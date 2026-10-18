"""`qlm bench [--quick]`: measure the running server as Con uses it (docs/architecture.md, "What qlm bench measures").

Prompt processing, generation, MTP draft acceptance, prompt-cache reuse in a Con-shaped session, and
peak VRAM, with MTP on and off. To measure with MTP off, the profile is swapped for a copy without the
LLAMA_ARG_SPEC_* lines and always put back, whatever ends the run. Results go to a JSON file and are
compared with the last comparable run. The numbers never fail the command; only a measurement that
couldn't be taken does.
"""

import hashlib
import json
import os
import random
import re
import statistics
import threading

from qlm import files, interrupts, service, smoke
from qlm.system import SERVER, ServerUnreachable

WARNING = ("Note: qlm bench replaces Con's cached prompt and, in a full run with MTP on, restarts the server "
           "twice. Don't use Con until it finishes.")
INTERRUPTED = "interrupted; nothing was changed"
CHAT_TIMEOUT_S = 300
TOKENIZE_TIMEOUT_S = 30
METRICS_TIMEOUT_S = 10
SIZES = (2048, 16384, 49152)
CONTEXT_MARGIN = 1024
RUNS = 3
GEN_TOKENS = 512
QUICK_GEN_TOKENS = 128
SEED = 42
SIZE_TOLERANCE = 0.02
MAX_TOKENIZE_CALLS = 4
CACHE_PREFIX_TOKENS = 13000
CACHE_RESULT_TOKENS = 1000
CACHE_TURNS = 4
CACHE_MAX_TOKENS = 64
REUSE_MARGIN = 1024
STREAM_ARGV = ("nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits", "--loop-ms=200")
RESULT_NAME = re.compile(r"^bench-(\d{8}T\d{6}Z)(?:-(\d+))?\.json$")

WORDS = (
    "about above across after again against along also always among animal answer around away back "
    "because before began behind being below best better between big black blue body book both bring "
    "build built call came can carry change children city close cold come common could country course "
    "cover cross dark day deep different does done door down draw during each early earth east easy "
    "end enough even ever every eye face fact family far farm fast father feel few field find fire "
    "first fish follow food foot form found four free friend front full game give good great green "
    "ground group grow half hand happen hard head hear heard help here high hold home horse hour house "
    "idea important inside island keep kind king land large last late learn leave left letter light "
    "line list little live long look low made main make many mark mean measure might mile mind money "
    "more morning most mother mountain move much music must name near need never next night north "
    "nothing notice number often old once only open order other over page paper part pass people "
    "picture place plain plan plant point power press problem pull question quick rain read ready real river"
).split()

CODE_PROMPT = "Rewrite this Python function with type hints and a docstring. Change nothing else."
CODE_FUNCTION = '''def summarize_orders(orders, currency, include_refunds, cutoff):
    totals = {}
    counts = {}
    refunds = 0
    skipped = []
    for order in orders:
        if order.get("status") == "cancelled":
            skipped.append(order["id"])
            continue
        if cutoff is not None and order["created"] > cutoff:
            skipped.append(order["id"])
            continue
        region = order.get("region", "unknown")
        amount = order["amount"]
        if order.get("currency") != currency:
            rate = order.get("rate")
            if rate is None:
                skipped.append(order["id"])
                continue
            amount = amount * rate
        if order.get("refunded"):
            refunds += amount
            if not include_refunds:
                continue
            amount = -amount
        totals[region] = totals.get(region, 0) + amount
        counts[region] = counts.get(region, 0) + 1
    report = []
    for region in sorted(totals):
        average = totals[region] / counts[region]
        report.append({
            "region": region,
            "total": round(totals[region], 2),
            "orders": counts[region],
            "average": round(average, 2),
        })
    return {
        "currency": currency,
        "regions": report,
        "refunds": round(refunds, 2),
        "skipped": skipped,
    }
'''
PROSE_PROMPT = "Explain how a hash table handles collisions. Cover chaining and open addressing, in plain paragraphs."
GEN_PROMPTS = (("code", f"{CODE_PROMPT}\n\n{CODE_FUNCTION}"), ("prose", PROSE_PROMPT))


class MeasureFailed(Exception):
    """A measurement couldn't be taken; the run stops."""


# Synthetic text


def filler(seed, n_words):
    """`n_words` words of sentences of 8 to 16 words. The same seed always gives the same words, so a
    shorter text is a prefix of a longer one."""
    rng = random.Random(seed)
    sentences, total = [], 0
    while total < n_words:
        length = rng.randint(8, 16)
        sentences.append([rng.choice(WORDS) for _ in range(length)])
        total += length
    out, left = [], n_words
    for sentence in sentences:
        part = sentence[:left]
        left -= len(part)
        if part:
            out.append(" ".join([part[0].capitalize(), *part[1:]]) + ".")
        if left == 0:
            break
    return " ".join(out)


def sized_text(count_tokens, seed, target):
    """Filler text whose token count is within 2% of `target`, or None after MAX_TOKENIZE_CALLS tries."""
    n_words = target
    for _ in range(MAX_TOKENIZE_CALLS):
        text = filler(seed, max(n_words, 1))
        tokens = count_tokens(text)
        if abs(tokens - target) <= SIZE_TOLERANCE * target:
            return text
        # Bounded, so a tokenizer that never answers sensibly can't make the text grow without limit.
        n_words = min(max(1, round(n_words * target / max(tokens, 1))), 4 * target)
    return None


# Pure helpers


def parse_counters(text):
    counters = {}
    for line in text.splitlines():
        m = re.match(r"^(llamacpp:[a-z_]+)\s+([0-9.e+]+)\s*$", line)
        if m:
            counters[m.group(1)] = float(m.group(2))
    return counters


def cache_verdicts(turns):
    """[(prompt_n, cache_n)] per request -> [{prompt_n, cache_n, verdict}]. A turn is reused when it
    re-processed at most REUSE_MARGIN tokens more than it added."""
    out, previous_total = [], None
    for prompt_n, cache_n in turns:
        verdict = None
        if previous_total is not None:
            new = prompt_n + cache_n - previous_total
            verdict = "reused" if prompt_n - new <= REUSE_MARGIN else "re-processed"
        out.append({"prompt_n": prompt_n, "cache_n": cache_n, "verdict": verdict})
        previous_total = prompt_n + cache_n
    return out


def peak_mib(lines, pid):
    peak = None
    for line in lines:
        fields = [f.strip() for f in line.split(",")]
        if len(fields) == 2 and fields[0].isdigit() and fields[1].isdigit() and int(fields[0]) == pid:
            peak = max(peak or 0, int(fields[1]))
    return peak


def mtp_off_variant(text):
    return "".join(line + "\n" for line in text.splitlines() if not line.startswith("LLAMA_ARG_SPEC_"))


def _result_key(path):
    m = RESULT_NAME.match(path.name)
    return (m.group(1), int(m.group(2) or 0)) if m else None


# Talking to the server


class _Client:
    def __init__(self, ctx, key):
        self.ctx, self.key = ctx, key
        self.build = None

    def _post(self, path, body, what, timeout_s):
        try:
            r = self.ctx.http.request("POST", f"{SERVER}{path}", headers={"Authorization": f"Bearer {self.key}"},
                                      json_body=body, timeout_s=timeout_s)
        except ServerUnreachable as e:
            raise MeasureFailed(f"{what}: {e}") from None
        if r.status != 200:
            raise MeasureFailed(f"{what}: HTTP {r.status}")
        try:
            return json.loads(r.body)
        except ValueError:
            raise MeasureFailed(f"{what}: the reply is not JSON") from None

    def tokens(self, text, what):
        reply = self._post("/tokenize", {"content": text}, what, TOKENIZE_TIMEOUT_S)
        tokens = reply.get("tokens") if isinstance(reply, dict) else None
        if not isinstance(tokens, list):
            raise MeasureFailed(f"{what}: /tokenize returned no tokens")
        return len(tokens)

    def chat(self, body, what):
        reply = self._post("/v1/chat/completions", body, what, CHAT_TIMEOUT_S)
        timings = reply.get("timings") if isinstance(reply, dict) else None
        if not isinstance(timings, dict):
            raise MeasureFailed(f"{what}: the reply has no timings")
        if self.build is None:
            self.build = reply.get("system_fingerprint")
        return timings

    def counters(self):
        """Speculative-decoding counters from /metrics, or (None, reason)."""
        try:
            r = self.ctx.http.request("GET", f"{SERVER}/metrics", headers={"Authorization": f"Bearer {self.key}"},
                                      timeout_s=METRICS_TIMEOUT_S)
        except ServerUnreachable as e:
            return None, str(e)
        if r.status != 200:
            return None, f"HTTP {r.status}"
        return parse_counters(r.body.decode(errors="replace")), None


def _text(client, seed, target, what):
    text = sized_text(lambda t: client.tokens(t, what), seed, target)
    if text is None:
        raise MeasureFailed(f"{what}: couldn't size the prompt to {target:,} tokens")
    return text


def _timing(timings, field, what):
    value = timings.get(field)
    if not isinstance(value, (int, float)):
        raise MeasureFailed(f"{what}: the reply's timings have no {field}")
    return value


# Peak VRAM


class _VramSampler:
    """Reads nvidia-smi's compute-process loop on a thread while one MTP mode is measured."""

    def __init__(self, ctx):
        self.lines, self.stream, self.thread = [], None, None
        if ctx.runner.which("nvidia-smi"):
            self.stream = ctx.runner.stream(list(STREAM_ARGV))
            self.thread = threading.Thread(target=self._read, daemon=True)
            self.thread.start()

    def _read(self):
        for line in self.stream.lines():
            self.lines.append(line)

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.thread.join()


# One MTP mode


class _Run:
    def __init__(self, ctx, client, quick, settings, vram_gib):
        self.ctx, self.client, self.quick, self.vram_gib = ctx, client, quick, vram_gib
        size = settings.get("LLAMA_ARG_CTX_SIZE", "")
        self.ctx_size = int(size) if size.isdigit() else None
        self.measures, self.cache_turns = {}, {}

    def say(self, mode, text):
        self.ctx.say(f"{mode} {text}")

    def mode(self, mode, pid):
        sampler = _VramSampler(self.ctx)
        try:
            self._prompt_processing(mode)
            self._generation(mode)
            if not self.quick:
                self._cache_reuse(mode)
        finally:
            sampler.stop()
        peak = peak_mib(sampler.lines, pid)
        if peak is None:
            self.measures[f"{mode}.peak_vram_gib"] = None
            self.say(mode, "peak VRAM: unavailable")
        else:
            self.measures[f"{mode}.peak_vram_gib"] = round(peak / 1024, 2)
            self.say(mode, f"peak VRAM: {peak / 1024:.1f} GiB (profile budget {self.vram_gib} GiB)")

    def _prompt_processing(self, mode):
        for i, size in enumerate(SIZES[:1] if self.quick else SIZES):
            what = f"prompt {size:,}"
            key = f"{mode}.pp_{size}"
            if self.ctx_size is not None and size + CONTEXT_MARGIN > self.ctx_size:
                self.measures[f"{key}.tok_s"] = self.measures[f"{key}.prompt_n"] = None
                self.say(mode, f"{what}: SKIP, doesn't fit the context ({self.ctx_size:,} tokens)")
                continue
            text = _text(self.client, SEED + i, size, what)
            body = {"messages": [{"role": "user", "content": text}], "max_tokens": 1, "cache_prompt": False,
                    "chat_template_kwargs": {"enable_thinking": False}}
            rates, prompt_n = [], None
            for _ in range(1 if self.quick else RUNS):
                timings = self.client.chat(body, what)
                rates.append(_timing(timings, "prompt_per_second", what))
                prompt_n = prompt_n if prompt_n is not None else _timing(timings, "prompt_n", what)
            rate = statistics.median(rates)
            self.measures[f"{key}.tok_s"], self.measures[f"{key}.prompt_n"] = rate, prompt_n
            self.say(mode, f"{what}: {rate:,.1f} tok/s ({prompt_n:,} tokens)")

    def _generation(self, mode):
        n_tokens = QUICK_GEN_TOKENS if self.quick else GEN_TOKENS
        for name, prompt in GEN_PROMPTS[:1] if self.quick else GEN_PROMPTS:
            what = f"generation ({name})"
            key = f"{mode}.gen_{name}"
            body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": n_tokens, "ignore_eos": True,
                    "seed": SEED, "chat_template_kwargs": {"enable_thinking": False}}
            before, problem = self.client.counters() if mode == "mtp_on" else (None, None)
            rates, drafted, accepted = [], 0, 0
            for _ in range(1 if self.quick else RUNS):
                timings = self.client.chat(body, what)
                if _timing(timings, "predicted_n", what) != n_tokens:
                    raise MeasureFailed(f"{what}: {timings['predicted_n']} tokens generated, not {n_tokens}")
                rates.append(_timing(timings, "predicted_per_second", what))
                drafted += timings.get("draft_n") or 0
                accepted += timings.get("draft_n_accepted") or 0
            rate = statistics.median(rates)
            self.measures[f"{key}.tok_s"] = rate
            self.say(mode, f"{what}: {rate:,.1f} tok/s")
            if mode != "mtp_on":
                continue
            after, problem_after = self.client.counters()
            self._acceptance(mode, key, what, drafted, accepted, before, after, problem or problem_after)

    def _acceptance(self, mode, key, what, drafted, accepted, before, after, problem):
        if drafted:
            pct = round(100 * accepted / drafted, 1)
            self.measures[f"{key}.accept_pct"] = pct
            self.say(mode, f"{what}: {pct:.1f}% of {drafted:,} drafted tokens accepted")
        else:
            self.measures[f"{key}.accept_pct"] = None
            self.say(mode, f"{what}: acceptance unavailable (no drafted tokens)")
        acc, steps = "llamacpp:spec_decode_num_accepted_tokens_total", "llamacpp:spec_decode_num_drafts_total"
        if problem is None and not all(c in d for d in (before, after) for c in (acc, steps)):
            problem = "a counter is missing from /metrics"
        if problem is None and after[steps] - before[steps] <= 0:
            problem = "no verification steps"
        if problem:
            self.measures[f"{key}.tokens_per_step"] = None
            self.say(mode, f"{what}: tokens per step unavailable ({problem})")
            return
        per_step = (after[acc] - before[acc]) / (after[steps] - before[steps])
        self.measures[f"{key}.tokens_per_step"] = round(per_step, 2)
        self.say(mode, f"{what}: {per_step:.1f} accepted tokens per step")

    def _cache_reuse(self, mode):
        what = "cache reuse"
        tool = {"type": "function", "function": {"name": smoke.TOOL_NAME, "description": smoke.TOOL_DESCRIPTION,
                                                 "parameters": smoke.TOOL_PARAMETERS}}
        messages = [{"role": "system", "content": _text(self.client, 100, CACHE_PREFIX_TOKENS, what)},
                    {"role": "user", "content": smoke.PROMPT}]
        raw = []
        for n in range(1, CACHE_TURNS + 1):
            if n > 1:
                call_id = f"call_{n}"
                messages = messages + [
                    {"role": "assistant", "content": "", "tool_calls": [{
                        "id": call_id, "type": "function",
                        "function": {"name": smoke.TOOL_NAME, "arguments": json.dumps({"order_id": smoke.ORDER_ID})},
                    }]},
                    {"role": "tool", "tool_call_id": call_id,
                     "content": _text(self.client, 100 + n, CACHE_RESULT_TOKENS, what)},
                ]
            timings = self.client.chat({"messages": messages, "tools": [tool], "max_tokens": CACHE_MAX_TOKENS},
                                       f"{what} turn {n}")
            raw.append((_timing(timings, "prompt_n", what), _timing(timings, "cache_n", what)))
        turns = cache_verdicts(raw)
        self.cache_turns[mode] = turns
        for n, turn in enumerate(turns, start=1):
            self.measures[f"{mode}.cache_turn_{n}.reprocessed"] = turn["prompt_n"]
            self.measures[f"{mode}.cache_turn_{n}.cache_n"] = turn["cache_n"]
            verdict = f", {turn['verdict']}" if turn["verdict"] else ""
            self.say(mode, f"{what} turn {n}: re-processed {turn['prompt_n']:,} tokens, "
                           f"{turn['cache_n']:,} from the cache{verdict}")


# The command


def run(ctx, quick=False):
    try:
        return _run(ctx, quick)
    except service.Stop as stop:
        return ctx.fail(stop.what, stop.todo)
    except MeasureFailed as e:
        return ctx.fail(f"bench stopped: {e}", "Nothing was saved. Check the server with qlm status and qlm logs")
    except KeyboardInterrupt:
        return ctx.fail(INTERRUPTED, "Run qlm bench again when you're ready")
    except BrokenPipeError:
        raise
    except OSError as e:
        return ctx.fail(f"file error: {e}", "Fix the cause and try again")


def _check_ready(ctx):
    p = ctx.paths
    service.leftover_bench_backup(ctx)
    if not p.api_key.exists() or not p.api_key.read_text().strip():
        raise service.Stop(f"the API key file {p.api_key} is missing or empty", "Run qlm install")
    profile = service.active_profile(ctx)
    state = service.unit_state(ctx)
    if not state.running or service.health_status(ctx) != 200:
        raise service.Stop("the server isn't ready", "Start it with qlm start; qlm bench never starts it")
    return profile, state


def _run(ctx, quick):
    p = ctx.paths
    profile, state = _check_ready(ctx)
    original = p.active_env.read_bytes()
    name = service.active_profile_name(ctx)
    mtp = "LLAMA_ARG_SPEC_TYPE" in profile.settings
    ctx.say(WARNING)
    client = _Client(ctx, p.api_key.read_text().strip())
    bench = _Run(ctx, client, quick, profile.settings, profile.vram_gib)

    with interrupts.sigterm_as_interrupt():
        if not mtp:
            ctx.say("MTP is off in this profile, so it is measured once, with MTP off.")
        bench.mode("mtp_on" if mtp else "mtp_off", state.pid)
        if mtp and not quick:
            _measure_mtp_off(ctx, bench, profile, state, original)

    record = {
        "version": 1,
        "time": ctx.clock.now().isoformat(),
        "quick": quick,
        "profile": name,
        "active_env_sha256": hashlib.sha256(original).hexdigest(),
        "llama_tag": os.path.basename(os.readlink(p.llama_current)) if p.llama_current.is_symlink() else None,
        "server_build": client.build,
        "ctx_size": bench.ctx_size,
        "vram_gib_budget": profile.vram_gib,
        "measures": bench.measures,
        "cache_turns": bench.cache_turns,
    }
    _compare(ctx, record)
    path = _save(ctx, record)
    ctx.say(f"Results: {path}")
    return 0


def _measure_mtp_off(ctx, bench, profile, state, original):
    """Switch to the MTP-off variant, measure, and always put the profile back."""
    p = ctx.paths
    service.check_ollama(ctx)
    service.check_vram(ctx, profile, credit_mib=service.server_vram_mib(ctx, state.pid) or 0)
    # A Ctrl-C halfway through would leave a partial backup, and the fix qlm prints for a leftover
    # backup (mv it over active.env) would then lose the profile.
    with interrupts.deferred():
        fd = os.open(p.bench_backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(original)
            f.flush()
            os.fsync(f.fileno())
    failure = None
    try:
        files.replace_atomically(p.active_env, mtp_off_variant(original.decode()).encode(), 0o600)
        if not service.bring_up(ctx, restart=True):
            raise MeasureFailed("the server didn't become ready with MTP off")
        bench.mode("mtp_off", service.unit_state(ctx).pid)
    except (Exception, KeyboardInterrupt) as e:  # anything, so the restore below always runs
        failure = e
    restored = _restore(ctx, original)
    if not restored:
        if failure is not None:
            ctx.warn(f"qlm: bench stopped: {_describe(failure)}")
        raise service.Stop("the server didn't become ready after the profile was restored, so it is stopped",
                           "Your profile is back in place; run qlm start")
    if isinstance(failure, KeyboardInterrupt):
        raise service.Stop("interrupted; the profile was restored and the server restarted", "Nothing was saved")
    if failure is not None:
        raise failure


def _describe(failure):
    if isinstance(failure, KeyboardInterrupt):
        return "interrupted"
    return str(getattr(failure, "what", failure))


def _restore(ctx, original):
    p = ctx.paths
    with interrupts.deferred():
        files.replace_atomically(p.active_env, original, 0o600)
        ready = service.bring_up(ctx, restart=True)
        p.bench_backup.unlink()
    return ready


# Saving and comparing


def _bench_dir(ctx):
    d = ctx.paths.bench
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(d, 0o700)
    return d


def _save(ctx, record):
    stamp = ctx.clock.now().strftime("%Y%m%dT%H%M%SZ")
    data = (json.dumps(record, indent=2) + "\n").encode()
    return files.create_new(_bench_dir(ctx), lambda n: f"bench-{files.stamped(stamp, n)}.json", data)


def _earlier(ctx, record):
    d = ctx.paths.bench
    if not d.is_dir():
        return None, None
    best = None
    for path in d.iterdir():
        key = _result_key(path)
        if key is None:
            continue
        try:
            data = json.loads(path.read_text())
            if not isinstance(data, dict) or not isinstance(data.get("measures"), dict):
                raise ValueError
        except (OSError, ValueError):
            ctx.warn(f"warning: skipped {path.name}, which isn't a bench result")
            continue
        if data.get("quick") != record["quick"] or data.get("profile") != record["profile"]:
            continue
        if best is None or key > best[0]:
            best = (key, path, data)
    return (best[1], best[2]) if best else (None, None)


def _fmt(value):
    return "-" if value is None else f"{value:,.2f}".rstrip("0").rstrip(".")


def _compare(ctx, record):
    path, earlier = _earlier(ctx, record)
    if earlier is None:
        ctx.say(f"There is nothing to compare with: no earlier {'quick' if record['quick'] else 'full'} run "
                f"with the {record['profile']} profile.")
        return
    ctx.say(f"Compared with {path.name}:")
    now, then = record["measures"], earlier["measures"]
    for name in sorted(set(now) | set(then)):
        cur, prev = now.get(name), then.get(name)
        if cur is None and prev is None:
            continue
        change = ""
        if isinstance(cur, (int, float)) and isinstance(prev, (int, float)) and prev:
            change = f" ({100 * (cur - prev) / prev:+.1f}%)"
        ctx.say(f"  {name}: {_fmt(cur)} vs {_fmt(prev)}{change}")
