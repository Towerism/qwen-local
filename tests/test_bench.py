"""qlm bench against a fake llama-server whose timings are fixed, so every number is known in advance."""

import json
import os
import shutil
import signal
import stat

import pytest

from qlm import bench, cli
from qlm.system import ServerUnreachable
from tests.fakes import FakeClock, FakeRunner, FakeServer, fail, make_ctx, ok, reply
from tests.test_service import (
    OLLAMA_PS, RESET, RESTART, RUNNING, SHOW, SMI_APPS, SMI_GPU, STOP, apps, gpu, install, rendered, runner, unit,
)

KEY = "k" * 43
STREAM = ("nvidia-smi", "--query-compute-apps=pid,used_memory")
SPEC_LINES = ("LLAMA_ARG_SPEC_TYPE", "LLAMA_ARG_SPEC_DRAFT_MODEL", "LLAMA_ARG_SPEC_DRAFT_N_MAX")
IDEAL_CACHE = [(13100, 0), (1104, 13096), (1104, 14196), (1104, 15296)]


def words(text):
    return len(text.split())


def kind(body):
    if body.get("max_tokens") == 1:
        return "pp"
    if body.get("tools"):
        return "cache"
    return "gen"


class BenchServer(FakeServer):
    """A FakeServer for bench: /tokenize counts words, chat replies carry the timings given here.

    pp_rates and gen_rates are used one per request, the last repeated. drafts is a list of
    (draft_n, draft_n_accepted) per generation request; None means MTP drafted nothing. cache is a
    list of (prompt_n, cache_n) per cache-reuse turn, repeated for each MTP mode.
    """

    def __init__(self, pp_rates=(2000.0,), gen_rates=(100.0,), drafts=((200, 150),), cache=IDEAL_CACHE,
                 metrics=None, predicted_n=None, tokenize=None, overrides=None):
        self.pp_rates, self.gen_rates, self.drafts = list(pp_rates), list(gen_rates), list(drafts)
        self.cache, self.predicted_n = list(cache), predicted_n
        self.cache_turn = 0
        replies = {
            ("GET", "/health"): reply(200, {"status": "ok"}),
            ("POST", "/tokenize"): tokenize or (lambda r: reply(200, {"tokens": list(range(words(r["json"]["content"])))})),
            ("POST", "/v1/chat/completions"): self.chat,
            ("GET", "/metrics"): metrics or reply(200, metrics_text(0, 0)),
        }
        replies.update(overrides or {})
        super().__init__(replies)

    @staticmethod
    def _next(values):
        return values.pop(0) if len(values) > 1 else values[0]

    def chat(self, record):
        body = record["json"]
        k = kind(body)
        if k == "pp":
            text = body["messages"][0]["content"]
            timings = {"cache_n": 0, "prompt_n": words(text) + 12, "prompt_per_second": self._next(self.pp_rates),
                       "predicted_n": 1, "predicted_per_second": 10.0}
        elif k == "gen":
            n = self.predicted_n if self.predicted_n is not None else body["max_tokens"]
            timings = {"cache_n": 0, "prompt_n": 40, "prompt_per_second": 500.0, "predicted_n": n,
                       "predicted_per_second": self._next(self.gen_rates)}
            draft = self._next(self.drafts)
            if draft is not None:
                timings["draft_n"], timings["draft_n_accepted"] = draft
        else:
            prompt_n, cache_n = self.cache[self.cache_turn % len(self.cache)]
            self.cache_turn += 1
            timings = {"cache_n": cache_n, "prompt_n": prompt_n, "prompt_per_second": 2000.0,
                       "predicted_n": 64, "predicted_per_second": 90.0}
        return reply(200, {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "x"}}],
                           "system_fingerprint": "b11669-620d2225c", "timings": timings})

    def chats(self, k):
        return [r["json"] for r in self.to("POST", "/v1/chat/completions") if kind(r["json"]) == k]


def metrics_text(accepted, steps):
    return (
        "# TYPE llamacpp:spec_decode_num_accepted_tokens_total counter\n"
        f"llamacpp:spec_decode_num_accepted_tokens_total {accepted}\n"
        "# TYPE llamacpp:spec_decode_num_drafts_total counter\n"
        f"llamacpp:spec_decode_num_drafts_total {steps}\n"
    )


def bench_runner(stream_lines=("4321, 19800", "4321, 20100", "9999, 23000", "4321, 19950"), overrides=None):
    script = {SHOW: RUNNING, STREAM: ok("".join(f"{l}\n" for l in stream_lines))}
    script.update(overrides or {})
    return runner(script)


def bench_home(home, repo, profile="agent"):
    """A home after qlm install and fetch, with current a symlink to the b11669 build."""
    p = install(home, repo, profile=profile, key=KEY)
    shutil.rmtree(p.llama_current)
    (p.tag_dir("b11669") / "bin").mkdir(parents=True)
    (p.tag_dir("b11669") / "bin/llama-server").write_bytes(b"server")
    p.llama_current.symlink_to(p.tag_dir("b11669"))
    return p


def run_bench(home, repo, server=None, run=None, args=("bench",), profile="agent"):
    p = bench_home(home, repo, profile)
    ctx = make_ctx(home, repo, runner=run or bench_runner(), http=server or BenchServer(), clock=FakeClock(advance=True))
    code = cli.main(list(args), ctx)
    return code, ctx, p


def out(ctx):
    return ctx.out.getvalue() + ctx.err.getvalue()


def results(p):
    return sorted((p.state_dir / "bench").glob("*.json")) if (p.state_dir / "bench").exists() else []


# Filler text


def test_filler_is_the_same_for_the_same_seed_and_sized_within_2_percent():
    calls = []

    def count(text):
        calls.append(text)
        return int(words(text) * 1.3)

    text = bench.sized_text(count, seed=7, target=2048)
    assert abs(count(text) - 2048) <= 0.02 * 2048
    assert bench.sized_text(lambda t: int(words(t) * 1.3), seed=7, target=2048) == text
    assert len(calls) <= 5  # four while sizing, one in this test
    assert set(w.strip(".").lower() for w in text.split()) <= set(bench.WORDS)


def test_filler_sentences_have_8_to_16_words():
    sentences = [s.split() for s in bench.filler(3, 400).split(".") if s.strip()]
    assert all(8 <= len(s) <= 16 for s in sentences[:-1])


def test_the_word_list_has_200_words():
    assert len(bench.WORDS) == 200 == len(set(bench.WORDS))


def test_text_that_never_sizes_fails_after_4_tokenize_calls(home, repo):
    calls = []

    def stubborn(record):
        calls.append(record)
        return reply(200, {"tokens": [0] * 10})

    code, ctx, p = run_bench(home, repo, server=BenchServer(tokenize=stubborn), args=("bench", "--quick"))
    assert code == 1
    assert len(calls) == 4
    assert "prompt 2,048" in ctx.err.getvalue()
    assert results(p) == []


# Checks before measuring


def test_a_leftover_backup_blocks_the_run(home, repo):
    p = install(home, repo, key=KEY)
    p.bench_backup.write_text("x")
    server = BenchServer()
    ctx = make_ctx(home, repo, runner=bench_runner(), http=server, clock=FakeClock(advance=True))
    assert cli.main(["bench"], ctx) == 1
    assert "mv " in ctx.err.getvalue()
    assert server.requests == []


def test_a_stopped_server_is_not_started(home, repo):
    run = bench_runner(overrides={SHOW: unit()})
    code, ctx, _ = run_bench(home, repo, run=run)
    assert code == 1
    assert "qlm start" in ctx.err.getvalue()
    assert run.ran("systemctl", "--user", "start") == [] and run.ran(*RESTART) == []


def test_a_loading_server_is_not_ready(home, repo):
    server = BenchServer(overrides={("GET", "/health"): reply(503, {})})
    code, ctx, _ = run_bench(home, repo, server=server)
    assert code == 1 and "qlm start" in ctx.err.getvalue()


def test_the_first_line_warns_about_con_and_the_restarts(home, repo):
    code, ctx, _ = run_bench(home, repo)
    assert code == 0, out(ctx)
    first = ctx.out.getvalue().splitlines()[0]
    assert "Con" in first and "restarts the server twice" in first


# Measurements


def test_prompt_processing_requests_and_the_median(home, repo):
    server = BenchServer(pp_rates=(1000.0, 3000.0, 2000.0))
    code, ctx, p = run_bench(home, repo, server=server, args=("bench", "--quick"))
    assert code == 0, out(ctx)
    pp = server.chats("pp")
    assert len(pp) == 1
    body = pp[0]
    assert body["cache_prompt"] is False and body["chat_template_kwargs"] == {"enable_thinking": False}
    assert [r["timeout_s"] for r in server.to("POST", "/v1/chat/completions")] == [300, 300]
    assert all(r["timeout_s"] == 30 for r in server.to("POST", "/tokenize"))


def test_a_full_run_sends_each_size_three_times_and_reports_the_median(home, repo):
    server = BenchServer(pp_rates=(1000.0, 3000.0, 2000.0, 2000.0))
    code, ctx, p = run_bench(home, repo, server=server)
    assert code == 0, out(ctx)
    measures = json.loads(results(p)[0].read_text())["measures"]
    assert measures["mtp_on.pp_2048.tok_s"] == 2000.0
    sizes = [words(b["messages"][0]["content"]) for b in server.chats("pp")]
    assert len(sizes) == 18  # 3 sizes x 3 runs x 2 modes
    assert sizes[0] == sizes[1] == sizes[2] and sizes[3] > sizes[0]


def test_a_size_that_does_not_fit_the_context_is_skipped(home, repo):
    server = BenchServer()
    code, ctx, p = run_bench(home, repo, server=server, profile="vision")
    assert code == 0, out(ctx)
    assert all(words(b["messages"][0]["content"]) < 40000 for b in server.chats("pp"))
    assert "doesn't fit the context" in ctx.out.getvalue()
    assert json.loads(results(p)[0].read_text())["measures"]["mtp_off.pp_49152.tok_s"] is None


def test_generation_requests_carry_only_the_specified_fields(home, repo):
    server = BenchServer()
    code, ctx, _ = run_bench(home, repo, server=server)
    assert code == 0, out(ctx)
    gens = server.chats("gen")
    assert len(gens) == 12  # 2 prompts x 3 runs x 2 modes
    for body in gens:
        assert body["max_tokens"] == 512 and body["ignore_eos"] is True and body["seed"] == 42
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        assert not {"temperature", "top_p", "top_k", "min_p"} & set(body)
    assert "Rewrite this Python function" in gens[0]["messages"][0]["content"]
    assert "hash table" in gens[3]["messages"][0]["content"]


def test_a_short_generation_fails_the_run(home, repo):
    code, ctx, p = run_bench(home, repo, server=BenchServer(predicted_n=300))
    assert code == 1
    assert "generation (code)" in ctx.err.getvalue()
    assert results(p) == []


def test_acceptance_rate_and_tokens_per_step(home, repo):
    server = BenchServer(
        drafts=((200, 150), (220, 160), (240, 170)),
        metrics=[reply(200, metrics_text(0, 0)), reply(200, metrics_text(480, 240))],
    )
    code, ctx, _ = run_bench(home, repo, server=server, args=("bench",))
    text = ctx.out.getvalue()
    assert "72.7%" in text
    assert "2.0 accepted tokens per step" in text


def test_unreadable_metrics_are_unavailable_not_a_failure(home, repo):
    server = BenchServer(metrics=reply(500, b"oops"))
    code, ctx, _ = run_bench(home, repo, server=server)
    assert code == 0, out(ctx)
    assert "unavailable (HTTP 500)" in ctx.out.getvalue()


def test_no_drafted_tokens_is_unavailable(home, repo):
    code, ctx, _ = run_bench(home, repo, server=BenchServer(drafts=(None,)))
    assert code == 0, out(ctx)
    assert "unavailable (no drafted tokens)" in ctx.out.getvalue()


def test_cache_reuse_verdicts_and_the_resent_history(home, repo):
    server = BenchServer(cache=[(13100, 0), (1104, 13096), (1104, 14196), (14000, 2400)])
    code, ctx, p = run_bench(home, repo, server=server)
    assert code == 0, out(ctx)
    turns = json.loads(results(p)[0].read_text())["cache_turns"]["mtp_on"]
    assert [t["verdict"] for t in turns] == [None, "reused", "reused", "re-processed"]
    second = server.chats("cache")[1]["messages"]
    assistant, tool = second[-2], second[-1]
    assert assistant["role"] == "assistant" and assistant["content"] == "" and "reasoning_content" not in assistant
    assert assistant["tool_calls"][0]["id"] == "call_2"
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"order_id": "A-7731"}
    assert tool == {"role": "tool", "tool_call_id": "call_2", "content": tool["content"]}
    assert server.chats("cache")[0]["max_tokens"] == 64
    assert "cache_prompt" not in server.chats("cache")[0]


def test_cache_verdict_rule():
    # Re-processed minus new tokens equals the previous total minus this turn's cache_n.
    turns = bench.cache_verdicts([(13100, 0), (1104, 13096), (2124, 13176), (2125, 14175)])
    assert [t["verdict"] for t in turns] == [None, "reused", "reused", "re-processed"]


def test_peak_vram_is_the_servers_largest_sample(home, repo):
    run = bench_runner()
    code, ctx, _ = run_bench(home, repo, run=run, args=("bench", "--quick"))
    assert code == 0, out(ctx)
    assert "19.6 GiB" in ctx.out.getvalue() and "20.3 GiB" in ctx.out.getvalue()
    assert run.streams and all(s.stopped for s in run.streams)
    assert run.streams[0].argv == ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits",
                                   "--loop-ms=200"]


def test_peak_vram_without_nvidia_smi_is_unavailable(home, repo):
    run = runner({SHOW: RUNNING, "which": {"nvidia-smi": None}})
    run.which_map.pop("nvidia-smi", None)
    code, ctx, _ = run_bench(home, repo, run=run, args=("bench", "--quick"))
    assert code == 0, out(ctx)
    assert "peak VRAM: unavailable" in ctx.out.getvalue()


# MTP on and off


def test_a_full_run_switches_mtp_off_and_back(home, repo):
    seen = {}
    p_holder = {}

    class Watching(BenchServer):
        def chat(self, record):
            p = p_holder["p"]
            if p.bench_backup.exists():
                seen["variant"] = p.active_env.read_text()
                seen["backup"] = p.bench_backup.read_text()
                seen["backup_mode"] = stat.S_IMODE(p.bench_backup.stat().st_mode)
            return super().chat(record)

    p = bench_home(home, repo)
    p_holder["p"] = p
    original = p.active_env.read_text()
    run = bench_runner()
    ctx = make_ctx(home, repo, runner=run, http=Watching(), clock=FakeClock(advance=True))
    code = cli.main(["bench"], ctx)
    assert code == 0, out(ctx)
    assert seen["backup"] == original and seen["backup_mode"] == 0o600
    assert seen["variant"] == "".join(l + "\n" for l in original.splitlines() if not l.startswith("LLAMA_ARG_SPEC_"))
    assert p.active_env.read_text() == original
    assert not p.bench_backup.exists()
    assert len(run.ran(*RESTART)) == 2


def test_mtp_on_is_measured_before_any_restart(home, repo):
    order = []
    run = bench_runner()
    original = run.script[RESTART]
    run.script[RESTART] = lambda argv: (order.append("restart"), original)[1]

    class Recording(BenchServer):
        def chat(self, record):
            order.append("chat")
            return super().chat(record)

    code, ctx, _ = run_bench(home, repo, server=Recording(), run=run)
    assert code == 0, out(ctx)
    first_restart = order.index("restart")
    assert first_restart > 0 and all(o == "chat" for o in order[:first_restart])


def test_a_profile_without_mtp_measures_once_without_restarting(home, repo):
    run = bench_runner()
    code, ctx, p = run_bench(home, repo, run=run, profile="long")
    assert code == 0, out(ctx)
    assert "MTP is off in this profile" in ctx.out.getvalue()
    assert run.ran(*RESTART) == []
    assert p.active_env.read_text() == rendered(home, repo, "long")
    measures = json.loads(results(p)[0].read_text())["measures"]
    assert measures["mtp_off.pp_2048.tok_s"] is not None and measures.get("mtp_on.pp_2048.tok_s") is None


def test_not_enough_vram_for_the_variant_changes_nothing(home, repo):
    run = bench_runner(overrides={SMI_GPU: gpu(used=24400), SMI_APPS: apps(("4321", "llama-server", "100"))})
    code, ctx, p = run_bench(home, repo, run=run)
    assert code == 1
    assert "not enough VRAM" in ctx.err.getvalue()
    assert not p.bench_backup.exists()
    assert p.active_env.read_text() == rendered(home, repo, "agent")
    assert run.ran(*RESTART) == []


def test_ctrl_c_with_mtp_off_restores_the_profile(home, repo):
    class Interrupting(BenchServer):
        def chat(self, record):
            if p_holder["p"].bench_backup.exists():
                raise KeyboardInterrupt
            return super().chat(record)

    p_holder = {"p": bench_home(home, repo)}
    p = p_holder["p"]
    original = p.active_env.read_text()
    run = bench_runner()
    ctx = make_ctx(home, repo, runner=run, http=Interrupting(), clock=FakeClock(advance=True))
    code = cli.main(["bench"], ctx)
    assert code == 1
    assert p.active_env.read_text() == original and not p.bench_backup.exists()
    assert len(run.ran(*RESTART)) == 2
    assert results(p) == []
    assert "Traceback" not in out(ctx)


def test_ctrl_c_before_the_backup_changes_nothing(home, repo):
    server = BenchServer(overrides={("POST", "/tokenize"): KeyboardInterrupt()})
    run = bench_runner()
    code, ctx, p = run_bench(home, repo, server=server, run=run)
    assert code == 1
    assert "interrupted; nothing was changed" in ctx.err.getvalue()
    assert run.ran(*RESTART) == [] and not p.bench_backup.exists()


def test_sigterm_during_measuring_restores_the_profile(home, repo):
    class Terminated(BenchServer):
        def chat(self, record):
            if p_holder["p"].bench_backup.exists():
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            return super().chat(record)

    before = signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)
    p_holder = {"p": bench_home(home, repo)}
    p = p_holder["p"]
    original = p.active_env.read_text()
    ctx = make_ctx(home, repo, runner=bench_runner(), http=Terminated(), clock=FakeClock(advance=True))
    assert cli.main(["bench"], ctx) == 1
    assert p.active_env.read_text() == original and not p.bench_backup.exists()
    assert (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)) == before


def test_a_restore_that_never_gets_ready_stops_the_unit_and_says_so(home, repo):
    # /health: ready at the start, ready after the switch to MTP off, then never again during the restore.
    run = bench_runner(overrides={SHOW: [RUNNING, RUNNING, RUNNING, unit("failed", "failed")]})
    server = BenchServer(overrides={("GET", "/health"): [reply(200, {"status": "ok"})] * 2 + [reply(503, {})]})
    code, ctx, p = run_bench(home, repo, server=server, run=run)
    assert code == 1
    assert p.active_env.read_text() == rendered(home, repo, "agent")
    assert not p.bench_backup.exists()
    assert run.ran(*STOP)
    assert "qlm start" in ctx.err.getvalue().rstrip().splitlines()[-1]


def test_a_failed_measurement_restores_and_writes_no_file(home, repo):
    server = BenchServer(pp_rates=(2000.0,))
    statuses = iter([200] * 9 + [500])

    def flaky(record):
        if kind(record["json"]) == "pp" and next(statuses) == 500:
            return reply(500, {"error": "boom"})
        return server.chat(record)

    server.replies[("POST", "/v1/chat/completions")] = flaky
    code, ctx, p = run_bench(home, repo, server=server)
    assert code == 1
    assert "HTTP 500" in ctx.err.getvalue() and "prompt" in ctx.err.getvalue()
    assert p.active_env.read_text() == rendered(home, repo, "agent")
    assert results(p) == []


def test_an_unreachable_server_mid_run_fails_the_measurement(home, repo):
    server = BenchServer()
    server.replies[("POST", "/v1/chat/completions")] = ServerUnreachable("timed out after 300 s")
    code, ctx, p = run_bench(home, repo, server=server, args=("bench", "--quick"))
    assert code == 1
    assert "timed out after 300 s" in ctx.err.getvalue()


def test_quick_mode_measures_little_and_never_restarts(home, repo):
    server, run = BenchServer(), bench_runner()
    code, ctx, p = run_bench(home, repo, server=server, run=run, args=("bench", "--quick"))
    assert code == 0, out(ctx)
    assert len(server.chats("pp")) == 1 and len(server.chats("gen")) == 1 and server.chats("cache") == []
    assert server.chats("gen")[0]["max_tokens"] == 128
    assert run.ran(*RESTART) == []
    assert p.active_env.read_text() == rendered(home, repo, "agent")


# Result files and the comparison


def test_two_runs_in_one_second_get_separate_private_files(home, repo):
    code, ctx, p = run_bench(home, repo, args=("bench", "--quick"))
    ctx2 = make_ctx(home, repo, runner=bench_runner(), http=BenchServer(), clock=FakeClock(advance=True))
    assert cli.main(["bench", "--quick"], ctx2) == 0
    names = [f.name for f in results(p)]
    assert names == ["bench-20261017T120000Z-1.json", "bench-20261017T120000Z.json"]
    for f in results(p):
        assert stat.S_IMODE(f.stat().st_mode) == 0o600
        assert KEY not in f.read_text()
    assert stat.S_IMODE((p.state_dir / "bench").stat().st_mode) == 0o700
    assert ctx2.out.getvalue().rstrip().splitlines()[-1].endswith("bench-20261017T120000Z-1.json")


def test_the_result_file_records_the_run(home, repo):
    code, ctx, p = run_bench(home, repo, args=("bench", "--quick"))
    data = json.loads(results(p)[0].read_text())
    assert data["version"] == 1 and data["quick"] is True and data["profile"] == "agent"
    assert data["llama_tag"] == "b11669" and data["server_build"] == "b11669-620d2225c"
    assert data["ctx_size"] == 65536 and data["vram_gib_budget"] == 20.3
    assert len(data["active_env_sha256"]) == 64
    assert data["measures"]["mtp_on.peak_vram_gib"] == pytest.approx(20100 / 1024, abs=0.05)


def write_result(p, name, quick, profile, measures):
    d = p.state_dir / "bench"
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    (d / name).write_text(json.dumps({"version": 1, "quick": quick, "profile": profile, "measures": measures}))


def test_the_comparison_uses_the_newest_matching_run(home, repo):
    p = bench_home(home, repo)
    write_result(p, "bench-20261016T100000Z.json", True, "agent", {"mtp_on.pp_2048.tok_s": 1.0})
    write_result(p, "bench-20261016T110000Z.json", True, "agent", {"mtp_on.pp_2048.tok_s": 1000.0})
    write_result(p, "bench-20261016T110000Z-1.json", True, "agent",
                 {"mtp_on.pp_2048.tok_s": 1600.0, "mtp_on.only_then": 5.0})
    write_result(p, "bench-20261016T120000Z.json", False, "agent", {"mtp_on.pp_2048.tok_s": 9.0})
    write_result(p, "bench-20261016T130000Z.json", True, "long", {"mtp_on.pp_2048.tok_s": 9.0})
    (p.state_dir / "bench/bench-20261016T140000Z.json").write_text("not json")
    ctx = make_ctx(home, repo, runner=bench_runner(), http=BenchServer(), clock=FakeClock(advance=True))
    assert cli.main(["bench", "--quick"], ctx) == 0, out(ctx)
    text = ctx.out.getvalue()
    assert "bench-20261016T110000Z-1.json" in text
    line = next(l for l in text.splitlines() if "mtp_on.pp_2048.tok_s" in l and "+" in l)
    assert "+25.0%" in line
    assert any("mtp_on.only_then" in l and " - " in l for l in text.splitlines())
    assert "bench-20261016T140000Z.json" in ctx.err.getvalue()


def test_with_no_earlier_run_there_is_nothing_to_compare(home, repo):
    code, ctx, _ = run_bench(home, repo, args=("bench", "--quick"))
    assert code == 0
    assert "nothing to compare with" in ctx.out.getvalue()


def test_the_key_never_appears_in_output(home, repo):
    code, ctx, p = run_bench(home, repo)
    assert KEY not in out(ctx)


def test_ctrl_c_and_sigterm_are_ignored_while_the_profile_is_restored(home, repo):
    handlers = []
    run = bench_runner()

    def restart(argv):
        handlers.append((signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)))
        return ok()

    run.script[RESTART] = restart
    code, ctx, _ = run_bench(home, repo, run=run)
    assert code == 0, out(ctx)
    assert handlers[0] != (signal.SIG_IGN, signal.SIG_IGN)
    assert handlers[1] == (signal.SIG_IGN, signal.SIG_IGN)


def test_a_file_error_after_the_backup_still_restores(home, repo, monkeypatch):
    from qlm import files

    real = files.replace_atomically
    calls = []

    def disk_full_once(path, data, mode):
        calls.append(path)
        if len(calls) == 1:
            raise OSError(28, "No space left on device")
        return real(path, data, mode)

    monkeypatch.setattr(files, "replace_atomically", disk_full_once)
    code, ctx, p = run_bench(home, repo)
    assert code == 1
    assert "No space left on device" in ctx.err.getvalue()
    assert p.active_env.read_text() == rendered(home, repo, "agent")
    assert not p.bench_backup.exists()


def test_ctrl_c_and_sigterm_are_ignored_while_the_backup_is_written(home, repo, monkeypatch):
    real_fsync = os.fsync
    handlers = []

    def recording_fsync(fd):
        handlers.append((signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)))
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    code, ctx, p = run_bench(home, repo)
    assert code == 0, out(ctx)
    assert handlers[0] == (signal.SIG_IGN, signal.SIG_IGN)  # the backup is the first file bench writes
