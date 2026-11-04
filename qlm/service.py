"""`qlm use`, `profile use`, `start`, `stop`, `status` and `logs`: the server's systemd user unit and its health.

The VRAM check before every start compares free video memory with the profile's `vram_gib`
plus 0.5 GiB (docs/architecture.md). A start that doesn't reach a healthy server is stopped,
so systemd doesn't keep restarting a profile that can't fit.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from qlm import files, pins, profiles, selection
from qlm.system import SERVER, ServerUnreachable

UNIT = "qlm-server"
HOST = "127.0.0.1"
PORT = 9931
HEADROOM_MIB = 512
READY_TIMEOUT_S = 180
POLL_S = 1
HEALTH_TIMEOUT_S = 5
METRICS_TIMEOUT_S = 10
JOURNAL_LINES = 30
TIMEOUT_S = 10
FILE_KEYS = ("LLAMA_ARG_MODEL", "LLAMA_ARG_SPEC_DRAFT_MODEL", "LLAMA_ARG_MMPROJ")
RUNNING_STATES = ("active", "activating")
STATE_LABELS = {"inactive": "stopped", "failed": "failed", "deactivating": "stopping"}
PROCESSOR = re.compile(r"\s\d+%(?:/\d+%)?\s+(CPU/GPU|GPU|CPU)(?:\s|$)")


class Stop(Exception):
    """A step failed; carries the summary line's two halves."""

    def __init__(self, what, todo):
        super().__init__(what)
        self.what, self.todo = what, todo


@dataclass
class UnitState:
    active: str
    sub: str
    pid: int
    entered: int | None

    @property
    def running(self):
        return self.active in RUNNING_STATES

    @property
    def exited(self):
        return self.active in ("failed", "inactive") or self.sub == "auto-restart"


def systemctl(ctx, *args):
    return ctx.runner.run(["systemctl", "--user", *args], timeout=TIMEOUT_S)


def unit_state(ctx):
    r = systemctl(ctx, "show", "-p", "ActiveState,SubState,MainPID,ActiveEnterTimestamp", "--timestamp=unix", UNIT)
    if r.code != 0:
        raise Stop(
            f"systemctl --user show failed: {r.output.strip()}",
            "Run qlm from your desktop session, where the systemd user manager is reachable",
        )
    fields = dict(line.split("=", 1) for line in r.stdout.splitlines() if "=" in line)
    pid = fields.get("MainPID", "0")
    entered = fields.get("ActiveEnterTimestamp", "")
    return UnitState(
        active=fields.get("ActiveState", "unknown"),
        sub=fields.get("SubState", "unknown"),
        pid=int(pid) if pid.isdigit() else 0,
        entered=int(entered[1:]) if re.fullmatch(r"@\d+", entered) else None,
    )


# Profiles


def leftover_bench_backup(ctx):
    p = ctx.paths
    if p.bench_backup.exists():
        raise Stop(
            f"{p.bench_backup} is left from a qlm bench that was killed before it restored the profile",
            f"Restore it with: mv {p.bench_backup} {p.active_env}, then run qlm stop and qlm start",
        )



def active_profile(ctx):
    """The parsed active.env. Raises Stop when it is missing or doesn't parse."""
    p = ctx.paths
    if not p.active_env.exists():
        raise Stop(f"{p.active_env} is missing", "Run qlm profile use agent")
    try:
        return profiles.parse("active", p.active_env.read_text())
    except profiles.ProfileError as e:
        raise Stop(f"{p.active_env} doesn't parse: {e}", "Run qlm profile use agent") from None


def model_profiles(ctx, model_id):
    return profiles.load_all(ctx.paths.model_profiles_dir(model_id))


def active_profile_name(ctx):
    """The profile of active.env's own model whose rendering active.env equals, or "custom"."""
    model_id = selection.active_alias(ctx)
    if model_id not in pins.model_ids():
        return "custom"
    text = ctx.paths.active_env.read_text()
    for name, profile in model_profiles(ctx, model_id).items():
        if profile.render(ctx.paths.home) == text:
            return name
    return "custom"


def missing_files(settings):
    return [(key, settings[key]) for key in FILE_KEYS if key in settings and not Path(settings[key]).is_file()]


def _fetch_hint(missing):
    return "Run qlm fetch --vision" if any(k == "LLAMA_ARG_MMPROJ" for k, _ in missing) else "Run qlm fetch"


# GPU and Ollama


def gpu_memory(ctx):
    """(total MiB, used MiB) from nvidia-smi. Raises Stop when it can't be read."""
    if not ctx.runner.which("nvidia-smi"):
        raise Stop("nvidia-smi not found", "Install the NVIDIA driver utilities")
    r = ctx.runner.run(
        ["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"], timeout=TIMEOUT_S
    )
    try:
        total, used = (int(f.strip()) for f in r.stdout.splitlines()[0].split(","))
    except (IndexError, ValueError):
        raise Stop(f"couldn't read VRAM from nvidia-smi: {r.output.strip()[:200]}", "Check that nvidia-smi works") from None
    if r.code != 0:
        raise Stop(f"nvidia-smi failed: {r.output.strip()[:200]}", "Check that nvidia-smi works")
    return total, used


def compute_apps(ctx):
    """[(pid, process name, MiB)] from nvidia-smi; empty when it can't be read."""
    r = ctx.runner.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        timeout=TIMEOUT_S,
    )
    rows = []
    for line in r.stdout.splitlines() if r.code == 0 else []:
        fields = [f.strip() for f in line.split(",")]
        # Process names can contain commas: the first field is the PID, the last the MiB.
        if len(fields) >= 3 and fields[0].isdigit() and fields[-1].isdigit():
            rows.append((int(fields[0]), ", ".join(fields[1:-1]), int(fields[-1])))
    return rows


def server_vram_mib(ctx, pid):
    """MiB held by the process with this PID, or None."""
    if not pid or not ctx.runner.which("nvidia-smi"):
        return None
    return next((mib for app_pid, _, mib in compute_apps(ctx) if app_pid == pid), None)


def check_ollama(ctx):
    if not ctx.runner.which("ollama"):
        return
    r = ctx.runner.run(["ollama", "ps"], timeout=TIMEOUT_S)
    if r.code != 0:
        return  # a stopped daemon holds no VRAM
    on_gpu = []
    for line in r.stdout.splitlines()[1:]:
        processor = PROCESSOR.search(line)  # "100% GPU", "100% CPU" or "48%/52% CPU/GPU"; not the model name
        if processor and "GPU" in processor.group(1):
            on_gpu.append(line.split()[0])
    if on_gpu:
        for name in on_gpu:
            ctx.warn(f"Ollama has {name} on the GPU; free it with: ollama stop {name}")
        raise Stop("Ollama holds a model on the GPU", "Stop it with the command above and try again")


def check_vram(ctx, profile, credit_mib=0):
    total, used = gpu_memory(ctx)
    free = total - used + credit_mib
    need = profile.vram_gib * 1024 + HEADROOM_MIB
    if free >= need:
        return
    for _, name, mib in compute_apps(ctx):
        ctx.warn(f"  {name} ({mib} MiB)")
    raise Stop(
        f"not enough VRAM: {free / 1024:.1f} GiB free, {need / 1024:.1f} GiB needed",
        "Close apps that hold VRAM (listed above) or use a smaller profile",
    )


# Starting and waiting


def health_status(ctx):
    try:
        return ctx.http.request("GET", f"{SERVER}/health", timeout_s=HEALTH_TIMEOUT_S).status
    except ServerUnreachable:
        return None


def _wait_ready(ctx, t0, polled=False):
    """Poll /health until 200, the unit exits, or READY_TIMEOUT_S passes. Returns the final UnitState or None.

    polled=True means the caller's own /health check was the first poll and wasn't 200.
    """
    while True:
        if not polled and health_status(ctx) == 200:
            return unit_state(ctx)
        polled = False
        state = unit_state(ctx)
        if state.exited:
            return None
        if (ctx.clock.now() - t0).total_seconds() >= READY_TIMEOUT_S:
            return None
        ctx.clock.sleep(POLL_S)


def _give_up(ctx, t0, todo):
    systemctl(ctx, "stop", UNIT)
    since = int(t0.timestamp())
    tail = ctx.runner.run(
        ["journalctl", "--user", "-u", UNIT, "--no-pager", "-n", str(JOURNAL_LINES), "--since", f"@{since}"],
        timeout=TIMEOUT_S,
    )
    for line in tail.output.rstrip().splitlines():
        ctx.warn(line)
    raise Stop("the server didn't become ready, so it was stopped", todo)


def _report_ready(ctx, t0, state, name):
    seconds = round((ctx.clock.now() - t0).total_seconds())
    mib = server_vram_mib(ctx, state.pid) if state else None
    vram = f"{mib / 1024:.1f} GiB VRAM" if mib is not None else "VRAM unknown"
    ctx.say(f"qlm-server is ready in {seconds} s with the {name} profile, using {vram}.")


NOT_READY_TODO = (
    "Close apps that hold VRAM, or lower LLAMA_ARG_CTX_SIZE in the profile, then run qlm start again"
)


def preflight(ctx, check_port=True):
    """qlm start's checks before a start. A running server holds the port, so callers restarting it skip that one."""
    p = ctx.paths
    if not p.unit_file.exists() or not (p.llama_current / "bin/llama-server").exists():
        raise Stop("llama-server or the qlm-server unit is not installed", "Run qlm install")
    profile = active_profile(ctx)
    if not p.api_key.exists() or not p.api_key.read_text().strip():
        raise Stop(f"the API key file {p.api_key} is missing or empty", "Run qlm install")
    if b"\r" in p.api_key.read_bytes():  # read_text would turn \r\n into \n and hide it
        ctx.warn(
            f"warning: {p.api_key} contains a carriage return; llama-server keeps it as part of the key, "
            "so requests from qlm and other clients will be refused"
        )
    missing = missing_files(profile.settings)
    if missing:
        raise Stop("missing model file: " + ", ".join(path for _, path in missing), _fetch_hint(missing))
    holder = ctx.env.listener_pid(HOST, PORT) if check_port else None
    if holder is not None:
        who = f"PID {holder}" if holder != "unknown" else "another user's process"
        raise Stop(f"port {PORT} is in use by {who}", "Stop that process or free the port")
    return profile


def start(ctx, name=None):
    """Start the server with the selected model's `name` profile, or with active.env when it follows the selection."""
    p = ctx.paths
    if name is None:
        name = _profile_for_selection(ctx)
    if name is not None:
        before = p.active_env.read_bytes() if p.active_env.exists() else None
        profile_use(ctx, name)
        if p.active_env.read_bytes() != before and unit_state(ctx).running:
            return 0  # profile_use restarted the running server with the new profile and waited until it was ready
    return _start(ctx)


def _profile_for_selection(ctx):
    """The profile plain qlm start renders first, or None when active.env already follows the selection.

    A missing active.env, or a stock profile of another model, becomes the selected model's profile
    of the same name (or agent). A custom profile of another model is never overwritten.
    """
    p = ctx.paths
    model_id = selection.current(ctx)
    alias = selection.active_alias(ctx)
    if alias == model_id or (p.active_env.exists() and alias not in pins.model_ids()):
        return None  # active.env is this model's, or one qlm can't place; start it as it is
    previous = active_profile_name(ctx)
    if p.active_env.exists() and previous == "custom":
        raise Stop(
            f"active.env is a custom profile for {alias}, but {model_id} is selected",
            f"Run qlm start agent to replace it, or qlm use {alias}",
        )
    return previous if previous in model_profiles(ctx, model_id) else "agent"


def _start(ctx):
    t0 = ctx.clock.now()
    state = unit_state(ctx)
    name = active_profile_name(ctx)
    if state.running:
        if health_status(ctx) == 200:
            ctx.say(f"qlm-server is already running with the {name} profile.")
            return 0
        ready = _wait_ready(ctx, t0, polled=True)
        if ready is None:
            _give_up(ctx, t0, NOT_READY_TODO)
        _report_ready(ctx, t0, ready, name)
        return 0
    profile = preflight(ctx)
    check_ollama(ctx)
    check_vram(ctx, profile)
    systemctl(ctx, "reset-failed", UNIT)  # clears an earlier start limit; failure here is harmless
    r = systemctl(ctx, "start", UNIT)
    if r.code != 0:
        raise Stop(f"systemctl --user start failed: {r.output.strip()}", "Check the unit with qlm logs")
    ready = _wait_ready(ctx, t0)
    if ready is None:
        _give_up(ctx, t0, NOT_READY_TODO)
    _report_ready(ctx, t0, ready, name)
    return 0


def bring_up(ctx, restart):
    """Restart (or start) the unit and wait as qlm start does. True when ready; otherwise the unit is stopped.

    An earlier failed state is cleared first, so a start limit left by a bad profile doesn't block this start.
    """
    t0 = ctx.clock.now()
    systemctl(ctx, "reset-failed", UNIT)
    r = systemctl(ctx, "restart" if restart else "start", UNIT)
    ready = _wait_ready(ctx, t0) if r.code == 0 else None
    if ready is None:
        try:
            _give_up(ctx, t0, "")
        except Stop:
            pass  # _give_up already stopped the unit and printed the journal
        return False
    _report_ready(ctx, t0, ready, active_profile_name(ctx))
    return True


def profile_use(ctx, name, model_id=None):
    p = ctx.paths
    profile = profiles.load(p.model_profiles_dir(model_id or selection.current(ctx)) / f"{name}.env")
    text = profile.render(p.home)
    missing = missing_files(profiles.parse(name, text).settings)
    if missing:
        raise Stop("missing model file: " + ", ".join(path for _, path in missing), _fetch_hint(missing))
    if p.active_env.exists() and p.active_env.read_text() == text:
        ctx.say(f"The {name} profile is already active.")
        return 0
    state = unit_state(ctx)
    previous = active_profile_name(ctx)
    if state.running:
        check_ollama(ctx)
        check_vram(ctx, profile, credit_mib=server_vram_mib(ctx, state.pid) or 0)
    p.config_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    files.replace_atomically(p.active_env, text.encode(), 0o600)
    if not state.running:
        ctx.say(f"Switched to the {name} profile; it takes effect at the next qlm start.")
        return 0
    t0 = ctx.clock.now()
    back = f"qlm profile use {previous}" if previous != "custom" else "qlm profile use agent"
    r = systemctl(ctx, "restart", UNIT)
    if r.code != 0:
        raise Stop(f"systemctl --user restart failed: {r.output.strip()}", f"Go back with {back}")
    ready = _wait_ready(ctx, t0)
    if ready is None:
        _give_up(ctx, t0, f"The {name} profile didn't start. Go back with {back}")
    _report_ready(ctx, t0, ready, name)
    return 0


def use(ctx, model_id):
    """Select a model and render its profile into active.env, restarting a running server.

    The selection is saved only once the profile is in place, except when the model's files are
    missing: then only the selection is saved, so qlm fetch knows what to fetch.
    """
    p = ctx.paths
    try:
        selected = selection.current(ctx)
    except Stop:
        selected = None  # an unreadable selection is what qlm use replaces
    if selected == model_id and selection.active_alias(ctx) == model_id:
        ctx.say(f"{model_id} is already selected.")
        return 0
    available = model_profiles(ctx, model_id)
    previous = active_profile_name(ctx)
    name = previous if previous in available else "agent"
    replacing_custom = p.active_env.exists() and previous == "custom"
    missing = missing_files(profiles.parse(name, available[name].render(p.home)).settings)
    if missing:
        selection.save(ctx, model_id)
        raise Stop(
            f"{model_id} is selected but its files are missing: " + ", ".join(path for _, path in missing),
            f"{_fetch_hint(missing)}, then qlm start {name}",
        )
    if not p.selection.exists():
        selection.save(ctx, selected)  # else a failed restart leaves active.env, and so the selection, on model_id
    before = p.active_env.read_bytes() if p.active_env.exists() else None
    profile_use(ctx, name, model_id)
    selection.save(ctx, model_id)
    if replacing_custom:
        ctx.say(f"Replaced the custom active.env with the {name} profile of {model_id}.")
    if p.active_env.read_bytes() == before and not unit_state(ctx).running:
        # active.env already held this profile, as a failed restart in qlm use leaves it, so nothing said so yet
        ctx.say(f"{model_id} is selected. qlm-server is stopped; start it with qlm start.")
        return 0
    ctx.say(f"{model_id} is selected.")
    return 0


# stop, status, logs


def stop(ctx):
    state = unit_state(ctx)
    if state.active == "failed":
        systemctl(ctx, "reset-failed", UNIT)
        ctx.say("qlm-server was in the failed state; cleared it.")
        return 0
    if not state.running:
        ctx.say("qlm-server is not running.")
        return 0
    r = systemctl(ctx, "stop", UNIT)
    if r.code != 0:
        raise Stop(f"systemctl --user stop failed: {r.output.strip()}", "Check the unit with qlm logs")
    ctx.say("qlm-server stopped.")
    return 0


def _uptime(seconds):
    h, rest = divmod(max(int(seconds), 0), 3600)
    m, s = divmod(rest, 60)
    return f"{h}h {m}m {s}s"


def _largest_request(ctx):
    try:
        key = ctx.paths.api_key.read_text().strip()
        r = ctx.http.request(
            "GET", f"{SERVER}/metrics", headers={"Authorization": f"Bearer {key}"}, timeout_s=METRICS_TIMEOUT_S
        )
    except (OSError, ServerUnreachable) as e:
        return None, str(e)
    if r.status != 200:
        return None, f"HTTP {r.status}"
    m = re.search(r"^llamacpp:n_tokens_max\s+([0-9.e+]+)\s*$", r.body.decode(errors="replace"), re.M)
    if not m:
        return None, "llamacpp:n_tokens_max not in /metrics"
    return int(float(m.group(1))), None


def status(ctx):
    if not ctx.paths.unit_file.exists():
        raise Stop("the qlm-server unit is not installed", "Run qlm install")
    state = unit_state(ctx)
    ready = state.running and health_status(ctx) == 200
    label = "running" if ready else "loading" if state.running else STATE_LABELS.get(state.active, state.active)
    ctx.say(f"state: {label}")
    alias = selection.active_alias(ctx)
    ctx.say(f"model: {alias or ('unknown' if ctx.paths.active_env.exists() else 'none')}")
    try:
        selected = selection.current(ctx)
    except Stop:
        selected = f"invalid ({ctx.paths.selection} names {ctx.paths.selection.read_text().strip()!r})"
    if selected != alias:
        ctx.say(f"selected: {selected}")
    try:
        profile = active_profile(ctx)
        ctx.say(f"profile: {active_profile_name(ctx)}")
    except Stop as e:
        profile = None
        reason = e.what.split("doesn't parse: ", 1)[-1] if "doesn't parse" in e.what else e.what
        ctx.say(f"profile: custom (active.env doesn't parse: {reason})")
    if not state.running:
        return 0
    ctx.say(f"pid: {state.pid}")
    if state.entered is not None:
        ctx.say(f"uptime: {_uptime(ctx.clock.now().timestamp() - state.entered)}")
    if not ctx.runner.which("nvidia-smi"):
        ctx.say("vram: unavailable (nvidia-smi not found)")
    else:
        mib = server_vram_mib(ctx, state.pid)
        ctx.say(f"vram: {mib / 1024:.1f} GiB" if mib is not None else "vram: unavailable (no GPU process for the server)")
    if ready:
        largest, problem = _largest_request(ctx)
        ctx_size = profile.settings.get("LLAMA_ARG_CTX_SIZE") if profile else None
        if problem:
            ctx.say(f"context: unavailable ({problem})")
        elif ctx_size and ctx_size.isdigit():
            ctx.say(f"context: largest request {largest:,} of {int(ctx_size):,} tokens")
        else:
            ctx.say(f"context: largest request {largest:,} tokens")
    return 0


def logs(ctx, lines=50, follow=False):
    argv = ["journalctl", "--user", "-u", UNIT, "--no-pager", "-n", str(lines)]
    if follow:
        argv.append("-f")
    try:
        return ctx.runner.passthrough(argv)
    except KeyboardInterrupt:
        return 0


def run(ctx, command, **kwargs):
    """Run one of this module's commands, turning a Stop into the standard failure line."""
    try:
        return command(ctx, **kwargs)
    except Stop as stop_:
        return ctx.fail(stop_.what, stop_.todo)
    except BrokenPipeError:
        raise  # the reader went away (qlm status | head); cli.main ends quietly
    except OSError as e:
        return ctx.fail(f"file error: {e}", "Fix the cause and try again")
