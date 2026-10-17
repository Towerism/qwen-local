"""qlm profile use, start, stop, status and logs, run through cli.main() in a temporary home."""

import os
import stat

import pytest

from qlm import cli, profiles
from qlm.system import ServerUnreachable
from tests.fakes import FakeClock, FakeEnv, FakeRunner, FakeServer, fail, make_ctx, ok, reply

SHOW = ("systemctl", "--user", "show")
START = ("systemctl", "--user", "start", "qlm-server")
RESTART = ("systemctl", "--user", "restart", "qlm-server")
STOP = ("systemctl", "--user", "stop", "qlm-server")
RESET = ("systemctl", "--user", "reset-failed", "qlm-server")
JOURNAL = ("journalctl",)
SMI_GPU = ("nvidia-smi", "--query-gpu=memory.total,memory.used")
SMI_APPS = ("nvidia-smi", "--query-compute-apps=pid,process_name,used_memory")
OLLAMA_PS = ("ollama", "ps")
OLLAMA_HEADER = "NAME    ID    SIZE    PROCESSOR    CONTEXT    UNTIL\n"
NOW_UNIX = 1792238400  # FakeClock's default instant, 2026-10-17 12:00:00 UTC
REFUSED = ServerUnreachable("Connection refused")


# Factories


def unit(active="inactive", sub="dead", pid=0, entered=""):
    return ok(f"ActiveState={active}\nSubState={sub}\nMainPID={pid}\nActiveEnterTimestamp={entered}\n")


RUNNING = unit("active", "running", 4321, f"@{NOW_UNIX - 3725}")


def gpu(total=24564, used=2458):
    return ok(f"{total}, {used}\n")


def apps(*rows):
    return ok("".join(f"{pid}, {name}, {mib}\n" for pid, name, mib in rows))


def health(status=200):
    return reply(status, {"status": "ok"} if status == 200 else {"error": {"message": "Loading model"}})


def install(home, repo, profile="agent", key="k" * 43):
    """A home as `qlm install` and `qlm fetch` leave it: unit, build, profile, key and model files."""
    p = make_ctx(home, repo).paths
    p.unit_file.parent.mkdir(parents=True)
    p.unit_file.write_text("[Unit]\n")
    server = p.llama_current / "bin/llama-server"
    server.parent.mkdir(parents=True)
    server.write_text("")
    p.config_dir.mkdir(parents=True)
    p.active_env.write_text(profiles.load(p.profiles_dir / f"{profile}.env").render(home))
    os.chmod(p.active_env, 0o600)
    p.api_key.write_text(key)
    p.models.mkdir(parents=True)
    for name in ("Qwen3.8-27B-UD-Q4_K_M.gguf", "mtp-Qwen3.8-27B-Q4_0.gguf"):
        (p.models / name).write_text("")
    return p


def rendered(home, repo, name):
    return profiles.load(repo / "profiles" / f"{name}.env").render(home)


def runner(overrides=None):
    overrides = overrides or {}
    script = {
        SHOW: unit(),
        START: ok(),
        RESTART: ok(),
        STOP: ok(),
        RESET: ok(),
        JOURNAL: ok("llama-server: failed to allocate CUDA buffer\n"),
        SMI_GPU: gpu(),
        SMI_APPS: apps(("1200", "/usr/bin/kwin_wayland", "300")),
        OLLAMA_PS: ok(OLLAMA_HEADER),
    }
    script.update({k: v for k, v in overrides.items() if not isinstance(k, str)})
    which = {"nvidia-smi": "/usr/bin/nvidia-smi", "ollama": "/usr/bin/ollama"}
    which.update(overrides.get("which", {}))
    return FakeRunner(script, which=which)


def ctx_for(home, repo, run=None, server=None, env=None):
    return make_ctx(
        home, repo, runner=run or runner(), http=server or FakeServer({("GET", "/health"): health()}),
        clock=FakeClock(advance=True), env=env,
    )


def out(ctx):
    return ctx.out.getvalue() + ctx.err.getvalue()


# profile use


def test_unknown_profile_is_a_usage_error_listing_the_profiles(home, repo):
    ctx = ctx_for(home, repo)
    assert cli.main(["profile", "use", "turbo"], ctx) == 2
    assert "agent, long, vision" in out(ctx)


def test_vision_without_the_projector_changes_nothing(home, repo):
    p = install(home, repo)
    before = p.active_env.read_bytes()
    ctx = ctx_for(home, repo)
    assert cli.main(["profile", "use", "vision"], ctx) == 1
    assert "qlm fetch --vision" in out(ctx)
    assert p.active_env.read_bytes() == before


def test_the_active_profile_is_left_alone_without_a_restart(home, repo):
    install(home, repo)
    run = runner({SHOW: RUNNING})
    ctx = ctx_for(home, repo, run=run)
    assert cli.main(["profile", "use", "agent"], ctx) == 0
    assert "already active" in out(ctx)
    assert run.ran(*RESTART) == []


def test_switching_while_stopped_writes_the_profile_for_the_next_start(home, repo):
    p = install(home, repo)
    run = runner()
    ctx = ctx_for(home, repo, run=run)
    assert cli.main(["profile", "use", "long"], ctx) == 0
    assert p.active_env.read_text() == rendered(home, repo, "long")
    assert stat.S_IMODE(p.active_env.stat().st_mode) == 0o600
    assert run.ran(*RESTART) == []
    assert "next qlm start" in out(ctx)


def test_switching_while_running_counts_the_servers_own_vram_as_free(home, repo):
    p = install(home, repo)
    run = runner({
        SHOW: RUNNING,
        SMI_GPU: gpu(24564, 23400),
        SMI_APPS: apps(("4321", "llama-server", "20800"), ("1200", "kwin_wayland", "2600")),
    })
    ctx = ctx_for(home, repo, run=run)
    assert cli.main(["profile", "use", "long"], ctx) == 0
    assert p.active_env.read_text() == rendered(home, repo, "long")
    assert len(run.ran(*RESTART)) == 1
    assert "ready" in out(ctx)


def test_switching_refuses_when_even_the_freed_vram_is_not_enough(home, repo):
    p = install(home, repo)
    before = p.active_env.read_bytes()
    run = runner({SHOW: RUNNING, SMI_GPU: gpu(24564, 24000), SMI_APPS: apps(("4321", "llama-server", "20000"))})
    ctx = ctx_for(home, repo, run=run)
    assert cli.main(["profile", "use", "long"], ctx) == 1
    assert p.active_env.read_bytes() == before
    assert run.ran(*RESTART) == []


def test_switching_refuses_while_ollama_holds_a_gpu_model(home, repo):
    p = install(home, repo)
    before = p.active_env.read_bytes()
    run = runner({SHOW: RUNNING, OLLAMA_PS: ok(OLLAMA_HEADER + "qwen3:8b  abc  6 GB  100% GPU  4096  4 minutes\n")})
    ctx = ctx_for(home, repo, run=run)
    assert cli.main(["profile", "use", "long"], ctx) == 1
    assert "ollama stop qwen3:8b" in out(ctx)
    assert p.active_env.read_bytes() == before


def test_a_failed_restart_stops_the_unit_and_names_the_way_back(home, repo):
    install(home, repo)
    run = runner({SHOW: [RUNNING, RUNNING, unit("failed", "failed", 0)], SMI_APPS: apps(("4321", "llama-server", "20800"))})
    server = FakeServer({("GET", "/health"): REFUSED})
    ctx = ctx_for(home, repo, run=run, server=server)
    assert cli.main(["profile", "use", "long"], ctx) == 1
    assert len(run.ran(*STOP)) == 1
    assert "failed to allocate CUDA buffer" in out(ctx)
    assert "qlm profile use agent" in out(ctx)


# start: already running


def test_start_when_already_ready_does_nothing(home, repo):
    install(home, repo)
    run = runner({SHOW: RUNNING})
    ctx = ctx_for(home, repo, run=run)
    assert cli.main(["start"], ctx) == 0
    assert "already running" in out(ctx) and "agent" in out(ctx)
    assert run.ran(*START) == []


def test_start_while_loading_waits_without_starting_again(home, repo):
    install(home, repo)
    run = runner({SHOW: unit("active", "running", 4321, f"@{NOW_UNIX}"), SMI_APPS: apps(("4321", "llama-server", "21094"))})
    server = FakeServer({("GET", "/health"): [health(503), health(200)]})
    ctx = ctx_for(home, repo, run=run, server=server)
    assert cli.main(["start"], ctx) == 0
    assert run.ran(*START) == []
    assert "ready in 1 s" in out(ctx)


# start: preconditions


@pytest.mark.parametrize(
    "breakage, fix",
    [
        ("unit", "qlm install"),
        ("active_env", "qlm profile use agent"),
        ("key", "qlm install"),
        ("model", "qlm fetch"),
        ("port", "PID 4242"),
    ],
)
def test_start_refuses_before_starting_anything(home, repo, breakage, fix):
    p = install(home, repo)
    env = FakeEnv(path=[str(p.local_bin)])
    if breakage == "unit":
        p.unit_file.unlink()
    elif breakage == "active_env":
        p.active_env.unlink()
    elif breakage == "key":
        p.api_key.write_text("")
    elif breakage == "model":
        (p.models / "Qwen3.8-27B-UD-Q4_K_M.gguf").unlink()
    else:
        env.listener = 4242
    run = runner()
    ctx = ctx_for(home, repo, run=run, env=env)
    assert cli.main(["start"], ctx) == 1
    assert fix in out(ctx)
    assert run.ran(*START) == []


def test_start_refuses_an_active_env_that_does_not_parse(home, repo):
    p = install(home, repo)
    p.active_env.write_text("# qlm: vram_gib=20.3\nnot a setting\n")
    ctx = ctx_for(home, repo)
    assert cli.main(["start"], ctx) == 1
    assert "line 2" in out(ctx) and "qlm profile use agent" in out(ctx)


def test_start_warns_about_a_carriage_return_in_the_key_file(home, repo):
    p = install(home, repo)
    p.api_key.write_text("k" * 43 + "\r\n")
    run = runner({SHOW: [unit(), unit("active", "running", 4321, f"@{NOW_UNIX}")]})
    ctx = ctx_for(home, repo, run=run)
    assert cli.main(["start"], ctx) == 0
    assert "carriage return" in ctx.err.getvalue()


# start: Ollama and VRAM


def test_a_gpu_model_in_ollama_blocks_the_start_but_a_cpu_model_does_not(home, repo):
    install(home, repo)
    listing = OLLAMA_HEADER + "qwen3.8:27b  a1  17 GB  100% GPU  4096  4 minutes\ncon-suggest:latest  b2  1 GB  100% CPU  4096  4 minutes\n"
    run = runner({OLLAMA_PS: ok(listing)})
    ctx = ctx_for(home, repo, run=run)
    assert cli.main(["start"], ctx) == 1
    assert "ollama stop qwen3.8:27b" in out(ctx)
    assert "con-suggest" not in out(ctx)
    assert run.ran(*START) == []


@pytest.mark.parametrize(
    "ollama",
    [
        {OLLAMA_PS: ok(OLLAMA_HEADER + "con-suggest:latest  b2  1 GB  100% CPU  4096  4 minutes\n")},
        {OLLAMA_PS: fail(stderr="could not connect to ollama app")},
        {"which": {"ollama": None}},
    ],
)
def test_the_ollama_check_passes_without_a_gpu_model(home, repo, ollama):
    install(home, repo)
    run = runner({SHOW: [unit(), unit("active", "running", 4321, f"@{NOW_UNIX}")], **ollama})
    ctx = ctx_for(home, repo, run=run)
    assert cli.main(["start"], ctx) == 0
    assert len(run.ran(*START)) == 1


def test_not_enough_vram_lists_who_holds_it(home, repo):
    install(home, repo)
    run = runner({SMI_GPU: gpu(24564, 4500), SMI_APPS: apps(("1200", "kwin_wayland", "1800"), ("3300", "firefox, web", "2700"))})
    ctx = ctx_for(home, repo, run=run)
    assert cli.main(["start"], ctx) == 1
    text = out(ctx)
    assert "19.6 GiB free, 20.8 GiB needed" in text
    assert "kwin_wayland (1800 MiB)" in text and "firefox, web (2700 MiB)" in text
    assert run.ran(*START) == []


def test_just_enough_vram_starts(home, repo):
    install(home, repo)
    run = runner({SMI_GPU: gpu(24564, 3000), SHOW: [unit(), unit("active", "running", 4321, f"@{NOW_UNIX}")]})
    ctx = ctx_for(home, repo, run=run)
    assert cli.main(["start"], ctx) == 0


@pytest.mark.parametrize("smi", [{"which": {"nvidia-smi": None}}, {SMI_GPU: ok("garbage\n")}])
def test_start_fails_without_a_readable_gpu(home, repo, smi):
    install(home, repo)
    run = runner(smi)
    ctx = ctx_for(home, repo, run=run)
    assert cli.main(["start"], ctx) == 1
    assert "nvidia-smi" in out(ctx)
    assert run.ran(*START) == []


# start: waiting


def test_start_waits_for_health_and_reports_time_and_vram(home, repo):
    install(home, repo)
    run = runner({
        SHOW: [unit(), unit("activating", "start", 4321), unit("active", "running", 4321, f"@{NOW_UNIX}")],
        SMI_APPS: apps(("1200", "kwin_wayland", "300"), ("4321", "llama-server", "21094")),
    })
    server = FakeServer({("GET", "/health"): [health(503), health(503), health(503), health(200)]})
    ctx = ctx_for(home, repo, run=run, server=server)
    assert cli.main(["start"], ctx) == 0
    argvs = [c["argv"] for c in run.calls]
    assert argvs.index(list(RESET)) < argvs.index(list(START))
    assert len(server.to("GET", "/health")) == 4
    assert "agent" in out(ctx) and "ready in 3 s" in out(ctx) and "20.6 GiB" in out(ctx)


def test_start_gives_up_after_180_seconds_and_stops_the_unit(home, repo):
    install(home, repo)
    run = runner({SHOW: [unit(), unit("activating", "start", 4321)]})
    server = FakeServer({("GET", "/health"): health(503)})
    ctx = ctx_for(home, repo, run=run, server=server)
    assert cli.main(["start"], ctx) == 1
    assert sum(ctx.clock.sleeps) == 180
    assert len(run.ran(*STOP)) == 1
    assert "failed to allocate CUDA buffer" in out(ctx)
    assert "LLAMA_ARG_CTX_SIZE" in out(ctx)


def test_start_stops_waiting_as_soon_as_the_server_exits(home, repo):
    install(home, repo)
    run = runner({SHOW: [unit(), unit("activating", "start", 4321), unit("activating", "auto-restart", 0)]})
    server = FakeServer({("GET", "/health"): REFUSED})
    ctx = ctx_for(home, repo, run=run, server=server)
    assert cli.main(["start"], ctx) == 1
    assert len(server.to("GET", "/health")) == 2
    assert sum(ctx.clock.sleeps) == 1
    assert len(run.ran(*STOP)) == 1


def test_journal_tail_covers_only_this_start(home, repo):
    install(home, repo)
    run = runner({SHOW: [unit(), unit("failed", "failed", 0)]})
    ctx = ctx_for(home, repo, run=run, server=FakeServer({("GET", "/health"): REFUSED}))
    cli.main(["start"], ctx)
    journal = run.ran(*JOURNAL)[0]
    assert journal[:6] == ["journalctl", "--user", "-u", "qlm-server", "--no-pager", "-n"]
    assert journal[6] == "30" and journal[journal.index("--since") + 1] == f"@{NOW_UNIX}"


def test_vram_is_unknown_when_no_process_matches_the_main_pid(home, repo):
    install(home, repo)
    run = runner({SHOW: [unit(), unit("active", "running", 4321, f"@{NOW_UNIX}")]})
    ctx = ctx_for(home, repo, run=run)
    assert cli.main(["start"], ctx) == 0
    assert "VRAM unknown" in out(ctx)


def test_a_failing_systemctl_start_is_reported(home, repo):
    install(home, repo)
    run = runner({START: fail(stderr="Unit qlm-server.service not found.")})
    ctx = ctx_for(home, repo, run=run)
    assert cli.main(["start"], ctx) == 1
    assert "Unit qlm-server.service not found." in out(ctx)


def test_an_unreachable_user_manager_is_explained(home, repo):
    install(home, repo)
    run = runner({SHOW: fail(stderr="Failed to connect to bus: No medium found")})
    ctx = ctx_for(home, repo, run=run)
    assert cli.main(["start"], ctx) == 1
    assert "No medium found" in out(ctx) and "desktop session" in out(ctx)


# stop


def test_stop_when_stopped_does_nothing(home, repo):
    install(home, repo)
    run = runner()
    ctx = ctx_for(home, repo, run=run)
    assert cli.main(["stop"], ctx) == 0
    assert "not running" in out(ctx)
    assert run.ran(*STOP) == []


def test_stop_stops_a_running_server(home, repo):
    install(home, repo)
    run = runner({SHOW: RUNNING})
    ctx = ctx_for(home, repo, run=run)
    assert cli.main(["stop"], ctx) == 0
    assert len(run.ran(*STOP)) == 1


def test_stop_clears_a_failed_unit_without_stopping(home, repo):
    install(home, repo)
    run = runner({SHOW: unit("failed", "failed", 0)})
    ctx = ctx_for(home, repo, run=run)
    assert cli.main(["stop"], ctx) == 0
    assert len(run.ran(*RESET)) == 1 and run.ran(*STOP) == []
    assert "cleared" in out(ctx)


def test_stop_reports_a_systemctl_failure(home, repo):
    install(home, repo)
    run = runner({SHOW: RUNNING, STOP: fail(stderr="Access denied")})
    ctx = ctx_for(home, repo, run=run)
    assert cli.main(["stop"], ctx) == 1
    assert "Access denied" in out(ctx)


# status


def metrics(n=12345):
    return reply(200, f"# HELP llamacpp:n_tokens_max x\nllamacpp:prompt_tokens_total 99\nllamacpp:n_tokens_max {n}\n")


def test_status_of_a_ready_server(home, repo):
    install(home, repo)
    run = runner({SHOW: RUNNING, SMI_APPS: apps(("4321", "llama-server", "21094"))})
    server = FakeServer({("GET", "/health"): health(), ("GET", "/metrics"): metrics()})
    ctx = ctx_for(home, repo, run=run, server=server)
    assert cli.main(["status"], ctx) == 0
    lines = ctx.out.getvalue().splitlines()
    assert lines == [
        "state: running",
        "profile: agent",
        "pid: 4321",
        "uptime: 1h 2m 5s",
        "vram: 20.6 GiB",
        "context: largest request 12,345 of 65,536 tokens",
    ]
    assert server.to("GET", "/metrics")[0]["headers"] == {"Authorization": "Bearer " + "k" * 43}


def test_status_names_a_switched_profile_and_a_hand_edited_one(home, repo):
    p = install(home, repo, profile="long")
    ctx = ctx_for(home, repo)
    cli.main(["status"], ctx)
    assert "profile: long" in ctx.out.getvalue()
    p.active_env.write_text(p.active_env.read_text().replace("98304", "90000"))
    ctx = ctx_for(home, repo)
    cli.main(["status"], ctx)
    assert "profile: custom" in ctx.out.getvalue()


def test_status_while_loading_has_no_context_line(home, repo):
    install(home, repo)
    run = runner({SHOW: RUNNING})
    ctx = ctx_for(home, repo, run=run, server=FakeServer({("GET", "/health"): health(503)}))
    assert cli.main(["status"], ctx) == 0
    assert "state: loading" in ctx.out.getvalue()
    assert "context:" not in ctx.out.getvalue()


def test_status_when_stopped(home, repo):
    install(home, repo)
    ctx = ctx_for(home, repo)
    assert cli.main(["status"], ctx) == 0
    assert ctx.out.getvalue().splitlines() == ["state: stopped", "profile: agent"]


@pytest.mark.parametrize("active, label", [("failed", "failed"), ("deactivating", "stopping")])
def test_status_labels_other_states(home, repo, active, label):
    install(home, repo)
    ctx = ctx_for(home, repo, run=runner({SHOW: unit(active, "x", 0)}))
    assert cli.main(["status"], ctx) == 0
    assert f"state: {label}" in ctx.out.getvalue()


def test_status_shows_unavailable_metrics_and_vram_without_failing(home, repo):
    install(home, repo)
    run = runner({SHOW: RUNNING, "which": {"nvidia-smi": None}})
    server = FakeServer({("GET", "/health"): health(), ("GET", "/metrics"): reply(500, "boom")})
    ctx = ctx_for(home, repo, run=run, server=server)
    assert cli.main(["status"], ctx) == 0
    assert "context: unavailable (HTTP 500)" in ctx.out.getvalue()
    assert "vram: unavailable" in ctx.out.getvalue()


def test_status_without_the_unit_file_fails(home, repo):
    p = install(home, repo)
    p.unit_file.unlink()
    ctx = ctx_for(home, repo)
    assert cli.main(["status"], ctx) == 1
    assert "qlm install" in out(ctx)


def test_status_with_an_unparseable_active_env(home, repo):
    p = install(home, repo)
    p.active_env.write_text("nonsense\n")
    ctx = ctx_for(home, repo)
    assert cli.main(["status"], ctx) == 0
    assert "profile: custom (active.env doesn't parse:" in ctx.out.getvalue()


# logs


def test_logs_defaults_to_fifty_lines(home, repo):
    run = FakeRunner({JOURNAL: ok()})
    ctx = ctx_for(home, repo, run=run)
    assert cli.main(["logs"], ctx) == 0
    assert run.passthrough_calls == [["journalctl", "--user", "-u", "qlm-server", "--no-pager", "-n", "50"]]


def test_logs_follows_and_returns_journalctls_exit_code(home, repo):
    run = FakeRunner({JOURNAL: fail(code=4)})
    ctx = ctx_for(home, repo, run=run)
    assert cli.main(["logs", "-n", "200", "-f"], ctx) == 4
    assert run.passthrough_calls == [["journalctl", "--user", "-u", "qlm-server", "--no-pager", "-n", "200", "-f"]]


def test_ctrl_c_while_following_logs_exits_cleanly(home, repo):
    run = FakeRunner({JOURNAL: KeyboardInterrupt()})
    ctx = ctx_for(home, repo, run=run)
    assert cli.main(["logs", "-f"], ctx) == 0


def test_logs_rejects_a_line_count_below_one(home, repo):
    ctx = ctx_for(home, repo)
    assert cli.main(["logs", "-n", "0"], ctx) == 2


# more failure paths


def test_a_refused_restart_says_how_to_go_back(home, repo):
    install(home, repo)
    run = runner({SHOW: RUNNING, RESTART: fail(stderr="Job failed"), SMI_APPS: apps(("4321", "llama-server", "20800"))})
    ctx = ctx_for(home, repo, run=run)
    assert cli.main(["profile", "use", "long"], ctx) == 1
    assert "Job failed" in out(ctx) and "qlm profile use agent" in out(ctx)


def test_going_back_from_a_hand_edited_profile_suggests_agent(home, repo):
    p = install(home, repo)
    p.active_env.write_text(p.active_env.read_text().replace("65536", "60000"))
    run = runner({SHOW: [RUNNING, RUNNING, unit("failed", "failed", 0)], SMI_APPS: apps(("4321", "llama-server", "20800"))})
    ctx = ctx_for(home, repo, run=run, server=FakeServer({("GET", "/health"): REFUSED}))
    assert cli.main(["profile", "use", "long"], ctx) == 1
    assert "qlm profile use agent" in out(ctx)


def test_start_gives_up_on_a_server_that_was_already_loading(home, repo):
    install(home, repo)
    run = runner({SHOW: [unit("activating", "start", 4321), unit("failed", "failed", 0)]})
    ctx = ctx_for(home, repo, run=run, server=FakeServer({("GET", "/health"): health(503)}))
    assert cli.main(["start"], ctx) == 1
    assert run.ran(*START) == [] and len(run.ran(*STOP)) == 1


def test_a_port_held_by_a_hidden_process_is_named_as_such(home, repo):
    install(home, repo)
    ctx = ctx_for(home, repo, env=FakeEnv(listener="unknown"))
    assert cli.main(["start"], ctx) == 1
    assert "another user's process" in out(ctx)
