"""qlm update: build a new llama.cpp tag beside the old one, switch, smoke-test, and roll back on failure."""

import os
import shutil
import signal

import pytest

from qlm import cli, interrupts, pins
from qlm.system import ServerUnreachable
from tests.fakes import FakeClock, FakeEnv, fail, make_ctx, ok, reply
from tests.test_install import help_text
from tests.test_service import (
    OLLAMA_HEADER, OLLAMA_PS, RESET, RESTART, RUNNING, SHOW, SMI_APPS, SMI_GPU, START, STOP, apps, gpu, install,
    runner, unit,
)
from tests.test_smoke import make_server

OLD, NEW = pins.LLAMA_TAG, "b11700"
NEW_COMMIT = "a1" * 20
CONFIGURE = ("cmake", "-S")
BUILD = ("cmake", "--build")
UPDATE = ["update", "--tag", NEW, "--commit", NEW_COMMIT]


def installed(home, repo):
    """A home after qlm install: like test_service.install, but with current a symlink to the old tag."""
    p = install(home, repo)
    shutil.rmtree(p.llama_current)
    server = p.tag_dir(OLD) / "bin/llama-server"
    server.parent.mkdir(parents=True)
    server.write_bytes(b"server-old")
    p.llama_current.symlink_to(p.tag_dir(OLD))
    return p


def update_runner(home, repo, p, commit=NEW_COMMIT, help_ok=True, running=True, overrides=None):
    """service's fake runner plus git and cmake building NEW, as test_install's Machine does for OLD."""
    src = str(p.llama_src)
    build_dir = p.build_dir(NEW)
    staged = p.tag_dir(NEW) / "bin" / ".staged-llama-server"

    def build(argv):
        (build_dir / "bin").mkdir(parents=True, exist_ok=True)
        (build_dir / "bin" / "llama-server").write_bytes(b"server-new")
        (build_dir / "bin" / "llama-bench").write_bytes(b"bench")
        return ok()

    script = {
        ("git", "-C", src, "fetch"): ok(),
        ("git", "-C", src, "rev-parse"): ok(commit + "\n"),
        ("git", "-C", src, "checkout"): ok(),
        CONFIGURE: ok(),
        BUILD: build,
        (str(staged), "--help"): ok(help_text(repo) if help_ok else help_text(repo, drop=("LLAMA_ARG_SPEC_TYPE",))),
        SHOW: RUNNING if running else unit(),
        SMI_APPS: apps(("4321", "llama-server", "19860"), ("1200", "/usr/bin/kwin_wayland", "300")),
    }
    script.update(overrides or {})
    (p.llama_src / ".git").mkdir(parents=True, exist_ok=True)
    return runner(script)


def run_update(home, repo, run, server=None, args=UPDATE, env=None):
    ctx = make_ctx(home, repo, runner=run, http=server or make_server(), clock=FakeClock(advance=True), env=env)
    code = cli.main(args, ctx)
    return code, ctx


def out(ctx):
    return ctx.out.getvalue() + ctx.err.getvalue()


def target(p):
    return os.readlink(p.llama_current)


# Arguments and checks before building


@pytest.mark.parametrize("args", [
    ["update", "--tag", "11700", "--commit", NEW_COMMIT],
    ["update", "--tag", NEW, "--commit", "abc"],
    ["update", "--tag", NEW, "--commit", NEW_COMMIT.upper()],
    ["update", "--tag", NEW],
    ["update", "--commit", NEW_COMMIT],
])
def test_bad_or_missing_options_are_a_usage_error(home, repo, args):
    ctx = make_ctx(home, repo)
    assert cli.main(args, ctx) == 2
    assert ctx.runner.calls == []


def test_without_current_it_says_to_run_install(home, repo):
    p = installed(home, repo)
    p.llama_current.unlink()
    run = update_runner(home, repo, p)
    code, ctx = run_update(home, repo, run)
    assert code == 1
    assert "qlm install" in ctx.err.getvalue()
    assert run.ran("git") == []


def test_a_current_outside_the_build_directory_is_refused(home, repo, tmp_path):
    p = installed(home, repo)
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "bin").mkdir(parents=True)
    (elsewhere / "bin/llama-server").write_bytes(b"x")
    p.llama_current.unlink()
    p.llama_current.symlink_to(elsewhere)
    run = update_runner(home, repo, p)
    code, ctx = run_update(home, repo, run)
    assert code == 1 and "qlm install" in ctx.err.getvalue()
    assert run.ran("git") == []


def test_already_on_the_tag_does_nothing(home, repo):
    p = installed(home, repo)
    run = update_runner(home, repo, p)
    code, ctx = run_update(home, repo, run, args=["update", "--tag", OLD, "--commit", NEW_COMMIT])
    assert code == 0
    assert f"already on {OLD}" in out(ctx)
    assert run.ran("git") == []


def test_a_leftover_bench_backup_blocks_the_update(home, repo):
    p = installed(home, repo)
    (p.config_dir / "active.env.bench-backup").write_text("x")
    run = update_runner(home, repo, p)
    code, ctx = run_update(home, repo, run)
    assert code == 1
    assert "mv " in ctx.err.getvalue() and "active.env.bench-backup" in ctx.err.getvalue()
    assert run.ran("git") == []


def test_the_first_line_warns_about_the_restart(home, repo):
    p = installed(home, repo)
    code, ctx = run_update(home, repo, update_runner(home, repo, p))
    assert code == 0, out(ctx)
    assert "restarts the server" in ctx.out.getvalue().splitlines()[0]


# Build failures change nothing


def test_a_moved_tag_stops_before_building(home, repo):
    p = installed(home, repo)
    run = update_runner(home, repo, p, commit="de" * 20)
    code, ctx = run_update(home, repo, run)
    assert code == 1
    assert "de" * 20 in ctx.err.getvalue() and NEW_COMMIT in ctx.err.getvalue()
    assert run.ran("cmake") == []
    assert target(p) == str(p.tag_dir(OLD))


def test_a_failed_help_check_keeps_current_and_the_server(home, repo):
    p = installed(home, repo)
    run = update_runner(home, repo, p, help_ok=False)
    code, ctx = run_update(home, repo, run)
    assert code == 1
    assert "LLAMA_ARG_SPEC_TYPE" in ctx.err.getvalue()
    assert target(p) == str(p.tag_dir(OLD))
    assert [f.name for f in (p.tag_dir(NEW) / "bin").iterdir() if f.name.startswith(".staged")] == []
    assert run.ran(*RESTART) == [] and run.ran(*START) == []


def test_the_build_log_is_named_for_update(home, repo):
    p = installed(home, repo)
    run_update(home, repo, update_runner(home, repo, p))
    assert [f.name for f in p.logs.iterdir()] == ["update-20261017T120000Z.log"]


# Preflight before the switch


def test_not_enough_vram_keeps_current_and_the_server(home, repo):
    p = installed(home, repo)
    run = update_runner(home, repo, p, overrides={
        SMI_GPU: gpu(used=24000), SMI_APPS: apps(("4321", "llama-server", "100")),
    })
    code, ctx = run_update(home, repo, run)
    assert code == 1
    assert "not enough VRAM" in ctx.err.getvalue()
    assert target(p) == str(p.tag_dir(OLD))
    assert run.ran(*RESTART) == []


def test_the_running_servers_vram_counts_as_free(home, repo):
    p = installed(home, repo)
    run = update_runner(home, repo, p, overrides={SMI_GPU: gpu(used=22500)})
    code, ctx = run_update(home, repo, run)
    assert code == 0, out(ctx)


def test_a_missing_key_with_the_server_stopped_fails_before_the_switch(home, repo):
    p = installed(home, repo)
    p.api_key.unlink()
    run = update_runner(home, repo, p, running=False)
    server = make_server()
    code, ctx = run_update(home, repo, run, server=server)
    assert code == 1
    assert "qlm install" in ctx.err.getvalue()
    assert target(p) == str(p.tag_dir(OLD))
    assert server.requests == []


def test_a_held_port_with_the_server_stopped_fails_before_the_switch(home, repo):
    p = installed(home, repo)
    run = update_runner(home, repo, p, running=False)
    code, ctx = run_update(home, repo, run, env=FakeEnv(listener=999))
    assert code == 1
    assert "port 9931" in ctx.err.getvalue()
    assert target(p) == str(p.tag_dir(OLD))


def test_ollama_on_the_gpu_keeps_current(home, repo):
    p = installed(home, repo)
    run = update_runner(home, repo, p, overrides={OLLAMA_PS: ok(OLLAMA_HEADER + "qwen  abc  9 GB  100% GPU  4096  4 min\n")})
    code, ctx = run_update(home, repo, run)
    assert code == 1
    assert target(p) == str(p.tag_dir(OLD))


# Success


def test_a_running_server_moves_to_the_new_tag_and_keeps_running(home, repo):
    p = installed(home, repo)
    run = update_runner(home, repo, p)
    code, ctx = run_update(home, repo, run)
    assert code == 0, out(ctx)
    assert target(p) == str(p.tag_dir(NEW))
    assert len(run.ran(*RESTART)) == 1
    assert run.ran(*STOP) == []
    text = out(ctx)
    assert f"from {OLD} to {NEW}" in text
    assert f"still pin {pins.LLAMA_TAG}" in text and "qlm/pins.py" in text and "ADR 0001" in text
    assert "k" * 43 not in text


def test_a_stopped_server_is_started_for_smoke_and_stopped_again(home, repo):
    p = installed(home, repo)
    run = update_runner(home, repo, p, running=False, overrides={SHOW: [unit(), RUNNING]})
    code, ctx = run_update(home, repo, run)
    assert code == 0, out(ctx)
    assert run.ran(*RESET) and run.ran(*START)
    assert run.ran(*RESTART) == []
    assert len(run.ran(*STOP)) == 1
    names = [c["argv"][2] for c in run.calls if c["argv"][:2] == ["systemctl", "--user"] and c["argv"][2] in ("start", "stop")]
    assert names == ["start", "stop"]


def test_an_earlier_rolled_back_tag_is_rebuilt(home, repo):
    p = installed(home, repo)
    old_new = p.tag_dir(NEW) / "bin/llama-server"
    old_new.parent.mkdir(parents=True)
    old_new.write_bytes(b"server-stale")
    code, ctx = run_update(home, repo, update_runner(home, repo, p))
    assert code == 0, out(ctx)
    assert old_new.read_bytes() == b"server-new"


# Rollback


def with_health(server, *statuses):
    """Answer /health with these statuses in turn, the last one repeated."""
    server.replies[("GET", "/health")] = [reply(st, {"status": "ok"} if st == 200 else {}) for st in statuses]
    return server


def broken_tool_call():
    body = {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "no tools today"}}]}
    return make_server(tool_call=body)


def test_a_smoke_failure_rolls_back_and_restarts_the_old_tag(home, repo):
    p = installed(home, repo)
    run = update_runner(home, repo, p)
    code, ctx = run_update(home, repo, run, server=broken_tool_call())
    assert code == 1
    assert target(p) == str(p.tag_dir(OLD))
    assert len(run.ran(*RESTART)) == 2
    assert (p.tag_dir(NEW) / "bin/llama-server").exists()
    last = ctx.err.getvalue().rstrip().splitlines()[-1]
    assert "tool call" in last and f"back on {OLD}" in last


def test_a_smoke_failure_with_the_server_stopped_before_stops_it(home, repo):
    p = installed(home, repo)
    run = update_runner(home, repo, p, running=False, overrides={SHOW: [unit(), RUNNING]})
    code, ctx = run_update(home, repo, run, server=broken_tool_call())
    assert code == 1
    assert target(p) == str(p.tag_dir(OLD))
    assert run.ran(*RESTART) == []
    assert len(run.ran(*STOP)) == 1


def test_a_new_server_that_never_gets_ready_rolls_back_without_smoke(home, repo):
    p = installed(home, repo)
    run = update_runner(home, repo, p, overrides={SHOW: [RUNNING, unit("failed", "failed"), RUNNING]})
    server = with_health(make_server(), 503, 200)
    code, ctx = run_update(home, repo, run, server=server)
    assert code == 1
    assert target(p) == str(p.tag_dir(OLD))
    assert server.to("POST", "/v1/chat/completions") == []
    assert "didn't become ready" in ctx.err.getvalue()


def test_when_the_old_tag_fails_too_it_says_both_failed(home, repo):
    p = installed(home, repo)
    run = update_runner(home, repo, p, overrides={SHOW: [RUNNING, unit("failed", "failed")]})
    server = with_health(make_server(), 503)
    code, ctx = run_update(home, repo, run, server=server)
    assert code == 1
    assert target(p) == str(p.tag_dir(OLD))
    last = ctx.err.getvalue().rstrip().splitlines()[-1]
    assert "qlm start" in last and "qlm logs" in last and "either" in last


def test_ctrl_c_during_smoke_rolls_back(home, repo):
    p = installed(home, repo)
    run = update_runner(home, repo, p)
    server = make_server(models=KeyboardInterrupt())
    code, ctx = run_update(home, repo, run, server=server)
    assert code == 1
    assert target(p) == str(p.tag_dir(OLD))
    assert len(run.ran(*RESTART)) == 2
    assert "interrupted" in ctx.err.getvalue()


def test_ctrl_c_during_the_build_changes_nothing(home, repo):
    p = installed(home, repo)
    run = update_runner(home, repo, p, overrides={BUILD: KeyboardInterrupt()})
    code, ctx = run_update(home, repo, run)
    assert code == 1
    assert "interrupted; nothing was changed" in ctx.err.getvalue()
    assert "Traceback" not in out(ctx)
    assert target(p) == str(p.tag_dir(OLD))
    assert run.ran(*RESTART) == []


def test_signal_handlers_are_put_back_after_an_update(home, repo):
    before = signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)
    p = installed(home, repo)
    run_update(home, repo, update_runner(home, repo, p), server=broken_tool_call())
    assert (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)) == before


def test_sigterm_is_turned_into_an_interrupt():
    with pytest.raises(KeyboardInterrupt):
        interrupts.raise_interrupt(signal.SIGTERM, None)


def test_ctrl_c_and_sigterm_are_ignored_during_the_rollback(home, repo):
    p = installed(home, repo)
    handlers = []
    run = update_runner(home, repo, p)

    def restart(argv):
        handlers.append((signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)))
        return ok()

    run.script[RESTART] = restart
    code, ctx = run_update(home, repo, run, server=broken_tool_call())
    assert code == 1
    assert handlers[0] != (signal.SIG_IGN, signal.SIG_IGN)
    assert handlers[1] == (signal.SIG_IGN, signal.SIG_IGN)
