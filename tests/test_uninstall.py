"""qlm uninstall: remove what qlm install put in place, ask first, and keep logs, backups and (by default) models."""

import os
import shutil

import pytest

from qlm import cli
from tests.fakes import FakeEnv, FakeRunner, fail, make_ctx, ok
from tests.test_service import RESET, RUNNING, SHOW, STOP, install, unit

RELOAD = ("systemctl", "--user", "daemon-reload")
CON_LINES = (
    'con.agent.provider = "openaicompatible"\n'
    'con.agent.providers.openaicompatible.base_url = "http://127.0.0.1:9931/v1"\n'
)


def full_install(home, repo):
    """Everything qlm install, fetch, smoke and con leave behind."""
    p = install(home, repo)
    shutil.rmtree(p.llama_current)
    (p.tag_dir("b11669") / "bin").mkdir(parents=True)
    (p.tag_dir("b11669") / "bin/llama-server").write_bytes(b"server")
    p.llama_current.symlink_to(p.tag_dir("b11669"))
    (p.llama_src / ".git").mkdir(parents=True)
    (p.build_dir("b11669") / "bin").mkdir(parents=True)
    p.local_bin.mkdir(parents=True)
    p.qlm_link.symlink_to(p.entry_point)
    (p.state_dir / "backups").mkdir(parents=True)
    (p.state_dir / "backups/con.conf.20261017T120000Z.bak").write_text("old con.conf\n")
    return p


def uninstall(home, repo, *args, state=None, run=None, env=None):
    run = run or FakeRunner({SHOW: state or unit(), STOP: ok(), RESET: ok(), RELOAD: ok()})
    ctx = make_ctx(home, repo, runner=run, env=env or FakeEnv())
    code = cli.main(["uninstall", *args], ctx)
    return code, ctx, run


def out(ctx):
    return ctx.out.getvalue() + ctx.err.getvalue()


def test_a_full_uninstall_removes_the_install_and_keeps_state_and_models(home, repo, tmp_path):
    p = full_install(home, repo)
    (p.llama_opt / "notes.txt").write_text("mine")
    outside = tmp_path / "cache-elsewhere"
    (outside / "llama.cpp").mkdir(parents=True)
    shutil.rmtree(p.home / ".cache/qlm")
    (p.home / ".cache/qlm").symlink_to(outside)

    code, ctx, run = uninstall(home, repo, "--yes", state=RUNNING)

    assert code == 0, out(ctx)
    assert run.ran(*STOP)
    for gone in (p.unit_file, p.llama_current, p.tag_dir("b11669"), p.config_dir, p.qlm_link, p.home / ".cache/qlm"):
        assert not os.path.lexists(gone), gone
    assert (p.llama_opt / "notes.txt").read_text() == "mine"
    assert (outside / "llama.cpp").is_dir()
    assert (p.state_dir / "backups/con.conf.20261017T120000Z.bak").exists()
    assert (p.models / "Qwen3.8-27B-UD-Q4_K_M.gguf").exists()
    assert run.ran(*RELOAD)


def test_the_list_names_what_goes_and_what_stays(home, repo):
    p = full_install(home, repo)
    code, ctx, _ = uninstall(home, repo, "--yes")
    text = ctx.out.getvalue()
    remove, keep = text.split("Will keep:", 1)
    assert str(p.unit_file) in remove and str(p.config_dir) in remove and str(p.tag_dir("b11669")) in remove
    assert str(p.state_dir) in keep and str(p.models) in keep and "con-suggest" in keep and "API key" in keep


def test_an_empty_build_directory_is_removed_too(home, repo):
    p = full_install(home, repo)
    uninstall(home, repo, "--yes")
    assert not p.llama_opt.exists()


def test_a_qlm_command_that_is_not_our_link_is_kept(home, repo):
    p = full_install(home, repo)
    p.qlm_link.unlink()
    p.qlm_link.write_text("#!/bin/sh\n")
    code, ctx, _ = uninstall(home, repo, "--yes")
    assert code == 0
    assert p.qlm_link.read_text() == "#!/bin/sh\n"
    assert "isn't a link to this repo's bin/qlm" in ctx.out.getvalue()


def test_models_go_only_with_the_models_option(home, repo):
    p = full_install(home, repo)
    code, ctx, _ = uninstall(home, repo, "--models", "--yes")
    assert code == 0
    assert not p.models.exists()
    assert not (p.home / ".local/share/qlm").exists()


def test_the_share_directory_stays_when_something_else_is_in_it(home, repo):
    p = full_install(home, repo)
    (p.home / ".local/share/qlm/notes.txt").write_text("mine")
    uninstall(home, repo, "--models", "--yes")
    assert not p.models.exists()
    assert (p.home / ".local/share/qlm/notes.txt").exists()


def test_nothing_installed_says_so_without_asking(home, repo):
    env = FakeEnv(stdin_tty=True, answers=["yes"])
    code, ctx, _ = uninstall(home, repo, env=env)
    assert code == 0
    assert ctx.out.getvalue().strip().endswith("Nothing to remove.")
    assert env.prompts == []


def test_any_answer_but_yes_cancels(home, repo):
    p = full_install(home, repo)
    env = FakeEnv(stdin_tty=True, answers=["y"])
    code, ctx, run = uninstall(home, repo, env=env)
    assert code == 1
    assert "uninstall cancelled; nothing was removed" in ctx.err.getvalue()
    assert env.prompts == ["Type yes to remove these:"]
    assert p.unit_file.exists() and p.config_dir.exists() and p.llama_current.exists()
    assert run.ran(*STOP) == []


def test_yes_at_the_prompt_removes(home, repo):
    p = full_install(home, repo)
    code, ctx, _ = uninstall(home, repo, env=FakeEnv(stdin_tty=True, answers=["yes"]))
    assert code == 0, out(ctx)
    assert not p.unit_file.exists()


def test_without_a_terminal_it_needs_the_yes_option(home, repo):
    p = full_install(home, repo)
    code, ctx, _ = uninstall(home, repo, env=FakeEnv(stdin_tty=False))
    assert code == 2
    assert "--yes" in ctx.err.getvalue()
    assert p.unit_file.exists()


def test_a_failed_stop_removes_nothing(home, repo):
    p = full_install(home, repo)
    run = FakeRunner({SHOW: RUNNING, STOP: fail(stderr="Access denied"), RESET: ok(), RELOAD: ok()})
    code, ctx, _ = uninstall(home, repo, "--yes", run=run)
    assert code == 1
    assert p.unit_file.exists() and p.llama_current.exists() and p.config_dir.exists()


def test_a_failed_removal_lists_what_was_removed_and_what_remains(home, repo, monkeypatch):
    p = full_install(home, repo)
    real_rmtree = shutil.rmtree

    def refuse_cache(path, *a, **k):
        if str(path) == str(p.home / ".cache/qlm"):
            raise PermissionError(13, "Permission denied", str(path))
        return real_rmtree(path, *a, **k)

    monkeypatch.setattr(shutil, "rmtree", refuse_cache)
    code, ctx, _ = uninstall(home, repo, "--yes")
    assert code == 1
    err = ctx.err.getvalue()
    removed, remaining = err.split("Still there:", 1)
    assert str(p.unit_file) in removed and str(p.tag_dir("b11669")) in removed
    for path in (p.home / ".cache/qlm", p.qlm_link, p.config_dir):
        assert str(path) in remaining
    assert p.config_dir.exists()


def test_a_failed_daemon_reload_is_only_a_warning(home, repo):
    p = full_install(home, repo)
    run = FakeRunner({SHOW: unit(), STOP: ok(), RESET: ok(), RELOAD: fail(stderr="no bus")})
    code, ctx, _ = uninstall(home, repo, "--yes", run=run)
    assert code == 0
    assert "daemon-reload" in ctx.err.getvalue()
    assert not p.config_dir.exists()


def test_it_says_how_to_restore_con_and_leaves_con_conf_alone(home, repo):
    p = full_install(home, repo)
    p.con_conf.parent.mkdir(parents=True)
    p.con_conf.write_text(CON_LINES)
    before = p.con_conf.read_bytes()
    code, ctx, _ = uninstall(home, repo, "--yes")
    assert code == 0
    assert "Con is still set to use the local server" in ctx.out.getvalue()
    assert f"{p.entry_point} con --revert" in ctx.out.getvalue()
    assert p.con_conf.read_bytes() == before


def test_no_con_note_when_con_uses_another_provider(home, repo):
    p = full_install(home, repo)
    p.con_conf.parent.mkdir(parents=True)
    p.con_conf.write_text('con.agent.provider = "anthropic"\n')
    code, ctx, _ = uninstall(home, repo, "--yes")
    assert "Con is still set to use the local server" not in ctx.out.getvalue()


def test_a_current_pointing_outside_is_removed_as_a_link_only(home, repo, tmp_path):
    p = full_install(home, repo)
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "bin").mkdir(parents=True)
    p.llama_current.unlink()
    p.llama_current.symlink_to(elsewhere)
    code, ctx, _ = uninstall(home, repo, "--yes")
    assert code == 0
    assert not os.path.lexists(p.llama_current)
    assert (elsewhere / "bin").is_dir()


def test_an_unreachable_user_manager_removes_nothing(home, repo):
    p = full_install(home, repo)
    run = FakeRunner({SHOW: fail(stderr="Failed to connect to bus")})
    code, ctx, _ = uninstall(home, repo, "--yes", run=run)
    assert code == 1
    assert p.unit_file.exists()


def test_ctrl_c_at_the_prompt_cancels_cleanly(home, repo):
    p = full_install(home, repo)

    class CtrlC(FakeEnv):
        def ask(self, prompt):
            raise KeyboardInterrupt

    code, ctx, _ = uninstall(home, repo, env=CtrlC(stdin_tty=True))
    assert code == 1
    assert "uninstall cancelled; nothing was removed" in ctx.err.getvalue()
    assert p.unit_file.exists()
