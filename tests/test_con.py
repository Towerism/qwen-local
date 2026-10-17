"""qlm con and qlm suggest: editing Con's config file, with backups, and the opt-in Ollama suggestion model."""

import os
import stat

import pytest

from qlm import cli, con
from tests.fakes import FakeRunner, FakeEnv, fail, make_ctx, ok
from tests.test_service import install, unit

KEY = "k" * 43
PGREP = ("pgrep", "-x", "con")
SHOW = ("systemctl", "--user", "show")
OLLAMA_LIST = ("ollama", "list")
OLLAMA_SHOW = ("ollama", "show", "con-suggest", "--modelfile")
OLLAMA_PULL = ("ollama", "pull", "qwen2.5-coder:1.5b")
OLLAMA_CREATE = ("ollama", "create", "con-suggest")
LIST_HEADER = "NAME    ID    SIZE    MODIFIED\n"

ORIGINAL = [
    "# my Con settings",
    "con.version = 1",
    'font-family = "JetBrains Mono"',
    'font-family = "Noto Color Emoji"',
    "con.skills.project_paths = .claude/skills",
    "con.skills.project_paths = .agents/skills",
    'con.agent.providers.anthropic.api_key = "sk-ant-secret-value"',
    "con.agent.suggestion_model.enabled = true",
    "con.agent.max_turns = 30",
    "con.agent.temperature = 0.3",
    "  con.agent.suggestion_model.enabled = true  ",
    "# con.agent.provider = \"commented\"",
    'con.agent.provider = "anthropic"',
    "con.agent.auto_approve_tools = false",
]

EXPECTED = {
    "con.agent.provider": "openaicompatible",
    "con.agent.provider_is_explicit": "true",
    "con.agent.providers.openaicompatible.base_url": "http://127.0.0.1:9931/v1",
    "con.agent.providers.openaicompatible.model": "qwen3.8-27b",
    "con.agent.providers.openaicompatible.api_key": KEY,
    "con.agent.suggestion_model.enabled": "false",
}


# Factories


def write_conf(home, lines, mode=0o600, newline="\n", final=True):
    conf = home / ".config/con/con.conf"
    conf.parent.mkdir(parents=True, exist_ok=True)
    text = newline.join(lines) + (newline if final else "")
    conf.write_bytes(text.encode())
    os.chmod(conf, mode)
    return conf


def runner(overrides=None):
    overrides = overrides or {}
    script = {
        PGREP: fail(code=1),
        SHOW: unit(),
        OLLAMA_LIST: ok(LIST_HEADER),
        OLLAMA_SHOW: ok("FROM qwen2.5-coder:1.5b\nPARAMETER num_gpu 0\n"),
        OLLAMA_PULL: ok(),
        OLLAMA_CREATE: ok(),
    }
    script.update({k: v for k, v in overrides.items() if not isinstance(k, str)})
    which = {"ollama": "/usr/bin/ollama"}
    which.update(overrides.get("which", {}))
    return FakeRunner(script, which=which)


def ctx_for(home, repo, run=None, xdg=None):
    if not (home / ".config/qlm/api-key").exists():
        install(home, repo, key=KEY)
    return make_ctx(home, repo, runner=run or runner(), env=FakeEnv(xdg=xdg))


def text_of(home):
    return (home / ".config/con/con.conf").read_text()


def backups(home):
    d = home / ".local/state/qlm/backups"
    return sorted(p.name for p in d.iterdir()) if d.exists() else []


def output(ctx):
    return ctx.out.getvalue() + ctx.err.getvalue()


# The text editor, as pure functions


def test_a_line_holds_a_key_only_outside_comments_and_with_exact_name():
    assert con.holds("  con.agent.provider = x ", "con.agent.provider")
    assert con.holds("con.agent.provider=x", "con.agent.provider")
    assert not con.holds('# con.agent.provider = "x"', "con.agent.provider")
    assert not con.holds("con.agent.provider_is_explicit = true", "con.agent.provider")
    assert not con.holds("con.agent.provider", "con.agent.provider")


def test_the_current_value_is_the_last_line_without_outer_quotes():
    lines = ['con.agent.provider = "anthropic"', "con.agent.provider = ollama ", 'x = "a"b"']
    assert con.current_value(lines, "con.agent.provider") == "ollama"
    assert con.current_value(['k = "v=w"'], "k") == "v=w"
    assert con.current_value(["k = \"\""], "k") == ""
    assert con.current_value(["other = 1"], "k") is None


def test_edit_replaces_the_first_line_drops_repeats_and_appends_missing_keys_in_order():
    text = "a = 1\nk = old\nb = 2\nk = older\n"
    new = con.edit(text, [("k", "new"), ("m", "1"), ("n", '"x"')], remove=("b",))
    assert new == "a = 1\nk = new\nm = 1\nn = \"x\"\n"


def test_edit_keeps_crlf_on_untouched_lines_and_adds_a_newline_before_appending():
    assert con.edit("a = 1\r\nk = old\r\nz = 9", [("k", "new"), ("m", "1")]) == "a = 1\r\nk = new\r\nz = 9\nm = 1\n"


def test_edit_leaves_comments_that_mention_a_key():
    text = '# con.agent.temperature = 1\ncon.agent.temperature = 0.3\n'
    assert con.edit(text, [], remove=("con.agent.temperature",)) == "# con.agent.temperature = 1\n"


# qlm con: usage and refusals


@pytest.mark.parametrize("argv", [["con"], ["con", "--print", "--apply"], ["suggest"], ["suggest", "--enable", "--disable"]])
def test_con_and_suggest_need_exactly_one_mode(home, repo, argv):
    ctx = ctx_for(home, repo)
    assert cli.main(argv, ctx) == 2


def test_a_different_xdg_config_home_is_refused(home, repo):
    write_conf(home, ORIGINAL)
    before = text_of(home)
    ctx = ctx_for(home, repo, xdg=str(home / "elsewhere"))
    assert cli.main(["con", "--apply"], ctx) == 1
    assert "XDG_CONFIG_HOME" in output(ctx)
    assert text_of(home) == before and backups(home) == []


def test_xdg_config_home_pointing_at_the_default_is_fine(home, repo):
    write_conf(home, ORIGINAL)
    ctx = ctx_for(home, repo, xdg=str(home / ".config") + "/")
    assert cli.main(["con", "--apply"], ctx) == 0


@pytest.mark.parametrize(
    "argv", [["con", "--apply"], ["con", "--revert"], ["suggest", "--disable"], ["suggest", "--enable"]]
)
def test_nothing_changes_while_con_runs(home, repo, argv):
    write_conf(home, ORIGINAL)
    before = text_of(home)
    run = runner({PGREP: ok("563280\n")})
    ctx = ctx_for(home, repo, run=run)
    assert cli.main(argv, ctx) == 1
    assert "563280" in output(ctx) and "quit Con" in output(ctx)
    assert text_of(home) == before and backups(home) == []
    assert run.ran("ollama") == []


def test_an_unclear_pgrep_result_also_refuses(home, repo):
    write_conf(home, ORIGINAL)
    ctx = ctx_for(home, repo, run=runner({PGREP: fail(code=2, stderr="pgrep: bad")}))
    assert cli.main(["con", "--apply"], ctx) == 1
    assert backups(home) == []


def test_dashed_provider_keys_are_refused_with_the_line_number(home, repo):
    lines = list(ORIGINAL)
    lines.insert(11, "con.agent.providers.openai-compatible.base_url = x")
    write_conf(home, lines)
    before = text_of(home)
    ctx = ctx_for(home, repo)
    assert cli.main(["con", "--apply"], ctx) == 1
    assert "line 12" in output(ctx) and "openaicompatible" in output(ctx)
    assert text_of(home) == before


# qlm con --print


def test_print_compares_without_changing_anything_or_showing_keys(home, repo):
    write_conf(home, ORIGINAL)
    before = text_of(home)
    run = runner({PGREP: ok("563280\n")})
    ctx = ctx_for(home, repo, run=run)
    assert cli.main(["con", "--print"], ctx) == 0
    shown = ctx.out.getvalue()
    states = {line.split()[0]: line.rsplit("(", 1)[-1].rstrip(")") for line in shown.splitlines() if line.startswith("con.")}
    assert states["con.agent.provider"] == "different"
    for key in list(EXPECTED)[1:5]:
        assert states[key] == "missing", key
    assert states["con.agent.suggestion_model.enabled"] == "different"
    assert states["con.agent.temperature"] == "will be removed"
    assert KEY not in shown and "sk-ant-secret-value" not in shown and "<hidden>" in shown
    assert text_of(home) == before and backups(home) == []


# qlm con --apply


def test_apply_edits_only_the_managed_lines(home, repo):
    write_conf(home, ORIGINAL)
    ctx = ctx_for(home, repo)
    assert cli.main(["con", "--apply"], ctx) == 0
    lines = text_of(home).splitlines()
    assert lines[:7] == ORIGINAL[:7]
    assert lines[7] == "con.agent.suggestion_model.enabled = false"
    assert lines[8] == "con.agent.max_turns = 30"
    assert lines[9] == '# con.agent.provider = "commented"'
    assert lines[10] == 'con.agent.provider = "openaicompatible"'
    assert lines[11] == "con.agent.auto_approve_tools = false"
    assert lines[12:] == [
        "con.agent.provider_is_explicit = true",
        'con.agent.providers.openaicompatible.base_url = "http://127.0.0.1:9931/v1"',
        'con.agent.providers.openaicompatible.model = "qwen3.8-27b"',
        f'con.agent.providers.openaicompatible.api_key = "{KEY}"',
    ]
    assert not any("temperature" in l for l in lines)


def test_apply_keeps_the_mode_and_backs_up_the_old_bytes_privately(home, repo):
    conf = write_conf(home, ORIGINAL, mode=0o640)
    old = conf.read_bytes()
    ctx = ctx_for(home, repo)
    assert cli.main(["con", "--apply"], ctx) == 0
    assert stat.S_IMODE(conf.stat().st_mode) == 0o640
    assert backups(home) == ["con.conf.20261017T120000Z.bak"]
    backup = home / ".local/state/qlm/backups/con.conf.20261017T120000Z.bak"
    assert backup.read_bytes() == old
    assert stat.S_IMODE(backup.stat().st_mode) == 0o600
    assert stat.S_IMODE(backup.parent.stat().st_mode) == 0o700
    assert "next" in output(ctx) and "manual" in output(ctx)
    assert KEY not in output(ctx)


def test_apply_mentions_starting_the_server_only_when_it_is_stopped(home, repo):
    write_conf(home, ORIGINAL)
    ctx = ctx_for(home, repo)
    cli.main(["con", "--apply"], ctx)
    assert "qlm start" in output(ctx)
    write_conf(home, ORIGINAL)
    ctx = ctx_for(home, repo, run=runner({SHOW: unit("active", "running", 4321)}))
    cli.main(["con", "--apply"], ctx)
    assert "qlm start" not in output(ctx)


def test_apply_twice_changes_nothing_the_second_time(home, repo):
    write_conf(home, ORIGINAL)
    cli.main(["con", "--apply"], ctx_for(home, repo))
    after = text_of(home)
    ctx = ctx_for(home, repo)
    assert cli.main(["con", "--apply"], ctx) == 0
    assert "already configured" in output(ctx)
    assert text_of(home) == after and len(backups(home)) == 1


def test_apply_without_a_config_file_starts_one_like_con_does(home, repo):
    ctx = ctx_for(home, repo)
    assert cli.main(["con", "--apply"], ctx) == 0
    conf = home / ".config/con/con.conf"
    lines = conf.read_text().splitlines()
    assert lines[:2] == ["# Con configuration", "con.version = 1"]
    assert {con.current_value(lines, k) for k in EXPECTED} == set(EXPECTED.values())
    assert stat.S_IMODE(conf.stat().st_mode) == 0o600
    assert backups(home) == ["con.conf.20261017T120000Z.absent"]


@pytest.mark.parametrize("legacy", ["config.toml", "config.ghostty"])
def test_apply_waits_for_con_to_migrate_an_older_config(home, repo, legacy):
    (home / ".config/con").mkdir(parents=True)
    (home / ".config/con" / legacy).write_text("x = 1\n")
    ctx = ctx_for(home, repo)
    assert cli.main(["con", "--apply"], ctx) == 1
    assert "Start Con once" in output(ctx)
    assert not (home / ".config/con/con.conf").exists() and backups(home) == []


def test_apply_leaves_suggestions_on_when_they_use_ollama(home, repo):
    lines = ORIGINAL[:7] + [
        'con.agent.suggestion_model.provider = "ollama"',
        "con.agent.suggestion_model.model = con-suggest",
        "con.agent.suggestion_model.enabled = true",
    ]
    write_conf(home, lines)
    ctx = ctx_for(home, repo)
    assert cli.main(["con", "--apply"], ctx) == 0
    assert con.current_value(text_of(home).splitlines(), "con.agent.suggestion_model.enabled") == "true"


def test_apply_writes_through_a_symlink_inside_home(home, repo):
    target = home / "dotfiles/con.conf"
    target.parent.mkdir()
    target.write_text("\n".join(ORIGINAL) + "\n")
    link = home / ".config/con/con.conf"
    link.parent.mkdir(parents=True)
    link.symlink_to(target)
    ctx = ctx_for(home, repo)
    assert cli.main(["con", "--apply"], ctx) == 0
    assert link.is_symlink() and os.readlink(link) == str(target)
    assert 'con.agent.provider = "openaicompatible"' in target.read_text()


def test_apply_refuses_a_symlink_that_leaves_home(home, repo, tmp_path):
    outside = tmp_path / "outside.conf"
    outside.write_text("\n".join(ORIGINAL) + "\n")
    link = home / ".config/con/con.conf"
    link.parent.mkdir(parents=True)
    link.symlink_to(outside)
    ctx = ctx_for(home, repo)
    assert cli.main(["con", "--apply"], ctx) == 1
    assert outside.read_text() == "\n".join(ORIGINAL) + "\n"


def test_a_write_that_does_not_verify_is_rolled_back(home, repo, monkeypatch):
    conf = write_conf(home, ORIGINAL)
    old = conf.read_bytes()
    real_write = con._write

    def drop_last_line(path, data, mode):
        real_write(path, data.rsplit(b"\n", 2)[0] + b"\n", mode)

    monkeypatch.setattr(con, "_write", drop_last_line)
    ctx = ctx_for(home, repo)
    assert cli.main(["con", "--apply"], ctx) == 1
    assert conf.read_bytes() == old
    assert KEY not in output(ctx)


def test_a_new_file_that_does_not_verify_is_removed_again(home, repo, monkeypatch):
    real_write = con._write
    monkeypatch.setattr(con, "_write", lambda path, data, mode: real_write(path, b"con.version = 1\n", mode))
    ctx = ctx_for(home, repo)
    assert cli.main(["con", "--apply"], ctx) == 1
    assert not (home / ".config/con/con.conf").exists()


def test_a_failed_write_leaves_the_file_and_no_temporary_file(home, repo, monkeypatch):
    conf = write_conf(home, ORIGINAL)
    old = conf.read_bytes()

    def full_disk(fd):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "fsync", full_disk)
    ctx = ctx_for(home, repo)
    assert cli.main(["con", "--apply"], ctx) == 1
    monkeypatch.undo()
    assert conf.read_bytes() == old
    assert sorted(p.name for p in conf.parent.iterdir()) == ["con.conf"]
    assert "No space left on device" in output(ctx)


def test_apply_needs_the_api_key(home, repo):
    write_conf(home, ORIGINAL)
    ctx = ctx_for(home, repo)
    (home / ".config/qlm/api-key").unlink()
    assert cli.main(["con", "--apply"], ctx) == 1
    assert "qlm install" in output(ctx)


# qlm con --revert


def make_backup(home, name, data):
    d = home / ".local/state/qlm/backups"
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_bytes(data)


def test_revert_restores_the_newest_backup_and_keeps_the_others(home, repo):
    write_conf(home, ["current = 1"])
    make_backup(home, "con.conf.20261017T100000Z.bak", b"ten = 1\n")
    make_backup(home, "con.conf.20261017T110000Z.bak", b"eleven = 1\n")
    make_backup(home, "con.conf.20261017T110000Z-1.bak", b"eleven later = 1\n")
    make_backup(home, "con.conf.20261017T115900Z.pre-revert", b"never = 1\n")
    ctx = ctx_for(home, repo)
    assert cli.main(["con", "--revert"], ctx) == 0
    assert text_of(home) == "eleven later = 1\n"
    names = backups(home)
    assert "con.conf.20261017T100000Z.bak" in names and "con.conf.20261017T110000Z-1.bak" in names
    pre = home / ".local/state/qlm/backups/con.conf.20261017T120000Z.pre-revert"
    assert pre.read_text() == "current = 1\n"


def test_revert_to_an_absent_marker_deletes_the_file(home, repo):
    write_conf(home, ["current = 1"])
    make_backup(home, "con.conf.20261017T100000Z.bak", b"ten = 1\n")
    make_backup(home, "con.conf.20261017T110000Z.absent", b"")
    ctx = ctx_for(home, repo)
    assert cli.main(["con", "--revert"], ctx) == 0
    assert not (home / ".config/con/con.conf").exists()


def test_revert_without_backups_fails(home, repo):
    write_conf(home, ["current = 1"])
    ctx = ctx_for(home, repo)
    assert cli.main(["con", "--revert"], ctx) == 1
    assert text_of(home) == "current = 1\n"


def test_apply_then_revert_gives_back_the_original_bytes(home, repo):
    conf = write_conf(home, ORIGINAL, newline="\r\n", final=False)
    old = conf.read_bytes()
    cli.main(["con", "--apply"], ctx_for(home, repo))
    assert conf.read_bytes() != old
    assert cli.main(["con", "--revert"], ctx_for(home, repo)) == 0
    assert conf.read_bytes() == old


# qlm suggest


def test_enable_pulls_creates_and_points_con_at_the_cpu_model(home, repo, monkeypatch):
    write_conf(home, ORIGINAL)
    seen = {}

    def create(argv):
        seen["modelfile"] = open(argv[argv.index("-f") + 1]).read()
        return ok()

    run = runner({OLLAMA_CREATE: create})
    ctx = ctx_for(home, repo, run=run)
    assert cli.main(["suggest", "--enable"], ctx) == 0
    assert run.passthrough_calls == [list(OLLAMA_PULL)]
    assert seen["modelfile"].splitlines() == ["FROM qwen2.5-coder:1.5b", "PARAMETER num_gpu 0"]
    lines = text_of(home).splitlines()
    assert con.current_value(lines, "con.agent.suggestion_model.enabled") == "true"
    assert con.current_value(lines, "con.agent.suggestion_model.provider") == "ollama"
    assert con.current_value(lines, "con.agent.suggestion_model.model") == "con-suggest"
    assert len(backups(home)) == 1


def test_enable_skips_the_pull_when_the_base_model_is_there(home, repo):
    write_conf(home, ORIGINAL)
    run = runner({OLLAMA_LIST: ok(LIST_HEADER + "qwen2.5-coder:1.5b  6d3abb8d2d53  986 MB  2 days ago\n")})
    ctx = ctx_for(home, repo, run=run)
    assert cli.main(["suggest", "--enable"], ctx) == 0
    assert run.passthrough_calls == [] and len(run.ran(*OLLAMA_CREATE)) == 1


def test_enable_reuses_an_existing_cpu_only_model(home, repo):
    write_conf(home, ORIGINAL)
    run = runner({OLLAMA_LIST: ok(LIST_HEADER + "con-suggest:latest  1a2b3c4d5e6f  986 MB  1 day ago\n")})
    ctx = ctx_for(home, repo, run=run)
    assert cli.main(["suggest", "--enable"], ctx) == 0
    assert run.passthrough_calls == [] and run.ran(*OLLAMA_CREATE) == []


def test_enable_refuses_an_existing_model_that_could_use_the_gpu(home, repo):
    write_conf(home, ORIGINAL)
    before = text_of(home)
    run = runner({
        OLLAMA_LIST: ok(LIST_HEADER + "con-suggest:latest  1a2b3c4d5e6f  986 MB  1 day ago\n"),
        OLLAMA_SHOW: ok("FROM qwen2.5-coder:1.5b\n"),
    })
    ctx = ctx_for(home, repo, run=run)
    assert cli.main(["suggest", "--enable"], ctx) == 1
    assert "ollama rm con-suggest" in output(ctx)
    assert text_of(home) == before


@pytest.mark.parametrize("ollama", [{OLLAMA_LIST: fail(stderr="could not connect")}, {"which": {"ollama": None}}])
def test_enable_needs_a_running_ollama(home, repo, ollama):
    write_conf(home, ORIGINAL)
    before = text_of(home)
    ctx = ctx_for(home, repo, run=runner(ollama))
    assert cli.main(["suggest", "--enable"], ctx) == 1
    assert "Ollama" in output(ctx)
    assert text_of(home) == before


def test_a_failed_pull_changes_nothing(home, repo):
    write_conf(home, ORIGINAL)
    before = text_of(home)
    run = runner({OLLAMA_PULL: fail(code=1)})
    ctx = ctx_for(home, repo, run=run)
    assert cli.main(["suggest", "--enable"], ctx) == 1
    assert run.ran(*OLLAMA_CREATE) == []
    assert text_of(home) == before and backups(home) == []


def test_a_failed_create_changes_nothing(home, repo):
    write_conf(home, ORIGINAL)
    before = text_of(home)
    ctx = ctx_for(home, repo, run=runner({OLLAMA_CREATE: fail(stderr="model too large")}))
    assert cli.main(["suggest", "--enable"], ctx) == 1
    assert "model too large" in output(ctx)
    assert text_of(home) == before and backups(home) == []


def test_disabling_twice_says_it_is_already_off(home, repo):
    write_conf(home, ORIGINAL)
    cli.main(["suggest", "--disable"], ctx_for(home, repo))
    ctx = ctx_for(home, repo)
    assert cli.main(["suggest", "--disable"], ctx) == 0
    assert "already off" in output(ctx) and len(backups(home)) == 1


def test_enable_checks_the_config_before_downloading(home, repo):
    write_conf(home, ORIGINAL + ["con.agent.providers.openai-compatible.model = x"])
    run = runner()
    ctx = ctx_for(home, repo, run=run)
    assert cli.main(["suggest", "--enable"], ctx) == 1
    assert run.ran("ollama") == []


def test_disable_turns_suggestions_off_and_leaves_the_rest(home, repo):
    lines = ORIGINAL[:7] + [
        'con.agent.suggestion_model.provider = "ollama"',
        'con.agent.suggestion_model.model = "con-suggest"',
        "con.agent.suggestion_model.enabled = true",
    ]
    write_conf(home, lines)
    run = runner()
    ctx = ctx_for(home, repo, run=run)
    assert cli.main(["suggest", "--disable"], ctx) == 0
    new = text_of(home).splitlines()
    assert new[7:9] == lines[7:9]
    assert con.current_value(new, "con.agent.suggestion_model.enabled") == "false"
    assert run.ran("ollama") == []
