import os
import pathlib
import runpy

import pytest

from qlm import cli
from qlm.paths import Paths
from tests.fakes import make_ctx

EXPECTED_FILES = [
    "bin/qlm",
    "qlm/__init__.py",
    "qlm/__main__.py",
    "qlm/cli.py",
    "qlm/paths.py",
    "qlm/pins.py",
    "qlm/profiles.py",
    "qlm/system.py",
    "qlm/doctor.py",
    "qlm/install.py",
    "qlm/fetch.py",
    "qlm/files.py",
    "qlm/con.py",
    "qlm/service.py",
    "qlm/smoke.py",
    "profiles/agent.env",
    "profiles/long.env",
    "profiles/vision.env",
    "systemd/qlm-server.service",
    "pyproject.toml",
]


def test_help_through_symlink_from_another_directory(tmp_path, repo, monkeypatch, capsys):
    link_dir = tmp_path / "bin"
    link_dir.mkdir()
    link = link_dir / "qlm"
    link.symlink_to(repo / "bin" / "qlm")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.setattr("sys.argv", [str(link), "--help"])

    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(link), run_name="__main__")

    assert exc.value.code == 0
    out = capsys.readouterr().out
    for name in ("doctor", "install", "fetch", "profile", "start", "stop", "status", "logs", "smoke", "con", "suggest"):
        assert name in out


def test_entry_point_has_isolated_python_shebang(repo):
    first = (repo / "bin" / "qlm").read_text().splitlines()[0]
    assert first == "#!/usr/bin/python3 -I"
    assert os.access(repo / "bin" / "qlm", os.X_OK)


def test_unknown_subcommand_is_a_usage_error(home, repo):
    ctx = make_ctx(home, repo)
    assert cli.main(["frobnicate"], ctx) == 2


def test_unknown_option_is_a_usage_error(home, repo):
    ctx = make_ctx(home, repo)
    assert cli.main(["fetch", "--nope"], ctx) == 2


def test_install_rejects_other_tags_before_running_anything(home, repo):
    ctx = make_ctx(home, repo)
    assert cli.main(["install", "--tag", "b11670"], ctx) == 2
    assert ctx.runner.calls == []
    assert "qlm update" in ctx.err.getvalue()


def test_repo_has_the_specified_layout(repo):
    missing = [p for p in EXPECTED_FILES if not (repo / p).exists()]
    assert missing == []


def test_every_path_lives_under_the_given_home(home, repo):
    paths = Paths(home=home, repo=repo)
    for name in (
        "llama_src", "llama_opt", "llama_current", "models", "config_dir", "active_env",
        "api_key", "unit_file", "state_dir", "logs", "backups", "bench", "local_bin", "qlm_link", "con_conf",
    ):
        value = getattr(paths, name)
        assert pathlib.Path(value).is_relative_to(home), name
    assert paths.build_dir("b11669").is_relative_to(home)
    assert paths.tag_dir("b11669").is_relative_to(home)
