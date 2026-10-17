"""Checks against the real tools on this machine. Opt in with: python3 -I -m pytest -m machine

These read real tool output and hash the real model files, so they take tens of seconds.
They never build, download or write outside a temporary directory.
"""

import io
import shutil
from pathlib import Path

import pytest

from qlm import cli, doctor, pins
from qlm.paths import Paths
from qlm.system import Clock, Ctx, Env, Http, Runner

pytestmark = pytest.mark.machine


def real_ctx(home, repo):
    return Ctx(Paths(home=home, repo=repo), Runner(), Http(), Clock(), Env(), io.StringIO(), io.StringIO())


def test_doctor_reports_all_twelve_checks_on_this_machine(repo):
    ctx = real_ctx(Path.home(), repo)
    doctor.run(ctx)
    lines = ctx.out.getvalue().splitlines()
    assert len(lines) == 12
    assert all(l.split()[0] in ("PASS", "WARN", "FAIL") for l in lines)
    assert lines[0].startswith("PASS gpu:")


def test_fetch_verifies_the_files_already_on_disk_without_downloading(repo, tmp_path):
    real_models = Path.home() / ".local/share/qlm/models"
    if not all((real_models / f.local_name).exists() for f in pins.files()):
        pytest.skip("model files not downloaded on this machine")
    home = tmp_path / "home"
    models = home / ".local/share/qlm/models"
    models.mkdir(parents=True)
    for f in pins.files():
        (models / f.local_name).symlink_to(real_models / f.local_name)

    class NoHttp:
        def get(self, *a, **k):
            raise AssertionError("fetch must not download files that are already present")

    ctx = real_ctx(home, repo)
    ctx.http = NoHttp()
    assert cli.main(["fetch"], ctx) == 0, ctx.err.getvalue()
    assert ctx.out.getvalue().count("already present and verified") == 2
