"""Checks against the real tools on this machine. Opt in with: python3 -I -m pytest -m machine

These read real tool output and hash the real model files, so they take tens of seconds.
They never build or download. The server test starts and stops the real qlm-server
(about 20 GiB of VRAM for a few minutes) and leaves a smoke log in ~/.local/state/qlm/smoke/;
it skips when the server is already running, so it never stops one you started. The bench test
does the same and leaves a result in ~/.local/state/qlm/bench/. The per-model smoke test selects,
starts and smokes each (model, profile) pair in turn, then puts back the selection and active.env;
it fails when the server is already running, so stop it first.
"""

import io
import json
from pathlib import Path

import pytest

from qlm import cli, doctor, pins
from qlm.paths import Paths
from qlm.system import Clock, Ctx, Env, Http, Runner
from tests.fakes import select_model

pytestmark = pytest.mark.machine


def real_ctx(home, repo):
    return Ctx(Paths(home=home, repo=repo), Runner(), Http(), Clock(), Env(), io.StringIO(), io.StringIO())


def test_doctor_reports_all_eleven_checks_on_this_machine(repo):
    ctx = real_ctx(Path.home(), repo)
    doctor.run(ctx)
    lines = ctx.out.getvalue().splitlines()
    assert len(lines) == 11
    assert all(line.split()[0] in ("PASS", "WARN", "FAIL") for line in lines)
    assert lines[0].startswith("PASS gpu:")


def test_fetch_verifies_the_files_already_on_disk_without_downloading(repo, tmp_path):
    real_models = Path.home() / ".local/share/qlm/models"
    if not all((real_models / f.local_name).exists() for f in pins.files(pins.model("qwen3.8-27b"))):
        pytest.skip("model files not downloaded on this machine")
    home = tmp_path / "home"
    models = home / ".local/share/qlm/models"
    models.mkdir(parents=True)
    for f in pins.files(pins.model("qwen3.8-27b")):
        (models / f.local_name).symlink_to(real_models / f.local_name)
    select_model(home, "qwen3.8-27b")

    class NoHttp:
        def get(self, *a, **k):
            raise AssertionError("fetch must not download files that are already present")

    ctx = real_ctx(home, repo)
    ctx.http = NoHttp()
    assert cli.main(["fetch"], ctx) == 0, ctx.err.getvalue()
    assert ctx.out.getvalue().count("already present and verified") == 2


def test_the_real_server_starts_passes_smoke_and_stops(repo):
    from qlm import service

    ctx = real_ctx(Path.home(), repo)
    if not ctx.paths.unit_file.exists():
        pytest.skip("qlm install has not been run on this machine")
    if service.unit_state(ctx).running:
        pytest.skip("qlm-server is already running; this test only stops a server it started")
    try:
        assert cli.main(["start"], ctx) == 0, ctx.err.getvalue()
        assert "ready in" in ctx.out.getvalue()
        status = real_ctx(Path.home(), repo)
        assert cli.main(["status"], status) == 0
        assert status.out.getvalue().splitlines()[0] == "state: running"
        smoke = real_ctx(Path.home(), repo)
        assert cli.main(["smoke"], smoke) == 0, smoke.out.getvalue() + smoke.err.getvalue()
    finally:
        assert cli.main(["stop"], real_ctx(Path.home(), repo)) == 0


def test_a_quick_bench_measures_the_real_server_and_saves_a_result(repo):
    import json

    from qlm import service

    ctx = real_ctx(Path.home(), repo)
    if not ctx.paths.unit_file.exists():
        pytest.skip("qlm install has not been run on this machine")
    if service.unit_state(ctx).running:
        pytest.skip("qlm-server is already running; this test only stops a server it started")
    try:
        assert cli.main(["start"], ctx) == 0, ctx.err.getvalue()
        bench = real_ctx(Path.home(), repo)
        assert cli.main(["bench", "--quick"], bench) == 0, bench.out.getvalue() + bench.err.getvalue()
        result = Path(bench.out.getvalue().rstrip().splitlines()[-1].split("Results: ", 1)[1])
        data = json.loads(result.read_text())
        assert data["quick"] is True
        assert data["measures"]["mtp_on.pp_2048.tok_s"] > 0
        assert data["measures"]["mtp_on.gen_code.tok_s"] > 0
        assert data["measures"]["mtp_on.peak_vram_gib"] > 0
    finally:
        assert cli.main(["stop"], real_ctx(Path.home(), repo)) == 0


@pytest.mark.parametrize(
    "model_id, profile",
    [("qwen3.6-35b-a3b", "agent"), ("qwen3.6-35b-a3b", "long"), ("qwen3.8-27b", "agent")],
)
def test_selected_model_profile_passes_smoke(repo, model_id, profile):
    from qlm import service
    from qlm.system import SERVER

    ctx = real_ctx(Path.home(), repo)
    assert not service.unit_state(ctx).running, "stop qlm-server first: this test starts and stops it"
    p = ctx.paths
    saved_selection = p.selection.read_bytes() if p.selection.exists() else None
    saved_active = p.active_env.read_bytes()
    try:
        use = real_ctx(Path.home(), repo)
        assert cli.main(["use", model_id], use) == 0, use.err.getvalue()
        start = real_ctx(Path.home(), repo)
        assert cli.main(["start", profile], start) == 0, start.err.getvalue()
        smoke = real_ctx(Path.home(), repo)
        assert cli.main(["smoke"], smoke) == 0, smoke.out.getvalue() + smoke.err.getvalue()
        key = p.api_key.read_text().strip()
        reply = ctx.http.request("GET", f"{SERVER}/v1/models", headers={"Authorization": f"Bearer {key}"})
        assert reply.status == 200
        assert model_id in [m["id"] for m in json.loads(reply.body)["data"]]
    finally:
        assert cli.main(["stop"], real_ctx(Path.home(), repo)) == 0
        p.active_env.write_bytes(saved_active)
        if saved_selection is None:
            p.selection.unlink(missing_ok=True)
        else:
            p.selection.write_bytes(saved_selection)
