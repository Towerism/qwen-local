import os

from qlm import cli, doctor
from tests.fakes import FakeEnv, FakeRunner, fail, make_ctx, ok

GPU = ("nvidia-smi", "--query-gpu=name,compute_cap,memory.total,memory.used,driver_version",
       "--format=csv,noheader,nounits")
APPS = ("nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits")
OLLAMA_HEADER = "NAME    ID    SIZE    PROCESSOR    CONTEXT    RUNNER    UNTIL\n"


def healthy_runner(overrides=None):
    script = {
        GPU: ok("NVIDIA GeForce RTX 4090, 8.9, 24564, 2563, 615.78.08\n"),
        APPS: ok("1001, /usr/bin/compositor, 331\n"),
        ("/opt/cuda/bin/nvcc", "--version"): ok("Cuda compilation tools, release 13.4, V13.4.92\n"),
        ("ollama", "ps"): ok(OLLAMA_HEADER),
        ("systemctl", "--user", "show", "-p", "MainPID", "--value", "qlm-server"): ok("0\n"),
    }
    script.update(overrides or {})
    which = {
        "nvidia-smi": "/usr/bin/nvidia-smi", "nvcc": "/opt/cuda/bin/nvcc", "g++": "/usr/bin/g++",
        "g++-15": "/usr/bin/g++-15", "git": "/usr/bin/git", "cmake": "/usr/bin/cmake", "ollama": "/usr/bin/ollama",
    }
    return FakeRunner(script, which)


def run_doctor(home, repo, runner=None, env=None):
    ctx = make_ctx(home, repo, runner=runner or healthy_runner(), env=env)
    code = cli.main(["doctor"], ctx)
    lines = ctx.out.getvalue().splitlines()
    return code, lines, ctx


def line_for(lines, name):
    matches = [l for l in lines if f" {name}:" in l]
    assert len(matches) == 1, (name, lines)
    return matches[0]


def test_healthy_machine_passes_every_check(home, repo):
    code, lines, _ = run_doctor(home, repo)
    assert code == 0
    assert len(lines) == 12
    assert all(l.startswith("PASS ") for l in lines), lines


def test_dashed_con_provider_key_fails(home, repo):
    conf = home / ".config/con/con.conf"
    conf.parent.mkdir(parents=True)
    conf.write_text('con.agent.provider = "anthropic"\n  con.agent.providers.openai-compatible.base_url = "x"\n')

    code, lines, _ = run_doctor(home, repo)

    assert code == 1
    assert line_for(lines, "con config").startswith("FAIL ")


def test_low_vram_warns_and_names_each_gpu_process(home, repo):
    runner = healthy_runner({
        GPU: ok("NVIDIA GeForce RTX 4090, 8.9, 24564, 5000, 615.78.08\n"),
        APPS: ok("1001, /usr/bin/compositor, 331\n2002, /usr/lib/browser --flag=a,b, 2100\n"),
    })
    code, lines, _ = run_doctor(home, repo, runner=runner)

    vram = line_for(lines, "vram")
    assert code == 0
    assert vram.startswith("WARN ")
    assert "/usr/bin/compositor (331 MiB)" in vram
    assert "/usr/lib/browser --flag=a,b (2100 MiB)" in vram


def test_doctor_changes_no_files(home, repo):
    (home / "existing").write_text("x")
    before = {p: p.stat().st_mtime_ns for p in home.rglob("*")}

    run_doctor(home, repo)

    after = {p: p.stat().st_mtime_ns for p in home.rglob("*")}
    assert after == before


def test_doctor_runs_only_read_only_commands(home, repo):
    _, _, ctx = run_doctor(home, repo)
    for call in ctx.runner.calls:
        assert call["argv"][:2] != ["systemctl", "--user"] or call["argv"][2] in ("show", "is-active"), call


def test_port_held_by_the_qlm_server_passes(home, repo):
    runner = healthy_runner({("systemctl", "--user", "show", "-p", "MainPID", "--value", "qlm-server"): ok("4242\n")})
    env = FakeEnv(listener=4242)
    env.path = [str(home / ".local/bin")]
    _, lines, _ = run_doctor(home, repo, runner=runner, env=env)
    assert line_for(lines, "port 9931").startswith("PASS ")


def test_port_held_by_another_process_fails_and_names_it(home, repo):
    env = FakeEnv(listener=777)
    env.path = [str(home / ".local/bin")]
    code, lines, _ = run_doctor(home, repo, env=env)
    port = line_for(lines, "port 9931")
    assert code == 1
    assert port.startswith("FAIL ") and "777" in port


def test_missing_nvidia_smi_fails_gpu_checks_but_others_still_run(home, repo):
    runner = healthy_runner()
    del runner.which_map["nvidia-smi"]
    code, lines, _ = run_doctor(home, repo, runner=runner)
    assert code == 1
    assert len(lines) == 12
    for name in ("gpu", "compute capability", "vram"):
        assert line_for(lines, name).startswith("FAIL "), name
    assert line_for(lines, "git and cmake").startswith("PASS ")


def test_missing_nvcc_git_or_cmake_fails(home, repo):
    runner = healthy_runner()
    del runner.which_map["nvcc"]
    del runner.which_map["cmake"]
    _, lines, _ = run_doctor(home, repo, runner=runner)
    assert line_for(lines, "nvcc").startswith("FAIL ")
    assert line_for(lines, "git and cmake").startswith("FAIL ") and "cmake" in line_for(lines, "git and cmake")


def test_compiler_without_fallback_warns(home, repo):
    runner = healthy_runner()
    del runner.which_map["g++-15"]
    _, lines, _ = run_doctor(home, repo, runner=runner)
    assert line_for(lines, "host compiler").startswith("WARN ")


def test_ollama_model_on_gpu_warns_and_names_it(home, repo):
    runner = healthy_runner({("ollama", "ps"): ok(OLLAMA_HEADER + "qwen3:8b  abc  6.1 GB  100% GPU  4096  x  4 minutes from now\n")})
    code, lines, _ = run_doctor(home, repo, runner=runner)
    assert code == 0
    assert line_for(lines, "ollama").startswith("WARN ") and "qwen3:8b" in line_for(lines, "ollama")


def test_ollama_not_installed_or_daemon_down_passes(home, repo):
    runner = healthy_runner()
    del runner.which_map["ollama"]
    _, lines, _ = run_doctor(home, repo, runner=runner)
    assert "not installed" in line_for(lines, "ollama")

    runner = healthy_runner({("ollama", "ps"): fail(stderr="could not connect to ollama")})
    _, lines, _ = run_doctor(home, repo, runner=runner)
    assert line_for(lines, "ollama").startswith("PASS ") and "daemon not running" in line_for(lines, "ollama")


def test_local_bin_missing_from_path_warns_with_the_line_to_add(home, repo):
    env = FakeEnv(path=["/usr/bin"])
    _, lines, _ = run_doctor(home, repo, env=env)
    path_line = line_for(lines, "~/.local/bin on PATH")
    assert path_line.startswith("WARN ")
    assert 'export PATH="$HOME/.local/bin:$PATH"' in path_line


def test_disk_space_counts_files_already_present(home, repo, monkeypatch):
    from tests.fakes import small_pins

    files, contents, _ = small_pins(monkeypatch)
    models = home / ".local/share/qlm/models"
    models.mkdir(parents=True)
    (models / files[0].local_name).write_bytes(contents[files[0].local_name])
    (models / (files[1].local_name + ".part")).write_bytes(b"x" * 100)
    needed = files[1].size - 100 + 6 * 2**30

    env = FakeEnv(path=[str(home / ".local/bin")], disk_free_bytes=needed)
    _, lines, _ = run_doctor(home, repo, env=env)
    assert line_for(lines, "disk space").startswith("PASS ")

    env = FakeEnv(path=[str(home / ".local/bin")], disk_free_bytes=needed - 1)
    code, lines, _ = run_doctor(home, repo, env=env)
    assert code == 1 and line_for(lines, "disk space").startswith("FAIL ")


def test_low_ram_warns(home, repo):
    env = FakeEnv(path=[str(home / ".local/bin")], total_ram_bytes=16 * 2**30)
    _, lines, _ = run_doctor(home, repo, env=env)
    assert line_for(lines, "ram").startswith("WARN ")


def test_other_gpu_warns_and_wrong_compute_capability_fails(home, repo):
    runner = healthy_runner({GPU: ok("NVIDIA GeForce RTX 3090, 8.6, 24576, 1000, 615.78.08\n")})
    code, lines, _ = run_doctor(home, repo, runner=runner)
    assert line_for(lines, "gpu").startswith("WARN ")
    assert line_for(lines, "compute capability").startswith("FAIL ")
    assert code == 1


def test_vram_passes_when_the_server_is_already_running(home, repo):
    runner = healthy_runner({
        ("systemctl", "--user", "show", "-p", "MainPID", "--value", "qlm-server"): ok("4242\n"),
        GPU: ok("NVIDIA GeForce RTX 4090, 8.9, 24564, 23000, 615.78.08\n"),
    })
    env = FakeEnv(path=[str(home / ".local/bin")], listener=4242)
    _, lines, _ = run_doctor(home, repo, runner=runner, env=env)
    assert line_for(lines, "vram").startswith("PASS ") and "server running" in line_for(lines, "vram")


def test_unparseable_nvidia_smi_output_fails_gpu_checks_without_crashing(home, repo):
    runner = healthy_runner({GPU: ok("NVIDIA GeForce RTX 4090, 8.9, 24564, [N/A], 615.78.08\n")})
    code, lines, _ = run_doctor(home, repo, runner=runner)
    assert code == 1
    assert len(lines) == 12
    assert "unparseable" in line_for(lines, "gpu")
    assert line_for(lines, "vram").startswith("FAIL ")


def test_path_entry_with_trailing_slash_counts(home, repo):
    env = FakeEnv(path=[str(home / ".local/bin") + "/"])
    _, lines, _ = run_doctor(home, repo, env=env)
    assert line_for(lines, "~/.local/bin on PATH").startswith("PASS ")


def test_doctor_commands_have_a_timeout(home, repo):
    _, _, ctx = run_doctor(home, repo)
    assert ctx.runner.calls and all(c["timeout"] == 10 for c in ctx.runner.calls)
