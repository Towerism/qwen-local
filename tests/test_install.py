import os
import stat

import pytest

from qlm import cli, install, pins, profiles
from tests.fakes import FakeRunner, fail, make_ctx, ok

TAG = pins.LLAMA_TAG
COMMIT = pins.LLAMA_COMMIT
CONFIGURE = ("cmake", "-S")
BUILD = ("cmake", "--build")


def help_text(repo, drop=()):
    names = [n for n in profiles.all_setting_names(repo / "profiles") if n not in drop and n != "LLAMA_ARG_SPEC_TYPE"]
    lines = [f"--opt-{i} VALUE              some option (env: {n})" for i, n in enumerate(names)]
    if "LLAMA_ARG_SPEC_TYPE" not in drop:
        lines += [
            "--spec-type TYPES            comma-separated speculative decoding types:",
            "                             none, draft-simple, draft-mtp, ngram-mod",
            "                             (env: LLAMA_ARG_SPEC_TYPE)",
        ]
    lines.append("--other X                    unrelated option mentioning draft-mtp-ish (env: LLAMA_ARG_OTHER)")
    return "\n".join(lines) + "\n"


class Machine:
    """A fake machine on which git, cmake and the built llama-server behave as scripted."""

    def __init__(self, home, repo, commit=COMMIT, configure=None, help_ok=True, binary=b"server-new", which=None):
        self.home, self.repo, self.binary = home, repo, binary
        ctx = make_ctx(home, repo)
        self.paths = ctx.paths
        src = str(self.paths.llama_src)
        build_dir = self.paths.build_dir(TAG)
        staged = self.paths.tag_dir(TAG) / "bin" / ".staged-llama-server"
        final = self.paths.tag_dir(TAG) / "bin" / "llama-server"
        good_help, bad_help = help_text(repo), help_text(repo, drop=("LLAMA_ARG_SPEC_TYPE",))

        def clone(argv):
            (self.paths.llama_src / ".git").mkdir(parents=True)
            return ok()

        def build(argv):
            (build_dir / "bin").mkdir(parents=True, exist_ok=True)
            (build_dir / "bin" / "llama-server").write_bytes(self.binary)
            (build_dir / "bin" / "llama-bench").write_bytes(b"bench")
            return ok("[100%] Built target llama-server\n")

        script = {
            ("git", "clone"): clone,
            ("git", "-C", src, "fetch"): ok(),
            ("git", "-C", src, "rev-parse"): ok(commit + "\n"),
            ("git", "-C", src, "checkout"): ok(),
            CONFIGURE: configure or ok("-- Configuring done\n"),
            BUILD: build,
            (str(staged), "--help"): ok(good_help if help_ok else bad_help),
            (str(final), "--help"): ok(good_help),
            ("systemctl", "--user", "daemon-reload"): ok(),
            ("systemctl", "--user", "show", "-p", "MainPID", "--value", "qlm-server"): ok("0\n"),
        }
        self.runner = FakeRunner(script, {"g++-15": "/usr/bin/g++-15"} if which is None else which)

    def install(self, *args):
        ctx = make_ctx(self.home, self.repo, runner=self.runner)
        code = cli.main(["install", *args], ctx)
        return code, ctx


def mode(path):
    return stat.S_IMODE(os.stat(path).st_mode)


def test_wrong_commit_stops_before_building(home, repo):
    m = Machine(home, repo, commit="deadbeef" * 5)
    code, ctx = m.install()
    assert code == 1
    assert COMMIT in ctx.err.getvalue() and "deadbeef" in ctx.err.getvalue()
    assert m.runner.ran("cmake") == []
    assert m.runner.ran("git", "-C", str(m.paths.llama_src), "checkout") == []


def test_successful_install_places_binaries_and_switches_current(home, repo):
    m = Machine(home, repo)
    code, ctx = m.install()
    assert code == 0, ctx.err.getvalue()
    bin_dir = m.paths.tag_dir(TAG) / "bin"
    assert (bin_dir / "llama-server").read_bytes() == b"server-new"
    assert (bin_dir / "llama-bench").exists()
    assert os.readlink(m.paths.llama_current) == str(m.paths.tag_dir(TAG))
    configure = m.runner.ran(*CONFIGURE)[0]
    for flag in ("-DGGML_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES=89", "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=OFF"):
        assert flag in configure
    assert m.runner.ran("git", "clone")[0][-2:] == [pins.LLAMA_REPO_URL, str(m.paths.llama_src)]
    assert "--no-recurse-submodules" in m.runner.ran("git", "clone")[0]
    assert m.runner.ran("git", "-C", str(m.paths.llama_src), "checkout")[0][-2:] == ["--detach", COMMIT]
    assert m.runner.ran(*BUILD)[0][-2:] == ["-j", "32"]


def test_existing_checkout_fetches_the_tag_instead_of_cloning(home, repo):
    m = Machine(home, repo)
    (m.paths.llama_src / ".git").mkdir(parents=True)
    code, _ = m.install()
    assert code == 0
    assert m.runner.ran("git", "clone") == []
    fetch = m.runner.ran("git", "-C", str(m.paths.llama_src), "fetch")[0]
    assert fetch[-3:] == ["--force", "origin", f"refs/tags/{TAG}:refs/tags/{TAG}"]


def test_unsupported_host_compiler_retries_with_gcc15(home, repo):
    attempts = []

    def configure(argv):
        attempts.append(argv)
        if len(attempts) == 1:
            return fail(stderr="#error -- unsupported GNU version! gcc versions later than 16 are not supported!")
        return ok()

    m = Machine(home, repo, configure=configure)
    code, ctx = m.install()
    assert code == 0, ctx.err.getvalue()
    assert len(attempts) == 2
    assert "-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-15" in attempts[1]
    assert "-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-15" not in attempts[0]


def test_unsupported_host_compiler_without_gcc15_stops(home, repo):
    m = Machine(home, repo, configure=fail(stderr="unsupported GNU version"), which={})
    code, ctx = m.install()
    assert code == 1
    assert "gcc15" in ctx.err.getvalue()
    assert len(m.runner.ran(*CONFIGURE)) == 1


def test_failed_help_check_names_the_missing_setting_and_keeps_current(home, repo):
    m = Machine(home, repo, help_ok=False)
    old = m.paths.llama_opt / "b11000"
    old.mkdir(parents=True)
    m.paths.llama_current.symlink_to(old)

    code, ctx = m.install()

    assert code == 1
    assert "LLAMA_ARG_SPEC_TYPE" in ctx.err.getvalue()
    assert os.readlink(m.paths.llama_current) == str(old)


def test_rebuild_with_failed_check_leaves_the_working_binary_untouched(home, repo):
    m = Machine(home, repo)
    assert m.install()[0] == 0
    working = (m.paths.tag_dir(TAG) / "bin" / "llama-server").read_bytes()

    bad = Machine(home, repo, help_ok=False, binary=b"server-broken")
    code, _ = bad.install("--rebuild")

    bin_dir = m.paths.tag_dir(TAG) / "bin"
    assert code == 1
    assert (bin_dir / "llama-server").read_bytes() == working
    assert [p.name for p in bin_dir.iterdir() if p.name.startswith(".staged")] == []
    assert os.readlink(m.paths.llama_current) == str(m.paths.tag_dir(TAG))


@pytest.mark.parametrize(
    "drop, missing",
    [(("LLAMA_ARG_MMPROJ",), ["LLAMA_ARG_MMPROJ"]), (("LLAMA_ARG_SPEC_TYPE",), ["LLAMA_ARG_SPEC_TYPE", "draft-mtp"])],
)
def test_help_check_needs_exact_names(repo, drop, missing):
    text = help_text(repo, drop=drop)
    required = profiles.all_setting_names(repo / "profiles")
    assert install.missing_from_help(text, required) == missing


def test_unit_is_written_and_reloaded_but_never_started(home, repo):
    m = Machine(home, repo)
    code, _ = m.install()
    assert code == 0
    assert m.paths.unit_file.read_text() == (repo / "systemd/qlm-server.service").read_text()
    assert m.runner.ran("systemctl", "--user", "daemon-reload")
    for verb in ("enable", "start", "stop", "restart"):
        assert m.runner.ran("systemctl", "--user", verb) == []


def test_existing_regular_file_at_the_link_path_stops_install(home, repo):
    m = Machine(home, repo)
    m.paths.local_bin.mkdir(parents=True)
    m.paths.qlm_link.write_text("someone else's script")
    code, ctx = m.install()
    assert code == 1
    assert str(m.paths.qlm_link) in ctx.err.getvalue()
    assert m.paths.qlm_link.read_text() == "someone else's script"


def test_link_to_this_repo_is_accepted_and_link_elsewhere_is_refused(home, repo, tmp_path):
    m = Machine(home, repo)
    m.paths.local_bin.mkdir(parents=True)
    m.paths.qlm_link.symlink_to(repo / "bin" / "qlm")
    assert m.install()[0] == 0
    assert os.readlink(m.paths.qlm_link) == str(repo / "bin" / "qlm")

    m.paths.qlm_link.unlink()
    m.paths.qlm_link.symlink_to(tmp_path / "elsewhere")
    code, ctx = m.install()
    assert code == 1
    assert os.readlink(m.paths.qlm_link) == str(tmp_path / "elsewhere")


def test_creates_link_and_local_bin_when_missing(home, repo):
    m = Machine(home, repo)
    assert m.install()[0] == 0
    assert os.path.realpath(m.paths.qlm_link) == os.path.realpath(repo / "bin" / "qlm")


def test_profile_and_key_are_created_once_with_private_modes(home, repo):
    m = Machine(home, repo)
    assert m.install()[0] == 0
    key = m.paths.api_key.read_bytes()
    env = m.paths.active_env.read_bytes()

    assert m.install()[0] == 0

    assert len(key) == 43 and key.decode().replace("-", "").replace("_", "").isalnum()
    assert mode(m.paths.api_key) == 0o600
    assert mode(m.paths.active_env) == 0o600
    assert mode(m.paths.config_dir) == 0o700
    assert m.paths.api_key.read_bytes() == key
    assert m.paths.active_env.read_bytes() == env
    assert env.decode() == profiles.load(repo / "profiles/agent.env").render(home)


def test_existing_config_dir_is_tightened_to_0700(home, repo):
    m = Machine(home, repo)
    m.paths.config_dir.mkdir(parents=True, mode=0o755)
    os.chmod(m.paths.config_dir, 0o755)
    assert m.install()[0] == 0
    assert mode(m.paths.config_dir) == 0o700


def test_installed_tag_skips_the_build_unless_rebuild(home, repo):
    m = Machine(home, repo)
    assert m.install()[0] == 0
    again = Machine(home, repo)
    (again.paths.unit_file).unlink()

    assert again.install()[0] == 0
    assert again.runner.ran("git") == [] and again.runner.ran("cmake") == []
    assert again.paths.unit_file.exists()

    third = Machine(home, repo)
    assert third.install("--rebuild")[0] == 0
    assert third.runner.ran("git") and third.runner.ran("cmake")


def test_failed_build_shows_the_log_tail_and_path_without_the_key(home, repo):
    m = Machine(home, repo)
    m.paths.config_dir.mkdir(parents=True)
    m.paths.api_key.write_text("SECRET-KEY-VALUE")
    long_output = "\n".join(f"compile line {i}" for i in range(100)) + "\nerror: nvcc fatal\n"
    m.runner.script[BUILD] = fail(stdout=long_output)

    code, ctx = m.install()

    err = ctx.err.getvalue()
    logs = list(m.paths.logs.glob("install-*.log"))
    assert code == 1
    assert "error: nvcc fatal" in err and "compile line 71" in err and "compile line 69" not in err
    assert len(logs) == 1 and str(logs[0]) in err
    assert mode(logs[0]) == 0o600
    assert "SECRET-KEY-VALUE" not in logs[0].read_text()
    assert err.rstrip().splitlines()[-1].startswith("qlm: ")


def test_missing_tag_upstream_stops_naming_the_tag(home, repo):
    m = Machine(home, repo)
    m.runner.script[("git", "-C", str(m.paths.llama_src), "rev-parse")] = fail(stderr="fatal: bad revision")
    code, ctx = m.install()
    assert code == 1 and TAG in ctx.err.getvalue()
    assert m.runner.ran("cmake") == []


def test_running_server_is_left_alone_with_a_note(home, repo):
    m = Machine(home, repo)
    m.runner.script[("systemctl", "--user", "show", "-p", "MainPID", "--value", "qlm-server")] = ok("4242\n")
    code, ctx = m.install()
    assert code == 0
    assert "next start" in ctx.out.getvalue()
    for verb in ("stop", "restart", "start"):
        assert m.runner.ran("systemctl", "--user", verb) == []


def test_two_runs_in_the_same_second_get_separate_logs(home, repo):
    m = Machine(home, repo)
    assert m.install()[0] == 0
    assert Machine(home, repo).install("--rebuild")[0] == 0
    assert len(list(m.paths.logs.glob("install-*.log"))) == 2
