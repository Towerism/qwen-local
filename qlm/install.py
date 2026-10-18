"""`qlm install`: build the pinned llama.cpp and put the service, profile, key and link in place.

A new build is staged and checked before it replaces anything (ADR 0001: a bad build never
replaces a working one).
"""

import os
import re
import shutil

from qlm import pins, profiles

CMAKE_FLAGS = (
    "-DGGML_CUDA=ON",
    "-DCMAKE_CUDA_ARCHITECTURES=89",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DBUILD_SHARED_LIBS=OFF",
)
BINARIES = ("llama-server", "llama-bench")
UNSUPPORTED_COMPILER = re.compile(r"unsupported (gnu version|host compiler)", re.I)
TAIL_LINES = 30
GCC15 = "/usr/bin/g++-15"


class _Stop(Exception):
    """A step failed; carries the summary line's two halves."""

    def __init__(self, what, todo):
        super().__init__(what)
        self.what, self.todo = what, todo


def missing_from_help(text, required):
    """Names from `required` that `llama-server --help` doesn't list, plus draft-mtp if absent."""
    missing = [name for name in required if f"(env: {name})" not in text]
    if not _spec_type_mentions_draft_mtp(text):
        missing.append("draft-mtp")
    return missing


def _spec_type_mentions_draft_mtp(text):
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.lstrip().startswith("--spec-type"):
            block = [line]
            for follow in lines[i + 1 :]:
                if follow.lstrip().startswith("-"):
                    break
                block.append(follow)
            return re.search(r"\bdraft-mtp\b", "\n".join(block)) is not None
    return False


def run(ctx, rebuild=False):
    tag = pins.LLAMA_TAG
    p = ctx.paths
    required = profiles.all_setting_names(p.profiles_dir)
    bin_dir = p.tag_dir(tag) / "bin"
    try:
        built = False
        if rebuild or not _installed_and_valid(ctx, bin_dir / "llama-server", required):
            with _Log(ctx) as log:
                _build(ctx, tag, required, bin_dir, log)
            built = True
        else:
            ctx.say(f"llama.cpp {tag} is already installed; skipping the build (use --rebuild to build again)")
        switched = _switch_current(ctx, tag)
        _write_unit(ctx)
        _link_entry_point(ctx)
        _write_private_files(ctx)
    except _Stop as stop:
        return ctx.fail(stop.what, stop.todo)
    except OSError as e:
        return ctx.fail(f"file error: {e}", "Nothing after this step was changed; fix the cause and run qlm install again")
    if (built or switched) and _server_pid(ctx):
        ctx.say("qlm-server is running; the new build takes effect at the next start.")
    ctx.say("Install complete.")
    return 0


def _installed_and_valid(ctx, server, required):
    if not server.exists():
        return False
    return not missing_from_help(ctx.runner.run([server, "--help"]).output, required)


class _Log:
    """The install log, created owner-only. Never receives the API key."""

    def __init__(self, ctx):
        ctx.paths.logs.mkdir(parents=True, exist_ok=True)
        stamp = ctx.clock.now().strftime("%Y%m%dT%H%M%SZ")
        for n in range(1000):
            path = ctx.paths.logs / (f"install-{stamp}.log" if n == 0 else f"install-{stamp}-{n}.log")
            try:
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                break
            except FileExistsError:
                continue
        else:
            raise _Stop(f"could not create a new log file in {ctx.paths.logs}", "Remove old install logs and try again")
        self.path = path
        self.file = os.fdopen(fd, "w")
        self.ctx = ctx

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.file.close()

    def step(self, label, argv, cwd=None, allow_fail=False):
        self.ctx.say(label)
        self.file.write(f"$ {' '.join(str(a) for a in argv)}\n")
        result = self.ctx.runner.run(argv, cwd=cwd)
        self.file.write(result.output)
        self.file.write(f"[exit {result.code}]\n")
        self.file.flush()
        if result.code != 0 and not allow_fail:
            self.show_tail(result)
            raise _Stop(f"{label} failed (exit {result.code})", f"See the full log at {self.path}")
        return result

    def show_tail(self, result):
        for line in result.output.rstrip().splitlines()[-TAIL_LINES:]:
            self.ctx.warn(line)
        self.ctx.warn(f"Full log: {self.path}")


def _build(ctx, tag, required, bin_dir, log):
    p = ctx.paths
    src = p.llama_src
    if (src / ".git").exists():
        log.step(
            f"Fetching tag {tag}",
            ["git", "-C", src, "fetch", "--force", pins.LLAMA_REPO_URL, f"refs/tags/{tag}:refs/tags/{tag}"],
        )
    else:
        src.parent.mkdir(parents=True, exist_ok=True)
        log.step("Cloning llama.cpp", ["git", "clone", "--no-recurse-submodules", pins.LLAMA_REPO_URL, src])

    resolved = log.step(f"Resolving {tag}", ["git", "-C", src, "rev-parse", f"refs/tags/{tag}^{{commit}}"], allow_fail=True)
    if resolved.code != 0:
        log.show_tail(resolved)
        raise _Stop(f"tag {tag} was not found in llama.cpp", "Check the pin in ADR 0001; qlm won't fall back to another tag")
    commit = resolved.stdout.strip()
    if commit != pins.LLAMA_COMMIT:
        raise _Stop(
            f"tag {tag} points at {commit}, but ADR 0001 pins {pins.LLAMA_COMMIT}",
            "Nothing was built. Check whether the tag was moved upstream",
        )
    log.step(f"Checking out {commit[:12]}", ["git", "-C", src, "checkout", "--detach", commit])

    build = p.build_dir(tag)
    build.parent.mkdir(parents=True, exist_ok=True)
    configure = ["cmake", "-S", src, "-B", build, *CMAKE_FLAGS]
    first = log.step("Configuring the CUDA build", configure, allow_fail=True)
    if first.code != 0:
        if not UNSUPPORTED_COMPILER.search(first.output):
            log.show_tail(first)
            raise _Stop("configuring the build failed", f"See the full log at {log.path}")
        if not ctx.runner.which(GCC15):
            log.show_tail(first)
            raise _Stop("nvcc rejected the host compiler and g++-15 is not installed", "Install the gcc15 package and run qlm install again")
        shutil.rmtree(build, ignore_errors=True)
        log.step("Configuring again with g++-15", [*configure, f"-DCMAKE_CUDA_HOST_COMPILER={GCC15}"])

    log.step("Building llama-server and llama-bench", ["cmake", "--build", build, "--target", *BINARIES, "-j", str(ctx.env.cpu_count())])

    bin_dir.mkdir(parents=True, exist_ok=True)
    staged = {name: bin_dir / f".staged-{name}" for name in BINARIES}
    try:
        for name, dest in staged.items():
            shutil.copyfile(build / "bin" / name, dest)
            os.chmod(dest, 0o755)
        help_out = log.step("Checking the new llama-server", [staged["llama-server"], "--help"], allow_fail=True)
        missing = missing_from_help(help_out.output, required)
        if missing:
            raise _Stop(
                "the new llama-server --help lacks: " + ", ".join(missing),
                "Nothing was replaced. The profiles or the pinned tag need updating",
            )
        for name, dest in staged.items():
            os.replace(dest, bin_dir / name)
    finally:
        for dest in staged.values():
            if dest.exists():
                dest.unlink()


def _switch_current(ctx, tag):
    """Point `current` at this tag, unless it already leads to another working build.

    A working build there was put in place by `qlm update`; switching back would undo it.
    """
    target = ctx.paths.tag_dir(tag)
    current = ctx.paths.llama_current
    if current.is_symlink() and os.readlink(current) == str(target):
        return False
    if current.is_symlink() and (current / "bin/llama-server").exists():
        ctx.say(f"current is on {os.path.basename(os.readlink(current))}, set by qlm update; leaving it there.")
        return False
    tmp = current.with_name(f".current.tmp-{os.getpid()}")
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    os.symlink(target, tmp)
    os.replace(tmp, current)
    ctx.say(f"{current} -> {target}")
    return True


def _write_unit(ctx):
    unit = ctx.paths.unit_file
    unit.parent.mkdir(parents=True, exist_ok=True)
    unit.write_text(ctx.paths.unit_template.read_text())
    os.chmod(unit, 0o644)
    r = ctx.runner.run(["systemctl", "--user", "daemon-reload"])
    if r.code != 0:
        raise _Stop("systemctl --user daemon-reload failed: " + r.output.strip(), "Check that the systemd user instance is running")


def _link_entry_point(ctx):
    link, target = ctx.paths.qlm_link, ctx.paths.entry_point
    link.parent.mkdir(parents=True, exist_ok=True)
    if not os.path.lexists(link):
        os.symlink(target, link)
        return
    if os.path.islink(link) and os.path.realpath(link) == os.path.realpath(target):
        return
    raise _Stop(f"{link} already exists and is not a link to this repo's bin/qlm", "Move it aside and run qlm install again")


def _write_private_files(ctx):
    p = ctx.paths
    p.config_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(p.config_dir, 0o700)
    for path in (p.active_env, p.api_key):
        if path.exists() and path.stat().st_size == 0:
            raise _Stop(f"{path} is empty", "Delete it and run qlm install again to create a new one")
    if not p.active_env.exists():
        agent = profiles.load(p.profiles_dir / "agent.env")
        _create_private(p.active_env, agent.render(p.home))
    if not p.api_key.exists():
        _create_private(p.api_key, ctx.env.random_token(32))


def _create_private(path, text):
    """Write owner-only to a temporary file, then hard-link it into place, so `path` never exists half-written."""
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.link(tmp, path)
    finally:
        tmp.unlink()


def _server_pid(ctx):
    r = ctx.runner.run(["systemctl", "--user", "show", "-p", "MainPID", "--value", "qlm-server"])
    pid = r.stdout.strip()
    return int(pid) if r.code == 0 and pid.isdigit() and int(pid) > 0 else None
