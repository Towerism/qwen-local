"""`qlm doctor`: read-only checks of this machine before installing or starting the server."""

import os
from dataclasses import dataclass

from qlm import pins, profiles

GIB = 2**30
MIB = 2**20
BUILD_SPACE = 6 * GIB
MIN_RAM = 32 * GIB
PORT = 9931
HOST = "127.0.0.1"
CON_BAD_KEY = "con.agent.providers.openai-compatible."
TIMEOUT_S = 10


@dataclass
class Check:
    status: str
    name: str
    detail: str

    def line(self):
        return f"{self.status} {self.name}: {self.detail}"


def run(ctx):
    facts = _Facts(ctx)
    checks = [
        _gpu(facts),
        _compute_capability(facts),
        _nvcc(ctx),
        _host_compiler(ctx),
        _git_and_cmake(ctx),
        _disk(ctx),
        _ram(ctx),
        _port(ctx, facts),
        _ollama(ctx),
        _vram(ctx, facts),
        _path(ctx),
        _con(ctx),
    ]
    for c in checks:
        ctx.say(c.line())
    if any(c.status == "FAIL" for c in checks):
        return ctx.fail("one or more checks failed", "Fix the FAIL lines above and run qlm doctor again")
    return 0


def _run(ctx, argv):
    return ctx.runner.run(argv, timeout=TIMEOUT_S)


class _Facts:
    """Tool output shared by several checks, gathered once."""

    def __init__(self, ctx):
        self.gpu = None
        self.gpu_problem = NO_SMI
        self.apps = []
        if ctx.runner.which("nvidia-smi"):
            r = _run(
                ctx,
                ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total,memory.used,driver_version",
                 "--format=csv,noheader,nounits"],
            )
            if r.code == 0 and r.stdout.strip():
                try:
                    name, cap, total, used, driver = [f.strip() for f in r.stdout.splitlines()[0].split(",")]
                    self.gpu = {"name": name, "cap": cap, "total": int(total), "used": int(used), "driver": driver}
                except ValueError:
                    self.gpu_problem = "unparseable nvidia-smi output: " + r.stdout.splitlines()[0].strip()
            if self.gpu is not None:
                a = _run(
                    ctx, ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"]
                )
                for line in a.stdout.splitlines() if a.code == 0 else []:
                    fields = [f.strip() for f in line.split(",")]
                    if len(fields) >= 3:
                        # Process names can contain commas: the first field is the PID, the last the MiB.
                        self.apps.append((fields[0], ", ".join(fields[1:-1]).replace(", ", ","), fields[-1]))
        r = _run(ctx, ["systemctl", "--user", "show", "-p", "MainPID", "--value", "qlm-server"])
        pid = r.stdout.strip()
        self.server_pid = int(pid) if r.code == 0 and pid.isdigit() and int(pid) > 0 else None


NO_SMI = "nvidia-smi not available"


def _gpu(f):
    if f.gpu is None:
        return Check("FAIL", "gpu", f.gpu_problem)
    detail = f"{f.gpu['name']}, driver {f.gpu['driver']}"
    return Check("PASS" if f.gpu["name"] == "NVIDIA GeForce RTX 4090" else "WARN", "gpu", detail)


def _compute_capability(f):
    if f.gpu is None:
        return Check("FAIL", "compute capability", f.gpu_problem)
    ok = f.gpu["cap"] == "8.9"
    return Check("PASS" if ok else "FAIL", "compute capability", f.gpu["cap"] + ("" if ok else ", the build targets 8.9"))


def _nvcc(ctx):
    path = ctx.runner.which("nvcc", extra_dirs=["/opt/cuda/bin"])
    if not path:
        return Check("FAIL", "nvcc", "not found on PATH or in /opt/cuda/bin; install the cuda package")
    r = _run(ctx, [path, "--version"])
    release = next((l.split("release", 1)[1].strip() for l in r.stdout.splitlines() if "release" in l), "unknown version")
    return Check("PASS", "nvcc", f"{path}, release {release}")


def _host_compiler(ctx):
    if not ctx.runner.which("g++"):
        return Check("FAIL", "host compiler", "g++ not found; install gcc")
    if ctx.runner.which("g++-15"):
        return Check("PASS", "host compiler", "g++ with g++-15 as fallback")
    return Check("WARN", "host compiler", "g++ only; the build has no fallback if nvcc rejects it (install gcc15)")


def _git_and_cmake(ctx):
    missing = [t for t in ("git", "cmake") if not ctx.runner.which(t)]
    if missing:
        return Check("FAIL", "git and cmake", "missing: " + ", ".join(missing))
    return Check("PASS", "git and cmake", "found")


def _missing_model_bytes(ctx):
    missing = 0
    for f in pins.files():
        final = ctx.paths.models / f.local_name
        part = ctx.paths.models / (f.local_name + ".part")
        if final.exists() and final.stat().st_size == f.size:
            continue
        have = part.stat().st_size if part.exists() else 0
        missing += max(f.size - have, 0)
    return missing


def _disk(ctx):
    need = _missing_model_bytes(ctx) + BUILD_SPACE
    free = ctx.env.disk_free(ctx.paths.home)
    detail = f"{free / GIB:.1f} GiB free, {need / GIB:.1f} GiB needed"
    return Check("PASS" if free >= need else "FAIL", "disk space", detail)


def _ram(ctx):
    total = ctx.env.total_ram()
    return Check("PASS" if total >= MIN_RAM else "WARN", "ram", f"{total / GIB:.0f} GiB")


def _port(ctx, f):
    pid = ctx.env.listener_pid(HOST, PORT)
    if pid is None:
        return Check("PASS", f"port {PORT}", "free")
    if f.server_pid is not None and pid == f.server_pid:
        return Check("PASS", f"port {PORT}", "held by qlm-server")
    who = f"PID {pid}" if pid != "unknown" else "another user's process"
    return Check("FAIL", f"port {PORT}", f"in use by {who}; stop it or free the port")


def _ollama(ctx):
    if not ctx.runner.which("ollama"):
        return Check("PASS", "ollama", "not installed")
    r = _run(ctx, ["ollama", "ps"])
    if r.code != 0:
        return Check("PASS", "ollama", "daemon not running")
    on_gpu = [l.split()[0] for l in r.stdout.splitlines()[1:] if l.strip() and "GPU" in l]
    if on_gpu:
        names = ", ".join(on_gpu)
        return Check("WARN", "ollama", f"model on the GPU: {names}; free it with: ollama stop {on_gpu[0]}")
    return Check("PASS", "ollama", "no model on the GPU")


def _vram(ctx, f):
    if f.gpu is None:
        return Check("FAIL", "vram", f.gpu_problem)
    if f.server_pid is not None:
        return Check("PASS", "vram", "server running")
    agent = profiles.load(ctx.paths.profiles_dir / "agent.env")
    need_mib = agent.vram_gib * 1024 + 512
    free_mib = f.gpu["total"] - f.gpu["used"]
    detail = f"{free_mib / 1024:.1f} GiB free, {need_mib / 1024:.1f} GiB needed by the agent profile"
    if free_mib >= need_mib:
        return Check("PASS", "vram", detail)
    users = "; ".join(f"{name} ({mib} MiB)" for _, name, mib in f.apps) or "no compute processes listed"
    return Check("WARN", "vram", f"{detail}; in use by: {users}")


def _path(ctx):
    wanted = os.path.normpath(ctx.paths.local_bin)
    if any(entry and os.path.normpath(entry) == wanted for entry in ctx.env.path):
        return Check("PASS", "~/.local/bin on PATH", "yes")
    return Check("WARN", "~/.local/bin on PATH", 'no; add to ~/.zshrc: export PATH="$HOME/.local/bin:$PATH"')


def _con(ctx):
    conf = ctx.paths.con_conf
    if not conf.exists():
        return Check("PASS", "con config", "Con config not found")
    for n, line in enumerate(conf.read_text(errors="replace").splitlines(), start=1):
        if line.lstrip().startswith(CON_BAD_KEY):
            return Check(
                "FAIL", "con config",
                f"line {n} uses {CON_BAD_KEY}*, which stops Con loading its config; rename it to con.agent.providers.openaicompatible.*",
            )
    return Check("PASS", "con config", "no invalid provider keys")
