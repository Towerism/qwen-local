"""Test doubles for the seams in qlm.system: Runner, Http, Clock and Env."""

import datetime
import hashlib
import io
import json
from dataclasses import dataclass, field

from qlm import pins
from qlm.paths import Paths
from qlm.system import Ctx, HttpResponse, LocalResponse, RunResult


class FakeRunner:
    """Answers commands from a script keyed by argv prefix and records every call.

    A script value is a RunResult, a callable taking argv and returning one, or a list of
    either, used one per call with the last repeated. The longest matching prefix wins.
    Unscripted commands fail loudly. `passthrough` answers from the same script with the
    result's exit code, or raises the value if it is an exception. `stream` answers with a
    FakeStream over the result's stdout lines; every stream handed out is kept in `streams`.
    """

    def __init__(self, script=None, which=None):
        self.script = dict(script or {})
        self.which_map = dict(which or {})
        self.calls = []
        self.passthrough_calls = []
        self.streams = []

    def _answer(self, argv):
        best = None
        for key, value in self.script.items():
            if tuple(argv[: len(key)]) == key and (best is None or len(key) > len(best[0])):
                best = (key, value)
        if best is None:
            raise AssertionError(f"unscripted command: {argv}")
        value = best[1]
        if isinstance(value, list):
            value = value.pop(0) if len(value) > 1 else value[0]
        if isinstance(value, BaseException):
            raise value
        return value(argv) if callable(value) else value

    def run(self, argv, cwd=None, env=None, timeout=None):
        argv = [str(a) for a in argv]
        self.calls.append({"argv": argv, "cwd": cwd, "env": env, "timeout": timeout})
        return self._answer(argv)

    def passthrough(self, argv):
        argv = [str(a) for a in argv]
        self.passthrough_calls.append(argv)
        self.calls.append({"argv": argv, "cwd": None, "env": None, "timeout": None})
        return self._answer(argv).code

    def stream(self, argv):
        argv = [str(a) for a in argv]
        self.calls.append({"argv": argv, "cwd": None, "env": None, "timeout": None})
        handle = FakeStream(argv, self._answer(argv).stdout.splitlines())
        self.streams.append(handle)
        return handle

    def which(self, name, extra_dirs=()):
        return self.which_map.get(name)

    def ran(self, *prefix):
        return [c["argv"] for c in self.calls if tuple(c["argv"][: len(prefix)]) == prefix]


class FakeStream:
    """A finished process's output: lines() yields the fixed lines, then ends."""

    def __init__(self, argv, lines):
        self.argv, self._lines, self.stopped = argv, list(lines), False

    def lines(self):
        return iter(self._lines)

    def stop(self):
        self.stopped = True


def ok(stdout="", stderr=""):
    return RunResult(0, stdout, stderr)


def fail(stdout="", stderr="", code=1):
    return RunResult(code, stdout, stderr)


class FakeHttp:
    """Serves in-memory files by URL, with switches for Range support, drops and errors.

    `drops` is a list of byte counts: each GET sends that many bytes and then raises
    ConnectionResetError, until the list is used up.
    """

    def __init__(self, files=None, ranges=True, drops=None, status=None, raise_on_get=None, chunk=1024):
        self.files = dict(files or {})
        self.ranges = ranges
        self.drops = list(drops or [])
        self.status = status
        self.raise_on_get = raise_on_get
        self.chunk = chunk
        self.requests = []

    def get(self, url, range_start=None, timeout_s=60):
        self.requests.append({"url": url, "range_start": range_start})
        if self.raise_on_get is not None:
            raise self.raise_on_get
        if self.status is not None:
            return HttpResponse(self.status, {}, iter(()))
        data = self.files[url]
        start = range_start or 0
        if start and not self.ranges:
            start, status = 0, 200
        elif start:
            if start >= len(data):
                return HttpResponse(416, {}, iter(()))
            status = 206
        else:
            status = 200
        body = data[start:]
        drop = self.drops.pop(0) if self.drops else None
        headers = {"content-length": str(len(body))}
        if status == 206:
            headers["content-range"] = f"bytes {start}-{len(data) - 1}/{len(data)}"
        return HttpResponse(status, headers, self._chunks(body, drop))

    def _chunks(self, body, drop):
        sent = 0
        for i in range(0, len(body), self.chunk):
            piece = body[i : i + self.chunk]
            if drop is not None and sent + len(piece) > drop:
                if drop > sent:
                    yield piece[: drop - sent]
                raise ConnectionResetError("connection reset by peer")
            sent += len(piece)
            yield piece
        if drop is not None:
            raise ConnectionResetError("connection reset by peer")


class FakeClock:
    """A fixed clock. With advance=True, each sleep moves now() forward by the time slept."""

    def __init__(self, now=datetime.datetime(2026, 10, 17, 12, 0, 0, tzinfo=datetime.timezone.utc), advance=False):
        self._now = now
        self.advance = advance
        self.sleeps = []

    def now(self):
        return self._now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        if self.advance:
            self._now += datetime.timedelta(seconds=seconds)


class FakeServer:
    """Answers Http.request like llama-server, from replies keyed by (method, path).

    A reply is a LocalResponse, an exception to raise, a callable taking the request
    record and returning either, or a list of those used one per call with the last
    repeated. Unscripted paths fail loudly. Every request is recorded.
    """

    def __init__(self, replies=None):
        self.replies = dict(replies or {})
        self.requests = []

    def request(self, method, url, headers=None, json_body=None, timeout_s=10):
        path = url.split("9931", 1)[1] if "9931" in url else url
        record = {"method": method, "path": path, "headers": dict(headers or {}), "json": json_body, "timeout_s": timeout_s}
        self.requests.append(record)
        if (method, path) not in self.replies:
            raise AssertionError(f"unscripted request: {method} {path}")
        value = self.replies[(method, path)]
        if isinstance(value, list):
            value = value.pop(0) if len(value) > 1 else value[0]
        if callable(value) and not isinstance(value, type):
            value = value(record)
        if isinstance(value, BaseException):
            raise value
        return value

    def to(self, method, path):
        return [r for r in self.requests if r["method"] == method and r["path"] == path]


def reply(status=200, body=b"", headers=None):
    if not isinstance(body, (bytes, str)):
        body = json.dumps(body)
    if isinstance(body, str):
        body = body.encode()
    return LocalResponse(status, dict(headers or {}), body)


@dataclass
class FakeEnv:
    path: list = field(default_factory=list)
    disk_free_bytes: int = 500 * 2**30
    total_ram_bytes: int = 62 * 2**30
    listener: object = None
    tty: bool = False
    token: str = "k" * 43
    cpus: int = 32
    xdg: object = None
    stdin_tty: bool = False
    answers: list = field(default_factory=list)
    prompts: list = field(default_factory=list)

    def stdin_is_tty(self):
        return self.stdin_tty

    def ask(self, prompt):
        self.prompts.append(prompt)
        return self.answers.pop(0) if self.answers else ""

    def xdg_config_home(self):
        return self.xdg

    def disk_free(self, path):
        return self.disk_free_bytes

    def total_ram(self):
        return self.total_ram_bytes

    def listener_pid(self, host, port):
        return self.listener

    def is_tty(self, stream):
        return self.tty

    def random_token(self, nbytes):
        return self.token

    def cpu_count(self):
        return self.cpus


def make_ctx(home, repo, runner=None, http=None, clock=None, env=None):
    paths = Paths(home=home, repo=repo)
    if env is None:
        env = FakeEnv(path=[str(paths.local_bin), "/usr/bin"])
    return Ctx(
        paths=paths,
        runner=runner or FakeRunner(),
        http=http or FakeHttp(),
        clock=clock or FakeClock(),
        env=env,
        out=io.StringIO(),
        err=io.StringIO(),
    )


def small_pins(monkeypatch):
    """Replace the real pinned files with small byte strings and their real hashes."""
    contents = {}
    files = []
    for f in pins.FILES:
        data = (f.local_name.encode() * 400)[: 3000 + len(files) * 500]
        contents[f.local_name] = data
        files.append(
            pins.PinnedFile(
                repo_path=f.repo_path,
                local_name=f.local_name,
                size=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
                vision_only=f.vision_only,
            )
        )
    monkeypatch.setattr(pins, "FILES", tuple(files))
    urls = {pins.download_url(f): contents[f.local_name] for f in files}
    return files, contents, urls
