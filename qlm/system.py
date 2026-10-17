"""The seams between qlm and the outside world: processes, HTTP, time and the host.

Commands receive these through a Ctx so tests can replace each one with a fake.
"""

import datetime
import os
import re
import secrets
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, TextIO

from qlm.paths import Paths

USER_AGENT = "qlm/0.1"
CHUNK = 1 << 20


@dataclass
class RunResult:
    code: int
    stdout: str
    stderr: str

    @property
    def output(self):
        return self.stdout + self.stderr


@dataclass
class HttpResponse:
    status: int
    headers: dict
    body: Iterable[bytes]


class RedirectRefused(Exception):
    """A download redirected somewhere other than HTTPS huggingface.co or *.hf.co."""


def redirect_allowed(url):
    parts = urllib.parse.urlsplit(url)
    host = (parts.hostname or "").lower()
    return parts.scheme == "https" and (host == "huggingface.co" or host.endswith(".hf.co"))


class SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not redirect_allowed(newurl):
            host = urllib.parse.urlsplit(newurl).hostname or newurl
            raise RedirectRefused(f"refused redirect to {urllib.parse.urlsplit(newurl).scheme}://{host}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class Runner:
    """Runs commands from argv lists, never through a shell."""

    def run(self, argv, cwd=None, env=None):
        full_env = None
        if env:
            full_env = dict(os.environ)
            full_env.update(env)
        try:
            p = subprocess.run(
                [str(a) for a in argv], cwd=cwd, env=full_env, capture_output=True, text=True, errors="replace"
            )
        except FileNotFoundError as e:
            return RunResult(127, "", str(e))
        return RunResult(p.returncode, p.stdout, p.stderr)

    def which(self, name, extra_dirs=()):
        found = shutil.which(name)
        if found:
            return found
        for d in extra_dirs:
            candidate = Path(d) / name
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
        return None


class Http:
    def __init__(self):
        self.opener = urllib.request.build_opener(SafeRedirectHandler)

    def get(self, url, range_start=None, timeout_s=60):
        headers = {"User-Agent": USER_AGENT}
        if range_start:
            headers["Range"] = f"bytes={range_start}-"
        req = urllib.request.Request(url, headers=headers)
        try:
            resp = self.opener.open(req, timeout=timeout_s)
        except urllib.error.HTTPError as e:
            return HttpResponse(e.code, dict(e.headers or {}), iter(()))
        if not redirect_allowed(resp.geturl()):
            resp.close()
            raise RedirectRefused(f"refused redirect to {resp.geturl()}")
        return HttpResponse(resp.status, {k.lower(): v for k, v in resp.headers.items()}, self._body(resp))

    @staticmethod
    def _body(resp):
        with resp:
            while True:
                chunk = resp.read(CHUNK)
                if not chunk:
                    return
                yield chunk


class Clock:
    def now(self):
        return datetime.datetime.now(datetime.timezone.utc)

    def sleep(self, seconds):
        time.sleep(seconds)


class Env:
    def __init__(self, path=None):
        self.path = path if path is not None else os.environ.get("PATH", "").split(os.pathsep)

    def disk_free(self, path):
        p = Path(path)
        while not p.exists():
            p = p.parent
        return shutil.disk_usage(p).free

    def total_ram(self):
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")

    def listener_pid(self, host, port):
        """PID listening on host:port, None if free, "unknown" if held but the PID is hidden."""
        try:
            p = subprocess.run(
                ["ss", "-Hltnp", f"sport = :{port}"], capture_output=True, text=True, errors="replace"
            )
        except FileNotFoundError:
            return "unknown"
        lines = [l for l in p.stdout.splitlines() if f"{host}:{port}" in l or f"*:{port}" in l or f"[::]:{port}" in l]
        if not lines:
            return None
        m = re.search(r"pid=(\d+)", lines[0])
        return int(m.group(1)) if m else "unknown"

    def is_tty(self, stream):
        return sys.stderr.isatty()

    def random_token(self, nbytes):
        return secrets.token_urlsafe(nbytes)

    def cpu_count(self):
        return os.cpu_count() or 1


@dataclass
class Ctx:
    paths: Paths
    runner: Any
    http: Any
    clock: Any
    env: Any
    out: TextIO
    err: TextIO

    def say(self, line):
        print(line, file=self.out)

    def warn(self, line):
        print(line, file=self.err)

    def fail(self, what, todo):
        print(f"qlm: {what}. {todo}", file=self.err)
        return 1


def default_ctx():
    repo = Path(__file__).resolve().parent.parent
    return Ctx(
        paths=Paths(home=Path.home(), repo=repo),
        runner=Runner(),
        http=Http(),
        clock=Clock(),
        env=Env(),
        out=sys.stdout,
        err=sys.stderr,
    )
