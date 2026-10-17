"""The seams between qlm and the outside world: processes, HTTP, time and the host.

Commands receive these through a Ctx so tests can replace each one with a fake.
"""

import datetime
import json
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


@dataclass
class LocalResponse:
    status: int
    headers: dict
    body: bytes


SERVER = "http://127.0.0.1:9931"


class RedirectRefused(Exception):
    """A download redirected somewhere other than HTTPS huggingface.co or *.hf.co."""


class ServerUnreachable(Exception):
    """A request to the local server got no HTTP answer: refused, timed out, redirected or not local."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


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

    def run(self, argv, cwd=None, env=None, timeout=None):
        full_env = None
        if env:
            full_env = dict(os.environ)
            full_env.update(env)
        try:
            p = subprocess.run(
                [str(a) for a in argv], cwd=cwd, env=full_env, capture_output=True, text=True, errors="replace",
                timeout=timeout,
            )
        except FileNotFoundError as e:
            return RunResult(127, "", str(e))
        except subprocess.TimeoutExpired:
            return RunResult(124, "", f"{argv[0]} timed out after {timeout} s")
        except OSError as e:
            return RunResult(126, "", str(e))
        return RunResult(p.returncode, p.stdout, p.stderr)

    def passthrough(self, argv):
        """Run a command on the terminal, uncaptured (progress bars, journalctl -f). Returns its exit code."""
        try:
            return subprocess.run([str(a) for a in argv]).returncode
        except FileNotFoundError:
            return 127
        except OSError:
            return 126

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
    def __init__(self, server=SERVER):
        self.opener = urllib.request.build_opener(SafeRedirectHandler)
        # Requests carrying the API key must reach 127.0.0.1 directly: no proxy, no redirects.
        self.local_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect)
        self.server = server

    def request(self, method, url, headers=None, json_body=None, timeout_s=10):
        """One request to the local llama-server. Never puts header values in an error message."""
        if not (url == self.server or url.startswith(self.server + "/")):
            raise ServerUnreachable(f"refused a request outside {self.server}")
        data = None
        all_headers = {"User-Agent": USER_AGENT, **(headers or {})}
        if json_body is not None:
            data = json.dumps(json_body).encode()
            all_headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=all_headers, method=method)
        try:
            with self.local_opener.open(req, timeout=timeout_s) as resp:
                return LocalResponse(resp.status, {k.lower(): v for k, v in resp.headers.items()}, resp.read())
        except urllib.error.HTTPError as e:
            if 300 <= e.code < 400:
                raise ServerUnreachable(f"the server answered with a redirect (HTTP {e.code})") from None
            body = e.read() if e.fp else b""
            return LocalResponse(e.code, {k.lower(): v for k, v in (e.headers or {}).items()}, body)
        except urllib.error.URLError as e:
            reason = e.reason
            if isinstance(reason, TimeoutError):
                raise ServerUnreachable(f"timed out after {timeout_s} s") from None
            raise ServerUnreachable(getattr(reason, "strerror", None) or str(reason)) from None
        except TimeoutError:
            raise ServerUnreachable(f"timed out after {timeout_s} s") from None
        except (ConnectionError, OSError) as e:
            raise ServerUnreachable(e.strerror or str(e)) from None

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
        """PID listening where it blocks host:port, None if free, "unknown" if held but the PID is hidden."""
        r = Runner().run(["ss", "-Hltnp", f"sport = :{port}"], timeout=10)
        if r.code != 0:
            return "unknown"
        return parse_ss_listener(r.stdout, host, port)

    def is_tty(self, stream):
        return sys.stderr.isatty()

    def random_token(self, nbytes):
        return secrets.token_urlsafe(nbytes)

    def cpu_count(self):
        return os.cpu_count() or 1

    def xdg_config_home(self):
        return os.environ.get("XDG_CONFIG_HOME") or None


def parse_ss_listener(stdout, host, port):
    """From `ss -Hltnp` output, the PID of a listener that stops a bind to host:port.

    A wildcard address (0.0.0.0, *, [::]) or host itself blocks it; another specific address doesn't.
    """
    blocking = {host, "0.0.0.0", "*", "[::]", f"[::ffff:{host}]", "[::ffff:0.0.0.0]"}
    for line in stdout.splitlines():
        fields = line.split()
        local = next((f for f in fields if f.endswith(f":{port}")), None)
        if local is None:
            continue
        addr = local.rsplit(":", 1)[0].split("%", 1)[0]
        if addr not in blocking:
            continue
        m = re.search(r"pid=(\d+)", line)
        return int(m.group(1)) if m else "unknown"
    return None


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
        print(line, file=self.out, flush=True)

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
