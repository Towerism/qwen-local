"""The seams added for running the server: local HTTP requests, terminal passthrough and XDG_CONFIG_HOME."""

import http.server
import io
import json
import subprocess
import sys
import threading
import urllib.error

import pytest

from qlm.system import Env, Http, Runner, ServerUnreachable


class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _reply(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "http://127.0.0.1:1/elsewhere")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.path == "/missing":
            payload = b'{"error":"nope"}'
            self.send_response(404)
        else:
            payload = json.dumps({
                "method": self.command,
                "auth": self.headers.get("Authorization"),
                "type": self.headers.get("Content-Type"),
                "body": body.decode(),
            }).encode()
            self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    do_GET = do_POST = _reply


@pytest.fixture
def local_server():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


@pytest.mark.loopback
def test_request_sends_json_and_headers_and_returns_the_whole_body(local_server):
    resp = Http(server=local_server).request(
        "POST", local_server + "/v1/x", headers={"Authorization": "Bearer k"}, json_body={"a": 1}, timeout_s=5
    )
    assert resp.status == 200
    assert json.loads(resp.body) == {"method": "POST", "auth": "Bearer k", "type": "application/json", "body": '{"a": 1}'}


@pytest.mark.loopback
def test_request_returns_error_statuses_with_their_body(local_server):
    resp = Http(server=local_server).request("GET", local_server + "/missing", timeout_s=5)
    assert resp.status == 404 and resp.body == b'{"error":"nope"}'


@pytest.mark.loopback
def test_request_treats_a_redirect_as_unreachable(local_server):
    with pytest.raises(ServerUnreachable, match="redirect"):
        Http(server=local_server).request("GET", local_server + "/redirect", timeout_s=5)


@pytest.mark.parametrize(
    "url", ["http://127.0.0.2:9931/health", "http://127.0.0.1:8080/health", "https://127.0.0.1:9931/health"]
)
def test_request_refuses_any_other_server_before_connecting(url):
    with pytest.raises(ServerUnreachable, match="refused"):
        Http().request("GET", url, timeout_s=5)


@pytest.mark.loopback
def test_request_goes_direct_even_with_a_proxy_configured(monkeypatch, local_server):
    monkeypatch.setenv("http_proxy", "http://10.255.255.1:3128")
    monkeypatch.setenv("HTTP_PROXY", "http://10.255.255.1:3128")
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.delenv("NO_PROXY", raising=False)
    resp = Http(server=local_server).request("GET", local_server + "/health", timeout_s=5)
    assert resp.status == 200


def test_request_reports_a_refused_connection_as_unreachable(monkeypatch):
    http = Http()

    def refused(req, timeout):
        raise urllib.error.URLError(ConnectionRefusedError(111, "Connection refused"))

    monkeypatch.setattr(http.local_opener, "open", refused)
    with pytest.raises(ServerUnreachable, match="Connection refused"):
        http.request("GET", "http://127.0.0.1:9931/health", timeout_s=5)


def test_request_reports_a_timeout_as_unreachable(monkeypatch):
    http = Http()

    def slow(req, timeout):
        raise TimeoutError("timed out")

    monkeypatch.setattr(http.local_opener, "open", slow)
    with pytest.raises(ServerUnreachable, match="timed out after 5 s"):
        http.request("GET", "http://127.0.0.1:9931/health", timeout_s=5)


def test_request_reports_a_reply_cut_off_mid_body_as_unreachable(monkeypatch):
    import http.client

    http_ = Http()

    def cut_off(req, timeout):
        raise http.client.IncompleteRead(b"partial", 100)

    monkeypatch.setattr(http_.local_opener, "open", cut_off)
    with pytest.raises(ServerUnreachable, match="IncompleteRead"):
        http_.request("POST", "http://127.0.0.1:9931/v1/chat/completions", headers={"Authorization": "Bearer k"}, timeout_s=5)


def test_passthrough_returns_the_exit_code_without_capturing(monkeypatch):
    seen = {}

    def run(argv, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(argv, 3)

    monkeypatch.setattr(subprocess, "run", run)
    assert Runner().passthrough(["journalctl", "-f"]) == 3
    assert "capture_output" not in seen and "stdout" not in seen


def test_passthrough_reports_a_missing_program_as_127(monkeypatch):
    def missing(argv, **kwargs):
        raise FileNotFoundError(argv[0])

    monkeypatch.setattr(subprocess, "run", missing)
    assert Runner().passthrough(["nope"]) == 127


def test_xdg_config_home_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", "/x/cfg")
    assert Env().xdg_config_home() == "/x/cfg"
    monkeypatch.delenv("XDG_CONFIG_HOME")
    assert Env().xdg_config_home() is None


class _FakePopen:
    """Stands in for subprocess.Popen: stdout yields fixed lines, and terminate ends the process."""

    started = []

    def __init__(self, argv, **kwargs):
        self.argv, self.kwargs = argv, kwargs
        self.stdout = io.StringIO("2096, 73\n9105, 19860\n")
        self.returncode = None
        self.terminated = False
        _FakePopen.started.append(self)

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


def test_stream_yields_lines_without_endings_and_stop_ends_the_process(monkeypatch):
    _FakePopen.started.clear()
    monkeypatch.setattr(subprocess, "Popen", _FakePopen)
    handle = Runner().stream(["nvidia-smi", "--loop-ms=200"])
    assert list(handle.lines()) == ["2096, 73", "9105, 19860"]
    handle.stop()
    proc = _FakePopen.started[0]
    assert proc.terminated
    assert proc.kwargs["stdout"] == subprocess.PIPE and "shell" not in proc.kwargs


def test_stream_of_a_missing_program_yields_nothing(monkeypatch):
    def missing(argv, **kwargs):
        raise FileNotFoundError(argv[0])

    monkeypatch.setattr(subprocess, "Popen", missing)
    handle = Runner().stream(["nvidia-smi"])
    assert list(handle.lines()) == []
    handle.stop()


def test_ask_prints_the_prompt_and_returns_one_line_without_its_ending(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO("yes\r\nignored\n"))
    assert Env().ask("Type yes to remove these:") == "yes"
    assert "Type yes to remove these:" in capsys.readouterr().out


def test_ask_returns_empty_at_end_of_input(monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    assert Env().ask("?") == ""


def test_stdin_is_tty_follows_standard_input(monkeypatch):
    class Tty(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.setattr(sys, "stdin", Tty())
    assert Env().stdin_is_tty() is True
    monkeypatch.setattr(sys, "stdin", io.StringIO())
    assert Env().stdin_is_tty() is False
    closed = io.StringIO()
    closed.close()
    monkeypatch.setattr(sys, "stdin", closed)
    assert Env().stdin_is_tty() is False
