"""The seams added for running the server: local HTTP requests, terminal passthrough and XDG_CONFIG_HOME."""

import http.server
import json
import subprocess
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
