import io
import subprocess
import urllib.error

import pytest

from qlm import system
from qlm.system import Env, Http, Runner, parse_ss_listener

SS_LINE = "LISTEN 0      4096   {addr}:9931 0.0.0.0:*    users:((\"llama-server\",pid=4242,fd=3))"


@pytest.mark.parametrize("addr", ["0.0.0.0", "127.0.0.1", "*", "[::]", "[::ffff:127.0.0.1]"])
def test_listeners_that_block_loopback_are_found(addr):
    assert parse_ss_listener(SS_LINE.format(addr=addr) + "\n", "127.0.0.1", 9931) == 4242


def test_listener_without_a_visible_pid_is_unknown():
    assert parse_ss_listener("LISTEN 0 4096 0.0.0.0:9931 0.0.0.0:*\n", "127.0.0.1", 9931) == "unknown"


@pytest.mark.parametrize("addr", ["192.168.1.5", "127.0.0.2", "[::1]"])
def test_listeners_on_other_addresses_do_not_block_loopback(addr):
    assert parse_ss_listener(SS_LINE.format(addr=addr) + "\n", "127.0.0.1", 9931) is None


def test_no_output_means_free():
    assert parse_ss_listener("", "127.0.0.1", 9931) is None


def test_random_token_is_43_url_safe_characters():
    token = Env().random_token(32)
    assert len(token) == 43
    assert set(token) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")
    assert Env().random_token(32) != token


def test_runner_turns_os_errors_and_timeouts_into_results(monkeypatch):
    def denied(*a, **k):
        raise PermissionError("permission denied")

    monkeypatch.setattr(subprocess, "run", denied)
    assert Runner().run(["x"]).code == 126

    def slow(*a, **k):
        raise subprocess.TimeoutExpired(cmd="x", timeout=10)

    monkeypatch.setattr(subprocess, "run", slow)
    result = Runner().run(["x"], timeout=10)
    assert result.code == 124 and "timed out" in result.stderr


class FakeResponse(io.BytesIO):
    def __init__(self, data, status, headers, url):
        super().__init__(data)
        self.status, self.headers, self._url = status, headers, url

    def geturl(self):
        return self._url


def test_http_sends_range_and_returns_the_body(monkeypatch):
    http = Http()
    seen = {}

    def open_(req, timeout):
        seen["range"] = req.get_header("Range")
        return FakeResponse(b"abc", 206, {"Content-Range": "bytes 10-12/13"}, "https://cdn-lfs.hf.co/f")

    monkeypatch.setattr(http.opener, "open", open_)
    resp = http.get("https://huggingface.co/f", range_start=10)
    assert seen["range"] == "bytes=10-"
    assert resp.status == 206 and resp.headers["content-range"] == "bytes 10-12/13"
    assert b"".join(resp.body) == b"abc"


@pytest.mark.parametrize("code", [302, 404, 416, 503])
def test_http_maps_error_statuses_to_responses(monkeypatch, code):
    http = Http()

    def open_(req, timeout):
        raise urllib.error.HTTPError(req.full_url, code, "x", {}, None)

    monkeypatch.setattr(http.opener, "open", open_)
    resp = http.get("https://huggingface.co/f")
    assert resp.status == code and list(resp.body) == []


def test_http_refuses_a_final_url_off_the_allowed_hosts(monkeypatch):
    http = Http()
    monkeypatch.setattr(http.opener, "open", lambda req, timeout: FakeResponse(b"", 200, {}, "https://example.com/f"))
    with pytest.raises(system.RedirectRefused):
        http.get("https://huggingface.co/f")
