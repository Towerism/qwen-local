import pathlib
import socket
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def no_real_io(request, monkeypatch):
    """Fail any test that opens a network connection or starts a real process."""
    if request.node.get_closest_marker("machine"):
        return

    def refuse(*args, **kwargs):
        raise AssertionError("tests must not use the network or real subprocesses")

    monkeypatch.setattr(subprocess, "Popen", refuse)
    if not request.node.get_closest_marker("loopback"):
        monkeypatch.setattr(socket.socket, "connect", refuse)
        return

    real_connect = socket.socket.connect

    def loopback_only(sock, address):
        if not (isinstance(address, tuple) and address[0] == "127.0.0.1"):
            refuse()
        return real_connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", loopback_only)


@pytest.fixture
def repo():
    return REPO


@pytest.fixture
def home(tmp_path):
    h = tmp_path / "home with space"
    h.mkdir()
    return h
