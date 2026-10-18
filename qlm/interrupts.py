"""Ctrl-C and SIGTERM handling for the commands that must put things back (qlm bench, qlm update)."""

import contextlib
import signal


def raise_interrupt(signum, frame):
    """The SIGTERM handler: treat it like Ctrl-C, so the same cleanup runs."""
    raise KeyboardInterrupt


@contextlib.contextmanager
def sigterm_as_interrupt():
    old = signal.signal(signal.SIGTERM, raise_interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, old)


@contextlib.contextmanager
def deferred():
    """Ignore Ctrl-C and SIGTERM while a restore runs, so a second Ctrl-C can't stop it halfway."""
    old_int = signal.signal(signal.SIGINT, signal.SIG_IGN)
    old_term = signal.signal(signal.SIGTERM, signal.SIG_IGN)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)
