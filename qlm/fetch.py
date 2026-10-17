"""`qlm fetch`: download the pinned model files with resume, and verify them against ADR 0002.

It never deletes or overwrites a final model file; a corrupt one is reported for the user to remove.
"""

import errno
import hashlib
import http.client
import os

from qlm import pins
from qlm.system import RedirectRefused

GIB = 2**30
HASH_CHUNK = 8 * 2**20
BACKOFF = (2, 4, 8, 16, 32)
TIMEOUT_S = 60


class _FileFailed(Exception):
    pass


def _open_for_hash(path):
    return open(path, "rb")


def sha256_of(path):
    h = hashlib.sha256()
    with _open_for_hash(path) as fh:
        while True:
            chunk = fh.read(HASH_CHUNK)
            if not chunk:
                return h.hexdigest()
            h.update(chunk)


def _write(fh, chunk):
    fh.write(chunk)


def run(ctx, vision=False):
    models = ctx.paths.models
    models.mkdir(parents=True, exist_ok=True)
    wanted = pins.files(vision)
    failed, pending = [], []

    for f in wanted:
        final = models / f.local_name
        if not final.exists():
            pending.append(f)
            continue
        if final.stat().st_size == f.size and sha256_of(final) == f.sha256:
            ctx.say(f"{f.local_name}: already present and verified")
        else:
            ctx.warn(f"{f.local_name}: exists but doesn't match the pinned size and SHA-256; delete {final} and run qlm fetch again")
            failed.append(f)

    if pending:
        need = sum(_bytes_still_needed(models, f) for f in pending) + GIB
        free = ctx.env.disk_free(models)
        if free < need:
            return ctx.fail(
                f"not enough disk space in {models}: {free / GIB:.1f} GiB free, {need / GIB:.1f} GiB needed",
                "Free some space and run qlm fetch again",
            )

    for f in pending:
        try:
            _download(ctx, f)
            ctx.say(f"{f.local_name}: downloaded and verified")
        except _FileFailed as e:
            ctx.warn(f"{f.local_name}: {e}")
            failed.append(f)

    if failed:
        names = ", ".join(f.local_name for f in failed)
        return ctx.fail(f"{len(failed)} file(s) failed: {names}", "See the messages above, then run qlm fetch again")
    _write_sums(models, wanted)
    return 0


def _bytes_still_needed(models, f):
    part = models / (f.local_name + ".part")
    have = part.stat().st_size if part.exists() else 0
    return f.size if have > f.size else f.size - have


def _download(ctx, f):
    part = ctx.paths.models / (f.local_name + ".part")
    url = pins.download_url(f)
    if part.exists() and part.stat().st_size > f.size:
        part.unlink()
    streak = 0
    progress = _Progress(ctx, f)
    while True:
        offset = part.stat().st_size if part.exists() else 0
        if offset == f.size:
            return _verify_and_place(ctx, f, part)
        got_bytes, error = _attempt(ctx, f, url, part, offset, progress)
        if error is None:
            size = part.stat().st_size if part.exists() else 0
            if size == f.size:
                return _verify_and_place(ctx, f, part)
            if size > f.size:
                part.unlink()
                raise _FileFailed(f"the server sent more than the pinned {f.size} bytes; the partial file was removed")
            error = "the connection ended early"
        streak = 1 if got_bytes else streak + 1
        if streak > len(BACKOFF):
            raise _FileFailed(f"giving up after {len(BACKOFF)} retries without progress ({error}); run qlm fetch again to resume")
        wait = BACKOFF[streak - 1]
        ctx.warn(f"{f.local_name}: {error}; retrying in {wait} s")
        ctx.clock.sleep(wait)


def _attempt(ctx, f, url, part, offset, progress):
    """One GET. Returns (received any bytes, retryable error or None). Raises _FileFailed for final errors."""
    try:
        resp = ctx.http.get(url, range_start=offset or None, timeout_s=TIMEOUT_S)
    except RedirectRefused as e:
        raise _FileFailed(str(e)) from None
    except (OSError, http.client.HTTPException) as e:
        return False, f"network error: {e}"

    if resp.status == 416:
        return False, None
    if resp.status in (404, 403) or 400 <= resp.status < 500:
        raise _FileFailed(f"HTTP {resp.status} from {url}")
    if resp.status >= 500:
        return False, f"HTTP {resp.status} from the server"
    mode = "ab" if resp.status == 206 and offset else "wb"

    got = False
    with open(part, mode) as fh:
        body = iter(resp.body)
        while True:
            try:
                chunk = next(body)
            except StopIteration:
                return got, None
            except (OSError, http.client.HTTPException) as e:
                return got, f"network error: {e}"
            try:
                _write(fh, chunk)
            except OSError as e:
                if e.errno == errno.ENOSPC:
                    raise _FileFailed("the disk is full; the partial file was kept so the next run can resume") from None
                raise
            got = got or bool(chunk)
            progress.update(fh.tell())


class _Progress:
    def __init__(self, ctx, f):
        self.ctx, self.f, self.last = ctx, f, None
        self.enabled = ctx.env.is_tty("stderr")

    def update(self, done):
        if not self.enabled:
            return
        pct = min(100, done * 100 // self.f.size)
        if pct != self.last:
            self.last = pct
            self.ctx.warn(f"{self.f.local_name}: {pct}%")


def _verify_and_place(ctx, f, part):
    if part.stat().st_size != f.size or sha256_of(part) != f.sha256:
        part.unlink()
        raise _FileFailed("the download doesn't match the pinned size and SHA-256; the partial file was removed")
    os.replace(part, ctx.paths.models / f.local_name)


def _write_sums(models, files):
    tmp = models / ".SHA256SUMS.tmp"
    tmp.write_text("".join(f"{f.sha256}  {f.local_name}\n" for f in files))
    os.replace(tmp, models / "SHA256SUMS")
