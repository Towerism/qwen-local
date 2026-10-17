import errno
import hashlib
import urllib.request

import pytest

from qlm import cli, fetch, pins
from qlm.system import RedirectRefused, SafeRedirectHandler, redirect_allowed
from tests.fakes import FakeEnv, FakeHttp, make_ctx, small_pins


@pytest.fixture
def pinned(monkeypatch):
    return small_pins(monkeypatch)


def run_fetch(home, repo, http, *args, env=None):
    ctx = make_ctx(home, repo, http=http, env=env)
    code = cli.main(["fetch", *args], ctx)
    return code, ctx


def models(home):
    d = home / ".local/share/qlm/models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_present_files_make_no_requests(home, repo, pinned):
    files, contents, urls = pinned
    for f in files[:2]:
        (models(home) / f.local_name).write_bytes(contents[f.local_name])
    http = FakeHttp(urls)

    code, ctx = run_fetch(home, repo, http)

    assert code == 0
    assert http.requests == []
    assert ctx.out.getvalue().count("already present") == 2


def test_partial_download_resumes_with_a_range_request(home, repo, pinned):
    files, contents, urls = pinned
    f = files[0]
    data = contents[f.local_name]
    (models(home) / files[1].local_name).write_bytes(contents[files[1].local_name])
    (models(home) / (f.local_name + ".part")).write_bytes(data[: len(data) // 2])
    http = FakeHttp(urls)

    code, _ = run_fetch(home, repo, http)

    assert code == 0
    assert http.requests == [{"url": pins.download_url(f), "range_start": len(data) // 2}]
    assert (models(home) / f.local_name).read_bytes() == data
    assert not (models(home) / (f.local_name + ".part")).exists()


def test_server_ignoring_range_restarts_from_zero(home, repo, pinned):
    files, contents, urls = pinned
    f = files[0]
    (models(home) / files[1].local_name).write_bytes(contents[files[1].local_name])
    (models(home) / (f.local_name + ".part")).write_bytes(b"stale bytes")
    http = FakeHttp(urls, ranges=False)

    code, _ = run_fetch(home, repo, http)

    assert code == 0
    assert (models(home) / f.local_name).read_bytes() == contents[f.local_name]


def test_wrong_hash_after_download_deletes_the_part_and_fails(home, repo, pinned):
    files, contents, urls = pinned
    f = files[0]
    (models(home) / files[1].local_name).write_bytes(contents[files[1].local_name])
    urls[pins.download_url(f)] = b"x" * f.size
    code, ctx = run_fetch(home, repo, FakeHttp(urls))

    assert code == 1
    assert not (models(home) / (f.local_name + ".part")).exists()
    assert not (models(home) / f.local_name).exists()
    assert "SHA-256" in ctx.err.getvalue()


def test_corrupt_final_file_is_reported_not_deleted(home, repo, pinned):
    files, contents, urls = pinned
    bad = b"y" * files[0].size
    (models(home) / files[0].local_name).write_bytes(bad)
    (models(home) / files[1].local_name).write_bytes(contents[files[1].local_name])

    code, ctx = run_fetch(home, repo, FakeHttp(urls))

    assert code == 1
    assert (models(home) / files[0].local_name).read_bytes() == bad
    assert "delete" in ctx.err.getvalue() and files[0].local_name in ctx.err.getvalue()


def test_dropped_connections_without_progress_back_off_2_then_4_seconds(home, repo, pinned):
    files, contents, urls = pinned
    f = files[0]
    data = contents[f.local_name]
    (models(home) / files[1].local_name).write_bytes(contents[files[1].local_name])
    (models(home) / (f.local_name + ".part")).write_bytes(data[:1000])
    http = FakeHttp(urls, drops=[0, 0])
    ctx = make_ctx(home, repo, http=http)

    code = cli.main(["fetch"], ctx)

    assert code == 0
    assert ctx.clock.sleeps == [2, 4]
    assert [r["range_start"] for r in http.requests] == [1000, 1000, 1000]
    assert (models(home) / f.local_name).read_bytes() == data


def test_retry_count_resets_after_bytes_arrive(home, repo, pinned):
    files, contents, urls = pinned
    (models(home) / files[1].local_name).write_bytes(contents[files[1].local_name])
    http = FakeHttp(urls, drops=[500, 500])
    ctx = make_ctx(home, repo, http=http)

    assert cli.main(["fetch"], ctx) == 0
    assert ctx.clock.sleeps == [2, 2]
    assert [r["range_start"] for r in http.requests] == [None, 500, 1000]


def test_gives_up_after_five_retries_without_progress_and_keeps_the_part(home, repo, pinned):
    files, contents, urls = pinned
    f = files[0]
    (models(home) / files[1].local_name).write_bytes(contents[files[1].local_name])
    (models(home) / (f.local_name + ".part")).write_bytes(contents[f.local_name][:100])
    http = FakeHttp(urls, drops=[0] * 6)
    ctx = make_ctx(home, repo, http=http)

    assert cli.main(["fetch"], ctx) == 1
    assert ctx.clock.sleeps == [2, 4, 8, 16, 32]
    assert (models(home) / (f.local_name + ".part")).stat().st_size == 100


def test_server_errors_are_retried(home, repo, pinned):
    files, contents, urls = pinned
    (models(home) / files[1].local_name).write_bytes(contents[files[1].local_name])

    class FlakyHttp(FakeHttp):
        def get(self, url, range_start=None, timeout_s=60):
            if not self.requests:
                self.requests.append({"url": url, "range_start": range_start})
                from qlm.system import HttpResponse
                return HttpResponse(503, {}, iter(()))
            return super().get(url, range_start, timeout_s)

    ctx = make_ctx(home, repo, http=FlakyHttp(urls))
    assert cli.main(["fetch"], ctx) == 0
    assert ctx.clock.sleeps == [2]


@pytest.mark.parametrize("status", [404, 403])
def test_missing_revision_fails_without_retrying(home, repo, pinned, status):
    files, contents, urls = pinned
    (models(home) / files[1].local_name).write_bytes(contents[files[1].local_name])
    http = FakeHttp(urls, status=status)
    ctx = make_ctx(home, repo, http=http)

    assert cli.main(["fetch"], ctx) == 1
    assert len(http.requests) == 1 and ctx.clock.sleeps == []
    assert str(status) in ctx.err.getvalue() and pins.download_url(files[0]) in ctx.err.getvalue()


def test_low_disk_space_fails_before_any_request(home, repo, pinned):
    files, contents, urls = pinned
    http = FakeHttp(urls)
    need = files[0].size + files[1].size + 2**30
    env = FakeEnv(disk_free_bytes=need - 1)

    code, ctx = run_fetch(home, repo, http, env=env)

    assert code == 1
    assert http.requests == []
    assert "disk" in ctx.err.getvalue().lower()


def test_vision_adds_the_projector_only_when_asked(home, repo, pinned):
    files, contents, urls = pinned
    http = FakeHttp(urls)
    assert run_fetch(home, repo, http)[0] == 0
    assert pins.download_url(files[2]) not in [r["url"] for r in http.requests]

    assert run_fetch(home, repo, http, "--vision")[0] == 0
    assert http.requests[-1]["url"] == pins.download_url(files[2])
    assert (models(home) / files[2].local_name).read_bytes() == contents[files[2].local_name]


def test_progress_only_on_a_terminal_and_one_failure_does_not_block_the_other(home, repo, pinned):
    files, contents, urls = pinned
    urls[pins.download_url(files[1])] = b"z" * files[1].size
    code, ctx = run_fetch(home, repo, FakeHttp(urls, chunk=10), env=FakeEnv(tty=False))
    assert code == 1
    assert "%" not in ctx.err.getvalue()
    assert (models(home) / files[0].local_name).exists()

    (models(home) / files[0].local_name).unlink()
    code, ctx = run_fetch(home, repo, FakeHttp(urls, chunk=10), env=FakeEnv(tty=True))
    lines = [l for l in ctx.err.getvalue().splitlines() if l.startswith(files[0].local_name) and l.endswith("%")]
    assert 1 <= len(lines) <= 101


def test_complete_part_is_verified_without_a_request(home, repo, pinned):
    files, contents, urls = pinned
    for f in files[:2]:
        (models(home) / (f.local_name + ".part")).write_bytes(contents[f.local_name])
    http = FakeHttp(urls)

    assert run_fetch(home, repo, http)[0] == 0
    assert http.requests == []
    assert (models(home) / files[0].local_name).exists()


def test_oversized_part_is_discarded_and_restarted(home, repo, pinned):
    files, contents, urls = pinned
    f = files[0]
    (models(home) / files[1].local_name).write_bytes(contents[files[1].local_name])
    (models(home) / (f.local_name + ".part")).write_bytes(b"q" * (f.size + 10))
    http = FakeHttp(urls)

    assert run_fetch(home, repo, http)[0] == 0
    assert http.requests[0]["range_start"] is None


def test_sha256sums_written_after_full_success_and_kept_on_failure(home, repo, pinned):
    files, contents, urls = pinned
    assert run_fetch(home, repo, FakeHttp(urls))[0] == 0
    sums = (models(home) / "SHA256SUMS").read_text()
    assert sums == "".join(f"{f.sha256}  {f.local_name}\n" for f in files[:2])

    (models(home) / files[0].local_name).write_bytes(b"broken" * 10)
    assert run_fetch(home, repo, FakeHttp(urls))[0] == 1
    assert (models(home) / "SHA256SUMS").read_text() == sums


def test_refused_redirect_fails_the_file_and_names_the_host(home, repo, pinned):
    files, contents, urls = pinned
    (models(home) / files[1].local_name).write_bytes(contents[files[1].local_name])
    http = FakeHttp(urls, raise_on_get=RedirectRefused("refused redirect to http://evil.example"))

    code, ctx = run_fetch(home, repo, http)

    assert code == 1
    assert "evil.example" in ctx.err.getvalue()
    assert not (models(home) / files[0].local_name).exists()


def test_disk_full_while_writing_keeps_the_part(home, repo, pinned, monkeypatch):
    files, contents, urls = pinned
    (models(home) / files[1].local_name).write_bytes(contents[files[1].local_name])
    calls = {"n": 0}
    real_write = fetch._write

    def write(fh, chunk):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError(errno.ENOSPC, "No space left on device")
        real_write(fh, chunk)

    monkeypatch.setattr(fetch, "_write", write)
    code, ctx = run_fetch(home, repo, FakeHttp(urls))

    assert code == 1
    assert (models(home) / (files[0].local_name + ".part")).exists()
    assert "disk" in ctx.err.getvalue().lower()


def test_server_sending_too_many_bytes_discards_the_part(home, repo, pinned):
    files, contents, urls = pinned
    (models(home) / files[1].local_name).write_bytes(contents[files[1].local_name])
    urls[pins.download_url(files[0])] = contents[files[0].local_name] + b"extra"

    code, _ = run_fetch(home, repo, FakeHttp(urls))
    assert code == 1
    assert not (models(home) / (files[0].local_name + ".part")).exists()


@pytest.mark.parametrize(
    "url, allowed",
    [
        ("https://huggingface.co/x", True),
        ("https://cas-bridge.xethub.hf.co/x", True),
        ("https://cdn-lfs.hf.co/x", True),
        ("http://cdn-lfs.hf.co/x", False),
        ("https://evilhf.co/x", False),
        ("https://huggingface.co.evil.example/x", False),
        ("https://example.com/x", False),
    ],
)
def test_redirect_policy(url, allowed):
    assert redirect_allowed(url) is allowed


def test_redirect_handler_refuses_disallowed_targets():
    handler = SafeRedirectHandler()
    req = urllib.request.Request("https://huggingface.co/x")
    with pytest.raises(RedirectRefused, match="example.com"):
        handler.redirect_request(req, None, 302, "Found", {}, "https://example.com/file")
    assert handler.redirect_request(req, None, 302, "Found", {}, "https://cdn-lfs.hf.co/file") is not None


def test_hashing_reads_in_bounded_chunks(tmp_path, monkeypatch):
    p = tmp_path / "f"
    p.write_bytes(b"a" * (20 * 2**20))
    sizes = []
    real_open = open

    class Spy:
        def __init__(self, fh):
            self.fh = fh

        def read(self, n=-1):
            sizes.append(n)
            return self.fh.read(n)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.fh.close()

    monkeypatch.setattr(fetch, "_open_for_hash", lambda path: Spy(real_open(path, "rb")))
    assert fetch.sha256_of(p) == hashlib.sha256(b"a" * (20 * 2**20)).hexdigest()
    assert sizes and max(sizes) <= 8 * 2**20 and -1 not in sizes
