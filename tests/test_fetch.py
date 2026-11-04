import errno
import os
import hashlib
import urllib.request

import pytest

from qlm import cli, fetch, pins
from qlm.system import RedirectRefused, SafeRedirectHandler, redirect_allowed
from tests.fakes import FakeEnv, FakeHttp, make_ctx, select_model, small_pins


@pytest.fixture
def pinned(monkeypatch, home):
    """The 27B files: the cases below count its main, draft and vision projector."""
    select_model(home, "qwen3.8-27b")
    return small_pins(monkeypatch, "qwen3.8-27b")


def run_fetch(home, repo, http, *args, env=None):
    ctx = make_ctx(home, repo, http=http, env=env)
    code = cli.main(["fetch", *args], ctx)
    return code, ctx


def models(home):
    d = home / ".local/share/qlm/models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_a_fresh_home_fetches_the_35b_files(home, repo, monkeypatch):
    files, contents, urls = small_pins(monkeypatch, "qwen3.6-35b-a3b")
    http = FakeHttp(urls)

    code, ctx = run_fetch(home, repo, http)

    assert code == 0, ctx.err.getvalue()
    assert sorted(r["url"] for r in http.requests) == sorted(pins.download_url(f) for f in files)
    for f in files:
        assert (models(home) / f.local_name).read_bytes() == contents[f.local_name]


def test_the_35b_files_come_from_their_own_pinned_repos_and_revisions(home, repo, monkeypatch):
    files, contents, urls = small_pins(monkeypatch, "qwen3.6-35b-a3b")
    http = FakeHttp(urls)

    code, ctx = run_fetch(home, repo, http)

    assert code == 0, ctx.err.getvalue()
    assert [r["url"] for r in http.requests] == [
        "https://huggingface.co/mradermacher/Qwen3.6-35B-A3B-GGUF/resolve/"
        "ac96549d2e8b902f1a772083a33d13b5ee01fe3c/Qwen3.6-35B-A3B.Q4_K_M.gguf",
        "https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-GGUF/resolve/"
        "7878dc9a905c7c9379e5e8d6a1aa9affdd6839c5/mtp-Qwen3.6-35B-A3B-Q4_0.gguf",
    ]
    assert ctx.out.getvalue().count("downloaded and verified") == 2


def test_a_wrong_35b_download_is_removed_and_reported(home, repo, monkeypatch):
    files, contents, urls = small_pins(monkeypatch, "qwen3.6-35b-a3b")
    mtp = files[1]
    urls[pins.download_url(mtp)] = b"x" * mtp.size

    code, ctx = run_fetch(home, repo, FakeHttp(urls))

    assert code == 1
    assert f"{mtp.local_name}: the download doesn't match the pinned size and SHA-256" in ctx.err.getvalue()
    assert not (models(home) / mtp.local_name).exists()
    assert not (models(home) / (mtp.local_name + ".part")).exists()
    assert (models(home) / files[0].local_name).read_bytes() == contents[files[0].local_name]


def test_sha256sums_keeps_the_27b_lines_after_a_35b_fetch(home, repo, monkeypatch):
    qwen35, contents, urls = small_pins(monkeypatch, "qwen3.6-35b-a3b")
    qwen27 = pins.model("qwen3.8-27b").files[:2]
    for f in qwen27:
        (models(home) / f.local_name).write_bytes(contents[f.local_name])

    assert run_fetch(home, repo, FakeHttp(urls))[0] == 0

    sums = (models(home) / "SHA256SUMS").read_text().splitlines()
    assert sorted(sums) == sorted(f"{f.sha256}  {f.local_name}" for f in (*qwen35, *qwen27))


def test_vision_for_a_model_without_a_projector_is_a_usage_error(home, repo, monkeypatch):
    _, _, urls = small_pins(monkeypatch, "qwen3.6-35b-a3b")
    http = FakeHttp(urls)

    code, ctx = run_fetch(home, repo, http, "--vision")

    assert code == 2
    assert "no vision files for qwen3.6-35b-a3b" in ctx.err.getvalue()
    assert http.requests == []


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
    lines = [line for line in ctx.err.getvalue().splitlines() if line.startswith(files[0].local_name) and line.endswith("%")]
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


def test_redirect_status_leaves_the_part_untouched(home, repo, pinned):
    files, contents, urls = pinned
    (models(home) / files[1].local_name).write_bytes(contents[files[1].local_name])
    part = models(home) / (files[0].local_name + ".part")
    part.write_bytes(contents[files[0].local_name][:2000])
    http = FakeHttp(urls, status=302)
    ctx = make_ctx(home, repo, http=http)

    assert cli.main(["fetch"], ctx) == 1
    assert part.read_bytes() == contents[files[0].local_name][:2000]
    assert len(http.requests) == 6


def test_partial_response_starting_elsewhere_is_not_appended(home, repo, pinned):
    files, contents, urls = pinned
    f = files[0]
    data = contents[f.local_name]
    (models(home) / files[1].local_name).write_bytes(contents[files[1].local_name])
    part = models(home) / (f.local_name + ".part")
    part.write_bytes(data[:1000])

    class WrongRange(FakeHttp):
        def get(self, url, range_start=None, timeout_s=60):
            resp = super().get(url, range_start, timeout_s)
            if resp.status == 206 and len(self.requests) == 1:
                resp.headers["content-range"] = f"bytes 0-{len(data) - 1}/{len(data)}"
            return resp

    ctx = make_ctx(home, repo, http=WrongRange(urls))
    assert cli.main(["fetch"], ctx) == 0
    assert (models(home) / f.local_name).read_bytes() == data
    assert ctx.clock.sleeps == [2]


def test_unexpected_os_error_ends_with_a_summary_line(home, repo, pinned, monkeypatch):
    files, contents, urls = pinned

    def write(fh, chunk):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(fetch, "_write", write)
    code, ctx = run_fetch(home, repo, FakeHttp(urls))
    assert code == 1
    assert ctx.err.getvalue().rstrip().splitlines()[-1].startswith("qlm: ")


def test_progress_restarts_counting_after_a_full_restart(home, repo, pinned):
    files, contents, urls = pinned
    (models(home) / files[1].local_name).write_bytes(contents[files[1].local_name])
    (models(home) / (files[0].local_name + ".part")).write_bytes(b"s" * 1000)
    http = FakeHttp(urls, ranges=False, chunk=10)
    code, ctx = run_fetch(home, repo, http, env=FakeEnv(tty=True))
    lines = [line for line in ctx.err.getvalue().splitlines() if line.startswith(files[0].local_name) and line.endswith("%")]
    assert code == 0
    assert lines[0].endswith(" 0%") and len(lines) <= 101


# --from: link a local copy after checking it against the pins


@pytest.fixture
def qwen35(monkeypatch, home, tmp_path):
    """The 35B files, selected, and a directory outside the home holding local copies of them."""
    select_model(home, "qwen3.6-35b-a3b")
    files, contents, urls = small_pins(monkeypatch, "qwen3.6-35b-a3b")
    copies = tmp_path / "ollama models"
    copies.mkdir()
    return files, contents, urls, copies


def copy_of(copies, data, name="copy.gguf"):
    path = copies / name
    path.write_bytes(data)
    return path


def tree(directory):
    """Every entry under a directory with its bytes, or None when the directory doesn't exist."""
    if not directory.exists():
        return None
    return {p.relative_to(directory): p.read_bytes() for p in sorted(directory.rglob("*")) if p.is_file()}


def test_from_links_a_matching_copy_and_downloads_only_the_rest(home, repo, qwen35, tmp_path):
    files, contents, urls, copies = qwen35
    main, mtp = files
    source = copy_of(copies, contents[main.local_name])
    (tmp_path / "via").symlink_to(copies)
    before = source.stat()
    http = FakeHttp(urls)

    code, ctx = run_fetch(home, repo, http, "--from", str(tmp_path / "via" / "copy.gguf"))

    assert code == 0, ctx.err.getvalue()
    link = home / ".local/share/qlm/models" / main.local_name
    assert link.is_symlink() and os.readlink(link) == str(source)
    assert (source.stat().st_ino, source.stat().st_size) == (before.st_ino, before.st_size)
    assert [r["url"] for r in http.requests] == [pins.download_url(mtp)]
    assert f"{main.local_name}: linked to {source} and verified" in ctx.out.getvalue()


@pytest.mark.parametrize("change", ["same size, wrong hash", "wrong size"])
@pytest.mark.parametrize("models_dir_exists", [False, True])
def test_from_refuses_a_copy_that_matches_no_pin_and_writes_nothing(home, repo, qwen35, change, models_dir_exists):
    files, contents, urls, copies = qwen35
    data = contents[files[0].local_name]
    data = b"x" * len(data) if change == "same size, wrong hash" else data + b"x"
    source = copy_of(copies, data)
    models_dir = home / ".local/share/qlm/models"
    if models_dir_exists:
        models_dir.mkdir(parents=True)
        (models_dir / "SHA256SUMS").write_text("earlier\n")
    before = tree(models_dir)
    http = FakeHttp(urls)

    code, ctx = run_fetch(home, repo, http, "--from", str(source))

    assert code == 1
    assert f"{source} doesn't match any pinned file of qwen3.6-35b-a3b" in ctx.err.getvalue()
    assert tree(models_dir) == before
    assert http.requests == []


def test_from_refuses_a_file_of_the_other_model(home, repo, qwen35):
    files, contents, urls, copies = qwen35
    other = pins.model("qwen3.8-27b").files[0]
    source = copy_of(copies, contents[other.local_name])

    code, ctx = run_fetch(home, repo, FakeHttp(urls), "--from", str(source))

    assert code == 1
    assert "doesn't match any pinned file of qwen3.6-35b-a3b" in ctx.err.getvalue()
    assert not (home / ".local/share/qlm/models").exists()


def test_from_refuses_to_replace_a_different_existing_file(home, repo, qwen35):
    files, contents, urls, copies = qwen35
    main = files[0]
    source = copy_of(copies, contents[main.local_name])
    existing = models(home) / main.local_name
    existing.write_bytes(b"something else")

    code, ctx = run_fetch(home, repo, FakeHttp(urls), "--from", str(source))

    assert code == 1
    assert f"{existing} already exists" in ctx.err.getvalue()
    assert not existing.is_symlink() and existing.read_bytes() == b"something else"


def test_from_again_with_the_same_copy_changes_nothing(home, repo, qwen35):
    files, contents, urls, copies = qwen35
    source = copy_of(copies, contents[files[0].local_name])
    assert run_fetch(home, repo, FakeHttp(urls), "--from", str(source))[0] == 0
    link = models(home) / files[0].local_name

    http = FakeHttp(urls)
    code, ctx = run_fetch(home, repo, http, "--from", str(source))

    assert code == 0, ctx.err.getvalue()
    assert link.is_symlink() and os.readlink(link) == str(source)
    assert http.requests == []


def test_a_broken_symlink_is_reported_and_not_downloaded_over(home, repo, qwen35):
    files, contents, urls, copies = qwen35
    main = files[0]
    gone = copies / "moved-away.gguf"
    link = models(home) / main.local_name
    link.symlink_to(gone)
    http = FakeHttp(urls)

    code, ctx = run_fetch(home, repo, http)

    assert code == 1
    assert f"{main.local_name}: symlink to {gone} is broken; remove it and run qlm fetch" in ctx.err.getvalue()
    assert link.is_symlink() and os.readlink(link) == str(gone)
    assert pins.download_url(main) not in [r["url"] for r in http.requests]
