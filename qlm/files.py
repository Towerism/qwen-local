"""Writing files so that no reader ever sees them half-written, and never overwriting a log or backup."""

import os
from pathlib import Path


def replace_atomically(path, data, mode):
    """Write `data` (bytes) to a temporary file beside `path`, flush it to disk, then rename it over `path`."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if os.path.lexists(tmp):
        tmp.unlink()
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, mode)  # the umask may have narrowed the mode given to open
        os.replace(tmp, path)
    except BaseException:
        if os.path.lexists(tmp):
            tmp.unlink()
        raise


def create_new(directory, name_for, data=b"", mode=0o600):
    """Create a file that didn't exist, named name_for(0), or name_for(1), name_for(2)... on a clash."""
    directory = Path(directory)
    for n in range(1000):
        path = directory / name_for(n)
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        except FileExistsError:
            continue
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())  # a backup must be on disk before the file it backs up is replaced
        return path
    raise FileExistsError(f"no free file name in {directory}")


def stamped(stamp, n):
    """`stamp` with a `-n` suffix after the first clash, so names still sort oldest first."""
    return stamp if n == 0 else f"{stamp}-{n}"
