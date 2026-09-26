"""Who owns Ruth's mind right now.

Every entry point used to open her independently:

    state = os.path.join(home, "brain.npz")
    Mind(Brain.load_or_create(state, cfg), home=home)

That gave the app one full copy in memory, the CLI another loaded from disk,
and the MCP bridge a third. Each saved over the others, so whichever process
exited last destroyed the rest. Measured on this box: the app held 7,262
moments in memory while the CLI had written a 689-moment copy to the same
file, ready to be overwritten in turn. Experience was being destroyed on a
timer.

A mind is not a file that several programs may rewrite. One owner at a time:

    with owner(home):        # the app, which holds her for as long as it runs
        ...load, live, save...

    with owner(home, write=False):   # a reader, allowed to look
        ...

and a writer that cannot get the lock is refused rather than allowed to
clobber. The lock is an advisory fcntl.flock on a file in her home (a
msvcrt byte lock on Windows, which has no fcntl), so it is released
automatically if the owner is killed, and it works across processes without a
daemon.
"""
from __future__ import annotations

import errno
import os
from contextlib import contextmanager

try:
    import fcntl
except ImportError:          # Windows: importing fcntl here used to stop her
    fcntl = None             # from starting at all on that system
    import msvcrt

LOCK_NAME = ".ruth-owner.lock"
# Windows byte locks are mandatory: a locked byte cannot even be read. The
# lock therefore sits far past the recorded owner line, so who_owns can still
# read who holds her.
WIN_LOCK_OFFSET = 1 << 30


class MindBusy(Exception):
    """Another process owns her mind; writing now would destroy its state."""


def lock_path(home: str) -> str:
    return os.path.join(home, LOCK_NAME)


def who_owns(home: str) -> str | None:
    """Best-effort description of the current owner, or None if free.

    Advisory locks do not carry a name, so this reports the pid recorded in
    the lock file. Stale entries (a killed owner) are reported as free,
    because the lock itself is what actually gates access.
    """
    path = lock_path(home)
    try:
        with open(path, encoding="utf-8") as fh:
            recorded = fh.read().strip()
    except OSError:
        return None
    if not recorded:
        return None
    pid = recorded.split()[0]
    if fcntl is None:
        # On Windows os.kill(pid, 0) is not a probe (signal 0 is CTRL_C_EVENT),
        # so ask the lock itself: if it can be taken, nobody holds her.
        fd = os.open(path, os.O_RDWR | getattr(os, "O_BINARY", 0))
        try:
            try:
                _lock(fd)
            except OSError:
                return recorded
            _unlock(fd)
            return None
        finally:
            os.close(fd)
    try:
        os.kill(int(pid), 0)
    except (ValueError, ProcessLookupError):
        return None                      # owner is gone
    except PermissionError:
        return recorded
    return recorded


def _lock(fd: int) -> None:
    if fcntl is not None:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    else:
        os.lseek(fd, WIN_LOCK_OFFSET, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)


def _unlock(fd: int) -> None:
    if fcntl is not None:
        fcntl.flock(fd, fcntl.LOCK_UN)
    else:
        os.lseek(fd, WIN_LOCK_OFFSET, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)


@contextmanager
def owner(home: str, write: bool = True, label: str = ""):
    """Take exclusive ownership of the mind in `home`.

    With write=False this still takes the lock, so a reader cannot observe a
    half-written file; it simply refuses nothing and reports failure as a
    MindBusy so the caller can decide to fall back to a read-only view.
    """
    os.makedirs(home, exist_ok=True)
    path = lock_path(home)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0), 0o600)
    try:
        try:
            _lock(fd)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN, getattr(errno, "EDEADLOCK", -1)):
                other = who_owns(home) or "another process"
                what = label or "this operation"
                raise MindBusy(
                    f"cannot {what}: {other} owns Ruth's mind right now. "
                    f"Stop it first, or use --read-only. Writing now would "
                    f"overwrite the live copy and destroy what it has learned."
                ) from None
            raise
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, f"{os.getpid()} {label or 'ruth'}\n".encode())
        os.fsync(fd)
        yield
    finally:
        try:
            _unlock(fd)
        except OSError:
            pass
        os.close(fd)


def is_free(home: str) -> bool:
    try:
        with owner(home, label="probe"):
            return True
    except MindBusy:
        return False
