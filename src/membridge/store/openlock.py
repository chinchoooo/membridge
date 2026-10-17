"""Is this database open in another process?

`membridge lock` encrypts the database and deletes the plaintext. If a dashboard or MCP server still has the file open,
that is a silent data-loss bug on POSIX (the server keeps writing to an unlinked inode) and a confusing PermissionError on
Windows. Every `Store` therefore holds a marker lock beside the database for as long as it is open, and `lock` refuses to
proceed while any marker is held.

  POSIX    flock(): shared by every open Store, exclusive for the probe.
  Windows  msvcrt byte-range locks (they cannot be shared): each Store locks one byte at a random offset in a 1 MiB window;
           the probe tries to lock the whole window, which fails if any single byte is held.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

WINDOW = 1 << 20


def marker_path(db_path: Path) -> Path:
    return db_path.with_suffix(db_path.suffix + ".open")


class OpenMarker:
    """Held for the lifetime of one Store. Best effort: failure to take it never stops a Store from opening."""

    def __init__(self, db_path: Path) -> None:
        self.fd: int | None = None
        self.offset = 0
        try:
            self.fd = os.open(marker_path(db_path), os.O_RDWR | os.O_CREAT)
            if sys.platform == "win32":
                import msvcrt

                for _ in range(32):
                    self.offset = random.randrange(0, WINDOW)
                    os.lseek(self.fd, self.offset, os.SEEK_SET)
                    try:
                        msvcrt.locking(self.fd, msvcrt.LK_NBLCK, 1)  # type: ignore[attr-defined,unused-ignore]
                        return
                    except OSError:
                        continue
                self._close()
            else:
                import fcntl

                fcntl.flock(self.fd, fcntl.LOCK_SH)
        except OSError:
            self._close()

    def _close(self) -> None:
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = None

    def release(self) -> None:
        if self.fd is None:
            return
        try:
            if sys.platform == "win32":
                import msvcrt

                os.lseek(self.fd, self.offset, os.SEEK_SET)
                msvcrt.locking(self.fd, msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined,unused-ignore]
        except OSError:
            pass
        self._close()


def held_by_anyone(db_path: Path) -> bool:
    """True if any Store (in this or another process) currently has the database open."""
    mp = marker_path(db_path)
    if not mp.exists():
        return False
    try:
        fd = os.open(mp, os.O_RDWR)
    except OSError:
        return True
    try:
        if sys.platform == "win32":
            import msvcrt

            os.lseek(fd, 0, os.SEEK_SET)
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, WINDOW)  # type: ignore[attr-defined,unused-ignore]
            except OSError:
                return True
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, WINDOW)  # type: ignore[attr-defined,unused-ignore]
            return False
        import fcntl

        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)
