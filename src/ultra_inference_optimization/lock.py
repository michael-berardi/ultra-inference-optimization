"""Exclusive inter-process lock so only one uio tune holds the machine at a time."""
from __future__ import annotations

import fcntl
import os

from .errors import Refused
from .util import uio_home


class RunLock:
    def __init__(self, path=None):
        self.path = path or (uio_home() / "lock")
        self._fh = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+")
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.seek(0)
            holder = fh.read().strip() or "unknown pid"
            fh.close()
            raise Refused("lock-held", f"another uio run holds {self.path} (holder pid {holder})")
        fh.seek(0)
        fh.truncate()
        fh.write(str(os.getpid()))
        fh.flush()
        self._fh = fh
        return self

    def __exit__(self, *exc):
        if self._fh:
            try:
                fcntl.flock(self._fh, fcntl.LOCK_UN)
            finally:
                self._fh.close()
                self._fh = None
        return False
