"""Small shared helpers: paths, durations, time stamps, logging to stderr."""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
import shutil
import sys
from pathlib import Path

from .errors import Refused


# Only the ten most recent default run folders are kept; older benchmark logs are deleted on the next run.
RUN_RETENTION = 10


def uio_home() -> Path:
    env = os.environ.get("UIO_HOME")
    return Path(env) if env else Path.home() / ".cache" / "uio"


def private_mkdir(path: Path) -> None:
    """Create (or tighten) a directory that holds local run data: owner read/write/execute only.

    Directories below UIO_HOME are tightened too (parents created by mkdir are 0755 by default);
    ancestors above UIO_HOME, and the parents of an explicit --workdir, are left alone.
    """
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    target = Path(os.path.abspath(path))
    base = Path(os.path.abspath(uio_home()))
    os.chmod(target, 0o700)
    if target != base and base in target.parents:
        parent = target.parent
        while parent != base:
            os.chmod(parent, 0o700)
            parent = parent.parent
        os.chmod(base, 0o700)


def prune_runs(home: Path | None = None, keep: int = RUN_RETENTION) -> list[Path]:
    """Bound retained logs: delete the oldest folders under <home>/runs, never a caller's --workdir."""
    root = (home or uio_home()) / "runs"
    if not root.is_dir():
        return []
    runs = sorted((p for p in root.iterdir() if p.is_dir() and not p.is_symlink()),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    removed = []
    for old in runs[keep:]:
        shutil.rmtree(old, ignore_errors=True)
        removed.append(old)
    return removed


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def log(msg: str) -> None:
    print(f"[uio] {msg}", file=sys.stderr, flush=True)


_DUR = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smhd]?)\s*$")


def parse_duration(text: str) -> float:
    m = _DUR.match(str(text))
    if not m:
        raise Refused("bad-duration", f"cannot parse duration {text!r} (examples: 90s, 45m, 2h)")
    unit = {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]
    return float(m.group(1)) * unit


def write_json(path: Path, obj, mode: int = 0o600) -> None:
    private_mkdir(path.parent)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    with os.fdopen(fd, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=False)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())  # the bytes are on disk before the rename makes them the file
    os.chmod(tmp, mode)
    os.replace(tmp, path)


def _append_fd(path: Path) -> int:
    private_mkdir(path.parent)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    os.fchmod(fd, 0o600)  # also tightens a file left by an older version
    return fd


def append_jsonl(path: Path, obj) -> None:
    with os.fdopen(_append_fd(path), "a") as fh:
        fh.write(json.dumps(obj) + "\n")


def append_text(path: Path, text: str) -> None:
    with os.fdopen(_append_fd(path), "a") as fh:
        fh.write(text)
