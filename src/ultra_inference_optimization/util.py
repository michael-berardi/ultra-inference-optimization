"""Small shared helpers: paths, durations, time stamps, logging to stderr."""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
import sys
from pathlib import Path

from .errors import Refused


def uio_home() -> Path:
    env = os.environ.get("UIO_HOME")
    return Path(env) if env else Path.home() / ".cache" / "uio"


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


def write_json(path: Path, obj, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode if mode is not None else 0o644)
    with os.fdopen(fd, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=False)
        fh.write("\n")
    if mode is not None:
        os.chmod(tmp, mode)
    os.replace(tmp, path)


def append_jsonl(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(obj) + "\n")


def append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(text)
