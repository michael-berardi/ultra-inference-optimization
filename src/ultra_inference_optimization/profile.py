"""Local profile JSON: read it, and write tuned extraArgs back safely."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from .errors import Refused
from .util import now_iso, stamp, uio_home, write_json


def load_profile(path: str) -> dict:
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (OSError, ValueError) as e:
        raise Refused("bad-profile", f"cannot read profile {path}: {e}")
    if not isinstance((data.get("source") or {}).get("path"), str):
        raise Refused("bad-profile", "profile needs source.path (the model file)")
    if not isinstance(data.get("ctx"), int) or data["ctx"] <= 0:
        raise Refused("bad-profile", "profile needs an integer ctx")
    extra = data.get("extraArgs", [])
    if not isinstance(extra, list) or not all(isinstance(x, str) for x in extra):
        raise Refused("bad-profile", "profile extraArgs must be a list of strings")
    return data


def extra_args_to_write(best_args: list[str], launch_args: list[str] | None) -> list[str]:
    """What goes back into extraArgs. The manager re-adds its launch flags itself and llama-server keeps the last value of
    a repeated flag, so an unchanged launch prefix is dropped and anything else is written whole."""
    launch = list(launch_args or [])
    if launch and best_args[:len(launch)] == launch:
        return list(best_args[len(launch):])
    return list(best_args)


def apply_extra_args(path: str, best_args: list[str]) -> dict:
    """Timestamped backup, atomic replace, mode 0600. Only extraArgs changes."""
    p = Path(path)
    with open(p) as fh:
        data = json.load(fh)
    backup = p.with_name(p.name + f".bak-{stamp()}")
    fd = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(p.read_bytes())
    os.chmod(backup, 0o600)
    data["extraArgs"] = list(best_args)
    write_json(p, data, mode=0o600)
    return {"written": str(p), "backup": str(backup)}


def last_tune_path(profile_path: str) -> Path:
    h = hashlib.sha256(os.path.realpath(profile_path).encode()).hexdigest()[:16]
    return uio_home() / "last" / f"{h}.json"


def record_last_tune(profile_path: str, info: dict) -> None:
    write_json(last_tune_path(profile_path), {"profile": os.path.realpath(profile_path), "finishedAt": now_iso(), **info})
