"""Local profile JSON: read it, and write tuned extraArgs back safely."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from . import argmap
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


def plan_write(best_args: list[str], launch_args: list[str] | None) -> tuple[list[str] | None, str, str]:
    """What goes into the profile's extraArgs: only the per-flag difference from the launch flags (the manager re-adds its
    launch flags itself). Returns (extraArgs, "ok", "") or (None, cause, detail) when the result cannot be expressed safely:
    a launch flag that the best config removed, or a repeatable flag (--lora-scaled ...) that a flag->value map cannot hold
    (both are `launch-flag-removed`). The result is verified: launch + written must mean exactly the tested args."""
    launch = list(launch_args or [])
    try:
        rep = argmap.repeatable_in(best_args) or argmap.repeatable_in(launch)
        if rep:
            return None, "launch-flag-removed", f"repeatable flag {', '.join(rep)} cannot be written as a per-flag difference"
        lmap, bmap = argmap.flag_map(launch), argmap.flag_map(best_args)
        removed = [f for f in lmap if f not in bmap]
        if removed:
            return None, "launch-flag-removed", f"the best config dropped launch flag(s) {', '.join(removed)}"
        seen: dict[str, str] = {}
        for f, spelling, _v in argmap.parse(best_args):
            seen[f] = spelling
        written: list[str] = []
        for f, spelling in seen.items():
            if f in lmap and lmap[f] == bmap[f]:
                continue
            written.append(spelling)
            if bmap[f] is not None:
                written.append(bmap[f])
        if not argmap.same_args(launch + written, best_args):
            return None, "verify-mismatch", "launch flags plus the written flags do not equal the tested arguments"
        return written, "ok", ""
    except ValueError as e:
        return None, "unparseable-args", str(e)


def _unique_backup(p: Path) -> tuple[Path, int]:
    base = p.name + f".bak-{stamp()}"
    for n in range(0, 1000):
        cand = p.with_name(base if n == 0 else f"{base}-{n}")
        try:
            return cand, os.open(cand, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            continue
    raise OSError("cannot find an unused backup name")


def apply_extra_args(path: str, extra_args: list[str], expect_before: list[str] | None = None) -> dict:
    """Timestamped unique backup, fsynced temp then atomic replace, mode 0600. Only extraArgs changes. Refuses with
    profile-changed when extraArgs is no longer what the run started with."""
    p = Path(path)
    with open(p) as fh:
        data = json.load(fh)
    if expect_before is not None and list(data.get("extraArgs", [])) != list(expect_before):
        return {"written": False, "extraArgs": None, "backup": None, "reason": "profile-changed"}
    backup, fd = _unique_backup(p)
    with os.fdopen(fd, "wb") as fh:
        fh.write(p.read_bytes())
        fh.flush()
        os.fsync(fh.fileno())
    os.chmod(backup, 0o600)
    data["extraArgs"] = list(extra_args)
    write_json(p, data, mode=0o600)
    return {"written": True, "extraArgs": list(extra_args), "backup": str(backup), "reason": "applied"}


def last_tune_path(profile_path: str) -> Path:
    h = hashlib.sha256(os.path.realpath(profile_path).encode()).hexdigest()[:16]
    return uio_home() / "last" / f"{h}.json"


def record_last_tune(profile_path: str, info: dict) -> None:
    write_json(last_tune_path(profile_path), {"profile": os.path.realpath(profile_path), "finishedAt": now_iso(), **info})
