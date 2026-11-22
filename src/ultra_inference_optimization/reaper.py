"""Remember the server process we started (PID + start time) so a run killed hard cannot leave a model server behind:
the next run finds the record and stops that exact process, never one matched by name."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

from .util import log, uio_home, write_json


def start_time(pid: int) -> str | None:
    """The process start time as ps prints it, or None when the PID is gone. The pair (pid, start time) identifies a process
    even after the OS reuses the PID."""
    try:
        out = subprocess.run(["ps", "-o", "stat=,lstart=", "-p", str(int(pid))], capture_output=True, text=True,
                             timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    if not out:
        return None
    stat, _, started = out.partition(" ")
    return None if stat.startswith("Z") else (started.strip() or None)  # a zombie is already dead


def global_marker() -> Path:
    return uio_home() / "child.json"


def record(pid: int, workdir: Path | None = None) -> None:
    st = start_time(pid)
    if st is None:
        return
    info = {"pid": pid, "startTime": st, "uioPid": os.getpid()}
    write_json(global_marker(), info)
    if workdir is not None:
        write_json(Path(workdir) / "child.json", info)


def clear(workdir: Path | None = None) -> None:
    for p in [global_marker()] + ([Path(workdir) / "child.json"] if workdir is not None else []):
        try:
            p.unlink()
        except OSError:
            pass


def reap_stale(grace: float = 10.0) -> str | None:
    """Stop the recorded child if it is still the same process. Returns what happened, None when nothing was recorded."""
    path = global_marker()
    try:
        info = json.loads(path.read_text())
        pid, recorded = int(info["pid"]), str(info["startTime"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    result = "gone"
    if pid != os.getpid() and start_time(pid) == recorded:
        log(f"stopping a server left over from an earlier run (pid {pid})")
        result = "terminated"
        try:
            os.kill(pid, signal.SIGTERM)
            end = time.monotonic() + grace
            while time.monotonic() < end and start_time(pid) == recorded:
                time.sleep(0.1)
            if start_time(pid) == recorded:
                os.kill(pid, signal.SIGKILL)
                result = "killed"
        except ProcessLookupError:
            pass
        except PermissionError:
            result = "not-permitted"
    try:
        path.unlink()
    except OSError:
        pass
    return result
