"""`uio probe`: what machine is this, how fast is its memory, which engines are installed."""
from __future__ import annotations

import os
import re
import shutil

from . import sysinfo


def find_llama_server(explicit: str | None = None) -> str | None:
    cand = explicit or os.environ.get("UIO_LLAMA_SERVER")
    if cand:
        return cand if os.path.isfile(cand) and os.access(cand, os.X_OK) else None
    return shutil.which("llama-server")


def llama_version(path: str) -> str | None:
    out = sysinfo.run([path, "--version"], timeout=20)
    m = re.search(r"version:\s*(.+)", out)
    return m.group(1).strip() if m else (out.strip().splitlines()[0] if out.strip() else None)


def probe(measure: bool = True, llama_server: str | None = None) -> dict:
    eng = {}
    path = find_llama_server(llama_server)
    eng["llama-server"] = {"path": path, "version": llama_version(path) if path else None}
    for name in ("mlx_lm.server", "vllm"):
        p = shutil.which(name)
        eng[name] = {"path": p, "version": None, "supported": False}
    return {"machine": sysinfo.machine(),
            "measured": sysinfo.measure_bandwidth() if measure else {"memBandwidthGBs": None, "method": "skipped"},
            "engines": eng}
