"""Machine facts and environment guards (power, free memory, GPU busy, per-process memory).

Every reader is a small module-level function so tests can replace it."""
from __future__ import annotations

import glob
import os
import platform
import re
import shutil
import subprocess
import threading
import time

from .errors import Refused


def run(cmd: list[str], timeout: float = 10.0) -> str:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (p.stdout or "") + (p.stderr if not p.stdout else "")
    except (OSError, subprocess.SubprocessError):
        return ""


def sysctl(name: str) -> str:
    return run(["sysctl", "-n", name]).strip()


def is_darwin() -> bool:
    return platform.system() == "Darwin"


# ---- power -------------------------------------------------------------------------
def read_power() -> dict:
    """{"powerSource": ac|battery|unknown, "powerProfile": normal|low-power|unknown}"""
    if is_darwin():
        batt = run(["pmset", "-g", "batt"])
        src = "unknown"
        if "AC Power" in batt:
            src = "ac"
        elif "Battery Power" in batt:
            src = "battery"
        prof = "normal"
        m = re.search(r"^\s*lowpowermode\s+(\d)", run(["pmset", "-g"]), re.M)
        if m and m.group(1) == "1":
            prof = "low-power"
        return {"powerSource": src, "powerProfile": prof}
    src = "unknown"
    for st in glob.glob("/sys/class/power_supply/BAT*/status"):
        try:
            with open(st) as fh:
                s = fh.read().strip().lower()
        except OSError:
            continue
        src = "battery" if s == "discharging" else "ac"
    return {"powerSource": src, "powerProfile": "unknown"}


def check_power() -> dict:
    p = read_power()
    if p["powerSource"] == "battery":
        raise Refused("on-battery", "refusing to benchmark on battery power: clocks and thermals are not comparable; plug in")
    if p["powerProfile"] == "low-power":
        raise Refused("low-power-mode", "refusing to benchmark in low-power mode: turn it off first")
    return p


# ---- memory ------------------------------------------------------------------------
def total_memory_bytes() -> int | None:
    if is_darwin():
        v = sysctl("hw.memsize")
        return int(v) if v.isdigit() else None
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def free_memory_bytes() -> int | None:
    """Memory that can be claimed without swapping: free + inactive + speculative + purgeable."""
    if is_darwin():
        out = run(["vm_stat"])
        m = re.search(r"page size of (\d+) bytes", out)
        if not m:
            return None
        page = int(m.group(1))
        pages = 0
        for label in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"):
            mm = re.search(rf"{label}:\s+(\d+)", out)
            if mm:
                pages += int(mm.group(1))
        return pages * page
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def process_memory_bytes(pid: int) -> int | None:
    """Memory attributed to one PID: `footprint` on macOS (counts GPU-wired pages), else RSS."""
    if is_darwin() and shutil.which("footprint"):
        out = run(["footprint", "-p", str(pid)], timeout=15)
        m = re.search(r"Footprint:\s*([\d.]+)\s*([KMGT]?B)", out)
        if m:
            mult = {"B": 1, "KB": 1024, "MB": 1024 ** 2, "GB": 1024 ** 3, "TB": 1024 ** 4}[m.group(2)]
            return int(float(m.group(1)) * mult)
    out = run(["ps", "-o", "rss=", "-p", str(pid)]).strip()
    return int(out) * 1024 if out.isdigit() else None


class MemoryWatchdog:
    """Polls one PID's memory and the machine's free memory; calls `on_trip` once if the PID exceeds the cap or free memory
    drops below `min_free_bytes`. Starts the moment it is entered (the tuner enters it right after the process is spawned,
    before the model finishes loading). `peak` and `min_free` are what was seen, for the ledger."""

    def __init__(self, pid: int, cap_bytes: float | None, on_trip, interval: float = 2.0, reader=None,
                 min_free_bytes: float | None = None, free_reader=None):
        self.pid, self.cap, self.on_trip, self.interval = pid, cap_bytes, on_trip, interval
        self.reader = reader or process_memory_bytes
        self.min_free_bytes = min_free_bytes
        self.free_reader = free_reader or free_memory_bytes
        self.tripped = False
        self.reason = None
        self.peak = 0
        self.min_free = None
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _trip(self, reason: str):
        self.tripped, self.reason = True, reason
        self.on_trip()

    def _loop(self):
        while True:
            v = self.reader(self.pid)
            if v is not None:
                self.peak = max(self.peak, v)
                if self.cap is not None and v > self.cap:
                    return self._trip(f"process memory {v} above cap {self.cap:.0f}")
            if self.min_free_bytes is not None:
                f = self.free_reader()
                if f is not None:
                    self.min_free = f if self.min_free is None else min(self.min_free, f)
                    if f < self.min_free_bytes:
                        return self._trip(f"system free memory {f} below {self.min_free_bytes:.0f}")
            if self._stop.wait(self.interval):
                return

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=self.interval + 5)
        return False


# ---- gpu ---------------------------------------------------------------------------
def gpu_utilization_percent() -> float | None:
    """Best-effort GPU busy percentage, None when unknowable."""
    if shutil.which("nvidia-smi"):
        out = run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"])
        vals = [float(x) for x in re.findall(r"^\s*(\d+)\s*$", out, re.M)]
        if vals:
            return max(vals)
    if is_darwin():
        out = run(["ioreg", "-r", "-d", "1", "-w0", "-c", "IOAccelerator"])
        vals = [float(x) for x in re.findall(r'"Device Utilization %"\s*=\s*(\d+)', out)]
        if vals:
            return max(vals)
    return None


def gpu_busy_median(samples: int = 5, interval: float = 1.0) -> float | None:
    """Median GPU utilization over a few seconds. One instant reading is not a measurement: on a Mac in use the
    compositor (WindowServer, browsers, the terminal's web view) holds the GPU at 30-40% in short bursts."""
    vals = []
    for i in range(samples):
        u = gpu_utilization_percent()
        if u is not None:
            vals.append(u)
        if i + 1 < samples:
            time.sleep(interval)
    if not vals:
        return None
    vals.sort()
    return vals[len(vals) // 2]


def check_gpu_idle(threshold: float = 60.0, samples: int = 5) -> float | None:
    """Refuse when another job is really using the GPU (median over `samples` seconds above `threshold`). Desktop
    compositing stays under the default; a second model server or a render does not. Returns the median measured."""
    u = gpu_busy_median(samples)
    if u is not None and u > threshold:
        raise Refused("gpu-busy", f"GPU utilization median {u:.0f}% over {samples} s (> {threshold:.0f}%): stop other GPU work "
                                  f"first, or pass --gpu-busy-max to accept this background load")
    return u


def port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    import socket
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---- bandwidth ---------------------------------------------------------------------
_WORKER = (
    "import os,sys,time\n"
    "size=64*1024*1024; secs=float(sys.argv[1])\n"
    "src=memoryview(bytearray(os.urandom(4096))*(size//4096)); dst=memoryview(bytearray(size)); dst[:]=src\n"
    "n=0; t0=time.perf_counter()\n"
    "while time.perf_counter()-t0<secs:\n"
    "    dst[:]=src; n+=size\n"
    "print(n, time.perf_counter()-t0)\n"
)


def measure_bandwidth(seconds: float = 1.0, procs: int | None = None) -> dict:
    """Copy-throughput probe using only the standard library: N worker processes each copy a private
    64 MiB buffer (memoryview slice assignment, a memcpy) for `seconds`. Reported as bytes copied per
    second, i.e. read-equivalent GB/s (each copy also writes the same amount). A lower bound on what
    the memory system can stream; real decode reads can be a little faster or slower."""
    import subprocess
    import sys
    n = procs or max(1, min(8, os.cpu_count() or 1))
    ps = [subprocess.Popen([sys.executable, "-I", "-c", _WORKER, str(seconds)], stdout=subprocess.PIPE, text=True)
          for _ in range(n)]
    moved, dts = 0.0, []
    for p in ps:
        try:
            out, _ = p.communicate(timeout=seconds + 60)
            b, dt = out.split()
            moved += float(b)
            dts.append(float(dt))
        except (subprocess.SubprocessError, ValueError):
            p.kill()
            p.communicate()
    if not dts:
        return {"memBandwidthGBs": None, "method": "stdlib memcpy worker processes failed"}
    gbs = moved / max(dts) / 1e9
    return {"memBandwidthGBs": round(gbs, 1),
            "method": f"stdlib memcpy, {len(dts)} worker processes x 64 MiB, {seconds:.1f}s, read-equivalent (lower bound)"}


# ---- machine -----------------------------------------------------------------------
def machine() -> dict:
    osname = platform.system().lower()
    arch = platform.machine()
    m: dict = {"os": f"{osname} {platform.release()}", "chip": None, "accelerator": None,
               "unifiedMemory": False, "memoryBytes": total_memory_bytes(),
               "perfCores": None, "effCores": None}
    if is_darwin():
        m["chip"] = sysctl("machdep.cpu.brand_string") or arch
        apple = arch == "arm64"
        m["unifiedMemory"] = apple
        m["accelerator"] = "apple-gpu (metal)" if apple else "unknown"
        p = sysctl("hw.perflevel0.physicalcpu")
        e = sysctl("hw.perflevel1.physicalcpu")
        m["perfCores"] = int(p) if p.isdigit() else (os.cpu_count() or None)
        m["effCores"] = int(e) if e.isdigit() else 0
    else:
        chip = None
        try:
            with open("/proc/cpuinfo") as fh:
                for line in fh:
                    if line.lower().startswith("model name"):
                        chip = line.split(":", 1)[1].strip()
                        break
        except OSError:
            pass
        m["chip"] = chip or arch
        m["perfCores"] = os.cpu_count()
        m["effCores"] = 0
        if shutil.which("nvidia-smi"):
            out = run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]).strip().splitlines()
            m["accelerator"] = "nvidia: " + "; ".join(out) if out else "nvidia"
        elif shutil.which("rocm-smi"):
            m["accelerator"] = "amd (rocm-smi present)"
        elif glob.glob("/sys/class/drm/card*/device/vendor"):
            m["accelerator"] = "gpu (drm)"
    m.update(read_power())
    return m
