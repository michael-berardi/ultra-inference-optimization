"""llama.cpp (llama-server) engine: launch, lever catalog, graceful stop."""
from __future__ import annotations

import os
import re
import resource
import signal
import subprocess
import time
import urllib.error
import urllib.request

from .. import probe, sysinfo
from ..util import private_mkdir
from ..errors import Failed, Refused
from .base import Engine, Lever, Server, Variant

# ---- argument list helpers -------------------------------------------------------------
def drop(args: list[str], names: tuple[str, ...], takes_value: bool = True) -> list[str]:
    out, i = [], 0
    while i < len(args):
        a = args[i]
        head = a.split("=", 1)[0]
        if head in names:
            if takes_value and "=" not in a and i + 1 < len(args) and not (args[i + 1].startswith("-") and len(args[i + 1]) > 1 and not args[i + 1][1].isdigit()):
                i += 1
            i += 1
            continue
        out.append(a)
        i += 1
    return out


def get_value(args: list[str], names: tuple[str, ...]) -> str | None:
    """Return the last effective value for any flag in `names`, matching llama-server last-wins semantics."""
    val = None
    i = 0
    while i < len(args):
        a = args[i]
        head = a.split("=", 1)[0]
        if head in names:
            if "=" in a:
                val = a.split("=", 1)[1]
                i += 1
            else:
                val = args[i + 1] if i + 1 < len(args) else ""
                i += 2
        else:
            i += 1
    return val


def has_flag(args: list[str], names: tuple[str, ...]) -> bool:
    return any(a.split("=", 1)[0] in names for a in args)


def set_value(args: list[str], names: tuple[str, ...], value: str) -> list[str]:
    return drop(args, names) + [names[0], str(value)]


def set_bare(args: list[str], names: tuple[str, ...]) -> list[str]:
    return drop(args, names, takes_value=False) + [names[0]]


SPEC_FLAGS = ("--spec-draft-n-max", "--draft-max", "--draft", "--draft-n", "--spec-type", "-md", "--model-draft",
              "--spec-draft-n-min", "--draft-min", "--spec-draft-p-min", "--draft-p-min", "--spec-ngram-size-n")


def help_lists(help_text: str, flag: str) -> bool:
    return re.search(rf"(?<![\w-]){re.escape(flag)}(?![\w-])", help_text) is not None


def _flash_attn_takes_value(help_text: str) -> bool:
    m = re.search(r"--flash-attn[^\n]*", help_text)
    return bool(m and re.search(r"on\|off|\[on", m.group(0)))


def _memlock_ok(model_bytes: int | None) -> bool:
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    except (ValueError, OSError):
        return False
    if soft == resource.RLIM_INFINITY:
        return True
    return bool(model_bytes and soft >= model_bytes)


def catalog(help_text: str, baseline: list[str], machine: dict, model_bytes: int | None = None) -> list[Lever]:
    """Levers ordered by typical payoff. Each lever is one change derived from the current best."""
    levers: list[Lever] = []

    def need(lv: Lever, *flags: str):
        missing = [f for f in flags if not help_lists(help_text, f)]
        if missing:
            lv.skip_reason = f"binary --help does not list {', '.join(missing)}"
        return lv

    # 1. speculative decoding: only meaningful when the baseline already has a draft source
    sv = [Variant(f"--spec-draft-n-max {n}", (lambda a, n=n: set_value(a, ("--spec-draft-n-max",), n)))
          for n in (2, 3, 4, 6, 8, 12, 16)]
    sv.append(Variant("speculation off", lambda a: drop(a, SPEC_FLAGS)))
    lv = need(Lever("speculative-draft-length", sv, True, "sweep draft length or switch speculation off"),
              "--spec-draft-n-max")
    if not lv.skip_reason and not has_flag(baseline, SPEC_FLAGS):
        lv.skip_reason = "baseline has no speculative draft source to sweep"
    levers.append(lv)

    # 1b. adaptive draft length: stop drafting when the draft's own confidence falls below p-min. With partial acceptance a fixed
    #     draft length wastes the drafted tail (each rejected draft token costs a draft step and verify width).
    if has_flag(baseline, SPEC_FLAGS):
        pv = [Variant(f"--spec-draft-p-min {p}", (lambda a, p=p: set_value(a, ("--spec-draft-p-min", "--draft-p-min"), p)))
              for p in ("0.5", "0.7", "0.85")]
        levers.append(need(Lever("speculative-confidence-cutoff", pv, True,
                                 "draft only while the draft is confident; target verification keeps output identical"),
                           "--spec-draft-p-min"))
        # Draft-side sampling on the accelerator (a CPU sampler costs milliseconds per step).
        levers.append(need(Lever("draft-backend-sampling",
                                 [Variant("--spec-draft-backend-sampling", lambda a: set_bare(a, ("--spec-draft-backend-sampling",)))],
                                 True, "draft sampling on the GPU backend"), "--spec-draft-backend-sampling"))

    # 2. flash attention
    if _flash_attn_takes_value(help_text):
        fa = [Variant("-fa on", lambda a: set_value(a, ("-fa", "--flash-attn"), "on")),
              Variant("-fa off", lambda a: set_value(a, ("-fa", "--flash-attn"), "off"))]
    else:
        fa = [Variant("-fa on", lambda a: set_bare(a, ("-fa", "--flash-attn"))),
              Variant("-fa off", lambda a: drop(a, ("-fa", "--flash-attn"), takes_value=False))]
    quant_v = get_value(baseline, ("-ctv", "--cache-type-v")) or "f16"
    if quant_v not in ("f16", "f32", "bf16"):
        # llama.cpp refuses a quantised V cache without flash attention ("quantized V cache requires flash_attn"): not a lever.
        fa = [v for v in fa if not v.label.startswith("-fa off")]
    levers.append(need(Lever("flash-attention", fa, True, "numerically equivalent in theory; the equivalence gate verifies"), "-fa"))

    # 3. KV cache type (lossy, labelled)
    kv = [Variant("-ctk q8_0 -ctv q8_0 (lossy: quantised KV cache)",
                  lambda a: set_value(set_value(a, ("-ctk", "--cache-type-k"), "q8_0"), ("-ctv", "--cache-type-v"), "q8_0")),
          Variant("-ctk f16 -ctv f16",
                  lambda a: set_value(set_value(a, ("-ctk", "--cache-type-k"), "f16"), ("-ctv", "--cache-type-v"), "f16"))]
    levers.append(need(Lever("kv-cache-type", kv, False, "q8_0 KV halves KV bytes but is lossy"), "-ctk", "-ctv"))

    # 4. batch sizes
    levers.append(need(Lever("ubatch-size", [Variant(f"-ub {n}", (lambda a, n=n: set_value(a, ("-ub", "--ubatch-size"), n)))
                                         for n in (256, 512, 1024, 2048)], True, "prefill throughput"), "-ub"))
    levers.append(need(Lever("batch-size", [Variant(f"-b {n}", (lambda a, n=n: set_value(a, ("-b", "--batch-size"), n)))
                                         for n in (512, 1024, 2048, 4096)], True, "prefill throughput"), "-b"))

    # 5. threads
    perf = int(machine.get("perfCores") or os.cpu_count() or 4)
    cand = sorted({max(1, perf // 2), max(1, perf - 1), perf, perf + int(machine.get("effCores") or 0)})
    levers.append(need(Lever("threads", [Variant(f"-t {n} -tb {n}", (lambda a, n=n: set_value(set_value(a, ("-t", "--threads"), n), ("-tb", "--threads-batch"), n)))
                                       for n in cand], True, "CPU-side work and CPU offload"), "-t", "-tb"))

    # 6. host prompt cache
    levers.append(need(Lever("cache-ram", [Variant(f"--cache-ram {n}", (lambda a, n=n: set_value(a, ("--cache-ram",), n)))
                                          for n in (0, 1024, 4096)], True, "MiB of host RAM for prompt-cache reuse"), "--cache-ram"))

    # 7. parallel slots
    levers.append(need(Lever("parallel-slots", [Variant(f"-np {n}", (lambda a, n=n: set_value(a, ("-np", "--parallel"), n)))
                                              for n in (1, 2, 4)], True, "slots split the context; the full-context gate guards it"), "-np"))

    # 8. mlock where allowed
    ml = need(Lever("mlock", [Variant("--mlock", lambda a: set_bare(a, ("--mlock",)))], True, "pin weights in RAM"), "--mlock")
    if not ml.skip_reason and not _memlock_ok(model_bytes):
        ml.skip_reason = "RLIMIT_MEMLOCK too low to pin the model"
    levers.append(ml)
    return levers


# ---- server process ---------------------------------------------------------------------
def stop_process(proc: subprocess.Popen, sigint_wait: float = 20.0, sigterm_wait: float = 10.0) -> str:
    """Stop a child we started: SIGINT, bounded wait, SIGTERM, bounded wait, last resort SIGKILL of that PID."""
    if proc.poll() is not None:
        return "already-exited"
    for sig, wait, name in ((signal.SIGINT, sigint_wait, "sigint"), (signal.SIGTERM, sigterm_wait, "sigterm")):
        try:
            proc.send_signal(sig)
        except ProcessLookupError:
            return name
        try:
            proc.wait(timeout=wait)
            return name
        except subprocess.TimeoutExpired:
            continue
    proc.kill()
    proc.wait(timeout=10)
    return "sigkill"


class LlamaServer(Server):
    def __init__(self, proc: subprocess.Popen, base_url: str, log_fh, sigint_wait=20.0, sigterm_wait=10.0):
        self.proc, self.base_url, self.pid = proc, base_url, proc.pid
        self._log, self._sw = log_fh, (sigint_wait, sigterm_wait)
        self.stopped_by = None

    def alive(self) -> bool:
        return self.proc.poll() is None

    def stop(self) -> None:
        try:
            self.stopped_by = stop_process(self.proc, *self._sw)
        finally:
            try:
                self._log.close()
            except OSError:
                pass


def wait_healthy(server: LlamaServer, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not server.alive():
            raise Failed("server-exited", f"llama-server exited early (code {server.proc.returncode}); see its log")
        try:
            with urllib.request.urlopen(server.base_url + "/health", timeout=3) as r:
                if r.status == 200:
                    return
        except urllib.error.HTTPError:
            pass  # 503 while loading
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.5)
    raise Failed("server-start-timeout", f"llama-server not healthy after {timeout:.0f}s")


class LlamaCppEngine(Engine):
    name = "llama.cpp"

    def __init__(self, binary: str | None = None, startup_timeout: float = 600.0):
        self.binary = probe.find_llama_server(binary)
        self.startup_timeout = startup_timeout
        self._help: str | None = None

    def require_binary(self) -> str:
        if not self.binary:
            raise Refused("engine-missing", "llama-server not found: put it on PATH, set UIO_LLAMA_SERVER, or pass --llama-server")
        return self.binary

    def info(self) -> dict:
        b = self.require_binary()
        return {"path": b, "version": probe.llama_version(b)}

    def help_text(self) -> str:
        if self._help is None:
            self._help = sysinfo.run([self.require_binary(), "--help"], timeout=30)
        return self._help

    def levers(self, baseline_args, ctx, machine, model_bytes=None):
        return catalog(self.help_text(), baseline_args, machine, model_bytes)

    def build_command(self, profile: dict, args: list[str], port: int) -> list[str]:
        src = (profile.get("source") or {}).get("path")
        ctx = profile.get("ctx")
        cmd = [self.require_binary(), "-m", str(src)]
        if ctx:
            cmd += ["-c", str(ctx)]
        args = drop(drop(drop(list(args), ("--port",)), ("--host",)), ("-m", "--model"))
        return cmd + ["--host", "127.0.0.1", "--port", str(port)] + args

    def start(self, profile, args, port, log_path) -> LlamaServer:
        cmd = self.build_command(profile, args, port)
        private_mkdir(log_path.parent)
        fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        os.fchmod(fd, 0o600)  # server logs can contain model paths and prompts
        fh = os.fdopen(fd, "ab")
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        srv = LlamaServer(proc, f"http://127.0.0.1:{port}", fh)
        try:
            hook = getattr(self, "spawn_hook", None)
            if hook:
                hook(srv)  # the caller's memory watchdog starts now, while the model is still loading
            wait_healthy(srv, self.startup_timeout)
        except BaseException:
            srv.stop()
            raise
        return srv
