"""Command line: every command prints one JSON object on stdout (--human for text)."""
from __future__ import annotations

import argparse
import json
import signal
import sys

from . import __version__, bench, ceiling, climb, gguf, probe, sysinfo
from .engines import get_engine
from .errors import Failed, Refused, UioError
from .util import parse_duration


def _human(obj, indent=0) -> str:
    pad = "  " * indent
    if isinstance(obj, dict):
        lines = []
        for k, v in obj.items():
            if k in ("perSample", "samples") and isinstance(v, list):
                lines.append(f"{pad}{k}: [{len(v)} values]")
            elif isinstance(v, (dict, list)) and v:
                lines.append(f"{pad}{k}:")
                lines.append(_human(v, indent + 1))
            else:
                lines.append(f"{pad}{k}: {v}")
        return "\n".join(lines)
    if isinstance(obj, list):
        if all(not isinstance(x, (dict, list)) for x in obj):
            return pad + "  ".join(str(x) for x in obj) if len(obj) <= 12 else "\n".join(pad + "- " + str(x) for x in obj)
        return "\n".join(f"{pad}-\n{_human(x, indent + 1)}" for x in obj)
    return f"{pad}{obj}"


def emit(obj, human: bool) -> None:
    if human:
        print(_human(obj))
    else:
        print(json.dumps(obj))


def cmd_probe(a):
    return probe.probe(measure=not a.no_measure, llama_server=a.llama_server)


def cmd_ceiling(a):
    g = gguf.read_gguf(a.gguf)
    bw = a.bandwidth
    if bw is None:
        bw = sysinfo.measure_bandwidth()["memBandwidthGBs"]
    if (a.draft_accept is None) != (a.draft_n is None):
        raise Refused("bad-args", "--draft-accept and --draft-n go together")
    cap = a.cap_gb * 1024 ** 3 if a.cap_gb else (sysinfo.total_memory_bytes() or 0) * 0.75 or None
    ctx = a.ctx or int(g.key("context_length", 4096) or 4096)
    d = gguf.read_gguf(a.draft_gguf) if a.draft_gguf else None
    return ceiling.compute(g, bw, ctx=ctx, kv_type=a.kv_type, draft_accept=a.draft_accept, draft_n=a.draft_n,
                           draft_gguf=d, tflops=a.tflops, cap_bytes=cap)


def cmd_bench(a):
    panel = bench.load_panel(a.panel, a.seed, a.long_context, a.long_words)
    res = bench.run_bench(a.base_url, a.samples, panel, a.seed, a.long_context, timeout=a.timeout)
    if not res["decodeTokS"]["samples"] and not res["prefillTokS"]["samples"]:
        res["error"] = "no successful samples"
        raise _BenchFailed(res)
    return res


class _BenchFailed(Failed):
    def __init__(self, payload):
        super().__init__("bench-failed", "no successful samples")
        self.payload = payload


def parse_launch_args(text):
    """--launch-args must be a JSON list of strings."""
    if text is None:
        return None
    try:
        val = json.loads(text)
    except ValueError as e:
        raise Refused("bad-launch-args", f"--launch-args is not valid JSON: {e}")
    if not isinstance(val, list) or not all(isinstance(x, str) for x in val):
        raise Refused("bad-launch-args", "--launch-args must be a JSON list of strings")
    return val


def cmd_tune(a):
    from .tune import Tuner
    eng = get_engine(a.engine, binary=a.llama_server) if a.engine == "llama.cpp" else get_engine(a.engine)
    from pathlib import Path
    t = Tuner(a.profile, eng, target=a.target, allow=a.allow, budget_s=parse_duration(a.budget),
              workdir=Path(a.workdir) if a.workdir else None, apply=a.apply, samples=a.samples,
              mem_cap_bytes=a.mem_cap_gb * 1024 ** 3 if a.mem_cap_gb else None, bandwidth_gbs=a.bandwidth,
              min_gain=a.min_gain, gpu_busy_max=a.gpu_busy_max, launch_args=parse_launch_args(a.launch_args))
    return t.tune()


def cmd_climb(a):
    params = climb.parse_params(a.param)
    c = climb.Climber(a.metric_cmd, params, a.apply_cmd, a.direction, a.samples, parse_duration(a.budget),
                      a.workdir, a.min_gain)
    return c.climb()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="uio", description="Measure, bound and hill-climb local LLM serving speed.")
    p.add_argument("--version", action="version", version=f"uio {__version__}")
    p.add_argument("--human", action="store_true", help="print text instead of JSON")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("probe", help="machine facts, measured memory bandwidth, installed engines")
    s.add_argument("--no-measure", action="store_true", help="skip the bandwidth probe")
    s.add_argument("--llama-server")
    s.set_defaults(fn=cmd_probe)

    s = sub.add_parser("ceiling", help="theoretical decode/prefill ceilings from a GGUF header")
    s.add_argument("--gguf", required=True)
    s.add_argument("--ctx", type=int)
    s.add_argument("--bandwidth", type=float, help="GB/s; measured when omitted")
    s.add_argument("--kv-type", default="f16", choices=sorted(ceiling.KV_BYTES))
    s.add_argument("--draft-accept", type=float, help="per-token acceptance probability, 0..1")
    s.add_argument("--draft-n", type=int, help="tokens drafted per pass")
    s.add_argument("--draft-gguf", help="draft model, adds its bytes to each pass")
    s.add_argument("--tflops", type=float, help="usable matmul peak for the prefill ceiling")
    s.add_argument("--cap-gb", type=float, help="memory cap for the sequence count (default 75%% of RAM)")
    s.set_defaults(fn=cmd_ceiling)

    s = sub.add_parser("bench", help="streaming benchmark of an OpenAI-compatible server")
    s.add_argument("--base-url", required=True)
    s.add_argument("--samples", type=int, default=6)
    s.add_argument("--panel", default="default")
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--long-context", action="store_true")
    s.add_argument("--long-words", type=int, default=6000)
    s.add_argument("--timeout", type=float, default=600.0)
    s.set_defaults(fn=cmd_bench)

    s = sub.add_parser("tune", help="hill-climb a local server's launch arguments")
    s.add_argument("--profile", required=True)
    s.add_argument("--engine", default="llama.cpp")
    s.add_argument("--llama-server")
    s.add_argument("--budget", default="45m")
    s.add_argument("--target", default="decode", choices=["decode", "prefill", "ttft", "balanced"])
    s.add_argument("--allow", default="lossless", choices=["lossless", "lossy"])
    s.add_argument("--workdir")
    s.add_argument("--apply", action="store_true")
    s.add_argument("--samples", type=int, default=6)
    s.add_argument("--min-gain", type=float, default=0.02)
    s.add_argument("--mem-cap-gb", type=float)
    s.add_argument("--launch-args", help="JSON list of the flags the model manager adds at launch (measured with the profile)")
    s.add_argument("--gpu-busy-max", type=float, default=60.0, help="refuse when the GPU median busy %% exceeds this (default 60)")
    s.add_argument("--bandwidth", type=float)
    s.set_defaults(fn=cmd_tune)

    s = sub.add_parser("climb", help="generic hill-climb of any numeric metric")
    s.add_argument("--metric-cmd", required=True)
    s.add_argument("--param", action="append", default=[], help="NAME=v1,v2,... (first value is the baseline)")
    s.add_argument("--apply-cmd", required=True)
    s.add_argument("--direction", default="max", choices=["min", "max"])
    s.add_argument("--samples", type=int, default=6)
    s.add_argument("--budget", default="45m")
    s.add_argument("--workdir")
    s.add_argument("--min-gain", type=float, default=0.02)
    s.set_defaults(fn=cmd_climb)
    return p


class _Signalled(KeyboardInterrupt):
    """SIGTERM / SIGHUP delivered as an exception so every `finally` (stopping the server, the lock) runs."""


def _raise_signalled(signum, _frame):
    raise _Signalled(f"signal {signum}")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    old = {}
    try:
        for sig in (signal.SIGTERM, signal.SIGHUP):
            old[sig] = signal.signal(sig, _raise_signalled)
    except ValueError:  # not the main thread (embedded use): leave the handlers alone
        old = {}
    try:
        emit(args.fn(args), args.human)
        return 0
    except _BenchFailed as e:
        emit({"error": e.cause, "message": e.message, **e.payload}, args.human)
        return 3
    except UioError as e:
        emit({"error": e.cause, "message": e.message}, args.human)
        return e.exit_code
    except KeyboardInterrupt as e:
        emit({"error": "interrupted", "message": str(e) or "interrupted"}, args.human)
        return 3
    except Exception as e:  # anything unexpected still ends as a JSON object and exit 3, with the cause
        emit({"error": "unexpected", "message": f"{type(e).__name__}: {e}"}, args.human)
        return 3
    finally:
        for sig, h in old.items():
            signal.signal(sig, h)
