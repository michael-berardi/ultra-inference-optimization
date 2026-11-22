"""`uio tune`: hill-climb a llama.cpp launch configuration."""
from __future__ import annotations

import itertools
import time
from pathlib import Path

from . import bench, ceiling, experiment, gates, gguf, probe, profile, sysinfo
from .engines import Engine
from .errors import Failed, Refused, UioError
from .experiment import METRIC_DIRECTION, TARGETS, decide, summarize
from .lock import RunLock
from .stats import median, point_ratio
from .util import append_jsonl, append_text, log, now_iso, stamp, uio_home, write_json

GIB = 1024 ** 3
PLATEAU_LEVERS = 3


class EnvironmentChanged(UioError):
    exit_code = 2


class Run:
    def __init__(self, rid, role, args):
        self.id, self.role, self.args = rid, role, list(args)
        self.ok = False
        self.error = None
        self.metrics: dict = {}
        self.result: dict = {}
        self.greedy: list = []
        self.flags: list = []
        self.finished = time.monotonic()
        self.watchdog = False


class Tuner:
    def __init__(self, profile_path: str, engine: Engine, target="decode", allow="lossless", budget_s=2700.0,
                 workdir: Path | None = None, apply=False, samples=6, min_gain=experiment.MIN_GAIN,
                 mem_cap_bytes: float | None = None, bandwidth_gbs: float | None = None,
                 headroom_bytes: int = 2 * GIB, levers=None, preflight=True, seed=1, control_max_age_s=900.0,
                 gpu_busy_max: float = 60.0, launch_args: list[str] | None = None):
        self.gpu_busy_max = gpu_busy_max
        self.launch_args = list(launch_args or [])
        self.gpu_background = None
        if target not in TARGETS:
            raise Refused("bad-target", f"target must be one of {', '.join(TARGETS)}")
        if allow not in ("lossless", "lossy"):
            raise Refused("bad-allow", "allow must be lossless or lossy")
        self.profile_path, self.engine, self.target_name, self.allow = profile_path, engine, target, allow
        self.budget_s, self.apply, self.samples, self.min_gain = budget_s, apply, samples, min_gain
        self.workdir = Path(workdir) if workdir else uio_home() / "runs" / stamp()
        self.mem_cap = mem_cap_bytes
        self.bandwidth = bandwidth_gbs
        self.headroom = headroom_bytes
        self.lever_override = levers
        self.do_preflight = preflight
        self.seed = seed
        self.control_max_age = control_max_age_s
        self.ids = itertools.count(1)
        self.profile = profile.load_profile(profile_path)
        if launch_args:
            self.profile = {**self.profile, "launchArgs": list(launch_args)}
        self.ledger = self.workdir / "ledger.jsonl"
        self.plan = self.workdir / "PLAN.md"
        self.env0 = None
        self.accepted: list = []
        self.rejected: list = []
        self.skipped: list = []

    # ---- preconditions ------------------------------------------------------------------
    def preflight(self, g: gguf.GGUF):
        self.env0 = sysinfo.check_power()
        port = self.profile.get("port")
        if isinstance(port, int) and sysinfo.port_in_use(port):
            raise Refused("port-busy", f"the profile's port {port} is in use: stop that server first")
        self.gpu_background = sysinfo.check_gpu_idle(self.gpu_busy_max)
        free = sysinfo.free_memory_bytes()
        need = g.file_size + ceiling.kv_bytes_per_token(g) * self.profile["ctx"] + self.headroom
        if free is not None and free < need:
            raise Refused("low-memory", f"free memory {free / GIB:.1f} GiB is below model+KV+headroom {need / GIB:.1f} GiB")

    def env_ok_after(self):
        now = sysinfo.read_power()
        if now["powerSource"] == "battery" or now["powerProfile"] == "low-power":
            raise EnvironmentChanged("environment-changed", f"power state became {now}: run voided")
        if self.env0 and now != self.env0:
            raise EnvironmentChanged("environment-changed", f"power state changed {self.env0} -> {now}: run voided")

    # ---- plumbing -----------------------------------------------------------------------
    def plan_line(self, text: str):
        append_text(self.plan, f"- {now_iso()} {text}\n")
        log(text)

    def metric_list(self, available: dict) -> tuple[list, list]:
        names = [n for n in ("decode", "prefill", "ttft") if available.get(n)]
        target = [(n, METRIC_DIRECTION[n]) for n in TARGETS[self.target_name] if n in names]
        if not target:
            raise Failed("no-target-samples", f"the panel produced no samples for target {self.target_name}")
        guard = [(n, METRIC_DIRECTION[n]) for n in names if n not in [t[0] for t in target]]
        return target, guard

    def run_config(self, args, role, lever=None, variant=None, needle=False, needle_only=False) -> Run:
        rid = next(self.ids)
        run = Run(rid, role, args)
        write_json(self.workdir / "configs" / f"{rid:03d}-{role}.json",
                   {"id": rid, "role": role, "lever": lever, "variant": variant, "extraArgs": list(args),
                    "ctx": self.profile["ctx"], "source": self.profile["source"]})
        sysinfo.check_power()
        port = sysinfo.free_port()
        log_path = self.workdir / "logs" / f"{rid:03d}-{role}.log"
        server = None
        t0 = time.monotonic()
        try:
            server = self.engine.start(self.profile, args, port, log_path)
            wd = sysinfo.MemoryWatchdog(server.pid, self.mem_cap, server.kill_for_watchdog) if (server.pid and self.mem_cap) else None
            try:
                if wd:
                    wd.__enter__()
                bench.stream_chat(server.base_url, bench.unique_prefix() + "Say hello.", 8, timeout=900)  # warmup, unrecorded
                if not needle_only:
                    res = bench.run_bench(server.base_url, self.samples, bench.build_panel(self.seed), self.seed)
                    run.result, run.metrics, run.flags = res, bench.metrics_of(res), res["flags"]
                    run.greedy = gates.greedy_outputs(server.base_url)
                if needle:
                    run.needle = gates.needle_gate(server.base_url, self.profile["ctx"])
            finally:
                if wd:
                    wd.__exit__(None, None, None)
                    run.watchdog = wd.tripped
            run.ok = not run.watchdog and "failed" not in run.flags and "looping" not in run.flags and \
                (needle_only or bool(run.metrics.get("decode")))
            if run.watchdog:
                run.error = "memory-watchdog: server exceeded the memory cap and was stopped"
            elif not run.ok:
                run.error = f"run flagged {run.flags or 'no decode samples'}"
        except EnvironmentChanged:
            raise
        except UioError as e:
            run.error = e.cause + ": " + e.message
        finally:
            if server is not None:
                server.stop()
        run.finished = time.monotonic()
        try:
            self.env_ok_after()
        except EnvironmentChanged:
            append_jsonl(self.ledger, {"ts": now_iso(), "id": rid, "role": role, "voided": True, "args": list(args)})
            raise
        append_jsonl(self.ledger, {"ts": now_iso(), "id": rid, "role": role, "lever": lever, "variant": variant,
                                   "args": list(args), "ok": run.ok, "error": run.error, "flags": run.flags,
                                   "metrics": summarize(run.metrics), "seconds": round(time.monotonic() - t0, 1),
                                   "draftAcceptance": run.result.get("draftAcceptance"), "voided": False})
        return run

    # ---- experiments --------------------------------------------------------------------
    def gate_candidate(self, lever, cand: Run, control: Run, cand_args) -> dict:
        gate = {"sanity": gates.sanity(cand.greedy, cand.flags)}
        eq = gates.compare_greedy(control.greedy, cand.greedy)
        gate["greedyEquivalence"] = {**eq, "required": lever.lossless}
        gate["greedyEquivalence"]["passed"] = eq["equal"] or not lever.lossless
        nd = self.run_config(cand_args, "gate", lever.name, None, needle=True, needle_only=True)
        gate["needle"] = getattr(nd, "needle", {"passed": False, "trials": [], "error": nd.error})
        if not nd.ok and not gate["needle"].get("trials"):
            gate["needle"]["passed"] = False
        gate["passed"] = gate["sanity"]["passed"] and gate["greedyEquivalence"]["passed"] and gate["needle"]["passed"]
        return gate

    def experiment(self, lever, variant, best_args, best_run: Run, target, guard):
        cand_args = variant.apply(best_args)
        if cand_args == best_args:
            return None, best_run
        if time.monotonic() - best_run.finished > self.control_max_age:
            best_run = self.run_config(best_args, "control-refresh")
        ctrl1 = best_run
        cand = self.run_config(cand_args, "candidate", lever.name, variant.label)
        entry = {"lever": lever.name, "variant": variant.label, "args": cand_args, "lossless": lever.lossless}
        if not cand.ok:
            entry.update(verdict="failed-run", reason=cand.error)
            self.plan_line(f"#{cand.id} {lever.name} / {variant.label}: FAILED RUN, {cand.error}")
            return entry, ctrl1
        if point_ratio(cand.metrics, ctrl1.metrics, target) <= 1.0:
            d = decide([ctrl1.metrics], cand.metrics, target, guard, self.min_gain, seed=self.seed)
            d["controls"] = 1
            entry.update(d)
            entry["verdict"] = d["verdict"] if d["verdict"] != "accept" else "noise"
            self.plan_line(f"#{cand.id} {lever.name} / {variant.label}: x{d['ratio']} CI {d['ci95']}, "
                           f"did not beat control #{ctrl1.id}: {entry['verdict']}")
            return entry, ctrl1
        ctrl2 = self.run_config(best_args, "control", lever.name, "after-candidate")
        if not ctrl2.ok:
            entry.update(verdict="failed-run", reason=f"closing control failed: {ctrl2.error}")
            return entry, ctrl1
        d = decide([ctrl1.metrics, ctrl2.metrics], cand.metrics, target, guard, self.min_gain, seed=self.seed)
        d["controls"] = 2
        entry.update(d)
        if d["accept"]:
            gate = self.gate_candidate(lever, cand, ctrl2, cand_args)
            entry["gates"] = gate
            if not gate["passed"]:
                entry.update(verdict="rejected-gate", accept=False, reason="a gate failed: " + ", ".join(
                    k for k in ("sanity", "greedyEquivalence", "needle") if not gate[k]["passed"]))
                self.plan_line(f"#{cand.id} {lever.name} / {variant.label}: x{d['ratio']} but {entry['reason']}")
                return entry, ctrl2
            entry["before"], entry["after"] = summarize(ctrl2.metrics), summarize(cand.metrics)
            self.plan_line(f"#{cand.id} {lever.name} / {variant.label}: ACCEPTED x{d['ratio']} CI {d['ci95']} "
                           f"(controls #{ctrl1.id}, #{ctrl2.id}; gates passed)")
            return entry, cand
        self.plan_line(f"#{cand.id} {lever.name} / {variant.label}: x{d['ratio']} CI {d['ci95']} {d['verdict']}: {d['reason']}")
        return entry, ctrl2

    # ---- driver -------------------------------------------------------------------------
    def tune(self) -> dict:
        g = gguf.read_gguf(self.profile["source"]["path"]) if self._model_exists() else None
        if g is None:
            raise Refused("model-not-found", f"source.path does not exist: {self.profile['source']['path']}")
        if self.do_preflight:
            self.preflight(g)
        else:
            self.env0 = sysinfo.read_power()
        with RunLock():
            return self._tune(g)

    def _model_exists(self):
        import os
        return os.path.isfile(self.profile["source"]["path"])

    def _tune(self, g) -> dict:
        t_start = time.monotonic()
        deadline = t_start + self.budget_s
        self.workdir.mkdir(parents=True, exist_ok=True)
        machine = sysinfo.machine()
        if self.mem_cap is None and machine.get("memoryBytes"):
            self.mem_cap = machine["memoryBytes"] * 0.85
        bw = self.bandwidth or sysinfo.measure_bandwidth()["memBandwidthGBs"]
        ctx = self.profile["ctx"]
        cap = ceiling.compute(g, bw, ctx=min(ctx, 1024), cap_bytes=self.mem_cap)
        ceil_full = ceiling.compute(g, bw, ctx=ctx, cap_bytes=self.mem_cap)
        write_json(self.workdir / "ceiling.json", ceil_full)
        append_text(self.workdir / "THEORY.md", "# Theory: ceilings\n\nBandwidth used: %s GB/s\n\n%s\n" %
                    (bw, "\n".join("- " + a for a in cap["arithmetic"])))
        append_text(self.plan, f"# Tuning plan\n\n- started {now_iso()}\n- target: {self.target_name}; allow: {self.allow}; "
                               f"budget {self.budget_s:.0f}s; samples {self.samples}; min gain {self.min_gain}\n"
                               f"- profile: {self.profile_path}\n\n## Experiments\n\n")
        baseline_args = self.engine.baseline_args(self.profile)
        base = self.run_config(baseline_args, "baseline")
        if not base.ok:
            raise Failed("baseline-failed", base.error or "baseline run failed")
        target, guard = self.metric_list(base.metrics)
        best_args, best_run = list(baseline_args), base
        self.plan_line(f"baseline #{base.id}: {summarize(base.metrics)}")
        levers = self.lever_override if self.lever_override is not None else \
            self.engine.levers(baseline_args, ctx, machine)
        plateau_count, stop = 0, "levers-exhausted"
        for lever in levers:
            if lever.skip_reason:
                self.skipped.append({"lever": lever.name, "reason": lever.skip_reason})
                continue
            if not lever.lossless and self.allow != "lossy":
                self.skipped.append({"lever": lever.name, "reason": "lossy lever and --allow lossless"})
                continue
            if time.monotonic() >= deadline:
                stop = "budget"
                break
            won = False
            try:
                for v in lever.variants:
                    if time.monotonic() >= deadline:
                        stop = "budget"
                        break
                    entry, best_run_new = self.experiment(lever, v, best_args, best_run, target, guard)
                    best_run = best_run_new
                    if entry is None:
                        continue
                    if entry.get("accept"):
                        won = True
                        best_args = list(entry["args"])
                        self.accepted.append(entry)
                    else:
                        self.rejected.append(entry)
            except EnvironmentChanged as e:
                stop = "environment-changed"
                self.plan_line(f"STOP: {e.message}")
                break
            if stop == "budget":
                break
            plateau_count = 0 if won else plateau_count + 1
            if plateau_count >= PLATEAU_LEVERS:
                stop = "plateau"
                break
        plateau = stop in ("plateau", "levers-exhausted")
        final_ok = stop in ("plateau", "levers-exhausted", "budget")
        base_s, best_s = summarize(base.metrics), summarize(best_run.metrics)
        out = {"baseline": {"args": baseline_args, **base_s}, "best": {"args": best_args, **best_s},
               "ceiling": {"decodeTokS": ceil_full["decodeTokS"], "kvBytesPerToken": ceil_full["kvBytesPerToken"],
                           "maxSequencesUnderCap": ceil_full["maxSequencesUnderCap"], "bandwidthGBs": bw,
                           "percentOfDecodeCeiling": (round(100 * best_s["decode"] / cap["decodeTokS"]["weightsOnly"], 1)
                                                      if best_s.get("decode") and cap["decodeTokS"]["weightsOnly"] else None)},
               "bestArgs": best_args, "accepted": self.accepted, "rejected": self.rejected, "skipped": self.skipped,
               "plateau": plateau, "stopReason": stop, "workdir": str(self.workdir),
               "elapsedS": round(time.monotonic() - t_start, 1), "applied": None}
        append_text(self.workdir / "THEORY.md", f"\n## Result\n\nBest decode {best_s.get('decode')} tok/s = "
                    f"{out['ceiling']['percentOfDecodeCeiling']}% of the short-context bandwidth ceiling.\n")
        write_json(self.workdir / "best-config.json", {"extraArgs": best_args, "baselineArgs": baseline_args,
                                                       "metrics": best_s, "baselineMetrics": base_s,
                                                       "stopReason": stop, "target": self.target_name})
        if self.apply:
            if not final_ok:
                out["applied"] = {"written": None, "reason": f"not applied: run ended with {stop}"}
            elif not self.accepted:
                out["applied"] = {"written": None, "reason": "nothing to apply: no change passed all gates"}
            else:
                out["applied"] = profile.apply_extra_args(self.profile_path, profile.extra_args_to_write(best_args, self.launch_args))
        try:
            profile.record_last_tune(self.profile_path, {"workdir": str(self.workdir), "plateau": plateau,
                                                         "stopReason": stop, "applied": bool(out["applied"] and out["applied"].get("written")),
                                                         "decodeTokS": {"before": base_s.get("decode"), "after": best_s.get("decode")}})
        except OSError:
            pass
        return out
