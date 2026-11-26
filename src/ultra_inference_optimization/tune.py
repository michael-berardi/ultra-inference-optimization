"""`uio tune`: hill-climb a llama.cpp launch configuration."""
from __future__ import annotations

import itertools
import time
from pathlib import Path

from . import argmap, bench, ceiling, experiment, gates, gguf, probe, profile, reaper, sysinfo
from .engines import Engine
from .errors import BudgetExceeded, Failed, Refused, UioError
from .experiment import METRIC_DIRECTION, TARGETS, control_gap, decide, pool, summarize
from .lock import RunLock
from .stats import median, point_ratio
from .util import append_jsonl, append_text, log, now_iso, private_mkdir, prune_runs, stamp, uio_home, write_json

GIB = 1024 ** 3
MIB = 1024 ** 2
EARLY_EXIT_GAPS = 3.0  # a lone control may end an experiment early only when the candidate is worse by > 3x the control gap
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
        self.needle = None
        self.peak = 0
        self.min_free = None

    def metrics_median(self, name: str) -> float:
        vals = sorted(v for v in (self.metrics.get(name) or []) if isinstance(v, (int, float)))
        return float(vals[len(vals) // 2]) if vals else 0.0


class Tuner:
    def __init__(self, profile_path: str, engine: Engine, target="decode", allow="lossless", budget_s=2700.0,
                 workdir: Path | None = None, apply=False, samples=6, min_gain=experiment.MIN_GAIN,
                 mem_cap_bytes: float | None = None, bandwidth_gbs: float | None = None,
                 headroom_bytes: int = 2 * GIB, levers=None, preflight=True, seed=1, control_max_age_s=900.0,
                 gpu_busy_max: float = 60.0, launch_args: list[str] | None = None,
                 min_free_bytes: float = 512 * MIB):
        self.gpu_busy_max = gpu_busy_max
        self.min_free_bytes = min_free_bytes
        self.deadline = None
        self.control_gap_seen = None
        self.baseline_needle = None
        self.launch_args = list(launch_args or [])
        self.gpu_background = None
        if target not in TARGETS:
            raise Refused("bad-target", f"target must be one of {', '.join(TARGETS)}")
        if allow not in ("lossless", "lossy"):
            raise Refused("bad-allow", "allow must be lossless or lossy")
        self.profile_path, self.engine, self.target_name, self.allow = profile_path, engine, target, allow
        self.budget_s, self.apply, self.samples, self.min_gain = budget_s, apply, samples, min_gain
        self.default_workdir = workdir is None
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
        self.extra_before = list(self.profile.get("extraArgs") or [])  # what the file said when the run started
        self.ledger = self.workdir / "ledger.jsonl"
        self.plan = self.workdir / "PLAN.md"
        self.env0 = None
        self.accepted: list = []
        self.all_runs: list = []
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

    def run_config(self, args, role, lever=None, variant=None, needle=False, needle_only=False, budgeted=True) -> Run:
        if budgeted and self.deadline is not None and time.monotonic() >= self.deadline:
            raise BudgetExceeded("budget", "time budget ended")
        rid = next(self.ids)
        run = Run(rid, role, args)
        write_json(self.workdir / "configs" / f"{rid:03d}-{role}.json",
                   {"id": rid, "role": role, "lever": lever, "variant": variant, "extraArgs": list(args),
                    "ctx": self.profile["ctx"], "source": self.profile["source"]})
        sysinfo.check_power()
        port = sysinfo.free_port()
        log_path = self.workdir / "logs" / f"{rid:03d}-{role}.log"
        server = None
        wd = None
        t0 = time.monotonic()

        def spawned(srv):
            """Called the moment the process exists (before the model has loaded): record it and start the watchdog."""
            nonlocal server, wd
            if server is not None:
                return
            server = srv
            if srv.pid:
                reaper.record(srv.pid, self.workdir)
                wd = sysinfo.MemoryWatchdog(srv.pid, self.mem_cap, srv.kill_for_watchdog, min_free_bytes=self.min_free_bytes)
                wd.__enter__()

        self.engine.spawn_hook = spawned
        try:
            started = self.engine.start(self.profile, args, port, log_path)
            spawned(started)
            deadline = self.deadline if budgeted else None
            bench.stream_chat(server.base_url, bench.unique_prefix() + "Say hello.", 8, timeout=900)  # warmup, unrecorded
            if not needle_only:
                res = bench.run_bench(server.base_url, self.samples, bench.build_panel(self.seed), self.seed,
                                      deadline=deadline)
                run.result, run.metrics, run.flags = res, bench.metrics_of(res), res["flags"]
                run.greedy = gates.greedy_outputs(server.base_url)
            if needle:
                run.needle = gates.needle_gate(server.base_url, self.profile["ctx"], deadline=deadline)
            run.ok = "failed" not in run.flags and "looping" not in run.flags and \
                (needle_only or bool(run.metrics.get("decode")))
            if not run.ok:
                run.error = f"run flagged {run.flags or 'no decode samples'}"
        except (EnvironmentChanged, BudgetExceeded):
            raise
        except UioError as e:
            run.error = e.cause + ": " + e.message
        finally:
            if wd:
                wd.__exit__(None, None, None)
                run.watchdog, run.peak, run.min_free = wd.tripped, wd.peak, wd.min_free
            self.engine.spawn_hook = None
            if server is not None:
                server.stop()
            reaper.clear(self.workdir)
        if run.watchdog:
            run.ok = False
            run.error = f"memory-watchdog: server stopped ({wd.reason})"
        run.finished = time.monotonic()
        try:
            self.env_ok_after()
        except EnvironmentChanged:
            append_jsonl(self.ledger, {"ts": now_iso(), "id": rid, "role": role, "voided": True, "args": list(args)})
            raise
        append_jsonl(self.ledger, {"ts": now_iso(), "id": rid, "role": role, "lever": lever, "variant": variant,
                                   "args": list(args), "ok": run.ok, "error": run.error, "flags": run.flags,
                                   "metrics": summarize(run.metrics), "seconds": round(time.monotonic() - t0, 1),
                                   "draftAcceptance": run.result.get("draftAcceptance"), "voided": False,
                                   "peakMemoryBytes": run.peak or None, "minSystemFreeBytes": run.min_free})
        self.all_runs.append(run)
        return run

    # ---- experiments --------------------------------------------------------------------
    def gate_candidate(self, lever, cand: Run, control: Run, cand_args) -> dict:
        gate = {"sanity": gates.sanity(cand.greedy, cand.flags)}
        eq = gates.compare_greedy(control.greedy, cand.greedy)
        gate["greedyEquivalence"] = {**eq, "required": lever.lossless}
        gate["greedyEquivalence"]["passed"] = eq["equal"] or not lever.lossless
        nd = self.run_config(cand_args, "gate", lever.name, None, needle=True, needle_only=True)
        gate["needle"] = nd.needle or {"passed": False, "trials": [], "error": nd.error}
        # no worse than the unchanged baseline measured once at the start (a model that cannot find the needle at all is
        # not held against the candidate, but may not do worse than it did)
        gate["needle"]["baseline"] = None if not self.baseline_needle else {
            k: self.baseline_needle.get(k) for k in ("exact", "total", "score")}
        gate["needle"]["passed"] = gates.needle_no_worse(gate["needle"], self.baseline_needle)
        gate["passed"] = gate["sanity"]["passed"] and gate["greedyEquivalence"]["passed"] and gate["needle"]["passed"]
        return gate

    def _early_worse(self, d: dict, ratio: float) -> bool:
        """One control is enough to stop only when the candidate is clearly worse: below the control by more than 3x the
        gap seen between controls of the same configuration (unknown gap: never)."""
        if self.control_gap_seen is None or d.get("verdict") != "worse":
            return False
        import math
        gap_log = max(math.log(self.control_gap_seen), 0.01)
        return ratio > 0 and math.log(ratio) < -EARLY_EXIT_GAPS * gap_log

    def experiment(self, lever, variant, best_args, best_run: Run, target, guard):
        cand_args = variant.apply(best_args)
        if argmap.same_args(cand_args, best_args):
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
        d1 = decide([ctrl1.metrics], cand.metrics, target, guard, self.min_gain, seed=self.seed)
        if self._early_worse(d1, d1["ratio"]):
            d1["controls"] = 1
            entry.update(d1)
            self.plan_line(f"#{cand.id} {lever.name} / {variant.label}: x{d1['ratio']} CI {d1['ci95']}, worse than control "
                           f"#{ctrl1.id} by more than {EARLY_EXIT_GAPS:g}x the control gap: worse")
            return entry, ctrl1
        ctrl2 = self.run_config(best_args, "control", lever.name, "after-candidate")
        if not ctrl2.ok:
            entry.update(verdict="failed-run", reason=f"closing control failed: {ctrl2.error}")
            return entry, ctrl1
        gap = control_gap([ctrl1.metrics, ctrl2.metrics], target)
        if gap is not None and gap != float("inf"):
            self.control_gap_seen = gap if self.control_gap_seen is None else max(self.control_gap_seen, gap)
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
                           f"(controls #{ctrl1.id}, #{ctrl2.id}; control gap x{d['controlGap']}; gates passed)")
            return entry, cand
        self.plan_line(f"#{cand.id} {lever.name} / {variant.label}: x{d['ratio']} CI {d['ci95']} {d['verdict']}: {d['reason']}")
        return entry, ctrl2

    def confirm(self, baseline_args, best_args, target, guard) -> tuple[dict, list]:
        """Fresh measurement of baseline vs best, control/best/control/best, after the climb. The discovery numbers are the
        best of many tries; this round must reproduce the gain or nothing is applied. Returns (contract block, runs)."""
        out = {"ran": True, "gainRatio": None, "ci": None, "passed": False}
        runs = []
        for role, args in (("confirm-control", baseline_args), ("confirm-best", best_args),
                           ("confirm-control", baseline_args), ("confirm-best", best_args)):
            r = self.run_config(args, role, "confirmation", None, budgeted=False)
            runs.append(r)
            if not r.ok:
                self.plan_line(f"CONFIRMATION failed: run #{r.id} {r.error}")
                return out, runs
        c1, x1, c2, x2 = runs
        d = decide([c1.metrics, c2.metrics], pool([x1.metrics, x2.metrics]), target, guard, self.min_gain, seed=self.seed)
        out["gainRatio"], out["ci"] = d.get("ratio"), d.get("ci95")
        each = [experiment.point_ratio(x.metrics, pool([c1.metrics, c2.metrics]), target) > 1.0 for x in (x1, x2)]
        out["passed"] = bool(d.get("accept")) and all(each)
        self.plan_line(f"CONFIRMATION: x{d.get('ratio')} CI {d.get('ci95')} control gap x{d.get('controlGap')}: "
                       f"{'reproduced' if out['passed'] else 'NOT reproduced (' + str(d.get('reason')) + ')'}")
        return out, runs

    # ---- driver -------------------------------------------------------------------------
    def tune(self) -> dict:
        if self.default_workdir:
            prune_runs()  # keep only the newest ten default run folders, including their logs
        g = gguf.read_gguf(self.profile["source"]["path"]) if self._model_exists() else None
        if g is None:
            raise Refused("model-not-found", f"source.path does not exist: {self.profile['source']['path']}")
        if self.do_preflight:
            self.preflight(g)
        else:
            self.env0 = sysinfo.read_power()
        with RunLock():
            reaper.reap_stale()
            try:
                return self._tune(g)
            finally:
                reaper.clear(self.workdir)

    def _model_exists(self):
        import os
        return os.path.isfile(self.profile["source"]["path"])

    @staticmethod
    def _draft_n(args: list[str]) -> int | None:
        from .engines.llamacpp import get_value
        v = get_value(args, ("--spec-draft-n-max", "--draft-max", "--draft-n", "--draft"))
        return int(v) if v and v.isdigit() and int(v) > 0 else None

    @staticmethod
    def _kv_type(args) -> str:
        from .engines.llamacpp import get_value
        kv = get_value(args, ("-ctk", "--cache-type-k")) or "f16"
        return kv if kv in ceiling.KV_BYTES else "f16"

    def _ceilings(self, g, bw, ctx, best_args, acceptance, kv_type="f16"):
        draft_gguf = None
        n = self._draft_n(best_args)
        if acceptance is not None and n:
            from .engines.llamacpp import get_value
            dpath = get_value(best_args, ("-md", "--model-draft"))
            if dpath:
                try:
                    draft_gguf = gguf.read_gguf(dpath)
                except Failed:
                    draft_gguf = None
        return ceiling.compute(g, bw, ctx=ctx, kv_type=kv_type, cap_bytes=self.mem_cap,
                               draft_accept=acceptance if n else None, draft_n=n if acceptance is not None else None,
                               draft_gguf=draft_gguf)

    def _tune(self, g) -> dict:
        t_start = time.monotonic()
        self.deadline = t_start + self.budget_s
        private_mkdir(self.workdir)
        machine = sysinfo.machine()
        if self.mem_cap is None and machine.get("memoryBytes"):
            self.mem_cap = machine["memoryBytes"] * 0.85
        bw = self.bandwidth or sysinfo.measure_bandwidth()["memBandwidthGBs"]
        if not bw:
            raise Refused("no-bandwidth", "memory bandwidth could not be measured; pass --bandwidth <GB/s>")
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
        t_base = time.monotonic()
        base = self.run_config(baseline_args, "baseline", needle=True, budgeted=False)
        if self.apply:
            # Reserve the confirmation round (4 runs without gates) inside the budget, so --apply never runs past it.
            per_run = max(30.0, (time.monotonic() - t_base) / 3.0)
            self.deadline = max(time.monotonic(), self.deadline - 4 * per_run)
        if not base.ok:
            raise Failed("baseline-failed", base.error or "baseline run failed")
        self.baseline_needle = base.needle
        target, guard = self.metric_list(base.metrics)
        best_args, best_run = list(baseline_args), base
        self.plan_line(f"baseline #{base.id}: {summarize(base.metrics)}; needle "
                       f"{(base.needle or {}).get('exact')}/{(base.needle or {}).get('total')}")
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
            if time.monotonic() >= self.deadline:
                stop = "budget"
                break
            won, measured, all_noise = False, 0, True
            try:
                for v in lever.variants:
                    entry, best_run_new = self.experiment(lever, v, best_args, best_run, target, guard)
                    best_run = best_run_new
                    if entry is None:
                        continue
                    measured += 1
                    if entry.get("verdict") != "noise" and not entry.get("accept"):
                        all_noise = False
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
            except BudgetExceeded as e:
                stop = "budget"
                self.plan_line(f"STOP: {e.message} (run discarded)")
                break
            # Plateau means three levers in a row where EVERY variant was indistinguishable from noise; a lever whose variants
            # were clearly worse, failed or were gate-rejected says something else and does not count.
            if won or (measured and not all_noise):
                plateau_count = 0
            elif measured:
                plateau_count += 1
            if plateau_count >= PLATEAU_LEVERS:
                stop = "plateau"
                break
        plateau = stop in ("plateau", "levers-exhausted")
        final_ok = stop in ("plateau", "levers-exhausted", "budget")
        confirmation = {"ran": False, "gainRatio": None, "ci": None, "passed": None}
        base_final, best_final = base.metrics, best_run.metrics
        if self.apply and final_ok and self.accepted and not argmap.same_args(best_args, baseline_args):
            try:
                confirmation, cruns = self.confirm(baseline_args, best_args, target, guard)
                if len(cruns) == 4 and all(r.ok for r in cruns):
                    base_final, best_final = pool([cruns[0].metrics, cruns[2].metrics]), pool([cruns[1].metrics, cruns[3].metrics])
                    best_run = cruns[3]
            except EnvironmentChanged as e:
                stop, final_ok = "environment-changed", False
                confirmation = {"ran": True, "gainRatio": None, "ci": None, "passed": False}
                self.plan_line(f"STOP: {e.message}")
        if argmap.same_args(best_args, baseline_args):
            best_final = base_final  # nothing changed: "after" is the baseline itself, not a later control's draw
        base_s, best_s = summarize(base_final), summarize(best_final)
        trio = lambda x: {k: x.get(k) for k in ("decode", "prefill", "ttft")}
        acceptance = best_run.result.get("draftAcceptance")
        # The comparison ceiling is on the same basis as the benchmark: short prompts (so KV at 1024 tokens, not the full
        # context), the KV cache type the server really runs, and the speculative ceiling only when acceptance was measured.
        kv_type = self._kv_type(best_args)
        ceil_best = self._ceilings(g, bw, min(ctx, 1024), best_args, acceptance, kv_type)
        spec = ceil_best["decodeTokS"].get("speculative")
        denom = spec if (acceptance is not None and spec) else ceil_best["decodeTokS"]["weightsOnly"]
        pct = round(100 * best_s["decode"] / denom, 1) if best_s.get("decode") and denom else None
        ceiling_note = None
        from .engines.llamacpp import SPEC_FLAGS, has_flag
        no_spec_max = max([r.metrics_median("decode") for r in self.all_runs if not has_flag(r.args, SPEC_FLAGS)] or [0.0])
        if pct is not None and (pct > 100.0 or (no_spec_max and no_spec_max > ceil_best["decodeTokS"]["weightsOnly"])):
            # A measured decode above its own ceiling means the bandwidth figure is too low (a CPU copy probe on unified
            # memory reads well below what the GPU streams): report no percentage rather than an impossible one.
            ceiling_note = (f"ceiling unreliable: measured decode exceeds the ceiling computed from {bw} GB/s; pass --bandwidth "
                            "with the device's real memory bandwidth")
            pct = None
        out = {"baseline": {"args": baseline_args, **trio(base_s)}, "best": {"args": best_args, **trio(best_s)},
               "ceiling": {"decodeTokS": ceil_best["decodeTokS"], "kvBytesPerToken": ceil_full["kvBytesPerToken"],
                           "maxSequencesUnderCap": ceil_full["maxSequencesUnderCap"], "bandwidthGBs": bw,
                           "draftAcceptance": acceptance,
                           "percentOfDecodeCeiling": (round(100 * best_s["decode"] / cap["decodeTokS"]["weightsOnly"], 1)
                                                      if best_s.get("decode") and cap["decodeTokS"]["weightsOnly"] else None)},
               "percentOfCeiling": pct, "ceilingNote": ceiling_note,
               "bestArgs": best_args, "accepted": self.accepted, "rejected": self.rejected, "skipped": self.skipped,
               "plateau": plateau, "stoppedBecause": stop, "stopReason": stop, "workdir": str(self.workdir),
               "elapsedS": round(time.monotonic() - t_start, 1), "confirmation": confirmation,
               "applied": {"written": False, "extraArgs": None, "backup": None, "reason": "--apply not given"}}
        append_text(self.workdir / "THEORY.md", f"\n## Result\n\nBest decode {best_s.get('decode')} tok/s = "
                    f"{pct}% of the ceiling used for the comparison "
                    f"({'speculative, from the measured acceptance' if denom is spec and spec else 'weights only'}).\n")
        write_json(self.workdir / "best-config.json", {"extraArgs": best_args, "baselineArgs": baseline_args,
                                                       "metrics": best_s, "baselineMetrics": base_s,
                                                       "stopReason": stop, "target": self.target_name})
        out["applied"] = self._apply(best_args, stop, final_ok, confirmation)
        try:
            profile.record_last_tune(self.profile_path, {"workdir": str(self.workdir), "plateau": plateau,
                                                         "stopReason": stop, "applied": bool(out["applied"]["written"]),
                                                         "decodeTokS": {"before": base_s.get("decode"), "after": best_s.get("decode")}})
        except OSError:
            pass
        return out

    def _apply(self, best_args, stop, final_ok, confirmation) -> dict:
        def no(reason):
            return {"written": False, "extraArgs": None, "backup": None, "reason": reason}
        if not self.apply:
            return no("--apply not given")
        if not final_ok:
            return no(f"run ended with {stop}")
        if not self.accepted:
            return no("no accepted change: nothing passed all gates")
        if not confirmation.get("passed"):
            return no("confirmation-failed")
        extra, cause, detail = profile.plan_write(best_args, self.launch_args)
        if extra is None:
            self.plan_line(f"NOT APPLIED: {cause}: {detail}")
            return no(cause)
        res = profile.apply_extra_args(self.profile_path, extra, expect_before=self.extra_before)
        if not res["written"]:
            self.plan_line(f"NOT APPLIED: {res['reason']}: extraArgs changed since the run started")
        return res
