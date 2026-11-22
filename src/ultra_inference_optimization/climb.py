"""`uio climb`: generic hill-climb of any numeric metric over a parameter grid.

The metric command prints a number (last number on stdout wins). The apply command is a shell
template with {NAME} placeholders, substituted verbatim with the candidate values."""
from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

from .errors import Failed, Refused
from .experiment import decide
from .stats import median, point_ratio
from .util import append_jsonl, append_text, log, now_iso, stamp, uio_home, write_json

NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def parse_params(specs: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for s in specs:
        if "=" not in s:
            raise Refused("bad-param", f"--param must look like NAME=v1,v2 (got {s!r})")
        name, vals = s.split("=", 1)
        vs = [v for v in vals.split(",") if v != ""]
        if not re.fullmatch(r"\w+", name) or len(vs) < 1:
            raise Refused("bad-param", f"--param needs a word NAME and at least one value (got {s!r})")
        out[name] = vs
    if not out:
        raise Refused("bad-param", "give at least one --param NAME=v1,v2")
    return out


def sh(cmd: str, timeout: float = 900.0) -> str:
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise Failed("command-timeout", f"timed out after {timeout:.0f}s: {cmd[:80]}")
    if p.returncode != 0:
        raise Failed("command-failed", f"exit {p.returncode}: {cmd[:80]}: {(p.stderr or p.stdout).strip()[:200]}")
    return p.stdout


def render(template: str, values: dict[str, str]) -> str:
    out = template
    for k, v in values.items():
        out = out.replace("{" + k + "}", str(v))
    return out


class Climber:
    def __init__(self, metric_cmd, params, apply_cmd, direction="max", samples=6, budget_s=2700.0,
                 workdir=None, min_gain=0.02, run=sh, seed=12345):
        if direction not in ("min", "max"):
            raise Refused("bad-direction", "direction must be min or max")
        if samples < 6:
            raise Refused("too-few-samples", "need >= 6 samples for a median")
        self.metric_cmd, self.params, self.apply_cmd = metric_cmd, params, apply_cmd
        self.hb = direction == "max"
        self.samples, self.budget_s, self.min_gain, self.run, self.seed = samples, budget_s, min_gain, run, seed
        self.workdir = Path(workdir) if workdir else uio_home() / "runs" / (stamp() + "-climb")
        self.ids = 0

    def measure(self, values: dict, role: str, param=None) -> dict:
        self.ids += 1
        self.run(render(self.apply_cmd, values))
        xs = []
        for _ in range(self.samples):
            out = self.run(self.metric_cmd)
            nums = NUM.findall(out)
            if not nums:
                raise Failed("metric-not-numeric", f"metric command printed no number: {out.strip()[:80]!r}")
            xs.append(float(nums[-1]))
        rec = {"ts": now_iso(), "id": self.ids, "role": role, "param": param, "values": dict(values),
               "median": median(xs), "samples": xs}
        append_jsonl(self.workdir / "ledger.jsonl", rec)
        write_json(self.workdir / "configs" / f"{self.ids:03d}-{role}.json", {"values": values})
        return rec

    def climb(self) -> dict:
        deadline = time.monotonic() + self.budget_s
        self.workdir.mkdir(parents=True, exist_ok=True)
        append_text(self.workdir / "PLAN.md", f"# Climb plan\n\n- started {now_iso()}\n- metric: {self.metric_cmd}\n"
                                              f"- direction: {'max' if self.hb else 'min'}\n\n")
        best = {k: v[0] for k, v in self.params.items()}
        metric = [("value", self.hb)]
        accepted, rejected, plateau_count, stop = [], [], 0, "params-exhausted"
        try:
            base = self.measure(best, "baseline")
            best_rec = base
            for name, vals in self.params.items():
                if time.monotonic() >= deadline:
                    stop = "budget"
                    break
                won = False
                for v in vals:
                    if v == best[name]:
                        continue
                    if time.monotonic() >= deadline:
                        stop = "budget"
                        break
                    cand_vals = {**best, name: v}
                    cand = self.measure(cand_vals, "candidate", name)
                    c, k1 = {"value": cand["samples"]}, {"value": best_rec["samples"]}
                    entry = {"param": name, "value": v, "values": cand_vals}
                    if point_ratio(c, k1, metric) <= 1.0:
                        d = decide([k1], c, metric, None, self.min_gain, seed=self.seed)
                        d.update(controls=1, verdict="noise" if d["verdict"] == "accept" else d["verdict"], accept=False)
                        rejected.append({**entry, **d})
                        append_text(self.workdir / "PLAN.md", f"- {name}={v}: ratio {d['ratio']} {d['verdict']}\n")
                        continue
                    ctrl2 = self.measure(best, "control", name)
                    d = decide([k1, {"value": ctrl2["samples"]}], c, metric, None, self.min_gain, seed=self.seed)
                    d["controls"] = 2
                    entry.update(d)
                    append_text(self.workdir / "PLAN.md", f"- {name}={v}: ratio {d['ratio']} CI {d['ci95']} {d['verdict']}\n")
                    if d["accept"]:
                        won = True
                        best = cand_vals
                        best_rec = cand
                        accepted.append(entry)
                    else:
                        best_rec = ctrl2
                        rejected.append(entry)
                if stop == "budget":
                    break
                plateau_count = 0 if won else plateau_count + 1
                if plateau_count >= 3:
                    stop = "plateau"
                    break
        finally:
            try:
                self.run(render(self.apply_cmd, best))  # leave the system in the best state found
            except Exception as e:  # report, do not mask the original error
                log(f"could not re-apply best values: {e}")
        out = {"baseline": {"values": {k: v[0] for k, v in self.params.items()}, "median": base["median"]},
               "best": {"values": best, "median": best_rec["median"]}, "bestParams": best,
               "accepted": accepted, "rejected": rejected, "plateau": stop in ("plateau", "params-exhausted"),
               "stopReason": stop, "workdir": str(self.workdir)}
        write_json(self.workdir / "best-config.json", out)
        return out
