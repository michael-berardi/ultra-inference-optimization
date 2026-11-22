"""The accept/reject rule shared by `tune` and `climb`."""
from __future__ import annotations

from typing import Mapping, Sequence

from .stats import median, point_ratio, ratio_ci

MIN_GAIN = 0.02          # a win must also be at least this large, not just statistically positive
MAX_REGRESSION = 0.03    # no other metric may get worse than this

# name -> higher is better
METRIC_DIRECTION = {"decode": True, "prefill": True, "ttft": False}
TARGETS = {"decode": ["decode"], "prefill": ["prefill"], "ttft": ["ttft"], "balanced": ["decode", "prefill"]}


def pool(runs: Sequence[Mapping[str, Sequence[float]]]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for r in runs:
        for k, v in r.items():
            out.setdefault(k, []).extend(v)
    return out


def decide(controls: Sequence[Mapping[str, Sequence[float]]], cand: Mapping[str, Sequence[float]],
           target: list[tuple[str, bool]], guard: list[tuple[str, bool]] | None = None,
           min_gain: float = MIN_GAIN, iters: int = 2000, seed: int = 12345) -> dict:
    """Accept only if the candidate beats every neighbouring control on the target, the bootstrap 95% CI of the
    pooled ratio is above 1.0 (and the point gain >= min_gain), and no guard metric regresses > 3%."""
    pooled = pool(controls)
    for n, _ in target + (guard or []):
        if not cand.get(n) or not pooled.get(n):
            return {"verdict": "invalid", "reason": f"missing samples for {n}", "accept": False}
    per_control = [point_ratio(cand, c, target) for c in controls]
    point, lo, hi = ratio_ci(cand, pooled, target, iters=iters, seed=seed)
    regressions = {}
    for n, hb in guard or []:
        r, _lo, hi_g = ratio_ci(cand, pooled, [(n, hb)], iters=iters, seed=seed)
        # a regression needs both a >3% point drop and evidence it is not noise (CI upper bound below 1.0)
        if r < 1 - MAX_REGRESSION and hi_g < 1.0:
            regressions[n] = round(r, 4)
    beats_all = all(r > 1.0 for r in per_control)
    out = {"ratio": round(point, 4), "ci95": [round(lo, 4), round(hi, 4)],
           "perControl": [round(r, 4) for r in per_control], "beatsAllControls": beats_all,
           "regressions": regressions, "minGain": min_gain}
    if lo > 1.0 and point >= 1 + min_gain and beats_all:
        if regressions:
            out.update(verdict="rejected-regression", accept=False,
                       reason="a non-target metric regressed more than 3%: " + ", ".join(f"{k} x{v}" for k, v in regressions.items()))
        else:
            out.update(verdict="accept", accept=True, reason="beats both controls with CI above 1.0")
    elif hi < 1.0:
        out.update(verdict="worse", accept=False, reason="confidence interval entirely below 1.0")
    else:
        out.update(verdict="noise", accept=False, reason="gain not distinguishable from noise or below minimum gain")
    return out


def summarize(samples: Mapping[str, Sequence[float]]) -> dict:
    return {k: (round(median(v), 3) if v else None) for k, v in samples.items()}
