"""The accept/reject rule shared by `tune` and `climb`."""
from __future__ import annotations

from typing import Mapping, Sequence

from .stats import Tagged, median, point_ratio, ratio_ci, tags_of

MIN_GAIN = 0.02          # a win must also be at least this large, not just statistically positive
MAX_REGRESSION = 0.03    # no other metric may get worse than this

# name -> higher is better
METRIC_DIRECTION = {"decode": True, "prefill": True, "ttft": False}
TARGETS = {"decode": ["decode"], "prefill": ["prefill"], "ttft": ["ttft"], "balanced": ["decode", "prefill"]}


def pool(runs: Sequence[Mapping[str, Sequence[float]]]) -> dict[str, list[float]]:
    """Concatenate runs. Round ids are offset per run so rounds of different runs stay distinct when resampled."""
    out: dict[str, Tagged] = {}
    for idx, r in enumerate(runs):
        for k, v in r.items():
            dst = out.setdefault(k, Tagged())
            base = (max((t[0] for t in dst.tags), default=-1) + 1) if dst.tags else 0
            for x, (rnd, prompt) in zip(v, tags_of(v)):
                dst.append(x)
                dst.tags.append((base + rnd, prompt))
    return out


def control_gap(controls: Sequence[Mapping[str, Sequence[float]]], target: list[tuple[str, bool]]) -> float | None:
    """How far apart two runs of the SAME configuration landed, as a ratio >= 1 (None with fewer than two controls)."""
    if len(controls) < 2:
        return None
    r = point_ratio(controls[1], controls[0], target)
    return max(r, 1.0 / r) if r > 0 else float("inf")


def decide(controls: Sequence[Mapping[str, Sequence[float]]], cand: Mapping[str, Sequence[float]],
           target: list[tuple[str, bool]], guard: list[tuple[str, bool]] | None = None,
           min_gain: float = MIN_GAIN, iters: int = 2000, seed: int = 12345) -> dict:
    """Accept only if the candidate beats every neighbouring control on the target, the bootstrap 95% CI of the
    pooled ratio is above 1.0, the point gain is >= min_gain AND larger than the gap between the two controls (the
    measured run-to-run drift of an unchanged config), and no guard metric is more than 3% worse."""
    pooled = pool(controls)
    for n, _ in target + (guard or []):
        if not cand.get(n) or not pooled.get(n):
            return {"verdict": "invalid", "reason": f"missing samples for {n}", "accept": False}
    per_control = [point_ratio(cand, c, target) for c in controls]
    point, lo, hi = ratio_ci(cand, pooled, target, iters=iters, seed=seed)
    gap = control_gap(controls, target)
    regressions = {}
    for n, hb in guard or []:
        r = point_ratio(cand, pooled, [(n, hb)])
        if r < 1 - MAX_REGRESSION:  # a drop below 0.97 rejects on its own, no CI needed
            regressions[n] = round(r, 4)
    beats_all = all(r > 1.0 for r in per_control)
    exceeds_gap = gap is None or point > gap
    out = {"ratio": round(point, 4), "ci95": [round(lo, 4), round(hi, 4)],
           "perControl": [round(r, 4) for r in per_control], "beatsAllControls": beats_all,
           "controlGap": None if gap is None else round(gap, 4), "exceedsControlGap": exceeds_gap,
           "regressions": regressions, "minGain": min_gain}
    if lo > 1.0 and point >= 1 + min_gain and beats_all and exceeds_gap:
        if regressions:
            out.update(verdict="rejected-regression", accept=False,
                       reason="a non-target metric regressed more than 3%: " + ", ".join(f"{k} x{v}" for k, v in regressions.items()))
        else:
            out.update(verdict="accept", accept=True, reason="beats both controls with CI above 1.0 and more than the control gap")
    elif hi < 1.0 and point < 1 - min_gain:
        out.update(verdict="worse", accept=False, reason="confidence interval entirely below 1.0")
    else:
        out.update(verdict="noise", accept=False, reason="gain not distinguishable from noise, below the minimum gain, or within the control gap")
    return out


def summarize(samples: Mapping[str, Sequence[float]]) -> dict:
    return {k: (round(median(v), 3) if v else None) for k, v in samples.items()}
