"""Statistics: medians and a bootstrap confidence interval on the ratio of medians."""
from __future__ import annotations

import math
import random
from statistics import median as _median
from typing import Mapping, Sequence


def median(xs: Sequence[float]) -> float:
    return float(_median(list(xs))) if xs else float("nan")


def _ratio(cand_med: float, ctrl_med: float, higher_better: bool) -> float:
    num, den = (cand_med, ctrl_med) if higher_better else (ctrl_med, cand_med)
    if den <= 0 or num <= 0:
        return 1.0 if den <= 0 and num <= 0 else (0.0 if num <= 0 else float("inf"))
    return num / den


def point_ratio(cand: Mapping[str, Sequence[float]], ctrl: Mapping[str, Sequence[float]],
                metrics: Sequence[tuple[str, bool]]) -> float:
    """Geometric mean over metrics of (candidate median / control median), oriented so >1 is better."""
    logs = []
    for name, hb in metrics:
        r = _ratio(median(cand[name]), median(ctrl[name]), hb)
        logs.append(math.log(r) if 0 < r < float("inf") else (-20.0 if r == 0 else 20.0))
    return math.exp(sum(logs) / len(logs))


def ratio_ci(cand: Mapping[str, Sequence[float]], ctrl: Mapping[str, Sequence[float]],
             metrics: Sequence[tuple[str, bool]], iters: int = 2000, seed: int = 12345,
             alpha: float = 0.05) -> tuple[float, float, float]:
    """(point, lo, hi): bootstrap percentile interval of the oriented ratio of medians."""
    rng = random.Random(seed)
    point = point_ratio(cand, ctrl, metrics)
    draws = []
    for _ in range(iters):
        c = {n: [rng.choice(cand[n]) for _ in cand[n]] for n, _h in metrics}
        t = {n: [rng.choice(ctrl[n]) for _ in ctrl[n]] for n, _h in metrics}
        draws.append(point_ratio(c, t, metrics))
    draws.sort()
    lo = draws[int((alpha / 2) * (iters - 1))]
    hi = draws[int((1 - alpha / 2) * (iters - 1))]
    return point, lo, hi
