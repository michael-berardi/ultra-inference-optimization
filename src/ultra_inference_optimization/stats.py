"""Statistics: medians and a bootstrap confidence interval on the ratio of medians."""
from __future__ import annotations

import math
import random
from statistics import median as _median
from typing import Mapping, Sequence


class Tagged(list):
    """Sample values that remember which round and which prompt each came from, so a resample can keep whole rounds
    and compare prompt against the same prompt. Plain lists are treated as one prompt, one value per round."""

    def __init__(self, values=(), tags=None):
        super().__init__(values)
        self.tags = list(tags) if tags is not None else [(i, "_") for i in range(len(self))]


def tags_of(xs) -> list:
    t = getattr(xs, "tags", None)
    return list(t) if t is not None and len(t) == len(xs) else [(i, "_") for i in range(len(xs))]


def median(xs: Sequence[float]) -> float:
    return float(_median(list(xs))) if xs else float("nan")


def _ratio(cand_med: float, ctrl_med: float, higher_better: bool) -> float:
    num, den = (cand_med, ctrl_med) if higher_better else (ctrl_med, cand_med)
    if den <= 0 or num <= 0:
        return 1.0 if den <= 0 and num <= 0 else (0.0 if num <= 0 else float("inf"))
    return num / den


def _by_prompt(xs) -> dict:
    out: dict = {}
    for v, (_r, p) in zip(xs, tags_of(xs)):
        out.setdefault(p, []).append(v)
    return out


def _clog(r: float) -> float:
    return math.log(r) if 0 < r < float("inf") else (-20.0 if r == 0 else 20.0)


def point_ratio(cand: Mapping[str, Sequence[float]], ctrl: Mapping[str, Sequence[float]],
                metrics: Sequence[tuple[str, bool]]) -> float:
    """Geometric mean over metrics and prompts of (candidate median / control median) for the same prompt, oriented so
    >1 is better. Untagged lists count as a single prompt."""
    logs = []
    for name, hb in metrics:
        c, k = _by_prompt(cand[name]), _by_prompt(ctrl[name])
        shared = [p for p in c if p in k] or None
        if shared is None:
            logs.append(_clog(_ratio(median(cand[name]), median(ctrl[name]), hb)))
            continue
        logs.append(sum(_clog(_ratio(median(c[p]), median(k[p]), hb)) for p in shared) / len(shared))
    return math.exp(sum(logs) / len(logs))


def _rounds(samples: Mapping[str, Sequence[float]], names) -> dict:
    """round id -> {metric: [(value, prompt)]}"""
    out: dict = {}
    for n in names:
        for v, (r, p) in zip(samples[n], tags_of(samples[n])):
            out.setdefault(r, {}).setdefault(n, []).append((v, p))
    return out


def _resample(rounds: dict, rng: random.Random, names) -> dict:
    ids = sorted(rounds)
    out = {n: Tagged() for n in names}
    for k in range(len(ids)):
        r = rng.choice(ids)
        for n in names:
            for v, p in rounds[r].get(n, ()):
                out[n].append(v)
                out[n].tags.append((k, p))
    return out


def ratio_ci(cand: Mapping[str, Sequence[float]], ctrl: Mapping[str, Sequence[float]],
             metrics: Sequence[tuple[str, bool]], iters: int = 2000, seed: int = 12345,
             alpha: float = 0.05) -> tuple[float, float, float]:
    """(point, lo, hi): bootstrap percentile interval of the oriented ratio of medians. Whole rounds are resampled (every
    prompt of a drawn round comes along) and each prompt is compared with itself."""
    rng = random.Random(seed)
    names = [n for n, _h in metrics]
    point = point_ratio(cand, ctrl, metrics)
    rc, rk = _rounds(cand, names), _rounds(ctrl, names)
    draws = []
    for _ in range(iters):
        c, t = _resample(rc, rng, names), _resample(rk, rng, names)
        if any(not c[n] or not t[n] for n in names):
            continue
        draws.append(point_ratio(c, t, metrics))
    if not draws:
        return point, point, point
    draws.sort()
    lo = draws[int((alpha / 2) * (len(draws) - 1))]
    hi = draws[int((1 - alpha / 2) * (len(draws) - 1))]
    return point, lo, hi
