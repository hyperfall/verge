"""Metrics: single-shot pass@1, distinct-path dedup, slope+CI. Pure stdlib."""
from __future__ import annotations

import math
import re

_NUMS = re.compile(r"-?\d+\.?\d*")


def pass_at_1(flags: list[bool]) -> float:
    return sum(flags) / len(flags) if flags else 0.0


def dedupe_reasoning_paths(pairs: list) -> list:
    """RFT key ingredient: keep only DISTINCT reasoning paths per problem.
    Fingerprint = (problem id, sequence of numbers in the trace)."""
    seen, out = set(), []
    for p, tr in pairs:
        sig = (p.id, tuple(_NUMS.findall(tr)))
        if sig in seen:
            continue
        seen.add(sig)
        out.append((p, tr))
    return out


_T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 8: 2.31,
        10: 2.23, 15: 2.13, 20: 2.09, 30: 2.04}


def _t(df):
    if df <= 0:
        return float("inf")
    for k in sorted(_T95):
        if df <= k:
            return _T95[k]
    return 1.96


def fit_slope(xs: list[float], ys: list[float]):
    """OLS slope of ys~xs with 95% CI. Returns (slope, ci_low, ci_high)."""
    n = len(xs)
    if n < 3:
        raise ValueError("need >= 3 points")
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    b = my - slope * mx
    resid = [y - (b + slope * x) for x, y in zip(xs, ys)]
    s2 = sum(r * r for r in resid) / (n - 2)
    se = math.sqrt(s2 / sxx) if sxx > 0 else float("inf")
    t = _t(n - 2)
    return slope, slope - t * se, slope + t * se
