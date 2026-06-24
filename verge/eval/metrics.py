"""Metrics for the harness — reused VERBATIM from m1-verified-reasoner/metrics.py.

`pass_at_1`, distinct-path dedup, and OLS `fit_slope` with a 95% CI are the measured,
pure-stdlib primitives. We import them from the unedited M1 module rather than
re-implement, so there is exactly one definition of "slope" in the project
(build constraint §2.6). Re-exported here as `verge.eval.metrics` for layer-agnostic use.
"""
from __future__ import annotations

from verge import _m1bridge

_m1bridge.ensure_on_path()

from metrics import (  # noqa: E402  (path set up above)
    dedupe_reasoning_paths,
    fit_slope,
    pass_at_1,
)

__all__ = ["pass_at_1", "fit_slope", "dedupe_reasoning_paths"]
