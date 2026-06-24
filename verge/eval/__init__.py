"""The layer-agnostic evaluation harness (verge-engineering.md §7).

The measurement spine — per the spec the most defensible original contribution.
Generalizes the M1 rig (`metrics.py` + `run.py --aggregate`) into `evaluate(...)`:
≥3 seeds on a FROZEN test set, OLS slope + 95% CI, named ablations, a pre-registered
go/no-go threshold, and a free CPU mock path.
"""
from verge.eval.harness import EvalProtocol, Report, evaluate
from verge.eval.metrics import dedupe_reasoning_paths, fit_slope, pass_at_1

__all__ = [
    "evaluate", "Report", "EvalProtocol",
    "pass_at_1", "fit_slope", "dedupe_reasoning_paths",
]
