"""L4 — Plasticity (STUB, but the discipline is built alongside everything).

Open component: **UPGD** (utility-based perturbed gradient descent) as an optimizer
wrapper over AdamW (~200 LOC: perturb low-utility units more, protect high-utility ones)
— supersedes continual-backprop, handling plasticity *and* forgetting. Plus the
**gradient-interference monitor**: first-epoch interference predicts forgetting severity,
so a bad task sequence can be forecast before committing.

This is also the layer that explains the §5 L3 collapse (catastrophic forgetting from
too-high LR) — its monitor is meant to wire into EVERY training run, including L3's. The
standard-backprop baseline is built first; local-credit / predictive-coding residuals are
a non-blocking frontier spike, NOT on the critical path (build constraint §2.4).
"""
from verge.l4_plasticity.service import L4Plasticity
from verge.l4_plasticity.upgd import (
    SGD,
    UPGD,
    GradientInterferenceMonitor,
    continual_plasticity_demo,
)

__all__ = [
    "L4Plasticity", "GradientInterferenceMonitor", "UPGD", "SGD",
    "continual_plasticity_demo",
]
