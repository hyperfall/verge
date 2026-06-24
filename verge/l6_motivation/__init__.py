"""L6 — Motivation (REAL seam; overseer-gated indefinitely).

MAGELLAN-style learning-progress predictor proposes goals; a degenerate-curiosity detector
filters noisy-TV / echo-chamber traps. The autonomy boundary is enforced at the type level:
`propose()` is autonomous; `act()` requires a `HumanApproval` token and a 100%-decomposable
goal (G6), and only logs to the Ring-0 append-only audit ledger. See `magellan.py`, `service.py`.
"""
from verge.l6_motivation.magellan import (
    DegenerateCuriosityDetector,
    LearningProgressPredictor,
)
from verge.l6_motivation.service import Goal, HumanApproval, L6Motivation, SubGoal

__all__ = [
    "L6Motivation", "Goal", "SubGoal", "HumanApproval",
    "LearningProgressPredictor", "DegenerateCuriosityDetector",
]
