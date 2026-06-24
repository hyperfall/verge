"""L5 — Social / Theory of Mind (REAL seam; frontier, don't gate the system on it).

Bayesian inverse planning over a mental-state subspace recovers another mind's belief/goal
from its behaviour — including *false* beliefs (the Sally-Anne structure). The same machinery
models overseer intent (where alignment lives). ExploreToM is the adversarial eval engine
at scale. See `inverse_planning.py` and `service.py`.
"""
from verge.l5_social.inverse_planning import (
    InversePlanner,
    MentalState,
    policy,
    sample_trajectory,
)
from verge.l5_social.service import L5Social

__all__ = ["L5Social", "InversePlanner", "MentalState", "policy", "sample_trajectory"]
