"""L2 — World model (REAL seam). Predict latent dynamics under intervention `do(x)`.

Predictive half: a learned linear transition (DreamerV3 injectable behind the latent
contract). Causal half: a linear-Gaussian Neural Causal Model over named latent factors,
skeleton from VERIFIED edges only, with calibrated `do(x)` uncertainty + an
out-of-skeleton flag (G2). See `ncm.py` and `service.py`.
"""
from verge.l2_world_model.ncm import (
    CausalEdge,
    CausalSkeleton,
    InterventionResult,
    NeuralCausalModel,
)
from verge.l2_world_model.service import L2WorldModel

__all__ = [
    "L2WorldModel", "CausalSkeleton", "CausalEdge", "NeuralCausalModel",
    "InterventionResult",
]
