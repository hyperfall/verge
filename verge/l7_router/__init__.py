"""L7 — Reflection / router (REAL seam; integration claim only).

A learned, capacity-limited select-and-broadcast bottleneck over the shared latent:
salience = free-energy-reduction × verifier-confidence × goal-relevance. No consciousness
vocabulary. Ships only if broadcast beats a no-broadcast ablation on cross-layer transfer
(G7). See `service.py` and `cross_layer_transfer_demo`.
"""
from verge.l7_router.service import L7Router, cross_layer_transfer_demo

__all__ = ["L7Router", "cross_layer_transfer_demo"]
