"""L1 — Perception (REAL seam). Encode multi-modal signal into the shared latent.

Open backbone (injectable, frozen): **V-JEPA 2** (video/control) · **DINOv2/3** (stills).
The learned part — the only part we train — is a projection head into `LATENT_DIM`
(unit-norm, VICReg) plus the cross-modal binding loss (G1). See `service.py`.
"""
from verge.l1_perception.service import (
    L1Perception,
    Signal,
    default_backbone,
    synth_pair,
)

__all__ = ["L1Perception", "Signal", "default_backbone", "synth_pair"]
