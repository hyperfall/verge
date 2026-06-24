"""L4 plasticity service (REAL seam) — UPGD optimizer + gradient-interference monitor.

L4 is a *discipline*, not a layer that writes to the bus: it keeps the learning substrate
acquiring capability without forgetting or ossifying (spec §3 L4; Thread C). Its concrete
deliverables are the **UPGD optimizer** (the default once its ablation wins) and the
**gradient-interference monitor** wired into every training run — including L3's, which is
the loop whose §5 collapse L4 retroactively explains (catastrophic forgetting from too-
high LR).

`encode`/`step` are pass-throughs (L4 acts on optimizer state, not the latent bus); the
real API is `wrap_optimizer()` and `monitor`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from verge.latent import Latent, LayerService
from verge.l4_plasticity.upgd import GradientInterferenceMonitor, UPGD


@dataclass
class L4Plasticity(LayerService):
    layer_id: str = "L4"
    monitor: GradientInterferenceMonitor = field(default_factory=GradientInterferenceMonitor)

    def wrap_optimizer(self, *, lr: float = 0.01, sigma: float = 0.01, seed: int = 0) -> UPGD:
        """Return the UPGD optimizer — perturb low-utility units, protect high-utility ones.
        In the real stack this wraps AdamW; here it is the standalone numpy UPGD. Ships as
        the default only once its ablation beats plain SGD (the bitter-lesson gate)."""
        return UPGD(lr=lr, sigma=sigma, seed=seed)

    def observe_interference(self, grad_a, grad_b) -> float:
        """Feed two task gradients to the monitor; first-epoch value forecasts forgetting."""
        return self.monitor.observe(grad_a, grad_b)

    # --- LayerService (pass-through: L4 is a substrate, not a bus producer) ---
    def encode(self, x) -> list[Latent]:
        return []

    def step(self, ctx: list[Latent]) -> list[Latent]:
        return ctx

    def health(self) -> dict:
        return {"layer": self.layer_id, "built": True, "optimizer": "UPGD",
                "monitor": "gradient-interference",
                "interference_breaker": self.monitor.circuit_break(),
                "gate": "G4 plasticity held / forgetting bounded"}
