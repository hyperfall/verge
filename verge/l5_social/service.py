"""L5 social service (REAL seam) — inverse-planning ToM over a mental-state subspace.

`encode` lifts an inferred mental state into the shared latent (a `concept`-modality
latent on the belief/goal subspace). `model_overseer` is the alignment-bearing call: infer
the overseer's intent from observed behaviour. `predict_action` uses the inferred mental
state to anticipate what the agent will do next.

Frontier (spec §3 L5): the engine is real and testable, but ToM is not "solved" — at
scale ExploreToM is the adversarial battery and the system is never gated on L5.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from verge.latent import LATENT_DIM, Latent, LayerService, make_latent
from verge.l5_social.inverse_planning import (
    InversePlanner,
    MentalState,
    policy,
)


def _target_vec(target: tuple) -> np.ndarray:
    """Deterministic embedding of a mental-state target into the shared latent subspace."""
    seed = int.from_bytes(hashlib.sha256(str(target).encode()).digest()[:8], "little")
    return np.random.default_rng(seed).standard_normal(LATENT_DIM)


@dataclass
class L5Social(LayerService):
    layer_id: str = "L5"
    grid: tuple = (5, 5)
    beta: float = 3.0
    planner: InversePlanner = field(default=None)  # type: ignore[assignment]

    def __post_init__(self):
        if self.planner is None:
            self.planner = InversePlanner(grid=self.grid, beta=self.beta)

    def infer_mental_state(self, trajectory, candidates, prior=None) -> dict:
        """Posterior over an agent's believed-target (its mental state)."""
        return self.planner.infer(trajectory, candidates, prior)

    def model_overseer(self, trajectory, candidates, prior=None) -> Latent:
        """Infer overseer intent and emit it as a latent the rest of the stack can read
        (confidence = posterior mass on the MAP intent)."""
        post = self.planner.infer(trajectory, candidates, prior)
        best = max(post, key=post.get)
        return make_latent(_target_vec(best), modality="concept", source_layer="L5",
                           confidence=float(post[best]))

    def predict_action(self, state, mental_state: MentalState) -> str:
        """Anticipate the agent's next action under its inferred mental state."""
        p = policy(state, mental_state.target, grid=self.grid, beta=self.beta)
        return max(p, key=p.get)

    # --- LayerService --------------------------------------------------------
    def encode(self, x) -> list[Latent]:
        """Encode a MentalState (or a target tuple) into the shared latent."""
        target = x.target if isinstance(x, MentalState) else tuple(x)
        return [make_latent(_target_vec(target), modality="concept", source_layer="L5")]

    def step(self, ctx: list[Latent]) -> list[Latent]:
        # TODO(L5): full multi-agent recursive ToM (modelling minds modelling minds).
        return ctx

    def health(self) -> dict:
        return {"layer": self.layer_id, "built": True, "engine": "inverse planning",
                "eval": "ExploreToM (frontier)", "gate": "G5 ToM battery", "frontier": True}
