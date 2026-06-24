"""L2 world model (REAL seam) — predictive rollouts + causal `do(x)` head.

Two halves (verge-engineering.md §5; spec §3 L2):
  - **Predictive** — a transition the reasoner can query: `rollout(state_latent,
    action_seq) -> list[Latent]`. Implemented here as a learned linear transition
    (TD-MPC2-style, fit by least squares) so it runs on CPU; **DreamerV3** is the
    injectable upgrade behind the same latent contract (the JAX process boundary).
  - **Causal** — a `NeuralCausalModel` over named latent factors: `do(x)` with calibrated
    uncertainty and an out-of-skeleton flag (the `ncm` module). Skeleton from verified
    edges only; functional forms learned.

A fixed orthonormal embedding maps the low-dim causal/state factor space into the shared
`LATENT_DIM` latent and back, so L2 reads and writes `Latent`s like every other layer.

G2 (M3 gate): `do(x)` predictions match held-out simulation above baseline, and
out-of-skeleton interventions fire the uncertainty flag.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from verge.latent import LATENT_DIM, Latent, LayerService, make_latent
from verge.l2_world_model.ncm import CausalSkeleton, NeuralCausalModel


def _orthonormal_embedding(d_factors: int, seed: int = 7) -> np.ndarray:
    """A fixed (d_factors, LATENT_DIM) embedding with orthonormal rows: factor space ↪
    shared latent. `E @ E.T = I`, so decode = project with E."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d_factors, LATENT_DIM))
    # Gram-Schmidt via QR on the transpose
    Q, _ = np.linalg.qr(A.T)
    return Q[:, :d_factors].T  # (d_factors, LATENT_DIM), orthonormal rows


@dataclass
class L2WorldModel(LayerService):
    layer_id: str = "L2"
    skeleton: CausalSkeleton | None = None
    ncm: NeuralCausalModel | None = None
    transition: object = None        # injectable DreamerV3; default = learned linear (A,B)
    _A: np.ndarray | None = field(default=None, repr=False)
    _B: np.ndarray | None = field(default=None, repr=False)
    _E: np.ndarray | None = field(default=None, repr=False)
    _d_factors: int = 0

    # --- causal half ---------------------------------------------------------
    def fit_causal(self, skeleton: CausalSkeleton, data) -> "L2WorldModel":
        self.skeleton = skeleton
        self.ncm = NeuralCausalModel(skeleton=skeleton).fit(data)
        self._d_factors = len(skeleton.variables)
        self._E = _orthonormal_embedding(self._d_factors)
        return self

    def do(self, intervention: dict, *, evidence: dict | None = None):
        """do(X=x) over the verified-edge skeleton; returns the NCM InterventionResult
        (values, per-var uncertainty, out-of-skeleton flag)."""
        if self.ncm is None:
            raise RuntimeError("L2.do called before fit_causal")
        return self.ncm.do(intervention, evidence=evidence)

    def do_latent(self, intervention: dict, *, evidence: dict | None = None) -> Latent:
        """do(X=x) projected into the shared latent (modality='state'), confidence set to
        1−normalized-uncertainty so downstream layers see the causal certainty."""
        res = self.do(intervention, evidence=evidence)
        vec = np.array([res.values[v] for v in self.skeleton.variables])
        conf = float(np.clip(1.0 / (1.0 + res.overall_uncertainty), 0.0, 1.0))
        return make_latent(self._embed(vec), modality="state", source_layer="L2",
                           confidence=conf)

    # --- predictive half -----------------------------------------------------
    def fit_transition(self, states: np.ndarray, actions: np.ndarray,
                       next_states: np.ndarray) -> "L2WorldModel":
        """Learn s' = A s + B a by least squares (TD-MPC2-style linear transition)."""
        S = np.atleast_2d(states); Ac = np.atleast_2d(actions); Sn = np.atleast_2d(next_states)
        X = np.hstack([S, Ac])
        W, *_ = np.linalg.lstsq(X, Sn, rcond=None)   # (ds+da, ds)
        ds = S.shape[1]
        self._A = W[:ds].T
        self._B = W[ds:].T
        if self._E is None:
            self._d_factors = ds
            self._E = _orthonormal_embedding(ds)
        return self

    def rollout(self, state_latent: Latent, action_seq: list) -> list[Latent]:
        """Predict a latent rollout the reasoner can query. Uses the injectable DreamerV3
        transition if provided, else the learned linear (A,B)."""
        s = self._project(state_latent.z)
        out = []
        for a in action_seq:
            a = np.atleast_1d(np.asarray(a, dtype=np.float64))
            if self.transition is not None:
                s = np.asarray(self.transition(s, a), dtype=np.float64)
            else:
                if self._A is None:
                    raise RuntimeError("L2.rollout needs a fitted transition (or inject one)")
                s = self._A @ s + self._B @ a
            out.append(make_latent(self._embed(s), modality="state", source_layer="L2"))
        return out

    # --- LayerService + latent embedding ------------------------------------
    def encode(self, x) -> list[Latent]:
        """Encode a factor/state vector into the shared latent (modality='state')."""
        return [make_latent(self._embed(np.asarray(x, dtype=np.float64)),
                            modality="state", source_layer="L2")]

    def step(self, ctx: list[Latent]) -> list[Latent]:
        """One predictive step from the latest state latent with a zero action."""
        if not ctx or self._A is None:
            return []
        return self.rollout(ctx[-1], [np.zeros(self._B.shape[1])])

    def health(self) -> dict:
        return {"layer": self.layer_id, "built": True,
                "predictive": "linear-transition (DreamerV3 injectable)",
                "causal": "linear-Gaussian NCM",
                "skeleton_vars": list(self.skeleton.variables) if self.skeleton else [],
                "gate": "G2 do(x) vs simulation + out-of-skeleton flag"}

    # --- factor <-> latent ---------------------------------------------------
    def _embed(self, factors: np.ndarray) -> np.ndarray:
        f = np.asarray(factors, dtype=np.float64).ravel()
        if f.shape[0] != self._d_factors:
            raise ValueError(f"expected {self._d_factors} factors, got {f.shape[0]}")
        return f @ self._E  # (LATENT_DIM,)

    def _project(self, z: np.ndarray) -> np.ndarray:
        return self._E @ np.asarray(z, dtype=np.float64)  # (d_factors,)
