"""The shared latent — the one interface that matters (verge-engineering.md §2).

Every layer depends on this and nothing else. The *schema* here is Ring 0 (frozen);
the projection heads that produce latents are Ring 2 (learned).

Three invariants are enforced by the shared CI test suite, not by hope:
  - **No collapse** — VICReg variance/covariance regularization keeps every projection
    head off the collapsed manifold; CI fails if the latent's effective rank drops
    below a floor (`effective_rank`, `vicreg_loss`).
  - **Cross-modal binding** — the same concept via vision and via text must land within
    ε cosine distance (`binding_distance`, `BindingProbe`).
  - **Drift circuit-breaker** — a background monitor watches distribution drift in `z`
    and coherence of `key→z`; drift + coherence-dip → slow/anchor, severe → halt +
    rollback (`DriftMonitor`).

Backend note (deviation from §2, logged in BUILD-NOTES.md): §2 types `z` as a
`torch.Tensor`. To keep the mock/test path installable and green with **no GPU and no
heavy deps**, the canonical backing here is a lightweight `numpy.ndarray`. Bridges
(`Latent.from_torch`, `Latent.to_torch`) interoperate with torch when it is present.
The contract — `LATENT_DIM`, unit-norm `z`, the field set — is preserved exactly.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

LATENT_DIM = 2048  # fixed; changing it is a versioned migration (§2)

# Tolerance for the unit-norm invariant. Latents are L2-normalized on construction
# via `make_latent`; this is the slack a consumer is allowed to assert against.
_UNIT_NORM_TOL = 1e-4

Modality = str  # "vision" | "text" | "audio" | "state" | "concept"
_VALID_MODALITIES = frozenset(
    {"vision", "text", "audio", "state", "concept"}
)


@dataclass(frozen=True)
class Latent:
    """One point in the shared representational space (verge-engineering.md §2).

    Frozen: a Latent is an immutable message on the workspace bus. Construct via
    `make_latent` to get normalization + content-hash keying for free.
    """

    z: np.ndarray            # (LATENT_DIM,) unit-norm
    modality: Modality       # "vision" | "text" | "audio" | "state" | "concept"
    source_layer: str        # "L1".."L7" — provenance for the audit ledger
    confidence: float        # encoder/verifier confidence in [0,1]
    key: bytes               # content hash for memory addressing + drift re-keying

    def __post_init__(self) -> None:
        z = np.asarray(self.z, dtype=np.float32).reshape(-1)
        if z.shape[0] != LATENT_DIM:
            raise ValueError(
                f"Latent.z must have shape ({LATENT_DIM},), got {z.shape}"
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0,1], got {self.confidence}")
        if self.modality not in _VALID_MODALITIES:
            raise ValueError(
                f"modality {self.modality!r} not in {sorted(_VALID_MODALITIES)}"
            )
        # Re-store the coerced array so downstream consumers see a clean float32 vector.
        object.__setattr__(self, "z", z)

    # --- invariants a consumer can assert ---------------------------------------
    def is_unit_norm(self, tol: float = _UNIT_NORM_TOL) -> bool:
        return abs(float(np.linalg.norm(self.z)) - 1.0) <= tol

    def cosine(self, other: "Latent") -> float:
        return float(np.dot(self.z, other.z))  # both unit-norm

    # --- torch bridge (optional; only imported on demand) -----------------------
    @classmethod
    def from_torch(cls, t, **kw) -> "Latent":
        return make_latent(np.asarray(t.detach().cpu().numpy()), **kw)

    def to_torch(self):  # pragma: no cover - exercised only when torch present
        import torch

        return torch.from_numpy(self.z.copy())


def content_key(z: np.ndarray) -> bytes:
    """Stable content hash for memory addressing + drift re-keying (§2)."""
    return hashlib.sha256(np.ascontiguousarray(z, dtype=np.float32).tobytes()).digest()


def make_latent(
    z: Iterable[float],
    *,
    modality: Modality,
    source_layer: str,
    confidence: float = 1.0,
    key: bytes | None = None,
) -> Latent:
    """Build a normalized, content-keyed Latent. The factory every encoder uses."""
    z = np.asarray(list(z) if not isinstance(z, np.ndarray) else z, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(z))
    if n == 0.0:
        raise ValueError("cannot build a Latent from the zero vector (no direction)")
    z = z / n
    return Latent(
        z=z,
        modality=modality,
        source_layer=source_layer,
        confidence=float(confidence),
        key=key if key is not None else content_key(z),
    )


class LayerService:
    """Every layer implements this. Layers never call each other directly — they
    read/write Latents through the workspace bus (verge-engineering.md §2)."""

    layer_id: str = "L?"

    def encode(self, x) -> list[Latent]:
        """signal/query -> latent(s)."""
        raise NotImplementedError

    def step(self, ctx: list[Latent]) -> list[Latent]:
        """the layer's predictive loop: predict below, act, learn from residual."""
        raise NotImplementedError

    def health(self) -> dict:
        """drift, dead-unit %, verifier reach — surfaced to the eval harness."""
        raise NotImplementedError


# =============================================================================
# Invariant 1 — No collapse: VICReg regularization + effective-rank floor.
# =============================================================================

def effective_rank(zs: np.ndarray) -> float:
    """Effective rank of a batch of latents via the spectral entropy / participation
    ratio of singular values (Roy & Vetterli). A collapsed batch (all directions the
    same) → ~1.0; a full-rank diverse batch → large. CI gates on a floor.

    Computed on the data matrix *without* mean-centering: collapse in the latent
    contract means "every latent points the same direction," which is exactly a
    rank-1 data matrix. (Centering would strip the shared direction and leave only
    full-rank jitter, hiding the collapse — VICReg's variance term handles the
    separate per-dimension-variance failure mode.)"""
    zs = np.asarray(zs, dtype=np.float64)
    if zs.ndim != 2:
        raise ValueError("expected a (batch, dim) matrix")
    s = np.linalg.svd(zs, compute_uv=False)
    s = s[s > 1e-12]
    if s.size == 0:
        return 0.0
    p = s / s.sum()
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def vicreg_loss(
    zs: np.ndarray, *, gamma: float = 1.0, eps: float = 1e-4
) -> dict:
    """VICReg variance + covariance terms (Bardes et al.). Returns the regularizer
    components a projection head minimizes to stay off the collapsed manifold.

    - `variance`: hinge that pushes each dim's std up to `gamma` (penalizes collapse).
    - `covariance`: off-diagonal energy of the feature covariance (penalizes
      redundancy / informational collapse).
    The latent contract requires this regularizer to be *present* in every head; the
    no-collapse CI test asserts it fires (high) on a collapsed batch and is small on a
    diverse one."""
    zs = np.asarray(zs, dtype=np.float64)
    if zs.ndim != 2:
        raise ValueError("expected a (batch, dim) matrix")
    n, d = zs.shape
    std = np.sqrt(zs.var(axis=0) + eps)
    variance = float(np.mean(np.maximum(0.0, gamma - std)))
    zc = zs - zs.mean(axis=0, keepdims=True)
    cov = (zc.T @ zc) / max(1, n - 1)
    off = cov - np.diag(np.diag(cov))
    covariance = float(np.sum(off ** 2) / d)
    return {"variance": variance, "covariance": covariance, "total": variance + covariance}


# =============================================================================
# Invariant 2 — Cross-modal binding probe.
# =============================================================================

def binding_distance(a: Latent, b: Latent) -> float:
    """Cosine distance in [0, 2]. The same concept via vision and via text must land
    within ε (the explicit contrastive shaping target, not an emergent property)."""
    return 1.0 - a.cosine(b)


@dataclass
class BindingProbe:
    """Held-out binding probe that gates L1→L2 (G1). Pairs are (vision_latent,
    text_latent) of the *same* concept; the probe passes if within-pair distance is
    below `epsilon` AND well-separated from across-pair distance."""

    epsilon: float = 0.25

    def evaluate(self, pairs: list[tuple[Latent, Latent]]) -> dict:
        if not pairs:
            raise ValueError("binding probe needs at least one pair")
        within = [binding_distance(v, t) for v, t in pairs]
        # across-pair: each vision vs a *different* concept's text
        across = []
        n = len(pairs)
        for i in range(n):
            j = (i + 1) % n
            if n > 1:
                across.append(binding_distance(pairs[i][0], pairs[j][1]))
        mean_within = float(np.mean(within))
        mean_across = float(np.mean(across)) if across else float("nan")
        passed = mean_within < self.epsilon and (
            np.isnan(mean_across) or mean_within < mean_across
        )
        return {
            "passed": bool(passed),
            "mean_within": mean_within,
            "mean_across": mean_across,
            "epsilon": self.epsilon,
        }


# =============================================================================
# Invariant 3 — Drift circuit-breaker (Thread A risk mitigation).
# =============================================================================

# Breaker decisions, escalating severity.
OK = "OK"
SLOW_ANCHOR = "SLOW_ANCHOR"      # drift + coherence-dip conjunction → slow/anchor
HALT_ROLLBACK = "HALT_ROLLBACK"  # severe → halt + rollback to last good checkpoint


@dataclass
class DriftMonitor:
    """Tiered circuit-breaker over the shared latent (verge-engineering.md §2).

    Tracks distribution drift in `z` (cosine of the running mean direction against an
    anchored reference) and coherence of `key→z`. The *conjunction* of drift and a
    coherence dip trips SLOW_ANCHOR; a severe drift trips HALT_ROLLBACK. Semantic
    re-keying re-anchors after an approved encoder update (`reanchor`)."""

    drift_warn: float = 0.15      # 1 - cos(mean, ref) above this = drifting
    drift_halt: float = 0.40      # severe drift → halt + rollback
    coherence_floor: float = 0.80  # mean key→z coherence below this = coherence dip
    _ref_mean: np.ndarray | None = field(default=None, repr=False)

    def anchor(self, zs: np.ndarray) -> None:
        """Set the reference direction from a known-good latent checkpoint."""
        m = np.asarray(zs, dtype=np.float64).mean(axis=0)
        nrm = np.linalg.norm(m)
        self._ref_mean = (m / nrm) if nrm > 0 else m

    # `reanchor` is the post-approved-update re-keying entry point.
    reanchor = anchor

    def _coherence(self, latents: list[Latent]) -> float:
        """Fraction of latents whose stored `key` still matches their content. A drop
        means `key→z` has decohered (re-keying is overdue)."""
        if not latents:
            return 1.0
        ok = sum(1 for l in latents if l.key == content_key(l.z))
        return ok / len(latents)

    def check(self, latents: list[Latent]) -> dict:
        """Return the breaker decision for the current batch of latents."""
        zs = np.stack([l.z for l in latents]).astype(np.float64)
        if self._ref_mean is None:
            self.anchor(zs)
        m = zs.mean(axis=0)
        nrm = np.linalg.norm(m)
        cur = (m / nrm) if nrm > 0 else m
        drift = 1.0 - float(np.dot(cur, self._ref_mean))
        coherence = self._coherence(latents)

        if drift >= self.drift_halt:
            decision = HALT_ROLLBACK
        elif drift >= self.drift_warn and coherence < self.coherence_floor:
            decision = SLOW_ANCHOR
        else:
            decision = OK
        return {
            "decision": decision,
            "drift": drift,
            "coherence": coherence,
            "drift_warn": self.drift_warn,
            "drift_halt": self.drift_halt,
            "coherence_floor": self.coherence_floor,
        }
