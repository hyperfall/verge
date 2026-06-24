"""L1 perception (REAL seam) — projection head + cross-modal binding.

The spec's rule (verge-engineering.md §4; spec §3 L1): **do not train an encoder.** Load a
pretrained backbone (V-JEPA 2 for video/control, DINOv2/3 for stills), freeze it, and
train only a small **projection head** into `LATENT_DIM` (unit-norm, VICReg-regularized)
plus a **cross-modal contrastive binding loss** so the same concept via vision and via
text lands at proximate latent points (the G1 gate).

This module implements exactly that seam, on CPU, in numpy:
  - the backbone is an **injectable callable** `features = backbone(signal)`. The default
    is a deterministic synthetic feature map so the layer runs and tests with no GPU/weights;
    inject a real V-JEPA 2 / DINOv2 forward to use pretrained perception.
  - the projection head (`verge._numpy_nn.Linear` → unit-norm) is the only learned part.
  - `train_binding` reduces paired-modality distance with an explicit alignment + VICReg
    objective (standard backprop), and `run_binding_gate` is the G1 probe.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from verge._numpy_nn import Linear, unit_norm, vicreg_variance_push
from verge.latent import (
    LATENT_DIM,
    BindingProbe,
    Latent,
    LayerService,
    content_key,
    effective_rank,
    make_latent,
    vicreg_loss,
)


@dataclass
class Signal:
    """A backbone's output for one observation: the frozen encoder's features plus the
    modality and an optional concept key (provenance / binding pairing)."""

    modality: str            # "vision" | "text" | "audio" | "state"
    features: np.ndarray     # backbone output (any dim; the head maps it to LATENT_DIM)
    key: bytes | None = None


def default_backbone(modality: str, concept: str, dim: int = 256,
                     noise: float = 0.15) -> np.ndarray:
    """Deterministic synthetic 'frozen backbone'. Same concept → a shared latent factor
    across modalities (so binding is *learnable*), plus modality-specific structure (so it
    is not already aligned). Stands in for V-JEPA 2 / DINOv2 features on CPU."""
    cseed = int.from_bytes(hashlib.sha256(concept.encode()).digest()[:8], "little")
    mseed = int.from_bytes(hashlib.sha256(modality.encode()).digest()[:8], "little")
    shared = np.random.default_rng(cseed).standard_normal(dim)
    # a fixed per-modality linear distortion + concept-conditioned modality noise
    distort = np.random.default_rng(mseed).standard_normal((dim, dim)) / np.sqrt(dim)
    mnoise = np.random.default_rng((cseed ^ mseed) & 0xFFFFFFFF).standard_normal(dim)
    return shared @ distort.T + noise * mnoise


@dataclass
class L1Perception(LayerService):
    layer_id: str = "L1"
    latent_dim: int = LATENT_DIM
    vicreg_gamma: float = 1.0
    binding: BindingProbe = field(default_factory=lambda: BindingProbe(epsilon=0.25))
    backbone: Callable[[str, str], np.ndarray] | None = None  # injectable; default synthetic
    _heads: dict = field(default_factory=dict, repr=False)
    _seed: int = 0

    # --- the learned projection head (one per modality, lazily sized) ---------
    def _head(self, modality: str, d_in: int) -> Linear:
        if modality not in self._heads:
            # seed per modality for reproducibility; near-isometric init = no collapse
            self._heads[modality] = Linear(d_in, self.latent_dim,
                                           seed=self._seed + hash(modality) % 1000)
        return self._heads[modality]

    def _project(self, sig: Signal) -> np.ndarray:
        return self._head(sig.modality, sig.features.shape[0]).forward(sig.features)[0]

    # --- LayerService -------------------------------------------------------
    def encode(self, x) -> list[Latent]:
        """Encode a Signal (or a (modality, concept) tuple via the default backbone)."""
        sig = self._as_signal(x)
        z = unit_norm(self._project(sig))
        return [Latent(z=z.astype(np.float32), modality=sig.modality, source_layer="L1",
                       confidence=1.0, key=sig.key or content_key(z.astype(np.float32)))]

    def step(self, ctx: list[Latent]) -> list[Latent]:
        """The JEPA loop: predict masked/future representations of itself. Here the
        predictive target is the running mean of context (a minimal self-prediction);
        a real V-JEPA 2 predictor head replaces this once the backbone is wired."""
        if not ctx:
            return []
        pred = unit_norm(np.mean([l.z for l in ctx], axis=0))
        return [make_latent(pred, modality="concept", source_layer="L1",
                            confidence=float(np.mean([l.confidence for l in ctx])))]

    def health(self) -> dict:
        return {"layer": self.layer_id, "built": True,
                "regularizer": "VICReg", "heads": sorted(self._heads),
                "backbone": "injectable (V-JEPA2/DINOv2); default=synthetic",
                "gate": "G1 binding probe"}

    # --- binding training (the only spec-mandated L1 training) ----------------
    def train_binding(self, pairs: list[tuple[Signal, Signal]], *, epochs: int = 200,
                      lr: float = 0.05, vicreg_weight: float = 0.01,
                      norm_weight: float = 0.1) -> list[float]:
        """Pull paired (vision, text) projections together (alignment) while VICReg keeps
        the head off the collapsed manifold and a unit-norm anchor keeps magnitudes bounded.
        Full-batch updates per epoch (stable). Returns the mean binding-distance curve."""
        if not pairs:
            return []
        vm, tm = pairs[0][0].modality, pairs[0][1].modality
        Xv = np.stack([v.features for v, _ in pairs])
        Xt = np.stack([t.features for _, t in pairs])
        hv = self._head(vm, Xv.shape[1])
        ht = self._head(tm, Xt.shape[1])
        curve = []
        for _ in range(epochs):
            Yv, Yt = hv.forward(Xv), ht.forward(Xt)
            mid = 0.5 * (Yv + Yt)                            # detached alignment target
            # alignment + unit-norm anchor (keeps ||y||≈1 so nothing explodes)
            nv = np.linalg.norm(Yv, axis=1, keepdims=True)
            nt = np.linalg.norm(Yt, axis=1, keepdims=True)
            dYv = (Yv - mid) + norm_weight * (nv - 1.0) * Yv / np.maximum(nv, 1e-9)
            dYt = (Yt - mid) + norm_weight * (nt - 1.0) * Yt / np.maximum(nt, 1e-9)
            # VICReg variance over the FULL stacked batch (stable, anti-collapse)
            stacked = np.vstack([Yv, Yt])
            dvar = vicreg_variance_push(stacked, self.vicreg_gamma)
            dYv = dYv + vicreg_weight * dvar[: len(pairs)]
            dYt = dYt + vicreg_weight * dvar[len(pairs):]
            hv.apply_output_grad(Xv, dYv, lr)
            ht.apply_output_grad(Xt, dYt, lr)
            within = [1.0 - float(np.dot(unit_norm(Yv[i]), unit_norm(Yt[i])))
                      for i in range(len(pairs))]
            curve.append(float(np.mean(within)))
        return curve

    def run_binding_gate(self, pairs: list[tuple[Signal, Signal]]) -> dict:
        encoded = [(self.encode(v)[0], self.encode(t)[0]) for v, t in pairs]
        report = self.binding.evaluate(encoded)
        zs = np.stack([l.z for pr in encoded for l in pr])
        report["effective_rank"] = effective_rank(zs)
        report["vicreg"] = vicreg_loss(zs)["total"]
        return report

    # --- helpers ------------------------------------------------------------
    def _as_signal(self, x) -> Signal:
        if isinstance(x, Signal):
            return x
        if isinstance(x, tuple) and len(x) == 2:  # (modality, concept) via default backbone
            modality, concept = x
            bb = self.backbone or default_backbone
            return Signal(modality=modality, features=np.asarray(bb(modality, concept)),
                          key=_concept_key(concept))
        raise TypeError("L1.encode expects a Signal or a (modality, concept) tuple")


def _concept_key(concept: str) -> bytes:
    return hashlib.sha256(("concept:" + str(concept)).encode()).digest()


def synth_pair(concept: str, *, dim: int = 256, noise: float = 0.15,
               backbone: Callable = default_backbone) -> tuple[Signal, Signal]:
    """Build a (vision, text) Signal pair for the same concept — for tests/demos."""
    return (Signal("vision", np.asarray(backbone("vision", concept, dim, noise)), _concept_key(concept)),
            Signal("text", np.asarray(backbone("text", concept, dim, noise)), _concept_key(concept)))
