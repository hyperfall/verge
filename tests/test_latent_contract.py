"""The latent-contract CI suite (verge-engineering.md §2, §10 step 1).

Nothing above L1 starts until this is green. All three invariants run on synthetic
latents, on CPU, with no GPU and no heavy deps.
"""
from __future__ import annotations

import numpy as np
import pytest

from verge.latent import (
    HALT_ROLLBACK,
    LATENT_DIM,
    OK,
    SLOW_ANCHOR,
    BindingProbe,
    DriftMonitor,
    Latent,
    binding_distance,
    content_key,
    effective_rank,
    make_latent,
    vicreg_loss,
)


def _rand_latent(rng, *, modality="concept", source="L1") -> Latent:
    return make_latent(rng.standard_normal(LATENT_DIM), modality=modality, source_layer=source)


# --- contract basics --------------------------------------------------------

def test_latent_is_unit_norm_and_correct_dim():
    rng = np.random.default_rng(0)
    l = _rand_latent(rng)
    assert l.z.shape == (LATENT_DIM,)
    assert l.is_unit_norm()
    assert l.key == content_key(l.z)


def test_latent_rejects_bad_shape_and_confidence():
    with pytest.raises(ValueError):
        Latent(z=np.ones(3, dtype=np.float32), modality="text", source_layer="L1",
               confidence=1.0, key=b"")
    with pytest.raises(ValueError):
        make_latent(np.ones(LATENT_DIM), modality="text", source_layer="L1", confidence=2.0)
    with pytest.raises(ValueError):
        make_latent(np.ones(LATENT_DIM), modality="not-a-modality", source_layer="L1")


def test_latent_is_frozen():
    rng = np.random.default_rng(1)
    l = _rand_latent(rng)
    with pytest.raises(Exception):
        l.confidence = 0.5  # type: ignore[misc]


# --- Invariant 1: no collapse ----------------------------------------------

def test_effective_rank_floor_distinguishes_collapse_from_diversity():
    rng = np.random.default_rng(2)
    diverse = np.stack([_rand_latent(rng).z for _ in range(64)])
    # Collapsed batch: every latent points the same way (+ tiny jitter).
    base = rng.standard_normal(LATENT_DIM)
    collapsed = np.stack([
        make_latent(base + 1e-3 * rng.standard_normal(LATENT_DIM),
                    modality="concept", source_layer="L1").z
        for _ in range(64)
    ])
    er_diverse = effective_rank(diverse)
    er_collapsed = effective_rank(collapsed)
    assert er_collapsed < 2.0, er_collapsed
    assert er_diverse > 10.0, er_diverse
    # The CI floor the harness gates on:
    FLOOR = 5.0
    assert er_diverse > FLOOR > er_collapsed


def test_vicreg_regularizer_fires_on_collapse():
    rng = np.random.default_rng(3)
    diverse = np.stack([_rand_latent(rng).z for _ in range(64)])
    base = rng.standard_normal(LATENT_DIM)
    base = base / np.linalg.norm(base)
    collapsed = np.stack([base for _ in range(64)])
    loss_collapsed = vicreg_loss(collapsed)
    loss_diverse = vicreg_loss(diverse)
    # Variance hinge must be much larger on the collapsed batch — the regularizer is
    # *present and active*, the contract's no-collapse guarantee.
    assert loss_collapsed["variance"] > loss_diverse["variance"]
    assert loss_collapsed["total"] > loss_diverse["total"]


# --- Invariant 2: cross-modal binding --------------------------------------

def _paired_concepts(rng, n=16, noise=0.05):
    """Synthetic vision/text encoders: same concept → same base direction + small
    modality-specific noise (the contrastive shaping target, simulated)."""
    pairs = []
    for _ in range(n):
        concept = rng.standard_normal(LATENT_DIM)
        v = make_latent(concept + noise * rng.standard_normal(LATENT_DIM),
                        modality="vision", source_layer="L1")
        t = make_latent(concept + noise * rng.standard_normal(LATENT_DIM),
                        modality="text", source_layer="L1")
        pairs.append((v, t))
    return pairs


def test_binding_probe_passes_when_modalities_align():
    rng = np.random.default_rng(4)
    pairs = _paired_concepts(rng, noise=0.05)
    report = BindingProbe(epsilon=0.25).evaluate(pairs)
    assert report["passed"], report
    assert report["mean_within"] < report["mean_across"]


def test_binding_probe_fails_when_modalities_unbound():
    rng = np.random.default_rng(5)
    # Independent random vision/text latents — no binding.
    pairs = [
        (make_latent(rng.standard_normal(LATENT_DIM), modality="vision", source_layer="L1"),
         make_latent(rng.standard_normal(LATENT_DIM), modality="text", source_layer="L1"))
        for _ in range(16)
    ]
    report = BindingProbe(epsilon=0.25).evaluate(pairs)
    assert not report["passed"], report
    # Random high-dim unit vectors are ~orthogonal → distance ~1.0.
    assert report["mean_within"] > 0.5


# --- Invariant 3: drift circuit-breaker ------------------------------------

def test_drift_breaker_ok_when_stable():
    rng = np.random.default_rng(6)
    ref = rng.standard_normal(LATENT_DIM)
    batch = [make_latent(ref + 0.01 * rng.standard_normal(LATENT_DIM),
                         modality="state", source_layer="L2") for _ in range(32)]
    mon = DriftMonitor()
    mon.anchor(np.stack([l.z for l in batch]))
    assert mon.check(batch)["decision"] == OK


def test_drift_breaker_slow_anchor_on_drift_plus_coherence_dip():
    rng = np.random.default_rng(7)
    ref = rng.standard_normal(LATENT_DIM)
    anchor_batch = [make_latent(ref + 0.01 * rng.standard_normal(LATENT_DIM),
                                modality="state", source_layer="L2") for _ in range(32)]
    mon = DriftMonitor(drift_warn=0.05, drift_halt=0.40, coherence_floor=0.8)
    mon.anchor(np.stack([l.z for l in anchor_batch]))
    # Moderately drifted direction AND decohered keys (key no longer matches content).
    drifted_dir = ref + 0.6 * rng.standard_normal(LATENT_DIM)
    drifted = []
    for _ in range(32):
        l = make_latent(drifted_dir + 0.05 * rng.standard_normal(LATENT_DIM),
                        modality="state", source_layer="L2")
        # simulate stale key → coherence dip
        drifted.append(Latent(z=l.z, modality=l.modality, source_layer=l.source_layer,
                              confidence=l.confidence, key=b"stale-key"))
    out = mon.check(drifted)
    assert out["decision"] == SLOW_ANCHOR, out
    assert out["coherence"] < 0.8


def test_drift_breaker_halt_rollback_on_severe_drift():
    rng = np.random.default_rng(8)
    ref = rng.standard_normal(LATENT_DIM)
    anchor_batch = [make_latent(ref, modality="state", source_layer="L2") for _ in range(8)]
    mon = DriftMonitor(drift_warn=0.05, drift_halt=0.30)
    mon.anchor(np.stack([l.z for l in anchor_batch]))
    # Orthogonal-ish direction → drift ~1.0 ≫ halt threshold.
    opposite = -ref + rng.standard_normal(LATENT_DIM)
    severe = [make_latent(opposite + 0.05 * rng.standard_normal(LATENT_DIM),
                          modality="state", source_layer="L2") for _ in range(8)]
    assert mon.check(severe)["decision"] == HALT_ROLLBACK
