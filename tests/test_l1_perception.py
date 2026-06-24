"""L1 perception (real seam): projection head + cross-modal binding (G1)."""
from __future__ import annotations

import numpy as np

from verge.l1_perception import L1Perception, Signal, synth_pair
from verge.latent import LayerService


def _pairs(n=24):
    return [synth_pair(f"c{i}") for i in range(n)]


def test_encode_emits_unit_norm_modality_latent():
    L1 = L1Perception()
    lat = L1.encode(("vision", "cat"))[0]
    assert lat.modality == "vision" and lat.source_layer == "L1"
    assert lat.is_unit_norm() and lat.z.shape == (2048,)


def test_untrained_head_does_not_collapse():
    # near-isometric init: the no-collapse floor holds before any training
    L1 = L1Perception()
    report = L1.run_binding_gate(_pairs())
    assert report["effective_rank"] > 5.0


def test_binding_training_reduces_distance_and_passes_gate():
    L1 = L1Perception()
    pairs = _pairs()
    before = L1.run_binding_gate(pairs)
    assert not before["passed"]              # untrained: vision/text unbound
    curve = L1.train_binding(pairs, epochs=400, lr=0.05)
    after = L1.run_binding_gate(pairs)
    assert curve[-1] < curve[0]              # alignment loss went down
    assert after["mean_within"] < before["mean_within"]
    assert after["passed"]                   # G1: same concept lands within ε
    assert after["mean_within"] < after["mean_across"]
    assert after["effective_rank"] > 5.0     # VICReg held off collapse


def test_injectable_backbone_is_used():
    calls = {}

    def fake_backbone(modality, concept, dim=256, noise=0.15):
        calls[(modality, concept)] = True
        return np.ones(dim) * (hash(concept) % 7 + 1)

    L1 = L1Perception(backbone=fake_backbone)
    L1.encode(("vision", "z"))
    assert ("vision", "z") in calls


def test_is_layerservice_and_healthy():
    L1 = L1Perception()
    assert isinstance(L1, LayerService)
    assert L1.health()["built"] is True
