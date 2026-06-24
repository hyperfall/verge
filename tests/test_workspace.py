"""Workspace bus: layers communicate only via Latents; broadcast/select + drift halt."""
from __future__ import annotations

import numpy as np
import pytest

from verge.latent import LATENT_DIM, DriftMonitor, LayerService, make_latent
from verge.workspace import DriftHalt, WorkspaceBus


def _latent(rng, conf=1.0, source="L3"):
    return make_latent(rng.standard_normal(LATENT_DIM), modality="concept",
                       source_layer=source, confidence=conf)


def test_register_requires_layer_id():
    bus = WorkspaceBus()
    svc = LayerService()
    svc.layer_id = ""
    with pytest.raises(ValueError):
        bus.register(svc)


def test_broadcast_logs_provenance_to_audit_sink():
    rng = np.random.default_rng(0)
    bus = WorkspaceBus()
    bus.broadcast([_latent(rng), _latent(rng)])
    trail = bus.audit_trail()
    events = [e["event"] for e in trail]
    assert events.count("broadcast") == 2
    assert all("source_layer" in e for e in trail if e["event"] == "broadcast")


def test_select_is_capacity_limited_by_salience():
    rng = np.random.default_rng(1)
    bus = WorkspaceBus(capacity=3)
    lats = [_latent(rng, conf=c) for c in (0.1, 0.9, 0.5, 0.7, 0.2)]
    bus.broadcast(lats)
    picked = bus.select()
    assert len(picked) == 3
    confs = [l.confidence for l in picked]
    assert confs == sorted(confs, reverse=True)
    assert min(confs) >= 0.5  # the three highest-salience survived


def test_bus_halts_on_severe_drift():
    rng = np.random.default_rng(2)
    ref = rng.standard_normal(LATENT_DIM)
    mon = DriftMonitor(drift_warn=0.05, drift_halt=0.30)
    mon.anchor(np.stack([make_latent(ref, modality="state", source_layer="L2").z]))
    bus = WorkspaceBus(monitor=mon)
    severe = [make_latent(-ref + 0.05 * rng.standard_normal(LATENT_DIM),
                          modality="state", source_layer="L2") for _ in range(4)]
    with pytest.raises(DriftHalt):
        bus.broadcast(severe)
