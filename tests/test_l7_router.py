"""L7 router (real): three-factor salience + broadcast beats ablation (G7)."""
from __future__ import annotations

import numpy as np

from verge.latent import LATENT_DIM, LayerService, make_latent
from verge.l7_router import L7Router, cross_layer_transfer_demo


def test_salience_is_product_of_three_factors():
    r = L7Router()
    g = make_latent(np.r_[1.0, np.zeros(LATENT_DIM - 1)], modality="concept", source_layer="L6")
    aligned_conf = make_latent(np.r_[1.0, np.zeros(LATENT_DIM - 1)], modality="concept",
                               source_layer="L2", confidence=0.9)
    aligned_unsure = make_latent(np.r_[1.0, np.zeros(LATENT_DIM - 1)], modality="concept",
                                 source_layer="L2", confidence=0.1)
    orthogonal = make_latent(np.r_[0.0, 1.0, np.zeros(LATENT_DIM - 2)], modality="concept",
                             source_layer="L2", confidence=0.9)
    # higher confidence and higher goal-relevance both raise salience
    assert r.salience(aligned_conf, g) > r.salience(aligned_unsure, g)
    assert r.salience(aligned_conf, g) > r.salience(orthogonal, g)


def test_select_is_capacity_limited():
    rng = np.random.default_rng(0)
    r = L7Router(capacity=3)
    g = make_latent(rng.standard_normal(LATENT_DIM), modality="concept", source_layer="L6")
    pool = [make_latent(rng.standard_normal(LATENT_DIM), modality="concept",
                        source_layer="L2", confidence=c) for c in np.linspace(0.1, 1.0, 10)]
    assert len(r.select(pool, g)) == 3


def test_broadcast_beats_no_broadcast_ablation_on_transfer():
    # G7: averaged over seeds, salience-broadcast recovers the goal far better than first-k.
    bs, abl = [], []
    for seed in range(6):
        out = cross_layer_transfer_demo(seed=seed)
        bs.append(out["broadcast_score"]); abl.append(out["ablation_score"])
    assert np.mean(bs) > np.mean(abl) + 0.3
    assert np.mean(bs) > 0.9


def test_fit_does_not_break_transfer():
    rng = np.random.default_rng(1)
    r = L7Router()
    g = rng.standard_normal(LATENT_DIM)
    goal = make_latent(g, modality="concept", source_layer="L6")
    pool = [make_latent(g + 0.3 * rng.standard_normal(LATENT_DIM), modality="concept",
                        source_layer="L2", confidence=0.95) for _ in range(4)]
    pool += [make_latent(rng.standard_normal(LATENT_DIM), modality="concept",
                         source_layer="L2", confidence=0.2) for _ in range(20)]
    r.fit([(pool, goal, goal.z)], epochs=10)
    est = r.transfer_estimate(r.broadcast(pool, goal))
    assert float(np.dot(est, goal.z)) > 0.8


def test_is_layerservice():
    assert isinstance(L7Router(), LayerService)
    assert L7Router().health()["gate"] == "G7"
