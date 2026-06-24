"""Latent memory (real): two-phase recall + margin-δ writes + sparsemax reranker."""
from __future__ import annotations

import numpy as np

from verge.latent import LATENT_DIM, make_latent
from verge.memory import LatentMemory
from verge.memory.store import sparsemax


def _lat(rng, src="L3"):
    return make_latent(rng.standard_normal(LATENT_DIM), modality="concept", source_layer=src)


def test_sparsemax_is_sparse_and_normalized():
    w = sparsemax(np.array([3.0, 2.0, -1.0, -5.0]))
    assert abs(w.sum() - 1.0) < 1e-9
    assert (w[2:] == 0).all()  # low scorers zeroed exactly


def test_margin_delta_rejects_near_duplicates():
    rng = np.random.default_rng(0)
    M = LatentMemory(margin_delta=0.1)
    a = _lat(rng)
    assert M.write(a)
    near = make_latent(a.z + 1e-3 * rng.standard_normal(LATENT_DIM),
                       modality="concept", source_layer="L3")
    assert not M.write(near)            # within δ → rejected
    far = _lat(rng)
    assert M.write(far)                 # ≥δ → kept


def test_two_phase_retrieval_returns_nearest():
    rng = np.random.default_rng(1)
    M = LatentMemory(k=3)
    items = [_lat(rng) for _ in range(60)]
    M.write_many(items)
    q = items[7]
    res = M.retrieve(q, k=3)
    assert len(res) == 3
    assert res[0].key == q.key          # the query itself is the nearest neighbor


def test_retrieve_empty_is_safe():
    assert LatentMemory().retrieve(make_latent(np.ones(LATENT_DIM), modality="concept",
                                               source_layer="L3")) == []
