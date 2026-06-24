"""L2 world model (real): NCM do(x) vs baseline (G2) + out-of-skeleton flag + rollout."""
from __future__ import annotations

import numpy as np

from verge.latent import LayerService, make_latent
from verge.l2_world_model import CausalEdge, CausalSkeleton, L2WorldModel


def _chain_scm(n=4000, seed=0, verified=True):
    rng = np.random.default_rng(seed)
    X0 = rng.standard_normal(n)
    X1 = 2.0 * X0 + 0.3 * rng.standard_normal(n)
    X2 = -1.0 * X1 + 0.3 * rng.standard_normal(n)
    X3 = 0.5 * X0 + 0.3 * rng.standard_normal(n)
    data = {"X0": X0, "X1": X1, "X2": X2, "X3": X3}
    sk = CausalSkeleton(("X0", "X1", "X2", "X3"),
                        (CausalEdge("X0", "X1", verified),
                         CausalEdge("X1", "X2", verified),
                         CausalEdge("X0", "X3", verified)))
    return sk, data


def test_do_x_matches_simulation_and_beats_baseline():
    sk, data = _chain_scm()
    L2 = L2WorldModel().fit_causal(sk, data)
    r = L2.do({"X0": 1.0})
    truth = {"X1": 2.0, "X2": -2.0, "X3": 0.5}
    scm_err = np.mean([abs(r.values[k] - truth[k]) for k in truth])
    base_err = np.mean([abs(0.0 - truth[k]) for k in truth])  # ignore-intervention baseline
    assert scm_err < 0.1
    assert scm_err < 0.1 * base_err          # G2: do(x) >> baseline
    assert not r.out_of_skeleton


def test_out_of_skeleton_intervention_fires_flag():
    sk, data = _chain_scm()
    L2 = L2WorldModel().fit_causal(sk, data)
    r = L2.do({"X9": 1.0})                    # variable not in the skeleton
    assert r.out_of_skeleton
    assert r.overall_uncertainty > 1.0


def test_unverified_edge_routes_fire_flag():
    # same structure but the X1->X2 edge is a `prior`, not verified
    rng = np.random.default_rng(1)
    n = 3000
    X0 = rng.standard_normal(n); X1 = 2 * X0 + 0.3 * rng.standard_normal(n)
    X2 = -X1 + 0.3 * rng.standard_normal(n)
    sk = CausalSkeleton(("X0", "X1", "X2"),
                        (CausalEdge("X0", "X1", True), CausalEdge("X1", "X2", False)))
    L2 = L2WorldModel().fit_causal(sk, {"X0": X0, "X1": X1, "X2": X2})
    # do(X0) reaches X2 only through the unverified X1->X2 edge → flagged
    assert L2.do({"X0": 1.0}).out_of_skeleton


def test_counterfactual_abduction_action_prediction():
    sk, data = _chain_scm()
    L2 = L2WorldModel().fit_causal(sk, data)
    cf = L2.ncm.counterfactual({"X0": 0.0, "X1": 0.0, "X2": 0.0, "X3": 0.0}, {"X0": 1.0})
    assert abs(cf["X2"] - (-2.0)) < 0.1       # had X0 been 1, X2 would be ≈ -2


def test_learned_transition_rollout():
    rng = np.random.default_rng(2)
    ds, da, n = 4, 2, 2000
    A = rng.standard_normal((ds, ds)) * 0.3
    B = rng.standard_normal((ds, da)) * 0.5
    S = rng.standard_normal((n, ds)); Ac = rng.standard_normal((n, da))
    Sn = S @ A.T + Ac @ B.T
    L2 = L2WorldModel().fit_transition(S, Ac, Sn)
    s0 = make_latent(L2._embed(np.ones(ds)), modality="state", source_layer="L2")
    roll = L2.rollout(s0, [np.zeros(da), np.ones(da)])
    assert len(roll) == 2 and all(l.is_unit_norm() for l in roll)
    # one-step prediction matches the true linear dynamics in DIRECTION (latents are
    # unit-norm by contract, so scale is normalized away; the internal state keeps it).
    pred1 = L2._project(roll[0].z)
    expected = A @ np.ones(ds)
    cos = np.dot(pred1, expected) / (np.linalg.norm(pred1) * np.linalg.norm(expected))
    assert cos > 0.999


def test_is_layerservice():
    sk, data = _chain_scm()
    L2 = L2WorldModel().fit_causal(sk, data)
    assert isinstance(L2, LayerService) and L2.health()["built"] is True
