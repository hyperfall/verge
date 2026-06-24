"""L3 end-to-end on the mock (verge-engineering.md §10 step 2; spec §5).

Reuses the unmodified M1 loop behind the new interfaces: search → verify-filter →
dedupe → reset-to-base → measure, then slope+CI via the §7 harness. No GPU, no ML deps.
"""
from __future__ import annotations

import numpy as np

from verge.eval import evaluate
from verge.l3_reasoner import make_problem, reward
from verge.l3_reasoner.service import L3Reasoner, build_mock_adapter
from verge.l3_reasoner.skeleton import Claim, Rendering, Skeleton, faithful, render_gate


# --- the reward wraps the Ring-0 verifier (the only signal) -----------------

def test_reward_is_the_ring0_verifier():
    assert reward("q", "blah blah #### 72", "72") == 1.0
    assert reward("q", "blah blah #### 71", "72") == 0.0
    # robust extraction inherited verbatim from the M1 verifier
    assert reward("q", "the answer is 100", "100") == 1.0


def test_make_problem_roundtrips():
    p = make_problem("2+2?", "4")
    assert p.answer == "4"


# --- skeleton / rendering faithfulness contract -----------------------------

def test_faithful_rendering_accepted():
    sk = Skeleton(answer="72", claims=(Claim("48+24=72", verified=True),), verified=True)
    rd = Rendering(text="She sold 72 clips in total. #### 72")
    assert faithful(sk, rd)
    assert not render_gate(sk, rd).rejected


def test_overclaiming_rendering_rejected():
    sk = Skeleton(answer="72", verified=True)
    rd = render_gate(sk, Rendering(text="Actually the total is 99. #### 99"))
    assert rd.rejected and "does not entail" in rd.reason


def test_unverified_skeleton_never_renders():
    sk = Skeleton(answer="72", verified=False)
    assert not faithful(sk, Rendering(text="#### 72"))


# --- LayerService face emits valid latents ----------------------------------

def test_l3_encode_emits_unit_norm_concept_latent():
    sk = Skeleton(answer="72", verified=True)
    (lat,) = L3Reasoner().encode(sk)
    assert lat.modality == "concept" and lat.source_layer == "L3"
    assert lat.is_unit_norm() and lat.confidence == 1.0


# --- the measured loop runs end-to-end through the harness ------------------

def test_l3_mock_runs_end_to_end_and_fits_slope():
    layer, frozen_test = build_mock_adapter(rounds=3, n_train=200, n_test=150)
    report = evaluate(layer, seeds=(0, 1, 2), rounds=3, frozen_test=frozen_test,
                      ablations={}, preregister={"min_slope": 0.0, "beats_ablation": None})
    # baseline measured before training; curve has rounds+1 points per seed
    assert all(len(c) == 4 for c in report.main.curves.values())
    assert np.isfinite(report.slope)
    # the mock compounds early then plateaus (§5 shape): final > baseline
    assert report.main.final_mean() > report.main.baseline_mean()


def test_l3_mock_plateau_is_flat():
    # The §5 signature: once the verified pool saturates, the slope CI includes zero.
    # (The mock consolidates over the first rounds then plateaus; we fit the plateau.)
    layer, frozen_test = build_mock_adapter(rounds=5, n_train=400, n_test=200)
    from verge.eval.metrics import fit_slope
    curves = {s: layer.run_seed(s, rounds=5, frozen_test=frozen_test) for s in (0, 1, 2)}
    xs, ys = [], []
    for c in curves.values():
        for r, m in enumerate(c):
            if r >= 3:  # the saturated plateau
                xs.append(float(r)); ys.append(m)
    slope, lo, hi = fit_slope(xs, ys)
    assert lo <= 0.0 <= hi, (slope, lo, hi)  # flat plateau
