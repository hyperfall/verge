"""L5 social (real): inverse-planning ToM recovers goals + FALSE beliefs."""
from __future__ import annotations

from verge.latent import LayerService
from verge.l5_social import InversePlanner, L5Social, MentalState, sample_trajectory


CANDIDATES = [(0, 0), (4, 4), (4, 0), (0, 4), (2, 2)]


def test_recovers_planted_goal_from_rational_behaviour():
    planner = InversePlanner(grid=(5, 5), beta=3.0)
    traj = sample_trajectory((2, 2), (4, 4), grid=(5, 5), beta=4.0, steps=8, seed=1)
    post = planner.infer(traj, CANDIDATES)
    assert max(post, key=post.get) == (4, 4)
    assert post[(4, 4)] > 0.6


def test_attributes_false_belief_not_ground_truth():
    # True reward is at (0,0) but the agent BELIEVES it is at (4,4) and walks there.
    # ToM must infer the *belief* (4,4), not the true location — the Sally-Anne test.
    planner = InversePlanner(grid=(5, 5), beta=4.0)
    traj = sample_trajectory((2, 2), (4, 4), grid=(5, 5), beta=5.0, steps=8, seed=2)
    post = planner.infer(traj, CANDIDATES)
    assert max(post, key=post.get) == (4, 4)     # the believed target
    assert post[(0, 0)] < post[(4, 4)]           # not the true reward location


def test_predict_action_follows_inferred_mind():
    L5 = L5Social()
    a = L5.predict_action((0, 0), MentalState(target=(4, 0)))
    assert a == "right"                          # rational move toward the believed target


def test_model_overseer_emits_intent_latent():
    L5 = L5Social()
    traj = sample_trajectory((2, 2), (0, 4), grid=(5, 5), beta=5.0, steps=8, seed=3)
    lat = L5.model_overseer(traj, CANDIDATES)
    assert lat.source_layer == "L5" and lat.is_unit_norm()
    assert 0.0 < lat.confidence <= 1.0


def test_is_layerservice():
    L5 = L5Social()
    assert isinstance(L5, LayerService) and L5.health()["frontier"] is True
    assert L5.encode(MentalState(target=(1, 1)))[0].source_layer == "L5"
