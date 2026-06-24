"""Cross-layer contract tests — every layer implements LayerService and is now BUILT.

(Previously these layers were stubs; they are implemented per-layer in test_l1..test_l7.
This file checks the uniform contract that holds across all of them.)
"""
from __future__ import annotations

import pytest

from verge.latent import LayerService
from verge.l1_perception import L1Perception
from verge.l2_world_model import CausalEdge, CausalSkeleton, L2WorldModel
from verge.l4_plasticity import L4Plasticity
from verge.l5_social import L5Social
from verge.l6_motivation import Goal, HumanApproval, L6Motivation, SubGoal
from verge.l7_router import L7Router
from verge.memory import LatentMemory
from verge.ring0 import Rung


def _l2_built():
    import numpy as np
    rng = np.random.default_rng(0)
    X0 = rng.standard_normal(500)
    sk = CausalSkeleton(("X0", "X1"), (CausalEdge("X0", "X1", True),))
    return L2WorldModel().fit_causal(sk, {"X0": X0, "X1": 2 * X0 + 0.1 * rng.standard_normal(500)})


ALL_LAYERS = [L1Perception(), _l2_built(), L4Plasticity(), L5Social(),
              L6Motivation(), L7Router()]


@pytest.mark.parametrize("layer", ALL_LAYERS, ids=lambda l: l.layer_id)
def test_implements_layerservice_and_is_built(layer):
    assert isinstance(layer, LayerService)
    assert layer.layer_id in {"L1", "L2", "L4", "L5", "L6", "L7"}
    assert layer.health()["built"] is True


def test_memory_is_built():
    assert LatentMemory().health()["built"] is True


def test_l6_act_is_type_gated():
    # The autonomy boundary is structural: act() rejects a missing approval token.
    L6 = L6Motivation()
    g = L6.decompose("prove X", [SubGoal("lemma", Rung.EXACT)])
    with pytest.raises(PermissionError):
        L6.act(g, approval=None)  # type: ignore[arg-type]
