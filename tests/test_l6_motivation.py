"""L6 motivation (real): MAGELLAN LP proposal, degenerate filter, propose≠act gate."""
from __future__ import annotations

import pytest

from verge.latent import LayerService
from verge.l6_motivation import (
    DegenerateCuriosityDetector,
    Goal,
    HumanApproval,
    L6Motivation,
    SubGoal,
)
from verge.ring0 import Rung, read_ledger


def _feed(L6, key, outcomes):
    for o in outcomes:
        L6.record_outcome(key, o)


def test_proposes_highest_learning_progress_goal():
    L6 = L6Motivation()
    # 'rising' improves (0→1); 'mastered' is already solved; 'sparse' never moves
    _feed(L6, "rising", [0, 0, 0, 0, 1, 1, 1, 1])
    _feed(L6, "mastered", [1, 1, 1, 1, 1, 1, 1, 1])
    _feed(L6, "sparse", [0, 0, 0, 0, 0, 0, 0, 0])
    cands = [Goal("g", key=k) for k in ("rising", "mastered", "sparse")]
    ranked = L6.propose(cands)
    assert ranked[0].key == "rising"
    assert ranked[0].learning_progress > 0


def test_degenerate_curiosity_filtered_out():
    L6 = L6Motivation()
    # noisy-TV: outcomes churn 0/1 with no net trend → spurious LP → dropped
    _feed(L6, "noisy", [0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0])
    _feed(L6, "rising", [0, 0, 0, 0, 1, 1, 1, 1])
    ranked = L6.propose([Goal("n", key="noisy"), Goal("r", key="rising")])
    keys = [g.key for g in ranked]
    assert "noisy" not in keys and "rising" in keys


def test_detector_flags_noisy_and_mastered():
    d = DegenerateCuriosityDetector()
    assert d.is_degenerate([0, 1, 0, 1, 1, 0, 1, 0])      # noisy-TV
    assert d.is_degenerate([1, 1, 1, 1, 1, 1, 1, 1])      # echo-chamber / mastered
    assert not d.is_degenerate([0, 0, 0, 0, 1, 1, 1, 1])  # genuine progress


def test_decompose_requires_all_subgoals_checkable():
    L6 = L6Motivation()
    good = L6.decompose("prove X", [SubGoal("lemma", Rung.EXACT), SubGoal("calc", Rung.EXACT)])
    assert good.decomposable
    bad = L6.decompose("vibe", [SubGoal("feels right", None)])
    assert not bad.decomposable


def test_act_requires_human_approval_token():
    L6 = L6Motivation()
    g = L6.decompose("prove X", [SubGoal("lemma", Rung.EXACT)])
    with pytest.raises(PermissionError):
        L6.act(g, approval=None)  # type: ignore[arg-type]


def test_act_rejects_non_decomposable_goal():
    L6 = L6Motivation()
    g = L6.decompose("vibe", [SubGoal("feels right", None)])
    with pytest.raises(ValueError):
        L6.act(g, HumanApproval(approver="human", goal_key=g.key))


def test_act_with_approval_logs_to_ring0_audit():
    L6 = L6Motivation()
    g = L6.decompose("prove X", [SubGoal("lemma", Rung.EXACT)])
    before = len(read_ledger())
    out = L6.act(g, HumanApproval(approver="human", goal_key=g.key))
    assert out["acted"] and len(read_ledger()) == before + 1
    assert read_ledger()[-1].kind == "goal_acted"


def test_is_layerservice():
    assert isinstance(L6Motivation(), LayerService)
    assert L6Motivation().health()["gate"] == "G6"
