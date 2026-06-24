"""The layer-agnostic eval harness (verge-engineering.md §7)."""
from __future__ import annotations

import pytest

from verge.eval import dedupe_reasoning_paths, evaluate, fit_slope, pass_at_1


class _LinearArm:
    """A synthetic EvalProtocol arm: metric = base + slope*round + per-seed jitter."""

    def __init__(self, name, base, slope, jitter=0.0):
        self.name, self.base, self.slope, self.jitter = name, base, slope, jitter

    def run_seed(self, seed, *, rounds, frozen_test):
        import random
        rng = random.Random(seed)
        return [self.base + self.slope * r + (rng.random() - 0.5) * self.jitter
                for r in range(rounds + 1)]


def test_metrics_are_the_m1_primitives():
    assert pass_at_1([True, False, True, True]) == 0.75
    slope, lo, hi = fit_slope([0, 1, 2, 3], [0.0, 1.0, 2.0, 3.0])
    assert abs(slope - 1.0) < 1e-9 and lo <= 1.0 <= hi


def test_evaluate_requires_three_seeds():
    with pytest.raises(ValueError):
        evaluate(_LinearArm("x", 0.3, 0.0), seeds=(0, 1), rounds=3,
                 frozen_test=None, preregister={})


def test_evaluate_ships_on_rising_slope():
    report = evaluate(_LinearArm("rising", 0.3, 0.05), seeds=(0, 1, 2), rounds=4,
                      frozen_test=None, preregister={"min_slope": 0.0})
    assert report.decision == "SHIP"
    assert report.ci_low > 0.0


def test_evaluate_flat_when_ci_includes_zero():
    report = evaluate(_LinearArm("flat", 0.36, 0.0, jitter=0.02), seeds=(0, 1, 2),
                      rounds=4, frozen_test=None, preregister={"min_slope": 0.0})
    assert report.is_flat
    assert report.decision == "FLAT"


def test_bitter_lesson_gate_cuts_layer_that_loses_to_ablation():
    # main rises but its FINAL does not beat the ablation arm -> CUT (must earn its place)
    main = _LinearArm("layer", base=0.30, slope=0.02)
    abl = _LinearArm("scale_base_plus_retrieval", base=0.60, slope=0.0)
    report = evaluate(main, seeds=(0, 1, 2), rounds=4, frozen_test=None,
                      ablations={"scale_base_plus_retrieval": abl},
                      preregister={"min_slope": 0.0,
                                   "beats_ablation": "scale_base_plus_retrieval"})
    assert report.decision == "CUT"
    assert any("does NOT beat" in r for r in report.reasons)


def test_dedupe_reasoning_paths_available():
    from verifiers import Problem  # via the m1 bridge already on path
    p = Problem("p1", "q", "3")
    pairs = [(p, "a 1 2 3"), (p, "a 1 2 3"), (p, "b 4 5 6")]
    assert len(dedupe_reasoning_paths(pairs)) == 2
