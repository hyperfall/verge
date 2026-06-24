"""L3 GRPO/DAPO engine swap (verge-engineering.md §3 upgrade 1; spec §5.4).

Validates the new loop on CPU: the reward is the Ring-0 verifier and nothing else, the
TRL reward callable has the right shape, DAPO knobs are the defaults, MATH/GSM8K loaders
are wired, and the mock GRPO engine runs end-to-end through the §7 harness. The real
GPU/TRL path is import-clean here and exercised only behind optional deps.
"""
from __future__ import annotations

import sys

import pytest

from verge.eval import evaluate
from verge.l3_reasoner.data import _extract_boxed, _gsm8k_answer, load_dataset_problems
from verge.l3_reasoner.grpo import (
    GRPOSettings,
    build_grpo_mock_adapter,
    make_trl_reward,
)


# --- the reward is the Ring-0 verifier, in TRL's callable shape -------------

def test_trl_reward_is_verifier_only_raw_completions():
    f = make_trl_reward()
    out = f(["q", "q"], ["work #### 72", "work #### 71"], answer=["72", "72"])
    assert out == [1.0, 0.0]


def test_trl_reward_handles_chat_format_completions():
    f = make_trl_reward()
    completions = [[{"role": "assistant", "content": "so #### 100"}]]
    assert f(["q"], completions, answer=["100"]) == [1.0]


# --- DAPO defaults: never vanilla GRPO from base ----------------------------

def test_dapo_defaults_are_set():
    s = GRPOSettings()
    assert s.epsilon_high > s.epsilon       # clip-higher (decoupled)
    assert s.dynamic_sampling is True        # drop zero-advantage groups
    assert s.scale_rewards is False          # length-bias fix
    assert s.beta == 0.0                     # no KL
    assert s.lr == 2e-6                       # calibrated, not the 1e-5 default


def test_to_trl_kwargs_filters_to_installed_trl():
    trl = pytest.importorskip("trl")  # only when verge[l3] present
    kwargs = GRPOSettings().to_trl_kwargs()
    import inspect
    accepted = set(inspect.signature(trl.GRPOConfig.__init__).parameters)
    assert set(kwargs).issubset(accepted)
    assert kwargs.get("learning_rate") == 2e-6


# --- modules import clean (no heavy deps at import time) --------------------

def test_grpo_path_imports_without_heavy_deps():
    import verge.l3_reasoner.data  # noqa: F401
    import verge.l3_reasoner.grpo  # noqa: F401
    import verge.l3_reasoner.search  # noqa: F401
    assert not ({"torch", "transformers", "trl", "vllm"} & set(sys.modules))


# --- data wiring ------------------------------------------------------------

def test_extract_boxed_balances_nested_braces():
    sol = r"thus the answer is $\boxed{\frac{1}{2}}$ done"
    assert _extract_boxed(sol) == r"\frac{1}{2}"
    assert _extract_boxed("no box here") is None


def test_gsm8k_answer_extraction():
    assert _gsm8k_answer("blah\n#### 42") == "42"


def test_unknown_dataset_rejected():
    with pytest.raises(ValueError):
        load_dataset_problems("not-a-dataset", 10, 10)


# --- the mock GRPO loop runs end-to-end through the harness -----------------

def test_grpo_mock_runs_end_to_end():
    layer, frozen_test = build_grpo_mock_adapter(rounds=3, k=8, n_train=200, n_test=150)
    report = evaluate(layer, seeds=(0, 1, 2), rounds=3, frozen_test=frozen_test,
                      ablations={}, preregister={"min_slope": 0.0, "beats_ablation": None})
    assert all(len(c) == 4 for c in report.main.curves.values())
    # bounded amplifier: final improves over baseline but stays a finite consolidation
    assert report.main.final_mean() >= report.main.baseline_mean()


def test_grpo_mock_dynamic_sampling_is_active():
    # With dynamic sampling on, the mock still trains (kept set is non-empty early), and
    # the loop is deterministic per seed.
    layer, test = build_grpo_mock_adapter(rounds=2, k=8, n_train=200, n_test=120)
    c0 = layer.run_seed(0, rounds=2, frozen_test=test)
    c0b = layer.run_seed(0, rounds=2, frozen_test=test)
    assert c0 == c0b  # deterministic
    assert c0[-1] >= c0[0]
