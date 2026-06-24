"""The free CPU mock path (verge-engineering.md §7) — validate the full pipeline before
any GPU spend. Builds synthetic `Problem`s for the M1 mock simulator, which models the
one dynamic that matters: single-shot skill rises when distilled on verified, deduped
traces, and search reaches a little past current skill. No ML deps.
"""
from __future__ import annotations

import random

from verge import _m1bridge

_m1bridge.ensure_on_path()

from verifiers import Problem  # noqa: E402


def make_mock_problems(n_train: int = 400, n_test: int = 300, seed: int = 1234):
    """Synthetic train/test splits for the mock simulator (mirrors run.py `_mock_data`)."""
    rng = random.Random(seed)
    train = [Problem(f"tr{i}", f"mock {i}", str(rng.randint(1, 99))) for i in range(n_train)]
    test = [Problem(f"te{i}", f"mocktest {i}", str(rng.randint(1, 99))) for i in range(n_test)]
    return train, test
