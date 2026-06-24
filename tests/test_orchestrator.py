"""End-to-end orchestration: the seven layers compose over the workspace bus."""
from __future__ import annotations

import warnings

from verge.orchestrator import run_demo


def test_run_demo_end_to_end():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = run_demo(verbose=False)
    assert out["binding_passed"] is True
    assert out["broadcasts"] >= 12
    assert out["workspace_size"] == 7
    assert out["proposals"] >= 1
