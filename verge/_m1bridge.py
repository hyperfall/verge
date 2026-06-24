"""Locate the untouched `m1-verified-reasoner/` and make its modules importable.

The measured L3 (`verifiers.py`, `metrics.py`, `expert_iteration.py`, `mock_model.py`,
...) uses bare top-level imports (`from verifiers import ...`). We import from it; we do
not fork or edit it (build constraint §2.6). This shim finds the directory wherever it
lives — repo root per `verge-engineering.md` §9, or under `Spec & Details/` as shipped —
and prepends it to `sys.path` exactly once, so the bare imports resolve.
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache

_CANDIDATE_SUBPATHS = (
    "m1-verified-reasoner",
    os.path.join("Spec & Details", "m1-verified-reasoner"),
)


@lru_cache(maxsize=1)
def m1_dir() -> str:
    """Absolute path to the m1-verified-reasoner directory."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
    # Walk up a couple levels in case verge/ is nested.
    roots = [here, os.path.dirname(here)]
    for root in roots:
        for sub in _CANDIDATE_SUBPATHS:
            cand = os.path.join(root, sub)
            if os.path.isfile(os.path.join(cand, "verifiers.py")):
                return cand
    raise FileNotFoundError(
        "Could not locate m1-verified-reasoner/ (looked for verifiers.py under "
        f"{_CANDIDATE_SUBPATHS} relative to {roots}). It must remain importable, unedited."
    )


def ensure_on_path() -> str:
    d = m1_dir()
    if d not in sys.path:
        sys.path.insert(0, d)
    return d
