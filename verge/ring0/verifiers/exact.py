"""The EXACT rung — `ExactMatchVerifier`, reused VERBATIM from m1-verified-reasoner.

This is the Ring-0 asset (verge-engineering.md §3): "do not touch it without an
independent test review." We import it; we do not copy or modify its logic (build
constraint §2.2). The `verge.ring0` namespace re-exports it so the rest of the stack
references the verifier only through Ring 0.
"""
from __future__ import annotations

from verge import _m1bridge

_m1bridge.ensure_on_path()

# Verbatim re-export from the measured, unedited m1-verified-reasoner/verifiers.py.
from verifiers import (  # noqa: E402  (path set up above)
    ExactMatchVerifier,
    Problem,
    extract_final_answer,
    normalize_number,
)

__all__ = ["ExactMatchVerifier", "Problem", "extract_final_answer", "normalize_number", "verify"]

# A module-level singleton + a thin functional alias, so callers (e.g. the L3 GRPO
# reward) can use `verify(problem, trace)` without re-instantiating. The instance is
# stateless and deterministic; this adds no learned state.
_VERIFIER = ExactMatchVerifier()


def verify(problem: Problem, trace: str) -> bool:
    """Deterministic, label-free V: (problem, trace) -> bool. The only trusted signal."""
    return _VERIFIER.verify(problem, trace)
