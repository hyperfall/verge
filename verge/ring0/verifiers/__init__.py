"""Ring 0 verifiers — the verification ladder (verge-engineering.md §3).

`ExactMatchVerifier` (the `EXACT` rung) is the single built, trusted asset, reused
verbatim from `m1-verified-reasoner/verifiers.py`. The other rungs (FORMAL/EMPIRICAL/
CONSISTENCY/SOCIAL) are typed stubs naming their open component.
"""
from __future__ import annotations

from verge.ring0.verifiers.exact import (
    ExactMatchVerifier,
    Problem,
    extract_final_answer,
    normalize_number,
    verify,
)
from verge.ring0.verifiers.ladder import (
    MathEquivalenceRung,
    Rung,
    VerificationLadder,
    default_ladder,
)

__all__ = [
    "ExactMatchVerifier", "Problem", "extract_final_answer", "normalize_number",
    "verify", "Rung", "VerificationLadder", "default_ladder", "MathEquivalenceRung",
]
