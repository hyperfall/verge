"""The L3 output contract: a verified skeleton + a fluent rendering (verge-engineering.md §3).

L3 emits a verified **skeleton** (the checkable claim graph) and a fluent **rendering**,
with a **faithfulness check** that rejects any rendering asserting something the skeleton
does not entail. The skeleton is what gets distilled and what enters memory; the
rendering is for humans.

The faithfulness check here is concrete and testable: the rendering's final answer (read
with the Ring-0 extractor) must equal the skeleton's verifier-confirmed answer. A
rendering that states a different answer than the skeleton entails is rejected — the
hallucinated-fluency failure mode the contract exists to stop.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from verge.ring0 import extract_final_answer, normalize_number


@dataclass(frozen=True)
class Claim:
    """One checkable node in the skeleton's claim graph."""

    text: str
    verified: bool          # confirmed by some rung of the verification ladder
    rung: str = "EXACT"


@dataclass(frozen=True)
class Skeleton:
    """The verified, checkable backbone of a solution. `answer` is the
    verifier-confirmed final answer; `claims` is the supporting claim graph. Only
    skeletons whose final answer is verified are distilled / written to memory."""

    answer: str
    claims: tuple[Claim, ...] = ()
    verified: bool = False

    def entailed_answer(self) -> str | None:
        return normalize_number(self.answer)


@dataclass
class Rendering:
    """The human-facing fluent text. Carries no authority of its own — it is only
    publishable if `faithful(skeleton, rendering)` holds."""

    text: str
    rejected: bool = False
    reason: str = ""
    meta: dict = field(default_factory=dict)


def faithful(skeleton: Skeleton, rendering: Rendering) -> bool:
    """True iff the rendering asserts nothing the skeleton does not entail.

    Concrete rule for the EXACT rung: the final answer the rendering states must match
    the skeleton's verified answer. (Higher rungs extend this to per-claim entailment
    once FORMAL/EMPIRICAL are built.)"""
    if not skeleton.verified:
        return False
    rendered = extract_final_answer(rendering.text)
    entailed = skeleton.entailed_answer()
    return rendered is not None and entailed is not None and rendered == entailed


def render_gate(skeleton: Skeleton, rendering: Rendering) -> Rendering:
    """Apply the faithfulness gate, marking the rendering rejected if it overclaims."""
    if not faithful(skeleton, rendering):
        rendering.rejected = True
        rendering.reason = (
            "rendering asserts an answer the skeleton does not entail"
            if skeleton.verified else "skeleton is unverified; nothing to render")
    return rendering
