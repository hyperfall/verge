"""The verification ladder (verge-engineering.md §3; spec Thread B).

A partially-ordered set of verifiers, each a `LayerService`-style check returning
`VerdictResult(passed, rung, confidence)`. The verge declines monotonically up the
ladder; only the EXACT rung is built (it wraps the Ring-0 verifier). The others are
typed stubs naming their open component — they raise `NotImplementedError` until built,
never silently pass.

    | Rung        | Verifier                                   | Domain                 |
    |-------------|--------------------------------------------|------------------------|
    | FORMAL      | Lean 4 / lean-dojo proof check             | math, logic            |
    | EXACT       | ExactMatchVerifier (BUILT)                 | arithmetic answers     |
    | EMPIRICAL   | sandboxed pytest / simulation              | code, physics rollouts |
    | CONSISTENCY | N-version agreement among formalizers      | autoformalization gate |
    | SOCIAL      | human-panel statistical validation         | L5 claims              |
"""
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from verge.ring0.verifiers.exact import Problem, verify


class Rung(enum.IntEnum):
    """Ordered by verifier strength (FORMAL strongest). The `verge` is how far up this
    ladder the signal can climb for a given claim."""

    FORMAL = 5
    EXACT = 4
    EMPIRICAL = 3
    CONSISTENCY = 2
    SOCIAL = 1


@dataclass(frozen=True)
class VerdictResult:
    passed: bool
    rung: Rung
    confidence: float  # in [0,1]; below EXACT this is genuinely uncertain


@runtime_checkable
class RungVerifier(Protocol):
    rung: Rung

    def check(self, problem: Problem, trace: str) -> VerdictResult: ...


class ExactRung:
    """The one built rung. Wraps the Ring-0 `ExactMatchVerifier`; deterministic,
    confidence 1.0 because the answer either matches or it does not."""

    rung = Rung.EXACT

    def check(self, problem: Problem, trace: str) -> VerdictResult:
        ok = verify(problem, trace)
        return VerdictResult(passed=ok, rung=Rung.EXACT, confidence=1.0)


class _StubRung:
    rung: Rung
    component: str

    def check(self, problem: Problem, trace: str) -> VerdictResult:
        raise NotImplementedError(
            f"{type(self).__name__} ({self.rung.name}) not built — open component: "
            f"{self.component}. Until built it must never silently pass a trace."
        )


class FormalRung(_StubRung):
    rung = Rung.FORMAL
    # TODO(L3): Lean 4 / lean-dojo proof check for math & logic claims.
    component = "Lean 4 / lean-dojo"


class EmpiricalRung(_StubRung):
    rung = Rung.EMPIRICAL
    # TODO(L3): sandboxed pytest / physics-rollout simulation (runs in the Wasm sandbox).
    component = "sandboxed pytest / simulation"


class ConsistencyRung(_StubRung):
    rung = Rung.CONSISTENCY
    # TODO(L3): N-version agreement among independent formalizers — the autoformalization
    # gate. A *vote*, never a self-confidence bet; residual correlated error is what the
    # verifier-independent Ring-0 audit exists to catch.
    component = "N-version independent-formalizer agreement"


class SocialRung(_StubRung):
    rung = Rung.SOCIAL
    # TODO(L5): human-panel statistical validation of L5 social claims.
    component = "human-panel statistical validation"


class MathEquivalenceRung(_StubRung):
    """Deterministic symbolic-equivalence check for general MATH answers (fractions,
    surds, symbolic) that the numeric EXACT verifier cannot read. Strength-equivalent to
    EXACT but a SEPARATE verifier — never a modification of `ExactMatchVerifier`. Needed
    for the §5.4 MATH headroom run beyond its numeric slice."""

    rung = Rung.EXACT
    # TODO(L3): sympy `simplify(a - b) == 0` equivalence (verge[world] ships sympy via
    # pyro, or add sympy directly). Independent unit review before it joins Ring 0.
    component = "sympy math-equivalence checker"


@dataclass
class VerificationLadder:
    """The partially-ordered verifier set. `verify_best` tries rungs strongest-first
    among those applicable, returning the highest-rung verdict that does not raise."""

    rungs: tuple[RungVerifier, ...]

    def by_rung(self, rung: Rung) -> RungVerifier:
        for v in self.rungs:
            if v.rung == rung:
                return v
        raise KeyError(f"no verifier for rung {rung}")

    def exact(self) -> ExactRung:
        return self.by_rung(Rung.EXACT)  # type: ignore[return-value]


def default_ladder() -> VerificationLadder:
    """The ladder as shipped: EXACT built, the rest typed stubs in strength order."""
    return VerificationLadder(rungs=(
        FormalRung(), ExactRung(), EmpiricalRung(), ConsistencyRung(), SocialRung(),
    ))
