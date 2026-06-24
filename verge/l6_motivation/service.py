"""L6 motivation service (REAL seam) — propose() autonomous, act() human-gated by type.

`propose()` ranks candidate goals by predicted learning progress (MAGELLAN), filtering
degenerate-curiosity traps — this is autonomous. `decompose()` checks a goal is 100%
verifier-checkable (G6), bounded by L3's reach. `act()` is side-effecting and requires a
`HumanApproval` token *by signature* — there is no autonomous path that mints one — and it
only logs the approved goal to the Ring-0 append-only audit ledger (it never lowers a tier
or self-grants capability). This is the spec's "propose ≠ act," indefinitely overseer-gated.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from verge.latent import Latent, LayerService
from verge.l6_motivation.magellan import (
    DegenerateCuriosityDetector,
    LearningProgressPredictor,
)
# Read-only Ring-0 handles only (the ring-enforcement guard checks this import):
from verge.ring0 import AuditEvent, Rung, record


@dataclass(frozen=True)
class SubGoal:
    """A decomposed sub-goal and the verification rung that can check it. `checkable` iff
    the rung is a real verifier rung (bounded by L3's reach)."""

    description: str
    rung: Rung | None = None

    @property
    def checkable(self) -> bool:
        return isinstance(self.rung, Rung)


@dataclass(frozen=True)
class Goal:
    """A proposed goal. `decomposable` iff every sub-goal is verifier-checkable (G6)."""

    description: str
    key: str = ""
    sub_goals: tuple[SubGoal, ...] = ()
    learning_progress: float = 0.0
    embedding: tuple = ()

    @property
    def decomposable(self) -> bool:
        return bool(self.sub_goals) and all(sg.checkable for sg in self.sub_goals)


@dataclass(frozen=True)
class HumanApproval:
    """An approval token. Only a human-initiated path constructs one; no autonomous
    constructor exists anywhere in the stack. Required by `act()`."""

    approver: str
    goal_key: str


def _goal_key(description: str) -> str:
    return hashlib.sha256(description.encode()).hexdigest()[:16]


@dataclass
class L6Motivation(LayerService):
    layer_id: str = "L6"
    lp: LearningProgressPredictor = field(default_factory=LearningProgressPredictor)
    degenerate: DegenerateCuriosityDetector = field(default_factory=DegenerateCuriosityDetector)

    # --- autonomous: proposing -----------------------------------------------
    def record_outcome(self, goal_key: str, success: float) -> None:
        self.lp.update(goal_key, success)

    def propose(self, candidates: list[Goal]) -> list[Goal]:
        """Rank candidates by (predicted) learning progress, dropping degenerate-curiosity
        traps. Autonomous — proposing is allowed; acting is not."""
        scored = []
        for g in candidates:
            hist = self.lp.history.get(g.key, [])
            if self.degenerate.is_degenerate(hist):
                continue
            lp = self.lp.predict_lp(g.key, list(g.embedding) or None)
            scored.append(Goal(description=g.description, key=g.key, sub_goals=g.sub_goals,
                               learning_progress=lp, embedding=g.embedding))
        return sorted(scored, key=lambda g: g.learning_progress, reverse=True)

    def decompose(self, description: str, sub_goals: list[SubGoal]) -> Goal:
        """Build a goal with its verifier-checkable decomposition (G6 requires 100%)."""
        return Goal(description=description, key=_goal_key(description),
                    sub_goals=tuple(sub_goals))

    # --- NOT autonomous: acting requires a human approval token ---------------
    def act(self, goal: Goal, approval: HumanApproval) -> dict:
        """Side-effecting, but only via the Ring-0 append-only audit log. Requires a
        `HumanApproval` token by signature and a 100%-decomposable goal (G6). The system
        never self-approves and never lowers a tier."""
        if not isinstance(approval, HumanApproval):
            raise PermissionError(
                "L6.act requires a HumanApproval token — no autonomous action path exists "
                "(spec §3 L6). propose() is autonomous; act() is not.")
        if approval.goal_key != goal.key:
            raise PermissionError("approval token does not match this goal")
        if not goal.decomposable:
            raise ValueError(
                "G6: a goal must decompose 100% into verifier-checkable sub-goals before "
                "it can be acted on (bounded by L3's reach).")
        record(AuditEvent(kind="goal_acted", source_layer="L6", key=goal.key,
                          detail={"approver": approval.approver, "goal": goal.description}))
        return {"acted": True, "goal_key": goal.key, "logged_to": "ring0.audit"}

    # --- LayerService --------------------------------------------------------
    def encode(self, x) -> list[Latent]:
        return []

    def step(self, ctx: list[Latent]) -> list[Latent]:
        return ctx

    def health(self) -> dict:
        return {"layer": self.layer_id, "built": True, "predictor": "MAGELLAN-LP",
                "autonomy": "propose-only; act() human-gated", "gate": "G6"}
