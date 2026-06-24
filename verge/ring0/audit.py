"""Ring 0 audit — append-only ledger, frozen governance thresholds, rollback ledger.

Verifier-*independent*, always-on, hard-gating (verge-engineering.md §8; spec §4).
The system never lowers its own approval tier: thresholds are a frozen dataclass with
no setter, and the ledger is append-only (`record` adds; nothing removes or rewrites).
The mutable ledger object itself (`_LEDGER`) is module-private and never exported — the
ring-enforcement guard fails the build if a self-improvement path imports it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

__all__ = [
    "AuditEvent", "record", "read_ledger", "GovernanceThresholds", "THRESHOLDS",
    "RollbackLedger",
]


@dataclass(frozen=True)
class AuditEvent:
    """One append-only audit record with full latent provenance."""

    kind: str            # "broadcast" | "goal_proposed" | "drift_check" | "skill_compiled" ...
    source_layer: str    # provenance (Latent.source_layer)
    key: str             # Latent.key hex — what was acted on
    detail: dict = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


# Module-private, mutable, append-only. NOT exported. Importing this from a
# self-improvement path is a ring-enforcement violation (it is a mutable Ring-0 handle).
_LEDGER: list[AuditEvent] = []


def record(event: AuditEvent) -> None:
    """Append-only write. The one mutation Ring 0 permits from outside; it cannot
    remove history, rewrite an entry, or lower a tier."""
    if not isinstance(event, AuditEvent):
        raise TypeError("record() takes an AuditEvent")
    _LEDGER.append(event)


def read_ledger() -> tuple[AuditEvent, ...]:
    """Immutable view of the audit history."""
    return tuple(_LEDGER)


@dataclass(frozen=True)
class GovernanceThresholds:
    """Frozen go/no-go thresholds (verge-engineering.md §7, §8). No setter exists; a
    new policy is a versioned migration reviewed independently, never a self-edit."""

    # Drift breaker (mirrors verge.latent.DriftMonitor defaults; Ring 0 owns the policy).
    drift_warn: float = 0.15
    drift_halt: float = 0.40
    coherence_floor: float = 0.80
    # Eval-harness autonomy gate: a layer ships only if its slope CI clears this and it
    # beats its ablation (the bitter-lesson gate).
    min_slope_for_ship: float = 0.0
    # L6: every proposed goal must decompose into verifier-checkable sub-goals.
    goal_decomposition_required: float = 1.0  # 100%


# The single frozen instance the stack reads. Read-only handle.
THRESHOLDS = GovernanceThresholds()


@dataclass
class RollbackLedger:
    """Records 'last known good' latent/weight checkpoints so the drift breaker's
    HALT_ROLLBACK has somewhere to roll back to. Read interface is exported; the write
    path is gated behind an approved learning job, never a self-edit."""

    _checkpoints: list[dict] = field(default_factory=list)

    def commit(self, label: str, meta: dict) -> int:
        """Record a good checkpoint. (Append-only; rollbacks select, never delete.)"""
        self._checkpoints.append({"label": label, "meta": dict(meta)})
        return len(self._checkpoints) - 1

    def last_good(self) -> dict | None:
        return self._checkpoints[-1] if self._checkpoints else None

    def history(self) -> tuple[dict, ...]:
        return tuple(self._checkpoints)
