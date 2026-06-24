"""Ring 0 — FROZEN. The only trusted asset, and it is never learned.

Holds the verifier kernel, the verification ladder, the audit layer, the governance
thresholds, and the rollback ledger (verge-engineering.md §8; spec §4 Thread D).

**No setters.** Ring 0 exposes read-only handles and a single append-only audit write
(`record`) that can never *lower* a tier. The CI ring-enforcement guard
(`verge.ring_check`) fails the build if any module reachable from a self-improvement
path imports anything from Ring 0 outside the read-only allowlist — see
`RING0_READONLY_EXPORTS` below, which is the single source of truth for that guard.

The verifier kernel (`ExactMatchVerifier`) is reused **verbatim** from the measured
`m1-verified-reasoner/`; its logic is not modified here.
"""
from __future__ import annotations

from verge.ring0.audit import (
    THRESHOLDS,
    AuditEvent,
    GovernanceThresholds,
    RollbackLedger,
    read_ledger,
    record,
)
from verge.ring0.sandbox import CompiledSkill, WasmSkillSandbox
from verge.ring0.verifiers import (
    ExactMatchVerifier,
    MathEquivalenceRung,
    Problem,
    Rung,
    VerificationLadder,
    default_ladder,
    extract_final_answer,
    normalize_number,
    verify,
)

# The single source of truth for the ring-enforcement guard: the only Ring-0 names a
# self-improvement path is permitted to import. Everything else (private internals, the
# mutable ledger object, any future setter) is a build-failing violation.
RING0_READONLY_EXPORTS = frozenset({
    # verifier kernel (read-only; deterministic; never learned)
    "ExactMatchVerifier", "Problem", "verify", "extract_final_answer", "normalize_number",
    # verification ladder (read-only checks)
    "Rung", "VerificationLadder", "default_ladder", "MathEquivalenceRung",
    # audit: append-only write + read (cannot lower a tier)
    "AuditEvent", "record", "read_ledger",
    # governance thresholds: read-only frozen instance + its type
    "GovernanceThresholds", "THRESHOLDS",
    # rollback ledger: read interface
    "RollbackLedger",
    # sandbox boundary (read-only interface)
    "WasmSkillSandbox", "CompiledSkill",
})

__all__ = sorted(RING0_READONLY_EXPORTS)
