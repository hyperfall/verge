"""Ring-enforcement CI guard (verge-engineering.md §8; spec §4 Thread D).

Two halves:
  1. The real tree is clean — no self-improvement path imports a mutable Ring-0 handle.
  2. The guard *fails* when a violation is introduced (deliberately-bad fixture, kept
     only as an in-test string so it never lives in the tree).
"""
from __future__ import annotations

import os

from verge import ring_check
from verge.ring_check import ring0_import_violations, scan


def _pkg_root() -> str:
    return os.path.dirname(os.path.abspath(ring_check.__file__))


# --- half 1: the real tree is clean ----------------------------------------

def test_real_tree_has_no_ring0_violations():
    violations = scan(_pkg_root())
    assert violations == [], "\n".join(str(v) for v in violations)


def test_walk_actually_reaches_self_improvement_modules():
    # Proves the reachability walk traverses rather than vacuously passing: L3's reward
    # module legitimately imports the read-only verifier and must be in the graph.
    reached = ring_check.reachable_modules(_pkg_root())
    assert "verge.l3_reasoner.reward" in reached, sorted(reached)


# --- half 2: the guard fails on a violation (bad fixtures, in-test only) ----

BAD_PRIVATE_LEDGER = (
    "from verge.ring0.audit import _LEDGER\n"
    "_LEDGER.clear()  # rewriting append-only history — must be caught\n"
)
BAD_WHOLE_MODULE = "import verge.ring0.audit as a\n"
BAD_WILDCARD = "from verge.ring0 import *\n"
GOOD_READONLY = (
    "from verge.ring0 import ExactMatchVerifier, verify, THRESHOLDS, record\n"
)


def test_guard_catches_private_mutable_handle():
    v = ring0_import_violations(BAD_PRIVATE_LEDGER, module_name="verge.l6_motivation.evil")
    assert v and any("_LEDGER" in str(x) for x in v)


def test_guard_catches_whole_module_import():
    v = ring0_import_violations(BAD_WHOLE_MODULE, module_name="verge.l7_router.evil")
    assert v and any("whole-module" in str(x) for x in v)


def test_guard_catches_wildcard_import():
    v = ring0_import_violations(BAD_WILDCARD, module_name="verge.l4_plasticity.evil")
    assert v and any("wildcard" in str(x) for x in v)


def test_guard_allows_readonly_handles():
    v = ring0_import_violations(GOOD_READONLY, module_name="verge.l3_reasoner.reward")
    assert v == [], "\n".join(str(x) for x in v)
