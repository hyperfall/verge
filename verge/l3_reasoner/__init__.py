"""L3 — the verified reasoner (BUILT; harden, then scale) — verge-engineering.md §3.

L3 is the spine and the only layer with a result (spec §5). This package wraps and
orchestrates the measured `m1-verified-reasoner/` code behind the new VERGE interfaces:

  - `reward.py`   — the GRPO reward: the Ring-0 `ExactMatchVerifier` as the ONLY signal.
  - `skeleton.py` — the skeleton/rendering output contract + faithfulness check.
  - `service.py`  — the `LayerService` + the `EvalProtocol` adapter that drives the M1
                    expert-iteration loop (search → verify-filter → dedupe →
                    reset-to-base → measure) through the §7 harness.
  - `run.py`      — `python -m verge.l3_reasoner.run --mock`.

Respecting spec §2: only verifier-confirmed traces ever enter training; the worst case
is wasted samples, never poisoned weights. There is no differentiable guide that labels
training data.
"""
from verge.l3_reasoner.grpo import (
    GRPOEngine,
    GRPOSettings,
    MockGRPOEngine,
    make_trl_reward,
)
from verge.l3_reasoner.reward import make_problem, reward
from verge.l3_reasoner.skeleton import Rendering, Skeleton, faithful

__all__ = [
    "reward", "make_problem", "Skeleton", "Rendering", "faithful",
    "GRPOSettings", "GRPOEngine", "MockGRPOEngine", "make_trl_reward",
]
