"""The L3 GRPO reward — the Ring-0 verifier is the ONLY signal (verge-engineering.md §3).

This is the GRPO/DAPO reward wrapper: GRPO samples a group, scores each completion with
this verifiable reward, and advantage-weights. The reward is deterministic and
label-free — it is the Ring-0 `ExactMatchVerifier`, never a learned reward model. Per
spec §2 this keeps the loop a *bounded amplifier*: only verifier-confirmed traces ever
gain advantage, so the worst case is wasted samples, never poisoned weights.

When the engine is swapped to TRL `GRPOTrainer` (then verl), this `reward(...)` is the
`reward_funcs` callable — same verifier, no other signal.
"""
from __future__ import annotations

# Read-only Ring-0 handles only (the ring-enforcement guard checks this import).
from verge.ring0 import ExactMatchVerifier, Problem

_V = ExactMatchVerifier()


def make_problem(prompt: str, answer: str, pid: str = "grpo") -> Problem:
    return Problem(id=pid, prompt=prompt, answer=str(answer))


def reward(prompt: str, completion: str, answer: str) -> float:
    """The ONLY signal; deterministic. 1.0 iff the completion's final answer exactly
    matches `answer` under the Ring-0 verifier, else 0.0."""
    return 1.0 if _V.verify(make_problem(prompt, answer), completion) else 0.0


def reward_batch(prompts: list[str], completions: list[str], answers: list[str]) -> list[float]:
    """Group-wise rewards for a GRPO sample group (TRL `reward_funcs` shape)."""
    return [reward(p, c, a) for p, c, a in zip(prompts, completions, answers)]
