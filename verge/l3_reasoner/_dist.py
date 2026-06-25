"""Tiny rank helpers for the multi-GPU (accelerate launch + DeepSpeed ZeRO) path.

When the run is launched with `accelerate launch --num_processes N ...`, the SAME script
runs once per GPU. These helpers let the single-process harness stay correct under that:
- pick the right local GPU for the pre-trainer baseline eval (each rank → its own device),
- print progress/reports on the main rank only (no N-times-duplicated output).

Under plain `python -m ...` (no distributed env), RANK/LOCAL_RANK are unset → rank 0,
single device — so the helpers are no-ops and the existing single-GPU path is unchanged.
"""
from __future__ import annotations

import os


def rank() -> int:
    """Global process rank across all nodes/GPUs (0 when not launched distributed)."""
    return int(os.environ.get("RANK", "0"))


def local_rank() -> int:
    """Process rank within this node — i.e. which local GPU this process owns."""
    return int(os.environ.get("LOCAL_RANK", "0"))


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_main() -> bool:
    """True on exactly one process — the only one that should print user-facing output."""
    return rank() == 0


def mprint(*args, **kwargs) -> None:
    """print(), but only on the main rank (avoids N-times-duplicated lines under ZeRO)."""
    if is_main():
        kwargs.setdefault("flush", True)
        print(*args, **kwargs)
