"""Ring 1 — the differentiability boundary (verge-engineering.md §8; spec §4 Thread D).

Compiled skills run as **Wasm modules under wasmtime**: opaque, sandboxed, and *not
FFI-able into the autograd graph*. This is the boundary that lets the system compile
skills for itself while only ever touching the non-differentiable algorithmic surface,
never the learned core.

The interface is defined; the compiled-skill execution path is a stub (`wasmtime` is an
optional dep — `verge[sandbox]`).
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["CompiledSkill", "WasmSkillSandbox"]


@dataclass(frozen=True)
class CompiledSkill:
    """A verifier-gated, compiled skill. `wasm` is opaque bytecode; by construction it
    cannot be differentiated through or FFI'd into autograd."""

    name: str
    wasm: bytes
    # The verifier rung that gated this skill in (it never enters without confirmation).
    gated_by: str = "EXACT"


class WasmSkillSandbox:
    """The execution boundary for Ring-1 compiled skills.

    Invariant (enforced by design + the ring-enforcement guard): nothing here returns a
    differentiable handle, and the sandbox has no path to mutate Ring-0 or the learned
    core. Calls are pure: opaque bytes in, plain values out.
    """

    def run(self, skill: CompiledSkill, args: dict) -> dict:
        # TODO(Ring1): execute `skill.wasm` under wasmtime with a no-FFI, no-host-call
        # store; return only plain (non-differentiable) values. Optional dep:
        # `pip install verge[sandbox]` (wasmtime>=21).
        raise NotImplementedError(
            "Wasm skill execution not built — open component: wasmtime (verge[sandbox]). "
            "The boundary contract is defined; the compiled-skill path is the stub."
        )
