"""The workspace bus — how layers compose through the latent contract, not direct calls.

Layers register as `LayerService`s and communicate *only* by reading/writing `Latent`
objects on this bus (verge-engineering.md §2, §9). No layer parses another layer's
internals. `broadcast` publishes a latent with full provenance to the audit ledger;
`select` is the capacity-limited salience pick that L7's router (when built) refines.

This is the load-bearing seam: it is real, not a stub.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from verge.latent import DriftMonitor, Latent, LayerService

__all__ = ["WorkspaceBus", "Broadcast", "LayerService", "Latent"]


@dataclass(frozen=True)
class Broadcast:
    """A single published message: the latent plus a monotonically increasing tick."""

    tick: int
    latent: Latent


def default_salience(latent: Latent) -> float:
    """Salience = free-energy-reduction × verifier-confidence × goal-relevance (spec
    §6 / L7). With only the confidence channel wired in the skeleton, salience reduces
    to confidence; L7's learned router supplies the other two factors when built."""
    return float(latent.confidence)


@dataclass
class WorkspaceBus:
    """In-process broadcast/select bus over the shared latent.

    `capacity` is the global-workspace bottleneck: `select` returns at most `capacity`
    latents, ranked by salience. The drift circuit-breaker is consulted on every
    broadcast batch so representational drift trips slow/anchor or halt/rollback before
    a poisoned latent propagates.
    """

    capacity: int = 7  # Miller's bottleneck; the integration claim, not a magic number
    _services: dict[str, LayerService] = field(default_factory=dict, repr=False)
    _log: list[Broadcast] = field(default_factory=list, repr=False)
    _tick: int = 0
    monitor: DriftMonitor | None = None
    # An append-only sink for provenance. Wired to ring0.audit by the orchestrator;
    # left as a plain list in the skeleton so the bus has no Ring-0 import dependency.
    _audit_sink: list = field(default_factory=list, repr=False)

    def register(self, service: LayerService) -> None:
        if not getattr(service, "layer_id", None):
            raise ValueError("LayerService must declare a layer_id before registering")
        self._services[service.layer_id] = service

    def broadcast(self, latents: list[Latent]) -> list[Broadcast]:
        """Publish latents to the bus. Returns the Broadcast records. Consults the
        drift breaker on the batch; a HALT_ROLLBACK decision raises so the orchestrator
        can roll back rather than propagate drift."""
        if self.monitor is not None and latents:
            decision = self.monitor.check(latents)
            self._audit_sink.append({"event": "drift_check", **decision})
            if decision["decision"] == "HALT_ROLLBACK":
                raise DriftHalt(decision)
        out = []
        for l in latents:
            self._tick += 1
            b = Broadcast(self._tick, l)
            self._log.append(b)
            self._audit_sink.append(
                {"event": "broadcast", "tick": b.tick,
                 "source_layer": l.source_layer, "key": l.key.hex()}
            )
            out.append(b)
        return out

    def select(self, *, salience=default_salience) -> list[Latent]:
        """Capacity-limited select over everything broadcast this far: the top-`capacity`
        latents by salience. This is the functional core L7's router specializes."""
        ranked = sorted(self._log, key=lambda b: salience(b.latent), reverse=True)
        return [b.latent for b in ranked[: self.capacity]]

    def history(self) -> list[Broadcast]:
        return list(self._log)

    def audit_trail(self) -> list:
        return list(self._audit_sink)


class DriftHalt(RuntimeError):
    """Raised by the bus when the drift breaker decides HALT_ROLLBACK."""

    def __init__(self, decision: dict):
        super().__init__(f"drift circuit-breaker tripped: {decision}")
        self.decision = decision
