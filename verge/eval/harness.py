"""The layer-agnostic eval harness (verge-engineering.md §7).

    evaluate(layer, *, seeds=(0,1,2), rounds, frozen_test, ablations, preregister) -> Report

Run ≥3 seeds on a FROZEN test set, compare against named ablations, fit slope + 95% CI
(via the M1 `fit_slope`), and decide go/no-go against the PRE-REGISTERED threshold. A
result that wasn't pre-registered doesn't count.

A *layer* here is any object implementing `EvalProtocol`: given a seed, a round budget,
and the frozen test set, it returns the per-round metric curve (e.g. single-shot pass@1
for L3, retrieval-conditioned pass@1 for M2). This is exactly the M1 loop generalized —
the harness does not know or care what is being measured.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from verge.eval.metrics import fit_slope


@runtime_checkable
class EvalProtocol(Protocol):
    """What a layer (or an ablation arm) provides to be measured."""

    name: str

    def run_seed(self, seed: int, *, rounds: int, frozen_test) -> list[float]:
        """Return the metric for rounds 0..rounds (length rounds+1). Round 0 is the
        baseline measured before any self-training, exactly as in M1."""
        ...


@dataclass
class ArmResult:
    name: str
    # curves[seed] = [m_round0, m_round1, ...]
    curves: dict[int, list[float]]

    def pooled_points(self) -> tuple[list[float], list[float]]:
        xs, ys = [], []
        for curve in self.curves.values():
            for r, m in enumerate(curve):
                xs.append(float(r))
                ys.append(float(m))
        return xs, ys

    def final_mean(self) -> float:
        finals = [c[-1] for c in self.curves.values() if c]
        return sum(finals) / len(finals) if finals else float("nan")

    def baseline_mean(self) -> float:
        bases = [c[0] for c in self.curves.values() if c]
        return sum(bases) / len(bases) if bases else float("nan")


@dataclass
class Report:
    layer: str
    rounds: int
    seeds: tuple[int, ...]
    main: ArmResult
    ablations: dict[str, ArmResult]
    slope: float
    ci_low: float
    ci_high: float
    preregister: dict
    decision: str          # "SHIP" | "CUT" | "FLAT"
    reasons: list[str] = field(default_factory=list)

    @property
    def is_flat(self) -> bool:
        return self.ci_low <= 0.0 <= self.ci_high

    def summary(self) -> str:
        lines = [
            f"=== eval report: {self.layer} ===",
            f"seeds={list(self.seeds)} rounds={self.rounds}",
            f"baseline pass@1 = {self.main.baseline_mean():.4f}  "
            f"-> final {self.main.final_mean():.4f}",
            f"slope = {self.slope:+.4f}  95% CI [{self.ci_low:+.4f}, {self.ci_high:+.4f}]  "
            f"({'FLAT — CI includes zero' if self.is_flat else 'CI excludes zero'})",
        ]
        for name, arm in self.ablations.items():
            lines.append(
                f"  ablation[{name}]: final {arm.final_mean():.4f}  "
                f"(Δ vs main = {self.main.final_mean() - arm.final_mean():+.4f})")
        lines.append(f"pre-registered: {self.preregister}")
        lines.append(f"DECISION: {self.decision}")
        for r in self.reasons:
            lines.append(f"  - {r}")
        return "\n".join(lines)


def _run_arm(arm: EvalProtocol, seeds, rounds, frozen_test) -> ArmResult:
    curves = {s: arm.run_seed(s, rounds=rounds, frozen_test=frozen_test) for s in seeds}
    return ArmResult(name=arm.name, curves=curves)


def evaluate(
    layer: EvalProtocol,
    *,
    seeds: tuple[int, ...] = (0, 1, 2),
    rounds: int,
    frozen_test,
    ablations: dict[str, EvalProtocol] | None = None,
    preregister: dict,
) -> Report:
    """Run the full §7 evaluation and return a go/no-go Report.

    `preregister` (written BEFORE the run) supports:
      - "hypothesis": str — the claim being tested.
      - "min_slope": float — slope-CI lower bound must exceed this to SHIP (default 0).
      - "beats_ablation": str | None — name of the ablation the layer's final metric
        must beat (the bitter-lesson gate). None = no ablation gate.
    """
    if len(seeds) < 3:
        raise ValueError("§7 requires >= 3 seeds (pre-registered); got %d" % len(seeds))
    ablations = ablations or {}
    main = _run_arm(layer, seeds, rounds, frozen_test)
    abl_results = {name: _run_arm(a, seeds, rounds, frozen_test) for name, a in ablations.items()}

    xs, ys = main.pooled_points()
    slope, lo, hi = fit_slope(xs, ys)

    min_slope = float(preregister.get("min_slope", 0.0))
    beats = preregister.get("beats_ablation")
    reasons: list[str] = []

    flat = lo <= 0.0 <= hi
    slope_clears = lo > min_slope
    if slope_clears:
        reasons.append(f"slope CI lower bound {lo:+.4f} > pre-registered {min_slope:+.4f}")
    else:
        reasons.append(f"slope CI lower bound {lo:+.4f} does NOT clear {min_slope:+.4f}"
                       + (" (flat: CI includes zero)" if flat else ""))

    ablation_ok = True
    if beats is not None:
        if beats not in abl_results:
            raise KeyError(f"pre-registered ablation {beats!r} was not provided")
        delta = main.final_mean() - abl_results[beats].final_mean()
        ablation_ok = delta > 0.0
        reasons.append(
            f"final vs ablation[{beats}] Δ={delta:+.4f} "
            f"({'beats' if ablation_ok else 'does NOT beat'} the bitter-lesson gate)")

    if slope_clears and ablation_ok:
        decision = "SHIP"
    elif flat:
        decision = "FLAT"
    else:
        decision = "CUT"

    return Report(
        layer=getattr(layer, "name", "layer"),
        rounds=rounds, seeds=tuple(seeds), main=main, ablations=abl_results,
        slope=slope, ci_low=lo, ci_high=hi, preregister=dict(preregister),
        decision=decision, reasons=reasons,
    )
