"""Neural Causal Model (linear-Gaussian, NCM-lite) — the causal half of L2.

The genuinely hard part of a world model is the causal *skeleton* — which edges exist
(verge-engineering.md §5; spec §3 L2). NCMs confirm that without a supplied causal
diagram, interventional/counterfactual queries are underdetermined. So:

  - the **skeleton is built from VERIFIED edges only** — never learned end-to-end (that
    conflates causation with correlation), never bootstrapped from text claims (circular);
  - we learn only the **functional forms** inside the supplied skeleton (here: linear-
    Gaussian structural equations, fit by least squares);
  - untested edges are flagged **`prior`, not `knowledge`**, and any `do(x)` whose effect
    routes through an unverified edge (or an out-of-skeleton variable) fires the
    **uncertainty flag**.

This is deliberately the linear-Gaussian special case so it runs and is exactly testable
on CPU; DoWhy/pyro + a deep-SCM head replace the functional forms at scale, behind the
same interface.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class CausalEdge:
    parent: str
    child: str
    verified: bool = False   # True only if tested by the L6→action loop; else a `prior`


@dataclass
class CausalSkeleton:
    """A DAG over named latent variables. Edges carry a `verified` flag."""

    variables: tuple[str, ...]
    edges: tuple[CausalEdge, ...]

    def parents(self, var: str) -> list[str]:
        return [e.parent for e in self.edges if e.child == var]

    def edge(self, parent: str, child: str) -> CausalEdge | None:
        for e in self.edges:
            if e.parent == parent and e.child == child:
                return e
        return None

    def has(self, var: str) -> bool:
        return var in self.variables

    def topological_order(self) -> list[str]:
        order, seen = [], set()

        def visit(v):
            if v in seen:
                return
            for p in self.parents(v):
                visit(p)
            seen.add(v)
            order.append(v)

        for v in self.variables:
            visit(v)
        return order

    def descendants(self, var: str) -> set[str]:
        out, stack = set(), [var]
        while stack:
            cur = stack.pop()
            for e in self.edges:
                if e.parent == cur and e.child not in out:
                    out.add(e.child)
                    stack.append(e.child)
        return out

    def path_uses_unverified(self, source: str, target: str) -> bool:
        """True if every/any directed path source→target traverses an unverified edge.
        We flag conservatively: if ANY edge on ANY source→target path is unverified."""
        # BFS collecting edges on paths to target
        on_path_unverified = False
        stack = [(source, [])]
        while stack:
            cur, trail = stack.pop()
            for e in self.edges:
                if e.parent != cur:
                    continue
                new_trail = trail + [e]
                if e.child == target:
                    if any(not x.verified for x in new_trail):
                        on_path_unverified = True
                else:
                    stack.append((e.child, new_trail))
        return on_path_unverified


@dataclass
class InterventionResult:
    values: dict          # variable -> predicted value under do(x)
    uncertainty: dict     # variable -> propagated std
    out_of_skeleton: bool # the uncertainty flag (untested edge / unknown var)
    overall_uncertainty: float


@dataclass
class NeuralCausalModel:
    """Linear-Gaussian SCM: x_i = b_i + Σ_{p∈parents(i)} w_ip x_p + ε_i, ε_i~N(0,σ_i²).

    `fit` learns (b, w, σ) for the supplied skeleton from observational data; `do`
    answers interventional queries; `counterfactual` does abduction→action→prediction.
    """

    skeleton: CausalSkeleton
    coef: dict = field(default_factory=dict)       # child -> {parent: weight}
    intercept: dict = field(default_factory=dict)  # child -> b
    noise_std: dict = field(default_factory=dict)  # child -> σ
    _fitted: bool = False

    def fit(self, data: dict | np.ndarray) -> "NeuralCausalModel":
        """`data` maps variable name -> 1D array of observations (or a (n, d) matrix in
        `skeleton.variables` order). Learns only the functional forms in the skeleton."""
        cols = self._as_columns(data)
        n = len(next(iter(cols.values())))
        for var in self.skeleton.variables:
            pa = self.skeleton.parents(var)
            y = cols[var]
            if pa:
                X = np.column_stack([cols[p] for p in pa] + [np.ones(n)])
                beta, *_ = np.linalg.lstsq(X, y, rcond=None)
                self.coef[var] = {p: float(beta[i]) for i, p in enumerate(pa)}
                self.intercept[var] = float(beta[-1])
                resid = y - X @ beta
            else:
                self.coef[var] = {}
                self.intercept[var] = float(np.mean(y))
                resid = y - np.mean(y)
            self.noise_std[var] = float(np.std(resid))
        self._fitted = True
        self._marginal_mean = {v: float(np.mean(cols[v])) for v in self.skeleton.variables}
        return self

    def do(self, intervention: dict, *, evidence: dict | None = None) -> InterventionResult:
        """Predict all variables under do(intervention). `evidence` (optional) sets
        observed exogenous/root values; otherwise roots take their learned mean."""
        if not self._fitted:
            raise RuntimeError("NCM.do called before fit")
        evidence = evidence or {}
        unknown = [v for v in intervention if not self.skeleton.has(v)]
        flag = bool(unknown)

        values, var_var = {}, {}
        for var in self.skeleton.topological_order():
            if var in intervention:
                values[var] = float(intervention[var])
                var_var[var] = 0.0
                continue
            if var in evidence:
                values[var] = float(evidence[var])
                var_var[var] = 0.0
                continue
            pa = self.skeleton.parents(var)
            mu = self.intercept[var] + sum(self.coef[var][p] * values[p] for p in pa)
            v = self.noise_std[var] ** 2 + sum(self.coef[var][p] ** 2 * var_var[p] for p in pa)
            values[var] = float(mu)
            var_var[var] = float(v)

        # the uncertainty flag: any do() whose effect routes through an unverified edge
        for iv in intervention:
            if not self.skeleton.has(iv):
                continue
            for desc in self.skeleton.descendants(iv):
                if self.skeleton.path_uses_unverified(iv, desc):
                    flag = True
        uncertainty = {v: float(np.sqrt(var_var.get(v, 0.0))) for v in values}
        affected = set()
        for iv in intervention:
            if self.skeleton.has(iv):
                affected |= self.skeleton.descendants(iv)
        overall = float(np.mean([uncertainty[v] for v in affected])) if affected else 0.0
        if flag:
            overall = max(overall, 1.0) * 3.0  # inflate when off-skeleton / unverified
        return InterventionResult(values=values, uncertainty=uncertainty,
                                  out_of_skeleton=flag, overall_uncertainty=overall)

    def counterfactual(self, factual: dict, intervention: dict) -> dict:
        """Abduction → action → prediction (Pearl's three steps), linear-Gaussian case.
        Infer each variable's exogenous noise from `factual`, then re-run under `do`."""
        noise = {}
        for var in self.skeleton.topological_order():
            pa = self.skeleton.parents(var)
            mu = self.intercept[var] + sum(self.coef[var][p] * factual[p] for p in pa)
            noise[var] = factual[var] - mu
        out = {}
        for var in self.skeleton.topological_order():
            if var in intervention:
                out[var] = float(intervention[var])
                continue
            pa = self.skeleton.parents(var)
            out[var] = float(self.intercept[var]
                             + sum(self.coef[var][p] * out[p] for p in pa) + noise[var])
        return out

    def _as_columns(self, data) -> dict:
        if isinstance(data, dict):
            return {k: np.asarray(v, dtype=np.float64) for k, v in data.items()}
        data = np.asarray(data, dtype=np.float64)
        return {v: data[:, i] for i, v in enumerate(self.skeleton.variables)}
