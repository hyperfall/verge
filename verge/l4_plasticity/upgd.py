"""UPGD — utility-based perturbed gradient descent (verge-engineering.md §6; spec §3 L4).

Dohare et al. (*Nature* 2024) showed standard networks under sustained continual learning
ossify — up to ~90% of units go dead — and that the fix needs a random, non-gradient
component that keeps injecting variability. **UPGD** (Elsayed & Mahmood 2024) supersedes
plain continual-backprop by handling plasticity *and* forgetting at once: it perturbs
**low-utility** units more (reviving dead capacity) and protects **high-utility** units
(bounding forgetting).

Update rule, per weight:
    u   = -w · g                                  # instantaneous utility (loss-↑ if removed)
    U   ← β U + (1-β) u                            # decayed utility trace
    s   = σ( U / (max|U| + ε) )                    # scaled utility in (0,1), high = useful
    w   ← w - lr · (g + ξ) · (1 - s),  ξ ~ N(0,σ_p²)

High utility → s→1 → gate (1-s)→0 → the weight is protected; low/negative utility → the
weight is both updated and perturbed → plasticity is restored. ~200 LOC over AdamW in the
real stack; here it is a clean numpy optimizer so L4's value is testable on CPU.

Also here: the **gradient-interference monitor** — first-epoch interference predicts
forgetting severity, so a bad task sequence can be forecast before it is committed. The
spec wires this into every training run, including L3's.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class UPGD:
    """Utility-based perturbed gradient descent. Operates on a list of numpy arrays."""

    lr: float = 0.01
    beta_utility: float = 0.99
    sigma: float = 0.01            # perturbation scale (the non-gradient variability)
    seed: int = 0
    _traces: list = field(default_factory=list, repr=False)
    _rng: object = None

    def _ensure(self, params):
        if not self._traces:
            self._traces = [np.zeros_like(p) for p in params]
            self._rng = np.random.default_rng(self.seed)

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        """In-place UPGD update of `params` given `grads`."""
        self._ensure(params)
        for i, (p, g) in enumerate(zip(params, grads)):
            u = -p * g
            self._traces[i] = self.beta_utility * self._traces[i] + (1 - self.beta_utility) * u
            tr = self._traces[i]
            scaled = _sigmoid(tr / (np.max(np.abs(tr)) + 1e-8))
            noise = self.sigma * self._rng.standard_normal(p.shape)
            p -= self.lr * (g + noise) * (1.0 - scaled)


@dataclass
class SGD:
    """Plain SGD baseline — the ossifying control arm UPGD is measured against."""

    lr: float = 0.01

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        for p, g in zip(params, grads):
            p -= self.lr * g


@dataclass
class GradientInterferenceMonitor:
    """Forecasts forgetting severity from gradient interference between tasks.

    Interference(a,b) = max(0, -cos(g_a, g_b)) — conflicting gradients (negative cosine)
    predict forgetting. The first-epoch value is the early-warning signal; `circuit_break`
    fires when interference exceeds a threshold so a bad task sequence is caught before it
    is committed."""

    threshold: float = 0.5
    history: list = field(default_factory=list)

    def observe(self, grad_a, grad_b) -> float:
        a = np.asarray(grad_a, dtype=np.float64).ravel()
        b = np.asarray(grad_b, dtype=np.float64).ravel()
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        interference = max(0.0, -cos)
        self.history.append(interference)
        return interference

    def circuit_break(self) -> bool:
        return bool(self.history and self.history[-1] > self.threshold)

    def predicted_forgetting(self) -> float:
        return float(self.history[0]) if self.history else 0.0


# --- a small numpy MLP so the continual-learning claim is testable on CPU ----

@dataclass
class _MLP:
    d_in: int
    hidden: int
    d_out: int
    seed: int = 0

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.W1 = rng.standard_normal((self.hidden, self.d_in)) / np.sqrt(self.d_in)
        self.b1 = np.zeros(self.hidden)
        self.W2 = rng.standard_normal((self.d_out, self.hidden)) / np.sqrt(self.hidden)
        self.b2 = np.zeros(self.d_out)

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def forward(self, X):
        Z1 = X @ self.W1.T + self.b1
        A1 = np.maximum(0.0, Z1)
        Y = A1 @ self.W2.T + self.b2
        return Y, (X, Z1, A1)

    def grads(self, cache, dY):
        X, Z1, A1 = cache
        n = X.shape[0]
        gW2 = dY.T @ A1 / n
        gb2 = dY.mean(axis=0)
        dA1 = dY @ self.W2
        dZ1 = dA1 * (Z1 > 0)
        gW1 = dZ1.T @ X / n
        gb1 = dZ1.mean(axis=0)
        return [gW1, gb1, gW2, gb2]

    def dead_unit_fraction(self, X) -> float:
        _, (_, Z1, _) = self.forward(X)
        alive = (Z1 > 0).any(axis=0)        # a unit is dead if it never activates
        return float(np.mean(~alive))


def continual_plasticity_demo(*, optimizer_name: str, tasks: int = 40, steps: int = 40,
                              d_in: int = 16, hidden: int = 32, seed: int = 0) -> dict:
    """Run a sequence of random-regression tasks (Dohare-style) and report the final
    dead-unit fraction. UPGD's perturbation keeps units alive; plain SGD ossifies."""
    rng = np.random.default_rng(seed)
    net = _MLP(d_in, hidden, 1, seed=seed)
    opt = UPGD(lr=0.05, sigma=0.05, seed=seed) if optimizer_name == "upgd" else SGD(lr=0.05)
    probe = rng.standard_normal((128, d_in))
    for _t in range(tasks):
        R = rng.standard_normal((1, d_in))          # a fresh target each task
        for _s in range(steps):
            X = rng.standard_normal((32, d_in))
            target = np.maximum(0.0, X @ R.T)        # nonneg target stresses relu units
            Y, cache = net.forward(X)
            dY = (Y - target)
            opt.step(net.params(), net.grads(cache, dY))
    return {"dead_unit_fraction": net.dead_unit_fraction(probe)}
