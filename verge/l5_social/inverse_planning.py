"""Inverse planning — the core of L5 Theory of Mind (verge-engineering.md §6; spec §3 L5).

Model other agents as approximately-rational planners (Baker et al.; inverse planning):
an agent with mental state (goal, belief) acts to maximize value *under its own belief*,
so observing its actions lets us invert to a **posterior over mental states** — including
*false* beliefs, the Sally-Anne structure that distinguishes ToM from assuming the agent
knows the truth.

Implemented on a small grid so it runs and is exactly testable on CPU. The same machinery
models the **overseer's intent** (alignment lives here: L5 correctly modelling overseer
intent + L6 deferring). ExploreToM is the adversarial data/eval engine at scale; this is
the inference engine it would stress.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# grid actions
ACTIONS = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0), "stay": (0, 0)}
_ANAMES = list(ACTIONS)


def _manhattan(a, b) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _move(state, action, grid):
    dx, dy = ACTIONS[action]
    return (int(np.clip(state[0] + dx, 0, grid[0] - 1)),
            int(np.clip(state[1] + dy, 0, grid[1] - 1)))


@dataclass(frozen=True)
class MentalState:
    """What another mind wants and believes. `target` is where the agent *believes* the
    reward is (its belief); under a true belief it equals the real reward location."""

    target: tuple


def policy(state, target, *, grid=(5, 5), beta: float = 3.0) -> dict:
    """A (Boltzmann-)rational planner's action distribution: softmax over how much each
    action reduces distance to the believed `target`. β is the rationality temperature."""
    vals = np.array([-_manhattan(_move(state, a, grid), target) for a in _ANAMES], dtype=float)
    e = np.exp(beta * (vals - vals.max()))
    p = e / e.sum()
    return dict(zip(_ANAMES, p))


def sample_trajectory(start, target, *, grid=(5, 5), beta=3.0, steps=8, seed=0):
    """Roll out a rational agent heading for `target` (its belief)."""
    rng = np.random.default_rng(seed)
    s, traj = start, []
    for _ in range(steps):
        p = policy(s, target, grid=grid, beta=beta)
        a = rng.choice(_ANAMES, p=list(p.values()))
        traj.append((s, a))
        s = _move(s, a, grid)
    return traj


@dataclass
class InversePlanner:
    """Bayesian inference of an agent's mental state from observed (state, action) pairs."""

    grid: tuple = (5, 5)
    beta: float = 3.0

    def infer(self, trajectory, candidates: list[tuple],
              prior: dict | None = None) -> dict:
        """Posterior over candidate believed-targets: P(target | trajectory) ∝
        P(target) · Π_t π(a_t | s_t, target). Returns {target: probability}."""
        log_post = {}
        for tgt in candidates:
            lp = np.log((prior or {}).get(tgt, 1.0 / len(candidates)) + 1e-12)
            for s, a in trajectory:
                lp += np.log(policy(s, tgt, grid=self.grid, beta=self.beta)[a] + 1e-12)
            log_post[tgt] = lp
        m = max(log_post.values())
        post = {t: float(np.exp(lp - m)) for t, lp in log_post.items()}
        z = sum(post.values())
        return {t: p / z for t, p in post.items()}

    def map_estimate(self, trajectory, candidates, prior=None) -> MentalState:
        post = self.infer(trajectory, candidates, prior)
        return MentalState(target=max(post, key=post.get))
