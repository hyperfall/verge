"""MAGELLAN-style learning-progress motivation (verge-engineering.md §6; spec §3 L6).

Replace raw free-energy with **learning-progress (LP) based goal selection** as the
primary signal: LP is the time-derivative of competence — propose goals where you are
*improving fastest*, not where reward is highest. MAGELLAN's contribution is that LP is
*predicted* by a model that generalizes across goals, so LP can be estimated for unseen
goals (not just measured after the fact).

Two safety-relevant pieces ship with it:
  - the **degenerate-curiosity detector** — catch noisy-TV (apparent LP from pure noise)
    and echo-chamber (trivially-mastered, repeated) traps, so the drive is not captured by
    spurious progress;
  - LP is the *signal*; whether to act on a proposed goal is gated elsewhere (propose ≠
    act, enforced at the type level in `service.py`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LearningProgressPredictor:
    """Tracks per-goal competence and estimates LP; optionally generalizes LP across
    goals from goal embeddings (the MAGELLAN move)."""

    window: int = 4
    history: dict = field(default_factory=dict)     # goal_key -> [outcomes in [0,1]]
    _gen_w: np.ndarray | None = field(default=None, repr=False)

    def update(self, goal_key: str, outcome: float) -> None:
        self.history.setdefault(goal_key, []).append(float(outcome))

    def competence(self, goal_key: str) -> float:
        h = self.history.get(goal_key, [])
        return float(np.mean(h[-self.window:])) if h else 0.0

    def learning_progress(self, goal_key: str) -> float:
        """Measured LP = competence(recent window) − competence(previous window)."""
        h = self.history.get(goal_key, [])
        if len(h) < 2 * self.window:
            return 0.0
        recent = np.mean(h[-self.window:])
        older = np.mean(h[-2 * self.window:-self.window])
        return float(recent - older)

    # --- MAGELLAN generalization: predict LP for unseen goals from embeddings -
    def fit_generalizer(self, embeddings: dict) -> None:
        keys = [k for k in embeddings if len(self.history.get(k, [])) >= 2 * self.window]
        if len(keys) < 2:
            return
        X = np.array([embeddings[k] for k in keys])
        y = np.array([self.learning_progress(k) for k in keys])
        X1 = np.hstack([X, np.ones((len(keys), 1))])
        self._gen_w, *_ = np.linalg.lstsq(X1, y, rcond=None)

    def predict_lp(self, goal_key: str, embedding=None) -> float:
        if embedding is not None and self._gen_w is not None:
            return float(np.append(np.asarray(embedding), 1.0) @ self._gen_w)
        return self.learning_progress(goal_key)


@dataclass
class DegenerateCuriosityDetector:
    """Flags goals whose 'progress' is a trap: noisy-TV (oscillation with no net trend) or
    echo-chamber (trivially mastered / repeated). Min observations before judging."""

    window: int = 4
    noise_var_threshold: float = 0.18
    lp_eps: float = 0.06
    mastered: float = 0.95

    def is_degenerate(self, history: list[float]) -> bool:
        if len(history) < 2 * self.window:
            return False
        recent = np.mean(history[-self.window:])
        older = np.mean(history[-2 * self.window:-self.window])
        lp = abs(recent - older)
        var = float(np.var(history[-2 * self.window:]))
        noisy_tv = lp < self.lp_eps and var > self.noise_var_threshold   # churn, no trend
        echo_chamber = recent > self.mastered and lp < self.lp_eps        # already mastered
        return bool(noisy_tv or echo_chamber)
