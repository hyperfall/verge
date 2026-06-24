"""CPU simulator so the whole pipeline runs free before any GPU spend.

NOT a language model. Simulates the one dynamic that matters: single-shot skill
rises when distilled on correct, deduped traces; search reaches a bit past current
skill. Lets you validate generate->verify->dedup->train->measure end to end.
"""
from __future__ import annotations

import hashlib
import math
import random

from verifiers import Problem, extract_final_answer, normalize_number

THETA0 = 0.20
EXPLORE = 0.30
_VOCAB = ("let x be the count add the parts carry the remainder divide by the rate "
          "subtract the discount multiply totals compute the sum compare simplify").split()


def _difficulty(pid: str) -> float:
    return (int(hashlib.sha256(pid.encode()).hexdigest()[:8], 16) % 10_000) / 10_000.0


class MockModel:
    def __init__(self, cfg, device=None):
        self.cfg = cfg
        self.rng = random.Random(getattr(cfg.train, "seed", 0))
        self.theta = THETA0

    def _trace(self, p: Problem, correct: bool) -> str:
        gold = normalize_number(p.answer) or "0"
        ans = gold if correct else str(int(float(gold)) + 1)
        phrase = " ".join(self.rng.sample(_VOCAB, k=6))
        nums = " ".join(str(self.rng.randint(1, 999)) for _ in range(3))
        return f"for {p.id} {phrase} intermediate {nums} so #### {ans}"

    def reset_to_base(self):
        self.theta = THETA0

    def single_shot_batch(self, problems):
        out = []
        for p in problems:
            noise = (self.rng.random() - 0.5) * 0.04
            out.append(self._trace(p, _difficulty(p.id) < self.theta + noise))
        return out

    def search_batch(self, problems):
        out = []
        for p in problems:
            d = _difficulty(p.id)
            out.append([self._trace(p, d < self.theta + EXPLORE * self.rng.random())
                        for _ in range(self.cfg.model.samples_per_problem)])
        return out

    def finetune(self, pairs):
        if not pairs:
            return
        n_correct = sum(1 for p, t in pairs if extract_final_answer(t) == normalize_number(p.answer))
        gain = 0.70 * (1.0 - math.exp(-max(0, n_correct) / 1500.0))
        self.theta = min(0.95, THETA0 + gain)
