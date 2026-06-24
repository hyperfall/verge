"""The expert-iteration loop (ReST^EM / RFT, calibrated):

  round 0: measure single-shot pass@1 on the FROZEN test set (baseline)
  each round:
    generate K samples/train-problem at temp 1.0   (search)
    keep traces the verifier accepts                (verify-filter)
    dedupe to distinct reasoning paths              (RFT key step)
    reset model to base, SFT on the accumulated pool (train-from-base)
    measure single-shot pass@1 on frozen test       (fixed greedy decoding)

The verifier is the strict filter — we only ever train on confirmed-correct traces,
so the worst case is wasted samples, never poisoned weights.
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass

from metrics import dedupe_reasoning_paths, pass_at_1
from verifiers import ExactMatchVerifier


@dataclass
class RoundLog:
    round: int
    pass_at_1: float
    n_correct_new: int
    pool_size: int


def run(cfg, seed: int, train, test):
    random.seed(seed)
    cfg.train.seed = seed if hasattr(cfg.train, "seed") else seed
    verifier = ExactMatchVerifier()
    t0 = time.time()
    budget = cfg.wall_clock_min * 60.0

    Model = _select_model(cfg)
    model = Model(cfg)

    def measure() -> float:
        traces = model.single_shot_batch(test)
        return pass_at_1([verifier.verify(p, t) for p, t in zip(test, traces)])

    logs = [RoundLog(0, measure(), 0, 0)]
    _save(cfg, seed, logs)
    pool: list = []

    for r in range(1, cfg.loop.rounds + 1):
        if budget and time.time() - t0 > budget:
            print(f"[seed{seed}] wall-clock budget hit at round {r-1}")
            break
        batched = model.search_batch(train)
        per_problem: dict[str, int] = {}
        kept = 0
        for p, traces in zip(train, batched):
            for tr in traces:
                if len(tr) > cfg.train.max_trace_chars:
                    continue
                if verifier.verify(p, tr) and per_problem.get(p.id, 0) < 4:
                    pool.append((p, tr))
                    per_problem[p.id] = per_problem.get(p.id, 0) + 1
                    kept += 1
        if cfg.train.dedupe_reasoning_paths:
            pool = dedupe_reasoning_paths(pool)
        if len(pool) > cfg.train.max_pool:
            pool = random.sample(pool, cfg.train.max_pool)
        if cfg.train.reset_each_round:
            model.reset_to_base()
        model.finetune(pool)
        logs.append(RoundLog(r, measure(), kept, len(pool)))
        _save(cfg, seed, logs)
    return logs


def _select_model(cfg):
    if getattr(cfg, "use_mock", False):
        from mock_model import MockModel
        return MockModel
    from model import Model
    return Model


def _save(cfg, seed: int, logs):
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, f"seed{seed}.json"), "w") as f:
        json.dump({"seed": seed, "logs": [asdict(x) for x in logs]}, f, indent=2)
