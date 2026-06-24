"""Entrypoint.

  python run.py --mock                      # CPU smoke test, no GPU/ML deps
  python run.py --seeds 0 1 2               # real run (needs torch+transformers+datasets)
  python run.py --model Qwen/Qwen2.5-1.5B-Instruct --rounds 4
Then: python run.py --aggregate
"""
from __future__ import annotations

import argparse
import glob
import json
import os

from config import Config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--aggregate", action="store_true", help="summarize runs/ -> slope")
    ap.add_argument("--model", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--train-size", type=int, default=None)
    ap.add_argument("--test-size", type=int, default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--wall-clock-min", type=float, default=0.0)
    args = ap.parse_args()

    cfg = Config()
    cfg.use_mock = args.mock
    cfg.wall_clock_min = args.wall_clock_min
    if args.model:
        cfg.model.name = args.model
    if args.rounds:
        cfg.loop.rounds = args.rounds
    if args.train_size:
        cfg.data.train_size = args.train_size
    if args.test_size:
        cfg.data.test_size = args.test_size
    if args.outdir:
        cfg.output_dir = args.outdir
    if args.mock:
        cfg.data.train_size = min(cfg.data.train_size, 400)
        cfg.data.test_size = min(cfg.data.test_size, 300)

    if args.aggregate:
        return _aggregate(cfg.output_dir)

    from expert_iteration import run as run_seed
    if cfg.use_mock:
        train, test = _mock_data(cfg)
    else:
        from data import load_problems
        train, test = load_problems(cfg.data)

    seeds = args.seeds if args.seeds is not None else list(cfg.loop.seeds)
    for s in seeds:
        logs = run_seed(cfg, s, train, test)
        print(f"\n[seed {s}] single-shot pass@1 by round:")
        for lg in logs:
            print(f"  r{lg.round}: pass@1={lg.pass_at_1:.4f}  new_correct={lg.n_correct_new}  pool={lg.pool_size}")
    print("\nAggregate with:  python run.py --aggregate" + (f" --outdir {cfg.output_dir}" if args.outdir else ""))


def _mock_data(cfg):
    import random
    from verifiers import Problem
    rng = random.Random(1234)
    train = [Problem(f"tr{i}", f"mock {i}", str(rng.randint(1, 99))) for i in range(cfg.data.train_size)]
    test = [Problem(f"te{i}", f"mocktest {i}", str(rng.randint(1, 99))) for i in range(cfg.data.test_size)]
    return train, test


def _aggregate(outdir):
    from metrics import fit_slope
    xs, ys, finals = [], [], []
    for path in sorted(glob.glob(os.path.join(outdir, "seed*.json"))):
        d = json.load(open(path))
        for lg in d["logs"]:
            xs.append(float(lg["round"]))
            ys.append(lg["pass_at_1"])
        last = max(d["logs"], key=lambda l: l["round"])
        finals.append(last["pass_at_1"])
        r0 = next(l["pass_at_1"] for l in d["logs"] if l["round"] == 0)
        print(f"{os.path.basename(path)}: r0={r0:.4f} -> final={last['pass_at_1']:.4f}")
    if len(xs) >= 3:
        slope, lo, hi = fit_slope(xs, ys)
        mean_final = sum(finals) / len(finals)
        print(f"\nmean final pass@1 = {mean_final:.4f}")
        print(f"slope = {slope:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"({'rising' if lo > 0 else 'flat/!'} )")
    else:
        print("need >=3 (round,seed) points to fit a slope")


if __name__ == "__main__":
    main()
