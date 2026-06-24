"""L3 end-to-end runner.

  # Free CPU pipeline checks (no GPU / no ML deps), through the §7 harness:
  python -m verge.l3_reasoner.run --mock                          # expert-iteration loop
  python -m verge.l3_reasoner.run --mock --engine grpo --k 8      # GRPO/DAPO loop shape

  # The §5.4 headroom run (needs verge[l3,infer] + a GPU; downloads the base):
  python -m verge.l3_reasoner.run --engine grpo --dataset gsm8k \
         --base Qwen/Qwen2.5-7B-Instruct --k 16 --rounds 4 --seeds 0 1 2

Does search → verify-filter (Ring-0 verifier) → advantage-weight/distil → reset/measure,
then fits slope + 95% CI through the new harness and prints a pre-registered go/no-go.
On the mock this validates the whole pipeline and reproduces the §5 rise-then-plateau.
The measured GSM8K result is flat from round 0 (RESULTS.md / spec §5); a positive slope
needs *headroom* (MATH, 7B+, K>>4), not a cleverer objective (spec §2).
"""
from __future__ import annotations

import argparse

from verge.eval import evaluate


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="store_true", help="CPU simulator (no GPU/ML deps)")
    ap.add_argument("--engine", choices=["expert_iteration", "grpo"],
                    default="expert_iteration")
    ap.add_argument("--dataset", choices=["gsm8k", "math"], default="gsm8k")
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--k", type=int, default=8, help="group size / samples per problem")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--train-size", type=int, default=400)
    ap.add_argument("--test-size", type=int, default=300)
    ap.add_argument("--steps-per-round", type=int, default=100)
    ap.add_argument("--outdir", default="runs/verge_l3")
    ap.add_argument("--no-vllm", action="store_true",
                    help="disable TRL's vLLM generation (use HF generate) — safer first run")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny real-GPU shakeout: 0.5B, K=4, 1 round, 1 seed, small splits")
    args = ap.parse_args()

    if args.smoke:
        args.base = "Qwen/Qwen2.5-0.5B-Instruct"
        args.k, args.rounds, args.seeds = 4, 1, [0]
        args.train_size, args.test_size, args.steps_per_round = 200, 200, 20
        args.no_vllm = True

    layer, frozen_test = _build(args)
    if layer is None:
        return 2

    # Pre-registered BEFORE the run (§7).
    if args.mock:
        preregister = {
            "hypothesis": "the full search->verify->advantage/distil->measure pipeline "
                          f"({args.engine}) runs end-to-end and the harness fits a slope+CI",
            "min_slope": 0.0, "beats_ablation": None}
    else:
        preregister = {
            "hypothesis": f"on {args.dataset} with {args.base} at K={args.k}, verified "
                          f"{args.engine} produces a positive single-shot slope (§5.4 headroom)",
            "min_slope": 0.0, "beats_ablation": None}

    report = evaluate(layer, seeds=tuple(args.seeds), rounds=args.rounds,
                      frozen_test=frozen_test, ablations={}, preregister=preregister)
    print(report.summary())

    if args.mock:
        _report_plateau(report, plateau_start=min(2, args.rounds))
        print("\nNOTE: the mock validates the pipeline; it models generation, not the "
              "verifier-reachable ceiling. The measured GSM8K result is FLAT from round 0 "
              "(slope CI includes zero) — spec §5 / m1-verified-reasoner/RESULTS.md. A "
              "positive slope needs headroom (MATH, 7B+, K>>4), not a cleverer objective.")
    return 0


def _build(args):
    """Return (EvalProtocol layer, frozen_test) or (None, None) on a wiring error."""
    if args.mock:
        if args.engine == "grpo":
            from verge.l3_reasoner.grpo import build_grpo_mock_adapter
            return build_grpo_mock_adapter(rounds=args.rounds, k=args.k,
                                           n_train=args.train_size, n_test=args.test_size,
                                           output_dir=args.outdir)
        from verge.l3_reasoner.service import build_mock_adapter
        return build_mock_adapter(rounds=args.rounds, n_train=args.train_size,
                                  n_test=args.test_size, output_dir=args.outdir)

    # --- real path: heavy deps + GPU + dataset download ---
    try:
        from verge.l3_reasoner.data import load_dataset_problems
        train, test = load_dataset_problems(args.dataset, args.train_size, args.test_size)
    except Exception as e:  # noqa: BLE001 — surface the missing-dep / network reason
        print(f"Could not load '{args.dataset}' ({type(e).__name__}: {e}).\n"
              "The real path needs `pip install -e .[l3,infer]`, a GPU, and network "
              "access to download the base + dataset. Validate free first with --mock.")
        return None, None

    if args.engine == "grpo":
        from verge.l3_reasoner.grpo import build_grpo_engine
        layer = build_grpo_engine(base_model=args.base, train=train, k=args.k,
                                  steps_per_round=args.steps_per_round,
                                  use_vllm=not args.no_vllm)
        return layer, test

    # real expert-iteration (the original M1 engine on a real model)
    from config import Config

    from verge.l3_reasoner.service import L3EvalAdapter
    cfg = Config()
    cfg.use_mock = False
    cfg.model.name = args.base
    cfg.model.samples_per_problem = args.k
    cfg.output_dir = args.outdir
    return L3EvalAdapter(cfg=cfg, train=train), test


def _report_plateau(report, *, plateau_start: int) -> None:
    from verge.eval.metrics import fit_slope

    xs, ys = [], []
    for curve in report.main.curves.values():
        for r, m in enumerate(curve):
            if r >= plateau_start:
                xs.append(float(r)); ys.append(float(m))
    if len(xs) >= 3:
        slope, lo, hi = fit_slope(xs, ys)
        flat = lo <= 0.0 <= hi
        print(f"\nplateau (rounds >= {plateau_start}): slope = {slope:+.4f}  "
              f"95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"({'FLAT — reproduces the §5 shape' if flat else 'still moving'})")


if __name__ == "__main__":
    raise SystemExit(main())
