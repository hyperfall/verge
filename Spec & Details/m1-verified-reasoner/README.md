# M1 — Verified Reasoner (Layer 3, expert-iteration)

The first **buildable, runnable** piece of VERGE: a self-improving math reasoner that
trains only on its own *verifier-confirmed* solutions. This is Layer 3 (System-2
reasoning) as a standalone, honest, shippable artifact.

## What it is (and what it isn't)

It is the calibrated **expert-iteration / ReST^EM / RFT** recipe — the proven way to
lift small models on GSM8K — done correctly:

1. **Format** — a few-shot prompt locks the `#### <number>` answer format so the verifier reads answers cleanly.
2. **Search** — sample K solutions per problem at temperature 1.0 (diversity).
3. **Verify** — keep only traces the deterministic exact-match verifier accepts. *The verifier is the strict filter; we never train on unconfirmed traces, so the worst case is wasted samples, never poisoned weights.*
4. **Dedup** — keep only **distinct reasoning paths** (RFT's key ingredient — training on near-duplicates overfits and degrades).
5. **Train from base** — reset to the base weights each round and SFT on the accumulated correct set (avoids the catastrophic-forgetting doom-loop).
6. **Measure** — single-shot greedy pass@1 on a **frozen** test set, decoding fixed across rounds.

It is **not** a ceiling-breaker. Per `../Proxy-Dilemma-Note.md`, expert-iteration is
bounded by base capability + verifier-confirmable search. The honest goal is a **clean,
rising pass@1 curve on the verifiable slice** — a real, working L3 you understand end to end.
The next-step upgrades (documented, not built here) are GRPO via TRL/verl, or the formal
Lean curriculum (DeepSeek-Prover-V2 style).

## Run it

**Step 0 — validate the whole pipeline for FREE (no GPU, no ML deps):**
```bash
python run.py --mock
python run.py --aggregate --outdir runs
```
The mock simulates the learning dynamic so you can confirm generate→verify→dedup→train→measure
and the aggregation all work before spending a GPU-hour.

**Step 1 — real run (one consumer GPU; start small):**
```bash
pip install -r requirements.txt
python run.py --model Qwen/Qwen2.5-0.5B-Instruct --seeds 0 --rounds 4 --train-size 400 --test-size 500
python run.py --aggregate
```

**Step 2 — scale once it works:** bump to `Qwen/Qwen2.5-1.5B-Instruct`, more seeds, `--train-size 1000`.

Knobs live in `config.py`. `--wall-clock-min N` stops a run after N minutes (partial logs saved).

## Files

```
config.py            all knobs (recipe defaults)
verifiers.py         ExactMatchVerifier + robust answer extraction   (pure stdlib)
data.py              GSM8K loader + contamination guard
model.py             HF wrapper: few-shot prompt, batched gen, reset-to-base, SFT
expert_iteration.py  the loop: search -> verify -> dedup -> train-from-base -> measure
metrics.py           pass@1, distinct-path dedup, slope+CI                (pure stdlib)
mock_model.py        CPU simulator for the free smoke test
run.py               entrypoint (+ --aggregate)
```

## How to read the result

`python run.py --aggregate` prints per-seed r0→final pass@1 and the overall slope + 95% CI.
- **Slope positive, CI excludes 0** → the layer works: verified self-training lifts single-shot reasoning. A real, defensible L3.
- **Flat** → at this model/size the base is already near its verifier-reachable ceiling; try a smaller base (more headroom) or confirm the recipe with the mock first.
- Use ≥3 seeds before trusting the slope; keep the test set frozen.

*Layer 3 of VERGE. Bounded by design (see `../Proxy-Dilemma-Note.md`); buildable today.*
