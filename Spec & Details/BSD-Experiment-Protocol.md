# Contrastive Boundary Guidance for Discrete-Diffusion Reasoning

*An honestly-scoped experiment protocol. June 2026. Companion to `Proxy-Dilemma-Note.md`.*

## 1. What this is (and is not)

This is **not** an attempt to break the verifier ceiling — the Proxy-Dilemma note shows that ceiling holds for any differentiable guide. This is the *bounded* idea that survives: a **variance-reduction / sample-efficiency** improvement to RL fine-tuning of diffusion reasoning models. The claim is narrow and falsifiable.

> **Hypothesis.** Replacing the sparse binary RLVR reward in diffusion-LLM reasoning (as in **d1**) with a **contrastive distance-to-correct-boundary guide** yields a smoother, lower-variance gradient field, so the loop reaches its (unchanged) accuracy ceiling in **fewer rollouts / less wall-clock**, *without* introducing new adversarial failure — because the **true discrete verifier V remains the strict filter** on the training set.

Expected result, stated up front for honesty: **wins on sample-efficiency and training stability; ties on final accuracy** with d1 / rStar-Math (the ceiling is set by search + verifier coverage, per the conservation argument, not by the guide).

## 2. Why it's sound (the one good design choice)

The guide d_ψ is Goodhart-able off-manifold (Proxy Dilemma, Horn 2) — but it is used **only to steer proposals**, never to label training data. The trustworthy discrete V filters every trajectory before it enters D_succ. So the worst case for a hacked guide is **wasted samples** (proposals that score low d_ψ but fail V and are discarded), not **poisoned weights**. This is the rStar-Math discipline (search proposes, true verifier disposes) ported to a diffusion proposer.

## 3. Method

1. **Base generator.** A pretrained discrete-diffusion LLM — **LLaDA** or **SEDD** (open). Reuse the **d1** codebase as the RL/training harness.
2. **Seed data.** One bounded **rStar-Math** run to collect verifier-labelled correct (y⁺) and incorrect (y⁻) reasoning traces on the target set (GSM8K, then MATH).
3. **Boundary guide d_ψ.** A frozen semantic encoder φ (SimCSE/SBERT) + a contrastive head trained with a margin triplet loss:

   L = E_{(y, y⁺, y⁻)} [ max(0, ‖φ(y) − φ(y⁺)‖² − ‖φ(y) − φ(y⁻)‖² + δ) ]

   d_ψ(y) approximates distance to the nearest *confirmed-correct* trajectory.
4. **Guided reverse process.** During denoising, apply guidance toward the boundary:

   ε_θ(y_t, t) ← ε_θ(y_t, t) − λ · ∇_{y_t} d_ψ( ŷ₀(y_t) )

   λ swept; classifier-free variant as an ablation.
5. **True-verifier filter + distill.** Generate N trajectories, keep only V(y) = 1, add to D_succ, retrain the diffusion model by MLE on D_succ. Iterate.

## 4. Baselines, metrics, ablations

**Baselines (the comparison is the point):**
- **d1** (diffusion LLM + standard RLVR/GRPO) — the primary baseline.
- **rStar-Math** (AR + MCTS + PRM) — the AR reference ceiling.
- **No-guide ablation** (diffusion + V-filter + distill, λ = 0) — isolates the guide's contribution.
- **Reward-guide ablation** (guide = reward proxy instead of distance) — isolates distance-vs-reward.

**Primary metric — sample efficiency:** rollouts (and wall-clock) to reach X% of the final accuracy plateau. *This is where the hypothesis lives.*

**Secondary — stability:** gradient/return variance across steps; entropy trajectory (does it avoid the GRPO entropy-collapse / mode-collapse failure?).

**Control — final accuracy:** pass@1 on GSM8K/MATH at plateau. **Predicted to tie** the baselines. If it *exceeds* them by more than seed noise, that is a surprise that would *contradict* the conservation argument — and must be scrutinized for a verifier/contamination leak before being believed.

**Eval rigor:** reuse the G3 discipline — frozen test split, ≥3 seeds, pre-registered thresholds, contamination check.

## 5. What would falsify the hypothesis

- **No sample-efficiency win** (BSD ≈ d1 in rollouts-to-plateau) → the contrastive guide isn't smoother in practice; idea dead.
- **Worse stability** (more collapse than d1) → the guide is injecting variance, not removing it.
- **Final accuracy *below* baselines** → the guide is steering proposals *away* from the confirmable region (d_ψ poorly calibrated); fixable or fatal depending on degree.

A clean negative on all three is still a publishable result (negative result on diffusion guidance), given the conservation note frames the expectation.

## 6. Risks (named)

- **Long-chain instability.** Guided discrete diffusion over 500–1000+ token reasoning chains is known-hard (mode collapse, incoherence). This is the real engineering risk; start short (GSM8K, ≤256 tokens) before MATH.
- **Guide miscalibration.** d_ψ trained on a small seed set may steer poorly; mitigate by refreshing d_ψ each round with hard negatives from the diffusion failures.
- **Compute.** Diffusion sampling is multi-step; budget accordingly. Start at the smallest viable LLaDA/SEDD checkpoint.

## 7. Honest positioning (for the writeup)

Cite **d1**, **LLaDA**, **SEDD**, and **Diffusion-LM** reward-guided variants explicitly. Frame BSD as a **stability / sample-efficiency improvement to the RL objective for diffusion reasoning**, with a true-verifier safety filter — *not* a paradigm shift, *not* a ceiling-breaker. The accompanying conservation note (Part 1) sets the expectation that it ties on accuracy by design, which makes the sample-efficiency claim the clean, defensible contribution.

## 8. Scope

Implementable in ~2–3 weeks on existing **LLaDA / d1** codebases for one person: seed run (days) → train d_ψ (hours) → guided loop on GSM8K (days) → ablations + writeup. MATH and long-chain stability are stretch goals, not gating.

---

*References: [d1 / diffusion-LLM RL](https://arxiv.org/html/2510.04019v1); [Discrete Diffusion LLM survey (2506.13759)](https://arxiv.org/pdf/2506.13759); [Simple Guidance for Discrete Diffusion](https://discrete-diffusion-guidance.github.io/); [rStar-Math (2501.04519)](https://arxiv.org/abs/2501.04519); [Reward-guided diffusion / overoptimization review (2501.09685)](https://arxiv.org/pdf/2501.09685).*
