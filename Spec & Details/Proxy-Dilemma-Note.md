# The Proxy Dilemma: Why Differentiable Guides Cannot Escape the Verifier's Reach

*A unifying negative result for verifier-guided self-improvement. Note / position draft, June 2026.*

## Abstract

Recent work shows that reinforcement learning with verifiable rewards (RLVR) does not expand a base model's reasoning support — it re-weights existing mass ([Yue et al.](https://openreview.net/forum?id=4OsgYD7em5); [The Invisible Leash, 2507.14843](https://arxiv.org/html/2507.14843v2)). We generalize this result along two axes. First, we show the bound is a property not of the *reward* specifically but of **any differentiable guide** trained on finite verifier labels — reward model, distance-to-boundary, energy function, or contrastive proxy — via a single dichotomy we call the **Proxy Dilemma**. Second, we show it is independent of the *generator's factorization*: it holds identically for autoregressive and discrete-diffusion samplers, because diffusion's full-support noise prior is irrelevant once the reverse process is trained. We state the resulting **effective-support recursion**, make explicit the one operator that *can* expand reach (discrete search, bounded by compute and verifier coverage), and argue this explains the saturation curves empirically observed in rStar-Math, d1, and ReST^EM. The contribution is a synthesis and generalization, not a new impossibility theorem; we are explicit about scope and assumptions.

## 1. Setup

- 𝒴: the (discrete) space of token sequences. 𝒴 is countable, not a manifold.
- φ: 𝒴 → ℝ^d: a fixed semantic encoder; φ(𝒴) is a measure-zero subset of ℝ^d (the "data manifold").
- V: 𝒴 → {0,1}: a **trustworthy discrete verifier** (exact-match, unit tests, proof check). Ground truth; the project's one trusted asset.
- G_θ: ℝ^d → ℝ: any **differentiable guide** (reward model, distance, energy, contrastive proxy), trained on a finite labelled set D_train ⊂ 𝒴.
- π_base: a fixed base/reference generator. For ε > 0, the **ε-effective support** is Supp_ε(π) := { y ∈ 𝒴 : π(y) ≥ ε }. (Note: for any softmax/diffusion model the *literal* support is all of 𝒴; only the effective support is operationally meaningful — nothing with probability ≪ ε is ever sampled in finite budgets.)

## 2. The Proxy Dilemma (the fork)

Any usable guide G_θ must be one of two things, and both fail to provide an escape:

**Horn 1 — G_θ is the true verifier V.** Then on 𝒴 it is integer-valued and piecewise constant; under any embedding it has no usable gradient (zero almost everywhere, undefined on the measure-zero decision boundary). There is no descent direction to follow. *No gradient to exploit.*

**Horn 2 — G_θ is a smooth learned approximation of V.** Then it is an interpolant fit on finite points D_train. Off the training manifold, the universal approximation theorem grants **no extrapolation guarantee**: G_θ(y) is not a consistent estimator of V(y), and its correlation with V is not lower-bounded. Gradient steps φ + α∇(−G_θ) therefore move toward the **adversarial optima of the interpolant** — points where G_θ predicts "correct" but V = 0. This is reward/proxy overoptimization, with the known scaling-law signature ([Gao et al.](https://arxiv.org/abs/2210.10760)): the proxy–truth gap grows with optimization pressure.

**Corollary (Proxy Dilemma).** Renaming the objective does not change the fork. "Maximize reward V_proxy", "minimize distance d_ψ", "descend energy E_θ" are all Horn 2: each is a smooth learned approximation of the same discrete oracle, and each is Goodhart-able off-support. Boundedness of the metric's range is irrelevant — Goodhart is driven by the proxy–truth gap, not by the existence of an unbounded maximum; a bounded reward (e.g. −distance, capped at 0) over-optimizes toward its bound exactly as an unbounded one does. Lipschitz continuity bounds the gradient's *magnitude*, hence step size, but not its *direction*: a bounded gradient pointing off-manifold is still wrong, merely slower.

## 3. Generator factorization is irrelevant (AR ≡ diffusion)

A common hope is that discrete diffusion escapes the leash because its forward noise prior has full support over 𝒴. It does not. Generation uses the **learned reverse process** p_θ, trained (e.g. by MLE on a success set D_succ). A model trained to approximate a distribution samples from that distribution, not uniformly over 𝒴 — its effective support concentrates on a neighborhood of its training data, exactly as a softmax autoregressive model's does. The full support belongs to the *noise*, not the *generator*. Hence the effective-support argument is factorization-invariant: it applies identically to autoregressive and diffusion samplers once they are fine-tuned on finite verifier signal. (Diffusion's non-left-to-right generation may yield a *differently shaped, plausibly broader proposal distribution* and better controllability — a real but bounded difference in *which* in-support sequences are easy to reach, not an escape from the support itself.)

## 4. The effective-support recursion (the actual "leash")

Consider one round of any guided self-improvement loop (propose with the current generator + guide, filter with V, distill the survivors). Then:

  Supp_ε(π_{k+1}) ⊆ Supp_ε(π_k) ∪ Search_V(π_k) ∪ Proj_G(Supp_ε(π_k))

where:
- **Search_V(π_k)** is the set reachable by *discrete combinatorial search* (beam, MCTS, sampling at temperature) from π_k and **confirmed** by V. This term *can* genuinely add new sequences π_k never sampled.
- **Proj_G(·)** is the contribution of the differentiable guide. By the Proxy Dilemma it is one of: (a) projection that stays near Supp_ε(π_k) to remain fluent — *adds nothing new*; or (b) projection off-manifold — *rejected by the V-filter* (V = 0), so it never enters D_succ.

**Claim (informal).** For a *fixed* base and a *fixed* verifier, the differentiable guide contributes no net expansion of the effective-confirmable support: all genuine expansion flows through Search_V. The guide can **re-weight and accelerate** sampling within the reachable set; it cannot enlarge it.

This is a *structural argument*, not a formal impossibility theorem: a rigorous version requires fixing ε, the search budget, and regularity assumptions on G_θ. We state it as a proposition with those caveats, not as a closed theorem.

## 5. Scope, caveats, and relation to prior work (read this before citing it)

- **Search is not leashed.** The result bounds the *differentiable guide*, not the whole system. With enough search compute, Search_V can be very powerful (AlphaProof finds proofs absent from the base). The honest one-liner: *gradients don't help you escape; search does, and search is bounded only by compute and verifier coverage.*
- **The bound is relative.** It is relative to a *fixed* base and a *fixed* verifier. A stronger base, or an expanding verifier surface (e.g. VSEP-style empirical/social rungs), moves the bound. This is not a claim about absolute limits of learning.
- **It is incremental.** This generalizes, and does not supersede, [The Invisible Leash](https://arxiv.org/html/2507.14843v2) (support preservation for RLVR) and [Yue et al.](https://openreview.net/forum?id=4OsgYD7em5) (RLVR bounded by base). The novelty is (i) the guide-agnostic Proxy Dilemma and (ii) the AR/diffusion-agnostic effective-support framing. Position it as a unifying note, not a landmark.
- **It is corroborated, not just argued.** The saturation curves of rStar-Math, d1, and ReST^EM are the predicted empirical fingerprint: gains compound while *search* finds genuinely new confirmable trajectories, then flatten when search saturates — never escaping to the global frontier.

## 6. Implication

Stop investing in differentiable-guide "escapes" from the verifier ceiling; the Proxy Dilemma says they relocate the bound, they don't remove it. Invest the same effort in the three levers that *do* move the effective-confirmable frontier: **(1) base-model quality**, **(2) search power** (better proposers, more compute, MCTS), and **(3) verifier coverage** (extending what can be confirmed). Everything that has actually worked — rStar-Math, AlphaProof, ReST^EM — moves one of these three, never the gradient guide.

---

*References: [The Invisible Leash (2507.14843)](https://arxiv.org/html/2507.14843v2); [Does RL incentivize reasoning beyond base? (Yue et al.)](https://openreview.net/forum?id=4OsgYD7em5); [Scaling Laws for Reward Model Overoptimization (Gao et al., 2210.10760)](https://arxiv.org/abs/2210.10760); [rStar-Math (2501.04519)](https://arxiv.org/abs/2501.04519); [ReST^EM / Beyond Human Data (2312.06585)](https://arxiv.org/abs/2312.06585); [d1 / diffusion-LLM RL](https://arxiv.org/html/2510.04019v1).*
