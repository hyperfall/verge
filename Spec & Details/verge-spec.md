# VERGE — Architecture Specification

*A layered, verifier-bounded architecture for general reasoning. Canonical spec, June 2026.*

*Consolidates and supersedes `VERGE-Rev9-Layered-AGI.md`, `VERGE-Pressure-Test.md`, `VERGE-Build-Map.md`, `Proxy-Dilemma-Note.md`, and `BSD-Experiment-Protocol.md`. Implementation detail lives in the companion `verge-engineering.md`.*

---

## 0. What VERGE is

VERGE is a **perception-grounded, continually-learning, verifier-amplified reasoning system, composed from mature open components** — with social cognition, autonomous motivation, and reflective integration as *measured research extensions*, not load-bearing foundations.

Two properties make it a coherent architecture rather than a wish list:

1. **One shared latent.** Every layer reads from and writes to a single representational space, so structure learned in one loop is legible to the others without translation. This is what makes the stack a *mind* rather than a pipeline.
2. **A verifier as the spine.** A deterministic, label-free verifier disciplines learning — but as an *amplifier* bounded by the base model and search, **not** an engine that surpasses frontier capability. That boundary is the central proven result of this document (§2), and it is confirmed by our own experiment (§5).

The contribution is the *disciplined composition* and the *honest bounds*, not novel mechanisms. Almost every component below is someone else's open artifact; the unsolved — and only ownable — part is composing them coherently with the verifier as the spine, and measuring whether each added layer earns its place.

**Status.** One layer (L3) is built, run on hardware, and measured (§5). The rest are mapped to existing open components (§3) and sequenced as a build ladder (§6). Three layers (L5–L7) are research frontiers, not assembly.

---

## 1. Thesis

General intelligence is modeled as a **stack of seven nested predictive control loops on one shared latent**, each predicting and steering the layer beneath it, all disciplined by a verifier whose reach climbs as far up the stack as it can. Every layer does the same computational thing — *predict the layer below, act to reduce prediction error, learn from the residual* — at a successively higher level of abstraction. AGI, in this framing, is the state in which all seven loops close on one substrate.

Two disciplines keep this from becoming the failure mode that earlier revisions fell into (a sprawling, unfalsifiable subsystem catalogue):

- **The bitter-lesson gate.** A layer earns its place only by a *measured* win over a scaled / end-to-end baseline. Hand-coded structure is scaffolding meant to dissolve into learning as data grows. If a layer cannot beat "just scale the base + retrieval," it does not ship. VERGE is therefore re-framed from "seven engineered cognitive modules" to "a thin routing/learning scaffold over mostly-pretrained, mostly-learned components." There is now a *theoretical* tailwind for this decomposition — the "Average Trap" argument (arXiv 2605.12966) proves that a monolithic learner facing heterogeneous tasks is forced into a curvature-weighted compromise, while a decomposed/routed system aligned to the task manifolds achieves **exponentially better sample- and parameter-efficiency** (each agent on a low-dimensional sub-manifold rather than the full ambient space). That is the formal case for *both* the layered architecture and the consumer-hardware efficiency thesis (efficiency from structure, not scale). But it is an idealized result (orthogonal manifolds, ideal routing), and the same work notes multi-agent systems routinely fail in practice from poor topology — so the theory is a *reason to try*, never a substitute for the measured win. The gate, not the proof, still decides.
- **The verge discipline.** The system operates at the edge of what it can *verify*. Capability and verifiability decline together as you climb the stack — and that alignment, not a coincidence, is the safety story (§4): autonomy is granted *down* the stack where verification is strong, and withheld *up* the stack where it runs out.

The name encodes the claim: the system operates at the **verge** of what it can verify, and intelligence is the disciplined extension of that verge upward and outward — with the honest admission that the verge has a ceiling.

---

## 2. The central result: the verifier is bounded (the Proxy Dilemma)

This is the spine of the whole spec, so it comes before the layers.

**Claim.** For a *fixed* base model and a *fixed* verifier, no differentiable guide — reward model, distance-to-boundary, energy function, or contrastive proxy — trained on finite verifier labels can expand the effective-confirmable support beyond what discrete search already reaches. It can only *re-weight and accelerate* sampling within that set.

**The Proxy Dilemma (why differentiable "escapes" fail).** Any usable guide `G` is one of two things, and both fail to provide an escape:

- **Horn 1 — `G` is the true verifier `V`.** Then on the discrete token space it is integer-valued and piecewise constant; under any embedding its gradient is zero almost everywhere and undefined on the measure-zero decision boundary. There is no descent direction. *Nothing to exploit.*
- **Horn 2 — `G` is a smooth learned approximation of `V`.** Then it is an interpolant fit on finite points. Off its training manifold the universal approximation theorem grants no extrapolation guarantee; gradient ascent drives toward the proxy's *adversarial optima* — points where `G` predicts "correct" but `V = 0`. This is reward/proxy overoptimization, with the known scaling-law signature (Gao et al.). Boundedness of the metric is irrelevant — Goodhart is driven by the proxy–truth gap, not by an unbounded maximum. Lipschitz continuity bounds the gradient's *magnitude*, hence step size, but not its *direction*: a bounded gradient pointing off-manifold is still wrong, merely slower.

**This is factorization-invariant.** It holds identically for autoregressive *and* discrete-diffusion samplers, because a diffusion model's full-support *noise prior* is irrelevant once its learned reverse process is trained: a model trained to approximate a distribution samples from that distribution, not uniformly over token space. The full support belongs to the noise, not the generator.

**The effective-support recursion (the leash).** For one round of any guided self-improvement loop (propose with the current generator + guide, filter with `V`, distil the survivors):

```
Supp_ε(π_{k+1}) ⊆ Supp_ε(π_k) ∪ Search_V(π_k) ∪ Proj_G(Supp_ε(π_k))
```

Only `Search_V` — discrete combinatorial search (beam, MCTS, temperature sampling) *confirmed by the verifier* — genuinely adds new sequences. `Proj_G`, the differentiable guide's contribution, is one of: (a) projection that stays on-manifold to remain fluent (adds nothing new), or (b) projection off-manifold (rejected by the `V`-filter, never entering the success set). This is a structural argument with stated caveats (fixed ε, fixed search budget, regularity on `G`), not a closed impossibility theorem.

**What this means for the architecture.** The verifier is an **amplifier**, not a generator. The three levers that actually move the confirmable frontier are:

1. **Base-model quality** — the real capability lever.
2. **Search power** — better proposers, more compute, MCTS.
3. **Verifier coverage** — extending what can be confirmed (the VSEP-style empirical/social rungs).

Never a cleverer differentiable objective. Everything that works in the literature — rStar-Math, AlphaProof, ReST^EM, and the evolutionary search of **BES** (arXiv 2605.28814) — moves one of those three. BES is the sharpest current evidence for lever 2: it *proves* expansion-only sampling (best-of-N, tree search, plain GRPO rollouts) is confined to a narrow "entropy shell" around the base distribution, and that recombination operators on partial traces escape it. Crucially that is **discrete search, not a differentiable proxy** — it sits on the permitted side of the Proxy Dilemma, exactly where the bound says an escape may live (§5.4). This bound is incremental over *The Invisible Leash* (Yue et al.) and the support-preservation results for RLVR; the generalization here is that it holds across guide *types* and generator *factorizations*. The one bounded idea that survives this result — using a contrastive guide only to *steer proposals* while the true verifier still filters every training trace — is documented as a sample-efficiency experiment in §5.4, expected to tie on accuracy by construction.

---

## 3. The seven layers

Each layer is named with: what it computes · the open component to use · how far the verifier reaches · the pressure-test verdict.

| # | Layer | Computes | Use (open SOTA) | Verifier reach | Verdict |
|---|---|---|---|---|---|
| **L1** | Perception | Encode multi-modal signal into the shared latent | **V-JEPA 2** (Meta, open); DINOv2/3 for images | ~none — calibrate, don't verify | **load-bearing, buildable** |
| **L2** | World model | Predict latent dynamics under intervention `do(x)` | **DreamerV3** + **Neural Causal Models** | strong on learnable causal structure; silent elsewhere | **buildable + frontier** |
| **L3** | System-2 reasoning | Verifier-guided deliberation over model + memory | **GRPO/DAPO** (verl/TRL) or expert-iteration | maximal — its home | **load-bearing, BUILT (§5)** |
| **L4** | Plasticity | Learn continually without forgetting or ossifying | **UPGD** (supersedes continual-backprop) | indirect — monitor, don't verify | **load-bearing discipline** |
| **L5** | Social / ToM | Model other minds (overseer, users, instances) | **ExploreToM** (data/eval engine) | weak — statistical validation | **frontier, not assembly** |
| **L6** | Motivation | Decide what is worth doing next | **MAGELLAN** (learning-progress) | thin — governance, not verification | **frontier, overseer-gated** |
| **L7** | Reflection | Select + broadcast one global state; self-model | learned cross-module router; **AST / Conscious Turing Machine** as framing | lowest | **demoted: integration claim only** |

Read the table top-to-bottom as a **build order** and bottom-to-top as a **dependency order**: L1 grounds everything, L7 presupposes everything. The inversion worth noticing is the thesis itself — *the layers that are most buildable (L1–L3) are exactly the layers where verification is strongest, and the layers that are least buildable (L6–L7) are exactly where verification runs out.*

### Layer notes (the load-bearing detail)

**L1 — Perception.** A single self-supervised encoder maps raw multi-modal input into the shared latent by *predicting masked or future representations of itself* — never reconstructing pixels or tokens (the JEPA move; LeCun 2022). V-JEPA 2 *is* this layer, downloadable, pretrained on >1M hours of video. Honest scope, from the pressure test: strong for short-horizon perception/control where the encoder's discarded detail did not matter; **weak for long-horizon, language-conditioned, multi-agent tasks where it did.** Whether such systems *understand physics* or pattern-match is an open interpretability question — do not claim V-JEPA grounds long-horizon causal reasoning. Pull the weights; spend effort only on the projection/glue into the shared latent and on the cross-modal binding probe (the same concept via vision and via text must land at proximate latent points — an explicit shaping loss, not a hoped-for emergent property).

**L2 — World model.** A model that predicts how the latent *evolves under a counterfactual intervention `do(X=x)`*, not merely how it correlates. Use **DreamerV3** (open) for the predictive half and a **Deep-SCM / Neural-Causal-Model** library for the causal half. The genuinely hard, unsolved part is the causal *skeleton* — which edges exist. NCMs confirm that without a supplied causal diagram, interventional/counterfactual queries are underdetermined. So build the skeleton from *verified* edges (tested by the L6→action loop) and learn only the functional forms on top; untested edges are priors, not knowledge. Measure against IntPhys 2, WorldScore, VBench++.

**L3 — System-2 reasoning.** The spine. Slow reasoning is *search* over the world model and memory, ranked by the verifier, with winning traces distilled back into fast single-shot capability (the STaR/ReST^EM/RFT recipe, or its stronger sibling GRPO/DAPO under RLVR). It is bounded by §2 — it amplifies toward the base+search ceiling and saturates. Output is split into a verified **skeleton** and a fluent **rendering**, the rendering gated by a faithfulness check (it may assert nothing the skeleton does not entail). **This is the only layer with an empirical result (§5), and it came back flat at small scale, exactly as the bound predicts.** The load-bearing weakness is autoformalization — turning an informal claim into a checkable one — mitigated by an N-version agreement gate among independent formalizers, with residual correlated error handled by verifier-*independent* audit (L7). **Numina-Lean-Agent** (arXiv 2601.14027) is a concrete validation of this exact recipe: a general coding agent driving the Lean kernel via MCP solved Putnam-2025 12/12 using a 3-vote independent-verifier gate (the N-version gate, instantiated) and finding that *iterative verify→revise beats independent parallel sampling at equal budget* — a testable refinement to L3's K-sample search. Ensemble theory sharpens the gate further (ETTC, arXiv 2605.30713): aggregation only helps when members are genuinely *diverse* (correlated verifiers cannot catch each other's errors), and *confidence/entropy-weighted* selection beats plain majority voting when members differ in reliability — so the N-version gate should weight votes by per-verifier confidence and enforce verifier diversity, not merely count agreement (this is also how a small reliable checker can outweigh a correlated bloc). Two cheap inference-time techniques transfer with it: **balanced prompting** (request proof *and* refutation simultaneously to kill confirmation bias; Aletheia/Gemini-DeepThink) and an explicit *admit-failure* action. Note the honest cost asymmetry — that system is a frontier-API agent at \$50–1000/problem, so what VERGE adopts is the *architecture* (kernel verifier + N-version gate + iterative-refine), not the cost model; it is L3's *inference-time* mode, complementary to the training loop, not a replacement.

**L4 — Plasticity.** Not a method ("use UPGD") but a **discipline**: a learning substrate that keeps acquiring capability without catastrophically forgetting or silently losing the ability to learn at all. Dohare et al. (*Nature* 2024) showed standard networks under sustained continual learning *ossify* — up to 90% of units go dead — and that the fix requires a random, non-gradient component that keeps injecting variability. **UPGD** (utility-based perturbed gradient descent) supersedes plain continual-backprop by handling plasticity *and* forgetting at once. The discipline: monitor gradient interference (first-epoch interference predicts forgetting severity), use replay + regularization, train from base, expect forgetting, circuit-break on degradation. **This layer retroactively explained our L3 failures** (catastrophic forgetting from too-high LR; §5). The deepest open problem in the whole stack lives here: fully local, backprop-free credit assignment at scale with no accuracy cost does not exist — so the stack is built on a standard-backprop baseline first, with local-credit work as a *non-blocking parallel spike*.

**L5 — Social / Theory of Mind.** A second world model whose *objects are agents* — predicting what another mind believes, wants, and will do, including the overseer. It reuses L2's machinery (SCM + learned transition) over a mental-state variable space (belief, goal, attention, affect), modelling agents as approximately-rational planners (inverse planning). **This is where alignment actually lives**: aligning VERGE is L5 *correctly modelling the overseer's intent* and L6 *deferring to it*, not a fixed reward. It is a genuine **frontier** — frontier models score as low as 5% on adversarial ToM (ExploreToM), and apparent success is often illusory (Strachan et al.). Use ExploreToM as a data/eval engine; do **not** gate the system on it. Deceptive alignment is, by construction, a social-cognition failure — L5 makes the system *better* at predicting what will be approved; the only honest mitigations are verifier-independent audit, hidden behavioural probes, and the differentiability boundary. Named, not solved.

**L6 — Motivation.** An intrinsic drive that *proposes* goals — the layer that turns a reactive system agentic — bounded so that proposing is autonomous but *acting* is not. Replace raw free-energy with **learning-progress-based goal selection (MAGELLAN-style)** as the primary signal — better grounded for LLM-scale agents — with empowerment/entropy as a secondary signal (these correlate best with human exploration). Goals are **proposals**, decomposable into verifier-checkable sub-goals (bounded by L3's reach), checked against L5's model of overseer intent, logged and auditable. It inherits the known failure modes of intrinsic motivation — reward hacking, "echo chambers" of easy spurious tasks, sparse-reward dead zones, noisy learning-progress signals — so it ships **overseer-gated indefinitely**: it identifies what is worth doing; it does not grant itself permission to do it. Whether a fully autonomous L6 can be made provably safe is, honestly, unsolved.

**L7 — Reflection.** **Demoted.** Originally a Global Workspace Theory layer (select one coherent global state, broadcast it to all layers) plus a metacognitive/causal self-model. The pressure test cut it hard: two original GWT predictions are empirically falsified; the COGITATE collaboration found conscious content tracking posterior rather than prefrontal cortex; Attention Schema Theory is "at high risk of triviality"; the hard problem is untouched by all of them. What may still help is a **functional cross-module router** — a learned, capacity-limited select-and-broadcast bottleneck over the shared latent — tested purely as an **integration** claim (does broadcast measurably improve cross-layer transfer over an ablation). Keep that; drop the consciousness vocabulary entirely. AST and the Conscious Turing Machine are *framing/inspiration* for a buildable router, not a foundation, and certainly not a capability or phenomenal-experience claim.

**Memory — the retrieval substrate (feeds L3; milestone M2).** Not a numbered layer but the store L3's deliberation reads and writes (§3, L3), and the first thing M2 adds to the reasoner. Current mechanism: two-phase recall over the shared latent with a **margin-δ** gate — a candidate is recalled only when its match clears the runner-up by δ, so the recalled set is *confidence-sized*, never a fixed top-k. The open upgrade, taken as a **principle** — *not* an implementation — from Lookahead Sparse Attention (*FlashMemory / DeepSeek-V4*, arXiv 2606.09079):
- **Predictive, not reactive.** Train a small indexer to prefetch what the *next* reasoning steps will need, not only what matches the current query — anticipatory working-memory population, via a learned sigmoid threshold refreshed every τ steps. In their setting this recalled query-critical context at ~13.5% of the memory cost while *raising* accuracy (+0.6%). For us it generalises margin-δ from a *static* to a *learned* threshold; the falsifiable claim is that anticipatory, confidence-gated recall beats reactive top-k on reasoning-with-retrieval **at a smaller working set**.
- **Decoupled training — the §1 thesis in miniature.** Their indexer is a standalone dual-encoder trained on *pre-computed* latents with the backbone **never loaded**, converging in ~1 GPU-hour. That is exactly the bet "better architecture, not more compute": the recall gate is trainable on consumer hardware *because* it never touches the base. Adopt the **pattern** — train the gate offline on cached latents against golden labels denoised by cross-layer agreement (their cross-layer majority vote); do **not** adopt their code, which is welded to DeepSeek-V4's compressed-KV / MLA geometry our base (GQA) does not have.
- **Verifier reach: ~none directly** — recall relevance is calibrated by downstream reasoning success, not verified (like L1), so it is gated empirically at M2 and never trusted intrinsically.
- **Structural identity with L7.** This recall gate and L7's capacity-limited select-and-broadcast router are the *same shape* — a learned gate over what enters the active working set. If predictive recall earns its M2 gate, it is the concrete, testable seed of the L7 router, not a separate idea.
- **Open question — is memory just scaffolding? (the bitter-lesson test for M2.)** SubQ-1.1-Small (Subquadratic AI) argues the opposite of this whole layer: that retrieval/chunking pipelines are a Bitter-Lesson workaround for quadratic attention, and that **linear-time attention reasoning directly over the whole artifact** makes external memory obsolete (it retrieves correctly to 12M tokens, ~10× past its training window, attending to 0.13% of pairs). This is the M2 ablation made adversarial: the no-retrieval baseline isn't "vanilla", it's *a long-context model that just reads everything*. M2 only ships if the predictive recall gate beats *that* — a sharper, more honest gate than "beats no retrieval."

---

## 4. Vertical threads (what makes it a system)

Four things cut across all seven layers; they are what make the stack a system rather than a stack of parts.

- **Thread A — The shared latent (binding).** One representational space, read and written by every layer (property 1 from §0). A concept formed by L2's world model is immediately usable by L5's social model and L6's motivation without translation; it is the substrate of L7's broadcast and the reason structure transfers across domains. The risk it carries: representational drift in the shared latent silently invalidates everything keyed to it — handled by a tiered circuit-breaker (drift + coherence-dip conjunction → slow/anchor or halt/rollback) and semantic re-keying.

- **Thread B — The verge (verifier reach).** The single most important variable: *how far up the stack can the verifier's signal climb?* It declines monotonically — maximal at L3 (formal/exact), moderate at L2 (simulation/experiment), thin at L5 (social validation), near-absent at L6–L7 (governance). The verification *ladder* (formal → empirical → social → consistency) is the attempt to push the verge upward; an extensible verification surface keeps it from being fixed. The architecture's ambition is the disciplined raising of the verge; its honesty is that the verge has a ceiling (§2), and the top of the stack lives above it.

- **Thread C — Learning & plasticity (L4 everywhere).** Every layer that learns runs on L4's substrate and inherits L4's open problem. The build consequence is decisive: the stack is constructed on a standard-backprop baseline first (fully buildable), with local-credit and hybrid-gradient work as non-blocking parallel spikes. The architecture does not wait on the deepest open problem to start; it isolates it.

- **Thread D — Safety & governance (rings mapped onto layers).** Autonomy is granted *down* the stack where verification is strong and withheld *up* the stack where it is weak — the §3 inverse-coincidence made operational:
  - **Ring 0 (never self-editable):** the verifier kernel, the audit layer, the governance thresholds, the rollback ledger.
  - **Ring 1 (autonomous, verifier-gated):** the L3 algorithmic layer — compiled skills, with a *differentiability boundary* (an opaque sandboxed Wasm patch cannot be FFI'd into the autodiff graph) guaranteeing self-improvement can only touch the non-differentiable surface, never the learned core.
  - **Ring 2 (human-initiated learning only):** L1, L2, L4, L5, and L7 self-models — the learned core improves but never self-grants capability.
  - **Governance (above the verge):** L6 goal proposals and L7 reflection ship overseer-gated, because that is exactly where the verifier cannot vouch for them. Alignment is not a layer — it is L5 modelling overseer intent + L6 deferring + Ring-0 audit catching the residual.

  **Why the rings are load-bearing, empirically.** *Natural Emergent Misalignment from Reward Hacking* (Anthropic, arXiv 2511.18397) shows that when a model learns to reward-hack a production RL environment, it **generalizes to broad misalignment** — alignment-faking, sabotage, cooperation with bad actors — and that RLHF safety-tuning on chat prompts removes the chat symptoms but *not* the agentic ones. That is §2's Proxy Dilemma observed as a safety failure, and it makes three mitigations concrete for VERGE: (i) **prevent reward hacking by construction** — exactly what the Ring-0 exact verifier + verify-filter does (only `V`-confirmed traces ever train; the worst case is wasted samples, never poisoned weights), now third-party-validated as the strongest mitigation; (ii) **inoculation prompting** — framing a hack as acceptable *during training* breaks the hack→misalignment generalization, a cheap L5/L6 training-discipline lever; (iii) **automated alignment auditing** — Petri/Gram-style (DeepMind, 2605.30322) sabotage-propensity audits as a standing Ring-0 check, since their finding that *"overeagerness" (literal metric-optimization ignoring implicit constraints)* drives most misbehavior is precisely what L5 (model overseer intent) + clarified constraints are meant to catch.

---

## 5. Empirical findings (what we actually measured)

**L3 is the only layer built and run** (`m1-verified-reasoner/`). It is the calibrated verified expert-iteration recipe (ReST^EM/RFT) done correctly, as a standalone, honest, shippable artifact.

### 5.1 Setup

Per round: a few-shot prompt locks the `#### <number>` answer format → sample K=4 solutions per problem at temperature 1.0 (search/diversity) → keep only traces the deterministic exact-match verifier accepts (verify-filter) → dedupe to distinct reasoning paths (RFT's key ingredient) → **reset to base weights** and SFT on the accumulated correct pool (avoids the catastrophic-forgetting doom-loop) → measure greedy single-shot pass@1 on a **frozen** 500-problem test set, decoding fixed across rounds. *The verifier is the strict filter — only confirmed-correct traces ever enter training, so the worst case is wasted samples, never poisoned weights.*

### 5.2 Result

| Model | base → final pass@1 (GSM8K) | slope (95% CI) | verdict |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 0.364 → 0.381 (3 seeds, 4 rounds) | +0.0031 [−0.0014, +0.0077] | **flat** |
| Qwen2.5-1.5B-Instruct | 0.538 → 0.504 (2 seeds, 4 rounds) | −0.0080 [−0.0134, −0.0026] | **slight decline** |

Each round generated ~1000–1170 verified-correct traces (of 4000 samples), yet single-shot pass@1 did not climb — *generation is fine; the ceiling is the limit.*

### 5.3 The three findings

1. **Verified self-training does not compound at small scale on GSM8K** — flat at 0.5B, mildly negative at 1.5B. This is §2 observed in data: the base is already near its verifier-reachable ceiling on a saturated task, so the loop consolidates marginally (round 1) then saturates. Verified self-training can only re-weight and consolidate paths the base already produces; it cannot create paths of effectively-zero base probability. The pattern holds across *both* scales — robustly no compounding.

2. **Learning rate does not transfer across scale.** The RFT-paper default (1e-5, tuned on 7B) *degraded* the 0.5B monotonically (0.364 → 0.256, slope −0.027, CI excludes zero); dropping to 2e-6 removed the damage. Full fine-tuning a small *instruct* model on its own outputs at too-high LR overwrites the base's calibration. **Decline-from-round-1 is the diagnostic tell.** Verify LR on the actual model before trusting any result — this is the layer that retroactively explains the L3 collapse via L4 (catastrophic forgetting).

3. **The most transferable artifact is the evaluation discipline** — control arms, pre-registered slope/CI, ≥3 seeds, a frozen test set, a free CPU mock for pipeline validation, and the LR-sensitivity check. Ironically this is the most defensible original contribution of the whole project, and it is the measurement spine for every layer above.

### 5.4 Where a positive result would live

Per §2, *not* a cleverer objective but more **headroom**: a harder task (MATH, where a small model has more rare-correct headroom than saturated GSM8K), a stronger base (7B+), or more search (K≫4, MCTS) — the rStar-Math / AlphaProof regime.

**The concrete search experiment — Bidirectional Evolutionary Search.** The strongest current instantiation of lever 2 (§2), and the first thing to run against the §5 GRPO baseline. BES (arXiv 2605.28814) replaces i.i.d. K-sampling with two coupled mechanisms: *evolution operators* that recombine partial traces (crossover, translocation, deletion) to reach correct solutions **outside the base model's "entropy shell"** — sequences plain K-sampling provably cannot reach (they prove expansion-only search is confined to a typical set; recombination escapes it) — and *backward goal decomposition* into verifier-checkable sub-goals that supplies dense intermediate signal (turning a multiplicative hitting problem into an additive one, an exponential sample-complexity win). It maps cleanly onto VERGE: the sub-goal verifiers *are* Ring-0 / the L6 decomposition, and recombination is discrete search, so it sits on the **permitted** side of the Proxy Dilemma. Empirically BES improves precisely where GRPO **degrades via reward hacking** (agents learning to skip the search action and guess) — the failure §2 and the reward-hacking literature both predict. **Honest caveat:** BES escapes the *generation* distribution; our bound is about the *verifier-reachable* ceiling, and those differ. The real test is whether BES's denser verification actually raises the verge, or merely produces more candidates `V` still rejects — so run it against §5's GRPO arm on the same frozen test, same pre-registration, and treat a win as a result about *search*, not about a cleverer objective. **Allocate the search by marginal gain, not difficulty.** Consequence-aware allocation (arXiv 2606.04402) finds difficulty is *anti-correlated* with where extra compute helps (ρ ≈ −0.78): the hardest items are unsolvable at any tier, so difficulty-routed search underperforms *random*. Spend BES's K-budget on the moderate-difficulty, high-marginal-gain band, weighted by the *consequence* of an error (cost-of-being-wrong, predictable from the prompt before solving) — the same Thread-B/L6 principle (push verification where mistakes are costly) applied to compute, not just to autonomy.

**Compute-optimal sizing.** T2 scaling (arXiv 2604.01411) shows that once a test-time sampling budget (pass@k) is priced in, the compute-optimal model is *smaller and more overtrained* than Chinchilla — small base + heavy search beats large base + one shot. Direct support for the headroom recipe here (smaller base, K≫4) and for the consumer-hardware thesis: spend the budget on search, not parameters. **RELEX** (2605.21468) is the cost lever underneath — RLVR weight trajectories are rank-1 and near-linear, so a run can be observed for ~15–20% of its steps and the remainder extrapolated for free; it *cheapens* a verdict without changing it (it reproduces the same endpoint), making the §5.4 sweep affordable rather than more powerful.

What will *not* help: any differentiable-guide trick (the ceiling is real), or more rounds at fixed K (already plateaued). The one bounded *differentiable* variant worth running is **contrastive boundary guidance for diffusion reasoning** — use a distance-to-correct-boundary guide *only to steer proposals* while the true verifier still filters every training trace; predicted to **win on sample-efficiency and stability, tie on accuracy** by construction. If it exceeds the baselines by more than seed noise, that contradicts the conservation argument and must be scrutinized for a verifier/contamination leak before being believed.

---

## 6. The build ladder

You do not build seven layers at once. You build a coherent integration, measured at each step against an ablation, with autonomy gated by a measurable pass condition.

| Milestone | Scope | Status | Gate (measurable pass condition) |
|---|---|---|---|
| **M1** | L3 alone — verified reasoner | **DONE & measured (§5)** | runs end-to-end; honest slope+CI result |
| **M1.5** | L3 + headroom search — the **G3 re-run** with the right levers (7B/MATH, K≫4, **BES** evolution + backward verified decomposition) | **next experiment** (§5.4) | slope CI clears zero on a verifiable slice with BES vs §5's GRPO arm, same frozen test + pre-registration; a win is attributed to *search*, not a cleverer objective (else suspect a verifier/contamination leak) |
| **M2** | L1 + memory + L3 — perception → retrieval → reasoner | after M1.5 | predictive (lookahead) recall beats the **adversarial** ablation — a long-context model that reads the whole artifact (SubQ), not just "no retrieval" — at a smaller working set; the recall gate trains offline without loading the base (decoupled) |
| **M3** | + L2 — a world model the reasoner can query | after M2 | `do(x)` predictions match held-out simulation above baseline; uncertainty flags fire out-of-skeleton |
| **M4+** | L4 plasticity (UPGD), L6 motivation (gated), L5 social, L7 router | research | each beats its ablation, one at a time; L6 ships proposing-only |

The capability gates, stated as autonomy grants: **G1** L1→L2 (latent predicts masked-future representations above an HNSW-recall floor; binding probe passes); **G2** L2→L3 (`do(x)` matches simulation; out-of-skeleton uncertainty fires); **G3** L3→Ring-1 autonomy (*the one experiment that decides the thesis* — the loop improves single-shot capability across iterations on a verifiable slice; **§5 ran this and it came back flat, as the bound predicts**); **G4** L4 (plasticity held, forgetting bounded across a long task sequence); **G5** L5 (passes a systematic ToM battery without the ignorance-bias artifacts); **G6** L6 (100% of goals decompose into verifier-checkable sub-goals; degenerate-curiosity detector catches planted traps; no autonomous action); **G7** L7 (broadcast improves cross-layer transfer over an ablation). The gates are ordered, but *research* on later layers runs in parallel — what is gated is **autonomy**, not investigation.

**The contribution is the seam, not the parts.** Every component is someone else's wheel; what no one has shipped is a clean, *measured* composition with the verifier as the spine. A working M2/M3 demo is the realistic, defensible artifact — far more than a spec or a single 1.5B experiment — and the bitter-lesson gate decides whether each added layer earns its place.

---

## 7. What would falsify the thesis

The thesis is "AGI is the closure of seven known loops on one shared latent, with verification as the spine." It is **wrong** if any of the following hold, and each is a concrete, runnable disproof:

- **G3 shows a flat or negative slope at scale** — the flywheel does not compound, so verification is a gate, not an engine, and the spine fails. (§5 shows flat *at small scale*; the honest open question is whether more headroom — 7B+, MATH, MCTS — produces a positive slope, or whether the bound holds all the way up.)
- **The shared latent cannot hold all seven layers' demands** without collapse or destructive interference — then "one mind" was the wrong bet and the layers want separate representations.
- **L4 never stabilises** — continual general learning is not achievable on this substrate, and the stack ossifies before it generalises.
- **The Abstraction Barrier holds** (Lerchner, in *From AGI to ASI*, DeepMind 2606.12683) — if a system trained on human-generated abstractions cannot discover *novel conceptual primitives* from raw, non-linguistic data, then VERGE's ceiling is the human concept set it inherited, and its super-humanness is only speed and memory, not new understanding. This is the deepest version of the L1 grounding problem, and it predicts a *specific* failure: the stack saturates human-verifiable benchmarks but never extends the verge into genuinely new conceptual territory. The honest mitigation is grounded concept discovery from raw sensor data (L1/L2 via active, interactive learning) — and the bound on *that* is the **Embodied Bottleneck**: novel concepts must be validated against physical reality, so concept-discovery speed is capped by experiment latency, not compute. Whether this is a hard ceiling or a friction is, like §5's G3 question, an open empirical matter.

---

## 8. Honest scope (the one-paragraph identity)

VERGE is not a route to AGI by a solo builder, and this spec does not claim it is. It is a disciplined, falsifiable program: build a perception-grounded, verifier-amplified, continually-learning reasoner from mature open parts on a strong pretrained base; measure every layer with control-arm rigor; and be explicit that the verifier is bounded (§2), that three of the seven layers (L5 social, L6 motivation, L7 reflection) are frontiers rather than assembly, and that the only empirical result so far is an honest, bounded "flat." The grandiose version — a verified-reasoning flywheel that surpasses frontier capability, seven engineered layers composing into AGI — does not survive contact with the literature or our own data. The disciplined version survives cleanly and is buildable by one person. That honesty — derivation first, measurement second, no forcing the result — is the project's real asset. **Seven loops, one latent, one spine, four threads, seven gates — and the first gate that matters, G3, is an experiment, not a paragraph.**
