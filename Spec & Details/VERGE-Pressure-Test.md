# VERGE Pressure Test — Critiques, Practitioner Lessons, and the Hardened Spec

**Date: June 2026. Companion to `VERGE-Rev9-Layered-AGI.md` and `VERGE-Build-Map.md`.**

This document stress-tests every load-bearing claim in the spec against (a) researcher critiques and (b) what people actually learned trying to use these components. The goal is not to defend the spec — it's to find where it's wrong and fix it. Each item is tagged: **[survives]**, **[downgrade]** (claim must weaken), **[demote]** (layer is less central than the spec implies), or **[frontier]** (unsolved, not assembly).

The single most important finding up front: **the literature's strongest critique lands on our spine (L3), and it confirms what our own G3 experiment showed.** That's good news for the spec's honesty and bad news for its ambition — exactly the trade a pressure test should surface.

---

## 0. The meta-test: the Bitter Lesson vs a 7-layer hand-designed architecture  **[downgrade]**

**The critique.** Sutton's bitter lesson — *flexible, learned methods beat handcrafted domain knowledge in the long run* — is the central objection to VERGE's entire shape. Hand-designed modular cognitive architectures (LIDA, SOAR, ACT-R lineage) "run end-to-end but have **not** scaled to web-scale learning" ([survey context](https://www.intelligencestrategy.org/blog-posts/agi-architectures-what-we-can-agree-on)). A seven-box architecture with hand-specified roles is precisely the pattern that has repeatedly lost to scaling.

**What practitioners/researchers learned.** The 2025–2026 consensus isn't "modularity is dead" — LeCun's own world-model + actor + configurator is modular, and ARC-AGI work shows recursive refinement/search winning without raw scale. It's that **the winning systems make every component *learned and scalable*, and keep hand-coded structure to a minimum.**

**Hardening.** Adopt a strict **bitter-lesson gate**: a layer earns its place only by a *measured* win over an end-to-end / scaled baseline, and any hand-coded structure is scaffolding meant to *dissolve into learning* as data grows. VERGE is re-framed from "seven engineered cognitive modules" to "a thin routing/learning scaffold over mostly-learned, mostly-pretrained components." If a layer can't beat just scaling the base model + retrieval, it doesn't ship.

---

## 1. L3 — The verifier spine: the hardest test  **[downgrade — important]**

**The critique (this is the big one).** A direct body of work asks whether RLVR/GRPO *teaches new reasoning* or merely *amplifies the base model*:
- *"Does RL Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?"* ([OpenReview](https://openreview.net/forum?id=4OsgYD7em5)) — systematic finding: **RLVR does not elicit fundamentally new reasoning patterns; performance is fundamentally constrained by the base model's capability**, biasing its output distribution toward reward-maximizing paths. RLVR-tuned models can *underperform* the base on pass@k (reduced diversity).
- **Exploration collapse:** when all sampled trajectories in a group are wrong, the group advantage is zero → *no gradient*. The method literally cannot learn what the base never gets right even once. (This is exactly our G3 "too weak to bootstrap" failure — now confirmed as a named, fundamental limit.)
- RLVR is **restricted to domains with verifiers** (STEM-ish).

**Counter-evidence (for honesty).** The debate isn't fully settled — pass@k is argued to be a flawed metric, and *"RLVR Implicitly Incentivizes Correct Reasoning in Base LLMs"* ([2506.14245](https://huggingface.co/papers/2506.14245)) finds genuine benefit. So the truthful position is *bounded*, not *worthless*.

**Practitioner lessons (GRPO in the wild).** Entropy collapse → mode collapse to a single repetitive chain; reward/training collapse when started from base; **length bias** (advantage ÷ length → long *wrong* answers under-penalized → the model learns to ramble); general instability ([Raschka's survey](https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training)).

**What this does to the thesis.** The spec's framing — "verifier as the *engine* that surpasses frontier reasoning" — must **downgrade** to: **the verifier is an efficient *amplifier* that reaches the base model's latent ceiling on verifiable tasks, not a generator of new capability.** Its ceiling is the base model. This is *precisely what G3 showed* (verified arm was protective and beat controls, but did not compound past the base — it can't, by this result). The honesty is now load-bearing, not decorative.

**Hardening.**
1. **The base model is the real capability lever**, not the RL loop. Pick the strongest base you can; RL is polish.
2. Use **DAPO / entropy-controlled variants** to fight collapse and length bias; don't run vanilla GRPO from base.
3. Keep the **verifiable-domain boundary** explicit — this is not a general-reasoning solution.
4. Reframe the spec's L3 claim accordingly (done in this doc's summary).

## 2. L1 / L2 — Perception & world model  **[downgrade]**

**Critique.** *"Sora and V-JEPA Have Not Learned the Complete Real World Model"* ([2407.10311](https://www.arxiv.org/pdf/2407.10311)). Whether these systems *understand physics* or perform *sophisticated pattern matching* is an **open question** — the interpretability to settle it hasn't been done at scale.

**Practitioner lesson.** V-JEPA 2 wins on **short-horizon perception and control**, where the information the encoder discards genuinely didn't matter. It loses on **long-horizon, language-conditioned, multi-agent** tasks where the discarded detail *does* matter. JEPA's core move (abstract away the unpredictable) is simultaneously its strength and its ceiling.

**Hardening.** Use V-JEPA for L1 perception/short-horizon grounding only. **Do not claim it grounds long-horizon causal reasoning.** The causal skeleton (L2, verified edges) remains the genuinely hard, unsolved part — and Neural Causal Models confirm it: without a supplied causal diagram, interventional/counterfactual queries are underdetermined.

## 3. L4 — Plasticity / continual learning  **[survives, as a discipline not a method]**

**Critique/lesson.** Catastrophic forgetting during sequential fine-tuning is real, mechanistically localized to specific circuits, and **was exactly our G3 collapse** ([Mechanistic Analysis, 2601.18699](https://arxiv.org/abs/2601.18699)). A practical tool emerged: **gradient interference in the first epoch predicts forgetting severity** — you can forecast a bad task sequence before committing to it.

**Hardening.** L4 is not "use UPGD." It's a **discipline**: monitor gradient alignment, use replay + regularization, expect forgetting, train from base. UPGD is one good tool (handles plasticity *and* forgetting); replay-based methods (e.g. FOREVER) are complementary. This layer *survives* the pressure test because forgetting is a confirmed, central, measurable problem — and it retroactively explains our experiment.

## 4. L5 — Social intelligence / Theory of Mind  **[frontier]**

**Critique.** Frontier models score **as low as 5%** on adversarially-generated ToM problems ([ExploreToM](https://ai.meta.com/research/publications/explore-theory-of-mind-program-guided-adversarial-data-generation-for-theory-of-mind-reasoning/)); evaluation metrics are ambiguous; apparent ToM is brittle and often illusory (Strachan et al.).

**Hardening.** Unchanged from the build map, now firmer: **L5 is a research frontier, not an assembly step.** Use ExploreToM as a data/eval engine. Do not let the system's value depend on it.

## 5. L6 — Motivation / intrinsic goals  **[downgrade]**

**Critique/lesson.** Intrinsic-motivation/open-ended systems are notoriously hard to make work: **reward hacking** (agents form "echo chambers" of easy/spurious tasks and need periodic re-alignment), sparse-reward dead zones, **noisy learning-progress signals**, heavy auxiliary-model engineering, poor reproducibility ([RLeXplore](https://arxiv.org/pdf/2405.19548), [Lil'Log on reward hacking](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/)).

**Hardening.** The spec's "free-energy frontier / autotelic goals" inherits all of this. Keep L6 **overseer-gated** (the spec already does), but state plainly that autonomy here is both *dangerous* and *brittle* — expect degenerate exploitation, budget for periodic re-alignment, and treat MAGELLAN-style learning-progress as a noisy heuristic, not a reliable drive.

## 6. L7 — Consciousness / reflection  **[demote — significant]**

**Critique.** This layer takes the heaviest hit. **Two original GWT predictions have been empirically falsified**, and the large adversarial **COGITATE** collaboration found conscious content tracking *posterior* cortex rather than prefrontal networks — meaning *broadcast, higher-order representation, and attention schemas may none be necessary*. Attention Schema Theory is flagged as **"at high risk of triviality"** (it explains why systems *claim* awareness, not why they have it). The hard problem is untouched by all of them. There's even a direct argument that LLMs *can't* be conscious without continual learning ([2512.12802](https://arxiv.org/pdf/2512.12802)), tying L7 back to L4.

**Hardening.** **Demote L7 from a capability layer to optional, clearly-labeled experimental scaffolding.** A functional global-workspace *router* (select + broadcast across modules) may still help **integration** — that's an engineering claim worth testing — but the *consciousness* framing is contested, partly falsified, and **not a capability driver**. Do not build the system's value or narrative on it. If kept, keep it as "a learned cross-module attention/routing bottleneck," drop the consciousness vocabulary from the engineering spec, and treat AST/CTM as *inspiration*, not foundation.

---

## What changes in the spec (the net)

1. **Spine reframed (L3):** verifier = *amplifier to the base model's ceiling on verifiable tasks*, **not** an engine that creates new reasoning or surpasses frontier. Confirmed by both the literature and our own G3 run.
2. **The base model is the capability lever.** RL/verification is efficiency and reliability, not new intelligence. Budget accordingly.
3. **Bitter-lesson gate** on every layer: prefer learned/scalable/pretrained components; hand-structure must win measurably or be cut; scaffold, don't engineer cognition.
4. **Load-bearing layers are L1 (perception), L3 (verified reasoning), L4 (continual-learning discipline)** — these survive the pressure test and are buildable now.
5. **L5 (social), L6 (motivation), L7 (consciousness) are research extensions, not foundations.** L7 in particular is demoted and stripped of capability claims.
6. **The honest revised thesis:** *VERGE is a perception-grounded, continually-learning, verifier-amplified reasoning system built on a strong pretrained base by composing mature open components — with social cognition, autonomous motivation, and reflective integration as measured research extensions, not load-bearing foundations. The contribution is the disciplined composition and the verifier-as-amplifier seam, honestly bounded.*

## What survives the pressure test (the solid core to build on)

- **Verifier-as-amplifier on the verifiable slice** — bounded, but real and reproducible (G3 + RLVR literature).
- **Perception grounding via V-JEPA** — strong for short-horizon, honestly scoped.
- **Continual-learning discipline** — a confirmed, central, measurable problem with practical tooling (gradient-interference prediction, replay, UPGD).
- **The evaluation rigor itself** — G3's control-arm + pre-registered go/no-go method is, ironically, the most defensible original contribution of the whole project. Keep it as the measurement spine for every layer.
- **Composition as the contribution** — honestly scoped to a 2–3 layer integrated demo (M1→M3 in the build map), not AGI.

## The one-line verdict

The pressure test doesn't kill VERGE — it *right-sizes* it. The grandiose version ("verified-reasoning flywheel surpasses frontier; seven engineered layers compose into AGI") does not survive contact with the literature. The disciplined version ("a perception-grounded, continually-learning, verifier-amplified reasoner on a strong base, composed from mature parts, measured honestly") survives cleanly and is buildable by one person. Build that one.

---

*Sources: [Does RL incentivize reasoning beyond base?](https://openreview.net/forum?id=4OsgYD7em5), [RLVR implicitly incentivizes correct reasoning](https://huggingface.co/papers/2506.14245), [State of RL for LLM reasoning](https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training), [Sora/V-JEPA incomplete world model](https://www.arxiv.org/pdf/2407.10311), [Mechanistic catastrophic forgetting](https://arxiv.org/abs/2601.18699), [LLM continual-learning survey](https://github.com/Wang-ML-Lab/llm-continual-learning-survey), [RLeXplore intrinsic motivation](https://arxiv.org/pdf/2405.19548), [Reward hacking (Lil'Log)](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/), [Disproof of LLM consciousness / continual learning](https://arxiv.org/pdf/2512.12802), [ExploreToM](https://ai.meta.com/research/publications/explore-theory-of-mind-program-guided-adversarial-data-generation-for-theory-of-mind-reasoning/).*
