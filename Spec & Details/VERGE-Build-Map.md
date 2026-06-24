# VERGE Build Map — State of the Art Per Layer (use this, don't build it)

**Date: June 2026. Companion to `VERGE-Rev9-Layered-AGI.md`. This supersedes the "build from scratch" reading of every layer.**

The strategy is the one Rahul named: don't reinvent wheels. For each of the seven layers we (a) name our spec's original approach, (b) find the open-source repo / model / paper that *already does it better*, (c) check whether a *stronger theory* exists for the theoretical layers, and (d) give a concrete recommendation. The honest meta-point holds throughout: **the parts already exist — the unsolved frontier is composing them. So the realistic deliverable is an integrated demo of 2–3 layers, not all seven at once.**

Tags: **[use]** a maintained artifact you can pull today · **[frontier]** active research, no clean drop-in · **[theory]** a conceptual choice, with a better option flagged.

---

## L1 — Perception  **[use]**

- **Our spec:** a JEPA-style self-supervised encoder into the shared latent.
- **State of the art:** **V-JEPA 2** (Meta, open weights — [arXiv:2506.09985](https://arxiv.org/abs/2506.09985), [ai.meta.com/vjepa](https://ai.meta.com/vjepa/)). Action-free joint-embedding predictive model trained on >1M hours of video; SOTA motion understanding and action anticipation; post-trainable as an action-conditioned model with <62h of robot video. This *is* our L1, already built and downloadable. For static images, DINOv2/v3 (Meta) are the open backbones.
- **Better/newer direction:** **Causal-JEPA** ([arXiv:2602.11389](https://arxiv.org/pdf/2602.11389), Feb 2026) — learns world models through *object-level latent interventions*, which fuses L1 perception with L2 causal reasoning. Worth tracking; it points where L1→L2 is heading.
- **Recommendation:** Pull V-JEPA 2 weights. Do not train a perception encoder. Spend effort only on the projection/glue into your shared latent.

## L2 — Intuitive physics / World model  **[use + frontier]**

- **Our spec:** a Structural Causal Model skeleton + a learned neural transition model (intervention/counterfactual).
- **State of the art (predictive world models):** **Genie 3** (DeepMind, Aug 2025 — real-time interactive worlds, 24fps, but *closed*); **DreamerV3** (Danijar Hafner, **open** — `danijar/dreamerv3`) is the strongest open model-based RL world model and the practical choice. Open video-gen world models: CogVideo, HunyuanVideo, WanVideo. Tracking list: `ziqihuangg/Awesome-From-Video-Generation-to-World-Model`.
- **State of the art (the causal half):** **Neural Causal Models (NCMs)** and **Deep Structural Causal Models** (NeurIPS 2020) neurally parameterize an SCM for observational/interventional/counterfactual queries — exactly our §18 design, already formalized. **Key confirmed limitation:** NCMs need the causal *diagram* supplied; without structural knowledge the queries are underdetermined. That validates our "SCM skeleton from verified edges, neural functions on top" choice — and tells you the skeleton is the hard part, not the neural fit.
- **Benchmarks to measure against:** **IntPhys 2** (Meta, intuitive-physics video benchmark), PhysicsMind, WorldScore, VBench++.
- **Recommendation:** Use DreamerV3 (open) for the predictive world model; use a Deep-SCM / NCM library for the causal layer. The frontier piece you'd actually contribute is wiring the verified-edge skeleton to the neural transitions.

## L3 — System-2 reasoning  **[use]**

- **Our spec / what we hand-rolled:** a STaR/RFT verifier-filtered self-distillation loop (the G3 experiment).
- **State of the art:** **GRPO** (DeepSeekMath → DeepSeek-R1) and its 2025 refinement **DAPO** (Yu et al. — decoupled clipping, dynamic sampling) under the **RLVR** paradigm. GRPO samples a group of completions, scores each with a *verifiable reward* (your exact-match / unit-test / proof check), and advantage-weights them — a strictly better version of what we built. Overview: [Post-Training in 2026](https://llm-stats.com/blog/research/post-training-techniques-2026).
- **Use these repos:** **[verl](https://github.com/verl-project/verl)** (SOTA RL post-training, EuroSys 2025), **[OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)** (~3× faster than TRL on GSM8K+GRPO; has small-model R1 reproductions), **TRL** `GRPOTrainer` (easiest entry), **Open-R1 / EasyR1** (full R1 pipelines), index: **[awesome-RLVR](https://github.com/opendilab/awesome-RLVR)**.
- **Recommendation:** **Retire our G3 harness.** Point verl or TRL-GRPO at GSM8K with your exact-match reward function. Layer 3 is done with battle-tested code — no bug-hunting. (Keep G3's *control-arm discipline* as your evaluation rig; that part was good.)

## L4 — Plasticity / Continual learning  **[use — clear upgrade]**

- **Our spec:** continual backpropagation (Dohare et al., *Nature* 2024 — `shibhansh/loss-of-plasticity`).
- **Supersedes it:** **UPGD — Utility-based Perturbed Gradient Descent** ([arXiv:2404.00781](https://arxiv.org/abs/2404.00781)). Continual backprop only fixes *loss of plasticity*; UPGD addresses **both plasticity *and* catastrophic forgetting at once** by perturbing low-utility units more and protecting high-utility ones. This is a direct, concrete upgrade to our L4 mechanism.
- **Recommendation:** Adopt UPGD as the L4 learning rule; keep continual-backprop as the baseline to beat. This is the cleanest "skip the grind" win in the stack.

## L5 — Social intelligence / Theory of Mind  **[frontier]**

- **Our spec:** an inverse-planning model over mental-state variables; alignment lives here.
- **State of the art:** **ExploreToM** (Meta — [Explore Theory-of-Mind](https://ai.meta.com/research/publications/explore-theory-of-mind-program-guided-adversarial-data-generation-for-theory-of-mind-reasoning/)): program-guided adversarial generation of ToM problems. The headline result is sobering and useful — frontier models (Llama-3.1-70B, GPT-4o) score **as low as 5%** on ExploreToM-generated data. So ToM is *far* from solved, and there is **no clean drop-in**. Community: ToM4AI workshop (AAAI 2026).
- **Recommendation:** This is a genuine **frontier** layer — use ExploreToM as a *training/eval data engine*, not as a solved component. Be honest in the spec: L5 is research, not assembly. Don't gate the whole system on it.

## L6 — Motivation / Intrinsic goals  **[use + frontier]**

- **Our spec:** a free-energy frontier as the acquisition function; a four-level goal stack.
- **Supersedes/sharpens it:** **MAGELLAN** ([arXiv:2502.07709](https://arxiv.org/pdf/2502.07709)) — autotelic LLM agents that use *metacognitive predictions of learning progress* to pick goals in large goal spaces. "Learning progress" is a more practical, better-grounded intrinsic signal for LLM-scale agents than raw free-energy, and it's built for exactly our setting. Also **HERAKLES** (hierarchical skill compilation for open-ended LLM agents, [arXiv:2508.14751](https://arxiv.org/pdf/2508.14751)) and **Voyager** (`MineDojo/Voyager` — open-ended embodied LLM agent). Finding worth noting: **entropy and empowerment** are the intrinsic objectives that best correlate with human exploration.
- **Recommendation:** Replace the free-energy framing with **learning-progress-based goal selection (MAGELLAN-style)** as the primary intrinsic drive; keep empowerment as a secondary signal. Voyager is the reference architecture for an autotelic agent loop.

## L7 — Consciousness / Reflection  **[theory — a better one exists]**

- **Our spec:** Global Workspace Theory (select + broadcast) + a metacognitive/causal self-model. Functional only; no phenomenal claim.
- **Is there a better theory?** For the *engineering* goal, two candidates are stronger than plain GWT:
  - **Attention Schema Theory (AST)** (Graziano — [Frontiers in Robotics & AI](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2017.00060/full)), explicitly framed as "a foundation for engineering artificial consciousness." The sharp line from the literature: *"GWT is essentially attention schema theory without the schema part"* — i.e. AST = a global workspace **plus a self-model of its own attention**, which is exactly the metacognition you already wanted in L7. AST subsumes our design more cleanly than GWT alone.
  - **Conscious Turing Machine (CTM)** (Blum & Blum — [arXiv:2107.13704](https://arxiv.org/pdf/2107.13704)): a formal, theoretical-computer-science model of GWT that is concrete enough to *implement*. If you want L7 to be buildable rather than hand-wavy, CTM is the spec.
- **Open implementation:** VanRullen & Kanai's "Global Latent Workspace" (deep-learning GWT) has open code; **recurrence** is repeatedly flagged as a fundamental architectural requirement.
- **Honest scope:** IIT is "phenomenology-first" and not buildable here. Recent results say GWT/IIT/Predictive-Processing are *complementary*, and none is settled. So L7 stays the most speculative layer — but **adopt AST as the framing and CTM as the buildable formalization**; that's a real upgrade over generic GWT.

---

## The composition map (what to actually build, in order)

You don't build seven layers. You build a **coherent 2–3 layer integration** that demonstrates the spine, then extend. Recommended milestone order — each is shippable on its own and uses the components above:

1. **M1 — Verified reasoner (L3 alone).** verl/TRL-GRPO + a verifiable reward on math/code. Reproduces RLVR cleanly. *This is a portfolio-grade artifact by itself and the cheapest to finish.*
2. **M2 — Perception → Memory → Reasoner (L1 + retrieval + L3).** V-JEPA 2 features → a vector memory → the M1 reasoner conditioned on retrieved context. This is the first thing that looks like *your* architecture and is genuinely novel as an integration.
3. **M3 — Add a world model (L2).** DreamerV3 or V-JEPA2-action-conditioned for predictive rollouts the reasoner can query. Now you have perception + world model + verified reasoning — a defensible research demo.
4. **M4+ — Plasticity (UPGD), motivation (MAGELLAN), social (ExploreToM), reflection (AST/CTM)** layered in one at a time, each measured against an ablation.

**The contribution is the seam, not the parts.** Every component above is someone else's wheel. What no one has shipped is a clean, measured *composition* with the verifier as the spine connecting them. That seam is buildable by one person over months — and a working M2/M3 demo is the realistic version of "get noticed," far more than a spec or a 1.5B experiment.

## Honest scope (unchanged)

Using SOTA parts skips *years* of grinding on the components — that's real and worth doing. It does **not** make AGI a solo near-term outcome, because the composition is the open problem and three of the seven layers (L5 social, L6 autonomy, L7 consciousness) are frontiers, not assembly. The right ambition: build M1→M3, measure honestly with G3's control discipline, and let a real integrated system — not a claim — speak.

---

*Sources are linked inline. Frameworks: [verl](https://github.com/verl-project/verl), [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), [awesome-RLVR](https://github.com/opendilab/awesome-RLVR). Models: V-JEPA 2, DreamerV3, DINOv2/3. Methods: GRPO/DAPO, UPGD, MAGELLAN, ExploreToM. Theory: AST, Conscious Turing Machine.*
