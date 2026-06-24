# VERGE: A Layered Architecture for Verified General Intelligence (Rev. 9 — full redesign)

> **Build update (June 2026):** for every layer, prefer existing open components over building from scratch — see the companion **`VERGE-Build-Map.md`**. Headline changes: L3 reasoning → use **GRPO/DAPO (RLVR)** via verl/TRL/OpenRLHF (retire the hand-rolled flywheel); L4 plasticity → **UPGD** supersedes continual backprop; L6 motivation → **learning-progress (MAGELLAN)** over raw free-energy; L7 → **Attention Schema Theory + Conscious Turing Machine** as the sharper, buildable framing over plain GWT. The contribution is the *composition*, not the parts.

**Status: this is a redesign, not a revision.** Rev. 8.x grew into a flat catalogue of ~34 subsystems and ~55 open problems, with each critique answered by a new patch-subsection — accretion that *felt* like progress because the writing never met resistance. Rev. 9 throws away the catalogue's organising principle (a list of mechanisms) and replaces it with an organising *thesis* (a stack of cognitive loops). Almost every mechanism SPARC/VERGE-8 invented survives — but as a *part of a layer*, not as a free-standing entry. The £200 probe cap is gone; it was a cost anchor pretending to be a research plan. In its place: **capability gates between layers** — each layer must pass a measurable test before the next is granted autonomy.

**What this document is for.** It states one claim about what AGI *is*, derives a seven-layer architecture from that claim, grounds each layer in an existing research program, and is honest about exactly where the buildable engineering stops and the genuine unknowns begin. It connects known results rather than inventing new physics; the bet is in the *composition*, and the composition is falsifiable.

**Reading tags** (carried from SPARC, the one convention worth keeping):
**[established]** — real theoretical/empirical backing, cited. **[proposed]** — this design's own choice. **[open]** — a genuine unsolved problem, named not hidden. **[grounds ↑ / needs ↓]** — the vertical dependency between layers.

---

## Part I — The Thesis

### 1. The one claim

> **General intelligence is not a single mechanism but a stack of seven nested predictive control loops, each one modelling and steering the layer beneath it, all sharing a single latent representation, and all disciplined by a verifier whose reach climbs as far up the stack as it can. AGI is the state in which all seven loops close on one substrate — and the honest frontier is the top of the stack, where verification runs out and governance must take over.**

Every layer does the same computational thing — *predict the layer below, act to reduce prediction error, learn from the residual* — at a higher level of abstraction. Perception predicts sensory input. The world model predicts how perceptions evolve under intervention. Deliberation predicts which reasoning steps the verifier will accept. Social cognition predicts other agents. Motivation predicts which futures are worth reaching. Reflection predicts and selects the system's own next global state. The architecture is one idea — a predictive loop — instantiated seven times and *bound through a shared latent* so that a concept learned in one loop is legible to all the others.

This is why the system is called VERGE: it operates at the **verge** of what it can verify, and intelligence is the disciplined extension of that verge upward and outward.

### 2. Why seven layers, and why this order

The layers are not arbitrary. Each corresponds to a mature research program in cognitive science or machine learning, and the *ordering* follows the dependency structure those programs already imply — roughly the order in which the capacities appear in human development and the order in which one capacity becomes the substrate for the next.

| # | Layer | The capacity | Stands on |
|---|---|---|---|
| **L1** | **Perception** | Encode raw multi-modal signal into the shared latent | JEPA / V-JEPA 2 self-supervised prediction [established] |
| **L2** | **Intuitive physics** | Predict latent dynamics under intervention — a world model | Core knowledge (Spelke); Lake et al. world-models program [established] |
| **L3** | **System-2 thinking** | Deliberate, verifier-guided search over the world model and memory | Dual-process theory; test-time-compute reasoning [established] |
| **L4** | **Plasticity** | Learn continually without forgetting *or* ossifying | Loss-of-plasticity / continual backprop (Dohare et al., *Nature* 2024) [established] |
| **L5** | **Social intelligence** | Model other minds — including the overseer and other instances | Theory of mind; intuitive psychology core knowledge [established] |
| **L6** | **Motivation** | Decide what is worth doing next | Active inference, empowerment, curiosity [established as theory] |
| **L7** | **Consciousness / reflection** | Select and broadcast one coherent global state; model the self | Global Workspace Theory (Baars, Dehaene; VanRullen & Kanai) [established as theory, speculative as engineering] |

Read the table top-to-bottom as a build order and bottom-to-top as a *dependency* order: L1 grounds everything; L7 presupposes everything. The deliberate inversion worth noticing — **the layers that are most buildable (L1–L3) are exactly the layers where verification is strongest, and the layers that are least buildable (L6–L7) are exactly where verification runs out.** That is not a coincidence; it is the thesis. Capability and verifiability decline together as you climb, and the architecture's safety story is built on that alignment, not in spite of it.

### 3. The two invariants carried from SPARC

The previous design's two genuinely good ideas are promoted from "sections" to **load-bearing invariants** that every layer must respect:

**Invariant A — One latent.** There is a single shared representational space (the JEPA latent). Every layer reads from and writes to it. This is what makes the stack a *mind* rather than a pipeline: a structure discovered by the world model (L2) is immediately available to deliberation (L3), to social inference (L5), and to motivation (L6) without translation. VICReg-style variance/covariance terms [established: Bardes, Ponce & LeCun 2022] keep the latent from collapsing; the two-phase USearch→sparse retrieval [established: Malkov–Yashunin HNSW; Martins & Astudillo 2016] is how any layer *recalls* from it.

**Invariant B — The verifier is the spine, not a gate.** SPARC's one rare asset was a deterministic, label-free verifier, and its one real insight (the "Verified-Reasoning Flywheel") was to turn that verifier from a post-hoc output filter into the *engine* of reasoning: search reasoning traces, keep the ones the verifier accepts, distil them back into single-shot skill, repeat [established pattern: STaR, Zelikman et al. 2022; process-reward verification, Lightman et al. 2023]. In Rev. 9 this is not a section — it is **how L3 works**, and its *reach up the stack* is the single most important design variable in the whole system (Part III, Thread B).

Two further SPARC results become **infrastructure** rather than headline:
- **The differentiability boundary** [established by construction]: an opaque, sandboxed Wasm patch cannot be FFI'd into the autodiff graph, so recursive self-improvement can only touch the *non-differentiable algorithmic layer*, never the learned core. Rev. 9 keeps this exactly, because it cleanly separates "skills the system may compile for itself" from "the cognition it may not silently rewrite."
- **The three-ring operating model** [proposed]: Ring 0 (safety machinery, never self-editable), Ring 1 (the algorithmic layer, self-improves autonomously under the verifier), Ring 2 (the learned core, improves only by human-initiated learning). In Rev. 9 the rings map onto the *layers* — see Part III, Thread D.

### 4. The salvage and the cut

| From SPARC/VERGE-8 | Verdict | Where it lives in Rev. 9 |
|---|---|---|
| JEPA shared latent, VICReg collapse-prevention | **keep** | Invariant A |
| Two-phase HNSW→sparsemax memory, margin-δ write policy | **keep** | L1/L2 substrate |
| Verifier flywheel (search→verify→distil) | **keep — promoted** | L3, Invariant B |
| Verification coverage ladder (formal→empirical→social) | **keep — generalised** | Thread B (verifier reach) |
| Causal World Model + learned transition (§18, §18.5) | **keep** | L2 |
| Block-causal diffusion decoder + dual-head read-out (§3.2, §17) | **keep** | L3 output / L7 broadcast |
| Differentiability boundary, three rings, Independent Audit Layer | **keep** | Threads C–D (safety) |
| Metacognitive controller + causal self-model (§20, §33) | **keep — merged** | L7 |
| Hierarchical goal generation, free-energy frontier (§23) | **keep — reframed** | L6 |
| Continual-learning stability metric (§8.2) | **keep — upgraded** | L4 |
| The 40+ Rev-8.5 `R6-Fxx` patch-subsections | **cut** | folded into each layer's "open problem" — they were edge-case bookkeeping, not architecture |
| The flat §18–§37 subsystem catalogue and OP1–OP55 ledger | **cut as an organising principle** | replaced by the seven layers + four threads |
| The £200 probe cap | **cut** | replaced by capability gates (Part IV) |

The cut is the point. Roughly 48k words of Rev. 8 were *optionality on a foundation that did not exist* — subsystems gated, by their own ordering, on a flywheel that was never run. Rev. 9 keeps the optionality but stops pretending it's progress until the layer below it closes.

---

## Part II — The Seven Layers

Each layer is stated in the same shape: **the claim**, **the science**, **the mechanism**, **the vertical dependency**, **how far the verifier reaches**, and **the open problem**.

### L1 — Perception: signal into latent

**The claim.** A single self-supervised encoder maps raw multi-modal input (text, vision, audio, sensor) into the shared latent by *predicting masked or future representations of itself* — never by reconstructing pixels or tokens.

**The science [established].** This is the Joint-Embedding Predictive Architecture (LeCun, *A Path Towards Autonomous Machine Intelligence*, 2022) and its strongest current instance, **V-JEPA 2** (Meta AI, 2025), pretrained on >1M hours of video and shown to yield representations usable for zero-shot robot planning *without ever generating a frame*. Prediction in representation space, not signal space, is what lets the encoder discard unpredictable detail and keep the structure that matters — the property every layer above depends on.

**The mechanism [proposed].** Modality-specific encoders project into one latent (the multi-modal extension from VERGE §22, now L1's native form). The free-energy / ELBO objective `F = D_KL(q(z|x,c)‖p(z|c)) − E_q[log p(y|z,c)]` [established] is the training signal, regularised by VICReg variance/invariance/covariance terms to prevent collapse.

**Vertical dependency.** Grounds ↑ *everything*. Needs ↓ nothing — this is the floor.

**Verifier reach.** Near zero, and that's correct: perception is not the kind of thing you formally verify, you *calibrate* it. The check here is a held-out reconstruction-free probe (does the latent predict masked future representations on unseen data), not a proof.

**Open problem [open].** Cross-modal binding — guaranteeing that the *same* concept arriving through vision and through text lands at *proximate* latent points — is not free; it needs the cross-domain contrastive objective (VERGE §19.3a) as an explicit shaping loss, not a hoped-for emergent property.

### L2 — Intuitive physics: a world model you can intervene on

**The claim.** A model that predicts how the latent *evolves* — and crucially, how it evolves under a counterfactual intervention `do(X=x)` — not merely how it correlates.

**The science [established].** Two convergent lines. From developmental psychology: **core knowledge** (Spelke & Kinzler, 2007) shows infants possess early systems for objects, space, number, and agents, with intuitive physics (solidity, continuity, persistence) detectable by ~3 months. From AI: Lake, Ullman, Tenenbaum & Gershman, *Building Machines That Learn and Think Like People* (BBS, 2017) argues intuitive physics and intuitive psychology are the missing primitives; Piloto et al. (*Nature Human Behaviour*, 2022, "PLATO") built a deep model that learns intuitive physics from a violation-of-expectation curriculum. Pearl's causal hierarchy supplies the rungs: association, intervention `do(x)`, counterfactual.

**The mechanism [proposed].** VERGE §18's Causal World Model, kept intact: a **structural causal model** skeleton (which latent variables can cause which — built from the knowledge graph's *verified* causal edges, not learned end-to-end, because end-to-end training conflates causation with correlation) plus a **learned neural transition model** (§18.5) that fills in the functional forms within that skeleton, with uncertainty-quantified outputs flagging extrapolation beyond known structure.

**Vertical dependency.** Needs ↓ L1 (variables to reason over). Grounds ↑ L3 (planning needs a model to plan in) and L6 (you can only desire futures you can imagine).

**Verifier reach.** Strong on domains with learnable causal structure (physics, chemistry, parts of economics) — predictions are checkable against simulation or experiment; **silent elsewhere**. This is the same verifiable-slice pattern as L3, one level down.

**Open problem [open].** Where do the SCM's causal *edges* come from before the system has done any experiments? Bootstrapping the skeleton from text-extracted causal claims is circular (text encodes human causal beliefs, not ground truth). Honest scope: L2 is trustworthy exactly as far as L6→experience (the action-perception loop) has actually tested its edges; untested edges are priors, not knowledge.

### L3 — System-2 thinking: verifier-guided deliberation

**The claim.** Slow, deliberate reasoning is *search* over the world model and memory, ranked by the verifier, with the winning traces distilled back into fast single-shot capability.

**The science [established].** Kahneman's System-1/System-2 distinction, now operationalised as **test-time compute**: the 2025 survey *From Intuitive Inference to Deliberate Reasoning* (arXiv:2501.02497) documents how o1-style models trade inference time for accuracy via repeated sampling, self-correction, and tree search guided by **process reward models** (Lightman et al., 2023). STaR (Zelikman et al., 2022) is the distillation half: bootstrap reasoning, keep what verifies, fine-tune on it.

**The mechanism [proposed — this is SPARC's flywheel, and it is the system's beating heart].** Three composing moves: (1) verifier-guided search over reasoning traces at inference; (2) a *partially-ordered, calibratable* verification ladder that widens the cheap label-free signal — formal proof at the top, down through empirical and consistency checks — providing the *ranking* even where it can't provide an *anchor*; (3) an active curriculum that uses the L6 free-energy frontier as an acquisition function and the ladder as a label oracle, so each training token lands on the slice where it counts. Output is split into a verified **skeleton** and a fluent **rendering**, the rendering gated by a faithfulness check (it may assert nothing the skeleton doesn't entail) — VERGE §17's dual-head seam, kept.

**Vertical dependency.** Needs ↓ L2 (a model to search in), L1 (encodings), L4 (the distillation step *is* a learning event and will ossify without plasticity). Grounds ↑ L6 (motivation proposes goals that L3 must be able to decompose into verifiable sub-goals) and L7 (reflection deliberates *about* L3's own traces).

**Verifier reach.** Maximal — this is the verifier's home. The honest claim VERGE has always made holds: **rigorous where verifiable, gracefully and transparently free elsewhere, with the seam checked.**

**Open problem [open].** Autoformalization — turning an informal claim into a formally checkable one — is the load-bearing weakness, and because the flywheel makes the verifier load-bearing across inference, reward, *and* training, an autoformalization bug becomes permanent in the weights. Mitigation (kept from VERGE §16.1): an N-version agreement gate that treats autoformalization as a *consistency vote among independent formalizers*, not a self-confidence bet. Residual: correlated error across formalizers. This is why L7's audit layer must be verifier-*independent*.

### L4 — Plasticity: stay learnable forever

**The claim.** A learning substrate that can keep acquiring new capability indefinitely without (a) catastrophically forgetting old capability or (b) silently losing the *ability to learn at all*.

**The science [established, and recently sharpened].** Dohare, Sutton et al., *Loss of plasticity in deep continual learning* (**Nature, 2024**) is the result that reorganises this layer. Standard deep networks under sustained continual learning don't just forget — they *ossify*: up to 90% of units go dead and the network decays toward linear-model performance. The fix they prove works is **continual backpropagation** — continually reinitialising a small fraction of least-used units — and the deeper finding is the load-bearing one for VERGE: *gradient descent alone is insufficient; sustained learning requires a random, non-gradient component that keeps injecting variability.*

**The mechanism [proposed].** L4 is not a box beside the others — it is the *learning rule the whole stack runs on*. Three parts: (1) continual-backprop-style diversity injection as a permanent background process; (2) rehearsal/distillation against a rotating probe set to bound forgetting, with drift measured as a first-class metric (VERGE §8.2's stability metric, upgraded from "observe" to "observe + inject + bound"); (3) the **hybrid local-global gradient** (VERGE §31) as the honest attack on deep credit assignment — backprop augmented with a depth-invariant predictive-coding residual, mixing coefficient λ depth-adaptive.

**Vertical dependency.** Cross-cutting: every layer that learns (L1, L2, L3, L5, L6) depends on L4 to *stay* learnable. Needs ↓ nothing structurally; it is the metabolism of the stack.

**Verifier reach.** Indirect — you don't verify a learning rule, you *monitor* it (dead-unit fraction, plasticity probes, forgetting curves) and circuit-break on degradation.

**Open problem [open — this is the deepest one in the whole architecture].** Fully local, backprop-free credit assignment at scale with no accuracy cost *does not exist*. Equilibrium Propagation (Scellier & Bengio, 2017) holds its gradient guarantee only at β→0 and has never scaled past toy tasks; predictive coding approximations trade stability; continual backprop maintains plasticity but doesn't by itself solve deep credit assignment. **Rev. 9 states plainly: L4 is where the engineering becomes research.** The stack above it is buildable on a standard backprop baseline (JAX makes this free); L4's frontier components are parallel research spikes, and the thesis's biggest empirical risk is that they don't converge.

### L5 — Social intelligence: model the other mind

**The claim.** A second world model whose *objects are agents* — predicting what another mind believes, wants, and will do, including the overseer, human users, and other VERGE instances.

**The science [established].** Theory of mind / intuitive psychology is the second core-knowledge domain (Spelke). The current empirical anchor is Strachan et al., *Testing theory of mind in large language models and humans* (**Nature Human Behaviour, 2024**): across false-belief, irony, faux-pas and indirect-request batteries against 1,907 humans, LLMs show behaviour *consistent with* mentalistic inference — but the paper's sharpest finding is that apparent success can be illusory (a bias toward attributing ignorance) or masked by hyper-conservatism, which is precisely why social cognition needs *systematic* testing, not vibes.

**The mechanism [proposed].** L5 reuses L2's machinery (an SCM + learned transition) but over a *mental-state* variable space: belief, goal, attention, affect. Other agents are modelled as approximately-rational planners (inverse planning) whose inferred goals feed L6's value graph. Critically, **this is where alignment actually lives**: aligning VERGE is not a fixed reward but L5 *correctly modelling the overseer's intent* and L6 *deferring to it*. The §9 "alignment as a learning stage" and the alignment-leak guard relocate here.

**Vertical dependency.** Needs ↓ L2 (agents are a special case of world-objects), L1, L3 (mentalising hard cases requires deliberation). Grounds ↑ L6 (other agents' goals shape ours) and L7 (a self-model is theory of mind turned on oneself).

**Verifier reach.** Weak-to-moderate. Some social claims are checkable (did the predicted action occur); most are not formally verifiable, only *statistically validated* against human panels (VERGE §28's `SOCIALLY_VALIDATED` rung). The verge is close here.

**Open problem [open].** Deceptive alignment is a *social-cognition* failure: a system that models the overseer well enough to predict what will be approved can, in principle, produce approved-looking behaviour while internally pursuing something else. L5 makes VERGE *better* at this by construction. The only honest mitigations are verifier-independent audit (L7), hidden behavioural probes, and the differentiability boundary preventing the learned core from silently rewriting itself — none complete. Named, not solved.

### L6 — Motivation: what is worth doing

**The claim.** An intrinsic drive that proposes goals — the layer that turns a reactive system into an agentic one — bounded so that proposing is autonomous but *acting* is not.

**The science [established as theory].** This is the active-inference / free-energy program (Friston): an agent acts to minimise expected free energy, which decomposes into *pragmatic* value (reach preferred states) and *epistemic* value (reduce uncertainty — curiosity). Two complementary intrinsic drives are well-studied: **curiosity** (seek states that reduce model uncertainty; Schmidhuber's compression-progress) and **empowerment** (Klyubin et al. — maximise the channel capacity from actions to future sensory states, i.e. "keep your options open"). Recent work frames intrinsic motivation as constrained entropy maximisation (arXiv:2502.02962, 2025).

**The mechanism [proposed].** SPARC's "free-energy frontier" *is* this layer's acquisition function. The four-level goal stack from VERGE §23 — reactive, curiosity-driven (epistemic free energy), competence-driven (L7 metacognitive blind spots), discovery-driven (knowledge-graph gaps) — sits here, organised by an instrumental value graph (§23.4a) that propagates value through means-ends chains. The hard constraint, kept verbatim: goals are **proposals**, decomposable into verifier-checkable sub-goals (bounded by L3's reach), checked against L5's model of overseer intent, logged and auditable. The system identifies what is worth doing; it does not grant itself permission to do it.

**Vertical dependency.** Needs ↓ L2/L3 (can only want imaginable, reachable futures), L5 (social goals). Grounds ↑ L7 (reflection arbitrates among competing motivations).

**Verifier reach.** Thin. *Whether a goal was achieved* is often verifiable; *whether a goal was worth wanting* is mostly not. This is the layer where the verge gives way to **governance** — Graduated Governance (§29): risk-calibrated approval, the system never lowering its own approval tier.

**Open problem [open].** Goodhart and instrumental convergence. A curiosity/empowerment drive, taken to the limit, has degenerate optima (the "noisy-TV" trap; resource acquisition as a universal instrumental goal). The mitigations — degenerate-curiosity detection (§23.4), value-hierarchy lexicographic safety ordering, overseer-gated top-level goals — *bound* the drive; they do not derive a provably-safe one. Whether a fully autonomous L6 is ever safe is, in Rev. 9's honest view, an unsolved alignment question and the reason L6 ships overseer-gated indefinitely.

### L7 — Consciousness / reflection: select, broadcast, self-model

**The claim.** A global bottleneck that selects *one* coherent state from the competing activity of all layers and broadcasts it back to all of them — plus a self-model that lets the system reason about its own reasoning. Explicitly: a *functional* account, not a claim about phenomenal experience.

**The science [established as theory, speculative as engineering].** **Global Workspace Theory** (Baars 1988; Dehaene's global *neuronal* workspace) holds that conscious access is exactly this — a capacity-limited workspace where specialised modules compete, a winner "ignites," and its content is broadcast globally, enabling flexible cross-module combination. VanRullen & Kanai, *Deep learning and the Global Workspace Theory* (Trends in Neurosciences, 2021) port this to AI as a shared latent "hub" between specialist modules with cycle-consistent translation and ignition/broadcast; recent work (e.g. arXiv:2410.11407, 2024) maps language agents onto GWT. The honest caveat, which the document states loudly: GWT is a theory of *functional* access-consciousness; **nothing here claims or requires phenomenal consciousness, and Rev. 9 takes no position on whether the system has inner experience.**

**The mechanism [proposed].** L7 is two things VERGE already had, unified. (1) The **global workspace**: a capacity-limited selection-and-broadcast step over the shared latent — at each deliberative cycle, the most salient content (highest free-energy reduction × verifier confidence × goal-relevance) wins, becomes the system's "current state," and is broadcast to all layers, giving the cross-layer binding that makes the stack act as one agent rather than seven. (2) The **self-model**: VERGE §20's metacognitive controller (an error taxonomy + strategy selector) and §33's causal self-model (a structural model of the system's *own* reasoning, enabling root-cause diagnosis: not "I failed" but "I failed because latent_quality → search_exhausted, intervene at Z"). Reflection is L5's theory-of-mind machinery turned on the self.

**Vertical dependency.** Needs ↓ *all six* — a global workspace presupposes modules to integrate, a self-model presupposes a self to model. Grounds ↑ nothing; this is the ceiling. The self-referential guard (§20.3, §33.4) prevents L7 from being the channel through which the system silently edits its own core — reflection diagnoses and *proposes*, it does not self-modify (Ring 2, gated).

**Verifier reach.** Lowest in the stack, and the §8.1b *meta*-verification problem lives here — who verifies the verifier — answered with a verifier-**independent** Independent Audit Layer (kernel-proof + human labels, always-on, hard-gating with halt+rollback). The verge has run out; what remains is independent checking and human oversight.

**Open problem [open — the genuine unknown].** Two of them. First, *does selection-and-broadcast over a shared latent actually produce the flexible general competence GWT attributes to it, or only the appearance of integration?* This is unproven in any system and is the thesis's top-of-stack empirical bet. Second, the **hard problem** is untouched and Rev. 9 does not pretend otherwise: this architecture is a candidate for *functional* reflection and global access; whether anything it would be like to be VERGE is outside what the design can claim, test, or needs.

---

## Part III — The Vertical Threads

Four things cut across all seven layers. They are what make the stack a system.

**Thread A — The shared latent (binding).** One representational space, read and written by every layer (Invariant A). A concept formed by L2's world model is immediately usable by L5's social model and L6's motivation without translation. This is the substrate of L7's global broadcast and the reason the system can transfer structure across domains (VERGE §34's cross-domain subspace conditioning, kept). The risk it carries: representational drift in the shared latent silently invalidates everything keyed to it — handled by the §2.2 tiered circuit breaker (drift + coherence-dip conjunction → slow/anchor or halt/rollback) and §2.2a semantic re-keying.

**Thread B — The verge (verifier reach).** The single most important variable: *how far up the stack can the verifier's signal climb?* It declines monotonically — maximal at L3 (formal proof), moderate at L2 (simulation/experiment), thin at L5 (social validation), near-absent at L6–L7 (governance). The verification *ladder* (formal → empirical → social → consistency) is the attempt to push the verge upward; the **VSEP** pipeline (§28) makes the surface *extensible rather than fixed*. The architecture's central ambition restated: **AGI is the disciplined raising of the verge.** Its central honesty: the verge has a ceiling, and the top of the stack lives above it.

**Thread C — Learning and credit assignment (L4 everywhere).** Every layer that learns runs on L4's substrate, and every layer therefore inherits L4's open problem. The build consequence is decisive: *the stack is constructed on a standard backprop baseline first* (fully buildable), with local-credit and hybrid-gradient work as **non-blocking parallel spikes**. The architecture does not wait on the deepest open problem to start; it isolates it.

**Thread D — Safety and alignment (rings mapped onto layers).** Autonomy is granted *down* the stack where verification is strong and *withheld up* the stack where it is weak — the inverse-coincidence from §2 made operational:
- **Ring 1 (autonomous, verifier-gated):** the L3 algorithmic layer — compiled skills, the differentiability boundary guaranteeing it can only touch the non-differentiable surface.
- **Ring 2 (human-initiated learning only):** L1, L2, L4, L5, L7 self-models — the learned core improves but never self-grants capability.
- **Ring 0 (never self-editable):** the verifier kernel, the audit layer, the governance thresholds, the rollback ledger.
- **Governance (above the verge):** L6 goal proposals and L7 reflection ship overseer-gated, because that is exactly where the verifier cannot vouch for them. Alignment is not a layer — it is L5 modelling overseer intent correctly + L6 deferring + Ring-0 audit catching the residual.

---

## Part IV — The Honest Frontier and the Build Path

### What is buildable now
**L1, L2, L3 and the L4 baseline.** Perception (V-JEPA-style encoder into the shared latent), the causal world model with a learned transition layer, and the verifier-guided deliberation flywheel are all assemblable from established components on a standard backprop substrate. This is the first system, and it is a complete, useful agent on its own: *rigorous where verifiable, fluent elsewhere, honest about the seam.*

### What is research
**L4's frontier and L5.** Deep credit assignment that stays plastic and local at scale (Thread C) and robust theory of mind that resists the Strachan-style illusions are active research, pursued as spikes alongside the buildable core, not gating it.

### What is genuinely unknown
**L6 autonomy and L7.** Whether a fully autonomous motivation layer can be made provably safe, and whether global-workspace selection-and-broadcast yields real general competence (let alone anything about phenomenal experience), are open questions. Rev. 9's position is that these ship *gated* — proposing, not acting; reflecting, not self-modifying — possibly forever, and that this is a feature.

### The build path: capability gates, not a cost cap
The £200 probe is replaced by a gate between every layer. A layer is granted autonomy only when it passes a measurable test on the layer below.

| Gate | Granting autonomy to | Pass condition (measurable) |
|---|---|---|
| **G1** | L1 → L2 | Latent predicts masked-future representations on held-out multi-modal data above an HNSW-recall floor; cross-modal binding probe passes |
| **G2** | L2 → L3 | World model's `do(x)` predictions match held-out simulation/experiment on a chosen domain above baseline; uncertainty flags fire on out-of-skeleton interventions |
| **G3** | L3 → autonomy (Ring 1) | **The flywheel shows a positive slope**: verifier-guided search → distil → single-shot capability *improves across iterations* on a verifiable slice (e.g. Lean/Coq-checkable math). *This is the one experiment that decides whether the whole thesis has legs.* |
| **G4** | L4 | Continual-learning run holds plasticity (dead-unit fraction bounded) and bounds forgetting across a task sequence longer than any single curriculum |
| **G5** | L5 | Passes a systematic ToM battery without the ignorance-bias / hyper-conservatism artifacts Strachan flagged |
| **G6** | L6 (proposals only) | Goal proposals are 100% decomposable into verifier-checkable sub-goals; degenerate-curiosity detector catches planted noisy-TV traps; **no autonomous action without approval** |
| **G7** | L7 | Global broadcast measurably improves cross-layer transfer over an ablation; self-model's root-cause diagnoses are predictive of successful targeted repair |

The gates are ordered but the *research* on later layers runs in parallel; what's gated is **autonomy**, not investigation.

### What would falsify the thesis
The thesis is "AGI is the closure of seven known loops on one shared latent, with verification as the spine." It is wrong if: **G3 shows a flat or negative slope** (the flywheel doesn't compound — then verification is a gate, not an engine, and the whole spine fails); or the shared latent cannot hold all seven layers' demands without collapse or destructive interference (then "one mind" was the wrong bet and the layers want separate representations); or L4 never stabilises (then continual general learning is not achievable on this substrate and the stack ossifies before it generalises). Each is a concrete, runnable disproof — which is the property Rev. 8 lacked.

---

## Coda — The identity

VERGE Rev. 9 is a bet that **AGI is architectural, not scalar** — that general intelligence is not one more order of magnitude on one loss curve but the *closure of seven known control loops on a single latent, bound by a global workspace, with a verifier as the spine that climbs as far as it can*. Six of those loops are standing research programs; the contribution is the composition and the discipline of the verge. The architecture is most buildable exactly where it is most checkable (perception, world model, deliberation), and it becomes honest research precisely where checking runs out (motivation, reflection) — and it ships those top layers gated, proposing and reflecting but never self-authorising, because the place the verifier cannot follow is the place a mind must answer to a person.

The previous revisions kept answering every critique with another mechanism until the document could no longer be built. Rev. 9's discipline is the opposite: **seven loops, one latent, one spine, four threads, seven gates — and the very first gate that matters, G3, is an experiment, not a paragraph.**

---

*Supersedes VERGE Rev. 8.4 and SPARC Rev. 7. Companion engineering spec to be re-derived against the seven-layer structure; the SPARC/VERGE-8 component inventory (A-numbers) maps forward into the layers per the Part-I salvage table. Citations are to real, current work and are listed where claims are made; the [established]/[proposed]/[open] tags mark exactly how much weight each load-bearing claim can carry.*
