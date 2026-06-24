# VERGE — Engineering Specification

*How to implement the architecture in `verge-spec.md`. June 2026. Companion to the spec; read §2 of the spec first — the verifier-is-bounded result is the constraint every engineering choice below respects.*

This document picks a concrete stack, defines the interfaces that hold the layers together, and gives a per-layer build with the open components named, wired, and measured. The governing principle from the spec is operational here: **use mature parts, build the seam, gate autonomy by measurement.** Nothing below trains a foundation model from scratch.

---

## 0. Design rules (non-negotiable, derived from the spec)

1. **The verifier is the only trusted asset and it is never learned.** It is deterministic, label-free, pure-stdlib where possible, and unit-tested independently. Only verifier-confirmed traces ever enter training. Worst case is wasted compute, never poisoned weights.
2. **One shared latent, one contract.** Every layer reads/writes a single typed vector interface (§2). No layer parses another layer's internal representation.
3. **Standard backprop baseline first.** L1–L3 + L4-baseline are built on stock PyTorch. Local-credit / hybrid-gradient work (the L4 frontier) is a *non-blocking parallel spike*, never on the critical path.
4. **Every layer ships behind the eval harness (§7).** Pre-registered slope+CI, ≥3 seeds, frozen test, a free CPU mock for pipeline validation. A layer that cannot beat "scale base + retrieval" is cut (bitter-lesson gate).
5. **Rings are enforced in code, not convention** (§8). Ring 0 (verifier, audit, governance thresholds, rollback ledger) is import-frozen and never reachable from a self-improvement path.

---

## 1. The stack

| Concern | Choice | Why |
|---|---|---|
| Language | **Python 3.11+** | ecosystem; matches the existing M1 code |
| Core DL | **PyTorch 2.x** (`torch.compile`, FSDP) | primary substrate; M1 already on it; widest component support |
| Models/tokenizers | **HuggingFace** `transformers`, `datasets`, `accelerate`, `peft` | weights + LoRA + sharded training |
| Fast inference / search | **vLLM** (paged-attention, continuous batching) | K-sample search at temperature is the throughput bottleneck; vLLM is 5–20× HF `generate` |
| L3 RL post-training | **TRL `GRPOTrainer`** to start → **verl** at scale | TRL is the easiest correct entry; verl (EuroSys '25) is SOTA for large-group RLVR |
| Verifiers | stdlib (math) · **sandboxed pytest** (code) · **Lean 4 / `lean-dojo`** (proofs) | the verification ladder, formal → empirical |
| L1 perception | **V-JEPA 2** (open weights) · **DINOv2/3** for stills | downloadable; do not train an encoder |
| Vector memory | **FAISS** (HNSW index) + a sparse reranker | two-phase HNSW → sparse retrieval from the spec |
| L2 world model | **DreamerV3** (`danijar/dreamerv3`, JAX) · **DoWhy/pyro** + a thin NCM head for the causal SCM | strongest open model-based world model; Deep-SCM for `do(x)` |
| L4 plasticity | **UPGD** (own impl, ~200 LOC) + a gradient-interference monitor | supersedes continual-backprop; handles plasticity *and* forgetting |
| L6 motivation | **MAGELLAN-style** learning-progress predictor; **Voyager** loop as reference | learning-progress > raw free-energy at LLM scale |
| L5 social | **ExploreToM** as a data/eval engine | frontier; a generator of tests, not a solved module |
| L7 router | small learned attention bottleneck (PyTorch) | functional cross-module select+broadcast; integration claim only |
| Safety sandbox | **wasmtime** (Wasm) for compiled skills | the differentiability boundary: opaque, non-FFI-able into autograd |
| Orchestration | a `LayerService` interface + an in-process **workspace bus**; **Ray** when distributed | layers compose through the latent contract, not direct calls |
| Config | **dataclasses** (as in M1) → **Hydra** when sweeps grow | one place for every knob |
| Tracking / artifacts | **Weights & Biases** + JSON run logs; **git-lfs / DVC** for weights | reproducibility is the spine of the eval discipline |
| Packaging / infra | **uv** or `pip`, **Docker**, single **H100** baseline | M1 ran on one H100; keep that floor |

**The JAX boundary, handled honestly.** DreamerV3 is JAX; everything else is PyTorch. Do not rewrite Dreamer. Run it behind a process boundary that speaks the latent contract (§2) over Arrow/IPC, so L2 is a *service* the rest of the stack queries. If the boundary becomes a bottleneck, the fallback is a PyTorch model-based world model (e.g. a TD-MPC2-style learned transition); the contract makes that swap local.

---

## 2. The shared latent — the one interface that matters

Every layer depends on this and nothing else. Get it right before building any layer above L1.

```python
# verge/latent.py  — Ring 2 (learned), but the SCHEMA is Ring 0 (frozen)
from dataclasses import dataclass
import torch

LATENT_DIM = 2048            # fixed; changing it is a versioned migration

@dataclass(frozen=True)
class Latent:
    """One point in the shared representational space."""
    z: torch.Tensor          # (LATENT_DIM,) unit-norm
    modality: str            # "vision" | "text" | "audio" | "state" | "concept"
    source_layer: str        # "L1".."L7" — provenance for the audit ledger
    confidence: float        # encoder/verifier confidence in [0,1]
    key: bytes               # content hash for memory addressing + drift re-keying

class LayerService:
    """Every layer implements this. Layers never call each other directly —
    they read/write Latents through the workspace bus."""
    def encode(self, x) -> list[Latent]: ...      # signal/query -> latent
    def step(self, ctx: list[Latent]) -> list[Latent]: ...  # the layer's predictive loop
    def health(self) -> dict: ...                 # drift, dead-unit %, verifier reach
```

Three invariants enforced by a shared test suite, not by hope:

- **No collapse.** VICReg variance/covariance terms regularize every projection head into the latent. CI fails if the latent's effective rank drops below a floor.
- **Cross-modal binding.** The same concept via vision and via text must land within ε cosine distance. An explicit contrastive shaping loss (not emergent); a held-out binding probe gates L1→L2 (G1).
- **Drift circuit-breaker.** A background monitor tracks distribution drift in `z` and coherence of `key→z`. Drift + coherence-dip conjunction triggers slow/anchor; severe → halt + rollback to the last good latent checkpoint. Semantic re-keying re-anchors `key`s after an approved encoder update.

**Memory** is the two-phase recall over this space: FAISS HNSW for the cheap approximate neighborhood, then a sparse (sparsemax-style) reranker for the precise top-k. Writes use a margin-δ policy (only store a latent if it is ≥δ from existing neighbors — avoids flooding memory with near-duplicates).

---

## 3. L3 — the verified reasoner (BUILT; harden, then scale)

L3 is the spine and the only layer with a result. The existing `m1-verified-reasoner/` is a correct, calibrated expert-iteration implementation — keep its *discipline*, swap its *engine* for GRPO at scale.

**What exists and stays (from `m1-verified-reasoner/`):**
- `verifiers.py` — `ExactMatchVerifier` + robust answer extraction (`####`, `\boxed`, "answer is", last-number fallback). Pure stdlib. *This is the Ring-0 asset; do not touch it without an independent test review.*
- `metrics.py` — `pass_at_1`, distinct-path dedup, OLS `fit_slope` with 95% CI. Pure stdlib.
- `expert_iteration.py` — the loop: search → verify-filter → dedupe → reset-to-base → SFT → measure-on-frozen-test.
- `mock_model.py` — CPU simulator for free pipeline validation. **Run this before every GPU-hour.**
- The calibrated config: `lr=2e-6` (critical — the 1e-5 RFT default degrades 0.5B), `gen_temperature=1.0`, dedup on, reset-to-base on, frozen 500-problem test.

**The two upgrades (spec §5.4 — the path to a positive slope is *headroom*, not a cleverer objective):**

1. **Swap the engine to GRPO/DAPO.** Replace the hand-rolled SFT-on-survivors loop with TRL `GRPOTrainer` (then verl). Same verifier as the reward function; GRPO samples a group, scores each with the verifiable reward, advantage-weights. Use **DAPO's** decoupled clipping + dynamic sampling to fight the known GRPO failures: entropy/mode collapse, and length-bias (advantage ÷ length → long *wrong* answers under-penalized). *Never run vanilla GRPO from base.*

   ```python
   # verge/l3/reward.py
   from m1_verified_reasoner.verifiers import ExactMatchVerifier
   V = ExactMatchVerifier()
   def reward(prompt, completion, answer):     # the ONLY signal; deterministic
       return 1.0 if V.verify(make_problem(prompt, answer), completion) else 0.0
   ```

2. **Swap generation to vLLM** for the K-sample search — the throughput lever that lets you raise K (more search = the one operator that genuinely expands the confirmable set) and move to MATH (more rare-correct headroom than saturated GSM8K).

**Output contract.** L3 emits a verified **skeleton** (the checkable claim graph) and a fluent **rendering**, with a faithfulness check that rejects any rendering asserting something the skeleton does not entail. The skeleton is what gets distilled and what enters memory; the rendering is for humans.

**The verification ladder** (Thread B made concrete) — a partially-ordered set of verifiers, each a `LayerService`-style check returning `(passed, rung, confidence)`:

| Rung | Verifier | Domain |
|---|---|---|
| `FORMAL` | Lean 4 / `lean-dojo` proof check | math, logic |
| `EXACT` | `ExactMatchVerifier` (built) | arithmetic answers |
| `EMPIRICAL` | sandboxed pytest / simulation | code, physics rollouts |
| `CONSISTENCY` | N-version agreement among independent formalizers | autoformalization gate |
| `SOCIAL` | human-panel statistical validation | L5 claims |

Autoformalization is the load-bearing weakness: the `CONSISTENCY` rung treats "informal → formal" as a *vote among independent formalizers*, never a self-confidence bet. Residual correlated error is what the Ring-0 audit (verifier-*independent*) exists to catch.

**Bounded experiment (optional, spec §5.4).** Contrastive boundary guidance for a diffusion proposer (LLaDA/SEDD + the d1 harness): a margin-triplet distance-to-correct-boundary guide steers proposals only; the true verifier still filters every training trace. Predicted: **wins sample-efficiency + stability, ties accuracy.** An accuracy win beyond seed noise is a red flag for a verifier/contamination leak, not a triumph — scrutinize before believing.

---

## 4. L1 + memory — the M2 milestone

The first thing that looks like *VERGE* rather than a reasoner.

- **L1 encode.** Load V-JEPA 2 weights; freeze the backbone. Train only a small projection head (`encoder_out → LATENT_DIM`, unit-norm, VICReg-regularized) plus the cross-modal contrastive binding loss against the text encoder. DINOv2/3 path for static images. Output `Latent(modality="vision"|...)`.
- **Memory.** FAISS HNSW index over stored latents; sparse reranker for top-k; margin-δ write policy. Expose `retrieve(query_latent, k) -> list[Latent]`.
- **Wire to L3.** The reasoner's prompt is conditioned on retrieved latents (decoded to text context, or projected as soft prompts).
- **M2 gate:** retrieval *measurably* improves L3 pass@1 over a no-retrieval ablation, under the §7 harness. If it doesn't, the retrieval design is wrong — fix or cut, don't ship.

---

## 5. L2 — world model — the M3 milestone

- **Predictive half.** DreamerV3 behind the process boundary (§1), or a PyTorch TD-MPC2-style transition if the JAX seam bites. It predicts latent rollouts the reasoner can query: `rollout(state_latent, action_seq) -> list[Latent]`.
- **Causal half.** A Deep-SCM / NCM head: an SCM *skeleton* (adjacency over latent variables) built from **verified** causal edges only — never learned end-to-end (that conflates causation with correlation), and never bootstrapped from text-extracted claims (circular). Learn only the functional forms inside the supplied skeleton. Untested edges are flagged `prior`, not `knowledge`. Use DoWhy/pyro for the interventional/counterfactual machinery.
- **Uncertainty.** Every `do(x)` prediction carries a calibrated uncertainty; out-of-skeleton interventions *must* fire the flag.
- **M3 gate (G2):** `do(x)` predictions match held-out simulation/experiment above baseline on one chosen domain; out-of-skeleton uncertainty fires reliably.

---

## 6. L4–L7 — research layers (one at a time, each vs an ablation)

- **L4 plasticity (discipline, build alongside everything).** Implement UPGD as the optimizer wrapper (perturb low-utility units more, protect high-utility ones — ~200 LOC over AdamW). Add the **gradient-interference monitor**: first-epoch interference predicts forgetting severity, so you can forecast a bad task sequence before committing. Track dead-unit fraction, plasticity probes, forgetting curves; circuit-break on degradation. *This is also the layer that explains the §5 L3 collapse — wire its monitor into every training run, including L3's.* Frontier spike (off critical path): predictive-coding / local-credit residuals.
- **L6 motivation (overseer-gated, always).** A MAGELLAN-style learning-progress predictor proposes goals; empowerment/entropy as secondary. Every goal must be 100% decomposable into verifier-checkable sub-goals (bounded by L3's reach) and checked against L5's overseer model. A degenerate-curiosity detector catches noisy-TV/echo-chamber traps. **`propose()` exists; `act()` requires human approval — enforced at the type level, no autonomous side-effecting call path.**
- **L5 social (frontier, don't gate the system on it).** Use ExploreToM as an adversarial data/eval engine. Model agents via inverse planning over a mental-state latent subspace (reuse L2's SCM machinery). Alignment lives here (overseer-intent modelling) — which is also why deceptive alignment is a structural risk; mitigations are verifier-independent audit + hidden behavioural probes + the differentiability boundary. None complete; named in the spec, not solved.
- **L7 router (integration claim only).** A small learned attention bottleneck: capacity-limited select-and-broadcast over the shared latent (salience = free-energy-reduction × verifier-confidence × goal-relevance). **No consciousness vocabulary.** Ship only if it beats an ablation on cross-layer transfer (G7); otherwise it's scaffolding, labeled as such.

---

## 7. The evaluation harness — the most reusable artifact

This is the measurement spine; per the spec it's the most defensible original contribution. Generalize the M1 rig (`metrics.py` + `run.py --aggregate`) into a layer-agnostic harness every milestone runs through.

```python
# verge/eval/harness.py
def evaluate(layer, *, seeds=(0,1,2), rounds, frozen_test, ablations: dict,
             preregister: dict) -> Report:
    """Run >=3 seeds on a FROZEN test set, compare against named ablations,
    fit slope + 95% CI, and decide go/no-go against the PRE-REGISTERED threshold.
    A result that wasn't pre-registered doesn't count."""
```

Required of every layer before it ships:
- **Frozen test split** + contamination guard (the M1 `data.py` pattern).
- **≥3 seeds**, OLS slope + 95% CI (`metrics.fit_slope`); "flat" = CI includes zero.
- **Named ablations** — the layer's contribution must beat *removing* it (bitter-lesson gate).
- **Pre-registered go/no-go** thresholds, written before the run.
- **Free CPU mock** to validate the full pipeline before any GPU spend.
- **LR-sensitivity check** — the §5 lesson: hyperparameters from large-model papers do not transfer; decline-from-round-1 is the tell.

---

## 8. Safety enforcement (in code)

The rings (spec §4 Thread D) are import-graph constraints checked in CI, not documentation:

- **Ring 0 — frozen.** `verge/ring0/` holds the verifier kernel, the audit layer, governance thresholds, and the rollback ledger. A CI guard fails the build if any module reachable from a self-improvement path imports a *mutable* handle to Ring 0. Ring 0 has no setters.
- **Ring 1 — autonomous, verifier-gated.** Compiled skills run as **Wasm modules under wasmtime** — opaque, sandboxed, and *not FFI-able into the autograd graph*. This is the differentiability boundary: the system may compile skills for itself but can only touch the non-differentiable algorithmic surface, never the learned core.
- **Ring 2 — human-initiated learning only.** L1/L2/L4/L5/L7 weights update only through an approved learning job, never from an L6/L7 self-edit path.
- **Governance.** L6 proposals and L7 outputs are logged to the append-only audit ledger with full latent provenance (`source_layer`, `key`). The audit layer is verifier-*independent* (kernel-proof + human labels), always-on, hard-gating with halt + rollback. **The system never lowers its own approval tier.**

Prohibited-action rules (enter credentials, move money, change permissions, hard-delete, accept terms) are enforced at the tool-boundary, not left to model judgment.

---

## 9. Repository layout

```
verge/
  latent.py            # the shared-latent contract (schema = Ring 0)
  workspace.py         # the bus; LayerService base; broadcast/select
  ring0/               # FROZEN: verifier kernel, audit, thresholds, rollback ledger
    verifiers/         #   exact (built), lean, pytest-sandbox, n-version gate
    audit.py
  l1_perception/       # V-JEPA2 + DINOv2/3 projection heads, binding loss
  memory/              # FAISS HNSW + sparse reranker, margin-δ writes
  l3_reasoner/         # GRPO/DAPO (TRL→verl) + vLLM search + skeleton/render
    reward.py          #   wraps ring0 verifier
  l2_world_model/      # DreamerV3 service bridge + Deep-SCM/NCM head
  l4_plasticity/       # UPGD optimizer + gradient-interference monitor
  l5_social/ l6_motivation/ l7_router/
  eval/                # the layer-agnostic harness (§7) + mock
  configs/             # dataclasses/Hydra; calibrated defaults per layer
m1-verified-reasoner/  # the built, measured L3 — kept as-is, imported by l3_reasoner
```

---

## 10. Build sequence (concrete, in order)

1. **Lock the latent contract (§2)** + its CI test suite (no-collapse, binding, drift). Nothing above L1 starts until this is green.
2. **Harden L3:** wrap the existing `ExactMatchVerifier` as the GRPO reward; move search to vLLM; bring up TRL `GRPOTrainer` on GSM8K; reproduce the §5 flat baseline through the new harness; then push for headroom (MATH, larger base, higher K) to test for a positive slope.
3. **M2:** V-JEPA2 projection head + FAISS memory → condition L3 → gate on retrieval-vs-ablation.
4. **M3:** DreamerV3 service + NCM head → reasoner queries rollouts → gate on `do(x)` vs simulation.
5. **L4 monitor everywhere** from step 2 onward; UPGD as the default optimizer once its ablation wins.
6. **M4+:** L6 (gated) → L5 (ExploreToM eval) → L7 (router vs ablation), each through §7, each cut if it loses to "scale base + retrieval."
7. **Ring enforcement (§8)** wired into CI from step 1 — safety scaffolding precedes autonomy, never follows it.

**Floor:** everything through M3 runs on a single H100, exactly as M1 did. The contribution is the measured seam — not scale, and not a claim.
