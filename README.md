# VERGE

A layered, verifier-bounded reasoning architecture, composed from mature open
components. **All seven layers are implemented as real, CPU-runnable, tested seams** —
the *learned glue* the spec actually asks a builder to own (projection heads, FAISS-style
memory, the Neural-Causal-Model `do(x)` head, UPGD, inverse-planning ToM, MAGELLAN
learning-progress, the router). The heavy **pretrained backbones** (V-JEPA 2, DreamerV3)
and the **GPU training** (GRPO/vLLM) are kept as *injectable, optional* dependencies, so
the whole stack runs and tests with **numpy only, no GPU, no model download**.

The discipline is the spec's: the verifier is a **bounded amplifier** (`verge-spec.md`
§2), it is the only trusted asset and is **never learned** (Ring 0), and every layer
ships behind the eval harness with a pre-registered slope+CI.

> Read `Spec & Details/verge-spec.md` (§2, §5) and `Spec & Details/verge-engineering.md`
> (the blueprint) for the why. This README is the how.

## What's real (CPU) vs. injectable (GPU/pretrained)

Every layer below is a **real, tested implementation** of its learned seam, runnable on
CPU. The "injectable" column is the heavy open component that drops in behind the same
interface for production scale — none is required to run or test the stack.

| Layer / component | Real, tested seam (CPU) | Injectable heavy component | Where |
|---|---|---|---|
| Shared latent + bus + Ring 0 + ring guard + eval harness | full | — | `latent.py`, `workspace.py`, `ring0/`, `ring_check.py`, `eval/` |
| **L1** perception | projection head → unit-norm → VICReg; cross-modal **binding** loss (G1) | V-JEPA 2 / DINOv2 backbone (`backbone=`) | [l1_perception/](verge/l1_perception/) |
| **memory** | two-phase recall (approx → sparsemax rerank); **margin-δ** writes | FAISS HNSW (`verge[memory]`) | [memory/](verge/memory/) |
| **L2** world model | **linear-Gaussian NCM**: `do(x)`, counterfactual, uncertainty, out-of-skeleton flag (G2); learned transition rollout | DreamerV3 (`transition=`), DoWhy/pyro | [l2_world_model/](verge/l2_world_model/) |
| **L3** reasoner | verified expert-iteration + **GRPO/DAPO** loop; reward = Ring-0 verifier; skeleton/render | TRL `GRPOTrainer` + vLLM (`verge[l3,infer]`) | [l3_reasoner/](verge/l3_reasoner/) |
| **L4** plasticity | **UPGD** optimizer + gradient-interference monitor | wrap AdamW on torch params | [l4_plasticity/](verge/l4_plasticity/) |
| **L5** social | **Bayesian inverse planning** (recovers goals + false beliefs); overseer-intent latent | ExploreToM eval engine | [l5_social/](verge/l5_social/) |
| **L6** motivation | **MAGELLAN** learning-progress + degenerate-curiosity filter; `propose()` autonomous, `act()` human-gated by type (G6) | — | [l6_motivation/](verge/l6_motivation/) |
| **L7** router | learned capacity-limited **select-and-broadcast**; salience = FE-reduction × verifier-confidence × goal-relevance; beats no-broadcast ablation (G7) | scale the attention bottleneck on torch | [l7_router/](verge/l7_router/) |

`ExactMatchVerifier`, `metrics` (`pass_at_1`/`fit_slope`/dedup), and the mock simulator
are **imported verbatim** from the unedited `Spec & Details/m1-verified-reasoner/` — they
are not forked or modified.

> **Honesty (spec §8):** these are working *seams*, not the frontier components. The
> linear-Gaussian NCM is the testable special case of a deep SCM; the synthetic backbone
> stands in for V-JEPA 2; the inverse planner is a gridworld engine, not ExploreToM at
> scale. Each is correct and measured at what it claims, and each names the heavy
> component that replaces it. L5/L6/L7 remain **frontiers** — implemented and gated, not
> "solved."

## Install (mock path: no GPU, no model download)

```bash
uv venv .venv && uv pip install -e ".[mock]" --python .venv/bin/python
# or: python -m venv .venv && .venv/bin/pip install -e ".[mock]"
```

The mock/test path needs only **numpy + pytest**. Heavy/GPU deps are isolated behind
optional extras and are never pulled by the mock path:

```bash
pip install -e ".[l3]"      # torch, transformers, datasets, accelerate, trl, peft
pip install -e ".[infer]"   # vLLM (K-sample search)
pip install -e ".[memory]"  # faiss-cpu
pip install -e ".[world]"   # dowhy, pyro-ppl
pip install -e ".[sandbox]" # wasmtime (Ring-1 compiled-skill boundary)
pip install -e ".[all]"     # everything (a single H100 box)
```

## Run the mock (L3 end-to-end, free)

```bash
# expert-iteration engine (the M1 loop, unmodified):
.venv/bin/python -m verge.l3_reasoner.run --mock --rounds 4 --seeds 0 1 2

# GRPO/DAPO engine (the §3 swap), same loop shape, validated free on CPU:
.venv/bin/python -m verge.l3_reasoner.run --mock --engine grpo --k 8
```

Does **search → verify-filter (Ring-0 verifier) → advantage-weight/distil → measure** and
fits **slope + 95% CI** through the new harness. The mock compounds early then plateaus;
the plateau slope is **flat (CI includes zero)** — the §5 shape. (The measured *GSM8K*
result is flat from round 0 on real hardware; see `m1-verified-reasoner/RESULTS.md`.)

## Run the headroom experiment (GPU; spec §5.4)

The L3 hardening — TRL `GRPOTrainer` with the Ring-0 verifier as the reward (DAPO
clip-higher + dynamic sampling + length-bias fix, `lr=2e-6`), vLLM K-sample search — is
wired behind the same `EvalProtocol`, so the only change is the engine and the headroom
knobs. Needs `verge[l3,infer]`, a GPU, and network access:

```bash
pip install -e ".[l3,infer]"
python -m verge.l3_reasoner.run --engine grpo --dataset gsm8k \
       --base Qwen/Qwen2.5-7B-Instruct --k 16 --rounds 4 --seeds 0 1 2
```

Per spec §2 the lever is **headroom** (a stronger base, more search, a harder task), never
a cleverer objective. `--dataset math` is wired; general MATH answers need the
`MathEquivalenceRung` (sympy) — a typed stub in the ladder, not a change to the EXACT
verifier — so run GSM8K (larger base + higher K) or the numeric MATH slice until that rung
is built. **This experiment is not run in this environment** (no GPU / no model download);
the path is mock-validated and ready.

## Run the full seven-layer stack (CPU, free)

```bash
.venv/bin/python -m verge.orchestrator   # all 7 layers composing over the workspace bus
```

Shows the seam end to end: L1 binds vision+text and encodes concepts → memory stores and
retrieves them (margin-δ) → L2 fits a causal model and answers `do(x)` → L3 broadcasts a
verified skeleton → L5 infers an agent's intent → L7 broadcasts the capacity-limited
global workspace → L6 proposes a goal by learning progress → L4's monitor stands ready →
the drift circuit-breaker halts a drifting stream. Every message is a `Latent` over the
bus; the verifier stays in Ring 0.

## Run the tests

```bash
.venv/bin/python -m pytest -q          # 91 passed, 1 skipped (TRL-only test)
.venv/bin/python -m verge.ring_check   # ring-0 enforcement guard (exits non-zero on violation)
```

- `tests/test_latent_contract.py` — no-collapse, cross-modal binding, drift breaker.
- `tests/test_workspace.py` — bus broadcast/select + drift halt.
- `tests/test_ring_enforcement.py` — clean tree + bad-fixture proofs the guard fails.
- `tests/test_l3_mock.py`, `test_l3_grpo.py` — L3 expert-iteration + GRPO/DAPO loops on the mock.
- `tests/test_eval_harness.py` — slope/CI, FLAT detection, bitter-lesson ablation gate.
- `tests/test_l1_perception.py` — binding training reduces distance, passes G1, no collapse.
- `tests/test_memory.py` — margin-δ writes, two-phase retrieval, sparsemax.
- `tests/test_l2_world_model.py` — `do(x)` beats baseline (G2), out-of-skeleton flag, counterfactual, rollout.
- `tests/test_l4_plasticity.py` — UPGD non-gradient variability + utility protection; interference monitor.
- `tests/test_l5_social.py` — inverse planning recovers goals and false beliefs.
- `tests/test_l6_motivation.py` — MAGELLAN LP proposal, degenerate filter, propose≠act gate, audit log.
- `tests/test_l7_router.py` — three-factor salience; broadcast beats no-broadcast ablation (G7).
- `tests/test_orchestrator.py` — the full seven-layer compose over the bus.

## Where each layer's heavy component plugs in (scale-up)

The seam is real; this is where the production-scale open component drops in behind it.

| Layer | Plug-in point | Open component |
|---|---|---|
| **L1** perception | `verge/l1_perception/service.py::L1Perception.encode` | **V-JEPA 2** (frozen) + trained projection head + cross-modal binding loss |
| **memory** | `verge/memory/store.py::LatentMemory.{write,retrieve}` | **FAISS** HNSW + sparse reranker, margin-δ writes |
| **L2** world model | `verge/l2_world_model/service.py::L2WorldModel.{rollout,do}` | **DreamerV3** (JAX, process boundary) + **NCM/DoWhy** causal head |
| **L4** plasticity | `verge/l4_plasticity/service.py::L4Plasticity.wrap_optimizer` + `GradientInterferenceMonitor` | **UPGD** over AdamW + gradient-interference monitor |
| **L5** social | `verge/l5_social/service.py::L5Social.model_overseer` | **ExploreToM** data/eval engine + inverse planning |
| **L6** motivation | `verge/l6_motivation/service.py::L6Motivation.propose` (act() stays human-gated) | **MAGELLAN** learning-progress predictor |
| **L7** router | `verge/l7_router/service.py::L7Router.{salience,select}` | small learned attention bottleneck |
| **L3** (scale-up) | `verge/l3_reasoner/service.py` engine swap | **TRL `GRPOTrainer`** → **verl** + **vLLM** search |

## Layout (verge-engineering.md §9)

```
verge/
  latent.py            shared-latent contract (schema = Ring 0)
  workspace.py         the bus; LayerService base; broadcast/select
  ring_check.py        ring-enforcement CI guard (import-graph walk)
  ring0/               FROZEN: verifier kernel, ladder, audit, thresholds, rollback, wasm boundary
  l1_perception/  memory/  l2_world_model/  l4_plasticity/  l5_social/  l6_motivation/  l7_router/
  l3_reasoner/         reward (wraps ring0) + skeleton/render + service + run
  eval/                the layer-agnostic harness (§7) + free CPU mock
  configs/             calibrated dataclass defaults per layer
Spec & Details/m1-verified-reasoner/   the built, measured L3 — kept as-is, imported
```

See [BUILD-NOTES.md](BUILD-NOTES.md) for decisions, deviations, and the next step per stub.
# verge
