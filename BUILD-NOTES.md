# BUILD-NOTES — VERGE skeleton

Decisions, deviations from `verge-engineering.md`, and the precise next step for each
stub. Build order followed `verge-engineering.md` §10: latent contract → ring enforcement
→ L3 → eval harness → stub layers → packaging/docs. Tests were kept green throughout
(never more than one component ahead of a passing test).

## Decisions

- **One definition of the verifier and the metrics.** `ExactMatchVerifier`, `metrics`
  (`pass_at_1`/`fit_slope`/`dedupe_reasoning_paths`), and `MockModel` are **imported
  verbatim** from `Spec & Details/m1-verified-reasoner/` via a path shim
  ([verge/_m1bridge.py](verge/_m1bridge.py)). Nothing in that directory was edited or
  forked (build constraint §2.6). `verge.ring0.verifiers.exact` and `verge.eval.metrics`
  are thin re-export modules.
- **The m1 directory location.** §9 shows `m1-verified-reasoner/` at repo root; it ships
  under `Spec & Details/`. The bridge locates it in either place (and unedited), so no
  files had to move.
- **Ring 0 has no setters.** `verge/ring0/` exposes only read-only handles plus one
  append-only audit write (`record`) that cannot lower a tier. The mutable ledger object
  (`_LEDGER`) is module-private and never exported. The single source of truth for what a
  self-improvement path may import is `verge.ring0.RING0_READONLY_EXPORTS`.
- **Ring-enforcement as an import-graph walk.** [verge/ring_check.py](verge/ring_check.py)
  walks imports from the self-improvement roots (L1–L7 + memory) and fails the build if
  any reachable non-Ring-0 module imports a Ring-0 name outside the allowlist, a whole
  Ring-0 module, or a wildcard. The bad-fixture proof lives *inside* the test as strings
  (`tests/test_ring_enforcement.py`), so no violating code ever sits in the tree.
- **Drift breaker tiers.** `DriftMonitor` trips `SLOW_ANCHOR` only on the **conjunction**
  of drift + a coherence dip (per §2), and `HALT_ROLLBACK` on severe drift alone. The bus
  raises `DriftHalt` on `HALT_ROLLBACK` so the orchestrator rolls back rather than
  propagating a poisoned latent.
- **No-collapse measured two ways.** Effective rank is computed on the *un-centered* data
  matrix (collapse = rank-1 directions); VICReg's variance hinge separately catches
  per-dimension variance collapse. Together they are the no-collapse CI gate.
- **L3 mock reproduces the §5 *shape*, honestly.** The M1 mock models *generation* (skill
  rises when distilled), not the verifier-reachable *ceiling*, so its full curve
  compounds. The runner additionally fits the slope over the **plateau** (post-
  consolidation rounds), which is **flat — CI includes zero**, the §5 signature. The real
  flat-from-round-0 result is GSM8K on hardware (`m1-verified-reasoner/RESULTS.md`).

## Deviations from verge-engineering.md (flagged, per the brief)

1. **`Latent.z` is `numpy.ndarray`, not `torch.Tensor` (§2 types it as `torch.Tensor`).**
   Rationale: the acceptance criteria require `pip install -e .[mock]` and `pytest` to be
   green with **no GPU and no heavy deps**; making the load-bearing contract depend on
   torch at import time would violate that. numpy is lightweight and always present. The
   *contract* — `LATENT_DIM=2048`, unit-norm `z`, the field set (`modality`,
   `source_layer`, `confidence`, `key`) — is preserved exactly, and `Latent.from_torch` /
   `Latent.to_torch` bridge to torch when it is installed (`verge[l3]`). This is the only
   semantic deviation; everything else follows the doc.
2. **Salience is confidence-only in the skeleton bus.** §6 defines salience as
   free-energy-reduction × verifier-confidence × goal-relevance; only the confidence
   channel is wired (the other two are L2/L6 signals). `WorkspaceBus.select` documents
   this; L7's learned router is where the full product lands.

No conflicts with the engineering doc were found beyond these; where the brief and the
doc could differ, the doc was treated as ground truth.

## Next step per stub (the open component to wire)

- **L1 perception** — load V-JEPA 2 weights, freeze the backbone, train only a projection
  head into `LATENT_DIM` (unit-norm, VICReg) + the cross-modal contrastive binding loss;
  wire `run_binding_gate` to the G1 probe. → `verge/l1_perception/service.py`.
- **memory** — build the FAISS HNSW index + sparse reranker; implement the margin-δ write
  policy; gate on retrieval-vs-no-retrieval under `verge.eval` (M2). →
  `verge/memory/store.py`.
- **L2 world model** — stand up DreamerV3 behind the latent-contract process boundary
  (Arrow/IPC) for `rollout`; build the NCM/DoWhy `do(x)` head over a **verified-edge-only**
  skeleton with calibrated out-of-skeleton uncertainty (G2). → `verge/l2_world_model/service.py`.
- **L4 plasticity** — implement UPGD (~200 LOC over AdamW) in `wrap_optimizer`; implement
  `GradientInterferenceMonitor.observe`; wire the monitor into the L3 training loop first
  (it explains the §5 collapse). Ship UPGD only once its ablation wins. → `verge/l4_plasticity/service.py`.
- **L5 social** — drive ExploreToM as the data/eval engine; implement inverse planning
  over a mental-state latent subspace reusing L2's SCM machinery; `model_overseer` is the
  alignment-bearing call. Frontier; do **not** gate the system on it. → `verge/l5_social/service.py`.
- **L6 motivation** — implement MAGELLAN learning-progress proposal in `propose`; implement
  `decompose` to 100% verifier-checkable sub-goals (G6); a degenerate-curiosity detector.
  `act()` stays human-gated **by type** (`HumanApproval` token) indefinitely. → `verge/l6_motivation/service.py`.
- **L7 router** — implement the learned attention bottleneck (`salience`, `select`); ship
  only if it beats an ablation on cross-layer transfer (G7). No consciousness vocabulary.
  → `verge/l7_router/service.py`.
- **L3 scale-up (the single highest-value next step)** — swap the engine from the M1
  expert-iteration loop to **TRL `GRPOTrainer`** (then verl) with `reward.py` as the
  reward function, move K-sample search to **vLLM**, and push for **headroom**: MATH and a
  7B+ base at higher K, to test for a positive slope (spec §5.4). Never vanilla GRPO from
  base; use DAPO's decoupled clipping + dynamic sampling. → `verge/l3_reasoner/service.py`.

## L3 hardening — the GRPO/DAPO engine swap (spec §5.4, the headroom path)

Built the §3 upgrade behind the existing `EvalProtocol`, so swapping the engine is local:

- **`verge/l3_reasoner/grpo.py`** — `GRPOEngine` (real TRL `GRPOTrainer`, all heavy imports
  lazy) + `MockGRPOEngine` (free CPU validation of the loop shape) + `GRPOSettings`. The
  reward is `make_trl_reward()` → the Ring-0 verifier and **nothing else**. Defaults encode
  **DAPO** (never vanilla GRPO from base): clip-higher (`epsilon_high=0.28 > epsilon=0.2`),
  dynamic sampling (drop zero-advantage groups), `scale_rewards=False` (length-bias fix),
  `beta=0.0` (no KL), and the calibrated `lr=2e-6`. `to_trl_kwargs()` filters to the fields
  the installed TRL actually accepts (version-tolerant).
- **`verge/l3_reasoner/search.py`** — `VLLMSearch`, the throughput lever for raising K
  (lazy vllm/transformers). TRL's in-trainer generation uses `use_vllm=True`.
- **`verge/l3_reasoner/data.py`** — GSM8K + MATH loaders into `Problem`, reusing the M1
  contamination guard verbatim. Honest caveat: the EXACT verifier is numeric, so MATH
  defaults to `numeric_only=True`; general MATH needs `MathEquivalenceRung` (sympy) —
  added as a **typed stub** in the ladder, never a modification of `ExactMatchVerifier`.
- **`run.py`** — `--engine {expert_iteration,grpo}`, `--dataset {gsm8k,math}`, `--base`,
  `--k`, `--steps-per-round`. The real path surfaces a clear missing-dep/network message.

What is **not** done here: the actual GPU headroom run. This environment has no GPU and no
network model download, so the run was not executed and **no positive-slope result is
claimed**. The GRPO loop is mock-validated on CPU (`--mock --engine grpo`): it exercises
the real Ring-0 reward over grouped samples, applies DAPO dynamic sampling, and reproduces
the bounded rise-then-plateau (plateau CI includes zero). The exact reproduce command is in
the README; correctness of a positive slope, if it appears, must still be scrutinized for a
verifier/contamination leak before being believed (spec §5.4).

## All seven layers — real CPU seams (the second milestone)

Every layer was taken from a typed stub to a **real, tested implementation of its learned
seam**, runnable on CPU with numpy. The heavy pretrained backbones and GPU training are
**injectable optionals** behind the same interfaces — none is needed to run or test.

- **Shared numpy substrate** ([verge/_numpy_nn.py](verge/_numpy_nn.py)) — `Linear` with
  explicit MSE / output-gradient steps (standard backprop baseline, §2.4), `unit_norm`,
  `softmax`, VICReg variance push. The small learned components are numpy, not foundation
  models (§2.1).
- **L1** — projection head (`Linear`→unit-norm) over an **injectable backbone** (default:
  a deterministic synthetic feature map; inject V-JEPA 2 / DINOv2). `train_binding` reduces
  paired vision/text distance (alignment + unit-norm anchor + full-batch VICReg) — verified
  to drop within-pair distance ~1.0→0.0 while effective rank stays >5 (no collapse), passing
  G1. The early per-pair VICReg push diverged (overflow on 2-sample batches); fixed by
  full-batch updates + a unit-norm anchor.
- **memory** — two-phase recall (FAISS HNSW when `verge[memory]` present, else an exact
  numpy scan with identical results) + a **sparsemax** reranker + **margin-δ** writes that
  reject near-duplicates. Tested: margin-δ rejection, query-is-own-nearest-neighbour.
- **L2** — a **linear-Gaussian Neural Causal Model** ([ncm.py](verge/l2_world_model/ncm.py)):
  skeleton from verified edges only, functional forms by least squares, `do(x)` by
  topological propagation with variance, counterfactual by abduction→action→prediction, and
  an **out-of-skeleton flag** that fires for unknown variables or effects routed through an
  unverified edge. Tested: `do(x)` error 0.006 vs baseline 1.5 (G2); counterfactual exact;
  flag fires. A fixed orthonormal embedding maps the factor space ↔ the 2048-d latent; the
  emitted latents are unit-norm by contract (scale is normalized away — rollout matches the
  true dynamics in *direction*). DreamerV3 is the injectable transition.
- **L4** — a real **UPGD** optimizer ([upgd.py](verge/l4_plasticity/upgd.py)) + the
  gradient-interference monitor. Rather than chase a fragile emergent ossification result in
  a toy net, the tests assert UPGD's *guaranteed* mechanism: it moves a zero-gradient weight
  (the non-gradient variability SGD lacks — Dohare's whole point) and protects high-utility
  weights. The monitor distinguishes conflicting from aligned task gradients and circuit-breaks.
- **L5** — **Bayesian inverse planning** on a gridworld
  ([inverse_planning.py](verge/l5_social/inverse_planning.py)): a Boltzmann-rational planner
  likelihood, posterior over believed-targets. Tested to recover a planted goal AND to
  attribute a **false belief** (the Sally-Anne structure — infers where the agent *thinks*
  the reward is, not the true location). Same machinery models overseer intent. ExploreToM
  is the scale-up eval engine.
- **L6** — **MAGELLAN** learning-progress predictor ([magellan.py](verge/l6_motivation/magellan.py))
  (competence derivative, with a least-squares LP generalizer for unseen goals) + a
  degenerate-curiosity detector (noisy-TV / echo-chamber). `propose()` is autonomous;
  `act()` requires a `HumanApproval` token **by signature** and a 100%-decomposable goal
  (G6), and only appends to the Ring-0 audit ledger — a safe, finished, indefinitely
  overseer-gated `act()` (it imports only allowlisted read-only Ring-0 handles, so the ring
  guard stays green).
- **L7** — a learned capacity-limited **select-and-broadcast** router: salience =
  FE-reduction × verifier-confidence × goal-relevance (weighted product, learnable
  log-weights). Tested: broadcast recovers a goal direction ~0.98 vs a no-broadcast (first-k)
  ablation ~0.03 averaged over seeds (G7).
- **Orchestration** ([verge/orchestrator.py](verge/orchestrator.py), `python -m
  verge.orchestrator`) — the seven layers composing over the `WorkspaceBus`, every message a
  `Latent`. The global drift monitor is demonstrated on a single coherent stream (broadcasting
  deliberately heterogeneous modalities legitimately trips the breaker — that is the breaker
  working; re-keying re-anchors on intentional context switches).

**Honest scope (unchanged discipline).** These are working seams, deliberately the testable
special cases: linear-Gaussian SCM (not a deep SCM), synthetic backbone (not V-JEPA 2),
gridworld inverse planner (not ExploreToM), numpy UPGD (not the AdamW wrapper at scale). Each
is correct at what it claims and names the heavy component that replaces it. L5/L6/L7 remain
frontiers — implemented and gated, not solved. Nothing here was run on a GPU or downloaded a
model.

## Ring-1 / safety boundary

`verge/ring0/sandbox.py` defines the Wasm/wasmtime differentiability-boundary interface
(`CompiledSkill`, `WasmSkillSandbox`); the compiled-skill execution path is a `# TODO`
behind `verge[sandbox]`. `verge/ring_check.py` is the always-on CI guard; run it (or the
ring-enforcement test) in CI before any autonomy work, per §10 step 7 — safety scaffolding
precedes autonomy, never follows it.
