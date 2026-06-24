# Build Prompt — VERGE Skeleton (hand this to a fresh Opus coding session)

You are implementing the first code milestone of the VERGE architecture. You have a full coding environment (file tools, a shell, Python 3.11+). Work in the repository root that contains `verge-spec.md`, `verge-engineering.md`, and the `m1-verified-reasoner/` folder.

## 0. Read first (do not skip)

Before writing any code, read these three sources in full and treat them as ground truth:

1. **`verge-engineering.md`** — the stack, the shared-latent contract (§2), the verification ladder (§3), the repo layout (§9), and the build sequence (§10). This is your blueprint.
2. **`verge-spec.md`** — the architecture and, critically, §2 (the verifier is a *bounded amplifier*, not a generator) and §5 (the measured L3 result). Every design choice must respect §2.
3. **`m1-verified-reasoner/`** — the already-built, already-measured L3. Read `verifiers.py`, `metrics.py`, `expert_iteration.py`, `model.py`, `config.py`, `mock_model.py`, and `RESULTS.md`. **You will reuse this code, not rewrite it.**

If anything in this prompt conflicts with `verge-engineering.md`, the engineering doc wins — flag the conflict in your summary rather than guessing.

## 1. What to build (scope: full 7-layer skeleton, L3 fully implemented)

Produce a coherent, **compiling, test-passing** Python package `verge/` that realizes the repo layout in `verge-engineering.md` §9, with this depth per component:

- **Shared-latent contract + workspace bus + ring enforcement — fully implemented and tested.** This is the load-bearing seam; it must be real, not a stub.
- **L3 reasoner — fully implemented**, by wrapping and orchestrating the existing `m1-verified-reasoner/` code behind the new interfaces (verifier as Ring-0 asset, the GRPO reward wrapper, the skeleton/rendering output contract, the verification-ladder abstraction). It must run end-to-end on the CPU mock with no GPU.
- **L1, L2, L4, L5, L6, L7 — typed, compiling stubs.** Each implements the `LayerService` interface, has correct type signatures, raises `NotImplementedError` with a `# TODO(Lx):` note citing the open component to use (V-JEPA 2, DreamerV3+NCM, UPGD, ExploreToM, MAGELLAN, learned router), and has a `@pytest.mark.skip(reason="stub")` test asserting the intended contract. A stub is a *promise with a signature*, not an empty file.
- **The eval harness (`verge/eval/`) — fully implemented**, generalizing `m1-verified-reasoner/metrics.py` (`pass_at_1`, `fit_slope` slope+95% CI, distinct-path dedup) into the layer-agnostic `evaluate(...)` of `verge-engineering.md` §7, with the free CPU mock path wired in.

## 2. Hard constraints (these are the point — violating them defeats the architecture)

1. **Never train a foundation model from scratch.** Every model is pretrained/open. You are building the *seam*.
2. **The verifier is Ring 0 and is never learned.** Reuse `m1-verified-reasoner/verifiers.py`'s `ExactMatchVerifier` verbatim under `verge/ring0/`. Do not modify its logic. Ring 0 (verifier kernel, audit, governance thresholds, rollback ledger) has **no setters** and must not be reachable as a mutable handle from any self-improvement path.
3. **One shared latent, one contract.** No layer parses another layer's internals; all cross-layer communication is `Latent` objects over the workspace bus. Implement the `Latent` dataclass and `LayerService` base exactly as in `verge-engineering.md` §2 (LATENT_DIM=2048, unit-norm `z`, `modality`/`source_layer`/`confidence`/`key`).
4. **Standard backprop baseline only.** Do not implement local-credit/hybrid-gradient work; leave it as a documented L4 frontier stub.
5. **Respect §2 of the spec in L3.** Only verifier-confirmed traces ever enter training; the worst case must be wasted samples, never poisoned weights. Do not add any differentiable guide that labels training data.
6. **Keep `m1-verified-reasoner/` intact.** Import from it; don't fork or edit it. If you need a change there, document it instead of making it.

## 3. Concrete deliverables

1. The `verge/` package per §9 layout, installable (`pip install -e .` or `uv`), with `pyproject.toml` pinning the §1 stack (PyTorch, transformers, vLLM, TRL, FAISS, wasmtime, etc. — but mark GPU/heavy deps optional so the mock path installs clean).
2. **The latent contract CI suite** (`verge/latent.py` + tests): no-collapse (effective-rank floor via VICReg regularizer presence), cross-modal binding probe (interface + a synthetic test), and the drift circuit-breaker (drift + coherence-dip → slow/anchor; severe → halt/rollback). Tests must pass on CPU with synthetic latents.
3. **Ring-enforcement CI guard**: a test that walks the import graph and **fails the build** if any module reachable from a self-improvement path imports a mutable Ring-0 handle. Wasm/wasmtime sandbox boundary stubbed with the interface defined and a `# TODO` for the compiled-skill path.
4. **L3 end-to-end on the mock**: a runnable command (e.g. `python -m verge.l3_reasoner.run --mock`) that does search → verify-filter → dedupe → reset-to-base → measure, reusing the M1 loop, and prints slope+CI through the new harness. Reproduce the §5 "flat" shape on the mock.
5. **`README.md`** at repo root for the `verge/` package: how to install, run the mock, run the tests, and the exact place each stubbed layer plugs in its open component.
6. A short **`BUILD-NOTES.md`** logging decisions, any deviations from `verge-engineering.md`, and the precise next step for each stub.

## 4. Definition of done (acceptance criteria — verify each before finishing)

- [ ] `pip install -e .` (mock extras) succeeds with no GPU and no network model download.
- [ ] `pytest` is green: all real tests pass; all stub tests are explicitly skipped with a reason.
- [ ] The latent-contract tests (no-collapse, binding, drift) pass on synthetic data.
- [ ] The ring-enforcement import-graph guard passes (and you've added a deliberately-bad fixture in a test to prove the guard *fails* when a violation is introduced, then removed it).
- [ ] L3 runs end-to-end on the mock and prints a slope + 95% CI via `verge/eval`.
- [ ] Every stub layer implements `LayerService`, compiles, and names its open component in a `# TODO(Lx):`.
- [ ] No edits to `m1-verified-reasoner/`; no modification to `ExactMatchVerifier` logic.
- [ ] `README.md` + `BUILD-NOTES.md` written.

## 5. How to work

Build in the order of `verge-engineering.md` §10: **(1)** latent contract + its CI suite first — nothing else starts until it's green; **(2)** ring enforcement wired into CI; **(3)** L3 hardened behind the new interfaces and running on the mock; **(4)** the eval harness; **(5)** the six stub layers; **(6)** packaging + docs. Run the CPU mock and the test suite continually — do not write more than one component ahead of a passing test. Use a task list to track the milestones.

When you finish, give a concise summary: what's real, what's stubbed, how to run it, the test status, and the single highest-value next step (hardening L3 toward a positive-slope headroom run on MATH/larger base, per spec §5.4). Do not overstate — the spec's whole discipline is honesty about what is measured versus what is promised.
