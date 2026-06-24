"""End-to-end orchestration demo — the seven layers composing over the workspace bus.

    python -m verge.orchestrator

Shows the *seam* working: every layer communicates only via `Latent`s on the
`WorkspaceBus`, the verifier stays in Ring 0, and nothing here needs a GPU or a model
download. This is an integration smoke demo (the M2/M3 shape), not a benchmark.
"""
from __future__ import annotations

import numpy as np

from verge.latent import DriftMonitor, make_latent
from verge.l1_perception import L1Perception, synth_pair
from verge.l2_world_model import CausalEdge, CausalSkeleton, L2WorldModel
from verge.l3_reasoner.skeleton import Skeleton
from verge.l3_reasoner.service import L3Reasoner
from verge.l4_plasticity import L4Plasticity
from verge.l5_social import L5Social, sample_trajectory
from verge.l6_motivation import Goal, L6Motivation
from verge.l7_router import L7Router
from verge.memory import LatentMemory
from verge.workspace import WorkspaceBus


def run_demo(verbose: bool = True) -> dict:
    log = (lambda *a: print(*a)) if verbose else (lambda *a: None)
    # The bus carries heterogeneous modalities here, so the drift breaker is demonstrated
    # separately below on a single coherent stream (where drift is actually meaningful).
    bus = WorkspaceBus(capacity=7)
    L1, L2, L3 = L1Perception(), L2WorldModel(), L3Reasoner()
    L4, L5, L6, L7 = L4Plasticity(), L5Social(), L6Motivation(), L7Router()
    for svc in (L1, L2, L3, L4, L5, L6, L7):
        bus.register(svc)

    # --- L1: bind vision+text, encode concepts onto the bus ------------------
    pairs = [synth_pair(f"concept{i}") for i in range(12)]
    L1.train_binding(pairs, epochs=300, lr=0.05)
    gate = L1.run_binding_gate(pairs)
    log(f"[L1] binding gate passed={gate['passed']} (within={gate['mean_within']:.3f}, "
        f"eff_rank={gate['effective_rank']:.1f})")
    concept_latents = [L1.encode(("vision", f"concept{i}"))[0] for i in range(12)]
    bus.broadcast(concept_latents)

    # --- memory: store concepts, retrieve a neighbourhood --------------------
    mem = LatentMemory(k=3)
    kept = mem.write_many(concept_latents)
    retrieved = mem.retrieve(concept_latents[0], k=3)
    log(f"[memory] stored {kept}/12 (margin-δ); retrieved {len(retrieved)} for a query")

    # --- L2: fit a tiny causal model, do(x), broadcast the predicted state ---
    rng = np.random.default_rng(0)
    X0 = rng.standard_normal(2000)
    X1 = 1.5 * X0 + 0.2 * rng.standard_normal(2000)
    sk = CausalSkeleton(("X0", "X1"), (CausalEdge("X0", "X1", True),))
    L2.fit_causal(sk, {"X0": X0, "X1": X1})
    do_result = L2.do({"X0": 1.0})
    state_latent = L2.do_latent({"X0": 1.0})
    bus.broadcast([state_latent])
    log(f"[L2] do(X0=1) -> X1≈{do_result.values['X1']:.2f} "
        f"(flag={do_result.out_of_skeleton}); state latent conf={state_latent.confidence:.2f}")

    # --- L3: verified skeleton -> latent on the bus --------------------------
    skel = Skeleton(answer="72", verified=True)
    l3_latent = L3.encode(skel)[0]
    bus.broadcast([l3_latent])
    log(f"[L3] verified skeleton latent broadcast (conf={l3_latent.confidence:.2f})")

    # --- L5: model an observed agent's intent --------------------------------
    traj = sample_trajectory((2, 2), (4, 4), grid=(5, 5), beta=5.0, steps=8, seed=1)
    intent = L5.model_overseer(traj, [(0, 0), (4, 4), (0, 4)])
    bus.broadcast([intent])
    log(f"[L5] inferred intent latent (conf={intent.confidence:.2f})")

    # --- L7: capacity-limited broadcast toward the inferred goal -------------
    selected = L7.broadcast(bus.select(), goal=intent)
    log(f"[L7] global workspace = {len(selected)} latents (capacity {L7.capacity})")

    # --- L6: propose goals by learning progress (autonomous; act() is gated) -
    for o in [0, 0, 0, 0, 1, 1, 1, 1]:   # competence rising → positive learning progress
        L6.record_outcome("solve-MATH", o)
    proposals = L6.propose([Goal("solve MATH", key="solve-MATH")])
    log(f"[L6] proposed {len(proposals)} goal(s); top LP="
        f"{proposals[0].learning_progress:+.2f}" if proposals else "[L6] none")

    # --- L4: the plasticity monitor is available to every training loop ------
    log(f"[L4] optimizer={L4.health()['optimizer']}, "
        f"interference_breaker={L4.health()['interference_breaker']}")

    # --- the drift circuit-breaker on a single coherent stream ---------------
    mon = DriftMonitor()
    ref = rng.standard_normal(2048)
    stable = [make_latent(ref + 0.02 * rng.standard_normal(2048), modality="state",
                          source_layer="L2") for _ in range(16)]
    mon.anchor(np.stack([l.z for l in stable]))
    drifted = [make_latent(-ref + 0.05 * rng.standard_normal(2048), modality="state",
                           source_layer="L2") for _ in range(16)]
    log(f"[drift] stable stream -> {mon.check(stable)['decision']}; "
        f"drifted stream -> {mon.check(drifted)['decision']}")

    audit = bus.audit_trail()
    log(f"[bus] {len([e for e in audit if e['event']=='broadcast'])} broadcasts logged "
        f"with full provenance over the shared latent contract")
    return {"binding_passed": gate["passed"], "broadcasts": len(bus.history()),
            "workspace_size": len(selected), "proposals": len(proposals)}


if __name__ == "__main__":
    run_demo()
