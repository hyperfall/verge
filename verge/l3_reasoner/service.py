"""L3 as a `LayerService` + the `EvalProtocol` adapter that drives the measured loop.

The adapter (`L3EvalAdapter`) orchestrates the *unmodified* M1 expert-iteration loop —
search → verify-filter → dedupe → reset-to-base → measure-on-frozen-test — behind the
new harness, returning the per-round single-shot pass@1 curve. The reward/filter is the
Ring-0 verifier (`reward.py`); the engine swap to TRL `GRPOTrainer` (then verl) replaces
only the inner `model.finetune` call, leaving this orchestration intact.

`L3Reasoner` is the `LayerService` face: it encodes problems/skeletons into the shared
latent and exposes `health()`. The text→latent map here is a deterministic stand-in for
the learned projection head (M2); the *contract* it satisfies is real and tested.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from verge import _m1bridge
from verge.latent import LATENT_DIM, Latent, LayerService, make_latent
from verge.l3_reasoner.skeleton import Skeleton

_m1bridge.ensure_on_path()


def _text_to_vec(text: str) -> np.ndarray:
    """Deterministic hashing feature map text -> R^LATENT_DIM. A stand-in for the
    learned projection head; deterministic and unit-normable so the bus/contract are
    demonstrable on CPU. NOT a semantic encoder (that is M2 / V-JEPA2 + binding loss)."""
    rng = np.random.default_rng(
        int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little"))
    return rng.standard_normal(LATENT_DIM)


@dataclass
class L3Reasoner(LayerService):
    """The verified reasoner as a workspace participant."""

    layer_id: str = "L3"
    _last_health: dict = field(default_factory=lambda: {"verifier_reach": "EXACT"})

    def encode(self, x) -> list[Latent]:
        """Encode a problem prompt or a verified skeleton's answer into the shared
        latent (modality='concept', source_layer='L3')."""
        text = x.answer if isinstance(x, Skeleton) else str(x)
        conf = 1.0 if isinstance(x, Skeleton) and x.verified else 0.5
        return [make_latent(_text_to_vec(text), modality="concept",
                            source_layer="L3", confidence=conf)]

    def step(self, ctx: list[Latent]) -> list[Latent]:
        # TODO(L3): full deliberation = verifier-ranked search over world-model (L2) +
        # memory (M2), distilling winning skeletons. The measured loop lives in
        # L3EvalAdapter; this bus-level step is wired once M2/M3 land.
        raise NotImplementedError(
            "L3.step (bus-level deliberation over L2+memory) lands with M2/M3; "
            "the measured search→verify→distil loop is L3EvalAdapter.run_seed.")

    def health(self) -> dict:
        return dict(self._last_health)


@dataclass
class L3EvalAdapter:
    """Drives the unmodified M1 expert-iteration loop through the §7 harness.

    Implements `EvalProtocol`: `run_seed` returns the single-shot pass@1 curve over
    rounds 0..rounds on the frozen test set.
    """

    cfg: object
    train: list
    name: str = "L3-verified-reasoner"

    def run_seed(self, seed: int, *, rounds: int, frozen_test) -> list[float]:
        from expert_iteration import run as run_loop  # M1, unedited

        self.cfg.loop.rounds = rounds
        logs = run_loop(self.cfg, seed, self.train, frozen_test)
        return [lg.pass_at_1 for lg in logs]


def build_mock_adapter(*, rounds: int = 4, n_train: int = 400, n_test: int = 300,
                       output_dir: str = "runs/verge_l3_mock"):
    """Assemble the calibrated mock L3 adapter + frozen test set (no GPU, no ML deps)."""
    from config import Config  # M1 calibrated defaults

    from verge.eval.mock import make_mock_problems

    cfg = Config()
    cfg.use_mock = True
    cfg.loop.rounds = rounds
    cfg.output_dir = output_dir
    cfg.data.train_size = min(cfg.data.train_size, n_train)
    cfg.data.test_size = min(cfg.data.test_size, n_test)
    train, test = make_mock_problems(cfg.data.train_size, cfg.data.test_size)
    return L3EvalAdapter(cfg=cfg, train=train), test
