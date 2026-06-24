"""Per-layer calibrated configs (verge-engineering.md §1: dataclasses → Hydra at scale).

L3's defaults are the M1-calibrated recipe (the critical `lr=2e-6`, temp 1.0, dedup on,
reset-to-base on, frozen test). The other layers carry their open-component knobs as
typed placeholders so the seam is configurable before the layer is built.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class L3Config:
    # The calibrated recipe (m1-verified-reasoner). lr=2e-6 is critical: the 1e-5 RFT
    # default degrades a 0.5B (spec §5 / RESULTS.md).
    lr: float = 2e-6
    gen_temperature: float = 1.0
    eval_temperature: float = 0.0   # greedy; FIXED across rounds
    samples_per_problem: int = 4    # K — the search lever; raise via vLLM
    dedupe_reasoning_paths: bool = True
    reset_each_round: bool = True
    engine: str = "expert_iteration"  # → "grpo" (TRL) → "verl" at scale
    dataset: str = "gsm8k"            # "math" for §5.4 headroom (needs a math verifier rung)
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"  # bump to 7B+ for headroom
    rounds: int = 4
    seeds: tuple[int, ...] = (0, 1, 2)
    # DAPO knobs (never vanilla GRPO from base): clip-higher, dynamic sampling, no KL,
    # length-bias fix. See verge/l3_reasoner/grpo.py::GRPOSettings.
    use_vllm: bool = True             # vLLM K-sample search (the throughput lever)
    epsilon: float = 0.2
    epsilon_high: float = 0.28        # decoupled clipping
    dynamic_sampling: bool = True
    scale_rewards: bool = False
    beta: float = 0.0


@dataclass
class L1Config:
    backbone: str = "vjepa2"        # or "dinov2" for stills
    latent_dim: int = 2048
    freeze_backbone: bool = True    # never train the encoder
    vicreg_gamma: float = 1.0
    binding_epsilon: float = 0.25   # G1 gate


@dataclass
class MemoryConfig:
    index: str = "faiss-hnsw"
    reranker: str = "sparsemax"
    margin_delta: float = 0.1
    k: int = 8


@dataclass
class L2Config:
    predictive: str = "dreamerv3"   # JAX, behind the latent-contract process boundary
    causal: str = "ncm+dowhy"
    skeleton_from_verified_edges_only: bool = True


@dataclass
class L4Config:
    optimizer: str = "upgd"         # over AdamW; ships once its ablation wins
    baseline_optimizer: str = "adamw"
    monitor_gradient_interference: bool = True


@dataclass
class L6Config:
    predictor: str = "magellan"     # learning-progress primary
    require_full_decomposition: bool = True   # G6: 100% verifier-checkable
    autonomous_action: bool = False           # propose-only, indefinitely


@dataclass
class VergeConfig:
    l3: L3Config = field(default_factory=L3Config)
    l1: L1Config = field(default_factory=L1Config)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    l2: L2Config = field(default_factory=L2Config)
    l4: L4Config = field(default_factory=L4Config)
    l6: L6Config = field(default_factory=L6Config)


__all__ = ["VergeConfig", "L3Config", "L1Config", "MemoryConfig", "L2Config",
           "L4Config", "L6Config"]
