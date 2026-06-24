"""M1 configuration — expert-iteration verified reasoner (calibrated to RFT/ReST^EM).

One place for every knob. Defaults are the recipe we validated:
few-shot format, temp-1.0 sampling, distinct-path dedup, train-from-base each round.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen2.5-0.5B-Instruct"   # bump to -1.5B-Instruct once it runs
    max_new_tokens: int = 320                    # GSM8K solutions fit; big time lever
    gen_temperature: float = 1.0                 # RFT: diversity -> dedup is the mechanism
    gen_top_p: float = 0.95
    samples_per_problem: int = 4                 # K
    gen_batch_problems: int = 64                 # batch width (raise on big-VRAM cards)
    eval_temperature: float = 0.0                # greedy; FIXED across rounds


@dataclass
class TrainConfig:
    lr: float = 1e-5                             # full-FT (RFT standard)
    epochs_per_round: int = 1
    batch_size: int = 8
    max_pool: int = 1500                         # cap accumulated training set
    max_trace_chars: int = 1600                  # drop degenerate over-long traces
    reset_each_round: bool = True                # STaR/ReST^EM: re-train FROM BASE
    dedupe_reasoning_paths: bool = True          # RFT's key ingredient


@dataclass
class DataConfig:
    dataset_name: str = "openai/gsm8k"
    train_size: int = 1000
    test_size: int = 500                         # FROZEN held-out
    contamination_check: bool = True


@dataclass
class LoopConfig:
    rounds: int = 4
    seeds: tuple[int, ...] = (0, 1, 2)


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    output_dir: str = "runs"
    use_mock: bool = False                       # CPU simulator, no ML deps
    wall_clock_min: float = 0.0                  # >0 => stop a run after N min (partial saved)
