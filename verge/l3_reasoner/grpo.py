"""The L3 engine swap: GRPO/DAPO under RLVR (verge-engineering.md §3, upgrade 1).

Replaces the hand-rolled SFT-on-survivors loop with **TRL `GRPOTrainer`** (then verl at
scale). The reward is the Ring-0 verifier and nothing else (`reward.py`) — GRPO samples a
group, scores each completion with the verifiable reward, and advantage-weights. Per spec
§2 this keeps the loop a bounded amplifier: only verifier-confirmed traces gain advantage,
so the worst case is wasted samples, never poisoned weights.

**Never vanilla GRPO from base.** `GRPOSettings` defaults encode **DAPO**'s fixes for the
known GRPO failures:
  - **clip-higher** — decoupled clipping (`epsilon_high > epsilon_low`) to fight entropy/
    mode collapse;
  - **dynamic sampling** — drop groups whose rewards are all-equal (zero advantage), so
    every step carries signal;
  - **length-bias fix** — token-level loss / `scale_rewards=False`, so long *wrong*
    answers are not under-penalized;
  - **no KL** (`beta=0.0`), as in DAPO.
And the calibrated `lr=2e-6` — NOT the 1e-5 RFT default that degraded the 0.5B (spec §5).

`GRPOEngine` is the real (GPU) path, all heavy imports lazy. `MockGRPOEngine` validates
the *new loop shape* (group sampling → verifier reward → dynamic-sampling filter → policy
step → measure) free on CPU, exercising the real Ring-0 reward. Both implement
`EvalProtocol`, so `verge.eval.evaluate` drives them unchanged.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field

from verge.l3_reasoner.reward import reward, reward_batch


@dataclass
class GRPOSettings:
    """The calibrated GRPO/DAPO knobs. `to_trl_kwargs` maps them onto TRL `GRPOConfig`,
    version-tolerantly (unknown fields are dropped, not crashed on)."""

    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"  # bump to 7B+ for headroom (§5.4)
    lr: float = 2e-6                 # calibrated; the 1e-5 default degrades small models
    num_generations: int = 8         # GRPO group size K — the search lever; raise it
    temperature: float = 1.0
    max_completion_length: int = 320
    # DAPO
    epsilon: float = 0.2             # lower clip
    epsilon_high: float = 0.28       # clip-higher (decoupled)
    dynamic_sampling: bool = True    # drop zero-advantage groups
    scale_rewards: bool = False      # length-bias fix
    beta: float = 0.0                # no KL (DAPO)
    # generation backend + schedule
    use_vllm: bool = True            # the throughput lever (verge[infer])
    vllm_mode: str = "colocate"      # single-GPU: vLLM shares the training GPU with the policy
    vllm_gpu_memory_utilization: float = 0.3  # vLLM's share; training keeps the rest of the GPU
    per_device_train_batch_size: int = 8
    gradient_checkpointing: bool = False  # trade compute for memory (full-FT 7B needs it)
    optim: str = "adamw_torch"       # "paged_adamw_8bit" to full-FT 7B on ONE 80GB GPU
    load_in_bf16: bool = False       # load weights in bf16 (halves param memory; 7B/1-GPU)
    eval_batch_size: int = 32        # frozen-test eval batched (was 1-at-a-time)
    steps_per_round: int = 100       # GRPO steps between frozen-test measurements
    eval_rounds: int = 4

    def to_trl_kwargs(self) -> dict:
        """Map to TRL `GRPOConfig` kwargs, filtered to fields the installed TRL accepts."""
        desired = {
            "learning_rate": self.lr,
            "num_generations": self.num_generations,
            "temperature": self.temperature,
            "max_completion_length": self.max_completion_length,
            "epsilon": self.epsilon,
            "epsilon_high": self.epsilon_high,
            "scale_rewards": self.scale_rewards,
            "beta": self.beta,
            "use_vllm": self.use_vllm,
            "optim": self.optim,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            # DAPO dynamic sampling surfaces under different names across versions:
            "mask_truncated_completions": True,
        }
        if self.use_vllm:  # only meaningful with vLLM; names vary across TRL versions
            desired["vllm_mode"] = self.vllm_mode
            desired["vllm_gpu_memory_utilization"] = self.vllm_gpu_memory_utilization
        from trl import GRPOConfig  # lazy

        accepted = set(inspect.signature(GRPOConfig.__init__).parameters)
        return {k: v for k, v in desired.items() if k in accepted}


def make_trl_reward(answer_key: str = "answer"):
    """A TRL `reward_funcs` callable: the Ring-0 verifier as the ONLY signal.

    TRL calls `f(prompts, completions, **cols)` where `cols` carries dataset columns
    (here the gold `answer`). Returns one reward per completion."""

    def _reward(prompts, completions, **cols):
        answers = cols.get(answer_key)
        # completions may be chat-format [{role,content}] or raw strings
        texts = [c[0]["content"] if isinstance(c, list) else c for c in completions]
        return reward_batch(list(prompts), texts, list(answers))

    _reward.__name__ = "ring0_exact_match_reward"
    return _reward


@dataclass
class GRPOEngine:
    """Real TRL GRPOTrainer path (needs verge[l3,infer] + a GPU). Implements EvalProtocol."""

    settings: GRPOSettings
    train: list
    name: str = "L3-GRPO"

    def run_seed(self, seed: int, *, rounds: int, frozen_test) -> list[float]:
        # All heavy imports are lazy so this module is import-clean on CPU.
        import contextlib
        import gc
        import torch  # noqa: F401
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer

        from model import _build_prompt  # m1, unedited (locks the #### format)
        from verge.l3_reasoner._dist import local_rank, mprint
        from verge.ring0 import verify, Problem

        s = self.settings
        tok = AutoTokenizer.from_pretrained(s.base_model)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        def to_row(p):
            return {"prompt": _build_prompt(tok, p), "answer": p.answer}

        ds = Dataset.from_list([to_row(p) for p in self.train])

        on_cuda = torch.cuda.is_available()

        def _autocast():
            # bf16 autocast for ~2x faster generation; weights stay fp32 (no accuracy hit).
            return torch.autocast("cuda", dtype=torch.bfloat16) if on_cuda \
                else contextlib.nullcontext()

        @torch.no_grad()
        def measure(model) -> float:
            model.eval()
            n = len(frozen_test)
            bs = s.eval_batch_size
            # left-pad so decoder-only generation lines up across a padded batch
            prev_side, tok.padding_side = tok.padding_side, "left"
            correct = 0
            try:
                for start in range(0, n, bs):
                    batch = frozen_test[start:start + bs]
                    prompts = [_build_prompt(tok, p) for p in batch]
                    enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
                    with _autocast():
                        out = model.generate(**enc, max_new_tokens=s.max_completion_length,
                                             do_sample=False, pad_token_id=tok.pad_token_id)
                    gen = out[:, enc["input_ids"].shape[1]:]
                    for p, g in zip(batch, gen):
                        text = tok.decode(g, skip_special_tokens=True)
                        correct += int(verify(Problem(p.id, p.prompt, p.answer), text))
                    done = min(start + bs, n)
                    mprint(f"  [eval] {done}/{n}  running acc={correct / done:.3f}")
            finally:
                tok.padding_side = prev_side
            return correct / n

        load_dtype = torch.bfloat16 if (s.load_in_bf16 and on_cuda) else None
        model = AutoModelForCausalLM.from_pretrained(s.base_model, torch_dtype=load_dtype)
        if on_cuda:
            # Each rank owns one local GPU (cuda:LOCAL_RANK). Plain `python -m` → cuda:0.
            # Without this the pre-trainer baseline eval would pile every rank onto cuda:0;
            # TRL only places the model itself later, inside .train().
            torch.cuda.set_device(local_rank())
            model = model.to(f"cuda:{local_rank()}")
        mprint(f"[seed {seed}] round 0/{rounds} — baseline eval")
        curve = [measure(model)]  # round 0 baseline, before any GRPO

        trainer = None
        for _r in range(1, rounds + 1):
            mprint(f"[seed {seed}] round {_r}/{rounds} — GRPO {s.steps_per_round} steps")
            # Reclaim the prior round's trainer/vLLM AND the reserved cache that HF generation
            # balloons during eval — vLLM colocate checks FREE (not live) GPU memory at init,
            # so without this it sees ~8 GB free next to a 14 GB model and refuses to start.
            if trainer is not None:
                del trainer
            gc.collect()
            if on_cuda:
                torch.cuda.empty_cache()
            cfg = GRPOConfig(output_dir=f"runs/grpo_seed{seed}", seed=seed,
                             max_steps=s.steps_per_round, logging_steps=10,
                             save_strategy="no", report_to=[], bf16=on_cuda,
                             gradient_checkpointing=s.gradient_checkpointing,
                             gradient_checkpointing_kwargs={"use_reentrant": False},
                             **s.to_trl_kwargs())
            trainer = GRPOTrainer(model=model, reward_funcs=[make_trl_reward()],
                                  args=cfg, train_dataset=ds, processing_class=tok)
            trainer.train()
            model = trainer.model
            mprint(f"[seed {seed}] round {_r}/{rounds} — post-train eval")
            curve.append(measure(model))
        return curve


@dataclass
class MockGRPOEngine:
    """Free CPU validation of the GRPO loop SHAPE. Exercises the real Ring-0 reward over
    grouped samples, applies DAPO dynamic sampling, and steps a simulated policy. No GPU,
    no ML deps. Reproduces the bounded rise-then-plateau (the §2/§5 ceiling)."""

    cfg: object            # M1 Config (mock); cfg.model.samples_per_problem == K
    settings: GRPOSettings
    train: list
    name: str = "L3-GRPO-mock"

    def run_seed(self, seed: int, *, rounds: int, frozen_test) -> list[float]:
        from mock_model import MockModel

        from verge.ring0 import verify, Problem

        self.cfg.train.seed = seed
        model = MockModel(self.cfg)

        def measure() -> float:
            traces = model.single_shot_batch(frozen_test)
            return sum(verify(Problem(p.id, p.prompt, p.answer), t)
                       for p, t in zip(frozen_test, traces)) / len(frozen_test)

        curve = [measure()]  # round 0 baseline
        for _r in range(1, rounds + 1):
            groups = model.search_batch(self.train)          # K samples/problem (the group)
            kept = []
            for p, traces in zip(self.train, groups):
                rewards = reward_batch([p.prompt] * len(traces), traces,
                                       [p.answer] * len(traces))
                # DAPO dynamic sampling: skip zero-advantage groups (all same reward).
                if self.settings.dynamic_sampling and len(set(rewards)) == 1:
                    continue
                # Advantage-weighting toward verifier-confirmed traces (reward==1).
                kept += [(p, tr) for tr, rw in zip(traces, rewards) if rw == 1.0]
            model.finetune(kept)                              # online policy step (no reset)
            curve.append(measure())
        return curve


def build_grpo_mock_adapter(*, rounds: int = 4, k: int = 8, n_train: int = 400,
                            n_test: int = 300, output_dir: str = "runs/verge_grpo_mock"):
    """Assemble the free CPU GRPO-mock adapter + frozen test (validates the new loop)."""
    from config import Config

    from verge.eval.mock import make_mock_problems

    cfg = Config()
    cfg.use_mock = True
    cfg.output_dir = output_dir
    cfg.model.samples_per_problem = k
    cfg.data.train_size = min(cfg.data.train_size, n_train)
    cfg.data.test_size = min(cfg.data.test_size, n_test)
    train, test = make_mock_problems(cfg.data.train_size, cfg.data.test_size)
    settings = GRPOSettings(num_generations=k, eval_rounds=rounds)
    return MockGRPOEngine(cfg=cfg, settings=settings, train=train), test


def build_grpo_engine(*, base_model: str, train, k: int = 8, lr: float = 2e-6,
                      steps_per_round: int = 100, use_vllm: bool = True,
                      per_device_batch: int = 8, gradient_checkpointing: bool = False,
                      optim: str = "adamw_torch", load_in_bf16: bool = False,
                      vllm_gpu_memory_utilization: float = 0.3):
    """Assemble the real (GPU) GRPO engine. Caller supplies train problems + frozen test.
    `use_vllm=False` falls back to TRL's HF generation — slower but no vLLM-server setup,
    the safer first-run path. `gradient_checkpointing` trades compute for memory (needed to
    full-FT 7B). `optim='paged_adamw_8bit'` + `load_in_bf16=True` full-FT 7B on ONE 80GB GPU
    (needs bitsandbytes). With vLLM colocate on one GPU, `vllm_gpu_memory_utilization` is
    vLLM's share — keep it low (~0.3) so the policy + optimizer keep the rest. NOTE: under
    multi-GPU, `per_device_batch * num_processes` must be a multiple of `k`."""
    settings = GRPOSettings(base_model=base_model, num_generations=k, lr=lr,
                            steps_per_round=steps_per_round, use_vllm=use_vllm,
                            per_device_train_batch_size=per_device_batch,
                            gradient_checkpointing=gradient_checkpointing,
                            optim=optim, load_in_bf16=load_in_bf16,
                            vllm_gpu_memory_utilization=vllm_gpu_memory_utilization)
    return GRPOEngine(settings=settings, train=train)
