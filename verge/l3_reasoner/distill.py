"""L3 distillation engine — the §2 control experiment (spec M1.6).

§2 caps SELF-training: the survivors that get distilled come from the model itself,
so the loop can only re-weight what the base already reaches (your §5 flat result,
and the field's "invisible leash", arXiv 2507.14843 / 2504.13837). It says NOTHING
against new information from a *stronger source*. This engine runs the identical
search -> Ring-0 verify -> train -> measure loop, swapping only the trace SOURCE:

  source="self"     the model's own samples              (leashed control; predict flat)
  source="teacher"  verified traces from a stronger      (predict: rises past the
                    teacher (e.g. open R1 distill data)    self-ceiling -> escape)

If, on the SAME frozen test, the self arm is flat and the teacher arm is positive,
you have isolated the variable: the loop compounds on *external* information, exactly
as the conservation law predicts. Honest bound: distillation escapes YOUR base's
leash but inherits the TEACHER's — it proves the machinery, it is not unbounded.

Engineering bonus: distillation is plain SFT (no vLLM colocate, no 8-bit-Adam knife
edge), so the real path sidesteps the single-GPU memory wall GRPO+vLLM hit.

Real-path heavy imports are lazy. MockDistillEngine validates the loop shape AND the
self-vs-teacher control free on CPU through the same §7 harness.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from verge.l3_reasoner.reward import reward_batch


@dataclass
class DistillSettings:
    """Distillation knobs. SFT, not RL — so no vLLM / no colocate memory wall."""

    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    source: str = "teacher"            # "self" (leashed control) | "teacher" (escape)
    teacher_traces: str = ""           # path to JSONL {"id","solution"} of teacher traces
    teacher_theta: float = 0.90        # MOCK only: the (frozen) teacher's skill ceiling
    lr: float = 1e-5                   # SFT lr (distillation tolerates the RFT default)
    k: int = 8                         # self-source: samples/problem to propose
    steps_per_round: int = 60
    per_device_train_batch_size: int = 8
    gradient_checkpointing: bool = True
    load_in_bf16: bool = True
    optim: str = "adamw_torch"         # "paged_adamw_8bit" to full-FT 7B on one 80GB GPU
    max_completion_length: int = 320
    eval_batch_size: int = 32
    eval_rounds: int = 3


def _load_teacher_traces(path: str, train) -> dict:
    """Load teacher traces from a JSONL of {"id": <problem id>, "solution": <text ####ans>}.
    Returns {problem_id: [solution, ...]}. Keep the format trivial so it is easy to prepare
    on the pod from open R1-distilled data — the only requirement is the `#### <answer>`
    final-answer format the Ring-0 verifier reads."""
    if not path:
        raise ValueError("source='teacher' needs --teacher <traces.jsonl>; see distill.py")
    by_id: dict[str, list[str]] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            by_id.setdefault(str(row["id"]), []).append(row["solution"])
    return by_id


@dataclass
class DistillEngine:
    """Real (GPU) distillation: SFT the student on Ring-0-verified traces from `source`.
    Mirrors the §5 recipe — accumulate verified traces, reset to base, SFT, measure —
    but the traces come from `source` (self = STaR/expert-iteration; teacher = escape)."""

    settings: DistillSettings
    train: list
    name: str = "L3-Distill"

    def run_seed(self, seed: int, *, rounds: int, frozen_test) -> list[float]:
        import contextlib
        import gc

        import torch  # noqa: F401
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                   DataCollatorForLanguageModeling, Trainer, TrainingArguments)

        from model import _build_prompt
        from verge.l3_reasoner._dist import local_rank, mprint
        from verge.ring0 import Problem, verify

        s = self.settings
        on_cuda = torch.cuda.is_available()
        dev = f"cuda:{local_rank()}" if on_cuda else "cpu"
        tok = AutoTokenizer.from_pretrained(s.base_model)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        teacher_traces = (_load_teacher_traces(s.teacher_traces, self.train)
                          if s.source == "teacher" else {})

        def load_base():
            m = AutoModelForCausalLM.from_pretrained(s.base_model)
            if s.load_in_bf16 and on_cuda:
                m = m.bfloat16()
            if on_cuda:
                torch.cuda.set_device(local_rank())
                m = m.to(dev)
            return m

        def _autocast():
            return torch.autocast("cuda", dtype=torch.bfloat16) if on_cuda \
                else contextlib.nullcontext()

        @torch.no_grad()
        def measure(model) -> float:
            model.eval()
            n = len(frozen_test)
            prev, tok.padding_side = tok.padding_side, "left"
            correct = 0
            try:
                for i in range(0, n, s.eval_batch_size):
                    batch = frozen_test[i:i + s.eval_batch_size]
                    enc = tok([_build_prompt(tok, p) for p in batch],
                              return_tensors="pt", padding=True).to(model.device)
                    with _autocast():
                        out = model.generate(**enc, max_new_tokens=s.max_completion_length,
                                             do_sample=False, pad_token_id=tok.pad_token_id)
                    for p, g in zip(batch, out[:, enc["input_ids"].shape[1]:]):
                        text = tok.decode(g, skip_special_tokens=True)
                        correct += int(verify(Problem(p.id, p.prompt, p.answer), text))
            finally:
                tok.padding_side = prev
            return correct / n

        @torch.no_grad()
        def self_propose(model) -> dict:
            """Sample k traces/problem from the current model (HF generate; no vLLM needed)."""
            model.eval()
            out: dict[str, list[str]] = {}
            prev, tok.padding_side = tok.padding_side, "left"
            try:
                for p in self.train:
                    enc = tok(_build_prompt(tok, p), return_tensors="pt").to(model.device)
                    with _autocast():
                        gen = model.generate(**enc, max_new_tokens=s.max_completion_length,
                                             do_sample=True, temperature=1.0, num_return_sequences=s.k,
                                             pad_token_id=tok.pad_token_id)
                    out[p.id] = [tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True)
                                 for g in gen]
            finally:
                tok.padding_side = prev
            return out

        def sft(model, pool):
            rows = [{"text": _build_prompt(tok, p) + t + tok.eos_token} for p, t in pool]
            ds = _hf_dataset(rows, tok, s.max_completion_length + 512)
            args = TrainingArguments(
                output_dir=f"runs/distill_seed{seed}", max_steps=s.steps_per_round,
                per_device_train_batch_size=s.per_device_train_batch_size, learning_rate=s.lr,
                bf16=on_cuda, gradient_checkpointing=s.gradient_checkpointing,
                gradient_checkpointing_kwargs={"use_reentrant": False}, optim=s.optim,
                logging_steps=10, save_strategy="no", report_to=[])
            collator = DataCollatorForLanguageModeling(tok, mlm=False)
            Trainer(model=model, args=args, train_dataset=ds, data_collator=collator).train()

        model = load_base()
        mprint(f"[seed {seed}] round 0/{rounds} — baseline eval (source={s.source})")
        curve = [measure(model)]
        pool: list = []  # accumulated Ring-0-verified (problem, trace)
        for _r in range(1, rounds + 1):
            mprint(f"[seed {seed}] round {_r}/{rounds} — collect ({s.source}) + Ring-0 filter")
            cand = teacher_traces if s.source == "teacher" else self_propose(model)
            for p in self.train:
                for t in cand.get(p.id, []):
                    if verify(Problem(p.id, p.prompt, p.answer), t):
                        pool.append((p, t))
            mprint(f"[seed {seed}] round {_r}/{rounds} — reset-to-base + SFT on {len(pool)} verified")
            del model
            gc.collect()
            if on_cuda:
                torch.cuda.empty_cache()
            model = load_base()              # reset-to-base (avoids the forgetting doom-loop, §5)
            if pool:
                sft(model, pool)
            curve.append(measure(model))
        return curve


def _hf_dataset(rows, tok, max_len):
    from datasets import Dataset

    def tok_fn(b):
        enc = tok(b["text"], truncation=True, max_length=max_len, padding=False)
        enc["labels"] = [ids[:] for ids in enc["input_ids"]]
        return enc

    return Dataset.from_list(rows).map(tok_fn, batched=True, remove_columns=["text"])


@dataclass
class MockDistillEngine:
    """Free CPU validation of the loop shape AND the self-vs-teacher control. Models the one
    dynamic that matters: a student trained on verified traces moves toward the *coverage
    ceiling of its source*. Self-source is leashed to its own reach (THETA0+EXPLORE → plateau);
    a stronger teacher pulls it toward teacher_theta (escape). Uses the real Ring-0 filter."""

    cfg: object
    settings: DistillSettings
    train: list
    name: str = "L3-Distill-mock"

    def run_seed(self, seed: int, *, rounds: int, frozen_test) -> list[float]:
        from mock_model import EXPLORE, THETA0, MockModel

        from verge.ring0 import Problem, verify

        self.cfg.train.seed = seed
        student = MockModel(self.cfg)
        student.reset_to_base()
        source = self.settings.source
        if source == "teacher":
            teacher = MockModel(self.cfg)
            teacher.theta = self.settings.teacher_theta   # frozen, stronger source
        else:
            teacher = student                              # leashed to the student itself
        # coverage ceiling the source's verified traces can teach toward:
        target = (THETA0 + EXPLORE) if source == "self" else self.settings.teacher_theta
        lr = 0.5

        def measure() -> float:
            traces = student.single_shot_batch(frozen_test)
            return sum(verify(Problem(p.id, p.prompt, p.answer), t)
                       for p, t in zip(frozen_test, traces)) / len(frozen_test)

        curve = [measure()]
        for _r in range(1, rounds + 1):
            groups = teacher.search_batch(self.train)          # source proposes K/problem
            verified = 0                                        # real Ring-0 filter
            for p, traces in zip(self.train, groups):
                rewards = reward_batch([p.prompt] * len(traces), traces,
                                       [p.answer] * len(traces))
                verified += int(any(rw == 1.0 for rw in rewards))
            if verified:                                        # learn only from confirmed traces
                student.theta = min(0.97, student.theta + lr * (target - student.theta))
            curve.append(measure())
        return curve


def build_distill_mock_adapter(*, rounds: int = 3, source: str = "teacher", k: int = 8,
                               n_train: int = 400, n_test: int = 300, teacher_theta: float = 0.90,
                               output_dir: str = "runs/verge_distill_mock"):
    """Assemble the free CPU distill-mock adapter + frozen test (validates the control)."""
    from config import Config

    from verge.eval.mock import make_mock_problems

    cfg = Config()
    cfg.use_mock = True
    cfg.output_dir = output_dir
    cfg.model.samples_per_problem = k
    cfg.data.train_size = min(cfg.data.train_size, n_train)
    cfg.data.test_size = min(cfg.data.test_size, n_test)
    train, test = make_mock_problems(cfg.data.train_size, cfg.data.test_size)
    settings = DistillSettings(source=source, k=k, teacher_theta=teacher_theta, eval_rounds=rounds)
    return MockDistillEngine(cfg=cfg, settings=settings, train=train), test


def build_distill_engine(*, base_model: str, train, source: str = "teacher",
                         teacher_traces: str = "", k: int = 8, steps_per_round: int = 60,
                         per_device_batch: int = 8, gradient_checkpointing: bool = True,
                         load_in_bf16: bool = True, optim: str = "adamw_torch"):
    """Assemble the real (GPU) distillation engine. `source='self'` is STaR/expert-iteration
    (leashed); `source='teacher'` needs `teacher_traces` (a JSONL of verified teacher traces)
    and is the proven escape. Plain SFT — no vLLM, no colocate memory wall."""
    settings = DistillSettings(base_model=base_model, source=source, teacher_traces=teacher_traces,
                               k=k, steps_per_round=steps_per_round,
                               per_device_train_batch_size=per_device_batch,
                               gradient_checkpointing=gradient_checkpointing,
                               load_in_bf16=load_in_bf16, optim=optim)
    return DistillEngine(settings=settings, train=train)
