"""HF model wrapper: few-shot prompt, batched generation, train-from-base, SFT.

Requires torch + transformers. The few-shot prompt locks the '#### <number>' format
so the verifier reads answers cleanly (the bug that wrecked the first attempts).
"""
from __future__ import annotations

from verifiers import Problem

_SYSTEM = ("Solve the math problem. Think step by step, then on the FINAL line write "
           "exactly '#### <number>' with only the numeric answer and nothing after it.")

_FEWSHOT = [
    ("Natalia sold clips to 48 friends in April, then sold half as many in May. "
     "How many clips did she sell altogether?",
     "April: 48. May: 48 / 2 = 24. Total: 48 + 24 = 72.\n#### 72"),
    ("Weng earns $12 an hour for babysitting. Yesterday she babysat 50 minutes. How much did she earn?",
     "50 minutes is 50/60 hour. Pay: 12 * 50/60 = 10.\n#### 10"),
]


def _build_prompt(tok, problem: Problem) -> str:
    msgs = [{"role": "system", "content": _SYSTEM}]
    for q, a in _FEWSHOT:
        msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": problem.prompt})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def _chunks(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


class Model:
    def __init__(self, cfg, device=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.cfg, self.torch = cfg, torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(cfg.model.name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model.name, torch_dtype=dtype).to(self.device)
        self.model.eval()
        self._base = {k: v.detach().to("cpu").clone() for k, v in self.model.state_dict().items()}

    def reset_to_base(self):
        with self.torch.no_grad():
            sd = self.model.state_dict()
            for k, v in sd.items():
                v.copy_(self._base[k].to(device=v.device, dtype=v.dtype))
        self.model.eval()

    def _gen(self, problems, *, n, temperature):
        torch = self.torch
        self.tok.padding_side = "left"
        out_all = []
        for chunk in _chunks(problems, self.cfg.model.gen_batch_problems):
            prompts = [_build_prompt(self.tok, p) for p in chunk]
            enc = self.tok(prompts, return_tensors="pt", padding=True, truncation=True,
                           max_length=1024).to(self.device)
            do_sample = temperature > 0.0
            with torch.no_grad():
                out = self.model.generate(
                    **enc, max_new_tokens=self.cfg.model.max_new_tokens, do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    top_p=self.cfg.model.gen_top_p if do_sample else None,
                    num_return_sequences=n, pad_token_id=self.tok.pad_token_id)
            gen = out[:, enc["input_ids"].shape[1]:]
            text = self.tok.batch_decode(gen, skip_special_tokens=True)
            for i in range(len(chunk)):
                out_all.append(text[i * n:(i + 1) * n])
        return out_all

    def single_shot_batch(self, problems):
        return [g[0] for g in self._gen(problems, n=1, temperature=self.cfg.model.eval_temperature)]

    def search_batch(self, problems):
        return self._gen(problems, n=self.cfg.model.samples_per_problem,
                         temperature=self.cfg.model.gen_temperature)

    def finetune(self, pairs):
        if not pairs:
            return
        torch = self.torch
        self.model.train()
        self.tok.padding_side = "right"
        opt = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad],
                                lr=self.cfg.train.lr)
        for _ in range(self.cfg.train.epochs_per_round):
            for batch in _chunks(pairs, self.cfg.train.batch_size):
                texts, plens = [], []
                for prob, trace in batch:
                    p = _build_prompt(self.tok, prob)
                    texts.append(p + trace + self.tok.eos_token)
                    plens.append(len(self.tok(p)["input_ids"]))
                enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True,
                               max_length=1024).to(self.device)
                labels = enc["input_ids"].clone()
                labels[enc["attention_mask"] == 0] = -100
                for row, pl in enumerate(plens):
                    labels[row, :pl] = -100
                loss = self.model(**enc, labels=labels).loss
                loss.backward()
                opt.step()
                opt.zero_grad()
        self.model.eval()
