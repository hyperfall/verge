"""K-sample search backends (verge-engineering.md §3, upgrade 2).

Search is the one operator that genuinely expands the verifier-confirmable set (spec §2).
vLLM (paged-attention, continuous batching) is 5–20× HF `generate`, which is the lever
that lets you raise **K** and move to harder tasks. Both backends are lazy-imported, so
this module imports clean on the CPU/mock path.

For TRL GRPO the in-trainer generation is enabled with `use_vllm=True` (see `grpo.py`);
this `VLLMSearch` is the standalone K-sample search used by the expert-iteration engine
and for offline best-of-K proposal mining.
"""
from __future__ import annotations

from dataclasses import dataclass

from verge import _m1bridge

_m1bridge.ensure_on_path()

from verifiers import Problem  # noqa: E402

# The few-shot prompt that locks the '#### <number>' format — reused from M1's model.py
# (the fix that made the verifier read answers cleanly).
from model import _build_prompt  # noqa: E402  (m1, unedited)


@dataclass
class VLLMSearch:
    """vLLM-backed K-sample generation. The throughput lever for raising K."""

    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_tokens: int = 320
    gpu_memory_utilization: float = 0.9
    _llm: object = None
    _tok: object = None

    def _ensure(self):
        if self._llm is not None:
            return
        # lazy: needs `verge[infer]` (vllm) + `verge[l3]` (transformers) + a GPU
        from transformers import AutoTokenizer
        from vllm import LLM

        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._llm = LLM(model=self.model_name,
                        gpu_memory_utilization=self.gpu_memory_utilization)

    def search_batch(self, problems: list[Problem], *, k: int, temperature: float = 1.0,
                     top_p: float = 0.95) -> list[list[str]]:
        """Return k sampled completions per problem (the M1 `search_batch` contract)."""
        self._ensure()
        from vllm import SamplingParams

        prompts = [_build_prompt(self._tok, p) for p in problems]
        params = SamplingParams(n=k, temperature=temperature, top_p=top_p,
                                max_tokens=self.max_tokens)
        outs = self._llm.generate(prompts, params)
        return [[o.text for o in out.outputs] for out in outs]

    def single_shot_batch(self, problems: list[Problem]) -> list[str]:
        """Greedy single-shot (fixed decoding) for measurement."""
        self._ensure()
        from vllm import SamplingParams

        prompts = [_build_prompt(self._tok, p) for p in problems]
        params = SamplingParams(n=1, temperature=0.0, max_tokens=self.max_tokens)
        outs = self._llm.generate(prompts, params)
        return [out.outputs[0].text for out in outs]
