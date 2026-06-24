"""GSM8K loader into Problem, with a contamination guard. `datasets` imported lazily."""
from __future__ import annotations

import re

from verifiers import Problem


def _gsm8k_answer(raw: str) -> str:
    m = re.search(r"####\s*(.+)", raw)
    return (m.group(1) if m else raw).strip()


def load_problems(dcfg) -> tuple[list[Problem], list[Problem]]:
    from datasets import load_dataset  # lazy

    ds = load_dataset(dcfg.dataset_name, "main")
    def take(split, n, pre):
        return [Problem(id=f"{pre}{i}", prompt=r["question"], answer=_gsm8k_answer(r["answer"]))
                for i, r in enumerate(split.select(range(min(n, len(split)))))]
    train = take(ds["train"], dcfg.train_size, "tr")
    test = take(ds["test"], dcfg.test_size, "te")
    if dcfg.contamination_check:
        check_contamination(train, test)
    return train, test


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def check_contamination(train: list[Problem], test: list[Problem]) -> int:
    tr = {_norm(p.prompt) for p in train}
    overlap = [p.id for p in test if _norm(p.prompt) in tr]
    if overlap:
        raise RuntimeError(f"CONTAMINATION: {len(overlap)} test prompts in train (e.g. {overlap[:3]})")
    return 0
