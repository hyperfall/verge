"""Datasets for the L3 headroom run (verge-engineering.md §3; spec §5.4).

GSM8K is saturated for a small base (the §5 flat result). The path to a *positive* slope
is **headroom**: a harder task (MATH), a stronger base, or more search. This module loads
either into the shared `Problem` type, reusing the M1 contamination guard verbatim.

`datasets` is imported lazily — the mock/test path needs none of this.

**Verifier-reach caveat (honest).** The built EXACT rung (`ExactMatchVerifier`) is
numeric/arithmetic. GSM8K answers and the numeric slice of MATH verify cleanly with it.
General MATH answers (fractions, surds, symbolic) need a **math-equivalence rung**
(sympy) — a typed stub in the verification ladder (`MathEquivalenceRung`), NOT a silent
modification of the EXACT verifier. Until that rung is built, run the headroom experiment
on GSM8K (larger base + higher K, same trusted verifier) or the numeric MATH slice.
"""
from __future__ import annotations

import re

from verge import _m1bridge

_m1bridge.ensure_on_path()

from verifiers import Problem  # noqa: E402

_BOXED = re.compile(r"\\boxed\{")


def _gsm8k_answer(raw: str) -> str:
    m = re.search(r"####\s*(.+)", raw)
    return (m.group(1) if m else raw).strip()


def _extract_boxed(solution: str) -> str | None:
    """Extract the content of the LAST \\boxed{...}, balancing nested braces."""
    idx = None
    for m in _BOXED.finditer(solution):
        idx = m.end()
    if idx is None:
        return None
    depth, out = 1, []
    for ch in solution[idx:]:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
    return "".join(out).strip()


def load_gsm8k(train_size: int = 1000, test_size: int = 500, contamination_check: bool = True):
    from datasets import load_dataset  # lazy

    ds = load_dataset("openai/gsm8k", "main")

    def take(split, n, pre):
        return [Problem(id=f"{pre}{i}", prompt=r["question"], answer=_gsm8k_answer(r["answer"]))
                for i, r in enumerate(split.select(range(min(n, len(split)))))]

    train = take(ds["train"], train_size, "tr")
    test = take(ds["test"], test_size, "te")
    if contamination_check:
        _check(train, test)
    return train, test


def load_math(train_size: int = 1000, test_size: int = 500, *,
              numeric_only: bool = True, contamination_check: bool = True):
    """Load MATH (`hendrycks/competition_math`) into Problems with boxed answers.

    `numeric_only=True` keeps only problems whose boxed answer the EXACT verifier can
    check today (so the existing Ring-0 verifier stays the only trusted signal). Set it
    False once `MathEquivalenceRung` (sympy) is built.
    """
    from datasets import load_dataset  # lazy

    from verge.ring0 import normalize_number  # numeric-checkability probe

    ds = load_dataset("hendrycks/competition_math")

    def take(split, n, pre):
        out = []
        for i, r in enumerate(split):
            ans = _extract_boxed(r.get("solution", "")) or ""
            if numeric_only and normalize_number(ans) is None:
                continue
            out.append(Problem(id=f"{pre}{len(out)}", prompt=r["problem"], answer=ans))
            if len(out) >= n:
                break
        return out

    train = take(ds["train"], train_size, "tr")
    test = take(ds["test"], test_size, "te")
    if contamination_check:
        _check(train, test)
    return train, test


def _check(train, test) -> None:
    """Reuse the M1 contamination guard verbatim (one definition of the check)."""
    from data import check_contamination  # m1, unedited

    check_contamination(train, test)


def load_dataset_problems(name: str, train_size: int, test_size: int):
    name = name.lower()
    if name == "gsm8k":
        return load_gsm8k(train_size, test_size)
    if name == "math":
        return load_math(train_size, test_size)
    raise ValueError(f"unknown dataset {name!r} (use 'gsm8k' or 'math')")
