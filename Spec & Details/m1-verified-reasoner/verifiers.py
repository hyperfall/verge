"""The trustworthy verifier: deterministic exact-match on the final answer.

Pure stdlib — testable anywhere. Robust extraction (####, \\boxed, 'answer is',
bolded number, last-number fallback) so a correct answer phrased oddly still scores.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


@dataclass
class Problem:
    id: str
    prompt: str
    answer: str           # ground-truth final answer


def normalize_number(s: str | None) -> str | None:
    if s is None:
        return None
    s = s.strip().replace(",", "").replace("$", "").replace("%", "").rstrip(".")
    m = _NUM.search(s)
    if not m:
        return None
    try:
        f = float(m.group(0).replace(",", ""))
    except ValueError:
        return None
    return str(int(f)) if f == int(f) else repr(f)


def extract_final_answer(trace: str | None) -> str | None:
    """Robust GSM8K answer extraction (priority order)."""
    if not trace:
        return None
    for pat in (r"####\s*([^\n]+)", r"\\boxed\{([^}]*)\}",
                r"(?:final answer|answer)\s*(?:is|:|=)\s*\$?\*{0,2}([-\d.,]+)",
                r"\*\*\s*\$?\s*([-\d][\d.,]*)"):
        m = re.findall(pat, trace, re.I)
        if m:
            n = normalize_number(m[-1])
            if n is not None:
                return n
    nums = _NUM.findall(trace)
    return normalize_number(nums[-1]) if nums else None


class ExactMatchVerifier:
    """V: (problem, trace) -> bool. The one trusted asset; never learned."""

    def verify(self, problem: Problem, trace: str) -> bool:
        pred = extract_final_answer(trace)
        return pred is not None and pred == normalize_number(problem.answer)
