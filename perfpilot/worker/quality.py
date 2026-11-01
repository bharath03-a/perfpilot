"""Quality guard: 50 fixed short-answer Qs, greedy decoding, max 32 tokens.

Pure part (this module): bank loading, answer normalization, grading,
pass/fail vs max_accuracy_drop_pts. GPU part (serving answers) happens in
the worker at runtime via the vLLM OpenAI-compatible /v1/completions
endpoint; scores feed tune.TuneOutcome quality records.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, Field

BANK_PATH = Path(__file__).resolve().parent.parent / "evals" / "quality_bank.json"
MAX_NEW_TOKENS = 32


class QualityQuestion(BaseModel):
    id: str
    prompt: str
    answer: str


class QualityResult(BaseModel):
    accuracy_pct: float
    baseline_accuracy_pct: float
    drop_pts: float
    passed: bool
    n: int = 50
    wrong_ids: list[str] = Field(default_factory=list)


def load_bank(path: str | Path = BANK_PATH) -> list[QualityQuestion]:
    data = json.loads(Path(path).read_text())
    return [QualityQuestion(**q) for q in data]


def normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,;:!?\"'")


def grade(pred: str, expected: str) -> bool:
    """First-line greedy output vs expected answer, normalized."""
    first = pred.splitlines()[0] if pred.splitlines() else ""
    return normalize(first) == normalize(expected)


def accuracy(
    answer_fn: Callable[[str], str], bank: list[QualityQuestion]
) -> tuple[float, list[str]]:
    """answer_fn maps prompt -> greedy completion. Returns (pct, wrong_ids)."""
    wrong = [q.id for q in bank if not grade(answer_fn(q.prompt), q.answer)]
    pct = (len(bank) - len(wrong)) / len(bank) * 100.0 if bank else 0.0
    return pct, wrong


def check_quality(
    candidate_fn: Callable[[str], str],
    baseline_fn: Callable[[str], str],
    bank: list[QualityQuestion] | None = None,
    max_drop_pts: float = 2.0,
) -> QualityResult:
    """Pass iff candidate accuracy drops no more than max_drop_pts vs baseline."""
    bank = bank if bank is not None else load_bank()
    base_pct, _ = accuracy(baseline_fn, bank)
    cand_pct, wrong = accuracy(candidate_fn, bank)
    drop = base_pct - cand_pct
    return QualityResult(
        accuracy_pct=cand_pct,
        baseline_accuracy_pct=base_pct,
        drop_pts=drop,
        passed=drop <= max_drop_pts,
        n=len(bank),
        wrong_ids=wrong,
    )


def vllm_answer_fn(base_url: str, model: str) -> Callable[[str], str]:
    """Greedy completion_fn against a running vLLM server (GPU runtime)."""
    import urllib.request

    def answer(prompt: str) -> str:
        body = json.dumps(
            {
                "model": model,
                "prompt": prompt,
                "max_tokens": MAX_NEW_TOKENS,
                "temperature": 0.0,
            }
        ).encode()
        req = urllib.request.Request(
            base_url.rstrip("/") + "/v1/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120.0) as resp:
            data = json.load(resp)
        return str(data["choices"][0].get("text", ""))

    return answer
