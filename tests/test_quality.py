"""Quality guard tests: bank, grading, pass/fail gate."""

from __future__ import annotations

from perfpilot.worker.quality import (
    check_quality,
    grade,
    load_bank,
    normalize,
)


def test_bank_has_50() -> None:
    bank = load_bank()
    assert len(bank) == 50
    assert all(q.prompt and q.answer for q in bank)


def test_normalize_and_grade() -> None:
    assert normalize("  Paris. ") == "paris"
    assert grade("Paris\nmore text", "paris")
    assert grade("  4 ", "4")
    assert not grade("London", "Paris")


def test_quality_gate_passes_and_fails() -> None:
    bank = load_bank()

    def perfect(prompt: str) -> str:
        return next(q.answer for q in bank if q.prompt == prompt)

    res = check_quality(perfect, perfect)
    assert res.passed and res.drop_pts == 0.0 and res.accuracy_pct == 100.0

    def sloppy(prompt: str) -> str:
        return "wrong"

    res2 = check_quality(sloppy, perfect)
    assert not res2.passed and res2.drop_pts > 2.0
    assert len(res2.wrong_ids) == 50


def test_small_drop_within_tolerance() -> None:
    bank = load_bank()
    answers = {q.prompt: q.answer for q in bank}

    def one_wrong(prompt: str) -> str:
        if prompt == bank[0].prompt:
            return "wrong"
        return answers[prompt]

    def perfect(prompt: str) -> str:
        return answers[prompt]

    res = check_quality(one_wrong, perfect)
    assert res.passed and res.drop_pts == 2.0
    assert res.wrong_ids == [bank[0].id]
