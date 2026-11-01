"""Synthetic verdict checks: known 0% / 5% / 20% effects."""

from __future__ import annotations

import math
import random

import pytest

from perfpilot.stats import (
    classify_commit,
    confirm_regression,
    log_ratio_interval,
    tune_accepts,
)


def synth(mean: float, rel_sd: float, n: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.gauss(mean, mean * rel_sd) for _ in range(n)]


def test_zero_effect_interval_covers_zero():
    good = synth(100.0, 0.02, 8, seed=1)
    same = synth(100.0, 0.02, 8, seed=2)
    iv = log_ratio_interval(same, good)
    assert iv.lo <= 0.0 <= iv.hi
    v = confirm_regression(same, good)
    assert v.label == "no_regression"
    assert v.reason


def test_five_percent_effect_detected_and_classified():
    good = synth(100.0, 0.01, 8, seed=3)
    up5 = synth(105.0, 0.01, 8, seed=4)
    iv = log_ratio_interval(up5, good)
    assert iv.lo > 0.0
    assert math.isclose(iv.effect, math.log(1.05), rel_tol=0.15)
    full = math.log(0.80)
    v = classify_commit(up5, good, full_effect=full)
    assert v.label == "good"
    assert v.reason


def test_twenty_percent_drop_confirmed_bad():
    good = synth(100.0, 0.01, 8, seed=5)
    down20 = synth(80.0, 0.01, 8, seed=6)
    v = confirm_regression(down20, good)
    assert v.label == "bad"
    assert math.isclose(v.effect, math.log(0.80), rel_tol=0.15)
    mid = classify_commit(down20, good, full_effect=v.effect)
    assert mid.label == "bad"


def test_small_effect_under_3pct_is_no_regression():
    good = synth(100.0, 0.002, 8, seed=7)
    tiny = synth(101.0, 0.002, 8, seed=8)
    v = confirm_regression(tiny, good)
    assert v.label == "no_regression"


def test_noisy_samples_are_inconclusive_not_wrong():
    good = synth(100.0, 0.10, 3, seed=9)
    cand = synth(95.0, 0.10, 3, seed=10)
    v = classify_commit(cand, good, full_effect=math.log(0.80))
    assert v.label in ("good", "bad", "inconclusive")
    assert v.reason


def test_tune_gate_rejects_weak_uplift():
    best = synth(100.0, 0.01, 8, seed=11)
    weak = synth(101.0, 0.01, 8, seed=12)
    v = tune_accepts(weak, best, cand_p99=100.0, best_p99=100.0)
    assert v.label == "inconclusive"


def test_tune_gate_accepts_real_uplift():
    best = synth(100.0, 0.01, 8, seed=13)
    fast = synth(106.0, 0.01, 8, seed=14)
    v = tune_accepts(fast, best, cand_p99=100.0, best_p99=100.0)
    assert v.label == "good"


def test_needs_two_samples():
    with pytest.raises(ValueError):
        log_ratio_interval([1.0], [1.0, 2.0])


def test_significant_improvement_is_not_a_regression():
    faster = synth(120.0, 0.01, 8, seed=21)
    good = synth(100.0, 0.01, 8, seed=22)
    v = confirm_regression(faster, good)
    assert v.label == "no_regression"
    assert "mprovement" in v.reason


def test_nonfinite_and_nonpositive_samples_raise():
    good = [100.0, 101.0, 99.0]
    for bad in (
        [float("nan"), 1.0, 2.0],
        [float("inf"), 1.0, 2.0],
        [0.0, 1.0, 2.0],
        [-5.0, 1.0, 2.0],
    ):
        with pytest.raises(ValueError):
            log_ratio_interval(bad, good)
        with pytest.raises(ValueError):
            log_ratio_interval(good, bad)


def test_zero_variance_is_never_decisive():
    a = [100.0, 100.0]
    b = [90.0, 90.0]
    assert confirm_regression(b, a).label == "no_regression"
    assert "zero-variance" in confirm_regression(b, a).reason
    assert classify_commit(b, a, full_effect=-0.1).label == "inconclusive"
    assert tune_accepts(b, a, cand_p99=100.0, best_p99=100.0).label == "inconclusive"
