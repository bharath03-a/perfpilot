"""Bisect driver over fake commit ranges: culprit, skips, drift, ambiguity."""

from __future__ import annotations

import math
import random

import pytest

from perfpilot.modes.bisect import (
    BisectParams,
    CommitLabel,
    TrialOutcome,
    default_pick,
    drift_exceeds,
    new_bisect_state,
    open_range,
    run_bisect,
    validate_pick,
    verify_culprit,
)

COMMITS = [f"c{i:02d}" for i in range(16)]
GOOD_MEAN = 1000.0
BAD_MEAN = 850.0


def synth(mean: float, rel_sd: float, n: int, rng: random.Random) -> list[float]:
    return [rng.gauss(mean, mean * rel_sd) for _ in range(n)]


def make_measure(
    levels: dict[str, float],
    *,
    noise: float = 0.004,
    seed: int = 0,
    skips: frozenset[str] = frozenset(),
    noisy: dict[str, float] | None = None,
):
    calls: list[tuple[str, int]] = []
    counts: dict[str, int] = {}

    def measure(sha: str, runs: int) -> TrialOutcome:
        calls.append((sha, runs))
        if sha in skips:
            return TrialOutcome(skipped=True, reason="no wheel")
        counts[sha] = counts.get(sha, 0) + 1
        rng = random.Random(f"{seed}:{sha}:{counts[sha]}")
        sd = (noisy or {}).get(sha, noise)
        return TrialOutcome(samples=synth(levels[sha], sd, runs, rng))

    measure.calls = calls  # type: ignore[attr-defined]
    return measure


def levels_for(culprit_idx: int) -> dict[str, float]:
    return {
        sha: (BAD_MEAN if i >= culprit_idx else GOOD_MEAN)
        for i, sha in enumerate(COMMITS)
    }


@pytest.mark.parametrize("culprit_idx", [1, 9, 15])
def test_full_16_commit_range_finds_planted_culprit(culprit_idx: int):
    measure = make_measure(levels_for(culprit_idx))
    out = run_bisect(COMMITS, measure)
    assert out.confirmed, out.reason
    assert out.culprit == COMMITS[culprit_idx]
    assert out.culprit_range == [COMMITS[culprit_idx]]
    assert out.verified, out.verify_reason
    assert out.steps <= 7
    assert out.drift_checks == out.steps // 4
    if culprit_idx == 9:
        assert out.drift_checks >= 1


def test_skips_widen_to_culprit_range():
    skips = frozenset({"c07", "c08"})
    measure = make_measure(levels_for(9), skips=skips)
    out = run_bisect(COMMITS, measure)
    assert out.confirmed
    assert out.status == "culprit_range"
    assert out.culprit == "c09"
    assert set(skips) <= set(out.culprit_range)
    assert set(skips) <= set(out.skipped)
    assert not out.verified
    assert out.verify_reason == "parent skipped"
    assert out.verify_ref == "c08"


def test_drift_triggers_endpoint_remeasure():
    levels = levels_for(9)
    calls: dict[str, int] = {}
    drift_injected = False

    def measure(sha: str, runs: int) -> TrialOutcome:
        nonlocal drift_injected
        calls[sha] = calls.get(sha, 0) + 1
        rng = random.Random(f"7:{sha}:{calls[sha]}")
        mean = levels[sha]
        if runs == 2 and not drift_injected:
            drift_injected = True
            mean *= 1.05
        return TrialOutcome(samples=synth(mean, 0.004, runs, rng))

    out = run_bisect(COMMITS, measure)
    assert out.drift_checks >= 1
    assert out.endpoint_remeasures == 1
    assert out.confirmed
    assert out.culprit == "c09"
    assert out.verified


def test_no_regression_stops_honestly():
    levels = dict.fromkeys(COMMITS, GOOD_MEAN)
    out = run_bisect(COMMITS, make_measure(levels))
    assert not out.confirmed
    assert out.status == "no_regression"
    assert out.culprit is None
    assert not out.verified


def test_tiny_regression_under_3pct_stops():
    levels = {sha: GOOD_MEAN * 0.99 for sha in COMMITS}
    levels["c00"] = GOOD_MEAN
    out = run_bisect(COMMITS, make_measure(levels, noise=0.001))
    assert not out.confirmed
    assert out.status == "no_regression"


def test_noisy_midpoint_goes_ambiguous_at_cap():
    levels = levels_for(9)
    levels["c08"] = GOOD_MEAN * math.exp(math.log(BAD_MEAN / GOOD_MEAN) / 2)
    measure = make_measure(levels, noise=0.004, noisy={"c08": 0.12})
    out = run_bisect(COMMITS, measure, params=BisectParams(min_runs=3, max_runs=8))
    assert out.confirmed
    assert "c08" in out.ambiguous
    assert out.culprit == "c09"
    assert out.status == "culprit_range"


def test_bad_pick_proposal_falls_back_to_middle():
    measure = make_measure(levels_for(9))
    out = run_bisect(COMMITS, measure, pick=lambda state: "c99")
    assert out.fallbacks >= 1
    assert out.culprit == "c09"
    assert out.verified


def test_default_pick_is_middle_untested_unskipped():
    state = new_bisect_state(COMMITS)
    assert default_pick(state) == "c08"
    assert validate_pick(state, "c08")
    assert not validate_pick(state, "c00")
    assert not validate_pick(state, "c99")
    state.labels["c08"] = CommitLabel.SKIPPED
    assert default_pick(state) in set(open_range(state))
    assert default_pick(state) != "c08"


def test_verify_rejects_wrong_direction():
    rng = random.Random(11)
    good = synth(GOOD_MEAN, 0.004, 8, rng)
    bad = synth(BAD_MEAN, 0.004, 8, rng)
    full = math.log(BAD_MEAN / GOOD_MEAN)
    assert verify_culprit(bad, good, full, ref="c08").verified
    flipped = verify_culprit(good, bad, full, ref="c08")
    assert not flipped.verified
    assert "direction" in flipped.reason


def test_verify_rejects_insignificant_gap():
    rng = random.Random(12)
    same_a = synth(GOOD_MEAN, 0.004, 8, rng)
    same_b = synth(GOOD_MEAN, 0.004, 8, rng)
    result = verify_culprit(same_a, same_b, math.log(0.85), ref="c00")
    assert not result.verified


def test_drift_helper_threshold():
    stable = drift_exceeds([100.0, 101.0], [100.5, 100.8])
    assert not stable.exceeded
    moved = drift_exceeds([100.0, 101.0], [105.0, 106.0])
    assert moved.exceeded
    assert moved.drift != 0.0


def test_needs_two_commits():
    with pytest.raises(ValueError):
        run_bisect(["only"], make_measure({"only": 1.0}))


def test_endpoint_skip_cannot_confirm():
    measure = make_measure(levels_for(9), skips=frozenset({"c00"}))
    out = run_bisect(COMMITS, measure)
    assert not out.confirmed
    assert out.status == "no_regression"


def test_endpoint_error_is_loud_not_no_regression():
    def measure(sha: str, runs: int) -> TrialOutcome:
        if sha == "c00":
            return TrialOutcome(skipped=True, reason="trial error: boom")
        return make_measure(levels_for(9))(sha, runs)

    out = run_bisect(COMMITS, measure)
    assert out.status == "endpoint_error"
    assert "boom" in out.reason


def test_extra_runs_non_int_value_counts_fallback():
    levels = levels_for(9)
    levels["c08"] = GOOD_MEAN * math.exp(math.log(BAD_MEAN / GOOD_MEAN) / 2)
    measure = make_measure(levels, noise=0.004, noisy={"c08": 0.12})
    out = run_bisect(COMMITS, measure, extra_runs=lambda sha, runs, v: object())
    assert out.fallbacks >= 1
    assert any("extra_runs fallback" in entry for entry in out.pick_log)
    assert out.confirmed


def test_pick_log_records_rejections():
    measure = make_measure(levels_for(9))
    out = run_bisect(COMMITS, measure, pick=lambda state: "c99")
    assert out.fallbacks >= 1
    assert out.pick_log
    assert any("rejected" in entry for entry in out.pick_log)


def test_verify_measures_literal_parent_when_skipped():
    skips = frozenset({"c08"})
    measure = make_measure(levels_for(9), skips=skips)
    out = run_bisect(COMMITS, measure)
    assert out.confirmed
    assert out.status == "culprit_range"
    assert not out.verified
    assert out.verify_reason == "parent skipped"
    assert out.verify_ref == "c08"


def test_tested_order_records_every_visit_with_labels():
    skips = frozenset({"c08"})
    measure = make_measure(levels_for(9), skips=skips)
    out = run_bisect(COMMITS, measure)
    assert out.confirmed
    assert "c00" in out.tested_order
    assert "c15" in out.tested_order
    assert "c08" in out.tested_order
    assert out.culprit in out.tested_order
    assert out.labels["c08"] == "skipped"
    assert out.labels[out.culprit] == "bad"
    assert out.labels["c00"] == "good"


def test_drift_records_match_drift_checks():
    levels = levels_for(9)
    calls: dict[str, int] = {}
    drift_injected = False

    def measure(sha: str, runs: int) -> TrialOutcome:
        nonlocal drift_injected
        calls[sha] = calls.get(sha, 0) + 1
        rng = random.Random(f"7:{sha}:{calls[sha]}")
        mean = levels[sha]
        if runs == 2 and not drift_injected:
            drift_injected = True
            mean *= 1.05
        return TrialOutcome(samples=synth(mean, 0.004, runs, rng))

    out = run_bisect(COMMITS, measure)
    assert out.drift_checks >= 1
    assert len(out.drift_records) == out.drift_checks
    assert any(r.exceeded for r in out.drift_records)


def test_ambiguous_at_cap_in_timeline_but_not_in_steps():
    levels = levels_for(9)
    levels["c08"] = GOOD_MEAN * math.exp(math.log(BAD_MEAN / GOOD_MEAN) / 2)
    measure = make_measure(levels, noise=0.004, noisy={"c08": 0.12})
    out = run_bisect(COMMITS, measure, params=BisectParams(min_runs=3, max_runs=8))
    assert "c08" in out.ambiguous
    assert "c08" in out.tested_order
    assert out.labels.get("c08") == "ambiguous"


def test_inconclusive_spends_extra_runs_within_cap():
    levels = levels_for(9)
    levels["c08"] = GOOD_MEAN * math.exp(math.log(BAD_MEAN / GOOD_MEAN) / 2)
    measure = make_measure(levels, noise=0.004, noisy={"c08": 0.12})
    out = run_bisect(COMMITS, measure)
    assert out.extra_runs_used > 0
    total_granted = out.extra_runs_used
    assert total_granted <= len(measure.calls) * 3
    for _sha, runs in measure.calls:
        assert runs <= 8


def _m2_eval9_measure():
    """Replay of trial-0 evidence: baseline contaminated once, rest clean."""
    calls: list[str] = []

    def measure(sha: str, runs: int) -> TrialOutcome:
        calls.append(sha)
        rng = random.Random(f"eval9:{sha}:{calls.count(sha)}")
        if sha == COMMITS[0] and calls.count(sha) == 1:
            return TrialOutcome(samples=[3569.0, 3594.0, 1703.0])
        mean = 1180.0 if COMMITS.index(sha) >= 1 else 3570.0
        return TrialOutcome(samples=synth(mean, 0.004, runs, rng))

    measure.calls = calls  # type: ignore[attr-defined]
    return measure


def test_contaminated_baseline_remeasured_finds_culprit():
    out = run_bisect(COMMITS[:8], _m2_eval9_measure(), params=BisectParams())
    assert out.status == "culprit"
    assert out.culprit == COMMITS[1]
    assert out.verified, out.verify_reason


def test_contaminated_baseline_without_remeasure_fails():
    """The same evidence defeats the old behavior: threshold disabled."""
    out = run_bisect(
        COMMITS[:8],
        _m2_eval9_measure(),
        params=BisectParams(baseline_max_spread=10.0),
    )
    assert not (out.status == "culprit" and out.culprit == COMMITS[1])


def test_persistently_noisy_endpoints_retry_once_and_terminate():
    calls: list[str] = []

    def measure(sha: str, runs: int) -> TrialOutcome:
        calls.append(sha)
        rng = random.Random(f"noisy:{sha}:{calls.count(sha)}")
        return TrialOutcome(samples=synth(1000.0, 0.40, runs, rng))

    out = run_bisect(COMMITS[:8], measure, params=BisectParams())
    # Endpoint stage retries exactly once per endpoint, never loops; later
    # re-probes (drift/verify) are separate decisions.
    assert calls[:4] == [COMMITS[0], COMMITS[0], COMMITS[7], COMMITS[7]]
    assert any("endpoint re-measured for good" in line for line in out.pick_log)
    assert any("endpoint re-measured for bad" in line for line in out.pick_log)
    assert out.status in ("culprit", "culprit_range", "no_regression")
