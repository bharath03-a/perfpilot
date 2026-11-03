"""Offline harness: simulated GPU, scenario suite, swarm referee, scoreboard."""

from __future__ import annotations

from pathlib import Path

from perfpilot.bench.harness import run_scenario, run_suite
from perfpilot.bench.scenarios import SCENARIOS, Scenario
from perfpilot.bench.scoreboard import append_entry, load_entries, trend_rows
from perfpilot.bench.sim import SimGPU
from perfpilot.bench.swarm import Proposal, referee
from perfpilot.modes.bisect import new_bisect_state


def _scenario(name: str) -> Scenario:
    return next(s for s in SCENARIOS if s.name == name)


def test_sim_is_seeded_and_plants_drop() -> None:
    sc = _scenario("clean_strong")
    a, b = SimGPU(sc, seed=3), SimGPU(sc, seed=3)
    first = sc.commits[0]
    last = sc.commits[-1]
    assert a.measure(first, 3).samples == b.measure(first, 3).samples
    fast = sum(a.measure(first, 3).samples) / 3
    slow = sum(a.measure(last, 3).samples) / 3
    assert slow < fast * (1 - sc.drop / 2)
    assert a.runs_used == 9 and a.probes == 3 and a.cost_usd > 0


def test_referee_majority_beats_lone_proposal() -> None:
    state = new_bisect_state(list("abcdefg")).model_copy(update={"bad_idx": 6})
    props = [
        Proposal(role="bisector", sha="d", confidence="low"),
        Proposal(role="statistician", sha="d", confidence="low"),
        Proposal(role="code_reader", sha="b", confidence="low"),
    ]
    assert referee(state, props).sha == "d"


def test_referee_high_confidence_hint_wins_alone() -> None:
    state = new_bisect_state(list("abcdefg")).model_copy(update={"bad_idx": 6})
    props = [
        Proposal(role="bisector", sha="d", confidence="low"),
        Proposal(role="code_reader", sha="b", confidence="high"),
    ]
    assert referee(state, props).sha == "b"


def test_referee_drops_out_of_range_and_empty() -> None:
    state = new_bisect_state(list("abcdefg")).model_copy(update={"bad_idx": 6})
    assert referee(state, [Proposal(role="x", sha="zzz", confidence="high")]) is None
    assert referee(state, []) is None


def test_fallback_finds_clean_culprit() -> None:
    res = run_scenario(_scenario("clean_strong"), variant="fallback", seed=0)
    assert res.correct and res.wrong == 0


def test_no_regression_scenario_is_not_a_wrong_verdict() -> None:
    res = run_scenario(_scenario("no_regression"), variant="swarm", seed=0)
    assert res.correct and res.wrong == 0


def test_swarm_uses_hint_to_cut_probes() -> None:
    sc = _scenario("code_hint")
    seeds = range(4)
    fb = [run_scenario(sc, "fallback", s).probes for s in seeds]
    sw = [run_scenario(sc, "swarm", s).probes for s in seeds]
    assert sum(sw) < sum(fb)


def test_misleading_hint_never_causes_wrong_verdict() -> None:
    sc = _scenario("misleading_hint")
    for s in range(4):
        assert run_scenario(sc, "swarm", s).wrong == 0


def test_suite_summary_and_scoreboard_trend(tmp_path: Path) -> None:
    summary = run_suite(["fallback", "swarm"], seeds=2)
    assert {r.variant for r in summary.rows} == {"fallback", "swarm"}
    board = tmp_path / "scoreboard.jsonl"
    append_entry(board, summary, git_sha="aaa", mode="scripted")
    append_entry(board, summary, git_sha="bbb", mode="scripted")
    entries = load_entries(board)
    assert [e["git_sha"] for e in entries] == ["aaa", "bbb"]
    rows = trend_rows(entries, variant="swarm", mode="scripted")
    assert rows[-1]["delta_score"] == 0.0


def test_bench_cli_records_and_shows_trend(tmp_path: Path, monkeypatch) -> None:
    from typer.testing import CliRunner

    from perfpilot.cli import app

    monkeypatch.chdir(tmp_path)
    board = tmp_path / "board.jsonl"
    args = ["bench", "--seeds", "1", "--variants", "fallback,swarm"]
    runner = CliRunner()
    for _ in range(2):
        res = runner.invoke(app, [*args, "--record", "--board", str(board)])
        assert res.exit_code == 0, res.output
    assert "| swarm |" in res.output and "trend: swarm" in res.output
    assert len(load_entries(board)) == 2


def test_bench_cli_rejects_unknown_variant() -> None:
    from typer.testing import CliRunner

    from perfpilot.cli import app

    res = CliRunner().invoke(app, ["bench", "--variants", "hivemind"])
    assert res.exit_code == 2


def test_outlier_in_drift_recheck_does_not_abort_bisect() -> None:
    # seeds 1, 12, 17 aborted as no_regression: one collapsed run in the
    # drift re-measure blew up the interval and ended a correct bisect
    sc = _scenario("outlier")
    for s in (1, 12, 17):
        res = run_scenario(sc, "fallback", s)
        assert res.correct, (s, res.status)


def _stress(name: str) -> Scenario:
    from perfpilot.bench.scenarios import STRESS

    return next(s for s in STRESS if s.name == name)


def test_transient_worker_errors_are_retried_not_fatal() -> None:
    sc = _stress("flaky_worker")
    results = [run_scenario(sc, "fallback", s) for s in range(20)]
    assert not [r.seed for r in results if r.status == "endpoint_error"]
    assert sum(r.exact for r in results) >= 18


def test_retry_wrapper_retries_transient_only() -> None:
    from perfpilot.modes.bisect import TrialOutcome, with_transient_retries

    calls: list[str] = []
    replies = {
        "flaky": [
            TrialOutcome(skipped=True, reason="trial error: worker preempted"),
            TrialOutcome(samples=[1.0, 2.0]),
        ],
        "broken": [TrialOutcome(skipped=True, reason="build failed: no wheel")] * 3,
        "dead": [TrialOutcome(skipped=True, reason="trial error: oom")] * 5,
    }

    def measure(sha: str, runs: int) -> TrialOutcome:
        calls.append(sha)
        return replies[sha][calls.count(sha) - 1]

    wrapped = with_transient_retries(measure, retries=2)
    assert wrapped("flaky", 2).samples == [1.0, 2.0]
    assert wrapped("broken", 2).skipped and calls.count("broken") == 1
    assert wrapped("dead", 2).skipped and calls.count("dead") == 3


def test_swarm_ignores_non_discriminative_hints() -> None:
    sc = _stress("hint_spam")
    fb = sum(run_scenario(sc, "fallback", s).probes for s in range(10))
    sw = sum(run_scenario(sc, "swarm", s).probes for s in range(10))
    assert sw <= fb * 1.05


def test_endpoint_pools_and_trims_when_both_sets_contaminated() -> None:
    from perfpilot.modes.bisect import TrialOutcome, run_bisect

    # exact seed-4 evidence: every good-endpoint set has one collapsed run
    good_sets = iter([[3635.0, 1708.0, 3615.0], [1713.0, 3639.0, 3630.0]])

    def measure(sha: str, runs: int) -> TrialOutcome:
        if sha == "a":
            try:
                return TrialOutcome(samples=next(good_sets))
            except StopIteration:
                return TrialOutcome(samples=[3610.0, 3625.0, 3640.0][:runs])
        return TrialOutcome(samples=[1801.0, 1793.0, 1812.0][:runs])

    out = run_bisect(list("abcd"), measure)
    assert out.status == "culprit" and out.culprit == "b"
    assert any("pooled" in line for line in out.pick_log)


def test_outlier_scenario_robust_across_seeds() -> None:
    sc = _scenario("outlier")
    misses = [s for s in range(40) if not run_scenario(sc, "fallback", s).correct]
    assert misses == []
