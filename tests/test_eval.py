"""Eval + injection unit tests. Pure: no GPU, no network."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from perfpilot.evals.inject import (
    OVERLAY_FLAGS,
    WORKER_INJECTS,
    Injection,
    apply_injection,
    is_injected,
    overlay_flags,
    should_explain,
    to_worker_inject,
    visible_flags,
)
from perfpilot.evals.run_eval import (
    FALLBACK_GPU_COST_PER_HOUR,
    EvalConfig,
    TrialOutcome,
    count_wrong_verdicts,
    format_eval_table,
    plan_trials,
    result_key,
    seeded_culprit_positions,
    summarize_trials,
    venv_key,
)
from perfpilot.report.run_dir import append_event, write_run_skeleton


def test_overlay_before_culprit_is_empty():
    inj = Injection(kind="enforce_eager", culprit_index=5)
    assert apply_injection(["--seed", "0"], 4, inj) == ["--seed", "0"]
    assert not is_injected(4, 5)


def test_overlay_at_and_after_culprit():
    inj = Injection(kind="enforce_eager", culprit_index=5)
    assert apply_injection([], 5, inj) == ["--enforce-eager"]
    assert apply_injection([], 15, inj) == ["--enforce-eager"]
    assert is_injected(5, 5)


def test_gpu_mem_overlay_flags():
    assert overlay_flags("gpu_mem_05") == ["--gpu-memory-utilization", "0.5"]
    inj = Injection(kind="gpu_mem_05", culprit_index=0)
    assert apply_injection([], 0, inj) == ["--gpu-memory-utilization", "0.5"]


def test_agent_view_strips_secret_overlay():
    server = ["--seed", "0", "--enforce-eager"]
    assert visible_flags(server, "enforce_eager") == ["--seed", "0"]
    server2 = ["--gpu-memory-utilization", "0.5", "--seed", "0"]
    assert visible_flags(server2, "gpu_mem_05") == ["--seed", "0"]


def test_eval_skips_explanation():
    assert should_explain(is_eval=True) is False
    assert should_explain(is_eval=False) is True


def test_seeded_positions_include_low_edge_and_last():
    pos = seeded_culprit_positions(n_trials=2, range_size=16, seed=0)
    assert pos == [1, 15]
    pos5 = seeded_culprit_positions(n_trials=5, range_size=16, seed=0)
    assert 1 in pos5 and 15 in pos5 and len(pos5) == 5
    assert len(set(pos5)) == 5  # edges never duplicated by interior draws


def test_plan_trials_two_trials_two_kinds():
    specs = plan_trials(EvalConfig())
    assert len(specs) == 2
    assert [s.culprit_index for s in specs] == [1, 15]
    assert specs[0].kind != specs[1].kind


def test_venvs_reused_results_never_reused():
    assert venv_key("abc") == venv_key("abc")
    assert result_key("abc", "h", "enforce_eager", 0) != result_key(
        "abc", "h", "enforce_eager", 1
    )
    assert result_key("abc", "h", "enforce_eager", 0) != result_key(
        "abc", "h", "gpu_mem_05", 0
    )


def test_count_wrong_verdicts():
    assert count_wrong_verdicts([(2, "bad"), (7, "good")], 5) == 2
    assert count_wrong_verdicts([(2, "good"), (7, "bad")], 5) == 0
    assert (
        count_wrong_verdicts([(2, "inconclusive"), (3, "skip"), (4, "ambiguous")], 5)
        == 0
    )


def test_summarize_two_for_two_zero_wrong():
    outcomes = [
        TrialOutcome(
            trial_id=0, culprit_index=0, found_index=0, steps=4, gpu_seconds=3600.0
        ),
        TrialOutcome(
            trial_id=1,
            culprit_index=15,
            found_index=15,
            steps=5,
            inconclusive_steps=1,
            extra_runs=2,
            gpu_seconds=1800.0,
        ),
    ]
    s = summarize_trials(outcomes)
    assert (s.correct, s.total) == (2, 2)
    assert s.wrong_verdicts == 0
    assert s.expected_steps == math.log2(16) == 4.0
    assert s.steps == [4, 5]
    assert s.inconclusive_steps == 1 and s.extra_runs == 2
    assert s.gpu_minutes == [60.0, 30.0]
    assert s.cost_usd == pytest.approx([2.10, 1.05])
    table = format_eval_table(s)
    assert "2/2" in table and "| Wrong verdicts | 0 |" in table


def test_eval_cost_defaults_to_measured_rate():
    from perfpilot.config import PerfPilotConfig

    assert FALLBACK_GPU_COST_PER_HOUR == 2.10
    assert EvalConfig().gpu_cost_per_hour == PerfPilotConfig().gpu_cost_per_hour


def test_default_pairing_is_strong_first_mild_last():
    specs = plan_trials(EvalConfig())
    assert specs[0].kind == "enforce_eager" and specs[0].culprit_index == 1
    assert specs[1].kind == "gpu_mem_05" and specs[1].culprit_index == 15


def test_to_worker_inject_maps_both_kinds():
    assert to_worker_inject("enforce_eager") == "enforce-eager"
    assert to_worker_inject("gpu_mem_05") == "gpu-memory-utilization-0.5"
    with pytest.raises(ValueError, match="unknown injection kind"):
        to_worker_inject("turbo-boost")  # type: ignore[arg-type]


def test_worker_inject_is_bijection_with_wire_contract():
    """Either side adding a kind must break the suite, not a GPU trial."""
    from perfpilot.worker.modal_app import INJECT_FLAGS

    assert set(WORKER_INJECTS.values()) == set(INJECT_FLAGS)
    assert set(WORKER_INJECTS) == {"enforce_eager", "gpu_mem_05"}


def test_visible_flags_removes_every_occurrence():
    doubled = ["--seed", "0", "--enforce-eager", "--x", "1", "--enforce-eager"]
    assert visible_flags(doubled, "enforce_eager") == ["--seed", "0", "--x", "1"]
    doubled_mem = (
        ["--gpu-memory-utilization", "0.5"]
        + ["--seed", "0"]
        + ["--gpu-memory-utilization", "0.5"]
    )
    assert visible_flags(doubled_mem, "gpu_mem_05") == ["--seed", "0"]
    assert visible_flags(["--seed", "0"], "enforce_eager") == ["--seed", "0"]


@pytest.mark.parametrize("kind", ["enforce_eager", "gpu_mem_05"])
def test_events_log_never_contains_overlay_or_culprit(kind, tmp_path: Path):
    inj = Injection(kind=kind, culprit_index=3)
    run_dir = write_run_skeleton("eval0", {"model": "m"}, runs_root=tmp_path)
    for n, idx in enumerate((2, 3)):
        server = apply_injection(["--seed", "0"], idx, inj)
        append_event(
            run_dir,
            {
                "ts": f"t{n}",
                "run_id": "eval0",
                "type": "trial_start",
                "sha": f"c{idx}",
                "flags": visible_flags(server, kind),
            },
        )
    log = (run_dir / "events.jsonl").read_text()
    assert "culprit_index" not in log
    for flag in OVERLAY_FLAGS[kind]:
        assert flag not in log
    logged = [json.loads(line)["flags"] for line in log.splitlines()]
    assert logged == [["--seed", "0"], ["--seed", "0"]]
    assert all(
        "enforce-eager" not in " ".join(map(str, f))
        and "gpu-memory-utilization" not in " ".join(map(str, f))
        for f in logged
    )


def _fake_transport(drops: dict[str, float], seed_salt: str = ""):
    """Fake make_measure: per-kind drops, 1% noise, ok status, 1 gpu-sec."""
    import random

    from perfpilot.modes import TrialOutcome as DriverOutcome

    def make_measure(inject_for, record):
        def measure(sha: str, runs: int) -> DriverOutcome:
            rng = random.Random(hash((seed_salt, sha, runs)) & 0xFFFFFFFF)
            kind = inject_for(sha)
            if kind is not None and kind not in drops:
                raise AssertionError(f"fake transport got unknown kind {kind!r}")
            mu = 3800.0 * (1.0 - drops.get(kind or "", 0.0))
            result = {
                "status": "ok",
                "gpu_uuid": "GPU-fake",
                "gpu_seconds": 1.0,
                "runs": [
                    {"output_throughput": rng.gauss(mu, mu * 0.01)} for _ in range(runs)
                ],
            }
            record.append({"sha": sha, "runs": runs, "result": result})
            return DriverOutcome(
                samples=[r["output_throughput"] for r in result["runs"]]
            )

        return measure

    return make_measure


def _to_trial_outcome(spec, summary, commits) -> TrialOutcome:
    from perfpilot.evals.run_eval import TrialOutcome as EvalOutcome

    found = commits.index(summary["culprit"]) if summary["culprit"] else None
    verdicts = summary["verdicts"]
    return EvalOutcome(
        trial_id=spec.trial_id,
        culprit_index=spec.culprit_index,
        found_index=found,
        found_range=[commits.index(s) for s in summary.get("culprit_range", [])],
        wrong_verdicts=count_wrong_verdicts(verdicts, spec.culprit_index),
        steps=summary["steps"],
        inconclusive_steps=sum(1 for _, v in verdicts if v == "ambiguous"),
        extra_runs=summary["extra_runs_used"],
        gpu_seconds=summary["gpu_seconds"],
    )


def test_eval_protocol_two_trials_fake_transport(tmp_path: Path) -> None:
    """E2E protocol: plan -> drive both spec'd trials -> 2/2, 0 wrong.

    Trial 0 plants the strong overlay at the low edge (index 1), trial 1
    the mild overlay at the last commit. No GPU, no network, no key.
    """
    from perfpilot.modes.driver import drive_eval_trial

    commits = [f"e{i:02d}{'0' * 36}" for i in range(16)]
    specs = plan_trials(EvalConfig(commits=commits, seed=0))
    assert [(s.culprit_index, s.kind) for s in specs] == [
        (1, "enforce_eager"),
        (15, "gpu_mem_05"),
    ]
    drops = {"enforce_eager": 0.30, "gpu_mem_05": 0.10}
    outcomes: list[TrialOutcome] = []
    for spec in specs:
        summary = drive_eval_trial(
            commits,
            culprit_index=spec.culprit_index,
            inject_kind=spec.kind,
            make_measure=_fake_transport(drops),
            run_id=f"proto-{spec.trial_id}",
            runs_root=str(tmp_path),
        )
        assert summary["correct"] and summary["verified"], spec
        assert summary["culprit"] == commits[spec.culprit_index]
        outcomes.append(_to_trial_outcome(spec, summary, commits))
        events = (tmp_path / f"proto-{spec.trial_id}" / "events.jsonl").read_text()
        assert "culprit_index" not in events and spec.kind not in events
    table = format_eval_table(summarize_trials(outcomes, 16, 2.10))
    assert "| Correct culprits | 2/2 |" in table
    assert "| Wrong verdicts | 0 |" in table


def test_eval_protocol_first_commit_reports_no_regression(tmp_path: Path) -> None:
    """Boundary lock-in: culprit at index 0 is honestly unfindable.

    Every commit at/after the culprit is slow, so the range contains no
    clean baseline and no transition. The only honest verdict is
    ``no_regression`` -- the driver must not fabricate a culprit. Only an
    external (pre-range) baseline could distinguish this case, which the
    driver does not have. Protocol decision: positions are [1, 15] (index 1
    is the lowest detectable position); this test locks the boundary
    behavior in case a range ever starts at a culprit.
    """
    from perfpilot.modes.driver import drive_eval_trial

    commits = [f"e{i:02d}{'0' * 36}" for i in range(16)]
    summary = drive_eval_trial(
        commits,
        culprit_index=0,
        inject_kind="enforce_eager",
        make_measure=_fake_transport({"enforce_eager": 0.30}),
        run_id="proto-first",
        runs_root=str(tmp_path),
    )
    assert summary["status"] == "no_regression"
    assert summary["culprit"] is None and not summary["correct"]
