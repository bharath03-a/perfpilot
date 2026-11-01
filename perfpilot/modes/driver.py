"""M2 driver: deterministic bisect trial with hidden injection.

Runs run_bisect with a measure callback (deployed Modal class in prod,
fakes in tests), writes the run folder + events.jsonl, and returns a
summary. The hidden culprit index NEVER enters events or the report.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from perfpilot.modes import BisectOutcome, BisectParams, TrialOutcome, run_bisect
from perfpilot.modes.tune import TuneOutcome, TuneParams, run_tuning
from perfpilot.orchestrator import EventLog
from perfpilot.report import render as render_mod
from perfpilot.report import run_dir as run_dir_mod

MeasureFn = Callable[[str, int], TrialOutcome]
InjectFn = Callable[[str], str | None]


def modal_measure_factory(
    app_name: str,
    cls_name: str,
    model: str,
    workload: dict[str, Any],
    inject_for: InjectFn,
    record: list[dict[str, Any]],
) -> MeasureFn:
    """Measure via the deployed Modal worker class (one warm container)."""
    import modal

    worker = modal.Cls.from_name(app_name, cls_name)()

    def measure(sha: str, runs: int) -> TrialOutcome:
        from perfpilot.evals.inject import to_worker_inject

        raw = inject_for(sha)
        inject = to_worker_inject(raw) if raw is not None else None
        result = worker.run_trial.remote(
            commit=sha,
            flags={},
            runs=runs,
            model=model,
            workload=workload,
            inject=inject,
        )
        record.append({"sha": sha, "runs": runs, "result": result})
        if result.get("status") == "skip":
            return TrialOutcome(skipped=True, reason=result.get("reason", ""))
        if result.get("status") != "ok":
            return TrialOutcome(
                skipped=True, reason=f"trial error: {result.get('reason', '')}"
            )
        return TrialOutcome(
            samples=[r["output_throughput"] for r in result.get("runs", [])]
        )

    return measure


def drive_eval_trial(
    commits: list[str],
    culprit_index: int,
    inject_kind: str,
    make_measure: Callable[[InjectFn, list[dict[str, Any]]], MeasureFn],
    run_id: str,
    runs_root: str = "runs",
    gpu_cost_per_hour: float = 2.10,
) -> dict[str, Any]:
    """One planted-culprit trial. Returns the summary; culprit stays hidden."""
    run_dir = run_dir_mod.write_run_skeleton(
        run_id, {"commits": commits}, runs_root=runs_root
    )
    events_path = run_dir / "events.jsonl"
    log = EventLog(run_id, events_path)
    log.emit(
        "run_start",
        mode="bisect-eval",
        n_commits=len(commits),
        gpu_cost_per_hour=gpu_cost_per_hour,
    )
    index_of = {sha: i for i, sha in enumerate(commits)}

    def inject_for(sha: str) -> str | None:
        return inject_kind if index_of[sha] >= culprit_index else None

    records: list[dict[str, Any]] = []
    raw_measure = make_measure(inject_for, records)

    def measure(sha: str, runs: int) -> TrialOutcome:
        # One immediate retry on ERROR (not skip): the first trial on a
        # freshly woken container flakes transiently; a persistent error
        # still fails loud on the second attempt.
        first = raw_measure(sha, runs)
        if first.skipped and first.reason.startswith("trial error"):
            log.emit(
                "trial_start",
                sha=sha,
                runs=runs,
                attempt="retry",
                reason=first.reason[:300],
            )
            return raw_measure(sha, runs)
        return first

    outcome: BisectOutcome = run_bisect(commits, measure, params=BisectParams())
    verdicts = []
    for sha in outcome.tested_order:
        label = str(outcome.labels.get(sha, "untested"))
        verdicts.append((index_of[sha], {"skipped": "skip"}.get(label, label)))
    for trial_n, rec in enumerate(records):
        res = rec["result"]
        run_dir_mod.write_trial_result(run_dir, rec["sha"], {}, trial_n, dict(res))
        log.emit(
            "trial_result",
            sha=rec["sha"],
            runs_requested=rec["runs"],
            runs=[r.get("output_throughput") for r in res.get("runs", [])],
            status=res.get("status"),
            reason=res.get("reason", ""),
            gpu_uuid=res.get("gpu_uuid"),
            gpu_seconds=res.get("gpu_seconds"),
        )
    log.emit(
        "culprit",
        culprit=outcome.culprit,
        culprit_range=outcome.culprit_range,
        verified=outcome.verified,
        steps=outcome.steps,
    )
    log.emit("run_end", status=outcome.status)
    html = render_mod.render_report(events_path)
    run_dir_mod.finalize_report(run_dir, html)

    uuids = {r["result"].get("gpu_uuid") for r in records}
    gpu_seconds = sum(r["result"].get("gpu_seconds", 0.0) for r in records)
    expected = commits[culprit_index]
    correct = outcome.culprit == expected or (
        bool(outcome.culprit_range) and expected in (outcome.culprit_range or [])
    )
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "status": outcome.status,
        "culprit": outcome.culprit,
        "expected": expected,
        "correct": bool(correct),
        "verified": outcome.verified,
        "steps": outcome.steps,
        "extra_runs_used": outcome.extra_runs_used,
        "verdicts": verdicts,
        "n_trials": len(records),
        "unique_gpus": len(uuids),
        "gpu_seconds": gpu_seconds,
        "cost_usd": gpu_seconds / 3600.0 * gpu_cost_per_hour,
    }


def drive_bisect_live(
    commits: list[str],
    measure: MeasureFn,
    run_id: str,
    params: BisectParams | None = None,
    pick=None,
    extra_runs=None,
    runs_root: str = "runs",
    gpu_cost_per_hour: float = 2.10,
) -> dict[str, Any]:
    """Real (non-eval) bisect: no injection, no hidden culprit."""
    from perfpilot.modes import run_bisect as _run_bisect

    run_dir = run_dir_mod.write_run_skeleton(
        run_id, {"commits": commits}, runs_root=runs_root
    )
    events_path = run_dir / "events.jsonl"
    log = EventLog(run_id, events_path)
    log.emit(
        "run_start",
        mode="bisect",
        n_commits=len(commits),
        gpu_cost_per_hour=gpu_cost_per_hour,
    )
    outcome: BisectOutcome = _run_bisect(
        commits,
        measure,
        params=params or BisectParams(),
        pick=pick,
        extra_runs=extra_runs,
    )
    log.emit(
        "culprit",
        culprit=outcome.culprit,
        culprit_range=outcome.culprit_range,
        verified=outcome.verified,
        steps=outcome.steps,
    )
    log.emit("run_end", status=outcome.status)
    html = render_mod.render_report(events_path)
    run_dir_mod.finalize_report(run_dir, html)
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "status": outcome.status,
        "culprit": outcome.culprit,
        "culprit_range": outcome.culprit_range,
        "verified": outcome.verified,
        "steps": outcome.steps,
    }


def drive_tune_live(
    base_commit: str,
    knob_space: dict[str, list[Any]],
    measure,
    quality,
    run_id: str,
    params=None,
    propose=None,
    runs_root: str = "runs",
    gpu_cost_per_hour: float = 2.10,
) -> dict[str, Any]:
    """Real TuneSmith run with events + report. Returns summary."""
    run_dir = run_dir_mod.write_run_skeleton(
        run_id,
        {"base_commit": base_commit, "knobs": knob_space},
        runs_root=runs_root,
    )
    events_path = run_dir / "events.jsonl"
    log = EventLog(run_id, events_path)
    log.emit(
        "run_start",
        mode="tune",
        base_commit=base_commit,
        gpu_cost_per_hour=gpu_cost_per_hour,
    )
    outcome: TuneOutcome = run_tuning(
        {},
        knob_space,
        measure,
        quality,
        params=params or TuneParams(),
        propose=propose,
    )
    for cand in outcome.candidates:
        log.emit(
            "tune_candidate",
            knob=cand.knob,
            value=cand.value,
            config=dict(cand.config),
            reason=cand.reason,
        )
        if cand.decision == "accepted":
            log.emit(
                "tune_accept",
                knob=cand.knob,
                value=cand.value,
                effect=cand.effect,
                lo=cand.lo,
                hi=cand.hi,
            )
    for qr in outcome.quality_checks:
        log.emit(
            "quality_check",
            accuracy=qr.accuracy,
            baseline_accuracy=qr.baseline_accuracy,
            drop_pts=qr.drop_pts,
            passed=qr.passed,
            reason=qr.reason,
        )
    log.emit("run_end", status="confirmed" if outcome.confirmed else "done")
    html = render_mod.render_report(events_path)
    run_dir_mod.finalize_report(run_dir, html)
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "status": "confirmed" if outcome.confirmed else "done",
        "best": dict(outcome.best_config),
        "uplift": outcome.uplift_effect,
        "uplift_interval": [outcome.uplift_lo, outcome.uplift_hi],
        "serve_command": outcome.serve_command,
        "trials_used": outcome.trials_used,
    }
