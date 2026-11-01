"""PerfPilot CLI. GPU calls happen only inside command execution."""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any

import typer

from perfpilot.config import PerfPilotConfig, load_config

app = typer.Typer(name="perfpilot", help="Keep a vLLM deployment fast.")


def _load(config: Path) -> PerfPilotConfig:
    cfg = load_config(config)
    if not cfg.model:
        typer.echo("error: set 'model:' in perfpilot.yaml", err=True)
        raise typer.Exit(code=2)
    return cfg


def _workload_dict(cfg: PerfPilotConfig) -> dict[str, Any]:
    w = cfg.workload
    return {
        "dataset_name": w.dataset,
        "input_len": w.input_len,
        "output_len": w.output_len,
        "num_prompts": w.num_prompts,
        "max_concurrency": w.max_concurrency,
        "seed": w.seed,
    }


def _deployed_worker():
    import modal

    return modal.Cls.from_name("perfpilot-worker", "Worker")()


def _run_id(prefix: str) -> str:
    return f"{prefix}-{_dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"


def _make_orchestrator(cfg: PerfPilotConfig, run_id: str, run_dir: Path):
    """AgentOrchestrator from config, or None when no key (fallbacks)."""
    from perfpilot.agent.client import SparkClient, read_api_key
    from perfpilot.orchestrator import AgentOrchestrator

    key = read_api_key()
    if not key:
        typer.echo("no agent key: deterministic fallbacks")
        return None
    client = SparkClient(
        api_key=key, base_url=cfg.agent.base_url, model=cfg.agent.model
    )
    return AgentOrchestrator(
        run_id=run_id, events_path=run_dir / "agent-events.jsonl", client=client
    )


@app.command()
def bisect(
    config: Path = typer.Option(Path("perfpilot.yaml"), "--config"),
    good: str | None = typer.Option(None, "--good"),
    bad: str | None = typer.Option(None, "--bad"),
) -> None:
    """Bisect a commit range for the first slow commit (M2; agent in M3)."""
    from perfpilot.git_info import commits_between
    from perfpilot.modes import BisectParams, TrialOutcome
    from perfpilot.modes.driver import drive_bisect_live
    from perfpilot.orchestrator import make_extra_runs_fn, make_pick_fn

    cfg = _load(config)
    good_sha = good or cfg.bisect.good
    bad_sha = bad or cfg.bisect.bad
    if not good_sha or not bad_sha:
        typer.echo("error: need --good/--bad or bisect.good/bad in config", err=True)
        raise typer.Exit(code=2)
    wl = _workload_dict(cfg)
    worker = _deployed_worker()
    run_id = _run_id("bisect")
    commits = commits_between(cfg.bisect.repo, good_sha, bad_sha)
    typer.echo(f"range: {len(commits)} commits")

    def measure(sha: str, runs: int) -> TrialOutcome:
        result = worker.run_trial.remote(
            commit=sha,
            flags={},
            runs=runs,
            model=cfg.model,
            workload=wl,
            inject=None,
        )
        if result.get("status") != "ok":
            return TrialOutcome(
                skipped=True, reason=str(result.get("reason", result.get("status")))
            )
        return TrialOutcome(samples=[r["output_throughput"] for r in result["runs"]])

    pick = extra = None
    orch = _make_orchestrator(cfg, run_id, Path("runs") / run_id)
    if orch is not None:
        pick = make_pick_fn(orch, metric=cfg.metrics.primary)
        extra = make_extra_runs_fn(orch, max_runs=cfg.stats.max_runs)
    outcome = drive_bisect_live(
        commits=commits,
        measure=measure,
        run_id=run_id,
        params=BisectParams(
            min_runs=cfg.stats.min_runs,
            max_runs=cfg.stats.max_runs,
            confidence=cfg.stats.confidence,
        ),
        pick=pick,
        extra_runs=extra,
    )
    typer.echo(f"status={outcome['status']} culprit={outcome['culprit']}")
    typer.echo(f"report: {outcome['run_dir']}/report.html")


@app.command()
def tune(
    config: Path = typer.Option(Path("perfpilot.yaml"), "--config"),
    commit: str | None = typer.Option(None, "--commit"),
) -> None:
    """Tune serve flags on one commit (M4)."""
    from perfpilot.modes.driver import drive_tune_live
    from perfpilot.modes.tune import TuneParams, serve_command
    from perfpilot.orchestrator import make_propose_fn
    from perfpilot.worker.quality import accuracy, load_bank

    cfg = _load(config)
    base_commit = commit or cfg.tune.base_commit
    if not base_commit:
        typer.echo("error: need --commit or tune.base_commit", err=True)
        raise typer.Exit(code=2)
    wl = _workload_dict(cfg)
    worker = _deployed_worker()
    bank = load_bank()
    run_id = _run_id("tune")

    def measure(tune_config: dict[str, Any], runs: int):
        from perfpilot.modes.tune import TuneTrial

        result = worker.run_trial.remote(
            commit=base_commit,
            flags=tune_config,
            runs=runs,
            model=cfg.model,
            workload=wl,
            inject=None,
        )
        if result.get("status") != "ok":
            raise RuntimeError(f"trial failed: {result.get('reason')}")
        runs_out = result["runs"]
        p99 = sorted(r["p99_ttft_ms"] for r in runs_out)[-1]
        return TuneTrial(
            samples=[r["output_throughput"] for r in runs_out],
            p99_ttft_ms=p99,
        )

    def quality(tune_config: dict[str, Any]) -> float:
        texts = worker.answer_batch.remote(
            commit=base_commit,
            flags=tune_config,
            model=cfg.model,
            prompts=[q.prompt for q in bank],
        )
        by_prompt = dict(zip([q.prompt for q in bank], texts, strict=False))
        acc, _ = accuracy(lambda p: by_prompt[p], bank)
        return acc

    propose = None
    orch = _make_orchestrator(cfg, run_id, Path("runs") / run_id)
    if orch is not None:
        propose = make_propose_fn(orch, cfg.tune.knobs)
    outcome = drive_tune_live(
        base_commit=base_commit,
        knob_space=dict(cfg.tune.knobs),
        measure=measure,
        quality=quality,
        propose=propose,
        run_id=run_id,
        params=TuneParams(
            min_runs=cfg.stats.min_runs,
            max_runs=cfg.stats.max_runs,
            confidence=cfg.stats.confidence,
            max_trials=cfg.tune.max_trials,
            max_accuracy_drop_pts=cfg.tune.quality.max_accuracy_drop_pts,
        ),
    )
    typer.echo(f"status={outcome['status']} best={outcome['best']}")
    typer.echo(serve_command(cfg.model, outcome["best"]))
    typer.echo(f"report: {outcome['run_dir']}/report.html")


@app.command()
def report(run: Path = typer.Argument(..., help="runs/<run_id>")) -> None:
    """Render report.html from a run's events.jsonl (M5)."""
    from perfpilot.report.render import render_report

    events = Path(run) / "events.jsonl"
    if not events.exists():
        typer.echo(f"error: no events.jsonl in {run}", err=True)
        raise typer.Exit(code=2)
    html = render_report(events)
    out = Path(run) / "report.html"
    out.write_text(html)
    typer.echo(f"wrote {out}")


@app.command()
def watch(run: Path = typer.Argument(..., help="runs/<run_id>")) -> None:
    """Render the live (auto-refreshing) report (M5)."""
    from perfpilot.report.render import render_report

    events = Path(run) / "events.jsonl"
    if not events.exists():
        typer.echo(f"error: no events.jsonl in {run}", err=True)
        raise typer.Exit(code=2)
    html = render_report(events, live=True)
    out = Path(run) / "report.html"
    out.write_text(html)
    typer.echo(f"wrote {out} (refreshes every 5s; open in a browser)")


@app.command()
def eval(
    config: Path = typer.Option(Path("perfpilot.yaml"), "--config"),
    trials: int = typer.Option(2, "--trials"),
) -> None:
    """Seeded injected-regression eval: 2 trials, first+last culprits (M3)."""
    from perfpilot.git_info import wheeled_range
    from perfpilot.modes.driver import drive_eval_trial, modal_measure_factory

    cfg = _load(config)
    wl = _workload_dict(cfg)
    commits = wheeled_range(cfg.bisect.repo, 16)
    positions = [0, len(commits) - 1][: max(trials, 1)]
    results = []
    for i, pos in enumerate(positions):
        summary = drive_eval_trial(
            commits,
            culprit_index=pos,
            inject_kind="enforce-eager",
            make_measure=lambda inject_for, record: modal_measure_factory(
                "perfpilot-worker",
                "Worker",
                cfg.model,
                wl,
                inject_for,
                record,
            ),
            run_id=_run_id(f"eval-t{i}"),
        )
        results.append(summary)
        typer.echo(
            f"trial {i}: culprit={summary['culprit']} "
            f"expected={summary['expected']} correct={summary['correct']} "
            f"verified={summary['verified']} cost=${summary['cost_usd']:.2f}"
        )
    typer.echo(f"{sum(r['correct'] for r in results)}/{len(results)} correct")
