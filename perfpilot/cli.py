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
    """Bisect a commit range for the first slow commit."""
    from perfpilot.git_info import commit_infos_between
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
    infos = commit_infos_between(cfg.bisect.repo, good_sha, bad_sha)
    commits = [c.sha for c in infos]
    messages = {c.sha: c.message for c in infos}
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
        reason = str(result.get("reason", result.get("status")))
        if result.get("status") == "skip":
            return TrialOutcome(skipped=True, reason=reason)
        if result.get("status") != "ok":
            # prefix marks it transient so run_bisect retries it
            return TrialOutcome(skipped=True, reason=f"trial error: {reason}")
        valid = [
            r
            for r in result.get("runs", [])
            if isinstance(r, dict) and not r.get("failed_requests", 0)
        ]
        if not valid and result.get("runs"):
            return TrialOutcome(skipped=True, reason="all runs had failed requests")
        return TrialOutcome(samples=[r["output_throughput"] for r in valid])

    pick = extra = explain_fn = draft_fn = None
    orch = _make_orchestrator(cfg, run_id, Path("runs") / run_id)
    if orch is not None:
        pick = make_pick_fn(orch, metric=cfg.metrics.primary, messages=messages)
        extra = make_extra_runs_fn(orch, max_runs=cfg.stats.max_runs)

        def draft_fn(sha, patch):
            decision = orch.draft_fix(sha, patch)
            if not decision.accepted or decision.value is None:
                return None
            return {"patch": decision.value.patch}

        def explain_fn(sha, message, paths, stats):
            decision = orch.explain_culprit(
                sha,
                message,
                paths,
                metric=cfg.metrics.primary,
                good_mean=stats["good_mean"],
                bad_mean=stats["bad_mean"],
                effect=stats["effect"],
            )
            if not decision.accepted:
                return None
            return decision.value.model_dump()

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
        explain_fn=explain_fn,
        draft_fn=draft_fn,
        repo=cfg.bisect.repo,
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

        last_reason = "unknown"
        for _ in (1, 2):  # one immediate retry on transient failure
            result = worker.run_trial.remote(
                commit=base_commit,
                flags=tune_config,
                runs=runs,
                model=cfg.model,
                workload=wl,
                inject=None,
            )
            if result.get("status") == "ok":
                break
            last_reason = str(result.get("reason", result.get("status")))
        if result.get("status") != "ok":
            raise RuntimeError(f"trial failed twice: {last_reason}")
        runs_out = result["runs"]
        valid = [r for r in runs_out if not r.get("failed_requests", 0)]
        if not valid:
            raise RuntimeError("trial failed: all runs had failed requests")
        p99 = sorted(r["p99_ttft_ms"] for r in valid)[-1]
        return TuneTrial(
            samples=[r["output_throughput"] for r in valid],
            p99_ttft_ms=p99,
        )

    def quality(tune_config: dict[str, Any]) -> float:
        last_exc: Exception | None = None
        texts = []
        for _ in (1, 2):  # one immediate retry on transient failure
            try:
                texts = worker.answer_batch.remote(
                    commit=base_commit,
                    flags=tune_config,
                    model=cfg.model,
                    prompts=[q.prompt for q in bank],
                )
                break
            except Exception as exc:  # noqa: BLE001 - retried once, then raised
                last_exc = exc
        else:
            raise last_exc or RuntimeError("answer_batch failed twice")
        by_prompt = dict(zip([q.prompt for q in bank], texts, strict=False))
        acc, _ = accuracy(lambda p: by_prompt[p], bank)
        return acc

    propose = None
    orch = _make_orchestrator(cfg, run_id, Path("runs") / run_id)
    if orch is not None:
        propose = make_propose_fn(orch, cfg.tune.knobs, hardware=cfg.gpu)
    outcome = drive_tune_live(
        base_commit=base_commit,
        knob_space=dict(cfg.tune.knobs),
        measure=measure,
        quality=quality,
        propose=propose,
        run_id=run_id,
        model=cfg.model,
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
def demo(
    run: Path = typer.Argument(..., help="runs/<run_id>"),
    speed: float = typer.Option(1.5, "--speed", help="seconds per event"),
    out: Path | None = typer.Option(None, "--out", help="scratch dir"),
) -> None:
    """Replay a recorded run into a scratch dir, re-rendering the report
    after each event (judge demo: narrowing-bisect movie, no GPU/key)."""
    import json
    import shutil
    import time

    from perfpilot.report.render import render_report

    events = Path(run) / "events.jsonl"
    if not events.exists():
        typer.echo(f"error: no events.jsonl in {run}", err=True)
        raise typer.Exit(code=2)
    lines = [ln for ln in events.read_text().splitlines() if ln.strip()]
    agent = Path(run) / "agent-events.jsonl"
    agent_lines = (
        [ln for ln in agent.read_text().splitlines() if ln.strip()]
        if agent.exists()
        else []
    )
    scratch = out or Path(f"{run}-demo")
    if scratch.exists():
        shutil.rmtree(scratch)
    (scratch).mkdir(parents=True)
    staged: list[str] = []
    agent_staged: list[str] = []
    main_ts = [json.loads(ln).get("ts", "") for ln in lines]
    agent_ts = [json.loads(ln).get("ts", "") for ln in agent_lines]
    ai = 0
    for ln, ts in zip(lines, main_ts, strict=True):
        staged.append(ln)
        while ai < len(agent_lines) and agent_ts[ai] <= ts:
            agent_staged.append(agent_lines[ai])
            ai += 1
        (scratch / "events.jsonl").write_text("\n".join(staged) + "\n")
        (scratch / "agent-events.jsonl").write_text(
            "\n".join(agent_staged) + ("\n" if agent_staged else "")
        )
        (scratch / "report.html").write_text(render_report(scratch / "events.jsonl"))
        if speed > 0:
            time.sleep(speed)
    while ai < len(agent_lines):
        agent_staged.append(agent_lines[ai])
        ai += 1
    (scratch / "agent-events.jsonl").write_text(
        "\n".join(agent_staged) + ("\n" if agent_staged else "")
    )
    (scratch / "report.html").write_text(render_report(scratch / "events.jsonl"))
    typer.echo(f"replayed {len(staged)} events -> {scratch}/report.html")


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
    from perfpilot.evals.run_eval import seeded_culprit_positions

    commits = wheeled_range(cfg.bisect.repo, 16)
    # index 0 has no clean in-range baseline; see README "Known limitation"
    positions = seeded_culprit_positions(max(trials, 1), len(commits))
    results = []
    for i, pos in enumerate(positions):
        summary = drive_eval_trial(
            commits,
            culprit_index=pos,
            inject_kind="enforce_eager",
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


@app.command()
def bench(
    variants: str = typer.Option("fallback,single,swarm", "--variants"),
    seeds: int = typer.Option(5, "--seeds", min=1, max=50),
    live_brain: bool = typer.Option(
        False, "--live-brain", help="Roles call the Baseten brain (needs key)"
    ),
    record: bool = typer.Option(False, "--record", help="Append to scoreboard"),
    board: Path = typer.Option(Path("bench/scoreboard.jsonl"), "--board"),
    suite: str = typer.Option("all", "--suite", help="core | stress | all"),
) -> None:
    """Score agent variants on a simulated GPU ($0 GPU). Tracks progress."""
    import subprocess

    from perfpilot.bench.format import scenario_table, trend_table, variant_table
    from perfpilot.bench.harness import VARIANTS, run_suite
    from perfpilot.bench.scenarios import ALL_SCENARIOS, SCENARIOS, STRESS
    from perfpilot.bench.scoreboard import append_entry, load_entries, trend_rows

    chosen = [v.strip() for v in variants.split(",") if v.strip()]
    unknown = [v for v in chosen if v not in VARIANTS]
    if unknown or not chosen:
        typer.echo(f"error: variants must be from {VARIANTS}, got {unknown}", err=True)
        raise typer.Exit(code=2)
    suites = {"core": SCENARIOS, "stress": STRESS, "all": ALL_SCENARIOS}
    if suite not in suites:
        typer.echo(f"error: --suite must be one of {sorted(suites)}", err=True)
        raise typer.Exit(code=2)
    mode = ("live" if live_brain else "scripted") + (
        "" if suite == "core" else f"-{suite}"
    )
    workdir = Path("runs") / _run_id("bench")
    summary = run_suite(
        chosen, seeds=seeds, scenarios=suites[suite], live=live_brain, workdir=workdir
    )
    typer.echo(f"mode={mode} seeds={seeds} scenarios={len(suites[suite])}\n")
    typer.echo(variant_table(summary) + "\n")
    typer.echo(scenario_table(summary) + "\n")
    if record:
        sha = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
            or "unknown"
        )
        append_entry(board, summary, git_sha=sha, mode=mode)
    for v in chosen:
        rows = trend_rows(load_entries(board), variant=v, mode=mode)
        if rows:
            typer.echo(f"trend: {v} ({mode})\n" + trend_table(rows[-10:]) + "\n")
