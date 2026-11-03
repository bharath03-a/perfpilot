"""Run variants x scenarios x seeds on the simulated GPU and score them."""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import BaseModel

from perfpilot.bench.scenarios import SCENARIOS, Scenario
from perfpilot.bench.sim import SimGPU
from perfpilot.bench.swarm import scripted_single, scripted_swarm
from perfpilot.modes.bisect import BisectOutcome, BisectState, run_bisect

VARIANTS = ("fallback", "single", "swarm")
#: score = 100*correct - 200*wrong - 2*probes (per-run means). A wrong
#: verdict costs two right ones; each GPU probe costs a little.
W_CORRECT, W_WRONG, W_PROBE = 100.0, 200.0, 2.0

PickFn = Callable[[BisectState], str | None]


class ScenarioResult(BaseModel):
    scenario: str
    variant: str
    seed: int
    status: str
    correct: bool
    exact: bool
    wrong: int
    probes: int
    runs: int
    cost_usd: float
    fallbacks: int


class VariantRow(BaseModel):
    variant: str
    n: int
    correct_rate: float
    exact_rate: float
    wrong: int
    mean_probes: float
    mean_cost_usd: float
    score: float


class SuiteSummary(BaseModel):
    rows: list[VariantRow]
    results: list[ScenarioResult]


def judge(sc: Scenario, out: BisectOutcome) -> tuple[bool, bool, int]:
    """(correct, exact, wrong). Missing a regression is incorrect, not wrong."""
    if not sc.truth:
        ok = out.status == "no_regression"
        return ok, ok, int(out.status in ("culprit", "culprit_range"))
    truth = {sc.commits[i] for i in sc.truth}
    exact = out.status == "culprit" and out.culprit in truth
    in_range = out.status == "culprit_range" and bool(truth & set(out.culprit_range))
    wrong = int(out.status == "culprit" and out.culprit not in truth)
    if out.status == "culprit_range" and not truth & set(out.culprit_range):
        wrong = 1
    return exact or in_range, exact, wrong


def _live_pick(variant: str, sc: Scenario, workdir: Path) -> PickFn:
    from perfpilot.agent.client import SparkClient
    from perfpilot.bench.live_roles import live_swarm
    from perfpilot.orchestrator import AgentOrchestrator, make_pick_fn

    orch = AgentOrchestrator(
        run_id=f"bench-{sc.name}",
        events_path=workdir / f"{sc.name}-{variant}.jsonl",
        client=SparkClient(),
    )
    if variant == "single":
        return make_pick_fn(orch, messages=sc.messages)
    return live_swarm(orch, sc.messages)


def make_pick(
    variant: str, sc: Scenario, live: bool, workdir: Path | None
) -> PickFn | None:
    if variant == "fallback":
        return None
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
    if live:
        return _live_pick(variant, sc, workdir or Path(tempfile.mkdtemp()))
    if variant == "single":
        return scripted_single(sc.messages)
    return scripted_swarm(sc.messages)


def run_scenario(
    sc: Scenario,
    variant: str,
    seed: int = 0,
    live: bool = False,
    workdir: Path | None = None,
) -> ScenarioResult:
    gpu = SimGPU(sc, seed)
    out = run_bisect(
        sc.commits, gpu.measure, pick=make_pick(variant, sc, live, workdir)
    )
    correct, exact, wrong = judge(sc, out)
    return ScenarioResult(
        scenario=sc.name,
        variant=variant,
        seed=seed,
        status=out.status,
        correct=correct,
        exact=exact,
        wrong=wrong,
        probes=gpu.probes,
        runs=gpu.runs_used,
        cost_usd=round(gpu.cost_usd, 4),
        fallbacks=out.fallbacks,
    )


def summarize(variant: str, results: Sequence[ScenarioResult]) -> VariantRow:
    n = len(results)
    correct = sum(r.correct for r in results) / n
    wrong = sum(r.wrong for r in results)
    probes = sum(r.probes for r in results) / n
    score = W_CORRECT * correct - W_WRONG * wrong / n - W_PROBE * probes
    return VariantRow(
        variant=variant,
        n=n,
        correct_rate=round(correct, 3),
        exact_rate=round(sum(r.exact for r in results) / n, 3),
        wrong=wrong,
        mean_probes=round(probes, 2),
        mean_cost_usd=round(sum(r.cost_usd for r in results) / n, 3),
        score=round(score, 2),
    )


def run_suite(
    variants: Sequence[str] = VARIANTS,
    seeds: int = 5,
    scenarios: Sequence[Scenario] = SCENARIOS,
    live: bool = False,
    workdir: Path | None = None,
) -> SuiteSummary:
    results = [
        run_scenario(sc, v, s, live, workdir)
        for v in variants
        for sc in scenarios
        for s in range(seeds)
    ]
    rows = [summarize(v, [r for r in results if r.variant == v]) for v in variants]
    return SuiteSummary(rows=rows, results=results)
