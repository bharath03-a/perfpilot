"""Eval protocol harness (PLAN.md 3). Pure: no GPU, no network.

Protocol: one FIXED 16-commit range where all wheels exist; 2 seeded
trials with culprit positions including the first and last commit.
Reuse cached venvs across trials, never reuse benchmark results
(injection changes them). Skip explanation in eval trials.

Reported per README/video: correct culprits (target 2/2), wrong verdicts
at any step (target 0), steps vs log2(range), inconclusive steps +
extra runs, GPU minutes + cost per trial.
"""

from __future__ import annotations

import math
import random
from typing import Literal

from pydantic import BaseModel, Field

from perfpilot.config import PerfPilotConfig
from perfpilot.evals.inject import InjectionKind

RANGE_SIZE = 16
N_TRIALS = 2
#: $/h default mirrors PerfPilotConfig.gpu_cost_per_hour (measured
#: A100-40GB rate); callers must pass config.gpu_cost_per_hour through.
#: Never hardcode a competing constant.
FALLBACK_GPU_COST_PER_HOUR = float(
    PerfPilotConfig.model_fields["gpu_cost_per_hour"].default
)


class EvalConfig(BaseModel):
    range_size: int = Field(
        default=RANGE_SIZE, description="Fixed eval range; PLAN.md pins 16"
    )
    n_trials: int = Field(default=N_TRIALS, ge=1)
    seed: int = 0
    commits: list[str] = Field(
        default_factory=list, description="Range SHAs, oldest first"
    )
    injection_kinds: list[InjectionKind] = Field(
        default_factory=lambda: ["enforce_eager", "gpu_mem_05"],
        description=(
            "Overlay per trial, cycled: trial 0 plants the strong "
            "enforce_eager overlay, trial 1 the milder gpu_mem_05 overlay, "
            "extra trials cycle in order over seeded interior positions"
        ),
    )
    gpu_cost_per_hour: float = FALLBACK_GPU_COST_PER_HOUR


class TrialSpec(BaseModel):
    """One planned eval trial (planning-only, never logged).

    Only EvalSummary aggregates enter reports; TrialSpec and TrialOutcome
    stay out of events.jsonl so the planted culprit_index/kind cannot leak
    to the agent through the log.
    """

    trial_id: int
    culprit_index: int = Field(description="Hidden planted position in range")
    kind: InjectionKind


class TrialOutcome(BaseModel):
    trial_id: int
    culprit_index: int = Field(description="Planted position")
    found_index: int | None = Field(default=None, description="Bisect result position")
    found_range: list[int] = Field(
        default_factory=list, description="Range if skips intervene"
    )
    wrong_verdicts: int = 0
    steps: int = 0
    inconclusive_steps: int = 0
    extra_runs: int = 0
    gpu_seconds: float = 0.0

    @property
    def correct(self) -> bool:
        if self.found_index is not None:
            return self.found_index == self.culprit_index
        return self.culprit_index in self.found_range


class EvalSummary(BaseModel):
    n_trials: int
    correct: int
    total: int
    wrong_verdicts: int
    expected_steps: float = Field(description="log2(range_size)")
    steps: list[int] = Field(description="Bisect steps per trial")
    inconclusive_steps: int
    extra_runs: int
    gpu_minutes: list[float] = Field(description="Per trial")
    cost_usd: list[float] = Field(description="Per trial at gpu_cost_per_hour")
    target_correct: str = "2/2"
    target_wrong: int = 0


def seeded_culprit_positions(
    n_trials: int = N_TRIALS, range_size: int = RANGE_SIZE, seed: int = 0
) -> list[int]:
    """Seeded culprit positions; always include the low edge and last commit.

    For the PLAN.md default (2 trials) this is [1, range_size-1]. Index 0
    is excluded by construction: a culprit at the range start leaves no
    clean in-range baseline, so no transition exists to detect and the only
    honest verdict is no_regression (see
    test_eval_protocol_first_commit_reports_no_regression). Index 1 is the
    lowest detectable position. Extra trials draw seeded interior positions
    without replacement, never duplicating the edges.
    """
    if n_trials <= 0:
        return []
    low = 1 if range_size > 1 else 0
    positions = [low, range_size - 1]
    if n_trials > 2:
        rng = random.Random(seed)
        pool = [i for i in range(1, range_size - 1) if i not in positions]
        positions.extend(rng.sample(pool, min(n_trials - 2, len(pool))))
    return positions[:n_trials]


def plan_trials(config: EvalConfig) -> list[TrialSpec]:
    """Trial specs for the fixed range: seeded positions, cycled overlays.

    Default pairing is trial 0 = strong overlay at the low edge (index 1,
    the lowest detectable position) and trial 1 = mild overlay at the last
    commit; any alternate pairing must still cover both overlays and both
    range edges.
    """
    positions = seeded_culprit_positions(
        config.n_trials, config.range_size, config.seed
    )
    kinds = config.injection_kinds or ["enforce_eager"]
    return [
        TrialSpec(trial_id=i, culprit_index=pos, kind=kinds[i % len(kinds)])
        for i, pos in enumerate(positions)
    ]


def venv_key(commit_sha: str) -> str:
    """Venv cache key: reused across trials."""
    return commit_sha


def result_key(
    commit_sha: str, flags_hash: str, kind: InjectionKind, trial_id: int
) -> str:
    """Result cache key: trial-scoped, so results are never reused."""
    return f"{commit_sha}__{flags_hash}__{kind}__trial{trial_id}"


def summarize_trials(
    outcomes: list[TrialOutcome],
    range_size: int = RANGE_SIZE,
    gpu_cost_per_hour: float = FALLBACK_GPU_COST_PER_HOUR,
) -> EvalSummary:
    """Aggregate per-trial outcomes into the README/video eval table."""
    correct = sum(1 for o in outcomes if o.correct)
    wrong = sum(o.wrong_verdicts for o in outcomes)
    gpu_minutes = [o.gpu_seconds / 60.0 for o in outcomes]
    cost = [m / 60.0 * gpu_cost_per_hour for m in gpu_minutes]
    return EvalSummary(
        n_trials=len(outcomes),
        correct=correct,
        total=len(outcomes),
        wrong_verdicts=wrong,
        expected_steps=math.log2(range_size) if range_size > 0 else 0.0,
        steps=[o.steps for o in outcomes],
        inconclusive_steps=sum(o.inconclusive_steps for o in outcomes),
        extra_runs=sum(o.extra_runs for o in outcomes),
        gpu_minutes=gpu_minutes,
        cost_usd=cost,
    )


VerdictKind = Literal["good", "bad", "inconclusive", "skip", "ambiguous"]


def count_wrong_verdicts(
    verdicts: list[tuple[int, VerdictKind]], culprit_index: int
) -> int:
    """Wrong verdicts at any step: commits before culprit called bad, or
    culprit-or-later called good. Skips/ambiguous/inconclusive never count."""
    wrong = 0
    for index, label in verdicts:
        truth_bad = index >= culprit_index
        if label == "bad" and not truth_bad:
            wrong += 1
        elif label == "good" and truth_bad:
            wrong += 1
    return wrong


def format_eval_table(summary: EvalSummary) -> str:
    """Markdown eval table for README + video."""
    correct = f"{summary.correct}/{summary.total}"
    steps = f"{summary.steps} vs {summary.expected_steps:.1f}"
    lines = [
        "| Metric | Value | Target |",
        "|---|---|---|",
        f"| Correct culprits | {correct} | {summary.target_correct} |",
        f"| Wrong verdicts | {summary.wrong_verdicts} | {summary.target_wrong} |",
        f"| Bisect steps vs log2(range) | {steps} | ~log2 |",
        f"| Inconclusive steps | {summary.inconclusive_steps} | reported |",
        f"| Extra runs | {summary.extra_runs} | reported |",
    ]
    pairs = zip(summary.gpu_minutes, summary.cost_usd, strict=True)
    for i, (m, c) in enumerate(pairs):
        lines.append(f"| Trial {i}: GPU min / cost | {m:.1f} / ${c:.2f} | reported |")
    return "\n".join(lines)
