"""Planted-regression scenarios. Each probes one failure mode of the loop."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

RANGE = 16
BLAND = ["docs: fix typo", "ci: bump runner", "test: add case", "chore: lint"]


class Scenario(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    culprit: int | None = Field(description="Hidden index; None = no regression")
    drop: float = 0.0
    cv: float = 0.01
    outlier_p: float = 0.0
    hints: dict[int, str] = Field(default_factory=dict)
    n: int = RANGE
    extra_regressions: dict[int, float] = Field(
        default_factory=dict, description="Further drops stacking at/after index"
    )
    fix_at: int | None = Field(default=None, description="Regression reverted here")
    broken: frozenset[int] = Field(default=frozenset(), description="Build fails")
    error_p: float = Field(default=0.0, description="Transient worker error rate")
    drift_per_probe: float = Field(default=0.0, description="Rig slowdown per probe")
    hint_all: str = Field(default="", description="Suffix on every message")
    family: str = "core"

    @property
    def truth(self) -> set[int]:
        """Acceptable culprit indices (any planted transition)."""
        if self.culprit is None or self.fix_at is not None or self.drop < 0:
            return set()
        return {self.culprit, *self.extra_regressions}

    @property
    def commits(self) -> list[str]:
        return [f"{self.name}-{i:02d}" for i in range(self.n)]

    @property
    def messages(self) -> dict[str, str]:
        return {
            sha: self.hints.get(i, BLAND[i % len(BLAND)]) + self.hint_all
            for i, sha in enumerate(self.commits)
        }

    def index_of(self, sha: str) -> int:
        return self.commits.index(sha)


SCENARIOS: list[Scenario] = [
    Scenario(
        name="clean_strong",
        description="40% drop, low noise: the baseline case",
        culprit=7,
        drop=0.40,
    ),
    Scenario(
        name="mild",
        description="12% drop near the bad end",
        culprit=10,
        drop=0.12,
    ),
    Scenario(
        name="noisy",
        description="25% drop under 4% run-to-run noise",
        culprit=5,
        drop=0.25,
        cv=0.04,
    ),
    Scenario(
        name="outlier",
        description="Culprit at index 1 with collapsed-run outliers (trial-0 failure)",
        culprit=1,
        drop=0.50,
        outlier_p=0.08,
    ),
    Scenario(
        name="code_hint",
        description="Culprit message names a hot path the workload exercises",
        culprit=12,
        drop=0.30,
        hints={12: "[Scheduler] rewrite batch admission loop"},
    ),
    Scenario(
        name="misleading_hint",
        description="A hot-path message on an innocent commit; real culprit is bland",
        culprit=11,
        drop=0.30,
        hints={3: "[Attention] refactor kernel dispatch"},
    ),
    Scenario(
        name="no_regression",
        description="Endpoints equal: the only right answer is no_regression",
        culprit=None,
    ),
]

STRESS: list[Scenario] = [
    Scenario(
        name="two_culprits",
        description="Two stacked regressions (20% at 4, 25% at 11); either is right",
        culprit=4,
        drop=0.20,
        extra_regressions={11: 0.25},
        family="stress",
    ),
    Scenario(
        name="broken_builds",
        description="Commits next to the culprit fail to build",
        culprit=9,
        drop=0.30,
        broken=frozenset({7, 8}),
        family="stress",
    ),
    Scenario(
        name="flaky_worker",
        description="10% of trials die (preemption/OOM) with a transient error",
        culprit=6,
        drop=0.30,
        error_p=0.10,
        family="stress",
    ),
    Scenario(
        name="rig_drift",
        description="GPU slows 0.6% per probe (thermal/neighbour); 15% real drop",
        culprit=10,
        drop=0.15,
        drift_per_probe=-0.006,
        family="stress",
    ),
    Scenario(
        name="borderline",
        description="5% drop under 1.5% noise: just above the 3% floor",
        culprit=8,
        drop=0.05,
        cv=0.015,
        family="stress",
    ),
    Scenario(
        name="transient_regression",
        description="Regressed at 4, fixed at 10: endpoints equal, no regression",
        culprit=4,
        drop=0.30,
        fix_at=10,
        family="stress",
    ),
    Scenario(
        name="speedup",
        description="Range got 20% faster: an improvement is not a regression",
        culprit=6,
        drop=-0.20,
        family="stress",
    ),
    Scenario(
        name="wide_range",
        description="64-commit range, culprit at 37",
        culprit=37,
        drop=0.25,
        n=64,
        family="stress",
    ),
    Scenario(
        name="hint_spam",
        description="Every message mentions a hot path: hints carry no signal",
        culprit=3,
        drop=0.30,
        hint_all=" [kernel]",
        family="stress",
    ),
    Scenario(
        name="culprit_last",
        description="Culprit is the bad endpoint's own commit",
        culprit=15,
        drop=0.30,
        family="stress",
    ),
]

ALL_SCENARIOS: list[Scenario] = [*SCENARIOS, *STRESS]
