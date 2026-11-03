"""Simulated A100 measurement plane: seeded noise, outliers, metered cost."""

from __future__ import annotations

import random

from perfpilot.bench.scenarios import Scenario
from perfpilot.modes.bisect import TrialOutcome

BASE_TPS = 3600.0
#: Collapsed-run factor seen live (1703 vs ~3590 in m2-eval-9).
OUTLIER_FACTOR = 0.47
#: Per-probe setup + per-run seconds, roughly matching recorded A100 trials.
SETUP_S = 120.0
RUN_S = 60.0
GPU_COST_PER_HOUR = 2.10


class SimGPU:
    """Deterministic for a given (scenario, seed, call sequence)."""

    def __init__(self, scenario: Scenario, seed: int = 0) -> None:
        self.scenario = scenario
        self._rng = random.Random(f"{scenario.name}:{seed}")
        self.runs_used = 0
        self.probes = 0
        self.gpu_seconds = 0.0

    def _true_tps(self, idx: int) -> float:
        sc = self.scenario
        tps = BASE_TPS
        live = sc.fix_at is None or idx < sc.fix_at
        if sc.culprit is not None and idx >= sc.culprit and live:
            tps *= 1.0 - sc.drop
        for at, drop in sc.extra_regressions.items():
            if idx >= at:
                tps *= 1.0 - drop
        return tps * (1.0 + sc.drift_per_probe) ** self.probes

    def _sample(self, idx: int) -> float:
        sc = self.scenario
        tps = self._true_tps(idx) * self._rng.gauss(1.0, sc.cv)
        if self._rng.random() < sc.outlier_p:
            tps *= OUTLIER_FACTOR
        return max(tps, 1.0)

    def measure(self, sha: str, runs: int) -> TrialOutcome:
        sc = self.scenario
        idx = sc.index_of(sha)
        self.probes += 1
        if idx in sc.broken:
            self.gpu_seconds += SETUP_S
            return TrialOutcome(skipped=True, reason="build failed: no wheel")
        if self._rng.random() < sc.error_p:
            self.gpu_seconds += SETUP_S
            return TrialOutcome(skipped=True, reason="trial error: worker preempted")
        self.runs_used += runs
        self.gpu_seconds += SETUP_S + RUN_S * runs
        return TrialOutcome(samples=[self._sample(idx) for _ in range(runs)])

    @property
    def cost_usd(self) -> float:
        return self.gpu_seconds / 3600.0 * GPU_COST_PER_HOUR
