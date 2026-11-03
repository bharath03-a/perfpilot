"""PerfBisect deterministic state machine.

Pure logic: no network, GPU, or subprocess. The orchestrator injects a
``measure`` callback that benchmarks commits; agent proposals enter through
the ``pick`` / ``extra_runs`` callbacks with SPEC.md fallbacks as defaults
(middle commit, 2 extra runs). Verdicts come only from
:mod:`perfpilot.stats`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from perfpilot.stats import (
    Verdict,
    classify_commit,
    confirm_regression,
    log_ratio_interval,
)

DEFAULT_EXTRA_RUNS = 2
MAX_EXTRA_RUNS_PER_STEP = 3


class CommitLabel(StrEnum):
    UNTESTED = "untested"
    GOOD = "good"
    BAD = "bad"
    SKIPPED = "skipped"
    AMBIGUOUS = "ambiguous"


class BisectParams(BaseModel):
    min_runs: int = Field(default=3, ge=2)
    max_runs: int = Field(default=8, ge=2)
    confidence: float = Field(default=0.90, gt=0.0, lt=1.0)
    drift_every: int = Field(default=4, ge=1)
    drift_runs: int = Field(default=2, ge=2)
    drift_threshold: float = Field(default=0.02, gt=0.0)
    transient_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Re-run a trial whose reason starts with 'trial error' "
        "(preemption, OOM, cold container). Build skips are never retried.",
    )
    baseline_max_spread: float = Field(
        default=0.30,
        gt=0.0,
        description="Re-measure an endpoint once when (max-min)/mean exceeds "
        "this: a single collapsed run poisons baseline variance and turns "
        "every later comparison ambiguous.",
    )

    @model_validator(mode="after")
    def _caps_agree(self) -> BisectParams:
        if self.max_runs < self.min_runs:
            raise ValueError("max_runs must be >= min_runs")
        return self


class TrialOutcome(BaseModel):
    """One injected benchmark result for a commit."""

    samples: list[float] = Field(default_factory=list)
    skipped: bool = False
    reason: str = ""


class PickResult(BaseModel):
    sha: str
    fallback_used: bool = False
    reason: str = ""


class DriftCheck(BaseModel):
    exceeded: bool
    drift: float = 0.0
    reason: str = ""


class BisectState(BaseModel):
    commits: list[str]
    good_idx: int = 0
    bad_idx: int = 0
    labels: dict[str, CommitLabel] = Field(default_factory=dict)
    samples: dict[str, list[float]] = Field(default_factory=dict)
    full_effect: float = 0.0
    steps: int = 0
    extra_runs_used: int = 0
    drift_checks: int = 0
    endpoint_remeasures: int = 0
    fallbacks: int = 0
    tested_order: list[str] = Field(default_factory=list)
    pick_log: list[str] = Field(default_factory=list)
    drift_records: list[DriftCheck] = Field(default_factory=list)


class CulpritResult(BaseModel):
    culprit: str
    culprit_range: list[str] = Field(default_factory=list)
    skipped_between: list[str] = Field(default_factory=list)


class VerifyResult(BaseModel):
    verified: bool
    effect: float = 0.0
    lo: float = 0.0
    hi: float = 0.0
    ref: str | None = None
    reason: str = ""


BisectStatus = Literal["culprit", "culprit_range", "no_regression", "endpoint_error"]


class BisectOutcome(BaseModel):
    """Bisect result. ``steps`` counts decisive good/bad verdicts only.

    Commits marked ambiguous at the run cap do not move a boundary, so they
    never increment ``steps``; they appear in ``tested_order``, ``ambiguous``,
    and ``labels`` instead.
    """

    status: BisectStatus = "culprit"
    confirmed: bool = False
    reason: str = ""
    full_effect: float = 0.0
    culprit: str | None = None
    culprit_range: list[str] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)
    ambiguous: list[str] = Field(default_factory=list)
    verified: bool = False
    verify_reason: str = ""
    verify_ref: str | None = None
    steps: int = 0
    extra_runs_used: int = 0
    drift_checks: int = 0
    endpoint_remeasures: int = 0
    fallbacks: int = 0
    tested_order: list[str] = Field(
        default_factory=list,
        description="Every measured SHA in visit order, including skips, "
        "ambiguous-at-cap commits, drift rechecks, and verify trials; "
        "remeasures repeat the SHA. Join with labels for the timeline.",
    )
    labels: dict[str, str] = Field(
        default_factory=dict,
        description="Final label per visited SHA (good/bad/skipped/ambiguous).",
    )
    pick_log: list[str] = Field(
        default_factory=list,
        description="One entry per pick/extra-runs decision, accepted or "
        "fallback, with the reason.",
    )
    drift_records: list[DriftCheck] = Field(
        default_factory=list,
        description="One DriftCheck per drift recheck, in order.",
    )


MeasureFn = Callable[[str, int], TrialOutcome]
PickFn = Callable[[BisectState], str | None]
ExtraRunsFn = Callable[[str, int, Verdict], int]


def new_bisect_state(commits: Sequence[str]) -> BisectState:
    items = list(commits)
    if len(items) < 2:
        raise ValueError(f"need >=2 commits, got {len(items)}")
    return BisectState(
        commits=items,
        good_idx=0,
        bad_idx=len(items) - 1,
        labels={sha: CommitLabel.UNTESTED for sha in items},
    )


def open_range(state: BisectState) -> list[str]:
    """Untested, unskipped commits strictly between the boundaries."""
    out = []
    for sha in state.commits[state.good_idx + 1 : state.bad_idx]:
        if state.labels.get(sha, CommitLabel.UNTESTED) == CommitLabel.UNTESTED:
            out.append(sha)
    return out


def default_pick(state: BisectState) -> str | None:
    """SPEC.md fallback: the middle untested, unskipped commit."""
    cands = open_range(state)
    if not cands:
        return None
    return cands[len(cands) // 2]


def validate_pick(state: BisectState, sha: str | None) -> bool:
    return sha in set(open_range(state))


def pick_next(state: BisectState, propose: PickFn | None = None) -> PickResult:
    """Agent proposal validated; any miss falls back to the middle commit."""
    fallback = default_pick(state)
    if fallback is None:
        raise ValueError("no untested commits left in range")
    if propose is None:
        return PickResult(sha=fallback, reason="no agent: middle commit")
    try:
        proposal = propose(state)
    except Exception as exc:
        return PickResult(
            sha=fallback, fallback_used=True, reason=f"pick raised: {exc}"
        )
    if validate_pick(state, proposal):
        assert proposal is not None
        return PickResult(sha=proposal, reason="agent proposal accepted")
    return PickResult(
        sha=fallback,
        fallback_used=True,
        reason=f"rejected proposal {proposal!r}: outside open range",
    )


def default_extra_runs(commit_sha: str, runs_used: int, verdict: Verdict) -> int:
    """SPEC.md fallback: 2 extra runs."""
    return DEFAULT_EXTRA_RUNS


def clamp_extra_runs(wanted: int, runs_used: int, max_runs: int) -> int:
    if wanted < 0:
        return 0
    return max(0, min(MAX_EXTRA_RUNS_PER_STEP, wanted, max_runs - runs_used))


def drift_exceeds(
    old_samples: Sequence[float],
    new_samples: Sequence[float],
    threshold: float = 0.02,
) -> DriftCheck:
    """|ln(mean_new / mean_old)| over threshold means the rig drifted."""
    import math

    old = list(old_samples)
    new = list(new_samples)
    if not old or not new:
        return DriftCheck(exceeded=False, reason="missing samples")
    drift = math.log((math.fsum(new) / len(new)) / (math.fsum(old) / len(old)))
    if abs(drift) > threshold:
        return DriftCheck(
            exceeded=True,
            drift=drift,
            reason=f"|drift|={abs(drift):.4f} > {threshold:.0%}: re-measure endpoints",
        )
    return DriftCheck(exceeded=False, drift=drift, reason="endpoints stable")


def culprit_of(state: BisectState) -> CulpritResult:
    """First bad commit; a range when skips/ambiguities intervene."""
    culprit = state.commits[state.bad_idx]
    between = state.commits[state.good_idx + 1 : state.bad_idx]
    skipped = [
        s
        for s in between
        if state.labels.get(s) in (CommitLabel.SKIPPED, CommitLabel.AMBIGUOUS)
    ]
    if skipped:
        return CulpritResult(
            culprit=culprit,
            culprit_range=[*between, culprit],
            skipped_between=skipped,
        )
    return CulpritResult(culprit=culprit, culprit_range=[culprit])


def verify_culprit(
    culprit_samples: Sequence[float],
    ref_samples: Sequence[float],
    full_effect: float,
    ref: str = "",
    confidence: float = 0.90,
) -> VerifyResult:
    """Back-to-back check: significant and in the same direction as R."""
    try:
        iv = log_ratio_interval(list(culprit_samples), list(ref_samples), confidence)
    except ValueError as exc:
        return VerifyResult(
            verified=False,
            ref=ref or None,
            reason=f"verify samples invalid: {exc}",
        )
    if iv.lo == iv.hi:
        return VerifyResult(
            effect=iv.effect,
            lo=iv.lo,
            hi=iv.hi,
            ref=ref or None,
            verified=False,
            reason="verify samples have zero variance: not counted",
        )
    if iv.lo <= 0.0 <= iv.hi:
        return VerifyResult(
            effect=iv.effect,
            lo=iv.lo,
            hi=iv.hi,
            ref=ref or None,
            verified=False,
            reason="verify interval covers 0: not significant",
        )
    if (iv.effect > 0) != (full_effect > 0):
        return VerifyResult(
            effect=iv.effect,
            lo=iv.lo,
            hi=iv.hi,
            ref=ref or None,
            verified=False,
            reason="verify direction differs from R: not counted",
        )
    return VerifyResult(
        effect=iv.effect,
        lo=iv.lo,
        hi=iv.hi,
        ref=ref or None,
        verified=True,
        reason="significant and same direction as R",
    )


def _fresh(trial: TrialOutcome) -> bool:
    return not trial.skipped and len(trial.samples) > 0


def _spread(samples: list[float]) -> float:
    """Relative spread (max-min)/mean; inf when unmeasurable."""
    if len(samples) < 2:
        return float("inf")
    mean = sum(samples) / len(samples)
    if mean == 0:
        return float("inf")
    return (max(samples) - min(samples)) / abs(mean)


def _trim_collapsed(samples: list[float], p: BisectParams) -> list[float] | None:
    """Drop runs far from the pooled median (collapsed runs); None if too few left."""
    ordered = sorted(samples)
    n = len(ordered)
    median = (ordered[(n - 1) // 2] + ordered[n // 2]) / 2.0
    if median <= 0:
        return None
    band = p.baseline_max_spread / 2.0
    kept = [x for x in samples if abs(x - median) / median <= band]
    return kept if len(kept) >= p.min_runs else None


def _safe_classify(
    samples: list[float], good: list[float], full_effect: float, confidence: float
) -> Verdict:
    """classify_commit that never raises: invalid samples become
    inconclusive so the extra-runs loop (or the ambiguous label) handles
    them instead of crashing the run."""
    try:
        return classify_commit(samples, good, full_effect, confidence)
    except ValueError as exc:
        return Verdict(
            label="inconclusive",
            effect=0.0,
            lo=0.0,
            hi=0.0,
            confidence=confidence,
            reason=f"commit samples invalid: {exc}",
        )


def _outcome_lists(state: BisectState) -> tuple[list[str], list[str]]:
    skipped = [s for s, v in state.labels.items() if v == CommitLabel.SKIPPED]
    ambiguous = [s for s, v in state.labels.items() if v == CommitLabel.AMBIGUOUS]
    return skipped, ambiguous


TRANSIENT_PREFIX = "trial error"


def with_transient_retries(measure: MeasureFn, retries: int) -> MeasureFn:
    """Retry transient worker errors; a persistent one still surfaces."""

    def wrapped(sha: str, runs: int) -> TrialOutcome:
        outcome = measure(sha, runs)
        for _ in range(retries):
            if not (outcome.skipped and outcome.reason.startswith(TRANSIENT_PREFIX)):
                break
            outcome = measure(sha, runs)
        return outcome

    return wrapped


def run_bisect(
    commits: Sequence[str],
    measure: MeasureFn,
    *,
    params: BisectParams | None = None,
    pick: PickFn | None = None,
    extra_runs: ExtraRunsFn | None = None,
) -> BisectOutcome:
    """Deterministic PerfBisect driver over an oldest-first commit list.

    Adapter contract: the orchestrator unwraps agent Decision objects before
    invoking ``pick`` / ``extra_runs`` (make_pick_fn/make_extra_runs_fn), so
    these callbacks receive plain values. Any non-conforming value that slips
    through -- including a raw Decision -- is treated as invalid and counted
    as a fallback with its reason in ``pick_log``; it never raises.
    """
    p = params or BisectParams()
    measure = with_transient_retries(measure, p.transient_retries)
    items = list(commits)
    state = new_bisect_state(items)
    extra_fn = extra_runs or default_extra_runs

    def snapshot(
        status: BisectStatus,
        confirmed: bool,
        reason: str,
        culprit: CulpritResult | None = None,
        verify: VerifyResult | None = None,
    ) -> BisectOutcome:
        skipped, ambiguous = _outcome_lists(state)
        return BisectOutcome(
            status=status,
            confirmed=confirmed,
            reason=reason,
            full_effect=state.full_effect,
            culprit=culprit.culprit if culprit else None,
            culprit_range=culprit.culprit_range if culprit else [],
            skipped=skipped,
            ambiguous=ambiguous,
            verified=verify.verified if verify else False,
            verify_reason=verify.reason if verify else "",
            verify_ref=verify.ref if verify and verify.ref else None,
            steps=state.steps,
            extra_runs_used=state.extra_runs_used,
            drift_checks=state.drift_checks,
            endpoint_remeasures=state.endpoint_remeasures,
            fallbacks=state.fallbacks,
            tested_order=list(state.tested_order),
            labels={
                sha: state.labels[sha].value
                for sha in state.tested_order
                if sha in state.labels
            },
            pick_log=list(state.pick_log),
            drift_records=list(state.drift_records),
        )

    def measured(sha: str, runs: int) -> TrialOutcome:
        state.tested_order.append(sha)
        return measure(sha, runs)

    def stable_endpoint(sha: str, label: str) -> TrialOutcome:
        """Measure an endpoint; re-measure once if samples look contaminated.

        Keeps the tighter of two sets and logs the swap. The retry bypasses
        ``measured`` so it costs a trial without entering the verdict order.
        """
        first = measured(sha, p.min_runs)
        if not _fresh(first) or _spread(first.samples) <= p.baseline_max_spread:
            return first
        second = measure(sha, p.min_runs)
        if not _fresh(second):
            return first
        first_spread, second_spread = _spread(first.samples), _spread(second.samples)
        trimmed = _trim_collapsed([*first.samples, *second.samples], p)
        if trimmed is not None and _spread(trimmed) < min(first_spread, second_spread):
            dropped = len(first.samples) + len(second.samples) - len(trimmed)
            state.pick_log.append(
                f"endpoint re-measured for {label} {sha[:9]}: both sets noisy "
                f"({first_spread:.2f}, {second_spread:.2f}); pooled, dropped "
                f"{dropped} run(s) >{p.baseline_max_spread / 2:.0%} from median"
            )
            return TrialOutcome(samples=trimmed)
        kept = "second" if second_spread < first_spread else "first"
        state.pick_log.append(
            f"endpoint re-measured for {label} {sha[:9]}: spread "
            f"{first_spread:.2f} -> {second_spread:.2f}, kept {kept}"
        )
        return second if kept == "second" else first

    good_sha, bad_sha = items[0], items[-1]
    good_trial = stable_endpoint(good_sha, "good")
    bad_trial = stable_endpoint(bad_sha, "bad")
    for _label, _trial in (("good", good_trial), ("bad", bad_trial)):
        if not _fresh(_trial) and _trial.reason.startswith("trial error"):
            return snapshot(
                "endpoint_error",
                False,
                f"endpoint {_label} errored: {_trial.reason}",
            )
    if not _fresh(good_trial) or not _fresh(bad_trial):
        return snapshot("no_regression", False, "endpoint skipped: cannot confirm")
    state.samples[good_sha] = list(good_trial.samples)
    state.samples[bad_sha] = list(bad_trial.samples)
    state.labels[good_sha] = CommitLabel.GOOD
    state.labels[bad_sha] = CommitLabel.BAD

    try:
        verdict = confirm_regression(
            state.samples[bad_sha], state.samples[good_sha], p.confidence
        )
    except ValueError as exc:
        return snapshot("no_regression", False, f"endpoint samples invalid: {exc}")
    if verdict.label == "no_regression":
        return snapshot("no_regression", False, verdict.reason)
    state.full_effect = verdict.effect

    while state.bad_idx - state.good_idx > 1:
        if not open_range(state):
            break
        choice = pick_next(state, pick)
        state.fallbacks += 1 if choice.fallback_used else 0
        state.pick_log.append(choice.reason)
        sha = choice.sha
        idx = items.index(sha)
        trial = measured(sha, p.min_runs)
        if not _fresh(trial):
            state.labels[sha] = CommitLabel.SKIPPED
            continue
        samples = list(trial.samples)
        runs_used = p.min_runs
        verdict = _safe_classify(
            samples,
            state.samples[items[state.good_idx]],
            state.full_effect,
            p.confidence,
        )
        hit_skip = False
        while verdict.label == "inconclusive" and runs_used < p.max_runs:
            try:
                wanted = int(extra_fn(sha, runs_used, verdict))
            except Exception as exc:
                wanted = DEFAULT_EXTRA_RUNS
                state.fallbacks += 1
                state.pick_log.append(f"extra_runs fallback for {sha}: {exc!r}")
            granted = clamp_extra_runs(wanted, runs_used, p.max_runs)
            if granted <= 0:
                break
            more = measured(sha, granted)
            if not _fresh(more):
                hit_skip = True
                break
            samples.extend(more.samples)
            runs_used += granted
            state.extra_runs_used += granted
            verdict = _safe_classify(
                samples,
                state.samples[items[state.good_idx]],
                state.full_effect,
                p.confidence,
            )
        if hit_skip:
            state.labels[sha] = CommitLabel.SKIPPED
            continue
        if verdict.label == "inconclusive":
            state.labels[sha] = CommitLabel.AMBIGUOUS
            state.samples[sha] = samples
            continue
        state.samples[sha] = samples
        if verdict.label == "good":
            state.good_idx = idx
            state.labels[sha] = CommitLabel.GOOD
        else:
            state.bad_idx = idx
            state.labels[sha] = CommitLabel.BAD
        state.steps += 1

        if state.steps % p.drift_every == 0:
            state.drift_checks += 1
            cur_good = items[state.good_idx]
            recheck = measured(cur_good, p.drift_runs)
            if _fresh(recheck):
                check = drift_exceeds(
                    state.samples[cur_good], recheck.samples, p.drift_threshold
                )
                state.drift_records.append(check)
                if check.exceeded:
                    state.endpoint_remeasures += 1
                    cur_bad = items[state.bad_idx]
                    fresh_good = stable_endpoint(cur_good, "good")
                    fresh_bad = stable_endpoint(cur_bad, "bad")
                    if _fresh(fresh_good) and _fresh(fresh_bad):
                        try:
                            reconfirm = confirm_regression(
                                list(fresh_bad.samples),
                                list(fresh_good.samples),
                                p.confidence,
                            )
                        except ValueError as exc:
                            return snapshot(
                                "no_regression",
                                False,
                                f"drift recheck samples invalid: {exc}",
                            )
                        if reconfirm.label == "no_regression":
                            return snapshot(
                                "no_regression",
                                False,
                                f"drift check: {reconfirm.reason}",
                            )
                        state.samples[cur_good] = list(fresh_good.samples)
                        state.samples[cur_bad] = list(fresh_bad.samples)
                        state.full_effect = reconfirm.effect
            else:
                state.drift_records.append(
                    DriftCheck(exceeded=False, reason="drift recheck skipped")
                )

    found = culprit_of(state)
    status: BisectStatus = "culprit_range" if found.skipped_between else "culprit"
    ref_sha = items[state.bad_idx - 1]
    culprit_trial = measured(found.culprit, p.max_runs)
    ref_trial = measured(ref_sha, p.max_runs)
    if not _fresh(culprit_trial) or not _fresh(ref_trial):
        return snapshot(
            status,
            True,
            "endpoints confirmed; verify skipped",
            culprit=found,
            verify=VerifyResult(
                verified=False,
                ref=ref_sha,
                reason="parent skipped"
                if not _fresh(ref_trial)
                else "culprit trial skipped",
            ),
        )
    verify = verify_culprit(
        list(culprit_trial.samples),
        list(ref_trial.samples),
        state.full_effect,
        ref=ref_sha,
        confidence=p.confidence,
    )
    return snapshot(
        status,
        True,
        "regression confirmed",
        culprit=found,
        verify=verify,
    )
