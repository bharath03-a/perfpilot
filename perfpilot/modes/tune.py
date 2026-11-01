"""TuneSmith one-knob-at-a-time search.

Pure logic: no network, GPU, or subprocess. Knob support arrives as an
input set (already filtered from ``vllm serve --help``); GPU numbers enter
through the ``measure`` callback and accuracy through ``quality``. Agent
ranking enters through ``propose`` with the config-order fallback as
default. Uplift verdicts come only from :mod:`perfpilot.stats`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from perfpilot.stats import log_ratio_interval, tune_accepts

TuneConfig = dict[str, Any]


class TuneParams(BaseModel):
    min_runs: int = Field(default=3, ge=2)
    max_runs: int = Field(default=8, ge=2)
    confidence: float = Field(default=0.90, gt=0.0, lt=1.0)
    max_trials: int = Field(default=8, ge=1)
    min_uplift: float = Field(default=0.02, gt=0.0)
    max_p99_slack: float = Field(default=0.10, gt=0.0)
    max_accuracy_drop_pts: float = Field(default=2.0, ge=0.0)
    quality_knobs: set[str] = Field(
        default_factory=lambda: {"kv_cache_dtype"},
        description="Knobs that change numerics and therefore require the "
        "accuracy quality gate. The orchestrator must extend this set for any "
        "new numeric knob added to the space (e.g. quantization or dtype "
        "flags); anything missing here skips the gate and gets only the "
        "final-config check.",
    )

    @model_validator(mode="after")
    def _caps_agree(self) -> TuneParams:
        if self.max_runs < self.min_runs:
            raise ValueError("max_runs must be >= min_runs")
        return self


class TuneTrial(BaseModel):
    samples: list[float]
    p99_ttft_ms: float


class KnobProposal(BaseModel):
    knob: str
    value: Any
    reason: str = ""


class ProposalCheck(BaseModel):
    valid: bool
    reason: str = ""
    candidate: TuneConfig = Field(default_factory=dict)


class CandidateRecord(BaseModel):
    knob: str = ""
    value: Any = None
    config: TuneConfig = Field(default_factory=dict)
    effect: float = 0.0
    lo: float = 0.0
    hi: float = 0.0
    decision: Literal["accepted", "rejected"] = "rejected"
    reason: str = ""


class QualityRecord(BaseModel):
    config: TuneConfig = Field(default_factory=dict)
    accuracy: float = 0.0
    baseline_accuracy: float = 0.0
    drop_pts: float = 0.0
    passed: bool = True
    reason: str = ""


class TuneOutcome(BaseModel):
    baseline_config: TuneConfig = Field(default_factory=dict)
    best_config: TuneConfig = Field(default_factory=dict)
    uplift_effect: float = 0.0
    uplift_lo: float = 0.0
    uplift_hi: float = 0.0
    confirmed: bool = False
    confirm_reason: str = ""
    trials_used: int = 0
    candidates: list[CandidateRecord] = Field(default_factory=list)
    quality_checks: list[QualityRecord] = Field(default_factory=list)
    stopped_reason: str = ""
    serve_command: str = ""
    fallbacks: int = 0
    rejections: int = 0
    propose_log: list[str] = Field(default_factory=list)


MeasureTuneFn = Callable[[TuneConfig, int], TuneTrial]
QualityFn = Callable[[TuneConfig], float]
ProposeFn = Callable[[TuneConfig, list[CandidateRecord]], KnobProposal | None]


def config_key(config: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((k, repr(v)) for k, v in config.items()))


def default_propose(
    best: Mapping[str, Any],
    knob_space: Mapping[str, list[Any]],
    tried: set[tuple[tuple[str, str], ...]],
    knob_order: Sequence[str] | None = None,
) -> KnobProposal | None:
    """SPEC.md fallback: next untried value in rank order, one knob.

    ``knob_order`` is the agent's rank_knobs ordering; when None, config
    (dict-insertion) order is used. Names missing from the space are ignored,
    and space knobs missing from the order keep config order at the end.
    """
    current = dict(best)
    if knob_order is None:
        names = list(knob_space)
    else:
        names = [k for k in knob_order if k in knob_space]
        names += [k for k in knob_space if k not in set(names)]
    for knob in names:
        for value in knob_space[knob]:
            if value == current.get(knob):
                continue
            if config_key({**current, knob: value}) in tried:
                continue
            return KnobProposal(knob=knob, value=value, reason="config order")
    return None


def validate_proposal(
    proposal: KnobProposal,
    best: Mapping[str, Any],
    knob_space: Mapping[str, list[Any]],
    tried: set[tuple[tuple[str, str], ...]],
) -> ProposalCheck:
    """Reject out-of-config values, no-ops, and already-tried configs."""
    if proposal.knob not in knob_space:
        return ProposalCheck(
            valid=False, reason=f"unknown knob {proposal.knob!r}: reject"
        )
    if proposal.value not in knob_space[proposal.knob]:
        return ProposalCheck(
            valid=False,
            reason=f"value {proposal.value!r} not in config for "
            f"{proposal.knob}: reject",
        )
    candidate = {**dict(best), proposal.knob: proposal.value}
    if candidate == dict(best):
        return ProposalCheck(valid=False, reason="no-op change: reject")
    if config_key(candidate) in tried:
        return ProposalCheck(valid=False, reason="config already tried: reject")
    return ProposalCheck(
        valid=True, reason="one-knob change accepted", candidate=candidate
    )


def quality_passes(
    candidate_acc: float,
    baseline_acc: float,
    max_drop_pts: float = 2.0,
) -> QualityRecord:
    """Reject configs whose accuracy drops more than the allowed points."""
    drop = baseline_acc - candidate_acc
    if drop > max_drop_pts:
        return QualityRecord(
            accuracy=candidate_acc,
            baseline_accuracy=baseline_acc,
            drop_pts=drop,
            passed=False,
            reason=f"accuracy drop {drop:.2f}pts > {max_drop_pts:.2f}pts: reject",
        )
    return QualityRecord(
        accuracy=candidate_acc,
        baseline_accuracy=baseline_acc,
        drop_pts=drop,
        passed=True,
        reason="quality gate passed",
    )


def serve_command(
    model: str, config: Mapping[str, Any], *, port: int = 8000, seed: int = 0
) -> str:
    """Ready-to-paste ``vllm serve`` command for a tried config."""
    parts = [f"vllm serve {model or '<model>'}", f"--port {port}", f"--seed {seed}"]
    for knob in sorted(config):
        value = config[knob]
        flag = f"--{knob.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                parts.append(flag)
        else:
            parts.append(f"{flag} {value}")
    return " ".join(parts)


def run_tuning(
    baseline_config: Mapping[str, Any],
    knob_space: Mapping[str, list[Any]],
    measure: MeasureTuneFn,
    quality: QualityFn,
    *,
    model: str = "",
    params: TuneParams | None = None,
    propose: ProposeFn | None = None,
    knob_order: list[str] | None = None,
) -> TuneOutcome:
    """One-knob coordinate descent from the baseline config.

    ``knob_order`` is the agent's rank_knobs ordering, threaded into the
    default propose fallback instead of silent dict-insertion order.

    Adapter contract: the orchestrator unwraps agent Decision objects before
    invoking ``propose`` (make_propose_fn), so it receives plain values. Any
    non-conforming proposal that slips through -- including a raw Decision --
    fails validation and is counted as a rejection, never raised. A
    ``propose`` callback that raises falls back to ``default_propose`` and is
    counted in ``fallbacks`` with its reason in ``propose_log``.

    Invalid proposals never consume a ``max_trials`` iteration; rejections
    are counted separately in ``rejections``.
    """
    p = params or TuneParams()
    baseline = dict(baseline_config)
    space = {k: list(v) for k, v in knob_space.items()}
    tried = {config_key(baseline)}

    base_trial = measure(dict(baseline), p.min_runs)
    base_acc = quality(dict(baseline))
    quality_checks = [
        QualityRecord(
            config=dict(baseline),
            accuracy=base_acc,
            baseline_accuracy=base_acc,
            drop_pts=0.0,
            passed=True,
            reason="baseline reference",
        )
    ]
    checked = {config_key(baseline)}
    best = dict(baseline)
    best_trial = base_trial
    candidates: list[CandidateRecord] = []
    trials_used = 0
    fallbacks = 0
    rejections = 0
    propose_log: list[str] = []
    stopped_reason = "max_trials reached"

    while trials_used < p.max_trials:
        if propose is not None:
            try:
                proposal = propose(dict(best), list(candidates))
            except Exception as exc:
                fallbacks += 1
                propose_log.append(f"propose raised {exc!r}: config-order fallback")
                proposal = default_propose(best, space, tried, knob_order)
        else:
            proposal = default_propose(best, space, tried, knob_order)
        if proposal is None:
            stopped_reason = "no further proposals"
            break
        check = validate_proposal(proposal, best, space, tried)
        if not check.valid:
            rejections += 1
            propose_log.append(
                f"rejected {proposal.knob}={proposal.value!r}: {check.reason}"
            )
            candidates.append(
                CandidateRecord(
                    knob=proposal.knob, value=proposal.value, reason=check.reason
                )
            )
            if rejections >= p.max_trials:
                stopped_reason = "too many rejected proposals"
                break
            continue
        cand = dict(check.candidate)
        tried.add(config_key(cand))
        try:
            trial = measure(dict(cand), p.min_runs)
        except Exception as exc:
            trials_used += 1
            reason = f"measure failed: {type(exc).__name__}"
            propose_log.append(f"rejected {proposal.knob}={proposal.value!r}: {reason}")
            candidates.append(
                CandidateRecord(
                    knob=proposal.knob,
                    value=proposal.value,
                    config=cand,
                    reason=reason,
                )
            )
            continue
        trials_used += 1
        try:
            verdict = tune_accepts(
                list(trial.samples),
                list(best_trial.samples),
                trial.p99_ttft_ms,
                best_trial.p99_ttft_ms,
                p.confidence,
                p.min_uplift,
            )
        except ValueError as exc:
            reason = f"verdict failed: {type(exc).__name__}: {exc}"
            propose_log.append(f"rejected {proposal.knob}={proposal.value!r}: {reason}")
            candidates.append(
                CandidateRecord(
                    knob=proposal.knob,
                    value=proposal.value,
                    config=cand,
                    reason=reason,
                )
            )
            continue
        if verdict.label != "good":
            candidates.append(
                CandidateRecord(
                    knob=proposal.knob,
                    value=proposal.value,
                    config=cand,
                    effect=verdict.effect,
                    lo=verdict.lo,
                    hi=verdict.hi,
                    reason=verdict.reason,
                )
            )
            continue
        if proposal.knob in p.quality_knobs:
            try:
                candidate_acc = quality(dict(cand))
            except Exception as exc:
                reason = f"quality failed: {type(exc).__name__}"
                propose_log.append(
                    f"rejected {proposal.knob}={proposal.value!r}: {reason}"
                )
                candidates.append(
                    CandidateRecord(
                        knob=proposal.knob,
                        value=proposal.value,
                        config=cand,
                        reason=reason,
                    )
                )
                continue
            record = quality_passes(candidate_acc, base_acc, p.max_accuracy_drop_pts)
            record.config = dict(cand)
            quality_checks.append(record)
            checked.add(config_key(cand))
            if not record.passed:
                candidates.append(
                    CandidateRecord(
                        knob=proposal.knob,
                        value=proposal.value,
                        config=cand,
                        effect=verdict.effect,
                        lo=verdict.lo,
                        hi=verdict.hi,
                        reason=record.reason,
                    )
                )
                continue
        best = cand
        best_trial = trial
        propose_log.append(
            f"accepted {proposal.knob}={proposal.value!r}: {verdict.reason}"
        )
        candidates.append(
            CandidateRecord(
                knob=proposal.knob,
                value=proposal.value,
                config=cand,
                effect=verdict.effect,
                lo=verdict.lo,
                hi=verdict.hi,
                decision="accepted",
                reason=verdict.reason,
            )
        )

    if config_key(best) not in checked:
        try:
            best_acc = quality(dict(best))
        except Exception as exc:
            best_acc = None
            propose_log.append(f"final quality failed: {type(exc).__name__}")
        if best_acc is None:
            record = QualityRecord(
                passed=False,
                reason="final quality failed: unevaluated",
            )
        else:
            record = quality_passes(best_acc, base_acc, p.max_accuracy_drop_pts)
        record.config = dict(best)
        quality_checks.append(record)
        checked.add(config_key(best))
        if not record.passed:
            for cand in candidates:
                if cand.decision == "accepted":
                    cand.decision = "rejected"
                    cand.reason = f"final quality failed: {record.reason}"
            propose_log.append(f"final quality failed: {record.reason}: reverted")
            best = dict(baseline)
            best_trial = base_trial

    if best == baseline:
        return TuneOutcome(
            baseline_config=dict(baseline),
            best_config=dict(best),
            trials_used=trials_used,
            candidates=candidates,
            quality_checks=quality_checks,
            stopped_reason=stopped_reason,
            serve_command=serve_command(model, best),
            confirm_reason="no candidate beat baseline: no confirmation needed",
            fallbacks=fallbacks,
            rejections=rejections,
            propose_log=list(propose_log),
        )
    try:
        confirm_best = measure(dict(best), p.max_runs)
        confirm_base = measure(dict(baseline), p.max_runs)
    except Exception as exc:
        propose_log.append(f"confirmation failed: {type(exc).__name__}")
        return TuneOutcome(
            baseline_config=dict(baseline),
            best_config=dict(best),
            trials_used=trials_used,
            candidates=candidates,
            quality_checks=quality_checks,
            stopped_reason=stopped_reason,
            serve_command=serve_command(model, best),
            confirm_reason="confirmation unmeasurable: unevaluated",
            fallbacks=fallbacks,
            rejections=rejections,
            propose_log=list(propose_log),
        )
    try:
        iv = log_ratio_interval(
            list(confirm_best.samples), list(confirm_base.samples), p.confidence
        )
    except ValueError as exc:
        return TuneOutcome(
            baseline_config=dict(baseline),
            best_config=dict(best),
            uplift_effect=0.0,
            uplift_lo=0.0,
            uplift_hi=0.0,
            confirmed=False,
            confirm_reason=f"confirmation samples invalid: {exc}",
            trials_used=trials_used,
            candidates=candidates,
            quality_checks=quality_checks,
            stopped_reason=stopped_reason,
            serve_command=serve_command(model, best),
            fallbacks=fallbacks,
            rejections=rejections,
            propose_log=list(propose_log),
        )
    confirmed = iv.lo > p.min_uplift
    return TuneOutcome(
        baseline_config=dict(baseline),
        best_config=dict(best),
        uplift_effect=iv.effect,
        uplift_lo=iv.lo,
        uplift_hi=iv.hi,
        confirmed=confirmed,
        confirm_reason=(
            "uplift confirmed at max_runs"
            if confirmed
            else f"confirm interval not fully above +{p.min_uplift:.0%}"
        ),
        trials_used=trials_used,
        candidates=candidates,
        quality_checks=quality_checks,
        stopped_reason=stopped_reason,
        serve_command=serve_command(model, best),
        fallbacks=fallbacks,
        rejections=rejections,
        propose_log=list(propose_log),
    )
