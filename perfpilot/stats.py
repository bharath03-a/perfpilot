"""Log-ratio Welch verdicts. Pure functions: no network, no GPU."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field

VerdictLabel = Literal["good", "bad", "inconclusive", "no_regression"]

NO_REGRESSION_THRESHOLD = 0.03
TUNE_MIN_UPLIFT = 0.02


class Interval(BaseModel):
    effect: float = Field(description="r = ln(mean_c / mean_g)")
    se: float
    df: float
    lo: float
    hi: float
    confidence: float


class Verdict(BaseModel):
    label: VerdictLabel
    effect: float
    lo: float
    hi: float
    confidence: float
    reason: str


def _t_critical(df: float, confidence: float) -> float:
    try:
        from scipy.stats import t as t_dist

        return float(t_dist.ppf((1.0 + confidence) / 2.0, df))
    except ImportError:
        z = {0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}
        closest = min(z, key=lambda k: abs(k - confidence))
        return z[closest]


def _mean(xs: Sequence[float]) -> float:
    return math.fsum(xs) / len(xs)


def _sample_var(xs: Sequence[float], mean: float) -> float:
    if len(xs) < 2:
        raise ValueError(f"need >=2 samples, got {len(xs)}")
    return math.fsum((x - mean) ** 2 for x in xs) / (len(xs) - 1)


def log_ratio_interval(
    cand: Sequence[float], ref: Sequence[float], confidence: float = 0.90
) -> Interval:
    """Log-ratio Welch interval for ln(mean_cand / mean_ref)."""
    cand = list(cand)
    ref = list(ref)
    if len(cand) < 2 or len(ref) < 2:
        raise ValueError("need >=2 samples per side")
    mc, mg = _mean(cand), _mean(ref)
    if mc <= 0 or mg <= 0:
        raise ValueError("means must be positive for a log ratio")
    sc2 = _sample_var(cand, mc)
    sg2 = _sample_var(ref, mg)
    nc, ng = len(cand), len(ref)
    vc = sc2 / (nc * mc * mc)
    vg = sg2 / (ng * mg * mg)
    se = math.sqrt(vc + vg)
    effect = math.log(mc / mg)
    if se == 0.0:
        return Interval(
            effect=effect, se=0.0, df=1.0, lo=effect, hi=effect, confidence=confidence
        )
    denom = (vc * vc / (nc - 1) if vc else 0.0) + (vg * vg / (ng - 1) if vg else 0.0)
    df = (vc + vg) ** 2 / denom if denom > 0 else 1.0
    t = _t_critical(df, confidence)
    return Interval(
        effect=effect,
        se=se,
        df=df,
        lo=effect - t * se,
        hi=effect + t * se,
        confidence=confidence,
    )


def confirm_regression(
    bad: Sequence[float], good: Sequence[float], confidence: float = 0.90
) -> Verdict:
    """Endpoint check: R = ln(bad / good). Significant + >=3% or no-go."""
    iv = log_ratio_interval(bad, good, confidence)
    if iv.lo <= 0.0 <= iv.hi:
        return Verdict(
            label="no_regression",
            effect=iv.effect,
            lo=iv.lo,
            hi=iv.hi,
            confidence=confidence,
            reason="interval covers 0: no confirmed regression",
        )
    if abs(iv.effect) < NO_REGRESSION_THRESHOLD:
        return Verdict(
            label="no_regression",
            effect=iv.effect,
            lo=iv.lo,
            hi=iv.hi,
            confidence=confidence,
            reason=f"|R|={abs(iv.effect):.4f} < 3%: no confirmed regression",
        )
    return Verdict(
        label="bad",
        effect=iv.effect,
        lo=iv.lo,
        hi=iv.hi,
        confidence=confidence,
        reason="endpoints differ significantly: regression confirmed",
    )


def classify_commit(
    commit: Sequence[float],
    good: Sequence[float],
    full_effect: float,
    confidence: float = 0.90,
) -> Verdict:
    """Bisect midpoint rule with m = R / 2; else inconclusive."""
    iv = log_ratio_interval(commit, good, confidence)
    m = full_effect / 2.0
    if full_effect < 0:
        if iv.hi < m:
            return Verdict(
                label="bad",
                effect=iv.effect,
                lo=iv.lo,
                hi=iv.hi,
                confidence=confidence,
                reason=f"whole interval below m={m:.4f}: bad",
            )
        if iv.lo > m:
            return Verdict(
                label="good",
                effect=iv.effect,
                lo=iv.lo,
                hi=iv.hi,
                confidence=confidence,
                reason=f"whole interval above m={m:.4f}: good",
            )
    else:
        if iv.lo > m:
            return Verdict(
                label="bad",
                effect=iv.effect,
                lo=iv.lo,
                hi=iv.hi,
                confidence=confidence,
                reason=f"whole interval above m={m:.4f}: bad",
            )
        if iv.hi < m:
            return Verdict(
                label="good",
                effect=iv.effect,
                lo=iv.lo,
                hi=iv.hi,
                confidence=confidence,
                reason=f"whole interval below m={m:.4f}: good",
            )
    return Verdict(
        label="inconclusive",
        effect=iv.effect,
        lo=iv.lo,
        hi=iv.hi,
        confidence=confidence,
        reason=f"interval straddles m={m:.4f}: inconclusive",
    )


def tune_accepts(
    candidate: Sequence[float],
    best: Sequence[float],
    cand_p99: float,
    best_p99: float,
    confidence: float = 0.90,
    min_uplift: float = TUNE_MIN_UPLIFT,
) -> Verdict:
    """TuneSmith gate: interval fully above +2% and p99 TTFT <=10% worse."""
    iv = log_ratio_interval(candidate, best, confidence)
    if iv.lo <= min_uplift:
        return Verdict(
            label="inconclusive",
            effect=iv.effect,
            lo=iv.lo,
            hi=iv.hi,
            confidence=confidence,
            reason=f"interval not fully above +{min_uplift:.0%}: reject",
        )
    if cand_p99 > best_p99 * 1.10:
        return Verdict(
            label="inconclusive",
            effect=iv.effect,
            lo=iv.lo,
            hi=iv.hi,
            confidence=confidence,
            reason="p99 TTFT >10% worse: reject",
        )
    return Verdict(
        label="good",
        effect=iv.effect,
        lo=iv.lo,
        hi=iv.hi,
        confidence=confidence,
        reason="uplift verified: accept",
    )
