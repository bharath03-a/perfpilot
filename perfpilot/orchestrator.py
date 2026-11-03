"""Orchestrator: owns every state-changing decision.

The agent only proposes; this module validates, rejects, falls back, and
logs every call to events.jsonl. Fallbacks per SPEC.md:

pick_commit -> middle, extra_runs -> 2, explain_culprit -> "No explanation"
(culprit still counts), rank_knobs -> config order,
propose_config -> next untried, draft_fix -> skip.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from perfpilot.agent.client import (
    AgentAPIError,
    AgentValidationError,
    SparkClient,
    complete_validated,
)
from perfpilot.agent.prompts import (
    SYSTEM_PROMPTS,
    commit_table,
    history_table,
)
from perfpilot.agent.tools import (
    DecisionType,
    DraftFix,
    ExplainCulprit,
    ExtraRuns,
    PickCommit,
    ProposeConfig,
    RankKnobs,
)
from perfpilot.modes.bisect import BisectState, open_range
from perfpilot.modes.tune import CandidateRecord, KnobProposal, config_key
from perfpilot.stats import Verdict, log_ratio_interval

EVENT_TYPES = (
    "run_start",
    "trial_start",
    "trial_result",
    "verdict",
    "agent_call",
    "boundary_update",
    "culprit",
    "verify",
    "explain",
    "fix",
    "tune_candidate",
    "tune_accept",
    "quality_check",
    "run_end",
)

EXTRA_RUNS_FALLBACK = 2
NO_EXPLANATION = "No explanation"

DIFF_EXCERPT_CAP = 2000
PATCH_EXCERPT_CAP = 500


def _truncate(text: str, limit: int) -> str:
    if len(text) > limit:
        return text[:limit] + "...truncated"
    return text


def _strict_eq(a: Any, b: Any) -> bool:
    """Type-strict equality: ``1`` never equals ``True``."""
    return type(a) is type(b) and a == b


def _best_config(best: Mapping[str, Any] | str | None) -> dict[str, Any]:
    if isinstance(best, Mapping):
        return dict(best)
    return {}


def _tried_keys(
    best: Mapping[str, Any], tried: Collection[Any] | None
) -> set[tuple[tuple[str, str], ...]]:
    """Normalize ``tried`` to full-config keys matching ``tune.config_key``.

    Accepts full configs (mappings), full-config keys (tuples of pairs),
    and legacy ``(knob, value)`` pairs (resolved against ``best``).
    """
    keys: set[tuple[tuple[str, str], ...]] = set()
    base = dict(best)
    for entry in tried or []:
        if isinstance(entry, Mapping):
            keys.add(config_key(dict(entry)))
        elif (
            isinstance(entry, (tuple, list))
            and len(entry) > 0
            and all(isinstance(e, (tuple, list)) and len(e) == 2 for e in entry)
        ):
            keys.add(tuple(tuple(e) for e in entry))  # type: ignore[arg-type]
        elif (
            isinstance(entry, (tuple, list))
            and len(entry) == 2
            and isinstance(entry[0], str)
        ):
            knob, value = entry
            keys.add(config_key({**base, knob: value}))
    return keys


@dataclass
class Decision[T]:
    value: T
    accepted: bool
    why: str


class EventLog:
    """Append-only events.jsonl writer; every event has ts/run_id/type."""

    def __init__(self, run_id: str, events_path: str | Path) -> None:
        self.run_id = run_id
        self.path = Path(events_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, type: str, **fields: Any) -> dict[str, Any]:
        import json as _json

        event = {
            "ts": datetime.now(UTC).isoformat(),
            "run_id": self.run_id,
            "type": type,
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(_json.dumps(event) + "\n")
        return event


LEARNINGS_CAP = 12
LEARNING_LEN = 160


class AgentOrchestrator:
    """Validates agent proposals for modes/ (bisect.py, tune.py).

    Every accepted or rejected decision appends one short learning; the
    most recent ones travel with each new prompt so the brain sees what
    already happened this run instead of deciding one-shot every time.
    """

    def __init__(
        self, run_id: str, events_path: str | Path, client: SparkClient | None = None
    ) -> None:
        self.events = EventLog(run_id, events_path)
        self.client = client or SparkClient()
        self.learnings: list[str] = []

    def _remember(self, text: str) -> None:
        """One compact learning; oldest drop off past the cap."""
        text = " ".join(text.split())
        if len(text) > LEARNING_LEN:
            text = text[:LEARNING_LEN] + "…"
        self.learnings.append(text)
        del self.learnings[:-LEARNINGS_CAP]

    def _context(self) -> str:
        if not self.learnings:
            return ""
        lines = "\n".join(f"- {item}" for item in self.learnings)
        return f"\n\nEarlier this run:\n{lines}"

    def _ask[
        T: ExplainCulprit
        | ExtraRuns
        | PickCommit
        | ProposeConfig
        | RankKnobs
        | DraftFix
    ](
        self, decision: DecisionType, user: str, model_cls: type[T]
    ) -> tuple[T | None, str]:
        prompt = user + self._context()
        try:
            proposal = complete_validated(
                system=SYSTEM_PROMPTS[decision],
                user=prompt,
                response_model=model_cls,
                transport=self.client.transport,
                api_key=self.client._api_key,
                base_url=self.client.base_url,
                model=self.client.model,
                timeout_s=self.client.timeout_s,
            )
            return proposal, "agent proposal validated"
        except (AgentAPIError, AgentValidationError) as e:
            return None, f"{type(e).__name__}: {e}"

    def _log(
        self,
        decision: DecisionType,
        proposal: Any,
        accepted: bool,
        why: str,
        fallback: Any = None,
    ) -> None:
        event: dict[str, Any] = {
            "decision": decision,
            "proposal": proposal,
            "accepted": accepted,
            "why": why,
        }
        if not accepted:
            event["fallback"] = fallback
        self.events.emit("agent_call", **event)

    @staticmethod
    def middle_commit(candidates: Sequence[str]) -> str:
        if not candidates:
            raise ValueError("pick_commit: empty candidate range")
        return list(candidates)[len(candidates) // 2]

    def pick_commit(
        self,
        candidates: Sequence[str],
        messages: Mapping[str, str] | None = None,
        history: Sequence[Mapping[str, Any]] | None = None,
        effect_pct: float = 0.0,
        metric: str = "output_throughput",
        good: str = "",
        bad: str = "",
    ) -> Decision[str]:
        cands = list(candidates)
        try:
            fallback = self.middle_commit(cands)
        except ValueError as e:
            why = f"empty candidate range: {e}"
            self.events.emit(
                "agent_call",
                decision="pick_commit",
                proposal=None,
                accepted=False,
                why=why,
                fallback=None,
            )
            raise
        user = (
            f"The regression is a {effect_pct}% drop in {metric} "
            f"between {good} and {bad}.\nUntested commits, oldest first:\n"
            f"{commit_table(cands, messages)}\n"
            f"Results so far:\n{history_table(list(history or []))}"
        )
        proposal, note = self._ask("pick_commit", user, PickCommit)
        if proposal is None:
            why = f"{note}; using middle"
            self._log("pick_commit", None, False, why, fallback)
            return Decision(fallback, False, why)
        if proposal.sha not in cands:
            why = f"sha {proposal.sha} outside range; using middle {fallback}"
            self._log("pick_commit", proposal.model_dump(), False, why, fallback)
            self._remember(f"rejected pick {proposal.sha}: outside range")
            return Decision(fallback, False, why)
        self._log("pick_commit", proposal.model_dump(), True, proposal.reason)
        self._remember(f"picked {proposal.sha}: {proposal.reason}")
        return Decision(proposal.sha, True, proposal.reason)

    def extra_runs(
        self,
        current_runs: int,
        max_runs: int,
        lo: float = 0.0,
        hi: float = 0.0,
        midpoint: float = 0.0,
    ) -> Decision[int]:
        budget = max_runs - current_runs
        if budget <= 0:
            self.events.emit(
                "agent_call",
                decision="extra_runs",
                proposal=None,
                accepted=False,
                why="at max_runs cap",
                fallback=0,
            )
            return Decision(0, False, "at max_runs cap")
        fallback = min(EXTRA_RUNS_FALLBACK, budget)
        user = (
            f"Current runs: {current_runs} (max {max_runs}). "
            f"Interval: [{lo:.4f}, {hi:.4f}] around midpoint m = {midpoint:.4f}."
        )
        proposal, note = self._ask("extra_runs", user, ExtraRuns)
        if proposal is None:
            why = f"{note}; using {fallback}"
            self._log("extra_runs", None, False, why, fallback)
            return Decision(fallback, False, why)
        runs = min(proposal.runs, budget)
        if runs != proposal.runs:
            why = f"clamped {proposal.runs} to budget {budget}"
            self._log("extra_runs", proposal.model_dump(), False, why, runs)
            self._remember(f"clamped extra runs {proposal.runs} to budget {budget}")
            return Decision(runs, False, why)
        self._log("extra_runs", proposal.model_dump(), True, proposal.reason)
        self._remember(f"granted {runs} extra runs: {proposal.reason}")
        return Decision(runs, True, proposal.reason)

    def explain_culprit(
        self,
        sha: str,
        message: str,
        diff_paths: Collection[str],
        diff_excerpt: str = "",
        metric: str = "output_throughput",
        good_mean: float = 0.0,
        bad_mean: float = 0.0,
        effect: float = 0.0,
    ) -> Decision[ExplainCulprit]:
        fallback = ExplainCulprit(
            summary=NO_EXPLANATION, mechanism="", files=[], confidence="low"
        )
        excerpt = _truncate(diff_excerpt, DIFF_EXCERPT_CAP)
        user = (
            f"Culprit commit: {sha} ({message}).\nBefore/after {metric}: "
            f"good mean {good_mean:.2f}, bad mean {bad_mean:.2f}, effect {effect}.\n"
            f"Diff paths: {sorted(diff_paths)}\nExcerpt:\n{excerpt}"
        )
        proposal, note = self._ask("explain_culprit", user, ExplainCulprit)
        if proposal is None:
            why = f"{note}; culprit still counts"
            self._log("explain_culprit", None, False, why, fallback.model_dump())
            return Decision(fallback, False, why)
        uncited = [p for p in proposal.files if p not in set(diff_paths)]
        if uncited:
            why = f"cited paths not in diff: {uncited}; culprit still counts"
            self._log(
                "explain_culprit",
                proposal.model_dump(),
                False,
                why,
                fallback.model_dump(),
            )
            return Decision(fallback, False, why)
        self._log("explain_culprit", proposal.model_dump(), True, note)
        self._remember(f"explained {sha}: {proposal.summary}")
        return Decision(proposal, True, note)

    def rank_knobs(
        self,
        knobs: Mapping[str, Sequence[Any]],
        baseline: str | Mapping[str, Any] = "",
        metric: str = "output_throughput",
        constraints: str = "",
    ) -> Decision[list[str]]:
        fallback = list(knobs.keys())
        user = (
            f"Baseline {metric}: {baseline}.\n"
            f"Supported knobs and values: {dict(knobs)}."
        ) + _constraints_line(constraints)
        proposal, note = self._ask("rank_knobs", user, RankKnobs)
        if proposal is None:
            why = f"{note}; using config order"
            self._log("rank_knobs", None, False, why, fallback)
            return Decision(fallback, False, why)
        unknown = [k for k in proposal.order if k not in knobs]
        if unknown or not proposal.order:
            why = f"unknown/empty knobs {unknown}; using config order"
            self._log("rank_knobs", proposal.model_dump(), False, why, fallback)
            return Decision(fallback, False, why)
        self._log("rank_knobs", proposal.model_dump(), True, proposal.reason)
        self._remember(f"knob order: {proposal.order}")
        return Decision(proposal.order, True, proposal.reason)

    @staticmethod
    def next_untried(
        knobs: Mapping[str, Sequence[Any]],
        tried: Collection[Any] = (),
        knob_order: Sequence[str] | None = None,
        best: Mapping[str, Any] | str | None = None,
    ) -> ProposeConfig:
        """Next untried one-knob change; full-config granularity like tune."""
        base = _best_config(best)
        tried_keys = _tried_keys(base, tried)
        rest = [k for k in knobs if k not in (knob_order or [])]
        for knob in list(knob_order or []) + rest:
            for value in knobs.get(knob, []):
                if knob in base and _strict_eq(value, base[knob]):
                    continue
                if config_key({**base, knob: value}) in tried_keys:
                    continue
                return ProposeConfig(
                    knob=knob,
                    value=value,
                    reason="next untried in order (fallback)",
                )
        raise ValueError("propose_config: no untried configs left")

    def propose_config(
        self,
        knobs: Mapping[str, Sequence[Any]],
        tried: Collection[Any] = (),
        knob_order: Sequence[str] | None = None,
        best: Mapping[str, Any] | str | None = None,
        history: Sequence[Mapping[str, Any]] | None = None,
        constraints: str = "",
    ) -> Decision[ProposeConfig | None]:
        base = _best_config(best)
        tried_keys = _tried_keys(base, tried)
        try:
            fallback = self.next_untried(knobs, tried_keys, knob_order, base)
        except ValueError:
            why = "no untried configs"
            self._log("propose_config", None, False, why, None)
            return Decision(None, False, why)
        user = (
            f"Best config so far: {base}.\n"
            f"Already tried: {sorted(tried_keys, key=str)}.\n"
            f"Allowed values: {dict(knobs)}.\n"
            f"Results so far:\n{history_table(list(history or []))}"
        ) + _constraints_line(constraints)
        proposal, note = self._ask("propose_config", user, ProposeConfig)
        if proposal is None:
            why = f"{note}; using {fallback.knob}={fallback.value}"
            self._log("propose_config", None, False, why, fallback.model_dump())
            return Decision(fallback, False, why)
        if proposal.knob not in knobs:
            why = (
                f"knob {proposal.knob} not in config; "
                f"using {fallback.knob}={fallback.value}"
            )
            self._log(
                "propose_config",
                proposal.model_dump(),
                False,
                why,
                fallback.model_dump(),
            )
            return Decision(fallback, False, why)
        if not any(_strict_eq(proposal.value, v) for v in knobs[proposal.knob]):
            why = (
                f"value {proposal.value!r} not in config for {proposal.knob}; "
                f"using {fallback.knob}={fallback.value}"
            )
            self._log(
                "propose_config",
                proposal.model_dump(),
                False,
                why,
                fallback.model_dump(),
            )
            return Decision(fallback, False, why)
        if proposal.knob in base and _strict_eq(proposal.value, base[proposal.knob]):
            why = (
                f"{proposal.knob}={proposal.value!r} is a no-op vs best; "
                f"using {fallback.knob}={fallback.value}"
            )
            self._log(
                "propose_config",
                proposal.model_dump(),
                False,
                why,
                fallback.model_dump(),
            )
            return Decision(fallback, False, why)
        if config_key({**base, proposal.knob: proposal.value}) in tried_keys:
            why = (
                f"{proposal.knob}={proposal.value!r} already tried; "
                f"using {fallback.knob}={fallback.value}"
            )
            self._log(
                "propose_config",
                proposal.model_dump(),
                False,
                why,
                fallback.model_dump(),
            )
            return Decision(fallback, False, why)
        self._log("propose_config", proposal.model_dump(), True, proposal.reason)
        self._remember(f"tried {proposal.knob}={proposal.value!r}: {proposal.reason}")
        return Decision(proposal, True, proposal.reason)

    def draft_fix(self, sha: str, diff: str = "") -> Decision[DraftFix | None]:
        user = f"Culprit commit: {sha}. Diff:\n{_truncate(diff, DIFF_EXCERPT_CAP)}"
        proposal, note = self._ask("draft_fix", user, DraftFix)
        if proposal is None:
            why = f"{note}; skipping fix"
            self._log("draft_fix", None, False, why, {"skipped": True})
            return Decision(None, False, why)
        if not proposal.patch.strip():
            why = "empty patch; skipping fix"
            self._log(
                "draft_fix",
                proposal.model_dump(),
                False,
                why,
                {"skipped": True},
            )
            return Decision(None, False, why)
        excerpt = _truncate(proposal.patch, PATCH_EXCERPT_CAP)
        self._log(
            "draft_fix",
            {"patch_len": len(proposal.patch), "patch_excerpt": excerpt},
            True,
            note,
        )
        return Decision(proposal, True, note)


#: Known capability gaps the brain otherwise ignores (live run proposed fp8 on A100).
GPU_NOTES: dict[str, str] = {
    "A100": "Ampere sm80: no FP8 compute or FP8 KV cache",
    "A10G": "Ampere sm86: no FP8, 24GB memory",
    "L4": "Ada sm89: FP8 supported, 24GB memory",
    "H100": "Hopper sm90: FP8 supported",
}


def hardware_note(gpu: str) -> str:
    if not gpu:
        return ""
    note = GPU_NOTES.get(gpu.upper())
    return f"Hardware: {gpu} ({note})." if note else f"Hardware: {gpu}."


def _constraints_line(constraints: str) -> str:
    return (
        f"\n{constraints} Do not propose values this hardware cannot run."
        if constraints
        else ""
    )


def bisect_history(state: BisectState) -> list[dict[str, Any]]:
    """Measured evidence per tested commit, effect = ln(commit / good)."""
    good = state.samples.get(state.commits[state.good_idx], [])
    rows: list[dict[str, Any]] = []
    for sha in state.tested_order:
        samples = state.samples.get(sha, [])
        label = state.labels.get(sha)
        row: dict[str, Any] = {
            "commit": sha,
            "runs": len(samples),
            "label": label.value if label is not None else "?",
        }
        try:
            iv = log_ratio_interval(samples, good)
            row = {
                **row,
                "effect": round(iv.effect, 3),
                "lo": round(iv.lo, 3),
                "hi": round(iv.hi, 3),
            }
        except ValueError:
            pass  # too few samples for an interval; row still shows label/runs
        rows.append(row)
    return rows


def candidate_history(candidates: Sequence[CandidateRecord]) -> list[dict[str, Any]]:
    """Measured uplift per tried flag so the brain stops re-proposing duds."""
    return [
        {
            "commit": f"{c.knob}={c.value!r}",
            "effect": round(c.effect, 3),
            "lo": round(c.lo, 3),
            "hi": round(c.hi, 3),
            "label": f"{c.decision}: {c.reason}"[:80],
        }
        for c in candidates
    ]


def make_pick_fn(
    orch: AgentOrchestrator,
    metric: str = "output_throughput",
    messages: Mapping[str, str] | None = None,
) -> Callable[[BisectState], str | None]:
    """Adapt ``pick_commit`` to bisect's ``PickFn``.

    Rejected proposals return ``None`` so ``bisect.pick_next`` owns the
    middle-commit fallback and the fallback count.
    """

    def pick(state: BisectState) -> str | None:
        cands = open_range(state)
        if not cands:
            return None
        history = bisect_history(state)
        decision = orch.pick_commit(
            cands,
            messages=messages,
            history=history,
            effect_pct=state.full_effect * 100.0,
            metric=metric,
            good=state.commits[state.good_idx],
            bad=state.commits[state.bad_idx],
        )
        if not decision.accepted:
            return None
        return decision.value

    return pick


def make_extra_runs_fn(
    orch: AgentOrchestrator, max_runs: int = 8
) -> Callable[[str, int, Verdict], int]:
    """Adapt ``extra_runs`` to bisect's ``ExtraRunsFn``."""

    def extra(commit_sha: str, runs_used: int, verdict: Verdict) -> int:
        midpoint = (
            verdict.midpoint
            if verdict.midpoint is not None
            else (verdict.lo + verdict.hi) / 2.0
        )
        decision = orch.extra_runs(
            current_runs=runs_used,
            max_runs=max_runs,
            lo=verdict.lo,
            hi=verdict.hi,
            midpoint=midpoint,
        )
        return decision.value

    return extra


def make_propose_fn(
    orch: AgentOrchestrator,
    knob_space: Mapping[str, Sequence[Any]],
    baseline: Mapping[str, Any] | str | None = None,
    metric: str = "output_throughput",
    hardware: str = "",
) -> Callable[[Mapping[str, Any], list[CandidateRecord]], KnobProposal | None]:
    """Adapt ``propose_config`` to tune's ``ProposeFn``.

    Consumes the ``rank_knobs`` order (config-order fallback), so the rank
    decision has a call path. ``Decision`` ``None`` stays ``None`` (stop).
    """
    space = dict(knob_space)
    constraints = hardware_note(hardware)
    knob_order = orch.rank_knobs(
        space, baseline=baseline or "", metric=metric, constraints=constraints
    ).value

    def propose(
        best: Mapping[str, Any], candidates: list[CandidateRecord]
    ) -> KnobProposal | None:
        tried = {config_key(dict(best))}
        for record in candidates:
            if record.config:
                tried.add(config_key(dict(record.config)))
        decision = orch.propose_config(
            space,
            tried=tried,
            knob_order=knob_order,
            best=best,
            history=candidate_history(candidates),
            constraints=constraints,
        )
        if decision.value is None:
            return None
        return KnobProposal(
            knob=decision.value.knob,
            value=decision.value.value,
            reason=decision.value.reason,
        )

    return propose
