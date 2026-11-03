"""Proposer swarm: role agents vote on the next commit; a referee decides.

Roles only propose. The referee is deterministic and every pick still goes
through bisect's own validation, so a bad role can cost probes but can
never produce a wrong verdict.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Literal

from pydantic import BaseModel

from perfpilot.modes.bisect import BisectState, CommitLabel, open_range

Confidence = Literal["low", "medium", "high"]
WEIGHT: dict[str, int] = {"low": 1, "medium": 2, "high": 3}
HOT_PATHS = (
    "scheduler",
    "attention",
    "kv cache",
    "kv-cache",
    "cuda graph",
    "sampler",
    "sampling",
    "kernel",
    "batch",
)


class Proposal(BaseModel):
    role: str
    sha: str | None
    confidence: Confidence = "low"
    reason: str = ""


class Pick(BaseModel):
    sha: str
    score: int
    votes: list[Proposal]


Role = Callable[[BisectState, Sequence[Pick]], Proposal]


def _balance(cands: Sequence[str], sha: str) -> int:
    i = list(cands).index(sha)
    return min(i, len(cands) - 1 - i)


def referee(state: BisectState, proposals: Sequence[Proposal]) -> Pick | None:
    """Sum confidence weights per valid SHA; ties go to the most even split."""
    cands = open_range(state)
    valid = [p for p in proposals if p.sha in cands]
    if not valid:
        return None
    scores: dict[str, int] = {}
    for p in valid:
        scores[p.sha] = scores.get(p.sha, 0) + WEIGHT[p.confidence]
    best = max(scores, key=lambda s: (scores[s], _balance(cands, s), -cands.index(s)))
    return Pick(sha=best, score=scores[best], votes=list(proposals))


def bisector() -> Role:
    def role(state: BisectState, history: Sequence[Pick]) -> Proposal:
        cands = open_range(state)
        sha = cands[len(cands) // 2] if cands else None
        return Proposal(role="bisector", sha=sha, reason="halve the range")

    return role


#: A hint is only evidence if it singles commits out; past this share of
#: the open range it is noise (live risk: conventional "[Kernel]" tags).
MAX_HINT_SHARE = 0.25


def code_reader(messages: Mapping[str, str]) -> Role:
    def role(state: BisectState, history: Sequence[Pick]) -> Proposal:
        cands = open_range(state)
        hits = [
            sha
            for sha in cands
            if any(k in messages.get(sha, "").lower() for k in HOT_PATHS)
        ]
        if not hits:
            return Proposal(role="code_reader", sha=None, reason="no hot-path message")
        if len(hits) > max(1, int(MAX_HINT_SHARE * len(cands))):
            return Proposal(
                role="code_reader",
                sha=None,
                reason=f"{len(hits)}/{len(cands)} messages match: no signal",
            )
        sha = hits[-1]
        return Proposal(
            role="code_reader",
            sha=sha,
            confidence="high",
            reason=f"message touches a hot path: {messages[sha]}",
        )

    return role


def statistician() -> Role:
    """After a hint lands bad, test its parent: one probe proves the edge."""

    def role(state: BisectState, history: Sequence[Pick]) -> Proposal:
        if not history:
            return Proposal(role="statistician", sha=None, reason="no history")
        last = history[-1]
        led_by_hint = any(
            v.role == "code_reader" and v.sha == last.sha for v in last.votes
        )
        if not led_by_hint or state.labels.get(last.sha) != CommitLabel.BAD:
            return Proposal(role="statistician", sha=None, reason="nothing to confirm")
        idx = state.commits.index(last.sha)
        parent = state.commits[idx - 1] if idx > 0 else None
        if parent not in open_range(state):
            return Proposal(role="statistician", sha=None, reason="parent known")
        return Proposal(
            role="statistician",
            sha=parent,
            confidence="high",
            reason="hinted commit is bad; parent good would pin the edge",
        )

    return role


class Swarm:
    """Callable ``PickFn``. Keeps its own pick history for the roles."""

    def __init__(
        self,
        roles: Sequence[Role],
        on_pick: Callable[[Pick | None], None] | None = None,
    ) -> None:
        self.roles = list(roles)
        self.history: list[Pick] = []
        self.on_pick = on_pick

    def __call__(self, state: BisectState) -> str | None:
        proposals = [role(state, self.history) for role in self.roles]
        pick = referee(state, proposals)
        if self.on_pick is not None:
            self.on_pick(pick)
        if pick is None:
            return None
        self.history = [*self.history, pick]
        return pick.sha


def scripted_swarm(messages: Mapping[str, str]) -> Swarm:
    return Swarm([bisector(), code_reader(messages), statistician()])


def scripted_single(messages: Mapping[str, str]) -> Swarm:
    """One reasoning agent: reads messages, else halves. No peer roles."""
    return Swarm([bisector(), code_reader(messages)])
