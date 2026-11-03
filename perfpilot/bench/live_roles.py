"""LLM-backed swarm roles: same referee, personas over the live brain."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import BaseModel

from perfpilot.agent.prompts import commit_table, history_table
from perfpilot.bench.swarm import (
    HOT_PATHS,
    Pick,
    Proposal,
    Role,
    Swarm,
    bisector,
    statistician,
)
from perfpilot.modes.bisect import BisectState, open_range
from perfpilot.orchestrator import AgentOrchestrator, bisect_history

PERSONAS: dict[str, str] = {
    "code_reader": (
        "You read commit messages. Pick the commit most likely to slow "
        "LLM serving throughput (scheduler, attention, KV cache, CUDA graphs, "
        "sampling, kernels). Say high confidence only if a message clearly fits."
    ),
    "statistician": (
        "You read the measured table. Rule 1: if the last probe was a commit "
        "picked because of its message and it measured bad, pick its parent "
        "(the commit just before it) with high confidence: if the parent is "
        "good, the culprit is proven in one probe. Otherwise pick the commit "
        "that best halves the open range, low confidence."
    ),
}


class RoleVote(BaseModel):
    sha: str
    confidence: str = "low"
    reason: str = ""


def _last_probe(state: BisectState, history: Sequence[Pick]) -> str:
    if not history:
        return "Last probe: none yet.\n"
    last = history[-1]
    label = state.labels.get(last.sha)
    roles = sorted({v.role for v in last.votes if v.sha == last.sha})
    idx = state.commits.index(last.sha)
    parent = state.commits[idx - 1] if idx > 0 else "none"
    return (
        f"Last probe: {last.sha} measured "
        f"{label.value if label else 'unknown'}, chosen by {roles}; "
        f"its parent is {parent}.\n"
    )


def llm_role(name: str, orch: AgentOrchestrator, messages: Mapping[str, str]) -> Role:
    persona = PERSONAS[name]

    def role(state: BisectState, history: Sequence[Pick]) -> Proposal:
        cands = open_range(state)
        user = (
            f"Role: {persona}\n{_last_probe(state, history)}"
            f"Untested commits, oldest first:\n"
            f"{commit_table(cands, messages)}\n"
            f"Results so far:\n{history_table(bisect_history(state))}\n"
            'Return JSON only: {"sha": "<from list>", "confidence": '
            '"<low|medium|high>", "reason": "<one sentence>"}'
        )
        vote, note = orch._ask("pick_commit", user, RoleVote)
        if vote is None:
            return Proposal(role=name, sha=None, reason=note)
        conf = (
            vote.confidence if vote.confidence in ("low", "medium", "high") else "low"
        )
        msg = messages.get(vote.sha, "").lower()
        if conf != "low" and not any(k in msg for k in HOT_PATHS):
            # live bench: ungrounded "high" votes turned noisy into a linear walk
            conf = "low"
        return Proposal(role=name, sha=vote.sha, confidence=conf, reason=vote.reason)

    return role


def live_swarm(orch: AgentOrchestrator, messages: Mapping[str, str]) -> Swarm:
    def log(pick: Pick | None) -> None:
        orch.events.emit(
            "swarm_pick",
            chosen=pick.sha if pick else None,
            score=pick.score if pick else 0,
            votes=[v.model_dump() for v in pick.votes] if pick else [],
        )

    # LLM where the signal is semantic (messages); arithmetic stays deterministic
    roles = [bisector(), llm_role("code_reader", orch, messages), statistician()]
    return Swarm(roles, on_pick=log)
