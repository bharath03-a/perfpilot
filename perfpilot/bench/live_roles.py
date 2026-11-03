"""LLM-backed swarm roles: same referee, personas over the live brain."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import BaseModel

from perfpilot.agent.prompts import commit_table, history_table
from perfpilot.bench.swarm import Pick, Proposal, Role, Swarm, bisector
from perfpilot.modes.bisect import BisectState, open_range
from perfpilot.orchestrator import AgentOrchestrator, bisect_history

PERSONAS: dict[str, str] = {
    "code_reader": (
        "You read commit messages. Pick the commit most likely to slow "
        "LLM serving throughput (scheduler, attention, KV cache, CUDA graphs, "
        "sampling, kernels). Say high confidence only if a message clearly fits."
    ),
    "statistician": (
        "You read the measured table. Pick the commit whose result would most "
        "shrink the suspect range; if a hinted commit was just labeled bad, "
        "pick its parent to confirm the transition edge."
    ),
}


class RoleVote(BaseModel):
    sha: str
    confidence: str = "low"
    reason: str = ""


def llm_role(name: str, orch: AgentOrchestrator, messages: Mapping[str, str]) -> Role:
    persona = PERSONAS[name]

    def role(state: BisectState, history: Sequence[Pick]) -> Proposal:
        cands = open_range(state)
        user = (
            f"Role: {persona}\nUntested commits, oldest first:\n"
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

    roles = [
        bisector(),
        llm_role("code_reader", orch, messages),
        llm_role("statistician", orch, messages),
    ]
    return Swarm(roles, on_pick=log)
