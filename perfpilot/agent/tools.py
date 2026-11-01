"""JSON schemas (pydantic) for the agent's allowed actions."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

DecisionType = Literal[
    "pick_commit",
    "extra_runs",
    "explain_culprit",
    "rank_knobs",
    "propose_config",
    "draft_fix",
]


class PickCommit(BaseModel):
    sha: str
    reason: str


class ExtraRuns(BaseModel):
    runs: int = Field(ge=1, le=3)
    reason: str


class ExplainCulprit(BaseModel):
    summary: str
    mechanism: str
    files: list[str] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"] = "low"


class RankKnobs(BaseModel):
    order: list[str]
    reason: str


class ProposeConfig(BaseModel):
    knob: str
    value: Any
    reason: str


class DraftFix(BaseModel):
    patch: str


DECISION_MODELS: dict[str, type[BaseModel]] = {
    "pick_commit": PickCommit,
    "extra_runs": ExtraRuns,
    "explain_culprit": ExplainCulprit,
    "rank_knobs": RankKnobs,
    "propose_config": ProposeConfig,
    "draft_fix": DraftFix,
}


def schema_for(decision: DecisionType) -> dict[str, Any]:
    return DECISION_MODELS[decision].model_json_schema()
