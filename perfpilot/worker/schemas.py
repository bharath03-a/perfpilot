"""TrialResult schemas matching SPEC.md (Data files, CLI and report)."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field

TrialStatus = Literal["ok", "skip", "error"]

VALID_STATUSES: tuple[str, str, str] = ("ok", "skip", "error")


class RunMetrics(BaseModel):
    output_throughput: float
    mean_ttft_ms: float
    p99_ttft_ms: float
    median_tpot_ms: float
    failed_requests: int = 0

    @property
    def valid(self) -> bool:
        return self.failed_requests == 0


class TrialResult(BaseModel):
    commit: str
    flags: dict[str, Any] = Field(default_factory=dict)
    status: TrialStatus
    reason: str = ""
    gpu_uuid: str = ""
    vllm_version: str = ""
    runs: list[RunMetrics] = Field(default_factory=list)
    gpu_seconds: float = 0.0

    @property
    def valid_runs(self) -> list[RunMetrics]:
        return [r for r in self.runs if r.valid]

    def to_spec_json(self) -> dict[str, Any]:
        return json.loads(self.model_dump_json())


def flags_hash(flags: dict[str, Any]) -> str:
    canonical = json.dumps(flags, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def trial_cache_key(
    commit: str, flags: dict[str, Any], inject: str | None
) -> tuple[str, str, str | None]:
    return (commit, flags_hash(flags), inject)
