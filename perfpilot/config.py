"""Pydantic models for perfpilot.yaml. Secrets come from env only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Workload(BaseModel):
    dataset: str = "random"
    input_len: int = 1024
    output_len: int = 128
    num_prompts: int = 100
    max_concurrency: int = 32
    seed: int = 0


class Metrics(BaseModel):
    primary: str = "output_throughput"
    guards: list[str] = Field(default_factory=lambda: ["p99_ttft_ms"])


class StatsConfig(BaseModel):
    min_runs: int = 3
    max_runs: int = 8
    confidence: float = 0.90


class BisectConfig(BaseModel):
    repo: str = "vllm-project/vllm"
    good: str = ""
    bad: str = ""


class QualityConfig(BaseModel):
    max_accuracy_drop_pts: float = 2.0


class TuneConfig(BaseModel):
    base_commit: str = ""
    knobs: dict[str, list[Any]] = Field(
        default_factory=lambda: {
            "max_num_seqs": [64, 128, 256, 512],
            "max_num_batched_tokens": [2048, 4096, 8192, 16384],
            "async_scheduling": [False, True],
            "enable_prefix_caching": [False, True],
            "kv_cache_dtype": ["auto", "fp8"],
        }
    )
    quality: QualityConfig = Field(default_factory=QualityConfig)
    max_trials: int = 8


class AgentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PERFPILOT_LLM_")

    base_url: str = "https://inference.baseten.co/v1"
    model: str = "meta-models/muse-spark-open-1026"
    api_key: str = ""


class PerfPilotConfig(BaseModel):
    model: str = ""
    gpu: str = "A100"
    # Measured Modal rate: $2.10/hr (A100 40GB), $2.50/hr (80GB). Reports
    # and evals must read this, never hardcode a $/hr constant.
    gpu_cost_per_hour: float = 2.10
    workload: Workload = Field(default_factory=Workload)
    metrics: Metrics = Field(default_factory=Metrics)
    stats: StatsConfig = Field(default_factory=StatsConfig)
    bisect: BisectConfig = Field(default_factory=BisectConfig)
    tune: TuneConfig = Field(default_factory=TuneConfig)
    agent: AgentSettings = Field(default_factory=AgentSettings)


def load_config(path: str | Path) -> PerfPilotConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p} (pass --config perfpilot.yaml)")
    data = yaml.safe_load(p.read_text()) or {}
    agent_env = AgentSettings()
    agent = {
        **(data.get("agent") or {}),
        "base_url": agent_env.base_url,
        "model": data.get("agent", {}).get("model", agent_env.model),
        "api_key": agent_env.api_key,
    }
    return PerfPilotConfig(**{**data, "agent": agent})
