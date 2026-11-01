"""Config loading: defaults, file parse, missing-file error."""

from __future__ import annotations

import pytest

from perfpilot.config import PerfPilotConfig, load_config


def test_defaults_match_plan():
    cfg = PerfPilotConfig()
    assert cfg.gpu == "A100"
    assert cfg.gpu_cost_per_hour == 2.10
    assert cfg.workload.num_prompts == 100
    assert cfg.stats.min_runs == 3
    assert cfg.stats.max_runs == 8
    assert cfg.stats.confidence == 0.90
    assert cfg.agent.model == "meta-models/muse-spark-open-1026"
    assert cfg.agent.base_url == "https://inference.baseten.co/v1"


def test_example_file_loads(tmp_path):
    src = (
        "model: test-model\n"
        "gpu: A100\n"
        "workload:\n  num_prompts: 100\n"
        "bisect:\n  good: abc\n  bad: def\n"
    )
    p = tmp_path / "perfpilot.yaml"
    p.write_text(src)
    cfg = load_config(p)
    assert cfg.model == "test-model"
    assert cfg.bisect.good == "abc"


def test_missing_file_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")
