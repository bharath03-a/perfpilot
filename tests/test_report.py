"""Report + run-folder unit tests. Pure: synthetic events.jsonl only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from perfpilot.report.render import (
    EVENT_TYPES,
    FALLBACK_GPU_COST_PER_HOUR,
    load_events,
    render_report,
    render_watch_page,
    write_report,
)
from perfpilot.report.run_dir import (
    append_event,
    finalize_report,
    flags_hash,
    run_start_event,
    write_fix_patch,
    write_log,
    write_run_skeleton,
    write_trial_result,
)


def _synthetic_events(run_id: str = "eval0") -> list[dict]:
    return [
        {
            "ts": "t0",
            "run_id": run_id,
            "type": "run_start",
            "model": "glimmer",
            "gpu": "A100",
            "gpu_uuid": "GPU-uuid-1",
            "vllm_version": "0.1",
            "bench_version": "0.1",
            "repo": "vllm-project/vllm",
            "good": "aaa",
            "bad": "ppp",
            "gpu_cost_per_hour": 2.10,
        },
        {"ts": "t1", "run_id": run_id, "type": "trial_start", "sha": "hhh", "runs": 3},
        {
            "ts": "t2",
            "run_id": run_id,
            "type": "trial_result",
            "sha": "hhh",
            "status": "ok",
            "mean": 100.0,
            "lo": 98.0,
            "hi": 102.0,
            "gpu_seconds": 600.0,
        },
        {
            "ts": "t3",
            "run_id": run_id,
            "type": "verdict",
            "sha": "hhh",
            "label": "good",
            "effect": 0.0,
            "lo": -0.02,
            "hi": 0.02,
            "reason": "near good",
        },
        {
            "ts": "t4",
            "run_id": run_id,
            "type": "agent_call",
            "decision": "pick_commit",
            "proposal": {"sha": "lll"},
            "accepted": True,
            "why": "middle of range",
        },
        {
            "ts": "t5",
            "run_id": run_id,
            "type": "agent_call",
            "decision": "extra_runs",
            "proposal": {"runs": 3},
            "accepted": False,
            "why": "over max_runs cap",
        },
        {
            "ts": "t6",
            "run_id": run_id,
            "type": "boundary_update",
            "good": "hhh",
            "bad": "ppp",
        },
        {
            "ts": "t7",
            "run_id": run_id,
            "type": "trial_result",
            "sha": "lll",
            "status": "skip",
            "label": "skipped",
            "reason": "no wheel",
        },
        {
            "ts": "t8",
            "run_id": run_id,
            "type": "trial_result",
            "sha": "mmm",
            "status": "ok",
            "mean": 80.0,
            "lo": 78.0,
            "hi": 82.0,
            "gpu_seconds": 600.0,
        },
        {
            "ts": "t9",
            "run_id": run_id,
            "type": "verdict",
            "sha": "mmm",
            "label": "bad",
            "effect": -0.22,
            "reason": "past midpoint",
        },
        {
            "ts": "t10",
            "run_id": run_id,
            "type": "culprit",
            "sha": "mmm",
            "effect": -0.22,
            "lo": -0.25,
            "hi": -0.19,
        },
        {
            "ts": "t11",
            "run_id": run_id,
            "type": "verify",
            "verified": True,
            "effect": -0.21,
            "reason": "same direction, significant",
        },
        {
            "ts": "t12",
            "run_id": run_id,
            "type": "explain",
            "sha": "mmm",
            "summary": "slower path",
            "mechanism": "eager fallback",
            "files": ["vllm/worker.py"],
            "confidence": "high",
        },
        {
            "ts": "t13",
            "run_id": run_id,
            "type": "tune_candidate",
            "config": {"max_num_seqs": 256},
            "reason": "raise throughput",
        },
        {
            "ts": "t14",
            "run_id": run_id,
            "type": "tune_accept",
            "config": {"max_num_seqs": 256},
            "accepted": True,
            "why": "uplift verified",
            "effect": 0.06,
        },
        {
            "ts": "t15",
            "run_id": run_id,
            "type": "quality_check",
            "config": {"max_num_seqs": 256},
            "passed": True,
            "accuracy_drop_pts": 0,
        },
        {"ts": "t16", "run_id": run_id, "type": "run_end", "status": "done"},
    ]


def _write_events(tmp_path: Path, events: list[dict]) -> Path:
    p = tmp_path / "events.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    return p


def test_all_spec_event_types_covered():
    covered = {e["type"] for e in _synthetic_events()}
    assert covered == set(EVENT_TYPES)


def test_render_covers_all_sections(tmp_path: Path):
    html_doc = render_report(_write_events(tmp_path, _synthetic_events()))
    assert "chart.umd.min.js" in html_doc  # Chart.js CDN
    assert "mmm" in html_doc  # verdict card culprit
    assert "verified" in html_doc
    assert "hhh" in html_doc and "lll" in html_doc  # timeline
    assert "skipped" in html_doc  # skip coloring present
    assert "tput" in html_doc  # throughput chart canvas
    assert "rejected" in html_doc  # agent panel accepted + rejected
    assert "middle of range" in html_doc and "over max_runs cap" in html_doc
    assert "vllm/worker.py" in html_doc and "github.com" in html_doc  # diff link
    assert "vllm serve" in html_doc and "max_num_seqs" in html_doc  # paste-ready cmd
    assert "GPU-uuid-1" in html_doc  # reproducibility block


def test_render_tolerates_empty_log(tmp_path: Path):
    html_doc = render_report(_write_events(tmp_path, []))
    assert "PerfPilot report" in html_doc


def test_watch_adds_autorefresh(tmp_path: Path):
    html_doc = render_watch_page(_write_events(tmp_path, _synthetic_events()))
    assert "http-equiv='refresh'" in html_doc or 'http-equiv="refresh"' in html_doc
    static = render_report(tmp_path / "events.jsonl")
    assert "refresh" not in static


def test_run_skeleton_layout_and_helpers(tmp_path: Path):
    run_dir = write_run_skeleton("r1", {"model": "glimmer"}, runs_root=tmp_path)
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "events.jsonl").exists()
    assert (run_dir / "results").is_dir() and (run_dir / "logs").is_dir()
    append_event(run_dir, {"run_id": "r1", "type": "run_start"})
    assert len((run_dir / "events.jsonl").read_text().splitlines()) == 1
    rp = write_trial_result(run_dir, "abc", {"a": 1}, 0, {"status": "ok"})
    assert rp.parent.name.startswith("abc__") and rp.name == "run_0.json"
    assert flags_hash({"a": 1}) == flags_hash({"a": 1})
    write_log(run_dir, "server", "ok")
    assert (run_dir / "logs" / "server.log").exists()
    finalize_report(run_dir, "<html></html>")
    write_fix_patch(run_dir, "diff")
    assert (run_dir / "report.html").exists() and (run_dir / "fix.patch").exists()
    out = write_report(run_dir / "events.jsonl")
    assert out == run_dir / "report.html" and Path(out).exists()


def test_flags_hash_single_sourced_with_worker():
    from perfpilot.worker.schemas import flags_hash as worker_hash

    assert flags_hash is worker_hash
    assert len(flags_hash({"a": 1})) == 16
    assert flags_hash({"b": 2, "a": 1}) == flags_hash({"a": 1, "b": 2})


def test_run_start_event_threads_config_rate(tmp_path: Path):
    from perfpilot.config import PerfPilotConfig

    assert FALLBACK_GPU_COST_PER_HOUR == 2.10
    assert PerfPilotConfig().gpu_cost_per_hour == 2.10
    run_dir = write_run_skeleton("r9", {"model": "m"}, runs_root=tmp_path)
    append_event(run_dir, run_start_event("r9", PerfPilotConfig().gpu_cost_per_hour))
    first = json.loads((run_dir / "events.jsonl").read_text().splitlines()[0])
    assert first["gpu_cost_per_hour"] == 2.10


def test_verdict_card_falls_back_to_210_without_rate(tmp_path: Path):
    events = [e for e in _synthetic_events() if e["type"] != "run_start"]
    events.insert(0, {"ts": "t0", "run_id": "eval0", "type": "run_start"})
    html_doc = render_report(_write_events(tmp_path, events))
    assert "$2.10/h" in html_doc


def test_timeline_shows_effect_interval_and_reason(tmp_path: Path):
    html_doc = render_report(_write_events(tmp_path, _synthetic_events()))
    assert "past midpoint" in html_doc  # verdict reason is auditable
    assert "near good" in html_doc
    assert "no wheel" in html_doc  # skip reason kept


def test_timeline_never_uses_raw_status(tmp_path: Path):
    events = [
        {
            "ts": "t0",
            "run_id": "r",
            "type": "trial_result",
            "sha": "zzz",
            "status": "ok",
            "mean": 50.0,
        },
    ]
    html_doc = render_report(_write_events(tmp_path, events))
    assert ">ok<" not in html_doc and "— ok<" not in html_doc


def test_agent_panel_renders_fallback_column(tmp_path: Path):
    events = _synthetic_events()
    events.append(
        {
            "ts": "t17",
            "run_id": "eval0",
            "type": "agent_call",
            "decision": "pick_commit",
            "proposal": {"sha": "zzz"},
            "accepted": False,
            "why": "outside range; using middle",
            "fallback": "mmm",
        }
    )
    html_doc = render_report(_write_events(tmp_path, events))
    assert "Fallback" in html_doc and "mmm" in html_doc


def test_emit_rejects_unknown_type_but_load_keeps(tmp_path: Path):
    run_dir = write_run_skeleton("r2", {}, runs_root=tmp_path)
    with pytest.raises(ValueError, match="unknown event type"):
        append_event(run_dir, {"run_id": "r2", "type": "nope"})
    p = _write_events(tmp_path, [{"type": "nope", "run_id": "r"}])
    with pytest.warns(UserWarning, match="unknown event type"):
        assert load_events(p) == [{"type": "nope", "run_id": "r"}]


def test_tune_shows_candidate_outcome_interval_and_winner_confirm(tmp_path: Path):
    events = _synthetic_events()
    events.insert(
        -1,
        {
            "ts": "t13b",
            "run_id": "eval0",
            "type": "tune_candidate",
            "config": {"max_num_seqs": 128},
            "decision": "rejected",
            "effect": -0.01,
            "lo": -0.03,
            "hi": 0.01,
            "reason": "no uplift",
        },
    )
    events.insert(
        -1,
        {
            "ts": "t14b",
            "run_id": "eval0",
            "type": "tune_accept",
            "config": {"max_num_seqs": 256},
            "accepted": True,
            "effect": 0.06,
            "lo": 0.04,
            "hi": 0.08,
            "why": "uplift verified",
        },
    )
    html_doc = render_report(_write_events(tmp_path, events))
    assert "rejected" in html_doc and "no uplift" in html_doc
    assert "uplift" in html_doc


def test_tune_command_uses_serve_command_bool_port_seed(tmp_path: Path):
    from perfpilot.modes.tune import serve_command

    events = _synthetic_events()
    for e in events:
        if e["type"] == "tune_accept":
            e["config"] = {"async_scheduling": True, "max_num_seqs": 256}
    html_doc = render_report(_write_events(tmp_path, events))
    assert "vllm serve" in html_doc
    assert "--async-scheduling" in html_doc  # bool renders bare, no value
    assert "--async-scheduling True" not in html_doc
    assert serve_command("glimmer", {"async_scheduling": True}) in html_doc


def test_quality_rows_show_accuracy_and_reason(tmp_path: Path):
    events = _synthetic_events()
    for e in events:
        if e["type"] == "quality_check":
            e.update(
                {
                    "accuracy": 0.91,
                    "baseline_accuracy": 0.93,
                    "reason": "within 2pt gate",
                }
            )
    html_doc = render_report(_write_events(tmp_path, events))
    assert "0.91" in html_doc and "0.93" in html_doc
    assert "within 2pt gate" in html_doc
