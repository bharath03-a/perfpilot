"""CLI surface tests: help, config errors, report on missing run (no GPU)."""

from __future__ import annotations

import re

from typer.testing import CliRunner

from perfpilot.cli import app

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _clean(output: str) -> str:
    """Strip ANSI escapes: rich may split '--trials' across styled spans."""
    return _ANSI_RE.sub("", output)


def test_help_lists_commands() -> None:
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    for cmd in ("bisect", "tune", "report", "watch", "demo", "eval"):
        assert cmd in _clean(res.output)


def test_bisect_needs_good_bad(tmp_path) -> None:
    cfg = tmp_path / "perfpilot.yaml"
    cfg.write_text("model: m\n")
    res = runner.invoke(app, ["bisect", "--config", str(cfg)])
    assert res.exit_code == 2
    assert "good" in res.output.lower()


def test_report_missing_run(tmp_path) -> None:
    res = runner.invoke(app, ["report", str(tmp_path / "nope")])
    assert res.exit_code == 2


def test_eval_defaults_trials() -> None:
    res = runner.invoke(app, ["eval", "--help"])
    assert res.exit_code == 0
    assert "--trials" in _clean(res.output)


def test_demo_replays_run_into_scratch_dir(tmp_path) -> None:
    import json

    src = tmp_path / "src-run"
    src.mkdir()
    events = [
        {"ts": "t0", "run_id": "r", "type": "run_start", "gpu_cost_per_hour": 2.10},
        {
            "ts": "t1",
            "run_id": "r",
            "type": "trial_result",
            "sha": "goodsha",
            "status": "ok",
            "runs": [3600.0, 3580.0],
        },
        {
            "ts": "t2",
            "run_id": "r",
            "type": "trial_result",
            "sha": "badsha",
            "status": "ok",
            "runs": [1800.0, 1790.0],
        },
        {
            "ts": "t3",
            "run_id": "r",
            "type": "culprit",
            "culprit": "badsha",
            "culprit_range": ["badsha"],
            "verified": False,
            "steps": 1,
        },
        {"ts": "t4", "run_id": "r", "type": "run_end", "status": "culprit"},
    ]
    (src / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    (src / "agent-events.jsonl").write_text(
        json.dumps(
            {
                "ts": "t1b",
                "run_id": "r",
                "type": "agent_call",
                "decision": "pick_commit",
                "proposal": {"sha": "badsha"},
                "accepted": True,
                "why": "middle",
            }
        )
        + "\n"
    )
    before = (src / "events.jsonl").read_bytes()
    scratch = tmp_path / "replay"
    res = runner.invoke(app, ["demo", str(src), "--speed", "0", "--out", str(scratch)])
    assert res.exit_code == 0
    assert (src / "events.jsonl").read_bytes() == before  # source untouched
    html = (scratch / "report.html").read_text()
    assert "badsha" in html  # final frame: verdict + timeline
    assert "pick_commit" in html  # merged agent rows
    staged = (scratch / "events.jsonl").read_text().strip().splitlines()
    assert len(staged) == len(events)


def test_demo_missing_run(tmp_path) -> None:
    res = runner.invoke(app, ["demo", str(tmp_path / "nope"), "--speed", "0"])
    assert res.exit_code == 2
