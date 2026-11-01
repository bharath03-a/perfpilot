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
    for cmd in ("bisect", "tune", "report", "watch", "eval"):
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
