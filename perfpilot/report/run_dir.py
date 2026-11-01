"""Run-folder writer. Pure filesystem: no GPU, no network.

Layout per SPEC.md/PLAN.md::

    runs/<run_id>/
      config.yaml
      events.jsonl          # source of truth for the report
      results/<sha>__<flags_hash>/run_<n>.json
      logs/<name>.log
      report.html
      fix.patch             # only if a fix was drafted

Signatures (CLI wiring comes later):

    write_run_skeleton(run_id: str | Path, config: ...) -> Path
    append_event(run_dir: str | Path, event: dict) -> None
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from perfpilot.config import PerfPilotConfig
from perfpilot.report.render import EVENT_TYPES
from perfpilot.worker.schemas import flags_hash

__all__ = [
    "EVENT_TYPES",
    "FALLBACK_GPU_COST_PER_HOUR",
    "flags_hash",
    "append_event",
    "finalize_report",
    "run_dir_for",
    "run_start_event",
    "write_fix_patch",
    "write_log",
    "write_run_skeleton",
    "write_trial_result",
]

#: $/h used only when the run_start event lacks gpu_cost_per_hour.
#: Mirrors PerfPilotConfig.gpu_cost_per_hour (measured A100-40GB rate);
#: never hardcode a competing constant.
FALLBACK_GPU_COST_PER_HOUR = float(
    PerfPilotConfig.model_fields["gpu_cost_per_hour"].default
)

DEFAULT_RUNS_ROOT = Path("runs")


def run_dir_for(run_id: str | Path, runs_root: str | Path = DEFAULT_RUNS_ROOT) -> Path:
    p = Path(run_id)
    return p if p.is_absolute() or len(p.parts) > 1 else Path(runs_root) / p


def write_run_skeleton(
    run_id: str,
    config: dict[str, Any] | str,
    runs_root: str | Path = DEFAULT_RUNS_ROOT,
) -> Path:
    """Create runs/<run_id>/ with config.yaml, empty events.jsonl, results/, logs/.

    Args:
        run_id: Run id (folder name) or path; bare names resolve under runs_root.
        config: Config mapping (written as config.yaml) or raw YAML string.
        runs_root: Parent dir for bare run ids.

    Returns:
        Path to the run dir.
    """
    run_dir = run_dir_for(run_id, runs_root)
    (run_dir / "results").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "config.yaml"
    if isinstance(config, str):
        cfg_path.write_text(config if config.endswith("\n") else config + "\n")
    else:
        cfg_path.write_text(yaml.safe_dump(dict(config), sort_keys=False))
    (run_dir / "events.jsonl").touch(exist_ok=True)
    return run_dir


def append_event(run_dir: str | Path, event: dict[str, Any]) -> None:
    """Append one event dict as a JSON line; fills ts when absent.

    Raises:
        ValueError: On an unknown event ``type`` (emit-side validation;
            readers warn-and-keep instead so old logs still render).
    """
    ev = dict(event)
    if "type" in ev and ev["type"] not in EVENT_TYPES:
        raise ValueError(
            f"unknown event type {ev['type']!r}; expected one of {EVENT_TYPES}"
        )
    ev.setdefault("ts", datetime.now(UTC).isoformat())
    with open(Path(run_dir) / "events.jsonl", "a") as f:
        f.write(json.dumps(ev) + "\n")


def run_start_event(
    run_id: str,
    gpu_cost_per_hour: float = FALLBACK_GPU_COST_PER_HOUR,
    **fields: Any,
) -> dict[str, Any]:
    """Build the run_start event threading the $/h rate from config.

    CLI wiring must pass ``config.gpu_cost_per_hour`` explicitly; the
    default applies only when no config value is available (e.g. tests).
    """
    return {
        "run_id": run_id,
        "type": "run_start",
        "gpu_cost_per_hour": gpu_cost_per_hour,
        **fields,
    }


def write_trial_result(
    run_dir: str | Path,
    commit_sha: str,
    flags: dict[str, Any],
    run_n: int,
    payload: dict[str, Any],
) -> Path:
    """Write results/<sha>__<flags_hash>/run_<n>.json; return its path."""
    d = Path(run_dir) / "results" / f"{commit_sha}__{flags_hash(flags)}"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"run_{run_n}.json"
    p.write_text(json.dumps(payload, indent=2))
    return p


def write_log(run_dir: str | Path, name: str, text: str) -> Path:
    p = Path(run_dir) / "logs" / (name if name.endswith(".log") else name + ".log")
    p.write_text(text)
    return p


def finalize_report(run_dir: str | Path, html: str) -> Path:
    p = Path(run_dir) / "report.html"
    p.write_text(html)
    return p


def write_fix_patch(run_dir: str | Path, patch: str) -> Path:
    p = Path(run_dir) / "fix.patch"
    p.write_text(patch)
    return p
