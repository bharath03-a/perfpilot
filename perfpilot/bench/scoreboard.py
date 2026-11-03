"""Append-only scoreboard: one line per bench run, keyed by git SHA."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from perfpilot.bench.harness import SuiteSummary


def append_entry(path: Path, summary: SuiteSummary, git_sha: str, mode: str) -> dict:
    entry = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha,
        "mode": mode,
        "variants": {r.variant: r.model_dump() for r in summary.rows},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def load_entries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{n}: corrupt scoreboard line: {e}") from e
    return out


def trend_rows(
    entries: list[dict[str, Any]], variant: str, mode: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prev: float | None = None
    for e in entries:
        row = e.get("variants", {}).get(variant)
        if e.get("mode") != mode or row is None:
            continue
        score = float(row["score"])
        delta = 0.0 if prev is None else round(score - prev, 2)
        rows.append(
            {
                "ts": e["ts"],
                "git_sha": e["git_sha"],
                "score": score,
                "delta_score": delta,
                "correct_rate": row["correct_rate"],
                "mean_probes": row["mean_probes"],
            }
        )
        prev = score
    return rows
