"""JSON export for the Pages bench section; numbers come from a real run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from perfpilot.bench.harness import SuiteSummary
from perfpilot.bench.scenarios import ALL_SCENARIOS


def scenario_cells(summary: SuiteSummary) -> list[dict[str, Any]]:
    by: dict[tuple[str, str], list] = {}
    for r in summary.results:
        by.setdefault((r.scenario, r.variant), []).append(r)
    family = {s.name: (s.family, s.description) for s in ALL_SCENARIOS}
    return [
        {
            "scenario": sc,
            "variant": v,
            "family": family.get(sc, ("core", ""))[0],
            "description": family.get(sc, ("", ""))[1],
            "n": len(rs),
            "correct": sum(x.correct for x in rs),
            "exact": sum(x.exact for x in rs),
            "wrong": sum(x.wrong for x in rs),
            "probes": round(sum(x.probes for x in rs) / len(rs), 2),
            "cost_usd": round(sum(x.cost_usd for x in rs) / len(rs), 3),
        }
        for (sc, v), rs in by.items()
    ]


def export_bench(
    path: Path, summary: SuiteSummary, entries: list[dict[str, Any]], meta: dict
) -> None:
    payload = {
        "meta": meta,
        "variants": [r.model_dump() for r in summary.rows],
        "cells": scenario_cells(summary),
        "scoreboard": entries,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1))


def hero_trace(run_dir: Path) -> dict[str, Any]:
    """Probe-ordered per-commit samples from a recorded bisect run."""
    import yaml

    cfg = yaml.safe_load((run_dir / "config.yaml").read_text()) or {}
    commits = [str(c) for c in cfg.get("commits", [])]
    index = {sha: i for i, sha in enumerate(commits)}
    probes: list[dict[str, Any]] = []
    culprit: str | None = None
    status = ""
    for line in (run_dir / "events.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        if e.get("type") == "trial_result" and e.get("sha") in index and e.get("runs"):
            probes.append(
                {"idx": index[e["sha"]], "sha": e["sha"][:9], "runs": e["runs"]}
            )
        elif e.get("type") == "culprit":
            culprit = e.get("sha") or e.get("culprit")
        elif e.get("type") == "run_end":
            status = e.get("status", "")
    if not probes:
        raise ValueError(f"{run_dir}: no trial_result events with samples")
    return {
        "run_id": run_dir.name,
        "n_commits": len(commits),
        "culprit_idx": index.get(culprit) if culprit else None,
        "status": status,
        "probes": probes,
    }
