"""Markdown tables for bench output and README embedding."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from perfpilot.bench.harness import SuiteSummary


def variant_table(summary: SuiteSummary) -> str:
    out = [
        "| Variant | Correct | Exact | Wrong | Probes/run | $/run (sim) | Score |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in summary.rows:
        out.append(
            f"| {r.variant} | {r.correct_rate:.0%} | {r.exact_rate:.0%} | "
            f"{r.wrong} | {r.mean_probes:.2f} | ${r.mean_cost_usd:.2f} | "
            f"{r.score:.1f} |"
        )
    return "\n".join(out)


def scenario_table(summary: SuiteSummary) -> str:
    variants = [r.variant for r in summary.rows]
    cells: dict[str, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
    for r in summary.results:
        cells[r.scenario][r.variant].append(r)
    out = [
        "| Scenario | " + " | ".join(f"{v} (ok · probes)" for v in variants) + " |",
        "|---|" + "---|" * len(variants),
    ]
    for scenario, by_variant in cells.items():
        parts = []
        for v in variants:
            rs = by_variant[v]
            ok = sum(x.correct for x in rs)
            probes = sum(x.probes for x in rs) / len(rs)
            parts.append(f"{ok}/{len(rs)} · {probes:.1f}")
        out.append(f"| {scenario} | " + " | ".join(parts) + " |")
    return "\n".join(out)


def trend_table(rows: list[dict[str, Any]]) -> str:
    out = [
        "| When | Commit | Score | Δ | Correct | Probes |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        out.append(
            f"| {r['ts'][:16]} | `{r['git_sha'][:8]}` | {r['score']:.1f} | "
            f"{r['delta_score']:+.1f} | {r['correct_rate']:.0%} | "
            f"{r['mean_probes']:.2f} |"
        )
    return "\n".join(out)
