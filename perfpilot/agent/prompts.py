"""System prompts, one per decision type, plus tiny-context tables."""

# ruff: noqa: E501 - prompt strings are model-facing prose, not code.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

PICK_COMMIT = """You are choosing the next vLLM commit to benchmark in a performance bisect.
The regression is a {effect}% drop in {metric} between good {good} and bad {bad}.
Untested commits, oldest first, with messages: {table}
Results so far: {history}
Return JSON only: {{"sha": "<one sha from the list>", "reason": "<one sentence>"}}.
Prefer the middle of the range unless a commit message clearly points at code
this workload exercises (scheduler, attention, sampling, KV cache, CUDA graphs)."""

EXTRA_RUNS = """You are deciding how many extra benchmark runs to add for an inconclusive commit.
Current runs: {current_runs} (max {max_runs}). Interval: [{lo:.4f}, {hi:.4f}] around
midpoint m = {midpoint:.4f} for ln(commit / good).
Return JSON only: {{"runs": <1-3>, "reason": "<one sentence>"}}.
Ask for fewer runs when the interval is close to resolving, more when it is wide."""

EXPLAIN_CULPRIT = """You are explaining a verified vLLM performance regression.
Culprit commit: {sha} ({message}).
Before/after {metric}: good mean {good_mean:.2f}, bad mean {bad_mean:.2f}, effect {effect}.
Diff excerpt (only cite paths appearing here): {diff}
Return JSON only: {{"summary": "<one sentence>", "mechanism": "<why it slows this workload>",
"files": ["<diff paths only>"], "confidence": "<low|medium|high>"}}."""

RANK_KNOBS = """You are ranking vLLM serve flags to try first for higher {metric}.
Baseline: {baseline}. Supported knobs and values: {knobs}.
Return JSON only: {{"order": ["<knob names, most promising first>"], "reason": "<one sentence>"}}.
High queuing/TTFT under load suggests batch knobs first; only name knobs from the list."""

PROPOSE_CONFIG = """You are proposing one vLLM serve flag change to benchmark.
Best config so far: {best}. Already tried: {tried}. Allowed values: {knobs}.
Results so far: {history}
Return JSON only: {{"knob": "<name>", "value": <allowed value>, "reason": "<one sentence>"}}.
Propose exactly one change that has not been tried."""

DRAFT_FIX = """You are drafting a minimal revert/fix patch for a vLLM performance regression.
Culprit commit: {sha}. Diff: {diff}
Return JSON only: {{"patch": "<unified diff text>"}}.
Keep it minimal and limited to the culprit's changed files."""

SYSTEM_PROMPTS = {
    "pick_commit": PICK_COMMIT,
    "extra_runs": EXTRA_RUNS,
    "explain_culprit": EXPLAIN_CULPRIT,
    "rank_knobs": RANK_KNOBS,
    "propose_config": PROPOSE_CONFIG,
    "draft_fix": DRAFT_FIX,
}


def commit_table(
    commits: Sequence[str], messages: Mapping[str, str] | None = None
) -> str:
    lines = []
    for i, sha in enumerate(commits):
        msg = (messages or {}).get(sha, "")
        lines.append(f"{i}: {sha} {msg}".rstrip())
    return "\n".join(lines)


def history_table(rows: Sequence[Mapping[str, Any]]) -> str:
    """Compact commit/runs/effect/interval table; never raw logs."""
    out = ["commit | runs | effect | interval"]
    for r in rows:
        out.append(
            f"{r.get('commit', '?')} | {r.get('runs', '?')} | "
            f"{r.get('effect', '?')} | [{r.get('lo', '?')}, {r.get('hi', '?')}]"
        )
    return "\n".join(out)
