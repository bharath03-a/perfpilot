# Video shot list (3 min)

Total GPU spend for all footage: $0 — record runs, never live-GPU.

## Beats (PLAN.md §5)

| Time | Shot | Source |
|---|---|---|
| 0:00 | tok/s cliff ("last deploy cost 32%") | `runs/m2-eval-9/report.html` — verdict card |
| 0:20 | Bisect narrowing, culprit + diff | `runs/m2-eval-9/report.html` — timeline section |
| 0:40 | Spark explains why | agent panel in report + `events.jsonl` proposal log |
| 0:55 | TuneSmith propose → bench → keep/revert, verified uplift | tune run report (M4 GPU) or `runs/` tune record when available |
| 1:20 | Eval table (correct, 0 wrong, cost stated) | README eval section |
| 1:40 | Architecture (control vs measurement plane) | `docs/architecture.html` section A |
| 2:00 | Cost model: only trials spend | billing + `docs/architecture.html` |
| 2:20 | "Every model in this loop is open weights" | Llama-3.2-1B + muse-spark-open |

## Rules

- No live GPU on camera. Screen-record the reports and docs pages.
- Say out loud: open weights + agentic loop with accepted/rejected proposals.
- End card: repo link + X post (impact line + 20s dashboard clip).
