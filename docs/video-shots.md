# Video shot list (3 min, 2:50 target, ~400 words)

Total GPU spend for all footage: $0 — record runs, never live-GPU.
Script outline: [story-research.md](story-research.md) §5.1. Every sentence
below is cross-checked against README/`runs/`; no promised uplift, no
unsupported numbers.

## Beats

| Time | Shot | Source |
|---|---|---|
| 0:00–0:15 | Hook: verdict card with the tok/s cliff | `runs/m2-eval-9/report.html` — Culprit `e5196e29`, regression −60.1% (derived from endpoint trials), unverified |
| 0:15–0:40 | Problem: vLLM regressions ship silently; hand-bisect is slow and foolable — "our first GPU eval was fooled exactly that way" | vLLM issue #39940 numbers + `docs/architecture.html` range diagram |
| 0:40–0:55 | Solution: two planes; "brain proposes, benchmarks dispose"; "every model in this loop is open weights" | `docs/architecture.html` planes diagram; `perfpilot bisect --good G --bad B` |
| 0:55–1:35 | Proof (bisect): `perfpilot demo runs/m2-eval-9` replay | Timeline narrows on screen; land on culprit + interval + GPU $. Include the `[3569, 3594, 1703]` baseline-outlier moment and the re-measure fix in one breath |
| 1:35–2:00 | Proof (tune, honest): tune report | `runs/tune-20261101-150900/report.html` — 6 candidates, intervals, fp8 rejected (A100 lacks Hopper FP8), "Baseline kept". Line: "an agent that can say 'no win here' is one you can trust when it says 'win'" |
| 2:00–2:35 | Numbers: local table and GPU table side by side, labelled as different instruments | README eval sections — local 2/2 ($0) vs GPU 0 wrong verdicts ($4.63) + mild-overlay premise measured −0.8%. Say the first-commit limitation out loud |
| 2:35–2:55 | Close: end card | Repo URL, `perfpilot report runs/<id>` re-renders offline, X handle. "Built this week, all open weights. Thank you." |

## Rules

- No live GPU on camera. Screen-record the reports and docs pages at 125% zoom; burn in captions (numbers must be readable paused).
- Say out loud: open weights + agentic loop with accepted/rejected proposals.
- Never merge the local (2/2) and GPU tables into one score.
- Upload unlisted, test the link logged out, then publish. Reserve the final afternoon solely for this.
- End card: repo link + X post (impact line + 20s dashboard clip).
