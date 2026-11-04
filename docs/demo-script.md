# Demo script — PerfPilot (2:50, ~400 words)

Record screen + voice. No face. 125% browser zoom, bookmarks bar hidden.
Tabs in order: ① verdict report ② demo replay ③ tune report ④ README ⑤ architecture end card.

Backdrop footage if you want it: `/tmp/demo-vid/*.webm` (55s silent B-roll:
bisect narrowing → tune → architecture). Play fullscreen behind the voiceover,
or just tab through the live pages.

| Time | Screen (what to show / click) | Say |
|---|---|---|
| 0:00–0:15 | ① m2-eval-9 report, verdict card | "Last deploy cost a cliff in tok/s — and nobody could say which commit did it. PerfPilot can. This is the verdict: culprit e5196e29, minus sixty percent, with a ninety-percent interval, on a real A100." |
| 0:15–0:40 | ⑤ architecture, Two planes diagram | "vLLM merges over night; hand-bisecting is slow, expensive, and foolable — our first GPU eval was fooled exactly that way. So PerfPilot splits the job: a free local plane decides and proves, a metered A100 plane only measures. The brain proposes, benchmarks dispose." |
| 0:40–0:55 | ⑤ architecture, Agentic loop section | "The agent is muse-spark-open on Baseten — every model in this loop is open weights. It picks commits and flags; validators gate every call; stats have the final word. No key, no run failure — deterministic fallbacks." |
| 0:55–1:35 | ② demo replay of m2-eval-9 (run it live: `uv run perfpilot demo runs/m2-eval-9 --speed 1.2`) | "Watch the range narrow. Each probe is one GPU trial with an interval — the outliers get re-measured, skips are skipped honestly. It lands on the culprit with the interval and the dollar cost on screen. The red moment in the middle is a collapsed run — poisoned baseline — that this run caught and the fix now re-measures." |
| 1:35–2:00 | ③ tune report | "TuneSmith tried six flags on the culprit's commit. Five cleared nothing. fp8 can't even run on an A100 — it was logged as a rejection and the run kept going. Baseline kept. An agent that can say 'no win here' is one you can trust when it says 'win.'" |
| 2:00–2:35 | ④ README eval tables | "Proof, kept in two separate instruments. Local protocol: two-of-two for zero dollars. Real A100: three trials, zero wrong verdicts, five-eighty-six total. Trial zero came back fifteen wide; the fixed driver re-ran it in eleven probes for a dollar twenty-three. One limitation, out loud: a regression at the very first commit has no clean baseline to compare — it's undetectable, and the docs say so." |
| 2:35–2:55 | ⑤ end card (docs landing) | "PerfPilot: built solo this week, every claim tied to an interval, every run re-renders offline. Repo below. Open weights, agentic loop, statistical proof. Thank you." |

## Checklist

- [ ] Tabs open in order, 125% zoom, notifications off (Do Not Disturb)
- [ ] QuickTime → New Screen Recording; one full take, mistakes included — pick the calmer take
- [ ] Export < 3:00; captions for every number (verdict, intervals, $)
- [ ] YouTube upload **unlisted**; test link in a private window
- [ ] Send the link → gets embedded on the site + goes in the submission form + X post
