# Video script (3 min, 2:50 target)

Record from recorded runs only, never live GPU. Screen-record the Pages site
(https://bharath03-a.github.io/perfpilot/) at 125% zoom with captions burned in.
Every number below is in the README or `runs/`.

| Time | On screen | Say |
|---|---|---|
| 0:00–0:15 | Site hero; the bisect replay animates and narrows onto the culprit band | "vLLM ships performance regressions silently. One commit can cost you half your throughput. PerfPilot finds it, and proves it." |
| 0:15–0:35 | A vLLM perf-regression issue, then the hero again | "Finding it by hand means hours of GPU benchmarking, and one noisy run fools you. Our own first GPU run got fooled exactly that way: one collapsed sample at 1,703 tok/s." |
| 0:35–0:55 | Architecture page, planes diagram | "Two planes. A free local plane decides and proves. A metered A100 on Modal only measures. The brain, Meta's muse-spark-open on Baseten, only advises. Brain proposes, benchmarks dispose. Every model in this loop is open weights: we serve Llama-3.2-1B, and the agent is muse-spark-open." |
| 0:55–1:25 | Run reports tab: `eval-t0` report, swarm votes panel, then the timeline | "The agent is a swarm: a bisector, a code reader that reads commit messages, and a statistician, plus a deterministic referee. Here's a real A100 run, end to end: we planted a slowdown at commit 7, the swarm voted, and the range closed around commit 7 with every commit labelled correctly. Cost: $1.17." |
| 1:25–2:00 | Bench section: heatmap, then dumbbells | "How do we know it's getting better? A gym. 17 scenarios with hidden truth: noise, outlier runs, broken builds, flaky workers, rig drift, misleading commit messages. Every change is scored and logged against its git SHA. Building it caught real bugs, like a contaminated baseline we now fix with a pooled median trim." |
| 2:00–2:20 | Bench live table: single vs swarm | "And we're honest about it. With the live model, the swarm wins when commit messages carry real signal, six probes versus ten, and loses on pure noise. So the referee is deterministic, and the agents never get the last word." |
| 2:20–2:40 | Tune report: 6 candidates, fp8 rejected, "Baseline kept" | "TuneSmith tries serve flags to win speed back. On this rig none beat baseline by the 2% bar, and FP8 is impossible on an A100. It said so. An agent that can say 'no win here' is one you can trust when it says 'win'." |
| 2:40–2:55 | End card: repo URL, site URL, `uv run perfpilot bench` | "Zero wrong verdicts, 224 tests, $8.63 of GPU for the whole project. Clone it and run the bench with no GPU and no key. Thanks." |

## Rules

- Say "open weights" and "agentic loop" out loud. Show an accepted and a rejected proposal.
- Never merge scripted bench, live bench and GPU numbers into one score. Label each one.
- Upload unlisted, test the link while logged out, then set `VIDEO_ID` in `docs/index.html`.
