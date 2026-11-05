# Submission text

## Name
PerfPilot

## Tagline (one line)
An agent swarm that finds the vLLM commit that made your model slow, and proves it with statistics, not vibes.

## Description
vLLM ships performance regressions silently. One commit can cost 10–60% throughput, and finding it by hand takes hours of GPU benchmarking that a single noisy run can fool.

PerfPilot bisects a vLLM commit range on a real A100 (Modal) serving Meta's Llama-3.2-1B. It returns the culprit commit, a log-ratio Welch interval and the GPU dollars spent. TuneSmith then tries serve flags to win the speed back, and accepts a flag only when the gain is proven and a quality gate holds.

The agentic loop is a proposer swarm running on Meta's open-weights muse-spark-open (Baseten). A bisector, a code reader that reads commit messages, and a statistician each propose which commit to probe next. A deterministic referee picks one, and the benchmark decides. Every proposal is schema-validated, with a deterministic fallback.

To measure whether the agent gets better, we built a gym: 17 planted-failure scenarios (noise, collapsed-run outliers, broken builds, flaky workers, rig drift, misleading hints, two culprits) with hidden truth, a judge and a scoreboard keyed to git SHA. It found real bugs that we fixed (baseline contamination, disk-full workers, ungrounded agent confidence).

Results: 0 wrong verdicts across 340 scripted runs, the live-model bench and every GPU run. End to end with the live swarm on an A100 for $1.17. $8.63 of GPU for the whole project, 224 tests.

## Meta AI used
- Llama-3.2-1B: the model being served and measured.
- muse-spark-open (Baseten): the swarm's brain, used for commit picking, extra-sample decisions and flag proposals.
Every model in the loop is open weights.

## Links
- Repo: https://github.com/bharath03-a/perfpilot
- Site: https://bharath03-a.github.io/perfpilot/
- Video: https://youtu.be/NJbV1aXLIMc

## X post
vLLM ships perf regressions silently. I built PerfPilot: an open-weights agent swarm (Llama + muse-spark-open) that bisects vLLM on a real A100 and names the slow commit, with a statistical interval and the GPU bill.

0 wrong verdicts across 17 failure scenarios. $8.63 total GPU.

🎥 <video>
🔗 https://github.com/bharath03-a/perfpilot
#MetaAIHackathon
