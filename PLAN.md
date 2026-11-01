# PerfPilot — Detailed Win Plan

## 0. Problem statement

**Problem:** LLM inference serving silently gets slower (or stays slower
than it could be) and nobody can say why. vLLM merges daily; nightly
benchmarks catch day-sized windows, not culprit commits; engineers then
hand-bisect across builds at $4/GPU-hour and argue from vibes. Our own
research: real regressions lingered months or went stale (#39940: 4.5 mo,
stale-bot; #48396: open, contested repro; #61989: manual 14-commit guess).

* **Who:** teams serving open models (vLLM ops, ML platform, indie
  hackers on rented GPUs). Judge-readable: anyone who has watched a
  deploy get slower for no reason.
* **What:** an agent that (1) finds the exact commit behind a slowdown
  and explains the mechanism, (2) finds serving flags that recover the
  speed — every claim benchmark-verified with intervals, never asserted.
* **When:** right after a nightly flags a regressed window, or any time
  serving feels slow. Runs in hours, unblocking the release.
* **Where:** one pinned GPU (Modal A100), vLLM + Meta open model,
  agent brain on Baseten Spark Open.
* **Why:** manual bisect is slow, expensive, and inconclusive; existing
  tools detect flakes/coverage, none close the loop to proof + fix.
* **How:** coordinate search (bisect commits / descend flags) with a
  deterministic statistical gate; LLM proposes, Welch-interval verdicts
  dispose; full event log renders the report and the demo.

**Framing checks used:** 5W1H above; Jobs-to-be-done ("when my nightly
goes red, help me name the commit today, not next sprint"); inversion
("what would make a judge call this a wrapper?" → answer: no verified
numbers, so eval table + intervals are the product); "what must be true"
(wheels install, one GPU pinnable, ≥10% injections measurable — all in
Verify-first/M1); before→after demo test (cliff → culprit → recovery,
all on screen).

## 0. Identity

* **Project:** PerfPilot — agent that keeps a vLLM deployment of a Meta
  open model fast.
* **Track:** Indie developer, theme Push the Frontier (software).
* **Prize targets:** Indie top-5 + Best Open-Weight ($10K) + Best Agentic ($5K).
* **Deadline:** Wed Nov 4, 7 PM Phoenix. Submit by noon. Code freeze 9 AM.
* **Repo:** public, all code written during the hackathon (Oct 26–Nov 4).
* **Impact line (README + video + X):** "Performance regressions ship
  silently into vLLM nightlies and take days of GPU benchmarking by hand.
  PerfPilot finds the culprit commit in ~5 benchmark runs and tunes it back,
  verified."

## 1. Architecture

One benchmark-and-verify engine, two search modes on top:

* **PerfBisect:** given good/bad vLLM commits, binary-search the first bad
  commit, verify it, explain the diff, hand off to TuneSmith.
* **TuneSmith:** on one fixed commit, coordinate-descent over
  `vllm serve` flags for max throughput under latency + quality guards.

Rules: the orchestrator owns every state-changing decision; the agent
(Muse Spark Open on Baseten, OpenAI-compatible client) returns JSON
proposals only. Every proposal is validated (pydantic), retried once, then
falls back. 60s timeout per agent call; a run never stalls on the agent.
Every GPU call is idempotent and cached by (commit, flags_hash, inject).
Fail loudly with a reason; never silently fall back without an event.

```
perfpilot/
  cli.py            # typer: bisect, tune, report, watch, eval
  config.py         # pydantic models for perfpilot.yaml
  orchestrator.py   # runs a mode, validates proposals, writes events
  stats.py          # log-ratio Welch verdicts (pure, unit-tested)
  modes/bisect.py   # bisect state machine
  modes/tune.py     # one-knob-at-a-time search
  agent/client.py   # Baseten Spark Open client
  agent/prompts.py  # one system prompt per decision type
  agent/tools.py    # JSON schemas for allowed actions
  worker/modal_app.py  # Modal A100: run_trial()
  worker/env.py     # venv per commit from vLLM wheels, cached on Volume
  worker/server.py  # start/health/stop vllm serve
  worker/bench.py   # fixed vllm bench serve client, parse JSON
  worker/quality.py # greedy-decoding answer check (50 Qs)
  git_info.py       # commit ranges via GitHub API
  report/render.py  # events.jsonl -> report.html
evals/inject.py     # --enforce-eager / --gpu-memory-utilization 0.5 overlays
tests/ runs/ SPEC.md README.md
```

Config `perfpilot.yaml`: model (Meta open, fits one A100 — Glimmer-class,
NOT 816B Spark), workload (random dataset, in 1024 / out 128,
200 prompts → cut to 100, concurrency 32, seed 0), primary metric
`output_throughput`, guard `p99_ttft_ms`, stats (min_runs 3, max_runs 8,
90% confidence), bisect good/bad SHAs, tune knobs
(max_num_seqs, max_num_batched_tokens, async_scheduling,
enable_prefix_caching, kv_cache_dtype) + quality (max 2-pt accuracy drop),
agent (base_url/model `meta-models/muse-spark-open-1026` from env).

## 2. Core engine

`run_trial(commit, flags, runs, inject=None) -> TrialResult`:
commit, flags, status (ok/skip/error), reason, gpu_uuid (`nvidia-smi -L`),
vllm_version, per-run {output_throughput, mean_ttft_ms, p99_ttft_ms,
median_tpot_ms, failed_requests} (any failed requests invalidate the run),
gpu_seconds.

Environments: `uv venv` per commit at `/cache/envs/<sha>`, install that
commit's wheel (`https://wheels.vllm.ai/<sha>`, confirm CUDA subdir on
first run), cache venvs + model weights on a Modal Volume. No wheel or
failed install → `skip` with reason, never build from source.
Server: start `vllm serve` in its own process group, poll `/health`
(15-min timeout → error + log tail), one warmup benchmark discarded,
always kill the group after. Benchmark client: ONE pinned vLLM version
for all commits (never let the ruler change). Same GPU always: one
Modal container max, long scaledown; stats refuses cross-GPU compares.

Verdict stats (`stats.py`, unit-tested): log-ratio Welch interval.
Effect r = ln(mean_c/mean_g), se from both samples,
Welch–Satterthwaite df, interval at configured confidence. Every verdict
returns interval + reason string. Bisect midpoint m = R/2 where
R = ln(bad/good); commit is bad/good iff its whole interval sits past m
toward that side, else inconclusive (agent proposes extra runs ≤ max_runs,
then ambiguous = skip). No confirmed regression (|R| < 3% or interval
covers 0) → honest "no confirmed regression" stop. Drift check: re-measure
good every 4 steps; >2% drift → re-measure both endpoints. Culprit =
first bad commit (or range if skips intervene); verify back-to-back with
max_runs, significant + same direction or it doesn't count. Explanation:
agent writes change/mechanism/confidence citing diff paths; orchestrator
rejects uncited paths. Stretch: draft revert .patch (first on cut list).

TuneSmith: baseline min_runs → drop unsupported knobs (`vllm serve
--help` per venv, logged) → agent ranks knobs, proposes one JSON change
at a time (orchestrator rejects out-of-config/already-tried) → accept
iff interval for ln(cand/best) fully above +2% AND p99 TTFT ≤ 10% worse →
stop at 8 candidates or no improvement → quality guard (numerics-changing
configs: 50 fixed Qs, greedy, ≤32 tokens, reject on >2-pt drop; final
config always checked) → confirm best vs baseline at max_runs → output
ready-to-paste `vllm serve` command + uplift interval + full tried-list.

Agent decisions: pick_commit (fallback: middle), extra_runs (fallback:
2), explain_culprit (fallback: "no explanation", culprit still counts),
rank_knobs (fallback: config order), propose_config (fallback: next
untried), draft_fix (fallback: skip). Keep context tiny: compact
commit/runs/effect/interval table, never raw logs. Log every call to
events.jsonl (type, proposal, accepted/rejected, why) — powers the agent
panel and the video.

## 3. Eval protocol (the credibility core)

Injected regressions: worker secretly adds a slowdown flag for commits at
or after a hidden index (agent never sees it). Two overlays, pre-measured
Friday, only kept if ≥10% drop on your workload: `--enforce-eager`
(strong, version-stable) and `--gpu-memory-utilization 0.5` (milder).
Protocol: fix one 16-commit range where all wheels exist; run 2 trials
with seeded culprit positions at the low edge (index 1 — index 0 leaves no
clean in-range baseline, so no transition exists to detect) and the last
commit; reuse cached venvs, never reuse results; skip explanation in eval
trials.
Report in README + video: correct culprits (target 2/2), wrong verdicts
at any step (target 0), steps vs log2(range), inconclusive steps + extra
runs used, GPU minutes + cost per trial.

CLI: `perfpilot bisect --config perfpilot.yaml [--good SHA --bad SHA]`,
`perfpilot tune --config perfpilot.yaml [--commit SHA]`,
`perfpilot report runs/<id>`, `perfpilot watch runs/<id>` (live refresh =
demo view), `perfpilot eval --trials 2`. Run folder per run:
config.yaml, events.jsonl, results/<sha>__<flags>/run_<n>.json, logs/,
report.html (+ fix.patch). Report page (self-contained HTML, Chart.js
CDN): verdict card (culprit, size + interval, verified?, GPU min + $),
bisect timeline (good/bad/skipped/ambiguous), throughput chart with
intervals, agent panel, explanation with diff links, TuneSmith winner +
paste-ready command, reproducibility block (GPU UUID, versions, config).

## 4. Schedule, budget, gates

Budget: $10 now + $30 Sunday reset ≈ 17 A100-hours (A100 ≈ $2–2.50/hr;
H100 is out — doubles burn for zero judging credit). Serve Glimmer-class
open model, 100 prompts, min_runs 3.

* **Fri night ($0 GPU):** SPEC.md from PDF into repo; M0 scaffold
  (package layout, pydantic config, typer stubs, stats.py + pytest incl.
  synthetic 0%/5%/20% effects classified correctly).
* **Sat (~3.5h GPU on $10):** M1 — one-commit venv, serve, bench, same-GPU
  pin confirmed; no-wheel commit returns skip. **GATE Sat noon: M1 not
  green by ~$8 spent → cut PerfBisect, ship TuneSmith-only on one commit.**
  Then M2 deterministic bisect + injection support; one injected trial
  finds the planted culprit.
* **Sun ($30 reset):** M3 agent brain wired (all calls logged); eval trial
  2; with key unset a bisect still completes on fallbacks (acceptance).
  M4 verify + explain; TuneSmith (6 candidates max) only if eval is 2/2.
* **Mon:** real-regression attempt only if it reproduces fast, else drop
  without regret. Feature freeze EOD.
* **Tue:** M5 report + watch; record + edit the 3-min video. Full day,
  no code.
* **Wed:** code freeze 9 AM Phoenix; fresh-clone README check
  (`perfpilot eval --trials 1` runnable); submit by noon; X post.

Cut order: workaround → TuneSmith → real regression → draft_fix →
watch. Never cut: verification, eval, video day.

## 5. Demo, submission, risks

Video beats: 0:00 tok/s cliff ("last deploy cost 32%"); 0:20 bisect
narrowing, culprit + diff, Spark explains why; 0:55 TuneSmith
propose→bench→keep/revert, verified uplift number; close on the eval
table (2/2, 0 wrong, cost stated). Say out loud: "every model in this
loop is open weights" (open-weight bonus) and show the accepted/rejected
proposal log (agentic bonus). Never live-GPU in the video; record runs.
X post: impact line + 20s dashboard clip + repo link.

Submission checklist: public repo, README (what/how/eval table/cost/
built-during section), video, X post, bonus eligibility named
(open models used, agentic loop described), Wed noon Phoenix.

Top risks: (1) Modal A100 capacity — M1 spike answers tonight;
(2) wheel index format wrong — confirm on ONE commit before scripting;
(3) benchmark JSON keys differ — pin parser to first real run;
(4) money runs dry mid-Sat — stop, Sunday refills;
(5) judges can't rerun GPU work — report.html + events.jsonl +
recorded video are the reproducible artifacts, say so in README.
Secrets only via env: PERFPILOT_LLM_BASE_URL, PERFPILOT_LLM_API_KEY,
GITHUB_TOKEN, Modal auth. No work code or data, ever.

## 6. Competition + reference links

* Hackathon: https://cerebralvalley.ai/e/meta-ai-developer-hackathon
  (Remote, Oct 26–Nov 4 9 PM ET; judging Nov 4–8; finals Nov 9)
* Prizes/rules mirror:
  https://madrobot.blog/2026/10/03/meta-global-ai-developer-hackathon-1-million-prizes-muse-model-api/
* Meta Model API docs: overview https://dev.meta.ai/docs/overview ·
  auth https://dev.meta.ai/docs/authentication · SDKs
  https://dev.meta.ai/docs/sdks · models https://dev.meta.ai/docs/models ·
  pricing https://dev.meta.ai/docs/pricing-rate-limits ·
  image-gen https://dev.meta.ai/docs/image-generation ·
  media-seg https://dev.meta.ai/docs/media-segmentation ·
  tool-calling https://dev.meta.ai/docs/tool-calling ·
  structured-output https://dev.meta.ai/docs/structured-output ·
  open-weights https://dev.meta.ai/docs/open-weight-models ·
  muse-code https://dev.meta.ai/docs/muse-code
* Partners: Baseten https://www.baseten.co · Modal https://modal.com
  (endpoints https://modal.com/docs/guide/endpoints, GPUs
  https://modal.com/docs/guide/gpu) · Fireworks https://fireworks.ai
* vLLM evidence (motivation): #39940
  https://github.com/vllm-project/vllm/issues/39940 · #48396
  https://github.com/vllm-project/vllm/issues/48396 · #61989
  https://github.com/vllm-project/vllm/pull/61989
* vLLM bench CLI https://docs.vllm.ai/en/stable/benchmarking/cli/ ·
  GuideLLM https://github.com/vllm-project/guidellm
* Winner-pattern comps: Opus 4.7
  https://claude.com/resources/articles/meet-the-winners-of-built-with-opus-4-7-claude-code-hackathon
  · Opus 4.6
  https://claude.com/blog/meet-the-winners-of-our-built-with-opus-4-6-claude-code-hackathon
  · LlamaCon https://ai.meta.com/blog/llamacon-hackathon/
* Observed in-hackathon builds to avoid colliding with:
  https://github.com/somnathpal-max/chronolens-swe (UI bug fix, indie) ·
  https://github.com/sholaola23/meta-hackathon-2026 (Baseten model id
  `meta-models/muse-spark-open-1026`) · https://github.com/medomohamed22/sketch2ui
  · https://github.com/PatriaAWijaya/Minime-Hackathon (agent orchestrator)
