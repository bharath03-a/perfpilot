# PerfPilot: Build Spec for Muse Code

Source: `PerfPilot Build Spec for Muse Code.pdf` (14 pages, Oct 30, 2026).
Faithful Markdown export. Where PLAN.md narrows the spec (A100-only GPU,
16-commit / 2-trial eval, prompt count cut to 100), PLAN.md governs the build.

PerfPilot is an agent that keeps a vLLM deployment of a Meta open model fast.
It has one benchmark-and-verify engine and two modes: PerfBisect searches
commits to find what made serving slower, and TuneSmith searches config flags
to find what makes it faster. The agent's reasoning runs on Muse Spark Open,
and the model being served and measured is a Meta open model.

How to use this spec: export it as Markdown, save it in the repo as SPEC.md,
and give Muse Code one milestone at a time from the milestones section. Each
milestone has acceptance criteria; don't start the next until the current one
passes. Items under "Verify first" are assumptions to confirm before building
on them.

## Scope

In scope:

- vLLM only, on one NVIDIA GPU type per run (spec PDF says Modal H100;
  PLAN.md and project instruction override to Modal A100 only)
- One Meta open model as the served workload
- PerfBisect over a range of vLLM commits on main that have pre-built wheels
- TuneSmith over 5 vllm serve knobs, with an output-quality guard
- A CLI, an event log, and a single-page HTML report
- An eval with injected regressions

Out of scope (roadmap only, mention in the video):

- SGLang, TensorRT-LLM or other engines
- Multi-GPU and multi-node setups
- Building vLLM from source for commits without wheels
- A hosted service, accounts or a database
- Continuous monitoring or CI integration

## Architecture

The orchestrator owns every decision that changes state: the agent only
proposes, and the orchestrator checks each proposal before running it. The GPU
worker is stateless apart from its caches, so a crashed run can be retried.

## Repo layout and config

Python 3.12, managed with uv. One package, perfpilot.

```text
perfpilot/
  cli.py            # typer CLI: bisect, tune, report, eval
  config.py         # pydantic models for perfpilot.yaml
  orchestrator.py   # runs a mode, validates agent proposals, writes events
  stats.py          # repeated-run verdicts (pure functions, unit-tested)
  modes/
    bisect.py       # PerfBisect state machine
    tune.py         # TuneSmith search
  agent/
    client.py       # OpenAI-compatible client for Muse Spark Open
    prompts.py      # system prompts, one per decision type
    tools.py        # JSON schemas for the agent's allowed actions
  worker/
    modal_app.py    # Modal app: GPU function run_trial()
    env.py          # venv per commit from vLLM wheels, cached
    server.py       # start / health-check / stop vllm serve
    bench.py        # run vllm bench serve, parse JSON
    quality.py      # greedy-decoding answer check
  git_info.py       # commit list and diffs via GitHub API
  report/
    render.py       # events.jsonl -> report.html
    template.html
  evals/
    inject.py       # injected-regression overlays
    run_eval.py
tests/
runs/               # one folder per run: events.jsonl, results/, report.html
SPEC.md
README.md
```

Config: `perfpilot.yaml`

```yaml
model: <meta-open-model-id>        # served by vLLM; verify size fits one GPU
gpu: H100                          # PLAN.md overrides: A100
workload:
  dataset: random
  input_len: 1024
  output_len: 128
  num_prompts: 200                 # PLAN.md: cut to 100
  max_concurrency: 32
  seed: 0
metrics:
  primary: output_throughput      # higher is better
  guards: [p99_ttft_ms]           # must not regress beyond threshold
stats:
  min_runs: 3
  max_runs: 8
  confidence: 0.90
bisect:
  repo: vllm-project/vllm
  good: <commit-sha>
  bad: <commit-sha>
tune:
  base_commit: <commit-sha>
  knobs:
    max_num_seqs: [64, 128, 256, 512]
    max_num_batched_tokens: [2048, 4096, 8192, 16384]
    async_scheduling: [false, true]
    enable_prefix_caching: [false, true]
    kv_cache_dtype: [auto, fp8]
  quality:
    max_accuracy_drop_pts: 2
agent:
  base_url: ${PERFPILOT_LLM_BASE_URL}
  model: meta-models/muse-spark-open-1026
  api_key: ${PERFPILOT_LLM_API_KEY}
```

The benchmark command the worker runs is built from workload, using vllm
bench serve flags that exist in recent versions: `--dataset-name random`,
`--random-input-len`, `--random-output-len`, `--num-prompts`,
`--max-concurrency`, `--seed`, `--save-result`, `--result-dir` (vLLM bench
serve docs). Older commits may lack some flags; see "Verify first".

## Core engine

Both modes call one function:
`run_trial(commit, flags, runs, inject=None) -> TrialResult`.
Everything else is search logic on top.

### Environments, one per commit

1. Create `/cache/envs/<sha>` with uv venv.
2. Install that commit's wheel: `uv pip install vllm --extra-index-url
   https://wheels.vllm.ai/<sha> --index-strategy first-index`. The URL may
   need a CUDA variant subdirectory; confirm on the first run.
3. Cache the venv on a Modal Volume keyed by SHA, so revisiting a commit
   costs nothing.
4. If install fails or no wheel exists, return `status="skip"` with the
   reason. Never build from source.
5. Cache model weights once on the Volume (`HF_HOME=/cache/hf`).

### Server

- Start `vllm serve <model> --port 8000 --seed 0 <flags>` from the commit's
  venv, in its own process group, with logs saved to the run folder.
- Poll `/health` until ready; time out after 15 minutes and return
  `status="error"` with the log tail.
- Warm up with one short benchmark and discard its result.
- Always kill the process group afterwards, even on error.

### Benchmark

- Use one fixed benchmark client for every commit, installed in its own venv
  at a pinned vLLM version. Only the server changes between commits; if the
  measuring tool changed too, a regression could be in the ruler.
- Run `vllm bench serve` runs times with the workload from the config,
  saving JSON per run.
- Normalize each run to `{output_throughput, mean_ttft_ms, p99_ttft_ms,
  median_tpot_ms, failed_requests}`. Check the exact JSON key names on the
  first real run and fix the parser to match.
- Any failed requests make that run invalid.

### Same GPU, every time

Run the worker as a Modal class held to one container (one container max,
long scaledown window), so a whole bisect stays on one machine. Record the
GPU UUID from `nvidia-smi -L` in every result. The stats layer refuses to
compare results from different GPUs and asks for references to be re-measured
instead.

### Verdict statistics (stats.py)

For a metric with samples from a candidate c and a reference g:

- Effect: `r = ln(mean_c / mean_g)`
- Standard error:
  `se = sqrt(s_c^2 / (n_c * mean_c^2) + s_g^2 / (n_g * mean_g^2))`
- Welch–Satterthwaite degrees of freedom; interval `r ± t(df) * se` at the
  configured confidence

This is the log-ratio Welch interval. Every verdict returns the interval and
a reason string, never just a label. Unit-test it with synthetic samples
where the true answer is known.

## Mode 1: PerfBisect

Goal: given a good commit and a slower bad commit, find the first commit that
is slow, prove it, and explain it.

1. List commits. Get the first-parent commits on main from good to bad via
   the GitHub compare API, oldest first.
2. Confirm there's a regression. Measure good and bad with min_runs each.
   Compute the full effect `R = ln(bad / good)` on the primary metric. If the
   interval for R includes zero, or |R| is under 3%, stop and report "no
   confirmed regression". That's an honest, valid outcome.
3. Set the midpoint. Use `m = R / 2`. A commit is bad if its whole interval
   for `ln(commit / good)` is past m toward bad, good if it's entirely on the
   good side, and inconclusive otherwise.
4. Loop while more than one commit separates good and bad:
   a. Default choice: the middle untested, unskipped commit. The agent may
      propose another commit from the same range, with a reason, such as a
      commit message touching the scheduler. The orchestrator rejects any
      proposal outside the range and falls back to the middle.
   b. Run `run_trial(commit, runs=min_runs)`.
   c. If inconclusive, the agent proposes how many runs to add, capped at
      max_runs. Still inconclusive at the cap means mark the commit ambiguous
      and treat it like a skip.
   d. If the trial returns skip, record the reason and choose another commit.
   e. Move the good or bad boundary to this commit.
   f. Every 4 steps, re-measure the good commit with 2 runs. If it has
      drifted more than 2%, re-measure both endpoints before continuing.
5. Name the culprit: the first bad commit. If skipped commits sit between the
   last good and first bad, report a culprit range instead, listing the
   skipped commits.
6. Verify. Re-run the culprit and its parent back to back with max_runs each.
   Mark the result verified only if the difference is significant and in the
   same direction as R.
7. Explain. Fetch the culprit's diff. The agent writes what changed, why it
   could slow this workload, and its confidence, citing file paths from the
   diff. The orchestrator checks every cited path is actually in the diff.
8. Draft a fix (stretch). Draft a revert or minimal fix, saved as a .patch
   file. It's not applied or measured unless time allows.
9. Hand off to TuneSmith on the culprit commit, with the goal of recovering
   the good commit's speed through flags.

## Mode 2: TuneSmith

Goal: on one fixed commit, find the vllm serve flags that maximize the
primary metric without breaking latency guards or output quality.

1. Baseline. Measure default flags with min_runs.
2. Check which knobs exist. Run `vllm serve --help` in that commit's venv and
   drop any knob or value it doesn't support. Log what was dropped.
3. Search one knob at a time. The full grid is about 128 combinations, too
   many for the GPU budget. Instead:
   a. The agent ranks knobs from the baseline numbers. For example, high TTFT
      under load suggests raising max_num_seqs first.
   b. The agent proposes one change at a time as JSON:
      `{knob, value, reason}`. The orchestrator rejects values not in the
      config, and configs already tried.
   c. Each candidate gets min_runs. Accept it as the new best only if the
      interval for `ln(candidate / best)` is entirely above +2%, and p99 TTFT
      is no more than 10% worse.
   d. Stop at max_trials (default 12; PLAN.md narrows to 8, M4 to 6) or when
      no knob improves.
4. Quality guard. Any config that changes numerics, such as
   `kv_cache_dtype=fp8`, must pass the quality check: 50 fixed short-answer
   questions with known answers in the repo, greedy decoding, max 32 tokens.
   Reject the config if accuracy drops more than 2 points versus the baseline
   flags. The final config always gets one quality check.
5. Confirm. Re-run the best config and the baseline with max_runs each.
   Report the uplift with its interval.
6. Output. The recommended flags as a ready-to-paste vllm serve command,
   plus every tried config and why it was kept or rejected.

Workaround mode, after PerfBisect: run TuneSmith on the culprit commit and
report what share of the regression the best flags recover, against the good
commit's speed.

## The agent

The agent is Muse Spark Open on Baseten, called through an OpenAI-compatible
chat completions client (`agent/client.py`). It never runs commands. It
answers one narrow question at a time with JSON, and the orchestrator decides
whether to accept the answer.

### Decision types

| Decision | Input | Output JSON | Fallback if invalid |
|---|---|---|---|
| pick_commit | Untested range, commit messages, results so far | {sha, reason} | Middle commit |
| extra_runs | Current interval and run count | {runs, reason}, runs 1–3 | 2 runs |
| explain_culprit | Commit message, diff excerpt, before/after metrics | {summary, mechanism, files[], confidence} | "No explanation", culprit still reported |
| rank_knobs | Baseline metrics, supported knobs | {order[], reason} | Config order |
| propose_config | Tried configs and results | {knob, value, reason} | Next untried value in order |
| draft_fix | Culprit diff | Unified diff patch | Skip the fix |

### Guardrails

- Validate every response against a pydantic model. On failure, retry once
  with the validation error appended, then use the fallback.
- 60-second timeout per call; on API error, use the fallback. A run never
  stalls on the agent.
- Strip any reasoning text the model returns before parsing JSON.
- Keep context small: pass history as a compact table of commit, runs, effect
  and interval, not raw logs.
- Log every call to events.jsonl: decision type, the proposal, accepted or
  rejected, and why. This log powers the agent panel in the report and the
  demo video.

### Prompt pattern

Each decision type gets a short system prompt in `agent/prompts.py`. Example
for pick_commit:

```text
You are choosing the next vLLM commit to benchmark in a performance bisect.
The regression is a {effect}% drop in {metric} between {good} and {bad}.
Untested commits, oldest first, with messages: {table}
Results so far: {history}
Return JSON only: {"sha": "<one sha from the list>", "reason": "<one sentence>"}.
Prefer the middle of the range unless a commit message clearly points at code
this workload exercises (scheduler, attention, sampling, KV cache, CUDA graphs).
```

## Data files, CLI and report

Run folder: `runs/<run_id>/`

- `config.yaml`: the exact config used
- `events.jsonl`: one event per line, the source of truth for the report
- `results/<sha>__<flags_hash>/run_<n>.json`: raw benchmark output per run
- `logs/`: server logs per trial
- `report.html`, plus `fix.patch` if drafted

### TrialResult

```json
{
  "commit": "abc1234",
  "flags": {"max_num_seqs": 256},
  "status": "ok | skip | error",
  "reason": "",
  "gpu_uuid": "GPU-...",
  "vllm_version": "0.x.y.devN+gabc1234",
  "runs": [{"output_throughput": 0.0, "mean_ttft_ms": 0.0, "p99_ttft_ms": 0.0,
             "median_tpot_ms": 0.0, "failed_requests": 0}],
  "gpu_seconds": 0.0
}
```

### Event types

`run_start`, `trial_start`, `trial_result`, `verdict`, `agent_call`,
`boundary_update`, `culprit`, `verify`, `explain`, `tune_candidate`,
`tune_accept`, `quality_check`, `run_end`. Every event has ts, run_id
and type.

### CLI

```text
perfpilot bisect --config perfpilot.yaml [--good SHA --bad SHA]
perfpilot tune   --config perfpilot.yaml [--commit SHA] [--workaround-for RUN_ID]
perfpilot report runs/<run_id>
perfpilot watch  runs/<run_id>     # live page that refreshes as events arrive
perfpilot eval   --trials 5        # PLAN.md narrows: --trials 2, range of 16
```

### Report page

One self-contained HTML file built from events.jsonl, with Chart.js from
a CDN.

1. Verdict card: culprit commit, regression size with interval, verified or
   not, GPU minutes and cost.
2. Bisect timeline: the range narrowing step by step, each commit colored
   good, bad, skipped or ambiguous.
3. Throughput chart: each tested commit in order, with its interval.
4. Agent panel: every proposal, whether it was accepted, and why.
5. Explanation: the agent's summary, with links to the diff files.
6. TuneSmith: tried configs, the winner, and a ready-to-paste vllm serve
   command.
7. Reproducibility: GPU UUID, vLLM versions, benchmark client version, config.

`perfpilot watch` is the demo view: the same page, refreshing every few
seconds while a run is in progress.

## Eval with injected regressions

The agent has to find culprits you planted before anyone trusts it on a real
one.

How injection works. In eval mode, the worker secretly adds a slowdown flag
to the server for every commit at or after a hidden culprit index. The agent
and orchestrator never see the flag; they only see the numbers. Two injection
types:

- `--enforce-eager` (disables CUDA graphs): a strong, version-stable slowdown
- `--gpu-memory-utilization 0.5`: a milder, workload-dependent slowdown

On Friday, measure the size of each on your model and workload. Use only
injections that cause at least a 10% drop.

Protocol (spec; PLAN.md narrows to a 16-commit range and 2 trials):

1. Fix one range of 32–64 consecutive main commits that all have wheels.
2. Run 5 trials with seeded culprit positions, including the first and last
   commit in the range.
3. Reuse the cached venvs across trials, but never reuse benchmark results,
   since injection changes them.
4. Skip the explanation step in eval trials; the real diff doesn't contain
   the planted change.

Report these in the README and the video:

| Metric | Pass bar |
|---|---|
| Correct culprit | 5 of 5 (PLAN.md: 2 of 2) |
| Wrong verdicts at any step | 0 |
| Bisect steps vs range size | About log2 of the range |
| Inconclusive steps and extra runs used | Reported |
| GPU minutes and cost per trial | Reported |

## Milestones

Give Muse Code one milestone at a time. Don't start the next until every
acceptance check passes.

- M0 · Fri: skeleton and statistics. Package layout, `perfpilot.yaml`
  loading with pydantic, typer CLI stubs. `stats.py`: log-ratio Welch
  interval and the good / bad / inconclusive classifier. Accept: pytest
  passes, including synthetic cases with a known 0%, 5% and 20% effect
  classified correctly.
- M1 · Fri–Sat: one trial on a GPU. Modal worker: env per commit from
  wheels, server start and stop, warmup, fixed benchmark client, N runs.
  TrialResult with GPU UUID and GPU seconds. Accept: two calls for the same
  commit land on the same GPU, the second skips the install, and throughput
  variation across runs is reported. Accept: a commit with no wheel returns
  skip with a reason instead of crashing.
- M2 · Sat: bisect without the agent. Deterministic midpoint bisect,
  endpoint confirmation, skip handling, drift re-checks. events.jsonl written
  for every step. Injection support for eval mode. Accept: one injected
  trial finds the planted culprit.
- M3 · Sun: the agent and the eval. OpenAI-compatible client for Muse Spark
  Open; pick_commit and extra_runs with validation, retry and fallback.
  `perfpilot eval --trials 5` (PLAN.md: 2). Accept: 5 of 5 culprits found
  (PLAN.md: 2 of 2); every agent call logged with accepted or rejected.
  Accept: with the API key unset, a bisect still completes using fallbacks.
- M4 · Mon: verify, explain, tune. Verification re-run; explain_culprit with
  cited-file checks. TuneSmith: knob detection, one-knob-at-a-time search,
  latency guard, quality check, final confirmation. Workaround mode after a
  bisect. Accept: `perfpilot tune` outputs a ready-to-paste vllm serve
  command with its measured uplift and interval. Accept: a config that fails
  the quality check is rejected and logged.
- M5 · Tue: report and story. `perfpilot report` and `perfpilot watch`.
  README: what PerfPilot is, how to run it, eval table, cost, and a "Built
  during the hackathon" section. Attempt the real regression; keep it only
  if it reproduces. Accept: a fresh clone can run
  `perfpilot eval --trials 1` by following the README.
- M6 · Wed: ship. Code freeze by 9 AM Phoenix; record and edit the 3-minute
  video. Submit by noon Phoenix (deadline 7 PM Phoenix); repo public; post
  on X.

Cut order if behind: draft_fix, then watch, then workaround mode. Never cut
verification or the eval. (PLAN.md cut order: workaround → TuneSmith → real
regression → draft_fix → watch. Never cut: verification, eval, video day.)

## Rules for Muse Code

- Python 3.12, uv, type hints everywhere, pydantic for every data model,
  pytest for tests.
- OpenAI-compatible client for Muse Spark Open; pick_commit and extra_runs
  with validation, retry and fallback.
- Keep stats.py and the bisect state machine pure and unit-tested; no network
  or GPU calls inside them.
- The orchestrator is the only component that executes anything. The agent
  returns JSON proposals only.
- Every GPU call is idempotent and cached by (commit, flags_hash, inject).
- Fail loudly with a reason string; never silently fall back without logging
  an event.
- Secrets only from environment variables: PERFPILOT_LLM_BASE_URL,
  PERFPILOT_LLM_API_KEY, GITHUB_TOKEN, plus Modal's own auth.
- All code is new and written in this repo during the hackathon. Don't copy
  code from other projects.

## Verify first

Confirm these on Friday before building on them. Each is an assumption in
this spec.

- Served model size. Does the chosen Meta open model fit and serve on one
  GPU? If Muse Spark Open doesn't, use Muse Glimmer. (PLAN.md: Glimmer-class,
  NOT 816B Spark; A100-only.)
- Agent endpoint. The Baseten base URL, the model ID
  `meta-models/muse-spark-open-1026` from the participant guide, and whether
  reasoning text comes back in the response.
- Wheel URLs. The exact per-commit wheel index format and CUDA variant
  subdirectory, tested on one commit.
- Benchmark JSON keys. The real key names in vllm bench serve output, and
  whether the pinned benchmark client works against older servers.
- Flag support. That the TuneSmith knobs exist in the chosen commit range.
- Modal limits. The function timeout for a long bisect, holding one container
  across calls, and GPU cost per hour against the credit.
- GitHub API. That the compare endpoint returns the whole commit range you
  pick; it may cap the number of commits.
- Injection size. That each injected slowdown causes at least a 10% drop on
  your workload.
