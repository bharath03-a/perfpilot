"""Fixed pinned benchmark client: run ``vllm bench serve`` N times, parse JSON.

Flag names verified against vllm/benchmarks/serve.py and
vllm/benchmarks/datasets/datasets.py at main c7a683bde (2026-10-31):
--backend/--model/--endpoint, --dataset-name/--random-input-len/
--random-output-len/--num-prompts/--seed, --max-concurrency,
--save-result/--result-dir/--result-filename.

The pinned bench client is vLLM 0.32.0 (latest stable 2026-10-31);
every flag above and every key below was re-verified in the v0.32.0 tag
source (vllm/benchmarks/serve.py + datasets/datasets.py).

Saved-JSON keys at that SHA: duration, completed, failed,
total_input/output_tokens, request/output/total_token_throughput,
mean/median/std/p99_ttft_ms, mean/median/p99_tpot_ms, itl + e2el
aggregates (defaults: --percentile-metrics ttft,tpot,itl,
--metric-percentiles 99). Older commits may lack some flags or keys;
the normalizer accepts aliases and fails loudly listing actual keys.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .schemas import RunMetrics

BENCH_PIN_VERSION = "0.32.0"
BENCH_BACKEND = "vllm"
BENCH_ENDPOINT = "/v1/completions"
WARMUP_PROMPTS = 5


@dataclass
class Workload:
    dataset_name: str = "random"
    input_len: int = 1024
    output_len: int = 128
    num_prompts: int = 100
    max_concurrency: int = 32
    seed: int = 0


def build_bench_command(
    bench_py: str,
    model: str,
    base_url: str,
    workload: Workload,
    result_dir: str,
    result_filename: str,
    num_prompts_override: int | None = None,
    seed_override: int | None = None,
) -> list[str]:
    # vllm bench serve uses --result-filename VERBATIM (no .json appended),
    # so the extension must be part of the value to match result_path_for.
    filename = (
        result_filename
        if result_filename.endswith(".json")
        else result_filename + ".json"
    )
    return [
        bench_py,
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
        "serve",
        "--backend",
        BENCH_BACKEND,
        "--model",
        model,
        "--base-url",
        base_url,
        "--endpoint",
        BENCH_ENDPOINT,
        "--dataset-name",
        workload.dataset_name,
        "--random-input-len",
        str(workload.input_len),
        "--random-output-len",
        str(workload.output_len),
        "--num-prompts",
        str(
            num_prompts_override
            if num_prompts_override is not None
            else workload.num_prompts
        ),
        "--max-concurrency",
        str(workload.max_concurrency),
        "--seed",
        str(seed_override if seed_override is not None else workload.seed),
        "--save-result",
        "--result-dir",
        result_dir,
        "--result-filename",
        filename,
        "--disable-tqdm",
    ]


def _num(doc: Mapping[str, object], *keys: str) -> float | None:
    for key in keys:
        if key in doc and isinstance(doc[key], (int, float)):
            return float(doc[key])
    return None


def _pct(doc: Mapping[str, object], stem: str, pct: int = 99) -> float | None:
    for key in (f"p{pct}_{stem}_ms", f"p{pct}.0_{stem}_ms", f"p{pct}_{stem}"):
        if key in doc and isinstance(doc[key], (int, float)):
            return float(doc[key])
    return None


def normalize_bench_result(doc: Mapping[str, object]) -> RunMetrics:
    """Normalize one saved bench JSON to RunMetrics. Loud on missing keys."""
    missing: list[str] = []
    throughput = _num(doc, "output_throughput")
    if throughput is None:
        missing.append("output_throughput")
    mean_ttft = _num(doc, "mean_ttft_ms")
    if mean_ttft is None:
        missing.append("mean_ttft_ms")
    p99_ttft = _pct(doc, "ttft")
    if p99_ttft is None:
        missing.append("p99_ttft_ms")
    median_tpot = _num(doc, "median_tpot_ms", "mean_tpot_ms")
    if median_tpot is None:
        missing.append("median_tpot_ms")
    failed = _num(doc, "failed", "failed_requests", "num_failed_requests")
    if failed is None:
        missing.append("failed")
    if missing:
        keys = sorted(str(k) for k in doc.keys())
        raise ValueError(
            f"bench JSON missing {missing}; have keys={keys}. "
            "Pin the parser to the first real run."
        )
    assert throughput is not None and mean_ttft is not None
    assert p99_ttft is not None and median_tpot is not None and failed is not None
    return RunMetrics(
        output_throughput=throughput,
        mean_ttft_ms=mean_ttft,
        p99_ttft_ms=p99_ttft,
        median_tpot_ms=median_tpot,
        failed_requests=int(failed),
    )


def read_result_file(path: str) -> Mapping[str, object]:
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    if isinstance(doc, list):
        if not doc:
            raise ValueError(f"bench result {path} is an empty list")
        doc = doc[0]
    if not isinstance(doc, dict):
        raise ValueError(f"bench result {path} is not a JSON object")
    return doc


def run_bench(
    cmd: list[str],
    result_path: str,
    timeout_s: float,
) -> RunMetrics:
    """Run one bench invocation, return normalized metrics. Raises loudly."""
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    if proc.returncode != 0:
        raise RuntimeError(
            f"vllm bench serve failed (rc={proc.returncode}): "
            f"{proc.stderr.strip()[-1000:]}"
        )
    if not os.path.isfile(result_path):
        raise RuntimeError(
            f"bench exited 0 but wrote no result at {result_path}; "
            f"stderr tail: {proc.stderr.strip()[-500:]}"
        )
    return normalize_bench_result(read_result_file(result_path))


def result_path_for(result_dir: str, filename: str) -> str:
    name = filename if filename.endswith(".json") else filename + ".json"
    return os.path.join(result_dir, name)


def summarize_runs(runs: Sequence[RunMetrics]) -> dict[str, float | int]:
    valid = [r for r in runs if r.valid]
    throughputs = sorted(r.output_throughput for r in valid)
    out: dict[str, float | int] = {"runs": len(runs), "valid_runs": len(valid)}
    if throughputs:
        out["min_throughput"] = throughputs[0]
        out["max_throughput"] = throughputs[-1]
        out["spread_pct"] = (
            100.0 * (throughputs[-1] - throughputs[0]) / throughputs[0]
            if throughputs[0] > 0
            else float("inf")
        )
    return out
