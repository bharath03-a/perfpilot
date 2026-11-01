"""Modal A100 app: GPU function run_trial().

Layout (SPEC.md M1): one Modal class held to a single container
(max_containers=1, long scaledown) so a whole bisect stays on one
machine; venvs + model weights cached on a Modal Volume at /cache;
secrets env-only (read via os.environ inside the function).

``modal`` is imported lazily so CPU-side unit tests and the
orchestrator (which calls run_trial over the network) never need it.
"""

from __future__ import annotations

import os
import time
from typing import Any

from .bench import (
    BENCH_PIN_VERSION,
    WARMUP_PROMPTS,
    Workload,
    build_bench_command,
    result_path_for,
    run_bench,
    summarize_runs,
)
from .env import (
    TOOLCHAIN_ROOT,
    cuda_home,
    ensure_bench_env,
    ensure_server_env,
    toolchain_paths,
    venv_python,
)
from .schemas import RunMetrics, TrialResult
from .server import (
    DEFAULT_PORT,
    HEALTH_TIMEOUT_S,
    build_serve_command,
    kill_process_group,
    log_tail,
    start_server,
    wait_for_health,
)

APP_NAME = "perfpilot-worker"
VOLUME_NAME = "perfpilot-cache"
CACHE_MOUNT = "/cache"
BENCH_TIMEOUT_S = 3600.0

INJECT_FLAGS: dict[str, list[str]] = {
    "enforce-eager": ["--enforce-eager"],
    "gpu-memory-utilization-0.5": ["--gpu-memory-utilization", "0.5"],
}

VALID_INJECTS = frozenset([None, *INJECT_FLAGS])


def gpu_uuid() -> str:
    """UUID of GPU 0 from nvidia-smi -L. Raises on any failure."""
    import subprocess

    proc = subprocess.run(
        ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=60.0
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"nvidia-smi -L failed (rc={proc.returncode}): {proc.stderr.strip()[-300:]}"
        )
    line = proc.stdout.strip().splitlines()
    if not line:
        raise RuntimeError("nvidia-smi -L returned no GPUs")
    return line[0].strip()


def _get_modal() -> Any:
    try:
        import modal  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "modal is required on the GPU worker only; "
            "install it to deploy perfpilot/worker/modal_app.py"
        ) from exc
    return modal


def _check_same_gpu(cached_uuid: str | None, fresh_uuid: str) -> None:
    if cached_uuid is not None and cached_uuid != fresh_uuid:
        raise RuntimeError(
            f"GPU changed mid-run: {cached_uuid} -> {fresh_uuid}. "
            "Refusing to compare across GPUs; re-measure references."
        )


def run_trial_local(
    commit: str,
    flags: dict[str, Any],
    runs: int,
    model: str,
    workload: Workload,
    inject: str | None = None,
    port: int = DEFAULT_PORT,
    log_dir: str = "/tmp/perfpilot-logs",
) -> TrialResult:
    """Full trial orchestration. Runs on the Modal container; also the
    documented flow for a local-GPU dry run. Never raises for a bad
    commit: no wheel / failed install -> status="skip"."""
    if inject not in VALID_INJECTS:
        return TrialResult(
            commit=commit,
            flags=flags,
            status="error",
            reason=f"unknown inject {inject!r}",
            gpu_seconds=0.0,
        )
    if runs < 1:
        return TrialResult(
            commit=commit,
            flags=flags,
            status="error",
            reason=f"runs must be >= 1, got {runs}",
            gpu_seconds=0.0,
        )
    t0 = time.monotonic()
    os.makedirs(log_dir, exist_ok=True)
    try:
        uuid = gpu_uuid()
    except RuntimeError as exc:
        return TrialResult(
            commit=commit,
            flags=flags,
            status="error",
            reason=str(exc),
            gpu_seconds=time.monotonic() - t0,
        )

    import shutil as _shutil

    try:
        _du = _shutil.disk_usage(CACHE_MOUNT)
        print(
            f"[perfpilot] trial start commit={commit[:12]} runs={runs} "
            f"cache_free_gb={_du.free / 1e9:.1f}",
            flush=True,
        )
    except OSError:
        print(f"[perfpilot] trial start commit={commit[:12]} runs={runs}", flush=True)
    server = ensure_server_env(commit)
    print(
        f"[perfpilot] server env cached={server.cached} "
        f"skipped={server.skipped} vllm={server.vllm_version}",
        flush=True,
    )
    if server.skipped:
        return TrialResult(
            commit=commit,
            flags=flags,
            status="skip",
            reason=server.reason,
            gpu_uuid=uuid,
            gpu_seconds=time.monotonic() - t0,
        )
    try:
        bench = ensure_bench_env(BENCH_PIN_VERSION)
    except RuntimeError as exc:
        return TrialResult(
            commit=commit,
            flags=flags,
            status="error",
            reason=str(exc),
            gpu_uuid=uuid,
            vllm_version=server.vllm_version,
            gpu_seconds=time.monotonic() - t0,
        )

    base_url = f"http://127.0.0.1:{port}"
    server_log = os.path.join(log_dir, f"serve-{commit[:12]}.log")
    extra = list(INJECT_FLAGS[inject]) if inject else []
    cmd = build_serve_command(venv_python(server.venv), model, port, flags, extra)
    print(f"[perfpilot] model={model!r} serve_argv={cmd}", flush=True)
    child_env = dict(os.environ)
    child_env["HF_HOME"] = f"{CACHE_MOUNT}/hf"
    # Uniform sampler backend: flashinfer's JIT sampler needs a matched
    # nvcc/CCCL pair that nightly venvs don't guarantee; =0 selects vLLM's
    # native sampler on EVERY trial, so comparisons stay apples-to-apples.
    child_env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    _ch, _tbin = toolchain_paths(TOOLCHAIN_ROOT)
    if not _ch:
        _ch = cuda_home(server.venv)
        _tbin = os.path.join(server.venv, "bin")
    if _ch:
        child_env["CUDA_HOME"] = _ch
        _path = f"{_ch}/bin"
        if _tbin:
            _path += f":{_tbin}"
        child_env["PATH"] = f"{_path}:{child_env.get('PATH', '')}"
        print(f"[perfpilot] CUDA_HOME={_ch}", flush=True)
    proc = start_server(cmd, server_log, env=child_env)
    metrics: list[RunMetrics] = []
    try:
        print(f"[perfpilot] server started, polling /health: {server_log}", flush=True)
        health = wait_for_health(
            base_url,
            timeout_s=HEALTH_TIMEOUT_S,
            proc=proc,
            log_path=server_log,
        )
        if not health.ready:
            return TrialResult(
                commit=commit,
                flags=flags,
                status="error",
                reason=f"server unhealthy: {health.reason}; "
                f"exit={proc.returncode}; log tail: {log_tail(server_log, 16000)}",
                gpu_uuid=uuid,
                vllm_version=server.vllm_version,
                gpu_seconds=time.monotonic() - t0,
            )
        try:
            _check_same_gpu(uuid, gpu_uuid())
        except RuntimeError as exc:
            return TrialResult(
                commit=commit,
                flags=flags,
                status="error",
                reason=str(exc),
                gpu_uuid=uuid,
                vllm_version=server.vllm_version,
                gpu_seconds=time.monotonic() - t0,
            )
        bench_py = venv_python(bench.venv)
        warm_cmd = build_bench_command(
            bench_py,
            model,
            base_url,
            workload,
            log_dir,
            f"warmup-{commit[:12]}",
            num_prompts_override=WARMUP_PROMPTS,
        )
        try:
            run_bench(
                warm_cmd,
                result_path_for(log_dir, f"warmup-{commit[:12]}"),
                BENCH_TIMEOUT_S,
            )
        except Exception as exc:
            return TrialResult(
                commit=commit,
                flags=flags,
                status="error",
                reason=(
                    f"warmup bench failed: {exc}; "
                    f"log tail: {log_tail(server_log, 16000)}"
                ),
                gpu_uuid=uuid,
                vllm_version=server.vllm_version,
                gpu_seconds=time.monotonic() - t0,
            )
        for i in range(runs):
            name = f"run-{commit[:12]}-{i}"
            bench_cmd = build_bench_command(
                bench_py,
                model,
                base_url,
                workload,
                log_dir,
                name,
                seed_override=workload.seed + i,
            )
            try:
                metrics.append(
                    run_bench(
                        bench_cmd, result_path_for(log_dir, name), BENCH_TIMEOUT_S
                    )
                )
            except Exception as exc:
                return TrialResult(
                    commit=commit,
                    flags=flags,
                    status="error",
                    reason=f"bench run {i} failed: {exc}; "
                    f"log tail: {log_tail(server_log, 16000)}",
                    gpu_uuid=uuid,
                    vllm_version=server.vllm_version,
                    runs=metrics,
                    gpu_seconds=time.monotonic() - t0,
                )
    finally:
        kill_process_group(proc)

    summary = summarize_runs(metrics)
    reason = ""
    if summary["valid_runs"] == 0 and metrics:
        return TrialResult(
            commit=commit,
            flags=flags,
            status="error",
            reason="every run had failed requests",
            gpu_uuid=uuid,
            vllm_version=server.vllm_version,
            runs=metrics,
            gpu_seconds=time.monotonic() - t0,
        )
    if summary["valid_runs"] < len(metrics):
        reason = (
            f"{len(metrics) - int(summary['valid_runs'])} of "
            f"{len(metrics)} runs had failed requests and were excluded"
        )
    return TrialResult(
        commit=commit,
        flags=flags,
        status="ok",
        reason=reason,
        gpu_uuid=uuid,
        vllm_version=server.vllm_version,
        runs=metrics,
        gpu_seconds=time.monotonic() - t0,
    )


try:
    import modal as _modal

    _volume = _modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    _image = (
        _modal.Image.debian_slim(python_version="3.12")
        .pip_install("uv", "pydantic>=2.0", "numpy>=1.26", "scipy>=1.11")
        .add_local_python_source("perfpilot")
    )
    # Unconditional: a conditional Secret reference evaluates differently in
    # local vs remote environments and breaks hydration. Create once with:
    #   modal secret create huggingface HF_TOKEN=<token>
    _secrets = [_modal.Secret.from_name("huggingface")]

    app: Any = _modal.App(APP_NAME)

    @app.cls(
        # Pinned 40GB variant: bare "A100" floats between 40GB and 80GB
        # ($2.10 vs $2.50/hr) and breaks the same-GPU rule across trials.
        gpu="A100-40GB",
        max_containers=1,
        # 10 min: driver calls are seconds apart; an hour of warm idle
        # after each session burned $2.10 a pop with no benefit.
        scaledown_window=600,
        volumes={CACHE_MOUNT: _volume},
        timeout=7200,
        retries=0,
        image=_image,
        secrets=_secrets,
    )
    class Worker:
        @_modal.method()
        def du_cache_remote(self) -> str:
            import shutil as _shutil
            import subprocess as _sp

            total, used, free = _shutil.disk_usage(CACHE_MOUNT)
            proc = _sp.run(
                f"du -sh {CACHE_MOUNT}/envs {CACHE_MOUNT}/hf; "
                f"du -sh {CACHE_MOUNT}/envs/*",
                shell=True,
                capture_output=True,
                text=True,
                timeout=300.0,
            )
            return (
                f"cache total={total / 1e9:.1f}GB "
                f"used={used / 1e9:.1f}GB free={free / 1e9:.1f}GB\n" + proc.stdout
            )

        @_modal.method()
        def run_trial(
            self,
            commit: str,
            flags: dict[str, Any],
            runs: int,
            model: str,
            workload: dict[str, Any],
            inject: str | None = None,
        ) -> dict[str, Any]:
            result = run_trial_local(
                commit, flags, runs, model, Workload(**workload), inject
            )
            _volume.commit()
            return result.to_spec_json()

        @_modal.method()
        def answer_batch(
            self,
            commit: str,
            flags: dict[str, Any],
            model: str,
            prompts: list[str],
        ) -> list[str]:
            """Greedy completions for quality-guard questions.

            Starts the server once, POSTs each prompt (temperature 0,
            32 tokens), tears down. Raises RuntimeError on any failure.
            """
            import json as _json
            import urllib.request as _url

            from .quality import MAX_NEW_TOKENS

            server = ensure_server_env(commit)
            if server.skipped:
                raise RuntimeError(f"server env skipped: {server.reason}")
            port = DEFAULT_PORT
            base_url = f"http://127.0.0.1:{port}"
            log_dir = "/tmp/perfpilot-logs"
            os.makedirs(log_dir, exist_ok=True)
            server_log = os.path.join(log_dir, f"quality-{commit[:12]}.log")
            cmd = build_serve_command(venv_python(server.venv), model, port, flags, [])
            child_env = dict(os.environ)
            child_env["HF_HOME"] = f"{CACHE_MOUNT}/hf"
            child_env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
            _ch, _tbin = toolchain_paths(TOOLCHAIN_ROOT)
            if _ch:
                child_env["CUDA_HOME"] = _ch
                _path = f"{_ch}/bin"
                if _tbin:
                    _path += f":{_tbin}"
                child_env["PATH"] = f"{_path}:{child_env.get('PATH', '')}"
            proc = start_server(cmd, server_log, env=child_env)
            try:
                health = wait_for_health(
                    base_url,
                    timeout_s=HEALTH_TIMEOUT_S,
                    proc=proc,
                    log_path=server_log,
                )
                if not health.ready:
                    raise RuntimeError(
                        f"server unhealthy: {health.reason}; "
                        f"log tail: {log_tail(server_log, 16000)}"
                    )
                texts: list[str] = []
                for prompt in prompts:
                    body = _json.dumps(
                        {
                            "model": model,
                            "prompt": prompt,
                            "max_tokens": MAX_NEW_TOKENS,
                            "temperature": 0.0,
                        }
                    ).encode()
                    req = _url.Request(
                        base_url + "/v1/completions",
                        data=body,
                        headers={"Content-Type": "application/json"},
                    )
                    with _url.urlopen(req, timeout=120.0) as resp:
                        data = _json.load(resp)
                    texts.append(str(data["choices"][0].get("text", "")))
                return texts
            finally:
                kill_process_group(proc)

    @app.local_entrypoint()
    def du_cache() -> None:
        """Print /cache usage and per-venv sizes. Costs cents."""
        print(Worker().du_cache_remote.remote(), flush=True)

    @app.local_entrypoint()
    def main(
        commit: str,
        model: str,
        runs: int = 3,
        workload_json: str = "{}",
        flags_json: str = "{}",
        inject: str | None = None,
    ) -> None:
        """One GPU trial: modal run perfpilot/worker/modal_app.py --commit SHA
        --model <id> [--runs 3]. Prints the TrialResult JSON."""
        import json as _json

        result = Worker().run_trial.remote(
            commit=commit,
            flags=_json.loads(flags_json),
            runs=runs,
            model=model,
            workload=_json.loads(workload_json),
            inject=inject,
        )
        print(_json.dumps(result, indent=2))
except ImportError:
    app = None
    Worker = None
