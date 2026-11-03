"""CPU-only tests for the GPU worker (no Modal, no GPU, no network)."""

from __future__ import annotations

import json
import subprocess

import pytest

from perfpilot.worker import env
from perfpilot.worker.bench import (
    BENCH_PIN_VERSION,
    Workload,
    build_bench_command,
    normalize_bench_result,
    read_result_file,
    result_path_for,
    summarize_runs,
)
from perfpilot.worker.env import (
    build_install_command,
    candidate_index_urls,
    classify_install_failure,
    find_wheel_filename,
    is_full_sha,
    venv_path,
    wheel_index_url,
)
from perfpilot.worker.schemas import (
    RunMetrics,
    TrialResult,
    flags_hash,
    trial_cache_key,
)
from perfpilot.worker.server import (
    HEALTH_TIMEOUT_S,
    build_serve_command,
    log_tail,
    wait_for_health,
)

SHA = "0aa72a450c9a98642c698e36a26e26c750400157"
CURRENT_SHA = "c7a683bde5909bf2862d5d4500239dd1e3bd44bc"

CANONICAL_BENCH_JSON = {
    "duration": 120.5,
    "completed": 100,
    "failed": 0,
    "total_input_tokens": 102400,
    "total_output_tokens": 12800,
    "request_throughput": 0.83,
    "request_goodput": None,
    "output_throughput": 106.25,
    "total_token_throughput": 960.0,
    "mean_ttft_ms": 71.54,
    "median_ttft_ms": 73.88,
    "std_ttft_ms": 5.1,
    "p99_ttft_ms": 79.49,
    "mean_tpot_ms": 7.91,
    "median_tpot_ms": 7.96,
    "std_tpot_ms": 0.4,
    "p99_tpot_ms": 8.03,
}


def test_trial_result_matches_spec_shape() -> None:
    r = TrialResult(
        commit="abc1234",
        flags={"max_num_seqs": 256},
        status="ok",
        reason="",
        gpu_uuid="GPU-abc",
        vllm_version="0.32.0",
        runs=[
            RunMetrics(
                output_throughput=100.0,
                mean_ttft_ms=70.0,
                p99_ttft_ms=80.0,
                median_tpot_ms=8.0,
                failed_requests=0,
            )
        ],
        gpu_seconds=600.0,
    )
    doc = r.to_spec_json()
    assert set(doc) == {
        "commit",
        "flags",
        "status",
        "reason",
        "gpu_uuid",
        "vllm_version",
        "runs",
        "gpu_seconds",
    }
    assert set(doc["runs"][0]) == {
        "output_throughput",
        "mean_ttft_ms",
        "p99_ttft_ms",
        "median_tpot_ms",
        "failed_requests",
    }
    assert doc["status"] in ("ok", "skip", "error")


def test_trial_result_rejects_bad_status() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        TrialResult(commit="x", flags={}, status="timeout")  # type: ignore[arg-type]


def test_flags_hash_stable_and_order_insensitive() -> None:
    assert flags_hash({"b": 1, "a": 2}) == flags_hash({"a": 2, "b": 1})
    assert flags_hash({"a": 1}) != flags_hash({"a": 2})
    assert trial_cache_key("c", {"a": 1}, None) == ("c", flags_hash({"a": 1}), None)


def test_valid_runs_filters_failed() -> None:
    ok = RunMetrics(
        output_throughput=1.0,
        mean_ttft_ms=1.0,
        p99_ttft_ms=1.0,
        median_tpot_ms=1.0,
        failed_requests=0,
    )
    bad = RunMetrics(
        output_throughput=1.0,
        mean_ttft_ms=1.0,
        p99_ttft_ms=1.0,
        median_tpot_ms=1.0,
        failed_requests=3,
    )
    assert ok.valid and not bad.valid
    t = TrialResult(commit="c", flags={}, status="ok", runs=[ok, bad])
    assert t.valid_runs == [ok]


def test_wheel_index_url_format() -> None:
    assert wheel_index_url(SHA) == f"https://wheels.vllm.ai/{SHA}"
    with pytest.raises(ValueError):
        wheel_index_url("main")
    with pytest.raises(ValueError):
        wheel_index_url(SHA[:12])
    assert not is_full_sha("v0.32.0") and is_full_sha(SHA)


def test_candidate_index_urls_prefers_per_sha() -> None:
    urls = candidate_index_urls(SHA)
    assert urls[0] == f"https://wheels.vllm.ai/{SHA}"
    assert any("cu130" in u for u in urls)


def test_install_command_never_builds_from_source() -> None:
    cmd = build_install_command(
        "/cache/envs/x/bin/python", f"https://wheels.vllm.ai/{SHA}"
    )
    text = " ".join(cmd)
    assert "--extra-index-url" in cmd and "--index-strategy" in cmd
    assert "first-index" in cmd
    assert "build" not in text and "git+" not in text and "compile" not in text


def test_find_wheel_filename_matches_real_listing() -> None:
    html = (
        '<a href="../../0aa72a450c9a98642c698e36a26e26c750400157/'
        'vllm-0.33.0rc1.dev411%2Bg0aa72a450-cp38-abi3-manylinux_2_28_x86_64.whl">'
        "vllm-0.33.0rc1.dev411+g0aa72a450-cp38-abi3-manylinux_2_28_x86_64.whl</a>"
        '<a href="../../0aa72a450c9a98642c698e36a26e26c750400157/'
        'vllm-0.33.0rc1.dev411%2Bg0aa72a450-cp38-abi3-manylinux_2_28_aarch64.whl">'
        "vllm-0.33.0rc1.dev411+g0aa72a450-cp38-abi3-manylinux_2_28_aarch64.whl</a>"
    )
    hit = find_wheel_filename(html, SHA)
    assert hit is not None and "x86_64" in hit and SHA[:9] in hit
    assert find_wheel_filename(html, SHA, arch="aarch64") is not None
    assert "aarch64" in (find_wheel_filename(html, SHA, arch="aarch64") or "")
    assert find_wheel_filename("<html></html>", SHA) is None
    assert find_wheel_filename(html, CURRENT_SHA) is None


def test_classify_install_failure_skip_paths() -> None:
    assert classify_install_failure(1, "ERROR: 404 Not Found").startswith(
        "no wheel for commit"
    )
    assert classify_install_failure(
        1, "ERROR: No matching distribution found for vllm"
    ).startswith("no wheel for commit")
    assert "timed out" in classify_install_failure(1, "connection timed out")
    assert "rc=2" in classify_install_failure(2, "something else broke")


def test_ensure_server_env_uses_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(env, "is_env_cached", lambda *a: True)
    monkeypatch.setattr(env, "read_vllm_version", lambda venv: "0.32.0.dev1")
    monkeypatch.setattr(env, "read_installed_sha", lambda venv: SHA[:12])
    monkeypatch.setattr(env, "ensure_toolchain", lambda venv: "")
    res = env.ensure_server_env(SHA, env_root=str(tmp_path))
    assert res.cached and not res.skipped and res.vllm_version == "0.32.0.dev1"


def test_ensure_server_env_missing_wheel_is_skip(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(env, "is_env_cached", lambda *a: False)
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], timeout_s: float):
        calls.append(cmd)
        if cmd[:2] == ["uv", "venv"]:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        idx = cmd.index("--extra-index-url") + 1
        return subprocess.CompletedProcess(
            cmd, 1, "", "ERROR: 404 Not Found: " + cmd[idx]
        )

    monkeypatch.setattr(env, "run_command", fake_run)
    res = env.ensure_server_env(SHA, env_root=str(tmp_path))
    assert res.skipped and not res.cached
    assert "no wheel" in res.reason
    assert all("git" not in " ".join(c) and "build" not in " ".join(c) for c in calls)


def test_ensure_server_env_failed_install_is_skip(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(env, "is_env_cached", lambda *a: False)

    def fake_run(cmd: list[str], timeout_s: float):
        if cmd[:2] == ["uv", "venv"]:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return subprocess.CompletedProcess(cmd, 1, "", "torch CUDA mismatch boom")

    monkeypatch.setattr(env, "run_command", fake_run)
    monkeypatch.setattr(
        env, "candidate_index_urls", lambda commit: ["https://example.invalid/x"]
    )
    res = env.ensure_server_env(SHA, env_root=str(tmp_path))
    assert res.skipped and "install failed" in res.reason


def test_venv_path_layout() -> None:
    assert venv_path(SHA) == f"/cache/envs/{SHA}"


def test_normalize_canonical_bench_json() -> None:
    m = normalize_bench_result(CANONICAL_BENCH_JSON)
    assert m.output_throughput == pytest.approx(106.25)
    assert m.mean_ttft_ms == pytest.approx(71.54)
    assert m.p99_ttft_ms == pytest.approx(79.49)
    assert m.median_tpot_ms == pytest.approx(7.96)
    assert m.failed_requests == 0 and m.valid


def test_normalize_maps_failed_alias_and_marks_invalid() -> None:
    doc = dict(CANONICAL_BENCH_JSON)
    del doc["failed"]
    doc["failed_requests"] = 2
    m = normalize_bench_result(doc)
    assert m.failed_requests == 2 and not m.valid


def test_normalize_missing_keys_fails_loudly() -> None:
    with pytest.raises(ValueError, match="missing.*output_throughput"):
        normalize_bench_result({"completed": 10})
    with pytest.raises(ValueError, match="have keys"):
        normalize_bench_result({})


def test_read_result_file_unwraps_list(tmp_path) -> None:
    p = tmp_path / "r.json"
    p.write_text(json.dumps([CANONICAL_BENCH_JSON]))
    assert normalize_bench_result(read_result_file(str(p))).valid
    p.write_text(json.dumps([]))
    with pytest.raises(ValueError, match="empty list"):
        read_result_file(str(p))
    p.write_text("[1, 2")
    with pytest.raises(json.JSONDecodeError):
        read_result_file(str(p))


def test_bench_command_has_verified_workload_flags(tmp_path) -> None:
    w = Workload(
        dataset_name="random",
        input_len=1024,
        output_len=128,
        num_prompts=100,
        max_concurrency=32,
        seed=0,
    )
    cmd = build_bench_command(
        "/venv/bin/python",
        "meta-model",
        "http://127.0.0.1:8000",
        w,
        str(tmp_path),
        "run-0",
    )
    for flag in (
        "--dataset-name",
        "random",
        "--random-input-len",
        "1024",
        "--random-output-len",
        "128",
        "--num-prompts",
        "100",
        "--max-concurrency",
        "32",
        "--seed",
        "0",
        "--save-result",
        "--result-dir",
        "--result-filename",
        "--model",
        "--base-url",
        "--endpoint",
        "--backend",
    ):
        assert flag in cmd, flag
    assert "--request-rate" not in cmd


def test_bench_command_overrides_for_warmup_and_reseed(tmp_path) -> None:
    w = Workload()
    warm = build_bench_command(
        "/p", "m", "u", w, str(tmp_path), "warm", num_prompts_override=5
    )
    assert warm[warm.index("--num-prompts") + 1] == "5"
    r = build_bench_command("/p", "m", "u", w, str(tmp_path), "run-2", seed_override=2)
    assert r[r.index("--seed") + 1] == "2"


def test_result_path_for_adds_json(tmp_path) -> None:
    assert result_path_for(str(tmp_path), "run-0").endswith("run-0.json")
    assert result_path_for(str(tmp_path), "r.json").endswith("r.json")


def test_bench_filename_matches_result_path(tmp_path) -> None:
    # Regression: vllm bench serve writes --result-filename VERBATIM, so the
    # value in argv must equal the basename run_bench will read.
    import os

    cmd = build_bench_command("/p", "m", "u", Workload(), str(tmp_path), "run-0")
    argv_name = cmd[cmd.index("--result-filename") + 1]
    assert os.path.basename(result_path_for(str(tmp_path), "run-0")) == argv_name


def test_summarize_runs_reports_spread() -> None:
    def m(t: float, f: int = 0) -> RunMetrics:
        return RunMetrics(
            output_throughput=t,
            mean_ttft_ms=1.0,
            p99_ttft_ms=1.0,
            median_tpot_ms=1.0,
            failed_requests=f,
        )

    s = summarize_runs([m(100.0), m(110.0), m(105.0, f=1)])
    assert s["runs"] == 3 and s["valid_runs"] == 2
    assert s["spread_pct"] == pytest.approx(10.0)
    assert summarize_runs([]) == {"runs": 0, "valid_runs": 0}


def test_serve_command_shape() -> None:
    cmd = build_serve_command(
        "/venv/bin/python",
        "meta-model",
        8000,
        {"max_num_seqs": 256, "enable_prefix_caching": True, "async_scheduling": False},
        ["--enforce-eager"],
    )
    assert cmd[:4] == [
        "/venv/bin/python",
        "-u",
        "-m",
        "vllm.entrypoints.openai.api_server",
    ]
    assert "--model" in cmd and "--port" in cmd and "--seed" in cmd
    assert "--max-num-seqs" in cmd and "256" in cmd
    assert "--enable-prefix-caching" in cmd
    assert "--no-async-scheduling" in cmd
    assert "--enforce-eager" in cmd
    assert "true" not in cmd and "false" not in cmd


def test_wait_for_health_success_and_timeout() -> None:
    calls = {"n": 0}

    def get(url: str, timeout: float) -> int:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("refused")
        return 200

    t = [0.0]
    res = wait_for_health(
        "http://127.0.0.1:8000",
        timeout_s=60.0,
        poll_s=1.0,
        get=get,
        clock=lambda: t[0],
        sleep=lambda s: t.__setitem__(0, t[0] + s),
    )
    assert res.ready and calls["n"] == 3

    res = wait_for_health(
        "http://x",
        timeout_s=5.0,
        poll_s=2.0,
        get=lambda u, to: (_ for _ in ()).throw(ConnectionError("down")),
        clock=lambda: t[0],
        sleep=lambda s: t.__setitem__(0, t[0] + s),
    )
    assert not res.ready and "5s" in res.reason


def test_health_timeout_default_is_15_min() -> None:
    assert HEALTH_TIMEOUT_S == 15 * 60


def test_log_tail_reads_end_of_file(tmp_path) -> None:
    p = tmp_path / "serve.log"
    p.write_bytes(b"x" * 10000 + b"TAIL MARKER")
    tail = log_tail(str(p), max_chars=100)
    assert tail.endswith("TAIL MARKER") and len(tail) <= 100
    assert "unreadable" in log_tail(str(tmp_path / "missing.log"))


def test_inject_flags_known_only() -> None:
    from perfpilot.worker.modal_app import INJECT_FLAGS, VALID_INJECTS

    assert set(INJECT_FLAGS) == {"enforce-eager", "gpu-memory-utilization-0.5"}
    assert INJECT_FLAGS["enforce-eager"] == ["--enforce-eager"]
    assert None in VALID_INJECTS


def test_run_trial_rejects_bad_input_without_gpu(tmp_path) -> None:
    from perfpilot.worker.modal_app import run_trial_local

    bad_inject = run_trial_local("c", {}, 3, "m", Workload(), inject="nope")
    assert bad_inject.status == "error" and "inject" in bad_inject.reason
    bad_runs = run_trial_local("c", {}, 0, "m", Workload())
    assert bad_runs.status == "error" and "runs" in bad_runs.reason


def test_bench_pin_is_verified_stable() -> None:
    assert BENCH_PIN_VERSION == "0.32.0"


def test_worker_imports_without_modal() -> None:
    try:
        __import__("modal")
        pytest.skip("modal installed; CPU-guard test not applicable")
    except ImportError:
        pass
    from perfpilot.worker import modal_app

    assert modal_app.app is None
    assert callable(modal_app.run_trial_local)


def test_toolchain_install_command_shape() -> None:
    from perfpilot.worker.env import build_toolchain_install_command

    cmd = build_toolchain_install_command("/venv/bin/python")
    assert cmd[:4] == ["uv", "pip", "install", "--python"]
    assert "nvidia-cuda-nvcc==13.4.92" in cmd
    assert "ninja" in cmd


def test_cuda_home_detects_venv_nvcc(tmp_path) -> None:
    from perfpilot.worker.env import cuda_home, ensure_toolchain

    assert cuda_home(str(tmp_path)) == ""
    nvcc_dir = (
        tmp_path
        / "lib"
        / "python3.12"
        / "site-packages"
        / "nvidia"
        / "cuda_nvcc"
        / "bin"
    )
    nvcc_dir.mkdir(parents=True)
    (nvcc_dir / "nvcc").write_text("fake")
    assert cuda_home(str(tmp_path)).endswith("cuda_nvcc")
    assert ensure_toolchain(str(tmp_path)) == ""


def test_shared_toolchain_created_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from perfpilot.worker import env as env_mod

    root = tmp_path / "toolchain"
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], timeout_s: float):
        calls.append(cmd)
        if cmd[:2] == ["uv", "venv"]:
            (root / "bin").mkdir(parents=True)
            (root / "bin" / "python").write_text("fake")
            return subprocess.CompletedProcess(cmd, 0, "", "")
        assert "ninja" in cmd
        nv = root / "lib" / "python3.12" / "site-packages" / "nvidia" / "cu13" / "bin"
        nv.mkdir(parents=True)
        (nv / "nvcc").write_text("fake")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(env_mod, "run_command", fake_run)
    assert env_mod.ensure_shared_toolchain(str(root)) == ""
    ch, _ = env_mod.toolchain_paths(str(root))
    assert ch.endswith("cu13")
    before = len(calls)
    assert env_mod.ensure_shared_toolchain(str(root)) == ""
    assert len(calls) == before


def _mk_env(root, name: str, age: int) -> None:
    import os

    d = root / name
    d.mkdir()
    os.utime(d, (1_000_000 + age, 1_000_000 + age))


def test_prune_envs_drops_oldest_commit_venvs_only(tmp_path) -> None:
    a, b, c = "a" * 40, "b" * 40, "c" * 40
    _mk_env(tmp_path, a, 1)
    _mk_env(tmp_path, b, 2)
    _mk_env(tmp_path, c, 3)
    _mk_env(tmp_path, "bench-client", 0)
    free = iter([5.0, 5.0, 15.0, 40.0])  # guard read, then one per candidate
    removed = env.prune_envs(
        keep=c, env_root=str(tmp_path), min_free_gb=25.0, free_gb=lambda: next(free)
    )
    assert removed == [a, b]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["bench-client", c]


def test_prune_envs_noop_when_space_is_fine(tmp_path) -> None:
    _mk_env(tmp_path, "a" * 40, 1)
    assert env.prune_envs(keep="", env_root=str(tmp_path), free_gb=lambda: 100.0) == []


def test_disk_full_install_prunes_and_retries(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(env, "is_env_cached", lambda *a: False)
    monkeypatch.setattr(env, "read_installed_sha", lambda venv: SHA[:12])
    monkeypatch.setattr(env, "read_vllm_version", lambda venv: "0.32.0")
    monkeypatch.setattr(env, "ensure_toolchain", lambda venv: "")
    pruned: list[float] = []
    monkeypatch.setattr(
        env,
        "prune_envs",
        lambda keep, env_root=None, min_free_gb=25.0, **k: (
            pruned.append(min_free_gb) or []
        ),
    )
    installs = {"n": 0}

    def fake_run(cmd: list[str], timeout_s: float):
        if cmd[:2] == ["uv", "venv"]:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        installs["n"] += 1
        if installs["n"] == 1:
            return subprocess.CompletedProcess(
                cmd, 2, "", "failed to copy file: No space left on device (os error 28)"
            )
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(env, "run_command", fake_run)
    res = env.ensure_server_env(SHA, env_root=str(tmp_path))
    assert not res.skipped and installs["n"] == 2
    assert len(pruned) == 2 and pruned[1] > pruned[0]


def test_disk_full_reason_is_distinct() -> None:
    reason = env.classify_install_failure(
        2, "copy x: No space left on device (os error 28)"
    )
    assert reason.startswith("cache volume full")
