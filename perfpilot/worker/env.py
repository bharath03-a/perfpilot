"""venv per commit from vLLM nightly wheels, cached on a Modal Volume.

Wheel-URL findings (verified 2026-10-31 with curl, CPU-only):
- https://wheels.vllm.ai/<full-sha> is a PEP 503 index (HTTP 200).
  Use it as ``--extra-index-url`` with ``--index-strategy first-index``.
- File listing: https://wheels.vllm.ai/<sha>/vllm/ contains
  vllm-<ver>+g<sha>-cp38-abi3-manylinux_2_28_{x86_64,aarch64}.whl.
  Exact older-commit x86_64 .whl confirmed HEAD 200.
- Older commits are NOT garbage-collected (2026-08-31 SHAs -> 200).
- Only commits newer than the last nightly build 404 (build lag);
  a 16-commit bisect range is feasible if its SHAs predate the latest
  nightly index (check https://wheels.vllm.ai/nightly/ timestamp).
- CUDA flavour: the cu130 tree mirrors the same layout
  (https://wheels.vllm.ai/nightly/cu130/vllm/); the per-SHA index is
  tried first, nightly/cu130 second.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass

WHEEL_ROOT = "https://wheels.vllm.ai"
NIGHTLY_INDEX = f"{WHEEL_ROOT}/nightly/"
CUDA_INDEX = f"{WHEEL_ROOT}/nightly/cu130"

SHA_RE = re.compile(r"^[0-9a-f]{40}$")
WHEEL_FILE_RE = re.compile(
    r"vllm-(?P<ver>[^+\"'<>]+)\+g(?P<sha>[0-9a-f]{7,40})-cp38-abi3-"
    r"manylinux_2_28_(?P<arch>x86_64|aarch64)\.whl"
)

ENV_ROOT = "/cache/envs"
BENCH_ENV = "/cache/envs/bench-client"
TOOLCHAIN_ROOT = "/cache/toolchain"


def toolchain_paths(
    root: str = TOOLCHAIN_ROOT,
) -> tuple[str, str]:
    """(cuda_home, venv_bin) for the shared toolchain; '' when absent."""
    ch = cuda_home(root)
    if not ch:
        return "", ""
    vbin = os.path.dirname(venv_python(root))
    return ch, vbin if os.path.isdir(vbin) else (ch, "")


def ensure_shared_toolchain(root: str = TOOLCHAIN_ROOT) -> str:
    """Create the shared nvcc+ninja toolchain once (one copy for all
    commits). Returns '' on success, else a skip reason."""
    if cuda_home(root):
        return ""

    shutil.rmtree(root, ignore_errors=True)
    proc = run_command(["uv", "venv", root], timeout_s=300.0)
    if proc.returncode != 0:
        return f"toolchain venv failed: {proc.stderr.strip()[-200:]}"
    proc = run_command(
        build_toolchain_install_command(venv_python(root)), timeout_s=1200.0
    )
    if proc.returncode != 0 or not cuda_home(root):
        return f"toolchain install failed: {proc.stderr.strip()[-300:]}"
    return ""


def is_full_sha(commit: str) -> bool:
    return bool(SHA_RE.match(commit))


def wheel_index_url(commit: str) -> str:
    if not is_full_sha(commit):
        raise ValueError(f"refusing non-SHA commit {commit!r}: wheels are per-SHA only")
    return f"{WHEEL_ROOT}/{commit}"


def candidate_index_urls(commit: str) -> list[str]:
    return [wheel_index_url(commit), CUDA_INDEX, NIGHTLY_INDEX]


def venv_path(commit: str, env_root: str = ENV_ROOT) -> str:
    return os.path.join(env_root, commit)


def venv_python(venv: str) -> str:
    return os.path.join(venv, "bin", "python")


def is_env_cached(commit: str, env_root: str = ENV_ROOT) -> bool:
    py = venv_python(venv_path(commit, env_root))
    return os.path.isfile(py) and os.access(py, os.X_OK)


def find_wheel_filename(
    index_html: str, commit: str, arch: str = "x86_64"
) -> str | None:
    for m in WHEEL_FILE_RE.finditer(index_html):
        if commit.startswith(m.group("sha")) and m.group("arch") == arch:
            return m.group(0)
    return None


def build_install_command(venv_py: str, index_url: str) -> list[str]:
    return [
        "uv",
        "pip",
        "install",
        "--python",
        venv_py,
        "vllm",
        "--extra-index-url",
        index_url,
        "--index-strategy",
        "first-index",
        "--prerelease",
        "allow",
    ]


def build_venv_command(venv_dir: str) -> list[str]:
    return ["uv", "venv", venv_dir]


@dataclass
class EnvResult:
    venv: str
    cached: bool
    vllm_version: str = ""
    skipped: bool = False
    reason: str = ""


#: Per-SHA venvs are several GB each; the eval-12 GPU run lost 6 of 12
#: trials to ENOSPC on the shared cache volume before pruning existed.
MIN_FREE_GB = 25.0
RETRY_FREE_GB = 60.0
DISK_FULL = "no space left on device"
#: Modal volumes cap inodes (500k); a torch+vLLM venv is ~85k files, so the
#: volume fills by file count long before bytes. eval-t0 hit this at 409 GB free.
MIN_FREE_INODE_FRAC = 0.25
MAX_ENVS = 3


def _free_gb(path: str) -> float:
    return shutil.disk_usage(path).free / 1e9


def _free_inode_frac(path: str) -> float:
    st = os.statvfs(path)
    return st.f_ffree / st.f_files if st.f_files else 1.0


def prune_envs(
    keep: str,
    env_root: str = ENV_ROOT,
    min_free_gb: float = MIN_FREE_GB,
    free_gb: Callable[[], float] | None = None,
    free_inodes: Callable[[], float] | None = None,
    max_envs: int = MAX_ENVS,
) -> list[str]:
    """Delete least-recently-used per-commit venvs until enough space is free.

    Only 40-hex SHA dirs are candidates: the current commit, the bench
    client and the toolchain are never touched.
    """
    import os

    gb = free_gb or (lambda: _free_gb(env_root))
    inodes = free_inodes or (lambda: _free_inode_frac(env_root))
    if not os.path.isdir(env_root):
        return []
    candidates = sorted(
        (
            entry
            for entry in os.scandir(env_root)
            if entry.is_dir() and is_full_sha(entry.name) and entry.name != keep
        ),
        key=lambda entry: entry.stat().st_mtime,
    )

    def roomy(remaining: int) -> bool:
        return (
            remaining < max_envs
            and gb() >= min_free_gb
            and inodes() >= MIN_FREE_INODE_FRAC
        )

    removed: list[str] = []
    for i, entry in enumerate(candidates):
        if roomy(len(candidates) - i):
            break
        shutil.rmtree(entry.path, ignore_errors=True)
        removed.append(entry.name)
        print(f"[perfpilot] pruned cached venv {entry.name[:12]} for space", flush=True)
    return removed


def classify_install_failure(returncode: int, stderr: str) -> str:
    blob = stderr.lower()
    if DISK_FULL in blob:
        tail = stderr.strip()[-200:]
        return f"cache volume full after pruning (rc={returncode}): {tail}"
    if "404" in blob or "not found" in blob or "no matching distribution" in blob:
        return "no wheel for commit (index 404 / no matching distribution)"
    if "network" in blob or "connection" in blob or "timed out" in blob:
        return f"wheel download failed (network): {stderr.strip()[-300:]}"
    return f"vllm wheel install failed (rc={returncode}): {stderr.strip()[-300:]}"


def run_command(cmd: list[str], timeout_s: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)


def ensure_server_env(
    commit: str,
    env_root: str = ENV_ROOT,
    install_timeout_s: float = 1200.0,
) -> EnvResult:
    """Create (or reuse) the per-commit server venv. Never builds from source.

    Returns EnvResult with skipped=True + reason when there is no wheel or
    the install fails; callers map that to TrialResult status="skip".
    """
    venv = venv_path(commit, env_root)
    if is_env_cached(commit, env_root):
        cached_sha = read_installed_sha(venv)
        if cached_sha and commit.startswith(cached_sha):
            version = read_vllm_version(venv)
            tc_err = ensure_toolchain(venv)
            if tc_err:
                return EnvResult(venv=venv, cached=True, skipped=True, reason=tc_err)
            return EnvResult(venv=venv, cached=True, vllm_version=version)

        print(
            f"[perfpilot] cached venv {commit[:12]} holds "
            f"{cached_sha or 'stable (no +gSHA)'}; rebuilding",
            flush=True,
        )
        shutil.rmtree(venv, ignore_errors=True)

    prune_envs(commit, env_root)
    index_url = wheel_index_url(commit)
    for attempt in (1, 2):
        shutil.rmtree(venv, ignore_errors=True)
        proc = run_command(build_venv_command(venv), timeout_s=300.0)
        if proc.returncode != 0:
            tail = proc.stderr.strip()[-300:]
            return EnvResult(
                venv=venv,
                cached=False,
                skipped=True,
                reason=f"uv venv failed (rc={proc.returncode}): {tail}",
            )
        # ONLY the per-commit index: installing any other commit's wheel (e.g.
        # latest nightly) would silently measure the wrong binary. Any failure
        # here is a skip, never a fallback.
        try:
            proc = run_command(
                build_install_command(venv_python(venv), index_url),
                timeout_s=install_timeout_s,
            )
        except subprocess.TimeoutExpired:
            shutil.rmtree(venv, ignore_errors=True)
            return EnvResult(
                venv=venv,
                cached=False,
                skipped=True,
                reason=f"wheel install timed out via {index_url}",
            )
        if proc.returncode == 0 or DISK_FULL not in proc.stderr.lower() or attempt == 2:
            break
        shutil.rmtree(venv, ignore_errors=True)
        prune_envs(commit, env_root, min_free_gb=RETRY_FREE_GB)
    if proc.returncode == 0:
        installed = read_installed_sha(venv)
        if not installed or not commit.startswith(installed):
            # Empty means a stable release (no +gSHA): uv fell through to
            # PyPI because the per-SHA index had no match. That binary is
            # NOT the commit — drop the venv and skip, never measure it.

            shutil.rmtree(venv, ignore_errors=True)
            return EnvResult(
                venv=venv,
                cached=False,
                skipped=True,
                reason=(
                    f"wheel SHA mismatch: want {commit[:12]}, "
                    f"installed {installed or 'stable (no +gSHA)'}"
                ),
            )
        tc_err = ensure_toolchain(venv)
        if tc_err:
            return EnvResult(venv=venv, cached=False, skipped=True, reason=tc_err)
        return EnvResult(venv=venv, cached=False, vllm_version=read_vllm_version(venv))

    shutil.rmtree(venv, ignore_errors=True)
    return EnvResult(
        venv=venv,
        cached=False,
        skipped=True,
        reason=classify_install_failure(proc.returncode, proc.stderr),
    )


def ensure_bench_env(
    pin_version: str,
    bench_root: str = BENCH_ENV,
    install_timeout_s: float = 1200.0,
) -> EnvResult:
    """Create (or reuse) the pinned benchmark-client venv.

    The ruler never changes: one fixed vLLM version for every commit.
    Failure here is an error, never a skip (our pin, not the commit's fault).
    """
    py = venv_python(bench_root)
    if os.path.isfile(py) and os.access(py, os.X_OK):
        return EnvResult(
            venv=bench_root, cached=True, vllm_version=read_vllm_version(bench_root)
        )
    proc = run_command(["uv", "venv", bench_root], timeout_s=300.0)
    if proc.returncode != 0:
        raise RuntimeError(
            f"bench-client uv venv failed (rc={proc.returncode}): "
            f"{proc.stderr.strip()[-300:]}"
        )
    proc = run_command(
        ["uv", "pip", "install", "--python", py, f"vllm=={pin_version}"],
        timeout_s=install_timeout_s,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"pinned bench client vllm=={pin_version} install failed "
            f"(rc={proc.returncode}): {proc.stderr.strip()[-500:]}"
        )
    return EnvResult(
        venv=bench_root, cached=False, vllm_version=read_vllm_version(bench_root)
    )


def cuda_home(venv: str) -> str:
    """Toolkit root containing bin/nvcc anywhere under the venv.

    Recursive: torch pre-installs stub/real nvidia packages in varying
    layouts, so never assume nvidia/<pkg>/bin/nvcc.
    """
    for root, dirs, files in os.walk(venv):
        if os.path.basename(root) == "bin" and "nvcc" in files:
            return os.path.dirname(root)
        # Don't descend into unrelated bulk.
        dirs[:] = [d for d in dirs if d != "__pycache__"]
    return ""


def build_toolchain_install_command(venv_py: str) -> list[str]:
    # Pinned real toolkit: torch pre-installs a stub that satisfies a bare
    # requirement (pip then no-ops), and cu-suffixed names are stubs too.
    # Layout verified: nvidia/cu13/bin/nvcc in the 13.4.92 linux wheel.
    # ninja: flashinfer JIT shells out to it after nvcc resolves.
    return [
        "uv",
        "pip",
        "install",
        "--python",
        venv_py,
        "nvidia-cuda-nvcc==13.4.92",
        "ninja",
    ]


def ensure_toolchain(venv: str) -> str:
    """Ensure nvcc+ninja are available for a venv. Uses the shared toolchain
    (one copy for all commits — per-venv copies exhaust volume inodes).
    Returns '' on success, else a skip reason."""
    if cuda_home(venv) or cuda_home(TOOLCHAIN_ROOT):
        return ""
    return ensure_shared_toolchain()


def read_installed_sha(venv: str) -> str:
    """Short commit SHA baked into the installed wheel version (+gSHA)."""
    try:
        proc = run_command(
            [
                venv_python(venv),
                "-c",
                "from importlib.metadata import version; print(version('vllm'))",
            ],
            timeout_s=120.0,
        )
    except subprocess.TimeoutExpired:
        return ""
    if proc.returncode != 0:
        return ""
    m = re.search(r"\+g([0-9a-f]{7,40})", proc.stdout.strip())
    return m.group(1) if m else ""


def read_vllm_version(venv: str) -> str:
    try:
        proc = run_command(
            [
                venv_python(venv),
                "-c",
                "from importlib.metadata import version; print(version('vllm'))",
            ],
            timeout_s=120.0,
        )
    except subprocess.TimeoutExpired:
        return ""
    return proc.stdout.strip() if proc.returncode == 0 else ""
