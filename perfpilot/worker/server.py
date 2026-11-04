"""Start / health-check / stop ``vllm serve`` in its own process group.

Contract (SPEC.md): start from the commit's venv, poll /health with a
15-minute timeout (-> error + log tail), always kill the process group.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

HEALTH_TIMEOUT_S = 15 * 60
HEALTH_PATH = "/health"
DEFAULT_PORT = 8000


def build_serve_command(
    venv_py: str,
    model: str,
    port: int,
    flags: dict[str, object],
    extra_args: Sequence[str] = (),
) -> list[str]:
    cmd = [
        venv_py,
        "-u",  # unbuffered: tracebacks must reach the log file on crash
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--port",
        str(port),
        "--seed",
        "0",
    ]
    for key in sorted(flags):
        name = f"--{key}".replace("_", "-")
        value = flags[key]
        if isinstance(value, bool):
            cmd.append(name if value else f"--no-{name[2:]}")
        else:
            cmd += [name, str(value)]
    cmd.extend(extra_args)
    return cmd


def start_server(
    cmd: list[str],
    log_path: str,
    env: dict[str, str] | None = None,
) -> subprocess.Popen[str]:
    """Launch the server as a new session leader so the whole group dies."""
    log_fh = open(log_path, "ab")
    return subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )


def kill_process_group(proc: subprocess.Popen[str], grace_s: float = 30.0) -> None:
    """SIGTERM the group, escalate to SIGKILL. Never raises on missing procs."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=grace_s)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        pass


#: Repeated per-process warnings that bury the real failure in the tail.
LOG_NOISE = ("torch._opaque_base is deprecated",)


def log_tail(log_path: str, max_chars: int = 4000) -> str:
    try:
        with open(log_path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - max_chars * 8))
            text = fh.read().decode(errors="replace")
    except OSError as exc:
        return f"<log unreadable: {exc}>"
    kept = [ln for ln in text.splitlines() if not any(n in ln for n in LOG_NOISE)]
    return "\n".join(kept)[-max_chars:]


@dataclass
class HealthResult:
    ready: bool
    elapsed_s: float
    reason: str = ""


def wait_for_health(
    base_url: str,
    timeout_s: float = HEALTH_TIMEOUT_S,
    poll_s: float = 2.0,
    get: Callable[[str, float], int] | None = None,
    clock: Callable[[], float] | None = None,
    sleep: Callable[[float], None] | None = None,
    proc: subprocess.Popen[str] | None = None,
    log_path: str | None = None,
) -> HealthResult:
    """Poll <base_url>/health until HTTP 200. Injectable for unit tests."""
    import urllib.request

    def _get(url: str, timeout: float) -> int:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status

    get = get or _get
    now = clock or time.monotonic
    doze = sleep or time.sleep
    start = now()
    attempts = 0
    while True:
        attempts += 1
        try:
            if get(base_url.rstrip("/") + HEALTH_PATH, 10.0) == 200:
                return HealthResult(ready=True, elapsed_s=now() - start)
        except Exception:
            pass
        # Fail fast: a dead server will never become healthy. This turns a
        # startup crash from a 900s burn into a seconds-long error.
        if proc is not None and proc.poll() is not None:
            return HealthResult(
                ready=False,
                elapsed_s=now() - start,
                reason=(
                    f"server process exited (rc={proc.returncode}) "
                    f"after {attempts} polls"
                ),
            )
        # Alive-but-broken: the process can wedge after printing a
        # traceback (exit stays None). Sniff the log ~1/min and abort early.
        if log_path is not None and attempts % 30 == 0:
            tail = log_tail(log_path)
            if "Traceback (most recent call last)" in tail:
                return HealthResult(
                    ready=False,
                    elapsed_s=now() - start,
                    reason=(
                        f"server traceback after {attempts} polls; "
                        f"log tail: {tail[-2000:]}"
                    ),
                )
        if now() - start >= timeout_s:
            return HealthResult(
                ready=False,
                elapsed_s=now() - start,
                reason=f"/health not ready after {timeout_s:.0f}s ({attempts} polls)",
            )
        doze(poll_s)
