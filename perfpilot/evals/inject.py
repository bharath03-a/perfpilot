"""Injected-regression overlays for eval trials. Pure: no GPU, no network.

In eval mode the worker secretly appends a slowdown flag to the server
command for every commit at or after a hidden culprit index. The agent and
orchestrator never see the flag; they only see benchmark numbers.

Two overlays (PLAN.md; pre-measured, kept only if >=10% drop):
- ``enforce_eager``: ``--enforce-eager`` (strong, version-stable)
- ``gpu_mem_05``: ``--gpu-memory-utilization 0.5`` (milder)

Eval trials skip the explanation step: the real diff does not contain
the planted change, so there is nothing to explain.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

InjectionKind = Literal["enforce_eager", "gpu_mem_05"]

#: Minimum measured drop required to keep an overlay (10%).
MIN_KEEP_DROP = 0.10

#: Server flags appended secretly per overlay kind.
OVERLAY_FLAGS: dict[str, list[str]] = {
    "enforce_eager": ["--enforce-eager"],
    "gpu_mem_05": ["--gpu-memory-utilization", "0.5"],
}

#: Eval trials never run the explain step.
SKIP_EXPLAIN_IN_EVAL = True

#: Eval kind -> worker wire key (modal_app.INJECT_FLAGS). The driver
#: translates at the boundary in modal_measure_factory; the worker never
#: sees eval-side names. Tested as a bijection against INJECT_FLAGS so
#: either side adding a kind breaks the suite, not a GPU trial.
WORKER_INJECTS: dict[InjectionKind, str] = {
    "enforce_eager": "enforce-eager",
    "gpu_mem_05": "gpu-memory-utilization-0.5",
}


def to_worker_inject(kind: InjectionKind) -> str:
    """Translate an eval overlay kind to the worker wire key. Loud on unknown."""
    try:
        return WORKER_INJECTS[kind]
    except KeyError:
        raise ValueError(f"unknown injection kind {kind!r}") from None


class Injection(BaseModel):
    kind: InjectionKind = Field(description="Overlay slowdown type")
    culprit_index: int = Field(ge=0, description="Hidden index; inject at/after this")
    range_size: int = Field(default=16, ge=1, description="Fixed eval range size")


def is_injected(commit_index: int, culprit_index: int) -> bool:
    """True when a commit at this position gets the secret overlay."""
    return commit_index >= culprit_index


def overlay_flags(kind: InjectionKind) -> list[str]:
    """Secret server flags for an overlay kind (worker-side only)."""
    return list(OVERLAY_FLAGS[kind])


def apply_injection(
    base_flags: list[str], commit_index: int, injection: Injection
) -> list[str]:
    """Server command flags: base + secret overlay at/after culprit index."""
    if is_injected(commit_index, injection.culprit_index):
        return list(base_flags) + overlay_flags(injection.kind)
    return list(base_flags)


def visible_flags(server_flags: list[str], kind: InjectionKind) -> list[str]:
    """Flags safe to show the agent/orchestrator: overlay stripped.

    Agent/orchestrator see only numbers; even flag lists must not leak
    the planted slowdown. Removes ALL occurrences: the overlay may appear
    more than once if flags are concatenated from several layers.

    M1/M2 wiring rule: every worker/server path that logs flags must route
    them through this helper first; raw server flags (and culprit_index)
    must never reach events.jsonl.
    """
    secret = overlay_flags(kind)
    out = list(server_flags)
    if not secret:
        return out
    i = 0
    while i <= len(out) - len(secret):
        if out[i : i + len(secret)] == secret:
            del out[i : i + len(secret)]
        else:
            i += 1
    return out


def should_explain(is_eval: bool) -> bool:
    """Eval skips explanation; the real diff lacks the planted change."""
    if is_eval and SKIP_EXPLAIN_IN_EVAL:
        return False
    return True
