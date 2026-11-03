"""The brain must see measured evidence, not just SHAs, when it decides."""

from __future__ import annotations

from pathlib import Path

from perfpilot.agent.client import SparkClient
from perfpilot.modes.bisect import CommitLabel, new_bisect_state
from perfpilot.modes.tune import CandidateRecord
from perfpilot.orchestrator import (
    AgentOrchestrator,
    make_extra_runs_fn,
    make_pick_fn,
    make_propose_fn,
)
from perfpilot.stats import Verdict, classify_commit


def _capturing_orch(tmp_path: Path, reply: str, seen: list[str]):
    def transport(system: str, user: str) -> str:
        seen.append(user)
        return reply

    return AgentOrchestrator(
        run_id="ctx",
        events_path=tmp_path / "events.jsonl",
        client=SparkClient(transport=transport),
    )


def test_pick_history_carries_labels_and_effects(tmp_path: Path) -> None:
    seen: list[str] = []
    o = _capturing_orch(tmp_path, '{"sha": "c", "reason": "mid"}', seen)
    state = new_bisect_state(["a", "b", "c", "d", "e"])
    state = state.model_copy(
        update={
            "bad_idx": 4,
            "full_effect": -0.5,
            "samples": {
                "a": [100.0, 101.0, 99.0],
                "e": [50.0, 51.0, 49.0],
            },
            "labels": {"a": CommitLabel.GOOD, "e": CommitLabel.BAD},
            "tested_order": ["a", "e"],
        }
    )
    make_pick_fn(o)(state)
    prompt = seen[-1]
    assert "| good" in prompt and "| bad" in prompt
    assert "-0.69" in prompt  # ln(50/100) for the bad endpoint
    assert "| 3 |" in prompt


def test_classify_exposes_decision_midpoint() -> None:
    v = classify_commit([80.0, 81.0, 79.0], [100.0, 101.0, 99.0], full_effect=-0.4)
    assert v.midpoint == -0.2


def test_extra_runs_prompt_uses_decision_midpoint(tmp_path: Path) -> None:
    seen: list[str] = []
    o = _capturing_orch(tmp_path, '{"runs": 1, "reason": "close"}', seen)
    verdict = Verdict(
        label="inconclusive",
        effect=-0.1,
        lo=-0.3,
        hi=0.1,
        confidence=0.9,
        reason="straddles",
        midpoint=-0.25,
    )
    make_extra_runs_fn(o)("c", 3, verdict)
    assert "m = -0.2500" in seen[-1]


def test_propose_sees_measured_candidate_results(tmp_path: Path) -> None:
    seen: list[str] = []
    o = _capturing_orch(
        tmp_path, '{"knob": "max_num_seqs", "value": 128, "reason": "r"}', seen
    )
    knobs = {"max_num_seqs": [64, 128], "async_scheduling": [False, True]}
    propose = make_propose_fn(o, knobs)
    tried = CandidateRecord(
        knob="async_scheduling",
        value=True,
        config={"async_scheduling": True},
        effect=0.011,
        lo=-0.004,
        hi=0.026,
        decision="rejected",
        reason="no uplift",
    )
    propose({}, [tried])
    prompt = seen[-1]
    assert "async_scheduling=True" in prompt
    assert "0.011" in prompt and "rejected" in prompt


def test_compare_range_returns_messages(monkeypatch) -> None:
    from perfpilot import git_info

    sha_a, sha_b = "a" * 40, "b" * 40
    payload = {
        "status": "ahead",
        "commits": [
            {"sha": sha_a, "commit": {"message": "[Sched] tweak\n\nbody"}},
            {"sha": sha_b, "commit": {"message": "docs: typo"}},
        ],
    }
    monkeypatch.setattr(git_info, "_api", lambda url: payload)
    infos = git_info.commit_infos_between("o/r", "g", "b")
    assert [(i.sha, i.message) for i in infos] == [
        (sha_a, "[Sched] tweak"),
        (sha_b, "docs: typo"),
    ]
    assert git_info.commits_between("o/r", "g", "b") == [sha_a, sha_b]


def test_tune_prompts_carry_hardware(tmp_path: Path) -> None:
    seen: list[str] = []
    o = _capturing_orch(
        tmp_path, '{"knob": "max_num_seqs", "value": 128, "reason": "r"}', seen
    )
    knobs = {"max_num_seqs": [64, 128]}
    make_propose_fn(o, knobs, hardware="A100")({}, [])
    assert all("A100" in p for p in seen)
    assert any("no FP8" in p for p in seen)
