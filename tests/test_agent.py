"""Agent brain tests: fake transport only, no network, no key."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from perfpilot.agent.client import (
    API_KEY_ENV,
    SparkClient,
    read_api_key,
    strip_to_json,
)
from perfpilot.agent.prompts import SYSTEM_PROMPTS
from perfpilot.agent.tools import DECISION_MODELS, schema_for
from perfpilot.orchestrator import AgentOrchestrator

KNOBS = {
    "max_num_seqs": [64, 128],
    "async_scheduling": [False, True],
}


def orch(tmp_path: Path, transport) -> AgentOrchestrator:
    return AgentOrchestrator(
        run_id="test-run",
        events_path=tmp_path / "events.jsonl",
        client=SparkClient(transport=transport),
    )


def read_events(tmp_path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]


def test_prompts_cover_all_decisions() -> None:
    assert set(SYSTEM_PROMPTS) == set(DECISION_MODELS)
    for prompt in SYSTEM_PROMPTS.values():
        assert "JSON only" in prompt


def test_schemas_cover_all_decisions() -> None:
    for decision in DECISION_MODELS:
        assert schema_for(decision)["type"] == "object"


def test_strip_reasoning_text() -> None:
    raw = (
        "Thinking: the middle looks good...\n```json\n"
        '{"sha": "abc", "reason": "middle"}\n```'
    )
    assert json.loads(strip_to_json(raw)) == {"sha": "abc", "reason": "middle"}


def test_strip_bare_json_with_prefix() -> None:
    raw = 'Here is my answer: {"runs": 2, "reason": "close"} trailing words'
    assert json.loads(strip_to_json(raw)) == {"runs": 2, "reason": "close"}


def test_retry_then_fallback(tmp_path: Path) -> None:
    calls = []

    def bad(sys: str, user: str) -> str:
        calls.append(user)
        return "not json at all"

    o = orch(tmp_path, bad)
    d = o.pick_commit(["a", "b", "c"])
    assert d.value == "b" and d.accepted is False
    assert len(calls) == 2
    assert "failed validation" in calls[1]


def test_retry_succeeds(tmp_path: Path) -> None:
    seen: list[str] = []

    def flaky(sys: str, user: str) -> str:
        seen.append(user)
        return "oops" if len(seen) == 1 else '{"sha": "c", "reason": "tip"}'

    o = orch(tmp_path, flaky)
    d = o.pick_commit(["a", "b", "c"])
    assert (d.value, d.accepted) == ("c", True)


def test_timeout_falls_back(tmp_path: Path) -> None:
    def slow(sys: str, user: str) -> str:
        raise TimeoutError("timed out")

    o = orch(tmp_path, slow)
    d = o.extra_runs(current_runs=3, max_runs=8)
    assert (d.value, d.accepted) == (2, False)


def test_api_error_falls_back(tmp_path: Path) -> None:
    def boom(sys: str, user: str) -> str:
        raise ConnectionError("down")

    o = orch(tmp_path, boom)
    assert o.rank_knobs(KNOBS).value == list(KNOBS)
    assert o.draft_fix("abc").value is None


def test_pick_commit_rejects_out_of_range(tmp_path: Path) -> None:
    o = orch(tmp_path, lambda s, u: '{"sha": "zzz", "reason": "wild"}')
    d = o.pick_commit(["a", "b", "c", "d", "e"])
    assert (d.value, d.accepted) == ("c", False)
    ev = read_events(tmp_path)[-1]
    assert ev["accepted"] is False and "outside range" in ev["why"]


def test_propose_config_rejections(tmp_path: Path) -> None:
    o = orch(tmp_path, lambda s, u: '{"knob": "nope", "value": 1, "reason": "x"}')
    d = o.propose_config(KNOBS, tried=[])
    assert (d.accepted, d.value.knob, d.value.value) == (False, "max_num_seqs", 64)

    o2 = orch(
        tmp_path, lambda s, u: '{"knob": "max_num_seqs", "value": 999, "reason": "x"}'
    )
    d2 = o2.propose_config(KNOBS, tried=[])
    assert d2.accepted is False

    o3 = orch(
        tmp_path, lambda s, u: '{"knob": "max_num_seqs", "value": 64, "reason": "x"}'
    )
    d3 = o3.propose_config(KNOBS, tried=[("max_num_seqs", 64)])
    assert (d3.accepted, d3.value.value) == (False, 128)


def test_explain_rejects_uncited_paths(tmp_path: Path) -> None:
    o = orch(
        tmp_path,
        lambda s, u: (
            '{"summary": "s", "mechanism": "m",'
            '"files": ["evil.py"], "confidence": "high"}'
        ),
    )
    d = o.explain_culprit("abc", "msg", diff_paths=["good.py"], diff_excerpt="x")
    assert d.accepted is False and d.value.summary == "No explanation"


def test_rank_knobs_rejects_unknown(tmp_path: Path) -> None:
    o = orch(tmp_path, lambda s, u: '{"order": ["mystery"], "reason": "x"}')
    d = o.rank_knobs(KNOBS)
    assert (d.accepted, d.value) == (False, list(KNOBS))


def test_events_schema(tmp_path: Path) -> None:
    o = orch(tmp_path, lambda s, u: '{"sha": "b", "reason": "ok"}')
    o.pick_commit(["a", "b", "c"])
    o.extra_runs(3, 8, lo=-0.1, hi=0.1, midpoint=-0.05)
    for ev in read_events(tmp_path):
        assert {"ts", "run_id", "type"} <= set(ev)
        assert ev["run_id"] == "test-run"
        if ev["type"] == "agent_call":
            assert {"decision", "proposal", "accepted", "why"} <= set(ev)


def test_full_sequence_with_key_unset(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    o = AgentOrchestrator(run_id="r", events_path=tmp_path / "events.jsonl")
    cands = [f"c{i}" for i in range(5)]
    assert o.pick_commit(cands).value == "c2"
    assert o.extra_runs(3, 8).value == 2
    exp = o.explain_culprit("c2", "m", diff_paths=["a.py"])
    assert exp.value.summary == "No explanation"
    assert o.rank_knobs(KNOBS).value == list(KNOBS)
    prop = o.propose_config(KNOBS, tried=[])
    assert (prop.value.knob, prop.value.value) == ("max_num_seqs", 64)
    assert o.draft_fix("c2").value is None
    assert len(read_events(tmp_path)) == 6


def test_propose_config_full_config_granularity(tmp_path: Path) -> None:
    from perfpilot.modes.tune import config_key

    best = {"max_num_seqs": 64, "async_scheduling": False}
    already = {"max_num_seqs": 128, "async_scheduling": False}
    o = orch(
        tmp_path,
        lambda s, u: '{"knob": "max_num_seqs", "value": 128, "reason": "x"}',
    )
    d = o.propose_config(KNOBS, tried={config_key(already)}, best=best)
    assert d.accepted is False and "already tried" in d.why
    assert (d.value.knob, d.value.value) == ("async_scheduling", True)
    o2 = orch(
        tmp_path,
        lambda s, u: '{"knob": "max_num_seqs", "value": 128, "reason": "x"}',
    )
    d2 = o2.propose_config(KNOBS, tried=[already], best=best)
    assert d2.accepted is False and "already tried" in d2.why


def test_propose_config_rejects_noop(tmp_path: Path) -> None:
    best = {"max_num_seqs": 128, "async_scheduling": False}
    o = orch(
        tmp_path,
        lambda s, u: '{"knob": "max_num_seqs", "value": 128, "reason": "x"}',
    )
    d = o.propose_config(KNOBS, tried=[], best=best)
    assert d.accepted is False and "no-op" in d.why
    assert d.value is not None


def test_propose_config_exhaustion_logs_no_raise(tmp_path: Path) -> None:
    seen: list[str] = []
    knobs = {"max_num_seqs": [64, 128]}
    best = {"max_num_seqs": 64}
    tried = [{"max_num_seqs": 64}, {"max_num_seqs": 128}]

    def transport(sys: str, user: str) -> str:
        seen.append(user)
        return '{"knob": "max_num_seqs", "value": 64, "reason": "x"}'

    o = orch(tmp_path, transport)
    d = o.propose_config(knobs, tried=tried, best=best)
    assert d.value is None and d.accepted is False
    assert d.why == "no untried configs"
    assert seen == []
    ev = read_events(tmp_path)[-1]
    assert ev["accepted"] is False and ev["why"] == "no untried configs"


def test_extra_runs_clamped_logs_rejected(tmp_path: Path) -> None:
    o = orch(tmp_path, lambda s, u: '{"runs": 3, "reason": "wide"}')
    d = o.extra_runs(current_runs=7, max_runs=8)
    assert (d.value, d.accepted) == (1, False)
    ev = read_events(tmp_path)[-1]
    assert ev["accepted"] is False and ev["why"] == d.why
    assert "clamped" in d.why


def test_diff_truncated_before_prompt(tmp_path: Path) -> None:
    seen: list[str] = []

    def cap(sys: str, user: str) -> str:
        seen.append(user)
        return '{"summary": "s", "mechanism": "m", "files": [], "confidence": "high"}'

    big = "x" * 5000
    o = orch(tmp_path, cap)
    d = o.explain_culprit("abc", "msg", diff_paths=[], diff_excerpt=big)
    assert d.accepted is True
    assert big not in seen[0] and seen[0].endswith("...truncated")


def test_draft_fix_logs_bounded_excerpt(tmp_path: Path) -> None:
    patch = "line\n" * 500

    def transport(sys: str, user: str) -> str:
        assert len(user) < 3000
        assert user.endswith("...truncated")
        return json.dumps({"patch": patch})

    o = orch(tmp_path, transport)
    d = o.draft_fix("abc", diff="d" * 5000)
    assert d.accepted is True
    ev = read_events(tmp_path)[-1]
    assert patch not in json.dumps(ev)
    assert ev["proposal"]["patch_excerpt"].endswith("...truncated")


def test_explain_rejects_bad_confidence(tmp_path: Path) -> None:
    o = orch(
        tmp_path,
        lambda s, u: (
            '{"summary": "s", "mechanism": "m","files": [], "confidence": "extreme"}'
        ),
    )
    d = o.explain_culprit("abc", "m", diff_paths=[])
    assert d.accepted is False and d.value.summary == "No explanation"


def test_openai_style_timeout_falls_back(tmp_path: Path) -> None:
    class APITimeoutError(Exception):
        pass

    def slow(sys: str, user: str) -> str:
        raise APITimeoutError("timed out")

    o = orch(tmp_path, slow)
    d = o.extra_runs(current_runs=3, max_runs=8)
    assert (d.value, d.accepted) == (2, False)


def test_propose_config_type_strict_bool(tmp_path: Path) -> None:
    knobs = {"async_scheduling": [False, True]}
    o = orch(
        tmp_path,
        lambda s, u: '{"knob": "async_scheduling", "value": 1, "reason": "x"}',
    )
    d = o.propose_config(knobs, tried=[], best={})
    assert d.accepted is False and "not in config" in d.why
    assert d.value.value is False


def test_pick_commit_empty_range_logs_then_raises(tmp_path: Path) -> None:
    o = orch(tmp_path, lambda s, u: '{"sha": "a", "reason": "x"}')
    with pytest.raises(ValueError):
        o.pick_commit([])
    ev = read_events(tmp_path)[-1]
    assert ev["decision"] == "pick_commit" and ev["accepted"] is False


def test_decision_why_matches_event_why(tmp_path: Path) -> None:
    o = orch(tmp_path, lambda s, u: '{"sha": "zzz", "reason": "wild"}')
    d = o.pick_commit(["a", "b", "c"])
    assert d.why == read_events(tmp_path)[-1]["why"]

    def boom(sys: str, user: str) -> str:
        raise ConnectionError("down")

    o2 = orch(tmp_path, boom)
    d2 = o2.extra_runs(3, 8)
    assert d2.why == read_events(tmp_path)[-1]["why"]

    o3 = orch(tmp_path, lambda s, u: '{"knob": "nope", "value": 1, "reason": "x"}')
    d3 = o3.propose_config(KNOBS, tried=[])
    assert d3.why == read_events(tmp_path)[-1]["why"]

    o4 = orch(tmp_path, lambda s, u: '{"patch": "  ", "reason": "x"}')
    d4 = o4.draft_fix("abc")
    assert d4.why == read_events(tmp_path)[-1]["why"]


def test_make_pick_fn_none_on_rejection(tmp_path: Path) -> None:
    from perfpilot.modes.bisect import new_bisect_state, pick_next
    from perfpilot.orchestrator import make_pick_fn

    o = orch(tmp_path, lambda s, u: '{"sha": "zzz", "reason": "wild"}')
    pick = make_pick_fn(o)
    state = new_bisect_state(["a", "b", "c"])
    assert pick(state) is None
    res = pick_next(state, pick)
    assert (res.sha, res.fallback_used) == ("b", True)

    o2 = orch(tmp_path, lambda s, u: '{"sha": "b", "reason": "mid"}')
    assert make_pick_fn(o2)(new_bisect_state(["a", "b", "c"])) == "b"


def test_make_extra_runs_fn(tmp_path: Path) -> None:
    from perfpilot.orchestrator import make_extra_runs_fn
    from perfpilot.stats import Verdict

    o = orch(tmp_path, lambda s, u: '{"runs": 2, "reason": "ok"}')
    verdict = Verdict(
        label="inconclusive",
        effect=0.0,
        lo=-0.1,
        hi=0.1,
        confidence=0.9,
        reason="x",
    )
    assert make_extra_runs_fn(o)("c", 3, verdict) == 2


def test_make_propose_fn_wires_rank_order(tmp_path: Path) -> None:
    from perfpilot.orchestrator import make_propose_fn

    def transport(sys: str, user: str) -> str:
        if "ranking" in sys:
            return '{"order": ["async_scheduling", "max_num_seqs"], "reason": "q"}'
        raise ConnectionError("agent down, use fallback")

    o = orch(tmp_path, transport)
    propose = make_propose_fn(o, KNOBS)
    got = propose({}, [])
    assert (got.knob, got.value) == ("async_scheduling", False)


def test_make_propose_fn_none_on_exhaustion(tmp_path: Path) -> None:
    from perfpilot.orchestrator import make_propose_fn

    o = orch(tmp_path, lambda s, u: "garbage")
    propose = make_propose_fn(o, {"k": [1]}, baseline={"k": 1})
    assert propose({"k": 1}, []) is None


def test_no_key_leak_in_errors(tmp_path: Path, monkeypatch) -> None:
    import os as _os

    from perfpilot.agent.client import complete_validated
    from perfpilot.agent.tools import PickCommit

    monkeypatch.setenv(API_KEY_ENV, "sekrit-key-123")

    def boom(sys: str, user: str) -> str:
        raise ConnectionError("down")

    with pytest.raises(Exception) as ei:
        complete_validated(
            system="s", user="u", response_model=PickCommit, transport=boom
        )
    assert "sekrit-key-123" not in str(ei.value)
    o = AgentOrchestrator(
        run_id="r",
        events_path=tmp_path / "events.jsonl",
        client=SparkClient(transport=boom),
    )
    o.pick_commit(["a", "b"])
    assert "sekrit-key-123" not in (tmp_path / "events.jsonl").read_text()
    assert _os.environ[API_KEY_ENV] == "sekrit-key-123"


def test_dotenv_fills_missing_key(monkeypatch, tmp_path):
    """./.env supplies the key when env has neither name (fake key)."""
    import perfpilot.agent.client as client_mod

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    monkeypatch.delenv("BASETEN_API_KEY", raising=False)
    monkeypatch.setattr(client_mod, "_dotenv_loaded_for", None)
    (tmp_path / ".env").write_text("BASETEN_API_KEY=fake-dotenv-key\n")
    assert read_api_key() == "fake-dotenv-key"


def test_real_env_beats_dotenv(monkeypatch, tmp_path):
    """.env never overrides an already-set env var."""
    import perfpilot.agent.client as client_mod

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(API_KEY_ENV, "real-env-key")
    monkeypatch.delenv("BASETEN_API_KEY", raising=False)
    monkeypatch.setattr(client_mod, "_dotenv_loaded_for", None)
    (tmp_path / ".env").write_text(
        f"{API_KEY_ENV}=dotenv-key\nBASETEN_API_KEY=dotenv-fallback\n"
    )
    assert read_api_key() == "real-env-key"
