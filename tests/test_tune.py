"""TuneSmith gates over fake configs: accept, reject, quality, stop."""

from __future__ import annotations

import random
from typing import Any

from perfpilot.modes.tune import (
    KnobProposal,
    TuneParams,
    TuneTrial,
    config_key,
    default_propose,
    quality_passes,
    run_tuning,
    serve_command,
    validate_proposal,
)

BASELINE: dict[str, Any] = {
    "max_num_seqs": 128,
    "max_num_batched_tokens": 4096,
    "async_scheduling": False,
    "enable_prefix_caching": False,
    "kv_cache_dtype": "auto",
}

SPACE: dict[str, list[Any]] = {
    "max_num_seqs": [64, 128, 256, 512],
    "max_num_batched_tokens": [2048, 4096, 8192],
    "async_scheduling": [False, True],
    "enable_prefix_caching": [False, True],
    "kv_cache_dtype": ["auto", "fp8"],
}


def throughput(config: dict[str, Any]) -> float:
    mult = 1.0
    seqs = config.get("max_num_seqs", 128)
    mult *= {64: 0.94, 128: 1.0, 256: 1.08, 512: 1.12}.get(seqs, 1.0)
    if config.get("enable_prefix_caching") is True:
        mult *= 1.005
    if config.get("async_scheduling") is True:
        mult *= 1.001
    if config.get("kv_cache_dtype") == "fp8":
        mult *= 1.10
    tokens = config.get("max_num_batched_tokens", 4096)
    mult *= {2048: 0.99, 4096: 1.0, 8192: 1.0}.get(tokens, 1.0)
    return 1000.0 * mult


def p99_for(config: dict[str, Any]) -> float:
    if config.get("max_num_seqs") == 512:
        return 130.0
    return 100.0


def accuracy_for(config: dict[str, Any]) -> float:
    if config.get("kv_cache_dtype") == "fp8":
        return 74.0
    return 80.0


def make_measure(noise: float = 0.004, seed: int = 0):
    def measure(config: dict[str, Any], runs: int) -> TuneTrial:
        rng = random.Random(f"{seed}:{config_key(config)}:{runs}")
        mean = throughput(config)
        return TuneTrial(
            samples=[rng.gauss(mean, mean * noise) for _ in range(runs)],
            p99_ttft_ms=p99_for(config),
        )

    return measure


def test_accepts_real_uplift_and_confirms():
    out = run_tuning(BASELINE, SPACE, make_measure(), accuracy_for, model="glimmer")
    assert out.best_config["max_num_seqs"] == 256
    assert out.confirmed, out.confirm_reason
    assert out.uplift_lo > 0.02
    accepted = [c for c in out.candidates if c.decision == "accepted"]
    assert accepted
    assert out.serve_command.startswith("vllm serve glimmer")
    assert "--max-num-seqs 256" in out.serve_command


def test_weak_uplift_rejected_below_2pct():
    out = run_tuning(BASELINE, SPACE, make_measure(), accuracy_for)
    weak = [
        c
        for c in out.candidates
        if c.knob == "enable_prefix_caching" and c.value is True
    ]
    assert weak and all(c.decision == "rejected" for c in weak)
    assert out.best_config.get("enable_prefix_caching") is not True


def test_p99_regression_rejected_despite_uplift():
    out = run_tuning(BASELINE, SPACE, make_measure(), accuracy_for)
    seq512 = [c for c in out.candidates if (c.knob, c.value) == ("max_num_seqs", 512)]
    assert seq512 and all(c.decision == "rejected" for c in seq512)
    assert any("p99" in c.reason for c in seq512)
    assert out.best_config.get("max_num_seqs") != 512


def test_quality_rejection_for_fp8():
    out = run_tuning(BASELINE, SPACE, make_measure(), accuracy_for)
    fp8 = [c for c in out.candidates if (c.knob, c.value) == ("kv_cache_dtype", "fp8")]
    assert fp8 and all(c.decision == "rejected" for c in fp8)
    assert any("accuracy" in c.reason for c in fp8)
    assert out.best_config.get("kv_cache_dtype") != "fp8"
    assert any(not q.passed for q in out.quality_checks)


def test_final_config_always_quality_checked():
    seen: list[dict[str, Any]] = []

    def quality(config: dict[str, Any]) -> float:
        seen.append(dict(config))
        return accuracy_for(config)

    out = run_tuning(BASELINE, SPACE, make_measure(), quality)
    assert seen
    assert config_key(out.best_config) in {config_key(c) for c in seen}


def test_failing_final_quality_reverts_to_baseline():
    def quality(config: dict[str, Any]) -> float:
        if config.get("max_num_seqs") == 256:
            return 70.0
        return 80.0

    params = TuneParams(quality_knobs=set())
    out = run_tuning(BASELINE, SPACE, make_measure(), quality, params=params)
    assert out.best_config == BASELINE
    assert not out.confirmed


def test_rejects_out_of_config_and_already_tried():
    tried = {config_key(BASELINE)}
    bad_knob = validate_proposal(
        KnobProposal(knob="nope", value=1), BASELINE, SPACE, tried
    )
    assert not bad_knob.valid
    bad_value = validate_proposal(
        KnobProposal(knob="max_num_seqs", value=999), BASELINE, SPACE, tried
    )
    assert not bad_value.valid
    noop = validate_proposal(
        KnobProposal(knob="max_num_seqs", value=128), BASELINE, SPACE, tried
    )
    assert not noop.valid
    ok = validate_proposal(
        KnobProposal(knob="max_num_seqs", value=256), BASELINE, SPACE, tried
    )
    assert ok.valid and ok.candidate["max_num_seqs"] == 256
    tried.add(config_key(ok.candidate))
    dup = validate_proposal(
        KnobProposal(knob="max_num_seqs", value=256), BASELINE, SPACE, tried
    )
    assert not dup.valid


def test_stops_at_max_trials():
    big_space = {f"knob{i}": [0, 1, 2, 3, 4] for i in range(6)}
    base = {f"knob{i}": 0 for i in range(6)}

    def measure(config: dict[str, Any], runs: int) -> TuneTrial:
        rng = random.Random(3)
        return TuneTrial(
            samples=[rng.gauss(1000.0, 4.0) for _ in range(runs)],
            p99_ttft_ms=100.0,
        )

    params = TuneParams(max_trials=8)
    out = run_tuning(base, big_space, measure, lambda c: 80.0, params=params)
    assert out.trials_used <= 8
    assert out.stopped_reason == "max_trials reached"


def test_agent_propose_none_stops_early():
    out = run_tuning(
        BASELINE, SPACE, make_measure(), accuracy_for, propose=lambda b, h: None
    )
    assert out.stopped_reason == "no further proposals"
    assert out.best_config == BASELINE


def test_quality_passes_helper():
    assert quality_passes(80.0, 80.0).passed
    assert not quality_passes(77.9, 80.0).passed
    assert quality_passes(78.0, 80.0).passed


def test_invalid_proposals_do_not_consume_trials():
    plan = [
        KnobProposal(knob="nope", value=1),
        KnobProposal(knob="max_num_seqs", value=256),
        KnobProposal(knob="max_num_seqs", value=512),
    ]

    def propose(best: dict[str, Any], history: list) -> KnobProposal | None:
        return plan.pop(0) if plan else None

    params = TuneParams(max_trials=2)
    out = run_tuning(
        BASELINE, SPACE, make_measure(), accuracy_for, params=params, propose=propose
    )
    assert out.rejections == 1
    assert out.trials_used == 2
    assert any("rejected" in entry for entry in out.propose_log)


def test_propose_exception_counts_fallback():
    def propose(best: dict[str, Any], history: list) -> KnobProposal | None:
        raise RuntimeError("agent down")

    params = TuneParams(max_trials=2)
    out = run_tuning(
        BASELINE, SPACE, make_measure(), accuracy_for, params=params, propose=propose
    )
    assert out.fallbacks >= 1
    assert any("propose raised" in entry for entry in out.propose_log)
    assert out.trials_used == 2


def test_knob_order_threads_into_default_propose():
    tried = {config_key(BASELINE)}
    first = default_propose(BASELINE, SPACE, tried, ["kv_cache_dtype"])
    assert first is not None and first.knob == "kv_cache_dtype"
    params = TuneParams(max_trials=1)
    out = run_tuning(
        BASELINE,
        SPACE,
        make_measure(),
        accuracy_for,
        params=params,
        knob_order=["kv_cache_dtype"],
    )
    assert out.candidates and out.candidates[0].knob == "kv_cache_dtype"


def test_final_quality_revert_annotates_accepted():
    def quality(config: dict[str, Any]) -> float:
        if config.get("max_num_seqs") == 256:
            return 70.0
        return 80.0

    params = TuneParams(quality_knobs=set())
    out = run_tuning(BASELINE, SPACE, make_measure(), quality, params=params)
    assert out.best_config == BASELINE
    assert not any(c.decision == "accepted" for c in out.candidates)
    assert any("final quality failed" in c.reason for c in out.candidates)
    assert any("reverted" in entry for entry in out.propose_log)


def test_serve_command_shape():
    cmd = serve_command(
        "glimmer",
        {"max_num_seqs": 256, "enable_prefix_caching": True, "async_scheduling": False},
    )
    assert cmd.startswith("vllm serve glimmer --port 8000 --seed 0")
    assert "--max-num-seqs 256" in cmd
    assert "--enable-prefix-caching" in cmd
    assert "async-scheduling" not in cmd


def test_measure_failure_rejects_candidate_and_continues():
    base = make_measure()
    seen: list[dict] = []

    def measure(config: dict[str, Any], runs: int) -> TuneTrial:
        if config.get("max_num_seqs") == 256:
            seen.append(dict(config))
            raise RuntimeError("transient serve crash")
        return base(config, runs)

    out = run_tuning(BASELINE, SPACE, measure, accuracy_for, model="glimmer")
    assert seen, "failing config was never tried"
    reasons = [c.reason for c in out.candidates]
    assert any("measure failed" in r for r in reasons)
    assert out.best_config, "run must still produce a best config"
    assert out.trials_used > 0


def test_quality_failure_rejects_candidate_fail_closed():
    def quality(config: dict[str, Any]) -> float:
        if config.get("kv_cache_dtype") == "fp8":
            raise RuntimeError("answer_batch down")
        return accuracy_for(config)

    out = run_tuning(BASELINE, SPACE, make_measure(), quality, model="glimmer")
    assert out.best_config.get("kv_cache_dtype") != "fp8"
    assert any("quality failed" in c.reason for c in out.candidates)
    assert not any(
        c.decision == "accepted" and c.config.get("kv_cache_dtype") == "fp8"
        for c in out.candidates
    )


def test_confirm_failure_yields_unconfirmed_not_crash():
    base = make_measure()

    def measure(config: dict[str, Any], runs: int) -> TuneTrial:
        if runs >= 8:
            raise RuntimeError("confirm serve crash")
        return base(config, runs)

    out = run_tuning(BASELINE, SPACE, measure, accuracy_for, model="glimmer")
    assert out.best_config
    assert out.confirmed is False
    assert "unevaluated" in out.confirm_reason


def test_nan_samples_become_rejections_not_crash():
    base = make_measure()

    def measure(config: dict[str, Any], runs: int) -> TuneTrial:
        trial = base(config, runs)
        if config.get("max_num_seqs") == 256:
            trial = TuneTrial(
                samples=[float("nan")] * runs, p99_ttft_ms=trial.p99_ttft_ms
            )
        return trial

    out = run_tuning(BASELINE, SPACE, measure, accuracy_for, model="glimmer")
    nan_recs = [c for c in out.candidates if c.config.get("max_num_seqs") == 256]
    assert nan_recs
    assert all("verdict failed" in c.reason for c in nan_recs)
    assert not any(
        c.decision == "accepted" and c.config.get("max_num_seqs") == 256
        for c in out.candidates
    )
