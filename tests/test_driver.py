"""M2 driver test: planted culprit found, index stays hidden."""

from __future__ import annotations

import json
import random

from perfpilot.modes import TrialOutcome
from perfpilot.modes.driver import drive_eval_trial


def _fake_factory(base: float, drop: float, seed: int, n: int = 5):
    def make_measure(inject_for, record):
        def measure(sha: str, runs: int) -> TrialOutcome:
            rng = random.Random(hash((seed, sha, runs)) & 0xFFFFFFFF)
            mu = base * (1.0 - drop) if inject_for(sha) else base
            result = {
                "status": "ok",
                "gpu_uuid": "GPU-fake",
                "gpu_seconds": 1.0,
                "runs": [
                    {"output_throughput": rng.gauss(mu, mu * 0.01)} for _ in range(n)
                ],
            }
            record.append({"sha": sha, "runs": runs, "result": result})
            return TrialOutcome(
                samples=[r["output_throughput"] for r in result["runs"]]
            )

        return measure

    return make_measure


def test_driver_finds_planted_culprit_and_hides_index(tmp_path) -> None:
    commits = [f"c{i:02d}{'0' * 36}" for i in range(16)]
    summary = drive_eval_trial(
        commits,
        culprit_index=9,
        inject_kind="enforce-eager",
        make_measure=_fake_factory(3800.0, 0.20, seed=7),
        run_id="test-eval-1",
        runs_root=str(tmp_path),
    )
    assert summary["correct"] and summary["culprit"] == commits[9]
    assert summary["verified"]
    assert summary["n_trials"] <= 12  # ~log2(16) + endpoints + verify
    events = (tmp_path / "test-eval-1" / "events.jsonl").read_text()
    assert "culprit_index" not in events and "enforce-eager" not in events
    assert (tmp_path / "test-eval-1" / "report.html").exists()
    types = {json.loads(line)["type"] for line in events.strip().splitlines()}
    assert {"run_start", "trial_result", "culprit", "run_end"} <= types


def test_driver_retries_error_once_then_proceeds(tmp_path) -> None:
    from perfpilot.modes.driver import drive_eval_trial

    calls: list[str] = []

    def make_measure(inject_for, record):
        def measure(sha: str, runs: int):
            from perfpilot.modes import TrialOutcome

            calls.append(sha)
            if calls.count(sha) == 1:
                record.append(
                    {
                        "sha": sha,
                        "runs": runs,
                        "result": {
                            "status": "error",
                            "reason": "trial error: flake",
                            "gpu_uuid": "g",
                            "gpu_seconds": 1.0,
                            "runs": [],
                        },
                    }
                )
                return TrialOutcome(skipped=True, reason="trial error: flake")
            return _ok(sha, runs, record, inject_for)

        return measure

    import random

    def _ok(sha, runs, record, inject_for):
        from perfpilot.modes import TrialOutcome

        rng = random.Random(hash(sha) & 0xFFFFFFFF)
        mu = 3000.0 if inject_for(sha) else 3800.0
        result = {
            "status": "ok",
            "gpu_uuid": "g",
            "gpu_seconds": 1.0,
            "runs": [{"output_throughput": rng.gauss(mu, mu * 0.01)} for _ in range(5)],
        }
        record.append({"sha": sha, "runs": runs, "result": result})
        return TrialOutcome(samples=[r["output_throughput"] for r in result["runs"]])

    commits = [f"d{i:02d}{'0' * 36}" for i in range(16)]
    summary = drive_eval_trial(
        commits,
        culprit_index=9,
        inject_kind="enforce-eager",
        make_measure=make_measure,
        run_id="retry-test",
        runs_root=str(tmp_path),
    )
    assert summary["correct"]
    assert summary["n_trials"] > 4  # every error retried exactly once


def test_bisect_live_completes_without_brain(tmp_path) -> None:
    """M3 acceptance: key unset -> deterministic fallbacks still find culprit."""
    from perfpilot.modes import TrialOutcome
    from perfpilot.modes.driver import drive_bisect_live

    commits = [f"b{i:02d}{'0' * 36}" for i in range(8)]

    def measure(sha: str, runs: int) -> TrialOutcome:
        rng = random.Random(hash(sha) & 0xFFFFFFFF)
        mu = 3040.0 if commits.index(sha) >= 5 else 3800.0
        return TrialOutcome(samples=[rng.gauss(mu, mu * 0.01) for _ in range(runs)])

    out = drive_bisect_live(
        commits, measure, run_id="no-brain-bisect", runs_root=str(tmp_path)
    )
    assert out["status"] == "culprit"
    assert out["culprit"] == commits[5]
    assert (tmp_path / "no-brain-bisect" / "report.html").exists()


def test_tune_live_completes_without_brain(tmp_path) -> None:
    """M4 acceptance: propose=None still tunes via default grid proposals."""
    from perfpilot.modes.driver import drive_tune_live
    from perfpilot.modes.tune import TuneTrial

    space: dict[str, list] = {"max_num_seqs": [64, 128, 256]}

    def measure(config: dict, runs: int) -> TuneTrial:
        rng = random.Random(f"tune:{sorted(config.items())}:{runs}")
        mult = {64: 0.94, 128: 1.0, 256: 1.08}[config.get("max_num_seqs", 128)]
        mean = 1000.0 * mult
        return TuneTrial(
            samples=[rng.gauss(mean, mean * 0.004) for _ in range(runs)],
            p99_ttft_ms=100.0,
        )

    out = drive_tune_live(
        "base000" + "0" * 33,
        space,
        measure,
        lambda config: 80.0,
        run_id="no-brain-tune",
        runs_root=str(tmp_path),
    )
    assert out["best"]["max_num_seqs"] == 256
    assert "vllm serve" in out["serve_command"]
    assert (tmp_path / "no-brain-tune" / "report.html").exists()
