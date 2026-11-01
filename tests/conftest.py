"""Suite-wide hermeticity: no test may see the developer's real keys or .env."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_secrets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Clear key env vars and run each test in an empty cwd.

    perfpilot.agent.client._load_dotenv reads ./.env, so without this a
    developer's real ~/.env-adjacent repo .env would leak live credentials
    (and live brain behavior) into fallback/key-unset tests.
    """
    monkeypatch.delenv("PERFPILOT_LLM_API_KEY", raising=False)
    monkeypatch.delenv("BASETEN_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
