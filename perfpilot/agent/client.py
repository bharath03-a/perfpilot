"""Baseten Muse Spark Open client (OpenAI-compatible chat completions).

The API key comes from env ``PERFPILOT_LLM_API_KEY`` (or ``BASETEN_API_KEY``
fallback), with a gitignored local ``.env`` filling gaps. It is never
printed, logged, or included in error messages.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, ValidationError

BASE_URL = "https://inference.baseten.co/v1"
MODEL_ID = "meta-models/muse-spark-open-1026"
API_KEY_ENV = "PERFPILOT_LLM_API_KEY"
_DOTENV_KEYS = (API_KEY_ENV, "BASETEN_API_KEY")
TIMEOUT_S = 60

_dotenv_loaded_for: Path | None = None


def _load_dotenv() -> None:
    """Load known key names from ./``.env`` without overriding real env."""
    global _dotenv_loaded_for
    dotenv = Path.cwd() / ".env"
    if dotenv == _dotenv_loaded_for or not dotenv.is_file():
        return
    for line in dotenv.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip()
        if name in _DOTENV_KEYS and name not in os.environ:
            value = value.strip().strip("\"'")
            if value:
                os.environ[name] = value
    _dotenv_loaded_for = dotenv


Transport = Callable[[str, str], str]


class AgentAPIError(RuntimeError):
    """Transport/timeout/key failure: caller must use the fallback."""


class AgentValidationError(RuntimeError):
    """Response failed schema validation even after one retry."""


def read_api_key() -> str:
    """PERFPILOT_LLM_API_KEY first, BASETEN_API_KEY fallback; .env fills gaps."""
    _load_dotenv()
    return os.environ.get(API_KEY_ENV, "") or os.environ.get("BASETEN_API_KEY", "")


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def strip_to_json(text: str) -> str:
    """Strip reasoning text / fences, returning a JSON document string."""
    m = _FENCE_RE.search(text)
    candidate = m.group(1) if m else text
    decoder = json.JSONDecoder()
    for i, ch in enumerate(candidate):
        if ch in "{[":
            try:
                obj, _ = decoder.raw_decode(candidate[i:])
                return json.dumps(obj)
            except json.JSONDecodeError:
                continue
    return candidate.strip()


def _real_call(
    system: str, user: str, base_url: str, model: str, api_key: str, timeout_s: float
) -> str:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""


def complete_validated[T: BaseModel](
    *,
    system: str,
    user: str,
    response_model: type[T],
    transport: Transport | None = None,
    api_key: str | None = None,
    base_url: str = BASE_URL,
    model: str = MODEL_ID,
    timeout_s: float = TIMEOUT_S,
) -> T:
    """One validated JSON proposal; retry once with the error appended."""
    key = api_key if api_key is not None else read_api_key()
    if transport is None and not key:
        raise AgentAPIError(f"no API key in {API_KEY_ENV}; using fallback")
    prompt = user
    last_error = ""
    for _ in (1, 2):
        try:
            if transport is not None:
                raw = transport(system, prompt)
            else:
                raw = _real_call(system, prompt, base_url, model, key, timeout_s)
        except AgentAPIError:
            raise
        except Exception as e:
            if isinstance(e, TimeoutError) or "Timeout" in type(e).__name__:
                raise AgentAPIError(f"agent call timed out: {type(e).__name__}") from e
            raise AgentAPIError(f"agent API error: {type(e).__name__}") from e
        try:
            return response_model.model_validate(json.loads(strip_to_json(raw)))
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = str(e)
            prompt = (
                user + "\n\nPrevious response failed validation: "
                f"{e}\nReturn JSON only matching the required schema."
            )
    raise AgentValidationError(f"invalid agent response after retry: {last_error}")


class SparkClient:
    """Thin wrapper; pass ``transport`` in tests to avoid network/key."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = BASE_URL,
        model: str = MODEL_ID,
        timeout_s: float = TIMEOUT_S,
        transport: Transport | None = None,
    ) -> None:
        self._api_key = api_key
        self.base_url = base_url
        self.model = model
        self.timeout_s = timeout_s
        self.transport = transport

    def complete[T: BaseModel](
        self, system: str, user: str, response_model: type[T]
    ) -> T:
        return complete_validated(
            system=system,
            user=user,
            response_model=response_model,
            transport=self.transport,
            api_key=self._api_key,
            base_url=self.base_url,
            model=self.model,
            timeout_s=self.timeout_s,
        )
