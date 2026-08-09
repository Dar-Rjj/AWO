"""Audited OpenRouter client compatible with the RealEvo reference."""

# Pydantic evaluates model annotations on Python 3.9, where ``X | None`` is invalid.
# ruff: noqa: UP045

from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from openai import APIConnectionError, APITimeoutError, OpenAI
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator


class LLMConfigurationError(ValueError):
    """Raised before a request when the OpenRouter configuration is invalid."""


class LLMRequestError(RuntimeError):
    """Raised after a request exhausts its bounded retry budget."""


class OpenRouterSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: Literal["openrouter"] = "openrouter"
    model: str
    base_url: str = "https://openrouter.ai/api/v1"
    temperature: float = 1.0
    n: int = 1
    timeout_seconds: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=8, ge=0)
    max_concurrency: int = Field(default=8, ge=1)
    max_tokens: int = Field(default=4096, ge=1)
    api_key: SecretStr

    @field_validator("n")
    @classmethod
    def require_single_choice(cls, value: int) -> int:
        if value != 1:
            raise ValueError("OpenRouter requests must use n=1")
        return value

    @classmethod
    def from_mapping(
        cls, config: Mapping[str, Any], api_key: str | None = None
    ) -> OpenRouterSettings:
        key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise LLMConfigurationError(
                "OPENROUTER_API_KEY is required; provide it via the environment"
            )
        values = dict(config)
        values.pop("schema_version", None)
        values["api_key"] = key
        try:
            return cls.model_validate(values)
        except ValueError as exc:
            raise LLMConfigurationError(str(exc)) from exc


class TokenUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: Optional[float] = None


class ChatResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    request_id: str
    response_id: Optional[str]
    requested_model: str
    actual_model: Optional[str]
    provider: Optional[str]
    content: str
    finish_reason: Optional[str]
    usage: TokenUsage
    attempts: int
    latency_seconds: float
    request_sha256: str


class JsonlRequestRecorder:
    """Thread-safe append-only request log with no credentials."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record(self, payload: Mapping[str, Any]) -> None:
        line = json.dumps(dict(payload), ensure_ascii=False, sort_keys=True) + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()


def _request_hash(
    model: str,
    messages: Sequence[Mapping[str, str]],
    temperature: float,
    max_tokens: int,
) -> str:
    payload = json.dumps(
        {
            "model": model,
            "messages": [dict(message) for message in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "n": 1,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _status_code(exc: Exception) -> int | None:
    value = getattr(exc, "status_code", None)
    return value if isinstance(value, int) else None


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return True
    return _status_code(exc) in {408, 409, 425, 429, 500, 502, 503, 504}


class OpenRouterClient:
    """One-choice chat client used by every optimizer and executor role."""

    def __init__(
        self,
        settings: OpenRouterSettings,
        recorder: JsonlRequestRecorder | None = None,
        sdk_client: Any | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        self.settings = settings
        self.recorder = recorder
        self._sleep = sleeper
        self._jitter = jitter
        self._client = sdk_client or OpenAI(
            api_key=settings.api_key.get_secret_value(),
            base_url=settings.base_url,
            timeout=settings.timeout_seconds,
            max_retries=0,
        )

    def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ChatResult:
        if not messages:
            raise LLMConfigurationError("messages must not be empty")
        normalized = [dict(message) for message in messages]
        for message in normalized:
            if set(message) != {"role", "content"} or not all(
                isinstance(value, str) for value in message.values()
            ):
                raise LLMConfigurationError(
                    "each message must contain string role and content fields"
                )

        request_id = str(uuid.uuid4())
        selected_temperature = (
            self.settings.temperature if temperature is None else temperature
        )
        selected_max_tokens = self.settings.max_tokens if max_tokens is None else max_tokens
        request_sha256 = _request_hash(
            self.settings.model, normalized, selected_temperature, selected_max_tokens
        )
        started_at = datetime.now(timezone.utc)
        started = time.perf_counter()
        errors: list[dict[str, Any]] = []
        total_attempts = self.settings.max_retries + 1

        for attempt in range(1, total_attempts + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.settings.model,
                    messages=normalized,
                    temperature=selected_temperature,
                    max_tokens=selected_max_tokens,
                    n=1,
                    stream=False,
                )
                if not response.choices:
                    raise LLMRequestError("OpenRouter returned no choices")
                choice = response.choices[0]
                content = choice.message.content
                if not isinstance(content, str) or not content:
                    raise LLMRequestError("OpenRouter returned empty content")

                usage_obj = getattr(response, "usage", None)
                usage_extra = getattr(usage_obj, "model_extra", None) or {}
                response_extra = getattr(response, "model_extra", None) or {}
                usage = TokenUsage(
                    prompt_tokens=getattr(usage_obj, "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(usage_obj, "completion_tokens", 0) or 0,
                    total_tokens=getattr(usage_obj, "total_tokens", 0) or 0,
                    cost=usage_extra.get("cost") or response_extra.get("cost"),
                )
                result = ChatResult(
                    request_id=request_id,
                    response_id=getattr(response, "id", None),
                    requested_model=self.settings.model,
                    actual_model=getattr(response, "model", None),
                    provider=response_extra.get("provider"),
                    content=content,
                    finish_reason=getattr(choice, "finish_reason", None),
                    usage=usage,
                    attempts=attempt,
                    latency_seconds=time.perf_counter() - started,
                    request_sha256=request_sha256,
                )
                self._record(
                    {
                        "status": "ok",
                        "started_at": started_at.isoformat(),
                        "messages": normalized,
                        "temperature": selected_temperature,
                        "max_tokens": selected_max_tokens,
                        "metadata": dict(metadata or {}),
                        "errors": errors,
                        "result": result.model_dump(mode="json"),
                    }
                )
                return result
            except Exception as exc:
                error = {
                    "attempt": attempt,
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "status_code": _status_code(exc),
                    "retryable": _is_retryable(exc),
                }
                errors.append(error)
                if not error["retryable"] or attempt == total_attempts:
                    self._record(
                        {
                            "status": "error",
                            "started_at": started_at.isoformat(),
                            "request_id": request_id,
                            "request_sha256": request_sha256,
                            "requested_model": self.settings.model,
                            "messages": normalized,
                            "temperature": selected_temperature,
                            "max_tokens": selected_max_tokens,
                            "metadata": dict(metadata or {}),
                            "attempts": attempt,
                            "latency_seconds": time.perf_counter() - started,
                            "errors": errors,
                        }
                    )
                    raise LLMRequestError(
                        f"OpenRouter request failed after {attempt} attempt(s): {exc}"
                    ) from exc

                delay = min(60.0, 2.0 ** (attempt - 1)) + self._jitter()
                self._sleep(delay)

        raise AssertionError("retry loop exited unexpectedly")

    def chat_many(
        self,
        conversations: Sequence[Sequence[Mapping[str, str]]],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> list[ChatResult]:
        with ThreadPoolExecutor(max_workers=self.settings.max_concurrency) as executor:
            return list(
                executor.map(
                    lambda messages: self.chat(messages, metadata=metadata),
                    conversations,
                )
            )

    def _record(self, payload: Mapping[str, Any]) -> None:
        if self.recorder is not None:
            self.recorder.record(payload)


def client_from_config(
    config: Mapping[str, Any],
    *,
    recorder: JsonlRequestRecorder | None = None,
    api_key: str | None = None,
    sdk_client: Any | None = None,
) -> OpenRouterClient:
    settings = settings_from_config(config, api_key=api_key)
    return OpenRouterClient(settings, recorder=recorder, sdk_client=sdk_client)


def settings_from_config(
    config: Mapping[str, Any], *, api_key: str | None = None
) -> OpenRouterSettings:
    """Accept either a direct model YAML or a resolved experiment configuration."""

    embedded = config.get("model")
    model_config = embedded if isinstance(embedded, Mapping) else config
    return OpenRouterSettings.from_mapping(model_config, api_key=api_key)
