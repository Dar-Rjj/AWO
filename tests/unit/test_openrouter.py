import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from awo.llm import (
    JsonlRequestRecorder,
    LLMConfigurationError,
    LLMRequestError,
    OpenRouterClient,
    OpenRouterSettings,
    settings_from_config,
)


class StatusError(Exception):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def fake_response(content: str = "AWO_PREFLIGHT_OK"):
    return SimpleNamespace(
        id="response-1",
        model="deepseek/deepseek-chat",
        model_extra={"provider": "fake-provider"},
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content), finish_reason="stop"
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=3,
            completion_tokens=2,
            total_tokens=5,
            model_extra={"cost": 0.001},
        ),
    )


def settings(**overrides):
    values = {
        "model": "deepseek/deepseek-chat",
        "api_key": "test-secret",
        "max_retries": 2,
    }
    values.update(overrides)
    return OpenRouterSettings.model_validate(values)


def client_with(outcomes, tmp_path, **setting_overrides):
    completions = FakeCompletions(outcomes)
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    record_path = tmp_path / "requests.jsonl"
    client = OpenRouterClient(
        settings(**setting_overrides),
        recorder=JsonlRequestRecorder(record_path),
        sdk_client=sdk,
        sleeper=lambda _delay: None,
        jitter=lambda: 0.0,
    )
    return client, completions, record_path


def test_success_records_auditable_secret_free_request(tmp_path) -> None:
    client, completions, record_path = client_with([fake_response()], tmp_path)

    result = client.chat(
        [{"role": "user", "content": "ping"}], metadata={"role": "test"}
    )

    assert result.content == "AWO_PREFLIGHT_OK"
    assert result.actual_model == "deepseek/deepseek-chat"
    assert result.provider == "fake-provider"
    assert result.usage.total_tokens == 5
    assert completions.calls[0]["n"] == 1
    persisted = record_path.read_text(encoding="utf-8")
    assert "test-secret" not in persisted
    payload = json.loads(persisted)
    assert payload["status"] == "ok"
    assert payload["result"]["request_sha256"] == result.request_sha256


def test_retry_is_bounded_and_records_prior_errors(tmp_path) -> None:
    retryable = StatusError("rate limited", 429)
    client, completions, record_path = client_with(
        [retryable, retryable, fake_response()], tmp_path
    )
    delays = []
    client._sleep = delays.append

    result = client.chat([{"role": "user", "content": "ping"}])

    assert result.attempts == 3
    assert len(completions.calls) == 3
    assert delays == [1.0, 2.0]
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    assert [item["status_code"] for item in payload["errors"]] == [429, 429]


def test_non_retryable_error_stops_immediately(tmp_path) -> None:
    client, completions, record_path = client_with(
        [StatusError("bad request", 400)], tmp_path
    )

    with pytest.raises(LLMRequestError, match="after 1 attempt"):
        client.chat([{"role": "user", "content": "ping"}])

    assert len(completions.calls) == 1
    assert json.loads(record_path.read_text(encoding="utf-8"))["status"] == "error"


def test_chat_many_preserves_input_order(tmp_path) -> None:
    client, _completions, _record_path = client_with(
        [fake_response("first"), fake_response("second")],
        tmp_path,
        max_concurrency=1,
    )

    results = client.chat_many(
        [
            [{"role": "user", "content": "one"}],
            [{"role": "user", "content": "two"}],
        ]
    )

    assert [result.content for result in results] == ["first", "second"]


def test_settings_reject_multiple_choices() -> None:
    with pytest.raises(ValidationError, match="n=1"):
        settings(n=2)


def test_settings_require_environment_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(LLMConfigurationError, match="OPENROUTER_API_KEY"):
        OpenRouterSettings.from_mapping({"model": "deepseek/deepseek-chat"})


def test_settings_accept_direct_and_embedded_model_config() -> None:
    direct = {"schema_version": 1, "provider": "openrouter", "model": "direct"}
    embedded = {"schema_version": 1, "model": {**direct, "model": "embedded"}}

    assert settings_from_config(direct, api_key="test").model == "direct"
    assert settings_from_config(embedded, api_key="test").model == "embedded"
