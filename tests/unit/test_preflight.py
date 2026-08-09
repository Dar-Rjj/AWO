from types import SimpleNamespace

from awo.llm.preflight import PREFLIGHT_MARKER, run_preflight


def test_preflight_uses_small_deterministic_request() -> None:
    result = SimpleNamespace(
        content=PREFLIGHT_MARKER,
        request_id="request-1",
        response_id="response-1",
        requested_model="deepseek/deepseek-chat",
        actual_model="deepseek/deepseek-chat",
        provider="fake",
        attempts=1,
        latency_seconds=0.01,
        usage=SimpleNamespace(model_dump=lambda **_kwargs: {"total_tokens": 8}),
        request_sha256="a" * 64,
    )
    client = SimpleNamespace(chat=lambda *args, **kwargs: result)

    summary = run_preflight(client)

    assert summary["ok"] is True
    assert summary["requested_model"] == "deepseek/deepseek-chat"
    assert summary["usage"] == {"total_tokens": 8}
