from __future__ import annotations

from typing import Any

import pytest

from awo.benchmarks.data import BenchmarkExample
from awo.experiments.strict_math_io import (
    STRICT_MATH_IO_PROTOCOL,
    StrictMathIOBaseline,
    build_strict_math_io_prompt,
    strict_prompt_sha256,
)
from awo.llm import ChatResult, TokenUsage


class FakeClient:
    def __init__(self, content: str) -> None:
        self.content = content
        self.messages: list[dict[str, str]] = []

    def chat(self, messages: Any, *, metadata: Any = None) -> ChatResult:
        self.messages = messages
        return ChatResult(
            request_id="request-1",
            response_id="response-1",
            requested_model="openai/gpt-4o-mini",
            actual_model="openai/gpt-4o-mini",
            provider="test",
            content=self.content,
            finish_reason="stop",
            usage=TokenUsage(),
            attempts=1,
            latency_seconds=0.01,
            request_sha256="a" * 64,
        )


def example(dataset: str = "math") -> BenchmarkExample:
    return BenchmarkExample(
        schema_version=1,
        dataset=dataset,
        split="test",
        sample_id=f"{dataset}-test-0000",
        prompt="What is 2+3?",
        reference=r"\boxed{5}",
        entry_point=None,
        test_code=None,
        metadata={},
    )


def test_strict_prompt_has_an_exact_json_boxed_schema_without_code_language() -> None:
    prompt = build_strict_math_io_prompt(example())
    assert r'{"solution":"\\boxed{<answer>}"}' in prompt.system
    assert "fully evaluated literal mathematical answer" in prompt.system
    assert "Python" not in prompt.system
    assert "program" not in prompt.system
    assert "code" not in prompt.system
    assert "only the JSON object" in prompt.user
    assert prompt.provenance == STRICT_MATH_IO_PROTOCOL


def test_strict_baseline_records_effective_prompt_and_protocol() -> None:
    client = FakeClient(r'{"solution":"\\boxed{5}"}')
    result = StrictMathIOBaseline(client).run(example())  # type: ignore[arg-type]
    assert result.prediction == r"\boxed{5}"
    assert result.prompt_sha256 == strict_prompt_sha256(example())
    assert result.protocol == STRICT_MATH_IO_PROTOCOL
    assert len(client.messages) == 2


def test_strict_prompt_rejects_non_math_dataset() -> None:
    with pytest.raises(ValueError, match="only supports the math dataset"):
        build_strict_math_io_prompt(example("gsm8k"))
