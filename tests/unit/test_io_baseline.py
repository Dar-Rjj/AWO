from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from awo.baselines import IOBaseline, build_io_prompt, parse_io_response
from awo.baselines.io import IO_TEMPLATES
from awo.benchmarks.data import BenchmarkExample
from awo.llm import ChatResult, TokenUsage


class FakeClient:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[tuple[Any, Any]] = []

    def chat(self, messages: Any, *, metadata: Any = None) -> ChatResult:
        self.calls.append((messages, metadata))
        return ChatResult(
            request_id="request-1",
            response_id="response-1",
            requested_model="deepseek/deepseek-chat",
            actual_model="deepseek/deepseek-chat",
            provider="test",
            content=self.content,
            finish_reason="stop",
            usage=TokenUsage(),
            attempts=1,
            latency_seconds=0.01,
            request_sha256="a" * 64,
        )


def example(dataset: str) -> BenchmarkExample:
    metadata = {}
    prompt = "Question text"
    if dataset == "hotpotqa":
        metadata = {"question": "Who?", "context": [["Title", ["One.", "Two."]]]}
    return BenchmarkExample(
        schema_version=1,
        dataset=dataset,
        split="validate",
        sample_id=f"{dataset}-0",
        prompt=prompt,
        reference="answer",
        entry_point="solve" if dataset in {"humaneval", "mbpp"} else None,
        test_code="def check(): pass" if dataset in {"humaneval", "mbpp"} else None,
        metadata=metadata,
    )


@pytest.mark.parametrize("dataset", ["hotpotqa", "drop", "gsm8k", "math", "humaneval", "mbpp"])
def test_io_uses_exactly_one_request(dataset: str) -> None:
    client = FakeClient('{"solution":"Answer is 4","answer":"direct"}')
    result = IOBaseline(client).run(example(dataset))  # type: ignore[arg-type]
    assert result.call_count == 1
    assert len(client.calls) == 1
    assert client.calls[0][1]["method"] == "io"
    assert client.calls[0][1]["dataset"] == dataset
    assert len(result.prompt_sha256) == 64


def test_hotpot_prompt_matches_historical_field_order() -> None:
    prompt = build_io_prompt(example("hotpotqa"))
    assert "Question: Who?" in prompt.user
    assert "The revelant context: One. Two." in prompt.user
    assert prompt.provenance == "upstream-user/adapted-schema"


@pytest.mark.parametrize("dataset", sorted(IO_TEMPLATES))
def test_documented_prompt_matches_runtime_template(dataset: str) -> None:
    path = Path(__file__).parents[2] / "prompts" / "baselines" / "io" / f"{dataset}.txt"
    assert path.read_text(encoding="utf-8") == IO_TEMPLATES[dataset]


@pytest.mark.parametrize(
    ("dataset", "content", "expected"),
    [
        ("drop", '{"answer": "20"}', "20"),
        ("hotpotqa", "Thought\nFinal Answer: Paris", "Paris"),
        ("gsm8k", '{"solution": "Answer is 36"}', "Answer is 36"),
        ("math", '{"solution": "Therefore \\\\boxed{5}"}', "Therefore \\boxed{5}"),
        ("humaneval", "```python\ndef solve():\n    return 1\n```", "def solve():\n    return 1"),
    ],
)
def test_parse_io_response(dataset: str, content: str, expected: str) -> None:
    assert parse_io_response(dataset, content) == expected
