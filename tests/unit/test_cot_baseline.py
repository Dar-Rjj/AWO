from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from awo.baselines import CoTBaseline, build_cot_prompt
from awo.baselines.cot import COT_TEMPLATES
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
            request_sha256="b" * 64,
        )


def example(dataset: str) -> BenchmarkExample:
    metadata = {}
    if dataset == "hotpotqa":
        metadata = {"question": "Where?", "context": [["Title", ["First.", "Second."]]]}
    return BenchmarkExample(
        1,
        dataset,
        "validate",
        f"{dataset}-0",
        "Question text",
        "4",
        entry_point="solve" if dataset in {"humaneval", "mbpp"} else None,
        test_code="def check(): pass" if dataset in {"humaneval", "mbpp"} else None,
        metadata=metadata,
    )


@pytest.mark.parametrize("dataset", sorted(COT_TEMPLATES))
def test_cot_uses_exactly_one_request(dataset: str) -> None:
    client = FakeClient('{"solution":"Reasoning. Answer is 4","answer":"direct"}')
    result = CoTBaseline(client).run(example(dataset))  # type: ignore[arg-type]
    assert result.call_count == 1
    assert len(client.calls) == 1
    assert client.calls[0][1]["method"] == "cot"
    assert client.calls[0][1]["role"] == "generator"
    assert len(result.prompt_sha256) == 64


def test_hotpot_cot_preserves_question_and_context_fields() -> None:
    prompt = build_cot_prompt(example("hotpotqa"))
    assert "Think step by step" in prompt.user
    assert "Question: Where?" in prompt.user
    assert "The revelant context: First. Second." in prompt.user


def test_drop_primary_prompt_is_explicitly_inferred() -> None:
    prompt = build_cot_prompt(example("drop"))
    assert prompt.provenance == "paper-faithful/inferred"
    assert "Question text" in prompt.user


@pytest.mark.parametrize("dataset", sorted(COT_TEMPLATES))
def test_documented_prompt_matches_runtime_template(dataset: str) -> None:
    path = Path(__file__).parents[2] / "prompts" / "baselines" / "cot" / f"{dataset}.txt"
    assert path.read_text(encoding="utf-8") == COT_TEMPLATES[dataset]
