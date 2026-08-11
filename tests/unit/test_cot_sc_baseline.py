from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from awo.baselines import CoTSCBaseline, build_selector_prompt, parse_selector_letter
from awo.baselines.cot_sc import (
    HOTPOT_SELECTOR_TEMPLATE,
    SELECTOR_TEMPLATE,
    CoTSCProtocolError,
)
from awo.benchmarks.data import BenchmarkExample
from awo.llm import ChatResult, TokenUsage


class FakeClient:
    def __init__(self, contents: list[str]) -> None:
        self.contents = iter(contents)
        self.calls: list[tuple[Any, Any]] = []

    def chat(self, messages: Any, *, metadata: Any = None) -> ChatResult:
        index = len(self.calls)
        self.calls.append((messages, metadata))
        return ChatResult(
            request_id=f"request-{index}",
            response_id=f"response-{index}",
            requested_model="deepseek/deepseek-chat",
            actual_model="deepseek/deepseek-chat",
            provider="test",
            content=next(self.contents),
            finish_reason="stop",
            usage=TokenUsage(),
            attempts=1,
            latency_seconds=0.01,
            request_sha256=str(index) * 64,
        )


def example(dataset: str = "gsm8k") -> BenchmarkExample:
    metadata = {}
    if dataset == "hotpotqa":
        metadata = {"question": "Where?", "context": [["Title", ["First.", "Second."]]]}
    return BenchmarkExample(
        1, dataset, "validate", f"{dataset}-0", "Question", "3", metadata=metadata
    )


@pytest.mark.parametrize("dataset", ["hotpotqa", "drop", "gsm8k", "math", "humaneval", "mbpp"])
def test_cot_sc_has_fixed_five_plus_one_call_graph(dataset: str) -> None:
    candidates = [f'{{"solution":"candidate {i}","answer":"candidate {i}"}}' for i in range(5)]
    client = FakeClient([*candidates, '{"solution_letter":"C"}'])
    result = CoTSCBaseline(client).run(example(dataset))  # type: ignore[arg-type]
    assert result.call_count == 6
    assert len(client.calls) == 6
    assert [call[1]["candidate_index"] for call in client.calls[:5]] == list(range(5))
    assert client.calls[5][1]["role"] == "selector"
    assert result.artifacts["selected_letter"] == "C"
    assert result.artifacts["selected_index"] == 2
    assert result.prediction == "candidate 2"


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ('{"solution_letter":"B"}', "B"),
        ("D", "D"),
        ("solution_letter: e", "E"),
        ('{"thought":"line one\nline two","solution_letter": "A"}', "A"),
    ],
)
def test_selector_letter_parser(content: str, expected: str) -> None:
    assert parse_selector_letter(content) == expected


@pytest.mark.parametrize("content", ["", "F", '{"solution_letter":"Z"}', "I choose option A"])
def test_selector_letter_parser_fails_closed(content: str) -> None:
    with pytest.raises(CoTSCProtocolError):
        parse_selector_letter(content)


def test_selector_lists_exactly_a_through_e() -> None:
    prompt = build_selector_prompt(example(), ["one", "two", "three", "four", "five"])
    for letter in "ABCDE":
        assert f"{letter}: \n" in prompt
    assert "F: \n" not in prompt


def test_documented_selector_prompts_match_runtime_templates() -> None:
    root = Path(__file__).parents[2] / "prompts" / "baselines" / "cot_sc"
    assert (root / "selector.txt").read_text(encoding="utf-8") == SELECTOR_TEMPLATE
    assert (root / "hotpotqa_selector.txt").read_text(encoding="utf-8") == (
        HOTPOT_SELECTOR_TEMPLATE
    )
