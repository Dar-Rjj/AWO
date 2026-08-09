from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from awo.baselines import SelfRefineBaseline
from awo.baselines.self_refine import (
    GSM8K_REVISE_TEMPLATE,
    HUMANEVAL_REVISE_TEMPLATE,
    MATH_REVISE_TEMPLATE,
    MBPP_REVISE_TEMPLATE,
    QA_REVISE_TEMPLATE,
    REVIEW_TEMPLATE,
    SelfRefineProtocolError,
    parse_review_decision,
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
        metadata = {"question": "Where?", "context": [["Title", ["Context."]]]}
    return BenchmarkExample(
        1, dataset, "validate", f"{dataset}-0", "Question", "3", metadata=metadata
    )


@pytest.mark.parametrize("dataset", ["hotpotqa", "drop", "gsm8k", "math", "humaneval", "mbpp"])
def test_accept_first_review_uses_two_calls(dataset: str) -> None:
    client = FakeClient(['{"solution":"3"}', '{"review_result":true,"feedback":"ok"}'])
    result = SelfRefineBaseline(client).run(example(dataset))  # type: ignore[arg-type]
    assert result.call_count == 2
    assert result.prediction == "3"
    assert result.artifacts["stop_reason"] == "review_accepted"
    assert [call[1]["role"] for call in client.calls] == ["generator", "reviewer"]


def test_three_rejections_use_maximum_seven_calls() -> None:
    contents = ['{"solution":"0"}']
    for index in range(3):
        contents.extend(
            [
                f'{{"review_result":false,"feedback":"error-{index}"}}',
                f'{{"solution":"{index + 1}"}}',
            ]
        )
    client = FakeClient(contents)
    result = SelfRefineBaseline(client).run(example())  # type: ignore[arg-type]
    assert result.call_count == 7
    assert result.prediction == "3"
    assert result.artifacts["stop_reason"] == "max_rounds_exhausted"
    assert len(result.artifacts["iterations"]) == 3
    assert result.artifacts["iterations"][-1]["solution_after"] == "3"


def test_solution_text_not_mapping_repr_is_used_in_review() -> None:
    client = FakeClient(['{"solution":"Answer is 3"}', '{"review_result":true,"feedback":""}'])
    SelfRefineBaseline(client).run(example())  # type: ignore[arg-type]
    review_prompt = client.calls[1][0][1]["content"]
    assert "solution: Answer is 3" in review_prompt
    assert "{'solution':" not in review_prompt


@pytest.mark.parametrize(
    "content",
    [
        '{"review_result":"false","feedback":"bad"}',
        '{"review_result":false,"feedback":"bad"}',
    ],
)
def test_explicit_false_is_never_truthy(content: str) -> None:
    assert parse_review_decision(content).accepted is False


@pytest.mark.parametrize(
    "content", ["True", '{"review_result":1,"feedback":""}', '{"feedback":""}']
)
def test_invalid_review_fails_closed(content: str) -> None:
    with pytest.raises(SelfRefineProtocolError):
        parse_review_decision(content)


def test_documented_templates_match_runtime() -> None:
    root = Path(__file__).parents[2] / "prompts" / "baselines" / "self_refine"
    assert (root / "review.txt").read_text() == REVIEW_TEMPLATE
    assert (root / "revise_gsm8k.txt").read_text() == GSM8K_REVISE_TEMPLATE
    assert (root / "revise_math.txt").read_text() == MATH_REVISE_TEMPLATE
    assert (root / "revise_humaneval.txt").read_text() == HUMANEVAL_REVISE_TEMPLATE
    assert (root / "revise_mbpp.txt").read_text() == MBPP_REVISE_TEMPLATE
    assert (root / "revise_qa.txt").read_text() == QA_REVISE_TEMPLATE
