from __future__ import annotations

import asyncio
from typing import Any

import pytest

from awo.aflow import AFlowValidationEvaluator, WorkflowCandidate, initial_candidate
from awo.benchmarks.data import BenchmarkExample
from awo.llm import ChatResult, TokenUsage


class FakeClient:
    def __init__(self, content: str) -> None:
        self.content = content

    def chat(self, messages: Any, *, metadata: Any = None, **_: Any) -> ChatResult:
        return ChatResult(
            request_id="request",
            response_id="response",
            requested_model="deepseek/deepseek-chat",
            actual_model="deepseek/deepseek-chat",
            provider="test",
            content=self.content,
            finish_reason="stop",
            usage=TokenUsage(total_tokens=10, cost=0.001),
            attempts=1,
            latency_seconds=0.01,
            request_sha256="a" * 64,
        )


def hotpot(split: str = "validate") -> BenchmarkExample:
    return BenchmarkExample(
        1,
        "hotpotqa",
        split,
        "hotpot-1",
        "Where?",
        "Paris",
        metadata={"question": "Where?", "context": []},
    )


def test_aflow_validation_rejects_test_split_and_scores_validation() -> None:
    with pytest.raises(ValueError, match="validation"):
        AFlowValidationEvaluator(FakeClient("Paris"), [hotpot("test")])  # type: ignore[arg-type]

    observation = asyncio.run(
        AFlowValidationEvaluator(  # type: ignore[arg-type]
            FakeClient("Paris"),
            [hotpot()],
        )(initial_candidate("hotpotqa"), 0, 1)
    )
    assert observation.score == 1.0
    assert observation.tokens == 10
    assert observation.cost == 0.001


def test_aflow_validation_keeps_cost_when_operator_parse_fails() -> None:
    candidate = WorkflowCandidate(
        modification="structured answer",
        graph={
            "schema_version": 1,
            "nodes": [
                {
                    "id": "answer",
                    "operator": "AnswerGenerate",
                    "inputs": {"input": "$problem"},
                }
            ],
            "output": "$answer.answer",
        },
    )
    observation = asyncio.run(
        AFlowValidationEvaluator(  # type: ignore[arg-type]
            FakeClient("not json"),
            [hotpot()],
        )(candidate, 0, 2)
    )
    assert observation.score == 0.0
    assert observation.tokens == 10
    assert observation.cost == 0.001
    assert observation.details["failure_count"] == 1
