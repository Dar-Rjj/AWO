from __future__ import annotations

import asyncio
from typing import Any

import pytest

from awo.aflow.dsl import (
    WorkflowCandidate,
    WorkflowDSLValidationError,
    execute_candidate,
    initial_candidate,
    validate_candidate,
)
from awo.aflow.runtime import AFlowRuntime
from awo.llm import ChatResult, TokenUsage


class FakeClient:
    def chat(self, messages: Any, *, metadata: Any = None) -> ChatResult:
        return ChatResult(
            request_id="request",
            response_id="response",
            requested_model="deepseek/deepseek-chat",
            actual_model="deepseek/deepseek-chat",
            provider="test",
            content="36",
            finish_reason="stop",
            usage=TokenUsage(total_tokens=10, cost=0.001),
            attempts=1,
            latency_seconds=0.01,
            request_sha256="a" * 64,
        )


def test_initial_candidate_executes_without_generated_python() -> None:
    candidate = initial_candidate("gsm8k")
    prediction, trace = asyncio.run(
        execute_candidate(
            candidate,
            AFlowRuntime(FakeClient()),  # type: ignore[arg-type]
            dataset="gsm8k",
            example={"question": "What is 6 squared?"},
        )
    )
    assert prediction == "36"
    assert [item["operator"] for item in trace] == ["Custom"]


@pytest.mark.parametrize(
    ("candidate", "error"),
    [
        (
            WorkflowCandidate(
                "bad op",
                {
                    "schema_version": 1,
                    "nodes": [
                        {
                            "id": "x",
                            "operator": "Programmer",
                            "inputs": {"problem": "$problem"},
                        }
                    ],
                    "output": "$x.response",
                },
            ),
            "not allowed",
        ),
        (
            WorkflowCandidate(
                "forward",
                {
                    "schema_version": 1,
                    "nodes": [
                        {
                            "id": "x",
                            "operator": "Custom",
                            "inputs": {
                                "input": "$later.response",
                                "instruction": "",
                            },
                        },
                        {
                            "id": "later",
                            "operator": "Custom",
                            "inputs": {"input": "$problem", "instruction": ""},
                        },
                    ],
                    "output": "$x.response",
                },
            ),
            "forward",
        ),
        (
            WorkflowCandidate(
                "missing prompt",
                {
                    "schema_version": 1,
                    "nodes": [
                        {
                            "id": "x",
                            "operator": "Custom",
                            "inputs": {
                                "input": "$problem",
                                "instruction": "$prompt.SOLVE",
                            },
                        }
                    ],
                    "output": "$x.response",
                },
            ),
            "undefined prompt",
        ),
    ],
)
def test_validation_rejects_unsafe_graphs(candidate: WorkflowCandidate, error: str) -> None:
    with pytest.raises(WorkflowDSLValidationError, match=error):
        validate_candidate(candidate, "hotpotqa")


def test_validation_rejects_excess_nodes_and_prompt_placeholders() -> None:
    nodes = [
        {
            "id": f"n{index}",
            "operator": "Custom",
            "inputs": {"input": "$problem", "instruction": ""},
        }
        for index in range(11)
    ]
    too_large = WorkflowCandidate(
        "too large",
        {"schema_version": 1, "nodes": nodes, "output": "$n0.response"},
    )
    with pytest.raises(WorkflowDSLValidationError, match="between 1 and 10"):
        validate_candidate(too_large, "drop")

    placeholder = WorkflowCandidate(
        "placeholder",
        initial_candidate("drop").graph,
        prompts={"SOLVE": "Solve {problem}"},
    )
    with pytest.raises(WorkflowDSLValidationError, match="placeholder"):
        validate_candidate(placeholder, "drop")
