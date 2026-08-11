from __future__ import annotations

import asyncio
from typing import Any

import pytest

from awo.adas import (
    ADASArchitecture,
    ADASValidationError,
    execute_architecture,
    initial_archive,
    run_architecture,
    validate_architecture,
)
from awo.benchmarks.data import BenchmarkExample
from awo.llm import ChatResult, TokenUsage


class FakeClient:
    def __init__(self, contents: list[str]) -> None:
        self.contents = iter(contents)
        self.calls: list[tuple[Any, Any, Any]] = []

    def chat(
        self,
        messages: Any,
        *,
        temperature: Any = None,
        metadata: Any = None,
    ) -> ChatResult:
        index = len(self.calls)
        self.calls.append((messages, temperature, metadata))
        return ChatResult(
            request_id=f"request-{index}",
            response_id=f"response-{index}",
            requested_model="deepseek/deepseek-chat",
            actual_model="deepseek/deepseek-chat",
            provider="test",
            content=next(self.contents),
            finish_reason="stop",
            usage=TokenUsage(total_tokens=10, cost=0.001),
            attempts=1,
            latency_seconds=0.01,
            request_sha256=str(index) * 64,
        )


def test_official_seven_seed_adaptations_are_valid() -> None:
    seeds = initial_archive()
    assert [item.name for item in seeds] == [
        "Chain-of-Thought",
        "Self-Consistency with Chain-of-Thought",
        "Self-Refine (Reflexion)",
        "LLM Debate",
        "Step-back Abstraction",
        "Quality-Diversity",
        "Dynamic Assignment of Roles",
    ]
    assert all(1 <= len(item.architecture["nodes"]) <= 12 for item in seeds)


def test_executor_interprets_dag_through_unified_client() -> None:
    client = FakeClient(['{"thinking":"six squared","answer":"36"}'])
    prediction, responses, trace = asyncio.run(
        execute_architecture(
            initial_archive()[0],
            client,  # type: ignore[arg-type]
            task="What is 6 squared?",
            dataset="gsm8k",
            sample_id="gsm8k-smoke",
        )
    )
    assert prediction == "36"
    assert len(responses) == 1
    assert trace[0]["id"] == "cot"
    assert client.calls[0][2]["role"] == "executor"


def test_code_adapter_includes_entry_point_but_not_tests() -> None:
    client = FakeClient(
        ['{"thinking":"implement directly","answer":"def target(x):\\n    return x"}']
    )
    example = BenchmarkExample(
        schema_version=1,
        dataset="humaneval",
        split="validate",
        sample_id="HumanEval/0",
        prompt="Write an identity function.",
        reference="",
        entry_point="target",
        test_code="assert target(1) == 1",
    )
    result = asyncio.run(
        run_architecture(
            initial_archive()[0],
            client,  # type: ignore[arg-type]
            example,
        )
    )
    user_prompt = client.calls[0][0][1]["content"]
    assert "entry-point function `target`" in user_prompt
    assert "assert target" not in user_prompt
    assert result.protocol == "protocol-compatible/official-meta-agent-safe-dag"


def test_validation_rejects_forward_reference_and_generated_code_field() -> None:
    forward = ADASArchitecture(
        thought="invalid",
        name="invalid",
        architecture={
            "schema_version": 1,
            "nodes": [
                {
                    "id": "first",
                    "role": "assistant",
                    "temperature": 0.5,
                    "output_fields": ["answer"],
                    "instruction": "answer",
                    "inputs": ["$later.answer"],
                }
            ],
            "output": "$first.answer",
        },
    )
    with pytest.raises(ADASValidationError, match="forward"):
        validate_architecture(forward)

    graph = dict(initial_archive()[0].architecture)
    graph["python"] = "exec('bad')"
    with pytest.raises(ADASValidationError, match="unsupported"):
        validate_architecture(
            ADASArchitecture(thought="invalid", name="invalid", architecture=graph)
        )
