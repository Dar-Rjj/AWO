from __future__ import annotations

import asyncio
import json
from typing import Any

from awo.adas import ADASMetaGenerator, initial_archive
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


def _payload(*, reflected: bool) -> dict[str, Any]:
    value: dict[str, Any] = {
        "thought": "Use a careful verifier.",
        "name": "Verifier DAG",
        "architecture": {
            "schema_version": 1,
            "nodes": [
                {
                    "id": "answer",
                    "role": "reasoning agent",
                    "temperature": 0.5,
                    "output_fields": ["thinking", "answer"],
                    "instruction": "Reason and verify before answering.",
                    "inputs": ["$task"],
                }
            ],
            "output": "$answer.answer",
        },
    }
    if reflected:
        value["reflection"] = "The references and output are valid."
    return value


def test_meta_generator_uses_proposal_and_two_reflections() -> None:
    client = FakeClient(
        [
            json.dumps(_payload(reflected=False)),
            json.dumps(_payload(reflected=True)),
            json.dumps(_payload(reflected=True)),
        ]
    )
    seed = initial_archive()[0]
    archive = [
        {
            "generation": "initial-1",
            "fitness": 0.5,
            "candidate": seed.to_dict(),
        }
    ]
    result = asyncio.run(
        ADASMetaGenerator(client).generate(  # type: ignore[arg-type]
            dataset="gsm8k",
            generation=1,
            archive=archive,
            previous=None,
        )
    )
    assert result.name == "Verifier DAG"
    assert result.metadata["tokens"] == 30
    assert result.metadata["cost"] == 0.003
    assert [call[2]["phase"] for call in client.calls] == [
        "proposal",
        "reflection_1",
        "reflection_2",
    ]
    assert all(call[1] == 0.8 for call in client.calls)
