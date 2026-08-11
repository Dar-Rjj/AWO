from __future__ import annotations

import asyncio
import json
from typing import Any

from awo.aflow.dsl import OPERATOR_SETS, initial_candidate
from awo.aflow.optimizer import OpenRouterCandidateGenerator
from awo.aflow.search import RoundRecord, SearchContext
from awo.llm import ChatResult, TokenUsage


class FakeClient:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[tuple[Any, Any, Any]] = []

    def chat(
        self,
        messages: Any,
        *,
        temperature: Any = None,
        metadata: Any = None,
    ) -> ChatResult:
        self.calls.append((messages, temperature, metadata))
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


def test_optimizer_parses_and_validates_json_candidate() -> None:
    payload = {
        "modification": "Ask for a concise derivation.",
        "graph": {
            "schema_version": 1,
            "nodes": [
                {
                    "id": "answer",
                    "operator": "Custom",
                    "inputs": {
                        "input": "$problem",
                        "instruction": "Derive, verify, and answer.",
                    },
                }
            ],
            "output": "$answer.response",
        },
        "prompts": {},
    }
    client = FakeClient(json.dumps(payload))
    parent = initial_candidate("gsm8k")
    record = RoundRecord(
        round=1,
        parent_round=None,
        candidate_sha256=parent.sha256,
        modification=parent.modification,
        scores=(0.0,),
        mean_score=0.0,
        std_score=0.0,
        total_cost=0.0,
        total_tokens=0,
        generation_attempts=0,
        improved=True,
    )
    context = SearchContext(
        dataset="gsm8k",
        generated_index=1,
        round=2,
        parent=parent,
        parent_record=record,
        parent_pool=(record,),
        selection_probabilities=(1.0,),
        experience="none",
        allowed_operators=tuple(sorted(OPERATOR_SETS["gsm8k"])),
    )
    candidate = asyncio.run(
        OpenRouterCandidateGenerator(client).generate(context)  # type: ignore[arg-type]
    )
    assert candidate.parent_round == 1
    assert candidate.metadata["request_sha256"]
    assert client.calls[0][2]["role"] == "aflow_optimizer"
