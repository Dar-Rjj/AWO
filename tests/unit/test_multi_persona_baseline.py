from __future__ import annotations

from typing import Any

import pytest

from awo.baselines import MultiPersonaBaseline
from awo.baselines.multi_persona import build_persona_prompt, roles_for_dataset
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
def test_multi_persona_has_fixed_seven_call_graph(dataset: str) -> None:
    turns = [
        f'{{"thinking":"thinking-{index}","answer":"answer-{index}"}}'
        for index in range(6)
    ]
    client = FakeClient([*turns, '{"solution":"3"}'])
    result = MultiPersonaBaseline(client).run(example(dataset))  # type: ignore[arg-type]
    assert result.call_count == 7
    assert len(client.calls) == 7
    assert [call[1]["round_index"] for call in client.calls[:6]] == [0, 0, 0, 1, 1, 1]
    assert [call[1]["persona_index"] for call in client.calls[:6]] == [0, 1, 2, 0, 1, 2]
    assert client.calls[-1][1]["role"] == "synthesizer"
    assert result.prediction == "3"
    assert len(result.artifacts["rounds"]) == 2


def test_round_two_receives_all_round_one_thinking() -> None:
    roles = roles_for_dataset("humaneval")
    prompt = build_persona_prompt(
        example("humaneval"),
        roles[1],
        round_index=1,
        prior_thinking=["alpha", "beta", "gamma"],
    )
    assert f"{roles[1]}'s previous round thinking: beta" in prompt
    assert f"{roles[0]}'s thinking: alpha" in prompt
    assert f"{roles[2]}'s thinking: gamma" in prompt
    assert "['alpha', 'beta', 'gamma']" not in prompt


def test_round_two_requires_exact_peer_context() -> None:
    role = roles_for_dataset("mbpp")[0]
    with pytest.raises(ValueError):
        build_persona_prompt(
            example("mbpp"), role, round_index=1, prior_thinking=["only one"]
        )


def test_drop_is_explicitly_inferred() -> None:
    contents = ['{"thinking":"t","answer":"a"}'] * 6 + ['{"solution":"3"}']
    result = MultiPersonaBaseline(FakeClient(contents)).run(example("drop"))  # type: ignore[arg-type]
    assert result.artifacts["prompt_provenance"] == "paper-faithful/inferred"
