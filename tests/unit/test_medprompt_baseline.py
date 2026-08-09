from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from awo.baselines import MedPromptBaseline, choose_vote_winner
from awo.baselines.medprompt import (
    CODE_VOTER_TEMPLATE,
    GENERAL_VOTER_TEMPLATE,
    GSM8K_CANDIDATE_TEMPLATE,
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
def test_medprompt_has_fixed_three_plus_five_call_graph(dataset: str) -> None:
    candidates = [f'{{"solution":"candidate {i}","answer":"candidate {i}"}}' for i in range(3)]
    voters = ['{"solution_letter":"A"}'] * 5
    client = FakeClient([*candidates, *voters])
    result = MedPromptBaseline(client, seed=17).run(example(dataset))  # type: ignore[arg-type]
    assert result.call_count == 8
    assert len(client.calls) == 8
    assert [call[1]["candidate_index"] for call in client.calls[:3]] == [0, 1, 2]
    assert [call[1]["vote_index"] for call in client.calls[3:]] == list(range(5))
    assert all(sorted(item) == [0, 1, 2] for item in result.artifacts["permutations"])
    assert sum(result.artifacts["vote_counts"]) == 5


def test_shuffle_is_deterministic_for_sample_and_seed() -> None:
    contents = ['{"solution":"same"}'] * 3 + ['{"solution_letter":"A"}'] * 5
    first = MedPromptBaseline(FakeClient(contents.copy()), seed=123).run(example())  # type: ignore[arg-type]
    second = MedPromptBaseline(FakeClient(contents.copy()), seed=123).run(example())  # type: ignore[arg-type]
    assert first.artifacts["permutations"] == second.artifacts["permutations"]
    assert first.artifacts["vote_original_indices"] == second.artifacts["vote_original_indices"]


def test_duplicate_candidate_text_does_not_collapse_index_mapping() -> None:
    contents = ['{"solution":"same"}'] * 3 + ['{"solution_letter":"A"}'] * 5
    result = MedPromptBaseline(FakeClient(contents), seed=7).run(example())  # type: ignore[arg-type]
    expected = [permutation[0] for permutation in result.artifacts["permutations"]]
    assert result.artifacts["vote_original_indices"] == expected


def test_vote_tie_breaks_to_lowest_original_index() -> None:
    assert choose_vote_winner([1, 0, 1, 0, 2]) == 0


@pytest.mark.parametrize("votes", [[], [0, 0, 1, 1], [0, 0, 1, 1, 3]])
def test_vote_winner_requires_five_valid_votes(votes: list[int]) -> None:
    with pytest.raises(ValueError):
        choose_vote_winner(votes)


def test_documented_medprompt_templates_match_runtime() -> None:
    root = Path(__file__).parents[2] / "prompts" / "baselines" / "medprompt"
    assert (root / "gsm8k_candidate.txt").read_text() == GSM8K_CANDIDATE_TEMPLATE
    assert (root / "general_voter.txt").read_text() == GENERAL_VOTER_TEMPLATE
    assert (root / "code_voter.txt").read_text() == CODE_VOTER_TEMPLATE
