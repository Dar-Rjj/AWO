from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from awo.adas import (
    ADASArchitecture,
    ADASConfig,
    ADASObservation,
    ADASSearch,
    ADASValidationEvaluator,
    initial_archive,
)
from awo.benchmarks.data import BenchmarkExample
from awo.llm import ChatResult, TokenUsage


class IncrementingGenerator:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls += 1
        return ADASArchitecture(
            thought=f"candidate {self.calls}",
            name=f"Candidate {self.calls}",
            architecture={
                "schema_version": 1,
                "nodes": [
                    {
                        "id": "answer",
                        "role": "assistant",
                        "temperature": 0.5,
                        "output_fields": ["answer"],
                        "instruction": f"new instruction {self.calls}",
                        "inputs": ["$task"],
                    }
                ],
                "output": "$answer.answer",
            },
            parent_generation=kwargs["generation"] - 1 or None,
            metadata={"tokens": 30, "cost": 0.003},
        )


def test_search_evaluates_seven_seeds_generates_and_resumes(tmp_path: Path) -> None:
    generator = IncrementingGenerator()
    evaluations: list[str | int] = []

    def evaluator(candidate, generation):  # type: ignore[no-untyped-def]
        evaluations.append(generation)
        return ADASObservation(score=len(evaluations) / 10, tokens=5, cost=0.01)

    config = ADASConfig(generations=2)
    summary = asyncio.run(
        ADASSearch(
            dataset="gsm8k",
            config=config,
            generator=generator,
            evaluator=evaluator,
            output_dir=tmp_path,
        ).run()
    )
    assert summary.completed_initial == 7
    assert summary.completed_generations == 2
    assert len(summary.records) == 9
    assert generator.calls == 2
    assert (tmp_path / "archive.json").exists()
    assert (tmp_path / "entries" / "entry_008" / "candidate.json").exists()

    resumed_generator = IncrementingGenerator()
    resumed = asyncio.run(
        ADASSearch(
            dataset="gsm8k",
            config=config,
            generator=resumed_generator,
            evaluator=evaluator,
            output_dir=tmp_path,
        ).run()
    )
    assert resumed.records == summary.records
    assert resumed_generator.calls == 0


def test_evaluation_failure_is_archived_as_zero(tmp_path: Path) -> None:
    generator = IncrementingGenerator()

    def evaluator(candidate, generation):  # type: ignore[no-untyped-def]
        raise RuntimeError("validation error")

    summary = asyncio.run(
        ADASSearch(
            dataset="drop",
            config=ADASConfig(generations=0),
            generator=generator,
            evaluator=evaluator,
            output_dir=tmp_path,
        ).run()
    )
    assert len(summary.records) == 7
    assert all(record.fitness == 0 for record in summary.records)
    assert all("validation error" in (record.error or "") for record in summary.records)


class FakeClient:
    def __init__(self, content: str = '{"thinking":"calculate","answer":"36"}') -> None:
        self.content = content

    def chat(  # type: ignore[no-untyped-def]
        self, messages, *, temperature=None, max_tokens=None, metadata=None
    ):
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


def test_validation_evaluator_rejects_test_split_and_scores_validation() -> None:
    test_example = BenchmarkExample(1, "gsm8k", "test", "test-0", "6 squared?", "36")
    try:
        ADASValidationEvaluator(FakeClient(), [test_example])  # type: ignore[arg-type]
    except ValueError as exc:
        assert "validation" in str(exc)
    else:
        raise AssertionError("test split entered ADAS search")

    validation_example = BenchmarkExample(1, "gsm8k", "validate", "validate-0", "6 squared?", "36")
    observation = asyncio.run(
        ADASValidationEvaluator(  # type: ignore[arg-type]
            FakeClient(), [validation_example]
        )(initial_archive()[0], "initial-1")
    )
    assert observation.score == 1.0
    assert observation.tokens == 10


def test_adas_validation_keeps_cost_when_field_parsing_fails() -> None:
    validation_example = BenchmarkExample(1, "gsm8k", "validate", "validate-0", "6 squared?", "36")
    observation = asyncio.run(
        ADASValidationEvaluator(  # type: ignore[arg-type]
            FakeClient("not json"), [validation_example]
        )(initial_archive()[0], "initial-1")
    )
    assert observation.score == 0.0
    assert observation.tokens == 10
    assert observation.cost == 0.001
    assert observation.details["failure_count"] == 1


@pytest.mark.parametrize("dataset", ["hotpotqa", "drop", "humaneval", "mbpp", "gsm8k", "math"])
def test_all_six_dataset_search_adapters_complete_seed_archive(
    dataset: str, tmp_path: Path
) -> None:
    def evaluator(candidate, generation):  # type: ignore[no-untyped-def]
        return ADASObservation(score=0.5)

    summary = asyncio.run(
        ADASSearch(
            dataset=dataset,
            config=ADASConfig(generations=0),
            generator=IncrementingGenerator(),
            evaluator=evaluator,
            output_dir=tmp_path / dataset,
        ).run()
    )
    assert summary.completed_initial == 7


def test_adas_resume_rejects_changed_run_fingerprint(tmp_path: Path) -> None:
    def evaluator(candidate, generation):  # type: ignore[no-untyped-def]
        return ADASObservation(score=0.5)

    config = ADASConfig(generations=0)
    asyncio.run(
        ADASSearch(
            dataset="gsm8k",
            config=config,
            generator=IncrementingGenerator(),
            evaluator=evaluator,
            output_dir=tmp_path,
            run_fingerprint={"sample_ids": ["one"], "commit": "one"},
        ).run()
    )
    with pytest.raises(ValueError, match="configuration"):
        asyncio.run(
            ADASSearch(
                dataset="gsm8k",
                config=config,
                generator=IncrementingGenerator(),
                evaluator=evaluator,
                output_dir=tmp_path,
                run_fingerprint={"sample_ids": ["two"], "commit": "one"},
            ).run()
        )
