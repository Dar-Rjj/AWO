from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from awo.aflow.dsl import WorkflowCandidate
from awo.aflow.search import (
    AFlowSearch,
    RoundRecord,
    SearchConfig,
    ValidationObservation,
    mixed_probabilities,
    select_parent_pool,
)


class IncrementingGenerator:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, context):  # type: ignore[no-untyped-def]
        self.calls += 1
        return WorkflowCandidate(
            modification=f"candidate {self.calls}",
            graph={
                "schema_version": 1,
                "nodes": [
                    {
                        "id": "answer",
                        "operator": "Custom",
                        "inputs": {
                            "input": "$problem",
                            "instruction": f"version {self.calls}",
                        },
                    }
                ],
                "output": "$answer.response",
            },
            parent_round=context.parent_record.round,
        )


class DuplicateThenValidGenerator(IncrementingGenerator):
    async def generate(self, context):  # type: ignore[no-untyped-def]
        self.calls += 1
        if self.calls == 1:
            return WorkflowCandidate(
                modification="rename the unchanged workflow",
                graph=context.parent.graph,
                parent_round=context.parent_record.round,
            )
        return WorkflowCandidate(
            modification="actually change the instruction",
            graph={
                "schema_version": 1,
                "nodes": [
                    {
                        "id": "answer",
                        "operator": "Custom",
                        "inputs": {
                            "input": "$problem",
                            "instruction": "verify once",
                        },
                    }
                ],
                "output": "$answer.response",
            },
            parent_round=context.parent_record.round,
        )


def _record(round_number: int, score: float) -> RoundRecord:
    return RoundRecord(
        round=round_number,
        parent_round=None,
        candidate_sha256=str(round_number),
        modification=str(round_number),
        scores=(score,),
        mean_score=score,
        std_score=0.0,
        total_cost=0.0,
        total_tokens=0,
        generation_attempts=1,
        improved=True,
    )


def test_mixed_probabilities_and_parent_pool() -> None:
    probabilities = mixed_probabilities([0.1, 0.9], alpha=0.4, lambda_weight=0.2)
    assert abs(sum(probabilities) - 1.0) < 1e-12
    assert probabilities[1] > probabilities[0]
    records = [_record(1, 0.0), _record(2, 0.2), _record(3, 0.9), _record(4, 0.5)]
    assert [item.round for item in select_parent_pool(records, 2)] == [1, 3, 4]


def test_search_persists_repeats_and_resumes(tmp_path: Path) -> None:
    generator = IncrementingGenerator()

    async def evaluator(candidate, repeat, round_number):  # type: ignore[no-untyped-def]
        return ValidationObservation(
            score=round_number / 10,
            cost=0.01,
            tokens=10 + repeat,
        )

    config = SearchConfig(
        max_generated_rounds=3,
        validation_repeats=2,
        early_stop_rounds=5,
    )
    search = AFlowSearch(
        dataset="gsm8k",
        config=config,
        generator=generator,
        evaluator=evaluator,
        output_dir=tmp_path,
    )
    summary = asyncio.run(search.run())
    assert summary.completed_generated_rounds == 3
    assert summary.best_round == 4
    assert generator.calls == 3
    assert len(list(tmp_path.glob("rounds/round_*/evaluations.jsonl"))) == 4
    selection = (tmp_path / "rounds" / "round_2" / "selection.json").read_text(encoding="utf-8")
    assert '"pool_rounds": [' in selection
    assert '"probabilities": [' in selection

    resumed_generator = IncrementingGenerator()
    resumed_summary = asyncio.run(
        AFlowSearch(
            dataset="gsm8k",
            config=config,
            generator=resumed_generator,
            evaluator=evaluator,
            output_dir=tmp_path,
        ).run()
    )
    assert resumed_summary.records == summary.records
    assert resumed_generator.calls == 0


def test_failures_count_as_zero_and_trigger_early_stop(tmp_path: Path) -> None:
    generator = IncrementingGenerator()

    async def evaluator(candidate, repeat, round_number):  # type: ignore[no-untyped-def]
        if repeat == 1:
            raise RuntimeError("validation failed")
        return ValidationObservation(score=1.0 if round_number == 1 else 0.5)

    config = SearchConfig(
        max_generated_rounds=10,
        validation_repeats=2,
        early_stop_rounds=2,
    )
    summary = asyncio.run(
        AFlowSearch(
            dataset="gsm8k",
            config=config,
            generator=generator,
            evaluator=evaluator,
            output_dir=tmp_path,
        ).run()
    )
    assert summary.stopped_reason == "early_stop"
    assert summary.completed_generated_rounds == 2
    assert all(len(record.scores) == 2 for record in summary.records)
    assert summary.records[0].scores == (1.0, 0.0)


def test_mapping_accepts_paper_parameter_names() -> None:
    config = SearchConfig.from_mapping(
        {
            "max_rounds": 20,
            "validation_rounds": 5,
            "alpha": 0.4,
            "lambda": 0.2,
        }
    )
    assert config == SearchConfig()


def test_duplicate_graph_retries_even_if_modification_text_changes(
    tmp_path: Path,
) -> None:
    generator = DuplicateThenValidGenerator()

    def evaluator(candidate, repeat, round_number):  # type: ignore[no-untyped-def]
        return ValidationObservation(score=float(round_number))

    summary = asyncio.run(
        AFlowSearch(
            dataset="gsm8k",
            config=SearchConfig(max_generated_rounds=1, validation_repeats=1),
            generator=generator,
            evaluator=evaluator,
            output_dir=tmp_path,
        ).run()
    )
    assert generator.calls == 2
    assert summary.records[1].generation_attempts == 2


def test_resume_rejects_candidate_tampering(tmp_path: Path) -> None:
    generator = IncrementingGenerator()

    def evaluator(candidate, repeat, round_number):  # type: ignore[no-untyped-def]
        return ValidationObservation(score=1.0)

    search = AFlowSearch(
        dataset="gsm8k",
        config=SearchConfig(max_generated_rounds=0, validation_repeats=1),
        generator=generator,
        evaluator=evaluator,
        output_dir=tmp_path,
    )
    asyncio.run(search.run())
    candidate_file = tmp_path / "rounds" / "round_1" / "candidate.json"
    candidate_file.write_text(
        candidate_file.read_text(encoding="utf-8").replace(
            "Initial single-agent workflow.", "tampered"
        ),
        encoding="utf-8",
    )
    try:
        asyncio.run(search.run())
    except ValueError as exc:
        assert "integrity" in str(exc)
    else:
        raise AssertionError("tampered candidate was accepted")


def test_resume_rejects_changed_run_fingerprint(tmp_path: Path) -> None:
    def evaluator(candidate, repeat, round_number):  # type: ignore[no-untyped-def]
        return ValidationObservation(score=1.0)

    config = SearchConfig(max_generated_rounds=0, validation_repeats=1)
    asyncio.run(
        AFlowSearch(
            dataset="gsm8k",
            config=config,
            generator=IncrementingGenerator(),
            evaluator=evaluator,
            output_dir=tmp_path,
            run_fingerprint={"dataset_sha256": "a" * 64, "commit": "one"},
        ).run()
    )
    with pytest.raises(ValueError, match="configuration"):
        asyncio.run(
            AFlowSearch(
                dataset="gsm8k",
                config=config,
                generator=IncrementingGenerator(),
                evaluator=evaluator,
                output_dir=tmp_path,
                run_fingerprint={"dataset_sha256": "a" * 64, "commit": "two"},
            ).run()
        )
