"""Shared scoring bridge from baseline outputs to the controlled evaluators."""

from __future__ import annotations

from awo.baselines.models import BaselineResult
from awo.benchmarks.code import score_code
from awo.benchmarks.data import BenchmarkExample
from awo.benchmarks.scoring import ScoreResult, score_prediction
from awo.sandbox import DockerSandbox


def score_baseline_result(
    example: BenchmarkExample,
    result: BaselineResult,
    *,
    sandbox: DockerSandbox | None = None,
    math_mode: str = "archive",
) -> ScoreResult:
    if result.sample_id != example.sample_id or result.dataset != example.dataset:
        raise ValueError("baseline result does not belong to the supplied example")
    if example.dataset in {"humaneval", "mbpp"}:
        if sandbox is None:
            raise ValueError("code benchmark scoring requires a Docker sandbox")
        if example.test_code is None:
            raise ValueError(f"missing test code for {example.sample_id}")
        return score_code(
            example.dataset,
            result.prediction,
            tests=example.test_code,
            entry_point=example.entry_point,
            sandbox=sandbox,
        )
    return score_prediction(
        example.dataset, example.reference, result.prediction, math_mode=math_mode
    )
