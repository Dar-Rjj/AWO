"""Validation-only evaluator for declarative AFlow workflow search."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from statistics import fmean
from typing import Any

from awo.aflow.dsl import WorkflowCandidate, execute_candidate
from awo.aflow.runtime import AFlowRuntime
from awo.aflow.search import ValidationObservation
from awo.baselines.models import BaselineResult
from awo.baselines.runner import score_baseline_result
from awo.benchmarks.data import BenchmarkExample
from awo.llm import OpenRouterClient
from awo.sandbox import DockerSandbox


class AFlowValidationEvaluator:
    """Evaluate one workflow only on a frozen validation slice."""

    def __init__(
        self,
        client: OpenRouterClient,
        examples: Sequence[BenchmarkExample],
        *,
        public_tests: Mapping[str, str] | None = None,
        sandbox: DockerSandbox | None = None,
        max_concurrency: int = 1,
    ) -> None:
        if not examples:
            raise ValueError("AFlow validation examples cannot be empty")
        datasets = {example.dataset for example in examples}
        if len(datasets) != 1:
            raise ValueError("AFlow validation evaluator requires one dataset")
        if any(example.split != "validate" for example in examples):
            raise ValueError("AFlow search may access validation examples only")
        dataset = next(iter(datasets))
        if dataset in {"humaneval", "mbpp", "gsm8k", "math"} and sandbox is None:
            raise ValueError(f"AFlow {dataset} search requires the Docker sandbox")
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")
        self.client = client
        self.examples = tuple(examples)
        self.dataset = dataset
        self.public_tests = dict(public_tests or {})
        self.sandbox = sandbox
        self.max_concurrency = max_concurrency

    async def __call__(
        self,
        candidate: WorkflowCandidate,
        repeat: int,
        round_number: int,
    ) -> ValidationObservation:
        semaphore = asyncio.Semaphore(self.max_concurrency)
        requires_public_tests = any(node["operator"] == "Test" for node in candidate.graph["nodes"])

        async def evaluate_one(
            example: BenchmarkExample,
        ) -> tuple[float, float, int, dict[str, Any] | None]:
            async with semaphore:
                runtime = AFlowRuntime(
                    self.client,
                    sandbox=self.sandbox,
                    run_metadata={
                        "dataset": self.dataset,
                        "sample_id": example.sample_id,
                        "search_round": round_number,
                        "validation_repeat": repeat,
                    },
                )
                try:
                    if requires_public_tests and example.sample_id not in self.public_tests:
                        raise ValueError(f"no frozen public tests for sample {example.sample_id}")
                    prediction, trace = await execute_candidate(
                        candidate,
                        runtime,
                        dataset=self.dataset,
                        example={
                            "problem": example.prompt,
                            "entry_point": example.entry_point or "",
                        },
                        public_tests=self.public_tests.get(example.sample_id, ""),
                    )
                    result = BaselineResult(
                        method="aflow",
                        dataset=self.dataset,
                        sample_id=example.sample_id,
                        prediction=prediction,
                        prompt_sha256=candidate.workflow_sha256,
                        responses=tuple(runtime.responses),
                        protocol="controlled-search/declarative_v1",
                        artifacts={
                            "candidate_sha256": candidate.sha256,
                            "trace": trace,
                        },
                    )
                    score = score_baseline_result(example, result, sandbox=self.sandbox)
                    return score.score, runtime.total_cost, runtime.total_tokens, None
                except Exception as exc:
                    return (
                        0.0,
                        runtime.total_cost,
                        runtime.total_tokens,
                        {
                            "sample_id": example.sample_id,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )

        values = await asyncio.gather(*(evaluate_one(example) for example in self.examples))
        failures = [failure for _, _, _, failure in values if failure is not None]
        return ValidationObservation(
            score=fmean(score for score, _, _, _ in values),
            cost=sum(cost for _, cost, _, _ in values),
            tokens=sum(tokens for _, _, tokens, _ in values),
            details={
                "dataset": self.dataset,
                "round": round_number,
                "repeat": repeat,
                "sample_count": len(values),
                "failure_count": len(failures),
                "failures": failures,
            },
        )
