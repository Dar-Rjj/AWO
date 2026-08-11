"""Validation-only evaluation bridge for ADAS architecture search."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from statistics import fmean
from typing import Any

from awo.adas.dsl import ADASArchitecture
from awo.adas.executor import run_architecture
from awo.adas.search import ADASObservation
from awo.baselines.runner import score_baseline_result
from awo.benchmarks.data import BenchmarkExample
from awo.llm import ChatResult, OpenRouterClient
from awo.sandbox import DockerSandbox


class _CallLedger:
    """Capture successful calls even when ADAS field parsing fails afterward."""

    def __init__(self, client: OpenRouterClient, generation: str | int) -> None:
        self.client = client
        self.generation = generation
        self.responses: list[ChatResult] = []

    def chat(
        self,
        messages: Any,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        metadata: Any = None,
    ) -> Any:
        enriched = {
            **dict(metadata or {}),
            "validation_generation": self.generation,
        }
        response = self.client.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            metadata=enriched,
        )
        self.responses.append(response)
        return response


class ADASValidationEvaluator:
    """Evaluate an architecture on frozen validation examples only."""

    def __init__(
        self,
        client: OpenRouterClient,
        examples: Sequence[BenchmarkExample],
        *,
        sandbox: DockerSandbox | None = None,
        max_concurrency: int = 1,
    ) -> None:
        if not examples:
            raise ValueError("ADAS validation examples cannot be empty")
        datasets = {example.dataset for example in examples}
        if len(datasets) != 1:
            raise ValueError("ADAS validation evaluator requires one dataset")
        if any(example.split != "validate" for example in examples):
            raise ValueError("ADAS search may access validation examples only")
        dataset = next(iter(datasets))
        if dataset in {"humaneval", "mbpp"} and sandbox is None:
            raise ValueError("ADAS code validation requires the Docker sandbox")
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")
        self.client = client
        self.examples = tuple(examples)
        self.sandbox = sandbox
        self.max_concurrency = max_concurrency

    async def __call__(self, candidate: ADASArchitecture, generation: str | int) -> ADASObservation:
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def evaluate_one(
            example: BenchmarkExample,
        ) -> tuple[float, float, int, dict[str, Any] | None]:
            async with semaphore:
                ledger = _CallLedger(self.client, generation)
                try:
                    result = await run_architecture(
                        candidate,
                        ledger,  # type: ignore[arg-type]
                        example,
                    )
                    score = score_baseline_result(
                        example,
                        result,
                        sandbox=self.sandbox,
                    )
                    cost = sum(response.usage.cost or 0.0 for response in ledger.responses)
                    tokens = sum(response.usage.total_tokens for response in ledger.responses)
                    return score.score, cost, tokens, None
                except Exception as exc:
                    return (
                        0.0,
                        sum(response.usage.cost or 0.0 for response in ledger.responses),
                        sum(response.usage.total_tokens for response in ledger.responses),
                        {
                            "sample_id": example.sample_id,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )

        values = await asyncio.gather(*(evaluate_one(example) for example in self.examples))
        failures = [failure for _, _, _, failure in values if failure is not None]
        return ADASObservation(
            score=fmean(score for score, _, _, _ in values),
            cost=sum(cost for _, cost, _, _ in values),
            tokens=sum(tokens for _, _, tokens, _ in values),
            details={
                "dataset": self.examples[0].dataset,
                "generation": generation,
                "sample_count": len(values),
                "failure_count": len(failures),
                "failures": failures,
            },
        )
