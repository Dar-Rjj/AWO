"""Resumable runner shared by frozen AFlow and ADAS executors."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from awo.aflow import AFlowRuntime, WorkflowCandidate, execute_candidate
from awo.baselines import score_baseline_result
from awo.baselines.models import BaselineResult
from awo.benchmarks.data import BenchmarkExample
from awo.llm import OpenRouterClient
from awo.sandbox import DockerSandbox

from .runner import (
    ExperimentRecord,
    ExperimentStore,
    SampleCallLedger,
    summarize_records,
)

AgenticExecutor = Callable[[SampleCallLedger, BenchmarkExample], Awaitable[BaselineResult]]


def aflow_candidate_executor(
    candidate: WorkflowCandidate,
    public_tests: Mapping[str, str],
    sandbox: DockerSandbox | None,
) -> AgenticExecutor:
    """Build a frozen declarative-AFlow executor for validation-independent test runs."""

    requires_public_tests = any(node["operator"] == "Test" for node in candidate.graph["nodes"])

    async def execute(
        client: SampleCallLedger,
        example: BenchmarkExample,
    ) -> BaselineResult:
        if requires_public_tests and example.sample_id not in public_tests:
            raise ValueError(f"no frozen public tests for sample {example.sample_id}")
        runtime = AFlowRuntime(
            client,  # type: ignore[arg-type]
            sandbox=sandbox,
            run_metadata={
                "dataset": example.dataset,
                "sample_id": example.sample_id,
                "candidate_sha256": candidate.sha256,
            },
        )
        prediction, trace = await execute_candidate(
            candidate,
            runtime,
            dataset=example.dataset,
            example={
                "problem": example.prompt,
                "entry_point": example.entry_point or "",
            },
            public_tests=public_tests.get(example.sample_id, ""),
        )
        return BaselineResult(
            method="aflow",
            dataset=example.dataset,
            sample_id=example.sample_id,
            prediction=prediction,
            prompt_sha256=candidate.workflow_sha256,
            responses=tuple(runtime.responses),
            protocol="controlled-search/declarative_v1",
            artifacts={
                "candidate_sha256": candidate.sha256,
                "workflow_sha256": candidate.workflow_sha256,
                "trace": trace,
            },
        )

    return execute


class AgenticExperimentRunner:
    """Execute one frozen agentic method with crash-safe per-sample records."""

    def __init__(
        self,
        *,
        client: OpenRouterClient,
        executor: AgenticExecutor,
        method: str,
        executor_fingerprint: Mapping[str, Any],
        examples: Sequence[BenchmarkExample],
        repeats: int,
        output_dir: Path,
        dataset_sha256: str,
        config_sha256: str,
        implementation_commit: str | None,
        sandbox: DockerSandbox | None = None,
    ) -> None:
        if not examples:
            raise ValueError("experiment examples cannot be empty")
        datasets = {example.dataset for example in examples}
        splits = {example.split for example in examples}
        if len(datasets) != 1 or len(splits) != 1:
            raise ValueError("runner requires examples from one dataset and split")
        if not method.strip() or repeats <= 0:
            raise ValueError("method cannot be empty and repeats must be positive")
        dataset = next(iter(datasets))
        if dataset in {"humaneval", "mbpp"} and sandbox is None:
            raise ValueError("code experiments require the Docker sandbox")
        self.client = client
        self.executor = executor
        self.method = method
        self.examples = tuple(examples)
        self.repeats = repeats
        self.sandbox = sandbox
        self.store = ExperimentStore(output_dir)
        self.spec = {
            "schema_version": 1,
            "dataset": dataset,
            "split": next(iter(splits)),
            "dataset_sha256": dataset_sha256,
            "config_sha256": config_sha256,
            "implementation_commit": implementation_commit,
            "sample_ids": [example.sample_id for example in examples],
            "methods": [method],
            "repeats": repeats,
            "executor_fingerprint": dict(executor_fingerprint),
            "cache_across_repeats": False,
        }

    def _key(self, repeat: int, example: BenchmarkExample) -> str:
        return f"{example.dataset}:{example.split}:{self.method}:{repeat}:{example.sample_id}"

    def _run_one(
        self,
        repeat: int,
        sample_index: int,
        example: BenchmarkExample,
    ) -> ExperimentRecord:
        key = self._key(repeat, example)
        ledger = SampleCallLedger(self.client, repeat=repeat)
        result: BaselineResult | None = None
        score_value = 0.0
        score_details: dict[str, Any] = {}
        status = "ok"
        error = None
        try:
            result = asyncio.run(self.executor(ledger, example))
            if result.method != self.method:
                raise ValueError(
                    f"executor returned method {result.method!r}, expected {self.method!r}"
                )
            score = score_baseline_result(example, result, sandbox=self.sandbox)
            score_value = score.score
            score_details = score.details
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
        responses = ledger.responses
        return ExperimentRecord(
            schema_version=1,
            key=key,
            dataset=example.dataset,
            split=example.split,
            method=self.method,
            repeat=repeat,
            sample_index=sample_index,
            sample_id=example.sample_id,
            status=status,
            score=score_value,
            score_details=score_details,
            call_count=len(responses),
            tokens=sum(response.usage.total_tokens for response in responses),
            cost=sum(response.usage.cost or 0.0 for response in responses),
            latency_seconds=sum(response.latency_seconds for response in responses),
            providers=tuple(response.provider or "unknown" for response in responses),
            result=result.to_dict() if result is not None else None,
            error=error,
        )

    def run(self) -> dict[str, Any]:
        self.store.initialize(self.spec)
        records: list[ExperimentRecord] = []
        for repeat in range(1, self.repeats + 1):
            for sample_index, example in enumerate(self.examples):
                key = self._key(repeat, example)
                record = self.store.get(key)
                if record is None:
                    record = self._run_one(repeat, sample_index, example)
                    self.store.save(record)
                records.append(record)
        summary = summarize_records(
            self.spec,
            records,
            [self.method],
            self.repeats,
        )
        self.store.save_summary(summary)
        return summary
