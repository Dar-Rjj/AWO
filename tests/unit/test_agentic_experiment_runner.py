from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from awo.aflow import WorkflowCandidate
from awo.baselines.models import BaselineResult
from awo.benchmarks.data import BenchmarkExample
from awo.experiments import AgenticExperimentRunner, aflow_candidate_executor
from awo.llm import ChatResult, TokenUsage


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    def chat(self, messages: Any, *, metadata: Any = None, **_: Any) -> ChatResult:
        self.calls.append((messages, metadata))
        return ChatResult(
            request_id=str(len(self.calls)),
            response_id=str(len(self.calls)),
            requested_model="deepseek/deepseek-chat",
            actual_model="deepseek/deepseek-chat",
            provider="test",
            content='{"answer":"36"}',
            finish_reason="stop",
            usage=TokenUsage(total_tokens=12, cost=0.002),
            attempts=1,
            latency_seconds=0.02,
            request_sha256="a" * 64,
        )


def example() -> BenchmarkExample:
    return BenchmarkExample(1, "gsm8k", "test", "sample-1", "6 squared?", "36")


async def successful_executor(client: Any, item: BenchmarkExample) -> BaselineResult:
    response = client.chat(
        [{"role": "user", "content": item.prompt}],
        metadata={"method": "adas"},
    )
    return BaselineResult(
        method="adas",
        dataset=item.dataset,
        sample_id=item.sample_id,
        prediction="36",
        prompt_sha256="b" * 64,
        responses=(response,),
    )


def make_runner(tmp_path: Path, client: FakeClient) -> AgenticExperimentRunner:
    return AgenticExperimentRunner(
        client=client,  # type: ignore[arg-type]
        executor=successful_executor,
        method="adas",
        executor_fingerprint={"architecture_sha256": "c" * 64},
        examples=[example()],
        repeats=2,
        output_dir=tmp_path,
        dataset_sha256="d" * 64,
        config_sha256="e" * 64,
        implementation_commit="commit-a",
    )


def test_agentic_runner_records_and_resumes(tmp_path: Path) -> None:
    client = FakeClient()
    summary = make_runner(tmp_path, client).run()
    assert summary["completed_records"] == 2
    assert summary["totals"] == {
        "failures": 0,
        "calls": 2,
        "tokens": 24,
        "cost": 0.004,
    }
    assert [call[1]["experiment_repeat"] for call in client.calls] == [1, 2]

    resumed_client = FakeClient()
    assert make_runner(tmp_path, resumed_client).run() == summary
    assert resumed_client.calls == []


def test_agentic_runner_keeps_call_cost_when_executor_parse_fails(tmp_path: Path) -> None:
    async def failing_executor(client: Any, item: BenchmarkExample) -> BaselineResult:
        client.chat([{"role": "user", "content": item.prompt}])
        raise ValueError("invalid structured response")

    summary = AgenticExperimentRunner(
        client=FakeClient(),  # type: ignore[arg-type]
        executor=failing_executor,
        method="adas",
        executor_fingerprint={"architecture_sha256": "f" * 64},
        examples=[example()],
        repeats=1,
        output_dir=tmp_path,
        dataset_sha256="1" * 64,
        config_sha256="2" * 64,
        implementation_commit="commit-b",
    ).run()
    assert summary["totals"] == {
        "failures": 1,
        "calls": 1,
        "tokens": 12,
        "cost": 0.002,
    }


def test_agentic_resume_rejects_changed_executor(tmp_path: Path) -> None:
    make_runner(tmp_path, FakeClient()).run()
    runner = AgenticExperimentRunner(
        client=FakeClient(),  # type: ignore[arg-type]
        executor=successful_executor,
        method="adas",
        executor_fingerprint={"architecture_sha256": "0" * 64},
        examples=[example()],
        repeats=2,
        output_dir=tmp_path,
        dataset_sha256="d" * 64,
        config_sha256="e" * 64,
        implementation_commit="commit-a",
    )
    with pytest.raises(ValueError, match="spec"):
        runner.run()


def test_searched_aflow_test_node_requires_sample_public_tests() -> None:
    candidate = WorkflowCandidate(
        modification="test generated code",
        graph={
            "schema_version": 1,
            "nodes": [
                {
                    "id": "test",
                    "operator": "Test",
                    "inputs": {
                        "problem": "$problem",
                        "solution": "def solve(): return 1",
                        "entry_point": "$entry_point",
                    },
                }
            ],
            "output": "$test.solution",
        },
    )
    client = FakeClient()
    item = BenchmarkExample(
        1,
        "humaneval",
        "test",
        "HumanEval/missing",
        "problem",
        "tests",
        entry_point="solve",
    )
    with pytest.raises(ValueError, match="no frozen public tests"):
        asyncio.run(aflow_candidate_executor(candidate, {}, None)(client, item))
    assert client.calls == []
