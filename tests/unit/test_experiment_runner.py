from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from awo.benchmarks.data import BenchmarkExample
from awo.experiments import ManualExperimentRunner
from awo.llm import ChatResult, TokenUsage


class FakeClient:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[Any, Any]] = []

    def chat(
        self,
        messages: Any,
        *,
        temperature: Any = None,
        max_tokens: Any = None,
        metadata: Any = None,
    ) -> ChatResult:
        self.calls.append((messages, metadata))
        if self.fail:
            raise RuntimeError("request failed")
        return ChatResult(
            request_id=str(len(self.calls)),
            response_id=str(len(self.calls)),
            requested_model="deepseek/deepseek-chat",
            actual_model="deepseek/deepseek-chat",
            provider="test",
            content='{"solution":"36","answer":"36"}',
            finish_reason="stop",
            usage=TokenUsage(total_tokens=10, cost=0.001),
            attempts=1,
            latency_seconds=0.01,
            request_sha256="a" * 64,
        )


def _examples() -> list[BenchmarkExample]:
    return [
        BenchmarkExample(1, "gsm8k", "test", f"sample-{index}", "6 squared?", "36")
        for index in range(2)
    ]


def test_runner_persists_each_sample_and_resumes_without_calls(tmp_path: Path) -> None:
    client = FakeClient()
    runner = ManualExperimentRunner(
        client=client,  # type: ignore[arg-type]
        examples=_examples(),
        methods=["io"],
        repeats=2,
        output_dir=tmp_path,
        dataset_sha256="a" * 64,
        config_sha256="1" * 64,
        implementation_commit="commit-a",
    )
    summary = runner.run()
    assert summary["completed_records"] == 4
    assert summary["totals"] == {
        "failures": 0,
        "calls": 4,
        "tokens": 40,
        "cost": 0.004,
    }
    assert all(call[1]["experiment_repeat"] in {1, 2} for call in client.calls)
    record_count = len(list((tmp_path / "records").glob("*.json")))
    assert record_count == 4

    resumed_client = FakeClient()
    resumed = ManualExperimentRunner(
        client=resumed_client,  # type: ignore[arg-type]
        examples=_examples(),
        methods=["io"],
        repeats=2,
        output_dir=tmp_path,
        dataset_sha256="a" * 64,
        config_sha256="1" * 64,
        implementation_commit="commit-a",
    ).run()
    assert resumed == summary
    assert resumed_client.calls == []


def test_runner_records_request_failure_as_zero(tmp_path: Path) -> None:
    summary = ManualExperimentRunner(
        client=FakeClient(fail=True),  # type: ignore[arg-type]
        examples=_examples()[:1],
        methods=["io"],
        repeats=1,
        output_dir=tmp_path,
        dataset_sha256="b" * 64,
        config_sha256="2" * 64,
        implementation_commit="commit-b",
    ).run()
    assert summary["totals"]["failures"] == 1
    assert summary["by_run"][0]["score"] == 0.0


def test_resume_rejects_changed_sample_slice(tmp_path: Path) -> None:
    ManualExperimentRunner(
        client=FakeClient(),  # type: ignore[arg-type]
        examples=_examples()[:1],
        methods=["io"],
        repeats=1,
        output_dir=tmp_path,
        dataset_sha256="c" * 64,
        config_sha256="3" * 64,
        implementation_commit="commit-c",
    ).run()
    with pytest.raises(ValueError, match="spec"):
        ManualExperimentRunner(
            client=FakeClient(),  # type: ignore[arg-type]
            examples=_examples(),
            methods=["io"],
            repeats=1,
            output_dir=tmp_path,
            dataset_sha256="c" * 64,
            config_sha256="3" * 64,
            implementation_commit="commit-c",
        ).run()


def test_resume_rejects_tampered_record(tmp_path: Path) -> None:
    ManualExperimentRunner(
        client=FakeClient(),  # type: ignore[arg-type]
        examples=_examples()[:1],
        methods=["io"],
        repeats=1,
        output_dir=tmp_path,
        dataset_sha256="d" * 64,
        config_sha256="4" * 64,
        implementation_commit="commit-d",
    ).run()
    record_path = next((tmp_path / "records").glob("*.json"))
    wrapper = json.loads(record_path.read_text(encoding="utf-8"))
    wrapper["record"]["score"] = 0.25
    record_path.write_text(json.dumps(wrapper), encoding="utf-8")

    with pytest.raises(ValueError, match="integrity"):
        ManualExperimentRunner(
            client=FakeClient(),  # type: ignore[arg-type]
            examples=_examples()[:1],
            methods=["io"],
            repeats=1,
            output_dir=tmp_path,
            dataset_sha256="d" * 64,
            config_sha256="4" * 64,
            implementation_commit="commit-d",
        ).run()
