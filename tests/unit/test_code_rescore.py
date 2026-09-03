from __future__ import annotations

from pathlib import Path

from awo.baselines import BaselineResult
from awo.benchmarks.data import BenchmarkExample
from awo.experiments.rescore import rescore_code_records
from awo.experiments.runner import ExperimentRecord, ExperimentStore
from awo.sandbox import PASS_MARKER, SandboxResult


class PassingSandbox:
    def run(self, source: str) -> SandboxResult:
        assert "def add_one" in source
        return SandboxResult("passed", 0, PASS_MARKER, "", 0.01, "test-image")


def test_rescore_reuses_prediction_and_updates_integrity_wrapper(tmp_path: Path) -> None:
    example = BenchmarkExample(
        1,
        "humaneval",
        "test",
        "HumanEval/0",
        "def add_one(value):",
        "    return value + 1",
        entry_point="add_one",
        test_code="def check(candidate):\n    assert candidate(1) == 2",
    )
    spec = {
        "schema_version": 1,
        "dataset": "humaneval",
        "split": "test",
        "dataset_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "implementation_commit": "commit-a",
        "sample_ids": [example.sample_id],
        "methods": ["io"],
        "repeats": 1,
        "seed": 42,
        "cache_across_repeats": False,
    }
    result = BaselineResult(
        method="io",
        dataset="humaneval",
        sample_id=example.sample_id,
        prediction="def add_one(value):\n    return value + 1",
        prompt_sha256="c" * 64,
        responses=(),
    )
    record = ExperimentRecord(
        schema_version=1,
        key="humaneval:test:io:1:HumanEval/0",
        dataset="humaneval",
        split="test",
        method="io",
        repeat=1,
        sample_index=0,
        sample_id=example.sample_id,
        status="ok",
        score=0.0,
        score_details={"sandbox_status": "sandbox_error"},
        call_count=1,
        tokens=10,
        cost=0.001,
        latency_seconds=0.1,
        providers=("test",),
        result=result.to_dict(),
        error=None,
    )
    store = ExperimentStore(tmp_path)
    store.initialize(spec)
    store.save(record)

    report = rescore_code_records(
        tmp_path,
        [example],
        sandbox=PassingSandbox(),  # type: ignore[arg-type]
    )

    assert report["rescored"] == 1
    assert report["totals"] == {
        "failures": 0,
        "calls": 1,
        "tokens": 10,
        "cost": 0.001,
    }
    updated = store.get(record.key)
    assert updated is not None
    assert updated.score == 1.0
    assert updated.score_details["sandbox_status"] == "passed"

    resumed = rescore_code_records(
        tmp_path,
        [example],
        sandbox=PassingSandbox(),  # type: ignore[arg-type]
    )
    assert resumed["rescored"] == 0
    assert resumed["skipped"] == 1
