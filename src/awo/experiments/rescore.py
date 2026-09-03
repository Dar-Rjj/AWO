"""Offline rescoring for persisted code-benchmark predictions."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from awo.baselines import BaselineResult, score_baseline_result
from awo.benchmarks.data import BenchmarkExample
from awo.experiments.runner import ExperimentRecord, ExperimentStore, summarize_records
from awo.sandbox import DockerSandbox


def _load_spec(output_dir: Path) -> dict[str, Any]:
    wrapper = json.loads((output_dir / "spec.json").read_text(encoding="utf-8"))
    spec = {key: value for key, value in wrapper.items() if key != "sha256"}
    ExperimentStore(output_dir).initialize(spec)
    return spec


def _stored_result(record: ExperimentRecord) -> BaselineResult:
    if record.result is None:
        raise ValueError(f"record {record.key} has no persisted baseline result")
    value = record.result
    return BaselineResult(
        method=str(value["method"]),
        dataset=str(value["dataset"]),
        sample_id=str(value["sample_id"]),
        prediction=str(value["prediction"]),
        prompt_sha256=str(value["prompt_sha256"]),
        responses=(),
        protocol=str(value.get("protocol", "paper-faithful")),
        artifacts=dict(value.get("artifacts", {})),
    )


def rescore_code_records(
    output_dir: Path,
    examples: Sequence[BenchmarkExample],
    *,
    sandbox: DockerSandbox,
) -> dict[str, Any]:
    """Replace only prior sandbox errors, without issuing any model calls."""

    spec = _load_spec(output_dir)
    dataset = str(spec["dataset"])
    if dataset not in {"humaneval", "mbpp"}:
        raise ValueError("offline code rescoring supports only HumanEval and MBPP")
    if str(spec["split"]) != "test":
        raise ValueError("offline code rescoring requires a test split")

    by_id = {example.sample_id: example for example in examples}
    expected_ids = [str(item) for item in spec["sample_ids"]]
    if list(by_id) != expected_ids:
        raise ValueError("dataset examples do not match the persisted sample order")

    store = ExperimentStore(output_dir)
    methods = [str(item) for item in spec["methods"]]
    repeats = int(spec["repeats"])
    records: list[ExperimentRecord] = []
    rescored = 0
    skipped = 0

    for repeat in range(1, repeats + 1):
        for method in methods:
            for sample_id in expected_ids:
                key = f"{dataset}:test:{method}:{repeat}:{sample_id}"
                record = store.get(key)
                if record is None:
                    raise ValueError(f"missing persisted record: {key}")
                prior_status = record.score_details.get("sandbox_status")
                if record.status != "ok" or prior_status != "sandbox_error":
                    skipped += 1
                    records.append(record)
                    continue

                score = score_baseline_result(
                    by_id[sample_id],
                    _stored_result(record),
                    sandbox=sandbox,
                )
                if score.details.get("sandbox_status") == "sandbox_error":
                    raise RuntimeError(
                        f"sandbox remains unavailable while rescoring {record.key}: "
                        f"{score.details.get('stderr', '')}"
                    )
                updated = replace(
                    record,
                    score=score.score,
                    score_details=score.details,
                )
                store.save(updated)
                records.append(updated)
                rescored += 1

    summary = summarize_records(spec, records, methods, repeats)
    store.save_summary(summary)
    return {
        "dataset": dataset,
        "output_dir": str(output_dir),
        "rescored": rescored,
        "skipped": skipped,
        "by_run": summary["by_run"],
        "totals": summary["totals"],
    }
