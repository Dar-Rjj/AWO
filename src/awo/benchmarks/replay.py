"""Re-score archived AFlow CSV files with the controlled evaluators."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from awo.benchmarks.code import score_code
from awo.benchmarks.data import load_and_normalize
from awo.benchmarks.scoring import score_prediction
from awo.sandbox import DockerSandbox


def detect_header_row(path: Path) -> int:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for index in range(10):
            line = handle.readline()
            if not line:
                break
            fields = {field.strip() for field in line.strip().split(",")}
            if "score" in fields and "prediction" in fields:
                return index
    raise ValueError(f"Could not find result CSV header in first 10 lines: {path}")


def load_result_csv(path: Path) -> pd.DataFrame:
    header_row = detect_header_row(path)
    frame = pd.read_csv(path, skiprows=header_row)
    required = {"prediction", "expected_output", "score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Result CSV {path} is missing columns: {sorted(missing)}")
    return frame


def replay_result_csv(
    dataset: str, path: Path, *, math_mode: str = "archive"
) -> dict[str, Any]:
    frame = load_result_csv(path)
    replayed = []
    extracted = []
    for row in frame.itertuples(index=False):
        result = score_prediction(
            dataset, row.expected_output, row.prediction, math_mode=math_mode
        )
        replayed.append(result.score)
        extracted.append(result)
    stored = frame["score"].astype(float)
    differences = [abs(float(old) - new) for old, new in zip(stored, replayed)]
    mismatches = sum(difference > 1e-9 for difference in differences)
    mismatch_examples = []
    for index, (old, new, difference, result) in enumerate(
        zip(stored, replayed, differences, extracted)
    ):
        if difference <= 1e-9:
            continue
        mismatch_examples.append(
            {
                "row_index": index,
                "stored_score": float(old),
                "replayed_score": new,
                "prediction_extracted": str(result.extracted),
                "expected_extracted": str(result.details.get("expected_extracted")),
            }
        )
        if len(mismatch_examples) == 20:
            break
    return {
        "dataset": dataset,
        "path": str(path),
        "records": len(frame),
        "stored_mean": float(stored.mean()),
        "replayed_mean": sum(replayed) / len(replayed) if replayed else 0.0,
        "mismatches": mismatches,
        "max_absolute_difference": max(differences, default=0.0),
        "math_mode": math_mode if dataset.lower() == "math" else None,
        "mismatch_examples": mismatch_examples,
    }


def replay_code_result_csv(
    dataset: str,
    path: Path,
    dataset_jsonl: Path,
    *,
    sandbox: DockerSandbox,
) -> dict[str, Any]:
    """Join archived predictions to frozen tests by prompt and execute them in Docker."""

    normalized = dataset.lower()
    if normalized not in {"humaneval", "mbpp"}:
        raise ValueError(f"unsupported code benchmark: {dataset}")
    frame = load_result_csv(path)
    examples = load_and_normalize(dataset_jsonl, normalized, "test")
    by_prompt = {example.prompt.strip(): example for example in examples}
    if len(by_prompt) != len(examples):
        raise ValueError(f"{dataset_jsonl} contains duplicate normalized prompts")
    if len(frame) != len(examples):
        raise ValueError(
            f"archive/data row-count mismatch: {len(frame)} predictions != {len(examples)} tests"
        )

    replayed = []
    statuses: Counter[str] = Counter()
    mismatch_examples = []
    seen_prompts: set[str] = set()
    for index, row in enumerate(frame.itertuples(index=False)):
        prompt = str(row.question).strip()
        if prompt in seen_prompts:
            raise ValueError(f"duplicate archived question at row {index}")
        seen_prompts.add(prompt)
        try:
            example = by_prompt[prompt]
        except KeyError as error:
            raise ValueError(f"archived question at row {index} is absent from dataset") from error
        if example.test_code is None:
            raise ValueError(f"missing test code for {example.sample_id}")
        result = score_code(
            dataset,
            row.prediction,
            tests=example.test_code,
            entry_point=example.entry_point,
            sandbox=sandbox,
        )
        replayed.append(result.score)
        status = str(result.details["sandbox_status"])
        statuses[status] += 1
        stored_score = float(row.score)
        if abs(stored_score - result.score) > 1e-9 and len(mismatch_examples) < 20:
            mismatch_examples.append(
                {
                    "row_index": index,
                    "sample_id": example.sample_id,
                    "stored_score": stored_score,
                    "replayed_score": result.score,
                    "sandbox_status": status,
                    "stderr": result.details.get("stderr", ""),
                }
            )

    stored = frame["score"].astype(float).tolist()
    differences = [abs(old - new) for old, new in zip(stored, replayed)]
    return {
        "dataset": dataset,
        "path": str(path),
        "dataset_jsonl": str(dataset_jsonl),
        "records": len(frame),
        "stored_mean": sum(stored) / len(stored) if stored else 0.0,
        "replayed_mean": sum(replayed) / len(replayed) if replayed else 0.0,
        "mismatches": sum(value > 1e-9 for value in differences),
        "max_absolute_difference": max(differences, default=0.0),
        "sandbox_statuses": dict(sorted(statuses.items())),
        "sandbox_image": sandbox.config.image,
        "mismatch_examples": mismatch_examples,
    }
