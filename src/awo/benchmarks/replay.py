"""Re-score archived AFlow CSV files with the controlled evaluators."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from awo.benchmarks.scoring import score_prediction


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
