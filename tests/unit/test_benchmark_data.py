import json
from pathlib import Path

import pytest

from awo.benchmarks.data import BenchmarkDataError, load_and_normalize


def write_jsonl(path: Path, records) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


def test_normalizes_hotpot_prompt(tmp_path: Path) -> None:
    path = tmp_path / "hotpotqa_test.jsonl"
    write_jsonl(
        path,
        [
            {
                "_id": "sample",
                "question": "Who?",
                "answer": "Ada",
                "context": [["title", ["First.", "Second."]]],
            }
        ],
    )

    example = load_and_normalize(path, "hotpotqa", "test")[0]

    assert example.sample_id == "sample"
    assert example.prompt == "Context: First. Second.\n\nQuestion: Who?\n\nAnswer:"
    assert example.reference == "Ada"


def test_normalizes_code_task(tmp_path: Path) -> None:
    path = tmp_path / "humaneval_validate.jsonl"
    write_jsonl(
        path,
        [
            {
                "task_id": "HumanEval/0",
                "prompt": "def f(x):",
                "canonical_solution": " return x",
                "test": "def check(candidate): assert candidate(1) == 1",
                "entry_point": "f",
            }
        ],
    )

    example = load_and_normalize(path, "humaneval", "validate")[0]

    assert example.entry_point == "f"
    assert "check" in example.test_code


def test_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "gsm8k_test.jsonl"
    record = {"id": "same", "question": "q", "answer": "1"}
    write_jsonl(path, [record, record])

    with pytest.raises(BenchmarkDataError, match="Duplicate sample_id"):
        load_and_normalize(path, "gsm8k", "test")
