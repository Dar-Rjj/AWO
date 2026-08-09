"""Normalize the frozen AFlow JSONL files into one explicit schema."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from awo.artifacts import sha256_file

DATASETS = ("hotpotqa", "drop", "humaneval", "mbpp", "gsm8k", "math")
SPLITS = ("validate", "test")


class BenchmarkDataError(ValueError):
    """Raised when a frozen dataset record does not match its expected schema."""


@dataclass(frozen=True)
class BenchmarkExample:
    schema_version: int
    dataset: str
    split: str
    sample_id: str
    prompt: str
    reference: Any
    entry_point: str | None = None
    test_code: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require(record: dict[str, Any], keys: Iterable[str], source: str) -> None:
    missing = [key for key in keys if key not in record]
    if missing:
        raise BenchmarkDataError(f"{source} is missing required fields: {missing}")


def _hotpot_prompt(record: dict[str, Any]) -> str:
    paragraphs = [item[1] for item in record["context"] if isinstance(item[1], list)]
    context = "\n".join(" ".join(paragraph) for paragraph in paragraphs)
    return f"Context: {context}\n\nQuestion: {record['question']}\n\nAnswer:"


def normalize_record(
    dataset: str, split: str, index: int, record: dict[str, Any]
) -> BenchmarkExample:
    source = f"{dataset}_{split}.jsonl line {index + 1}"
    if dataset == "hotpotqa":
        _require(record, ("_id", "question", "answer", "context"), source)
        return BenchmarkExample(
            1,
            dataset,
            split,
            str(record["_id"]),
            _hotpot_prompt(record),
            record["answer"],
            metadata={
                "question": record["question"],
                "context": record["context"],
                "supporting_facts": record.get("supporting_facts", []),
                "level": record.get("level"),
                "type": record.get("type"),
            },
        )
    if dataset == "drop":
        _require(record, ("id", "context", "ref_text"), source)
        return BenchmarkExample(
            1,
            dataset,
            split,
            str(record["id"]),
            record["context"],
            record["ref_text"],
            metadata={"completion": record.get("completion")},
        )
    if dataset == "gsm8k":
        _require(record, ("id", "question", "answer"), source)
        return BenchmarkExample(
            1,
            dataset,
            split,
            str(record["id"]),
            record["question"],
            record["answer"],
            metadata={"cot": record.get("cot")},
        )
    if dataset == "math":
        _require(record, ("problem", "solution", "level", "type"), source)
        return BenchmarkExample(
            1,
            dataset,
            split,
            f"math-{split}-{index:04d}",
            record["problem"],
            record["solution"],
            metadata={"level": record["level"], "type": record["type"]},
        )
    if dataset == "humaneval":
        _require(
            record,
            ("task_id", "prompt", "canonical_solution", "test", "entry_point"),
            source,
        )
        return BenchmarkExample(
            1,
            dataset,
            split,
            str(record["task_id"]),
            record["prompt"],
            record["canonical_solution"],
            entry_point=record["entry_point"],
            test_code=record["test"],
        )
    if dataset == "mbpp":
        _require(record, ("task_id", "prompt", "code", "test", "entry_point"), source)
        return BenchmarkExample(
            1,
            dataset,
            split,
            f"mbpp-{record['task_id']}",
            record["prompt"],
            record["code"],
            entry_point=record["entry_point"],
            test_code=record["test"],
            metadata={
                "test_imports": record.get("test_imports", []),
                "test_list": record.get("test_list", []),
            },
        )
    raise BenchmarkDataError(f"Unknown dataset: {dataset}")


def load_and_normalize(path: Path, dataset: str, split: str) -> list[BenchmarkExample]:
    examples = []
    seen_ids = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BenchmarkDataError(f"Invalid JSON at {path}:{index + 1}") from exc
            if not isinstance(record, dict):
                raise BenchmarkDataError(f"Expected object at {path}:{index + 1}")
            example = normalize_record(dataset, split, index, record)
            if example.sample_id in seen_ids:
                raise BenchmarkDataError(f"Duplicate sample_id in {path}: {example.sample_id}")
            seen_ids.add(example.sample_id)
            examples.append(example)
    return examples


def _write_jsonl(examples: list[BenchmarkExample], output: Path) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with output.open("w", encoding="utf-8") as handle:
        for example in examples:
            line = json.dumps(
                example.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            encoded = (line + "\n").encode("utf-8")
            handle.write(encoded.decode("utf-8"))
            digest.update(encoded)
    return digest.hexdigest()


def prepare_all_datasets(raw_dir: Path, output_dir: Path) -> dict[str, Any]:
    files = []
    for dataset in DATASETS:
        for split in SPLITS:
            source = Path(raw_dir) / f"{dataset}_{split}.jsonl"
            examples = load_and_normalize(source, dataset, split)
            destination = Path(output_dir) / f"{dataset}_{split}.jsonl"
            output_hash = _write_jsonl(examples, destination)
            files.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "records": len(examples),
                    "source": str(source),
                    "source_sha256": sha256_file(source),
                    "output": str(destination),
                    "output_sha256": output_hash,
                }
            )
    manifest = {"schema_version": 1, "files": files}
    manifest_path = Path(output_dir) / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest
