"""Safe planning and orchestration for the Table 1 experiment matrix."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from awo.benchmarks.data import load_and_normalize
from awo.config import config_fingerprint, load_config

DATASETS = ("hotpotqa", "drop", "humaneval", "mbpp", "gsm8k", "math")
MANUAL_METHOD_MAP = {
    "io": "io",
    "cot": "cot",
    "cot_sc": "cot_sc",
    "medprompt": "medprompt",
    "multipersona": "multi_persona",
    "multi_persona": "multi_persona",
    "self_refine": "self_refine",
}
AGENTIC_METHODS = ("adas", "aflow")
MANUAL_CALL_BOUNDS = {
    "io": (1, 1),
    "cot": (1, 1),
    "cot_sc": (6, 6),
    "medprompt": (8, 8),
    "multi_persona": (7, 7),
    "self_refine": (2, 7),
}


@dataclass(frozen=True)
class Table1Job:
    """One resumable subprocess in the registered matrix."""

    job_id: str
    phase: str
    dataset: str
    method: str
    command: tuple[str, ...]
    output_dir: str
    paid_llm_calls: bool = True

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["command"] = list(self.command)
        return value


def _jsonl(data_root: Path, dataset: str, split: str) -> Path:
    return data_root / f"{dataset}_{split}.jsonl"


def _public_tests(data_root: Path, dataset: str) -> Path | None:
    if dataset not in {"humaneval", "mbpp"}:
        return None
    return data_root / f"{dataset}_public_test.jsonl"


def _ensure_files(config: dict[str, Any], data_root: Path) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for dataset in config["datasets"]:
        if dataset not in DATASETS:
            raise ValueError(f"unsupported dataset: {dataset}")
        counts[dataset] = {}
        for split in ("validate", "test"):
            path = _jsonl(data_root, dataset, split)
            if not path.is_file():
                raise FileNotFoundError(path)
            counts[dataset][split] = len(load_and_normalize(path, dataset, split))
        public_tests = _public_tests(data_root, dataset)
        if public_tests is not None and not public_tests.is_file():
            raise FileNotFoundError(public_tests)
    return counts


def _logical_call_bounds(
    config: dict[str, Any], selected_counts: dict[str, dict[str, int]]
) -> dict[str, Any]:
    """Protocol-derived logical chat-call bounds, excluding transport retries."""

    methods = [MANUAL_METHOD_MAP.get(item, item) for item in config["methods"]]
    repeats = int(config["test_repeats"])
    manual_min = sum(MANUAL_CALL_BOUNDS[item][0] for item in methods if item in MANUAL_CALL_BOUNDS)
    manual_max = sum(MANUAL_CALL_BOUNDS[item][1] for item in methods if item in MANUAL_CALL_BOUNDS)
    result = {
        "manual_test": {"minimum": 0, "maximum": 0},
        "adas_search": {"minimum": 0, "maximum": 0},
        "adas_test": {"minimum": 0, "maximum": 0},
        "aflow_search": {"minimum": 0, "maximum": 0},
        "aflow_test": {"minimum": 0, "maximum": 0},
    }
    for dataset in config["datasets"]:
        validation = selected_counts[dataset]["validation"]
        test = selected_counts[dataset]["test"] * repeats
        result["manual_test"]["minimum"] += manual_min * test
        result["manual_test"]["maximum"] += manual_max * test
        if "adas" in methods:
            generations = int(config["adas"].get("max_rounds", 30))
            attempts = int(config["adas"].get("max_generation_attempts", 3))
            # Seven seeds contain 34 nodes in total. Every generation makes three
            # meta calls per attempt and evaluates one generated 1..12-node DAG.
            result["adas_search"]["minimum"] += 34 * validation
            result["adas_search"]["maximum"] += 34 * validation
            result["adas_search"]["minimum"] += generations * (3 + validation)
            result["adas_search"]["maximum"] += generations * (
                3 * attempts + 12 * validation
            )
            result["adas_test"]["minimum"] += test
            result["adas_test"]["maximum"] += 12 * test
        if "aflow" in methods:
            rounds = int(config["aflow"].get("max_rounds", 20))
            validation_repeats = int(config["aflow"].get("validation_rounds", 5))
            attempts = int(config["aflow"].get("max_generation_attempts", 3))
            evaluations = validation * validation_repeats
            # Initial workflow has one LLM node; generated workflows have 1..10.
            result["aflow_search"]["minimum"] += evaluations
            result["aflow_search"]["maximum"] += evaluations
            result["aflow_search"]["minimum"] += rounds * (1 + evaluations)
            result["aflow_search"]["maximum"] += rounds * (attempts + 10 * evaluations)
            result["aflow_test"]["minimum"] += test
            result["aflow_test"]["maximum"] += 10 * test
    result["all"] = {
        "minimum": sum(value["minimum"] for value in result.values()),
        "maximum": sum(value["maximum"] for value in result.values()),
    }
    return result


def build_plan(
    config_path: Path,
    data_root: Path,
    output_root: Path,
    *,
    max_concurrency: int = 1,
) -> dict[str, Any]:
    """Build a deterministic search-then-test plan without issuing LLM calls."""

    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be positive")
    config = load_config(config_path)
    counts = _ensure_files(config, data_root)
    methods = list(config["methods"])
    unknown = set(methods) - set(MANUAL_METHOD_MAP) - set(AGENTIC_METHODS)
    if unknown:
        raise ValueError(f"unsupported methods: {sorted(unknown)}")
    repeats = int(config["test_repeats"])
    limit = config.get("sample_limit")
    run_root = output_root / str(config["name"])
    jobs: list[Table1Job] = []

    for dataset in config["datasets"]:
        test_path = _jsonl(data_root, dataset, "test")
        validation_path = _jsonl(data_root, dataset, "validate")
        public_tests = _public_tests(data_root, dataset)
        manual = [MANUAL_METHOD_MAP[item] for item in methods if item in MANUAL_METHOD_MAP]
        if manual:
            command = [
                sys.executable,
                "scripts/run_manual_experiment.py",
                dataset,
                str(test_path),
                str(run_root / dataset / "test" / "manual"),
                "--methods",
                *manual,
                "--repeats",
                str(repeats),
                "--config",
                str(config_path),
            ]
            if limit is not None:
                command.extend(("--limit", str(int(limit))))
            jobs.append(
                Table1Job(
                    job_id=f"test:{dataset}:manual",
                    phase="test",
                    dataset=dataset,
                    method="manual",
                    command=tuple(command),
                    output_dir=str(run_root / dataset / "test" / "manual"),
                )
            )

        for method in AGENTIC_METHODS:
            if method not in methods:
                continue
            search_dir = run_root / dataset / "search" / method
            command = [
                sys.executable,
                "scripts/run_search.py",
                method,
                dataset,
                str(validation_path),
                str(search_dir),
                "--max-concurrency",
                str(max_concurrency),
                "--config",
                str(config_path),
            ]
            if public_tests is not None and method == "aflow":
                command.extend(("--public-tests", str(public_tests)))
            jobs.append(
                Table1Job(
                    job_id=f"search:{dataset}:{method}",
                    phase="search",
                    dataset=dataset,
                    method=method,
                    command=tuple(command),
                    output_dir=str(search_dir),
                )
            )

            test_dir = run_root / dataset / "test" / method
            command = [
                sys.executable,
                "scripts/run_agentic_experiment.py",
                method,
                dataset,
                str(test_path),
                str(test_dir),
                f"--{method}-candidate",
                str(search_dir / "best_candidate.json"),
                "--repeats",
                str(repeats),
                "--config",
                str(config_path),
            ]
            if limit is not None:
                command.extend(("--limit", str(int(limit))))
            if public_tests is not None and method == "aflow":
                command.extend(("--public-tests", str(public_tests)))
            jobs.append(
                Table1Job(
                    job_id=f"test:{dataset}:{method}",
                    phase="test",
                    dataset=dataset,
                    method=method,
                    command=tuple(command),
                    output_dir=str(test_dir),
                )
            )

    bounded = limit is not None
    selected_counts = {
        dataset: {
            "validation": (
                min(counts[dataset]["validate"], int(limit))
                if bounded
                else counts[dataset]["validate"]
            ),
            "test": (
                min(counts[dataset]["test"], int(limit))
                if bounded
                else counts[dataset]["test"]
            ),
        }
        for dataset in config["datasets"]
    }
    return {
        "schema_version": 1,
        "name": config["name"],
        "config": str(config_path.resolve()),
        "config_sha256": config_fingerprint(config),
        "data_root": str(data_root.resolve()),
        "output_root": str(run_root),
        "full_table1": not bounded,
        "sample_limit": limit,
        "test_repeats": repeats,
        "selected_counts": selected_counts,
        "logical_call_bounds": _logical_call_bounds(config, selected_counts),
        "job_count": len(jobs),
        "jobs": [job.to_dict() for job in jobs],
    }


def write_plan(plan: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def execute_plan(plan: dict[str, Any], *, phases: Sequence[str] = ("search", "test")) -> None:
    """Execute jobs in dependency order; underlying runners provide sample-level resume."""

    selected = set(phases)
    if not selected or selected - {"search", "test"}:
        raise ValueError("phases must contain search and/or test")
    for raw_job in plan["jobs"]:
        if raw_job["phase"] not in selected:
            continue
        subprocess.run(raw_job["command"], check=True)


def require_full_table_confirmation(plan: dict[str, Any], confirmed: bool) -> None:
    """Fail closed before any job when an unbounded plan lacks explicit confirmation."""

    if plan["full_table1"] and not confirmed:
        raise PermissionError("full Table 1 execution requires explicit confirmation")
