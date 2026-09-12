#!/usr/bin/env python3
"""Test the frozen best upstream-Python AFlow HumanEval workflow once."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from awo.aflow.official import OfficialPythonCandidate, OfficialSandboxRuntime
from awo.benchmarks.code import score_code
from awo.benchmarks.data import BenchmarkExample, load_and_normalize
from awo.config import load_config
from awo.llm import JsonlRequestRecorder, client_from_config
from awo.sandbox import create_sandbox
from awo.tracking import build_manifest, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "search_dir",
        type=Path,
        nargs="?",
        default=Path(
            "experiments/runs/aflow-humaneval-claude35-gpt4omini-upstream-optimizer/search"
        ),
    )
    parser.add_argument(
        "test_jsonl",
        type=Path,
        nargs="?",
        default=Path("data/raw/datasets/humaneval_test.jsonl"),
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        default=Path(
            "experiments/runs/aflow-humaneval-claude35-gpt4omini-upstream-optimizer/test"
        ),
    )
    parser.add_argument(
        "--humaneval-public-tests",
        type=Path,
        default=Path("data/raw/datasets/humaneval_public_test.jsonl"),
    )
    parser.add_argument(
        "--mbpp-public-tests",
        type=Path,
        default=Path("data/raw/datasets/mbpp_public_test.jsonl"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/aflow/original_humaneval.yaml"),
    )
    parser.add_argument("--max-concurrency", type=int, default=8)
    return parser.parse_args()


def load_public_tests(path: Path) -> dict[str, Any]:
    tests: dict[str, Any] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            entry_point = item.get("entry_point")
            assertions = item.get("test")
            if not isinstance(entry_point, str) or not isinstance(assertions, list):
                raise ValueError(f"invalid public tests at {path}:{line_number}")
            tests[entry_point] = assertions
    return tests


async def evaluate_one(
    *,
    index: int,
    example: BenchmarkExample,
    candidate: OfficialPythonCandidate,
    best_round: int,
    runtime: OfficialSandboxRuntime,
    semaphore: asyncio.Semaphore,
) -> tuple[int, dict[str, Any]]:
    async with semaphore:
        try:
            assert example.entry_point is not None
            assert example.test_code is not None
            execution = await asyncio.to_thread(
                runtime.execute,
                candidate,
                round_number=best_round,
                problem=example.prompt,
                entry_point=example.entry_point,
                metadata={"sample_id": example.sample_id, "split": "test"},
            )
            score = await asyncio.to_thread(
                score_code,
                "humaneval",
                execution.prediction,
                tests=example.test_code,
                sandbox=runtime.sandbox,
                entry_point=example.entry_point,
            )
            return index, {
                "sample_id": example.sample_id,
                "prediction": execution.prediction,
                "score": score.score,
                "cost": execution.relay_cost,
                "tokens": execution.tokens,
                "calls": execution.calls,
                "sandbox_runs": execution.sandbox_runs,
                "score_details": score.details,
                "error": None,
            }
        except Exception as error:
            return index, {
                "sample_id": example.sample_id,
                "prediction": "",
                "score": 0.0,
                "cost": 0.0,
                "tokens": 0,
                "calls": 0,
                "sandbox_runs": 0,
                "score_details": {},
                "error": f"{type(error).__name__}: {error}",
            }


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


async def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_concurrency <= 0:
        raise ValueError("--max-concurrency must be positive")
    search_summary = json.loads((args.search_dir / "summary.json").read_text())
    best_round = int(search_summary["best_round"])
    candidate = OfficialPythonCandidate.from_dict(
        json.loads((args.search_dir / "best_candidate.json").read_text())
    )
    examples = load_and_normalize(args.test_jsonl, "humaneval", "test")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    if not manifest_path.exists():
        write_manifest(build_manifest(args.config), manifest_path)
    client = client_from_config(
        load_config(args.config),
        recorder=JsonlRequestRecorder(args.output_dir / "requests.jsonl"),
    )
    runtime = OfficialSandboxRuntime(
        client,
        create_sandbox(),
        humaneval_public_tests=load_public_tests(args.humaneval_public_tests),
        mbpp_public_tests=load_public_tests(args.mbpp_public_tests),
    )
    semaphore = asyncio.Semaphore(args.max_concurrency)
    tasks = [
        asyncio.create_task(
            evaluate_one(
                index=index,
                example=example,
                candidate=candidate,
                best_round=best_round,
                runtime=runtime,
                semaphore=semaphore,
            )
        )
        for index, example in enumerate(examples)
    ]
    rows: dict[int, dict[str, Any]] = {}
    for task in asyncio.as_completed(tasks):
        index, row = await task
        rows[index] = row
        ordered = [rows[key] for key in sorted(rows)]
        write_json(
            args.output_dir / "progress.json",
            {
                "completed": len(ordered),
                "total": len(examples),
                "passed": sum(int(item["score"] == 1.0) for item in ordered),
                "failed": sum(int(item["score"] == 0.0) for item in ordered),
                "cost": sum(float(item["cost"]) for item in ordered),
                "tokens": sum(int(item["tokens"]) for item in ordered),
            },
        )
    ordered = [rows[key] for key in sorted(rows)]
    (args.output_dir / "results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in ordered),
        encoding="utf-8",
    )
    summary = {
        "dataset": "humaneval",
        "split": "test",
        "test_repeats": 1,
        "sample_count": len(ordered),
        "best_round": best_round,
        "pass_count": sum(int(row["score"] == 1.0) for row in ordered),
        "pass_at_1": sum(float(row["score"]) for row in ordered) / len(ordered),
        "total_cost": sum(float(row["cost"]) for row in ordered),
        "total_tokens": sum(int(row["tokens"]) for row in ordered),
        "total_calls": sum(int(row["calls"]) for row in ordered),
        "errors": sum(int(row["error"] is not None) for row in ordered),
    }
    write_json(args.output_dir / "summary.json", summary)
    return summary


def main() -> int:
    print(json.dumps(asyncio.run(run(parse_args())), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
