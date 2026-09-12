#!/usr/bin/env python3
"""Run the pinned upstream AFlow Python search on HumanEval via the sandbox relay."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from awo.aflow.official import OfficialSandboxRuntime
from awo.aflow.official_search import OfficialAFlowSearch, OfficialSearchConfig
from awo.benchmarks.data import load_and_normalize
from awo.config import load_config
from awo.llm import JsonlRequestRecorder, client_from_config
from awo.sandbox import create_sandbox
from awo.tracking import build_manifest, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "validation_jsonl",
        type=Path,
        nargs="?",
        default=Path("data/raw/datasets/humaneval_validate.jsonl"),
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        default=Path(
            "experiments/runs/aflow-humaneval-gpt-4o-mini-upstream-optimizer/search"
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
        help="Loaded because the pinned HumanEval template literally requests the MBPP file.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/aflow/original_humaneval.yaml"),
    )
    parser.add_argument("--max-rounds", type=int)
    parser.add_argument("--initial-round", type=int)
    parser.add_argument("--validation-rounds", type=int)
    parser.add_argument("--max-concurrency", type=int)
    parser.add_argument("--limit", type=int)
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
            if not all(isinstance(assertion, str) for assertion in assertions):
                raise ValueError(f"invalid assertion at {path}:{line_number}")
            tests[entry_point] = assertions
    return tests


async def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    aflow = dict(config["aflow"])
    examples = load_and_normalize(args.validation_jsonl, "humaneval", "validate")
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        examples = examples[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    if not manifest_path.exists():
        write_manifest(build_manifest(args.config), manifest_path)
    client = client_from_config(
        config,
        recorder=JsonlRequestRecorder(args.output_dir / "requests.jsonl"),
    )
    sandbox = create_sandbox()
    runtime = OfficialSandboxRuntime(
        client,
        sandbox,
        humaneval_public_tests=load_public_tests(args.humaneval_public_tests),
        mbpp_public_tests=load_public_tests(args.mbpp_public_tests),
    )
    search_config = OfficialSearchConfig(
        sample=int(aflow.get("sample", 4)),
        initial_round=(
            args.initial_round
            if args.initial_round is not None
            else int(aflow.get("initial_round", 1))
        ),
        max_rounds=(
            args.max_rounds if args.max_rounds is not None else int(aflow.get("max_rounds", 20))
        ),
        validation_rounds=(
            args.validation_rounds
            if args.validation_rounds is not None
            else int(aflow.get("validation_rounds", 5))
        ),
        check_convergence=bool(aflow.get("check_convergence", True)),
        max_concurrency=(
            args.max_concurrency
            if args.max_concurrency is not None
            else int(aflow.get("max_concurrency", 8))
        ),
    )
    summary = await OfficialAFlowSearch(
        client=client,
        runtime=runtime,
        examples=examples,
        output_dir=args.output_dir,
        config=search_config,
    ).run()
    return {
        "dataset": "humaneval",
        "validation_sample_count": len(examples),
        "summary": summary.__dict__,
    }


def main() -> int:
    result = asyncio.run(run(parse_args()))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
