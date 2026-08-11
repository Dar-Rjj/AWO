#!/usr/bin/env python3
"""Run and resume validation-only AFlow or ADAS search, then freeze the best candidate."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from awo.adas import (
    ADASConfig,
    ADASMetaGenerator,
    ADASSearch,
    ADASValidationEvaluator,
)
from awo.aflow import (
    AFlowSearch,
    AFlowValidationEvaluator,
    OpenRouterCandidateGenerator,
    SearchConfig,
    load_public_tests,
)
from awo.artifacts import sha256_file
from awo.benchmarks.data import load_and_normalize
from awo.config import config_fingerprint, load_config
from awo.llm import JsonlRequestRecorder, client_from_config
from awo.sandbox import DockerSandbox
from awo.tracking import build_manifest, write_manifest

METHODS = ("aflow", "adas")
DATASETS = ("hotpotqa", "drop", "humaneval", "mbpp", "gsm8k", "math")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=METHODS)
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("validation_jsonl", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--public-tests", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.yaml"))
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.max_concurrency <= 0:
        parser.error("--max-concurrency must be positive")
    if args.method == "aflow" and args.dataset in {"humaneval", "mbpp"}:
        if args.public_tests is None:
            parser.error("--public-tests is required for AFlow code search")
    elif args.public_tests is not None:
        parser.error("--public-tests is only valid for AFlow code search")
    return args


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


async def run() -> dict[str, Any]:
    args = parse_args()
    config = load_config(args.config)
    all_examples = load_and_normalize(
        args.validation_jsonl,
        args.dataset,
        "validate",
    )
    configured_limit = config.get("sample_limit")
    limit = args.limit if args.limit is not None else configured_limit
    selected = all_examples if limit is None else all_examples[: int(limit)]
    if not selected:
        raise ValueError("validation slice cannot be empty")

    public_tests = (
        load_public_tests(args.public_tests, all_examples) if args.public_tests is not None else {}
    )
    sandbox_datasets = {"humaneval", "mbpp"}
    if args.method == "aflow":
        sandbox_datasets.update({"gsm8k", "math"})
    sandbox = DockerSandbox() if args.dataset in sandbox_datasets else None

    manifest = build_manifest(args.config)
    run_fingerprint = {
        "dataset_sha256": sha256_file(args.validation_jsonl),
        "config_sha256": config_fingerprint(config),
        "implementation_commit": manifest["repository"]["commit"],
        "validation_sample_ids": [example.sample_id for example in selected],
        "public_tests_sha256": (
            sha256_file(args.public_tests) if args.public_tests is not None else None
        ),
        "cache_across_repeats": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    if not manifest_path.exists():
        write_manifest(manifest, manifest_path)
    client = client_from_config(
        config,
        recorder=JsonlRequestRecorder(args.output_dir / "requests.jsonl"),
    )

    if args.method == "aflow":
        search_config = SearchConfig.from_mapping({**dict(config["aflow"]), "seed": config["seed"]})
        search = AFlowSearch(
            dataset=args.dataset,
            config=search_config,
            generator=OpenRouterCandidateGenerator(client),
            evaluator=AFlowValidationEvaluator(
                client,
                selected,
                public_tests=public_tests,
                sandbox=sandbox,
                max_concurrency=args.max_concurrency,
            ),
            output_dir=args.output_dir,
            run_fingerprint=run_fingerprint,
        )
        summary = await search.run()
        candidate_path = (
            args.output_dir / "rounds" / f"round_{summary.best_round}" / "candidate.json"
        )
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    else:
        search_config = ADASConfig.from_mapping(config["adas"])
        search = ADASSearch(
            dataset=args.dataset,
            config=search_config,
            generator=ADASMetaGenerator(client),
            evaluator=ADASValidationEvaluator(
                client,
                selected,
                sandbox=sandbox,
                max_concurrency=args.max_concurrency,
            ),
            output_dir=args.output_dir,
            run_fingerprint=run_fingerprint,
        )
        summary = await search.run()
        archive = json.loads((args.output_dir / "archive.json").read_text(encoding="utf-8"))
        candidate = archive[summary.best_archive_index]["candidate"]

    write_json(args.output_dir / "best_candidate.json", candidate)
    output = {
        "method": args.method,
        "dataset": args.dataset,
        "validation_sample_count": len(selected),
        "best_candidate": str(args.output_dir / "best_candidate.json"),
        "summary": asdict(summary),
    }
    write_json(args.output_dir / "summary.json", output)
    return output


def main() -> int:
    print(json.dumps(asyncio.run(run()), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
