#!/usr/bin/env python3
"""Run the resumable strict-boxed MATH IO protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from awo.artifacts import sha256_file
from awo.benchmarks.data import load_and_normalize
from awo.config import config_fingerprint, load_config
from awo.experiments.strict_math_io import StrictMathIOExperimentRunner
from awo.llm import JsonlRequestRecorder, client_from_config
from awo.tracking import build_manifest, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_jsonl", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/math_io_gpt_4o_mini_strict_boxed.yaml"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repeats <= 0 or args.start < 0 or (args.limit is not None and args.limit <= 0):
        raise SystemExit("--repeats/--limit must be positive and --start cannot be negative")
    examples = load_and_normalize(args.dataset_jsonl, "math", "test")
    stop = None if args.limit is None else args.start + args.limit
    selected = examples[args.start : stop]
    if not selected:
        raise SystemExit("the requested dataset slice is empty")

    config = load_config(args.config)
    manifest = build_manifest(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    if not manifest_path.exists():
        write_manifest(manifest, manifest_path)
    recorder = JsonlRequestRecorder(args.output_dir / "requests.jsonl")
    client = client_from_config(config, recorder=recorder)
    summary = StrictMathIOExperimentRunner(
        client=client,
        examples=selected,
        methods=("io",),
        repeats=args.repeats,
        output_dir=args.output_dir,
        dataset_sha256=sha256_file(args.dataset_jsonl),
        config_sha256=config_fingerprint(config),
        implementation_commit=manifest["repository"]["commit"],
        seed=42,
    ).run()
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
