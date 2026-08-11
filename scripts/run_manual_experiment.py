#!/usr/bin/env python3
"""Run a resumable manual-baseline slice through the unified evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from awo.artifacts import sha256_file
from awo.benchmarks.data import load_and_normalize
from awo.config import config_fingerprint, load_config
from awo.experiments import MANUAL_METHODS, ManualExperimentRunner
from awo.llm import JsonlRequestRecorder, client_from_config
from awo.sandbox import DockerSandbox
from awo.tracking import build_manifest, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        choices=("hotpotqa", "drop", "humaneval", "mbpp", "gsm8k", "math"),
    )
    parser.add_argument("dataset_jsonl", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--split", choices=("validate", "test"), default="test")
    parser.add_argument("--methods", nargs="+", choices=MANUAL_METHODS, default=MANUAL_METHODS)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/paper/table1.yaml"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    examples = load_and_normalize(args.dataset_jsonl, args.dataset, args.split)
    stop = None if args.limit is None else args.start + args.limit
    selected = examples[args.start : stop]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args.config)
    manifest_path = args.output_dir / "manifest.json"
    if not manifest_path.exists():
        write_manifest(manifest, manifest_path)
    recorder = JsonlRequestRecorder(args.output_dir / "requests.jsonl")
    client = client_from_config(load_config(args.config), recorder=recorder)
    sandbox = DockerSandbox() if args.dataset in {"humaneval", "mbpp"} else None
    summary = ManualExperimentRunner(
        client=client,
        examples=selected,
        methods=args.methods,
        repeats=args.repeats,
        output_dir=args.output_dir,
        dataset_sha256=sha256_file(args.dataset_jsonl),
        config_sha256=config_fingerprint(load_config(args.config)),
        implementation_commit=manifest["repository"]["commit"],
        seed=42,
        sandbox=sandbox,
    ).run()
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
