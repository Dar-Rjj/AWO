#!/usr/bin/env python3
"""Rescore saved HumanEval/MBPP predictions without calling an LLM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from awo.benchmarks.data import load_and_normalize
from awo.experiments.rescore import rescore_code_records
from awo.sandbox import DockerSandbox


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("humaneval", "mbpp"))
    parser.add_argument("dataset_jsonl", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    examples = load_and_normalize(args.dataset_jsonl, args.dataset, "test")
    report = rescore_code_records(
        args.output_dir,
        examples,
        sandbox=DockerSandbox(),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
