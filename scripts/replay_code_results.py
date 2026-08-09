#!/usr/bin/env python3
"""Re-score an archived HumanEval or MBPP CSV in the Docker sandbox."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from awo.benchmarks.replay import replay_code_result_csv
from awo.sandbox import DockerSandbox, SandboxConfig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("HumanEval", "MBPP"))
    parser.add_argument("csv", type=Path)
    parser.add_argument("dataset_jsonl", type=Path)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()
    sandbox = DockerSandbox(SandboxConfig(timeout_seconds=args.timeout))
    report = replay_code_result_csv(
        args.dataset, args.csv, args.dataset_jsonl, sandbox=sandbox
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["mismatches"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
