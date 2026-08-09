#!/usr/bin/env python3
"""Replay deterministic scores from official AFlow result CSVs."""

import argparse
import json
from pathlib import Path

from awo.benchmarks.replay import replay_result_csv

DATASET_DIRS = {
    "hotpotqa": "HotpotQA",
    "drop": "DROP",
    "gsm8k": "GSM8K",
    "math": "MATH",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=[*DATASET_DIRS, "all"], default="all")
    parser.add_argument(
        "--math-mode", choices=["archive", "source", "corrected"], default="archive"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    selected = list(DATASET_DIRS) if args.dataset == "all" else [args.dataset]
    results = []
    for dataset in selected:
        pattern_root = args.results_dir / DATASET_DIRS[dataset] / "graphs_test"
        paths = sorted(pattern_root.glob("**/*.csv"))
        if not paths:
            raise FileNotFoundError(f"No archived test CSVs found under {pattern_root}")
        results.extend(
            replay_result_csv(dataset, path, math_mode=args.math_mode) for path in paths
        )
    payload = {"schema_version": 1, "results": results}
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if all(item["mismatches"] == 0 for item in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
