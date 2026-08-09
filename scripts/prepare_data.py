#!/usr/bin/env python3
"""Normalize all frozen validation and test splits."""

import argparse
import json
from pathlib import Path

from awo.benchmarks import prepare_all_datasets

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data/raw/datasets")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data/processed")
    args = parser.parse_args()
    manifest = prepare_all_datasets(args.raw_dir, args.output_dir)
    summary = {
        "files": len(manifest["files"]),
        "records": sum(item["records"] for item in manifest["files"]),
        "output_dir": str(args.output_dir),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
