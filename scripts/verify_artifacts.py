#!/usr/bin/env python3
"""Verify cached and extracted official AFlow artifacts without downloading."""

import argparse
import json
from pathlib import Path

from awo.artifacts import (
    load_artifact_manifest,
    verify_archive,
    verify_dataset_files,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "data/manifests/aflow-artifacts.json",
    )
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "data/raw/archives")
    parser.add_argument("--dataset-dir", type=Path, default=REPO_ROOT / "data/raw/datasets")
    parser.add_argument(
        "--archive",
        action="append",
        choices=["datasets", "results", "initial_rounds"],
        help="verify a cached archive; repeat to select several",
    )
    parser.add_argument("--skip-dataset-files", action="store_true")
    args = parser.parse_args()

    archives, dataset_files = load_artifact_manifest(args.manifest)
    summary = {"archives": {}, "dataset_files": []}
    for name in args.archive or []:
        spec = archives[name]
        summary["archives"][name] = verify_archive(args.cache_dir / spec.filename, spec)
    if not args.skip_dataset_files:
        summary["dataset_files"] = verify_dataset_files(args.dataset_dir, dataset_files)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
