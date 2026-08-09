#!/usr/bin/env python3
"""Download and safely extract frozen AFlow artifacts."""

import argparse
import json
from pathlib import Path

from awo.artifacts import (
    download_artifact,
    extract_artifact,
    load_artifact_manifest,
    verify_dataset_files,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "data/manifests/aflow-artifacts.json"

DESTINATIONS = {
    "datasets": (Path("raw/datasets"), None),
    "results": (Path("results"), "results"),
    "initial_rounds": (Path("workspace"), None),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact",
        action="append",
        choices=[*DESTINATIONS, "all"],
        help="artifact to fetch; repeat to select several (default: datasets)",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "data/raw/archives")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    args = parser.parse_args()

    archives, dataset_files = load_artifact_manifest(args.manifest)
    requested = args.artifact or ["datasets"]
    names = list(DESTINATIONS) if "all" in requested else list(dict.fromkeys(requested))
    summary = []
    for name in names:
        spec = archives[name]
        archive = download_artifact(spec, args.cache_dir)
        relative_destination, strip_prefix = DESTINATIONS[name]
        destination = extract_artifact(
            archive,
            args.data_root / relative_destination,
            spec,
            strip_prefix=strip_prefix,
        )
        entry = {"artifact": name, "archive": str(archive), "destination": str(destination)}
        if name == "datasets":
            entry["verified_files"] = len(verify_dataset_files(destination, dataset_files))
        summary.append(entry)

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
