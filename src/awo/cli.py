"""Command-line entry point for repository-level checks."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from awo import __version__
from awo.config import config_fingerprint, load_config, public_config
from awo.tracking import build_manifest, write_manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="awo")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    config_parser = subparsers.add_parser("config-check", help="resolve and validate YAML")
    config_parser.add_argument("--config", type=Path, required=True)

    manifest_parser = subparsers.add_parser("manifest", help="write a run manifest")
    manifest_parser.add_argument("--config", type=Path, required=True)
    manifest_parser.add_argument("--output", type=Path, required=True)
    manifest_parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "config-check":
        config = load_config(args.config)
        summary = {
            "config": public_config(config),
            "sha256": config_fingerprint(config),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.command == "manifest":
        manifest = build_manifest(args.config, args.repo_root)
        output = write_manifest(manifest, args.output)
        print(output)
        return 0

    raise AssertionError(f"Unhandled command: {args.command}")
