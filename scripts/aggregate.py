#!/usr/bin/env python3
"""Aggregate Table 1 ledgers into JSON and Markdown reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from awo.reporting import aggregate_runs, write_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = aggregate_runs(args.runs)
    write_report(report, args.output)
    print(json.dumps(report["totals"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
