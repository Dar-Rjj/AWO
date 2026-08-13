#!/usr/bin/env python3
"""Plan or explicitly execute the registered Table 1 experiment matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from awo.table1 import (
    build_plan,
    execute_plan,
    require_full_table_confirmation,
    write_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/datasets"))
    parser.add_argument("--output-root", type=Path, default=Path("experiments/runs"))
    parser.add_argument("--plan-output", type=Path)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--phase", choices=("search", "test"), action="append")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="issue paid requests; without this flag the command is a dry run",
    )
    parser.add_argument(
        "--confirm-full-table1",
        action="store_true",
        help="second gate required for configs without a sample limit",
    )
    args = parser.parse_args()
    plan = build_plan(
        args.config,
        args.data_root,
        args.output_root,
        max_concurrency=args.max_concurrency,
    )
    plan_output = args.plan_output or Path(plan["output_root"]) / "plan.json"
    write_plan(plan, plan_output)
    print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
    if not args.execute:
        return 0
    try:
        require_full_table_confirmation(plan, args.confirm_full_table1)
    except PermissionError as exc:
        parser.error(f"{exc}; pass --confirm-full-table1 only after pilot budget approval")
    execute_plan(plan, phases=tuple(args.phase or ("search", "test")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
