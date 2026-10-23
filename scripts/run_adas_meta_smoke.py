#!/usr/bin/env python3
"""Run one paid ADAS proposal + two-reflection generation."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from awo.adas import TASK_DESCRIPTIONS, ADASMetaGenerator, initial_archive
from awo.config import load_config
from awo.llm import JsonlRequestRecorder, client_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="gsm8k", choices=sorted(TASK_DESCRIPTIONS))
    parser.add_argument("--record-file", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/models/openrouter_deepseek_chat.yaml"),
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    seeds = initial_archive()
    archive = [
        {
            "generation": f"initial-{index}",
            "fitness": 0.0,
            "candidate": candidate.to_dict(),
        }
        for index, candidate in enumerate(seeds, start=1)
    ]
    recorder = JsonlRequestRecorder(args.record_file) if args.record_file else None
    client = client_from_config(load_config(args.config), recorder=recorder)
    candidate = await ADASMetaGenerator(client).generate(
        dataset=args.dataset,
        generation=1,
        archive=archive,
        previous=None,
    )
    print(json.dumps(candidate.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
