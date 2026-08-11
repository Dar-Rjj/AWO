#!/usr/bin/env python3
"""Run one official-seed ADAS architecture on one frozen example."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from awo.adas import initial_archive, run_architecture
from awo.baselines.runner import score_baseline_result
from awo.benchmarks.data import load_and_normalize
from awo.config import load_config
from awo.llm import JsonlRequestRecorder, client_from_config
from awo.sandbox import DockerSandbox


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        choices=("hotpotqa", "drop", "humaneval", "mbpp", "gsm8k", "math"),
    )
    parser.add_argument("dataset_jsonl", type=Path)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--seed-index", type=int, default=0, choices=range(7))
    parser.add_argument("--record-file", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/models/openrouter_deepseek_chat.yaml"),
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    example = load_and_normalize(args.dataset_jsonl, args.dataset, "validate")[args.index]
    candidate = initial_archive()[args.seed_index]
    recorder = JsonlRequestRecorder(args.record_file) if args.record_file else None
    client = client_from_config(load_config(args.config), recorder=recorder)
    result = await run_architecture(candidate, client, example)
    sandbox = DockerSandbox() if args.dataset in {"humaneval", "mbpp"} else None
    score = score_baseline_result(example, result, sandbox=sandbox)
    output = result.to_dict()
    output["score"] = score.score
    output["score_details"] = score.details
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
