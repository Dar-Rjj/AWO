#!/usr/bin/env python3
"""Run one auditable adaptive-call Self-Refine sample and score it."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from awo.baselines import SelfRefineBaseline, score_baseline_result
from awo.benchmarks.data import load_and_normalize
from awo.config import load_config
from awo.llm import JsonlRequestRecorder, client_from_config
from awo.sandbox import DockerSandbox


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset", choices=("hotpotqa", "drop", "humaneval", "mbpp", "gsm8k", "math")
    )
    parser.add_argument("dataset_jsonl", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/models/openrouter_deepseek_chat.yaml"),
    )
    parser.add_argument("--split", choices=("validate", "test"), default="validate")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--record-file", type=Path)
    args = parser.parse_args()

    examples = load_and_normalize(args.dataset_jsonl, args.dataset, args.split)
    if not 0 <= args.index < len(examples):
        raise IndexError(f"index {args.index} outside dataset of size {len(examples)}")
    recorder = JsonlRequestRecorder(args.record_file) if args.record_file else None
    baseline = SelfRefineBaseline(client_from_config(load_config(args.config), recorder=recorder))
    example = examples[args.index]
    result = baseline.run(example)
    sandbox = DockerSandbox() if args.dataset in {"humaneval", "mbpp"} else None
    score = score_baseline_result(example, result, sandbox=sandbox)
    output = result.to_dict()
    output["score"] = score.score
    output["score_details"] = score.details
    output["provider_counts"] = dict(
        sorted(Counter(response.provider or "unknown" for response in result.responses).items())
    )
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
