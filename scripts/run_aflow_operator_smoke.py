#!/usr/bin/env python3
"""Run the AFlow blank-workflow Custom operator on one auditable sample."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from awo.aflow import AFlowRuntime, Custom
from awo.baselines.models import BaselineResult
from awo.baselines.runner import score_baseline_result
from awo.benchmarks.data import load_and_normalize
from awo.config import load_config
from awo.llm import JsonlRequestRecorder, client_from_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("hotpotqa", "drop", "gsm8k", "math"))
    parser.add_argument("dataset_jsonl", type=Path)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--split", choices=("validate", "test"), default="validate")
    parser.add_argument("--instruction", default="")
    parser.add_argument("--record-file", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/models/openrouter_deepseek_chat.yaml"),
    )
    args = parser.parse_args()

    examples = load_and_normalize(args.dataset_jsonl, args.dataset, args.split)
    example = examples[args.index]
    recorder = JsonlRequestRecorder(args.record_file) if args.record_file else None
    runtime = AFlowRuntime(
        client_from_config(load_config(args.config), recorder=recorder),
        run_metadata={"dataset": args.dataset, "sample_id": example.sample_id},
    )
    result = asyncio.run(Custom(runtime)(example.prompt, args.instruction))
    baseline_result = BaselineResult(
        method="aflow_blank",
        dataset=example.dataset,
        sample_id=example.sample_id,
        prediction=result["response"],
        prompt_sha256=runtime.responses[0].request_sha256,
        responses=tuple(runtime.responses),
        protocol="paper-faithful/blank-workflow",
    )
    score = score_baseline_result(example, baseline_result)
    output = baseline_result.to_dict()
    output["usage"] = runtime.usage_summary()
    output["score"] = score.score
    output["score_details"] = score.details
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
