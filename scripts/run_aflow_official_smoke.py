#!/usr/bin/env python3
"""Run one hash-verified official-best AFlow workflow through DeepSeek Chat."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from awo.aflow import (
    AFlowRuntime,
    OfficialBestWorkflow,
    load_official_manifest,
    load_public_tests,
    verify_official_bundle,
)
from awo.baselines.models import BaselineResult
from awo.baselines.runner import score_baseline_result
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
    parser.add_argument("results_root", type=Path)
    parser.add_argument("--public-tests", type=Path)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--split", choices=("validate", "test"), default="validate")
    parser.add_argument("--record-file", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/aflow/official_best.yaml"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/models/openrouter_deepseek_chat.yaml"),
    )
    args = parser.parse_args()

    artifact_hash, specs = load_official_manifest(args.manifest)
    bundle = verify_official_bundle(
        args.results_root,
        specs[args.dataset],
        expected_artifact_sha256=artifact_hash,
    )
    examples = load_and_normalize(args.dataset_jsonl, args.dataset, args.split)
    example = examples[args.index]
    recorder = JsonlRequestRecorder(args.record_file) if args.record_file else None
    sandbox = DockerSandbox() if args.dataset in {"humaneval", "mbpp", "gsm8k", "math"} else None
    runtime = AFlowRuntime(
        client_from_config(load_config(args.config), recorder=recorder),
        sandbox=sandbox,
        run_metadata={
            "dataset": args.dataset,
            "sample_id": example.sample_id,
            "official_round": bundle.spec.round,
        },
    )
    public_tests = load_public_tests(args.public_tests, examples) if args.public_tests else None
    result = asyncio.run(
        OfficialBestWorkflow(runtime, bundle, public_tests=public_tests).run(example)
    )
    baseline_result = BaselineResult(
        method="aflow_official_best",
        dataset=example.dataset,
        sample_id=example.sample_id,
        prediction=result.prediction,
        prompt_sha256=result.prompt_sha256,
        responses=tuple(runtime.responses),
        protocol="official-best/native-safe-adapter",
        artifacts=result.to_dict(),
    )
    score = score_baseline_result(example, baseline_result, sandbox=sandbox)
    output = baseline_result.to_dict()
    output["score"] = score.score
    output["score_details"] = score.details
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
