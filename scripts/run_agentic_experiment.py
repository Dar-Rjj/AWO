#!/usr/bin/env python3
"""Run resumable frozen AFlow or ADAS experiments through one ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from awo.adas import (
    ADASArchitecture,
    initial_archive,
    run_architecture,
    validate_architecture,
)
from awo.aflow import (
    AFlowRuntime,
    OfficialBestWorkflow,
    OfficialWorkflowBundle,
    WorkflowCandidate,
    load_official_manifest,
    load_public_tests,
    validate_candidate,
    verify_official_bundle,
)
from awo.artifacts import sha256_file
from awo.baselines.models import BaselineResult
from awo.benchmarks.data import BenchmarkExample, load_and_normalize
from awo.config import config_fingerprint, load_config
from awo.experiments import AgenticExperimentRunner, aflow_candidate_executor
from awo.llm import JsonlRequestRecorder, OpenRouterClient, client_from_config
from awo.sandbox import DockerSandbox
from awo.tracking import build_manifest, write_manifest

METHODS = ("aflow", "aflow_official_best", "adas")
DATASETS = ("hotpotqa", "drop", "humaneval", "mbpp", "gsm8k", "math")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=METHODS)
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("dataset_jsonl", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--split", choices=("validate", "test"), default="test")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--public-tests", type=Path)
    parser.add_argument("--aflow-candidate", type=Path)
    adas_source = parser.add_mutually_exclusive_group()
    adas_source.add_argument("--adas-seed-index", type=int, choices=range(7))
    adas_source.add_argument("--adas-candidate", type=Path)
    parser.add_argument(
        "--official-manifest",
        type=Path,
        default=Path("configs/aflow/official_best.yaml"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/paper/table1.yaml"),
    )
    return parser


def aflow_executor(
    bundle: OfficialWorkflowBundle,
    public_tests: dict[str, str] | None,
    sandbox: DockerSandbox | None,
):
    async def execute(
        client: OpenRouterClient,
        example: BenchmarkExample,
    ) -> BaselineResult:
        runtime = AFlowRuntime(
            client,
            sandbox=sandbox,
            run_metadata={
                "dataset": example.dataset,
                "sample_id": example.sample_id,
                "official_round": bundle.spec.round,
            },
        )
        workflow_result = await OfficialBestWorkflow(
            runtime,
            bundle,
            public_tests=public_tests,
        ).run(example)
        return BaselineResult(
            method="aflow_official_best",
            dataset=example.dataset,
            sample_id=example.sample_id,
            prediction=workflow_result.prediction,
            prompt_sha256=workflow_result.prompt_sha256,
            responses=tuple(runtime.responses),
            protocol="official-best/native-safe-adapter",
            artifacts=workflow_result.to_dict(),
        )

    return execute


def adas_executor(candidate: ADASArchitecture):
    async def execute(
        client: OpenRouterClient,
        example: BenchmarkExample,
    ) -> BaselineResult:
        return await run_architecture(candidate, client, example)

    return execute


def load_adas_candidate(path: Path) -> ADASArchitecture:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("ADAS candidate file must contain one JSON object")
    candidate = ADASArchitecture.from_dict(payload)
    validate_architecture(candidate)
    return candidate


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.repeats <= 0 or args.start < 0 or (args.limit is not None and args.limit <= 0):
        parser.error("--repeats/--limit must be positive and --start cannot be negative")
    if args.method == "aflow_official_best":
        if args.results_root is None:
            parser.error("--results-root is required for aflow_official_best")
        if args.dataset in {"humaneval", "mbpp"} and args.public_tests is None:
            parser.error("--public-tests is required for AFlow code workflows")
        if args.adas_seed_index is not None or args.adas_candidate is not None:
            parser.error("ADAS candidate options cannot be used with AFlow")
        if args.aflow_candidate is not None:
            parser.error("--aflow-candidate is only valid for searched AFlow")
    elif args.method == "aflow":
        if args.aflow_candidate is None:
            parser.error("--aflow-candidate is required for searched AFlow")
        if args.results_root is not None:
            parser.error("--results-root is only valid for official AFlow replay")
        if args.dataset in {"humaneval", "mbpp"} and args.public_tests is None:
            parser.error("--public-tests is required for AFlow code workflows")
        if args.adas_seed_index is not None or args.adas_candidate is not None:
            parser.error("ADAS candidate options cannot be used with AFlow")
    elif (
        args.results_root is not None
        or args.public_tests is not None
        or args.aflow_candidate is not None
    ):
        parser.error("AFlow options cannot be used with ADAS")

    examples = load_and_normalize(args.dataset_jsonl, args.dataset, args.split)
    stop = None if args.limit is None else args.start + args.limit
    selected = examples[args.start : stop]
    if not selected:
        parser.error("the requested dataset slice is empty")

    config = load_config(args.config)
    manifest = build_manifest(args.config)
    sandbox_datasets = {"humaneval", "mbpp"}
    if args.method in {"aflow", "aflow_official_best"}:
        sandbox_datasets.update({"gsm8k", "math"})
    sandbox = DockerSandbox() if args.dataset in sandbox_datasets else None

    if args.method == "aflow_official_best":
        assert args.results_root is not None
        artifact_hash, specs = load_official_manifest(args.official_manifest)
        bundle = verify_official_bundle(
            args.results_root,
            specs[args.dataset],
            expected_artifact_sha256=artifact_hash,
        )
        public_tests = (
            load_public_tests(args.public_tests, examples)
            if args.public_tests is not None
            else None
        )
        executor = aflow_executor(bundle, public_tests, sandbox)
        fingerprint = {
            "protocol": "official-best/native-safe-adapter",
            "results_artifact_sha256": artifact_hash,
            "official_round": bundle.spec.round,
            "graph_sha256": bundle.spec.graph_sha256,
            "prompt_sha256": bundle.spec.prompt_sha256,
            "public_tests_sha256": (
                sha256_file(args.public_tests) if args.public_tests is not None else None
            ),
        }
    elif args.method == "aflow":
        assert args.aflow_candidate is not None
        payload = json.loads(args.aflow_candidate.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("AFlow candidate file must contain one JSON object")
        candidate = WorkflowCandidate.from_dict(payload)
        validate_candidate(candidate, args.dataset)
        public_tests = (
            load_public_tests(args.public_tests, examples) if args.public_tests is not None else {}
        )
        executor = aflow_candidate_executor(candidate, public_tests, sandbox)
        fingerprint = {
            "protocol": "controlled-search/declarative_v1",
            "candidate_file_sha256": sha256_file(args.aflow_candidate),
            "candidate_sha256": candidate.sha256,
            "workflow_sha256": candidate.workflow_sha256,
            "public_tests_sha256": (
                sha256_file(args.public_tests) if args.public_tests is not None else None
            ),
        }
    else:
        seed_index = args.adas_seed_index if args.adas_seed_index is not None else 0
        candidate = (
            load_adas_candidate(args.adas_candidate)
            if args.adas_candidate is not None
            else initial_archive()[seed_index]
        )
        executor = adas_executor(candidate)
        fingerprint = {
            "protocol": "protocol-compatible/official-meta-agent-safe-dag",
            "source": "frozen_candidate" if args.adas_candidate is not None else "seed",
            "seed_index": None if args.adas_candidate is not None else seed_index,
            "candidate_file_sha256": (
                sha256_file(args.adas_candidate) if args.adas_candidate is not None else None
            ),
            "architecture_name": candidate.name,
            "architecture_sha256": candidate.architecture_sha256,
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    if not manifest_path.exists():
        write_manifest(manifest, manifest_path)
    recorder = JsonlRequestRecorder(args.output_dir / "requests.jsonl")
    client = client_from_config(config, recorder=recorder)

    summary = AgenticExperimentRunner(
        client=client,
        executor=executor,
        method=args.method,
        executor_fingerprint=fingerprint,
        examples=selected,
        repeats=args.repeats,
        output_dir=args.output_dir,
        dataset_sha256=sha256_file(args.dataset_jsonl),
        config_sha256=config_fingerprint(config),
        implementation_commit=manifest["repository"]["commit"],
        sandbox=sandbox,
    ).run()
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
