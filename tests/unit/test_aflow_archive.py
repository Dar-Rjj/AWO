from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from awo.aflow import (
    AFlowRuntime,
    ArchivedWorkflowError,
    OfficialBestWorkflow,
    OfficialWorkflowBundle,
    OfficialWorkflowSpec,
    load_literal_prompts,
    load_official_manifest,
    load_public_tests,
    verify_official_bundle,
)
from awo.benchmarks.data import BenchmarkExample
from awo.llm import ChatResult, TokenUsage


class FakeClient:
    def __init__(self, contents: list[str]) -> None:
        self.contents = iter(contents)
        self.calls: list[tuple[Any, Any]] = []

    def chat(self, messages: Any, *, metadata: Any = None) -> ChatResult:
        index = len(self.calls)
        self.calls.append((messages, metadata))
        return ChatResult(
            request_id=f"request-{index}",
            response_id=f"response-{index}",
            requested_model="deepseek/deepseek-chat",
            actual_model="deepseek/deepseek-chat",
            provider="test",
            content=next(self.contents),
            finish_reason="stop",
            usage=TokenUsage(total_tokens=10, cost=0.001),
            attempts=1,
            latency_seconds=0.01,
            request_sha256=str(index) * 64,
        )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def test_frozen_manifest_names_all_six_official_best_rounds() -> None:
    root = Path(__file__).parents[2]
    artifact_hash, specs = load_official_manifest(
        root / "configs" / "aflow" / "official_best.yaml"
    )
    assert len(artifact_hash) == 64
    assert {name: spec.round for name, spec in specs.items()} == {
        "drop": 3,
        "gsm8k": 10,
        "hotpotqa": 3,
        "humaneval": 5,
        "math": 5,
        "mbpp": 14,
    }


def test_literal_prompt_loader_never_executes_python(tmp_path: Path) -> None:
    safe = tmp_path / "safe.py"
    safe.write_text('PROMPT = "hello"\n')
    assert load_literal_prompts(safe) == {"PROMPT": "hello"}
    unsafe = tmp_path / "unsafe.py"
    unsafe.write_text('import os\nPROMPT = "hello"\n')
    with pytest.raises(ArchivedWorkflowError):
        load_literal_prompts(unsafe)


def test_bundle_requires_marker_and_exact_file_hashes(tmp_path: Path) -> None:
    graph = "class Workflow:\n    pass\n"
    prompt = 'PROMPT = "hello"\n'
    directory = tmp_path / "GSM8K" / "graphs_test" / "round_10"
    directory.mkdir(parents=True)
    (directory / "graph.py").write_text(graph)
    (directory / "prompt.py").write_text(prompt)
    (tmp_path / ".artifact.json").write_text(json.dumps({"sha256": "a" * 64}))
    spec = OfficialWorkflowSpec("gsm8k", "GSM8K", 10, sha256_text(graph), sha256_text(prompt))
    bundle = verify_official_bundle(tmp_path, spec, expected_artifact_sha256="a" * 64)
    assert bundle.prompts == {"PROMPT": "hello"}
    (directory / "graph.py").write_text(graph + "# drift\n")
    with pytest.raises(ArchivedWorkflowError):
        verify_official_bundle(tmp_path, spec, expected_artifact_sha256="a" * 64)


def test_public_test_loader_is_entry_point_keyed(tmp_path: Path) -> None:
    path = tmp_path / "public.jsonl"
    path.write_text('{"entry_point":"solve","test":["assert candidate() == 1"]}\n')
    assert load_public_tests(path) == {"solve": "assert candidate() == 1"}


def test_native_hotpot_adapter_reconstructs_five_call_topology() -> None:
    contents = [
        '{"thought":"t","answer":"a"}',
        '{"thought":"t","answer":"a"}',
        '{"thought":"t","answer":"a"}',
        '{"solution_letter":"A"}',
        "a",
    ]
    runtime = AFlowRuntime(FakeClient(contents))  # type: ignore[arg-type]
    spec = OfficialWorkflowSpec("hotpotqa", "HotpotQA", 3, "g" * 64, "p" * 64)
    bundle = OfficialWorkflowBundle(
        spec,
        Path("graph.py"),
        Path("prompt.py"),
        {"FORMAT_ANSWER_PROMPT": "format:"},
    )
    example = BenchmarkExample(1, "hotpotqa", "validate", "id", "problem", "a")
    result = asyncio.run(OfficialBestWorkflow(runtime, bundle).run(example))
    assert result.prediction == "a"
    assert result.call_count == 5
    assert result.operator_trace == (
        "AnswerGenerate",
        "AnswerGenerate",
        "AnswerGenerate",
        "ScEnsemble",
        "Custom",
    )


def test_workflow_usage_is_per_sample_not_cumulative() -> None:
    contents = [
        *(['{"thought":"t","answer":"a"}'] * 3),
        '{"solution_letter":"A"}',
        "a",
    ] * 2
    runtime = AFlowRuntime(FakeClient(contents))  # type: ignore[arg-type]
    spec = OfficialWorkflowSpec("hotpotqa", "HotpotQA", 3, "g" * 64, "p" * 64)
    bundle = OfficialWorkflowBundle(
        spec, Path("graph.py"), Path("prompt.py"), {"FORMAT_ANSWER_PROMPT": "format:"}
    )
    workflow = OfficialBestWorkflow(runtime, bundle)
    first = asyncio.run(
        workflow.run(BenchmarkExample(1, "hotpotqa", "validate", "one", "problem", "a"))
    )
    second = asyncio.run(
        workflow.run(BenchmarkExample(1, "hotpotqa", "validate", "two", "problem", "a"))
    )
    assert first.call_count == second.call_count == 5
    assert first.total_tokens == second.total_tokens == 50
    assert len(runtime.responses) == 10
