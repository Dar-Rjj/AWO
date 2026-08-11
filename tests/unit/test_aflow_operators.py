from __future__ import annotations

import asyncio
from typing import Any

import pytest

from awo.aflow import (
    AFlowRuntime,
    AnswerGenerate,
    Custom,
    CustomCodeGenerate,
    Programmer,
    ScEnsemble,
)
from awo.aflow import (
    Test as PublicTestOperator,
)
from awo.llm import ChatResult, TokenUsage
from awo.sandbox import PASS_MARKER, SandboxResult


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


class FakeSandbox:
    def __init__(self, results: list[SandboxResult]) -> None:
        self.results = iter(results)
        self.sources: list[str] = []

    def run(self, source: str) -> SandboxResult:
        self.sources.append(source)
        return next(self.results)


def sandbox_result(status: str, *, stdout: str = "", stderr: str = "") -> SandboxResult:
    return SandboxResult(status, 0, stdout, stderr, 0.01, "test-image")  # type: ignore[arg-type]


def test_custom_preserves_historical_prompt_concatenation_and_usage() -> None:
    client = FakeClient(["answer"])
    runtime = AFlowRuntime(client, run_metadata={"dataset": "gsm8k"})  # type: ignore[arg-type]
    result = asyncio.run(Custom(runtime)("problem", "instruction:"))
    assert result == {"response": "answer"}
    assert client.calls[0][0] == [{"role": "user", "content": "instruction:problem"}]
    assert client.calls[0][1]["method"] == "aflow"
    assert runtime.usage_summary() == {
        "call_count": 1,
        "total_tokens": 10,
        "total_cost": 0.001,
    }


def test_answer_generate_uses_explicit_structured_schema() -> None:
    client = FakeClient(['{"thought":"reason","answer":"final"}'])
    runtime = AFlowRuntime(client)  # type: ignore[arg-type]
    assert asyncio.run(AnswerGenerate(runtime)("question")) == {
        "thought": "reason",
        "answer": "final",
    }


def test_custom_code_generate_requires_entry_point() -> None:
    client = FakeClient(["```python\ndef target(x):\n    return x\n```"])
    runtime = AFlowRuntime(client)  # type: ignore[arg-type]
    result = asyncio.run(CustomCodeGenerate(runtime)("problem", "target", "Solve: "))
    assert "def target" in result["response"]
    assert "function named 'target'" in client.calls[0][0][0]["content"]


def test_sc_ensemble_selects_explicit_letter() -> None:
    client = FakeClient(['{"solution_letter":"B"}'])
    runtime = AFlowRuntime(client)  # type: ignore[arg-type]
    result = asyncio.run(ScEnsemble(runtime)(["first", "second"], "problem"))
    assert result == {"response": "second"}


def test_programmer_retries_only_inside_networkless_sandbox() -> None:
    client = FakeClient(
        [
            "```python\ndef solve():\n    raise ValueError('bad')\n```",
            "```python\ndef solve():\n    return 36\n```",
        ]
    )
    sandbox = FakeSandbox(
        [
            sandbox_result("failed", stderr="ValueError"),
            sandbox_result(
                "passed", stdout='__AWO_PROGRAMMER_OUTPUT__36\n' + PASS_MARKER + "\n"
            ),
        ]
    )
    runtime = AFlowRuntime(client, sandbox=sandbox)  # type: ignore[arg-type]
    result = asyncio.run(Programmer(runtime)("problem"))
    assert result["output"] == "36"
    assert result["sandbox_statuses"] == ["failed", "passed"]
    assert len(client.calls) == 2
    assert all("solve()" in source for source in sandbox.sources)


def test_public_test_reflection_never_executes_on_host() -> None:
    client = FakeClient(["```python\ndef candidate(x):\n    return x + 1\n```"])
    sandbox = FakeSandbox(
        [sandbox_result("failed", stderr="AssertionError"), sandbox_result("passed")]
    )
    runtime = AFlowRuntime(client, sandbox=sandbox)  # type: ignore[arg-type]
    result = asyncio.run(
        PublicTestOperator(runtime)(
            "problem",
            "def candidate(x): return x",
            "candidate",
            "assert candidate(1) == 2",
        )
    )
    assert result["result"] is True
    assert len(sandbox.sources) == 2
    assert len(client.calls) == 1


def test_sandbox_is_mandatory_for_code_execution_operators() -> None:
    runtime = AFlowRuntime(FakeClient([]))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Programmer(runtime)
    with pytest.raises(ValueError):
        PublicTestOperator(runtime)
