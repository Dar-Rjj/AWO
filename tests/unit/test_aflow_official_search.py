import inspect
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from awo.aflow.official import OfficialExecution, OfficialSandboxRuntime
from awo.aflow.official_search import (
    OfficialAFlowSearch,
    OfficialSearchConfig,
    load_upstream_bindings,
)
from awo.benchmarks.data import BenchmarkExample
from awo.config import load_config
from awo.sandbox import PASS_MARKER, SandboxConfig, SandboxResult


class FakeClient:
    settings = type("Settings", (), {"model": "openai/gpt-4o-mini"})()

    def chat(self, _messages, **_kwargs):
        graph = """class Workflow:
    def __init__(self, name, llm_config, dataset):
        pass

    async def __call__(self, problem, entry_point):
        return problem, 0.0"""
        return SimpleNamespace(
            content=(
                "<modification>keep one code generation step</modification>"
                f"<graph>{graph}</graph>"
                "<prompt>CODE_GENERATION_PROMPT = \"solve\"</prompt>"
            ),
            usage=SimpleNamespace(cost=0.001, total_tokens=10),
        )


class UnusedSandbox:
    config = SandboxConfig()

    def run(self, _source):
        raise AssertionError("build_optimizer must not execute a graph")


class PassingSandbox:
    config = SandboxConfig()

    def run(self, _source):
        return SandboxResult("passed", 0, PASS_MARKER, "", 0.01, "test-sandbox")


class FakeRuntime:
    def __init__(self) -> None:
        self.client = FakeClient()
        self.sandbox = PassingSandbox()
        self.validated: list[int] = []

    def validate(self, _candidate, *, round_number: int) -> None:
        self.validated.append(round_number)

    def execute(self, _candidate, *, round_number, problem, entry_point, metadata):
        del round_number, problem, entry_point, metadata
        return OfficialExecution(
            prediction="def add(a, b):\n    return a + b",
            reported_cost=0.001,
            relay_cost=0.001,
            tokens=10,
            calls=1,
            sandbox_runs=1,
        )


def test_original_parent_probability_uses_upstream_constants() -> None:
    bindings = load_upstream_bindings()
    probabilities = bindings.data_utils_class(".")._compute_probabilities([80, 70])
    expected_first = 0.3 * 0.5 + 0.7 * (math.exp(16) / (math.exp(16) + math.exp(14)))
    assert probabilities[0] == pytest.approx(expected_first)
    assert sum(probabilities) == pytest.approx(1.0)


def test_original_humaneval_config_uses_upstream_search_defaults() -> None:
    config = load_config(Path("configs/aflow/original_humaneval.yaml"))
    aflow = config["aflow"]
    search = OfficialSearchConfig(
        sample=aflow["sample"],
        max_rounds=aflow["max_rounds"],
        validation_rounds=aflow["validation_rounds"],
        check_convergence=aflow["check_convergence"],
        max_concurrency=aflow["max_concurrency"],
    )
    assert search.sample == 4
    assert search.max_rounds == 20
    assert search.validation_rounds == 5
    assert aflow["optimizer_model"] == "anthropic/claude-3.5-sonnet"
    assert aflow["executor_model"] == "openai/gpt-4o-mini"


def test_search_constructs_exact_upstream_optimizer(tmp_path: Path) -> None:
    runtime = OfficialSandboxRuntime(FakeClient(), UnusedSandbox(), humaneval_public_tests={})
    example = BenchmarkExample(
        1,
        "humaneval",
        "validate",
        "HumanEval/0",
        "def add(a, b):",
        "return a + b",
        entry_point="add",
        test_code="def check(candidate): assert candidate(1, 2) == 3",
    )
    search = OfficialAFlowSearch(
        optimizer_client=FakeClient(),
        runtime=runtime,
        examples=[example],
        output_dir=tmp_path / "search",
        config=OfficialSearchConfig(max_rounds=1, validation_rounds=1),
    )
    optimizer = search.build_optimizer()
    source = Path(inspect.getsourcefile(type(optimizer)) or "").resolve()
    assert source.name == "optimizer.py"
    assert "third_party/aflow_paper" in str(source)
    assert type(optimizer.data_utils).__module__ == "scripts.optimizer_utils.data_utils"
    assert type(optimizer.experience_utils).__module__ == (
        "scripts.optimizer_utils.experience_utils"
    )
    assert type(optimizer.evaluation_utils).__module__ == (
        "scripts.optimizer_utils.evaluation_utils"
    )
    assert type(optimizer.convergence_utils).__module__ == (
        "scripts.optimizer_utils.convergence_utils"
    )


def test_upstream_optimizer_runs_one_experimental_round(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bindings = load_upstream_bindings()
    monkeypatch.setattr(bindings.optimizer_module.time, "sleep", lambda _seconds: None)
    example = BenchmarkExample(
        1,
        "humaneval",
        "validate",
        "HumanEval/0",
        "def add(a, b):",
        "return a + b",
        entry_point="add",
        test_code="def check(candidate): assert candidate(1, 2) == 3",
    )
    runtime = FakeRuntime()
    search = OfficialAFlowSearch(
        optimizer_client=FakeClient(),
        runtime=runtime,
        examples=[example],
        output_dir=tmp_path / "search",
        config=OfficialSearchConfig(
            sample=1,
            max_rounds=1,
            validation_rounds=1,
            check_convergence=False,
            max_concurrency=1,
        ),
    )
    summary = search._run_sync()
    assert summary.search_driver == "scripts.optimizer.Optimizer"
    assert summary.generated_attempts == 1
    assert summary.final_round == 2
    assert summary.best_score == 1.0
    assert runtime.validated == [1, 2]
