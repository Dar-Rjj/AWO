import inspect
import math
from pathlib import Path

import pytest

from awo.aflow.official import OfficialSandboxRuntime
from awo.aflow.official_search import (
    OfficialAFlowSearch,
    OfficialSearchConfig,
    load_upstream_bindings,
)
from awo.benchmarks.data import BenchmarkExample
from awo.config import load_config
from awo.sandbox import SandboxConfig


class FakeClient:
    settings = type("Settings", (), {"model": "openai/gpt-4o-mini"})()


class UnusedSandbox:
    config = SandboxConfig()

    def run(self, _source):
        raise AssertionError("build_optimizer must not execute a graph")


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
        client=FakeClient(),
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
