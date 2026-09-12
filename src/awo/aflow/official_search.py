"""Run the pinned upstream AFlow Optimizer with only external-boundary adapters."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import shutil
import sys
import threading
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Literal

from awo.aflow.official import (
    PAPER_COMMIT,
    SNAPSHOT_ROOT,
    OfficialPythonCandidate,
    OfficialSandboxRuntime,
)
from awo.benchmarks.code import score_code
from awo.benchmarks.data import BenchmarkExample
from awo.llm import OpenRouterClient


@dataclass(frozen=True)
class OfficialSearchConfig:
    """Arguments passed directly to the upstream Optimizer."""

    sample: int = 4
    initial_round: int = 1
    max_rounds: int = 20
    validation_rounds: int = 5
    check_convergence: bool = True
    max_concurrency: int = 8

    def __post_init__(self) -> None:
        if min(
            self.sample,
            self.initial_round,
            self.max_rounds,
            self.validation_rounds,
            self.max_concurrency,
        ) <= 0:
            raise ValueError("AFlow search counts and concurrency must be positive")


@dataclass(frozen=True)
class OfficialSearchSummary:
    stopped_reason: str
    generated_attempts: int
    final_round: int
    best_round: int
    best_score: float
    upstream_commit: str = PAPER_COMMIT
    search_driver: str = "scripts.optimizer.Optimizer"


@dataclass(frozen=True)
class UpstreamBindings:
    optimizer_module: Any
    optimizer_class: type
    graph_utils_class: type
    data_utils_class: type
    experience_utils_class: type
    evaluation_utils_class: type
    convergence_utils_class: type
    format_error_class: type[Exception]


@dataclass(frozen=True)
class SandboxGraphReference:
    round_number: int
    candidate: OfficialPythonCandidate


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class _EvaluatorImportBoundary:
    """The object imported by upstream EvaluationUtils as scripts.evaluator.Evaluator."""

    def __init__(self, eval_path: str) -> None:
        self.eval_path = eval_path

    async def graph_evaluate(
        self,
        dataset: str,
        graph: SandboxGraphReference,
        params: dict[str, Any],
        path: str,
        is_test: bool = False,
    ) -> tuple[float, float, float]:
        del params
        if is_test:
            raise ValueError("search adapter only evaluates the validation split")
        if dataset != "HumanEval":
            raise ValueError(f"unsupported sandbox evaluator dataset: {dataset}")
        if _ACTIVE_EVALUATOR is None:
            raise RuntimeError("no active AFlow sandbox evaluator")
        return await _ACTIVE_EVALUATOR.evaluate(graph, Path(path))


_BINDINGS: UpstreamBindings | None = None
_ACTIVE_EVALUATOR: _SandboxEvaluationBoundary | None = None
_RUN_LOCK = threading.Lock()


def load_upstream_bindings() -> UpstreamBindings:
    """Import the exact vendored Optimizer while replacing only its evaluator import."""

    global _BINDINGS
    if _BINDINGS is not None:
        return _BINDINGS

    snapshot = str(SNAPSHOT_ROOT.resolve())
    if snapshot not in sys.path:
        sys.path.insert(0, snapshot)

    scripts_package = sys.modules.get("scripts")
    scripts_path = str((SNAPSHOT_ROOT / "scripts").resolve())
    if scripts_package is not None and hasattr(scripts_package, "__path__"):
        paths = scripts_package.__path__
        if scripts_path not in paths:
            paths.insert(0, scripts_path)

    if "tree_sitter" not in sys.modules:
        tree_sitter = types.ModuleType("tree_sitter")
        tree_sitter.Node = type("Node", (), {})
        sys.modules["tree_sitter"] = tree_sitter

    evaluator = types.ModuleType("scripts.evaluator")
    evaluator.DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP"]
    evaluator.Evaluator = _EvaluatorImportBoundary
    sys.modules["scripts.evaluator"] = evaluator

    optimizer_module = importlib.import_module("scripts.optimizer")
    graph_utils = importlib.import_module("scripts.optimizer_utils.graph_utils")
    data_utils = importlib.import_module("scripts.optimizer_utils.data_utils")
    experience_utils = importlib.import_module("scripts.optimizer_utils.experience_utils")
    evaluation_utils = importlib.import_module("scripts.optimizer_utils.evaluation_utils")
    convergence_utils = importlib.import_module("scripts.optimizer_utils.convergence_utils")
    formatter = importlib.import_module("scripts.formatter")

    optimizer_source = Path(inspect.getsourcefile(optimizer_module.Optimizer) or "").resolve()
    expected_source = (SNAPSHOT_ROOT / "scripts/optimizer.py").resolve()
    if optimizer_source != expected_source:
        raise RuntimeError(f"unexpected AFlow Optimizer source: {optimizer_source}")

    _BINDINGS = UpstreamBindings(
        optimizer_module=optimizer_module,
        optimizer_class=optimizer_module.Optimizer,
        graph_utils_class=graph_utils.GraphUtils,
        data_utils_class=data_utils.DataUtils,
        experience_utils_class=experience_utils.ExperienceUtils,
        evaluation_utils_class=evaluation_utils.EvaluationUtils,
        convergence_utils_class=convergence_utils.ConvergenceUtils,
        format_error_class=formatter.FormatError,
    )
    return _BINDINGS


class _OpenRouterOptimizerBoundary:
    """Upstream AsyncLLM-compatible optimizer model backed by OpenRouter."""

    def __init__(
        self,
        client: OpenRouterClient,
        format_error_class: type[Exception],
    ) -> None:
        self.client = client
        self.format_error_class = format_error_class
        self.total_cost = 0.0
        self.total_tokens = 0
        self.call_count = 0

    async def __call__(self, prompt: str) -> str:
        result = await asyncio.to_thread(
            self.client.chat,
            [{"role": "user", "content": prompt}],
            metadata={
                "role": "aflow_optimizer",
                "dataset": "HumanEval",
                "upstream_commit": PAPER_COMMIT,
            },
        )
        self.total_cost += float(result.usage.cost or 0.0)
        self.total_tokens += int(result.usage.total_tokens)
        self.call_count += 1
        return result.content

    async def call_with_format(self, prompt: str, formatter: Any) -> dict[str, str]:
        response = await self(formatter.prepare_prompt(prompt))
        valid, parsed = formatter.validate_response(response)
        if not valid:
            raise self.format_error_class(
                f"{formatter.format_error_message()}. Raw response: {response}"
            )
        return parsed

    def get_usage_summary(self) -> dict[str, Any]:
        return {
            "total_cost": self.total_cost,
            "total_tokens": self.total_tokens,
            "call_count": self.call_count,
        }


class _SandboxEvaluationBoundary:
    """Run upstream graph objects and HumanEval candidate code in the sandbox."""

    def __init__(
        self,
        runtime: OfficialSandboxRuntime,
        examples: list[BenchmarkExample],
        max_concurrency: int,
    ) -> None:
        self.runtime = runtime
        self.examples = examples
        self.max_concurrency = max_concurrency

    async def _evaluate_one(
        self,
        graph: SandboxGraphReference,
        example: BenchmarkExample,
        semaphore: asyncio.Semaphore,
    ) -> dict[str, Any]:
        async with semaphore:
            try:
                if example.entry_point is None or example.test_code is None:
                    raise ValueError(f"invalid HumanEval example: {example.sample_id}")
                execution = await asyncio.to_thread(
                    self.runtime.execute,
                    graph.candidate,
                    round_number=graph.round_number,
                    problem=example.prompt,
                    entry_point=example.entry_point,
                    metadata={"sample_id": example.sample_id, "split": "validate"},
                )
                score = await asyncio.to_thread(
                    score_code,
                    "humaneval",
                    execution.prediction,
                    tests=example.test_code,
                    sandbox=self.runtime.sandbox,
                    entry_point=example.entry_point,
                )
                return {
                    "sample_id": example.sample_id,
                    "prediction": execution.prediction,
                    "score": score.score,
                    "cost": execution.relay_cost,
                    "tokens": execution.tokens,
                    "calls": execution.calls,
                    "sandbox_runs": execution.sandbox_runs,
                    "score_details": score.details,
                    "error": None,
                }
            except Exception as error:
                return {
                    "sample_id": example.sample_id,
                    "prediction": "",
                    "score": 0.0,
                    "cost": 0.0,
                    "tokens": 0,
                    "calls": 0,
                    "sandbox_runs": 0,
                    "score_details": {},
                    "error": f"{type(error).__name__}: {error}",
                }

    async def evaluate(
        self,
        graph: SandboxGraphReference,
        directory: Path,
    ) -> tuple[float, float, float]:
        semaphore = asyncio.Semaphore(self.max_concurrency)
        rows = await asyncio.gather(
            *(self._evaluate_one(graph, example, semaphore) for example in self.examples)
        )
        score = fmean(float(row["score"]) for row in rows)
        total_cost = max((float(row["cost"]) for row in rows), default=0.0)
        average_cost = total_cost / len(rows)

        repeat = len(list(directory.glob("evaluation_*.jsonl"))) + 1
        (directory / f"evaluation_{repeat}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        log_path = directory / "log.json"
        log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else []
        for row, example in zip(rows, self.examples):
            if not row["score"]:
                log.append(
                    {
                        "question": example.prompt,
                        "right_answer": example.reference,
                        "model_output": row["prediction"],
                        "extracted_output": row["score"],
                        "error": row["error"],
                    }
                )
        _write_json(log_path, log)
        return score, average_cost, total_cost


def _sandbox_graph_utils_class(bindings: UpstreamBindings, runtime: OfficialSandboxRuntime) -> type:
    class SandboxGraphUtils(bindings.graph_utils_class):
        def load_graph(self, round_number: int, workflows_path: str) -> SandboxGraphReference:
            prompt, graph_source = self.read_graph_files(round_number, workflows_path)
            graph = self.extract_solve_graph(graph_source)
            if len(graph) != 1:
                raise RuntimeError(f"round {round_number} does not contain one Workflow class")
            candidate = OfficialPythonCandidate(
                modification="",
                graph=graph[0],
                prompt=prompt,
                parent_round=None,
                metadata={"upstream_commit": PAPER_COMMIT},
            )
            runtime.validate(candidate, round_number=round_number)
            return SandboxGraphReference(round_number, candidate)

    return SandboxGraphUtils


class OfficialAFlowSearch:
    """Thin boundary adapter around the exact upstream Optimizer class."""

    def __init__(
        self,
        *,
        client: OpenRouterClient,
        runtime: OfficialSandboxRuntime,
        examples: list[BenchmarkExample],
        output_dir: Path,
        config: OfficialSearchConfig,
    ) -> None:
        if not examples or any(item.dataset != "humaneval" for item in examples):
            raise ValueError("upstream Python search currently requires HumanEval examples")
        self.client = client
        self.runtime = runtime
        self.examples = examples
        self.output_dir = Path(output_dir)
        self.config = config

    def _prepare_workspace(self) -> None:
        workflows = self.output_dir / "workflows"
        workflows.mkdir(parents=True, exist_ok=True)
        for name in ("template", "round_1"):
            source = SNAPSHOT_ROOT / "workspace/HumanEval/workflows" / name
            destination = workflows / name
            if not destination.exists():
                shutil.copytree(source, destination)

    def build_optimizer(self) -> Any:
        """Construct the official Optimizer and replace only its two external boundaries."""

        self._prepare_workspace()
        bindings = load_upstream_bindings()
        optimizer_llm = _OpenRouterOptimizerBoundary(self.client, bindings.format_error_class)
        bindings.optimizer_module.create_llm_instance = lambda _config: optimizer_llm
        optimizer = bindings.optimizer_class(
            dataset="HumanEval",
            question_type="code",
            opt_llm_config={"model": self.client.settings.model},
            exec_llm_config={"model": self.client.settings.model},
            operators=["Custom", "CustomCodeGenerate", "ScEnsemble", "Test"],
            sample=self.config.sample,
            check_convergence=self.config.check_convergence,
            optimized_path=str(self.output_dir.parent),
            initial_round=self.config.initial_round,
            max_rounds=self.config.max_rounds,
            validation_rounds=self.config.validation_rounds,
        )

        root = str(self.output_dir)
        optimizer.root_path = root
        graph_utils_class = _sandbox_graph_utils_class(bindings, self.runtime)
        optimizer.graph_utils = graph_utils_class(root)
        optimizer.data_utils = bindings.data_utils_class(root)
        optimizer.experience_utils = bindings.experience_utils_class(root)
        optimizer.evaluation_utils = bindings.evaluation_utils_class(root)
        optimizer.convergence_utils = bindings.convergence_utils_class(root)
        return optimizer

    def _best_candidate(self, optimizer: Any, best_round: int) -> OfficialPythonCandidate:
        workflows = self.output_dir / "workflows"
        prompt, graph_source = optimizer.graph_utils.read_graph_files(best_round, str(workflows))
        graph = optimizer.graph_utils.extract_solve_graph(graph_source)
        if len(graph) != 1:
            raise RuntimeError(f"best round {best_round} does not contain one Workflow class")
        experience_path = workflows / f"round_{best_round}" / "experience.json"
        experience = (
            json.loads(experience_path.read_text(encoding="utf-8"))
            if experience_path.exists()
            else {}
        )
        return OfficialPythonCandidate(
            modification=str(experience.get("modification", "")),
            graph=graph[0],
            prompt=prompt,
            parent_round=experience.get("father node"),
            metadata={
                "upstream_commit": PAPER_COMMIT,
                "search_driver": "scripts.optimizer.Optimizer",
            },
        )

    def _run_sync(self) -> OfficialSearchSummary:
        global _ACTIVE_EVALUATOR
        with _RUN_LOCK:
            optimizer = self.build_optimizer()
            _ACTIVE_EVALUATOR = _SandboxEvaluationBoundary(
                self.runtime,
                self.examples,
                self.config.max_concurrency,
            )
            try:
                optimizer.optimize("Graph")
            finally:
                _ACTIVE_EVALUATOR = None

        results = optimizer.data_utils.load_results(str(self.output_dir / "workflows"))
        if not results:
            raise RuntimeError("upstream Optimizer produced no validation results")
        scores: dict[int, list[float]] = {}
        for item in results:
            scores.setdefault(int(item["round"]), []).append(float(item["score"]))
        means = {round_number: fmean(values) for round_number, values in scores.items()}
        best_score = max(means.values())
        best_round = next(
            round_number for round_number in sorted(means) if means[round_number] == best_score
        )
        converged, _, _ = optimizer.convergence_utils.check_convergence(top_k=3)
        summary = OfficialSearchSummary(
            stopped_reason=(
                "convergence" if converged and self.config.check_convergence else "max_rounds"
            ),
            generated_attempts=optimizer.round - self.config.initial_round,
            final_round=optimizer.round,
            best_round=best_round,
            best_score=best_score,
        )
        best_candidate = self._best_candidate(optimizer, best_round)
        _write_json(self.output_dir / "best_candidate.json", best_candidate.to_dict())
        _write_json(self.output_dir / "summary.json", asdict(summary))
        return summary

    async def run(self) -> OfficialSearchSummary:
        return await asyncio.to_thread(self._run_sync)
