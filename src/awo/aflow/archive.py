"""Hash-verified native adapters for the six published best AFlow workflows."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from awo.aflow.operators import (
    AnswerGenerate,
    Custom,
    CustomCodeGenerate,
    Programmer,
    ScEnsemble,
    Test,
)
from awo.aflow.runtime import AFlowRuntime
from awo.artifacts import sha256_file
from awo.benchmarks.data import BenchmarkExample


class ArchivedWorkflowError(RuntimeError):
    """Raised when an archived workflow is untrusted or incompatible."""


@dataclass(frozen=True)
class OfficialWorkflowSpec:
    dataset: str
    official_dataset: str
    round: int
    graph_sha256: str
    prompt_sha256: str

    @property
    def relative_directory(self) -> Path:
        return Path(self.official_dataset) / "graphs_test" / f"round_{self.round}"


@dataclass(frozen=True)
class OfficialWorkflowBundle:
    spec: OfficialWorkflowSpec
    graph_path: Path
    prompt_path: Path
    prompts: dict[str, str]


@dataclass(frozen=True)
class AFlowWorkflowResult:
    dataset: str
    sample_id: str
    prediction: str
    official_round: int
    graph_sha256: str
    prompt_sha256: str
    call_count: int
    total_tokens: int
    total_cost: float
    operator_trace: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "sample_id": self.sample_id,
            "prediction": self.prediction,
            "official_round": self.official_round,
            "graph_sha256": self.graph_sha256,
            "prompt_sha256": self.prompt_sha256,
            "call_count": self.call_count,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "operator_trace": list(self.operator_trace),
            "protocol": "official-best/native-safe-adapter",
        }


def load_official_manifest(
    path: Path,
) -> tuple[str, dict[str, OfficialWorkflowSpec]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ArchivedWorkflowError("official workflow manifest must use schema_version 1")
    artifact_hash = payload.get("results_artifact_sha256")
    workflows = payload.get("workflows")
    if not isinstance(artifact_hash, str) or not isinstance(workflows, dict):
        raise ArchivedWorkflowError("official workflow manifest is incomplete")
    specs = {}
    for dataset, values in workflows.items():
        if not isinstance(dataset, str) or not isinstance(values, dict):
            raise ArchivedWorkflowError("invalid workflow manifest entry")
        specs[dataset] = OfficialWorkflowSpec(dataset=dataset, **values)
    return artifact_hash, specs


def load_literal_prompts(path: Path) -> dict[str, str]:
    """Read only uppercase string assignments; never import or execute archived Python."""

    source = Path(path).read_text(encoding="utf-8")
    try:
        module = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise ArchivedWorkflowError(f"invalid archived prompt syntax: {path}") from exc
    prompts = {}
    for statement in module.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            raise ArchivedWorkflowError("archived prompt may only contain direct assignments")
        target = statement.targets[0]
        if not isinstance(target, ast.Name) or not target.id.isupper():
            raise ArchivedWorkflowError("archived prompt names must be uppercase identifiers")
        try:
            value = ast.literal_eval(statement.value)
        except (ValueError, TypeError) as exc:
            raise ArchivedWorkflowError("archived prompt value must be a literal") from exc
        if not isinstance(value, str):
            raise ArchivedWorkflowError("archived prompt values must be strings")
        prompts[target.id] = value
    if not prompts:
        raise ArchivedWorkflowError("archived prompt contains no constants")
    return prompts


def verify_official_bundle(
    results_root: Path,
    spec: OfficialWorkflowSpec,
    *,
    expected_artifact_sha256: str,
) -> OfficialWorkflowBundle:
    root = Path(results_root)
    marker = root / ".artifact.json"
    try:
        marker_payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArchivedWorkflowError(f"missing or invalid results marker: {marker}") from exc
    if marker_payload.get("sha256") != expected_artifact_sha256:
        raise ArchivedWorkflowError("results artifact marker SHA256 mismatch")
    directory = root / spec.relative_directory
    graph_path = directory / "graph.py"
    prompt_path = directory / "prompt.py"
    if sha256_file(graph_path) != spec.graph_sha256:
        raise ArchivedWorkflowError(f"archived graph SHA256 mismatch: {graph_path}")
    if sha256_file(prompt_path) != spec.prompt_sha256:
        raise ArchivedWorkflowError(f"archived prompt SHA256 mismatch: {prompt_path}")
    return OfficialWorkflowBundle(
        spec=spec,
        graph_path=graph_path,
        prompt_path=prompt_path,
        prompts=load_literal_prompts(prompt_path),
    )


def load_public_tests(path: Path) -> dict[str, str]:
    tests = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ArchivedWorkflowError(
                    f"invalid public-test JSON at {path}:{line_number}"
                ) from exc
            entry_point = record.get("entry_point")
            assertions = record.get("test")
            if not isinstance(entry_point, str) or not isinstance(assertions, list):
                raise ArchivedWorkflowError(f"invalid public-test schema at {path}:{line_number}")
            if entry_point in tests or not all(isinstance(item, str) for item in assertions):
                raise ArchivedWorkflowError(f"invalid public tests for {entry_point!r}")
            tests[entry_point] = "\n".join(assertions)
    return tests


class OfficialBestWorkflow:
    """Native reconstruction selected only after graph and prompt hash verification."""

    def __init__(
        self,
        runtime: AFlowRuntime,
        bundle: OfficialWorkflowBundle,
        *,
        public_tests: dict[str, str] | None = None,
    ) -> None:
        self.runtime = runtime
        self.bundle = bundle
        self.public_tests = public_tests or {}
        self.trace: list[str] = []

    async def _custom(self, problem: str, instruction: str) -> str:
        self.trace.append("Custom")
        return (await Custom(self.runtime)(problem, instruction))["response"]

    async def _answer(self, problem: str) -> dict[str, str]:
        self.trace.append("AnswerGenerate")
        return await AnswerGenerate(self.runtime)(problem)

    async def _ensemble(self, solutions: list[str], problem: str, *, qa: bool = False) -> str:
        self.trace.append("ScEnsemble")
        return (await ScEnsemble(self.runtime, qa_mode=qa)(solutions, problem))["response"]

    async def _program(self, problem: str, analysis: str = "None") -> dict[str, Any]:
        self.trace.append("Programmer")
        return await Programmer(self.runtime)(problem, analysis)

    async def _code(self, problem: str, entry_point: str, instruction: str) -> str:
        self.trace.append("CustomCodeGenerate")
        return (await CustomCodeGenerate(self.runtime)(problem, entry_point, instruction))[
            "response"
        ]

    async def _test(self, problem: str, solution: str, entry_point: str) -> dict[str, Any]:
        if entry_point not in self.public_tests:
            raise ArchivedWorkflowError(f"no frozen public tests for entry point {entry_point}")
        self.trace.append("Test")
        return await Test(self.runtime)(
            problem,
            solution,
            entry_point,
            self.public_tests[entry_point],
        )

    async def run(self, example: BenchmarkExample) -> AFlowWorkflowResult:
        if example.dataset != self.bundle.spec.dataset:
            raise ValueError("example dataset does not match official workflow bundle")
        self.trace = []
        response_start = len(self.runtime.responses)
        method = getattr(self, f"_run_{example.dataset}")
        prediction = await method(example)
        sample_responses = self.runtime.responses[response_start:]
        return AFlowWorkflowResult(
            dataset=example.dataset,
            sample_id=example.sample_id,
            prediction=prediction,
            official_round=self.bundle.spec.round,
            graph_sha256=self.bundle.spec.graph_sha256,
            prompt_sha256=self.bundle.spec.prompt_sha256,
            call_count=len(sample_responses),
            total_tokens=sum(response.usage.total_tokens for response in sample_responses),
            total_cost=sum(response.usage.cost or 0.0 for response in sample_responses),
            operator_trace=tuple(self.trace),
        )

    async def _run_drop(self, example: BenchmarkExample) -> str:
        solutions = [(await self._answer(example.prompt))["answer"] for _ in range(3)]
        selected = await self._ensemble(solutions, example.prompt, qa=True)
        problem = f"Question: {example.prompt}\nBest solution: {selected}"
        return await self._custom(problem, self.bundle.prompts["REFINE_ANSWER_PROMPT"])

    async def _run_hotpotqa(self, example: BenchmarkExample) -> str:
        solutions = [(await self._answer(example.prompt))["answer"] for _ in range(3)]
        selected = await self._ensemble(solutions, example.prompt, qa=True)
        problem = f"Question: {example.prompt}\nBest answer: {selected}"
        return await self._custom(problem, self.bundle.prompts["FORMAT_ANSWER_PROMPT"])

    async def _run_gsm8k(self, example: BenchmarkExample) -> str:
        instruction = self.bundle.prompts["MATH_SOLVE_PROMPT"]
        solutions = [await self._custom(example.prompt, instruction) for _ in range(5)]
        selected = await self._ensemble(solutions, example.prompt)
        verification = await self._program(example.prompt, selected)
        return str(verification["output"]) if verification["output"] else selected

    async def _run_math(self, example: BenchmarkExample) -> str:
        programmed = await self._program(example.prompt)
        refined = await self._custom(
            example.prompt + f"\nCode output: {programmed['output']}",
            self.bundle.prompts["REFINE_ANSWER_PROMPT"],
        )
        detailed = await self._custom(
            example.prompt, self.bundle.prompts["DETAILED_SOLUTION_PROMPT"]
        )
        solutions = [refined, detailed]
        for _ in range(2):
            solutions.append(
                await self._custom(example.prompt, self.bundle.prompts["GENERATE_SOLUTION_PROMPT"])
            )
        return await self._ensemble(solutions, example.prompt)

    async def _run_humaneval(self, example: BenchmarkExample) -> str:
        assert example.entry_point is not None
        solution = await self._code(example.prompt, example.entry_point, "")
        tested = await self._test(example.prompt, solution, example.entry_point)
        if tested["result"]:
            return str(tested["solution"])
        return await self._code(
            example.prompt,
            example.entry_point,
            self.bundle.prompts["IMPROVE_CODE_PROMPT"],
        )

    async def _run_mbpp(self, example: BenchmarkExample) -> str:
        assert example.entry_point is not None
        instruction = self.bundle.prompts["CODE_GENERATE_PROMPT"]
        solutions = [
            await self._code(example.prompt, example.entry_point, instruction) for _ in range(3)
        ]
        selected = await self._ensemble(solutions, example.prompt)
        tested = await self._test(example.prompt, selected, example.entry_point)
        if tested["result"]:
            return str(tested["solution"])
        problem = (
            f"Problem: {example.prompt}\nFailed solution: {selected}\nError: {tested['solution']}"
        )
        return await self._custom(problem, self.bundle.prompts["FIX_CODE_PROMPT"])
