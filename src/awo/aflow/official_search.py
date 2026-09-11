"""AFlow paper-commit search loop for Python-represented HumanEval workflows."""

from __future__ import annotations

import asyncio
import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

import numpy as np

from awo.aflow.official import (
    PAPER_COMMIT,
    OfficialCandidateGenerator,
    OfficialPythonCandidate,
    OfficialSandboxRuntime,
    initial_humaneval_candidate,
    render_graph_source,
)
from awo.benchmarks.code import score_code
from awo.benchmarks.data import BenchmarkExample
from awo.sandbox import PASS_MARKER


@dataclass(frozen=True)
class OfficialSearchConfig:
    """Defaults from upstream run.py and Optimizer."""

    sample: int = 4
    max_rounds: int = 20
    validation_rounds: int = 5
    check_convergence: bool = True
    max_concurrency: int = 8

    def __post_init__(self) -> None:
        if min(self.sample, self.validation_rounds, self.max_concurrency) <= 0:
            raise ValueError("sample, validation_rounds and max_concurrency must be positive")
        if self.max_rounds < 0:
            raise ValueError("max_rounds must be non-negative")


@dataclass(frozen=True)
class OfficialSearchSummary:
    stopped_reason: str
    generated_attempts: int
    final_round: int
    best_round: int
    best_score: float
    upstream_commit: str = PAPER_COMMIT


def original_mixed_probabilities(scores: list[float]) -> list[float]:
    """DataUtils._compute_probabilities with upstream alpha/lambda defaults."""

    values = np.asarray(scores, dtype=np.float64) * 100
    uniform = np.full(len(values), 1.0 / len(values), dtype=np.float64)
    weights = np.exp(0.2 * (values - np.max(values)))
    score_probability = weights / np.sum(weights)
    mixed = 0.3 * uniform + 0.7 * score_probability
    if not np.isclose(np.sum(mixed), 1.0):
        mixed = mixed / np.sum(mixed)
    return [float(item) for item in mixed]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


class OfficialAFlowSearch:
    """Preserve the upstream parent selection, experience and convergence loop."""

    def __init__(
        self,
        *,
        generator: OfficialCandidateGenerator,
        runtime: OfficialSandboxRuntime,
        examples: list[BenchmarkExample],
        output_dir: Path,
        config: OfficialSearchConfig,
    ) -> None:
        if not examples or any(item.dataset != "humaneval" for item in examples):
            raise ValueError("official Python search currently requires HumanEval examples")
        self.generator = generator
        self.runtime = runtime
        self.examples = examples
        self.output_dir = Path(output_dir)
        self.workflows = self.output_dir / "workflows"
        self.results_path = self.workflows / "results.json"
        self.config = config

    def _save_candidate(self, round_number: int, candidate: OfficialPythonCandidate) -> Path:
        directory = self.workflows / f"round_{round_number}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "__init__.py").write_text("", encoding="utf-8")
        (directory / "graph.py").write_text(
            render_graph_source(candidate, round_number), encoding="utf-8"
        )
        (directory / "prompt.py").write_text(candidate.prompt, encoding="utf-8")
        _write_json(directory / "candidate.json", candidate.to_dict())
        return directory

    def _candidate(self, round_number: int) -> OfficialPythonCandidate:
        payload = _read_json(self.workflows / f"round_{round_number}" / "candidate.json", None)
        if payload is None:
            raise RuntimeError(f"missing candidate for completed round {round_number}")
        return OfficialPythonCandidate.from_dict(payload)

    def _results(self) -> list[dict[str, Any]]:
        return list(_read_json(self.results_path, []))

    def _round_scores(self) -> dict[int, list[float]]:
        grouped: dict[int, list[float]] = defaultdict(list)
        for item in self._results():
            grouped[int(item["round"])].append(float(item["score"]))
        return dict(grouped)

    def _top_rounds(self) -> list[dict[str, float | int]]:
        means = {
            round_number: fmean(scores) for round_number, scores in self._round_scores().items()
        }
        ordered = sorted(means.items(), key=lambda item: item[1], reverse=True)
        unique: list[dict[str, float | int]] = []
        seen: set[int] = set()
        if 1 in means:
            unique.append({"round": 1, "score": means[1]})
            seen.add(1)
        for round_number, score in ordered:
            if round_number not in seen:
                unique.append({"round": round_number, "score": score})
                seen.add(round_number)
                if len(unique) >= self.config.sample:
                    break
        return unique

    def _select_parent(self) -> tuple[int, float, list[float]]:
        candidates = sorted(self._top_rounds(), key=lambda item: float(item["score"]), reverse=True)
        probabilities = original_mixed_probabilities([float(item["score"]) for item in candidates])
        index = int(np.random.choice(len(candidates), p=probabilities))
        selected = candidates[index]
        return int(selected["round"]), float(selected["score"]), probabilities

    def _processed_experience(self) -> dict[int, dict[str, Any]]:
        processed: dict[int, dict[str, Any]] = {}
        for directory in self.workflows.glob("round_*"):
            experience_path = directory / "experience.json"
            if not experience_path.exists():
                continue
            item = _read_json(experience_path, {})
            father = int(item["father node"])
            record = processed.setdefault(
                father, {"score": item["before"], "success": {}, "failure": {}}
            )
            branch = "success" if item["succeed"] else "failure"
            child_round = int(directory.name.removeprefix("round_"))
            record[branch][child_round] = {
                "modification": item["modification"],
                "score": item["after"],
            }
        serializable = {str(key): value for key, value in processed.items()}
        _write_json(self.workflows / "processed_experience.json", serializable)
        return processed

    @staticmethod
    def _format_experience(processed: dict[int, dict[str, Any]], parent: int) -> str:
        data = processed.get(parent)
        if not data:
            return f"No experience data found for round {parent}."
        experience = f"Original Score: {data['score']}\n"
        experience += "These are some conclusions drawn from experience:\n\n"
        for value in data["failure"].values():
            experience += (
                f"-Absolutely prohibit {value['modification']} (Score: {value['score']})\n"
            )
        for value in data["success"].values():
            experience += f"-Absolutely prohibit {value['modification']} \n"
        experience += (
            "\n\nNote: Take into account past failures and avoid repeating the same "
            "mistakes, as these failures indicate that these approaches are ineffective. "
            "You must fundamentally change your way of thinking, rather than simply using "
            "more advanced Python syntax like for, if, else, etc., or modifying the prompt."
        )
        return experience

    @staticmethod
    def _modification_is_new(
        processed: dict[int, dict[str, Any]], parent: int, modification: str
    ) -> bool:
        data = processed.get(parent)
        if not data:
            return True
        previous = [*data["failure"].values(), *data["success"].values()]
        return all(item["modification"] != modification for item in previous)

    def _sample_log(self, parent: int) -> str:
        path = self.workflows / f"round_{parent}" / "log.json"
        values = _read_json(path, [])
        if isinstance(values, dict):
            values = [values]
        if not values:
            return ""
        selected = random.sample(values, min(3, len(values)))
        return "".join(json.dumps(item, indent=4, ensure_ascii=False) + "\n\n" for item in selected)

    def _syntax_check(self, candidate: OfficialPythonCandidate, round_number: int) -> None:
        graph = render_graph_source(candidate, round_number)
        source = f"compile({graph!r}, 'graph.py', 'exec')\nprint({PASS_MARKER!r})"
        result = self.runtime.sandbox.run(source)
        if not result.passed:
            raise SyntaxError(
                f"generated graph.py failed sandbox compilation: {result.stderr[-1000:]}"
            )

    async def _evaluate_one(
        self,
        candidate: OfficialPythonCandidate,
        round_number: int,
        repeat: int,
        example: BenchmarkExample,
        semaphore: asyncio.Semaphore,
    ) -> dict[str, Any]:
        async with semaphore:
            try:
                assert example.entry_point is not None
                assert example.test_code is not None
                execution = await asyncio.to_thread(
                    self.runtime.execute,
                    candidate,
                    round_number=round_number,
                    problem=example.prompt,
                    entry_point=example.entry_point,
                    metadata={
                        "sample_id": example.sample_id,
                        "validation_repeat": repeat,
                    },
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

    async def _evaluate(self, candidate: OfficialPythonCandidate, round_number: int) -> float:
        results = [item for item in self._results() if int(item["round"]) != round_number]
        _write_json(self.results_path, results)
        directory = self.workflows / f"round_{round_number}"
        log: list[dict[str, Any]] = []
        scores: list[float] = []
        for repeat in range(self.config.validation_rounds):
            semaphore = asyncio.Semaphore(self.config.max_concurrency)
            rows = await asyncio.gather(
                *(
                    self._evaluate_one(candidate, round_number, repeat, example, semaphore)
                    for example in self.examples
                )
            )
            score = fmean(float(row["score"]) for row in rows)
            costs = [float(row["cost"]) for row in rows]
            upstream_total_cost = max(costs, default=0.0)
            results.append(
                {
                    "round": round_number,
                    "score": score,
                    "avg_cost": upstream_total_cost / len(rows),
                    "total_cost": upstream_total_cost,
                    "time": datetime.now(timezone.utc).isoformat(),
                }
            )
            _write_json(self.results_path, results)
            evaluation_path = directory / f"evaluation_{repeat + 1}.jsonl"
            evaluation_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
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
            _write_json(directory / "log.json", log)
            scores.append(score)
        _write_json(
            directory / "record.json",
            {
                "round": round_number,
                "scores": scores,
                "mean_score": fmean(scores),
                "std_score": pstdev(scores),
                "candidate_sha256": candidate.sha256,
                "protocol": "upstream-python/sandbox-relay",
                "upstream_commit": PAPER_COMMIT,
            },
        )
        return fmean(scores)

    def _convergence(self) -> tuple[bool, int | None, int | None]:
        grouped = self._round_scores()
        means = [fmean(grouped[key]) for key in sorted(grouped)]
        stds = [pstdev(grouped[key]) for key in sorted(grouped)]
        top_k = 3
        if len(means) < top_k + 1:
            return False, None, None
        count = 0
        previous_y = None
        for index in range(len(means)):
            indices = np.argsort(means[: index + 1])[::-1][:top_k]
            current_y = float(np.mean([means[item] for item in indices]))
            _ = float(np.sqrt(np.sum([stds[item] ** 2 for item in indices]) / top_k**2))
            if previous_y is not None:
                if abs(current_y - previous_y) <= 0:
                    count += 1
                    if count >= 5:
                        return True, index - 5 + 1, index
                else:
                    count = 0
            previous_y = current_y
        return False, None, None

    async def _generate_round(self, round_number: int) -> float:
        directory = self.workflows / f"round_{round_number}"
        directory.mkdir(parents=True, exist_ok=True)
        while True:
            parent_round, parent_score, probabilities = self._select_parent()
            processed = self._processed_experience()
            candidate = await self.generator.generate(
                parent=self._candidate(parent_round),
                parent_round=parent_round,
                score=parent_score,
                experience=self._format_experience(processed, parent_round),
                log=self._sample_log(parent_round),
                round_number=round_number,
            )
            if self._modification_is_new(processed, parent_round, candidate.modification):
                break
        _write_json(
            directory / "selection.json",
            {
                "parent_round": parent_round,
                "parent_score": parent_score,
                "probabilities": probabilities,
            },
        )
        self._save_candidate(round_number, candidate)
        self._syntax_check(candidate, round_number)
        experience = {
            "father node": parent_round,
            "modification": candidate.modification,
            "before": parent_score,
            "after": None,
            "succeed": None,
        }
        score = await self._evaluate(candidate, round_number)
        experience["after"] = score
        experience["succeed"] = bool(score > parent_score)
        _write_json(directory / "experience.json", experience)
        return score

    async def run(self) -> OfficialSearchSummary:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        state_path = self.output_dir / "state.json"
        state = _read_json(state_path, {})
        if not self._round_scores():
            initial = initial_humaneval_candidate()
            self._save_candidate(1, initial)
            self._syntax_check(initial, 1)
            await self._evaluate(initial, 1)
            state = {"generated_attempts": 0, "current_round": 1}
            _write_json(state_path, state)

        generated_attempts = int(state.get("generated_attempts", 0))
        current_round = int(state.get("current_round", max(self._round_scores())))
        stopped_reason = "max_rounds"
        while generated_attempts < self.config.max_rounds:
            next_round = current_round + 1
            try:
                await self._generate_round(next_round)
            except Exception as error:
                directory = self.workflows / f"round_{next_round}"
                directory.mkdir(parents=True, exist_ok=True)
                _write_json(
                    directory / "error.json",
                    {"type": type(error).__name__, "message": str(error)},
                )
            generated_attempts += 1
            current_round = next_round
            converged, convergence_round, final_round = self._convergence()
            state = {
                "generated_attempts": generated_attempts,
                "current_round": current_round,
                "converged": converged,
                "convergence_round": convergence_round,
                "convergence_final_round": final_round,
            }
            _write_json(state_path, state)
            await asyncio.sleep(5)
            if converged and self.config.check_convergence:
                stopped_reason = "convergence"
                break

        means = {
            round_number: fmean(scores) for round_number, scores in self._round_scores().items()
        }
        best_score = max(means.values())
        best_round = next(
            round_number for round_number in sorted(means) if means[round_number] == best_score
        )
        best = self._candidate(best_round)
        _write_json(self.output_dir / "best_candidate.json", best.to_dict())
        summary = OfficialSearchSummary(
            stopped_reason=stopped_reason,
            generated_attempts=generated_attempts,
            final_round=current_round,
            best_round=best_round,
            best_score=best_score,
        )
        _write_json(self.output_dir / "summary.json", asdict(summary))
        return summary
