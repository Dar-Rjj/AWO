"""Deterministic, resumable AFlow workflow search."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import random
from collections.abc import Awaitable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Callable, Protocol, Union

from awo.aflow.dsl import (
    OPERATOR_SETS,
    WorkflowCandidate,
    WorkflowDSLValidationError,
    initial_candidate,
    normalize_dataset,
    validate_candidate,
)


@dataclass(frozen=True)
class SearchConfig:
    max_generated_rounds: int = 20
    validation_repeats: int = 5
    top_k: int = 3
    early_stop_rounds: int = 5
    alpha: float = 0.4
    lambda_weight: float = 0.2
    seed: int = 42
    max_generation_attempts: int = 3

    def __post_init__(self) -> None:
        if self.max_generated_rounds < 0:
            raise ValueError("max_generated_rounds must be non-negative")
        if self.validation_repeats <= 0:
            raise ValueError("validation_repeats must be positive")
        if self.top_k <= 0 or self.early_stop_rounds <= 0:
            raise ValueError("top_k and early_stop_rounds must be positive")
        if self.alpha < 0 or not 0 <= self.lambda_weight <= 1:
            raise ValueError("alpha/lambda_weight are outside their valid ranges")
        if self.max_generation_attempts <= 0:
            raise ValueError("max_generation_attempts must be positive")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SearchConfig:
        return cls(
            max_generated_rounds=int(
                value.get("max_generated_rounds", value.get("max_rounds", 20))
            ),
            validation_repeats=int(
                value.get("validation_repeats", value.get("validation_rounds", 5))
            ),
            top_k=int(value.get("top_k", 3)),
            early_stop_rounds=int(value.get("early_stop_rounds", 5)),
            alpha=float(value.get("alpha", 0.4)),
            lambda_weight=float(value.get("lambda_weight", value.get("lambda", 0.2))),
            seed=int(value.get("seed", 42)),
            max_generation_attempts=int(value.get("max_generation_attempts", 3)),
        )


@dataclass(frozen=True)
class ValidationObservation:
    score: float
    cost: float = 0.0
    tokens: int = 0
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RoundRecord:
    round: int
    parent_round: int | None
    candidate_sha256: str
    modification: str
    scores: tuple[float, ...]
    mean_score: float
    std_score: float
    total_cost: float
    total_tokens: int
    generation_attempts: int
    improved: bool
    selection_probability: float | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RoundRecord:
        return cls(
            round=int(value["round"]),
            parent_round=(
                int(value["parent_round"]) if value.get("parent_round") is not None else None
            ),
            candidate_sha256=str(value["candidate_sha256"]),
            modification=str(value["modification"]),
            scores=tuple(float(item) for item in value["scores"]),
            mean_score=float(value["mean_score"]),
            std_score=float(value["std_score"]),
            total_cost=float(value["total_cost"]),
            total_tokens=int(value["total_tokens"]),
            generation_attempts=int(value["generation_attempts"]),
            improved=bool(value["improved"]),
            selection_probability=(
                float(value["selection_probability"])
                if value.get("selection_probability") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class SearchContext:
    dataset: str
    generated_index: int
    round: int
    parent: WorkflowCandidate
    parent_record: RoundRecord
    parent_pool: tuple[RoundRecord, ...]
    selection_probabilities: tuple[float, ...]
    experience: str
    allowed_operators: tuple[str, ...]


@dataclass(frozen=True)
class SearchSummary:
    stopped_reason: str
    completed_generated_rounds: int
    best_round: int
    best_score: float
    records: tuple[RoundRecord, ...]


class CandidateGenerator(Protocol):
    async def generate(self, context: SearchContext) -> WorkflowCandidate:
        """Generate one candidate for the selected parent."""


Evaluator = Callable[
    [WorkflowCandidate, int, int],
    Union[ValidationObservation, Awaitable[ValidationObservation]],  # noqa: UP007
]


def mixed_probabilities(
    scores: list[float] | tuple[float, ...],
    *,
    alpha: float,
    lambda_weight: float,
) -> tuple[float, ...]:
    """Paper/upstream score-softmax mixed with a uniform distribution."""

    if not scores:
        raise ValueError("scores cannot be empty")
    if alpha < 0 or not 0 <= lambda_weight <= 1:
        raise ValueError("invalid alpha/lambda_weight")
    scaled = [float(score) * 100.0 for score in scores]
    peak = max(scaled)
    weights = [math.exp(alpha * (score - peak)) for score in scaled]
    denominator = sum(weights)
    softmax = [weight / denominator for weight in weights]
    uniform = 1.0 / len(scores)
    return tuple(
        lambda_weight * uniform + (1.0 - lambda_weight) * probability for probability in softmax
    )


def select_parent_pool(
    records: list[RoundRecord] | tuple[RoundRecord, ...],
    top_k: int,
) -> tuple[RoundRecord, ...]:
    if not records:
        raise ValueError("records cannot be empty")
    initial = min(records, key=lambda item: item.round)
    generated = sorted(
        (item for item in records if item.round != initial.round),
        key=lambda item: (-item.mean_score, item.round),
    )
    return (initial, *generated[:top_k])


class SearchStore:
    """Atomic, human-auditable state for interruption-safe search."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.rounds_root = root / "rounds"

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    @staticmethod
    def _config_payload(dataset: str, config: SearchConfig) -> dict[str, Any]:
        return {"dataset": dataset, "search": asdict(config)}

    def initialize(self, dataset: str, config: SearchConfig) -> None:
        payload = self._config_payload(dataset, config)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        config_hash = hashlib.sha256(encoded).hexdigest()
        config_file = self.root / "config.json"
        if config_file.exists():
            existing = json.loads(config_file.read_text(encoding="utf-8"))
            if existing.get("sha256") != config_hash:
                raise ValueError("resume configuration does not match existing search")
            return
        self._write_json(config_file, {**payload, "sha256": config_hash})

    def save_candidate(self, round_number: int, candidate: WorkflowCandidate) -> None:
        self._write_json(
            self.rounds_root / f"round_{round_number}" / "candidate.json",
            candidate.to_dict(),
        )

    def save_evaluations(
        self, round_number: int, observations: list[ValidationObservation]
    ) -> None:
        path = self.rounds_root / f"round_{round_number}" / "evaluations.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".jsonl.tmp")
        temporary.write_text(
            "".join(
                json.dumps(asdict(item), ensure_ascii=False, sort_keys=True) + "\n"
                for item in observations
            ),
            encoding="utf-8",
        )
        temporary.replace(path)

    def save_record(self, record: RoundRecord) -> None:
        self._write_json(
            self.rounds_root / f"round_{record.round}" / "record.json",
            asdict(record),
        )

    def save_selection(
        self,
        *,
        round_number: int,
        generated_index: int,
        parent_round: int,
        pool: tuple[RoundRecord, ...],
        probabilities: tuple[float, ...],
        seed: int,
    ) -> None:
        self._write_json(
            self.rounds_root / f"round_{round_number}" / "selection.json",
            {
                "generated_index": generated_index,
                "parent_round": parent_round,
                "pool_rounds": [item.round for item in pool],
                "probabilities": list(probabilities),
                "seed": seed,
            },
        )

    def save_state(self, value: Mapping[str, Any]) -> None:
        self._write_json(self.root / "state.json", dict(value))

    def load(self) -> tuple[list[WorkflowCandidate], list[RoundRecord]]:
        candidates: list[WorkflowCandidate] = []
        records: list[RoundRecord] = []
        if not self.rounds_root.exists():
            return candidates, records
        round_dirs = sorted(
            self.rounds_root.glob("round_*"),
            key=lambda path: int(path.name.removeprefix("round_")),
        )
        for expected_round, directory in enumerate(round_dirs, start=1):
            actual_round = int(directory.name.removeprefix("round_"))
            candidate_file = directory / "candidate.json"
            record_file = directory / "record.json"
            if not candidate_file.exists() or not record_file.exists():
                if any(
                    (later / "candidate.json").exists() and (later / "record.json").exists()
                    for later in round_dirs[expected_round:]
                ):
                    raise ValueError("completed search rounds are not contiguous")
                continue
            if actual_round != expected_round:
                raise ValueError("completed search rounds are not contiguous")
            candidate = WorkflowCandidate.from_dict(
                json.loads(candidate_file.read_text(encoding="utf-8"))
            )
            record = RoundRecord.from_dict(json.loads(record_file.read_text(encoding="utf-8")))
            if record.round != actual_round or record.candidate_sha256 != candidate.sha256:
                raise ValueError(f"round {actual_round} failed candidate integrity checks")
            candidates.append(candidate)
            records.append(record)
        return candidates, records


class AFlowSearch:
    """Run the preregistered initial + generated-round search protocol."""

    def __init__(
        self,
        *,
        dataset: str,
        config: SearchConfig,
        generator: CandidateGenerator,
        evaluator: Evaluator,
        output_dir: Path,
    ) -> None:
        self.dataset = normalize_dataset(dataset)
        self.config = config
        self.generator = generator
        self.evaluator = evaluator
        self.store = SearchStore(output_dir)

    async def _evaluate(
        self,
        candidate: WorkflowCandidate,
        round_number: int,
    ) -> list[ValidationObservation]:
        observations: list[ValidationObservation] = []
        for repeat in range(self.config.validation_repeats):
            try:
                value = self.evaluator(candidate, repeat, round_number)
                if inspect.isawaitable(value):
                    value = await value
                if not isinstance(value, ValidationObservation):
                    raise TypeError("evaluator must return ValidationObservation")
                if not math.isfinite(value.score):
                    raise ValueError("validation score must be finite")
                observations.append(value)
            except Exception as exc:
                observations.append(
                    ValidationObservation(
                        score=0.0,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
        return observations

    @staticmethod
    def _experience(
        parent_round: int,
        records: list[RoundRecord],
        candidates: list[WorkflowCandidate],
    ) -> str:
        messages: list[str] = []
        for candidate, record in zip(candidates, records):
            if candidate.parent_round != parent_round:
                continue
            verdict = "REUSE" if record.improved else "AVOID"
            messages.append(
                f"{verdict}: {candidate.modification} (validation={record.mean_score:.6f})"
            )
        return "\n".join(messages) or "No child workflow has been tried for this parent."

    async def _record_round(
        self,
        candidate: WorkflowCandidate,
        *,
        round_number: int,
        parent_round: int | None,
        attempts: int,
        prior_best: float,
        selection_probability: float | None = None,
    ) -> RoundRecord:
        observations = await self._evaluate(candidate, round_number)
        scores = tuple(item.score for item in observations)
        mean_score = fmean(scores)
        record = RoundRecord(
            round=round_number,
            parent_round=parent_round,
            candidate_sha256=candidate.sha256,
            modification=candidate.modification,
            scores=scores,
            mean_score=mean_score,
            std_score=pstdev(scores),
            total_cost=sum(item.cost for item in observations),
            total_tokens=sum(item.tokens for item in observations),
            generation_attempts=attempts,
            improved=mean_score > prior_best,
            selection_probability=selection_probability,
        )
        self.store.save_candidate(round_number, candidate)
        self.store.save_evaluations(round_number, observations)
        self.store.save_record(record)
        return record

    async def run(self) -> SearchSummary:
        self.store.initialize(self.dataset, self.config)
        candidates, records = self.store.load()
        if any(len(record.scores) != self.config.validation_repeats for record in records):
            raise ValueError("stored validation repeat count does not match search config")
        if not records:
            blank = initial_candidate(self.dataset)
            initial_record = await self._record_round(
                blank,
                round_number=1,
                parent_round=None,
                attempts=0,
                prior_best=float("-inf"),
            )
            candidates.append(blank)
            records.append(initial_record)

        generated_done = len(records) - 1
        best_score = max(record.mean_score for record in records)
        best_round = min(record.round for record in records if record.mean_score == best_score)
        no_improvement = 0
        for record in reversed(records[1:]):
            if record.improved:
                break
            no_improvement += 1
        stopped_reason = "max_generated_rounds"

        for generated_index in range(generated_done + 1, self.config.max_generated_rounds + 1):
            if no_improvement >= self.config.early_stop_rounds:
                stopped_reason = "early_stop"
                break
            pool = select_parent_pool(records, self.config.top_k)
            probabilities = mixed_probabilities(
                [item.mean_score for item in pool],
                alpha=self.config.alpha,
                lambda_weight=self.config.lambda_weight,
            )
            rng = random.Random(f"{self.config.seed}:{generated_index}")
            parent_record = rng.choices(pool, weights=probabilities, k=1)[0]
            parent_position = next(
                index for index, item in enumerate(pool) if item.round == parent_record.round
            )
            parent_candidate = candidates[
                next(
                    index for index, item in enumerate(records) if item.round == parent_record.round
                )
            ]
            round_number = generated_index + 1
            self.store.save_selection(
                round_number=round_number,
                generated_index=generated_index,
                parent_round=parent_record.round,
                pool=pool,
                probabilities=probabilities,
                seed=self.config.seed,
            )
            context = SearchContext(
                dataset=self.dataset,
                generated_index=generated_index,
                round=round_number,
                parent=parent_candidate,
                parent_record=parent_record,
                parent_pool=pool,
                selection_probabilities=probabilities,
                experience=self._experience(parent_record.round, records, candidates),
                allowed_operators=tuple(sorted(OPERATOR_SETS[self.dataset])),
            )
            candidate: WorkflowCandidate | None = None
            generation_error: Exception | None = None
            attempts = 0
            known_workflows = {item.workflow_sha256 for item in candidates}
            for attempt_number in range(1, self.config.max_generation_attempts + 1):
                attempts = attempt_number
                try:
                    proposed = await self.generator.generate(context)
                    if proposed.parent_round != parent_record.round:
                        proposed = WorkflowCandidate(
                            modification=proposed.modification,
                            graph=proposed.graph,
                            prompts=proposed.prompts,
                            parent_round=parent_record.round,
                            metadata=proposed.metadata,
                        )
                    validate_candidate(proposed, self.dataset)
                    if proposed.workflow_sha256 in known_workflows:
                        raise WorkflowDSLValidationError("optimizer proposed a duplicate workflow")
                    candidate = proposed
                    break
                except Exception as exc:
                    generation_error = exc
            if candidate is None:
                raise RuntimeError(
                    f"candidate generation failed after {attempts} attempts"
                ) from generation_error

            previous_best = best_score
            record = await self._record_round(
                candidate,
                round_number=round_number,
                parent_round=parent_record.round,
                attempts=attempts,
                prior_best=previous_best,
                selection_probability=probabilities[parent_position],
            )
            candidates.append(candidate)
            records.append(record)
            if record.mean_score > best_score:
                best_score = record.mean_score
                best_round = record.round
                no_improvement = 0
            else:
                no_improvement += 1
            self.store.save_state(
                {
                    "best_round": best_round,
                    "best_score": best_score,
                    "completed_generated_rounds": len(records) - 1,
                    "no_improvement_rounds": no_improvement,
                }
            )
        else:
            stopped_reason = "max_generated_rounds"

        if no_improvement >= self.config.early_stop_rounds:
            stopped_reason = "early_stop"
        self.store.save_state(
            {
                "best_round": best_round,
                "best_score": best_score,
                "completed_generated_rounds": len(records) - 1,
                "no_improvement_rounds": no_improvement,
                "stopped_reason": stopped_reason,
            }
        )
        return SearchSummary(
            stopped_reason=stopped_reason,
            completed_generated_rounds=len(records) - 1,
            best_round=best_round,
            best_score=best_score,
            records=tuple(records),
        )
