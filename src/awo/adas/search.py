"""Resumable Meta Agent Search over controlled ADAS architectures."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, Union

from awo.adas.dsl import (
    ADASArchitecture,
    ADASValidationError,
    initial_archive,
    validate_architecture,
)


@dataclass(frozen=True)
class ADASConfig:
    generations: int = 30
    max_generation_attempts: int = 3

    def __post_init__(self) -> None:
        if self.generations < 0:
            raise ValueError("generations must be non-negative")
        if self.max_generation_attempts <= 0:
            raise ValueError("max_generation_attempts must be positive")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ADASConfig:
        return cls(
            generations=int(value.get("generations", value.get("max_rounds", 30))),
            max_generation_attempts=int(value.get("max_generation_attempts", 3)),
        )


@dataclass(frozen=True)
class ADASObservation:
    score: float
    cost: float = 0.0
    tokens: int = 0
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ADASRecord:
    archive_index: int
    generation: str | int
    candidate_sha256: str
    architecture_sha256: str
    fitness: float
    evaluation_cost: float
    evaluation_tokens: int
    meta_cost: float
    meta_tokens: int
    generation_attempts: int
    error: str | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ADASRecord:
        generation_value = value["generation"]
        generation: str | int = (
            int(generation_value) if isinstance(generation_value, int) else str(generation_value)
        )
        return cls(
            archive_index=int(value["archive_index"]),
            generation=generation,
            candidate_sha256=str(value["candidate_sha256"]),
            architecture_sha256=str(value["architecture_sha256"]),
            fitness=float(value["fitness"]),
            evaluation_cost=float(value["evaluation_cost"]),
            evaluation_tokens=int(value["evaluation_tokens"]),
            meta_cost=float(value["meta_cost"]),
            meta_tokens=int(value["meta_tokens"]),
            generation_attempts=int(value["generation_attempts"]),
            error=str(value["error"]) if value.get("error") is not None else None,
        )


@dataclass(frozen=True)
class ADASSummary:
    completed_initial: int
    completed_generations: int
    best_archive_index: int
    best_fitness: float
    records: tuple[ADASRecord, ...]


class MetaGenerator(Protocol):
    async def generate(
        self,
        *,
        dataset: str,
        generation: int,
        archive: Sequence[Mapping[str, Any]],
        previous: ADASArchitecture | None,
    ) -> ADASArchitecture:
        """Generate one reflected ADAS candidate."""


ADASEvaluator = Callable[
    [ADASArchitecture, Union[str, int]],  # noqa: UP007
    Union[ADASObservation, Awaitable[ADASObservation]],  # noqa: UP007
]


class ADASStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.entries_root = root / "entries"

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def initialize(
        self,
        dataset: str,
        config: ADASConfig,
        run_fingerprint: Mapping[str, Any],
    ) -> None:
        payload = {
            "dataset": dataset,
            "search": asdict(config),
            "run_fingerprint": dict(run_fingerprint),
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        path = self.root / "config.json"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("sha256") != digest:
                raise ValueError("resume configuration does not match ADAS search")
            return
        self._write_json(path, {**payload, "sha256": digest})

    def save(
        self,
        candidates: list[ADASArchitecture],
        records: list[ADASRecord],
    ) -> None:
        candidate = candidates[-1]
        record = records[-1]
        entry_root = self.entries_root / f"entry_{record.archive_index:03d}"
        self._write_json(entry_root / "candidate.json", candidate.to_dict())
        self._write_json(entry_root / "record.json", asdict(record))
        archive = [
            {"candidate": item.to_dict(), **asdict(item_record)}
            for item, item_record in zip(candidates, records)
        ]
        self._write_json(self.root / "archive.json", archive)

    def save_state(self, value: Mapping[str, Any]) -> None:
        self._write_json(self.root / "state.json", dict(value))

    def load(self) -> tuple[list[ADASArchitecture], list[ADASRecord]]:
        archive_path = self.root / "archive.json"
        if not archive_path.exists():
            return [], []
        values = json.loads(archive_path.read_text(encoding="utf-8"))
        if not isinstance(values, list):
            raise ValueError("ADAS archive must be a list")
        candidates: list[ADASArchitecture] = []
        records: list[ADASRecord] = []
        for expected_index, value in enumerate(values):
            candidate = ADASArchitecture.from_dict(value["candidate"])
            record = ADASRecord.from_dict(value)
            if (
                record.archive_index != expected_index
                or record.candidate_sha256 != candidate.sha256
                or record.architecture_sha256 != candidate.architecture_sha256
            ):
                raise ValueError(f"ADAS archive entry {expected_index} failed integrity checks")
            validate_architecture(candidate)
            candidates.append(candidate)
            records.append(record)
        return candidates, records


class ADASSearch:
    """Evaluate seven seeds, then run the official 30-generation archive loop."""

    def __init__(
        self,
        *,
        dataset: str,
        config: ADASConfig,
        generator: MetaGenerator,
        evaluator: ADASEvaluator,
        output_dir: Path,
        run_fingerprint: Mapping[str, Any] | None = None,
    ) -> None:
        if dataset not in {"hotpotqa", "drop", "humaneval", "mbpp", "gsm8k", "math"}:
            raise ValueError(f"unsupported ADAS dataset: {dataset}")
        self.dataset = dataset
        self.config = config
        self.generator = generator
        self.evaluator = evaluator
        self.store = ADASStore(output_dir)
        self.run_fingerprint = dict(run_fingerprint or {})

    async def _evaluate(
        self, candidate: ADASArchitecture, generation: str | int
    ) -> ADASObservation:
        try:
            value = self.evaluator(candidate, generation)
            if inspect.isawaitable(value):
                value = await value
            if not isinstance(value, ADASObservation):
                raise TypeError("ADAS evaluator must return ADASObservation")
            if not math.isfinite(value.score):
                raise ValueError("ADAS fitness must be finite")
            return value
        except Exception as exc:
            return ADASObservation(
                score=0.0,
                error=f"{type(exc).__name__}: {exc}",
            )

    @staticmethod
    def _archive_view(
        candidates: list[ADASArchitecture], records: list[ADASRecord]
    ) -> list[dict[str, Any]]:
        return [
            {
                "generation": record.generation,
                "fitness": record.fitness,
                "candidate": candidate.to_dict(),
            }
            for candidate, record in zip(candidates, records)
        ]

    async def _append(
        self,
        candidates: list[ADASArchitecture],
        records: list[ADASRecord],
        candidate: ADASArchitecture,
        *,
        generation: str | int,
        attempts: int,
    ) -> None:
        observation = await self._evaluate(candidate, generation)
        metadata = candidate.metadata
        record = ADASRecord(
            archive_index=len(records),
            generation=generation,
            candidate_sha256=candidate.sha256,
            architecture_sha256=candidate.architecture_sha256,
            fitness=observation.score,
            evaluation_cost=observation.cost,
            evaluation_tokens=observation.tokens,
            meta_cost=float(metadata.get("cost", 0.0)),
            meta_tokens=int(metadata.get("tokens", 0)),
            generation_attempts=attempts,
            error=observation.error,
        )
        candidates.append(candidate)
        records.append(record)
        self.store.save(candidates, records)

    async def run(self) -> ADASSummary:
        self.store.initialize(self.dataset, self.config, self.run_fingerprint)
        candidates, records = self.store.load()
        seeds = initial_archive()
        completed_initial = 0
        for record in records:
            if not isinstance(record.generation, str):
                break
            completed_initial += 1
        if any(isinstance(record.generation, str) for record in records[completed_initial:]):
            raise ValueError("ADAS initial architectures must precede generated entries")
        if completed_initial > len(seeds):
            raise ValueError("ADAS archive contains too many initial architectures")
        for index in range(completed_initial):
            if candidates[index].architecture_sha256 != seeds[index].architecture_sha256:
                raise ValueError("stored ADAS initial archive does not match frozen seeds")
        for seed_index in range(completed_initial, len(seeds)):
            await self._append(
                candidates,
                records,
                seeds[seed_index],
                generation=f"initial-{seed_index + 1}",
                attempts=0,
            )

        generated_records = [record for record in records if isinstance(record.generation, int)]
        if [record.generation for record in generated_records] != list(
            range(1, len(generated_records) + 1)
        ):
            raise ValueError("stored ADAS generations are not contiguous")
        generated_done = len(generated_records)
        previous = candidates[records.index(generated_records[-1])] if generated_records else None
        for generation in range(generated_done + 1, self.config.generations + 1):
            known = {candidate.architecture_sha256 for candidate in candidates}
            candidate: ADASArchitecture | None = None
            generation_error: Exception | None = None
            attempts = 0
            for attempt_number in range(1, self.config.max_generation_attempts + 1):
                attempts = attempt_number
                try:
                    proposed = await self.generator.generate(
                        dataset=self.dataset,
                        generation=generation,
                        archive=self._archive_view(candidates, records),
                        previous=previous,
                    )
                    expected_parent = generation - 1 if generation > 1 else None
                    if proposed.parent_generation != expected_parent:
                        proposed = ADASArchitecture(
                            thought=proposed.thought,
                            name=proposed.name,
                            architecture=proposed.architecture,
                            parent_generation=expected_parent,
                            metadata=proposed.metadata,
                        )
                    validate_architecture(proposed)
                    if proposed.architecture_sha256 in known:
                        raise ADASValidationError("meta-agent proposed a duplicate architecture")
                    candidate = proposed
                    break
                except Exception as exc:
                    generation_error = exc
            if candidate is None:
                raise RuntimeError(
                    f"ADAS generation {generation} failed after {attempts} attempts"
                ) from generation_error
            await self._append(
                candidates,
                records,
                candidate,
                generation=generation,
                attempts=attempts,
            )
            previous = candidate
            best = max(records, key=lambda record: (record.fitness, -record.archive_index))
            self.store.save_state(
                {
                    "best_archive_index": best.archive_index,
                    "best_fitness": best.fitness,
                    "completed_generations": generation,
                    "completed_initial": len(seeds),
                }
            )

        best = max(records, key=lambda record: (record.fitness, -record.archive_index))
        self.store.save_state(
            {
                "best_archive_index": best.archive_index,
                "best_fitness": best.fitness,
                "completed_generations": len(
                    [record for record in records if isinstance(record.generation, int)]
                ),
                "completed_initial": len(seeds),
                "status": "complete",
            }
        )
        return ADASSummary(
            completed_initial=len(seeds),
            completed_generations=len(
                [record for record in records if isinstance(record.generation, int)]
            ),
            best_archive_index=best.archive_index,
            best_fitness=best.fitness,
            records=tuple(records),
        )
