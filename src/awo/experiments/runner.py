"""Resumable, per-sample runner for the six controlled manual baselines."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from awo.baselines import (
    CoTBaseline,
    CoTSCBaseline,
    IOBaseline,
    MedPromptBaseline,
    MultiPersonaBaseline,
    SelfRefineBaseline,
    score_baseline_result,
)
from awo.baselines.models import BaselineResult
from awo.benchmarks.data import BenchmarkExample
from awo.llm import ChatResult, OpenRouterClient
from awo.sandbox import DockerSandbox

MANUAL_METHODS = (
    "io",
    "cot",
    "cot_sc",
    "medprompt",
    "multi_persona",
    "self_refine",
)


def build_manual_baseline(
    method: str,
    client: OpenRouterClient,
    *,
    seed: int = 42,
) -> Any:
    factories = {
        "io": lambda: IOBaseline(client),
        "cot": lambda: CoTBaseline(client),
        "cot_sc": lambda: CoTSCBaseline(client),
        "medprompt": lambda: MedPromptBaseline(client, seed=seed),
        "multi_persona": lambda: MultiPersonaBaseline(client),
        "self_refine": lambda: SelfRefineBaseline(client),
    }
    try:
        return factories[method]()
    except KeyError as exc:
        raise ValueError(f"unsupported manual baseline: {method}") from exc


class SampleCallLedger:
    """Duck-typed client proxy that captures calls even if parsing later fails."""

    def __init__(
        self,
        client: OpenRouterClient,
        *,
        repeat: int,
    ) -> None:
        self.client = client
        self.repeat = repeat
        self.responses: list[ChatResult] = []

    def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ChatResult:
        enriched = {**dict(metadata or {}), "experiment_repeat": self.repeat}
        result = self.client.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            metadata=enriched,
        )
        self.responses.append(result)
        return result


@dataclass(frozen=True)
class ExperimentRecord:
    schema_version: int
    key: str
    dataset: str
    split: str
    method: str
    repeat: int
    sample_index: int
    sample_id: str
    status: str
    score: float
    score_details: dict[str, Any]
    call_count: int
    tokens: int
    cost: float
    latency_seconds: float
    providers: tuple[str, ...]
    result: dict[str, Any] | None
    error: str | None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ExperimentRecord:
        return cls(
            schema_version=int(value["schema_version"]),
            key=str(value["key"]),
            dataset=str(value["dataset"]),
            split=str(value["split"]),
            method=str(value["method"]),
            repeat=int(value["repeat"]),
            sample_index=int(value["sample_index"]),
            sample_id=str(value["sample_id"]),
            status=str(value["status"]),
            score=float(value["score"]),
            score_details=dict(value["score_details"]),
            call_count=int(value["call_count"]),
            tokens=int(value["tokens"]),
            cost=float(value["cost"]),
            latency_seconds=float(value["latency_seconds"]),
            providers=tuple(str(item) for item in value["providers"]),
            result=dict(value["result"]) if value.get("result") is not None else None,
            error=str(value["error"]) if value.get("error") is not None else None,
        )


class ExperimentStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.records_root = root / "records"

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def initialize(self, spec: Mapping[str, Any]) -> None:
        canonical = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(canonical).hexdigest()
        path = self.root / "spec.json"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("sha256") != digest:
                raise ValueError("experiment resume spec does not match existing run")
            return
        self._write_json(path, {**dict(spec), "sha256": digest})

    @staticmethod
    def _filename(key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest() + ".json"

    def get(self, key: str) -> ExperimentRecord | None:
        path = self.records_root / self._filename(key)
        if not path.exists():
            return None
        wrapper = json.loads(path.read_text(encoding="utf-8"))
        payload = wrapper.get("record")
        if not isinstance(payload, dict):
            raise ValueError("experiment record wrapper is invalid")
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if wrapper.get("sha256") != digest:
            raise ValueError("experiment record failed integrity check")
        record = ExperimentRecord.from_dict(payload)
        if record.key != key:
            raise ValueError("experiment record key hash collision")
        return record

    def save(self, record: ExperimentRecord) -> None:
        payload = asdict(record)
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self._write_json(
            self.records_root / self._filename(record.key),
            {"record": payload, "sha256": digest},
        )

    def save_summary(self, summary: Mapping[str, Any]) -> None:
        self._write_json(self.root / "summary.json", dict(summary))


class ManualExperimentRunner:
    """Run every requested method/repeat/sample exactly once with crash recovery."""

    def __init__(
        self,
        *,
        client: OpenRouterClient,
        examples: Sequence[BenchmarkExample],
        methods: Sequence[str],
        repeats: int,
        output_dir: Path,
        dataset_sha256: str,
        config_sha256: str,
        implementation_commit: str | None,
        seed: int = 42,
        sandbox: DockerSandbox | None = None,
    ) -> None:
        if not examples:
            raise ValueError("experiment examples cannot be empty")
        datasets = {example.dataset for example in examples}
        splits = {example.split for example in examples}
        if len(datasets) != 1 or len(splits) != 1:
            raise ValueError("runner requires examples from one dataset and split")
        unknown = set(methods) - set(MANUAL_METHODS)
        if unknown:
            raise ValueError(f"unsupported manual baselines: {sorted(unknown)}")
        if not methods or repeats <= 0:
            raise ValueError("methods cannot be empty and repeats must be positive")
        dataset = next(iter(datasets))
        if dataset in {"humaneval", "mbpp"} and sandbox is None:
            raise ValueError("code experiments require the Docker sandbox")
        self.client = client
        self.examples = tuple(examples)
        self.methods = tuple(methods)
        self.repeats = repeats
        self.seed = seed
        self.sandbox = sandbox
        self.store = ExperimentStore(output_dir)
        self.spec = {
            "schema_version": 1,
            "dataset": dataset,
            "split": next(iter(splits)),
            "dataset_sha256": dataset_sha256,
            "config_sha256": config_sha256,
            "implementation_commit": implementation_commit,
            "sample_ids": [example.sample_id for example in examples],
            "methods": list(methods),
            "repeats": repeats,
            "seed": seed,
            "cache_across_repeats": False,
        }

    @staticmethod
    def _key(method: str, repeat: int, example: BenchmarkExample) -> str:
        return f"{example.dataset}:{example.split}:{method}:{repeat}:{example.sample_id}"

    def _run_one(
        self,
        method: str,
        repeat: int,
        sample_index: int,
        example: BenchmarkExample,
    ) -> ExperimentRecord:
        key = self._key(method, repeat, example)
        ledger = SampleCallLedger(self.client, repeat=repeat)
        result: BaselineResult | None = None
        score_value = 0.0
        score_details: dict[str, Any] = {}
        status = "ok"
        error = None
        try:
            baseline = build_manual_baseline(
                method,
                ledger,  # type: ignore[arg-type]
                seed=self.seed,
            )
            result = baseline.run(example)
            score = score_baseline_result(
                example,
                result,
                sandbox=self.sandbox,
            )
            score_value = score.score
            score_details = score.details
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
        responses = ledger.responses
        return ExperimentRecord(
            schema_version=1,
            key=key,
            dataset=example.dataset,
            split=example.split,
            method=method,
            repeat=repeat,
            sample_index=sample_index,
            sample_id=example.sample_id,
            status=status,
            score=score_value,
            score_details=score_details,
            call_count=len(responses),
            tokens=sum(response.usage.total_tokens for response in responses),
            cost=sum(response.usage.cost or 0.0 for response in responses),
            latency_seconds=sum(response.latency_seconds for response in responses),
            providers=tuple(response.provider or "unknown" for response in responses),
            result=result.to_dict() if result is not None else None,
            error=error,
        )

    def run(self) -> dict[str, Any]:
        self.store.initialize(self.spec)
        records: list[ExperimentRecord] = []
        for repeat in range(1, self.repeats + 1):
            for method in self.methods:
                for sample_index, example in enumerate(self.examples):
                    key = self._key(method, repeat, example)
                    record = self.store.get(key)
                    if record is None:
                        record = self._run_one(method, repeat, sample_index, example)
                        self.store.save(record)
                    records.append(record)

        by_run: list[dict[str, Any]] = []
        for repeat in range(1, self.repeats + 1):
            for method in self.methods:
                selected = [
                    record
                    for record in records
                    if record.repeat == repeat and record.method == method
                ]
                by_run.append(
                    {
                        "method": method,
                        "repeat": repeat,
                        "score": fmean(record.score for record in selected),
                        "sample_count": len(selected),
                        "failure_count": sum(record.status != "ok" for record in selected),
                        "call_count": sum(record.call_count for record in selected),
                        "tokens": sum(record.tokens for record in selected),
                        "cost": sum(record.cost for record in selected),
                        "latency_seconds": sum(record.latency_seconds for record in selected),
                    }
                )
        summary = {
            "schema_version": 1,
            "spec": self.spec,
            "completed_records": len(records),
            "by_run": by_run,
            "totals": {
                "failures": sum(record.status != "ok" for record in records),
                "calls": sum(record.call_count for record in records),
                "tokens": sum(record.tokens for record in records),
                "cost": sum(record.cost for record in records),
            },
        }
        self.store.save_summary(summary)
        return summary
