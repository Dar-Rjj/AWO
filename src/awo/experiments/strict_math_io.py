"""MATH-only IO protocol with an unambiguous final-answer schema."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from awo.baselines.io import IOPrompt, parse_io_response
from awo.baselines.models import BaselineResult
from awo.baselines.runner import score_baseline_result
from awo.benchmarks.data import BenchmarkExample
from awo.experiments.runner import (
    ExperimentRecord,
    ManualExperimentRunner,
    SampleCallLedger,
)
from awo.llm import OpenRouterClient

STRICT_MATH_IO_PROTOCOL = "controlled/strict-output-v1"
STRICT_MATH_IO_SYSTEM = (
    "Return exactly one valid JSON object and no surrounding markdown. "
    "It must have exactly one string field named 'solution'. "
    r'Use this exact JSON shape: {"solution":"\\boxed{<answer>}"}. '
    "Replace <answer> with the fully evaluated literal mathematical answer. "
    "The boxed value must not contain named variables, placeholders, assignments, "
    "or instructions to perform a calculation."
)
STRICT_MATH_IO_USER = (
    "{question}\n"
    "Solve the problem internally and return only the JSON object required by the system.\n"
)


def build_strict_math_io_prompt(example: BenchmarkExample) -> IOPrompt:
    """Build the registered MATH-only strict-output prompt."""

    if example.dataset.lower() != "math":
        raise ValueError("strict MATH IO only supports the math dataset")
    return IOPrompt(
        dataset="math",
        system=STRICT_MATH_IO_SYSTEM,
        user=STRICT_MATH_IO_USER.format(question=example.prompt),
        provenance=STRICT_MATH_IO_PROTOCOL,
    )


class StrictMathIOBaseline:
    """One-call MATH IO baseline isolated from generic code-task instructions."""

    method = "io"
    expected_calls = 1

    def __init__(self, client: OpenRouterClient) -> None:
        self.client = client

    def run(self, example: BenchmarkExample) -> BaselineResult:
        prompt = build_strict_math_io_prompt(example)
        response = self.client.chat(
            [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ],
            metadata={
                "method": self.method,
                "dataset": example.dataset,
                "sample_id": example.sample_id,
                "role": "generator",
                "prompt_provenance": prompt.provenance,
            },
        )
        return BaselineResult(
            method=self.method,
            dataset=example.dataset,
            sample_id=example.sample_id,
            prediction=parse_io_response(example.dataset, response.content),
            prompt_sha256=prompt.sha256,
            responses=(response,),
            protocol=STRICT_MATH_IO_PROTOCOL,
        )


class StrictMathIOExperimentRunner(ManualExperimentRunner):
    """Use the standard resumable ledger with the strict MATH IO baseline."""

    def _run_one(
        self,
        method: str,
        repeat: int,
        sample_index: int,
        example: BenchmarkExample,
    ) -> ExperimentRecord:
        if method != "io":
            raise ValueError("strict MATH IO runner only supports method='io'")
        key = self._key(method, repeat, example)
        ledger = SampleCallLedger(self.client, repeat=repeat)
        result: BaselineResult | None = None
        score_value = 0.0
        score_details: dict[str, Any] = {}
        status = "ok"
        error = None
        try:
            result = StrictMathIOBaseline(ledger).run(example)  # type: ignore[arg-type]
            score = score_baseline_result(example, result, sandbox=self.sandbox)
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


def strict_prompt_sha256(example: BenchmarkExample) -> str:
    """Expose the effective prompt hash for external audit checks."""

    prompt = build_strict_math_io_prompt(example)
    payload = json.dumps(
        {"system": prompt.system, "user": prompt.user},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
