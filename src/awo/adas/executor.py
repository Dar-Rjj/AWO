"""Six-task adapter from frozen benchmark examples to ADAS agent DAGs."""

from __future__ import annotations

import hashlib
import json

from awo.adas.dsl import ADASArchitecture, execute_architecture
from awo.baselines.models import BaselineResult
from awo.benchmarks.data import BenchmarkExample
from awo.llm import OpenRouterClient


def build_adas_task(example: BenchmarkExample) -> str:
    task = example.prompt
    if example.dataset in {"humaneval", "mbpp"}:
        if not example.entry_point:
            raise ValueError(f"missing code entry point for {example.sample_id}")
        task += (
            "\n\nReturn ONLY a complete runnable Python program. It must define "
            f"the entry-point function `{example.entry_point}`."
        )
    return task


async def run_architecture(
    candidate: ADASArchitecture,
    client: OpenRouterClient,
    example: BenchmarkExample,
) -> BaselineResult:
    """Run a frozen candidate without giving it references or test code."""

    task = build_adas_task(example)
    prediction, responses, trace = await execute_architecture(
        candidate,
        client,
        task=task,
        dataset=example.dataset,
        sample_id=example.sample_id,
    )
    prompt_hash = hashlib.sha256(
        json.dumps(
            {
                "architecture_sha256": candidate.architecture_sha256,
                "task": task,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return BaselineResult(
        method="adas",
        dataset=example.dataset,
        sample_id=example.sample_id,
        prediction=prediction,
        prompt_sha256=prompt_hash,
        responses=tuple(responses),
        protocol="protocol-compatible/official-meta-agent-safe-dag",
        artifacts={
            "architecture_name": candidate.name,
            "architecture_sha256": candidate.architecture_sha256,
            "trace": trace,
        },
    )
