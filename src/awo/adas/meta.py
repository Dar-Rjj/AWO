"""Three-call ADAS meta-agent generation adapted to a safe architecture DSL."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from typing import Any

from awo.adas.dsl import ADASArchitecture, validate_architecture
from awo.llm import ChatResult, OpenRouterClient

TASK_DESCRIPTIONS = {
    "hotpotqa": (
        "multi-hop question answering over supplied context; return a short answer "
        "without explanation"
    ),
    "drop": (
        "reading comprehension with discrete reasoning over a passage; return the "
        "short answer span, number, list, or date"
    ),
    "humaneval": (
        "Python function synthesis; the final answer must be only a complete program "
        "defining the requested entry point"
    ),
    "mbpp": (
        "Python function synthesis from a natural-language specification; the final "
        "answer must be only a complete program defining the requested entry point"
    ),
    "gsm8k": ("grade-school mathematical reasoning; return the final numeric answer clearly"),
    "math": (
        "competition mathematics; derive carefully and put the final answer in "
        "standard mathematical form"
    ),
}

META_SYSTEM = "You are a helpful ADAS meta-agent. Return exactly one well-formed JSON object."

META_PROMPT = """# Overview
You are an expert machine-learning researcher designing an agentic architecture for:
{task_description}

The discovered architecture archive is below. Fitness is the validation score; your
goal is to maximize it. Do not use test examples or test results.

{archive}

# Controlled architecture language
Return exactly the keys "thought", "name", and "architecture".
"architecture" must be:
{{
  "schema_version": 1,
  "nodes": [
    {{
      "id": "lowercase_identifier",
      "role": "literal role description",
      "temperature": 0.5,
      "output_fields": ["thinking", "answer"],
      "instruction": "literal instruction",
      "inputs": ["$task", "$earlier_node.field"]
    }}
  ],
  "output": "$node.field"
}}

Use 1-12 nodes. A node may reference only "$task" and fields declared by earlier
nodes. Use 1-5 unique lowercase output fields, 1-30 inputs, and temperature 0-2.
The output must reference a declared field. The architecture is interpreted as
data: do not emit Python, imports, tools, network calls, file operations, loops,
conditions, templates, or runtime placeholders.

In "thought", explain the insight, overall idea, and implementation. Propose one
interesting architecture that differs materially from the archive.
"""

REFLECTION_ONE = """Carefully review the proposed architecture:
1. Compare it with every archived architecture and assess interestingness.
2. Check every node schema, backward reference, declared output field, and final output.
3. Improve effectiveness without changing more than needed.

Return exactly "reflection", "thought", "name", and "architecture". The architecture
must be a complete corrected object in the same controlled language.
"""

REFLECTION_TWO = """Revise the architecture one final time. Check that it uses 1-12 nodes,
contains no generated Python or unsupported control flow, references only earlier declared
fields, and has one valid final output. Return exactly "reflection", "thought", "name",
and "architecture"; repeat the final thought/name and include the complete final object.
"""


def _json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    value = json.loads(stripped)
    if not isinstance(value, dict):
        raise ValueError("ADAS meta-agent response must be one JSON object")
    return value


def _candidate_from_payload(
    payload: Mapping[str, Any],
    *,
    parent_generation: int | None,
    metadata: Mapping[str, Any] | None = None,
    reflected: bool,
    validate: bool = True,
) -> ADASArchitecture:
    expected = (
        {"reflection", "thought", "name", "architecture"}
        if reflected
        else {"thought", "name", "architecture"}
    )
    if set(payload) != expected:
        raise ValueError(f"ADAS meta-agent fields must be exactly {sorted(expected)}")
    candidate = ADASArchitecture(
        thought=str(payload["thought"]),
        name=str(payload["name"]),
        architecture=dict(payload["architecture"]),
        parent_generation=parent_generation,
        metadata=dict(metadata or {}),
    )
    if validate:
        validate_architecture(candidate)
    return candidate


def render_archive(archive: Sequence[Mapping[str, Any]]) -> str:
    compact = []
    for item in archive:
        compact.append(
            {
                "generation": item["generation"],
                "fitness": item["fitness"],
                "name": item["candidate"]["name"],
                "thought": item["candidate"]["thought"],
                "architecture": item["candidate"]["architecture"],
            }
        )
    return json.dumps(compact, ensure_ascii=False, sort_keys=True)


class ADASMetaGenerator:
    """Official proposal + two reflections, all through the unified client."""

    def __init__(self, client: OpenRouterClient) -> None:
        self.client = client

    async def _call(
        self,
        messages: list[dict[str, str]],
        *,
        dataset: str,
        generation: int,
        phase: str,
    ) -> ChatResult:
        return await asyncio.to_thread(
            self.client.chat,
            messages,
            temperature=0.8,
            metadata={
                "method": "adas",
                "role": "meta_agent",
                "dataset": dataset,
                "generation": generation,
                "phase": phase,
            },
        )

    async def generate(
        self,
        *,
        dataset: str,
        generation: int,
        archive: Sequence[Mapping[str, Any]],
        previous: ADASArchitecture | None,
    ) -> ADASArchitecture:
        if dataset not in TASK_DESCRIPTIONS:
            raise ValueError(f"unsupported ADAS dataset: {dataset}")
        prompt = META_PROMPT.format(
            task_description=TASK_DESCRIPTIONS[dataset],
            archive=render_archive(archive),
        )
        messages = [
            {"role": "system", "content": META_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        proposal_response = await self._call(
            messages, dataset=dataset, generation=generation, phase="proposal"
        )
        proposal_payload = _json_object(proposal_response.content)
        _candidate_from_payload(
            proposal_payload,
            parent_generation=generation - 1 if generation > 1 else None,
            reflected=False,
            validate=False,
        )

        prior = (
            "\nPrevious generated architecture:\n"
            + json.dumps(previous.to_dict(), ensure_ascii=False, sort_keys=True)
            if previous is not None
            else ""
        )
        messages.extend(
            [
                {"role": "assistant", "content": proposal_response.content},
                {"role": "user", "content": prior + "\n" + REFLECTION_ONE},
            ]
        )
        reflection_one = await self._call(
            messages, dataset=dataset, generation=generation, phase="reflection_1"
        )
        reflection_one_payload = _json_object(reflection_one.content)
        _candidate_from_payload(
            reflection_one_payload,
            parent_generation=generation - 1 if generation > 1 else None,
            reflected=True,
            validate=False,
        )

        messages.extend(
            [
                {"role": "assistant", "content": reflection_one.content},
                {"role": "user", "content": REFLECTION_TWO},
            ]
        )
        reflection_two = await self._call(
            messages, dataset=dataset, generation=generation, phase="reflection_2"
        )
        final_payload = _json_object(reflection_two.content)
        responses = (proposal_response, reflection_one, reflection_two)
        return _candidate_from_payload(
            final_payload,
            parent_generation=generation - 1 if generation > 1 else None,
            metadata={
                "protocol": "official-three-call-reflection/safe-dag",
                "request_sha256": [item.request_sha256 for item in responses],
                "providers": [item.provider for item in responses],
                "tokens": sum(item.usage.total_tokens for item in responses),
                "cost": sum(item.usage.cost or 0.0 for item in responses),
            },
            reflected=True,
        )
