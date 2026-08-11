"""Safe agent-DAG replacement for ADAS-generated Python forward functions."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from awo.llm import ChatResult, OpenRouterClient


class ADASValidationError(ValueError):
    """Raised when an ADAS architecture is outside the controlled language."""


_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")
_REFERENCE = re.compile(r"^\$(task|[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*)$")


@dataclass(frozen=True)
class ADASArchitecture:
    thought: str
    name: str
    architecture: dict[str, Any]
    parent_generation: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def architecture_sha256(self) -> str:
        encoded = json.dumps(
            self.architecture,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            {
                "architecture": self.architecture,
                "name": self.name,
                "parent_generation": self.parent_generation,
                "thought": self.thought,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["architecture_sha256"] = self.architecture_sha256
        value["sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ADASArchitecture:
        return cls(
            thought=str(value["thought"]),
            name=str(value["name"]),
            architecture=dict(value["architecture"]),
            parent_generation=(
                int(value["parent_generation"])
                if value.get("parent_generation") is not None
                else None
            ),
            metadata=dict(value.get("metadata", {})),
        )


def _node(
    node_id: str,
    output_fields: list[str],
    instruction: str,
    inputs: list[str],
    *,
    role: str = "helpful assistant",
    temperature: float = 0.5,
) -> dict[str, Any]:
    return {
        "id": node_id,
        "role": role,
        "temperature": temperature,
        "output_fields": output_fields,
        "instruction": instruction,
        "inputs": inputs,
    }


def _architecture(nodes: list[dict[str, Any]], output: str) -> dict[str, Any]:
    return {"schema_version": 1, "nodes": nodes, "output": output}


def initial_archive() -> tuple[ADASArchitecture, ...]:
    """Translate the seven official DROP seed architectures into safe DAGs."""

    cot_instruction = "Please think step by step and then solve the task."
    seeds: list[ADASArchitecture] = []
    seeds.append(
        ADASArchitecture(
            thought=("Chain-of-thought exposes intermediate reasoning before a concise answer."),
            name="Chain-of-Thought",
            architecture=_architecture(
                [_node("cot", ["thinking", "answer"], cot_instruction, ["$task"])],
                "$cot.answer",
            ),
        )
    )

    sc_nodes = [
        _node(
            f"cot_{index}",
            ["thinking", "answer"],
            cot_instruction,
            ["$task"],
            temperature=0.8,
        )
        for index in range(1, 6)
    ]
    sc_inputs = ["$task"]
    for index in range(1, 6):
        sc_inputs.extend([f"$cot_{index}.thinking", f"$cot_{index}.answer"])
    sc_nodes.append(
        _node(
            "decision",
            ["thinking", "answer"],
            (
                "Given all solutions above, reason over them carefully and provide "
                "the most reliable final answer."
            ),
            sc_inputs,
            role="final decision agent",
            temperature=0.1,
        )
    )
    seeds.append(
        ADASArchitecture(
            thought="Generate five diverse reasoning paths and ensemble them.",
            name="Self-Consistency with Chain-of-Thought",
            architecture=_architecture(sc_nodes, "$decision.answer"),
        )
    )

    refine_nodes = [_node("draft", ["thinking", "answer"], cot_instruction, ["$task"])]
    previous = "draft"
    for index in range(1, 6):
        critic = f"critic_{index}"
        revision = f"revision_{index}"
        refine_nodes.append(
            _node(
                critic,
                ["feedback", "correct"],
                (
                    "Review the latest answer and identify any error. Put exactly "
                    "'True' or 'False' in correct."
                ),
                ["$task", f"${previous}.thinking", f"${previous}.answer"],
                role="critic agent",
            )
        )
        refine_nodes.append(
            _node(
                revision,
                ["thinking", "answer"],
                ("Use the critique to reconsider the latest attempt and solve the task better."),
                [
                    "$task",
                    f"${previous}.thinking",
                    f"${previous}.answer",
                    f"${critic}.feedback",
                    f"${critic}.correct",
                ],
            )
        )
        previous = revision
    seeds.append(
        ADASArchitecture(
            thought=(
                "Iteratively critique and revise a chain-of-thought answer. The safe "
                "DAG unrolls the official maximum of five revisions."
            ),
            name="Self-Refine (Reflexion)",
            architecture=_architecture(refine_nodes, f"${previous}.answer"),
        )
    )

    debate_nodes: list[dict[str, Any]] = []
    roles = (
        "reading comprehension specialist",
        "logical reasoning strategist",
        "multidisciplinary knowledge integrator",
    )
    for index, role in enumerate(roles, start=1):
        debate_nodes.append(
            _node(
                f"debate_{index}_1",
                ["thinking", "answer"],
                cot_instruction,
                ["$task"],
                role=role,
                temperature=0.8,
            )
        )
    first_round = [
        value
        for index in range(1, 4)
        for value in (f"$debate_{index}_1.thinking", f"$debate_{index}_1.answer")
    ]
    for index, role in enumerate(roles, start=1):
        debate_nodes.append(
            _node(
                f"debate_{index}_2",
                ["thinking", "answer"],
                (
                    "Consider the other proposed solutions as advice, then update "
                    "your reasoning and answer."
                ),
                ["$task", *first_round],
                role=role,
                temperature=0.8,
            )
        )
    second_round = [
        value
        for index in range(1, 4)
        for value in (f"$debate_{index}_2.thinking", f"$debate_{index}_2.answer")
    ]
    debate_nodes.append(
        _node(
            "debate_decision",
            ["thinking", "answer"],
            "Synthesize the debate and provide the best final answer.",
            ["$task", *second_round],
            role="final decision agent",
            temperature=0.1,
        )
    )
    seeds.append(
        ADASArchitecture(
            thought="Use two rounds of three-role debate followed by synthesis.",
            name="LLM Debate",
            architecture=_architecture(debate_nodes, "$debate_decision.answer"),
        )
    )

    seeds.append(
        ADASArchitecture(
            thought="First identify general principles, then solve using them.",
            name="Step-back Abstraction",
            architecture=_architecture(
                [
                    _node(
                        "principles",
                        ["thinking", "principle"],
                        (
                            "Identify and explain the principles or concepts needed "
                            "to solve this task."
                        ),
                        ["$task"],
                        role="principle agent",
                    ),
                    _node(
                        "stepback_answer",
                        ["thinking", "answer"],
                        "Use the task and identified principles to solve step by step.",
                        ["$task", "$principles.thinking", "$principles.principle"],
                    ),
                ],
                "$stepback_answer.answer",
            ),
        )
    )

    qd_nodes: list[dict[str, Any]] = [
        _node("diverse_1", ["thinking", "answer"], cot_instruction, ["$task"])
    ]
    diverse_inputs = ["$task", "$diverse_1.thinking", "$diverse_1.answer"]
    for index in range(2, 5):
        qd_nodes.append(
            _node(
                f"diverse_{index}",
                ["thinking", "answer"],
                "Try another substantially different way to solve the task.",
                list(diverse_inputs),
            )
        )
        diverse_inputs.extend([f"$diverse_{index}.thinking", f"$diverse_{index}.answer"])
    qd_nodes.append(
        _node(
            "qd_decision",
            ["thinking", "answer"],
            "Compare all diverse solutions and return the best final answer.",
            diverse_inputs,
            role="final decision agent",
            temperature=0.1,
        )
    )
    seeds.append(
        ADASArchitecture(
            thought="Explore several diverse solution paths before selection.",
            name="Quality-Diversity",
            architecture=_architecture(qd_nodes, "$qd_decision.answer"),
        )
    )

    seeds.append(
        ADASArchitecture(
            thought="Route the task to a dynamically described expert role.",
            name="Dynamic Assignment of Roles",
            architecture=_architecture(
                [
                    _node(
                        "router",
                        ["choice"],
                        (
                            "Choose the most useful expert role for this task and "
                            "describe that role concisely."
                        ),
                        ["$task"],
                        role="routing agent",
                    ),
                    _node(
                        "dynamic_expert",
                        ["thinking", "answer"],
                        (
                            "Adopt the expert role selected above, think step by step, "
                            "and answer the task."
                        ),
                        ["$task", "$router.choice"],
                        role="dynamically assigned expert",
                    ),
                ],
                "$dynamic_expert.answer",
            ),
        )
    )
    for seed in seeds:
        validate_architecture(seed)
    return tuple(seeds)


def validate_architecture(candidate: ADASArchitecture) -> None:
    if (
        not candidate.thought.strip()
        or not candidate.name.strip()
        or len(candidate.thought) > 8000
        or len(candidate.name) > 200
    ):
        raise ADASValidationError("thought and name must be non-empty")
    graph = candidate.architecture
    if set(graph) != {"schema_version", "nodes", "output"}:
        raise ADASValidationError("architecture has unsupported fields")
    if graph.get("schema_version") != 1:
        raise ADASValidationError("architecture.schema_version must equal 1")
    nodes = graph["nodes"]
    if not isinstance(nodes, list) or not 1 <= len(nodes) <= 12:
        raise ADASValidationError("architecture must contain between 1 and 12 nodes")
    fields_by_node: dict[str, set[str]] = {}
    for node in nodes:
        expected = {
            "id",
            "role",
            "temperature",
            "output_fields",
            "instruction",
            "inputs",
        }
        if not isinstance(node, dict) or set(node) != expected:
            raise ADASValidationError(f"agent node must contain exactly {sorted(expected)}")
        node_id = node["id"]
        if (
            not isinstance(node_id, str)
            or not _IDENTIFIER.fullmatch(node_id)
            or node_id in fields_by_node
        ):
            raise ADASValidationError(f"invalid or duplicate node id: {node_id}")
        if not isinstance(node["role"], str) or not node["role"].strip() or len(node["role"]) > 300:
            raise ADASValidationError(f"node {node_id} has an invalid role")
        if (
            not isinstance(node["instruction"], str)
            or not node["instruction"].strip()
            or len(node["instruction"]) > 4000
        ):
            raise ADASValidationError(f"node {node_id} has an invalid instruction")
        temperature = node["temperature"]
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not 0 <= float(temperature) <= 2
        ):
            raise ADASValidationError(f"node {node_id} has an invalid temperature")
        output_fields = node["output_fields"]
        if (
            not isinstance(output_fields, list)
            or not 1 <= len(output_fields) <= 5
            or len(set(output_fields)) != len(output_fields)
            or any(
                not isinstance(field, str) or not _IDENTIFIER.fullmatch(field)
                for field in output_fields
            )
        ):
            raise ADASValidationError(f"node {node_id} has invalid output_fields")
        inputs = node["inputs"]
        if not isinstance(inputs, list) or not 1 <= len(inputs) <= 30:
            raise ADASValidationError(f"node {node_id} has invalid inputs")
        for reference in inputs:
            if not isinstance(reference, str) or not _REFERENCE.fullmatch(reference):
                raise ADASValidationError(f"node {node_id} has invalid input: {reference}")
            if reference == "$task":
                continue
            source, field_name = reference[1:].split(".", maxsplit=1)
            if source not in fields_by_node or field_name not in fields_by_node[source]:
                raise ADASValidationError(
                    f"node {node_id} has forward or unknown input: {reference}"
                )
        fields_by_node[node_id] = set(output_fields)
    output = graph["output"]
    if not isinstance(output, str) or not _REFERENCE.fullmatch(output) or output == "$task":
        raise ADASValidationError("architecture.output must be a node field reference")
    source, field_name = output[1:].split(".", maxsplit=1)
    if source not in fields_by_node or field_name not in fields_by_node[source]:
        raise ADASValidationError("architecture.output is unknown")


def _parse_fields(content: str, output_fields: list[str]) -> dict[str, str]:
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    value = json.loads(stripped)
    if (
        not isinstance(value, dict)
        or set(value) != set(output_fields)
        or any(not isinstance(value[field], str) for field in output_fields)
    ):
        raise ADASValidationError(
            f"agent response must contain exactly string fields {output_fields}"
        )
    return {field: value[field] for field in output_fields}


async def execute_architecture(
    candidate: ADASArchitecture,
    client: OpenRouterClient,
    *,
    task: str,
    dataset: str,
    sample_id: str,
) -> tuple[str, list[ChatResult], list[dict[str, Any]]]:
    """Execute a candidate as data; optimizer-produced Python is never evaluated."""

    validate_architecture(candidate)
    results: dict[str, dict[str, str]] = {}
    responses: list[ChatResult] = []
    trace: list[dict[str, Any]] = []
    for node in candidate.architecture["nodes"]:
        output_fields = node["output_fields"]
        descriptions = {
            field_name: (
                (
                    f"Your {field_name}. Return a complete Python program as plain "
                    "text with the requested entry point and no markdown."
                )
                if dataset in {"humaneval", "mbpp"} and "answer" in field_name
                else f"Your {field_name}. Directly answer the task and keep it concise."
                if "answer" in field_name
                else f"Your {field_name}."
            )
            for field_name in output_fields
        }
        system = (
            f"You are a {node['role']}.\n\n"
            "Reply with exactly one well-formed JSON object using these string fields:\n"
            f"{json.dumps(descriptions, ensure_ascii=False, sort_keys=True)}"
        )
        blocks: list[str] = []
        for reference in node["inputs"]:
            if reference == "$task":
                blocks.append(f"# Your Task:\n{task}")
                continue
            source, field_name = reference[1:].split(".", maxsplit=1)
            blocks.append(f"### {field_name} by {source}:\n{results[source][field_name]}")
        user = "\n\n".join([*blocks, f"# Instruction:\n{node['instruction']}"])
        response = await asyncio.to_thread(
            client.chat,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=float(node["temperature"]),
            metadata={
                "method": "adas",
                "role": "executor",
                "dataset": dataset,
                "sample_id": sample_id,
                "architecture": candidate.name,
                "node": node["id"],
            },
        )
        parsed = _parse_fields(response.content, output_fields)
        results[node["id"]] = parsed
        responses.append(response)
        trace.append(
            {
                "id": node["id"],
                "role": node["role"],
                "temperature": float(node["temperature"]),
                "output_fields": list(output_fields),
            }
        )
    source, field_name = candidate.architecture["output"][1:].split(".", maxsplit=1)
    return results[source][field_name], responses, trace
