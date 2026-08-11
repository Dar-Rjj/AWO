"""A safe, declarative workflow language for controlled AFlow reproduction."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from awo.aflow.operators import (
    AnswerGenerate,
    Custom,
    CustomCodeGenerate,
    Programmer,
    ScEnsemble,
    Test,
)
from awo.aflow.runtime import AFlowRuntime


class WorkflowDSLValidationError(ValueError):
    """Raised when a generated workflow is outside the controlled DSL."""


OPERATOR_SETS: dict[str, frozenset[str]] = {
    "drop": frozenset({"Custom", "AnswerGenerate", "ScEnsemble"}),
    "gsm8k": frozenset({"Custom", "ScEnsemble", "Programmer"}),
    "hotpotqa": frozenset({"Custom", "AnswerGenerate", "ScEnsemble"}),
    "humaneval": frozenset({"Custom", "CustomCodeGenerate", "ScEnsemble", "Test"}),
    "math": frozenset({"Custom", "ScEnsemble", "Programmer"}),
    "mbpp": frozenset({"Custom", "CustomCodeGenerate", "ScEnsemble", "Test"}),
}

_CODE_DATASETS = frozenset({"humaneval", "mbpp"})
_PROMPT_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
_REFERENCE = re.compile(
    r"^\$(problem|entry_point|public_tests|prompt\.[A-Z][A-Z0-9_]*|"
    r"[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*)$"
)
_INPUTS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "Custom": (frozenset({"input", "instruction"}), frozenset({"input", "instruction"})),
    "AnswerGenerate": (frozenset({"input"}), frozenset({"input"})),
    "CustomCodeGenerate": (
        frozenset({"problem", "entry_point", "instruction"}),
        frozenset({"problem", "entry_point", "instruction"}),
    ),
    "ScEnsemble": (
        frozenset({"solutions", "problem"}),
        frozenset({"solutions", "problem"}),
    ),
    "Programmer": (
        frozenset({"problem"}),
        frozenset({"problem", "analysis"}),
    ),
    "Test": (
        frozenset({"problem", "solution", "entry_point"}),
        frozenset({"problem", "solution", "entry_point", "public_tests", "test_loop"}),
    ),
}


def normalize_dataset(dataset: str) -> str:
    normalized = dataset.strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "drop": "drop",
        "gsm8k": "gsm8k",
        "hotpotqa": "hotpotqa",
        "humaneval": "humaneval",
        "math": "math",
        "mbpp": "mbpp",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise WorkflowDSLValidationError(f"unsupported dataset: {dataset}") from exc


@dataclass(frozen=True)
class WorkflowCandidate:
    """One immutable workflow proposed by the optimizer."""

    modification: str
    graph: dict[str, Any]
    prompts: dict[str, str] = field(default_factory=dict)
    parent_round: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def sha256(self) -> str:
        payload = {
            "graph": self.graph,
            "modification": self.modification,
            "parent_round": self.parent_round,
            "prompts": self.prompts,
        }
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    @property
    def workflow_sha256(self) -> str:
        payload = {"graph": self.graph, "prompts": self.prompts}
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["sha256"] = self.sha256
        result["workflow_sha256"] = self.workflow_sha256
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> WorkflowCandidate:
        return cls(
            modification=str(value["modification"]),
            graph=dict(value["graph"]),
            prompts={str(k): str(v) for k, v in dict(value.get("prompts", {})).items()},
            parent_round=(
                int(value["parent_round"]) if value.get("parent_round") is not None else None
            ),
            metadata=dict(value.get("metadata", {})),
        )


def initial_candidate(dataset: str) -> WorkflowCandidate:
    """Return the separately evaluated blank/initial workflow."""

    normalized = normalize_dataset(dataset)
    if normalized in _CODE_DATASETS:
        operator = "CustomCodeGenerate"
        inputs: dict[str, Any] = {
            "problem": "$problem",
            "entry_point": "$entry_point",
            "instruction": "",
        }
    else:
        operator = "Custom"
        inputs = {"input": "$problem", "instruction": ""}
    return WorkflowCandidate(
        modification="Initial single-agent workflow.",
        graph={
            "schema_version": 1,
            "nodes": [{"id": "answer", "operator": operator, "inputs": inputs}],
            "output": "$answer.response",
        },
    )


def _walk_references(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.startswith("$") else []
    if isinstance(value, list):
        references: list[str] = []
        for item in value:
            references.extend(_walk_references(item))
        return references
    if isinstance(value, dict):
        references = []
        for item in value.values():
            references.extend(_walk_references(item))
        return references
    return []


def _validate_expression(value: Any) -> None:
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str) and value.startswith("$") and not _REFERENCE.fullmatch(value):
            raise WorkflowDSLValidationError(f"invalid reference: {value}")
        return
    if isinstance(value, list):
        for item in value:
            _validate_expression(item)
        return
    if isinstance(value, dict):
        if set(value) not in ({"concat"}, {"list"}):
            raise WorkflowDSLValidationError(
                "expression objects must contain exactly 'concat' or 'list'"
            )
        sequence = next(iter(value.values()))
        if not isinstance(sequence, list):
            raise WorkflowDSLValidationError("concat/list expression must be a list")
        for item in sequence:
            _validate_expression(item)
        return
    raise WorkflowDSLValidationError(f"unsupported expression value: {type(value).__name__}")


def validate_candidate(candidate: WorkflowCandidate, dataset: str) -> None:
    """Validate a candidate without importing or executing generated Python."""

    normalized = normalize_dataset(dataset)
    if not candidate.modification.strip():
        raise WorkflowDSLValidationError("modification must be non-empty")
    graph = candidate.graph
    if graph.get("schema_version") != 1:
        raise WorkflowDSLValidationError("graph.schema_version must equal 1")
    if set(graph) != {"schema_version", "nodes", "output"}:
        raise WorkflowDSLValidationError("graph has unsupported fields")
    nodes = graph["nodes"]
    if not isinstance(nodes, list) or not 1 <= len(nodes) <= 10:
        raise WorkflowDSLValidationError("graph must contain between 1 and 10 nodes")
    for name, prompt in candidate.prompts.items():
        if not _PROMPT_NAME.fullmatch(name):
            raise WorkflowDSLValidationError(f"invalid prompt name: {name}")
        if not isinstance(prompt, str) or not prompt.strip():
            raise WorkflowDSLValidationError(f"prompt {name} must be non-empty")
        if any(marker in prompt for marker in ("{problem}", "{input}", "{entry_point}")):
            raise WorkflowDSLValidationError(
                f"prompt {name} contains a prohibited runtime placeholder"
            )

    seen: set[str] = set()
    for node in nodes:
        if not isinstance(node, dict) or set(node) != {"id", "operator", "inputs"}:
            raise WorkflowDSLValidationError(
                "each node must contain exactly id, operator and inputs"
            )
        node_id = node["id"]
        if (
            not isinstance(node_id, str)
            or not re.fullmatch(r"[a-z][a-z0-9_]*", node_id)
            or node_id in seen
        ):
            raise WorkflowDSLValidationError(f"invalid or duplicate node id: {node_id}")
        operator = node["operator"]
        if operator not in OPERATOR_SETS[normalized]:
            raise WorkflowDSLValidationError(f"operator {operator} is not allowed for {normalized}")
        inputs = node["inputs"]
        if not isinstance(inputs, dict):
            raise WorkflowDSLValidationError(f"node {node_id} inputs must be an object")
        required, allowed = _INPUTS[operator]
        input_keys = frozenset(inputs)
        if not required <= input_keys or not input_keys <= allowed:
            raise WorkflowDSLValidationError(f"invalid inputs for {operator}: {sorted(input_keys)}")
        for value in inputs.values():
            _validate_expression(value)
            for reference in _walk_references(value):
                target = reference[1:].split(".", maxsplit=1)[0]
                if target in {"problem", "entry_point", "public_tests"}:
                    continue
                if target == "prompt":
                    name = reference.removeprefix("$prompt.")
                    if name not in candidate.prompts:
                        raise WorkflowDSLValidationError(f"undefined prompt reference: {reference}")
                    continue
                if target not in seen:
                    raise WorkflowDSLValidationError(
                        f"node {node_id} has a forward or unknown reference: {reference}"
                    )
        seen.add(node_id)

    _validate_expression(graph["output"])
    output_references = _walk_references(graph["output"])
    if len(output_references) != 1:
        raise WorkflowDSLValidationError("graph.output must contain one node reference")
    output_target = output_references[0][1:].split(".", maxsplit=1)[0]
    if output_target not in seen:
        raise WorkflowDSLValidationError("graph.output must reference an existing node")


def _resolve(
    value: Any,
    *,
    context: Mapping[str, Any],
    prompts: Mapping[str, str],
    results: Mapping[str, Mapping[str, Any]],
) -> Any:
    if isinstance(value, str):
        if not value.startswith("$"):
            return value
        path = value[1:].split(".")
        if path[0] == "prompt":
            return prompts[path[1]]
        if path[0] in context:
            resolved: Any = context[path[0]]
        else:
            resolved = results[path[0]]
        for part in path[1:]:
            if not isinstance(resolved, Mapping) or part not in resolved:
                raise WorkflowDSLValidationError(f"unresolved reference: {value}")
            resolved = resolved[part]
        return resolved
    if isinstance(value, list):
        return [_resolve(item, context=context, prompts=prompts, results=results) for item in value]
    if isinstance(value, dict):
        if "list" in value:
            return [
                _resolve(item, context=context, prompts=prompts, results=results)
                for item in value["list"]
            ]
        parts = [
            _resolve(item, context=context, prompts=prompts, results=results)
            for item in value["concat"]
        ]
        return "".join(str(part) for part in parts)
    return value


async def execute_candidate(
    candidate: WorkflowCandidate,
    runtime: AFlowRuntime,
    *,
    dataset: str,
    example: Mapping[str, Any],
    public_tests: str = "",
) -> tuple[str, list[dict[str, Any]]]:
    """Interpret a validated workflow using only registered safe operators."""

    normalized = normalize_dataset(dataset)
    validate_candidate(candidate, normalized)
    context = {
        "problem": str(example.get("problem", example.get("question", ""))),
        "entry_point": str(example.get("entry_point", "")),
        "public_tests": public_tests,
    }
    results: dict[str, dict[str, Any]] = {}
    trace: list[dict[str, Any]] = []
    for node in candidate.graph["nodes"]:
        node_id = node["id"]
        operator_name = node["operator"]
        inputs = {
            key: _resolve(
                value,
                context=context,
                prompts=candidate.prompts,
                results=results,
            )
            for key, value in node["inputs"].items()
        }
        if operator_name == "Custom":
            result = await Custom(runtime)(**inputs)
        elif operator_name == "AnswerGenerate":
            result = await AnswerGenerate(runtime)(**inputs)
        elif operator_name == "CustomCodeGenerate":
            result = await CustomCodeGenerate(runtime)(**inputs)
        elif operator_name == "ScEnsemble":
            result = await ScEnsemble(runtime, qa_mode=normalized in {"drop", "hotpotqa"})(**inputs)
        elif operator_name == "Programmer":
            result = await Programmer(runtime)(**inputs)
        elif operator_name == "Test":
            inputs.setdefault("public_tests", public_tests)
            result = await Test(runtime)(**inputs)
        else:  # pragma: no cover
            raise WorkflowDSLValidationError(f"unknown operator: {operator_name}")
        result_dict = dict(result)
        results[node_id] = result_dict
        trace.append({"id": node_id, "operator": operator_name, "result": result_dict})
    prediction = _resolve(
        candidate.graph["output"],
        context=context,
        prompts=candidate.prompts,
        results=results,
    )
    return str(prediction), trace
