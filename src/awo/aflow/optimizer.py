"""OpenRouter-backed optimizer constrained to the safe workflow DSL."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from awo.aflow.dsl import WorkflowCandidate, validate_candidate
from awo.aflow.search import SearchContext
from awo.llm import OpenRouterClient

OPTIMIZE_PROMPT = """You optimize an agent workflow for {dataset}.

Return exactly one JSON object with keys:
- "modification": one concise, specific change from the parent;
- "graph": a schema_version=1 declarative graph;
- "prompts": an object mapping UPPERCASE prompt names to literal instructions.

The graph has 1-10 nodes. Each node has exactly "id", "operator", and "inputs".
Allowed operators: {operators}.
References are "$problem", "$entry_point", "$public_tests", "$prompt.NAME", or
"$prior_node.field". Inputs may also use {{"concat": [...]}} or {{"list": [...]}}.
Nodes may only reference earlier nodes. Do not emit Python, imports, code fences,
runtime prompt placeholders such as {{problem}}, or more than one modification.

Operator schemas (use only operators listed as allowed above):
- Custom inputs: input, instruction; output: response.
- AnswerGenerate inputs: input; outputs: thought, answer.
- CustomCodeGenerate inputs: problem, entry_point, instruction; output: response.
- ScEnsemble inputs: solutions (a list), problem; output: response.
- Programmer inputs: problem, optional analysis; outputs: code, output.
- Test inputs: problem, solution, entry_point, optional public_tests/test_loop;
  outputs: result, solution.

Parent candidate:
{parent}

Parent validation mean: {parent_score:.6f}
Prior experience for this parent:
{experience}
"""


def _json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    value = json.loads(stripped)
    if not isinstance(value, dict):
        raise ValueError("optimizer response must be one JSON object")
    return value


class OpenRouterCandidateGenerator:
    """Generate one candidate per call and reject everything outside the DSL."""

    def __init__(self, client: OpenRouterClient) -> None:
        self.client = client

    async def generate(self, context: SearchContext) -> WorkflowCandidate:
        prompt = OPTIMIZE_PROMPT.format(
            dataset=context.dataset,
            operators=", ".join(context.allowed_operators),
            parent=json.dumps(context.parent.to_dict(), ensure_ascii=False, sort_keys=True),
            parent_score=context.parent_record.mean_score,
            experience=context.experience,
        )
        response = await asyncio.to_thread(
            self.client.chat,
            [
                {
                    "role": "system",
                    "content": ("You are the AFlow workflow optimizer. Output strict JSON only."),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            metadata={
                "role": "aflow_optimizer",
                "dataset": context.dataset,
                "round": context.round,
                "parent_round": context.parent_record.round,
            },
        )
        payload = _json_object(response.content)
        if set(payload) != {"modification", "graph", "prompts"}:
            raise ValueError("optimizer JSON must contain exactly modification, graph and prompts")
        candidate = WorkflowCandidate(
            modification=str(payload["modification"]),
            graph=dict(payload["graph"]),
            prompts={str(key): str(value) for key, value in dict(payload["prompts"]).items()},
            parent_round=context.parent_record.round,
            metadata={
                "request_sha256": response.request_sha256,
                "provider": response.provider,
                "cost": response.usage.cost,
                "tokens": response.usage.total_tokens,
            },
        )
        validate_candidate(candidate, context.dataset)
        return candidate
