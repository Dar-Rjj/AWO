"""Five-candidate universal self-consistency with one LLM selector."""

from __future__ import annotations

import hashlib
import json
import re

from awo.baselines.cot import build_cot_prompt
from awo.baselines.io import _hotpot_context, _json_object, parse_io_response
from awo.baselines.models import BaselineResult
from awo.benchmarks.data import BenchmarkExample
from awo.llm import OpenRouterClient

SELECTOR_TEMPLATE = (
    "Given the question described as follows: {question}\n"
    "Several solutions have been generated to address the given question. They are as follows:\n"
    "{solutions}\n\n"
    "Carefully evaluate these solutions and identify the answer that appears most frequently "
    "across them. This consistency in answers is crucial for determining the most reliable "
    "solution.\n\n"
    'In the "thought" field, provide a detailed explanation of your thought process. In the '
    '"solution_letter" field, output only the single letter ID (A, B, C, etc.) corresponding '
    "to the most consistent solution. Do not include any additional text or explanation in "
    'the "solution_letter" field.\n'
)

HOTPOT_SELECTOR_TEMPLATE = (
    "Given the question descripted as follows: {question}\n"
    "And the relevant context is provided as follows: {context}\n"
    "some solutions to the question are generated as follows:\n"
    "{solutions}\n\n"
    "Evaluate these solutions and select the most consistent solution based on majority "
    "consensus.\n"
    "Give your answer with a single id of solution (without anything else).\n"
)


class CoTSCProtocolError(ValueError):
    """Raised when the selector does not return one of the five candidate IDs."""


def _solution_listing(candidates: list[str]) -> str:
    return "".join(
        f"{chr(65 + index)}: \n{candidate}\n\n\n" for index, candidate in enumerate(candidates)
    )


def build_selector_prompt(example: BenchmarkExample, candidates: list[str]) -> str:
    if len(candidates) != 5:
        raise ValueError("CoT-SC requires exactly five candidates")
    values = {"question": example.prompt, "solutions": _solution_listing(candidates)}
    if example.dataset == "hotpotqa":
        values["question"] = str(example.metadata["question"])
        values["context"] = _hotpot_context(example)
        return HOTPOT_SELECTOR_TEMPLATE.format(**values)
    return SELECTOR_TEMPLATE.format(**values)


def parse_selector_letter(content: str, candidate_count: int = 5) -> str:
    allowed = {chr(65 + index) for index in range(candidate_count)}
    parsed = _json_object(content)
    if parsed is not None:
        value = parsed.get("solution_letter")
        if isinstance(value, str):
            match = re.search(r"[A-Z]", value.upper())
            if match and match.group() in allowed:
                return match.group()
    stripped = content.strip().upper().strip("\"'` .")
    if stripped in allowed:
        return stripped
    match = re.search(
        r"(?i)[\"']?solution_letter[\"']?\s*(?:is|:|=)\s*[\"']?([A-Z])",
        content,
    )
    if match and match.group(1).upper() in allowed:
        return match.group(1).upper()
    raise CoTSCProtocolError("selector did not return one candidate letter A-E")


class CoTSCBaseline:
    method = "cot_sc"
    candidate_count = 5
    expected_calls = 6

    def __init__(self, client: OpenRouterClient) -> None:
        self.client = client

    def run(self, example: BenchmarkExample) -> BaselineResult:
        candidate_prompt = build_cot_prompt(example)
        candidates = []
        responses = []
        for index in range(self.candidate_count):
            response = self.client.chat(
                [
                    {"role": "system", "content": candidate_prompt.system},
                    {"role": "user", "content": candidate_prompt.user},
                ],
                metadata={
                    "method": self.method,
                    "dataset": example.dataset,
                    "sample_id": example.sample_id,
                    "role": "candidate_generator",
                    "candidate_index": index,
                    "prompt_provenance": candidate_prompt.provenance,
                },
            )
            responses.append(response)
            candidates.append(parse_io_response(example.dataset, response.content))

        selector_prompt = build_selector_prompt(example, candidates)
        selector_system = (
            "Return one valid JSON object with a single string field 'solution_letter'. "
            "Its value must be exactly one of A, B, C, D, or E."
        )
        selector_response = self.client.chat(
            [
                {"role": "system", "content": selector_system},
                {"role": "user", "content": selector_prompt},
            ],
            metadata={
                "method": self.method,
                "dataset": example.dataset,
                "sample_id": example.sample_id,
                "role": "selector",
                "candidate_count": self.candidate_count,
                "prompt_provenance": (
                    "upstream-user/adapted-schema"
                    if example.dataset != "drop"
                    else "paper-faithful/inferred"
                ),
            },
        )
        responses.append(selector_response)
        letter = parse_selector_letter(selector_response.content, self.candidate_count)
        selected_index = ord(letter) - 65
        prompt_hash = hashlib.sha256(
            json.dumps(
                {
                    "candidate_prompt_sha256": candidate_prompt.sha256,
                    "selector_system": selector_system,
                    "selector_user": selector_prompt,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return BaselineResult(
            method=self.method,
            dataset=example.dataset,
            sample_id=example.sample_id,
            prediction=candidates[selected_index],
            prompt_sha256=prompt_hash,
            responses=tuple(responses),
            artifacts={
                "candidates": candidates,
                "selected_index": selected_index,
                "selected_letter": letter,
                "candidate_prompt_sha256": candidate_prompt.sha256,
            },
        )
