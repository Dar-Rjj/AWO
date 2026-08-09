"""One-request Input-Output baseline for all six Table 1 datasets."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from awo.baselines.models import BaselineResult
from awo.benchmarks.code import extract_python_code
from awo.benchmarks.data import BenchmarkExample
from awo.llm import OpenRouterClient

IO_TEMPLATES = {
    "hotpotqa": (
        "Given a question and a context, please answer the question.\n"
        '1. In the "thought" field, explain your thinking process.\n'
        '2. In the "answer" field, provide the final answer concisely and clearly. '
        "The answer should be a direct response to the question, without including "
        "explanations or reasoning.\n"
        "Question: {question}\n"
        "The revelant context: {context}\n"
    ),
    "drop": (
        "Given the passage and question below, provide the final answer concisely. "
        "The answer must be a direct response without explanation.\n"
        "{question}\n"
    ),
    "gsm8k": (
        "{question}\n"
        "Generate an answer to this question. At the end, provide the final answer in "
        'the format "Answer is <number>", where <number> is a single number.\n'
    ),
    "math": (
        "{question}\n"
        "Please generate a solution for the problem. At the end, provide the final answer "
        'in the format "\\boxed{{<number>}}", where <number> is a math answer'
        "(an expression or number), without any additional information or explanation.\n"
    ),
    "humaneval": (
        "{question}\n"
        "Generate an answer to this question, without any additional test cases.\n"
    ),
    "mbpp": (
        "{question}\n"
        "Generate an answer to this question, ensure the output code is self-contained, "
        "meaning it should have the correct function name and return statement, but "
        "without any additional test cases.\n"
    ),
}

IO_PROVENANCE = {
    "hotpotqa": "upstream-user/adapted-schema",
    "drop": "paper-faithful/inferred",
    "gsm8k": "upstream-user/adapted-schema",
    "math": "upstream-user/adapted-schema",
    "humaneval": "upstream-user/adapted-schema",
    "mbpp": "upstream-user/adapted-schema",
}


@dataclass(frozen=True)
class IOPrompt:
    dataset: str
    system: str
    user: str
    provenance: str

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            {"system": self.system, "user": self.user},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _hotpot_context(example: BenchmarkExample) -> str:
    raw_context = example.metadata.get("context", [])
    paragraphs = [item[1] for item in raw_context if isinstance(item, list) and len(item) > 1]
    return "\n".join(
        " ".join(str(sentence) for sentence in paragraph)
        for paragraph in paragraphs
        if isinstance(paragraph, list)
    )


def build_io_prompt(example: BenchmarkExample) -> IOPrompt:
    dataset = example.dataset.lower()
    if dataset not in IO_TEMPLATES:
        raise ValueError(f"unsupported IO dataset: {example.dataset}")
    if dataset == "hotpotqa":
        user = IO_TEMPLATES[dataset].format(
            question=example.metadata["question"], context=_hotpot_context(example)
        )
    else:
        user = IO_TEMPLATES[dataset].format(question=example.prompt)
    output_key = "answer" if dataset in {"hotpotqa", "drop"} else "solution"
    system = (
        "Return one valid JSON object and no surrounding markdown. "
        f"The required string field is {output_key!r}. "
        "For code tasks, put the complete Python program in that field."
    )
    return IOPrompt(dataset, system, user, IO_PROVENANCE[dataset])


def _json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def parse_io_response(dataset: str, content: str) -> str:
    normalized = dataset.lower()
    parsed = _json_object(content)
    keys = ("answer", "solution") if normalized in {"hotpotqa", "drop"} else (
        "solution",
        "answer",
    )
    if parsed is not None:
        for key in keys:
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return extract_python_code(value) if normalized in {"humaneval", "mbpp"} else value

    if normalized in {"humaneval", "mbpp"}:
        return extract_python_code(content)
    if normalized in {"hotpotqa", "drop"}:
        matches = re.findall(
            r"(?im)^\s*(?:final\s+)?answer\s*(?:is|:)\s*(.+?)\s*$", content
        )
        if matches:
            return matches[-1].strip().strip('"')
    return content.strip()


class IOBaseline:
    method = "io"
    expected_calls = 1

    def __init__(self, client: OpenRouterClient) -> None:
        self.client = client

    def run(self, example: BenchmarkExample) -> BaselineResult:
        prompt = build_io_prompt(example)
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
        )
