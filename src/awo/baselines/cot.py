"""One-request chain-of-thought baseline for all six Table 1 datasets."""

from __future__ import annotations

from awo.baselines.io import IOPrompt, _hotpot_context, parse_io_response
from awo.baselines.models import BaselineResult
from awo.benchmarks.data import BenchmarkExample
from awo.llm import OpenRouterClient

COT_TEMPLATES = {
    "hotpotqa": (
        "Think step by step and solve the problem.\n"
        '1. In the "thought" field, explain your thinking process in detail.\n'
        '2. In the "answer" field, provide the final answer concisely and clearly. '
        "The answer should be a direct response to the question, without including "
        "explanations or reasoning.\n"
        "Question: {question}\n"
        "The revelant context: {context}\n"
    ),
    "drop": (
        "{question}\n"
        "Please reason step by step. At the end, provide only the final answer after "
        '"Answer is:" without any additional information after it.\n'
    ),
    "gsm8k": (
        "{question}\n"
        "Please reason step by step. At the end, provide the final answer in the format "
        '"Answer is <number>", where <number> is a single number, without any additional '
        "information or explanation.\n"
    ),
    "math": (
        "{question}\n"
        "Please reason step by step. At the end, provide the final answer in the format "
        '"\\boxed{{<number>}}", where <number> is a math answer'
        "(an expression or number), without any additional information or explanation.\n"
    ),
    "humaneval": (
        "{question}\n"
        "Please provide a step-by-step explanation in text, followed by your Python "
        "function without any additional text or test cases.\n"
    ),
    "mbpp": (
        "{question}\n"
        "Please provide a step-by-step explanation in text, followed by your Python "
        "function, ensure the output code is self-contained, meaning it should have the "
        "correct function name and return statement, without any additional text."
        "\n"
    ),
}

COT_PROVENANCE = {
    "hotpotqa": "upstream-user/adapted-schema",
    "drop": "paper-faithful/inferred",
    "gsm8k": "upstream-user/adapted-schema",
    "math": "upstream-user/adapted-schema",
    "humaneval": "upstream-user/adapted-schema",
    "mbpp": "upstream-user/adapted-schema",
}


def build_cot_prompt(example: BenchmarkExample) -> IOPrompt:
    dataset = example.dataset.lower()
    if dataset not in COT_TEMPLATES:
        raise ValueError(f"unsupported CoT dataset: {example.dataset}")
    if dataset == "hotpotqa":
        user = COT_TEMPLATES[dataset].format(
            question=example.metadata["question"], context=_hotpot_context(example)
        )
    else:
        user = COT_TEMPLATES[dataset].format(question=example.prompt)
    output_key = "answer" if dataset in {"hotpotqa", "drop"} else "solution"
    system = (
        "Return one valid JSON object and no surrounding markdown. "
        f"The required string field is {output_key!r}. Preserve the requested reasoning "
        "inside the field. For code tasks, end that field with the complete Python program."
    )
    return IOPrompt(dataset, system, user, COT_PROVENANCE[dataset])


class CoTBaseline:
    method = "cot"
    expected_calls = 1

    def __init__(self, client: OpenRouterClient) -> None:
        self.client = client

    def run(self, example: BenchmarkExample) -> BaselineResult:
        prompt = build_cot_prompt(example)
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
