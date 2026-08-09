"""Initial solution followed by at most three review/revise rounds."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from awo.baselines.cot import build_cot_prompt
from awo.baselines.io import _json_object, parse_io_response
from awo.baselines.models import BaselineResult
from awo.benchmarks.data import BenchmarkExample
from awo.llm import OpenRouterClient

REVIEW_TEMPLATE = (
    "\nGiven a problem and a thoughtful solution, your task is to using critical thinking "
    "(questioning) to review the solution's correctness and provide a review result in boolean "
    "format.\n\nproblem: {problem}\nsolution: {solution}\n\nIf you are more than 95 percent "
    "confident that the final answer is incorrect, please return False and give a feedback for "
    "the error. Otherwise, please return True and give a explanation for the correctness.\n"
)

GSM8K_REVISE_TEMPLATE = (
    "\nGiven a problem and a thoughtful solution which is just reviewed as incorrect, your task "
    "is to revise the solution to solve the question and ensure the final answer in the format "
    '"Answer is <number>", where <number> is a single number.\n\nproblem: {problem}\nsolution: '
    "{solution}\nfeedback: {feedback}\n"
)

MATH_REVISE_TEMPLATE = (
    "\nGiven a problem and a thoughtful solution which is just reviewed as incorrect, your task "
    "is to revise the solution to solve the question and ensure the final answer in the format "
    '"\\boxed{{<number>}}", where <number> is a math answer(an expression or number), without '
    "any additional information or explanation.\n\nproblem: {problem}\nsolution: {solution}\n"
    "feedback: {feedback}\n"
)

HUMANEVAL_REVISE_TEMPLATE = (
    "\nGiven a problem and a thoughtful solution which is just reviewed as incorrect, your task "
    "is to revise the solution to solve the question and ensure the final code solution is "
    "wrapped with ```python```.\n\nproblem: {problem}\nsolution: {solution}\nfeedback: "
    "{feedback}\n\nEnsure the output code is self-contained, and without any additional text "
    "or test cases.\n"
)

MBPP_REVISE_TEMPLATE = (
    "\nGiven a problem and a thoughtful solution which is just reviewed as incorrect, your task "
    "is to revise the solution to solve the question and ensure the final code solution is "
    "wrapped with ```python```.\n\nproblem: {problem}\nsolution: {solution}\nfeedback: "
    "{feedback}\n\nEnsure the output code is self-contained, meaning it should have the correct "
    "function name and return statement, without any additional text.\n"
)

QA_REVISE_TEMPLATE = (
    "\nGiven a problem and a thoughtful solution which is just reviewed as incorrect, revise "
    "the solution using the feedback. Return a concise final answer without explanation.\n\n"
    "problem: {problem}\nsolution: {solution}\nfeedback: {feedback}\n"
)


class SelfRefineProtocolError(ValueError):
    """Raised when the reviewer does not provide an explicit boolean decision."""


@dataclass(frozen=True)
class ReviewDecision:
    accepted: bool
    feedback: str


def build_review_prompt(example: BenchmarkExample, solution: str) -> str:
    return REVIEW_TEMPLATE.format(problem=example.prompt, solution=solution)


def build_revise_prompt(example: BenchmarkExample, solution: str, feedback: str) -> str:
    if example.dataset == "gsm8k":
        template = GSM8K_REVISE_TEMPLATE
    elif example.dataset == "math":
        template = MATH_REVISE_TEMPLATE
    elif example.dataset == "humaneval":
        template = HUMANEVAL_REVISE_TEMPLATE
    elif example.dataset == "mbpp":
        template = MBPP_REVISE_TEMPLATE
    elif example.dataset in {"hotpotqa", "drop"}:
        template = QA_REVISE_TEMPLATE
    else:
        raise ValueError(f"unsupported Self-Refine dataset: {example.dataset}")
    return template.format(problem=example.prompt, solution=solution, feedback=feedback)


def parse_review_decision(content: str) -> ReviewDecision:
    parsed = _json_object(content)
    if parsed is None:
        raise SelfRefineProtocolError("reviewer did not return a JSON object")
    raw_result = parsed.get("review_result")
    if isinstance(raw_result, bool):
        accepted = raw_result
    elif isinstance(raw_result, str) and raw_result.strip().lower() in {"true", "false"}:
        accepted = raw_result.strip().lower() == "true"
    else:
        raise SelfRefineProtocolError("review_result must be an explicit boolean")
    feedback = parsed.get("feedback", "")
    if not isinstance(feedback, str):
        raise SelfRefineProtocolError("review feedback must be a string")
    return ReviewDecision(accepted, feedback)


class SelfRefineBaseline:
    method = "self_refine"
    max_rounds = 3
    minimum_calls = 2
    maximum_calls = 7

    def __init__(self, client: OpenRouterClient) -> None:
        self.client = client

    def run(self, example: BenchmarkExample) -> BaselineResult:
        generation_prompt = build_cot_prompt(example)
        responses = []
        runtime_prompts = []
        generation_messages = [
            {"role": "system", "content": generation_prompt.system},
            {"role": "user", "content": generation_prompt.user},
        ]
        runtime_prompts.append(generation_messages)
        response = self.client.chat(
            generation_messages,
            metadata={
                "method": self.method,
                "dataset": example.dataset,
                "sample_id": example.sample_id,
                "role": "generator",
                "prompt_provenance": generation_prompt.provenance,
            },
        )
        responses.append(response)
        solution = parse_io_response(example.dataset, response.content)
        initial_solution = solution
        iterations = []
        stop_reason = "max_rounds_exhausted"
        inferred = example.dataset in {"hotpotqa", "drop"}

        review_system = (
            "Return one valid JSON object and no surrounding markdown. Required fields are "
            "'review_result' (a JSON boolean) and 'feedback' (a string)."
        )
        revise_system = (
            "Return one valid JSON object and no surrounding markdown. The required string field "
            "is 'solution'. For code tasks, put the complete Python program in that field."
        )
        for round_index in range(self.max_rounds):
            review_user = build_review_prompt(example, solution)
            review_messages = [
                {"role": "system", "content": review_system},
                {"role": "user", "content": review_user},
            ]
            runtime_prompts.append(review_messages)
            review_response = self.client.chat(
                review_messages,
                metadata={
                    "method": self.method,
                    "dataset": example.dataset,
                    "sample_id": example.sample_id,
                    "role": "reviewer",
                    "round_index": round_index,
                    "prompt_provenance": (
                        "paper-faithful/inferred"
                        if inferred
                        else "upstream-user/adapted-schema"
                    ),
                },
            )
            responses.append(review_response)
            decision = parse_review_decision(review_response.content)
            iteration = {
                "round_index": round_index,
                "solution_before": solution,
                "accepted": decision.accepted,
                "feedback": decision.feedback,
                "solution_after": None,
            }
            if decision.accepted:
                iterations.append(iteration)
                stop_reason = "review_accepted"
                break

            revise_user = build_revise_prompt(example, solution, decision.feedback)
            revise_messages = [
                {"role": "system", "content": revise_system},
                {"role": "user", "content": revise_user},
            ]
            runtime_prompts.append(revise_messages)
            revise_response = self.client.chat(
                revise_messages,
                metadata={
                    "method": self.method,
                    "dataset": example.dataset,
                    "sample_id": example.sample_id,
                    "role": "reviser",
                    "round_index": round_index,
                    "prompt_provenance": (
                        "paper-faithful/inferred"
                        if inferred
                        else "upstream-user/adapted-schema"
                    ),
                },
            )
            responses.append(revise_response)
            solution = parse_io_response(example.dataset, revise_response.content)
            iteration["solution_after"] = solution
            iterations.append(iteration)

        prompt_hash = hashlib.sha256(
            json.dumps(
                runtime_prompts,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return BaselineResult(
            method=self.method,
            dataset=example.dataset,
            sample_id=example.sample_id,
            prediction=solution,
            prompt_sha256=prompt_hash,
            responses=tuple(responses),
            artifacts={
                "initial_solution": initial_solution,
                "iterations": iterations,
                "stop_reason": stop_reason,
                "max_rounds": self.max_rounds,
                "prompt_provenance": (
                    "paper-faithful/inferred"
                    if inferred
                    else "upstream-user/adapted-schema"
                ),
            },
        )
