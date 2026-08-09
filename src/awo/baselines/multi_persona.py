"""Three-persona, two-round debate followed by one synthesis call."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from awo.baselines.io import _hotpot_context, _json_object, parse_io_response
from awo.baselines.models import BaselineResult
from awo.benchmarks.data import BenchmarkExample
from awo.llm import OpenRouterClient

ROLE_PREFIX = "You are a {role}. Based on your professional knowledge and thinking style,"

MATH_ROLES = (
    "Innovative Math Thinker - Math PhD",
    "Critical Reasoning Expert - Math Professor",
    "Computational Thinking Specialist - Math And Computer Science Researcher",
)
CODE_ROLES = (
    "Innovative CS Thinker - ICPC Competitor",
    "Critical Reasoning Expert - Math Professor",
    "Computational Thinking Specialist - Computer Science Researcher",
)
QA_ROLES = (
    "Comprehensive Knowledge Maven - Information Scientist",
    "Analytical Insight Specialist - Cognitive Psychologist",
    "Fact Verification Expert - Data Analyst",
)

TEMPLATES = {
    "gsm8k": {
        "initial": "\n{question}\nPlease think step by step and then solve this task.\n",
        "debate": (
            "\n{question}\nConsidering the solutions provided by other agents as additional "
            "suggestions. Please think carefully and provide an updated answer.\n"
        ),
        "final": (
            "\n{question}\nConsidering all the thinking processes and answers:\n{all_thinking}\n"
            "{all_answers}\nPlease reason carefully and provide the final answer. To ensure "
            "accuracy, At the end, provide the final answer in solution field with the format "
            '"Answer is <number>", where <number> is a single number, without any additional '
            "information or explanation.\n"
        ),
    },
    "math": {
        "initial": (
            "\n{question}\nPlease reason step by step, the reason process can be put in the "
            'thinking field. At the end, provide the final answer in the answer field with the '
            'format "\\boxed{{<number>}}", where <number> is a math answer(an expression or '
            "number), without any additional information or explanation.\nMake sure the output "
            "is wrapped with correct xml tags!\n"
        ),
        "debate": (
            "\n{question}\nConsidering the solutions provided by other agents as additional "
            "suggestions, the reason process can be put in the thinking field. Please think "
            'carefully and provide an updated answer in the answer field with the format '
            '"\\boxed{{<number>}}", where <number> is a math answer(an expression or number), '
            "without any additional information or explanation.\nMake sure the output is "
            "wrapped with correct xml tags!\n"
        ),
        "final": (
            "\n{question}\nConsidering all the thinking processes and answers:\n{all_thinking}\n"
            "{all_answers}\n\nThe thinking process can be put in the thinking field.\nPlease "
            'reason carefully and provide the final answer in the answer field with the format '
            '"\\boxed{{<number>}}", where <number> is a math answer(an expression or number), '
            "without any additional information or explanation.\nMake sure the output is "
            "wrapped with correct xml tags!\n"
        ),
    },
    "hotpotqa": {
        "initial": (
            "\nGiven a question and context, please think step by step and then solve this "
            "task.\n\n"
            "Question: {question}\nContext: {relevant_context}\n"
        ),
        "debate": (
            "\nGiven a question and context,\n\nQuestion: {question}\nContext: "
            "{relevant_context}\n\nConsidering the solutions provided by other agents as "
            "additional "
            "suggestions. Please think carefully and provide an updated answer.\n"
        ),
        "final": (
            "\nGiven a question and context,\n\nQuestion: {question}\nContext: "
            "{relevant_context}\n\nConsidering all the thinking processes and answers:\n"
            "{all_thinking}\n{all_answers}\nPlease reason carefully and provide the final answer. "
            "Give the final answer in solution field. You MUST Keep the answer very concise in "
            "a few words, without any additional information.\n"
        ),
    },
    "humaneval": {
        "initial": (
            "\n{question}\nPlease provide a step-by-step explanation in text, followed by your "
            "Python function without any additional text or test cases. \n"
        ),
        "debate": (
            "\n{question}\nConsidering the solutions provided by other agents as additional "
            "suggestions. Please think carefully and provide an updated python function without "
            "any additional text or test cases. \n"
        ),
        "final": (
            "\n{question}\nConsidering all the thinking processes and answers:\n{all_thinking}\n"
            "{all_answers}\nPlease reason carefully and provide the final answer. Make sure the "
            "code output is wrapped with ```python``` without any additional text or test cases.\n"
        ),
    },
    "mbpp": {
        "initial": (
            "\n{question}\nPlease provide a step-by-step explanation in text, followed by your "
            "Python function, ensure the output code is self-contained, meaning it should have "
            "the correct function name and return statement, without any additional text."
        ),
        "debate": (
            "\n{question}\nConsidering the solutions provided by other agents as additional "
            "suggestions. Please think carefully and provide an updated self-contained python "
            "function which meaning it should have the correct function name and return "
            "statement, but it shouldn't have any additional text or test cases. \n"
        ),
        "final": (
            "\n{question}\nConsidering all the thinking processes and answers:\n{all_thinking}\n"
            "{all_answers}\nPlease reason carefully and provide the final answer. Make sure the "
            "output code is self-contained, meaning it should have the correct function name "
            "and return statement, without any additional text."
        ),
    },
    "drop": {
        "initial": "\n{question}\nPlease think step by step and then solve this task.\n",
        "debate": (
            "\n{question}\nConsidering the solutions provided by other agents as additional "
            "suggestions. Please think carefully and provide an updated answer.\n"
        ),
        "final": (
            "\n{question}\nConsidering all the thinking processes and answers:\n{all_thinking}\n"
            "{all_answers}\nPlease reason carefully and provide the final answer. Give the final "
            "answer in solution field. Keep the answer concise, without explanation.\n"
        ),
    },
}


@dataclass(frozen=True)
class PersonaTurn:
    thinking: str
    answer: str


def roles_for_dataset(dataset: str) -> tuple[str, str, str]:
    if dataset in {"gsm8k", "math"}:
        return MATH_ROLES
    if dataset in {"humaneval", "mbpp"}:
        return CODE_ROLES
    if dataset in {"hotpotqa", "drop"}:
        return QA_ROLES
    raise ValueError(f"unsupported MultiPersona dataset: {dataset}")


def _template_values(example: BenchmarkExample) -> dict[str, str]:
    values = {"question": example.prompt, "relevant_context": ""}
    if example.dataset == "hotpotqa":
        values["question"] = str(example.metadata["question"])
        values["relevant_context"] = _hotpot_context(example)
    return values


def build_persona_prompt(
    example: BenchmarkExample,
    role: str,
    *,
    round_index: int,
    prior_thinking: list[str] | None = None,
) -> str:
    if round_index not in {0, 1}:
        raise ValueError("MultiPersona has exactly two rounds")
    template = TEMPLATES[example.dataset]["initial" if round_index == 0 else "debate"]
    prompt = ROLE_PREFIX.format(role=role) + template.format(**_template_values(example))
    if round_index == 1:
        if prior_thinking is None or len(prior_thinking) != 3:
            raise ValueError("round two requires all three prior thinking traces")
        roles = roles_for_dataset(example.dataset)
        own_index = roles.index(role)
        context = [f"{role}'s previous round thinking: {prior_thinking[own_index]}"]
        context.extend(
            f"{other_role}'s thinking: {prior_thinking[index]}"
            for index, other_role in enumerate(roles)
            if index != own_index
        )
        prompt += "\n".join(context)
    return prompt


def build_synthesis_prompt(
    example: BenchmarkExample, roles: tuple[str, str, str], final_turns: list[PersonaTurn]
) -> str:
    if len(final_turns) != 3:
        raise ValueError("synthesis requires exactly three final persona turns")
    values = _template_values(example)
    values["all_thinking"] = "\n".join(
        f"{role}'s final thinking: {turn.thinking}"
        for role, turn in zip(roles, final_turns)
    )
    values["all_answers"] = "\n".join(
        f"{role}'s final answer: {turn.answer}"
        for role, turn in zip(roles, final_turns)
    )
    return TEMPLATES[example.dataset]["final"].format(**values)


def parse_persona_turn(content: str) -> PersonaTurn:
    parsed = _json_object(content)
    if parsed is not None:
        thinking = parsed.get("thinking")
        answer = parsed.get("answer")
        if isinstance(thinking, str) and isinstance(answer, str) and answer.strip():
            return PersonaTurn(thinking, answer)
    return PersonaTurn(content.strip(), content.strip())


class MultiPersonaBaseline:
    method = "multi_persona"
    persona_count = 3
    round_count = 2
    expected_calls = 7

    def __init__(self, client: OpenRouterClient) -> None:
        self.client = client

    def run(self, example: BenchmarkExample) -> BaselineResult:
        roles = roles_for_dataset(example.dataset)
        responses = []
        rounds: list[list[PersonaTurn]] = []
        runtime_prompts = []
        provenance = (
            "paper-faithful/inferred"
            if example.dataset == "drop"
            else "upstream-user/adapted-schema"
        )
        debate_system = (
            "Return one valid JSON object and no surrounding markdown. Required string fields "
            "are 'thinking' and 'answer'. For code tasks, put the complete Python program in "
            "the 'answer' field."
        )
        for round_index in range(self.round_count):
            turns = []
            prior = [turn.thinking for turn in rounds[0]] if round_index == 1 else None
            for persona_index, role in enumerate(roles):
                user = build_persona_prompt(
                    example, role, round_index=round_index, prior_thinking=prior
                )
                runtime_prompts.append({"system": debate_system, "user": user})
                response = self.client.chat(
                    [
                        {"role": "system", "content": debate_system},
                        {"role": "user", "content": user},
                    ],
                    metadata={
                        "method": self.method,
                        "dataset": example.dataset,
                        "sample_id": example.sample_id,
                        "role": "debater",
                        "persona_index": persona_index,
                        "persona": role,
                        "round_index": round_index,
                        "prompt_provenance": provenance,
                    },
                )
                responses.append(response)
                turns.append(parse_persona_turn(response.content))
            rounds.append(turns)

        synthesis_user = build_synthesis_prompt(example, roles, rounds[-1])
        synthesis_system = (
            "Return one valid JSON object and no surrounding markdown. The required string field "
            "is 'solution'. For code tasks, put the complete Python program in that field."
        )
        runtime_prompts.append({"system": synthesis_system, "user": synthesis_user})
        response = self.client.chat(
            [
                {"role": "system", "content": synthesis_system},
                {"role": "user", "content": synthesis_user},
            ],
            metadata={
                "method": self.method,
                "dataset": example.dataset,
                "sample_id": example.sample_id,
                "role": "synthesizer",
                "prompt_provenance": provenance,
            },
        )
        responses.append(response)
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
            prediction=parse_io_response(example.dataset, response.content),
            prompt_sha256=prompt_hash,
            responses=tuple(responses),
            artifacts={
                "roles": list(roles),
                "rounds": [
                    [
                        {"thinking": turn.thinking, "answer": turn.answer}
                        for turn in round_turns
                    ]
                    for round_turns in rounds
                ],
                "round_count": self.round_count,
                "persona_count": self.persona_count,
                "prompt_provenance": provenance,
            },
        )
