"""Three-candidate MedPrompt ensemble with five shuffled LLM votes."""

from __future__ import annotations

import hashlib
import json
import random

from awo.baselines.cot import build_cot_prompt
from awo.baselines.cot_sc import parse_selector_letter
from awo.baselines.io import IOPrompt, parse_io_response
from awo.baselines.models import BaselineResult
from awo.benchmarks.data import BenchmarkExample
from awo.llm import OpenRouterClient

GSM8K_CANDIDATE_TEMPLATE = (
    "{question}\n"
    "Please reason step by step, and to ensure accuracy, provide the correct answer in "
    "the final, without any additional text.\n"
)

GENERAL_VOTER_TEMPLATE = (
    "You are given a problem:\n"
    "{question}\n\n"
    "Here is a list of possible solutions to the problem:\n"
    "{solutions}\n\n"
    "Using the inputs above, your goal is to choose the best solution to the problem.\n"
    "The main consideration is that the solution can fully solve the problem in a correct "
    "and robust manner.\n"
    "Provide your final decision by writing the chosen solution letter.\n\n"
    "Please follow the required format in your response.\n"
)

CODE_VOTER_TEMPLATE = (
    "Given the question described as follows: {question}\n"
    "Several solutions have been generated to address the given question. They are as follows:\n"
    "{solutions}\n\n"
    "Carefully evaluate these solutions and identify the solution that is more capable of "
    "solving the problem compared to other solutions, as this is crucial for problem-solving.\n\n"
    'In the "thought" field, provide a detailed explanation of your thought process. In the '
    '"solution_letter" field, output only the single letter ID (A, B, C, etc.) corresponding '
    "to the solution. Do not include any additional text or explanation in the "
    '"solution_letter" field.\n'
)


def build_medprompt_candidate_prompt(example: BenchmarkExample) -> IOPrompt:
    if example.dataset != "gsm8k":
        return build_cot_prompt(example)
    user = GSM8K_CANDIDATE_TEMPLATE.format(question=example.prompt)
    system = (
        "Return one valid JSON object and no surrounding markdown. "
        "The required string field is 'solution'. Preserve the requested reasoning inside it."
    )
    return IOPrompt("gsm8k", system, user, "upstream-user/adapted-schema")


def _permuted_listing(candidates: list[str], permutation: list[int]) -> str:
    return "".join(
        f"{chr(65 + displayed)}: \n{candidates[original]}\n\n\n"
        for displayed, original in enumerate(permutation)
    )


def build_medprompt_voter_prompt(
    example: BenchmarkExample, candidates: list[str], permutation: list[int]
) -> str:
    if sorted(permutation) != list(range(len(candidates))) or len(candidates) != 3:
        raise ValueError("MedPrompt requires a permutation of exactly three candidates")
    template = (
        CODE_VOTER_TEMPLATE
        if example.dataset in {"humaneval", "mbpp"}
        else GENERAL_VOTER_TEMPLATE
    )
    return template.format(
        question=example.prompt,
        solutions=_permuted_listing(candidates, permutation),
    )


def choose_vote_winner(votes: list[int], candidate_count: int = 3) -> int:
    if len(votes) != 5 or any(not 0 <= vote < candidate_count for vote in votes):
        raise ValueError("MedPrompt requires exactly five valid votes")
    counts = [votes.count(index) for index in range(candidate_count)]
    return max(range(candidate_count), key=lambda index: (counts[index], -index))


class MedPromptBaseline:
    method = "medprompt"
    candidate_count = 3
    vote_count = 5
    expected_calls = 8

    def __init__(self, client: OpenRouterClient, *, seed: int = 0) -> None:
        self.client = client
        self.seed = seed

    def run(self, example: BenchmarkExample) -> BaselineResult:
        candidate_prompt = build_medprompt_candidate_prompt(example)
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

        rng = random.Random(f"{self.seed}:{example.sample_id}")
        permutations = []
        vote_letters = []
        vote_indices = []
        voter_prompts = []
        voter_system = (
            "Return one valid JSON object with a single string field 'solution_letter'. "
            "Its value must be exactly one of A, B, or C."
        )
        for vote_index in range(self.vote_count):
            permutation = list(range(self.candidate_count))
            rng.shuffle(permutation)
            permutations.append(permutation)
            voter_prompt = build_medprompt_voter_prompt(example, candidates, permutation)
            voter_prompts.append(voter_prompt)
            response = self.client.chat(
                [
                    {"role": "system", "content": voter_system},
                    {"role": "user", "content": voter_prompt},
                ],
                metadata={
                    "method": self.method,
                    "dataset": example.dataset,
                    "sample_id": example.sample_id,
                    "role": "voter",
                    "vote_index": vote_index,
                    "permutation": permutation,
                    "seed": self.seed,
                    "prompt_provenance": (
                        "upstream-user/adapted-schema"
                        if example.dataset in {"gsm8k", "math", "humaneval", "mbpp"}
                        else "paper-faithful/inferred"
                    ),
                },
            )
            responses.append(response)
            letter = parse_selector_letter(response.content, self.candidate_count)
            vote_letters.append(letter)
            vote_indices.append(permutation[ord(letter) - 65])

        selected_index = choose_vote_winner(vote_indices, self.candidate_count)
        vote_counts = [vote_indices.count(index) for index in range(self.candidate_count)]
        prompt_hash = hashlib.sha256(
            json.dumps(
                {
                    "candidate_prompt_sha256": candidate_prompt.sha256,
                    "voter_system": voter_system,
                    "voter_prompts": voter_prompts,
                    "seed": self.seed,
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
                "permutations": permutations,
                "vote_letters": vote_letters,
                "vote_original_indices": vote_indices,
                "vote_counts": vote_counts,
                "selected_index": selected_index,
                "seed": self.seed,
            },
        )
