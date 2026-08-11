"""Controlled implementations of the operator set used by AFlow Table 1."""

from __future__ import annotations

import json
from typing import Any

from awo.aflow.runtime import AFlowRuntime
from awo.baselines.cot_sc import parse_selector_letter
from awo.baselines.io import _json_object
from awo.benchmarks.code import extract_python_code
from awo.sandbox import PASS_MARKER, SandboxResult

ANSWER_GENERATION_PROMPT = (
    "\nThink step by step and solve the problem.\n"
    '1. In the "thought" field, explain your thinking process in detail.\n'
    '2. In the "answer" field, provide the final answer concisely and clearly. The answer '
    "should be a direct response to the question, without including explanations or reasoning.\n"
    "Your task: {input}\n"
)

GENERAL_SC_ENSEMBLE_PROMPT = (
    "\nGiven the question described as follows: {problem}\n"
    "Several solutions have been generated to address the given question. They are as follows:\n"
    "{solutions}\n\nCarefully evaluate these solutions and identify the answer that appears "
    "most frequently across them. This consistency in answers is crucial for determining the "
    "most reliable solution.\n\nIn the \"thought\" field, provide a detailed explanation of your "
    'thought process. In the "solution_letter" field, output only the single letter ID (A, B, '
    "C, etc.) corresponding to the most consistent solution. Do not include any additional text "
    'or explanation in the "solution_letter" field.\n'
)

QA_SC_ENSEMBLE_PROMPT = (
    "\nSeveral answers have been generated to a same question. They are as follows:\n"
    "{solutions}\n\nIdentify the concise answer that appears most frequently across them. This "
    "consistency in answers is crucial for determining the most reliable solution.\n\nIn the "
    '"thought" field, provide a detailed explanation of your thought process. In the '
    '"solution_letter" field, output only the single letter ID (A, B, C, etc.) corresponding '
    "to the most consistent solution. Do not include any additional text or explanation in the "
    '"solution_letter" field.\n'
)

PYTHON_CODE_VERIFIER_PROMPT = (
    "\nYou are a professional Python programmer. Your task is to write complete, self-contained "
    "code based on a given mathematical problem and output the answer. The code should include "
    "all necessary imports and dependencies, and be ready to run without additional setup or "
    "environment configuration.\n\nProblem description: {problem}\nOther analysis: "
    "{analysis}\n{feedback}\n\nYour code should:\n1. Implement the calculation steps described "
    "in the problem.\n2. Define a function named `solve` that performs the calculation and "
    "returns the result. The `solve` function should not require any input parameters; instead, "
    "it should obtain all necessary inputs from within the function or from globally defined "
    "variables.\n3. `solve` function return the final calculation result.\n\nPlease ensure your "
    "code is efficient, well-commented, and follows Python best practices. The output should be "
    "limited to basic data types such as strings, integers, and floats. It is prohibited to "
    "transmit images or other file formats. The code output is intended for a text-based "
    "language model.\n"
)

REFLECTION_ON_PUBLIC_TEST_PROMPT = (
    "\nGiven a code problem and a python code solution which failed to pass test or execute, you "
    "need to analyze the reason for the failure and propose a better code solution.: \n### "
    "problem\n{problem}\n\n### Code Solution\n{solution}\n\n### Execution Result\n"
    "{exec_pass}\n\n#### Failed Test Case\n{test_fail}\n\nPlease provide a reflection on the "
    "failed test cases and code solution, followed by a better code solution without any "
    "additional text or test cases.\n"
)

CODE_OUTPUT_INSTRUCTION = (
    "\n\nPlease write your code solution in Python. Return ONLY the complete, runnable code "
    "without explanations. Use proper Python syntax and formatting.\nMake sure to include a "
    "function named '{entry_point}' in your solution. This function will be the entry point for "
    "the program."
)

PROGRAMMER_OUTPUT_MARKER = "__AWO_PROGRAMMER_OUTPUT__"


class AFlowOperatorError(RuntimeError):
    """Raised when an operator cannot produce a protocol-valid result."""


def _structured_fields(content: str, required: tuple[str, ...]) -> dict[str, Any]:
    parsed = _json_object(content)
    if parsed is None or any(not isinstance(parsed.get(key), str) for key in required):
        raise AFlowOperatorError(f"operator response must contain string fields {required}")
    return parsed


def _solution_listing(solutions: list[str]) -> str:
    if not 1 <= len(solutions) <= 26:
        raise ValueError("ScEnsemble requires between 1 and 26 solutions")
    return "".join(
        f"{chr(65 + index)}: \n{solution}\n\n\n"
        for index, solution in enumerate(solutions)
    )


class Custom:
    def __init__(self, runtime: AFlowRuntime) -> None:
        self.runtime = runtime

    async def __call__(self, input: str, instruction: str) -> dict[str, str]:
        response = await self.runtime.chat(
            instruction + input, operator="Custom", metadata={"format": "text"}
        )
        return {"response": response.content}


class AnswerGenerate:
    def __init__(self, runtime: AFlowRuntime) -> None:
        self.runtime = runtime

    async def __call__(self, input: str) -> dict[str, str]:
        prompt = ANSWER_GENERATION_PROMPT.format(input=input)
        response = await self.runtime.chat(
            prompt,
            operator="AnswerGenerate",
            system=(
                "Return one valid JSON object and no surrounding markdown. Required string "
                "fields are 'thought' and 'answer'."
            ),
            metadata={"format": "adapted-json"},
        )
        parsed = _structured_fields(response.content, ("thought", "answer"))
        return {"thought": parsed["thought"], "answer": parsed["answer"]}


class CustomCodeGenerate:
    def __init__(self, runtime: AFlowRuntime) -> None:
        self.runtime = runtime

    async def __call__(
        self, problem: str, entry_point: str, instruction: str
    ) -> dict[str, str]:
        prompt = instruction + problem + CODE_OUTPUT_INSTRUCTION.format(entry_point=entry_point)
        response = await self.runtime.chat(
            prompt,
            operator="CustomCodeGenerate",
            metadata={"format": "code", "entry_point": entry_point},
        )
        code = extract_python_code(response.content)
        if not code.strip():
            raise AFlowOperatorError("CustomCodeGenerate returned no Python code")
        return {"response": code}


class ScEnsemble:
    def __init__(self, runtime: AFlowRuntime, *, qa_mode: bool = False) -> None:
        self.runtime = runtime
        self.qa_mode = qa_mode

    async def __call__(self, solutions: list[str], problem: str) -> dict[str, str]:
        listing = _solution_listing(solutions)
        template = QA_SC_ENSEMBLE_PROMPT if self.qa_mode else GENERAL_SC_ENSEMBLE_PROMPT
        prompt = template.format(problem=problem, solutions=listing)
        allowed = ", ".join(chr(65 + index) for index in range(len(solutions)))
        response = await self.runtime.chat(
            prompt,
            operator="ScEnsemble",
            system=(
                "Return one valid JSON object and no surrounding markdown. The required string "
                f"field is 'solution_letter' and must be exactly one of: {allowed}."
            ),
            metadata={"format": "adapted-json", "candidate_count": len(solutions)},
        )
        letter = parse_selector_letter(response.content, len(solutions))
        return {"response": solutions[ord(letter) - 65]}


def _programmer_source(code: str) -> str:
    return (
        f"{code}\n\n"
        "import json as __awo_json\n"
        "__awo_value = solve()\n"
        f"print({PROGRAMMER_OUTPUT_MARKER!r} + __awo_json.dumps(__awo_value))\n"
        f"print({PASS_MARKER!r})\n"
    )


def _programmer_output(result: SandboxResult) -> str | None:
    if not result.passed:
        return None
    for line in reversed(result.stdout.splitlines()):
        if line.startswith(PROGRAMMER_OUTPUT_MARKER):
            payload = line[len(PROGRAMMER_OUTPUT_MARKER) :]
            try:
                return str(json.loads(payload))
            except json.JSONDecodeError:
                return None
    return None


class Programmer:
    def __init__(self, runtime: AFlowRuntime, *, max_attempts: int = 3) -> None:
        if runtime.sandbox is None:
            raise ValueError("Programmer requires the controlled Docker sandbox")
        self.runtime = runtime
        self.max_attempts = max_attempts

    async def __call__(self, problem: str, analysis: str = "None") -> dict[str, Any]:
        feedback = ""
        code = ""
        output = "No code generated"
        statuses = []
        for attempt in range(self.max_attempts):
            prompt = PYTHON_CODE_VERIFIER_PROMPT.format(
                problem=problem, analysis=analysis, feedback=feedback
            ) + CODE_OUTPUT_INSTRUCTION.format(entry_point="solve")
            response = await self.runtime.chat(
                prompt,
                operator="Programmer",
                metadata={"format": "code", "attempt": attempt},
            )
            code = extract_python_code(response.content)
            sandbox_result = self.runtime.sandbox.run(_programmer_source(code))
            parsed_output = _programmer_output(sandbox_result)
            statuses.append(sandbox_result.status)
            if parsed_output is not None:
                return {"code": code, "output": parsed_output, "sandbox_statuses": statuses}
            output = (sandbox_result.stderr or sandbox_result.stdout).strip() or (
                f"sandbox status: {sandbox_result.status}"
            )
            feedback = (
                "\nThe result of the error from the code you wrote in the previous round:\n"
                f"Code: {code}\n\nStatus: Error, {output}"
            )
        return {"code": code, "output": output, "sandbox_statuses": statuses}


def _public_test_source(solution: str, public_tests: str) -> str:
    return f"{solution}\n\n{public_tests}\n\nprint({PASS_MARKER!r})\n"


class Test:
    def __init__(self, runtime: AFlowRuntime) -> None:
        if runtime.sandbox is None:
            raise ValueError("Test requires the controlled Docker sandbox")
        self.runtime = runtime

    async def __call__(
        self,
        problem: str,
        solution: str,
        entry_point: str,
        public_tests: str,
        test_loop: int = 3,
    ) -> dict[str, Any]:
        del entry_point  # Kept in the compatibility signature; tests already call the entry point.
        statuses = []
        for round_index in range(test_loop):
            result = self.runtime.sandbox.run(_public_test_source(solution, public_tests))
            statuses.append(result.status)
            if result.passed:
                return {"result": True, "solution": solution, "sandbox_statuses": statuses}
            failure = (result.stderr or result.stdout).strip() or result.status
            prompt = REFLECTION_ON_PUBLIC_TEST_PROMPT.format(
                problem=problem,
                solution=solution,
                exec_pass=f"executed unsuccessfully, error:\n{failure}",
                test_fail=failure,
            ) + CODE_OUTPUT_INSTRUCTION.format(entry_point="candidate")
            response = await self.runtime.chat(
                prompt,
                operator="Test",
                metadata={"format": "code", "round_index": round_index},
            )
            solution = extract_python_code(response.content)
        result = self.runtime.sandbox.run(_public_test_source(solution, public_tests))
        statuses.append(result.status)
        return {
            "result": result.passed,
            "solution": solution,
            "sandbox_statuses": statuses,
        }
