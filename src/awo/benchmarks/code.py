"""Extraction, harness construction, and scoring for code benchmarks."""

from __future__ import annotations

import ast
import re
from collections.abc import Sequence
from typing import Any

from awo.benchmarks.scoring import ScoreResult
from awo.sandbox import PASS_MARKER, DockerSandbox

FENCE_PATTERN = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)
PRELUDE = """\
import math
import hashlib
import re
import collections
import itertools
import functools
import statistics
from typing import Any, Dict, List, Optional, Set, Tuple

def encode_cyclic(s):
    groups = [s[(3 * i):min((3 * i + 3), len(s))] for i in range((len(s) + 2) // 3)]
    groups = [(group[1:] + group[0]) if len(group) == 3 else group for group in groups]
    return "".join(groups)

def encode_shift(s):
    return "".join(chr(((ord(ch) + 5 - ord("a")) % 26) + ord("a")) for ch in s)

def poly(xs, x):
    return sum(coeff * math.pow(x, i) for i, coeff in enumerate(xs))
"""


def _valid_python(value: str) -> bool:
    try:
        ast.parse(value)
        return True
    except SyntaxError:
        return False


def extract_python_code(response: Any) -> str:
    """Choose the longest syntactically valid fenced block, then try raw text."""

    text = str(response).strip()
    fenced = [block.strip() for block in FENCE_PATTERN.findall(text)]
    valid = [block for block in fenced if _valid_python(block)]
    if valid:
        return max(valid, key=len)
    if _valid_python(text):
        return text
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if re.match(r"\s*(?:from|import|def|class|@)\b", line):
            candidate = "\n".join(lines[index:]).strip()
            if _valid_python(candidate):
                return candidate
    return text


def build_humaneval_harness(solution: str, test: str, entry_point: str) -> str:
    return "\n".join(
        [
            PRELUDE,
            extract_python_code(solution),
            test,
            f"assert callable(globals()[{entry_point!r}])",
            f"check(globals()[{entry_point!r}])",
            f"print({PASS_MARKER!r})",
        ]
    )


def build_mbpp_harness(solution: str, tests: str | Sequence[str]) -> str:
    test_source = tests if isinstance(tests, str) else "\n".join(str(item) for item in tests)
    tree = ast.parse(test_source)
    defines_check = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "check"
        for node in tree.body
    )
    check_call = "check()" if defines_check else ""
    return "\n".join(
        [
            PRELUDE,
            extract_python_code(solution),
            test_source,
            check_call,
            f"print({PASS_MARKER!r})",
        ]
    )


def score_code(
    dataset: str,
    prediction: Any,
    *,
    tests: str | Sequence[str],
    sandbox: DockerSandbox,
    entry_point: str | None = None,
) -> ScoreResult:
    normalized = dataset.lower()
    if normalized == "humaneval":
        if not entry_point:
            raise ValueError("HumanEval scoring requires entry_point")
        source = build_humaneval_harness(str(prediction), str(tests), entry_point)
    elif normalized == "mbpp":
        source = build_mbpp_harness(str(prediction), tests)
    else:
        raise ValueError(f"unsupported code benchmark: {dataset}")
    result = sandbox.run(source)
    return ScoreResult(
        float(result.passed),
        extract_python_code(prediction),
        {
            "sandbox_status": result.status,
            "exit_code": result.exit_code,
            "duration_seconds": result.duration_seconds,
            "image": result.image,
            "stderr": result.stderr[-2000:],
        },
    )
