"""Deterministic scorers for QA and mathematical AFlow benchmarks."""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from sympy import N, simplify
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

NUMBER_PATTERN = re.compile(
    r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)
TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)


@dataclass(frozen=True)
class ScoreResult:
    score: float
    extracted: Any
    details: dict[str, Any] = field(default_factory=dict)


def normalize_answer(value: Any) -> str:
    text = str(value).lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(reference: Any, prediction: Any) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    if not prediction_tokens or not reference_tokens:
        return 0.0
    common = Counter(prediction_tokens) & Counter(reference_tokens)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(prediction_tokens)
    recall = same / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def best_token_f1(reference: Any, prediction: Any) -> ScoreResult:
    references = [part.strip() for part in str(reference).split("|") if part.strip()]
    predictions = [part.strip() for part in str(prediction).split("|") if part.strip()]
    scores = [token_f1(gold, candidate) for gold in references for candidate in predictions]
    return ScoreResult(max(scores, default=0.0), str(prediction))


def extract_last_number(value: Any) -> float | None:
    matches = NUMBER_PATTERN.findall(str(value))
    if not matches:
        return None
    try:
        return float(matches[-1].replace(",", ""))
    except ValueError:
        return None


def score_gsm8k(reference: Any, prediction: Any) -> ScoreResult:
    expected = extract_last_number(reference)
    extracted = extract_last_number(prediction)
    if expected is None or extracted is None:
        return ScoreResult(0.0, extracted, {"expected_number": expected})
    score = 1.0 if math.isclose(expected, extracted, abs_tol=1e-6) else 0.0
    return ScoreResult(score, extracted, {"expected_number": expected})


def extract_boxed_answer(value: Any) -> str:
    text = str(value)
    answers = []
    start = 0
    marker = r"\boxed{"
    while True:
        marker_index = text.find(marker, start)
        if marker_index < 0:
            break
        content_start = marker_index + len(marker)
        depth = 1
        index = content_start
        while index < len(text) and depth:
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
            index += 1
        if depth == 0:
            answers.append(text[content_start : index - 1].strip())
            start = index
        else:
            start = content_start
    if answers:
        return answers[-1]
    sentences = [part.strip() for part in re.split(r"(?<!\d)[.!?]\s+", text) if part.strip()]
    return sentences[-1] if sentences else ""


def _normalize_math(value: Any) -> str:
    text = str(value).strip()
    if text.startswith("$") and text.endswith("$"):
        text = text[1:-1]
    replacements = {
        r"\left": "",
        r"\right": "",
        r"\dfrac": r"\frac",
        r"\tfrac": r"\frac",
        r"\%": "%",
        "−": "-",
        "–": "-",
        "π": "pi",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\s+", "", text)
    text = text.rstrip(".")
    return text


def _parse_numeric(value: Any) -> float | None:
    text = _normalize_math(value).replace(",", "")
    percentage = text.endswith("%")
    if percentage:
        text = text[:-1].rstrip("\\")
    fraction = re.fullmatch(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", text)
    try:
        if fraction:
            number = float(fraction.group(1)) / float(fraction.group(2))
        elif re.fullmatch(r"[-+]?\d+(?:\.\d+)?/[-+]?\d+(?:\.\d+)?", text):
            numerator, denominator = text.split("/", 1)
            number = float(numerator) / float(denominator)
        else:
            number = float(text)
        return number / 100 if percentage else number
    except (ValueError, ZeroDivisionError):
        return None


def _sympy_text(value: Any) -> str:
    text = _normalize_math(value)
    text = text.replace(r"\cdot", "*").replace(r"\times", "*")
    text = text.replace(r"\pi", "pi")
    text = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", text)
    while True:
        updated = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", text)
        if updated == text:
            break
        text = updated
    text = text.replace("{", "(").replace("}", ")")
    return text


def _archive_numeric(value: Any) -> float | None:
    text = re.sub(",", "", str(value))
    try:
        return float(text)
    except ValueError:
        if text.endswith("%"):
            text = text[:-1]
            if text.endswith("\\"):
                text = text[:-1]
            try:
                return float(text) / 100
            except ValueError:
                pass
    return None


def source_math_equal(prediction: Any, reference: Any) -> bool:
    """Mirror the published AFlow evaluator, including its limited parsing."""

    if str(prediction) == str(reference):
        return True
    predicted_number = _archive_numeric(prediction)
    expected_number = _archive_numeric(reference)
    if predicted_number is not None and expected_number is not None:
        return math.isclose(predicted_number, expected_number, abs_tol=1e-3)

    def parse(value: Any) -> Any:
        try:
            return parse_expr(str(value))
        except Exception:
            return value

    predicted_expr = parse(prediction)
    expected_expr = parse(reference)
    try:
        if simplify(predicted_expr - expected_expr) == 0:
            return True
    except Exception:
        pass
    try:
        return bool(math.isclose(N(predicted_expr), N(expected_expr), abs_tol=1e-3))
    except Exception:
        return False


def archive_math_equal(prediction: Any, reference: Any) -> bool:
    """Match archived CSV labels, including two inferred presentation rules."""

    if source_math_equal(prediction, reference):
        return True
    predicted_compact = re.sub(r"\s+", "", str(prediction))
    expected_compact = re.sub(r"\s+", "", str(reference))
    bare_tuple = all(
        text.startswith("(") and text.endswith(")") and "," in text
        for text in (str(prediction).strip(), str(reference).strip())
    )
    matrix = all(r"\begin{pmatrix}" in text for text in (str(prediction), str(reference)))
    if (bare_tuple or matrix) and predicted_compact == expected_compact:
        return True
    predicted_percent = predicted_compact.removesuffix(r"\%").removesuffix("%")
    expected_percent = expected_compact.removesuffix(r"\%").removesuffix("%")
    if (predicted_percent, expected_percent) != (predicted_compact, expected_compact):
        try:
            return math.isclose(
                float(predicted_percent), float(expected_percent), abs_tol=1e-3
            )
        except ValueError:
            pass
    return False


def corrected_math_equal(prediction: Any, reference: Any) -> bool:
    predicted = _normalize_math(prediction)
    expected = _normalize_math(reference)
    if predicted == expected:
        return True

    predicted_number = _parse_numeric(predicted)
    expected_number = _parse_numeric(expected)
    if predicted_number is not None and expected_number is not None:
        return math.isclose(predicted_number, expected_number, abs_tol=1e-3)

    try:
        predicted_expr = parse_expr(
            _sympy_text(predicted), transformations=TRANSFORMATIONS, evaluate=True
        )
        expected_expr = parse_expr(
            _sympy_text(expected), transformations=TRANSFORMATIONS, evaluate=True
        )
        if simplify(predicted_expr - expected_expr) == 0:
            return True
        return bool(math.isclose(float(N(predicted_expr)), float(N(expected_expr)), abs_tol=1e-3))
    except Exception:
        return False


def math_equal(prediction: Any, reference: Any, mode: str = "archive") -> bool:
    if mode == "archive":
        return archive_math_equal(prediction, reference)
    if mode == "source":
        return source_math_equal(prediction, reference)
    if mode == "corrected":
        return corrected_math_equal(prediction, reference)
    raise ValueError(f"Unknown MATH scoring mode: {mode}")


def _extract_archive_boxed_answer(value: Any) -> str:
    text = str(value)
    matches = re.findall(r"\\boxed{((?:[^{}]|{[^{}]*})*)}", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    sentences = [part.strip() for part in re.split(r"(?<!\d)[.!?]\s+", text) if part.strip()]
    return sentences[-1] if sentences else ""


def score_math(reference: Any, prediction: Any, mode: str = "archive") -> ScoreResult:
    extractor = (
        _extract_archive_boxed_answer
        if mode in {"archive", "source"}
        else extract_boxed_answer
    )
    expected = extractor(reference)
    extracted = extractor(prediction)
    return ScoreResult(
        1.0 if math_equal(extracted, expected, mode=mode) else 0.0,
        extracted,
        {"expected_extracted": expected, "math_mode": mode},
    )


def score_prediction(
    dataset: str, reference: Any, prediction: Any, *, math_mode: str = "archive"
) -> ScoreResult:
    normalized_dataset = dataset.lower()
    if normalized_dataset in {"hotpotqa", "drop"}:
        return best_token_f1(reference, prediction)
    if normalized_dataset == "gsm8k":
        return score_gsm8k(reference, prediction)
    if normalized_dataset == "math":
        return score_math(reference, prediction, mode=math_mode)
    if normalized_dataset in {"humaneval", "mbpp"}:
        raise ValueError("Code datasets require the sandbox evaluator")
    raise ValueError(f"Unknown dataset: {dataset}")
