import pytest

from awo.benchmarks import (
    extract_boxed_answer,
    extract_last_number,
    math_equal,
    score_prediction,
    token_f1,
)


def test_qa_normalization_and_f1() -> None:
    assert token_f1("The Eiffel Tower", "eiffel tower!") == 1.0
    assert token_f1("red blue", "red green") == 0.5
    assert token_f1("", "") == 0.0


def test_drop_uses_best_reference_and_prediction_part() -> None:
    result = score_prediction("drop", "12|twelve", "reasoning|Twelve.")

    assert result.score == 1.0


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("answer: 1,234.5", 1234.5),
        ("work 2 then -3.2e2", -320.0),
        ("nothing", None),
    ],
)
def test_extract_last_number(text, expected) -> None:
    assert extract_last_number(text) == expected


def test_gsm8k_scores_last_number() -> None:
    assert score_prediction("gsm8k", "36", "First 64, therefore 36").score == 1.0
    assert score_prediction("gsm8k", "36", "35").score == 0.0


def test_nested_and_last_boxed_answer() -> None:
    assert extract_boxed_answer(r"first \boxed{1}, last \boxed{\frac{2}{3}}") == r"\frac{2}{3}"


@pytest.mark.parametrize(
    ("prediction", "reference"),
    [
        ("0.5", r"\frac{1}{2}"),
        ("25%", "0.25"),
        ("x+x", "2*x"),
        (r"\sqrt{4}", "2"),
    ],
)
def test_math_equivalence(prediction, reference) -> None:
    assert math_equal(prediction, reference, mode="corrected")


def test_archive_math_mode_uses_declared_upstream_fallback() -> None:
    assert math_equal("x+x", "2*x", mode="source")
    assert not math_equal("0.5", r"\frac{1}{2}", mode="source")
    assert math_equal("0.5", r"\frac{1}{2}", mode="corrected")


def test_archive_math_mode_matches_inferred_csv_presentation_rules() -> None:
    assert not math_equal("30", r"30\%", mode="source")
    assert math_equal("30", r"30\%", mode="archive")
    assert math_equal("(2, -1)", "(2,-1)", mode="archive")


def test_math_score_extracts_boxed() -> None:
    result = score_prediction("math", r"solution \boxed{23}", "Thus \\boxed{23}.")

    assert result.score == 1.0
    assert result.details["expected_extracted"] == "23"
