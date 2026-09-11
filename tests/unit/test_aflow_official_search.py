import math
from pathlib import Path

import pytest

from awo.aflow.official_search import OfficialSearchConfig, original_mixed_probabilities
from awo.config import load_config


def test_original_parent_probability_uses_upstream_constants() -> None:
    probabilities = original_mixed_probabilities([0.8, 0.7])
    expected_first = 0.3 * 0.5 + 0.7 * (math.exp(16) / (math.exp(16) + math.exp(14)))
    assert probabilities[0] == pytest.approx(expected_first)
    assert sum(probabilities) == pytest.approx(1.0)


def test_original_humaneval_config_uses_upstream_search_defaults() -> None:
    config = load_config(Path("configs/aflow/original_humaneval.yaml"))
    aflow = config["aflow"]
    search = OfficialSearchConfig(
        sample=aflow["sample"],
        max_rounds=aflow["max_rounds"],
        validation_rounds=aflow["validation_rounds"],
        check_convergence=aflow["check_convergence"],
        max_concurrency=aflow["max_concurrency"],
    )
    assert search.sample == 4
    assert search.max_rounds == 20
    assert search.validation_rounds == 5
