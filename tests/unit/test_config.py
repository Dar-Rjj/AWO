from pathlib import Path

import pytest

from awo.config import ConfigError, config_fingerprint, load_config

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_table1_config_resolves_model() -> None:
    config = load_config(REPO_ROOT / "configs/paper/table1.yaml")

    assert config["name"] == "aflow-table1-deepseek-chat"
    assert config["model"]["provider"] == "openrouter"
    assert config["model"]["model"] == "deepseek/deepseek-chat"
    assert config["model"]["temperature"] == 1.0
    assert config["test_repeats"] == 3
    assert len(config["datasets"]) == 6
    assert len(config["methods"]) == 8


def test_fingerprint_excludes_runtime_path() -> None:
    config = load_config(REPO_ROOT / "configs/smoke.yaml")
    copied = dict(config)
    copied["_config_path"] = "/different/machine/config.yaml"

    assert config_fingerprint(config) == config_fingerprint(copied)


def test_extends_and_environment_expansion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    parent = tmp_path / "parent.yaml"
    parent.write_text("schema_version: 1\nnested:\n  first: 1\n  second: 2\n", encoding="utf-8")
    child = tmp_path / "child.yaml"
    child.write_text(
        "extends: parent.yaml\nschema_version: 1\nnested:\n  second: 3\n"
        "value: $ENV{AWO_TEST_VALUE}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AWO_TEST_VALUE", "resolved")

    config = load_config(child)

    assert config["nested"] == {"first": 1, "second": 3}
    assert config["value"] == "resolved"


def test_missing_environment_variable_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "schema_version: 1\nvalue: $ENV{AWO_MISSING_VALUE}\n", encoding="utf-8"
    )
    monkeypatch.delenv("AWO_MISSING_VALUE", raising=False)

    with pytest.raises(ConfigError, match="AWO_MISSING_VALUE"):
        load_config(config_path)
