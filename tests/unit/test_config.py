from pathlib import Path

import pytest

from awo.config import ConfigError, config_fingerprint, load_config

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_active_configs_resolve_gpt_4o_mini() -> None:
    expected_names = {
        "configs/smoke.yaml": "aflow-table1-smoke-gpt-4o-mini",
        "configs/pilot.yaml": "aflow-table1-pilot-gpt-4o-mini",
        "configs/paper/table1.yaml": "aflow-table1-gpt-4o-mini",
    }

    for relative_path, expected_name in expected_names.items():
        config = load_config(REPO_ROOT / relative_path)
        assert config["name"] == expected_name
        assert config["model"]["provider"] == "openrouter"
        assert config["model"]["model"] == "openai/gpt-4o-mini"
        assert config["model"]["temperature"] == 1.0

    table1 = load_config(REPO_ROOT / "configs/paper/table1.yaml")
    assert table1["test_repeats"] == 3
    assert len(table1["datasets"]) == 6
    assert len(table1["methods"]) == 8


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
