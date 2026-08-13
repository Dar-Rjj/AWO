from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from awo.table1 import build_plan, execute_plan, require_full_table_confirmation


def _write_jsonl(path: Path, dataset: str, split: str) -> None:
    rows = []
    for index in range(2):
        if dataset == "gsm8k":
            rows.append({"id": f"{split}-{index}", "question": "1+1?", "answer": "2"})
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_plan_orders_search_before_frozen_test_and_is_bounded(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    _write_jsonl(data / "gsm8k_validate.jsonl", "gsm8k", "validate")
    _write_jsonl(data / "gsm8k_test.jsonl", "gsm8k", "test")
    model = tmp_path / "model.yaml"
    model.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "provider": "openrouter",
                "model": "deepseek/deepseek-chat",
                "base_url": "https://openrouter.ai/api/v1",
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "pilot.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "unit-pilot",
                "model_config": "model.yaml",
                "seed": 42,
                "test_repeats": 1,
                "sample_limit": 1,
                "datasets": ["gsm8k"],
                "methods": ["io", "multipersona", "adas", "aflow"],
                "aflow": {"max_rounds": 1},
                "adas": {"max_rounds": 1},
            }
        ),
        encoding="utf-8",
    )
    plan = build_plan(config, data, tmp_path / "runs")
    assert plan["full_table1"] is False
    assert plan["selected_counts"]["gsm8k"] == {"validation": 1, "test": 1}
    assert [job["job_id"] for job in plan["jobs"]] == [
        "test:gsm8k:manual",
        "search:gsm8k:adas",
        "test:gsm8k:adas",
        "search:gsm8k:aflow",
        "test:gsm8k:aflow",
    ]
    manual_command = plan["jobs"][0]["command"]
    assert "multi_persona" in manual_command
    assert "--limit" in manual_command
    assert "--adas-candidate" in plan["jobs"][2]["command"]
    assert "--aflow-candidate" in plan["jobs"][4]["command"]
    assert plan["logical_call_bounds"]["all"] == {"minimum": 32, "maximum": 143}


def test_unbounded_plan_requires_second_confirmation() -> None:
    plan = {"full_table1": True}
    with pytest.raises(PermissionError, match="explicit confirmation"):
        require_full_table_confirmation(plan, False)
    require_full_table_confirmation(plan, True)


def test_parallel_execution_preserves_order_inside_each_dataset(monkeypatch) -> None:
    calls: list[str] = []

    def fake_run(command, *, check):
        assert check is True
        calls.append(command[0])

    monkeypatch.setattr("awo.table1.subprocess.run", fake_run)
    jobs = []
    for dataset in ("gsm8k", "math"):
        for index, phase in enumerate(("test", "search", "test"), start=1):
            jobs.append(
                {
                    "dataset": dataset,
                    "phase": phase,
                    "command": [f"{dataset}-{index}"],
                }
            )
    execute_plan({"jobs": jobs}, job_concurrency=2)
    for dataset in ("gsm8k", "math"):
        assert [item for item in calls if item.startswith(dataset)] == [
            f"{dataset}-1",
            f"{dataset}-2",
            f"{dataset}-3",
        ]
