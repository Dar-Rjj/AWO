from __future__ import annotations

import json
from pathlib import Path

from awo.reporting import aggregate_runs, render_markdown


def test_aggregate_separates_search_and_test_cost(tmp_path: Path) -> None:
    test_dir = tmp_path / "gsm8k" / "test" / "manual"
    test_dir.mkdir(parents=True)
    (test_dir / "summary.json").write_text(
        json.dumps(
            {
                "spec": {"dataset": "gsm8k"},
                "by_run": [
                    {
                        "method": "io",
                        "repeat": 1,
                        "score": 0.5,
                        "sample_count": 2,
                        "failure_count": 0,
                        "call_count": 2,
                        "tokens": 20,
                        "cost": 0.02,
                        "latency_seconds": 1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    request = {
        "status": "ok",
        "metadata": {"role": "generator"},
        "result": {
            "provider": "P",
            "requested_model": "deepseek/deepseek-chat",
            "actual_model": "deepseek/deepseek-chat",
            "latency_seconds": 1,
            "usage": {
                "prompt_tokens": 8,
                "completion_tokens": 2,
                "total_tokens": 10,
                "cost": 0.01,
            },
        },
    }
    (test_dir / "requests.jsonl").write_text(json.dumps(request) + "\n", encoding="utf-8")
    search_dir = tmp_path / "gsm8k" / "search" / "aflow"
    search_dir.mkdir(parents=True)
    (search_dir / "summary.json").write_text(
        json.dumps(
            {
                "method": "aflow",
                "dataset": "gsm8k",
                "validation_sample_count": 1,
                "best_candidate": "candidate.json",
            }
        ),
        encoding="utf-8",
    )
    (search_dir / "requests.jsonl").write_text(json.dumps(request) + "\n", encoding="utf-8")

    report = aggregate_runs(tmp_path)
    assert report["cells"][0]["mean"] == 50.0
    assert report["totals"]["test_cost"] == 0.01
    assert report["totals"]["search_cost"] == 0.01
    assert "Frozen test: 1 requests" in render_markdown(report)
