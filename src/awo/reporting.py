"""Aggregate resumable Table 1 ledgers and OpenRouter audit logs."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

PAPER_SCORES = {
    "io": [68.1, 68.3, 87.0, 71.8, 92.7, 48.6],
    "cot": [67.9, 78.5, 88.6, 71.8, 92.4, 48.8],
    "cot_sc": [68.9, 78.8, 91.6, 73.6, 92.7, 50.4],
    "medprompt": [68.3, 78.0, 91.6, 73.6, 90.0, 50.0],
    "multi_persona": [69.2, 74.4, 89.3, 73.6, 92.8, 50.8],
    "self_refine": [60.8, 70.2, 87.8, 69.8, 89.6, 46.1],
    "adas": [64.5, 76.6, 82.4, 53.4, 90.8, 35.4],
    "aflow": [73.5, 80.6, 94.7, 83.4, 93.5, 56.2],
}
DATASETS = ("hotpotqa", "drop", "humaneval", "mbpp", "gsm8k", "math")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _empty_request_stats() -> dict[str, Any]:
    return {
        "requests": 0,
        "tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost": 0.0,
        "latency_seconds": 0.0,
        "statuses": Counter(),
        "providers": Counter(),
        "requested_models": Counter(),
        "actual_models": Counter(),
        "roles": Counter(),
    }


def _request_stats(path: Path) -> dict[str, Any]:
    stats = _empty_request_stats()
    if not path.is_file():
        return stats
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            result = row.get("result") or {}
            usage = result.get("usage") or {}
            metadata = row.get("metadata") or {}
            stats["requests"] += 1
            stats["tokens"] += int(usage.get("total_tokens") or 0)
            stats["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
            stats["completion_tokens"] += int(usage.get("completion_tokens") or 0)
            stats["cost"] += float(usage.get("cost") or 0.0)
            stats["latency_seconds"] += float(result.get("latency_seconds") or 0.0)
            stats["statuses"][str(row.get("status") or "unknown")] += 1
            stats["providers"][str(result.get("provider") or "unknown")] += 1
            stats["requested_models"][str(result.get("requested_model") or "unknown")] += 1
            stats["actual_models"][str(result.get("actual_model") or "unknown")] += 1
            stats["roles"][str(metadata.get("role") or "unknown")] += 1
    return stats


def _plain(value: Any) -> Any:
    if isinstance(value, Counter):
        return dict(sorted(value.items()))
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return value


def _merge_request_stats(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key in (
        "requests",
        "tokens",
        "prompt_tokens",
        "completion_tokens",
        "cost",
        "latency_seconds",
    ):
        target[key] += source[key]
    for key in (
        "statuses",
        "providers",
        "requested_models",
        "actual_models",
        "roles",
    ):
        target[key].update(source[key])


def aggregate_runs(runs: Path) -> dict[str, Any]:
    """Aggregate all final-test summaries and keep search costs separate."""

    rows: list[dict[str, Any]] = []
    totals = defaultdict(float)
    search_runs: list[dict[str, Any]] = []
    audit_totals = {
        "search": _empty_request_stats(),
        "test": _empty_request_stats(),
    }
    seen: set[Path] = set()
    for summary_path in sorted(runs.rglob("summary.json")):
        summary = _read_json(summary_path)
        if "by_run" in summary and "spec" in summary:
            run_dir = summary_path.parent
            if run_dir in seen:
                continue
            seen.add(run_dir)
            spec = summary["spec"]
            dataset = str(spec["dataset"])
            audit = _request_stats(run_dir / "requests.jsonl")
            _merge_request_stats(audit_totals["test"], audit)
            for item in summary["by_run"]:
                method = str(item["method"])
                paper = None
                if method in PAPER_SCORES and dataset in DATASETS:
                    paper = PAPER_SCORES[method][DATASETS.index(dataset)]
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "repeat": int(item["repeat"]),
                        "score": float(item["score"]) * 100.0,
                        "paper_score": paper,
                        "absolute_difference": (
                            abs(float(item["score"]) * 100.0 - paper)
                            if paper is not None
                            else None
                        ),
                        "sample_count": int(item["sample_count"]),
                        "failures": int(item["failure_count"]),
                        "calls": int(item["call_count"]),
                        "tokens": int(item["tokens"]),
                        "cost": float(item["cost"]),
                        "latency_seconds": float(item["latency_seconds"]),
                        "run_dir": str(run_dir),
                    }
                )
            for key in (
                "requests",
                "tokens",
                "prompt_tokens",
                "completion_tokens",
                "cost",
                "latency_seconds",
            ):
                totals[f"test_{key}"] += audit[key]
        elif summary.get("method") in {"aflow", "adas"} and "best_candidate" in summary:
            run_dir = summary_path.parent
            audit = _request_stats(run_dir / "requests.jsonl")
            _merge_request_stats(audit_totals["search"], audit)
            search_runs.append(
                {
                    "dataset": summary["dataset"],
                    "method": summary["method"],
                    "validation_sample_count": summary["validation_sample_count"],
                    "run_dir": str(run_dir),
                    "audit": _plain(audit),
                }
            )
            for key in (
                "requests",
                "tokens",
                "prompt_tokens",
                "completion_tokens",
                "cost",
                "latency_seconds",
            ):
                totals[f"search_{key}"] += audit[key]

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["method"])].append(row)
    cells: list[dict[str, Any]] = []
    for (dataset, method), selected in sorted(grouped.items()):
        scores = [row["score"] for row in selected]
        deviation = stdev(scores) if len(scores) > 1 else 0.0
        ci95 = 1.96 * deviation / math.sqrt(len(scores)) if len(scores) > 1 else 0.0
        cells.append(
            {
                "dataset": dataset,
                "method": method,
                "scores": scores,
                "mean": fmean(scores),
                "standard_deviation": deviation,
                "ci95_half_width": ci95,
                "paper_score": selected[0]["paper_score"],
                "absolute_difference": (
                    abs(fmean(scores) - selected[0]["paper_score"])
                    if selected[0]["paper_score"] is not None
                    else None
                ),
                "samples": sum(row["sample_count"] for row in selected),
                "failures": sum(row["failures"] for row in selected),
                "calls": sum(row["calls"] for row in selected),
                "tokens": sum(row["tokens"] for row in selected),
                "cost": sum(row["cost"] for row in selected),
            }
        )
    return {
        "schema_version": 1,
        "runs_root": str(runs.resolve()),
        "cells": cells,
        "test_runs": rows,
        "search_runs": search_runs,
        "totals": dict(sorted(totals.items())),
        "audit": _plain(audit_totals),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Table 1 reproduction report",
        "",
        "| Dataset | Method | Repeats | Mean | SD | 95% CI | Paper | Δ | Samples | "
        "Failures | Calls | Cost (USD) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cell in report["cells"]:
        paper = "—" if cell["paper_score"] is None else f'{cell["paper_score"]:.2f}'
        delta = "—" if cell["absolute_difference"] is None else f'{cell["absolute_difference"]:.2f}'
        lines.append(
            f'| {cell["dataset"]} | {cell["method"]} | {len(cell["scores"])} | '
            f'{cell["mean"]:.2f} | {cell["standard_deviation"]:.2f} | '
            f'±{cell["ci95_half_width"]:.2f} | {paper} | {delta} | '
            f'{cell["samples"]} | {cell["failures"]} | {cell["calls"]} | '
            f'${cell["cost"]:.6f} |'
        )
    totals = report["totals"]
    lines.extend(
        [
            "",
            "## Cost separation",
            "",
            f'- Search: {int(totals.get("search_requests", 0))} requests, '
            f'${totals.get("search_cost", 0.0):.6f}.',
            f'- Frozen test: {int(totals.get("test_requests", 0))} requests, '
            f'${totals.get("test_cost", 0.0):.6f}.',
            "",
        ]
    )
    return "\n".join(lines)


def write_report(report: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "report.md").write_text(render_markdown(report), encoding="utf-8")
