import json
from pathlib import Path
from types import SimpleNamespace

from awo.benchmarks.replay import (
    detect_header_row,
    replay_code_result_csv,
    replay_result_csv,
)
from awo.sandbox import PASS_MARKER, SandboxResult


class PassingSandbox:
    def __init__(self) -> None:
        self.config = SimpleNamespace(image="fake-image")
        self.sources: list[str] = []

    def run(self, source: str) -> SandboxResult:
        self.sources.append(source)
        return SandboxResult("passed", 0, PASS_MARKER, "", 0.01, "sha256:fake")


def test_replay_accepts_extra_metadata_row(tmp_path: Path) -> None:
    path = tmp_path / "result.csv"
    path.write_text(
        "0.50000_20240926_112034\n"
        "question,prediction,expected_output,score,cost\n"
        'q,"The answer",answer,1.0,0.1\n'
        'q2,wrong,right,0.0,0.2\n',
        encoding="utf-8",
    )

    assert detect_header_row(path) == 1
    replay = replay_result_csv("hotpotqa", path)
    assert replay["records"] == 2
    assert replay["mismatches"] == 0
    assert replay["stored_mean"] == replay["replayed_mean"] == 0.5


def test_replay_code_joins_prompt_and_builds_harness(tmp_path: Path) -> None:
    dataset = tmp_path / "humaneval_test.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "task_id": "HumanEval/0",
                "prompt": "\ndef add(a, b):\n",
                "canonical_solution": "    return a + b\n",
                "test": "def check(candidate): assert candidate(2, 3) == 5",
                "entry_point": "add",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = tmp_path / "result.csv"
    result.write_text(
        "question,prediction,expected_output,score,cost\n"
        '"def add(a, b):","def add(a, b): return a + b",ok,1,0\n',
        encoding="utf-8",
    )
    sandbox = PassingSandbox()

    replay = replay_code_result_csv(
        "HumanEval", result, dataset, sandbox=sandbox  # type: ignore[arg-type]
    )

    assert replay["records"] == 1
    assert replay["mismatches"] == 0
    assert replay["sandbox_statuses"] == {"passed": 1}
    assert "check(globals()['add'])" in sandbox.sources[0]
