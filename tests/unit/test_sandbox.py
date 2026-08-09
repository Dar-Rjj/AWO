from __future__ import annotations

import pytest

from awo.benchmarks.code import (
    build_humaneval_harness,
    build_mbpp_harness,
    extract_python_code,
)
from awo.sandbox import DEFAULT_IMAGE, DockerSandbox, SandboxConfig


def test_image_must_be_digest_pinned() -> None:
    with pytest.raises(ValueError, match="pinned"):
        SandboxConfig(image="python:3.9-slim")


def test_docker_command_contains_required_isolation_flags() -> None:
    command = DockerSandbox().build_command("exact-name")
    joined = " ".join(command)
    for expected in (
        "--interactive",
        "--network none",
        "--read-only",
        "--user 65534:65534",
        "--cap-drop ALL",
        "--security-opt no-new-privileges",
        "--memory-swap 256m",
        "--ipc none",
        "--tmpfs /tmp:rw,noexec,nosuid,nodev",
        "python -I -B -",
    ):
        assert expected in joined
    assert DEFAULT_IMAGE in command
    assert "--volume" not in command
    assert "--privileged" not in command


def test_extracts_longest_valid_python_fence() -> None:
    response = "explanation\n```python\nx = 1\n```\n```py\ndef add(a, b):\n    return a + b\n```"
    assert extract_python_code(response) == "def add(a, b):\n    return a + b"


def test_harness_invokes_entry_point_check() -> None:
    harness = build_humaneval_harness(
        "def add(a, b): return a + b",
        "def check(candidate): assert candidate(2, 3) == 5",
        "add",
    )
    assert "check(globals()['add'])" in harness
    assert "__AWO_TESTS_PASSED__" in harness


def test_mbpp_harness_accepts_test_list() -> None:
    harness = build_mbpp_harness("def add(a, b): return a + b", ["assert add(2, 3) == 5"])
    assert "assert add(2, 3) == 5" in harness


def test_mbpp_harness_calls_named_check() -> None:
    harness = build_mbpp_harness(
        "def add(a, b): return a + b",
        "def check(): assert add(2, 3) == 5",
    )
    assert "\ncheck()\n" in harness
