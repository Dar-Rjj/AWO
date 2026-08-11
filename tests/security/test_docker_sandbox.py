from __future__ import annotations

import asyncio
import os

import pytest

from awo.aflow import AFlowRuntime
from awo.aflow import Test as PublicTestOperator
from awo.sandbox import PASS_MARKER, DockerSandbox, SandboxConfig

pytestmark = pytest.mark.skipif(
    os.environ.get("AWO_RUN_DOCKER_TESTS") != "1",
    reason="set AWO_RUN_DOCKER_TESTS=1 to run Docker security tests",
)


@pytest.fixture
def sandbox() -> DockerSandbox:
    return DockerSandbox(SandboxConfig(timeout_seconds=1.5, max_output_bytes=32_768))


def test_passes_normal_program(sandbox: DockerSandbox) -> None:
    assert sandbox.run(f"print({PASS_MARKER!r})").status == "passed"


def test_is_non_root_and_root_filesystem_is_read_only(sandbox: DockerSandbox) -> None:
    source = f"""
import os
assert os.getuid() == 65534 and os.getgid() == 65534
try:
    open('/should-not-exist', 'w')
except OSError:
    pass
else:
    raise AssertionError('root filesystem writable')
print({PASS_MARKER!r})
"""
    assert sandbox.run(source).status == "passed"


def test_network_is_unreachable(sandbox: DockerSandbox) -> None:
    source = f"""
import socket
s = socket.socket()
assert s.connect_ex(('1.1.1.1', 53)) != 0
print({PASS_MARKER!r})
"""
    assert sandbox.run(source).status == "passed"


def test_infinite_loop_times_out(sandbox: DockerSandbox) -> None:
    assert sandbox.run("while True: pass").status == "timeout"


def test_output_flood_is_stopped(sandbox: DockerSandbox) -> None:
    result = sandbox.run("while True: print('x' * 1000)")
    assert result.status == "output_limit"
    total_output = len(result.stdout.encode()) + len(result.stderr.encode())
    assert total_output <= sandbox.config.max_output_bytes


def test_aflow_public_test_alias_runs_only_in_sandbox(sandbox: DockerSandbox) -> None:
    runtime = AFlowRuntime(object(), sandbox=sandbox)  # type: ignore[arg-type]
    result = asyncio.run(
        PublicTestOperator(runtime)(
            "increment",
            "def increment(value):\n    return value + 1",
            "increment",
            "assert candidate(1) == 2",
            test_loop=1,
        )
    )
    assert result["result"] is True
    assert runtime.responses == []
