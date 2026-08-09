"""Run untrusted Python in a tightly constrained, networkless Docker container."""

from __future__ import annotations

import json
import math
import os
import selectors
import subprocess
import time
import uuid
from dataclasses import dataclass
from typing import Literal

DEFAULT_IMAGE = "awo-code-sandbox:py3.9-numpy2.0.2"
EXPECTED_LABELS = {
    "org.awo.profile": "python3.9-numpy2.0.2",
    "org.awo.base.digest": (
        "sha256:2d97f6910b16bd338d3060f261f53f144965f755599aab1acda1e13cf1731b1b"
    ),
    "org.awo.numpy.sha256": (
        "f26b258c385842546006213344c50655ff1555a9338e2e5e02a0756dc3e803dd"
    ),
}
PASS_MARKER = "__AWO_TESTS_PASSED__"
SandboxStatus = Literal[
    "passed", "failed", "timeout", "output_limit", "resource_limit", "sandbox_error"
]


@dataclass(frozen=True)
class SandboxConfig:
    image: str = DEFAULT_IMAGE
    timeout_seconds: float = 10.0
    memory_mb: int = 256
    pids_limit: int = 64
    cpus: float = 1.0
    max_output_bytes: int = 1_048_576
    tmpfs_mb: int = 16

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if min(self.memory_mb, self.pids_limit, self.max_output_bytes, self.tmpfs_mb) <= 0:
            raise ValueError("sandbox resource limits must be positive")
        if self.cpus <= 0:
            raise ValueError("cpus must be positive")
        pinned = "@sha256:" in self.image or self.image.startswith("sha256:")
        if not pinned and self.image != DEFAULT_IMAGE:
            raise ValueError("sandbox image must be pinned or use the verified AWO build tag")


@dataclass(frozen=True)
class SandboxResult:
    status: SandboxStatus
    exit_code: int | None
    stdout: str
    stderr: str
    duration_seconds: float
    image: str

    @property
    def passed(self) -> bool:
        return self.status == "passed"


class DockerSandbox:
    """A fail-closed Docker runner; it never mounts host files into the container."""

    def __init__(self, config: SandboxConfig | None = None, docker_binary: str = "docker"):
        self.config = config or SandboxConfig()
        self.docker_binary = docker_binary
        self._resolved_image: str | None = None

    def build_command(self, container_name: str, image: str | None = None) -> list[str]:
        config = self.config
        memory = f"{config.memory_mb}m"
        cpu_seconds = max(1, math.ceil(config.timeout_seconds))
        return [
            self.docker_binary,
            "run",
            "--interactive",
            "--name",
            container_name,
            "--network",
            "none",
            "--read-only",
            "--user",
            "65534:65534",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            str(config.pids_limit),
            "--memory",
            memory,
            "--memory-swap",
            memory,
            "--cpus",
            str(config.cpus),
            "--ipc",
            "none",
            "--tmpfs",
            f"/tmp:rw,noexec,nosuid,nodev,size={config.tmpfs_mb}m,mode=1777",
            "--workdir",
            "/tmp",
            "--ulimit",
            f"cpu={cpu_seconds}:{cpu_seconds}",
            "--ulimit",
            "nofile=64:64",
            "--ulimit",
            "fsize=1024:1024",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            "PYTHONHASHSEED=0",
            image or config.image,
            "python",
            "-I",
            "-B",
            "-",
        ]

    def _docker_control(self, *arguments: str) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            [self.docker_binary, *arguments],
            capture_output=True,
            timeout=5,
            check=False,
        )

    def _resolve_image(self) -> str:
        if self._resolved_image is not None:
            return self._resolved_image
        image = self.config.image
        if image.startswith("sha256:") or "@sha256:" in image:
            self._resolved_image = image
            return self._resolved_image
        inspected = self._docker_control("image", "inspect", image)
        if inspected.returncode != 0:
            message = inspected.stderr.decode("utf-8", "replace").strip()
            raise RuntimeError(
                f"verified sandbox image is unavailable; build docker/code-sandbox: {message}"
            )
        records = json.loads(inspected.stdout)
        if len(records) != 1:
            raise RuntimeError(f"unexpected Docker inspect result for {image}")
        record = records[0]
        labels = record.get("Config", {}).get("Labels") or {}
        wrong = {
            key: labels.get(key)
            for key, expected in EXPECTED_LABELS.items()
            if labels.get(key) != expected
        }
        if wrong:
            raise RuntimeError(f"sandbox image labels failed verification: {wrong}")
        image_id = str(record.get("Id", ""))
        if not image_id.startswith("sha256:"):
            raise RuntimeError(f"Docker returned an invalid image ID for {image}")
        self._resolved_image = image_id
        return self._resolved_image

    def run(self, source: str) -> SandboxResult:
        name = f"awo-sandbox-{uuid.uuid4().hex}"
        started = time.monotonic()
        process: subprocess.Popen[bytes] | None = None
        stdout = bytearray()
        stderr = bytearray()
        terminal_status: SandboxStatus | None = None
        oom_killed = False
        resolved_image = self.config.image
        try:
            resolved_image = self._resolve_image()
            process = subprocess.Popen(
                self.build_command(name, resolved_image),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={"PATH": os.environ.get("PATH", "")},
            )
            assert process.stdin is not None
            assert process.stdout is not None
            assert process.stderr is not None
            process.stdin.write(source.encode("utf-8"))
            process.stdin.close()

            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ, stdout)
            selector.register(process.stderr, selectors.EVENT_READ, stderr)
            deadline = started + self.config.timeout_seconds
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    terminal_status = "timeout"
                    break
                events = selector.select(min(remaining, 0.1))
                if not events and process.poll() is not None:
                    events = [(key, selectors.EVENT_READ) for key in selector.get_map().values()]
                for key, _ in events:
                    chunk = os.read(key.fd, 65_536)
                    if chunk:
                        key.data.extend(chunk)
                        if len(stdout) + len(stderr) > self.config.max_output_bytes:
                            terminal_status = "output_limit"
                            break
                    else:
                        selector.unregister(key.fileobj)
                if terminal_status is not None:
                    break

            if terminal_status is not None and process.poll() is None:
                # Stop the local attach stream first. Otherwise an output-flooding
                # container can fill its pipe and make the daemon-side kill block.
                process.kill()
                process.wait(timeout=2)
                self._docker_control("kill", name)
            try:
                exit_code = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                exit_code = process.wait(timeout=5)

            inspect = self._docker_control("inspect", "--format", "{{.State.OOMKilled}}", name)
            oom_killed = inspect.returncode == 0 and inspect.stdout.strip() == b"true"
            if terminal_status is None:
                if oom_killed or exit_code in (137, 143):
                    terminal_status = "resource_limit"
                elif exit_code == 0 and PASS_MARKER in stdout.decode("utf-8", "replace"):
                    terminal_status = "passed"
                else:
                    terminal_status = "failed"
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            terminal_status = "sandbox_error"
            exit_code = process.poll() if process is not None else None
            stderr.extend(f"\nDocker sandbox error: {error}".encode())
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()
            try:
                self._docker_control("rm", "--force", name)
            except (OSError, subprocess.SubprocessError):
                pass

        if oom_killed:
            terminal_status = "resource_limit"
        assert terminal_status is not None
        stdout_bytes = bytes(stdout[: self.config.max_output_bytes])
        stderr_budget = self.config.max_output_bytes - len(stdout_bytes)
        stderr_bytes = bytes(stderr[:stderr_budget])
        return SandboxResult(
            status=terminal_status,
            exit_code=exit_code,
            stdout=stdout_bytes.decode("utf-8", "replace"),
            stderr=stderr_bytes.decode("utf-8", "replace"),
            duration_seconds=time.monotonic() - started,
            image=resolved_image,
        )
