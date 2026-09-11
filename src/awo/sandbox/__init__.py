"""Isolated execution support for generated benchmark code."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

from awo.sandbox.docker import (
    DEFAULT_IMAGE,
    PASS_MARKER,
    DockerSandbox,
    SandboxConfig,
    SandboxResult,
)
from awo.sandbox.remote import DEFAULT_SOCKET_PATH, RemoteSandbox


class SandboxRunner(Protocol):
    """Common interface implemented by local-Docker and remote sandboxes."""

    config: SandboxConfig

    def run(self, source: str) -> SandboxResult: ...


def create_sandbox(
    config: SandboxConfig | None = None,
    backend: str | None = None,
) -> SandboxRunner:
    """Create the configured sandbox backend (Docker by default)."""

    configured_backend = backend or os.environ.get("AWO_SANDBOX_BACKEND")
    configured_socket = os.environ.get("AWO_SANDBOX_SOCKET")
    project_socket = Path("../.sandbox-ipc/sandbox.sock")
    if configured_backend is None and (configured_socket or project_socket.is_socket()):
        selected = "remote"
    else:
        selected = (configured_backend or "docker").lower()
    if selected == "docker":
        return DockerSandbox(config)
    if selected == "remote":
        socket_path = configured_socket or (project_socket if project_socket.is_socket() else None)
        return RemoteSandbox(config, socket_path=socket_path)
    raise ValueError(f"unknown AWO sandbox backend: {selected!r}")

__all__ = [
    "DEFAULT_IMAGE",
    "DEFAULT_SOCKET_PATH",
    "PASS_MARKER",
    "DockerSandbox",
    "SandboxConfig",
    "SandboxResult",
    "SandboxRunner",
    "RemoteSandbox",
    "create_sandbox",
]
