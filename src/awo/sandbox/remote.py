"""Client for the networkless AWO sandbox execution service."""

from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Any

from awo.sandbox.docker import PASS_MARKER, SandboxConfig, SandboxResult
from awo.sandbox.protocol import ProtocolError, receive_message, send_message

DEFAULT_SOCKET_PATH = Path("/run/code-sandbox/sandbox.sock")
EXPECTED_PROFILE = "python3.9-numpy2.0.2"


class RemoteSandbox:
    """Fail-closed sandbox client using a local Unix domain socket."""

    def __init__(
        self,
        config: SandboxConfig | None = None,
        socket_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self.config = config or SandboxConfig()
        configured_path = socket_path or os.environ.get("AWO_SANDBOX_SOCKET")
        self.socket_path = Path(configured_path) if configured_path else DEFAULT_SOCKET_PATH
        self._service_image: str | None = None

    def _request(self, message: dict[str, Any], timeout: float) -> dict[str, Any]:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(timeout)
            connection.connect(str(self.socket_path))
            send_message(connection, message)
            response = receive_message(connection)
        if response.get("ok") is not True:
            error = response.get("error", "sandbox service rejected the request")
            raise RuntimeError(str(error))
        return response

    def health(self) -> dict[str, Any]:
        """Verify that the service exposes the locked AWO runtime profile."""

        response = self._request({"operation": "health"}, timeout=5.0)
        if response.get("profile") != EXPECTED_PROFILE:
            raise RuntimeError(
                "sandbox service profile mismatch: "
                f"expected {EXPECTED_PROFILE!r}, got {response.get('profile')!r}"
            )
        if response.get("numpy_version") != "2.0.2":
            raise RuntimeError("sandbox service NumPy version mismatch")
        python_version = str(response.get("python_version", ""))
        if not python_version.startswith("3.9."):
            raise RuntimeError("sandbox service Python version mismatch")
        self._service_image = str(response.get("image", EXPECTED_PROFILE))
        return response

    def run(self, source: str) -> SandboxResult:
        """Execute source remotely and preserve the existing SandboxResult contract."""

        started_image = self._service_image or EXPECTED_PROFILE
        try:
            if self._service_image is None:
                self.health()
            response = self._request(
                {
                    "operation": "run",
                    "source": source,
                    "timeout_seconds": self.config.timeout_seconds,
                    "max_output_bytes": self.config.max_output_bytes,
                    "memory_mb": self.config.memory_mb,
                    "pids_limit": self.config.pids_limit,
                },
                timeout=self.config.timeout_seconds + 8.0,
            )
            result = response.get("result")
            if not isinstance(result, dict):
                raise ProtocolError("sandbox response is missing a result object")
            remote_status = result.get("status")
            allowed_statuses = {
                "exited",
                "timeout",
                "output_limit",
                "resource_limit",
                "sandbox_error",
            }
            if remote_status not in allowed_statuses:
                raise ProtocolError(f"invalid sandbox status: {remote_status!r}")
            exit_code = result.get("exit_code")
            if exit_code is not None and not isinstance(exit_code, int):
                raise ProtocolError("sandbox exit_code must be an integer or null")
            stdout = str(result.get("stdout", ""))
            status = remote_status
            if remote_status == "exited":
                status = "passed" if exit_code == 0 and PASS_MARKER in stdout else "failed"
            return SandboxResult(
                status=status,
                exit_code=exit_code,
                stdout=stdout,
                stderr=str(result.get("stderr", "")),
                duration_seconds=float(result.get("duration_seconds", 0.0)),
                image=str(result.get("image", self._service_image or started_image)),
            )
        except (OSError, ProtocolError, RuntimeError, TypeError, ValueError) as error:
            return SandboxResult(
                status="sandbox_error",
                exit_code=None,
                stdout="",
                stderr=f"Remote sandbox error: {error}",
                duration_seconds=0.0,
                image=self._service_image or started_image,
            )
