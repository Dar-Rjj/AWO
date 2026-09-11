from __future__ import annotations

import socket
import threading
from pathlib import Path

from awo.sandbox import PASS_MARKER, RemoteSandbox, create_sandbox
from awo.sandbox.protocol import receive_message, send_message


def _serve_two_requests(socket_path: Path) -> None:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(socket_path))
        server.listen(2)
        with server.accept()[0] as connection:
            request = receive_message(connection)
            assert request["operation"] == "health"
            send_message(
                connection,
                {
                    "ok": True,
                    "profile": "python3.9-numpy2.0.2",
                    "python_version": "3.9.25",
                    "numpy_version": "2.0.2",
                    "image": "test-service",
                },
            )
        with server.accept()[0] as connection:
            request = receive_message(connection)
            assert request["operation"] == "run"
            assert PASS_MARKER in request["source"]
            send_message(
                connection,
                {
                    "ok": True,
                    "result": {
                        "status": "exited",
                        "exit_code": 0,
                        "stdout": PASS_MARKER,
                        "stderr": "",
                        "duration_seconds": 0.01,
                        "image": "test-service",
                    },
                },
            )


def test_remote_sandbox_preserves_result_contract(tmp_path: Path) -> None:
    socket_path = tmp_path / "sandbox.sock"
    server = threading.Thread(target=_serve_two_requests, args=(socket_path,), daemon=True)
    server.start()
    while not socket_path.exists():
        server.join(0.001)

    result = RemoteSandbox(socket_path=socket_path).run(f"print({PASS_MARKER!r})")

    server.join(timeout=1)
    assert not server.is_alive()
    assert result.passed
    assert result.image == "test-service"


def test_remote_sandbox_fails_closed_when_service_is_absent(tmp_path: Path) -> None:
    result = RemoteSandbox(socket_path=tmp_path / "missing.sock").run("print('unsafe')")

    assert result.status == "sandbox_error"
    assert result.exit_code is None
    assert "Remote sandbox error" in result.stderr


def test_factory_selects_remote_backend(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AWO_SANDBOX_BACKEND", "remote")
    monkeypatch.setenv("AWO_SANDBOX_SOCKET", str(tmp_path / "sandbox.sock"))

    sandbox = create_sandbox()

    assert isinstance(sandbox, RemoteSandbox)


def test_factory_auto_detects_project_socket(monkeypatch, tmp_path: Path) -> None:
    project_directory = tmp_path / "AWO"
    project_directory.mkdir()
    socket_directory = tmp_path / ".sandbox-ipc"
    socket_directory.mkdir()
    socket_path = socket_directory / "sandbox.sock"
    monkeypatch.delenv("AWO_SANDBOX_BACKEND", raising=False)
    monkeypatch.delenv("AWO_SANDBOX_SOCKET", raising=False)
    monkeypatch.chdir(project_directory)

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(socket_path))
        sandbox = create_sandbox()

    assert isinstance(sandbox, RemoteSandbox)
    assert sandbox.socket_path == Path("../.sandbox-ipc/sandbox.sock")
