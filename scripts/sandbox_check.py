#!/usr/bin/env python3
"""Run small, explicit security probes against the configured Docker sandbox."""

from __future__ import annotations

import json

from awo.sandbox import PASS_MARKER, DockerSandbox, SandboxConfig


def main() -> int:
    sandbox = DockerSandbox(SandboxConfig(timeout_seconds=2, max_output_bytes=65_536))
    probes = {
        "basic": f"print({PASS_MARKER!r})",
        "identity": (
            "import os\n"
            "assert (os.getuid(), os.getgid()) == (65534, 65534)\n"
            f"print({PASS_MARKER!r})"
        ),
        "network": (
            "import socket\n"
            "s = socket.socket()\n"
            "assert s.connect_ex(('1.1.1.1', 53)) != 0\n"
            f"print({PASS_MARKER!r})"
        ),
        "read_only": (
            "try:\n"
            "    open('/blocked', 'w')\n"
            "except OSError:\n"
            "    pass\n"
            "else:\n"
            "    raise AssertionError('root filesystem is writable')\n"
            f"print({PASS_MARKER!r})"
        ),
    }
    summary = {name: sandbox.run(source).status for name, source in probes.items()}
    summary["timeout"] = sandbox.run("while True: pass").status
    print(json.dumps(summary, sort_keys=True))
    expected = {"basic", "identity", "network", "read_only"}
    successful = all(summary[name] == "passed" for name in expected)
    return 0 if successful and summary["timeout"] == "timeout" else 1


if __name__ == "__main__":
    raise SystemExit(main())
