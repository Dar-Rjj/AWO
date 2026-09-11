#!/usr/bin/env python3
"""Run explicit security probes against the configured sandbox backend."""

from __future__ import annotations

import json

from awo.sandbox import PASS_MARKER, SandboxConfig, create_sandbox


def main() -> int:
    sandbox = create_sandbox(SandboxConfig(timeout_seconds=2, max_output_bytes=65_536))
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
        "service_isolation": (
            "import os\n"
            "blocked = [\n"
            "    '/run/code-sandbox/sandbox.sock',\n"
            "    '/root/.env',\n"
            "    '/root/workspace/rjj/AWO',\n"
            "]\n"
            "assert not any(os.path.exists(path) for path in blocked)\n"
            f"print({PASS_MARKER!r})"
        ),
    }
    summary = {name: sandbox.run(source).status for name, source in probes.items()}
    summary["timeout"] = sandbox.run("while True: pass").status
    summary["output_flood"] = sandbox.run("while True: print('x' * 4096)").status
    print(json.dumps(summary, sort_keys=True))
    expected = {"basic", "identity", "network", "read_only", "service_isolation"}
    successful = all(summary[name] == "passed" for name in expected)
    limits_work = summary["timeout"] == "timeout" and summary["output_flood"] == "output_limit"
    return 0 if successful and limits_work else 1


if __name__ == "__main__":
    raise SystemExit(main())
