"""Build immutable, secret-free experiment manifests."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from awo.config import config_fingerprint, load_config, public_config


def _git(repo_root: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(config_path: Path, repo_root: Path | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    fingerprint = config_fingerprint(config)
    now = datetime.now(timezone.utc)
    root = Path(repo_root or Path.cwd()).resolve()
    status = _git(root, "status", "--porcelain")
    model = config.get("model", {})

    return {
        "schema_version": 1,
        "run_id": f"{now.strftime('%Y%m%dT%H%M%SZ')}-{fingerprint[:12]}",
        "created_at": now.isoformat(),
        "config_sha256": fingerprint,
        "config": public_config(config),
        "repository": {
            "root": str(root),
            "commit": _git(root, "rev-parse", "HEAD"),
            "branch": _git(root, "branch", "--show-current"),
            "dirty": bool(status),
        },
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "model": {
            "provider": model.get("provider"),
            "requested_model": model.get("model"),
            "base_url": model.get("base_url"),
            "temperature": model.get("temperature"),
        },
        "credentials": {
            "openrouter_api_key_present": bool(os.getenv("OPENROUTER_API_KEY")),
        },
    }


def write_manifest(manifest: dict[str, Any], output_path: Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return output
