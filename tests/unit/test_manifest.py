import json
from pathlib import Path

from awo.tracking.manifest import build_manifest, sha256_file, write_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_sha256_file(tmp_path: Path) -> None:
    path = tmp_path / "payload.txt"
    path.write_text("awo\n", encoding="utf-8")

    assert sha256_file(path) == "980ff4fe948d40983fefc01743477d8d3500dbdd21a8ad481c9cd3800d54ebcc"


def test_manifest_is_secret_free(tmp_path: Path, monkeypatch) -> None:
    secret = "not-written-to-disk"
    monkeypatch.setenv("OPENROUTER_API_KEY", secret)

    manifest = build_manifest(REPO_ROOT / "configs/smoke.yaml", REPO_ROOT)
    output = write_manifest(manifest, tmp_path / "manifest.json")
    persisted = output.read_text(encoding="utf-8")
    parsed = json.loads(persisted)

    assert secret not in persisted
    assert parsed["credentials"]["openrouter_api_key_present"] is True
    assert parsed["model"]["requested_model"] == "deepseek/deepseek-chat"
    assert len(parsed["config_sha256"]) == 64
