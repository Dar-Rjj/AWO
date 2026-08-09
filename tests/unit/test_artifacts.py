import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from awo.artifacts import (
    ArtifactError,
    ArtifactSpec,
    DatasetFileSpec,
    extract_artifact,
    load_artifact_manifest,
    verify_archive,
    verify_dataset_files,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def make_archive(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as tar:
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))


def spec_for(path: Path) -> ArtifactSpec:
    content = path.read_bytes()
    return ArtifactSpec(
        name="fixture",
        google_drive_id="unused",
        filename=path.name,
        size_bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )


def test_repository_manifest_has_expected_archives_and_files() -> None:
    archives, files = load_artifact_manifest(
        REPO_ROOT / "data/manifests/aflow-artifacts.json"
    )

    assert set(archives) == {"datasets", "results", "initial_rounds"}
    assert len(files) == 15
    assert sum(spec.records or 0 for spec in files) == 5101


def test_verify_archive_checks_size_and_hash(tmp_path: Path) -> None:
    archive = tmp_path / "fixture.tar.gz"
    make_archive(archive, {"data.jsonl": b"{}\n"})
    spec = spec_for(archive)

    assert verify_archive(archive, spec)["sha256"] == spec.sha256
    archive.write_bytes(b"changed")
    with pytest.raises(ArtifactError, match="Size mismatch"):
        verify_archive(archive, spec)


def test_extract_is_safe_atomic_and_idempotent(tmp_path: Path) -> None:
    archive = tmp_path / "fixture.tar.gz"
    make_archive(archive, {"root/data.jsonl": b'{"id": 1}\n'})
    spec = spec_for(archive)
    destination = tmp_path / "extracted"

    first = extract_artifact(archive, destination, spec, strip_prefix="root")
    second = extract_artifact(archive, destination, spec, strip_prefix="root")

    assert first == second == destination
    assert (destination / "data.jsonl").read_text(encoding="utf-8") == '{"id": 1}\n'
    marker = json.loads((destination / ".artifact.json").read_text(encoding="utf-8"))
    assert marker["sha256"] == spec.sha256


def test_extract_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.tar.gz"
    make_archive(archive, {"../escaped": b"unsafe"})

    with pytest.raises(ArtifactError, match="Unsafe archive path"):
        extract_artifact(archive, tmp_path / "output", spec_for(archive))
    assert not (tmp_path / "escaped").exists()


def test_extract_rejects_links(tmp_path: Path) -> None:
    archive = tmp_path / "link.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tar.addfile(info)

    with pytest.raises(ArtifactError, match="Unsupported archive member type"):
        extract_artifact(archive, tmp_path / "output", spec_for(archive))


def test_verify_dataset_files_checks_records(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text('{"id": 1}\n{"id": 2}\n', encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    specs = [DatasetFileSpec(path=path.name, records=2, sha256=digest)]

    assert verify_dataset_files(tmp_path, specs)[0]["records"] == 2

    wrong = [DatasetFileSpec(path=path.name, records=3, sha256=digest)]
    with pytest.raises(ArtifactError, match="Record mismatch"):
        verify_dataset_files(tmp_path, wrong)
