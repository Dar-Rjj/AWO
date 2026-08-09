"""Hash-first downloads and traversal-safe extraction for official artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

import gdown


class ArtifactError(RuntimeError):
    """Raised when an artifact is missing, unsafe, or differs from the manifest."""


@dataclass(frozen=True)
class ArtifactSpec:
    name: str
    google_drive_id: str
    filename: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class DatasetFileSpec:
    path: str
    records: int | None
    sha256: str


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_artifact_manifest(path: Path) -> tuple[dict[str, ArtifactSpec], list[DatasetFileSpec]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != 1:
        raise ArtifactError("Only artifact manifest schema_version=1 is supported")

    archives = {
        name: ArtifactSpec(name=name, **values)
        for name, values in payload["archives"].items()
    }
    files = [DatasetFileSpec(**values) for values in payload["files"]]
    return archives, files


def verify_archive(path: Path, spec: ArtifactSpec) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ArtifactError(f"Archive does not exist: {path}")
    actual_size = path.stat().st_size
    if actual_size != spec.size_bytes:
        raise ArtifactError(
            f"Size mismatch for {path.name}: expected {spec.size_bytes}, got {actual_size}"
        )
    actual_hash = sha256_file(path)
    if actual_hash != spec.sha256:
        raise ArtifactError(
            f"SHA256 mismatch for {path.name}: expected {spec.sha256}, got {actual_hash}"
        )
    return {"path": str(path), "size_bytes": actual_size, "sha256": actual_hash}


def download_artifact(spec: ArtifactSpec, cache_dir: Path) -> Path:
    """Reuse a verified archive or atomically publish a newly verified download."""

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / spec.filename
    if destination.exists():
        verify_archive(destination, spec)
        return destination

    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    downloaded = gdown.download(id=spec.google_drive_id, output=str(temporary), quiet=False)
    if downloaded is None or not temporary.is_file():
        raise ArtifactError(f"Download failed for artifact {spec.name}")
    verify_archive(temporary, spec)
    temporary.replace(destination)
    return destination


def _safe_relative_path(name: str, strip_prefix: str | None) -> Path | None:
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts:
        raise ArtifactError(f"Unsafe archive path: {name}")
    parts = list(pure.parts)
    if strip_prefix is not None:
        if not parts or parts[0] != strip_prefix:
            raise ArtifactError(
                f"Archive member {name!r} does not start with {strip_prefix!r}"
            )
        parts = parts[1:]
    if not parts:
        return None
    return Path(*parts)


def _copy_member(source: BinaryIO, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as output:
        shutil.copyfileobj(source, output)


def _safe_extract_tar(archive: Path, destination: Path, strip_prefix: str | None) -> None:
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            if member.issym() or member.islnk() or member.isdev():
                raise ArtifactError(f"Unsupported archive member type: {member.name}")
            if not (member.isdir() or member.isfile()):
                raise ArtifactError(f"Unsupported archive member type: {member.name}")
            _safe_relative_path(member.name, strip_prefix)

        for member in members:
            relative = _safe_relative_path(member.name, strip_prefix)
            if relative is None:
                continue
            target = destination / relative
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            source = tar.extractfile(member)
            if source is None:
                raise ArtifactError(f"Could not read archive member: {member.name}")
            with source:
                _copy_member(source, target)


def extract_artifact(
    archive: Path,
    destination: Path,
    spec: ArtifactSpec,
    *,
    strip_prefix: str | None = None,
) -> Path:
    """Extract into a staging directory and atomically publish the result."""

    verify_archive(archive, spec)
    destination = Path(destination)
    marker = destination / ".artifact.json"
    if marker.is_file():
        with marker.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("sha256") == spec.sha256:
            return destination
    if destination.exists():
        raise ArtifactError(
            f"Destination already exists without the expected marker: {destination}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        _safe_extract_tar(Path(archive), staging, strip_prefix)
        with (staging / ".artifact.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "schema_version": 1,
                    "artifact": spec.name,
                    "filename": spec.filename,
                    "size_bytes": spec.size_bytes,
                    "sha256": spec.sha256,
                },
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
        staging.replace(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def verify_dataset_files(
    dataset_dir: Path, specs: list[DatasetFileSpec]
) -> list[dict[str, Any]]:
    verified = []
    for spec in specs:
        path = Path(dataset_dir) / spec.path
        if not path.is_file():
            raise ArtifactError(f"Dataset file does not exist: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != spec.sha256:
            raise ArtifactError(
                f"SHA256 mismatch for {spec.path}: expected {spec.sha256}, got {actual_hash}"
            )
        actual_records = None
        if spec.records is not None:
            with path.open("r", encoding="utf-8") as handle:
                actual_records = sum(1 for line in handle if line.strip())
            if actual_records != spec.records:
                raise ArtifactError(
                    f"Record mismatch for {spec.path}: expected {spec.records}, "
                    f"got {actual_records}"
                )
        verified.append(
            {"path": spec.path, "records": actual_records, "sha256": actual_hash}
        )
    return verified
