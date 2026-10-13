"""Versioned reading guides beside validated-behavior exports.

These companion files are outside the immutable Parquet generation. Their
selection record binds one guide to the exact export manifest, but is not a
scientific publication or production authority.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

from .publication import safe_component, sha256_file
from .validated_behavior_dataset import ValidatedBehaviorExportDataset

HANDOFF_SCHEMA_ID = "palette.analytics.validated_behavior_handoff"
HANDOFF_SCHEMA_VERSION = 1
_HANDOFF_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "export_run_id",
        "export_manifest_record_sha256",
        "version",
        "document_path",
        "document_sha256",
        "record_sha256",
    }
)


@dataclass(frozen=True)
class ValidatedBehaviorHandoff:
    version: str
    path: Path
    document_sha256: str
    record_path: Path
    record_sha256: str


def _version(value: str) -> str:
    if not re.fullmatch(r"v[0-9]{3}", value):
        raise ValueError("handoff version must be v followed by three digits")
    return value


def _handoff_dir(root: Path, run_id: str) -> Path:
    run = safe_component(run_id, label="export run ID")
    return root / "handoffs" / f"export_run_id={run}"


def _checked_path(root: Path, path: Path, *, label: str) -> Path:
    root = root.resolve()
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes the publication root") from exc
    current = path
    while current != root:
        if current.is_symlink():
            raise ValueError(f"{label} must not use a symlink: {current}")
        current = current.parent
    return path


def _record_payload(
    dataset: ValidatedBehaviorExportDataset, *, version: str, digest: str
) -> dict[str, Any]:
    relative = (
        Path("handoffs")
        / f"export_run_id={dataset.export_run_id}"
        / "versions"
        / f"{version}.md"
    )
    value: dict[str, Any] = {
        "schema_id": HANDOFF_SCHEMA_ID,
        "schema_version": HANDOFF_SCHEMA_VERSION,
        "export_run_id": dataset.export_run_id,
        "export_manifest_record_sha256": dataset.manifest["record_sha256"],
        "version": version,
        "document_path": relative.as_posix(),
        "document_sha256": digest,
    }
    return {**value, "record_sha256": canonical_json_sha256(value)}


def _read_record(
    dataset: ValidatedBehaviorExportDataset, record_path: Path
) -> ValidatedBehaviorHandoff:
    root = dataset.root
    _checked_path(root, record_path, label="handoff record")
    if record_path.is_symlink() or not record_path.is_file():
        raise ValueError("handoff record is not a regular file")
    value = json.loads(record_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != _HANDOFF_FIELDS:
        raise ValueError("handoff record fields are inexact")
    version = _version(str(value["version"]))
    expected = _record_payload(
        dataset, version=version, digest=str(value["document_sha256"])
    )
    if value != expected:
        raise ValueError("handoff record does not bind this exact export and guide")
    guide = _checked_path(root, root / expected["document_path"], label="handoff guide")
    if guide.is_symlink() or not guide.is_file():
        raise ValueError("handoff guide is not a regular file")
    if sha256_file(guide) != expected["document_sha256"]:
        raise ValueError("handoff guide bytes differ from the selected digest")
    text = guide.read_text(encoding="utf-8")
    if (
        dataset.export_run_id not in text
        or expected["export_manifest_record_sha256"] not in text
    ):
        raise ValueError("handoff guide lacks the exact export run or manifest digest")
    return ValidatedBehaviorHandoff(
        version=version,
        path=guide,
        document_sha256=expected["document_sha256"],
        record_path=record_path,
        record_sha256=expected["record_sha256"],
    )


def read_validated_behavior_handoff(
    dataset: ValidatedBehaviorExportDataset,
) -> ValidatedBehaviorHandoff | None:
    """Resolve only the explicit companion record, never a sorted guide filename."""

    record_path = _handoff_dir(dataset.root, dataset.export_run_id) / "handoff.json"
    if not record_path.exists() and not record_path.is_symlink():
        return None
    return _read_record(dataset, record_path)


def publish_validated_behavior_handoff(
    *,
    publication_root: str | Path,
    export_run_id: str,
    source: str | Path,
    version: str,
) -> ValidatedBehaviorHandoff:
    """Copy a reviewed guide beside an exact export and select that version."""

    dataset = ValidatedBehaviorExportDataset.open(
        publication_root, export_run_id, validate=True, full_part_hashes=False
    )
    version = _version(version)
    source_path = Path(source).expanduser()
    if source_path.is_symlink() or not source_path.is_file():
        raise ValueError("handoff source must be a regular file")
    content = source_path.read_bytes()
    text = content.decode("utf-8")
    manifest_digest = str(dataset.manifest["record_sha256"])
    if dataset.export_run_id not in text or manifest_digest not in text:
        raise ValueError(
            "handoff source must name the exact export run and manifest digest"
        )
    directory = _checked_path(
        dataset.root,
        _handoff_dir(dataset.root, dataset.export_run_id),
        label="handoff directory",
    )
    versions_dir = _checked_path(
        dataset.root, directory / "versions", label="handoff versions"
    )
    versions_dir.mkdir(parents=True, exist_ok=True)
    guide_path = _checked_path(
        dataset.root, versions_dir / f"{version}.md", label="handoff guide"
    )
    record_path = _checked_path(
        dataset.root, directory / "handoff.json", label="handoff record"
    )
    lock_path = _checked_path(
        dataset.root, directory / ".handoff.lock", label="handoff lock"
    )
    digest = hashlib.sha256(content).hexdigest()
    expected = _record_payload(dataset, version=version, digest=digest)

    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        current = read_validated_behavior_handoff(dataset)
        if current is not None:
            if current.version == version and current.document_sha256 == digest:
                return current
            if version <= current.version:
                raise ValueError("handoff revision must use a newer version")
        if guide_path.exists() or guide_path.is_symlink():
            if guide_path.is_symlink() or sha256_file(guide_path) != digest:
                raise FileExistsError(
                    "handoff version already exists with different bytes"
                )
        else:
            temporary = versions_dir / f".{version}.{uuid.uuid4().hex}.tmp"
            try:
                with temporary.open("xb") as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.chmod(temporary, 0o644)
                os.link(temporary, guide_path)
            finally:
                temporary.unlink(missing_ok=True)
        temporary_record = directory / f".handoff.{uuid.uuid4().hex}.json.tmp"
        try:
            with temporary_record.open("x", encoding="utf-8") as handle:
                json.dump(expected, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary_record, 0o644)
            os.replace(temporary_record, record_path)
        finally:
            temporary_record.unlink(missing_ok=True)
        return _read_record(dataset, record_path)
