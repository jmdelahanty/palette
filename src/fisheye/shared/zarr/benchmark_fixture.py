"""Guarded publication of immutable, noncanonical benchmark source fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping
import uuid


FIXTURE_MANIFEST_SCHEMA_ID = "palette.storage_benchmark_fixture"
FIXTURE_MANIFEST_SCHEMA_VERSION = 1
TREE_DIGEST_SCHEMA_ID = "palette.tree_sha256.path_size_content.v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class TreeInventory:
    """Deterministic exact identity and storage accounting for one file tree."""

    file_count: int
    apparent_bytes: int
    allocated_bytes: int
    tree_sha256: str

    def as_manifest(self) -> dict[str, object]:
        return {
            "digest_schema_id": TREE_DIGEST_SCHEMA_ID,
            "file_count": self.file_count,
            "apparent_bytes": self.apparent_bytes,
            "allocated_bytes": self.allocated_bytes,
            "tree_sha256": self.tree_sha256,
        }


def inventory_tree(root: Path) -> TreeInventory:
    """Hash relative paths, file sizes, and contents in deterministic order."""

    resolved = root.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Tree does not exist: {resolved}")
    digest = hashlib.sha256()
    file_count = 0
    apparent_bytes = 0
    allocated_bytes = 0
    for path in sorted(candidate for candidate in resolved.rglob("*") if candidate.is_file()):
        if path.is_symlink():
            raise ValueError(f"Benchmark fixture trees cannot contain symlinks: {path}")
        relative = path.relative_to(resolved).as_posix().encode("utf-8")
        stat = path.stat()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(int(stat.st_size).to_bytes(8, "little"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        file_count += 1
        apparent_bytes += int(stat.st_size)
        allocated_bytes += int(stat.st_blocks * 512)
    return TreeInventory(
        file_count=file_count,
        apparent_bytes=apparent_bytes,
        allocated_bytes=allocated_bytes,
        tree_sha256=digest.hexdigest(),
    )


def load_noncanonical_source_manifest(
    manifest_path: Path,
    *,
    expected_source: Path,
) -> dict[str, object]:
    """Require explicit proof that a source is already a disposable copy."""

    resolved_manifest = manifest_path.expanduser().resolve()
    payload = json.loads(resolved_manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Source manifest must be a JSON object.")
    required_false = ("canonical", "registry_registered", "selector_eligible")
    for field in required_false:
        if payload.get(field) is not False:
            raise ValueError(f"Source manifest must declare {field}=false.")
    purpose = str(payload.get("purpose", "")).lower()
    if "benchmark" not in purpose:
        raise ValueError("Source manifest purpose must identify benchmark use.")
    declared_destination = payload.get("destination")
    if not isinstance(declared_destination, str) or (
        Path(declared_destination).expanduser().resolve()
        != expected_source.expanduser().resolve()
    ):
        raise ValueError("Source manifest destination does not match the source tree.")
    return payload


def require_safe_fixture_destination(
    destination: Path,
    *,
    benchmark_root: Path,
) -> Path:
    """Require a fresh destination below a benchmark-only fixtures namespace."""

    root = benchmark_root.expanduser().resolve()
    resolved = destination.expanduser().resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError(f"Fixture destination must be below {root}.")
    relative = resolved.relative_to(root)
    if "fixtures" not in relative.parts:
        raise ValueError("Fixture destination must be below a fixtures namespace.")
    if resolved.name in {"", ".", "..", "fixtures"}:
        raise ValueError("Fixture destination must identify one concrete fixture.")
    if resolved.exists():
        raise FileExistsError(f"Fixture destination already exists: {resolved}")
    return resolved


def plan_benchmark_fixture(
    *,
    fixture_id: str,
    source: Path,
    source_manifest_path: Path,
    destination: Path,
    benchmark_root: Path,
) -> dict[str, object]:
    """Validate and inventory a proposed copy without changing destination state."""

    resolved_source = source.expanduser().resolve()
    if not resolved_source.is_dir():
        raise FileNotFoundError(f"Benchmark fixture source does not exist: {resolved_source}")
    metadata_path = resolved_source / "zarr.json"
    if not metadata_path.is_file():
        raise ValueError("Benchmark fixture source must be a Zarr v3 group.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("zarr_format") != 3 or metadata.get("node_type") != "group":
        raise ValueError("Benchmark fixture source must be a Zarr v3 group.")
    source_manifest = load_noncanonical_source_manifest(
        source_manifest_path,
        expected_source=resolved_source,
    )
    resolved_destination = require_safe_fixture_destination(
        destination,
        benchmark_root=benchmark_root,
    )
    identifier = str(fixture_id).strip()
    if not identifier or identifier != resolved_destination.name:
        raise ValueError("fixture_id must equal the destination directory name.")
    if any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789_-"
        for character in identifier
    ):
        raise ValueError(
            "fixture_id must contain only lowercase letters, digits, '_' or '-'."
        )
    return {
        "schema_id": FIXTURE_MANIFEST_SCHEMA_ID,
        "schema_version": FIXTURE_MANIFEST_SCHEMA_VERSION,
        "status": "planned",
        "fixture_id": identifier,
        "benchmark_only": True,
        "canonical": False,
        "registry_registered": False,
        "selector_eligible": False,
        "source": str(resolved_source),
        "source_manifest": str(source_manifest_path.expanduser().resolve()),
        "source_manifest_sha256": _sha256_file(source_manifest_path.expanduser().resolve()),
        "source_manifest_payload": source_manifest,
        "source_zarr_format": 3,
        "source_inventory": inventory_tree(resolved_source).as_manifest(),
        "destination": str(resolved_destination),
        "copied_zarr_relative_path": "source.zarr",
        "copy_method": "shutil.copytree_exclusive_then_atomic_rename",
        "destination_collision": False,
        "payload_io_performed": False,
    }


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    rendered = json.dumps(
        payload,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    with path.open("x", encoding="utf-8") as handle:
        handle.write(rendered + "\n")


def _freeze_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_symlink():
            raise ValueError(f"Cannot freeze a symlinked fixture path: {path}")
        os.chmod(path, 0o555 if path.is_dir() else 0o444)
    os.chmod(root, 0o555)


def _thaw_tree_for_cleanup(root: Path) -> None:
    """Restore owner write bits only for an unpublished temporary tree."""

    os.chmod(root, 0o755)
    for path in root.rglob("*"):
        if path.is_dir():
            os.chmod(path, 0o755)
        elif path.is_file():
            os.chmod(path, 0o644)


def publish_benchmark_fixture(
    *,
    fixture_id: str,
    source: Path,
    source_manifest_path: Path,
    destination: Path,
    benchmark_root: Path,
) -> dict[str, object]:
    """Copy, verify, manifest, atomically publish, and freeze one fixture."""

    plan = plan_benchmark_fixture(
        fixture_id=fixture_id,
        source=source,
        source_manifest_path=source_manifest_path,
        destination=destination,
        benchmark_root=benchmark_root,
    )
    resolved_source = Path(str(plan["source"]))
    resolved_destination = Path(str(plan["destination"]))
    resolved_destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved_destination.parent / (
        f".{resolved_destination.name}.incomplete.{uuid.uuid4().hex}"
    )
    if temporary.exists():
        raise FileExistsError(f"Temporary fixture path already exists: {temporary}")

    try:
        copied_tree = temporary / "source.zarr"
        shutil.copytree(resolved_source, copied_tree)
        copied_inventory = inventory_tree(copied_tree)
        source_inventory = TreeInventory(
            file_count=int(plan["source_inventory"]["file_count"]),
            apparent_bytes=int(plan["source_inventory"]["apparent_bytes"]),
            allocated_bytes=int(plan["source_inventory"]["allocated_bytes"]),
            tree_sha256=str(plan["source_inventory"]["tree_sha256"]),
        )
        if (
            copied_inventory.file_count != source_inventory.file_count
            or copied_inventory.apparent_bytes != source_inventory.apparent_bytes
            or copied_inventory.tree_sha256 != source_inventory.tree_sha256
        ):
            raise RuntimeError("Copied benchmark fixture tree does not match source.")
        manifest = {
            **plan,
            "status": "published_immutable",
            "created_at_utc": _utc_now(),
            "copied_inventory": copied_inventory.as_manifest(),
            "exact_relative_path_size_content_match": True,
            "payload_io_performed": True,
            "immutability": {
                "files_mode": "0444",
                "directories_mode": "0555",
                "jobs_must_open_read_only": True,
            },
        }
        _write_json_exclusive(temporary / "fixture_manifest.json", manifest)
        _freeze_tree(temporary)
        temporary.rename(resolved_destination)
        return manifest
    except BaseException:
        if temporary.exists():
            _thaw_tree_for_cleanup(temporary)
            shutil.rmtree(temporary)
        raise


__all__ = [
    "FIXTURE_MANIFEST_SCHEMA_ID",
    "FIXTURE_MANIFEST_SCHEMA_VERSION",
    "TREE_DIGEST_SCHEMA_ID",
    "TreeInventory",
    "inventory_tree",
    "load_noncanonical_source_manifest",
    "plan_benchmark_fixture",
    "publish_benchmark_fixture",
    "require_safe_fixture_destination",
]
