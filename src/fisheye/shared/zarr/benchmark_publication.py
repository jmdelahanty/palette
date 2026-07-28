"""Exclusive copy-back publication for immutable benchmark candidates."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import time
import uuid

import zarr

from fisheye.shared.zarr.benchmark_fixture import (
    freeze_tree,
    inventory_tree,
    thaw_tree_for_cleanup,
)


def _require_publication_destination(
    destination: Path,
    *,
    workflow_root: Path,
) -> Path:
    root = workflow_root.expanduser().resolve()
    resolved = destination.expanduser().resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError(f"Published candidate must be below {root}.")
    if "candidates" not in resolved.relative_to(root).parts:
        raise ValueError("Published candidate must be below a candidates namespace.")
    if resolved.suffix != ".zarr":
        raise ValueError("Published candidate destination must end in .zarr.")
    if resolved.exists():
        raise FileExistsError(f"Published candidate already exists: {resolved}")
    return resolved


def publish_benchmark_candidate(
    *,
    source: Path,
    destination: Path,
    workflow_root: Path,
) -> dict[str, object]:
    """Copy one validated local candidate to fresh shared benchmark storage."""

    source_path = source.expanduser().resolve()
    if not (source_path / "zarr.json").is_file():
        raise ValueError(f"Candidate source is not a Zarr v3 group: {source_path}")
    source_metadata = json.loads(
        (source_path / "zarr.json").read_text(encoding="utf-8")
    )
    attributes = source_metadata.get("attributes")
    if not isinstance(attributes, dict) or (
        attributes.get("benchmark_only") is not True
        or attributes.get("canonical") is not False
        or attributes.get("registry_registered") is not False
        or attributes.get("selector_eligible") is not False
    ):
        raise ValueError("Candidate source lacks benchmark-only safety attributes.")
    destination_path = _require_publication_destination(
        destination,
        workflow_root=workflow_root,
    )
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_path.parent / (
        f".{destination_path.name}.incomplete.{uuid.uuid4().hex}"
    )
    source_inventory = inventory_tree(source_path)
    started = time.perf_counter()
    try:
        copy_started = time.perf_counter()
        shutil.copytree(source_path, temporary)
        copy_seconds = float(time.perf_counter() - copy_started)
        validation_started = time.perf_counter()
        copied_inventory = inventory_tree(temporary)
        exact = (
            copied_inventory.file_count == source_inventory.file_count
            and copied_inventory.apparent_bytes == source_inventory.apparent_bytes
            and copied_inventory.tree_sha256 == source_inventory.tree_sha256
        )
        if not exact:
            raise RuntimeError("Published benchmark candidate copy is not exact.")
        zarr.open_group(str(temporary), mode="r", use_consolidated=True)
        validation_seconds = float(time.perf_counter() - validation_started)
        freeze_tree(temporary)
        temporary.rename(destination_path)
        published_open_started = time.perf_counter()
        zarr.open_group(str(destination_path), mode="r", use_consolidated=True)
        published_open_seconds = float(time.perf_counter() - published_open_started)
        return {
            "schema_id": "palette.storage_benchmark_publication",
            "schema_version": 1,
            "status": "published_immutable",
            "source": str(source_path),
            "destination": str(destination_path),
            "copy_method": "shutil.copytree_exclusive_then_atomic_rename",
            "source_inventory": source_inventory.as_manifest(),
            "published_inventory": copied_inventory.as_manifest(),
            "exact_relative_path_size_content_match": True,
            "timing": {
                "copy_seconds": copy_seconds,
                "validation_seconds": validation_seconds,
                "published_consolidated_open_seconds": published_open_seconds,
                "publication_seconds": float(time.perf_counter() - started),
            },
            "immutability": {
                "files_mode": "0444",
                "directories_mode": "0555",
            },
        }
    except BaseException:
        if temporary.exists():
            thaw_tree_for_cleanup(temporary)
            shutil.rmtree(temporary)
        raise


__all__ = ["publish_benchmark_candidate"]
