"""Filesystem primitives for immutable in-archive coordinate successors."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
from typing import Any
import shutil

from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


PAYLOAD_FILE_EQUIVALENCE_SCHEMA_ID = (
    "palette.coordinate_successor_payload_file_equivalence"
)
PAYLOAD_FILE_EQUIVALENCE_SCHEMA_VERSION = 1
PAYLOAD_FILE_EQUIVALENCE_DIGEST_ALGORITHM = "sha256_canonical_json_v1"


class PayloadFileEquivalenceError(ValueError):
    """Raised when a copied successor payload is not safely equivalent."""


def _safe_archive_relative_label(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise PayloadFileEquivalenceError(
            f"{name} must be one non-empty archive-relative label."
        )
    if (
        value.startswith("/")
        or value.endswith("/")
        or "\\" in value
        or "\x00" in value
        or any(ord(character) < 0x20 for character in value)
        or ":" in value
    ):
        raise PayloadFileEquivalenceError(
            f"{name} is not a safe archive-relative label."
        )
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise PayloadFileEquivalenceError(
            f"{name} is not a safe archive-relative label."
        )
    return value


def _payload_inventory(path: Path, *, label: str) -> list[dict[str, Any]]:
    try:
        root_stat = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise PayloadFileEquivalenceError(
            f"{label} payload root cannot be inspected: {exc}"
        ) from exc
    if not stat.S_ISDIR(root_stat.st_mode):
        raise PayloadFileEquivalenceError(
            f"{label} payload root must be a regular directory."
        )

    inventory: list[dict[str, Any]] = []

    def visit(directory: Path, relative_directory: str) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise PayloadFileEquivalenceError(
                f"{label} payload directory cannot be inspected at "
                f"{relative_directory or '.'}: {exc}"
            ) from exc
        for entry in entries:
            relative = (
                entry.name
                if not relative_directory
                else f"{relative_directory}/{entry.name}"
            )
            try:
                entry_stat = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise PayloadFileEquivalenceError(
                    f"{label} payload entry {relative!r} cannot be inspected: {exc}"
                ) from exc
            mode = entry_stat.st_mode
            if stat.S_ISDIR(mode):
                visit(Path(entry.path), relative)
            elif entry.name.endswith(".json"):
                continue
            elif stat.S_ISREG(mode):
                inventory.append(
                    {"path": relative, "size_bytes": int(entry_stat.st_size)}
                )
            else:
                raise PayloadFileEquivalenceError(
                    f"{label} payload entry {relative!r} is non-regular."
                )

    visit(path, "")
    return inventory


def validate_payload_file_equivalence(
    source: Path,
    target: Path,
    *,
    source_label: str,
    target_label: str,
) -> dict[str, Any]:
    """Validate hard-linked successor payload identity without reading bytes."""

    source_label = _safe_archive_relative_label(source_label, name="source_label")
    target_label = _safe_archive_relative_label(target_label, name="target_label")
    source = Path(source)
    target = Path(target)
    source_inventory = _payload_inventory(source, label=source_label)
    target_inventory = _payload_inventory(target, label=target_label)
    source_by_path = {entry["path"]: entry for entry in source_inventory}
    target_by_path = {entry["path"]: entry for entry in target_inventory}

    missing = sorted(set(source_by_path) - set(target_by_path))
    if missing:
        raise PayloadFileEquivalenceError(
            f"{target_label} is missing payload files: {missing}"
        )
    extra = sorted(set(target_by_path) - set(source_by_path))
    if extra:
        raise PayloadFileEquivalenceError(
            f"{target_label} has extra payload files: {extra}"
        )

    for relative in sorted(source_by_path):
        source_entry = source_by_path[relative]
        target_entry = target_by_path[relative]
        if source_entry["size_bytes"] != target_entry["size_bytes"]:
            raise PayloadFileEquivalenceError(
                f"Payload size differs for {relative!r}: "
                f"{source_entry['size_bytes']} != {target_entry['size_bytes']}"
            )
        try:
            hardlinked = os.path.samefile(source / relative, target / relative)
        except OSError as exc:
            raise PayloadFileEquivalenceError(
                f"Payload hardlink cannot be verified for {relative!r}: {exc}"
            ) from exc
        if not hardlinked:
            raise PayloadFileEquivalenceError(
                f"Payload files are not hard-linked for {relative!r}."
            )

    inventory_body = {
        "schema_id": PAYLOAD_FILE_EQUIVALENCE_SCHEMA_ID,
        "schema_version": PAYLOAD_FILE_EQUIVALENCE_SCHEMA_VERSION,
        "digest_algorithm": PAYLOAD_FILE_EQUIVALENCE_DIGEST_ALGORITHM,
        "payload_files": source_inventory,
    }
    body = {
        **inventory_body,
        "source_label": source_label,
        "target_label": target_label,
        "payload_file_count": len(source_inventory),
        "inventory_digest": canonical_json_sha256(inventory_body),
        "hardlink_validation": "samefile_for_every_payload_file_v1",
    }
    return {**body, "receipt_digest": canonical_json_sha256(body)}


def metadata_tree_sha256(path: Path) -> str:
    """Hash all direct Zarr metadata without reading array payload objects."""

    digest = hashlib.sha256()
    for candidate in sorted(path.rglob("zarr.json")):
        digest.update(candidate.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(candidate.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def copy_metadata_and_link_payload(source: Path, target: Path) -> dict[str, int]:
    """Copy mutable metadata while hard-linking immutable payload files."""

    if target.exists():
        raise FileExistsError(f"Immutable successor target exists: {target}")
    target.mkdir(parents=True, exist_ok=False)
    linked = 0
    copied = 0
    try:
        for source_path in sorted(source.rglob("*")):
            relative = source_path.relative_to(source)
            target_path = target / relative
            if source_path.is_dir():
                target_path.mkdir(exist_ok=False)
            elif source_path.is_file():
                if source_path.name == "zarr.json" or source_path.suffix == ".json":
                    shutil.copy2(source_path, target_path)
                    copied += 1
                else:
                    os.link(source_path, target_path)
                    linked += 1
            else:
                raise ValueError(f"Unsupported source entry: {source_path}")
    except BaseException:
        shutil.rmtree(target, ignore_errors=True)
        raise
    return {"metadata_files_copied": copied, "payload_files_hardlinked": linked}


__all__ = [
    "PAYLOAD_FILE_EQUIVALENCE_DIGEST_ALGORITHM",
    "PAYLOAD_FILE_EQUIVALENCE_SCHEMA_ID",
    "PAYLOAD_FILE_EQUIVALENCE_SCHEMA_VERSION",
    "PayloadFileEquivalenceError",
    "copy_metadata_and_link_payload",
    "metadata_tree_sha256",
    "validate_payload_file_equivalence",
]
