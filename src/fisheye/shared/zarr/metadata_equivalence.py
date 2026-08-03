"""Fail-closed direct/consolidated Zarr v3 metadata comparison.

The archive-root inline metadata map is the remote-reader view.  Publication
and registry finalization must not treat it as equivalent merely because both
the direct and consolidated APIs can open: Zarr may silently fall back to
direct metadata when consolidation is absent or stale.  This module compares
the persisted JSON declarations themselves for one exact subtree.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .manifest_digest import (
    canonical_json_sha256,
    metadata_without_empty_group_consolidation,
)


METADATA_EQUIVALENCE_SCHEMA_ID = "palette.zarr.metadata_equivalence"
METADATA_EQUIVALENCE_SCHEMA_VERSION = 1
_INLINE_ENVELOPE_FIELDS = frozenset({"kind", "must_understand", "metadata"})


class ZarrMetadataEquivalenceError(RuntimeError):
    """Direct and archive-root consolidated metadata are not exactly equivalent."""


@dataclass(frozen=True)
class MetadataEquivalenceReceipt:
    subtree_path: str
    node_count: int
    group_count: int
    array_count: int
    declarations_sha256: str

    def to_json(self) -> dict[str, object]:
        return {
            "schema_id": METADATA_EQUIVALENCE_SCHEMA_ID,
            "schema_version": METADATA_EQUIVALENCE_SCHEMA_VERSION,
            "subtree_path": self.subtree_path,
            "node_count": self.node_count,
            "group_count": self.group_count,
            "array_count": self.array_count,
            "declarations_sha256": self.declarations_sha256,
        }


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ZarrMetadataEquivalenceError(
            f"Cannot read strict {label} JSON at {path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ZarrMetadataEquivalenceError(
            f"{label} metadata must be one JSON object: {path}"
        )
    return dict(payload)


def _normalize_subtree_path(value: str) -> str:
    if type(value) is not str:
        raise TypeError("subtree_path must be an exact string")
    normalized = value.strip().strip("/")
    components = normalized.split("/") if normalized else []
    if not components or any(part in {"", ".", ".."} for part in components):
        raise ValueError("subtree_path must name one non-root canonical subtree")
    return "/".join(components)


def _normalize_declaration(value: object, *, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ZarrMetadataEquivalenceError(
            f"Zarr metadata declaration at {path!r} is not an object"
        )
    declaration = dict(value)
    node_type = declaration.get("node_type")
    if node_type not in {"group", "array"}:
        raise ZarrMetadataEquivalenceError(
            f"Zarr metadata declaration at {path!r} has unsupported node_type"
        )
    try:
        return metadata_without_empty_group_consolidation(
            declaration,
            path=path,
        )
    except ValueError as exc:
        raise ZarrMetadataEquivalenceError(str(exc)) from exc


def validate_direct_consolidated_subtree(
    archive_path: str | Path,
    *,
    subtree_path: str,
) -> MetadataEquivalenceReceipt:
    """Compare every direct declaration in one subtree to the inline root map.

    The comparison permits only the Zarr-Python representation difference for
    a leaf group: omitted/null ``consolidated_metadata`` versus the exact empty
    inline envelope.  Every array declaration, group attribute, descendant
    path, codec, chunk, shard, dtype, shape, and fill declaration remains exact.
    """

    archive = Path(archive_path).expanduser().resolve()
    if not archive.is_dir():
        raise FileNotFoundError(f"Zarr archive does not exist: {archive}")
    relative_subtree = _normalize_subtree_path(subtree_path)
    subtree = archive.joinpath(*relative_subtree.split("/"))
    if not subtree.is_dir():
        raise FileNotFoundError(f"Zarr subtree does not exist: {subtree}")
    try:
        subtree.resolve().relative_to(archive)
    except ValueError as exc:
        raise ValueError("subtree_path resolves outside the archive") from exc

    root = _read_json_object(archive / "zarr.json", label="archive-root")
    envelope = root.get("consolidated_metadata")
    if not isinstance(envelope, Mapping):
        raise ZarrMetadataEquivalenceError(
            "Archive root has no persisted consolidated_metadata envelope"
        )
    if set(envelope) != _INLINE_ENVELOPE_FIELDS:
        raise ZarrMetadataEquivalenceError(
            "Archive consolidated_metadata envelope has unexpected fields"
        )
    if envelope.get("kind") != "inline" or envelope.get("must_understand") is not False:
        raise ZarrMetadataEquivalenceError(
            "Archive consolidated_metadata must be supported inline metadata"
        )
    inline_raw = envelope.get("metadata")
    if not isinstance(inline_raw, Mapping) or not all(
        isinstance(path, str) for path in inline_raw
    ):
        raise ZarrMetadataEquivalenceError(
            "Archive consolidated_metadata.metadata must be a string-keyed object"
        )

    direct_raw: dict[str, dict[str, Any]] = {}
    for candidate in subtree.rglob("*"):
        if candidate.is_symlink():
            raise ZarrMetadataEquivalenceError(
                f"Selected Zarr subtree contains a symlink: {candidate}"
            )
    for metadata_path in sorted(subtree.rglob("zarr.json")):
        if not metadata_path.is_file():
            continue
        try:
            metadata_path.resolve().relative_to(subtree.resolve())
        except ValueError as exc:
            raise ZarrMetadataEquivalenceError(
                f"Zarr metadata path escapes the selected subtree: {metadata_path}"
            ) from exc
        node_path = metadata_path.parent.relative_to(archive).as_posix()
        direct_raw[node_path] = _read_json_object(
            metadata_path,
            label="direct-node",
        )

    prefix = f"{relative_subtree}/"
    inline_subtree = {
        path: value
        for path, value in inline_raw.items()
        if path == relative_subtree or path.startswith(prefix)
    }
    if set(direct_raw) != set(inline_subtree):
        missing = sorted(set(direct_raw) - set(inline_subtree))
        unexpected = sorted(set(inline_subtree) - set(direct_raw))
        raise ZarrMetadataEquivalenceError(
            "Direct/consolidated subtree paths differ: "
            f"missing_inline={missing}, unexpected_inline={unexpected}"
        )

    normalized: dict[str, dict[str, Any]] = {}
    group_count = 0
    array_count = 0
    for path in sorted(direct_raw):
        direct = _normalize_declaration(direct_raw[path], path=path)
        inline = _normalize_declaration(inline_subtree[path], path=path)
        if direct != inline:
            raise ZarrMetadataEquivalenceError(
                f"Direct/consolidated declaration differs at {path!r}"
            )
        normalized[path] = direct
        if direct["node_type"] == "group":
            group_count += 1
        else:
            array_count += 1

    if not normalized or relative_subtree not in normalized:
        raise ZarrMetadataEquivalenceError(
            "Selected subtree has no direct root Zarr declaration"
        )
    return MetadataEquivalenceReceipt(
        subtree_path=relative_subtree,
        node_count=len(normalized),
        group_count=group_count,
        array_count=array_count,
        declarations_sha256=canonical_json_sha256(normalized),
    )


__all__ = [
    "METADATA_EQUIVALENCE_SCHEMA_ID",
    "METADATA_EQUIVALENCE_SCHEMA_VERSION",
    "MetadataEquivalenceReceipt",
    "ZarrMetadataEquivalenceError",
    "validate_direct_consolidated_subtree",
]
