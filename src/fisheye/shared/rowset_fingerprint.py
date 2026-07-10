"""Deterministic identity fingerprints for sparse observation rowsets.

This is deliberately separate from run-lineage fingerprints. A rowset
fingerprint answers whether the exact set of observation identities consumed by
a row-aligned stage is unchanged; it is not an artifact hash or an animal
appearance signature.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

import numpy as np


ROWSET_FINGERPRINT_SCHEMA_ID = "palette.observation_rowset_fingerprint"
ROWSET_FINGERPRINT_SCHEMA_VERSION = 1
ROWSET_FINGERPRINT_CANONICALIZATION = "sha256_canonical_json_sorted_uint64_instance_keys_v1"
ROWSET_FINGERPRINT_STATUS_COMPLETE = "complete"
ROWSET_FINGERPRINT_STATUS_MISSING_INSTANCE_KEY = "missing_instance_key"


class RowsetFingerprintError(ValueError):
    """Raised when a rowset cannot satisfy the modern identity contract."""


@dataclass(frozen=True)
class RowsetFingerprint:
    """Canonical identity summary for one exact observation rowset."""

    source_rowset_path: str
    row_count: int
    source_edit_revision: int | None
    instance_key_digest: str | None
    fingerprint: str | None
    status: str

    @property
    def is_complete(self) -> bool:
        return self.status == ROWSET_FINGERPRINT_STATUS_COMPLETE

    def to_attrs(self, *, prefix: str = "source_rowset_") -> dict[str, Any]:
        """Return JSON-safe attrs with an explicit namespace prefix."""

        attrs: dict[str, Any] = {
            f"{prefix}fingerprint_schema_id": ROWSET_FINGERPRINT_SCHEMA_ID,
            f"{prefix}fingerprint_schema_version": ROWSET_FINGERPRINT_SCHEMA_VERSION,
            f"{prefix}fingerprint_canonicalization": ROWSET_FINGERPRINT_CANONICALIZATION,
            f"{prefix}fingerprint_status": self.status,
            f"{prefix}path": self.source_rowset_path,
            f"{prefix}row_count": self.row_count,
        }
        if self.source_edit_revision is not None:
            attrs[f"{prefix}edit_revision"] = self.source_edit_revision
        if self.instance_key_digest is not None:
            attrs[f"{prefix}instance_key_digest"] = self.instance_key_digest
        if self.fingerprint is not None:
            attrs[f"{prefix}fingerprint"] = self.fingerprint
        return attrs


def _normalize_edit_revision(value: object | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise RowsetFingerprintError(f"Invalid source edit revision: {value!r}") from exc


def resolve_rowset_edit_revision(*attr_sources: Mapping[str, Any] | None) -> int | None:
    """Resolve the first declared authoring/edit revision from source attrs."""

    for attrs in attr_sources:
        if attrs is None:
            continue
        for name in ("edit_revision", "authoring_revision", "rowset_revision"):
            if name in attrs and attrs.get(name) is not None:
                return _normalize_edit_revision(attrs.get(name))
    return None


def instance_key_digest(instance_keys: np.ndarray, *, expected_row_count: int | None = None) -> str:
    """Hash sorted unique uint64 observation keys with platform-neutral bytes."""

    keys = np.asarray(instance_keys, dtype=np.uint64).reshape(-1)
    if expected_row_count is not None and int(keys.shape[0]) != int(expected_row_count):
        raise RowsetFingerprintError(
            "instance_key row count does not match the source rowset "
            f"({int(keys.shape[0])} != {int(expected_row_count)})."
        )
    if int(np.unique(keys).shape[0]) != int(keys.shape[0]):
        raise RowsetFingerprintError("Duplicate instance_key values in source rowset.")
    canonical = np.sort(keys, kind="stable").astype("<u8", copy=False)
    hasher = hashlib.sha256()
    hasher.update(ROWSET_FINGERPRINT_SCHEMA_ID.encode("utf-8"))
    hasher.update(b"\x00instance_keys\x00")
    hasher.update(canonical.tobytes(order="C"))
    return hasher.hexdigest()


def build_rowset_fingerprint(
    *,
    source_rowset_path: str,
    row_count: int,
    instance_keys: np.ndarray | None,
    source_edit_revision: object | None = None,
) -> RowsetFingerprint:
    """Build a rowset fingerprint or an explicit legacy/missing-key result."""

    path = str(source_rowset_path).strip()
    if not path:
        raise RowsetFingerprintError("source_rowset_path must be non-empty.")
    count = int(row_count)
    if count < 0:
        raise RowsetFingerprintError("row_count must be nonnegative.")
    edit_revision = _normalize_edit_revision(source_edit_revision)
    if instance_keys is None:
        return RowsetFingerprint(
            source_rowset_path=path,
            row_count=count,
            source_edit_revision=edit_revision,
            instance_key_digest=None,
            fingerprint=None,
            status=ROWSET_FINGERPRINT_STATUS_MISSING_INSTANCE_KEY,
        )

    key_digest = instance_key_digest(instance_keys, expected_row_count=count)
    payload = {
        "schema_id": ROWSET_FINGERPRINT_SCHEMA_ID,
        "schema_version": ROWSET_FINGERPRINT_SCHEMA_VERSION,
        "source_rowset_path": path,
        "row_count": count,
        "source_edit_revision": edit_revision,
        "instance_key_digest": key_digest,
    }
    canonical_json = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return RowsetFingerprint(
        source_rowset_path=path,
        row_count=count,
        source_edit_revision=edit_revision,
        instance_key_digest=key_digest,
        fingerprint=hashlib.sha256(canonical_json.encode("utf-8")).hexdigest(),
        status=ROWSET_FINGERPRINT_STATUS_COMPLETE,
    )


def build_group_rowset_fingerprint(
    group: Any,
    *,
    source_rowset_path: str,
    source_edit_revision: object | None = None,
) -> RowsetFingerprint:
    """Build a rowset fingerprint from a Zarr-like row group."""

    if "frame_indices" not in group:
        raise RowsetFingerprintError("Source rowset is missing frame_indices.")
    row_count = int(group["frame_indices"].shape[0])
    keys = (
        np.asarray(group["instance_key"][:], dtype=np.uint64).reshape(-1)
        if "instance_key" in group
        else None
    )
    revision = source_edit_revision
    if revision is None and hasattr(group, "attrs"):
        revision = resolve_rowset_edit_revision(group.attrs)
    return build_rowset_fingerprint(
        source_rowset_path=source_rowset_path,
        row_count=row_count,
        instance_keys=keys,
        source_edit_revision=revision,
    )


def assert_rowset_fingerprint_matches(
    expected: RowsetFingerprint,
    actual: RowsetFingerprint,
    *,
    require_complete: bool = False,
) -> None:
    """Fail closed when two complete rowset identities disagree."""

    if require_complete and (not expected.is_complete or not actual.is_complete):
        raise RowsetFingerprintError(
            "A complete source rowset fingerprint is required for this operation."
        )
    if expected.source_rowset_path != actual.source_rowset_path:
        raise RowsetFingerprintError(
            "Source rowset path changed during processing "
            f"({expected.source_rowset_path!r} != {actual.source_rowset_path!r})."
        )
    if expected.row_count != actual.row_count:
        raise RowsetFingerprintError(
            "Source row count changed during processing "
            f"({expected.row_count} != {actual.row_count})."
        )
    if expected.source_edit_revision != actual.source_edit_revision:
        raise RowsetFingerprintError(
            "Source edit revision changed during processing "
            f"({expected.source_edit_revision!r} != {actual.source_edit_revision!r})."
        )
    if expected.is_complete and actual.is_complete and expected.fingerprint != actual.fingerprint:
        raise RowsetFingerprintError("Source rowset fingerprint changed during processing.")
    if expected.is_complete != actual.is_complete:
        raise RowsetFingerprintError("Source rowset fingerprint availability changed during processing.")


__all__ = [
    "ROWSET_FINGERPRINT_SCHEMA_ID",
    "ROWSET_FINGERPRINT_SCHEMA_VERSION",
    "ROWSET_FINGERPRINT_CANONICALIZATION",
    "ROWSET_FINGERPRINT_STATUS_COMPLETE",
    "ROWSET_FINGERPRINT_STATUS_MISSING_INSTANCE_KEY",
    "RowsetFingerprint",
    "RowsetFingerprintError",
    "resolve_rowset_edit_revision",
    "instance_key_digest",
    "build_rowset_fingerprint",
    "build_group_rowset_fingerprint",
    "assert_rowset_fingerprint_matches",
]
