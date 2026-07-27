"""Strict canonical-JSON helpers shared by persisted Zarr manifests."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


CANONICAL_JSON_DIGEST_ALGORITHM = "sha256_canonical_json_v1"
EMPTY_INLINE_CONSOLIDATED_METADATA = {
    "kind": "inline",
    "must_understand": False,
    "metadata": {},
}


def canonical_json_bytes(value: object) -> bytes:
    """Return deterministic strict UTF-8 JSON suitable for contract digests."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: object) -> str:
    """Return the lowercase SHA-256 of :func:`canonical_json_bytes`."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def metadata_without_empty_group_consolidation(
    value: Mapping[str, Any],
    *,
    path: str,
) -> dict[str, Any]:
    """Canonicalize equivalent absent, null, and empty leaf-group encodings.

    Zarr-Python 3.1.3 writes ``consolidated_metadata: null`` to a group's
    direct ``zarr.json``.  During archive-root consolidation it deliberately
    represents the same child group with an empty inline envelope so the
    consolidated hierarchy records that the group has no nested metadata.
    An omitted field is also equivalent under the Zarr declaration model.
    Only those exact representations are equivalent.  Array-level, non-empty,
    or otherwise malformed envelopes remain contract errors.
    """

    normalized = dict(value)
    has_envelope = "consolidated_metadata" in normalized
    envelope = normalized.pop("consolidated_metadata", None)
    if not has_envelope:
        return normalized
    if normalized.get("node_type") != "group":
        raise ValueError(
            f"Only Zarr groups may declare consolidated_metadata at {path!r}."
        )
    if envelope is not None and envelope != EMPTY_INLINE_CONSOLIDATED_METADATA:
        raise ValueError(
            f"Zarr group consolidated_metadata at {path!r} must be null or "
            "the exact empty inline group envelope."
        )
    return normalized


__all__ = [
    "CANONICAL_JSON_DIGEST_ALGORITHM",
    "EMPTY_INLINE_CONSOLIDATED_METADATA",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "metadata_without_empty_group_consolidation",
]
