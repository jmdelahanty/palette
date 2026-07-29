"""Strict persisted envelope for a stage's coordinate-contract catalog."""

from __future__ import annotations

from typing import Any, Mapping

from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_bytes,
    canonical_json_sha256,
)


COORDINATE_CATALOG_ENVELOPE_SCHEMA_ID = "palette.persisted_coordinate_catalog"
COORDINATE_CATALOG_ENVELOPE_SCHEMA_VERSION = 1


def build_coordinate_catalog_envelope(
    document: Mapping[str, Any],
) -> dict[str, object]:
    """Bind one exact schema-level coordinate catalog by canonical JSON digest."""

    normalized = dict(document)
    canonical_json_bytes(normalized)
    envelope: dict[str, object] = {
        "schema_id": COORDINATE_CATALOG_ENVELOPE_SCHEMA_ID,
        "schema_version": COORDINATE_CATALOG_ENVELOPE_SCHEMA_VERSION,
        "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
        "digest": canonical_json_sha256(normalized),
        "document": normalized,
    }
    canonical_json_bytes(envelope)
    return envelope


def validate_coordinate_catalog_envelope(
    value: Any,
    *,
    expected_document: Mapping[str, Any],
) -> tuple[str, ...]:
    """Deeply validate identity, digest, and the complete frozen catalog."""

    if not isinstance(value, Mapping):
        return ("coordinate catalog envelope must be an object",)
    expected_fields = {
        "schema_id",
        "schema_version",
        "digest_algorithm",
        "digest",
        "document",
    }
    errors: list[str] = []
    if set(value) != expected_fields:
        errors.append("coordinate catalog envelope has an unexpected field set")
    document = value.get("document")
    if not isinstance(document, Mapping):
        return (*errors, "coordinate catalog document must be an object")
    try:
        expected = build_coordinate_catalog_envelope(expected_document)
        canonical_json_bytes(dict(value))
    except (TypeError, ValueError) as exc:
        return (*errors, f"coordinate catalog envelope is not strict JSON: {exc}")
    if dict(value) != expected:
        errors.append("coordinate catalog differs from the frozen stage catalog")
    return tuple(dict.fromkeys(errors))


__all__ = [
    "COORDINATE_CATALOG_ENVELOPE_SCHEMA_ID",
    "COORDINATE_CATALOG_ENVELOPE_SCHEMA_VERSION",
    "build_coordinate_catalog_envelope",
    "validate_coordinate_catalog_envelope",
]
