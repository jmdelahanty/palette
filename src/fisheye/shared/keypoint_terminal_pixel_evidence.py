"""Typed pixel-origin evidence for canonical keypoint terminal artifacts."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from fisheye.shared.zarr.crop_consumer import (
    CROP_RUN_REFERENCE_SIGNED_PROFILE,
    validate_crop_run_reference,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


TERMINAL_PIXEL_EVIDENCE_SCHEMA_ID = "palette.keypoint.terminal_pixel_evidence"
TERMINAL_PIXEL_EVIDENCE_SCHEMA_VERSION = 1
DIRECT_HYBRID_TERMINAL_EVIDENCE_PROFILE = (
    "signed_hybrid_pixels_with_strict_crop_v2_geometry_v1"
)
DIRECT_HYBRID_SOURCE_ROSTER_SCHEMA_ID = (
    "palette.keypoint.direct_hybrid_source_shard_roster"
)
DIRECT_HYBRID_SOURCE_ROSTER_SCHEMA_VERSION = 1
_SHA256_HEX = frozenset("0123456789abcdef")


def _require_sha256(value: object, *, name: str) -> str:
    text = str(value).strip()
    if len(text) != 64 or any(character not in _SHA256_HEX for character in text):
        raise ValueError(f"{name} must be a lowercase hexadecimal SHA-256 digest.")
    return text


def _require_component(value: object, *, name: str) -> str:
    text = str(value).strip()
    if not text or "/" in text:
        raise ValueError(f"{name} must be one nonempty path-safe component.")
    return text


def build_direct_hybrid_terminal_pixel_evidence(
    *,
    provider_run: str,
    provider_reference: Mapping[str, Any],
    provider_binding: Mapping[str, Any],
    geometry_crop_run: str,
    geometry_crop_manifest_digest: str,
    source_shard_runs: Sequence[str],
    source_shard_evidence_digest: str,
) -> dict[str, Any]:
    """Build the exact direct-hybrid evidence profile after full admission."""

    reference = validate_crop_run_reference(dict(provider_reference))
    if reference.get("profile") != CROP_RUN_REFERENCE_SIGNED_PROFILE:
        raise ValueError("Direct-hybrid terminal evidence requires a signed crop reference.")
    resolved_provider_run = _require_component(provider_run, name="provider_run")
    if reference.get("run_id") != resolved_provider_run:
        raise ValueError("Direct-hybrid provider reference binds another run.")
    resolved_geometry_run = _require_component(
        geometry_crop_run,
        name="geometry_crop_run",
    )
    runs = [_require_component(value, name="source_shard_run") for value in source_shard_runs]
    if not runs or len(set(runs)) != len(runs):
        raise ValueError("Direct-hybrid source shard roster must be nonempty and unique.")
    binding_fields = {
        "provider_record_sha256": _require_sha256(
            provider_binding.get("provider_record_sha256"),
            name="provider_record_sha256",
        ),
        "source_pixel_fingerprint": _require_sha256(
            provider_binding.get("source_pixel_fingerprint"),
            name="source_pixel_fingerprint",
        ),
        "source_rowset_fingerprint": _require_sha256(
            provider_binding.get("source_rowset_fingerprint"),
            name="source_rowset_fingerprint",
        ),
        "source_row_signature_spec_digest": _require_sha256(
            provider_binding.get("source_row_signature_spec_digest"),
            name="source_row_signature_spec_digest",
        ),
    }
    roster = {
        "schema_id": DIRECT_HYBRID_SOURCE_ROSTER_SCHEMA_ID,
        "schema_version": DIRECT_HYBRID_SOURCE_ROSTER_SCHEMA_VERSION,
        "shard_count": len(runs),
        "shard_runs": runs,
        "evidence_digest": _require_sha256(
            source_shard_evidence_digest,
            name="source_shard_evidence_digest",
        ),
    }
    payload = {
        "schema_id": TERMINAL_PIXEL_EVIDENCE_SCHEMA_ID,
        "schema_version": TERMINAL_PIXEL_EVIDENCE_SCHEMA_VERSION,
        "profile": DIRECT_HYBRID_TERMINAL_EVIDENCE_PROFILE,
        "provider_run": resolved_provider_run,
        "provider_reference": reference,
        "provider_binding": binding_fields,
        "geometry_crop_run": resolved_geometry_run,
        "geometry_crop_manifest_digest": _require_sha256(
            geometry_crop_manifest_digest,
            name="geometry_crop_manifest_digest",
        ),
        "ordered_geometry_coverage_exact": True,
        "source_shard_roster": roster,
    }
    return validate_direct_hybrid_terminal_pixel_evidence(payload)


def validate_direct_hybrid_terminal_pixel_evidence(value: Any) -> dict[str, Any]:
    """Validate and return one exact direct-hybrid evidence declaration."""

    if not isinstance(value, Mapping):
        raise ValueError("Terminal pixel evidence must be an object.")
    evidence = dict(value)
    expected = {
        "schema_id",
        "schema_version",
        "profile",
        "provider_run",
        "provider_reference",
        "provider_binding",
        "geometry_crop_run",
        "geometry_crop_manifest_digest",
        "ordered_geometry_coverage_exact",
        "source_shard_roster",
    }
    if set(evidence) != expected:
        raise ValueError("Terminal pixel evidence has an unexpected field set.")
    if (
        evidence.get("schema_id") != TERMINAL_PIXEL_EVIDENCE_SCHEMA_ID
        or evidence.get("schema_version") != TERMINAL_PIXEL_EVIDENCE_SCHEMA_VERSION
        or evidence.get("profile") != DIRECT_HYBRID_TERMINAL_EVIDENCE_PROFILE
        or evidence.get("ordered_geometry_coverage_exact") is not True
    ):
        raise ValueError("Terminal pixel evidence schema/profile mismatch.")
    provider_run = _require_component(evidence.get("provider_run"), name="provider_run")
    _require_component(evidence.get("geometry_crop_run"), name="geometry_crop_run")
    _require_sha256(
        evidence.get("geometry_crop_manifest_digest"),
        name="geometry_crop_manifest_digest",
    )
    reference = validate_crop_run_reference(evidence.get("provider_reference"))
    if (
        reference.get("profile") != CROP_RUN_REFERENCE_SIGNED_PROFILE
        or reference.get("run_id") != provider_run
    ):
        raise ValueError("Terminal pixel evidence provider reference is incompatible.")
    binding = evidence.get("provider_binding")
    required_binding = {
        "provider_record_sha256",
        "source_pixel_fingerprint",
        "source_rowset_fingerprint",
        "source_row_signature_spec_digest",
    }
    if not isinstance(binding, Mapping) or set(binding) != required_binding:
        raise ValueError("Terminal pixel evidence provider binding is incomplete.")
    for name in sorted(required_binding):
        _require_sha256(binding.get(name), name=name)
    roster = evidence.get("source_shard_roster")
    required_roster = {
        "schema_id",
        "schema_version",
        "shard_count",
        "shard_runs",
        "evidence_digest",
    }
    if not isinstance(roster, Mapping) or set(roster) != required_roster:
        raise ValueError("Direct-hybrid source shard roster is incomplete.")
    runs = roster.get("shard_runs")
    if (
        roster.get("schema_id") != DIRECT_HYBRID_SOURCE_ROSTER_SCHEMA_ID
        or roster.get("schema_version")
        != DIRECT_HYBRID_SOURCE_ROSTER_SCHEMA_VERSION
        or not isinstance(runs, list)
        or not runs
        or roster.get("shard_count") != len(runs)
    ):
        raise ValueError("Direct-hybrid source shard roster schema mismatch.")
    normalized_runs = [_require_component(item, name="source_shard_run") for item in runs]
    if len(set(normalized_runs)) != len(normalized_runs):
        raise ValueError("Direct-hybrid source shard roster contains duplicates.")
    _require_sha256(roster.get("evidence_digest"), name="source_shard_evidence_digest")
    canonical_json_sha256(evidence)
    return evidence


__all__ = [
    "DIRECT_HYBRID_SOURCE_ROSTER_SCHEMA_ID",
    "DIRECT_HYBRID_SOURCE_ROSTER_SCHEMA_VERSION",
    "DIRECT_HYBRID_TERMINAL_EVIDENCE_PROFILE",
    "TERMINAL_PIXEL_EVIDENCE_SCHEMA_ID",
    "TERMINAL_PIXEL_EVIDENCE_SCHEMA_VERSION",
    "build_direct_hybrid_terminal_pixel_evidence",
    "validate_direct_hybrid_terminal_pixel_evidence",
]
