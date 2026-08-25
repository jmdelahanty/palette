from __future__ import annotations

import pytest

from fisheye.shared.keypoint_terminal_pixel_evidence import (
    DIRECT_HYBRID_TERMINAL_EVIDENCE_PROFILE,
    build_direct_hybrid_terminal_pixel_evidence,
    validate_direct_hybrid_terminal_pixel_evidence,
)


def _reference() -> dict[str, object]:
    return {
        "schema_id": "palette.crop_geometry.run_reference",
        "schema_version": 1,
        "profile": "signed_current_source_v1",
        "run_id": "provider_v1",
        "crop_signature": {"provider": "test"},
        "crop_revision": 1,
    }


def _binding() -> dict[str, str]:
    return {
        "provider_record_sha256": "a" * 64,
        "source_pixel_fingerprint": "b" * 64,
        "source_rowset_fingerprint": "c" * 64,
        "source_row_signature_spec_digest": "d" * 64,
    }


def test_direct_hybrid_terminal_pixel_evidence_is_exact_and_typed() -> None:
    evidence = build_direct_hybrid_terminal_pixel_evidence(
        provider_run="provider_v1",
        provider_reference=_reference(),
        provider_binding=_binding(),
        geometry_crop_run="crop_v2",
        geometry_crop_manifest_digest="e" * 64,
        source_shard_runs=("shard_0", "shard_1"),
        source_shard_evidence_digest="f" * 64,
    )

    assert evidence["profile"] == DIRECT_HYBRID_TERMINAL_EVIDENCE_PROFILE
    assert evidence["ordered_geometry_coverage_exact"] is True
    assert evidence["source_shard_roster"]["shard_count"] == 2
    assert validate_direct_hybrid_terminal_pixel_evidence(evidence) == evidence


def test_direct_hybrid_terminal_pixel_evidence_rejects_tamper() -> None:
    evidence = build_direct_hybrid_terminal_pixel_evidence(
        provider_run="provider_v1",
        provider_reference=_reference(),
        provider_binding=_binding(),
        geometry_crop_run="crop_v2",
        geometry_crop_manifest_digest="e" * 64,
        source_shard_runs=("shard_0",),
        source_shard_evidence_digest="f" * 64,
    )

    evidence["provider_binding"]["source_pixel_fingerprint"] = "not-a-digest"
    with pytest.raises(ValueError, match="source_pixel_fingerprint"):
        validate_direct_hybrid_terminal_pixel_evidence(evidence)

    with pytest.raises(ValueError, match="unique"):
        build_direct_hybrid_terminal_pixel_evidence(
            provider_run="provider_v1",
            provider_reference=_reference(),
            provider_binding=_binding(),
            geometry_crop_run="crop_v2",
            geometry_crop_manifest_digest="e" * 64,
            source_shard_runs=("shard_0", "shard_0"),
            source_shard_evidence_digest="f" * 64,
        )
