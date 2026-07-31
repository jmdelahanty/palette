from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest

from fisheye.shared.mask_rle import encode_mask_component_rle
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.refined_subject_mask_extensions import (
    REFINED_SUBJECT_MASK_DRAFT_AUDIT_SCHEMA_V1,
    SUBJECT_MASK_CACHE_VALIDATION_MODE,
    SubjectMaskDerivedCacheKind,
    SubjectMaskDerivedCacheReceipt,
    default_subject_mask_sampled_contour_profile,
    published_subject_mask_cache_extension_manifest,
    validate_published_subject_mask_cache_extension,
    validate_subject_mask_cache_arrays,
    validate_subject_mask_cache_receipt,
)
from fisheye.shared.zarr.subject_mask_schema import (
    SubjectMaskComponentRegistry,
    SubjectMaskDimensions,
)


def _dimensions() -> SubjectMaskDimensions:
    return SubjectMaskDimensions(
        n_frames=4,
        n_rois=3,
        n_channels=2,
        roi_height=4,
        roi_width=5,
    )


def test_default_contour_profile_prefers_fixed_samples_and_makes_full_optional() -> (
    None
):
    components = SubjectMaskComponentRegistry(
        ("subject_body", "eye_left", "eye_right", "swim_bladder")
    )
    profile = default_subject_mask_sampled_contour_profile(components)
    manifest = profile.as_manifest(components=components)

    assert manifest["default_cache"]["component_sample_counts"] == {
        "subject_body": 128,
        "eye_left": 64,
        "eye_right": 64,
        "swim_bladder": 32,
    }
    assert manifest["default_cache"]["winding"] == "clockwise_in_roi_y_down"
    assert manifest["default_cache"]["start_point"] == ("topmost_then_leftmost_vertex")
    assert manifest["full_contours"]["required_for_profile"] is False


def _fixed_utf8(values: tuple[str, ...], width: int) -> np.ndarray:
    result = np.zeros((len(values), width), dtype=np.uint8)
    for row, value in enumerate(values):
        encoded = value.encode("utf-8")
        result[row, : len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
    return result


def _draft_arrays() -> dict[str, np.ndarray]:
    edit = np.asarray(
        [[False, True], [False, False], [True, False]],
        dtype=bool,
    )
    timestamp = "2026-07-30T12:00:00+00:00"
    arrays: dict[str, np.ndarray] = {"edit_applied": edit}
    for index, label in enumerate(("body", "left_eye")):
        component_edit = edit[:, index].copy()
        revisions = (
            np.asarray([0, 0, 2], dtype=np.int64)
            if label == "body"
            else np.asarray([1, 0, 0], dtype=np.int64)
        )
        timestamps = tuple(timestamp if value else "" for value in revisions)
        reasons = tuple("manual_mask_edit" if value else "" for value in revisions)
        prefix = f"components/{label}"
        arrays[f"{prefix}/edit_applied"] = component_edit
        arrays[f"{prefix}/manual_override"] = component_edit.copy()
        arrays[f"{prefix}/row_revision"] = revisions
        arrays[f"{prefix}/row_updated_at_utc_bytes"] = _fixed_utf8(timestamps, 40)
        arrays[f"{prefix}/row_update_reason_bytes"] = _fixed_utf8(reasons, 128)
    return arrays


def test_editable_draft_audit_schema_accepts_exact_revision_state() -> None:
    dimensions = _dimensions()
    components = SubjectMaskComponentRegistry(("body", "left_eye"))
    arrays = _draft_arrays()

    REFINED_SUBJECT_MASK_DRAFT_AUDIT_SCHEMA_V1.require(
        arrays,
        dimensions=dimensions,
        components=components,
    )

    manifest = REFINED_SUBJECT_MASK_DRAFT_AUDIT_SCHEMA_V1.as_manifest(
        dimensions=dimensions,
        components=components,
    )
    assert manifest["selector_eligible"] is False
    assert manifest["write_contract"]["concurrency"] == "compare_and_swap_row_revision"


def test_editable_draft_audit_schema_rejects_mirror_and_audit_tampering() -> None:
    arrays = _draft_arrays()
    arrays["components/body/manual_override"][2] = False
    arrays["components/left_eye/row_updated_at_utc_bytes"][0] = 0

    issues = REFINED_SUBJECT_MASK_DRAFT_AUDIT_SCHEMA_V1.validate(
        arrays,
        dimensions=_dimensions(),
        components=SubjectMaskComponentRegistry(("body", "left_eye")),
    )

    codes = {issue.code for issue in issues}
    assert "manual_override_mismatch" in codes
    assert "invalid_row_revision_timestamp" in codes


def _receipt() -> SubjectMaskDerivedCacheReceipt:
    return SubjectMaskDerivedCacheReceipt(
        cache_kind=SubjectMaskDerivedCacheKind.BITPACKED,
        cache_path="mask_bitpacked",
        source_dense_core_manifest_digest="1" * 64,
        source_dense_array_values_sha256="2" * 64,
        component_registry_digest="3" * 64,
        logical_content_digest="4" * 64,
        generator_id="palette.mask_bitpacked",
        generator_version=1,
        generated_at_utc="2026-07-30T12:00:00+00:00",
    )


def test_published_cache_receipt_is_digest_bound_and_fail_closed() -> None:
    document = _receipt().as_manifest()
    assert validate_subject_mask_cache_receipt(document) == ()
    assert (
        document["payload"]["validation"]["mode"] == SUBJECT_MASK_CACHE_VALIDATION_MODE
    )

    tampered = copy.deepcopy(document)
    tampered["payload"]["stale"] = True
    tampered["payload_digest"] = canonical_json_sha256(tampered["payload"])

    errors = validate_subject_mask_cache_receipt(tampered)
    assert "published cache receipt must declare stale=false" in errors


def test_published_cache_extension_requires_unique_receipts() -> None:
    receipt = _receipt()
    manifest = published_subject_mask_cache_extension_manifest((receipt,))
    assert manifest["freshness_rule"] == (
        "only_full_regeneration_and_dense_equivalence_may_clear_stale"
    )
    assert manifest["receipts_digest"] == canonical_json_sha256(manifest["receipts"])
    assert validate_published_subject_mask_cache_extension(manifest) == ()

    tampered = copy.deepcopy(manifest)
    tampered["receipts"][0]["payload"]["authoritative_pixels"] = True
    tampered["receipts"][0]["payload_digest"] = canonical_json_sha256(
        tampered["receipts"][0]["payload"]
    )
    tampered["receipts_digest"] = canonical_json_sha256(tampered["receipts"])
    errors = validate_published_subject_mask_cache_extension(tampered)
    assert any("authoritative_pixels=false" in error for error in errors)


def test_published_cache_extension_requires_one_dense_authority() -> None:
    first = _receipt()
    second = replace(
        first,
        cache_kind=SubjectMaskDerivedCacheKind.RLE,
        cache_path="mask_rle",
        source_dense_array_values_sha256="9" * 64,
    )

    with pytest.raises(ValueError, match="same dense authority"):
        published_subject_mask_cache_extension_manifest((first, second))


def test_cache_receipt_path_must_match_cache_kind() -> None:
    with pytest.raises(ValueError, match="does not match cache kind"):
        replace(
            _receipt(),
            cache_kind=SubjectMaskDerivedCacheKind.FULL_CONTOURS,
            cache_path="mask_bitpacked",
        )


def test_bitpacked_cache_contract_uses_packed_width_not_roi_width() -> None:
    arrays = {"masks_packed": np.zeros((3, 2, 4, 1), dtype=np.uint8)}
    assert (
        validate_subject_mask_cache_arrays(
            SubjectMaskDerivedCacheKind.BITPACKED,
            arrays,
            dimensions=_dimensions(),
        )
        == ()
    )

    arrays["masks_packed"] = np.zeros((3, 2, 4, 5), dtype=np.uint8)
    issues = validate_subject_mask_cache_arrays(
        SubjectMaskDerivedCacheKind.BITPACKED,
        arrays,
        dimensions=_dimensions(),
    )
    assert {issue.code for issue in issues} == {"array_contract_violation"}


def test_rle_cache_contract_validates_component_arrays() -> None:
    masks = np.zeros((3, 4, 5), dtype=np.uint8)
    masks[0, 1:3, 1:4] = 1
    masks[2, :, 4] = 1
    encoded = encode_mask_component_rle(
        masks,
        component_name="body",
        component_index=0,
    )
    arrays = {
        "counts": encoded.counts,
        "indptr": encoded.indptr,
        "present": encoded.present,
        "area_px": encoded.area_px,
        "bbox_xyxy": encoded.bbox_xyxy,
    }

    assert (
        validate_subject_mask_cache_arrays(
            SubjectMaskDerivedCacheKind.RLE,
            arrays,
            dimensions=_dimensions(),
        )
        == ()
    )
    arrays["present"] = np.zeros(3, dtype=bool)
    issues = validate_subject_mask_cache_arrays(
        SubjectMaskDerivedCacheKind.RLE,
        arrays,
        dimensions=_dimensions(),
    )
    assert "rle_presence_mismatch" in {issue.code for issue in issues}


def test_sampled_contour_cache_requires_canonical_nan_invalid_rows() -> None:
    points = np.full((3, 4, 2), np.nan, dtype=np.float32)
    points[0] = np.asarray(
        [[0, 0], [1, 0], [1, 1], [0, 1]],
        dtype=np.float32,
    )
    arrays = {
        "points_xy": points,
        "valid": np.asarray([True, False, False], dtype=bool),
        "source_point_count": np.asarray([12, 0, 1], dtype=np.int32),
    }
    assert (
        validate_subject_mask_cache_arrays(
            SubjectMaskDerivedCacheKind.SAMPLED_CONTOURS,
            arrays,
            dimensions=_dimensions(),
            sample_count=4,
        )
        == ()
    )

    arrays["points_xy"][1] = 0
    issues = validate_subject_mask_cache_arrays(
        SubjectMaskDerivedCacheKind.SAMPLED_CONTOURS,
        arrays,
        dimensions=_dimensions(),
        sample_count=4,
    )
    assert "noncanonical_invalid_samples" in {issue.code for issue in issues}


def test_full_contour_cache_rejects_orphan_append_history() -> None:
    arrays = {
        "ptr": np.asarray([0, -1, 3], dtype=np.int64),
        "len": np.asarray([3, 0, 2], dtype=np.int32),
        "points_xy": np.asarray(
            [[0, 0], [1, 0], [0, 1], [2, 2], [3, 2]],
            dtype=np.float32,
        ),
    }
    assert (
        validate_subject_mask_cache_arrays(
            SubjectMaskDerivedCacheKind.FULL_CONTOURS,
            arrays,
            dimensions=_dimensions(),
        )
        == ()
    )

    arrays["points_xy"] = np.concatenate(
        (arrays["points_xy"], np.asarray([[99, 99]], dtype=np.float32)),
        axis=0,
    )
    issues = validate_subject_mask_cache_arrays(
        SubjectMaskDerivedCacheKind.FULL_CONTOURS,
        arrays,
        dimensions=_dimensions(),
    )
    assert "orphan_contour_points" in {issue.code for issue in issues}
