from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from fisheye.shared.coordinate_descriptor import (
    CANONICAL_OVERLAY_DIRECT,
    CanonicalFrameRecord,
    DigestBoundCoordinateRecordRef,
    PIXEL_FRAME_AUTHORITY_RECORD_KIND,
    build_canonical_coordinate_descriptor,
)
from fisheye.shared.coordinate_identity import (
    OBSERVATION_INSTANCE_DOMAIN,
    build_row_identity_contract,
)
from fisheye.shared.coordinate_surface_contract import (
    SOURCE_CAMERA_NORMALIZED_POINT_XY,
    SOURCE_CAMERA_POINT_XY,
)
from fisheye.shared.subject_position_storage import (
    SubjectPositionStorageValidationError,
    canonical_observation_position_arrays_sha256,
    canonical_observation_position_logical_metadata,
    canonical_observation_position_schema_descriptor,
    canonical_position_failure_reason_map,
    canonical_position_failure_reason_precedence,
    canonical_source_camera_coordinate_metadata,
    canonical_track_sample_position_schema_descriptor,
    collect_observation_position_storage_issues,
    observation_position_schema_digest,
    position_failure_reason_map_digest,
    position_failure_reason_precedence_digest,
    track_sample_position_schema_digest,
    validate_observation_position_arrays,
)
from fisheye.shared.subject_position_types import (
    CANONICAL_FLOAT32_QNAN_BITS,
    POSITION_FAILURE_REASON_CODES,
    POSITION_FAILURE_REASON_PRECEDENCE,
    empty_position_xy,
)

FRAME_SHA256 = "a" * 64


def _record(token: str = "a") -> DigestBoundCoordinateRecordRef:
    return DigestBoundCoordinateRecordRef(
        record_ref="/coordinate_frames/source_camera@pixel_frame_authority",
        record_sha256=token * 64,
    )


def _descriptor(
    *,
    row_count: int = 3,
    width_px: int = 640,
    height_px: int = 480,
    surface=SOURCE_CAMERA_POINT_XY,
    instance_keys: np.ndarray | None = None,
):
    if instance_keys is None:
        instance_keys = np.arange(100, 100 + row_count, dtype=np.uint64)
    identity = build_row_identity_contract(
        domain=OBSERVATION_INSTANCE_DOMAIN,
        values=instance_keys,
    )
    frame = _record()
    kwargs = surface.descriptor_kwargs()
    transform_refs = ()
    if kwargs["source_camera_overlay_status"] != CANONICAL_OVERLAY_DIRECT:
        transform_refs = (
            DigestBoundCoordinateRecordRef(
                record_ref="/coordinate_transforms/to_source_camera",
                record_sha256="b" * 64,
            ),
        )
    return build_canonical_coordinate_descriptor(
        **kwargs,
        reference_width=width_px,
        reference_height=height_px,
        reference_authority=frame,
        reference_selector="record",
        row_identity_contract=identity,
        row_identity_record_ref=(
            "/analysis/subject_position_source@row_identity_contract"
        ),
        overlay_transform_refs=transform_refs,
        frame_record=CanonicalFrameRecord(
            kind=PIXEL_FRAME_AUTHORITY_RECORD_KIND,
            record_ref=frame.record_ref,
            record_sha256=frame.record_sha256,
        ),
    )


def _metadata(*, row_count: int = 3, width_px: int = 640, height_px: int = 480):
    descriptor = _descriptor(
        row_count=row_count,
        width_px=width_px,
        height_px=height_px,
    )
    coordinate = canonical_source_camera_coordinate_metadata(descriptor)
    return coordinate, canonical_observation_position_logical_metadata(coordinate)


def _arrays(row_count: int = 3) -> dict[str, np.ndarray]:
    positions = empty_position_xy(row_count)
    valid = np.zeros(row_count, dtype=np.bool_)
    reasons = np.full(
        row_count,
        POSITION_FAILURE_REASON_CODES["required_anchor_invalid"],
        dtype=np.uint16,
    )
    if row_count:
        positions[0] = (12.5, 24.25)
        valid[0] = True
        reasons[0] = POSITION_FAILURE_REASON_CODES["ok"]
    return {
        "position_xy": positions,
        "valid": valid,
        "failure_reason_codes": reasons,
        "instance_key": np.arange(100, 100 + row_count, dtype=np.uint64),
        "source_acquisition_frame_index": np.arange(row_count, dtype=np.int64),
        "source_row_index": np.arange(row_count, dtype=np.int64),
    }


def _assert_invalid(arrays, coordinate, manifest, pattern: str) -> None:
    with pytest.raises(SubjectPositionStorageValidationError, match=pattern):
        validate_observation_position_arrays(
            arrays,
            coordinate_metadata=coordinate,
            manifest_metadata=manifest,
        )


def test_canonical_descriptors_and_digests_are_stable() -> None:
    observation = canonical_observation_position_schema_descriptor()
    track_sample = canonical_track_sample_position_schema_descriptor()
    assert observation["row_axis"] == "observation_instance"
    assert track_sample["row_axis"] == "track_sample"
    assert observation_position_schema_digest() == observation_position_schema_digest()
    assert track_sample_position_schema_digest() != observation_position_schema_digest()
    assert position_failure_reason_map_digest() == position_failure_reason_map_digest()
    assert canonical_position_failure_reason_map()["codes"]["ok"] == 0
    assert canonical_position_failure_reason_precedence()["reason_tags"] == list(
        POSITION_FAILURE_REASON_PRECEDENCE
    )


def test_valid_mixed_rows_pass_and_report_exact_metadata() -> None:
    arrays = _arrays()
    coordinate, manifest = _metadata()
    report = validate_observation_position_arrays(
        arrays,
        coordinate_metadata=coordinate,
        manifest_metadata=manifest,
    )
    assert report.row_count == 3
    assert report.support_point_count is None
    assert report.storage_schema_sha256 == observation_position_schema_digest()
    assert (
        report.coordinate_descriptor_sha256
        == coordinate["coordinate_descriptor_sha256"]
    )
    assert report.reason_code_map_sha256 == position_failure_reason_map_digest()
    assert (
        report.reason_precedence_sha256 == position_failure_reason_precedence_digest()
    )


def test_zero_row_publication_passes_with_exact_shapes() -> None:
    arrays = _arrays(0)
    coordinate, manifest = _metadata(row_count=0)
    report = validate_observation_position_arrays(
        arrays,
        coordinate_metadata=coordinate,
        manifest_metadata=manifest,
    )
    assert report.row_count == 0
    assert arrays["position_xy"].shape == (0, 2)


def test_optional_support_arrays_require_shared_p_and_canonical_invalid_points() -> (
    None
):
    arrays = _arrays()
    support_xy = empty_position_xy(6).reshape(3, 2, 2)
    support_xy[0, 0] = (10.0, 20.0)
    support_xy[0, 1] = (11.0, 21.0)
    support_xy[1, 0] = (30.0, 40.0)
    support_valid = np.array([[True, True], [True, False], [False, False]], dtype=bool)
    arrays.update(
        {
            "support/source_points_xy": support_xy,
            "support/source_points_valid": support_valid,
            "support/source_point_reason_codes": np.array(
                [[0, 0], [0, 4], [4, 4]], dtype=np.uint16
            ),
            "support/source_point_confidence": np.array(
                [[0.9, 0.8], [0.7, 0.0], [0.0, 0.0]], dtype=np.float32
            ),
        }
    )
    coordinate, manifest = _metadata()
    report = validate_observation_position_arrays(
        arrays,
        coordinate_metadata=coordinate,
        manifest_metadata=manifest,
    )
    assert report.support_point_count == 2


@pytest.mark.parametrize(
    ("path", "replacement", "pattern"),
    [
        ("position_xy", np.zeros((3, 2), dtype=np.float64), "array_dtype_mismatch"),
        ("valid", np.zeros(3, dtype=np.uint8), "array_dtype_mismatch"),
        ("failure_reason_codes", np.zeros(3, dtype=np.int64), "array_dtype_mismatch"),
        ("instance_key", np.zeros(3, dtype=np.int64), "array_dtype_mismatch"),
        (
            "source_acquisition_frame_index",
            np.zeros(3, dtype=np.int32),
            "array_dtype_mismatch",
        ),
        ("source_row_index", np.zeros((3, 1), dtype=np.int64), "array_shape_mismatch"),
        ("position_xy", np.zeros((2, 3), dtype=np.float32), "array_shape_mismatch"),
    ],
)
def test_wrong_dtype_or_rank_is_rejected(path, replacement, pattern) -> None:
    arrays = _arrays()
    arrays[path] = replacement
    coordinate, manifest = _metadata()
    _assert_invalid(arrays, coordinate, manifest, pattern)


def test_duplicate_instance_keys_are_rejected() -> None:
    arrays = _arrays()
    arrays["instance_key"][1] = arrays["instance_key"][0]
    coordinate, manifest = _metadata()
    _assert_invalid(arrays, coordinate, manifest, "duplicate_instance_key")


def test_descriptor_row_identity_must_match_exact_ordered_instance_keys() -> None:
    arrays = _arrays()
    descriptor = _descriptor(instance_keys=np.array([101, 100, 102], dtype=np.uint64))
    coordinate = canonical_source_camera_coordinate_metadata(descriptor)
    manifest = canonical_observation_position_logical_metadata(coordinate)
    _assert_invalid(
        arrays,
        coordinate,
        manifest,
        "coordinate_descriptor_row_identity_record_digest_mismatch",
    )


def test_duplicate_source_row_indices_are_rejected() -> None:
    arrays = _arrays()
    arrays["source_row_index"][2] = arrays["source_row_index"][1]
    coordinate, manifest = _metadata()
    _assert_invalid(arrays, coordinate, manifest, "duplicate_source_row_index")


def test_negative_frame_and_source_row_indices_are_rejected() -> None:
    arrays = _arrays()
    arrays["source_acquisition_frame_index"][1] = -1
    arrays["source_row_index"][2] = -2
    coordinate, manifest = _metadata()
    _assert_invalid(arrays, coordinate, manifest, "negative_index")


def test_unknown_reason_code_is_rejected() -> None:
    arrays = _arrays()
    arrays["failure_reason_codes"][1] = np.uint16(65535)
    coordinate, manifest = _metadata()
    _assert_invalid(arrays, coordinate, manifest, "unknown_reason_code")


def test_partial_nan_is_rejected() -> None:
    arrays = _arrays()
    arrays["valid"][0] = False
    arrays["failure_reason_codes"][0] = POSITION_FAILURE_REASON_CODES[
        "required_anchor_invalid"
    ]
    arrays["position_xy"][0, 0] = np.float32(np.nan)
    arrays["position_xy"][0, 1] = np.float32(5.0)
    coordinate, manifest = _metadata()
    _assert_invalid(arrays, coordinate, manifest, "invalidity_payload_mismatch")


def test_wrong_nan_payload_is_rejected() -> None:
    arrays = _arrays()
    bits = arrays["position_xy"].view(np.uint32)
    bits[1, :] = np.uint32(int(CANONICAL_FLOAT32_QNAN_BITS) + 1)
    coordinate, manifest = _metadata()
    _assert_invalid(arrays, coordinate, manifest, "invalidity_payload_mismatch")


def test_infinity_and_validity_reason_mismatch_are_rejected() -> None:
    arrays = _arrays()
    arrays["position_xy"][0, 0] = np.float32(np.inf)
    coordinate, manifest = _metadata()
    _assert_invalid(arrays, coordinate, manifest, "validity_payload_mismatch")


def test_support_shape_mismatch_is_rejected() -> None:
    arrays = _arrays()
    arrays["support/source_points_xy"] = np.zeros((3, 2, 3), dtype=np.float32)
    arrays["support/source_points_valid"] = np.zeros((3, 2), dtype=bool)
    coordinate, manifest = _metadata()
    _assert_invalid(arrays, coordinate, manifest, "support_shape_mismatch")


def test_support_invalid_coordinate_requires_exact_canonical_nan() -> None:
    arrays = _arrays()
    support_xy = empty_position_xy(6).reshape(3, 2, 2)
    support_valid = np.zeros((3, 2), dtype=bool)
    support_xy[1, 0, 0] = np.float32(1.0)
    arrays["support/source_points_xy"] = support_xy
    arrays["support/source_points_valid"] = support_valid
    coordinate, manifest = _metadata()
    _assert_invalid(
        arrays, coordinate, manifest, "support_invalid_coordinate_not_canonical_nan"
    )


def test_optional_support_arrays_need_coordinate_validity_pair() -> None:
    arrays = _arrays()
    arrays["support/source_points_xy"] = np.zeros((3, 1, 2), dtype=np.float32)
    coordinate, manifest = _metadata()
    _assert_invalid(arrays, coordinate, manifest, "support_pair_missing")


def test_stale_schema_digest_is_rejected() -> None:
    arrays = _arrays()
    coordinate, manifest = _metadata()
    manifest["storage_schema_sha256"] = "b" * 64
    _assert_invalid(arrays, coordinate, manifest, "storage_schema_digest_mismatch")


def test_stale_coordinate_descriptor_digest_is_rejected() -> None:
    arrays = _arrays()
    coordinate, manifest = _metadata()
    coordinate["coordinate_descriptor_sha256"] = "d" * 64
    _assert_invalid(
        arrays, coordinate, manifest, "coordinate_descriptor_digest_mismatch"
    )


def test_legitimate_normalized_v2_descriptor_is_rejected_for_position_storage() -> None:
    arrays = _arrays()
    descriptor = _descriptor(surface=SOURCE_CAMERA_NORMALIZED_POINT_XY)
    coordinate = canonical_source_camera_coordinate_metadata(descriptor)
    manifest = canonical_observation_position_logical_metadata(coordinate)
    _assert_invalid(
        arrays, coordinate, manifest, "coordinate_surface_contract_mismatch"
    )


def test_wrong_extent_or_frame_binding_is_rejected() -> None:
    arrays = _arrays()
    coordinate, manifest = _metadata()
    descriptor = deepcopy(coordinate["coordinate_descriptor"])
    descriptor["frame_record"]["record_sha256"] = "e" * 64
    coordinate = {
        "coordinate_descriptor": descriptor,
        "coordinate_descriptor_sha256": coordinate["coordinate_descriptor_sha256"],
    }
    _assert_invalid(
        arrays,
        coordinate,
        manifest,
        "coordinate_descriptor_frame_authority_mismatch",
    )


def test_noninteger_source_camera_extent_is_rejected_by_canonical_v2_parser() -> None:
    arrays = _arrays()
    coordinate, manifest = _metadata()
    descriptor = deepcopy(coordinate["coordinate_descriptor"])
    descriptor["reference_extent"]["width"] = 640.5
    coordinate = {
        "coordinate_descriptor": descriptor,
        "coordinate_descriptor_sha256": coordinate["coordinate_descriptor_sha256"],
    }
    _assert_invalid(
        arrays,
        coordinate,
        manifest,
        "coordinate_descriptor_pixel_reference_extent_not_integer",
    )


def test_reason_code_map_digest_is_bound_and_stale_maps_fail_closed() -> None:
    arrays = _arrays()
    coordinate, manifest = _metadata()
    manifest["reason_code_map_sha256"] = "c" * 64
    _assert_invalid(arrays, coordinate, manifest, "reason_code_map_digest_mismatch")
    manifest = canonical_observation_position_logical_metadata(coordinate)
    manifest["reason_code_map"]["codes"]["ok"] = 99
    _assert_invalid(arrays, coordinate, manifest, "reason_code_map_mismatch")


def test_reason_precedence_and_digest_are_bound_exactly() -> None:
    arrays = _arrays()
    coordinate, manifest = _metadata()
    manifest["reason_precedence_sha256"] = "c" * 64
    _assert_invalid(arrays, coordinate, manifest, "reason_precedence_digest_mismatch")

    manifest = canonical_observation_position_logical_metadata(coordinate)
    manifest["reason_precedence"]["reason_tags"] = list(
        reversed(POSITION_FAILURE_REASON_PRECEDENCE)
    )
    _assert_invalid(arrays, coordinate, manifest, "reason_precedence_mismatch")


def test_track_sample_descriptor_is_defined_but_observation_validator_rejects_it() -> (
    None
):
    arrays = _arrays()
    arrays["track_sample_key"] = np.zeros((3, 2), dtype=np.int64)
    coordinate, manifest = _metadata()
    manifest["row_axis"] = "track_sample"
    issues = collect_observation_position_storage_issues(
        arrays,
        coordinate_metadata=coordinate,
        manifest_metadata=manifest,
    )
    assert any(issue.code == "unexpected_array" for issue in issues)
    assert any(issue.code == "row_axis_mismatch" for issue in issues)


def test_array_digest_preserves_dtype_shape_and_exact_nan_payload() -> None:
    arrays = _arrays()
    first = canonical_observation_position_arrays_sha256(arrays)
    changed = dict(arrays)
    changed["position_xy"] = arrays["position_xy"].copy()
    changed["position_xy"].view(np.uint32)[1, :] = np.uint32(
        int(CANONICAL_FLOAT32_QNAN_BITS) + 1
    )
    second = canonical_observation_position_arrays_sha256(changed)
    assert first != second


def test_array_digest_rejects_wrong_dtype_shape_n_and_support_p() -> None:
    arrays = _arrays()

    wrong_dtype = dict(arrays)
    wrong_dtype["valid"] = arrays["valid"].astype(np.uint8)
    with pytest.raises(ValueError, match="expected exact dtype bool"):
        canonical_observation_position_arrays_sha256(wrong_dtype)

    wrong_shape = dict(arrays)
    wrong_shape["position_xy"] = np.zeros((3, 2, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="expected rank 2"):
        canonical_observation_position_arrays_sha256(wrong_shape)

    wrong_n = dict(arrays)
    wrong_n["source_row_index"] = np.arange(2, dtype=np.int64)
    with pytest.raises(ValueError, match="leading dimension N=3"):
        canonical_observation_position_arrays_sha256(wrong_n)

    wrong_p = dict(arrays)
    wrong_p["support/source_points_xy"] = np.zeros((3, 2, 2), dtype=np.float32)
    wrong_p["support/source_points_valid"] = np.zeros((3, 3), dtype=np.bool_)
    with pytest.raises(ValueError, match="expected shape \\(3, 2\\)"):
        canonical_observation_position_arrays_sha256(wrong_p)


def test_array_digest_requires_complete_known_support_pair() -> None:
    arrays = _arrays()

    missing = dict(arrays)
    del missing["source_row_index"]
    with pytest.raises(ValueError, match="missing mandatory arrays"):
        canonical_observation_position_arrays_sha256(missing)

    unknown = dict(arrays)
    unknown["track_sample_key"] = np.zeros((3, 2), dtype=np.int64)
    with pytest.raises(ValueError, match="unknown observation-position arrays"):
        canonical_observation_position_arrays_sha256(unknown)

    incomplete_support = dict(arrays)
    incomplete_support["support/source_points_xy"] = np.zeros(
        (3, 2, 2), dtype=np.float32
    )
    with pytest.raises(ValueError, match="both present or both absent"):
        canonical_observation_position_arrays_sha256(incomplete_support)

    orphan_confidence = dict(arrays)
    orphan_confidence["support/source_point_confidence"] = np.ones(
        (3, 2), dtype=np.float32
    )
    with pytest.raises(ValueError, match="require the complete support"):
        canonical_observation_position_arrays_sha256(orphan_confidence)
