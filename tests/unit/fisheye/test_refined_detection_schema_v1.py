from __future__ import annotations

import json

import numpy as np
import pytest

from fisheye.shared.zarr.detection_schema import derive_canonical_detection_geometry
from fisheye.shared.zarr.refined_detection_schema import (
    REFINED_DETECTION_SCHEMA_V1,
    SOURCE_DECISION_CODE_MAP,
    SOURCE_KIND_CODE_MAP,
    RefinedDetectionDimensions,
    RefinedDetectionLineageProfile,
    RefinedDetectionSchemaError,
)


def _p(group: str, name: str) -> str:
    return f"{group}/{name}"


def _payload(
    *,
    clipped: bool = False,
) -> tuple[RefinedDetectionDimensions, dict[str, np.ndarray]]:
    dimensions = RefinedDetectionDimensions(
        n_frames=4,
        n_instances=3,
        n_source_detections=3,
        source_width=640,
        source_height=480,
        lineage_profile=(
            RefinedDetectionLineageProfile.CLIPPED_RECORDING_SNAPSHOT
            if clipped
            else RefinedDetectionLineageProfile.FULL_ACQUISITION
        ),
    )
    source_bbox = np.asarray(
        [
            [0.50, 0.50, 0.20, 0.20],
            [0.25, 0.30, 0.10, 0.20],
            [0.75, 0.60, 0.20, 0.10],
        ],
        dtype=np.float32,
    )
    source_bbox_img, source_centers = derive_canonical_detection_geometry(
        source_bbox,
        source_width=dimensions.source_width,
        source_height=dimensions.source_height,
    )
    instance_bbox = np.asarray(
        [
            source_bbox[0],
            [0.35, 0.40, 0.12, 0.16],
            source_bbox[2],
        ],
        dtype=np.float32,
    )
    instance_bbox_img, instance_centers = derive_canonical_detection_geometry(
        instance_bbox,
        source_width=dimensions.source_width,
        source_height=dimensions.source_height,
    )
    arrays = {
        _p("instances", "frame_indices"): np.asarray([0, 0, 3], dtype=np.int32),
        _p("instances", "source_acquisition_frame_index"): np.asarray(
            [0, 0, 3], dtype=np.int64
        ),
        _p("instances", "instance_key"): np.asarray(
            [100, 200, 102], dtype=np.uint64
        ),
        _p("instances", "refined_row_ids"): np.asarray(
            [10, 12, 11], dtype=np.int64
        ),
        _p("instances", "bbox_norm_coords"): instance_bbox,
        _p("instances", "bbox_img_xyxy"): instance_bbox_img,
        _p("instances", "centers_img_xy"): instance_centers,
        _p("instances", "scores"): np.asarray(
            [0.9, 0.0, 0.7], dtype=np.float32
        ),
        _p("instances", "score_valid"): np.asarray(
            [True, False, True], dtype=bool
        ),
        _p("instances", "class_ids"): np.asarray([1, 2, 1], dtype=np.int32),
        _p("instances", "source_kind_codes"): np.asarray(
            [
                SOURCE_KIND_CODE_MAP["raw_detect"],
                SOURCE_KIND_CODE_MAP["manual"],
                SOURCE_KIND_CODE_MAP["raw_detect"],
            ],
            dtype=np.uint8,
        ),
        _p("instances", "manual_edit_flags"): np.asarray(
            [False, True, False], dtype=bool
        ),
        _p("instances", "source_detect_row_index"): np.asarray(
            [0, -1, 2], dtype=np.int64
        ),
        _p("instances", "reason_codes"): np.asarray([0, 1, 0], dtype=np.uint16),
        _p("instances", "frame_row_offsets"): np.asarray(
            [0, 2, 2, 2, 3], dtype=np.int64
        ),
        _p("source_detections", "source_detect_row_index"): np.arange(
            3, dtype=np.int64
        ),
        _p("source_detections", "frame_indices"): np.asarray(
            [0, 1, 3], dtype=np.int32
        ),
        _p("source_detections", "source_acquisition_frame_index"): np.asarray(
            [0, 1, 3], dtype=np.int64
        ),
        _p("source_detections", "instance_key"): np.asarray(
            [100, 101, 102], dtype=np.uint64
        ),
        _p("source_detections", "bbox_norm_coords"): source_bbox,
        _p("source_detections", "bbox_img_xyxy"): source_bbox_img,
        _p("source_detections", "centers_img_xy"): source_centers,
        _p("source_detections", "scores"): np.asarray(
            [0.9, 0.8, 0.7], dtype=np.float32
        ),
        _p("source_detections", "class_ids"): np.asarray(
            [1, 3, 1], dtype=np.int32
        ),
        _p("source_detections", "decision_codes"): np.asarray(
            [
                SOURCE_DECISION_CODE_MAP["accepted"],
                SOURCE_DECISION_CODE_MAP["filtered"],
                SOURCE_DECISION_CODE_MAP["accepted"],
            ],
            dtype=np.uint8,
        ),
        _p("source_detections", "resolved_refined_row_id"): np.asarray(
            [10, -1, 11], dtype=np.int64
        ),
        _p("source_detections", "reason_codes"): np.asarray(
            [0, 2, 0], dtype=np.uint16
        ),
        _p("source_detections", "frame_row_offsets"): np.asarray(
            [0, 1, 2, 2, 3], dtype=np.int64
        ),
    }
    if clipped:
        arrays.update(
            {
                _p("instances", "source_recording_frame_ids"): np.asarray(
                    [1, 1, 4], dtype=np.int64
                ),
                _p("instances", "source_clip_indices"): np.asarray(
                    [0, 0, 1], dtype=np.int32
                ),
                _p("instances", "source_clip_local_frame_indices"): np.asarray(
                    [0, 0, 1], dtype=np.int32
                ),
                _p("instances", "source_clip_detect_row_index"): np.asarray(
                    [0, -1, 0], dtype=np.int64
                ),
                _p("instances", "source_refined_row_ids"): np.asarray(
                    [0, 5, 0], dtype=np.int64
                ),
                _p("source_detections", "source_recording_frame_ids"): np.asarray(
                    [1, 2, 4], dtype=np.int64
                ),
                _p("source_detections", "source_clip_indices"): np.asarray(
                    [0, 0, 1], dtype=np.int32
                ),
                _p(
                    "source_detections", "source_clip_local_frame_indices"
                ): np.asarray([0, 1, 1], dtype=np.int32),
                _p("source_detections", "source_clip_detect_row_index"): np.asarray(
                    [0, 1, 0], dtype=np.int64
                ),
                _p(
                    "source_detections", "source_resolved_refined_row_id"
                ): np.asarray([0, -1, 0], dtype=np.int64),
            }
        )
    return dimensions, arrays


def _codes(
    dimensions: RefinedDetectionDimensions,
    arrays: dict[str, np.ndarray],
) -> set[str]:
    return {
        issue.code
        for issue in REFINED_DETECTION_SCHEMA_V1.validate(
            arrays,
            dimensions=dimensions,
        )
    }


def test_full_acquisition_snapshot_accepts_zero_one_or_many_instances() -> None:
    dimensions, arrays = _payload()

    assert REFINED_DETECTION_SCHEMA_V1.validate(arrays, dimensions=dimensions) == ()
    REFINED_DETECTION_SCHEMA_V1.require(arrays, dimensions=dimensions)
    assert tuple(np.diff(arrays[_p("instances", "frame_row_offsets")])) == (
        2,
        0,
        0,
        1,
    )


def test_clipped_profile_requires_and_accepts_exact_lineage_extension() -> None:
    dimensions, arrays = _payload(clipped=True)
    assert REFINED_DETECTION_SCHEMA_V1.validate(arrays, dimensions=dimensions) == ()
    assert len(REFINED_DETECTION_SCHEMA_V1.binding_paths_for(dimensions)) == 38

    del arrays[_p("instances", "source_clip_indices")]
    assert "missing_required_array" in _codes(dimensions, arrays)


def test_manual_rows_use_explicit_source_and_score_validity_encoding() -> None:
    dimensions, arrays = _payload()
    arrays[_p("instances", "source_detect_row_index")][1] = 1
    arrays[_p("instances", "score_valid")][1] = True

    codes = _codes(dimensions, arrays)
    assert "manual_source_row_mismatch" in codes
    assert "manual_row_semantics_invalid" in codes


def test_manual_instance_keys_cannot_collide_with_source_candidate_keys() -> None:
    dimensions, arrays = _payload()
    arrays[_p("instances", "instance_key")][1] = np.uint64(101)

    assert "manual_instance_key_collision" in _codes(dimensions, arrays)


def test_raw_backed_rows_must_join_exact_source_identity_and_resolution() -> None:
    dimensions, arrays = _payload()
    arrays[_p("instances", "instance_key")][2] = np.uint64(999)
    arrays[_p("source_detections", "resolved_refined_row_id")][2] = np.int64(10)

    codes = _codes(dimensions, arrays)
    assert "raw_source_join_mismatch" in codes
    assert "raw_resolution_join_mismatch" in codes


def test_raw_bbox_and_class_corrections_require_manual_edit_flag() -> None:
    dimensions, arrays = _payload()
    arrays[_p("instances", "class_ids")][0] = np.int32(7)

    assert "unedited_raw_value_mismatch" in _codes(dimensions, arrays)

    arrays[_p("instances", "manual_edit_flags")][0] = True
    assert REFINED_DETECTION_SCHEMA_V1.validate(arrays, dimensions=dimensions) == ()


def test_accepted_source_rows_are_exactly_the_raw_backed_rowset() -> None:
    dimensions, arrays = _payload()
    arrays[_p("source_detections", "decision_codes")][1] = np.uint8(
        SOURCE_DECISION_CODE_MAP["accepted"]
    )
    arrays[_p("source_detections", "resolved_refined_row_id")][1] = np.int64(12)

    assert "accepted_source_rowset_mismatch" in _codes(dimensions, arrays)


def test_recording_level_frame_identities_join_exactly() -> None:
    dimensions, arrays = _payload()
    arrays[_p("instances", "source_acquisition_frame_index")][2] = np.int64(2)

    assert "acquisition_frame_join_mismatch" in _codes(dimensions, arrays)

    clipped_dimensions, clipped_arrays = _payload(clipped=True)
    clipped_arrays[_p("source_detections", "source_recording_frame_ids")][2] = 5
    assert "recording_frame_join_mismatch" in _codes(
        clipped_dimensions,
        clipped_arrays,
    )


def test_source_decisions_and_offsets_fail_closed() -> None:
    dimensions, arrays = _payload()
    arrays[_p("source_detections", "decision_codes")][1] = np.uint8(99)
    arrays[_p("source_detections", "resolved_refined_row_id")][1] = np.int64(12)
    arrays[_p("instances", "frame_row_offsets")] = np.asarray(
        [0, 1, 2, 2, 3], dtype=np.int64
    )

    codes = _codes(dimensions, arrays)
    assert "unknown_source_decision_code" in codes
    assert "unaccepted_source_has_resolution" in codes
    assert "frame_row_offsets_mismatch" in codes


def test_exact_dtypes_and_exact_canonical_array_set_are_required() -> None:
    dimensions, arrays = _payload()
    arrays[_p("instances", "bbox_norm_coords")] = arrays[
        _p("instances", "bbox_norm_coords")
    ].astype(np.float64)
    arrays[_p("instances", "confidence_scores")] = np.zeros(3, dtype=np.float32)

    codes = _codes(dimensions, arrays)
    assert "array_contract_violation" in codes
    assert "forbidden_legacy_binding" in codes
    assert "unexpected_array" in codes
    with pytest.raises(RefinedDetectionSchemaError):
        REFINED_DETECTION_SCHEMA_V1.require(arrays, dimensions=dimensions)


def test_manifest_freezes_identity_sentinels_dtypes_and_lifecycle() -> None:
    dimensions, _arrays = _payload()
    manifest = REFINED_DETECTION_SCHEMA_V1.as_manifest(dimensions=dimensions)

    assert json.loads(json.dumps(manifest)) == manifest
    assert manifest["schema_id"] == "palette.stage.refined_detection"
    assert manifest["schema_version"] == 1
    assert manifest["dimensions"]["lineage_profile"] == "full_acquisition"
    assert manifest["invariants"]["artifact_mutability"] == "immutable_snapshot"
    assert manifest["invariants"]["frame_counts"] == "derived_not_persisted"
    assert manifest["invariants"]["manual_source_row_encoding"] == -1
    assert manifest["invariants"]["score_missing_encoding"] == {
        "score_valid": False,
        "scores": 0.0,
    }
    paths = {binding["path"] for binding in manifest["bindings"]}
    assert _p("instances", "frame_row_offsets") in paths
    assert _p("source_detections", "frame_row_offsets") in paths
    assert _p("instances", "frame_counts") not in paths
    assert _p("instances", "review_notes") not in paths
