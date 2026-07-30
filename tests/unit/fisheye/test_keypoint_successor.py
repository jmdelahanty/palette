from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.shared.zarr.crop_manifest import CropPixelAuthority
from fisheye.shared.zarr.crop_shadow import (
    prepare_crop_geometry_from_refined_source,
    publish_selector_ineligible_crop_geometry_snapshot,
)
from fisheye.shared.zarr.crop_successor import (
    publish_selector_ineligible_crop_geometry_successor,
)
from fisheye.shared.zarr.keypoint_manifest import (
    KeypointPreprocessingReference,
    keypoint_skeleton_digest,
)
from fisheye.shared.zarr.keypoint_publication import (
    prepare_raw_keypoint_v2_snapshot,
    publish_selector_ineligible_keypoint_snapshot,
    validate_keypoint_shadow_publication,
)
from fisheye.shared.zarr.keypoint_schema import (
    KeypointDimensions,
    derive_keypoint_row_signatures,
)
from fisheye.shared.zarr.keypoint_successor import (
    RAW_KEYPOINT_SUCCESSOR_PUBLICATION_RECEIPT_NAME,
    RAW_KEYPOINT_SUCCESSOR_SCHEMA_ID,
    RawKeypointSuccessorError,
    TerminalKeypointInferenceBatch,
    prepare_raw_keypoint_successor,
    publish_selector_ineligible_raw_keypoint_successor,
    validate_raw_keypoint_successor_publication_receipt,
)
from fisheye.shared.zarr.refined_detection_compaction import (
    compact_frozen_refined_detection_delta_generation,
)
from fisheye.shared.zarr.refined_detection_crop_source import (
    bind_refined_detection_crop_source,
)
from fisheye.shared.zarr.refined_detection_delta import (
    REFINED_DETECTION_DELTA_ARRAYS,
    REFINED_DETECTION_DELTA_OPERATION_CODE_MAP,
    RefinedDetectionDeltaBatch,
)
from fisheye.shared.zarr.refined_detection_delta_storage import (
    RefinedDetectionDeltaLineageBinding,
    create_refined_detection_delta_lineage,
    freeze_refined_detection_delta_generation,
    write_refined_detection_delta_partition,
)
from fisheye.shared.zarr.refined_detection_schema import SOURCE_KIND_CODE_MAP
from tests.unit.fisheye.test_crop_shadow import _policy
from tests.unit.fisheye.test_keypoint_publication import _pose_binding
from tests.unit.fisheye.test_refined_detection_compaction import (
    BASE_SNAPSHOT_ID,
    CREATED_AT,
    DELTA_LINEAGE_ID,
    RECORDING_IDENTITY,
    SUCCESSOR_SNAPSHOT_ID,
    _base_publication,
    _frozen_delta,
)


def _preprocessing() -> KeypointPreprocessingReference:
    return KeypointPreprocessingReference(
        profile_id="yolo_pose_crop_v1",
        profile_version=1,
        input_mode="crop_pixel_package",
        document={
            "decoded_dtype": "uint8",
            "channels": "grayscale",
            "resize": "letterbox_model_shape",
            "normalization": "ultralytics_runtime_v1",
        },
    )


def _crop_successor(tmp_path: Path):  # type: ignore[no-untyped-def]
    base = _base_publication(tmp_path)
    compaction_root = tmp_path / "compactions"
    compacted = compact_frozen_refined_detection_delta_generation(
        delta_root=_frozen_delta(tmp_path, base),
        delta_lineage_id=DELTA_LINEAGE_ID,
        generation_ordinal=0,
        base_manifest=base.manifest,
        base_arrays=base.arrays,
        destination=compaction_root / "successor.zarr",
        run_id="refined_successor",
        snapshot_id=SUCCESSOR_SNAPSHOT_ID,
        created_by="keypoint_successor_test",
        safe_root=compaction_root,
    )
    parent_source = bind_refined_detection_crop_source(
        base.output_path,
        run_id="refined_base",
        allow_selector_ineligible_benchmark=True,
    )
    target_source = bind_refined_detection_crop_source(
        compacted.publication.output_path,
        run_id="refined_successor",
        allow_selector_ineligible_benchmark=True,
        parent_manifest=base.manifest,
        parent_arrays=base.arrays,
    )
    pixels = CropPixelAuthority(
        authority_id="test_camera_video#decode=uint8_v1",
        authority_manifest_digest="c" * 64,
        recording_identity=RECORDING_IDENTITY,
        camera_identity="cam_test",
        n_frames=4,
        source_width=100,
        source_height=80,
    )
    parent_prepared = prepare_crop_geometry_from_refined_source(
        parent_source,
        policy=_policy(),
        pixel_authority=pixels,
    )
    target_prepared = prepare_crop_geometry_from_refined_source(
        target_source,
        policy=_policy(),
        pixel_authority=pixels,
    )
    crop_root = tmp_path / "crops"
    parent_crop = publish_selector_ineligible_crop_geometry_snapshot(
        parent_prepared,
        destination=crop_root / "parent.zarr",
        run_id="crop_parent",
        shadow_root=crop_root,
        coordinate_catalog=True,
    )
    successor = publish_selector_ineligible_crop_geometry_successor(
        parent_crop,
        target_prepared,
        destination=crop_root / "successor.zarr",
        run_id="crop_successor",
        shadow_root=crop_root,
        created_by="keypoint_successor_test",
    )
    return parent_crop, successor


def _replacement_delete_delta(tmp_path: Path, base):  # type: ignore[no-untyped-def]
    base_key = int(base.arrays["instances/instance_key"][0])
    deleted_key = int(base.arrays["instances/instance_key"][1])
    events = [
        {
            "event_sequence": 1,
            "expected_previous_event_sequence": 0,
            "operation_codes": REFINED_DETECTION_DELTA_OPERATION_CODE_MAP[
                "replace_instance"
            ],
            "instance_key": base_key,
            "refined_row_ids": 0,
            "row_index_hint": 0,
            "timestamp_ns": 1,
            "reason_codes": 1,
            "payload_valid": True,
            "frame_indices": 0,
            "source_acquisition_frame_index": 0,
            "bbox_norm_coords": [0.25, 0.20, 0.10, 0.10],
            "scores": 0.9,
            "score_valid": True,
            "class_ids": 1,
            "source_kind_codes": SOURCE_KIND_CODE_MAP["raw_detect"],
            "manual_edit_flags": True,
            "source_detect_row_index": 0,
        },
        {
            "event_sequence": 2,
            "expected_previous_event_sequence": 0,
            "operation_codes": REFINED_DETECTION_DELTA_OPERATION_CODE_MAP[
                "delete_instance"
            ],
            "instance_key": deleted_key,
            "refined_row_ids": 1,
            "row_index_hint": 1,
            "timestamp_ns": 2,
            "reason_codes": 2,
            "payload_valid": False,
            "frame_indices": -1,
            "source_acquisition_frame_index": -1,
            "bbox_norm_coords": [0.0, 0.0, 0.0, 0.0],
            "scores": 0.0,
            "score_valid": False,
            "class_ids": -1,
            "source_kind_codes": 0,
            "manual_edit_flags": False,
            "source_detect_row_index": -1,
        },
    ]
    arrays = {
        declaration.name: np.asarray(
            [event[declaration.name] for event in events],
            dtype=np.dtype(declaration.dtype),
        ).reshape(len(events), *declaration.trailing_shape)
        for declaration in REFINED_DETECTION_DELTA_ARRAYS
    }
    batch = RefinedDetectionDeltaBatch(
        delta_lineage_id=DELTA_LINEAGE_ID,
        base_snapshot_id=BASE_SNAPSHOT_ID,
        base_manifest_digest=str(base.manifest["payload_digest"]),
        generation_ordinal=0,
        partition_id="partition_replace_delete",
        actor_id="reviewer@example.org",
        reason_code_map={0: "none", 1: "bbox_corrected", 2: "false_positive"},
        arrays=arrays,
    )
    root = zarr.open_group(
        str(tmp_path / "replace_delete_delta.zarr"),
        mode="w",
        zarr_format=3,
    )
    create_refined_detection_delta_lineage(
        root,
        binding=RefinedDetectionDeltaLineageBinding(
            delta_lineage_id=DELTA_LINEAGE_ID,
            base_run_path="refined_detect_runs/refined_base",
            base_snapshot_id=BASE_SNAPSHOT_ID,
            base_manifest_digest=str(base.manifest["payload_digest"]),
            base_logical_content_digest=str(
                base.receipt["logical_content_digest"]
            ),
            recording_identity=RECORDING_IDENTITY,
            base_next_refined_row_id=3,
        ),
        created_by="reviewer@example.org",
        created_at_utc=CREATED_AT,
    )
    write_refined_detection_delta_partition(
        root,
        batch=batch,
        created_at_utc=CREATED_AT,
    )
    freeze_refined_detection_delta_generation(
        root,
        delta_lineage_id=DELTA_LINEAGE_ID,
        generation_ordinal=0,
        frozen_by="compactor",
        frozen_at_utc="2026-07-27T13:00:00+00:00",
    )
    return root


def _changed_and_retired_crop_successor(
    tmp_path: Path,
):  # type: ignore[no-untyped-def]
    base = _base_publication(tmp_path)
    compaction_root = tmp_path / "replace_delete_compactions"
    compacted = compact_frozen_refined_detection_delta_generation(
        delta_root=_replacement_delete_delta(tmp_path, base),
        delta_lineage_id=DELTA_LINEAGE_ID,
        generation_ordinal=0,
        base_manifest=base.manifest,
        base_arrays=base.arrays,
        destination=compaction_root / "successor.zarr",
        run_id="refined_replace_delete_successor",
        snapshot_id=SUCCESSOR_SNAPSHOT_ID,
        created_by="keypoint_successor_test",
        safe_root=compaction_root,
    )
    parent_source = bind_refined_detection_crop_source(
        base.output_path,
        run_id="refined_base",
        allow_selector_ineligible_benchmark=True,
    )
    target_source = bind_refined_detection_crop_source(
        compacted.publication.output_path,
        run_id="refined_replace_delete_successor",
        allow_selector_ineligible_benchmark=True,
        parent_manifest=base.manifest,
        parent_arrays=base.arrays,
    )
    pixels = CropPixelAuthority(
        authority_id="test_camera_video#decode=uint8_v1",
        authority_manifest_digest="c" * 64,
        recording_identity=RECORDING_IDENTITY,
        camera_identity="cam_test",
        n_frames=4,
        source_width=100,
        source_height=80,
    )
    parent_prepared = prepare_crop_geometry_from_refined_source(
        parent_source,
        policy=_policy(),
        pixel_authority=pixels,
    )
    target_prepared = prepare_crop_geometry_from_refined_source(
        target_source,
        policy=_policy(),
        pixel_authority=pixels,
    )
    crop_root = tmp_path / "changed_retired_crops"
    parent_crop = publish_selector_ineligible_crop_geometry_snapshot(
        parent_prepared,
        destination=crop_root / "parent.zarr",
        run_id="crop_parent",
        shadow_root=crop_root,
        coordinate_catalog=True,
    )
    successor = publish_selector_ineligible_crop_geometry_successor(
        parent_crop,
        target_prepared,
        destination=crop_root / "successor.zarr",
        run_id="crop_replace_delete_successor",
        shadow_root=crop_root,
        created_by="keypoint_successor_test",
    )
    return parent_crop, successor


def _parent_keypoints(tmp_path: Path, parent_crop):  # type: ignore[no-untyped-def]
    crop = parent_crop.arrays
    rows = parent_crop.dimensions.n_instances
    keypoints = 3
    points = np.repeat(
        np.asarray([[[1, 4], [6, 2], [6, 6]]], dtype=np.float32),
        rows,
        axis=0,
    )
    confidences = np.full((rows, keypoints), np.float32(0.9), dtype=np.float32)
    pose_confidence = np.full(rows, np.float32(0.95), dtype=np.float32)
    bbox_roi = np.repeat(
        np.asarray([[0.5, 0.5, 7.5, 7.5]], dtype=np.float32),
        rows,
        axis=0,
    )
    origins = np.asarray(crop["roi_coordinates_full"][:], dtype=np.float32)
    valid = np.ones((rows, keypoints), dtype=bool)
    binding = _pose_binding()
    arrays = {
        "instance_key": np.asarray(crop["instance_key"][:], dtype=np.uint64),
        "source_crop_row_ids": np.arange(rows, dtype=np.int64),
        "source_acquisition_frame_index": np.asarray(
            crop["source_acquisition_frame_index"][:], dtype=np.int64
        ),
        "frame_indices": np.asarray(crop["frame_indices"][:], dtype=np.int64),
        "frame_row_offsets": np.asarray(
            crop["frame_row_offsets"][:], dtype=np.int64
        ),
        "source_crop_row_signature": np.asarray(
            crop["source_row_signature"][:], dtype=np.uint8
        ),
        "keypoints_roi": points,
        "keypoints_img": points + origins[:, None, :],
        "keypoint_confidences": confidences,
        "keypoint_valid": valid,
        "pose_confidence": pose_confidence,
        "pose_bbox_xyxy_roi": bbox_roi,
        "pose_bbox_xyxy_img": bbox_roi
        + np.column_stack((origins, origins)).astype(np.float32),
        "pose_success": np.ones(rows, dtype=bool),
    }
    arrays["keypoint_row_signature"] = derive_keypoint_row_signatures(
        instance_key=arrays["instance_key"],
        source_crop_row_signature=arrays["source_crop_row_signature"],
        keypoints_roi=points,
        keypoint_valid=valid,
        skeleton_digest=keypoint_skeleton_digest(binding),
    )
    dimensions = KeypointDimensions(
        n_frames=parent_crop.dimensions.n_frames,
        n_instances=rows,
        n_keypoints=keypoints,
        source_width=parent_crop.dimensions.source_width,
        source_height=parent_crop.dimensions.source_height,
    )
    prepared = prepare_raw_keypoint_v2_snapshot(
        arrays,
        dimensions=dimensions,
        source_crop_arrays=crop,
        source_crop_manifest=parent_crop.manifest,
        pose_model_schema_binding=binding,
        preprocessing=_preprocessing(),
    )
    root = tmp_path / "keypoints"
    return publish_selector_ineligible_keypoint_snapshot(
        prepared,
        destination=root / "parent.zarr",
        run_id="raw_keypoint_parent",
        shadow_root=root,
        created_by="keypoint_successor_test",
    )


def _failed_batch(key: int) -> TerminalKeypointInferenceBatch:
    return TerminalKeypointInferenceBatch(
        instance_key=np.asarray([key], dtype=np.uint64),
        keypoints_roi=np.full((1, 3, 2), np.nan, dtype=np.float32),
        keypoint_confidences=np.full((1, 3), np.nan, dtype=np.float32),
        pose_confidence=np.full(1, np.nan, dtype=np.float32),
        pose_bbox_xyxy_roi=np.full((1, 4), np.nan, dtype=np.float32),
        pose_success=np.zeros(1, dtype=bool),
    )


def _successful_batch(key: int) -> TerminalKeypointInferenceBatch:
    return TerminalKeypointInferenceBatch(
        instance_key=np.asarray([key], dtype=np.uint64),
        keypoints_roi=np.asarray(
            [[ [1, 4], [6, 2], [6, 6] ]],
            dtype=np.float32,
        ),
        keypoint_confidences=np.asarray([[0.8, 0.9, 0.85]], dtype=np.float32),
        pose_confidence=np.asarray([0.92], dtype=np.float32),
        pose_bbox_xyxy_roi=np.asarray([[0.5, 0.5, 7.5, 7.5]], dtype=np.float32),
        pose_success=np.ones(1, dtype=bool),
    )


def test_added_detection_failed_pose_becomes_complete_failed_keypoint_row(
    tmp_path: Path,
) -> None:
    parent_crop, crop_successor = _crop_successor(tmp_path)
    parent_keypoints = _parent_keypoints(tmp_path, parent_crop)
    added_key = int(crop_successor.plan.added_instance_keys[0])

    result = prepare_raw_keypoint_successor(
        parent_keypoints,
        crop_successor,
        _failed_batch(added_key),
    )

    arrays = result.prepared.arrays
    keys = np.asarray(arrays["instance_key"], dtype=np.uint64)
    added_row = int(np.flatnonzero(keys == np.uint64(added_key))[0])
    assert result.receipt["schema_id"] == RAW_KEYPOINT_SUCCESSOR_SCHEMA_ID
    assert result.receipt["row_coverage"] == "complete_target_crop_rowset"
    assert result.receipt["pending_row_count"] == 0
    assert result.receipt["instance_keys"]["reused"]["count"] == 3
    assert result.receipt["instance_keys"]["inference_failed"]["values"] == [
        added_key
    ]
    assert arrays["pose_success"][added_row] is np.False_
    assert np.isnan(arrays["keypoints_roi"][added_row]).all()
    np.testing.assert_array_equal(
        arrays["frame_row_offsets"],
        crop_successor.publication.arrays["frame_row_offsets"][:],
    )

    output = publish_selector_ineligible_raw_keypoint_successor(
        parent_keypoints,
        crop_successor,
        _failed_batch(added_key),
        destination=tmp_path / "keypoints/successor.zarr",
        run_id="raw_keypoint_successor",
        shadow_root=tmp_path / "keypoints",
        created_by="keypoint_successor_test",
    )
    assert validate_keypoint_shadow_publication(output.publication) == ()
    assert validate_raw_keypoint_successor_publication_receipt(output.receipt) == ()
    assert (
        output.publication.output_path
        / RAW_KEYPOINT_SUCCESSOR_PUBLICATION_RECEIPT_NAME
    ).is_file()
    tampered = copy.deepcopy(output.receipt)
    tampered["payload"]["selector_eligible"] = True
    errors = validate_raw_keypoint_successor_publication_receipt(tampered)
    assert "raw keypoint successor receipt payload digest mismatch" in errors
    assert "raw keypoint successor receipt must remain selector-ineligible" in errors

    successful = prepare_raw_keypoint_successor(
        parent_keypoints,
        crop_successor,
        _successful_batch(added_key),
    )
    successful_keys = np.asarray(
        successful.prepared.arrays["instance_key"], dtype=np.uint64
    )
    successful_row = int(
        np.flatnonzero(successful_keys == np.uint64(added_key))[0]
    )
    assert successful.receipt["instance_keys"]["inference_succeeded"][
        "values"
    ] == [added_key]
    assert bool(successful.prepared.arrays["pose_success"][successful_row])
    assert np.isfinite(
        successful.prepared.arrays["keypoints_roi"][successful_row]
    ).all()


def test_added_detection_requires_terminal_inference_evidence(tmp_path: Path) -> None:
    parent_crop, crop_successor = _crop_successor(tmp_path)
    parent_keypoints = _parent_keypoints(tmp_path, parent_crop)
    empty = TerminalKeypointInferenceBatch(
        instance_key=np.empty(0, dtype=np.uint64),
        keypoints_roi=np.empty((0, 3, 2), dtype=np.float32),
        keypoint_confidences=np.empty((0, 3), dtype=np.float32),
        pose_confidence=np.empty(0, dtype=np.float32),
        pose_bbox_xyxy_roi=np.empty((0, 4), dtype=np.float32),
        pose_success=np.empty(0, dtype=bool),
    )

    with pytest.raises(RawKeypointSuccessorError, match="must exactly equal"):
        prepare_raw_keypoint_successor(parent_keypoints, crop_successor, empty)


def test_terminal_failure_rejects_finite_landmark_payload() -> None:
    with pytest.raises(RawKeypointSuccessorError, match="exact NaN"):
        TerminalKeypointInferenceBatch(
            instance_key=np.asarray([7], dtype=np.uint64),
            keypoints_roi=np.zeros((1, 3, 2), dtype=np.float32),
            keypoint_confidences=np.zeros((1, 3), dtype=np.float32),
            pose_confidence=np.full(1, np.nan, dtype=np.float32),
            pose_bbox_xyxy_roi=np.full((1, 4), np.nan, dtype=np.float32),
            pose_success=np.zeros(1, dtype=bool),
        )


def test_removed_detection_disappears_and_changed_crop_requires_inference(
    tmp_path: Path,
) -> None:
    parent_crop, crop_successor = _changed_and_retired_crop_successor(tmp_path)
    parent_keypoints = _parent_keypoints(tmp_path, parent_crop)
    compute_keys = np.concatenate(
        (
            crop_successor.plan.added_instance_keys,
            crop_successor.plan.changed_instance_keys,
        )
    ).astype(np.uint64, copy=False)
    rows = int(compute_keys.shape[0])
    inferred_points = np.repeat(
        np.asarray([[[2, 5], [7, 3], [7, 7]]], dtype=np.float32),
        rows,
        axis=0,
    )
    inference = TerminalKeypointInferenceBatch(
        instance_key=compute_keys,
        keypoints_roi=inferred_points,
        keypoint_confidences=np.full((rows, 3), np.float32(0.8), dtype=np.float32),
        pose_confidence=np.full(rows, np.float32(0.9), dtype=np.float32),
        pose_bbox_xyxy_roi=np.repeat(
            np.asarray([[1, 1, 8, 8]], dtype=np.float32),
            rows,
            axis=0,
        ),
        pose_success=np.ones(rows, dtype=bool),
    )

    result = prepare_raw_keypoint_successor(
        parent_keypoints,
        crop_successor,
        inference,
    )

    output_keys = np.asarray(result.prepared.arrays["instance_key"], dtype=np.uint64)
    assert not np.isin(
        crop_successor.plan.retired_instance_keys,
        output_keys,
    ).any()
    assert crop_successor.plan.added_instance_keys.shape == (0,)
    assert crop_successor.plan.changed_instance_keys.shape == (1,)
    assert result.receipt["instance_keys"]["retired"]["count"] == 1
    assert result.receipt["instance_keys"]["inference_succeeded"][
        "count"
    ] == int(compute_keys.shape[0])
    for key, expected_points in zip(compute_keys, inferred_points, strict=True):
        row = int(np.flatnonzero(output_keys == key)[0])
        np.testing.assert_array_equal(
            result.prepared.arrays["keypoints_roi"][row],
            expected_points,
        )

    empty = TerminalKeypointInferenceBatch(
        instance_key=np.empty(0, dtype=np.uint64),
        keypoints_roi=np.empty((0, 3, 2), dtype=np.float32),
        keypoint_confidences=np.empty((0, 3), dtype=np.float32),
        pose_confidence=np.empty(0, dtype=np.float32),
        pose_bbox_xyxy_roi=np.empty((0, 4), dtype=np.float32),
        pose_success=np.empty(0, dtype=bool),
    )
    with pytest.raises(RawKeypointSuccessorError, match="must exactly equal"):
        prepare_raw_keypoint_successor(parent_keypoints, crop_successor, empty)
