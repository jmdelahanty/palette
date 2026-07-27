from __future__ import annotations

import copy

import numpy as np
import pytest

from fisheye.shared.instance_keys import mint_manual_curation_instance_keys
from fisheye.shared.zarr.detection_schema import derive_canonical_detection_geometry
from fisheye.shared.zarr.refined_detection_delta import (
    REFINED_DETECTION_DELTA_ARRAYS,
    REFINED_DETECTION_DELTA_OPERATION_CODE_MAP,
    RefinedDetectionDeltaBatch,
    RefinedDetectionDeltaError,
    refined_detection_delta_schema_manifest,
    resolve_refined_detection_deltas,
)
from fisheye.shared.zarr.refined_detection_schema import (
    REFINED_DETECTION_SCHEMA_V1,
    SOURCE_DECISION_CODE_MAP,
    SOURCE_KIND_CODE_MAP,
    RefinedDetectionDimensions,
)


RECORDING_IDENTITY = "delta_v2_recording"
BASE_SNAPSHOT_ID = "11111111-1111-4111-8111-111111111111"
DELTA_LINEAGE_ID = "22222222-2222-4222-8222-222222222222"
BASE_MANIFEST_DIGEST = "a" * 64


def _offsets(frames: np.ndarray, n_frames: int = 4) -> np.ndarray:
    counts = np.bincount(frames.astype(np.int64), minlength=n_frames)
    result = np.zeros(n_frames + 1, dtype=np.int64)
    result[1:] = np.cumsum(counts, dtype=np.int64)
    return result


def _base() -> tuple[RefinedDetectionDimensions, dict[str, np.ndarray]]:
    dimensions = RefinedDetectionDimensions(
        n_frames=4,
        n_instances=2,
        n_source_detections=3,
        source_width=100,
        source_height=80,
    )
    source_frames = np.asarray([0, 1, 2], dtype=np.int32)
    source_bbox = np.asarray(
        [
            [0.25, 0.25, 0.20, 0.20],
            [0.50, 0.50, 0.20, 0.20],
            [0.75, 0.75, 0.20, 0.20],
        ],
        dtype=np.float32,
    )
    source_bbox_img, source_centers = derive_canonical_detection_geometry(
        source_bbox,
        source_width=dimensions.source_width,
        source_height=dimensions.source_height,
    )
    instance_rows = np.asarray([0, 2], dtype=np.int64)
    instance_frames = source_frames[instance_rows]
    instance_bbox = source_bbox[instance_rows]
    instance_bbox_img, instance_centers = derive_canonical_detection_geometry(
        instance_bbox,
        source_width=dimensions.source_width,
        source_height=dimensions.source_height,
    )
    arrays = {
        "instances/frame_indices": instance_frames,
        "instances/source_acquisition_frame_index": instance_frames.astype(np.int64),
        "instances/instance_key": np.asarray([10, 30], dtype=np.uint64),
        "instances/refined_row_ids": np.asarray([5, 7], dtype=np.int64),
        "instances/bbox_norm_coords": instance_bbox,
        "instances/bbox_img_xyxy": instance_bbox_img,
        "instances/centers_img_xy": instance_centers,
        "instances/scores": np.asarray([0.9, 0.8], dtype=np.float32),
        "instances/score_valid": np.asarray([True, True], dtype=np.bool_),
        "instances/class_ids": np.asarray([1, 2], dtype=np.int32),
        "instances/source_kind_codes": np.asarray(
            [SOURCE_KIND_CODE_MAP["raw_detect"]] * 2, dtype=np.uint8
        ),
        "instances/manual_edit_flags": np.asarray([False, False], dtype=np.bool_),
        "instances/source_detect_row_index": np.asarray([0, 2], dtype=np.int64),
        "instances/reason_codes": np.asarray([0, 0], dtype=np.uint16),
        "instances/frame_row_offsets": _offsets(instance_frames),
        "source_detections/source_detect_row_index": np.arange(3, dtype=np.int64),
        "source_detections/frame_indices": source_frames,
        "source_detections/source_acquisition_frame_index": source_frames.astype(
            np.int64
        ),
        "source_detections/instance_key": np.asarray([10, 20, 30], dtype=np.uint64),
        "source_detections/bbox_norm_coords": source_bbox,
        "source_detections/bbox_img_xyxy": source_bbox_img,
        "source_detections/centers_img_xy": source_centers,
        "source_detections/scores": np.asarray([0.9, 0.2, 0.8], dtype=np.float32),
        "source_detections/class_ids": np.asarray([1, 1, 2], dtype=np.int32),
        "source_detections/decision_codes": np.asarray(
            [
                SOURCE_DECISION_CODE_MAP["accepted"],
                SOURCE_DECISION_CODE_MAP["filtered"],
                SOURCE_DECISION_CODE_MAP["accepted"],
            ],
            dtype=np.uint8,
        ),
        "source_detections/resolved_refined_row_id": np.asarray(
            [5, -1, 7], dtype=np.int64
        ),
        "source_detections/reason_codes": np.asarray([0, 1, 0], dtype=np.uint16),
        "source_detections/frame_row_offsets": _offsets(source_frames),
    }
    assert REFINED_DETECTION_SCHEMA_V1.validate(arrays, dimensions=dimensions) == ()
    return dimensions, arrays


def _sentinel_payload() -> dict[str, object]:
    return {
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
    }


def _event(
    *,
    sequence: int,
    operation: str,
    instance_key: int,
    refined_row_id: int,
    predecessor: int = 0,
    hint: int = -1,
    reason_code: int = 0,
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "event_sequence": sequence,
        "expected_previous_event_sequence": predecessor,
        "operation_codes": REFINED_DETECTION_DELTA_OPERATION_CODE_MAP[operation],
        "instance_key": instance_key,
        "refined_row_ids": refined_row_id,
        "row_index_hint": hint,
        "timestamp_ns": sequence * 100,
        "reason_codes": reason_code,
        **(_sentinel_payload() if payload is None else payload),
    }


def _payload(
    *,
    frame: int,
    bbox: list[float],
    score: float,
    score_valid: bool,
    class_id: int,
    source_kind: int,
    source_row: int,
) -> dict[str, object]:
    return {
        "payload_valid": True,
        "frame_indices": frame,
        "source_acquisition_frame_index": frame,
        "bbox_norm_coords": bbox,
        "scores": score,
        "score_valid": score_valid,
        "class_ids": class_id,
        "source_kind_codes": source_kind,
        "manual_edit_flags": True,
        "source_detect_row_index": source_row,
    }


def _batch(
    events: list[dict[str, object]],
    *,
    partition_id: str = "partition_0001",
    generation_ordinal: int = 1,
) -> RefinedDetectionDeltaBatch:
    arrays: dict[str, np.ndarray] = {}
    for declaration in REFINED_DETECTION_DELTA_ARRAYS:
        values = [event[declaration.name] for event in events]
        arrays[declaration.name] = np.asarray(
            values,
            dtype=np.dtype(declaration.dtype),
        ).reshape(len(events), *declaration.trailing_shape)
    return RefinedDetectionDeltaBatch(
        delta_lineage_id=DELTA_LINEAGE_ID,
        base_snapshot_id=BASE_SNAPSHOT_ID,
        base_manifest_digest=BASE_MANIFEST_DIGEST,
        generation_ordinal=generation_ordinal,
        partition_id=partition_id,
        actor_id="reviewer@example.org",
        reason_code_map={
            0: "none",
            1: "bbox_corrected",
            2: "missed_detection",
            3: "false_positive",
            4: "restored_after_review",
        },
        arrays=arrays,
    )


def _resolve(*batches: RefinedDetectionDeltaBatch):
    dimensions, arrays = _base()
    return resolve_refined_detection_deltas(
        base_dimensions=dimensions,
        base_arrays=arrays,
        base_instance_reason_codes={0: "none"},
        base_source_reason_codes={0: "none", 1: "low_score"},
        recording_identity=RECORDING_IDENTITY,
        base_snapshot_id=BASE_SNAPSHOT_ID,
        base_manifest_digest=BASE_MANIFEST_DIGEST,
        next_refined_row_id=8,
        batches=batches,
    )


def _resolve_empty(*batches: RefinedDetectionDeltaBatch):
    _dimensions, populated = _base()
    dimensions = RefinedDetectionDimensions(
        n_frames=4,
        n_instances=0,
        n_source_detections=0,
        source_width=100,
        source_height=80,
    )
    arrays: dict[str, np.ndarray] = {}
    for path, values in populated.items():
        if path.endswith("frame_row_offsets"):
            arrays[path] = np.zeros(5, dtype=np.int64)
        elif path.endswith("bbox_norm_coords") or path.endswith("bbox_img_xyxy"):
            arrays[path] = np.empty((0, 4), dtype=np.float32)
        elif path.endswith("centers_img_xy"):
            arrays[path] = np.empty((0, 2), dtype=np.float32)
        else:
            arrays[path] = np.empty((0,), dtype=values.dtype)
    assert REFINED_DETECTION_SCHEMA_V1.validate(arrays, dimensions=dimensions) == ()
    return resolve_refined_detection_deltas(
        base_dimensions=dimensions,
        base_arrays=arrays,
        base_instance_reason_codes={0: "none"},
        base_source_reason_codes={0: "none"},
        recording_identity=RECORDING_IDENTITY,
        base_snapshot_id=BASE_SNAPSHOT_ID,
        base_manifest_digest=BASE_MANIFEST_DIGEST,
        next_refined_row_id=0,
        batches=batches,
    )


def _manual_key(row_id: int, frame: int, bbox: list[float], class_id: int) -> int:
    return int(
        mint_manual_curation_instance_keys(
            recording_identity=RECORDING_IDENTITY,
            refined_row_ids=np.asarray([row_id], dtype=np.int64),
            frame_indices=np.asarray([frame], dtype=np.int32),
            bbox_norm_coords=np.asarray([bbox], dtype=np.float32),
            class_ids=np.asarray([class_id], dtype=np.int32),
        )[0]
    )


def test_schema_manifest_freezes_exact_columns_and_storage_boundary() -> None:
    manifest = refined_detection_delta_schema_manifest()

    assert manifest["schema_id"] == "palette.refined_detection.delta"
    assert manifest["schema_version"] == 2
    assert manifest["merge_order"] == ["event_sequence"]
    assert manifest["lineage_profiles"] == ["full_acquisition"]
    assert manifest["storage"] == {
        "write_ownership": "one_immutable_partition_per_writer",
        "chunks": "one_complete_ordinary_chunk_per_array",
        "shards": None,
        "consolidated_metadata": False,
    }
    assert [row["name"] for row in manifest["array_declarations"]] == [
        declaration.name for declaration in REFINED_DETECTION_DELTA_ARRAYS
    ]


def test_add_replace_delete_restore_resolves_complete_multisubject_snapshot() -> None:
    corrected_bbox = [0.30, 0.25, 0.20, 0.20]
    added_bbox = [0.55, 0.55, 0.10, 0.10]
    added_key = _manual_key(8, 0, added_bbox, 3)
    batch = _batch(
        [
            _event(
                sequence=1,
                operation="replace_instance",
                instance_key=10,
                refined_row_id=5,
                hint=0,
                reason_code=1,
                payload=_payload(
                    frame=0,
                    bbox=corrected_bbox,
                    score=0.9,
                    score_valid=True,
                    class_id=4,
                    source_kind=SOURCE_KIND_CODE_MAP["raw_detect"],
                    source_row=0,
                ),
            ),
            _event(
                sequence=2,
                operation="add_instance",
                instance_key=added_key,
                refined_row_id=8,
                reason_code=2,
                payload=_payload(
                    frame=0,
                    bbox=added_bbox,
                    score=0.0,
                    score_valid=False,
                    class_id=3,
                    source_kind=SOURCE_KIND_CODE_MAP["manual"],
                    source_row=-1,
                ),
            ),
            _event(
                sequence=3,
                operation="delete_instance",
                instance_key=30,
                refined_row_id=7,
                hint=1,
                reason_code=3,
            ),
            _event(
                sequence=4,
                operation="restore_instance",
                instance_key=30,
                refined_row_id=7,
                predecessor=3,
                hint=1,
                reason_code=4,
            ),
        ]
    )

    result = _resolve(batch)

    assert result.dimensions.n_instances == 3
    assert result.next_refined_row_id == 9
    np.testing.assert_array_equal(result.arrays["instances/frame_indices"], [0, 0, 2])
    np.testing.assert_array_equal(result.arrays["instances/refined_row_ids"], [5, 8, 7])
    np.testing.assert_array_equal(
        result.arrays["instances/instance_key"], [10, added_key, 30]
    )
    np.testing.assert_array_equal(
        result.arrays["instances/frame_row_offsets"], [0, 2, 2, 3, 3]
    )
    assert result.arrays["instances/manual_edit_flags"].tolist() == [True, True, False]
    assert result.arrays["instances/class_ids"].tolist() == [4, 3, 2]
    assert result.arrays["source_detections/decision_codes"].tolist() == [0, 1, 0]
    assert result.arrays["source_detections/resolved_refined_row_id"].tolist() == [
        5,
        -1,
        7,
    ]
    assert result.report["operation_counts"] == {
        "add_instance": 1,
        "replace_instance": 1,
        "delete_instance": 1,
        "restore_instance": 1,
    }
    assert result.report["added_instance_keys"] == [added_key]
    assert result.report["rowset_changed"] is True
    assert (
        REFINED_DETECTION_SCHEMA_V1.validate(
            result.arrays,
            dimensions=result.dimensions,
        )
        == ()
    )


def test_delete_updates_source_audit_and_rebuilds_offsets() -> None:
    batch = _batch(
        [
            _event(
                sequence=1,
                operation="delete_instance",
                instance_key=30,
                refined_row_id=7,
                hint=1,
                reason_code=3,
            )
        ]
    )

    result = _resolve(batch)

    assert result.arrays["instances/instance_key"].tolist() == [10]
    assert result.arrays["instances/frame_row_offsets"].tolist() == [0, 1, 1, 1, 1]
    assert result.arrays["source_detections/decision_codes"].tolist() == [0, 1, 3]
    assert result.arrays["source_detections/resolved_refined_row_id"].tolist() == [
        5,
        -1,
        -1,
    ]
    deleted_reason_code = int(result.arrays["source_detections/reason_codes"][2])
    assert result.source_reason_codes[deleted_reason_code] == "false_positive"
    assert result.report["deleted_instance_keys"] == [30]


def test_manual_add_to_all_empty_base_is_a_complete_presentable_snapshot() -> None:
    bbox = [0.40, 0.40, 0.10, 0.10]
    key = _manual_key(0, 2, bbox, 6)
    batch = _batch(
        [
            _event(
                sequence=1,
                operation="add_instance",
                instance_key=key,
                refined_row_id=0,
                reason_code=2,
                payload=_payload(
                    frame=2,
                    bbox=bbox,
                    score=0.0,
                    score_valid=False,
                    class_id=6,
                    source_kind=SOURCE_KIND_CODE_MAP["manual"],
                    source_row=-1,
                ),
            )
        ]
    )

    result = _resolve_empty(batch)

    assert result.dimensions.n_instances == 1
    assert result.dimensions.n_source_detections == 0
    assert result.arrays["instances/frame_indices"].tolist() == [2]
    assert result.arrays["instances/frame_row_offsets"].tolist() == [0, 0, 0, 1, 1]
    assert result.arrays["source_detections/frame_row_offsets"].tolist() == [
        0,
        0,
        0,
        0,
        0,
    ]


def test_stale_expected_predecessor_fails_closed() -> None:
    corrected = _payload(
        frame=0,
        bbox=[0.30, 0.25, 0.20, 0.20],
        score=0.9,
        score_valid=True,
        class_id=1,
        source_kind=SOURCE_KIND_CODE_MAP["raw_detect"],
        source_row=0,
    )
    batch = _batch(
        [
            _event(
                sequence=1,
                operation="replace_instance",
                instance_key=10,
                refined_row_id=5,
                hint=0,
                payload=corrected,
            ),
            _event(
                sequence=2,
                operation="replace_instance",
                instance_key=10,
                refined_row_id=5,
                predecessor=0,
                hint=0,
                payload=corrected,
            ),
        ]
    )

    with pytest.raises(RefinedDetectionDeltaError, match="stale predecessor"):
        _resolve(batch)


def test_add_requires_allocated_row_id_and_frozen_manual_key() -> None:
    bbox = [0.55, 0.55, 0.10, 0.10]
    event = _event(
        sequence=1,
        operation="add_instance",
        instance_key=999,
        refined_row_id=7,
        payload=_payload(
            frame=1,
            bbox=bbox,
            score=0.0,
            score_valid=False,
            class_id=3,
            source_kind=SOURCE_KIND_CODE_MAP["manual"],
            source_row=-1,
        ),
    )

    with pytest.raises(RefinedDetectionDeltaError, match="monotonic allocator"):
        _resolve(_batch([event]))

    event["refined_row_ids"] = 8
    with pytest.raises(RefinedDetectionDeltaError, match="frozen allocator"):
        _resolve(_batch([event]))


def test_replace_cannot_change_sealed_frame_or_source_lineage() -> None:
    batch = _batch(
        [
            _event(
                sequence=1,
                operation="replace_instance",
                instance_key=10,
                refined_row_id=5,
                hint=0,
                payload=_payload(
                    frame=1,
                    bbox=[0.30, 0.25, 0.20, 0.20],
                    score=0.9,
                    score_valid=True,
                    class_id=1,
                    source_kind=SOURCE_KIND_CODE_MAP["raw_detect"],
                    source_row=0,
                ),
            )
        ]
    )

    with pytest.raises(RefinedDetectionDeltaError, match="sealed"):
        _resolve(batch)


def test_restore_is_limited_to_latest_uncompacted_tombstone() -> None:
    restore_without_delete = _batch(
        [
            _event(
                sequence=1,
                operation="restore_instance",
                instance_key=30,
                refined_row_id=7,
                hint=1,
            )
        ]
    )
    with pytest.raises(RefinedDetectionDeltaError, match="same open generation"):
        _resolve(restore_without_delete)


def test_restore_cannot_cross_a_frozen_generation_boundary() -> None:
    deleted = _batch(
        [
            _event(
                sequence=1,
                operation="delete_instance",
                instance_key=30,
                refined_row_id=7,
                hint=1,
            )
        ],
        generation_ordinal=1,
    )
    restored = _batch(
        [
            _event(
                sequence=2,
                operation="restore_instance",
                instance_key=30,
                refined_row_id=7,
                predecessor=1,
                hint=1,
            )
        ],
        partition_id="partition_0002",
        generation_ordinal=2,
    )

    with pytest.raises(RefinedDetectionDeltaError, match="same open generation"):
        _resolve(deleted, restored)


def test_global_event_sequence_is_unique_across_partitions() -> None:
    first = _batch(
        [
            _event(
                sequence=1,
                operation="delete_instance",
                instance_key=10,
                refined_row_id=5,
                hint=0,
            )
        ],
        partition_id="partition_a",
    )
    second = _batch(
        [
            _event(
                sequence=1,
                operation="delete_instance",
                instance_key=30,
                refined_row_id=7,
                hint=1,
            )
        ],
        partition_id="partition_b",
    )

    with pytest.raises(RefinedDetectionDeltaError, match="globally unique"):
        _resolve(first, second)


def test_partition_event_sequence_must_be_strictly_increasing_without_uint64_wrap() -> (
    None
):
    events = [
        _event(
            sequence=2,
            operation="delete_instance",
            instance_key=10,
            refined_row_id=5,
            hint=0,
        ),
        _event(
            sequence=1,
            operation="delete_instance",
            instance_key=30,
            refined_row_id=7,
            hint=1,
        ),
    ]

    with pytest.raises(RefinedDetectionDeltaError, match="strictly increasing"):
        _batch(events)


def test_batch_requires_exact_dtypes_fields_and_tombstone_sentinels() -> None:
    event = _event(
        sequence=1,
        operation="delete_instance",
        instance_key=10,
        refined_row_id=5,
        hint=0,
    )
    batch = _batch([event])

    wrong_dtype = dict(batch.arrays)
    wrong_dtype["class_ids"] = wrong_dtype["class_ids"].astype(np.int64)
    with pytest.raises(RefinedDetectionDeltaError, match="dtype must be int32"):
        RefinedDetectionDeltaBatch(
            delta_lineage_id=DELTA_LINEAGE_ID,
            base_snapshot_id=BASE_SNAPSHOT_ID,
            base_manifest_digest=BASE_MANIFEST_DIGEST,
            generation_ordinal=1,
            partition_id="wrong_dtype",
            actor_id="reviewer",
            reason_code_map={0: "none"},
            arrays=wrong_dtype,
        )

    bad_sentinel = copy.deepcopy(event)
    bad_sentinel["class_ids"] = 0
    with pytest.raises(RefinedDetectionDeltaError, match="class_ids sentinel"):
        _batch([bad_sentinel])
