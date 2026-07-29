from __future__ import annotations

import hashlib

import numpy as np
import pytest
import zarr

from fisheye.shared.immutable_yolo_storage import (
    IMMUTABLE_YOLO_STORAGE_ATTR,
    IMMUTABLE_YOLO_STORAGE_SCHEMA,
    validate_immutable_yolo_storage,
)
from fisheye.shared.zarr.columnar import store_array


def _create_array(
    group: zarr.Group,
    name: str,
    values: np.ndarray,
    *,
    sharded: bool,
) -> None:
    values = np.asarray(values)
    chunks = (2, *values.shape[1:])
    kwargs: dict[str, object] = {"data": values, "chunks": chunks}
    if sharded:
        kwargs["shards"] = (8, *values.shape[1:])
    group.create_array(name, **kwargs)


def _hashes(names: list[str]) -> dict[str, str]:
    return {name: hashlib.sha256(name.encode()).hexdigest() for name in names}


def _detect_run(tmp_path, *, sharded: bool = True) -> zarr.Group:
    root = zarr.open_group(tmp_path / "detect.zarr", mode="w", zarr_format=3)
    run = root.create_group("detect")
    row_values = {
        "frame_indices": np.asarray([0, 1, 1], dtype=np.int32),
        "bbox_norm_coords": np.arange(12, dtype=np.float32).reshape(3, 4),
        "scores": np.asarray([0.8, 0.9, 0.7], dtype=np.float32),
        "class_ids": np.zeros(3, dtype=np.int32),
        "instance_key": np.asarray([101, 102, 103], dtype=np.uint64),
    }
    frame_values = {
        "frame_counts": np.asarray([1, 2], dtype=np.int32),
        "n_detections": np.asarray([1, 2], dtype=np.int32),
    }
    for name, values in row_values.items():
        _create_array(run, name, values, sharded=sharded)
    for name, values in frame_values.items():
        _create_array(run, name, values, sharded=sharded)
    run.attrs.update(
        {
            "detect_storage_layout": (
                "indexed_sharding_v1" if sharded else "regular_chunks_v1"
            ),
            "detect_storage_policy": (
                "default_indexed_sharding_v1"
                if sharded
                else "explicit_regular_chunks_override"
            ),
            "detect_row_shard_rows": 8 if sharded else None,
            "detect_frame_shard_rows": 8 if sharded else None,
            "detect_shard_write": (
                {
                    "status": "complete",
                    "exact_match": True,
                    "detection_row_count": 3,
                    "source_sha256_by_array": _hashes(list(row_values) + list(frame_values)),
                    "destination_sha256_by_array": _hashes(list(row_values) + list(frame_values)),
                }
                if sharded
                else None
            ),
        }
    )
    return run


def _keypoint_run(tmp_path) -> zarr.Group:
    root = zarr.open_group(tmp_path / "keypoints.zarr", mode="w", zarr_format=3)
    run = root.create_group("keypoints")
    row_count = 3
    point_shape = (row_count, 5, 2)
    row_values = {
        "keypoints_roi": np.zeros(point_shape, dtype=np.float64),
        "keypoints_img": np.zeros(point_shape, dtype=np.float64),
        "keypoints_norm": np.zeros(point_shape, dtype=np.float64),
        "confidence": np.ones(row_count, dtype=np.float64),
        "keypoint_confidences": np.ones((row_count, 5), dtype=np.float64),
        "detection_success": np.ones(row_count, dtype=bool),
        "pose_bbox_xyxy_roi": np.zeros((row_count, 4), dtype=np.float32),
        "heading": np.zeros(row_count, dtype=np.float64),
        "heading_finite": np.ones(row_count, dtype=bool),
        "heading_usable": np.ones(row_count, dtype=bool),
        "effective_threshold": np.ones(row_count, dtype=np.float64),
        "effective_se2_radius": np.ones(row_count, dtype=np.float64),
        "detection_source": np.ones(row_count, dtype=np.int8),
        "frame_indices": np.asarray([0, 1, 1], dtype=np.int32),
        "source_crop_row_ids": np.arange(row_count, dtype=np.int64),
        "instance_key": np.asarray([201, 202, 203], dtype=np.uint64),
    }
    frame_values = {
        "frame_counts": np.asarray([1, 2], dtype=np.int32),
        "n_keypoints": np.asarray([5, 10], dtype=np.int32),
        "n_rois": np.asarray([1, 2], dtype=np.int32),
    }
    for name, values in row_values.items():
        _create_array(run, name, values, sharded=True)
    for name, values in frame_values.items():
        _create_array(run, name, values, sharded=True)
    hashed = _hashes(["keypoints_roi", "confidence"])
    run.attrs.update(
        {
            "keypoint_storage_layout": "indexed_sharding_v1",
            "keypoint_storage_policy": "default_indexed_sharding_v1",
            "keypoint_roi_shard_rows": 8,
            "keypoint_frame_shard_rows": 8,
            "keypoint_shard_write": {
                "status": "complete",
                "exact_match": True,
                "row_count": row_count,
                "source_sha256_by_array": hashed,
                "destination_sha256_by_array": hashed,
            },
        }
    )
    return run


def test_validates_sharded_detection_completion_contract(tmp_path) -> None:
    run = _detect_run(tmp_path, sharded=True)

    report = validate_immutable_yolo_storage(
        run,
        stage="detect",
        row_shard_rows=8,
        frame_shard_rows=8,
    )

    assert report["schema_id"] == IMMUTABLE_YOLO_STORAGE_SCHEMA
    assert report["status"] == "ok"
    assert report["eligible_arrays_checked"] == 7
    assert report["instance_key_unique"] is True
    assert run.attrs[IMMUTABLE_YOLO_STORAGE_ATTR]["status"] == "ok"


def test_validates_explicit_regular_detection_completion_contract(tmp_path) -> None:
    run = _detect_run(tmp_path, sharded=False)

    report = validate_immutable_yolo_storage(
        run,
        stage="detect",
        row_shard_rows=None,
        frame_shard_rows=8,
    )

    assert report["storage_layout"] == "regular_chunks_v1"
    assert all(item["shards"] is None for item in report["arrays"])


def test_read_only_validation_preserves_persisted_completion_report(tmp_path) -> None:
    run = _detect_run(tmp_path, sharded=True)
    validate_immutable_yolo_storage(
        run,
        stage="detect",
        row_shard_rows=8,
        frame_shard_rows=8,
    )
    persisted = dict(run.attrs[IMMUTABLE_YOLO_STORAGE_ATTR])

    report = validate_immutable_yolo_storage(
        run,
        stage="detect",
        row_shard_rows=8,
        frame_shard_rows=8,
        persist_report=False,
    )

    assert report["status"] == "ok"
    assert dict(run.attrs[IMMUTABLE_YOLO_STORAGE_ATTR]) == persisted


def test_read_only_validation_does_not_persist_failure_report(tmp_path) -> None:
    run = _detect_run(tmp_path)
    del run["instance_key"]
    assert IMMUTABLE_YOLO_STORAGE_ATTR not in run.attrs

    with pytest.raises(RuntimeError, match="missing required arrays.*instance_key"):
        validate_immutable_yolo_storage(
            run,
            stage="detect",
            row_shard_rows=8,
            frame_shard_rows=8,
            persist_report=False,
        )

    assert IMMUTABLE_YOLO_STORAGE_ATTR not in run.attrs


def test_rejects_missing_modern_identity(tmp_path) -> None:
    run = _detect_run(tmp_path)
    del run["instance_key"]

    with pytest.raises(RuntimeError, match="missing required arrays.*instance_key"):
        validate_immutable_yolo_storage(
            run,
            stage="detect",
            row_shard_rows=8,
            frame_shard_rows=8,
        )

    failure = run.attrs[IMMUTABLE_YOLO_STORAGE_ATTR]
    assert failure["status"] == "error"
    assert any("instance_key" in error for error in failure["errors"])


def test_rejects_one_ordinary_array_in_sharded_run(tmp_path) -> None:
    run = _detect_run(tmp_path)
    values = np.asarray(run["scores"][:])
    del run["scores"]
    run.create_array("scores", data=values, chunks=(2,))

    with pytest.raises(RuntimeError, match="scores shards=None"):
        validate_immutable_yolo_storage(
            run,
            stage="detect",
            row_shard_rows=8,
            frame_shard_rows=8,
        )


def test_rejects_incomplete_shard_write_summary(tmp_path) -> None:
    run = _detect_run(tmp_path)
    summary = dict(run.attrs["detect_shard_write"])
    summary["exact_match"] = False
    run.attrs["detect_shard_write"] = summary

    with pytest.raises(RuntimeError, match="exact_match is not true"):
        validate_immutable_yolo_storage(
            run,
            stage="detect",
            row_shard_rows=8,
            frame_shard_rows=8,
        )


def test_validates_keypoint_contract_and_rejects_duplicate_identity(tmp_path) -> None:
    run = _keypoint_run(tmp_path)
    report = validate_immutable_yolo_storage(
        run,
        stage="keypoints",
        row_shard_rows=8,
        frame_shard_rows=8,
    )
    assert report["status"] == "ok"
    assert report["eligible_arrays_checked"] == 19

    run["instance_key"][:] = np.asarray([201, 201, 203], dtype=np.uint64)
    with pytest.raises(RuntimeError, match="duplicate values"):
        validate_immutable_yolo_storage(
            run,
            stage="keypoints",
            row_shard_rows=8,
            frame_shard_rows=8,
        )


def test_accepts_exact_columnar_short_array_optimization(tmp_path) -> None:
    run = _keypoint_run(tmp_path)
    signature = store_array(
        run,
        "source_row_signature",
        np.zeros((3, 32), dtype=np.uint8),
        shard_rows=8,
    )
    assert signature.shards is None

    report = validate_immutable_yolo_storage(
        run,
        stage="keypoints",
        row_shard_rows=8,
        frame_shard_rows=8,
    )

    checked = {item["name"]: item for item in report["arrays"]}
    assert checked["source_row_signature"]["expected_shards"] is None


def test_rejects_tampered_columnar_short_array_declaration(tmp_path) -> None:
    run = _keypoint_run(tmp_path)
    signature = store_array(
        run,
        "source_row_signature",
        np.zeros((3, 32), dtype=np.uint8),
        shard_rows=8,
    )
    signature.attrs["palette_shard_rows_requested"] = 16

    with pytest.raises(RuntimeError, match="palette_shard_rows_requested"):
        validate_immutable_yolo_storage(
            run,
            stage="keypoints",
            row_shard_rows=8,
            frame_shard_rows=8,
        )
