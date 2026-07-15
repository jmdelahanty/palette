from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr

from fisheye.shared.refined_detect_curation import (
    REFINED_SOURCE_DETECTION_DECISION_CODE_MAP,
    REFINED_SOURCE_KIND_CODE_MAP,
)
from fisheye.shared.zarr_run_completion import mark_run_complete, mark_run_started
from fisheye.utils.backfill_refined_subject_mask_instance_keys import (
    LINEAGE_ARRAYS,
    backfill_refined_subject_mask_instance_keys,
)
from fisheye.utils.publish_clipped_refined_detect_snapshot import (
    publish_clipped_refined_detect_snapshot,
)


def _reason_bytes(rows: int, label: str) -> np.ndarray:
    values = np.zeros((rows, 64), dtype=np.uint8)
    encoded = label.encode("utf-8")
    values[:, : len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
    return values


def _write_clip_run(
    root: zarr.Group,
    *,
    clip_index: int,
    frame_count: int,
    instance_frames: list[int],
    instance_keys: list[int],
) -> dict[str, object]:
    clip_id = f"clip_{clip_index:06d}"
    camera = "2010095"
    run_name = f"refined_{clip_id}"
    refined_path = f"clips/{clip_id}/cameras/{camera}/refined_detect_runs/{run_name}"
    parent = root.require_group(f"clips/{clip_id}/cameras/{camera}/refined_detect_runs")
    run = parent.create_group(run_name)
    mark_run_started(run, run_name=run_name, stage="refined_detect")
    instances = run.create_group("instances")
    source = run.create_group("source_detections")

    frames = np.asarray(instance_frames, dtype=np.int32)
    rows = int(frames.shape[0])
    local_refined_ids = np.asarray([10 + value for value in range(rows)], dtype=np.int64)
    bbox = np.asarray(
        [[float(frame), 1.0, float(frame) + 2.0, 3.0] for frame in instance_frames],
        dtype=np.float64,
    )
    counts = np.bincount(frames, minlength=frame_count).astype(np.int32)
    offsets = np.zeros(frame_count + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts, dtype=np.int64)
    accepted = int(REFINED_SOURCE_DETECTION_DECISION_CODE_MAP["accepted"])
    filtered = int(REFINED_SOURCE_DETECTION_DECISION_CODE_MAP["filtered"])
    raw = int(REFINED_SOURCE_KIND_CODE_MAP["raw_detect"])

    instance_values = {
        "refined_row_ids": local_refined_ids,
        "frame_indices": frames,
        "frame_offsets": offsets,
        "bbox_img_xyxy": bbox,
        "bbox_norm_coords": bbox / 10.0,
        "source_kind_codes": np.full(rows, raw, dtype=np.int8),
        "manual_edit_flags": np.zeros(rows, dtype=bool),
        "source_detect_row_index": frames.copy(),
        "instance_key": np.asarray(instance_keys, dtype=np.uint64),
        "instance_key_origin_codes": np.zeros(rows, dtype=np.int8),
        "frame_counts": counts,
        "reason_bytes": _reason_bytes(rows, "accepted"),
        "confidence_scores": np.full(rows, 0.9, dtype=np.float32),
        "class_ids": np.zeros(rows, dtype=np.int32),
    }
    for name, values in instance_values.items():
        trailing = tuple(int(value) for value in values.shape[1:])
        chunks = (max(1, int(values.shape[0])), *trailing)
        instances.create_array(name, data=values, chunks=chunks, overwrite=True)
    instances.attrs.update(
        {
            "source_kind_code_map": dict(REFINED_SOURCE_KIND_CODE_MAP),
            "row_sort_order": ["frame_indices", "refined_row_ids"],
            "reason_encoding": "utf8-null-terminated",
            "reason_bytes_width": 64,
            "reason_bytes_null_terminated": True,
        }
    )

    source_frames = np.arange(frame_count, dtype=np.int32)
    source_bbox = np.asarray(
        [[float(frame), 1.0, float(frame) + 2.0, 3.0] for frame in source_frames],
        dtype=np.float64,
    )
    decisions = np.full(frame_count, filtered, dtype=np.int8)
    resolved = np.full(frame_count, -1, dtype=np.int64)
    source_keys = np.asarray(
        [clip_index * 1000 + 500 + frame for frame in range(frame_count)], dtype=np.uint64
    )
    instance_by_frame = dict(zip(instance_frames, local_refined_ids.tolist(), strict=True))
    key_by_frame = dict(zip(instance_frames, instance_keys, strict=True))
    for frame, refined_id in instance_by_frame.items():
        decisions[frame] = accepted
        resolved[frame] = refined_id
        source_keys[frame] = np.uint64(key_by_frame[frame])
    source_values = {
        "source_detect_row_index": np.arange(frame_count, dtype=np.int32),
        "frame_indices": source_frames,
        "bbox_img_xyxy": source_bbox,
        "bbox_norm_coords": source_bbox / 10.0,
        "decision_codes": decisions,
        "resolved_refined_row_id": resolved,
        "instance_key": source_keys,
        "reason_bytes": _reason_bytes(frame_count, "source"),
        "confidence_scores": np.full(frame_count, 0.8, dtype=np.float32),
        "class_ids": np.zeros(frame_count, dtype=np.int32),
    }
    for name, values in source_values.items():
        trailing = tuple(int(value) for value in values.shape[1:])
        chunks = (max(1, int(values.shape[0])), *trailing)
        source.create_array(name, data=values, chunks=chunks, overwrite=True)
    source.attrs.update(
        {
            "decision_code_map": dict(REFINED_SOURCE_DETECTION_DECISION_CODE_MAP),
            "reason_encoding": "utf8-null-terminated",
            "reason_bytes_width": 64,
            "reason_bytes_null_terminated": True,
        }
    )
    run.attrs.update(
        {
            "curated_primary_surface": "instances",
            "row_identity_policy": "stable_sparse_refined_row_id",
            "source_detect_path": f"clips/{clip_id}/detect_runs/raw",
            "source_detect_run": "raw",
            "source_quality_run": "quality",
            "refined_family_path": f"clips/{clip_id}/cameras/{camera}/refined_detect_runs",
        }
    )
    mark_run_complete(
        run,
        run_name=run_name,
        allow_missing_run_provenance=True,
        missing_run_provenance_reason="unit fixture",
    )
    return {
        "work_unit_id": clip_id,
        "clip_id": clip_id,
        "clip_index": clip_index,
        "camera_serial": camera,
        "frame_count": frame_count,
        "detect_run": f"detect_{clip_id}",
        "detect_quality_run": f"quality_{clip_id}",
        "refined_detect_run": run_name,
        "detect_group_path": f"clips/{clip_id}/detect_runs/raw",
        "refined_group_path": refined_path,
    }


def _fixture(tmp_path: Path) -> tuple[Path, str, list[int]]:
    zarr_path = tmp_path / "analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w", zarr_format=3)
    root.attrs["recording_id"] = "recording_a"
    selected = [
        _write_clip_run(
            root,
            clip_index=0,
            frame_count=3,
            instance_frames=[0, 2],
            instance_keys=[101, 103],
        ),
        _write_clip_run(
            root,
            clip_index=1,
            frame_count=2,
            instance_frames=[0, 1],
            instance_keys=[201, 202],
        ),
    ]
    frame_index_path = tmp_path / "recording_frame_index.parquet"
    pq.write_table(
        pa.table(
            {
                "camera_serial": ["2010095"] * 5,
                "clip_id": ["clip_000000"] * 3 + ["clip_000001"] * 2,
                "clip_local_frame_index": [0, 1, 2, 0, 1],
                "parent_frame_index": [0, 1, 2, 3, 4],
                "recording_frame_id": [1, 2, 3, 4, 5],
            }
        ),
        frame_index_path,
    )
    root.attrs["recording_frame_index_path"] = str(frame_index_path)
    collection_id = "collection_a"
    collection = root.require_group("experiment_index/finalized_runs").create_group(
        collection_id
    )
    collection.attrs.update(
        {
            "schema_version": "palette.refined_detect_clip_collection.v1",
            "collection_id": collection_id,
            "selected_runs": selected,
        }
    )
    refined_parent = root.require_group("refined_detect_runs")
    refined_parent.attrs["latest_collection"] = collection_id
    refined_parent.attrs["latest_collection_path"] = (
        f"experiment_index/finalized_runs/{collection_id}"
    )

    keypoint_parent = root.require_group("refined_keypoints_runs")
    keypoints = keypoint_parent.create_group("keypoints_a")
    mark_run_started(keypoints, run_name="keypoints_a", stage="refined_keypoints")
    keys = [101, 103, 201, 202]
    keypoints.create_array(
        "instance_key",
        data=np.asarray(keys, dtype=np.uint64),
        chunks=(2,),
        overwrite=True,
    )
    lineage_values = {
        "detection_indices": np.arange(4, dtype=np.int64),
        "detection_source": np.zeros(4, dtype=np.int8),
        "frame_counts": np.asarray([1, 0, 1, 1, 1], dtype=np.int32),
        "frame_indices": np.asarray([0, 2, 3, 4], dtype=np.int64),
        "source_clip_indices": np.asarray([0, 0, 1, 1], dtype=np.int64),
        "source_clip_local_frame_indices": np.asarray([0, 2, 0, 1], dtype=np.int64),
        "source_crop_row_ids": np.arange(4, dtype=np.int64),
        "source_detect_row_index": np.asarray([0, 2, 0, 1], dtype=np.int64),
        "source_frame_indices": np.asarray([0, 2, 3, 4], dtype=np.int64),
        "source_refined_row_ids": np.asarray([10, 11, 10, 11], dtype=np.int64),
    }
    for name, values in lineage_values.items():
        keypoints.create_array(name, data=values, chunks=(2,), overwrite=True)
    mark_run_complete(
        keypoints,
        parent_group=keypoint_parent,
        run_name="keypoints_a",
        allow_missing_run_provenance=True,
        missing_run_provenance_reason="unit fixture",
    )

    mask_parent = root.require_group("refined_subject_masks_runs")
    masks = mask_parent.create_group("masks_a")
    mark_run_started(masks, run_name="masks_a", stage="refined_subject_masks")
    for name, values in lineage_values.items():
        masks.create_array(name, data=values, chunks=(2,), overwrite=True)
    mark_run_complete(
        masks,
        parent_group=mask_parent,
        run_name="masks_a",
        allow_missing_run_provenance=True,
        missing_run_provenance_reason="unit fixture",
    )
    return zarr_path, collection_id, keys


def test_collection_snapshot_is_recording_mapped_sharded_and_promoted(tmp_path: Path) -> None:
    zarr_path, collection_id, keys = _fixture(tmp_path)

    result = publish_clipped_refined_detect_snapshot(
        zarr_path=zarr_path,
        collection_id=collection_id,
        output_run="recording_snapshot",
        shard_rows=4,
        apply=True,
    )

    assert result["status"] == "complete"
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    parent = root["refined_detect_runs"]
    assert parent.attrs["latest"] == "recording_snapshot"
    assert parent.attrs["latest_complete"] == "recording_snapshot"
    assert parent.attrs["latest_collection"] == collection_id
    target = parent["recording_snapshot"]
    instances = target["instances"]
    source = target["source_detections"]
    assert target.attrs["artifact_mutability"] == "immutable_snapshot"
    assert target.attrs["frame_indices_semantics"] == (
        "recording_parent_frame_index_0_based"
    )
    assert instances["bbox_img_xyxy"].shards == (4, 4)
    assert instances["instance_key"].shards == (4,)
    assert "reason" not in instances
    assert "reason" not in source
    np.testing.assert_array_equal(instances["instance_key"][:], np.asarray(keys, dtype=np.uint64))
    np.testing.assert_array_equal(instances["frame_indices"][:], [0, 2, 3, 4])
    np.testing.assert_array_equal(instances["source_frame_indices"][:], [0, 2, 3, 4])
    np.testing.assert_array_equal(instances["source_recording_frame_ids"][:], [1, 3, 4, 5])
    np.testing.assert_array_equal(instances["source_clip_indices"][:], [0, 0, 1, 1])
    np.testing.assert_array_equal(
        instances["source_clip_local_frame_indices"][:], [0, 2, 0, 1]
    )
    np.testing.assert_array_equal(instances["refined_row_ids"][:], [0, 1, 2, 3])
    np.testing.assert_array_equal(instances["source_refined_row_ids"][:], [10, 11, 10, 11])
    np.testing.assert_array_equal(instances["source_detect_row_index"][:], [0, 2, 3, 4])
    np.testing.assert_array_equal(
        instances["source_clip_detect_row_index"][:], [0, 2, 0, 1]
    )
    np.testing.assert_array_equal(instances["frame_counts"][:], [1, 0, 1, 1, 1])
    np.testing.assert_array_equal(instances["frame_offsets"][:], [0, 1, 1, 2, 3, 4])
    np.testing.assert_array_equal(source["source_detect_row_index"][:], [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(
        source["source_clip_detect_row_index"][:], [0, 1, 2, 0, 1]
    )
    np.testing.assert_array_equal(source["frame_indices"][:], [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(source["source_recording_frame_ids"][:], [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(source["resolved_refined_row_id"][:], [0, -1, 1, 2, 3])
    np.testing.assert_array_equal(
        source["source_resolved_refined_row_id"][:], [10, -1, 11, 10, 11]
    )
    assert "clips/clip_000000/cameras/2010095/refined_detect_runs/refined_clip_000000" in root
    assert result["contract_validation"]["status"] == "ok"


def test_collection_snapshot_defaults_to_read_only_plan(tmp_path: Path) -> None:
    zarr_path, collection_id, _keys = _fixture(tmp_path)

    result = publish_clipped_refined_detect_snapshot(
        zarr_path=zarr_path,
        collection_id=collection_id,
        output_run="recording_snapshot",
        shard_rows=4,
    )

    assert result["status"] == "planned"
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    assert "recording_snapshot" not in root["refined_detect_runs"]


def test_mask_instance_key_backfill_requires_exact_lineage_and_is_atomic(tmp_path: Path) -> None:
    zarr_path, _collection_id, keys = _fixture(tmp_path)

    planned = backfill_refined_subject_mask_instance_keys(
        zarr_path=zarr_path,
        block_rows=4,
        inner_rows=2,
    )
    assert planned["status"] == "planned"
    assert planned["planned_action"] == "add"
    assert [item["array"] for item in planned["lineage_validation"]] == list(
        LINEAGE_ARRAYS
    )

    applied = backfill_refined_subject_mask_instance_keys(
        zarr_path=zarr_path,
        block_rows=4,
        inner_rows=2,
        apply=True,
    )
    assert applied["status"] == "complete"
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    masks = root["refined_subject_masks_runs/masks_a"]
    np.testing.assert_array_equal(masks["instance_key"][:], np.asarray(keys, dtype=np.uint64))
    assert masks["instance_key"].shards == (4,)
    assert masks.attrs["instance_key_lineage_validation_status"] == "exact"


def test_mask_instance_key_backfill_rejects_lineage_mismatch(tmp_path: Path) -> None:
    zarr_path, _collection_id, _keys = _fixture(tmp_path)
    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    root["refined_subject_masks_runs/masks_a/source_crop_row_ids"][2] = 99

    try:
        backfill_refined_subject_mask_instance_keys(
            zarr_path=zarr_path,
            block_rows=4,
            inner_rows=2,
        )
    except ValueError as exc:
        assert "Lineage mismatch for source_crop_row_ids" in str(exc)
    else:
        raise AssertionError("Expected exact-lineage mask key repair to fail closed.")
