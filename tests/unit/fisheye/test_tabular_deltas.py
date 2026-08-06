from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.shared.tabular_deltas import (
    KEYPOINT_OPERATION_CODE_MAP,
    apply_keypoint_delta_overlay,
    create_delta_generation,
    freeze_delta_generation,
    resolve_keypoint_delta_overlay,
    write_delta_partition,
)


def _base_root(path: Path) -> zarr.Group:
    root = zarr.open_group(str(path / "sample.zarr"), mode="w", zarr_format=3)
    base = root.require_group("refined_keypoints_runs").create_group("snapshot")
    base.create_array(
        "instance_key",
        data=np.array([10, 20, 30], dtype=np.uint64),
        chunks=(3,),
        shards=(6,),
        overwrite=True,
    )
    base.attrs["artifact_mutability"] = "immutable_snapshot"
    return root


def test_partitioned_keypoint_delta_binds_by_instance_key_and_freezes(tmp_path: Path) -> None:
    root = _base_root(tmp_path)
    created = create_delta_generation(
        root,
        delta_run="review_001",
        generation="generation_000001",
        generation_ordinal=1,
        target_kind="keypoints",
        base_run_path="refined_keypoints_runs/snapshot",
        created_by="reviewer@example.org",
    )
    assert created["base_instance_key_count"] == 3

    result = write_delta_partition(
        root,
        delta_run="review_001",
        generation="generation_000001",
        partition="worker_00_batch_0001",
        editor="reviewer@example.org",
        instance_keys=np.array([20], dtype=np.uint64),
        row_index_hints=np.array([1], dtype=np.int64),
        operation_codes=np.array([KEYPOINT_OPERATION_CODE_MAP["replace_xy"]], dtype=np.uint8),
        revisions=np.array([1], dtype=np.uint64),
        timestamp_ns=np.array([123], dtype=np.int64),
        reason_codes=np.array([7], dtype=np.uint16),
        keypoint_index=np.array([2], dtype=np.int16),
        new_xy=np.array([[12.5, 18.5]], dtype=np.float64),
        valid=np.array([True]),
        reason_code_map={"manual_correction": 7},
    )
    assert result["row_count"] == 1
    partition = root[
        "edit_delta_runs/review_001/generations/generation_000001/partitions/worker_00_batch_0001"
    ]
    np.testing.assert_array_equal(partition["instance_key"][:], np.array([20], dtype=np.uint64))
    assert partition["instance_key"].shards is None

    frozen = freeze_delta_generation(
        root,
        delta_run="review_001",
        generation="generation_000001",
        frozen_by="compaction-planner",
    )
    assert frozen["status"] == "frozen"
    assert frozen["partition_count"] == 1
    with pytest.raises(ValueError, match="not open"):
        write_delta_partition(
            root,
            delta_run="review_001",
            generation="generation_000001",
            partition="late",
            editor="reviewer@example.org",
            instance_keys=[10],
            row_index_hints=[0],
            operation_codes=[KEYPOINT_OPERATION_CODE_MAP["replace_xy"]],
            revisions=[2],
            timestamp_ns=[124],
            reason_codes=[7],
            keypoint_index=[0],
            new_xy=[[1.0, 2.0]],
            valid=[True],
        )


def test_delta_refuses_stale_row_index_hint(tmp_path: Path) -> None:
    root = _base_root(tmp_path)
    create_delta_generation(
        root,
        delta_run="review_001",
        generation="generation_000001",
        generation_ordinal=1,
        target_kind="keypoints",
        base_run_path="refined_keypoints_runs/snapshot",
        created_by="reviewer",
    )
    with pytest.raises(ValueError, match="does not resolve"):
        write_delta_partition(
            root,
            delta_run="review_001",
            generation="generation_000001",
            partition="bad_hint",
            editor="reviewer",
            instance_keys=[20],
            row_index_hints=[2],
            operation_codes=[KEYPOINT_OPERATION_CODE_MAP["replace_xy"]],
            revisions=[1],
            timestamp_ns=[123],
            reason_codes=[0],
            keypoint_index=[0],
            new_xy=[[1.0, 2.0]],
            valid=[True],
        )


def test_keypoint_overlay_verifies_and_resolves_newest_edit(tmp_path: Path) -> None:
    root = _base_root(tmp_path)
    base = root["refined_keypoints_runs/snapshot"]
    base.create_array(
        "keypoints_roi",
        data=np.asarray(
            [
                [[1.0, 1.0], [2.0, 2.0]],
                [[3.0, 3.0], [4.0, 4.0]],
                [[5.0, 5.0], [6.0, 6.0]],
            ],
            dtype=np.float32,
        ),
        chunks=(3, 2, 2),
    )
    create_delta_generation(
        root,
        delta_run="review_001",
        generation="generation_000001",
        generation_ordinal=1,
        target_kind="keypoints",
        base_run_path="refined_keypoints_runs/snapshot",
        created_by="reviewer",
    )
    for partition, revision, xy in (
        ("edit_a", 1, [12.0, 13.0]),
        ("edit_b", 2, [22.0, 23.0]),
    ):
        write_delta_partition(
            root,
            delta_run="review_001",
            generation="generation_000001",
            partition=partition,
            editor="reviewer",
            instance_keys=[20],
            row_index_hints=[1],
            operation_codes=[KEYPOINT_OPERATION_CODE_MAP["replace_xy"]],
            revisions=[revision],
            timestamp_ns=[100 + revision],
            reason_codes=[7],
            keypoint_index=[1],
            new_xy=[xy],
            valid=[True],
            reason_code_map={"manual_correction": 7},
        )

    overlay = resolve_keypoint_delta_overlay(
        root,
        delta_run="review_001",
        generation="generation_000001",
        n_keypoints=2,
    )
    assert overlay.event_count == 2
    assert overlay.partition_count == 2
    assert overlay.max_revision == 2
    assert len(overlay.edits) == 1
    assert overlay.edits[0].new_xy == (22.0, 23.0)
    points, valid = apply_keypoint_delta_overlay(
        base["keypoints_roi"],
        instance_keys=base["instance_key"],
        overlay=overlay,
    )
    np.testing.assert_array_equal(points[1, 1], np.asarray([22.0, 23.0]))
    assert bool(valid[1, 1])
    np.testing.assert_array_equal(base["keypoints_roi"][1, 1], np.asarray([4.0, 4.0], dtype=np.float32))


def test_keypoint_overlay_rejects_recomputed_payload_tampering(tmp_path: Path) -> None:
    root = _base_root(tmp_path)
    create_delta_generation(
        root,
        delta_run="review_001",
        generation="generation_000001",
        generation_ordinal=1,
        target_kind="keypoints",
        base_run_path="refined_keypoints_runs/snapshot",
        created_by="reviewer",
    )
    write_delta_partition(
        root,
        delta_run="review_001",
        generation="generation_000001",
        partition="edit_a",
        editor="reviewer",
        instance_keys=[20],
        row_index_hints=[1],
        operation_codes=[KEYPOINT_OPERATION_CODE_MAP["replace_xy"]],
        revisions=[1],
        timestamp_ns=[101],
        reason_codes=[7],
        keypoint_index=[1],
        new_xy=[[12.0, 13.0]],
        valid=[True],
        reason_code_map={"manual_correction": 7},
    )
    partition = root["edit_delta_runs/review_001/generations/generation_000001/partitions/edit_a"]
    partition["new_xy"][0] = np.asarray([99.0, 100.0])
    with pytest.raises(ValueError, match="digest mismatch"):
        resolve_keypoint_delta_overlay(
            root,
            delta_run="review_001",
            generation="generation_000001",
            n_keypoints=2,
        )
