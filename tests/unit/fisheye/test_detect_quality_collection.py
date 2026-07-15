from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.refinement.detect_quality_collection import (
    BLIP,
    CLEAN,
    JUMP,
    COLLECTION_QUALITY_SCHEMA,
    TEMPORAL_POLICY_SCHEMA,
    reconcile_temporal_v2,
    run_collection_detect_quality,
)


def test_temporal_v2_accepts_sustained_relocation_across_boundary() -> None:
    frames = np.arange(6, dtype=np.int64)
    centroids = np.asarray(
        [[10, 10], [11, 10], [80, 80], [81, 80], [79, 81], [80, 79]],
        dtype=np.float32,
    )

    labels, summary = reconcile_temporal_v2(
        frames,
        centroids,
        jump_threshold_pixels=20.0,
        blip_gap_threshold=10,
        relocation_confirm_count=3,
        relocation_cluster_radius_pixels=10.0,
    )

    np.testing.assert_array_equal(labels, np.full(6, CLEAN, dtype=np.int8))
    assert summary["accepted_relocations"] == 1
    assert summary["jump_frames"].size == 0


def test_temporal_v2_retains_isolated_excursion_as_jump() -> None:
    labels, summary = reconcile_temporal_v2(
        np.arange(4, dtype=np.int64),
        np.asarray([[10, 10], [11, 10], [80, 80], [12, 10]], dtype=np.float32),
        jump_threshold_pixels=20.0,
        blip_gap_threshold=10,
        relocation_confirm_count=3,
        relocation_cluster_radius_pixels=10.0,
    )

    np.testing.assert_array_equal(labels, np.asarray([CLEAN, CLEAN, JUMP, CLEAN]))
    np.testing.assert_array_equal(summary["jump_frames"], np.asarray([2]))
    assert summary["accepted_relocations"] == 0


def test_temporal_v2_resets_baseline_after_long_gap() -> None:
    labels, summary = reconcile_temporal_v2(
        np.asarray([0, 1, 20, 21], dtype=np.int64),
        np.asarray([[10, 10], [11, 10], [80, 80], [81, 80]], dtype=np.float32),
        jump_threshold_pixels=20.0,
        blip_gap_threshold=10,
        relocation_confirm_count=3,
        relocation_cluster_radius_pixels=10.0,
    )

    np.testing.assert_array_equal(labels, np.asarray([CLEAN, CLEAN, BLIP, CLEAN]))
    np.testing.assert_array_equal(summary["blip_frames"], np.asarray([20]))


def _write_source(
    path: Path,
    *,
    duplicate_key: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    root = zarr.open_group(path, mode="w", zarr_format=3)
    root.attrs.update({"width": 100, "height": 100, "total_frames": 8})
    parent = root.create_group("detect_runs")
    parent.attrs["latest"] = "detect_source"
    source = parent.create_group("detect_source")
    frames = np.arange(6, dtype=np.int64)
    boxes = np.asarray(
        [
            [0.10, 0.10, 0.10, 0.10],
            [0.11, 0.10, 0.10, 0.10],
            [0.80, 0.80, 0.10, 0.10],
            [0.81, 0.80, 0.10, 0.10],
            [0.79, 0.81, 0.10, 0.10],
            [0.80, 0.79, 0.10, 0.10],
        ],
        dtype=np.float32,
    )
    keys = np.arange(100, 106, dtype=np.uint64)
    if duplicate_key:
        keys[-1] = keys[-2]
    source.create_array(
        "frame_indices",
        data=frames,
        chunks=(2,),
        shards=(4,),
    )
    source.create_array(
        "bbox_norm_coords",
        data=boxes,
        chunks=(2, 4),
        shards=(4, 4),
    )
    source.create_array(
        "instance_key",
        data=keys,
        chunks=(2,),
        shards=(4,),
    )
    return frames, keys


def _run(
    path: Path,
    *,
    output_run: str,
    workers: int,
    apply: bool = True,
) -> dict[str, object]:
    return run_collection_detect_quality(
        zarr_path=path,
        source_group_path="detect_runs/detect_source",
        output_run=output_run,
        recording_frame_count=8,
        width=100,
        height=100,
        jump_threshold=20,
        threshold_mode="pixels",
        blip_gap_threshold=10,
        relocation_confirm_count=3,
        relocation_cluster_radius_fraction=0.5,
        shard_rows=4,
        row_chunk_rows=2,
        frame_chunk_rows=2,
        workers=workers,
        work_dir=path.parent / "traces",
        apply=apply,
    )


def test_collection_quality_parallel_output_is_keyed_sharded_and_promoted(
    tmp_path: Path,
) -> None:
    path = tmp_path / "analysis.zarr"
    frames, source_keys = _write_source(path)

    result = _run(path, output_run="quality_parallel", workers=2)

    assert result["status"] == "complete"
    root = zarr.open_group(path, mode="r", use_consolidated=False)
    parent = root["detect_quality_runs"]
    assert parent.attrs["latest"] == "quality_parallel"
    assert parent.attrs["latest_complete"] == "quality_parallel"
    quality = parent["quality_parallel"]
    assert quality.attrs["schema_id"] == COLLECTION_QUALITY_SCHEMA
    assert quality.attrs["temporal_artifact_policy"] == TEMPORAL_POLICY_SCHEMA
    assert quality.attrs["palette_run_completion_status"] == "complete"
    assert quality["quality_flags"].shards == (4,)
    assert quality["detection_quality_labels"].shards == (4,)
    assert quality["instance_key"].shards == (4,)
    np.testing.assert_array_equal(quality["instance_key"][:], source_keys)
    np.testing.assert_array_equal(
        quality["detection_quality_labels"][:],
        quality["quality_flags"][:][frames],
    )
    np.testing.assert_array_equal(
        quality["quality_flags"][:],
        np.asarray([CLEAN, CLEAN, CLEAN, CLEAN, CLEAN, CLEAN, -1, -1], dtype=np.int8),
    )
    validation = quality.attrs["collection_quality_validation"]
    assert validation["status"] == "complete"
    assert validation["instance_key_exact"] is True


def test_parallel_and_serial_finalizers_are_bit_exact(tmp_path: Path) -> None:
    path = tmp_path / "analysis.zarr"
    _write_source(path)

    _run(path, output_run="quality_serial", workers=1)
    _run(path, output_run="quality_parallel", workers=2)

    root = zarr.open_group(path, mode="r", use_consolidated=False)
    serial = root["detect_quality_runs/quality_serial"]
    parallel = root["detect_quality_runs/quality_parallel"]
    for name in ("quality_flags", "detection_quality_labels", "instance_key"):
        np.testing.assert_array_equal(serial[name][:], parallel[name][:])
    assert serial.attrs["quality_score"] == parallel.attrs["quality_score"]
    assert serial.attrs["detection_quality_summary"] == parallel.attrs[
        "detection_quality_summary"
    ]


def test_collection_quality_duplicate_modern_keys_fail_before_promotion(
    tmp_path: Path,
) -> None:
    path = tmp_path / "analysis.zarr"
    _write_source(path, duplicate_key=True)

    with pytest.raises(ValueError, match="not unique"):
        _run(path, output_run="quality_bad", workers=2)

    root = zarr.open_group(path, mode="r", use_consolidated=False)
    assert "detect_quality_runs" not in root


def test_collection_quality_dry_run_does_not_mutate_archive(tmp_path: Path) -> None:
    path = tmp_path / "analysis.zarr"
    _write_source(path)

    result = _run(path, output_run="quality_plan", workers=2, apply=False)

    assert result["status"] == "planned"
    assert result["worker_tasks"] == 2
    root = zarr.open_group(path, mode="r", use_consolidated=False)
    assert "detect_quality_runs" not in root


def test_collection_quality_supports_empty_modern_detection_surface(
    tmp_path: Path,
) -> None:
    path = tmp_path / "analysis.zarr"
    root = zarr.open_group(path, mode="w", zarr_format=3)
    root.attrs.update({"width": 100, "height": 100, "total_frames": 4})
    source = root.create_group("detect_runs").create_group("empty")
    source.create_array(
        "frame_indices", data=np.empty((0,), dtype=np.int64), chunks=(2,), shards=(4,)
    )
    source.create_array(
        "bbox_norm_coords",
        data=np.empty((0, 4), dtype=np.float32),
        chunks=(2, 4),
        shards=(4, 4),
    )
    source.create_array(
        "instance_key", data=np.empty((0,), dtype=np.uint64), chunks=(2,), shards=(4,)
    )

    result = run_collection_detect_quality(
        zarr_path=path,
        source_group_path="detect_runs/empty",
        output_run="quality_empty",
        recording_frame_count=4,
        width=100,
        height=100,
        shard_rows=4,
        row_chunk_rows=2,
        frame_chunk_rows=2,
        workers=2,
        apply=True,
    )

    assert result["status"] == "complete"
    quality = zarr.open_group(path, mode="r", use_consolidated=False)[
        "detect_quality_runs/quality_empty"
    ]
    np.testing.assert_array_equal(
        quality["quality_flags"][:], np.full(4, -1, dtype=np.int8)
    )
    assert quality["detection_quality_labels"].shape == (0,)
    assert quality["instance_key"].shape == (0,)


def test_collection_quality_multi_subject_skips_global_temporal_labels(
    tmp_path: Path,
) -> None:
    path = tmp_path / "analysis.zarr"
    root = zarr.open_group(path, mode="w", zarr_format=3)
    root.attrs.update({"width": 100, "height": 100, "total_frames": 4})
    source = root.create_group("detect_runs").create_group("multi")
    frames = np.asarray([0, 0, 1, 1, 2, 2, 2], dtype=np.int64)
    boxes = np.asarray(
        [
            [0.1, 0.1, 0.1, 0.1],
            [0.9, 0.9, 0.1, 0.1],
            [0.2, 0.8, 0.1, 0.1],
            [0.8, 0.2, 0.1, 0.1],
            [0.1, 0.9, 0.1, 0.1],
            [0.5, 0.5, 0.1, 0.1],
            [0.9, 0.1, 0.1, 0.1],
        ],
        dtype=np.float32,
    )
    keys = np.arange(200, 207, dtype=np.uint64)
    source.create_array("frame_indices", data=frames, chunks=(2,), shards=(4,))
    source.create_array(
        "bbox_norm_coords", data=boxes, chunks=(2, 4), shards=(4, 4)
    )
    source.create_array("instance_key", data=keys, chunks=(2,), shards=(4,))

    result = run_collection_detect_quality(
        zarr_path=path,
        source_group_path="detect_runs/multi",
        output_run="quality_multi",
        recording_frame_count=4,
        width=100,
        height=100,
        expected_subject_count=2,
        shard_rows=4,
        row_chunk_rows=2,
        frame_chunk_rows=2,
        workers=2,
        apply=True,
    )

    assert result["status"] == "complete"
    quality = zarr.open_group(path, mode="r", use_consolidated=False)[
        "detect_quality_runs/quality_multi"
    ]
    np.testing.assert_array_equal(
        quality["quality_flags"][:], np.asarray([CLEAN, CLEAN, 4, -1], dtype=np.int8)
    )
    assert quality.attrs["artifact_summary"]["temporal_artifact_policy"] == (
        "skipped_expected_subject_count_gt_1"
    )
