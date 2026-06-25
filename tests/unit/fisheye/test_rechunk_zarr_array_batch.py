from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from fisheye.utils.rechunk_zarr_array_batch import (
    find_batch_candidates,
    rechunk_zarr_array_batch,
)


def _write_array(group: zarr.Group, name: str, data: np.ndarray, chunks: tuple[int, ...]) -> None:
    group.create_array(name, data=data, chunks=chunks, overwrite=True)


def _make_store(path: Path) -> Path:
    zarr_path = path / "sample.zarr"
    root = zarr.open_group(str(zarr_path), mode="w", zarr_format=3)
    crop = root.require_group("crop_runs").require_group("crop_a")
    _write_array(crop, "frame_indices", np.arange(20, dtype=np.int32), (5,))
    _write_array(crop, "source_refined_row_ids", np.arange(20, dtype=np.int64), (5,))
    _write_array(crop, "roi_images", np.zeros((20, 8, 8), dtype=np.uint8), (5, 8, 8))

    keypoints = root.require_group("keypoints_runs").require_group("kp_a")
    _write_array(keypoints, "frame_indices", np.arange(20, dtype=np.int32), (5,))
    _write_array(keypoints, "keypoints_img", np.zeros((20, 5, 2), dtype=np.float32), (5, 5, 2))

    refined = root.require_group("refined_keypoints_runs").require_group("rkp_a")
    _write_array(refined, "source_detect_row_index", np.arange(20, dtype=np.int32), (5,))
    _write_array(refined, "keypoints_img", np.zeros((20, 5, 2), dtype=np.float32), (5, 5, 2))

    detect = root.require_group("detect_runs").require_group("detect_a")
    _write_array(detect, "frame_counts", np.zeros((20,), dtype=np.int32), (5,))
    _write_array(detect, "bbox_norm_coords", np.zeros((20, 4), dtype=np.float32), (5, 4))

    track = (
        root.require_group("analysis")
        .require_group("track_kinematics_runs")
        .require_group("offline")
        .require_group("run_a")
        .require_group("tracks")
        .require_group("id_0")
    )
    _write_array(track, "frame_indices", np.arange(20, dtype=np.int32), (5,))
    _write_array(track, "positions_px", np.zeros((20, 2), dtype=np.float32), (5, 2))
    _write_array(track, "speed_raw_px", np.zeros((20,), dtype=np.float32), (5,))
    return zarr_path


def test_find_batch_candidates_matches_only_crimson_lineage_arrays(tmp_path: Path) -> None:
    zarr_path = _make_store(tmp_path)

    candidates = find_batch_candidates(
        zarr_path,
        preset="crimson-lineage-v1",
        row_chunk=16,
    )
    paths = [row.array_path for row in candidates]

    assert paths == [
        "crop_runs/crop_a/frame_indices",
        "crop_runs/crop_a/source_refined_row_ids",
        "detect_runs/detect_a/frame_counts",
        "keypoints_runs/kp_a/frame_indices",
        "refined_keypoints_runs/rkp_a/source_detect_row_index",
    ]
    assert "crop_runs/crop_a/roi_images" not in paths
    assert "keypoints_runs/kp_a/keypoints_img" not in paths
    assert "detect_runs/detect_a/bbox_norm_coords" not in paths


def test_rechunk_zarr_array_batch_applies_allowlisted_arrays(tmp_path: Path) -> None:
    zarr_path = _make_store(tmp_path)

    summary = rechunk_zarr_array_batch(
        zarr_path,
        preset="crimson-lineage-v1",
        row_chunk=16,
        reason="unit batch",
        apply=True,
    )

    assert summary.matched_count == 5
    assert summary.updated_count == 5
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    assert root["crop_runs/crop_a/frame_indices"].chunks == (16,)
    assert root["crop_runs/crop_a/source_refined_row_ids"].chunks == (16,)
    assert root["keypoints_runs/kp_a/frame_indices"].chunks == (16,)
    assert root["refined_keypoints_runs/rkp_a/source_detect_row_index"].chunks == (16,)
    assert root["detect_runs/detect_a/frame_counts"].chunks == (16,)
    assert root["crop_runs/crop_a/roi_images"].chunks == (5, 8, 8)
    assert root["keypoints_runs/kp_a/keypoints_img"].chunks == (5, 5, 2)
    assert root["detect_runs/detect_a/bbox_norm_coords"].chunks == (5, 4)
    assert root["crop_runs/crop_a/frame_indices"].attrs["rechunk_provenance"]["reason"] == "unit batch"


def test_track_kinematics_preset_matches_only_track_arrays(tmp_path: Path) -> None:
    zarr_path = _make_store(tmp_path)

    candidates = find_batch_candidates(
        zarr_path,
        preset="track-kinematics-v1",
        row_chunk=16,
    )
    paths = [row.array_path for row in candidates]

    assert paths == [
        "analysis/track_kinematics_runs/offline/run_a/tracks/id_0/frame_indices",
        "analysis/track_kinematics_runs/offline/run_a/tracks/id_0/positions_px",
        "analysis/track_kinematics_runs/offline/run_a/tracks/id_0/speed_raw_px",
    ]
    assert "crop_runs/crop_a/frame_indices" not in paths
    assert "keypoints_runs/kp_a/frame_indices" not in paths


def test_track_kinematics_preset_applies_only_track_arrays(tmp_path: Path) -> None:
    zarr_path = _make_store(tmp_path)

    summary = rechunk_zarr_array_batch(
        zarr_path,
        preset="track-kinematics-v1",
        row_chunk=16,
        reason="track batch",
        apply=True,
    )

    assert summary.matched_count == 3
    assert summary.updated_count == 3
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    assert root[
        "analysis/track_kinematics_runs/offline/run_a/tracks/id_0/frame_indices"
    ].chunks == (16,)
    assert root[
        "analysis/track_kinematics_runs/offline/run_a/tracks/id_0/positions_px"
    ].chunks == (16, 2)
    assert root["crop_runs/crop_a/frame_indices"].chunks == (5,)
    assert root["keypoints_runs/kp_a/frame_indices"].chunks == (5,)
