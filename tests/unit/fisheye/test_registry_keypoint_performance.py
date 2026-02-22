"""Unit tests for keypoint performance registry extraction and latest views."""

from pathlib import Path
import sys

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.registry.db import Registry


def _create_keypoint_archive(path: Path, *, session_uuid: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(path), mode="w")
    root.attrs["session_uuid"] = session_uuid
    raw = root.create_group("raw_video")
    raw.create_array("images_ds", data=np.zeros((4, 8, 8), dtype=np.uint8), chunks=(1, 8, 8))

    keypoint_parent = root.create_group("keypoints_runs")
    keypoint_parent.attrs["latest"] = "keypoints_new"

    keypoint_old = keypoint_parent.create_group("keypoints_old")
    keypoint_old.attrs["keypoints_timestamp_utc"] = "2026-02-08T00:00:00+00:00"
    keypoint_old.attrs["method"] = "traditional_pose"
    keypoint_old.attrs["source_crop_run"] = "crop_001"
    keypoint_old.attrs["source_detect_run"] = "refined_detect_001"
    keypoint_old.attrs["keypoints_processed"] = 4
    keypoint_old.attrs["success_rate"] = 50.0
    keypoint_old.attrs["duration_seconds"] = 2.0
    keypoint_old.attrs["summary_statistics"] = {
        "total_rois": 4,
        "successful_detections": 2,
        "failed_detections": 2,
        "success_rate_percent": 50.0,
    }
    keypoint_old.create_array("keypoints_roi", data=np.zeros((4, 3, 2), dtype=np.float32), chunks=(1, 3, 2))

    keypoint_new = keypoint_parent.create_group("keypoints_new")
    keypoint_new.attrs["keypoints_timestamp_utc"] = "2026-02-09T00:00:00+00:00"
    keypoint_new.attrs["method"] = "yolo_pose"
    keypoint_new.attrs["source_crop_run"] = "crop_002"
    keypoint_new.attrs["source_detect_run"] = "detect_002"
    keypoint_new.attrs["source_refined_run"] = "refined_detect_002"
    keypoint_new.attrs["keypoints_processed"] = 4
    keypoint_new.attrs["inference_duration_seconds"] = 1.0
    keypoint_new.attrs["inference_average_fps"] = 4.0
    keypoint_new.attrs["inference_poses_per_second"] = 4.0
    keypoint_new.attrs["model_resolution_selected_run_id"] = "pose_run_001"
    keypoint_new.attrs["model_resolution_selected_set_id"] = "pose_set_001"
    keypoint_new.attrs["model_resolution_selected_model_path"] = "/tmp/models/pose_best.pt"
    keypoint_new.attrs["summary_statistics"] = {
        "total_rois": 4,
        "successful_detections": 3,
        "failed_detections": 1,
        "success_rate_percent": 75.0,
        "mean_confidence": 0.88,
    }
    keypoint_new.attrs["parameters"] = {
        "batch_size": 256,
        "imgsz": 256,
        "confidence_threshold": 0.25,
        "iou_threshold": 0.5,
    }
    keypoint_new.create_array("keypoints_roi", data=np.zeros((4, 3, 2), dtype=np.float32), chunks=(1, 3, 2))


def _create_keypointless_archive(path: Path, *, session_uuid: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(path), mode="w")
    root.attrs["session_uuid"] = session_uuid
    raw = root.create_group("raw_video")
    raw.create_array("images_ds", data=np.zeros((4, 8, 8), dtype=np.uint8), chunks=(1, 8, 8))


def test_register_from_root_populates_keypoint_performance_all_runs_and_latest(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    zarr_path = tmp_path / "recordings" / "rec_pose_a" / "zarr" / "rec_pose_a_analysis.zarr"
    _create_keypoint_archive(zarr_path, session_uuid="rec_pose_a_uuid")

    dataset_id = registry.register_from_root(zarr.open_group(str(zarr_path), mode="r"), zarr_path)

    count = registry.conn.execute(
        "SELECT COUNT(*) AS n FROM keypoint_performance WHERE dataset_id = ?;",
        (dataset_id,),
    ).fetchone()
    assert count is not None
    assert int(count["n"]) == 2

    latest = registry.conn.execute(
        """
        SELECT
            keypoint_run,
            keypoint_method,
            model_run_id,
            model_set_id,
            model_name,
            total_rois,
            success_rate_percent,
            keypoints_per_second,
            conf_threshold,
            iou_threshold
        FROM keypoint_performance_latest
        WHERE dataset_id = ?;
        """,
        (dataset_id,),
    ).fetchone()
    assert latest is not None
    assert str(latest["keypoint_run"]) == "keypoints_new"
    assert str(latest["keypoint_method"]) == "yolo_pose"
    assert str(latest["model_run_id"]) == "pose_run_001"
    assert str(latest["model_set_id"]) == "pose_set_001"
    assert str(latest["model_name"]) == "pose_best.pt"
    assert int(latest["total_rois"]) == 4
    assert float(latest["success_rate_percent"]) == 75.0
    assert float(latest["keypoints_per_second"]) == 4.0
    assert float(latest["conf_threshold"]) == 0.25
    assert float(latest["iou_threshold"]) == 0.5

    rec_latest = registry.conn.execute(
        """
        SELECT recording_id, dataset_id, keypoint_run
        FROM recording_keypoint_performance_latest
        WHERE recording_id = ?;
        """,
        ("rec_pose_a_uuid",),
    ).fetchone()
    assert rec_latest is not None
    assert str(rec_latest["dataset_id"]) == dataset_id
    assert str(rec_latest["keypoint_run"]) == "keypoints_new"
    registry.close()


def test_register_from_root_handles_archive_without_keypoint_runs(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    zarr_path = tmp_path / "recordings" / "rec_pose_none" / "zarr" / "rec_pose_none_analysis.zarr"
    _create_keypointless_archive(zarr_path, session_uuid="rec_pose_none_uuid")

    dataset_id = registry.register_from_root(zarr.open_group(str(zarr_path), mode="r"), zarr_path)
    count = registry.conn.execute(
        "SELECT COUNT(*) AS n FROM keypoint_performance WHERE dataset_id = ?;",
        (dataset_id,),
    ).fetchone()
    assert count is not None
    assert int(count["n"]) == 0
    registry.close()
