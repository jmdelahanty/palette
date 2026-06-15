"""Unit tests for keypoint performance registry extraction and latest views."""

import json
from pathlib import Path
import sys

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.registry.db import Registry, _extract_keypoint_performance_rows


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
    keypoint_old.attrs["source_crop_storage_mode"] = "materialized"
    keypoint_old.attrs["source_crop_signature"] = "crop_sig_old"
    keypoint_old.attrs["source_crop_revision"] = 1
    keypoint_old.attrs["source_roi_image_representation"] = "uint8_grayscale_roi_v1"
    keypoint_old.attrs["source_roi_pixel_contract"] = {
        "name": "orange_mono_pynvvc_luma_uint8_v1",
        "image_representation": "uint8_grayscale_roi_v1",
    }
    keypoint_old.attrs["source_roi_pixel_contract_name"] = "orange_mono_pynvvc_luma_uint8_v1"
    keypoint_old.attrs["source_roi_read_mode"] = "materialized_crop_run"
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
    keypoint_new.attrs["source_crop_storage_mode"] = "geometry_only"
    keypoint_new.attrs["source_crop_signature"] = "crop_sig_new"
    keypoint_new.attrs["source_crop_revision"] = 2
    keypoint_new.attrs["source_roi_image_representation"] = "nv12_luma_plane_uint8"
    keypoint_new.attrs["source_roi_pixel_contract"] = {
        "name": "nv12_luma_plane_uint8",
        "image_representation": "nv12_luma_plane_uint8",
    }
    keypoint_new.attrs["source_roi_pixel_contract_name"] = "nv12_luma_plane_uint8"
    keypoint_new.attrs["source_roi_read_mode"] = "flat_bin_roi_cache"
    keypoint_new.attrs["roi_cache_policy"] = "required"
    keypoint_new.attrs["source_roi_cache_used"] = True
    keypoint_new.attrs["source_roi_cache_backend"] = "pynvvc_luma"
    keypoint_new.attrs["source_roi_live_acceleration_effective"] = "gpu"
    keypoint_new.attrs["source_roi_live_gpu_chunk_frames"] = 384
    keypoint_new.attrs["input_mode_requested"] = "auto"
    keypoint_new.attrs["input_mode_effective"] = "tensor"
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
            iou_threshold,
            source_crop_storage_mode,
            source_crop_signature,
            source_crop_revision,
            source_roi_image_representation,
            source_roi_pixel_contract_name,
            source_roi_pixel_contract_json,
            source_roi_read_mode,
            roi_cache_policy,
            source_roi_cache_used,
            source_roi_cache_backend,
            source_roi_live_acceleration_effective,
            source_roi_live_gpu_chunk_frames,
            input_mode_requested,
            input_mode_effective
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
    assert str(latest["source_crop_storage_mode"]) == "geometry_only"
    assert str(latest["source_crop_signature"]) == "crop_sig_new"
    assert int(latest["source_crop_revision"]) == 2
    assert str(latest["source_roi_image_representation"]) == "nv12_luma_plane_uint8"
    assert str(latest["source_roi_pixel_contract_name"]) == "nv12_luma_plane_uint8"
    assert json.loads(str(latest["source_roi_pixel_contract_json"]))["name"] == "nv12_luma_plane_uint8"
    assert str(latest["source_roi_read_mode"]) == "flat_bin_roi_cache"
    assert str(latest["roi_cache_policy"]) == "required"
    assert int(latest["source_roi_cache_used"]) == 1
    assert str(latest["source_roi_cache_backend"]) == "pynvvc_luma"
    assert str(latest["source_roi_live_acceleration_effective"]) == "gpu"
    assert int(latest["source_roi_live_gpu_chunk_frames"]) == 384
    assert str(latest["input_mode_requested"]) == "auto"
    assert str(latest["input_mode_effective"]) == "tensor"

    old_row = registry.conn.execute(
        """
        SELECT source_crop_storage_mode, source_roi_pixel_contract_name, source_roi_read_mode
        FROM keypoint_performance
        WHERE dataset_id = ? AND keypoint_run = 'keypoints_old';
        """,
        (dataset_id,),
    ).fetchone()
    assert old_row is not None
    assert str(old_row["source_crop_storage_mode"]) == "materialized"
    assert str(old_row["source_roi_pixel_contract_name"]) == "orange_mono_pynvvc_luma_uint8_v1"
    assert str(old_row["source_roi_read_mode"]) == "materialized_crop_run"

    rec_latest = registry.conn.execute(
        """
        SELECT recording_id, dataset_id, keypoint_run, source_roi_pixel_contract_name, input_mode_effective
        FROM recording_keypoint_performance_latest
        WHERE recording_id = ?;
        """,
        ("rec_pose_a_uuid",),
    ).fetchone()
    assert rec_latest is not None
    assert str(rec_latest["dataset_id"]) == dataset_id
    assert str(rec_latest["keypoint_run"]) == "keypoints_new"
    assert str(rec_latest["source_roi_pixel_contract_name"]) == "nv12_luma_plane_uint8"
    assert str(rec_latest["input_mode_effective"]) == "tensor"
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


def test_extract_keypoint_performance_rows_prefers_created_at_utc(tmp_path: Path) -> None:
    zarr_path = tmp_path / "keypoint_created_at_analysis.zarr"
    _create_keypoint_archive(zarr_path, session_uuid="keypoint_created_at_uuid")
    root = zarr.open_group(str(zarr_path), mode="a")
    run = root["keypoints_runs"]["keypoints_new"]
    run.attrs["created_at_utc"] = "2026-02-09T00:10:00+00:00"
    run.attrs["created_utc"] = "2026-02-09T00:08:00+00:00"
    run.attrs["timestamp_utc"] = "2026-02-09T00:07:00+00:00"
    run.attrs["provenance"] = {"created_at_utc": "2026-02-09T00:06:00+00:00"}

    rows = _extract_keypoint_performance_rows(
        zarr.open_group(str(zarr_path), mode="r"),
        zarr_path=zarr_path,
        recording_id="keypoint_created_at_uuid",
        zarr_use="analysis",
    )

    latest_row = next(row for row in rows if str(row["keypoint_run"]) == "keypoints_new")
    assert latest_row["keypoint_created_utc"] == "2026-02-09T00:10:00+00:00"


def test_extract_keypoint_performance_rows_reads_pixel_contract_from_provenance_inputs(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "keypoint_provenance_inputs_analysis.zarr"
    _create_keypoint_archive(zarr_path, session_uuid="keypoint_provenance_inputs_uuid")
    root = zarr.open_group(str(zarr_path), mode="a")
    run = root["keypoints_runs"]["keypoints_new"]
    for key in (
        "source_roi_pixel_contract",
        "source_roi_pixel_contract_name",
        "source_roi_read_mode",
        "source_roi_cache_backend",
        "source_roi_cache_used",
        "input_mode_effective",
    ):
        if key in run.attrs:
            del run.attrs[key]
    run.attrs["provenance"] = {
        "inputs": {
            "source_roi_pixel_contract": {
                "name": "orange_mono_pynvvc_luma_uint8_v1",
                "image_representation": "uint8_grayscale_roi_v1",
            },
            "source_roi_read_mode": "materialized_crop_run",
            "roi_cache_used": False,
            "roi_cache_backend": "none",
            "input_mode_effective": "numpy-list",
        }
    }

    rows = _extract_keypoint_performance_rows(
        zarr.open_group(str(zarr_path), mode="r"),
        zarr_path=zarr_path,
        recording_id="keypoint_provenance_inputs_uuid",
        zarr_use="analysis",
    )

    latest_row = next(row for row in rows if str(row["keypoint_run"]) == "keypoints_new")
    assert latest_row["source_roi_pixel_contract_name"] == "orange_mono_pynvvc_luma_uint8_v1"
    assert json.loads(str(latest_row["source_roi_pixel_contract_json"]))["name"] == (
        "orange_mono_pynvvc_luma_uint8_v1"
    )
    assert latest_row["source_roi_read_mode"] == "materialized_crop_run"
    assert latest_row["source_roi_cache_used"] == 0
    assert latest_row["source_roi_cache_backend"] == "none"
    assert latest_row["input_mode_effective"] == "numpy-list"
