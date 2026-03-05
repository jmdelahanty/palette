"""Unit tests for registry maintenance helpers."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import sys
from typing import Dict, Optional

import numpy as np
import pytest
import zarr

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.registry.db import Registry
from fisheye.registry.maintenance import (
    _backfill_keypoint_profiles,
    _backfill_eye_mask_profiles,
    _backfill_crop_quality,
    _backfill_dataset_lineage,
    _backfill_detect_quality,
    _backfill_detect_performance,
    _backfill_eye_mask_quality,
    _backfill_eye_mask_performance,
    _backfill_keypoint_performance,
    _backfill_recording_step_status,
    _backfill_recording_entities,
    _backfill_subject_dish_cross_entities,
    _backfill_subjects,
    _backfill_keypoint_quality,
    _backfill_model_tables,
    _remap_training_set_dataset_ids,
    _check_registry_integrity,
    _summarize_dataset_lineage_audit,
    _collect_empty_training_set_candidates,
    _collect_failed_run_candidates,
    _collect_invalid_dataset_candidates,
    _collect_missing_dataset_candidates,
    _collect_stale_in_progress_run_candidates,
    _delete_training_set_ids,
    _delete_training_run_ids,
    _delete_dataset_ids,
    _is_nested_zarr_subpath,
    _normalize_set_ids,
    _collect_set_delete_candidates,
    _collect_run_ids_for_set_ids,
    _collect_set_artifact_paths,
    _build_file_delete_plan,
    _collect_run_artifact_paths,
    _delete_paths,
    _is_safe_artifact_path,
    _normalize_run_ids,
    _resolve_existing_run_ids,
    _reconcile_stale_in_progress_runs,
    _normalize_status_values,
    main as maintenance_main,
)


def _create_quality_zarr(path: Path) -> None:
    root = zarr.open_group(str(path), mode="w")
    root.attrs["session_uuid"] = "quality_session"
    raw = root.create_group("raw_video")
    raw.create_array("images_ds", data=np.zeros((4, 8, 8), dtype=np.uint8), chunks=(1, 8, 8))
    crop_parent = root.create_group("crop_runs")
    crop = crop_parent.create_group("crop_001")
    crop.attrs["detection_source_type"] = "filtered"
    crop.create_array("roi_images", data=np.zeros((4, 8, 8), dtype=np.uint8), chunks=(1, 8, 8))
    kp_parent = root.create_group("keypoints_runs")
    kp = kp_parent.create_group("kp_001")
    kp.attrs["method"] = "traditional_pose"
    kp.attrs["source_crop_run"] = "crop_001"
    kp.attrs["success_rate"] = 0.75
    kp.create_array("keypoints_roi", data=np.zeros((4, 3, 2), dtype=np.float32), chunks=(1, 3, 2))
    refined_parent = root.create_group("refined_keypoints_runs")
    refined = refined_parent.create_group("refined_001")
    refined.attrs["source_keypoints_run"] = "kp_001"
    refined.attrs["created_utc"] = "2026-02-08T00:00:00+00:00"
    refined.attrs["keypoint_review_status"] = {
        "state": "approved",
        "intended_use": "training",
        "reviewer": "pytest",
        "timestamp_utc": "2026-02-08T00:00:00+00:00",
    }
    refined.create_array(
        "usable_keypoints",
        data=np.array([True, True, True, False], dtype=np.bool_),
        chunks=(4,),
    )


def _create_detect_performance_zarr(path: Path) -> None:
    root = zarr.open_group(str(path), mode="w")
    root.attrs["session_uuid"] = "detect_perf_session"
    raw = root.create_group("raw_video")
    raw.create_array("images_ds", data=np.zeros((4, 8, 8), dtype=np.uint8), chunks=(1, 8, 8))
    detect_parent = root.create_group("detect_runs")
    detect_parent.attrs["latest"] = "detect_001"
    detect = detect_parent.create_group("detect_001")
    detect.attrs["detect_timestamp_utc"] = "2026-02-09T00:00:00+00:00"
    detect.attrs["detection_method"] = "yolo"
    detect.attrs["model_path"] = "/tmp/model.pt"
    detect.attrs["model_name"] = "model.pt"
    detect.attrs["inference_average_fps"] = 80.0
    detect.attrs["inference_avg_read_ms"] = 120.0
    detect.attrs["parameters"] = {"conf_threshold": 0.4, "iou_threshold": 0.8, "batch_size": 16}
    detect.create_array("frame_counts", data=np.array([1, 0, 1, 0], dtype=np.int32), chunks=(4,))


def _create_keypoint_performance_zarr(path: Path) -> None:
    root = zarr.open_group(str(path), mode="w")
    root.attrs["session_uuid"] = "keypoint_perf_session"
    raw = root.create_group("raw_video")
    raw.create_array("images_ds", data=np.zeros((4, 8, 8), dtype=np.uint8), chunks=(1, 8, 8))

    kp_parent = root.create_group("keypoints_runs")
    kp_parent.attrs["latest"] = "keypoints_001"
    kp = kp_parent.create_group("keypoints_001")
    kp.attrs["keypoints_timestamp_utc"] = "2026-02-09T01:00:00+00:00"
    kp.attrs["method"] = "yolo_pose"
    kp.attrs["model_path"] = "/tmp/pose.pt"
    kp.attrs["source_crop_run"] = "crop_001"
    kp.attrs["source_detect_run"] = "detect_001"
    kp.attrs["source_refined_run"] = "refined_001"
    kp.attrs["duration_seconds"] = 2.0
    kp.attrs["inference_duration_seconds"] = 1.5
    kp.attrs["inference_poses_per_second"] = 2.0
    kp.attrs["inference_average_fps"] = 5.0
    kp.attrs["parameters"] = {
        "batch_size": 16,
        "imgsz": 640,
        "conf_threshold": 0.3,
        "iou_threshold": 0.7,
    }
    kp.attrs["summary_statistics"] = {
        "total_rois": 4,
        "successful_detections": 3,
        "failed_detections": 1,
        "success_rate_percent": 75.0,
        "frames_with_keypoints": 3,
        "mean_confidence": 0.91,
    }
    kp.create_array("keypoints_roi", data=np.zeros((4, 3, 2), dtype=np.float32), chunks=(1, 3, 2))


def _create_keypoint_profile_zarr(
    path: Path,
    *,
    profile_run: str = "keypoint_profile_001",
    zarr_use: str = "analysis",
    usable_rate: float = 0.75,
) -> object:
    _create_fake_zarr_store(path)
    root = _FakeGroup(attrs={"session_uuid": "keypoint_profile_session", "zarr_purpose": zarr_use})
    analysis = root.add_group("analysis")
    profile_parent = analysis.add_group("keypoint_profile_runs", attrs={"latest": profile_run})
    profile = profile_parent.add_group(profile_run)
    profile.attrs["profile_summary"] = {
        "created_at_utc": "2026-02-24T00:00:00+00:00",
        "dataset": {"recording_id": "recording_profile", "zarr_use": zarr_use},
        "source": {
            "keypoint_method": "traditional_pose",
            "keypoint_path": "keypoints_runs/keypoints_001",
            "keypoint_run": "keypoints_001",
            "skeleton_id": "fish_v1",
            "kpt_shape": [3, 3],
        },
        "quality": {
            "rows_total": 4,
            "rows_usable": 3,
            "usable_keypoints_total": 9,
            "usable_rate": usable_rate,
            "confidence_valid_rate": 0.95,
            "geometry_valid_rate": 0.96,
        },
        "geometry": {
            "triangle_area": {"stats": {"p10": 0.1, "p50": 0.2, "p90": 0.3}},
            "min_angle": {"stats": {"p10": 10.0, "p50": 20.0, "p90": 30.0}},
            "heading": {"stats": {"p10": -0.1, "p50": 0.0, "p90": 0.1}},
        },
        "composition": {
            "rig_id": "omnifin0",
            "camera_id": "2010094",
            "arena_id": "arena_2",
            "dish_design": "cedar",
            "canvas_name": "shadow",
            "protocol_name": "DefaultScreen",
            "genotype": "Tg(elavl3:gcamp7f)",
            "dpf_at_acquisition": 7,
        },
    }
    return root


def _ensure_eye_mask_data_profile_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS eye_mask_data_profile (
            dataset_id TEXT NOT NULL,
            profile_run TEXT NOT NULL,
            recording_id TEXT,
            zarr_use TEXT,
            eye_mask_method TEXT,
            source_eye_masks_run TEXT,
            source_refined_eye_masks_run TEXT,
            source_crop_run TEXT,
            source_keypoints_run TEXT,
            profile_created_utc TEXT,
            rows_total INTEGER,
            rows_usable INTEGER,
            usable_rate REAL,
            profile_json TEXT,
            genotype TEXT,
            dpf_at_acquisition INTEGER,
            zarr_mtime_ns INTEGER,
            updated_utc TEXT,
            PRIMARY KEY (dataset_id, profile_run)
        );
        """
    )
    conn.commit()


def _eye_mask_profile_row_payload(
    *,
    profile_run: str,
    usable_rate: float,
) -> Dict[str, object]:
    return {
        "profile_run": profile_run,
        "recording_id": "recording_eye_profile",
        "zarr_use": "analysis",
        "eye_mask_method": "traditional_eye_segmentation",
        "source_eye_masks_run": "eye_masks_001",
        "source_refined_eye_masks_run": "refined_eye_masks_001",
        "source_crop_run": "crop_001",
        "source_keypoints_run": "keypoints_001",
        "profile_created_utc": "2026-02-24T00:00:00+00:00",
        "rows_total": 10,
        "rows_usable": 8,
        "usable_rate": usable_rate,
        "profile_json": json.dumps({"usable_rate": usable_rate}, sort_keys=True),
        "genotype": "Tg(elavl3:gcamp7f)",
        "dpf_at_acquisition": 7,
        "zarr_mtime_ns": 123,
        "updated_utc": "2026-02-25T00:00:00+00:00",
    }


def _create_crop_quality_zarr(path: Path) -> None:
    root = zarr.open_group(str(path), mode="w")
    root.attrs["session_uuid"] = "crop_quality_session"
    raw = root.create_group("raw_video")
    raw.create_array("images_ds", data=np.zeros((4, 8, 8), dtype=np.uint8), chunks=(1, 8, 8))
    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "crop_001"
    crop = crop_parent.create_group("crop_001")
    crop.attrs["created_at_utc"] = "2026-02-10T00:00:00+00:00"
    crop.attrs["detection_source_type"] = "manual"
    crop.attrs["detection_source_path"] = "refined_detect_runs/refined_001/manual"
    crop.attrs["source_detect_run"] = "detect_001"
    crop.attrs["source_refined_run"] = "refined_001"
    crop.attrs["includes_interpolated"] = True
    crop.attrs["n_real_detections"] = 3
    crop.attrs["n_interpolated_detections"] = 1
    crop.attrs["summary_statistics"] = {
        "total_frames": 4,
        "frames_with_crops": 4,
        "total_rois_cropped": 4,
        "percent_frames_with_crops": 100.0,
    }
    crop.attrs["crop_review_status"] = {
        "state": "approved",
        "method": "manual",
        "intended_use": "training",
        "reviewer": "pytest",
        "timestamp": "2026-02-10T01:00:00+00:00",
        "notes": "ok",
    }
    crop.create_array("frame_counts", data=np.array([1, 1, 1, 1], dtype=np.int32), chunks=(4,))
    crop.create_array("bbox_norm_coords", data=np.zeros((4, 4), dtype=np.float32), chunks=(4, 4))
    crop.create_array("detection_source", data=np.array([0, 1, 0, 0], dtype=np.int8), chunks=(4,))


def _create_eye_mask_performance_zarr(path: Path) -> None:
    root = zarr.open_group(str(path), mode="w")
    root.attrs["session_uuid"] = "eye_mask_perf_session"
    raw = root.create_group("raw_video")
    raw.create_array("images_ds", data=np.zeros((4, 8, 8), dtype=np.uint8), chunks=(1, 8, 8))

    eye_parent = root.create_group("eye_masks_runs")
    eye_parent.attrs["latest"] = "eye_masks_001"
    eye_run = eye_parent.create_group("eye_masks_001")
    eye_run.attrs["created_utc"] = "2026-02-11T00:00:00+00:00"
    eye_run.attrs["method"] = "traditional_eye_segmentation"
    eye_run.attrs["source_crop_run"] = "crop_001"
    eye_run.attrs["source_keypoint_group"] = "keypoints_runs"
    eye_run.attrs["source_keypoints_run"] = "kp_001"
    eye_run.attrs["total_rois"] = 4
    eye_run.attrs["successful_eyes"] = 6
    eye_run.attrs["successful_roi_pairs"] = 3
    eye_run.attrs["successful_roi_pair_rate"] = 0.75
    eye_run.attrs["duration_seconds"] = 2.0
    eye_run.attrs["reason_counts"] = {"clean": 3, "too_close": 1}
    eye_run.attrs["summary_statistics"] = {"segmenter": {"successful_roi_pairs": 3}}

    refined_parent = root.create_group("refined_eye_masks_runs")
    refined_parent.attrs["latest"] = "refined_eye_masks_001"
    refined_run = refined_parent.create_group("refined_eye_masks_001")
    refined_run.attrs["created_utc"] = "2026-02-11T00:10:00+00:00"
    refined_run.attrs["method"] = "refine_eye_masks"
    refined_run.attrs["source_eye_masks_run"] = "eye_masks_001"
    refined_run.attrs["source_eye_masks_method"] = "traditional_eye_segmentation"
    refined_run.attrs["source_crop_run"] = "crop_001"
    refined_run.attrs["source_keypoint_group"] = "keypoints_runs"
    refined_run.attrs["source_keypoints_run"] = "kp_001"
    refined_run.attrs["total_rois"] = 4
    refined_run.attrs["successful_eyes"] = 7
    refined_run.attrs["successful_roi_pairs"] = 4
    refined_run.attrs["successful_roi_pair_rate"] = 1.0
    refined_run.attrs["duration_seconds"] = 1.0
    refined_run.attrs["eye_mask_review_status"] = {
        "state": "approved",
        "method": "manual",
        "intended_use": "training",
        "reviewer": "pytest",
        "timestamp": "2026-02-11T00:15:00+00:00",
    }
    refined_run.attrs["source_keypoint_stale"] = {
        "state": "stale",
        "reason": "keypoint_manual_correction",
        "timestamp": "2026-02-11T00:20:00+00:00",
        "source_keypoint_group": "keypoints_runs",
        "source_keypoints_run": "kp_001",
        "roi_indices": [2],
        "frame_indices": [10],
    }
    refined_run.attrs["summary_statistics"] = {
        "refine": {"smoothed_rois": 2},
        "postprocess": {"manual_fix_count": 1},
    }


class _FakeArray:
    def __init__(self, data: np.ndarray) -> None:
        self._data = np.asarray(data)
        self.shape = self._data.shape

    def __getitem__(self, key):
        return self._data[key]


class _FakeGroup:
    def __init__(self, *, attrs: Optional[Dict[str, object]] = None) -> None:
        self.attrs: Dict[str, object] = dict(attrs or {})
        self._children: Dict[str, object] = {}

    def add_group(self, name: str, *, attrs: Optional[Dict[str, object]] = None) -> "_FakeGroup":
        group = _FakeGroup(attrs=attrs)
        self._children[name] = group
        return group

    def add_array(self, name: str, data: np.ndarray) -> _FakeArray:
        arr = _FakeArray(np.asarray(data))
        self._children[name] = arr
        return arr

    def get(self, key: str):
        return self._children.get(key)

    def __contains__(self, key: str) -> bool:
        return key in self._children

    def __getitem__(self, key: str):
        return self._children[key]

    def keys(self):
        return self._children.keys()

    def group_keys(self):
        return [name for name, value in self._children.items() if isinstance(value, _FakeGroup)]


class _GetMissFakeGroup(_FakeGroup):
    def add_group(self, name: str, *, attrs: Optional[Dict[str, object]] = None) -> "_GetMissFakeGroup":
        group = _GetMissFakeGroup(attrs=attrs)
        self._children[name] = group
        return group

    def get(self, key: str):
        _ = key
        return None


class _FakeZarrModule:
    def __init__(self, roots_by_path: Dict[str, _FakeGroup]) -> None:
        self._roots_by_path = roots_by_path

    def open_group(self, path: str, mode: str = "r", consolidated: Optional[bool] = None) -> _FakeGroup:
        _ = mode, consolidated
        return self._roots_by_path[str(path)]


def _create_fake_zarr_store(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "zarr.json").write_text("{}", encoding="utf-8")


def _create_recording_step_status_zarr(path: Path) -> _FakeGroup:
    _create_fake_zarr_store(path)

    root = _FakeGroup(attrs={"session_uuid": "recording_step_session", "zarr_purpose": "analysis"})
    raw = root.add_group("raw_video")
    raw.add_array("images_ds", np.zeros((4, 8, 8), dtype=np.uint8))

    background_parent = root.add_group("background_runs", attrs={"latest": "background_001"})
    background = background_parent.add_group(
        "background_001",
        attrs={"created_utc": "2026-02-15T00:00:00+00:00", "method": "running_mean"},
    )
    background.add_array("background_full", np.zeros((8, 8), dtype=np.float32))
    background.add_array("background_ds", np.zeros((8, 8), dtype=np.float32))

    detect_parent = root.add_group("detect_runs", attrs={"latest": "detect_001"})
    detect = detect_parent.add_group(
        "detect_001",
        attrs={"detect_timestamp_utc": "2026-02-15T00:00:00+00:00", "detection_method": "yolo"},
    )
    detect.add_array("frame_counts", np.array([1, 0, 1, 1], dtype=np.int32))
    detect_quality_parent = detect.add_group("quality_reports", attrs={"latest": "detect_quality_001"})
    detect_quality_parent.add_group(
        "detect_quality_001",
        attrs={
            "quality_score": {"grade": "A", "overall_score": 98.4},
            "detection_quality_summary": {
                "clean_percentage": 97.0,
                "blip_detections": 1,
                "jump_detections": 1,
                "multi_detections": 1,
            },
        },
    )

    refined_detect_parent = root.add_group("refined_detect_runs", attrs={"latest": "refined_detect_001"})
    refined_detect_parent.add_group(
        "refined_detect_001",
        attrs={
            "created_utc": "2026-02-15T00:05:00+00:00",
            "source_detect_run": "detect_001",
            "parameters": {"refine_mode": "interpolated"},
            "coverage_comparison": {"interpolated": {"coverage_percent": 75.0}},
            "detect_review_status": {
                "state": "approved",
                "method": "manual",
                "reviewer": "pytest",
                "timestamp_utc": "2026-02-15T00:06:00+00:00",
            },
        },
    )

    crop_parent = root.add_group("crop_runs", attrs={"latest": "crop_001"})
    crop = crop_parent.add_group(
        "crop_001",
        attrs={
            "created_at_utc": "2026-02-15T00:10:00+00:00",
            "status": "completed",
            "detection_source_type": "manual",
            "crop_review_status": {
                "state": "approved",
                "intended_use": "training",
                "reviewer": "pytest",
                "timestamp_utc": "2026-02-15T00:11:00+00:00",
            },
        },
    )
    crop.add_array("frame_counts", np.array([1, 1, 1, 1], dtype=np.int32))

    keypoints_parent = root.add_group("keypoints_runs", attrs={"latest": "kp_001"})
    keypoints_parent.add_group(
        "kp_001",
        attrs={"created_utc": "2026-02-15T00:20:00+00:00", "method": "traditional_pose"},
    )

    refined_keypoints_parent = root.add_group("refined_keypoints_runs", attrs={"latest": "refined_kp_001"})
    refined_keypoints_parent.add_group(
        "refined_kp_001",
        attrs={
            "created_utc": "2026-02-15T00:30:00+00:00",
            "method": "refine_keypoints",
            "source_keypoints_run": "kp_001",
            "summary_statistics": {"postprocess": {"total_rois": 4, "usable_keypoints": 3}},
            "keypoint_review_status": {
                "state": "approved",
                "intended_use": "training",
                "reviewer": "pytest",
                "timestamp_utc": "2026-02-15T00:31:00+00:00",
            },
        },
    )

    eye_masks_parent = root.add_group("eye_masks_runs", attrs={"latest": "eye_masks_001"})
    eye_masks_parent.add_group(
        "eye_masks_001",
        attrs={
            "created_utc": "2026-02-15T00:40:00+00:00",
            "method": "traditional_eye_segmentation",
            "source_keypoints_run": "refined_kp_001",
            "source_keypoint_group": "refined_keypoints_runs",
            "successful_roi_pair_rate": 0.75,
        },
    )

    refined_eye_masks_parent = root.add_group("refined_eye_masks_runs", attrs={"latest": "refined_eye_masks_001"})
    refined_eye_masks_parent.add_group(
        "refined_eye_masks_001",
        attrs={
            "created_utc": "2026-02-15T00:45:00+00:00",
            "method": "refine_eye_masks",
            "source_eye_masks_run": "eye_masks_001",
            "successful_roi_pair_rate": 1.0,
            "eye_mask_review_status": {
                "state": "approved",
                "intended_use": "training",
                "reviewer": "pytest",
                "timestamp_utc": "2026-02-15T00:46:00+00:00",
            },
        },
    )

    id_parent = root.add_group("id_assignment_runs", attrs={"latest": "id_assign_001"})
    id_parent.add_group(
        "id_assign_001",
        attrs={"created_utc": "2026-02-15T00:50:00+00:00", "method": "hungarian"},
    )

    tracks_parent = root.add_group("tracking_runs", attrs={"latest": "tracks_001"})
    tracks_parent.add_group(
        "tracks_001",
        attrs={"created_utc": "2026-02-15T00:55:00+00:00", "method": "trackpy"},
    )

    analysis = root.add_group("analysis")
    stimulus_parent = analysis.add_group("stimulus_runs", attrs={"latest": "stimulus_001"})
    stimulus_parent.add_group(
        "stimulus_001",
        attrs={"created_utc": "2026-02-15T00:56:00+00:00"},
    )

    root.add_group("calibration", attrs={"created_utc": "2026-02-15T00:57:00+00:00"})
    analysis_meta = root.add_group("analysis_metadata")
    analysis_meta.attrs.update(
        {
            "dish_mask": {"ready": True},
            "detection_tuning": {"ready": True},
            "keypoint_tuning": {"ready": True},
            "eye_mask_tuning": {"ready": True},
            "subdish_mask_tuning": {"ready": True},
        }
    )
    return root


def _create_detectless_zarr(path: Path, *, session_uuid: str = "detectless_session") -> None:
    root = zarr.open_group(str(path), mode="w")
    root.attrs["session_uuid"] = session_uuid
    raw = root.create_group("raw_video")
    raw.create_array("images_ds", data=np.zeros((4, 8, 8), dtype=np.uint8), chunks=(1, 8, 8))


def _create_detect_quality_fake_zarr(path: Path, *, refined_runs: tuple[str, ...]) -> _FakeGroup:
    _create_fake_zarr_store(path)

    root = _FakeGroup(attrs={"session_uuid": "detect_quality_session", "zarr_purpose": "analysis"})
    raw = root.add_group("raw_video")
    raw.add_array("images_ds", np.zeros((4, 8, 8), dtype=np.uint8))

    detect_parent = root.add_group("detect_runs", attrs={"latest": "detect_001"})
    detect_parent.add_group(
        "detect_001",
        attrs={
            "detect_timestamp_utc": "2026-02-16T00:00:00+00:00",
            "detection_method": "yolo",
        },
    )

    if not refined_runs:
        return root

    refined_parent = root.add_group("refined_detect_runs", attrs={"latest": refined_runs[-1]})
    for index, refined_run in enumerate(refined_runs, start=1):
        refined_group = refined_parent.add_group(
            refined_run,
            attrs={
                "created_utc": f"2026-02-16T00:{index:02d}:00+00:00",
                "source_detect_run": "detect_001",
                "detect_review_status": {
                    "state": "approved",
                    "intended_use": "training",
                    "reviewer": "pytest",
                    "timestamp_utc": f"2026-02-16T00:{index + 10:02d}:00+00:00",
                    "resolved_group": "filtered",
                },
            },
        )
        resolved = refined_group.add_group("filtered")
        resolved.add_array("detection_source", np.array([0, 0, 1, 0], dtype=np.int8))

    return root


def test_is_nested_zarr_subpath() -> None:
    assert _is_nested_zarr_subpath("/data/a/session.zarr/detect_runs")
    assert _is_nested_zarr_subpath("/data/a/session.zarr/detect_runs/run_01")
    assert not _is_nested_zarr_subpath("/data/a/session.zarr")
    assert not _is_nested_zarr_subpath("/data/a/session.zarr/subset.zarr")


def test_schema_has_fk_indexes_for_skeleton_columns(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    expected_indexes = {
        "idx_training_sets_skeleton_id",
        "idx_training_runs_skeleton_id",
        "idx_onnx_models_skeleton_id",
        "idx_tensorrt_models_skeleton_id",
    }
    rows = registry.conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'index' AND name IN (?, ?, ?, ?)
        ORDER BY name;
        """,
        tuple(sorted(expected_indexes)),
    ).fetchall()
    found = {str(row["name"]) for row in rows}
    assert found == expected_indexes
    registry.close()


def test_schema_has_training_task_type_columns_and_indexes(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")

    set_cols = {
        str(row["name"])
        for row in registry.conn.execute("PRAGMA table_info(training_sets);").fetchall()
    }
    run_cols = {
        str(row["name"])
        for row in registry.conn.execute("PRAGMA table_info(training_runs);").fetchall()
    }
    assert "task_type" in set_cols
    assert "task_type" in run_cols

    idx_rows = registry.conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type='index'
          AND name IN ('idx_training_sets_task_type', 'idx_training_runs_task_type');
        """
    ).fetchall()
    idx_names = {str(row["name"]) for row in idx_rows}
    assert idx_names == {"idx_training_sets_task_type", "idx_training_runs_task_type"}
    registry.close()


def test_training_task_type_inferred_and_backfilled(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")

    registry.upsert_training_set(
        set_id="detect_example_v001",
        name="detect example",
        query_filter={"task": "detect"},
        dataset_ids=[],
    )
    registry.record_training_run(
        run_id="host_detect_example_v001_detect_20260209-000000_deadbeef",
        set_id="detect_example_v001",
        config_path=None,
        manifest_path=None,
        model_path=None,
        metrics_path=None,
        status="success",
    )

    set_row = registry.conn.execute(
        "SELECT task_type FROM training_sets WHERE set_id = 'detect_example_v001';"
    ).fetchone()
    run_row = registry.conn.execute(
        "SELECT task_type FROM training_runs WHERE run_id = 'host_detect_example_v001_detect_20260209-000000_deadbeef';"
    ).fetchone()
    assert set_row is not None and str(set_row["task_type"]) == "detect"
    assert run_row is not None and str(run_row["task_type"]) == "detect"

    registry.conn.execute("UPDATE training_sets SET task_type = NULL WHERE set_id = 'detect_example_v001';")
    registry.conn.execute(
        "UPDATE training_runs SET task_type = NULL WHERE run_id = 'host_detect_example_v001_detect_20260209-000000_deadbeef';"
    )
    registry.conn.commit()

    registry._migration_009_training_task_type_columns()  # noqa: SLF001

    set_row2 = registry.conn.execute(
        "SELECT task_type FROM training_sets WHERE set_id = 'detect_example_v001';"
    ).fetchone()
    run_row2 = registry.conn.execute(
        "SELECT task_type FROM training_runs WHERE run_id = 'host_detect_example_v001_detect_20260209-000000_deadbeef';"
    ).fetchone()
    assert set_row2 is not None and str(set_row2["task_type"]) == "detect"
    assert run_row2 is not None and str(run_row2["task_type"]) == "detect"
    registry.close()


def test_schema_has_detect_performance_table_views_and_indexes(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    table = registry.conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table' AND name = 'detect_performance';
        """
    ).fetchone()
    assert table is not None

    views = registry.conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'view' AND name IN (
            'detect_performance_latest',
            'recording_detect_performance_latest',
            'detect_model_performance_latest',
            'recording_detect_model_performance_latest',
            'detect_model_performance_summary',
            'recording_detect_model_performance_summary'
        );
        """
    ).fetchall()
    view_names = {str(row["name"]) for row in views}
    assert view_names == {
        "detect_performance_latest",
        "recording_detect_performance_latest",
        "detect_model_performance_latest",
        "recording_detect_model_performance_latest",
        "detect_model_performance_summary",
        "recording_detect_model_performance_summary",
    }

    idx = registry.conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'index' AND name IN (
            'idx_detect_perf_recording',
            'idx_detect_perf_coverage',
            'idx_detect_perf_runtime',
            'idx_detect_perf_method',
            'idx_detect_perf_model_path'
        );
        """
    ).fetchall()
    idx_names = {str(row["name"]) for row in idx}
    assert idx_names == {
        "idx_detect_perf_recording",
        "idx_detect_perf_coverage",
        "idx_detect_perf_runtime",
        "idx_detect_perf_method",
        "idx_detect_perf_model_path",
    }
    registry.close()


def test_schema_has_keypoint_performance_table_views_and_indexes(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    table = registry.conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table' AND name = 'keypoint_performance';
        """
    ).fetchone()
    assert table is not None

    views = registry.conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'view' AND name IN (
            'keypoint_performance_latest',
            'recording_keypoint_performance_latest'
        );
        """
    ).fetchall()
    view_names = {str(row["name"]) for row in views}
    assert view_names == {
        "keypoint_performance_latest",
        "recording_keypoint_performance_latest",
    }

    idx = registry.conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'index' AND name IN (
            'idx_keypoint_perf_recording',
            'idx_keypoint_perf_method',
            'idx_keypoint_perf_runtime',
            'idx_keypoint_perf_source'
        );
        """
    ).fetchall()
    idx_names = {str(row["name"]) for row in idx}
    assert idx_names == {
        "idx_keypoint_perf_recording",
        "idx_keypoint_perf_method",
        "idx_keypoint_perf_runtime",
        "idx_keypoint_perf_source",
    }
    registry.close()


def test_schema_has_eye_mask_performance_table_views_and_indexes(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    table = registry.conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table' AND name = 'eye_mask_performance';
        """
    ).fetchone()
    assert table is not None

    views = registry.conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'view' AND name IN (
            'eye_mask_performance_latest',
            'recording_eye_mask_performance_latest'
        );
        """
    ).fetchall()
    view_names = {str(row["name"]) for row in views}
    assert view_names == {
        "eye_mask_performance_latest",
        "recording_eye_mask_performance_latest",
    }

    idx = registry.conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'index' AND name IN (
            'idx_eye_mask_perf_recording',
            'idx_eye_mask_perf_stage_method',
            'idx_eye_mask_perf_runtime',
            'idx_eye_mask_perf_source',
            'idx_eye_mask_perf_review'
        );
        """
    ).fetchall()
    idx_names = {str(row["name"]) for row in idx}
    assert idx_names == {
        "idx_eye_mask_perf_recording",
        "idx_eye_mask_perf_stage_method",
        "idx_eye_mask_perf_runtime",
        "idx_eye_mask_perf_source",
        "idx_eye_mask_perf_review",
    }
    registry.close()


def test_schema_has_eye_mask_quality_table_views_and_indexes(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    table = registry.conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table' AND name = 'eye_mask_quality';
        """
    ).fetchone()
    assert table is not None

    views = registry.conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'view' AND name IN (
            'eye_mask_quality_current',
            'eye_mask_quality_overview',
            'recording_eye_mask_quality_overview'
        );
        """
    ).fetchall()
    view_names = {str(row["name"]) for row in views}
    assert view_names == {
        "eye_mask_quality_current",
        "eye_mask_quality_overview",
        "recording_eye_mask_quality_overview",
    }

    idx = registry.conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'index' AND name IN (
            'idx_eye_mask_quality_dataset_id',
            'idx_eye_mask_quality_gate',
            'idx_eye_mask_quality_stage_method',
            'idx_eye_mask_quality_recording'
        );
        """
    ).fetchall()
    idx_names = {str(row["name"]) for row in idx}
    assert idx_names == {
        "idx_eye_mask_quality_dataset_id",
        "idx_eye_mask_quality_gate",
        "idx_eye_mask_quality_stage_method",
        "idx_eye_mask_quality_recording",
    }
    registry.close()


def test_schema_has_phase2_subject_dish_cross_tables(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    expected_tables = {
        "crosses",
        "dishes",
        "recording_subjects",
    }
    rows = registry.conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table' AND name IN (?, ?, ?)
        ORDER BY name;
        """,
        tuple(sorted(expected_tables)),
    ).fetchall()
    found = {str(row["name"]) for row in rows}
    assert found == expected_tables
    registry.close()


def test_schema_has_phase6_subject_indexes(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    expected_indexes = {
        "idx_crosses_genotype",
        "idx_subjects_dish_id",
        "idx_recording_subjects_subject_dpf",
        "idx_recording_subjects_recording_id",
    }
    rows = registry.conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'index' AND name IN (?, ?, ?, ?)
        ORDER BY name;
        """,
        tuple(sorted(expected_indexes)),
    ).fetchall()
    found = {str(row["name"]) for row in rows}
    assert found == expected_indexes
    registry.close()


def test_schema_has_recording_subject_overview_view(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    rows = registry.conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'view' AND name = 'recording_subject_overview';
        """
    ).fetchall()
    assert len(rows) == 1
    assert str(rows[0]["name"]) == "recording_subject_overview"
    registry.close()


def test_schema_has_recording_step_status_tables_and_views(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    table_rows = registry.conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table' AND name IN ('recording_step_status', 'recording_step_status_history')
        ORDER BY name;
        """
    ).fetchall()
    assert {str(row["name"]) for row in table_rows} == {
        "recording_step_status",
        "recording_step_status_history",
    }

    view_rows = registry.conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'view' AND name IN (
            'recording_step_status_latest',
            'recording_step_overview',
            'recording_step_status_wide'
        )
        ORDER BY name;
        """
    ).fetchall()
    assert {str(row["name"]) for row in view_rows} == {
        "recording_step_overview",
        "recording_step_status_latest",
        "recording_step_status_wide",
    }

    index_rows = registry.conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'index' AND name IN (
            'idx_recording_step_status_recording_step',
            'idx_recording_step_status_dataset_step',
            'idx_recording_step_status_status'
        )
        ORDER BY name;
        """
    ).fetchall()
    assert {str(row["name"]) for row in index_rows} == {
        "idx_recording_step_status_dataset_step",
        "idx_recording_step_status_recording_step",
        "idx_recording_step_status_status",
    }
    registry.close()


def test_recording_step_status_latest_view_includes_dataset_context(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    dataset_path = tmp_path / "recording_a_training.zarr"
    registry.upsert_dataset(
        dataset_id="dataset_a",
        session_uuid="session_a",
        zarr_path=dataset_path,
        recording_id="recording_a",
        artifact_kind="source_recording",
        zarr_use="training",
    )
    registry.upsert_provenance(
        "dataset_a",
        provenance={},
        context={"rig_id": "rig_1", "arena_id": "arena_1", "camera_id": "cam_1"},
        protocol_name="DefaultScreen",
        protocol_hash=None,
        acquisition={"dish_design": "cedar"},
        zarr_purpose="training",
    )
    registry.conn.execute(
        """
        INSERT INTO recording_step_status (
            dataset_id, recording_id, step_name, status, run_name, method, coverage_pct,
            source, updated_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            "dataset_a",
            "recording_a",
            "detect",
            "ok",
            "detect_2026-02-22_00-00-00",
            "yolo",
            99.5,
            "unit_test",
            "2026-02-22T00:00:00+00:00",
        ),
    )
    registry.conn.commit()

    row = registry.conn.execute(
        """
        SELECT recording_id, dataset_id, zarr_use, rig_id, arena_id, camera_id, step_name, status
        FROM recording_step_status_latest
        WHERE dataset_id = ? AND step_name = ?;
        """,
        ("dataset_a", "detect"),
    ).fetchone()
    assert row is not None
    assert row["recording_id"] == "recording_a"
    assert row["zarr_use"] == "training"
    assert row["rig_id"] == "rig_1"
    assert row["arena_id"] == "arena_1"
    assert row["camera_id"] == "cam_1"
    assert row["step_name"] == "detect"
    assert row["status"] == "ok"
    registry.close()


def test_recording_step_overview_reports_non_ok_steps(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.upsert_dataset(
        dataset_id="dataset_a",
        session_uuid="session_a",
        zarr_path=tmp_path / "a.zarr",
        recording_id="recording_a",
        artifact_kind="source_recording",
        zarr_use="training",
    )
    registry.upsert_dataset(
        dataset_id="dataset_b",
        session_uuid="session_b",
        zarr_path=tmp_path / "b.zarr",
        recording_id="recording_a",
        artifact_kind="source_recording",
        zarr_use="analysis",
    )
    registry.conn.executemany(
        """
        INSERT INTO recording_step_status (
            dataset_id, recording_id, step_name, status, run_name, source, updated_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """,
        [
            ("dataset_a", "recording_a", "detect", "ok", "detect_a", "unit_test", "2026-02-22T00:00:00+00:00"),
            ("dataset_b", "recording_a", "detect", "ok", "detect_b", "unit_test", "2026-02-22T00:00:01+00:00"),
            ("dataset_a", "recording_a", "keypoints", "missing", None, "unit_test", "2026-02-22T00:00:02+00:00"),
            ("dataset_b", "recording_a", "keypoints", "ok", "keypoints_b", "unit_test", "2026-02-22T00:00:03+00:00"),
            ("dataset_a", "recording_a", "crop", "ok", "crop_a", "unit_test", "2026-02-22T00:00:04+00:00"),
            ("dataset_b", "recording_a", "crop", "missing", None, "unit_test", "2026-02-22T00:00:05+00:00"),
        ],
    )
    registry.conn.commit()

    row = registry.conn.execute(
        """
        SELECT
            recording_id,
            dataset_count,
            step_rows_total,
            missing_rows,
            detect_ok_count,
            detect_non_ok_count,
            keypoints_ok_count,
            keypoints_non_ok_count,
            crop_ok_count,
            crop_non_ok_count,
            blocking_steps_csv
        FROM recording_step_overview
        WHERE recording_id = ?;
        """,
        ("recording_a",),
    ).fetchone()
    assert row is not None
    assert row["recording_id"] == "recording_a"
    assert int(row["dataset_count"]) == 2
    assert int(row["step_rows_total"]) == 6
    assert int(row["missing_rows"]) == 2
    assert int(row["detect_ok_count"]) == 2
    assert int(row["detect_non_ok_count"]) == 0
    assert int(row["keypoints_ok_count"]) == 1
    assert int(row["keypoints_non_ok_count"]) == 1
    assert int(row["crop_ok_count"]) == 1
    assert int(row["crop_non_ok_count"]) == 1
    blocking = str(row["blocking_steps_csv"] or "")
    assert "crop" in blocking
    assert "keypoints" in blocking
    registry.close()


def test_recording_step_status_wide_view_formats_check_recording_steps_columns(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.upsert_dataset(
        dataset_id="dataset_a",
        session_uuid="session_a",
        zarr_path=tmp_path / "recording_a_training.zarr",
        recording_id="recording_a",
        artifact_kind="source_recording",
        zarr_use="training",
    )
    registry.upsert_provenance(
        "dataset_a",
        provenance={},
        context={"camera_id": "cam_1"},
        protocol_name="DefaultScreen",
        protocol_hash=None,
        acquisition={},
        zarr_purpose="training",
    )

    def _json_text(payload: Optional[Dict[str, object]]) -> Optional[str]:
        if payload is None:
            return None
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    rows = [
        (
            "dataset_a",
            "recording_a",
            "raw",
            "ok",
            None,
            None,
            None,
            None,
            _json_text(
                {
                    "raw_present": True,
                    "full_present": True,
                    "ds_present": True,
                    "sampled_present": False,
                    "pipeline_type": "analysis",
                    "zarr_purpose": "analysis",
                    "has_raw_video_attr": True,
                }
            ),
            "unit_test",
            "2026-02-23T01:00:00+00:00",
        ),
        (
            "dataset_a",
            "recording_a",
            "background",
            "ok",
            "background_001",
            None,
            None,
            None,
            _json_text({"full_present": True, "ds_present": True}),
            "unit_test",
            "2026-02-23T01:00:01+00:00",
        ),
        (
            "dataset_a",
            "recording_a",
            "detect",
            "ok",
            "detect_001",
            "yolo",
            75.0,
            None,
            _json_text(
                {
                    "detect_quality_grade": "A",
                    "detect_quality_score": 98.4,
                    "detect_quality_clean_percent": 97.0,
                    "detect_quality_artifacts": 3,
                }
            ),
            "unit_test",
            "2026-02-23T01:00:02+00:00",
        ),
        (
            "dataset_a",
            "recording_a",
            "refined_detect",
            "ok",
            "refined_detect_001",
            "passthrough",
            80.0,
            _json_text(
                {
                    "state": "approved",
                    "method": "manual",
                    "intended_use": "training",
                    "resolved_group": "refined_detect_runs/refined_detect_001",
                }
            ),
            None,
            "unit_test",
            "2026-02-23T01:00:03+00:00",
        ),
        (
            "dataset_a",
            "recording_a",
            "crop",
            "ok",
            "crop_001",
            None,
            None,
            _json_text({"state": "pending", "method": "manual", "intended_use": "training"}),
            _json_text({"run_state": "completed"}),
            "unit_test",
            "2026-02-23T01:00:04+00:00",
        ),
        ("dataset_a", "recording_a", "keypoints", "ok", "kp_001", None, None, None, None, "unit_test", "2026-02-23T01:00:05+00:00"),
        (
            "dataset_a",
            "recording_a",
            "refined_keypoints",
            "ok",
            "refined_kp_001",
            None,
            90.0,
            _json_text({"state": "approved", "method": "manual", "intended_use": "training"}),
            _json_text({"usable_keypoints_pct": 85.0}),
            "unit_test",
            "2026-02-23T01:00:06+00:00",
        ),
        ("dataset_a", "recording_a", "eye_masks", "ok", "eye_001", None, None, None, None, "unit_test", "2026-02-23T01:00:07+00:00"),
        (
            "dataset_a",
            "recording_a",
            "refined_eye_masks",
            "ok",
            "refined_eye_001",
            None,
            None,
            _json_text({"state": "approved", "method": "manual", "intended_use": "training"}),
            None,
            "unit_test",
            "2026-02-23T01:00:08+00:00",
        ),
        ("dataset_a", "recording_a", "id_assignment", "missing", None, None, None, None, None, "unit_test", "2026-02-23T01:00:09+00:00"),
        ("dataset_a", "recording_a", "tracks", "absent", None, None, None, None, None, "unit_test", "2026-02-23T01:00:10+00:00"),
        (
            "dataset_a",
            "recording_a",
            "stimulus",
            "ok",
            None,
            None,
            None,
            None,
            _json_text({"stimulus_runs": 3}),
            "unit_test",
            "2026-02-23T01:00:11+00:00",
        ),
        ("dataset_a", "recording_a", "calibration", "ok", None, None, None, None, None, "unit_test", "2026-02-23T01:00:12+00:00"),
        ("dataset_a", "recording_a", "dish_mask", "ok", None, None, None, None, None, "unit_test", "2026-02-23T01:00:13+00:00"),
        ("dataset_a", "recording_a", "detection_tuning", "missing", None, None, None, None, None, "unit_test", "2026-02-23T01:00:14+00:00"),
        ("dataset_a", "recording_a", "keypoint_tuning", "ok", None, None, None, None, None, "unit_test", "2026-02-23T01:00:15+00:00"),
        ("dataset_a", "recording_a", "eye_mask_tuning", "ok", None, None, None, None, None, "unit_test", "2026-02-23T01:00:16+00:00"),
        ("dataset_a", "recording_a", "subdish_mask_tuning", "na", None, None, None, None, None, "unit_test", "2026-02-23T01:00:17+00:00"),
    ]
    registry.conn.executemany(
        """
        INSERT INTO recording_step_status (
            dataset_id,
            recording_id,
            step_name,
            status,
            run_name,
            method,
            coverage_pct,
            review_status_json,
            details_json,
            source,
            updated_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        rows,
    )
    registry.conn.commit()

    row = registry.conn.execute(
        """
        SELECT *
        FROM recording_step_status_wide
        WHERE "Recording" = ?;
        """,
        ("recording_a",),
    ).fetchone()
    assert row is not None
    assert row["Camera"] == "cam_1"
    assert row["Zarr"] == "OK"
    assert row["Use"] == "training"
    assert row["Purpose"] == "analysis"
    assert row["Import"] == "OK"
    assert row["BG Full"] == "OK"
    assert row["BG DS"] == "OK"
    assert row["Detect"] == "OK (75.0%, registry, yolo)"
    assert row["Detect Quality"] == "OK (A 98.4, clean 97.0%, art 3)"
    assert row["Refine Detect"] == "80.0% (passthrough)"
    assert row["Detect Group"] == "refined_detect_runs/refined_detect_001"
    assert row["Detect Review"] == "approved (manual, training, group=refined_detect_runs/refined_detect_001)"
    assert row["Crop"] == "completed"
    assert row["Crop Review"] == "pending (manual, training)"
    assert row["Keypoints"] == "OK"
    assert row["Refined Keypoints (analysis/train)"] == "90.0% (train 85.0%)"
    assert row["Keypoint Review"] == "approved (manual, training)"
    assert row["Eye Masks"] == "OK"
    assert row["Refined Eye Masks"] == "OK"
    assert row["Eye Mask Review"] == "approved (manual, training)"
    assert row["Assign IDs"] == "MISS"
    assert row["Track"] == "MISS"
    assert row["Stimulus"] == "3 (OK)"
    assert row["Calib"] == "OK"
    assert row["Tuning"] == "3/5"
    assert row["dish_mask"] == "OK"
    assert row["detection_tuning"] == "MISS"
    assert row["keypoint_tuning"] == "OK"
    assert row["eye_mask_tuning"] == "OK"
    assert row["subdish_mask_tuning"] == "N/A"
    registry.close()


def test_recording_subject_overview_exposes_genotype_and_dpf(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.conn.execute(
        """
        INSERT INTO recordings (
            recording_id, session_uuid, recording_name, recording_path, recording_type,
            recording_subtype, behavior_mode, artifact_schema_id, protocol_name,
            created_utc, updated_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'));
        """,
        (
            "recording_a",
            "recording_a",
            "recording_a",
            str(tmp_path / "recordings" / "recording_a"),
            "behavior",
            "free",
            "free",
            "behavior_v1",
            "protocol_a",
        ),
    )
    registry.conn.execute(
        """
        INSERT INTO crosses (
            cross_id, genotype, line_strain, created_utc, updated_utc
        )
        VALUES (?, ?, ?, datetime('now'), datetime('now'));
        """,
        ("cross_a", "genotype_y", "line_a"),
    )
    registry.conn.execute(
        """
        INSERT INTO dishes (
            dish_id, cross_id, species, created_utc, updated_utc
        )
        VALUES (?, ?, ?, datetime('now'), datetime('now'));
        """,
        ("dish_a", "cross_a", "danio_rerio"),
    )
    registry.conn.execute(
        """
        INSERT INTO subjects (
            subject_id, dish_id, species, sex, created_utc, updated_utc
        )
        VALUES (?, ?, ?, ?, datetime('now'), datetime('now'));
        """,
        ("subject_a", "dish_a", "danio_rerio", "unknown"),
    )
    registry.conn.execute(
        """
        INSERT INTO recording_subjects (
            recording_id, subject_id, dish_id, cross_id, dpf_at_acquisition, created_utc, updated_utc
        )
        VALUES (?, ?, ?, ?, ?, datetime('now'), datetime('now'));
        """,
        ("recording_a", "subject_a", None, None, 8),
    )
    registry.conn.commit()

    row = registry.conn.execute(
        """
        SELECT
            recording_id, subject_id, dish_id, cross_id, genotype, dpf_at_acquisition,
            protocol_name, recording_type
        FROM recording_subject_overview
        WHERE recording_id = ? AND subject_id = ?;
        """,
        ("recording_a", "subject_a"),
    ).fetchone()
    assert row is not None
    assert row["dish_id"] == "dish_a"
    assert row["cross_id"] == "cross_a"
    assert row["genotype"] == "genotype_y"
    assert int(row["dpf_at_acquisition"]) == 8
    assert row["protocol_name"] == "protocol_a"
    assert row["recording_type"] == "behavior"
    registry.close()


def test_backfill_recording_entities_and_integrity_for_behavior_manifest(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    recording_dir = (
        tmp_path
        / "recordings"
        / "2026-01-28T19-22-28Z_arena_1_DefaultScreen"
    )
    zarr_path = recording_dir / "zarr" / "2026-01-28T19-22-28Z_arena_1_DefaultScreen.zarr"
    zarr_path.mkdir(parents=True)

    # Minimal required artifact files for behavior_v1.
    h5_path = recording_dir / "raw" / "session_data.h5"
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    h5_path.write_bytes(b"h5")
    timing_path = recording_dir / "raw" / "session_update_timing.csv"
    timing_path.write_text("t,dt\n0,0\n", encoding="utf-8")
    cam_video_path = recording_dir / "cams" / "cam0.mp4"
    cam_video_path.parent.mkdir(parents=True, exist_ok=True)
    cam_video_path.write_bytes(b"mp4")
    cam_meta_path = recording_dir / "cams" / "cam0.csv"
    cam_meta_path.write_text("frame,time\n0,0\n", encoding="utf-8")

    manifest_path = recording_dir / "recording_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "session_uuid": "2026-01-28T19-22-28Z_arena_1",
                "recording_type": "behavior",
                "recording_subtype": "free",
                "behavior_mode": "free",
                "artifact_schema_id": "behavior_v1",
                "rig_id": "omnifin0",
                "arena_id": "arena_1",
                "camera_id": "2010093",
                "files": {
                    "raw": [
                        "raw/session_data.h5",
                        "raw/session_update_timing.csv",
                    ],
                    "cams": [
                        "cams/cam0.mp4",
                        "cams/cam0.csv",
                    ],
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    registry.upsert_dataset(
        dataset_id="2026-01-28T19-22-28Z_arena_1",
        session_uuid="2026-01-28T19-22-28Z_arena_1",
        zarr_path=zarr_path,
    )

    summary = _backfill_recording_entities(registry, dry_run=False)
    assert summary["recordings_scanned"] == 1
    assert summary["manifests_missing"] == 0
    assert summary["recordings_upserted"] == 1
    assert summary["datasets_linked"] in {0, 1}
    assert summary["artifacts_seen"] == 4
    assert summary["artifacts_upserted"] == 4

    dataset_row = registry.conn.execute(
        """
        SELECT recording_id, artifact_kind
        FROM datasets
        WHERE dataset_id = ?;
        """,
        ("2026-01-28T19-22-28Z_arena_1",),
    ).fetchone()
    assert dataset_row is not None
    assert dataset_row["recording_id"] == "2026-01-28T19-22-28Z_arena_1"
    assert dataset_row["artifact_kind"] == "source_recording"

    recording_row = registry.conn.execute(
        """
        SELECT recording_type, recording_subtype, behavior_mode, artifact_schema_id
        FROM recordings
        WHERE recording_id = ?;
        """,
        ("2026-01-28T19-22-28Z_arena_1",),
    ).fetchone()
    assert recording_row is not None
    assert recording_row["recording_type"] == "behavior"
    assert recording_row["recording_subtype"] == "free"
    assert recording_row["behavior_mode"] == "free"
    assert recording_row["artifact_schema_id"] == "behavior_v1"

    issues = _check_registry_integrity(registry)
    recording_issue_codes = [issue.code for issue in issues if issue.code.startswith("recording_")]
    assert recording_issue_codes == []
    registry.close()


def test_backfill_subject_dish_cross_entities_from_source_provenance(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    zarr_path = tmp_path / "recordings" / "session_a" / "zarr" / "session_a_training.zarr"
    registry.upsert_dataset(
        dataset_id="session_a:z111",
        session_uuid="session_a",
        zarr_path=zarr_path,
        recording_id="session_a",
        artifact_kind="source_recording",
    )
    registry.conn.execute(
        """
        INSERT INTO recordings (
            recording_id, session_uuid, recording_name, recording_path, recording_type,
            recording_subtype, behavior_mode, artifact_schema_id, created_utc, updated_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'));
        """,
        (
            "session_a",
            "session_a",
            "session_a",
            str(tmp_path / "recordings" / "session_a"),
            "behavior",
            "free",
            "free",
            "behavior_v1",
        ),
    )
    registry.conn.execute(
        """
        INSERT INTO provenance (
            dataset_id, fish_id, dish_id, cross_id, line_strain, genotype, species,
            sex, dpf_at_acquisition, parents_json, subject_count, snapshot_status
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            "session_a:z111",
            "fish-uuid-1",
            "dish-001",
            "cross-001",
            "line_a",
            "wt",
            "danio_rerio",
            "unknown",
            6,
            json.dumps([{"identifier": "p1", "sex": "F"}]),
            1,
            "complete",
        ),
    )
    registry.conn.commit()

    dry_summary = _backfill_subject_dish_cross_entities(registry, dry_run=True)
    assert dry_summary["source_rows_scanned"] == 1
    assert dry_summary["crosses_unique_seen"] == 1
    assert dry_summary["crosses_would_insert"] == 1
    assert dry_summary["crosses_upserted"] == 1
    assert dry_summary["dishes_unique_seen"] == 1
    assert dry_summary["dishes_would_insert"] == 1
    assert dry_summary["dishes_upserted"] == 1
    assert dry_summary["recording_subjects_unique_seen"] == 1
    assert dry_summary["recording_subjects_would_insert"] == 1
    assert dry_summary["recording_subjects_upserted"] == 1
    assert registry.conn.execute("SELECT COUNT(*) FROM crosses;").fetchone()[0] == 0
    assert registry.conn.execute("SELECT COUNT(*) FROM dishes;").fetchone()[0] == 0
    assert registry.conn.execute("SELECT COUNT(*) FROM recording_subjects;").fetchone()[0] == 0

    apply_summary = _backfill_subject_dish_cross_entities(registry, dry_run=False)
    assert apply_summary["source_rows_scanned"] == 1
    assert apply_summary["crosses_unique_seen"] == 1
    assert apply_summary["crosses_would_insert"] == 1
    assert apply_summary["crosses_upserted"] == 1
    assert apply_summary["dishes_unique_seen"] == 1
    assert apply_summary["dishes_would_insert"] == 1
    assert apply_summary["dishes_upserted"] == 1
    assert apply_summary["recording_subjects_unique_seen"] == 1
    assert apply_summary["recording_subjects_would_insert"] == 1
    assert apply_summary["recording_subjects_upserted"] == 1

    cross_row = registry.conn.execute(
        "SELECT cross_id, line_strain, genotype FROM crosses WHERE cross_id = ?;",
        ("cross-001",),
    ).fetchone()
    assert cross_row is not None
    assert cross_row["line_strain"] == "line_a"
    assert cross_row["genotype"] == "wt"

    dish_row = registry.conn.execute(
        "SELECT dish_id, cross_id, species FROM dishes WHERE dish_id = ?;",
        ("dish-001",),
    ).fetchone()
    assert dish_row is not None
    assert dish_row["cross_id"] == "cross-001"
    assert dish_row["species"] == "danio_rerio"

    subject_row = registry.conn.execute(
        """
        SELECT recording_id, subject_id, dataset_id, dish_id, cross_id, dpf_at_acquisition
        FROM recording_subjects
        WHERE recording_id = ? AND subject_id = ?;
        """,
        ("session_a", "fish-uuid-1"),
    ).fetchone()
    assert subject_row is not None
    assert subject_row["dataset_id"] == "session_a:z111"
    assert subject_row["dish_id"] == "dish-001"
    assert subject_row["cross_id"] == "cross-001"
    assert int(subject_row["dpf_at_acquisition"]) == 6
    registry.close()


def test_backfill_subjects_from_recording_subjects(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    zarr_path = tmp_path / "recordings" / "session_a" / "zarr" / "session_a_training.zarr"
    registry.upsert_dataset(
        dataset_id="session_a:z111",
        session_uuid="session_a",
        zarr_path=zarr_path,
        recording_id="session_a",
        artifact_kind="source_recording",
    )
    registry.conn.execute(
        """
        INSERT INTO recordings (
            recording_id, session_uuid, recording_name, recording_path, recording_type,
            recording_subtype, behavior_mode, artifact_schema_id, created_utc, updated_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'));
        """,
        (
            "session_a",
            "session_a",
            "session_a",
            str(tmp_path / "recordings" / "session_a"),
            "behavior",
            "free",
            "free",
            "behavior_v1",
        ),
    )
    registry.conn.execute(
        """
        INSERT INTO provenance (
            dataset_id, fish_id, dish_id, cross_id, line_strain, genotype, species,
            sex, dpf_at_acquisition, subject_count, snapshot_status
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            "session_a:z111",
            "fish-uuid-1",
            "dish-001",
            "cross-001",
            "line_a",
            "wt",
            "danio_rerio",
            "unknown",
            6,
            1,
            "complete",
        ),
    )
    registry.conn.commit()

    _backfill_subject_dish_cross_entities(registry, dry_run=False)
    dry_summary = _backfill_subjects(registry, dry_run=True)
    assert dry_summary["subject_rows_scanned"] == 1
    assert dry_summary["subject_ids_unique_seen"] == 1
    assert dry_summary["subjects_would_insert"] == 1
    assert dry_summary["subjects_would_enrich"] == 0
    assert dry_summary["subjects_conflict_dish_id"] == 0
    assert dry_summary["subjects_conflict_species"] == 0
    assert dry_summary["subjects_conflict_sex"] == 0
    assert registry.conn.execute("SELECT COUNT(*) FROM subjects;").fetchone()[0] == 0

    apply_summary = _backfill_subjects(registry, dry_run=False)
    assert apply_summary["subject_rows_scanned"] == 1
    assert apply_summary["subject_ids_unique_seen"] == 1
    assert apply_summary["subjects_would_insert"] == 1

    subject_row = registry.conn.execute(
        """
        SELECT subject_id, dish_id, species, sex
        FROM subjects
        WHERE subject_id = ?;
        """,
        ("fish-uuid-1",),
    ).fetchone()
    assert subject_row is not None
    assert subject_row["dish_id"] == "dish-001"
    assert subject_row["species"] == "danio_rerio"
    assert subject_row["sex"] == "unknown"
    registry.close()


def test_backfill_subjects_reports_dish_conflict(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    for index, dish_id in enumerate(("dish-001", "dish-002"), start=1):
        dataset_id = f"session_a:z11{index}"
        recording_id = f"session_{index}"
        zarr_path = tmp_path / "recordings" / recording_id / "zarr" / f"{recording_id}_training.zarr"
        registry.upsert_dataset(
            dataset_id=dataset_id,
            session_uuid=recording_id,
            zarr_path=zarr_path,
            recording_id=recording_id,
            artifact_kind="source_recording",
        )
        registry.conn.execute(
            """
            INSERT INTO recordings (
                recording_id, session_uuid, recording_name, recording_path, recording_type,
                recording_subtype, behavior_mode, artifact_schema_id, created_utc, updated_utc
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'));
            """,
            (
                recording_id,
                recording_id,
                recording_id,
                str(tmp_path / "recordings" / recording_id),
                "behavior",
                "free",
                "free",
                "behavior_v1",
            ),
        )
        registry.conn.execute(
            """
            INSERT INTO provenance (
                dataset_id, fish_id, dish_id, cross_id, line_strain, genotype, species,
                sex, dpf_at_acquisition, subject_count, snapshot_status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                dataset_id,
                "fish-uuid-1",
                dish_id,
                "cross-001",
                "line_a",
                "wt",
                "danio_rerio",
                "unknown",
                6,
                1,
                "complete",
            ),
        )
    registry.conn.commit()

    _backfill_subject_dish_cross_entities(registry, dry_run=False)
    summary = _backfill_subjects(registry, dry_run=True)
    assert summary["subject_rows_scanned"] == 2
    assert summary["subject_ids_unique_seen"] == 1
    assert summary["subjects_conflict_dish_id"] == 1
    registry.close()


def test_register_from_root_disambiguates_dataset_id_for_same_session_uuid(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    zarr_a = tmp_path / "recordings" / "session_a" / "zarr" / "a.zarr"
    zarr_b = tmp_path / "recordings" / "session_a" / "zarr" / "b.zarr"
    zarr_a.parent.mkdir(parents=True, exist_ok=True)
    zarr_b.parent.mkdir(parents=True, exist_ok=True)

    root_a = zarr.open_group(str(zarr_a), mode="w")
    root_a.attrs["session_uuid"] = "session_a"
    root_b = zarr.open_group(str(zarr_b), mode="w")
    root_b.attrs["session_uuid"] = "session_a"

    dataset_a = registry.register_from_root(zarr.open_group(str(zarr_a), mode="r"), zarr_a)
    dataset_b = registry.register_from_root(zarr.open_group(str(zarr_b), mode="r"), zarr_b)

    assert dataset_a.startswith("session_a:z")
    assert dataset_b != dataset_a
    assert dataset_b.startswith("session_a:z")

    rows = registry.conn.execute(
        """
        SELECT dataset_id, session_uuid
        FROM datasets
        ORDER BY dataset_id;
        """
    ).fetchall()
    assert len(rows) == 2
    assert {str(row["session_uuid"]) for row in rows} == {"session_a"}
    registry.close()


def test_register_from_root_maps_days_post_fertilization_to_dpf(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    zarr_path = tmp_path / "recordings" / "session_dpf" / "zarr" / "session_dpf_training.zarr"
    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(zarr_path), mode="w")
    root.attrs["session_uuid"] = "session_dpf"
    analysis_meta = root.create_group("analysis_metadata")
    analysis_meta.attrs["subject_metadata"] = json.dumps(
        {
            "fish_id": "fish-dpf-1",
            "dish_id": "dish-dpf-1",
            "cross_id": "cross-dpf-1",
            "days_post_fertilization": "12",
            "genotype": "genotype_dpf",
            "line_strain": "line_dpf",
            "species": "Danio rerio",
            "sex": "unknown",
            "subject_count": "1",
        }
    )

    dataset_id = registry.register_from_root(zarr.open_group(str(zarr_path), mode="r"), zarr_path)
    row = registry.conn.execute(
        """
        SELECT fish_id, dish_id, cross_id, dpf_at_acquisition
        FROM provenance
        WHERE dataset_id = ?;
        """,
        (dataset_id,),
    ).fetchone()
    assert row is not None
    assert row["fish_id"] == "fish-dpf-1"
    assert row["dish_id"] == "dish-dpf-1"
    assert row["cross_id"] == "cross-dpf-1"
    assert int(row["dpf_at_acquisition"]) == 12
    registry.close()


def test_backfill_dataset_lineage_from_training_set_membership(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    source_a = tmp_path / "recording_a.zarr"
    source_b = tmp_path / "recording_b.zarr"
    merged = tmp_path / "training" / "datasets" / "detect_set_v001" / "zarr" / "detect_set_v001_merged.zarr"
    registry.upsert_dataset("source_a", session_uuid="source_a", zarr_path=source_a)
    registry.upsert_dataset("source_b", session_uuid="source_b", zarr_path=source_b)
    registry.upsert_dataset("detect_set_v001_merged", session_uuid="detect_set_v001_merged", zarr_path=merged)
    registry.upsert_training_set(
        set_id="detect_set_v001",
        name="detect_set",
        query_filter=None,
        dataset_ids=["source_a", "source_b", "detect_set_v001_merged"],
    )

    summary = _backfill_dataset_lineage(registry, dry_run=False)
    assert summary["sets_scanned"] == 1
    assert summary["merged_scanned"] == 1
    assert summary["relationships_changed"] == 1

    rows = registry.conn.execute(
        """
        SELECT parent_dataset_id
        FROM dataset_lineage_current
        WHERE child_dataset_id = ? AND relationship_type = 'training_merge_source'
        ORDER BY parent_dataset_id;
        """,
        ("detect_set_v001_merged",),
    ).fetchall()
    assert [str(row["parent_dataset_id"]) for row in rows] == ["source_a", "source_b"]
    registry.close()


def test_integrity_flags_merged_dataset_missing_lineage(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    merged = tmp_path / "training" / "datasets" / "pose_set_v001" / "zarr" / "pose_set_v001_merged.zarr"
    registry.upsert_dataset("pose_set_v001_merged", session_uuid="pose_set_v001_merged", zarr_path=merged)

    issues = _check_registry_integrity(registry)
    codes = {issue.code for issue in issues}
    assert "merged_dataset_missing_lineage" in codes
    registry.close()


def test_integrity_flags_dataset_lineage_cycle(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    dataset_a = tmp_path / "a.zarr"
    dataset_b = tmp_path / "b.zarr"
    registry.upsert_dataset("dataset_a", session_uuid="dataset_a", zarr_path=dataset_a)
    registry.upsert_dataset("dataset_b", session_uuid="dataset_b", zarr_path=dataset_b)
    registry.replace_dataset_lineage(
        child_dataset_id="dataset_a",
        parent_dataset_ids=["dataset_b"],
        relationship_type="training_merge_source",
    )
    registry.replace_dataset_lineage(
        child_dataset_id="dataset_b",
        parent_dataset_ids=["dataset_a"],
        relationship_type="training_merge_source",
    )

    issues = _check_registry_integrity(registry)
    cycle_codes = [issue.code for issue in issues if issue.code == "dataset_lineage_cycle"]
    assert cycle_codes
    registry.close()


def test_integrity_flags_derived_dataset_missing_recording_id_single_parent(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    source = tmp_path / "source.zarr"
    child = tmp_path / "derived.zarr"
    registry.upsert_dataset(
        "source_a",
        session_uuid="source_a",
        zarr_path=source,
        recording_id="rec_a",
        artifact_kind="source_recording",
    )
    registry.upsert_dataset(
        "derived_a",
        session_uuid="derived_a",
        zarr_path=child,
        recording_id=None,
        artifact_kind="derived_analysis",
    )
    registry.replace_dataset_lineage(
        child_dataset_id="derived_a",
        parent_dataset_ids=["source_a"],
        relationship_type="analysis_source",
    )

    issues = _check_registry_integrity(registry)
    codes = {issue.code for issue in issues}
    assert "derived_dataset_missing_recording_id_single_parent" in codes
    registry.close()


def test_integrity_flags_derived_dataset_non_null_recording_id_multi_parent(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    source_a = tmp_path / "source_a.zarr"
    source_b = tmp_path / "source_b.zarr"
    child = tmp_path / "derived.zarr"
    registry.upsert_dataset(
        "source_a",
        session_uuid="source_a",
        zarr_path=source_a,
        recording_id="rec_a",
        artifact_kind="source_recording",
    )
    registry.upsert_dataset(
        "source_b",
        session_uuid="source_b",
        zarr_path=source_b,
        recording_id="rec_b",
        artifact_kind="source_recording",
    )
    registry.upsert_dataset(
        "derived_multi",
        session_uuid="derived_multi",
        zarr_path=child,
        recording_id="rec_a",
        artifact_kind="derived_analysis",
    )
    registry.replace_dataset_lineage(
        child_dataset_id="derived_multi",
        parent_dataset_ids=["source_a", "source_b"],
        relationship_type="analysis_source",
    )

    issues = _check_registry_integrity(registry)
    codes = {issue.code for issue in issues}
    assert "derived_dataset_recording_id_non_null_multi_parent" in codes
    registry.close()


def test_integrity_accepts_derived_dataset_single_parent_matching_recording_id(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    source = tmp_path / "source.zarr"
    child = tmp_path / "derived.zarr"
    registry.upsert_dataset(
        "source_a",
        session_uuid="source_a",
        zarr_path=source,
        recording_id="rec_a",
        artifact_kind="source_recording",
    )
    registry.upsert_dataset(
        "derived_ok",
        session_uuid="derived_ok",
        zarr_path=child,
        recording_id="rec_a",
        artifact_kind="derived_analysis",
    )
    registry.replace_dataset_lineage(
        child_dataset_id="derived_ok",
        parent_dataset_ids=["source_a"],
        relationship_type="analysis_source",
    )

    issues = _check_registry_integrity(registry)
    codes = {issue.code for issue in issues}
    assert "derived_dataset_missing_recording_id_single_parent" not in codes
    assert "derived_dataset_recording_id_mismatch_single_parent" not in codes
    assert "derived_dataset_recording_id_non_null_multi_parent" not in codes
    registry.close()


def test_integrity_flags_source_protocol_name_mismatch(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    source = tmp_path / "source.zarr"
    registry.upsert_dataset(
        "source_a",
        session_uuid="source_a",
        zarr_path=source,
        recording_id="rec_a",
        artifact_kind="source_recording",
    )
    registry.conn.execute(
        """
        INSERT INTO recordings (
            recording_id, session_uuid, recording_name, recording_path, recording_type,
            recording_subtype, behavior_mode, artifact_schema_id, protocol_name,
            created_utc, updated_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'));
        """,
        ("rec_a", "source_a", "rec_a", str(tmp_path), "behavior", "free", "free", "behavior_v1", "ProtocolA"),
    )
    registry.conn.execute(
        """
        INSERT INTO provenance (dataset_id, protocol_name, subject_count)
        VALUES (?, ?, ?);
        """,
        ("source_a", "ProtocolB", 1),
    )
    registry.conn.commit()

    issues = _check_registry_integrity(registry)
    codes = {issue.code for issue in issues}
    assert "source_protocol_name_mismatch" in codes
    registry.close()


def test_integrity_flags_source_subject_count_invalid(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    source = tmp_path / "source.zarr"
    registry.upsert_dataset(
        "source_a",
        session_uuid="source_a",
        zarr_path=source,
        recording_id="rec_a",
        artifact_kind="source_recording",
    )
    registry.conn.execute(
        """
        INSERT INTO recordings (
            recording_id, session_uuid, recording_name, recording_path, recording_type,
            recording_subtype, behavior_mode, artifact_schema_id, protocol_name,
            created_utc, updated_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'));
        """,
        ("rec_a", "source_a", "rec_a", str(tmp_path), "behavior", "free", "free", "behavior_v1", "ProtocolA"),
    )
    registry.conn.execute(
        """
        INSERT INTO provenance (dataset_id, protocol_name, subject_count)
        VALUES (?, ?, ?);
        """,
        ("source_a", "ProtocolA", 0),
    )
    registry.conn.commit()

    issues = _check_registry_integrity(registry)
    codes = {issue.code for issue in issues}
    assert "source_subject_count_invalid" in codes
    registry.close()


def test_integrity_flags_source_dish_design_mismatch(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    source = tmp_path / "source.zarr"
    registry.upsert_dataset(
        "source_a",
        session_uuid="source_a",
        zarr_path=source,
        recording_id="rec_a",
        artifact_kind="source_recording",
    )
    registry.conn.execute(
        """
        INSERT INTO recordings (
            recording_id, session_uuid, recording_name, recording_path, recording_type,
            recording_subtype, behavior_mode, artifact_schema_id, protocol_name, dish_design,
            created_utc, updated_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'));
        """,
        (
            "rec_a",
            "source_a",
            "rec_a",
            str(tmp_path),
            "behavior",
            "free",
            "free",
            "behavior_v1",
            "ProtocolA",
            "6_well_plate",
        ),
    )
    registry.conn.execute(
        """
        INSERT INTO provenance (dataset_id, protocol_name, dish_design, subject_count)
        VALUES (?, ?, ?, ?);
        """,
        ("source_a", "ProtocolA", "12_well_plate", 1),
    )
    registry.conn.commit()

    issues = _check_registry_integrity(registry)
    codes = {issue.code for issue in issues}
    assert "source_dish_design_mismatch" in codes
    registry.close()


def test_integrity_accepts_source_protocol_and_subject_count_consistent(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    source = tmp_path / "source.zarr"
    registry.upsert_dataset(
        "source_a",
        session_uuid="source_a",
        zarr_path=source,
        recording_id="rec_a",
        artifact_kind="source_recording",
    )
    registry.conn.execute(
        """
        INSERT INTO recordings (
            recording_id, session_uuid, recording_name, recording_path, recording_type,
            recording_subtype, behavior_mode, artifact_schema_id, protocol_name, dish_design,
            created_utc, updated_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'));
        """,
        (
            "rec_a",
            "source_a",
            "rec_a",
            str(tmp_path),
            "behavior",
            "free",
            "free",
            "behavior_v1",
            "ProtocolA",
            "6_well_plate",
        ),
    )
    registry.conn.execute(
        """
        INSERT INTO provenance (dataset_id, protocol_name, dish_design, subject_count)
        VALUES (?, ?, ?, ?);
        """,
        ("source_a", "ProtocolA", "6_well_plate", 1),
    )
    registry.conn.commit()

    issues = _check_registry_integrity(registry)
    codes = {issue.code for issue in issues}
    assert "source_protocol_name_mismatch" not in codes
    assert "source_dish_design_mismatch" not in codes
    assert "source_subject_count_invalid" not in codes
    registry.close()


def test_integrity_flags_required_view_missing(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.conn.execute("DROP VIEW IF EXISTS recording_overview;")
    registry.conn.commit()

    issues = _check_registry_integrity(registry)
    issue_codes = {(issue.code, issue.run_id) for issue in issues}
    assert ("required_view_missing", "recording_overview") in issue_codes
    registry.close()


def test_integrity_flags_required_view_query_error(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.conn.execute("DROP VIEW IF EXISTS recording_overview;")
    registry.conn.execute(
        """
        CREATE VIEW recording_overview AS
        SELECT * FROM does_not_exist_table;
        """
    )
    registry.conn.commit()

    issues = _check_registry_integrity(registry)
    issue_codes = {(issue.code, issue.run_id) for issue in issues}
    assert ("required_view_query_error", "recording_overview") in issue_codes
    registry.close()


def test_integrity_flags_required_view_missing_recording_subject_overview(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.conn.execute("DROP VIEW IF EXISTS recording_subject_overview;")
    registry.conn.commit()

    issues = _check_registry_integrity(registry)
    issue_codes = {(issue.code, issue.run_id) for issue in issues}
    assert ("required_view_missing", "recording_subject_overview") in issue_codes
    registry.close()


def test_integrity_flags_recording_subject_missing_subject(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.conn.execute(
        """
        INSERT INTO recordings (
            recording_id, session_uuid, recording_name, recording_path, recording_type,
            recording_subtype, behavior_mode, artifact_schema_id, created_utc, updated_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'));
        """,
        (
            "recording_a",
            "recording_a",
            "recording_a",
            str(tmp_path / "recordings" / "recording_a"),
            "behavior",
            "free",
            "free",
            "behavior_v1",
        ),
    )
    registry.conn.execute(
        """
        INSERT INTO crosses (cross_id, genotype, created_utc, updated_utc)
        VALUES (?, ?, datetime('now'), datetime('now'));
        """,
        ("cross_a", "wt"),
    )
    registry.conn.execute(
        """
        INSERT INTO dishes (dish_id, cross_id, created_utc, updated_utc)
        VALUES (?, ?, datetime('now'), datetime('now'));
        """,
        ("dish_a", "cross_a"),
    )
    registry.conn.execute(
        """
        INSERT INTO recording_subjects (
            recording_id, subject_id, dish_id, cross_id, created_utc, updated_utc
        )
        VALUES (?, ?, ?, ?, datetime('now'), datetime('now'));
        """,
        ("recording_a", "subject_missing", "dish_a", "cross_a"),
    )
    registry.conn.commit()

    issues = _check_registry_integrity(registry)
    codes = {issue.code for issue in issues}
    assert "recording_subject_missing_subject" in codes
    registry.close()


def test_dataset_lineage_self_edge_rejected_by_db_trigger(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    dataset_a = tmp_path / "a.zarr"
    registry.upsert_dataset("dataset_a", session_uuid="dataset_a", zarr_path=dataset_a)
    with pytest.raises(sqlite3.IntegrityError, match="self-edge"):
        registry.conn.execute(
            """
            INSERT INTO dataset_lineage (
                child_dataset_id, parent_dataset_id, relationship_type, created_utc, updated_utc
            )
            VALUES (?, ?, ?, datetime('now'), datetime('now'));
            """,
            ("dataset_a", "dataset_a", "training_merge_source"),
        )
    registry.close()


def test_dataset_lineage_audit_summary_counts(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    source_a = tmp_path / "recording_a.zarr"
    merged = tmp_path / "training" / "datasets" / "pose_set_v001" / "zarr" / "pose_set_v001_merged.zarr"
    registry.upsert_dataset("source_a", session_uuid="source_a", zarr_path=source_a)
    registry.upsert_dataset("pose_set_v001_merged", session_uuid="pose_set_v001_merged", zarr_path=merged)
    registry.upsert_training_set(
        set_id="pose_set_v001",
        name="pose_set",
        query_filter=None,
        dataset_ids=["source_a", "pose_set_v001_merged"],
    )

    before = _summarize_dataset_lineage_audit(registry)
    assert before.edge_count == 0
    assert before.merged_dataset_count == 1
    assert before.merged_missing_lineage_count == 1
    assert before.training_set_lineage_mismatch_count == 1

    _backfill_dataset_lineage(registry, dry_run=False)
    after = _summarize_dataset_lineage_audit(registry)
    assert after.edge_count == 1
    assert after.merged_dataset_count == 1
    assert after.merged_missing_lineage_count == 0
    assert after.training_set_lineage_mismatch_count == 0
    registry.close()


def test_remap_training_set_dataset_ids_maps_legacy_source_ids(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    source_a = tmp_path / "recording_a_training.zarr"
    source_b = tmp_path / "recording_b_training.zarr"
    merged = tmp_path / "training" / "datasets" / "detect_set_v001" / "zarr" / "detect_set_v001_merged.zarr"
    registry.upsert_dataset(
        "session_a:z111",
        session_uuid="session_a",
        zarr_path=source_a,
        artifact_kind="source_recording",
    )
    registry.upsert_dataset(
        "session_b:z222",
        session_uuid="session_b",
        zarr_path=source_b,
        artifact_kind="source_recording",
    )
    registry.upsert_dataset(
        "detect_set_v001_merged",
        session_uuid="detect_set_v001_merged",
        zarr_path=merged,
        artifact_kind="derived_training_merge",
    )
    registry.upsert_training_set(
        set_id="detect_set_v001",
        name="detect_set",
        query_filter=None,
        dataset_ids=["session_a", "session_b", "detect_set_v001_merged"],
    )

    dry_summary = _remap_training_set_dataset_ids(registry, dry_run=True)
    assert dry_summary["sets_scanned"] == 1
    assert dry_summary["sets_changed"] == 1
    assert dry_summary["ids_remapped"] == 2
    assert dry_summary["ids_unresolved"] == 0
    original_json = registry.conn.execute(
        "SELECT dataset_ids_json FROM training_sets WHERE set_id = ?;",
        ("detect_set_v001",),
    ).fetchone()
    assert original_json is not None
    assert json.loads(str(original_json["dataset_ids_json"])) == [
        "detect_set_v001_merged",
        "session_a",
        "session_b",
    ]

    apply_summary = _remap_training_set_dataset_ids(registry, dry_run=False)
    assert apply_summary["sets_scanned"] == 1
    assert apply_summary["sets_changed"] == 1
    assert apply_summary["ids_remapped"] == 2
    assert apply_summary["ids_unresolved"] == 0
    remapped_json = registry.conn.execute(
        "SELECT dataset_ids_json FROM training_sets WHERE set_id = ?;",
        ("detect_set_v001",),
    ).fetchone()
    assert remapped_json is not None
    assert json.loads(str(remapped_json["dataset_ids_json"])) == [
        "detect_set_v001_merged",
        "session_a:z111",
        "session_b:z222",
    ]
    registry.close()


def test_collect_and_delete_invalid_candidates(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    good = tmp_path / "good.zarr"
    nested = tmp_path / "good.zarr" / "detect_runs"
    nested_zarr = tmp_path / "good.zarr" / "subset.zarr"
    stale = tmp_path / "missing.zarr"

    registry.upsert_dataset("good", session_uuid="good", zarr_path=good)
    registry.upsert_dataset("nested", session_uuid=None, zarr_path=nested)
    registry.upsert_dataset("nested_zarr", session_uuid=None, zarr_path=nested_zarr)
    registry.upsert_dataset("stale", session_uuid=None, zarr_path=stale)
    registry.conn.execute("UPDATE datasets SET status = 'missing' WHERE dataset_id = 'stale';")
    registry.conn.commit()

    candidates = _collect_invalid_dataset_candidates(registry)
    by_id = {candidate.dataset_id: candidate for candidate in candidates}
    assert set(by_id) == {"nested", "stale"}
    assert by_id["nested"].reasons == ("nested_zarr_subpath",)
    assert by_id["stale"].reasons == ("status_missing",)

    # Dry run must not delete.
    assert _delete_dataset_ids(registry, ["nested", "stale"], dry_run=True) == 0
    still_present = {
        row["dataset_id"]
        for row in registry.conn.execute("SELECT dataset_id FROM datasets ORDER BY dataset_id;").fetchall()
    }
    assert still_present == {"good", "nested", "nested_zarr", "stale"}

    deleted = _delete_dataset_ids(registry, ["nested", "stale"], dry_run=False)
    assert deleted == 2
    remaining = {
        row["dataset_id"]
        for row in registry.conn.execute("SELECT dataset_id FROM datasets ORDER BY dataset_id;").fetchall()
    }
    assert remaining == {"good", "nested_zarr"}
    registry.close()


def test_collect_candidates_can_infer_missing_without_status(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.upsert_dataset("missing_active", session_uuid=None, zarr_path=tmp_path / "not_there.zarr")
    nested_path = tmp_path / "recording.zarr" / "detect_runs"
    nested_path.mkdir(parents=True)
    (nested_path / "zarr.json").write_text('{"zarr_format": 3, "node_type": "group"}', encoding="utf-8")
    registry.upsert_dataset(
        "nested_active",
        session_uuid=None,
        zarr_path=nested_path,
    )

    candidates = _collect_invalid_dataset_candidates(
        registry,
        include_missing_scan=True,
    )
    by_id = {candidate.dataset_id: candidate for candidate in candidates}
    assert set(by_id) == {"missing_active", "nested_active"}
    assert by_id["missing_active"].reasons == ("status_missing",)
    assert by_id["nested_active"].reasons == ("nested_zarr_subpath",)
    registry.close()


def test_collect_missing_dataset_candidates_excludes_nested_only(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.upsert_dataset("missing_active", session_uuid=None, zarr_path=tmp_path / "not_there.zarr")
    nested_path = tmp_path / "recording.zarr" / "detect_runs"
    nested_path.mkdir(parents=True)
    (nested_path / "zarr.json").write_text('{"zarr_format": 3, "node_type": "group"}', encoding="utf-8")
    registry.upsert_dataset("nested_active", session_uuid=None, zarr_path=nested_path)

    inferred = _collect_missing_dataset_candidates(
        registry,
        include_missing_scan=True,
    )
    by_id = {candidate.dataset_id: candidate for candidate in inferred}
    assert set(by_id) == {"missing_active"}
    assert by_id["missing_active"].reasons == ("status_missing",)

    registry.conn.execute("UPDATE datasets SET status = 'missing' WHERE dataset_id = 'nested_active';")
    registry.conn.commit()
    status_only = _collect_missing_dataset_candidates(registry)
    by_id = {candidate.dataset_id: candidate for candidate in status_only}
    assert set(by_id) == {"nested_active"}
    registry.close()


def test_collect_and_delete_failed_training_runs(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.record_training_run(
        run_id="run_failed",
        set_id="set_a",
        config_path=None,
        manifest_path=None,
        skeleton_id=None,
        model_path=None,
        metrics_path=None,
        status="failed",
    )
    registry.record_training_run(
        run_id="run_failed_caps",
        set_id="set_a",
        config_path=None,
        manifest_path=None,
        skeleton_id=None,
        model_path=None,
        metrics_path=None,
        status="FAILED",
    )
    registry.record_training_run(
        run_id="run_success",
        set_id="set_a",
        config_path=None,
        manifest_path=None,
        skeleton_id=None,
        model_path=None,
        metrics_path=None,
        status="success",
    )
    registry.record_training_run(
        run_id="run_in_progress",
        set_id="set_a",
        config_path=None,
        manifest_path=None,
        skeleton_id=None,
        model_path=None,
        metrics_path=None,
        status="in_progress",
    )
    registry.record_model_export(run_id="run_failed", export_type="onnx", path=tmp_path / "run_failed.onnx")
    registry.record_model_export(run_id="run_success", export_type="onnx", path=tmp_path / "run_success.onnx")

    status_values = _normalize_status_values(["failed"])
    candidates = _collect_failed_run_candidates(registry, status_values=status_values)
    candidate_ids = {candidate.run_id for candidate in candidates}
    assert candidate_ids == {"run_failed", "run_failed_caps"}

    # Dry run must not delete.
    assert _delete_training_run_ids(registry, sorted(candidate_ids), dry_run=True) == 0
    still_present = {
        row["run_id"]
        for row in registry.conn.execute("SELECT run_id FROM training_runs ORDER BY run_id;").fetchall()
    }
    assert still_present == {"run_failed", "run_failed_caps", "run_success", "run_in_progress"}

    deleted = _delete_training_run_ids(registry, sorted(candidate_ids), dry_run=False)
    assert deleted == 2
    remaining = {
        row["run_id"]
        for row in registry.conn.execute("SELECT run_id FROM training_runs ORDER BY run_id;").fetchall()
    }
    assert remaining == {"run_success", "run_in_progress"}

    export_rows = registry.conn.execute(
        "SELECT run_id FROM onnx_models ORDER BY run_id;"
    ).fetchall()
    assert [row["run_id"] for row in export_rows] == ["run_success"]
    registry.close()


def test_collect_stale_in_progress_run_candidates_filters_by_age_and_task(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.record_training_run(
        run_id="run_pose_stale",
        set_id="pose_set_v001",
        task_type="pose",
        config_path=None,
        manifest_path=None,
        skeleton_id=None,
        model_path=None,
        metrics_path=None,
        status="in_progress",
        final_metrics={"stage": "preflight_and_training", "status_detail": "training_started"},
    )
    registry.record_training_run(
        run_id="run_pose_fresh",
        set_id="pose_set_v001",
        task_type="pose",
        config_path=None,
        manifest_path=None,
        skeleton_id=None,
        model_path=None,
        metrics_path=None,
        status="in_progress",
        final_metrics={"stage": "preflight_and_training", "status_detail": "training_started"},
    )
    registry.record_training_run(
        run_id="run_detect_stale",
        set_id="detect_set_v001",
        task_type="detect",
        config_path=None,
        manifest_path=None,
        skeleton_id=None,
        model_path=None,
        metrics_path=None,
        status="in_progress",
        final_metrics={"stage": "preflight_and_training", "status_detail": "training_started"},
    )
    registry.record_training_run(
        run_id="run_success",
        set_id="pose_set_v001",
        task_type="pose",
        config_path=None,
        manifest_path=None,
        skeleton_id=None,
        model_path=None,
        metrics_path=None,
        status="success",
        final_metrics={"stage": "completed", "status_detail": "training_complete"},
    )
    registry.conn.execute(
        "UPDATE training_runs SET created_utc = ? WHERE run_id = ?;",
        ("2026-02-20T00:00:00+00:00", "run_pose_stale"),
    )
    registry.conn.execute(
        "UPDATE training_runs SET created_utc = ? WHERE run_id = ?;",
        ("2026-02-22T23:00:00+00:00", "run_pose_fresh"),
    )
    registry.conn.execute(
        "UPDATE training_runs SET created_utc = ? WHERE run_id = ?;",
        ("2026-02-20T00:00:00+00:00", "run_detect_stale"),
    )
    registry.conn.commit()

    now_utc = datetime(2026, 2, 23, 0, 0, tzinfo=timezone.utc)

    pose_candidates = _collect_stale_in_progress_run_candidates(
        registry,
        max_age_hours=24.0,
        task_filter="pose",
        now_utc=now_utc,
    )
    assert {candidate.run_id for candidate in pose_candidates} == {"run_pose_stale"}

    all_candidates = _collect_stale_in_progress_run_candidates(
        registry,
        max_age_hours=24.0,
        task_filter="all",
        now_utc=now_utc,
    )
    assert {candidate.run_id for candidate in all_candidates} == {
        "run_pose_stale",
        "run_detect_stale",
    }
    registry.close()


def test_reconcile_stale_in_progress_runs_marks_failed_with_audit_payload(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    run_id = "run_pose_stale"
    registry.record_training_run(
        run_id=run_id,
        set_id="pose_set_v001",
        task_type="pose",
        config_path=None,
        manifest_path=None,
        skeleton_id=None,
        model_path=None,
        metrics_path=None,
        status="in_progress",
        final_metrics={"stage": "preflight_and_training", "status_detail": "training_started"},
    )
    registry.conn.execute(
        "UPDATE training_runs SET created_utc = ? WHERE run_id = ?;",
        ("2026-02-20T00:00:00+00:00", run_id),
    )
    registry.conn.commit()

    now_utc = datetime(2026, 2, 23, 0, 0, tzinfo=timezone.utc)
    candidates = _collect_stale_in_progress_run_candidates(
        registry,
        max_age_hours=24.0,
        task_filter="pose",
        now_utc=now_utc,
    )
    assert {candidate.run_id for candidate in candidates} == {run_id}

    dry_run_reconciled = _reconcile_stale_in_progress_runs(
        registry,
        candidates=candidates,
        max_age_hours=24.0,
        task_filter="pose",
        dry_run=True,
        now_utc=now_utc,
    )
    assert dry_run_reconciled == 0
    status_before = registry.conn.execute(
        "SELECT status FROM training_runs WHERE run_id = ?;",
        (run_id,),
    ).fetchone()
    assert status_before is not None
    assert status_before["status"] == "in_progress"

    reconciled = _reconcile_stale_in_progress_runs(
        registry,
        candidates=candidates,
        max_age_hours=24.0,
        task_filter="pose",
        dry_run=False,
        now_utc=now_utc,
    )
    assert reconciled == 1

    run_row = registry.conn.execute(
        "SELECT status, final_metrics_json FROM training_runs WHERE run_id = ?;",
        (run_id,),
    ).fetchone()
    assert run_row is not None
    assert run_row["status"] == "failed"
    final_metrics = json.loads(str(run_row["final_metrics_json"] or "{}"))
    assert final_metrics["stage"] == "maintenance_reconcile"
    assert final_metrics["status_detail"] == "stale_in_progress_reconciled"
    assert final_metrics["error_type"] == "StaleInProgressRun"
    assert "older than 24h" in str(final_metrics["error_message"])
    assert final_metrics["reconciled_by"] == "fisheye.registry.maintenance"
    assert final_metrics["previous_run_status"] == "in_progress"
    assert final_metrics["in_progress_since_utc"] == "2026-02-20T00:00:00+00:00"

    model_row = registry.conn.execute(
        "SELECT status, final_metrics_json FROM training_models WHERE run_id = ?;",
        (run_id,),
    ).fetchone()
    assert model_row is not None
    assert model_row["status"] == "failed"
    model_metrics = json.loads(str(model_row["final_metrics_json"] or "{}"))
    assert model_metrics["status_detail"] == "stale_in_progress_reconciled"
    registry.close()


def test_normalize_run_ids_supports_repeat_and_comma_input() -> None:
    assert _normalize_run_ids(["run_a, run_b", "run_c", "run_b"]) == ("run_a", "run_b", "run_c")
    assert _normalize_run_ids(None) == ()


def test_resolve_existing_run_ids_splits_existing_and_missing(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.record_training_run(
        run_id="run_exists",
        set_id="set_a",
        config_path=None,
        manifest_path=None,
        skeleton_id=None,
        model_path=None,
        metrics_path=None,
        status="success",
    )
    existing, missing = _resolve_existing_run_ids(registry, ["run_exists", "run_missing"])
    assert existing == ["run_exists"]
    assert missing == ["run_missing"]
    registry.close()


def test_normalize_set_ids_supports_repeat_and_comma_input() -> None:
    assert _normalize_set_ids(["set_a, set_b", "set_c", "set_b"]) == ("set_a", "set_b", "set_c")
    assert _normalize_set_ids(None) == ()


def test_collect_set_delete_candidates_and_linked_runs(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.upsert_training_set(
        set_id="set_empty",
        name="empty",
        query_filter=None,
        dataset_ids=[],
    )
    registry.upsert_training_set(
        set_id="set_linked",
        name="linked",
        query_filter=None,
        dataset_ids=[],
    )
    registry.record_training_run(
        run_id="run_1",
        set_id="set_linked",
        config_path=None,
        manifest_path=None,
        skeleton_id=None,
        model_path=None,
        metrics_path=None,
        status="success",
    )
    registry.record_training_run(
        run_id="run_2",
        set_id="set_linked",
        config_path=None,
        manifest_path=None,
        skeleton_id=None,
        model_path=None,
        metrics_path=None,
        status="failed",
    )

    candidates = _collect_set_delete_candidates(registry, ["set_empty", "set_linked", "set_missing"])
    by_id = {candidate.set_id: candidate for candidate in candidates}
    assert by_id["set_empty"].exists is True
    assert by_id["set_empty"].run_count == 0
    assert by_id["set_linked"].exists is True
    assert by_id["set_linked"].run_count == 2
    assert by_id["set_missing"].exists is False
    assert by_id["set_missing"].run_count == 0

    run_ids = _collect_run_ids_for_set_ids(registry, ["set_linked"])
    assert run_ids == ["run_2", "run_1"]
    registry.close()


def test_is_safe_artifact_path_blocks_recordings_and_outside(tmp_path: Path) -> None:
    root = tmp_path / "training"
    root.mkdir()
    safe_path = (root / "set_a" / "file.txt").resolve()
    safe_path.parent.mkdir(parents=True)
    safe_path.write_text("x", encoding="utf-8")
    ok, reason = _is_safe_artifact_path(safe_path, [root.resolve()])
    assert ok is True
    assert reason == "ok"

    outside_path = (tmp_path / "other" / "file.txt").resolve()
    outside_path.parent.mkdir(parents=True)
    outside_path.write_text("y", encoding="utf-8")
    ok, reason = _is_safe_artifact_path(outside_path, [root.resolve()])
    assert ok is False
    assert reason == "outside_training_artifact_roots"

    recordings_path = Path("/nvme1/recordings/example.zarr")
    ok, reason = _is_safe_artifact_path(recordings_path, [Path("/nvme1").resolve()])
    assert ok is False
    assert reason == "recordings_path_blocked"


def test_collect_run_artifact_paths_and_delete_plan(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    run_id = "run_cleanup"
    run_dir = tmp_path / "models" / run_id
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True)
    model_path = weights_dir / "best.pt"
    metrics_path = run_dir / "results.csv"
    onnx_path = run_dir / "exports" / "onnx" / f"{run_id}.onnx"
    onnx_path.parent.mkdir(parents=True)
    model_path.write_text("model", encoding="utf-8")
    metrics_path.write_text("metrics", encoding="utf-8")
    onnx_path.write_text("onnx", encoding="utf-8")

    registry.record_training_run(
        run_id=run_id,
        set_id="set_a",
        config_path=None,
        manifest_path=None,
        skeleton_id=None,
        model_path=model_path,
        metrics_path=metrics_path,
        status="success",
    )
    registry.record_onnx_model(
        run_id=run_id,
        set_id="set_a",
        skeleton_id=None,
        detection_model_run_id=run_id,
        path=onnx_path,
        sha256=None,
        manifest_path=None,
        manifest_sha256=None,
        metadata=None,
    )

    candidates = _collect_run_artifact_paths(registry, [run_id])
    assert run_dir.resolve() in candidates
    plan = _build_file_delete_plan(candidates, artifact_roots=[(tmp_path / "models").resolve()])
    assert run_dir.resolve() in plan.existing_paths
    assert plan.existing_bytes > 0

    deleted = _delete_paths(plan.existing_paths, dry_run=False)
    assert deleted >= 1
    assert not run_dir.exists()
    registry.close()


def test_collect_set_artifact_paths_includes_model_task_subdirs(tmp_path: Path) -> None:
    roots = [
        (tmp_path / "datasets").resolve(),
        (tmp_path / "models").resolve(),
    ]
    set_id = "detect_cedar_shadow_v005"
    paths = _collect_set_artifact_paths([set_id], roots)
    path_set = {path.resolve() for path in paths}
    assert (roots[0] / set_id).resolve() in path_set
    assert (roots[1] / "detect" / set_id).resolve() in path_set
    assert (roots[1] / "pose" / set_id).resolve() in path_set


def test_collect_and_delete_empty_training_sets(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.upsert_training_set(
        set_id="set_empty_1",
        name="empty one",
        query_filter=None,
        dataset_ids=["dataset_a"],
    )
    registry.upsert_training_set(
        set_id="set_linked",
        name="linked",
        query_filter=None,
        dataset_ids=["dataset_b"],
    )
    registry.upsert_training_set(
        set_id="set_empty_2",
        name="empty two",
        query_filter=None,
        dataset_ids=[],
    )
    registry.record_training_run(
        run_id="run_linked",
        set_id="set_linked",
        config_path=None,
        manifest_path=None,
        skeleton_id=None,
        model_path=None,
        metrics_path=None,
        status="success",
    )
    registry.record_training_run(
        run_id="run_unlinked",
        set_id=None,
        config_path=None,
        manifest_path=None,
        skeleton_id=None,
        model_path=None,
        metrics_path=None,
        status="success",
    )

    candidates = _collect_empty_training_set_candidates(registry)
    candidate_ids = {candidate.set_id for candidate in candidates}
    assert candidate_ids == {"set_empty_1", "set_empty_2"}

    assert _delete_training_set_ids(registry, sorted(candidate_ids), dry_run=True) == 0
    still_present = {
        row["set_id"]
        for row in registry.conn.execute("SELECT set_id FROM training_sets ORDER BY set_id;").fetchall()
    }
    assert still_present == {"set_empty_1", "set_empty_2", "set_linked"}

    deleted = _delete_training_set_ids(registry, sorted(candidate_ids), dry_run=False)
    assert deleted == 2
    remaining = {
        row["set_id"]
        for row in registry.conn.execute("SELECT set_id FROM training_sets ORDER BY set_id;").fetchall()
    }
    assert remaining == {"set_linked"}
    registry.close()


def test_backfill_model_tables_from_legacy_rows(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    config = tmp_path / "cfg.yaml"
    model = tmp_path / "best.pt"
    metrics = tmp_path / "results.csv"
    onnx = tmp_path / "best.onnx"
    trt = tmp_path / "best_fp16.engine"
    config.write_text("cfg", encoding="utf-8")
    model.write_text("model", encoding="utf-8")
    metrics.write_text("metrics", encoding="utf-8")
    onnx.write_text("onnx", encoding="utf-8")
    trt.write_text("trt", encoding="utf-8")

    registry.record_training_run(
        run_id="run_a",
        set_id="set_a",
        config_path=config,
        manifest_path=None,
        skeleton_id=None,
        model_path=model,
        metrics_path=metrics,
        status="success",
        final_metrics={"mAP50": 0.9},
    )
    # Legacy model_exports rows that backfill reads from.
    registry.conn.execute(
        """
        INSERT INTO model_exports (run_id, export_type, path, manifest_path, metadata_json, created_utc)
        VALUES (?, 'onnx', ?, NULL, ?, datetime('now'));
        """,
        (
            "run_a",
            str(onnx),
            '{"sha256":"onnx_sha","manifest_sha256":"onnx_manifest_sha","nms":{"conf":0.31,"iou":0.67,"topk":2}}',
        ),
    )
    registry.conn.execute(
        """
        INSERT INTO model_exports (run_id, export_type, path, manifest_path, metadata_json, created_utc)
        VALUES (?, 'tensorrt', ?, NULL, ?, datetime('now'));
        """,
        (
            "run_a",
            str(trt),
            '{"sha256":"trt_sha","manifest_sha256":"trt_manifest_sha","precision":"fp16","nms_conf":0.29,"nms_iou":0.63,"nms_topk":4}',
        ),
    )
    registry.conn.commit()

    # Simulate pre-migration registry by removing new tables.
    registry.conn.execute("DELETE FROM training_models;")
    registry.conn.execute("DELETE FROM onnx_models;")
    registry.conn.execute("DELETE FROM tensorrt_models;")
    registry.conn.commit()

    dry = _backfill_model_tables(registry, dry_run=True)
    assert dry["detection_missing"] == 1
    assert dry["onnx_missing"] == 1
    assert dry["tensorrt_missing"] == 1
    assert dry["detection_inserted"] == 0
    assert dry["onnx_inserted"] == 0
    assert dry["tensorrt_inserted"] == 0

    applied = _backfill_model_tables(registry, dry_run=False)
    assert applied["detection_inserted"] == 1
    assert applied["onnx_inserted"] == 1
    assert applied["tensorrt_inserted"] == 1

    detection_count = registry.conn.execute("SELECT COUNT(*) AS n FROM training_models;").fetchone()["n"]
    onnx_count = registry.conn.execute("SELECT COUNT(*) AS n FROM onnx_models;").fetchone()["n"]
    trt_count = registry.conn.execute("SELECT COUNT(*) AS n FROM tensorrt_models;").fetchone()["n"]
    assert detection_count == 1
    assert onnx_count == 1
    assert trt_count == 1
    onnx_row = registry.conn.execute(
        "SELECT nms_conf, nms_iou, nms_topk FROM onnx_models WHERE run_id='run_a';"
    ).fetchone()
    trt_row = registry.conn.execute(
        "SELECT nms_conf, nms_iou, nms_topk FROM tensorrt_models WHERE run_id='run_a';"
    ).fetchone()
    assert onnx_row is not None
    assert trt_row is not None
    assert float(onnx_row["nms_conf"]) == pytest.approx(0.31)
    assert float(onnx_row["nms_iou"]) == pytest.approx(0.67)
    assert int(onnx_row["nms_topk"]) == 2
    assert float(trt_row["nms_conf"]) == pytest.approx(0.29)
    assert float(trt_row["nms_iou"]) == pytest.approx(0.63)
    assert int(trt_row["nms_topk"]) == 4

    # Idempotent on repeat.
    repeat = _backfill_model_tables(registry, dry_run=False)
    assert repeat["detection_inserted"] == 0
    assert repeat["onnx_inserted"] == 0
    assert repeat["tensorrt_inserted"] == 0
    registry.close()


def test_backfill_keypoint_quality_dry_run_and_apply(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    zarr_path = tmp_path / "quality_sample.zarr"
    _create_quality_zarr(zarr_path)
    root = zarr.open_group(str(zarr_path), mode="r")
    dataset_id = registry.register_from_root(root, zarr_path)
    registry.conn.execute("DELETE FROM keypoint_quality WHERE dataset_id = ?;", (dataset_id,))
    registry.conn.commit()

    dry = _backfill_keypoint_quality(
        registry,
        dry_run=True,
        scope_paths=None,
        refresh=False,
    )
    assert dry["datasets_scanned"] == 1
    assert dry["rows_inserted"] == 1
    assert dry["rows_updated"] == 0
    assert dry["rows_deleted"] == 0

    applied = _backfill_keypoint_quality(
        registry,
        dry_run=False,
        scope_paths=None,
        refresh=False,
    )
    assert applied["rows_inserted"] == 1
    row = registry.conn.execute(
        "SELECT review_state, review_intended_use, usable_keypoints_rate FROM keypoint_quality_current WHERE dataset_id = ?;",
        (dataset_id,),
    ).fetchone()
    assert row is not None
    assert row["review_state"] == "approved"
    assert row["review_intended_use"] == "training"
    assert float(row["usable_keypoints_rate"]) == 0.75
    registry.close()


def test_refresh_keypoint_quality_deletes_stale_rows(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    zarr_path = tmp_path / "quality_refresh.zarr"
    _create_quality_zarr(zarr_path)
    root = zarr.open_group(str(zarr_path), mode="r")
    dataset_id = registry.register_from_root(root, zarr_path)
    registry.conn.execute(
        """
        INSERT INTO keypoint_quality (
            dataset_id, refined_run, source_keypoint_run, quality_updated_utc
        ) VALUES (?, 'stale_refined', 'kp_old', datetime('now'));
        """,
        (dataset_id,),
    )
    registry.conn.commit()

    summary = _backfill_keypoint_quality(
        registry,
        dry_run=False,
        scope_paths=None,
        refresh=True,
    )
    assert summary["rows_deleted"] >= 1
    stale = registry.conn.execute(
        "SELECT COUNT(*) AS n FROM keypoint_quality WHERE dataset_id = ? AND refined_run = 'stale_refined';",
        (dataset_id,),
    ).fetchone()["n"]
    assert stale == 0
    registry.close()


def test_backfill_keypoint_profiles_dry_run_and_apply(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    zarr_path = tmp_path / "recordings" / "rec_profile" / "zarr" / "rec_profile_analysis.zarr"
    fake_root = _create_keypoint_profile_zarr(zarr_path, zarr_use="analysis")
    monkeypatch.setattr(
        "fisheye.registry.maintenance._import_zarr",
        lambda: _FakeZarrModule({str(zarr_path): fake_root}),
    )
    dataset_id = "dataset_profile_a"
    registry.upsert_dataset(
        dataset_id=dataset_id,
        session_uuid="session_profile_a",
        zarr_path=zarr_path,
        recording_id="recording_profile_a",
        artifact_kind="source_recording",
        zarr_use="analysis",
    )
    registry.conn.execute("DELETE FROM keypoint_data_profile WHERE dataset_id = ?;", (dataset_id,))
    registry.conn.commit()

    dry = _backfill_keypoint_profiles(
        registry,
        dry_run=True,
        scope_paths=None,
        refresh=False,
    )
    assert dry["datasets_scanned"] == 1
    assert dry["rows_inserted"] == 1
    assert dry["rows_updated"] == 0
    assert dry["rows_deleted"] == 0

    applied = _backfill_keypoint_profiles(
        registry,
        dry_run=False,
        scope_paths=None,
        refresh=False,
    )
    assert applied["rows_inserted"] == 1
    row = registry.conn.execute(
        """
        SELECT profile_run, keypoint_method, usable_rate, genotype, dpf_at_acquisition
        FROM keypoint_data_profile_latest
        WHERE dataset_id = ?;
        """,
        (dataset_id,),
    ).fetchone()
    assert row is not None
    assert str(row["profile_run"]) == "keypoint_profile_001"
    assert str(row["keypoint_method"]) == "traditional_pose"
    assert float(row["usable_rate"]) == pytest.approx(0.75)
    assert str(row["genotype"]) == "Tg(elavl3:gcamp7f)"
    assert int(row["dpf_at_acquisition"]) == 7
    registry.close()


def test_backfill_keypoint_profiles_handles_missing_get_lookup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    zarr_path = tmp_path / "recordings" / "rec_profile_get_miss" / "zarr" / "rec_profile_get_miss_training.zarr"
    _create_fake_zarr_store(zarr_path)
    root = _GetMissFakeGroup(attrs={"session_uuid": "keypoint_profile_session", "zarr_purpose": "training"})
    analysis = root.add_group("analysis")
    profile_parent = analysis.add_group("keypoint_profile_runs", attrs={"latest": "keypoint_profile_001"})
    profile = profile_parent.add_group("keypoint_profile_001")
    profile.attrs["profile_summary"] = {
        "created_at_utc": "2026-02-24T00:00:00+00:00",
        "dataset": {"recording_id": "recording_profile", "zarr_use": "training"},
        "source": {
            "keypoint_method": "traditional_pose",
            "keypoint_path": "keypoints_runs/keypoints_001",
            "keypoint_run": "keypoints_001",
            "skeleton_id": "fish_v1",
            "kpt_shape": [3, 3],
        },
        "quality": {
            "rows_total": 4,
            "rows_usable": 3,
            "usable_keypoints_total": 9,
            "usable_rate": 0.75,
        },
        "geometry": {
            "triangle_area": {"stats": {"p10": 0.1, "p50": 0.2, "p90": 0.3}},
            "min_angle": {"stats": {"p10": 10.0, "p50": 20.0, "p90": 30.0}},
            "heading": {"stats": {"p10": -0.1, "p50": 0.0, "p90": 0.1}},
        },
        "composition": {"genotype": "Tg(elavl3:gcamp7f)", "dpf_at_acquisition": 7},
    }
    monkeypatch.setattr(
        "fisheye.registry.maintenance._import_zarr",
        lambda: _FakeZarrModule({str(zarr_path): root}),
    )
    dataset_id = "dataset_profile_get_miss"
    registry.upsert_dataset(
        dataset_id=dataset_id,
        session_uuid="session_profile_get_miss",
        zarr_path=zarr_path,
        recording_id="recording_profile_get_miss",
        artifact_kind="source_recording",
        zarr_use="training",
    )

    summary = _backfill_keypoint_profiles(
        registry,
        dry_run=False,
        scope_paths=None,
        refresh=False,
    )
    assert summary["rows_inserted"] == 1
    row = registry.conn.execute(
        "SELECT profile_run FROM keypoint_data_profile_latest WHERE dataset_id = ?;",
        (dataset_id,),
    ).fetchone()
    assert row is not None
    assert str(row["profile_run"]) == "keypoint_profile_001"
    registry.close()


def test_backfill_keypoint_profiles_scope_defaults_to_source_all_uses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    analysis_path = tmp_path / "recordings" / "rec_profile_a" / "zarr" / "rec_profile_a_analysis.zarr"
    training_path = tmp_path / "recordings" / "rec_profile_b" / "zarr" / "rec_profile_b_training.zarr"
    fake_analysis = _create_keypoint_profile_zarr(analysis_path, zarr_use="analysis")
    fake_training = _create_keypoint_profile_zarr(training_path, zarr_use="training")
    monkeypatch.setattr(
        "fisheye.registry.maintenance._import_zarr",
        lambda: _FakeZarrModule(
            {
                str(analysis_path): fake_analysis,
                str(training_path): fake_training,
            }
        ),
    )

    analysis_id = "dataset_profile_analysis"
    training_id = "dataset_profile_training"
    registry.upsert_dataset(
        dataset_id=analysis_id,
        session_uuid="session_profile_analysis",
        zarr_path=analysis_path,
        recording_id="recording_profile_analysis",
        artifact_kind="source_recording",
        zarr_use="analysis",
    )
    registry.upsert_dataset(
        dataset_id=training_id,
        session_uuid="session_profile_training",
        zarr_path=training_path,
        recording_id="recording_profile_training",
        artifact_kind="source_recording",
        zarr_use="training",
    )
    registry.conn.execute("DELETE FROM keypoint_data_profile WHERE dataset_id IN (?, ?);", (analysis_id, training_id))
    registry.conn.commit()

    dry = _backfill_keypoint_profiles(
        registry,
        dry_run=True,
        scope_paths=None,
        refresh=False,
    )
    assert dry["datasets_scanned"] == 2
    assert dry["rows_inserted"] == 2

    applied = _backfill_keypoint_profiles(
        registry,
        dry_run=False,
        scope_paths=None,
        refresh=False,
    )
    assert applied["rows_inserted"] == 2
    analysis_rows = registry.conn.execute(
        "SELECT COUNT(*) AS n FROM keypoint_data_profile WHERE dataset_id = ?;",
        (analysis_id,),
    ).fetchone()
    training_rows = registry.conn.execute(
        "SELECT COUNT(*) AS n FROM keypoint_data_profile WHERE dataset_id = ?;",
        (training_id,),
    ).fetchone()
    assert analysis_rows is not None and int(analysis_rows["n"]) == 1
    assert training_rows is not None and int(training_rows["n"]) == 1
    registry.close()


def test_refresh_keypoint_profiles_deletes_stale_rows_and_is_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    zarr_path = tmp_path / "recordings" / "rec_profile" / "zarr" / "rec_profile_refresh.zarr"
    fake_root = _create_keypoint_profile_zarr(zarr_path, zarr_use="analysis")
    monkeypatch.setattr(
        "fisheye.registry.maintenance._import_zarr",
        lambda: _FakeZarrModule({str(zarr_path): fake_root}),
    )

    dataset_id = "dataset_profile_refresh"
    registry.upsert_dataset(
        dataset_id=dataset_id,
        session_uuid="session_profile_refresh",
        zarr_path=zarr_path,
        recording_id="recording_profile_refresh",
        artifact_kind="source_recording",
        zarr_use="analysis",
    )
    registry.conn.execute(
        """
        INSERT INTO keypoint_data_profile (dataset_id, profile_run, updated_utc)
        VALUES
            (?, 'keypoint_profile_001', datetime('now')),
            (?, 'stale_profile', datetime('now'));
        """,
        (dataset_id, dataset_id),
    )
    registry.conn.commit()

    dry = _backfill_keypoint_profiles(
        registry,
        dry_run=True,
        scope_paths=None,
        refresh=True,
    )
    applied = _backfill_keypoint_profiles(
        registry,
        dry_run=False,
        scope_paths=None,
        refresh=True,
    )
    assert dry["rows_inserted"] == 0
    assert dry["rows_updated"] == 1
    assert dry["rows_deleted"] == 1
    assert dry["rows_skipped"] == 0
    assert applied["rows_inserted"] == 0
    assert applied["rows_updated"] == 1
    assert applied["rows_deleted"] == 1
    assert applied["rows_skipped"] == 0

    stale = registry.conn.execute(
        "SELECT COUNT(*) AS n FROM keypoint_data_profile WHERE dataset_id = ? AND profile_run = 'stale_profile';",
        (dataset_id,),
    ).fetchone()
    assert stale is not None and int(stale["n"]) == 0
    rows = registry.conn.execute(
        "SELECT profile_run FROM keypoint_data_profile WHERE dataset_id = ?;",
        (dataset_id,),
    ).fetchall()
    assert len(rows) == 1
    assert str(rows[0]["profile_run"]) == "keypoint_profile_001"
    registry.close()


def test_backfill_eye_mask_profiles_dry_run_and_apply(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    _ensure_eye_mask_data_profile_table(registry.conn)

    zarr_path = tmp_path / "recordings" / "rec_eye_profile" / "zarr" / "rec_eye_profile_analysis.zarr"
    _create_fake_zarr_store(zarr_path)
    fake_root = _FakeGroup(attrs={"session_uuid": "eye_profile_session"})
    monkeypatch.setattr(
        "fisheye.registry.maintenance._import_zarr",
        lambda: _FakeZarrModule({str(zarr_path): fake_root}),
    )
    monkeypatch.setattr(
        "fisheye.registry.maintenance._extract_eye_mask_profile_rows_for_maintenance",
        lambda *_args, **_kwargs: [
            _eye_mask_profile_row_payload(profile_run="eye_mask_profile_001", usable_rate=0.8),
        ],
    )

    dataset_id = "dataset_eye_profile_a"
    registry.upsert_dataset(
        dataset_id=dataset_id,
        session_uuid="session_eye_profile_a",
        zarr_path=zarr_path,
        recording_id="recording_eye_profile_a",
        artifact_kind="source_recording",
        zarr_use="analysis",
    )
    registry.conn.execute("DELETE FROM eye_mask_data_profile WHERE dataset_id = ?;", (dataset_id,))
    registry.conn.commit()

    dry = _backfill_eye_mask_profiles(
        registry,
        dry_run=True,
        scope_paths=None,
        refresh=False,
    )
    assert dry["datasets_scanned"] == 1
    assert dry["rows_inserted"] == 1
    assert dry["rows_updated"] == 0
    assert dry["rows_deleted"] == 0

    applied = _backfill_eye_mask_profiles(
        registry,
        dry_run=False,
        scope_paths=None,
        refresh=False,
    )
    assert applied["rows_inserted"] == 1
    row = registry.conn.execute(
        "SELECT profile_run FROM eye_mask_data_profile WHERE dataset_id = ?;",
        (dataset_id,),
    ).fetchone()
    assert row is not None
    assert str(row["profile_run"]) == "eye_mask_profile_001"
    registry.close()


def test_refresh_eye_mask_profiles_deletes_stale_rows_and_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    _ensure_eye_mask_data_profile_table(registry.conn)

    zarr_path = tmp_path / "recordings" / "rec_eye_profile" / "zarr" / "rec_eye_profile_refresh.zarr"
    _create_fake_zarr_store(zarr_path)
    fake_root = _FakeGroup(attrs={"session_uuid": "eye_profile_refresh"})
    monkeypatch.setattr(
        "fisheye.registry.maintenance._import_zarr",
        lambda: _FakeZarrModule({str(zarr_path): fake_root}),
    )
    monkeypatch.setattr(
        "fisheye.registry.maintenance._extract_eye_mask_profile_rows_for_maintenance",
        lambda *_args, **_kwargs: [
            _eye_mask_profile_row_payload(profile_run="eye_mask_profile_keep", usable_rate=0.85),
        ],
    )

    dataset_id = "dataset_eye_profile_refresh"
    registry.upsert_dataset(
        dataset_id=dataset_id,
        session_uuid="session_eye_profile_refresh",
        zarr_path=zarr_path,
        recording_id="recording_eye_profile_refresh",
        artifact_kind="source_recording",
        zarr_use="analysis",
    )
    registry.conn.execute(
        """
        INSERT INTO eye_mask_data_profile (dataset_id, profile_run, usable_rate, updated_utc)
        VALUES
            (?, 'eye_mask_profile_keep', 0.25, datetime('now')),
            (?, 'eye_mask_profile_stale', 0.75, datetime('now'));
        """,
        (dataset_id, dataset_id),
    )
    registry.conn.commit()

    dry = _backfill_eye_mask_profiles(
        registry,
        dry_run=True,
        scope_paths=None,
        refresh=True,
    )
    applied = _backfill_eye_mask_profiles(
        registry,
        dry_run=False,
        scope_paths=None,
        refresh=True,
    )
    for key in ("rows_inserted", "rows_updated", "rows_deleted", "rows_skipped"):
        assert dry[key] == applied[key]
    assert dry["rows_inserted"] == 0
    assert dry["rows_updated"] == 1
    assert dry["rows_deleted"] == 1
    assert dry["rows_skipped"] == 0

    rows = registry.conn.execute(
        "SELECT profile_run FROM eye_mask_data_profile WHERE dataset_id = ? ORDER BY profile_run;",
        (dataset_id,),
    ).fetchall()
    assert [str(row["profile_run"]) for row in rows] == ["eye_mask_profile_keep"]

    repeat = _backfill_eye_mask_profiles(
        registry,
        dry_run=False,
        scope_paths=None,
        refresh=True,
    )
    assert repeat["rows_inserted"] == 0
    assert repeat["rows_updated"] == 0
    assert repeat["rows_deleted"] == 0
    assert repeat["rows_skipped"] == 1
    registry.close()


def test_backfill_keypoint_performance_dry_run_and_apply(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    zarr_path = tmp_path / "recordings" / "rec_a" / "zarr" / "rec_a_analysis.zarr"
    _create_keypoint_performance_zarr(zarr_path)
    root = zarr.open_group(str(zarr_path), mode="r")
    dataset_id = registry.register_from_root(root, zarr_path)
    registry.conn.execute(
        "UPDATE datasets SET zarr_use = 'analysis' WHERE dataset_id = ?;",
        (dataset_id,),
    )
    registry.conn.execute("DELETE FROM keypoint_performance WHERE dataset_id = ?;", (dataset_id,))
    registry.conn.commit()

    dry = _backfill_keypoint_performance(
        registry,
        dry_run=True,
        scope_paths=None,
        refresh=False,
    )
    assert dry["datasets_scanned"] == 1
    assert dry["rows_inserted"] == 1
    assert dry["rows_updated"] == 0
    assert dry["rows_deleted"] == 0

    applied = _backfill_keypoint_performance(
        registry,
        dry_run=False,
        scope_paths=None,
        refresh=False,
    )
    assert applied["rows_inserted"] == 1
    row = registry.conn.execute(
        """
        SELECT keypoint_method, success_rate_percent, keypoints_per_second
        FROM keypoint_performance_latest
        WHERE dataset_id = ?;
        """,
        (dataset_id,),
    ).fetchone()
    assert row is not None
    assert str(row["keypoint_method"]) == "yolo_pose"
    assert float(row["success_rate_percent"]) == pytest.approx(75.0)
    assert float(row["keypoints_per_second"]) == pytest.approx(2.0)
    registry.close()


def test_backfill_keypoint_performance_scope_defaults_to_source_analysis(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    analysis_path = tmp_path / "recordings" / "rec_a" / "zarr" / "rec_a_analysis.zarr"
    training_path = tmp_path / "recordings" / "rec_a" / "zarr" / "rec_a_training.zarr"
    _create_keypoint_performance_zarr(analysis_path)
    _create_keypoint_performance_zarr(training_path)

    analysis_id = registry.register_from_root(zarr.open_group(str(analysis_path), mode="r"), analysis_path)
    training_id = registry.register_from_root(zarr.open_group(str(training_path), mode="r"), training_path)
    registry.conn.execute(
        "UPDATE datasets SET zarr_use = 'analysis' WHERE dataset_id = ?;",
        (analysis_id,),
    )
    registry.conn.execute(
        "UPDATE datasets SET zarr_use = 'training' WHERE dataset_id = ?;",
        (training_id,),
    )
    registry.conn.execute(
        "DELETE FROM keypoint_performance WHERE dataset_id IN (?, ?);",
        (analysis_id, training_id),
    )
    registry.conn.commit()

    dry_default = _backfill_keypoint_performance(
        registry,
        dry_run=True,
        scope_paths=None,
        refresh=False,
    )
    assert dry_default["datasets_scanned"] == 1
    assert dry_default["rows_inserted"] == 1

    applied_default = _backfill_keypoint_performance(
        registry,
        dry_run=False,
        scope_paths=None,
        refresh=False,
    )
    assert applied_default["rows_inserted"] == 1
    analysis_rows = registry.conn.execute(
        "SELECT COUNT(*) AS n FROM keypoint_performance WHERE dataset_id = ?;",
        (analysis_id,),
    ).fetchone()
    training_rows = registry.conn.execute(
        "SELECT COUNT(*) AS n FROM keypoint_performance WHERE dataset_id = ?;",
        (training_id,),
    ).fetchone()
    assert analysis_rows is not None and int(analysis_rows["n"]) == 1
    assert training_rows is not None and int(training_rows["n"]) == 0

    registry.conn.execute("DELETE FROM keypoint_performance;")
    registry.conn.commit()
    dry_all = _backfill_keypoint_performance(
        registry,
        dry_run=True,
        scope_paths=None,
        refresh=False,
        include_all_datasets=True,
    )
    assert dry_all["datasets_scanned"] == 2
    assert dry_all["rows_inserted"] == 2
    registry.close()


def test_main_no_action_message_includes_profile_flags() -> None:
    with pytest.raises(SystemExit) as exc:
        maintenance_main([])
    message = str(exc.value)
    assert "--backfill-keypoint-profiles" in message
    assert "--refresh-keypoint-profiles" in message
    assert "--backfill-eye-mask-profiles" in message
    assert "--refresh-eye-mask-profiles" in message


def test_main_backfill_keypoint_profiles_wiring_and_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    registry_path = tmp_path / "registry.sqlite"
    Registry(registry_path).close()
    calls: list[Dict[str, object]] = []

    def _fake_backfill(
        _registry: Registry,
        *,
        dry_run: bool,
        scope_paths: Optional[list[Path]],
        refresh: bool,
    ) -> Dict[str, int]:
        calls.append(
            {
                "dry_run": dry_run,
                "scope_paths": scope_paths,
                "refresh": refresh,
            }
        )
        return {
            "datasets_scanned": 5,
            "datasets_skipped_existing": 3,
            "datasets_missing": 1,
            "datasets_errors": 0,
            "datasets_no_profile": 2,
            "rows_inserted": 2,
            "rows_updated": 1,
            "rows_skipped": 3,
            "rows_deleted": 0,
        }

    monkeypatch.setattr(
        "fisheye.registry.maintenance._backfill_keypoint_profiles",
        _fake_backfill,
    )

    maintenance_main(
        [
            "--registry",
            str(registry_path),
            "--backfill-keypoint-profiles",
            "--dry-run",
        ]
    )
    assert calls == [{"dry_run": True, "scope_paths": None, "refresh": False}]
    output = capsys.readouterr().out
    assert "Keypoint profiles backfill: scope=source-recording-all-uses" in output
    assert "Dry run: would apply inserted=2 updated=1 deleted=0 unchanged=3 row(s)." in output


def test_main_backfill_eye_mask_profiles_wiring_and_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    registry_path = tmp_path / "registry.sqlite"
    Registry(registry_path).close()
    calls: list[Dict[str, object]] = []

    def _fake_backfill(
        _registry: Registry,
        *,
        dry_run: bool,
        scope_paths: Optional[list[Path]],
        refresh: bool,
    ) -> Dict[str, int]:
        calls.append(
            {
                "dry_run": dry_run,
                "scope_paths": scope_paths,
                "refresh": refresh,
            }
        )
        return {
            "datasets_scanned": 7,
            "datasets_skipped_existing": 2,
            "datasets_missing": 1,
            "datasets_errors": 0,
            "datasets_no_profile": 3,
            "rows_inserted": 2,
            "rows_updated": 1,
            "rows_skipped": 3,
            "rows_deleted": 4,
        }

    monkeypatch.setattr(
        "fisheye.registry.maintenance._backfill_eye_mask_profiles",
        _fake_backfill,
    )

    maintenance_main(
        [
            "--registry",
            str(registry_path),
            "--backfill-eye-mask-profiles",
            "--dry-run",
        ]
    )
    assert calls == [{"dry_run": True, "scope_paths": None, "refresh": False}]
    output = capsys.readouterr().out
    assert "Eye-mask profiles backfill: scope=source-recording-all-uses" in output
    assert "Dry run: would apply inserted=2 updated=1 deleted=4 unchanged=3 row(s)." in output


def test_backfill_detect_quality_refresh_apply_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    zarr_path = tmp_path / "recordings" / "detect_quality" / "zarr" / "detect_quality_analysis.zarr"
    roots_by_path = {
        str(zarr_path): _create_detect_quality_fake_zarr(
            zarr_path,
            refined_runs=("refined_detect_001",),
        )
    }
    monkeypatch.setattr(
        "fisheye.registry.maintenance._import_zarr",
        lambda: _FakeZarrModule(roots_by_path),
    )

    registry.upsert_dataset(
        dataset_id="dataset_detect_quality_idempotent",
        session_uuid="session_detect_quality_idempotent",
        zarr_path=zarr_path,
        recording_id="recording_detect_quality_idempotent",
        artifact_kind="source_recording",
        zarr_use="analysis",
    )

    first = _backfill_detect_quality(
        registry,
        dry_run=False,
        scope_paths=None,
        refresh=True,
    )
    assert first["datasets_scanned"] == 1
    assert first["rows_inserted"] == 1
    assert first["rows_updated"] == 0
    assert first["rows_deleted"] == 0
    assert first["rows_skipped"] == 0

    repeat_1 = _backfill_detect_quality(
        registry,
        dry_run=False,
        scope_paths=None,
        refresh=True,
    )
    repeat_2 = _backfill_detect_quality(
        registry,
        dry_run=False,
        scope_paths=None,
        refresh=True,
    )
    assert repeat_1 == repeat_2
    assert repeat_1["rows_inserted"] == 0
    assert repeat_1["rows_updated"] == 0
    assert repeat_1["rows_deleted"] == 0
    assert repeat_1["rows_skipped"] == 1

    count_row = registry.conn.execute(
        "SELECT COUNT(*) AS n FROM detect_quality WHERE dataset_id = ?;",
        ("dataset_detect_quality_idempotent",),
    ).fetchone()
    assert count_row is not None and int(count_row["n"]) == 1
    registry.close()


def test_backfill_detect_quality_refresh_apply_deletes_rows_when_source_disappears(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    zarr_path = tmp_path / "recordings" / "detect_quality" / "zarr" / "detect_quality_analysis.zarr"
    roots_by_path = {
        str(zarr_path): _create_detect_quality_fake_zarr(
            zarr_path,
            refined_runs=("refined_detect_001",),
        )
    }
    monkeypatch.setattr(
        "fisheye.registry.maintenance._import_zarr",
        lambda: _FakeZarrModule(roots_by_path),
    )

    registry.upsert_dataset(
        dataset_id="dataset_detect_quality_disappears",
        session_uuid="session_detect_quality_disappears",
        zarr_path=zarr_path,
        recording_id="recording_detect_quality_disappears",
        artifact_kind="source_recording",
        zarr_use="analysis",
    )

    seeded = _backfill_detect_quality(
        registry,
        dry_run=False,
        scope_paths=None,
        refresh=True,
    )
    assert seeded["rows_inserted"] == 1

    roots_by_path[str(zarr_path)] = _create_detect_quality_fake_zarr(
        zarr_path,
        refined_runs=(),
    )
    refreshed = _backfill_detect_quality(
        registry,
        dry_run=False,
        scope_paths=None,
        refresh=True,
    )
    assert refreshed["datasets_scanned"] == 1
    assert refreshed["datasets_no_quality"] == 1
    assert refreshed["rows_inserted"] == 0
    assert refreshed["rows_updated"] == 0
    assert refreshed["rows_deleted"] == 1
    assert refreshed["rows_skipped"] == 0

    count_row = registry.conn.execute(
        "SELECT COUNT(*) AS n FROM detect_quality WHERE dataset_id = ?;",
        ("dataset_detect_quality_disappears",),
    ).fetchone()
    assert count_row is not None and int(count_row["n"]) == 0
    registry.close()


def test_backfill_detect_quality_refresh_dry_run_and_apply_counts_are_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    zarr_path = tmp_path / "recordings" / "detect_quality" / "zarr" / "detect_quality_analysis.zarr"
    roots_by_path = {
        str(zarr_path): _create_detect_quality_fake_zarr(
            zarr_path,
            refined_runs=("refined_keep", "refined_update", "refined_new"),
        )
    }
    monkeypatch.setattr(
        "fisheye.registry.maintenance._import_zarr",
        lambda: _FakeZarrModule(roots_by_path),
    )

    dataset_id = "dataset_detect_quality_deterministic"
    registry.upsert_dataset(
        dataset_id=dataset_id,
        session_uuid="session_detect_quality_deterministic",
        zarr_path=zarr_path,
        recording_id="recording_detect_quality_deterministic",
        artifact_kind="source_recording",
        zarr_use="analysis",
    )

    seeded = _backfill_detect_quality(
        registry,
        dry_run=False,
        scope_paths=None,
        refresh=True,
    )
    assert seeded["rows_inserted"] == 3

    registry.conn.execute(
        "DELETE FROM detect_quality WHERE dataset_id = ? AND refined_run = 'refined_new';",
        (dataset_id,),
    )
    registry.conn.execute(
        "UPDATE detect_quality SET review_state = 'needs_review' WHERE dataset_id = ? AND refined_run = 'refined_update';",
        (dataset_id,),
    )
    registry.conn.execute(
        """
        INSERT INTO detect_quality (
            dataset_id, refined_run, source_detect_run, quality_updated_utc
        ) VALUES (?, 'refined_stale', 'detect_001', datetime('now'));
        """,
        (dataset_id,),
    )
    registry.conn.commit()

    dry = _backfill_detect_quality(
        registry,
        dry_run=True,
        scope_paths=None,
        refresh=True,
    )
    applied = _backfill_detect_quality(
        registry,
        dry_run=False,
        scope_paths=None,
        refresh=True,
    )
    for key in ("rows_inserted", "rows_updated", "rows_deleted", "rows_skipped"):
        assert dry[key] == applied[key]

    assert dry["rows_inserted"] == 1
    assert dry["rows_updated"] == 1
    assert dry["rows_deleted"] == 1
    assert dry["rows_skipped"] == 1

    refined_runs = [
        str(row["refined_run"])
        for row in registry.conn.execute(
            "SELECT refined_run FROM detect_quality WHERE dataset_id = ? ORDER BY refined_run;",
            (dataset_id,),
        ).fetchall()
    ]
    assert refined_runs == ["refined_keep", "refined_new", "refined_update"]
    registry.close()


def test_backfill_detect_performance_dry_run_and_apply(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    zarr_path = tmp_path / "recordings" / "rec_a" / "zarr" / "rec_a_analysis.zarr"
    _create_detect_performance_zarr(zarr_path)
    root = zarr.open_group(str(zarr_path), mode="r")
    dataset_id = registry.register_from_root(root, zarr_path)
    registry.conn.execute(
        "UPDATE datasets SET zarr_use = 'analysis' WHERE dataset_id = ?;",
        (dataset_id,),
    )
    registry.conn.execute("DELETE FROM detect_performance WHERE dataset_id = ?;", (dataset_id,))
    registry.conn.commit()

    dry = _backfill_detect_performance(
        registry,
        dry_run=True,
        scope_paths=None,
        refresh=False,
    )
    assert dry["datasets_scanned"] == 1
    assert dry["rows_inserted"] == 1
    assert dry["rows_updated"] == 0
    assert dry["rows_deleted"] == 0

    applied = _backfill_detect_performance(
        registry,
        dry_run=False,
        scope_paths=None,
        refresh=False,
    )
    assert applied["rows_inserted"] == 1
    row = registry.conn.execute(
        "SELECT detection_method, coverage_percent, inference_average_fps FROM detect_performance_latest WHERE dataset_id = ?;",
        (dataset_id,),
    ).fetchone()
    assert row is not None
    assert str(row["detection_method"]) == "yolo"
    assert float(row["coverage_percent"]) == pytest.approx(50.0)
    assert float(row["inference_average_fps"]) == pytest.approx(80.0)
    registry.close()


def test_backfill_detect_performance_scope_defaults_to_source_analysis(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    analysis_path = tmp_path / "recordings" / "rec_a" / "zarr" / "rec_a_analysis.zarr"
    training_path = tmp_path / "recordings" / "rec_a" / "zarr" / "rec_a_training.zarr"
    _create_detect_performance_zarr(analysis_path)
    _create_detect_performance_zarr(training_path)

    analysis_id = registry.register_from_root(zarr.open_group(str(analysis_path), mode="r"), analysis_path)
    training_id = registry.register_from_root(zarr.open_group(str(training_path), mode="r"), training_path)
    registry.conn.execute(
        "UPDATE datasets SET zarr_use = 'analysis' WHERE dataset_id = ?;",
        (analysis_id,),
    )
    registry.conn.execute(
        "UPDATE datasets SET zarr_use = 'training' WHERE dataset_id = ?;",
        (training_id,),
    )
    registry.conn.execute(
        "DELETE FROM detect_performance WHERE dataset_id IN (?, ?);",
        (analysis_id, training_id),
    )
    registry.conn.commit()

    dry_default = _backfill_detect_performance(
        registry,
        dry_run=True,
        scope_paths=None,
        refresh=False,
    )
    assert dry_default["datasets_scanned"] == 1
    assert dry_default["rows_inserted"] == 1

    applied_default = _backfill_detect_performance(
        registry,
        dry_run=False,
        scope_paths=None,
        refresh=False,
    )
    assert applied_default["rows_inserted"] == 1
    analysis_rows = registry.conn.execute(
        "SELECT COUNT(*) AS n FROM detect_performance WHERE dataset_id = ?;",
        (analysis_id,),
    ).fetchone()
    training_rows = registry.conn.execute(
        "SELECT COUNT(*) AS n FROM detect_performance WHERE dataset_id = ?;",
        (training_id,),
    ).fetchone()
    assert analysis_rows is not None and int(analysis_rows["n"]) == 1
    assert training_rows is not None and int(training_rows["n"]) == 0

    registry.conn.execute("DELETE FROM detect_performance;")
    registry.conn.commit()
    dry_all = _backfill_detect_performance(
        registry,
        dry_run=True,
        scope_paths=None,
        refresh=False,
        include_all_datasets=True,
    )
    assert dry_all["datasets_scanned"] == 2
    assert dry_all["rows_inserted"] == 2
    registry.close()


def test_backfill_crop_quality_dry_run_and_apply(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    zarr_path = tmp_path / "recordings" / "rec_a" / "zarr" / "rec_a_analysis.zarr"
    _create_crop_quality_zarr(zarr_path)
    dataset_id = registry.register_from_root(zarr.open_group(str(zarr_path), mode="r"), zarr_path)
    registry.conn.execute(
        "UPDATE datasets SET zarr_use = 'analysis' WHERE dataset_id = ?;",
        (dataset_id,),
    )
    registry.conn.execute("DELETE FROM crop_quality WHERE dataset_id = ?;", (dataset_id,))
    registry.conn.commit()

    dry = _backfill_crop_quality(
        registry,
        dry_run=True,
        scope_paths=None,
        refresh=False,
    )
    assert dry["datasets_scanned"] == 1
    assert dry["rows_inserted"] == 1
    assert dry["rows_updated"] == 0
    assert dry["rows_deleted"] == 0

    applied = _backfill_crop_quality(
        registry,
        dry_run=False,
        scope_paths=None,
        refresh=False,
    )
    assert applied["rows_inserted"] == 1
    row = registry.conn.execute(
        """
        SELECT review_state, review_intended_use, detection_source_type, percent_frames_with_crops
        FROM crop_quality_current
        WHERE dataset_id = ?;
        """,
        (dataset_id,),
    ).fetchone()
    assert row is not None
    assert str(row["review_state"]) == "approved"
    assert str(row["review_intended_use"]) == "training"
    assert str(row["detection_source_type"]) == "manual"
    assert float(row["percent_frames_with_crops"]) == pytest.approx(100.0)
    registry.close()


def test_backfill_crop_quality_scope_defaults_to_source_analysis(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    analysis_path = tmp_path / "recordings" / "rec_a" / "zarr" / "rec_a_analysis.zarr"
    training_path = tmp_path / "recordings" / "rec_a" / "zarr" / "rec_a_training.zarr"
    _create_crop_quality_zarr(analysis_path)
    _create_crop_quality_zarr(training_path)

    analysis_id = registry.register_from_root(zarr.open_group(str(analysis_path), mode="r"), analysis_path)
    training_id = registry.register_from_root(zarr.open_group(str(training_path), mode="r"), training_path)
    registry.conn.execute(
        "UPDATE datasets SET zarr_use = 'analysis' WHERE dataset_id = ?;",
        (analysis_id,),
    )
    registry.conn.execute(
        "UPDATE datasets SET zarr_use = 'training' WHERE dataset_id = ?;",
        (training_id,),
    )
    registry.conn.execute(
        "DELETE FROM crop_quality WHERE dataset_id IN (?, ?);",
        (analysis_id, training_id),
    )
    registry.conn.commit()

    dry_default = _backfill_crop_quality(
        registry,
        dry_run=True,
        scope_paths=None,
        refresh=False,
    )
    assert dry_default["datasets_scanned"] == 1
    assert dry_default["rows_inserted"] == 1

    applied_default = _backfill_crop_quality(
        registry,
        dry_run=False,
        scope_paths=None,
        refresh=False,
    )
    assert applied_default["rows_inserted"] == 1
    analysis_rows = registry.conn.execute(
        "SELECT COUNT(*) AS n FROM crop_quality WHERE dataset_id = ?;",
        (analysis_id,),
    ).fetchone()
    training_rows = registry.conn.execute(
        "SELECT COUNT(*) AS n FROM crop_quality WHERE dataset_id = ?;",
        (training_id,),
    ).fetchone()
    assert analysis_rows is not None and int(analysis_rows["n"]) == 1
    assert training_rows is not None and int(training_rows["n"]) == 0

    registry.conn.execute("DELETE FROM crop_quality;")
    registry.conn.commit()
    dry_all = _backfill_crop_quality(
        registry,
        dry_run=True,
        scope_paths=None,
        refresh=False,
        include_all_datasets=True,
    )
    assert dry_all["datasets_scanned"] == 2
    assert dry_all["rows_inserted"] == 2
    registry.close()


def test_backfill_detect_performance_handles_with_and_without_detect_runs(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    with_detect_path = tmp_path / "recordings" / "rec_with" / "zarr" / "rec_with_analysis.zarr"
    without_detect_path = tmp_path / "recordings" / "rec_without" / "zarr" / "rec_without_analysis.zarr"
    _create_detect_performance_zarr(with_detect_path)
    _create_detectless_zarr(without_detect_path, session_uuid="detectless_session_b")

    with_detect_id = registry.register_from_root(zarr.open_group(str(with_detect_path), mode="r"), with_detect_path)
    without_detect_id = registry.register_from_root(
        zarr.open_group(str(without_detect_path), mode="r"),
        without_detect_path,
    )
    registry.conn.execute(
        "UPDATE datasets SET zarr_use = 'analysis' WHERE dataset_id IN (?, ?);",
        (with_detect_id, without_detect_id),
    )
    registry.conn.execute(
        "DELETE FROM detect_performance WHERE dataset_id IN (?, ?);",
        (with_detect_id, without_detect_id),
    )
    registry.conn.commit()

    dry = _backfill_detect_performance(
        registry,
        dry_run=True,
        scope_paths=None,
        refresh=False,
    )
    assert dry["datasets_scanned"] == 2
    assert dry["rows_inserted"] == 1
    assert dry["datasets_no_performance"] == 1

    applied = _backfill_detect_performance(
        registry,
        dry_run=False,
        scope_paths=None,
        refresh=False,
    )
    assert applied["rows_inserted"] == 1
    assert applied["datasets_no_performance"] == 1

    repeat = _backfill_detect_performance(
        registry,
        dry_run=False,
        scope_paths=None,
        refresh=False,
    )
    assert repeat["datasets_scanned"] == 2
    assert repeat["datasets_skipped_existing"] == 1
    assert repeat["datasets_no_performance"] == 1
    assert repeat["rows_inserted"] == 0
    assert repeat["rows_updated"] == 0
    assert repeat["rows_deleted"] == 0
    assert repeat["rows_skipped"] >= 1
    registry.close()


def test_backfill_detect_performance_refresh_dry_run_is_deterministic(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    zarr_path = tmp_path / "recordings" / "rec_a" / "zarr" / "rec_a_analysis.zarr"
    _create_detect_performance_zarr(zarr_path)
    dataset_id = registry.register_from_root(zarr.open_group(str(zarr_path), mode="r"), zarr_path)
    registry.conn.execute(
        "UPDATE datasets SET zarr_use = 'analysis' WHERE dataset_id = ?;",
        (dataset_id,),
    )
    registry.conn.commit()

    dry_refresh_1 = _backfill_detect_performance(
        registry,
        dry_run=True,
        scope_paths=None,
        refresh=True,
    )
    dry_refresh_2 = _backfill_detect_performance(
        registry,
        dry_run=True,
        scope_paths=None,
        refresh=True,
    )
    assert dry_refresh_1 == dry_refresh_2
    registry.close()


def test_backfill_eye_mask_quality_dry_run_and_apply(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    zarr_path = tmp_path / "recordings" / "rec_a" / "zarr" / "rec_a_analysis.zarr"
    _create_eye_mask_performance_zarr(zarr_path)
    dataset_id = registry.register_from_root(zarr.open_group(str(zarr_path), mode="r"), zarr_path)
    registry.conn.execute(
        "UPDATE datasets SET zarr_use = 'analysis' WHERE dataset_id = ?;",
        (dataset_id,),
    )
    registry.conn.execute("DELETE FROM eye_mask_quality WHERE dataset_id = ?;", (dataset_id,))
    registry.conn.commit()

    dry = _backfill_eye_mask_quality(
        registry,
        dry_run=True,
        scope_paths=None,
        refresh=False,
    )
    assert dry["datasets_scanned"] == 1
    assert dry["rows_inserted"] == 1
    assert dry["rows_updated"] == 0
    assert dry["rows_deleted"] == 0

    applied = _backfill_eye_mask_quality(
        registry,
        dry_run=False,
        scope_paths=None,
        refresh=False,
    )
    assert applied["rows_inserted"] == 1
    rows = registry.conn.execute(
        """
        SELECT
            stage_group,
            run_name,
            eye_mask_method,
            successful_roi_pair_rate,
            review_state,
            review_intended_use,
            source_keypoint_stale_state,
            lifecycle_state
        FROM eye_mask_quality_current
        WHERE dataset_id = ?;
        """,
        (dataset_id,),
    ).fetchall()
    assert len(rows) == 1
    quality = rows[0]
    assert str(quality["stage_group"]) == "refined_eye_masks_runs"
    assert str(quality["run_name"]) == "refined_eye_masks_001"
    assert str(quality["eye_mask_method"]) == "refine_eye_masks"
    assert float(quality["successful_roi_pair_rate"]) == pytest.approx(1.0)
    assert str(quality["review_state"]) == "approved"
    assert str(quality["review_intended_use"]) == "training"
    assert str(quality["source_keypoint_stale_state"]) == "stale"
    assert str(quality["lifecycle_state"]) == "stale"
    registry.close()


def test_backfill_eye_mask_performance_dry_run_and_apply(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    zarr_path = tmp_path / "recordings" / "rec_a" / "zarr" / "rec_a_analysis.zarr"
    _create_eye_mask_performance_zarr(zarr_path)
    dataset_id = registry.register_from_root(zarr.open_group(str(zarr_path), mode="r"), zarr_path)
    registry.conn.execute(
        "UPDATE datasets SET zarr_use = 'analysis' WHERE dataset_id = ?;",
        (dataset_id,),
    )
    registry.conn.execute("DELETE FROM eye_mask_performance WHERE dataset_id = ?;", (dataset_id,))
    registry.conn.commit()

    dry = _backfill_eye_mask_performance(
        registry,
        dry_run=True,
        scope_paths=None,
        refresh=False,
    )
    assert dry["datasets_scanned"] == 1
    assert dry["rows_inserted"] == 2
    assert dry["rows_updated"] == 0
    assert dry["rows_deleted"] == 0

    applied = _backfill_eye_mask_performance(
        registry,
        dry_run=False,
        scope_paths=None,
        refresh=False,
    )
    assert applied["rows_inserted"] == 2
    rows = registry.conn.execute(
        """
        SELECT
            stage_group,
            run_name,
            method,
            total_rois,
            rois_per_second,
            review_state,
            review_intended_use,
            source_keypoint_stale_state,
            lifecycle_state
        FROM eye_mask_performance_latest
        WHERE dataset_id = ?
        ORDER BY stage_group;
        """,
        (dataset_id,),
    ).fetchall()
    assert len(rows) == 2
    by_stage = {str(row["stage_group"]): row for row in rows}
    assert str(by_stage["eye_masks_runs"]["run_name"]) == "eye_masks_001"
    assert str(by_stage["eye_masks_runs"]["method"]) == "traditional_eye_segmentation"
    assert int(by_stage["eye_masks_runs"]["total_rois"]) == 4
    assert float(by_stage["eye_masks_runs"]["rois_per_second"]) == pytest.approx(2.0)
    assert by_stage["eye_masks_runs"]["review_state"] is None
    assert by_stage["eye_masks_runs"]["source_keypoint_stale_state"] is None
    assert str(by_stage["refined_eye_masks_runs"]["run_name"]) == "refined_eye_masks_001"
    assert str(by_stage["refined_eye_masks_runs"]["method"]) == "refine_eye_masks"
    assert int(by_stage["refined_eye_masks_runs"]["total_rois"]) == 4
    assert float(by_stage["refined_eye_masks_runs"]["rois_per_second"]) == pytest.approx(4.0)
    assert str(by_stage["refined_eye_masks_runs"]["review_state"]) == "approved"
    assert str(by_stage["refined_eye_masks_runs"]["review_intended_use"]) == "training"
    assert str(by_stage["refined_eye_masks_runs"]["source_keypoint_stale_state"]) == "stale"
    assert str(by_stage["refined_eye_masks_runs"]["lifecycle_state"]) == "stale"
    registry.close()


def test_backfill_eye_mask_performance_summary_includes_stale_counters_when_zero(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    zarr_path = tmp_path / "recordings" / "rec_a" / "zarr" / "rec_a_analysis.zarr"
    _create_eye_mask_performance_zarr(zarr_path)
    root = zarr.open_group(str(zarr_path), mode="a")
    root["refined_eye_masks_runs"]["refined_eye_masks_001"].attrs.pop("source_keypoint_stale", None)

    dataset_id = registry.register_from_root(zarr.open_group(str(zarr_path), mode="r"), zarr_path)
    registry.conn.execute(
        "UPDATE datasets SET zarr_use = 'analysis' WHERE dataset_id = ?;",
        (dataset_id,),
    )
    registry.conn.execute("DELETE FROM eye_mask_performance WHERE dataset_id = ?;", (dataset_id,))
    registry.conn.commit()

    dry = _backfill_eye_mask_performance(
        registry,
        dry_run=True,
        scope_paths=None,
        refresh=True,
    )
    assert dry["datasets_scanned"] == 1
    assert dry["rows_inserted"] == 2
    assert dry["rows_stale"] == 0
    assert dry["rows_in_progress"] == 0
    registry.close()


def test_backfill_eye_mask_performance_scope_defaults_to_source_analysis(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    analysis_path = tmp_path / "recordings" / "rec_a" / "zarr" / "rec_a_analysis.zarr"
    training_path = tmp_path / "recordings" / "rec_a" / "zarr" / "rec_a_training.zarr"
    _create_eye_mask_performance_zarr(analysis_path)
    _create_eye_mask_performance_zarr(training_path)

    analysis_id = registry.register_from_root(zarr.open_group(str(analysis_path), mode="r"), analysis_path)
    training_id = registry.register_from_root(zarr.open_group(str(training_path), mode="r"), training_path)
    registry.conn.execute(
        "UPDATE datasets SET zarr_use = 'analysis' WHERE dataset_id = ?;",
        (analysis_id,),
    )
    registry.conn.execute(
        "UPDATE datasets SET zarr_use = 'training' WHERE dataset_id = ?;",
        (training_id,),
    )
    registry.conn.execute(
        "DELETE FROM eye_mask_performance WHERE dataset_id IN (?, ?);",
        (analysis_id, training_id),
    )
    registry.conn.commit()

    dry_default = _backfill_eye_mask_performance(
        registry,
        dry_run=True,
        scope_paths=None,
        refresh=False,
    )
    assert dry_default["datasets_scanned"] == 1
    assert dry_default["rows_inserted"] == 2

    applied_default = _backfill_eye_mask_performance(
        registry,
        dry_run=False,
        scope_paths=None,
        refresh=False,
    )
    assert applied_default["rows_inserted"] == 2
    analysis_rows = registry.conn.execute(
        "SELECT COUNT(*) AS n FROM eye_mask_performance WHERE dataset_id = ?;",
        (analysis_id,),
    ).fetchone()
    training_rows = registry.conn.execute(
        "SELECT COUNT(*) AS n FROM eye_mask_performance WHERE dataset_id = ?;",
        (training_id,),
    ).fetchone()
    assert analysis_rows is not None and int(analysis_rows["n"]) == 2
    assert training_rows is not None and int(training_rows["n"]) == 0

    registry.conn.execute("DELETE FROM eye_mask_performance;")
    registry.conn.commit()
    dry_all = _backfill_eye_mask_performance(
        registry,
        dry_run=True,
        scope_paths=None,
        refresh=False,
        include_all_datasets=True,
    )
    assert dry_all["datasets_scanned"] == 2
    assert dry_all["rows_inserted"] == 4
    registry.close()


def test_backfill_recording_step_status_dry_run_no_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    zarr_path = tmp_path / "recordings" / "rec_step_a" / "zarr" / "rec_step_a_analysis.zarr"
    fake_root = _create_recording_step_status_zarr(zarr_path)
    monkeypatch.setattr(
        "fisheye.registry.maintenance._import_zarr",
        lambda: _FakeZarrModule({str(zarr_path): fake_root}),
    )

    registry.upsert_dataset(
        dataset_id="dataset_step_a",
        session_uuid="session_step_a",
        zarr_path=zarr_path,
        recording_id="recording_step_a",
        artifact_kind="source_recording",
        zarr_use="analysis",
    )

    summary = _backfill_recording_step_status(
        registry,
        dry_run=True,
        scope_paths=None,
        recording_ids=None,
        zarr_use_filter="all",
    )
    assert summary["datasets_scanned"] == 1
    assert summary["rows_inserted"] == 18
    assert summary["rows_updated"] == 0
    assert summary["rows_skipped"] == 0

    rows_by_status = summary["rows_by_status"]
    assert isinstance(rows_by_status, dict)
    assert int(rows_by_status["ok"]) == 18
    assert int(rows_by_status["missing"]) == 0
    assert int(rows_by_status["absent"]) == 0
    assert int(rows_by_status["na"]) == 0
    assert int(rows_by_status["error"]) == 0

    current_count = registry.conn.execute(
        "SELECT COUNT(*) AS n FROM recording_step_status;"
    ).fetchone()
    history_count = registry.conn.execute(
        "SELECT COUNT(*) AS n FROM recording_step_status_history;"
    ).fetchone()
    assert current_count is not None and int(current_count["n"]) == 0
    assert history_count is not None and int(history_count["n"]) == 0
    registry.close()


def test_backfill_recording_step_status_apply_and_convergent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    zarr_path = tmp_path / "recordings" / "rec_step_a" / "zarr" / "rec_step_a_analysis.zarr"
    fake_root = _create_recording_step_status_zarr(zarr_path)
    monkeypatch.setattr(
        "fisheye.registry.maintenance._import_zarr",
        lambda: _FakeZarrModule({str(zarr_path): fake_root}),
    )

    registry.upsert_dataset(
        dataset_id="dataset_step_a",
        session_uuid="session_step_a",
        zarr_path=zarr_path,
        recording_id="recording_step_a",
        artifact_kind="source_recording",
        zarr_use="analysis",
    )

    applied = _backfill_recording_step_status(
        registry,
        dry_run=False,
        scope_paths=None,
        recording_ids=None,
        zarr_use_filter="all",
    )
    assert applied["rows_inserted"] == 18
    assert applied["rows_updated"] == 0
    assert applied["rows_skipped"] == 0
    assert applied["history_rows_inserted"] == 18

    rows = registry.conn.execute(
        """
        SELECT step_name, status, run_name, method, coverage_pct
        FROM recording_step_status
        WHERE dataset_id = ?
        ORDER BY step_name;
        """,
        ("dataset_step_a",),
    ).fetchall()
    assert len(rows) == 18
    by_step = {str(row["step_name"]): row for row in rows}
    assert set(by_step.keys()) == {
        "background",
        "calibration",
        "crop",
        "detect",
        "detection_tuning",
        "dish_mask",
        "eye_masks",
        "eye_mask_tuning",
        "id_assignment",
        "keypoints",
        "keypoint_tuning",
        "raw",
        "refined_detect",
        "refined_eye_masks",
        "refined_keypoints",
        "stimulus",
        "subdish_mask_tuning",
        "tracks",
    }
    assert all(str(row["status"]) == "ok" for row in rows)
    assert str(by_step["detect"]["run_name"]) == "detect_001"
    assert str(by_step["detect"]["method"]) == "yolo"
    assert float(by_step["detect"]["coverage_pct"]) == pytest.approx(75.0)
    detect_details_row = registry.conn.execute(
        """
        SELECT details_json
        FROM recording_step_status
        WHERE dataset_id = ? AND step_name = 'detect';
        """,
        ("dataset_step_a",),
    ).fetchone()
    assert detect_details_row is not None
    detect_details = json.loads(str(detect_details_row["details_json"]))
    assert detect_details["detect_quality_run"] == "detect_quality_001"
    assert detect_details["detect_quality_grade"] == "A"
    assert float(detect_details["detect_quality_score"]) == pytest.approx(98.4)
    assert float(detect_details["detect_quality_clean_percent"]) == pytest.approx(97.0)
    assert int(detect_details["detect_quality_artifacts"]) == 3
    assert str(by_step["id_assignment"]["run_name"]) == "id_assign_001"
    assert str(by_step["stimulus"]["run_name"]) == "stimulus_001"
    assert str(by_step["tracks"]["run_name"]) == "tracks_001"

    repeat = _backfill_recording_step_status(
        registry,
        dry_run=False,
        scope_paths=None,
        recording_ids=None,
        zarr_use_filter="all",
    )
    assert repeat["rows_inserted"] == 0
    assert repeat["rows_updated"] == 0
    assert repeat["rows_skipped"] == 18
    assert repeat["history_rows_inserted"] == 0

    history_count = registry.conn.execute(
        "SELECT COUNT(*) AS n FROM recording_step_status_history WHERE dataset_id = ?;",
        ("dataset_step_a",),
    ).fetchone()
    assert history_count is not None and int(history_count["n"]) == 18
    registry.close()


def test_backfill_recording_step_status_marks_refined_keypoints_stale_when_source_mismatches_latest_keypoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    zarr_path = tmp_path / "recordings" / "rec_step_b" / "zarr" / "rec_step_b_analysis.zarr"
    fake_root = _create_recording_step_status_zarr(zarr_path)
    keypoints_parent = fake_root["keypoints_runs"]
    keypoints_parent.add_group(
        "kp_002",
        attrs={"created_utc": "2026-02-15T01:10:00+00:00", "method": "yolo_pose"},
    )
    keypoints_parent.attrs["latest"] = "kp_002"
    monkeypatch.setattr(
        "fisheye.registry.maintenance._import_zarr",
        lambda: _FakeZarrModule({str(zarr_path): fake_root}),
    )

    registry.upsert_dataset(
        dataset_id="dataset_step_b",
        session_uuid="session_step_b",
        zarr_path=zarr_path,
        recording_id="recording_step_b",
        artifact_kind="source_recording",
        zarr_use="analysis",
    )

    _backfill_recording_step_status(
        registry,
        dry_run=False,
        scope_paths=None,
        recording_ids=None,
        zarr_use_filter="all",
    )

    refined_row = registry.conn.execute(
        """
        SELECT status, run_name, coverage_pct, details_json
        FROM recording_step_status
        WHERE dataset_id = ? AND step_name = 'refined_keypoints';
        """,
        ("dataset_step_b",),
    ).fetchone()
    assert refined_row is not None
    assert str(refined_row["status"]) == "missing"
    assert refined_row["run_name"] is None
    assert refined_row["coverage_pct"] is None
    refined_details = json.loads(str(refined_row["details_json"]))
    assert refined_details["reason"] == "stale_vs_latest_keypoints"
    assert refined_details["expected_source_keypoints_run"] == "kp_002"
    assert refined_details["latest_refined_run"] == "refined_kp_001"
    assert refined_details["latest_refined_source_keypoints_run"] == "kp_001"

    keypoints_row = registry.conn.execute(
        """
        SELECT status, run_name, method
        FROM recording_step_status
        WHERE dataset_id = ? AND step_name = 'keypoints';
        """,
        ("dataset_step_b",),
    ).fetchone()
    assert keypoints_row is not None
    assert str(keypoints_row["status"]) == "ok"
    assert str(keypoints_row["run_name"]) == "kp_002"
    assert str(keypoints_row["method"]) == "yolo_pose"
    registry.close()


def test_backfill_recording_step_status_marks_refined_detect_stale_when_source_mismatches_latest_detect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    zarr_path = tmp_path / "recordings" / "rec_step_c" / "zarr" / "rec_step_c_analysis.zarr"
    fake_root = _create_recording_step_status_zarr(zarr_path)
    detect_parent = fake_root["detect_runs"]
    detect_parent.add_group(
        "detect_002",
        attrs={"detect_timestamp_utc": "2026-02-15T02:00:00+00:00", "detection_method": "yolo"},
    )
    detect_parent.attrs["latest"] = "detect_002"
    monkeypatch.setattr(
        "fisheye.registry.maintenance._import_zarr",
        lambda: _FakeZarrModule({str(zarr_path): fake_root}),
    )

    registry.upsert_dataset(
        dataset_id="dataset_step_c",
        session_uuid="session_step_c",
        zarr_path=zarr_path,
        recording_id="recording_step_c",
        artifact_kind="source_recording",
        zarr_use="analysis",
    )

    _backfill_recording_step_status(
        registry,
        dry_run=False,
        scope_paths=None,
        recording_ids=None,
        zarr_use_filter="all",
    )

    refined_row = registry.conn.execute(
        """
        SELECT status, run_name, coverage_pct, details_json
        FROM recording_step_status
        WHERE dataset_id = ? AND step_name = 'refined_detect';
        """,
        ("dataset_step_c",),
    ).fetchone()
    assert refined_row is not None
    assert str(refined_row["status"]) == "missing"
    assert refined_row["run_name"] is None
    assert refined_row["coverage_pct"] is None
    refined_details = json.loads(str(refined_row["details_json"]))
    assert refined_details["reason"] == "stale_vs_latest_detect"
    assert refined_details["expected_source_detect_run"] == "detect_002"
    assert refined_details["latest_refined_detect_run"] == "refined_detect_001"
    assert refined_details["latest_refined_detect_source_run"] == "detect_001"

    detect_row = registry.conn.execute(
        """
        SELECT status, run_name
        FROM recording_step_status
        WHERE dataset_id = ? AND step_name = 'detect';
        """,
        ("dataset_step_c",),
    ).fetchone()
    assert detect_row is not None
    assert str(detect_row["status"]) == "ok"
    assert str(detect_row["run_name"]) == "detect_002"
    registry.close()


def test_backfill_recording_step_status_marks_eye_masks_stale_when_source_mismatches_latest_keypoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    zarr_path = tmp_path / "recordings" / "rec_step_d" / "zarr" / "rec_step_d_analysis.zarr"
    fake_root = _create_recording_step_status_zarr(zarr_path)
    keypoints_parent = fake_root["keypoints_runs"]
    keypoints_parent.add_group(
        "kp_002",
        attrs={"created_utc": "2026-02-15T03:00:00+00:00", "method": "yolo_pose"},
    )
    keypoints_parent.attrs["latest"] = "kp_002"

    refined_parent = fake_root["refined_keypoints_runs"]
    refined_parent.add_group(
        "refined_kp_002",
        attrs={
            "created_utc": "2026-02-15T03:05:00+00:00",
            "method": "refine_keypoints",
            "source_keypoints_run": "kp_002",
            "summary_statistics": {"postprocess": {"total_rois": 4, "usable_keypoints": 4}},
        },
    )
    refined_parent.attrs["latest"] = "refined_kp_002"
    monkeypatch.setattr(
        "fisheye.registry.maintenance._import_zarr",
        lambda: _FakeZarrModule({str(zarr_path): fake_root}),
    )

    registry.upsert_dataset(
        dataset_id="dataset_step_d",
        session_uuid="session_step_d",
        zarr_path=zarr_path,
        recording_id="recording_step_d",
        artifact_kind="source_recording",
        zarr_use="analysis",
    )

    _backfill_recording_step_status(
        registry,
        dry_run=False,
        scope_paths=None,
        recording_ids=None,
        zarr_use_filter="all",
    )

    eye_masks_row = registry.conn.execute(
        """
        SELECT status, run_name, coverage_pct, details_json
        FROM recording_step_status
        WHERE dataset_id = ? AND step_name = 'eye_masks';
        """,
        ("dataset_step_d",),
    ).fetchone()
    assert eye_masks_row is not None
    assert str(eye_masks_row["status"]) == "missing"
    assert eye_masks_row["run_name"] is None
    assert eye_masks_row["coverage_pct"] is None
    eye_masks_details = json.loads(str(eye_masks_row["details_json"]))
    assert eye_masks_details["reason"] == "stale_vs_latest_keypoints"
    assert eye_masks_details["expected_source_keypoints_run"] == "refined_kp_002"
    assert eye_masks_details["latest_eye_masks_run"] == "eye_masks_001"
    assert eye_masks_details["latest_eye_masks_source_keypoints_run"] == "refined_kp_001"

    refined_keypoints_row = registry.conn.execute(
        """
        SELECT status, run_name
        FROM recording_step_status
        WHERE dataset_id = ? AND step_name = 'refined_keypoints';
        """,
        ("dataset_step_d",),
    ).fetchone()
    assert refined_keypoints_row is not None
    assert str(refined_keypoints_row["status"]) == "ok"
    assert str(refined_keypoints_row["run_name"]) == "refined_kp_002"
    registry.close()


def test_backfill_recording_step_status_marks_refined_eye_masks_stale_when_source_mismatches_latest_eye_masks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    zarr_path = tmp_path / "recordings" / "rec_step_e" / "zarr" / "rec_step_e_analysis.zarr"
    fake_root = _create_recording_step_status_zarr(zarr_path)
    eye_masks_parent = fake_root["eye_masks_runs"]
    eye_masks_parent.add_group(
        "eye_masks_002",
        attrs={
            "created_utc": "2026-02-15T04:00:00+00:00",
            "method": "traditional_eye_segmentation",
            "source_keypoints_run": "refined_kp_001",
            "source_keypoint_group": "refined_keypoints_runs",
            "successful_roi_pair_rate": 0.8,
        },
    )
    eye_masks_parent.attrs["latest"] = "eye_masks_002"
    monkeypatch.setattr(
        "fisheye.registry.maintenance._import_zarr",
        lambda: _FakeZarrModule({str(zarr_path): fake_root}),
    )

    registry.upsert_dataset(
        dataset_id="dataset_step_e",
        session_uuid="session_step_e",
        zarr_path=zarr_path,
        recording_id="recording_step_e",
        artifact_kind="source_recording",
        zarr_use="analysis",
    )

    _backfill_recording_step_status(
        registry,
        dry_run=False,
        scope_paths=None,
        recording_ids=None,
        zarr_use_filter="all",
    )

    refined_eye_row = registry.conn.execute(
        """
        SELECT status, run_name, coverage_pct, details_json
        FROM recording_step_status
        WHERE dataset_id = ? AND step_name = 'refined_eye_masks';
        """,
        ("dataset_step_e",),
    ).fetchone()
    assert refined_eye_row is not None
    assert str(refined_eye_row["status"]) == "missing"
    assert refined_eye_row["run_name"] is None
    assert refined_eye_row["coverage_pct"] is None
    refined_eye_details = json.loads(str(refined_eye_row["details_json"]))
    assert refined_eye_details["reason"] == "stale_vs_latest_eye_masks"
    assert refined_eye_details["expected_source_eye_masks_run"] == "eye_masks_002"
    assert refined_eye_details["latest_refined_eye_masks_run"] == "refined_eye_masks_001"
    assert refined_eye_details["latest_refined_eye_masks_source_run"] == "eye_masks_001"

    eye_masks_row = registry.conn.execute(
        """
        SELECT status, run_name
        FROM recording_step_status
        WHERE dataset_id = ? AND step_name = 'eye_masks';
        """,
        ("dataset_step_e",),
    ).fetchone()
    assert eye_masks_row is not None
    assert str(eye_masks_row["status"]) == "ok"
    assert str(eye_masks_row["run_name"]) == "eye_masks_002"
    registry.close()


def test_backfill_recording_step_status_scoped_filters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    zarr_a = tmp_path / "recordings" / "scope_a" / "zarr" / "scope_a_analysis.zarr"
    zarr_b = tmp_path / "recordings" / "scope_b" / "zarr" / "scope_b_training.zarr"
    fake_root_a = _create_recording_step_status_zarr(zarr_a)
    fake_root_b = _create_recording_step_status_zarr(zarr_b)
    monkeypatch.setattr(
        "fisheye.registry.maintenance._import_zarr",
        lambda: _FakeZarrModule({str(zarr_a): fake_root_a, str(zarr_b): fake_root_b}),
    )

    registry.upsert_dataset(
        dataset_id="dataset_scope_a",
        session_uuid="session_scope_a",
        zarr_path=zarr_a,
        recording_id="recording_scope_a",
        artifact_kind="source_recording",
        zarr_use="analysis",
    )
    registry.upsert_dataset(
        dataset_id="dataset_scope_b",
        session_uuid="session_scope_b",
        zarr_path=zarr_b,
        recording_id="recording_scope_b",
        artifact_kind="source_recording",
        zarr_use="training",
    )

    by_recording = _backfill_recording_step_status(
        registry,
        dry_run=True,
        scope_paths=None,
        recording_ids=("recording_scope_a",),
        zarr_use_filter="all",
    )
    assert by_recording["rows_inserted"] == 18
    assert by_recording["datasets_skipped_recording_filter"] == 1

    by_use = _backfill_recording_step_status(
        registry,
        dry_run=True,
        scope_paths=None,
        recording_ids=None,
        zarr_use_filter="analysis",
    )
    assert by_use["rows_inserted"] == 18
    assert by_use["datasets_skipped_zarr_use_filter"] == 1

    by_scope = _backfill_recording_step_status(
        registry,
        dry_run=True,
        scope_paths=[tmp_path / "recordings" / "scope_a"],
        recording_ids=None,
        zarr_use_filter="all",
    )
    assert by_scope["rows_inserted"] == 18
    assert by_scope["datasets_skipped_path"] == 1

    current_count = registry.conn.execute(
        "SELECT COUNT(*) AS n FROM recording_step_status;"
    ).fetchone()
    assert current_count is not None and int(current_count["n"]) == 0
    registry.close()


def test_check_registry_integrity_passes_for_valid_rows(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    cfg = tmp_path / "cfg.yaml"
    model = tmp_path / "best.pt"
    metrics = tmp_path / "results.csv"
    onnx = tmp_path / "best.onnx"
    onnx_manifest = tmp_path / "best.onnx.manifest.json"
    trt = tmp_path / "best_fp16.engine"
    trt_manifest = tmp_path / "best_fp16.tensorrt.manifest.json"
    cfg.write_text("cfg", encoding="utf-8")
    model.write_text("model", encoding="utf-8")
    metrics.write_text("metrics", encoding="utf-8")
    onnx.write_text("onnx", encoding="utf-8")
    onnx_manifest.write_text("{}", encoding="utf-8")
    trt.write_text("trt", encoding="utf-8")
    trt_manifest.write_text("{}", encoding="utf-8")

    registry.record_training_run(
        run_id="run_ok",
        set_id="set_ok",
        config_path=cfg,
        manifest_path=None,
        skeleton_id=None,
        model_path=model,
        metrics_path=metrics,
        status="success",
        final_metrics={"mAP50": 0.9},
    )
    registry.record_model_export(
        run_id="run_ok",
        export_type="onnx",
        path=onnx,
        manifest_path=onnx_manifest,
        metadata={"sha256": "onnx_sha", "manifest_sha256": "onnx_manifest_sha"},
    )
    registry.record_model_export(
        run_id="run_ok",
        export_type="tensorrt",
        path=trt,
        manifest_path=trt_manifest,
        metadata={"sha256": "trt_sha", "manifest_sha256": "trt_manifest_sha", "precision": "fp16"},
    )

    issues = _check_registry_integrity(registry)
    assert issues == []
    registry.close()


def test_check_registry_integrity_reports_missing_detection_model_rows(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("cfg", encoding="utf-8")
    registry.record_training_run(
        run_id="run_missing_dm",
        set_id="set_a",
        config_path=cfg,
        manifest_path=None,
        skeleton_id=None,
        model_path=None,
        metrics_path=None,
        status="in_progress",
        final_metrics={"stage": "start"},
    )
    # Simulate inconsistent state.
    registry.conn.execute("DELETE FROM training_models WHERE run_id = 'run_missing_dm';")
    registry.conn.commit()

    issues = _check_registry_integrity(registry)
    assert any(issue.code == "missing_detection_model_row" and issue.run_id == "run_missing_dm" for issue in issues)
    registry.close()


def test_check_registry_integrity_reports_missing_artifact_files(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("cfg", encoding="utf-8")
    model = tmp_path / "missing_best.pt"
    metrics = tmp_path / "missing_results.csv"
    onnx = tmp_path / "missing_best.onnx"
    trt = tmp_path / "missing_best_fp16.engine"

    registry.record_training_run(
        run_id="run_missing_files",
        set_id="set_missing",
        config_path=cfg,
        manifest_path=None,
        skeleton_id=None,
        model_path=model,
        metrics_path=metrics,
        status="success",
        final_metrics={"mAP50": 0.9},
    )
    registry.record_model_export(
        run_id="run_missing_files",
        export_type="onnx",
        path=onnx,
        metadata={"sha256": "onnx_sha"},
    )
    registry.record_model_export(
        run_id="run_missing_files",
        export_type="tensorrt",
        path=trt,
        metadata={"sha256": "trt_sha", "precision": "fp16"},
    )

    issues = _check_registry_integrity(registry)
    codes = {issue.code for issue in issues}
    assert "missing_model_file" in codes
    assert "missing_metrics_file" in codes
    assert "onnx_file_missing" in codes
    assert "trt_file_missing" in codes
    registry.close()


def test_check_registry_integrity_reports_trt_plugin_contract_mismatch(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    cfg = tmp_path / "cfg.yaml"
    model = tmp_path / "best.pt"
    metrics = tmp_path / "results.csv"
    onnx = tmp_path / "best.onnx"
    onnx_manifest = tmp_path / "best.onnx.manifest.json"
    trt = tmp_path / "best_fp16.engine"
    trt_manifest = tmp_path / "best_fp16.tensorrt.manifest.json"
    cfg.write_text("cfg", encoding="utf-8")
    model.write_text("model", encoding="utf-8")
    metrics.write_text("metrics", encoding="utf-8")
    onnx.write_text("onnx", encoding="utf-8")
    onnx_manifest.write_text("{}", encoding="utf-8")
    trt.write_text("trt", encoding="utf-8")
    trt_manifest.write_text("{}", encoding="utf-8")

    registry.record_training_run(
        run_id="run_plugin_mismatch",
        set_id="set_ok",
        config_path=cfg,
        manifest_path=None,
        skeleton_id=None,
        model_path=model,
        metrics_path=metrics,
        status="success",
        final_metrics={"mAP50": 0.9},
    )
    registry.record_model_export(
        run_id="run_plugin_mismatch",
        export_type="onnx",
        path=onnx,
        manifest_path=onnx_manifest,
        metadata={
            "sha256": "onnx_sha",
            "manifest_sha256": "onnx_manifest_sha",
            "requires_plugins": True,
            "plugin_ops": ["TRT::EfficientNMS_TRT"],
            "plugin_versions": {"TRT::EfficientNMS_TRT": "1"},
        },
    )
    registry.record_model_export(
        run_id="run_plugin_mismatch",
        export_type="tensorrt",
        path=trt,
        manifest_path=trt_manifest,
        metadata={
            "sha256": "trt_sha",
            "manifest_sha256": "trt_manifest_sha",
            "precision": "fp16",
            # Deliberately incomplete to trigger integrity findings.
            "requires_plugins": True,
        },
    )

    issues = _check_registry_integrity(registry)
    codes = {issue.code for issue in issues}
    assert "trt_plugins_missing_ops" in codes
    assert "trt_plugin_contract_mismatch" in codes
    registry.close()


def test_registry_schema_version_initialized(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    latest_version = max(version for version, _name, _fn in registry._schema_migrations())
    row = registry.conn.execute("SELECT MAX(version) AS version FROM schema_version;").fetchone()
    assert row is not None
    assert int(row["version"]) == latest_version
    pragma_row = registry.conn.execute("PRAGMA user_version;").fetchone()
    assert pragma_row is not None
    assert int(pragma_row[0]) == latest_version
    registry.close()


def test_registry_schema_version_bootstrap_for_existing_registry(tmp_path: Path) -> None:
    path = tmp_path / "registry.sqlite"
    registry = Registry(path)
    latest_version = max(version for version, _name, _fn in registry._schema_migrations())
    registry.close()

    with sqlite3.connect(str(path)) as conn:
        conn.execute("DROP TABLE schema_version;")
        conn.commit()

    reopened = Registry(path)
    row = reopened.conn.execute("SELECT MAX(version) AS version FROM schema_version;").fetchone()
    assert row is not None
    assert int(row["version"]) == latest_version
    reopened.close()


class _FailingMigrationRegistry(Registry):
    def _schema_migrations(self):
        migrations = list(super()._schema_migrations())
        next_version = max(version for version, _name, _fn in migrations) + 1
        migrations.append((next_version, "intentional_failure_for_test", self._migration_002_fail))
        return migrations

    def _migration_002_fail(self) -> None:
        self.conn.execute("CREATE TABLE should_rollback (id INTEGER PRIMARY KEY);")
        raise RuntimeError("boom")


def test_registry_migration_failure_does_not_advance_version(tmp_path: Path) -> None:
    path = tmp_path / "registry.sqlite"
    base = Registry(path)
    latest_version = max(version for version, _name, _fn in base._schema_migrations())
    base.close()

    with pytest.raises(RuntimeError, match="boom"):
        _FailingMigrationRegistry(path)

    with sqlite3.connect(str(path)) as conn:
        version = conn.execute("SELECT MAX(version) FROM schema_version;").fetchone()[0]
        assert int(version) == latest_version
        table_row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='should_rollback';"
        ).fetchone()
        assert table_row is None
