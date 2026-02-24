"""Tests for registry_query subject-lineage filters."""

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.registry.db import Registry
from fisheye.utils.registry_query import main as registry_query_main


def _seed_registry_for_subject_filters(registry_path: Path) -> None:
    registry = Registry(registry_path)
    # Minimal dataset rows.
    registry.upsert_dataset(
        "dataset_a",
        session_uuid="session_a",
        zarr_path=registry_path.parent / "a.zarr",
        recording_id="recording_a",
        artifact_kind="source_recording",
    )
    registry.upsert_dataset(
        "dataset_b",
        session_uuid="session_b",
        zarr_path=registry_path.parent / "b.zarr",
        recording_id="recording_b",
        artifact_kind="source_recording",
    )
    registry.upsert_provenance(
        "dataset_a",
        provenance={},
        context={},
        protocol_name=None,
        protocol_hash=None,
        acquisition={},
        zarr_purpose=None,
    )
    registry.upsert_provenance(
        "dataset_b",
        provenance={},
        context={},
        protocol_name=None,
        protocol_hash=None,
        acquisition={},
        zarr_purpose=None,
    )
    # Recording context rows for view joins.
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
            "session_a",
            "recording_a",
            str(registry_path.parent / "recording_a"),
            "behavior",
            "free",
            "free",
            "behavior_v1",
        ),
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
            "recording_b",
            "session_b",
            "recording_b",
            str(registry_path.parent / "recording_b"),
            "behavior",
            "free",
            "free",
            "behavior_v1",
        ),
    )
    # Lineage entities.
    registry.conn.execute(
        """
        INSERT INTO crosses (cross_id, genotype, created_utc, updated_utc)
        VALUES ('cross_a', 'genotype_x', datetime('now'), datetime('now'));
        """
    )
    registry.conn.execute(
        """
        INSERT INTO crosses (cross_id, genotype, created_utc, updated_utc)
        VALUES ('cross_b', 'genotype_y', datetime('now'), datetime('now'));
        """
    )
    registry.conn.execute(
        """
        INSERT INTO dishes (dish_id, cross_id, created_utc, updated_utc)
        VALUES ('dish_a', 'cross_a', datetime('now'), datetime('now'));
        """
    )
    registry.conn.execute(
        """
        INSERT INTO dishes (dish_id, cross_id, created_utc, updated_utc)
        VALUES ('dish_b', 'cross_b', datetime('now'), datetime('now'));
        """
    )
    registry.conn.execute(
        """
        INSERT INTO subjects (subject_id, dish_id, created_utc, updated_utc)
        VALUES ('subject_a', 'dish_a', datetime('now'), datetime('now'));
        """
    )
    registry.conn.execute(
        """
        INSERT INTO subjects (subject_id, dish_id, created_utc, updated_utc)
        VALUES ('subject_b', 'dish_b', datetime('now'), datetime('now'));
        """
    )
    registry.conn.execute(
        """
        INSERT INTO recording_subjects (
            recording_id, subject_id, dataset_id, dish_id, cross_id, dpf_at_acquisition, created_utc, updated_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'));
        """,
        ("recording_a", "subject_a", "dataset_a", "dish_a", "cross_a", 8),
    )
    registry.conn.execute(
        """
        INSERT INTO recording_subjects (
            recording_id, subject_id, dataset_id, dish_id, cross_id, dpf_at_acquisition, created_utc, updated_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'));
        """,
        ("recording_b", "subject_b", "dataset_b", "dish_b", "cross_b", 12),
    )
    registry.conn.commit()
    registry.close()


def _seed_registry_for_detect_filters(registry_path: Path) -> None:
    registry = Registry(registry_path)
    for (
        dataset_id,
        session_uuid,
        recording_id,
        filename,
        rig_id,
        arena_id,
        camera_id,
        dish_design,
    ) in (
        ("dataset_a", "session_a", "recording_a", "a.zarr", "rig_a", "arena_x", "cam_1", "cedar"),
        ("dataset_b", "session_b", "recording_b", "b.zarr", "rig_a", "arena_x", "cam_2", "cedar"),
        ("dataset_c", "session_c", "recording_c", "c.zarr", "rig_b", "arena_y", "cam_3", "maple"),
    ):
        registry.upsert_dataset(
            dataset_id,
            session_uuid=session_uuid,
            zarr_path=registry_path.parent / filename,
            recording_id=recording_id,
            artifact_kind="source_recording",
            zarr_use="analysis",
        )
        registry.upsert_provenance(
            dataset_id,
            provenance={},
            context={
                "rig_id": rig_id,
                "arena_id": arena_id,
                "camera_id": camera_id,
            },
            protocol_name=None,
            protocol_hash=None,
            acquisition={
                "dish_design": dish_design,
            },
            zarr_purpose="analysis",
        )

    registry.upsert_detect_performance(
        dataset_id="dataset_a",
        detect_run="detect_a",
        detect_created_utc="2026-02-09T00:00:00+00:00",
        recording_id="recording_a",
        zarr_use="analysis",
        detection_method="yolo",
        model_run_id="run_detect_model_v1",
        model_set_id="detect_set_v1",
        model_path="/models/detect_model_v1.pt",
        model_name="detect_model_v1.pt",
        coverage_percent=95.0,
        frames_with_detections=95,
        frames_zero_detections=5,
        total_frames=100,
        mean_confidence=0.9,
        min_confidence=0.5,
        max_confidence=1.0,
        inference_duration_seconds=10.0,
        inference_average_fps=120.0,
        inference_avg_batch_ms=50.0,
        inference_avg_read_ms=80.0,
        conf_threshold=0.4,
        iou_threshold=0.8,
        batch_size=16,
        inference_width=640,
        inference_height=640,
    )
    registry.upsert_detect_performance(
        dataset_id="dataset_b",
        detect_run="detect_b",
        detect_created_utc="2026-02-09T00:00:00+00:00",
        recording_id="recording_b",
        zarr_use="analysis",
        detection_method="traditional",
        model_run_id=None,
        model_set_id=None,
        model_path=None,
        model_name=None,
        coverage_percent=70.0,
        frames_with_detections=70,
        frames_zero_detections=30,
        total_frames=100,
        mean_confidence=0.8,
        min_confidence=0.4,
        max_confidence=1.0,
        inference_duration_seconds=20.0,
        inference_average_fps=60.0,
        inference_avg_batch_ms=80.0,
        inference_avg_read_ms=130.0,
        conf_threshold=None,
        iou_threshold=None,
        batch_size=16,
        inference_width=640,
        inference_height=640,
    )
    registry.upsert_detect_performance(
        dataset_id="dataset_c",
        detect_run="detect_c",
        detect_created_utc="2026-02-09T00:00:00+00:00",
        recording_id="recording_c",
        zarr_use="analysis",
        detection_method="yolo",
        model_run_id="run_detect_model_v2",
        model_set_id="detect_set_v2",
        model_path="/models/detect_model_v2.pt",
        model_name="detect_model_v2.pt",
        coverage_percent=85.0,
        frames_with_detections=85,
        frames_zero_detections=15,
        total_frames=100,
        mean_confidence=0.88,
        min_confidence=0.45,
        max_confidence=1.0,
        inference_duration_seconds=12.0,
        inference_average_fps=100.0,
        inference_avg_batch_ms=55.0,
        inference_avg_read_ms=90.0,
        conf_threshold=0.4,
        iou_threshold=0.8,
        batch_size=16,
        inference_width=640,
        inference_height=640,
    )

    def _crop_record(
        *,
        crop_run: str,
        recording_id: str,
        zarr_use: str,
        source_type: str,
        percent_frames: float,
        review_state: str | None,
        review_intended_use: str | None,
        review_method: str | None = "manual",
    ) -> dict[str, object]:
        return {
            "crop_run": crop_run,
            "recording_id": recording_id,
            "zarr_use": zarr_use,
            "crop_created_utc": "2026-02-09T00:00:00+00:00",
            "source_detect_run": None,
            "source_refined_run": "refined_detect_2026-02-09_00-00-00",
            "detection_source_type": source_type,
            "detection_source_path": "refined_detect_runs/refined_detect_2026-02-09_00-00-00/manual",
            "total_rois": 1000,
            "frames_with_crops": int(percent_frames),
            "total_frames": 100,
            "percent_frames_with_crops": percent_frames,
            "includes_interpolated": 0,
            "n_real_detections": 1000,
            "n_interpolated_detections": 0,
            "review_state": review_state,
            "review_method": review_method,
            "review_intended_use": review_intended_use,
            "review_reviewer": "tester",
            "review_timestamp_utc": "2026-02-09T00:05:00+00:00",
            "review_notes": None,
            "zarr_mtime_ns": 123456789,
            "updated_utc": "2026-02-09T00:05:00+00:00",
        }

    registry.replace_crop_quality(
        "dataset_a",
        [
            _crop_record(
                crop_run="crop_a",
                recording_id="recording_a",
                zarr_use="analysis",
                source_type="full_recording",
                percent_frames=95.0,
                review_state="approved",
                review_intended_use="training",
            )
        ],
    )
    registry.replace_crop_quality(
        "dataset_b",
        [
            _crop_record(
                crop_run="crop_b",
                recording_id="recording_b",
                zarr_use="analysis",
                source_type="raw",
                percent_frames=70.0,
                review_state=None,
                review_intended_use=None,
            )
        ],
    )
    registry.replace_crop_quality(
        "dataset_c",
        [
            _crop_record(
                crop_run="crop_c",
                recording_id="recording_c",
                zarr_use="analysis",
                source_type="interpolated",
                percent_frames=85.0,
                review_state="needs_review",
                review_intended_use="full_recording",
            )
        ],
    )
    registry.close()


def _seed_eye_mask_performance_rows(registry_path: Path) -> None:
    registry = Registry(registry_path)

    registry.upsert_eye_mask_performance(
        dataset_id="dataset_a",
        stage_group="eye_masks_runs",
        run_name="eye_masks_a",
        run_created_utc="2026-02-11T00:00:00+00:00",
        recording_id="recording_a",
        zarr_use="analysis",
        method="traditional_eye_segmentation",
        source_crop_run="crop_a",
        source_keypoint_group="refined_keypoints_runs",
        source_keypoints_run="refined_kp_a",
        source_eye_masks_run=None,
        source_eye_masks_method=None,
        total_rois=200,
        successful_eyes=360,
        successful_roi_pairs=160,
        successful_roi_pair_rate=0.8,
        duration_seconds=100.0,
        rois_per_second=2.0,
        inference_duration_seconds=None,
        inference_average_fps=2.0,
        reason_counts_json=None,
        summary_statistics_json=None,
        lifecycle_state=None,
    )
    registry.upsert_eye_mask_performance(
        dataset_id="dataset_a",
        stage_group="refined_eye_masks_runs",
        run_name="refined_eye_masks_a",
        run_created_utc="2026-02-11T00:10:00+00:00",
        recording_id="recording_a",
        zarr_use="analysis",
        method="refine_eye_masks",
        source_crop_run="crop_a",
        source_keypoint_group="refined_keypoints_runs",
        source_keypoints_run="refined_kp_a",
        source_eye_masks_run="eye_masks_a",
        source_eye_masks_method="traditional_eye_segmentation",
        total_rois=200,
        successful_eyes=392,
        successful_roi_pairs=196,
        successful_roi_pair_rate=0.98,
        duration_seconds=40.0,
        rois_per_second=5.0,
        inference_duration_seconds=None,
        inference_average_fps=5.0,
        reason_counts_json=None,
        summary_statistics_json=None,
        review_state="approved",
        review_method="manual",
        review_intended_use="training",
        review_reviewer="alice",
        review_timestamp_utc="2026-02-11T00:20:00+00:00",
        lifecycle_state="approved",
    )

    registry.upsert_eye_mask_performance(
        dataset_id="dataset_b",
        stage_group="refined_eye_masks_runs",
        run_name="refined_eye_masks_b",
        run_created_utc="2026-02-11T01:10:00+00:00",
        recording_id="recording_b",
        zarr_use="analysis",
        method="refine_eye_masks",
        source_crop_run="crop_b",
        source_keypoint_group="refined_keypoints_runs",
        source_keypoints_run="refined_kp_b",
        source_eye_masks_run="eye_masks_b",
        source_eye_masks_method="traditional_eye_segmentation",
        total_rois=180,
        successful_eyes=320,
        successful_roi_pairs=150,
        successful_roi_pair_rate=0.8333,
        duration_seconds=55.0,
        rois_per_second=3.2727,
        inference_duration_seconds=None,
        inference_average_fps=3.2727,
        reason_counts_json=None,
        summary_statistics_json=None,
        review_state="needs_review",
        review_method="manual",
        review_intended_use="training",
        review_reviewer="bob",
        review_timestamp_utc="2026-02-11T01:20:00+00:00",
        source_keypoint_stale_state="stale",
        source_keypoint_stale_reason="keypoint_manual_correction",
        source_keypoint_stale_timestamp_utc="2026-02-11T01:25:00+00:00",
        source_keypoint_stale_json=json.dumps({"state": "stale", "reason": "keypoint_manual_correction"}),
        lifecycle_state="stale",
        lifecycle_reason="keypoint_manual_correction",
    )

    registry.upsert_eye_mask_performance(
        dataset_id="dataset_c",
        stage_group="eye_masks_runs",
        run_name="eye_masks_c",
        run_created_utc="2026-02-11T02:00:00+00:00",
        recording_id="recording_c",
        zarr_use="analysis",
        method="traditional_eye_segmentation",
        source_crop_run="crop_c",
        source_keypoint_group="refined_keypoints_runs",
        source_keypoints_run="refined_kp_c",
        source_eye_masks_run=None,
        source_eye_masks_method=None,
        total_rois=220,
        successful_eyes=430,
        successful_roi_pairs=210,
        successful_roi_pair_rate=0.9545,
        duration_seconds=70.0,
        rois_per_second=3.1428,
        inference_duration_seconds=None,
        inference_average_fps=3.1428,
        reason_counts_json=None,
        summary_statistics_json=None,
        lifecycle_state=None,
    )
    registry.upsert_eye_mask_performance(
        dataset_id="dataset_c",
        stage_group="refined_eye_masks_runs",
        run_name="refined_eye_masks_c",
        run_created_utc="2026-02-11T02:05:00+00:00",
        recording_id="recording_c",
        zarr_use="analysis",
        method="refine_eye_masks",
        source_crop_run="crop_c",
        source_keypoint_group="refined_keypoints_runs",
        source_keypoints_run="refined_kp_c",
        source_eye_masks_run="eye_masks_c",
        source_eye_masks_method="traditional_eye_segmentation",
        total_rois=220,
        successful_eyes=410,
        successful_roi_pairs=200,
        successful_roi_pair_rate=0.9090,
        duration_seconds=90.0,
        rois_per_second=2.4444,
        inference_duration_seconds=None,
        inference_average_fps=2.4444,
        reason_counts_json=None,
        summary_statistics_json=None,
        review_state="pending",
        review_method="manual",
        review_intended_use="training",
        review_reviewer="carol",
        review_timestamp_utc="2026-02-11T02:10:00+00:00",
        lifecycle_state="in_progress",
        lifecycle_reason="pending",
    )

    registry.close()


def _seed_keypoint_quality_and_performance_rows(registry_path: Path) -> None:
    registry = Registry(registry_path)

    def _quality_record(
        *,
        refined_run: str,
        refined_created_utc: str,
        source_keypoint_run: str,
        keypoint_method: str,
        review_state: str | None,
        review_intended_use: str | None,
        review_reviewer: str | None,
        review_timestamp_utc: str | None,
        usable_rate: float | None,
        usable_count: int,
        total_count: int,
        raw_success_rate: float,
        raw_successful: int,
    ) -> dict[str, object]:
        return {
            "refined_run": refined_run,
            "refined_created_utc": refined_created_utc,
            "source_keypoint_run": source_keypoint_run,
            "keypoint_method": keypoint_method,
            "review_state": review_state,
            "review_intended_use": review_intended_use,
            "review_reviewer": review_reviewer,
            "review_timestamp_utc": review_timestamp_utc,
            "usable_keypoints": usable_count,
            "total_keypoints": total_count,
            "usable_keypoints_rate": usable_rate,
            "raw_keypoints_success_rate": raw_success_rate,
            "raw_keypoints_successful": raw_successful,
            "quality_updated_utc": "2026-02-11T03:00:00+00:00",
            "zarr_mtime_ns": 123456789,
        }

    registry.replace_keypoint_quality(
        "dataset_a",
        [
            _quality_record(
                refined_run="refined_kp_a_trad",
                refined_created_utc="2026-02-11T00:30:00+00:00",
                source_keypoint_run="keypoints_a_trad",
                keypoint_method="traditional_pose",
                review_state="approved",
                review_intended_use="training",
                review_reviewer="alice",
                review_timestamp_utc="2026-02-11T00:45:00+00:00",
                usable_rate=0.95,
                usable_count=95,
                total_count=100,
                raw_success_rate=0.97,
                raw_successful=97,
            ),
            _quality_record(
                refined_run="refined_kp_a_yolo",
                refined_created_utc="2026-02-11T00:50:00+00:00",
                source_keypoint_run="keypoints_a_yolo",
                keypoint_method="yolo_pose",
                review_state="needs_review",
                review_intended_use="training",
                review_reviewer="alice",
                review_timestamp_utc="2026-02-11T00:55:00+00:00",
                usable_rate=0.82,
                usable_count=82,
                total_count=100,
                raw_success_rate=0.9,
                raw_successful=90,
            ),
        ],
    )
    registry.replace_keypoint_quality(
        "dataset_b",
        [
            _quality_record(
                refined_run="refined_kp_b_trad",
                refined_created_utc="2026-02-11T01:30:00+00:00",
                source_keypoint_run="keypoints_b_trad",
                keypoint_method="traditional_pose",
                review_state=None,
                review_intended_use=None,
                review_reviewer=None,
                review_timestamp_utc=None,
                usable_rate=0.65,
                usable_count=65,
                total_count=100,
                raw_success_rate=0.7,
                raw_successful=70,
            ),
        ],
    )
    registry.replace_keypoint_quality(
        "dataset_c",
        [
            _quality_record(
                refined_run="refined_kp_c_yolo",
                refined_created_utc="2026-02-11T02:30:00+00:00",
                source_keypoint_run="keypoints_c_yolo",
                keypoint_method="yolo_pose",
                review_state="approved",
                review_intended_use="training",
                review_reviewer="carol",
                review_timestamp_utc="2026-02-11T02:35:00+00:00",
                usable_rate=0.91,
                usable_count=91,
                total_count=100,
                raw_success_rate=0.92,
                raw_successful=92,
            ),
        ],
    )

    registry.upsert_keypoint_performance(
        dataset_id="dataset_a",
        keypoint_run="keypoints_2026-02-11_00-10-00",
        keypoint_created_utc="2026-02-11T00:10:00+00:00",
        recording_id="recording_a",
        zarr_use="analysis",
        keypoint_method="yolo_pose",
        model_run_id="run_pose_model_v1",
        model_set_id="pose_set_v1",
        model_path="/models/pose_model_v1.pt",
        model_name="pose_model_v1.pt",
        source_crop_run="crop_a",
        source_detect_run="detect_a",
        source_refined_run="refined_detect_a",
        total_rois=100,
        successful_detections=96,
        failed_detections=4,
        success_rate_percent=96.0,
        frames_with_keypoints=95,
        mean_confidence=0.92,
        duration_seconds=20.0,
        inference_duration_seconds=18.0,
        keypoints_per_second=240.0,
        inference_average_fps=220.0,
        batch_size=16,
        imgsz="[256,256]",
        conf_threshold=0.25,
        iou_threshold=0.7,
        summary_statistics_json=None,
    )
    registry.upsert_keypoint_performance(
        dataset_id="dataset_b",
        keypoint_run="keypoints_2026-02-11_01-10-00",
        keypoint_created_utc="2026-02-11T01:10:00+00:00",
        recording_id="recording_b",
        zarr_use="analysis",
        keypoint_method="traditional_pose",
        model_run_id=None,
        model_set_id=None,
        model_path=None,
        model_name=None,
        source_crop_run="crop_b",
        source_detect_run="detect_b",
        source_refined_run="refined_detect_b",
        total_rois=100,
        successful_detections=72,
        failed_detections=28,
        success_rate_percent=72.0,
        frames_with_keypoints=70,
        mean_confidence=0.74,
        duration_seconds=45.0,
        inference_duration_seconds=45.0,
        keypoints_per_second=120.0,
        inference_average_fps=115.0,
        batch_size=16,
        imgsz="[256,256]",
        conf_threshold=None,
        iou_threshold=None,
        summary_statistics_json=None,
    )
    registry.upsert_keypoint_performance(
        dataset_id="dataset_c",
        keypoint_run="keypoints_2026-02-11_02-10-00",
        keypoint_created_utc="2026-02-11T02:10:00+00:00",
        recording_id="recording_c",
        zarr_use="analysis",
        keypoint_method="yolo_pose",
        model_run_id="run_pose_model_v2",
        model_set_id="pose_set_v2",
        model_path="/models/pose_model_v2.pt",
        model_name="pose_model_v2.pt",
        source_crop_run="crop_c",
        source_detect_run="detect_c",
        source_refined_run="refined_detect_c",
        total_rois=100,
        successful_detections=88,
        failed_detections=12,
        success_rate_percent=88.0,
        frames_with_keypoints=86,
        mean_confidence=0.89,
        duration_seconds=30.0,
        inference_duration_seconds=28.0,
        keypoints_per_second=180.0,
        inference_average_fps=175.0,
        batch_size=16,
        imgsz="[256,256]",
        conf_threshold=0.25,
        iou_threshold=0.7,
        summary_statistics_json=None,
    )
    registry.close()


def _seed_detect_quality_rows(registry_path: Path) -> None:
    registry = Registry(registry_path)
    registry.replace_detect_quality(
        "dataset_a",
        [
            {
                "refined_run": "refined_detect_a_yolo",
                "refined_created_utc": "2026-02-11T00:20:00+00:00",
                "source_detect_run": "detect_a",
                "detect_method": "yolo",
                "review_state": "approved",
                "review_intended_use": "training",
                "review_reviewer": "alice",
                "review_timestamp_utc": "2026-02-11T00:30:00+00:00",
                "review_resolved_group": "manual",
                "total_detections": 1000,
                "real_detections": 960,
                "interpolated_detections": 40,
                "interpolated_detections_rate": 0.04,
                "quality_updated_utc": "2026-02-11T00:31:00+00:00",
                "zarr_mtime_ns": 123456789,
            },
            {
                "refined_run": "refined_detect_a_trad",
                "refined_created_utc": "2026-02-11T00:40:00+00:00",
                "source_detect_run": "detect_a",
                "detect_method": "traditional",
                "review_state": "rejected",
                "review_intended_use": "full_recording",
                "review_reviewer": "bob",
                "review_timestamp_utc": "2026-02-11T00:50:00+00:00",
                "review_resolved_group": "auto",
                "total_detections": 1000,
                "real_detections": 700,
                "interpolated_detections": 300,
                "interpolated_detections_rate": 0.3,
                "quality_updated_utc": "2026-02-11T00:51:00+00:00",
                "zarr_mtime_ns": 123456789,
            },
        ],
    )
    registry.close()


def _seed_detection_data_profile_rows(registry_path: Path) -> None:
    registry = Registry(registry_path)
    profile_records = [
        {
            "dataset_id": "dataset_a",
            "profile_run": "profile_a_v1",
            "recording_id": "recording_a",
            "zarr_use": "analysis",
            "detection_type": "filtered",
            "detection_path": "refined_detect_runs/refined_a_v1/filtered",
            "profile_created_utc": "2026-02-12T00:00:00+00:00",
            "frames_total": 100,
            "frames_with_detections": 80,
            "coverage_percent": 80.0,
            "detections_total": 800,
            "detections_per_frame_p50": 8.0,
            "detections_per_frame_p90": 9.0,
            "w_p10": 0.1,
            "w_p50": 0.2,
            "w_p90": 0.3,
            "h_p10": 0.1,
            "h_p50": 0.2,
            "h_p90": 0.3,
            "area_p10": 0.01,
            "area_p50": 0.04,
            "area_p90": 0.09,
            "aspect_ratio_p10": 0.8,
            "aspect_ratio_p50": 1.0,
            "aspect_ratio_p90": 1.2,
            "edge_proximity_rate": 0.10,
            "rig_id": "rig_a",
            "camera_id": "cam_1",
            "arena_id": "arena_x",
            "dish_design": "cedar",
            "canvas_name": "canvas_a",
            "protocol_name": "DefaultScreen",
            "profile_json": json.dumps({"run": "profile_a_v1"}),
            "zarr_mtime_ns": 1000,
        },
        {
            "dataset_id": "dataset_a",
            "profile_run": "profile_a_v2",
            "recording_id": "recording_a",
            "zarr_use": "analysis",
            "detection_type": "manual",
            "detection_path": "refined_detect_runs/refined_a_v2/manual",
            "profile_created_utc": "2026-02-12T02:00:00+00:00",
            "frames_total": 100,
            "frames_with_detections": 97,
            "coverage_percent": 97.0,
            "detections_total": 970,
            "detections_per_frame_p50": 9.0,
            "detections_per_frame_p90": 10.0,
            "w_p10": 0.1,
            "w_p50": 0.2,
            "w_p90": 0.3,
            "h_p10": 0.1,
            "h_p50": 0.2,
            "h_p90": 0.3,
            "area_p10": 0.01,
            "area_p50": 0.04,
            "area_p90": 0.09,
            "aspect_ratio_p10": 0.8,
            "aspect_ratio_p50": 1.0,
            "aspect_ratio_p90": 1.2,
            "edge_proximity_rate": 0.02,
            "rig_id": "rig_a",
            "camera_id": "cam_1",
            "arena_id": "arena_x",
            "dish_design": "cedar",
            "canvas_name": "canvas_a",
            "protocol_name": "DefaultScreen",
            "profile_json": json.dumps({"run": "profile_a_v2"}),
            "zarr_mtime_ns": 2000,
        },
        {
            "dataset_id": "dataset_b",
            "profile_run": "profile_b_v1",
            "recording_id": "recording_b",
            "zarr_use": "analysis",
            "detection_type": "interpolated",
            "detection_path": "refined_detect_runs/refined_b_v1/interpolated",
            "profile_created_utc": "2026-02-12T01:00:00+00:00",
            "frames_total": 100,
            "frames_with_detections": 88,
            "coverage_percent": 88.0,
            "detections_total": 880,
            "detections_per_frame_p50": 8.0,
            "detections_per_frame_p90": 9.0,
            "w_p10": 0.1,
            "w_p50": 0.2,
            "w_p90": 0.3,
            "h_p10": 0.1,
            "h_p50": 0.2,
            "h_p90": 0.3,
            "area_p10": 0.01,
            "area_p50": 0.04,
            "area_p90": 0.09,
            "aspect_ratio_p10": 0.8,
            "aspect_ratio_p50": 1.0,
            "aspect_ratio_p90": 1.2,
            "edge_proximity_rate": 0.07,
            "rig_id": "rig_a",
            "camera_id": "cam_2",
            "arena_id": "arena_x",
            "dish_design": "cedar",
            "canvas_name": "canvas_a",
            "protocol_name": "DefaultScreen",
            "profile_json": json.dumps({"run": "profile_b_v1"}),
            "zarr_mtime_ns": 1500,
        },
    ]
    for record in profile_records:
        registry.upsert_detection_data_profile(
            dataset_id=record["dataset_id"],
            profile_run=record["profile_run"],
            recording_id=record["recording_id"],
            zarr_use=record["zarr_use"],
            detection_type=record["detection_type"],
            detection_path=record["detection_path"],
            profile_created_utc=record["profile_created_utc"],
            frames_total=record["frames_total"],
            frames_with_detections=record["frames_with_detections"],
            coverage_percent=record["coverage_percent"],
            detections_total=record["detections_total"],
            detections_per_frame_p50=record["detections_per_frame_p50"],
            detections_per_frame_p90=record["detections_per_frame_p90"],
            w_p10=record["w_p10"],
            w_p50=record["w_p50"],
            w_p90=record["w_p90"],
            h_p10=record["h_p10"],
            h_p50=record["h_p50"],
            h_p90=record["h_p90"],
            area_p10=record["area_p10"],
            area_p50=record["area_p50"],
            area_p90=record["area_p90"],
            aspect_ratio_p10=record["aspect_ratio_p10"],
            aspect_ratio_p50=record["aspect_ratio_p50"],
            aspect_ratio_p90=record["aspect_ratio_p90"],
            edge_proximity_rate=record["edge_proximity_rate"],
            rig_id=record["rig_id"],
            camera_id=record["camera_id"],
            arena_id=record["arena_id"],
            dish_design=record["dish_design"],
            canvas_name=record["canvas_name"],
            protocol_name=record["protocol_name"],
            profile_json=record["profile_json"],
            zarr_mtime_ns=record["zarr_mtime_ns"],
        )
    registry.close()


def _rewrite_detect_quality_current_view(
    registry_path: Path,
    *,
    include_shared_review_columns: bool,
    use_legacy_timestamp_column: bool,
) -> None:
    registry = Registry(registry_path)
    select_columns = [
        "dataset_id",
        "refined_run",
        "refined_created_utc",
        "source_detect_run",
        "detect_method",
        "review_state",
        "review_intended_use",
        "review_reviewer",
        "review_timestamp_utc AS timestamp" if use_legacy_timestamp_column else "review_timestamp_utc",
        "review_resolved_group",
        "total_detections",
        "real_detections",
        "interpolated_detections",
        "interpolated_detections_rate",
        "quality_updated_utc",
        "zarr_mtime_ns",
    ]
    if include_shared_review_columns:
        select_columns.extend(
            [
                "'manual' AS review_method",
                "'detect-shared-note' AS review_notes",
            ]
        )
    registry.conn.execute("DROP VIEW IF EXISTS detect_quality_current;")
    registry.conn.execute(
        f"""
        CREATE VIEW detect_quality_current AS
        WITH ranked AS (
            SELECT
                dq.*,
                ROW_NUMBER() OVER (
                    PARTITION BY dq.dataset_id, COALESCE(dq.detect_method, '')
                    ORDER BY
                        COALESCE(dq.review_timestamp_utc, dq.refined_created_utc, dq.quality_updated_utc) DESC,
                        COALESCE(dq.refined_created_utc, '') DESC,
                        dq.refined_run DESC
                ) AS _rn
            FROM detect_quality dq
        )
        SELECT
            {", ".join(select_columns)}
        FROM ranked
        WHERE _rn = 1;
        """
    )
    registry.conn.commit()
    registry.close()


def _rewrite_keypoint_quality_current_view(
    registry_path: Path,
    *,
    include_shared_review_columns: bool,
    use_legacy_timestamp_column: bool,
) -> None:
    registry = Registry(registry_path)
    select_columns = [
        "dataset_id",
        "refined_run",
        "refined_created_utc",
        "source_keypoint_run",
        "keypoint_method",
        "review_state",
        "review_intended_use",
        "review_reviewer",
        "review_timestamp_utc AS timestamp" if use_legacy_timestamp_column else "review_timestamp_utc",
        "usable_keypoints",
        "total_keypoints",
        "usable_keypoints_rate",
        "raw_keypoints_success_rate",
        "raw_keypoints_successful",
        "quality_updated_utc",
        "zarr_mtime_ns",
    ]
    if include_shared_review_columns:
        select_columns.extend(
            [
                "'hybrid' AS review_method",
                "'keypoint-shared-note' AS review_notes",
            ]
        )
    registry.conn.execute("DROP VIEW IF EXISTS keypoint_quality_current;")
    registry.conn.execute(
        f"""
        CREATE VIEW keypoint_quality_current AS
        WITH ranked AS (
            SELECT
                kq.*,
                ROW_NUMBER() OVER (
                    PARTITION BY kq.dataset_id, COALESCE(kq.keypoint_method, '')
                    ORDER BY
                        COALESCE(kq.review_timestamp_utc, kq.refined_created_utc, kq.quality_updated_utc) DESC,
                        COALESCE(kq.refined_created_utc, '') DESC,
                        kq.refined_run DESC
                ) AS _rn
            FROM keypoint_quality kq
        )
        SELECT
            {", ".join(select_columns)}
        FROM ranked
        WHERE _rn = 1;
        """
    )
    registry.conn.commit()
    registry.close()


def _seed_recording_step_status_rows(registry_path: Path) -> None:
    registry = Registry(registry_path)
    registry.conn.executemany(
        """
        INSERT INTO recording_step_status (
            dataset_id, recording_id, step_name, status, run_name, source, updated_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """,
        [
            ("dataset_a", "recording_a", "detect", "ok", "detect_a", "unit_test", "2026-02-11T00:00:00+00:00"),
            ("dataset_a", "recording_a", "keypoints", "missing", None, "unit_test", "2026-02-11T00:10:00+00:00"),
            ("dataset_b", "recording_b", "detect", "missing", None, "unit_test", "2026-02-11T01:00:00+00:00"),
            ("dataset_b", "recording_b", "keypoints", "ok", "keypoints_b", "unit_test", "2026-02-11T01:10:00+00:00"),
            ("dataset_c", "recording_c", "detect", "ok", "detect_c", "unit_test", "2026-02-11T02:00:00+00:00"),
            ("dataset_c", "recording_c", "keypoints", "ok", "keypoints_c", "unit_test", "2026-02-11T02:10:00+00:00"),
        ],
    )
    registry.conn.commit()
    registry.close()


def test_registry_query_filters_by_cross_id(tmp_path: Path, capsys) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry_for_subject_filters(registry_path)

    rc = registry_query_main(
        [
            "--registry",
            str(registry_path),
            "--cross-id",
            "cross_a",
            "--json",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    dataset_ids = {row["dataset_id"] for row in payload}
    assert dataset_ids == {"dataset_a"}


def test_registry_query_filters_by_genotype_and_dpf(tmp_path: Path, capsys) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry_for_subject_filters(registry_path)

    rc = registry_query_main(
        [
            "--registry",
            str(registry_path),
            "--genotype",
            "genotype_y",
            "--dpf",
            "12",
            "--json",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    dataset_ids = {row["dataset_id"] for row in payload}
    assert dataset_ids == {"dataset_b"}


def test_registry_query_filters_by_dpf_range(tmp_path: Path, capsys) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry_for_subject_filters(registry_path)

    rc = registry_query_main(
        [
            "--registry",
            str(registry_path),
            "--dpf-min",
            "9",
            "--dpf-max",
            "12",
            "--json",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    dataset_ids = {row["dataset_id"] for row in payload}
    assert dataset_ids == {"dataset_b"}


def test_registry_query_rejects_invalid_dpf_range(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry_for_subject_filters(registry_path)

    try:
        registry_query_main(
            [
                "--registry",
                str(registry_path),
                "--dpf-min",
                "13",
                "--dpf-max",
                "12",
                "--json",
            ]
        )
    except SystemExit as exc:
        assert "--dpf-min must be <= --dpf-max." in str(exc)
    else:  # pragma: no cover - defensive branch
        raise AssertionError("Expected SystemExit for invalid DPF range.")


def test_registry_query_filters_by_detect_coverage_min(tmp_path: Path, capsys) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry_for_detect_filters(registry_path)

    rc = registry_query_main(
        [
            "--registry",
            str(registry_path),
            "--detect-coverage-min",
            "90",
            "--json",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    dataset_ids = {row["dataset_id"] for row in payload}
    assert dataset_ids == {"dataset_a"}


def test_registry_query_filters_by_step_name_and_status(tmp_path: Path, capsys) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry_for_detect_filters(registry_path)
    _seed_recording_step_status_rows(registry_path)

    rc = registry_query_main(
        [
            "--registry",
            str(registry_path),
            "--step-name",
            "keypoints",
            "--step-status",
            "missing",
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert {row["dataset_id"] for row in payload} == {"dataset_a"}
    assert payload[0]["recording_step_name"] == "keypoints"
    assert payload[0]["recording_step_status"] == "missing"


def test_registry_query_filters_by_non_ok_step_status(tmp_path: Path, capsys) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry_for_detect_filters(registry_path)
    _seed_recording_step_status_rows(registry_path)

    rc = registry_query_main(
        [
            "--registry",
            str(registry_path),
            "--step-status",
            "non-ok",
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert {row["dataset_id"] for row in payload} == {"dataset_a", "dataset_b"}
    statuses = {row["recording_step_status"] for row in payload}
    assert statuses == {"missing"}


def test_registry_query_rejects_invalid_step_status(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry_for_detect_filters(registry_path)
    _seed_recording_step_status_rows(registry_path)

    with pytest.raises(SystemExit, match="--step-status must be one of"):
        registry_query_main(
            [
                "--registry",
                str(registry_path),
                "--step-status",
                "broken",
                "--json",
            ]
        )


def test_registry_query_detect_model_only_and_model_like(tmp_path: Path, capsys) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry_for_detect_filters(registry_path)

    rc = registry_query_main(
        [
            "--registry",
            str(registry_path),
            "--detect-model-only",
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    dataset_ids = {row["dataset_id"] for row in payload}
    assert dataset_ids == {"dataset_a", "dataset_c"}
    runs = {row.get("detect_model_run_id") for row in payload}
    assert runs == {"run_detect_model_v1", "run_detect_model_v2"}

    rc2 = registry_query_main(
        [
            "--registry",
            str(registry_path),
            "--detect-model-like",
            "v2",
            "--json",
        ]
    )
    assert rc2 == 0
    payload2 = json.loads(capsys.readouterr().out)
    dataset_ids2 = {row["dataset_id"] for row in payload2}
    assert dataset_ids2 == {"dataset_c"}


def test_registry_query_group_by_model_json(tmp_path: Path, capsys) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry_for_detect_filters(registry_path)

    rc = registry_query_main(
        [
            "--registry",
            str(registry_path),
            "--group-by-model",
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 2
    names = {row["model_name"] for row in payload}
    assert names == {"detect_model_v1.pt", "detect_model_v2.pt"}
    run_ids = {row["model_run_id"] for row in payload}
    assert run_ids == {"run_detect_model_v1", "run_detect_model_v2"}
    counts = {row["model_name"]: row["recordings"] for row in payload}
    assert counts["detect_model_v1.pt"] == 1
    assert counts["detect_model_v2.pt"] == 1


def test_registry_query_group_by_dimension_json_includes_percentiles(tmp_path: Path, capsys) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry_for_detect_filters(registry_path)

    rc = registry_query_main(
        [
            "--registry",
            str(registry_path),
            "--group-by",
            "rig",
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 2
    by_group = {row["group_value"]: row for row in payload}
    assert set(by_group.keys()) == {"rig_a", "rig_b"}

    rig_a = by_group["rig_a"]
    assert rig_a["datasets"] == 2
    assert rig_a["recordings"] == 2
    assert rig_a["coverage_avg"] == 82.5
    assert rig_a["coverage_p50"] == 82.5
    assert rig_a["fps_p50"] == 90.0
    assert rig_a["read_ms_p50"] == 105.0

    for key in (
        "coverage_p10",
        "coverage_p50",
        "coverage_p90",
        "fps_p10",
        "fps_p50",
        "fps_p90",
        "read_ms_p10",
        "read_ms_p50",
        "read_ms_p90",
    ):
        assert key in rig_a


def test_registry_query_group_by_model_alias_matches_explicit(tmp_path: Path, capsys) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry_for_detect_filters(registry_path)

    rc_alias = registry_query_main(
        [
            "--registry",
            str(registry_path),
            "--group-by-model",
            "--json",
        ]
    )
    assert rc_alias == 0
    alias_payload = json.loads(capsys.readouterr().out)

    rc_explicit = registry_query_main(
        [
            "--registry",
            str(registry_path),
            "--group-by",
            "model",
            "--json",
        ]
    )
    assert rc_explicit == 0
    explicit_payload = json.loads(capsys.readouterr().out)
    assert alias_payload == explicit_payload


def test_registry_query_rejects_group_by_model_conflict(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry_for_detect_filters(registry_path)

    try:
        registry_query_main(
            [
                "--registry",
                str(registry_path),
                "--group-by-model",
                "--group-by",
                "rig",
            ]
        )
    except SystemExit as exc:
        assert "--group-by-model cannot be combined with --group-by non-model values." in str(exc)
    else:  # pragma: no cover - defensive branch
        raise AssertionError("Expected SystemExit for conflicting group-by args.")


def test_registry_query_detect_model_summary_mode_json(tmp_path: Path, capsys) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry_for_detect_filters(registry_path)

    rc = registry_query_main(
        [
            "--registry",
            str(registry_path),
            "--detect-model-summary",
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 2
    run_ids = {row["model_run_id"] for row in payload}
    assert run_ids == {"run_detect_model_v1", "run_detect_model_v2"}
    for row in payload:
        assert "coverage_p50" in row
        assert "fps_p50" in row
        assert "read_ms_p50" in row


def test_registry_query_detect_model_summary_mode_filters(tmp_path: Path, capsys) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry_for_detect_filters(registry_path)

    rc = registry_query_main(
        [
            "--registry",
            str(registry_path),
            "--detect-model-summary",
            "--detect-model-like",
            "v2",
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 1
    assert payload[0]["model_run_id"] == "run_detect_model_v2"


def test_registry_query_detect_model_summary_rejects_output_file_list(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry_for_detect_filters(registry_path)
    out_file = tmp_path / "rows.txt"

    try:
        registry_query_main(
            [
                "--registry",
                str(registry_path),
                "--detect-model-summary",
                "--output-file-list",
                str(out_file),
            ]
        )
    except SystemExit as exc:
        assert "--output-file-list is only supported for dataset-row query mode." in str(exc)
    else:  # pragma: no cover - defensive branch
        raise AssertionError("Expected SystemExit for invalid output-file-list usage.")


def test_registry_query_detection_data_profile_latest_mode_json(tmp_path: Path, capsys) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry_for_detect_filters(registry_path)
    _seed_detection_data_profile_rows(registry_path)

    rc = registry_query_main(
        [
            "--registry",
            str(registry_path),
            "--detection-data-profile-latest",
            "--profile-detection-type",
            "manual",
            "--profile-coverage-min",
            "90",
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert {row["dataset_id"] for row in payload} == {"dataset_a"}
    row = payload[0]
    assert row["profile_run"] == "profile_a_v2"
    assert row["detection_type"] == "manual"
    assert row["coverage_percent"] == pytest.approx(97.0)


def test_registry_query_recording_detection_data_profile_latest_mode_json(tmp_path: Path, capsys) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry_for_detect_filters(registry_path)
    _seed_detection_data_profile_rows(registry_path)

    rc = registry_query_main(
        [
            "--registry",
            str(registry_path),
            "--recording-detection-data-profile-latest",
            "--profile-recording-id",
            "recording_b",
            "--profile-dataset-id",
            "dataset_b",
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 1
    row = payload[0]
    assert row["recording_id"] == "recording_b"
    assert row["dataset_id"] == "dataset_b"
    assert row["profile_run"] == "profile_b_v1"


def test_registry_query_detection_data_profile_modes_are_mutually_exclusive(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry_for_detect_filters(registry_path)
    _seed_detection_data_profile_rows(registry_path)

    with pytest.raises(
        SystemExit,
        match="--detection-data-profile-latest and --recording-detection-data-profile-latest are mutually exclusive.",
    ):
        registry_query_main(
            [
                "--registry",
                str(registry_path),
                "--detection-data-profile-latest",
                "--recording-detection-data-profile-latest",
            ]
        )


def test_registry_query_detection_data_profile_mode_rejects_output_file_list(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry_for_detect_filters(registry_path)
    _seed_detection_data_profile_rows(registry_path)
    out_file = tmp_path / "rows.txt"

    with pytest.raises(SystemExit, match="--output-file-list is only supported for dataset-row query mode."):
        registry_query_main(
            [
                "--registry",
                str(registry_path),
                "--detection-data-profile-latest",
                "--output-file-list",
                str(out_file),
            ]
        )


def test_registry_query_filters_by_crop_review_state(tmp_path: Path, capsys) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry_for_detect_filters(registry_path)

    rc = registry_query_main(
        [
            "--registry",
            str(registry_path),
            "--crop-review-state",
            "approved",
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    dataset_ids = {row["dataset_id"] for row in payload}
    assert dataset_ids == {"dataset_a"}
    assert payload[0]["crop_review_state"] == "approved"


def test_registry_query_filters_by_crop_missing_review_state(tmp_path: Path, capsys) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry_for_detect_filters(registry_path)

    rc = registry_query_main(
        [
            "--registry",
            str(registry_path),
            "--crop-review-state",
            "missing",
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    dataset_ids = {row["dataset_id"] for row in payload}
    assert dataset_ids == {"dataset_b"}
    assert payload[0]["crop_review_state"] is None


def test_registry_query_filters_by_crop_source_and_intended_use(tmp_path: Path, capsys) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry_for_detect_filters(registry_path)

    rc = registry_query_main(
        [
            "--registry",
            str(registry_path),
            "--crop-review-intended-use",
            "full_recording",
            "--crop-source-type",
            "interpolated",
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    dataset_ids = {row["dataset_id"] for row in payload}
    assert dataset_ids == {"dataset_c"}
    assert payload[0]["crop_source_type"] == "interpolated"


def test_registry_query_filters_by_crop_percent_frames_threshold(tmp_path: Path, capsys) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry_for_detect_filters(registry_path)

    rc = registry_query_main(
        [
            "--registry",
            str(registry_path),
            "--crop-percent-frames-min",
            "90",
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    dataset_ids = {row["dataset_id"] for row in payload}
    assert dataset_ids == {"dataset_a"}
    assert payload[0]["crop_percent_frames_with_crops"] == 95.0


def test_registry_query_filters_by_eye_mask_review_state(tmp_path: Path, capsys) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry_for_detect_filters(registry_path)
    _seed_eye_mask_performance_rows(registry_path)

    rc = registry_query_main(
        [
            "--registry",
            str(registry_path),
            "--eye-mask-review-state",
            "approved",
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert {row["dataset_id"] for row in payload} == {"dataset_a"}
    assert payload[0]["eye_mask_stage_group"] == "refined_eye_masks_runs"
    assert payload[0]["eye_mask_review_state"] == "approved"


def test_registry_query_filters_by_eye_mask_stale_and_lifecycle_state(tmp_path: Path, capsys) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry_for_detect_filters(registry_path)
    _seed_eye_mask_performance_rows(registry_path)

    rc = registry_query_main(
        [
            "--registry",
            str(registry_path),
            "--eye-mask-stale-state",
            "stale",
            "--eye-mask-lifecycle-state",
            "stale",
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert {row["dataset_id"] for row in payload} == {"dataset_b"}
    assert payload[0]["eye_mask_source_keypoint_stale_state"] == "stale"
    assert payload[0]["eye_mask_lifecycle_state"] == "stale"


def test_registry_query_filters_by_eye_mask_stage_and_success_rate(tmp_path: Path, capsys) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry_for_detect_filters(registry_path)
    _seed_eye_mask_performance_rows(registry_path)

    rc = registry_query_main(
        [
            "--registry",
            str(registry_path),
            "--eye-mask-stage",
            "eye_masks_runs",
            "--eye-mask-method",
            "traditional_eye_segmentation",
            "--eye-mask-success-rate-min",
            "0.9",
            "--eye-mask-rois-per-second-min",
            "3.0",
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert {row["dataset_id"] for row in payload} == {"dataset_c"}
    assert payload[0]["eye_mask_stage_group"] == "eye_masks_runs"
    assert payload[0]["eye_mask_successful_roi_pair_rate"] == pytest.approx(0.9545)


def test_registry_query_filters_by_keypoint_review_and_usable_rate(tmp_path: Path, capsys) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry_for_detect_filters(registry_path)
    _seed_keypoint_quality_and_performance_rows(registry_path)

    rc = registry_query_main(
        [
            "--registry",
            str(registry_path),
            "--keypoint-method",
            "yolo_pose",
            "--keypoint-review-state",
            "approved",
            "--keypoint-review-intended-use",
            "training",
            "--keypoint-usable-rate-min",
            "0.9",
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert {row["dataset_id"] for row in payload} == {"dataset_c"}
    assert payload[0]["keypoint_review_state"] == "approved"
    assert payload[0]["keypoint_usable_keypoints_rate"] == pytest.approx(0.91)
    assert payload[0]["keypoint_method"] == "yolo_pose"


def test_registry_query_filters_by_keypoint_missing_review_state(tmp_path: Path, capsys) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry_for_detect_filters(registry_path)
    _seed_keypoint_quality_and_performance_rows(registry_path)

    rc = registry_query_main(
        [
            "--registry",
            str(registry_path),
            "--keypoint-review-state",
            "missing",
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert {row["dataset_id"] for row in payload} == {"dataset_b"}
    assert payload[0]["keypoint_review_state"] is None


def test_registry_query_detect_and_keypoint_shared_review_fields_when_available(tmp_path: Path, capsys) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry_for_detect_filters(registry_path)
    _seed_detect_quality_rows(registry_path)
    _seed_keypoint_quality_and_performance_rows(registry_path)
    _rewrite_detect_quality_current_view(
        registry_path,
        include_shared_review_columns=True,
        use_legacy_timestamp_column=False,
    )
    _rewrite_keypoint_quality_current_view(
        registry_path,
        include_shared_review_columns=True,
        use_legacy_timestamp_column=False,
    )

    rc_detect = registry_query_main(
        [
            "--registry",
            str(registry_path),
            "--detect-coverage-min",
            "90",
            "--json",
        ]
    )
    assert rc_detect == 0
    detect_payload = json.loads(capsys.readouterr().out)
    assert {row["dataset_id"] for row in detect_payload} == {"dataset_a"}
    detect_row = detect_payload[0]
    assert detect_row["detect_quality_method"] == "yolo"
    assert detect_row["detect_review_state"] == "approved"
    assert detect_row["detect_review_intended_use"] == "training"
    assert detect_row["detect_review_method"] == "manual"
    assert detect_row["detect_review_notes"] == "detect-shared-note"
    assert detect_row["detect_review_timestamp_utc"] == "2026-02-11T00:30:00+00:00"

    rc_keypoint = registry_query_main(
        [
            "--registry",
            str(registry_path),
            "--keypoint-method",
            "yolo_pose",
            "--keypoint-review-state",
            "approved",
            "--json",
        ]
    )
    assert rc_keypoint == 0
    keypoint_payload = json.loads(capsys.readouterr().out)
    assert {row["dataset_id"] for row in keypoint_payload} == {"dataset_c"}
    keypoint_row = keypoint_payload[0]
    assert keypoint_row["keypoint_review_state"] == "approved"
    assert keypoint_row["keypoint_review_intended_use"] == "training"
    assert keypoint_row["keypoint_review_method"] == "hybrid"
    assert keypoint_row["keypoint_review_notes"] == "keypoint-shared-note"
    assert keypoint_row["keypoint_review_timestamp_utc"] == "2026-02-11T02:35:00+00:00"


def test_registry_query_detect_and_keypoint_review_field_legacy_fallback(tmp_path: Path, capsys) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry_for_detect_filters(registry_path)
    _seed_detect_quality_rows(registry_path)
    _seed_keypoint_quality_and_performance_rows(registry_path)
    _rewrite_detect_quality_current_view(
        registry_path,
        include_shared_review_columns=False,
        use_legacy_timestamp_column=True,
    )
    _rewrite_keypoint_quality_current_view(
        registry_path,
        include_shared_review_columns=False,
        use_legacy_timestamp_column=True,
    )

    rc_detect = registry_query_main(
        [
            "--registry",
            str(registry_path),
            "--detect-coverage-min",
            "90",
            "--json",
        ]
    )
    assert rc_detect == 0
    detect_payload = json.loads(capsys.readouterr().out)
    assert {row["dataset_id"] for row in detect_payload} == {"dataset_a"}
    detect_row = detect_payload[0]
    assert detect_row["detect_quality_method"] == "yolo"
    assert detect_row["detect_review_timestamp_utc"] == "2026-02-11T00:30:00+00:00"
    assert detect_row["detect_review_method"] is None
    assert detect_row["detect_review_notes"] is None

    rc_keypoint = registry_query_main(
        [
            "--registry",
            str(registry_path),
            "--keypoint-method",
            "yolo_pose",
            "--keypoint-review-state",
            "approved",
            "--json",
        ]
    )
    assert rc_keypoint == 0
    keypoint_payload = json.loads(capsys.readouterr().out)
    assert {row["dataset_id"] for row in keypoint_payload} == {"dataset_c"}
    keypoint_row = keypoint_payload[0]
    assert keypoint_row["keypoint_review_timestamp_utc"] == "2026-02-11T02:35:00+00:00"
    assert keypoint_row["keypoint_review_method"] is None
    assert keypoint_row["keypoint_review_notes"] is None


def test_registry_query_filters_by_keypoint_runtime_and_model_like(tmp_path: Path, capsys) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry_for_detect_filters(registry_path)
    _seed_keypoint_quality_and_performance_rows(registry_path)

    rc = registry_query_main(
        [
            "--registry",
            str(registry_path),
            "--keypoint-success-rate-min",
            "90",
            "--keypoint-kps-min",
            "200",
            "--keypoint-duration-max",
            "25",
            "--keypoint-model-like",
            "v1",
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert {row["dataset_id"] for row in payload} == {"dataset_a"}
    assert payload[0]["keypoint_model_run_id"] == "run_pose_model_v1"
    assert payload[0]["keypoint_success_rate_percent"] == pytest.approx(96.0)
    assert payload[0]["keypoint_keypoints_per_second"] == pytest.approx(240.0)


def test_registry_query_keypoint_group_by_method_json_includes_percentiles(tmp_path: Path, capsys) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry_for_detect_filters(registry_path)
    _seed_keypoint_quality_and_performance_rows(registry_path)

    rc = registry_query_main(
        [
            "--registry",
            str(registry_path),
            "--keypoint-group-by",
            "method",
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 2
    by_method = {row["group_value"]: row for row in payload}
    assert set(by_method.keys()) == {"traditional_pose", "yolo_pose"}

    yolo = by_method["yolo_pose"]
    assert yolo["datasets"] == 2
    assert yolo["recordings"] == 2
    assert yolo["success_rate_p50"] == pytest.approx(92.0)
    assert yolo["kps_p50"] == pytest.approx(210.0)
    assert yolo["duration_p50"] == pytest.approx(25.0)
    for key in (
        "success_rate_p10",
        "success_rate_p50",
        "success_rate_p90",
        "kps_p10",
        "kps_p50",
        "kps_p90",
        "duration_p10",
        "duration_p50",
        "duration_p90",
    ):
        assert key in yolo


def test_registry_query_keypoint_group_by_model_with_model_only(tmp_path: Path, capsys) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry_for_detect_filters(registry_path)
    _seed_keypoint_quality_and_performance_rows(registry_path)

    rc = registry_query_main(
        [
            "--registry",
            str(registry_path),
            "--keypoint-group-by",
            "model",
            "--keypoint-model-only",
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 2
    names = {row["model_name"] for row in payload}
    assert names == {"pose_model_v1.pt", "pose_model_v2.pt"}
    run_ids = {row["model_run_id"] for row in payload}
    assert run_ids == {"run_pose_model_v1", "run_pose_model_v2"}


def test_registry_query_rejects_detect_and_keypoint_group_by_conflict(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    _seed_registry_for_detect_filters(registry_path)
    _seed_keypoint_quality_and_performance_rows(registry_path)

    try:
        registry_query_main(
            [
                "--registry",
                str(registry_path),
                "--group-by",
                "rig",
                "--keypoint-group-by",
                "method",
            ]
        )
    except SystemExit as exc:
        assert "--group-by cannot be combined with --keypoint-group-by." in str(exc)
    else:  # pragma: no cover - defensive branch
        raise AssertionError("Expected SystemExit for detect/keypoint summary conflict.")
