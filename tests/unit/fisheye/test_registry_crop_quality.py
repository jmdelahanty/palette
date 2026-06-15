"""Unit tests for crop review quality registry extraction and views."""

import json
from pathlib import Path
import sys

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.registry.db import Registry
from fisheye.shared.roi_pixel_contract import crop_run_pixel_contract


def _create_crop_archive(path: Path, *, session_uuid: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(path), mode="w")
    root.attrs["session_uuid"] = session_uuid
    raw = root.create_group("raw_video")
    raw.create_array("images_ds", data=np.zeros((4, 8, 8), dtype=np.uint8), chunks=(1, 8, 8))

    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "crop_new"

    crop_old = crop_parent.create_group("crop_old")
    crop_old.attrs["created_at_utc"] = "2026-02-09T00:00:00+00:00"
    crop_old.attrs["detection_source_type"] = "detect"
    crop_old.attrs["detection_source_path"] = "detect_runs/detect_old"
    crop_old.attrs["source_detect_run"] = "detect_old"
    crop_old.attrs["status"] = "completed"
    crop_old.attrs["summary_statistics"] = {
        "total_frames": 4,
        "frames_with_crops": 2,
        "total_rois_cropped": 2,
        "percent_frames_with_crops": 50.0,
    }
    crop_old.create_array("frame_counts", data=np.array([1, 0, 1, 0], dtype=np.int32), chunks=(4,))
    crop_old.create_array("bbox_norm_coords", data=np.zeros((2, 4), dtype=np.float32), chunks=(2, 4))

    crop_new = crop_parent.create_group("crop_new")
    crop_new.attrs["created_at_utc"] = "2026-02-10T00:00:00+00:00"
    crop_new.attrs["detection_source_type"] = "manual"
    crop_new.attrs["detection_source_path"] = "refined_detect_runs/refined_001/manual"
    crop_new.attrs["source_detect_run"] = "detect_001"
    crop_new.attrs["source_refined_run"] = "refined_001"
    crop_new.attrs["status"] = "completed"
    crop_new.attrs["crop_storage_mode"] = "materialized"
    crop_new.attrs["video_source_type"] = "external"
    crop_new.attrs["roi_live_acceleration_effective"] = "gpu"
    crop_contract = crop_run_pixel_contract(
        crop_storage_mode="materialized",
        video_source_type="external",
        acceleration="gpu",
    )
    crop_new.attrs["roi_pixel_contract"] = crop_contract
    crop_new.attrs["roi_pixel_contract_name"] = crop_contract["name"]
    crop_new.attrs["includes_interpolated"] = True
    crop_new.attrs["n_real_detections"] = 3
    crop_new.attrs["n_interpolated_detections"] = 1
    crop_new.attrs["summary_statistics"] = {
        "total_frames": 4,
        "frames_with_crops": 4,
        "total_rois_cropped": 4,
        "percent_frames_with_crops": 100.0,
    }
    crop_new.attrs["crop_review_status"] = {
        "state": "approved",
        "method": "manual",
        "intended_use": "training",
        "reviewer": "pytest",
        "timestamp": "2026-02-10T01:00:00+00:00",
        "notes": "looks good",
    }
    crop_new.create_array("frame_counts", data=np.array([1, 1, 1, 1], dtype=np.int32), chunks=(4,))
    crop_new.create_array("bbox_norm_coords", data=np.zeros((4, 4), dtype=np.float32), chunks=(4, 4))
    crop_new.create_array("detection_source", data=np.array([0, 1, 0, 0], dtype=np.int8), chunks=(4,))


def _create_cropless_archive(path: Path, *, session_uuid: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(path), mode="w")
    root.attrs["session_uuid"] = session_uuid
    raw = root.create_group("raw_video")
    raw.create_array("images_ds", data=np.zeros((4, 8, 8), dtype=np.uint8), chunks=(1, 8, 8))


def test_schema_has_crop_quality_table_views_and_indexes(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")

    table = registry.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='crop_quality';"
    ).fetchone()
    assert table is not None

    views = registry.conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type='view' AND name IN ('crop_quality_current', 'recording_crop_quality_current');
        """
    ).fetchall()
    assert {str(row["name"]) for row in views} == {"crop_quality_current", "recording_crop_quality_current"}

    indexes = registry.conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type='index' AND name IN (
            'idx_crop_quality_dataset_id',
            'idx_crop_quality_review_gate',
            'idx_crop_quality_source',
            'idx_crop_quality_recording'
        );
        """
    ).fetchall()
    assert {str(row["name"]) for row in indexes} == {
        "idx_crop_quality_dataset_id",
        "idx_crop_quality_review_gate",
        "idx_crop_quality_source",
        "idx_crop_quality_recording",
    }
    registry.close()


def test_register_from_root_populates_crop_quality_and_latest_views(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    zarr_path = tmp_path / "recordings" / "rec_a" / "zarr" / "rec_a_analysis.zarr"
    _create_crop_archive(zarr_path, session_uuid="rec_a_uuid")

    dataset_id = registry.register_from_root(zarr.open_group(str(zarr_path), mode="r"), zarr_path)

    count = registry.conn.execute(
        "SELECT COUNT(*) AS n FROM crop_quality WHERE dataset_id = ?;",
        (dataset_id,),
    ).fetchone()
    assert count is not None
    assert int(count["n"]) == 2

    latest = registry.conn.execute(
        """
        SELECT
            crop_run,
            review_state,
            review_method,
            review_intended_use,
            review_reviewer,
            review_notes,
            source_refined_run,
            detection_source_type,
            crop_storage_mode,
            roi_image_representation,
            roi_pixel_contract_name,
            roi_pixel_contract_json,
            includes_interpolated,
            n_real_detections,
            n_interpolated_detections,
            percent_frames_with_crops
        FROM crop_quality_current
        WHERE dataset_id = ?;
        """,
        (dataset_id,),
    ).fetchone()
    assert latest is not None
    assert str(latest["crop_run"]) == "crop_new"
    assert str(latest["review_state"]) == "approved"
    assert str(latest["review_method"]) == "manual"
    assert str(latest["review_intended_use"]) == "training"
    assert str(latest["review_reviewer"]) == "pytest"
    assert str(latest["review_notes"]) == "looks good"
    assert str(latest["source_refined_run"]) == "refined_001"
    assert str(latest["detection_source_type"]) == "manual"
    assert str(latest["crop_storage_mode"]) == "materialized"
    assert str(latest["roi_image_representation"]) == "uint8_grayscale_roi_v1"
    assert str(latest["roi_pixel_contract_name"]) == "decord_rgb_channel_mean_uint8"
    assert json.loads(str(latest["roi_pixel_contract_json"]))["name"] == "decord_rgb_channel_mean_uint8"
    assert int(latest["includes_interpolated"]) == 1
    assert int(latest["n_real_detections"]) == 3
    assert int(latest["n_interpolated_detections"]) == 1
    assert float(latest["percent_frames_with_crops"]) == 100.0

    rec_latest = registry.conn.execute(
        """
        SELECT recording_id, dataset_id, crop_run, review_state, roi_pixel_contract_name
        FROM recording_crop_quality_current
        WHERE recording_id = ?;
        """,
        ("rec_a_uuid",),
    ).fetchone()
    assert rec_latest is not None
    assert str(rec_latest["dataset_id"]) == dataset_id
    assert str(rec_latest["crop_run"]) == "crop_new"
    assert str(rec_latest["review_state"]) == "approved"
    assert str(rec_latest["roi_pixel_contract_name"]) == "decord_rgb_channel_mean_uint8"
    registry.close()


def test_register_from_root_handles_archive_without_crop_runs(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    zarr_path = tmp_path / "recordings" / "rec_no_crop" / "zarr" / "rec_no_crop_analysis.zarr"
    _create_cropless_archive(zarr_path, session_uuid="rec_no_crop_uuid")

    dataset_id = registry.register_from_root(zarr.open_group(str(zarr_path), mode="r"), zarr_path)
    count = registry.conn.execute(
        "SELECT COUNT(*) AS n FROM crop_quality WHERE dataset_id = ?;",
        (dataset_id,),
    ).fetchone()
    assert count is not None
    assert int(count["n"]) == 0
    registry.close()


def test_refresh_crop_quality_from_root_populates_pixel_contract(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    zarr_path = tmp_path / "recordings" / "rec_refresh" / "zarr" / "rec_refresh_analysis.zarr"
    _create_crop_archive(zarr_path, session_uuid="rec_refresh_uuid")

    dataset_id, row_count = registry.refresh_crop_quality_from_root(
        zarr.open_group(str(zarr_path), mode="r"),
        zarr_path,
    )

    assert row_count == 2
    latest = registry.conn.execute(
        """
        SELECT crop_run, roi_pixel_contract_name
        FROM crop_quality_current
        WHERE dataset_id = ?;
        """,
        (dataset_id,),
    ).fetchone()
    assert latest is not None
    assert str(latest["crop_run"]) == "crop_new"
    assert str(latest["roi_pixel_contract_name"]) == "decord_rgb_channel_mean_uint8"
    registry.close()


def test_crop_step_status_refreshes_crop_quality_pixel_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from fisheye.tracking import crop as crop_module

    registry_path = tmp_path / "registry.sqlite"
    zarr_path = tmp_path / "recordings" / "rec_inline" / "zarr" / "rec_inline_analysis.zarr"
    _create_crop_archive(zarr_path, session_uuid="rec_inline_uuid")
    monkeypatch.setenv("PALETTE_REGISTRY_PATH", str(registry_path))

    captured_details = {}

    def fake_emit_stage_completion(*args, **kwargs):
        captured_details.update(dict(kwargs["details_json"]))
        return True

    monkeypatch.setattr(crop_module, "emit_stage_completion", fake_emit_stage_completion)
    crop_module._emit_crop_step_status(
        root=zarr.open_group(str(zarr_path), mode="r"),
        zarr_path=str(zarr_path),
        status="ok",
        run_name="crop_new",
        method="pytest",
        coverage_pct=100.0,
        review_status=None,
        details={"reason": "present"},
        console=None,
    )

    registry = Registry(registry_path)
    latest = registry.conn.execute(
        """
        SELECT crop_run, roi_pixel_contract_name
        FROM crop_quality_current
        WHERE recording_id = ?;
        """,
        ("rec_inline_uuid",),
    ).fetchone()
    assert latest is not None
    assert str(latest["crop_run"]) == "crop_new"
    assert str(latest["roi_pixel_contract_name"]) == "decord_rgb_channel_mean_uint8"
    assert captured_details["crop_quality_refresh_status"] == "ok"
    assert captured_details["crop_quality_refresh_run"] == "crop_new"
    assert captured_details["crop_quality_refresh_run_present"] is True
    registry.close()
