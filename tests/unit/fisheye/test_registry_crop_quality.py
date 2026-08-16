"""Unit tests for crop review quality registry extraction and views."""

import json
from pathlib import Path
import sys

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.registry.db import Registry
from fisheye.registry.extractors.crop import _crop_run_names, _get_group
from fisheye.shared.roi_pixel_contract import crop_run_pixel_contract
from fisheye.shared.zarr_run_completion import mark_run_complete
from fisheye.utils.finalize_crop_flat_roi_cache_batch_registry import (
    finalize_crop_flat_roi_cache_batch_registry,
)


class _StaleMetadataParent:
    def __init__(self, *, store, path: str) -> None:
        self.store = store
        self.path = path

    def group_keys(self):
        return []

    def keys(self):
        return []

    def get(self, _key):
        return None

    def __getitem__(self, _key):
        raise KeyError(_key)


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


def _create_hybrid_provider_archive(path: Path, *, session_uuid: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(path), mode="w")
    root.attrs["session_uuid"] = session_uuid
    run = root.create_group("crop_runs").create_group("crop_hybrid_v3")
    run.attrs.update(
        {
            "schema_id": "palette.hybrid_acquisition_offline_crop_run.v3",
            "created_at_utc": "2026-08-15T12:00:00+00:00",
            "source_pixels": "hybrid_acquisition_crop_video_offline_supplement",
            "source_pixel_routing_policy": "goodbatbadbat_crop_pixel_routing_v1",
            "crop_policy_id": "zebrafish_crop_384_v1",
            "source_acquisition_mode": "canonical_ledger",
            "source_acquisition_crop_ledger_record_sha256": "a" * 64,
            "provider_record_sha256": "b" * 64,
            "source_refined_detect_run": "refined_v1",
            "summary_statistics": {
                "total_frames": 4,
                "reviewed_refined_rows": 4,
                "acquisition_video_rows_reused": 3,
                "supplemental_rows_materialized": 1,
            },
        }
    )
    run.create_array("bbox_norm_coords", data=np.zeros((4, 4), dtype=np.float32))
    run.create_array("frame_counts", data=np.ones(4, dtype=np.int32))


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


def test_crop_run_names_falls_back_to_filesystem_when_metadata_is_stale(tmp_path: Path) -> None:
    zarr_path = tmp_path / "stale_crop_metadata.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    root.create_group("crop_runs/crop_hidden")

    parent = _StaleMetadataParent(store=root.store, path="crop_runs")

    assert _crop_run_names(parent) == ["crop_hidden"]
    assert getattr(_get_group(parent, "crop_hidden"), "path") == "crop_runs/crop_hidden"


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


def test_registry_exposes_hybrid_crop_pixel_routing_readiness(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    zarr_path = (
        tmp_path / "recordings" / "rec_hybrid" / "zarr" / "rec_hybrid_analysis.zarr"
    )
    _create_hybrid_provider_archive(zarr_path, session_uuid="rec_hybrid_uuid")

    dataset_id = registry.register_from_root(
        zarr.open_group(str(zarr_path), mode="r"), zarr_path
    )
    row = registry.conn.execute(
        """
        SELECT crop_schema_id, source_pixels, provider_record_sha256,
               routing_policy_id, crop_policy_id, source_acquisition_mode,
               source_acquisition_ledger_record_sha256,
               acquisition_video_rows, full_frame_recovery_rows,
               crop_pixel_routing_ready, source_refined_run
        FROM crop_quality_current
        WHERE dataset_id = ?;
        """,
        (dataset_id,),
    ).fetchone()

    assert row is not None
    assert row["crop_schema_id"] == "palette.hybrid_acquisition_offline_crop_run.v3"
    assert row["provider_record_sha256"] == "b" * 64
    assert row["routing_policy_id"] == "goodbatbadbat_crop_pixel_routing_v1"
    assert row["crop_policy_id"] == "zebrafish_crop_384_v1"
    assert row["source_acquisition_mode"] == "canonical_ledger"
    assert row["source_acquisition_ledger_record_sha256"] == "a" * 64
    assert int(row["acquisition_video_rows"]) == 3
    assert int(row["full_frame_recovery_rows"]) == 1
    assert int(row["crop_pixel_routing_ready"]) == 1
    assert row["source_refined_run"] == "refined_v1"
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


def test_crop_step_status_defer_env_skips_registry_write(monkeypatch, tmp_path: Path) -> None:
    from fisheye.tracking import crop as crop_module

    zarr_path = tmp_path / "recordings" / "rec_defer" / "zarr" / "rec_defer_analysis.zarr"
    _create_crop_archive(zarr_path, session_uuid="rec_defer_uuid")
    monkeypatch.setenv("PALETTE_DISABLE_REGISTRY_WRITES", "1")

    called = False

    def fake_emit_stage_completion(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("registry write should be deferred")

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

    assert called is False


def test_finalize_crop_flat_roi_cache_batch_registry_syncs_crop_status(
    monkeypatch,
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "registry.sqlite"
    zarr_path = tmp_path / "recordings" / "rec_finalizer" / "zarr" / "rec_finalizer_analysis.zarr"
    _create_crop_archive(zarr_path, session_uuid="rec_finalizer_uuid")

    root = zarr.open_group(str(zarr_path), mode="a")
    crop_parent = root["crop_runs"]
    mark_run_complete(crop_parent["crop_new"], parent_group=crop_parent, run_name="crop_new")
    monkeypatch.setattr(
        "fisheye.utils.finalize_crop_flat_roi_cache_batch_registry."
        "load_persisted_ordinary_crop_observation_geometry",
        lambda *_args, **_kwargs: object(),
    )

    run_root = tmp_path / "batch_run"
    status_dir = run_root / "per_recording" / "rec_finalizer"
    status_dir.mkdir(parents=True)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache_manifest = cache_dir / "rec_finalizer_analysis.flat_roi_cache.json"
    cache_payload = cache_dir / "rec_finalizer_analysis.flat_roi_cache.bin"
    cache_payload.write_bytes(b"cache")
    cache_manifest.write_text(
        json.dumps({"schema": "palette_roi_cache_flat_bin_v1", "layout": "flat_bin_v1"}),
        encoding="utf-8",
    )

    crop_status = {
        "status": "ok",
        "stage": "crop_materialized",
        "job_id": "crop123",
        "host": "pytest-host",
        "finished_at_utc": "2026-06-22T00:00:00+00:00",
        "zarr_path": str(zarr_path.resolve()),
        "crop_run": "crop_new",
        "crop_run_attrs": {
            "crop_storage_mode": "materialized",
            "detection_source_type": "manual",
            "detection_source_path": "refined_detect_runs/refined_001/manual",
            "total_detections": 4,
        },
    }
    (status_dir / "rec_finalizer.crop.123.json").write_text(
        json.dumps(crop_status),
        encoding="utf-8",
    )
    cache_status = {
        "status": "ok",
        "stage": "flat_roi_cache_publish",
        "finished_at_utc": "2026-06-22T00:01:00+00:00",
        "published_manifest": str(cache_manifest),
        "published_bin": str(cache_payload),
        "published_bin_size_bytes": cache_payload.stat().st_size,
        "layout": "flat_bin_v1",
        "source": {
            "archive_path": str(zarr_path.resolve()),
            "crop_run_name": "crop_new",
        },
    }
    (status_dir / "rec_finalizer.cache.456.json").write_text(
        json.dumps(cache_status),
        encoding="utf-8",
    )

    report = finalize_crop_flat_roi_cache_batch_registry(
        run_root,
        registry_path=registry_path,
        apply=True,
    )

    assert report["status"] == "ok"
    assert report["finalized_count"] == 1
    assert report["upserted_status_rows"] == 1
    registry = Registry(registry_path)
    row = registry.conn.execute(
        """
        SELECT status, run_name, source
        FROM recording_step_status
        WHERE step_name = 'crop';
        """
    ).fetchone()
    assert dict(row) == {
        "status": "ok",
        "run_name": "crop_new",
        "source": "serial_crop_flat_roi_cache_batch_finalizer",
    }
    quality = registry.conn.execute("SELECT crop_run FROM crop_quality_current;").fetchone()
    assert quality is not None
    assert str(quality["crop_run"]) == "crop_new"
    registry.close()
