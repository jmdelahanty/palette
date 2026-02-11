from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.registry.db import Registry
from fisheye.utils import check_recording_steps as mod


def test_check_zarr_reports_detect_coverage_full_basis(tmp_path: Path) -> None:
    zarr_path = tmp_path / "full_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    raw = root.create_group("raw_video")
    raw.create_array("images_full", data=np.zeros((4,), dtype=np.uint8))
    detect_parent = root.create_group("detect_runs")
    detect_parent.attrs["latest"] = "detect_full"
    run = detect_parent.create_group("detect_full")
    run.create_array("frame_counts", data=np.array([1, 0, 2, 1], dtype=np.int32))

    info = mod._check_zarr(zarr_path, tuning_keys=[])  # noqa: SLF001

    assert info["detect_present"] is True
    assert info["detect_method"] is None
    assert info["detect_coverage_basis"] == "full"
    assert info["detect_coverage"] == pytest.approx(75.0)


def test_check_zarr_reports_detect_coverage_sampled_basis(tmp_path: Path) -> None:
    zarr_path = tmp_path / "sampled_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    raw = root.create_group("raw_video")
    raw.create_array("original_frame_indices", data=np.array([0, 2, 4], dtype=np.int32))
    detect_parent = root.create_group("detect_runs")
    detect_parent.attrs["latest"] = "detect_sampled"
    run = detect_parent.create_group("detect_sampled")
    run.create_array("frame_counts", data=np.array([1, 0, 1], dtype=np.int32))

    info = mod._check_zarr(zarr_path, tuning_keys=[])  # noqa: SLF001

    assert info["detect_present"] is True
    assert info["detect_coverage_basis"] == "sampled"
    assert info["detect_coverage"] == pytest.approx(66.6667, abs=1e-3)


def test_check_zarr_reports_detect_coverage_inferred_basis(tmp_path: Path) -> None:
    zarr_path = tmp_path / "inferred_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    detect_parent = root.create_group("detect_runs")
    detect_parent.attrs["latest"] = "detect_inferred"
    run = detect_parent.create_group("detect_inferred")
    run.create_array("frame_indices", data=np.array([0, 0, 2, 5], dtype=np.int32))

    info = mod._check_zarr(zarr_path, tuning_keys=[])  # noqa: SLF001

    assert info["detect_present"] is True
    assert info["detect_coverage_basis"] == "inferred"
    assert info["detect_coverage"] == pytest.approx(50.0)


def test_check_zarr_reports_detect_missing_when_no_detect_runs(tmp_path: Path) -> None:
    zarr_path = tmp_path / "empty_analysis.zarr"
    zarr.open_group(str(zarr_path), mode="w")

    info = mod._check_zarr(zarr_path, tuning_keys=[])  # noqa: SLF001

    assert info["detect_present"] is False
    assert info["detect_method"] is None
    assert info["detect_coverage"] is None
    assert info["detect_coverage_basis"] is None
    assert info["detect_quality_present"] is False
    assert info["detect_quality_run"] is None


def test_check_zarr_reads_detect_method_from_provenance(tmp_path: Path) -> None:
    zarr_path = tmp_path / "method_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    detect_parent = root.create_group("detect_runs")
    detect_parent.attrs["latest"] = "detect_prov"
    run = detect_parent.create_group("detect_prov")
    run.attrs["provenance"] = {"method": "yolo"}
    run.create_array("frame_counts", data=np.array([1, 1], dtype=np.int32))

    info = mod._check_zarr(zarr_path, tuning_keys=[])  # noqa: SLF001

    assert info["detect_present"] is True
    assert info["detect_method"] == "yolo"


def test_check_zarr_reads_detect_quality_summary(tmp_path: Path) -> None:
    zarr_path = tmp_path / "quality_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    detect_parent = root.create_group("detect_runs")
    detect_parent.attrs["latest"] = "detect_quality_source"
    run = detect_parent.create_group("detect_quality_source")
    run.create_array("frame_counts", data=np.array([1, 1, 0], dtype=np.int32))
    quality_parent = run.create_group("quality_reports")
    quality_parent.attrs["latest"] = "detect_quality_2026-02-09_12-00-00"
    quality = quality_parent.create_group("detect_quality_2026-02-09_12-00-00")
    quality.attrs["quality_score"] = {"grade": "A", "overall_score": 99.2}
    quality.attrs["detection_quality_summary"] = {
        "clean_percentage": 98.7,
        "blip_detections": 3,
        "jump_detections": 2,
        "multi_detections": 1,
    }

    info = mod._check_zarr(zarr_path, tuning_keys=[])  # noqa: SLF001

    assert info["detect_quality_present"] is True
    assert info["detect_quality_run"] == "detect_quality_2026-02-09_12-00-00"
    assert info["detect_quality_grade"] == "A"
    assert info["detect_quality_score"] == pytest.approx(99.2)
    assert info["detect_quality_clean_percent"] == pytest.approx(98.7)
    assert info["detect_quality_artifacts"] == 6


def test_check_zarr_reads_crop_status_from_latest_run(tmp_path: Path) -> None:
    zarr_path = tmp_path / "crop_status_latest_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "crop_2026-02-10_12-00-00"
    crop_run = crop_parent.create_group("crop_2026-02-10_12-00-00")
    crop_run.attrs["status"] = "completed"

    info = mod._check_zarr(zarr_path, tuning_keys=[])  # noqa: SLF001

    assert info["crop_present"] is True
    assert info["crop_status"] == "completed"


def test_check_zarr_reads_crop_status_from_fallback_latest_name(tmp_path: Path) -> None:
    zarr_path = tmp_path / "crop_status_fallback_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    crop_parent = root.create_group("crop_runs")
    crop_parent.create_group("crop_2026-02-10_11-00-00").attrs["status"] = "failed"
    crop_parent.create_group("crop_2026-02-10_12-00-00").attrs["status"] = "running"

    info = mod._check_zarr(zarr_path, tuning_keys=[])  # noqa: SLF001

    assert info["crop_present"] is True
    assert info["crop_status"] == "running"


def test_crop_status_format_helpers() -> None:
    assert mod._crop_status_text(False, None) == "MISS"  # noqa: SLF001
    assert mod._crop_status_text(True, None) == "OK"  # noqa: SLF001
    assert mod._crop_status_text(True, "FAILED") == "failed"  # noqa: SLF001
    assert mod._crop_status_rich(True, "completed") == "[chartreuse1]completed[/chartreuse1]"  # noqa: SLF001
    assert mod._crop_status_rich(True, "running") == "[yellow]running[/yellow]"  # noqa: SLF001
    assert mod._crop_status_rich(True, "failed") == "[red]failed[/red]"  # noqa: SLF001


def test_registry_crop_review_status_for_zarr_returns_latest_review_fields(tmp_path: Path) -> None:
    zarr_path = tmp_path / "a_analysis.zarr"
    zarr.open_group(str(zarr_path), mode="w")

    registry_path = tmp_path / "registry.sqlite"
    registry = Registry(registry_path)
    registry.upsert_dataset(
        "dataset_a",
        session_uuid="session_a",
        zarr_path=zarr_path,
        recording_id="recording_a",
        artifact_kind="source_recording",
        zarr_use="analysis",
    )
    registry.upsert_provenance(
        "dataset_a",
        provenance={},
        context={},
        protocol_name=None,
        protocol_hash=None,
        acquisition={},
        zarr_purpose="analysis",
    )
    registry.replace_crop_quality(
        "dataset_a",
        [
            {
                "crop_run": "crop_2026-02-10_10-00-00",
                "recording_id": "recording_a",
                "zarr_use": "analysis",
                "crop_created_utc": "2026-02-10T10:00:00+00:00",
                "source_detect_run": None,
                "source_refined_run": None,
                "detection_source_type": "manual",
                "detection_source_path": None,
                "total_rois": 100,
                "frames_with_crops": 90,
                "total_frames": 100,
                "percent_frames_with_crops": 90.0,
                "includes_interpolated": 0,
                "n_real_detections": 100,
                "n_interpolated_detections": 0,
                "review_state": "approved",
                "review_method": "manual",
                "review_intended_use": "training",
                "review_reviewer": "alice",
                "review_timestamp_utc": "2026-02-10T10:30:00+00:00",
                "review_notes": "looks good",
                "zarr_mtime_ns": 12345,
                "updated_utc": "2026-02-10T10:30:00+00:00",
            }
        ],
    )

    payload = mod._registry_crop_review_status_for_zarr(  # noqa: SLF001
        registry=registry,
        zarr_path=zarr_path,
    )
    registry.close()

    assert payload is not None
    assert payload["crop_run"] == "crop_2026-02-10_10-00-00"
    status = payload["crop_review_status"]
    assert isinstance(status, dict)
    assert status["state"] == "approved"
    assert status["method"] == "manual"
    assert status["intended_use"] == "training"
    assert status["reviewer"] == "alice"
    assert status["timestamp_utc"] == "2026-02-10T10:30:00+00:00"
