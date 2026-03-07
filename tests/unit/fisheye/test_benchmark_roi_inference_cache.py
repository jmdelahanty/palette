from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from fisheye.diagnostics import benchmark_roi_inference_cache as mod


def _make_benchmark_fixture(tmp_path: Path) -> Path:
    zarr_path = tmp_path / "recording_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")

    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "crop_materialized"
    crop_parent.attrs["latest_materialized"] = "crop_materialized"
    crop_parent.attrs["latest_any"] = "crop_geometry"

    crop_materialized = crop_parent.create_group("crop_materialized")
    crop_materialized.attrs["crop_storage_mode"] = "materialized"
    crop_materialized.create_array(
        "roi_images",
        data=np.zeros((2, 8, 8), dtype=np.uint8),
        overwrite=True,
    )

    crop_geometry = crop_parent.create_group("crop_geometry")
    crop_geometry.attrs["crop_storage_mode"] = "geometry_only"
    crop_geometry.create_array(
        "roi_coordinates_full",
        data=np.array([[0, 0], [1, 1]], dtype=np.int32),
        overwrite=True,
    )
    crop_geometry.create_array(
        "frame_indices",
        data=np.array([0, 1], dtype=np.int32),
        overwrite=True,
    )
    crop_geometry.attrs["roi_size"] = [8, 8]

    return zarr_path


def test_resolve_crop_runs_picks_materialized_and_geometry_defaults(tmp_path: Path) -> None:
    zarr_path = _make_benchmark_fixture(tmp_path)

    result = mod._resolve_crop_runs(
        zarr_path=zarr_path,
        materialized_crop_run=None,
        geometry_crop_run=None,
    )

    assert result.materialized_run == "crop_materialized"
    assert result.geometry_run == "crop_geometry"


def test_scenario_specs_share_geometry_cache_dir_between_build_and_reuse(tmp_path: Path) -> None:
    crop_runs = mod.CropRunSelection(
        materialized_run="crop_materialized",
        geometry_run="crop_geometry",
    )

    specs = mod._scenario_specs(
        crop_runs=crop_runs,
        scenario_names=[
            mod.SCENARIO_MATERIALIZED,
            mod.SCENARIO_GEOMETRY_CACHE_BUILD,
            mod.SCENARIO_GEOMETRY_CACHE_REUSE,
        ],
        cache_root=tmp_path / "cache-root",
    )

    assert [spec.name for spec in specs] == [
        mod.SCENARIO_MATERIALIZED,
        mod.SCENARIO_GEOMETRY_CACHE_BUILD,
        mod.SCENARIO_GEOMETRY_CACHE_REUSE,
    ]
    assert specs[0].roi_cache_dir is None
    assert specs[1].roi_cache_dir == specs[2].roi_cache_dir
    assert specs[1].roi_cache_policy == "always"
    assert specs[2].roi_cache_policy == "always"


def test_build_keypoint_and_eye_commands_include_cache_settings(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    cache_dir = tmp_path / "cache"

    key_cmd = mod._build_keypoint_command(
        zarr_path=zarr_path,
        model_path=Path("/tmp/pose.pt"),
        crop_run="crop_geometry",
        run_name="kp_run",
        roi_cache_policy="always",
        roi_cache_dir=cache_dir,
        batch_size=128,
        device="cuda:0",
        imgsz=512,
    )
    eye_cmd = mod._build_eye_mask_command(
        zarr_path=zarr_path,
        checkpoint_path=Path("/tmp/eye.pt"),
        crop_run="crop_geometry",
        keypoints_run="kp_run",
        run_name="eye_run",
        roi_cache_policy="always",
        roi_cache_dir=cache_dir,
        batch_size=64,
        device="cuda:0",
        write_binary_masks=True,
    )

    assert "--roi-cache-policy" in key_cmd
    assert "--roi-cache-dir" in key_cmd
    assert "--device" in key_cmd
    assert "--imgsz" in key_cmd
    assert "kp_run" in key_cmd

    assert "--roi-cache-policy" in eye_cmd
    assert "--roi-cache-dir" in eye_cmd
    assert "--device" in eye_cmd
    assert "--write-binary-masks" in eye_cmd
    assert "kp_run" in eye_cmd


def test_stage_metrics_reports_cache_usage_fields() -> None:
    metrics = mod._stage_metrics(
        stage="keypoints",
        attrs={
            "total_rois": 100,
            "inference_duration_seconds": 4.0,
            "inference_poses_per_second": 25.0,
            "source_crop_run": "crop_geometry",
            "source_crop_storage_mode": "geometry_only",
            "source_crop_signature": "sig-001",
            "source_roi_read_mode": "temporary_cache",
            "source_roi_cache_used": True,
            "source_roi_cache_path": "/tmp/cache.zarr",
            "source_roi_cache_key": "abc123",
            "roi_cache_policy": "always",
        },
        wall_seconds=5.0,
    )

    assert metrics["stage"] == "keypoints"
    assert metrics["wall_seconds"] == 5.0
    assert metrics["recorded_duration_seconds"] == 4.0
    assert metrics["throughput_per_second"] == 25.0
    assert metrics["source_roi_read_mode"] == "temporary_cache"
    assert metrics["source_roi_cache_used"] is True
