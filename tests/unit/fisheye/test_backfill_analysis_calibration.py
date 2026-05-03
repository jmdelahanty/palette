from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import zarr

from fisheye.utils import backfill_analysis_calibration as mod


HOMOGRAPHY = np.array(
    [
        [1.0, 0.0, 10.0],
        [0.0, 1.0, 20.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def _write_stimulus_h5(path: Path, *, camera_id: str = "2010094") -> None:
    arena_config = {
        "active_camera_id": camera_id,
        "calculated_z_eff_mm": 0.0,
        "camera_calibrations": [
            {
                "camera_id": camera_id,
                "native_width_px": 4512,
                "native_height_px": 4512,
                "pixels_per_mm_camera": 50.0,
                "pixels_per_mm_projector": 5.0,
                "real_world_ref_mm": 10.0,
            }
        ],
        "experimental_area_shape": "CIRCLE",
        "experimental_area_center_x_px": 172.0,
        "experimental_area_center_y_px": 172.0,
        "experimental_area_radius_px": 166.0,
        "experimental_area_radius_mm": 40.0,
    }
    homography_yml = """%YAML:1.0
---
homography_matrix:
  rows: 3
  cols: 3
  dt: d
  data: [1, 0, 10, 0, 1, 20, 0, 0, 1]
"""
    with h5py.File(path, "w") as h5:
        calib = h5.create_group("calibration_snapshot")
        calib.create_dataset("arena_config_json", data=json.dumps(arena_config).encode("utf-8"))
        cam = calib.create_group(camera_id)
        cam.attrs["pixels_per_mm_camera"] = 50.0
        cam.attrs["pixels_per_mm_projector"] = 5.0
        cam.create_dataset("homography_matrix_yml", data=homography_yml.encode("utf-8"))


def _seed_analysis_zarr(zarr_path: Path, *, source_h5: Path | None, zarr_purpose: str = "analysis") -> None:
    root = zarr.open_group(str(zarr_path), mode="w")
    root.attrs["zarr_purpose"] = zarr_purpose
    analysis = root.create_group("analysis")
    runs_parent = analysis.create_group("stimulus_runs")
    run = runs_parent.create_group("stimulus_001")
    if source_h5 is not None:
        run.attrs["source_h5"] = str(source_h5)
    runs_parent.attrs["latest"] = "stimulus_001"


def test_plan_or_backfill_one_writes_analysis_calibration_from_stimulus_source_h5(tmp_path: Path) -> None:
    h5_path = tmp_path / "session.h5"
    zarr_path = tmp_path / "sample_analysis.zarr"
    _write_stimulus_h5(h5_path)
    _seed_analysis_zarr(zarr_path, source_h5=h5_path)

    dry = mod.plan_or_backfill_one(  # noqa: SLF001
        zarr_path,
        apply=False,
        overwrite_existing=False,
    )
    assert dry.status == "would_backfill"
    assert dry.h5_path == h5_path
    assert dry.run_name == "stimulus_001"

    result = mod.plan_or_backfill_one(  # noqa: SLF001
        zarr_path,
        apply=True,
        overwrite_existing=False,
    )
    assert result.status == "backfilled"

    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    calib = root["analysis"]["calibration"]
    assert calib.attrs["active_camera_id"] == "2010094"
    assert calib.attrs["source_h5"] == str(h5_path)
    assert calib.attrs["source_stimulus_run"] == "stimulus_001"
    assert calib.attrs["pixel_to_mm"] == 1.0 / 50.0
    np.testing.assert_allclose(calib["homography_matrix"][:], HOMOGRAPHY)


def test_plan_or_backfill_one_skips_complete_calibration_without_overwrite(tmp_path: Path) -> None:
    h5_path = tmp_path / "session.h5"
    zarr_path = tmp_path / "sample_analysis.zarr"
    _write_stimulus_h5(h5_path)
    _seed_analysis_zarr(zarr_path, source_h5=h5_path)

    assert mod.plan_or_backfill_one(zarr_path, apply=True, overwrite_existing=False).status == "backfilled"
    assert mod.plan_or_backfill_one(zarr_path, apply=False, overwrite_existing=False).status == "skipped_existing"
    assert mod.plan_or_backfill_one(zarr_path, apply=False, overwrite_existing=True).status == "would_overwrite"


def test_source_h5_falls_back_to_recording_raw_directory(tmp_path: Path) -> None:
    recording_dir = tmp_path / "2026-01-01T00-00-00Z_arena_1"
    raw_dir = recording_dir / "raw"
    zarr_dir = recording_dir / "zarr"
    raw_dir.mkdir(parents=True)
    zarr_dir.mkdir()
    h5_path = raw_dir / "2026-01-01T00-00-00Z_arena_1.h5"
    zarr_path = zarr_dir / "2026-01-01T00-00-00Z_arena_1_analysis.zarr"

    _write_stimulus_h5(h5_path)
    _seed_analysis_zarr(zarr_path, source_h5=None)

    result = mod.plan_or_backfill_one(zarr_path, apply=False, overwrite_existing=False)
    assert result.status == "would_backfill"
    assert result.h5_path == h5_path


def test_training_source_h5_falls_back_to_recording_raw_directory(tmp_path: Path) -> None:
    recording_dir = tmp_path / "2026-01-01T00-00-00Z_arena_1"
    raw_dir = recording_dir / "raw"
    zarr_dir = recording_dir / "zarr"
    raw_dir.mkdir(parents=True)
    zarr_dir.mkdir()
    h5_path = raw_dir / "2026-01-01T00-00-00Z_arena_1.h5"
    zarr_path = zarr_dir / "2026-01-01T00-00-00Z_arena_1_training.zarr"

    _write_stimulus_h5(h5_path)
    _seed_analysis_zarr(zarr_path, source_h5=None, zarr_purpose="training")

    result = mod.plan_or_backfill_one(
        zarr_path,
        apply=False,
        overwrite_existing=False,
        zarr_use="training",
    )
    assert result.status == "would_backfill"
    assert result.h5_path == h5_path


def test_main_dry_run_reports_candidate(tmp_path: Path, capsys) -> None:
    h5_path = tmp_path / "session.h5"
    zarr_path = tmp_path / "sample_analysis.zarr"
    _write_stimulus_h5(h5_path)
    _seed_analysis_zarr(zarr_path, source_h5=h5_path)

    rc = mod.main([str(zarr_path)])
    assert rc == 0

    out = capsys.readouterr().out
    assert "would_backfill" in out
    assert "would_backfill=1" in out


def test_training_archive_is_filtered_by_default_but_processed_when_requested(tmp_path: Path) -> None:
    h5_path = tmp_path / "session.h5"
    zarr_path = tmp_path / "sample_training.zarr"
    _write_stimulus_h5(h5_path)
    _seed_analysis_zarr(zarr_path, source_h5=h5_path, zarr_purpose="training")

    default_result = mod.plan_or_backfill_one(zarr_path, apply=False, overwrite_existing=False)
    assert default_result.status == "filtered_zarr_use"

    training_result = mod.plan_or_backfill_one(
        zarr_path,
        apply=False,
        overwrite_existing=False,
        zarr_use="training",
    )
    assert training_result.status == "would_backfill"

    any_result = mod.plan_or_backfill_one(zarr_path, apply=False, overwrite_existing=False, zarr_use="any")
    assert any_result.status == "would_backfill"
