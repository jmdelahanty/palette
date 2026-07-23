from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
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


def _seed_donor_calibration(zarr_path: Path, *, with_scale: bool = True) -> None:
    root = zarr.open_group(str(zarr_path), mode="a")
    root.attrs["zarr_purpose"] = "analysis"
    root.attrs["recording_id"] = "donor_recording"
    calib = root.require_group("analysis").require_group("calibration")
    calib.attrs["schema_version"] = 1
    calib.attrs["source"] = "h5_calibration_snapshot"
    calib.attrs["source_h5"] = "/example/donor.h5"
    calib.attrs["source_stimulus_run"] = "stimulus_donor"
    calib.attrs["primary_camera_id"] = "2010095"
    if with_scale:
        calib.attrs["pixels_per_mm_camera"] = 53.4031982421875
        calib.attrs["pixel_to_mm"] = 0.018725470251143482
    calib.create_array("homography_matrix", data=HOMOGRAPHY, chunks=(3, 3), overwrite=True)


def _zarr_metadata_snapshot(zarr_path: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(zarr_path)): path.read_bytes()
        for path in sorted(zarr_path.rglob("zarr.json"))
    }


def test_plan_or_backfill_one_reports_h5_candidate_without_mutation(tmp_path: Path) -> None:
    h5_path = tmp_path / "session.h5"
    zarr_path = tmp_path / "sample_analysis.zarr"
    _write_stimulus_h5(h5_path)
    _seed_analysis_zarr(zarr_path, source_h5=h5_path)

    dry = mod.plan_or_backfill_one(  # noqa: SLF001
        zarr_path,
        apply=False,
        overwrite_existing=False,
    )
    assert dry.status == mod.H5_CANDIDATE_STATUS
    assert dry.h5_path == h5_path
    assert dry.run_name == "stimulus_001"

    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    assert "calibration" not in root["analysis"]


@pytest.mark.parametrize("with_donor", [False, True])
def test_apply_rejected_before_any_zarr_open_and_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    with_donor: bool,
) -> None:
    h5_path = tmp_path / "session.h5"
    zarr_path = tmp_path / "sample_analysis.zarr"
    donor_zarr = tmp_path / "donor_analysis.zarr"
    _write_stimulus_h5(h5_path)
    _seed_analysis_zarr(zarr_path, source_h5=h5_path)
    if with_donor:
        _seed_donor_calibration(donor_zarr)
    before = _zarr_metadata_snapshot(zarr_path)

    def _unexpected_open(*_args, **_kwargs):
        raise AssertionError("retired apply path opened a Zarr archive")

    monkeypatch.setattr(mod, "_open_zarr", _unexpected_open)
    with pytest.raises(
        mod.AnalysisCalibrationApplyRetiredError,
        match="Legacy global analysis/calibration writes are retired",
    ):
        mod.plan_or_backfill_one(
            zarr_path,
            apply=True,
            overwrite_existing=False,
            donor_zarr_path=donor_zarr if with_donor else None,
        )

    assert _zarr_metadata_snapshot(zarr_path) == before


def test_plan_or_backfill_one_skips_complete_calibration_without_overwrite(tmp_path: Path) -> None:
    h5_path = tmp_path / "session.h5"
    zarr_path = tmp_path / "sample_analysis.zarr"
    _write_stimulus_h5(h5_path)
    _seed_analysis_zarr(zarr_path, source_h5=h5_path)
    _seed_donor_calibration(zarr_path)

    assert mod.plan_or_backfill_one(zarr_path, apply=False, overwrite_existing=False).status == "skipped_existing"
    assert (
        mod.plan_or_backfill_one(
            zarr_path,
            apply=False,
            overwrite_existing=True,
        ).status
        == mod.H5_OVERWRITE_CANDIDATE_STATUS
    )


def test_plan_or_backfill_one_reports_donor_candidate_without_copying(tmp_path: Path) -> None:
    donor_zarr = tmp_path / "donor_analysis.zarr"
    target_zarr = tmp_path / "target_analysis.zarr"
    _seed_donor_calibration(donor_zarr)
    _seed_analysis_zarr(target_zarr, source_h5=None)

    dry = mod.plan_or_backfill_one(
        target_zarr,
        apply=False,
        overwrite_existing=False,
        donor_zarr_path=donor_zarr,
        donor_note="same camera/rig configuration verified",
    )
    assert dry.status == mod.DONOR_CANDIDATE_STATUS
    assert dry.donor_zarr_path == donor_zarr
    assert "source_h5=/example/donor.h5" in dry.message

    root = zarr.open_group(str(target_zarr), mode="r", use_consolidated=False)
    assert "calibration" not in root["analysis"]


def test_plan_or_backfill_one_rejects_donor_without_usable_scale(tmp_path: Path) -> None:
    donor_zarr = tmp_path / "donor_analysis.zarr"
    target_zarr = tmp_path / "target_analysis.zarr"
    _seed_donor_calibration(donor_zarr, with_scale=False)
    _seed_analysis_zarr(target_zarr, source_h5=None)

    result = mod.plan_or_backfill_one(
        target_zarr,
        apply=False,
        overwrite_existing=False,
        donor_zarr_path=donor_zarr,
    )
    assert result.status == "donor_calibration_missing_scale"


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
    assert result.status == mod.H5_CANDIDATE_STATUS
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
    assert result.status == mod.H5_CANDIDATE_STATUS
    assert result.h5_path == h5_path


def test_main_dry_run_reports_candidate(tmp_path: Path, capsys) -> None:
    h5_path = tmp_path / "session.h5"
    zarr_path = tmp_path / "sample_analysis.zarr"
    _write_stimulus_h5(h5_path)
    _seed_analysis_zarr(zarr_path, source_h5=h5_path)

    rc = mod.main([str(zarr_path)])
    assert rc == 0

    out = capsys.readouterr().out
    assert mod.H5_CANDIDATE_STATUS in out
    assert f"{mod.H5_CANDIDATE_STATUS}=1" in out


def test_main_apply_fails_before_inventory(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def _unexpected_inventory(*_args, **_kwargs):
        raise AssertionError("retired apply path started archive discovery")

    monkeypatch.setattr(mod, "_iter_zarr", _unexpected_inventory)
    rc = mod.main(["--apply", "/does/not/need/to/exist"])

    assert rc == 2
    captured = capsys.readouterr()
    assert "Legacy global analysis/calibration writes are retired" in captured.err


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
    assert training_result.status == mod.H5_CANDIDATE_STATUS

    any_result = mod.plan_or_backfill_one(zarr_path, apply=False, overwrite_existing=False, zarr_use="any")
    assert any_result.status == mod.H5_CANDIDATE_STATUS
