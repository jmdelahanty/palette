from __future__ import annotations

import argparse
from pathlib import Path

import zarr

from fisheye.utils.detect_quality_batch import _build_cmd, _build_plans


def _make_zarr(
    recordings_root: Path,
    recording_name: str,
    *,
    detect_run: str | None,
    with_quality: bool,
) -> Path:
    zarr_path = recordings_root / recording_name / "zarr" / f"{recording_name}_analysis.zarr"
    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(zarr_path), mode="w")
    if detect_run is not None:
        detect_parent = root.create_group("detect_runs")
        detect_group = detect_parent.create_group(detect_run)
        detect_parent.attrs["latest"] = detect_run
        if with_quality:
            quality_parent = detect_group.create_group("quality_reports")
            quality_parent.create_group("detect_quality_2026-02-09_12-00-00")
            quality_parent.attrs["latest"] = "detect_quality_2026-02-09_12-00-00"
    return zarr_path


def test_build_plans_missing_detect_runs(tmp_path: Path) -> None:
    recordings = tmp_path / "recordings"
    zarr_path = _make_zarr(recordings, "rec_missing", detect_run=None, with_quality=False)
    plans = _build_plans([recordings], recursive=True, detect_run=None, skip_existing=True)
    assert len(plans) == 1
    assert plans[0].zarr_path == zarr_path
    assert plans[0].status == "missing"
    assert plans[0].reason == "detect_runs missing"


def test_build_plans_skips_existing_quality_by_default(tmp_path: Path) -> None:
    recordings = tmp_path / "recordings"
    zarr_path = _make_zarr(recordings, "rec_quality", detect_run="detect_2026_01", with_quality=True)
    plans = _build_plans([recordings], recursive=True, detect_run=None, skip_existing=True)
    assert len(plans) == 1
    plan = plans[0]
    assert plan.zarr_path == zarr_path
    assert plan.status == "skipped"
    assert plan.detect_run == "detect_2026_01"
    assert plan.quality_present is True


def test_build_plans_allows_existing_quality_with_no_skip(tmp_path: Path) -> None:
    recordings = tmp_path / "recordings"
    zarr_path = _make_zarr(recordings, "rec_quality", detect_run="detect_2026_01", with_quality=True)
    plans = _build_plans([recordings], recursive=True, detect_run=None, skip_existing=False)
    assert len(plans) == 1
    plan = plans[0]
    assert plan.zarr_path == zarr_path
    assert plan.status == "ok"
    assert plan.detect_run == "detect_2026_01"


def test_build_plans_detect_run_not_found(tmp_path: Path) -> None:
    recordings = tmp_path / "recordings"
    _make_zarr(recordings, "rec_a", detect_run="detect_2026_01", with_quality=False)
    plans = _build_plans([recordings], recursive=True, detect_run="detect_missing", skip_existing=True)
    assert len(plans) == 1
    assert plans[0].status == "missing"
    assert plans[0].reason == "detect_run not found"


def test_build_cmd_includes_requested_options(tmp_path: Path) -> None:
    args = argparse.Namespace(
        threshold=123.5,
        threshold_mode="scaled",
        threshold_reference_width=640.0,
        no_save=False,
    )
    zarr_path = tmp_path / "rec.zarr"
    cmd = _build_cmd(args, zarr_path, detect_run="detect_2026_01")
    assert cmd[1:3] == ["-m", "fisheye.refinement.detect_quality"]
    assert str(zarr_path) in cmd
    assert "--threshold" in cmd
    assert "123.5" in cmd
    assert "--threshold-mode" in cmd
    assert "scaled" in cmd
    assert "--threshold-reference-width" in cmd
    assert "640.0" in cmd
    assert "--run" in cmd
    assert "detect_2026_01" in cmd
    assert "--save" in cmd


def test_build_cmd_respects_no_save(tmp_path: Path) -> None:
    args = argparse.Namespace(
        threshold=100.0,
        threshold_mode="scaled",
        threshold_reference_width=640.0,
        no_save=True,
    )
    zarr_path = tmp_path / "rec.zarr"
    cmd = _build_cmd(args, zarr_path, detect_run=None)
    assert "--no-save" in cmd
    assert "--save" not in cmd
