from __future__ import annotations

import argparse
import json
from pathlib import Path

from fisheye.utils import run_movement_bout_batch_pipeline as mod


def _write_zarr_group(path: Path, attrs: dict | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    payload = {"zarr_format": 3, "node_type": "group", "attributes": attrs or {}}
    (path / "zarr.json").write_text(json.dumps(payload), encoding="utf-8")


def _plan(zarr_path: Path) -> mod.ArchivePlan:
    return mod.ArchivePlan(
        zarr_path=str(zarr_path),
        crop_run="crop",
        refined_keypoint_run="kp",
        refined_subject_run=None,
        track_run="tk",
        eye_angle_run="eye",
        swim_bout_run="bouts",
        bout_kinematics_run="bk",
        run_arena_assignment=False,
        run_track_kinematics=False,
        run_track_visualization=False,
        run_eye_angles=False,
        run_swim_bouts=False,
        run_bout_kinematics=False,
        include_eye_gaze=False,
    )


def _write_common_outputs(zarr_path: Path) -> None:
    _write_zarr_group(zarr_path / "analysis" / "track_kinematics_runs" / "offline" / "tk")
    _write_zarr_group(
        zarr_path
        / "analysis"
        / "bout_kinematics_runs"
        / "bk"
        / "movement"
        / "per_bout_metrics"
    )


def test_validate_plan_outputs_accepts_compact_swim_bout_run(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    _write_common_outputs(zarr_path)
    _write_zarr_group(
        zarr_path / "analysis" / "swim_bout_runs" / "bouts",
        {"layout": "compact_tabular_v2"},
    )
    _write_zarr_group(zarr_path / "analysis" / "swim_bout_runs" / "bouts" / "tables" / "bouts")

    assert mod._validate_plan_outputs(_plan(zarr_path)) == ("ok", "")


def test_validate_plan_outputs_accepts_hierarchical_swim_bout_run(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    _write_common_outputs(zarr_path)
    _write_zarr_group(
        zarr_path / "analysis" / "swim_bout_runs" / "bouts",
        {"default_level": "speed_exponential"},
    )
    _write_zarr_group(
        zarr_path
        / "analysis"
        / "swim_bout_runs"
        / "bouts"
        / "speed_exponential"
        / "bouts"
    )

    assert mod._validate_plan_outputs(_plan(zarr_path)) == ("ok", "")


def test_validate_plan_outputs_accepts_compact_bout_kinematics_run(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    _write_zarr_group(zarr_path / "analysis" / "track_kinematics_runs" / "offline" / "tk")
    _write_zarr_group(
        zarr_path / "analysis" / "swim_bout_runs" / "bouts",
        {"layout": "compact_tabular_v2"},
    )
    _write_zarr_group(zarr_path / "analysis" / "swim_bout_runs" / "bouts" / "tables" / "bouts")
    _write_zarr_group(
        zarr_path / "analysis" / "bout_kinematics_runs" / "bk",
        {"layout": "compact_tabular_v2"},
    )
    _write_zarr_group(zarr_path / "analysis" / "bout_kinematics_runs" / "bk" / "movement_metrics")

    assert mod._validate_plan_outputs(_plan(zarr_path)) == ("ok", "")


def test_validate_plan_outputs_accepts_compact_bout_kinematics_eye_gaze(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    plan = mod.ArchivePlan(**{**_plan(zarr_path).__dict__, "include_eye_gaze": True})
    _write_zarr_group(zarr_path / "analysis" / "track_kinematics_runs" / "offline" / "tk")
    _write_zarr_group(zarr_path / "analysis" / "eye_angle_runs" / "eye")
    _write_zarr_group(
        zarr_path / "analysis" / "swim_bout_runs" / "bouts",
        {"layout": "compact_tabular_v2"},
    )
    _write_zarr_group(zarr_path / "analysis" / "swim_bout_runs" / "bouts" / "tables" / "bouts")
    _write_zarr_group(
        zarr_path / "analysis" / "bout_kinematics_runs" / "bk",
        {"layout": "compact_tabular_v2"},
    )
    _write_zarr_group(zarr_path / "analysis" / "bout_kinematics_runs" / "bk" / "movement_metrics")
    _write_zarr_group(zarr_path / "analysis" / "bout_kinematics_runs" / "bk" / "eye_gaze_metrics")

    assert mod._validate_plan_outputs(plan) == ("ok", "")


def test_validate_eye_angle_only_plan_does_not_require_existing_bout_eye_gaze(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    plan = mod.ArchivePlan(
        **{
            **_plan(zarr_path).__dict__,
            "run_eye_angles": True,
            "run_bout_kinematics": False,
            "include_eye_gaze": True,
        }
    )
    _write_zarr_group(zarr_path / "analysis" / "eye_angle_runs" / "eye")
    _write_zarr_group(zarr_path / "analysis" / "track_kinematics_runs" / "offline" / "tk")
    _write_zarr_group(
        zarr_path / "analysis" / "swim_bout_runs" / "bouts",
        {"layout": "compact_tabular_v2"},
    )
    _write_zarr_group(zarr_path / "analysis" / "swim_bout_runs" / "bouts" / "tables" / "bouts")
    _write_zarr_group(
        zarr_path / "analysis" / "bout_kinematics_runs" / "bk",
        {"layout": "compact_tabular_v2"},
    )
    _write_zarr_group(zarr_path / "analysis" / "bout_kinematics_runs" / "bk" / "movement_metrics")

    assert mod._validate_plan_outputs(plan) == ("ok", "")


def test_validate_plan_outputs_reports_missing_logical_swim_bouts(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    _write_common_outputs(zarr_path)
    _write_zarr_group(
        zarr_path / "analysis" / "swim_bout_runs" / "bouts",
        {"layout": "compact_tabular_v2"},
    )

    status, detail = mod._validate_plan_outputs(_plan(zarr_path))

    assert status == "failed"
    assert "analysis/swim_bout_runs/bouts logical bouts" in detail


def test_batch_eye_angle_defaults_are_compact_dense_v2() -> None:
    parser = mod._build_parser()
    args = parser.parse_args(["/tmp/example_analysis.zarr"])

    assert args.eye_angle_run == "eye_angle_compact_dense_v2_batch_20260511"
    assert mod.DEFAULT_EYE_ANGLE_LAYOUT == "compact_dense_v2"


def test_eye_angle_command_pins_compact_layout() -> None:
    plan = _plan(Path("/tmp/example_analysis.zarr"))
    args = argparse.Namespace(
        eye_angle_chunk_size=8192,
        eye_angle_execution_backend=None,
        eye_angle_scheduler="processes",
        eye_angle_num_workers=24,
    )

    cmd = mod._eye_angle_command(plan, args)

    layout_idx = cmd.index("--layout")
    assert cmd[layout_idx + 1] == "compact_dense_v2"
