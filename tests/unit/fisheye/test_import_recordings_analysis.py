from __future__ import annotations

from pathlib import Path

import h5py
import zarr

from fisheye.utils import import_recordings_analysis as analysis_import


def _write_h5(path: Path, *, camera_id: str | None = None, ipc_source_name: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        if camera_id is not None:
            h5.attrs["camera_id"] = camera_id
        if ipc_source_name is not None:
            h5.attrs["ipc_source_name"] = ipc_source_name


def test_build_plans_uses_recording_analysis_naming(tmp_path: Path) -> None:
    recording_dir = tmp_path / "2026-01-28T21-47-47Z_arena_1_DefaultScreen"
    h5_path = recording_dir / "raw" / "session.h5"
    _write_h5(h5_path, camera_id="2010093")

    cams = recording_dir / "cams"
    cams.mkdir(parents=True, exist_ok=True)
    wanted = cams / "Cam2010093_foo.mp4"
    wanted.touch()

    plans = analysis_import._build_plans(  # noqa: SLF001
        root=tmp_path,
        recursive=True,
        skip_existing=True,
        check_stimulus=False,
    )

    assert len(plans) == 1
    plan = plans[0]
    assert plan.status == "ok"
    assert plan.cam_video == wanted
    assert plan.zarr_path == recording_dir / "zarr" / f"{recording_dir.name}_analysis.zarr"


def test_build_plans_marks_multi_camera_recording_as_unsupported(tmp_path: Path) -> None:
    recording_dir = tmp_path / "2026-01-28T21-47-47Z_arena_1_DefaultScreen"
    h5_path = recording_dir / "raw" / "session.h5"
    _write_h5(h5_path, camera_id="2010093")

    cams = recording_dir / "cams"
    cams.mkdir(parents=True, exist_ok=True)
    (cams / "Cam2010093_foo.mp4").touch()
    (cams / "Cam9999999_bar.mp4").touch()

    plans = analysis_import._build_plans(  # noqa: SLF001
        root=tmp_path,
        recursive=True,
        skip_existing=True,
        check_stimulus=False,
    )

    assert len(plans) == 1
    plan = plans[0]
    assert plan.status == "missing"
    assert "multi-camera analysis import is not yet supported" in (plan.reason or "")


def test_build_plans_skips_existing_analysis_zarr(tmp_path: Path) -> None:
    recording_dir = tmp_path / "2026-01-28T22-22-57Z_arena_2_Feeding"
    h5_path = recording_dir / "raw" / "session.h5"
    _write_h5(h5_path, camera_id="2010094")

    cams = recording_dir / "cams"
    cams.mkdir(parents=True, exist_ok=True)
    (cams / "Cam2010094_foo.mp4").touch()

    zarr_path = recording_dir / "zarr" / f"{recording_dir.name}_analysis.zarr"
    zarr.open_group(str(zarr_path), mode="w")

    plans = analysis_import._build_plans(  # noqa: SLF001
        root=tmp_path,
        recursive=True,
        skip_existing=True,
        check_stimulus=True,
    )

    assert len(plans) == 1
    plan = plans[0]
    assert plan.status == "skipped"
    assert "already exists" in (plan.reason or "")


def test_stimulus_runs_present_detects_existing_run(tmp_path: Path) -> None:
    zarr_path = tmp_path / "sample_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    stim_parent = root.require_group("analysis").require_group("stimulus_runs")
    stim_parent.require_group("stimulus_20260209_000000")

    assert analysis_import._stimulus_runs_present(zarr_path)  # noqa: SLF001


def test_set_analysis_purpose_overwrites_root_attr(tmp_path: Path) -> None:
    zarr_path = tmp_path / "sample_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    root.attrs["zarr_purpose"] = "production"

    analysis_import._set_analysis_purpose(zarr_path)  # noqa: SLF001

    root2 = zarr.open_group(str(zarr_path), mode="r")
    assert root2.attrs.get("zarr_purpose") == "analysis"


def test_build_plans_marks_multi_h5_recording_as_unsupported(tmp_path: Path) -> None:
    recording_dir = tmp_path / "2026-02-09T00-00-00Z_arena_multi"
    h5_1 = recording_dir / "raw" / "cam1.h5"
    h5_2 = recording_dir / "raw" / "cam2.h5"
    _write_h5(h5_1, camera_id="2010001")
    _write_h5(h5_2, camera_id="2010002")

    plans = analysis_import._build_plans(  # noqa: SLF001
        root=tmp_path,
        recursive=True,
        skip_existing=True,
        check_stimulus=False,
    )

    assert len(plans) == 2
    assert all(plan.status == "missing" for plan in plans)
    assert all("multi-camera analysis import is not yet supported" in (plan.reason or "") for plan in plans)
