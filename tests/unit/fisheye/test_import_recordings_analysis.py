from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

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


def test_main_logs_recording_failure_and_returns_nonzero(monkeypatch, tmp_path: Path) -> None:
    recording_dir = tmp_path / "2026-01-28T21-47-47Z_arena_1_DefaultScreen"
    h5_path = recording_dir / "raw" / "session.h5"
    _write_h5(h5_path, camera_id="2010093")
    cams = recording_dir / "cams"
    cams.mkdir(parents=True, exist_ok=True)
    (cams / "Cam2010093_foo.mp4").touch()
    log_dir = tmp_path / "logs"

    def _fake_process(*_args, **_kwargs):
        return SimpleNamespace(
            ok=False,
            failed_step="detect_yolo",
            error="detect step failed",
            returncode=17,
            dataset_id=None,
        )

    monkeypatch.setattr(analysis_import, "process_recording_analysis_pipeline", _fake_process)

    rc = analysis_import.main(
        [
            str(tmp_path),
            "--recursive",
            "--apply",
            "--log-dir",
            str(log_dir),
        ]
    )

    assert rc == 1
    logs = sorted(log_dir.glob("import_recordings_analysis_*.jsonl"))
    assert len(logs) == 1

    events = [json.loads(line) for line in logs[0].read_text(encoding="utf-8").splitlines() if line.strip()]
    failed_events = [evt for evt in events if evt.get("event") == "recording_failed"]
    assert len(failed_events) == 1
    failed = failed_events[0]
    assert failed.get("step") == "detect_yolo"
    assert failed.get("returncode") == 17


def test_main_dry_run_with_register_does_not_open_registry(monkeypatch, tmp_path: Path) -> None:
    recording_dir = tmp_path / "2026-01-28T21-47-47Z_arena_1_DefaultScreen"
    h5_path = recording_dir / "raw" / "session.h5"
    _write_h5(h5_path, camera_id="2010093")
    cams = recording_dir / "cams"
    cams.mkdir(parents=True, exist_ok=True)
    (cams / "Cam2010093_foo.mp4").touch()

    def _unexpected_registry(*_args, **_kwargs):
        raise AssertionError("Registry should not be opened during dry-run")

    monkeypatch.setattr(analysis_import, "Registry", _unexpected_registry)

    rc = analysis_import.main(
        [
            str(tmp_path),
            "--recursive",
            "--dry-run",
            "--register",
            "--no-log",
        ]
    )

    assert rc == 0


def test_main_forwards_keypoint_stage_toggles_to_pipeline(monkeypatch, tmp_path: Path) -> None:
    recording_dir = tmp_path / "2026-01-28T21-47-47Z_arena_1_DefaultScreen"
    h5_path = recording_dir / "raw" / "session.h5"
    _write_h5(h5_path, camera_id="2010093")
    cams = recording_dir / "cams"
    cams.mkdir(parents=True, exist_ok=True)
    (cams / "Cam2010093_foo.mp4").touch()

    captured: dict[str, object] = {}

    def _fake_process(_plan, opts, **_kwargs):
        captured["run_keypoints"] = opts.run_keypoints
        captured["refine_keypoints"] = opts.refine_keypoints
        captured["keypoints_config"] = opts.keypoints_config
        return SimpleNamespace(
            ok=True,
            failed_step=None,
            error=None,
            returncode=0,
            dataset_id=None,
        )

    monkeypatch.setattr(analysis_import, "process_recording_analysis_pipeline", _fake_process)

    rc = analysis_import.main(
        [
            str(tmp_path),
            "--recursive",
            "--apply",
            "--no-log",
            "--keypoints",
            "--refine-keypoints",
            "--keypoints-config",
            "configs/fisheye/default.yaml",
        ]
    )

    assert rc == 0
    assert captured["run_keypoints"] is True
    assert captured["refine_keypoints"] is True
    assert captured["keypoints_config"] == Path("configs/fisheye/default.yaml")


def test_main_rejects_deprecated_refine_max_gap(tmp_path: Path) -> None:
    try:
        analysis_import.main([str(tmp_path), "--refine-max-gap", "5"])
    except SystemExit as exc:
        assert "Interpolation overrides are deprecated and unsupported" in str(exc)
        assert "--refine-max-gap" in str(exc)
    else:  # pragma: no cover - defensive branch
        raise AssertionError("Expected SystemExit when --refine-max-gap is passed.")
