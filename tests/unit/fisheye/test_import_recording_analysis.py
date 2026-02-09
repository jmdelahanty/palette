from __future__ import annotations

from pathlib import Path

import zarr

from fisheye.utils import import_recording_analysis as mod


def _opts() -> mod.RecordingImportOptions:
    return mod.RecordingImportOptions(
        import_video_metadata=False,
        video_metadata_overwrite=False,
        import_stimulus=False,
        stimulus_always=False,
        stimulus_run_name=None,
        stimulus_overwrite=False,
        stimulus_quiet=True,
    )


def test_process_recording_import_returns_stimulus_failure(monkeypatch, tmp_path: Path) -> None:
    plan = mod.RecordingAnalysisPlan(
        recording_dir=tmp_path / "rec",
        h5_path=tmp_path / "rec" / "raw" / "session.h5",
        cam_video=tmp_path / "rec" / "cams" / "cam.mp4",
        zarr_path=tmp_path / "rec" / "zarr" / "rec_analysis.zarr",
    )
    opts = _opts()
    opts.import_stimulus = True
    opts.stimulus_always = True

    def _fake_stim(_plan: mod.RecordingAnalysisPlan, _opts: mod.RecordingImportOptions):
        return False, 5, ["stimulus"]

    monkeypatch.setattr(mod, "run_stimulus_import", _fake_stim)
    result = mod.process_recording_import(plan, opts, logger=None)

    assert not result.ok
    assert result.failed_step == "import_stimulus_to_zarr"
    assert result.returncode == 5


def test_stimulus_runs_present_detects_existing_run(tmp_path: Path) -> None:
    zarr_path = tmp_path / "sample_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    stim_parent = root.require_group("analysis").require_group("stimulus_runs")
    stim_parent.require_group("stimulus_20260209_000000")

    assert mod.stimulus_runs_present(zarr_path)


def test_ensure_analysis_archive_sets_purpose(tmp_path: Path) -> None:
    plan = mod.RecordingAnalysisPlan(
        recording_dir=tmp_path / "rec",
        h5_path=tmp_path / "rec" / "raw" / "session.h5",
        cam_video=tmp_path / "rec" / "cams" / "cam.mp4",
        zarr_path=tmp_path / "rec" / "zarr" / "rec_analysis.zarr",
    )

    mod.ensure_analysis_archive(plan)

    root = zarr.open_group(str(plan.zarr_path), mode="r")
    assert root.attrs.get("zarr_purpose") == "analysis"


def test_resolve_single_recording_plan_uses_default_paths(tmp_path: Path) -> None:
    rec = tmp_path / "2026-01-28T19-22-28Z_arena_1_DefaultScreen"
    (rec / "cams").mkdir(parents=True, exist_ok=True)
    (rec / "raw").mkdir(parents=True, exist_ok=True)
    video = rec / "cams" / "Cam2010093_foo.mp4"
    h5 = rec / "raw" / "session.h5"
    video.touch()
    h5.touch()

    plan = mod.resolve_single_recording_plan(recording_dir=rec)

    assert plan.recording_dir == rec.resolve()
    assert plan.cam_video == video.resolve()
    assert plan.h5_path == h5.resolve()
    assert plan.zarr_path == (rec / "zarr" / f"{rec.name}_analysis.zarr").resolve()


def test_resolve_single_recording_plan_fails_on_ambiguous_video(tmp_path: Path) -> None:
    rec = tmp_path / "rec"
    (rec / "cams").mkdir(parents=True, exist_ok=True)
    (rec / "raw").mkdir(parents=True, exist_ok=True)
    (rec / "cams" / "a.mp4").touch()
    (rec / "cams" / "b.mp4").touch()
    (rec / "raw" / "session.h5").touch()

    try:
        mod.resolve_single_recording_plan(recording_dir=rec)
    except ValueError as exc:
        assert "multiple .mp4 files" in str(exc)
    else:
        raise AssertionError("expected ValueError for ambiguous cams/*.mp4")


def test_main_defaults_to_dry_run_and_does_not_create_archive(tmp_path: Path) -> None:
    rec = tmp_path / "2026-01-28T19-22-28Z_arena_1_DefaultScreen"
    (rec / "cams").mkdir(parents=True, exist_ok=True)
    (rec / "raw").mkdir(parents=True, exist_ok=True)
    (rec / "cams" / "Cam2010093_foo.mp4").touch()
    (rec / "raw" / "session.h5").touch()
    out = rec / "zarr" / f"{rec.name}_analysis.zarr"

    rc = mod.main(["--recording-dir", str(rec)])

    assert rc == 0
    assert not out.exists()
