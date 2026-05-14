from __future__ import annotations

import json
from pathlib import Path

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
        allow_preflight_failures=False,
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

    monkeypatch.setattr(mod, "ensure_analysis_archive", lambda _plan: None)
    monkeypatch.setattr(mod, "stimulus_runs_present", lambda _path: False)
    monkeypatch.setattr(mod, "run_stimulus_import", _fake_stim)
    result = mod.process_recording_import(plan, opts, logger=None)

    assert not result.ok
    assert result.failed_step == "import_stimulus_to_zarr"
    assert result.returncode == 5


def test_stimulus_runs_present_detects_existing_run(monkeypatch, tmp_path: Path) -> None:
    class _FakeGroup:
        def __init__(self, groups: dict[str, object] | None = None, keys: list[str] | None = None) -> None:
            self._groups = groups or {}
            self._keys = keys or []

        def get(self, name: str):
            return self._groups.get(name)

        def group_keys(self):
            return list(self._keys)

    fake_root = _FakeGroup(
        groups={
            "analysis": _FakeGroup(
                groups={
                    "stimulus_runs": _FakeGroup(keys=["stimulus_20260209_000000"]),
                }
            )
        }
    )
    monkeypatch.setattr(mod.zarr, "open", lambda *_args, **_kwargs: fake_root)

    assert mod.stimulus_runs_present(tmp_path / "sample_analysis.zarr")


def test_ensure_analysis_archive_sets_purpose(monkeypatch, tmp_path: Path) -> None:
    class _FakeAttrs(dict):
        def put(self, payload):
            self.clear()
            self.update(payload)

    class _FakeGroup:
        def __init__(self) -> None:
            self.attrs = _FakeAttrs()

    fake_root = _FakeGroup()
    monkeypatch.setattr(mod.zarr, "open_group", lambda *_args, **_kwargs: fake_root)

    plan = mod.RecordingAnalysisPlan(
        recording_dir=tmp_path / "rec",
        h5_path=tmp_path / "rec" / "raw" / "session.h5",
        cam_video=tmp_path / "rec" / "cams" / "cam.mp4",
        zarr_path=tmp_path / "rec" / "zarr" / "rec_analysis.zarr",
    )

    mod.ensure_analysis_archive(plan)

    assert fake_root.attrs.get("zarr_purpose") == "analysis"
    assert fake_root.attrs.get("session_uuid") == "rec"
    assert fake_root.attrs.get("recording_id") == "rec"
    assert fake_root.attrs.get("recording_name") == "rec"
    assert fake_root.attrs.get("recording_type") == "behavior"
    assert fake_root.attrs.get("recording_subtype") == "free"
    assert fake_root.attrs.get("behavior_mode") == "free"
    assert fake_root.attrs.get("artifact_schema_id") == "recording_analysis_v1"


def test_ensure_analysis_archive_marks_recording_only_context(monkeypatch, tmp_path: Path) -> None:
    class _FakeAttrs(dict):
        def put(self, payload):
            self.clear()
            self.update(payload)

    class _FakeGroup:
        def __init__(self) -> None:
            self.attrs = _FakeAttrs()

    fake_root = _FakeGroup()
    monkeypatch.setattr(mod.zarr, "open_group", lambda *_args, **_kwargs: fake_root)

    plan = mod.RecordingAnalysisPlan(
        recording_dir=tmp_path / "rec",
        h5_path=None,
        cam_video=tmp_path / "rec" / "cams" / "cam.mp4",
        zarr_path=tmp_path / "rec" / "zarr" / "rec_analysis.zarr",
    )

    mod.ensure_analysis_archive(plan)

    assert fake_root.attrs.get("zarr_purpose") == "analysis"
    assert fake_root.attrs.get("experiment_context_status") == "absent"
    assert fake_root.attrs.get("experiment_context_source") == "none"
    assert fake_root.attrs.get("stimulus_runs_available") is False


def test_ensure_analysis_archive_copies_recording_manifest_context(monkeypatch, tmp_path: Path) -> None:
    class _FakeAttrs(dict):
        def put(self, payload):
            self.clear()
            self.update(payload)

    class _FakeGroup:
        def __init__(self) -> None:
            self.attrs = _FakeAttrs()

    recording_dir = tmp_path / "sickyfish_2026_02_23_16_23_35_cam2010093"
    recording_dir.mkdir()
    (recording_dir / "recording_manifest.json").write_text(
        json.dumps(
            {
                "recording_id": "sickyfish_2026_02_23_16_23_35",
                "session_uuid": "sickyfish_2026_02_23_16_23_35_cam2010093",
                "recording_name": "sickyfish_2026_02_23_16_23_35_cam2010093",
                "session_start_iso8601_utc": "2026-02-23T21:23:35Z",
                "camera_id": "2010093",
                "dish_design": "polar",
                "protocol_name": "sickyfish",
                "num_dishes": "1",
                "fish_per_dish": "1",
            }
        ),
        encoding="utf-8",
    )
    fake_root = _FakeGroup()
    monkeypatch.setattr(mod.zarr, "open_group", lambda *_args, **_kwargs: fake_root)

    plan = mod.RecordingAnalysisPlan(
        recording_dir=recording_dir,
        h5_path=None,
        cam_video=recording_dir / "cams" / "cam.mp4",
        zarr_path=recording_dir / "zarr" / "rec_analysis.zarr",
    )

    mod.ensure_analysis_archive(plan)

    assert fake_root.attrs.get("recording_id") == recording_dir.name
    assert fake_root.attrs.get("organizer_recording_id") == "sickyfish_2026_02_23_16_23_35"
    assert fake_root.attrs.get("camera_id") == "2010093"
    assert fake_root.attrs.get("dish_design") == "polar"
    assert fake_root.attrs.get("protocol_name") == "sickyfish"
    assert fake_root.attrs.get("num_dishes") == "1"
    assert fake_root.attrs.get("fish_per_dish") == "1"
    assert fake_root.attrs.get("session_start_iso8601_utc") == "2026-02-23T21:23:35Z"


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


def test_resolve_single_recording_plan_allows_missing_h5_when_not_required(tmp_path: Path) -> None:
    rec = tmp_path / "2026-01-28T19-22-28Z_arena_1_DefaultScreen"
    (rec / "cams").mkdir(parents=True, exist_ok=True)
    video = rec / "cams" / "Cam2010093_foo.mp4"
    video.touch()

    plan = mod.resolve_single_recording_plan(recording_dir=rec, require_h5=False)

    assert plan.recording_dir == rec.resolve()
    assert plan.cam_video == video.resolve()
    assert plan.h5_path is None
    assert plan.zarr_path == (rec / "zarr" / f"{rec.name}_analysis.zarr").resolve()


def test_resolve_single_recording_plan_still_requires_h5_by_default(tmp_path: Path) -> None:
    rec = tmp_path / "2026-01-28T19-22-28Z_arena_1_DefaultScreen"
    (rec / "cams").mkdir(parents=True, exist_ok=True)
    (rec / "cams" / "Cam2010093_foo.mp4").touch()

    try:
        mod.resolve_single_recording_plan(recording_dir=rec)
    except ValueError as exc:
        assert "no .h5 files" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing raw/*.h5")


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


def test_process_recording_import_blocks_failed_preflight(tmp_path: Path) -> None:
    recording_dir = tmp_path / "rec"
    recording_dir.mkdir()
    (recording_dir / "recording_manifest.json").write_text(
        json.dumps({"preflight": {"status": "fail", "video": {"media_status": "fail"}}}),
        encoding="utf-8",
    )
    plan = mod.RecordingAnalysisPlan(
        recording_dir=recording_dir,
        h5_path=recording_dir / "raw" / "session.h5",
        cam_video=recording_dir / "cams" / "cam.mp4",
        zarr_path=recording_dir / "zarr" / "rec_analysis.zarr",
    )

    result = mod.process_recording_import(plan, _opts(), logger=None)

    assert not result.ok
    assert result.failed_step == "preflight_gate"
    assert "preflight failed" in (result.error or "")


def test_process_recording_import_rejects_stimulus_import_without_h5(monkeypatch, tmp_path: Path) -> None:
    plan = mod.RecordingAnalysisPlan(
        recording_dir=tmp_path / "rec",
        h5_path=None,
        cam_video=tmp_path / "rec" / "cams" / "cam.mp4",
        zarr_path=tmp_path / "rec" / "zarr" / "rec_analysis.zarr",
    )
    opts = _opts()
    opts.import_stimulus = True
    monkeypatch.setattr(mod, "ensure_analysis_archive", lambda _plan: None)

    result = mod.process_recording_import(plan, opts, logger=None)

    assert not result.ok
    assert result.failed_step == "import_stimulus_to_zarr"
    assert result.returncode == 2
    assert "no H5" in (result.error or "")


def test_process_recording_import_allows_failed_preflight_when_overridden(monkeypatch, tmp_path: Path) -> None:
    recording_dir = tmp_path / "rec"
    recording_dir.mkdir()
    (recording_dir / "recording_manifest.json").write_text(
        json.dumps({"preflight": {"status": "fail", "video": {"media_status": "fail"}}}),
        encoding="utf-8",
    )
    plan = mod.RecordingAnalysisPlan(
        recording_dir=recording_dir,
        h5_path=recording_dir / "raw" / "session.h5",
        cam_video=recording_dir / "cams" / "cam.mp4",
        zarr_path=recording_dir / "zarr" / "rec_analysis.zarr",
    )
    opts = _opts()
    opts.allow_preflight_failures = True
    seen: dict[str, bool] = {"ensure": False}

    def _fake_ensure(_plan: mod.RecordingAnalysisPlan) -> None:
        seen["ensure"] = True

    monkeypatch.setattr(mod, "ensure_analysis_archive", _fake_ensure)

    result = mod.process_recording_import(plan, opts, logger=None)

    assert result.ok
    assert seen["ensure"] is True
