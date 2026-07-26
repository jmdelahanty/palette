from __future__ import annotations

import json
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
    monkeypatch.setattr(mod, "apply_acquisition_frame_clock", lambda _plan: {})
    monkeypatch.setattr(mod, "import_experiment_setup", lambda _plan: None)
    monkeypatch.setattr(mod, "stimulus_runs_present", lambda _path: False)
    monkeypatch.setattr(mod, "run_stimulus_import", _fake_stim)
    result = mod.process_recording_import(plan, opts, logger=None)

    assert not result.ok
    assert result.failed_step == "import_stimulus_to_zarr"
    assert result.returncode == 5


def test_run_stimulus_import_forwards_metadata_and_calibration_only(
    monkeypatch,
    tmp_path: Path,
) -> None:
    plan = mod.RecordingAnalysisPlan(
        recording_dir=tmp_path / "rec",
        h5_path=tmp_path / "rec" / "raw" / "session.h5",
        cam_video=tmp_path / "rec" / "cams" / "cam.mp4",
        zarr_path=tmp_path / "rec" / "zarr" / "rec_analysis.zarr",
    )
    opts = _opts()
    opts.stimulus_metadata_and_calibration_only = True
    captured: dict[str, object] = {}

    class _Result:
        returncode = 0

    def _fake_run(cmd, *, check):
        captured["cmd"] = list(cmd)
        captured["check"] = check
        return _Result()

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

    ok, returncode, cmd = mod.run_stimulus_import(plan, opts)

    assert ok is True
    assert returncode == 0
    assert "--metadata-and-calibration-only" in cmd
    assert captured == {"cmd": cmd, "check": False}


def test_stimulus_runs_present_detects_existing_run(monkeypatch, tmp_path: Path) -> None:
    class _FakeGroup:
        def __init__(
            self,
            groups: dict[str, object] | None = None,
            keys: list[str] | None = None,
            attrs: dict[str, object] | None = None,
        ) -> None:
            self._groups = groups or {}
            self._keys = keys or []
            self.attrs = attrs or {}

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


def test_stimulus_runs_present_rejects_failed_strict_run(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class _FakeGroup:
        def __init__(
            self,
            groups: dict[str, object] | None = None,
            keys: list[str] | None = None,
            attrs: dict[str, object] | None = None,
        ) -> None:
            self._groups = groups or {}
            self._keys = keys or []
            self.attrs = attrs or {}

        def get(self, name: str):
            return self._groups.get(name)

        def group_keys(self):
            return list(self._keys)

    failed = _FakeGroup(
        attrs={
            "palette_run_completion_status": "failed",
            "stage_selector_eligible": False,
        }
    )
    parent = _FakeGroup(
        groups={"stimulus_failed": failed},
        keys=["stimulus_failed"],
        attrs={
            "palette_completion_epoch": 2,
            "latest": "stimulus_failed",
        },
    )
    fake_root = _FakeGroup(
        groups={
            "analysis": _FakeGroup(groups={"stimulus_runs": parent}),
        }
    )
    monkeypatch.setattr(mod.zarr, "open", lambda *_args, **_kwargs: fake_root)

    assert not mod.stimulus_runs_present(tmp_path / "sample_analysis.zarr")


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


def test_ensure_analysis_archive_rejects_conflicting_camera_identity(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class _FakeAttrs(dict):
        def put(self, payload):
            self.clear()
            self.update(payload)

    class _FakeGroup:
        def __init__(self) -> None:
            self.attrs = _FakeAttrs(camera_id="2010094")

    recording_dir = tmp_path / "rec"
    recording_dir.mkdir()
    (recording_dir / "recording_manifest.json").write_text(
        json.dumps({"camera_id": "2010093"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(mod.zarr, "open_group", lambda *_args, **_kwargs: _FakeGroup())
    plan = mod.RecordingAnalysisPlan(
        recording_dir=recording_dir,
        h5_path=None,
        cam_video=recording_dir / "cams" / "cam.mp4",
        zarr_path=recording_dir / "zarr" / "rec_analysis.zarr",
    )

    try:
        mod.ensure_analysis_archive(plan)
    except ValueError as exc:
        assert "camera_id conflicts" in str(exc)
    else:
        raise AssertionError("expected conflicting archive and manifest camera IDs to fail")


def test_apply_video_metadata_stamps_source_h5_fingerprint(monkeypatch, tmp_path: Path) -> None:
    rec = tmp_path / "rec"
    video = rec / "cams" / "cam.mp4"
    h5_path = rec / "raw" / "session.h5"
    zarr_path = rec / "zarr" / "rec_analysis.zarr"
    video.parent.mkdir(parents=True)
    h5_path.parent.mkdir(parents=True)
    zarr_path.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    h5_path.write_bytes(b"h5")
    root = zarr.open_group(str(zarr_path), mode="w", zarr_format=3)
    root.attrs.update(
        {
            "recording_id": "rec",
            "camera_id": "2010093",
            "source_video_metadata": {
                "imageio_metadata": {"nframes": 1.7976931348623157e308}
            },
        }
    )

    def _fake_probe(_path: Path, **_kwargs: object) -> dict[str, object]:
        return {
            "source_video": video.name,
            "source_path": str(video),
            "width": 4512,
            "height": 4512,
            "total_frames": 100,
            "fps": 100.0,
            "duration_seconds": 1.0,
            "codec": "hevc",
            "pix_fmt": "yuv420p",
        }

    monkeypatch.setattr(mod, "probe_video_metadata", _fake_probe)
    plan = mod.RecordingAnalysisPlan(
        recording_dir=rec,
        h5_path=h5_path,
        cam_video=video,
        zarr_path=zarr_path,
    )

    updates = mod.apply_video_metadata(plan, overwrite=False)

    root = zarr.open_group(str(zarr_path), mode="r")
    raw = root["raw_video"]
    assert updates["root_attrs_updated"] > 0
    assert updates["raw_video_attrs_updated"] > 0
    assert root.attrs["source_h5"] == "session.h5"
    assert raw.attrs["source_h5"] == "session.h5"
    assert root.attrs["source_h5_fingerprint_strategy"] == "stat_v1"
    assert raw.attrs["source_h5_fingerprint_strategy"] == "stat_v1"
    assert root.attrs["source_h5_fingerprint"] == raw.attrs["source_h5_fingerprint"]
    assert raw.attrs["source_h5_size_bytes"] == len(b"h5")
    assert root.attrs["source_video_metadata"]["schema_id"] == (
        "palette.source_video_metadata.v2"
    )
    assert root.attrs["source_video_metadata"]["locator"] == {
        "kind": "recording_relative",
        "relative_path": f"cams/{video.name}",
    }
    assert root.attrs["source_video_metadata"]["camera_id"] == "2010093"
    assert "imageio_metadata" not in root.attrs["source_video_metadata"]
    authority = root["analysis/acquisition_camera_frames/2010093"]
    assert authority.attrs["acquisition_import_ownership"]["mode"] == (
        "external_video_v1"
    )
    assert authority.attrs["acquisition_camera_frame"]["width_px"] == 4512
    assert authority.attrs["acquisition_camera_frame"]["height_px"] == 4512


def test_producer_video_metadata_selects_manifest_stream_for_source_video(tmp_path: Path) -> None:
    rec = tmp_path / "rec"
    video = rec / "cams" / "cam.mp4"
    video.parent.mkdir(parents=True)
    video.touch()
    (rec / "recording_manifest.json").write_text(
        json.dumps(
            {
                "video_streams": {
                    "streams": {
                        "full": {
                            "video": "cams/cam.mp4",
                            "frame_count": 139295,
                            "frame_rate": 100,
                            "codec": "hevc",
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    plan = mod.RecordingAnalysisPlan(
        recording_dir=rec,
        h5_path=None,
        cam_video=video,
        zarr_path=rec / "zarr" / "rec_analysis.zarr",
    )

    assert mod._producer_video_metadata(plan) == {  # noqa: SLF001
        "_source": "recording_manifest.video_streams.streams.full",
        "total_frames": 139295,
        "fps": 100,
        "codec": "hevc",
    }


def test_ensure_analysis_archive_imports_acquisition_video_stream_inventory(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class _FakeAttrs(dict):
        def put(self, payload):
            self.clear()
            self.update(payload)

    class _FakeGroup:
        def __init__(self) -> None:
            self.attrs = _FakeAttrs()
            self.groups: dict[str, _FakeGroup] = {}

        def require_group(self, name: str):
            group = self
            for part in name.split("/"):
                group = group.groups.setdefault(part, _FakeGroup())
            return group

    recording_dir = tmp_path / "2026-06-14T21-12-08Z_arena_1_GoodCopBadCop"
    (recording_dir / "cams").mkdir(parents=True)
    crop_dir = recording_dir / "derived" / "external_crop_recorder"
    crop_dir.mkdir(parents=True)
    (recording_dir / "cams" / "Cam2010093_sample.mp4").write_bytes(b"full")
    (crop_dir / "Cam2010093_sample_crop_external.mp4").write_bytes(b"crop")
    (crop_dir / "Cam2010093_sample_crop_meta.csv").write_text(
        "recording_frame_id,crop_x,crop_y,crop_w,crop_h,has_detection,blank_frame\n"
        "1,10,20,256,256,true,false\n"
        "2,10,20,256,256,false,true\n",
        encoding="utf-8",
    )
    (crop_dir / "Cam2010093_sample_crop_external_summary.json").write_text(
        json.dumps({"status": "completed", "frames_encoded": 2, "frames_dropped": 0}),
        encoding="utf-8",
    )
    (crop_dir / "Cam2010093_sample_crop_external_status.json").write_text(
        json.dumps({"status": "completed"}),
        encoding="utf-8",
    )
    (recording_dir / "recording_manifest.json").write_text(
        json.dumps(
            {
                "video_streams": {
                    "schema_id": "orange_runtime_video_streams_v1",
                    "frame_clock": "recording_frame_id",
                    "streams": {
                        "full": {
                            "role": "ingest_authoritative_full_frame",
                            "output_kind": "full",
                            "video": "cams/Cam2010093_sample.mp4",
                            "frame_clock": "recording_frame_id",
                            "frame_count": 2,
                        },
                        "crop": {
                            "role": "runtime_derived_acquisition_input",
                            "output_kind": "crop",
                            "video": (
                                "derived/external_crop_recorder/"
                                "Cam2010093_sample_crop_external.mp4"
                            ),
                            "metadata": (
                                "derived/external_crop_recorder/"
                                "Cam2010093_sample_crop_meta.csv"
                            ),
                            "summary": (
                                "derived/external_crop_recorder/"
                                "Cam2010093_sample_crop_external_summary.json"
                            ),
                            "frame_clock": "recording_frame_id",
                            "video_pixel_coordinate_space": "crop_frame_pixels",
                            "source_geometry_coordinate_space": "full_frame_pixels",
                            "geometry_columns": ["crop_x", "crop_y", "crop_w", "crop_h"],
                            "blank_frame_policy": "encode_black_frame_when_no_detection",
                            "selection_policy": "largest_detection_by_confidence",
                            "width": 256,
                            "height": 256,
                            "frame_count": 2,
                        },
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    fake_root = _FakeGroup()
    monkeypatch.setattr(mod.zarr, "open_group", lambda *_args, **_kwargs: fake_root)

    plan = mod.RecordingAnalysisPlan(
        recording_dir=recording_dir,
        h5_path=None,
        cam_video=recording_dir / "cams" / "Cam2010093_sample.mp4",
        zarr_path=recording_dir / "zarr" / "rec_analysis.zarr",
    )

    result = mod.ensure_analysis_archive(plan)

    assert result is not None
    assert result["schema_id"] == "palette.acquisition_video_streams.v1"
    assert result["stream_count"] == 2
    assert result["crop_stream_available"] is True
    assert fake_root.attrs["acquisition_crop_video_available"] is True
    inventory = fake_root.groups["analysis"].groups["acquisition_video_streams"]
    crop = inventory.groups["streams"].groups["crop"]
    assert crop.attrs["availability_status"] == "ok"
    assert crop.attrs["files"]["metadata"]["data_row_count"] == 2
    assert crop.attrs["contract"]["video_pixel_coordinate_space"] == "crop_frame_pixels"
    assert crop.attrs["summary"]["frames_encoded"] == 2


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


def test_resolve_single_recording_plan_uses_manifest_full_video_to_disambiguate(
    tmp_path: Path,
) -> None:
    rec = tmp_path / "rec"
    (rec / "cams").mkdir(parents=True, exist_ok=True)
    (rec / "raw").mkdir(parents=True, exist_ok=True)
    full = rec / "cams" / "Cam2010096_full.mp4"
    derived = rec / "cams" / "Cam2010096_preview.mp4"
    full.touch()
    derived.touch()
    (rec / "raw" / "session.h5").touch()
    (rec / "recording_manifest.json").write_text(
        json.dumps(
            {
                "video_streams": {
                    "streams": {
                        "full": {"video": "cams/Cam2010096_full.mp4"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    plan = mod.resolve_single_recording_plan(recording_dir=rec)

    assert plan.cam_video == full.resolve()


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
    monkeypatch.setattr(mod, "apply_acquisition_frame_clock", lambda _plan: {})

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
    monkeypatch.setattr(mod, "apply_acquisition_frame_clock", lambda _plan: {})
    monkeypatch.setattr(mod, "import_experiment_setup", lambda _plan: None)

    result = mod.process_recording_import(plan, opts, logger=None)

    assert result.ok
    assert seen["ensure"] is True
