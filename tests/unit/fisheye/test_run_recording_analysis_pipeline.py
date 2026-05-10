from __future__ import annotations

from pathlib import Path

from fisheye.utils import run_recording_analysis_pipeline as mod
from fisheye.utils.import_recording_analysis import RecordingAnalysisPlan, RecordingImportOptions, RecordingImportResult


def _opts(tmp_path: Path) -> mod.RecordingPipelineOptions:
    return mod.RecordingPipelineOptions(
        model_source="explicit",
        model=None,
        detect_config=None,
        conf=None,
        iou=None,
        max_det=None,
        batch_size=None,
        cpu=False,
        set_id=None,
        require_unique=False,
        include_non_success=False,
        top_k=5,
        refine_detect=False,
        refine_config=None,
        register=False,
        registry_path=tmp_path / "registry.sqlite",
        import_opts=RecordingImportOptions(
            import_video_metadata=False,
            video_metadata_overwrite=False,
            import_stimulus=False,
            stimulus_always=False,
            stimulus_run_name=None,
            stimulus_overwrite=False,
            stimulus_quiet=True,
            allow_preflight_failures=False,
        ),
    )


def test_process_pipeline_returns_detect_failure(monkeypatch, tmp_path: Path) -> None:
    plan = RecordingAnalysisPlan(
        recording_dir=tmp_path / "rec",
        h5_path=tmp_path / "rec" / "raw" / "session.h5",
        cam_video=tmp_path / "rec" / "cams" / "cam.mp4",
        zarr_path=tmp_path / "rec" / "zarr" / "rec_analysis.zarr",
    )
    opts = _opts(tmp_path)

    monkeypatch.setattr(mod, "process_recording_import", lambda *a, **k: RecordingImportResult(ok=True))
    monkeypatch.setattr(mod, "run_detect_yolo", lambda *a, **k: (False, 4, ["detect"]))

    result = mod.process_recording_analysis_pipeline(plan, opts, logger=None)

    assert not result.ok
    assert result.failed_step == "detect_yolo"
    assert result.returncode == 4


def test_run_detect_registry_model_builds_expected_command(monkeypatch, tmp_path: Path) -> None:
    called: list[list[str]] = []

    def _fake_run(cmd: list[str], check: bool = False):  # noqa: FBT002
        called.append(cmd)

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

    plan = RecordingAnalysisPlan(
        recording_dir=tmp_path / "rec",
        h5_path=tmp_path / "rec" / "raw" / "session.h5",
        cam_video=tmp_path / "rec" / "cams" / "Cam2010093.mp4",
        zarr_path=tmp_path / "rec" / "zarr" / "rec_analysis.zarr",
    )
    opts = _opts(tmp_path)
    opts.model_source = "registry"
    opts.detect_config = Path("configs/fisheye/yolo_detect_config.yaml")
    opts.conf = 0.4
    opts.iou = 0.8
    opts.max_det = 1
    opts.batch_size = 16
    opts.set_id = "detect_cedar_shadow_v007"
    opts.require_unique = True
    opts.top_k = 7

    ok, rc, cmd = mod.run_detect_registry_model(plan, opts)

    assert ok
    assert rc == 0
    assert called
    assert cmd == called[0]
    assert "-m" in cmd
    assert "fisheye.utils.run_detect_with_registry_model" in cmd
    assert "--recording-dir" in cmd
    assert "--registry" in cmd
    assert "--write-raw-video-metadata" in cmd
    assert "--set-id" in cmd
    assert "--require-unique" in cmd
    assert "--top-k" in cmd


def test_run_keypoints_batch_builds_expected_command(monkeypatch, tmp_path: Path) -> None:
    called: list[list[str]] = []

    def _fake_run(cmd: list[str], check: bool = False):  # noqa: FBT002
        called.append(cmd)

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

    plan = RecordingAnalysisPlan(
        recording_dir=tmp_path / "rec",
        h5_path=tmp_path / "rec" / "raw" / "session.h5",
        cam_video=tmp_path / "rec" / "cams" / "Cam2010093.mp4",
        zarr_path=tmp_path / "rec" / "zarr" / "rec_analysis.zarr",
    )
    opts = _opts(tmp_path)
    opts.keypoints_config = Path("configs/fisheye/default.yaml")

    ok, rc, cmd = mod.run_keypoints_batch(plan, opts)

    assert ok
    assert rc == 0
    assert called
    assert cmd == called[0]
    assert "-m" in cmd
    assert "fisheye.utils.run_keypoints_batch" in cmd
    assert "--apply" in cmd
    assert "--quiet" in cmd
    assert "--no-log" in cmd
    assert "--config" in cmd


def test_run_detect_quality_builds_expected_command(monkeypatch, tmp_path: Path) -> None:
    called: list[list[str]] = []

    def _fake_run(cmd: list[str], check: bool = False):  # noqa: FBT002
        called.append(cmd)

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

    plan = RecordingAnalysisPlan(
        recording_dir=tmp_path / "rec",
        h5_path=tmp_path / "rec" / "raw" / "session.h5",
        cam_video=tmp_path / "rec" / "cams" / "Cam2010093.mp4",
        zarr_path=tmp_path / "rec" / "zarr" / "rec_analysis.zarr",
    )
    opts = _opts(tmp_path)

    ok, rc, cmd = mod.run_detect_quality(plan, opts)

    assert ok
    assert rc == 0
    assert called
    assert cmd == called[0]
    assert "-m" in cmd
    assert "fisheye.refinement.detect_quality" in cmd
    assert str(plan.zarr_path) in cmd


def test_main_defaults_to_dry_run_and_does_not_create_archive(monkeypatch, tmp_path: Path) -> None:
    rec = tmp_path / "2026-01-28T19-22-28Z_arena_1_DefaultScreen"
    (rec / "cams").mkdir(parents=True, exist_ok=True)
    (rec / "raw").mkdir(parents=True, exist_ok=True)
    (rec / "cams" / "Cam2010093_foo.mp4").touch()
    (rec / "raw" / "session.h5").touch()
    out = rec / "zarr" / f"{rec.name}_analysis.zarr"

    def _unexpected_call(*_args, **_kwargs):
        raise AssertionError("process_recording_analysis_pipeline should not run in dry-run mode")

    monkeypatch.setattr(mod, "process_recording_analysis_pipeline", _unexpected_call)

    rc = mod.main(["--recording-dir", str(rec)])

    assert rc == 0
    assert not out.exists()


def test_main_recording_only_dry_run_allows_missing_h5(monkeypatch, tmp_path: Path) -> None:
    rec = tmp_path / "2026-01-28T19-22-28Z_arena_1_DefaultScreen"
    (rec / "cams").mkdir(parents=True, exist_ok=True)
    (rec / "cams" / "Cam2010093_foo.mp4").touch()
    out = rec / "zarr" / f"{rec.name}_analysis.zarr"

    def _unexpected_call(*_args, **_kwargs):
        raise AssertionError("process_recording_analysis_pipeline should not run in dry-run mode")

    monkeypatch.setattr(mod, "process_recording_analysis_pipeline", _unexpected_call)

    rc = mod.main(["--recording-dir", str(rec), "--recording-only"])

    assert rc == 0
    assert not out.exists()


def test_main_dry_run_with_register_does_not_open_registry(monkeypatch, tmp_path: Path) -> None:
    rec = tmp_path / "2026-01-28T19-22-28Z_arena_1_DefaultScreen"
    (rec / "cams").mkdir(parents=True, exist_ok=True)
    (rec / "raw").mkdir(parents=True, exist_ok=True)
    (rec / "cams" / "Cam2010093_foo.mp4").touch()
    (rec / "raw" / "session.h5").touch()

    def _unexpected_registry(*_args, **_kwargs):
        raise AssertionError("Registry should not be opened during dry-run")

    monkeypatch.setattr(mod, "Registry", _unexpected_registry)

    rc = mod.main(["--recording-dir", str(rec), "--register"])

    assert rc == 0


def test_main_rejects_deprecated_refine_max_gap(tmp_path: Path) -> None:
    rec = tmp_path / "2026-01-28T19-22-28Z_arena_1_DefaultScreen"
    rec.mkdir(parents=True, exist_ok=True)

    try:
        mod.main(["--recording-dir", str(rec), "--refine-max-gap", "5"])
    except SystemExit as exc:
        assert "Interpolation overrides are deprecated and unsupported" in str(exc)
        assert "--refine-max-gap" in str(exc)
    else:  # pragma: no cover - defensive branch
        raise AssertionError("Expected SystemExit when --refine-max-gap is passed.")


def test_process_pipeline_happy_path_runs_stages_in_order(monkeypatch, tmp_path: Path) -> None:
    plan = RecordingAnalysisPlan(
        recording_dir=tmp_path / "rec",
        h5_path=tmp_path / "rec" / "raw" / "session.h5",
        cam_video=tmp_path / "rec" / "cams" / "cam.mp4",
        zarr_path=tmp_path / "rec" / "zarr" / "rec_analysis.zarr",
    )
    opts = _opts(tmp_path)
    opts.refine_detect = True
    opts.register = True
    order: list[str] = []

    def _import(*_args, **_kwargs):
        order.append("import")
        return RecordingImportResult(ok=True)

    def _detect(*_args, **_kwargs):
        order.append("detect")
        return True, 0, ["detect"]

    def _quality(*_args, **_kwargs):
        order.append("detect_quality")
        return True, 0, ["detect_quality"]

    def _refine(*_args, **_kwargs):
        order.append("refine")
        return True, 0, ["refine"]

    class _Registry:
        def scan_zarr(self, _zarr_path: Path) -> str:
            order.append("register")
            return "rec:zdataset"

    monkeypatch.setattr(mod, "process_recording_import", _import)
    monkeypatch.setattr(mod, "run_detect_yolo", _detect)
    monkeypatch.setattr(mod, "run_detect_quality", _quality)
    monkeypatch.setattr(mod, "run_refine_detect", _refine)

    result = mod.process_recording_analysis_pipeline(plan, opts, registry=_Registry(), logger=None)

    assert result.ok is True
    assert result.dataset_id == "rec:zdataset"
    assert order == ["import", "detect", "detect_quality", "refine", "register"]


def test_process_pipeline_returns_detect_quality_failure(monkeypatch, tmp_path: Path) -> None:
    plan = RecordingAnalysisPlan(
        recording_dir=tmp_path / "rec",
        h5_path=tmp_path / "rec" / "raw" / "session.h5",
        cam_video=tmp_path / "rec" / "cams" / "cam.mp4",
        zarr_path=tmp_path / "rec" / "zarr" / "rec_analysis.zarr",
    )
    opts = _opts(tmp_path)
    opts.refine_detect = True

    monkeypatch.setattr(mod, "process_recording_import", lambda *a, **k: RecordingImportResult(ok=True))
    monkeypatch.setattr(mod, "run_detect_yolo", lambda *a, **k: (True, 0, ["detect"]))
    monkeypatch.setattr(mod, "run_detect_quality", lambda *a, **k: (False, 7, ["detect_quality"]))
    monkeypatch.setattr(
        mod,
        "run_refine_detect",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("refine should not run")),
    )

    result = mod.process_recording_analysis_pipeline(plan, opts, logger=None)

    assert not result.ok
    assert result.failed_step == "detect_quality"
    assert result.returncode == 7


def test_process_pipeline_full_stack_runs_stages_in_order(monkeypatch, tmp_path: Path) -> None:
    plan = RecordingAnalysisPlan(
        recording_dir=tmp_path / "rec",
        h5_path=tmp_path / "rec" / "raw" / "session.h5",
        cam_video=tmp_path / "rec" / "cams" / "cam.mp4",
        zarr_path=tmp_path / "rec" / "zarr" / "rec_analysis.zarr",
    )
    opts = _opts(tmp_path)
    opts.refine_detect = True
    opts.run_keypoints = True
    opts.refine_keypoints = True
    opts.register = True
    order: list[str] = []

    def _import(*_args, **_kwargs):
        order.append("import")
        return RecordingImportResult(ok=True)

    def _detect(*_args, **_kwargs):
        order.append("detect")
        return True, 0, ["detect"]

    def _quality(*_args, **_kwargs):
        order.append("detect_quality")
        return True, 0, ["detect_quality"]

    def _refine(*_args, **_kwargs):
        order.append("refine")
        return True, 0, ["refine"]

    def _keypoints(*_args, **_kwargs):
        order.append("keypoints")
        return True, 0, ["keypoints"]

    def _refine_keypoints(*_args, **_kwargs):
        order.append("refine_keypoints")
        return True, 0, ["refine_keypoints"]

    class _Registry:
        def scan_zarr(self, _zarr_path: Path) -> str:
            order.append("register")
            return "rec:zdataset"

    monkeypatch.setattr(mod, "process_recording_import", _import)
    monkeypatch.setattr(mod, "run_detect_yolo", _detect)
    monkeypatch.setattr(mod, "run_detect_quality", _quality)
    monkeypatch.setattr(mod, "run_refine_detect", _refine)
    monkeypatch.setattr(mod, "run_keypoints_batch", _keypoints)
    monkeypatch.setattr(mod, "run_refine_keypoints", _refine_keypoints)

    result = mod.process_recording_analysis_pipeline(plan, opts, registry=_Registry(), logger=None)

    assert result.ok is True
    assert result.dataset_id == "rec:zdataset"
    assert order == ["import", "detect", "detect_quality", "refine", "keypoints", "refine_keypoints", "register"]
