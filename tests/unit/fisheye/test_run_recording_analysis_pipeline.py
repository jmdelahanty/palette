from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fisheye.utils import run_recording_analysis_pipeline as mod
from fisheye.utils.import_recording_analysis import RecordingAnalysisPlan, RecordingImportOptions, RecordingImportResult


def _opts(tmp_path: Path) -> mod.RecordingPipelineOptions:
    return mod.RecordingPipelineOptions(
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
        expected_subject_count=None,
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


def test_sync_pipeline_registry_uses_shadow_publication(
    monkeypatch,
    tmp_path: Path,
) -> None:
    plan = RecordingAnalysisPlan(
        recording_dir=tmp_path / "rec",
        h5_path=None,
        cam_video=tmp_path / "rec" / "cams" / "cam.mp4",
        zarr_path=tmp_path / "rec" / "zarr" / "rec_analysis.zarr",
    )

    registry_path = tmp_path / "registry.sqlite"
    receipt = object()
    observed: dict[str, object] = {}

    def _shadow(**kwargs: object) -> object:
        observed.update(kwargs)
        return SimpleNamespace(mutation_result={"dataset_id": "rec:bound"})

    monkeypatch.setattr(mod, "shadow_synchronize_recording_import", _shadow)

    assert (
        mod._sync_pipeline_registry(
            registry_path=registry_path,
            plan=plan,
            receipt=receipt,
        )
        == "rec:bound"
    )
    assert observed == {
        "canonical_registry": registry_path,
        "zarr_path": plan.zarr_path,
        "receipt": receipt,
        "decided_by": "fisheye.utils.run_recording_analysis_pipeline",
    }


def test_process_pipeline_returns_detect_failure(monkeypatch, tmp_path: Path) -> None:
    plan = RecordingAnalysisPlan(
        recording_dir=tmp_path / "rec",
        h5_path=tmp_path / "rec" / "raw" / "session.h5",
        cam_video=tmp_path / "rec" / "cams" / "cam.mp4",
        zarr_path=tmp_path / "rec" / "zarr" / "rec_analysis.zarr",
    )
    opts = _opts(tmp_path)

    monkeypatch.setattr(mod, "process_recording_import", lambda *a, **k: RecordingImportResult(ok=True))
    monkeypatch.setattr(
        mod,
        "run_detect_registry_model",
        lambda *a, **k: (False, 4, ["detect"]),
    )

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
    assert "--write-raw-video-metadata" not in cmd
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
    opts.expected_subject_count = 4

    ok, rc, cmd = mod.run_detect_quality(plan, opts)

    assert ok
    assert rc == 0
    assert called
    assert cmd == called[0]
    assert "-m" in cmd
    assert "fisheye.refinement.detect_quality" in cmd
    assert str(plan.zarr_path) in cmd
    assert "--expected-subject-count" in cmd
    assert "4" in cmd


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

    def _unexpected_publish(*_args, **_kwargs):
        raise AssertionError("Registry should not be published during dry-run")

    monkeypatch.setattr(mod, "shadow_synchronize_recording_import", _unexpected_publish)

    rc = mod.main(["--recording-dir", str(rec), "--register"])

    assert rc == 0


def test_process_pipeline_rejects_current_manifest_without_register_before_import(
    monkeypatch,
    tmp_path: Path,
) -> None:
    recording_dir = tmp_path / "rec"
    recording_dir.mkdir(parents=True)
    (recording_dir / "recording_manifest.json").write_text(
        json.dumps(
            {
                mod.SOURCE_RECORDING_IDENTITY_PROFILE_ATTR: (
                    mod.SOURCE_RECORDING_IDENTITY_PROFILE
                )
            }
        ),
        encoding="utf-8",
    )
    plan = RecordingAnalysisPlan(
        recording_dir=recording_dir,
        h5_path=None,
        cam_video=recording_dir / "cams" / "cam.mp4",
        zarr_path=recording_dir / "zarr" / "rec_analysis.zarr",
    )
    imported = False

    def _unexpected_import(*_args, **_kwargs):
        nonlocal imported
        imported = True
        raise AssertionError("current processing must fail before import")

    monkeypatch.setattr(mod, "process_recording_import", _unexpected_import)

    result = mod.process_recording_analysis_pipeline(plan, _opts(tmp_path), logger=None)

    assert result.ok is False
    assert result.failed_step == "recording_import_preflight"
    assert "receipt-bound registry publication" in (result.error or "")
    assert imported is False


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
    receipt = object()
    expected_receipt = receipt

    def _import(*_args, **_kwargs):
        order.append("import")
        return RecordingImportResult(ok=True, receipt=receipt)  # type: ignore[arg-type]

    def _detect(*_args, **_kwargs):
        order.append("detect")
        return True, 0, ["detect"]

    def _quality(*_args, **_kwargs):
        order.append("detect_quality")
        return True, 0, ["detect_quality"]

    def _refine(*_args, **_kwargs):
        order.append("refine")
        return True, 0, ["refine"]

    def _shadow(
        *,
        canonical_registry: Path,
        zarr_path: Path,
        receipt: object | None,
        decided_by: str,
    ) -> object:
        assert canonical_registry == opts.registry_path
        assert zarr_path == plan.zarr_path
        assert decided_by == "fisheye.utils.run_recording_analysis_pipeline"
        order.append("register" if receipt is expected_receipt else "refresh")
        return SimpleNamespace(mutation_result={"dataset_id": "rec:zdataset"})

    monkeypatch.setattr(mod, "process_recording_import", _import)
    monkeypatch.setattr(mod, "run_detect_registry_model", _detect)
    monkeypatch.setattr(mod, "run_detect_quality", _quality)
    monkeypatch.setattr(mod, "run_refine_detect", _refine)
    monkeypatch.setattr(mod, "shadow_synchronize_recording_import", _shadow)
    monkeypatch.setattr(
        mod,
        "load_source_recording_identity_profile",
        lambda _path: mod.SOURCE_RECORDING_IDENTITY_PROFILE,
    )

    result = mod.process_recording_analysis_pipeline(plan, opts, logger=None)

    assert result.ok is True
    assert result.dataset_id == "rec:zdataset"
    assert order == [
        "import",
        "register",
        "detect",
        "detect_quality",
        "refine",
        "refresh",
    ]


def test_process_pipeline_bound_current_import_skips_all_import_writers(
    monkeypatch,
    tmp_path: Path,
) -> None:
    plan = RecordingAnalysisPlan(
        recording_dir=tmp_path / "rec",
        h5_path=None,
        cam_video=tmp_path / "rec" / "cams" / "cam.mp4",
        zarr_path=tmp_path / "rec" / "zarr" / "rec_analysis.zarr",
    )
    plan.zarr_path.mkdir(parents=True)
    opts = _opts(tmp_path)
    opts.register = True
    order: list[str] = []

    def _receipt_paths(zarr_path: Path) -> tuple[Path, ...]:
        assert zarr_path == plan.zarr_path
        order.append("receipt_paths")
        return (zarr_path / ".imports" / "receipt.json",)

    def _shadow(
        *,
        canonical_registry: Path,
        zarr_path: Path,
        receipt: object | None,
        decided_by: str,
    ) -> object:
        assert canonical_registry == opts.registry_path
        assert zarr_path == plan.zarr_path
        assert receipt is None
        assert decided_by == "fisheye.utils.run_recording_analysis_pipeline"
        order.append("refresh")
        return SimpleNamespace(mutation_result={"dataset_id": "rec:bound"})

    monkeypatch.setattr(
        mod,
        "load_source_recording_identity_profile",
        lambda _path: mod.SOURCE_RECORDING_IDENTITY_PROFILE,
    )
    monkeypatch.setattr(mod, "recording_import_receipt_paths", _receipt_paths)
    monkeypatch.setattr(mod, "shadow_synchronize_recording_import", _shadow)
    monkeypatch.setattr(
        mod,
        "process_recording_import",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("bound replay must not invoke import writers")
        ),
    )
    monkeypatch.setattr(
        mod,
        "run_detect_registry_model",
        lambda *_args, **_kwargs: order.append("detect") or (True, 0, ["detect"]),
    )

    result = mod.process_recording_analysis_pipeline(
        plan,
        opts,
        logger=None,
    )

    assert result.ok is True
    assert result.dataset_id == "rec:bound"
    assert order == ["receipt_paths", "refresh", "detect", "refresh"]


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
    monkeypatch.setattr(
        mod,
        "run_detect_registry_model",
        lambda *a, **k: (True, 0, ["detect"]),
    )
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
    monkeypatch.setattr(
        mod,
        "require_future_normal_refined_keypoint_publication",
        lambda: None,
    )
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
    receipt = object()
    expected_receipt = receipt

    def _import(*_args, **_kwargs):
        order.append("import")
        return RecordingImportResult(ok=True, receipt=receipt)  # type: ignore[arg-type]

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

    def _shadow(
        *,
        canonical_registry: Path,
        zarr_path: Path,
        receipt: object | None,
        decided_by: str,
    ) -> object:
        assert canonical_registry == opts.registry_path
        assert zarr_path == plan.zarr_path
        assert decided_by == "fisheye.utils.run_recording_analysis_pipeline"
        order.append("register" if receipt is expected_receipt else "refresh")
        return SimpleNamespace(mutation_result={"dataset_id": "rec:zdataset"})

    monkeypatch.setattr(mod, "process_recording_import", _import)
    monkeypatch.setattr(mod, "run_detect_registry_model", _detect)
    monkeypatch.setattr(mod, "run_detect_quality", _quality)
    monkeypatch.setattr(mod, "run_refine_detect", _refine)
    monkeypatch.setattr(mod, "run_keypoints_batch", _keypoints)
    monkeypatch.setattr(mod, "run_refine_keypoints", _refine_keypoints)
    monkeypatch.setattr(mod, "shadow_synchronize_recording_import", _shadow)
    monkeypatch.setattr(
        mod,
        "load_source_recording_identity_profile",
        lambda _path: mod.SOURCE_RECORDING_IDENTITY_PROFILE,
    )

    result = mod.process_recording_analysis_pipeline(plan, opts, logger=None)

    assert result.ok is True
    assert result.dataset_id == "rec:zdataset"
    assert order == [
        "import",
        "register",
        "detect",
        "detect_quality",
        "refine",
        "keypoints",
        "refine_keypoints",
        "refresh",
    ]


def test_process_pipeline_rejects_refined_keypoints_before_import(
    monkeypatch,
    tmp_path: Path,
) -> None:
    plan = RecordingAnalysisPlan(
        recording_dir=tmp_path / "rec",
        h5_path=tmp_path / "rec" / "raw" / "session.h5",
        cam_video=tmp_path / "rec" / "cams" / "cam.mp4",
        zarr_path=tmp_path / "rec" / "zarr" / "rec_analysis.zarr",
    )
    opts = _opts(tmp_path)
    opts.refine_keypoints = True
    imported = False

    def _unexpected_import(*_args, **_kwargs):
        nonlocal imported
        imported = True
        raise AssertionError("import must not run")

    monkeypatch.setattr(mod, "process_recording_import", _unexpected_import)

    with pytest.raises(RuntimeError, match="disabled for future-normal processing"):
        mod.process_recording_analysis_pipeline(plan, opts, logger=None)

    assert imported is False
