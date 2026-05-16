from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import zarr

from fisheye.utils import run_eye_masks_batch as mod


def _create_latest_run(root: zarr.Group, group_name: str, run_name: str) -> None:
    parent = root.require_group(group_name)
    parent.create_group(run_name)
    parent.attrs["latest"] = run_name


def _create_h5(path: Path, camera_id: str = "3") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.attrs["camera_id"] = camera_id


@pytest.mark.parametrize(
    ("with_crop", "with_keypoints", "with_eye_masks", "skip_existing", "refine_only", "status", "reason"),
    [
        (False, True, False, False, False, "missing", "crop missing"),
        (True, False, False, False, False, "missing", "keypoints missing"),
        (True, True, True, True, False, "skipped", "eye masks already present"),
        (False, False, False, False, True, "missing", "eye_masks missing"),
    ],
)
def test_plan_from_zarr_prerequisites(
    tmp_path: Path,
    with_crop: bool,
    with_keypoints: bool,
    with_eye_masks: bool,
    skip_existing: bool,
    refine_only: bool,
    status: str,
    reason: str,
) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    if with_crop:
        _create_latest_run(root, "crop_runs", "crop_001")
    if with_keypoints:
        _create_latest_run(root, "keypoints_runs", "keypoints_001")
    if with_eye_masks:
        _create_latest_run(root, "eye_masks_runs", "eye_masks_001")

    plan = mod._plan_from_zarr(
        zarr_path=zarr_path,
        recording_dir=tmp_path,
        h5_path=tmp_path / "raw" / "recording.h5",
        camera_id="3",
        skip_existing=skip_existing,
        require_crop=True,
        require_keypoints=True,
        refine_only=refine_only,
    )

    assert plan.status == status
    assert plan.reason == reason


def test_plan_from_zarr_refine_missing_only_skips_when_refined_exists(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_training.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    _create_latest_run(root, "eye_masks_runs", "eye_masks_001")
    _create_latest_run(root, "refined_eye_masks_runs", "refined_eye_masks_001")

    plan = mod._plan_from_zarr(
        zarr_path=zarr_path,
        recording_dir=tmp_path,
        h5_path=tmp_path / "raw" / "recording.h5",
        camera_id="3",
        skip_existing=True,
        require_crop=False,
        require_keypoints=False,
        refine_only=True,
        refine_missing_only=True,
    )

    assert plan.status == "skipped"
    assert plan.reason == "refined eye masks already present"


def test_plan_from_zarr_model_eye_masks_accept_geometry_only_latest_any(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    crop_parent = root.require_group("crop_runs")
    crop_parent.attrs["latest_any"] = "crop_geometry"
    crop_geometry = crop_parent.create_group("crop_geometry")
    crop_geometry.attrs["crop_storage_mode"] = "geometry_only"
    _create_latest_run(root, "keypoints_runs", "keypoints_001")

    plan = mod._plan_from_zarr(
        zarr_path=zarr_path,
        recording_dir=tmp_path,
        h5_path=tmp_path / "raw" / "recording.h5",
        camera_id="3",
        skip_existing=True,
        require_crop=True,
        require_keypoints=True,
        refine_only=False,
        crop_storage_requirement="any",
    )

    assert plan.status == "ok"
    assert plan.crop_present is True


def test_plan_from_zarr_traditional_eye_masks_require_materialized_crop(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    crop_parent = root.require_group("crop_runs")
    crop_parent.attrs["latest_any"] = "crop_geometry"
    crop_geometry = crop_parent.create_group("crop_geometry")
    crop_geometry.attrs["crop_storage_mode"] = "geometry_only"
    _create_latest_run(root, "keypoints_runs", "keypoints_001")

    plan = mod._plan_from_zarr(
        zarr_path=zarr_path,
        recording_dir=tmp_path,
        h5_path=tmp_path / "raw" / "recording.h5",
        camera_id="3",
        skip_existing=True,
        require_crop=True,
        require_keypoints=True,
        refine_only=False,
        crop_storage_requirement="materialized",
    )

    assert plan.status == "missing"
    assert plan.reason == "crop missing"
    assert plan.crop_present is False

    crop_parent.attrs["latest_materialized"] = "crop_materialized"
    crop_materialized = crop_parent.create_group("crop_materialized")
    crop_materialized.attrs["crop_storage_mode"] = "materialized"
    crop_materialized.create_array(
        "roi_images",
        data=np.zeros((1, 4, 4), dtype=np.uint8),
        overwrite=True,
    )

    plan = mod._plan_from_zarr(
        zarr_path=zarr_path,
        recording_dir=tmp_path,
        h5_path=tmp_path / "raw" / "recording.h5",
        camera_id="3",
        skip_existing=True,
        require_crop=True,
        require_keypoints=True,
        refine_only=False,
        crop_storage_requirement="materialized",
    )

    assert plan.status == "ok"
    assert plan.crop_present is True


@pytest.mark.parametrize(
    ("zarr_use", "expected_name"),
    [
        ("training", "recording_training.zarr"),
        ("analysis", "recording_analysis.zarr"),
    ],
)
def test_build_plans_respects_zarr_use_suffix(
    tmp_path: Path,
    zarr_use: str,
    expected_name: str,
) -> None:
    recording_dir = tmp_path / "recording_a"
    h5_path = recording_dir / "raw" / "recording.h5"
    _create_h5(h5_path)

    for suffix in ("analysis", "training"):
        zarr_path = recording_dir / "zarr" / f"recording_{suffix}.zarr"
        root = zarr.open_group(str(zarr_path), mode="w")
        _create_latest_run(root, "crop_runs", "crop_001")
        _create_latest_run(root, "keypoints_runs", "keypoints_001")

    plans = mod._build_plans(
        [tmp_path],
        recursive=True,
        skip_existing=True,
        require_crop=True,
        require_keypoints=True,
        refine_only=False,
        zarr_use=zarr_use,
    )

    assert len(plans) == 1
    assert plans[0].zarr_path.name == expected_name
    assert plans[0].status == "ok"


def test_build_plans_all_includes_analysis_and_training_when_present(tmp_path: Path) -> None:
    recording_dir = tmp_path / "recording_a"
    h5_path = recording_dir / "raw" / "recording.h5"
    _create_h5(h5_path)

    for suffix in ("analysis", "training"):
        zarr_path = recording_dir / "zarr" / f"recording_{suffix}.zarr"
        root = zarr.open_group(str(zarr_path), mode="w")
        _create_latest_run(root, "crop_runs", "crop_001")
        _create_latest_run(root, "keypoints_runs", "keypoints_001")

    plans = mod._build_plans(
        [tmp_path],
        recursive=True,
        skip_existing=True,
        require_crop=True,
        require_keypoints=True,
        refine_only=False,
        zarr_use="all",
    )

    assert sorted(plan.zarr_path.name for plan in plans) == [
        "recording_analysis.zarr",
        "recording_training.zarr",
    ]
    assert all(plan.status == "ok" for plan in plans)


def test_build_plans_from_zarr_filters_by_zarr_use(tmp_path: Path) -> None:
    analysis_path = tmp_path / "recording_analysis.zarr"
    analysis_root = zarr.open_group(str(analysis_path), mode="w")
    _create_latest_run(analysis_root, "crop_runs", "crop_001")
    _create_latest_run(analysis_root, "keypoints_runs", "keypoints_001")

    training_path = tmp_path / "recording_training.zarr"
    training_root = zarr.open_group(str(training_path), mode="w")
    _create_latest_run(training_root, "crop_runs", "crop_001")
    _create_latest_run(training_root, "keypoints_runs", "keypoints_001")

    plans = mod._build_plans_from_zarr(
        [analysis_path, training_path],
        skip_existing=True,
        require_crop=True,
        require_keypoints=True,
        refine_only=False,
        zarr_use="training",
    )

    assert len(plans) == 1
    assert plans[0].zarr_path == training_path


def test_build_plans_from_zarr_uses_source_recording_context_for_copied_archive(tmp_path: Path) -> None:
    source_recording = tmp_path / "source_recording"
    source_video = source_recording / "cams" / "cam_1.mp4"
    source_video.parent.mkdir(parents=True, exist_ok=True)
    source_video.write_bytes(b"")

    copied_path = tmp_path / "smoke" / "copied_analysis.zarr"
    copied_root = zarr.open_group(str(copied_path), mode="w")
    copied_root.attrs["source_video_path"] = str(source_video.resolve())
    analysis_metadata = copied_root.require_group("analysis_metadata")
    analysis_metadata.attrs["session_uuid"] = "2026-01-28T19-22-28Z_arena_1"
    _create_latest_run(copied_root, "crop_runs", "crop_001")
    _create_latest_run(copied_root, "keypoints_runs", "keypoints_001")

    plans = mod._build_plans_from_zarr(
        [copied_path],
        skip_existing=True,
        require_crop=True,
        require_keypoints=True,
        refine_only=False,
        zarr_use="analysis",
    )

    assert len(plans) == 1
    assert plans[0].status == "ok"
    assert plans[0].recording_dir == source_recording.resolve()
    assert plans[0].zarr_path == copied_path


def test_resolve_method_prefers_override() -> None:
    cfg = {"eye_masks": {"method": "traditional"}}
    assert mod._resolve_method(cfg, "yolo") == "yolo"
    assert mod._resolve_method({"eye_masks": {"method": "UnEt"}}, None) == "unet"
    assert mod._resolve_method({"eye_masks": {"method": "yolo_eye_segmentation"}}, None) == "yolo"


def test_validate_method_requirements_traditional_has_no_model_requirement() -> None:
    mod._validate_method_requirements({"eye_masks": {"method": "traditional"}}, "traditional", refine_only=False)


def test_validate_method_requirements_yolo_requires_model_when_running_segmentation() -> None:
    with pytest.raises(ValueError, match="Method 'yolo' requires eye_masks.model_path or eye_masks.model"):
        mod._validate_method_requirements({"eye_masks": {}}, "yolo", refine_only=False)


def test_validate_method_requirements_unet_requires_checkpoint_when_running_segmentation() -> None:
    with pytest.raises(ValueError, match="Method 'unet' requires eye_masks.checkpoint"):
        mod._validate_method_requirements({"eye_masks": {}}, "unet", refine_only=False)


def test_validate_method_requirements_refine_only_skips_model_checks() -> None:
    mod._validate_method_requirements({"eye_masks": {}}, "yolo", refine_only=True)
    mod._validate_method_requirements({"eye_masks": {}}, "unet", refine_only=True)


@pytest.mark.parametrize(
    ("method", "expected_runner"),
    [
        ("traditional", "traditional"),
        ("yolo", "yolo"),
        ("yolo_eye_segmentation", "yolo"),
        ("unet", "unet"),
        ("unet_eye_masks", "unet"),
    ],
)
def test_run_plan_dispatch_and_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    method: str,
    expected_runner: str,
) -> None:
    called: list[str] = []

    def _traditional(*args, **kwargs):  # noqa: ANN002, ANN003
        called.append("traditional")
        return "eye_masks_traditional_001"

    def _yolo(*args, **kwargs):  # noqa: ANN002, ANN003
        called.append("yolo")
        return "eye_masks_yolo_001"

    def _unet(*args, **kwargs):  # noqa: ANN002, ANN003
        called.append("unet")
        return "eye_masks_unet_001"

    def _collect(zarr_path: Path, group_name: str, run_name: str) -> dict:
        return {
            "group": group_name,
            "run_name": run_name,
            "method": "mock",
            "duration_seconds": 0.25,
            "source_runs": {"source_keypoints_run": "keypoints_001"},
        }

    def _project(_zarr_path: Path, eye_run: str) -> dict:
        return {
            "group": "subject_mask_runs",
            "run_name": f"subject_masks_from_{eye_run}",
            "method": "mock",
            "label_schema_id": "subject_v1_union",
            "projection_mode": "eyes_union_from_pair",
            "run_semantics": "eye_mask_runtime_projection",
            "source_runs": {
                "source_eye_masks_run": eye_run,
                "source_mask_stage": "eye_masks_runs",
            },
        }

    monkeypatch.setattr(mod, "_run_traditional", _traditional)
    monkeypatch.setattr(mod, "_run_yolo", _yolo)
    monkeypatch.setattr(mod, "_run_unet", _unet)
    monkeypatch.setattr(mod, "_collect_stage_payload", _collect)
    monkeypatch.setattr(mod, "_project_subject_masks_after_eye_run", _project)
    monkeypatch.setattr(
        mod,
        "_sync_eye_mask_registry_rows_after_run",
        lambda **kwargs: {"synced": True, "dataset_id": "dataset_x"},  # noqa: ANN003
    )

    plan = mod.EyeMaskPlan(
        recording_dir=tmp_path,
        h5_path=tmp_path / "raw" / "recording.h5",
        zarr_path=tmp_path / "zarr" / "recording_analysis.zarr",
        camera_id="3",
        status="ok",
    )
    result = mod._run_plan(
        plan,
        config={},
        method=method,
        scheduler=None,
        num_workers=None,
        quiet=True,
        refine=False,
        refine_only=False,
    )

    assert called == [expected_runner]
    assert result["method"] == expected_runner
    assert result["status"] == "ok"
    assert "duration_seconds" in result
    assert result["eye_masks"]["run_name"].startswith("eye_masks_")
    assert result["eye_masks"]["source_runs"]["source_keypoints_run"] == "keypoints_001"
    assert result["subject_masks"]["group"] == "subject_mask_runs"
    assert result["subject_masks"]["source_runs"]["source_eye_masks_run"] == result["eye_masks"]["run_name"]
    assert result["registry_sync"]["synced"] is True
    assert "refined_eye_masks" not in result


def test_run_plan_refine_only_uses_latest_eye_mask_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen: dict[str, str] = {}

    monkeypatch.setattr(mod, "_latest_run", lambda _zarr_path, _group_name: "eye_masks_010")

    def _run_refine(*args, **kwargs):  # noqa: ANN002, ANN003
        seen["source_run"] = kwargs["source_run"]
        return "refined_eye_masks_011"

    def _collect(zarr_path: Path, group_name: str, run_name: str) -> dict:
        return {
            "group": group_name,
            "run_name": run_name,
            "method": "refine",
            "duration_seconds": 0.4,
            "source_runs": {"source_eye_masks_run": "eye_masks_010"},
        }

    monkeypatch.setattr(mod, "_run_refine", _run_refine)
    monkeypatch.setattr(mod, "_collect_stage_payload", _collect)
    monkeypatch.setattr(
        mod,
        "_project_subject_masks_after_eye_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("raw projection should not run in refine-only mode")),
    )
    monkeypatch.setattr(
        mod,
        "_sync_eye_mask_registry_rows_after_run",
        lambda **kwargs: {"synced": True, "dataset_id": "dataset_x"},  # noqa: ANN003
    )

    plan = mod.EyeMaskPlan(
        recording_dir=tmp_path,
        h5_path=tmp_path / "raw" / "recording.h5",
        zarr_path=tmp_path / "zarr" / "recording_analysis.zarr",
        camera_id="3",
        status="ok",
    )
    result = mod._run_plan(
        plan,
        config={},
        method="traditional",
        scheduler=None,
        num_workers=None,
        quiet=True,
        refine=False,
        refine_only=True,
    )

    assert seen["source_run"] == "eye_masks_010"
    assert "eye_masks" not in result
    assert result["refined_eye_masks"]["group"] == "refined_eye_masks_runs"
    assert result["refined_eye_masks"]["run_name"] == "refined_eye_masks_011"
    assert result["registry_sync"]["synced"] is True


def test_sync_eye_mask_registry_rows_after_run_status_context_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(mod, "_resolve_step_status_context", lambda *_args, **_kwargs: None)

    result = mod._sync_eye_mask_registry_rows_after_run(
        zarr_path=tmp_path / "recording_analysis.zarr",
        stage_payloads={},
    )

    assert result == {"synced": False, "reason": "status_context_unavailable"}


def test_sync_eye_mask_registry_rows_after_run_writes_status_and_refreshes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        mod,
        "_resolve_step_status_context",
        lambda *_args, **_kwargs: {
            "registry_path": tmp_path / "registry.sqlite",
            "dataset_id": "dataset_001",
            "recording_id": "recording_001",
        },
    )

    class _Cursor:
        def __init__(self, row: dict[str, object]):
            self._row = row

        def fetchone(self) -> dict[str, object]:
            return self._row

    class _Conn:
        def execute(self, _sql: str, _params: tuple[str]) -> _Cursor:
            return _Cursor({"recording_id": "recording_db", "zarr_use": "analysis"})

    class _FakeRegistry:
        def __init__(self, _path: Path):
            self.conn = _Conn()

        def refresh_eye_mask_performance_for_dataset(self, *_args, **_kwargs) -> int:
            return 4

        def refresh_eye_mask_quality_for_dataset(self, *_args, **_kwargs) -> int:
            return 2

        def refresh_subject_mask_performance_for_dataset(self, *_args, **_kwargs) -> int:
            return 3

        def refresh_subject_mask_component_quality_for_dataset(self, *_args, **_kwargs) -> int:
            return 5

        def close(self) -> None:
            return None

    step_calls: list[dict[str, object]] = []

    def _fake_upsert(_registry, **kwargs):  # noqa: ANN001, ANN003
        step_calls.append(kwargs)

    monkeypatch.setattr(mod, "Registry", _FakeRegistry)
    monkeypatch.setattr(mod, "upsert_recording_step_status", _fake_upsert)
    monkeypatch.setattr(mod, "_zarr_mtime_ns", lambda _path: 123)

    result = mod._sync_eye_mask_registry_rows_after_run(
        zarr_path=tmp_path / "recording_analysis.zarr",
        stage_payloads={
            "eye_masks": {
                "run_name": "eye_masks_001",
                "method": "yolo_eye_segmentation",
                "source_runs": {"source_crop_run": "crop_001"},
                "successful_roi_pair_rate": 0.75,
            },
            "refined_eye_masks": {
                "run_name": "refined_eye_masks_001",
                "method": "refine_eye_masks",
                "source_runs": {"source_eye_masks_run": "eye_masks_001"},
                "successful_roi_pair_rate": 80.0,
            },
            "subject_masks": {
                "run_name": "subject_masks_001",
                "method": "yolo_eye_segmentation",
                "label_schema_id": "subject_v1_union",
                "projection_mode": "eyes_union_from_pair",
                "run_semantics": "eye_mask_runtime_projection",
                "source_runs": {
                    "source_eye_masks_run": "eye_masks_001",
                    "source_mask_stage": "eye_masks_runs",
                },
            },
        },
    )

    assert result["synced"] is True
    assert result["dataset_id"] == "dataset_001"
    assert result["recording_id"] == "recording_db"
    assert result["eye_mask_performance_rows"] == 4
    assert result["eye_mask_quality_rows"] == 2
    assert result["subject_mask_performance_rows"] == 3
    assert result["subject_mask_component_quality_rows"] == 5
    assert result["step_status_written"] == ["eye_masks", "refined_eye_masks", "subject_masks"]
    assert result["step_status_errors"] == {}
    assert [call["step_name"] for call in step_calls] == ["eye_masks", "refined_eye_masks", "subject_masks"]
    assert step_calls[0]["coverage_pct"] == pytest.approx(75.0)
    assert step_calls[1]["coverage_pct"] == pytest.approx(80.0)
    assert step_calls[2]["coverage_pct"] is None
    assert step_calls[2]["details_json"]["projection_mode"] == "eyes_union_from_pair"


def test_sync_eye_mask_registry_rows_after_run_refresh_failure_is_non_fatal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        mod,
        "_resolve_step_status_context",
        lambda *_args, **_kwargs: {
            "registry_path": tmp_path / "registry.sqlite",
            "dataset_id": "dataset_001",
            "recording_id": "recording_001",
        },
    )

    class _Cursor:
        def __init__(self, row: dict[str, object]):
            self._row = row

        def fetchone(self) -> dict[str, object]:
            return self._row

    class _Conn:
        def execute(self, _sql: str, _params: tuple[str]) -> _Cursor:
            return _Cursor({"recording_id": "recording_db", "zarr_use": "analysis"})

    class _FakeRegistry:
        def __init__(self, _path: Path):
            self.conn = _Conn()

        def refresh_eye_mask_performance_for_dataset(self, *_args, **_kwargs) -> int:
            raise RuntimeError("refresh boom")

        def refresh_eye_mask_quality_for_dataset(self, *_args, **_kwargs) -> int:
            return 0

        def refresh_subject_mask_performance_for_dataset(self, *_args, **_kwargs) -> int:
            return 0

        def refresh_subject_mask_component_quality_for_dataset(self, *_args, **_kwargs) -> int:
            return 0

        def close(self) -> None:
            return None

    monkeypatch.setattr(mod, "Registry", _FakeRegistry)
    monkeypatch.setattr(mod, "upsert_recording_step_status", lambda *_args, **_kwargs: None)

    result = mod._sync_eye_mask_registry_rows_after_run(
        zarr_path=tmp_path / "recording_analysis.zarr",
        stage_payloads={"eye_masks": {"run_name": "eye_masks_001", "source_runs": {}}},
    )

    assert result["synced"] is False
    assert result["reason"] == "refresh_failed"
    assert "refresh boom" in str(result["error"])


def test_main_json_failure_row_includes_failure_reason(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan = mod.EyeMaskPlan(
        recording_dir=tmp_path / "recording_a",
        h5_path=tmp_path / "recording_a" / "raw" / "recording_a.h5",
        zarr_path=tmp_path / "recording_a" / "zarr" / "recording_a_analysis.zarr",
        camera_id="1",
        status="ok",
    )
    monkeypatch.setattr(mod, "_resolve_root", lambda _paths: [tmp_path])
    monkeypatch.setattr(mod, "_load_config", lambda _path: {"eye_masks": {"method": "traditional"}})
    monkeypatch.setattr(mod, "_build_plans", lambda *args, **kwargs: [plan])  # noqa: ARG005
    monkeypatch.setattr(mod, "_build_plans_from_zarr", lambda *args, **kwargs: [])  # noqa: ARG005

    def _fail(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("segmentation exploded")

    monkeypatch.setattr(mod, "_run_plan", _fail)

    rc = mod.main(["--apply", "--json", "--no-log", str(tmp_path)])
    assert rc == 0

    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert rows
    failure = rows[-1]
    assert failure["status"] == "failed"
    assert failure["method"] == "traditional"
    assert failure["failure_reason"] == "segmentation exploded"


def test_main_preflight_validation_fails_before_planning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(mod, "_resolve_root", lambda _paths: [tmp_path])
    monkeypatch.setattr(mod, "_load_config", lambda _path: {"eye_masks": {"method": "yolo"}})
    monkeypatch.setattr(
        mod,
        "_build_plans",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("_build_plans should not run")),  # noqa: ARG005
    )
    monkeypatch.setattr(mod, "_build_plans_from_zarr", lambda *args, **kwargs: [])  # noqa: ARG005

    with pytest.raises(SystemExit, match="Method 'yolo' requires eye_masks.model_path or eye_masks.model"):
        mod.main(["--apply", "--no-log", str(tmp_path)])


def test_main_model_source_registry_skips_config_model_requirement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = mod.EyeMaskPlan(
        recording_dir=tmp_path / "recording_a",
        h5_path=tmp_path / "recording_a" / "raw" / "recording_a.h5",
        zarr_path=tmp_path / "recording_a" / "zarr" / "recording_a_analysis.zarr",
        camera_id="1",
        status="ok",
    )
    registry_path = tmp_path / "registry.sqlite"
    registry_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(mod, "_resolve_root", lambda _paths: [tmp_path])
    monkeypatch.setattr(mod, "_load_config", lambda _path: {"eye_masks": {"method": "yolo"}})
    monkeypatch.setattr(mod, "_build_plans", lambda *args, **kwargs: [plan])  # noqa: ARG005
    monkeypatch.setattr(mod, "_build_plans_from_zarr", lambda *args, **kwargs: [])  # noqa: ARG005

    rc = mod.main(
        [
            "--dry-run",
            "--no-log",
            "--model-source",
            "registry",
            "--registry",
            str(registry_path),
            str(tmp_path),
        ]
    )
    assert rc == 0


def test_main_apply_model_source_registry_uses_pre_resolved_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = mod.EyeMaskPlan(
        recording_dir=tmp_path / "recording_a",
        h5_path=tmp_path / "recording_a" / "raw" / "recording_a.h5",
        zarr_path=tmp_path / "recording_a" / "zarr" / "recording_a_analysis.zarr",
        camera_id="1",
        status="ok",
    )
    registry_path = tmp_path / "registry.sqlite"
    registry_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(mod, "_resolve_root", lambda _paths: [tmp_path])
    monkeypatch.setattr(
        mod,
        "_load_config",
        lambda _path: {
            "eye_masks": {
                "method": "yolo",
                "batch_size": 16,
                "use_crop": True,
                "roi_cache_policy": "always",
                "roi_cache_dir": "/tmp/roi-cache",
            }
        },
    )
    monkeypatch.setattr(mod, "_build_plans", lambda *args, **kwargs: [plan])  # noqa: ARG005
    monkeypatch.setattr(mod, "_build_plans_from_zarr", lambda *args, **kwargs: [])  # noqa: ARG005
    monkeypatch.setattr(
        mod,
        "_resolve_registry_models_for_plans",
        lambda **_kwargs: (  # noqa: ARG005
            {
                str(plan.zarr_path.resolve()): mod.ResolvedModel(
                    model_path="/models/eye_masks_yolo.pt",
                    payload={
                        "selected": {
                            "run_id": "eye_run_001",
                            "set_id": "eye_masks_set_001",
                            "model_path": "/models/eye_masks_yolo.pt",
                        }
                    },
                )
            },
            {},
        ),
    )

    captured: dict[str, object] = {}

    def _fake_run_plan_with_registry_model(*args, **kwargs):  # noqa: ANN002, ANN003
        captured["resolved_model"] = kwargs.get("resolved_model")
        return {
            "recording": str(plan.recording_dir),
            "zarr": str(plan.zarr_path),
            "status": "ok",
            "method": "yolo",
            "registry_sync": {"synced": True},
        }

    monkeypatch.setattr(mod, "_run_plan_with_registry_model", _fake_run_plan_with_registry_model)

    rc = mod.main(
        [
            "--apply",
            "--no-log",
            "--method",
            "yolo",
            "--model-source",
            "registry",
            "--registry",
            str(registry_path),
            "--model-set-id",
            "eye_masks_set_001",
            "--model-top-k",
            "7",
            str(tmp_path),
        ]
    )
    assert rc == 0
    resolved_model = captured.get("resolved_model")
    assert isinstance(resolved_model, mod.ResolvedModel)
    assert resolved_model.model_path == "/models/eye_masks_yolo.pt"


def test_main_apply_model_source_registry_for_unet_uses_pre_resolved_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = mod.EyeMaskPlan(
        recording_dir=tmp_path / "recording_b",
        h5_path=tmp_path / "recording_b" / "raw" / "recording_b.h5",
        zarr_path=tmp_path / "recording_b" / "zarr" / "recording_b_analysis.zarr",
        camera_id="2",
        status="ok",
    )
    registry_path = tmp_path / "registry.sqlite"
    registry_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(mod, "_resolve_root", lambda _paths: [tmp_path])
    monkeypatch.setattr(
        mod,
        "_load_config",
        lambda _path: {
            "eye_masks": {
                "method": "unet",
                "checkpoint": "/tmp/unet.pt",
                "batch_size": 24,
                "use_crop": True,
                "roi_cache_policy": "always",
                "roi_cache_dir": "/tmp/unet-roi-cache",
            }
        },
    )
    monkeypatch.setattr(mod, "_build_plans", lambda *args, **kwargs: [plan])  # noqa: ARG005
    monkeypatch.setattr(mod, "_build_plans_from_zarr", lambda *args, **kwargs: [])  # noqa: ARG005
    monkeypatch.setattr(
        mod,
        "_resolve_registry_models_for_plans",
        lambda **_kwargs: (  # noqa: ARG005
            {
                str(plan.zarr_path.resolve()): mod.ResolvedModel(
                    model_path="/models/eye_masks_unet.pt",
                    payload={
                        "selected": {
                            "run_id": "eye_run_002",
                            "set_id": "eye_masks_set_002",
                            "model_path": "/models/eye_masks_unet.pt",
                        }
                    },
                )
            },
            {},
        ),
    )

    captured: dict[str, object] = {}

    def _fake_run_plan_with_registry_model(*args, **kwargs):  # noqa: ANN002, ANN003
        captured["resolved_model"] = kwargs.get("resolved_model")
        return {
            "recording": str(plan.recording_dir),
            "zarr": str(plan.zarr_path),
            "status": "ok",
            "method": "unet",
            "registry_sync": {"synced": True},
        }

    monkeypatch.setattr(mod, "_run_plan_with_registry_model", _fake_run_plan_with_registry_model)

    rc = mod.main(
        [
            "--apply",
            "--no-log",
            "--method",
            "unet",
            "--model-source",
            "registry",
            "--registry",
            str(registry_path),
            "--model-set-id",
            "eye_masks_set_002",
            str(tmp_path),
        ]
    )

    assert rc == 0
    resolved_model = captured.get("resolved_model")
    assert isinstance(resolved_model, mod.ResolvedModel)
    assert resolved_model.model_path == "/models/eye_masks_unet.pt"


def test_run_unet_forwards_roi_cache_args(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def _fake_infer(argv: list[str] | None = None) -> None:
        captured["argv"] = list(argv or [])

    monkeypatch.setattr(mod, "infer_unet_eye_masks_main", _fake_infer)
    monkeypatch.setattr(mod, "_latest_run", lambda *_args, **_kwargs: "eye_masks_unet_003")

    run_name = mod._run_unet(
        tmp_path / "recording_analysis.zarr",
        {
            "eye_masks": {
                "checkpoint": "/tmp/unet.pt",
                "batch_size": 32,
                "mask_probs_chunk_rois": 96,
                "mask_probs_dtype": "uint8",
                "roi_cache_policy": "always",
                "roi_cache_dir": "/tmp/unet-roi-cache",
            }
        },
    )

    assert run_name == "eye_masks_unet_003"
    argv = captured.get("argv")
    assert isinstance(argv, list)
    assert "--checkpoint" in argv
    assert "--mask-probs-chunk-rois" in argv
    assert "--mask-probs-dtype" in argv
    assert "--roi-cache-policy" in argv
    assert "--roi-cache-dir" in argv
    assert "96" in argv
    assert "uint8" in argv
    assert "always" in argv
    assert "/tmp/unet-roi-cache" in argv


def test_main_refine_missing_only_requires_refine_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(mod, "_resolve_root", lambda _paths: [tmp_path])
    monkeypatch.setattr(mod, "_load_config", lambda _path: {"eye_masks": {"method": "traditional"}})

    with pytest.raises(SystemExit, match="Use --refine-missing-only only with --refine-only."):
        mod.main(["--refine-missing-only", "--no-log", str(tmp_path)])


def test_main_model_source_registry_rejects_traditional_segmentation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "registry.sqlite"
    registry_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(mod, "_resolve_root", lambda _paths: [tmp_path])
    monkeypatch.setattr(mod, "_load_config", lambda _path: {"eye_masks": {"method": "traditional"}})

    with pytest.raises(SystemExit, match="--model-source registry requires model-based eye mask method"):
        mod.main(
            [
                "--apply",
                "--no-log",
                "--model-source",
                "registry",
                "--registry",
                str(registry_path),
                str(tmp_path),
            ]
        )


def test_decode_source_runs_prefers_canonical_with_legacy_fallback() -> None:
    legacy = mod._decode_source_runs({"source_keypoint_run": "kp_legacy"})
    canonical = mod._decode_source_runs(
        {
            "source_keypoints_run": "kp_new",
            "source_keypoint_run": "kp_old",
            "source_crop_run": "crop_001",
        }
    )

    assert legacy["source_keypoints_run"] == "kp_legacy"
    assert canonical["source_keypoints_run"] == "kp_new"
    assert canonical["source_crop_run"] == "crop_001"


def test_latest_run_reads_live_metadata_when_consolidated_is_stale(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    _create_latest_run(root, "eye_masks_runs", "eye_masks_001")
    zarr.consolidate_metadata(str(zarr_path))

    eye_masks_parent = root["eye_masks_runs"]
    eye_masks_parent.create_group("eye_masks_002")
    eye_masks_parent.attrs["latest"] = "eye_masks_002"

    assert mod._latest_run(zarr_path, "eye_masks_runs") == "eye_masks_002"


def test_collect_stage_payload_reads_live_run_when_consolidated_is_stale(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    parent = root.require_group("eye_masks_runs")
    run_1 = parent.create_group("eye_masks_001")
    run_1.attrs["method"] = "traditional"
    parent.attrs["latest"] = "eye_masks_001"
    zarr.consolidate_metadata(str(zarr_path))

    run_2 = parent.create_group("eye_masks_002")
    run_2.attrs.update(
        {
            "method": "traditional_eye_segmentation",
            "source_crop_run": "crop_001",
            "source_keypoints_run": "kp_001",
        }
    )
    parent.attrs["latest"] = "eye_masks_002"

    payload = mod._collect_stage_payload(zarr_path, "eye_masks_runs", "eye_masks_002")
    assert payload["run_name"] == "eye_masks_002"
    assert payload["method"] == "traditional_eye_segmentation"
    assert payload["source_runs"]["source_crop_run"] == "crop_001"
    assert payload["source_runs"]["source_keypoints_run"] == "kp_001"
