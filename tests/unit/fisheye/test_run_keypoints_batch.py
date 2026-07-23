from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.utils import run_keypoints_batch as mod


def _plan_from_presence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    crop_present: bool,
    background_present: bool,
    require_background: bool,
) -> mod.KeypointPlan:
    zarr_path = tmp_path / "recording_training.zarr"
    zarr_path.touch()
    recording_dir = tmp_path / "recording"
    h5_path = recording_dir / "raw" / "recording.h5"

    monkeypatch.setattr(mod.zarr, "open", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(mod, "_has_crop", lambda _root: crop_present)
    monkeypatch.setattr(mod, "_has_background", lambda _root: background_present)
    monkeypatch.setattr(mod, "_has_keypoints", lambda _root: False)
    monkeypatch.setattr(mod, "_has_keypoint_tuning", lambda _root: False)

    return mod._plan_from_zarr(
        zarr_path=zarr_path,
        recording_dir=recording_dir,
        h5_path=h5_path,
        camera_id="1",
        skip_existing=True,
        require_crop=True,
        require_background=require_background,
        require_tuning=False,
        refine_only=False,
    )


def test_collect_stage_payload_reads_live_run_when_consolidated_is_stale(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_training.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    parent = root.require_group("keypoints_runs")
    run_1 = parent.create_group("keypoints_001")
    run_1.attrs["method"] = "traditional_pose"
    parent.attrs["latest"] = "keypoints_001"
    zarr.consolidate_metadata(str(zarr_path))

    run_2 = parent.create_group("keypoints_002")
    run_2.attrs.update(
        {
            "method": "traditional_pose",
            "source_crop_run": "crop_001",
            "source_detect_run": "refined_detect_001",
            "summary_statistics": {
                "total_rois": 12,
                "successful_detections": 11,
                "success_rate_percent": 91.67,
            },
        }
    )
    parent.attrs["latest"] = "keypoints_002"

    payload = mod._collect_stage_payload(zarr_path, "keypoints_runs", "keypoints_002")

    assert payload["run_name"] == "keypoints_002"
    assert payload["method"] == "traditional_pose"
    assert payload["source_runs"]["source_crop_run"] == "crop_001"
    assert payload["source_runs"]["source_detect_run"] == "refined_detect_001"
    assert payload["summary_statistics"]["total_rois"] == 12


def test_plan_from_zarr_marks_missing_when_background_required(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = _plan_from_presence(
        monkeypatch,
        tmp_path,
        crop_present=True,
        background_present=False,
        require_background=True,
    )

    assert plan.status == "missing"
    assert plan.reason == "background missing"
    assert plan.crop_present is True
    assert plan.background_present is False


def test_plan_from_zarr_allows_missing_background_when_not_required(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = _plan_from_presence(
        monkeypatch,
        tmp_path,
        crop_present=True,
        background_present=False,
        require_background=False,
    )

    assert plan.status == "ok"
    assert plan.reason is None
    assert plan.crop_present is True
    assert plan.background_present is False


def test_plan_from_zarr_yolo_accepts_geometry_only_latest_any(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    crop_parent = root.require_group("crop_runs")
    crop_parent.attrs["latest_any"] = "crop_geometry"
    crop_geometry = crop_parent.create_group("crop_geometry")
    crop_geometry.attrs["crop_storage_mode"] = "geometry_only"

    plan = mod._plan_from_zarr(
        zarr_path=zarr_path,
        recording_dir=tmp_path,
        h5_path=tmp_path / "raw" / "recording.h5",
        camera_id="1",
        skip_existing=True,
        require_crop=True,
        require_background=False,
        require_tuning=False,
        refine_only=False,
        crop_storage_requirement="any",
    )

    assert plan.status == "ok"
    assert plan.crop_present is True


def test_plan_from_zarr_traditional_requires_latest_materialized(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    crop_parent = root.require_group("crop_runs")
    crop_parent.attrs["latest_any"] = "crop_geometry"
    crop_geometry = crop_parent.create_group("crop_geometry")
    crop_geometry.attrs["crop_storage_mode"] = "geometry_only"

    plan = mod._plan_from_zarr(
        zarr_path=zarr_path,
        recording_dir=tmp_path,
        h5_path=tmp_path / "raw" / "recording.h5",
        camera_id="1",
        skip_existing=True,
        require_crop=True,
        require_background=False,
        require_tuning=False,
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
        camera_id="1",
        skip_existing=True,
        require_crop=True,
        require_background=False,
        require_tuning=False,
        refine_only=False,
        crop_storage_requirement="materialized",
    )

    assert plan.status == "ok"
    assert plan.crop_present is True


@pytest.mark.parametrize(
    ("config_method", "cli_flags", "expected_require_background"),
    [
        ("traditional", [], True),
        ("yolo", [], False),
        ("yolo", ["--require-background"], True),
        ("traditional", ["--no-require-background"], False),
    ],
)
def test_main_background_requirement_defaults_by_method_and_honors_cli_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    config_method: str,
    cli_flags: list[str],
    expected_require_background: bool,
) -> None:
    captured: dict[str, bool] = {}

    def _capture_build_plans(*_args, **kwargs):  # noqa: ANN002, ANN003
        captured["require_background"] = kwargs["require_background"]
        return []

    monkeypatch.setattr(mod, "_resolve_root", lambda _paths: [tmp_path])
    monkeypatch.setattr(mod, "_load_config", lambda _path: {"keypoints": {"method": config_method}})
    monkeypatch.setattr(mod, "_build_plans", _capture_build_plans)
    monkeypatch.setattr(mod, "_build_plans_from_zarr", lambda *args, **kwargs: [])  # noqa: ARG005

    rc = mod.main([*cli_flags, "--no-log", str(tmp_path)])
    assert rc == 0
    assert captured["require_background"] is expected_require_background


def test_run_plan_rejects_refine_before_keypoint_detection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    called: list[str] = []

    def _traditional(*args, **kwargs):  # noqa: ANN002, ANN003
        called.append("traditional")
        return {"success_rate_percent": 99.0}

    def _refine(*args, **kwargs):  # noqa: ANN002, ANN003
        called.append("refine")
        return "refined_keypoints_001"

    def _collect(_zarr_path: Path, group_name: str, run_name: str) -> dict:
        return {
            "group": group_name,
            "run_name": run_name,
            "method": "mock",
            "summary_statistics": {"total_rois": 10},
            "source_runs": {"source_crop_run": "crop_001"},
        }

    monkeypatch.setattr(mod, "_run_traditional", _traditional)
    monkeypatch.setattr(mod, "_latest_keypoints_run", lambda _zarr_path: "keypoints_001")
    monkeypatch.setattr(mod, "_keypoints_total_rois", lambda _zarr_path, _run_name: 10)
    monkeypatch.setattr(mod, "_run_refine", _refine)
    monkeypatch.setattr(mod, "_collect_stage_payload", _collect)
    monkeypatch.setattr(
        mod,
        "_sync_keypoint_registry_rows_after_run",
        lambda **kwargs: {"synced": True, "dataset_id": "dataset_x"},  # noqa: ANN003
    )

    plan = mod.KeypointPlan(
        recording_dir=tmp_path,
        h5_path=tmp_path / "raw" / "recording.h5",
        zarr_path=tmp_path / "zarr" / "recording_training.zarr",
        camera_id="3",
        status="ok",
    )
    with pytest.raises(mod.RefinedKeypointCoordinatePublicationUnavailable):
        mod._run_plan(
            plan,
            config={},
            method="traditional",
            scheduler=None,
            num_workers=None,
            quiet=True,
            dask_progress=False,
            refine=True,
            refine_only=False,
            json_output=False,
        )

    assert called == []


def test_run_plan_refine_only_fails_before_archive_discovery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(mod, "_latest_keypoints_run", lambda _zarr_path: "keypoints_001")
    monkeypatch.setattr(mod, "_keypoints_total_rois", lambda _zarr_path, _run_name: 0)
    monkeypatch.setattr(
        mod,
        "_run_refine",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("_run_refine should not be called")),  # noqa: ANN002, ANN003
    )

    plan = mod.KeypointPlan(
        recording_dir=tmp_path,
        h5_path=tmp_path / "raw" / "recording.h5",
        zarr_path=tmp_path / "zarr" / "recording_training.zarr",
        camera_id="3",
        status="ok",
    )
    with pytest.raises(mod.RefinedKeypointCoordinatePublicationUnavailable):
        mod._run_plan(
            plan,
            config={},
            method="traditional",
            scheduler=None,
            num_workers=None,
            quiet=True,
            dask_progress=False,
            refine=False,
            refine_only=True,
            json_output=False,
        )


def test_run_plan_auto_review_syncs_step_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        mod,
        "require_future_normal_refined_keypoint_publication",
        lambda: None,
    )
    monkeypatch.setattr(mod, "_run_traditional", lambda *args, **kwargs: {"success_rate_percent": 99.0})  # noqa: ANN002, ANN003
    monkeypatch.setattr(mod, "_latest_keypoints_run", lambda _zarr_path: "keypoints_001")
    monkeypatch.setattr(mod, "_keypoints_total_rois", lambda _zarr_path, _run_name: 10)
    monkeypatch.setattr(mod, "_run_refine", lambda *args, **kwargs: "refined_keypoints_001")  # noqa: ANN002, ANN003

    def _collect(_zarr_path: Path, group_name: str, run_name: str) -> dict:
        if group_name == "keypoints_runs":
            return {
                "group": group_name,
                "run_name": run_name,
                "method": "yolo_pose",
                "summary_statistics": {"total_rois": 10},
                "source_runs": {},
            }
        return {
            "group": group_name,
            "run_name": run_name,
            "method": "refine_keypoints",
            "summary_statistics": {"total_rois": 10, "refined_success": 10, "usable_keypoints": 10, "pass_rate_percent": 100.0},
            "source_runs": {
                "source_keypoints_run": "keypoints_001",
                "source_crop_run": "crop_001",
                "source_detect_run": "refined_detect_001",
            },
        }

    monkeypatch.setattr(mod, "_collect_stage_payload", _collect)
    monkeypatch.setattr(
        mod,
        "_maybe_auto_approve_refined_keypoints",
        lambda **kwargs: {  # noqa: ANN003
            "enabled": True,
            "applied": True,
            "reason": "threshold_met",
            "review_status": {
                "state": "approved",
                "method": "algorithmic",
                "intended_use": "full_recording",
            },
        },
    )

    captured: dict[str, object] = {}

    def _sync(**kwargs):  # noqa: ANN003
        captured.update(kwargs)
        return {"synced": True}

    monkeypatch.setattr(mod, "_sync_refined_keypoint_step_status_after_auto_review", _sync)

    plan = mod.KeypointPlan(
        recording_dir=tmp_path,
        h5_path=tmp_path / "raw" / "recording.h5",
        zarr_path=tmp_path / "zarr" / "recording_training.zarr",
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
        dask_progress=False,
        refine=True,
        refine_only=False,
        json_output=False,
        auto_approve_min_usable_rate=1.0,
    )

    assert result["auto_review"]["step_status_sync"]["synced"] is True
    assert captured["refined_run"] == "refined_keypoints_001"
    assert captured["zarr_path"] == plan.zarr_path


def test_main_logs_rich_keypoint_results_to_jsonl(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    recording_dir = tmp_path / "recording_a"
    plan = mod.KeypointPlan(
        recording_dir=recording_dir,
        h5_path=recording_dir / "raw" / "recording_a.h5",
        zarr_path=recording_dir / "zarr" / "recording_a_training.zarr",
        camera_id="1",
        status="ok",
    )
    run_payload = {
        "recording": str(recording_dir),
        "zarr": str(plan.zarr_path),
        "camera_id": "1",
        "status": "ok",
        "method": "traditional",
        "keypoints": {
            "group": "keypoints_runs",
            "run_name": "keypoints_001",
            "summary_statistics": {"total_rois": 10, "success_rate_percent": 90.0},
        },
        "refined_keypoints": {
            "group": "refined_keypoints_runs",
            "run_name": "refined_keypoints_001",
            "summary_statistics": {"total_rois": 10, "usable_keypoints": 9},
        },
    }

    monkeypatch.setattr(mod, "_resolve_root", lambda _paths: [tmp_path])
    monkeypatch.setattr(mod, "_load_config", lambda _path: {"keypoints": {"method": "traditional"}})
    monkeypatch.setattr(mod, "_build_plans", lambda *args, **kwargs: [plan])  # noqa: ARG005
    monkeypatch.setattr(mod, "_build_plans_from_zarr", lambda *args, **kwargs: [])  # noqa: ARG005
    monkeypatch.setattr(mod, "_run_plan", lambda *args, **kwargs: run_payload)  # noqa: ARG005

    log_dir = tmp_path / "logs"
    rc = mod.main(["--apply", "--quiet", "--log-dir", str(log_dir), str(tmp_path)])
    assert rc == 0

    log_files = sorted(log_dir.glob("run_keypoints_batch_*.jsonl"))
    assert log_files, "expected keypoints batch log file"
    entries = [
        json.loads(line)
        for line in log_files[-1].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    keypoints_ok = [entry for entry in entries if entry.get("event") == "keypoints_ok"]
    assert len(keypoints_ok) == 1
    assert keypoints_ok[0]["results"]["keypoints"]["run_name"] == "keypoints_001"
    assert keypoints_ok[0]["results"]["refined_keypoints"]["run_name"] == "refined_keypoints_001"


def test_main_refine_only_fails_before_delegation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, list[str]] = {}

    def _delegate(argv: list[str]) -> int:
        captured["argv"] = list(argv)
        return 0

    monkeypatch.setattr(mod.refine_keypoints_batch_mod, "main", _delegate)

    file_list = tmp_path / "targets.txt"
    file_list.write_text("# test\n", encoding="utf-8")
    rc = mod.main(
        [
            "--refine-only",
            "--recursive",
            "--apply",
            "--scheduler",
            "single-threaded",
            "--num-workers",
            "3",
            "--no-log",
            "--json",
            "--file-list",
            str(file_list),
            str(tmp_path),
        ]
    )
    assert rc == 2
    stderr = capsys.readouterr().err
    assert "disabled for future-normal processing" in stderr
    assert captured == {}


# ---------------------------------------------------------------------------
# Registry discovery tests
# ---------------------------------------------------------------------------


def _make_fake_row(zarr_path: str) -> dict:
    return {"zarr_path": zarr_path}


def test_discover_zarrs_from_registry_skip_existing_passes_exclude_step_ok(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When skip_existing=True, exclude_step_ok='keypoints' is passed."""
    registry_path = tmp_path / "registry.sqlite"
    registry_path.write_text("", encoding="utf-8")

    captured_kwargs: list[dict] = []

    class _FakeRegistry:
        def __init__(self, _path):
            pass

        def query_datasets(self, **kwargs):
            captured_kwargs.append(kwargs)
            return []

        def close(self):
            pass

    monkeypatch.setattr(mod, "Registry", _FakeRegistry)

    mod._discover_zarrs_from_registry(
        registry_path=registry_path,
        scope_paths=[],
        skip_existing=True,
    )

    assert len(captured_kwargs) == 1
    assert captured_kwargs[0]["exclude_step_ok"] == "keypoints"
    assert captured_kwargs[0]["require_steps_ok"] == ["detect", "crop"]


def test_discover_zarrs_from_registry_no_skip_omits_exclude_step_ok(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When skip_existing=False (default), exclude_step_ok is not passed."""
    registry_path = tmp_path / "registry.sqlite"
    registry_path.write_text("", encoding="utf-8")

    captured_kwargs: list[dict] = []

    class _FakeRegistry:
        def __init__(self, _path):
            pass

        def query_datasets(self, **kwargs):
            captured_kwargs.append(kwargs)
            return []

        def close(self):
            pass

    monkeypatch.setattr(mod, "Registry", _FakeRegistry)

    mod._discover_zarrs_from_registry(
        registry_path=registry_path,
        scope_paths=[],
        skip_existing=False,
    )

    assert len(captured_kwargs) == 1
    assert "exclude_step_ok" not in captured_kwargs[0]
    assert captured_kwargs[0]["require_steps_ok"] == ["detect", "crop"]


def test_main_source_registry_missing_registry_fails(tmp_path: Path) -> None:
    """--source registry with missing registry file returns exit code 1."""
    rc = mod.main(
        [
            "--source",
            "registry",
            "--registry",
            str(tmp_path / "nonexistent.sqlite"),
            "--no-log",
            str(tmp_path),
        ]
    )
    assert rc == 1


def test_main_emit_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--emit-paths prints discovered paths and exits 0."""
    registry_path = tmp_path / "registry.sqlite"
    registry_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        mod,
        "_discover_registry_entries",
        lambda **_kw: [
            mod.RegistryZarrEntry(zarr_path=Path("/data/rec_a_analysis.zarr"), camera_id="1"),
            mod.RegistryZarrEntry(zarr_path=Path("/data/rec_b_analysis.zarr"), camera_id="2"),
        ],
    )

    rc = mod.main(
        [
            "--source",
            "registry",
            "--emit-paths",
            "--registry",
            str(registry_path),
            "--no-log",
            str(tmp_path),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    lines = [l for l in out.strip().splitlines() if l.strip()]
    assert "/data/rec_a_analysis.zarr" in lines
    assert "/data/rec_b_analysis.zarr" in lines


def test_main_source_registry_uses_registry_camera_ids_in_dry_run_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    registry_path = tmp_path / "registry.sqlite"
    registry_path.write_text("", encoding="utf-8")
    recording_dir = tmp_path / "recording_a"
    zarr_path = recording_dir / "zarr" / "recording_a_analysis.zarr"
    discovered = [mod.RegistryZarrEntry(zarr_path=zarr_path, camera_id="7")]

    def _fake_build_plans_from_zarr(*args, **kwargs):  # noqa: ANN002, ANN003
        return [
            mod.KeypointPlan(
                recording_dir=recording_dir,
                h5_path=None,
                zarr_path=zarr_path,
                camera_id=None,
                status="ok",
            )
        ]

    monkeypatch.setattr(mod, "_discover_registry_entries", lambda **_kw: discovered)
    monkeypatch.setattr(mod, "_load_config", lambda _path: {"keypoints": {"method": "yolo"}})
    monkeypatch.setattr(mod, "_build_plans_from_zarr", _fake_build_plans_from_zarr)

    rc = mod.main(
        [
            "--source",
            "registry",
            "--method",
            "yolo",
            "--registry",
            str(registry_path),
            "--dry-run",
            "--json",
            "--no-log",
            str(tmp_path),
        ]
    )
    assert rc == 0
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    assert lines
    payload = json.loads(lines[0])
    assert payload["camera_id"] == "7"


def test_run_yolo_prefers_model_path_override(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _fake_detect_keypoints_yolo(**kwargs):  # noqa: ANN003
        captured.update(kwargs)
        return "keypoints_001"

    monkeypatch.setattr(mod, "detect_keypoints_yolo", _fake_detect_keypoints_yolo)
    run_name = mod._run_yolo(
        "recording_analysis.zarr",
        {
            "keypoints": {
                "model": "/models/from_config.pt",
                "batch_size": 64,
                "roi_cache_policy": "always",
                "roi_cache_dir": "/tmp/roi-cache",
            }
        },
        quiet=False,
        model_path_override="/models/from_registry.pt",
    )
    assert run_name == "keypoints_001"
    assert captured["model_path"] == "/models/from_registry.pt"
    assert captured["roi_cache_policy"] == "always"
    assert captured["roi_cache_dir"] == "/tmp/roi-cache"
    assert captured["keypoint_roi_shard_rows"] == 131_072
    assert captured["keypoint_frame_shard_rows"] == 131_072


def test_run_yolo_allows_regular_chunk_storage_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        mod,
        "detect_keypoints_yolo",
        lambda **kwargs: captured.update(kwargs) or "keypoints_001",
    )

    mod._run_yolo(
        "recording_analysis.zarr",
        {
            "keypoints": {
                "model": "/models/pose.pt",
                "keypoint_roi_shard_rows": None,
            }
        },
        quiet=False,
    )

    assert captured["keypoint_roi_shard_rows"] is None
    assert captured["keypoint_frame_shard_rows"] == 131_072


def test_resolve_registry_models_for_plans_collects_resolution_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan_ok = mod.KeypointPlan(
        recording_dir=tmp_path / "rec_ok",
        h5_path=None,
        zarr_path=tmp_path / "rec_ok" / "zarr" / "rec_ok_analysis.zarr",
        camera_id=None,
        status="ok",
    )
    plan_fail = mod.KeypointPlan(
        recording_dir=tmp_path / "rec_fail",
        h5_path=None,
        zarr_path=tmp_path / "rec_fail" / "zarr" / "rec_fail_analysis.zarr",
        camera_id=None,
        status="ok",
    )

    def _fake_resolve(**kwargs):  # noqa: ANN003
        plan = kwargs["plan"]
        if plan.recording_dir.name == "rec_fail":
            raise RuntimeError("resolution boom")
        return mod.ResolvedModel(
            model_path="/models/pose.pt",
            payload={"selected": {"run_id": "pose_run_001", "set_id": "pose_set_001", "model_path": "/models/pose.pt"}},
        )

    monkeypatch.setattr(mod, "_resolve_registry_model_for_plan", _fake_resolve)
    resolved, errors = mod._resolve_registry_models_for_plans(
        plans=[plan_ok, plan_fail],
        registry_path=tmp_path / "registry.sqlite",
        set_id_filter=None,
        require_unique=False,
        top_k=5,
        include_non_success=False,
        config={"keypoints": {}},
    )
    assert str(plan_ok.zarr_path.resolve()) in resolved
    assert str(plan_fail.zarr_path.resolve()) in errors
    assert "resolution boom" in errors[str(plan_fail.zarr_path.resolve())]


def test_resolve_registry_models_for_plans_resolves_once_per_recording(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recording_dir = tmp_path / "rec_shared"
    plan_a = mod.KeypointPlan(
        recording_dir=recording_dir,
        h5_path=None,
        zarr_path=recording_dir / "zarr" / "rec_shared_analysis.zarr",
        camera_id=None,
        status="ok",
    )
    plan_b = mod.KeypointPlan(
        recording_dir=recording_dir,
        h5_path=None,
        zarr_path=recording_dir / "zarr" / "rec_shared_training.zarr",
        camera_id=None,
        status="ok",
    )
    calls: list[str] = []

    def _fake_resolve(**kwargs):  # noqa: ANN003
        plan = kwargs["plan"]
        calls.append(str(plan.zarr_path))
        return mod.ResolvedModel(
            model_path="/models/pose.pt",
            payload={"selected": {"run_id": "pose_run_001", "set_id": "pose_set_001", "model_path": "/models/pose.pt"}},
        )

    monkeypatch.setattr(mod, "_resolve_registry_model_for_plan", _fake_resolve)
    resolved, errors = mod._resolve_registry_models_for_plans(
        plans=[plan_a, plan_b],
        registry_path=tmp_path / "registry.sqlite",
        set_id_filter=None,
        require_unique=False,
        top_k=5,
        include_non_success=False,
        config={"keypoints": {}},
    )
    assert not errors
    assert len(calls) == 1
    assert str(plan_a.zarr_path.resolve()) in resolved
    assert str(plan_b.zarr_path.resolve()) in resolved
    assert resolved[str(plan_a.zarr_path.resolve())] == resolved[str(plan_b.zarr_path.resolve())]


def test_resolve_registry_models_for_plans_emits_progress_events(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recording_dir = tmp_path / "rec_shared"
    plan_a = mod.KeypointPlan(
        recording_dir=recording_dir,
        h5_path=None,
        zarr_path=recording_dir / "zarr" / "rec_shared_analysis.zarr",
        camera_id="1",
        status="ok",
    )
    plan_b = mod.KeypointPlan(
        recording_dir=recording_dir,
        h5_path=None,
        zarr_path=recording_dir / "zarr" / "rec_shared_training.zarr",
        camera_id="1",
        status="ok",
    )
    events: list[dict[str, object]] = []

    def _fake_resolve(**kwargs):  # noqa: ANN003
        return mod.ResolvedModel(
            model_path="/models/pose.pt",
            payload={"selected": {"run_id": "pose_run_001", "set_id": "pose_set_001", "model_path": "/models/pose.pt"}},
        )

    monkeypatch.setattr(mod, "_resolve_registry_model_for_plan", _fake_resolve)
    resolved, errors = mod._resolve_registry_models_for_plans(
        plans=[plan_a, plan_b],
        registry_path=tmp_path / "registry.sqlite",
        set_id_filter=None,
        require_unique=False,
        top_k=5,
        include_non_success=False,
        config={"keypoints": {}},
        on_event=lambda payload: events.append(payload),
    )
    assert not errors
    assert str(plan_a.zarr_path.resolve()) in resolved
    assert str(plan_b.zarr_path.resolve()) in resolved
    assert [event["event"] for event in events] == [
        "model_resolution_start",
        "model_resolution_ok",
        "model_resolution_cached",
    ]
    assert events[0]["index"] == 1
    assert events[2]["index"] == 2


def test_main_model_source_registry_passes_pre_resolved_model_to_run_plan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registry_path = tmp_path / "registry.sqlite"
    registry_path.write_text("", encoding="utf-8")
    recording_dir = tmp_path / "recording_a"
    plan = mod.KeypointPlan(
        recording_dir=recording_dir,
        h5_path=recording_dir / "raw" / "recording_a.h5",
        zarr_path=recording_dir / "zarr" / "recording_a_analysis.zarr",
        camera_id="1",
        status="ok",
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(mod, "_resolve_root", lambda _paths: [tmp_path])
    monkeypatch.setattr(mod, "_load_config", lambda _path: {"keypoints": {"method": "yolo"}})
    monkeypatch.setattr(mod, "_build_plans", lambda *args, **kwargs: [plan])  # noqa: ARG005
    monkeypatch.setattr(mod, "_build_plans_from_zarr", lambda *args, **kwargs: [])  # noqa: ARG005
    monkeypatch.setattr(
        mod,
        "_resolve_registry_models_for_plans",
        lambda **kwargs: (  # noqa: ARG005
            {
                str(plan.zarr_path.resolve()): mod.ResolvedModel(
                    model_path="/models/resolved.pt",
                    payload={
                        "selected": {
                            "run_id": "pose_run_001",
                            "set_id": "pose_set_001",
                            "model_path": "/models/resolved.pt",
                        }
                    },
                )
            },
            {},
        ),
    )

    def _fake_run_plan(*args, **kwargs):  # noqa: ANN002, ANN003
        captured["resolved_model"] = kwargs.get("resolved_model")
        return {
            "recording": str(recording_dir),
            "zarr": str(plan.zarr_path),
            "status": "ok",
            "method": "yolo",
        }

    monkeypatch.setattr(mod, "_run_plan", _fake_run_plan)

    rc = mod.main(
        [
            "--apply",
            "--model-source",
            "registry",
            "--registry",
            str(registry_path),
            "--no-log",
            str(tmp_path),
        ]
    )
    assert rc == 0
    resolved_model = captured.get("resolved_model")
    assert isinstance(resolved_model, mod.ResolvedModel)
    assert resolved_model.model_path == "/models/resolved.pt"


def test_main_model_source_registry_requires_yolo_method(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(mod, "_load_config", lambda _path: {"keypoints": {"method": "traditional"}})
    rc = mod.main(
        [
            "--apply",
            "--model-source",
            "registry",
            "--no-log",
            str(tmp_path),
        ]
    )
    assert rc == 1


def test_maybe_auto_approve_refined_keypoints_applies_when_threshold_met(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    calls: list[dict] = []

    def _fake_apply_auto_review(path: Path, **kwargs):  # noqa: ANN003
        assert path == zarr_path
        calls.append(dict(kwargs))
        if kwargs["dry_run"]:
            return {
                "skipped": False,
                "passed": True,
                "payload": {
                    "auto_review": {
                        "evidence": {
                            "total_rois": 8,
                            "usable_keypoints": 8,
                            "usable_keypoints_rate": 1.0,
                        }
                    }
                },
            }
        payload = {
            "state": "approved",
            "method": "algorithmic",
            "intended_use": "full_recording",
            "reviewer": "auto-batch",
            "auto_review": {
                "policy_id": "keypoint_auto_review_v1",
                "policy_version": 1,
            },
        }
        return {"skipped": False, "payload": payload}

    monkeypatch.setattr(mod, "apply_auto_review", _fake_apply_auto_review)

    result = mod._maybe_auto_approve_refined_keypoints(
        zarr_path=zarr_path,
        refined_run="refined_keypoints_001",
        min_usable_rate=1.0,
        intended_use="full_recording",
        reviewer="auto-batch",
    )

    assert result["applied"] is True
    assert result["reason"] == "threshold_met"
    assert result["usable_rate"] == pytest.approx(1.0)
    review_status = dict(result["review_status"])
    assert review_status["state"] == "approved"
    assert review_status["method"] == "algorithmic"
    assert review_status["intended_use"] == "full_recording"
    assert review_status["reviewer"] == "auto-batch"
    assert review_status["auto_review"]["policy_id"] == "keypoint_auto_review_v1"
    assert review_status["auto_review"]["policy_version"] == 1
    assert len(calls) == 2
    assert calls[0]["dry_run"] is True
    assert calls[1]["dry_run"] is False


def test_maybe_auto_approve_refined_keypoints_skips_when_threshold_not_met(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    calls: list[dict] = []

    def _fake_apply_auto_review(path: Path, **kwargs):  # noqa: ANN003
        assert path == zarr_path
        calls.append(dict(kwargs))
        assert kwargs["dry_run"] is True
        return {
            "skipped": False,
            "passed": False,
            "payload": {
                "auto_review": {
                    "evidence": {
                        "total_rois": 8,
                        "usable_keypoints": 7,
                        "usable_keypoints_rate": 7.0 / 8.0,
                    }
                }
            },
        }

    monkeypatch.setattr(mod, "apply_auto_review", _fake_apply_auto_review)

    result = mod._maybe_auto_approve_refined_keypoints(
        zarr_path=zarr_path,
        refined_run="refined_keypoints_001",
        min_usable_rate=1.0,
        intended_use="full_recording",
        reviewer=None,
    )

    assert result["applied"] is False
    assert result["reason"] == "threshold_not_met"
    assert result["usable_rate"] == pytest.approx(7.0 / 8.0)
    assert len(calls) == 1


def test_main_auto_approve_requires_refine_flag(tmp_path: Path) -> None:
    rc = mod.main(
        [
            "--apply",
            "--auto-approve-perfect",
            "--no-log",
            str(tmp_path),
        ]
    )
    assert rc == 1


def test_run_plan_retry_failed_only_routes_to_retry_helper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        mod,
        "require_future_normal_refined_keypoint_publication",
        lambda: None,
    )
    captured_retry_kwargs: dict[str, object] = {}
    captured_provenance: dict[str, object] = {}

    def _fake_retry(*args, **kwargs):  # noqa: ANN002, ANN003
        captured_retry_kwargs.update(kwargs)
        return {
            "run_name": "keypoints_retry_001",
            "source_keypoints_run": "keypoints_001",
            "source_refined_run": "refined_keypoints_001",
            "retry_target_count": 7,
            "retry_replaced_count": 6,
            "retry_selector": "failed_only",
            "retry_policy": "replace_on_success_only",
            "updated": True,
            "created_new_run": True,
            "reused_existing": False,
        }

    def _fake_collect(_zarr_path: Path, group_name: str, run_name: str) -> dict:
        if group_name == "keypoints_runs":
            return {
                "group": group_name,
                "run_name": run_name,
                "method": "yolo_pose",
                "summary_statistics": {"total_rois": 10},
                "source_runs": {},
            }
        return {
            "group": group_name,
            "run_name": run_name,
            "method": "refine_keypoints",
            "summary_statistics": {"total_rois": 10, "usable_keypoints": 10},
            "source_runs": {"source_keypoints_run": "keypoints_retry_001"},
        }

    monkeypatch.setattr(mod, "_run_yolo_retry_failed_only", _fake_retry)
    monkeypatch.setattr(mod, "_collect_stage_payload", _fake_collect)
    monkeypatch.setattr(mod, "_keypoints_total_rois", lambda _zarr_path, _run_name: 10)
    monkeypatch.setattr(mod, "_run_refine", lambda *args, **kwargs: "refined_keypoints_retry_001")  # noqa: ANN002, ANN003
    monkeypatch.setattr(
        mod,
        "_sync_keypoint_registry_rows_after_run",
        lambda **kwargs: {"synced": True, "dataset_id": "dataset_retry"},  # noqa: ANN003
    )
    monkeypatch.setattr(
        mod,
        "write_keypoint_model_resolution_provenance",
        lambda **kwargs: captured_provenance.update(kwargs),  # noqa: ANN003
    )

    plan = mod.KeypointPlan(
        recording_dir=tmp_path / "recording",
        h5_path=None,
        zarr_path=tmp_path / "recording" / "zarr" / "recording_analysis.zarr",
        camera_id="7",
        status="ok",
    )
    resolved_model = mod.ResolvedModel(
        model_path="/models/pose_retry.pt",
        payload={"selected": {"model_path": "/models/pose_retry.pt", "run_id": "pose_run_001", "set_id": "pose_set_001"}},
    )
    result = mod._run_plan(
        plan,
        config={},
        method="yolo",
        scheduler=None,
        num_workers=None,
        quiet=True,
        dask_progress=False,
        refine=True,
        refine_only=False,
        json_output=False,
        resolved_model=resolved_model,
        retry_failed_only=True,
        retry_source_keypoints_run="keypoints_001",
        retry_refined_run="refined_keypoints_001",
        retry_include_fish_present_no_keypoints=True,
        retry_force_new=True,
    )

    assert captured_retry_kwargs["model_path_override"] == "/models/pose_retry.pt"
    assert captured_retry_kwargs["source_keypoints_run"] == "keypoints_001"
    assert captured_retry_kwargs["refined_run"] == "refined_keypoints_001"
    assert captured_retry_kwargs["include_fish_present_no_keypoints"] is True
    assert captured_retry_kwargs["force_new"] is True
    assert result["keypoint_retry"]["run_name"] == "keypoints_retry_001"
    assert result["keypoints"]["run_name"] == "keypoints_retry_001"
    assert result["refined_keypoints"]["run_name"] == "refined_keypoints_retry_001"
    assert result["registry_sync"]["synced"] is True
    assert captured_provenance["run_name"] == "keypoints_retry_001"


def test_main_retry_options_require_retry_failed_only(tmp_path: Path) -> None:
    rc = mod.main(
        [
            "--retry-refined-run",
            "refined_keypoints_001",
            "--no-log",
            str(tmp_path),
        ]
    )
    assert rc == 1


def test_main_retry_failed_only_requires_yolo_method(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(mod, "_load_config", lambda _path: {"keypoints": {"method": "traditional"}})
    rc = mod.main(
        [
            "--apply",
            "--retry-failed-only",
            "--no-log",
            str(tmp_path),
        ]
    )
    assert rc == 1


def test_main_retry_failed_only_forces_skip_existing_false(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def _capture_build_plans(*_args, **kwargs):  # noqa: ANN002, ANN003
        captured["skip_existing"] = kwargs["skip_existing"]
        return []

    monkeypatch.setattr(mod, "_resolve_root", lambda _paths: [tmp_path])
    monkeypatch.setattr(mod, "_load_config", lambda _path: {"keypoints": {"method": "yolo"}})
    monkeypatch.setattr(mod, "_build_plans", _capture_build_plans)
    monkeypatch.setattr(mod, "_build_plans_from_zarr", lambda *args, **kwargs: [])  # noqa: ARG005

    rc = mod.main(
        [
            "--retry-failed-only",
            "--no-log",
            str(tmp_path),
        ]
    )
    assert rc == 0
    assert captured["skip_existing"] is False


def test_main_retry_failed_only_passes_retry_args_to_run_plan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    recording_dir = tmp_path / "recording_a"
    plan = mod.KeypointPlan(
        recording_dir=recording_dir,
        h5_path=recording_dir / "raw" / "recording_a.h5",
        zarr_path=recording_dir / "zarr" / "recording_a_analysis.zarr",
        camera_id="1",
        status="ok",
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(mod, "_resolve_root", lambda _paths: [tmp_path])
    monkeypatch.setattr(mod, "_load_config", lambda _path: {"keypoints": {"method": "yolo"}})
    monkeypatch.setattr(mod, "_build_plans", lambda *args, **kwargs: [plan])  # noqa: ARG005
    monkeypatch.setattr(mod, "_build_plans_from_zarr", lambda *args, **kwargs: [])  # noqa: ARG005

    def _fake_run_plan(*args, **kwargs):  # noqa: ANN002, ANN003
        captured.update(kwargs)
        return {
            "recording": str(recording_dir),
            "zarr": str(plan.zarr_path),
            "status": "ok",
            "method": "yolo",
        }

    monkeypatch.setattr(mod, "_run_plan", _fake_run_plan)

    rc = mod.main(
        [
            "--apply",
            "--retry-failed-only",
            "--retry-source-keypoints-run",
            "keypoints_123",
            "--retry-refined-run",
            "refined_keypoints_123",
            "--retry-include-fish-present-no-keypoints",
            "--retry-force-new",
            "--no-log",
            str(tmp_path),
        ]
    )
    assert rc == 0
    assert captured["retry_failed_only"] is True
    assert captured["retry_source_keypoints_run"] == "keypoints_123"
    assert captured["retry_refined_run"] == "refined_keypoints_123"
    assert captured["retry_include_fish_present_no_keypoints"] is True
    assert captured["retry_force_new"] is True
