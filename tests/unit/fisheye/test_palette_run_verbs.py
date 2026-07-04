from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

import zarr

from fisheye.cli import palette
from fisheye.cli.envelope import build_run_provenance
from fisheye.shared.zarr_run_completion import (
    AUTHORITATIVE_RUN_ATTR,
    AUTHORITATIVE_RUN_PROVENANCE_ATTR,
    mark_run_complete,
    mark_run_started,
    require_runs_parent,
)
from fisheye.utils.crop_batch import CropPlan


def _run_json(capsys, *args: str) -> tuple[int, dict]:
    rc = palette.main([*args, "--json"])
    out = capsys.readouterr().out
    return rc, json.loads(out)


def _open_tmp_store(path: Path):
    return zarr.open_group(str(path), mode="w")


def _create_raw(root) -> None:
    raw = root.require_group("raw_video")
    if "images_full" not in raw:
        raw.create_array("images_full", shape=(1, 1, 1), dtype="uint8")


def _complete_run(root, parent_path: str, run_name: str):
    parent = root
    for part in parent_path.split("/"):
        parent = parent.require_group(part)
    run = parent.require_group(run_name)
    mark_run_complete(run, parent_group=parent, run_name=run_name)
    return run


def _runner_result(**payload):
    return SimpleNamespace(to_dict=lambda: dict(payload))


def test_run_provenance_config_hash_is_stable() -> None:
    left = build_run_provenance(command="palette detect", params={"a": 1, "path": Path("/tmp/a")})
    right = build_run_provenance(command="palette detect", params={"path": Path("/tmp/a"), "a": 1})
    changed = build_run_provenance(command="palette detect", params={"a": 2, "path": Path("/tmp/a")})

    assert left["config_hash"] == right["config_hash"]
    assert left["config_hash"] != changed["config_hash"]


def test_approve_dry_run_does_not_write_pointer(tmp_path, capsys) -> None:
    zarr_path = tmp_path / "approve_ready.zarr"
    root = _open_tmp_store(zarr_path)
    _create_raw(root)
    _complete_run(root, "subject_mask_runs", "subject_full")

    rc, payload = _run_json(
        capsys,
        "approve",
        str(zarr_path),
        "subject_masks",
        "subject_full",
        "--by",
        "jeremy",
    )

    assert rc == palette.EXIT_OK
    assert payload["status"] == "dry_run"
    assert payload["reason_code"] == "DRY_RUN"
    assert payload["run"] == "subject_full"
    assert payload["metrics"]["stage"] == "subject_masks"
    assert payload["metrics"]["parent_path"] == "subject_mask_runs"
    assert payload["approval"]["approved_by"] == "jeremy"
    assert "--apply" in payload["next_hints"][0]
    reopened = zarr.open_group(str(zarr_path), mode="r")
    assert AUTHORITATIVE_RUN_ATTR not in reopened["subject_mask_runs"].attrs


def test_approve_apply_writes_authoritative_pointer_and_status_uses_it(tmp_path, capsys) -> None:
    zarr_path = tmp_path / "approve_apply.zarr"
    root = _open_tmp_store(zarr_path)
    _create_raw(root)
    _complete_run(root, "subject_mask_runs", "subject_full")
    _complete_run(root, "subject_mask_runs", "subject_smoke")

    rc, payload = _run_json(
        capsys,
        "approve",
        str(zarr_path),
        "subject_masks",
        "subject_full",
        "--apply",
        "--by",
        "jeremy",
        "--note",
        "reviewed full run",
    )

    assert rc == palette.EXIT_OK
    assert payload["status"] == "ok"
    assert payload["reason_code"] == "OK"
    assert payload["metrics"]["previous_authoritative_run"] is None
    reopened = zarr.open_group(str(zarr_path), mode="r")
    parent = reopened["subject_mask_runs"]
    assert parent.attrs[AUTHORITATIVE_RUN_ATTR] == "subject_full"
    provenance = dict(parent.attrs[AUTHORITATIVE_RUN_PROVENANCE_ATTR])
    assert provenance["approved_by"] == "jeremy"
    assert provenance["note"] == "reviewed full run"
    assert "approved_at" in provenance

    rc, status_payload = _run_json(capsys, "status", str(zarr_path))

    assert rc == palette.EXIT_OK
    subject_masks = next(stage for stage in status_payload["stages"] if stage["stage"] == "subject_masks")
    assert subject_masks["run"] == "subject_full"
    assert subject_masks["latest_complete_run"] == "subject_smoke"
    assert subject_masks["run_resolution"] == "authoritative"


def test_approve_blocks_incomplete_run(tmp_path, capsys) -> None:
    zarr_path = tmp_path / "approve_incomplete.zarr"
    root = _open_tmp_store(zarr_path)
    parent = require_runs_parent(root, "subject_mask_runs")
    run = parent.require_group("subject_pending")
    mark_run_started(run, run_name="subject_pending", stage="subject_masks")

    rc, payload = _run_json(
        capsys,
        "approve",
        str(zarr_path),
        "subject_masks",
        "subject_pending",
        "--apply",
    )

    assert rc == palette.EXIT_BLOCKED
    assert payload["status"] == "blocked"
    assert payload["reason_code"] == "RUN_NOT_COMPLETE"
    reopened = zarr.open_group(str(zarr_path), mode="r")
    assert AUTHORITATIVE_RUN_ATTR not in reopened["subject_mask_runs"].attrs


def test_approve_callable_writes_authoritative_pointer(tmp_path) -> None:
    zarr_path = tmp_path / "approve_callable.zarr"
    root = _open_tmp_store(zarr_path)
    _create_raw(root)
    _complete_run(root, "subject_mask_runs", "subject_full")

    payload = palette.approve(
        palette.ApproveRequest(
            recording=zarr_path,
            stage="subject_masks",
            run="subject_full",
            approved_by="jeremy",
            note="callable approval",
            apply=True,
        )
    )

    assert payload["status"] == "ok"
    assert payload["reason_code"] == "OK"
    assert payload["metrics"]["authoritative_run"] == "subject_full"
    reopened = zarr.open_group(str(zarr_path), mode="r")
    parent = reopened["subject_mask_runs"]
    assert parent.attrs[AUTHORITATIVE_RUN_ATTR] == "subject_full"
    provenance = dict(parent.attrs[AUTHORITATIVE_RUN_PROVENANCE_ATTR])
    assert provenance["approved_by"] == "jeremy"
    assert provenance["note"] == "callable approval"


def test_approve_callable_dry_run_does_not_write_pointer(tmp_path) -> None:
    zarr_path = tmp_path / "approve_callable_dry_run.zarr"
    root = _open_tmp_store(zarr_path)
    _create_raw(root)
    _complete_run(root, "subject_mask_runs", "subject_full")

    payload = palette.approve(
        palette.ApproveRequest(
            recording=zarr_path,
            stage="subject_masks",
            run="subject_full",
            approved_by="jeremy",
            apply=False,
        )
    )

    assert payload["status"] == "dry_run"
    assert payload["reason_code"] == "DRY_RUN"
    assert "--apply" in payload["next_hints"][0]
    reopened = zarr.open_group(str(zarr_path), mode="r")
    assert AUTHORITATIVE_RUN_ATTR not in reopened["subject_mask_runs"].attrs


def test_status_callable_returns_envelope(tmp_path) -> None:
    zarr_path = tmp_path / "status_callable.zarr"
    root = _open_tmp_store(zarr_path)
    _create_raw(root)

    payload = palette.status(palette.StatusRequest(recording=zarr_path))

    assert payload["command"] == "palette status"
    assert payload["status"] == "ok"
    assert payload["reason_code"] == "OK"
    raw = next(stage for stage in payload["stages"] if stage["stage"] == "raw")
    assert raw["state"] == "complete"


def test_plan_callable_returns_envelope(tmp_path) -> None:
    zarr_path = tmp_path / "plan_callable.zarr"
    root = _open_tmp_store(zarr_path)
    _create_raw(root)

    payload = palette.plan(palette.PlanRequest(recording=zarr_path))

    assert payload["command"] == "palette plan"
    assert payload["status"] == "ok"
    assert payload["reason_code"] == "OK"
    assert any(item["stage"] == "detect" for item in payload["next"])
    assert payload["next_hints"]


def test_detect_default_dry_run_uses_runner_without_writing(monkeypatch, tmp_path, capsys) -> None:
    zarr_path = tmp_path / "detect_ready.zarr"
    root = _open_tmp_store(zarr_path)
    _create_raw(root)
    _complete_run(root, "background_runs", "background_001")
    calls: dict = {}

    def fake_detect(**kwargs):
        calls.update(kwargs)
        return _runner_result(
            ok=True,
            status="dry_run",
            output_zarr=str(kwargs["output"]),
            detect_run=None,
            selected_model_path="/models/detect.pt",
            selected_run_id="detect_model_run",
            selected_set_id="detect_set",
        )

    import fisheye.utils.run_detect_with_registry_model as runner

    monkeypatch.setattr(runner, "run_detect_with_registry_model", fake_detect)

    rc, payload = _run_json(capsys, "detect", str(zarr_path))

    assert rc == palette.EXIT_OK
    assert payload["status"] == "dry_run"
    assert payload["reason_code"] == "DRY_RUN"
    assert payload["resolved_command"].startswith("palette detect")
    assert "--dry-run" in payload["resolved_command"]
    assert payload["next_hints"][0].startswith("palette detect")
    assert "--apply" in payload["next_hints"][0]
    assert calls["dry_run"] is True
    assert calls["output"] == zarr_path.resolve()
    assert calls["cli_provenance"] is None


def test_detect_apply_passes_cli_provenance(monkeypatch, tmp_path, capsys) -> None:
    zarr_path = tmp_path / "detect_apply.zarr"
    root = _open_tmp_store(zarr_path)
    _create_raw(root)
    _complete_run(root, "background_runs", "background_001")
    calls: dict = {}

    def fake_detect(**kwargs):
        calls.update(kwargs)
        return _runner_result(
            ok=True,
            status="ok",
            output_zarr=str(kwargs["output"]),
            detect_run="detect_palette_001",
            selected_model_path="/models/detect.pt",
            selected_run_id="detect_model_run",
            selected_set_id="detect_set",
        )

    import fisheye.utils.run_detect_with_registry_model as runner

    monkeypatch.setattr(runner, "run_detect_with_registry_model", fake_detect)

    rc, payload = _run_json(capsys, "detect", str(zarr_path), "--apply")

    assert rc == palette.EXIT_OK
    assert payload["status"] == "ok"
    assert payload["run"] == "detect_palette_001"
    assert calls["dry_run"] is False
    assert calls["cli_provenance"]["command"] == "palette detect"
    assert calls["cli_provenance"]["config_hash"]


def test_detect_no_longer_blocks_on_missing_background(monkeypatch, tmp_path, capsys) -> None:
    zarr_path = tmp_path / "detect_raw_only.zarr"
    root = _open_tmp_store(zarr_path)
    _create_raw(root)
    calls: dict = {}

    def fake_detect(**kwargs):
        calls.update(kwargs)
        return _runner_result(
            ok=True,
            status="dry_run",
            output_zarr=str(kwargs["output"]),
            detect_run=None,
            selected_model_path="/models/detect.pt",
            selected_run_id="detect_model_run",
            selected_set_id="detect_set",
        )

    import fisheye.utils.run_detect_with_registry_model as runner

    monkeypatch.setattr(runner, "run_detect_with_registry_model", fake_detect)

    rc, payload = _run_json(capsys, "detect", str(zarr_path))

    assert rc == palette.EXIT_OK
    assert payload["status"] == "dry_run"
    assert payload["reason_code"] == "DRY_RUN"
    assert calls["dry_run"] is True
    assert calls["cli_provenance"] is None


def test_keypoints_apply_constructs_registry_runner_invocation(monkeypatch, tmp_path, capsys) -> None:
    zarr_path = tmp_path / "keypoints_ready.zarr"
    root = _open_tmp_store(zarr_path)
    _complete_run(root, "crop_runs", "crop_001")
    calls: dict = {}

    def fake_keypoints(**kwargs):
        calls.update(kwargs)
        return _runner_result(
            ok=True,
            status="ok",
            output_zarr=str(kwargs["output"]),
            keypoint_run="keypoints_palette_001",
            selected_model_path="/models/keypoints.pt",
            selected_run_id="keypoint_model_run",
            selected_set_id="keypoint_set",
        )

    fake_module = ModuleType("fisheye.utils.run_keypoints_with_registry_model")
    fake_module.run_keypoints_with_registry_model = fake_keypoints
    monkeypatch.setitem(sys.modules, "fisheye.utils.run_keypoints_with_registry_model", fake_module)

    rc, payload = _run_json(capsys, "keypoints", str(zarr_path), "--apply", "--pose-schema", "traditional_v3")

    assert rc == palette.EXIT_OK
    assert payload["status"] == "ok"
    assert payload["run"] == "keypoints_palette_001"
    assert calls["output"] == zarr_path.resolve()
    assert calls["pose_schema"] == "traditional_v3"
    assert calls["cli_provenance"]["input_run_ids"] == {"crop": "crop_001"}


def test_keypoints_force_overrides_missing_crop_with_loud_provenance(monkeypatch, tmp_path, capsys) -> None:
    zarr_path = tmp_path / "keypoints_force.zarr"
    _open_tmp_store(zarr_path)
    calls: dict = {}

    def fake_keypoints(**kwargs):
        calls.update(kwargs)
        return _runner_result(
            ok=True,
            status="dry_run",
            output_zarr=str(kwargs["output"]),
            keypoint_run=None,
            selected_model_path="/models/keypoints.pt",
            selected_run_id="keypoint_model_run",
            selected_set_id="keypoint_set",
        )

    fake_module = ModuleType("fisheye.utils.run_keypoints_with_registry_model")
    fake_module.run_keypoints_with_registry_model = fake_keypoints
    monkeypatch.setitem(sys.modules, "fisheye.utils.run_keypoints_with_registry_model", fake_module)

    rc, payload = _run_json(capsys, "keypoints", str(zarr_path), "--force")

    assert rc == palette.EXIT_OK
    assert payload["status"] == "dry_run"
    assert payload["forced"] is True
    assert payload["blocked_by"] == ["crop"]
    assert "--force" in payload["resolved_command"]
    overrides = payload["provenance"]["forced_dependency_overrides"]
    assert overrides == [
        {
            "blocked_by": ["crop"],
            "reason_code": "BLOCKED_BY_CROP",
            "stage": "keypoints",
            "warning": "Catalog dependency gate explicitly overridden by --force.",
        }
    ]
    assert calls["dry_run"] is True
    assert calls["cli_provenance"] is None


def test_keypoints_force_apply_passes_override_in_cli_provenance(monkeypatch, tmp_path, capsys) -> None:
    zarr_path = tmp_path / "keypoints_force_apply.zarr"
    _open_tmp_store(zarr_path)
    calls: dict = {}

    def fake_keypoints(**kwargs):
        calls.update(kwargs)
        return _runner_result(
            ok=True,
            status="ok",
            output_zarr=str(kwargs["output"]),
            keypoint_run="keypoints_force_001",
            selected_model_path="/models/keypoints.pt",
            selected_run_id="keypoint_model_run",
            selected_set_id="keypoint_set",
        )

    fake_module = ModuleType("fisheye.utils.run_keypoints_with_registry_model")
    fake_module.run_keypoints_with_registry_model = fake_keypoints
    monkeypatch.setitem(sys.modules, "fisheye.utils.run_keypoints_with_registry_model", fake_module)

    rc, payload = _run_json(capsys, "keypoints", str(zarr_path), "--apply", "--force")

    assert rc == palette.EXIT_OK
    assert payload["status"] == "ok"
    assert payload["forced"] is True
    assert payload["blocked_by"] == ["crop"]
    overrides = calls["cli_provenance"]["forced_dependency_overrides"]
    assert overrides == [
        {
            "blocked_by": ["crop"],
            "reason_code": "BLOCKED_BY_CROP",
            "stage": "keypoints",
            "warning": "Catalog dependency gate explicitly overridden by --force.",
        }
    ]


def test_crop_apply_writes_cli_provenance_to_real_tmp_zarr(monkeypatch, tmp_path, capsys) -> None:
    zarr_path = tmp_path / "crop_ready.zarr"
    root = _open_tmp_store(zarr_path)
    _complete_run(root, "refined_detect_runs", "refined_001")

    def fake_plan(zarr_path_arg, config, source_type, source_path, selection_policy, force_new, crop_storage_mode):
        return CropPlan(
            zarr_path=Path(zarr_path_arg),
            status="ok",
            source_type="refined",
            source_path="refined_detect_runs/refined_001/instances",
            roi_size=(512, 512),
            crop_storage_mode="geometry_only",
            selection_policy=selection_policy,
        )

    def fake_crop_detections(**kwargs):
        target = zarr.open_group(str(kwargs["zarr_path"]), mode="a")
        parent = target.require_group("crop_runs")
        run = parent.require_group("crop_palette_001")
        run.attrs["cli_provenance"] = dict(kwargs["cli_provenance"])
        return {
            "run_name": "crop_palette_001",
            "total_crops": 7,
            "frames_with_crops": 7,
            "percent_cropped": 100.0,
            "duration_seconds": 0.01,
            "detection_source_type": kwargs["source_type"],
            "detection_source_path": kwargs["source_path"],
            "crop_storage_mode": kwargs["crop_storage_mode"],
        }

    import fisheye.tracking.crop as crop_module
    import fisheye.utils.crop_batch as crop_batch

    monkeypatch.setattr(crop_batch, "_build_plan", fake_plan)
    monkeypatch.setattr(crop_module, "crop_detections", fake_crop_detections)

    rc, payload = _run_json(capsys, "crop", str(zarr_path), "--apply")

    assert rc == palette.EXIT_OK
    assert payload["status"] == "ok"
    assert payload["run"] == "crop_palette_001"
    reopened = zarr.open_group(str(zarr_path), mode="r")
    stamped = dict(reopened["crop_runs"]["crop_palette_001"].attrs["cli_provenance"])
    assert stamped["command"] == "palette crop"
    assert stamped["config_hash"]
    assert stamped["input_run_ids"] == {"refined_detect": "refined_001"}


def test_crop_apply_reports_no_rows_as_failed_envelope(monkeypatch, tmp_path, capsys) -> None:
    zarr_path = tmp_path / "crop_no_rows.zarr"
    root = _open_tmp_store(zarr_path)
    _complete_run(root, "refined_detect_runs", "refined_001")

    def fake_plan(zarr_path_arg, config, source_type, source_path, selection_policy, force_new, crop_storage_mode):
        return CropPlan(
            zarr_path=Path(zarr_path_arg),
            status="ok",
            source_type="refined",
            source_path="refined_detect_runs/refined_001/instances",
            roi_size=(512, 512),
            crop_storage_mode="geometry_only",
            selection_policy=selection_policy,
        )

    def fake_crop_detections(**kwargs):
        return {
            "run_name": "crop_empty_001",
            "total_crops": 0,
            "frames_with_crops": 0,
            "percent_cropped": 0.0,
            "duration_seconds": 0.01,
            "detection_source_type": kwargs["source_type"],
            "detection_source_path": kwargs["source_path"],
            "crop_storage_mode": kwargs["crop_storage_mode"],
        }

    import fisheye.tracking.crop as crop_module
    import fisheye.utils.crop_batch as crop_batch

    monkeypatch.setattr(crop_batch, "_build_plan", fake_plan)
    monkeypatch.setattr(crop_module, "crop_detections", fake_crop_detections)

    rc, payload = _run_json(capsys, "crop", str(zarr_path), "--apply")

    assert rc == palette.EXIT_FAILED
    assert payload["status"] == "failed"
    assert payload["reason_code"] == "CROP_NO_ROWS_WRITTEN"
    assert payload["metrics"]["rows_written"] == 0
    assert payload["metrics"]["rows_failed"] is None
    assert payload["metrics"]["total_crops"] == 0


def test_crop_apply_reports_all_failed_rows_as_failed_envelope(monkeypatch, tmp_path, capsys) -> None:
    zarr_path = tmp_path / "crop_all_failed.zarr"
    root = _open_tmp_store(zarr_path)
    _complete_run(root, "refined_detect_runs", "refined_001")

    def fake_plan(zarr_path_arg, config, source_type, source_path, selection_policy, force_new, crop_storage_mode):
        return CropPlan(
            zarr_path=Path(zarr_path_arg),
            status="ok",
            source_type="refined",
            source_path="refined_detect_runs/refined_001/instances",
            roi_size=(512, 512),
            crop_storage_mode="geometry_only",
            selection_policy=selection_policy,
        )

    def fake_crop_detections(**kwargs):
        return {
            "run_name": "crop_failed_001",
            "total_crops": 0,
            "rows_failed": 5,
            "frames_with_crops": 0,
            "percent_cropped": 0.0,
            "duration_seconds": 0.01,
            "detection_source_type": kwargs["source_type"],
            "detection_source_path": kwargs["source_path"],
            "crop_storage_mode": kwargs["crop_storage_mode"],
        }

    import fisheye.tracking.crop as crop_module
    import fisheye.utils.crop_batch as crop_batch

    monkeypatch.setattr(crop_batch, "_build_plan", fake_plan)
    monkeypatch.setattr(crop_module, "crop_detections", fake_crop_detections)

    rc, payload = _run_json(capsys, "crop", str(zarr_path), "--apply")

    assert rc == palette.EXIT_FAILED
    assert payload["status"] == "failed"
    assert payload["reason_code"] == "CROP_ALL_ROWS_FAILED"
    assert payload["metrics"]["rows_written"] == 0
    assert payload["metrics"]["rows_failed"] == 5
