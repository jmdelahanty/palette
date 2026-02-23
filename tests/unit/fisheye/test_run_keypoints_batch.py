from __future__ import annotations

import json
from pathlib import Path

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


def test_run_plan_returns_rich_payload_with_optional_refine(
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
    )

    assert called == ["traditional", "refine"]
    assert result["status"] == "ok"
    assert result["method"] == "traditional"
    assert result["keypoints"]["run_name"] == "keypoints_001"
    assert result["refined_keypoints"]["run_name"] == "refined_keypoints_001"
    assert "duration_seconds" in result


def test_run_plan_refine_only_skips_refine_when_zero_rois(
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
    result = mod._run_plan(
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

    assert result["status"] == "ok"
    assert result["source_keypoints_run"] == "keypoints_001"
    assert result["refine_skipped_reason"] == "zero_keypoint_rois"
    assert "refined_keypoints" not in result


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


def test_main_refine_only_delegates_to_refine_keypoints_batch(
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
    assert rc == 0
    stderr = capsys.readouterr().err
    assert "deprecated" in stderr.lower()
    assert "single-threaded" in stderr

    argv = captured["argv"]
    assert str(tmp_path) in argv
    assert "--file-list" in argv
    assert str(file_list) in argv
    assert "--recursive" in argv
    assert "--apply" in argv
    assert "--zarr-use" in argv
    assert "any" in argv
    assert "--no-skip-existing" in argv
    assert "--num-workers" in argv
    assert "3" in argv
    assert "--no-log" in argv
    assert "--json" in argv
    assert "--scheduler" not in argv
