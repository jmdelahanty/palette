from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from fisheye.utils import run_detections_batch as mod


def _write_root_metadata(zarr_path: Path) -> None:
    zarr_path.mkdir(parents=True, exist_ok=True)
    (zarr_path / "zarr.json").write_text(
        json.dumps(
            {
                "attributes": {},
                "zarr_format": 3,
                "consolidated_metadata": None,
                "node_type": "group",
            }
        ),
        encoding="utf-8",
    )


def _write_group_attrs(zarr_path: Path, group_name: str, attrs: dict[str, object]) -> None:
    group_dir = zarr_path / group_name
    group_dir.mkdir(parents=True, exist_ok=True)
    (group_dir / "zarr.json").write_text(
        json.dumps(
            {
                "attributes": attrs,
                "zarr_format": 3,
                "consolidated_metadata": None,
                "node_type": "group",
            }
        ),
        encoding="utf-8",
    )


def test_discover_analysis_zarrs_is_deterministic(tmp_path: Path) -> None:
    rec_b = tmp_path / "rec_b"
    rec_a = tmp_path / "rec_a"
    (rec_b / "zarr" / "rec_b_analysis.zarr").mkdir(parents=True, exist_ok=True)
    (rec_a / "zarr" / "rec_a_analysis.zarr").mkdir(parents=True, exist_ok=True)

    found = mod._discover_analysis_zarrs(  # noqa: SLF001
        [tmp_path, rec_b / "zarr" / "rec_b_analysis.zarr"],
        recursive=True,
    )

    found_str = [str(path) for path in found]
    assert found_str == sorted(found_str)
    assert found_str == [
        str((rec_a / "zarr" / "rec_a_analysis.zarr").resolve()),
        str((rec_b / "zarr" / "rec_b_analysis.zarr").resolve()),
    ]


def test_build_plans_applies_status_reason_taxonomy(tmp_path: Path) -> None:
    rec_ok = tmp_path / "rec_ok"
    rec_skip = tmp_path / "rec_skip"
    rec_no_video = tmp_path / "rec_no_video"

    zarr_ok = rec_ok / "zarr" / "rec_ok_analysis.zarr"
    zarr_skip = rec_skip / "zarr" / "rec_skip_analysis.zarr"
    zarr_no_video = rec_no_video / "zarr" / "rec_no_video_analysis.zarr"

    _write_root_metadata(zarr_ok)
    _write_root_metadata(zarr_skip)
    _write_root_metadata(zarr_no_video)

    _write_group_attrs(zarr_ok, "detect_runs", {"latest": None})
    _write_group_attrs(zarr_ok, "background_runs", {"latest": "bg_1"})
    _write_group_attrs(zarr_ok, "analysis_metadata", {"detection_tuning": {"enabled": True}})

    _write_group_attrs(zarr_skip, "detect_runs", {"latest": "detect_1"})
    _write_group_attrs(zarr_skip, "background_runs", {"latest": "bg_1"})
    _write_group_attrs(zarr_skip, "analysis_metadata", {"detection_tuning": {"enabled": True}})

    _write_group_attrs(zarr_no_video, "detect_runs", {"latest": None})
    _write_group_attrs(zarr_no_video, "background_runs", {"latest": "bg_1"})
    _write_group_attrs(zarr_no_video, "analysis_metadata", {"detection_tuning": {"enabled": True}})

    (rec_ok / "cams").mkdir(parents=True, exist_ok=True)
    (rec_skip / "cams").mkdir(parents=True, exist_ok=True)
    (rec_ok / "cams" / "cam_1.mp4").write_bytes(b"")
    (rec_skip / "cams" / "cam_1.mp4").write_bytes(b"")

    plans = mod._build_plans(  # noqa: SLF001
        [zarr_no_video, zarr_skip, zarr_ok],
        skip_existing=True,
        require_background=False,
        require_tuning=False,
    )

    by_path = {str(plan.zarr_path): plan for plan in plans}

    ok_plan = by_path[str(zarr_ok.resolve())]
    assert ok_plan.status == mod.STATUS_OK
    assert ok_plan.reason is None

    skip_plan = by_path[str(zarr_skip.resolve())]
    assert skip_plan.status == mod.STATUS_SKIPPED
    assert skip_plan.reason == mod.REASON_DETECT_ALREADY_PRESENT

    missing_plan = by_path[str(zarr_no_video.resolve())]
    assert missing_plan.status == mod.STATUS_MISSING
    assert missing_plan.reason == mod.REASON_CAMS_DIR_MISSING


def test_main_apply_invokes_registry_runner_and_returns_failure(monkeypatch, tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    registry_path.write_text("", encoding="utf-8")

    zarr_path = tmp_path / "rec_a" / "zarr" / "rec_a_analysis.zarr"
    zarr_path.mkdir(parents=True, exist_ok=True)
    recording_dir = zarr_path.parent.parent
    video_path = recording_dir / "cams" / "cam_1.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(b"")

    plan = mod.DetectPlan(
        zarr_path=zarr_path.resolve(),
        recording_dir=recording_dir.resolve(),
        video_path=video_path.resolve(),
        status=mod.STATUS_OK,
    )

    monkeypatch.setattr(mod, "_resolve_input_paths", lambda *_args, **_kwargs: [tmp_path])
    monkeypatch.setattr(mod, "_discover_analysis_zarrs", lambda *_args, **_kwargs: [zarr_path.resolve()])
    monkeypatch.setattr(mod, "_build_plans", lambda *_args, **_kwargs: [plan])

    calls: list[dict[str, object]] = []

    def _fake_runner(**kwargs):  # noqa: ANN003
        calls.append(kwargs)
        return SimpleNamespace(
            ok=False,
            recording_dir=str(recording_dir.resolve()),
            output_zarr=str(zarr_path.resolve()),
            reason="detect_inference_failed",
            error="oom",
            remediation="retry",
            selected_model_path="/tmp/model.pt",
            selected_run_id="run_1",
            selected_set_id="set_1",
            resolved_at_utc=None,
            detect_run=None,
        )

    monkeypatch.setattr(mod, "run_detect_with_registry_model", _fake_runner)

    rc = mod.main(["--apply", "--json", "--no-log", "--registry", str(registry_path), str(tmp_path)])

    assert rc == 1
    assert len(calls) == 1
    assert calls[0]["recording_dir"] == recording_dir.resolve()
    assert calls[0]["video"] == video_path.resolve()
    assert calls[0]["output"] == zarr_path.resolve()
    assert calls[0]["registry"] == registry_path.resolve()


def test_main_dry_run_marks_registry_missing(monkeypatch, tmp_path: Path, capsys) -> None:
    zarr_path = tmp_path / "rec_a" / "zarr" / "rec_a_analysis.zarr"
    zarr_path.mkdir(parents=True, exist_ok=True)
    recording_dir = zarr_path.parent.parent
    video_path = recording_dir / "cams" / "cam_1.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(b"")

    plan = mod.DetectPlan(
        zarr_path=zarr_path.resolve(),
        recording_dir=recording_dir.resolve(),
        video_path=video_path.resolve(),
        status=mod.STATUS_OK,
    )

    monkeypatch.setattr(mod, "_resolve_input_paths", lambda *_args, **_kwargs: [tmp_path])
    monkeypatch.setattr(mod, "_discover_analysis_zarrs", lambda *_args, **_kwargs: [zarr_path.resolve()])
    monkeypatch.setattr(mod, "_build_plans", lambda *_args, **_kwargs: [plan])

    def _unexpected_runner(**_kwargs):
        raise AssertionError("run_detect_with_registry_model should not be called in dry-run")

    monkeypatch.setattr(mod, "run_detect_with_registry_model", _unexpected_runner)

    missing_registry = tmp_path / "missing_registry.sqlite"
    rc = mod.main(["--dry-run", "--json", "--no-log", "--registry", str(missing_registry), str(tmp_path)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "registry_missing" in out
