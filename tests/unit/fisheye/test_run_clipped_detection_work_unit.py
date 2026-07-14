from __future__ import annotations

import json
from pathlib import Path

from fisheye.utils import run_clipped_detection_work_unit as worker


def test_work_unit_builds_imports_and_validates(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}
    scratch = tmp_path / "scratch"
    monkeypatch.setattr(worker, "_scratch_root", lambda: scratch)

    def fake_build(**kwargs):
        calls["build"] = kwargs
        Path(kwargs["tarball_output"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["tarball_output"]).write_bytes(b"tar")
        return {"status": "ok", "run_name": kwargs["run_name"]}

    def fake_import(**kwargs):
        calls["import"] = kwargs
        return {"status": "ok", "applied": True}

    def fake_validate(**kwargs):
        calls["validate"] = kwargs
        return {"status": "pass"}

    monkeypatch.setattr(worker, "build_detection_artifact", fake_build)
    monkeypatch.setattr(worker, "apply_import", fake_import)
    monkeypatch.setattr(worker, "validate_imported_run_group", fake_validate)
    report_path = tmp_path / "reports" / "clip.json"
    report = worker.run_work_unit(
        video_path=tmp_path / "clip.mp4",
        target_zarr=tmp_path / "analysis.zarr",
        target_group_path="clips/clip_000000/cameras/2010093/detect_runs/detect_run",
        model_path=tmp_path / "model.pt",
        model_sha256="a" * 64,
        model_registry_set_id="detect_set",
        model_registry_run_id="detect_model_run",
        config_path=tmp_path / "config.yaml",
        workflow_id="campaign",
        recording_id="recording:z1",
        clip_id="clip_000000",
        clip_index=0,
        camera_serial="2010093",
        run_name="detect_run",
        report_path=report_path,
    )

    assert report["status"] == "ok"
    assert report["model"]["registry_run_id"] == "detect_model_run"
    assert calls["build"]["model_sha256"] == "a" * 64
    assert calls["build"]["model_registry_set_id"] == "detect_set"
    assert calls["import"]["use_intended_target"] is True
    assert calls["validate"]["target_group_path"].endswith("/detect_run")
    assert json.loads(report_path.read_text(encoding="utf-8"))["status"] == "ok"
