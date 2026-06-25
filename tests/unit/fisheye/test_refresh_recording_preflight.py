from __future__ import annotations

import json
from pathlib import Path

from fisheye.utils import refresh_recording_preflight as mod


def test_plan_from_manifest_preserves_recording_context(tmp_path: Path) -> None:
    rec = tmp_path / "2026-06-21T18-18-31Z_arena_1_GoodCopBadCop"
    rec.mkdir()
    manifest_path = rec / "recording_manifest.json"
    manifest_path.write_text(
        json.dumps({"camera_id": "2010093", "session_uuid": "session-1", "source_dir": str(tmp_path / "staging")}),
        encoding="utf-8",
    )

    plan = mod._plan_from_manifest(manifest_path)

    assert plan.name == rec.name
    assert plan.dest_dir == rec
    assert plan.camera_id == "2010093"
    assert plan.meta["session_uuid"] == "session-1"


def test_refresh_manifest_preflight_uses_existing_diagnostic_hooks(monkeypatch, tmp_path: Path) -> None:
    rec = tmp_path / "rec"
    rec.mkdir()
    manifest_path = rec / "recording_manifest.json"
    manifest_path.write_text(json.dumps({"camera_id": "2010093"}), encoding="utf-8")
    calls: list[str] = []

    def _fake_video(plan, logger):
        calls.append(f"video:{plan.name}")
        return object()

    def _fake_h5(plan, logger):
        calls.append(f"h5:{plan.name}")
        return object()

    def _fake_persist(plan, *, video_result, h5_result):
        calls.append(f"persist:{plan.name}:{video_result is not None}:{h5_result is not None}")
        return None

    monkeypatch.setattr(mod, "_run_video_diagnostics_for_plan", _fake_video)
    monkeypatch.setattr(mod, "_run_h5_diagnostics_for_plan", _fake_h5)
    monkeypatch.setattr(mod, "_persist_preflight_to_manifest", _fake_persist)

    status, error = mod.refresh_manifest_preflight(
        manifest_path,
        run_video=True,
        run_h5=True,
        apply=True,
    )

    assert status == "updated"
    assert error is None
    assert calls == ["video:rec", "h5:rec", "persist:rec:True:True"]
