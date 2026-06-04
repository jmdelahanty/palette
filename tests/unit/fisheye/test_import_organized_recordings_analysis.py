from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from fisheye.utils import import_organized_recordings_analysis as mod


def _recording(root: Path, name: str = "2026-05-29T18-11-16Z_arena_1_GoodCopBadCop") -> Path:
    rec = root / name
    (rec / "cams").mkdir(parents=True, exist_ok=True)
    (rec / "raw").mkdir(parents=True, exist_ok=True)
    (rec / "cams" / "Cam2010093_2026-05-29T18-11-16Z_arena_1.mp4").touch()
    (rec / "raw" / f"{name}.h5").touch()
    return rec


def test_discover_recording_dirs_finds_organized_h5_recording(tmp_path: Path) -> None:
    rec = _recording(tmp_path)

    discovered = mod.discover_recording_dirs(
        tmp_path,
        recursive=False,
        import_stimulus=True,
    )

    assert discovered == [rec]


def test_discover_recording_dirs_can_read_organize_log(tmp_path: Path) -> None:
    rec = _recording(tmp_path)
    log_path = tmp_path / "organize.jsonl"
    log_path.write_text(
        "\n".join(
            [
                json.dumps({"event": "recording_plan", "dest_dir": "/ignored"}),
                json.dumps({"event": "recording_applied", "dest_dir": str(rec)}),
            ]
        ),
        encoding="utf-8",
    )

    discovered = mod.discover_recording_dirs(
        tmp_path / "not_scanned",
        recursive=False,
        import_stimulus=True,
        organize_logs=[log_path],
    )

    assert discovered == [rec.resolve()]


def test_build_plans_skips_existing_analysis_zarr(tmp_path: Path) -> None:
    rec = _recording(tmp_path)
    zarr_path = rec / "zarr" / f"{rec.name}_analysis.zarr"
    zarr_path.mkdir(parents=True)

    plans = mod.build_plans(
        [rec],
        import_stimulus=True,
        skip_existing=True,
        allow_preflight_failures=False,
        check_stimulus=False,
    )

    assert len(plans) == 1
    assert plans[0].status == "skipped"
    assert plans[0].reason == "analysis zarr already exists"


def test_main_apply_uses_import_only_process(monkeypatch, tmp_path: Path) -> None:
    rec = _recording(tmp_path)
    calls: list[tuple[Path, bool]] = []

    def _fake_process(plan, opts, **_kwargs):
        calls.append((plan.recording_dir, opts.import_stimulus))
        return SimpleNamespace(ok=True, failed_step=None, error=None, returncode=None)

    monkeypatch.setattr(mod, "process_recording_import", _fake_process)

    rc = mod.main(
        [
            str(tmp_path),
            "--apply",
            "--no-log",
        ]
    )

    assert rc == 0
    assert calls == [(rec.resolve(), True)]


def test_main_recording_only_discovers_video_recording(monkeypatch, tmp_path: Path) -> None:
    rec = tmp_path / "video_only"
    (rec / "cams").mkdir(parents=True)
    (rec / "cams" / "Cam2010093_video_only.mp4").touch()
    calls: list[tuple[Path, object]] = []

    def _fake_process(plan, opts, **_kwargs):
        calls.append((plan.recording_dir, plan.h5_path))
        return SimpleNamespace(ok=True, failed_step=None, error=None, returncode=None)

    monkeypatch.setattr(mod, "process_recording_import", _fake_process)

    rc = mod.main(
        [
            str(tmp_path),
            "--recording-only",
            "--apply",
            "--no-log",
        ]
    )

    assert rc == 0
    assert calls == [(rec.resolve(), None)]
