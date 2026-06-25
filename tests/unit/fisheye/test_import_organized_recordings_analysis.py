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
    (rec / "recording_manifest.json").write_text(
        json.dumps(
            {
                "recording_name": name,
                "recording_type": "behavior",
                "recording_subtype": "free",
                "behavior_mode": "free",
                "artifact_schema_id": "orange_external_ipc_single_clip_v1",
                "preflight": {"status": "not_run", "checked_at_utc": None, "video": None, "h5": None},
            }
        ),
        encoding="utf-8",
    )
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
    manifest = json.loads((rec / "recording_manifest.json").read_text(encoding="utf-8"))
    assert manifest["import_status"] == "ok"
    assert manifest["analysis_zarr_path"].endswith(f"{rec.name}_analysis.zarr")


def test_main_apply_syncs_successful_import_when_registry_is_provided(monkeypatch, tmp_path: Path) -> None:
    rec = _recording(tmp_path)
    registry_path = tmp_path / "registry.sqlite"
    process_calls: list[Path] = []
    registry_calls: list[tuple[Path, Path]] = []

    def _fake_process(plan, _opts, **_kwargs):
        process_calls.append(plan.recording_dir)
        return SimpleNamespace(ok=True, failed_step=None, error=None, returncode=None)

    class _Registry:
        def __init__(self, path: Path) -> None:
            self.path = path

        def scan_zarr(self, zarr_path: Path) -> str:
            registry_calls.append((self.path, zarr_path))
            return "dataset-1"

        def close(self) -> None:
            pass

    monkeypatch.setattr(mod, "process_recording_import", _fake_process)
    monkeypatch.setattr(mod, "Registry", _Registry)

    rc = mod.main(
        [
            str(tmp_path),
            "--apply",
            "--registry",
            str(registry_path),
            "--no-log",
        ]
    )

    expected_zarr = rec.resolve() / "zarr" / f"{rec.name}_analysis.zarr"
    assert rc == 0
    assert process_calls == [rec.resolve()]
    assert registry_calls == [(registry_path.resolve(), expected_zarr)]
    manifest = json.loads((rec / "recording_manifest.json").read_text(encoding="utf-8"))
    assert manifest["import_status"] == "ok"
    assert manifest["registry_dataset_id"] == "dataset-1"


def test_main_syncs_skipped_existing_zarr_when_registry_is_provided(monkeypatch, tmp_path: Path) -> None:
    rec = _recording(tmp_path)
    zarr_path = rec / "zarr" / f"{rec.name}_analysis.zarr"
    zarr_path.mkdir(parents=True)
    registry_path = tmp_path / "registry.sqlite"
    registry_calls: list[Path] = []

    def _unexpected_process(*_args, **_kwargs):
        raise AssertionError("existing zarr should be skipped, not imported")

    class _Registry:
        def __init__(self, _path: Path) -> None:
            pass

        def scan_zarr(self, path: Path) -> str:
            registry_calls.append(path)
            return "dataset-existing"

        def close(self) -> None:
            pass

    monkeypatch.setattr(mod, "process_recording_import", _unexpected_process)
    monkeypatch.setattr(mod, "Registry", _Registry)

    rc = mod.main(
        [
            str(tmp_path),
            "--apply",
            "--registry",
            str(registry_path),
            "--no-log",
        ]
    )

    assert rc == 0
    assert registry_calls == [zarr_path.resolve()]
    manifest = json.loads((rec / "recording_manifest.json").read_text(encoding="utf-8"))
    assert manifest["import_status"] == "ok"
    assert manifest["registry_dataset_id"] == "dataset-existing"


def test_main_registry_sync_failure_marks_run_failed(monkeypatch, tmp_path: Path) -> None:
    _recording(tmp_path)

    def _fake_process(_plan, _opts, **_kwargs):
        return SimpleNamespace(ok=True, failed_step=None, error=None, returncode=None)

    class _Registry:
        def __init__(self, _path: Path) -> None:
            pass

        def scan_zarr(self, _path: Path) -> str:
            raise RuntimeError("registry unavailable")

        def close(self) -> None:
            pass

    monkeypatch.setattr(mod, "process_recording_import", _fake_process)
    monkeypatch.setattr(mod, "Registry", _Registry)

    rc = mod.main(
        [
            str(tmp_path),
            "--apply",
            "--registry",
            str(tmp_path / "registry.sqlite"),
            "--no-log",
        ]
    )

    assert rc == 1


def test_main_recording_only_discovers_video_recording(monkeypatch, tmp_path: Path) -> None:
    rec = tmp_path / "video_only"
    (rec / "cams").mkdir(parents=True)
    (rec / "cams" / "Cam2010093_video_only.mp4").touch()
    (rec / "recording_manifest.json").write_text(
        json.dumps(
            {
                "recording_name": rec.name,
                "recording_type": "behavior",
                "recording_subtype": "free",
                "behavior_mode": "free",
                "artifact_schema_id": "video_only_v1",
            }
        ),
        encoding="utf-8",
    )
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
