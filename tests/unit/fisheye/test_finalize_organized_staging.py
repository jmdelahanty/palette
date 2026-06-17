from __future__ import annotations

import json
from pathlib import Path

from fisheye.utils import finalize_organized_staging as mod


def _write(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _make_recording(
    root: Path,
    *,
    name: str,
    batch: Path,
    include_file: bool = True,
    include_zarr: bool = True,
) -> Path:
    recording = root / name
    if include_file:
        _write(recording / "raw" / "capture.h5", "h5")
        _write(recording / "cams" / "Cam2010093_example.mp4", "video")
    manifest = {
        "recording_name": name,
        "source_dir": str(batch),
        "files": {
            "raw": ["raw/capture.h5"],
            "cams": ["cams/Cam2010093_example.mp4"],
            "derived": [],
        },
        "preflight": {"status": "not_run"},
    }
    _write(recording / "recording_manifest.json", json.dumps(manifest))
    if include_zarr:
        (recording / "zarr" / f"{name}_analysis.zarr").mkdir(parents=True)
    return recording


def test_build_finalize_plan_ready_when_manifest_files_and_zarr_exist(tmp_path: Path) -> None:
    staging = tmp_path / "staging" / "batch_a"
    staging.mkdir(parents=True)
    recordings_root = tmp_path / "recordings"
    recording = _make_recording(recordings_root, name="rec_a", batch=staging)

    plan = mod.build_finalize_plan(staging, recordings_root=recordings_root)

    assert plan.status == "ready"
    assert plan.action == "move"
    assert plan.target_path == staging.parent / ".processed" / "batch_a"
    assert plan.recordings == [recording]
    assert plan.blockers == []


def test_build_finalize_plan_blocks_missing_manifest_file_and_zarr(tmp_path: Path) -> None:
    staging = tmp_path / "staging" / "batch_a"
    staging.mkdir(parents=True)
    recordings_root = tmp_path / "recordings"
    _make_recording(
        recordings_root,
        name="rec_a",
        batch=staging,
        include_file=False,
        include_zarr=False,
    )

    plan = mod.build_finalize_plan(staging, recordings_root=recordings_root)

    assert plan.status == "blocked"
    assert any("manifest-listed file missing" in blocker for blocker in plan.blockers)
    assert any("analysis zarr missing" in blocker for blocker in plan.blockers)


def test_apply_finalize_plan_moves_batch_to_processed(tmp_path: Path) -> None:
    staging = tmp_path / "staging" / "batch_a"
    _write(staging / "source.txt", "source")
    recordings_root = tmp_path / "recordings"
    _make_recording(recordings_root, name="rec_a", batch=staging)

    plan = mod.build_finalize_plan(staging, recordings_root=recordings_root)
    mod.apply_finalize_plan(plan)

    assert plan.applied is True
    assert not staging.exists()
    assert (staging.parent / ".processed" / "batch_a" / "source.txt").read_text(encoding="utf-8") == "source"


def test_import_log_failure_blocks_finalization(tmp_path: Path) -> None:
    staging = tmp_path / "staging" / "batch_a"
    staging.mkdir(parents=True)
    recordings_root = tmp_path / "recordings"
    recording = _make_recording(recordings_root, name="rec_a", batch=staging)
    import_log = tmp_path / "import.jsonl"
    _write(
        import_log,
        json.dumps(
            {
                "event": "recording_failed",
                "recording_dir": str(recording),
                "step": "import_stimulus_to_zarr",
            }
        )
        + "\n",
    )
    import_status = mod.read_import_log_status([import_log])

    plan = mod.build_finalize_plan(
        staging,
        recordings_root=recordings_root,
        import_status=import_status,
    )

    assert plan.status == "blocked"
    assert any("import failed" in blocker for blocker in plan.blockers)


def test_organize_log_can_discover_recording_for_batch(tmp_path: Path) -> None:
    staging = tmp_path / "staging" / "batch_a"
    staging.mkdir(parents=True)
    recordings_root = tmp_path / "recordings"
    recording = _make_recording(recordings_root, name="rec_a", batch=staging)
    organize_log = tmp_path / "organize.jsonl"
    _write(
        organize_log,
        "\n".join(
            [
                json.dumps({"event": "batch_start", "batch_source": str(staging)}),
                json.dumps({"event": "recording_applied", "dest_dir": str(recording)}),
            ]
        )
        + "\n",
    )

    plan = mod.build_finalize_plan(
        staging,
        recordings_root=tmp_path / "empty_recordings_root",
        organize_logs=[organize_log],
    )

    assert plan.status == "ready"
    assert plan.recordings == [recording]


def test_log_discovered_recording_blocks_if_manifest_source_is_different(tmp_path: Path) -> None:
    staging = tmp_path / "staging" / "batch_a"
    other_staging = tmp_path / "staging" / "batch_b"
    staging.mkdir(parents=True)
    other_staging.mkdir(parents=True)
    recordings_root = tmp_path / "recordings"
    recording = _make_recording(recordings_root, name="rec_a", batch=other_staging)
    organize_log = tmp_path / "organize.jsonl"
    _write(
        organize_log,
        "\n".join(
            [
                json.dumps({"event": "batch_start", "batch_source": str(staging)}),
                json.dumps({"event": "recording_applied", "dest_dir": str(recording)}),
            ]
        )
        + "\n",
    )

    plan = mod.build_finalize_plan(
        staging,
        recordings_root=tmp_path / "empty_recordings_root",
        organize_logs=[organize_log],
    )

    assert plan.status == "blocked"
    assert any("manifest source_dir does not point into batch" in blocker for blocker in plan.blockers)
