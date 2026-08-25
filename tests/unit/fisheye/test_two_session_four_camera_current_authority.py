from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from fisheye.registry.db import Registry
from fisheye.shared import run_provenance
from fisheye.shared.source_recording_identity import (
    SOURCE_RECORDING_IDENTITY_PROFILE,
    SOURCE_RECORDING_IDENTITY_PROFILE_ATTR,
)
from fisheye.utils import draft_video_only_organizer_manifest as draft_manifest
from fisheye.utils import import_recording_analysis as importer
from fisheye.utils import organize_recordings


CAMERA_IDS = ("2010093", "2010094", "2010095", "2010096")
SESSION_UUIDS = ("session_alpha", "session_beta")


def _organize_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    session_uuid: str,
) -> list[Path]:
    """Draft and apply one real four-camera organizer manifest."""

    source_root = tmp_path / "staging" / session_uuid
    source_root.mkdir(parents=True)
    (source_root / "recording_snapshot.json").write_text(
        json.dumps(
            {
                "recording_id": session_uuid,
                "session_uuid": session_uuid,
                "timestamp_utc": "2026-08-25T12:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    for camera_id in CAMERA_IDS:
        (source_root / f"Cam{camera_id}.mp4").write_bytes(
            f"{session_uuid}-{camera_id}".encode("ascii")
        )

    metadata_csv = tmp_path / "manifests" / f"{session_uuid}.csv"
    assert (
        draft_manifest.main(
            [
                str(source_root),
                "--output",
                str(metadata_csv),
                "--session-uuid",
                session_uuid,
                "--recording-id",
                session_uuid,
            ]
        )
        == 0
    )

    # The organizer normally runs this container check against every video.
    # Keep the media fixture tiny while still exercising the real move and
    # manifest-writing path.
    monkeypatch.setattr(
        organize_recordings,
        "check_hevc_keyframe_flags",
        lambda _path: {
            "codec": "hevc",
            "has_stss": True,
            "needs_fix": False,
            "message": "fixture",
        },
    )
    dest_root = tmp_path / "recordings"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "organize_recordings.py",
            str(source_root),
            "--video-only",
            "--metadata-csv",
            str(metadata_csv),
            "--dest-root",
            str(dest_root),
            "--apply",
        ],
    )
    assert organize_recordings.main() == 0

    recording_dirs = [
        dest_root / f"{session_uuid}_cam{camera_id}" for camera_id in CAMERA_IDS
    ]
    for recording_dir, camera_id in zip(recording_dirs, CAMERA_IDS):
        manifest_path = recording_dir / "recording_manifest.json"
        assert manifest_path.is_file()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        # These are assertions about the producer's manifest, not test-added
        # identity attributes. The analysis importer must copy this declaration
        # into the root and the authority must verify it later.
        assert manifest[SOURCE_RECORDING_IDENTITY_PROFILE_ATTR] == (
            SOURCE_RECORDING_IDENTITY_PROFILE
        )
        assert manifest["recording_id"] == f"{session_uuid}_cam{camera_id}"
        assert manifest["session_uuid"] == session_uuid
        assert manifest["camera_id"] == camera_id
    return recording_dirs


def _publish_analysis_imports(
    recording_dirs: list[Path],
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[Path, importer.RecordingImportResult]]:
    def probe(path: Path, **_kwargs: object) -> dict[str, object]:
        return {
            "source_video": path.name,
            "source_path": str(path),
            "width": 4512,
            "height": 4512,
            "total_frames": 4,
            "fps": 100.0,
            "duration_seconds": 0.04,
            "codec": "hevc",
            "pix_fmt": "yuv420p",
        }

    # These doubles only replace external media/provenance probes. The Zarr
    # identity writer, receipt publisher, and registry verified reader remain
    # real.
    monkeypatch.setattr(importer, "probe_video_metadata", probe)
    monkeypatch.setattr(
        importer,
        "apply_acquisition_frame_clock",
        lambda _plan: {"available": False, "status": "fixture"},
    )
    clean_code = {"git_sha": "1" * 40, "git_dirty": False}
    monkeypatch.setattr(importer, "git_identity", lambda **_kwargs: clean_code)
    monkeypatch.setattr(run_provenance, "git_identity", lambda **_kwargs: clean_code)

    options = importer.RecordingImportOptions(
        import_video_metadata=True,
        video_metadata_overwrite=False,
        import_stimulus=False,
        stimulus_always=False,
        stimulus_run_name=None,
        stimulus_overwrite=False,
        stimulus_quiet=True,
        allow_preflight_failures=True,
    )
    published: list[tuple[Path, importer.RecordingImportResult]] = []
    for recording_dir in recording_dirs:
        camera_id = recording_dir.name.rsplit("_cam", 1)[1]
        video = recording_dir / "cams" / f"Cam{camera_id}_{recording_dir.name.split('_cam', 1)[0]}.mp4"
        plan = importer.RecordingAnalysisPlan(
            recording_dir=recording_dir,
            h5_path=None,
            cam_video=video,
            zarr_path=recording_dir / "zarr" / f"{recording_dir.name}_analysis.zarr",
        )
        result = importer.process_recording_import(plan, options)
        assert result.ok is True, result
        assert result.receipt is not None
        published.append((plan.zarr_path, result))
    return published


def test_two_sessions_four_cameras_bind_eight_current_recordings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recording_dirs: list[Path] = []
    for session_uuid in SESSION_UUIDS:
        recording_dirs.extend(
            _organize_session(tmp_path, monkeypatch, session_uuid=session_uuid)
        )

    published = _publish_analysis_imports(recording_dirs, monkeypatch)
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        verified_rows = []
        for zarr_path, result in published:
            assert result.receipt is not None
            dataset_id = registry.synchronize_recording_import(
                zarr_path=zarr_path,
                receipt=result.receipt,
                decided_by="pytest:two-session-four-camera",
            )
            verified = registry.read_verified_recording_import(dataset_id)
            verified_by_path = registry.read_verified_recording_import_by_path(zarr_path)
            assert verified_by_path == verified
            assert verified.receipt.receipt_sha256 == result.receipt.receipt_sha256
            verified_rows.append(
                (
                    verified.identity.recording_id,
                    verified.identity.session_uuid,
                    verified.acquisition_frame.record.camera_id,
                )
            )

        expected = {
            (f"{session_uuid}_cam{camera_id}", session_uuid, camera_id)
            for session_uuid in SESSION_UUIDS
            for camera_id in CAMERA_IDS
        }
        assert len(verified_rows) == 8
        assert len({row[0] for row in verified_rows}) == 8
        assert {row[1] for row in verified_rows} == set(SESSION_UUIDS)
        assert {row[2] for row in verified_rows} == set(CAMERA_IDS)
        assert set(verified_rows) == expected
        assert registry.conn.execute("SELECT COUNT(*) FROM recordings;").fetchone()[0] == 8
        assert registry.conn.execute("SELECT COUNT(*) FROM datasets;").fetchone()[0] == 8
        assert (
            registry.conn.execute(
                "SELECT COUNT(*) FROM recording_import_receipt_bindings;"
            ).fetchone()[0]
            == 8
        )
    finally:
        registry.close()
