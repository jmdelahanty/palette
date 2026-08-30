from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.shared.source_recording_identity import (
    SOURCE_RECORDING_IDENTITY_PROFILE,
    SOURCE_RECORDING_IDENTITY_PROFILE_ATTR,
)
from fisheye.utils import backfill_video_only_sidecars as mod


def _write_manifest_csv(path: Path, *, source_root: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_video",
                "source_camera_metadata_csv",
                SOURCE_RECORDING_IDENTITY_PROFILE_ATTR,
                "camera_id",
                "session_uuid",
                "recording_id",
                "recording_name",
                "dish_design",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "source_video": "Cam2010093.mp4",
                "source_camera_metadata_csv": "Cam2010093_meta.csv",
                SOURCE_RECORDING_IDENTITY_PROFILE_ATTR: (
                    SOURCE_RECORDING_IDENTITY_PROFILE
                ),
                "camera_id": "2010093",
                "session_uuid": "sleepyfish_2026_05_05_17_45_30_cam2010093",
                "recording_id": "2026_05_05_17_45_30",
                "recording_name": "sleepyfish_2026_05_05_17_45_30_cam2010093",
                "dish_design": "palm",
            }
        )


def test_backfill_video_only_sidecars_repairs_partial_organization(tmp_path: Path) -> None:
    source_root = tmp_path / "staging"
    source_root.mkdir()
    (source_root / "Cam2010093_keyframe.json").write_text("{}", encoding="utf-8")
    (source_root / "Cam2010093_pipeline_perf.csv").write_text("metric,value\n", encoding="utf-8")
    (source_root / "Cam2010093_acquisition_cadence_probe.csv").write_text("metric,value\n", encoding="utf-8")
    ptp_path = source_root / "ptp_sync_summary.json"
    ptp_path.write_text("{}", encoding="utf-8")
    snapshot_path = source_root / "recording_snapshot.json"
    snapshot_path.write_text("{}", encoding="utf-8")

    metadata_csv = tmp_path / "video_only_manifest.csv"
    _write_manifest_csv(metadata_csv, source_root=source_root)

    recording_dir = tmp_path / "recordings" / "sleepyfish_2026_05_05_17_45_30_cam2010093"
    (recording_dir / "cams").mkdir(parents=True)
    (recording_dir / "cams" / "Cam2010093_sleepyfish_2026_05_05_17_45_30_cam2010093.mp4").write_bytes(b"video")
    (recording_dir / "cams" / "Cam2010093_sleepyfish_2026_05_05_17_45_30_cam2010093_meta.csv").write_text(
        "frame_id,timestamp,timestamp_sys\n",
        encoding="utf-8",
    )
    manifest_path = recording_dir / "recording_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "recording_name": "sleepyfish_2026_05_05_17_45_30_cam2010093",
                "files": {
                    "raw": [],
                    "cams": [
                        "cams/Cam2010093_sleepyfish_2026_05_05_17_45_30_cam2010093.mp4",
                        "cams/Cam2010093_sleepyfish_2026_05_05_17_45_30_cam2010093_meta.csv",
                    ],
                    "derived": [],
                },
            }
        ),
        encoding="utf-8",
    )

    rc = mod.main(
        [
            str(source_root),
            "--metadata-csv",
            str(metadata_csv),
            "--dest-root",
            str(tmp_path / "recordings"),
            "--apply",
        ]
    )

    assert rc == 0
    assert (recording_dir / "raw" / "ptp_sync_summary.json").exists()
    assert (recording_dir / "raw" / "recording_snapshot_runtime.json").exists()
    assert ptp_path.exists()
    assert snapshot_path.exists()
    assert (recording_dir / "cams" / "Cam2010093_sleepyfish_2026_05_05_17_45_30_cam2010093_keyframe.json").exists()
    assert (
        recording_dir / "derived" / "Cam2010093_sleepyfish_2026_05_05_17_45_30_cam2010093_pipeline_perf.csv"
    ).exists()
    assert (
        recording_dir
        / "derived"
        / "Cam2010093_sleepyfish_2026_05_05_17_45_30_cam2010093_acquisition_cadence_probe.csv"
    ).exists()
    assert not (source_root / "Cam2010093_keyframe.json").exists()
    assert not (source_root / "Cam2010093_pipeline_perf.csv").exists()
    assert not (source_root / "Cam2010093_acquisition_cadence_probe.csv").exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["files"]["raw"] == [
        "raw/ptp_sync_summary.json",
        "raw/recording_snapshot_runtime.json",
    ]
    assert manifest["files"]["cams"] == [
        "cams/Cam2010093_sleepyfish_2026_05_05_17_45_30_cam2010093.mp4",
        "cams/Cam2010093_sleepyfish_2026_05_05_17_45_30_cam2010093_meta.csv",
        "cams/Cam2010093_sleepyfish_2026_05_05_17_45_30_cam2010093_keyframe.json",
    ]
    assert manifest["files"]["derived"] == [
        "derived/Cam2010093_sleepyfish_2026_05_05_17_45_30_cam2010093_pipeline_perf.csv",
        "derived/Cam2010093_sleepyfish_2026_05_05_17_45_30_cam2010093_acquisition_cadence_probe.csv",
    ]


def test_backfill_video_only_sidecars_dry_run_does_not_mutate(tmp_path: Path) -> None:
    source_root = tmp_path / "staging"
    source_root.mkdir()
    sidecar = source_root / "Cam2010093_keyframe.json"
    sidecar.write_text("{}", encoding="utf-8")
    metadata_csv = tmp_path / "video_only_manifest.csv"
    _write_manifest_csv(metadata_csv, source_root=source_root)

    recording_dir = tmp_path / "recordings" / "sleepyfish_2026_05_05_17_45_30_cam2010093"
    recording_dir.mkdir(parents=True)
    manifest_path = recording_dir / "recording_manifest.json"
    manifest_path.write_text(json.dumps({"files": {"raw": [], "cams": [], "derived": []}}), encoding="utf-8")

    rc = mod.main(
        [
            str(source_root),
            "--metadata-csv",
            str(metadata_csv),
            "--dest-root",
            str(tmp_path / "recordings"),
            "--dry-run",
        ]
    )

    assert rc == 0
    assert sidecar.exists()
    assert not (recording_dir / "cams" / "Cam2010093_sleepyfish_2026_05_05_17_45_30_cam2010093_keyframe.json").exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["files"]["cams"] == []
    assert manifest["files"]["derived"] == []


def test_backfill_video_only_sidecars_rejects_unmarked_identity(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "staging"
    source_root.mkdir()
    sidecar = source_root / "Cam2010093_keyframe.json"
    sidecar.write_text("{}", encoding="utf-8")
    metadata_csv = tmp_path / "unmarked_video_only_manifest.csv"
    metadata_csv.write_text(
        "source_video,camera_id,session_uuid,recording_id,recording_name\n"
        "Cam2010093.mp4,2010093,session,recording,recording\n",
        encoding="utf-8",
    )

    rc = mod.main(
        [
            str(source_root),
            "--metadata-csv",
            str(metadata_csv),
            "--dest-root",
            str(tmp_path / "recordings"),
            "--dry-run",
        ]
    )

    assert rc == 1
    assert sidecar.exists()
    assert not (tmp_path / "recordings").exists()
