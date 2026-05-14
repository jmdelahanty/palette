from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.utils import draft_video_only_organizer_manifest as mod
from fisheye.utils import organize_recordings


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_draft_manifest_from_snapshot_and_flags_roundtrips_to_organizer(tmp_path: Path) -> None:
    source_root = tmp_path / "staging" / "2026_05_05_17_45_30"
    source_root.mkdir(parents=True)
    (source_root / "Cam2010093.mp4").write_bytes(b"video")
    (source_root / "Cam2010093_meta.csv").write_text(
        "frame_id,timestamp,timestamp_sys\n0,1.0,2.0\n",
        encoding="utf-8",
    )
    (source_root / "recording_snapshot.json").write_text(
        json.dumps(
            {
                "recording_id": "2026_05_05_17_45_30",
                "timestamp_utc": "2026-05-05T21:45:30Z",
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "video_only_manifest.csv"

    rc = mod.main(
        [
            str(source_root),
            "--output",
            str(output),
            "--dish-design",
            "cedar",
            "--rig-id",
            "omnifin0",
            "--arena-id",
            "arena_1",
            "--num-dishes",
            "1",
            "--fish-per-dish",
            "1",
        ]
    )

    assert rc == 0
    rows = _read_csv(output)
    assert len(rows) == 1
    row = rows[0]
    assert row["source_video"] == "Cam2010093.mp4"
    assert row["source_camera_metadata_csv"] == "Cam2010093_meta.csv"
    assert row["camera_id"] == "2010093"
    assert row["session_uuid"] == "2026_05_05_17_45_30_cam2010093"
    assert row["recording_id"] == "2026_05_05_17_45_30"
    assert row["recording_name"] == "2026_05_05_17_45_30_cam2010093"
    assert row["session_start_iso8601_utc"] == "2026-05-05T21:45:30Z"
    assert row["recording_type"] == "behavior"
    assert row["recording_subtype"] == "free"
    assert row["behavior_mode"] == "free"
    assert row["artifact_schema_id"] == "video_only_v1"
    assert row["dish_design"] == "cedar"
    assert row["rig_id"] == "omnifin0"
    assert row["arena_id"] == "arena_1"
    assert row["num_dishes"] == "1"
    assert row["fish_per_dish"] == "1"

    loaded = organize_recordings._load_video_only_rows(output, source_root=source_root)
    plan = organize_recordings._build_video_only_plan(
        loaded[0],
        dest_root=tmp_path / "recordings",
        rename_cams=True,
    )
    assert plan.name == "2026_05_05_17_45_30_cam2010093"
    assert plan.cam_files[0].dest_name == "Cam2010093_2026_05_05_17_45_30_cam2010093.mp4"
    assert plan.cam_files[1].dest_name == "Cam2010093_2026_05_05_17_45_30_cam2010093_meta.csv"
    assert plan.meta["dish_design"] == "cedar"
    assert plan.meta["num_dishes"] == "1"
    assert plan.meta["fish_per_dish"] == "1"


def test_draft_manifest_can_require_camera_metadata_csv(tmp_path: Path) -> None:
    source_root = tmp_path / "staging"
    source_root.mkdir()
    (source_root / "Cam2010094.mp4").write_bytes(b"video")
    output = tmp_path / "manifest.csv"

    rc = mod.main(
        [
            str(source_root),
            "--output",
            str(output),
            "--require-camera-metadata-csv",
        ]
    )

    assert rc == 1
    assert not output.exists()
