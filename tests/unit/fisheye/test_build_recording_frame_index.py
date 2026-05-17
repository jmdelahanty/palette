from __future__ import annotations

import csv
import json
from pathlib import Path

import pyarrow.parquet as pq

from fisheye.utils.build_recording_frame_index import build_recording_frame_index


def _write_metadata(path: Path, frame_ids: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame_id", "timestamp", "timestamp_sys"])
        writer.writeheader()
        for idx, frame_id in enumerate(frame_ids):
            writer.writerow(
                {
                    "frame_id": frame_id,
                    "timestamp": 1000 + idx,
                    "timestamp_sys": 2000 + idx,
                }
            )


def _write_clip_bundle(root: Path, clip_id: str, frame_ids: list[int]) -> dict[str, object]:
    clip_index = int(clip_id.rsplit("_", 1)[1])
    clip_dir = root / "clips" / clip_id
    video = clip_dir / "Cam2010093_example.mp4"
    metadata = clip_dir / "Cam2010093_example_meta.csv"
    keyframe = clip_dir / "Cam2010093_example_keyframe.json"
    manifest = clip_dir / "clip_manifest.json"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"video")
    _write_metadata(metadata, frame_ids)
    keyframe.write_text(
        json.dumps({"total_frames": len(frame_ids), "fps": 2, "keyframe_frames": [0]}),
        encoding="utf-8",
    )
    manifest.write_text(json.dumps({"clip_id": clip_id}), encoding="utf-8")
    return {
        "recording_id": "rec_a",
        "session_id": "rec_a",
        "producer": "test",
        "recording_backend_mode": "rolling_clips",
        "camera_serial": "2010093",
        "clip_index": clip_index,
        "clip_id": clip_id,
        "clip_directory": f"clips/{clip_id}",
        "video_path": f"clips/{clip_id}/{video.name}",
        "metadata_path": f"clips/{clip_id}/{metadata.name}",
        "keyframe_path": f"clips/{clip_id}/{keyframe.name}",
        "clip_manifest_path": f"clips/{clip_id}/clip_manifest.json",
        "frame_count": len(frame_ids),
        "first_recording_frame_id": frame_ids[0],
        "last_recording_frame_id": frame_ids[-1],
    }


def test_build_recording_frame_index_from_clipped_recording(tmp_path: Path) -> None:
    root = tmp_path / "rec_a"
    first = _write_clip_bundle(root, "clip_000000", [1, 2, 3])
    second = _write_clip_bundle(root, "clip_000001", [4, 5])
    (root / "recording_clip_index.json").write_text(
        json.dumps(
            {
                "recording_id": "rec_a",
                "session_id": "rec_a",
                "producer": "test_index",
                "recording_backend_mode": "materialized_stream_copy",
                "clips": [first, second],
            }
        ),
        encoding="utf-8",
    )

    result = build_recording_frame_index(root, write_csv=True)

    assert result["status"] == "ok"
    assert result["source_layout"] == "rolling_clips"
    assert result["row_count"] == 5
    assert Path(result["parquet_path"]).exists()
    assert Path(result["manifest_path"]).exists()
    assert Path(result["csv_path"]).exists()

    table = pq.read_table(result["parquet_path"])
    rows = table.to_pylist()
    assert rows[0]["clip_id"] == "clip_000000"
    assert rows[0]["recording_frame_id"] == 1
    assert rows[0]["parent_frame_index"] == 0
    assert rows[0]["clip_local_frame_index"] == 0
    assert rows[3]["clip_id"] == "clip_000001"
    assert rows[3]["recording_frame_id"] == 4
    assert rows[3]["parent_frame_index"] == 3
    assert rows[3]["clip_local_frame_index"] == 0

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["status"] == "ok"
    assert manifest["source_authority"] == "recording_clip_index + per_clip_metadata_csv"
    assert manifest["recording_frame_id_min"] == 1
    assert manifest["recording_frame_id_max"] == 5


def test_build_recording_frame_index_single_video_fallback(tmp_path: Path) -> None:
    root = tmp_path / "rec_single"
    video = root / "cams" / "Cam2010094_single.mp4"
    metadata = root / "cams" / "Cam2010094_single_meta.csv"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    _write_metadata(metadata, [0, 1, 2])

    result = build_recording_frame_index(root)

    assert result["status"] == "ok"
    assert result["source_layout"] == "single_video"
    table = pq.read_table(result["parquet_path"])
    rows = table.to_pylist()
    assert [row["recording_frame_id"] for row in rows] == [0, 1, 2]
    assert [row["parent_frame_index"] for row in rows] == [0, 1, 2]
    assert {row["clip_id"] for row in rows} == {"full_video"}
    assert {row["clip_local_frame_index"] for row in rows} == {0, 1, 2}
    assert rows[0]["video_path"].endswith("Cam2010094_single.mp4")


def test_build_recording_frame_index_reports_clip_index_mismatch_in_dry_run(tmp_path: Path) -> None:
    root = tmp_path / "rec_bad"
    row = _write_clip_bundle(root, "clip_000000", [1, 2])
    row["frame_count"] = 3
    (root / "recording_clip_index.json").write_text(
        json.dumps({"recording_id": "rec_bad", "clips": [row]}),
        encoding="utf-8",
    )

    result = build_recording_frame_index(root, dry_run=True)

    assert result["status"] == "fail"
    assert result["wrote_parquet"] is False
    failures = [check for check in result["checks"] if check["status"] != "ok"]
    assert any(check["code"] == "metadata_rows_match_clip_index_frame_count" for check in failures)
