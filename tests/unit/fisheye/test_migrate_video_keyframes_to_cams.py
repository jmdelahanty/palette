from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.utils import migrate_video_keyframes_to_cams as mod


def _make_recording(tmp_path: Path) -> tuple[Path, Path]:
    recording_dir = tmp_path / "recordings" / "sleepyfish_2026_05_05_17_45_30_cam2010093"
    (recording_dir / "cams").mkdir(parents=True)
    (recording_dir / "derived").mkdir()
    video = recording_dir / "cams" / "Cam2010093_sleepyfish_2026_05_05_17_45_30_cam2010093.mp4"
    video.write_bytes(b"video")
    keyframe = recording_dir / "derived" / "Cam2010093_sleepyfish_2026_05_05_17_45_30_cam2010093_keyframe.json"
    keyframe.write_text("{}", encoding="utf-8")
    (recording_dir / "recording_manifest.json").write_text(
        json.dumps(
            {
                "files": {
                    "cams": [f"cams/{video.name}"],
                    "derived": [f"derived/{keyframe.name}"],
                }
            }
        ),
        encoding="utf-8",
    )
    return recording_dir, keyframe


def test_migrate_video_keyframes_to_cams_apply(tmp_path: Path) -> None:
    recording_dir, keyframe = _make_recording(tmp_path)

    rc = mod.main([str(recording_dir), "--apply", "--migration-id", "testmigration"])

    assert rc == 0
    dest = recording_dir / "cams" / keyframe.name
    assert dest.exists()
    assert not keyframe.exists()
    manifest = json.loads((recording_dir / "recording_manifest.json").read_text(encoding="utf-8"))
    assert f"cams/{keyframe.name}" in manifest["files"]["cams"]
    assert f"derived/{keyframe.name}" not in manifest["files"]["derived"]
    assert manifest["metadata_migrations"][0]["migration_type"] == "video_keyframe_to_cams_v1"


def test_migrate_video_keyframes_to_cams_dry_run_does_not_mutate(tmp_path: Path) -> None:
    recording_dir, keyframe = _make_recording(tmp_path)

    rc = mod.main([str(recording_dir), "--dry-run", "--migration-id", "testmigration"])

    assert rc == 0
    assert keyframe.exists()
    assert not (recording_dir / "cams" / keyframe.name).exists()
    manifest = json.loads((recording_dir / "recording_manifest.json").read_text(encoding="utf-8"))
    assert f"derived/{keyframe.name}" in manifest["files"]["derived"]
