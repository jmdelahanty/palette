from __future__ import annotations

from pathlib import Path

import pytest
import zarr

from fisheye.shared.pixel_frame_authority import (
    load_persisted_acquisition_camera_authority,
)
from fisheye.utils import migrate_external_video_acquisition_authority as migration


def _archive(tmp_path: Path, *, total_frames: int | None = None) -> tuple[Path, Path]:
    recording_id = "sleepyfish_2026_05_05_17_45_30_cam2010093"
    recording = tmp_path / recording_id
    source = recording / "cams" / f"Cam2010093_{recording_id}.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"encoded video identity")
    zarr_path = recording / "zarr" / f"{recording_id}_analysis.zarr"
    zarr_path.parent.mkdir()
    root = zarr.open_group(str(zarr_path), mode="w")
    root.attrs.update(
        {
            "recording_id": recording_id,
            "recording_path": str(recording),
            "camera_serials": ["2010093"],
            "width": 4512,
            "height": 4512,
        }
    )
    if total_frames is not None:
        root.attrs["total_frames"] = total_frames
    root.require_group("raw_video")
    return zarr_path, source


def _probe(source: Path) -> dict[str, object]:
    return {
        "source_video": source.name,
        "source_path": str(source),
        "width": 4512,
        "height": 4512,
        "total_frames": 1_188_000,
        "fps": 30.0,
        "duration_seconds": 39_600.0,
        "codec": "hevc",
        "pix_fmt": "gray",
        "imageio_metadata": {"nframes": float("inf")},
    }


def test_apply_migrates_missing_locator_and_seals_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zarr_path, source = _archive(tmp_path)
    monkeypatch.setattr(migration, "probe_video_metadata", _probe)

    dry_run, _ = migration.plan_migration(zarr_path)
    assert dry_run.status == "would_migrate_and_seal"
    assert dry_run.camera_id == "2010093"
    assert dry_run.source_video_path == str(source)

    result = migration.apply_migration(zarr_path, consolidate_metadata=False)
    assert result.status == "migrated_and_sealed"
    assert result.authority_path == "analysis/acquisition_camera_frames/2010093"

    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    assert root.attrs["camera_id"] == "2010093"
    assert root.attrs["source_video_metadata"]["camera_id"] == "2010093"
    assert "imageio_metadata" not in root.attrs["source_video_metadata"]
    assert root.attrs["source_video_metadata"]["file_fingerprint"]["size_bytes"] > 0
    ownership, frame = load_persisted_acquisition_camera_authority(
        root,
        expected_camera_id="2010093",
    )
    assert ownership.record.mode == "external_video_v1"
    assert frame.record.width_px == 4512
    assert frame.record.frame_count == 1_188_000


def test_conflicting_legacy_geometry_fails_without_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zarr_path, _source = _archive(tmp_path, total_frames=100)
    monkeypatch.setattr(migration, "probe_video_metadata", _probe)

    with pytest.raises(ValueError, match="total_frames conflicts"):
        migration.plan_migration(zarr_path)

    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    assert root.attrs.get("camera_id") is None
    assert "acquisition_camera_frames" not in root["analysis"] if "analysis" in root else True


def test_camera_evidence_must_agree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zarr_path, _source = _archive(tmp_path)
    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    root.attrs["camera_id"] = "2010094"
    monkeypatch.setattr(migration, "probe_video_metadata", _probe)

    with pytest.raises(ValueError, match="Camera identity evidence conflicts"):
        migration.plan_migration(zarr_path)
