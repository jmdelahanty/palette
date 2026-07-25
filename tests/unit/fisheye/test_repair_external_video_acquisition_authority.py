from __future__ import annotations

from pathlib import Path

import zarr

from fisheye.shared.acquisition_publication_status import (
    ACQUISITION_AUTHORITY_PUBLISHED,
    ACQUISITION_AUTHORITY_STATUS_ATTR,
    load_acquisition_authority_publication_status,
)
from fisheye.shared.import_source_fingerprint import source_stat_fingerprint_attrs
from fisheye.shared.import_video_metadata import (
    publish_external_video_acquisition_authority,
)
from fisheye.shared.pixel_frame_authority import (
    load_persisted_acquisition_camera_authority,
)
from fisheye.shared.source_video_metadata import build_source_video_metadata_v2
from fisheye.utils.repair_external_video_acquisition_authority import (
    repair_external_video_acquisition_authority,
)


def _metadata_only_archive(
    tmp_path: Path,
    *,
    include_legacy_imageio: bool = False,
) -> tuple[Path, Path]:
    recording = tmp_path / "recording"
    video_path = recording / "cams" / "camera-01.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"exact external video bytes")
    source = {
        "source_path": str(video_path.resolve()),
        "camera_id": "camera-01",
        "width": 4512,
        "height": 4512,
        "total_frames": 100,
        "fps": 100.0,
        "codec": "hevc",
        "pix_fmt": "yuv420p",
    }
    fingerprint = source_stat_fingerprint_attrs(
        video_path,
        attr_prefix="source_video",
        extra={
            "codec": source["codec"],
            "pix_fmt": source["pix_fmt"],
            "width": source["width"],
            "height": source["height"],
            "fps": source["fps"],
            "frame_count": source["total_frames"],
        },
    )
    metadata = build_source_video_metadata_v2(
        source,
        recording_path=recording,
        fingerprint_attrs=fingerprint,
    )
    if include_legacy_imageio:
        metadata["imageio_metadata"] = {
            "plugin": "ffmpeg",
            "nframes": float("inf"),
        }
    zarr_path = recording / "zarr" / "recording_analysis.zarr"
    root = zarr.open_group(zarr_path, mode="w", zarr_format=3)
    root.attrs.update(
        {
            "recording_id": "recording-id",
            "camera_id": "camera-01",
            "recording_path": str(recording.resolve()),
            "source_video_path": str(video_path.resolve()),
            "source_path": str(video_path.resolve()),
            "source_video_metadata": metadata,
        }
    )
    if include_legacy_imageio:
        root.attrs["imageio_metadata"] = metadata["imageio_metadata"]
    root.require_group("raw_video")
    return zarr_path, video_path


def test_repair_external_video_authority_dry_run_apply_and_idempotence(
    tmp_path: Path,
) -> None:
    zarr_path, video_path = _metadata_only_archive(tmp_path)

    dry_run = repair_external_video_acquisition_authority(zarr_path)
    assert dry_run["status"] == "ok"
    assert dry_run["action"] == "would_publish"
    assert dry_run["live_fingerprint_match"] is True
    assert dry_run["source_video_path"] == str(video_path.resolve())

    applied = repair_external_video_acquisition_authority(zarr_path, apply=True)
    assert applied["status"] == "ok"
    assert applied["action"] == "publish"

    root = zarr.open_group(zarr_path, mode="r", use_consolidated=False)
    status = load_acquisition_authority_publication_status(root)
    assert status.status == ACQUISITION_AUTHORITY_PUBLISHED
    ownership, frame = load_persisted_acquisition_camera_authority(
        root,
        expected_camera_id="camera-01",
    )
    assert ownership.record.mode == "external_video_v1"
    assert frame.record.source_total_frames == 100

    second = repair_external_video_acquisition_authority(zarr_path, apply=True)
    assert second["status"] == "ok"
    assert second["action"] == "already_complete"


def test_repair_external_video_authority_rejects_changed_source(tmp_path: Path) -> None:
    zarr_path, video_path = _metadata_only_archive(tmp_path)
    video_path.write_bytes(b"changed external video bytes with a new size")

    report = repair_external_video_acquisition_authority(zarr_path, apply=True)

    assert report["status"] == "blocked"
    assert report["live_fingerprint_match"] is False
    assert "differs from the persisted stat_v1 fingerprint" in report["error"]
    root = zarr.open_group(zarr_path, mode="r", use_consolidated=False)
    assert ACQUISITION_AUTHORITY_STATUS_ATTR not in root.attrs


def test_repair_external_video_authority_rejects_statusless_existing_node(
    tmp_path: Path,
) -> None:
    zarr_path, _video_path = _metadata_only_archive(tmp_path)
    root = zarr.open_group(zarr_path, mode="a", use_consolidated=False)
    publish_external_video_acquisition_authority(root)
    del root.attrs[ACQUISITION_AUTHORITY_STATUS_ATTR]
    del root["raw_video"].attrs[ACQUISITION_AUTHORITY_STATUS_ATTR]

    report = repair_external_video_acquisition_authority(zarr_path, apply=True)

    assert report["status"] == "blocked"
    assert "Statusless acquisition authority is ambiguous" in report["error"]


def test_repair_removes_legacy_imageio_before_publication(tmp_path: Path) -> None:
    zarr_path, _video_path = _metadata_only_archive(
        tmp_path,
        include_legacy_imageio=True,
    )

    dry_run = repair_external_video_acquisition_authority(zarr_path)
    assert dry_run["status"] == "ok"
    assert dry_run["action"] == "would_publish"
    assert dry_run["legacy_imageio_metadata_present"] is True
    assert dry_run["legacy_imageio_metadata_action"] == "would_remove"

    applied = repair_external_video_acquisition_authority(zarr_path, apply=True)
    assert applied["status"] == "ok"
    assert applied["legacy_imageio_metadata_action"] == "removed"

    root = zarr.open_group(zarr_path, mode="r", use_consolidated=False)
    assert "imageio_metadata" not in root.attrs
    assert "imageio_metadata" not in root.attrs["source_video_metadata"]
    assert load_acquisition_authority_publication_status(root).status == (
        ACQUISITION_AUTHORITY_PUBLISHED
    )
