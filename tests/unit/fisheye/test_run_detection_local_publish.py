from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.shared.import_source_fingerprint import source_stat_fingerprint_attrs
from fisheye.shared.import_video_metadata import (
    publish_external_video_acquisition_authority,
)
from fisheye.shared.pixel_frame_authority import (
    load_persisted_acquisition_camera_authority,
)
from fisheye.shared.source_video_metadata import build_source_video_metadata_v2
from fisheye.utils import run_detection_local_publish as mod


def _external_archive(tmp_path: Path) -> tuple[Path, Path]:
    recording = tmp_path / "recording"
    video = recording / "cams" / "camera-01.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"canonical external video")
    source = {
        "source_path": str(video.resolve()),
        "camera_id": "camera-01",
        "width": 4512,
        "height": 4512,
        "total_frames": 100,
        "fps": 100.0,
        "codec": "hevc",
        "pix_fmt": "yuv420p",
    }
    fingerprint = source_stat_fingerprint_attrs(
        video,
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
    archive = recording / "zarr" / "recording_analysis.zarr"
    root = zarr.open_group(archive, mode="w", zarr_format=3)
    root.attrs.update(
        {
            "recording_id": "recording-id",
            "camera_id": "camera-01",
            "recording_path": str(recording.resolve()),
            "source_video_path": str(video.resolve()),
            "source_path": str(video.resolve()),
            "source_video_metadata": metadata,
        }
    )
    root.require_group("raw_video")
    publish_external_video_acquisition_authority(root)
    return archive, video


def test_prepare_local_overlay_copies_only_verified_acquisition_metadata(
    tmp_path: Path,
) -> None:
    source, _video = _external_archive(tmp_path)
    local = tmp_path / "scratch" / "analysis.zarr"

    report = mod._prepare_local_overlay(source, local)  # noqa: SLF001

    assert report == {
        "authority_mode": "external_video_v1",
        "authority_path": "analysis/acquisition_camera_frames/camera-01",
        "camera_id": "camera-01",
        "recording_id": "recording-id",
        "source_total_frames": 100,
        "source_width_px": 4512,
        "source_height_px": 4512,
        "staged_raw_video_arrays": 0,
    }
    staged = zarr.open_group(local, mode="r", use_consolidated=False)
    assert tuple(staged["raw_video"].array_keys()) == ()
    ownership, acquisition = load_persisted_acquisition_camera_authority(staged)
    ownership.assert_verified()
    acquisition.assert_verified()


def test_prepare_local_overlay_rejects_materialized_raw_video(tmp_path: Path) -> None:
    source, _video = _external_archive(tmp_path)
    root = zarr.open_group(source, mode="a", use_consolidated=False)
    root["raw_video"].create_array(
        "frames",
        data=np.zeros((1, 2, 2), dtype=np.uint8),
    )

    with pytest.raises(RuntimeError, match="refuses to stage raw_video arrays"):
        mod._prepare_local_overlay(  # noqa: SLF001
            source,
            tmp_path / "scratch" / "analysis.zarr",
        )


def test_shared_source_camera_authorities_are_complete_and_idempotent(
    tmp_path: Path,
) -> None:
    source, _video = _external_archive(tmp_path)

    first = mod._ensure_shared_source_camera_authorities(source)  # noqa: SLF001
    second = mod._ensure_shared_source_camera_authorities(source)  # noqa: SLF001

    assert second == first
    assert first["point_record_ref"].endswith("/continuous@pixel_frame_authority")
    assert first["bbox_record_ref"].endswith(
        "/pixel_edge_half_open@pixel_frame_authority"
    )


def test_verify_model_requires_matching_registered_digest(tmp_path: Path) -> None:
    model = tmp_path / "model.pt"
    model.write_bytes(b"registered model")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()

    verified = mod._verify_model(model, digest)  # noqa: SLF001

    assert verified["sha256"] == digest
    with pytest.raises(RuntimeError, match="digest mismatch"):
        mod._verify_model(model, "0" * 64)  # noqa: SLF001
