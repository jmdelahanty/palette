from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.shared.import_source_fingerprint import source_stat_fingerprint_attrs
from fisheye.shared.detection_candidate import (
    DETECTION_CANDIDATE_BUILD_AUTHORITY_ATTR,
    node_local_detection_candidate_authority,
)
from fisheye.shared.import_video_metadata import (
    publish_external_video_acquisition_authority,
)
from fisheye.shared.pixel_frame_authority import (
    load_persisted_acquisition_camera_authority,
)
from fisheye.shared.run_provenance import validate_run_provenance
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
    assert staged.attrs[DETECTION_CANDIDATE_BUILD_AUTHORITY_ATTR] == (
        node_local_detection_candidate_authority()
    )
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


def test_default_local_publish_provenance_passes_completion_gate(
    tmp_path: Path,
) -> None:
    provenance = mod._resolve_detection_run_provenance(  # noqa: SLF001
        supplied=None,
        source_zarr=tmp_path / "recording_analysis.zarr",
        video_path=tmp_path / "camera.mp4",
        model_path=tmp_path / "model.pt",
        model_sha256="a" * 64,
        model_run_id="model-run",
        model_set_id="model-set",
        model_created_utc="2026-07-25T00:00:00Z",
        run_name="detect_test",
        config_path="detect.yaml",
        conf_threshold=0.25,
        iou_threshold=0.7,
        max_det=10,
        batch_size=16,
        resize_dims=[640, 640],
        imgsz=[640, 640],
        decode_backend="pynvvc_nv12_rgb",
        detect_row_shard_rows=131_072,
        detect_frame_shard_rows=131_072,
        use_gpu=True,
        copy_backend="python",
    )

    validation = validate_run_provenance(provenance)
    assert validation.valid, validation.errors
    assert provenance["command"] == "fisheye.utils.run_detection_local_publish"
    assert provenance["input_run_ids"] == {
        "model_run": "model-run",
        "model_set": "model-set",
    }
    assert provenance["params"]["output_policy"] == mod.PUBLISH_POLICY


def test_local_publish_rejects_incomplete_supplied_provenance_before_inference(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="requires valid run provenance before inference",
    ):
        mod._resolve_detection_run_provenance(  # noqa: SLF001
            supplied={"schema_id": "transport-policy-only"},
            source_zarr=tmp_path / "recording_analysis.zarr",
            video_path=tmp_path / "camera.mp4",
            model_path=tmp_path / "model.pt",
            model_sha256="a" * 64,
            model_run_id="model-run",
            model_set_id="model-set",
            model_created_utc=None,
            run_name="detect_test",
            config_path=None,
            conf_threshold=None,
            iou_threshold=None,
            max_det=None,
            batch_size=None,
            resize_dims=None,
            imgsz=None,
            decode_backend=None,
            detect_row_shard_rows=131_072,
            detect_frame_shard_rows=131_072,
            use_gpu=None,
            copy_backend="python",
        )
