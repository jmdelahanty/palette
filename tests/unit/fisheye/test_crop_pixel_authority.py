from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from fisheye.shared.acquisition_publication_status import (
    ACQUISITION_AUTHORITY_PUBLISHED,
    EXTERNAL_ACQUISITION_AUTHORITY_MODE,
    EXTERNAL_ACQUISITION_PUBLISHED_REASON,
    build_acquisition_authority_publication_status,
)
from fisheye.shared.import_source_fingerprint import source_stat_fingerprint_attrs
from fisheye.shared.zarr import crop_pixel_authority as module
from fisheye.shared.zarr.crop_pixel_authority import (
    CROP_SOURCE_PIXEL_BINDING_SCHEMA_ID,
    CROP_SOURCE_VIDEO_DECODE_PROFILE,
    BoundCropPixelAuthority,
    CropSourcePixelAuthorityError,
    bind_external_video_crop_pixel_authority,
    bind_refined_crop_source_pixel_authority,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.refined_detection_crop_source import (
    BoundRefinedDetectionCropSource,
)


class _FakeNode:
    def __init__(self, *, attrs=None, children=None) -> None:
        self.attrs = dict(attrs or {})
        self._children = dict(children or {})

    def get(self, name):
        return self._children.get(name)


@dataclass
class _Evidence:
    record: object
    record_ref: str
    record_sha256: str
    verified: int = 0

    def assert_verified(self) -> None:
        self.verified += 1


def _live_fingerprint(video: Path, metadata: dict[str, object]) -> dict[str, object]:
    attrs = source_stat_fingerprint_attrs(
        video,
        attr_prefix="source_video",
        extra={
            "codec": metadata["codec"],
            "pix_fmt": metadata["pix_fmt"],
            "width": metadata["width"],
            "height": metadata["height"],
            "fps": metadata["fps"],
            "frame_count": metadata["total_frames"],
        },
    )
    return {
        "strategy": attrs["source_video_fingerprint_strategy"],
        "value": attrs["source_video_fingerprint"],
        "size_bytes": attrs["source_video_size_bytes"],
        "mtime_ns": attrs["source_video_mtime_ns"],
        "relocation_stable": False,
    }


def _fixture(monkeypatch, tmp_path: Path):
    recording = tmp_path / "recording"
    video = recording / "cams" / "camera.mp4"
    archive = recording / "zarr" / "recording_analysis.zarr"
    video.parent.mkdir(parents=True)
    archive.mkdir(parents=True)
    video.write_bytes(b"exact source video")

    metadata: dict[str, object] = {
        "schema_id": "palette.source_video_metadata.v2",
        "layout": "single_video",
        "camera_id": "cam2010095",
        "locator": {
            "kind": "recording_relative",
            "relative_path": "cams/camera.mp4",
        },
        "source_path": str(video.resolve()),
        "width": 100,
        "height": 80,
        "total_frames": 4,
        "fps": 100.0,
        "codec": "hevc",
        "pix_fmt": "yuv420p",
    }
    metadata["file_fingerprint"] = _live_fingerprint(video, metadata)
    root = _FakeNode(
        attrs={
            "recording_id": "recording-001",
            "recording_path": str(recording),
            "source_video_metadata": metadata,
        },
        children={"raw_video": _FakeNode()},
    )
    status = build_acquisition_authority_publication_status(
        status=ACQUISITION_AUTHORITY_PUBLISHED,
        reason_code=EXTERNAL_ACQUISITION_PUBLISHED_REASON,
        authority_mode=EXTERNAL_ACQUISITION_AUTHORITY_MODE,
        authority_path="analysis/acquisition_camera_frames/cam2010095",
    )
    ownership = _Evidence(
        record=SimpleNamespace(mode=EXTERNAL_ACQUISITION_AUTHORITY_MODE),
        record_ref=(
            "/analysis/acquisition_camera_frames/cam2010095"
            "@acquisition_import_ownership"
        ),
        record_sha256="1" * 64,
    )
    record = SimpleNamespace(
        recording_id="recording-001",
        camera_id="cam2010095",
        source_total_frames=4,
        frame_count=4,
        width_px=100,
        height_px=80,
        source_video_metadata=metadata,
        source_video_metadata_sha256=canonical_json_sha256(metadata),
    )
    acquisition = _Evidence(
        record=record,
        record_ref=(
            "/analysis/acquisition_camera_frames/cam2010095@acquisition_camera_frame"
        ),
        record_sha256="2" * 64,
    )
    monkeypatch.setattr(module, "open_zarr_root", lambda *_args, **_kwargs: root)
    monkeypatch.setattr(
        module,
        "load_acquisition_authority_publication_status",
        lambda _root: status,
    )
    monkeypatch.setattr(
        module,
        "load_persisted_acquisition_camera_authority",
        lambda _root, *, expected_camera_id=None: (ownership, acquisition),
    )
    return archive, video, ownership, acquisition


def _bind(archive: Path) -> BoundCropPixelAuthority:
    return bind_external_video_crop_pixel_authority(
        archive,
        expected_recording_identity="recording-001",
        expected_camera_identity="cam2010095",
        expected_n_frames=4,
        expected_source_width=100,
        expected_source_height=80,
    )


def test_external_video_adapter_binds_live_acquisition_and_decode_contract(
    monkeypatch,
    tmp_path: Path,
) -> None:
    archive, video, ownership, acquisition = _fixture(monkeypatch, tmp_path)

    bound = _bind(archive)

    assert bound.source_video_path == video.resolve()
    assert bound.pixel_authority.recording_identity == "recording-001"
    assert bound.pixel_authority.camera_identity == "cam2010095"
    assert bound.pixel_authority.n_frames == 4
    assert bound.pixel_authority.source_width == 100
    assert bound.pixel_authority.source_height == 80
    assert bound.binding_document_digest == canonical_json_sha256(
        bound.binding_document
    )
    assert bound.binding_document["schema_id"] == (CROP_SOURCE_PIXEL_BINDING_SCHEMA_ID)
    assert bound.binding_document["decoded_pixel_contract"]["profile"] == (
        CROP_SOURCE_VIDEO_DECODE_PROFILE
    )
    assert bound.binding_document["decoded_pixel_contract"]["source_pixels"] == (
        "raw_camera_video"
    )
    assert ownership.verified == 1
    assert acquisition.verified == 1
    bound.assert_verified()
    assert ownership.verified == 2
    assert acquisition.verified == 2


def test_external_video_adapter_detects_live_source_replacement(
    monkeypatch,
    tmp_path: Path,
) -> None:
    archive, video, _ownership, _acquisition = _fixture(monkeypatch, tmp_path)
    bound = _bind(archive)

    video.write_bytes(b"replacement source video with different bytes")

    with pytest.raises(CropSourcePixelAuthorityError, match="Live source video"):
        bound.assert_verified()


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"expected_recording_identity": "another-recording"}, "recording identity"),
        ({"expected_camera_identity": "another-camera"}, "camera identity"),
        ({"expected_n_frames": 5}, "frame count"),
        ({"expected_source_width": 101}, "source width"),
        ({"expected_source_height": 81}, "source height"),
    ],
)
def test_external_video_adapter_rejects_refined_source_mismatch(
    monkeypatch,
    tmp_path: Path,
    override: dict[str, object],
    match: str,
) -> None:
    archive, _video, _ownership, _acquisition = _fixture(monkeypatch, tmp_path)
    kwargs: dict[str, object] = {
        "expected_recording_identity": "recording-001",
        "expected_camera_identity": "cam2010095",
        "expected_n_frames": 4,
        "expected_source_width": 100,
        "expected_source_height": 80,
        **override,
    }

    with pytest.raises(CropSourcePixelAuthorityError, match=match):
        bind_external_video_crop_pixel_authority(archive, **kwargs)


def test_external_video_adapter_rejects_non_external_publication(
    monkeypatch,
    tmp_path: Path,
) -> None:
    archive, _video, _ownership, _acquisition = _fixture(monkeypatch, tmp_path)
    status = SimpleNamespace(
        status=ACQUISITION_AUTHORITY_PUBLISHED,
        authority_mode="materialized_source_frames_v1",
        authority_path="analysis/acquisition_camera_frames/cam2010095",
    )
    monkeypatch.setattr(
        module,
        "load_acquisition_authority_publication_status",
        lambda _root: status,
    )

    with pytest.raises(CropSourcePixelAuthorityError, match="external-video"):
        _bind(archive)


def test_bound_authority_constructor_is_sealed(monkeypatch, tmp_path: Path) -> None:
    archive, _video, ownership, acquisition = _fixture(monkeypatch, tmp_path)
    valid = _bind(archive)

    with pytest.raises(CropSourcePixelAuthorityError, match="cannot be constructed"):
        BoundCropPixelAuthority(
            archive_path=archive,
            source_video_path=valid.source_video_path,
            pixel_authority=valid.pixel_authority,
            binding_document=valid.binding_document,
            import_ownership=ownership,
            acquisition_frame=acquisition,
            expected_recording_identity="recording-001",
            expected_camera_identity="cam2010095",
            expected_n_frames=4,
            expected_source_width=100,
            expected_source_height=80,
        )


def test_refined_handoff_supplies_exact_source_dimensions(
    monkeypatch,
    tmp_path: Path,
) -> None:
    archive = tmp_path / "source.zarr"
    archive.mkdir()
    source = BoundRefinedDetectionCropSource(
        archive_path=archive,
        run_id="refined",
        run_path="refined_detect_runs/refined",
        instances_path="refined_detect_runs/refined/instances",
        selection_mode="approved_authoritative_refined_v1",
        manifest={
            "payload": {
                "snapshot_lineage": {
                    "manual_instance_key_allocator": {
                        "recording_identity": "recording-001"
                    }
                }
            }
        },
        dimensions=SimpleNamespace(
            n_frames=4,
            source_width=100,
            source_height=80,
        ),
        arrays={},
        run_group=None,
        instances_group=None,
        logical_content_digest="3" * 64,
        handoff_manifest={},
    )
    sentinel = object()
    captured = {}

    def fake_bind(path, **kwargs):
        captured["path"] = path
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(module, "bind_external_video_crop_pixel_authority", fake_bind)

    result = bind_refined_crop_source_pixel_authority(
        source,
        expected_camera_identity="cam2010095",
    )

    assert result is sentinel
    assert captured == {
        "path": archive,
        "expected_recording_identity": "recording-001",
        "expected_camera_identity": "cam2010095",
        "expected_n_frames": 4,
        "expected_source_width": 100,
        "expected_source_height": 80,
    }
