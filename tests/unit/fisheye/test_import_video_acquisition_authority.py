from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr
from zarr.storage import LocalStore

import fisheye.capture.import_video as import_video_module
from fisheye.capture.import_video import (
    ACQUISITION_AUTHORITY_NOT_PUBLISHED,
    ACQUISITION_AUTHORITY_PENDING,
    ACQUISITION_AUTHORITY_PUBLISHED,
    ACQUISITION_AUTHORITY_STATUS_ATTR,
    COMPLETE_ENCODED_OBJECT_WRITE_POLICY,
    _build_import_source_video_metadata,
    _load_organized_recording_context,
    mark_materialized_import_noncanonical,
    publish_completed_materialized_acquisition_authority,
    stamp_stored_zarr_frame_identity_mapping,
)
from fisheye.shared.acquisition_publication_status import (
    ACQUISITION_AUTHORITY_PUBLISHED as SHARED_ACQUISITION_AUTHORITY_PUBLISHED,
    EXTERNAL_ACQUISITION_AUTHORITY_MODE,
    load_acquisition_authority_publication_status,
)
from fisheye.shared.import_video_metadata import (
    publish_external_video_acquisition_authority,
)
from fisheye.shared.pixel_frame_authority import (
    ACQUISITION_PHYSICAL_CHUNK_MANIFEST_ATTR,
    PixelFrameAuthorityError,
    load_persisted_acquisition_camera_authority,
)


def _organized_source(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    recording = tmp_path / "recording"
    source = recording / "cams" / "camera.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"exact encoded source identity")
    manifest = {
        "recording_id": "recording-id",
        "camera_id": "camera-01",
    }
    (recording / "recording_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    context = _load_organized_recording_context(source)
    assert context is not None
    return source, context


def test_materialized_import_metadata_uses_decoded_observation_and_ffprobe(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "camera.mp4"
    source.write_bytes(b"video")
    monkeypatch.setattr(
        import_video_module,
        "probe_ffprobe_video_metadata",
        lambda _path: {
            "codec": "hevc",
            "pix_fmt": "yuv420p",
            "format_tags": {"encoder": "orange"},
        },
    )

    class _Reader:
        @staticmethod
        def get_avg_fps() -> float:
            return 100.0

    metadata = import_video_module._get_video_metadata(  # noqa: SLF001
        source,
        _Reader(),
        4512,
        4512,
        139295,
    )

    assert metadata["total_frames"] == 139295
    assert metadata["fps"] == 100.0
    assert metadata["codec"] == "hevc"
    assert metadata["pix_fmt"] == "yuv420p"
    assert "imageio_metadata" not in metadata


def _build_complete_archive(
    tmp_path: Path,
    *,
    sampled: bool = False,
    sharded: bool = True,
) -> Path:
    source, context = _organized_source(tmp_path)
    source_metadata, fingerprint_attrs = _build_import_source_video_metadata(
        {
            "source_video": source.name,
            "source_path": str(source),
            "width": 4,
            "height": 3,
            "total_frames": 4,
            "fps": 20.0,
            "duration_seconds": 0.2,
            "codec": "test",
            "pix_fmt": "gray",
        },
        organized_context=context,
    )
    zarr_path = tmp_path / "recording" / "zarr" / "analysis.zarr"
    zarr_path.parent.mkdir(parents=True)
    with zarr.config.set({"array.write_empty_chunks": True}):
        root = zarr.open_group(
            store=LocalStore(str(zarr_path)),
            mode="w",
            zarr_format=3,
        )
        root.attrs.update(
            {
                "recording_id": context["recording_id"],
                "camera_id": context["camera_id"],
                "recording_path": context["recording_path"],
                "recording_manifest_path": context["recording_manifest_path"],
                "source_video_metadata": source_metadata,
                "zarr_purpose": "training" if sampled else "analysis",
                **fingerprint_attrs,
            }
        )
        raw = root.create_group("raw_video")
        raw.attrs.update(
            {
                "import_method": "standard_zarr",
                "import_stage": "complete",
                "import_mode": "sampled" if sampled else "full",
                "decode_backend": "test_decoder",
                "source_decode_surface": "test_gray_uint8",
                "source_path": str(source.resolve()),
                "write_empty_chunks": True,
                "complete_encoded_object_write_policy": (
                    COMPLETE_ENCODED_OBJECT_WRITE_POLICY
                ),
            }
        )
        array_kwargs: dict[str, object] = {
            "data": np.zeros((4, 3, 4), dtype=np.uint8),
            "chunks": (1, 3, 4),
            "compressors": [],
        }
        if sharded:
            array_kwargs["shards"] = (2, 3, 4)
        raw.create_array("images_full", **array_kwargs)
        stamp_stored_zarr_frame_identity_mapping(raw, 4)
        if sampled:
            raw.create_array(
                "original_frame_indices",
                data=np.arange(4, dtype=np.int64),
                chunks=(4,),
            )
        root.store.close()
    return zarr_path


def test_organized_manifest_requires_exact_explicit_identities(tmp_path: Path) -> None:
    source = tmp_path / "recording" / "cams" / "camera.mp4"
    source.parent.mkdir(parents=True)
    source.touch()
    manifest_path = tmp_path / "recording" / "recording_manifest.json"
    manifest_path.write_text(json.dumps({"camera_id": "camera-01"}), encoding="utf-8")

    with pytest.raises(ValueError, match="recording_id"):
        _load_organized_recording_context(source)

    manifest_path.write_text(
        json.dumps({"recording_id": "recording-id", "camera_id": " camera-01"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="camera_id"):
        _load_organized_recording_context(source)

    for invalid_id in (".", "..", "camera/01", "camera\\01", "camera 01"):
        manifest_path.write_text(
            json.dumps(
                {"recording_id": "recording-id", "camera_id": invalid_id}
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="canonical-path-segment camera_id"):
            _load_organized_recording_context(source)


def test_ad_hoc_import_has_no_inferred_recording_identity(tmp_path: Path) -> None:
    source = tmp_path / "camera.mp4"
    source.touch()

    assert _load_organized_recording_context(source) is None


def test_all_zero_sharded_full_import_publishes_actual_outer_objects(
    tmp_path: Path,
) -> None:
    zarr_path = _build_complete_archive(tmp_path, sharded=True)

    result = publish_completed_materialized_acquisition_authority(zarr_path)

    root = zarr.open_group(store=LocalStore(str(zarr_path)), mode="r")
    status = root.attrs[ACQUISITION_AUTHORITY_STATUS_ATTR]
    assert status["status"] == ACQUISITION_AUTHORITY_PUBLISHED
    assert status["authority_mode"] == "materialized_source_frames_v1"
    assert result["authority_path"] == "analysis/acquisition_camera_frames/camera-01"
    manifest = root["raw_video/manifests/images_full_materialization"].attrs[
        ACQUISITION_PHYSICAL_CHUNK_MANIFEST_ATTR
    ]
    assert manifest["entry_count"] == 2
    assert [entry["chunk_indices"] for entry in manifest["entries"]] == [
        [0, 0, 0],
        [1, 0, 0],
    ]
    assert all(entry["encoded_size_bytes"] > 0 for entry in manifest["entries"])
    ownership, frame = load_persisted_acquisition_camera_authority(
        root,
        expected_camera_id="camera-01",
    )
    assert ownership.record.mode == "materialized_source_frames_v1"
    assert frame.record.frame_count == 4
    root.store.close()

    with pytest.raises(PixelFrameAuthorityError, match="cannot be downgraded"):
        mark_materialized_import_noncanonical(
            zarr_path,
            reason_code="organized_recording_identity_absent",
        )


def test_external_video_publication_is_mode_explicit_and_idempotent(
    tmp_path: Path,
) -> None:
    source, context = _organized_source(tmp_path)
    source_metadata, _fingerprint_attrs = _build_import_source_video_metadata(
        {
            "source_video": source.name,
            "source_path": str(source),
            "width": 4,
            "height": 3,
            "total_frames": 4,
            "fps": 20.0,
            "duration_seconds": 0.2,
            "codec": "test",
            "pix_fmt": "gray",
        },
        organized_context=context,
    )
    zarr_path = tmp_path / "external-analysis.zarr"
    root = zarr.open_group(
        store=LocalStore(str(zarr_path)),
        mode="w",
        zarr_format=3,
    )
    root.attrs.update(
        {
            "recording_id": context["recording_id"],
            "camera_id": context["camera_id"],
            "source_video_metadata": source_metadata,
        }
    )
    root.require_group("raw_video")

    first = publish_external_video_acquisition_authority(root)
    second = publish_external_video_acquisition_authority(root)

    assert first == second
    status = load_acquisition_authority_publication_status(root)
    assert status.status == SHARED_ACQUISITION_AUTHORITY_PUBLISHED
    assert status.authority_mode == EXTERNAL_ACQUISITION_AUTHORITY_MODE
    assert status.authority_path == first["authority_path"]
    assert "manifests" not in root["raw_video"]
    ownership, frame = load_persisted_acquisition_camera_authority(
        root,
        expected_camera_id=context["camera_id"],
    )
    assert ownership.record.mode == EXTERNAL_ACQUISITION_AUTHORITY_MODE
    assert frame.record.frame_array is None
    del root.attrs[ACQUISITION_AUTHORITY_STATUS_ATTR]
    del root["raw_video"].attrs[ACQUISITION_AUTHORITY_STATUS_ATTR]
    with pytest.raises(PixelFrameAuthorityError, match="requires explicit repair"):
        publish_external_video_acquisition_authority(root)
    root.store.close()


def test_stale_consolidated_metadata_cannot_hide_publication_nodes(
    tmp_path: Path,
) -> None:
    zarr_path = _build_complete_archive(tmp_path)
    zarr.consolidate_metadata(LocalStore(str(zarr_path)))

    publish_completed_materialized_acquisition_authority(zarr_path)

    root = zarr.open_group(
        store=LocalStore(str(zarr_path)),
        mode="r",
        use_consolidated=False,
    )
    assert "raw_video/manifests/images_full_materialization" in root
    assert "analysis/acquisition_camera_frames/camera-01" in root
    load_persisted_acquisition_camera_authority(
        root,
        expected_camera_id="camera-01",
    )
    root.store.close()


def test_statusless_existing_materialized_authority_requires_explicit_repair(
    tmp_path: Path,
) -> None:
    zarr_path = _build_complete_archive(tmp_path)
    publish_completed_materialized_acquisition_authority(zarr_path)
    root = zarr.open_group(
        store=LocalStore(str(zarr_path)),
        mode="r+",
        use_consolidated=False,
    )
    del root.attrs[ACQUISITION_AUTHORITY_STATUS_ATTR]
    del root["raw_video"].attrs[ACQUISITION_AUTHORITY_STATUS_ATTR]
    root.store.close()

    with pytest.raises(PixelFrameAuthorityError, match="requires explicit repair"):
        publish_completed_materialized_acquisition_authority(zarr_path)


def test_sampled_training_import_cannot_publish_acquisition_authority(
    tmp_path: Path,
) -> None:
    zarr_path = _build_complete_archive(tmp_path, sampled=True)

    with pytest.raises(PixelFrameAuthorityError, match="Training archives"):
        publish_completed_materialized_acquisition_authority(zarr_path)

    root = zarr.open_group(store=LocalStore(str(zarr_path)), mode="r")
    assert "analysis" not in root
    root.store.close()


def test_late_failure_is_explicitly_pending_and_exact_retry_completes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zarr_path = _build_complete_archive(tmp_path)

    with monkeypatch.context() as patch:
        patch.setattr(
            import_video_module,
            "stamp_acquisition_camera_frame",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("injected late frame stamp failure")
            ),
        )
        with pytest.raises(RuntimeError, match="injected late frame stamp failure"):
            publish_completed_materialized_acquisition_authority(zarr_path)

    root = zarr.open_group(store=LocalStore(str(zarr_path)), mode="r")
    pending = root.attrs[ACQUISITION_AUTHORITY_STATUS_ATTR]
    assert pending == root["raw_video"].attrs[ACQUISITION_AUTHORITY_STATUS_ATTR]
    assert pending["status"] == ACQUISITION_AUTHORITY_PENDING
    assert pending["resumption_policy"] == (
        "retry_exact_archive_completion_idempotent_v1"
    )
    assert "raw_video/manifests/images_full_materialization" in root
    assert "analysis/acquisition_camera_frames/camera-01" in root
    root.store.close()

    publish_completed_materialized_acquisition_authority(zarr_path)

    root = zarr.open_group(store=LocalStore(str(zarr_path)), mode="r")
    assert root.attrs[ACQUISITION_AUTHORITY_STATUS_ATTR]["status"] == (
        ACQUISITION_AUTHORITY_PUBLISHED
    )
    load_persisted_acquisition_camera_authority(
        root,
        expected_camera_id="camera-01",
    )
    root.store.close()


def test_noncanonical_status_does_not_create_coordinate_authority(tmp_path: Path) -> None:
    zarr_path = _build_complete_archive(tmp_path)

    mark_materialized_import_noncanonical(
        zarr_path,
        reason_code="organized_recording_identity_absent",
    )

    root = zarr.open_group(store=LocalStore(str(zarr_path)), mode="r")
    status = root.attrs[ACQUISITION_AUTHORITY_STATUS_ATTR]
    assert status == root["raw_video"].attrs[ACQUISITION_AUTHORITY_STATUS_ATTR]
    assert status["status"] == ACQUISITION_AUTHORITY_NOT_PUBLISHED
    assert status["reason_code"] == "organized_recording_identity_absent"
    assert status["authority_mode"] is None
    assert "analysis" not in root
    root.store.close()

    with pytest.raises(PixelFrameAuthorityError, match="silently upgraded"):
        publish_completed_materialized_acquisition_authority(zarr_path)


def test_missing_empty_chunk_policy_fails_before_manifest_publication(
    tmp_path: Path,
) -> None:
    zarr_path = _build_complete_archive(tmp_path)
    root = zarr.open_group(
        store=LocalStore(str(zarr_path)),
        mode="r+",
        use_consolidated=False,
    )
    root["raw_video"].attrs["write_empty_chunks"] = False
    root.store.close()

    with pytest.raises(PixelFrameAuthorityError, match="write_empty_chunks=True"):
        publish_completed_materialized_acquisition_authority(zarr_path)

    root = zarr.open_group(store=LocalStore(str(zarr_path)), mode="r")
    assert "manifests" not in root["raw_video"]
    root.store.close()
