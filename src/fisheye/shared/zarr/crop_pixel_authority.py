"""Bind geometry-only crops to one live, published source-pixel authority.

The crop schema deliberately persists no pixels.  This strict binder therefore
reopens the archive-owned acquisition authority, verifies its publication
status and live source-file fingerprints, and converts that evidence into the
``CropPixelAuthority`` stored by a crop run.

Single external full-frame videos and recording-wide clipped collections use
separate typed binders because their frame maps differ. Materialized arrays and
acquisition crop videos remain unsupported. No compatibility aliases, dtype
probing, inference, or fallback are permitted here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from fisheye.shared.acquisition_publication_status import (
    ACQUISITION_AUTHORITY_PUBLISHED,
    CLIPPED_EXTERNAL_ACQUISITION_AUTHORITY_MODE,
    EXTERNAL_ACQUISITION_AUTHORITY_MODE,
    AcquisitionPublicationStatusError,
    load_acquisition_authority_publication_status,
)
from fisheye.shared.clipped_video_collection import (
    ClippedVideoCollectionEvidenceError,
    VerifiedClippedVideoCollectionFiles,
    verify_clipped_video_collection_live_files,
)
from fisheye.shared.import_source_fingerprint import source_stat_fingerprint_attrs
from fisheye.shared.pixel_frame_authority import (
    BoundAcquisitionCameraFrame,
    BoundAcquisitionImportOwnership,
    PixelFrameAuthorityError,
    load_persisted_acquisition_camera_authority,
)
from fisheye.shared.source_video_metadata import (
    SourceVideoMetadataError,
    resolve_source_video,
)
from fisheye.shared.zarr.crop_manifest import CropPixelAuthority
from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_sha256,
)
from fisheye.shared.zarr.refined_detection_crop_source import (
    BoundRefinedDetectionCropSource,
)
from fisheye.shared.zarr_io import open_zarr_root


CROP_SOURCE_PIXEL_BINDING_SCHEMA_ID = (
    "palette.crop_geometry.source_pixel_authority_binding"
)
CROP_SOURCE_PIXEL_BINDING_SCHEMA_VERSION = 1
CROP_SOURCE_VIDEO_DECODE_PROFILE = "orange_mono_pynvvc_luma_uint8_v1"
CROP_CLIPPED_SOURCE_VIDEO_DECODE_PROFILE = (
    "orange_mono_pynvvc_luma_uint8_clipped_collection_v1"
)

_BINDING_SEAL = object()


class CropSourcePixelAuthorityError(RuntimeError):
    """Raised when source pixels cannot be bound without inference."""


def _required_text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise CropSourcePixelAuthorityError(
            f"{name} must be an exact unpadded nonempty string."
        )
    return value


def _required_positive_int(value: object, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise CropSourcePixelAuthorityError(
            f"{name} must be an exact positive integer."
        )
    return value


def _expected_live_fingerprint(
    video_path: Path,
    metadata: Mapping[str, Any],
) -> dict[str, object]:
    live = source_stat_fingerprint_attrs(
        video_path,
        attr_prefix="source_video",
        extra={
            "codec": metadata.get("codec"),
            "pix_fmt": metadata.get("pix_fmt"),
            "width": metadata.get("width"),
            "height": metadata.get("height"),
            "fps": metadata.get("fps"),
            "frame_count": metadata.get("total_frames"),
        },
    )
    return {
        "strategy": live["source_video_fingerprint_strategy"],
        "value": live["source_video_fingerprint"],
        "size_bytes": live["source_video_size_bytes"],
        "mtime_ns": live["source_video_mtime_ns"],
        "relocation_stable": False,
    }


def _decoded_pixel_contract() -> dict[str, object]:
    """Return the exact full-frame decode contract used by crop materializers."""

    return {
        "schema_id": "palette.crop_geometry.source_video_decode",
        "schema_version": 1,
        "profile": CROP_SOURCE_VIDEO_DECODE_PROFILE,
        "source_pixels": "raw_camera_video",
        "source_frame_representation": "orange_mono_encoded_full_frame_video",
        "decode_backend": "pynvvc_luma",
        "decoded_surface": "nv12_y_plane_uint8",
        "dtype": "uint8",
        "channels": "grayscale",
        "axis_order": "yx",
        "range_semantics": "orange_mono8_full_range_0_255",
        "container_color_range_handling": (
            "read_direct_y_plane_without_decoder_range_remap"
        ),
        "crop_sampling": "integer_half_open_xywh",
    }


def _clipped_decoded_pixel_contract() -> dict[str, object]:
    """Return the exact collection routing and per-clip decode contract."""

    return {
        "schema_id": "palette.crop_geometry.source_video_decode",
        "schema_version": 1,
        "profile": CROP_CLIPPED_SOURCE_VIDEO_DECODE_PROFILE,
        "source_pixels": "raw_camera_video_clipped_collection",
        "source_frame_representation": (
            "orange_mono_encoded_full_frame_video_clips"
        ),
        "frame_routing": (
            "recording_acquisition_frame_to_indexed_clip_local_frame_v1"
        ),
        "decode_backend": "pynvvc_luma",
        "decoded_surface": "nv12_y_plane_uint8",
        "dtype": "uint8",
        "channels": "grayscale",
        "axis_order": "yx",
        "range_semantics": "orange_mono8_full_range_0_255",
        "container_color_range_handling": (
            "read_direct_y_plane_without_decoder_range_remap"
        ),
        "crop_sampling": "integer_half_open_xywh",
    }


def _binding_document(
    *,
    status: Any,
    ownership: BoundAcquisitionImportOwnership,
    acquisition: BoundAcquisitionCameraFrame,
) -> dict[str, object]:
    record = acquisition.record
    metadata = record.source_video_metadata
    return {
        "schema_id": CROP_SOURCE_PIXEL_BINDING_SCHEMA_ID,
        "schema_version": CROP_SOURCE_PIXEL_BINDING_SCHEMA_VERSION,
        "provider_profile": "published_external_full_frame_video_v1",
        "publication_status": status.to_dict(),
        "acquisition_import_ownership": {
            "record_ref": ownership.record_ref,
            "record_sha256": ownership.record_sha256,
            "mode": ownership.record.mode,
        },
        "acquisition_camera_frame": {
            "record_ref": acquisition.record_ref,
            "record_sha256": acquisition.record_sha256,
            "recording_identity": record.recording_id,
            "camera_identity": record.camera_id,
            "frame_index_domain": "zero_based_acquisition_camera_frame",
            "n_frames": record.source_total_frames,
            "source_width": record.width_px,
            "source_height": record.height_px,
        },
        "source_video": {
            "metadata_digest": record.source_video_metadata_sha256,
            "locator": metadata["locator"],
            "file_fingerprint": metadata["file_fingerprint"],
        },
        "decoded_pixel_contract": _decoded_pixel_contract(),
        "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
    }


def _clipped_binding_document(
    *,
    status: Any,
    ownership: BoundAcquisitionImportOwnership,
    acquisition: BoundAcquisitionCameraFrame,
) -> dict[str, object]:
    record = acquisition.record
    metadata = record.source_video_metadata
    collection = metadata["collection"]
    return {
        "schema_id": CROP_SOURCE_PIXEL_BINDING_SCHEMA_ID,
        "schema_version": CROP_SOURCE_PIXEL_BINDING_SCHEMA_VERSION,
        "provider_profile": "published_external_clipped_video_collection_v1",
        "publication_status": status.to_dict(),
        "acquisition_import_ownership": {
            "record_ref": ownership.record_ref,
            "record_sha256": ownership.record_sha256,
            "mode": ownership.record.mode,
        },
        "acquisition_camera_frame": {
            "record_ref": acquisition.record_ref,
            "record_sha256": acquisition.record_sha256,
            "recording_identity": record.recording_id,
            "camera_identity": record.camera_id,
            "frame_index_domain": "zero_based_acquisition_camera_frame",
            "n_frames": record.source_total_frames,
            "source_width": record.width_px,
            "source_height": record.height_px,
        },
        "source_video_collection": {
            "metadata_digest": record.source_video_metadata_sha256,
            "schema_id": metadata["schema_id"],
            "layout": metadata["layout"],
            "locator": metadata["locator"],
            "recording_frame_index": collection["recording_frame_index"],
            "member_count": len(collection["members"]),
            "collection_sha256": collection["collection_sha256"],
        },
        "decoded_pixel_contract": _clipped_decoded_pixel_contract(),
        "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
    }


@dataclass(frozen=True, init=False)
class BoundCropPixelAuthority:
    """Sealed crop authority reconstructed from live acquisition evidence."""

    archive_path: Path
    source_video_path: Path | None
    source_video_paths: tuple[Path, ...]
    source_index_paths: tuple[Path, ...]
    pixel_authority: CropPixelAuthority
    binding_document: Mapping[str, Any]
    binding_document_digest: str
    import_ownership: BoundAcquisitionImportOwnership = field(repr=False, compare=False)
    acquisition_frame: BoundAcquisitionCameraFrame = field(repr=False, compare=False)
    _expected_recording_identity: str = field(repr=False, compare=False)
    _expected_camera_identity: str | None = field(repr=False, compare=False)
    _expected_n_frames: int = field(repr=False, compare=False)
    _expected_source_width: int = field(repr=False, compare=False)
    _expected_source_height: int = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        archive_path: Path,
        source_video_path: Path | None,
        source_video_paths: tuple[Path, ...],
        source_index_paths: tuple[Path, ...],
        pixel_authority: CropPixelAuthority,
        binding_document: Mapping[str, Any],
        import_ownership: BoundAcquisitionImportOwnership,
        acquisition_frame: BoundAcquisitionCameraFrame,
        expected_recording_identity: str,
        expected_camera_identity: str | None,
        expected_n_frames: int,
        expected_source_width: int,
        expected_source_height: int,
        _verification_seal: object | None = None,
    ) -> None:
        if _verification_seal is not _BINDING_SEAL:
            raise CropSourcePixelAuthorityError(
                "Bound crop pixel authorities cannot be constructed directly."
            )
        document = dict(binding_document)
        digest = canonical_json_sha256(document)
        if digest != pixel_authority.authority_manifest_digest:
            raise CropSourcePixelAuthorityError(
                "Crop pixel authority digest does not bind its source document."
            )
        if (
            not source_video_paths
            or any(not isinstance(item, Path) for item in source_video_paths)
            or any(not isinstance(item, Path) for item in source_index_paths)
            or (
                source_video_path is not None
                and source_video_paths != (source_video_path,)
            )
        ):
            raise CropSourcePixelAuthorityError(
                "Bound crop pixel authority has invalid source-file evidence."
            )
        object.__setattr__(self, "archive_path", archive_path)
        object.__setattr__(self, "source_video_path", source_video_path)
        object.__setattr__(self, "source_video_paths", source_video_paths)
        object.__setattr__(self, "source_index_paths", source_index_paths)
        object.__setattr__(self, "pixel_authority", pixel_authority)
        object.__setattr__(self, "binding_document", document)
        object.__setattr__(self, "binding_document_digest", digest)
        object.__setattr__(self, "import_ownership", import_ownership)
        object.__setattr__(self, "acquisition_frame", acquisition_frame)
        object.__setattr__(
            self, "_expected_recording_identity", expected_recording_identity
        )
        object.__setattr__(self, "_expected_camera_identity", expected_camera_identity)
        object.__setattr__(self, "_expected_n_frames", expected_n_frames)
        object.__setattr__(self, "_expected_source_width", expected_source_width)
        object.__setattr__(self, "_expected_source_height", expected_source_height)
        object.__setattr__(self, "_seal", _verification_seal)

    def assert_verified(self) -> None:
        """Reopen all evidence and reject any post-binding drift."""

        if self._seal is not _BINDING_SEAL:
            raise CropSourcePixelAuthorityError(
                "Crop pixel authority is not sealed verification evidence."
            )
        current = bind_crop_pixel_authority(
            self.archive_path,
            expected_recording_identity=self._expected_recording_identity,
            expected_camera_identity=self._expected_camera_identity,
            expected_n_frames=self._expected_n_frames,
            expected_source_width=self._expected_source_width,
            expected_source_height=self._expected_source_height,
        )
        if (
            current.pixel_authority != self.pixel_authority
            or current.binding_document != self.binding_document
            or current.source_video_path != self.source_video_path
            or current.source_video_paths != self.source_video_paths
            or current.source_index_paths != self.source_index_paths
        ):
            raise CropSourcePixelAuthorityError(
                "Source pixel authority changed after it was bound."
            )


def bind_external_video_crop_pixel_authority(
    archive_path: str | Path,
    *,
    expected_recording_identity: str,
    expected_n_frames: int,
    expected_source_width: int,
    expected_source_height: int,
    expected_camera_identity: str | None = None,
) -> BoundCropPixelAuthority:
    """Reopen one external-video authority and bind it for crop publication."""

    path = Path(archive_path).expanduser().resolve()
    if not path.is_dir() or path.suffix != ".zarr":
        raise CropSourcePixelAuthorityError(
            f"Crop source is not a Zarr directory: {path}"
        )
    recording_identity = _required_text(
        expected_recording_identity,
        name="expected_recording_identity",
    )
    camera_identity = (
        None
        if expected_camera_identity is None
        else _required_text(
            expected_camera_identity,
            name="expected_camera_identity",
        )
    )
    n_frames = _required_positive_int(expected_n_frames, name="expected_n_frames")
    source_width = _required_positive_int(
        expected_source_width,
        name="expected_source_width",
    )
    source_height = _required_positive_int(
        expected_source_height,
        name="expected_source_height",
    )

    root = open_zarr_root(path, mode="r")
    try:
        status = load_acquisition_authority_publication_status(root)
        if (
            status.status != ACQUISITION_AUTHORITY_PUBLISHED
            or status.authority_mode != EXTERNAL_ACQUISITION_AUTHORITY_MODE
            or status.authority_path is None
        ):
            raise CropSourcePixelAuthorityError(
                "Crop source requires one published external-video acquisition "
                "authority."
            )
        ownership, acquisition = load_persisted_acquisition_camera_authority(
            root,
            expected_camera_id=camera_identity,
        )
        ownership.assert_verified()
        acquisition.assert_verified()
    except (AcquisitionPublicationStatusError, PixelFrameAuthorityError) as exc:
        raise CropSourcePixelAuthorityError(
            f"Invalid persisted acquisition authority: {exc}"
        ) from exc

    record = acquisition.record
    expected_authority_path = f"analysis/acquisition_camera_frames/{record.camera_id}"
    if (
        ownership.record.mode != EXTERNAL_ACQUISITION_AUTHORITY_MODE
        or status.authority_path != expected_authority_path
    ):
        raise CropSourcePixelAuthorityError(
            "Acquisition publication status disagrees with its ownership/frame "
            "authority."
        )
    expected_values = (
        ("recording identity", record.recording_id, recording_identity),
        ("frame count", record.source_total_frames, n_frames),
        ("source width", record.width_px, source_width),
        ("source height", record.height_px, source_height),
    )
    if camera_identity is not None and record.camera_id != camera_identity:
        raise CropSourcePixelAuthorityError(
            f"Acquisition camera identity {record.camera_id!r} differs from crop "
            f"source {camera_identity!r}."
        )
    for label, actual, expected in expected_values:
        if type(actual) is not type(expected) or actual != expected:
            raise CropSourcePixelAuthorityError(
                f"Acquisition {label} {actual!r} differs from crop source {expected!r}."
            )
    if record.frame_count != record.source_total_frames:
        raise CropSourcePixelAuthorityError(
            "External-video authority does not cover the complete acquisition "
            "frame domain."
        )

    try:
        resolved = resolve_source_video(
            root,
            zarr_path=path,
            require_exists=True,
        )
    except SourceVideoMetadataError as exc:
        raise CropSourcePixelAuthorityError(
            f"Cannot resolve the authoritative source video: {exc}"
        ) from exc
    metadata = record.source_video_metadata
    try:
        live_fingerprint = _expected_live_fingerprint(resolved.path, metadata)
    except OSError as exc:
        raise CropSourcePixelAuthorityError(
            f"Cannot stat the authoritative source video: {exc}"
        ) from exc
    if metadata.get("file_fingerprint") != live_fingerprint:
        raise CropSourcePixelAuthorityError(
            "Live source video differs from the persisted stat_v1 fingerprint."
        )

    document = _binding_document(
        status=status,
        ownership=ownership,
        acquisition=acquisition,
    )
    document_digest = canonical_json_sha256(document)
    authority = CropPixelAuthority(
        authority_id=(
            f"{acquisition.record_ref}#decode={CROP_SOURCE_VIDEO_DECODE_PROFILE}"
        ),
        authority_manifest_digest=document_digest,
        recording_identity=record.recording_id,
        camera_identity=record.camera_id,
        n_frames=record.source_total_frames,
        source_width=record.width_px,
        source_height=record.height_px,
    )
    return BoundCropPixelAuthority(
        archive_path=path,
        source_video_path=resolved.path,
        source_video_paths=(resolved.path,),
        source_index_paths=(),
        pixel_authority=authority,
        binding_document=document,
        import_ownership=ownership,
        acquisition_frame=acquisition,
        expected_recording_identity=recording_identity,
        expected_camera_identity=camera_identity,
        expected_n_frames=n_frames,
        expected_source_width=source_width,
        expected_source_height=source_height,
        _verification_seal=_BINDING_SEAL,
    )


def _recording_directory(root: Any, *, archive_path: Path) -> Path:
    attrs = getattr(root, "attrs", None)
    declared = attrs.get("recording_path") if isinstance(attrs, Mapping) else None
    if declared is not None:
        if type(declared) is not str or not declared or declared != declared.strip():
            raise CropSourcePixelAuthorityError(
                "Clipped crop source recording_path is malformed."
            )
        recording = Path(declared).expanduser().resolve()
    elif archive_path.parent.name == "zarr":
        recording = archive_path.parent.parent
    else:
        raise CropSourcePixelAuthorityError(
            "Clipped crop source requires an exact recording directory."
        )
    if not recording.is_dir():
        raise CropSourcePixelAuthorityError(
            f"Clipped crop source recording directory not found: {recording}"
        )
    return recording


def bind_clipped_video_collection_crop_pixel_authority(
    archive_path: str | Path,
    *,
    expected_recording_identity: str,
    expected_n_frames: int,
    expected_source_width: int,
    expected_source_height: int,
    expected_camera_identity: str | None = None,
) -> BoundCropPixelAuthority:
    """Bind one published recording-wide clipped-video collection."""

    path = Path(archive_path).expanduser().resolve()
    if not path.is_dir() or path.suffix != ".zarr":
        raise CropSourcePixelAuthorityError(
            f"Crop source is not a Zarr directory: {path}"
        )
    recording_identity = _required_text(
        expected_recording_identity,
        name="expected_recording_identity",
    )
    camera_identity = (
        None
        if expected_camera_identity is None
        else _required_text(
            expected_camera_identity,
            name="expected_camera_identity",
        )
    )
    n_frames = _required_positive_int(expected_n_frames, name="expected_n_frames")
    source_width = _required_positive_int(
        expected_source_width,
        name="expected_source_width",
    )
    source_height = _required_positive_int(
        expected_source_height,
        name="expected_source_height",
    )

    root = open_zarr_root(path, mode="r")
    try:
        status = load_acquisition_authority_publication_status(root)
        if (
            status.status != ACQUISITION_AUTHORITY_PUBLISHED
            or status.authority_mode
            != CLIPPED_EXTERNAL_ACQUISITION_AUTHORITY_MODE
            or status.authority_path is None
        ):
            raise CropSourcePixelAuthorityError(
                "Crop source requires one published clipped-video collection "
                "acquisition authority."
            )
        ownership, acquisition = load_persisted_acquisition_camera_authority(
            root,
            expected_camera_id=camera_identity,
        )
        ownership.assert_verified()
        acquisition.assert_verified()
    except (AcquisitionPublicationStatusError, PixelFrameAuthorityError) as exc:
        raise CropSourcePixelAuthorityError(
            f"Invalid persisted acquisition authority: {exc}"
        ) from exc

    record = acquisition.record
    expected_authority_path = f"analysis/acquisition_camera_frames/{record.camera_id}"
    if (
        ownership.record.mode != CLIPPED_EXTERNAL_ACQUISITION_AUTHORITY_MODE
        or status.authority_path != expected_authority_path
    ):
        raise CropSourcePixelAuthorityError(
            "Acquisition publication status disagrees with its clipped "
            "ownership/frame authority."
        )
    expected_values = (
        ("recording identity", record.recording_id, recording_identity),
        ("frame count", record.source_total_frames, n_frames),
        ("source width", record.width_px, source_width),
        ("source height", record.height_px, source_height),
    )
    if camera_identity is not None and record.camera_id != camera_identity:
        raise CropSourcePixelAuthorityError(
            f"Acquisition camera identity {record.camera_id!r} differs from crop "
            f"source {camera_identity!r}."
        )
    for label, actual, expected in expected_values:
        if type(actual) is not type(expected) or actual != expected:
            raise CropSourcePixelAuthorityError(
                f"Acquisition {label} {actual!r} differs from crop source "
                f"{expected!r}."
            )
    if record.frame_count != record.source_total_frames:
        raise CropSourcePixelAuthorityError(
            "Clipped-video authority does not cover the complete acquisition "
            "frame domain."
        )

    recording = _recording_directory(root, archive_path=path)
    try:
        live_files: VerifiedClippedVideoCollectionFiles = (
            verify_clipped_video_collection_live_files(
                recording,
                record.source_video_metadata,
            )
        )
    except ClippedVideoCollectionEvidenceError as exc:
        raise CropSourcePixelAuthorityError(
            f"Invalid live clipped-video collection: {exc}"
        ) from exc

    document = _clipped_binding_document(
        status=status,
        ownership=ownership,
        acquisition=acquisition,
    )
    document_digest = canonical_json_sha256(document)
    authority = CropPixelAuthority(
        authority_id=(
            f"{acquisition.record_ref}"
            f"#decode={CROP_CLIPPED_SOURCE_VIDEO_DECODE_PROFILE}"
            f"#collection={live_files.collection_sha256}"
        ),
        authority_manifest_digest=document_digest,
        recording_identity=record.recording_id,
        camera_identity=record.camera_id,
        n_frames=record.source_total_frames,
        source_width=record.width_px,
        source_height=record.height_px,
    )
    return BoundCropPixelAuthority(
        archive_path=path,
        source_video_path=None,
        source_video_paths=live_files.member_paths,
        source_index_paths=live_files.index_paths,
        pixel_authority=authority,
        binding_document=document,
        import_ownership=ownership,
        acquisition_frame=acquisition,
        expected_recording_identity=recording_identity,
        expected_camera_identity=camera_identity,
        expected_n_frames=n_frames,
        expected_source_width=source_width,
        expected_source_height=source_height,
        _verification_seal=_BINDING_SEAL,
    )


def bind_crop_pixel_authority(
    archive_path: str | Path,
    *,
    expected_recording_identity: str,
    expected_n_frames: int,
    expected_source_width: int,
    expected_source_height: int,
    expected_camera_identity: str | None = None,
) -> BoundCropPixelAuthority:
    """Dispatch only between explicitly published source-pixel profiles."""

    path = Path(archive_path).expanduser().resolve()
    if not path.is_dir() or path.suffix != ".zarr":
        raise CropSourcePixelAuthorityError(
            f"Crop source is not a Zarr directory: {path}"
        )
    try:
        root = open_zarr_root(path, mode="r")
        status = load_acquisition_authority_publication_status(root)
    except AcquisitionPublicationStatusError as exc:
        raise CropSourcePixelAuthorityError(
            f"Invalid acquisition publication status: {exc}"
        ) from exc
    kwargs = {
        "expected_recording_identity": expected_recording_identity,
        "expected_camera_identity": expected_camera_identity,
        "expected_n_frames": expected_n_frames,
        "expected_source_width": expected_source_width,
        "expected_source_height": expected_source_height,
    }
    if status.authority_mode == EXTERNAL_ACQUISITION_AUTHORITY_MODE:
        return bind_external_video_crop_pixel_authority(path, **kwargs)
    if status.authority_mode == CLIPPED_EXTERNAL_ACQUISITION_AUTHORITY_MODE:
        return bind_clipped_video_collection_crop_pixel_authority(path, **kwargs)
    raise CropSourcePixelAuthorityError(
        "Crop source acquisition authority mode is unsupported: "
        f"{status.authority_mode!r}."
    )


def bind_refined_crop_source_pixel_authority(
    source: BoundRefinedDetectionCropSource,
    *,
    expected_camera_identity: str | None = None,
) -> BoundCropPixelAuthority:
    """Bind source pixels using the exact dimensions of a refined handoff."""

    if type(source) is not BoundRefinedDetectionCropSource:
        raise CropSourcePixelAuthorityError(
            "A validated BoundRefinedDetectionCropSource is required."
        )
    lineage = source.manifest["payload"]["snapshot_lineage"]
    allocator = lineage["manual_instance_key_allocator"]
    return bind_crop_pixel_authority(
        source.archive_path,
        expected_recording_identity=allocator["recording_identity"],
        expected_camera_identity=expected_camera_identity,
        expected_n_frames=source.dimensions.n_frames,
        expected_source_width=source.dimensions.source_width,
        expected_source_height=source.dimensions.source_height,
    )


__all__ = [
    "CROP_CLIPPED_SOURCE_VIDEO_DECODE_PROFILE",
    "CROP_SOURCE_PIXEL_BINDING_SCHEMA_ID",
    "CROP_SOURCE_PIXEL_BINDING_SCHEMA_VERSION",
    "CROP_SOURCE_VIDEO_DECODE_PROFILE",
    "BoundCropPixelAuthority",
    "CropSourcePixelAuthorityError",
    "bind_clipped_video_collection_crop_pixel_authority",
    "bind_crop_pixel_authority",
    "bind_external_video_crop_pixel_authority",
    "bind_refined_crop_source_pixel_authority",
]
