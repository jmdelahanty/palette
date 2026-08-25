"""Sealed, space-specific pixel-frame authorities for canonical transforms.

An extent is not a coordinate frame.  This module promotes an exact persisted
extent to one controlled pixel frame only when the producer also supplies the
typed lineage that gives that frame meaning.  There is deliberately no public
``space_id=...`` constructor:

* source-camera frames require a sealed acquisition-owned frame-domain record;
* selected stimulus canvases require builder-sealed display evidence;
* arena-relative canvases require a selected canvas plus exact persisted
  integer arena placement geometry;
* ROI frames require exact crop-placement bytes and observation identity; and
* detector model-input frames require exact persisted preprocessing bytes and
  a target ROI frame.

Selected-calibration H5 evidence does not mint an acquisition camera frame. It
is used later, by the camera-to-display transform authority. All Zarr-side
records and arrays retained here are revalidated live.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import hashlib
import itertools
import json
import os
import re
import stat
from pathlib import PurePosixPath
from typing import Any, Mapping

import numpy as np

from fisheye.shared.acquisition_publication_status import (
    CLIPPED_EXTERNAL_ACQUISITION_AUTHORITY_MODE,
    EXTERNAL_ACQUISITION_AUTHORITY_MODE,
    MATERIALIZED_ACQUISITION_AUTHORITY_MODE,
)
from fisheye.shared.archive_identity import (
    ArchiveIdentity,
    ArchiveIdentityError,
    archive_identity,
    require_same_archive,
)
from fisheye.shared.clipped_video_collection import (
    SOURCE_VIDEO_COLLECTION_FINGERPRINT_STRATEGY,
    SOURCE_VIDEO_COLLECTION_LAYOUT,
    SOURCE_VIDEO_COLLECTION_LOCATOR_KIND,
    SOURCE_VIDEO_COLLECTION_METADATA_SCHEMA_ID,
    SOURCE_VIDEO_COLLECTION_SCHEMA_ID,
)
from fisheye.shared.coordinate_descriptor import PIXEL_CONVENTIONS
from fisheye.shared.coordinate_identity import (
    INSTANCE_KEY_ARRAY_REF,
    INSTANCE_KEY_MODE,
    OBSERVATION_INSTANCE_DOMAIN,
    BoundRowIdentityContract,
    RowIdentityContractError,
)
from fisheye.shared.coordinate_reference import (
    BoundReferenceExtent,
    CoordinateReferenceError,
    bind_array_reference_extent,
    canonical_node_path,
    verify_bound_reference_extent,
)
from fisheye.shared.model_input_transform import ModelInputTransform
from fisheye.shared.proof_verification import verify_persisted_proof


PIXEL_FRAME_AUTHORITY_SCHEMA_ID = "palette.pixel_frame_authority"
PIXEL_FRAME_AUTHORITY_SCHEMA_VERSION = 2
PIXEL_FRAME_AUTHORITY_ATTR = "pixel_frame_authority"
PIXEL_FRAME_AUTHORITY_DIGEST_ATTR = "pixel_frame_authority_sha256"
PIXEL_FRAME_AUTHORITY_CANONICALIZATION = "canonical_json_sort_keys_v1"
ARRAY_VALUES_CANONICALIZATION = "numpy_dtype_shape_c_order_bytes_v1"

ACQUISITION_CAMERA_FRAME_SCHEMA_ID = "palette.acquisition_camera_frame"
ACQUISITION_CAMERA_FRAME_SCHEMA_VERSION = 2
ACQUISITION_CAMERA_FRAME_ATTR = "acquisition_camera_frame"
ACQUISITION_CAMERA_FRAME_DIGEST_ATTR = "acquisition_camera_frame_sha256"

ACQUISITION_IMPORT_OWNERSHIP_SCHEMA_ID = "palette.acquisition_import_ownership"
ACQUISITION_IMPORT_OWNERSHIP_SCHEMA_VERSION = 1
ACQUISITION_IMPORT_OWNERSHIP_ATTR = "acquisition_import_ownership"
ACQUISITION_IMPORT_OWNERSHIP_DIGEST_ATTR = "acquisition_import_ownership_sha256"
ACQUISITION_IMPORT_PRODUCER = "palette.acquisition_import_writer.v1"
ACQUISITION_MATERIALIZATION_MANIFEST_SCHEMA_ID = (
    "palette.acquisition_materialization_manifest"
)
ACQUISITION_MATERIALIZATION_MANIFEST_SCHEMA_VERSION = 2
ACQUISITION_MATERIALIZATION_MANIFEST_ATTR = "acquisition_materialization_manifest"
ACQUISITION_MATERIALIZATION_MANIFEST_DIGEST_ATTR = (
    "acquisition_materialization_manifest_sha256"
)
ACQUISITION_PHYSICAL_CHUNK_MANIFEST_SCHEMA_ID = (
    "palette.acquisition_physical_chunk_manifest"
)
ACQUISITION_PHYSICAL_CHUNK_MANIFEST_SCHEMA_VERSION = 2
ACQUISITION_PHYSICAL_CHUNK_MANIFEST_ATTR = "acquisition_physical_chunk_manifest"
ACQUISITION_PHYSICAL_CHUNK_MANIFEST_DIGEST_ATTR = (
    "acquisition_physical_chunk_manifest_sha256"
)
ACQUISITION_MATERIALIZATION_MANIFEST_PATH = (
    "raw_video/manifests/images_full_materialization"
)
ACQUISITION_MATERIALIZATION_WRITE_POLICY = "write_once_immutable_v1"
ACQUISITION_CHUNK_MANIFEST_SCOPE = (
    "complete_physical_storage_object_grid_encoded_payloads_v2"
)
ACQUISITION_CHUNK_CONTENT_EVIDENCE_SCOPE = (
    "importer_hashed_encoded_physical_storage_objects_at_completion_v2"
)
ACQUISITION_CHUNK_ENTRY_CANONICALIZATION = (
    "canonical_json_ordered_physical_storage_entries_v2"
)
ACQUISITION_METADATA_ONLY_VERIFICATION_SCOPE = (
    "manifest_and_storage_metadata_only_no_live_payload_rehash_v1"
)

CROP_PLACEMENT_OWNERSHIP_SCHEMA_ID = "palette.crop_placement_ownership"
CROP_PLACEMENT_OWNERSHIP_SCHEMA_VERSION = 1
CROP_PLACEMENT_OWNERSHIP_ATTR = "crop_placement_ownership"
CROP_PLACEMENT_OWNERSHIP_DIGEST_ATTR = "crop_placement_ownership_sha256"
CROP_PLACEMENT_PIXEL_CENTER_OWNERSHIP_ATTR = "crop_placement_ownership_pixel_center"
CROP_PLACEMENT_PIXEL_EDGE_OWNERSHIP_ATTR = (
    "crop_placement_ownership_pixel_edge_half_open"
)
# Historical coordinate-only successors may retain the producer's requested
# zero-padded windows.  These names are deliberately separate from the
# ordinary crop-writer ownership attrs: the latter continue to mean contained
# source-camera windows and must continue to fail closed for negative or
# out-of-frame origins.
CROP_PLACEMENT_PADDED_OWNERSHIP_ATTR = (
    "coordinate_successor_padded_crop_placement_ownership"
)
CROP_PLACEMENT_PADDED_PIXEL_CENTER_OWNERSHIP_ATTR = (
    "coordinate_successor_padded_crop_placement_ownership_pixel_center"
)
CROP_PLACEMENT_PADDED_PIXEL_EDGE_OWNERSHIP_ATTR = (
    "coordinate_successor_padded_crop_placement_ownership_pixel_edge_half_open"
)
CROP_PLACEMENT_PADDED_OWNERSHIP_SCHEMA_ID = (
    "palette.coordinate_successor.padded_crop_placement_ownership"
)
CROP_PLACEMENT_PADDED_OWNERSHIP_SCHEMA_VERSION = 1
CROP_PLACEMENT_PADDED_WINDOW_GEOMETRY_SCHEMA_ID = (
    "palette.coordinate_successor.padded_crop_window_geometry"
)
CROP_PLACEMENT_PADDED_PROVENANCE_SCHEMA_ID = (
    "palette.coordinate_successor.padded_crop_placement_provenance"
)
CROP_PLACEMENT_PADDED_PROVENANCE_SCHEMA_VERSION = 1
CROP_PLACEMENT_PADDED_PROVENANCE_ATTR = (
    "coordinate_successor_padded_crop_placement_provenance"
)
CROP_PLACEMENT_PADDED_PROVENANCE_DIGEST_ATTR = (
    f"{CROP_PLACEMENT_PADDED_PROVENANCE_ATTR}_sha256"
)
CROP_PLACEMENT_OWNERSHIP_ATTRS = frozenset(
    {
        CROP_PLACEMENT_OWNERSHIP_ATTR,
        CROP_PLACEMENT_PIXEL_CENTER_OWNERSHIP_ATTR,
        CROP_PLACEMENT_PIXEL_EDGE_OWNERSHIP_ATTR,
        CROP_PLACEMENT_PADDED_OWNERSHIP_ATTR,
        CROP_PLACEMENT_PADDED_PIXEL_CENTER_OWNERSHIP_ATTR,
        CROP_PLACEMENT_PADDED_PIXEL_EDGE_OWNERSHIP_ATTR,
    }
)
CROP_PLACEMENT_PRODUCER = "palette.crop_writer.v1"
CROP_PLACEMENT_WINDOW_POLICY = "contained_actual_source_window_v1"
CROP_PLACEMENT_PADDED_PRODUCER = "palette.sealed_geometry_crop_profile.v1"
CROP_PLACEMENT_PADDED_LEGACY_PRODUCER = (
    "palette.coordinate_successor.historical_geometry_only_crop_adapter.v1"
)
CROP_PLACEMENT_PADDED_WINDOW_POLICY = "requested_window_zero_padded_v1"

SOURCE_CAMERA_FRAME_KIND = "acquisition_source_camera"
SELECTED_CANVAS_FRAME_KIND = "selected_stimulus_canvas"
ARENA_RELATIVE_CANVAS_FRAME_KIND = "arena_relative_canvas"
ROI_FRAME_KIND = "crop_roi"
MODEL_INPUT_FRAME_KIND = "detector_model_input"
SOURCE_CAMERA_NORMALIZED_FRAME_KIND = "source_camera_normalized"
DETECTOR_NORMALIZED_FRAME_KIND = "detector_normalized"
PIXEL_FRAME_KINDS = frozenset(
    {
        SOURCE_CAMERA_FRAME_KIND,
        SELECTED_CANVAS_FRAME_KIND,
        ARENA_RELATIVE_CANVAS_FRAME_KIND,
        ROI_FRAME_KIND,
        MODEL_INPUT_FRAME_KIND,
        SOURCE_CAMERA_NORMALIZED_FRAME_KIND,
        DETECTOR_NORMALIZED_FRAME_KIND,
    }
)

SOURCE_CAMERA_IMAGE_SPACE_ID = "source_camera_image_px"
STIMULUS_CANVAS_SPACE_ID = "stimulus_canvas_px"
ARENA_RELATIVE_CANVAS_SPACE_ID = "arena_relative_canvas_px"
ROI_LOCAL_SPACE_ID = "roi_local_px"
DETECTOR_MODEL_INPUT_SPACE_ID = "detector_model_input_px"
SOURCE_CAMERA_NORMALIZED_SPACE_ID = "source_camera_normalized_xy"
DETECTOR_NORMALIZED_SPACE_ID = "detector_normalized_xy"

PROJECTIVE_XY_DIRECT_V1 = "projective_xy_direct_v1"
TRANSLATION_XY_DIRECT_V1 = "translation_xy_direct_v1"
SCALE_XY_EDGE_ALIGNED_V1 = "target=offset+source*target_extent/source_extent"
SCALE_XY_PIXEL_CENTER_V1 = "target=offset+(source+0.5)*target_extent/source_extent-0.5"
NORMALIZED_TO_PIXEL_EDGE_EXTENT_V1 = "target_px=source_normalized*reference_extent_px"
NORMALIZED_TO_PIXEL_CENTER_INDEX_V1 = (
    "target_px=source_normalized*(reference_extent_px-1)"
)

_FRAME_ID_RE = re.compile(r"^[A-Za-z0-9_.:+-]+$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_BOUND_PIXEL_FRAME_SEAL = object()
_BOUND_ACQUISITION_FRAME_SEAL = object()
_BOUND_ACQUISITION_OWNERSHIP_SEAL = object()
_BOUND_CROP_PLACEMENT_OWNERSHIP_SEAL = object()


def _bound_authority_proof_key(value: Any) -> tuple[Any, ...]:
    """Identify one exact persisted authority without reopening its store."""

    identity = value.archive_identity
    return (
        "palette.pixel_frame_authority_proof.v1",
        type(value).__qualname__,
        identity.kind,
        identity.key,
        value.record_ref,
        value.record_sha256,
    )


_VERIFIED_ACQUISITION_MATERIALIZATION_SEAL = object()
_ACQUISITION_PHYSICAL_OBJECT_EVIDENCE_SEAL = object()
_SUPPORTED_PIXEL_CONVENTIONS = PIXEL_CONVENTIONS - {"not_applicable"}


class PixelFrameAuthorityError(ValueError):
    """Raised when a pixel-frame authority is forged, stale, or incoherent."""


def require_trusted_coordinate_attrs(node: Any, *, label: str) -> Any:
    """Return the exact supported attrs transaction type or fail pre-write.

    Capability-shaped mappings and ``dict`` subclasses are not a safe
    scientific transaction boundary: they can partially mutate, coerce nested
    values, or sabotage rollback.  Canonical coordinate/transform publication
    accepts only the exact built-in mapping used by deterministic fakes or the
    exact Zarr Attributes implementation used by persisted archives.
    """

    attrs = getattr(node, "attrs", None)
    trusted = type(attrs) is dict
    if not trusted:
        try:
            from zarr.core.attributes import Attributes as ZarrAttributes
        except ImportError:  # pragma: no cover - Palette depends on zarr
            ZarrAttributes = None  # type: ignore[assignment,misc]
        trusted = ZarrAttributes is not None and type(attrs) is ZarrAttributes
    if not trusted:
        raise PixelFrameAuthorityError(
            f"{label} attrs type is not an exact trusted dict or Zarr Attributes implementation; no write was attempted."
        )
    for operation in ("update", "__setitem__", "__delitem__"):
        if not callable(getattr(attrs, operation, None)):
            raise PixelFrameAuthorityError(
                f"{label} attrs lacks required {operation} mutation support; no write was attempted."
            )
    return attrs


@dataclass(frozen=True, init=False)
class VerifiedAcquisitionMaterialization:
    """Sealed importer attestation plus metadata-only manifest validation.

    The normal loader validates persisted storage metadata, frame-map content,
    and the import-time per-chunk evidence record.  It does not re-read encoded
    image chunks and therefore is not proof against post-import byte mutation
    or a perfect copy of both payload and all bound metadata.
    """

    operation: Mapping[str, Any]
    _seal: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        operation: Mapping[str, Any],
        _verification_seal: object | None = None,
    ) -> None:
        if _verification_seal is not _VERIFIED_ACQUISITION_MATERIALIZATION_SEAL:
            raise PixelFrameAuthorityError(
                "Verified acquisition materialization cannot be constructed directly."
            )
        object.__setattr__(self, "operation", copy.deepcopy(dict(operation)))
        object.__setattr__(self, "_seal", _verification_seal)


@dataclass(frozen=True, init=False)
class AcquisitionPhysicalObjectEvidence:
    """Sealed hashes of the actual encoded objects behind ``images_full``.

    Evidence is minted only by
    :func:`collect_acquisition_importer_physical_object_evidence`, which reads
    every expected outer chunk-grid object from the array's own local store.
    For sharded Zarr v3 arrays those objects are complete encoded shards, not
    the logical chunks exposed by ``Array.chunks``.
    """

    array_path: str
    array_storage_sha256: str
    entries: tuple[Mapping[str, Any], ...]
    _frame_node: Any = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        array_path: str,
        array_storage_sha256: str,
        entries: list[dict[str, Any]],
        frame_node: Any,
        _verification_seal: object | None = None,
    ) -> None:
        if _verification_seal is not _ACQUISITION_PHYSICAL_OBJECT_EVIDENCE_SEAL:
            raise PixelFrameAuthorityError(
                "Acquisition physical-object evidence cannot be constructed directly."
            )
        object.__setattr__(self, "array_path", array_path)
        object.__setattr__(self, "array_storage_sha256", array_storage_sha256)
        object.__setattr__(
            self,
            "entries",
            tuple(copy.deepcopy(dict(entry)) for entry in entries),
        )
        object.__setattr__(self, "_frame_node", frame_node)
        object.__setattr__(self, "_seal", _verification_seal)

    def assert_verified(self, frame_node: Any) -> list[dict[str, Any]]:
        """Rehash the live encoded objects and return canonical entries."""

        if (
            self._seal is not _ACQUISITION_PHYSICAL_OBJECT_EVIDENCE_SEAL
            or frame_node is not self._frame_node
        ):
            raise PixelFrameAuthorityError(
                "Acquisition physical-object evidence is not sealed for the exact live frame array."
            )
        storage = _array_storage_identity(frame_node)
        if (
            self.array_path != canonical_node_path(frame_node)
            or self.array_storage_sha256 != storage["record_sha256"]
        ):
            raise PixelFrameAuthorityError(
                "Acquisition physical-object evidence no longer matches live array storage metadata."
            )
        current = _hash_live_encoded_physical_objects(frame_node)
        persisted = [copy.deepcopy(dict(entry)) for entry in self.entries]
        if not _exact_json_equal(current, persisted):
            raise PixelFrameAuthorityError(
                "Encoded acquisition storage objects changed after importer evidence was collected."
            )
        return current


@dataclass(frozen=True)
class AcquisitionImportOwnershipRecord:
    """Exact source ownership minted by the acquisition import writer."""

    recording_id: str
    camera_id: str
    producer: str
    mode: str
    source_video_metadata_sha256: str
    frame_array: Mapping[str, Any] | None
    frame_index: Mapping[str, Any] | None
    import_operation: Mapping[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": ACQUISITION_IMPORT_OWNERSHIP_SCHEMA_ID,
            "schema_version": ACQUISITION_IMPORT_OWNERSHIP_SCHEMA_VERSION,
            "recording_id": self.recording_id,
            "camera_id": self.camera_id,
            "producer": self.producer,
            "mode": self.mode,
            "source_video_metadata_sha256": self.source_video_metadata_sha256,
            "frame_array": (
                copy.deepcopy(dict(self.frame_array))
                if self.frame_array is not None
                else None
            ),
            "frame_index": (
                copy.deepcopy(dict(self.frame_index))
                if self.frame_index is not None
                else None
            ),
            "import_operation": (
                copy.deepcopy(dict(self.import_operation))
                if self.import_operation is not None
                else None
            ),
            "canonicalization": PIXEL_FRAME_AUTHORITY_CANONICALIZATION,
        }

    def digest(self) -> str:
        return _mapping_sha256(self.to_dict())


@dataclass(frozen=True, init=False)
class BoundAcquisitionImportOwnership:
    record: AcquisitionImportOwnershipRecord
    record_ref: str
    record_sha256: str
    _archive_identity: ArchiveIdentity = field(repr=False, compare=False)
    _root_node: Any = field(repr=False, compare=False)
    _authority_node: Any = field(repr=False, compare=False)
    _frame_node: Any | None = field(repr=False, compare=False)
    _frame_index_node: Any | None = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        record: AcquisitionImportOwnershipRecord,
        archive: ArchiveIdentity,
        root_node: Any,
        authority_node: Any,
        frame_node: Any | None,
        frame_index_node: Any | None,
        _verification_seal: object | None = None,
    ) -> None:
        if _verification_seal is not _BOUND_ACQUISITION_OWNERSHIP_SEAL:
            raise PixelFrameAuthorityError(
                "Acquisition import ownership cannot be constructed directly."
            )
        object.__setattr__(self, "record", record)
        object.__setattr__(
            self,
            "record_ref",
            f"/{canonical_node_path(authority_node)}@{ACQUISITION_IMPORT_OWNERSHIP_ATTR}",
        )
        object.__setattr__(self, "record_sha256", record.digest())
        object.__setattr__(self, "_archive_identity", archive)
        object.__setattr__(self, "_root_node", root_node)
        object.__setattr__(self, "_authority_node", authority_node)
        object.__setattr__(self, "_frame_node", frame_node)
        object.__setattr__(self, "_frame_index_node", frame_index_node)
        object.__setattr__(self, "_seal", _verification_seal)

    @property
    def archive_identity(self) -> ArchiveIdentity:
        return self._archive_identity

    def assert_verified(self) -> None:
        if self._seal is not _BOUND_ACQUISITION_OWNERSHIP_SEAL:
            raise PixelFrameAuthorityError(
                "Acquisition import ownership is not sealed evidence."
            )
        current = load_acquisition_import_ownership(
            self._root_node,
            self._authority_node,
            frame_node=self._frame_node,
            frame_index_node=self._frame_index_node,
        )
        if (
            current.record != self.record
            or current.record_ref != self.record_ref
            or current.record_sha256 != self.record_sha256
            or current.archive_identity != self.archive_identity
        ):
            raise PixelFrameAuthorityError(
                "Persisted acquisition import ownership changed after binding."
            )


@dataclass(frozen=True)
class AcquisitionCameraFrameRecord:
    recording_id: str
    camera_id: str
    source_video_metadata: Mapping[str, Any]
    source_video_metadata_sha256: str
    width_px: int
    height_px: int
    source_total_frames: int
    frame_domain: Mapping[str, Any]
    frame_array: Mapping[str, Any] | None
    frame_index: Mapping[str, Any] | None
    frame_count: int
    import_ownership: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": ACQUISITION_CAMERA_FRAME_SCHEMA_ID,
            "schema_version": ACQUISITION_CAMERA_FRAME_SCHEMA_VERSION,
            "recording_id": self.recording_id,
            "camera_id": self.camera_id,
            "source_video_metadata": copy.deepcopy(dict(self.source_video_metadata)),
            "source_video_metadata_sha256": self.source_video_metadata_sha256,
            "width_px": self.width_px,
            "height_px": self.height_px,
            "source_total_frames": self.source_total_frames,
            "frame_domain": copy.deepcopy(dict(self.frame_domain)),
            "frame_array": (
                copy.deepcopy(dict(self.frame_array))
                if self.frame_array is not None
                else None
            ),
            "frame_index": (
                copy.deepcopy(dict(self.frame_index))
                if self.frame_index is not None
                else None
            ),
            "frame_count": self.frame_count,
            "import_ownership": copy.deepcopy(dict(self.import_ownership)),
            "canonicalization": PIXEL_FRAME_AUTHORITY_CANONICALIZATION,
        }

    def digest(self) -> str:
        return _mapping_sha256(self.to_dict())


@dataclass(frozen=True, init=False)
class BoundAcquisitionCameraFrame:
    record: AcquisitionCameraFrameRecord
    record_ref: str
    record_sha256: str
    selector: str
    width: int
    height: int
    units: str
    _archive_identity: ArchiveIdentity = field(repr=False, compare=False)
    _root_node: Any = field(repr=False, compare=False)
    _authority_node: Any = field(repr=False, compare=False)
    import_ownership: BoundAcquisitionImportOwnership = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        record: AcquisitionCameraFrameRecord,
        archive: ArchiveIdentity,
        root_node: Any,
        authority_node: Any,
        import_ownership: BoundAcquisitionImportOwnership,
        _verification_seal: object | None = None,
    ) -> None:
        if _verification_seal is not _BOUND_ACQUISITION_FRAME_SEAL:
            raise PixelFrameAuthorityError(
                "Bound acquisition camera frames cannot be constructed directly."
            )
        object.__setattr__(self, "record", record)
        object.__setattr__(
            self,
            "record_ref",
            f"/{canonical_node_path(authority_node)}@{ACQUISITION_CAMERA_FRAME_ATTR}",
        )
        object.__setattr__(self, "record_sha256", record.digest())
        object.__setattr__(self, "selector", ACQUISITION_CAMERA_FRAME_ATTR)
        object.__setattr__(self, "width", record.width_px)
        object.__setattr__(self, "height", record.height_px)
        object.__setattr__(self, "units", "px")
        object.__setattr__(self, "_archive_identity", archive)
        object.__setattr__(self, "_root_node", root_node)
        object.__setattr__(self, "_authority_node", authority_node)
        object.__setattr__(self, "import_ownership", import_ownership)
        object.__setattr__(self, "_seal", _verification_seal)

    @property
    def archive_identity(self) -> ArchiveIdentity:
        return self._archive_identity

    def assert_verified(self) -> None:
        if self._seal is not _BOUND_ACQUISITION_FRAME_SEAL:
            raise PixelFrameAuthorityError(
                "Acquisition camera frame is not sealed evidence."
            )
        current = load_acquisition_camera_frame(
            self._root_node,
            self._authority_node,
            import_ownership=self.import_ownership,
        )
        if (
            current.record != self.record
            or current.record_ref != self.record_ref
            or current.record_sha256 != self.record_sha256
            or current.archive_identity != self.archive_identity
        ):
            raise PixelFrameAuthorityError(
                "Persisted acquisition camera frame changed after binding."
            )


def _required_text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise PixelFrameAuthorityError(
            f"{field_name} must be a non-empty string without surrounding whitespace."
        )
    return value


def _sha256(value: Any, *, field_name: str) -> str:
    text = _required_text(value, field_name=field_name)
    if _SHA256_RE.fullmatch(text) is None:
        raise PixelFrameAuthorityError(
            f"{field_name} must be a lowercase 64-character SHA-256 digest."
        )
    return text


def _canonical_json(value: Any) -> str:
    try:
        canonical = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        canonical.encode("utf-8")
        return canonical
    except (TypeError, ValueError, UnicodeError) as exc:
        raise PixelFrameAuthorityError(
            "Pixel-frame authority records must contain finite canonical JSON values "
            "with valid Unicode scalar strings."
        ) from exc


def _mapping_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _exact_json_equal(left: Any, right: Any) -> bool:
    """Type-strict JSON equality (so bool/int, int/float, list/tuple differ)."""

    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return (
            type(left) is type(right)
            and set(left) == set(right)
            and all(_exact_json_equal(left[name], right[name]) for name in left)
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _exact_json_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return type(left) is type(right) and left == right


def _expected_attrs_after_update(
    snapshot: Mapping[str, Any],
    intended: Mapping[str, Any],
) -> dict[str, Any]:
    expected = copy.deepcopy(dict(snapshot))
    expected.update(copy.deepcopy(dict(intended)))
    return expected


def _require_exact_attrs_state(
    attrs: Any,
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    current = dict(attrs)
    if not _exact_json_equal(current, dict(expected)):
        raise PixelFrameAuthorityError(
            f"{label} attrs differ from the exact snapshot plus intended payload."
        )


def _restore_exact_attrs(attrs: Any, snapshot: Mapping[str, Any]) -> None:
    for name in tuple(attrs.keys()):
        del attrs[name]
    attrs.update(copy.deepcopy(dict(snapshot)))
    if not _exact_json_equal(dict(attrs), dict(snapshot)):
        raise RuntimeError("restored attrs differ type-strictly from snapshot")


def _exact_fields(
    value: Any,
    *,
    expected: frozenset[str],
    field_name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PixelFrameAuthorityError(f"{field_name} must be a mapping.")
    actual = frozenset(value)
    if actual != expected:
        raise PixelFrameAuthorityError(
            f"{field_name} fields are invalid; missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}."
        )
    return value


def _array_values(node: Any) -> np.ndarray:
    try:
        values = np.asarray(node[:])
    except Exception as exc:
        raise PixelFrameAuthorityError(
            "Unable to read exact persisted pixel-frame lineage values."
        ) from exc
    if values.dtype.hasobject:
        raise PixelFrameAuthorityError(
            "Pixel-frame lineage arrays cannot use object dtype."
        )
    return np.ascontiguousarray(values)


def _array_has_exact_dtype(node: Any, expected: str) -> bool:
    try:
        return np.dtype(getattr(node, "dtype")) == np.dtype(expected)
    except (AttributeError, TypeError, ValueError):
        return False


def array_values_sha256(node: Any) -> str:
    values = _array_values(node)
    header = {
        "canonicalization": ARRAY_VALUES_CANONICALIZATION,
        "dtype": np.lib.format.dtype_to_descr(values.dtype),
        "shape": [int(item) for item in values.shape],
    }
    digest = hashlib.sha256()
    digest.update(_canonical_json(header).encode("utf-8"))
    digest.update(b"\x00")
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _array_pointer(node: Any) -> dict[str, str]:
    return {
        "record_ref": f"/{canonical_node_path(node)}@array_values",
        "record_sha256": array_values_sha256(node),
        "selector": "array_values",
    }


def _exact_positive_shape(
    value: Any,
    *,
    dimensions: int,
    field_name: str,
) -> tuple[int, ...]:
    if (
        not isinstance(value, (tuple, list))
        or len(value) != dimensions
        or any(type(item) is not int or item <= 0 for item in value)
    ):
        raise PixelFrameAuthorityError(
            f"{field_name} must contain one exact positive integer per array dimension."
        )
    return tuple(value)


def _metadata_chunk_shapes(
    metadata: Mapping[str, Any],
    *,
    dimensions: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return ``(logical, physical)`` shapes from exact Zarr v3 metadata."""

    chunk_grid = metadata.get("chunk_grid")
    if not isinstance(chunk_grid, Mapping) or chunk_grid.get("name") != "regular":
        raise PixelFrameAuthorityError(
            "Array storage identity requires a regular Zarr v3 chunk grid."
        )
    configuration = chunk_grid.get("configuration")
    if not isinstance(configuration, Mapping):
        raise PixelFrameAuthorityError(
            "Array storage identity requires exact chunk-grid configuration."
        )
    physical = _exact_positive_shape(
        configuration.get("chunk_shape"),
        dimensions=dimensions,
        field_name="Zarr physical chunk-grid shape",
    )
    codecs = metadata.get("codecs")
    if type(codecs) is not list:
        raise PixelFrameAuthorityError(
            "Array storage identity requires an exact Zarr v3 codecs list."
        )
    sharding_codecs = [
        codec
        for codec in codecs
        if isinstance(codec, Mapping) and codec.get("name") == "sharding_indexed"
    ]
    if not sharding_codecs:
        return physical, physical
    if len(sharding_codecs) != 1:
        raise PixelFrameAuthorityError(
            "Array storage identity supports exactly one outer sharding_indexed codec."
        )
    shard_configuration = sharding_codecs[0].get("configuration")
    if not isinstance(shard_configuration, Mapping):
        raise PixelFrameAuthorityError(
            "Sharded array storage identity requires exact codec configuration."
        )
    logical = _exact_positive_shape(
        shard_configuration.get("chunk_shape"),
        dimensions=dimensions,
        field_name="Zarr logical chunk shape",
    )
    if any(
        outer < inner or outer % inner != 0
        for inner, outer in zip(logical, physical, strict=True)
    ):
        raise PixelFrameAuthorityError(
            "Zarr physical shard shape must be an integer multiple of its logical chunk shape."
        )
    return logical, physical


def _array_storage_identity(node: Any) -> dict[str, Any]:
    """Bind exact persisted array metadata without reading the array payload.

    Zarr attributes are deliberately excluded because the materialization
    records themselves live in attrs.  Real Zarr arrays provide a metadata
    object with ``to_dict``; deterministic test doubles must provide the exact
    JSON mapping as ``_coordinate_storage_metadata``.  Shape/dtype/chunks are
    repeated outside that mapping so a malformed metadata implementation cannot
    hide a disagreement with the live array interface.
    """

    shape_raw = getattr(node, "shape", None)
    dtype_raw = getattr(node, "dtype", None)
    if not isinstance(shape_raw, (tuple, list)) or not shape_raw:
        raise PixelFrameAuthorityError("Array storage identity requires exact shape.")
    if any(type(item) is not int or item <= 0 for item in shape_raw):
        raise PixelFrameAuthorityError("Array shape must use exact positive integers.")
    dimensions = len(shape_raw)
    chunks_raw = _exact_positive_shape(
        getattr(node, "chunks", None),
        dimensions=dimensions,
        field_name="Array logical chunks",
    )
    try:
        shards_raw = getattr(node, "shards", None)
    except (AttributeError, NotImplementedError):
        shards_raw = None
    physical_raw = (
        chunks_raw
        if shards_raw is None
        else _exact_positive_shape(
            shards_raw,
            dimensions=dimensions,
            field_name="Array physical shards",
        )
    )
    try:
        dtype = np.dtype(dtype_raw)
    except TypeError as exc:
        raise PixelFrameAuthorityError("Array storage dtype is invalid.") from exc
    metadata_object = getattr(node, "metadata", None)
    if metadata_object is not None and callable(
        getattr(metadata_object, "to_dict", None)
    ):
        metadata_raw = metadata_object.to_dict()
    else:
        metadata_raw = getattr(node, "_coordinate_storage_metadata", None)
    if type(metadata_raw) is not dict:
        raise PixelFrameAuthorityError(
            "Array storage identity requires exact Zarr metadata without reading payload bytes."
        )
    metadata_without_attrs = copy.deepcopy(metadata_raw)
    metadata_without_attrs.pop("attributes", None)
    try:
        canonical_metadata = json.loads(_canonical_json(metadata_without_attrs))
    except PixelFrameAuthorityError as exc:
        raise PixelFrameAuthorityError(
            "Array Zarr metadata is not canonical finite JSON."
        ) from exc
    if (
        type(canonical_metadata.get("zarr_format")) is not int
        or canonical_metadata.get("zarr_format") != 3
        or canonical_metadata.get("node_type") != "array"
    ):
        raise PixelFrameAuthorityError(
            "Array storage identity requires exact Zarr v3 array metadata."
        )
    metadata_shape = _exact_positive_shape(
        canonical_metadata.get("shape"),
        dimensions=dimensions,
        field_name="Zarr metadata shape",
    )
    if metadata_shape != tuple(shape_raw):
        raise PixelFrameAuthorityError(
            "Live array shape disagrees with exact Zarr storage metadata."
        )
    try:
        metadata_dtype = np.dtype(canonical_metadata.get("data_type"))
    except (TypeError, ValueError) as exc:
        raise PixelFrameAuthorityError(
            "Array Zarr metadata data_type is invalid."
        ) from exc
    if metadata_dtype != dtype:
        raise PixelFrameAuthorityError(
            "Live array dtype disagrees with exact Zarr storage metadata."
        )
    metadata_logical, metadata_physical = _metadata_chunk_shapes(
        canonical_metadata,
        dimensions=dimensions,
    )
    if metadata_logical != chunks_raw or metadata_physical != physical_raw:
        raise PixelFrameAuthorityError(
            "Live logical/physical chunk shapes disagree with exact Zarr storage metadata."
        )
    record = {
        "record_ref": f"/{canonical_node_path(node)}@array_storage_metadata",
        "selector": "array_storage_metadata",
        "shape": [int(item) for item in shape_raw],
        "dtype": np.lib.format.dtype_to_descr(dtype),
        "logical_chunk_shape": [int(item) for item in chunks_raw],
        "physical_chunk_shape": [int(item) for item in physical_raw],
        "zarr_metadata_without_attrs": canonical_metadata,
    }
    return {**record, "record_sha256": _mapping_sha256(record)}


def _same_resolved_persisted_node(resolved: Any, supplied: Any) -> bool:
    """Compare logical Zarr nodes without trusting detached path lookalikes.

    Real Zarr indexing returns a fresh Python wrapper for the same persisted
    node, so object identity is too strict.  Exact test doubles intentionally
    retain the stronger ``is`` rule.  For real Zarr wrappers we require the
    exact public node type, canonical path, stable archive/store identity, and
    (for arrays) identical live storage metadata.
    """

    if resolved is supplied:
        return True
    try:
        import zarr
    except ImportError:  # pragma: no cover - Palette depends on zarr
        return False
    if type(resolved) is not type(supplied) or type(resolved) not in {
        zarr.Array,
        zarr.Group,
    }:
        return False
    try:
        if canonical_node_path(resolved) != canonical_node_path(
            supplied
        ) or archive_identity(resolved) != archive_identity(supplied):
            return False
        if type(resolved) is zarr.Array:
            return _exact_json_equal(
                _array_storage_identity(resolved),
                _array_storage_identity(supplied),
            )
    except (ArchiveIdentityError, CoordinateReferenceError, PixelFrameAuthorityError):
        return False
    return True


def _expected_physical_chunk_indices(node: Any) -> list[list[int]]:
    storage = _array_storage_identity(node)
    ranges = [
        range((size + chunk - 1) // chunk)
        for size, chunk in zip(
            storage["shape"],
            storage["physical_chunk_shape"],
            strict=True,
        )
    ]
    return [list(index) for index in itertools.product(*ranges)]


def _expected_physical_chunk_storage_key(
    storage: Mapping[str, Any],
    chunk_indices: list[int],
) -> str:
    metadata = storage.get("zarr_metadata_without_attrs")
    if not isinstance(metadata, Mapping):
        raise PixelFrameAuthorityError(
            "Physical chunk keys require exact Zarr chunk-key metadata."
        )
    encoding = metadata.get("chunk_key_encoding")
    if not isinstance(encoding, Mapping):
        raise PixelFrameAuthorityError(
            "Physical chunk keys require a declared Zarr chunk_key_encoding."
        )
    name = encoding.get("name")
    configuration = encoding.get("configuration")
    if not isinstance(configuration, Mapping):
        raise PixelFrameAuthorityError(
            "Zarr chunk_key_encoding configuration is missing."
        )
    separator = configuration.get("separator")
    if separator not in {"/", "."}:
        raise PixelFrameAuthorityError(
            "Zarr chunk_key_encoding separator is unsupported."
        )
    suffix = separator.join(str(item) for item in chunk_indices)
    if name == "default":
        return f"c{separator}{suffix}"
    if name == "v2":
        return suffix
    raise PixelFrameAuthorityError(
        "Zarr chunk_key_encoding name is unsupported for physical evidence."
    )


def _local_encoded_storage_object_locator(
    frame_node: Any,
    storage_key: str,
) -> tuple[str, tuple[str, ...]]:
    """Return a local root and canonical relative segments for fd-safe traversal."""

    store_path = getattr(frame_node, "store_path", None)
    store = getattr(store_path, "store", None)
    root_raw = getattr(store, "root", None)
    store_node_path = getattr(store_path, "path", None)
    array_path = canonical_node_path(frame_node)
    if not isinstance(root_raw, (str, os.PathLike)) or store_node_path != array_path:
        raise PixelFrameAuthorityError(
            "Importer physical-object evidence requires the exact local Zarr store path owned by the frame array."
        )
    root_fspath = os.fspath(root_raw)
    if not isinstance(root_fspath, str):
        raise PixelFrameAuthorityError(
            "Importer physical-object evidence requires a text local-store root path."
        )
    root = os.path.abspath(os.path.expanduser(root_fspath))
    relative_parts = (*array_path.split("/"), *storage_key.split("/"))
    for part in relative_parts:
        if part in {"", ".", ".."} or "\\" in part or "\x00" in part:
            raise PixelFrameAuthorityError(
                "Importer physical-object storage key contains a noncanonical path segment."
            )
    return root, relative_parts


def _hash_encoded_storage_object(
    root: str,
    relative_parts: tuple[str, ...],
    *,
    storage_key: str,
) -> tuple[str, int]:
    """Hash a local object through retained no-follow directory descriptors.

    Every component is opened relative to the already-open parent directory.
    A concurrent pathname replacement therefore cannot redirect traversal, and
    the opened file's identity and size are checked again after the read.
    """

    if (
        not relative_parts
        or not hasattr(os, "O_NOFOLLOW")
        or not hasattr(os, "O_DIRECTORY")
    ):
        raise PixelFrameAuthorityError(
            "Importer physical-object evidence requires no-follow descriptor traversal."
        )
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    file_flags = os.O_RDONLY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        directory_flags |= os.O_CLOEXEC
        file_flags |= os.O_CLOEXEC
    opened_fds: list[int] = []
    digest = hashlib.sha256()
    size = 0
    try:
        current_fd = os.open(root, directory_flags)
        opened_fds.append(current_fd)
        if not stat.S_ISDIR(os.fstat(current_fd).st_mode):
            raise PixelFrameAuthorityError(
                "Importer physical-object evidence requires a non-symlink local store directory."
            )
        for part in relative_parts[:-1]:
            current_fd = os.open(part, directory_flags, dir_fd=current_fd)
            opened_fds.append(current_fd)
            if not stat.S_ISDIR(os.fstat(current_fd).st_mode):
                raise PixelFrameAuthorityError(
                    "Encoded acquisition storage-object parent is not a directory."
                )
        payload_fd = os.open(relative_parts[-1], file_flags, dir_fd=current_fd)
        opened_fds.append(payload_fd)
        before = os.fstat(payload_fd)
        if not stat.S_ISREG(before.st_mode):
            raise PixelFrameAuthorityError(
                "Encoded acquisition storage object is not a regular file."
            )
        if before.st_size <= 0:
            raise PixelFrameAuthorityError(
                "Encoded acquisition storage objects must contain positive bytes."
            )
        while True:
            block = os.read(payload_fd, 8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            size += len(block)
        after = os.fstat(payload_fd)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if before_identity != after_identity or size != before.st_size:
            raise PixelFrameAuthorityError(
                "Encoded acquisition storage object changed while it was hashed."
            )
    except OSError as exc:
        raise PixelFrameAuthorityError(
            f"Unable to hash expected encoded acquisition storage object: {storage_key}."
        ) from exc
    finally:
        for opened_fd in reversed(opened_fds):
            try:
                os.close(opened_fd)
            except OSError:
                pass
    return digest.hexdigest(), size


def _hash_live_encoded_physical_objects(frame_node: Any) -> list[dict[str, Any]]:
    storage = _array_storage_identity(frame_node)
    entries: list[dict[str, Any]] = []
    for indices in _expected_physical_chunk_indices(frame_node):
        storage_key = _expected_physical_chunk_storage_key(storage, indices)
        root, relative_parts = _local_encoded_storage_object_locator(
            frame_node, storage_key
        )
        payload_sha256, size_bytes = _hash_encoded_storage_object(
            root,
            relative_parts,
            storage_key=storage_key,
        )
        entries.append(
            {
                "chunk_indices": list(indices),
                "storage_key": storage_key,
                "encoded_payload_sha256": payload_sha256,
                "encoded_size_bytes": size_bytes,
            }
        )
    return _canonical_physical_chunk_entries(entries, frame_node=frame_node)


def collect_acquisition_importer_physical_object_evidence(
    frame_node: Any,
) -> AcquisitionPhysicalObjectEvidence:
    """Hash every encoded outer-grid object and return sealed importer evidence.

    This is intentionally a local-store completion operation.  Importers using
    another write store must close it and reopen the finished archive through a
    local Zarr store before publishing canonical acquisition authority.
    """

    storage = _array_storage_identity(frame_node)
    entries = _hash_live_encoded_physical_objects(frame_node)
    return AcquisitionPhysicalObjectEvidence(
        array_path=canonical_node_path(frame_node),
        array_storage_sha256=storage["record_sha256"],
        entries=entries,
        frame_node=frame_node,
        _verification_seal=_ACQUISITION_PHYSICAL_OBJECT_EVIDENCE_SEAL,
    )


def _canonical_physical_chunk_entries(
    value: Any,
    *,
    frame_node: Any,
) -> list[dict[str, Any]]:
    if type(value) is not list:
        raise PixelFrameAuthorityError(
            "physical_chunk_entries must be an exact list emitted by the importer."
        )
    expected_indices = _expected_physical_chunk_indices(frame_node)
    if len(value) != len(expected_indices):
        raise PixelFrameAuthorityError(
            "Physical chunk evidence must cover the complete array chunk grid."
        )
    canonical: list[dict[str, Any]] = []
    storage_keys: set[str] = set()
    storage = _array_storage_identity(frame_node)
    for position, (raw, expected_index) in enumerate(
        zip(value, expected_indices, strict=True)
    ):
        if type(raw) is not dict or set(raw) != {
            "chunk_indices",
            "storage_key",
            "encoded_payload_sha256",
            "encoded_size_bytes",
        }:
            raise PixelFrameAuthorityError(
                f"Physical chunk entry {position} has invalid exact fields."
            )
        indices = raw["chunk_indices"]
        if (
            type(indices) is not list
            or any(type(item) is not int or item < 0 for item in indices)
            or indices != expected_index
        ):
            raise PixelFrameAuthorityError(
                "Physical chunk entries must be ordered by the complete canonical chunk grid."
            )
        storage_key = raw["storage_key"]
        expected_storage_key = _expected_physical_chunk_storage_key(
            storage,
            expected_index,
        )
        if (
            type(storage_key) is not str
            or not storage_key
            or storage_key != storage_key.strip()
            or storage_key in storage_keys
            or storage_key != expected_storage_key
        ):
            raise PixelFrameAuthorityError(
                "Physical chunk storage keys must exactly match the declared Zarr chunk-key encoding and be unique."
            )
        storage_keys.add(storage_key)
        encoded_size = raw["encoded_size_bytes"]
        if type(encoded_size) is not int or encoded_size <= 0:
            raise PixelFrameAuthorityError(
                "Physical chunk encoded_size_bytes must be an exact positive integer."
            )
        canonical.append(
            {
                "chunk_indices": list(indices),
                "storage_key": storage_key,
                "encoded_payload_sha256": _sha256(
                    raw["encoded_payload_sha256"],
                    field_name=(
                        f"physical_chunk_entries[{position}].encoded_payload_sha256"
                    ),
                ),
                "encoded_size_bytes": encoded_size,
            }
        )
    if not _exact_json_equal(value, canonical):
        raise PixelFrameAuthorityError(
            "Physical chunk entries are not their exact canonical JSON form."
        )
    return canonical


def _canonical_import_decode(
    import_operation_attrs: Mapping[str, Any],
    *,
    source_path: str,
) -> dict[str, str]:
    if not isinstance(import_operation_attrs, Mapping):
        raise PixelFrameAuthorityError("import_operation_attrs must be a mapping.")
    import_method = _required_text(
        import_operation_attrs.get("import_method"), field_name="import_method"
    )
    import_stage = _required_text(
        import_operation_attrs.get("import_stage"), field_name="import_stage"
    )
    import_mode = _required_text(
        import_operation_attrs.get("import_mode"), field_name="import_mode"
    )
    if import_mode != "full" or import_stage not in {"complete", "full_resolution"}:
        raise PixelFrameAuthorityError(
            "Canonical materialized acquisition authority requires a completed full import."
        )
    if (
        _required_text(
            import_operation_attrs.get("source_path"), field_name="source_path"
        )
        != source_path
    ):
        raise PixelFrameAuthorityError(
            "Import operation source_path conflicts with source-video metadata."
        )
    return {
        "import_method": import_method,
        "import_stage": import_stage,
        "import_mode": import_mode,
        "decode_backend": _required_text(
            import_operation_attrs.get("decode_backend"), field_name="decode_backend"
        ),
        "source_decode_surface": _required_text(
            import_operation_attrs.get("source_decode_surface"),
            field_name="source_decode_surface",
        ),
    }


def _acquisition_manifest_records(
    root_node: Any,
    *,
    frame_node: Any,
    frame_index_node: Any,
    manifest_node: Any,
    import_operation_attrs: Mapping[str, Any],
    physical_object_evidence: AcquisitionPhysicalObjectEvidence,
) -> tuple[dict[str, Any], dict[str, Any]]:
    recording_id, camera_id, metadata = _source_video_metadata(root_node)
    resolved_frame, resolved_index = _resolve_materialized_acquisition_nodes(root_node)
    resolved_manifest = _resolve_acquisition_materialization_manifest_node(root_node)
    if not (
        _same_resolved_persisted_node(resolved_frame, frame_node)
        and _same_resolved_persisted_node(resolved_index, frame_index_node)
        and _same_resolved_persisted_node(resolved_manifest, manifest_node)
    ):
        raise PixelFrameAuthorityError(
            "Acquisition importer manifest requires the exact nodes resolved from the archive root."
        )
    try:
        require_same_archive(root_node, frame_node, frame_index_node, manifest_node)
    except ArchiveIdentityError as exc:
        raise PixelFrameAuthorityError(str(exc)) from exc
    decode = _canonical_import_decode(
        import_operation_attrs,
        source_path=_required_text(
            metadata.get("source_path"), field_name="source_path"
        ),
    )
    storage = _array_storage_identity(frame_node)
    frame_map = _array_pointer(frame_index_node)
    if type(physical_object_evidence) is not AcquisitionPhysicalObjectEvidence:
        raise PixelFrameAuthorityError(
            "Acquisition manifest publication requires sealed importer physical-object evidence."
        )
    entries = physical_object_evidence.assert_verified(frame_node)
    chunk_record = {
        "schema_id": ACQUISITION_PHYSICAL_CHUNK_MANIFEST_SCHEMA_ID,
        "schema_version": ACQUISITION_PHYSICAL_CHUNK_MANIFEST_SCHEMA_VERSION,
        "producer": ACQUISITION_IMPORT_PRODUCER,
        "array_ref": f"/{canonical_node_path(frame_node)}@array_values",
        "array_storage": storage,
        "scope": ACQUISITION_CHUNK_MANIFEST_SCOPE,
        "content_evidence_scope": ACQUISITION_CHUNK_CONTENT_EVIDENCE_SCOPE,
        "digest_algorithm": "sha256",
        "entries": entries,
        "entry_count": len(entries),
        "entries_sha256": hashlib.sha256(
            _canonical_json(entries).encode("utf-8")
        ).hexdigest(),
        "entries_canonicalization": ACQUISITION_CHUNK_ENTRY_CANONICALIZATION,
        "canonicalization": PIXEL_FRAME_AUTHORITY_CANONICALIZATION,
    }
    chunk_pointer = {
        "record_ref": (
            f"/{canonical_node_path(manifest_node)}@"
            f"{ACQUISITION_PHYSICAL_CHUNK_MANIFEST_ATTR}"
        ),
        "record_sha256": _mapping_sha256(chunk_record),
        "entry_count": len(entries),
        "scope": ACQUISITION_CHUNK_MANIFEST_SCOPE,
        "content_evidence_scope": ACQUISITION_CHUNK_CONTENT_EVIDENCE_SCOPE,
        "metadata_only_verification_scope": (
            ACQUISITION_METADATA_ONLY_VERIFICATION_SCOPE
        ),
    }
    identity_basis = {
        "recording_id": recording_id,
        "camera_id": camera_id,
        "source_video_metadata_sha256": _mapping_sha256(metadata),
        "decode": decode,
        "images_full_storage": storage,
        "frame_map": frame_map,
        "physical_chunk_manifest": chunk_pointer,
    }
    materialization_record = {
        "schema_id": ACQUISITION_MATERIALIZATION_MANIFEST_SCHEMA_ID,
        "schema_version": ACQUISITION_MATERIALIZATION_MANIFEST_SCHEMA_VERSION,
        "producer": ACQUISITION_IMPORT_PRODUCER,
        "recording_id": recording_id,
        "camera_id": camera_id,
        "materialization_id": _mapping_sha256(identity_basis),
        "write_policy": ACQUISITION_MATERIALIZATION_WRITE_POLICY,
        "completed": True,
        "source_video_metadata_sha256": _mapping_sha256(metadata),
        "decode": decode,
        "images_full_storage": storage,
        "frame_map": frame_map,
        "physical_chunk_manifest": chunk_pointer,
        "canonicalization": PIXEL_FRAME_AUTHORITY_CANONICALIZATION,
    }
    return chunk_record, materialization_record


def stamp_acquisition_import_writer_materialization_manifest(
    root_node: Any,
    *,
    frame_node: Any,
    frame_index_node: Any,
    manifest_node: Any,
    import_operation_attrs: Mapping[str, Any],
    physical_object_evidence: AcquisitionPhysicalObjectEvidence,
) -> dict[str, Any]:
    """Persist the importer's exact chunk evidence and materialization record.

    The aggregate digest is derived internally from the complete canonical
    entry list.  Callers cannot provide digests or sizes: the required sealed
    evidence was minted by hashing the array's actual encoded outer-grid
    objects, and this function rehashes them before publication.  The write is
    immutable and idempotent only for the exact same records.
    """

    chunk_record, materialization_record = _acquisition_manifest_records(
        root_node,
        frame_node=frame_node,
        frame_index_node=frame_index_node,
        manifest_node=manifest_node,
        import_operation_attrs=import_operation_attrs,
        physical_object_evidence=physical_object_evidence,
    )
    attrs = require_trusted_coordinate_attrs(
        manifest_node,
        label="Acquisition materialization manifest",
    )
    intended = {
        ACQUISITION_PHYSICAL_CHUNK_MANIFEST_ATTR: chunk_record,
        ACQUISITION_PHYSICAL_CHUNK_MANIFEST_DIGEST_ATTR: _mapping_sha256(chunk_record),
        ACQUISITION_MATERIALIZATION_MANIFEST_ATTR: materialization_record,
        ACQUISITION_MATERIALIZATION_MANIFEST_DIGEST_ATTR: _mapping_sha256(
            materialization_record
        ),
    }
    controlled = set(intended)
    occupied = controlled & set(attrs)
    if occupied:
        if occupied != controlled:
            raise PixelFrameAuthorityError(
                "Occupied acquisition materialization has an incomplete immutable record set; explicit migration is required."
            )
        current = {name: copy.deepcopy(attrs[name]) for name in intended}
        if not _exact_json_equal(current, intended):
            raise PixelFrameAuthorityError(
                "Acquisition materialization is write-once and differs from the requested importer evidence."
            )
        return _load_acquisition_materialization_manifest(
            root_node,
            frame_node,
            frame_index_node,
        )
    snapshot = copy.deepcopy(dict(attrs))
    expected = _expected_attrs_after_update(snapshot, intended)
    try:
        attrs.update(copy.deepcopy(intended))
        _require_exact_attrs_state(
            attrs,
            expected,
            label="Acquisition materialization post-write",
        )
        loaded = _load_acquisition_materialization_manifest(
            root_node,
            frame_node,
            frame_index_node,
        )
        _require_exact_attrs_state(
            require_trusted_coordinate_attrs(
                manifest_node,
                label="Reloaded acquisition materialization manifest",
            ),
            expected,
            label="Acquisition materialization post-reload",
        )
        return loaded
    except Exception as exc:
        try:
            _restore_exact_attrs(attrs, snapshot)
        except Exception as rollback_exc:  # pragma: no cover - storage failure
            raise PixelFrameAuthorityError(
                "Acquisition materialization stamp failed and rollback was incomplete: "
                f"{rollback_exc}"
            ) from exc
        if isinstance(exc, PixelFrameAuthorityError):
            raise
        raise PixelFrameAuthorityError(
            f"Acquisition materialization stamp failed: {exc}"
        ) from exc


def _load_acquisition_materialization_manifest(
    root_node: Any,
    frame_node: Any,
    frame_index_node: Any,
) -> dict[str, Any]:
    resolved_frame, resolved_index = _resolve_materialized_acquisition_nodes(root_node)
    manifest_node = _resolve_acquisition_materialization_manifest_node(root_node)
    if not (
        _same_resolved_persisted_node(resolved_frame, frame_node)
        and _same_resolved_persisted_node(resolved_index, frame_index_node)
    ):
        raise PixelFrameAuthorityError(
            "Acquisition materialization must use the exact live nodes resolved from the archive root."
        )
    attrs = getattr(manifest_node, "attrs", None)
    if not isinstance(attrs, Mapping):
        raise PixelFrameAuthorityError(
            "Canonical acquisition materialization manifest must expose attrs."
        )
    chunk_raw = attrs.get(ACQUISITION_PHYSICAL_CHUNK_MANIFEST_ATTR)
    if type(chunk_raw) is not dict:
        raise PixelFrameAuthorityError(
            "Persisted acquisition physical-chunk manifest record is missing."
        )
    chunk_stored_digest = _sha256(
        attrs.get(ACQUISITION_PHYSICAL_CHUNK_MANIFEST_DIGEST_ATTR),
        field_name=ACQUISITION_PHYSICAL_CHUNK_MANIFEST_DIGEST_ATTR,
    )
    if chunk_stored_digest != _mapping_sha256(chunk_raw):
        raise PixelFrameAuthorityError(
            "Acquisition physical-chunk manifest digest is stale."
        )
    chunk_payload = _exact_fields(
        chunk_raw,
        expected=frozenset(
            {
                "schema_id",
                "schema_version",
                "producer",
                "array_ref",
                "array_storage",
                "scope",
                "content_evidence_scope",
                "digest_algorithm",
                "entries",
                "entry_count",
                "entries_sha256",
                "entries_canonicalization",
                "canonicalization",
            }
        ),
        field_name="acquisition_physical_chunk_manifest",
    )
    storage = _array_storage_identity(frame_node)
    entries = _canonical_physical_chunk_entries(
        chunk_payload["entries"],
        frame_node=frame_node,
    )
    if (
        chunk_payload["schema_id"] != ACQUISITION_PHYSICAL_CHUNK_MANIFEST_SCHEMA_ID
        or type(chunk_payload["schema_version"]) is not int
        or chunk_payload["schema_version"]
        != ACQUISITION_PHYSICAL_CHUNK_MANIFEST_SCHEMA_VERSION
        or chunk_payload["producer"] != ACQUISITION_IMPORT_PRODUCER
        or chunk_payload["array_ref"]
        != f"/{canonical_node_path(frame_node)}@array_values"
        or not _exact_json_equal(chunk_payload["array_storage"], storage)
        or chunk_payload["scope"] != ACQUISITION_CHUNK_MANIFEST_SCOPE
        or chunk_payload["content_evidence_scope"]
        != ACQUISITION_CHUNK_CONTENT_EVIDENCE_SCOPE
        or chunk_payload["digest_algorithm"] != "sha256"
        or type(chunk_payload["entry_count"]) is not int
        or chunk_payload["entry_count"] != len(entries)
        or _sha256(
            chunk_payload["entries_sha256"],
            field_name="acquisition_physical_chunk_manifest.entries_sha256",
        )
        != hashlib.sha256(_canonical_json(entries).encode("utf-8")).hexdigest()
        or chunk_payload["entries_canonicalization"]
        != ACQUISITION_CHUNK_ENTRY_CANONICALIZATION
        or chunk_payload["canonicalization"] != PIXEL_FRAME_AUTHORITY_CANONICALIZATION
    ):
        raise PixelFrameAuthorityError(
            "Acquisition physical-chunk manifest does not exactly bind live storage metadata and importer content evidence."
        )
    canonical_chunk = json.loads(_canonical_json(chunk_payload))
    if not _exact_json_equal(chunk_raw, canonical_chunk):
        raise PixelFrameAuthorityError(
            "Raw acquisition physical-chunk manifest is not canonical."
        )
    if chunk_stored_digest != _mapping_sha256(canonical_chunk):
        raise PixelFrameAuthorityError(
            "Acquisition physical-chunk manifest digest is stale."
        )

    raw = attrs.get(ACQUISITION_MATERIALIZATION_MANIFEST_ATTR)
    if type(raw) is not dict:
        raise PixelFrameAuthorityError(
            "Persisted acquisition materialization manifest is missing."
        )
    stored_manifest_digest = _sha256(
        attrs.get(ACQUISITION_MATERIALIZATION_MANIFEST_DIGEST_ATTR),
        field_name=ACQUISITION_MATERIALIZATION_MANIFEST_DIGEST_ATTR,
    )
    if stored_manifest_digest != _mapping_sha256(raw):
        raise PixelFrameAuthorityError(
            "Acquisition materialization manifest digest is stale."
        )
    payload = _exact_fields(
        raw,
        expected=frozenset(
            {
                "schema_id",
                "schema_version",
                "producer",
                "recording_id",
                "camera_id",
                "materialization_id",
                "write_policy",
                "completed",
                "source_video_metadata_sha256",
                "decode",
                "images_full_storage",
                "frame_map",
                "physical_chunk_manifest",
                "canonicalization",
            }
        ),
        field_name="acquisition_materialization_manifest",
    )
    if (
        payload["schema_id"] != ACQUISITION_MATERIALIZATION_MANIFEST_SCHEMA_ID
        or type(payload["schema_version"]) is not int
        or payload["schema_version"]
        != ACQUISITION_MATERIALIZATION_MANIFEST_SCHEMA_VERSION
        or payload["producer"] != ACQUISITION_IMPORT_PRODUCER
        or payload["write_policy"] != ACQUISITION_MATERIALIZATION_WRITE_POLICY
        or payload["completed"] is not True
        or payload["canonicalization"] != PIXEL_FRAME_AUTHORITY_CANONICALIZATION
    ):
        raise PixelFrameAuthorityError(
            "Acquisition materialization manifest is not a completed immutable importer record."
        )
    recording_id, camera_id, metadata = _source_video_metadata(root_node)
    if payload["recording_id"] != recording_id or payload["camera_id"] != camera_id:
        raise PixelFrameAuthorityError(
            "Acquisition materialization manifest recording/camera identity changed."
        )
    metadata_digest = _mapping_sha256(metadata)
    if (
        _sha256(
            payload["source_video_metadata_sha256"],
            field_name="source_video_metadata_sha256",
        )
        != metadata_digest
    ):
        raise PixelFrameAuthorityError(
            "Acquisition materialization manifest source metadata is stale."
        )
    decode = _exact_fields(
        payload["decode"],
        expected=frozenset(
            {
                "import_method",
                "import_stage",
                "import_mode",
                "decode_backend",
                "source_decode_surface",
            }
        ),
        field_name="acquisition_materialization_manifest.decode",
    )
    canonical_decode = _canonical_import_decode(
        {**dict(decode), "source_path": metadata.get("source_path")},
        source_path=_required_text(
            metadata.get("source_path"), field_name="source_path"
        ),
    )
    frame_map = _array_pointer(frame_index_node)
    if (
        not _exact_json_equal(decode, canonical_decode)
        or not _exact_json_equal(payload["images_full_storage"], storage)
        or not _exact_json_equal(payload["frame_map"], frame_map)
    ):
        raise PixelFrameAuthorityError(
            "Acquisition materialization manifest does not bind canonical decode, storage metadata, and frame map."
        )
    chunks = _exact_fields(
        payload["physical_chunk_manifest"],
        expected=frozenset(
            {
                "record_ref",
                "record_sha256",
                "entry_count",
                "scope",
                "content_evidence_scope",
                "metadata_only_verification_scope",
            }
        ),
        field_name="acquisition_materialization_manifest.physical_chunk_manifest",
    )
    expected_chunk_pointer = {
        "record_ref": (
            f"/{canonical_node_path(manifest_node)}@"
            f"{ACQUISITION_PHYSICAL_CHUNK_MANIFEST_ATTR}"
        ),
        "record_sha256": chunk_stored_digest,
        "entry_count": len(entries),
        "scope": ACQUISITION_CHUNK_MANIFEST_SCOPE,
        "content_evidence_scope": ACQUISITION_CHUNK_CONTENT_EVIDENCE_SCOPE,
        "metadata_only_verification_scope": (
            ACQUISITION_METADATA_ONLY_VERIFICATION_SCOPE
        ),
    }
    if not _exact_json_equal(chunks, expected_chunk_pointer):
        raise PixelFrameAuthorityError(
            "Acquisition materialization does not dereference the exact physical-chunk manifest."
        )
    identity_basis = {
        "recording_id": recording_id,
        "camera_id": camera_id,
        "source_video_metadata_sha256": metadata_digest,
        "decode": canonical_decode,
        "images_full_storage": storage,
        "frame_map": frame_map,
        "physical_chunk_manifest": expected_chunk_pointer,
    }
    if _sha256(
        payload["materialization_id"], field_name="materialization_id"
    ) != _mapping_sha256(identity_basis):
        raise PixelFrameAuthorityError("Acquisition materialization identity is stale.")
    canonical = json.loads(_canonical_json(payload))
    if not _exact_json_equal(raw, canonical):
        raise PixelFrameAuthorityError(
            "Raw acquisition materialization manifest is not canonical."
        )
    if stored_manifest_digest != _mapping_sha256(canonical):
        raise PixelFrameAuthorityError(
            "Acquisition materialization manifest digest is stale."
        )
    return canonical


def _exact_positive_int(value: Any, *, field_name: str) -> int:
    if type(value) is not int or value <= 0:
        raise PixelFrameAuthorityError(
            f"{field_name} must be an exact positive integer."
        )
    return value


def _exact_posix_file_path(
    value: Any,
    *,
    field_name: str,
    absolute: bool,
) -> str:
    path = _required_text(value, field_name=field_name)
    if "\x00" in path or "\\" in path:
        raise PixelFrameAuthorityError(
            f"{field_name} must use an exact POSIX path without NUL/backslash."
        )
    parts = path.split("/")
    path_is_absolute = path.startswith("/")
    body = parts[1:] if path_is_absolute else parts
    if (
        path_is_absolute != absolute
        or not body
        or any(part in {"", ".", ".."} for part in body)
        or PurePosixPath(path).as_posix() != path
    ):
        kind = "absolute" if absolute else "recording-relative"
        raise PixelFrameAuthorityError(
            f"{field_name} must be one canonical {kind} POSIX file path."
        )
    return path


def _parse_clipped_collection_file_evidence(
    value: Any,
    *,
    field_name: str,
) -> dict[str, Any]:
    payload = _exact_fields(
        value,
        expected=frozenset({"relative_path", "sha256", "size_bytes", "mtime_ns"}),
        field_name=field_name,
    )
    return {
        "relative_path": _exact_posix_file_path(
            payload["relative_path"],
            field_name=f"{field_name}.relative_path",
            absolute=False,
        ),
        "sha256": _sha256(payload["sha256"], field_name=f"{field_name}.sha256"),
        "size_bytes": _exact_positive_int(
            payload["size_bytes"], field_name=f"{field_name}.size_bytes"
        ),
        "mtime_ns": _exact_positive_int(
            payload["mtime_ns"], field_name=f"{field_name}.mtime_ns"
        ),
    }


def _parse_clipped_collection_fingerprint(
    value: Any,
    *,
    field_name: str,
) -> dict[str, Any]:
    payload = _exact_fields(
        value,
        expected=frozenset(
            {"strategy", "value", "size_bytes", "mtime_ns", "relocation_stable"}
        ),
        field_name=field_name,
    )
    if payload["strategy"] != "stat_v1":
        raise PixelFrameAuthorityError(f"{field_name}.strategy must be exact stat_v1.")
    if payload["relocation_stable"] is not False:
        raise PixelFrameAuthorityError(
            f"{field_name}.relocation_stable must be exact false."
        )
    return {
        "strategy": "stat_v1",
        "value": _sha256(payload["value"], field_name=f"{field_name}.value"),
        "size_bytes": _exact_positive_int(
            payload["size_bytes"], field_name=f"{field_name}.size_bytes"
        ),
        "mtime_ns": _exact_positive_int(
            payload["mtime_ns"], field_name=f"{field_name}.mtime_ns"
        ),
        "relocation_stable": False,
    }


def _parse_clipped_video_collection_metadata(value: Any) -> dict[str, Any]:
    metadata = _exact_fields(
        value,
        expected=frozenset(
            {
                "schema_id",
                "layout",
                "camera_id",
                "width",
                "height",
                "total_frames",
                "fps",
                "codec",
                "pix_fmt",
                "locator",
                "collection",
            }
        ),
        field_name="source_video_metadata",
    )
    if metadata["schema_id"] != SOURCE_VIDEO_COLLECTION_METADATA_SCHEMA_ID:
        raise PixelFrameAuthorityError("Unsupported clipped source metadata schema_id.")
    if metadata["layout"] != SOURCE_VIDEO_COLLECTION_LAYOUT:
        raise PixelFrameAuthorityError("Unsupported clipped source metadata layout.")
    camera_id = _required_text(
        metadata["camera_id"], field_name="source_video_metadata.camera_id"
    )
    if _FRAME_ID_RE.fullmatch(camera_id) is None:
        raise PixelFrameAuthorityError(
            "source_video_metadata.camera_id must be one canonical path segment."
        )
    width = _exact_positive_int(
        metadata["width"], field_name="source_video_metadata.width"
    )
    height = _exact_positive_int(
        metadata["height"], field_name="source_video_metadata.height"
    )
    total_frames = _exact_positive_int(
        metadata["total_frames"], field_name="source_video_metadata.total_frames"
    )
    fps_raw = metadata["fps"]
    if isinstance(fps_raw, bool) or not isinstance(fps_raw, (int, float)):
        raise PixelFrameAuthorityError("source_video_metadata.fps must be numeric.")
    fps = float(fps_raw)
    if not np.isfinite(fps) or fps <= 0:
        raise PixelFrameAuthorityError("source_video_metadata.fps must be positive.")
    for name in ("codec", "pix_fmt"):
        if metadata[name] is not None:
            _required_text(metadata[name], field_name=f"source_video_metadata.{name}")

    locator = _exact_fields(
        metadata["locator"],
        expected=frozenset({"kind", "relative_path"}),
        field_name="source_video_metadata.locator",
    )
    if locator["kind"] != SOURCE_VIDEO_COLLECTION_LOCATOR_KIND:
        raise PixelFrameAuthorityError("Unsupported clipped collection locator kind.")
    frame_index_relative = _exact_posix_file_path(
        locator["relative_path"],
        field_name="source_video_metadata.locator.relative_path",
        absolute=False,
    )

    raw_collection = _exact_fields(
        metadata["collection"],
        expected=frozenset(
            {
                "schema_id",
                "schema_version",
                "fingerprint_strategy",
                "recording_clip_index",
                "recording_frame_index",
                "recording_frame_index_manifest",
                "members",
                "collection_sha256",
            }
        ),
        field_name="source_video_metadata.collection",
    )
    if (
        raw_collection["schema_id"] != SOURCE_VIDEO_COLLECTION_SCHEMA_ID
        or raw_collection["schema_version"] != 1
        or raw_collection["fingerprint_strategy"]
        != SOURCE_VIDEO_COLLECTION_FINGERPRINT_STRATEGY
    ):
        raise PixelFrameAuthorityError("Unsupported clipped collection contract.")
    file_evidence = {
        name: _parse_clipped_collection_file_evidence(
            raw_collection[name],
            field_name=f"source_video_metadata.collection.{name}",
        )
        for name in (
            "recording_clip_index",
            "recording_frame_index",
            "recording_frame_index_manifest",
        )
    }
    if file_evidence["recording_frame_index"]["relative_path"] != frame_index_relative:
        raise PixelFrameAuthorityError(
            "Clipped collection locator and frame-index evidence disagree."
        )
    raw_members = raw_collection["members"]
    if type(raw_members) is not list or not raw_members:
        raise PixelFrameAuthorityError("Clipped collection requires source members.")
    members: list[dict[str, Any]] = []
    expected_start = 0
    previous_clip_index = -1
    seen_ids: set[str] = set()
    seen_indices: set[int] = set()
    for member_index, raw_member in enumerate(raw_members):
        field = f"source_video_metadata.collection.members[{member_index}]"
        member = _exact_fields(
            raw_member,
            expected=frozenset(
                {
                    "clip_id",
                    "clip_index",
                    "relative_path",
                    "frame_count",
                    "first_frame_index",
                    "last_frame_index_inclusive",
                    "width",
                    "height",
                    "fps",
                    "codec",
                    "pix_fmt",
                    "file_fingerprint",
                }
            ),
            field_name=field,
        )
        clip_id = _required_text(member["clip_id"], field_name=f"{field}.clip_id")
        clip_index = member["clip_index"]
        if type(clip_index) is not int or clip_index < 0:
            raise PixelFrameAuthorityError(f"{field}.clip_index must be nonnegative.")
        if clip_index <= previous_clip_index:
            raise PixelFrameAuthorityError(
                "Clipped collection members must be ordered by increasing clip_index."
            )
        previous_clip_index = clip_index
        if clip_id in seen_ids or clip_index in seen_indices:
            raise PixelFrameAuthorityError(
                "Clipped collection member identity is duplicated."
            )
        seen_ids.add(clip_id)
        seen_indices.add(clip_index)
        frame_count = _exact_positive_int(
            member["frame_count"], field_name=f"{field}.frame_count"
        )
        if (
            type(member["first_frame_index"]) is not int
            or member["first_frame_index"] != expected_start
            or type(member["last_frame_index_inclusive"]) is not int
            or member["last_frame_index_inclusive"] != expected_start + frame_count - 1
        ):
            raise PixelFrameAuthorityError(
                "Clipped collection members must exactly tile the acquisition timeline."
            )
        member_fps_raw = member["fps"]
        if isinstance(member_fps_raw, bool) or not isinstance(
            member_fps_raw, (int, float)
        ):
            raise PixelFrameAuthorityError(f"{field}.fps must be numeric.")
        member_fps = float(member_fps_raw)
        if (
            _exact_positive_int(member["width"], field_name=f"{field}.width") != width
            or _exact_positive_int(member["height"], field_name=f"{field}.height")
            != height
            or not np.isfinite(member_fps)
            or not np.isclose(member_fps, fps, rtol=0.0, atol=1e-6)
        ):
            raise PixelFrameAuthorityError(
                "Clipped collection member geometry/frame rate is inconsistent."
            )
        for name in ("codec", "pix_fmt"):
            if member[name] is not None:
                _required_text(member[name], field_name=f"{field}.{name}")
            if metadata[name] is not None and member[name] != metadata[name]:
                raise PixelFrameAuthorityError(
                    f"Clipped collection member {name} differs from collection metadata."
                )
        members.append(
            {
                "clip_id": clip_id,
                "clip_index": clip_index,
                "relative_path": _exact_posix_file_path(
                    member["relative_path"],
                    field_name=f"{field}.relative_path",
                    absolute=False,
                ),
                "frame_count": frame_count,
                "first_frame_index": expected_start,
                "last_frame_index_inclusive": expected_start + frame_count - 1,
                "width": width,
                "height": height,
                "fps": member_fps,
                "codec": member["codec"],
                "pix_fmt": member["pix_fmt"],
                "file_fingerprint": _parse_clipped_collection_fingerprint(
                    member["file_fingerprint"], field_name=f"{field}.file_fingerprint"
                ),
            }
        )
        expected_start += frame_count
    if expected_start != total_frames:
        raise PixelFrameAuthorityError(
            "Clipped collection member coverage differs from total_frames."
        )
    collection_basis = {
        "schema_id": SOURCE_VIDEO_COLLECTION_SCHEMA_ID,
        "schema_version": 1,
        "fingerprint_strategy": SOURCE_VIDEO_COLLECTION_FINGERPRINT_STRATEGY,
        **file_evidence,
        "members": members,
    }
    collection_sha = _sha256(
        raw_collection["collection_sha256"],
        field_name="source_video_metadata.collection.collection_sha256",
    )
    if collection_sha != _mapping_sha256(collection_basis):
        raise PixelFrameAuthorityError("Clipped collection digest is stale.")
    return {
        "schema_id": SOURCE_VIDEO_COLLECTION_METADATA_SCHEMA_ID,
        "layout": SOURCE_VIDEO_COLLECTION_LAYOUT,
        "camera_id": camera_id,
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "fps": fps,
        "codec": metadata["codec"],
        "pix_fmt": metadata["pix_fmt"],
        "locator": {
            "kind": SOURCE_VIDEO_COLLECTION_LOCATOR_KIND,
            "relative_path": frame_index_relative,
        },
        "collection": {**collection_basis, "collection_sha256": collection_sha},
    }


def parse_source_video_metadata(value: Any) -> dict[str, Any]:
    """Parse exact source metadata used as acquisition identity evidence."""

    if not isinstance(value, Mapping):
        raise PixelFrameAuthorityError(
            "Canonical acquisition authority requires source_video_metadata."
        )
    metadata = json.loads(_canonical_json(value))
    if metadata.get("schema_id") == SOURCE_VIDEO_COLLECTION_METADATA_SCHEMA_ID:
        return _parse_clipped_video_collection_metadata(metadata)
    if metadata.get("schema_id") != "palette.source_video_metadata.v2":
        raise PixelFrameAuthorityError(
            "Acquisition authority requires source_video_metadata v2."
        )
    if metadata.get("layout") != "single_video":
        raise PixelFrameAuthorityError(
            "Acquisition authority currently requires single_video layout."
        )
    camera_id = _required_text(
        metadata.get("camera_id"), field_name="source_video_metadata.camera_id"
    )
    if _FRAME_ID_RE.fullmatch(camera_id) is None:
        raise PixelFrameAuthorityError(
            "source_video_metadata.camera_id must be one canonical path segment."
        )
    for name in ("width", "height", "total_frames"):
        _exact_positive_int(
            metadata.get(name), field_name=f"source_video_metadata.{name}"
        )

    locator = metadata.get("locator")
    if type(locator) is not dict:
        raise PixelFrameAuthorityError(
            "Acquisition authority requires an exact source-video locator object."
        )
    locator_kind = _required_text(
        locator.get("kind"), field_name="source_video_metadata.locator.kind"
    )
    if locator_kind == "recording_relative":
        if set(locator) != {"kind", "relative_path"}:
            raise PixelFrameAuthorityError(
                "Recording-relative source-video locator has unexpected fields."
            )
        _exact_posix_file_path(
            locator.get("relative_path"),
            field_name="source_video_metadata.locator.relative_path",
            absolute=False,
        )
    elif locator_kind == "absolute":
        if set(locator) != {"kind", "path"}:
            raise PixelFrameAuthorityError(
                "Absolute source-video locator has unexpected fields."
            )
        absolute_path = _exact_posix_file_path(
            locator.get("path"),
            field_name="source_video_metadata.locator.path",
            absolute=True,
        )
        if "source_path" in metadata and metadata.get("source_path") != absolute_path:
            raise PixelFrameAuthorityError(
                "Absolute source_path mirror must equal the exact canonical locator."
            )
    else:
        raise PixelFrameAuthorityError(
            "Acquisition authority source-video locator kind is unsupported."
        )
    if "source_path" in metadata:
        _exact_posix_file_path(
            metadata.get("source_path"),
            field_name="source_video_metadata.source_path",
            absolute=True,
        )

    fingerprint = metadata.get("file_fingerprint")
    exact_fingerprint_fields = {
        "strategy",
        "value",
        "size_bytes",
        "mtime_ns",
        "relocation_stable",
    }
    if type(fingerprint) is not dict or set(fingerprint) != exact_fingerprint_fields:
        raise PixelFrameAuthorityError(
            "Acquisition authority requires an exact source-video file fingerprint."
        )
    _required_text(
        fingerprint.get("strategy"),
        field_name="source_video_metadata.file_fingerprint.strategy",
    )
    _required_text(
        fingerprint.get("value"),
        field_name="source_video_metadata.file_fingerprint.value",
    )
    for name in ("size_bytes", "mtime_ns"):
        field_value = fingerprint.get(name)
        if type(field_value) is not int or field_value < 0:
            raise PixelFrameAuthorityError(
                f"source_video_metadata.file_fingerprint.{name} must be an exact nonnegative integer."
            )
    if fingerprint.get("relocation_stable") is not False:
        raise PixelFrameAuthorityError(
            "Source-video fingerprint must explicitly declare relocation_stable=false."
        )
    return metadata


def _source_video_metadata(root_node: Any) -> tuple[str, str, dict[str, Any]]:
    attrs = getattr(root_node, "attrs", None)
    if not isinstance(attrs, Mapping):
        raise PixelFrameAuthorityError("Archive root must expose persisted attrs.")
    recording_id = _required_text(attrs.get("recording_id"), field_name="recording_id")
    raw_metadata = attrs.get("source_video_metadata")
    metadata = parse_source_video_metadata(raw_metadata)
    if not _exact_json_equal(raw_metadata, metadata):
        raise PixelFrameAuthorityError(
            "Raw source_video_metadata is not its exact parsed canonical form."
        )
    camera_id = metadata["camera_id"]
    return recording_id, camera_id, metadata


def build_verified_acquisition_materialization(
    root_node: Any,
    *,
    frame_node: Any,
    frame_index_node: Any,
    import_operation_attrs: Mapping[str, Any],
) -> VerifiedAcquisitionMaterialization:
    """Bind importer evidence without re-reading materialized image chunks.

    This detects missing/stale manifests, path or storage-metadata replacement,
    and frame-map changes.  It intentionally cannot detect post-hoc encoded
    chunk mutation when storage metadata and the persisted importer evidence
    remain unchanged; a maintenance verifier must re-hash chunks for that.
    """

    recording_id, camera_id, metadata = _source_video_metadata(root_node)
    resolved_frame, resolved_index = _resolve_materialized_acquisition_nodes(root_node)
    if not (
        _same_resolved_persisted_node(resolved_frame, frame_node)
        and _same_resolved_persisted_node(resolved_index, frame_index_node)
    ):
        raise PixelFrameAuthorityError(
            "Acquisition materialization receipt requires the exact canonical nodes resolved from the archive root; detached lookalikes are forbidden."
        )
    try:
        require_same_archive(root_node, frame_node, frame_index_node)
    except ArchiveIdentityError as exc:
        raise PixelFrameAuthorityError(str(exc)) from exc
    if (
        canonical_node_path(frame_node) != "raw_video/images_full"
        or canonical_node_path(frame_index_node)
        != "raw_video/frame_domain_maps/stored_zarr_frame_to_acquisition_frame"
    ):
        raise PixelFrameAuthorityError(
            "Acquisition materialization receipt requires canonical image and frame-map nodes."
        )
    locator = metadata.get("locator")
    fingerprint = metadata.get("file_fingerprint")
    if not isinstance(locator, Mapping) or not isinstance(fingerprint, Mapping):
        raise PixelFrameAuthorityError(
            "Acquisition materialization requires exact source locator and fingerprint."
        )
    decode = _canonical_import_decode(
        import_operation_attrs,
        source_path=_required_text(
            metadata.get("source_path"), field_name="source_path"
        ),
    )
    persisted_manifest = _load_acquisition_materialization_manifest(
        root_node,
        frame_node,
        frame_index_node,
    )
    if not _exact_json_equal(persisted_manifest["decode"], decode):
        raise PixelFrameAuthorityError(
            "Live import-operation evidence disagrees with the immutable persisted materialization manifest."
        )
    operation = {
        "schema_id": "palette.acquisition_materialization_receipt",
        "schema_version": 1,
        "producer": ACQUISITION_IMPORT_PRODUCER,
        "recording_id": recording_id,
        "camera_id": camera_id,
        "source_locator": json.loads(_canonical_json(locator)),
        "source_fingerprint": json.loads(_canonical_json(fingerprint)),
        "decode": decode,
        "materialization_manifest": {
            "record_ref": (
                f"/{ACQUISITION_MATERIALIZATION_MANIFEST_PATH}@"
                f"{ACQUISITION_MATERIALIZATION_MANIFEST_ATTR}"
            ),
            "record_sha256": _mapping_sha256(persisted_manifest),
            "materialization_id": persisted_manifest["materialization_id"],
            "physical_chunk_manifest": copy.deepcopy(
                dict(persisted_manifest["physical_chunk_manifest"])
            ),
            "verification_scope": ACQUISITION_METADATA_ONLY_VERIFICATION_SCOPE,
        },
        "canonicalization": PIXEL_FRAME_AUTHORITY_CANONICALIZATION,
    }
    return VerifiedAcquisitionMaterialization(
        operation=operation,
        _verification_seal=_VERIFIED_ACQUISITION_MATERIALIZATION_SEAL,
    )


def _require_verified_acquisition_materialization(
    value: Any,
    *,
    root_node: Any,
    frame_node: Any,
    frame_index_node: Any,
) -> VerifiedAcquisitionMaterialization:
    if (
        type(value) is not VerifiedAcquisitionMaterialization
        or value._seal is not _VERIFIED_ACQUISITION_MATERIALIZATION_SEAL
    ):
        raise PixelFrameAuthorityError(
            "Materialized frames require a sealed acquisition-import receipt."
        )
    operation = value.operation
    if not isinstance(operation, Mapping):
        raise PixelFrameAuthorityError(
            "Acquisition materialization receipt is invalid."
        )
    rebuilt = build_verified_acquisition_materialization(
        root_node,
        frame_node=frame_node,
        frame_index_node=frame_index_node,
        import_operation_attrs={
            **dict(operation.get("decode", {})),
            "source_path": _source_video_metadata(root_node)[2].get("source_path"),
        },
    )
    if not _exact_json_equal(operation, rebuilt.operation):
        raise PixelFrameAuthorityError(
            "Acquisition materialization receipt changed or no longer binds the canonical importer manifest, storage metadata, and frame map."
        )
    return value


def _bind_persisted_acquisition_materialization(
    operation: Mapping[str, Any],
    *,
    root_node: Any,
    frame_node: Any,
    frame_index_node: Any,
) -> VerifiedAcquisitionMaterialization:
    if not isinstance(operation, Mapping):
        raise PixelFrameAuthorityError(
            "Persisted acquisition materialization receipt is missing."
        )
    decode = operation.get("decode")
    if not isinstance(decode, Mapping):
        raise PixelFrameAuthorityError(
            "Persisted acquisition materialization decode record is missing."
        )
    metadata = _source_video_metadata(root_node)[2]
    rebuilt = build_verified_acquisition_materialization(
        root_node,
        frame_node=frame_node,
        frame_index_node=frame_index_node,
        import_operation_attrs={
            **dict(decode),
            "source_path": metadata.get("source_path"),
        },
    )
    if not _exact_json_equal(operation, rebuilt.operation):
        raise PixelFrameAuthorityError(
            "Persisted acquisition receipt conflicts with source, decode, importer chunk evidence, storage metadata, or frame map."
        )
    return rebuilt


def _acquisition_ownership_record(
    root_node: Any,
    authority_node: Any,
    *,
    frame_node: Any | None,
    frame_index_node: Any | None,
    materialization: VerifiedAcquisitionMaterialization | None,
) -> AcquisitionImportOwnershipRecord:
    recording_id, camera_id, metadata = _source_video_metadata(root_node)
    expected_authority_path = f"analysis/acquisition_camera_frames/{camera_id}"
    if canonical_node_path(authority_node) != expected_authority_path:
        raise PixelFrameAuthorityError(
            "Acquisition ownership must be persisted at the canonical camera authority path."
        )
    if (frame_node is None) != (frame_index_node is None):
        raise PixelFrameAuthorityError(
            "Materialized acquisition ownership requires both frames and source indices."
        )
    if frame_node is None:
        if materialization is not None:
            raise PixelFrameAuthorityError(
                "External-video authority cannot claim a materialization receipt."
            )
        mode = (
            CLIPPED_EXTERNAL_ACQUISITION_AUTHORITY_MODE
            if metadata.get("schema_id") == SOURCE_VIDEO_COLLECTION_METADATA_SCHEMA_ID
            else EXTERNAL_ACQUISITION_AUTHORITY_MODE
        )
        frame_array = None
        frame_index = None
        import_operation = None
    else:
        expected_frame_path = "raw_video/images_full"
        expected_index_path = (
            "raw_video/frame_domain_maps/stored_zarr_frame_to_acquisition_frame"
        )
        if canonical_node_path(frame_node) != expected_frame_path:
            raise PixelFrameAuthorityError(
                "Materialized acquisition frames must use the canonical raw-video path; "
                "same-sized debug arrays are not acquisition authority."
            )
        if canonical_node_path(frame_index_node) != expected_index_path:
            raise PixelFrameAuthorityError(
                "Acquisition source indices must use the canonical raw-video lineage path."
            )
        frame_attrs = getattr(frame_node, "attrs", None)
        if not isinstance(frame_attrs, Mapping):
            raise PixelFrameAuthorityError(
                "Materialized acquisition frames must expose persisted attrs."
            )
        index_attrs = getattr(frame_index_node, "attrs", None)
        if not isinstance(index_attrs, Mapping) or (
            index_attrs.get("source_domain") != "stored_zarr_frame"
            or index_attrs.get("target_domain") != "acquisition_frame"
        ):
            raise PixelFrameAuthorityError(
                "Acquisition ownership requires the canonical stored-Zarr to acquisition-frame map."
            )
        if index_attrs.get("semantics") not in {
            "identity_map_zero_based_full_import",
            "explicit_stored_zarr_to_acquisition_frame_v1",
        }:
            raise PixelFrameAuthorityError(
                "Acquisition frame-domain map semantics are unsupported."
            )
        try:
            extent = bind_array_reference_extent(frame_node, units="px")
        except CoordinateReferenceError as exc:
            raise PixelFrameAuthorityError(
                f"Materialized acquisition frame array is invalid: {exc}"
            ) from exc
        if extent.width != metadata["width"] or extent.height != metadata["height"]:
            raise PixelFrameAuthorityError(
                "Materialized acquisition dimensions conflict with source-video metadata."
            )
        indices = _array_values(frame_index_node)
        shape = tuple(int(item) for item in getattr(frame_node, "shape", ()))
        if (
            len(shape) != 3
            or shape[0] <= 0
            or not _array_has_exact_dtype(frame_node, "uint8")
            or indices.ndim != 1
            or indices.dtype != np.dtype("<i8")
            or indices.shape[0] != shape[0]
        ):
            raise PixelFrameAuthorityError(
                "Materialized acquisition frames must be rank-3 uint8 and use a "
                "row-aligned little-endian int64 source-index map."
            )
        normalized = indices.astype(np.int64)
        if (
            np.any(normalized < 0)
            or np.any(normalized >= int(metadata["total_frames"]))
            or (normalized.size > 1 and np.any(np.diff(normalized) <= 0))
        ):
            raise PixelFrameAuthorityError(
                "Acquisition source indices must be strictly increasing and in range."
            )
        mode = MATERIALIZED_ACQUISITION_AUTHORITY_MODE
        receipt = _require_verified_acquisition_materialization(
            materialization,
            root_node=root_node,
            frame_node=frame_node,
            frame_index_node=frame_index_node,
        )
        frame_array = {
            **_extent_pointer(extent),
            "storage_identity": _array_storage_identity(frame_node),
        }
        frame_index = _array_pointer(frame_index_node)
        import_operation = copy.deepcopy(dict(receipt.operation))
    return AcquisitionImportOwnershipRecord(
        recording_id=recording_id,
        camera_id=camera_id,
        producer=ACQUISITION_IMPORT_PRODUCER,
        mode=mode,
        source_video_metadata_sha256=_mapping_sha256(metadata),
        frame_array=frame_array,
        frame_index=frame_index,
        import_operation=import_operation,
    )


def parse_acquisition_import_ownership(
    value: Any,
) -> AcquisitionImportOwnershipRecord:
    if isinstance(value, AcquisitionImportOwnershipRecord):
        value = value.to_dict()
    payload = _exact_fields(
        value,
        expected=frozenset(
            {
                "schema_id",
                "schema_version",
                "recording_id",
                "camera_id",
                "producer",
                "mode",
                "source_video_metadata_sha256",
                "frame_array",
                "frame_index",
                "import_operation",
                "canonicalization",
            }
        ),
        field_name="acquisition_import_ownership",
    )
    if payload["schema_id"] != ACQUISITION_IMPORT_OWNERSHIP_SCHEMA_ID:
        raise PixelFrameAuthorityError("Unsupported acquisition-ownership schema_id.")
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != ACQUISITION_IMPORT_OWNERSHIP_SCHEMA_VERSION
    ):
        raise PixelFrameAuthorityError(
            "Unsupported acquisition-ownership schema_version."
        )
    if payload["producer"] != ACQUISITION_IMPORT_PRODUCER:
        raise PixelFrameAuthorityError("Unsupported acquisition ownership producer.")
    if payload["canonicalization"] != PIXEL_FRAME_AUTHORITY_CANONICALIZATION:
        raise PixelFrameAuthorityError(
            "Unsupported acquisition ownership canonicalization."
        )
    mode = payload["mode"]
    if mode not in {
        CLIPPED_EXTERNAL_ACQUISITION_AUTHORITY_MODE,
        EXTERNAL_ACQUISITION_AUTHORITY_MODE,
        MATERIALIZED_ACQUISITION_AUTHORITY_MODE,
    }:
        raise PixelFrameAuthorityError("Unsupported acquisition ownership mode.")
    frame_array = payload["frame_array"]
    frame_index = payload["frame_index"]
    import_operation = payload["import_operation"]
    if mode in {
        CLIPPED_EXTERNAL_ACQUISITION_AUTHORITY_MODE,
        EXTERNAL_ACQUISITION_AUTHORITY_MODE,
    }:
        if (
            frame_array is not None
            or frame_index is not None
            or import_operation is not None
        ):
            raise PixelFrameAuthorityError(
                "External-video ownership cannot claim materialized arrays."
            )
    elif (
        not isinstance(frame_array, Mapping)
        or not isinstance(frame_index, Mapping)
        or not isinstance(import_operation, Mapping)
    ):
        raise PixelFrameAuthorityError(
            "Materialized acquisition ownership requires exact array pointers."
        )
    return AcquisitionImportOwnershipRecord(
        recording_id=_required_text(payload["recording_id"], field_name="recording_id"),
        camera_id=_required_text(payload["camera_id"], field_name="camera_id"),
        producer=ACQUISITION_IMPORT_PRODUCER,
        mode=mode,
        source_video_metadata_sha256=_sha256(
            payload["source_video_metadata_sha256"],
            field_name="source_video_metadata_sha256",
        ),
        frame_array=(
            json.loads(_canonical_json(frame_array))
            if frame_array is not None
            else None
        ),
        frame_index=(
            json.loads(_canonical_json(frame_index))
            if frame_index is not None
            else None
        ),
        import_operation=(
            json.loads(_canonical_json(import_operation))
            if import_operation is not None
            else None
        ),
    )


def load_acquisition_import_ownership(
    root_node: Any,
    authority_node: Any,
    *,
    frame_node: Any | None = None,
    frame_index_node: Any | None = None,
) -> BoundAcquisitionImportOwnership:
    archive = _acquisition_archive(
        root_node, authority_node, frame_node, frame_index_node
    )
    attrs = getattr(authority_node, "attrs", None)
    if not isinstance(attrs, Mapping):
        raise PixelFrameAuthorityError("Acquisition ownership must expose attrs.")
    raw = attrs.get(ACQUISITION_IMPORT_OWNERSHIP_ATTR)
    record = parse_acquisition_import_ownership(raw)
    if not isinstance(raw, Mapping) or not _exact_json_equal(raw, record.to_dict()):
        raise PixelFrameAuthorityError(
            "Raw acquisition ownership is not its exact canonical mapping."
        )
    stored = _sha256(
        attrs.get(ACQUISITION_IMPORT_OWNERSHIP_DIGEST_ATTR),
        field_name=ACQUISITION_IMPORT_OWNERSHIP_DIGEST_ATTR,
    )
    if stored != record.digest():
        raise PixelFrameAuthorityError("Acquisition ownership digest is stale.")
    receipt = (
        _bind_persisted_acquisition_materialization(
            record.import_operation,
            root_node=root_node,
            frame_node=frame_node,
            frame_index_node=frame_index_node,
        )
        if record.mode == MATERIALIZED_ACQUISITION_AUTHORITY_MODE
        else None
    )
    expected = _acquisition_ownership_record(
        root_node,
        authority_node,
        frame_node=frame_node,
        frame_index_node=frame_index_node,
        materialization=receipt,
    )
    if record != expected:
        raise PixelFrameAuthorityError(
            "Acquisition ownership conflicts with exact source/materialization lineage."
        )
    return BoundAcquisitionImportOwnership(
        record=record,
        archive=archive,
        root_node=root_node,
        authority_node=authority_node,
        frame_node=frame_node,
        frame_index_node=frame_index_node,
        _verification_seal=_BOUND_ACQUISITION_OWNERSHIP_SEAL,
    )


def _require_canonical_child(
    parent: Any,
    name: str,
    *,
    expected_path: str,
) -> Any:
    """Resolve one named child and reject aliases or path-spoofed objects."""

    try:
        child = parent[name]
    except Exception as exc:
        raise PixelFrameAuthorityError(
            f"Required canonical acquisition node {expected_path!r} is missing."
        ) from exc
    if canonical_node_path(child) != expected_path:
        raise PixelFrameAuthorityError(
            f"Resolved acquisition node does not have exact path {expected_path!r}."
        )
    return child


def _resolve_materialized_acquisition_nodes(root_node: Any) -> tuple[Any, Any]:
    """Resolve the only materialized acquisition pixel/map pair from root."""

    raw_video = _require_canonical_child(
        root_node,
        "raw_video",
        expected_path="raw_video",
    )
    frame_node = _require_canonical_child(
        raw_video,
        "images_full",
        expected_path="raw_video/images_full",
    )
    frame_maps = _require_canonical_child(
        raw_video,
        "frame_domain_maps",
        expected_path="raw_video/frame_domain_maps",
    )
    frame_index_node = _require_canonical_child(
        frame_maps,
        "stored_zarr_frame_to_acquisition_frame",
        expected_path=(
            "raw_video/frame_domain_maps/stored_zarr_frame_to_acquisition_frame"
        ),
    )
    return frame_node, frame_index_node


def _resolve_acquisition_materialization_manifest_node(root_node: Any) -> Any:
    """Resolve the importer's sole canonical materialization-manifest node."""

    raw_video = _require_canonical_child(
        root_node,
        "raw_video",
        expected_path="raw_video",
    )
    manifests = _require_canonical_child(
        raw_video,
        "manifests",
        expected_path="raw_video/manifests",
    )
    return _require_canonical_child(
        manifests,
        "images_full_materialization",
        expected_path=ACQUISITION_MATERIALIZATION_MANIFEST_PATH,
    )


def load_persisted_acquisition_camera_authority(
    root_node: Any,
    *,
    expected_camera_id: str | None = None,
) -> tuple[BoundAcquisitionImportOwnership, BoundAcquisitionCameraFrame]:
    """Resolve one canonical persisted acquisition authority without inference.

    The nested ``source_video_metadata.camera_id`` selects the sole allowed
    authority path.  Materialized nodes are opened only after the exact,
    digest-verified ownership record declares materialized mode; external mode
    never probes or inherits raw-video arrays.
    """

    _, camera_id, _ = _source_video_metadata(root_node)
    if expected_camera_id is not None:
        expected = _required_text(
            expected_camera_id,
            field_name="expected_camera_id",
        )
        if _FRAME_ID_RE.fullmatch(expected) is None or expected != camera_id:
            raise PixelFrameAuthorityError(
                "Expected camera_id does not equal source_video_metadata.camera_id."
            )
    analysis = _require_canonical_child(
        root_node,
        "analysis",
        expected_path="analysis",
    )
    authorities = _require_canonical_child(
        analysis,
        "acquisition_camera_frames",
        expected_path="analysis/acquisition_camera_frames",
    )
    authority_node = _require_canonical_child(
        authorities,
        camera_id,
        expected_path=f"analysis/acquisition_camera_frames/{camera_id}",
    )
    attrs = getattr(authority_node, "attrs", None)
    if not isinstance(attrs, Mapping):
        raise PixelFrameAuthorityError(
            "Persisted acquisition authority must expose attrs."
        )
    raw = attrs.get(ACQUISITION_IMPORT_OWNERSHIP_ATTR)
    record = parse_acquisition_import_ownership(raw)
    if not isinstance(raw, Mapping) or not _exact_json_equal(raw, record.to_dict()):
        raise PixelFrameAuthorityError(
            "Raw acquisition ownership is not its exact canonical mapping."
        )
    stored = _sha256(
        attrs.get(ACQUISITION_IMPORT_OWNERSHIP_DIGEST_ATTR),
        field_name=ACQUISITION_IMPORT_OWNERSHIP_DIGEST_ATTR,
    )
    if stored != record.digest():
        raise PixelFrameAuthorityError("Acquisition ownership digest is stale.")
    if record.camera_id != camera_id:
        raise PixelFrameAuthorityError(
            "Persisted acquisition ownership camera disagrees with source metadata."
        )

    frame_node: Any | None = None
    frame_index_node: Any | None = None
    if record.mode == MATERIALIZED_ACQUISITION_AUTHORITY_MODE:
        frame_node, frame_index_node = _resolve_materialized_acquisition_nodes(
            root_node
        )
    elif record.mode not in {
        CLIPPED_EXTERNAL_ACQUISITION_AUTHORITY_MODE,
        EXTERNAL_ACQUISITION_AUTHORITY_MODE,
    }:  # defensive after strict parse
        raise PixelFrameAuthorityError("Unsupported acquisition ownership mode.")

    ownership = load_acquisition_import_ownership(
        root_node,
        authority_node,
        frame_node=frame_node,
        frame_index_node=frame_index_node,
    )
    frame = load_acquisition_camera_frame(
        root_node,
        authority_node,
        import_ownership=ownership,
    )
    return ownership, frame


def stamp_acquisition_import_ownership(
    root_node: Any,
    authority_node: Any,
    *,
    frame_node: Any | None = None,
    frame_index_node: Any | None = None,
    materialization: VerifiedAcquisitionMaterialization | None = None,
) -> BoundAcquisitionImportOwnership:
    _acquisition_archive(root_node, authority_node, frame_node, frame_index_node)
    record = _acquisition_ownership_record(
        root_node,
        authority_node,
        frame_node=frame_node,
        frame_index_node=frame_index_node,
        materialization=materialization,
    )
    attrs = require_trusted_coordinate_attrs(
        authority_node,
        label="Acquisition ownership",
    )
    snapshot = copy.deepcopy(dict(attrs))
    intended = {
        ACQUISITION_IMPORT_OWNERSHIP_ATTR: record.to_dict(),
        ACQUISITION_IMPORT_OWNERSHIP_DIGEST_ATTR: record.digest(),
    }
    expected = _expected_attrs_after_update(snapshot, intended)
    try:
        attrs.update(copy.deepcopy(intended))
        _require_exact_attrs_state(
            attrs,
            expected,
            label="Acquisition ownership post-write",
        )
        bound = load_acquisition_import_ownership(
            root_node,
            authority_node,
            frame_node=frame_node,
            frame_index_node=frame_index_node,
        )
        _require_exact_attrs_state(
            require_trusted_coordinate_attrs(
                authority_node,
                label="Reloaded acquisition ownership",
            ),
            expected,
            label="Acquisition ownership post-reload",
        )
        return bound
    except Exception as exc:
        try:
            _restore_exact_attrs(attrs, snapshot)
        except Exception as rollback_exc:  # pragma: no cover
            raise PixelFrameAuthorityError(
                f"Acquisition ownership stamp failed and rollback was incomplete: {rollback_exc}"
            ) from exc
        if isinstance(exc, PixelFrameAuthorityError):
            raise
        raise PixelFrameAuthorityError(
            f"Acquisition ownership stamp failed: {exc}"
        ) from exc


def require_bound_acquisition_import_ownership(
    value: Any,
) -> BoundAcquisitionImportOwnership:
    if (
        type(value) is not BoundAcquisitionImportOwnership
        or value._seal is not _BOUND_ACQUISITION_OWNERSHIP_SEAL
    ):
        raise PixelFrameAuthorityError(
            "A sealed acquisition import/materialization ownership record is required."
        )
    verify_persisted_proof(_bound_authority_proof_key(value), value.assert_verified)
    return value


def _acquisition_record(
    root_node: Any,
    *,
    import_ownership: BoundAcquisitionImportOwnership,
) -> AcquisitionCameraFrameRecord:
    ownership = require_bound_acquisition_import_ownership(import_ownership)
    if (
        ownership._root_node is not root_node
        or ownership.archive_identity != archive_identity(root_node)
    ):
        raise PixelFrameAuthorityError(
            "Acquisition camera frame must use ownership bound to the exact root node."
        )
    frame_node = ownership._frame_node
    frame_index_node = ownership._frame_index_node
    recording_id, camera_id, metadata = _source_video_metadata(root_node)
    width = int(metadata["width"])
    height = int(metadata["height"])
    source_total = int(metadata["total_frames"])
    if (frame_node is None) != (frame_index_node is None):
        raise PixelFrameAuthorityError(
            "Materialized acquisition frames require both frame array and exact frame-index array."
        )
    if frame_node is None:
        frame_array = None
        frame_index = None
        frame_count = source_total
        if ownership.record.mode == CLIPPED_EXTERNAL_ACQUISITION_AUTHORITY_MODE:
            collection = metadata["collection"]
            frame_index_evidence = collection["recording_frame_index"]
            frame_domain = {
                "mode": "external_clipped_recording_frame_index_v1",
                "source": "source_video_metadata.collection.recording_frame_index",
                "first_frame_index": 0,
                "last_frame_index_inclusive": source_total - 1,
                "recording_frame_index_relative_path": frame_index_evidence[
                    "relative_path"
                ],
                "recording_frame_index_sha256": frame_index_evidence["sha256"],
                "collection_sha256": collection["collection_sha256"],
            }
        else:
            frame_domain = {
                "mode": "external_video_sequential_frame_index_v1",
                "source": "source_video_metadata.total_frames",
                "first_frame_index": 0,
                "last_frame_index_inclusive": source_total - 1,
            }
    else:
        try:
            extent = bind_array_reference_extent(frame_node, units="px")
        except CoordinateReferenceError as exc:
            raise PixelFrameAuthorityError(
                f"Materialized acquisition frame array is invalid: {exc}"
            ) from exc
        if extent.width != width or extent.height != height:
            raise PixelFrameAuthorityError(
                "Materialized acquisition frame dimensions conflict with source video metadata."
            )
        shape = tuple(int(item) for item in getattr(frame_node, "shape", ()))
        if (
            len(shape) != 3
            or shape[0] <= 0
            or not _array_has_exact_dtype(frame_node, "uint8")
        ):
            raise PixelFrameAuthorityError(
                "Materialized acquisition frames require exact rank-3 uint8 storage."
            )
        indices = _array_values(frame_index_node)
        if indices.ndim != 1 or indices.dtype != np.dtype("<i8"):
            raise PixelFrameAuthorityError(
                "Acquisition frame-index lineage must be a one-dimensional "
                "little-endian int64 array."
            )
        normalized = indices.astype(np.int64)
        if (
            normalized.shape[0] != shape[0]
            or np.any(normalized < 0)
            or np.any(normalized >= source_total)
            or (normalized.size > 1 and np.any(np.diff(normalized) <= 0))
        ):
            raise PixelFrameAuthorityError(
                "Acquisition frame-index lineage must be strictly increasing, in-range, and row-aligned."
            )
        frame_array = _extent_pointer(extent)
        frame_index = _array_pointer(frame_index_node)
        frame_count = shape[0]
        frame_domain = {
            "mode": "explicit_stored_zarr_to_acquisition_frame_map_v1",
            "index_record_ref": frame_index["record_ref"],
            "index_record_sha256": frame_index["record_sha256"],
        }
    return AcquisitionCameraFrameRecord(
        recording_id=recording_id,
        camera_id=camera_id,
        source_video_metadata=metadata,
        source_video_metadata_sha256=_mapping_sha256(metadata),
        width_px=width,
        height_px=height,
        source_total_frames=source_total,
        frame_domain=frame_domain,
        frame_array=frame_array,
        frame_index=frame_index,
        frame_count=frame_count,
        import_ownership={
            "record_ref": ownership.record_ref,
            "record_sha256": ownership.record_sha256,
        },
    )


def parse_acquisition_camera_frame(value: Any) -> AcquisitionCameraFrameRecord:
    if isinstance(value, AcquisitionCameraFrameRecord):
        value = value.to_dict()
    payload = _exact_fields(
        value,
        expected=frozenset(
            {
                "schema_id",
                "schema_version",
                "recording_id",
                "camera_id",
                "source_video_metadata",
                "source_video_metadata_sha256",
                "width_px",
                "height_px",
                "source_total_frames",
                "frame_domain",
                "frame_array",
                "frame_index",
                "frame_count",
                "import_ownership",
                "canonicalization",
            }
        ),
        field_name="acquisition_camera_frame",
    )
    if payload["schema_id"] != ACQUISITION_CAMERA_FRAME_SCHEMA_ID:
        raise PixelFrameAuthorityError("Unsupported acquisition-camera schema_id.")
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != ACQUISITION_CAMERA_FRAME_SCHEMA_VERSION
    ):
        raise PixelFrameAuthorityError("Unsupported acquisition-camera schema_version.")
    if payload["canonicalization"] != PIXEL_FRAME_AUTHORITY_CANONICALIZATION:
        raise PixelFrameAuthorityError(
            "Unsupported acquisition-camera canonicalization."
        )
    normalized_metadata = parse_source_video_metadata(payload["source_video_metadata"])
    digest = _sha256(
        payload["source_video_metadata_sha256"],
        field_name="source_video_metadata_sha256",
    )
    if digest != _mapping_sha256(normalized_metadata):
        raise PixelFrameAuthorityError("Source-video metadata digest is stale.")
    for name in ("frame_domain", "import_ownership"):
        if not isinstance(payload[name], Mapping):
            raise PixelFrameAuthorityError(f"{name} must be a mapping.")
    for name in ("frame_array", "frame_index"):
        if payload[name] is not None and not isinstance(payload[name], Mapping):
            raise PixelFrameAuthorityError(f"{name} must be a mapping or null.")
    camera_id = _required_text(payload["camera_id"], field_name="camera_id")
    if _FRAME_ID_RE.fullmatch(camera_id) is None:
        raise PixelFrameAuthorityError(
            "Acquisition camera_id must be one canonical path segment."
        )
    record = AcquisitionCameraFrameRecord(
        recording_id=_required_text(payload["recording_id"], field_name="recording_id"),
        camera_id=camera_id,
        source_video_metadata=normalized_metadata,
        source_video_metadata_sha256=digest,
        width_px=_exact_positive_int(payload["width_px"], field_name="width_px"),
        height_px=_exact_positive_int(payload["height_px"], field_name="height_px"),
        source_total_frames=_exact_positive_int(
            payload["source_total_frames"], field_name="source_total_frames"
        ),
        frame_domain=json.loads(_canonical_json(payload["frame_domain"])),
        frame_array=(
            json.loads(_canonical_json(payload["frame_array"]))
            if payload["frame_array"] is not None
            else None
        ),
        frame_index=(
            json.loads(_canonical_json(payload["frame_index"]))
            if payload["frame_index"] is not None
            else None
        ),
        frame_count=_exact_positive_int(
            payload["frame_count"], field_name="frame_count"
        ),
        import_ownership=json.loads(_canonical_json(payload["import_ownership"])),
    )
    ownership_pointer = _exact_fields(
        record.import_ownership,
        expected=frozenset({"record_ref", "record_sha256"}),
        field_name="import_ownership",
    )
    expected_ownership_ref = (
        f"/analysis/acquisition_camera_frames/{record.camera_id}"
        f"@{ACQUISITION_IMPORT_OWNERSHIP_ATTR}"
    )
    if (
        _required_text(
            ownership_pointer["record_ref"],
            field_name="import_ownership.record_ref",
        )
        != expected_ownership_ref
    ):
        raise PixelFrameAuthorityError(
            "Acquisition frame must reference exact import ownership."
        )
    _sha256(
        ownership_pointer["record_sha256"],
        field_name="import_ownership.record_sha256",
    )
    if (
        not _exact_json_equal(
            record.source_video_metadata.get("camera_id"), record.camera_id
        )
        or not _exact_json_equal(
            record.source_video_metadata.get("width"), record.width_px
        )
        or not _exact_json_equal(
            record.source_video_metadata.get("height"), record.height_px
        )
        or not _exact_json_equal(
            record.source_video_metadata.get("total_frames"),
            record.source_total_frames,
        )
    ):
        raise PixelFrameAuthorityError(
            "Acquisition camera fields conflict with exact source-video metadata."
        )
    external = record.frame_array is None and record.frame_index is None
    if external:
        if (
            record.source_video_metadata.get("schema_id")
            == SOURCE_VIDEO_COLLECTION_METADATA_SCHEMA_ID
        ):
            collection = record.source_video_metadata["collection"]
            frame_index_evidence = collection["recording_frame_index"]
            expected_frame_domain = {
                "mode": "external_clipped_recording_frame_index_v1",
                "source": "source_video_metadata.collection.recording_frame_index",
                "first_frame_index": 0,
                "last_frame_index_inclusive": record.source_total_frames - 1,
                "recording_frame_index_relative_path": frame_index_evidence[
                    "relative_path"
                ],
                "recording_frame_index_sha256": frame_index_evidence["sha256"],
                "collection_sha256": collection["collection_sha256"],
            }
        else:
            expected_frame_domain = {
                "mode": "external_video_sequential_frame_index_v1",
                "source": "source_video_metadata.total_frames",
                "first_frame_index": 0,
                "last_frame_index_inclusive": record.source_total_frames - 1,
            }
        if (
            record.frame_count != record.source_total_frames
            or record.frame_domain != expected_frame_domain
        ):
            raise PixelFrameAuthorityError(
                "External acquisition frame domain is invalid."
            )
    elif record.frame_array is None or record.frame_index is None:
        raise PixelFrameAuthorityError(
            "Materialized acquisition lineage is incomplete."
        )
    return record


def _acquisition_archive(
    root_node: Any,
    authority_node: Any,
    frame_node: Any | None,
    frame_index_node: Any | None,
) -> ArchiveIdentity:
    nodes = [root_node, authority_node]
    if frame_node is not None:
        nodes.extend((frame_node, frame_index_node))
    try:
        return require_same_archive(*nodes)
    except ArchiveIdentityError as exc:
        raise PixelFrameAuthorityError(str(exc)) from exc


def load_acquisition_camera_frame(
    root_node: Any,
    authority_node: Any,
    *,
    import_ownership: BoundAcquisitionImportOwnership,
) -> BoundAcquisitionCameraFrame:
    ownership = require_bound_acquisition_import_ownership(import_ownership)
    if ownership._authority_node is not authority_node:
        raise PixelFrameAuthorityError(
            "Acquisition frame and import ownership must use the exact authority node."
        )
    archive = _acquisition_archive(
        root_node,
        authority_node,
        ownership._frame_node,
        ownership._frame_index_node,
    )
    if archive != ownership.archive_identity:
        raise PixelFrameAuthorityError(
            "Acquisition frame and import ownership come from different archives."
        )
    attrs = getattr(authority_node, "attrs", None)
    if not isinstance(attrs, Mapping):
        raise PixelFrameAuthorityError("Acquisition authority must expose attrs.")
    raw = attrs.get(ACQUISITION_CAMERA_FRAME_ATTR)
    record = parse_acquisition_camera_frame(raw)
    if not isinstance(raw, Mapping) or not _exact_json_equal(raw, record.to_dict()):
        raise PixelFrameAuthorityError(
            "Raw acquisition-camera mapping is not its parsed canonical form."
        )
    stored = _sha256(
        attrs.get(ACQUISITION_CAMERA_FRAME_DIGEST_ATTR),
        field_name=ACQUISITION_CAMERA_FRAME_DIGEST_ATTR,
    )
    if stored != record.digest():
        raise PixelFrameAuthorityError("Acquisition-camera digest is stale.")
    expected = _acquisition_record(
        root_node,
        import_ownership=ownership,
    )
    if record != expected:
        raise PixelFrameAuthorityError(
            "Acquisition-camera record conflicts with live source metadata/frame lineage."
        )
    return BoundAcquisitionCameraFrame(
        record=record,
        archive=archive,
        root_node=root_node,
        authority_node=authority_node,
        import_ownership=ownership,
        _verification_seal=_BOUND_ACQUISITION_FRAME_SEAL,
    )


def stamp_acquisition_camera_frame(
    root_node: Any,
    authority_node: Any,
    *,
    import_ownership: BoundAcquisitionImportOwnership,
) -> BoundAcquisitionCameraFrame:
    ownership = require_bound_acquisition_import_ownership(import_ownership)
    if ownership._authority_node is not authority_node:
        raise PixelFrameAuthorityError(
            "Acquisition frame must use the exact ownership authority node."
        )
    _acquisition_archive(
        root_node,
        authority_node,
        ownership._frame_node,
        ownership._frame_index_node,
    )
    record = _acquisition_record(
        root_node,
        import_ownership=ownership,
    )
    attrs = require_trusted_coordinate_attrs(
        authority_node,
        label="Acquisition authority",
    )
    snapshot = copy.deepcopy(dict(attrs))
    intended = {
        ACQUISITION_CAMERA_FRAME_ATTR: record.to_dict(),
        ACQUISITION_CAMERA_FRAME_DIGEST_ATTR: record.digest(),
    }
    expected = _expected_attrs_after_update(snapshot, intended)
    try:
        attrs.update(copy.deepcopy(intended))
        _require_exact_attrs_state(
            attrs,
            expected,
            label="Acquisition camera post-write",
        )
        bound = load_acquisition_camera_frame(
            root_node,
            authority_node,
            import_ownership=ownership,
        )
        _require_exact_attrs_state(
            require_trusted_coordinate_attrs(
                authority_node,
                label="Reloaded acquisition camera",
            ),
            expected,
            label="Acquisition camera post-reload",
        )
        return bound
    except Exception as exc:
        try:
            _restore_exact_attrs(attrs, snapshot)
        except Exception as rollback_exc:  # pragma: no cover
            raise PixelFrameAuthorityError(
                f"Acquisition-camera stamp failed and rollback was incomplete: {rollback_exc}"
            ) from exc
        if isinstance(exc, PixelFrameAuthorityError):
            raise
        raise PixelFrameAuthorityError(
            f"Acquisition-camera stamp failed: {exc}"
        ) from exc


def require_bound_acquisition_camera_frame(value: Any) -> BoundAcquisitionCameraFrame:
    if (
        type(value) is not BoundAcquisitionCameraFrame
        or value._seal is not _BOUND_ACQUISITION_FRAME_SEAL
    ):
        raise PixelFrameAuthorityError(
            "A sealed acquisition-owned camera frame is required."
        )
    verify_persisted_proof(_bound_authority_proof_key(value), value.assert_verified)
    return value


PixelFrameReference = BoundReferenceExtent | BoundAcquisitionCameraFrame


def _extent_pointer(extent: PixelFrameReference) -> dict[str, Any]:
    return {
        "record_ref": extent.record_ref,
        "record_sha256": extent.record_sha256,
        "selector": extent.selector,
        "width": extent.width,
        "height": extent.height,
        "units": extent.units,
    }


def _identity_pointer(identity: BoundRowIdentityContract) -> dict[str, Any]:
    return {
        "record_ref": identity.record_ref,
        "record_sha256": identity.record_sha256,
        "leading_dimension": identity.leading_dimension,
    }


@dataclass(frozen=True)
class PixelFrameEndpoint:
    """Transform-facing endpoint derived only from a sealed frame authority."""

    space_id: str
    width: int
    height: int
    units: str
    pixel_convention: str
    record_ref: str
    record_sha256: str
    selector: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "space_id": self.space_id,
            "width": self.width,
            "height": self.height,
            "units": self.units,
            "pixel_convention": self.pixel_convention,
            "authority": {
                "record_ref": self.record_ref,
                "record_sha256": self.record_sha256,
                "selector": self.selector,
            },
        }


@dataclass(frozen=True)
class PixelFrameRecord:
    frame_id: str
    kind: str
    space_id: str
    coordinate_units: str
    pixel_convention: str
    reference_extent: Mapping[str, Any]
    lineage: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": PIXEL_FRAME_AUTHORITY_SCHEMA_ID,
            "schema_version": PIXEL_FRAME_AUTHORITY_SCHEMA_VERSION,
            "frame_id": self.frame_id,
            "kind": self.kind,
            "space_id": self.space_id,
            "coordinate_units": self.coordinate_units,
            "pixel_convention": self.pixel_convention,
            "reference_extent": copy.deepcopy(dict(self.reference_extent)),
            "lineage": copy.deepcopy(dict(self.lineage)),
            "canonicalization": PIXEL_FRAME_AUTHORITY_CANONICALIZATION,
        }

    def digest(self) -> str:
        return _mapping_sha256(self.to_dict())


@dataclass(frozen=True, init=False)
class BoundPixelFrameAuthority:
    """Sealed proof for one persisted, controlled pixel frame."""

    record: PixelFrameRecord
    authority_path: str
    record_ref: str
    record_sha256: str
    endpoint: PixelFrameEndpoint
    reference_extent: PixelFrameReference = field(repr=False, compare=False)
    row_identity: BoundRowIdentityContract | None = field(repr=False, compare=False)
    _archive_identity: ArchiveIdentity = field(repr=False, compare=False)
    _authority_node: Any = field(repr=False, compare=False)
    _context: Mapping[str, Any] = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        record: PixelFrameRecord,
        authority_path: str,
        reference_extent: PixelFrameReference,
        row_identity: BoundRowIdentityContract | None,
        archive: ArchiveIdentity,
        authority_node: Any,
        context: Mapping[str, Any],
        _verification_seal: object | None = None,
    ) -> None:
        if _verification_seal is not _BOUND_PIXEL_FRAME_SEAL:
            raise PixelFrameAuthorityError(
                "Bound pixel-frame authorities cannot be constructed directly."
            )
        record_ref = f"/{authority_path}@{PIXEL_FRAME_AUTHORITY_ATTR}"
        record_sha256 = record.digest()
        object.__setattr__(self, "record", record)
        object.__setattr__(self, "authority_path", authority_path)
        object.__setattr__(self, "record_ref", record_ref)
        object.__setattr__(self, "record_sha256", record_sha256)
        object.__setattr__(
            self,
            "endpoint",
            PixelFrameEndpoint(
                space_id=record.space_id,
                width=int(reference_extent.width),
                height=int(reference_extent.height),
                units=record.coordinate_units,
                pixel_convention=record.pixel_convention,
                record_ref=record_ref,
                record_sha256=record_sha256,
                selector=PIXEL_FRAME_AUTHORITY_ATTR,
            ),
        )
        object.__setattr__(self, "reference_extent", reference_extent)
        object.__setattr__(self, "row_identity", row_identity)
        object.__setattr__(self, "_archive_identity", archive)
        object.__setattr__(self, "_authority_node", authority_node)
        object.__setattr__(self, "_context", context)
        object.__setattr__(self, "_seal", _verification_seal)

    @property
    def archive_identity(self) -> ArchiveIdentity:
        return self._archive_identity

    @property
    def space_id(self) -> str:
        return self.record.space_id

    @property
    def pixel_convention(self) -> str:
        return self.record.pixel_convention

    def assert_verified(self) -> None:
        if self._seal is not _BOUND_PIXEL_FRAME_SEAL:
            raise PixelFrameAuthorityError(
                "Pixel-frame authority is not sealed evidence."
            )
        current = _load_by_kind(
            self.record.kind,
            self._authority_node,
            reference_extent=self.reference_extent,
            context=self._context,
        )
        if (
            current.record != self.record
            or current.record_ref != self.record_ref
            or current.record_sha256 != self.record_sha256
            or current.archive_identity != self.archive_identity
        ):
            raise PixelFrameAuthorityError(
                "Persisted pixel-frame authority changed after it was bound."
            )


def _parse_extent(value: Any) -> dict[str, Any]:
    payload = _exact_fields(
        value,
        expected=frozenset(
            {"record_ref", "record_sha256", "selector", "width", "height", "units"}
        ),
        field_name="reference_extent",
    )
    if payload["units"] != "px":
        raise PixelFrameAuthorityError("Pixel-frame reference units must be 'px'.")
    result = {
        "record_ref": _required_text(
            payload["record_ref"], field_name="reference_extent.record_ref"
        ),
        "record_sha256": _sha256(
            payload["record_sha256"], field_name="reference_extent.record_sha256"
        ),
        "selector": _required_text(
            payload["selector"], field_name="reference_extent.selector"
        ),
        "width": payload["width"],
        "height": payload["height"],
        "units": "px",
    }
    if not result["record_ref"].startswith("/"):
        raise PixelFrameAuthorityError("reference_extent.record_ref must be absolute.")
    for name in ("width", "height"):
        number = result[name]
        if type(number) is not int or number <= 0:
            raise PixelFrameAuthorityError(
                f"reference_extent.{name} must be an exact positive integer."
            )
    return result


def parse_pixel_frame_record(value: Any) -> PixelFrameRecord:
    if isinstance(value, PixelFrameRecord):
        value = value.to_dict()
    payload = _exact_fields(
        value,
        expected=frozenset(
            {
                "schema_id",
                "schema_version",
                "frame_id",
                "kind",
                "space_id",
                "coordinate_units",
                "pixel_convention",
                "reference_extent",
                "lineage",
                "canonicalization",
            }
        ),
        field_name="pixel_frame_authority",
    )
    if payload["schema_id"] != PIXEL_FRAME_AUTHORITY_SCHEMA_ID:
        raise PixelFrameAuthorityError("Unsupported pixel-frame schema_id.")
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != PIXEL_FRAME_AUTHORITY_SCHEMA_VERSION
    ):
        raise PixelFrameAuthorityError("Unsupported pixel-frame schema_version.")
    if payload["canonicalization"] != PIXEL_FRAME_AUTHORITY_CANONICALIZATION:
        raise PixelFrameAuthorityError("Unsupported pixel-frame canonicalization.")
    frame_id = _required_text(payload["frame_id"], field_name="frame_id")
    if _FRAME_ID_RE.fullmatch(frame_id) is None:
        raise PixelFrameAuthorityError("frame_id is not canonical.")
    kind = _required_text(payload["kind"], field_name="kind")
    expected_spaces = {
        SOURCE_CAMERA_FRAME_KIND: SOURCE_CAMERA_IMAGE_SPACE_ID,
        SELECTED_CANVAS_FRAME_KIND: STIMULUS_CANVAS_SPACE_ID,
        ARENA_RELATIVE_CANVAS_FRAME_KIND: ARENA_RELATIVE_CANVAS_SPACE_ID,
        ROI_FRAME_KIND: ROI_LOCAL_SPACE_ID,
        MODEL_INPUT_FRAME_KIND: DETECTOR_MODEL_INPUT_SPACE_ID,
        SOURCE_CAMERA_NORMALIZED_FRAME_KIND: SOURCE_CAMERA_NORMALIZED_SPACE_ID,
        DETECTOR_NORMALIZED_FRAME_KIND: DETECTOR_NORMALIZED_SPACE_ID,
    }
    if kind not in expected_spaces:
        raise PixelFrameAuthorityError(f"Unsupported pixel-frame kind {kind!r}.")
    if payload["space_id"] != expected_spaces[kind]:
        raise PixelFrameAuthorityError(
            "Pixel-frame kind and controlled space disagree."
        )
    expected_units = (
        "normalized"
        if kind
        in {
            SOURCE_CAMERA_NORMALIZED_FRAME_KIND,
            DETECTOR_NORMALIZED_FRAME_KIND,
        }
        else "px"
    )
    if payload["coordinate_units"] != expected_units:
        raise PixelFrameAuthorityError(
            "Pixel-frame kind and controlled coordinate units disagree."
        )
    convention = _required_text(
        payload["pixel_convention"], field_name="pixel_convention"
    )
    if convention not in _SUPPORTED_PIXEL_CONVENTIONS:
        raise PixelFrameAuthorityError(f"Unsupported pixel convention {convention!r}.")
    if not isinstance(payload["lineage"], Mapping):
        raise PixelFrameAuthorityError("lineage must be a mapping.")
    # Canonical JSON round-trip rejects non-JSON and preserves an inert copy.
    lineage = json.loads(_canonical_json(payload["lineage"]))
    return PixelFrameRecord(
        frame_id=frame_id,
        kind=kind,
        space_id=expected_spaces[kind],
        coordinate_units=expected_units,
        pixel_convention=convention,
        reference_extent=_parse_extent(payload["reference_extent"]),
        lineage=lineage,
    )


def _same_frame(
    left: BoundPixelFrameAuthority, right: BoundPixelFrameAuthority
) -> bool:
    return (
        left.archive_identity == right.archive_identity
        and left.record_ref == right.record_ref
        and left.record_sha256 == right.record_sha256
        and left.endpoint == right.endpoint
    )


def _require_archive(
    authority_node: Any,
    extent: PixelFrameReference,
    *nodes: Any,
    require_colocated: bool = True,
) -> ArchiveIdentity:
    try:
        if type(extent) is BoundAcquisitionCameraFrame:
            extent = require_bound_acquisition_camera_frame(extent)
        else:
            extent = verify_bound_reference_extent(extent)
        common = require_same_archive(
            authority_node,
            extent._authority_node,
            *nodes,
        )
    except (
        ArchiveIdentityError,
        CoordinateReferenceError,
        PixelFrameAuthorityError,
    ) as exc:
        raise PixelFrameAuthorityError(str(exc)) from exc
    if common != extent.archive_identity:
        raise PixelFrameAuthorityError(
            "Pixel-frame evidence comes from different archives/stores."
        )
    if require_colocated and canonical_node_path(authority_node) != canonical_node_path(
        extent._authority_node
    ):
        raise PixelFrameAuthorityError(
            "Pixel-frame attrs must be persisted on the exact extent-authority node."
        )
    return common


def _record(
    *,
    frame_id: str,
    kind: str,
    space_id: str,
    pixel_convention: str,
    reference_extent: PixelFrameReference,
    lineage: Mapping[str, Any],
    coordinate_units: str = "px",
) -> PixelFrameRecord:
    return parse_pixel_frame_record(
        {
            "schema_id": PIXEL_FRAME_AUTHORITY_SCHEMA_ID,
            "schema_version": PIXEL_FRAME_AUTHORITY_SCHEMA_VERSION,
            "frame_id": frame_id,
            "kind": kind,
            "space_id": space_id,
            "coordinate_units": coordinate_units,
            "pixel_convention": pixel_convention,
            "reference_extent": _extent_pointer(reference_extent),
            "lineage": copy.deepcopy(dict(lineage)),
            "canonicalization": PIXEL_FRAME_AUTHORITY_CANONICALIZATION,
        }
    )


def _raw_equals_canonical(raw: Any, parsed: PixelFrameRecord) -> bool:
    return isinstance(raw, Mapping) and _exact_json_equal(raw, parsed.to_dict())


def _load_record(node: Any) -> PixelFrameRecord:
    attrs = getattr(node, "attrs", None)
    if not isinstance(attrs, Mapping):
        raise PixelFrameAuthorityError("Pixel-frame node must expose persisted attrs.")
    if PIXEL_FRAME_AUTHORITY_ATTR not in attrs:
        raise PixelFrameAuthorityError("Persisted pixel-frame authority is missing.")
    raw = attrs[PIXEL_FRAME_AUTHORITY_ATTR]
    record = parse_pixel_frame_record(raw)
    if not _raw_equals_canonical(raw, record):
        raise PixelFrameAuthorityError(
            "Raw persisted pixel-frame mapping is not its parsed canonical form."
        )
    stored = _sha256(
        attrs.get(PIXEL_FRAME_AUTHORITY_DIGEST_ATTR),
        field_name=PIXEL_FRAME_AUTHORITY_DIGEST_ATTR,
    )
    if stored != record.digest():
        raise PixelFrameAuthorityError(
            "Persisted pixel-frame digest does not match canonical content."
        )
    return record


def _validate_common(
    record: PixelFrameRecord,
    *,
    expected_kind: str,
    reference_extent: PixelFrameReference,
) -> None:
    if record.kind != expected_kind:
        raise PixelFrameAuthorityError(
            f"Expected {expected_kind!r} pixel-frame authority, found {record.kind!r}."
        )
    try:
        if type(reference_extent) is BoundAcquisitionCameraFrame:
            extent = require_bound_acquisition_camera_frame(reference_extent)
        else:
            extent = verify_bound_reference_extent(reference_extent)
    except (CoordinateReferenceError, PixelFrameAuthorityError) as exc:
        raise PixelFrameAuthorityError(f"Reference authority is stale: {exc}") from exc
    if extent.units != "px":
        raise PixelFrameAuthorityError("Pixel-frame extent must use px units.")
    if record.reference_extent != _extent_pointer(extent):
        raise PixelFrameAuthorityError(
            "Pixel-frame record does not bind the exact live reference extent."
        )


def _source_camera_lineage(
    acquisition_frame: BoundAcquisitionCameraFrame,
) -> dict[str, Any]:
    verified = require_bound_acquisition_camera_frame(acquisition_frame)
    return {
        "acquisition_camera_frame": {
            "record_ref": verified.record_ref,
            "record_sha256": verified.record_sha256,
            "selector": verified.selector,
        },
        "acquisition_import_ownership": copy.deepcopy(
            dict(verified.record.import_ownership)
        ),
        "recording_id": verified.record.recording_id,
        "camera_id": verified.record.camera_id,
    }


def _selected_canvas_lineage(snapshot: Any) -> dict[str, Any]:
    from fisheye.shared.selected_calibration import (
        require_bound_selected_calibration_snapshot,
    )

    selected = require_bound_selected_calibration_snapshot(snapshot)
    return {
        "selected_calibration_manifest": {
            "record_ref": selected.manifest_record_ref,
            "record_sha256": selected.manifest_sha256,
        },
        "source_display": {
            "record_ref": selected.display_record_ref,
            "record_sha256": selected.manifest.source_display.digest(),
        },
        "stimulus_run": selected.stimulus_run,
        "camera_id": selected.camera_id,
        "external_h5_freshness": "persisted_import_snapshot",
    }


def _validate_source_camera(
    record: PixelFrameRecord,
    *,
    acquisition_frame: BoundAcquisitionCameraFrame,
) -> None:
    acquisition = require_bound_acquisition_camera_frame(acquisition_frame)
    _validate_common(
        record,
        expected_kind=SOURCE_CAMERA_FRAME_KIND,
        reference_extent=acquisition,
    )
    if record.lineage != _source_camera_lineage(acquisition):
        raise PixelFrameAuthorityError(
            "Source-camera frame does not match exact acquisition-owned lineage."
        )
    if (
        record.reference_extent["width"] != acquisition.record.width_px
        or record.reference_extent["height"] != acquisition.record.height_px
    ):
        raise PixelFrameAuthorityError(
            "Source-camera frame dimensions conflict with acquisition metadata."
        )


def _source_camera_authority_path(
    acquisition_frame: BoundAcquisitionCameraFrame,
    pixel_convention: str,
) -> str:
    acquisition = require_bound_acquisition_camera_frame(acquisition_frame)
    convention = _required_text(pixel_convention, field_name="pixel_convention")
    if convention not in _SUPPORTED_PIXEL_CONVENTIONS:
        raise PixelFrameAuthorityError(
            f"Unsupported source-camera pixel convention {convention!r}."
        )
    return (
        "analysis/coordinate_frames/source_camera/"
        f"{acquisition.record.camera_id}/{convention}"
    )


def _require_source_camera_authority_path(
    node: Any,
    *,
    acquisition_frame: BoundAcquisitionCameraFrame,
    pixel_convention: str,
) -> None:
    expected = _source_camera_authority_path(
        acquisition_frame,
        pixel_convention,
    )
    if canonical_node_path(node) != expected:
        raise PixelFrameAuthorityError(
            "Source-camera authority must be persisted at the exact derived path "
            f"{expected!r}; camera/convention aliases are forbidden."
        )


def _validate_selected_canvas(
    record: PixelFrameRecord,
    *,
    reference_extent: BoundReferenceExtent,
    selected_calibration_snapshot: Any,
) -> None:
    from fisheye.shared.selected_calibration import (
        require_bound_selected_calibration_snapshot,
    )

    selected = require_bound_selected_calibration_snapshot(
        selected_calibration_snapshot
    )
    _validate_common(
        record,
        expected_kind=SELECTED_CANVAS_FRAME_KIND,
        reference_extent=reference_extent,
    )
    lineage = _selected_canvas_lineage(selected)
    if record.lineage != lineage:
        raise PixelFrameAuthorityError(
            "Canvas frame does not match exact selected-display evidence."
        )
    if (
        record.reference_extent["width"] != selected.target_reference_extent.width
        or record.reference_extent["height"] != selected.target_reference_extent.height
    ):
        raise PixelFrameAuthorityError(
            "Canvas dimensions conflict with selected-display evidence."
        )


def _arena_relative_canvas_lineage(
    geometry_node: Any,
    selected_canvas_frame: BoundPixelFrameAuthority,
) -> dict[str, Any]:
    selected = require_selected_canvas_pixel_frame_authority(selected_canvas_frame)
    values = _array_values(geometry_node)
    if values.shape != (4,) or not np.issubdtype(values.dtype, np.integer):
        raise PixelFrameAuthorityError(
            "Arena placement must be one exact integer (x,y,width,height) array."
        )
    x, y, width, height = (int(item) for item in values.tolist())
    if (
        x < 0
        or y < 0
        or width <= 0
        or height <= 0
        or x + width > selected.endpoint.width
        or y + height > selected.endpoint.height
    ):
        raise PixelFrameAuthorityError(
            "Arena placement must be positive and contained in the exact selected canvas."
        )
    return {
        "arena_geometry": _array_pointer(geometry_node),
        "layout": "selected_canvas_xywh_px",
        "origin": "arena_top_left",
        "origin_in_selected_canvas_px": {"x": x, "y": y},
        "selected_canvas_frame": {
            "record_ref": selected.record_ref,
            "record_sha256": selected.record_sha256,
        },
    }


def _validate_arena_relative_canvas(
    record: PixelFrameRecord,
    *,
    reference_extent: BoundReferenceExtent,
    geometry_node: Any,
    selected_canvas_frame: BoundPixelFrameAuthority,
) -> None:
    _validate_common(
        record,
        expected_kind=ARENA_RELATIVE_CANVAS_FRAME_KIND,
        reference_extent=reference_extent,
    )
    selected = require_selected_canvas_pixel_frame_authority(selected_canvas_frame)
    lineage = _arena_relative_canvas_lineage(geometry_node, selected)
    if record.lineage != lineage:
        raise PixelFrameAuthorityError(
            "Arena-relative canvas lineage changed or is incomplete."
        )
    if record.pixel_convention != selected.pixel_convention:
        raise PixelFrameAuthorityError(
            "Arena-relative and selected-canvas frames must use one exact pixel convention."
        )
    geometry = _array_values(geometry_node)
    if record.reference_extent["width"] != int(geometry[2]) or record.reference_extent[
        "height"
    ] != int(geometry[3]):
        raise PixelFrameAuthorityError(
            "Arena-relative extent conflicts with exact persisted arena geometry."
        )


@dataclass(frozen=True)
class CropPlacementOwnershipRecord:
    producer: str
    layout: str
    window_policy: str
    crop_placement: Mapping[str, Any]
    row_identity: Mapping[str, Any]
    source_camera_frame: Mapping[str, Any]
    camera_id: str
    schema_id: str = CROP_PLACEMENT_OWNERSHIP_SCHEMA_ID
    schema_version: int = CROP_PLACEMENT_OWNERSHIP_SCHEMA_VERSION
    window_geometry: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "producer": self.producer,
            "layout": self.layout,
            "window_policy": self.window_policy,
            "crop_placement": copy.deepcopy(dict(self.crop_placement)),
            "row_identity": copy.deepcopy(dict(self.row_identity)),
            "source_camera_frame": copy.deepcopy(dict(self.source_camera_frame)),
            "camera_id": self.camera_id,
            "canonicalization": PIXEL_FRAME_AUTHORITY_CANONICALIZATION,
        }
        if self.window_geometry is not None:
            result["window_geometry"] = copy.deepcopy(dict(self.window_geometry))
        return result

    def digest(self) -> str:
        return _mapping_sha256(self.to_dict())


@dataclass(frozen=True, init=False)
class BoundCropPlacementOwnership:
    record: CropPlacementOwnershipRecord
    record_ref: str
    record_sha256: str
    attr_name: str
    row_identity: BoundRowIdentityContract = field(repr=False, compare=False)
    source_camera_frame: BoundPixelFrameAuthority = field(repr=False, compare=False)
    _archive_identity: ArchiveIdentity = field(repr=False, compare=False)
    _placement_node: Any = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        record: CropPlacementOwnershipRecord,
        row_identity: BoundRowIdentityContract,
        source_camera_frame: BoundPixelFrameAuthority,
        archive: ArchiveIdentity,
        placement_node: Any,
        attr_name: str,
        _verification_seal: object | None = None,
    ) -> None:
        if _verification_seal is not _BOUND_CROP_PLACEMENT_OWNERSHIP_SEAL:
            raise PixelFrameAuthorityError(
                "Crop placement ownership cannot be constructed directly."
            )
        object.__setattr__(self, "record", record)
        object.__setattr__(
            self,
            "record_ref",
            f"/{canonical_node_path(placement_node)}@{attr_name}",
        )
        object.__setattr__(self, "record_sha256", record.digest())
        object.__setattr__(self, "attr_name", attr_name)
        object.__setattr__(self, "row_identity", row_identity)
        object.__setattr__(self, "source_camera_frame", source_camera_frame)
        object.__setattr__(self, "_archive_identity", archive)
        object.__setattr__(self, "_placement_node", placement_node)
        object.__setattr__(self, "_seal", _verification_seal)

    @property
    def archive_identity(self) -> ArchiveIdentity:
        return self._archive_identity

    def assert_verified(self) -> None:
        if self._seal is not _BOUND_CROP_PLACEMENT_OWNERSHIP_SEAL:
            raise PixelFrameAuthorityError(
                "Crop placement ownership is not sealed evidence."
            )
        current = load_crop_placement_ownership(
            self._placement_node,
            row_identity=self.row_identity,
            source_camera_frame=self.source_camera_frame,
            attr_name=self.attr_name,
        )
        if (
            current.record != self.record
            or current.record_ref != self.record_ref
            or current.record_sha256 != self.record_sha256
            or current.archive_identity != self.archive_identity
        ):
            raise PixelFrameAuthorityError(
                "Persisted crop placement ownership changed after binding."
            )


def _crop_placement_values(
    placement_node: Any,
    row_identity: BoundRowIdentityContract,
    source_camera_frame: BoundPixelFrameAuthority,
    *,
    allow_zero_padded: bool = False,
) -> np.ndarray:
    values = _array_values(placement_node)
    # ``np.number`` includes complex, and bool can be coerced to float.  Crop
    # geometry admits only persisted integer/unsigned/real floating payloads.
    if values.dtype.kind not in {"i", "u", "f"}:
        raise PixelFrameAuthorityError(
            "Crop placement must use an integer or real floating dtype."
        )
    numeric = values.astype(np.float64)
    if (
        numeric.ndim != 2
        or numeric.shape[1:] != (4,)
        or not np.isfinite(numeric).all()
        or np.any(numeric[:, 2:] <= 0)
        or numeric.shape[0] != row_identity.leading_dimension
    ):
        raise PixelFrameAuthorityError(
            "Crop placement must be finite row-aligned (N,4) xywh with positive extents."
        )
    placement_path = canonical_node_path(placement_node)
    if placement_path != f"{row_identity.rowset_path}/source_crop_xywh":
        raise PixelFrameAuthorityError(
            "Crop placement must use the crop writer's canonical source_crop_xywh path."
        )
    camera = require_source_camera_pixel_frame_authority(source_camera_frame)
    if not allow_zero_padded and (
        np.any(numeric[:, :2] < 0)
        or np.any(numeric[:, 0] + numeric[:, 2] > camera.endpoint.width)
        or np.any(numeric[:, 1] + numeric[:, 3] > camera.endpoint.height)
    ):
        raise PixelFrameAuthorityError(
            "Canonical crop placement records only contained actual source-camera windows; negative/out-of-frame padded windows require a separate explicit lineage contract."
        )
    if allow_zero_padded:
        if not np.equal(numeric, np.floor(numeric)).all():
            raise PixelFrameAuthorityError(
                "Historical padded crop placement must use exact integer-valued xywh coordinates."
            )
        requested_width = numeric[:, 2]
        requested_height = numeric[:, 3]
        if (
            np.any(requested_width != requested_width[0])
            or np.any(requested_height != requested_height[0])
            or requested_width[0] <= 0
            or requested_height[0] <= 0
        ):
            raise PixelFrameAuthorityError(
                "Historical padded crop placement must use one positive fixed requested extent."
            )
        clipped_left = np.maximum(numeric[:, 0], 0.0)
        clipped_top = np.maximum(numeric[:, 1], 0.0)
        clipped_right = np.minimum(
            numeric[:, 0] + numeric[:, 2], float(camera.endpoint.width)
        )
        clipped_bottom = np.minimum(
            numeric[:, 1] + numeric[:, 3], float(camera.endpoint.height)
        )
        if np.any(clipped_right <= clipped_left) or np.any(
            clipped_bottom <= clipped_top
        ):
            raise PixelFrameAuthorityError(
                "Historical padded crop placement must retain a positive source-camera intersection."
            )
    return values


def _crop_ownership_record(
    placement_node: Any,
    row_identity: BoundRowIdentityContract,
    source_camera_frame: BoundPixelFrameAuthority,
    *,
    allow_zero_padded: bool = False,
    padded_producer: str = CROP_PLACEMENT_PADDED_PRODUCER,
) -> CropPlacementOwnershipRecord:
    try:
        row_identity.assert_verified()
    except RowIdentityContractError as exc:
        raise PixelFrameAuthorityError(f"Crop row identity is stale: {exc}") from exc
    camera = require_source_camera_pixel_frame_authority(source_camera_frame)
    if (
        row_identity.contract.domain != OBSERVATION_INSTANCE_DOMAIN
        or row_identity.contract.mode != INSTANCE_KEY_MODE
        or row_identity.contract.key_array.ref != INSTANCE_KEY_ARRAY_REF
    ):
        raise PixelFrameAuthorityError(
            "Crop ownership requires exact observation instance_key identity."
        )
    values = _crop_placement_values(
        placement_node,
        row_identity,
        camera,
        allow_zero_padded=allow_zero_padded,
    )
    window_geometry: dict[str, Any] | None = None
    producer = CROP_PLACEMENT_PRODUCER
    window_policy = CROP_PLACEMENT_WINDOW_POLICY
    schema_id = CROP_PLACEMENT_OWNERSHIP_SCHEMA_ID
    schema_version = CROP_PLACEMENT_OWNERSHIP_SCHEMA_VERSION
    if allow_zero_padded:
        if padded_producer not in {
            CROP_PLACEMENT_PADDED_PRODUCER,
            CROP_PLACEMENT_PADDED_LEGACY_PRODUCER,
        }:
            raise PixelFrameAuthorityError(
                "Unsupported padded crop placement producer."
            )
        attrs = getattr(placement_node, "attrs", None)
        provenance = (
            attrs.get(CROP_PLACEMENT_PADDED_PROVENANCE_ATTR)
            if isinstance(attrs, Mapping)
            else None
        )
        if not isinstance(provenance, Mapping):
            raise PixelFrameAuthorityError(
                "Padded crop placement requires its separately persisted successor provenance record."
            )
        provenance_digest = _mapping_sha256(provenance)
        if attrs.get(CROP_PLACEMENT_PADDED_PROVENANCE_DIGEST_ATTR) != provenance_digest:
            raise PixelFrameAuthorityError(
                "Padded crop placement provenance digest is missing or stale."
            )
        provenance_fields = {
            "schema_id",
            "schema_version",
            "crop_policy_payload_digest",
            "origin_authority_digest",
            "provider_record_sha256",
        }
        if (
            set(provenance) != provenance_fields
            or provenance.get("schema_id") != CROP_PLACEMENT_PADDED_PROVENANCE_SCHEMA_ID
            or provenance.get("schema_version")
            != CROP_PLACEMENT_PADDED_PROVENANCE_SCHEMA_VERSION
        ):
            raise PixelFrameAuthorityError(
                "Padded crop placement provenance has an invalid explicit field set."
            )
        for name in (
            "crop_policy_payload_digest",
            "origin_authority_digest",
            "provider_record_sha256",
        ):
            _sha256(provenance.get(name), field_name=name)
        numeric = values.astype(np.float64, copy=False)
        width = int(camera.endpoint.width)
        height = int(camera.endpoint.height)
        requested = np.rint(numeric).astype(np.int64)
        clipped_x0 = np.maximum(requested[:, 0], 0)
        clipped_y0 = np.maximum(requested[:, 1], 0)
        clipped_x1 = np.minimum(requested[:, 0] + requested[:, 2], width)
        clipped_y1 = np.minimum(requested[:, 1] + requested[:, 3], height)
        left = np.maximum(0, -requested[:, 0])
        top = np.maximum(0, -requested[:, 1])
        right = np.maximum(0, requested[:, 0] + requested[:, 2] - width)
        bottom = np.maximum(0, requested[:, 1] + requested[:, 3] - height)
        clipped = np.column_stack(
            (clipped_x0, clipped_y0, clipped_x1 - clipped_x0, clipped_y1 - clipped_y0)
        ).astype("<i8", copy=False)
        padding = np.column_stack((left, top, right, bottom)).astype("<i8", copy=False)
        padded_rows = np.any(padding != 0, axis=1)
        window_geometry = {
            "schema_id": CROP_PLACEMENT_PADDED_WINDOW_GEOMETRY_SCHEMA_ID,
            "schema_version": 1,
            "requested_extent": {
                "width": int(requested[0, 2]),
                "height": int(requested[0, 3]),
                "units": "px",
            },
            "source_camera_extent": {
                "width": width,
                "height": height,
                "units": "px",
            },
            "clipped_source_camera_intersection": {
                "formula": "x0=max(requested_x,0); y0=max(requested_y,0); x1=min(requested_x+width,W); y1=min(requested_y+height,H)",
                "dtype": clipped.dtype.str,
                "shape": [int(value) for value in clipped.shape],
                "sha256": array_values_sha256(clipped),
            },
            "zero_padding_offsets_ltrb": {
                "formula": "left=max(0,-x); top=max(0,-y); right=max(0,x+width-W); bottom=max(0,y+height-H)",
                "dtype": padding.dtype.str,
                "shape": [int(value) for value in padding.shape],
                "sha256": array_values_sha256(padding),
            },
            "padded_row_count": int(np.count_nonzero(padded_rows)),
            "padded_row_fraction": float(np.mean(padded_rows))
            if padding.shape[0]
            else 0.0,
            "max_padding_ltrb": [
                int(value) for value in padding.max(axis=0, initial=0)
            ],
            "crop_local_to_source_camera": {
                "formula": "source_xy = requested_origin_xy + crop_local_xy",
                "source_pixel_authority": "clipped_source_camera_intersection_only",
                "source_pixels_outside_extent": "synthetic_zero_padding_no_source_pixel_correspondence",
            },
            "crop_policy_provenance": copy.deepcopy(dict(provenance)),
        }
        producer = padded_producer
        window_policy = CROP_PLACEMENT_PADDED_WINDOW_POLICY
        schema_id = CROP_PLACEMENT_PADDED_OWNERSHIP_SCHEMA_ID
        schema_version = CROP_PLACEMENT_PADDED_OWNERSHIP_SCHEMA_VERSION
    return CropPlacementOwnershipRecord(
        producer=producer,
        layout="xywh",
        window_policy=window_policy,
        crop_placement=_array_pointer(placement_node),
        row_identity=_identity_pointer(row_identity),
        source_camera_frame={
            "record_ref": camera.record_ref,
            "record_sha256": camera.record_sha256,
        },
        camera_id=camera.record.lineage["camera_id"],
        schema_id=schema_id,
        schema_version=schema_version,
        window_geometry=window_geometry,
    )


def parse_crop_placement_ownership(value: Any) -> CropPlacementOwnershipRecord:
    if isinstance(value, CropPlacementOwnershipRecord):
        value = value.to_dict()
    base_fields = frozenset(
        {
            "schema_id",
            "schema_version",
            "producer",
            "layout",
            "window_policy",
            "crop_placement",
            "row_identity",
            "source_camera_frame",
            "camera_id",
            "canonicalization",
        }
    )
    if not isinstance(value, Mapping):
        raise PixelFrameAuthorityError("Crop ownership must be a mapping.")
    schema_id = value.get("schema_id")
    if schema_id == CROP_PLACEMENT_OWNERSHIP_SCHEMA_ID:
        payload = _exact_fields(
            value, expected=base_fields, field_name="crop_placement_ownership"
        )
        if payload["schema_version"] != CROP_PLACEMENT_OWNERSHIP_SCHEMA_VERSION:
            raise PixelFrameAuthorityError("Unsupported crop-ownership schema_version.")
        ordinary = (
            payload["producer"] == CROP_PLACEMENT_PRODUCER
            and payload["layout"] == "xywh"
            and payload["window_policy"] == CROP_PLACEMENT_WINDOW_POLICY
        )
        if not ordinary:
            raise PixelFrameAuthorityError(
                "Unsupported ordinary crop ownership producer/layout."
            )
        padded = False
    elif schema_id == CROP_PLACEMENT_PADDED_OWNERSHIP_SCHEMA_ID:
        payload = _exact_fields(
            value,
            expected=base_fields | {"window_geometry"},
            field_name="padded_crop_placement_ownership",
        )
        if payload["schema_version"] != CROP_PLACEMENT_PADDED_OWNERSHIP_SCHEMA_VERSION:
            raise PixelFrameAuthorityError(
                "Unsupported padded crop-ownership schema_version."
            )
        padded = True
        if not (
            payload["producer"]
            in {
                CROP_PLACEMENT_PADDED_PRODUCER,
                CROP_PLACEMENT_PADDED_LEGACY_PRODUCER,
            }
            and payload["layout"] == "xywh"
            and payload["window_policy"] == CROP_PLACEMENT_PADDED_WINDOW_POLICY
        ):
            raise PixelFrameAuthorityError(
                "Unsupported padded crop ownership producer/layout."
            )
    else:
        raise PixelFrameAuthorityError("Unsupported crop-ownership schema_id.")
    if payload["canonicalization"] != PIXEL_FRAME_AUTHORITY_CANONICALIZATION:
        raise PixelFrameAuthorityError("Unsupported crop ownership canonicalization.")
    for name in ("crop_placement", "row_identity", "source_camera_frame"):
        if not isinstance(payload[name], Mapping):
            raise PixelFrameAuthorityError(f"{name} must be an exact pointer mapping.")
    if padded:
        geometry = payload.get("window_geometry")
        if not isinstance(geometry, Mapping):
            raise PixelFrameAuthorityError(
                "Padded crop ownership requires its explicit window_geometry record."
            )
        expected_geometry_fields = {
            "schema_id",
            "schema_version",
            "requested_extent",
            "source_camera_extent",
            "clipped_source_camera_intersection",
            "zero_padding_offsets_ltrb",
            "padded_row_count",
            "padded_row_fraction",
            "max_padding_ltrb",
            "crop_local_to_source_camera",
            "crop_policy_provenance",
        }
        if set(geometry) != expected_geometry_fields or (
            geometry.get("schema_id") != CROP_PLACEMENT_PADDED_WINDOW_GEOMETRY_SCHEMA_ID
            or geometry.get("schema_version") != 1
            or geometry.get("crop_local_to_source_camera", {}).get(
                "source_pixels_outside_extent"
            )
            != "synthetic_zero_padding_no_source_pixel_correspondence"
        ):
            raise PixelFrameAuthorityError(
                "Padded crop ownership has an invalid window_geometry contract."
            )
        provenance = geometry.get("crop_policy_provenance")
        if not isinstance(provenance, Mapping) or set(provenance) != {
            "schema_id",
            "schema_version",
            "crop_policy_payload_digest",
            "origin_authority_digest",
            "provider_record_sha256",
        }:
            raise PixelFrameAuthorityError(
                "Padded crop ownership lacks its exact crop-policy provenance."
            )
        if (
            provenance.get("schema_id") != CROP_PLACEMENT_PADDED_PROVENANCE_SCHEMA_ID
            or provenance.get("schema_version")
            != CROP_PLACEMENT_PADDED_PROVENANCE_SCHEMA_VERSION
        ):
            raise PixelFrameAuthorityError(
                "Padded crop ownership has an invalid crop-policy provenance identity."
            )
        for name in (
            "crop_policy_payload_digest",
            "origin_authority_digest",
            "provider_record_sha256",
        ):
            _sha256(provenance.get(name), field_name=name)
    elif "window_geometry" in payload:
        raise PixelFrameAuthorityError(
            "Contained crop ownership cannot carry padded window geometry."
        )
    return CropPlacementOwnershipRecord(
        producer=str(payload["producer"]),
        layout="xywh",
        window_policy=str(payload["window_policy"]),
        crop_placement=json.loads(_canonical_json(payload["crop_placement"])),
        row_identity=json.loads(_canonical_json(payload["row_identity"])),
        source_camera_frame=json.loads(_canonical_json(payload["source_camera_frame"])),
        camera_id=_required_text(payload["camera_id"], field_name="camera_id"),
        schema_id=str(payload["schema_id"]),
        schema_version=int(payload["schema_version"]),
        window_geometry=(
            json.loads(_canonical_json(payload["window_geometry"])) if padded else None
        ),
    )


def load_crop_placement_ownership(
    placement_node: Any,
    *,
    row_identity: BoundRowIdentityContract,
    source_camera_frame: BoundPixelFrameAuthority,
    attr_name: str = CROP_PLACEMENT_OWNERSHIP_ATTR,
) -> BoundCropPlacementOwnership:
    if attr_name not in CROP_PLACEMENT_OWNERSHIP_ATTRS:
        raise PixelFrameAuthorityError(
            f"Unsupported crop-placement ownership attr {attr_name!r}."
        )
    camera = require_source_camera_pixel_frame_authority(source_camera_frame)
    expected_convention = {
        CROP_PLACEMENT_PIXEL_CENTER_OWNERSHIP_ATTR: "pixel_center",
        CROP_PLACEMENT_PIXEL_EDGE_OWNERSHIP_ATTR: "pixel_edge_half_open",
        CROP_PLACEMENT_PADDED_PIXEL_CENTER_OWNERSHIP_ATTR: "pixel_center",
        CROP_PLACEMENT_PADDED_PIXEL_EDGE_OWNERSHIP_ATTR: "pixel_edge_half_open",
    }.get(attr_name)
    if (
        expected_convention is not None
        and camera.pixel_convention != expected_convention
    ):
        raise PixelFrameAuthorityError(
            f"The {expected_convention} crop-placement ownership attr requires "
            f"a {expected_convention} source-camera frame."
        )
    try:
        archive = require_same_archive(
            placement_node,
            row_identity._rowset_node,
            row_identity._key_array_node,
            source_camera_frame._authority_node,
        )
    except ArchiveIdentityError as exc:
        raise PixelFrameAuthorityError(str(exc)) from exc
    attrs = getattr(placement_node, "attrs", None)
    if not isinstance(attrs, Mapping):
        raise PixelFrameAuthorityError("Crop placement must expose persisted attrs.")
    digest_attr = f"{attr_name}_sha256"
    raw = attrs.get(attr_name)
    record = parse_crop_placement_ownership(raw)
    expected = _crop_ownership_record(
        placement_node,
        row_identity,
        camera,
        allow_zero_padded=attr_name.startswith("coordinate_successor_padded_"),
        padded_producer=record.producer,
    )
    if not isinstance(raw, Mapping) or not _exact_json_equal(raw, record.to_dict()):
        raise PixelFrameAuthorityError(
            "Raw crop ownership is not its exact canonical mapping."
        )
    stored = _sha256(
        attrs.get(digest_attr),
        field_name=digest_attr,
    )
    if stored != record.digest() or record != expected:
        raise PixelFrameAuthorityError(
            "Crop placement ownership digest or live lineage is stale."
        )
    return BoundCropPlacementOwnership(
        record=record,
        row_identity=row_identity,
        source_camera_frame=source_camera_frame,
        archive=archive,
        placement_node=placement_node,
        attr_name=attr_name,
        _verification_seal=_BOUND_CROP_PLACEMENT_OWNERSHIP_SEAL,
    )


def stamp_crop_placement_ownership(
    placement_node: Any,
    *,
    row_identity: BoundRowIdentityContract,
    source_camera_frame: BoundPixelFrameAuthority,
    attr_name: str = CROP_PLACEMENT_OWNERSHIP_ATTR,
) -> BoundCropPlacementOwnership:
    if attr_name not in CROP_PLACEMENT_OWNERSHIP_ATTRS:
        raise PixelFrameAuthorityError(
            f"Unsupported crop-placement ownership attr {attr_name!r}."
        )
    camera = require_source_camera_pixel_frame_authority(source_camera_frame)
    expected_convention = {
        CROP_PLACEMENT_PIXEL_CENTER_OWNERSHIP_ATTR: "pixel_center",
        CROP_PLACEMENT_PIXEL_EDGE_OWNERSHIP_ATTR: "pixel_edge_half_open",
        CROP_PLACEMENT_PADDED_PIXEL_CENTER_OWNERSHIP_ATTR: "pixel_center",
        CROP_PLACEMENT_PADDED_PIXEL_EDGE_OWNERSHIP_ATTR: "pixel_edge_half_open",
    }.get(attr_name)
    if (
        expected_convention is not None
        and camera.pixel_convention != expected_convention
    ):
        raise PixelFrameAuthorityError(
            f"The {expected_convention} crop-placement ownership attr requires "
            f"a {expected_convention} source-camera frame."
        )
    record = _crop_ownership_record(
        placement_node,
        row_identity,
        camera,
        allow_zero_padded=attr_name.startswith("coordinate_successor_padded_"),
    )
    attrs = require_trusted_coordinate_attrs(
        placement_node,
        label="Crop placement",
    )
    snapshot = copy.deepcopy(dict(attrs))
    digest_attr = f"{attr_name}_sha256"
    intended = {
        attr_name: record.to_dict(),
        digest_attr: record.digest(),
    }
    expected = _expected_attrs_after_update(snapshot, intended)
    try:
        attrs.update(copy.deepcopy(intended))
        _require_exact_attrs_state(
            attrs,
            expected,
            label="Crop placement post-write",
        )
        bound = load_crop_placement_ownership(
            placement_node,
            row_identity=row_identity,
            source_camera_frame=camera,
            attr_name=attr_name,
        )
        _require_exact_attrs_state(
            require_trusted_coordinate_attrs(
                placement_node,
                label="Reloaded crop placement",
            ),
            expected,
            label="Crop placement post-reload",
        )
        return bound
    except Exception as exc:
        try:
            _restore_exact_attrs(attrs, snapshot)
        except Exception as rollback_exc:  # pragma: no cover
            raise PixelFrameAuthorityError(
                f"Crop ownership stamp failed and rollback was incomplete: {rollback_exc}"
            ) from exc
        if isinstance(exc, PixelFrameAuthorityError):
            raise
        raise PixelFrameAuthorityError(f"Crop ownership stamp failed: {exc}") from exc


def require_bound_crop_placement_ownership(
    value: Any,
) -> BoundCropPlacementOwnership:
    if (
        type(value) is not BoundCropPlacementOwnership
        or value._seal is not _BOUND_CROP_PLACEMENT_OWNERSHIP_SEAL
    ):
        raise PixelFrameAuthorityError(
            "A sealed crop-writer placement ownership record is required."
        )
    verify_persisted_proof(_bound_authority_proof_key(value), value.assert_verified)
    return value


def _roi_lineage(
    crop_placement_ownership: BoundCropPlacementOwnership,
) -> dict[str, Any]:
    ownership = require_bound_crop_placement_ownership(crop_placement_ownership)
    return {
        "crop_placement_ownership": {
            "record_ref": ownership.record_ref,
            "record_sha256": ownership.record_sha256,
        },
        "crop_placement": copy.deepcopy(dict(ownership.record.crop_placement)),
        "layout": "xywh",
        "window_policy": ownership.record.window_policy,
        "row_identity": copy.deepcopy(dict(ownership.record.row_identity)),
        "source_camera_frame": copy.deepcopy(
            dict(ownership.record.source_camera_frame)
        ),
        "camera_id": ownership.record.camera_id,
    }


def _validate_roi(
    record: PixelFrameRecord,
    *,
    reference_extent: BoundReferenceExtent,
    crop_placement_ownership: BoundCropPlacementOwnership,
) -> None:
    ownership = require_bound_crop_placement_ownership(crop_placement_ownership)
    row_identity = ownership.row_identity
    _validate_common(
        record,
        expected_kind=ROI_FRAME_KIND,
        reference_extent=reference_extent,
    )
    path = canonical_node_path(reference_extent._authority_node)
    if not path.startswith(f"{row_identity.rowset_path}/"):
        raise PixelFrameAuthorityError(
            "ROI extent authority must be a descendant of its exact rowset."
        )
    if record.lineage != _roi_lineage(ownership):
        raise PixelFrameAuthorityError("ROI lineage changed or is incomplete.")


def _model_semantics(transform: ModelInputTransform) -> dict[str, Any]:
    values = {
        "name": transform.name,
        "native_height": transform.native_height,
        "native_width": transform.native_width,
        "model_height": transform.model_height,
        "model_width": transform.model_width,
        "pad_top": transform.pad_top,
        "pad_bottom": transform.pad_bottom,
        "pad_left": transform.pad_left,
        "pad_right": transform.pad_right,
    }
    for name, value in values.items():
        if name == "name":
            continue
        if type(value) is not int or value < 0:
            raise PixelFrameAuthorityError(
                f"Model preprocessing {name} must be an exact nonnegative integer."
            )
    if (
        min(
            transform.native_height,
            transform.native_width,
            transform.model_height,
            transform.model_width,
        )
        <= 0
    ):
        raise PixelFrameAuthorityError("Model/native dimensions must be positive.")
    if (
        transform.pad_top + transform.native_height + transform.pad_bottom
        != transform.model_height
        or transform.pad_left + transform.native_width + transform.pad_right
        != transform.model_width
    ):
        raise PixelFrameAuthorityError(
            "Padding does not account exactly for model/native dimensions."
        )
    if transform.name not in {"identity", "pad_to_size"}:
        raise PixelFrameAuthorityError("Unsupported model preprocessing policy.")
    if transform.name == "identity" and not transform.is_identity:
        raise PixelFrameAuthorityError("Identity preprocessing is inconsistent.")
    if transform.name == "pad_to_size":
        dy = transform.model_height - transform.native_height
        dx = transform.model_width - transform.native_width
        if (
            transform.pad_top != dy // 2
            or transform.pad_bottom != dy - dy // 2
            or transform.pad_left != dx // 2
            or transform.pad_right != dx - dx // 2
        ):
            raise PixelFrameAuthorityError(
                "pad_to_size must record exact centered padding."
            )
    return values


def model_input_to_roi_matrix(transform: ModelInputTransform) -> np.ndarray:
    checked = _model_semantics(transform)
    return np.asarray(
        [
            [1.0, 0.0, -float(checked["pad_left"])],
            [0.0, 1.0, -float(checked["pad_top"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _model_lineage(
    preprocessing_node: Any,
    transform: ModelInputTransform,
    roi_frame: BoundPixelFrameAuthority,
) -> dict[str, Any]:
    roi = require_roi_pixel_frame_authority(roi_frame)
    matrix = _array_values(preprocessing_node)
    if matrix.dtype.str != "<f8" or matrix.shape != (3, 3):
        raise PixelFrameAuthorityError(
            "Model preprocessing matrix must be persisted little-endian float64 3x3."
        )
    if not np.array_equal(matrix, model_input_to_roi_matrix(transform)):
        raise PixelFrameAuthorityError(
            "Persisted model preprocessing matrix conflicts with its policy."
        )
    return {
        "preprocessing_payload": _array_pointer(preprocessing_node),
        "preprocessing": _model_semantics(transform),
        "roi_frame": {
            "record_ref": roi.record_ref,
            "record_sha256": roi.record_sha256,
        },
    }


def _validate_model_input(
    record: PixelFrameRecord,
    *,
    reference_extent: BoundReferenceExtent,
    preprocessing_node: Any,
    transform: ModelInputTransform,
    roi_frame: BoundPixelFrameAuthority,
) -> None:
    _validate_common(
        record,
        expected_kind=MODEL_INPUT_FRAME_KIND,
        reference_extent=reference_extent,
    )
    roi = require_roi_pixel_frame_authority(roi_frame)
    if record.pixel_convention != roi.pixel_convention:
        raise PixelFrameAuthorityError(
            "Model-input and ROI frames must use one exact pixel convention."
        )
    if (
        record.reference_extent["width"] != transform.model_width
        or record.reference_extent["height"] != transform.model_height
        or roi.endpoint.width != transform.native_width
        or roi.endpoint.height != transform.native_height
    ):
        raise PixelFrameAuthorityError(
            "Model preprocessing dimensions conflict with exact frame extents."
        )
    if record.lineage != _model_lineage(preprocessing_node, transform, roi):
        raise PixelFrameAuthorityError(
            "Model-input preprocessing lineage changed or is incomplete."
        )


def _normalized_formula_for_pixel_frame(
    pixel_frame: BoundPixelFrameAuthority,
) -> str:
    pixel = require_bound_pixel_frame_authority(pixel_frame)
    if pixel.pixel_convention == "pixel_center":
        if pixel.endpoint.width <= 1 or pixel.endpoint.height <= 1:
            raise PixelFrameAuthorityError(
                "Pixel-center normalized frames require both reference dimensions > 1."
            )
        return NORMALIZED_TO_PIXEL_CENTER_INDEX_V1
    return NORMALIZED_TO_PIXEL_EDGE_EXTENT_V1


def normalized_to_pixel_matrix(
    pixel_frame: BoundPixelFrameAuthority,
) -> np.ndarray:
    """Return the exact controlled normalized-to-pixel affine payload."""

    pixel = require_bound_pixel_frame_authority(pixel_frame)
    formula = _normalized_formula_for_pixel_frame(pixel)
    if formula == NORMALIZED_TO_PIXEL_CENTER_INDEX_V1:
        sx = float(pixel.endpoint.width - 1)
        sy = float(pixel.endpoint.height - 1)
    else:
        sx = float(pixel.endpoint.width)
        sy = float(pixel.endpoint.height)
    return np.asarray(
        [[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]],
        dtype="<f8",
    )


def _normalized_lineage(
    pixel_frame: BoundPixelFrameAuthority,
) -> dict[str, Any]:
    pixel = require_bound_pixel_frame_authority(pixel_frame)
    if pixel.record.kind not in {SOURCE_CAMERA_FRAME_KIND, MODEL_INPUT_FRAME_KIND}:
        raise PixelFrameAuthorityError(
            "Normalized frames are supported only for source-camera or detector model-input pixels."
        )
    return {
        "pixel_frame": {
            "record_ref": pixel.record_ref,
            "record_sha256": pixel.record_sha256,
        },
        "normalization_formula": _normalized_formula_for_pixel_frame(pixel),
        "target_pixel_convention": pixel.pixel_convention,
        "reference_width_px": pixel.endpoint.width,
        "reference_height_px": pixel.endpoint.height,
    }


def _validate_normalized(
    record: PixelFrameRecord,
    *,
    reference_extent: PixelFrameReference,
    pixel_frame: BoundPixelFrameAuthority,
) -> None:
    pixel = require_bound_pixel_frame_authority(pixel_frame)
    expected_kind = (
        SOURCE_CAMERA_NORMALIZED_FRAME_KIND
        if pixel.record.kind == SOURCE_CAMERA_FRAME_KIND
        else DETECTOR_NORMALIZED_FRAME_KIND
    )
    _validate_common(
        record,
        expected_kind=expected_kind,
        reference_extent=reference_extent,
    )
    if record.coordinate_units != "normalized":
        raise PixelFrameAuthorityError("Normalized frame units must be normalized.")
    if record.pixel_convention != "continuous":
        raise PixelFrameAuthorityError(
            "Normalized coordinate frames must use the continuous convention."
        )
    if (
        record.reference_extent["width"] != pixel.endpoint.width
        or record.reference_extent["height"] != pixel.endpoint.height
        or record.lineage != _normalized_lineage(pixel)
    ):
        raise PixelFrameAuthorityError(
            "Normalized frame does not bind the exact pixel reference and formula."
        )


def _load_by_kind(
    kind: str,
    node: Any,
    *,
    reference_extent: PixelFrameReference,
    context: Mapping[str, Any],
) -> BoundPixelFrameAuthority:
    record = _load_record(node)
    if record.kind != kind:
        raise PixelFrameAuthorityError("Pixel-frame authority kind changed.")
    extra_nodes: list[Any] = []
    row_identity: BoundRowIdentityContract | None = None
    if kind == SOURCE_CAMERA_FRAME_KIND:
        extra_nodes.append(context["acquisition_frame"]._authority_node)
        _require_source_camera_authority_path(
            node,
            acquisition_frame=context["acquisition_frame"],
            pixel_convention=record.pixel_convention,
        )
        _validate_source_camera(
            record,
            acquisition_frame=context["acquisition_frame"],
        )
    elif kind == SELECTED_CANVAS_FRAME_KIND:
        selected = context["selected_calibration_snapshot"]
        extra_nodes.extend(
            (
                selected._calibration_node,
                selected._camera_node,
                selected._display_node,
                selected._matrix_node,
            )
        )
        _validate_selected_canvas(
            record,
            reference_extent=reference_extent,
            selected_calibration_snapshot=selected,
        )
    elif kind == ARENA_RELATIVE_CANVAS_FRAME_KIND:
        selected = context["selected_canvas_frame"]
        extra_nodes.extend((context["geometry_node"], selected._authority_node))
        _validate_arena_relative_canvas(
            record,
            reference_extent=reference_extent,
            geometry_node=context["geometry_node"],
            selected_canvas_frame=selected,
        )
    elif kind == ROI_FRAME_KIND:
        ownership = context["crop_placement_ownership"]
        ownership = require_bound_crop_placement_ownership(ownership)
        row_identity = ownership.row_identity
        extra_nodes.extend(
            (
                ownership._placement_node,
                row_identity._rowset_node,
                row_identity._key_array_node,
                ownership.source_camera_frame._authority_node,
            )
        )
        _validate_roi(
            record,
            reference_extent=reference_extent,
            crop_placement_ownership=ownership,
        )
    elif kind == MODEL_INPUT_FRAME_KIND:
        roi = context["roi_frame"]
        row_identity = roi.row_identity
        extra_nodes.extend((context["preprocessing_node"], roi._authority_node))
        _validate_model_input(
            record,
            reference_extent=reference_extent,
            preprocessing_node=context["preprocessing_node"],
            transform=context["transform"],
            roi_frame=roi,
        )
    elif kind in {
        SOURCE_CAMERA_NORMALIZED_FRAME_KIND,
        DETECTOR_NORMALIZED_FRAME_KIND,
    }:
        pixel = context["pixel_frame"]
        row_identity = pixel.row_identity
        extra_nodes.append(pixel._authority_node)
        _validate_normalized(
            record,
            reference_extent=reference_extent,
            pixel_frame=pixel,
        )
    else:  # pragma: no cover - parser already guards
        raise PixelFrameAuthorityError(f"Unsupported pixel-frame kind {kind!r}.")
    archive = _require_archive(
        node,
        reference_extent,
        *extra_nodes,
        require_colocated=(
            kind
            not in {
                SOURCE_CAMERA_FRAME_KIND,
                SOURCE_CAMERA_NORMALIZED_FRAME_KIND,
                DETECTOR_NORMALIZED_FRAME_KIND,
            }
        ),
    )
    return BoundPixelFrameAuthority(
        record=record,
        authority_path=canonical_node_path(node),
        reference_extent=reference_extent,
        row_identity=row_identity,
        archive=archive,
        authority_node=node,
        context=context,
        _verification_seal=_BOUND_PIXEL_FRAME_SEAL,
    )


def _stamp(
    node: Any,
    *,
    record: PixelFrameRecord,
    reference_extent: PixelFrameReference,
    context: Mapping[str, Any],
) -> BoundPixelFrameAuthority:
    attrs = require_trusted_coordinate_attrs(node, label="Pixel-frame node")
    # Complete validation and archive preflight before mutation.
    try:
        require_same_archive(node, reference_extent._authority_node)
    except ArchiveIdentityError as exc:
        raise PixelFrameAuthorityError(str(exc)) from exc
    snapshot = copy.deepcopy(dict(attrs))
    intended = {
        PIXEL_FRAME_AUTHORITY_ATTR: record.to_dict(),
        PIXEL_FRAME_AUTHORITY_DIGEST_ATTR: record.digest(),
    }
    expected = _expected_attrs_after_update(snapshot, intended)
    try:
        attrs.update(copy.deepcopy(intended))
        _require_exact_attrs_state(
            attrs,
            expected,
            label="Pixel-frame post-write",
        )
        bound = _load_by_kind(
            record.kind,
            node,
            reference_extent=reference_extent,
            context=context,
        )
        _require_exact_attrs_state(
            require_trusted_coordinate_attrs(
                node,
                label="Reloaded pixel frame",
            ),
            expected,
            label="Pixel-frame post-reload",
        )
        return bound
    except Exception as exc:
        try:
            _restore_exact_attrs(attrs, snapshot)
        except Exception as rollback_exc:  # pragma: no cover - hostile mapping
            raise PixelFrameAuthorityError(
                f"Pixel-frame stamp failed and rollback was incomplete: {rollback_exc}"
            ) from exc
        if isinstance(exc, PixelFrameAuthorityError):
            raise
        raise PixelFrameAuthorityError(f"Pixel-frame stamp failed: {exc}") from exc


def stamp_source_camera_pixel_frame_authority(
    authority_node: Any,
    *,
    frame_id: str,
    pixel_convention: str,
    acquisition_frame: BoundAcquisitionCameraFrame,
) -> BoundPixelFrameAuthority:
    acquisition = require_bound_acquisition_camera_frame(acquisition_frame)
    existing_attrs = require_trusted_coordinate_attrs(
        authority_node,
        label="Source-camera pixel frame",
    )
    _require_source_camera_authority_path(
        authority_node,
        acquisition_frame=acquisition,
        pixel_convention=pixel_convention,
    )
    record = _record(
        frame_id=frame_id,
        kind=SOURCE_CAMERA_FRAME_KIND,
        space_id=SOURCE_CAMERA_IMAGE_SPACE_ID,
        pixel_convention=pixel_convention,
        reference_extent=acquisition,
        lineage=_source_camera_lineage(acquisition),
    )
    _validate_source_camera(
        record,
        acquisition_frame=acquisition,
    )
    occupied = {
        PIXEL_FRAME_AUTHORITY_ATTR,
        PIXEL_FRAME_AUTHORITY_DIGEST_ATTR,
    } & set(existing_attrs)
    if occupied:
        if occupied != {
            PIXEL_FRAME_AUTHORITY_ATTR,
            PIXEL_FRAME_AUTHORITY_DIGEST_ATTR,
        }:
            raise PixelFrameAuthorityError(
                "Occupied source-camera authority has an incomplete record/digest pair; explicit migration is required."
            )
        try:
            existing = _load_by_kind(
                SOURCE_CAMERA_FRAME_KIND,
                authority_node,
                reference_extent=acquisition,
                context={"acquisition_frame": acquisition},
            )
        except PixelFrameAuthorityError as exc:
            raise PixelFrameAuthorityError(
                "Occupied source-camera authority is stale or belongs to a different camera/convention; explicit migration is required."
            ) from exc
        if existing.record != record:
            raise PixelFrameAuthorityError(
                "Source-camera authority is write-once for one camera, convention, and frame identity."
            )
        return existing
    return _stamp(
        authority_node,
        record=record,
        reference_extent=acquisition,
        context={"acquisition_frame": acquisition},
    )


def load_source_camera_pixel_frame_authority(
    node: Any,
    *,
    acquisition_frame: BoundAcquisitionCameraFrame,
) -> BoundPixelFrameAuthority:
    acquisition = require_bound_acquisition_camera_frame(acquisition_frame)
    record = _load_record(node)
    _require_source_camera_authority_path(
        node,
        acquisition_frame=acquisition,
        pixel_convention=record.pixel_convention,
    )
    return _load_by_kind(
        SOURCE_CAMERA_FRAME_KIND,
        node,
        reference_extent=acquisition,
        context={"acquisition_frame": acquisition},
    )


def stamp_selected_canvas_pixel_frame_authority(
    reference_extent: BoundReferenceExtent,
    *,
    frame_id: str,
    pixel_convention: str,
    selected_calibration_snapshot: Any,
) -> BoundPixelFrameAuthority:
    from fisheye.shared.selected_calibration import (
        require_bound_selected_calibration_snapshot,
    )

    extent = verify_bound_reference_extent(reference_extent)
    selected = require_bound_selected_calibration_snapshot(
        selected_calibration_snapshot
    )
    record = _record(
        frame_id=frame_id,
        kind=SELECTED_CANVAS_FRAME_KIND,
        space_id=STIMULUS_CANVAS_SPACE_ID,
        pixel_convention=pixel_convention,
        reference_extent=extent,
        lineage=_selected_canvas_lineage(selected),
    )
    _validate_selected_canvas(
        record,
        reference_extent=extent,
        selected_calibration_snapshot=selected,
    )
    return _stamp(
        extent._authority_node,
        record=record,
        reference_extent=extent,
        context={"selected_calibration_snapshot": selected},
    )


def load_selected_canvas_pixel_frame_authority(
    node: Any,
    *,
    reference_extent: BoundReferenceExtent,
    selected_calibration_snapshot: Any,
) -> BoundPixelFrameAuthority:
    return _load_by_kind(
        SELECTED_CANVAS_FRAME_KIND,
        node,
        reference_extent=reference_extent,
        context={
            "selected_calibration_snapshot": selected_calibration_snapshot,
        },
    )


def stamp_arena_relative_canvas_pixel_frame_authority(
    reference_extent: BoundReferenceExtent,
    *,
    frame_id: str,
    pixel_convention: str,
    geometry_node: Any,
    selected_canvas_frame: BoundPixelFrameAuthority,
) -> BoundPixelFrameAuthority:
    extent = verify_bound_reference_extent(reference_extent)
    selected = require_selected_canvas_pixel_frame_authority(selected_canvas_frame)
    context = {
        "geometry_node": geometry_node,
        "selected_canvas_frame": selected,
    }
    record = _record(
        frame_id=frame_id,
        kind=ARENA_RELATIVE_CANVAS_FRAME_KIND,
        space_id=ARENA_RELATIVE_CANVAS_SPACE_ID,
        pixel_convention=pixel_convention,
        reference_extent=extent,
        lineage=_arena_relative_canvas_lineage(geometry_node, selected),
    )
    _validate_arena_relative_canvas(
        record,
        reference_extent=extent,
        **context,
    )
    return _stamp(
        extent._authority_node,
        record=record,
        reference_extent=extent,
        context=context,
    )


def load_arena_relative_canvas_pixel_frame_authority(
    node: Any,
    *,
    reference_extent: BoundReferenceExtent,
    geometry_node: Any,
    selected_canvas_frame: BoundPixelFrameAuthority,
) -> BoundPixelFrameAuthority:
    return _load_by_kind(
        ARENA_RELATIVE_CANVAS_FRAME_KIND,
        node,
        reference_extent=reference_extent,
        context={
            "geometry_node": geometry_node,
            "selected_canvas_frame": selected_canvas_frame,
        },
    )


def stamp_roi_pixel_frame_authority(
    reference_extent: BoundReferenceExtent,
    *,
    frame_id: str,
    pixel_convention: str,
    crop_placement_ownership: BoundCropPlacementOwnership,
) -> BoundPixelFrameAuthority:
    extent = verify_bound_reference_extent(reference_extent)
    ownership = require_bound_crop_placement_ownership(crop_placement_ownership)
    context = {"crop_placement_ownership": ownership}
    record = _record(
        frame_id=frame_id,
        kind=ROI_FRAME_KIND,
        space_id=ROI_LOCAL_SPACE_ID,
        pixel_convention=pixel_convention,
        reference_extent=extent,
        lineage=_roi_lineage(ownership),
    )
    _validate_roi(record, reference_extent=extent, **context)
    return _stamp(
        extent._authority_node,
        record=record,
        reference_extent=extent,
        context=context,
    )


def load_roi_pixel_frame_authority(
    node: Any,
    *,
    reference_extent: BoundReferenceExtent,
    crop_placement_ownership: BoundCropPlacementOwnership,
) -> BoundPixelFrameAuthority:
    return _load_by_kind(
        ROI_FRAME_KIND,
        node,
        reference_extent=reference_extent,
        context={
            "crop_placement_ownership": crop_placement_ownership,
        },
    )


def stamp_model_input_pixel_frame_authority(
    reference_extent: BoundReferenceExtent,
    *,
    frame_id: str,
    pixel_convention: str,
    preprocessing_node: Any,
    transform: ModelInputTransform,
    roi_frame: BoundPixelFrameAuthority,
) -> BoundPixelFrameAuthority:
    extent = verify_bound_reference_extent(reference_extent)
    context = {
        "preprocessing_node": preprocessing_node,
        "transform": transform,
        "roi_frame": roi_frame,
    }
    record = _record(
        frame_id=frame_id,
        kind=MODEL_INPUT_FRAME_KIND,
        space_id=DETECTOR_MODEL_INPUT_SPACE_ID,
        pixel_convention=pixel_convention,
        reference_extent=extent,
        lineage=_model_lineage(preprocessing_node, transform, roi_frame),
    )
    _validate_model_input(record, reference_extent=extent, **context)
    return _stamp(
        extent._authority_node,
        record=record,
        reference_extent=extent,
        context=context,
    )


def load_model_input_pixel_frame_authority(
    node: Any,
    *,
    reference_extent: BoundReferenceExtent,
    preprocessing_node: Any,
    transform: ModelInputTransform,
    roi_frame: BoundPixelFrameAuthority,
) -> BoundPixelFrameAuthority:
    return _load_by_kind(
        MODEL_INPUT_FRAME_KIND,
        node,
        reference_extent=reference_extent,
        context={
            "preprocessing_node": preprocessing_node,
            "transform": transform,
            "roi_frame": roi_frame,
        },
    )


def stamp_normalized_pixel_frame_authority(
    authority_node: Any,
    *,
    frame_id: str,
    pixel_frame: BoundPixelFrameAuthority,
) -> BoundPixelFrameAuthority:
    """Stamp a normalized endpoint with one explicit W/H sampling formula."""

    pixel = require_bound_pixel_frame_authority(pixel_frame)
    if pixel.record.kind == SOURCE_CAMERA_FRAME_KIND:
        kind = SOURCE_CAMERA_NORMALIZED_FRAME_KIND
        space_id = SOURCE_CAMERA_NORMALIZED_SPACE_ID
    elif pixel.record.kind == MODEL_INPUT_FRAME_KIND:
        kind = DETECTOR_NORMALIZED_FRAME_KIND
        space_id = DETECTOR_NORMALIZED_SPACE_ID
    else:
        raise PixelFrameAuthorityError(
            "Normalized endpoints require source-camera or detector model-input pixels."
        )
    record = _record(
        frame_id=frame_id,
        kind=kind,
        space_id=space_id,
        coordinate_units="normalized",
        pixel_convention="continuous",
        reference_extent=pixel.reference_extent,
        lineage=_normalized_lineage(pixel),
    )
    context = {"pixel_frame": pixel}
    _validate_normalized(
        record,
        reference_extent=pixel.reference_extent,
        pixel_frame=pixel,
    )
    return _stamp(
        authority_node,
        record=record,
        reference_extent=pixel.reference_extent,
        context=context,
    )


def load_normalized_pixel_frame_authority(
    authority_node: Any,
    *,
    pixel_frame: BoundPixelFrameAuthority,
) -> BoundPixelFrameAuthority:
    pixel = require_bound_pixel_frame_authority(pixel_frame)
    if pixel.record.kind == SOURCE_CAMERA_FRAME_KIND:
        kind = SOURCE_CAMERA_NORMALIZED_FRAME_KIND
    elif pixel.record.kind == MODEL_INPUT_FRAME_KIND:
        kind = DETECTOR_NORMALIZED_FRAME_KIND
    else:
        raise PixelFrameAuthorityError(
            "Normalized endpoints require source-camera or detector model-input pixels."
        )
    return _load_by_kind(
        kind,
        authority_node,
        reference_extent=pixel.reference_extent,
        context={"pixel_frame": pixel},
    )


def require_bound_pixel_frame_authority(
    value: Any,
    *,
    expected_kind: str | None = None,
) -> BoundPixelFrameAuthority:
    if (
        type(value) is not BoundPixelFrameAuthority
        or value._seal is not _BOUND_PIXEL_FRAME_SEAL
    ):
        raise PixelFrameAuthorityError(
            "A sealed, typed pixel-frame authority is required."
        )
    verify_persisted_proof(_bound_authority_proof_key(value), value.assert_verified)
    if expected_kind is not None and value.record.kind != expected_kind:
        raise PixelFrameAuthorityError(
            f"Expected pixel-frame kind {expected_kind!r}, found {value.record.kind!r}."
        )
    return value


def require_source_camera_pixel_frame_authority(value: Any) -> BoundPixelFrameAuthority:
    return require_bound_pixel_frame_authority(
        value, expected_kind=SOURCE_CAMERA_FRAME_KIND
    )


def require_selected_canvas_pixel_frame_authority(
    value: Any,
) -> BoundPixelFrameAuthority:
    return require_bound_pixel_frame_authority(
        value, expected_kind=SELECTED_CANVAS_FRAME_KIND
    )


def require_arena_relative_canvas_pixel_frame_authority(
    value: Any,
) -> BoundPixelFrameAuthority:
    return require_bound_pixel_frame_authority(
        value,
        expected_kind=ARENA_RELATIVE_CANVAS_FRAME_KIND,
    )


def require_roi_pixel_frame_authority(value: Any) -> BoundPixelFrameAuthority:
    return require_bound_pixel_frame_authority(value, expected_kind=ROI_FRAME_KIND)


def require_model_input_pixel_frame_authority(value: Any) -> BoundPixelFrameAuthority:
    return require_bound_pixel_frame_authority(
        value, expected_kind=MODEL_INPUT_FRAME_KIND
    )


def require_normalized_pixel_frame_authority(
    value: Any,
) -> BoundPixelFrameAuthority:
    bound = require_bound_pixel_frame_authority(value)
    if bound.record.kind not in {
        SOURCE_CAMERA_NORMALIZED_FRAME_KIND,
        DETECTOR_NORMALIZED_FRAME_KIND,
    }:
        raise PixelFrameAuthorityError(
            "A source-camera or detector normalized frame is required."
        )
    return bound


__all__ = [
    "ACQUISITION_IMPORT_OWNERSHIP_ATTR",
    "ACQUISITION_IMPORT_OWNERSHIP_DIGEST_ATTR",
    "ACQUISITION_IMPORT_OWNERSHIP_SCHEMA_ID",
    "ACQUISITION_IMPORT_OWNERSHIP_SCHEMA_VERSION",
    "ACQUISITION_MATERIALIZATION_MANIFEST_SCHEMA_ID",
    "ACQUISITION_MATERIALIZATION_MANIFEST_SCHEMA_VERSION",
    "ACQUISITION_MATERIALIZATION_MANIFEST_ATTR",
    "ACQUISITION_MATERIALIZATION_MANIFEST_DIGEST_ATTR",
    "ACQUISITION_MATERIALIZATION_MANIFEST_PATH",
    "ACQUISITION_MATERIALIZATION_WRITE_POLICY",
    "ACQUISITION_CHUNK_MANIFEST_SCOPE",
    "ACQUISITION_CHUNK_CONTENT_EVIDENCE_SCOPE",
    "ACQUISITION_CHUNK_ENTRY_CANONICALIZATION",
    "ACQUISITION_METADATA_ONLY_VERIFICATION_SCOPE",
    "ACQUISITION_PHYSICAL_CHUNK_MANIFEST_SCHEMA_ID",
    "ACQUISITION_PHYSICAL_CHUNK_MANIFEST_SCHEMA_VERSION",
    "ACQUISITION_PHYSICAL_CHUNK_MANIFEST_ATTR",
    "ACQUISITION_PHYSICAL_CHUNK_MANIFEST_DIGEST_ATTR",
    "ACQUISITION_CAMERA_FRAME_ATTR",
    "ACQUISITION_CAMERA_FRAME_DIGEST_ATTR",
    "ACQUISITION_CAMERA_FRAME_SCHEMA_ID",
    "ACQUISITION_CAMERA_FRAME_SCHEMA_VERSION",
    "ARRAY_VALUES_CANONICALIZATION",
    "ARENA_RELATIVE_CANVAS_FRAME_KIND",
    "ARENA_RELATIVE_CANVAS_SPACE_ID",
    "CROP_PLACEMENT_OWNERSHIP_ATTR",
    "CROP_PLACEMENT_OWNERSHIP_ATTRS",
    "CROP_PLACEMENT_OWNERSHIP_DIGEST_ATTR",
    "CROP_PLACEMENT_PIXEL_CENTER_OWNERSHIP_ATTR",
    "CROP_PLACEMENT_PIXEL_EDGE_OWNERSHIP_ATTR",
    "CROP_PLACEMENT_PADDED_OWNERSHIP_ATTR",
    "CROP_PLACEMENT_PADDED_PIXEL_CENTER_OWNERSHIP_ATTR",
    "CROP_PLACEMENT_PADDED_PIXEL_EDGE_OWNERSHIP_ATTR",
    "CROP_PLACEMENT_PADDED_OWNERSHIP_SCHEMA_ID",
    "CROP_PLACEMENT_PADDED_OWNERSHIP_SCHEMA_VERSION",
    "CROP_PLACEMENT_PADDED_PROVENANCE_ATTR",
    "CROP_PLACEMENT_PADDED_PROVENANCE_DIGEST_ATTR",
    "CROP_PLACEMENT_PADDED_PROVENANCE_SCHEMA_ID",
    "CROP_PLACEMENT_PADDED_PROVENANCE_SCHEMA_VERSION",
    "CROP_PLACEMENT_PADDED_WINDOW_GEOMETRY_SCHEMA_ID",
    "CROP_PLACEMENT_OWNERSHIP_SCHEMA_ID",
    "CROP_PLACEMENT_OWNERSHIP_SCHEMA_VERSION",
    "CROP_PLACEMENT_WINDOW_POLICY",
    "CROP_PLACEMENT_PADDED_PRODUCER",
    "CROP_PLACEMENT_PADDED_WINDOW_POLICY",
    "DETECTOR_MODEL_INPUT_SPACE_ID",
    "DETECTOR_NORMALIZED_FRAME_KIND",
    "DETECTOR_NORMALIZED_SPACE_ID",
    "MODEL_INPUT_FRAME_KIND",
    "NORMALIZED_TO_PIXEL_CENTER_INDEX_V1",
    "NORMALIZED_TO_PIXEL_EDGE_EXTENT_V1",
    "PIXEL_FRAME_AUTHORITY_ATTR",
    "PIXEL_FRAME_AUTHORITY_CANONICALIZATION",
    "PIXEL_FRAME_AUTHORITY_DIGEST_ATTR",
    "PIXEL_FRAME_AUTHORITY_SCHEMA_ID",
    "PIXEL_FRAME_AUTHORITY_SCHEMA_VERSION",
    "PROJECTIVE_XY_DIRECT_V1",
    "ROI_FRAME_KIND",
    "ROI_LOCAL_SPACE_ID",
    "SCALE_XY_EDGE_ALIGNED_V1",
    "SCALE_XY_PIXEL_CENTER_V1",
    "SELECTED_CANVAS_FRAME_KIND",
    "SOURCE_CAMERA_FRAME_KIND",
    "SOURCE_CAMERA_IMAGE_SPACE_ID",
    "SOURCE_CAMERA_NORMALIZED_FRAME_KIND",
    "SOURCE_CAMERA_NORMALIZED_SPACE_ID",
    "STIMULUS_CANVAS_SPACE_ID",
    "TRANSLATION_XY_DIRECT_V1",
    "AcquisitionCameraFrameRecord",
    "AcquisitionImportOwnershipRecord",
    "AcquisitionPhysicalObjectEvidence",
    "BoundAcquisitionCameraFrame",
    "BoundAcquisitionImportOwnership",
    "BoundCropPlacementOwnership",
    "CropPlacementOwnershipRecord",
    "BoundPixelFrameAuthority",
    "PixelFrameAuthorityError",
    "PixelFrameEndpoint",
    "PixelFrameRecord",
    "VerifiedAcquisitionMaterialization",
    "array_values_sha256",
    "collect_acquisition_importer_physical_object_evidence",
    "load_acquisition_camera_frame",
    "load_acquisition_import_ownership",
    "load_persisted_acquisition_camera_authority",
    "load_arena_relative_canvas_pixel_frame_authority",
    "load_model_input_pixel_frame_authority",
    "load_normalized_pixel_frame_authority",
    "load_crop_placement_ownership",
    "load_roi_pixel_frame_authority",
    "load_selected_canvas_pixel_frame_authority",
    "load_source_camera_pixel_frame_authority",
    "model_input_to_roi_matrix",
    "normalized_to_pixel_matrix",
    "parse_acquisition_camera_frame",
    "parse_acquisition_import_ownership",
    "parse_crop_placement_ownership",
    "parse_pixel_frame_record",
    "parse_source_video_metadata",
    "require_bound_acquisition_camera_frame",
    "require_bound_acquisition_import_ownership",
    "require_bound_crop_placement_ownership",
    "require_bound_pixel_frame_authority",
    "require_trusted_coordinate_attrs",
    "require_arena_relative_canvas_pixel_frame_authority",
    "require_model_input_pixel_frame_authority",
    "require_normalized_pixel_frame_authority",
    "require_roi_pixel_frame_authority",
    "require_selected_canvas_pixel_frame_authority",
    "require_source_camera_pixel_frame_authority",
    "stamp_acquisition_camera_frame",
    "stamp_acquisition_import_ownership",
    "stamp_acquisition_import_writer_materialization_manifest",
    "build_verified_acquisition_materialization",
    "stamp_arena_relative_canvas_pixel_frame_authority",
    "stamp_model_input_pixel_frame_authority",
    "stamp_normalized_pixel_frame_authority",
    "stamp_crop_placement_ownership",
    "stamp_roi_pixel_frame_authority",
    "stamp_selected_canvas_pixel_frame_authority",
    "stamp_source_camera_pixel_frame_authority",
]
