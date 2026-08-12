"""Strict coordinate contract for canonical Citrus stimulus imports.

The source H5 is validated read-only before Palette opens a destination Zarr.
Historical inference and compatibility belong in separate migration tooling;
this module only accepts complete, digest-bound canonical source metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping

import h5py
import numpy as np
import zarr

from fisheye.shared.coordinate_descriptor import (
    COORDINATE_DESCRIPTOR_ATTR,
    COORDINATE_DESCRIPTOR_DIGEST_SUFFIX,
    PIXEL_FRAME_AUTHORITY_RECORD_KIND,
    CanonicalCoordinateDescriptor,
    CoordinateDescriptorError,
    load_canonical_coordinate_descriptor_attrs,
)
from fisheye.shared.canonical_coordinate_publication import (
    BoundCanonicalCoordinateDescriptor,
    build_bound_canonical_coordinate_descriptor,
    load_bound_canonical_coordinate_descriptor,
    stamp_bound_canonical_coordinate_descriptors,
)
from fisheye.shared.coordinate_identity import (
    ROW_IDENTITY_CONTRACT_ATTR,
    ROW_IDENTITY_CONTRACT_REF_ATTR,
    ROW_IDENTITY_CONTRACT_SHA256_ATTR,
    ROW_IDENTITY_KEY_ATTR,
    ROW_IDENTITY_KEY_DIGEST_ATTR,
    STIMULUS_STATE_DOMAIN,
    STIMULUS_STATE_KEY_ARRAY_REF,
    STIMULUS_STATE_KEY_MODE,
    SOURCE_ROW_ACQUISITION_FRAME_INDEX_REF,
    BoundSourceRowTemporalAuthority,
    BoundRowIdentityContract,
    RowIdentityContract,
    build_row_identity_contract,
    identity_array_content_sha256,
    load_bound_row_identity_contract,
    load_row_identity_contract_attrs,
    load_row_identity_key_attrs,
    stamp_and_bind_row_identity_contract,
    stamp_source_row_temporal_authority,
    validate_row_identity_values,
)
from fisheye.shared.coordinate_record import (
    BoundCoordinateRecord,
    bind_persisted_coordinate_record,
    stamp_and_bind_persisted_coordinate_record,
)
from fisheye.shared.coordinate_reference import (
    BoundReferenceExtent,
    bind_persisted_record_reference_extent,
)
from fisheye.shared.archive_identity import (
    ArchiveIdentity,
    ArchiveIdentityError,
    archive_identity,
    require_same_archive,
)
from fisheye.shared.selected_calibration import (
    SelectedCalibrationSnapshot,
    VerifiedSelectedCameraSourceEvidence,
    VerifiedSelectedDisplaySourceEvidence,
    VerifiedSelectedHomographySourceEvidence,
    build_selected_camera_source_evidence_from_h5_values,
    build_selected_display_source_evidence_from_h5_values,
    build_selected_homography_source_evidence_from_h5_values,
)
from fisheye.shared.pixel_frame_authority import (
    ARENA_RELATIVE_CANVAS_FRAME_KIND,
    ARENA_RELATIVE_CANVAS_SPACE_ID,
    PIXEL_FRAME_AUTHORITY_ATTR,
    PIXEL_FRAME_AUTHORITY_CANONICALIZATION,
    PIXEL_FRAME_AUTHORITY_DIGEST_ATTR,
    PIXEL_FRAME_AUTHORITY_SCHEMA_ID,
    PIXEL_FRAME_AUTHORITY_SCHEMA_VERSION,
    load_persisted_acquisition_camera_authority,
    parse_pixel_frame_record,
)
from fisheye.shared.stimulus_frame_transform import (
    BoundStimulusFrameTransformEvidence,
    load_bound_stimulus_frame_transform_evidence,
    publish_stimulus_frame_transform_evidence,
)
from fisheye.shared.zarr.columnar import store_array
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_CONTRACT,
    RUN_COMPLETION_CONTRACT_ATTR,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
)


COORDINATE_CONTRACT_EPOCH = 1
STIMULUS_IMPORT_VERSION = "2.0.0"
SOURCE_COORDINATE_POLICY_CANONICAL = "canonical_required_v1"
SOURCE_COORDINATE_POLICY_METADATA_ONLY = (
    "omit_uncontracted_surfaces_metadata_and_calibration_only_v1"
)
COORDINATE_SURFACE_MANIFEST_ATTR = "coordinate_surface_manifest"
COORDINATE_SURFACE_MANIFEST_DIGEST_ATTR = (
    f"{COORDINATE_SURFACE_MANIFEST_ATTR}_sha256"
)
COORDINATE_SURFACE_MANIFEST_SCHEMA = (
    "palette.columnar_coordinate_surface_manifest"
)
COORDINATE_SURFACE_MANIFEST_VERSION = 1
COORDINATE_IMPORT_LINEAGE_ATTR = "coordinate_import_lineage"
COORDINATE_IMPORT_LINEAGE_DIGEST_ATTR = f"{COORDINATE_IMPORT_LINEAGE_ATTR}_sha256"
COORDINATE_OUTPUT_MANIFEST_ATTR = "coordinate_output_manifest"
COORDINATE_OUTPUT_MANIFEST_DIGEST_ATTR = (
    f"{COORDINATE_OUTPUT_MANIFEST_ATTR}_sha256"
)
COORDINATE_OUTPUT_MANIFEST_SCHEMA_ID = (
    "palette.stimulus_coordinate_output_manifest"
)
COORDINATE_OUTPUT_MANIFEST_SCHEMA_VERSION = 1
CAMERA_MAPPING_RECORD_ATTR = "camera_mapping_record"
CAMERA_MAPPING_RECORD_DIGEST_ATTR = f"{CAMERA_MAPPING_RECORD_ATTR}_sha256"
CAMERA_MAPPING_SCHEMA_ID = "palette.stimulus_camera_mapping"
CAMERA_MAPPING_SCHEMA_VERSION = 1
CAMERA_FRAME_IDS_ARRAY = "camera_frame_ids"
SOURCE_ROW_INDICES_ARRAY = "source_row_indices"
SOURCE_ACQUISITION_FRAME_INDEX_ARRAY = SOURCE_ROW_ACQUISITION_FRAME_INDEX_REF
STIMULUS_STATE_KEY_ARRAY = STIMULUS_STATE_KEY_ARRAY_REF
LEGACY_SOURCE_ROW_IDENTITY_ARRAY = "coordinate_row_identity"

SOURCE_ACQUISITION_MAPPING_RECORD_ATTR = "source_acquisition_mapping_record"
SOURCE_ACQUISITION_MAPPING_RECORD_DIGEST_ATTR = (
    f"{SOURCE_ACQUISITION_MAPPING_RECORD_ATTR}_sha256"
)
SOURCE_ACQUISITION_MAPPING_SCHEMA_ID = (
    "citrus.stimulus_source_acquisition_mapping"
)
SOURCE_ACQUISITION_MAPPING_SCHEMA_VERSION = 1
SOURCE_ACQUISITION_MAPPING_ARRAY_PATH = (
    f"/tracking_data/{SOURCE_ACQUISITION_FRAME_INDEX_ARRAY}"
)

TARGET_SOURCE_ACQUISITION_FRAME_INDEX_ARRAY = (
    "target_source_acquisition_frame_index"
)
TARGET_SOURCE_ACQUISITION_FRAME_VALID_ARRAY = (
    "target_source_acquisition_frame_valid"
)
TARGET_SOURCE_ACQUISITION_MAPPING_ARRAY_PATH = (
    f"/tracking_data/{TARGET_SOURCE_ACQUISITION_FRAME_INDEX_ARRAY}"
)
TARGET_SOURCE_ACQUISITION_VALID_ARRAY_PATH = (
    f"/tracking_data/{TARGET_SOURCE_ACQUISITION_FRAME_VALID_ARRAY}"
)
TARGET_SOURCE_ACQUISITION_MAPPING_RECORD_ATTR = (
    "target_source_acquisition_mapping_record"
)
TARGET_SOURCE_ACQUISITION_MAPPING_RECORD_DIGEST_ATTR = (
    f"{TARGET_SOURCE_ACQUISITION_MAPPING_RECORD_ATTR}_sha256"
)
TARGET_SOURCE_ACQUISITION_MAPPING_RECORD_REF_ATTR = (
    f"{TARGET_SOURCE_ACQUISITION_MAPPING_RECORD_ATTR}_ref"
)
TARGET_SOURCE_ACQUISITION_MAPPING_SCHEMA_ID = (
    "citrus.stimulus_target_source_acquisition_mapping"
)
TARGET_SOURCE_ACQUISITION_MAPPING_SCHEMA_VERSION = 1

CHASER_STATES_SCHEMA_ID = "citrus.tracking.chaser_states"
CHASER_STATES_SCHEMA_VERSION = 5
# Version 5 is the immutable Batman-derivative wire contract.  Version 6 is
# intentionally *not* a permissive extension of that reader: it is consumed
# only by the explicit lossless adapter below.
CHASER_STATES_V6_SCHEMA_VERSION = 6

STIMULUS_COORDINATE_V6_RECEIPT_PATH = "/stimulus_coordinate_v6"
STIMULUS_COORDINATE_V6_RECEIPT_SCHEMA_ID = (
    "citrus.stimulus_coordinate_v6.receipt"
)
STIMULUS_COORDINATE_V6_RECEIPT_SCHEMA_VERSION = 1
STIMULUS_COORDINATE_V6_SOURCE_RECORD_PATH = (
    "/stimulus_coordinate_v6/source_semantic_record_json"
)
STIMULUS_COORDINATE_V6_NORMALIZED_RECORD_PATH = (
    "/stimulus_coordinate_v6/normalized_semantic_record_json"
)
STIMULUS_COORDINATE_V6_FRAME_METADATA_PATH = "/video_metadata/frame_metadata"
STIMULUS_COORDINATE_V6_TARGET_VALID_ARRAY = (
    "target_source_acquisition_frame_valid"
)

STIMULUS_RENDERER_SNAPSHOT_PATH = "/stimulus_renderer_snapshot"
LEGACY_STIMULUS_RENDERER_SNAPSHOT_PATH = "/stimulus_coordinates"
STIMULUS_RENDERER_SNAPSHOT_SCHEMA_ID = (
    "citrus.stimulus_renderer_snapshot"
)
STIMULUS_RENDERER_SNAPSHOT_SCHEMA_VERSION = 1
STIMULUS_RENDERER_SNAPSHOT_CAPTURE_PHASE = (
    "experiment_start_after_arena_initialization"
)

SOURCE_CALIBRATION_SCHEMA_ID = "citrus.selected_calibration_source"
SOURCE_CALIBRATION_SCHEMA_VERSION = 1
SOURCE_CALIBRATION_GROUP = "/calibration_snapshot"
SOURCE_ARENA_CONFIG_PATH = "/calibration_snapshot/arena_config_json"
SOURCE_DISPLAY_GROUP = "/display_snapshot"
SOURCE_DISPLAY_DATASET_PATH = "/display_snapshot/selected_output_block"
DESCRIPTOR_STATUS_ATTR = "coordinate_descriptor_status"
DESCRIPTOR_ISSUE_CODE_ATTR = "coordinate_descriptor_issue_code"
DESCRIPTOR_ISSUE_MESSAGE_ATTR = "coordinate_descriptor_issue_message"
ARENA_GEOMETRY_RECORD_ATTR = "arena_geometry_record"
ARENA_GEOMETRY_RECORD_DIGEST_ATTR = f"{ARENA_GEOMETRY_RECORD_ATTR}_sha256"
SOURCE_ARENA_FRAME_ID = "citrus_arena_relative_canvas"

_SURFACE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*_xy$")
_CONTROLLED_SURFACES = {
    "chaser_position": (
        "chaser_position_xy",
        ("chaser_pos_x", "chaser_pos_y"),
    ),
    "target_position": (
        "target_position_xy",
        ("target_pos_x", "target_pos_y"),
    ),
    "target_clamped_position": (
        "target_clamped_position_xy",
        ("target_clamped_pos_x", "target_clamped_pos_y"),
    ),
}
_CONTROLLED_ROLES = frozenset(_CONTROLLED_SURFACES)
_REQUIRED_ROLES = frozenset({"target_position"})
_FIELD_CLASSIFICATIONS = frozenset(
    {"row_identity", "coordinate_component", "non_spatial"}
)
_LEGACY_COORDINATE_ATTRS = frozenset(
    {
        "coordinate_frame",
        "coordinate_units",
        "coordinate_origin",
        "position_fields",
        "x_axis_direction",
        "y_axis_direction",
        "pixel_convention",
    }
)
_ARENA_ATTRS = (
    "arena_region_width_px",
    "arena_region_height_px",
    "arena_origin_in_canvas_x_px",
    "arena_origin_in_canvas_y_px",
)


class StimulusCoordinateContractError(ValueError):
    """Raised when a source or staged output is unsafe to publish."""


@dataclass(frozen=True)
class SourceSelectedCalibration:
    """Three sealed evidence records built from exact values on one H5 handle."""

    source_camera: VerifiedSelectedCameraSourceEvidence
    source_display: VerifiedSelectedDisplaySourceEvidence
    source_homography: VerifiedSelectedHomographySourceEvidence
    source_evidence_sha256: str

    @property
    def active_camera_id(self) -> str:
        return self.source_camera.active_camera_id

    @property
    def homography_matrix(self) -> np.ndarray:
        matrix = self.source_homography.matrix
        matrix.setflags(write=False)
        return matrix


@dataclass(frozen=True)
class StimulusCoordinatePreflight:
    """Immutable result of read-only source-H5 coordinate validation."""

    source_h5: Path
    source_file_identity: Mapping[str, Any]
    selected_calibration: SourceSelectedCalibration
    source_contract_sha256: str
    has_chaser_states: bool
    descriptor: CanonicalCoordinateDescriptor | None = None
    manifest: Mapping[str, Any] | None = None
    row_identity_contract: RowIdentityContract | None = None
    row_identity_fields: tuple[str, ...] = ()
    row_identity_values: np.ndarray | None = dataclass_field(
        default=None,
        repr=False,
        compare=False,
    )
    row_identity_sha256: str | None = None
    row_identity_attrs: Mapping[str, Any] | None = None
    surfaces: tuple[Mapping[str, Any], ...] = ()
    source_dataset_sha256: str | None = None
    source_arena_record: Mapping[str, Any] | None = None
    source_arena_record_sha256: str | None = None
    source_arena_record_ref: str | None = None
    source_acquisition_frame_index: np.ndarray | None = dataclass_field(
        default=None,
        repr=False,
        compare=False,
    )
    source_acquisition_mapping_record: Mapping[str, Any] | None = None
    source_acquisition_mapping_record_sha256: str | None = None
    target_source_acquisition_frame_index: np.ndarray | None = dataclass_field(
        default=None,
        repr=False,
        compare=False,
    )
    target_source_acquisition_frame_valid: np.ndarray | None = dataclass_field(
        default=None,
        repr=False,
        compare=False,
    )
    target_source_acquisition_mapping_record: Mapping[str, Any] | None = None
    target_source_acquisition_mapping_record_sha256: str | None = None
    renderer_snapshot: Mapping[str, Any] | None = None
    renderer_snapshot_source_path: str | None = None
    renderer_snapshot_sha256: str | None = None
    source_coordinate_policy: str = SOURCE_COORDINATE_POLICY_CANONICAL
    omitted_coordinate_source_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class V6StimulusCoordinateArtifact:
    """Read-only validation result for the producer-native v6 golden wire.

    This is intentionally not a ``StimulusCoordinatePreflight``.  Normal
    import cannot accept a v6 artifact until its complete calibration and
    transform normalization path is implemented end-to-end.
    """

    source_h5: Path
    source_file_identity: Mapping[str, Any]
    source_contract_sha256: str
    status: str
    source_semantic_record: Mapping[str, Any]
    source_semantic_record_sha256: str
    normalized_semantic_record: Mapping[str, Any]
    normalized_semantic_record_sha256: str
    stimulus_state_key: np.ndarray = dataclass_field(repr=False, compare=False)
    source_acquisition_frame_index: np.ndarray = dataclass_field(repr=False, compare=False)
    target_source_acquisition_frame_index: np.ndarray = dataclass_field(repr=False, compare=False)
    target_source_acquisition_frame_valid: np.ndarray = dataclass_field(repr=False, compare=False)


def _present_coordinate_source_paths(h5: h5py.File) -> tuple[str, ...]:
    paths: list[str] = []
    if STIMULUS_COORDINATE_V6_RECEIPT_PATH in h5:
        paths.append(STIMULUS_COORDINATE_V6_RECEIPT_PATH)
    if "/stimulus_coordinates" in h5:
        paths.append("/stimulus_coordinates")
    if STIMULUS_RENDERER_SNAPSHOT_PATH in h5:
        paths.append(STIMULUS_RENDERER_SNAPSHOT_PATH)
    for path in (
        "/tracking_data/bounding_boxes",
        "/tracking_data/chaser_states",
        f"/tracking_data/{STIMULUS_STATE_KEY_ARRAY}",
        f"/tracking_data/{LEGACY_SOURCE_ROW_IDENTITY_ARRAY}",
        f"/tracking_data/{SOURCE_ACQUISITION_FRAME_INDEX_ARRAY}",
        TARGET_SOURCE_ACQUISITION_MAPPING_ARRAY_PATH,
        TARGET_SOURCE_ACQUISITION_VALID_ARRAY_PATH,
    ):
        if path in h5:
            paths.append(path)
    return tuple(paths)


@dataclass(frozen=True)
class BoundStimulusCoordinateEvidence:
    """Exact same-archive evidence for one published stimulus rowset."""

    archive_identity: ArchiveIdentity
    row_identity: BoundRowIdentityContract
    arena_reference: BoundReferenceExtent
    surface_manifest: BoundCoordinateRecord
    camera_mapping: BoundCoordinateRecord
    frame_transform: BoundStimulusFrameTransformEvidence = dataclass_field(
        repr=False,
        compare=False,
    )
    source_temporal_authority: BoundSourceRowTemporalAuthority = dataclass_field(
        repr=False,
        compare=False,
    )
    import_lineage: BoundCoordinateRecord
    output_manifest: BoundCoordinateRecord
    target_source_acquisition_mapping: BoundCoordinateRecord | None
    stimulus_state_key: np.ndarray = dataclass_field(repr=False, compare=False)
    camera_frame_ids: np.ndarray = dataclass_field(repr=False, compare=False)
    source_acquisition_frame_index: np.ndarray = dataclass_field(
        repr=False,
        compare=False,
    )
    source_row_indices: np.ndarray = dataclass_field(repr=False, compare=False)
    target_source_acquisition_frame_index: np.ndarray | None = dataclass_field(
        default=None,
        repr=False,
        compare=False,
    )
    target_source_acquisition_frame_valid: np.ndarray | None = dataclass_field(
        default=None,
        repr=False,
        compare=False,
    )


def canonical_mapping_digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def numpy_content_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    header = {
        "canonicalization": "numpy_dtype_shape_c_order_bytes_v1",
        "dtype": np.lib.format.dtype_to_descr(array.dtype),
        "shape": [int(item) for item in array.shape],
    }
    digest = sha256()
    digest.update(
        json.dumps(
            header,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    digest.update(b"\x00")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _h5_dataset_content_digest(dataset: h5py.Dataset) -> str:
    string_info = h5py.check_string_dtype(dataset.dtype)
    if string_info is not None:
        dtype_record: Any = {
            "kind": "h5_string",
            "encoding": string_info.encoding,
            "length": string_info.length,
        }
    elif dataset.dtype.names:
        dtype_record = dataset.dtype.descr
    else:
        dtype_record = dataset.dtype.str
    header = {
        "canonicalization": "numpy_dtype_shape_c_order_bytes_v1",
        "dtype": dtype_record,
        "shape": [int(item) for item in dataset.shape],
    }
    digest = sha256()
    digest.update(
        json.dumps(
            header,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    digest.update(b"\x00")
    if not dataset.shape:
        digest.update(np.ascontiguousarray(np.asarray(dataset[()])).tobytes(order="C"))
        return digest.hexdigest()
    rows = int(dataset.shape[0])
    batch = max(1, min(rows, 65_536))
    for start in range(0, rows, batch):
        digest.update(
            np.ascontiguousarray(np.asarray(dataset[start : start + batch])).tobytes(
                order="C"
            )
        )
    return digest.hexdigest()


def _normalize_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="strict")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _mapping_attr(value: Any, *, label: str) -> dict[str, Any]:
    value = _normalize_attr(value)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise StimulusCoordinateContractError(
                f"{label} is not valid JSON: {exc.msg}."
            ) from exc
    if not isinstance(value, Mapping):
        raise StimulusCoordinateContractError(f"{label} must be a JSON object.")
    return dict(value)


def _renderer_snapshot_text(value: Any, *, label: str) -> str:
    normalized = _normalize_attr(value)
    if not isinstance(normalized, str) or not normalized.strip():
        raise StimulusCoordinateContractError(f"{label} must be non-empty text.")
    return normalized.strip()


def _classify_renderer_snapshot(
    h5: h5py.File,
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    """Accept only the exact attribute-only Citrus renderer snapshot shapes."""

    present = [
        path
        for path in (
            LEGACY_STIMULUS_RENDERER_SNAPSHOT_PATH,
            STIMULUS_RENDERER_SNAPSHOT_PATH,
        )
        if path in h5
    ]
    if not present:
        return None, None, None
    if len(present) != 1:
        raise StimulusCoordinateContractError(
            "Source must not contain both stimulus_coordinates and "
            "stimulus_renderer_snapshot."
        )
    path = present[0]
    root = h5[path]
    if not isinstance(root, h5py.Group):
        raise StimulusCoordinateContractError(f"{path} must be an H5 group.")

    root_attrs = {
        str(name): _normalize_attr(value) for name, value in root.attrs.items()
    }
    if path == LEGACY_STIMULUS_RENDERER_SNAPSHOT_PATH:
        if root_attrs:
            raise StimulusCoordinateContractError(
                "Legacy stimulus_coordinates is accepted only as the exact "
                "attribute-only renderer snapshot; root attrs are forbidden."
            )
        source_classification = "legacy_static_attribute_only_v1"
    else:
        expected_root_attrs = {
            "schema_id": STIMULUS_RENDERER_SNAPSHOT_SCHEMA_ID,
            "schema_version": STIMULUS_RENDERER_SNAPSHOT_SCHEMA_VERSION,
            "capture_phase": STIMULUS_RENDERER_SNAPSHOT_CAPTURE_PHASE,
        }
        if root_attrs != expected_root_attrs:
            raise StimulusCoordinateContractError(
                "stimulus_renderer_snapshot root attrs do not exactly match "
                "citrus.stimulus_renderer_snapshot v1."
            )
        source_classification = "canonical_renderer_snapshot_v1"

    arena_names = list(root.keys())
    if (
        len(arena_names) != 1
        or re.fullmatch(r"arena_[1-9][0-9]*", arena_names[0]) is None
    ):
        raise StimulusCoordinateContractError(
            f"{path} must contain exactly one arena_<positive integer> group."
        )
    arena_name = arena_names[0]
    arena = root[arena_name]
    if not isinstance(arena, h5py.Group):
        raise StimulusCoordinateContractError(
            f"{path}/{arena_name} must be an H5 group."
        )
    arena_attrs = {
        str(name): _normalize_attr(value) for name, value in arena.attrs.items()
    }
    expected_arena_attrs = {
        "active_stimulus_mode",
        "texture_height_px",
        "texture_origin",
        "texture_width_px",
    }
    if set(arena_attrs) != expected_arena_attrs:
        raise StimulusCoordinateContractError(
            f"{path}/{arena_name} attrs must be exactly "
            f"{sorted(expected_arena_attrs)!r}."
        )
    active_mode = _renderer_snapshot_text(
        arena_attrs["active_stimulus_mode"],
        label=f"{path}/{arena_name}@active_stimulus_mode",
    )
    texture_origin = _renderer_snapshot_text(
        arena_attrs["texture_origin"],
        label=f"{path}/{arena_name}@texture_origin",
    )
    if texture_origin != "top_left":
        raise StimulusCoordinateContractError(
            f"{path}/{arena_name}@texture_origin must be 'top_left'."
        )
    width = _positive_number(
        arena_attrs["texture_width_px"],
        label=f"{path}/{arena_name}@texture_width_px",
    )
    height = _positive_number(
        arena_attrs["texture_height_px"],
        label=f"{path}/{arena_name}@texture_height_px",
    )
    if type(width) is not int or type(height) is not int:
        raise StimulusCoordinateContractError(
            "Renderer snapshot texture extents must be exact integers."
        )

    if list(arena.keys()) != ["custom_coordinates"]:
        raise StimulusCoordinateContractError(
            f"{path}/{arena_name} must contain only custom_coordinates."
        )
    custom = arena["custom_coordinates"]
    if not isinstance(custom, h5py.Group) or list(custom.keys()):
        raise StimulusCoordinateContractError(
            f"{path}/{arena_name}/custom_coordinates must be an attribute-only group."
        )
    custom_attrs = {
        str(name): _normalize_attr(value) for name, value in custom.attrs.items()
    }
    if set(custom_attrs) != {"texture_center_x", "texture_center_y"}:
        raise StimulusCoordinateContractError(
            f"{path}/{arena_name}/custom_coordinates attrs must be exactly "
            "texture_center_x and texture_center_y."
        )
    center_x = _finite_number(
        custom_attrs["texture_center_x"],
        label=f"{path}/{arena_name}/custom_coordinates@texture_center_x",
    )
    center_y = _finite_number(
        custom_attrs["texture_center_y"],
        label=f"{path}/{arena_name}/custom_coordinates@texture_center_y",
    )
    if not (0 <= float(center_x) <= width and 0 <= float(center_y) <= height):
        raise StimulusCoordinateContractError(
            "Renderer snapshot texture center lies outside its declared extent."
        )

    record = {
        "schema_id": STIMULUS_RENDERER_SNAPSHOT_SCHEMA_ID,
        "schema_version": STIMULUS_RENDERER_SNAPSHOT_SCHEMA_VERSION,
        "capture_phase": STIMULUS_RENDERER_SNAPSHOT_CAPTURE_PHASE,
        "source_classification": source_classification,
        "source_path": path,
        "arena_id": arena_name,
        "active_stimulus_mode": active_mode,
        "texture_extent": {
            "width": width,
            "height": height,
            "units": "px",
        },
        "texture_origin": "top_left",
        "texture_center": {
            "x": center_x,
            "y": center_y,
            "units": "px",
        },
        "scientific_coordinate_arrays_present": False,
        "canonicalization": "canonical_json_sort_keys_v1",
    }
    return record, path, canonical_mapping_digest(record)


def _positive_number(value: Any, *, label: str) -> int | float:
    value = _normalize_attr(value)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StimulusCoordinateContractError(f"{label} must be numeric.")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0:
        raise StimulusCoordinateContractError(f"{label} must be positive and finite.")
    return int(numeric) if numeric.is_integer() else numeric


def _finite_number(value: Any, *, label: str) -> int | float:
    value = _normalize_attr(value)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StimulusCoordinateContractError(f"{label} must be numeric.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise StimulusCoordinateContractError(f"{label} must be finite.")
    return int(numeric) if numeric.is_integer() else numeric


def arena_geometry_record(attrs: Mapping[str, Any]) -> dict[str, Any]:
    missing = [name for name in _ARENA_ATTRS if name not in attrs]
    if missing:
        raise StimulusCoordinateContractError(
            f"arena_geometry is missing required attrs: {', '.join(missing)}."
        )
    return {
        "schema_id": "palette.arena_geometry_reference",
        "schema_version": 1,
        "units": "px",
        "arena_region_width_px": _positive_number(
            attrs["arena_region_width_px"], label="arena_region_width_px"
        ),
        "arena_region_height_px": _positive_number(
            attrs["arena_region_height_px"], label="arena_region_height_px"
        ),
        "arena_origin_in_canvas_x_px": _finite_number(
            attrs["arena_origin_in_canvas_x_px"],
            label="arena_origin_in_canvas_x_px",
        ),
        "arena_origin_in_canvas_y_px": _finite_number(
            attrs["arena_origin_in_canvas_y_px"],
            label="arena_origin_in_canvas_y_px",
        ),
    }


def source_arena_pixel_frame_record(
    arena_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exact typed Citrus arena frame referenced by source descriptors."""

    arena = arena_geometry_record(arena_record)
    width = arena["arena_region_width_px"]
    height = arena["arena_region_height_px"]
    origin_x = arena["arena_origin_in_canvas_x_px"]
    origin_y = arena["arena_origin_in_canvas_y_px"]
    if any(type(value) is not int for value in (width, height, origin_x, origin_y)):
        raise StimulusCoordinateContractError(
            "Source arena pixel-frame authority requires exact integer placement."
        )
    if origin_x < 0 or origin_y < 0:
        raise StimulusCoordinateContractError(
            "Source arena pixel-frame authority requires nonnegative placement."
        )
    geometry_ref = (
        f"/calibration_snapshot/arena_geometry@{ARENA_GEOMETRY_RECORD_ATTR}"
    )
    geometry_digest = canonical_mapping_digest(arena)
    payload = {
        "schema_id": PIXEL_FRAME_AUTHORITY_SCHEMA_ID,
        "schema_version": PIXEL_FRAME_AUTHORITY_SCHEMA_VERSION,
        "frame_id": SOURCE_ARENA_FRAME_ID,
        "kind": ARENA_RELATIVE_CANVAS_FRAME_KIND,
        "space_id": ARENA_RELATIVE_CANVAS_SPACE_ID,
        "coordinate_units": "px",
        "pixel_convention": "continuous",
        "reference_extent": {
            "record_ref": geometry_ref,
            "record_sha256": geometry_digest,
            "selector": (
                "attrs[arena_region_width_px,arena_region_height_px]"
            ),
            "width": width,
            "height": height,
            "units": "px",
        },
        "lineage": {
            "arena_geometry": {
                "record_ref": geometry_ref,
                "record_sha256": geometry_digest,
            },
            "layout": "selected_canvas_xywh_px",
            "origin": "arena_top_left",
            "origin_in_selected_canvas_px": {
                "x": origin_x,
                "y": origin_y,
            },
            "producer_contract": "citrus.stimulus_arena_pixel_frame.v1",
        },
        "canonicalization": PIXEL_FRAME_AUTHORITY_CANONICALIZATION,
    }
    try:
        return parse_pixel_frame_record(payload).to_dict()
    except Exception as exc:
        raise StimulusCoordinateContractError(
            f"Source arena pixel-frame record is invalid: {exc}."
        ) from exc


def _load_manifest(attrs: Mapping[str, Any]) -> tuple[
    dict[str, Any], tuple[str, ...], tuple[Mapping[str, Any], ...]
]:
    if COORDINATE_SURFACE_MANIFEST_ATTR not in attrs:
        raise StimulusCoordinateContractError(
            "Canonical chaser_states requires coordinate_surface_manifest."
        )
    if COORDINATE_SURFACE_MANIFEST_DIGEST_ATTR not in attrs:
        raise StimulusCoordinateContractError(
            "Canonical chaser_states requires coordinate_surface_manifest_sha256."
        )
    manifest = _mapping_attr(
        attrs[COORDINATE_SURFACE_MANIFEST_ATTR],
        label=COORDINATE_SURFACE_MANIFEST_ATTR,
    )
    expected = {
        "schema_id",
        "schema_version",
        "coordinate_fields_complete",
        "field_classifications",
        "row_identity_fields",
        "surfaces",
    }
    if set(manifest) != expected:
        raise StimulusCoordinateContractError(
            f"coordinate_surface_manifest fields must be exactly {sorted(expected)!r}."
        )
    if manifest.get("schema_id") != COORDINATE_SURFACE_MANIFEST_SCHEMA:
        raise StimulusCoordinateContractError("Unsupported surface-manifest schema_id.")
    if manifest.get("schema_version") != COORDINATE_SURFACE_MANIFEST_VERSION:
        raise StimulusCoordinateContractError("Unsupported surface-manifest version.")
    if manifest.get("coordinate_fields_complete") is not True:
        raise StimulusCoordinateContractError(
            "coordinate_surface_manifest must assert coordinate_fields_complete=true."
        )

    raw_rows = manifest.get("row_identity_fields")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise StimulusCoordinateContractError("row_identity_fields must be non-empty.")
    row_fields = tuple(str(value).strip() for value in raw_rows)
    if any(not value for value in row_fields) or len(set(row_fields)) != len(row_fields):
        raise StimulusCoordinateContractError("row_identity_fields must be unique names.")

    raw_classifications = manifest.get("field_classifications")
    if not isinstance(raw_classifications, Mapping) or not raw_classifications:
        raise StimulusCoordinateContractError(
            "field_classifications must be a non-empty object."
        )
    classifications: dict[str, str] = {}
    for raw_name, raw_classification in raw_classifications.items():
        if not isinstance(raw_name, str) or not raw_name.strip() or raw_name != raw_name.strip():
            raise StimulusCoordinateContractError(
                "field_classifications keys must be non-empty field names."
            )
        if raw_classification not in _FIELD_CLASSIFICATIONS:
            raise StimulusCoordinateContractError(
                f"Field {raw_name!r} has unsupported classification "
                f"{raw_classification!r}."
            )
        classifications[raw_name] = str(raw_classification)

    raw_surfaces = manifest.get("surfaces")
    if not isinstance(raw_surfaces, list) or not raw_surfaces:
        raise StimulusCoordinateContractError("surfaces must be non-empty.")
    surfaces: list[dict[str, Any]] = []
    roles: set[str] = set()
    arrays: set[str] = set()
    component_fields_seen: set[str] = set()
    for index, item in enumerate(raw_surfaces):
        if not isinstance(item, Mapping):
            raise StimulusCoordinateContractError(f"surface {index} must be an object.")
        if set(item) != {"array_name", "semantic_role", "component_fields"}:
            raise StimulusCoordinateContractError(f"surface {index} fields are invalid.")
        role = item.get("semantic_role")
        array_name = item.get("array_name")
        fields = item.get("component_fields")
        if not isinstance(role, str) or role not in _CONTROLLED_ROLES:
            raise StimulusCoordinateContractError(
                f"surface {index} has unsupported semantic_role {role!r}."
            )
        if role in roles:
            raise StimulusCoordinateContractError(f"semantic_role {role!r} is duplicated.")
        roles.add(str(role))
        expected_name, expected_fields = _CONTROLLED_SURFACES[str(role)]
        if array_name != expected_name or _SURFACE_NAME_RE.fullmatch(str(array_name)) is None:
            raise StimulusCoordinateContractError(
                f"semantic_role {role!r} requires array_name {expected_name!r}."
            )
        if array_name in arrays:
            raise StimulusCoordinateContractError(f"array_name {array_name!r} is duplicated.")
        arrays.add(str(array_name))
        if (
            not isinstance(fields, list)
            or len(fields) != 2
            or any(not isinstance(field, str) or not field.strip() for field in fields)
            or len(set(fields)) != 2
        ):
            raise StimulusCoordinateContractError(
                f"surface {array_name!r} requires two unique component_fields."
            )
        normalized_fields = [str(field).strip() for field in fields]
        if tuple(normalized_fields) != expected_fields:
            raise StimulusCoordinateContractError(
                f"semantic_role {role!r} requires component_fields "
                f"{list(expected_fields)!r}."
            )
        overlap = component_fields_seen.intersection(normalized_fields)
        if overlap:
            raise StimulusCoordinateContractError(
                f"coordinate component fields are reused: {sorted(overlap)!r}."
            )
        component_fields_seen.update(normalized_fields)
        surfaces.append(
            {
                "array_name": str(array_name),
                "semantic_role": str(role),
                "component_fields": normalized_fields,
            }
        )
    missing_roles = sorted(_REQUIRED_ROLES - roles)
    if missing_roles:
        raise StimulusCoordinateContractError(
            f"Required coordinate semantic roles are missing: {missing_roles!r}."
        )

    classified_rows = {
        name for name, classification in classifications.items()
        if classification == "row_identity"
    }
    if classified_rows != set(row_fields):
        raise StimulusCoordinateContractError(
            "row_identity_fields disagree with field_classifications."
        )
    classified_components = {
        name for name, classification in classifications.items()
        if classification == "coordinate_component"
    }
    if classified_components != component_fields_seen:
        raise StimulusCoordinateContractError(
            "surface component_fields disagree with field_classifications."
        )

    normalized = {
        "schema_id": COORDINATE_SURFACE_MANIFEST_SCHEMA,
        "schema_version": COORDINATE_SURFACE_MANIFEST_VERSION,
        "coordinate_fields_complete": True,
        "field_classifications": {
            name: classifications[name] for name in sorted(classifications)
        },
        "row_identity_fields": list(row_fields),
        "surfaces": surfaces,
    }
    stored_digest = _normalize_attr(attrs[COORDINATE_SURFACE_MANIFEST_DIGEST_ATTR])
    if stored_digest != canonical_mapping_digest(normalized):
        raise StimulusCoordinateContractError(
            "coordinate_surface_manifest_sha256 does not match canonical content."
        )
    return normalized, row_fields, tuple(surfaces)


def _load_descriptor(
    attrs: Mapping[str, Any],
    *,
    row_identity_contract: RowIdentityContract,
    owner_shape: tuple[int, ...],
) -> CanonicalCoordinateDescriptor:
    normalized_attrs = dict(attrs)
    if COORDINATE_DESCRIPTOR_ATTR in normalized_attrs:
        normalized_attrs[COORDINATE_DESCRIPTOR_ATTR] = _mapping_attr(
            normalized_attrs[COORDINATE_DESCRIPTOR_ATTR],
            label=f"chaser_states@{COORDINATE_DESCRIPTOR_ATTR}",
        )
    legacy = sorted(_LEGACY_COORDINATE_ATTRS.intersection(attrs))
    if legacy:
        raise StimulusCoordinateContractError(
            f"Canonical source must not carry legacy coordinate attrs: {legacy!r}."
        )
    try:
        descriptor = load_canonical_coordinate_descriptor_attrs(
            normalized_attrs,
            row_identity_contract=row_identity_contract,
            expected_row_identity_record_ref=(
                f"/tracking_data/chaser_states@{ROW_IDENTITY_CONTRACT_ATTR}"
            ),
            owner_shape=owner_shape,
        )
    except CoordinateDescriptorError as exc:
        raise StimulusCoordinateContractError(
            f"Canonical chaser coordinate descriptor is invalid: {exc}"
        ) from exc
    if descriptor.profile_id != "arena_relative_canvas_px.top_left_y_down.v1":
        raise StimulusCoordinateContractError(
            "Chaser descriptor must use the canonical arena-relative profile."
        )
    if descriptor.geometry_type != "point_xy":
        raise StimulusCoordinateContractError("Chaser descriptor must use point_xy.")
    if descriptor.components != ("x", "y"):
        raise StimulusCoordinateContractError("Chaser descriptor components must be x,y.")
    if descriptor.space_id != "arena_relative_canvas_px":
        raise StimulusCoordinateContractError(
            f"Unsupported stimulus chaser coordinate space {descriptor.space_id!r}; "
            "only arena_relative_canvas_px has an exact authority resolver."
        )
    if descriptor.component_units != ("px", "px"):
        raise StimulusCoordinateContractError(
            "arena_relative_canvas_px components must use px units."
        )
    if descriptor.origin != "arena_top_left":
        raise StimulusCoordinateContractError(
            "arena_relative_canvas_px must use arena_top_left origin."
        )
    if (
        descriptor.positive_directions.x != "right"
        or descriptor.positive_directions.y != "down"
    ):
        raise StimulusCoordinateContractError(
            "arena_relative_canvas_px must use positive x right and positive y down."
        )
    if descriptor.pixel_convention != "continuous":
        raise StimulusCoordinateContractError(
            "Stimulus chaser positions must use the continuous pixel convention."
        )
    if descriptor.source_camera_overlay.status != "not_suitable":
        raise StimulusCoordinateContractError(
            "Arena-relative stimulus coordinates must explicitly be not_suitable "
            "for source-camera overlay."
        )
    if descriptor.source_camera_overlay.transform_refs:
        raise StimulusCoordinateContractError(
            "Overlay transform refs are unsupported for arena-relative stimulus coordinates."
        )
    return descriptor


def _validate_source_rows_and_fields(
    dataset: h5py.Dataset,
    *,
    manifest: Mapping[str, Any],
    row_fields: tuple[str, ...],
    surfaces: tuple[Mapping[str, Any], ...],
) -> np.ndarray:
    names = dataset.dtype.names or ()
    classifications = manifest["field_classifications"]
    if set(classifications) != set(names):
        missing = sorted(set(names) - set(classifications))
        extra = sorted(set(classifications) - set(names))
        raise StimulusCoordinateContractError(
            "field_classifications must cover the structured dtype exactly; "
            f"unclassified={missing!r}, absent={extra!r}."
        )
    rows: list[np.ndarray] = []
    for field in row_fields:
        if field not in names or dataset.dtype[field].kind not in "iu":
            raise StimulusCoordinateContractError(
                f"Row identity field {field!r} must be an integer source field."
            )
        values = np.asarray(dataset[field])
        if (
            values.dtype.kind == "u"
            and values.size
            and int(values.max()) > np.iinfo(np.int64).max
        ):
            raise StimulusCoordinateContractError(
                f"Row identity field {field!r} exceeds int64."
            )
        rows.append(np.asarray(values, dtype=np.int64))
    row_key = rows[0] if len(rows) == 1 else np.column_stack(rows)
    unique = (
        np.unique(row_key).shape[0]
        if row_key.ndim == 1
        else np.unique(row_key, axis=0).shape[0]
    )
    if unique != int(dataset.shape[0]):
        raise StimulusCoordinateContractError("Source row identity is not unique.")
    for surface in surfaces:
        for field in surface["component_fields"]:
            if field not in names or dataset.dtype[field].kind not in "fiu":
                raise StimulusCoordinateContractError(
                    f"Coordinate component field {field!r} must be numeric."
                )
    return np.asarray(row_key, dtype=np.int64)


def _load_source_row_identity(
    h5: h5py.File,
    *,
    rowset: h5py.Dataset,
    expected_values: np.ndarray,
    row_fields: tuple[str, ...],
) -> tuple[np.ndarray, str, dict[str, Any], RowIdentityContract]:
    path = f"/tracking_data/{STIMULUS_STATE_KEY_ARRAY}"
    if path not in h5 or not isinstance(h5[path], h5py.Dataset):
        raise StimulusCoordinateContractError(
            f"Normal canonical import requires persisted {path}; historical "
            "coordinate_row_identity sources require an explicit migration workflow."
        )
    node = h5[path]
    if node.dtype.kind != "i" or node.dtype.itemsize != 8:
        raise StimulusCoordinateContractError(
            f"{path} must use signed int64 values."
        )
    expected_shape = tuple(int(value) for value in expected_values.shape)
    if tuple(int(value) for value in node.shape) != expected_shape:
        raise StimulusCoordinateContractError(
            f"{path} shape does not match declared structured row fields."
        )
    values = np.asarray(node[:], dtype=np.int64)
    if not np.array_equal(values, expected_values):
        raise StimulusCoordinateContractError(
            f"{path} values do not exactly equal the declared structured row fields."
        )
    attrs = {str(key): _normalize_attr(value) for key, value in node.attrs.items()}
    required_attrs = {
        ROW_IDENTITY_KEY_ATTR,
        ROW_IDENTITY_KEY_DIGEST_ATTR,
        ROW_IDENTITY_CONTRACT_REF_ATTR,
        ROW_IDENTITY_CONTRACT_SHA256_ATTR,
    }
    if set(attrs) != required_attrs:
        raise StimulusCoordinateContractError(
            f"{path} attrs must be exactly {sorted(required_attrs)!r}."
        )
    rowset_attrs = {
        str(key): _normalize_attr(value) for key, value in rowset.attrs.items()
    }
    try:
        rowset_attrs[ROW_IDENTITY_CONTRACT_ATTR] = _mapping_attr(
            rowset_attrs[ROW_IDENTITY_CONTRACT_ATTR],
            label=f"/tracking_data/chaser_states@{ROW_IDENTITY_CONTRACT_ATTR}",
        )
        contract = load_row_identity_contract_attrs(rowset_attrs)
        attrs[ROW_IDENTITY_KEY_ATTR] = _mapping_attr(
            attrs[ROW_IDENTITY_KEY_ATTR],
            label=f"{path}@{ROW_IDENTITY_KEY_ATTR}",
        )
        key = load_row_identity_key_attrs(attrs, contract=contract)
    except Exception as exc:
        raise StimulusCoordinateContractError(
            f"Canonical stimulus row-identity metadata is invalid: {exc}"
        ) from exc
    if (
        contract.domain != STIMULUS_STATE_DOMAIN
        or contract.mode != STIMULUS_STATE_KEY_MODE
        or key.ref != STIMULUS_STATE_KEY_ARRAY
        or key.components != row_fields
    ):
        raise StimulusCoordinateContractError(
            "Source row-identity contract does not exactly match the declared "
            "stimulus-state key fields."
        )
    issues = validate_row_identity_values(contract, values)
    if issues:
        detail = "; ".join(f"{item.code}: {item.message}" for item in issues)
        raise StimulusCoordinateContractError(
            f"{path} values do not satisfy the canonical row-identity contract: {detail}"
        )
    digest = identity_array_content_sha256(values)
    values.setflags(write=False)
    return values, digest, attrs, contract


def _load_source_acquisition_mapping(
    h5: h5py.File,
    *,
    row_count: int,
    row_identity_sha256: str,
    row_identity_contract_sha256: str,
) -> tuple[np.ndarray, dict[str, Any], str]:
    """Load explicit Citrus-to-acquisition time evidence without ID inference."""

    path = SOURCE_ACQUISITION_MAPPING_ARRAY_PATH
    if path not in h5 or not isinstance(h5[path], h5py.Dataset):
        raise StimulusCoordinateContractError(
            f"Future canonical stimulus import requires explicit {path}; "
            "triggering_camera_frame_id is external provenance and is never "
            "reinterpreted as acquisition_frame_index."
        )
    node = h5[path]
    if node.dtype != np.dtype("<i8") or tuple(int(v) for v in node.shape) != (
        row_count,
    ):
        raise StimulusCoordinateContractError(
            f"{path} must be exact little-endian signed int64 shape ({row_count},)."
        )
    values = np.asarray(node[:], dtype=np.int64)
    if values.size and int(values.min()) < 0:
        raise StimulusCoordinateContractError(
            f"{path} contains negative acquisition-frame indices."
        )
    attrs = {str(key): _normalize_attr(value) for key, value in node.attrs.items()}
    if set(attrs) != {
        SOURCE_ACQUISITION_MAPPING_RECORD_ATTR,
        SOURCE_ACQUISITION_MAPPING_RECORD_DIGEST_ATTR,
    }:
        raise StimulusCoordinateContractError(
            f"{path} attrs must contain only the sealed acquisition mapping record."
        )
    record = _mapping_attr(
        attrs[SOURCE_ACQUISITION_MAPPING_RECORD_ATTR],
        label=f"{path}@{SOURCE_ACQUISITION_MAPPING_RECORD_ATTR}",
    )
    expected_fields = {
        "schema_id",
        "schema_version",
        "mapping_method",
        "source_rowset_ref",
        "source_row_identity_ref",
        "source_row_identity_sha256",
        "source_row_identity_contract_sha256",
        "acquisition_recording_id",
        "acquisition_camera_id",
        "source_total_frames",
        "target_domain",
        "array_ref",
        "array_dtype",
        "array_shape",
        "array_content_sha256",
        "canonicalization",
    }
    if set(record) != expected_fields:
        raise StimulusCoordinateContractError(
            "Source acquisition mapping record fields are not closed."
        )
    recording_id = _required_text(
        record.get("acquisition_recording_id"),
        label="acquisition_recording_id",
    )
    camera_id = _required_text(
        record.get("acquisition_camera_id"),
        label="acquisition_camera_id",
    )
    total_frames = record.get("source_total_frames")
    if (
        isinstance(total_frames, bool)
        or not isinstance(total_frames, int)
        or total_frames <= 0
    ):
        raise StimulusCoordinateContractError(
            "source_total_frames must be a positive integer."
        )
    if values.size and int(values.max()) >= total_frames:
        raise StimulusCoordinateContractError(
            f"{path} contains values outside its declared acquisition frame domain."
        )
    expected = {
        "schema_id": SOURCE_ACQUISITION_MAPPING_SCHEMA_ID,
        "schema_version": SOURCE_ACQUISITION_MAPPING_SCHEMA_VERSION,
        "mapping_method": "explicit_per_stimulus_state_v1",
        "source_rowset_ref": "/tracking_data/chaser_states",
        "source_row_identity_ref": f"/tracking_data/{STIMULUS_STATE_KEY_ARRAY}",
        "source_row_identity_sha256": row_identity_sha256,
        "source_row_identity_contract_sha256": row_identity_contract_sha256,
        "acquisition_recording_id": recording_id,
        "acquisition_camera_id": camera_id,
        "source_total_frames": total_frames,
        "target_domain": "acquisition_frame_index",
        "array_ref": path,
        "array_dtype": np.dtype("<i8").str,
        "array_shape": [row_count],
        "array_content_sha256": numpy_content_digest(values),
        "canonicalization": "canonical_json_sort_keys_v1",
    }
    digest = canonical_mapping_digest(expected)
    if (
        record != expected
        or attrs.get(SOURCE_ACQUISITION_MAPPING_RECORD_DIGEST_ATTR) != digest
    ):
        raise StimulusCoordinateContractError(
            "Source acquisition mapping record or digest is stale."
        )
    values.setflags(write=False)
    return values, expected, digest


def _load_target_source_acquisition_mapping(
    h5: h5py.File,
    *,
    row_count: int,
    row_identity_sha256: str,
    row_identity_contract_sha256: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], str]:
    """Load target-detection provenance separately from state acquisition time."""

    index_path = TARGET_SOURCE_ACQUISITION_MAPPING_ARRAY_PATH
    valid_path = TARGET_SOURCE_ACQUISITION_VALID_ARRAY_PATH
    for path in (index_path, valid_path):
        if path not in h5 or not isinstance(h5[path], h5py.Dataset):
            raise StimulusCoordinateContractError(
                "Chaser rows carrying target_source_frame_id require separate "
                f"canonical target provenance arrays; missing {path}."
            )
    index_node = h5[index_path]
    valid_node = h5[valid_path]
    expected_shape = (row_count,)
    if (
        index_node.dtype != np.dtype("<i8")
        or tuple(int(value) for value in index_node.shape) != expected_shape
    ):
        raise StimulusCoordinateContractError(
            f"{index_path} must be exact little-endian signed int64 shape "
            f"{expected_shape}."
        )
    if (
        valid_node.dtype != np.dtype("bool")
        or tuple(int(value) for value in valid_node.shape) != expected_shape
    ):
        raise StimulusCoordinateContractError(
            f"{valid_path} must be exact boolean shape {expected_shape}."
        )
    indices = np.asarray(index_node[:], dtype=np.int64)
    valid = np.asarray(valid_node[:], dtype=bool)
    if np.any(indices[~valid] != -1) or np.any(indices[valid] < 0):
        raise StimulusCoordinateContractError(
            "Target-source acquisition indices must be nonnegative when valid "
            "and exactly -1 when invalid."
        )

    index_attrs = {
        str(key): _normalize_attr(value) for key, value in index_node.attrs.items()
    }
    if set(index_attrs) != {
        TARGET_SOURCE_ACQUISITION_MAPPING_RECORD_ATTR,
        TARGET_SOURCE_ACQUISITION_MAPPING_RECORD_DIGEST_ATTR,
    }:
        raise StimulusCoordinateContractError(
            f"{index_path} attrs must contain only the sealed target-source "
            "acquisition mapping record."
        )
    record = _mapping_attr(
        index_attrs[TARGET_SOURCE_ACQUISITION_MAPPING_RECORD_ATTR],
        label=(
            f"{index_path}@{TARGET_SOURCE_ACQUISITION_MAPPING_RECORD_ATTR}"
        ),
    )
    expected_fields = {
        "schema_id",
        "schema_version",
        "mapping_method",
        "source_rowset_ref",
        "source_row_identity_ref",
        "source_row_identity_sha256",
        "source_row_identity_contract_sha256",
        "source_target_frame_field",
        "source_target_camera_field",
        "acquisition_recording_id",
        "acquisition_camera_id",
        "source_total_frames",
        "target_domain",
        "array_ref",
        "array_dtype",
        "array_shape",
        "array_content_sha256",
        "validity_array_ref",
        "validity_array_dtype",
        "validity_array_shape",
        "validity_array_content_sha256",
        "invalid_index_sentinel",
        "canonicalization",
    }
    if set(record) != expected_fields:
        raise StimulusCoordinateContractError(
            "Target-source acquisition mapping record fields are not closed."
        )
    recording_id = _required_text(
        record.get("acquisition_recording_id"),
        label="target acquisition_recording_id",
    )
    camera_id = _required_text(
        record.get("acquisition_camera_id"),
        label="target acquisition_camera_id",
    )
    total_frames = record.get("source_total_frames")
    if (
        isinstance(total_frames, bool)
        or not isinstance(total_frames, int)
        or total_frames <= 0
    ):
        raise StimulusCoordinateContractError(
            "Target source_total_frames must be a positive integer."
        )
    if np.any(indices[valid] >= total_frames):
        raise StimulusCoordinateContractError(
            f"{index_path} contains values outside its acquisition frame domain."
        )
    expected = {
        "schema_id": TARGET_SOURCE_ACQUISITION_MAPPING_SCHEMA_ID,
        "schema_version": TARGET_SOURCE_ACQUISITION_MAPPING_SCHEMA_VERSION,
        "mapping_method": "explicit_per_stimulus_state_target_provenance_v1",
        "source_rowset_ref": "/tracking_data/chaser_states",
        "source_row_identity_ref": f"/tracking_data/{STIMULUS_STATE_KEY_ARRAY}",
        "source_row_identity_sha256": row_identity_sha256,
        "source_row_identity_contract_sha256": row_identity_contract_sha256,
        "source_target_frame_field": (
            "/tracking_data/chaser_states#target_source_frame_id"
        ),
        "source_target_camera_field": (
            "/tracking_data/chaser_states#target_source_camera_id"
        ),
        "acquisition_recording_id": recording_id,
        "acquisition_camera_id": camera_id,
        "source_total_frames": total_frames,
        "target_domain": "acquisition_frame_index",
        "array_ref": index_path,
        "array_dtype": np.dtype("<i8").str,
        "array_shape": [row_count],
        "array_content_sha256": numpy_content_digest(indices),
        "validity_array_ref": valid_path,
        "validity_array_dtype": np.dtype("bool").str,
        "validity_array_shape": [row_count],
        "validity_array_content_sha256": numpy_content_digest(valid),
        "invalid_index_sentinel": -1,
        "canonicalization": "canonical_json_sort_keys_v1",
    }
    digest = canonical_mapping_digest(expected)
    if (
        record != expected
        or index_attrs.get(TARGET_SOURCE_ACQUISITION_MAPPING_RECORD_DIGEST_ATTR)
        != digest
    ):
        raise StimulusCoordinateContractError(
            "Target-source acquisition mapping record or digest is stale."
        )
    valid_attrs = {
        str(key): _normalize_attr(value) for key, value in valid_node.attrs.items()
    }
    expected_valid_attrs = {
        TARGET_SOURCE_ACQUISITION_MAPPING_RECORD_REF_ATTR: (
            f"{index_path}@{TARGET_SOURCE_ACQUISITION_MAPPING_RECORD_ATTR}"
        ),
        TARGET_SOURCE_ACQUISITION_MAPPING_RECORD_DIGEST_ATTR: digest,
    }
    if valid_attrs != expected_valid_attrs:
        raise StimulusCoordinateContractError(
            "Target-source validity metadata does not bind the sealed mapping record."
        )
    indices.setflags(write=False)
    valid.setflags(write=False)
    return indices, valid, expected, digest


def _validate_source_arena(
    h5: h5py.File,
    *,
    source_h5: Path,
    descriptor: CanonicalCoordinateDescriptor,
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    if descriptor.space_id != "arena_relative_canvas_px":
        raise StimulusCoordinateContractError(
            f"Unsupported stimulus chaser coordinate space {descriptor.space_id!r}."
        )
    path = "/calibration_snapshot/arena_geometry"
    if path not in h5 or not isinstance(h5[path], h5py.Group):
        raise StimulusCoordinateContractError(
            "arena_relative_canvas_px requires /calibration_snapshot/arena_geometry."
        )
    attrs = {
        str(key): _normalize_attr(value) for key, value in h5[path].attrs.items()
    }
    record = arena_geometry_record(attrs)
    digest = canonical_mapping_digest(record)
    if ARENA_GEOMETRY_RECORD_ATTR not in attrs:
        raise StimulusCoordinateContractError(
            f"{path} lacks {ARENA_GEOMETRY_RECORD_ATTR}."
        )
    persisted_record = _mapping_attr(
        attrs[ARENA_GEOMETRY_RECORD_ATTR],
        label=f"{path}@{ARENA_GEOMETRY_RECORD_ATTR}",
    )
    if persisted_record != record or attrs.get(ARENA_GEOMETRY_RECORD_DIGEST_ATTR) != digest:
        raise StimulusCoordinateContractError(
            "Persisted arena_geometry record or digest disagrees with its exact attrs."
        )
    expected_frame = source_arena_pixel_frame_record(record)
    frame_digest = canonical_mapping_digest(expected_frame)
    persisted_frame = _mapping_attr(
        attrs.get(PIXEL_FRAME_AUTHORITY_ATTR),
        label=f"{path}@{PIXEL_FRAME_AUTHORITY_ATTR}",
    )
    if (
        persisted_frame != expected_frame
        or attrs.get(PIXEL_FRAME_AUTHORITY_DIGEST_ATTR) != frame_digest
    ):
        raise StimulusCoordinateContractError(
            "Persisted source arena pixel-frame record or digest is stale."
        )
    authority = descriptor.reference_extent.authority
    frame_ref = f"{path}@{PIXEL_FRAME_AUTHORITY_ATTR}"
    if authority.record_ref != frame_ref or authority.record_sha256 != frame_digest:
        raise StimulusCoordinateContractError(
            "Source reference authority does not name the exact typed H5 arena frame."
        )
    if authority.selector != "record":
        raise StimulusCoordinateContractError(
            "Source arena reference authority uses an unsupported selector."
        )
    frame = descriptor.frame_record
    if (
        frame is None
        or frame.kind != PIXEL_FRAME_AUTHORITY_RECORD_KIND
        or frame.record_ref != frame_ref
        or frame.record_sha256 != frame_digest
    ):
        raise StimulusCoordinateContractError(
            "Source descriptor does not bind its exact typed arena frame record."
        )
    if (
        descriptor.reference_extent.width != record["arena_region_width_px"]
        or descriptor.reference_extent.height != record["arena_region_height_px"]
    ):
        raise StimulusCoordinateContractError(
            "Source descriptor extent disagrees with source arena_geometry."
        )
    matching = [item for item in descriptor.lineage_refs if item.record_ref == frame_ref]
    if len(matching) != 1 or matching[0].record_sha256 != frame_digest:
        raise StimulusCoordinateContractError(
            "Source typed arena-frame lineage ref is missing or has a stale digest."
        )
    return (
        record,
        digest,
        f"{source_h5.resolve()}#{path}@{ARENA_GEOMETRY_RECORD_ATTR}",
    )


def _required_text(value: Any, *, label: str) -> str:
    value = _normalize_attr(value)
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise StimulusCoordinateContractError(
            f"{label} must be a non-empty trimmed string."
        )
    return value


def _decode_exact_h5_text(value: Any, *, label: str) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise StimulusCoordinateContractError(
                f"{label} must contain UTF-8 text."
            ) from exc
    if not isinstance(value, str):
        raise StimulusCoordinateContractError(f"{label} must contain UTF-8 text.")
    return value


def _source_file_identity(h5: h5py.File, *, source_h5: Path) -> dict[str, Any]:
    actual = Path(str(h5.filename)).expanduser().resolve()
    expected = source_h5.expanduser().resolve()
    if actual != expected:
        raise StimulusCoordinateContractError(
            f"Open H5 filename {actual} does not match requested source {expected}."
        )
    try:
        handle = h5.id.get_vfd_handle()
        if isinstance(handle, tuple):
            handle = handle[0]
        stat = os.fstat(int(handle))
        path_stat = os.stat(expected)
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        raise StimulusCoordinateContractError(
            "Unable to bind the open H5 handle to an exact file identity."
        ) from exc
    if (stat.st_dev, stat.st_ino) != (path_stat.st_dev, path_stat.st_ino):
        raise StimulusCoordinateContractError(
            "Requested source H5 path no longer identifies the open file handle."
        )
    return {
        "resolved_path": str(expected),
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "size_bytes": int(stat.st_size),
        "mtime_unix_ns": int(stat.st_mtime_ns),
    }


def _preflight_selected_calibration(
    h5: h5py.File,
    *,
    source_h5: Path,
) -> SourceSelectedCalibration:
    if SOURCE_ARENA_CONFIG_PATH not in h5 or not isinstance(
        h5[SOURCE_ARENA_CONFIG_PATH], h5py.Dataset
    ):
        raise StimulusCoordinateContractError(
            f"Canonical stimulus import requires {SOURCE_ARENA_CONFIG_PATH}."
        )
    arena_node = h5[SOURCE_ARENA_CONFIG_PATH]
    arena_raw = arena_node[()]
    arena_text = _decode_exact_h5_text(
        arena_raw,
        label=SOURCE_ARENA_CONFIG_PATH,
    )
    try:
        arena_config = json.loads(arena_text)
    except json.JSONDecodeError as exc:
        raise StimulusCoordinateContractError(
            f"{SOURCE_ARENA_CONFIG_PATH} is not valid JSON."
        ) from exc
    if not isinstance(arena_config, Mapping):
        raise StimulusCoordinateContractError(
            f"{SOURCE_ARENA_CONFIG_PATH} must contain a JSON object."
        )
    camera_id = _required_text(
        arena_config.get("active_camera_id"),
        label=f"{SOURCE_ARENA_CONFIG_PATH}.active_camera_id",
    )
    camera_path = f"{SOURCE_CALIBRATION_GROUP}/{camera_id}"
    if camera_path not in h5 or not isinstance(h5[camera_path], h5py.Group):
        raise StimulusCoordinateContractError(
            f"Active camera calibration group {camera_path} is missing."
        )
    camera_group = h5[camera_path]
    camera_attrs = {
        str(key): _normalize_attr(value) for key, value in camera_group.attrs.items()
    }
    try:
        source_camera = build_selected_camera_source_evidence_from_h5_values(
            source_h5_path=str(source_h5.resolve()),
            arena_config_raw=arena_raw,
            camera_group_path=camera_path,
            camera_group_attrs=camera_attrs,
            expected_camera_id=camera_id,
        )
    except Exception as exc:
        raise StimulusCoordinateContractError(
            f"Active-camera source evidence is invalid: {exc}"
        ) from exc
    numeric_path = f"{camera_path}/homography_matrix"
    yaml_path = f"{camera_path}/homography_matrix_yml"
    for path in (numeric_path, yaml_path):
        if path not in h5 or not isinstance(h5[path], h5py.Dataset):
            raise StimulusCoordinateContractError(
                f"Active camera calibration requires exact dataset {path}."
            )
    numeric_node = h5[numeric_path]
    yaml_node = h5[yaml_path]
    numeric_attrs = {
        str(key): _normalize_attr(value) for key, value in numeric_node.attrs.items()
    }
    yaml_attrs = {
        str(key): _normalize_attr(value) for key, value in yaml_node.attrs.items()
    }
    try:
        source_homography = build_selected_homography_source_evidence_from_h5_values(
            source_h5_path=str(source_h5.resolve()),
            expected_camera_id=camera_id,
            numeric_dataset_path=numeric_path,
            numeric_matrix=numeric_node[:],
            numeric_dataset_attrs=numeric_attrs,
            yaml_dataset_path=yaml_path,
            yaml_dataset_raw=yaml_node[()],
            yaml_dataset_attrs=yaml_attrs,
        )
    except Exception as exc:
        raise StimulusCoordinateContractError(
            f"Active-camera homography source evidence is invalid: {exc}"
        ) from exc

    if SOURCE_DISPLAY_GROUP not in h5 or not isinstance(
        h5[SOURCE_DISPLAY_GROUP], h5py.Group
    ):
        raise StimulusCoordinateContractError(
            f"Canonical stimulus import requires {SOURCE_DISPLAY_GROUP}."
        )
    display_group = h5[SOURCE_DISPLAY_GROUP]
    if SOURCE_DISPLAY_DATASET_PATH not in h5 or not isinstance(
        h5[SOURCE_DISPLAY_DATASET_PATH], h5py.Dataset
    ):
        raise StimulusCoordinateContractError(
            f"Canonical stimulus import requires {SOURCE_DISPLAY_DATASET_PATH}."
        )
    display_attrs = {
        str(key): _normalize_attr(value) for key, value in display_group.attrs.items()
    }
    output_node = h5[SOURCE_DISPLAY_DATASET_PATH]
    try:
        source_display = build_selected_display_source_evidence_from_h5_values(
            source_h5_path=str(source_h5.resolve()),
            display_group_path=SOURCE_DISPLAY_GROUP,
            display_group_attrs=display_attrs,
            selected_output_dataset_path=SOURCE_DISPLAY_DATASET_PATH,
            selected_output_block_raw=output_node[()],
        )
    except Exception as exc:
        raise StimulusCoordinateContractError(
            f"Selected-display source evidence is invalid: {exc}"
        ) from exc

    evidence = {
        "schema_id": SOURCE_CALIBRATION_SCHEMA_ID,
        "schema_version": SOURCE_CALIBRATION_SCHEMA_VERSION,
        "source_camera": source_camera.to_dict(),
        "source_display": source_display.to_dict(),
        "source_homography": source_homography.to_dict(),
    }
    return SourceSelectedCalibration(
        source_camera=source_camera,
        source_display=source_display,
        source_homography=source_homography,
        source_evidence_sha256=canonical_mapping_digest(evidence),
    )


def _v6_canonical_json(value: Mapping[str, Any]) -> str:
    """The named v6 JSON canonicalization, deliberately separate from v5."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _v6_sha256(value: bytes) -> str:
    return f"sha256:{sha256(value).hexdigest()}"


def _v6_mapping_attr(value: Any, *, label: str) -> dict[str, Any]:
    """Parse a v6 JSON attribute without accepting alternate encodings."""

    return _mapping_attr(value, label=label)


def _v6_read_scalar_utf8(
    h5: h5py.File,
    *,
    path: str,
    attr_name: str,
    expected_attr_value: str,
) -> tuple[bytes, dict[str, Any]]:
    if path not in h5 or not isinstance(h5[path], h5py.Dataset):
        raise StimulusCoordinateContractError(f"v6 receipt requires {path}.")
    node = h5[path]
    if node.shape != ():
        raise StimulusCoordinateContractError(f"{path} must be a scalar UTF-8 dataset.")
    string_info = h5py.check_string_dtype(node.dtype)
    if string_info is None or string_info.encoding != "utf-8" or string_info.length is not None:
        raise StimulusCoordinateContractError(
            f"{path} must use variable-length UTF-8 HDF5 string storage."
        )
    raw = node[()]
    if isinstance(raw, str):
        payload = raw.encode("utf-8")
    elif isinstance(raw, bytes):
        payload = raw
        try:
            payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise StimulusCoordinateContractError(
                f"{path} must contain UTF-8 payload bytes."
            ) from exc
    else:
        raise StimulusCoordinateContractError(f"{path} must contain UTF-8 text.")
    attrs = {str(key): _normalize_attr(value) for key, value in node.attrs.items()}
    if attrs.get(attr_name) != expected_attr_value:
        raise StimulusCoordinateContractError(
            f"{path}@{attr_name} must be {expected_attr_value!r}."
        )
    checksum_name = (
        "observed_checksum_sha256" if attr_name == "serialization" else "checksum_sha256"
    )
    if attrs.get(checksum_name) != _v6_sha256(payload):
        raise StimulusCoordinateContractError(f"{path} {checksum_name} is stale.")
    try:
        parsed = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StimulusCoordinateContractError(f"{path} is not valid UTF-8 JSON.") from exc
    if not isinstance(parsed, dict):
        raise StimulusCoordinateContractError(f"{path} must contain a JSON object.")
    return payload, parsed


def _v6_require_sha256(value: Any, *, label: str) -> str:
    value = _required_text(value, label=label)
    if re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
        raise StimulusCoordinateContractError(f"{label} must be a lowercase sha256 digest.")
    return value


def _v6_validate_orange_source_record(
    record: Mapping[str, Any],
    *,
    recording_id: str,
    camera_serial: str,
) -> int:
    """Validate the closed Orange v1 map; no frame-ID aliases are accepted."""

    expected_root = {
        "camera_streams", "canonicalization", "frame_identity_contract_ref",
        "frame_identity_contract_sha256", "recording_id", "schema_id",
        "schema_version", "status",
    }
    if set(record) != expected_root:
        raise StimulusCoordinateContractError("v6 Orange source record is not closed.")
    if (
        record.get("schema_id") != "orange.recording.acquisition_index_mapping"
        or record.get("schema_version") != 1
        or record.get("status") != "finalized"
        or record.get("canonicalization")
        != "canonical_json_utf8_sort_keys_compact_v1"
        or record.get("recording_id") != recording_id
    ):
        raise StimulusCoordinateContractError("v6 Orange source record is unsupported or unsealed.")
    _required_text(record.get("frame_identity_contract_ref"), label="v6 frame_identity_contract_ref")
    _v6_require_sha256(record.get("frame_identity_contract_sha256"), label="v6 frame_identity_contract_sha256")
    streams = record.get("camera_streams")
    if not isinstance(streams, Mapping) or not streams or camera_serial not in streams:
        raise StimulusCoordinateContractError("v6 Orange source record camera stream is not exact.")

    def validate_stream(serial: str, stream: Any) -> int:
        if not isinstance(serial, str) or not serial or not isinstance(stream, Mapping):
            raise StimulusCoordinateContractError("v6 Orange camera streams are malformed.")
        if set(stream) != {
            "camera_serial", "coverage", "conversion", "destination_identity",
            "producer_identity", "source_metadata_artifact",
        } or stream.get("camera_serial") != serial:
            raise StimulusCoordinateContractError("v6 Orange camera stream is not closed.")
        producer = stream["producer_identity"]
        destination = stream["destination_identity"]
        conversion = stream["conversion"]
        coverage = stream["coverage"]
        artifact = stream["source_metadata_artifact"]
        if not all(isinstance(value, Mapping) for value in (producer, destination, conversion, coverage, artifact)):
            raise StimulusCoordinateContractError("v6 Orange camera stream has malformed nested records.")
        relative_path = artifact.get("relative_path")
        normalized_path = (
            isinstance(relative_path, str)
            and relative_path
            and "\\" not in relative_path
            and "//" not in relative_path
            and not relative_path.startswith("/")
            and all(part not in {"", ".", ".."} for part in relative_path.split("/"))
        )
        if (
            set(producer) != {"assignment_event", "dtype", "field", "index_base", "scope"}
            or producer != {
                "assignment_event": "orange_acquisition_recording_frame_sequence",
                "dtype": "uint64", "field": "recording_frame_id", "index_base": 1,
                "scope": "recording_session_and_camera_stream",
            }
            or set(destination) != {"dtype", "field", "index_base"}
            or destination != {"field": "source_acquisition_frame_index", "dtype": "int64", "index_base": 0}
            or set(conversion) != {"constant", "expression", "method"}
            or conversion != {
                "method": "subtract_constant_v1",
                "expression": "source_acquisition_frame_index = recording_frame_id - 1",
                "constant": 1,
            }
            or set(coverage) != {
                "first_recording_frame_id", "gap_count", "last_recording_frame_id",
                "metadata_row_count", "total_acquisitions", "gap_policy",
            }
            or set(artifact) != {"media_type", "relative_path", "sha256"}
            or artifact.get("media_type") != "text/csv"
            or not normalized_path
        ):
            raise StimulusCoordinateContractError("v6 Orange conversion or artifact path is unsupported.")
        _v6_require_sha256(artifact.get("sha256"), label=f"v6 {serial} source_metadata_artifact.sha256")
        total = coverage.get("total_acquisitions")
        if (
            isinstance(total, bool) or not isinstance(total, int) or total <= 0
            or coverage.get("first_recording_frame_id") != 1
            or coverage.get("last_recording_frame_id") != total
            or coverage.get("metadata_row_count") != total
            or coverage.get("gap_count") != 0
            or coverage.get("gap_policy") != "none"
        ):
            raise StimulusCoordinateContractError("v6 Orange coverage is unsealed or contains gaps.")
        return total

    totals = {serial: validate_stream(serial, stream) for serial, stream in streams.items()}
    return totals[camera_serial]


def _v6_exact_dataset(
    h5: h5py.File, path: str, *, dtype: np.dtype, shape: tuple[int, ...]
) -> tuple[h5py.Dataset, np.ndarray]:
    if path not in h5 or not isinstance(h5[path], h5py.Dataset):
        raise StimulusCoordinateContractError(f"v6 requires dataset {path}.")
    node = h5[path]
    if node.dtype != dtype or tuple(int(item) for item in node.shape) != shape:
        raise StimulusCoordinateContractError(
            f"v6 dataset {path} must be {dtype.str} with shape {shape!r}."
        )
    return node, np.asarray(node[:])


def _v6_validate_array_digest(
    node: h5py.Dataset, values: np.ndarray, *, expected_attrs: Mapping[str, Any],
    dtype_descriptor: Any | None = None,
) -> str:
    attrs = {str(key): _normalize_attr(value) for key, value in node.attrs.items()}
    if attrs != dict(expected_attrs):
        raise StimulusCoordinateContractError(f"v6 dataset {node.name} attributes are not closed.")
    digest = _v6_sha256(_v6_array_bytes(values, dtype_descriptor=dtype_descriptor))
    if attrs.get("content_sha256") != digest:
        raise StimulusCoordinateContractError(f"v6 dataset {node.name} content digest is stale.")
    return digest


def _v6_array_bytes(values: np.ndarray, *, dtype_descriptor: Any | None = None) -> bytes:
    array = np.ascontiguousarray(np.asarray(values))
    header = _v6_canonical_json({
        "canonicalization": "numpy_dtype_shape_c_order_bytes_v1",
        "dtype": (
            np.lib.format.dtype_to_descr(array.dtype)
            if dtype_descriptor is None else dtype_descriptor
        ),
        "shape": [int(item) for item in array.shape],
    }).encode("utf-8")
    return header + b"\x00" + array.tobytes(order="C")


def _v6_validate_compound_dataset_contract(
    node: h5py.Dataset,
    values: np.ndarray,
    *,
    schema_id: str,
    field_order: tuple[str, ...],
    dtype_fields: Mapping[str, str],
) -> None:
    """Verify v6's explicit LE row serialization, never host struct bytes."""

    attrs = {str(key): _normalize_attr(value) for key, value in node.attrs.items()}
    contract = _v6_mapping_attr(attrs.get("dataset_contract"), label=f"{node.name}@dataset_contract")
    expected_keys = {"schema_id", "schema_version", "dtype", "field_order", "shape", "rows_packed_le_sha256"}
    if (
        set(contract) != expected_keys
        or contract.get("schema_id") != schema_id or contract.get("schema_version") != 1
        or contract.get("dtype") != dict(dtype_fields)
        or contract.get("field_order") != list(field_order)
        or contract.get("shape") != [int(values.shape[0])]
        or attrs.get("dataset_contract_sha256") != _v6_sha256(_v6_canonical_json(contract).encode("utf-8"))
    ):
        raise StimulusCoordinateContractError(f"v6 dataset contract for {node.name} is stale or unsupported.")
    header = _v6_canonical_json({
        "canonicalization": "packed_row_fields_little_endian_v1",
        "dtype": dict(dtype_fields), "shape": [int(values.shape[0])],
    }).encode("utf-8")
    payload = bytearray(header + b"\x00")
    for row in values:
        for name in field_order:
            payload.extend(np.asarray(row[name], dtype=np.dtype(dtype_fields[name])).tobytes())
    if contract["rows_packed_le_sha256"] != _v6_sha256(bytes(payload)):
        raise StimulusCoordinateContractError(f"v6 packed-row digest for {node.name} is stale.")


def validate_citrus_stimulus_coordinate_v6_artifact(
    h5: h5py.File,
    *,
    source_h5: Path,
) -> V6StimulusCoordinateArtifact:
    """Validate the producer-native v6 golden representation losslessly.

    This route deliberately shares no v5 wire assumptions.  It preserves the
    exact Orange bytes and validates a separately canonicalized normalized
    record before inspecting the scientific arrays.
    """

    file_identity = _source_file_identity(h5, source_h5=source_h5)
    if STIMULUS_COORDINATE_V6_RECEIPT_PATH not in h5:
        raise StimulusCoordinateContractError("v6 artifact lacks /stimulus_coordinate_v6 receipt.")
    receipt = h5[STIMULUS_COORDINATE_V6_RECEIPT_PATH]
    if not isinstance(receipt, h5py.Group):
        raise StimulusCoordinateContractError("v6 receipt must be an H5 group.")
    receipt_attrs = {str(key): _normalize_attr(value) for key, value in receipt.attrs.items()}
    required_receipt = {
        "schema_id", "schema_version", "status", "reason_code", "recording_id",
        "camera_serial", "source_record_observed_sha256",
        "source_record_declared_sha256", "normalized_record_sha256",
    }
    if set(receipt_attrs) != required_receipt:
        raise StimulusCoordinateContractError("v6 receipt attributes are not closed.")
    if (
        receipt_attrs.get("schema_id") != STIMULUS_COORDINATE_V6_RECEIPT_SCHEMA_ID
        or receipt_attrs.get("schema_version") != STIMULUS_COORDINATE_V6_RECEIPT_SCHEMA_VERSION
        or receipt_attrs.get("status") != "sealed"
        or receipt_attrs.get("reason_code") != "sealed"
    ):
        raise StimulusCoordinateContractError("v6 receipt is unsealed or unsupported.")
    recording_id = _required_text(receipt_attrs.get("recording_id"), label="v6 receipt recording_id")
    camera_serial = _required_text(receipt_attrs.get("camera_serial"), label="v6 receipt camera_serial")
    source_digest = _v6_require_sha256(receipt_attrs.get("source_record_observed_sha256"), label="v6 receipt source_record_observed_sha256")
    declared_source_digest = _v6_require_sha256(receipt_attrs.get("source_record_declared_sha256"), label="v6 receipt source_record_declared_sha256")
    normalized_digest = _v6_require_sha256(receipt_attrs.get("normalized_record_sha256"), label="v6 receipt normalized_record_sha256")
    source_bytes, source_record = _v6_read_scalar_utf8(
        h5, path=STIMULUS_COORDINATE_V6_SOURCE_RECORD_PATH,
        attr_name="serialization", expected_attr_value="exact_source_bytes",
    )
    if _v6_sha256(source_bytes) != source_digest:
        raise StimulusCoordinateContractError("v6 receipt source-record digest is stale.")
    if source_bytes != _v6_canonical_json(source_record).encode("utf-8"):
        raise StimulusCoordinateContractError("v6 source semantic record is not canonical JSON bytes.")
    source_node_attrs = {str(key): _normalize_attr(value) for key, value in h5[STIMULUS_COORDINATE_V6_SOURCE_RECORD_PATH].attrs.items()}
    if source_node_attrs != {
        "observed_checksum_sha256": source_digest,
        "declared_checksum_sha256": declared_source_digest,
        "serialization": "exact_source_bytes",
    } or declared_source_digest != source_digest:
        raise StimulusCoordinateContractError("v6 source record declared/observed digest binding is invalid.")
    source_total_frames = _v6_validate_orange_source_record(
        source_record, recording_id=recording_id, camera_serial=camera_serial
    )
    normalized_bytes, normalized_record = _v6_read_scalar_utf8(
        h5, path=STIMULUS_COORDINATE_V6_NORMALIZED_RECORD_PATH,
        attr_name="canonicalization",
        expected_attr_value="canonical_json_utf8_sort_keys_compact_v1",
    )
    expected_normalized = {
        "canonicalization": "canonical_json_utf8_sort_keys_compact_v1",
        "orange_source_record": source_record,
        "orange_source_record_sha256": source_digest,
        "schema_id": "citrus.stimulus_coordinate_v6.mapping_normalized",
        "schema_version": 1,
    }
    normalized_node_attrs = {
        str(key): _normalize_attr(value)
        for key, value in h5[STIMULUS_COORDINATE_V6_NORMALIZED_RECORD_PATH].attrs.items()
    }
    if (
        normalized_node_attrs != {
            "checksum_sha256": normalized_digest,
            "canonicalization": "canonical_json_utf8_sort_keys_compact_v1",
        }
        or normalized_record != expected_normalized
        or normalized_bytes != _v6_canonical_json(expected_normalized).encode("utf-8")
        or _v6_sha256(normalized_bytes) != normalized_digest
    ):
        raise StimulusCoordinateContractError(
            "v6 normalized semantic record is not the exact lossless adapter record."
        )

    renderer_snapshot, renderer_path, renderer_digest = _classify_renderer_snapshot(h5)
    if renderer_path != STIMULUS_RENDERER_SNAPSHOT_PATH:
        raise StimulusCoordinateContractError("v6 requires the canonical renderer snapshot.")
    if LEGACY_STIMULUS_RENDERER_SNAPSHOT_PATH in h5:
        raise StimulusCoordinateContractError("v6 must not contain legacy stimulus_coordinates.")

    state_path = "/tracking_data/chaser_states"
    if state_path not in h5 or not isinstance(h5[state_path], h5py.Dataset):
        raise StimulusCoordinateContractError("sealed v6 receipt requires chaser_states.")
    states = h5[state_path]
    state_values = np.asarray(states[:])
    expected_state_dtype = np.dtype([
        ("stimulus_frame_num", "<u8"), ("chaser_index", "<i8"),
        ("chaser_pos_x", "<f4"), ("chaser_pos_y", "<f4"),
        ("target_pos_x", "<f4"), ("target_pos_y", "<f4"),
        ("target_clamped_pos_x", "<f4"), ("target_clamped_pos_y", "<f4"),
    ])
    if states.dtype != expected_state_dtype or states.ndim != 1 or states.shape[0] <= 0:
        raise StimulusCoordinateContractError("v6 chaser_states dtype or shape is unsupported.")
    if not all(
        np.all(np.isfinite(state_values[field]))
        for field in (
            "chaser_pos_x", "chaser_pos_y", "target_pos_x", "target_pos_y",
            "target_clamped_pos_x", "target_clamped_pos_y",
        )
    ):
        raise StimulusCoordinateContractError("v6 chaser coordinate rows contain non-finite values.")
    state_attrs = {str(key): _normalize_attr(value) for key, value in states.attrs.items()}
    expected_state_attr_names = {
        "schema_id", "schema_version", "coordinate_descriptor",
        "coordinate_descriptor_sha256", "coordinate_surface_manifest",
        "coordinate_surface_manifest_sha256", "source_acquisition_mapping_record",
        "source_acquisition_mapping_record_sha256", "target_source_acquisition_mapping_record",
        "target_source_acquisition_mapping_record_sha256", "dataset_contract",
        "dataset_contract_sha256",
    }
    if set(state_attrs) != expected_state_attr_names or (
        state_attrs.get("schema_id") != CHASER_STATES_SCHEMA_ID
        or state_attrs.get("schema_version") != CHASER_STATES_V6_SCHEMA_VERSION
    ):
        raise StimulusCoordinateContractError("v6 chaser_states schema or attributes are unsupported.")
    _v6_validate_compound_dataset_contract(
        states, state_values,
        schema_id="citrus.tracking.chaser_states.dataset_contract",
        field_order=(
            "stimulus_frame_num", "chaser_index", "chaser_pos_x", "chaser_pos_y",
            "target_pos_x", "target_pos_y", "target_clamped_pos_x", "target_clamped_pos_y",
        ),
        dtype_fields={
            "stimulus_frame_num": "<u8", "chaser_index": "<i8",
            "chaser_pos_x": "<f4", "chaser_pos_y": "<f4",
            "target_pos_x": "<f4", "target_pos_y": "<f4",
            "target_clamped_pos_x": "<f4", "target_clamped_pos_y": "<f4",
        },
    )
    descriptor = _v6_mapping_attr(state_attrs["coordinate_descriptor"], label="v6 coordinate_descriptor")
    expected_descriptor = {
        "schema_id": "citrus.tracking.chaser_states.coordinate_descriptor",
        "schema_version": 1,
        "canonicalization": "canonical_json_utf8_sort_keys_compact_v1",
        "coordinate_space": "arena_relative_canvas_px",
        "origin": "top_left_of_active_arena", "units": "px",
        "x_axis": "right", "y_axis": "down",
    }
    if descriptor != expected_descriptor or state_attrs["coordinate_descriptor_sha256"] != _v6_sha256(_v6_canonical_json(descriptor).encode("utf-8")):
        raise StimulusCoordinateContractError("v6 coordinate descriptor is stale or unsupported.")
    v6_manifest = _v6_mapping_attr(state_attrs["coordinate_surface_manifest"], label="v6 coordinate_surface_manifest")
    expected_v6_manifest = {
        "schema_id": "citrus.tracking.chaser_states.surface_manifest",
        "schema_version": 1,
        "canonicalization": "canonical_json_utf8_sort_keys_compact_v1",
        "row_identity": ["chaser_index", "stimulus_frame_num"],
        "surfaces": {
            "chaser_position_xy": ["chaser_pos_x", "chaser_pos_y"],
            "target_position_xy": ["target_pos_x", "target_pos_y"],
            "target_clamped_position_xy": ["target_clamped_pos_x", "target_clamped_pos_y"],
        },
    }
    if v6_manifest != expected_v6_manifest or state_attrs["coordinate_surface_manifest_sha256"] != _v6_sha256(_v6_canonical_json(v6_manifest).encode("utf-8")):
        raise StimulusCoordinateContractError("v6 coordinate surface manifest is stale or unsupported.")

    row_count = int(states.shape[0])
    key_node, key_values = _v6_exact_dataset(
        h5, "/tracking_data/stimulus_state_key", dtype=np.dtype("<i8"), shape=(row_count, 2)
    )
    key_digest = _v6_validate_array_digest(
        key_node, key_values, expected_attrs={
            "row_identity_contract": _normalize_attr(key_node.attrs.get("row_identity_contract")),
            "row_identity_contract_sha256": _normalize_attr(key_node.attrs.get("row_identity_contract_sha256")),
            "content_sha256": _normalize_attr(key_node.attrs.get("content_sha256")),
        },
    )
    row_contract = _v6_mapping_attr(key_node.attrs["row_identity_contract"], label="v6 row_identity_contract")
    expected_row_contract = {
        "schema_id": "citrus.tracking.chaser_states.row_identity", "schema_version": 1,
        "components": ["chaser_index", "stimulus_frame_num"],
        "key_array_ref": "/tracking_data/stimulus_state_key", "key_array_sha256": key_digest,
    }
    if (
        row_contract != expected_row_contract
        or key_node.attrs.get("row_identity_contract_sha256") != _v6_sha256(_v6_canonical_json(row_contract).encode("utf-8"))
        or not np.array_equal(key_values[:, 0], state_values["chaser_index"])
        or not np.array_equal(key_values[:, 1], state_values["stimulus_frame_num"].astype(np.int64))
        or np.any(key_values < 0)
        or np.unique(key_values, axis=0).shape[0] != row_count
    ):
        raise StimulusCoordinateContractError("v6 composite stimulus-state key is invalid.")

    source_node, source_values = _v6_exact_dataset(
        h5, SOURCE_ACQUISITION_MAPPING_ARRAY_PATH, dtype=np.dtype("<i8"), shape=(row_count,)
    )
    source_array_digest = _v6_validate_array_digest(
        source_node, source_values, expected_attrs={
            "content_sha256": _normalize_attr(source_node.attrs.get("content_sha256")),
            "logical_name": "current_orange_recording_acquisition",
        },
    )
    target_node, target_values = _v6_exact_dataset(
        h5, TARGET_SOURCE_ACQUISITION_MAPPING_ARRAY_PATH, dtype=np.dtype("<i8"), shape=(row_count,)
    )
    target_array_digest = _v6_validate_array_digest(
        target_node, target_values, expected_attrs={
            "content_sha256": _normalize_attr(target_node.attrs.get("content_sha256")),
            "logical_name": "held_target_orange_recording_acquisition",
        },
    )
    valid_path = f"/tracking_data/{STIMULUS_COORDINATE_V6_TARGET_VALID_ARRAY}"
    valid_node, valid_values = _v6_exact_dataset(
        h5, valid_path, dtype=np.dtype("u1"), shape=(row_count,)
    )
    valid_digest = _v6_validate_array_digest(
        valid_node, valid_values, expected_attrs={
            "content_sha256": _normalize_attr(valid_node.attrs.get("content_sha256")),
            "invalid_sentinel": -1, "logical_type": "bool",
        }, dtype_descriptor="|u1",
    )
    if np.any((valid_values != 0) & (valid_values != 1)):
        raise StimulusCoordinateContractError("v6 target validity must use only logical 0/1 values.")
    target_valid = valid_values.astype(bool)
    if np.any(target_values[~target_valid] != -1) or np.any(target_values[target_valid] < 0):
        raise StimulusCoordinateContractError("v6 target index and validity are inconsistent.")
    if np.any(source_values < 0) or np.any(source_values >= source_total_frames) or np.any(target_values[target_valid] >= source_total_frames):
        raise StimulusCoordinateContractError("v6 acquisition index is outside Orange coverage.")
    source_mapping = _v6_mapping_attr(state_attrs["source_acquisition_mapping_record"], label="v6 source acquisition mapping")
    expected_source_mapping = {
        "schema_id": "citrus.stimulus_source_acquisition_mapping", "schema_version": 2,
        "array_ref": SOURCE_ACQUISITION_MAPPING_ARRAY_PATH, "array_sha256": source_array_digest,
        "camera_serial": camera_serial, "orange_source_record_sha256": source_digest,
        "recording_id": recording_id, "row_identity_ref": "/tracking_data/stimulus_state_key",
        "source_total_frames": source_total_frames,
    }
    if source_mapping != expected_source_mapping or state_attrs["source_acquisition_mapping_record_sha256"] != _v6_sha256(_v6_canonical_json(source_mapping).encode("utf-8")):
        raise StimulusCoordinateContractError("v6 source acquisition mapping is stale or unsupported.")
    target_mapping = _v6_mapping_attr(state_attrs["target_source_acquisition_mapping_record"], label="v6 target acquisition mapping")
    expected_target_mapping = {
        "schema_id": "citrus.stimulus_target_source_acquisition_mapping", "schema_version": 2,
        "array_ref": TARGET_SOURCE_ACQUISITION_MAPPING_ARRAY_PATH, "array_sha256": target_array_digest,
        "validity_array_ref": valid_path, "validity_array_sha256": valid_digest,
        "invalid_sentinel": -1,
        "recording_id": recording_id, "camera_serial": camera_serial,
        "source_total_frames": source_total_frames,
        "row_identity_ref": "/tracking_data/stimulus_state_key",
        "orange_source_record_sha256": source_digest,
    }
    if target_mapping != expected_target_mapping or state_attrs["target_source_acquisition_mapping_record_sha256"] != _v6_sha256(_v6_canonical_json(target_mapping).encode("utf-8")):
        raise StimulusCoordinateContractError("v6 target acquisition mapping is stale or unsupported.")

    frame_node = h5.get(STIMULUS_COORDINATE_V6_FRAME_METADATA_PATH)
    frame_dtype = np.dtype([
        ("stimulus_frame_num", "<u8"), ("source_recording_frame_id", "<u8"),
        ("source_acquisition_frame_index", "<i8"),
    ])
    if not isinstance(frame_node, h5py.Dataset) or frame_node.dtype != frame_dtype or frame_node.ndim != 1:
        raise StimulusCoordinateContractError("v6 frame metadata dtype or shape is unsupported.")
    frames = np.asarray(frame_node[:])
    frame_attrs = {str(key): _normalize_attr(value) for key, value in frame_node.attrs.items()}
    if frame_attrs != {
        "schema_id": "citrus.stimulus_frame_metadata", "schema_version": 2,
        "source_acquisition_mapping_ref": STIMULUS_COORDINATE_V6_SOURCE_RECORD_PATH,
        "dataset_contract": _normalize_attr(frame_node.attrs.get("dataset_contract")),
        "dataset_contract_sha256": _normalize_attr(frame_node.attrs.get("dataset_contract_sha256")),
    } or frames.size == 0:
        raise StimulusCoordinateContractError("v6 frame metadata attributes are unsupported.")
    _v6_validate_compound_dataset_contract(
        frame_node, frames,
        schema_id="citrus.stimulus_frame_metadata.contract",
        field_order=("stimulus_frame_num", "source_recording_frame_id", "source_acquisition_frame_index"),
        dtype_fields={"stimulus_frame_num": "<u8", "source_recording_frame_id": "<u8", "source_acquisition_frame_index": "<i8"},
    )
    frame_keys = frames["stimulus_frame_num"].astype(np.int64)
    frame_indices = frames["source_acquisition_frame_index"]
    frame_recording_ids = frames["source_recording_frame_id"]
    if (
        np.any(frame_keys < 0) or np.unique(frame_keys).size != frame_keys.size
        or np.any(frame_indices < 0) or np.any(frame_indices >= source_total_frames)
        or not np.array_equal(frame_recording_ids.astype(np.int64), frame_indices + 1)
    ):
        raise StimulusCoordinateContractError("v6 frame metadata does not honor the sealed Orange conversion.")
    frame_lookup = {int(key): int(index) for key, index in zip(frame_keys, frame_indices, strict=True)}
    if any(int(key) not in frame_lookup for key in key_values[:, 1]) or not np.array_equal(
        source_values, np.asarray([frame_lookup[int(key)] for key in key_values[:, 1]], dtype=np.int64)
    ):
        raise StimulusCoordinateContractError("v6 chaser rows contain a terminal orphan or stale current-frame mapping.")

    # The transform/authority chain is source evidence.  The v6 adapter does
    # not manufacture a v5 calibration object when the fixture lacks one.
    arena = h5.get("/calibration_snapshot/arena_geometry")
    if not isinstance(arena, h5py.Group):
        raise StimulusCoordinateContractError("v6 requires arena_geometry authority.")
    arena_attrs = {str(key): _normalize_attr(value) for key, value in arena.attrs.items()}
    required_arena_attrs = {
        "arena_region_width_px", "arena_region_height_px",
        "arena_origin_in_canvas_x_px", "arena_origin_in_canvas_y_px",
        "coordinate_frame_authority", "coordinate_frame_authority_sha256",
    }
    if set(arena_attrs) != required_arena_attrs:
        raise StimulusCoordinateContractError("v6 arena geometry authority is not closed.")
    authority = _v6_mapping_attr(arena_attrs["coordinate_frame_authority"], label="v6 coordinate_frame_authority")
    expected_authority_keys = {
        "arena_id", "calibration_authority_ref", "calibration_authority_sha256", "camera_serial",
        "coordinate_frame", "final_display_extent_px", "native_source_extent_px", "origin",
        "runtime_geometry_contract_ref", "runtime_geometry_contract_sha256", "units", "x_axis", "y_axis",
    }
    if (
        set(authority) != expected_authority_keys
        or authority.get("arena_id") != (renderer_snapshot or {}).get("arena_id")
        or authority.get("camera_serial") != camera_serial
        or authority.get("coordinate_frame") != "arena_relative_canvas_px"
        or authority.get("origin") != "top_left_of_active_arena"
        or authority.get("units") != "px" or authority.get("x_axis") != "right" or authority.get("y_axis") != "down"
        or arena_attrs["coordinate_frame_authority_sha256"] != _v6_sha256(_v6_canonical_json(authority).encode("utf-8"))
    ):
        raise StimulusCoordinateContractError("v6 coordinate-frame authority is stale or unsupported.")
    def positive_integer_pair(value: Any, *, label: str) -> tuple[int, int]:
        if not isinstance(value, list) or len(value) != 2:
            raise StimulusCoordinateContractError(f"v6 {label} must be a two-element integer extent.")
        result: list[int] = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
                raise StimulusCoordinateContractError(f"v6 {label} must contain positive integers.")
            result.append(item)
        return result[0], result[1]

    native_width, native_height = positive_integer_pair(
        authority.get("native_source_extent_px"), label="native_source_extent_px"
    )
    display_width, display_height = positive_integer_pair(
        authority.get("final_display_extent_px"), label="final_display_extent_px"
    )
    _ = native_width, native_height
    for name in ("runtime_geometry_contract_ref", "calibration_authority_ref"):
        _required_text(authority.get(name), label=f"v6 {name}")
    for name in ("runtime_geometry_contract_sha256", "calibration_authority_sha256"):
        _v6_require_sha256(authority.get(name), label=f"v6 {name}")
    origin_x = arena_attrs["arena_origin_in_canvas_x_px"]
    origin_y = arena_attrs["arena_origin_in_canvas_y_px"]
    if (
        isinstance(origin_x, bool) or isinstance(origin_y, bool)
        or not isinstance(origin_x, (int, float)) or not isinstance(origin_y, (int, float))
        or not math.isfinite(float(origin_x)) or not math.isfinite(float(origin_y))
        or float(origin_x) < 0 or float(origin_y) < 0
        or isinstance(arena_attrs["arena_region_width_px"], bool)
        or isinstance(arena_attrs["arena_region_height_px"], bool)
        or not isinstance(arena_attrs["arena_region_width_px"], int)
        or not isinstance(arena_attrs["arena_region_height_px"], int)
        or arena_attrs["arena_region_width_px"] <= 0
        or arena_attrs["arena_region_height_px"] <= 0
        or float(origin_x) + arena_attrs["arena_region_width_px"] > display_width
        or float(origin_y) + arena_attrs["arena_region_height_px"] > display_height
        or renderer_snapshot is None
        or renderer_snapshot["texture_extent"] != {
            "width": arena_attrs["arena_region_width_px"],
            "height": arena_attrs["arena_region_height_px"],
            "units": "px",
        }
    ):
        raise StimulusCoordinateContractError("v6 arena extent does not match its authority.")
    homographies: dict[str, np.ndarray] = {}
    for name, direction, source_space, destination_space in (
        ("source_camera_to_final_display_homography", "source_camera_to_final_display", "source_camera_px", "final_display_canvas_px"),
        ("final_display_to_source_camera_homography", "final_display_to_source_camera", "final_display_canvas_px", "source_camera_px"),
    ):
        node, values = _v6_exact_dataset(h5, f"/calibration_snapshot/arena_geometry/{name}", dtype=np.dtype("<f8"), shape=(3, 3))
        _v6_validate_array_digest(node, values, expected_attrs={
            "content_sha256": _normalize_attr(node.attrs.get("content_sha256")),
            "transform_direction": direction, "source_space": source_space,
            "destination_space": destination_space,
        })
        if not np.all(np.isfinite(values)):
            raise StimulusCoordinateContractError("v6 homography contains non-finite values.")
        if abs(float(np.linalg.det(values))) <= 1e-12:
            raise StimulusCoordinateContractError("v6 homography is singular.")
        homographies[name] = values
    if not np.allclose(
        homographies["source_camera_to_final_display_homography"]
        @ homographies["final_display_to_source_camera_homography"],
        np.eye(3), rtol=1e-9, atol=1e-9,
    ):
        raise StimulusCoordinateContractError("v6 homographies are not mutual inverses.")
    if not np.allclose(
        homographies["final_display_to_source_camera_homography"]
        @ homographies["source_camera_to_final_display_homography"],
        np.eye(3), rtol=1e-9, atol=1e-9,
    ):
        raise StimulusCoordinateContractError("v6 inverse homography does not round-trip.")

    source_dataset_digest = _h5_dataset_content_digest(states)
    evidence = {
        "source_file_identity": dict(file_identity), "source_schema_version": 6,
        "v6_source_semantic_record_sha256": source_digest,
        "v6_normalized_semantic_record_sha256": normalized_digest,
        "v6_chaser_states_sha256": source_dataset_digest,
        "v6_stimulus_state_key_sha256": key_digest,
        "v6_renderer_snapshot_sha256": renderer_digest,
    }
    for values in (key_values, source_values, target_values, target_valid):
        values.setflags(write=False)
    return V6StimulusCoordinateArtifact(
        source_h5=source_h5.expanduser().resolve(), source_file_identity=file_identity,
        source_contract_sha256=canonical_mapping_digest(evidence),
        status="pre_materialization_only",
        source_semantic_record=source_record, source_semantic_record_sha256=source_digest,
        normalized_semantic_record=normalized_record,
        normalized_semantic_record_sha256=normalized_digest,
        stimulus_state_key=key_values,
        source_acquisition_frame_index=source_values,
        target_source_acquisition_frame_index=target_values,
        target_source_acquisition_frame_valid=target_valid,
    )


def preflight_stimulus_coordinate_contract(
    h5: h5py.File,
    *,
    source_h5: Path,
    metadata_and_calibration_only: bool = False,
) -> StimulusCoordinatePreflight:
    """Validate one already-open H5 handle before any destination mutation."""

    source = source_h5.expanduser().resolve()
    file_identity = _source_file_identity(h5, source_h5=source)
    try:
        selected_calibration = _preflight_selected_calibration(
            h5,
            source_h5=source,
        )
    except StimulusCoordinateContractError:
        raise
    except Exception as exc:
        raise StimulusCoordinateContractError(
            f"Selected calibration evidence is invalid: {exc}"
        ) from exc

    omitted_paths = _present_coordinate_source_paths(h5)
    if metadata_and_calibration_only:
        evidence = {
            "source_file_identity": file_identity,
            "selected_calibration_sha256": (
                selected_calibration.source_evidence_sha256
            ),
            "source_coordinate_policy": SOURCE_COORDINATE_POLICY_METADATA_ONLY,
            "omitted_coordinate_source_paths": list(omitted_paths),
            "has_chaser_states": False,
        }
        return StimulusCoordinatePreflight(
            source_h5=source,
            source_file_identity=file_identity,
            selected_calibration=selected_calibration,
            source_contract_sha256=canonical_mapping_digest(evidence),
            has_chaser_states=False,
            source_coordinate_policy=SOURCE_COORDINATE_POLICY_METADATA_ONLY,
            omitted_coordinate_source_paths=omitted_paths,
        )
    (
        renderer_snapshot,
        renderer_snapshot_source_path,
        renderer_snapshot_sha256,
    ) = _classify_renderer_snapshot(h5)
    if "/tracking_data/bounding_boxes" in h5:
        bboxes = h5["/tracking_data/bounding_boxes"]
        if isinstance(bboxes, h5py.Dataset) and int(bboxes.size) > 0:
            raise StimulusCoordinateContractError(
                "Stimulus bounding_boxes lacks canonical array-specific geometry support."
            )

    path = "/tracking_data/chaser_states"
    if path not in h5:
        orphan_keys = [
            name
            for name in (
                STIMULUS_STATE_KEY_ARRAY,
                LEGACY_SOURCE_ROW_IDENTITY_ARRAY,
                SOURCE_ACQUISITION_FRAME_INDEX_ARRAY,
            )
            if f"/tracking_data/{name}" in h5
        ]
        if orphan_keys:
            raise StimulusCoordinateContractError(
                f"Orphan stimulus row-identity arrays exist without chaser_states: "
                f"{orphan_keys!r}."
            )
        evidence = {
            "source_file_identity": file_identity,
            "selected_calibration_sha256": (
                selected_calibration.source_evidence_sha256
            ),
            "has_chaser_states": False,
            "renderer_snapshot_sha256": renderer_snapshot_sha256,
        }
        return StimulusCoordinatePreflight(
            source_h5=source,
            source_file_identity=file_identity,
            selected_calibration=selected_calibration,
            source_contract_sha256=canonical_mapping_digest(evidence),
            has_chaser_states=False,
            renderer_snapshot=renderer_snapshot,
            renderer_snapshot_source_path=renderer_snapshot_source_path,
            renderer_snapshot_sha256=renderer_snapshot_sha256,
        )
    dataset = h5[path]
    if not isinstance(dataset, h5py.Dataset):
        raise StimulusCoordinateContractError(
            "/tracking_data/chaser_states must be an H5 dataset."
        )
    if int(dataset.size) == 0:
        raise StimulusCoordinateContractError(
            "Persisted empty chaser_states is unsupported; omit the geometry surface."
        )
    if f"/tracking_data/{LEGACY_SOURCE_ROW_IDENTITY_ARRAY}" in h5:
        raise StimulusCoordinateContractError(
            "Normal canonical import does not consume coordinate_row_identity; "
            "use an explicit historical migration workflow."
        )
    attrs = {str(key): _normalize_attr(value) for key, value in dataset.attrs.items()}
    if (
        attrs.get("schema_id") != CHASER_STATES_SCHEMA_ID
        or attrs.get("schema_version") != CHASER_STATES_SCHEMA_VERSION
    ):
        raise StimulusCoordinateContractError(
            "Canonical chaser_states requires schema_id "
            f"{CHASER_STATES_SCHEMA_ID!r} and schema_version "
            f"{CHASER_STATES_SCHEMA_VERSION}."
        )
    manifest, row_fields, surfaces = _load_manifest(attrs)
    structured_rows = _validate_source_rows_and_fields(
        dataset,
        manifest=manifest,
        row_fields=row_fields,
        surfaces=surfaces,
    )
    row_values, row_digest, row_attrs, row_contract = _load_source_row_identity(
        h5,
        rowset=dataset,
        expected_values=structured_rows,
        row_fields=row_fields,
    )
    (
        source_acquisition_frames,
        source_acquisition_record,
        source_acquisition_digest,
    ) = _load_source_acquisition_mapping(
        h5,
        row_count=int(dataset.shape[0]),
        row_identity_sha256=row_digest,
        row_identity_contract_sha256=row_contract.digest(),
    )
    target_field_names = {"target_source_frame_id", "target_source_camera_id"}
    source_field_names = set(dataset.dtype.names or ())
    target_mapping_paths_present = {
        path
        for path in (
            TARGET_SOURCE_ACQUISITION_MAPPING_ARRAY_PATH,
            TARGET_SOURCE_ACQUISITION_VALID_ARRAY_PATH,
        )
        if path in h5
    }
    target_source_frames: np.ndarray | None = None
    target_source_valid: np.ndarray | None = None
    target_source_record: dict[str, Any] | None = None
    target_source_digest: str | None = None
    if source_field_names.intersection(target_field_names):
        if not target_field_names.issubset(source_field_names):
            raise StimulusCoordinateContractError(
                "Canonical target provenance requires both target_source_frame_id "
                "and target_source_camera_id fields."
            )
        (
            target_source_frames,
            target_source_valid,
            target_source_record,
            target_source_digest,
        ) = _load_target_source_acquisition_mapping(
            h5,
            row_count=int(dataset.shape[0]),
            row_identity_sha256=row_digest,
            row_identity_contract_sha256=row_contract.digest(),
        )
    elif target_mapping_paths_present:
        raise StimulusCoordinateContractError(
            "Orphan target-source acquisition arrays exist without target-source "
            "fields in chaser_states."
        )
    descriptor = _load_descriptor(
        attrs,
        row_identity_contract=row_contract,
        # The Citrus structured dataset carries one descriptor template for
        # the point surfaces enumerated by its sealed manifest. Validate that
        # template against the materialized point shape, never the structured
        # rowset's rank-1 storage shape.
        owner_shape=(int(dataset.shape[0]), 2),
    )
    arena_record, arena_digest, arena_ref = _validate_source_arena(
        h5,
        source_h5=source,
        descriptor=descriptor,
    )
    dataset_digest = _h5_dataset_content_digest(dataset)
    evidence = {
        "source_file_identity": file_identity,
        "selected_calibration_sha256": selected_calibration.source_evidence_sha256,
        "has_chaser_states": True,
        "chaser_dataset_sha256": dataset_digest,
        "chaser_dataset_attrs_sha256": canonical_mapping_digest(attrs),
        "stimulus_state_key_sha256": row_digest,
        "stimulus_state_key_attrs_sha256": canonical_mapping_digest(row_attrs),
        "row_identity_contract_sha256": row_contract.digest(),
        "source_acquisition_frame_index_sha256": numpy_content_digest(
            source_acquisition_frames
        ),
        "source_acquisition_mapping_record_sha256": source_acquisition_digest,
        "target_source_acquisition_frame_index_sha256": (
            numpy_content_digest(target_source_frames)
            if target_source_frames is not None
            else None
        ),
        "target_source_acquisition_frame_valid_sha256": (
            numpy_content_digest(target_source_valid)
            if target_source_valid is not None
            else None
        ),
        "target_source_acquisition_mapping_record_sha256": target_source_digest,
        "arena_geometry_sha256": arena_digest,
        "renderer_snapshot_sha256": renderer_snapshot_sha256,
    }
    return StimulusCoordinatePreflight(
        source_h5=source,
        source_file_identity=file_identity,
        selected_calibration=selected_calibration,
        source_contract_sha256=canonical_mapping_digest(evidence),
        has_chaser_states=True,
        descriptor=descriptor,
        manifest=manifest,
        row_identity_contract=row_contract,
        row_identity_fields=row_fields,
        row_identity_values=row_values,
        row_identity_sha256=row_digest,
        row_identity_attrs=row_attrs,
        surfaces=surfaces,
        source_dataset_sha256=dataset_digest,
        source_arena_record=arena_record,
        source_arena_record_sha256=arena_digest,
        source_arena_record_ref=arena_ref,
        source_acquisition_frame_index=source_acquisition_frames,
        source_acquisition_mapping_record=source_acquisition_record,
        source_acquisition_mapping_record_sha256=source_acquisition_digest,
        target_source_acquisition_frame_index=target_source_frames,
        target_source_acquisition_frame_valid=target_source_valid,
        target_source_acquisition_mapping_record=target_source_record,
        target_source_acquisition_mapping_record_sha256=target_source_digest,
        renderer_snapshot=renderer_snapshot,
        renderer_snapshot_source_path=renderer_snapshot_source_path,
        renderer_snapshot_sha256=renderer_snapshot_sha256,
    )


def reverify_stimulus_coordinate_contract(
    h5: h5py.File,
    *,
    preflight: StimulusCoordinatePreflight,
) -> None:
    """Re-read the exact open source handle immediately before publication."""

    current = preflight_stimulus_coordinate_contract(
        h5,
        source_h5=preflight.source_h5,
        metadata_and_calibration_only=(
            preflight.source_coordinate_policy
            == SOURCE_COORDINATE_POLICY_METADATA_ONLY
        ),
    )
    if current.source_file_identity != preflight.source_file_identity:
        raise StimulusCoordinateContractError(
            "Open source H5 file identity changed after preflight."
        )
    if current.source_contract_sha256 != preflight.source_contract_sha256:
        raise StimulusCoordinateContractError(
            "Source H5 coordinate/calibration evidence changed after preflight."
        )


def _selected_reference_authority(
    run_group: zarr.Group,
    *,
    preflight: StimulusCoordinatePreflight,
) -> tuple[BoundReferenceExtent, BoundCoordinateRecord]:
    descriptor = preflight.descriptor
    assert descriptor is not None
    if descriptor.space_id != "arena_relative_canvas_px":
        raise StimulusCoordinateContractError(
            f"Unsupported canonical stimulus space {descriptor.space_id!r}."
        )
    calibration = run_group.get("calibration")
    arena = calibration.get("arena_geometry") if calibration is not None else None
    if not isinstance(arena, zarr.Group):
        raise StimulusCoordinateContractError(
            "Selected stimulus run lacks calibration/arena_geometry."
        )
    record = arena_geometry_record(dict(arena.attrs))
    digest = canonical_mapping_digest(record)
    if digest != preflight.source_arena_record_sha256:
        raise StimulusCoordinateContractError(
            "Selected arena_geometry snapshot differs from the verified source record."
        )
    bound_record = stamp_and_bind_persisted_coordinate_record(
        arena,
        record,
        attr_name=ARENA_GEOMETRY_RECORD_ATTR,
        digest_attr_name=ARENA_GEOMETRY_RECORD_DIGEST_ATTR,
    )
    reference = bind_persisted_record_reference_extent(
        arena,
        record_attr=ARENA_GEOMETRY_RECORD_ATTR,
        digest_attr=ARENA_GEOMETRY_RECORD_DIGEST_ATTR,
        width_field="arena_region_width_px",
        height_field="arena_region_height_px",
        units_field="units",
    )
    if (
        reference.record_sha256 != digest
        or reference.width != descriptor.reference_extent.width
        or reference.height != descriptor.reference_extent.height
    ):
        raise StimulusCoordinateContractError(
            "Selected arena extent differs from the verified source descriptor."
        )
    return reference, bound_record


def _output_rows(
    chaser_group: zarr.Group,
    *,
    fields: tuple[str, ...],
) -> tuple[np.ndarray, int]:
    columns: list[np.ndarray] = []
    row_count: int | None = None
    for field in fields:
        node = chaser_group.get(field)
        if node is None or len(node.shape) != 1 or np.dtype(node.dtype).kind not in "iu":
            raise StimulusCoordinateContractError(
                f"Output row field {field!r} must be a 1-D integer array."
            )
        if row_count is None:
            row_count = int(node.shape[0])
        elif int(node.shape[0]) != row_count:
            raise StimulusCoordinateContractError("Output row fields have mismatched lengths.")
        values = np.asarray(node[:])
        if values.dtype.kind == "u" and values.size and int(values.max()) > np.iinfo(np.int64).max:
            raise StimulusCoordinateContractError(f"Output row field {field!r} exceeds int64.")
        columns.append(np.asarray(values, dtype=np.int64))
    assert row_count is not None
    if len(columns) == 1:
        key = columns[0]
        unique = np.unique(key).shape[0]
    else:
        key = np.column_stack(columns)
        unique = np.unique(key, axis=0).shape[0]
    if unique != row_count:
        raise StimulusCoordinateContractError("Output row identity is not unique.")
    return key, row_count


def _exact_integer_array(node: Any, *, label: str) -> np.ndarray:
    if not isinstance(node, zarr.Array) or len(node.shape) != 1:
        raise StimulusCoordinateContractError(f"{label} must be a rank-1 array.")
    dtype = np.dtype(node.dtype)
    if dtype.kind not in "iu":
        raise StimulusCoordinateContractError(
            f"{label} must use an integer dtype; fractional mappings are forbidden."
        )
    values = np.asarray(node[:])
    if dtype.kind == "u" and values.size and int(values.max()) > np.iinfo(np.int64).max:
        raise StimulusCoordinateContractError(f"{label} exceeds signed int64.")
    result = np.asarray(values, dtype=np.int64)
    if result.size and int(result.min()) < 0:
        raise StimulusCoordinateContractError(f"{label} contains negative identifiers.")
    return result


def _resolve_exact_integer_field(
    group: zarr.Group,
    *,
    candidates: tuple[str, ...],
    label: str,
) -> tuple[str, zarr.Array, np.ndarray]:
    present = [name for name in candidates if name in group]
    if len(present) != 1:
        raise StimulusCoordinateContractError(
            f"{label} must resolve exactly one field from {list(candidates)!r}; "
            f"found {present!r}."
        )
    name = present[0]
    node = group[name]
    return name, node, _exact_integer_array(node, label=f"{label}/{name}")


def _camera_mapping_inputs(
    run_group: zarr.Group,
    chaser_group: zarr.Group,
    *,
    identity_components: tuple[str, ...],
) -> tuple[np.ndarray, dict[str, Any], tuple[Any, ...]]:
    video = run_group.get("video_metadata")
    metadata = video.get("frame_metadata") if isinstance(video, zarr.Group) else None
    if not isinstance(metadata, zarr.Group):
        raise StimulusCoordinateContractError(
            "Canonical camera mapping requires video_metadata/frame_metadata."
        )
    stimulus_name, stimulus_node, stimulus_values = _resolve_exact_integer_field(
        metadata,
        candidates=("stimulus_frame_num", "frame_number", "stim_frame_num"),
        label="video_metadata/frame_metadata stimulus identity",
    )
    camera_name, camera_node, camera_values = _resolve_exact_integer_field(
        metadata,
        candidates=("triggering_camera_frame_id", "camera_frame_id"),
        label="video_metadata/frame_metadata camera identity",
    )
    if stimulus_values.shape != camera_values.shape:
        raise StimulusCoordinateContractError(
            "Frame-metadata stimulus and camera fields have mismatched lengths."
        )
    unique_stimulus, stimulus_counts = np.unique(stimulus_values, return_counts=True)
    duplicate_stimulus = unique_stimulus[stimulus_counts > 1]
    if duplicate_stimulus.size:
        raise StimulusCoordinateContractError(
            "Frame metadata contains duplicate stimulus-frame mappings: "
            f"{duplicate_stimulus.tolist()!r}."
        )
    stimulus_components = [
        name
        for name in ("stimulus_frame_num", "frame_number", "stim_frame_num")
        if name in identity_components
    ]
    if len(stimulus_components) != 1:
        raise StimulusCoordinateContractError(
            "stimulus_state_key must contain exactly one stimulus-frame component."
        )
    row_component = stimulus_components[0]
    if row_component not in chaser_group:
        raise StimulusCoordinateContractError(
            f"chaser_states lacks identity component {row_component!r}."
        )
    row_stimulus = _exact_integer_array(
        chaser_group[row_component],
        label=f"tracking_data/chaser_states/{row_component}",
    )
    mapping = {
        int(stimulus): int(camera)
        for stimulus, camera in zip(stimulus_values, camera_values, strict=True)
    }
    missing = sorted({int(value) for value in row_stimulus if int(value) not in mapping})
    if missing:
        raise StimulusCoordinateContractError(
            "Stimulus-state rows lack exact camera-frame mappings: "
            f"{missing!r}."
        )
    camera_frame_ids = np.asarray(
        [mapping[int(value)] for value in row_stimulus],
        dtype=np.int64,
    )
    details = {
        "frame_metadata_rowset_ref": f"/{metadata.path}",
        "frame_metadata_stimulus_field": stimulus_name,
        "frame_metadata_stimulus_ref": f"/{stimulus_node.path}",
        "frame_metadata_stimulus_sha256": numpy_content_digest(stimulus_values),
        "frame_metadata_camera_field": camera_name,
        "frame_metadata_camera_ref": f"/{camera_node.path}",
        "frame_metadata_camera_sha256": numpy_content_digest(camera_values),
        "stimulus_frame_component": row_component,
    }
    return camera_frame_ids, details, (
        metadata,
        stimulus_node,
        camera_node,
        chaser_group[row_component],
    )


def _array_manifest_entry(
    node: zarr.Array,
    *,
    kind: str,
    semantic_role: str | None,
    components: tuple[str, ...],
    source_component_fields: tuple[str, ...] = (),
) -> dict[str, Any]:
    values = np.asarray(node[:])
    return {
        "array_ref": f"/{node.path}",
        "dtype": np.dtype(node.dtype).str,
        "shape": [int(value) for value in node.shape],
        "content_sha256": numpy_content_digest(values),
        "kind": kind,
        "semantic_role": semantic_role,
        "components": list(components),
        "source_component_fields": list(source_component_fields),
    }


def _coordinate_output_array_entries(
    chaser_group: zarr.Group,
    *,
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    expected_role_nodes = {
        str(surface["semantic_role"]): str(surface["array_name"])
        for surface in manifest["surfaces"]
    }
    actual_role_nodes: dict[str, str] = {}
    for child_name in chaser_group.keys():
        child = chaser_group.get(child_name)
        attrs = getattr(child, "attrs", None)
        if not isinstance(attrs, Mapping) or "semantic_role" not in attrs:
            continue
        role = attrs["semantic_role"]
        if not isinstance(role, str) or role not in expected_role_nodes:
            raise StimulusCoordinateContractError(
                f"Child {child_name!r} claims unsupported semantic_role {role!r}."
            )
        if role in actual_role_nodes:
            raise StimulusCoordinateContractError(
                f"semantic_role {role!r} is claimed by multiple output arrays."
            )
        actual_role_nodes[role] = str(child_name)
    if actual_role_nodes != expected_role_nodes:
        raise StimulusCoordinateContractError(
            "Output semantic-role claims do not exactly equal the surface manifest."
        )
    key_node = chaser_group.get(STIMULUS_STATE_KEY_ARRAY)
    camera_node = chaser_group.get(CAMERA_FRAME_IDS_ARRAY)
    source_rows_node = chaser_group.get(SOURCE_ROW_INDICES_ARRAY)
    source_acquisition_node = chaser_group.get(
        SOURCE_ACQUISITION_FRAME_INDEX_ARRAY
    )
    if not all(
        isinstance(node, zarr.Array)
        for node in (
            key_node,
            camera_node,
            source_rows_node,
            source_acquisition_node,
        )
    ):
        raise StimulusCoordinateContractError(
            "Canonical output is missing identity or camera-mapping arrays."
        )
    entries[STIMULUS_STATE_KEY_ARRAY] = _array_manifest_entry(
        key_node,
        kind="row_identity",
        semantic_role=None,
        components=(),
    )
    entries[CAMERA_FRAME_IDS_ARRAY] = _array_manifest_entry(
        camera_node,
        kind="camera_mapping",
        semantic_role=None,
        components=("camera_frame_id",),
    )
    entries[SOURCE_ROW_INDICES_ARRAY] = _array_manifest_entry(
        source_rows_node,
        kind="camera_mapping",
        semantic_role=None,
        components=("source_row_index",),
    )
    entries[SOURCE_ACQUISITION_FRAME_INDEX_ARRAY] = _array_manifest_entry(
        source_acquisition_node,
        kind="source_temporal_mapping",
        semantic_role=None,
        components=("acquisition_frame_index",),
    )
    target_source_node = chaser_group.get(
        TARGET_SOURCE_ACQUISITION_FRAME_INDEX_ARRAY
    )
    target_source_valid_node = chaser_group.get(
        TARGET_SOURCE_ACQUISITION_FRAME_VALID_ARRAY
    )
    if (target_source_node is None) != (target_source_valid_node is None):
        raise StimulusCoordinateContractError(
            "Target-source acquisition index and validity arrays must be paired."
        )
    if target_source_node is not None and target_source_valid_node is not None:
        if not isinstance(target_source_node, zarr.Array) or not isinstance(
            target_source_valid_node, zarr.Array
        ):
            raise StimulusCoordinateContractError(
                "Target-source acquisition provenance must use arrays."
            )
        entries[TARGET_SOURCE_ACQUISITION_FRAME_INDEX_ARRAY] = (
            _array_manifest_entry(
                target_source_node,
                kind="target_source_temporal_mapping",
                semantic_role=None,
                components=("acquisition_frame_index",),
            )
        )
        entries[TARGET_SOURCE_ACQUISITION_FRAME_VALID_ARRAY] = (
            _array_manifest_entry(
                target_source_valid_node,
                kind="target_source_temporal_mapping_validity",
                semantic_role=None,
                components=("valid",),
            )
        )
    for surface in manifest["surfaces"]:
        name = str(surface["array_name"])
        role = str(surface["semantic_role"])
        fields = tuple(str(value) for value in surface["component_fields"])
        point_node = chaser_group.get(name)
        component_nodes = tuple(chaser_group.get(field) for field in fields)
        if not isinstance(point_node, zarr.Array) or not all(
            isinstance(node, zarr.Array) for node in component_nodes
        ):
            raise StimulusCoordinateContractError(
                f"Canonical surface {name!r} or its components are missing."
            )
        point_values = np.asarray(point_node[:])
        component_values = tuple(np.asarray(node[:]) for node in component_nodes)
        stacked = np.column_stack(component_values)
        if point_values.dtype != stacked.dtype or not np.array_equal(
            point_values,
            stacked,
            equal_nan=True,
        ):
            raise StimulusCoordinateContractError(
                f"{name} does not exactly equal its scalar component arrays."
            )
        if point_node.attrs.get("semantic_role") != role:
            raise StimulusCoordinateContractError(
                f"{name} lacks its exact controlled semantic_role."
            )
        if point_node.attrs.get("source_component_fields") != list(fields):
            raise StimulusCoordinateContractError(
                f"{name} does not bind its exact scalar components."
            )
        entries[name] = _array_manifest_entry(
            point_node,
            kind="point_surface",
            semantic_role=role,
            components=("x", "y"),
            source_component_fields=fields,
        )
        for index, (field, node) in enumerate(zip(fields, component_nodes, strict=True)):
            component = ("x", "y")[index]
            if "semantic_role" in node.attrs:
                raise StimulusCoordinateContractError(
                    f"Scalar component {field!r} must not claim a point semantic_role."
                )
            if (
                node.attrs.get("parent_semantic_role") != role
                or node.attrs.get("coordinate_component") != component
                or node.attrs.get("coordinate_surface_array_ref") != name
            ):
                raise StimulusCoordinateContractError(
                    f"Scalar component {field!r} lacks exact parent-surface metadata."
                )
            entries[field] = _array_manifest_entry(
                node,
                kind="coordinate_component",
                semantic_role=role,
                components=(component,),
                source_component_fields=(field,),
            )
    return {name: entries[name] for name in sorted(entries)}


def validate_stimulus_destination_acquisition_authority(
    root_node: zarr.Group,
    *,
    preflight: StimulusCoordinatePreflight,
) -> Any:
    """Bind Citrus time evidence to the destination before run mutation."""

    if not preflight.has_chaser_states:
        return None
    source_record = preflight.source_acquisition_mapping_record
    if not isinstance(source_record, Mapping):
        raise StimulusCoordinateContractError(
            "Verified source acquisition-frame mapping is absent."
        )
    camera_id = str(source_record.get("acquisition_camera_id", ""))
    try:
        _, acquisition_frame = load_persisted_acquisition_camera_authority(
            root_node,
            expected_camera_id=camera_id,
        )
    except Exception as exc:
        raise StimulusCoordinateContractError(
            f"Canonical acquisition authority cannot bind stimulus time: {exc}."
        ) from exc
    acquisition = acquisition_frame.record
    selected_camera = preflight.selected_calibration.source_camera
    if (
        camera_id != preflight.selected_calibration.active_camera_id
        or acquisition.recording_id
        != source_record.get("acquisition_recording_id")
        or acquisition.camera_id != camera_id
        or acquisition.source_total_frames != source_record.get("source_total_frames")
        or acquisition.width_px != selected_camera.native_width_px
        or acquisition.height_px != selected_camera.native_height_px
    ):
        raise StimulusCoordinateContractError(
            "Citrus acquisition mapping, selected camera, and destination "
            "acquisition authority disagree."
        )
    return acquisition_frame


def materialize_stimulus_coordinate_contract(
    run_group: zarr.Group,
    *,
    root_node: zarr.Group,
    preflight: StimulusCoordinatePreflight,
    selected_calibration: SelectedCalibrationSnapshot,
) -> None:
    """Write canonical row, point, and scalar surfaces into one staged run."""

    run_group.attrs["coordinate_contract_epoch"] = COORDINATE_CONTRACT_EPOCH
    if not preflight.has_chaser_states:
        run_group.attrs["chaser_states_coordinate_descriptor_status"] = "not_present"
        return
    descriptor = preflight.descriptor
    manifest = preflight.manifest
    assert descriptor is not None and manifest is not None
    tracking = run_group.get("tracking_data")
    chaser = tracking.get("chaser_states") if tracking is not None else None
    if not isinstance(chaser, zarr.Group):
        raise StimulusCoordinateContractError("Staged run lacks chaser_states.")

    reference_extent, selected_arena_record = _selected_reference_authority(
        run_group,
        preflight=preflight,
    )
    row_key, row_count = _output_rows(
        chaser,
        fields=preflight.row_identity_fields,
    )
    source_row_key = preflight.row_identity_values
    if source_row_key is None or not np.array_equal(row_key, source_row_key):
        raise StimulusCoordinateContractError(
            "Staged chaser row fields do not exactly match the verified source "
            "stimulus_state_key array."
        )
    collisions = [
        name
        for name in (
            STIMULUS_STATE_KEY_ARRAY,
            CAMERA_FRAME_IDS_ARRAY,
            SOURCE_ROW_INDICES_ARRAY,
            SOURCE_ACQUISITION_FRAME_INDEX_ARRAY,
            TARGET_SOURCE_ACQUISITION_FRAME_INDEX_ARRAY,
            TARGET_SOURCE_ACQUISITION_FRAME_VALID_ARRAY,
        )
        if name in chaser
    ]
    if collisions:
        raise StimulusCoordinateContractError(
            f"Canonical identity/mapping array name collision: {collisions!r}."
        )
    if LEGACY_SOURCE_ROW_IDENTITY_ARRAY in chaser:
        raise StimulusCoordinateContractError(
            "Historical coordinate_row_identity must not be materialized in canonical output."
        )

    surface_values: dict[str, np.ndarray] = {}
    for surface in preflight.surfaces:
        name = str(surface["array_name"])
        if name in chaser:
            raise StimulusCoordinateContractError(f"Coordinate surface collision: {name!r}.")
        values: list[np.ndarray] = []
        for field in surface["component_fields"]:
            node = chaser.get(field)
            if (
                node is None
                or tuple(int(item) for item in node.shape) != (row_count,)
                or np.dtype(node.dtype).kind not in "fiu"
            ):
                raise StimulusCoordinateContractError(
                    f"Output coordinate field {field!r} is missing or invalid."
                )
            values.append(np.asarray(node[:]))
        surface_values[name] = np.column_stack(values)

    interpolation = {
        "operation": "fisheye.analysis.chaser_state_interpolator.interpolate_chaser_states",
        "algorithm_version": "chaser_state_interpolation_v1",
        "applied": False,
        "mask_ref": None,
        "mask_sha256": None,
    }
    mask = tracking.get("chaser_interpolation_mask")
    if mask is not None:
        values = np.asarray(mask[:], dtype=bool)
        if values.shape != (row_count,):
            raise StimulusCoordinateContractError("Interpolation mask length mismatch.")
        interpolation.update(
            {
                "applied": bool(np.any(~values)),
                "mask_ref": f"/{tracking.path}/chaser_interpolation_mask",
                "mask_sha256": numpy_content_digest(values),
            }
        )

    identity_contract = build_row_identity_contract(
        domain=STIMULUS_STATE_DOMAIN,
        values=source_row_key,
        components=preflight.row_identity_fields,
    )
    if (
        preflight.row_identity_contract is None
        or identity_contract != preflight.row_identity_contract
    ):
        raise StimulusCoordinateContractError(
            "Output stimulus-state identity differs from the verified source contract."
        )
    key_node = store_array(chaser, STIMULUS_STATE_KEY_ARRAY, source_row_key, {})
    bound_identity = stamp_and_bind_row_identity_contract(
        chaser,
        key_node,
        contract=identity_contract,
    )
    key_node.attrs.update(
        {
            "source_dataset_ref": (
                f"{preflight.source_h5}#/tracking_data/"
                f"{STIMULUS_STATE_KEY_ARRAY}"
            ),
            "source_content_sha256": preflight.row_identity_sha256,
            "source_row_identity_contract_sha256": (
                preflight.row_identity_contract.digest()
            ),
        }
    )
    source_acquisition_values = preflight.source_acquisition_frame_index
    source_acquisition_record = preflight.source_acquisition_mapping_record
    if (
        source_acquisition_values is None
        or source_acquisition_record is None
        or source_acquisition_values.shape != (row_count,)
    ):
        raise StimulusCoordinateContractError(
            "Verified source acquisition-frame mapping is absent or misaligned."
        )
    source_acquisition_node = store_array(
        chaser,
        SOURCE_ACQUISITION_FRAME_INDEX_ARRAY,
        np.asarray(source_acquisition_values, dtype=np.int64),
        {},
    )
    source_acquisition_node.attrs.update(
        {
            "temporal_role": "source_acquisition_frame_index",
            "target_domain": "acquisition_frame",
            "primary_row_identity": False,
        }
    )
    bound_source_acquisition_mapping = (
        stamp_and_bind_persisted_coordinate_record(
            source_acquisition_node,
            dict(source_acquisition_record),
            attr_name=SOURCE_ACQUISITION_MAPPING_RECORD_ATTR,
            digest_attr_name=(
                SOURCE_ACQUISITION_MAPPING_RECORD_DIGEST_ATTR
            ),
        )
    )
    if (
        bound_source_acquisition_mapping.record_sha256
        != preflight.source_acquisition_mapping_record_sha256
    ):
        raise StimulusCoordinateContractError(
            "Persisted source acquisition mapping differs from preflight."
        )
    acquisition_frame = validate_stimulus_destination_acquisition_authority(
        root_node,
        preflight=preflight,
    )
    try:
        source_temporal_authority = stamp_source_row_temporal_authority(
            chaser,
            source_acquisition_node,
            source_row_identity=bound_identity,
            acquisition_frame=acquisition_frame,
        )
    except Exception as exc:
        raise StimulusCoordinateContractError(
            f"Unable to seal stimulus acquisition-frame authority: {exc}."
        ) from exc

    target_source_node: zarr.Array | None = None
    target_source_valid_node: zarr.Array | None = None
    bound_target_source_mapping: BoundCoordinateRecord | None = None
    target_source_values = preflight.target_source_acquisition_frame_index
    target_source_valid = preflight.target_source_acquisition_frame_valid
    target_source_record = preflight.target_source_acquisition_mapping_record
    target_parts_present = (
        target_source_values is not None,
        target_source_valid is not None,
        target_source_record is not None,
    )
    if any(target_parts_present) and not all(target_parts_present):
        raise StimulusCoordinateContractError(
            "Verified target-source acquisition provenance is incomplete."
        )
    if all(target_parts_present):
        assert target_source_values is not None
        assert target_source_valid is not None
        assert target_source_record is not None
        if (
            target_source_values.shape != (row_count,)
            or target_source_valid.shape != (row_count,)
            or target_source_record.get("acquisition_recording_id")
            != source_acquisition_record.get("acquisition_recording_id")
            or target_source_record.get("acquisition_camera_id")
            != source_acquisition_record.get("acquisition_camera_id")
            or target_source_record.get("source_total_frames")
            != source_acquisition_record.get("source_total_frames")
        ):
            raise StimulusCoordinateContractError(
                "Target-source provenance does not bind the same acquisition "
                "authority and stimulus rows as state acquisition time."
            )
        target_source_node = store_array(
            chaser,
            TARGET_SOURCE_ACQUISITION_FRAME_INDEX_ARRAY,
            np.asarray(target_source_values, dtype=np.int64),
            {},
        )
        target_source_valid_node = store_array(
            chaser,
            TARGET_SOURCE_ACQUISITION_FRAME_VALID_ARRAY,
            np.asarray(target_source_valid, dtype=bool),
            {},
        )
        target_source_node.attrs.update(
            {
                "temporal_role": "target_detection_source_acquisition_frame_index",
                "target_domain": "acquisition_frame",
                "primary_row_identity": False,
            }
        )
        bound_target_source_mapping = (
            stamp_and_bind_persisted_coordinate_record(
                target_source_node,
                dict(target_source_record),
                attr_name=TARGET_SOURCE_ACQUISITION_MAPPING_RECORD_ATTR,
                digest_attr_name=(
                    TARGET_SOURCE_ACQUISITION_MAPPING_RECORD_DIGEST_ATTR
                ),
            )
        )
        if (
            bound_target_source_mapping.record_sha256
            != preflight.target_source_acquisition_mapping_record_sha256
        ):
            raise StimulusCoordinateContractError(
                "Persisted target-source mapping differs from preflight."
            )
        target_source_valid_node.attrs.update(
            {
                TARGET_SOURCE_ACQUISITION_MAPPING_RECORD_REF_ATTR: (
                    bound_target_source_mapping.record_ref
                ),
                TARGET_SOURCE_ACQUISITION_MAPPING_RECORD_DIGEST_ATTR: (
                    bound_target_source_mapping.record_sha256
                ),
                "temporal_role": "target_detection_source_mapping_validity",
                "primary_row_identity": False,
            }
        )

    surface_nodes: dict[str, zarr.Array] = {}
    component_nodes: dict[str, zarr.Array] = {}
    component_fields: list[str] = []
    for surface in preflight.surfaces:
        name = str(surface["array_name"])
        role = str(surface["semantic_role"])
        fields = tuple(str(value) for value in surface["component_fields"])
        point_node = store_array(chaser, name, surface_values[name], {})
        point_node.attrs.update(
            {
                "semantic_role": role,
                "source_component_fields": list(fields),
                COORDINATE_SURFACE_MANIFEST_DIGEST_ATTR: canonical_mapping_digest(manifest),
            }
        )
        surface_nodes[name] = point_node
        for index, field in enumerate(fields):
            scalar_node = chaser[field]
            if "semantic_role" in scalar_node.attrs:
                del scalar_node.attrs["semantic_role"]
            scalar_node.attrs.update(
                {
                    "parent_semantic_role": role,
                    "coordinate_component": ("x", "y")[index],
                    "coordinate_surface_array_ref": name,
                }
            )
            component_nodes[field] = scalar_node
            component_fields.append(field)

    surface_manifest_record = stamp_and_bind_persisted_coordinate_record(
        chaser,
        manifest,
        attr_name=COORDINATE_SURFACE_MANIFEST_ATTR,
        digest_attr_name=COORDINATE_SURFACE_MANIFEST_DIGEST_ATTR,
    )

    camera_frame_ids, mapping_details, mapping_nodes = _camera_mapping_inputs(
        run_group,
        chaser,
        identity_components=tuple(preflight.row_identity_fields),
    )
    source_row_indices = np.arange(row_count, dtype=np.int64)
    camera_node = store_array(chaser, CAMERA_FRAME_IDS_ARRAY, camera_frame_ids, {})
    source_rows_node = store_array(
        chaser,
        SOURCE_ROW_INDICES_ARRAY,
        source_row_indices,
        {},
    )
    try:
        optional_target_nodes = tuple(
            node
            for node in (target_source_node, target_source_valid_node)
            if node is not None
        )
        require_same_archive(
            run_group,
            chaser,
            key_node,
            camera_node,
            source_rows_node,
            source_acquisition_node,
            *optional_target_nodes,
            *mapping_nodes,
        )
    except ArchiveIdentityError as exc:
        raise StimulusCoordinateContractError(str(exc)) from exc
    camera_mapping_record = {
        "schema_id": CAMERA_MAPPING_SCHEMA_ID,
        "schema_version": CAMERA_MAPPING_SCHEMA_VERSION,
        "coordinate_contract_epoch": COORDINATE_CONTRACT_EPOCH,
        "method": "exact_stimulus_frame_to_camera_frame_lookup",
        "row_count": row_count,
        "row_identity_record_ref": bound_identity.record_ref,
        "row_identity_record_sha256": bound_identity.record_sha256,
        "stimulus_state_key_ref": f"/{key_node.path}",
        "stimulus_state_key_sha256": identity_array_content_sha256(source_row_key),
        **mapping_details,
        "camera_frame_ids_ref": f"/{camera_node.path}",
        "camera_frame_ids_sha256": numpy_content_digest(camera_frame_ids),
        "source_row_indices_ref": f"/{source_rows_node.path}",
        "source_row_indices_sha256": numpy_content_digest(source_row_indices),
        "source_acquisition_frame_index_ref": (
            f"/{source_acquisition_node.path}"
        ),
        "source_acquisition_frame_index_sha256": numpy_content_digest(
            np.asarray(source_acquisition_node[:])
        ),
        "source_acquisition_mapping_record_ref": (
            bound_source_acquisition_mapping.record_ref
        ),
        "source_acquisition_mapping_record_sha256": (
            bound_source_acquisition_mapping.record_sha256
        ),
        "source_row_temporal_authority_ref": (
            source_temporal_authority.record_ref
        ),
        "source_row_temporal_authority_sha256": (
            source_temporal_authority.record_sha256
        ),
    }
    bound_camera_mapping = stamp_and_bind_persisted_coordinate_record(
        chaser,
        camera_mapping_record,
        attr_name=CAMERA_MAPPING_RECORD_ATTR,
        digest_attr_name=CAMERA_MAPPING_RECORD_DIGEST_ATTR,
    )

    try:
        frame_transform = publish_stimulus_frame_transform_evidence(
            root_node,
            run_group,
            chaser,
            stimulus_run=selected_calibration.stimulus_run,
            selected_calibration=selected_calibration,
            arena_reference=reference_extent,
            arena_record=selected_arena_record,
            source_temporal_authority=source_temporal_authority,
        )
    except Exception as exc:
        raise StimulusCoordinateContractError(
            f"Unable to publish typed stimulus frame/transform evidence: {exc}."
        ) from exc

    output_entries = _coordinate_output_array_entries(chaser, manifest=manifest)
    lineage = {
        "schema_id": "palette.coordinate_import_lineage",
        "schema_version": 1,
        "coordinate_contract_epoch": COORDINATE_CONTRACT_EPOCH,
        "operation": "stimulus_h5_chaser_coordinate_import",
        "source_file_identity": dict(preflight.source_file_identity),
        "selected_calibration_source_evidence_sha256": (
            preflight.selected_calibration.source_evidence_sha256
        ),
        "source_dataset_ref": f"{preflight.source_h5}#/tracking_data/chaser_states",
        "source_dataset_sha256": preflight.source_dataset_sha256,
        "source_dataset_digest_canonicalization": "numpy_dtype_shape_c_order_bytes_v1",
        "source_row_identity_ref": (
            f"{preflight.source_h5}#/tracking_data/"
            f"{STIMULUS_STATE_KEY_ARRAY}"
        ),
        "source_row_identity_sha256": preflight.row_identity_sha256,
        "source_row_identity_contract": (
            preflight.row_identity_contract.to_dict()
        ),
        "source_row_identity_contract_sha256": (
            preflight.row_identity_contract.digest()
        ),
        "source_coordinate_descriptor": descriptor.to_dict(),
        "source_coordinate_descriptor_sha256": descriptor.digest(),
        "source_coordinate_surface_manifest": dict(manifest),
        "source_coordinate_surface_manifest_sha256": canonical_mapping_digest(manifest),
        "source_arena_geometry": dict(preflight.source_arena_record or {}),
        "source_arena_geometry_sha256": preflight.source_arena_record_sha256,
        "source_arena_geometry_ref": preflight.source_arena_record_ref,
        "selected_arena_geometry": selected_arena_record.record,
        "selected_arena_geometry_sha256": reference_extent.record_sha256,
        "selected_arena_geometry_ref": reference_extent.record_ref,
        "output_row_identity_ref": f"/{chaser.path}/{STIMULUS_STATE_KEY_ARRAY}",
        "output_row_identity_sha256": identity_array_content_sha256(source_row_key),
        "output_row_identity_contract_ref": bound_identity.record_ref,
        "output_row_identity_contract_sha256": identity_contract.digest(),
        "output_array_sha256": {
            name: entry["content_sha256"]
            for name, entry in output_entries.items()
        },
        "camera_mapping_record_ref": bound_camera_mapping.record_ref,
        "camera_mapping_record_sha256": bound_camera_mapping.record_sha256,
        "source_row_temporal_authority_ref": (
            source_temporal_authority.record_ref
        ),
        "source_row_temporal_authority_sha256": (
            source_temporal_authority.record_sha256
        ),
        "stimulus_frame_transform_manifest_ref": (
            frame_transform.manifest.record_ref
        ),
        "stimulus_frame_transform_manifest_sha256": (
            frame_transform.manifest.record_sha256
        ),
        "interpolation": interpolation,
    }
    if bound_target_source_mapping is not None:
        lineage["target_source_acquisition_mapping_record_ref"] = (
            bound_target_source_mapping.record_ref
        )
        lineage["target_source_acquisition_mapping_record_sha256"] = (
            bound_target_source_mapping.record_sha256
        )
    bound_lineage = stamp_and_bind_persisted_coordinate_record(
        chaser,
        lineage,
        attr_name=COORDINATE_IMPORT_LINEAGE_ATTR,
        digest_attr_name=COORDINATE_IMPORT_LINEAGE_DIGEST_ATTR,
    )
    output_records = {
        "surface_manifest": {
            "record_ref": surface_manifest_record.record_ref,
            "record_sha256": surface_manifest_record.record_sha256,
        },
        "camera_mapping": {
            "record_ref": bound_camera_mapping.record_ref,
            "record_sha256": bound_camera_mapping.record_sha256,
        },
        "frame_transform": {
            "record_ref": frame_transform.manifest.record_ref,
            "record_sha256": frame_transform.manifest.record_sha256,
        },
        "import_lineage": {
            "record_ref": bound_lineage.record_ref,
            "record_sha256": bound_lineage.record_sha256,
        },
    }
    if bound_target_source_mapping is not None:
        output_records["target_source_acquisition_mapping"] = {
            "record_ref": bound_target_source_mapping.record_ref,
            "record_sha256": bound_target_source_mapping.record_sha256,
        }
    output_manifest = {
        "schema_id": COORDINATE_OUTPUT_MANIFEST_SCHEMA_ID,
        "schema_version": COORDINATE_OUTPUT_MANIFEST_SCHEMA_VERSION,
        "coordinate_contract_epoch": COORDINATE_CONTRACT_EPOCH,
        "row_count": row_count,
        "row_identity": {
            "record_ref": bound_identity.record_ref,
            "record_sha256": bound_identity.record_sha256,
        },
        "reference_extent": {
            "record_ref": reference_extent.record_ref,
            "record_sha256": reference_extent.record_sha256,
            "selector": reference_extent.selector,
            "width": reference_extent.width,
            "height": reference_extent.height,
            "units": reference_extent.units,
        },
        "records": output_records,
        "arrays": output_entries,
    }
    bound_output_manifest = stamp_and_bind_persisted_coordinate_record(
        chaser,
        output_manifest,
        attr_name=COORDINATE_OUTPUT_MANIFEST_ATTR,
        digest_attr_name=COORDINATE_OUTPUT_MANIFEST_DIGEST_ATTR,
    )

    lineage_records = (
        surface_manifest_record,
        bound_camera_mapping,
        frame_transform.manifest,
        bound_lineage,
        bound_output_manifest,
        *(
            (bound_target_source_mapping,)
            if bound_target_source_mapping is not None
            else ()
        ),
    )
    descriptor_bindings: list[BoundCanonicalCoordinateDescriptor] = []
    for surface in preflight.surfaces:
        name = str(surface["array_name"])
        fields = tuple(str(value) for value in surface["component_fields"])
        descriptor_bindings.append(
            build_bound_canonical_coordinate_descriptor(
                surface_nodes[name],
                profile_id="arena_relative_canvas_px.top_left_y_down.v1",
                geometry_type="point_xy",
                components=("x", "y"),
                component_units=("px", "px"),
                pixel_convention=descriptor.pixel_convention,
                row_identity=bound_identity,
                reference_frame_authority=frame_transform.arena_relative_frame,
                source_camera_overlay_status="requires_transform",
                transform_chain=frame_transform.transform_chain,
                lineage_records=lineage_records,
            )
        )
        for index, field in enumerate(fields):
            descriptor_bindings.append(
                build_bound_canonical_coordinate_descriptor(
                    component_nodes[field],
                    profile_id="arena_relative_canvas_px.top_left_y_down.v1",
                    geometry_type="coordinate_component",
                    components=(("x", "y")[index],),
                    component_units=("px",),
                    pixel_convention=descriptor.pixel_convention,
                    row_identity=bound_identity,
                    reference_frame_authority=(
                        frame_transform.arena_relative_frame
                    ),
                    source_camera_overlay_status="requires_transform",
                    transform_chain=frame_transform.transform_chain,
                    lineage_records=lineage_records,
                )
            )
    stamp_bound_canonical_coordinate_descriptors(descriptor_bindings)

    for key in (
        COORDINATE_DESCRIPTOR_ATTR,
        f"{COORDINATE_DESCRIPTOR_ATTR}{COORDINATE_DESCRIPTOR_DIGEST_SUFFIX}",
    ):
        if key in chaser.attrs:
            del chaser.attrs[key]
    chaser.attrs[DESCRIPTOR_STATUS_ATTR] = "canonical"
    chaser.attrs["coordinate_descriptor_surface_arrays"] = [
        str(surface["array_name"]) for surface in preflight.surfaces
    ]
    chaser.attrs["coordinate_descriptor_component_fields"] = component_fields
    for key in (DESCRIPTOR_ISSUE_CODE_ATTR, DESCRIPTOR_ISSUE_MESSAGE_ATTR):
        if key in chaser.attrs:
            del chaser.attrs[key]
    run_group.attrs["chaser_states_coordinate_descriptor_status"] = "canonical"
    _load_bound_stimulus_coordinate_evidence_before_selection(
        run_group,
        chaser,
        root_node=root_node,
        require_complete=False,
    )


def _record_pointer(record: BoundCoordinateRecord) -> dict[str, str]:
    return {
        "record_ref": record.record_ref,
        "record_sha256": record.record_sha256,
    }


def _require_canonical_run_state(
    run_group: zarr.Group,
    *,
    require_complete: bool,
    require_selector_eligible: bool,
) -> None:
    raw_epoch = run_group.attrs.get("coordinate_contract_epoch")
    if (
        isinstance(raw_epoch, (bool, np.bool_))
        or not isinstance(raw_epoch, (int, np.integer))
        or int(raw_epoch) != COORDINATE_CONTRACT_EPOCH
    ):
        raise StimulusCoordinateContractError(
            "Stimulus run lacks the exact canonical coordinate-contract epoch."
        )
    if run_group.attrs.get("chaser_states_coordinate_descriptor_status") != "canonical":
        raise StimulusCoordinateContractError(
            "Stimulus run does not declare canonical chaser-state coordinates."
        )
    if not require_complete:
        return
    if (
        run_group.attrs.get(RUN_COMPLETION_CONTRACT_ATTR) != RUN_COMPLETION_CONTRACT
        or run_group.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
    ):
        raise StimulusCoordinateContractError(
            "Canonical stimulus coordinates may only be consumed from a complete run."
        )
    if run_group.attrs.get("import_version") != STIMULUS_IMPORT_VERSION:
        raise StimulusCoordinateContractError(
            "Complete stimulus run uses an unsupported canonical import version."
        )
    if (
        require_selector_eligible
        and run_group.attrs.get("stage_selector_eligible") is not True
    ):
        raise StimulusCoordinateContractError(
            "Canonical stimulus coordinates may only be consumed from a complete, "
            "explicitly selector-eligible run."
        )


def _load_bound_stimulus_coordinate_evidence_impl(
    run_group: zarr.Group,
    chaser_group: zarr.Group,
    *,
    root_node: zarr.Group,
    require_complete: bool,
    require_selector_eligible: bool,
) -> BoundStimulusCoordinateEvidence:
    """Bind and freshly verify one exact canonical stimulus-coordinate rowset."""

    if (
        not isinstance(root_node, zarr.Group)
        or not isinstance(run_group, zarr.Group)
        or not isinstance(chaser_group, zarr.Group)
    ):
        raise StimulusCoordinateContractError(
            "Canonical stimulus evidence requires exact persisted Zarr groups."
        )
    expected_chaser_path = f"{run_group.path}/tracking_data/chaser_states"
    if chaser_group.path != expected_chaser_path:
        raise StimulusCoordinateContractError(
            "chaser_states does not belong to the selected stimulus run."
        )
    _require_canonical_run_state(
        run_group,
        require_complete=require_complete,
        require_selector_eligible=require_selector_eligible,
    )
    if chaser_group.attrs.get(DESCRIPTOR_STATUS_ATTR) != "canonical":
        raise StimulusCoordinateContractError(
            "chaser_states is not a complete canonical coordinate rowset."
        )

    key_node = chaser_group.get(STIMULUS_STATE_KEY_ARRAY)
    camera_node = chaser_group.get(CAMERA_FRAME_IDS_ARRAY)
    source_rows_node = chaser_group.get(SOURCE_ROW_INDICES_ARRAY)
    source_acquisition_node = chaser_group.get(
        SOURCE_ACQUISITION_FRAME_INDEX_ARRAY
    )
    target_source_node = chaser_group.get(
        TARGET_SOURCE_ACQUISITION_FRAME_INDEX_ARRAY
    )
    target_source_valid_node = chaser_group.get(
        TARGET_SOURCE_ACQUISITION_FRAME_VALID_ARRAY
    )
    if (target_source_node is None) != (target_source_valid_node is None):
        raise StimulusCoordinateContractError(
            "Target-source acquisition index and validity arrays must be paired."
        )
    target_nodes = tuple(
        node
        for node in (target_source_node, target_source_valid_node)
        if node is not None
    )
    if target_nodes and not all(
        isinstance(node, zarr.Array) for node in target_nodes
    ):
        raise StimulusCoordinateContractError(
            "Target-source acquisition provenance must use arrays."
        )
    calibration = run_group.get("calibration")
    arena = calibration.get("arena_geometry") if isinstance(calibration, zarr.Group) else None
    if not all(
        isinstance(node, zarr.Array)
        for node in (
            key_node,
            camera_node,
            source_rows_node,
            source_acquisition_node,
        )
    ) or not isinstance(arena, zarr.Group):
        raise StimulusCoordinateContractError(
            "Canonical stimulus evidence is missing identity, mapping, or arena nodes."
        )
    try:
        common_archive = require_same_archive(
            run_group,
            chaser_group,
            root_node,
            key_node,
            camera_node,
            source_rows_node,
            source_acquisition_node,
            *target_nodes,
            arena,
        )
        bound_identity = load_bound_row_identity_contract(chaser_group, key_node)
        arena_record = bind_persisted_coordinate_record(
            arena,
            attr_name=ARENA_GEOMETRY_RECORD_ATTR,
            digest_attr_name=ARENA_GEOMETRY_RECORD_DIGEST_ATTR,
        )
        reference_extent = bind_persisted_record_reference_extent(
            arena,
            record_attr=ARENA_GEOMETRY_RECORD_ATTR,
            digest_attr=ARENA_GEOMETRY_RECORD_DIGEST_ATTR,
            width_field="arena_region_width_px",
            height_field="arena_region_height_px",
            units_field="units",
        )
        surface_manifest_record = bind_persisted_coordinate_record(
            chaser_group,
            attr_name=COORDINATE_SURFACE_MANIFEST_ATTR,
            digest_attr_name=COORDINATE_SURFACE_MANIFEST_DIGEST_ATTR,
        )
        camera_mapping = bind_persisted_coordinate_record(
            chaser_group,
            attr_name=CAMERA_MAPPING_RECORD_ATTR,
            digest_attr_name=CAMERA_MAPPING_RECORD_DIGEST_ATTR,
        )
        source_acquisition_mapping = bind_persisted_coordinate_record(
            source_acquisition_node,
            attr_name=SOURCE_ACQUISITION_MAPPING_RECORD_ATTR,
            digest_attr_name=SOURCE_ACQUISITION_MAPPING_RECORD_DIGEST_ATTR,
        )
        target_source_mapping = (
            bind_persisted_coordinate_record(
                target_source_node,
                attr_name=TARGET_SOURCE_ACQUISITION_MAPPING_RECORD_ATTR,
                digest_attr_name=(
                    TARGET_SOURCE_ACQUISITION_MAPPING_RECORD_DIGEST_ATTR
                ),
            )
            if isinstance(target_source_node, zarr.Array)
            else None
        )
        import_lineage = bind_persisted_coordinate_record(
            chaser_group,
            attr_name=COORDINATE_IMPORT_LINEAGE_ATTR,
            digest_attr_name=COORDINATE_IMPORT_LINEAGE_DIGEST_ATTR,
        )
        output_manifest_record = bind_persisted_coordinate_record(
            chaser_group,
            attr_name=COORDINATE_OUTPUT_MANIFEST_ATTR,
            digest_attr_name=COORDINATE_OUTPUT_MANIFEST_DIGEST_ATTR,
        )
        frame_transform = load_bound_stimulus_frame_transform_evidence(
            root_node,
            run_group,
            chaser_group,
            stimulus_run=run_group.path.rsplit("/", 1)[-1],
            row_identity=bound_identity,
        )
    except Exception as exc:
        raise StimulusCoordinateContractError(
            f"Canonical persisted stimulus evidence is invalid: {exc}"
        ) from exc

    bound_records = (
        surface_manifest_record,
        camera_mapping,
        source_acquisition_mapping,
        import_lineage,
        output_manifest_record,
        *((target_source_mapping,) if target_source_mapping is not None else ()),
    )
    if (
        bound_identity.archive_identity != common_archive
        or reference_extent.archive_identity != common_archive
        or arena_record.archive_identity != common_archive
        or frame_transform.archive_identity != common_archive
        or any(record.archive_identity != common_archive for record in bound_records)
    ):
        raise StimulusCoordinateContractError(
            "Canonical stimulus evidence crosses archive/store boundaries."
        )

    manifest, row_fields, surfaces = _load_manifest(dict(chaser_group.attrs))
    if surface_manifest_record.record != manifest:
        raise StimulusCoordinateContractError(
            "Bound coordinate surface manifest differs from canonical parsed content."
        )
    output_target_fields_present = {
        name
        for name in ("target_source_frame_id", "target_source_camera_id")
        if name in chaser_group
    }
    if output_target_fields_present and (
        output_target_fields_present
        != {"target_source_frame_id", "target_source_camera_id"}
        or target_source_mapping is None
    ):
        raise StimulusCoordinateContractError(
            "Canonical chaser target-source fields require their separate "
            "acquisition provenance mapping."
        )
    if not output_target_fields_present and target_source_mapping is not None:
        raise StimulusCoordinateContractError(
            "Orphan target-source acquisition provenance is present."
        )
    contract = bound_identity.contract
    if (
        contract.domain != STIMULUS_STATE_DOMAIN
        or contract.mode != STIMULUS_STATE_KEY_MODE
        or contract.key_array.ref != STIMULUS_STATE_KEY_ARRAY
        or contract.key_array.components != row_fields
    ):
        raise StimulusCoordinateContractError(
            "Stimulus row identity does not match the exact surface manifest."
        )
    row_values, row_count = _output_rows(chaser_group, fields=row_fields)
    key_values = np.asarray(key_node[:])
    if key_values.dtype.kind != "i" or key_values.dtype.itemsize != 8:
        raise StimulusCoordinateContractError(
            "Persisted stimulus_state_key must use signed int64."
        )
    if not np.array_equal(key_values, row_values):
        raise StimulusCoordinateContractError(
            "stimulus_state_key differs from its exact owning row fields."
        )

    derived_camera_ids, mapping_details, mapping_nodes = _camera_mapping_inputs(
        run_group,
        chaser_group,
        identity_components=row_fields,
    )
    try:
        require_same_archive(run_group, chaser_group, *mapping_nodes)
    except ArchiveIdentityError as exc:
        raise StimulusCoordinateContractError(str(exc)) from exc
    camera_values = _exact_integer_array(
        camera_node,
        label=f"/{camera_node.path}",
    )
    source_row_indices = _exact_integer_array(
        source_rows_node,
        label=f"/{source_rows_node.path}",
    )
    source_acquisition_values = _exact_integer_array(
        source_acquisition_node,
        label=f"/{source_acquisition_node.path}",
    )
    if (
        np.dtype(camera_node.dtype).kind != "i"
        or np.dtype(camera_node.dtype).itemsize != 8
        or np.dtype(source_rows_node.dtype).kind != "i"
        or np.dtype(source_rows_node.dtype).itemsize != 8
        or np.dtype(source_acquisition_node.dtype).kind != "i"
        or np.dtype(source_acquisition_node.dtype).itemsize != 8
    ):
        raise StimulusCoordinateContractError(
            "Camera mapping arrays must use signed int64."
        )
    source_acquisition_record = source_acquisition_mapping.record
    expected_source_acquisition_fields = {
        "schema_id",
        "schema_version",
        "mapping_method",
        "source_rowset_ref",
        "source_row_identity_ref",
        "source_row_identity_sha256",
        "source_row_identity_contract_sha256",
        "acquisition_recording_id",
        "acquisition_camera_id",
        "source_total_frames",
        "target_domain",
        "array_ref",
        "array_dtype",
        "array_shape",
        "array_content_sha256",
        "canonicalization",
    }
    acquisition_record = frame_transform.acquisition_frame.record
    if (
        set(source_acquisition_record) != expected_source_acquisition_fields
        or source_acquisition_record.get("schema_id")
        != SOURCE_ACQUISITION_MAPPING_SCHEMA_ID
        or source_acquisition_record.get("schema_version")
        != SOURCE_ACQUISITION_MAPPING_SCHEMA_VERSION
        or source_acquisition_record.get("mapping_method")
        != "explicit_per_stimulus_state_v1"
        or source_acquisition_record.get("source_rowset_ref")
        != "/tracking_data/chaser_states"
        or source_acquisition_record.get("source_row_identity_ref")
        != f"/tracking_data/{STIMULUS_STATE_KEY_ARRAY}"
        or source_acquisition_record.get("source_row_identity_sha256")
        != identity_array_content_sha256(key_values)
        or source_acquisition_record.get(
            "source_row_identity_contract_sha256"
        )
        != contract.digest()
        or source_acquisition_record.get("acquisition_recording_id")
        != acquisition_record.recording_id
        or source_acquisition_record.get("acquisition_camera_id")
        != acquisition_record.camera_id
        or source_acquisition_record.get("source_total_frames")
        != acquisition_record.source_total_frames
        or source_acquisition_record.get("target_domain")
        != "acquisition_frame_index"
        or source_acquisition_record.get("array_ref")
        != SOURCE_ACQUISITION_MAPPING_ARRAY_PATH
        or source_acquisition_record.get("array_dtype") != np.dtype("<i8").str
        or source_acquisition_record.get("array_shape") != [row_count]
        or source_acquisition_record.get("array_content_sha256")
        != numpy_content_digest(source_acquisition_values)
        or source_acquisition_record.get("canonicalization")
        != "canonical_json_sort_keys_v1"
        or (
            source_acquisition_values.size
            and int(source_acquisition_values.max())
            >= acquisition_record.source_total_frames
        )
        or source_acquisition_node.attrs.get("temporal_role")
        != "source_acquisition_frame_index"
        or source_acquisition_node.attrs.get("target_domain")
        != "acquisition_frame"
        or source_acquisition_node.attrs.get("primary_row_identity") is not False
    ):
        raise StimulusCoordinateContractError(
            "Persisted source acquisition mapping is stale or does not bind "
            "the exact row and acquisition authorities."
        )

    target_source_values: np.ndarray | None = None
    target_source_valid: np.ndarray | None = None
    if target_source_mapping is not None:
        assert isinstance(target_source_node, zarr.Array)
        assert isinstance(target_source_valid_node, zarr.Array)
        target_source_values = np.asarray(target_source_node[:])
        target_source_valid = np.asarray(target_source_valid_node[:])
        if (
            np.dtype(target_source_node.dtype) != np.dtype("<i8")
            or target_source_values.shape != (row_count,)
            or np.dtype(target_source_valid_node.dtype) != np.dtype("bool")
            or target_source_valid.shape != (row_count,)
        ):
            raise StimulusCoordinateContractError(
                "Persisted target-source acquisition arrays have invalid dtype or shape."
            )
        target_source_values = np.asarray(target_source_values, dtype=np.int64)
        target_source_valid = np.asarray(target_source_valid, dtype=bool)
        target_record = target_source_mapping.record
        expected_target_record = {
            "schema_id": TARGET_SOURCE_ACQUISITION_MAPPING_SCHEMA_ID,
            "schema_version": TARGET_SOURCE_ACQUISITION_MAPPING_SCHEMA_VERSION,
            "mapping_method": (
                "explicit_per_stimulus_state_target_provenance_v1"
            ),
            "source_rowset_ref": "/tracking_data/chaser_states",
            "source_row_identity_ref": (
                f"/tracking_data/{STIMULUS_STATE_KEY_ARRAY}"
            ),
            "source_row_identity_sha256": identity_array_content_sha256(
                key_values
            ),
            "source_row_identity_contract_sha256": contract.digest(),
            "source_target_frame_field": (
                "/tracking_data/chaser_states#target_source_frame_id"
            ),
            "source_target_camera_field": (
                "/tracking_data/chaser_states#target_source_camera_id"
            ),
            "acquisition_recording_id": acquisition_record.recording_id,
            "acquisition_camera_id": acquisition_record.camera_id,
            "source_total_frames": acquisition_record.source_total_frames,
            "target_domain": "acquisition_frame_index",
            "array_ref": TARGET_SOURCE_ACQUISITION_MAPPING_ARRAY_PATH,
            "array_dtype": np.dtype("<i8").str,
            "array_shape": [row_count],
            "array_content_sha256": numpy_content_digest(target_source_values),
            "validity_array_ref": TARGET_SOURCE_ACQUISITION_VALID_ARRAY_PATH,
            "validity_array_dtype": np.dtype("bool").str,
            "validity_array_shape": [row_count],
            "validity_array_content_sha256": numpy_content_digest(
                target_source_valid
            ),
            "invalid_index_sentinel": -1,
            "canonicalization": "canonical_json_sort_keys_v1",
        }
        if (
            target_record != expected_target_record
            or np.any(target_source_values[~target_source_valid] != -1)
            or np.any(target_source_values[target_source_valid] < 0)
            or np.any(
                target_source_values[target_source_valid]
                >= acquisition_record.source_total_frames
            )
            or target_source_node.attrs.get("temporal_role")
            != "target_detection_source_acquisition_frame_index"
            or target_source_node.attrs.get("target_domain")
            != "acquisition_frame"
            or target_source_node.attrs.get("primary_row_identity") is not False
            or target_source_valid_node.attrs.get(
                TARGET_SOURCE_ACQUISITION_MAPPING_RECORD_REF_ATTR
            )
            != target_source_mapping.record_ref
            or target_source_valid_node.attrs.get(
                TARGET_SOURCE_ACQUISITION_MAPPING_RECORD_DIGEST_ATTR
            )
            != target_source_mapping.record_sha256
            or target_source_valid_node.attrs.get("temporal_role")
            != "target_detection_source_mapping_validity"
            or target_source_valid_node.attrs.get("primary_row_identity") is not False
        ):
            raise StimulusCoordinateContractError(
                "Persisted target-source acquisition mapping is stale or does "
                "not bind the exact row and acquisition authorities."
            )
    expected_source_rows = np.arange(row_count, dtype=np.int64)
    if not np.array_equal(source_row_indices, expected_source_rows):
        raise StimulusCoordinateContractError(
            "source_row_indices must be the exact imported row mapping."
        )
    if not np.array_equal(camera_values, derived_camera_ids):
        raise StimulusCoordinateContractError(
            "Persisted camera_frame_ids differ from exact frame-metadata mapping."
        )
    expected_camera_record = {
        "schema_id": CAMERA_MAPPING_SCHEMA_ID,
        "schema_version": CAMERA_MAPPING_SCHEMA_VERSION,
        "coordinate_contract_epoch": COORDINATE_CONTRACT_EPOCH,
        "method": "exact_stimulus_frame_to_camera_frame_lookup",
        "row_count": row_count,
        "row_identity_record_ref": bound_identity.record_ref,
        "row_identity_record_sha256": bound_identity.record_sha256,
        "stimulus_state_key_ref": f"/{key_node.path}",
        "stimulus_state_key_sha256": identity_array_content_sha256(key_values),
        **mapping_details,
        "camera_frame_ids_ref": f"/{camera_node.path}",
        "camera_frame_ids_sha256": numpy_content_digest(camera_values),
        "source_row_indices_ref": f"/{source_rows_node.path}",
        "source_row_indices_sha256": numpy_content_digest(source_row_indices),
        "source_acquisition_frame_index_ref": (
            f"/{source_acquisition_node.path}"
        ),
        "source_acquisition_frame_index_sha256": numpy_content_digest(
            source_acquisition_values
        ),
        "source_acquisition_mapping_record_ref": (
            source_acquisition_mapping.record_ref
        ),
        "source_acquisition_mapping_record_sha256": (
            source_acquisition_mapping.record_sha256
        ),
        "source_row_temporal_authority_ref": (
            frame_transform.source_temporal_authority.record_ref
        ),
        "source_row_temporal_authority_sha256": (
            frame_transform.source_temporal_authority.record_sha256
        ),
    }
    if camera_mapping.record != expected_camera_record:
        raise StimulusCoordinateContractError(
            "Persisted camera-mapping record is stale or does not name exact inputs."
        )

    output_entries = _coordinate_output_array_entries(
        chaser_group,
        manifest=manifest,
    )
    lineage = import_lineage.record
    if (
        lineage.get("schema_id") != "palette.coordinate_import_lineage"
        or lineage.get("schema_version") != 1
        or lineage.get("coordinate_contract_epoch") != COORDINATE_CONTRACT_EPOCH
        or lineage.get("output_row_identity_contract_ref")
        != bound_identity.record_ref
        or lineage.get("output_row_identity_contract_sha256")
        != bound_identity.record_sha256
        or lineage.get("selected_arena_geometry_ref") != reference_extent.record_ref
        or lineage.get("selected_arena_geometry_sha256")
        != reference_extent.record_sha256
        or lineage.get("camera_mapping_record_ref") != camera_mapping.record_ref
        or lineage.get("camera_mapping_record_sha256")
        != camera_mapping.record_sha256
        or lineage.get("source_row_temporal_authority_ref")
        != frame_transform.source_temporal_authority.record_ref
        or lineage.get("source_row_temporal_authority_sha256")
        != frame_transform.source_temporal_authority.record_sha256
        or lineage.get("stimulus_frame_transform_manifest_ref")
        != frame_transform.manifest.record_ref
        or lineage.get("stimulus_frame_transform_manifest_sha256")
        != frame_transform.manifest.record_sha256
        or lineage.get("output_array_sha256")
        != {
            name: entry["content_sha256"]
            for name, entry in output_entries.items()
        }
        or (
            target_source_mapping is None
            and (
                "target_source_acquisition_mapping_record_ref" in lineage
                or "target_source_acquisition_mapping_record_sha256" in lineage
            )
        )
        or (
            target_source_mapping is not None
            and (
                lineage.get("target_source_acquisition_mapping_record_ref")
                != target_source_mapping.record_ref
                or lineage.get(
                    "target_source_acquisition_mapping_record_sha256"
                )
                != target_source_mapping.record_sha256
            )
        )
    ):
        raise StimulusCoordinateContractError(
            "Coordinate import lineage does not bind the exact published output."
        )

    expected_output_records = {
        "surface_manifest": _record_pointer(surface_manifest_record),
        "camera_mapping": _record_pointer(camera_mapping),
        "frame_transform": _record_pointer(frame_transform.manifest),
        "import_lineage": _record_pointer(import_lineage),
    }
    if target_source_mapping is not None:
        expected_output_records["target_source_acquisition_mapping"] = (
            _record_pointer(target_source_mapping)
        )
    expected_output_manifest = {
        "schema_id": COORDINATE_OUTPUT_MANIFEST_SCHEMA_ID,
        "schema_version": COORDINATE_OUTPUT_MANIFEST_SCHEMA_VERSION,
        "coordinate_contract_epoch": COORDINATE_CONTRACT_EPOCH,
        "row_count": row_count,
        "row_identity": {
            "record_ref": bound_identity.record_ref,
            "record_sha256": bound_identity.record_sha256,
        },
        "reference_extent": {
            "record_ref": reference_extent.record_ref,
            "record_sha256": reference_extent.record_sha256,
            "selector": reference_extent.selector,
            "width": reference_extent.width,
            "height": reference_extent.height,
            "units": reference_extent.units,
        },
        "records": expected_output_records,
        "arrays": output_entries,
    }
    if output_manifest_record.record != expected_output_manifest:
        raise StimulusCoordinateContractError(
            "Coordinate output manifest does not equal exact published array content."
        )

    lineage_records = (
        surface_manifest_record,
        camera_mapping,
        frame_transform.manifest,
        import_lineage,
        output_manifest_record,
        *((target_source_mapping,) if target_source_mapping is not None else ()),
    )
    for surface in surfaces:
        point_name = str(surface["array_name"])
        point_node = chaser_group[point_name]
        try:
            point_descriptor = load_bound_canonical_coordinate_descriptor(
                point_node,
                row_identity=bound_identity,
                reference_frame_authority=(
                    frame_transform.arena_relative_frame
                ),
                transform_chain=frame_transform.transform_chain,
                lineage_records=lineage_records,
            ).descriptor
        except Exception as exc:
            raise StimulusCoordinateContractError(
                f"Coordinate point surface {point_name!r} is not exactly bound: {exc}"
            ) from exc
        if (
            point_descriptor.profile_id
            != "arena_relative_canvas_px.top_left_y_down.v1"
            or point_descriptor.geometry_type != "point_xy"
            or point_descriptor.components != ("x", "y")
            or point_descriptor.component_units != ("px", "px")
            or point_descriptor.source_camera_overlay.status
            != "requires_transform"
        ):
            raise StimulusCoordinateContractError(
                f"Coordinate point surface {point_name!r} has invalid semantics."
            )
        for index, field in enumerate(surface["component_fields"]):
            try:
                component_descriptor = load_bound_canonical_coordinate_descriptor(
                    chaser_group[str(field)],
                    row_identity=bound_identity,
                    reference_frame_authority=(
                        frame_transform.arena_relative_frame
                    ),
                    transform_chain=frame_transform.transform_chain,
                    lineage_records=lineage_records,
                ).descriptor
            except Exception as exc:
                raise StimulusCoordinateContractError(
                    f"Coordinate scalar component {field!r} is not exactly bound: {exc}"
                ) from exc
            if (
                component_descriptor.geometry_type != "coordinate_component"
                or component_descriptor.components != (("x", "y")[index],)
                or component_descriptor.component_units != ("px",)
                or component_descriptor.source_camera_overlay.status
                != "requires_transform"
            ):
                raise StimulusCoordinateContractError(
                    f"Coordinate scalar component {field!r} has invalid semantics."
                )

    for values in (
        key_values,
        camera_values,
        source_row_indices,
        source_acquisition_values,
    ):
        values.setflags(write=False)
    if target_source_values is not None:
        target_source_values.setflags(write=False)
    if target_source_valid is not None:
        target_source_valid.setflags(write=False)
    return BoundStimulusCoordinateEvidence(
        archive_identity=archive_identity(run_group),
        row_identity=bound_identity,
        arena_reference=reference_extent,
        surface_manifest=surface_manifest_record,
        camera_mapping=camera_mapping,
        frame_transform=frame_transform,
        source_temporal_authority=frame_transform.source_temporal_authority,
        import_lineage=import_lineage,
        output_manifest=output_manifest_record,
        target_source_acquisition_mapping=target_source_mapping,
        stimulus_state_key=key_values,
        camera_frame_ids=camera_values,
        source_acquisition_frame_index=source_acquisition_values,
        source_row_indices=source_row_indices,
        target_source_acquisition_frame_index=target_source_values,
        target_source_acquisition_frame_valid=target_source_valid,
    )


def _load_bound_stimulus_coordinate_evidence_before_selection(
    run_group: zarr.Group,
    chaser_group: zarr.Group,
    *,
    root_node: zarr.Group,
    require_complete: bool,
) -> BoundStimulusCoordinateEvidence:
    """Validate a running or complete stimulus publication before selection."""

    if run_group.attrs.get("stage_selector_eligible") is not False:
        raise StimulusCoordinateContractError(
            "Pre-selection stimulus validation requires literal "
            "stage_selector_eligible=false."
        )
    return _load_bound_stimulus_coordinate_evidence_impl(
        run_group,
        chaser_group,
        root_node=root_node,
        require_complete=require_complete,
        require_selector_eligible=False,
    )


def load_bound_stimulus_coordinate_evidence(
    run_group: zarr.Group,
    chaser_group: zarr.Group,
    *,
    root_node: zarr.Group,
) -> BoundStimulusCoordinateEvidence:
    """Freshly verify a complete, explicitly selectable stimulus rowset."""

    return _load_bound_stimulus_coordinate_evidence_impl(
        run_group,
        chaser_group,
        root_node=root_node,
        require_complete=True,
        require_selector_eligible=True,
    )


__all__ = [
    "ARENA_GEOMETRY_RECORD_ATTR",
    "ARENA_GEOMETRY_RECORD_DIGEST_ATTR",
    "BoundStimulusCoordinateEvidence",
    "CAMERA_FRAME_IDS_ARRAY",
    "CAMERA_MAPPING_RECORD_ATTR",
    "CAMERA_MAPPING_RECORD_DIGEST_ATTR",
    "CAMERA_MAPPING_SCHEMA_ID",
    "CAMERA_MAPPING_SCHEMA_VERSION",
    "CHASER_STATES_SCHEMA_ID",
    "CHASER_STATES_SCHEMA_VERSION",
    "CHASER_STATES_V6_SCHEMA_VERSION",
    "COORDINATE_CONTRACT_EPOCH",
    "COORDINATE_IMPORT_LINEAGE_ATTR",
    "COORDINATE_IMPORT_LINEAGE_DIGEST_ATTR",
    "COORDINATE_OUTPUT_MANIFEST_ATTR",
    "COORDINATE_OUTPUT_MANIFEST_DIGEST_ATTR",
    "COORDINATE_OUTPUT_MANIFEST_SCHEMA_ID",
    "COORDINATE_OUTPUT_MANIFEST_SCHEMA_VERSION",
    "COORDINATE_SURFACE_MANIFEST_ATTR",
    "COORDINATE_SURFACE_MANIFEST_DIGEST_ATTR",
    "COORDINATE_SURFACE_MANIFEST_SCHEMA",
    "COORDINATE_SURFACE_MANIFEST_VERSION",
    "STIMULUS_IMPORT_VERSION",
    "STIMULUS_STATE_KEY_ARRAY",
    "SOURCE_ACQUISITION_FRAME_INDEX_ARRAY",
    "SOURCE_ACQUISITION_MAPPING_ARRAY_PATH",
    "SOURCE_ACQUISITION_MAPPING_RECORD_ATTR",
    "SOURCE_ACQUISITION_MAPPING_RECORD_DIGEST_ATTR",
    "SOURCE_ACQUISITION_MAPPING_SCHEMA_ID",
    "SOURCE_ACQUISITION_MAPPING_SCHEMA_VERSION",
    "SOURCE_ROW_INDICES_ARRAY",
    "SOURCE_ARENA_FRAME_ID",
    "SOURCE_COORDINATE_POLICY_CANONICAL",
    "SOURCE_COORDINATE_POLICY_METADATA_ONLY",
    "STIMULUS_RENDERER_SNAPSHOT_CAPTURE_PHASE",
    "STIMULUS_RENDERER_SNAPSHOT_PATH",
    "STIMULUS_RENDERER_SNAPSHOT_SCHEMA_ID",
    "STIMULUS_RENDERER_SNAPSHOT_SCHEMA_VERSION",
    "TARGET_SOURCE_ACQUISITION_FRAME_INDEX_ARRAY",
    "TARGET_SOURCE_ACQUISITION_FRAME_VALID_ARRAY",
    "TARGET_SOURCE_ACQUISITION_MAPPING_ARRAY_PATH",
    "TARGET_SOURCE_ACQUISITION_MAPPING_RECORD_ATTR",
    "TARGET_SOURCE_ACQUISITION_MAPPING_RECORD_DIGEST_ATTR",
    "TARGET_SOURCE_ACQUISITION_MAPPING_RECORD_REF_ATTR",
    "TARGET_SOURCE_ACQUISITION_MAPPING_SCHEMA_ID",
    "TARGET_SOURCE_ACQUISITION_MAPPING_SCHEMA_VERSION",
    "TARGET_SOURCE_ACQUISITION_VALID_ARRAY_PATH",
    "LEGACY_SOURCE_ROW_IDENTITY_ARRAY",
    "SourceSelectedCalibration",
    "StimulusCoordinateContractError",
    "StimulusCoordinatePreflight",
    "V6StimulusCoordinateArtifact",
    "arena_geometry_record",
    "canonical_mapping_digest",
    "load_bound_stimulus_coordinate_evidence",
    "materialize_stimulus_coordinate_contract",
    "numpy_content_digest",
    "preflight_stimulus_coordinate_contract",
    "reverify_stimulus_coordinate_contract",
    "source_arena_pixel_frame_record",
    "validate_stimulus_destination_acquisition_authority",
    "validate_citrus_stimulus_coordinate_v6_artifact",
]
