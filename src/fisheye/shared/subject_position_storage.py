"""Logical storage schema and in-memory validation for subject positions.

This module is deliberately independent of Zarr.  It validates the logical
arrays and the digest-bound metadata that a future immutable publisher will
write.  In particular, it does not open stores, inspect consolidated
metadata, or perform any publication.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Final, Mapping, Sequence

import numpy as np

from fisheye.shared.coordinate_descriptor import (
    CanonicalCoordinateDescriptor,
    CoordinateDescriptorError,
    PIXEL_FRAME_AUTHORITY_RECORD_KIND,
    canonical_coordinate_descriptor_v2_digest,
    parse_canonical_coordinate_descriptor,
    verify_canonical_coordinate_descriptor_identity,
)
from fisheye.shared.coordinate_identity import (
    OBSERVATION_INSTANCE_DOMAIN,
    RowIdentityContractError,
    build_row_identity_contract,
)
from fisheye.shared.coordinate_surface_contract import SOURCE_CAMERA_POINT_XY
from fisheye.shared.subject_position_types import (
    CANONICAL_FLOAT32_QNAN_BITS,
    OBSERVATION_POSITION_ROW_AXIS,
    POSITION_FAILURE_REASON_CODES,
    POSITION_FAILURE_REASON_PRECEDENCE,
    POSITION_FAILURE_REASON_TAGS,
    SOURCE_CAMERA_POSITION_PROFILE_ID,
    SUBJECT_POSITION_STORAGE_SCHEMA_ID,
    SUBJECT_POSITION_STORAGE_SCHEMA_VERSION,
    TRACK_SAMPLE_POSITION_ROW_AXIS,
)
from fisheye.shared.zarr.manifest_digest import (
    canonical_json_bytes,
    canonical_json_sha256,
)

OBSERVATION_POSITION_NAMESPACE: Final = (
    "analysis/subject_position_runs/observation/<run>"
)
TRACK_SAMPLE_POSITION_NAMESPACE: Final = (
    "analysis/subject_position_runs/track_sample/<run>"
)
POSITION_FAILURE_REASON_MAP_SCHEMA_ID: Final = (
    "palette.subject_position.failure_reason_map"
)
POSITION_FAILURE_REASON_MAP_SCHEMA_VERSION: Final = 1
POSITION_FAILURE_REASON_PRECEDENCE_SCHEMA_ID: Final = (
    "palette.subject_position.failure_reason_precedence"
)
POSITION_FAILURE_REASON_PRECEDENCE_SCHEMA_VERSION: Final = 1
CANONICAL_JSON_DIGEST_ALGORITHM: Final = "sha256_canonical_json_v1"

OBSERVATION_POSITION_MANDATORY_ARRAYS: Final = (
    "position_xy",
    "valid",
    "failure_reason_codes",
    "instance_key",
    "source_acquisition_frame_index",
    "source_row_index",
)
OBSERVATION_POSITION_OPTIONAL_ARRAYS: Final = (
    "support/source_points_xy",
    "support/source_points_valid",
    "support/source_point_reason_codes",
    "support/source_point_confidence",
)
OBSERVATION_POSITION_ARRAYS: Final = (
    *OBSERVATION_POSITION_MANDATORY_ARRAYS,
    *OBSERVATION_POSITION_OPTIONAL_ARRAYS,
)

TRACK_SAMPLE_POSITION_CORE_ARRAYS: Final = (
    "position_xy",
    "valid",
    "failure_reason_codes",
    "track_sample_key",
    "source_acquisition_frame_index",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FLOAT32_DTYPE = np.dtype(np.float32)
_BOOL_DTYPE = np.dtype(np.bool_)
_UINT16_DTYPE = np.dtype(np.uint16)
_UINT64_DTYPE = np.dtype(np.uint64)
_INT64_DTYPE = np.dtype(np.int64)


def _array_descriptor(
    *,
    path: str,
    dtype: str,
    shape: Sequence[object],
    meaning: str,
    required: bool,
) -> dict[str, object]:
    return {
        "path": path,
        "dtype": dtype,
        "shape": list(shape),
        "meaning": meaning,
        "required": required,
    }


def canonical_observation_position_schema_descriptor() -> dict[str, object]:
    """Return the canonical JSON-compatible observation schema descriptor."""

    return {
        "schema_id": SUBJECT_POSITION_STORAGE_SCHEMA_ID,
        "schema_version": SUBJECT_POSITION_STORAGE_SCHEMA_VERSION,
        "namespace": OBSERVATION_POSITION_NAMESPACE,
        "row_axis": OBSERVATION_POSITION_ROW_AXIS,
        "arrays": [
            _array_descriptor(
                path="position_xy",
                dtype="float32",
                shape=("N", 2),
                meaning="continuous source-camera x,y position",
                required=True,
            ),
            _array_descriptor(
                path="valid",
                dtype="bool",
                shape=("N",),
                meaning="estimator validity",
                required=True,
            ),
            _array_descriptor(
                path="failure_reason_codes",
                dtype="uint16",
                shape=("N",),
                meaning="controlled row-result reason code",
                required=True,
            ),
            _array_descriptor(
                path="instance_key",
                dtype="uint64",
                shape=("N",),
                meaning="observation identity",
                required=True,
            ),
            _array_descriptor(
                path="source_acquisition_frame_index",
                dtype="int64",
                shape=("N",),
                meaning="recording acquisition-frame identity",
                required=True,
            ),
            _array_descriptor(
                path="source_row_index",
                dtype="int64",
                shape=("N",),
                meaning="row offset in the immutable source snapshot",
                required=True,
            ),
            _array_descriptor(
                path="support/source_points_xy",
                dtype="float32",
                shape=("N", "P", 2),
                meaning="ordered contributing source points",
                required=False,
            ),
            _array_descriptor(
                path="support/source_points_valid",
                dtype="bool",
                shape=("N", "P"),
                meaning="per-anchor source validity",
                required=False,
            ),
            _array_descriptor(
                path="support/source_point_reason_codes",
                dtype="uint16",
                shape=("N", "P"),
                meaning="optional per-anchor reason code",
                required=False,
            ),
            _array_descriptor(
                path="support/source_point_confidence",
                dtype="float32",
                shape=("N", "P"),
                meaning="optional source confidence",
                required=False,
            ),
        ],
        "canonicalization": CANONICAL_JSON_DIGEST_ALGORITHM,
        "invalid_position": {
            "representation": "paired_float32_nan",
            "uint32_bits": f"0x{int(CANONICAL_FLOAT32_QNAN_BITS):08x}",
        },
    }


def observation_position_schema_descriptor() -> dict[str, object]:
    """Compatibility alias for the canonical observation descriptor."""

    return canonical_observation_position_schema_descriptor()


def canonical_track_sample_position_schema_descriptor() -> dict[str, object]:
    """Return the reserved future track-sample core descriptor.

    The descriptor is defined now so its row axis cannot accidentally be
    confused with observation rows.  The observation validator explicitly
    rejects this descriptor and its ``track_sample`` row axis.
    """

    return {
        "schema_id": SUBJECT_POSITION_STORAGE_SCHEMA_ID,
        "schema_version": SUBJECT_POSITION_STORAGE_SCHEMA_VERSION,
        "namespace": TRACK_SAMPLE_POSITION_NAMESPACE,
        "row_axis": TRACK_SAMPLE_POSITION_ROW_AXIS,
        "arrays": [
            _array_descriptor(
                path="position_xy",
                dtype="float32",
                shape=("N", 2),
                meaning="continuous source-camera x,y position",
                required=True,
            ),
            _array_descriptor(
                path="valid",
                dtype="bool",
                shape=("N",),
                meaning="estimator validity",
                required=True,
            ),
            _array_descriptor(
                path="failure_reason_codes",
                dtype="uint16",
                shape=("N",),
                meaning="controlled row-result reason code",
                required=True,
            ),
            _array_descriptor(
                path="track_sample_key",
                dtype="int64",
                shape=("N", 2),
                meaning="track identity and acquisition frame sample key",
                required=True,
            ),
            _array_descriptor(
                path="source_acquisition_frame_index",
                dtype="int64",
                shape=("N",),
                meaning="recording acquisition-frame identity",
                required=True,
            ),
        ],
        "canonicalization": CANONICAL_JSON_DIGEST_ALGORITHM,
        "invalid_position": {
            "representation": "paired_float32_nan",
            "uint32_bits": f"0x{int(CANONICAL_FLOAT32_QNAN_BITS):08x}",
        },
    }


def canonical_observation_position_schema_json() -> bytes:
    """Return canonical JSON bytes for the observation logical schema."""

    return canonical_json_bytes(canonical_observation_position_schema_descriptor())


def observation_position_schema_digest() -> str:
    """Return the digest bound by observation position manifests."""

    return canonical_json_sha256(canonical_observation_position_schema_descriptor())


def canonical_track_sample_position_schema_json() -> bytes:
    """Return canonical JSON bytes for the reserved track-sample schema."""

    return canonical_json_bytes(canonical_track_sample_position_schema_descriptor())


def track_sample_position_schema_digest() -> str:
    """Return the digest of the reserved track-sample core descriptor."""

    return canonical_json_sha256(canonical_track_sample_position_schema_descriptor())


def canonical_position_failure_reason_map() -> dict[str, object]:
    """Return the digest-bound controlled failure-reason map."""

    return {
        "schema_id": POSITION_FAILURE_REASON_MAP_SCHEMA_ID,
        "schema_version": POSITION_FAILURE_REASON_MAP_SCHEMA_VERSION,
        "codes": {
            tag: int(code)
            for tag, code in sorted(POSITION_FAILURE_REASON_CODES.items())
        },
        "tags": {
            str(int(code)): tag
            for code, tag in sorted(POSITION_FAILURE_REASON_TAGS.items())
        },
    }


def canonical_position_failure_reason_map_json() -> bytes:
    """Return canonical JSON bytes for the controlled reason-code map."""

    return canonical_json_bytes(canonical_position_failure_reason_map())


def position_failure_reason_map_digest() -> str:
    """Return the digest that must be bound by a position manifest."""

    return canonical_json_sha256(canonical_position_failure_reason_map())


def canonical_position_failure_reason_precedence() -> dict[str, object]:
    """Return the explicit highest-priority-first reason precedence."""

    return {
        "schema_id": POSITION_FAILURE_REASON_PRECEDENCE_SCHEMA_ID,
        "schema_version": POSITION_FAILURE_REASON_PRECEDENCE_SCHEMA_VERSION,
        "order": "highest_priority_first",
        "reason_tags": list(POSITION_FAILURE_REASON_PRECEDENCE),
    }


def canonical_position_failure_reason_precedence_json() -> bytes:
    return canonical_json_bytes(canonical_position_failure_reason_precedence())


def position_failure_reason_precedence_digest() -> str:
    return canonical_json_sha256(canonical_position_failure_reason_precedence())


def canonical_source_camera_coordinate_metadata(
    coordinate_descriptor: object,
) -> dict[str, object]:
    """Bind an existing canonical-v2 source-camera point descriptor.

    This helper never invents frame or extent authority.  The caller supplies
    a descriptor built from the repository's canonical coordinate and row-
    identity APIs; this function parses it and preserves its exact v2 digest.
    """

    descriptor = parse_canonical_coordinate_descriptor(coordinate_descriptor)
    return {
        "coordinate_descriptor": descriptor.to_dict(),
        "coordinate_descriptor_sha256": descriptor.digest(),
        "coordinate_surface_contract": SOURCE_CAMERA_POINT_XY.as_manifest(),
    }


def canonical_observation_position_logical_metadata(
    coordinate_metadata: object,
) -> dict[str, object]:
    """Build Phase 1 logical metadata, not a publication manifest.

    Immutable publication additionally requires the estimator, source arrays,
    source-schema/anatomy binding, policies, software, completion evidence,
    and physical storage plan defined by the storage contract.
    """

    descriptor = _coordinate_descriptor_from_metadata(coordinate_metadata)
    descriptor_digest = descriptor.digest()
    return {
        "storage_schema_id": SUBJECT_POSITION_STORAGE_SCHEMA_ID,
        "storage_schema_version": SUBJECT_POSITION_STORAGE_SCHEMA_VERSION,
        "storage_schema_sha256": observation_position_schema_digest(),
        "row_axis": OBSERVATION_POSITION_ROW_AXIS,
        "coordinate_descriptor": descriptor.to_dict(),
        "coordinate_descriptor_sha256": descriptor_digest,
        "coordinate_surface_contract": SOURCE_CAMERA_POINT_XY.as_manifest(),
        "reason_code_map": canonical_position_failure_reason_map(),
        "reason_code_map_sha256": position_failure_reason_map_digest(),
        "reason_precedence": canonical_position_failure_reason_precedence(),
        "reason_precedence_sha256": position_failure_reason_precedence_digest(),
    }


def canonical_observation_position_arrays_sha256(
    arrays: Mapping[str, np.ndarray],
) -> str:
    """Digest decoded logical array payloads in canonical schema order.

    A content digest exists only for a complete, logically schema-shaped array
    set.  Scientific validity and coordinate/manifest bindings remain the
    responsibility of :func:`validate_observation_position_arrays`.

    The digest includes dtype and shape and hashes exact C-order bytes,
    preserving canonical NaN payloads and rejecting accidental value-based NaN
    normalization.
    """

    if not isinstance(arrays, Mapping):
        raise TypeError("arrays must be a mapping of NumPy arrays.")
    paths = set(arrays)
    missing = set(OBSERVATION_POSITION_MANDATORY_ARRAYS) - paths
    if missing:
        raise ValueError(
            "Cannot digest a partial observation-position array set; missing "
            f"mandatory arrays: {sorted(missing)!r}."
        )
    unknown = paths - set(OBSERVATION_POSITION_ARRAYS)
    if unknown:
        raise ValueError(
            "Cannot digest unknown observation-position arrays: "
            f"{sorted(unknown)!r}."
        )
    support_xy = "support/source_points_xy" in paths
    support_valid = "support/source_points_valid" in paths
    if support_xy != support_valid:
        raise ValueError(
            "support/source_points_xy and support/source_points_valid must be "
            "both present or both absent before digesting."
        )
    if not support_xy and paths.intersection(
        {
            "support/source_point_reason_codes",
            "support/source_point_confidence",
        }
    ):
        raise ValueError(
            "Optional support reason/confidence arrays require the complete "
            "support coordinate/validity pair."
        )

    mandatory_specs = {
        "position_xy": (_FLOAT32_DTYPE, 2, (2,)),
        "valid": (_BOOL_DTYPE, 1, ()),
        "failure_reason_codes": (_UINT16_DTYPE, 1, ()),
        "instance_key": (_UINT64_DTYPE, 1, ()),
        "source_acquisition_frame_index": (_INT64_DTYPE, 1, ()),
        "source_row_index": (_INT64_DTYPE, 1, ()),
    }
    row_count: int | None = None
    for path, (dtype, ndim, trailing_shape) in mandatory_specs.items():
        value = arrays[path]
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{path} must be a NumPy array.")
        if value.dtype != dtype:
            raise ValueError(
                f"Cannot digest {path}: expected exact dtype {dtype}, "
                f"got {value.dtype}."
            )
        if value.ndim != ndim or tuple(value.shape[1:]) != trailing_shape:
            raise ValueError(
                f"Cannot digest {path}: expected rank {ndim} with trailing "
                f"shape {trailing_shape}, got {value.shape}."
            )
        if row_count is None:
            row_count = int(value.shape[0])
        elif value.shape[0] != row_count:
            raise ValueError(
                f"Cannot digest {path}: expected leading dimension N={row_count}, "
                f"got {value.shape[0]}."
            )

    if support_xy:
        support_coords = arrays["support/source_points_xy"]
        support_validity = arrays["support/source_points_valid"]
        if not isinstance(support_coords, np.ndarray):
            raise TypeError("support/source_points_xy must be a NumPy array.")
        if not isinstance(support_validity, np.ndarray):
            raise TypeError("support/source_points_valid must be a NumPy array.")
        if support_coords.dtype != _FLOAT32_DTYPE:
            raise ValueError(
                "Cannot digest support/source_points_xy: expected exact dtype "
                f"{_FLOAT32_DTYPE}, got {support_coords.dtype}."
            )
        if (
            support_coords.ndim != 3
            or support_coords.shape[0] != row_count
            or support_coords.shape[2:] != (2,)
        ):
            raise ValueError(
                "Cannot digest support/source_points_xy: expected shape "
                f"({row_count}, P, 2), got {support_coords.shape}."
            )
        point_count = int(support_coords.shape[1])
        if support_validity.dtype != _BOOL_DTYPE:
            raise ValueError(
                "Cannot digest support/source_points_valid: expected exact "
                f"dtype {_BOOL_DTYPE}, got {support_validity.dtype}."
            )
        if support_validity.ndim != 2 or support_validity.shape != (
            row_count,
            point_count,
        ):
            raise ValueError(
                "Cannot digest support/source_points_valid: expected shape "
                f"({row_count}, {point_count}), got {support_validity.shape}."
            )

        optional_support_specs = {
            "support/source_point_reason_codes": _UINT16_DTYPE,
            "support/source_point_confidence": _FLOAT32_DTYPE,
        }
        for path, dtype in optional_support_specs.items():
            if path not in arrays:
                continue
            value = arrays[path]
            if not isinstance(value, np.ndarray):
                raise TypeError(f"{path} must be a NumPy array.")
            if value.dtype != dtype:
                raise ValueError(
                    f"Cannot digest {path}: expected exact dtype {dtype}, "
                    f"got {value.dtype}."
                )
            if value.ndim != 2 or value.shape != (row_count, point_count):
                raise ValueError(
                    f"Cannot digest {path}: expected shape "
                    f"({row_count}, {point_count}), got {value.shape}."
                )

    entries: list[dict[str, object]] = []
    for path in OBSERVATION_POSITION_ARRAYS:
        if path not in arrays:
            continue
        value = arrays[path]
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{path} must be a NumPy array.")
        contiguous = np.ascontiguousarray(value)
        entries.append(
            {
                "path": path,
                "dtype": value.dtype.str,
                "shape": list(value.shape),
                "sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
            }
        )
    return canonical_json_sha256(
        {
            "schema_sha256": observation_position_schema_digest(),
            "arrays": entries,
        }
    )


def observation_position_arrays_sha256(
    arrays: Mapping[str, np.ndarray],
) -> str:
    """Compatibility alias for :func:`canonical_observation_position_arrays_sha256`."""

    return canonical_observation_position_arrays_sha256(arrays)


@dataclass(frozen=True)
class SubjectPositionStorageIssue:
    code: str
    path: str
    message: str


class SubjectPositionStorageValidationError(ValueError):
    """Raised when a logical subject-position publication is invalid."""

    def __init__(self, issues: Sequence[SubjectPositionStorageIssue]):
        self.issues = tuple(issues)
        if not self.issues:
            raise ValueError("At least one validation issue is required.")
        detail = "; ".join(
            f"{issue.code} at {issue.path}: {issue.message}" for issue in self.issues
        )
        super().__init__(
            f"Subject-position storage validation failed with "
            f"{len(self.issues)} issue(s): {detail}"
        )


SubjectPositionStorageError = SubjectPositionStorageValidationError


@dataclass(frozen=True)
class SubjectPositionStorageValidationReport:
    row_count: int
    support_point_count: int | None
    storage_schema_sha256: str
    coordinate_descriptor_sha256: str
    reason_code_map_sha256: str
    reason_precedence_sha256: str


def _issue(code: str, path: str, message: str) -> SubjectPositionStorageIssue:
    return SubjectPositionStorageIssue(code=code, path=path, message=message)


def _coordinate_descriptor_payload(coordinate_metadata: object) -> object:
    if isinstance(coordinate_metadata, CanonicalCoordinateDescriptor):
        return coordinate_metadata
    if not isinstance(coordinate_metadata, Mapping):
        raise TypeError(
            "coordinate_metadata must be a CanonicalCoordinateDescriptor or mapping."
        )
    if "coordinate_descriptor" in coordinate_metadata:
        return coordinate_metadata["coordinate_descriptor"]
    if "descriptor" in coordinate_metadata:
        return coordinate_metadata["descriptor"]
    return coordinate_metadata


def _coordinate_descriptor_from_metadata(
    coordinate_metadata: object,
) -> CanonicalCoordinateDescriptor:
    return parse_canonical_coordinate_descriptor(
        _coordinate_descriptor_payload(coordinate_metadata)
    )


def _declared_coordinate_digest(coordinate_metadata: object) -> object:
    if not isinstance(coordinate_metadata, Mapping):
        return None
    if (
        "coordinate_descriptor" not in coordinate_metadata
        and "descriptor" not in coordinate_metadata
    ):
        return None
    return coordinate_metadata.get(
        "coordinate_descriptor_sha256",
        coordinate_metadata.get("descriptor_sha256"),
    )


def _validate_coordinate_metadata(
    coordinate_metadata: object,
) -> tuple[
    list[SubjectPositionStorageIssue],
    CanonicalCoordinateDescriptor | None,
    str | None,
]:
    issues: list[SubjectPositionStorageIssue] = []
    try:
        descriptor = _coordinate_descriptor_from_metadata(coordinate_metadata)
    except CoordinateDescriptorError as exc:
        return (
            [
                _issue(
                    f"coordinate_descriptor_{item.code}",
                    f"$coordinate{item.path[1:]}",
                    item.message,
                )
                for item in exc.issues
            ],
            None,
            None,
        )
    except (TypeError, ValueError) as exc:
        return (
            [_issue("coordinate_descriptor_missing", "$coordinate", str(exc))],
            None,
            None,
        )

    descriptor_digest = canonical_coordinate_descriptor_v2_digest(descriptor)
    declared_digest = _declared_coordinate_digest(coordinate_metadata)
    if declared_digest is not None and (
        type(declared_digest) is not str
        or _SHA256_RE.fullmatch(declared_digest) is None
    ):
        issues.append(
            _issue(
                "coordinate_descriptor_digest_invalid",
                "$coordinate.coordinate_descriptor_sha256",
                "Coordinate descriptor digest must be a lowercase SHA-256 value.",
            )
        )
    elif declared_digest is not None and declared_digest != descriptor_digest:
        issues.append(
            _issue(
                "coordinate_descriptor_digest_mismatch",
                "$coordinate.coordinate_descriptor_sha256",
                "The declared digest differs from the exact canonical-v2 descriptor digest.",
            )
        )

    surface = SOURCE_CAMERA_POINT_XY
    expected_descriptor = surface.descriptor_kwargs()
    invariants = (
        ("profile_id", descriptor.profile_id, expected_descriptor["profile_id"]),
        ("space_id", descriptor.space_id, surface.domain_id),
        ("geometry_type", descriptor.geometry_type, surface.geometry_type),
        ("components", descriptor.components, surface.components),
        ("component_units", descriptor.component_units, surface.component_units),
        (
            "pixel_convention",
            descriptor.pixel_convention,
            surface.pixel_convention,
        ),
        (
            "source_camera_overlay.status",
            descriptor.source_camera_overlay.status,
            expected_descriptor["source_camera_overlay_status"],
        ),
    )
    for field, actual, expected in invariants:
        if actual != expected:
            issues.append(
                _issue(
                    "coordinate_surface_contract_mismatch",
                    f"$coordinate.{field}",
                    f"Expected SOURCE_CAMERA_POINT_XY invariant {expected!r}, got {actual!r}.",
                )
            )
    if descriptor.profile_id != SOURCE_CAMERA_POSITION_PROFILE_ID:
        issues.append(
            _issue(
                "coordinate_profile_mismatch",
                "$coordinate.profile_id",
                "Subject-position types and SOURCE_CAMERA_POINT_XY disagree on profile identity.",
            )
        )
    extent = descriptor.reference_extent
    if (
        type(extent.width) is not int
        or extent.width <= 0
        or type(extent.height) is not int
        or extent.height <= 0
        or extent.units != "px"
    ):
        issues.append(
            _issue(
                "coordinate_extent_invalid",
                "$coordinate.reference_extent",
                "SOURCE_CAMERA_POINT_XY requires positive exact integer width/height in px.",
            )
        )
    frame = descriptor.frame_record
    if frame is None or frame.kind != PIXEL_FRAME_AUTHORITY_RECORD_KIND:
        issues.append(
            _issue(
                "coordinate_frame_authority_missing",
                "$coordinate.frame_record",
                "SOURCE_CAMERA_POINT_XY requires canonical pixel-frame authority.",
            )
        )
    elif (
        frame.record_ref != extent.authority.record_ref
        or frame.record_sha256 != extent.authority.record_sha256
    ):
        issues.append(
            _issue(
                "coordinate_frame_authority_mismatch",
                "$coordinate.frame_record",
                "Frame record must be the exact source-camera extent authority.",
            )
        )

    outer_surface = (
        coordinate_metadata.get("coordinate_surface_contract")
        if isinstance(coordinate_metadata, Mapping)
        else None
    )
    if outer_surface is not None and outer_surface != surface.as_manifest():
        issues.append(
            _issue(
                "coordinate_surface_contract_mismatch",
                "$coordinate.coordinate_surface_contract",
                "Coordinate surface metadata differs from SOURCE_CAMERA_POINT_XY.",
            )
        )
    return issues, descriptor, descriptor_digest


def _validate_manifest_metadata(
    manifest_metadata: object,
    *,
    coordinate_descriptor: CanonicalCoordinateDescriptor | None,
    coordinate_digest: str | None,
) -> tuple[list[SubjectPositionStorageIssue], str | None, str | None]:
    issues: list[SubjectPositionStorageIssue] = []
    if not isinstance(manifest_metadata, Mapping):
        return (
            [
                _issue(
                    "manifest_metadata_not_mapping",
                    "$manifest",
                    "Manifest metadata must be a mapping.",
                )
            ],
            None,
            None,
        )

    expected_schema_digest = observation_position_schema_digest()
    actual_schema_id = manifest_metadata.get("storage_schema_id")
    actual_schema_version = manifest_metadata.get("storage_schema_version")
    actual_schema_digest = manifest_metadata.get("storage_schema_sha256")
    if actual_schema_id != SUBJECT_POSITION_STORAGE_SCHEMA_ID:
        issues.append(
            _issue(
                "storage_schema_id_mismatch",
                "$manifest.storage_schema_id",
                "Manifest is not the subject-position storage schema.",
            )
        )
    if actual_schema_version != SUBJECT_POSITION_STORAGE_SCHEMA_VERSION:
        issues.append(
            _issue(
                "storage_schema_version_mismatch",
                "$manifest.storage_schema_version",
                "Manifest has an unsupported storage-schema version.",
            )
        )
    if (
        type(actual_schema_digest) is not str
        or _SHA256_RE.fullmatch(actual_schema_digest) is None
    ):
        issues.append(
            _issue(
                "storage_schema_digest_missing",
                "$manifest.storage_schema_sha256",
                "Manifest must bind a lowercase schema SHA-256 digest.",
            )
        )
    elif actual_schema_digest != expected_schema_digest:
        issues.append(
            _issue(
                "storage_schema_digest_mismatch",
                "$manifest.storage_schema_sha256",
                "Manifest storage-schema digest is stale or incorrect.",
            )
        )
    if manifest_metadata.get("row_axis") != OBSERVATION_POSITION_ROW_AXIS:
        issues.append(
            _issue(
                "row_axis_mismatch",
                "$manifest.row_axis",
                "Observation validator accepts only observation_instance rows.",
            )
        )

    manifest_descriptor_raw = manifest_metadata.get("coordinate_descriptor")
    manifest_descriptor: CanonicalCoordinateDescriptor | None = None
    try:
        manifest_descriptor = parse_canonical_coordinate_descriptor(
            manifest_descriptor_raw
        )
    except CoordinateDescriptorError as exc:
        issues.append(
            _issue(
                "manifest_coordinate_descriptor_invalid",
                "$manifest.coordinate_descriptor",
                f"Manifest must contain canonical descriptor v2: {exc}",
            )
        )
    if (
        manifest_descriptor is not None
        and coordinate_descriptor is not None
        and manifest_descriptor.to_dict() != coordinate_descriptor.to_dict()
    ):
        issues.append(
            _issue(
                "manifest_coordinate_descriptor_mismatch",
                "$manifest.coordinate_descriptor",
                "Manifest coordinate descriptor differs from coordinate metadata.",
            )
        )
    manifest_coordinate_digest = manifest_metadata.get("coordinate_descriptor_sha256")
    if (
        type(manifest_coordinate_digest) is not str
        or _SHA256_RE.fullmatch(manifest_coordinate_digest) is None
    ):
        issues.append(
            _issue(
                "manifest_coordinate_digest_missing",
                "$manifest.coordinate_descriptor_sha256",
                "Manifest must bind a coordinate descriptor digest.",
            )
        )
    elif (
        manifest_descriptor is not None
        and manifest_coordinate_digest
        != canonical_coordinate_descriptor_v2_digest(manifest_descriptor)
    ):
        issues.append(
            _issue(
                "manifest_coordinate_digest_mismatch",
                "$manifest.coordinate_descriptor_sha256",
                "Manifest coordinate digest does not match its descriptor.",
            )
        )
    if (
        coordinate_digest is not None
        and manifest_coordinate_digest != coordinate_digest
    ):
        issues.append(
            _issue(
                "coordinate_digest_binding_mismatch",
                "$manifest.coordinate_descriptor_sha256",
                "Manifest and coordinate metadata bind different descriptors.",
            )
        )
    if (
        manifest_metadata.get("coordinate_surface_contract")
        != SOURCE_CAMERA_POINT_XY.as_manifest()
    ):
        issues.append(
            _issue(
                "coordinate_surface_contract_mismatch",
                "$manifest.coordinate_surface_contract",
                "Manifest must bind the exact SOURCE_CAMERA_POINT_XY surface contract.",
            )
        )
    expected_reason_digest = position_failure_reason_map_digest()
    reason_map = manifest_metadata.get("reason_code_map")
    if reason_map != canonical_position_failure_reason_map():
        issues.append(
            _issue(
                "reason_code_map_mismatch",
                "$manifest.reason_code_map",
                "Manifest reason-code map is not the controlled v1 map.",
            )
        )
    actual_reason_digest = manifest_metadata.get("reason_code_map_sha256")
    if (
        type(actual_reason_digest) is not str
        or _SHA256_RE.fullmatch(actual_reason_digest) is None
    ):
        issues.append(
            _issue(
                "reason_code_map_digest_missing",
                "$manifest.reason_code_map_sha256",
                "Manifest must bind a lowercase reason-code-map SHA-256 digest.",
            )
        )
    elif (
        actual_reason_digest != expected_reason_digest
        or actual_reason_digest != canonical_json_sha256(reason_map)
    ):
        issues.append(
            _issue(
                "reason_code_map_digest_mismatch",
                "$manifest.reason_code_map_sha256",
                "Manifest reason-code-map digest is stale or incorrect.",
            )
        )
    expected_precedence = canonical_position_failure_reason_precedence()
    precedence = manifest_metadata.get("reason_precedence")
    if precedence != expected_precedence:
        issues.append(
            _issue(
                "reason_precedence_mismatch",
                "$manifest.reason_precedence",
                "Manifest reason precedence must match the shared highest-priority-first order.",
            )
        )
    expected_precedence_digest = position_failure_reason_precedence_digest()
    actual_precedence_digest = manifest_metadata.get("reason_precedence_sha256")
    if (
        type(actual_precedence_digest) is not str
        or _SHA256_RE.fullmatch(actual_precedence_digest) is None
    ):
        issues.append(
            _issue(
                "reason_precedence_digest_missing",
                "$manifest.reason_precedence_sha256",
                "Manifest must bind a lowercase reason-precedence SHA-256 digest.",
            )
        )
    elif (
        actual_precedence_digest != expected_precedence_digest
        or actual_precedence_digest != canonical_json_sha256(precedence)
    ):
        issues.append(
            _issue(
                "reason_precedence_digest_mismatch",
                "$manifest.reason_precedence_sha256",
                "Manifest reason-precedence digest is stale or incorrect.",
            )
        )
    return (
        issues,
        actual_schema_digest if isinstance(actual_schema_digest, str) else None,
        actual_reason_digest if isinstance(actual_reason_digest, str) else None,
    )


def _require_array(
    arrays: Mapping[str, object],
    path: str,
    *,
    dtype: np.dtype,
    ndim: int,
    shape_suffix: tuple[int, ...],
    issues: list[SubjectPositionStorageIssue],
) -> np.ndarray | None:
    value = arrays.get(path)
    if value is None:
        issues.append(_issue("missing_array", path, "Mandatory array is absent."))
        return None
    if not isinstance(value, np.ndarray):
        issues.append(_issue("array_not_numpy", path, "Array must be a NumPy ndarray."))
        return None
    if value.dtype != dtype:
        issues.append(
            _issue(
                "array_dtype_mismatch",
                path,
                f"Expected exact dtype {dtype}, got {value.dtype}.",
            )
        )
    if value.ndim != ndim or tuple(value.shape[1:]) != shape_suffix:
        issues.append(
            _issue(
                "array_shape_mismatch",
                path,
                f"Expected rank {ndim} with trailing shape {shape_suffix}, got {value.shape}.",
            )
        )
    return value


def _validate_support_arrays(
    arrays: Mapping[str, object],
    *,
    row_count: int,
    issues: list[SubjectPositionStorageIssue],
) -> int | None:
    coordinate_path = "support/source_points_xy"
    valid_path = "support/source_points_valid"
    coords_present = coordinate_path in arrays
    valid_present = valid_path in arrays
    support_present = coords_present or valid_present
    if coords_present != valid_present:
        issues.append(
            _issue(
                "support_pair_missing",
                "support",
                "source_points_xy and source_points_valid must be supplied together.",
            )
        )
    if not support_present:
        for path in (
            "support/source_point_reason_codes",
            "support/source_point_confidence",
        ):
            if path in arrays:
                issues.append(
                    _issue(
                        "support_dependency_missing",
                        path,
                        "Optional support evidence requires the coordinate/validity pair.",
                    )
                )
        return None

    coords = arrays.get(coordinate_path)
    valid = arrays.get(valid_path)
    if not isinstance(coords, np.ndarray) or not isinstance(valid, np.ndarray):
        return None
    if coords.dtype != _FLOAT32_DTYPE:
        issues.append(
            _issue(
                "array_dtype_mismatch",
                coordinate_path,
                "Support coordinates require exact float32 dtype.",
            )
        )
    if valid.dtype != _BOOL_DTYPE:
        issues.append(
            _issue(
                "array_dtype_mismatch",
                valid_path,
                "Support validity requires exact bool dtype.",
            )
        )
    if coords.ndim != 3 or coords.shape[0] != row_count or coords.shape[2:] != (2,):
        issues.append(
            _issue(
                "support_shape_mismatch",
                coordinate_path,
                f"Expected float32[N,P,2] with N={row_count}, got {coords.shape}.",
            )
        )
        return None
    point_count = int(coords.shape[1])
    support_shape_valid = valid.ndim == 2 and valid.shape == (row_count, point_count)
    if not support_shape_valid:
        issues.append(
            _issue(
                "support_shape_mismatch",
                valid_path,
                f"Expected bool[{row_count},{point_count}], got {valid.shape}.",
            )
        )
    elif coords.dtype == _FLOAT32_DTYPE and valid.dtype == _BOOL_DTYPE:
        valid_mask = valid
        bits = coords.view(np.uint32)
        finite = np.isfinite(coords).all(axis=2)
        canonical_nan = np.all(bits == CANONICAL_FLOAT32_QNAN_BITS, axis=2)
        if np.any(valid_mask & ~finite):
            issues.append(
                _issue(
                    "support_valid_coordinate_nonfinite",
                    coordinate_path,
                    "Valid support points must have two finite coordinates.",
                )
            )
        if np.any(~valid_mask & ~canonical_nan):
            issues.append(
                _issue(
                    "support_invalid_coordinate_not_canonical_nan",
                    coordinate_path,
                    "Invalid support points must use paired canonical float32 NaNs.",
                )
            )

    reason_path = "support/source_point_reason_codes"
    if reason_path in arrays:
        reasons = arrays[reason_path]
        if not isinstance(reasons, np.ndarray):
            issues.append(
                _issue(
                    "array_not_numpy",
                    reason_path,
                    "Support reason codes must be a NumPy ndarray.",
                )
            )
        else:
            if reasons.dtype != _UINT16_DTYPE:
                issues.append(
                    _issue(
                        "array_dtype_mismatch",
                        reason_path,
                        "Support reason codes require exact uint16 dtype.",
                    )
                )
            if reasons.ndim != 2 or reasons.shape != (row_count, point_count):
                issues.append(
                    _issue(
                        "support_shape_mismatch",
                        reason_path,
                        f"Expected uint16[{row_count},{point_count}], got {reasons.shape}.",
                    )
                )
            elif (
                reasons.dtype == _UINT16_DTYPE
                and valid.dtype == _BOOL_DTYPE
                and valid.ndim == 2
                and valid.shape == reasons.shape
            ):
                known = np.isin(
                    reasons, np.fromiter(POSITION_FAILURE_REASON_TAGS, dtype=np.uint16)
                )
                if np.any(~known):
                    issues.append(
                        _issue(
                            "unknown_reason_code",
                            reason_path,
                            "Support reason codes contain an unknown controlled code.",
                        )
                    )
                if np.any(valid & (reasons != 0)) or np.any(~valid & (reasons == 0)):
                    issues.append(
                        _issue(
                            "support_validity_reason_mismatch",
                            reason_path,
                            "Support validity and reason codes must agree.",
                        )
                    )

    confidence_path = "support/source_point_confidence"
    if confidence_path in arrays:
        confidence = arrays[confidence_path]
        if not isinstance(confidence, np.ndarray):
            issues.append(
                _issue(
                    "array_not_numpy",
                    confidence_path,
                    "Support confidence must be a NumPy ndarray.",
                )
            )
        else:
            if confidence.dtype != _FLOAT32_DTYPE:
                issues.append(
                    _issue(
                        "array_dtype_mismatch",
                        confidence_path,
                        "Support confidence requires exact float32 dtype.",
                    )
                )
            if confidence.ndim != 2 or confidence.shape != (row_count, point_count):
                issues.append(
                    _issue(
                        "support_shape_mismatch",
                        confidence_path,
                        f"Expected float32[{row_count},{point_count}], got {confidence.shape}.",
                    )
                )
            elif (
                confidence.dtype == _FLOAT32_DTYPE and not np.isfinite(confidence).all()
            ):
                issues.append(
                    _issue(
                        "support_confidence_nonfinite",
                        confidence_path,
                        "Support confidence must be finite when supplied.",
                    )
                )
    return point_count


def collect_observation_position_storage_issues(
    arrays: Mapping[str, object],
    *,
    coordinate_metadata: object,
    manifest_metadata: Mapping[str, object],
) -> tuple[SubjectPositionStorageIssue, ...]:
    """Collect all pure logical validation issues without opening Zarr."""

    issues: list[SubjectPositionStorageIssue] = []
    if not isinstance(arrays, Mapping):
        return (
            _issue(
                "arrays_not_mapping", "$arrays", "Arrays must be supplied as a mapping."
            ),
        )
    expected_paths = set(OBSERVATION_POSITION_ARRAYS)
    for path in sorted(set(arrays) - expected_paths):
        issues.append(
            _issue(
                "unexpected_array",
                path,
                "Array is not declared by observation position schema v1.",
            )
        )

    coordinate_issues, descriptor, coordinate_digest = _validate_coordinate_metadata(
        coordinate_metadata
    )
    issues.extend(coordinate_issues)
    manifest_issues, _, _ = _validate_manifest_metadata(
        manifest_metadata,
        coordinate_descriptor=descriptor,
        coordinate_digest=coordinate_digest,
    )
    issues.extend(manifest_issues)

    position = _require_array(
        arrays,
        "position_xy",
        dtype=_FLOAT32_DTYPE,
        ndim=2,
        shape_suffix=(2,),
        issues=issues,
    )
    valid = _require_array(
        arrays, "valid", dtype=_BOOL_DTYPE, ndim=1, shape_suffix=(), issues=issues
    )
    reasons = _require_array(
        arrays,
        "failure_reason_codes",
        dtype=_UINT16_DTYPE,
        ndim=1,
        shape_suffix=(),
        issues=issues,
    )
    instance_key = _require_array(
        arrays,
        "instance_key",
        dtype=_UINT64_DTYPE,
        ndim=1,
        shape_suffix=(),
        issues=issues,
    )
    frame_index = _require_array(
        arrays,
        "source_acquisition_frame_index",
        dtype=_INT64_DTYPE,
        ndim=1,
        shape_suffix=(),
        issues=issues,
    )
    source_row = _require_array(
        arrays,
        "source_row_index",
        dtype=_INT64_DTYPE,
        ndim=1,
        shape_suffix=(),
        issues=issues,
    )

    row_count: int | None = None
    mandatory_values = (position, valid, reasons, instance_key, frame_index, source_row)
    for value in mandatory_values:
        if value is not None and value.ndim >= 1:
            if row_count is None:
                row_count = int(value.shape[0])
            elif value.shape[0] != row_count:
                issues.append(
                    _issue(
                        "leading_dimension_mismatch",
                        "$arrays",
                        "All mandatory arrays must have the same leading dimension N.",
                    )
                )
    if row_count is None:
        row_count = 0

    if (
        instance_key is not None
        and instance_key.dtype == _UINT64_DTYPE
        and instance_key.ndim == 1
        and instance_key.shape[0] == row_count
    ):
        if np.unique(instance_key).size != instance_key.size:
            issues.append(
                _issue(
                    "duplicate_instance_key",
                    "instance_key",
                    "instance_key values must be unique within the observation run.",
                )
            )
        elif (
            descriptor is not None
            and position is not None
            and position.dtype == _FLOAT32_DTYPE
            and position.ndim == 2
            and position.shape == (row_count, 2)
        ):
            try:
                exact_row_identity = build_row_identity_contract(
                    domain=OBSERVATION_INSTANCE_DOMAIN,
                    values=instance_key,
                )
                verify_canonical_coordinate_descriptor_identity(
                    descriptor,
                    row_identity_contract=exact_row_identity,
                    expected_row_identity_record_ref=(
                        descriptor.row_identity.record_ref
                    ),
                    owner_shape=position.shape,
                )
            except RowIdentityContractError as exc:
                issues.extend(
                    _issue(
                        f"row_identity_{item.code}",
                        f"instance_key{item.path[1:]}",
                        item.message,
                    )
                    for item in exc.issues
                )
            except CoordinateDescriptorError as exc:
                issues.extend(
                    _issue(
                        f"coordinate_descriptor_{item.code}",
                        f"$coordinate{item.path[1:]}",
                        item.message,
                    )
                    for item in exc.issues
                )
    for path, value in (
        ("source_acquisition_frame_index", frame_index),
        ("source_row_index", source_row),
    ):
        if (
            value is not None
            and value.dtype == _INT64_DTYPE
            and value.ndim == 1
            and value.shape[0] == row_count
            and np.any(value < 0)
        ):
            issues.append(
                _issue(
                    "negative_index",
                    path,
                    "Frame and source-row indices must be nonnegative.",
                )
            )
    if (
        source_row is not None
        and source_row.dtype == _INT64_DTYPE
        and source_row.ndim == 1
        and source_row.shape[0] == row_count
        and np.unique(source_row).size != source_row.size
    ):
        issues.append(
            _issue(
                "duplicate_source_row_index",
                "source_row_index",
                "source_row_index values must be unique within the bound source snapshot.",
            )
        )

    if (
        reasons is not None
        and reasons.dtype == _UINT16_DTYPE
        and reasons.ndim == 1
        and reasons.shape[0] == row_count
    ):
        known_codes = np.fromiter(POSITION_FAILURE_REASON_TAGS, dtype=np.uint16)
        if np.any(~np.isin(reasons, known_codes)):
            issues.append(
                _issue(
                    "unknown_reason_code",
                    "failure_reason_codes",
                    "Failure reason codes contain an unknown controlled code.",
                )
            )

    if (
        position is not None
        and position.dtype == _FLOAT32_DTYPE
        and position.ndim == 2
        and position.shape[0] == row_count
        and valid is not None
        and valid.dtype == _BOOL_DTYPE
        and valid.ndim == 1
        and valid.shape[0] == row_count
        and reasons is not None
        and reasons.dtype == _UINT16_DTYPE
        and reasons.ndim == 1
        and reasons.shape[0] == row_count
    ):
        finite = np.isfinite(position).all(axis=1)
        bits = position.view(np.uint32)
        canonical_nan = np.all(bits == CANONICAL_FLOAT32_QNAN_BITS, axis=1)
        if np.any(valid & ((reasons != 0) | ~finite)):
            issues.append(
                _issue(
                    "validity_payload_mismatch",
                    "position_xy",
                    "Valid rows require finite paired XY and reason code zero.",
                )
            )
        if np.any(~valid & ((reasons == 0) | ~canonical_nan)):
            issues.append(
                _issue(
                    "invalidity_payload_mismatch",
                    "position_xy",
                    "Invalid rows require a nonzero reason and exact paired canonical float32 NaNs.",
                )
            )

    _validate_support_arrays(arrays, row_count=row_count, issues=issues)
    return tuple(issues)


def validate_observation_position_arrays(
    arrays: Mapping[str, object],
    *,
    coordinate_metadata: object,
    manifest_metadata: Mapping[str, object],
) -> SubjectPositionStorageValidationReport:
    """Strictly validate observation arrays and return their bound summary."""

    issues = collect_observation_position_storage_issues(
        arrays,
        coordinate_metadata=coordinate_metadata,
        manifest_metadata=manifest_metadata,
    )
    if issues:
        raise SubjectPositionStorageValidationError(issues)

    descriptor = _coordinate_descriptor_from_metadata(coordinate_metadata)
    row_count = int(arrays["position_xy"].shape[0])
    support = arrays.get("support/source_points_xy")
    return SubjectPositionStorageValidationReport(
        row_count=row_count,
        support_point_count=None if support is None else int(support.shape[1]),
        storage_schema_sha256=observation_position_schema_digest(),
        coordinate_descriptor_sha256=descriptor.digest(),
        reason_code_map_sha256=position_failure_reason_map_digest(),
        reason_precedence_sha256=position_failure_reason_precedence_digest(),
    )


def validate_observation_position_storage(
    arrays: Mapping[str, object],
    *,
    coordinate_metadata: object,
    manifest_metadata: Mapping[str, object],
) -> SubjectPositionStorageValidationReport:
    """Compatibility alias for the strict observation-array validator."""

    return validate_observation_position_arrays(
        arrays,
        coordinate_metadata=coordinate_metadata,
        manifest_metadata=manifest_metadata,
    )


__all__ = [
    "CANONICAL_JSON_DIGEST_ALGORITHM",
    "OBSERVATION_POSITION_ARRAYS",
    "OBSERVATION_POSITION_MANDATORY_ARRAYS",
    "OBSERVATION_POSITION_NAMESPACE",
    "OBSERVATION_POSITION_OPTIONAL_ARRAYS",
    "POSITION_FAILURE_REASON_MAP_SCHEMA_ID",
    "POSITION_FAILURE_REASON_MAP_SCHEMA_VERSION",
    "POSITION_FAILURE_REASON_PRECEDENCE_SCHEMA_ID",
    "POSITION_FAILURE_REASON_PRECEDENCE_SCHEMA_VERSION",
    "SubjectPositionStorageError",
    "SubjectPositionStorageIssue",
    "SubjectPositionStorageValidationError",
    "SubjectPositionStorageValidationReport",
    "TRACK_SAMPLE_POSITION_CORE_ARRAYS",
    "TRACK_SAMPLE_POSITION_NAMESPACE",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "canonical_observation_position_arrays_sha256",
    "canonical_observation_position_logical_metadata",
    "canonical_observation_position_schema_descriptor",
    "canonical_observation_position_schema_json",
    "canonical_position_failure_reason_map",
    "canonical_position_failure_reason_map_json",
    "canonical_position_failure_reason_precedence",
    "canonical_position_failure_reason_precedence_json",
    "canonical_source_camera_coordinate_metadata",
    "canonical_track_sample_position_schema_descriptor",
    "canonical_track_sample_position_schema_json",
    "collect_observation_position_storage_issues",
    "observation_position_arrays_sha256",
    "observation_position_schema_descriptor",
    "observation_position_schema_digest",
    "position_failure_reason_map_digest",
    "position_failure_reason_precedence_digest",
    "track_sample_position_schema_digest",
    "validate_observation_position_arrays",
    "validate_observation_position_storage",
]
