"""Shared logical types for subject-position contracts and evaluators."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Mapping

import numpy as np

from fisheye.shared.coordinate_surface_contract import SOURCE_CAMERA_PROFILE_ID


SUBJECT_POSITION_STORAGE_SCHEMA_ID: Final = "palette.subject_position_storage"
SUBJECT_POSITION_STORAGE_SCHEMA_VERSION: Final = 1
OBSERVATION_POSITION_ROW_AXIS: Final = "observation_instance"
TRACK_SAMPLE_POSITION_ROW_AXIS: Final = "track_sample"
SOURCE_CAMERA_POSITION_PROFILE_ID: Final = SOURCE_CAMERA_PROFILE_ID

CANONICAL_FLOAT32_QNAN_BITS: Final = np.uint32(0x7FC00000)

POSITION_FAILURE_REASON_CODES: Final[Mapping[str, int]] = MappingProxyType(
    {
        "ok": 0,
        "source_observation_rejected": 1,
        "required_anchor_invalid": 2,
        "required_anchor_low_confidence": 3,
        "empty_mask_component": 4,
        "nonfinite_source_geometry": 5,
        "degenerate_source_geometry": 6,
    }
)
POSITION_FAILURE_REASON_TAGS: Final[Mapping[int, str]] = MappingProxyType(
    {code: tag for tag, code in POSITION_FAILURE_REASON_CODES.items()}
)
# Highest-priority applicable reason comes first.  This order is part of the
# estimator/storage contract; numeric code order alone is not precedence.
POSITION_FAILURE_REASON_PRECEDENCE: Final[tuple[str, ...]] = (
    "nonfinite_source_geometry",
    "degenerate_source_geometry",
    "empty_mask_component",
    "required_anchor_low_confidence",
    "required_anchor_invalid",
    "source_observation_rejected",
)


def canonical_float32_nan() -> np.float32:
    """Return the contract's exact quiet-NaN float32 payload."""

    return CANONICAL_FLOAT32_QNAN_BITS.reshape(()).view(np.float32)[()]


FLOAT32_QUANTIZATION_SCHEMA_ID: Final = "palette.subject_position_float32_quantization"
FLOAT32_QUANTIZATION_SCHEMA_VERSION: Final = 1
_FLOAT32_QUANTIZATION_FIELDS: Final = frozenset(
    {"schema_id", "schema_version", "scope", "valid_row_count", "max_abs_error"}
)


def float32_quantization_record(
    values64: np.ndarray,
    values32: np.ndarray,
    valid: np.ndarray,
) -> dict[str, object]:
    """Measure the float64-to-float32 publication error over valid rows.

    The error is in ``position_xy`` coordinate units.  Invalid rows carry
    canonical NaNs and are excluded.
    """

    values64 = np.asarray(values64, dtype=np.float64)
    values32 = np.asarray(values32)
    valid = np.asarray(valid, dtype=bool)
    if values32.dtype != np.dtype(np.float32) or values64.shape != values32.shape:
        raise ValueError("Quantization requires paired float64/float32 position arrays.")
    if values64.ndim != 2 or values64.shape[1] != 2 or valid.shape != (values64.shape[0],):
        raise ValueError("Quantization requires [N,2] positions and [N] validity.")
    errors = np.abs(values32[valid].astype(np.float64) - values64[valid])
    max_error = float(errors.max()) if errors.size else 0.0
    if not np.isfinite(max_error):
        raise ValueError("Quantization error is nonfinite over valid rows.")
    return {
        "schema_id": FLOAT32_QUANTIZATION_SCHEMA_ID,
        "schema_version": FLOAT32_QUANTIZATION_SCHEMA_VERSION,
        "scope": "position_xy_valid_rows",
        "valid_row_count": int(valid.sum()),
        "max_abs_error": max_error,
    }


def validate_float32_quantization_record(
    record: Mapping[str, object],
    *,
    position_xy: np.ndarray | None = None,
    valid: np.ndarray | None = None,
) -> dict[str, object]:
    """Validate a quantization record, and its bound when arrays are given.

    Round-to-nearest casting cannot move a value by more than half the float32
    spacing at the published magnitude, so a larger recorded error is stale or
    tampered.
    """

    if not isinstance(record, Mapping) or set(record) != _FLOAT32_QUANTIZATION_FIELDS:
        raise ValueError("Float32 quantization record field set is not exact.")
    if (
        record["schema_id"] != FLOAT32_QUANTIZATION_SCHEMA_ID
        or record["schema_version"] != FLOAT32_QUANTIZATION_SCHEMA_VERSION
        or record["scope"] != "position_xy_valid_rows"
    ):
        raise ValueError("Float32 quantization record identity differs.")
    count = record["valid_row_count"]
    error = record["max_abs_error"]
    if type(count) is not int or count < 0:
        raise ValueError("Float32 quantization valid_row_count must be a non-negative integer.")
    if type(error) is not float or not np.isfinite(error) or error < 0.0:
        raise ValueError("Float32 quantization max_abs_error must be a finite non-negative float.")
    if count == 0 and error != 0.0:
        raise ValueError("Float32 quantization error requires valid rows.")
    if (position_xy is None) != (valid is None):
        raise ValueError("Quantization bound checks require both position_xy and valid.")
    if position_xy is not None:
        valid_mask = np.asarray(valid, dtype=bool)
        if int(valid_mask.sum()) != count:
            raise ValueError("Float32 quantization valid_row_count differs from the arrays.")
        published = np.asarray(position_xy, dtype=np.float32)[valid_mask]
        bound = float(0.5 * np.spacing(np.abs(published)).max()) if published.size else 0.0
        if error > bound:
            raise ValueError("Float32 quantization error exceeds the half-spacing bound.")
    return dict(record)


def empty_position_xy(row_count: int) -> np.ndarray:
    """Return canonical all-invalid ``float32[N,2]`` position storage."""

    count = int(row_count)
    if count < 0:
        raise ValueError("row_count must be non-negative.")
    values = np.empty((count, 2), dtype=np.float32)
    values.view(np.uint32)[:] = CANONICAL_FLOAT32_QNAN_BITS
    return values


@dataclass(frozen=True)
class PositionEvaluationResult:
    """Unbound numeric output before source authorization and publication.

    This value alone does not prove coordinate frame, source schema, or row
    identity.  A Phase 2 source adapter must bind those authorities before an
    immutable position run can be published.
    """

    position_xy: np.ndarray
    valid: np.ndarray
    failure_reason_codes: np.ndarray
    source_points_xy: np.ndarray | None = None
    source_points_valid: np.ndarray | None = None
    source_point_reason_codes: np.ndarray | None = None
    source_point_confidence: np.ndarray | None = None
    float32_quantization: Mapping[str, object] | None = None


__all__ = [
    "CANONICAL_FLOAT32_QNAN_BITS",
    "FLOAT32_QUANTIZATION_SCHEMA_ID",
    "FLOAT32_QUANTIZATION_SCHEMA_VERSION",
    "OBSERVATION_POSITION_ROW_AXIS",
    "POSITION_FAILURE_REASON_CODES",
    "POSITION_FAILURE_REASON_PRECEDENCE",
    "POSITION_FAILURE_REASON_TAGS",
    "PositionEvaluationResult",
    "SOURCE_CAMERA_POSITION_PROFILE_ID",
    "SUBJECT_POSITION_STORAGE_SCHEMA_ID",
    "SUBJECT_POSITION_STORAGE_SCHEMA_VERSION",
    "TRACK_SAMPLE_POSITION_ROW_AXIS",
    "canonical_float32_nan",
    "empty_position_xy",
    "float32_quantization_record",
    "validate_float32_quantization_record",
]
