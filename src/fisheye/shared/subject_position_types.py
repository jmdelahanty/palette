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


__all__ = [
    "CANONICAL_FLOAT32_QNAN_BITS",
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
]
