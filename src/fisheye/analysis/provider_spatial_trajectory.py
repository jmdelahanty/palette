"""Pure provider-bound trajectory preparation over exact track samples.

This module is deliberately an in-memory boundary.  It does not open Zarr,
resolve a selector, compile a stimulus expression, publish an artifact, or
perform any interpolation or smoothing.  Callers provide one exact track-
sample rowset, one exact resolved frame-membership object, and one explicit
camera-to-arena transform.

The result keeps selection, provider validity, coordinate transformation, and
grid membership independent.  That makes missing provider values and spatial
coverage visible without allowing a later consumer to silently substitute a
different provider or clip an out-of-grid point into the nearest bin.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


PROVIDER_SPATIAL_TRAJECTORY_SCHEMA_ID = "palette.provider_spatial_trajectory"
PROVIDER_SPATIAL_TRAJECTORY_SCHEMA_VERSION = 2
TRACK_SAMPLE_INPUT_SCHEMA_ID = "palette.provider_track_samples"
TRACK_SAMPLE_INPUT_SCHEMA_VERSION = 1
SELECTED_FRAME_MEMBERSHIP_SCHEMA_ID = "palette.selected_frame_membership"
SELECTED_FRAME_MEMBERSHIP_SCHEMA_VERSION = 2
CAMERA_TO_ARENA_MM_TRANSFORM_SCHEMA_ID = "palette.camera_to_arena_mm_transform"
CAMERA_TO_ARENA_MM_TRANSFORM_SCHEMA_VERSION = 2
SOURCE_CAMERA_EXTENT_POLICY_ID = "half_open_source_camera_extent_px_v1"

_MUTABLE_ID_ALIASES = frozenset(
    {
        "active",
        "authoritative",
        "current",
        "default",
        "fallback",
        "latest",
        "latest_complete",
        "none",
        "null",
        "selected",
        "stale",
        "unknown",
    }
)
_REASON_CODES = frozenset(
    {
        "ok",
        "not_in_selection",
        "provider_missing",
        "provider_invalid",
        "source_position_nonfinite",
        "source_position_out_of_extent",
        "transform_invalid",
        "out_of_grid",
    }
)


class ProviderSpatialTrajectoryError(ValueError):
    """Raised when an exact provider trajectory input is not safe to use."""


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            _json_ready(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ProviderSpatialTrajectoryError(
            "Trajectory contract value is not strict JSON."
        ) from exc


def _json_ready(value: Any) -> Any:
    """Convert numpy values and nonfinite evidence to deterministic JSON."""

    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float):
        if np.isnan(value):
            return {"__nonfinite_float__": "nan"}
        if np.isposinf(value):
            return {"__nonfinite_float__": "+inf"}
        if np.isneginf(value):
            return {"__nonfinite_float__": "-inf"}
    return value


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _read_only_array(value: Any, *, dtype: np.dtype[Any]) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _require_identity(value: object, *, name: str) -> str:
    if (
        not isinstance(value, (str, np.str_))
        or not value
        or value != value.strip()
        or any(character.isspace() for character in value)
        or value.lower() in _MUTABLE_ID_ALIASES
    ):
        raise ProviderSpatialTrajectoryError(
            f"{name} must be one nonempty immutable identity, not a selector "
            "or stale/unknown placeholder."
        )
    return str(value)


def _require_reason(value: object, *, name: str) -> str:
    if (
        not isinstance(value, (str, np.str_))
        or not value
        or value != value.strip()
        or any(character.isspace() for character in value)
    ):
        raise ProviderSpatialTrajectoryError(
            f"{name} must be one nonempty reason code."
        )
    return str(value)


def _string_vector(
    values: Sequence[object] | np.ndarray,
    *,
    name: str,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ProviderSpatialTrajectoryError(f"{name} must be a one-dimensional vector.")
    try:
        items = tuple(values)
    except TypeError as exc:
        raise ProviderSpatialTrajectoryError(
            f"{name} must be a one-dimensional vector."
        ) from exc
    result = tuple(_require_identity(item, name=f"{name}[{index}]") for index, item in enumerate(items))
    if not allow_empty and not result:
        raise ProviderSpatialTrajectoryError(f"{name} must not be empty.")
    return result


def _reason_vector(
    values: Sequence[object] | np.ndarray,
    *,
    name: str,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ProviderSpatialTrajectoryError(f"{name} must be a one-dimensional vector.")
    try:
        items = tuple(values)
    except TypeError as exc:
        raise ProviderSpatialTrajectoryError(
            f"{name} must be a one-dimensional vector."
        ) from exc
    return tuple(
        _require_reason(item, name=f"{name}[{index}]")
        for index, item in enumerate(items)
    )


def _membership_vector(
    values: Sequence[object] | np.ndarray,
    *,
    name: str,
) -> tuple[tuple[str, ...], ...]:
    """Canonicalize one ordered set of identities for every frame row."""

    if isinstance(values, (str, bytes)):
        raise ProviderSpatialTrajectoryError(
            f"{name} must be a vector of per-frame identity sets."
        )
    try:
        rows = tuple(values)
    except TypeError as exc:
        raise ProviderSpatialTrajectoryError(
            f"{name} must be a vector of per-frame identity sets."
        ) from exc
    result: list[tuple[str, ...]] = []
    for row_index, row in enumerate(rows):
        try:
            row_values = (
                (row,)
                if isinstance(row, (str, np.str_))
                else tuple(row)  # type: ignore[arg-type]
            )
        except TypeError as exc:
            raise ProviderSpatialTrajectoryError(
                f"{name}[{row_index}] must be an identity sequence."
            ) from exc
        identities = tuple(
            _require_identity(value, name=f"{name}[{row_index}]")
            for value in row_values
        )
        if len(set(identities)) != len(identities):
            raise ProviderSpatialTrajectoryError(
                f"{name}[{row_index}] contains duplicate identities."
            )
        result.append(identities)
    return tuple(result)


def _integer_vector(
    values: Sequence[object] | np.ndarray,
    *,
    name: str,
    allow_empty: bool = True,
) -> np.ndarray:
    if isinstance(values, (str, bytes)):
        raise ProviderSpatialTrajectoryError(f"{name} must be an integer vector.")
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ProviderSpatialTrajectoryError(f"{name} must be an integer vector.") from exc
    if raw.ndim != 1 or raw.dtype.kind not in "iu":
        raise ProviderSpatialTrajectoryError(
            f"{name} must be one-dimensional and contain integer values."
        )
    if not allow_empty and raw.size == 0:
        raise ProviderSpatialTrajectoryError(f"{name} must not be empty.")
    if np.any(raw < 0):
        raise ProviderSpatialTrajectoryError(f"{name} must not contain negative values.")
    return _read_only_array(raw, dtype=np.dtype(np.int64))


def _track_sample_key_array(value: object) -> np.ndarray:
    """Validate Palette's canonical ``[track_id, acquisition_frame]`` key."""

    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ProviderSpatialTrajectoryError(
            "track_sample_key must be one integer array with shape (N, 2)."
        ) from exc
    if raw.ndim != 2 or raw.shape[1:] != (2,) or raw.dtype.kind not in "iu":
        raise ProviderSpatialTrajectoryError(
            "track_sample_key must be one integer array with shape (N, 2)."
        )
    if np.any(raw < 0):
        raise ProviderSpatialTrajectoryError(
            "track_sample_key must not contain negative values."
        )
    return _read_only_array(raw, dtype=np.dtype(np.int64))


def _boolean_vector(
    values: Sequence[object] | np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    if isinstance(values, (str, bytes)):
        raise ProviderSpatialTrajectoryError(f"{name} must be a boolean vector.")
    try:
        items = tuple(values)
    except TypeError as exc:
        raise ProviderSpatialTrajectoryError(f"{name} must be a boolean vector.") from exc
    if any(type(item) not in {bool, np.bool_} for item in items):
        raise ProviderSpatialTrajectoryError(f"{name} must contain only boolean values.")
    return _read_only_array(items, dtype=np.dtype(bool))


def _position_array(value: object, *, row_count: int, name: str) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ProviderSpatialTrajectoryError(f"{name} must be a numeric array.") from exc
    if raw.ndim != 2 or raw.shape != (row_count, 2) or raw.dtype.kind not in "iuf":
        raise ProviderSpatialTrajectoryError(
            f"{name} must have shape ({row_count}, 2) and a numeric dtype."
        )
    return _read_only_array(raw, dtype=np.dtype(np.float64))


def _extent(value: object, *, name: str) -> tuple[float, float, float, float]:
    if isinstance(value, (str, bytes)):
        raise ProviderSpatialTrajectoryError(
            f"{name} must be (x_min, x_max, y_min, y_max)."
        )
    try:
        items = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ProviderSpatialTrajectoryError(
            f"{name} must be (x_min, x_max, y_min, y_max)."
        ) from exc
    if len(items) != 4 or any(
        isinstance(item, bool) or not isinstance(item, (int, float, np.number))
        for item in items
    ):
        raise ProviderSpatialTrajectoryError(
            f"{name} must contain four finite numeric values."
        )
    result = tuple(float(item) for item in items)
    if not np.all(np.isfinite(result)) or not (result[0] < result[1] and result[2] < result[3]):
        raise ProviderSpatialTrajectoryError(
            f"{name} must have finite increasing x and y bounds."
        )
    return result  # type: ignore[return-value]


@dataclass(frozen=True)
class TrajectoryAuthorityIdentities:
    """All immutable authorities required by one trajectory preparation."""

    recording_id: str
    provider_id: str
    track_sample_policy_id: str
    estimator_id: str
    source_id: str
    timing_authority_id: str
    timeline_authority_id: str
    coordinate_authority_id: str
    selection_authority_id: str

    def __post_init__(self) -> None:
        for name in (
            "recording_id",
            "provider_id",
            "track_sample_policy_id",
            "estimator_id",
            "source_id",
            "timing_authority_id",
            "timeline_authority_id",
            "coordinate_authority_id",
            "selection_authority_id",
        ):
            object.__setattr__(self, name, _require_identity(getattr(self, name), name=name))

    def as_record(self) -> dict[str, str]:
        return {
            "recording_id": self.recording_id,
            "provider_id": self.provider_id,
            "track_sample_policy_id": self.track_sample_policy_id,
            "estimator_id": self.estimator_id,
            "source_id": self.source_id,
            "timing_authority_id": self.timing_authority_id,
            "timeline_authority_id": self.timeline_authority_id,
            "coordinate_authority_id": self.coordinate_authority_id,
            "selection_authority_id": self.selection_authority_id,
        }


@dataclass(frozen=True)
class SelectedFrameMembership:
    """One exact, already-resolved acquisition-frame membership input.

    The contract has one row per selected acquisition frame. A row may retain several
    aligned membership keys, occurrence identities, and roles when source
    intervals overlap; pooled consumers still count that acquisition frame
    only once.
    """

    recording_id: str
    timeline_authority_id: str
    selection_authority_id: str
    acquisition_frames: Sequence[object] | np.ndarray
    membership_keys: Sequence[object] | np.ndarray
    occurrence_ids: Sequence[object] | np.ndarray
    roles: Sequence[object] | np.ndarray

    def __post_init__(self) -> None:
        recording_id = _require_identity(self.recording_id, name="selection.recording_id")
        timeline_id = _require_identity(
            self.timeline_authority_id, name="selection.timeline_authority_id"
        )
        selection_id = _require_identity(
            self.selection_authority_id, name="selection.selection_authority_id"
        )
        frames = _integer_vector(self.acquisition_frames, name="selection.acquisition_frames")
        membership_keys = _membership_vector(
            self.membership_keys, name="selection.membership_keys"
        )
        occurrence_ids = _membership_vector(
            self.occurrence_ids, name="selection.occurrence_ids"
        )
        roles = _membership_vector(self.roles, name="selection.roles")
        expected = frames.size
        if any(len(values) != expected for values in (membership_keys, occurrence_ids, roles)):
            raise ProviderSpatialTrajectoryError(
                "Selection frame and membership vectors have mismatched cardinality."
            )
        if np.unique(frames).size != expected:
            raise ProviderSpatialTrajectoryError(
                "Selection contains duplicate acquisition frames. Resolve overlap "
                "before preparing a trajectory."
            )
        if expected and not np.all(np.diff(frames) > 0):
            raise ProviderSpatialTrajectoryError(
                "Selection acquisition frames are reordered; they must be strictly "
                "increasing."
            )
        for index, (keys, occurrences, role_values) in enumerate(
            zip(membership_keys, occurrence_ids, roles, strict=True)
        ):
            if not keys or len(keys) != len(occurrences) or len(keys) != len(role_values):
                raise ProviderSpatialTrajectoryError(
                    "Selection frame memberships must be nonempty and have aligned "
                    f"keys, occurrences, and roles (row {index})."
                )
        object.__setattr__(self, "recording_id", recording_id)
        object.__setattr__(self, "timeline_authority_id", timeline_id)
        object.__setattr__(self, "selection_authority_id", selection_id)
        object.__setattr__(self, "acquisition_frames", frames)
        object.__setattr__(self, "membership_keys", membership_keys)
        object.__setattr__(self, "occurrence_ids", occurrence_ids)
        object.__setattr__(self, "roles", roles)

    def as_record(self) -> dict[str, Any]:
        return {
            "schema_id": SELECTED_FRAME_MEMBERSHIP_SCHEMA_ID,
            "schema_version": SELECTED_FRAME_MEMBERSHIP_SCHEMA_VERSION,
            "recording_id": self.recording_id,
            "timeline_authority_id": self.timeline_authority_id,
            "selection_authority_id": self.selection_authority_id,
            "acquisition_frames": self.acquisition_frames.tolist(),
            "membership_keys": [list(value) for value in self.membership_keys],
            "occurrence_ids": [list(value) for value in self.occurrence_ids],
            "roles": [list(value) for value in self.roles],
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.as_record())


@dataclass(frozen=True)
class ProviderTrackSamples:
    """Exact source rows for the single-subject track-sample profile."""

    track_sample_key: Sequence[object] | np.ndarray
    acquisition_frame: Sequence[object] | np.ndarray
    subject_identity: Sequence[object] | np.ndarray
    track_identity: Sequence[object] | np.ndarray
    source_position_xy: object
    provider_present: Sequence[object] | np.ndarray
    provider_valid: Sequence[object] | np.ndarray
    provider_reason_code: Sequence[object] | np.ndarray
    recording_ids: Sequence[object] | np.ndarray | None = None
    timeline_authority_ids: Sequence[object] | np.ndarray | None = None

    def __post_init__(self) -> None:
        keys = _track_sample_key_array(self.track_sample_key)
        frames = _integer_vector(self.acquisition_frame, name="acquisition_frame")
        subjects = _string_vector(self.subject_identity, name="subject_identity")
        tracks = _string_vector(self.track_identity, name="track_identity")
        present = _boolean_vector(self.provider_present, name="provider_present")
        valid = _boolean_vector(self.provider_valid, name="provider_valid")
        reasons = _reason_vector(
            self.provider_reason_code, name="provider_reason_code"
        )
        row_count = int(keys.shape[0])
        vectors = {
            "acquisition_frame": frames.size,
            "subject_identity": len(subjects),
            "track_identity": len(tracks),
            "provider_present": present.size,
            "provider_valid": valid.size,
            "provider_reason_code": len(reasons),
        }
        if any(size != row_count for size in vectors.values()):
            raise ProviderSpatialTrajectoryError(
                f"Track-sample vectors have mismatched cardinality: {vectors!r}."
            )
        positions = _position_array(
            self.source_position_xy, row_count=row_count, name="source_position_xy"
        )
        if not np.array_equal(keys[:, 1], frames):
            raise ProviderSpatialTrajectoryError(
                "track_sample_key acquisition-frame values disagree with "
                "acquisition_frame."
            )
        if np.unique(keys, axis=0).shape[0] != row_count:
            raise ProviderSpatialTrajectoryError("track_sample_key contains duplicates.")
        composite = tuple(zip(subjects, tracks, frames.tolist(), strict=True))
        if len(set(composite)) != row_count:
            raise ProviderSpatialTrajectoryError(
                "Duplicate subject-track + acquisition-frame rows are not allowed."
            )
        canonical_order = np.lexsort((keys[:, 1], keys[:, 0])).tolist()
        if canonical_order != list(range(row_count)):
            raise ProviderSpatialTrajectoryError(
                "Track-sample keys are reordered; rows must be canonical track, "
                "acquisition-frame order."
            )
        if np.any(valid & ~present):
            raise ProviderSpatialTrajectoryError(
                "provider_valid cannot be true when provider_present is false."
            )
        if np.unique(frames).size != frames.size:
            raise ProviderSpatialTrajectoryError(
                "The single-subject profile has more than one track sample for "
                "an acquisition frame."
            )
        for name, values in (("subject_identity", subjects), ("track_identity", tracks)):
            if any(not value for value in values):
                raise ProviderSpatialTrajectoryError(f"{name} cannot contain empty identities.")

        recording_ids = self._optional_row_authorities(
            self.recording_ids, row_count=row_count, name="recording_ids"
        )
        timeline_ids = self._optional_row_authorities(
            self.timeline_authority_ids,
            row_count=row_count,
            name="timeline_authority_ids",
        )
        object.__setattr__(self, "track_sample_key", keys)
        object.__setattr__(self, "acquisition_frame", frames)
        object.__setattr__(self, "subject_identity", subjects)
        object.__setattr__(self, "track_identity", tracks)
        object.__setattr__(self, "source_position_xy", positions)
        object.__setattr__(self, "provider_present", present)
        object.__setattr__(self, "provider_valid", valid)
        object.__setattr__(self, "provider_reason_code", reasons)
        object.__setattr__(self, "recording_ids", recording_ids)
        object.__setattr__(self, "timeline_authority_ids", timeline_ids)

    @staticmethod
    def _optional_row_authorities(
        values: Sequence[object] | np.ndarray | None,
        *,
        row_count: int,
        name: str,
    ) -> tuple[str, ...] | None:
        if values is None:
            return None
        result = _string_vector(values, name=name)
        if len(result) != row_count:
            raise ProviderSpatialTrajectoryError(
                f"{name} has mismatched cardinality: expected {row_count}, got {len(result)}."
            )
        return result

    def as_record(self) -> dict[str, Any]:
        return {
            "schema_id": TRACK_SAMPLE_INPUT_SCHEMA_ID,
            "schema_version": TRACK_SAMPLE_INPUT_SCHEMA_VERSION,
            "track_sample_key": self.track_sample_key.tolist(),
            "acquisition_frame": self.acquisition_frame.tolist(),
            "subject_identity": list(self.subject_identity),
            "track_identity": list(self.track_identity),
            "source_position_xy": self.source_position_xy.tolist(),
            "provider_present": self.provider_present.tolist(),
            "provider_valid": self.provider_valid.tolist(),
            "provider_reason_code": list(self.provider_reason_code),
            "recording_ids": None if self.recording_ids is None else list(self.recording_ids),
            "timeline_authority_ids": (
                None
                if self.timeline_authority_ids is None
                else list(self.timeline_authority_ids)
            ),
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.as_record())


@dataclass(frozen=True)
class SourceCameraToArenaMMTransform:
    """One explicit 3x3 source-camera to arena-mm homogeneous transform.

    ``source_camera_extent_px`` is authoritative for the source camera pixel
    frame. Its policy is half-open: ``x_min <= x < x_max`` and
    ``y_min <= y < y_max``. Preparation fails closed when the extent is
    absent; it is not inferred from the transform or source rows.
    """

    source_coordinate_authority_id: str
    target_coordinate_authority_id: str
    matrix: object
    grid_extent_mm: tuple[float, float, float, float]
    source_camera_extent_px: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        source_id = _require_identity(
            self.source_coordinate_authority_id,
            name="transform.source_coordinate_authority_id",
        )
        target_id = _require_identity(
            self.target_coordinate_authority_id,
            name="transform.target_coordinate_authority_id",
        )
        try:
            matrix = np.asarray(self.matrix)
        except (TypeError, ValueError) as exc:
            raise ProviderSpatialTrajectoryError("transform.matrix must be numeric.") from exc
        if matrix.shape != (3, 3) or matrix.dtype.kind not in "iuf":
            raise ProviderSpatialTrajectoryError(
                "transform.matrix must be a numeric 3x3 homogeneous matrix."
            )
        matrix = _read_only_array(matrix, dtype=np.dtype(np.float64))
        if not np.all(np.isfinite(matrix)) or abs(float(np.linalg.det(matrix))) <= 1e-15:
            raise ProviderSpatialTrajectoryError(
                "transform.matrix must be finite and non-singular."
            )
        grid_extent = _extent(self.grid_extent_mm, name="transform.grid_extent_mm")
        source_extent = (
            None
            if self.source_camera_extent_px is None
            else _extent(
                self.source_camera_extent_px,
                name="transform.source_camera_extent_px",
            )
        )
        object.__setattr__(self, "source_coordinate_authority_id", source_id)
        object.__setattr__(self, "target_coordinate_authority_id", target_id)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "grid_extent_mm", grid_extent)
        object.__setattr__(self, "source_camera_extent_px", source_extent)

    def as_record(self) -> dict[str, Any]:
        return {
            "schema_id": CAMERA_TO_ARENA_MM_TRANSFORM_SCHEMA_ID,
            "schema_version": CAMERA_TO_ARENA_MM_TRANSFORM_SCHEMA_VERSION,
            "source_coordinate_authority_id": self.source_coordinate_authority_id,
            "target_coordinate_authority_id": self.target_coordinate_authority_id,
            "matrix": self.matrix.tolist(),
            "grid_extent_mm": list(self.grid_extent_mm),
            "source_camera_extent_policy": SOURCE_CAMERA_EXTENT_POLICY_ID,
            "source_camera_extent_px": (
                None
                if self.source_camera_extent_px is None
                else list(self.source_camera_extent_px)
            ),
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.as_record())


@dataclass(frozen=True)
class TrajectoryCounts:
    """Deterministic source-wide and selected-frame coverage evidence."""

    expected_selected_frames: int
    source_rows: int
    selected_source_rows: int
    missing_selected_frames: int
    provider_present_rows: int
    provider_valid_rows: int
    valid_position_rows: int
    source_extent_valid_rows: int
    transform_valid_rows: int
    in_grid_rows: int
    missing_provider_rows: int
    invalid_provider_rows: int
    nonfinite_position_rows: int
    source_position_out_of_extent_rows: int
    transform_invalid_rows: int
    out_of_grid_rows: int
    selected_provider_present_rows: int
    selected_provider_valid_rows: int
    selected_valid_position_rows: int
    selected_source_extent_valid_rows: int
    selected_transform_valid_rows: int
    selected_in_grid_rows: int
    selected_missing_provider_rows: int
    selected_invalid_provider_rows: int
    selected_nonfinite_position_rows: int
    selected_source_position_out_of_extent_rows: int
    selected_transform_invalid_rows: int
    selected_out_of_grid_rows: int

    def as_record(self) -> dict[str, int]:
        return {
            name: int(getattr(self, name))
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class ProviderSpatialTrajectory:
    """Immutable in-memory trajectory preparation over the exact source rows."""

    authorities: TrajectoryAuthorityIdentities
    selection: SelectedFrameMembership
    transform: SourceCameraToArenaMMTransform
    source_row_index: np.ndarray
    track_sample_key: np.ndarray
    acquisition_frame: np.ndarray
    subject_identity: tuple[str, ...]
    track_identity: tuple[str, ...]
    source_position_xy: np.ndarray
    arena_position_xy: np.ndarray
    provider_present: np.ndarray
    provider_valid: np.ndarray
    source_position_valid: np.ndarray
    source_extent_valid: np.ndarray
    in_selection: np.ndarray
    transform_valid: np.ndarray
    in_grid: np.ndarray
    selection_membership_key: tuple[tuple[str, ...], ...]
    selection_occurrence_id: tuple[tuple[str, ...], ...]
    selection_role: tuple[tuple[str, ...], ...]
    reason_codes: tuple[tuple[str, ...], ...]
    reason_counts: Mapping[str, int]
    selected_reason_counts: Mapping[str, int]
    counts: TrajectoryCounts
    source_rows_sha256: str
    trajectory_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "source_row_index",
            "track_sample_key",
            "acquisition_frame",
            "source_position_xy",
            "arena_position_xy",
            "provider_present",
            "provider_valid",
            "source_position_valid",
            "source_extent_valid",
            "in_selection",
            "transform_valid",
            "in_grid",
        ):
            array = getattr(self, name)
            if not isinstance(array, np.ndarray):
                raise ProviderSpatialTrajectoryError(f"result.{name} must be an array.")
            array.setflags(write=False)
        object.__setattr__(self, "reason_counts", MappingProxyType(dict(self.reason_counts)))
        object.__setattr__(
            self,
            "selected_reason_counts",
            MappingProxyType(dict(self.selected_reason_counts)),
        )

    def as_record(self) -> dict[str, Any]:
        return {
            "schema_id": PROVIDER_SPATIAL_TRAJECTORY_SCHEMA_ID,
            "schema_version": PROVIDER_SPATIAL_TRAJECTORY_SCHEMA_VERSION,
            "row_axis": "track_samples",
            "authorities": self.authorities.as_record(),
            "selection": {
                "sha256": self.selection.sha256,
                "record": self.selection.as_record(),
            },
            "transform": {
                "sha256": self.transform.sha256,
                "record": self.transform.as_record(),
            },
            "source_rows_sha256": self.source_rows_sha256,
            "trajectory_sha256": self.trajectory_sha256,
            "counts": self.counts.as_record(),
            "reason_counts": dict(self.reason_counts),
            "selected_reason_counts": dict(self.selected_reason_counts),
            "state_fields": [
                "provider_present",
                "provider_valid",
                "source_extent_valid",
                "in_selection",
                "transform_valid",
                "in_grid",
            ],
            "smoothing": "none",
            "interpolation": "none",
            "fallback": "none",
        }


def _validate_authority_binding(
    *,
    authorities: TrajectoryAuthorityIdentities,
    selection: SelectedFrameMembership,
    transform: SourceCameraToArenaMMTransform,
    rows: ProviderTrackSamples,
) -> None:
    if selection.recording_id != authorities.recording_id:
        raise ProviderSpatialTrajectoryError("Selection recording identity is mixed.")
    if selection.timeline_authority_id != authorities.timeline_authority_id:
        raise ProviderSpatialTrajectoryError("Selection timeline authority is mixed.")
    if selection.selection_authority_id != authorities.selection_authority_id:
        raise ProviderSpatialTrajectoryError("Selection authority identity is mixed.")
    if transform.source_coordinate_authority_id != authorities.coordinate_authority_id:
        raise ProviderSpatialTrajectoryError("Transform coordinate authority is stale or mixed.")
    if rows.recording_ids is not None and any(
        value != authorities.recording_id for value in rows.recording_ids
    ):
        raise ProviderSpatialTrajectoryError("Track-sample rows contain mixed recording identities.")
    if rows.timeline_authority_ids is not None and any(
        value != authorities.timeline_authority_id for value in rows.timeline_authority_ids
    ):
        raise ProviderSpatialTrajectoryError(
            "Track-sample rows contain mixed timeline authorities."
        )


def _reason_counts(
    reasons: Sequence[tuple[str, ...]],
    selected: np.ndarray,
) -> tuple[dict[str, int], dict[str, int]]:
    all_counts = {reason: 0 for reason in sorted(_REASON_CODES)}
    selected_counts = {reason: 0 for reason in sorted(_REASON_CODES)}
    for index, row_reasons in enumerate(reasons):
        primary = next(
            (
                reason
                for reason in row_reasons
                if reason
                in {
                    "provider_missing",
                    "provider_invalid",
                    "source_position_nonfinite",
                    "source_position_out_of_extent",
                    "transform_invalid",
                    "out_of_grid",
                }
            ),
            "ok",
        )
        all_counts[primary] = all_counts.get(primary, 0) + 1
        if selected[index]:
            selected_counts[primary] = selected_counts.get(primary, 0) + 1
    return all_counts, selected_counts


def prepare_provider_spatial_trajectory(
    *,
    authorities: TrajectoryAuthorityIdentities,
    rows: ProviderTrackSamples,
    selection: SelectedFrameMembership,
    transform: SourceCameraToArenaMMTransform,
) -> ProviderSpatialTrajectory:
    """Prepare one exact, unsmoothed provider trajectory in arena millimetres.

    Selection is a frame lookup, never a positional or same-length join.  All
    source rows are retained in their validated order, including rows outside
    the selected frame set and rows whose provider value is unavailable.
    """

    if type(authorities) is not TrajectoryAuthorityIdentities:
        raise ProviderSpatialTrajectoryError("authorities must be an exact identity record.")
    if type(rows) is not ProviderTrackSamples:
        raise ProviderSpatialTrajectoryError("rows must be an exact ProviderTrackSamples record.")
    if type(selection) is not SelectedFrameMembership:
        raise ProviderSpatialTrajectoryError("selection must be an exact membership record.")
    if type(transform) is not SourceCameraToArenaMMTransform:
        raise ProviderSpatialTrajectoryError("transform must be an exact transform record.")
    if transform.source_camera_extent_px is None:
        raise ProviderSpatialTrajectoryError(
            "source_camera_extent_px is required for the camera-pixel transform "
            "path; source extent policy fails closed when absent."
        )
    _validate_authority_binding(
        authorities=authorities,
        selection=selection,
        transform=transform,
        rows=rows,
    )

    source_count = len(rows.track_sample_key)
    source_row_index = _read_only_array(np.arange(source_count), dtype=np.dtype(np.int64))
    frames = rows.acquisition_frame
    selected_frame_to_membership = {
        int(frame): index for index, frame in enumerate(selection.acquisition_frames.tolist())
    }
    in_selection = _read_only_array(
        np.array([int(frame) in selected_frame_to_membership for frame in frames], dtype=bool),
        dtype=np.dtype(bool),
    )
    membership_keys = tuple(
        selection.membership_keys[selected_frame_to_membership[int(frame)]]
        if int(frame) in selected_frame_to_membership
        else ()
        for frame in frames
    )
    occurrence_ids = tuple(
        selection.occurrence_ids[selected_frame_to_membership[int(frame)]]
        if int(frame) in selected_frame_to_membership
        else ()
        for frame in frames
    )
    roles = tuple(
        selection.roles[selected_frame_to_membership[int(frame)]]
        if int(frame) in selected_frame_to_membership
        else ()
        for frame in frames
    )

    source_position_finite = np.all(np.isfinite(rows.source_position_xy), axis=1)
    source_position_valid = rows.provider_present & rows.provider_valid & source_position_finite
    source_xmin, source_xmax, source_ymin, source_ymax = transform.source_camera_extent_px
    source_extent_valid = source_position_finite & (
        (rows.source_position_xy[:, 0] >= source_xmin)
        & (rows.source_position_xy[:, 0] < source_xmax)
        & (rows.source_position_xy[:, 1] >= source_ymin)
        & (rows.source_position_xy[:, 1] < source_ymax)
    )
    arena_position = np.full((source_count, 2), np.nan, dtype=np.float64)
    transform_valid = np.zeros(source_count, dtype=bool)
    transform_input_valid = source_position_valid & source_extent_valid
    if np.any(transform_input_valid):
        source_homogeneous = np.column_stack(
            (rows.source_position_xy[transform_input_valid], np.ones(int(np.count_nonzero(transform_input_valid))))
        )
        transformed_homogeneous = source_homogeneous @ transform.matrix.T
        weights = transformed_homogeneous[:, 2]
        finite_weights = np.isfinite(weights) & (np.abs(weights) > 1e-15)
        finite_xy = np.all(np.isfinite(transformed_homogeneous[:, :2]), axis=1)
        local_valid = finite_weights & finite_xy
        valid_indices = np.flatnonzero(transform_input_valid)
        if np.any(local_valid):
            arena_position[valid_indices[local_valid]] = (
                transformed_homogeneous[local_valid, :2]
                / weights[local_valid, None]
            )
            transform_valid[valid_indices[local_valid]] = True
    transform_valid = _read_only_array(transform_valid, dtype=np.dtype(bool))
    arena_position = _read_only_array(arena_position, dtype=np.dtype(np.float64))
    source_position_finite = _read_only_array(source_position_finite, dtype=np.dtype(bool))
    source_position_valid = _read_only_array(source_position_valid, dtype=np.dtype(bool))
    source_extent_valid = _read_only_array(source_extent_valid, dtype=np.dtype(bool))

    xmin, xmax, ymin, ymax = transform.grid_extent_mm
    in_grid_values = np.zeros(source_count, dtype=bool)
    if np.any(transform_valid):
        x = arena_position[:, 0]
        y = arena_position[:, 1]
        in_grid_values = transform_valid & (
            (x >= xmin)
            & ((x < xmax) | (x == xmax))
            & (y >= ymin)
            & ((y < ymax) | (y == ymax))
        )
    in_grid = _read_only_array(in_grid_values, dtype=np.dtype(bool))

    reasons: list[tuple[str, ...]] = []
    for index in range(source_count):
        row_reasons: list[str] = []
        if not in_selection[index]:
            row_reasons.append("not_in_selection")
        if not rows.provider_present[index]:
            row_reasons.append("provider_missing")
        elif not rows.provider_valid[index]:
            row_reasons.append("provider_invalid")
        elif not source_position_finite[index]:
            row_reasons.append("source_position_nonfinite")
        elif not source_extent_valid[index]:
            row_reasons.append("source_position_out_of_extent")
        elif not transform_valid[index]:
            row_reasons.append("transform_invalid")
        elif not in_grid[index]:
            row_reasons.append("out_of_grid")
        else:
            row_reasons.append("ok")
        input_reason = rows.provider_reason_code[index]
        if input_reason != "ok" and input_reason not in row_reasons:
            row_reasons.append(input_reason)
        reasons.append(tuple(row_reasons))
    reason_counts, selected_reason_counts = _reason_counts(reasons, in_selection)

    source_frame_set = set(int(frame) for frame in frames.tolist())
    missing_selected_frames = sum(
        int(frame) not in source_frame_set
        for frame in selection.acquisition_frames.tolist()
    )
    selected = in_selection
    missing_provider = ~rows.provider_present
    invalid_provider = rows.provider_present & ~rows.provider_valid
    nonfinite_position = rows.provider_present & rows.provider_valid & ~source_position_finite
    out_of_source_extent = source_position_valid & ~source_extent_valid
    transform_invalid = source_position_valid & source_extent_valid & ~transform_valid
    out_of_grid = transform_valid & ~in_grid

    def count(mask: np.ndarray) -> int:
        return int(np.count_nonzero(mask))

    counts = TrajectoryCounts(
        expected_selected_frames=int(selection.acquisition_frames.size),
        source_rows=source_count,
        selected_source_rows=count(selected),
        missing_selected_frames=int(missing_selected_frames),
        provider_present_rows=count(rows.provider_present),
        provider_valid_rows=count(rows.provider_valid),
        valid_position_rows=count(source_position_valid),
        source_extent_valid_rows=count(source_extent_valid),
        transform_valid_rows=count(transform_valid),
        in_grid_rows=count(in_grid),
        missing_provider_rows=count(missing_provider),
        invalid_provider_rows=count(invalid_provider),
        nonfinite_position_rows=count(nonfinite_position),
        source_position_out_of_extent_rows=count(out_of_source_extent),
        transform_invalid_rows=count(transform_invalid),
        out_of_grid_rows=count(out_of_grid),
        selected_provider_present_rows=count(selected & rows.provider_present),
        selected_provider_valid_rows=count(selected & rows.provider_valid),
        selected_valid_position_rows=count(selected & source_position_valid),
        selected_source_extent_valid_rows=count(selected & source_extent_valid),
        selected_transform_valid_rows=count(selected & transform_valid),
        selected_in_grid_rows=count(selected & in_grid),
        selected_missing_provider_rows=count(selected & missing_provider),
        selected_invalid_provider_rows=count(selected & invalid_provider),
        selected_nonfinite_position_rows=count(selected & nonfinite_position),
        selected_source_position_out_of_extent_rows=count(selected & out_of_source_extent),
        selected_transform_invalid_rows=count(selected & transform_invalid),
        selected_out_of_grid_rows=count(selected & out_of_grid),
    )

    source_rows_sha256 = rows.sha256
    result_record = {
        "authorities": authorities.as_record(),
        "selection_sha256": selection.sha256,
        "transform_sha256": transform.sha256,
        "source_rows_sha256": source_rows_sha256,
        "counts": counts.as_record(),
        "reason_counts": reason_counts,
        "selected_reason_counts": selected_reason_counts,
        "track_sample_key": rows.track_sample_key.tolist(),
        "acquisition_frame": rows.acquisition_frame.tolist(),
        "arena_position_xy": arena_position.tolist(),
        "in_selection": in_selection.tolist(),
        "transform_valid": transform_valid.tolist(),
        "source_extent_valid": source_extent_valid.tolist(),
        "in_grid": in_grid.tolist(),
        "reason_codes": [list(value) for value in reasons],
    }
    trajectory_sha256 = _sha256(result_record)
    return ProviderSpatialTrajectory(
        authorities=authorities,
        selection=selection,
        transform=transform,
        source_row_index=source_row_index,
        track_sample_key=rows.track_sample_key,
        acquisition_frame=rows.acquisition_frame,
        subject_identity=rows.subject_identity,
        track_identity=rows.track_identity,
        source_position_xy=rows.source_position_xy,
        arena_position_xy=arena_position,
        provider_present=rows.provider_present,
        provider_valid=rows.provider_valid,
        source_position_valid=source_position_valid,
        source_extent_valid=source_extent_valid,
        in_selection=in_selection,
        transform_valid=transform_valid,
        in_grid=in_grid,
        selection_membership_key=membership_keys,
        selection_occurrence_id=occurrence_ids,
        selection_role=roles,
        reason_codes=tuple(reasons),
        reason_counts=reason_counts,
        selected_reason_counts=selected_reason_counts,
        counts=counts,
        source_rows_sha256=source_rows_sha256,
        trajectory_sha256=trajectory_sha256,
    )


def prepare_provider_track_sample_trajectory(**kwargs: Any) -> ProviderSpatialTrajectory:
    """Explicitly named alias for callers using the track-sample terminology."""

    return prepare_provider_spatial_trajectory(**kwargs)


def selected_frame_membership_from_compiled_selection(
    compiled: object,
) -> SelectedFrameMembership:
    """Adapt the pure selection compiler without dropping overlap membership."""

    from fisheye.analysis_workflows.composable_stimulus_selection import (
        CompiledSelection,
        canonical_sha256,
    )

    if type(compiled) is not CompiledSelection:
        raise TypeError("compiled must be one exact CompiledSelection.")
    frames: list[int] = []
    membership_keys: list[tuple[str, ...]] = []
    occurrence_ids: list[tuple[str, ...]] = []
    roles: list[tuple[str, ...]] = []
    for interval in compiled.resolved_intervals:
        membership_digests = tuple(
            canonical_sha256(membership.to_dict())
            for membership in interval.source_memberships
        )
        occurrences = tuple(
            membership.occurrence_id for membership in interval.source_memberships
        )
        role_values = tuple(
            "unassigned" if membership.role is None else membership.role.role
            for membership in interval.source_memberships
        )
        for frame in range(interval.start_frame, interval.end_frame):
            frames.append(frame)
            membership_keys.append(
                tuple(
                    f"membership:{digest}"
                    for digest in membership_digests
                )
            )
            occurrence_ids.append(occurrences)
            roles.append(role_values)
    return SelectedFrameMembership(
        recording_id=compiled.authority.recording_id,
        timeline_authority_id=compiled.authority.timeline_id,
        selection_authority_id=compiled.resolved_digest,
        acquisition_frames=frames,
        membership_keys=membership_keys,
        occurrence_ids=occurrence_ids,
        roles=roles,
    )


__all__ = [
    "CAMERA_TO_ARENA_MM_TRANSFORM_SCHEMA_ID",
    "CAMERA_TO_ARENA_MM_TRANSFORM_SCHEMA_VERSION",
    "SOURCE_CAMERA_EXTENT_POLICY_ID",
    "ProviderSpatialTrajectory",
    "ProviderSpatialTrajectoryError",
    "ProviderTrackSamples",
    "SelectedFrameMembership",
    "SourceCameraToArenaMMTransform",
    "TRACK_SAMPLE_INPUT_SCHEMA_ID",
    "TRACK_SAMPLE_INPUT_SCHEMA_VERSION",
    "TrajectoryAuthorityIdentities",
    "TrajectoryCounts",
    "prepare_provider_spatial_trajectory",
    "prepare_provider_track_sample_trajectory",
    "selected_frame_membership_from_compiled_selection",
]
