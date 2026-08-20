"""Pure, keyed chaser-relative frame derivation.

This module deliberately stops at an in-memory result.  It is the common
framewise foundation for chaser analyses; publication, selectors, and
protocol-specific policies belong above it.

All positions use one source-camera coordinate system: top-left origin, +X
right, and +Y down.  A body frame is expressed in that same coordinate system.
Consequently, an egocentric bearing is simply ``atan2(left, forward)`` with
positive angles toward anatomical left.  No Y reflection, velocity heading,
interpolation, or invalid-row substitution is performed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


class ChaserRelativeFrameError(ValueError):
    """Raised when a chaser-relative input cannot be joined safely."""


RELATIVE_REASON_CODES = (
    "valid",
    "selection_excluded",
    "occurrence_excluded",
    "chaser_inactive",
    "fish_invalid",
    "chaser_invalid",
    "nonfinite_coordinate",
)

NEAREST_REASON_CODES = RELATIVE_REASON_CODES + (
    "no_chaser_axis",
    "no_valid_chaser",
)

BODY_REASON_CODES = (
    "valid",
    "body_frame_unavailable",
    "body_frame_invalid",
    "body_frame_nonfinite",
)

EGOCENTRIC_REASON_CODES = RELATIVE_REASON_CODES + (
    "body_frame_unavailable",
    "body_frame_invalid",
    "body_frame_nonfinite",
    "zero_relative_vector",
)

TRANSITION_REASON_CODES = (
    "valid",
    "no_predecessor",
    "nonconsecutive_acquisition_frame",
    "timestamp_unavailable",
    "nonpositive_timestamp_delta",
    "selection_boundary",
    "occurrence_boundary",
    "trial_boundary",
    "invalid_current_or_previous_position",
    "invalid_current_or_previous_body_frame",
)


def _error(message: str) -> None:
    raise ChaserRelativeFrameError(message)


def _text(value: object, *, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        _error(f"{label} must be a non-empty string.")
    return value


def _readonly_array(value: object, *, label: str) -> np.ndarray:
    try:
        array = np.array(value, copy=True)
    except Exception as exc:  # pragma: no cover - defensive conversion boundary
        _error(f"{label} cannot be converted to an array: {exc}")
    array.setflags(write=False)
    return array


def _readonly_float_array(value: object, *, label: str) -> np.ndarray:
    array = _readonly_array(value, label=label)
    if not np.issubdtype(array.dtype, np.number):
        _error(f"{label} must have a numeric dtype.")
    converted = array.astype(np.float64, copy=False)
    converted.setflags(write=False)
    return converted


def _require_shape(array: np.ndarray, shape: tuple[object, ...], *, label: str) -> None:
    if len(array.shape) != len(shape):
        _error(f"{label} must have shape {shape}; got {array.shape}.")
    for actual, expected in zip(array.shape, shape):
        if expected != "*" and actual != expected:
            _error(f"{label} must have shape {shape}; got {array.shape}.")


def _require_bool(array: np.ndarray, *, label: str) -> None:
    if array.dtype != np.dtype(bool):
        _error(f"{label} must have dtype bool; got {array.dtype}.")


def _require_int64(array: np.ndarray, *, label: str) -> None:
    if array.dtype != np.dtype(np.int64):
        _error(f"{label} must have dtype int64; got {array.dtype}.")


def _require_unique(array: np.ndarray, *, label: str) -> None:
    if array.ndim != 1:
        _error(f"{label} must be one-dimensional before uniqueness validation.")
    try:
        unique_count = np.unique(array).size
    except TypeError as exc:
        _error(f"{label} contains unsupported key values: {exc}")
    if unique_count != array.size:
        _error(f"{label} contains duplicate keys.")


def _same_array(left: np.ndarray, right: np.ndarray) -> bool:
    return left.dtype == right.dtype and left.shape == right.shape and np.array_equal(
        left, right, equal_nan=False
    )


@dataclass(frozen=True, slots=True)
class AcquisitionFrameKeys:
    """The exact acquisition-frame and track-sample row identity."""

    recording_id: str
    acquisition_frame_id: np.ndarray
    track_sample_id: np.ndarray
    row_axis_authority_id: str
    row_axis_authority_digest: str
    timestamp_ns: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "recording_id", _text(self.recording_id, label="recording_id"))
        object.__setattr__(
            self,
            "row_axis_authority_id",
            _text(self.row_axis_authority_id, label="row_axis_authority_id"),
        )
        object.__setattr__(
            self,
            "row_axis_authority_digest",
            _text(self.row_axis_authority_digest, label="row_axis_authority_digest"),
        )
        frame = _readonly_array(self.acquisition_frame_id, label="acquisition_frame_id")
        samples = _readonly_array(self.track_sample_id, label="track_sample_id")
        _require_shape(frame, ("*",), label="acquisition_frame_id")
        _require_shape(samples, (frame.shape[0],), label="track_sample_id")
        _require_int64(frame, label="acquisition_frame_id")
        _require_unique(frame, label="acquisition_frame_id")
        _require_unique(samples, label="track_sample_id")
        object.__setattr__(self, "acquisition_frame_id", frame)
        object.__setattr__(self, "track_sample_id", samples)
        if self.timestamp_ns is not None:
            timestamps = _readonly_array(self.timestamp_ns, label="timestamp_ns")
            _require_shape(timestamps, (frame.shape[0],), label="timestamp_ns")
            _require_int64(timestamps, label="timestamp_ns")
            object.__setattr__(self, "timestamp_ns", timestamps)

    @property
    def row_count(self) -> int:
        return int(self.acquisition_frame_id.shape[0])


@dataclass(frozen=True, slots=True)
class ProviderSourceAuthority:
    """Digest-bound authority for one in-memory input surface."""

    recording_id: str
    source_authority_id: str
    source_digest: str
    provider_id: str
    provider_digest: str
    coordinate_authority_id: str
    scale_authority_id: str
    timing_authority_id: str
    row_axis_authority_id: str
    row_axis_authority_digest: str

    def __post_init__(self) -> None:
        for name in (
            "recording_id",
            "source_authority_id",
            "source_digest",
            "provider_id",
            "provider_digest",
            "coordinate_authority_id",
            "scale_authority_id",
            "timing_authority_id",
            "row_axis_authority_id",
            "row_axis_authority_digest",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), label=name))


@dataclass(frozen=True, slots=True)
class CoordinatePolicy:
    """The source-camera coordinate convention shared by all position inputs."""

    coordinate_authority_id: str
    coordinate_frame: str
    policy_id: str = "source_camera_y_down_v1"
    origin: str = "top_left"
    x_axis_direction: str = "right"
    y_axis_direction: str = "down"

    def __post_init__(self) -> None:
        for name in (
            "coordinate_authority_id",
            "coordinate_frame",
            "policy_id",
            "origin",
            "x_axis_direction",
            "y_axis_direction",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), label=name))
        if self.origin != "top_left":
            _error("Coordinate policy must use a top_left origin.")
        if self.x_axis_direction != "right":
            _error("Coordinate policy must use +X right.")
        if self.y_axis_direction != "down":
            _error("Coordinate policy must use +Y down.")


@dataclass(frozen=True, slots=True)
class ScalePolicy:
    """The scale binding carried with the relative frame."""

    scale_authority_id: str
    scale_digest: str
    pixels_per_unit: float
    unit: str = "mm"
    policy_id: str = "source_camera_scale_v1"

    def __post_init__(self) -> None:
        for name in ("scale_authority_id", "scale_digest", "unit", "policy_id"):
            object.__setattr__(self, name, _text(getattr(self, name), label=name))
        value = float(self.pixels_per_unit)
        if not np.isfinite(value) or value <= 0:
            _error("pixels_per_unit must be finite and greater than zero.")
        object.__setattr__(self, "pixels_per_unit", value)


@dataclass(frozen=True, slots=True)
class TimingPolicy:
    """The acquisition timing/key interpretation binding."""

    timing_authority_id: str
    timing_digest: str
    recording_id: str
    frame_key_name: str = "acquisition_frame_id"
    track_sample_key_name: str = "track_sample_id"
    timestamp_field: Optional[str] = "timestamp_ns"
    policy_id: str = "acquisition_camera_timing_v1"

    def __post_init__(self) -> None:
        for name in (
            "timing_authority_id",
            "timing_digest",
            "recording_id",
            "frame_key_name",
            "track_sample_key_name",
            "policy_id",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), label=name))
        if self.timestamp_field is not None:
            object.__setattr__(
                self, "timestamp_field", _text(self.timestamp_field, label="timestamp_field")
            )


@dataclass(frozen=True, slots=True)
class ChaserObservations:
    """Complete chaser axis on the acquisition-frame row axis."""

    identities: tuple[str, ...]
    behavior_roles: np.ndarray
    xy: np.ndarray
    valid: np.ndarray
    source_row_index: np.ndarray
    authority: ProviderSourceAuthority
    trial_ids: Optional[np.ndarray] = None
    active: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        identities = tuple(_text(item, label="chaser identity") for item in self.identities)
        if len(set(identities)) != len(identities):
            _error("chaser identities contain duplicates.")
        object.__setattr__(self, "identities", identities)
        xy = _readonly_float_array(self.xy, label="chaser xy")
        valid = _readonly_array(self.valid, label="chaser valid")
        roles = _readonly_array(self.behavior_roles, label="chaser behavior roles")
        source_rows = _readonly_array(
            self.source_row_index,
            label="chaser source_row_index",
        )
        _require_shape(xy, ("*", len(identities), 2), label="chaser xy")
        _require_shape(valid, (xy.shape[0], len(identities)), label="chaser valid")
        _require_shape(
            roles,
            (xy.shape[0], len(identities)),
            label="chaser behavior roles",
        )
        _require_shape(
            source_rows,
            (xy.shape[0], len(identities)),
            label="chaser source_row_index",
        )
        _require_bool(valid, label="chaser valid")
        if roles.dtype.kind not in "US":
            _error("chaser behavior roles must use a fixed-width string dtype.")
        if np.any(np.char.str_len(roles.astype(str)) == 0):
            _error("chaser behavior roles cannot contain empty values.")
        _require_int64(source_rows, label="chaser source_row_index")
        object.__setattr__(self, "xy", xy)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "behavior_roles", roles)
        object.__setattr__(self, "source_row_index", source_rows)
        for name, value in (("trial_ids", self.trial_ids), ("active", self.active)):
            if value is None:
                continue
            array = _readonly_array(value, label=f"chaser {name}")
            _require_shape(array, valid.shape, label=f"chaser {name}")
            if name == "active":
                _require_bool(array, label="chaser active")
            else:
                _require_int64(array, label="chaser trial_ids")
            object.__setattr__(self, name, array)

    @property
    def row_count(self) -> int:
        return int(self.xy.shape[0])

    @property
    def chaser_count(self) -> int:
        return int(self.xy.shape[1])


@dataclass(frozen=True, slots=True)
class BodyFrameInput:
    """Separately keyed anatomical body-frame input."""

    frame_keys: AcquisitionFrameKeys
    origin_xy: np.ndarray
    forward_axis_xy: np.ndarray
    left_axis_xy: np.ndarray
    axis_valid: np.ndarray
    source_row_index: np.ndarray
    authority: ProviderSourceAuthority

    def __post_init__(self) -> None:
        origin = _readonly_float_array(self.origin_xy, label="origin_xy")
        forward = _readonly_float_array(self.forward_axis_xy, label="forward_axis_xy")
        left = _readonly_float_array(self.left_axis_xy, label="left_axis_xy")
        valid = _readonly_array(self.axis_valid, label="axis_valid")
        source_rows = _readonly_array(
            self.source_row_index,
            label="body frame source_row_index",
        )
        n = self.frame_keys.row_count
        _require_shape(origin, (n, 2), label="origin_xy")
        _require_shape(forward, (n, 2), label="forward_axis_xy")
        _require_shape(left, (n, 2), label="left_axis_xy")
        _require_shape(valid, (n,), label="axis_valid")
        _require_shape(source_rows, (n,), label="body frame source_row_index")
        _require_bool(valid, label="axis_valid")
        _require_int64(source_rows, label="body frame source_row_index")
        finite = (
            np.isfinite(origin).all(axis=1)
            & np.isfinite(forward).all(axis=1)
            & np.isfinite(left).all(axis=1)
        )
        if not np.array_equal(valid, finite):
            _error(
                "axis_valid must exactly identify complete finite body-frame geometry."
            )
        if np.any(~valid) and (
            np.any(np.isfinite(origin[~valid]))
            or np.any(np.isfinite(forward[~valid]))
            or np.any(np.isfinite(left[~valid]))
        ):
            _error("invalid body-frame rows must use NaN geometry.")
        if np.any(valid):
            forward_norm = np.linalg.norm(forward[valid], axis=1)
            left_norm = np.linalg.norm(left[valid], axis=1)
            orthogonal = np.sum(forward[valid] * left[valid], axis=1)
            determinant = (
                forward[valid, 0] * left[valid, 1]
                - forward[valid, 1] * left[valid, 0]
            )
            if np.any(~np.isclose(forward_norm, 1.0, atol=1e-6)):
                _error("valid forward axes must be unit vectors.")
            if np.any(~np.isclose(left_norm, 1.0, atol=1e-6)):
                _error("valid left axes must be unit vectors.")
            if np.any(~np.isclose(orthogonal, 0.0, atol=1e-6)):
                _error("valid body axes must be orthogonal.")
            if np.any(~np.isclose(determinant, -1.0, atol=1e-6)):
                _error(
                    "valid body axes must use anatomical-left orientation "
                    "with determinant -1 in camera XY."
                )
        object.__setattr__(self, "origin_xy", origin)
        object.__setattr__(self, "forward_axis_xy", forward)
        object.__setattr__(self, "left_axis_xy", left)
        object.__setattr__(self, "axis_valid", valid)
        object.__setattr__(self, "source_row_index", source_rows)


@dataclass(frozen=True, slots=True)
class ChaserRelativeFrameInput:
    """All immutable, keyed inputs for one relative-frame computation."""

    frame_keys: AcquisitionFrameKeys
    fish_xy: np.ndarray
    fish_valid: np.ndarray
    fish_source_row_index: np.ndarray
    fish_authority: ProviderSourceAuthority
    chasers: ChaserObservations
    selection_membership: np.ndarray
    occurrence_membership: np.ndarray
    coordinate_policy: CoordinatePolicy
    scale_policy: ScalePolicy
    timing_policy: TimingPolicy
    body_frame: Optional[BodyFrameInput] = None

    def __post_init__(self) -> None:
        n = self.frame_keys.row_count
        fish_xy = _readonly_float_array(self.fish_xy, label="fish xy")
        fish_valid = _readonly_array(self.fish_valid, label="fish valid")
        fish_source_rows = _readonly_array(
            self.fish_source_row_index,
            label="fish source_row_index",
        )
        selection = _readonly_array(self.selection_membership, label="selection membership")
        occurrence = _readonly_array(self.occurrence_membership, label="occurrence membership")
        _require_shape(fish_xy, (n, 2), label="fish xy")
        _require_shape(fish_valid, (n,), label="fish valid")
        _require_shape(fish_source_rows, (n,), label="fish source_row_index")
        _require_shape(selection, (n,), label="selection membership")
        _require_shape(occurrence, (n, self.chasers.chaser_count), label="occurrence membership")
        _require_bool(fish_valid, label="fish valid")
        _require_int64(fish_source_rows, label="fish source_row_index")
        _require_bool(selection, label="selection membership")
        _require_bool(occurrence, label="occurrence membership")
        if self.chasers.row_count != n:
            _error("chaser arrays must share the exact acquisition-frame row count.")
        object.__setattr__(self, "fish_xy", fish_xy)
        object.__setattr__(self, "fish_valid", fish_valid)
        object.__setattr__(self, "fish_source_row_index", fish_source_rows)
        object.__setattr__(self, "selection_membership", selection)
        object.__setattr__(self, "occurrence_membership", occurrence)


@dataclass(frozen=True, slots=True)
class ChaserRelativeFrameResult:
    """Immutable framewise relative geometry and optional body-frame outputs."""

    frame_keys: AcquisitionFrameKeys
    chaser_identities: tuple[str, ...]
    chaser_behavior_roles: np.ndarray
    selection_membership: np.ndarray
    occurrence_membership: np.ndarray
    chaser_trial_ids: Optional[np.ndarray]
    chaser_active: Optional[np.ndarray]
    fish_xy: np.ndarray
    fish_valid: np.ndarray
    fish_source_row_index: np.ndarray
    chaser_xy: np.ndarray
    chaser_valid: np.ndarray
    chaser_source_row_index: np.ndarray
    acquisition_frame_delta: np.ndarray
    timestamp_delta_ns: np.ndarray
    fish_transition_valid: np.ndarray
    fish_transition_reason_code: np.ndarray
    relative_xy: np.ndarray
    relative_xy_physical: np.ndarray
    distance_px: np.ndarray
    distance_physical: np.ndarray
    relative_valid: np.ndarray
    relative_reason_code: np.ndarray
    relative_transition_valid: np.ndarray
    relative_transition_reason_code: np.ndarray
    nearest_chaser_index: np.ndarray
    nearest_chaser_identity: tuple[Optional[str], ...]
    nearest_distance_px: np.ndarray
    nearest_distance_physical: np.ndarray
    nearest_valid: np.ndarray
    nearest_reason_code: np.ndarray
    body_frame_present: bool
    body_frame_origin_xy: np.ndarray
    body_frame_forward_axis_xy: np.ndarray
    body_frame_left_axis_xy: np.ndarray
    body_frame_heading_deg: np.ndarray
    body_frame_source_row_index: np.ndarray
    body_frame_valid: np.ndarray
    body_frame_reason_code: np.ndarray
    heading_transition_valid: np.ndarray
    heading_transition_reason_code: np.ndarray
    body_relative_xy: np.ndarray
    body_relative_xy_physical: np.ndarray
    forward_coordinate_px: np.ndarray
    left_coordinate_px: np.ndarray
    forward_coordinate_physical: np.ndarray
    left_coordinate_physical: np.ndarray
    egocentric_bearing_deg: np.ndarray
    egocentric_valid: np.ndarray
    egocentric_reason_code: np.ndarray
    fish_authority: ProviderSourceAuthority
    chaser_authority: ProviderSourceAuthority
    body_frame_authority: Optional[ProviderSourceAuthority]
    coordinate_policy: CoordinatePolicy
    scale_policy: ScalePolicy
    timing_policy: TimingPolicy

    def __post_init__(self) -> None:
        n = self.frame_keys.row_count
        m = len(self.chaser_identities)
        array_specs = (
            ("chaser_behavior_roles", (n, m)),
            ("selection_membership", (n,)),
            ("occurrence_membership", (n, m)),
            ("fish_xy", (n, 2)),
            ("fish_valid", (n,)),
            ("fish_source_row_index", (n,)),
            ("chaser_xy", (n, m, 2)),
            ("chaser_valid", (n, m)),
            ("chaser_source_row_index", (n, m)),
            ("acquisition_frame_delta", (n,)),
            ("timestamp_delta_ns", (n,)),
            ("fish_transition_valid", (n,)),
            ("fish_transition_reason_code", (n,)),
            ("relative_xy", (n, m, 2)),
            ("relative_xy_physical", (n, m, 2)),
            ("distance_px", (n, m)),
            ("distance_physical", (n, m)),
            ("relative_valid", (n, m)),
            ("relative_reason_code", (n, m)),
            ("relative_transition_valid", (n, m)),
            ("relative_transition_reason_code", (n, m)),
            ("nearest_chaser_index", (n,)),
            ("nearest_distance_px", (n,)),
            ("nearest_distance_physical", (n,)),
            ("nearest_valid", (n,)),
            ("nearest_reason_code", (n,)),
            ("body_frame_origin_xy", (n, 2)),
            ("body_frame_forward_axis_xy", (n, 2)),
            ("body_frame_left_axis_xy", (n, 2)),
            ("body_frame_heading_deg", (n,)),
            ("body_frame_source_row_index", (n,)),
            ("body_frame_valid", (n,)),
            ("body_frame_reason_code", (n,)),
            ("heading_transition_valid", (n,)),
            ("heading_transition_reason_code", (n,)),
            ("body_relative_xy", (n, m, 2)),
            ("body_relative_xy_physical", (n, m, 2)),
            ("forward_coordinate_px", (n, m)),
            ("left_coordinate_px", (n, m)),
            ("forward_coordinate_physical", (n, m)),
            ("left_coordinate_physical", (n, m)),
            ("egocentric_bearing_deg", (n, m)),
            ("egocentric_valid", (n, m)),
            ("egocentric_reason_code", (n, m)),
        )
        for name, shape in array_specs:
            array = _readonly_array(getattr(self, name), label=name)
            _require_shape(array, shape, label=name)
            if name.endswith("membership") or name.endswith("valid"):
                _require_bool(array, label=name)
            object.__setattr__(self, name, array)
        identities = tuple(
            _text(value, label="chaser identity") for value in self.chaser_identities
        )
        if len(set(identities)) != len(identities):
            _error("chaser identities contain duplicates.")
        object.__setattr__(self, "chaser_identities", identities)
        if self.chaser_behavior_roles.dtype.kind not in "US":
            _error("chaser_behavior_roles must use a fixed-width string dtype.")
        for name in (
            "fish_source_row_index",
            "chaser_source_row_index",
            "body_frame_source_row_index",
            "acquisition_frame_delta",
            "timestamp_delta_ns",
        ):
            _require_int64(getattr(self, name), label=name)
        if len(self.nearest_chaser_identity) != n:
            _error("nearest_chaser_identity must have one entry per acquisition frame.")
        for name, value in (("chaser_trial_ids", self.chaser_trial_ids), ("chaser_active", self.chaser_active)):
            if value is not None:
                array = _readonly_array(value, label=name)
                _require_shape(array, (n, m), label=name)
                if name == "chaser_active":
                    _require_bool(array, label=name)
                else:
                    _require_int64(array, label=name)
                object.__setattr__(self, name, array)


def _require_exact_frame_keys(
    expected: AcquisitionFrameKeys,
    observed: AcquisitionFrameKeys,
    *,
    label: str,
) -> None:
    if expected.recording_id != observed.recording_id:
        _error(f"{label} recording identity mismatch.")
    if expected.row_axis_authority_id != observed.row_axis_authority_id:
        _error(f"{label} row-axis authority ID mismatch.")
    if expected.row_axis_authority_digest != observed.row_axis_authority_digest:
        _error(f"{label} row-axis authority digest mismatch.")
    if not _same_array(expected.acquisition_frame_id, observed.acquisition_frame_id):
        _error(f"{label} acquisition-frame key/order mismatch.")
    if not _same_array(expected.track_sample_id, observed.track_sample_id):
        _error(f"{label} track-sample key/order mismatch.")
    if (expected.timestamp_ns is None) != (observed.timestamp_ns is None):
        _error(f"{label} timestamp key presence mismatch.")
    if expected.timestamp_ns is not None and not _same_array(
        expected.timestamp_ns, observed.timestamp_ns  # type: ignore[arg-type]
    ):
        _error(f"{label} timestamp key/order mismatch.")


def _validate_authority(
    authority: ProviderSourceAuthority,
    *,
    frame_keys: AcquisitionFrameKeys,
    coordinate: CoordinatePolicy,
    scale: ScalePolicy,
    timing: TimingPolicy,
    label: str,
) -> None:
    if authority.recording_id != frame_keys.recording_id:
        _error(f"{label} recording authority mismatch.")
    if authority.row_axis_authority_id != frame_keys.row_axis_authority_id:
        _error(f"{label} row-axis authority ID mismatch.")
    if authority.row_axis_authority_digest != frame_keys.row_axis_authority_digest:
        _error(f"{label} row-axis authority digest mismatch.")
    if authority.coordinate_authority_id != coordinate.coordinate_authority_id:
        _error(f"{label} coordinate-authority mismatch.")
    if authority.scale_authority_id != scale.scale_authority_id:
        _error(f"{label} scale-authority mismatch.")
    if authority.timing_authority_id != timing.timing_authority_id:
        _error(f"{label} timing-authority mismatch.")
    if not authority.source_digest or not authority.provider_digest:
        _error(f"{label} source/provider digest is missing.")


def _validate_provider_digest_consistency(
    authorities: tuple[tuple[str, ProviderSourceAuthority], ...]
) -> None:
    by_provider: dict[str, str] = {}
    by_source: dict[str, str] = {}
    for label, authority in authorities:
        previous_provider = by_provider.get(authority.provider_id)
        if previous_provider is not None and previous_provider != authority.provider_digest:
            _error(
                f"provider digest mismatch for provider {authority.provider_id!r} "
                f"({label})."
            )
        by_provider[authority.provider_id] = authority.provider_digest
        previous_source = by_source.get(authority.source_authority_id)
        if previous_source is not None and previous_source != authority.source_digest:
            _error(
                f"source digest mismatch for source {authority.source_authority_id!r} "
                f"({label})."
            )
        by_source[authority.source_authority_id] = authority.source_digest


def _reason_array(shape: tuple[int, ...], value: str) -> np.ndarray:
    return np.full(shape, value, dtype="<U48")


def _base_transition_reasons(
    frames: AcquisitionFrameKeys,
    selection: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = frames.row_count
    frame_delta = np.full(n, -1, dtype=np.int64)
    timestamp_delta = np.full(n, -1, dtype=np.int64)
    reason = _reason_array((n,), "no_predecessor")
    if n <= 1:
        return frame_delta, timestamp_delta, reason
    frame_delta[1:] = np.diff(frames.acquisition_frame_id)
    if frames.timestamp_ns is not None:
        timestamp_delta[1:] = np.diff(frames.timestamp_ns)
    for row in range(1, n):
        if frame_delta[row] != 1:
            reason[row] = "nonconsecutive_acquisition_frame"
        elif frames.timestamp_ns is None:
            reason[row] = "timestamp_unavailable"
        elif timestamp_delta[row] <= 0:
            reason[row] = "nonpositive_timestamp_delta"
        elif not (selection[row - 1] and selection[row]):
            reason[row] = "selection_boundary"
        else:
            reason[row] = "valid"
    return frame_delta, timestamp_delta, reason


def compute_chaser_relative_frame(
    inputs: ChaserRelativeFrameInput,
) -> ChaserRelativeFrameResult:
    """Compute exact framewise fish-to-chaser geometry in memory.

    The chaser axis is never reduced.  ``nearest_chaser_index`` is only a
    convenience projection over the complete ``(frame, chaser)`` arrays and
    uses the lowest chaser index when distances tie exactly.
    """

    frames = inputs.frame_keys
    chasers = inputs.chasers
    coordinate = inputs.coordinate_policy
    scale = inputs.scale_policy
    timing = inputs.timing_policy
    if timing.recording_id != frames.recording_id:
        _error("timing policy recording identity mismatch.")
    _validate_authority(
        inputs.fish_authority,
        frame_keys=frames,
        coordinate=coordinate,
        scale=scale,
        timing=timing,
        label="fish",
    )
    _validate_authority(
        chasers.authority,
        frame_keys=frames,
        coordinate=coordinate,
        scale=scale,
        timing=timing,
        label="chaser",
    )
    authorities = [("fish", inputs.fish_authority), ("chaser", chasers.authority)]
    body_authority: Optional[ProviderSourceAuthority] = None
    body = inputs.body_frame
    if body is not None:
        _require_exact_frame_keys(frames, body.frame_keys, label="body frame")
        _validate_authority(
            body.authority,
            frame_keys=frames,
            coordinate=coordinate,
            scale=scale,
            timing=timing,
            label="body frame",
        )
        authorities.append(("body frame", body.authority))
        body_authority = body.authority
    _validate_provider_digest_consistency(tuple(authorities))

    n = frames.row_count
    m = chasers.chaser_count
    fish_xy = inputs.fish_xy
    chaser_xy = chasers.xy
    selection = inputs.selection_membership
    occurrence = inputs.occurrence_membership
    fish_finite = np.isfinite(fish_xy).all(axis=1)
    chaser_finite = np.isfinite(chaser_xy).all(axis=2)
    active = (
        np.ones((n, m), dtype=bool)
        if chasers.active is None
        else np.asarray(chasers.active, dtype=bool)
    )
    relative_xy = np.full((n, m, 2), np.nan, dtype=np.float64)
    distance_px = np.full((n, m), np.nan, dtype=np.float64)
    reason = _reason_array((n, m), "valid")

    def apply_reason(mask: np.ndarray, value: str) -> None:
        reason[np.broadcast_to(mask, reason.shape)] = value

    apply_reason(~selection[:, None], "selection_excluded")
    apply_reason(selection[:, None] & ~occurrence, "occurrence_excluded")
    apply_reason(selection[:, None] & occurrence & ~active, "chaser_inactive")
    apply_reason(selection[:, None] & occurrence & active & ~inputs.fish_valid[:, None], "fish_invalid")
    apply_reason(
        selection[:, None]
        & occurrence
        & active
        & inputs.fish_valid[:, None]
        & ~chasers.valid,
        "chaser_invalid",
    )
    candidate = (
        selection[:, None]
        & occurrence
        & active
        & inputs.fish_valid[:, None]
        & chasers.valid
    )
    apply_reason(candidate & ~(fish_finite[:, None] & chaser_finite), "nonfinite_coordinate")
    relative_valid = reason == "valid"
    if np.any(relative_valid):
        relative_xy[relative_valid] = (
            chaser_xy - fish_xy[:, None, :]
        )[relative_valid]
        distance_px[relative_valid] = np.linalg.norm(relative_xy[relative_valid], axis=1)
    relative_xy.setflags(write=False)
    distance_px.setflags(write=False)
    relative_xy_physical = relative_xy / scale.pixels_per_unit
    distance_physical = distance_px / scale.pixels_per_unit
    relative_xy_physical.setflags(write=False)
    distance_physical.setflags(write=False)

    frame_delta, timestamp_delta, base_transition_reason = (
        _base_transition_reasons(frames, selection)
    )
    fish_transition_reason = base_transition_reason.copy()
    fish_transition_valid = np.zeros(n, dtype=bool)
    relative_transition_reason = np.broadcast_to(
        base_transition_reason[:, None],
        (n, m),
    ).copy()
    relative_transition_valid = np.zeros((n, m), dtype=bool)
    for row in range(1, n):
        if base_transition_reason[row] != "valid":
            continue
        if not (
            inputs.fish_valid[row - 1]
            and inputs.fish_valid[row]
            and fish_finite[row - 1]
            and fish_finite[row]
        ):
            fish_transition_reason[row] = (
                "invalid_current_or_previous_position"
            )
        else:
            fish_transition_reason[row] = "valid"
            fish_transition_valid[row] = True
        for column in range(m):
            if not (
                occurrence[row - 1, column]
                and occurrence[row, column]
            ):
                relative_transition_reason[row, column] = "occurrence_boundary"
                continue
            if chasers.trial_ids is not None and (
                chasers.trial_ids[row - 1, column]
                != chasers.trial_ids[row, column]
            ):
                relative_transition_reason[row, column] = "trial_boundary"
                continue
            if not (
                relative_valid[row - 1, column]
                and relative_valid[row, column]
            ):
                relative_transition_reason[row, column] = (
                    "invalid_current_or_previous_position"
                )
                continue
            relative_transition_reason[row, column] = "valid"
            relative_transition_valid[row, column] = True
    relative_valid.setflags(write=False)
    reason.setflags(write=False)

    nearest_index = np.full(n, -1, dtype=np.int64)
    nearest_distance = np.full(n, np.nan, dtype=np.float64)
    nearest_distance_physical = np.full(n, np.nan, dtype=np.float64)
    nearest_valid = np.zeros(n, dtype=bool)
    nearest_reason = _reason_array((n,), "no_valid_chaser")
    nearest_identity: list[Optional[str]] = [None] * n
    if m == 0:
        nearest_reason[:] = "no_chaser_axis"
    else:
        for frame_index in range(n):
            valid_indices = np.flatnonzero(relative_valid[frame_index])
            if valid_indices.size:
                # flatnonzero preserves axis order; argmin therefore makes the
                # lowest index the deterministic exact-distance tie winner.
                local = valid_indices[np.argmin(distance_px[frame_index, valid_indices])]
                nearest_index[frame_index] = int(local)
                nearest_distance[frame_index] = distance_px[frame_index, local]
                nearest_distance_physical[frame_index] = distance_physical[
                    frame_index,
                    local,
                ]
                nearest_valid[frame_index] = True
                nearest_reason[frame_index] = "valid"
                nearest_identity[frame_index] = chasers.identities[int(local)]
            elif not selection[frame_index]:
                nearest_reason[frame_index] = "selection_excluded"
            elif not np.any(occurrence[frame_index]):
                nearest_reason[frame_index] = "occurrence_excluded"

    body_frame_present = body is not None
    body_origin = np.full((n, 2), np.nan, dtype=np.float64)
    body_forward = np.full((n, 2), np.nan, dtype=np.float64)
    body_left = np.full((n, 2), np.nan, dtype=np.float64)
    body_heading = np.full(n, np.nan, dtype=np.float64)
    body_source_rows = np.full(n, -1, dtype=np.int64)
    body_valid = np.zeros(n, dtype=bool)
    body_reason = _reason_array(
        (n,), "body_frame_unavailable" if body is None else "body_frame_invalid"
    )
    forward = np.full((n, m), np.nan, dtype=np.float64)
    left = np.full((n, m), np.nan, dtype=np.float64)
    body_relative = np.full((n, m, 2), np.nan, dtype=np.float64)
    bearing = np.full((n, m), np.nan, dtype=np.float64)
    ego_valid = np.zeros((n, m), dtype=bool)
    ego_reason = reason.copy()
    heading_transition_valid = np.zeros(n, dtype=bool)
    heading_transition_reason = base_transition_reason.copy()
    if body is not None:
        body_origin = np.asarray(body.origin_xy, dtype=np.float64).copy()
        body_forward = np.asarray(body.forward_axis_xy, dtype=np.float64).copy()
        body_left = np.asarray(body.left_axis_xy, dtype=np.float64).copy()
        body_source_rows = np.asarray(body.source_row_index, dtype=np.int64).copy()
        axis_finite = (
            np.isfinite(body_origin).all(axis=1)
            & np.isfinite(body_forward).all(axis=1)
            & np.isfinite(body_left).all(axis=1)
        )
        body_valid = body.axis_valid & axis_finite
        body_reason[:] = "body_frame_invalid"
        body_reason[~body.axis_valid & axis_finite] = "body_frame_invalid"
        body_reason[body.axis_valid & ~axis_finite] = "body_frame_nonfinite"
        body_reason[body_valid] = "valid"
        body_heading[body_valid] = np.rad2deg(
            np.arctan2(
                -body_forward[body_valid, 1],
                body_forward[body_valid, 0],
            )
        )
        for row in range(1, n):
            if base_transition_reason[row] != "valid":
                continue
            if not (body_valid[row - 1] and body_valid[row]):
                heading_transition_reason[row] = (
                    "invalid_current_or_previous_body_frame"
                )
                continue
            heading_transition_reason[row] = "valid"
            heading_transition_valid[row] = True
        body_relative_values = chaser_xy - body_origin[:, None, :]
        body_pair = (
            selection[:, None]
            & occurrence
            & active
            & chasers.valid
            & chaser_finite
            & body_valid[:, None]
        )
        body_relative[body_pair] = body_relative_values[body_pair]
        if np.any(body_pair):
            forward_values = np.sum(
                body_relative_values * body_forward[:, None, :], axis=2
            )
            left_values = np.sum(
                body_relative_values * body_left[:, None, :],
                axis=2,
            )
            forward[body_pair] = forward_values[body_pair]
            left[body_pair] = left_values[body_pair]
            body_distance = np.linalg.norm(body_relative_values, axis=2)
            nonzero = body_pair & (body_distance > 0.0)
            bearing[nonzero] = np.rad2deg(np.arctan2(left[nonzero], forward[nonzero]))
            ego_valid[nonzero] = True
            ego_reason[nonzero] = "valid"
            zero = body_pair & ~nonzero
            ego_reason[zero] = "zero_relative_vector"
        ego_candidate = (
            selection[:, None]
            & occurrence
            & active
            & chasers.valid
            & chaser_finite
        )
        ego_reason[ego_candidate & ~body_valid[:, None]] = np.broadcast_to(
            body_reason[:, None],
            (n, m),
        )[ego_candidate & ~body_valid[:, None]]
        for frame_index in range(n):
            if body_valid[frame_index]:
                continue
            ego_reason[frame_index, ego_candidate[frame_index]] = body_reason[frame_index]
    else:
        ego_reason[:] = np.where(relative_valid, "body_frame_unavailable", reason)
        heading_transition_reason[base_transition_reason == "valid"] = (
            "invalid_current_or_previous_body_frame"
        )

    forward_physical = forward / scale.pixels_per_unit
    left_physical = left / scale.pixels_per_unit
    body_relative_physical = body_relative / scale.pixels_per_unit

    return ChaserRelativeFrameResult(
        frame_keys=frames,
        chaser_identities=chasers.identities,
        chaser_behavior_roles=chasers.behavior_roles,
        selection_membership=selection,
        occurrence_membership=occurrence,
        chaser_trial_ids=chasers.trial_ids,
        chaser_active=chasers.active,
        fish_xy=inputs.fish_xy,
        fish_valid=inputs.fish_valid,
        fish_source_row_index=inputs.fish_source_row_index,
        chaser_xy=chasers.xy,
        chaser_valid=chasers.valid,
        chaser_source_row_index=chasers.source_row_index,
        acquisition_frame_delta=frame_delta,
        timestamp_delta_ns=timestamp_delta,
        fish_transition_valid=fish_transition_valid,
        fish_transition_reason_code=fish_transition_reason,
        relative_xy=relative_xy,
        relative_xy_physical=relative_xy_physical,
        distance_px=distance_px,
        distance_physical=distance_physical,
        relative_valid=relative_valid,
        relative_reason_code=reason,
        relative_transition_valid=relative_transition_valid,
        relative_transition_reason_code=relative_transition_reason,
        nearest_chaser_index=nearest_index,
        nearest_chaser_identity=tuple(nearest_identity),
        nearest_distance_px=nearest_distance,
        nearest_distance_physical=nearest_distance_physical,
        nearest_valid=nearest_valid,
        nearest_reason_code=nearest_reason,
        body_frame_present=body_frame_present,
        body_frame_origin_xy=body_origin,
        body_frame_forward_axis_xy=body_forward,
        body_frame_left_axis_xy=body_left,
        body_frame_heading_deg=body_heading,
        body_frame_source_row_index=body_source_rows,
        body_frame_valid=body_valid,
        body_frame_reason_code=body_reason,
        heading_transition_valid=heading_transition_valid,
        heading_transition_reason_code=heading_transition_reason,
        body_relative_xy=body_relative,
        body_relative_xy_physical=body_relative_physical,
        forward_coordinate_px=forward,
        left_coordinate_px=left,
        forward_coordinate_physical=forward_physical,
        left_coordinate_physical=left_physical,
        egocentric_bearing_deg=bearing,
        egocentric_valid=ego_valid,
        egocentric_reason_code=ego_reason,
        fish_authority=inputs.fish_authority,
        chaser_authority=chasers.authority,
        body_frame_authority=body_authority,
        coordinate_policy=coordinate,
        scale_policy=scale,
        timing_policy=timing,
    )


# These aliases make the pure foundation easy to discover without introducing
# a second implementation name in downstream callers.
derive_chaser_relative_frame = compute_chaser_relative_frame
ProviderAuthority = ProviderSourceAuthority


__all__ = [
    "AcquisitionFrameKeys",
    "BodyFrameInput",
    "ChaserObservations",
    "ChaserRelativeFrameError",
    "ChaserRelativeFrameInput",
    "ChaserRelativeFrameResult",
    "CoordinatePolicy",
    "ProviderAuthority",
    "ProviderSourceAuthority",
    "ScalePolicy",
    "TimingPolicy",
    "BODY_REASON_CODES",
    "EGOCENTRIC_REASON_CODES",
    "NEAREST_REASON_CODES",
    "RELATIVE_REASON_CODES",
    "compute_chaser_relative_frame",
    "derive_chaser_relative_frame",
]
