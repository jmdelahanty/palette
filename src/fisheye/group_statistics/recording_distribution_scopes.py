"""Paradigm-neutral scope records and exact row-membership builders.

Scopes describe *where* a metric is reduced.  They deliberately do not know
which metric is being reduced or how its histogram is binned.  A protocol
adapter may therefore supply semantic frame intervals while a video-only
recording supplies only the whole-session scope, and both use the same reducer.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


SCOPE_SCHEMA_ID = "palette.analysis.recording_distribution_scope"
SCOPE_SCHEMA_VERSION = 1
SCOPE_REGISTRY_SCHEMA_ID = "palette.analysis.recording_distribution_scope_registry"
SCOPE_REGISTRY_SCHEMA_VERSION = 1

WHOLE_SESSION_SCOPE_ID = "whole_session"
WHOLE_SESSION_PROVIDER_ID = "whole_session.v1"

_SAFE_ID = re.compile(r"[a-z][a-z0-9_.-]*\Z")
_AXIS_KINDS = frozenset({"all", "acquisition_frame", "session_time_ns"})
_OVERLAP_POLICIES = frozenset(
    {"not_applicable", "mutually_exclusive_within_provider", "overlap_allowed"}
)


class RecordingDistributionScopeError(ValueError):
    """A scope or its row projection is ambiguous or internally inconsistent."""


def _text(value: object, *, field: str, identifier: bool = False) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise RecordingDistributionScopeError(
            f"{field} must be one nonempty stripped string."
        )
    if identifier and _SAFE_ID.fullmatch(value) is None:
        raise RecordingDistributionScopeError(
            f"{field} must match {_SAFE_ID.pattern!r}."
        )
    return value


def _plain_mapping(value: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RecordingDistributionScopeError(f"{field} must be one object.")
    result = json_attr_safe(dict(value))
    if not isinstance(result, dict):  # defensive: json_attr_safe preserves mappings
        raise RecordingDistributionScopeError(f"{field} is not JSON-object-safe.")
    return result


@dataclass(frozen=True, slots=True)
class RecordingDistributionScope:
    """One ordered scope with an exact coordinate interval and source binding."""

    scope_id: str
    scope_label: str
    scope_family: str
    scope_provider_id: str
    order: int
    axis_kind: str
    overlap_policy: str
    start_inclusive: int | None
    end_exclusive: int | None
    source_binding: Mapping[str, Any]

    def __post_init__(self) -> None:
        for field in ("scope_id", "scope_family", "scope_provider_id"):
            _text(getattr(self, field), field=field, identifier=True)
        _text(self.scope_label, field="scope_label")
        if type(self.order) is not int or self.order < 0:
            raise RecordingDistributionScopeError(
                "scope order must be one nonnegative integer."
            )
        if self.axis_kind not in _AXIS_KINDS:
            raise RecordingDistributionScopeError(
                f"Unsupported scope axis_kind {self.axis_kind!r}."
            )
        if self.overlap_policy not in _OVERLAP_POLICIES:
            raise RecordingDistributionScopeError(
                f"Unsupported overlap_policy {self.overlap_policy!r}."
            )
        start = self.start_inclusive
        stop = self.end_exclusive
        if self.axis_kind == "all":
            if (
                start is not None
                or stop is not None
                or self.overlap_policy != "not_applicable"
            ):
                raise RecordingDistributionScopeError(
                    "The all-rows scope cannot carry interval bounds or overlap."
                )
        elif (
            type(start) is not int
            or type(stop) is not int
            or start < 0
            or stop <= start
        ):
            raise RecordingDistributionScopeError(
                "An interval scope requires nonnegative increasing integer bounds."
            )
        plain = _plain_mapping(self.source_binding, field="source_binding")
        if not plain:
            raise RecordingDistributionScopeError(
                "scope source_binding must not be empty."
            )
        object.__setattr__(self, "source_binding", MappingProxyType(plain))

    @property
    def record(self) -> Mapping[str, Any]:
        body = {
            "schema_id": SCOPE_SCHEMA_ID,
            "schema_version": SCOPE_SCHEMA_VERSION,
            "scope_id": self.scope_id,
            "scope_label": self.scope_label,
            "scope_family": self.scope_family,
            "scope_provider_id": self.scope_provider_id,
            "order": self.order,
            "axis_kind": self.axis_kind,
            "overlap_policy": self.overlap_policy,
            "start_inclusive": self.start_inclusive,
            "end_exclusive": self.end_exclusive,
            "interval_semantics": (
                None if self.axis_kind == "all" else "half_open_start_inclusive"
            ),
            "source_binding": dict(self.source_binding),
        }
        return MappingProxyType(
            {**body, "scope_sha256": canonical_json_sha256(body)}
        )


@dataclass(frozen=True, slots=True)
class ScopeMaskProjection:
    """Masks plus rows that could not be placed on a requested time axis."""

    masks: Mapping[str, np.ndarray]
    uncovered: Mapping[str, np.ndarray]
    membership_policy_id: str


def whole_session_scope() -> RecordingDistributionScope:
    return RecordingDistributionScope(
        scope_id=WHOLE_SESSION_SCOPE_ID,
        scope_label="Whole session",
        scope_family="whole_session",
        scope_provider_id=WHOLE_SESSION_PROVIDER_ID,
        order=0,
        axis_kind="all",
        overlap_policy="not_applicable",
        start_inclusive=None,
        end_exclusive=None,
        source_binding={"selection": "all_source_rows_or_events"},
    )


def frame_interval_scope(
    *,
    scope_id: str,
    scope_label: str,
    scope_family: str,
    scope_provider_id: str,
    order: int,
    start_frame: int,
    end_frame_exclusive: int,
    source_binding: Mapping[str, Any],
    overlap_policy: str = "mutually_exclusive_within_provider",
) -> RecordingDistributionScope:
    return RecordingDistributionScope(
        scope_id=scope_id,
        scope_label=scope_label,
        scope_family=scope_family,
        scope_provider_id=scope_provider_id,
        order=order,
        axis_kind="acquisition_frame",
        overlap_policy=overlap_policy,
        start_inclusive=start_frame,
        end_exclusive=end_frame_exclusive,
        source_binding=source_binding,
    )


def session_time_bracket_scope(
    *,
    scope_id: str,
    scope_label: str,
    order: int,
    start_timestamp_ns_session: int,
    end_timestamp_ns_session_exclusive: int,
    timebase_binding: Mapping[str, Any],
) -> RecordingDistributionScope:
    return RecordingDistributionScope(
        scope_id=scope_id,
        scope_label=scope_label,
        scope_family="named_session_time_bracket",
        scope_provider_id="named_session_time_brackets.v1",
        order=order,
        axis_kind="session_time_ns",
        overlap_policy="overlap_allowed",
        start_inclusive=start_timestamp_ns_session,
        end_exclusive=end_timestamp_ns_session_exclusive,
        source_binding={
            "timebase": _plain_mapping(timebase_binding, field="timebase_binding"),
            "requested_bounds_unit": "ns_session",
            "interpolation": "prohibited",
        },
    )


def validate_scope_registry(
    scopes: Sequence[RecordingDistributionScope],
) -> tuple[RecordingDistributionScope, ...]:
    result = tuple(scopes)
    if not result:
        raise RecordingDistributionScopeError("At least one scope is required.")
    ids = [scope.scope_id for scope in result]
    orders = [scope.order for scope in result]
    if len(set(ids)) != len(ids):
        raise RecordingDistributionScopeError("Scope IDs must be unique.")
    if len(set(orders)) != len(orders) or sorted(orders) != list(range(len(result))):
        raise RecordingDistributionScopeError(
            "Scope orders must be unique and gapless from zero."
        )
    ordered = tuple(sorted(result, key=lambda scope: scope.order))
    first = ordered[0]
    if (
        first.scope_id != WHOLE_SESSION_SCOPE_ID
        or first.axis_kind != "all"
        or first.scope_provider_id != WHOLE_SESSION_PROVIDER_ID
    ):
        raise RecordingDistributionScopeError(
            "The first scope must be the canonical whole-session scope."
        )
    bounded = [scope for scope in ordered if scope.axis_kind != "all"]
    for index, left in enumerate(bounded):
        for right in bounded[index + 1 :]:
            if (
                left.scope_provider_id == right.scope_provider_id
                and left.axis_kind == right.axis_kind
                and left.overlap_policy == "mutually_exclusive_within_provider"
                and right.overlap_policy == "mutually_exclusive_within_provider"
                and max(int(left.start_inclusive), int(right.start_inclusive))
                < min(int(left.end_exclusive), int(right.end_exclusive))
            ):
                raise RecordingDistributionScopeError(
                    f"Scopes {left.scope_id!r} and {right.scope_id!r} overlap "
                    "inside one provider axis."
                )
    return ordered


def scope_registry_record(
    scopes: Sequence[RecordingDistributionScope],
) -> Mapping[str, Any]:
    ordered = validate_scope_registry(scopes)
    body = {
        "schema_id": SCOPE_REGISTRY_SCHEMA_ID,
        "schema_version": SCOPE_REGISTRY_SCHEMA_VERSION,
        "scope_order": [scope.scope_id for scope in ordered],
        "scopes": [dict(scope.record) for scope in ordered],
    }
    return MappingProxyType(
        {**body, "scope_registry_sha256": canonical_json_sha256(body)}
    )


def _aligned_int(values: Any, *, field: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in "iu":
        raise RecordingDistributionScopeError(
            f"{field} must be one one-dimensional integer array."
        )
    return array.astype(np.int64, copy=False)


def sample_scope_masks(
    scopes: Sequence[RecordingDistributionScope],
    *,
    acquisition_frame_id: Any | None = None,
    timestamp_ns_session: Any | None = None,
    timestamp_valid: Any | None = None,
) -> ScopeMaskProjection:
    """Project frame/sample rows without interpolation."""

    ordered = validate_scope_registry(scopes)
    frames = (
        None
        if acquisition_frame_id is None
        else _aligned_int(acquisition_frame_id, field="acquisition_frame_id")
    )
    timestamps = (
        None
        if timestamp_ns_session is None
        else _aligned_int(timestamp_ns_session, field="timestamp_ns_session")
    )
    if frames is None and timestamps is None:
        raise RecordingDistributionScopeError(
            "Sample projection requires a frame or session-time axis."
        )
    row_count = int((frames if frames is not None else timestamps).shape[0])
    if frames is not None and frames.shape != (row_count,):
        raise RecordingDistributionScopeError("Frame axes do not align.")
    if timestamps is not None and timestamps.shape != (row_count,):
        raise RecordingDistributionScopeError("Time axes do not align.")
    if timestamp_valid is None:
        time_valid = np.ones(row_count, dtype=bool)
    else:
        time_valid = np.asarray(timestamp_valid, dtype=bool)
        if time_valid.shape != (row_count,):
            raise RecordingDistributionScopeError("timestamp_valid does not align.")

    masks: dict[str, np.ndarray] = {}
    uncovered: dict[str, np.ndarray] = {}
    for scope in ordered:
        if scope.axis_kind == "all":
            mask = np.ones(row_count, dtype=bool)
            missing = np.zeros(row_count, dtype=bool)
        elif scope.axis_kind == "acquisition_frame":
            if frames is None:
                raise RecordingDistributionScopeError(
                    f"Scope {scope.scope_id!r} requires acquisition frames."
                )
            mask = (frames >= int(scope.start_inclusive)) & (
                frames < int(scope.end_exclusive)
            )
            missing = np.zeros(row_count, dtype=bool)
        else:
            if timestamps is None:
                raise RecordingDistributionScopeError(
                    f"Scope {scope.scope_id!r} requires session timestamps."
                )
            in_bounds = (timestamps >= int(scope.start_inclusive)) & (
                timestamps < int(scope.end_exclusive)
            )
            mask = time_valid & in_bounds
            missing = ~time_valid
        masks[scope.scope_id] = mask
        uncovered[scope.scope_id] = missing
    return ScopeMaskProjection(
        masks=MappingProxyType(masks),
        uncovered=MappingProxyType(uncovered),
        membership_policy_id="sample_coordinate_half_open_no_interpolation_v1",
    )


def transition_scope_masks(
    scopes: Sequence[RecordingDistributionScope],
    *,
    acquisition_frame_id: Any | None = None,
    acquisition_frame_delta: Any | None = None,
    timestamp_ns_session: Any | None = None,
    timestamp_delta_ns: Any | None = None,
    timestamp_valid: Any | None = None,
) -> ScopeMaskProjection:
    """Require both exact transition endpoints inside every bounded scope."""

    current = sample_scope_masks(
        scopes,
        acquisition_frame_id=acquisition_frame_id,
        timestamp_ns_session=timestamp_ns_session,
        timestamp_valid=timestamp_valid,
    )
    previous_frames = None
    if acquisition_frame_id is not None:
        frames = _aligned_int(acquisition_frame_id, field="acquisition_frame_id")
        deltas = _aligned_int(
            acquisition_frame_delta, field="acquisition_frame_delta"
        )
        if frames.shape != deltas.shape:
            raise RecordingDistributionScopeError(
                "Frame transition deltas do not align."
            )
        previous_frames = frames - deltas
    previous_timestamps = None
    if timestamp_ns_session is not None:
        timestamps = _aligned_int(
            timestamp_ns_session, field="timestamp_ns_session"
        )
        deltas_ns = _aligned_int(timestamp_delta_ns, field="timestamp_delta_ns")
        if timestamps.shape != deltas_ns.shape:
            raise RecordingDistributionScopeError(
                "Timestamp transition deltas do not align."
            )
        previous_timestamps = timestamps - deltas_ns
    previous = sample_scope_masks(
        scopes,
        acquisition_frame_id=previous_frames,
        timestamp_ns_session=previous_timestamps,
        timestamp_valid=timestamp_valid,
    )
    masks = {
        scope.scope_id: (
            current.masks[scope.scope_id]
            if scope.axis_kind == "all"
            else current.masks[scope.scope_id] & previous.masks[scope.scope_id]
        )
        for scope in validate_scope_registry(scopes)
    }
    uncovered = {
        scope.scope_id: (
            current.uncovered[scope.scope_id]
            | previous.uncovered[scope.scope_id]
        )
        for scope in validate_scope_registry(scopes)
    }
    return ScopeMaskProjection(
        masks=MappingProxyType(masks),
        uncovered=MappingProxyType(uncovered),
        membership_policy_id="both_transition_endpoints_inside_scope_v1",
    )


def exact_source_membership_masks(
    scopes: Sequence[RecordingDistributionScope],
    *,
    source_scope_id: Sequence[object],
) -> ScopeMaskProjection:
    """Project producer-authored event membership; unassigned events stay whole-only."""

    ordered = validate_scope_registry(scopes)
    values = np.asarray(source_scope_id, dtype=object).reshape(-1)
    allowed = {scope.scope_id for scope in ordered[1:]}
    observed = {str(value) for value in values if value is not None}
    unknown = sorted(observed - allowed)
    if unknown:
        raise RecordingDistributionScopeError(
            f"Source-authored membership names unknown scopes: {unknown!r}."
        )
    masks = {WHOLE_SESSION_SCOPE_ID: np.ones(values.shape, dtype=bool)}
    masks.update(
        {
            scope.scope_id: np.asarray(
                [value == scope.scope_id for value in values], dtype=bool
            )
            for scope in ordered[1:]
        }
    )
    uncovered = {
        scope.scope_id: np.zeros(values.shape, dtype=bool) for scope in ordered
    }
    return ScopeMaskProjection(
        masks=MappingProxyType(masks),
        uncovered=MappingProxyType(uncovered),
        membership_policy_id="producer_authored_exact_scope_membership_v1",
    )


def fully_contained_frame_event_masks(
    scopes: Sequence[RecordingDistributionScope],
    *,
    start_acquisition_frame_id: Any,
    end_acquisition_frame_id: Any,
) -> ScopeMaskProjection:
    """Assign events only when both inclusive event endpoints are in a scope."""

    starts = _aligned_int(
        start_acquisition_frame_id, field="start_acquisition_frame_id"
    )
    ends = _aligned_int(end_acquisition_frame_id, field="end_acquisition_frame_id")
    if starts.shape != ends.shape or np.any(ends < starts):
        raise RecordingDistributionScopeError("Event frame bounds are invalid.")
    ordered = validate_scope_registry(scopes)
    masks: dict[str, np.ndarray] = {}
    for scope in ordered:
        if scope.axis_kind == "all":
            masks[scope.scope_id] = np.ones(starts.shape, dtype=bool)
        elif scope.axis_kind == "acquisition_frame":
            masks[scope.scope_id] = (starts >= int(scope.start_inclusive)) & (
                ends < int(scope.end_exclusive)
            )
        else:
            raise RecordingDistributionScopeError(
                "Frame-bounded events cannot be projected into a time-only scope."
            )
    uncovered = {
        scope.scope_id: np.zeros(starts.shape, dtype=bool) for scope in ordered
    }
    return ScopeMaskProjection(
        masks=MappingProxyType(masks),
        uncovered=MappingProxyType(uncovered),
        membership_policy_id="event_fully_contained_in_half_open_frame_scope_v1",
    )


def fully_contained_time_event_masks(
    scopes: Sequence[RecordingDistributionScope],
    *,
    start_timestamp_ns_session: Any,
    end_timestamp_ns_session: Any,
    timestamp_valid: Any | None = None,
) -> ScopeMaskProjection:
    """Assign events only when both exact timestamps lie in a time scope."""

    starts = _aligned_int(
        start_timestamp_ns_session, field="start_timestamp_ns_session"
    )
    ends = _aligned_int(
        end_timestamp_ns_session, field="end_timestamp_ns_session"
    )
    if starts.shape != ends.shape or np.any(ends < starts):
        raise RecordingDistributionScopeError("Event timestamp bounds are invalid.")
    valid = (
        np.ones(starts.shape, dtype=bool)
        if timestamp_valid is None
        else np.asarray(timestamp_valid, dtype=bool)
    )
    if valid.shape != starts.shape:
        raise RecordingDistributionScopeError(
            "Event timestamp validity does not align."
        )
    ordered = validate_scope_registry(scopes)
    masks: dict[str, np.ndarray] = {}
    uncovered: dict[str, np.ndarray] = {}
    for scope in ordered:
        if scope.axis_kind == "all":
            masks[scope.scope_id] = np.ones(starts.shape, dtype=bool)
            uncovered[scope.scope_id] = np.zeros(starts.shape, dtype=bool)
        elif scope.axis_kind == "session_time_ns":
            masks[scope.scope_id] = (
                valid
                & (starts >= int(scope.start_inclusive))
                & (ends < int(scope.end_exclusive))
            )
            uncovered[scope.scope_id] = ~valid
        else:
            raise RecordingDistributionScopeError(
                "Time-bounded events cannot be projected into a frame-only scope."
            )
    return ScopeMaskProjection(
        masks=MappingProxyType(masks),
        uncovered=MappingProxyType(uncovered),
        membership_policy_id="event_fully_contained_in_half_open_time_scope_v1",
    )


__all__ = [
    "RecordingDistributionScope",
    "RecordingDistributionScopeError",
    "SCOPE_REGISTRY_SCHEMA_ID",
    "SCOPE_REGISTRY_SCHEMA_VERSION",
    "SCOPE_SCHEMA_ID",
    "SCOPE_SCHEMA_VERSION",
    "ScopeMaskProjection",
    "WHOLE_SESSION_PROVIDER_ID",
    "WHOLE_SESSION_SCOPE_ID",
    "exact_source_membership_masks",
    "frame_interval_scope",
    "fully_contained_frame_event_masks",
    "fully_contained_time_event_masks",
    "sample_scope_masks",
    "scope_registry_record",
    "session_time_bracket_scope",
    "transition_scope_masks",
    "validate_scope_registry",
    "whole_session_scope",
]
