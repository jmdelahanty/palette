"""Pure, authority-bound composable stimulus selection for analytics.

This module deliberately has no Zarr, registry, filesystem, or clock-reader
dependencies.  It is the in-memory v1 compiler used to freeze a requested
selection and its resolved acquisition-frame membership before a metric is
computed.

The contract has two related digests:

``request_digest``
    Digest of the canonical expression and its authority binding.  This is
    independent of the resolved intervals and detects a changed request.

``resolved_digest``
    Digest of the concrete, ordered, de-duplicated frame intervals and their
    source memberships.  This is the lineage handle a downstream metric can
    bind to.

Distinct atomic references may overlap.  The compiler de-duplicates frames
while retaining every source membership.  Repeating the same atomic
reference in one expression is rejected because it is normally an accidental
duplicate and cannot be distinguished from an intended second occurrence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
from types import MappingProxyType
from typing import Literal, Mapping, TypeAlias, Union

SCHEMA_VERSION = 1
ATOMIC_INTERVAL_REFERENCE_SCHEMA_ID = (
    "palette.composable_stimulus_atomic_interval_reference.v1"
)
SELECTION_EXPRESSION_SCHEMA_ID = "palette.composable_stimulus_selection_expression.v1"
RESOLVED_FRAME_SET_SCHEMA_ID = "palette.composable_stimulus_resolved_frame_set.v1"
SELECTION_REQUEST_SCHEMA_ID = "palette.composable_stimulus_selection_request.v1"
INTERVAL_POLICY_ID = "half_open_acquisition_frame_v1"
TRIM_ROUNDING_POLICY_ID = "ceil_seconds_times_fps_v1"

ReferenceKind: TypeAlias = Literal["stimulus_step", "interval_annotation"]
AggregationPolicy: TypeAlias = Literal["keep_occurrences", "pool_intervals"]


class StimulusSelectionError(ValueError):
    """Base error for invalid or unsupported selection contracts."""


class AuthorityMismatchError(StimulusSelectionError):
    """Raised when references do not bind to one exact timeline authority."""


class UnsupportedExpressionError(StimulusSelectionError):
    """Raised when an expression uses an operation outside the v1 vocabulary."""


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StimulusSelectionError(f"{field_name} must be a non-empty string")
    return value


def _require_digest(value: object, field_name: str) -> str:
    text = _require_text(value, field_name).lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise StimulusSelectionError(f"{field_name} must be a lowercase SHA-256 digest")
    return text


def _require_frame(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise StimulusSelectionError(f"{field_name} must be an integer frame index")
    return value


def _require_finite_nonnegative(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StimulusSelectionError(
            f"{field_name} must be a finite non-negative number"
        )
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise StimulusSelectionError(
            f"{field_name} must be a finite non-negative number"
        )
    return number


def _canonicalize(value: object) -> object:
    """Convert supported contract values to deterministic JSON values."""

    if hasattr(value, "to_dict"):
        return _canonicalize(value.to_dict())  # type: ignore[union-attr]
    if isinstance(value, Mapping):
        items = sorted(
            ((str(key), item) for key, item in value.items()), key=lambda pair: pair[0]
        )
        return {key: _canonicalize(item) for key, item in items}
    if isinstance(value, (tuple, list)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise StimulusSelectionError(
                "canonical JSON cannot contain NaN or infinity"
            )
        return 0.0 if value == 0.0 else value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise StimulusSelectionError(
        f"unsupported value {type(value).__name__} in canonical contract payload"
    )


def canonical_json(value: object) -> str:
    """Return canonical UTF-8 JSON text for a contract value."""

    return json.dumps(
        _canonicalize(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_sha256(value: object) -> str:
    """Return the SHA-256 digest of canonical JSON for a contract value."""

    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _freeze_json(value: object) -> object:
    """Recursively freeze one already validated strict-JSON value."""

    canonical = _canonicalize(value)
    if isinstance(canonical, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in canonical.items()}
        )
    if isinstance(canonical, list):
        return tuple(_freeze_json(item) for item in canonical)
    return canonical


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True)
class TimelineAuthority:
    """The complete authority tuple shared by every atomic reference."""

    recording_id: str
    timeline_id: str
    stimulus_authority_id: str
    stimulus_authority_sha256: str
    acquisition_frame_domain: str
    acquisition_frame_count: int
    source_video_metadata_ref: str
    source_video_metadata_sha256: str
    acquisition_clock_authority_ref: str
    acquisition_clock_authority_sha256: str
    source_metadata_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "recording_id",
            "timeline_id",
            "stimulus_authority_id",
            "acquisition_frame_domain",
            "source_video_metadata_ref",
            "acquisition_clock_authority_ref",
        ):
            _require_text(getattr(self, name), name)
        if (
            isinstance(self.acquisition_frame_count, bool)
            or not isinstance(self.acquisition_frame_count, int)
            or self.acquisition_frame_count <= 0
        ):
            raise StimulusSelectionError(
                "acquisition_frame_count must be a positive integer"
            )
        for name in (
            "stimulus_authority_sha256",
            "source_video_metadata_sha256",
            "acquisition_clock_authority_sha256",
            "source_metadata_sha256",
        ):
            _require_digest(getattr(self, name), name)

    def to_dict(self) -> dict[str, object]:
        return {
            "recording_id": self.recording_id,
            "timeline_id": self.timeline_id,
            "stimulus_authority_id": self.stimulus_authority_id,
            "stimulus_authority_sha256": self.stimulus_authority_sha256,
            "acquisition_frame_domain": self.acquisition_frame_domain,
            "acquisition_frame_count": self.acquisition_frame_count,
            "source_video_metadata_ref": self.source_video_metadata_ref,
            "source_video_metadata_sha256": self.source_video_metadata_sha256,
            "acquisition_clock_authority_ref": self.acquisition_clock_authority_ref,
            "acquisition_clock_authority_sha256": self.acquisition_clock_authority_sha256,
            "source_metadata_sha256": self.source_metadata_sha256,
        }


@dataclass(frozen=True)
class RoleMetadata:
    """Explicit saved semantic role; never inferred from a step label."""

    role: str
    label: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.role, "role")
        if self.label is not None:
            _require_text(self.label, "role label")
        if not isinstance(self.metadata, Mapping):
            raise StimulusSelectionError("role metadata must be a mapping")
        frozen = _freeze_json(self.metadata)
        if not isinstance(frozen, Mapping):  # pragma: no cover - defensive
            raise StimulusSelectionError("role metadata must remain a mapping")
        object.__setattr__(self, "metadata", frozen)

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "label": self.label,
            "metadata": _thaw_json(self.metadata),
        }


@dataclass(frozen=True)
class TrimSpec:
    """Directional trim with an explicit nominal frame-clock policy."""

    leading_seconds: float = 0.0
    trailing_seconds: float = 0.0
    fps: float = 0.0
    rounding_policy: str = TRIM_ROUNDING_POLICY_ID

    def __post_init__(self) -> None:
        leading = _require_finite_nonnegative(self.leading_seconds, "leading_seconds")
        trailing = _require_finite_nonnegative(
            self.trailing_seconds, "trailing_seconds"
        )
        if isinstance(self.fps, bool) or not isinstance(self.fps, (int, float)):
            raise StimulusSelectionError("fps must be a finite positive number")
        fps = float(self.fps)
        if not math.isfinite(fps) or fps <= 0:
            raise StimulusSelectionError("fps must be a finite positive number")
        if self.rounding_policy != TRIM_ROUNDING_POLICY_ID:
            raise StimulusSelectionError(
                f"unsupported trim rounding policy {self.rounding_policy!r}"
            )
        object.__setattr__(self, "leading_seconds", leading)
        object.__setattr__(self, "trailing_seconds", trailing)
        object.__setattr__(self, "fps", fps)

    @property
    def leading_frames(self) -> int:
        return math.ceil(self.leading_seconds * self.fps)

    @property
    def trailing_frames(self) -> int:
        return math.ceil(self.trailing_seconds * self.fps)

    def to_dict(self) -> dict[str, object]:
        return {
            "leading_seconds": self.leading_seconds,
            "trailing_seconds": self.trailing_seconds,
            "fps": self.fps,
            "rounding_policy": self.rounding_policy,
            "leading_frame_count": self.leading_frames,
            "trailing_frame_count": self.trailing_frames,
        }


@dataclass(frozen=True)
class AtomicIntervalReference:
    """One exact half-open step or annotation interval."""

    reference_kind: ReferenceKind
    reference_id: str
    label: str
    start_frame: int
    end_frame: int
    authority: TimelineAuthority
    occurrence_id: str | None = None

    def __post_init__(self) -> None:
        if self.reference_kind not in ("stimulus_step", "interval_annotation"):
            raise StimulusSelectionError(
                f"unsupported atomic reference kind {self.reference_kind!r}"
            )
        _require_text(self.reference_id, "reference_id")
        _require_text(self.label, "label")
        start = _require_frame(self.start_frame, "start_frame")
        end = _require_frame(self.end_frame, "end_frame")
        if start < 0 or end <= start:
            raise StimulusSelectionError(
                "atomic intervals must be non-empty half-open intervals with start < end"
            )
        if end > self.authority.acquisition_frame_count:
            raise StimulusSelectionError(
                "atomic interval exceeds the bound acquisition frame domain"
            )
        if self.occurrence_id is not None:
            _require_text(self.occurrence_id, "occurrence_id")

    @property
    def effective_occurrence_id(self) -> str:
        return self.occurrence_id or f"{self.reference_kind}:{self.reference_id}"

    @property
    def identity_key(self) -> tuple[str, str, str]:
        return (self.reference_kind, self.reference_id, self.effective_occurrence_id)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_id": ATOMIC_INTERVAL_REFERENCE_SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "reference_kind": self.reference_kind,
            "reference_id": self.reference_id,
            "label": self.label,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "interval_policy_id": INTERVAL_POLICY_ID,
            "authority": self.authority.to_dict(),
            "occurrence_id": self.occurrence_id,
        }


def stimulus_step_reference(
    *,
    reference_id: str,
    label: str,
    start_frame: int,
    end_frame: int,
    authority: TimelineAuthority,
    occurrence_id: str | None = None,
) -> AtomicIntervalReference:
    return AtomicIntervalReference(
        reference_kind="stimulus_step",
        reference_id=reference_id,
        label=label,
        start_frame=start_frame,
        end_frame=end_frame,
        authority=authority,
        occurrence_id=occurrence_id,
    )


def interval_annotation_reference(
    *,
    reference_id: str,
    label: str,
    start_frame: int,
    end_frame: int,
    authority: TimelineAuthority,
    occurrence_id: str | None = None,
) -> AtomicIntervalReference:
    return AtomicIntervalReference(
        reference_kind="interval_annotation",
        reference_id=reference_id,
        label=label,
        start_frame=start_frame,
        end_frame=end_frame,
        authority=authority,
        occurrence_id=occurrence_id,
    )


@dataclass(frozen=True)
class MemberExpression:
    reference: AtomicIntervalReference
    role: RoleMetadata | None = None
    trim: TrimSpec | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_id": SELECTION_EXPRESSION_SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "operation": "member",
            "reference": self.reference.to_dict(),
            "role": self.role.to_dict() if self.role is not None else None,
            "trim": self.trim.to_dict() if self.trim is not None else None,
        }


SelectionExpression: TypeAlias = Union[
    MemberExpression,
    "UnionExpression",
    "IntersectionExpression",
    "DifferenceExpression",
]


def _canonical_expression_key(expression: object) -> str:
    if not hasattr(expression, "to_dict"):
        raise UnsupportedExpressionError(
            f"unsupported expression type {type(expression).__name__}"
        )
    return canonical_json(expression.to_dict())  # type: ignore[union-attr]


@dataclass(frozen=True)
class UnionExpression:
    children: tuple[SelectionExpression, ...]

    def __post_init__(self) -> None:
        children = tuple(self.children)
        if not children:
            raise StimulusSelectionError("union requires at least one child")
        object.__setattr__(
            self, "children", tuple(sorted(children, key=_canonical_expression_key))
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_id": SELECTION_EXPRESSION_SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "operation": "union",
            "children": [child.to_dict() for child in self.children],
        }


@dataclass(frozen=True)
class IntersectionExpression:
    children: tuple[SelectionExpression, ...]

    def __post_init__(self) -> None:
        children = tuple(self.children)
        if not children:
            raise StimulusSelectionError("intersection requires at least one child")
        object.__setattr__(
            self, "children", tuple(sorted(children, key=_canonical_expression_key))
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_id": SELECTION_EXPRESSION_SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "operation": "intersection",
            "children": [child.to_dict() for child in self.children],
        }


@dataclass(frozen=True)
class DifferenceExpression:
    left: SelectionExpression
    right: SelectionExpression

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_id": SELECTION_EXPRESSION_SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "operation": "difference",
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }


def member(
    reference: AtomicIntervalReference,
    *,
    role: RoleMetadata | None = None,
    trim: TrimSpec | None = None,
) -> MemberExpression:
    if not isinstance(reference, AtomicIntervalReference):
        raise StimulusSelectionError("member requires an AtomicIntervalReference")
    if role is not None and not isinstance(role, RoleMetadata):
        raise StimulusSelectionError("role must be RoleMetadata or None")
    if trim is not None and not isinstance(trim, TrimSpec):
        raise StimulusSelectionError("trim must be TrimSpec or None")
    return MemberExpression(reference=reference, role=role, trim=trim)


def union(*children: SelectionExpression) -> UnionExpression:
    return UnionExpression(tuple(children))


def intersection(*children: SelectionExpression) -> IntersectionExpression:
    return IntersectionExpression(tuple(children))


def difference(
    left: SelectionExpression, right: SelectionExpression
) -> DifferenceExpression:
    return DifferenceExpression(left=left, right=right)


@dataclass(frozen=True)
class SelectionSpec:
    """A requested expression and its explicit aggregation policy."""

    selection_id: str
    expression: SelectionExpression
    aggregation_policy: AggregationPolicy
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.selection_id, "selection_id")
        if self.aggregation_policy not in ("keep_occurrences", "pool_intervals"):
            raise StimulusSelectionError(
                f"unsupported aggregation policy {self.aggregation_policy!r}"
            )
        if not isinstance(self.metadata, Mapping):
            raise StimulusSelectionError("selection metadata must be a mapping")
        frozen = _freeze_json(self.metadata)
        if not isinstance(frozen, Mapping):  # pragma: no cover - defensive
            raise StimulusSelectionError("selection metadata must remain a mapping")
        object.__setattr__(self, "metadata", frozen)
        _canonical_expression_key(self.expression)

    def to_dict(self, *, authority: TimelineAuthority) -> dict[str, object]:
        return {
            "schema_id": SELECTION_REQUEST_SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "selection_id": self.selection_id,
            "aggregation_policy": self.aggregation_policy,
            "interval_policy_id": INTERVAL_POLICY_ID,
            "authority": authority.to_dict(),
            "expression": self.expression.to_dict(),
            "metadata": _thaw_json(self.metadata),
        }


@dataclass(frozen=True)
class SourceMembership:
    reference_kind: ReferenceKind
    reference_id: str
    occurrence_id: str
    label: str
    original_start_frame: int
    original_end_frame: int
    selected_start_frame: int
    selected_end_frame: int
    role: RoleMetadata | None
    trim: TrimSpec | None

    @property
    def identity_key(self) -> tuple[str, str, str]:
        return (self.reference_kind, self.reference_id, self.occurrence_id)

    def to_dict(self) -> dict[str, object]:
        return {
            "reference_kind": self.reference_kind,
            "reference_id": self.reference_id,
            "occurrence_id": self.occurrence_id,
            "label": self.label,
            "original_interval": [self.original_start_frame, self.original_end_frame],
            "selected_interval": [self.selected_start_frame, self.selected_end_frame],
            "role": self.role.to_dict() if self.role is not None else None,
            "trim": self.trim.to_dict() if self.trim is not None else None,
        }


@dataclass(frozen=True)
class ResolvedInterval:
    start_frame: int
    end_frame: int
    source_memberships: tuple[SourceMembership, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "source_memberships": [
                membership.to_dict() for membership in self.source_memberships
            ],
        }


@dataclass(frozen=True)
class ResolvedOccurrence:
    occurrence_id: str
    reference_kind: ReferenceKind
    reference_id: str
    label: str
    role: RoleMetadata | None
    intervals: tuple[tuple[int, int], ...]

    @property
    def frame_count(self) -> int:
        return sum(end - start for start, end in self.intervals)

    def to_dict(self) -> dict[str, object]:
        return {
            "occurrence_id": self.occurrence_id,
            "reference_kind": self.reference_kind,
            "reference_id": self.reference_id,
            "label": self.label,
            "role": self.role.to_dict() if self.role is not None else None,
            "intervals": [list(interval) for interval in self.intervals],
            "frame_count": self.frame_count,
        }


@dataclass(frozen=True)
class CompiledSelection:
    selection_id: str
    aggregation_policy: AggregationPolicy
    authority: TimelineAuthority
    requested: Mapping[str, object]
    request_digest: str
    resolved_intervals: tuple[ResolvedInterval, ...]
    pooled_intervals: tuple[tuple[int, int], ...]
    occurrences: tuple[ResolvedOccurrence, ...]
    resolved_digest: str

    def __post_init__(self) -> None:
        frozen = _freeze_json(self.requested)
        if not isinstance(frozen, Mapping):  # pragma: no cover - defensive
            raise StimulusSelectionError("requested selection must remain a mapping")
        object.__setattr__(self, "requested", frozen)

    @property
    def selected_frame_count(self) -> int:
        return sum(
            interval.end_frame - interval.start_frame
            for interval in self.resolved_intervals
        )

    @property
    def empty(self) -> bool:
        return not self.resolved_intervals

    def resolved_payload(self) -> dict[str, object]:
        return {
            "schema_id": RESOLVED_FRAME_SET_SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "selection_id": self.selection_id,
            "aggregation_policy": self.aggregation_policy,
            "interval_policy_id": INTERVAL_POLICY_ID,
            "authority": self.authority.to_dict(),
            "request_digest": self.request_digest,
            "resolved_intervals": [
                interval.to_dict() for interval in self.resolved_intervals
            ],
            "pooled_intervals": [list(interval) for interval in self.pooled_intervals],
            "occurrences": [occurrence.to_dict() for occurrence in self.occurrences],
        }

    def to_dict(self) -> dict[str, object]:
        payload = self.resolved_payload()
        payload["requested"] = _thaw_json(self.requested)
        payload["request_digest"] = self.request_digest
        payload["resolved_digest"] = self.resolved_digest
        return payload


def _trimmed_interval(
    reference: AtomicIntervalReference, trim: TrimSpec | None
) -> tuple[int, int]:
    if trim is None:
        return reference.start_frame, reference.end_frame
    start = reference.start_frame + trim.leading_frames
    end = reference.end_frame - trim.trailing_frames
    if start > end:
        raise StimulusSelectionError(
            f"trims invert interval {reference.reference_kind}:{reference.reference_id}"
        )
    return start, end


def _membership_for(
    expression: MemberExpression,
) -> tuple[dict[int, dict[tuple[str, str, str], SourceMembership]], SourceMembership]:
    reference = expression.reference
    start, end = _trimmed_interval(reference, expression.trim)
    membership = SourceMembership(
        reference_kind=reference.reference_kind,
        reference_id=reference.reference_id,
        occurrence_id=reference.effective_occurrence_id,
        label=reference.label,
        original_start_frame=reference.start_frame,
        original_end_frame=reference.end_frame,
        selected_start_frame=start,
        selected_end_frame=end,
        role=expression.role,
        trim=expression.trim,
    )
    frames = {
        frame: {membership.identity_key: membership} for frame in range(start, end)
    }
    return frames, membership


def _merge_memberships(
    left: dict[tuple[str, str, str], SourceMembership],
    right: dict[tuple[str, str, str], SourceMembership],
) -> dict[tuple[str, str, str], SourceMembership]:
    merged = dict(left)
    for key, value in right.items():
        existing = merged.get(key)
        if existing is not None and existing != value:
            raise StimulusSelectionError(f"conflicting source membership for {key!r}")
        merged[key] = value
    return merged


def _evaluate(
    expression: SelectionExpression,
) -> tuple[
    dict[int, dict[tuple[str, str, str], SourceMembership]],
    tuple[SourceMembership, ...],
]:
    if isinstance(expression, MemberExpression):
        frames, membership = _membership_for(expression)
        return frames, (membership,)
    if isinstance(expression, UnionExpression):
        result: dict[int, dict[tuple[str, str, str], SourceMembership]] = {}
        members: list[SourceMembership] = []
        for child in expression.children:
            child_frames, child_members = _evaluate(child)
            members.extend(child_members)
            for frame, child_memberships in child_frames.items():
                result[frame] = _merge_memberships(
                    result.get(frame, {}), child_memberships
                )
        return result, tuple(members)
    if isinstance(expression, IntersectionExpression):
        child_evaluations = [_evaluate(child) for child in expression.children]
        result = dict(child_evaluations[0][0])
        members: list[SourceMembership] = list(child_evaluations[0][1])
        for child_frames, child_members in child_evaluations[1:]:
            result = {
                frame: _merge_memberships(result[frame], child_frames[frame])
                for frame in result.keys() & child_frames.keys()
            }
            members.extend(child_members)
        return result, tuple(members)
    if isinstance(expression, DifferenceExpression):
        left_frames, left_members = _evaluate(expression.left)
        right_frames, _right_members = _evaluate(expression.right)
        return (
            {
                frame: memberships
                for frame, memberships in left_frames.items()
                if frame not in right_frames
            },
            left_members,
        )
    raise UnsupportedExpressionError(
        f"unsupported expression type {type(expression).__name__}"
    )


def _collect_members(expression: SelectionExpression) -> tuple[MemberExpression, ...]:
    if isinstance(expression, MemberExpression):
        return (expression,)
    if isinstance(expression, (UnionExpression, IntersectionExpression)):
        members: list[MemberExpression] = []
        for child in expression.children:
            members.extend(_collect_members(child))
        return tuple(members)
    if isinstance(expression, DifferenceExpression):
        return _collect_members(expression.left) + _collect_members(expression.right)
    raise UnsupportedExpressionError(
        f"unsupported expression type {type(expression).__name__}"
    )


def _validate_members(
    expression: SelectionExpression,
    expected_authority: TimelineAuthority | None,
) -> tuple[TimelineAuthority, tuple[MemberExpression, ...]]:
    members = _collect_members(expression)
    if not members:
        raise StimulusSelectionError("selection expression has no atomic members")
    authority = members[0].reference.authority
    if expected_authority is not None and authority != expected_authority:
        raise AuthorityMismatchError(
            "selection authority does not match expected authority"
        )
    seen: set[tuple[str, str, str]] = set()
    for item in members:
        item_authority = item.reference.authority
        if item_authority != authority:
            raise AuthorityMismatchError(
                "all atomic references must share one recording, timeline, and authority"
            )
        if item.reference.identity_key in seen:
            raise StimulusSelectionError(
                "duplicate atomic reference/occurrence in selection expression: "
                f"{item.reference.identity_key!r}"
            )
        seen.add(item.reference.identity_key)
    return authority, members


def _coalesce_frames(
    frames: set[int] | list[int] | tuple[int, ...],
) -> tuple[tuple[int, int], ...]:
    ordered = sorted(set(frames))
    if not ordered:
        return ()
    intervals: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for frame in ordered[1:]:
        if frame != previous + 1:
            intervals.append((start, previous + 1))
            start = frame
        previous = frame
    intervals.append((start, previous + 1))
    return tuple(intervals)


def _membership_intervals(
    frame_memberships: Mapping[int, Mapping[tuple[str, str, str], SourceMembership]],
) -> tuple[ResolvedInterval, ...]:
    if not frame_memberships:
        return ()
    ordered = sorted(frame_memberships)
    intervals: list[ResolvedInterval] = []
    start = previous = ordered[0]
    previous_memberships = tuple(
        sorted(frame_memberships[start].values(), key=lambda value: value.identity_key)
    )
    for frame in ordered[1:]:
        memberships = tuple(
            sorted(
                frame_memberships[frame].values(), key=lambda value: value.identity_key
            )
        )
        if frame != previous + 1 or memberships != previous_memberships:
            intervals.append(
                ResolvedInterval(
                    start_frame=start,
                    end_frame=previous + 1,
                    source_memberships=previous_memberships,
                )
            )
            start = frame
            previous_memberships = memberships
        previous = frame
    intervals.append(
        ResolvedInterval(
            start_frame=start,
            end_frame=previous + 1,
            source_memberships=previous_memberships,
        )
    )
    return tuple(intervals)


def _resolved_occurrences(
    members: tuple[MemberExpression, ...],
    frame_memberships: Mapping[int, Mapping[tuple[str, str, str], SourceMembership]],
) -> tuple[ResolvedOccurrence, ...]:
    occurrences: list[ResolvedOccurrence] = []
    for member_expression in members:
        source = member_expression.reference
        key = source.identity_key
        frames = {
            frame
            for frame, memberships in frame_memberships.items()
            if key in memberships
        }
        occurrences.append(
            ResolvedOccurrence(
                occurrence_id=source.effective_occurrence_id,
                reference_kind=source.reference_kind,
                reference_id=source.reference_id,
                label=source.label,
                role=member_expression.role,
                intervals=_coalesce_frames(frames),
            )
        )
    return tuple(occurrences)


def compile_selection(
    spec: SelectionSpec,
    *,
    expected_authority: TimelineAuthority | None = None,
) -> CompiledSelection:
    """Validate and deterministically compile a v1 selection expression."""

    if not isinstance(spec, SelectionSpec):
        raise StimulusSelectionError("compile_selection requires SelectionSpec")
    authority, members = _validate_members(spec.expression, expected_authority)
    requested = spec.to_dict(authority=authority)
    request_digest = canonical_sha256(requested)
    frame_memberships, _evaluated_members = _evaluate(spec.expression)
    resolved_intervals = _membership_intervals(frame_memberships)
    pooled_intervals = _coalesce_frames(set(frame_memberships))
    occurrences = _resolved_occurrences(members, frame_memberships)
    resolved_payload = {
        "schema_id": RESOLVED_FRAME_SET_SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "selection_id": spec.selection_id,
        "aggregation_policy": spec.aggregation_policy,
        "interval_policy_id": INTERVAL_POLICY_ID,
        "authority": authority.to_dict(),
        "request_digest": request_digest,
        "resolved_intervals": [interval.to_dict() for interval in resolved_intervals],
        "pooled_intervals": [list(interval) for interval in pooled_intervals],
        "occurrences": [occurrence.to_dict() for occurrence in occurrences],
    }
    resolved_digest = canonical_sha256(resolved_payload)
    return CompiledSelection(
        selection_id=spec.selection_id,
        aggregation_policy=spec.aggregation_policy,
        authority=authority,
        requested=requested,
        request_digest=request_digest,
        resolved_intervals=resolved_intervals,
        pooled_intervals=pooled_intervals,
        occurrences=occurrences,
        resolved_digest=resolved_digest,
    )


__all__ = [
    "ATOMIC_INTERVAL_REFERENCE_SCHEMA_ID",
    "AggregationPolicy",
    "AtomicIntervalReference",
    "AuthorityMismatchError",
    "CompiledSelection",
    "DifferenceExpression",
    "INTERVAL_POLICY_ID",
    "IntersectionExpression",
    "MemberExpression",
    "RESOLVED_FRAME_SET_SCHEMA_ID",
    "RoleMetadata",
    "SELECTION_EXPRESSION_SCHEMA_ID",
    "SELECTION_REQUEST_SCHEMA_ID",
    "SCHEMA_VERSION",
    "SelectionSpec",
    "SourceMembership",
    "StimulusSelectionError",
    "TimelineAuthority",
    "TrimSpec",
    "TRIM_ROUNDING_POLICY_ID",
    "UnionExpression",
    "UnsupportedExpressionError",
    "canonical_json",
    "canonical_sha256",
    "compile_selection",
    "difference",
    "intersection",
    "interval_annotation_reference",
    "member",
    "stimulus_step_reference",
    "union",
]
