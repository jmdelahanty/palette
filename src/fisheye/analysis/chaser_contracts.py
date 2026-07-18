"""Canonical identity and long-form role contracts for chaser analyses."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from fisheye.analysis.chaser_behavior import (
    BEHAVIOR_CLASS_LABELS,
    canonical_behavior_label,
    resolve_configured_chaser_behaviors,
)

CHASER_SOURCE_SCHEMA_ID = "palette.chaser.source_contract"
CHASER_SOURCE_SCHEMA_VERSION = 1
ROLE_INTERVAL_BOUNDARY_POLICY = "inclusive_start_inclusive_end"


@dataclass(frozen=True)
class ChaserIdentity:
    """Stable stimulus identity connected to one persisted chaser track."""

    stimulus_instance_id: str
    chaser_index: int
    source_track_key: str
    raw_color_rgba: tuple[float, float, float, float]


@dataclass(frozen=True)
class ChaserRoleInterval:
    """One long-form role assignment for a chaser over a frame interval."""

    stimulus_instance_id: str
    role_class_id: int
    role: str
    start_frame: int
    end_frame: int | None
    source: str


@dataclass(frozen=True)
class CanonicalChaserSet:
    """Validated, variable-length chaser identities and role intervals."""

    identities: tuple[ChaserIdentity, ...]
    role_intervals: tuple[ChaserRoleInterval, ...]

    def __post_init__(self) -> None:
        validate_chaser_set(self.identities, self.role_intervals)

    def identity_rows(self) -> tuple[dict[str, Any], ...]:
        return tuple(asdict(identity) for identity in self.identities)

    def role_rows(self) -> tuple[dict[str, Any], ...]:
        return tuple(asdict(interval) for interval in self.role_intervals)


def _validate_frame_interval(interval: ChaserRoleInterval) -> None:
    if int(interval.start_frame) < 0:
        raise ValueError("chaser role start_frame must be nonnegative")
    if interval.end_frame is not None and int(interval.end_frame) < int(
        interval.start_frame
    ):
        raise ValueError(
            "chaser role end_frame must be greater than or equal to start_frame"
        )


def validate_chaser_set(
    identities: Sequence[ChaserIdentity],
    role_intervals: Sequence[ChaserRoleInterval],
) -> None:
    """Validate identity uniqueness and non-overlapping role assignments."""

    if not identities:
        raise ValueError("a canonical chaser set must contain at least one identity")
    instance_ids = [
        str(identity.stimulus_instance_id).strip() for identity in identities
    ]
    if any(not value for value in instance_ids):
        raise ValueError("stimulus_instance_id cannot be empty")
    if len(set(instance_ids)) != len(instance_ids):
        raise ValueError("stimulus_instance_id values must be unique")
    chaser_indices = [int(identity.chaser_index) for identity in identities]
    if len(set(chaser_indices)) != len(chaser_indices):
        raise ValueError("chaser_index values must be unique")

    known = set(instance_ids)
    by_instance: dict[str, list[ChaserRoleInterval]] = {
        value: [] for value in instance_ids
    }
    for interval in role_intervals:
        instance_id = str(interval.stimulus_instance_id).strip()
        if instance_id not in known:
            raise ValueError(
                f"role interval references unknown stimulus instance: {instance_id!r}"
            )
        _validate_frame_interval(interval)
        role = canonical_behavior_label(interval.role)
        expected = BEHAVIOR_CLASS_LABELS.get(int(interval.role_class_id))
        if expected is None or role != expected:
            raise ValueError(
                "role and role_class_id disagree: "
                f"role={role!r}, role_class_id={interval.role_class_id!r}"
            )
        by_instance[instance_id].append(interval)

    for instance_id, intervals in by_instance.items():
        if not intervals:
            raise ValueError(f"chaser identity has no role interval: {instance_id!r}")
        ordered = sorted(intervals, key=lambda value: int(value.start_frame))
        for previous, current in zip(ordered, ordered[1:]):
            if previous.end_frame is None or int(current.start_frame) <= int(
                previous.end_frame
            ):
                raise ValueError(
                    f"overlapping role intervals for stimulus instance {instance_id!r}"
                )


def canonical_chaser_set_from_protocol_payload(
    payload: Mapping[str, Any],
    *,
    total_frames: int | None = None,
    source: str = "protocol_json.steps[].parameters.chasers[]",
) -> CanonicalChaserSet:
    """Translate a variable-length protocol chaser list into canonical rows."""

    configured = resolve_configured_chaser_behaviors(payload)
    if not configured:
        raise ValueError("protocol payload contains no configured chasers")
    end_frame = None if total_frames is None else max(0, int(total_frames) - 1)
    identities = tuple(
        ChaserIdentity(
            stimulus_instance_id=f"chaser:{int(chaser.chaser_index)}",
            chaser_index=int(chaser.chaser_index),
            source_track_key=f"chaser_index:{int(chaser.chaser_index)}",
            raw_color_rgba=tuple(float(value) for value in chaser.raw_color_rgba),
        )
        for chaser in configured
    )
    intervals = tuple(
        ChaserRoleInterval(
            stimulus_instance_id=f"chaser:{int(chaser.chaser_index)}",
            role_class_id=int(chaser.behavior_class_id),
            role=canonical_behavior_label(chaser.behavior_class),
            start_frame=0,
            end_frame=end_frame,
            source=str(source),
        )
        for chaser in configured
    )
    return CanonicalChaserSet(identities=identities, role_intervals=intervals)


__all__ = [
    "CHASER_SOURCE_SCHEMA_ID",
    "CHASER_SOURCE_SCHEMA_VERSION",
    "ROLE_INTERVAL_BOUNDARY_POLICY",
    "CanonicalChaserSet",
    "ChaserIdentity",
    "ChaserRoleInterval",
    "canonical_chaser_set_from_protocol_payload",
    "validate_chaser_set",
]
