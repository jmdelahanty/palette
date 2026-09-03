"""Exact-time hysteretic near-field visit segmentation.

This module is the single state-machine primitive shared by the compact
radial/near-field aggregate and the durable individual-visit successor.  It
does not discover sources, choose providers, interpolate gaps, or publish
anything.  Inputs are one already selected epoch/chaser vector in exact
acquisition-frame order.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

LEFT_CENSOR_NONE = 0
LEFT_CENSOR_PHASE_START = 1
LEFT_CENSOR_INVALID_GAP = 2

RIGHT_CENSOR_NONE = 0
RIGHT_CENSOR_PHASE_END = 1
RIGHT_CENSOR_INVALID_GAP = 2

LEFT_CENSOR_REASONS = {
    LEFT_CENSOR_NONE: "none",
    LEFT_CENSOR_PHASE_START: "phase_start_inside",
    LEFT_CENSOR_INVALID_GAP: "invalid_gap_before_inside",
}
RIGHT_CENSOR_REASONS = {
    RIGHT_CENSOR_NONE: "none",
    RIGHT_CENSOR_PHASE_END: "phase_end_inside",
    RIGHT_CENSOR_INVALID_GAP: "invalid_gap_while_active",
}


class ExactNearFieldVisitError(ValueError):
    """Raised when exact visit segmentation inputs are inconsistent."""


@dataclass(frozen=True, slots=True)
class ExactNearFieldVisit:
    """One observed or censored contiguous hysteretic visit episode.

    ``first_sample_index`` through ``last_inside_index`` is an inclusive,
    contiguous, fully observed source slice.  The first valid sample strictly
    inside the entry radius is retained even when its entry transition is
    left-censored.  A valid first-outside sample is represented separately by
    ``exit_index`` and is not a visit-member sample.
    """

    ordinal: int
    first_sample_index: int
    last_inside_index: int
    entry_index: int | None
    exit_index: int | None
    entry_observed: bool
    exit_observed: bool
    left_censor_reason_code: int
    right_censor_reason_code: int

    @property
    def complete(self) -> bool:
        return self.entry_observed and self.exit_observed

    @property
    def sample_count(self) -> int:
        return self.last_inside_index - self.first_sample_index + 1


@dataclass(frozen=True, slots=True)
class ExactNearFieldVisitSegmentation:
    """Visit rows plus the exact aggregate values used by the radial child."""

    visits: tuple[ExactNearFieldVisit, ...]
    near_dwell_s: float
    valid_tracked_duration_s: float
    entry_count: int
    entry_rate_per_min: float
    complete_visit_median_dwell_s: float
    complete_visit_total_dwell_s: float
    invalid_gap_count: int
    censor_event_count: int
    boundary_censor_event_count: int
    invalid_gap_censor_event_count: int


def _vectors(
    *,
    frame_id: np.ndarray,
    timestamp_ns: np.ndarray,
    timestamp_valid: np.ndarray,
    distance_mm: np.ndarray,
    distance_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame = np.asarray(frame_id, dtype=np.int64).reshape(-1)
    timestamp = np.asarray(timestamp_ns, dtype=np.int64).reshape(-1)
    timestamp_ok = np.asarray(timestamp_valid, dtype=bool).reshape(-1)
    distance = np.asarray(distance_mm, dtype=np.float64).reshape(-1)
    distance_ok = np.asarray(distance_valid, dtype=bool).reshape(-1)
    if not (
        frame.size
        == timestamp.size
        == timestamp_ok.size
        == distance.size
        == distance_ok.size
    ):
        raise ExactNearFieldVisitError(
            "Exact temporal vectors have inconsistent lengths."
        )
    if frame.size > 1 and np.any(np.diff(frame) <= 0):
        raise ExactNearFieldVisitError(
            "Exact temporal frame IDs must be strictly increasing."
        )
    if np.any(timestamp_ok & (timestamp < 0)):
        raise ExactNearFieldVisitError(
            "Declared-valid session timestamps must be non-negative."
        )
    observed = distance_ok & timestamp_ok & np.isfinite(distance)
    return frame, timestamp, distance, observed


def segment_exact_time_near_field_visits(
    *,
    frame_id: np.ndarray,
    timestamp_ns: np.ndarray,
    timestamp_valid: np.ndarray,
    distance_mm: np.ndarray,
    distance_valid: np.ndarray,
    near_zone_mm: float,
    enter_mm: float,
    exit_mm: float,
) -> ExactNearFieldVisitSegmentation:
    """Segment visits without bridging invalid or nonadjacent frame evidence.

    Entry and exit comparisons are deliberately strict: entry requires
    ``distance < enter_mm`` after a known outside state, while exit requires
    ``distance > exit_mm``.  Equality is retained in the hysteresis state and
    cannot manufacture a transition.
    """

    near = float(near_zone_mm)
    enter = float(enter_mm)
    exit_ = float(exit_mm)
    if not all(math.isfinite(value) and value >= 0.0 for value in (near, enter)):
        raise ExactNearFieldVisitError(
            "Near-zone and entry radii must be finite and non-negative."
        )
    if not math.isfinite(exit_) or exit_ <= enter:
        raise ExactNearFieldVisitError(
            "Exit radius must be finite and strictly greater than entry radius."
        )
    frame, timestamp, distance, observed = _vectors(
        frame_id=frame_id,
        timestamp_ns=timestamp_ns,
        timestamp_valid=timestamp_valid,
        distance_mm=distance_mm,
        distance_valid=distance_valid,
    )

    state = "unknown"
    unknown_origin = "boundary"
    active_first: int | None = None
    active_last: int | None = None
    active_entry: int | None = None
    active_left_censor = LEFT_CENSOR_NONE
    visits: list[ExactNearFieldVisit] = []
    complete_dwell: list[float] = []
    near_dwell_s = 0.0
    tracked_s = 0.0
    entries = 0
    gaps = 0
    censors = 0
    boundary_censors = 0
    gap_censors = 0
    gap_open = False

    def start(
        index: int,
        *,
        entry_observed: bool,
        left_censor_reason: int,
    ) -> None:
        nonlocal active_first, active_last, active_entry, active_left_censor
        active_first = index
        active_last = index
        active_entry = index if entry_observed else None
        active_left_censor = left_censor_reason

    def close(
        *,
        exit_index: int | None,
        right_censor_reason: int,
    ) -> None:
        nonlocal active_first, active_last, active_entry, active_left_censor
        if active_first is None or active_last is None:
            raise ExactNearFieldVisitError(
                "Active near-field state lacks its exact source bounds."
            )
        entry_observed = active_entry is not None
        exit_observed = exit_index is not None
        visits.append(
            ExactNearFieldVisit(
                ordinal=len(visits),
                first_sample_index=active_first,
                last_inside_index=active_last,
                entry_index=active_entry,
                exit_index=exit_index,
                entry_observed=entry_observed,
                exit_observed=exit_observed,
                left_censor_reason_code=active_left_censor,
                right_censor_reason_code=right_censor_reason,
            )
        )
        if entry_observed and exit_observed:
            dwell_s = float(timestamp[exit_index] - timestamp[active_entry]) / 1e9
            if dwell_s <= 0.0:
                raise ExactNearFieldVisitError(
                    "Complete visit duration must be strictly positive."
                )
            complete_dwell.append(dwell_s)
        active_first = None
        active_last = None
        active_entry = None
        active_left_censor = LEFT_CENSOR_NONE

    for index in range(frame.size):
        continuous = index > 0 and bool(
            observed[index - 1]
            and observed[index]
            and frame[index] == frame[index - 1] + 1
            and timestamp[index] > timestamp[index - 1]
        )
        if continuous:
            delta_s = float(timestamp[index] - timestamp[index - 1]) / 1e9
            tracked_s += delta_s
            if distance[index - 1] <= near:
                near_dwell_s += delta_s
            gap_open = False
        elif index > 0:
            if not gap_open:
                gaps += 1
            gap_open = True
            if state in {"inside", "censored_inside"}:
                close(
                    exit_index=None,
                    right_censor_reason=RIGHT_CENSOR_INVALID_GAP,
                )
                censors += 1
                gap_censors += 1
            state = "unknown"
            unknown_origin = "gap"

        if not observed[index]:
            continue
        value = float(distance[index])
        if state == "unknown":
            if value > exit_:
                state = "outside"
            elif value < enter:
                state = "censored_inside"
                left_reason = (
                    LEFT_CENSOR_PHASE_START
                    if unknown_origin == "boundary"
                    else LEFT_CENSOR_INVALID_GAP
                )
                start(
                    index,
                    entry_observed=False,
                    left_censor_reason=left_reason,
                )
                censors += 1
                if unknown_origin == "boundary":
                    boundary_censors += 1
                else:
                    gap_censors += 1
            continue
        if state == "outside":
            if value < enter:
                state = "inside"
                start(
                    index,
                    entry_observed=True,
                    left_censor_reason=LEFT_CENSOR_NONE,
                )
                entries += 1
            continue
        if state in {"inside", "censored_inside"}:
            if value > exit_:
                close(exit_index=index, right_censor_reason=RIGHT_CENSOR_NONE)
                state = "outside"
            else:
                active_last = index
            continue
        raise ExactNearFieldVisitError(f"Unknown visit state {state!r}.")

    if state in {"inside", "censored_inside"}:
        close(exit_index=None, right_censor_reason=RIGHT_CENSOR_PHASE_END)
        censors += 1
        boundary_censors += 1

    dwell = np.asarray(complete_dwell, dtype=np.float64)
    return ExactNearFieldVisitSegmentation(
        visits=tuple(visits),
        near_dwell_s=near_dwell_s,
        valid_tracked_duration_s=tracked_s,
        entry_count=entries,
        entry_rate_per_min=(
            float(entries) / (tracked_s / 60.0) if tracked_s > 0 else math.nan
        ),
        complete_visit_median_dwell_s=(
            float(np.median(dwell)) if dwell.size else math.nan
        ),
        complete_visit_total_dwell_s=(float(np.sum(dwell)) if dwell.size else 0.0),
        invalid_gap_count=gaps,
        censor_event_count=censors,
        boundary_censor_event_count=boundary_censors,
        invalid_gap_censor_event_count=gap_censors,
    )


__all__ = [
    "LEFT_CENSOR_INVALID_GAP",
    "LEFT_CENSOR_NONE",
    "LEFT_CENSOR_PHASE_START",
    "LEFT_CENSOR_REASONS",
    "RIGHT_CENSOR_INVALID_GAP",
    "RIGHT_CENSOR_NONE",
    "RIGHT_CENSOR_PHASE_END",
    "RIGHT_CENSOR_REASONS",
    "ExactNearFieldVisit",
    "ExactNearFieldVisitError",
    "ExactNearFieldVisitSegmentation",
    "segment_exact_time_near_field_visits",
]
