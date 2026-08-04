"""Exact source binding and aggregation policy for activity/spatial time bins.

The portable table is deliberately geometry-honest: it summarizes verified
physical-mm track positions and mapped swim-bout events, but it does not claim
arena-normalized occupancy without an experimental-area geometry authority.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from fisheye.analysis._exact_tabular_run_schema import MANIFEST_ATTRIBUTE
from fisheye.analysis.swim_bout_frame_axis import (
    FRAME_AXIS_CONTRACT_ATTR,
    FRAME_AXIS_CONTRACT_SHA256_ATTR,
    canonical_frame_axis_sha256,
    resolve_swim_bout_frame_axis,
)
from fisheye.analysis.swim_bout_io import (
    SwimBoutCandidate,
    SwimBoutEvents,
    SwimBoutSignalVariant,
    load_swim_bout_events,
    resolve_swim_bout_candidate,
)
from fisheye.analysis.swim_bout_schema import (
    SWIM_BOUT_LAYOUT,
    SWIM_BOUT_RUN_SCHEMA_ID,
    SWIM_BOUT_RUN_SCHEMA_VERSION,
    validate_swim_bout_array_manifest,
)
from fisheye.analytics_exports import kinematics_samples as track_export
from fisheye.analytics_exports.publication import safe_component
from fisheye.shared.coordinate_frame_record import array_values_sha256
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr_run_completion import (
    is_run_complete_in_parent,
    is_run_selector_eligible,
)


ACTIVITY_SPATIAL_SOURCE_BINDING_SCHEMA_ID = (
    "palette.activity_spatial_time_bins.source_binding"
)
ACTIVITY_SPATIAL_SOURCE_BINDING_SCHEMA_VERSION = 1
ACTIVITY_SPATIAL_BINNING_SCHEMA_ID = "palette.activity_spatial_time_bins.binning"
ACTIVITY_SPATIAL_BINNING_SCHEMA_VERSION = 1
ACTIVITY_SPATIAL_SOURCE_SPEED_LEVEL = "filtered"
ACTIVITY_SPATIAL_BINNING_POLICY = (
    "global_acquisition_frame_fixed_width_round_half_up_v1"
)

_REQUIRED_BOUT_FIELDS = frozenset(
    {
        "candidate_id",
        "signal_id",
        "track_id",
        "bout_id",
        "start_frame",
        "end_frame",
        "duration_s",
        "path_length_mm",
    }
)


@dataclass(frozen=True)
class BoundSwimBoutSource:
    """One exact track-owned swim-bout source and its selected event rows."""

    binding: Mapping[str, Any]
    events: SwimBoutEvents
    frame_axis: np.ndarray


@dataclass(frozen=True)
class BoundActivitySpatialSources:
    """The track-motion authority plus one explicit bout run per track."""

    track_source: Any
    bout_sources: Mapping[int, BoundSwimBoutSource]
    binding: Mapping[str, Any]


def activity_spatial_binning_contract(
    *,
    source_sample_rate_hz: float,
    requested_bin_size_s: float,
) -> dict[str, Any]:
    """Return the deterministic global acquisition-frame binning policy."""

    for label, value in (
        ("source sample rate", source_sample_rate_hz),
        ("requested bin size", requested_bin_size_s),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise ValueError(f"{label} must be positive and finite.")
    source_rate = float(source_sample_rate_hz)
    requested = float(requested_bin_size_s)
    bin_frames = max(1, int(math.floor(source_rate * requested + 0.5)))
    body: dict[str, Any] = {
        "schema_id": ACTIVITY_SPATIAL_BINNING_SCHEMA_ID,
        "schema_version": ACTIVITY_SPATIAL_BINNING_SCHEMA_VERSION,
        "source_sample_rate_hz": source_rate,
        "requested_bin_size_s": requested,
        "bin_size_frames": bin_frames,
        "effective_bin_size_s": float(bin_frames) / source_rate,
        "binning_policy": ACTIVITY_SPATIAL_BINNING_POLICY,
        "bin_index_expression": (
            "source_acquisition_frame_index // bin_size_frames"
        ),
        "edge_policy": "clip_expected_denominator_to_track_frame_span",
        "gap_policy": "emit_empty_internal_bins",
        "source_speed_level": ACTIVITY_SPATIAL_SOURCE_SPEED_LEVEL,
        "position_policy": "sample_valid_and_position_finite_physical_mm",
        "speed_policy": "transition_valid_and_finite_filtered_mm_s",
        "bout_count_policy": "assign_whole_bout_by_start_frame",
        "bout_occupancy_policy": (
            "union_inclusive_bout_frame_intervals_clipped_to_bin"
        ),
        "invalid_float_semantics": "ieee_nan_not_arrow_null",
    }
    return {**body, "payload_sha256": canonical_json_sha256(body)}


def _attrs(group: Any) -> dict[str, Any]:
    attrs = getattr(group, "attrs", {})
    return dict(attrs.asdict() if hasattr(attrs, "asdict") else dict(attrs))


def _default_signal(candidate: SwimBoutCandidate) -> SwimBoutSignalVariant:
    if candidate.default_signal_id is None:
        raise ValueError(
            f"Swim-bout run {candidate.run_name!r} has no explicit default signal."
        )
    matches = tuple(
        signal
        for signal in candidate.signals
        if int(signal.signal_id) == int(candidate.default_signal_id)
    )
    if len(matches) != 1 or not matches[0].is_default:
        raise ValueError(
            f"Swim-bout run {candidate.run_name!r} does not resolve exactly one "
            "declared default signal."
        )
    return matches[0]


def _bind_swim_bout_source(
    root: Any,
    *,
    run_name: str,
    track_record: Mapping[str, Any],
    track_binding: Mapping[str, Any],
) -> BoundSwimBoutSource:
    track_id = int(track_record["track_id"])
    parent = root["analysis"]["swim_bout_runs"]
    run = parent[run_name]
    attrs = _attrs(run)
    if (
        attrs.get("schema_id") != SWIM_BOUT_RUN_SCHEMA_ID
        or attrs.get("schema_version") != SWIM_BOUT_RUN_SCHEMA_VERSION
        or attrs.get("layout") != SWIM_BOUT_LAYOUT
    ):
        raise ValueError(
            f"Swim-bout run {run_name!r} is not the maintained exact schema."
        )
    if not is_run_selector_eligible(run) or not is_run_complete_in_parent(
        parent,
        run,
        legacy_default=False,
    ):
        raise ValueError(
            f"Swim-bout run {run_name!r} must be complete and selector-eligible."
        )
    errors = validate_swim_bout_array_manifest(run)
    if errors:
        raise ValueError(
            f"Swim-bout run {run_name!r} violates its exact manifest: "
            + "; ".join(errors)
        )
    manifest = attrs.get(MANIFEST_ATTRIBUTE)
    if not isinstance(manifest, Mapping):
        raise ValueError(f"Swim-bout run {run_name!r} lacks its exact manifest.")
    manifest_sha = canonical_json_sha256(manifest)
    if attrs.get("source_track_kinematics_run") != track_binding["run_name"]:
        raise ValueError(
            f"Swim-bout run {run_name!r} binds a different track-motion run."
        )
    if attrs.get("source_track_motion_manifest_sha256") != track_binding[
        "source_manifest_sha256"
    ]:
        raise ValueError(
            f"Swim-bout run {run_name!r} binds a different track-motion manifest."
        )
    if type(attrs.get("track_id")) is not int or attrs["track_id"] != track_id:
        raise ValueError(
            f"Swim-bout run {run_name!r} does not belong to track {track_id}."
        )
    fps = attrs.get("fps")
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or float(fps) != float(track_binding["source_sample_rate_hz"])
    ):
        raise ValueError(f"Swim-bout run {run_name!r} source FPS differs.")

    candidate = resolve_swim_bout_candidate(
        root,
        run_name=run_name,
        legacy_compatibility=False,
    )
    if candidate.run_name != run_name or candidate.track_id != track_id:
        raise ValueError(
            f"Swim-bout run {run_name!r} candidate track identity differs."
        )
    signal = _default_signal(candidate)
    events = load_swim_bout_events(
        root,
        candidate=candidate,
        signal=signal,
        legacy_compatibility=False,
    )
    bouts = np.asarray(events.bouts)
    names = frozenset(bouts.dtype.names or ())
    if not _REQUIRED_BOUT_FIELDS.issubset(names):
        missing = sorted(_REQUIRED_BOUT_FIELDS - names)
        raise ValueError(
            f"Swim-bout run {run_name!r} event fields are incomplete: {missing!r}."
        )
    if bouts.size and (
        np.any(np.asarray(bouts["candidate_id"]) != candidate.candidate_id)
        or np.any(np.asarray(bouts["signal_id"]) != signal.signal_id)
        or np.any(np.asarray(bouts["track_id"]) != track_id)
    ):
        raise ValueError(
            f"Swim-bout run {run_name!r} contains cross-selection event rows."
        )
    if bouts.size:
        bout_ids = np.asarray(bouts["bout_id"], dtype=np.int64)
        starts = np.asarray(bouts["start_frame"], dtype=np.int64)
        ends = np.asarray(bouts["end_frame"], dtype=np.int64)
        durations = np.asarray(bouts["duration_s"], dtype=np.float64)
        paths = np.asarray(bouts["path_length_mm"], dtype=np.float64)
        if np.unique(bout_ids).size != bout_ids.size:
            raise ValueError(f"Swim-bout run {run_name!r} has duplicate bout IDs.")
        if (
            np.any(starts < 0)
            or np.any(ends < starts)
            or np.any(~np.isfinite(durations))
            or np.any(durations < 0)
            or np.any(~np.isfinite(paths))
            or np.any(paths < 0)
        ):
            raise ValueError(
                f"Swim-bout run {run_name!r} has invalid physical event values."
            )

    sample_count = int(track_record["sample_count"])
    frame_axis = resolve_swim_bout_frame_axis(
        root,
        run,
        expected_length=sample_count,
    )
    if frame_axis is None:
        raise ValueError(f"Swim-bout run {run_name!r} has no exact frame axis.")
    frame_axis = np.asarray(frame_axis, dtype=np.int64)
    track_frame_record = track_record["selected_surfaces"][
        "source_acquisition_frame_index"
    ]
    if array_values_sha256(frame_axis) != track_frame_record["content_sha256"]:
        raise ValueError(
            f"Swim-bout run {run_name!r} frame axis differs from track {track_id}."
        )
    raw_axis_contract = attrs.get(FRAME_AXIS_CONTRACT_ATTR)
    if not isinstance(raw_axis_contract, Mapping):
        raise ValueError(
            f"Swim-bout run {run_name!r} requires the canonical frame-axis contract."
        )
    axis_contract_sha = canonical_json_sha256(raw_axis_contract)
    if attrs.get(FRAME_AXIS_CONTRACT_SHA256_ATTR) != axis_contract_sha:
        raise ValueError(
            f"Swim-bout run {run_name!r} frame-axis digest is invalid."
        )

    body: dict[str, Any] = {
        "schema_id": ACTIVITY_SPATIAL_SOURCE_BINDING_SCHEMA_ID,
        "schema_version": ACTIVITY_SPATIAL_SOURCE_BINDING_SCHEMA_VERSION,
        "stage_id": "swim_bouts",
        "track_id": track_id,
        "run_name": run_name,
        "run_path": f"analysis/swim_bout_runs/{run_name}",
        "source_schema_id": attrs["schema_id"],
        "source_schema_version": attrs["schema_version"],
        "source_array_manifest_sha256": manifest_sha,
        "source_track_kinematics_run": attrs["source_track_kinematics_run"],
        "source_track_motion_manifest_sha256": attrs[
            "source_track_motion_manifest_sha256"
        ],
        "source_sample_rate_hz": float(fps),
        "candidate_id": int(candidate.candidate_id),
        "candidate_name": str(candidate.candidate_name),
        "signal_id": int(signal.signal_id),
        "signal_name": str(signal.signal_name),
        "speed_level": str(signal.speed_level),
        "frame_axis_contract_sha256": axis_contract_sha,
        "frame_axis_content_sha256": canonical_frame_axis_sha256(frame_axis),
        "frame_axis_array_values_sha256": array_values_sha256(frame_axis),
        "bout_count": int(bouts.shape[0]),
        "bout_dtype": bouts.dtype.descr,
        "bout_content_sha256": array_values_sha256(bouts),
        "selection_snapshot": {
            "mode": "explicit_per_track_run",
            "parent_latest": parent.attrs.get("latest"),
            "parent_latest_complete": parent.attrs.get("latest_complete"),
            "parent_completion_epoch": parent.attrs.get("palette_completion_epoch"),
        },
        "completion_snapshot": {
            "status": attrs.get("palette_run_completion_status"),
            "completed_at_utc": attrs.get("palette_run_completed_at_utc"),
            "selector_eligible": attrs.get("stage_selector_eligible"),
        },
    }
    return BoundSwimBoutSource(
        binding={**body, "payload_sha256": canonical_json_sha256(body)},
        events=events,
        frame_axis=frame_axis,
    )


def bind_activity_spatial_sources(
    root: Any,
    *,
    zarr_path: str | Path,
    recording_id: str,
    track_kinematics_run: str,
    track_scope: str,
    swim_bout_runs_by_track: Mapping[int, str],
) -> BoundActivitySpatialSources:
    """Bind one track authority and exactly one maintained bout run per track."""

    track_source = track_export._source_binding(
        root,
        zarr_path=Path(zarr_path).expanduser().resolve(),
        recording_id=str(recording_id),
        run_name=safe_component(
            track_kinematics_run,
            label="track-kinematics run ID",
        ),
        scope=track_scope,
    )
    track_records = {
        int(record["track_id"]): record for record in track_source.binding["tracks"]
    }
    normalized: dict[int, str] = {}
    for raw_track_id, raw_run_name in swim_bout_runs_by_track.items():
        if isinstance(raw_track_id, bool) or type(raw_track_id) is not int:
            raise ValueError("Swim-bout run-map keys must be exact integer track IDs.")
        normalized[raw_track_id] = safe_component(
            raw_run_name,
            label=f"track {raw_track_id} swim-bout run ID",
        )
    if set(normalized) != set(track_records):
        raise ValueError(
            "Swim-bout run map must bind every and only track-motion track ID."
        )
    if len(set(normalized.values())) != len(normalized):
        raise ValueError(
            "Maintained one-track swim-bout runs cannot be reused across track IDs."
        )
    bout_sources = {
        track_id: _bind_swim_bout_source(
            root,
            run_name=normalized[track_id],
            track_record=track_records[track_id],
            track_binding=track_source.binding,
        )
        for track_id in sorted(track_records)
    }
    body: dict[str, Any] = {
        "schema_id": ACTIVITY_SPATIAL_SOURCE_BINDING_SCHEMA_ID,
        "schema_version": ACTIVITY_SPATIAL_SOURCE_BINDING_SCHEMA_VERSION,
        "recording_id": str(recording_id),
        "zarr_path": str(Path(zarr_path).expanduser().resolve()),
        "track_source_binding": track_source.binding,
        "swim_bout_runs_by_track": {
            str(track_id): bout_sources[track_id].binding
            for track_id in sorted(bout_sources)
        },
    }
    return BoundActivitySpatialSources(
        track_source=track_source,
        bout_sources=bout_sources,
        binding={**body, "payload_sha256": canonical_json_sha256(body)},
    )


def _fraction(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else float("nan")


def _interval_union_count(
    starts: np.ndarray,
    ends_inclusive: np.ndarray,
    *,
    lower: int,
    upper_exclusive: int,
) -> int:
    intervals = sorted(
        (
            max(int(start), lower),
            min(int(end) + 1, upper_exclusive),
        )
        for start, end in zip(starts, ends_inclusive, strict=True)
        if int(end) >= lower and int(start) < upper_exclusive
    )
    occupied = 0
    merged_end = lower
    for start, end in intervals:
        if end <= start:
            continue
        if start >= merged_end:
            occupied += end - start
            merged_end = end
        elif end > merged_end:
            occupied += end - merged_end
            merged_end = end
    return occupied


def summarize_activity_spatial_track(
    *,
    track_id: int,
    source_acquisition_frame_index: np.ndarray,
    source_observed: np.ndarray,
    sample_valid: np.ndarray,
    position_finite: np.ndarray,
    transition_valid: np.ndarray,
    positions_mm: np.ndarray,
    filtered_speed_mm_s: np.ndarray,
    filtered_path_distance_mm: np.ndarray,
    bouts: np.ndarray,
    source_sample_rate_hz: float,
    requested_bin_size_s: float,
) -> list[dict[str, Any]]:
    """Aggregate one track into deterministic global acquisition-frame bins."""

    policy = activity_spatial_binning_contract(
        source_sample_rate_hz=source_sample_rate_hz,
        requested_bin_size_s=requested_bin_size_s,
    )
    frames = np.asarray(source_acquisition_frame_index, dtype=np.int64)
    count = int(frames.shape[0])
    one_dimensional = {
        "source_observed": np.asarray(source_observed, dtype=bool),
        "sample_valid": np.asarray(sample_valid, dtype=bool),
        "position_finite": np.asarray(position_finite, dtype=bool),
        "transition_valid": np.asarray(transition_valid, dtype=bool),
        "filtered_speed_mm_s": np.asarray(filtered_speed_mm_s, dtype=np.float64),
        "filtered_path_distance_mm": np.asarray(
            filtered_path_distance_mm,
            dtype=np.float64,
        ),
    }
    for name, values in one_dimensional.items():
        if values.shape != (count,):
            raise ValueError(f"{name} shape must equal the track frame axis.")
    positions = np.asarray(positions_mm, dtype=np.float64)
    if positions.shape != (count, 2):
        raise ValueError("positions_mm shape must be (track_sample, xy=2).")
    if count == 0:
        if np.asarray(bouts).shape[0] != 0:
            raise ValueError("An empty track cannot own swim-bout events.")
        return []
    if np.any(frames < 0) or np.any(np.diff(frames) <= 0):
        raise ValueError("Track acquisition frames must be nonnegative and increasing.")

    events = np.asarray(bouts)
    names = frozenset(events.dtype.names or ())
    if not _REQUIRED_BOUT_FIELDS.issubset(names):
        raise ValueError("Bout events do not contain the exact summary fields.")
    if events.size and np.any(np.asarray(events["track_id"]) != int(track_id)):
        raise ValueError("Bout events belong to a different track.")
    bout_starts = np.asarray(events["start_frame"], dtype=np.int64)
    bout_ends = np.asarray(events["end_frame"], dtype=np.int64)
    bout_durations = np.asarray(events["duration_s"], dtype=np.float64)
    bout_paths = np.asarray(events["path_length_mm"], dtype=np.float64)

    bin_frames = int(policy["bin_size_frames"])
    first_frame = int(frames[0])
    last_frame = int(frames[-1])
    first_bin = first_frame // bin_frames
    last_bin = last_frame // bin_frames
    rows: list[dict[str, Any]] = []
    for bin_index in range(first_bin, last_bin + 1):
        full_start = bin_index * bin_frames
        full_end = full_start + bin_frames
        span_start = max(full_start, first_frame)
        span_end = min(full_end, last_frame + 1)
        expected = span_end - span_start
        left = int(np.searchsorted(frames, span_start, side="left"))
        right = int(np.searchsorted(frames, span_end, side="left"))
        local = slice(left, right)
        source_count = right - left
        observed_count = int(np.count_nonzero(one_dimensional["source_observed"][local]))
        sample_valid_count = int(np.count_nonzero(one_dimensional["sample_valid"][local]))
        position_mask = (
            one_dimensional["sample_valid"][local]
            & one_dimensional["position_finite"][local]
            & np.all(np.isfinite(positions[local]), axis=1)
        )
        valid_positions = positions[local][position_mask]
        position_count = int(valid_positions.shape[0])
        position_metrics_valid = position_count > 0
        if position_metrics_valid:
            mean = np.mean(valid_positions, axis=0, dtype=np.float64)
            std = np.std(valid_positions, axis=0, ddof=0, dtype=np.float64)
            centered = valid_positions - mean
            covariance = float(np.mean(centered[:, 0] * centered[:, 1]))
            minimum = np.min(valid_positions, axis=0)
            maximum = np.max(valid_positions, axis=0)
            net_displacement = (
                float(np.linalg.norm(valid_positions[-1] - valid_positions[0]))
                if position_count >= 2
                else float("nan")
            )
        else:
            mean = std = minimum = maximum = np.full(2, np.nan, dtype=np.float64)
            covariance = net_displacement = float("nan")

        transition_mask = one_dimensional["transition_valid"][local]
        transition_count = int(np.count_nonzero(transition_mask))
        speeds = one_dimensional["filtered_speed_mm_s"][local]
        paths = one_dimensional["filtered_path_distance_mm"][local]
        speed_mask = transition_mask & np.isfinite(speeds)
        path_mask = transition_mask & np.isfinite(paths)
        valid_speeds = speeds[speed_mask]
        speed_metrics_valid = valid_speeds.size > 0 and np.count_nonzero(path_mask) > 0
        if valid_speeds.size:
            mean_speed = float(np.mean(valid_speeds, dtype=np.float64))
            median_speed = float(np.median(valid_speeds))
            p95_speed = float(np.percentile(valid_speeds, 95))
        else:
            mean_speed = median_speed = p95_speed = float("nan")
        path_sum = (
            float(np.sum(paths[path_mask], dtype=np.float64))
            if np.any(path_mask)
            else float("nan")
        )

        started = (bout_starts >= full_start) & (bout_starts < full_end)
        bout_count = int(np.count_nonzero(started))
        duration_sum = float(np.sum(bout_durations[started], dtype=np.float64))
        bout_path_sum = float(np.sum(bout_paths[started], dtype=np.float64))
        occupied = _interval_union_count(
            bout_starts,
            bout_ends,
            lower=span_start,
            upper_exclusive=span_end,
        )
        bin_valid = source_count > 0 and position_metrics_valid
        reason_code = 0 if bin_valid else (1 if source_count == 0 else 2)
        rows.append(
            {
                "track_id": int(track_id),
                "time_bin_index": bin_index,
                "start_acquisition_frame_index": full_start,
                "end_acquisition_frame_index_exclusive": full_end,
                "start_time_seconds": float(full_start) / float(source_sample_rate_hz),
                "end_time_seconds": float(full_end) / float(source_sample_rate_hz),
                "bin_duration_seconds": float(policy["effective_bin_size_s"]),
                "expected_track_frame_count": expected,
                "source_sample_count": source_count,
                "source_observed_count": observed_count,
                "source_observed_fraction": _fraction(observed_count, expected),
                "sample_valid_count": sample_valid_count,
                "sample_valid_fraction": _fraction(sample_valid_count, expected),
                "position_valid_count": position_count,
                "position_valid_fraction": _fraction(position_count, expected),
                "transition_valid_count": transition_count,
                "transition_valid_fraction": _fraction(transition_count, expected),
                "mean_position_x_mm": float(mean[0]),
                "mean_position_y_mm": float(mean[1]),
                "std_position_x_mm": float(std[0]),
                "std_position_y_mm": float(std[1]),
                "covariance_xy_mm2": covariance,
                "min_position_x_mm": float(minimum[0]),
                "max_position_x_mm": float(maximum[0]),
                "min_position_y_mm": float(minimum[1]),
                "max_position_y_mm": float(maximum[1]),
                "net_displacement_mm": net_displacement,
                "mean_speed_mm_s": mean_speed,
                "median_speed_mm_s": median_speed,
                "p95_speed_mm_s": p95_speed,
                "path_distance_mm_sum": path_sum,
                "bout_count_started": bout_count,
                "bout_duration_s_started_sum": duration_sum,
                "bout_path_length_mm_started_sum": bout_path_sum,
                "bout_occupied_frame_count": occupied,
                "bout_occupancy_fraction": _fraction(occupied, expected),
                "position_metrics_valid": position_metrics_valid,
                "speed_metrics_valid": speed_metrics_valid,
                "bout_metrics_valid": True,
                "bin_valid": bin_valid,
                "bin_reason_code": reason_code,
            }
        )
    return rows


__all__ = [
    "ACTIVITY_SPATIAL_BINNING_POLICY",
    "ACTIVITY_SPATIAL_BINNING_SCHEMA_ID",
    "ACTIVITY_SPATIAL_SOURCE_BINDING_SCHEMA_ID",
    "BoundActivitySpatialSources",
    "BoundSwimBoutSource",
    "activity_spatial_binning_contract",
    "bind_activity_spatial_sources",
    "summarize_activity_spatial_track",
]
