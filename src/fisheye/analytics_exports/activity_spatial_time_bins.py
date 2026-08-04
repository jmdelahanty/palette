"""Exact source binding and aggregation policy for activity/spatial time bins.

The portable table is deliberately geometry-honest: it summarizes verified
physical-mm track positions and mapped swim-bout events, but it does not claim
arena-normalized occupancy without an experimental-area geometry authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import socket
import struct
from typing import Any, Mapping
import uuid

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
from fisheye.analytics_exports.arrow_contracts import (
    ARROW_TABLE_CONTRACTS,
    arrow_contract_envelope,
    exact_arrow_schema,
    validate_arrow_schema,
)
from fisheye.analytics_exports.capabilities import resolve_capabilities
from fisheye.analytics_exports.contracts import (
    ACTIVITY_SPATIAL_TIME_BINS_TABLE,
    EXPORT_SCHEMA_ID,
    EXPORT_SCHEMA_VERSION,
    TABLE_CONTRACTS,
    contract_snapshot,
)
from fisheye.analytics_exports.publication import (
    PUBLICATION_SCHEMA_ID,
    PUBLICATION_SCHEMA_VERSION,
    commit_staged_publication,
    export_manifest_path,
    generation_relative_path,
    manifest_identity,
    manifest_selected_part_files_from_payload,
    publication_generation_root,
    publication_staging_root,
    safe_component,
    sha256_file,
)
from fisheye.analytics_exports.runtime_telemetry import ExportRuntimePhaseRecorder
from fisheye.shared.coordinate_frame_record import array_values_sha256
from fisheye.shared.system_metadata import get_git_info
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.manifest_digest import canonical_json_bytes
from fisheye.shared.zarr_io import open_zarr_root
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
ACTIVITY_SPATIAL_EXPORT_SCHEMA_ID = (
    "palette.analytics_export.activity_spatial_time_bins"
)
ACTIVITY_SPATIAL_EXPORT_SCHEMA_VERSION = 1
ACTIVITY_SPATIAL_DECODED_PAYLOAD_SCHEMA_ID = (
    "palette.activity_spatial_time_bins.decoded_payload"
)
ACTIVITY_SPATIAL_DECODED_PAYLOAD_SCHEMA_VERSION = 1
ACTIVITY_SPATIAL_PARQUET_POLICY_SCHEMA_ID = (
    "palette.activity_spatial_time_bins.parquet_policy"
)
ACTIVITY_SPATIAL_PARQUET_POLICY_SCHEMA_VERSION = 1

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
_SOURCE_BINDING_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "recording_id",
        "zarr_path",
        "track_source_binding",
        "swim_bout_runs_by_track",
        "payload_sha256",
    }
)
_BOUT_BINDING_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "stage_id",
        "track_id",
        "run_name",
        "run_path",
        "source_schema_id",
        "source_schema_version",
        "source_array_manifest_sha256",
        "source_track_kinematics_run",
        "source_track_motion_manifest_sha256",
        "source_sample_rate_hz",
        "candidate_id",
        "candidate_name",
        "signal_id",
        "signal_name",
        "speed_level",
        "frame_axis_contract_sha256",
        "frame_axis_content_sha256",
        "frame_axis_array_values_sha256",
        "frame_axis_first_frame",
        "frame_axis_last_frame",
        "bout_count",
        "bout_dtype",
        "bout_content_sha256",
        "selection_snapshot",
        "completion_snapshot",
        "payload_sha256",
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
        "bin_index_expression": ("source_acquisition_frame_index // bin_size_frames"),
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
    if (
        attrs.get("source_track_motion_manifest_sha256")
        != track_binding["source_manifest_sha256"]
    ):
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
    if frame_axis.size:
        if bouts.size and (
            np.any(np.asarray(bouts["start_frame"], dtype=np.int64) < frame_axis[0])
            or np.any(np.asarray(bouts["end_frame"], dtype=np.int64) > frame_axis[-1])
        ):
            raise ValueError(
                f"Swim-bout run {run_name!r} has events outside its track frame span."
            )
    elif bouts.size:
        raise ValueError(
            f"Swim-bout run {run_name!r} has events for an empty track frame axis."
        )
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
        raise ValueError(f"Swim-bout run {run_name!r} frame-axis digest is invalid.")

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
        "frame_axis_first_frame": (int(frame_axis[0]) if frame_axis.size else None),
        "frame_axis_last_frame": (int(frame_axis[-1]) if frame_axis.size else None),
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
    track_frame_span: tuple[int, int] | None = None,
    time_bin_range: tuple[int, int] | None = None,
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
    if count == 0 and track_frame_span is None:
        if np.asarray(bouts).shape[0] != 0:
            raise ValueError("An empty track cannot own swim-bout events.")
        return []
    if count and (np.any(frames < 0) or np.any(np.diff(frames) <= 0)):
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
    if track_frame_span is None:
        first_frame = int(frames[0])
        last_frame = int(frames[-1])
    else:
        if (
            len(track_frame_span) != 2
            or isinstance(track_frame_span[0], bool)
            or isinstance(track_frame_span[1], bool)
            or type(track_frame_span[0]) is not int
            or type(track_frame_span[1]) is not int
            or track_frame_span[0] < 0
            or track_frame_span[1] < track_frame_span[0]
        ):
            raise ValueError("track_frame_span must be one inclusive frame pair.")
        first_frame, last_frame = track_frame_span
        if count and (frames[0] < first_frame or frames[-1] > last_frame):
            raise ValueError("Local frames fall outside the declared track span.")
    if time_bin_range is None:
        first_bin = first_frame // bin_frames
        last_bin = last_frame // bin_frames
    else:
        if (
            len(time_bin_range) != 2
            or isinstance(time_bin_range[0], bool)
            or isinstance(time_bin_range[1], bool)
            or type(time_bin_range[0]) is not int
            or type(time_bin_range[1]) is not int
            or time_bin_range[0] < 0
            or time_bin_range[1] < time_bin_range[0]
        ):
            raise ValueError("time_bin_range must be one inclusive bin pair.")
        first_bin, last_bin = time_bin_range
        expected_first = first_frame // bin_frames
        expected_last = last_frame // bin_frames
        if first_bin < expected_first or last_bin > expected_last:
            raise ValueError("Requested bins fall outside the declared track span.")
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
        observed_count = int(
            np.count_nonzero(one_dimensional["source_observed"][local])
        )
        sample_valid_count = int(
            np.count_nonzero(one_dimensional["sample_valid"][local])
        )
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


def activity_spatial_parquet_policy(*, row_group_rows: int) -> dict[str, Any]:
    """Return the exact one-part Parquet encoding policy for this table."""

    if type(row_group_rows) is not int or row_group_rows <= 0:
        raise ValueError("row_group_rows must be a positive exact integer.")
    string_columns = [
        field.name
        for field in ARROW_TABLE_CONTRACTS[ACTIVITY_SPATIAL_TIME_BINS_TABLE].fields
        if field.arrow_type == "string"
    ]
    body: dict[str, Any] = {
        "schema_id": ACTIVITY_SPATIAL_PARQUET_POLICY_SCHEMA_ID,
        "schema_version": ACTIVITY_SPATIAL_PARQUET_POLICY_SCHEMA_VERSION,
        "part_count": 1,
        "row_group_rows": row_group_rows,
        "compression": "zstd",
        "compression_level": 3,
        "dictionary_columns": string_columns,
        "statistics": True,
        "write_policy": "one_recording_all_tracks_one_part_v1",
    }
    return {**body, "payload_sha256": canonical_json_sha256(body)}


def _footer_metadata() -> dict[bytes, bytes]:
    return {
        b"palette.export_schema_id": EXPORT_SCHEMA_ID.encode("utf-8"),
        b"palette.export_schema_version": str(EXPORT_SCHEMA_VERSION).encode("ascii"),
        b"palette.table_contract": json.dumps(
            TABLE_CONTRACTS[ACTIVITY_SPATIAL_TIME_BINS_TABLE].to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"),
    }


def _source_lineage_sha256(
    source_binding: Mapping[str, Any],
    binning_contract: Mapping[str, Any],
) -> str:
    return canonical_json_sha256(
        {
            "source_binding_sha256": source_binding["payload_sha256"],
            "binning_contract_sha256": binning_contract["payload_sha256"],
        }
    )


_DECODED_NUMPY_DTYPES: Mapping[str, np.dtype[Any]] = {
    "int16": np.dtype("<i2"),
    "int32": np.dtype("<i4"),
    "int64": np.dtype("<i8"),
    "double": np.dtype("<f8"),
    "bool": np.dtype("?"),
}


class _DecodedPayloadHasher:
    """Hash every exact decoded Arrow column without depending on IPC bytes."""

    def __init__(self) -> None:
        schema = exact_arrow_schema(
            ACTIVITY_SPATIAL_TIME_BINS_TABLE,
            metadata={},
        )
        self._types = {field.name: str(field.type) for field in schema}
        unsupported = sorted(
            set(self._types.values()) - ({"string"} | set(_DECODED_NUMPY_DTYPES))
        )
        if unsupported:  # pragma: no cover - installed contract is statically tested.
            raise ValueError(f"Unsupported decoded Arrow types: {unsupported!r}.")
        self._hashers: dict[str, Any] = {}
        for name, type_name in self._types.items():
            digest = hashlib.sha256()
            digest.update(
                canonical_json_bytes({"column_name": name, "arrow_type": type_name})
            )
            digest.update(b"\x00")
            self._hashers[name] = digest
        self.row_count = 0

    def update(self, columns: Mapping[str, Any]) -> None:
        if set(columns) != set(self._types):
            raise ValueError("Activity/spatial decoded column inventory changed.")
        lengths = {len(columns[name]) for name in self._types}
        if len(lengths) != 1:
            raise ValueError("Activity/spatial decoded columns have unequal lengths.")
        count = lengths.pop()
        for name, type_name in self._types.items():
            raw = columns[name]
            if any(value is None for value in raw):
                raise ValueError(
                    f"{name}: activity/spatial columns forbid Arrow nulls."
                )
            if type_name == "string":
                for value in raw:
                    if not isinstance(value, str):
                        raise ValueError(f"{name}: decoded string value is invalid.")
                    encoded = value.encode("utf-8")
                    self._hashers[name].update(struct.pack("<Q", len(encoded)))
                    self._hashers[name].update(encoded)
            else:
                values = np.asarray(raw, dtype=_DECODED_NUMPY_DTYPES[type_name])
                if values.shape != (count,):
                    raise ValueError(f"{name}: decoded column must be one-dimensional.")
                self._hashers[name].update(
                    np.ascontiguousarray(values).tobytes(order="C")
                )
        self.row_count += count

    def finish(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "schema_id": ACTIVITY_SPATIAL_DECODED_PAYLOAD_SCHEMA_ID,
            "schema_version": ACTIVITY_SPATIAL_DECODED_PAYLOAD_SCHEMA_VERSION,
            "row_count": self.row_count,
            "column_sha256": {
                name: self._hashers[name].hexdigest() for name in self._types
            },
        }
        return {**body, "payload_sha256": canonical_json_sha256(body)}


def _static_row_values(
    *,
    source_binding: Mapping[str, Any],
    bout_binding: Mapping[str, Any],
    binning_contract: Mapping[str, Any],
) -> dict[str, Any]:
    track = source_binding["track_source_binding"]
    return {
        "export_schema_version": EXPORT_SCHEMA_VERSION,
        "table_name": ACTIVITY_SPATIAL_TIME_BINS_TABLE,
        "recording_id": source_binding["recording_id"],
        "zarr_path": source_binding["zarr_path"],
        "source_lineage_hash": _source_lineage_sha256(
            source_binding,
            binning_contract,
        ),
        "source_track_kinematics_scope": track["scope"],
        "source_track_kinematics_run": track["run_name"],
        "source_track_kinematics_path": track["run_path"],
        "source_track_motion_manifest_schema_id": track["source_manifest_schema_id"],
        "source_track_motion_manifest_schema_version": track[
            "source_manifest_schema_version"
        ],
        "source_track_motion_manifest_sha256": track["source_manifest_sha256"],
        "source_track_binding_sha256": track["payload_sha256"],
        "source_swim_bout_run": bout_binding["run_name"],
        "source_swim_bout_path": bout_binding["run_path"],
        "source_swim_bout_schema_id": bout_binding["source_schema_id"],
        "source_swim_bout_schema_version": bout_binding["source_schema_version"],
        "source_swim_bout_manifest_sha256": bout_binding[
            "source_array_manifest_sha256"
        ],
        "source_swim_bout_binding_sha256": bout_binding["payload_sha256"],
        "source_swim_bout_candidate_id": bout_binding["candidate_id"],
        "source_swim_bout_signal_id": bout_binding["signal_id"],
        "source_speed_level": ACTIVITY_SPATIAL_SOURCE_SPEED_LEVEL,
        "source_sample_rate_hz": binning_contract["source_sample_rate_hz"],
        "requested_bin_size_s": binning_contract["requested_bin_size_s"],
        "bin_size_frames": binning_contract["bin_size_frames"],
        "effective_bin_size_s": binning_contract["effective_bin_size_s"],
        "binning_policy": binning_contract["binning_policy"],
        "position_coordinate_space": track["position_coordinate_space"],
        "position_coordinate_descriptor_sha256": track[
            "position_coordinate_descriptor_sha256"
        ],
        "physical_authority_sha256": track["physical_authority_sha256"],
    }


def _rows_to_arrow_table(rows: list[dict[str, Any]]) -> Any:
    import pyarrow as pa

    schema = exact_arrow_schema(
        ACTIVITY_SPATIAL_TIME_BINS_TABLE,
        metadata=_footer_metadata(),
    )
    return pa.Table.from_pylist(rows, schema=schema)


def _write_streaming_part(
    bound: BoundActivitySpatialSources,
    *,
    part_path: Path,
    binning_contract: Mapping[str, Any],
    row_group_rows: int,
) -> dict[str, Any]:
    """Read one global bin at a time and write one bounded exact part."""

    import pyarrow.parquet as pq

    schema = exact_arrow_schema(
        ACTIVITY_SPATIAL_TIME_BINS_TABLE,
        metadata=_footer_metadata(),
    )
    dictionary_columns = [
        field.name
        for field in ARROW_TABLE_CONTRACTS[ACTIVITY_SPATIAL_TIME_BINS_TABLE].fields
        if field.arrow_type == "string"
    ]
    part_path.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(
        part_path,
        schema,
        compression="zstd",
        compression_level=3,
        use_dictionary=dictionary_columns,
        write_statistics=True,
    )
    decoded_hasher = _DecodedPayloadHasher()
    track_source = bound.binding["track_source_binding"]
    bin_frames = int(binning_contract["bin_size_frames"])
    try:
        for track_record in track_source["tracks"]:
            track_id = int(track_record["track_id"])
            track_group = bound.track_source.run_group["tracks"][f"id_{track_id}"]
            bout_source = bound.bout_sources[track_id]
            frame_axis = np.asarray(bout_source.frame_axis, dtype=np.int64)
            source_hasher = track_export._SelectedSourcePayloadHasher(track_record)
            if frame_axis.size == 0:
                if np.asarray(bout_source.events.bouts).size:
                    raise ValueError(f"Track {track_id} has bouts but no source rows.")
                source_hasher.finish()
                continue
            first_frame = int(frame_axis[0])
            last_frame = int(frame_axis[-1])
            first_bin = first_frame // bin_frames
            last_bin = last_frame // bin_frames
            static = _static_row_values(
                source_binding=bound.binding,
                bout_binding=bout_source.binding,
                binning_contract=binning_contract,
            )
            track_rows: list[dict[str, Any]] = []
            for bin_index in range(first_bin, last_bin + 1):
                lower = bin_index * bin_frames
                upper = lower + bin_frames
                start = int(np.searchsorted(frame_axis, lower, side="left"))
                stop = int(np.searchsorted(frame_axis, upper, side="left"))
                if stop > start:
                    columns, source_frames = track_export._read_projected_window(
                        track_group,
                        track_id=track_id,
                        start=start,
                        stop=stop,
                        stride=1,
                        source_rate_hz=float(binning_contract["source_sample_rate_hz"]),
                        source_hasher=source_hasher,
                    )
                    if not np.array_equal(source_frames, frame_axis[start:stop]):
                        raise ValueError(
                            f"Track {track_id} source frame axis changed while reading."
                        )
                    positions = np.column_stack(
                        (columns["position_x_mm"], columns["position_y_mm"])
                    )
                else:
                    columns = {
                        "source_observed": np.empty(0, dtype=bool),
                        "sample_valid": np.empty(0, dtype=bool),
                        "position_finite": np.empty(0, dtype=bool),
                        "transition_valid": np.empty(0, dtype=bool),
                        "speed_mm_s": np.empty(0, dtype=np.float32),
                        "frame_path_distance_mm": np.empty(0, dtype=np.float32),
                    }
                    source_frames = np.empty(0, dtype=np.int64)
                    positions = np.empty((0, 2), dtype=np.float64)
                metric_rows = summarize_activity_spatial_track(
                    track_id=track_id,
                    source_acquisition_frame_index=source_frames,
                    source_observed=columns["source_observed"],
                    sample_valid=columns["sample_valid"],
                    position_finite=columns["position_finite"],
                    transition_valid=columns["transition_valid"],
                    positions_mm=positions,
                    filtered_speed_mm_s=columns["speed_mm_s"],
                    filtered_path_distance_mm=columns["frame_path_distance_mm"],
                    bouts=bout_source.events.bouts,
                    source_sample_rate_hz=float(
                        binning_contract["source_sample_rate_hz"]
                    ),
                    requested_bin_size_s=float(
                        binning_contract["requested_bin_size_s"]
                    ),
                    track_frame_span=(first_frame, last_frame),
                    time_bin_range=(bin_index, bin_index),
                )
                if len(metric_rows) != 1:
                    raise RuntimeError(
                        "One requested bin did not yield exactly one row."
                    )
                track_rows.append({**static, **metric_rows[0]})
            source_hasher.finish()
            table = _rows_to_arrow_table(track_rows)
            decoded_hasher.update(table.to_pydict())
            writer.write_table(table, row_group_size=row_group_rows)
    finally:
        writer.close()
    return decoded_hasher.finish()


_EXPORT_ENVELOPE_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "source_binding",
        "binning_contract",
        "decoded_payload",
        "parquet_policy",
        "payload_sha256",
    }
)


def _validate_payload_digest(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object.")
    body = dict(value)
    digest = body.pop("payload_sha256", None)
    if digest != canonical_json_sha256(body):
        raise ValueError(f"{label} payload digest is invalid.")
    return value


def _validate_source_binding_payload(value: object) -> Mapping[str, Any]:
    source = _validate_payload_digest(value, label="activity/spatial source binding")
    if set(source) != _SOURCE_BINDING_FIELDS or (
        source.get("schema_id") != ACTIVITY_SPATIAL_SOURCE_BINDING_SCHEMA_ID
        or source.get("schema_version")
        != ACTIVITY_SPATIAL_SOURCE_BINDING_SCHEMA_VERSION
    ):
        raise ValueError("Activity/spatial source-binding schema is invalid.")
    track = _validate_payload_digest(
        source.get("track_source_binding"),
        label="track-motion source binding",
    )
    track_export._validate_source_binding(track)
    if source.get("recording_id") != track.get("recording_id") or source.get(
        "zarr_path"
    ) != track.get("zarr_path"):
        raise ValueError("Activity/spatial root and track source identities differ.")
    runs = source.get("swim_bout_runs_by_track")
    if not isinstance(runs, Mapping):
        raise ValueError("Activity/spatial swim-bout source map is invalid.")
    track_records = track.get("tracks")
    if not isinstance(track_records, list):
        raise ValueError("Activity/spatial track source inventory is invalid.")
    expected_ids = {
        str(record.get("track_id"))
        for record in track_records
        if isinstance(record, Mapping) and type(record.get("track_id")) is int
    }
    if len(expected_ids) != len(track_records) or set(runs) != expected_ids:
        raise ValueError("Activity/spatial per-track source inventories differ.")
    seen_runs: set[str] = set()
    for track_text in sorted(runs, key=int):
        bout = _validate_payload_digest(
            runs[track_text],
            label=f"track {track_text} swim-bout source binding",
        )
        if set(bout) != _BOUT_BINDING_FIELDS or (
            bout.get("schema_id") != ACTIVITY_SPATIAL_SOURCE_BINDING_SCHEMA_ID
            or bout.get("schema_version")
            != ACTIVITY_SPATIAL_SOURCE_BINDING_SCHEMA_VERSION
            or bout.get("stage_id") != "swim_bouts"
            or bout.get("track_id") != int(track_text)
        ):
            raise ValueError(f"Track {track_text} swim-bout binding is invalid.")
        run_name = bout.get("run_name")
        if not isinstance(run_name, str) or not run_name or run_name in seen_runs:
            raise ValueError("Swim-bout run identities must be nonempty and unique.")
        seen_runs.add(run_name)
        if (
            bout.get("run_path") != f"analysis/swim_bout_runs/{run_name}"
            or bout.get("source_schema_id") != SWIM_BOUT_RUN_SCHEMA_ID
            or bout.get("source_schema_version") != SWIM_BOUT_RUN_SCHEMA_VERSION
            or bout.get("source_track_kinematics_run") != track.get("run_name")
            or bout.get("source_track_motion_manifest_sha256")
            != track.get("source_manifest_sha256")
            or bout.get("source_sample_rate_hz") != track.get("source_sample_rate_hz")
        ):
            raise ValueError(f"Track {track_text} swim-bout lineage is invalid.")
        for digest_name in (
            "source_array_manifest_sha256",
            "source_track_motion_manifest_sha256",
            "frame_axis_contract_sha256",
            "frame_axis_content_sha256",
            "frame_axis_array_values_sha256",
            "bout_content_sha256",
        ):
            track_export._sha256_text(
                bout.get(digest_name),
                label=f"track {track_text} {digest_name}",
            )
        selection = bout.get("selection_snapshot")
        if (
            not isinstance(selection, Mapping)
            or set(selection)
            != {
                "mode",
                "parent_latest",
                "parent_latest_complete",
                "parent_completion_epoch",
            }
            or selection.get("mode") != "explicit_per_track_run"
        ):
            raise ValueError(f"Track {track_text} selection snapshot is invalid.")
        completion = bout.get("completion_snapshot")
        if (
            not isinstance(completion, Mapping)
            or set(completion)
            != {
                "status",
                "completed_at_utc",
                "selector_eligible",
            }
            or completion.get("status") != "complete"
            or completion.get("selector_eligible") is not True
        ):
            raise ValueError(f"Track {track_text} completion snapshot is invalid.")
        first = bout.get("frame_axis_first_frame")
        last = bout.get("frame_axis_last_frame")
        if (first is None) != (last is None) or (
            first is not None
            and (
                type(first) is not int
                or type(last) is not int
                or first < 0
                or last < first
            )
        ):
            raise ValueError(f"Track {track_text} frame span is invalid.")
    return source


def _validate_export_envelope(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    envelope = manifest.get("activity_spatial_time_bins_export")
    if not isinstance(envelope, Mapping) or set(envelope) != _EXPORT_ENVELOPE_FIELDS:
        raise ValueError(
            "Activity/spatial export envelope has an unexpected field set."
        )
    body = dict(envelope)
    digest = body.pop("payload_sha256")
    if digest != canonical_json_sha256(body):
        raise ValueError("Activity/spatial export-envelope digest is invalid.")
    if (
        body.get("schema_id") != ACTIVITY_SPATIAL_EXPORT_SCHEMA_ID
        or body.get("schema_version") != ACTIVITY_SPATIAL_EXPORT_SCHEMA_VERSION
    ):
        raise ValueError("Activity/spatial export-envelope schema is invalid.")
    _validate_source_binding_payload(body.get("source_binding"))
    binning = _validate_payload_digest(
        body.get("binning_contract"),
        label="activity/spatial binning contract",
    )
    expected_binning = activity_spatial_binning_contract(
        source_sample_rate_hz=binning.get("source_sample_rate_hz"),
        requested_bin_size_s=binning.get("requested_bin_size_s"),
    )
    if dict(binning) != expected_binning:
        raise ValueError("Activity/spatial binning contract differs from policy.")
    decoded = _validate_payload_digest(
        body.get("decoded_payload"),
        label="activity/spatial decoded payload",
    )
    if (
        decoded.get("schema_id") != ACTIVITY_SPATIAL_DECODED_PAYLOAD_SCHEMA_ID
        or decoded.get("schema_version")
        != ACTIVITY_SPATIAL_DECODED_PAYLOAD_SCHEMA_VERSION
        or type(decoded.get("row_count")) is not int
        or decoded["row_count"] < 0
    ):
        raise ValueError("Activity/spatial decoded-payload declaration is invalid.")
    policy = _validate_payload_digest(
        body.get("parquet_policy"),
        label="activity/spatial Parquet policy",
    )
    expected_policy = activity_spatial_parquet_policy(
        row_group_rows=policy.get("row_group_rows")
    )
    if dict(policy) != expected_policy:
        raise ValueError(
            "Activity/spatial Parquet policy differs from installed policy."
        )
    return envelope


def _same_float(actual: Any, expected: Any) -> bool:
    if isinstance(actual, float) and isinstance(expected, float):
        return (math.isnan(actual) and math.isnan(expected)) or actual == expected
    return actual == expected


def _validate_decoded_rows(
    columns: Mapping[str, list[Any]],
    *,
    source_binding: Mapping[str, Any],
    binning: Mapping[str, Any],
) -> None:
    row_count = len(columns["track_id"])
    track_source = source_binding["track_source_binding"]
    bout_sources = source_binding["swim_bout_runs_by_track"]
    static_by_track = {
        int(track_text): _static_row_values(
            source_binding=source_binding,
            bout_binding=bout,
            binning_contract=binning,
        )
        for track_text, bout in bout_sources.items()
    }
    keys: list[tuple[Any, ...]] = []
    observed_bins: dict[int, list[int]] = {track_id: [] for track_id in static_by_track}
    for index in range(row_count):
        track_id = columns["track_id"][index]
        if type(track_id) is not int or track_id not in static_by_track:
            raise ValueError("Activity/spatial row has an unknown track identity.")
        for name, expected in static_by_track[track_id].items():
            if not _same_float(columns[name][index], expected):
                raise ValueError(
                    f"Activity/spatial row field {name!r} is inconsistent."
                )
        bin_index = columns["time_bin_index"][index]
        if type(bin_index) is not int or bin_index < 0:
            raise ValueError("Activity/spatial time-bin identity is invalid.")
        observed_bins[track_id].append(bin_index)
        key = (
            columns["recording_id"][index],
            columns["source_track_kinematics_scope"][index],
            columns["source_track_kinematics_run"][index],
            columns["source_swim_bout_run"][index],
            track_id,
            bin_index,
        )
        keys.append(key)
        bin_frames = int(binning["bin_size_frames"])
        start = bin_index * bin_frames
        end = start + bin_frames
        rate = float(binning["source_sample_rate_hz"])
        exact_values = {
            "start_acquisition_frame_index": start,
            "end_acquisition_frame_index_exclusive": end,
            "start_time_seconds": float(start) / rate,
            "end_time_seconds": float(end) / rate,
            "bin_duration_seconds": float(binning["effective_bin_size_s"]),
        }
        for name, expected in exact_values.items():
            if not _same_float(columns[name][index], expected):
                raise ValueError(f"Activity/spatial row field {name!r} is invalid.")
        expected_count = columns["expected_track_frame_count"][index]
        if type(expected_count) is not int or expected_count <= 0:
            raise ValueError("Activity/spatial expected frame count is invalid.")
        for count_name in (
            "source_sample_count",
            "source_observed_count",
            "sample_valid_count",
            "position_valid_count",
            "transition_valid_count",
            "bout_occupied_frame_count",
        ):
            count = columns[count_name][index]
            if type(count) is not int or count < 0 or count > expected_count:
                raise ValueError(f"Activity/spatial count {count_name!r} is invalid.")
        fractions = {
            "source_observed_fraction": columns["source_observed_count"][index],
            "sample_valid_fraction": columns["sample_valid_count"][index],
            "position_valid_fraction": columns["position_valid_count"][index],
            "transition_valid_fraction": columns["transition_valid_count"][index],
            "bout_occupancy_fraction": columns["bout_occupied_frame_count"][index],
        }
        for fraction_name, numerator in fractions.items():
            expected_fraction = float(numerator) / float(expected_count)
            if columns[fraction_name][index] != expected_fraction:
                raise ValueError(
                    f"Activity/spatial fraction {fraction_name!r} is invalid."
                )
        source_count = columns["source_sample_count"][index]
        position_valid = columns["position_metrics_valid"][index]
        expected_reason = (
            0
            if source_count > 0 and position_valid
            else (1 if source_count == 0 else 2)
        )
        if (
            columns["bin_valid"][index] != (expected_reason == 0)
            or columns["bin_reason_code"][index] != expected_reason
            or columns["bout_metrics_valid"][index] is not True
        ):
            raise ValueError("Activity/spatial validity/reason semantics are invalid.")
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise ValueError(
            "Activity/spatial primary keys are not strictly sorted and unique."
        )
    track_records = {
        int(record["track_id"]): record for record in track_source["tracks"]
    }
    bin_frames = int(binning["bin_size_frames"])
    for track_id, track_record in track_records.items():
        bout = bout_sources[str(track_id)]
        first = bout["frame_axis_first_frame"]
        last = bout["frame_axis_last_frame"]
        expected_bins = (
            []
            if first is None
            else list(range(first // bin_frames, last // bin_frames + 1))
        )
        if observed_bins[track_id] != expected_bins:
            raise ValueError(
                f"Track {track_id} exported time-bin coverage is incomplete."
            )
        if (first is None) != (int(track_record["sample_count"]) == 0):
            raise ValueError(f"Track {track_id} frame span/sample count disagree.")


def _decoded_part_payload(
    part_path: Path,
) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(part_path)
    hasher = _DecodedPayloadHasher()
    columns: dict[str, list[Any]] = {
        field.name: []
        for field in ARROW_TABLE_CONTRACTS[ACTIVITY_SPATIAL_TIME_BINS_TABLE].fields
    }
    for batch in parquet_file.iter_batches():
        values = batch.to_pydict()
        hasher.update(values)
        for name in columns:
            columns[name].extend(values[name])
    return hasher.finish(), columns


def validate_activity_spatial_time_bins_export_payload(
    export_root: str | Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the exact decoded table selected by one immutable manifest."""

    import pyarrow.parquet as pq

    root = Path(export_root).expanduser().resolve()
    envelope = _validate_export_envelope(manifest)
    if manifest.get("tables_requested") != [ACTIVITY_SPATIAL_TIME_BINS_TABLE]:
        raise ValueError("Activity/spatial export must select exactly its one table.")
    if manifest.get("source_zarrs") != [envelope["source_binding"]["zarr_path"]]:
        raise ValueError(
            "Activity/spatial export source path differs from its binding."
        )
    parts = manifest_selected_part_files_from_payload(
        root,
        manifest,
        ACTIVITY_SPATIAL_TIME_BINS_TABLE,
        allow_legacy_layout=False,
    )
    if len(parts) != 1:
        raise ValueError(
            "Activity/spatial export must select exactly one Parquet part."
        )
    part = parts[0]
    parquet_file = pq.ParquetFile(part)
    schema = parquet_file.schema_arrow
    validate_arrow_schema(ACTIVITY_SPATIAL_TIME_BINS_TABLE, schema)
    metadata = schema.metadata or {}
    if metadata.get(b"palette.export_schema_id") != EXPORT_SCHEMA_ID.encode("utf-8"):
        raise ValueError("Activity/spatial Parquet footer schema ID is invalid.")
    if metadata.get(b"palette.export_schema_version") != str(
        EXPORT_SCHEMA_VERSION
    ).encode("ascii"):
        raise ValueError("Activity/spatial Parquet footer schema version is invalid.")
    footer_contract = json.loads(
        metadata.get(b"palette.table_contract", b"null").decode("utf-8")
    )
    if footer_contract != TABLE_CONTRACTS[ACTIVITY_SPATIAL_TIME_BINS_TABLE].to_dict():
        raise ValueError("Activity/spatial Parquet footer contract is invalid.")
    policy = envelope["parquet_policy"]
    maximum_rows = int(policy["row_group_rows"])
    string_columns = set(policy["dictionary_columns"])
    for row_group_index in range(parquet_file.metadata.num_row_groups):
        row_group = parquet_file.metadata.row_group(row_group_index)
        if row_group.num_rows > maximum_rows:
            raise ValueError("Activity/spatial Parquet row group exceeds its bound.")
        for column_index in range(row_group.num_columns):
            column = row_group.column(column_index)
            if column.compression.upper() != "ZSTD":
                raise ValueError("Activity/spatial Parquet compression differs.")
            uses_dictionary = any(
                encoding in {"PLAIN_DICTIONARY", "RLE_DICTIONARY"}
                for encoding in column.encodings
            )
            if (column.path_in_schema in string_columns) != uses_dictionary:
                raise ValueError(
                    "Activity/spatial Parquet dictionary policy differs for "
                    f"{column.path_in_schema!r}."
                )
    observed_payload, columns = _decoded_part_payload(part)
    if observed_payload != envelope["decoded_payload"]:
        raise ValueError("Activity/spatial decoded payload differs from its receipt.")
    if manifest.get("row_counts_by_table") != {
        ACTIVITY_SPATIAL_TIME_BINS_TABLE: observed_payload["row_count"]
    }:
        raise ValueError("Activity/spatial manifest row count is invalid.")
    _validate_decoded_rows(
        columns,
        source_binding=envelope["source_binding"],
        binning=envelope["binning_contract"],
    )
    return {
        "valid": True,
        "row_count": observed_payload["row_count"],
        "decoded_payload_sha256": observed_payload["payload_sha256"],
        "source_binding_sha256": envelope["source_binding"]["payload_sha256"],
    }


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def export_activity_spatial_time_bins(
    zarr_path: str | Path,
    *,
    track_kinematics_run: str,
    track_scope: str,
    requested_bin_size_s: float,
    output_root: str | Path,
    export_run_id: str,
    scratch_root: str | Path,
    swim_bout_runs_by_track: Mapping[int, str] | None = None,
    single_track_swim_bout_run: str | None = None,
    row_group_rows: int = 65_536,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Publish one bounded, exact, selector-ineligible activity summary."""

    source_path = Path(zarr_path).expanduser().resolve()
    destination = Path(output_root).expanduser().resolve()
    scratch = Path(scratch_root).expanduser().resolve()
    if (
        destination == scratch
        or _path_is_within(destination, scratch)
        or _path_is_within(scratch, destination)
    ):
        raise ValueError("Export and scratch roots must not overlap.")
    if _path_is_within(destination, source_path) or _path_is_within(
        scratch, source_path
    ):
        raise ValueError(
            "Export and scratch roots must not be inside the source archive."
        )
    run_id = safe_component(export_run_id, label="export run ID")
    source_run = safe_component(
        track_kinematics_run,
        label="track-kinematics run ID",
    )
    if track_scope not in {"online", "offline"}:
        raise ValueError("track_scope must be 'online' or 'offline'.")
    policy = activity_spatial_parquet_policy(row_group_rows=row_group_rows)
    recording_id = track_export._recording_id(source_path)
    manifest_path = export_manifest_path(destination, run_id)
    baseline_identity = manifest_identity(manifest_path)
    if baseline_identity is not None and not overwrite:
        raise FileExistsError(f"Export manifest already exists: {manifest_path}")

    runtime = ExportRuntimePhaseRecorder()
    with runtime.measure("source_binding_before"):
        root = open_zarr_root(source_path, mode="r")
        if (swim_bout_runs_by_track is None) == (single_track_swim_bout_run is None):
            raise ValueError(
                "Provide exactly one of a per-track swim-bout map or one explicit "
                "single-track swim-bout run."
            )
        if single_track_swim_bout_run is not None:
            prebound_track = track_export._source_binding(
                root,
                zarr_path=source_path,
                recording_id=recording_id,
                run_name=source_run,
                scope=track_scope,
            )
            track_records = prebound_track.binding["tracks"]
            if len(track_records) != 1:
                raise ValueError(
                    "The workflow's single swim-bout dependency is valid only for an "
                    "exactly one-track source; use the explicit per-track run map."
                )
            resolved_bout_runs = {
                int(track_records[0]["track_id"]): single_track_swim_bout_run
            }
        else:
            assert swim_bout_runs_by_track is not None
            resolved_bout_runs = dict(swim_bout_runs_by_track)
        before = bind_activity_spatial_sources(
            root,
            zarr_path=source_path,
            recording_id=recording_id,
            track_kinematics_run=source_run,
            track_scope=track_scope,
            swim_bout_runs_by_track=resolved_bout_runs,
        )
        binning = activity_spatial_binning_contract(
            source_sample_rate_hz=float(
                before.binding["track_source_binding"]["source_sample_rate_hz"]
            ),
            requested_bin_size_s=requested_bin_size_s,
        )
    generation_id = uuid.uuid4().hex
    final_generation_path = generation_relative_path(run_id, generation_id)
    staging = publication_staging_root(destination, run_id, generation_id)
    final_generation = publication_generation_root(destination, run_id, generation_id)
    if staging.exists() or final_generation.exists():
        raise FileExistsError(
            f"Analytics export generation already exists: {generation_id}"
        )
    scratch_generation = scratch / f"palette_activity_spatial_{run_id}_{generation_id}"
    if scratch_generation.exists():
        raise FileExistsError(
            f"Activity/spatial scratch generation already exists: {scratch_generation}"
        )
    source_hash = hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()[:10]
    part_name = f"part-00000-{source_hash}.parquet"
    scratch_part = (
        scratch_generation / "tables" / ACTIVITY_SPATIAL_TIME_BINS_TABLE / part_name
    )
    try:
        with runtime.measure("scratch_parquet_write"):
            decoded_payload = _write_streaming_part(
                before,
                part_path=scratch_part,
                binning_contract=binning,
                row_group_rows=row_group_rows,
            )
        with runtime.measure("source_binding_after"):
            after_root = open_zarr_root(source_path, mode="r")
            after = bind_activity_spatial_sources(
                after_root,
                zarr_path=source_path,
                recording_id=recording_id,
                track_kinematics_run=source_run,
                track_scope=track_scope,
                swim_bout_runs_by_track=resolved_bout_runs,
            )
            if after.binding != before.binding:
                raise RuntimeError(
                    "Activity/spatial source selection, completion, or manifests "
                    "changed during extraction."
                )
        staged_part = staging / "tables" / ACTIVITY_SPATIAL_TIME_BINS_TABLE / part_name
        with runtime.measure("scratch_to_staging_copy"):
            staged_part.parent.mkdir(parents=True, exist_ok=False)
            shutil.copy2(scratch_part, staged_part)
            staged_sha256 = sha256_file(staged_part)
            if staged_sha256 != sha256_file(scratch_part):
                raise RuntimeError("Activity/spatial scratch copy digest mismatch.")
        with runtime.measure("staged_decoded_validation"):
            staged_payload, _ = _decoded_part_payload(staged_part)
            if staged_payload != decoded_payload:
                raise RuntimeError("Activity/spatial staged decoded payload differs.")

        relative_part = (
            final_generation_path
            / "tables"
            / ACTIVITY_SPATIAL_TIME_BINS_TABLE
            / part_name
        ).as_posix()
        row_count = int(decoded_payload["row_count"])
        inventory = {
            ACTIVITY_SPATIAL_TIME_BINS_TABLE: [
                {
                    "path": relative_part,
                    "sha256": staged_sha256,
                    "size_bytes": int(staged_part.stat().st_size),
                    "row_count": row_count,
                }
            ]
        }
        columns = tuple(
            field.name
            for field in ARROW_TABLE_CONTRACTS[ACTIVITY_SPATIAL_TIME_BINS_TABLE].fields
        )
        capability_statuses = resolve_capabilities(
            {ACTIVITY_SPATIAL_TIME_BINS_TABLE: columns}
        )
        envelope_body: dict[str, Any] = {
            "schema_id": ACTIVITY_SPATIAL_EXPORT_SCHEMA_ID,
            "schema_version": ACTIVITY_SPATIAL_EXPORT_SCHEMA_VERSION,
            "source_binding": before.binding,
            "binning_contract": binning,
            "decoded_payload": decoded_payload,
            "parquet_policy": policy,
        }
        export_envelope = {
            **envelope_body,
            "payload_sha256": canonical_json_sha256(envelope_body),
        }
        git = get_git_info(Path(__file__).resolve().parents[3])
        manifest: dict[str, Any] = {
            "export_run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "schema_id": EXPORT_SCHEMA_ID,
            "schema_version": EXPORT_SCHEMA_VERSION,
            "tool": "fisheye.analytics_exports.activity_spatial_time_bins",
            "hostname": socket.gethostname(),
            "palette_git_commit": git.get("commit_hash"),
            "palette_git_dirty": git.get("is_dirty"),
            "source_recording_count": 1,
            "source_zarrs": [str(source_path)],
            "tables_requested": [ACTIVITY_SPATIAL_TIME_BINS_TABLE],
            "table_contracts": contract_snapshot((ACTIVITY_SPATIAL_TIME_BINS_TABLE,)),
            "arrow_schema_contracts": arrow_contract_envelope(
                (ACTIVITY_SPATIAL_TIME_BINS_TABLE,)
            ),
            "capabilities": [
                item.capability_id for item in capability_statuses if item.available
            ],
            "capability_statuses": [item.to_dict() for item in capability_statuses],
            "row_counts_by_table": {ACTIVITY_SPATIAL_TIME_BINS_TABLE: row_count},
            "part_files_by_table": {ACTIVITY_SPATIAL_TIME_BINS_TABLE: [relative_part]},
            "publication": {
                "schema_id": PUBLICATION_SCHEMA_ID,
                "schema_version": PUBLICATION_SCHEMA_VERSION,
                "state": "complete",
                "generation_id": generation_id,
                "generation_path": final_generation_path.as_posix(),
                "parts_by_table": inventory,
            },
            "diagnostics": [],
            "collection_manifest": None,
            "export_parameters": {
                "registry_indexing": False,
                "selector_activation": False,
                "source_mutation": False,
                "scratch_root": str(scratch),
                "requested_bin_size_s": float(requested_bin_size_s),
                "overwrite": bool(overwrite),
            },
            "activity_spatial_time_bins_export": export_envelope,
        }
        with runtime.measure("manifest_validation"):
            _validate_export_envelope(manifest)
        committed = commit_staged_publication(
            destination,
            staging,
            manifest,
            baseline_manifest_identity=baseline_identity,
            runtime_recorder=runtime,
        )
        with runtime.measure("published_payload_validation"):
            published = json.loads(committed.read_text(encoding="utf-8"))
            validation = validate_activity_spatial_time_bins_export_payload(
                destination,
                published,
            )
        return {
            **published,
            "manifest_path": str(committed),
            "activity_spatial_time_bins_validation": validation,
            "runtime_telemetry": runtime.snapshot(),
        }
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    finally:
        if scratch_generation.exists():
            shutil.rmtree(scratch_generation)


__all__ = [
    "ACTIVITY_SPATIAL_BINNING_POLICY",
    "ACTIVITY_SPATIAL_BINNING_SCHEMA_ID",
    "ACTIVITY_SPATIAL_EXPORT_SCHEMA_ID",
    "ACTIVITY_SPATIAL_SOURCE_BINDING_SCHEMA_ID",
    "BoundActivitySpatialSources",
    "BoundSwimBoutSource",
    "activity_spatial_binning_contract",
    "activity_spatial_parquet_policy",
    "bind_activity_spatial_sources",
    "export_activity_spatial_time_bins",
    "summarize_activity_spatial_track",
    "validate_activity_spatial_time_bins_export_payload",
]
