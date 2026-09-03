"""Compact receipt-bound distributions over a validated-behavior cohort.

The parent export remains the lossless sample authority.  This module adds two
small observation tables that Phase C does not expose (semantic bout rows and
inter-bout intervals), then persists sparse recording histograms and
equal-recording-weight cohort summaries.  No source selector is discovered and
no recording Zarr is mutated.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from fisheye.analysis_workflows.provider_epoch_behavior_summary_source_handle import (
    load_provider_epoch_behavior_summary_source_handle,
)
from fisheye.analysis_workflows.validated_recording_behavior_source import (
    ValidatedRecordingBehaviorSource,
)
from fisheye.analytics_exports.publication import safe_component, sha256_file
from fisheye.analytics_exports.validated_behavior_dataset import (
    ValidatedBehaviorExportDataset,
    ValidatedBehaviorTable,
)
from fisheye.group_statistics.validated_behavior_appearance import (
    build_chaser_appearance_dimension,
    validate_chaser_appearance_dimension,
)
from fisheye.shared.json_safety import (
    json_attr_safe,
    strict_json_dumps,
    write_json_atomic,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

from .validated_behavior_distribution_specs import (
    SCOPE_LABELS,
    SCOPE_ORDER,
    DistributionMetricSpec,
    validate_distribution_metric_specs,
)

SCHEMA_ID = "palette.analytics.validated_behavior.distributions"
SCHEMA_VERSION = 1
METHOD_ID = "receipt_bound_sparse_recording_histograms_v1"
STATUS = "selector_ineligible_exploratory_candidate"
MAX_RESOLVED_BINS = 10_000

_EPOCH_ROLES = SCOPE_ORDER[1:]
_EVENT_SURFACES = frozenset({"bout_observations", "inter_bout_interval_observations"})
_LINEAR_MOTION_VALIDITY = "linear_sample_valid_and_transition_valid_positive_time_v1"
_ANGULAR_MOTION_VALIDITY = "angular_sample_valid_and_transition_valid_positive_time_v1"
_DISTANCE_VALIDITY = (
    "exact_occurrence_relative_physical_and_time_transition_valid_v1"
)

_EPOCH_BOUT_FIELDS = (
    "track_id",
    "window_id",
    "window_index",
    "window_label",
    "start_frame",
    "end_frame",
    "start_time_s",
    "end_time_s",
    "duration_s",
    "bout_source_row",
    "bout_id",
    "bout_event_frame",
    "bout_event_time_s",
    "bout_start_frame",
    "bout_end_frame",
    "bout_start_time_s",
    "bout_end_time_s",
    "bout_duration_s",
    "bout_path_length_mm",
    "bout_net_heading_change_deg",
    "abs_bout_net_heading_change_deg",
    "bout_heading_path_deg",
    "analysis_role",
    "source_interval_sha256",
    "protocol_semantic_hash",
    "protocol_semantic_step_index",
    "protocol_semantic_step_ref",
)
_EPOCH_IBI_HISTOGRAM_FIELDS = (
    "metric_name",
    "window_id",
    "bin_index",
    "bin_left",
    "bin_right",
    "hist_count",
    "source_sample_count",
    "finite_sample_count",
    "analysis_role",
    "source_interval_sha256",
)

_CANONICAL_BOUT_COLUMNS = (
    "recording_id",
    "membership_member_sha256",
    "bundle_set_member_sha256",
    "bundle_record_sha256",
    "source_child_key",
    "source_run_path",
    "source_manifest_sha256",
    "source_payload_sha256",
    "source_receipt_sha256",
    "swim_bout_run_path",
    "swim_bout_lineage_sha256",
    "track_id",
    "source_signal_id",
    "bout_id",
    "bout_row_id",
    "start_acquisition_frame_id",
    "end_acquisition_frame_id",
    "duration_s",
    "path_length_mm",
    "net_displacement_mm",
    "mean_speed_mm_s",
    "peak_speed_mm_s",
    "tortuosity",
)
_MOTION_COLUMNS = (
    "recording_id",
    "provider_role",
    "position_provider_id",
    "position_provider_digest",
    "source_run_path",
    "source_manifest_sha256",
    "source_verification_digest",
    "track_id",
    "track_sample_row_id",
    "acquisition_frame_id",
    "time_s",
    "linear_sample_valid",
    "angular_sample_valid",
    "transition_valid",
    "delta_frames",
    "delta_s",
    "heading_deg",
    "smoothed_heading_deg",
    "speed_filtered_mm_s",
    "speed_smoothed_mm_s",
    "frame_path_distance_smoothed_mm",
    "delta_heading_smoothed_deg",
    "angular_velocity_smoothed_deg_s",
    "angular_speed_smoothed_deg_s",
)
_DISTANCE_COLUMNS = (
    "recording_id",
    "provider_role",
    "position_provider_id",
    "position_provider_digest",
    "source_run_path",
    "source_manifest_sha256",
    "source_receipt_sha256",
    "relative_frame_row_id",
    "acquisition_frame_id",
    "timestamp_ns_session",
    "timestamp_valid",
    "timestamp_delta_ns",
    "chaser_identity_code",
    "chaser_identity",
    "behavior_role",
    "chaser_behavior_role_valid",
    "selection_member",
    "chaser_occurrence_member",
    "row_valid",
    "acquisition_frame_delta",
    "relative_distance_mm",
    "relative_physical_valid",
    "relative_transition_valid",
)


class ValidatedBehaviorDistributionError(ValueError):
    """Raised when a distribution computation or publication is unsafe."""


def _fail(message: str) -> None:
    raise ValidatedBehaviorDistributionError(message)


def _plain(value: Any) -> Any:
    return json_attr_safe(value)


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _decode(value: Any) -> Any:
    item = value.item() if isinstance(value, np.generic) else value
    if isinstance(item, bytes):
        return item.rstrip(b"\x00").decode("utf-8")
    return item


def _json_identity(record: Mapping[str, Any]) -> tuple[str, str]:
    plain = _plain(record)
    encoded = strict_json_dumps(plain)
    return encoded, canonical_json_sha256(plain)


@dataclass(frozen=True, slots=True)
class ValidatedBehaviorDistributionConfig:
    distribution_run_id: str
    metric_specs: tuple[DistributionMetricSpec, ...]
    heading_match_atol_deg: float = 1e-6

    def __post_init__(self) -> None:
        safe_component(self.distribution_run_id, label="distribution run ID")
        object.__setattr__(
            self,
            "metric_specs",
            validate_distribution_metric_specs(self.metric_specs),
        )
        tolerance = float(self.heading_match_atol_deg)
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("heading_match_atol_deg must be finite and nonnegative")

    @property
    def record(self) -> Mapping[str, object]:
        body = {
            "distribution_run_id": self.distribution_run_id,
            "method_id": METHOD_ID,
            "scope_order": list(SCOPE_ORDER),
            "scope_labels": dict(SCOPE_LABELS),
            "metric_specs": [spec.to_dict() for spec in self.metric_specs],
            "heading_match_atol_deg": float(self.heading_match_atol_deg),
            "experimental_unit": "recording_id",
            "cohort_weighting": "equal_weight_per_finite_recording",
            "pooled_observation_statistic": "diagnostic_only",
            "interpolation": "prohibited",
        }
        return MappingProxyType({**body, "config_sha256": canonical_json_sha256(body)})


@dataclass(frozen=True, slots=True)
class ValidatedBehaviorDistributionResult:
    config: ValidatedBehaviorDistributionConfig
    source_export: Mapping[str, object]
    cohort_summary: Mapping[str, object]
    source_queries: tuple[Mapping[str, object], ...]
    epoch_child_receipts: tuple[Mapping[str, object], ...]
    histogram_recipes: tuple[Mapping[str, object], ...]
    chaser_appearance_dimension: Mapping[str, object]
    bout_observations: tuple[Mapping[str, object], ...]
    inter_bout_interval_observations: tuple[Mapping[str, object], ...]
    recording_support: tuple[Mapping[str, object], ...]
    recording_nonzero_bins: tuple[Mapping[str, object], ...]
    cohort_bins: tuple[Mapping[str, object], ...]


@dataclass(slots=True)
class _AxisAudit:
    candidate_count: int = 0
    valid_count: int = 0
    minimum: float | None = None
    maximum: float | None = None

    def update(self, values: np.ndarray, *, candidate_count: int) -> None:
        self.candidate_count += int(candidate_count)
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        self.valid_count += int(finite.size)
        if finite.size:
            minimum = float(np.min(finite))
            maximum = float(np.max(finite))
            self.minimum = (
                minimum if self.minimum is None else min(self.minimum, minimum)
            )
            self.maximum = (
                maximum if self.maximum is None else max(self.maximum, maximum)
            )


@dataclass(slots=True)
class _SparseAccumulator:
    support_rows: list[dict[str, Any]] = field(default_factory=list)
    sparse_rows: list[dict[str, Any]] = field(default_factory=list)
    axis_audits: dict[str, _AxisAudit] = field(default_factory=dict)


def wrap_heading_delta_degrees(values: np.ndarray | float) -> np.ndarray:
    """Wrap signed angles to the exact ``[-180, 180]`` convention."""

    array = np.asarray(values, dtype=np.float64)
    return (array + 180.0) % 360.0 - 180.0


def derive_bout_heading_values(
    *,
    acquisition_frames: np.ndarray,
    smoothed_heading_deg: np.ndarray,
    angular_sample_valid: np.ndarray,
    bout_start_frames: np.ndarray,
    bout_end_frames: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the producer's inclusive-frame bout-heading reducer."""

    frames = np.asarray(acquisition_frames, dtype=np.int64).reshape(-1)
    headings = np.asarray(smoothed_heading_deg, dtype=np.float64).reshape(-1)
    valid = np.asarray(angular_sample_valid, dtype=bool).reshape(-1)
    starts = np.asarray(bout_start_frames, dtype=np.int64).reshape(-1)
    ends = np.asarray(bout_end_frames, dtype=np.int64).reshape(-1)
    if frames.shape != headings.shape or frames.shape != valid.shape:
        raise ValueError("Motion heading arrays do not share one row axis")
    if starts.shape != ends.shape:
        raise ValueError("Bout start/end arrays do not share one row axis")
    if frames.size and np.any(np.diff(frames) <= 0):
        raise ValueError("Acquisition frames must be strictly increasing")
    net = np.full(starts.shape, np.nan, dtype=np.float64)
    path = np.full(starts.shape, np.nan, dtype=np.float64)
    finite_valid = valid & np.isfinite(headings)
    for index, (raw_start, raw_end) in enumerate(zip(starts, ends, strict=True)):
        start = min(int(raw_start), int(raw_end))
        end = max(int(raw_start), int(raw_end))
        left = int(np.searchsorted(frames, start, side="left"))
        right = int(np.searchsorted(frames, end, side="right"))
        values = headings[left:right][finite_valid[left:right]]
        if values.size < 2:
            continue
        net[index] = float(wrap_heading_delta_degrees(values[-1] - values[0]))
        path[index] = float(np.sum(np.abs(wrap_heading_delta_degrees(np.diff(values)))))
    return net, path


def _resolve_axis(
    spec: DistributionMetricSpec,
    audit: _AxisAudit,
) -> dict[str, object]:
    if audit.valid_count <= 0 or audit.minimum is None or audit.maximum is None:
        _fail(f"{spec.metric_id}: no valid values resolve a histogram axis")
    width = float(spec.bin_width)
    if spec.coverage_policy == "fixed_closed_terminal":
        lower = float(spec.lower_bound)
        assert spec.upper_bound is not None
        upper = float(spec.upper_bound)
    elif spec.coverage_policy == "zero_anchored_cover_valid_max":
        if audit.minimum < 0:
            _fail(f"{spec.metric_id}: a nonnegative metric contains negative evidence")
        lower = 0.0
        upper = (math.floor(audit.maximum / width) + 1) * width
    elif spec.coverage_policy == "log10_cover_valid_positive_range":
        if audit.minimum <= 0:
            _fail(f"{spec.metric_id}: logarithmic evidence must be positive")
        lower_grid = math.floor(math.log10(audit.minimum) / width)
        upper_grid = math.floor(math.log10(audit.maximum) / width) + 1
        bin_count = int(upper_grid - lower_grid)
        if bin_count <= 0 or bin_count > MAX_RESOLVED_BINS:
            _fail(
                f"{spec.metric_id}: resolved {bin_count} logarithmic bins, outside "
                f"the safe range 1..{MAX_RESOLVED_BINS}"
            )
        edges = [
            10.0 ** ((lower_grid + index) * width) for index in range(bin_count + 1)
        ]
        body = {
            "metric_id": spec.metric_id,
            "metric_spec_sha256": spec.spec_sha256,
            "resolved_lower_bound": edges[0],
            "resolved_upper_bound": edges[-1],
            "bin_width": width,
            "bin_width_domain": "log10_metric_value",
            "bin_count": bin_count,
            "bin_edges": edges,
            "axis_scale": "log10",
            "grid_index_offset": -int(lower_grid),
            "terminal_bin_policy": "right_closed_only_for_final_bin",
            "open_range_resolution": "strict_log10_edges_outside_observed_range_v1",
            "source_audit": {
                "candidate_count": int(audit.candidate_count),
                "valid_count": int(audit.valid_count),
                "minimum": float(audit.minimum),
                "maximum": float(audit.maximum),
            },
        }
        return {**body, "histogram_recipe_sha256": canonical_json_sha256(body)}
    else:
        half_bins = max(
            1,
            math.floor(max(abs(audit.minimum), abs(audit.maximum)) / width) + 1,
        )
        lower = -float(half_bins) * width
        upper = float(half_bins) * width
    bin_count = int(round((upper - lower) / width))
    if bin_count <= 0 or bin_count > MAX_RESOLVED_BINS:
        _fail(
            f"{spec.metric_id}: resolved {bin_count} bins, outside the safe "
            f"range 1..{MAX_RESOLVED_BINS}"
        )
    edges = [lower + index * width for index in range(bin_count + 1)]
    body = {
        "metric_id": spec.metric_id,
        "metric_spec_sha256": spec.spec_sha256,
        "resolved_lower_bound": lower,
        "resolved_upper_bound": upper,
        "bin_width": width,
        "bin_width_domain": "metric_value",
        "bin_count": bin_count,
        "bin_edges": edges,
        "axis_scale": "linear",
        "grid_index_offset": (
            int(round(-lower / width))
            if spec.coverage_policy == "symmetric_cover_valid_abs_max"
            else 0
        ),
        "terminal_bin_policy": "right_closed_only_for_final_bin",
        "open_range_resolution": "strict_upper_edge_above_observed_max_v1",
        "source_audit": {
            "candidate_count": int(audit.candidate_count),
            "valid_count": int(audit.valid_count),
            "minimum": float(audit.minimum),
            "maximum": float(audit.maximum),
        },
    }
    return {**body, "histogram_recipe_sha256": canonical_json_sha256(body)}


def _bin_indices(values: np.ndarray, recipe: Mapping[str, object]) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    lower = float(recipe["resolved_lower_bound"])
    upper = float(recipe["resolved_upper_bound"])
    width = float(recipe["bin_width"])
    count = int(recipe["bin_count"])
    if np.any(~np.isfinite(data)) or np.any(data < lower) or np.any(data > upper):
        raise ValueError("Histogram values lie outside their resolved axis")
    if recipe.get("axis_scale") == "log10":
        if np.any(data <= 0):
            raise ValueError("Logarithmic histogram values must be positive")
        grid = np.floor(np.log10(data) / width).astype(np.int64)
        indices = grid + int(recipe["grid_index_offset"])
    else:
        indices = np.floor((data - lower) / width).astype(np.int64)
    indices[data == upper] = count - 1
    if np.any(indices < 0) or np.any(indices >= count):
        raise ValueError("Histogram index escaped its resolved axis")
    return indices


def _scope_masks(
    frames: np.ndarray,
    epochs: Sequence[Mapping[str, Any]],
) -> Mapping[str, np.ndarray]:
    values = np.asarray(frames, dtype=np.int64).reshape(-1)
    result: dict[str, np.ndarray] = {"whole_session": np.ones(values.shape, dtype=bool)}
    covered = np.zeros(values.shape, dtype=np.int8)
    by_role = {str(row["analysis_role"]): row for row in epochs}
    if set(by_role) != set(_EPOCH_ROLES) or len(by_role) != len(epochs):
        raise ValueError(
            "Each recording must expose exactly one pre/training/post epoch"
        )
    for role in _EPOCH_ROLES:
        row = by_role[role]
        mask = (values >= int(row["start_frame"])) & (
            values < int(row["end_frame_exclusive"])
        )
        result[role] = mask
        covered += mask.astype(np.int8)
    if np.any(covered > 1):
        raise ValueError("Semantic epoch frame intervals overlap")
    return MappingProxyType(result)


def _transition_scope_masks(
    frames: np.ndarray,
    delta_frames: np.ndarray,
    epochs: Sequence[Mapping[str, Any]],
) -> Mapping[str, np.ndarray]:
    """Require both endpoints of a weighted transition inside an epoch."""

    current = _scope_masks(frames, epochs)
    previous_frames = np.asarray(frames, dtype=np.int64).reshape(-1) - np.asarray(
        delta_frames, dtype=np.int64
    ).reshape(-1)
    if previous_frames.shape != np.asarray(frames).shape:
        raise ValueError("Transition frame deltas do not align with the frame axis")
    previous = _scope_masks(previous_frames, epochs)
    return MappingProxyType(
        {
            "whole_session": current["whole_session"],
            **{
                role: current[role] & previous[role]
                for role in _EPOCH_ROLES
            },
        }
    )


def _part_paths_for_member(
    table: ValidatedBehaviorTable,
    *,
    ordinal: int,
    recording_id: str,
) -> tuple[Path, ...]:
    member = f"member={int(ordinal):06d}-{recording_id}"
    paths = tuple(path for path in table.part_paths if path.parent.name == member)
    if not paths:
        _fail(f"{table.name}: no manifest-selected part for {member}")
    return paths


def _read_member_table(
    table: ValidatedBehaviorTable,
    *,
    ordinal: int,
    recording_id: str,
    columns: Sequence[str],
) -> Any:
    import polars as pl

    paths = _part_paths_for_member(table, ordinal=ordinal, recording_id=recording_id)
    frame = pl.read_parquet([str(path) for path in paths], columns=list(columns))
    if frame.height and frame.get_column("recording_id").unique().to_list() != [
        recording_id
    ]:
        _fail(f"{table.name}: selected member part contains another recording")
    return frame


def _array_rows(
    arrays: Mapping[str, np.ndarray], fields: Sequence[str]
) -> list[dict[str, Any]]:
    lengths = {int(np.asarray(arrays[name]).shape[0]) for name in fields}
    if len(lengths) != 1:
        raise ValueError("Targeted columnar arrays do not share one row axis")
    count = lengths.pop()
    return [
        {name: _decode(np.asarray(arrays[name])[index]) for name in fields}
        for index in range(count)
    ]


def _open_member_source(
    member: Mapping[str, Any],
) -> ValidatedRecordingBehaviorSource:
    if member.get("bundle_state") != "complete":
        _fail("Scientific distribution extraction requires a complete bundle member")
    bundle = member.get("bundle")
    if not isinstance(bundle, Mapping):
        _fail("Complete bundle-set member lacks its exact bundle binding")
    path = Path(str(bundle.get("path"))).expanduser().resolve()
    if not path.is_file() or sha256_file(path) != bundle.get("file_sha256"):
        _fail("Recording bundle file is absent or differs from the bundle set")
    source = ValidatedRecordingBehaviorSource(
        path,
        expected_analysis_zarr=member["analysis_zarr"],
        expected_recording_id=str(member["recording_id"]),
        validate_current_sources=False,
    )
    if source.bundle_sha256 != bundle.get("record_sha256"):
        _fail("Recording bundle record differs from its bundle-set member")
    return source


def _load_epoch_child_arrays(
    source: ValidatedRecordingBehaviorSource,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    capability = source.scientific_child("epoch_behavior")
    binding = capability.binding
    run_path = str(binding["run_path"])
    paths = tuple(
        [f"per_epoch_bouts/{name}" for name in _EPOCH_BOUT_FIELDS]
        + [
            f"per_epoch_inter_bout_interval_histograms/{name}"
            for name in _EPOCH_IBI_HISTOGRAM_FIELDS
        ]
    )
    handle = load_provider_epoch_behavior_summary_source_handle(
        source.analysis_zarr,
        run_name=run_path.rsplit("/", 1)[-1],
        expected_recording_id=source.recording_id,
        direct_validation_receipt=binding["receipt_path"],
        required_array_paths=paths,
    )
    observed = {
        "run_path": handle.run_path,
        "manifest_sha256": handle.manifest_sha256,
        "payload_digest": handle.payload_digest,
        "receipt_sha256": handle.receipt_digest,
    }
    expected = {
        "run_path": run_path,
        "manifest_sha256": binding["manifest_sha256"],
        "payload_digest": binding["payload_digest"],
        "receipt_sha256": binding["receipt_sha256"],
    }
    if observed != expected:
        _fail("Targeted epoch child differs from its recording bundle binding")
    handle.require_verified_arrays(paths)
    arrays = {path: handle.array(path) for path in paths}
    receipt_record = {
        "recording_id": source.recording_id,
        "bundle_record_sha256": source.bundle_sha256,
        "run_path": run_path,
        "manifest_sha256": handle.manifest_sha256,
        "payload_digest": handle.payload_digest,
        "receipt_path": str(binding["receipt_path"]),
        "receipt_sha256": handle.receipt_digest,
        "verification_mode": handle.verification_mode,
        "verified_array_paths": list(handle.verified_array_paths),
    }
    receipt_record["binding_sha256"] = canonical_json_sha256(receipt_record)
    return arrays, receipt_record


def _fps_from_epoch_rows(rows: Sequence[Mapping[str, Any]]) -> float:
    rates: list[float] = []
    for row in rows:
        duration = float(row["duration_s"])
        frames = int(row["total_span_frames"])
        if not math.isfinite(duration) or duration <= 0 or frames <= 0:
            _fail("Epoch summary cannot resolve exact positive FPS")
        rate = float(frames) / duration
        if not math.isfinite(rate) or rate <= 0:
            _fail("Epoch summary resolved an invalid FPS")
        rates.append(rate)
    if not rates:
        _fail("Recording lacks exact epoch-summary rows")
    fps = float(np.median(np.asarray(rates, dtype=np.float64)))
    if not np.allclose(rates, fps, rtol=1e-9, atol=1e-9):
        _fail("Epoch rows disagree on the exact recording FPS")
    return fps


def _load_bound_swim_bout_tables(source: ValidatedRecordingBehaviorSource) -> Any:
    """Open the exact canonical source named by the validated bundle."""

    try:
        return source.canonical_swim_bout_tables()
    except ValueError as exc:
        raise ValidatedBehaviorDistributionError(str(exc)) from exc


def _materialize_bound_intervals(
    *,
    canonical_rows: Sequence[Mapping[str, Any]],
    tables: Any,
    epochs: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], float]:
    """Project producer-authored interval values after exact bout-axis checks."""

    fps = float(tables.run_attrs["fps"])
    ordered_bouts = sorted(
        canonical_rows,
        key=lambda row: (
            int(row["start_acquisition_frame_id"]),
            int(row["end_acquisition_frame_id"]),
            int(row["bout_row_id"]),
        ),
    )
    intervals = np.asarray(tables.inter_bout_intervals)
    if intervals.shape != (max(0, len(ordered_bouts) - 1),):
        _fail("Canonical bout and producer interval axes do not close")
    epoch_rows = tuple(epochs)
    output: list[dict[str, Any]] = []
    for index, raw in enumerate(intervals):
        previous = ordered_bouts[index]
        following = ordered_bouts[index + 1]
        previous_end = int(previous["end_acquisition_frame_id"])
        next_start = int(following["start_acquisition_frame_id"])
        gap_frames = max(0, next_start - previous_end)
        if (
            int(raw["interval_id"]) != index
            or int(raw["prev_bout_id"]) != int(previous["bout_id"])
            or int(raw["next_bout_id"]) != int(following["bout_id"])
            or int(raw["prev_end_frame"]) != previous_end
            or int(raw["next_start_frame"]) != next_start
            or int(raw["interval_frames"]) != gap_frames
        ):
            _fail("Producer interval row differs from the canonical bout axis")
        interval_s = float(raw["interval_s"])
        interval_valid = bool(raw["valid"])
        if interval_valid and (
            not math.isfinite(interval_s)
            or interval_s < 0
            or not math.isclose(
                interval_s,
                float(gap_frames) / fps,
                rel_tol=1e-5,
                abs_tol=1e-8,
            )
        ):
            _fail("Producer interval duration violates its frame/FPS convention")
        matches = [
            epoch
            for epoch in epoch_rows
            if previous_end >= int(epoch["start_frame"])
            and next_start < int(epoch["end_frame_exclusive"])
        ]
        if len(matches) > 1:
            _fail("One producer interval belongs to multiple semantic epochs")
        epoch = matches[0] if matches else None
        output.append(
            {
                "interval_row_id": index,
                "previous_bout_row_id": int(previous["bout_row_id"]),
                "previous_bout_id": int(previous["bout_id"]),
                "next_bout_row_id": int(following["bout_row_id"]),
                "next_bout_id": int(following["bout_id"]),
                "previous_end_frame": previous_end,
                "next_start_frame": next_start,
                "interval_frames": gap_frames,
                "previous_end_time_s": float(raw["prev_end_time_s"]),
                "next_start_time_s": float(raw["next_start_time_s"]),
                "interval_s": interval_s,
                "interval_valid": interval_valid,
                "epoch_window_id": (
                    None if epoch is None else int(epoch["epoch_window_id"])
                ),
                "analysis_role": (
                    None if epoch is None else str(epoch["analysis_role"])
                ),
                "source_interval_sha256": (
                    None if epoch is None else str(epoch["source_interval_sha256"])
                ),
                "epoch_membership_state": (
                    "cross_epoch_or_outside" if epoch is None else "exact_member"
                ),
            }
        )
    return output, fps


def _validate_motion_fps(frame: Any, *, fps: float) -> None:
    delta_frames = frame.get_column("delta_frames").to_numpy().astype(np.float64)
    delta_s = frame.get_column("delta_s").to_numpy().astype(np.float64)
    valid = (
        frame.get_column("transition_valid").to_numpy().astype(bool)
        & np.isfinite(delta_s)
        & (delta_s > 0)
        & (delta_frames > 0)
    )
    if not np.any(valid):
        _fail("Provider motion has no valid transition from which to verify FPS")
    observed = delta_frames[valid] / delta_s[valid]
    if not np.allclose(observed, fps, rtol=5e-5, atol=5e-5):
        _fail("Provider-motion elapsed time disagrees with epoch-summary FPS")


def _epochs_from_frames(
    semantic_rows: Sequence[Mapping[str, Any]],
    epoch_summary_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    by_id = {int(row["epoch_window_id"]): row for row in epoch_summary_rows}
    if len(by_id) != len(epoch_summary_rows):
        _fail("Epoch-summary window identities are duplicated")
    result: list[dict[str, Any]] = []
    for row in semantic_rows:
        window_id = int(row["epoch_window_id"])
        try:
            summary = by_id[window_id]
        except KeyError as exc:
            raise ValidatedBehaviorDistributionError(
                "Semantic epoch lacks its exact behavior-summary row"
            ) from exc
        expected_end = int(row["end_frame_exclusive"]) - 1
        if (
            str(row["analysis_role"]) != str(summary["analysis_role"])
            or int(row["start_frame"]) != int(summary["start_frame"])
            or expected_end != int(summary["end_frame"])
            or str(row["source_interval_sha256"])
            != str(summary["source_interval_sha256"])
        ):
            _fail("Semantic-epoch and epoch-summary intervals disagree")
        result.append(
            {
                "epoch_window_id": window_id,
                "analysis_role": str(row["analysis_role"]),
                "start_frame": int(row["start_frame"]),
                "end_frame_exclusive": int(row["end_frame_exclusive"]),
                "source_interval_sha256": str(row["source_interval_sha256"]),
                "protocol_semantic_hash": str(row["protocol_semantic_hash"]),
                "protocol_semantic_step_index": int(
                    row["protocol_semantic_step_index"]
                ),
                "protocol_semantic_step_ref": str(row["protocol_semantic_step_ref"]),
                "valid_tracked_duration_s": float(summary["valid_tracked_duration_s"]),
            }
        )
    _scope_masks(np.zeros(0, dtype=np.int64), result)
    ordered = tuple(sorted(result, key=lambda row: row["start_frame"]))
    for previous, following in zip(ordered[:-1], ordered[1:], strict=True):
        if previous["end_frame_exclusive"] > following["start_frame"]:
            _fail("Semantic epoch intervals overlap")
    return ordered


def _float_matches(left: Any, right: Any, *, atol: float = 1e-8) -> bool:
    a = float(left)
    b = float(right)
    if math.isnan(a) and math.isnan(b):
        return True
    return (
        math.isfinite(a)
        and math.isfinite(b)
        and math.isclose(a, b, rel_tol=1e-7, abs_tol=atol)
    )


def _materialize_bout_observations(
    *,
    config: ValidatedBehaviorDistributionConfig,
    source_export_run_id: str,
    source_export_manifest_sha256: str,
    recording_id: str,
    canonical_frame: Any,
    motion_frame: Any,
    epochs: Sequence[Mapping[str, Any]],
    epoch_arrays: Mapping[str, np.ndarray],
    epoch_binding: Mapping[str, Any],
) -> list[dict[str, Any]]:
    canonical = canonical_frame.sort("bout_row_id").to_dicts()
    if not canonical:
        return []
    row_ids = [int(row["bout_row_id"]) for row in canonical]
    if row_ids != list(range(len(row_ids))):
        _fail(f"{recording_id}: canonical bout-row IDs are not gapless")
    if len({int(row["track_id"]) for row in canonical}) != 1:
        _fail(f"{recording_id}: canonical bouts span multiple tracks")

    frames = motion_frame.get_column("acquisition_frame_id").to_numpy()
    headings = motion_frame.get_column("smoothed_heading_deg").to_numpy()
    angular_valid = motion_frame.get_column("angular_sample_valid").to_numpy()
    starts = np.asarray(
        [int(row["start_acquisition_frame_id"]) for row in canonical],
        dtype=np.int64,
    )
    ends = np.asarray(
        [int(row["end_acquisition_frame_id"]) for row in canonical],
        dtype=np.int64,
    )
    net, heading_path = derive_bout_heading_values(
        acquisition_frames=frames,
        smoothed_heading_deg=headings,
        angular_sample_valid=angular_valid,
        bout_start_frames=starts,
        bout_end_frames=ends,
    )

    epoch_bout_arrays = {
        name: epoch_arrays[f"per_epoch_bouts/{name}"] for name in _EPOCH_BOUT_FIELDS
    }
    epoch_bouts = _array_rows(epoch_bout_arrays, _EPOCH_BOUT_FIELDS)
    epoch_by_id = {int(row["epoch_window_id"]): row for row in epochs}
    row_id_set = set(row_ids)
    mapped: dict[int, Mapping[str, Any]] = {}
    for row in epoch_bouts:
        source_row = int(row["bout_source_row"])
        if source_row in mapped or source_row not in row_id_set:
            _fail(f"{recording_id}: epoch bout mapping is duplicated or out of range")
        try:
            epoch = epoch_by_id[int(row["window_id"])]
        except KeyError as exc:
            raise ValidatedBehaviorDistributionError(
                f"{recording_id}: epoch bout references an absent semantic window"
            ) from exc
        if (
            str(row["analysis_role"]) != epoch["analysis_role"]
            or str(row["source_interval_sha256"]) != epoch["source_interval_sha256"]
            or int(row["start_frame"]) != epoch["start_frame"]
            or int(row["end_frame"]) != epoch["end_frame_exclusive"] - 1
        ):
            _fail(f"{recording_id}: epoch bout semantic identity is stale")
        mapped[source_row] = row

    output: list[dict[str, Any]] = []
    for index, canonical_row in enumerate(canonical):
        epoch_row = mapped.get(index)
        if epoch_row is not None:
            comparisons = (
                int(epoch_row["track_id"]) == int(canonical_row["track_id"]),
                int(epoch_row["bout_id"]) == int(canonical_row["bout_id"]),
                int(epoch_row["bout_start_frame"])
                == int(canonical_row["start_acquisition_frame_id"]),
                int(epoch_row["bout_end_frame"])
                == int(canonical_row["end_acquisition_frame_id"]),
                _float_matches(
                    epoch_row["bout_duration_s"], canonical_row["duration_s"]
                ),
                _float_matches(
                    epoch_row["bout_path_length_mm"], canonical_row["path_length_mm"]
                ),
            )
            if not all(comparisons):
                _fail(f"{recording_id}: canonical and epoch bout facts disagree")
            for calculated, persisted, label in (
                (net[index], epoch_row["bout_net_heading_change_deg"], "net heading"),
                (
                    abs(net[index]) if math.isfinite(net[index]) else np.nan,
                    epoch_row["abs_bout_net_heading_change_deg"],
                    "absolute net heading",
                ),
                (
                    heading_path[index],
                    epoch_row["bout_heading_path_deg"],
                    "heading path",
                ),
            ):
                if not _float_matches(
                    calculated,
                    persisted,
                    atol=float(config.heading_match_atol_deg),
                ):
                    _fail(
                        f"{recording_id}: derived {label} differs from epoch evidence"
                    )

        heading_valid = bool(
            math.isfinite(net[index]) and math.isfinite(heading_path[index])
        )
        output.append(
            {
                "distribution_run_id": config.distribution_run_id,
                "source_export_run_id": source_export_run_id,
                "source_export_manifest_sha256": source_export_manifest_sha256,
                "recording_id": recording_id,
                "membership_member_sha256": str(
                    canonical_row["membership_member_sha256"]
                ),
                "bundle_set_member_sha256": str(
                    canonical_row["bundle_set_member_sha256"]
                ),
                "bundle_record_sha256": str(canonical_row["bundle_record_sha256"]),
                "canonical_source_run_path": str(canonical_row["source_run_path"]),
                "canonical_source_manifest_sha256": str(
                    canonical_row["source_manifest_sha256"]
                ),
                "canonical_source_payload_sha256": str(
                    canonical_row["source_payload_sha256"]
                ),
                "epoch_source_run_path": str(epoch_binding["run_path"]),
                "epoch_source_manifest_sha256": str(epoch_binding["manifest_sha256"]),
                "epoch_source_payload_sha256": str(epoch_binding["payload_digest"]),
                "epoch_source_receipt_sha256": str(epoch_binding["receipt_sha256"]),
                "track_id": int(canonical_row["track_id"]),
                "source_signal_id": int(canonical_row["source_signal_id"]),
                "bout_row_id": index,
                "bout_id": int(canonical_row["bout_id"]),
                "start_acquisition_frame_id": int(
                    canonical_row["start_acquisition_frame_id"]
                ),
                "end_acquisition_frame_id": int(
                    canonical_row["end_acquisition_frame_id"]
                ),
                "bout_event_frame": (
                    None if epoch_row is None else int(epoch_row["bout_event_frame"])
                ),
                "bout_event_time_s": (
                    None
                    if epoch_row is None
                    else _finite_or_none(epoch_row["bout_event_time_s"])
                ),
                "epoch_window_id": (
                    None if epoch_row is None else int(epoch_row["window_id"])
                ),
                "analysis_role": (
                    None if epoch_row is None else str(epoch_row["analysis_role"])
                ),
                "source_interval_sha256": (
                    None
                    if epoch_row is None
                    else str(epoch_row["source_interval_sha256"])
                ),
                "protocol_semantic_hash": (
                    None
                    if epoch_row is None
                    else str(epoch_row["protocol_semantic_hash"])
                ),
                "protocol_semantic_step_index": (
                    None
                    if epoch_row is None
                    else int(epoch_row["protocol_semantic_step_index"])
                ),
                "protocol_semantic_step_ref": (
                    None
                    if epoch_row is None
                    else str(epoch_row["protocol_semantic_step_ref"])
                ),
                "epoch_membership_state": (
                    "unassigned_whole_session_evidence"
                    if epoch_row is None
                    else "exact_member"
                ),
                "duration_s": _finite_or_none(canonical_row["duration_s"]),
                "path_length_mm": _finite_or_none(canonical_row["path_length_mm"]),
                "net_displacement_mm": _finite_or_none(
                    canonical_row["net_displacement_mm"]
                ),
                "mean_speed_mm_s": _finite_or_none(canonical_row["mean_speed_mm_s"]),
                "peak_speed_mm_s": _finite_or_none(canonical_row["peak_speed_mm_s"]),
                "tortuosity": _finite_or_none(canonical_row["tortuosity"]),
                "net_heading_change_deg": _finite_or_none(net[index]),
                "abs_net_heading_change_deg": _finite_or_none(abs(net[index])),
                "heading_path_deg": _finite_or_none(heading_path[index]),
                "heading_valid": heading_valid,
                "heading_derivation_id": "inclusive_angular_valid_smoothed_heading_v1",
                "epoch_heading_crosscheck_state": (
                    "not_epoch_member"
                    if epoch_row is None
                    else "matches_persisted_epoch_child"
                ),
            }
        )
    return output


def _validate_epoch_ibi_histograms(
    *,
    recording_id: str,
    intervals: Sequence[Mapping[str, Any]],
    epochs: Sequence[Mapping[str, Any]],
    epoch_arrays: Mapping[str, np.ndarray],
) -> None:
    arrays = {
        name: epoch_arrays[f"per_epoch_inter_bout_interval_histograms/{name}"]
        for name in _EPOCH_IBI_HISTOGRAM_FIELDS
    }
    rows = [
        row
        for row in _array_rows(arrays, _EPOCH_IBI_HISTOGRAM_FIELDS)
        if str(row["metric_name"]) == "inter_bout_interval_s"
    ]
    if not rows:
        _fail(f"{recording_id}: epoch child lacks its IBI histogram")
    by_epoch = {int(row["epoch_window_id"]): row for row in epochs}
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["window_id"])].append(row)
    if set(grouped) != set(by_epoch):
        _fail(f"{recording_id}: IBI histogram does not close the semantic epoch axis")
    for window_id, histogram in grouped.items():
        epoch = by_epoch[window_id]
        ordered = sorted(histogram, key=lambda row: int(row["bin_index"]))
        if [int(row["bin_index"]) for row in ordered] != list(range(len(ordered))):
            _fail(f"{recording_id}: IBI histogram bins are not gapless")
        if any(
            str(row["analysis_role"]) != epoch["analysis_role"]
            or str(row["source_interval_sha256"]) != epoch["source_interval_sha256"]
            for row in ordered
        ):
            _fail(f"{recording_id}: IBI histogram semantic identity is stale")
        values = np.asarray(
            [
                float(row["interval_s"])
                for row in intervals
                if row["epoch_window_id"] == window_id
                and bool(row.get("interval_valid", True))
            ],
            dtype=np.float64,
        )
        source_counts = {int(row["source_sample_count"]) for row in ordered}
        finite_counts = {int(row["finite_sample_count"]) for row in ordered}
        if source_counts != {values.size} or finite_counts != {values.size}:
            _fail(
                f"{recording_id}: IBI histogram denominator differs from derived gaps"
            )
        edges = np.asarray(
            [float(row["bin_left"]) for row in ordered]
            + [float(ordered[-1]["bin_right"])],
            dtype=np.float64,
        )
        if np.any(np.diff(edges) <= 0) or any(
            not math.isclose(
                float(left["bin_right"]),
                float(right["bin_left"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for left, right in zip(ordered[:-1], ordered[1:], strict=True)
        ):
            _fail(f"{recording_id}: IBI histogram edges are discontinuous")
        observed, _ = np.histogram(values, bins=edges)
        expected = np.asarray(
            [int(row["hist_count"]) for row in ordered], dtype=np.int64
        )
        if not np.array_equal(observed, expected):
            _fail(f"{recording_id}: derived IBI values differ from epoch evidence")


def _event_scope_masks(analysis_roles: Sequence[Any]) -> Mapping[str, np.ndarray]:
    roles = np.asarray(analysis_roles, dtype=object).reshape(-1)
    return MappingProxyType(
        {
            "whole_session": np.ones(roles.shape, dtype=bool),
            **{role: roles == role for role in _EPOCH_ROLES},
        }
    )


def _grid_indices(values: np.ndarray, spec: DistributionMetricSpec) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    if np.any(~np.isfinite(data)):
        raise ValueError("Only finite values may be quantized")
    if spec.coverage_policy == "fixed_closed_terminal":
        assert spec.upper_bound is not None
        lower = float(spec.lower_bound)
        upper = float(spec.upper_bound)
        if np.any(data < lower) or np.any(data > upper):
            _fail(f"{spec.metric_id}: declared-valid values escape the fixed domain")
        count = int(round((upper - lower) / float(spec.bin_width)))
        result = np.floor((data - lower) / float(spec.bin_width)).astype(np.int64)
        result[data == upper] = count - 1
        return result
    if spec.coverage_policy == "zero_anchored_cover_valid_max":
        if np.any(data < 0):
            _fail(f"{spec.metric_id}: declared-valid values violate nonnegativity")
    if spec.coverage_policy == "log10_cover_valid_positive_range":
        if np.any(data <= 0):
            _fail(f"{spec.metric_id}: declared-valid values violate positivity")
        return np.floor(np.log10(data) / float(spec.bin_width)).astype(np.int64)
    return np.floor(data / float(spec.bin_width)).astype(np.int64)


def _group_roster(
    group_arrays: Mapping[str, np.ndarray],
    *,
    row_count: int,
) -> tuple[tuple[tuple[Any, ...], np.ndarray], ...]:
    if not group_arrays:
        return (((), np.ones(row_count, dtype=bool)),)
    names = tuple(group_arrays)
    arrays = tuple(
        np.asarray(group_arrays[name], dtype=object).reshape(-1) for name in names
    )
    if any(array.shape != (row_count,) for array in arrays):
        raise ValueError("Distribution group arrays do not share the value row axis")
    if any(any(value is None for value in array) for array in arrays):
        _fail("Distribution group dimensions contain null evidence")
    keys = sorted(
        set(zip(*arrays, strict=True)), key=lambda value: tuple(map(str, value))
    )
    return tuple(
        (
            tuple(_decode(value) for value in key),
            np.logical_and.reduce(
                [array == value for array, value in zip(arrays, key, strict=True)]
            ),
        )
        for key in keys
    )


def _source_identity_for_group(
    *,
    identity_arrays: Mapping[str, np.ndarray],
    group_mask: np.ndarray,
) -> Mapping[str, Any]:
    result: dict[str, Any] = {}
    for name, raw in identity_arrays.items():
        values = np.asarray(raw, dtype=object).reshape(-1)[group_mask]
        unique = {_decode(value) for value in values}
        if len(unique) != 1:
            _fail(f"Distribution group does not map to one exact {name}")
        result[name] = unique.pop()
    return MappingProxyType(result)


def _reduce_metric_values(
    accumulator: _SparseAccumulator,
    *,
    config: ValidatedBehaviorDistributionConfig,
    spec: DistributionMetricSpec,
    source_export_run_id: str,
    source_export_manifest_sha256: str,
    recording_id: str,
    values: np.ndarray,
    scope_masks: Mapping[str, np.ndarray],
    base_valid: np.ndarray,
    group_arrays: Mapping[str, np.ndarray],
    identity_arrays: Mapping[str, np.ndarray],
    time_weights_s: np.ndarray | None,
    valid_duration_by_scope: Mapping[str, float],
    time_scope_masks: Mapping[str, np.ndarray] | None = None,
) -> None:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    valid = np.asarray(base_valid, dtype=bool).reshape(-1) & np.isfinite(data)
    if data.shape != valid.shape or any(
        np.asarray(mask).shape != data.shape for mask in scope_masks.values()
    ):
        raise ValueError("Distribution values, validity, and scopes do not align")
    if tuple(scope_masks) != SCOPE_ORDER:
        raise ValueError("Distribution scopes do not follow the exact registry")
    if spec.coverage_policy == "zero_anchored_cover_valid_max" and np.any(
        data[valid] < 0
    ):
        _fail(f"{spec.metric_id}: declared-valid evidence violates nonnegativity")
    if spec.coverage_policy == "log10_cover_valid_positive_range" and np.any(
        data[valid] <= 0
    ):
        _fail(f"{spec.metric_id}: declared-valid evidence violates positivity")
    if spec.coverage_policy == "fixed_closed_terminal":
        assert spec.upper_bound is not None
        if np.any(data[valid] < spec.lower_bound) or np.any(
            data[valid] > spec.upper_bound
        ):
            _fail(f"{spec.metric_id}: declared-valid evidence escapes its fixed range")
    audit = accumulator.axis_audits.setdefault(spec.metric_id, _AxisAudit())
    audit.update(data[valid], candidate_count=data.size)

    groups = _group_roster(group_arrays, row_count=data.size)
    group_names = tuple(group_arrays)
    for group_values, group_mask in groups:
        group = {
            name: _decode(value)
            for name, value in zip(group_names, group_values, strict=True)
        }
        group_json, group_sha256 = _json_identity(group)
        identity = _source_identity_for_group(
            identity_arrays=identity_arrays,
            group_mask=group_mask,
        )
        identity_json, identity_sha256 = _json_identity(identity)
        for scope_id in SCOPE_ORDER:
            for weighting_id in spec.weighting_ids:
                selected_scope_masks = (
                    time_scope_masks
                    if weighting_id == "time" and time_scope_masks is not None
                    else scope_masks
                )
                scope_mask = (
                    np.asarray(selected_scope_masks[scope_id], dtype=bool) & group_mask
                )
                candidate_count = int(np.count_nonzero(scope_mask))
                base = scope_mask & valid
                if weighting_id == "time":
                    if time_weights_s is None:
                        raise ValueError(
                            f"{spec.metric_id}: time weighting lacks elapsed time"
                        )
                    weights = np.asarray(time_weights_s, dtype=np.float64).reshape(-1)
                    if weights.shape != data.shape:
                        raise ValueError("Distribution time weights do not align")
                    selected = base & np.isfinite(weights) & (weights > 0)
                    selected_weights = weights[selected]
                    weight_unit = "s"
                else:
                    selected = base
                    selected_weights = np.ones(
                        int(np.count_nonzero(selected)), dtype=np.float64
                    )
                    weight_unit = "count"
                selected_values = data[selected]
                valid_count = int(selected_values.size)
                denominator_weight = float(np.sum(selected_weights))
                valid_duration = float(valid_duration_by_scope[scope_id])
                rate = (
                    float(valid_count) / (valid_duration / 60.0)
                    if weighting_id == "event" and valid_duration > 0
                    else None
                )
                support_key = canonical_json_sha256(
                    {
                        "metric_id": spec.metric_id,
                        "recording_id": recording_id,
                        "scope_id": scope_id,
                        "group_key_sha256": group_sha256,
                        "weighting_id": weighting_id,
                    }
                )
                accumulator.support_rows.append(
                    {
                        "distribution_run_id": config.distribution_run_id,
                        "source_export_run_id": source_export_run_id,
                        "source_export_manifest_sha256": source_export_manifest_sha256,
                        "metric_id": spec.metric_id,
                        "metric_spec_sha256": spec.spec_sha256,
                        "metric_family": spec.metric_family,
                        "source_surface": spec.source_surface,
                        "recording_id": recording_id,
                        "scope_id": scope_id,
                        "group_key_json": group_json,
                        "group_key_sha256": group_sha256,
                        "source_identity_key_json": identity_json,
                        "source_identity_key_sha256": identity_sha256,
                        "weighting_id": weighting_id,
                        "weight_unit": weight_unit,
                        "candidate_count": candidate_count,
                        "valid_count": valid_count,
                        "excluded_count": candidate_count - valid_count,
                        "denominator_weight": denominator_weight,
                        "valid_duration_s": valid_duration,
                        "event_rate_per_valid_min": rate,
                        "minimum_value": (
                            None if not valid_count else float(np.min(selected_values))
                        ),
                        "maximum_value": (
                            None if not valid_count else float(np.max(selected_values))
                        ),
                        "support_state": (
                            "finite" if denominator_weight > 0 else "zero_denominator"
                        ),
                        "support_key_sha256": support_key,
                    }
                )
                if not valid_count:
                    continue
                indices = _grid_indices(selected_values, spec)
                unique_indices, inverse = np.unique(indices, return_inverse=True)
                bin_counts = np.bincount(inverse).astype(np.int64)
                bin_weights = np.bincount(inverse, weights=selected_weights).astype(
                    np.float64
                )
                for grid_index, count, weight in zip(
                    unique_indices, bin_counts, bin_weights, strict=True
                ):
                    accumulator.sparse_rows.append(
                        {
                            "metric_id": spec.metric_id,
                            "recording_id": recording_id,
                            "scope_id": scope_id,
                            "group_key_sha256": group_sha256,
                            "weighting_id": weighting_id,
                            "support_key_sha256": support_key,
                            "grid_index": int(grid_index),
                            "bin_count": int(count),
                            "bin_weight": float(weight),
                        }
                    )


def _finalize_recording_bins(
    *,
    config: ValidatedBehaviorDistributionConfig,
    source_export_run_id: str,
    source_export_manifest_sha256: str,
    accumulator: _SparseAccumulator,
) -> tuple[
    tuple[Mapping[str, object], ...],
    tuple[Mapping[str, object], ...],
    tuple[Mapping[str, object], ...],
]:
    specs = {spec.metric_id: spec for spec in config.metric_specs}
    recipes = {
        metric_id: _resolve_axis(specs[metric_id], audit)
        for metric_id, audit in sorted(accumulator.axis_audits.items())
    }
    if set(recipes) != set(specs):
        missing = sorted(set(specs) - set(recipes))
        _fail(f"Distribution metrics have no source audit: {missing}")
    support_by_key = {
        str(row["support_key_sha256"]): row for row in accumulator.support_rows
    }
    if len(support_by_key) != len(accumulator.support_rows):
        _fail("Recording distribution support primary key is duplicated")
    output: list[dict[str, object]] = []
    for sparse in accumulator.sparse_rows:
        spec = specs[str(sparse["metric_id"])]
        recipe = recipes[spec.metric_id]
        grid_index = int(sparse["grid_index"])
        bin_index = grid_index + int(recipe["grid_index_offset"])
        count = int(recipe["bin_count"])
        if bin_index < 0 or bin_index >= count:
            _fail(f"{spec.metric_id}: sparse bin escaped its resolved recipe")
        support = support_by_key[str(sparse["support_key_sha256"])]
        denominator = float(support["denominator_weight"])
        if denominator <= 0:
            _fail("A nonzero sparse bin references a zero denominator")
        left = float(recipe["bin_edges"][bin_index])
        right = float(recipe["bin_edges"][bin_index + 1])
        output.append(
            {
                "distribution_run_id": config.distribution_run_id,
                "source_export_run_id": source_export_run_id,
                "source_export_manifest_sha256": source_export_manifest_sha256,
                "metric_id": spec.metric_id,
                "metric_spec_sha256": spec.spec_sha256,
                "histogram_recipe_sha256": recipe["histogram_recipe_sha256"],
                "metric_family": spec.metric_family,
                "recording_id": str(sparse["recording_id"]),
                "scope_id": str(sparse["scope_id"]),
                "group_key_sha256": str(sparse["group_key_sha256"]),
                "weighting_id": str(sparse["weighting_id"]),
                "support_key_sha256": str(sparse["support_key_sha256"]),
                "bin_index": bin_index,
                "bin_left": left,
                "bin_right": right,
                "bin_center": (left + right) / 2.0,
                "bin_count": int(sparse["bin_count"]),
                "bin_weight": float(sparse["bin_weight"]),
                "fraction": float(sparse["bin_weight"]) / denominator,
            }
        )
    output.sort(
        key=lambda row: (
            row["metric_id"],
            row["recording_id"],
            SCOPE_ORDER.index(str(row["scope_id"])),
            row["group_key_sha256"],
            row["weighting_id"],
            row["bin_index"],
        )
    )
    support = tuple(
        sorted(
            accumulator.support_rows,
            key=lambda row: (
                row["metric_id"],
                row["recording_id"],
                SCOPE_ORDER.index(str(row["scope_id"])),
                row["group_key_sha256"],
                row["weighting_id"],
            ),
        )
    )
    return tuple(recipes[key] for key in sorted(recipes)), support, tuple(output)


def _cohort_bin_rows(
    *,
    config: ValidatedBehaviorDistributionConfig,
    source_export_run_id: str,
    source_export_manifest_sha256: str,
    parent_recording_count: int,
    recipes: Sequence[Mapping[str, object]],
    support_rows: Sequence[Mapping[str, object]],
    sparse_rows: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    recipe_by_metric = {str(row["metric_id"]): row for row in recipes}
    support_groups: dict[tuple[str, str, str, str], list[Mapping[str, object]]] = (
        defaultdict(list)
    )
    for row in support_rows:
        support_groups[
            (
                str(row["metric_id"]),
                str(row["scope_id"]),
                str(row["group_key_sha256"]),
                str(row["weighting_id"]),
            )
        ].append(row)
    sparse_lookup = {
        (str(row["support_key_sha256"]), int(row["bin_index"])): row
        for row in sparse_rows
    }
    if len(sparse_lookup) != len(sparse_rows):
        _fail("Recording nonzero-bin primary key is duplicated")
    output: list[dict[str, object]] = []
    for key, contributors in sorted(support_groups.items()):
        metric_id, scope_id, group_sha256, weighting_id = key
        recipe = recipe_by_metric[metric_id]
        finite = [row for row in contributors if float(row["denominator_weight"]) > 0]
        contributor_count = len(contributors)
        finite_count = len(finite)
        group_json_values = {str(row["group_key_json"]) for row in contributors}
        if len(group_json_values) != 1:
            _fail("One cohort histogram group digest maps to multiple group payloads")
        group_json = group_json_values.pop()
        for bin_index in range(int(recipe["bin_count"])):
            fractions: list[float] = []
            source_bin_count_sum = 0
            source_bin_weight_sum = 0.0
            source_denominator_count_sum = 0
            source_denominator_weight_sum = 0.0
            for support in finite:
                sparse = sparse_lookup.get(
                    (str(support["support_key_sha256"]), bin_index)
                )
                bin_count = 0 if sparse is None else int(sparse["bin_count"])
                bin_weight = 0.0 if sparse is None else float(sparse["bin_weight"])
                denominator = float(support["denominator_weight"])
                fractions.append(bin_weight / denominator)
                source_bin_count_sum += bin_count
                source_bin_weight_sum += bin_weight
                source_denominator_count_sum += int(support["valid_count"])
                source_denominator_weight_sum += denominator
            values = np.asarray(fractions, dtype=np.float64)
            mean = float(np.mean(values)) if values.size else None
            median = float(np.median(values)) if values.size else None
            std = float(np.std(values, ddof=1)) if values.size > 1 else None
            sem = std / math.sqrt(values.size) if std is not None else None
            left = float(recipe["bin_edges"][bin_index])
            right = float(recipe["bin_edges"][bin_index + 1])
            output.append(
                {
                    "distribution_run_id": config.distribution_run_id,
                    "source_export_run_id": source_export_run_id,
                    "source_export_manifest_sha256": source_export_manifest_sha256,
                    "metric_id": metric_id,
                    "metric_spec_sha256": next(
                        str(row["metric_spec_sha256"]) for row in contributors
                    ),
                    "histogram_recipe_sha256": str(recipe["histogram_recipe_sha256"]),
                    "metric_family": str(contributors[0]["metric_family"]),
                    "scope_id": scope_id,
                    "group_key_json": group_json,
                    "group_key_sha256": group_sha256,
                    "weighting_id": weighting_id,
                    "weight_unit": str(contributors[0]["weight_unit"]),
                    "bin_index": bin_index,
                    "bin_left": left,
                    "bin_right": right,
                    "bin_center": (left + right) / 2.0,
                    "parent_recording_count": int(parent_recording_count),
                    "contributor_recording_count": contributor_count,
                    "finite_recording_count": finite_count,
                    "excluded_zero_denominator_recording_count": (
                        contributor_count - finite_count
                    ),
                    "noncontributor_recording_count": (
                        int(parent_recording_count) - contributor_count
                    ),
                    "source_bin_count_sum": source_bin_count_sum,
                    "source_bin_weight_sum": source_bin_weight_sum,
                    "source_denominator_count_sum": source_denominator_count_sum,
                    "source_denominator_weight_sum": source_denominator_weight_sum,
                    "pooled_fraction": (
                        source_bin_weight_sum / source_denominator_weight_sum
                        if source_denominator_weight_sum > 0
                        else None
                    ),
                    "mean_recording_fraction": mean,
                    "median_recording_fraction": median,
                    "sample_std_recording_fraction": std,
                    "sem_recording_fraction": sem,
                    "minimum_recording_fraction": (
                        None if not values.size else float(np.min(values))
                    ),
                    "p25_recording_fraction": (
                        None if not values.size else float(np.quantile(values, 0.25))
                    ),
                    "p75_recording_fraction": (
                        None if not values.size else float(np.quantile(values, 0.75))
                    ),
                    "maximum_recording_fraction": (
                        None if not values.size else float(np.max(values))
                    ),
                }
            )
    return tuple(output)


def _numeric_column(rows: Sequence[Mapping[str, Any]], name: str) -> np.ndarray:
    return np.asarray(
        [np.nan if row.get(name) is None else float(row[name]) for row in rows],
        dtype=np.float64,
    )


def _object_column(rows: Sequence[Mapping[str, Any]], name: str) -> np.ndarray:
    return np.asarray([row.get(name) for row in rows], dtype=object)


def _constant_array(value: Any, count: int) -> np.ndarray:
    result = np.empty(count, dtype=object)
    result[:] = value
    return result


def _source_queries(
    dataset: ValidatedBehaviorExportDataset,
) -> tuple[Mapping[str, object], ...]:
    selections: Mapping[str, tuple[str, ...]] = {
        "cohort_recordings": (
            "recording_id",
            "analysis_unit_kind",
            "analysis_unit_id",
            "membership_state",
        ),
        "canonical_swim_bouts": _CANONICAL_BOUT_COLUMNS,
        "semantic_epochs": (
            "recording_id",
            "epoch_window_id",
            "analysis_role",
            "start_frame",
            "end_frame_exclusive",
            "source_interval_sha256",
            "protocol_semantic_hash",
            "protocol_semantic_step_index",
            "protocol_semantic_step_ref",
        ),
        "epoch_behavior_summary": (
            "recording_id",
            "track_id",
            "epoch_window_id",
            "analysis_role",
            "start_frame",
            "end_frame",
            "start_time_s",
            "end_time_s",
            "duration_s",
            "total_span_frames",
            "valid_tracked_frame_count",
            "valid_tracked_duration_s",
            "source_interval_sha256",
        ),
        "provider_motion_samples": _MOTION_COLUMNS,
        "chaser_relative_samples": _DISTANCE_COLUMNS,
        "chaser_occurrences": tuple(
            item.name
            for item in dataset.table("chaser_occurrences").spec.contract.fields
        ),
    }
    rows: list[Mapping[str, object]] = []
    for name, columns in selections.items():
        query = dataset.table(name).query_identity(
            columns=columns,
            predicate_description=(
                "manifest-selected exact rows; dense sources read one bundle-set "
                "member shard at a time; semantic scopes use exact half-open frame "
                "intervals; no interpolation or selector discovery"
            ),
        )
        rows.append({**query, "query_sha256": canonical_json_sha256(query)})
    return tuple(rows)


def _validate_analysis_unit_policy(
    dataset: ValidatedBehaviorExportDataset,
) -> tuple[int, tuple[str, ...]]:
    import polars as pl

    policy = dataset.manifest.get("analysis_unit_policy")
    if not isinstance(policy, Mapping) or not isinstance(policy.get("record"), Mapping):
        _fail("Parent export lacks its analysis-unit policy")
    record = policy["record"]
    if (
        record.get("analysis_unit_kind") != "recording"
        or record.get("member_id_field") != "recording_id"
    ):
        _fail("Distribution successor requires recording-scoped analysis units")
    frame = (
        dataset.table("cohort_recordings")
        .scan(
            columns=(
                "recording_id",
                "analysis_unit_kind",
                "analysis_unit_id",
                "membership_state",
            )
        )
        .collect()
    )
    if frame.height == 0 or frame.get_column("recording_id").n_unique() != frame.height:
        _fail("Parent recording roster is empty or duplicated")
    if frame.filter(
        (pl.col("analysis_unit_kind") != "recording")
        | (pl.col("analysis_unit_id") != pl.col("recording_id"))
    ).height:
        _fail("Parent recording roster violates its analysis-unit policy")
    return frame.height, tuple(sorted(frame.get_column("recording_id").to_list()))


def _augment_intervals(
    *,
    config: ValidatedBehaviorDistributionConfig,
    source_export_run_id: str,
    source_export_manifest_sha256: str,
    recording_id: str,
    intervals: Sequence[Mapping[str, Any]],
    fps: float,
    canonical_rows: Sequence[Mapping[str, Any]],
    epoch_binding: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not canonical_rows:
        return []
    first = canonical_rows[0]
    return [
        {
            "distribution_run_id": config.distribution_run_id,
            "source_export_run_id": source_export_run_id,
            "source_export_manifest_sha256": source_export_manifest_sha256,
            "recording_id": recording_id,
            "membership_member_sha256": str(first["membership_member_sha256"]),
            "bundle_set_member_sha256": str(first["bundle_set_member_sha256"]),
            "bundle_record_sha256": str(first["bundle_record_sha256"]),
            "canonical_source_run_path": str(first["source_run_path"]),
            "canonical_source_manifest_sha256": str(first["source_manifest_sha256"]),
            "epoch_source_run_path": str(epoch_binding["run_path"]),
            "epoch_source_manifest_sha256": str(epoch_binding["manifest_sha256"]),
            "epoch_source_receipt_sha256": str(epoch_binding["receipt_sha256"]),
            **dict(row),
            "fps": float(fps),
            "interval_derivation_id": (
                "producer_authored_interval_after_canonical_bout_axis_check_v1"
            ),
        }
        for row in intervals
    ]


def _event_identity_arrays(
    rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, np.ndarray]:
    names = (
        "membership_member_sha256",
        "bundle_set_member_sha256",
        "bundle_record_sha256",
        "canonical_source_run_path",
        "canonical_source_manifest_sha256",
        "epoch_source_run_path",
        "epoch_source_manifest_sha256",
        "epoch_source_receipt_sha256",
    )
    return MappingProxyType({name: _object_column(rows, name) for name in names})


def _reduce_event_surfaces(
    accumulator: _SparseAccumulator,
    *,
    config: ValidatedBehaviorDistributionConfig,
    source_export_run_id: str,
    source_export_manifest_sha256: str,
    recording_id: str,
    bout_rows: Sequence[Mapping[str, Any]],
    interval_rows: Sequence[Mapping[str, Any]],
    valid_duration_by_scope: Mapping[str, float],
) -> None:
    by_surface = {
        "bout_observations": bout_rows,
        "inter_bout_interval_observations": interval_rows,
    }
    for spec in config.metric_specs:
        if spec.source_surface not in _EVENT_SURFACES:
            continue
        rows = by_surface[spec.source_surface]
        values = _numeric_column(rows, spec.value_column)
        roles = _object_column(rows, "analysis_role")
        base_valid = np.ones(values.shape, dtype=bool)
        if spec.validity_policy_id == "derived_angular_valid_and_epoch_crosschecked_v1":
            base_valid &= np.asarray(
                [bool(row["heading_valid"]) for row in rows], dtype=bool
            )
        elif spec.source_surface == "inter_bout_interval_observations":
            base_valid &= np.asarray(
                [bool(row["interval_valid"]) for row in rows], dtype=bool
            )
        _reduce_metric_values(
            accumulator,
            config=config,
            spec=spec,
            source_export_run_id=source_export_run_id,
            source_export_manifest_sha256=source_export_manifest_sha256,
            recording_id=recording_id,
            values=values,
            scope_masks=_event_scope_masks(roles),
            base_valid=base_valid,
            group_arrays={},
            identity_arrays=_event_identity_arrays(rows),
            time_weights_s=None,
            time_scope_masks=None,
            valid_duration_by_scope=valid_duration_by_scope,
        )


def _reduce_motion_surfaces(
    accumulator: _SparseAccumulator,
    *,
    config: ValidatedBehaviorDistributionConfig,
    source_export_run_id: str,
    source_export_manifest_sha256: str,
    recording_id: str,
    motion_frame: Any,
    epochs: Sequence[Mapping[str, Any]],
    valid_duration_by_scope: Mapping[str, float],
) -> None:
    frames = motion_frame.get_column("acquisition_frame_id").to_numpy()
    scopes = _scope_masks(frames, epochs)
    transition_scopes = _transition_scope_masks(
        frames,
        motion_frame.get_column("delta_frames").to_numpy(),
        epochs,
    )
    linear_valid = motion_frame.get_column("linear_sample_valid").to_numpy().astype(
        bool
    ) & motion_frame.get_column("transition_valid").to_numpy().astype(bool)
    angular_valid = motion_frame.get_column("angular_sample_valid").to_numpy().astype(
        bool
    ) & motion_frame.get_column("transition_valid").to_numpy().astype(bool)
    time_weights = motion_frame.get_column("delta_s").to_numpy().astype(np.float64)
    group_arrays = {
        "provider_role": motion_frame.get_column("provider_role").to_numpy()
    }
    identity_arrays = {
        name: motion_frame.get_column(name).to_numpy()
        for name in (
            "position_provider_id",
            "position_provider_digest",
            "source_run_path",
            "source_manifest_sha256",
            "source_verification_digest",
            "track_id",
        )
    }
    for spec in config.metric_specs:
        if spec.source_surface != "provider_motion_samples":
            continue
        base_valid = (
            angular_valid
            if spec.validity_policy_id == _ANGULAR_MOTION_VALIDITY
            else linear_valid
        )
        _reduce_metric_values(
            accumulator,
            config=config,
            spec=spec,
            source_export_run_id=source_export_run_id,
            source_export_manifest_sha256=source_export_manifest_sha256,
            recording_id=recording_id,
            values=motion_frame.get_column(spec.value_column).to_numpy(),
            scope_masks=scopes,
            base_valid=base_valid,
            group_arrays=group_arrays,
            identity_arrays=identity_arrays,
            time_weights_s=time_weights,
            time_scope_masks=transition_scopes,
            valid_duration_by_scope=valid_duration_by_scope,
        )


def _reduce_distance_surfaces(
    accumulator: _SparseAccumulator,
    *,
    config: ValidatedBehaviorDistributionConfig,
    source_export_run_id: str,
    source_export_manifest_sha256: str,
    recording_id: str,
    distance_frame: Any,
    epochs: Sequence[Mapping[str, Any]],
    valid_duration_by_scope: Mapping[str, float],
) -> None:
    if distance_frame.height == 0:
        _fail(f"{recording_id}: chaser-relative source unexpectedly has zero rows")
    frames = distance_frame.get_column("acquisition_frame_id").to_numpy()
    scopes = _scope_masks(frames, epochs)
    transition_scopes = _transition_scope_masks(
        frames,
        distance_frame.get_column("acquisition_frame_delta").to_numpy(),
        epochs,
    )
    base_valid = np.logical_and.reduce(
        [
            distance_frame.get_column(name).to_numpy().astype(bool)
            for name in (
                "chaser_behavior_role_valid",
                "selection_member",
                "chaser_occurrence_member",
                "row_valid",
                "relative_physical_valid",
            )
        ]
    )
    timestamp_valid = (
        distance_frame.get_column("timestamp_valid").to_numpy().astype(bool)
    )
    timestamp_delta = (
        distance_frame.get_column("timestamp_delta_ns").to_numpy().astype(np.float64)
        / 1_000_000_000.0
    )
    timestamp_delta[~timestamp_valid] = np.nan
    relative_transition_valid = (
        distance_frame.get_column("relative_transition_valid")
        .to_numpy()
        .astype(bool)
    )
    timestamp_delta[~relative_transition_valid] = np.nan
    group_arrays = {
        "provider_role": distance_frame.get_column("provider_role").to_numpy(),
        "behavior_role": distance_frame.get_column("behavior_role").to_numpy(),
    }
    identity_arrays = {
        name: distance_frame.get_column(name).to_numpy()
        for name in (
            "position_provider_id",
            "position_provider_digest",
            "source_run_path",
            "source_manifest_sha256",
            "source_receipt_sha256",
            "chaser_identity_code",
            "chaser_identity",
        )
    }
    for spec in config.metric_specs:
        if spec.source_surface != "chaser_relative_samples":
            continue
        if spec.validity_policy_id != _DISTANCE_VALIDITY:
            _fail(f"{spec.metric_id}: unsupported chaser-distance validity policy")
        _reduce_metric_values(
            accumulator,
            config=config,
            spec=spec,
            source_export_run_id=source_export_run_id,
            source_export_manifest_sha256=source_export_manifest_sha256,
            recording_id=recording_id,
            values=distance_frame.get_column(spec.value_column).to_numpy(),
            scope_masks=scopes,
            base_valid=base_valid,
            group_arrays=group_arrays,
            identity_arrays=identity_arrays,
            time_weights_s=timestamp_delta,
            time_scope_masks=transition_scopes,
            valid_duration_by_scope=valid_duration_by_scope,
        )


def compute_validated_behavior_distributions(
    dataset: ValidatedBehaviorExportDataset,
    config: ValidatedBehaviorDistributionConfig,
    *,
    progress: Callable[[str], None] | None = None,
) -> ValidatedBehaviorDistributionResult:
    """Compute one compact successor from an exact validated parent export."""

    required_tables = {
        "cohort_recordings",
        "canonical_swim_bouts",
        "semantic_epochs",
        "epoch_behavior_summary",
        "provider_motion_samples",
        "chaser_relative_samples",
        "chaser_occurrences",
    }
    missing = sorted(required_tables - set(dataset.table_names))
    if missing:
        _fail(f"Parent export lacks required distribution tables: {missing}")
    parent_count, parent_recording_ids = _validate_analysis_unit_policy(dataset)
    appearance = build_chaser_appearance_dimension(dataset)
    validate_chaser_appearance_dimension(
        appearance,
        expected_export_manifest_sha256=dataset.cache_identity,
    )

    members = tuple(
        sorted(
            (
                member
                for member in dataset.bundle_set["members"]
                if member.get("bundle_state") == "complete"
            ),
            key=lambda row: int(row["ordinal"]),
        )
    )
    recording_ids = tuple(str(member["recording_id"]) for member in members)
    if len(set(recording_ids)) != len(recording_ids) or not set(recording_ids).issubset(
        parent_recording_ids
    ):
        _fail("Complete bundle-set recording membership is duplicated or foreign")

    tables = {name: dataset.table(name) for name in required_tables}
    accumulator = _SparseAccumulator()
    bout_observations: list[Mapping[str, object]] = []
    interval_observations: list[Mapping[str, object]] = []
    epoch_receipts: list[Mapping[str, object]] = []

    for member_index, member in enumerate(members, start=1):
        ordinal = int(member["ordinal"])
        recording_id = str(member["recording_id"])
        if progress is not None:
            progress(f"recording {member_index}/{len(members)}: {recording_id}")
        source = _open_member_source(member)
        epoch_arrays, epoch_receipt = _load_epoch_child_arrays(source)
        epoch_receipts.append(epoch_receipt)

        semantic_frame = _read_member_table(
            tables["semantic_epochs"],
            ordinal=ordinal,
            recording_id=recording_id,
            columns=(
                "recording_id",
                "epoch_window_id",
                "analysis_role",
                "start_frame",
                "end_frame_exclusive",
                "source_interval_sha256",
                "protocol_semantic_hash",
                "protocol_semantic_step_index",
                "protocol_semantic_step_ref",
            ),
        )
        epoch_summary_frame = _read_member_table(
            tables["epoch_behavior_summary"],
            ordinal=ordinal,
            recording_id=recording_id,
            columns=(
                "recording_id",
                "track_id",
                "epoch_window_id",
                "analysis_role",
                "start_frame",
                "end_frame",
                "start_time_s",
                "end_time_s",
                "duration_s",
                "total_span_frames",
                "valid_tracked_frame_count",
                "valid_tracked_duration_s",
                "source_interval_sha256",
            ),
        )
        epochs = _epochs_from_frames(
            semantic_frame.to_dicts(), epoch_summary_frame.to_dicts()
        )
        epoch_fps = _fps_from_epoch_rows(epoch_summary_frame.to_dicts())

        canonical_frame = _read_member_table(
            tables["canonical_swim_bouts"],
            ordinal=ordinal,
            recording_id=recording_id,
            columns=_CANONICAL_BOUT_COLUMNS,
        )
        motion_frame = _read_member_table(
            tables["provider_motion_samples"],
            ordinal=ordinal,
            recording_id=recording_id,
            columns=_MOTION_COLUMNS,
        ).sort("acquisition_frame_id")
        if motion_frame.height == 0:
            _fail(f"{recording_id}: provider-motion source unexpectedly has zero rows")
        if (
            motion_frame.get_column("acquisition_frame_id").n_unique()
            != motion_frame.height
        ):
            _fail(f"{recording_id}: provider-motion frame axis is not unique")
        swim_tables = _load_bound_swim_bout_tables(source)
        raw_intervals, fps = _materialize_bound_intervals(
            canonical_rows=canonical_frame.to_dicts(),
            tables=swim_tables,
            epochs=epochs,
        )
        if not math.isclose(epoch_fps, fps, rel_tol=1e-9, abs_tol=1e-9):
            _fail(f"{recording_id}: swim-bout and epoch-summary FPS disagree")
        _validate_motion_fps(motion_frame, fps=fps)

        epoch_binding = source.scientific_child("epoch_behavior").binding
        recording_bouts = _materialize_bout_observations(
            config=config,
            source_export_run_id=dataset.export_run_id,
            source_export_manifest_sha256=dataset.cache_identity,
            recording_id=recording_id,
            canonical_frame=canonical_frame,
            motion_frame=motion_frame,
            epochs=epochs,
            epoch_arrays=epoch_arrays,
            epoch_binding=epoch_binding,
        )
        _validate_epoch_ibi_histograms(
            recording_id=recording_id,
            intervals=raw_intervals,
            epochs=epochs,
            epoch_arrays=epoch_arrays,
        )
        recording_intervals = _augment_intervals(
            config=config,
            source_export_run_id=dataset.export_run_id,
            source_export_manifest_sha256=dataset.cache_identity,
            recording_id=recording_id,
            intervals=raw_intervals,
            fps=fps,
            canonical_rows=canonical_frame.to_dicts(),
            epoch_binding=epoch_binding,
        )
        bout_observations.extend(recording_bouts)
        interval_observations.extend(recording_intervals)

        linear_valid_count = int(
            np.count_nonzero(
                motion_frame.get_column("linear_sample_valid").to_numpy().astype(bool)
            )
        )
        valid_duration_by_scope = {
            "whole_session": float(linear_valid_count) / fps,
            **{
                str(epoch["analysis_role"]): float(epoch["valid_tracked_duration_s"])
                for epoch in epochs
            },
        }
        _reduce_event_surfaces(
            accumulator,
            config=config,
            source_export_run_id=dataset.export_run_id,
            source_export_manifest_sha256=dataset.cache_identity,
            recording_id=recording_id,
            bout_rows=recording_bouts,
            interval_rows=recording_intervals,
            valid_duration_by_scope=valid_duration_by_scope,
        )
        _reduce_motion_surfaces(
            accumulator,
            config=config,
            source_export_run_id=dataset.export_run_id,
            source_export_manifest_sha256=dataset.cache_identity,
            recording_id=recording_id,
            motion_frame=motion_frame,
            epochs=epochs,
            valid_duration_by_scope=valid_duration_by_scope,
        )
        if any(
            spec.source_surface == "chaser_relative_samples"
            for spec in config.metric_specs
        ):
            distance_frame = _read_member_table(
                tables["chaser_relative_samples"],
                ordinal=ordinal,
                recording_id=recording_id,
                columns=_DISTANCE_COLUMNS,
            )
            _reduce_distance_surfaces(
                accumulator,
                config=config,
                source_export_run_id=dataset.export_run_id,
                source_export_manifest_sha256=dataset.cache_identity,
                recording_id=recording_id,
                distance_frame=distance_frame,
                epochs=epochs,
                valid_duration_by_scope=valid_duration_by_scope,
            )

    recipes, support, sparse = _finalize_recording_bins(
        config=config,
        source_export_run_id=dataset.export_run_id,
        source_export_manifest_sha256=dataset.cache_identity,
        accumulator=accumulator,
    )
    cohort_bins = _cohort_bin_rows(
        config=config,
        source_export_run_id=dataset.export_run_id,
        source_export_manifest_sha256=dataset.cache_identity,
        parent_recording_count=parent_count,
        recipes=recipes,
        support_rows=support,
        sparse_rows=sparse,
    )
    if progress is not None:
        progress(
            "finalized "
            f"{len(support):,} recording supports, {len(sparse):,} nonzero bins, "
            f"and {len(cohort_bins):,} cohort bins"
        )
    source_export = {
        "path": str(dataset.root),
        "export_run_id": dataset.export_run_id,
        "export_manifest_record_sha256": dataset.cache_identity,
        "export_plan_sha256": dataset.manifest["export_plan"]["plan_sha256"],
        "export_profile": _plain(dataset.manifest["export_profile"]),
        "analysis_unit_policy": _plain(dataset.manifest["analysis_unit_policy"]),
        "membership": _plain(dataset.manifest["membership"]),
        "bundle_set": _plain(dataset.manifest["bundle_set"]),
        "validation_receipt": _plain(dataset.manifest["validation_receipt"]),
        "validation_mode": dataset.validation_mode,
    }
    cohort_summary = {
        "parent_recording_count": parent_count,
        "contributing_recording_count": len(recording_ids),
        "noncontributing_recording_count": parent_count - len(recording_ids),
        "parent_recording_ids_sha256": canonical_json_sha256(
            list(parent_recording_ids)
        ),
        "contributing_recording_ids_sha256": canonical_json_sha256(
            list(sorted(recording_ids))
        ),
        "bout_observation_count": len(bout_observations),
        "inter_bout_interval_observation_count": len(interval_observations),
        "recording_support_count": len(support),
        "recording_nonzero_bin_count": len(sparse),
        "cohort_bin_count": len(cohort_bins),
    }
    return ValidatedBehaviorDistributionResult(
        config=config,
        source_export=MappingProxyType(source_export),
        cohort_summary=MappingProxyType(cohort_summary),
        source_queries=_source_queries(dataset),
        epoch_child_receipts=tuple(epoch_receipts),
        histogram_recipes=recipes,
        chaser_appearance_dimension=appearance,
        bout_observations=tuple(bout_observations),
        inter_bout_interval_observations=tuple(interval_observations),
        recording_support=support,
        recording_nonzero_bins=sparse,
        cohort_bins=cohort_bins,
    )


_BOUT_OBSERVATION_SCHEMA = (
    ("distribution_run_id", "string", False),
    ("source_export_run_id", "string", False),
    ("source_export_manifest_sha256", "string", False),
    ("recording_id", "string", False),
    ("membership_member_sha256", "string", False),
    ("bundle_set_member_sha256", "string", False),
    ("bundle_record_sha256", "string", False),
    ("canonical_source_run_path", "string", False),
    ("canonical_source_manifest_sha256", "string", False),
    ("canonical_source_payload_sha256", "string", False),
    ("epoch_source_run_path", "string", False),
    ("epoch_source_manifest_sha256", "string", False),
    ("epoch_source_payload_sha256", "string", False),
    ("epoch_source_receipt_sha256", "string", False),
    ("track_id", "int64", False),
    ("source_signal_id", "int64", False),
    ("bout_row_id", "int64", False),
    ("bout_id", "int64", False),
    ("start_acquisition_frame_id", "int64", False),
    ("end_acquisition_frame_id", "int64", False),
    ("bout_event_frame", "int64", True),
    ("bout_event_time_s", "float64", True),
    ("epoch_window_id", "int64", True),
    ("analysis_role", "string", True),
    ("source_interval_sha256", "string", True),
    ("protocol_semantic_hash", "string", True),
    ("protocol_semantic_step_index", "int32", True),
    ("protocol_semantic_step_ref", "string", True),
    ("epoch_membership_state", "string", False),
    ("duration_s", "float64", True),
    ("path_length_mm", "float64", True),
    ("net_displacement_mm", "float64", True),
    ("mean_speed_mm_s", "float64", True),
    ("peak_speed_mm_s", "float64", True),
    ("tortuosity", "float64", True),
    ("net_heading_change_deg", "float64", True),
    ("abs_net_heading_change_deg", "float64", True),
    ("heading_path_deg", "float64", True),
    ("heading_valid", "bool", False),
    ("heading_derivation_id", "string", False),
    ("epoch_heading_crosscheck_state", "string", False),
)
_INTERVAL_OBSERVATION_SCHEMA = (
    ("distribution_run_id", "string", False),
    ("source_export_run_id", "string", False),
    ("source_export_manifest_sha256", "string", False),
    ("recording_id", "string", False),
    ("membership_member_sha256", "string", False),
    ("bundle_set_member_sha256", "string", False),
    ("bundle_record_sha256", "string", False),
    ("canonical_source_run_path", "string", False),
    ("canonical_source_manifest_sha256", "string", False),
    ("epoch_source_run_path", "string", False),
    ("epoch_source_manifest_sha256", "string", False),
    ("epoch_source_receipt_sha256", "string", False),
    ("interval_row_id", "int64", False),
    ("previous_bout_row_id", "int64", False),
    ("previous_bout_id", "int64", False),
    ("next_bout_row_id", "int64", False),
    ("next_bout_id", "int64", False),
    ("previous_end_frame", "int64", False),
    ("next_start_frame", "int64", False),
    ("interval_frames", "int64", False),
    ("previous_end_time_s", "float64", False),
    ("next_start_time_s", "float64", False),
    ("interval_s", "float64", False),
    ("interval_valid", "bool", False),
    ("epoch_window_id", "int64", True),
    ("analysis_role", "string", True),
    ("source_interval_sha256", "string", True),
    ("epoch_membership_state", "string", False),
    ("fps", "float64", False),
    ("interval_derivation_id", "string", False),
)
_SUPPORT_SCHEMA = (
    ("distribution_run_id", "string", False),
    ("source_export_run_id", "string", False),
    ("source_export_manifest_sha256", "string", False),
    ("metric_id", "string", False),
    ("metric_spec_sha256", "string", False),
    ("metric_family", "string", False),
    ("source_surface", "string", False),
    ("recording_id", "string", False),
    ("scope_id", "string", False),
    ("group_key_json", "string", False),
    ("group_key_sha256", "string", False),
    ("source_identity_key_json", "string", False),
    ("source_identity_key_sha256", "string", False),
    ("weighting_id", "string", False),
    ("weight_unit", "string", False),
    ("candidate_count", "int64", False),
    ("valid_count", "int64", False),
    ("excluded_count", "int64", False),
    ("denominator_weight", "float64", False),
    ("valid_duration_s", "float64", False),
    ("event_rate_per_valid_min", "float64", True),
    ("minimum_value", "float64", True),
    ("maximum_value", "float64", True),
    ("support_state", "string", False),
    ("support_key_sha256", "string", False),
)
_NONZERO_BIN_SCHEMA = (
    ("distribution_run_id", "string", False),
    ("source_export_run_id", "string", False),
    ("source_export_manifest_sha256", "string", False),
    ("metric_id", "string", False),
    ("metric_spec_sha256", "string", False),
    ("histogram_recipe_sha256", "string", False),
    ("metric_family", "string", False),
    ("recording_id", "string", False),
    ("scope_id", "string", False),
    ("group_key_sha256", "string", False),
    ("weighting_id", "string", False),
    ("support_key_sha256", "string", False),
    ("bin_index", "int32", False),
    ("bin_left", "float64", False),
    ("bin_right", "float64", False),
    ("bin_center", "float64", False),
    ("bin_count", "int64", False),
    ("bin_weight", "float64", False),
    ("fraction", "float64", False),
)
_COHORT_BIN_SCHEMA = (
    ("distribution_run_id", "string", False),
    ("source_export_run_id", "string", False),
    ("source_export_manifest_sha256", "string", False),
    ("metric_id", "string", False),
    ("metric_spec_sha256", "string", False),
    ("histogram_recipe_sha256", "string", False),
    ("metric_family", "string", False),
    ("scope_id", "string", False),
    ("group_key_json", "string", False),
    ("group_key_sha256", "string", False),
    ("weighting_id", "string", False),
    ("weight_unit", "string", False),
    ("bin_index", "int32", False),
    ("bin_left", "float64", False),
    ("bin_right", "float64", False),
    ("bin_center", "float64", False),
    ("parent_recording_count", "int32", False),
    ("contributor_recording_count", "int32", False),
    ("finite_recording_count", "int32", False),
    ("excluded_zero_denominator_recording_count", "int32", False),
    ("noncontributor_recording_count", "int32", False),
    ("source_bin_count_sum", "int64", False),
    ("source_bin_weight_sum", "float64", False),
    ("source_denominator_count_sum", "int64", False),
    ("source_denominator_weight_sum", "float64", False),
    ("pooled_fraction", "float64", True),
    ("mean_recording_fraction", "float64", True),
    ("median_recording_fraction", "float64", True),
    ("sample_std_recording_fraction", "float64", True),
    ("sem_recording_fraction", "float64", True),
    ("minimum_recording_fraction", "float64", True),
    ("p25_recording_fraction", "float64", True),
    ("p75_recording_fraction", "float64", True),
    ("maximum_recording_fraction", "float64", True),
)


def _arrow_schema(fields: Sequence[tuple[str, str, bool]]) -> Any:
    import pyarrow as pa

    types = {
        "string": pa.string(),
        "bool": pa.bool_(),
        "int32": pa.int32(),
        "int64": pa.int64(),
        "float64": pa.float64(),
    }
    return pa.schema(
        [
            pa.field(name, types[type_name], nullable=nullable)
            for name, type_name, nullable in fields
        ]
    )


def _schema_sha256(schema: Any) -> str:
    return canonical_json_sha256(
        {
            "fields": [
                {
                    "name": field.name,
                    "type": str(field.type),
                    "nullable": field.nullable,
                }
                for field in schema
            ]
        }
    )


def _write_parquet(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fields: Sequence[tuple[str, str, bool]],
) -> dict[str, object]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = _arrow_schema(fields)
    table = pa.Table.from_pylist([dict(row) for row in rows], schema=schema)
    pq.write_table(table, path, compression="zstd")
    return {
        "path": path.name,
        "row_count": table.num_rows,
        "size_bytes": path.stat().st_size,
        "file_sha256": sha256_file(path),
        "arrow_schema_sha256": _schema_sha256(schema),
    }


_OUTPUT_SCHEMAS: Mapping[str, tuple[tuple[str, str, bool], ...]] = MappingProxyType(
    {
        "bout_observations.parquet": _BOUT_OBSERVATION_SCHEMA,
        "inter_bout_interval_observations.parquet": _INTERVAL_OBSERVATION_SCHEMA,
        "recording_distribution_support.parquet": _SUPPORT_SCHEMA,
        "recording_distribution_nonzero_bins.parquet": _NONZERO_BIN_SCHEMA,
        "cohort_distribution_bins.parquet": _COHORT_BIN_SCHEMA,
    }
)
_PRIMARY_KEYS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "bout_observations.parquet": (
            "distribution_run_id",
            "recording_id",
            "track_id",
            "bout_row_id",
        ),
        "inter_bout_interval_observations.parquet": (
            "distribution_run_id",
            "recording_id",
            "interval_row_id",
        ),
        "recording_distribution_support.parquet": (
            "distribution_run_id",
            "metric_id",
            "recording_id",
            "scope_id",
            "group_key_sha256",
            "weighting_id",
        ),
        "recording_distribution_nonzero_bins.parquet": (
            "distribution_run_id",
            "metric_id",
            "recording_id",
            "scope_id",
            "group_key_sha256",
            "weighting_id",
            "bin_index",
        ),
        "cohort_distribution_bins.parquet": (
            "distribution_run_id",
            "metric_id",
            "scope_id",
            "group_key_sha256",
            "weighting_id",
            "bin_index",
        ),
    }
)


def _validate_json_identity_columns(
    frame: Any,
    *,
    json_column: str,
    digest_column: str,
    table_name: str,
) -> None:
    for row in frame.select((json_column, digest_column)).unique().to_dicts():
        try:
            value = json.loads(str(row[json_column]))
        except json.JSONDecodeError as exc:
            raise ValidatedBehaviorDistributionError(
                f"{table_name}: identity column is not strict JSON"
            ) from exc
        if (
            not isinstance(value, dict)
            or canonical_json_sha256(value) != row[digest_column]
        ):
            _fail(f"{table_name}: JSON identity digest is stale")


def _validate_distribution_directory(
    root: Path,
    manifest: Mapping[str, object],
) -> None:
    import polars as pl
    import pyarrow.parquet as pq

    if (
        manifest.get("schema_id") != SCHEMA_ID
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("method_id") != METHOD_ID
        or manifest.get("status") != STATUS
    ):
        _fail("Distribution manifest schema, method, or status is unsupported")
    for field_name in (
        "selector_eligible",
        "production_authority",
        "selector_activation",
        "registry_update",
        "source_export_mutation",
        "recording_zarr_mutation",
    ):
        if manifest.get(field_name) is not False:
            _fail(f"Distribution safety flag is not false: {field_name}")
    body = {key: value for key, value in manifest.items() if key != "record_sha256"}
    if manifest.get("record_sha256") != canonical_json_sha256(body):
        _fail("Distribution manifest self digest is stale")
    source = manifest.get("source_export")
    if not isinstance(source, Mapping):
        _fail("Distribution manifest lacks its source export")
    source_digest = source.get("export_manifest_record_sha256")
    appearance = manifest.get("chaser_appearance_dimension")
    if not isinstance(appearance, Mapping):
        _fail("Distribution manifest lacks its chaser appearance dimension")
    validate_chaser_appearance_dimension(
        appearance, expected_export_manifest_sha256=str(source_digest)
    )
    configuration = manifest.get("configuration")
    recipes = manifest.get("histogram_recipes")
    if not isinstance(configuration, Mapping) or not isinstance(recipes, list):
        _fail("Distribution manifest lacks its configuration or recipes")
    specs = configuration.get("metric_specs")
    if not isinstance(specs, list):
        _fail("Distribution configuration lacks metric specifications")
    spec_ids = {
        str(record.get("metric_id")) for record in specs if isinstance(record, Mapping)
    }
    recipe_ids: set[str] = set()
    recipe_digests: set[str] = set()
    for recipe in recipes:
        if not isinstance(recipe, Mapping):
            _fail("Distribution histogram recipe is malformed")
        digest = recipe.get("histogram_recipe_sha256")
        recipe_body = {
            key: value
            for key, value in recipe.items()
            if key != "histogram_recipe_sha256"
        }
        if digest != canonical_json_sha256(recipe_body):
            _fail("Distribution histogram recipe digest is stale")
        recipe_ids.add(str(recipe.get("metric_id")))
        recipe_digests.add(str(digest))
    if recipe_ids != spec_ids or len(recipes) != len(specs):
        _fail("Distribution metric and resolved recipe rosters differ")

    outputs = manifest.get("outputs")
    if not isinstance(outputs, list):
        _fail("Distribution manifest lacks output records")
    by_name = {
        str(record.get("path")): record
        for record in outputs
        if isinstance(record, Mapping)
    }
    if set(by_name) != set(_OUTPUT_SCHEMAS):
        _fail("Distribution output inventory is not exact")
    if {path.name for path in root.iterdir()} != {"manifest.json", *_OUTPUT_SCHEMAS}:
        _fail("Distribution directory contains an unrecorded file")

    frames: dict[str, Any] = {}
    for name, fields in _OUTPUT_SCHEMAS.items():
        record = by_name[name]
        path = root / name
        if Path(name).name != name or not path.is_file():
            _fail(f"Distribution output is unsafe or absent: {name}")
        if path.stat().st_size != record.get("size_bytes") or sha256_file(
            path
        ) != record.get("file_sha256"):
            _fail(f"Distribution output bytes differ from the manifest: {name}")
        parquet = pq.ParquetFile(path)
        schema = _arrow_schema(fields)
        if (
            parquet.metadata.num_rows != record.get("row_count")
            or not parquet.schema_arrow.equals(schema, check_metadata=False)
            or _schema_sha256(schema) != record.get("arrow_schema_sha256")
        ):
            _fail(f"Distribution output schema or row count differs: {name}")
        frame = pl.read_parquet(path)
        frames[name] = frame
        if frame.height:
            duplicates = (
                frame.group_by(_PRIMARY_KEYS[name])
                .agg(pl.len().alias("row_count"))
                .filter(pl.col("row_count") != 1)
            )
            if duplicates.height:
                _fail(f"Distribution output primary key is duplicated: {name}")
            if frame.get_column("distribution_run_id").unique().to_list() != [
                manifest["distribution_run_id"]
            ]:
                _fail(f"Distribution run identity differs in {name}")
            if frame.get_column("source_export_manifest_sha256").unique().to_list() != [
                source_digest
            ]:
                _fail(f"Source export identity differs in {name}")

    support = frames["recording_distribution_support.parquet"]
    sparse = frames["recording_distribution_nonzero_bins.parquet"]
    cohort = frames["cohort_distribution_bins.parquet"]
    _validate_json_identity_columns(
        support,
        json_column="group_key_json",
        digest_column="group_key_sha256",
        table_name="recording_distribution_support.parquet",
    )
    _validate_json_identity_columns(
        support,
        json_column="source_identity_key_json",
        digest_column="source_identity_key_sha256",
        table_name="recording_distribution_support.parquet",
    )
    _validate_json_identity_columns(
        cohort,
        json_column="group_key_json",
        digest_column="group_key_sha256",
        table_name="cohort_distribution_bins.parquet",
    )
    bad_support = support.filter(
        (pl.col("candidate_count") < 0)
        | (pl.col("valid_count") < 0)
        | (pl.col("excluded_count") < 0)
        | (
            pl.col("candidate_count")
            != pl.col("valid_count") + pl.col("excluded_count")
        )
        | (pl.col("denominator_weight") < 0)
        | ((pl.col("support_state") == "finite") & (pl.col("denominator_weight") <= 0))
        | (
            (pl.col("support_state") == "zero_denominator")
            & (pl.col("denominator_weight") != 0)
        )
        | ~pl.col("scope_id").is_in(SCOPE_ORDER)
        | ~pl.col("weighting_id").is_in(("event", "frame", "time"))
    )
    if bad_support.height:
        _fail("Distribution support denominator accounting is invalid")
    if sparse.height:
        if not set(
            sparse.get_column("histogram_recipe_sha256").unique().to_list()
        ).issubset(recipe_digests):
            _fail("A sparse bin references an unknown histogram recipe")
        bad_sparse = sparse.filter(
            (pl.col("bin_count") <= 0)
            | (pl.col("bin_weight") <= 0)
            | ~pl.col("fraction").is_between(0.0, 1.0, closed="both")
        )
        if bad_sparse.height:
            _fail("Distribution sparse bins contain an invalid count or fraction")
        totals = sparse.group_by("support_key_sha256").agg(
            pl.col("bin_count").sum().alias("bin_count_sum"),
            pl.col("bin_weight").sum().alias("bin_weight_sum"),
        )
        finite_support = support.filter(pl.col("denominator_weight") > 0).join(
            totals, on="support_key_sha256", how="left"
        )
        missing_bins = finite_support.filter(pl.col("bin_count_sum").is_null())
        bad_totals = finite_support.filter(
            (pl.col("bin_count_sum") != pl.col("valid_count"))
            | ((pl.col("bin_weight_sum") - pl.col("denominator_weight")).abs() > 1e-7)
        )
        if missing_bins.height or bad_totals.height:
            _fail("Sparse-bin totals differ from their recording support")
    if cohort.height:
        bad_counts = cohort.filter(
            (
                pl.col("finite_recording_count")
                + pl.col("excluded_zero_denominator_recording_count")
                != pl.col("contributor_recording_count")
            )
            | (
                pl.col("contributor_recording_count")
                + pl.col("noncontributor_recording_count")
                != pl.col("parent_recording_count")
            )
        )
        fraction_columns = (
            "pooled_fraction",
            "mean_recording_fraction",
            "median_recording_fraction",
            "minimum_recording_fraction",
            "p25_recording_fraction",
            "p75_recording_fraction",
            "maximum_recording_fraction",
        )
        bad_fraction = cohort.filter(
            pl.any_horizontal(
                [
                    pl.col(name).is_not_null()
                    & ~pl.col(name).is_between(0.0, 1.0, closed="both")
                    for name in fraction_columns
                ]
            )
        )
        if bad_counts.height or bad_fraction.height:
            _fail("Cohort-bin support or fraction accounting is invalid")


def write_validated_behavior_distributions(
    result: ValidatedBehaviorDistributionResult,
    output_dir: str | Path,
) -> Mapping[str, object]:
    """Atomically publish one immutable selector-ineligible distribution result."""

    target = Path(output_dir).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite distribution output: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    )
    try:
        assert temporary is not None
        outputs = (
            _write_parquet(
                temporary / "bout_observations.parquet",
                result.bout_observations,
                _BOUT_OBSERVATION_SCHEMA,
            ),
            _write_parquet(
                temporary / "inter_bout_interval_observations.parquet",
                result.inter_bout_interval_observations,
                _INTERVAL_OBSERVATION_SCHEMA,
            ),
            _write_parquet(
                temporary / "recording_distribution_support.parquet",
                result.recording_support,
                _SUPPORT_SCHEMA,
            ),
            _write_parquet(
                temporary / "recording_distribution_nonzero_bins.parquet",
                result.recording_nonzero_bins,
                _NONZERO_BIN_SCHEMA,
            ),
            _write_parquet(
                temporary / "cohort_distribution_bins.parquet",
                result.cohort_bins,
                _COHORT_BIN_SCHEMA,
            ),
        )
        epoch_receipts = [_plain(row) for row in result.epoch_child_receipts]
        body: dict[str, object] = {
            "schema_id": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "method_id": METHOD_ID,
            "status": STATUS,
            "distribution_run_id": result.config.distribution_run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_export": _plain(result.source_export),
            "cohort_summary": _plain(result.cohort_summary),
            "configuration": _plain(result.config.record),
            "source_queries": _plain(result.source_queries),
            "epoch_child_receipts": epoch_receipts,
            "epoch_child_receipts_sha256": canonical_json_sha256(epoch_receipts),
            "histogram_recipes": _plain(result.histogram_recipes),
            "chaser_appearance_dimension": _plain(result.chaser_appearance_dimension),
            "outputs": list(outputs),
            "trace_view_recipe": {
                "source_table": "provider_motion_samples",
                "recording_selection": "one_exact_recording_id",
                "frame_coordinate": "acquisition_frame_id",
                "time_coordinate": "time_s",
                "coordinate_choice_semantics": "display_only_same_exact_rows",
                "available_value_columns": [
                    spec.value_column
                    for spec in result.config.metric_specs
                    if spec.source_surface == "provider_motion_samples"
                ],
                "default_max_display_points": 5000,
                "decimation": "deterministic_even_index_endpoint_preserving_display_only_v1",
            },
            "scientific_claim": "exploratory_recording_normalized_distributions",
            "experimental_unit": "recording_id",
            "cohort_weighting": "equal_weight_per_finite_recording",
            "pooled_observation_statistic": "diagnostic_only",
            "selector_eligible": False,
            "production_authority": False,
            "selector_activation": False,
            "registry_update": False,
            "source_export_mutation": False,
            "recording_zarr_mutation": False,
            "implementation_module_sha256": sha256_file(Path(__file__)),
        }
        manifest = {**body, "record_sha256": canonical_json_sha256(body)}
        write_json_atomic(temporary / "manifest.json", manifest, overwrite=False)
        _validate_distribution_directory(temporary, manifest)
        os.replace(temporary, target)
        temporary = None
        return MappingProxyType(
            {**manifest, "manifest_path": str(target / "manifest.json")}
        )
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)


def read_validated_behavior_distributions(
    output_dir: str | Path,
) -> Mapping[str, object]:
    """Strictly reopen one exact distribution generation."""

    root = Path(output_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidatedBehaviorDistributionError(
            f"Cannot read distribution manifest: {manifest_path}"
        ) from exc
    if not isinstance(manifest, dict):
        _fail("Distribution manifest must be one JSON object")
    _validate_distribution_directory(root, manifest)
    return MappingProxyType({**manifest, "manifest_path": str(manifest_path)})


__all__ = [
    "METHOD_ID",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "STATUS",
    "ValidatedBehaviorDistributionConfig",
    "ValidatedBehaviorDistributionError",
    "ValidatedBehaviorDistributionResult",
    "compute_validated_behavior_distributions",
    "derive_bout_heading_values",
    "read_validated_behavior_distributions",
    "wrap_heading_delta_degrees",
    "write_validated_behavior_distributions",
]
