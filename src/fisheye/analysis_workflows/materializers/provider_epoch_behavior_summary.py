"""Publish selector-ineligible provider-bound stimulus-epoch behavior summaries.

The legacy mode binds one exact stimulus-epoch v2 candidate. The semantic-v2
successor additionally requires the strict protocol-semantic selection handle,
computes only ``chaser_pre``, ``chaser_training``, and ``chaser_post``, and
repeats the producer step identity on every scientific row. Both modes bind one
exact provider-motion run and one exact selector-ineligible swim-bout run.
Neither resolves or mutates a selector, claims physical stimulus-presentation
alignment, or infers spatial/chaser metrics from the motion source.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
from types import MappingProxyType, SimpleNamespace
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import zarr

from fisheye.analysis.chaser_distance_runs import ChaserDistanceWindow
from fisheye.analysis.chaser_epoch_behavior_summary import (
    _bout_heading_values,
    _finite_summary,
    _first_nonnegative_frame,
    _make_per_epoch_bout_histograms,
    _make_per_epoch_bouts,
    _make_per_epoch_inter_bout_interval_histograms,
    _structured_field,
    _window_time_mask,
)
from fisheye.analysis.swim_bout_frame_axis import canonical_frame_axis_sha256
from fisheye.analysis.swim_bout_io import (
    SwimBoutTables,
    load_exact_selector_ineligible_default_swim_bout_tables,
)
from fisheye.analysis_workflows.provider_analysis_bindings import (
    build_provider_analysis_offer,
    provider_motion_identity,
    temporal_selection_identity,
)
from fisheye.analysis_workflows.provider_analysis_offers import (
    ProviderRequirements,
    ScientificReadiness,
)
from fisheye.analysis_workflows.provider_track_motion_source_handle import (
    ProviderTrackMotionSourceHandle,
    load_provider_track_motion_source_handle,
)
from fisheye.analysis_workflows.protocol_semantic_chaser_selection_publication import (
    ProtocolSemanticChaserSelectionSourceHandle,
    load_protocol_semantic_chaser_selection_source_handle,
)
from fisheye.analysis_workflows.protocol_semantic_chaser_selection import (
    CHASER_WINDOW_ROLES,
)
from fisheye.analysis_workflows.resolved_epoch_selection import (
    ResolvedEpochSelection,
    resolve_exact_stimulus_epoch_selection,
)
from fisheye.shared.atomic_run_publisher import (
    AtomicRunPublishSpec,
    atomic_publish_run_group,
)
from fisheye.shared.coordinate_frame_record import array_values_sha256
from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.run_provenance import build_writer_run_provenance
from fisheye.shared.zarr.columnar import (
    load_structured_dataset,
    write_columnar_dataset,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.metadata_equivalence import (
    validate_direct_consolidated_subtree,
)
from fisheye.shared.zarr_helpers import (
    consolidate_metadata_capture_expected_warnings,
)
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_CONTRACT,
    RUN_COMPLETION_CONTRACT_ATTR,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_NAME_ATTR,
    RUN_STATUS_COMPLETE,
    mark_run_complete,
    mark_run_started,
    require_runs_parent,
)


PARENT_PATH = "analysis/stimulus_epoch_behavior_summary_runs"
SCHEMA_ID = "palette.stimulus_epoch_behavior_summary"
LEGACY_SCHEMA_VERSION = 1
SEMANTIC_SCHEMA_VERSION = 2
METHOD_ID = "provider_epoch_motion_bouts"
LEGACY_METHOD_VERSION = 1
SEMANTIC_METHOD_VERSION = 2
ANALYSIS_CLASS_ID = "stimulus_epoch_motion_bout_summary"
ANALYSIS_CLASS_VERSION = 1
LEGACY_MATERIALIZATION_SCHEMA_ID = (
    "palette.provider_epoch_behavior_summary_materialization.v1"
)
SEMANTIC_MATERIALIZATION_SCHEMA_ID = (
    "palette.provider_epoch_behavior_summary_materialization.v2"
)
LEGACY_PUBLISH_SCHEMA_ID = "palette.provider_epoch_behavior_summary_publish.v1"
SEMANTIC_PUBLISH_SCHEMA_ID = "palette.provider_epoch_behavior_summary_publish.v2"
MANIFEST_ATTR = "provider_epoch_behavior_summary_manifest"
MANIFEST_DIGEST_ATTR = "provider_epoch_behavior_summary_manifest_sha256"
# Preserve the original public constants for legacy callers. Semantic-v2 plans
# select their successor identifiers from the result binding mode.
MATERIALIZATION_SCHEMA_ID = LEGACY_MATERIALIZATION_SCHEMA_ID
PUBLISH_SCHEMA_ID = LEGACY_PUBLISH_SCHEMA_ID
LEGACY_EPOCH_BINDING_MODE = "exact_epoch_selection_v1"
SEMANTIC_EPOCH_BINDING_MODE = "protocol_semantic_selection_v2"
DEFAULT_SPEED_LEVEL = "filtered"
SUPPORTED_SPEED_LEVELS = ("raw", "filtered", "smoothed", "averaged")
_SELECTOR_ATTRS = (
    "latest",
    "latest_complete",
    "latest_pending",
    "authoritative_run",
)


class ProviderEpochBehaviorSummaryError(ValueError):
    """Raised when exact sources cannot produce one provider-bound summary."""


@dataclass(frozen=True)
class ProviderEpochBehaviorSummaryResult:
    recording_id: str
    run_name: str
    track_id: int
    speed_level: str
    fps: float
    schema_version: int
    method_version: int
    epoch_binding_mode: str
    epoch_selection: ResolvedEpochSelection
    protocol_semantic_selection_run_name: str | None
    protocol_semantic_selection_run_path: str | None
    protocol_semantic_selection_manifest_sha256: str | None
    motion_run_path: str
    motion_manifest_sha256: str
    motion_verification_digest: str
    swim_bout_run_name: str
    swim_bout_run_path: str
    swim_bout_lineage_hash: str
    swim_bout_frame_axis_sha256: str
    analysis_offer: Mapping[str, Any]
    analysis_offer_sha256: str
    source_bindings: Mapping[str, Any]
    per_epoch_fish: np.ndarray
    per_epoch_bouts: np.ndarray
    per_epoch_bout_histograms: np.ndarray
    per_epoch_inter_bout_interval_histograms: np.ndarray


@dataclass(frozen=True)
class ProviderEpochBehaviorSummaryPlan:
    source_zarr: Path
    scratch_root: Path
    local_zarr: Path
    run_name: str
    epoch_run_name: str
    protocol_semantic_selection_run_name: str | None
    motion_run_path: str
    swim_bout_run_name: str
    track_id: int
    speed_level: str
    result: ProviderEpochBehaviorSummaryResult
    parent_selector_attrs: Mapping[str, Any]

    @property
    def run_path(self) -> str:
        return f"{PARENT_PATH}/{self.run_name}"

    @property
    def local_run_path(self) -> Path:
        return self.local_zarr.joinpath(*self.run_path.split("/"))

    @property
    def target_run_path(self) -> Path:
        return self.source_zarr.joinpath(*self.run_path.split("/"))

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_id": (
                SEMANTIC_MATERIALIZATION_SCHEMA_ID
                if self.result.epoch_binding_mode == SEMANTIC_EPOCH_BINDING_MODE
                else LEGACY_MATERIALIZATION_SCHEMA_ID
            ),
            "source_zarr": str(self.source_zarr),
            "scratch_root": str(self.scratch_root),
            "local_zarr": str(self.local_zarr),
            "run_name": self.run_name,
            "run_path": self.run_path,
            "epoch_run_name": self.epoch_run_name,
            "protocol_semantic_selection_run_name": (
                self.protocol_semantic_selection_run_name
            ),
            "epoch_binding_mode": self.result.epoch_binding_mode,
            "schema_version": self.result.schema_version,
            "method_version": self.result.method_version,
            "motion_run_path": self.motion_run_path,
            "swim_bout_run_name": self.swim_bout_run_name,
            "track_id": self.track_id,
            "speed_level": self.speed_level,
            "analysis_offer_sha256": self.result.analysis_offer_sha256,
            "epoch_count": int(self.result.per_epoch_fish.shape[0]),
            "bout_count": int(self.result.per_epoch_bouts.shape[0]),
            "parent_selector_attrs": dict(self.parent_selector_attrs),
            "stage_selector_eligible": False,
        }


def _safe_name(value: object, *, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be one exact string.")
    if (
        not value
        or value != value.strip()
        or value in {".", "..", "latest", "latest_complete"}
        or "/" in value
        or "\\" in value
        or any(character.isspace() for character in value)
    ):
        raise ProviderEpochBehaviorSummaryError(
            f"{label} must be one exact non-selector child name."
        )
    return value


def _motion_run_path(value: object) -> str:
    from fisheye.analysis_workflows.materializers.provider_track_motion import (
        PROVIDER_TRACK_MOTION_PARENT_PATH,
    )

    if type(value) is not str:
        raise TypeError("motion_run must be one exact string.")
    prefix = f"{PROVIDER_TRACK_MOTION_PARENT_PATH}/"
    if value and "/" not in value:
        return f"{prefix}{_safe_name(value, label='motion run')}"
    if value.startswith(prefix):
        child = value[len(prefix) :]
        _safe_name(child, label="motion run")
        return value
    raise ProviderEpochBehaviorSummaryError(
        "motion_run must be one bare provider-motion child name or exact provider path."
    )


def _selector_snapshot(parent: Any | None) -> dict[str, Any]:
    if parent is None:
        return {}
    return {
        name: json_attr_safe(parent.attrs[name])
        for name in _SELECTOR_ATTRS
        if name in parent.attrs
    }


def _source_bindings_sha256(value: Mapping[str, Any]) -> str:
    """Hash the same JSON-safe source-binding form stored in Zarr attrs."""

    return canonical_json_sha256(json_attr_safe(dict(value)))


def _fps_from_motion(provider: ProviderTrackMotionSourceHandle) -> float:
    parameters = provider.computation_record.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ProviderEpochBehaviorSummaryError(
            "Provider-motion computation lacks an exact parameters object."
        )
    value = parameters.get("fps")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProviderEpochBehaviorSummaryError("Provider-motion FPS is absent.")
    fps = float(value)
    if not math.isfinite(fps) or fps <= 0:
        raise ProviderEpochBehaviorSummaryError("Provider-motion FPS is invalid.")
    return fps


def _track_slice(
    provider: ProviderTrackMotionSourceHandle,
    *,
    track_id: int,
) -> slice:
    ids = np.asarray(provider.track_ids, dtype=np.int64).reshape(-1)
    offsets = np.asarray(provider.track_row_offsets, dtype=np.int64).reshape(-1)
    matches = np.flatnonzero(ids == int(track_id))
    if matches.size != 1:
        raise ProviderEpochBehaviorSummaryError(
            f"Provider-motion track_id {track_id!r} did not resolve exactly once."
        )
    index = int(matches[0])
    if offsets.shape != (ids.shape[0] + 1,):
        raise ProviderEpochBehaviorSummaryError(
            "Provider-motion track offsets do not bind the track axis."
        )
    return slice(int(offsets[index]), int(offsets[index + 1]))


def _track_adapter(
    provider: ProviderTrackMotionSourceHandle,
    *,
    rows: slice,
) -> Any:
    def values(path: str, *, dtype: Any = np.float64) -> np.ndarray:
        try:
            return np.asarray(provider.array(path)[rows], dtype=dtype)
        except KeyError as exc:
            raise ProviderEpochBehaviorSummaryError(
                f"Provider-motion physical array {path!r} is required."
            ) from exc

    return SimpleNamespace(
        frame_indices=values("source_acquisition_frame_index", dtype=np.int64),
        linear_sample_valid=values("linear_sample_valid", dtype=bool),
        sample_valid=values("angular_sample_valid", dtype=bool),
        transition_valid=values("transition_valid", dtype=bool),
        speed_mm_by_level={
            level: values(f"speed_{level}_mm") for level in SUPPORTED_SPEED_LEVELS
        },
        frame_path_distance_mm_by_level={
            level: values(f"frame_path_distance_{level}_mm")
            for level in ("raw", "filtered", "smoothed")
        },
        smoothed_heading_degrees=values("smoothed_heading_degrees"),
        heading_degrees=values("heading_degrees"),
    )


def _windows(selection: ResolvedEpochSelection) -> tuple[ChaserDistanceWindow, ...]:
    return tuple(
        ChaserDistanceWindow(
            window_id=int(interval.window_id),
            label=str(interval.label),
            start_frame=int(interval.start_frame),
            end_frame=int(interval.end_frame) - 1,
            start_time_s=float(interval.start_time_s),
            end_time_s=float(interval.end_time_s),
            duration_s=float(interval.duration_s),
        )
        for interval in selection.intervals
    )


def _windows_from_protocol_semantic_binding(
    selection: ResolvedEpochSelection,
    binding: Mapping[str, Any],
) -> tuple[ChaserDistanceWindow, ...]:
    """Project the three strict semantic roles into motion/bout windows."""

    if binding.get("source_epoch_selection") != selection.selection_record:
        raise ProviderEpochBehaviorSummaryError(
            "Protocol-semantic selection and epoch authority identify different "
            "exact sources."
        )
    if (
        binding.get("selector_eligible") is not False
        or binding.get("production_authority") is not False
        or binding.get("roles") != list(CHASER_WINDOW_ROLES)
    ):
        raise ProviderEpochBehaviorSummaryError(
            "Protocol-semantic epoch authority is not the exact selector-ineligible "
            "chaser role set."
        )
    epoch_records = binding.get("position_suite_epochs")
    role_records = binding.get("semantic_role_bindings")
    if not isinstance(epoch_records, list) or not isinstance(role_records, list):
        raise ProviderEpochBehaviorSummaryError(
            "Protocol-semantic source binding lacks its exact role records."
        )
    if canonical_json_sha256(epoch_records) != binding.get(
        "position_suite_epochs_sha256"
    ) or canonical_json_sha256(role_records) != binding.get(
        "semantic_role_bindings_sha256"
    ):
        raise ProviderEpochBehaviorSummaryError(
            "Protocol-semantic role-record digest is stale."
        )
    observed_roles = tuple(record.get("analysis_role") for record in epoch_records)
    if (
        observed_roles != CHASER_WINDOW_ROLES
        or tuple(record.get("analysis_role") for record in role_records)
        != CHASER_WINDOW_ROLES
    ):
        raise ProviderEpochBehaviorSummaryError(
            "Protocol-semantic epoch roles must be exact chaser pre/training/post."
        )
    source_by_id = {
        int(interval.window_id): interval for interval in selection.intervals
    }
    semantic_by_role = {str(record["analysis_role"]): record for record in role_records}
    windows: list[ChaserDistanceWindow] = []
    for record in epoch_records:
        role = str(record["analysis_role"])
        source_id = record.get("window_id")
        start = record.get("start_frame")
        end = record.get("end_frame_exclusive")
        if (
            type(source_id) is not int
            or type(start) is not int
            or type(end) is not int
            or end <= start
            or source_id not in source_by_id
        ):
            raise ProviderEpochBehaviorSummaryError(
                f"Protocol-semantic epoch {role!r} has invalid selected bounds."
            )
        source = source_by_id[source_id]
        semantic = semantic_by_role[role]
        if (
            record.get("source_label") != source.label
            or record.get("source_interval_sha256") != source.source_interval_digest
            or semantic.get("source_window_id") != source_id
            or semantic.get("source_interval_sha256") != source.source_interval_digest
            or semantic.get("selected_start_frame") != start
            or semantic.get("selected_end_frame_exclusive") != end
            or start < source.start_frame
            or end > source.end_frame
        ):
            raise ProviderEpochBehaviorSummaryError(
                f"Protocol-semantic epoch {role!r} differs from its exact source interval."
            )
        duration = (end - start) / selection.fps
        windows.append(
            ChaserDistanceWindow(
                window_id=source_id,
                label=role,
                start_frame=start,
                end_frame=end - 1,
                start_time_s=start / selection.fps,
                end_time_s=end / selection.fps,
                duration_s=duration,
            )
        )
    for previous, current in zip(windows, windows[1:]):
        if current.start_frame <= previous.end_frame:
            raise ProviderEpochBehaviorSummaryError(
                "Protocol-semantic motion/bout windows overlap or are unordered."
            )
    return tuple(windows)


def _swim_bout_binding(
    tables: SwimBoutTables,
    *,
    provider: ProviderTrackMotionSourceHandle,
    rows: slice,
    track_id: int,
) -> tuple[dict[str, Any], str, str]:
    attrs = dict(tables.run_attrs)
    if attrs.get("source_track_kinematics_scope") != "provider":
        raise ProviderEpochBehaviorSummaryError(
            "Swim-bout candidate is not bound to provider motion."
        )
    if attrs.get("source_track_kinematics_run") != provider.run_name:
        raise ProviderEpochBehaviorSummaryError(
            "Swim-bout and provider-motion run identities disagree."
        )
    if int(attrs.get("track_id", -1)) != int(track_id):
        raise ProviderEpochBehaviorSummaryError(
            "Swim-bout and provider-motion track identities disagree."
        )
    if (
        attrs.get("source_track_motion_manifest_sha256")
        != provider.provider_manifest_sha256
    ):
        raise ProviderEpochBehaviorSummaryError(
            "Swim-bout provider-motion manifest binding is stale."
        )
    authority = attrs.get("source_track_motion_authority")
    if not isinstance(authority, Mapping):
        raise ProviderEpochBehaviorSummaryError(
            "Swim-bout provider read authority is absent."
        )
    expected_authority = {
        "motion_manifest_sha256": provider.provider_manifest_sha256,
        "provider_verification_digest": provider.verification_digest,
        "track_id": int(track_id),
        "track_row_start": int(rows.start or 0),
        "track_row_stop": int(rows.stop or 0),
    }
    for key, expected in expected_authority.items():
        if authority.get(key) != expected:
            raise ProviderEpochBehaviorSummaryError(
                f"Swim-bout provider read authority differs at {key!r}."
            )
    frame_contract = attrs.get("frame_axis_contract")
    if not isinstance(frame_contract, Mapping):
        raise ProviderEpochBehaviorSummaryError(
            "Swim-bout frame-axis contract is absent."
        )
    frames = np.asarray(provider.source_acquisition_frame_index[rows], dtype=np.int64)
    frame_sha256 = canonical_frame_axis_sha256(frames)
    if frame_contract.get("content_sha256") != frame_sha256:
        raise ProviderEpochBehaviorSummaryError(
            "Swim-bout frame axis differs from the selected provider track."
        )
    lineage_hash = attrs.get("lineage_hash")
    if type(lineage_hash) is not str or len(lineage_hash) != 64:
        raise ProviderEpochBehaviorSummaryError(
            "Swim-bout candidate lacks its exact lineage digest."
        )
    binding = {
        "schema_id": "palette.selector_ineligible_swim_bout_binding.v1",
        "run_name": tables.run_name,
        "run_path": tables.run_path,
        "lineage_hash": lineage_hash,
        "frame_axis_sha256": frame_sha256,
        "source_track_motion_manifest_sha256": provider.provider_manifest_sha256,
        "source_track_motion_verification_digest": provider.verification_digest,
        "track_id": int(track_id),
        "track_row_start": int(rows.start or 0),
        "track_row_stop": int(rows.stop or 0),
        "default_candidate_id": int(tables.candidate.candidate_id),
        "default_signal_id": int(tables.signal.signal_id),
        "default_signal_level": str(tables.signal.speed_level),
    }
    binding["sha256"] = canonical_json_sha256(binding)
    return binding, lineage_hash, frame_sha256


def _make_per_epoch_fish(
    *,
    windows: Sequence[ChaserDistanceWindow],
    track: Any,
    track_id: int,
    speed_level: str,
    swim_tables: SwimBoutTables,
    fps: float,
) -> np.ndarray:
    dtype = np.dtype(
        [
            ("track_id", np.int64),
            ("window_id", np.int32),
            ("window_index", np.int32),
            ("window_label", "S96"),
            ("start_frame", np.int64),
            ("end_frame", np.int64),
            ("start_time_s", np.float64),
            ("end_time_s", np.float64),
            ("duration_s", np.float64),
            ("total_span_frames", np.int64),
            ("provider_sample_count", np.int64),
            ("valid_tracked_frame_count", np.int64),
            ("missing_frame_count", np.int64),
            ("tracking_dropout_fraction", np.float64),
            ("valid_tracked_duration_s", np.float64),
            ("motion_valid_sample_count", np.int64),
            ("speed_sample_count", np.int64),
            ("mean_speed_mm_s", np.float64),
            ("median_speed_mm_s", np.float64),
            ("p05_speed_mm_s", np.float64),
            ("p95_speed_mm_s", np.float64),
            ("max_speed_mm_s", np.float64),
            ("total_path_mm", np.float64),
            ("bout_count", np.int64),
            ("bout_rate_per_min", np.float64),
            ("median_bout_duration_s", np.float64),
            ("mean_bout_duration_s", np.float64),
            ("median_bout_path_length_mm", np.float64),
            ("mean_bout_path_length_mm", np.float64),
            ("bout_heading_sample_count", np.int64),
            ("mean_bout_net_heading_change_deg", np.float64),
            ("median_bout_net_heading_change_deg", np.float64),
            ("mean_abs_bout_net_heading_change_deg", np.float64),
            ("median_abs_bout_net_heading_change_deg", np.float64),
            ("mean_bout_heading_path_deg", np.float64),
            ("median_bout_heading_path_deg", np.float64),
            ("inter_bout_interval_count", np.int64),
            ("mean_inter_bout_interval_s", np.float64),
            ("median_inter_bout_interval_s", np.float64),
            ("p05_inter_bout_interval_s", np.float64),
            ("p95_inter_bout_interval_s", np.float64),
            ("inter_bout_interval_rate_per_min", np.float64),
            ("rate_denominator", "S64"),
            ("motion_validity_rule", "S64"),
        ]
    )
    out = np.zeros(len(windows), dtype=dtype)
    for name in out.dtype.names or ():
        if out.dtype[name].kind == "f":
            out[name] = np.nan
    frames = np.asarray(track.frame_indices, dtype=np.int64)
    linear_valid = np.asarray(track.linear_sample_valid, dtype=bool)
    transition_valid = np.asarray(track.transition_valid, dtype=bool)
    speed = np.asarray(track.speed_mm_by_level[speed_level], dtype=np.float64)
    path_source = (
        speed_level if speed_level in {"raw", "filtered", "smoothed"} else "smoothed"
    )
    path = np.asarray(
        track.frame_path_distance_mm_by_level[path_source], dtype=np.float64
    )
    if not (
        frames.shape
        == linear_valid.shape
        == transition_valid.shape
        == speed.shape
        == path.shape
    ):
        raise ProviderEpochBehaviorSummaryError(
            "Provider-motion arrays do not share one exact track row axis."
        )
    bouts = swim_tables.bouts
    intervals = swim_tables.inter_bout_intervals
    bout_event_frame = _first_nonnegative_frame(
        bouts, "peak_frame", "core_start_frame", "start_frame"
    )
    bout_time_s = _structured_field(bouts, "peak_time_s", "start_time_s", "start_s")
    bout_duration_s = _structured_field(
        bouts, "duration_s", "observed_duration_s", "elapsed_duration_s"
    )
    bout_path_mm = _structured_field(bouts, "path_length_mm")
    interval_s = _structured_field(intervals, "interval_s")
    interval_prev_end_frame = _structured_field(intervals, "prev_end_frame")
    interval_next_start_frame = _structured_field(intervals, "next_start_frame")
    interval_prev_end_s = _structured_field(intervals, "prev_end_time_s")
    interval_next_start_s = _structured_field(intervals, "next_start_time_s")
    interval_valid = _structured_field(intervals, "valid")
    bout_heading, bout_heading_path = _bout_heading_values(
        bouts=bouts,
        track=track,
        fps=fps,
    )

    for index, window in enumerate(windows):
        membership = (frames >= window.start_frame) & (frames <= window.end_frame)
        valid = membership & linear_valid
        motion_valid = valid & transition_valid
        total_span = max(0, int(window.end_frame) - int(window.start_frame) + 1)
        provider_count = int(np.count_nonzero(membership))
        valid_count = int(np.count_nonzero(valid))
        missing_count = max(0, total_span - valid_count)
        valid_duration = float(valid_count) / fps
        speed_summary = _finite_summary(speed[motion_valid])
        finite_path = path[motion_valid]
        total_path = (
            float(np.nansum(finite_path[np.isfinite(finite_path)]))
            if finite_path.size
            else np.nan
        )
        if bout_event_frame is not None:
            event_frame = np.asarray(bout_event_frame, dtype=np.int64)
            bout_mask = (event_frame >= int(window.start_frame)) & (
                event_frame <= int(window.end_frame)
            )
        else:
            bout_mask = _window_time_mask(
                bout_time_s,
                start_s=window.start_time_s,
                end_s=window.end_time_s,
            )
        bout_count = int(np.count_nonzero(bout_mask))
        bout_rate = (
            float(bout_count) / (valid_duration / 60.0)
            if valid_duration > 0
            else np.nan
        )
        duration_summary = _finite_summary(
            np.asarray(bout_duration_s, dtype=np.float64)[bout_mask]
            if bout_duration_s is not None
            else np.asarray([])
        )
        path_summary = _finite_summary(
            np.asarray(bout_path_mm, dtype=np.float64)[bout_mask]
            if bout_path_mm is not None
            else np.asarray([])
        )
        heading_values = (
            bout_heading[bout_mask]
            if bout_mask.shape[0] == bout_heading.shape[0]
            else np.asarray([])
        )
        heading_summary = _finite_summary(heading_values)
        abs_heading_summary = _finite_summary(np.abs(heading_values))
        heading_path_values = (
            bout_heading_path[bout_mask]
            if bout_mask.shape[0] == bout_heading_path.shape[0]
            else np.asarray([])
        )
        heading_path_summary = _finite_summary(heading_path_values)
        if (
            interval_s is not None
            and interval_prev_end_frame is not None
            and interval_next_start_frame is not None
        ):
            interval_values = np.asarray(interval_s, dtype=np.float64)
            interval_mask = (
                np.isfinite(interval_values)
                & (
                    np.asarray(interval_prev_end_frame, dtype=np.int64)
                    >= int(window.start_frame)
                )
                & (
                    np.asarray(interval_next_start_frame, dtype=np.int64)
                    <= int(window.end_frame)
                )
            )
            if interval_valid is not None:
                interval_mask &= np.asarray(interval_valid, dtype=bool)
        elif (
            interval_s is not None
            and interval_prev_end_s is not None
            and interval_next_start_s is not None
        ):
            interval_values = np.asarray(interval_s, dtype=np.float64)
            interval_mask = (
                np.isfinite(interval_values)
                & (
                    np.asarray(interval_prev_end_s, dtype=np.float64)
                    >= float(window.start_time_s)
                )
                & (
                    np.asarray(interval_next_start_s, dtype=np.float64)
                    <= float(window.end_time_s)
                )
            )
            if interval_valid is not None:
                interval_mask &= np.asarray(interval_valid, dtype=bool)
        else:
            interval_values = np.asarray([], dtype=np.float64)
            interval_mask = np.zeros(0, dtype=bool)
        interval_summary = _finite_summary(
            interval_values[interval_mask] if interval_values.size else np.asarray([])
        )
        interval_rate = (
            float(interval_summary[0]) / (valid_duration / 60.0)
            if valid_duration > 0
            else np.nan
        )
        out[index] = (
            int(track_id),
            int(window.window_id),
            int(index),
            str(window.label).encode("utf-8", "ignore")[:95],
            int(window.start_frame),
            int(window.end_frame),
            float(window.start_time_s),
            float(window.end_time_s),
            float(window.duration_s),
            int(total_span),
            int(provider_count),
            int(valid_count),
            int(missing_count),
            float(missing_count / total_span) if total_span else np.nan,
            float(valid_duration),
            int(np.count_nonzero(motion_valid)),
            int(speed_summary[0]),
            float(speed_summary[1]),
            float(speed_summary[2]),
            float(speed_summary[3]),
            float(speed_summary[4]),
            float(speed_summary[5]),
            float(total_path),
            int(bout_count),
            float(bout_rate),
            float(duration_summary[2]),
            float(duration_summary[1]),
            float(path_summary[2]),
            float(path_summary[1]),
            int(heading_summary[0]),
            float(heading_summary[1]),
            float(heading_summary[2]),
            float(abs_heading_summary[1]),
            float(abs_heading_summary[2]),
            float(heading_path_summary[1]),
            float(heading_path_summary[2]),
            int(interval_summary[0]),
            float(interval_summary[1]),
            float(interval_summary[2]),
            float(interval_summary[3]),
            float(interval_summary[4]),
            float(interval_rate),
            b"valid_tracked_duration_s",
            b"linear_sample_valid_and_transition_valid",
        )
    return out


def _bind_track_id(records: np.ndarray, *, track_id: int) -> np.ndarray:
    """Add the selected track identity to a structured fact table."""

    source = np.asarray(records)
    if source.dtype.names is None:
        raise ProviderEpochBehaviorSummaryError(
            "Per-epoch bout facts must be one structured table."
        )
    dtype = np.dtype(
        [("track_id", np.int64)]
        + [(name, source.dtype.fields[name][0]) for name in source.dtype.names]
    )
    result = np.empty(source.shape, dtype=dtype)
    result["track_id"] = int(track_id)
    for name in source.dtype.names:
        result[name] = source[name]
    return result


def _array_declarations(
    result: ProviderEpochBehaviorSummaryResult,
) -> list[dict[str, Any]]:
    """Declare every physical column written by the immutable summary."""

    tables = {
        "per_epoch_fish": result.per_epoch_fish,
        "per_epoch_bouts": result.per_epoch_bouts,
        "per_epoch_bout_histograms": result.per_epoch_bout_histograms,
        "per_epoch_inter_bout_interval_histograms": (
            result.per_epoch_inter_bout_interval_histograms
        ),
    }
    declarations: list[dict[str, Any]] = []
    for table_name, table in tables.items():
        if table.dtype.names is None:
            raise ProviderEpochBehaviorSummaryError(
                f"{table_name} must be one structured table."
            )
        for field_name in table.dtype.names:
            values = np.asarray(table[field_name])
            declarations.append(
                {
                    "path": f"{table_name}/{field_name}",
                    "dtype": values.dtype.str,
                    "shape": list(values.shape),
                    "content_sha256": array_values_sha256(values),
                }
            )
    return declarations


def _summary_manifest(
    result: ProviderEpochBehaviorSummaryResult,
    *,
    run_path: str,
    source_refs: Mapping[str, Any],
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "scientific_schema": {
            "schema_id": SCHEMA_ID,
            "schema_version": result.schema_version,
        },
        "method_id": METHOD_ID,
        "method_version": result.method_version,
        "epoch_binding_mode": result.epoch_binding_mode,
        "run_path": run_path,
        "recording_id": result.recording_id,
        "dimensions": {
            "n_epoch_rows": int(result.per_epoch_fish.shape[0]),
            "n_bout_rows": int(result.per_epoch_bouts.shape[0]),
            "n_bout_histogram_rows": int(result.per_epoch_bout_histograms.shape[0]),
            "n_inter_bout_interval_histogram_rows": int(
                result.per_epoch_inter_bout_interval_histograms.shape[0]
            ),
        },
        "sources": dict(source_refs),
        "parameters": dict(parameters),
        "analysis_offer_sha256": result.analysis_offer_sha256,
        "array_declarations": _array_declarations(result),
        "selector_eligible": False,
        "selection": "none",
        "production_authority": False,
        "registry_update": False,
    }
    return {**body, "payload_digest": canonical_json_sha256(body)}


def _bind_protocol_semantic_row_identity(
    records: np.ndarray,
    *,
    binding: Mapping[str, Any],
) -> np.ndarray:
    """Add producer step identity to every semantic epoch summary row."""

    source = np.asarray(records)
    if (
        source.ndim != 1
        or source.dtype.names is None
        or "window_id" not in source.dtype.names
    ):
        raise ProviderEpochBehaviorSummaryError(
            "Protocol-semantic summary rows require one structured window_id axis."
        )
    added_names = (
        "analysis_role",
        "source_interval_sha256",
        "protocol_semantic_hash",
        "protocol_semantic_step_index",
        "protocol_semantic_step_ref",
    )
    if any(name in source.dtype.names for name in added_names):
        raise ProviderEpochBehaviorSummaryError(
            "Protocol-semantic row identity fields already exist."
        )
    role_records = binding.get("semantic_role_bindings")
    if not isinstance(role_records, list):
        raise ProviderEpochBehaviorSummaryError(
            "Protocol-semantic source binding lacks role identities."
        )
    by_window: dict[int, Mapping[str, Any]] = {}
    for record in role_records:
        if not isinstance(record, Mapping):
            raise ProviderEpochBehaviorSummaryError(
                "Protocol-semantic role identity is malformed."
            )
        window_id = record.get("source_window_id")
        if type(window_id) is not int or window_id in by_window:
            raise ProviderEpochBehaviorSummaryError(
                "Protocol-semantic role identities do not bind unique windows."
            )
        by_window[window_id] = record
    dtype = np.dtype(
        [(name, source.dtype.fields[name][0]) for name in source.dtype.names]
        + [
            ("analysis_role", "S32"),
            ("source_interval_sha256", "S64"),
            ("protocol_semantic_hash", "S72"),
            ("protocol_semantic_step_index", np.int32),
            ("protocol_semantic_step_ref", "S112"),
        ]
    )
    result = np.empty(source.shape, dtype=dtype)
    for name in source.dtype.names:
        result[name] = source[name]
    for row_index, window_value in enumerate(source["window_id"]):
        window_id = int(window_value)
        record = by_window.get(window_id)
        if record is None:
            raise ProviderEpochBehaviorSummaryError(
                f"Summary row references non-semantic window_id={window_id}."
            )
        role = record.get("analysis_role")
        source_interval_sha256 = record.get("source_interval_sha256")
        semantic_hash = record.get("protocol_semantic_hash")
        step_index = record.get("protocol_semantic_step_index")
        step_ref = record.get("protocol_semantic_step_ref")
        expected_step_ref = (
            f"protocol_semantic_snapshot@recipe.steps[{step_index}]"
            if type(step_index) is int
            else None
        )
        if (
            type(role) is not str
            or role not in CHASER_WINDOW_ROLES
            or type(source_interval_sha256) is not str
            or len(source_interval_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in source_interval_sha256
            )
            or type(semantic_hash) is not str
            or len(semantic_hash) != 71
            or not semantic_hash.startswith("sha256:")
            or any(
                character not in "0123456789abcdef" for character in semantic_hash[7:]
            )
            or semantic_hash != binding.get("protocol_semantic_hash")
            or type(step_index) is not int
            or step_index < 0
            or type(step_ref) is not str
            or step_ref != expected_step_ref
            or len(role.encode("utf-8")) > 32
            or len(step_ref.encode("utf-8")) > 112
        ):
            raise ProviderEpochBehaviorSummaryError(
                "Protocol-semantic row identity is incomplete."
            )
        result["analysis_role"][row_index] = role.encode("utf-8")
        result["source_interval_sha256"][row_index] = source_interval_sha256.encode(
            "utf-8"
        )
        result["protocol_semantic_hash"][row_index] = semantic_hash.encode("utf-8")
        result["protocol_semantic_step_index"][row_index] = step_index
        result["protocol_semantic_step_ref"][row_index] = step_ref.encode("utf-8")
    return result


def _compute_result(
    source_zarr: Path,
    *,
    run_name: str,
    epoch_run_name: str,
    protocol_semantic_selection_run_name: str | None,
    motion_run_path: str,
    swim_bout_run_name: str,
    track_id: int,
    speed_level: str,
) -> ProviderEpochBehaviorSummaryResult:
    selection = resolve_exact_stimulus_epoch_selection(
        source_zarr,
        run_name=epoch_run_name,
    )
    provider = load_provider_track_motion_source_handle(
        source_zarr,
        motion_run_path,
        use_consolidated=True,
        require_authoritative_timing=False,
    )
    rows = _track_slice(provider, track_id=track_id)
    root = open_zarr_root(source_zarr, mode="r", use_consolidated=True)
    swim_tables = load_exact_selector_ineligible_default_swim_bout_tables(
        root,
        run_name=swim_bout_run_name,
    )
    swim_binding, swim_lineage, frame_sha256 = _swim_bout_binding(
        swim_tables,
        provider=provider,
        rows=rows,
        track_id=track_id,
    )
    fps = _fps_from_motion(provider)
    if not math.isclose(fps, selection.fps, rel_tol=0.0, abs_tol=1e-12):
        raise ProviderEpochBehaviorSummaryError(
            "Stimulus-epoch and provider-motion FPS disagree."
        )
    bout_fps = swim_tables.run_attrs.get("fps")
    if not isinstance(bout_fps, (int, float)) or not math.isclose(
        float(bout_fps), fps, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ProviderEpochBehaviorSummaryError(
            "Swim-bout and provider-motion FPS disagree."
        )
    motion_identity = provider_motion_identity(provider)
    temporal_identity = temporal_selection_identity(selection)
    semantic_handle: ProtocolSemanticChaserSelectionSourceHandle | None = None
    semantic_binding: dict[str, Any] | None = None
    if protocol_semantic_selection_run_name is not None:
        semantic_handle = load_protocol_semantic_chaser_selection_source_handle(
            source_zarr,
            run_name=protocol_semantic_selection_run_name,
            expected_recording_id=str(temporal_identity.recording_id),
            use_consolidated=True,
            deep_audit=True,
        )
        semantic_binding = semantic_handle.source_binding()
        windows = _windows_from_protocol_semantic_binding(
            selection,
            semantic_binding,
        )
        schema_version = SEMANTIC_SCHEMA_VERSION
        method_version = SEMANTIC_METHOD_VERSION
        epoch_binding_mode = SEMANTIC_EPOCH_BINDING_MODE
    else:
        windows = _windows(selection)
        schema_version = LEGACY_SCHEMA_VERSION
        method_version = LEGACY_METHOD_VERSION
        epoch_binding_mode = LEGACY_EPOCH_BINDING_MODE
    offer = build_provider_analysis_offer(
        analysis_class_id=ANALYSIS_CLASS_ID,
        analysis_class_version=ANALYSIS_CLASS_VERSION,
        computation_id=METHOD_ID,
        computation_version=method_version,
        temporal_selection=temporal_identity,
        provider_requirements=ProviderRequirements(motion=motion_identity),
    )
    if offer.scientific_readiness is not ScientificReadiness.READY:
        raise ProviderEpochBehaviorSummaryError(
            "Provider epoch behavior summary is not scientifically ready: "
            f"{offer.scientific_readiness.value}."
        )
    offer_record = offer.record
    offer_sha256 = offer.sha256
    track = _track_adapter(provider, rows=rows)
    per_epoch_fish = _make_per_epoch_fish(
        windows=windows,
        track=track,
        track_id=track_id,
        speed_level=speed_level,
        swim_tables=swim_tables,
        fps=fps,
    )
    per_epoch_bouts = _bind_track_id(
        _make_per_epoch_bouts(
            windows=windows,
            swim_tables=swim_tables,
            track=track,
            fps=fps,
        ),
        track_id=track_id,
    )
    per_epoch_bout_histograms = _make_per_epoch_bout_histograms(
        windows=windows,
        per_epoch_bouts=per_epoch_bouts,
    )
    per_epoch_inter_bout_interval_histograms = (
        _make_per_epoch_inter_bout_interval_histograms(
            windows=windows,
            swim_tables=swim_tables,
        )
    )
    if semantic_binding is not None:
        per_epoch_fish = _bind_protocol_semantic_row_identity(
            per_epoch_fish,
            binding=semantic_binding,
        )
        per_epoch_bouts = _bind_protocol_semantic_row_identity(
            per_epoch_bouts,
            binding=semantic_binding,
        )
        per_epoch_bout_histograms = _bind_protocol_semantic_row_identity(
            per_epoch_bout_histograms,
            binding=semantic_binding,
        )
        per_epoch_inter_bout_interval_histograms = _bind_protocol_semantic_row_identity(
            per_epoch_inter_bout_interval_histograms,
            binding=semantic_binding,
        )
    source_bindings = {
        "epoch_binding_mode": epoch_binding_mode,
        "epoch_selection": {
            "record": selection.selection_record,
            "sha256": selection.selection_digest,
        },
        "provider_motion": {
            "run_path": provider.run_path,
            "manifest_sha256": provider.provider_manifest_sha256,
            "verification_digest": provider.verification_digest,
            "track_id": int(track_id),
            "track_row_start": int(rows.start or 0),
            "track_row_stop": int(rows.stop or 0),
        },
        "swim_bouts": swim_binding,
        **(
            {"protocol_semantic_selection": semantic_binding}
            if semantic_binding is not None
            else {}
        ),
    }
    return ProviderEpochBehaviorSummaryResult(
        recording_id=str(temporal_identity.recording_id),
        run_name=run_name,
        track_id=int(track_id),
        speed_level=speed_level,
        fps=fps,
        schema_version=schema_version,
        method_version=method_version,
        epoch_binding_mode=epoch_binding_mode,
        epoch_selection=selection,
        protocol_semantic_selection_run_name=(
            semantic_handle.run_name if semantic_handle is not None else None
        ),
        protocol_semantic_selection_run_path=(
            semantic_handle.run_path if semantic_handle is not None else None
        ),
        protocol_semantic_selection_manifest_sha256=(
            semantic_handle.manifest_sha256 if semantic_handle is not None else None
        ),
        motion_run_path=provider.run_path,
        motion_manifest_sha256=provider.provider_manifest_sha256,
        motion_verification_digest=provider.verification_digest,
        swim_bout_run_name=swim_tables.run_name,
        swim_bout_run_path=swim_tables.run_path,
        swim_bout_lineage_hash=swim_lineage,
        swim_bout_frame_axis_sha256=frame_sha256,
        analysis_offer=MappingProxyType(offer_record),
        analysis_offer_sha256=offer_sha256,
        source_bindings=MappingProxyType(source_bindings),
        per_epoch_fish=per_epoch_fish,
        per_epoch_bouts=per_epoch_bouts,
        per_epoch_bout_histograms=per_epoch_bout_histograms,
        per_epoch_inter_bout_interval_histograms=(
            per_epoch_inter_bout_interval_histograms
        ),
    )


def build_provider_epoch_behavior_summary_plan(
    source_zarr: str | Path,
    *,
    scratch_root: str | Path,
    run_name: str,
    epoch_run_name: str,
    protocol_semantic_selection_run_name: str | None = None,
    motion_run: str,
    swim_bout_run_name: str,
    track_id: int = 0,
    speed_level: str = DEFAULT_SPEED_LEVEL,
) -> ProviderEpochBehaviorSummaryPlan:
    source = Path(source_zarr).expanduser().resolve()
    scratch = Path(scratch_root).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Analysis Zarr does not exist: {source}.")
    if scratch == source or scratch.is_relative_to(source):
        raise ProviderEpochBehaviorSummaryError(
            "Scratch root must be outside the authoritative archive."
        )
    run = _safe_name(run_name, label="summary run")
    epoch = _safe_name(epoch_run_name, label="epoch run")
    semantic = (
        _safe_name(
            protocol_semantic_selection_run_name,
            label="protocol-semantic selection run",
        )
        if protocol_semantic_selection_run_name is not None
        else None
    )
    bout = _safe_name(swim_bout_run_name, label="swim-bout run")
    motion_path = _motion_run_path(motion_run)
    if type(track_id) is not int or track_id < 0:
        raise ProviderEpochBehaviorSummaryError(
            "track_id must be one nonnegative exact integer."
        )
    level = str(speed_level).strip().replace("speed_", "", 1)
    if level not in SUPPORTED_SPEED_LEVELS:
        raise ProviderEpochBehaviorSummaryError(
            f"speed_level must be one of {SUPPORTED_SPEED_LEVELS!r}."
        )
    if semantic is not None and level == "raw":
        raise ProviderEpochBehaviorSummaryError(
            "Semantic-v2 publication rejects raw speed; choose filtered, "
            "smoothed, or averaged physical speed."
        )
    local = scratch / "provider-epoch-behavior-summary.zarr"
    target = source.joinpath(*f"{PARENT_PATH}/{run}".split("/"))
    if local.exists() or target.exists():
        raise FileExistsError("Provider epoch summary output path is occupied.")
    root = open_zarr_root(source, mode="r", use_consolidated=False)
    parent = root.get(PARENT_PATH)
    result = _compute_result(
        source,
        run_name=run,
        epoch_run_name=epoch,
        protocol_semantic_selection_run_name=semantic,
        motion_run_path=motion_path,
        swim_bout_run_name=bout,
        track_id=track_id,
        speed_level=level,
    )
    return ProviderEpochBehaviorSummaryPlan(
        source_zarr=source,
        scratch_root=scratch,
        local_zarr=local,
        run_name=run,
        epoch_run_name=epoch,
        protocol_semantic_selection_run_name=semantic,
        motion_run_path=motion_path,
        swim_bout_run_name=bout,
        track_id=track_id,
        speed_level=level,
        result=result,
        parent_selector_attrs=MappingProxyType(_selector_snapshot(parent)),
    )


def _write_local(plan: ProviderEpochBehaviorSummaryPlan) -> None:
    plan.scratch_root.mkdir(parents=True, exist_ok=False)
    root = open_zarr_root(plan.local_zarr, mode="w", use_consolidated=False)
    analysis = root.require_group("analysis")
    parent = require_runs_parent(analysis, PARENT_PATH.split("/", 1)[1])
    run = parent.create_group(plan.run_name)
    mark_run_started(run, run_name=plan.run_name, stage=METHOD_ID)
    run.attrs["stage_selector_eligible"] = False
    write_columnar_dataset(
        run,
        "per_epoch_fish",
        plan.result.per_epoch_fish,
        {
            "row_axis": "track_x_stimulus_epoch",
            "unit_of_analysis": "track_epoch",
            "epoch_binding_mode": plan.result.epoch_binding_mode,
            "rate_denominator": "valid_tracked_duration_s",
            "motion_validity_rule": "linear_sample_valid_and_transition_valid",
        },
    )
    write_columnar_dataset(
        run,
        "per_epoch_bouts",
        plan.result.per_epoch_bouts,
        {
            "row_axis": "stimulus_epoch_x_swim_bout",
            "unit_of_analysis": "swim_bout",
            "epoch_binding_mode": plan.result.epoch_binding_mode,
            "epoch_assignment_rule": "first nonnegative peak/core_start/start frame within inclusive epoch",
        },
    )
    write_columnar_dataset(
        run,
        "per_epoch_bout_histograms",
        plan.result.per_epoch_bout_histograms,
        {
            "row_axis": "stimulus_epoch_x_bout_metric_x_bin",
            "epoch_binding_mode": plan.result.epoch_binding_mode,
            "source_table": "per_epoch_bouts",
        },
    )
    write_columnar_dataset(
        run,
        "per_epoch_inter_bout_interval_histograms",
        plan.result.per_epoch_inter_bout_interval_histograms,
        {
            "row_axis": "stimulus_epoch_x_inter_bout_interval_bin",
            "epoch_binding_mode": plan.result.epoch_binding_mode,
            "source_table": "source_swim_bout_run/inter_bout_intervals",
        },
    )
    source_refs = json_attr_safe(dict(plan.result.source_bindings))
    source_refs_sha256 = _source_bindings_sha256(plan.result.source_bindings)
    parameters = {
        "track_id": plan.track_id,
        "epoch_binding_mode": plan.result.epoch_binding_mode,
        "physical_speed_level": plan.speed_level,
        "bout_signal_selection": "source_swim_bout_default_signal",
        "bout_assignment_rule": "first_nonnegative_peak_core_start_start_frame_inclusive_epoch",
        "rate_denominator": "valid_tracked_duration_s",
        "valid_tracked_duration_source": "linear_sample_valid_count_over_exact_fps",
        "motion_validity_rule": "linear_sample_valid_and_transition_valid",
        "spatial_metrics": "omitted_requires_separately_selected_position_provider",
        "protocol_to_acquisition_alignment": (
            "sealed_epoch_selection_proxy_not_physical_presentation"
            if plan.result.epoch_binding_mode == SEMANTIC_EPOCH_BINDING_MODE
            else "legacy_exact_epoch_selection_without_protocol_role_authority"
        ),
    }
    manifest = _summary_manifest(
        plan.result,
        run_path=plan.run_path,
        source_refs=source_refs,
        parameters=parameters,
    )
    run.attrs.update(
        json_attr_safe(
            {
                "schema_id": SCHEMA_ID,
                "schema_version": plan.result.schema_version,
                "method": METHOD_ID,
                "method_version": plan.result.method_version,
                "epoch_binding_mode": plan.result.epoch_binding_mode,
                "run_name": plan.run_name,
                "run_path": plan.run_path,
                "recording_id": plan.result.recording_id,
                "row_axis": "track_x_stimulus_epoch",
                "fps": plan.result.fps,
                "track_id": plan.track_id,
                "source_refs": source_refs,
                "source_refs_sha256": source_refs_sha256,
                "parameters": parameters,
                "analysis_offer": dict(plan.result.analysis_offer),
                "analysis_offer_sha256": plan.result.analysis_offer_sha256,
                MANIFEST_ATTR: manifest,
                MANIFEST_DIGEST_ATTR: canonical_json_sha256(manifest),
                "production_authority": False,
                "registry_update": False,
                "selection": "none",
                "summary": {
                    "epoch_count": int(plan.result.per_epoch_fish.shape[0]),
                    "bout_count": int(plan.result.per_epoch_bouts.shape[0]),
                    "epoch_labels": [
                        bytes(value).rstrip(b"\x00").decode("utf-8")
                        for value in plan.result.per_epoch_fish["window_label"]
                    ],
                    "bout_counts": plan.result.per_epoch_fish["bout_count"]
                    .astype(int)
                    .tolist(),
                },
                "run_provenance": build_writer_run_provenance(
                    command="provider_epoch_behavior_summary_materializer",
                    params=parameters,
                    input_run_ids={
                        "epoch": plan.epoch_run_name,
                        "motion": plan.motion_run_path,
                        "swim_bouts": plan.swim_bout_run_name,
                        **(
                            {
                                "protocol_semantic_selection": (
                                    plan.protocol_semantic_selection_run_name
                                )
                            }
                            if plan.protocol_semantic_selection_run_name is not None
                            else {}
                        ),
                    },
                    cwd=plan.source_zarr,
                    include_system_context=False,
                ),
            }
        )
    )
    mark_run_complete(
        run,
        parent_group=None,
        run_name=plan.run_name,
        run_provenance=run.attrs.get("run_provenance"),
    )
    run.attrs["stage_selector_eligible"] = False
    consolidate_metadata_capture_expected_warnings(plan.local_zarr)


def _arrays_equal(expected: np.ndarray, observed: np.ndarray) -> bool:
    if expected.dtype != observed.dtype or expected.shape != observed.shape:
        return False
    for name in expected.dtype.names or ():
        left = expected[name]
        right = observed[name]
        if left.dtype.kind == "f":
            if not np.array_equal(left, right, equal_nan=True):
                return False
        elif not np.array_equal(left, right):
            return False
    return True


def _validate_group(
    run: zarr.Group,
    *,
    result: ProviderEpochBehaviorSummaryResult,
) -> dict[str, Any]:
    attrs = run.attrs
    errors: list[str] = []
    if (
        attrs.get("schema_id") != SCHEMA_ID
        or attrs.get("schema_version") != result.schema_version
        or attrs.get("method_version") != result.method_version
        or attrs.get("epoch_binding_mode") != result.epoch_binding_mode
    ):
        errors.append("schema identity mismatch")
    if attrs.get(RUN_COMPLETION_CONTRACT_ATTR) != RUN_COMPLETION_CONTRACT:
        errors.append("completion contract mismatch")
    if attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
        errors.append("run is not complete")
    if attrs.get(RUN_NAME_ATTR) != result.run_name:
        errors.append("run-name binding mismatch")
    if attrs.get("stage_selector_eligible") is not False:
        errors.append("run is not selector-ineligible")
    if attrs.get("analysis_offer_sha256") != result.analysis_offer_sha256:
        errors.append("analysis-offer digest mismatch")
    if (
        canonical_json_sha256(attrs.get("analysis_offer"))
        != result.analysis_offer_sha256
    ):
        errors.append("analysis-offer payload is stale")
    expected_manifest = _summary_manifest(
        result,
        run_path=f"{PARENT_PATH}/{result.run_name}",
        source_refs=json_attr_safe(dict(result.source_bindings)),
        parameters=attrs.get("parameters", {}),
    )
    if attrs.get(MANIFEST_ATTR) != expected_manifest:
        errors.append("immutable summary manifest mismatch")
    if attrs.get(MANIFEST_DIGEST_ATTR) != canonical_json_sha256(expected_manifest):
        errors.append("immutable summary manifest digest mismatch")
    if (
        attrs.get("production_authority") is not False
        or attrs.get("registry_update") is not False
        or attrs.get("selection") != "none"
    ):
        errors.append("immutable summary safety state mismatch")
    source_refs = attrs.get("source_refs")
    if not isinstance(source_refs, Mapping):
        errors.append("source bindings are absent")
    else:
        source_refs_sha256 = canonical_json_sha256(source_refs)
        if attrs.get("source_refs_sha256") != source_refs_sha256:
            errors.append("source bindings digest is stale")
        if source_refs_sha256 != _source_bindings_sha256(result.source_bindings):
            errors.append("source bindings differ from the planned sources")
    expected_tables = {
        "per_epoch_fish": result.per_epoch_fish,
        "per_epoch_bouts": result.per_epoch_bouts,
        "per_epoch_bout_histograms": result.per_epoch_bout_histograms,
        "per_epoch_inter_bout_interval_histograms": (
            result.per_epoch_inter_bout_interval_histograms
        ),
    }
    for name, expected in expected_tables.items():
        try:
            observed, table_attrs = load_structured_dataset(run, name)
        except Exception as exc:
            errors.append(f"{name} is unreadable: {exc}")
            continue
        if not _arrays_equal(expected, observed):
            errors.append(f"{name} decoded values differ")
        if table_attrs.get("epoch_binding_mode") != result.epoch_binding_mode:
            errors.append(f"{name} epoch binding mode differs")
    if errors:
        raise ProviderEpochBehaviorSummaryError(
            "Invalid provider epoch summary: " + "; ".join(errors)
        )
    return {
        "valid": True,
        "epoch_count": int(result.per_epoch_fish.shape[0]),
        "bout_count": int(result.per_epoch_bouts.shape[0]),
    }


def _validate_run(
    zarr_path: Path,
    *,
    run_path: str,
    result: ProviderEpochBehaviorSummaryResult,
    use_consolidated: bool,
) -> dict[str, Any]:
    root = open_zarr_root(
        zarr_path,
        mode="r",
        use_consolidated=use_consolidated,
    )
    payload = _validate_group(root[run_path], result=result)
    payload["use_consolidated"] = bool(use_consolidated)
    return payload


def materialize_provider_epoch_behavior_summary(
    source_zarr: str | Path,
    *,
    scratch_root: str | Path,
    run_name: str,
    epoch_run_name: str,
    protocol_semantic_selection_run_name: str | None = None,
    motion_run: str,
    swim_bout_run_name: str,
    track_id: int = 0,
    speed_level: str = DEFAULT_SPEED_LEVEL,
    copy_backend: str = "rsync",
    apply: bool = False,
    keep_scratch: bool = False,
) -> dict[str, Any]:
    plan = build_provider_epoch_behavior_summary_plan(
        source_zarr,
        scratch_root=scratch_root,
        run_name=run_name,
        epoch_run_name=epoch_run_name,
        protocol_semantic_selection_run_name=(protocol_semantic_selection_run_name),
        motion_run=motion_run,
        swim_bout_run_name=swim_bout_run_name,
        track_id=track_id,
        speed_level=speed_level,
    )
    result: dict[str, Any] = {"status": "planned", "plan": plan.to_json()}
    if not apply:
        return result
    succeeded = False
    try:
        _write_local(plan)
        local_validation = _validate_run(
            plan.local_zarr,
            run_path=plan.run_path,
            result=plan.result,
            use_consolidated=False,
        )
        _validate_run(
            plan.local_zarr,
            run_path=plan.run_path,
            result=plan.result,
            use_consolidated=True,
        )
        refreshed = _compute_result(
            plan.source_zarr,
            run_name=plan.run_name,
            epoch_run_name=plan.epoch_run_name,
            protocol_semantic_selection_run_name=(
                plan.protocol_semantic_selection_run_name
            ),
            motion_run_path=plan.motion_run_path,
            swim_bout_run_name=plan.swim_bout_run_name,
            track_id=plan.track_id,
            speed_level=plan.speed_level,
        )
        if (
            refreshed.analysis_offer_sha256 != plan.result.analysis_offer_sha256
            or _source_bindings_sha256(refreshed.source_bindings)
            != _source_bindings_sha256(plan.result.source_bindings)
            or not _arrays_equal(refreshed.per_epoch_fish, plan.result.per_epoch_fish)
            or not _arrays_equal(refreshed.per_epoch_bouts, plan.result.per_epoch_bouts)
            or not _arrays_equal(
                refreshed.per_epoch_bout_histograms,
                plan.result.per_epoch_bout_histograms,
            )
            or not _arrays_equal(
                refreshed.per_epoch_inter_bout_interval_histograms,
                plan.result.per_epoch_inter_bout_interval_histograms,
            )
        ):
            raise ProviderEpochBehaviorSummaryError(
                "Provider epoch summary sources changed during materialization."
            )

        def validate(path: Path) -> dict[str, Any]:
            run = open_zarr_root(path, mode="r", use_consolidated=False)
            return _validate_group(run, result=plan.result)

        def prepare(root: zarr.Group) -> tuple[zarr.Group]:
            analysis = root.require_group("analysis")
            return (
                require_runs_parent(
                    analysis,
                    PARENT_PATH.split("/", 1)[1],
                ),
            )

        def complete(_root: zarr.Group, _parent: zarr.Group, run: zarr.Group) -> None:
            mark_run_complete(
                run,
                parent_group=None,
                run_name=plan.run_name,
                run_provenance=run.attrs.get("run_provenance"),
            )
            run.attrs["stage_selector_eligible"] = False

        def verify(root: zarr.Group) -> None:
            parent = root[PARENT_PATH]
            if _selector_snapshot(parent) != dict(plan.parent_selector_attrs):
                raise ProviderEpochBehaviorSummaryError(
                    "Provider epoch summary publication changed parent selectors."
                )
            run = parent.get(plan.run_name)
            if not isinstance(run, zarr.Group):
                raise ProviderEpochBehaviorSummaryError(
                    "Published provider epoch summary is absent."
                )
            if (
                run.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
                or run.attrs.get("stage_selector_eligible") is not False
            ):
                raise ProviderEpochBehaviorSummaryError(
                    "Published provider epoch summary lost its fail-closed lifecycle."
                )

        published_validation: dict[str, Any] = {}

        def finalize(root: zarr.Group, _parent: zarr.Group, _run: zarr.Group) -> None:
            published_validation.update(
                _validate_run(
                    plan.source_zarr,
                    run_path=plan.run_path,
                    result=plan.result,
                    use_consolidated=False,
                )
            )
            consolidate_metadata_capture_expected_warnings(plan.source_zarr)
            validate_direct_consolidated_subtree(
                plan.source_zarr,
                subtree_path=plan.run_path,
            )
            _validate_run(
                plan.source_zarr,
                run_path=plan.run_path,
                result=plan.result,
                use_consolidated=True,
            )

        publication = atomic_publish_run_group(
            AtomicRunPublishSpec(
                source_zarr=plan.source_zarr,
                local_run_path=plan.local_run_path,
                target_run_path=plan.target_run_path,
                run_name=plan.run_name,
                lock_suffix="provider-epoch-behavior-summary",
                publish_schema_id=(
                    SEMANTIC_PUBLISH_SCHEMA_ID
                    if plan.result.epoch_binding_mode == SEMANTIC_EPOCH_BINDING_MODE
                    else LEGACY_PUBLISH_SCHEMA_ID
                ),
                policy=(
                    "provider_epoch_behavior_summary_atomic_nonpromoting_"
                    f"v{plan.result.schema_version}"
                ),
                rollback_policy="retain_failed_public_tombstone_leave_selectors_untouched",
                content_checksum=True,
            ),
            copy_backend=copy_backend,
            validate_run=validate,
            prepare_parents=prepare,
            complete_run=complete,
            verify_pointers=verify,
            activate_run=finalize,
            repair_failed_publication_visibility=(
                lambda _path: consolidate_metadata_capture_expected_warnings(
                    plan.source_zarr
                )
            ),
            payload_metadata={
                "analysis_offer_sha256": plan.result.analysis_offer_sha256,
                "epoch_count": int(plan.result.per_epoch_fish.shape[0]),
                "bout_count": int(plan.result.per_epoch_bouts.shape[0]),
            },
        )
        result.update(
            status="complete",
            local_validation=local_validation,
            published_validation=published_validation,
            publication=publication,
            run_path=plan.run_path,
            summary=json_attr_safe(
                {
                    "epoch_labels": [
                        bytes(value).rstrip(b"\x00").decode("utf-8")
                        for value in plan.result.per_epoch_fish["window_label"]
                    ],
                    "bout_counts": plan.result.per_epoch_fish["bout_count"]
                    .astype(int)
                    .tolist(),
                    "bout_rates_per_min": plan.result.per_epoch_fish[
                        "bout_rate_per_min"
                    ]
                    .astype(float)
                    .tolist(),
                }
            ),
        )
        succeeded = True
        return json_attr_safe(result)
    finally:
        if succeeded and not keep_scratch and plan.scratch_root.exists():
            shutil.rmtree(plan.scratch_root)


def _default_scratch(run_name: str) -> Path:
    user = os.environ.get("USER") or "unknown"
    job = os.environ.get("LSB_JOBID") or "manual"
    base = Path("/scratch") / user
    if base.is_dir() and os.access(base, os.W_OK | os.X_OK):
        return base / job / f"palette_provider_epoch_summary_{run_name}"
    return Path(os.environ.get("TMPDIR") or "/tmp") / (
        f"palette_provider_epoch_summary_{job}_{run_name}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epoch-run", required=True)
    parser.add_argument("--protocol-semantic-selection-run")
    parser.add_argument("--motion-run", required=True)
    parser.add_argument("--swim-bout-run", required=True)
    parser.add_argument("--track-id", type=int, default=0)
    parser.add_argument(
        "--speed-level",
        choices=SUPPORTED_SPEED_LEVELS,
        default=DEFAULT_SPEED_LEVEL,
    )
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--copy-backend", choices=("rsync", "python"), default="rsync")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--keep-scratch", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    payload = materialize_provider_epoch_behavior_summary(
        args.zarr_path,
        scratch_root=args.scratch_root or _default_scratch(args.run_name),
        run_name=args.run_name,
        epoch_run_name=args.epoch_run,
        protocol_semantic_selection_run_name=(args.protocol_semantic_selection_run),
        motion_run=args.motion_run,
        swim_bout_run_name=args.swim_bout_run,
        track_id=args.track_id,
        speed_level=args.speed_level,
        copy_backend=args.copy_backend,
        apply=args.apply,
        keep_scratch=args.keep_scratch,
    )
    print(json.dumps(payload, indent=None if args.json else 2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ANALYSIS_CLASS_ID",
    "MANIFEST_ATTR",
    "MANIFEST_DIGEST_ATTR",
    "METHOD_ID",
    "PARENT_PATH",
    "SEMANTIC_EPOCH_BINDING_MODE",
    "SEMANTIC_SCHEMA_VERSION",
    "ProviderEpochBehaviorSummaryError",
    "ProviderEpochBehaviorSummaryPlan",
    "ProviderEpochBehaviorSummaryResult",
    "build_provider_epoch_behavior_summary_plan",
    "materialize_provider_epoch_behavior_summary",
]
