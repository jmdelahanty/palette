"""Compute per-bout heading metrics from track kinematics and swim-bout candidates."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import matplotlib

# Bout-kinematics plot artifacts are persisted PNG bytes, not interactive GUI
# windows. Force a non-GUI backend so workstation display settings cannot make
# CLI artifact generation depend on Tk/Qt teardown behavior.
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import zarr

from fisheye.analysis.chaser_state_interpolator import (
    load_structured_dataset,
    write_columnar_dataset,
)
from fisheye.analysis.detect_bouts_multi_level import normalize_speed_level
from fisheye.analysis.eye_angle_io import EyeAngleIOError, load_eye_gaze_frame_series
from fisheye.analysis.swim_bout_io import load_swim_bout_tables
from fisheye.shared.plot_artifacts import (
    write_interactive_plot_spec_artifact,
    write_png_visualization_artifact,
)
from fisheye.shared.json_safety import decode_null_terminated_text
from fisheye.shared.run_lineage_fingerprint import write_best_effort_run_lineage_attrs
from fisheye.shared.stage_provenance import build_stage_provenance, write_stage_provenance
from fisheye.shared.zarr_run_completion import (
    mark_run_complete,
    mark_run_failed,
    mark_run_started,
    require_runs_parent,
)
from fisheye.shared.system_metadata import get_environment_info, get_git_info
from fisheye.shared.zarr_io import open_zarr_root


SCHEMA_ID = "analysis.bout_kinematics_runs"
SCHEMA_VERSION = 7
METHOD = "heading_window_and_within_bout_metrics"
METHOD_VERSION = "bout_kinematics.v7"
BOUT_KINEMATICS_PLOT_SPEC_SCHEMA_ID = "palette.plot_spec.bout_kinematics_summary.v1"
BOUT_EYE_GAZE_PLOT_SPEC_SCHEMA_ID = "palette.plot_spec.bout_eye_gaze_summary.v1"
BOUT_MOVEMENT_PLOT_SPEC_SCHEMA_ID = "palette.plot_spec.bout_movement_summary.v1"
BOUT_KINEMATICS_PLOT_RENDERER = "matplotlib_static_plotly_spec.v1"
BOUT_KINEMATICS_PNG_PREFIX = "bout_kinematics_summary"
BOUT_EYE_GAZE_PNG_PREFIX = "bout_eye_gaze_summary"
BOUT_MOVEMENT_PNG_PREFIX = "bout_movement_summary"

HEADING_LEVEL_TO_ARRAY = {
    "heading_smoothed": "smoothed_heading_degrees",
    "heading_raw": "heading_degrees",
}
HEADING_LEVEL_ALIASES = {
    "smoothed": "heading_smoothed",
    "raw": "heading_raw",
    **{level: level for level in HEADING_LEVEL_TO_ARRAY},
}
WITHIN_WINDOWS = ("bout_start_end", "core_start_end")
PRE_POST_MODES = ("fixed_window", "interbout_epoch")
EYE_GAZE_LEVEL = "eye_gaze"
MOVEMENT_LEVEL = "movement"
EYE_ANGLE_FAMILIES = ("gaze",)
PHYSICAL_ACTIVE_BOUNDARY_POLICY = "physical_active"
PHYSICAL_ACTIVE_BOUNDARY_CONSTRAINTS = (
    "clip_to_detector",
    "search_with_margin",
    "allow_extension",
)
PHYSICAL_ACTIVE_SPEED_LEVELS = ("speed_raw", "speed_filtered", "speed_smoothed")
LAYOUT_HIERARCHICAL_V1 = "hierarchical_v1"
LAYOUT_COMPACT_TABULAR_V2 = "compact_tabular_v2"
BOUT_KINEMATICS_LAYOUTS = (LAYOUT_HIERARCHICAL_V1, LAYOUT_COMPACT_TABULAR_V2)
BOUT_KINEMATICS_LAYOUT_DEFAULT = LAYOUT_COMPACT_TABULAR_V2
COMPACT_LEVEL_INDEX = "level_index"
COMPACT_MOVEMENT_TABLE = "movement_metrics"
COMPACT_HEADING_TABLE = "heading_metrics"
COMPACT_EYE_GAZE_TABLE = "eye_gaze_metrics"
_COMPACT_LEVEL_FIELDS = (
    "analysis_level_id",
    "analysis_level_bytes",
    "heading_level_id",
    "heading_level_bytes",
)


def normalize_heading_level(value: str) -> str:
    """Normalize user-facing heading-level names to stored subgroup names."""

    normalized = HEADING_LEVEL_ALIASES.get(str(value).strip())
    if normalized not in HEADING_LEVEL_TO_ARRAY:
        expected = ", ".join(sorted(HEADING_LEVEL_ALIASES))
        raise ValueError(f"Unsupported heading level {value!r}; expected one of: {expected}")
    return normalized


def _fixed_bytes(value: object, *, width: int = 64) -> bytes:
    raw = str(value).encode("utf-8", errors="replace")
    return raw[:width]


def _decode_fixed_bytes(value: object) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return decode_null_terminated_text(value)
    return str(value)


def _with_compact_level_columns(
    records: np.ndarray,
    *,
    analysis_level: str,
    analysis_level_id: int,
    heading_level: Optional[str] = None,
    heading_level_id: int = -1,
) -> np.ndarray:
    if records.dtype.names is None:
        raise ValueError("Compact bout-kinematics tables require structured records.")
    dtype = np.dtype(
        [
            ("analysis_level_id", "i2"),
            ("analysis_level_bytes", "S64"),
            ("heading_level_id", "i2"),
            ("heading_level_bytes", "S64"),
            *records.dtype.descr,
        ]
    )
    out = np.empty(records.shape[0], dtype=dtype)
    out["analysis_level_id"] = int(analysis_level_id)
    out["analysis_level_bytes"] = _fixed_bytes(analysis_level)
    out["heading_level_id"] = int(heading_level_id)
    out["heading_level_bytes"] = _fixed_bytes(heading_level or "")
    for name in records.dtype.names:
        out[name] = records[name]
    return out


def _drop_compact_level_columns(records: np.ndarray) -> np.ndarray:
    names = records.dtype.names or ()
    keep = [name for name in names if name not in _COMPACT_LEVEL_FIELDS]
    if len(keep) == len(names):
        return records
    if not keep:
        return np.empty(records.shape[0], dtype=[])
    return records[keep].copy()


def _empty_structured_records() -> np.ndarray:
    return np.empty(0, dtype=[])


def _column_group_to_structured_records(table_group: zarr.Group) -> np.ndarray:
    columns: list[tuple[str, np.ndarray]] = []
    row_count: Optional[int] = None
    for name in sorted(table_group.keys()):
        try:
            values = np.asarray(table_group[name][:])
        except Exception:
            continue
        if values.ndim == 0:
            values = values.reshape(1)
        if row_count is None:
            row_count = int(values.shape[0])
        elif int(values.shape[0]) != row_count:
            raise ValueError(
                f"Column {name!r} length {values.shape[0]} does not match expected {row_count}."
            )
        columns.append((str(name), values))
    if row_count is None:
        return _empty_structured_records()

    dtype_fields: list[tuple[object, ...]] = []
    for name, values in columns:
        if values.ndim <= 1:
            dtype_fields.append((name, values.dtype))
        else:
            dtype_fields.append((name, values.dtype, values.shape[1:]))
    records = np.empty(row_count, dtype=np.dtype(dtype_fields))
    for name, values in columns:
        records[name] = values
    return records


def _load_table_or_empty(parent: zarr.Group, name: str) -> tuple[np.ndarray, dict[str, object]]:
    try:
        records, attrs = load_structured_dataset(parent, name)
    except Exception:
        try:
            table_group = parent[name]
            return _column_group_to_structured_records(table_group), dict(table_group.attrs)
        except Exception:
            return _empty_structured_records(), {}
    return np.asarray(records), dict(attrs)


def _compact_level_index_records(
    *,
    heading_levels: Sequence[str],
    default_heading_level: str,
    movement_count: int,
    heading_counts: Mapping[str, int],
    eye_gaze_count: Optional[int],
) -> np.ndarray:
    dtype = np.dtype(
        [
            ("analysis_level_id", "i2"),
            ("analysis_level_bytes", "S64"),
            ("measurement_family_bytes", "S64"),
            ("heading_level_id", "i2"),
            ("heading_level_bytes", "S64"),
            ("is_default_heading_level", "?"),
            ("row_count", "i8"),
        ]
    )
    rows: list[tuple[int, bytes, bytes, int, bytes, bool, int]] = [
        (
            0,
            _fixed_bytes(MOVEMENT_LEVEL),
            _fixed_bytes(MOVEMENT_LEVEL),
            -1,
            _fixed_bytes(""),
            False,
            int(movement_count),
        )
    ]
    for idx, level in enumerate(heading_levels):
        rows.append(
            (
                idx + 1,
                _fixed_bytes(level),
                _fixed_bytes("heading"),
                idx,
                _fixed_bytes(level),
                level == default_heading_level,
                int(heading_counts.get(level, 0)),
            )
        )
    if eye_gaze_count is not None:
        rows.append(
            (
                len(rows),
                _fixed_bytes(EYE_GAZE_LEVEL),
                _fixed_bytes(EYE_GAZE_LEVEL),
                -1,
                _fixed_bytes(""),
                False,
                int(eye_gaze_count),
            )
        )
    return np.asarray(rows, dtype=dtype)


def _concat_heading_compact_records(
    metrics_by_level: Mapping[str, np.ndarray],
    *,
    heading_levels: Sequence[str],
) -> np.ndarray:
    pieces: list[np.ndarray] = []
    for idx, level in enumerate(heading_levels):
        records = metrics_by_level.get(level)
        if records is None:
            continue
        pieces.append(
            _with_compact_level_columns(
                records,
                analysis_level=level,
                analysis_level_id=idx + 1,
                heading_level=level,
                heading_level_id=idx,
            )
        )
    if not pieces:
        return _empty_structured_records()
    return np.concatenate(pieces)


def resolve_bout_kinematics_tables(
    run_group: zarr.Group,
    *,
    heading_level: Optional[str] = None,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    """Return logical bout-kinematics tables independent of physical layout.

    The returned mappings are keyed by logical analysis level: ``movement``,
    concrete heading levels such as ``heading_smoothed``, and optional
    ``eye_gaze``. Compact-v2 rows have layout index columns stripped so callers
    see the same record schema as hierarchical-v1 readers.
    """

    layout = str(run_group.attrs.get("layout", LAYOUT_HIERARCHICAL_V1))
    if layout == LAYOUT_COMPACT_TABULAR_V2:
        return _resolve_compact_bout_kinematics_tables(run_group, heading_level=heading_level)
    return _resolve_hierarchical_bout_kinematics_tables(run_group, heading_level=heading_level)


def _resolve_requested_levels(run_group: zarr.Group, heading_level: Optional[str]) -> tuple[str, ...]:
    if heading_level is not None:
        if str(heading_level).strip() == EYE_GAZE_LEVEL:
            return (EYE_GAZE_LEVEL,)
        if str(heading_level).strip() == MOVEMENT_LEVEL:
            return (MOVEMENT_LEVEL,)
        return (normalize_heading_level(heading_level),)

    levels = tuple(str(level) for level in run_group.attrs.get("heading_levels", []))
    if not levels:
        levels = tuple(level for level in ("heading_smoothed", "heading_raw") if level in run_group)
    if MOVEMENT_LEVEL in run_group or COMPACT_MOVEMENT_TABLE in run_group:
        levels = (MOVEMENT_LEVEL, *levels)
    if EYE_GAZE_LEVEL in run_group or COMPACT_EYE_GAZE_TABLE in run_group:
        levels = (*levels, EYE_GAZE_LEVEL)
    return levels


def _resolve_hierarchical_bout_kinematics_tables(
    run_group: zarr.Group,
    *,
    heading_level: Optional[str],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    records_by_level: dict[str, np.ndarray] = {}
    level_attrs_by_level: dict[str, dict[str, object]] = {}
    table_attrs_by_level: dict[str, dict[str, object]] = {}
    for level in _resolve_requested_levels(run_group, heading_level):
        if level not in run_group:
            continue
        records, attrs = _load_table_or_empty(run_group[level], "per_bout_metrics")
        records_by_level[level] = records
        level_attrs_by_level[level] = dict(run_group[level].attrs)
        table_attrs_by_level[level] = attrs
    return records_by_level, level_attrs_by_level, table_attrs_by_level


def _resolve_compact_bout_kinematics_tables(
    run_group: zarr.Group,
    *,
    heading_level: Optional[str],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    requested = set(_resolve_requested_levels(run_group, heading_level))
    records_by_level: dict[str, np.ndarray] = {}
    level_attrs_by_level: dict[str, dict[str, object]] = {}
    table_attrs_by_level: dict[str, dict[str, object]] = {}

    if MOVEMENT_LEVEL in requested and COMPACT_MOVEMENT_TABLE in run_group:
        records, attrs = _load_table_or_empty(run_group, COMPACT_MOVEMENT_TABLE)
        records_by_level[MOVEMENT_LEVEL] = _drop_compact_level_columns(records)
        level_attrs_by_level[MOVEMENT_LEVEL] = {
            "analysis_level": MOVEMENT_LEVEL,
            **attrs,
        }
        table_attrs_by_level[MOVEMENT_LEVEL] = attrs

    if COMPACT_HEADING_TABLE in run_group:
        heading_records, heading_attrs = _load_table_or_empty(run_group, COMPACT_HEADING_TABLE)
        names = heading_records.dtype.names or ()
        if "heading_level_bytes" in names:
            labels = np.asarray([_decode_fixed_bytes(value) for value in heading_records["heading_level_bytes"]])
            heading_levels = tuple(str(level) for level in run_group.attrs.get("heading_levels", []))
            if not heading_levels:
                heading_levels = tuple(dict.fromkeys(str(label) for label in labels if str(label)))
            for level in heading_levels:
                if level not in requested:
                    continue
                mask = labels == level
                level_records = heading_records[mask]
                records_by_level[level] = _drop_compact_level_columns(level_records)
                level_attrs_by_level[level] = {
                    "analysis_level": level,
                    "heading_level": level,
                    "is_default_heading_level": level == run_group.attrs.get("default_heading_level"),
                    **heading_attrs,
                }
                table_attrs_by_level[level] = heading_attrs

    if EYE_GAZE_LEVEL in requested and COMPACT_EYE_GAZE_TABLE in run_group:
        records, attrs = _load_table_or_empty(run_group, COMPACT_EYE_GAZE_TABLE)
        records_by_level[EYE_GAZE_LEVEL] = _drop_compact_level_columns(records)
        level_attrs_by_level[EYE_GAZE_LEVEL] = {
            "analysis_level": EYE_GAZE_LEVEL,
            **attrs,
        }
        table_attrs_by_level[EYE_GAZE_LEVEL] = attrs

    return records_by_level, level_attrs_by_level, table_attrs_by_level


def _write_compact_bout_kinematics_tables(
    run_group: zarr.Group,
    *,
    movement_metrics: np.ndarray,
    movement_attrs: Mapping[str, object],
    metrics_by_level: Mapping[str, np.ndarray],
    heading_levels: Sequence[str],
    default_heading_level: str,
    heading_table_attrs: Mapping[str, object],
    eye_gaze_metrics: Optional[np.ndarray],
    eye_gaze_attrs: Optional[Mapping[str, object]],
) -> None:
    level_index = _compact_level_index_records(
        heading_levels=heading_levels,
        default_heading_level=default_heading_level,
        movement_count=int(len(movement_metrics)),
        heading_counts={level: int(len(records)) for level, records in metrics_by_level.items()},
        eye_gaze_count=None if eye_gaze_metrics is None else int(len(eye_gaze_metrics)),
    )
    write_columnar_dataset(
        run_group,
        COMPACT_LEVEL_INDEX,
        level_index,
        attrs={
            "schema_id": f"{SCHEMA_ID}.compact_v2.level_index",
            "schema_version": SCHEMA_VERSION,
            "layout": LAYOUT_COMPACT_TABULAR_V2,
        },
    )
    write_columnar_dataset(
        run_group,
        COMPACT_MOVEMENT_TABLE,
        _with_compact_level_columns(
            movement_metrics,
            analysis_level=MOVEMENT_LEVEL,
            analysis_level_id=0,
        ),
        attrs={
            **dict(movement_attrs),
            "schema_id": f"{SCHEMA_ID}.compact_v2.movement_metrics",
            "schema_version": SCHEMA_VERSION,
            "layout": LAYOUT_COMPACT_TABULAR_V2,
            "analysis_level": MOVEMENT_LEVEL,
        },
    )

    heading_metrics = _concat_heading_compact_records(metrics_by_level, heading_levels=heading_levels)
    if heading_metrics.dtype.names:
        write_columnar_dataset(
            run_group,
            COMPACT_HEADING_TABLE,
            heading_metrics,
            attrs={
                **dict(heading_table_attrs),
                "schema_id": f"{SCHEMA_ID}.compact_v2.heading_metrics",
                "schema_version": SCHEMA_VERSION,
                "layout": LAYOUT_COMPACT_TABULAR_V2,
                "analysis_level": "heading",
                "heading_levels": list(heading_levels),
                "default_heading_level": default_heading_level,
            },
        )

    if eye_gaze_metrics is not None and eye_gaze_attrs is not None:
        write_columnar_dataset(
            run_group,
            COMPACT_EYE_GAZE_TABLE,
            _with_compact_level_columns(
                eye_gaze_metrics,
                analysis_level=EYE_GAZE_LEVEL,
                analysis_level_id=len(heading_levels) + 1,
            ),
            attrs={
                **dict(eye_gaze_attrs),
                "schema_id": f"{SCHEMA_ID}.compact_v2.eye_gaze_metrics",
                "schema_version": SCHEMA_VERSION,
                "layout": LAYOUT_COMPACT_TABULAR_V2,
                "analysis_level": EYE_GAZE_LEVEL,
            },
        )


def _metrics_dtype() -> np.dtype:
    return np.dtype(
        [
            ("bout_id", "i4"),
            ("source_start_frame", "i8"),
            ("source_end_frame", "i8"),
            ("source_core_start_frame", "i8"),
            ("source_core_end_frame", "i8"),
            ("source_core_start_time_s_interpolated", "f8"),
            ("source_core_end_time_s_interpolated", "f8"),
            ("source_core_duration_s_interpolated", "f8"),
            ("source_core_start_time_interpolated_valid", "?"),
            ("source_core_end_time_interpolated_valid", "?"),
            ("source_peak_frame", "i8"),
            ("source_peak_time_s", "f8"),
            ("source_peak_signal_value_mm_s", "f8"),
            ("source_peak_prominence_mm_s", "f8"),
            ("source_peak_width_s", "f8"),
            ("source_peak_width_height_mm_s", "f8"),
            ("source_peak_left_width_frame_interpolated", "f8"),
            ("source_peak_right_width_frame_interpolated", "f8"),
            ("source_peak_left_width_time_s", "f8"),
            ("source_peak_right_width_time_s", "f8"),
            ("source_peak_boundary_mode_bytes", "S64"),
            ("source_peak_shape_split_policy_bytes", "S64"),
            ("pre_epoch_start_frame", "i8"),
            ("pre_epoch_end_frame", "i8"),
            ("post_epoch_start_frame", "i8"),
            ("post_epoch_end_frame", "i8"),
            ("pre_heading_mean_deg", "f8"),
            ("post_heading_mean_deg", "f8"),
            ("net_delta_heading_deg", "f8"),
            ("abs_net_delta_heading_deg", "f8"),
            ("pre_position_mean_x_mm", "f8"),
            ("pre_position_mean_y_mm", "f8"),
            ("post_position_mean_x_mm", "f8"),
            ("post_position_mean_y_mm", "f8"),
            ("interbout_epoch_displacement_mm", "f8"),
            ("pre_position_mean_x_px", "f8"),
            ("pre_position_mean_y_px", "f8"),
            ("post_position_mean_x_px", "f8"),
            ("post_position_mean_y_px", "f8"),
            ("interbout_epoch_displacement_px", "f8"),
            ("within_heading_range_deg", "f8"),
            ("within_heading_peak_to_peak_deg", "f8"),
            ("within_heading_path_deg", "f8"),
            ("within_heading_std_deg", "f8"),
            ("within_heading_zero_crossings", "i4"),
            ("within_heading_dominant_frequency_hz", "f8"),
            ("within_angular_velocity_mean_deg_s", "f8"),
            ("within_angular_speed_mean_deg_s", "f8"),
            ("within_angular_speed_max_deg_s", "f8"),
            ("within_angular_velocity_std_deg_s", "f8"),
            ("pre_window_valid", "?"),
            ("post_window_valid", "?"),
            ("pre_position_valid", "?"),
            ("post_position_valid", "?"),
            ("within_window_valid", "?"),
            ("within_angular_velocity_valid", "?"),
            ("dominant_frequency_valid", "?"),
            ("pre_window_sample_count", "i4"),
            ("post_window_sample_count", "i4"),
            ("pre_position_sample_count", "i4"),
            ("post_position_sample_count", "i4"),
            ("within_window_sample_count", "i4"),
            ("within_angular_velocity_transition_count", "i4"),
            ("failure_reason_bytes", "S256"),
        ]
    )


def _eye_gaze_metrics_dtype() -> np.dtype:
    return np.dtype(
        [
            ("bout_id", "i4"),
            ("source_start_frame", "i8"),
            ("source_end_frame", "i8"),
            ("source_core_start_frame", "i8"),
            ("source_core_end_frame", "i8"),
            ("pre_epoch_start_frame", "i8"),
            ("pre_epoch_end_frame", "i8"),
            ("post_epoch_start_frame", "i8"),
            ("post_epoch_end_frame", "i8"),
            ("within_epoch_start_frame", "i8"),
            ("within_epoch_end_frame", "i8"),
            ("pre_left_gaze_mean_deg", "f8"),
            ("pre_right_gaze_mean_deg", "f8"),
            ("pre_vergence_gaze_mean_deg", "f8"),
            ("pre_vergence_gaze_signed_mean_deg", "f8"),
            ("pre_vergence_gaze_std_deg", "f8"),
            ("pre_vergence_gaze_valid_fraction", "f8"),
            ("pre_converged_fraction", "f8"),
            ("post_left_gaze_mean_deg", "f8"),
            ("post_right_gaze_mean_deg", "f8"),
            ("post_vergence_gaze_mean_deg", "f8"),
            ("post_vergence_gaze_signed_mean_deg", "f8"),
            ("post_vergence_gaze_std_deg", "f8"),
            ("post_vergence_gaze_valid_fraction", "f8"),
            ("post_converged_fraction", "f8"),
            ("within_bout_left_gaze_mean_deg", "f8"),
            ("within_bout_right_gaze_mean_deg", "f8"),
            ("within_bout_vergence_gaze_mean_deg", "f8"),
            ("within_bout_vergence_gaze_signed_mean_deg", "f8"),
            ("within_bout_vergence_gaze_max_deg", "f8"),
            ("within_bout_vergence_gaze_range_deg", "f8"),
            ("within_bout_vergence_gaze_std_deg", "f8"),
            ("within_bout_vergence_gaze_valid_fraction", "f8"),
            ("within_bout_converged_fraction", "f8"),
            ("pre_eye_window_valid", "?"),
            ("post_eye_window_valid", "?"),
            ("within_eye_window_valid", "?"),
            ("pre_eye_sample_count", "i4"),
            ("post_eye_sample_count", "i4"),
            ("within_eye_sample_count", "i4"),
            ("failure_reason_bytes", "S256"),
        ]
    )


def _movement_metrics_dtype() -> np.dtype:
    return np.dtype(
        [
            ("bout_id", "i4"),
            ("source_start_frame", "i8"),
            ("source_end_frame", "i8"),
            ("source_core_start_frame", "i8"),
            ("source_core_end_frame", "i8"),
            ("detector_duration_s", "f8"),
            ("detector_observed_duration_s", "f8"),
            ("detector_core_duration_s", "f8"),
            ("physical_active_start_frame", "i8"),
            ("physical_active_end_frame", "i8"),
            ("physical_active_start_time_s", "f8"),
            ("physical_active_end_time_s", "f8"),
            ("physical_active_duration_s", "f8"),
            ("physical_active_observed_duration_s", "f8"),
            ("physical_active_start_time_s_interpolated", "f8"),
            ("physical_active_end_time_s_interpolated", "f8"),
            ("physical_active_duration_s_interpolated", "f8"),
            ("physical_active_start_time_interpolated_valid", "?"),
            ("physical_active_end_time_interpolated_valid", "?"),
            ("physical_active_sample_count", "i4"),
            ("physical_active_valid_transition_count", "i4"),
            ("physical_active_valid_transition_fraction", "f8"),
            ("physical_active_path_length_mm", "f8"),
            ("physical_active_path_length_px", "f8"),
            ("physical_active_mean_speed_mm_s", "f8"),
            ("physical_active_peak_speed_mm_s", "f8"),
            ("physical_active_threshold_mm_s", "f8"),
            ("physical_active_boundary_margin_s", "f8"),
            ("physical_active_boundary_policy_bytes", "S64"),
            ("physical_active_boundary_constraint_bytes", "S64"),
            ("physical_active_valid", "?"),
            ("failure_reason_bytes", "S256"),
        ]
    )


def _wrap_degrees(delta: float) -> float:
    if not np.isfinite(delta):
        return float("nan")
    return float((delta + 180.0) % 360.0 - 180.0)


def _circular_mean_deg(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    radians = np.deg2rad(finite)
    return float(np.rad2deg(np.arctan2(np.mean(np.sin(radians)), np.mean(np.cos(radians)))))


def _unwrap_degrees(values: np.ndarray) -> np.ndarray:
    return np.rad2deg(np.unwrap(np.deg2rad(np.asarray(values, dtype=np.float64))))


def _std_unwrapped_deg(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.std(_unwrap_degrees(finite)))


def _zero_crossings(
    headings: np.ndarray,
    times: np.ndarray,
    *,
    derivative_threshold_deg_s: float,
) -> int:
    finite_mask = np.isfinite(headings) & np.isfinite(times)
    if np.count_nonzero(finite_mask) < 3:
        return 0
    values = _unwrap_degrees(np.asarray(headings, dtype=np.float64)[finite_mask])
    t = np.asarray(times, dtype=np.float64)[finite_mask]
    dt = np.diff(t)
    valid_dt = dt > 0
    if np.count_nonzero(valid_dt) < 2:
        return 0
    velocity = np.diff(values)[valid_dt] / dt[valid_dt]
    threshold = abs(float(derivative_threshold_deg_s))
    signs = np.sign(velocity)
    if threshold > 0:
        signs[np.abs(velocity) < threshold] = 0
    signs = signs[signs != 0]
    if signs.size < 2:
        return 0
    return int(np.count_nonzero(signs[1:] != signs[:-1]))


def _angular_velocity_steps(
    headings: np.ndarray,
    times: np.ndarray,
    *,
    transition_valid: Optional[np.ndarray] = None,
    sample_valid: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, bool, int, Optional[str]]:
    """Return valid per-transition angular velocity values for a heading window."""

    heading_values = np.asarray(headings, dtype=np.float64)
    time_values = np.asarray(times, dtype=np.float64)
    if heading_values.size < 2 or time_values.size != heading_values.size:
        return np.asarray([], dtype=np.float64), False, 0, "insufficient_angular_velocity_samples"

    dt = np.diff(time_values)
    unwrapped = np.rad2deg(np.unwrap(np.deg2rad(heading_values)))
    delta = np.diff(unwrapped)
    valid = (
        np.isfinite(heading_values[1:])
        & np.isfinite(heading_values[:-1])
        & np.isfinite(dt)
        & (dt > 0)
    )
    if transition_valid is not None:
        transition = np.asarray(transition_valid, dtype=bool)
        if transition.shape[0] == heading_values.shape[0]:
            valid &= transition[1:]
    if sample_valid is not None:
        samples = np.asarray(sample_valid, dtype=bool)
        if samples.shape[0] == heading_values.shape[0]:
            valid &= samples[1:] & samples[:-1]

    transition_count = int(valid.size)
    if transition_count == 0:
        return np.asarray([], dtype=np.float64), False, 0, "insufficient_angular_velocity_samples"
    if not bool(np.all(valid)):
        return np.asarray([], dtype=np.float64), False, transition_count, "heading_transition_contains_gap"

    return delta[valid] / dt[valid], True, transition_count, None


def _dominant_frequency_hz(
    headings: np.ndarray,
    times: np.ndarray,
    *,
    enabled: bool,
    min_samples: int,
    detrend: bool,
) -> tuple[float, bool, Optional[str]]:
    if not enabled:
        return float("nan"), False, "dominant_frequency_disabled"

    finite_mask = np.isfinite(headings) & np.isfinite(times)
    if np.count_nonzero(finite_mask) < int(min_samples):
        return float("nan"), False, "dominant_frequency_insufficient_samples"

    values = _unwrap_degrees(np.asarray(headings, dtype=np.float64)[finite_mask])
    t = np.asarray(times, dtype=np.float64)[finite_mask]
    dt = np.diff(t)
    finite_dt = dt[np.isfinite(dt) & (dt > 0)]
    if finite_dt.size == 0:
        return float("nan"), False, "dominant_frequency_insufficient_samples"

    sample_spacing = float(np.median(finite_dt))
    if sample_spacing <= 0:
        return float("nan"), False, "dominant_frequency_insufficient_samples"

    if detrend:
        values = values - np.linspace(values[0], values[-1], values.size)
    values = values - np.mean(values)
    spectrum = np.abs(np.fft.rfft(values))
    freqs = np.fft.rfftfreq(values.size, d=sample_spacing)
    if spectrum.size <= 1:
        return float("nan"), False, "dominant_frequency_insufficient_samples"
    peak_idx = int(np.argmax(spectrum[1:]) + 1)
    return float(freqs[peak_idx]), True, None


def _artifact_signature(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _png_bytes_from_figure(fig: plt.Figure, *, dpi: int) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def _field_or_default(bouts: np.ndarray, field: str, default: int) -> np.ndarray:
    if field in (bouts.dtype.names or ()):
        return np.asarray(bouts[field])
    return np.full(len(bouts), default)


def _float_field_or_nan(bouts: np.ndarray, field: str) -> np.ndarray:
    if field in (bouts.dtype.names or ()):
        return np.asarray(bouts[field], dtype=np.float64)
    return np.full(len(bouts), float("nan"), dtype=np.float64)


def _bool_field_or_false(bouts: np.ndarray, field: str) -> np.ndarray:
    if field in (bouts.dtype.names or ()):
        return np.asarray(bouts[field], dtype=bool)
    return np.zeros(len(bouts), dtype=bool)


def _bytes_field_or_empty(records: Optional[np.ndarray], field: str, count: int) -> np.ndarray:
    output = np.full(count, b"", dtype="S64")
    if records is None or field not in (records.dtype.names or ()):
        return output
    for idx, value in enumerate(records[field]):
        if idx >= count:
            break
        if isinstance(value, bytes):
            output[idx] = value[:64]
        else:
            output[idx] = str(value).encode("utf-8")[:64]
    return output


def _records_align_by_bout_id(bouts: np.ndarray, records: np.ndarray) -> bool:
    bout_names = bouts.dtype.names or ()
    record_names = records.dtype.names or ()
    if "bout_id" not in bout_names or "bout_id" not in record_names:
        return True
    try:
        return bool(np.array_equal(bouts["bout_id"], records["bout_id"]))
    except Exception:
        return False


def _epoch_bounds(frames: np.ndarray, epoch_slice: slice) -> tuple[int, int]:
    if epoch_slice.stop <= epoch_slice.start:
        return -1, -1
    return int(frames[epoch_slice.start]), int(frames[epoch_slice.stop - 1])


def _position_epoch_stats(
    positions: Optional[np.ndarray],
    epoch_slice: slice,
) -> tuple[float, float, int, bool]:
    if positions is None:
        return float("nan"), float("nan"), 0, False
    epoch = np.asarray(positions[epoch_slice], dtype=np.float64)
    if epoch.ndim != 2 or epoch.shape[1] != 2 or epoch.shape[0] == 0:
        return float("nan"), float("nan"), 0, False
    finite_rows = np.isfinite(epoch).all(axis=1)
    finite_count = int(np.count_nonzero(finite_rows))
    valid = finite_count == int(epoch.shape[0])
    if not valid:
        return float("nan"), float("nan"), finite_count, False
    mean = np.mean(epoch, axis=0)
    return float(mean[0]), float(mean[1]), finite_count, True


def _distance_2d(x0: float, y0: float, x1: float, y1: float) -> float:
    values = np.asarray([x0, y0, x1, y1], dtype=np.float64)
    if not np.isfinite(values).all():
        return float("nan")
    return float(np.hypot(float(x1) - float(x0), float(y1) - float(y0)))


def _speed_level_suffix(level: str) -> str:
    normalized = normalize_speed_level(level)
    if normalized == "speed_exponential":
        raise ValueError("Physical movement estimators cannot use speed_exponential.")
    return normalized.removeprefix("speed_")


def _threshold_crossing_time_from_samples(
    *,
    values: np.ndarray,
    times: np.ndarray,
    threshold: float,
    below_idx: int,
    above_idx: int,
) -> tuple[float, bool]:
    if below_idx < 0 or above_idx < 0 or below_idx >= values.size or above_idx >= values.size:
        return float("nan"), False
    if abs(int(above_idx) - int(below_idx)) != 1:
        return float("nan"), False
    below_value = float(values[below_idx])
    above_value = float(values[above_idx])
    below_time = float(times[below_idx])
    above_time = float(times[above_idx])
    if not (
        np.isfinite(below_value)
        and np.isfinite(above_value)
        and np.isfinite(below_time)
        and np.isfinite(above_time)
    ):
        return float("nan"), False
    if below_value == above_value:
        return float("nan"), False
    threshold = float(threshold)
    lower = min(below_value, above_value)
    upper = max(below_value, above_value)
    if threshold < lower or threshold > upper:
        return float("nan"), False
    fraction = (threshold - below_value) / (above_value - below_value)
    if not np.isfinite(fraction):
        return float("nan"), False
    fraction = float(np.clip(fraction, 0.0, 1.0))
    return below_time + fraction * (above_time - below_time), True


def _physical_search_bounds(
    *,
    start_idx: int,
    end_idx: int,
    previous_end_idx: int,
    next_start_idx: int,
    sample_count: int,
    margin_frames: int,
    boundary_constraint: str,
) -> tuple[int, int]:
    if boundary_constraint == "clip_to_detector":
        return int(start_idx), int(end_idx)
    if boundary_constraint == "search_with_margin":
        left = int(start_idx) - int(margin_frames)
        right = int(end_idx) + int(margin_frames)
    elif boundary_constraint == "allow_extension":
        left = 0
        right = int(sample_count) - 1
    else:
        expected = ", ".join(PHYSICAL_ACTIVE_BOUNDARY_CONSTRAINTS)
        raise ValueError(f"Unsupported physical active boundary constraint {boundary_constraint!r}; expected one of: {expected}")

    if previous_end_idx >= 0:
        left = max(left, int(previous_end_idx) + 1)
    if next_start_idx >= 0:
        right = min(right, int(next_start_idx) - 1)
    left = max(0, min(int(left), int(sample_count) - 1))
    right = max(0, min(int(right), int(sample_count) - 1))
    return left, right


def _resolve_track_run(
    root: zarr.Group,
    track_kinematics_run: str,
    *,
    track_scope: str,
) -> tuple[zarr.Group, str, str, str]:
    parent = root.get("analysis/track_kinematics_runs")
    if parent is None:
        raise ValueError("No analysis/track_kinematics_runs group found.")

    spec = str(track_kinematics_run).strip().strip("/")
    parts = spec.split("/")
    if spec.startswith("analysis/track_kinematics_runs/") and len(parts) >= 4:
        scope, run_name = parts[2], parts[3]
    elif len(parts) == 2 and parts[0] in parent:
        scope, run_name = parts
    else:
        scope = track_scope
        if scope not in parent:
            raise ValueError(f"Track kinematics scope {scope!r} not found.")
        run_name = parent[scope].attrs.get("latest") if spec == "latest" else spec

    if not run_name:
        raise ValueError(f"No track kinematics run resolved for {track_kinematics_run!r}.")
    if scope not in parent or run_name not in parent[scope]:
        raise ValueError(f"Track kinematics run {scope}/{run_name} not found.")

    run_path = f"analysis/track_kinematics_runs/{scope}/{run_name}"
    return parent[scope][run_name], str(run_name), run_path, str(scope)


def _load_eye_gaze_frame_series(
    root: zarr.Group,
    *,
    eye_angle_run: str,
    eye_angle_family: str,
    frames: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    try:
        return load_eye_gaze_frame_series(
            root,
            eye_angle_run=eye_angle_run,
            eye_angle_family=eye_angle_family,
            frames=frames,
            allowed_families=EYE_ANGLE_FAMILIES,
        )
    except EyeAngleIOError as exc:
        raise ValueError(str(exc)) from exc


def _eye_epoch_stats(
    *,
    series: Mapping[str, np.ndarray],
    epoch_slice: slice,
    min_valid_fraction: float,
    vergence_threshold_deg: Optional[float],
) -> dict[str, Any]:
    window_size = max(0, int(epoch_slice.stop) - int(epoch_slice.start))
    if window_size <= 0:
        return {
            "sample_count": 0,
            "valid_fraction": 0.0,
            "window_valid": False,
            "left_mean": float("nan"),
            "right_mean": float("nan"),
            "vergence_mean": float("nan"),
            "vergence_signed_mean": float("nan"),
            "vergence_max": float("nan"),
            "vergence_range": float("nan"),
            "vergence_std": float("nan"),
            "converged_fraction": float("nan"),
            "reason": "insufficient_eye_window",
        }

    left = np.asarray(series["left_gaze_deg"][epoch_slice], dtype=np.float64)
    right = np.asarray(series["right_gaze_deg"][epoch_slice], dtype=np.float64)
    vergence = np.asarray(series["vergence_gaze_deg"][epoch_slice], dtype=np.float64)
    signed = np.asarray(series["vergence_gaze_signed_deg"][epoch_slice], dtype=np.float64)
    valid = np.asarray(series["valid_frame"][epoch_slice], dtype=bool)
    valid &= np.isfinite(left) & np.isfinite(right) & np.isfinite(vergence)
    valid_count = int(np.count_nonzero(valid))
    valid_fraction = float(valid_count / window_size) if window_size > 0 else 0.0
    window_valid = bool(valid_count > 0 and valid_fraction >= float(min_valid_fraction))
    if not window_valid:
        reason = "insufficient_eye_window" if valid_count == 0 else "eye_angle_contains_gap"
        return {
            "sample_count": valid_count,
            "valid_fraction": valid_fraction,
            "window_valid": False,
            "left_mean": float("nan"),
            "right_mean": float("nan"),
            "vergence_mean": float("nan"),
            "vergence_signed_mean": float("nan"),
            "vergence_max": float("nan"),
            "vergence_range": float("nan"),
            "vergence_std": float("nan"),
            "converged_fraction": float("nan"),
            "reason": reason,
        }

    valid_left = left[valid]
    valid_right = right[valid]
    valid_vergence = vergence[valid]
    valid_signed = signed[valid & np.isfinite(signed)]
    if vergence_threshold_deg is None:
        converged_fraction = float("nan")
    else:
        converged_fraction = float(
            np.count_nonzero(valid_vergence >= float(vergence_threshold_deg)) / valid_vergence.size
        )

    return {
        "sample_count": valid_count,
        "valid_fraction": valid_fraction,
        "window_valid": True,
        "left_mean": float(np.mean(valid_left)),
        "right_mean": float(np.mean(valid_right)),
        "vergence_mean": float(np.mean(valid_vergence)),
        "vergence_signed_mean": float(np.mean(valid_signed)) if valid_signed.size else float("nan"),
        "vergence_max": float(np.max(valid_vergence)),
        "vergence_range": float(np.max(valid_vergence) - np.min(valid_vergence)),
        "vergence_std": float(np.std(valid_vergence)),
        "converged_fraction": converged_fraction,
        "reason": None,
    }


def _build_metrics_for_movement(
    *,
    bouts: np.ndarray,
    frames: np.ndarray,
    times: np.ndarray,
    physical_speed_mm: np.ndarray,
    fps: float,
    threshold_mm_s: float,
    boundary_constraint: str,
    boundary_margin_frames: int,
    boundary_margin_s: float,
    delta_seconds: Optional[np.ndarray],
    transition_valid: Optional[np.ndarray],
    sample_valid: Optional[np.ndarray],
    path_distance_mm: Optional[np.ndarray],
    path_distance_px: Optional[np.ndarray],
) -> np.ndarray:
    metrics = np.zeros(len(bouts), dtype=_movement_metrics_dtype())
    if len(metrics) == 0:
        return metrics

    speed = np.asarray(physical_speed_mm, dtype=np.float64)
    frame_values = np.asarray(frames, dtype=np.int64)
    time_values = np.asarray(times, dtype=np.float64)
    if speed.shape[0] != frame_values.shape[0] or time_values.shape[0] != frame_values.shape[0]:
        raise ValueError("Physical speed, frames, and times must have matching lengths.")
    if boundary_constraint not in PHYSICAL_ACTIVE_BOUNDARY_CONSTRAINTS:
        expected = ", ".join(PHYSICAL_ACTIVE_BOUNDARY_CONSTRAINTS)
        raise ValueError(f"Unsupported physical active boundary constraint {boundary_constraint!r}; expected one of: {expected}")

    if delta_seconds is None:
        delta_seconds_arr = np.zeros(frame_values.shape[0], dtype=np.float64)
        if frame_values.shape[0] > 1:
            delta = np.diff(time_values)
            delta_seconds_arr[1:] = np.where(np.isfinite(delta) & (delta > 0), delta, 0.0)
    else:
        delta_seconds_arr = np.asarray(delta_seconds, dtype=np.float64)
        if delta_seconds_arr.shape[0] != frame_values.shape[0]:
            raise ValueError("delta_seconds length must match frames length.")

    transition_mask = (
        np.asarray(transition_valid, dtype=bool)
        if transition_valid is not None
        else np.ones(frame_values.shape[0], dtype=bool)
    )
    sample_mask = (
        np.asarray(sample_valid, dtype=bool)
        if sample_valid is not None
        else np.ones(frame_values.shape[0], dtype=bool)
    )
    if transition_mask.shape[0] != frame_values.shape[0] or sample_mask.shape[0] != frame_values.shape[0]:
        raise ValueError("Validity arrays must match frames length.")

    path_mm = None if path_distance_mm is None else np.asarray(path_distance_mm, dtype=np.float64)
    path_px = None if path_distance_px is None else np.asarray(path_distance_px, dtype=np.float64)
    if path_mm is not None and path_mm.shape[0] != frame_values.shape[0]:
        raise ValueError("path_distance_mm length must match frames length.")
    if path_px is not None and path_px.shape[0] != frame_values.shape[0]:
        raise ValueError("path_distance_px length must match frames length.")

    frame_to_index = {int(frame): idx for idx, frame in enumerate(frame_values)}
    start_frames = np.asarray(bouts["start_frame"], dtype=np.int64)
    end_frames = np.asarray(bouts["end_frame"], dtype=np.int64)
    bout_ids = (
        np.asarray(bouts["bout_id"], dtype=np.int32)
        if "bout_id" in (bouts.dtype.names or ())
        else np.arange(1, len(bouts) + 1, dtype=np.int32)
    )
    core_starts = _field_or_default(bouts, "core_start_frame", -1).astype(np.int64)
    core_ends = _field_or_default(bouts, "core_end_frame", -1).astype(np.int64)
    detector_duration_s = _float_field_or_nan(bouts, "duration_s")
    detector_observed_duration_s = _float_field_or_nan(bouts, "observed_duration_s")
    detector_core_duration_s = _float_field_or_nan(bouts, "core_duration_s")

    sorted_rows = np.argsort(start_frames)
    previous_end_indices = np.full(len(bouts), -1, dtype=np.int64)
    next_start_indices = np.full(len(bouts), -1, dtype=np.int64)
    for order_idx, row_idx in enumerate(sorted_rows):
        if order_idx > 0:
            previous_row = int(sorted_rows[order_idx - 1])
            previous_end_indices[int(row_idx)] = int(frame_to_index.get(int(end_frames[previous_row]), -1))
        if order_idx + 1 < len(sorted_rows):
            next_row = int(sorted_rows[order_idx + 1])
            next_start_indices[int(row_idx)] = int(frame_to_index.get(int(start_frames[next_row]), -1))

    threshold = float(threshold_mm_s)
    if threshold < 0:
        raise ValueError("physical_active_threshold_mm_s must be >= 0.")

    for row_idx, (bout_id, start_frame, end_frame, core_start, core_end) in enumerate(
        zip(bout_ids, start_frames, end_frames, core_starts, core_ends)
    ):
        reasons: list[str] = []
        metrics[row_idx]["bout_id"] = int(bout_id)
        metrics[row_idx]["source_start_frame"] = int(start_frame)
        metrics[row_idx]["source_end_frame"] = int(end_frame)
        metrics[row_idx]["source_core_start_frame"] = int(core_start)
        metrics[row_idx]["source_core_end_frame"] = int(core_end)
        metrics[row_idx]["detector_duration_s"] = float(detector_duration_s[row_idx])
        metrics[row_idx]["detector_observed_duration_s"] = float(detector_observed_duration_s[row_idx])
        metrics[row_idx]["detector_core_duration_s"] = float(detector_core_duration_s[row_idx])
        metrics[row_idx]["physical_active_start_frame"] = -1
        metrics[row_idx]["physical_active_end_frame"] = -1
        metrics[row_idx]["physical_active_threshold_mm_s"] = threshold
        metrics[row_idx]["physical_active_boundary_margin_s"] = float(boundary_margin_s)
        metrics[row_idx]["physical_active_boundary_policy_bytes"] = PHYSICAL_ACTIVE_BOUNDARY_POLICY.encode("utf-8")
        metrics[row_idx]["physical_active_boundary_constraint_bytes"] = str(boundary_constraint).encode("utf-8")
        for field in (
            "physical_active_start_time_s",
            "physical_active_end_time_s",
            "physical_active_duration_s",
            "physical_active_observed_duration_s",
            "physical_active_start_time_s_interpolated",
            "physical_active_end_time_s_interpolated",
            "physical_active_duration_s_interpolated",
            "physical_active_valid_transition_fraction",
            "physical_active_path_length_mm",
            "physical_active_path_length_px",
            "physical_active_mean_speed_mm_s",
            "physical_active_peak_speed_mm_s",
        ):
            metrics[row_idx][field] = float("nan")

        start_idx = frame_to_index.get(int(start_frame))
        end_idx = frame_to_index.get(int(end_frame))
        if start_idx is None or end_idx is None or end_idx < start_idx:
            reasons.append("source_bout_missing")
            metrics[row_idx]["failure_reason_bytes"] = ";".join(reasons).encode("utf-8")
            continue

        search_start, search_end = _physical_search_bounds(
            start_idx=start_idx,
            end_idx=end_idx,
            previous_end_idx=int(previous_end_indices[row_idx]),
            next_start_idx=int(next_start_indices[row_idx]),
            sample_count=frame_values.shape[0],
            margin_frames=int(boundary_margin_frames),
            boundary_constraint=boundary_constraint,
        )
        if search_end < search_start:
            reasons.append("physical_active_search_window_invalid")
            metrics[row_idx]["failure_reason_bytes"] = ";".join(reasons).encode("utf-8")
            continue

        search_slice = slice(search_start, search_end + 1)
        active_mask = (
            np.isfinite(speed[search_slice])
            & sample_mask[search_slice]
            & (speed[search_slice] > threshold)
        )
        active_offsets = np.flatnonzero(active_mask)
        if active_offsets.size == 0:
            reasons.append("no_physical_active_samples")
            metrics[row_idx]["failure_reason_bytes"] = ";".join(reasons).encode("utf-8")
            continue

        active_start_idx = int(search_start + active_offsets[0])
        active_end_idx = int(search_start + active_offsets[-1])
        active_span = slice(active_start_idx, active_end_idx + 1)
        span_transition_valid = transition_mask[active_span]
        span_valid_count = int(np.count_nonzero(span_transition_valid))
        span_count = int(active_end_idx - active_start_idx + 1)
        valid_fraction = float(span_valid_count / span_count) if span_count > 0 else float("nan")

        metrics[row_idx]["physical_active_start_frame"] = int(frame_values[active_start_idx])
        metrics[row_idx]["physical_active_end_frame"] = int(frame_values[active_end_idx])
        metrics[row_idx]["physical_active_start_time_s"] = float(time_values[active_start_idx])
        metrics[row_idx]["physical_active_end_time_s"] = float(time_values[active_end_idx])
        metrics[row_idx]["physical_active_duration_s"] = (
            float((int(frame_values[active_end_idx]) - int(frame_values[active_start_idx]) + 1) / float(fps))
            if fps > 0
            else float("nan")
        )
        metrics[row_idx]["physical_active_observed_duration_s"] = float(
            np.sum(delta_seconds_arr[active_span][span_transition_valid])
        )
        metrics[row_idx]["physical_active_sample_count"] = int(active_offsets.size)
        metrics[row_idx]["physical_active_valid_transition_count"] = span_valid_count
        metrics[row_idx]["physical_active_valid_transition_fraction"] = valid_fraction
        if path_mm is not None:
            metrics[row_idx]["physical_active_path_length_mm"] = float(
                np.sum(path_mm[active_span][span_transition_valid])
            )
        if path_px is not None:
            metrics[row_idx]["physical_active_path_length_px"] = float(
                np.sum(path_px[active_span][span_transition_valid])
            )
        observed_duration = float(metrics[row_idx]["physical_active_observed_duration_s"])
        path_length_mm = float(metrics[row_idx]["physical_active_path_length_mm"])
        metrics[row_idx]["physical_active_mean_speed_mm_s"] = (
            path_length_mm / observed_duration
            if observed_duration > 0 and np.isfinite(path_length_mm)
            else float("nan")
        )
        active_speed_values = speed[search_slice][active_mask]
        metrics[row_idx]["physical_active_peak_speed_mm_s"] = float(np.max(active_speed_values))

        start_time_interp, start_interp_valid = _threshold_crossing_time_from_samples(
            values=speed,
            times=time_values,
            threshold=threshold,
            below_idx=active_start_idx - 1,
            above_idx=active_start_idx,
        )
        end_time_interp, end_interp_valid = _threshold_crossing_time_from_samples(
            values=speed,
            times=time_values,
            threshold=threshold,
            below_idx=active_end_idx + 1,
            above_idx=active_end_idx,
        )
        metrics[row_idx]["physical_active_start_time_s_interpolated"] = start_time_interp
        metrics[row_idx]["physical_active_end_time_s_interpolated"] = end_time_interp
        metrics[row_idx]["physical_active_start_time_interpolated_valid"] = start_interp_valid
        metrics[row_idx]["physical_active_end_time_interpolated_valid"] = end_interp_valid
        if start_interp_valid and end_interp_valid and end_time_interp >= start_time_interp:
            metrics[row_idx]["physical_active_duration_s_interpolated"] = float(
                end_time_interp - start_time_interp
            )

        if span_valid_count != span_count:
            reasons.append("physical_active_contains_gap")
        metrics[row_idx]["physical_active_valid"] = len(reasons) == 0
        metrics[row_idx]["failure_reason_bytes"] = (
            ";".join(reasons).encode("utf-8") if reasons else b"ok"
        )

    return metrics


def _build_metrics_for_eye_gaze(
    *,
    bouts: np.ndarray,
    frames: np.ndarray,
    eye_series: Mapping[str, np.ndarray],
    pre_post_mode: str,
    pre_window_frames: int,
    post_window_frames: int,
    within_window: str,
    eye_validity_min_fraction: float,
    vergence_threshold_deg: Optional[float],
) -> np.ndarray:
    metrics = np.zeros(len(bouts), dtype=_eye_gaze_metrics_dtype())
    if len(metrics) == 0:
        return metrics

    for field in (
        "pre_left_gaze_mean_deg",
        "pre_right_gaze_mean_deg",
        "pre_vergence_gaze_mean_deg",
        "pre_vergence_gaze_signed_mean_deg",
        "pre_vergence_gaze_std_deg",
        "pre_vergence_gaze_valid_fraction",
        "pre_converged_fraction",
        "post_left_gaze_mean_deg",
        "post_right_gaze_mean_deg",
        "post_vergence_gaze_mean_deg",
        "post_vergence_gaze_signed_mean_deg",
        "post_vergence_gaze_std_deg",
        "post_vergence_gaze_valid_fraction",
        "post_converged_fraction",
        "within_bout_left_gaze_mean_deg",
        "within_bout_right_gaze_mean_deg",
        "within_bout_vergence_gaze_mean_deg",
        "within_bout_vergence_gaze_signed_mean_deg",
        "within_bout_vergence_gaze_max_deg",
        "within_bout_vergence_gaze_range_deg",
        "within_bout_vergence_gaze_std_deg",
        "within_bout_vergence_gaze_valid_fraction",
        "within_bout_converged_fraction",
    ):
        metrics[field] = float("nan")

    frame_to_index = {int(frame): idx for idx, frame in enumerate(np.asarray(frames, dtype=np.int64))}
    start_frames = np.asarray(bouts["start_frame"], dtype=np.int64)
    end_frames = np.asarray(bouts["end_frame"], dtype=np.int64)
    bout_ids = (
        np.asarray(bouts["bout_id"], dtype=np.int32)
        if "bout_id" in (bouts.dtype.names or ())
        else np.arange(1, len(bouts) + 1, dtype=np.int32)
    )
    core_starts = _field_or_default(bouts, "core_start_frame", -1).astype(np.int64)
    core_ends = _field_or_default(bouts, "core_end_frame", -1).astype(np.int64)

    sorted_rows = np.argsort(start_frames)
    previous_end_indices = np.full(len(bouts), -1, dtype=np.int64)
    next_start_indices = np.full(len(bouts), -1, dtype=np.int64)
    for order_idx, row_idx in enumerate(sorted_rows):
        if order_idx > 0:
            previous_row = int(sorted_rows[order_idx - 1])
            previous_end_indices[int(row_idx)] = int(frame_to_index.get(int(end_frames[previous_row]), -1))
        if order_idx + 1 < len(sorted_rows):
            next_row = int(sorted_rows[order_idx + 1])
            next_start_indices[int(row_idx)] = int(frame_to_index.get(int(start_frames[next_row]), -1))

    for row_idx, (bout_id, start_frame, end_frame, core_start, core_end) in enumerate(
        zip(bout_ids, start_frames, end_frames, core_starts, core_ends)
    ):
        reasons: list[str] = []
        metrics[row_idx]["bout_id"] = int(bout_id)
        metrics[row_idx]["source_start_frame"] = int(start_frame)
        metrics[row_idx]["source_end_frame"] = int(end_frame)
        metrics[row_idx]["source_core_start_frame"] = int(core_start)
        metrics[row_idx]["source_core_end_frame"] = int(core_end)
        metrics[row_idx]["pre_epoch_start_frame"] = -1
        metrics[row_idx]["pre_epoch_end_frame"] = -1
        metrics[row_idx]["post_epoch_start_frame"] = -1
        metrics[row_idx]["post_epoch_end_frame"] = -1
        metrics[row_idx]["within_epoch_start_frame"] = -1
        metrics[row_idx]["within_epoch_end_frame"] = -1

        start_idx = frame_to_index.get(int(start_frame))
        end_idx = frame_to_index.get(int(end_frame))
        if start_idx is None or end_idx is None or end_idx < start_idx:
            reasons.append("source_bout_missing")
            metrics[row_idx]["failure_reason_bytes"] = ";".join(reasons).encode("utf-8")
            continue

        if pre_post_mode == "fixed_window":
            pre_slice = slice(max(0, start_idx - pre_window_frames), start_idx)
            post_slice = slice(end_idx + 1, min(len(frames), end_idx + 1 + post_window_frames))
        else:
            previous_end_idx = int(previous_end_indices[row_idx])
            next_start_idx = int(next_start_indices[row_idx])
            pre_slice = slice(previous_end_idx + 1, start_idx) if 0 <= previous_end_idx < start_idx else slice(start_idx, start_idx)
            post_slice = slice(end_idx + 1, next_start_idx) if next_start_idx >= 0 and end_idx < next_start_idx else slice(end_idx + 1, end_idx + 1)

        within_start_frame = core_start if within_window == "core_start_end" and core_start >= 0 else start_frame
        within_end_frame = core_end if within_window == "core_start_end" and core_end >= 0 else end_frame
        within_start_idx = frame_to_index.get(int(within_start_frame))
        within_end_idx = frame_to_index.get(int(within_end_frame))
        if within_start_idx is None or within_end_idx is None or within_end_idx < within_start_idx:
            within_slice = slice(start_idx, start_idx)
            reasons.append("source_bout_missing")
        else:
            within_slice = slice(within_start_idx, within_end_idx + 1)

        (
            metrics[row_idx]["pre_epoch_start_frame"],
            metrics[row_idx]["pre_epoch_end_frame"],
        ) = _epoch_bounds(frames, pre_slice)
        (
            metrics[row_idx]["post_epoch_start_frame"],
            metrics[row_idx]["post_epoch_end_frame"],
        ) = _epoch_bounds(frames, post_slice)
        (
            metrics[row_idx]["within_epoch_start_frame"],
            metrics[row_idx]["within_epoch_end_frame"],
        ) = _epoch_bounds(frames, within_slice)

        pre_stats = _eye_epoch_stats(
            series=eye_series,
            epoch_slice=pre_slice,
            min_valid_fraction=eye_validity_min_fraction,
            vergence_threshold_deg=vergence_threshold_deg,
        )
        post_stats = _eye_epoch_stats(
            series=eye_series,
            epoch_slice=post_slice,
            min_valid_fraction=eye_validity_min_fraction,
            vergence_threshold_deg=vergence_threshold_deg,
        )
        within_stats = _eye_epoch_stats(
            series=eye_series,
            epoch_slice=within_slice,
            min_valid_fraction=eye_validity_min_fraction,
            vergence_threshold_deg=vergence_threshold_deg,
        )

        metrics[row_idx]["pre_left_gaze_mean_deg"] = pre_stats["left_mean"]
        metrics[row_idx]["pre_right_gaze_mean_deg"] = pre_stats["right_mean"]
        metrics[row_idx]["pre_vergence_gaze_mean_deg"] = pre_stats["vergence_mean"]
        metrics[row_idx]["pre_vergence_gaze_signed_mean_deg"] = pre_stats["vergence_signed_mean"]
        metrics[row_idx]["pre_vergence_gaze_std_deg"] = pre_stats["vergence_std"]
        metrics[row_idx]["pre_vergence_gaze_valid_fraction"] = pre_stats["valid_fraction"]
        metrics[row_idx]["pre_converged_fraction"] = pre_stats["converged_fraction"]
        metrics[row_idx]["pre_eye_window_valid"] = pre_stats["window_valid"]
        metrics[row_idx]["pre_eye_sample_count"] = pre_stats["sample_count"]

        metrics[row_idx]["post_left_gaze_mean_deg"] = post_stats["left_mean"]
        metrics[row_idx]["post_right_gaze_mean_deg"] = post_stats["right_mean"]
        metrics[row_idx]["post_vergence_gaze_mean_deg"] = post_stats["vergence_mean"]
        metrics[row_idx]["post_vergence_gaze_signed_mean_deg"] = post_stats["vergence_signed_mean"]
        metrics[row_idx]["post_vergence_gaze_std_deg"] = post_stats["vergence_std"]
        metrics[row_idx]["post_vergence_gaze_valid_fraction"] = post_stats["valid_fraction"]
        metrics[row_idx]["post_converged_fraction"] = post_stats["converged_fraction"]
        metrics[row_idx]["post_eye_window_valid"] = post_stats["window_valid"]
        metrics[row_idx]["post_eye_sample_count"] = post_stats["sample_count"]

        metrics[row_idx]["within_bout_left_gaze_mean_deg"] = within_stats["left_mean"]
        metrics[row_idx]["within_bout_right_gaze_mean_deg"] = within_stats["right_mean"]
        metrics[row_idx]["within_bout_vergence_gaze_mean_deg"] = within_stats["vergence_mean"]
        metrics[row_idx]["within_bout_vergence_gaze_signed_mean_deg"] = within_stats["vergence_signed_mean"]
        metrics[row_idx]["within_bout_vergence_gaze_max_deg"] = within_stats["vergence_max"]
        metrics[row_idx]["within_bout_vergence_gaze_range_deg"] = within_stats["vergence_range"]
        metrics[row_idx]["within_bout_vergence_gaze_std_deg"] = within_stats["vergence_std"]
        metrics[row_idx]["within_bout_vergence_gaze_valid_fraction"] = within_stats["valid_fraction"]
        metrics[row_idx]["within_bout_converged_fraction"] = within_stats["converged_fraction"]
        metrics[row_idx]["within_eye_window_valid"] = within_stats["window_valid"]
        metrics[row_idx]["within_eye_sample_count"] = within_stats["sample_count"]

        for prefix, stats in (
            ("pre", pre_stats),
            ("post", post_stats),
            ("within", within_stats),
        ):
            if stats["reason"] is not None:
                reasons.append(f"{prefix}_{stats['reason']}")

        unique_reasons = list(dict.fromkeys(reasons))
        metrics[row_idx]["failure_reason_bytes"] = (
            ";".join(unique_reasons).encode("utf-8") if unique_reasons else b"ok"
        )

    return metrics


def _build_metrics_for_heading(
    *,
    bouts: np.ndarray,
    peak_events: Optional[np.ndarray],
    frames: np.ndarray,
    times: np.ndarray,
    headings: np.ndarray,
    transition_valid: Optional[np.ndarray],
    sample_valid: Optional[np.ndarray],
    positions_mm: Optional[np.ndarray],
    positions_px: Optional[np.ndarray],
    fps: float,
    pre_post_mode: str,
    pre_window_frames: int,
    post_window_frames: int,
    within_window: str,
    derivative_threshold_deg_s: float,
    dominant_frequency_enabled: bool,
    dominant_frequency_min_samples: int,
    dominant_frequency_detrend: bool,
) -> np.ndarray:
    metrics = np.zeros(len(bouts), dtype=_metrics_dtype())
    if len(metrics) == 0:
        return metrics

    frame_to_index = {int(frame): idx for idx, frame in enumerate(np.asarray(frames, dtype=np.int64))}
    start_frames = np.asarray(bouts["start_frame"], dtype=np.int64)
    end_frames = np.asarray(bouts["end_frame"], dtype=np.int64)
    bout_ids = (
        np.asarray(bouts["bout_id"], dtype=np.int32)
        if "bout_id" in (bouts.dtype.names or ())
        else np.arange(1, len(bouts) + 1, dtype=np.int32)
    )
    core_starts = _field_or_default(bouts, "core_start_frame", -1).astype(np.int64)
    core_ends = _field_or_default(bouts, "core_end_frame", -1).astype(np.int64)
    source_core_start_time_s_interpolated = _float_field_or_nan(
        bouts,
        "core_start_time_s_interpolated",
    )
    source_core_end_time_s_interpolated = _float_field_or_nan(
        bouts,
        "core_end_time_s_interpolated",
    )
    source_core_duration_s_interpolated = _float_field_or_nan(
        bouts,
        "core_duration_s_interpolated",
    )
    source_core_start_time_interpolated_valid = _bool_field_or_false(
        bouts,
        "core_start_time_interpolated_valid",
    )
    source_core_end_time_interpolated_valid = _bool_field_or_false(
        bouts,
        "core_end_time_interpolated_valid",
    )
    aligned_peak_events = (
        peak_events
        if peak_events is not None
        and len(peak_events) == len(bouts)
        and peak_events.dtype.names is not None
        and _records_align_by_bout_id(bouts, peak_events)
        else None
    )
    if aligned_peak_events is None:
        source_peak_frame = np.full(len(bouts), -1, dtype=np.int64)
        source_peak_time_s = np.full(len(bouts), float("nan"), dtype=np.float64)
        source_peak_signal_value_mm_s = np.full(len(bouts), float("nan"), dtype=np.float64)
        source_peak_prominence_mm_s = np.full(len(bouts), float("nan"), dtype=np.float64)
        source_peak_width_s = np.full(len(bouts), float("nan"), dtype=np.float64)
        source_peak_width_height_mm_s = np.full(len(bouts), float("nan"), dtype=np.float64)
        source_peak_left_width_frame_interpolated = np.full(len(bouts), float("nan"), dtype=np.float64)
        source_peak_right_width_frame_interpolated = np.full(len(bouts), float("nan"), dtype=np.float64)
    else:
        source_peak_frame = _field_or_default(aligned_peak_events, "peak_frame", -1).astype(np.int64)
        source_peak_time_s = _float_field_or_nan(aligned_peak_events, "peak_time_s")
        source_peak_signal_value_mm_s = _float_field_or_nan(aligned_peak_events, "peak_signal_value_mm_s")
        source_peak_prominence_mm_s = _float_field_or_nan(aligned_peak_events, "peak_prominence_mm_s")
        source_peak_width_s = _float_field_or_nan(aligned_peak_events, "peak_width_s")
        source_peak_width_height_mm_s = _float_field_or_nan(aligned_peak_events, "peak_width_height_mm_s")
        source_peak_left_width_frame_interpolated = _float_field_or_nan(
            aligned_peak_events,
            "left_width_frame_interpolated",
        )
        source_peak_right_width_frame_interpolated = _float_field_or_nan(
            aligned_peak_events,
            "right_width_frame_interpolated",
        )
    source_peak_left_width_time_s = (
        source_peak_left_width_frame_interpolated / float(fps)
        if fps > 0
        else np.full(len(bouts), float("nan"), dtype=np.float64)
    )
    source_peak_right_width_time_s = (
        source_peak_right_width_frame_interpolated / float(fps)
        if fps > 0
        else np.full(len(bouts), float("nan"), dtype=np.float64)
    )
    source_peak_boundary_mode_bytes = _bytes_field_or_empty(aligned_peak_events, "boundary_mode", len(bouts))
    source_peak_shape_split_policy_bytes = _bytes_field_or_empty(
        aligned_peak_events,
        "shape_split_policy",
        len(bouts),
    )
    sorted_rows = np.argsort(start_frames)
    previous_end_indices = np.full(len(bouts), -1, dtype=np.int64)
    next_start_indices = np.full(len(bouts), -1, dtype=np.int64)
    for order_idx, row_idx in enumerate(sorted_rows):
        if order_idx > 0:
            previous_row = int(sorted_rows[order_idx - 1])
            previous_end_indices[int(row_idx)] = int(frame_to_index.get(int(end_frames[previous_row]), -1))
        if order_idx + 1 < len(sorted_rows):
            next_row = int(sorted_rows[order_idx + 1])
            next_start_indices[int(row_idx)] = int(frame_to_index.get(int(start_frames[next_row]), -1))

    for row_idx, (bout_id, start_frame, end_frame, core_start, core_end) in enumerate(
        zip(bout_ids, start_frames, end_frames, core_starts, core_ends)
    ):
        reasons: list[str] = []
        metrics[row_idx]["bout_id"] = int(bout_id)
        metrics[row_idx]["source_start_frame"] = int(start_frame)
        metrics[row_idx]["source_end_frame"] = int(end_frame)
        metrics[row_idx]["source_core_start_frame"] = int(core_start)
        metrics[row_idx]["source_core_end_frame"] = int(core_end)
        metrics[row_idx]["source_core_start_time_s_interpolated"] = float(
            source_core_start_time_s_interpolated[row_idx]
        )
        metrics[row_idx]["source_core_end_time_s_interpolated"] = float(
            source_core_end_time_s_interpolated[row_idx]
        )
        metrics[row_idx]["source_core_duration_s_interpolated"] = float(
            source_core_duration_s_interpolated[row_idx]
        )
        metrics[row_idx]["source_core_start_time_interpolated_valid"] = bool(
            source_core_start_time_interpolated_valid[row_idx]
        )
        metrics[row_idx]["source_core_end_time_interpolated_valid"] = bool(
            source_core_end_time_interpolated_valid[row_idx]
        )
        metrics[row_idx]["source_peak_frame"] = int(source_peak_frame[row_idx])
        metrics[row_idx]["source_peak_time_s"] = float(source_peak_time_s[row_idx])
        metrics[row_idx]["source_peak_signal_value_mm_s"] = float(source_peak_signal_value_mm_s[row_idx])
        metrics[row_idx]["source_peak_prominence_mm_s"] = float(source_peak_prominence_mm_s[row_idx])
        metrics[row_idx]["source_peak_width_s"] = float(source_peak_width_s[row_idx])
        metrics[row_idx]["source_peak_width_height_mm_s"] = float(source_peak_width_height_mm_s[row_idx])
        metrics[row_idx]["source_peak_left_width_frame_interpolated"] = float(
            source_peak_left_width_frame_interpolated[row_idx]
        )
        metrics[row_idx]["source_peak_right_width_frame_interpolated"] = float(
            source_peak_right_width_frame_interpolated[row_idx]
        )
        metrics[row_idx]["source_peak_left_width_time_s"] = float(source_peak_left_width_time_s[row_idx])
        metrics[row_idx]["source_peak_right_width_time_s"] = float(source_peak_right_width_time_s[row_idx])
        metrics[row_idx]["source_peak_boundary_mode_bytes"] = source_peak_boundary_mode_bytes[row_idx]
        metrics[row_idx]["source_peak_shape_split_policy_bytes"] = source_peak_shape_split_policy_bytes[row_idx]
        metrics[row_idx]["pre_epoch_start_frame"] = -1
        metrics[row_idx]["pre_epoch_end_frame"] = -1
        metrics[row_idx]["post_epoch_start_frame"] = -1
        metrics[row_idx]["post_epoch_end_frame"] = -1
        for field in (
            "pre_heading_mean_deg",
            "post_heading_mean_deg",
            "net_delta_heading_deg",
            "abs_net_delta_heading_deg",
            "pre_position_mean_x_mm",
            "pre_position_mean_y_mm",
            "post_position_mean_x_mm",
            "post_position_mean_y_mm",
            "interbout_epoch_displacement_mm",
            "pre_position_mean_x_px",
            "pre_position_mean_y_px",
            "post_position_mean_x_px",
            "post_position_mean_y_px",
            "interbout_epoch_displacement_px",
            "within_heading_range_deg",
            "within_heading_peak_to_peak_deg",
            "within_heading_path_deg",
            "within_heading_std_deg",
            "within_heading_dominant_frequency_hz",
            "within_angular_velocity_mean_deg_s",
            "within_angular_speed_mean_deg_s",
            "within_angular_speed_max_deg_s",
            "within_angular_velocity_std_deg_s",
        ):
            metrics[row_idx][field] = float("nan")
        metrics[row_idx]["within_heading_zero_crossings"] = 0

        start_idx = frame_to_index.get(int(start_frame))
        end_idx = frame_to_index.get(int(end_frame))
        if start_idx is None or end_idx is None or end_idx < start_idx:
            reasons.append("source_bout_missing")
            metrics[row_idx]["failure_reason_bytes"] = ";".join(reasons).encode("utf-8")
            continue

        if pre_post_mode == "fixed_window":
            pre_slice = slice(max(0, start_idx - pre_window_frames), start_idx)
            post_slice = slice(end_idx + 1, min(len(headings), end_idx + 1 + post_window_frames))
        else:
            previous_end_idx = int(previous_end_indices[row_idx])
            next_start_idx = int(next_start_indices[row_idx])
            if previous_end_idx >= 0 and previous_end_idx < start_idx:
                pre_slice = slice(previous_end_idx + 1, start_idx)
            else:
                pre_slice = slice(start_idx, start_idx)
            if next_start_idx >= 0 and end_idx < next_start_idx:
                post_slice = slice(end_idx + 1, next_start_idx)
            else:
                post_slice = slice(end_idx + 1, end_idx + 1)
        (
            metrics[row_idx]["pre_epoch_start_frame"],
            metrics[row_idx]["pre_epoch_end_frame"],
        ) = _epoch_bounds(frames, pre_slice)
        (
            metrics[row_idx]["post_epoch_start_frame"],
            metrics[row_idx]["post_epoch_end_frame"],
        ) = _epoch_bounds(frames, post_slice)

        pre = np.asarray(headings[pre_slice], dtype=np.float64)
        post = np.asarray(headings[post_slice], dtype=np.float64)

        pre_valid_count = int(np.count_nonzero(np.isfinite(pre)))
        post_valid_count = int(np.count_nonzero(np.isfinite(post)))
        metrics[row_idx]["pre_window_sample_count"] = pre_valid_count
        metrics[row_idx]["post_window_sample_count"] = post_valid_count
        if pre_post_mode == "fixed_window":
            pre_valid = (
                pre.size == pre_window_frames
                and pre_window_frames > 0
                and pre_valid_count == pre_window_frames
            )
            post_valid = (
                post.size == post_window_frames
                and post_window_frames > 0
                and post_valid_count == post_window_frames
            )
        else:
            pre_valid = pre.size > 0 and pre_valid_count == pre.size
            post_valid = post.size > 0 and post_valid_count == post.size
        metrics[row_idx]["pre_window_valid"] = pre_valid
        metrics[row_idx]["post_window_valid"] = post_valid
        if not pre_valid:
            reasons.append(
                "insufficient_pre_window"
                if pre.size == 0 or (pre_post_mode == "fixed_window" and pre.size < pre_window_frames)
                else "heading_contains_gap"
            )
        if not post_valid:
            reasons.append(
                "insufficient_post_window"
                if post.size == 0 or (pre_post_mode == "fixed_window" and post.size < post_window_frames)
                else "heading_contains_gap"
            )

        if pre_valid:
            metrics[row_idx]["pre_heading_mean_deg"] = _circular_mean_deg(pre)
        if post_valid:
            metrics[row_idx]["post_heading_mean_deg"] = _circular_mean_deg(post)
        if pre_valid and post_valid:
            delta = _wrap_degrees(
                float(metrics[row_idx]["post_heading_mean_deg"])
                - float(metrics[row_idx]["pre_heading_mean_deg"])
            )
            metrics[row_idx]["net_delta_heading_deg"] = delta
            metrics[row_idx]["abs_net_delta_heading_deg"] = abs(delta)

        pre_x_mm, pre_y_mm, pre_count_mm, pre_valid_mm = _position_epoch_stats(positions_mm, pre_slice)
        post_x_mm, post_y_mm, post_count_mm, post_valid_mm = _position_epoch_stats(positions_mm, post_slice)
        pre_x_px, pre_y_px, pre_count_px, pre_valid_px = _position_epoch_stats(positions_px, pre_slice)
        post_x_px, post_y_px, post_count_px, post_valid_px = _position_epoch_stats(positions_px, post_slice)
        metrics[row_idx]["pre_position_mean_x_mm"] = pre_x_mm
        metrics[row_idx]["pre_position_mean_y_mm"] = pre_y_mm
        metrics[row_idx]["post_position_mean_x_mm"] = post_x_mm
        metrics[row_idx]["post_position_mean_y_mm"] = post_y_mm
        metrics[row_idx]["pre_position_mean_x_px"] = pre_x_px
        metrics[row_idx]["pre_position_mean_y_px"] = pre_y_px
        metrics[row_idx]["post_position_mean_x_px"] = post_x_px
        metrics[row_idx]["post_position_mean_y_px"] = post_y_px
        metrics[row_idx]["interbout_epoch_displacement_mm"] = _distance_2d(
            pre_x_mm,
            pre_y_mm,
            post_x_mm,
            post_y_mm,
        )
        metrics[row_idx]["interbout_epoch_displacement_px"] = _distance_2d(
            pre_x_px,
            pre_y_px,
            post_x_px,
            post_y_px,
        )
        pre_position_valid = pre_valid_mm or pre_valid_px
        post_position_valid = post_valid_mm or post_valid_px
        metrics[row_idx]["pre_position_valid"] = pre_position_valid
        metrics[row_idx]["post_position_valid"] = post_position_valid
        metrics[row_idx]["pre_position_sample_count"] = max(pre_count_mm, pre_count_px)
        metrics[row_idx]["post_position_sample_count"] = max(post_count_mm, post_count_px)
        has_position_source = positions_mm is not None or positions_px is not None
        if not pre_position_valid:
            reasons.append("missing_position_source" if not has_position_source else "insufficient_pre_position")
        if not post_position_valid:
            reasons.append("missing_position_source" if not has_position_source else "insufficient_post_position")

        within_start_frame = core_start if within_window == "core_start_end" and core_start >= 0 else start_frame
        within_end_frame = core_end if within_window == "core_start_end" and core_end >= 0 else end_frame
        within_start_idx = frame_to_index.get(int(within_start_frame))
        within_end_idx = frame_to_index.get(int(within_end_frame))
        if within_start_idx is None or within_end_idx is None or within_end_idx < within_start_idx:
            reasons.append("source_bout_missing")
            within = np.asarray([], dtype=np.float64)
            within_times = np.asarray([], dtype=np.float64)
            within_transition_valid = None
            within_sample_valid = None
        else:
            within = np.asarray(headings[within_start_idx : within_end_idx + 1], dtype=np.float64)
            within_times = np.asarray(times[within_start_idx : within_end_idx + 1], dtype=np.float64)
            within_transition_valid = (
                np.asarray(transition_valid[within_start_idx : within_end_idx + 1], dtype=bool)
                if transition_valid is not None
                else None
            )
            within_sample_valid = (
                np.asarray(sample_valid[within_start_idx : within_end_idx + 1], dtype=bool)
                if sample_valid is not None
                else None
            )

        within_valid_count = int(np.count_nonzero(np.isfinite(within)))
        metrics[row_idx]["within_window_sample_count"] = within_valid_count
        within_valid = within.size >= 2 and within_valid_count == within.size
        metrics[row_idx]["within_window_valid"] = within_valid
        if not within_valid:
            reasons.append(
                "insufficient_within_bout_samples" if within_valid_count < 2 else "heading_contains_gap"
            )
        if within_valid:
            unwrapped = _unwrap_degrees(within)
            diffs = np.diff(unwrapped)
            heading_range = float(np.max(unwrapped) - np.min(unwrapped))
            metrics[row_idx]["within_heading_range_deg"] = heading_range
            metrics[row_idx]["within_heading_peak_to_peak_deg"] = heading_range
            metrics[row_idx]["within_heading_path_deg"] = float(np.sum(np.abs(diffs)))
            metrics[row_idx]["within_heading_std_deg"] = _std_unwrapped_deg(within)
            metrics[row_idx]["within_heading_zero_crossings"] = _zero_crossings(
                within,
                within_times,
                derivative_threshold_deg_s=derivative_threshold_deg_s,
            )
            frequency, frequency_valid, frequency_reason = _dominant_frequency_hz(
                within,
                within_times,
                enabled=dominant_frequency_enabled,
                min_samples=dominant_frequency_min_samples,
                detrend=dominant_frequency_detrend,
            )
            metrics[row_idx]["within_heading_dominant_frequency_hz"] = frequency
            metrics[row_idx]["dominant_frequency_valid"] = frequency_valid
            if frequency_reason is not None:
                reasons.append(frequency_reason)
            angular_velocity, angular_valid, angular_count, angular_reason = _angular_velocity_steps(
                within,
                within_times,
                transition_valid=within_transition_valid,
                sample_valid=within_sample_valid,
            )
            metrics[row_idx]["within_angular_velocity_transition_count"] = angular_count
            metrics[row_idx]["within_angular_velocity_valid"] = angular_valid
            if angular_valid:
                angular_speed = np.abs(angular_velocity)
                metrics[row_idx]["within_angular_velocity_mean_deg_s"] = float(np.mean(angular_velocity))
                metrics[row_idx]["within_angular_speed_mean_deg_s"] = float(np.mean(angular_speed))
                metrics[row_idx]["within_angular_speed_max_deg_s"] = float(np.max(angular_speed))
                metrics[row_idx]["within_angular_velocity_std_deg_s"] = float(np.std(angular_velocity))
            elif angular_reason is not None:
                reasons.append(angular_reason)

        unique_reasons = list(dict.fromkeys(reasons))
        metrics[row_idx]["failure_reason_bytes"] = (
            ";".join(unique_reasons).encode("utf-8") if unique_reasons else b"ok"
        )

    return metrics


def _safe_metric_values(metrics: np.ndarray, field: str) -> np.ndarray:
    if field not in (metrics.dtype.names or ()):
        return np.asarray([], dtype=np.float64)
    values = np.asarray(metrics[field], dtype=np.float64)
    return values[np.isfinite(values)]


def _plot_bout_kinematics_summary(
    *,
    metrics_by_level: Mapping[str, np.ndarray],
    default_heading_level: str,
    source_speed_level: str,
    bins: int,
) -> bytes:
    default_metrics = metrics_by_level.get(default_heading_level)
    if default_metrics is None:
        default_metrics = next(iter(metrics_by_level.values()))

    fig, axes = plt.subplots(2, 3, figsize=(17, 8))
    axes_flat = axes.ravel()
    metric_specs = [
        ("net_delta_heading_deg", "Net heading change (deg)", (-180.0, 180.0)),
        ("abs_net_delta_heading_deg", "Absolute net heading change (deg)", (0.0, 180.0)),
        ("within_heading_range_deg", "Within-bout heading range (deg)", None),
        ("within_heading_path_deg", "Within-bout heading path (deg)", None),
        ("within_angular_speed_mean_deg_s", "Mean angular speed (deg/s)", None),
        ("within_angular_speed_max_deg_s", "Peak angular speed (deg/s)", None),
    ]
    for ax, (field, label, xlim) in zip(axes_flat, metric_specs):
        for level, metrics in metrics_by_level.items():
            values = _safe_metric_values(metrics, field)
            if values.size == 0:
                continue
            ax.hist(
                values,
                bins=int(bins),
                alpha=0.55,
                label=level.replace("heading_", ""),
            )
        ax.set_title(label)
        ax.set_xlabel(label)
        ax.set_ylabel("Bout count")
        ax.grid(alpha=0.25)
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ax.has_data():
            ax.legend()

    n_bouts = int(len(default_metrics))
    fig.suptitle(
        f"Bout heading kinematics ({source_speed_level}, {n_bouts} bouts)",
        fontsize=14,
    )
    fig.tight_layout()
    return _png_bytes_from_figure(fig, dpi=150)


def _plot_bout_eye_gaze_summary(
    *,
    metrics: np.ndarray,
    source_speed_level: str,
    bins: int,
) -> bytes:
    fig, axes = plt.subplots(2, 3, figsize=(17, 8))
    axes_flat = axes.ravel()
    metric_specs = [
        ("pre_vergence_gaze_mean_deg", "Pre-bout vergence (deg)", None),
        ("post_vergence_gaze_mean_deg", "Post-bout vergence (deg)", None),
        ("within_bout_vergence_gaze_mean_deg", "Within-bout mean vergence (deg)", None),
        ("within_bout_vergence_gaze_max_deg", "Within-bout max vergence (deg)", None),
        ("within_bout_vergence_gaze_range_deg", "Within-bout vergence range (deg)", None),
        ("within_bout_converged_fraction", "Within-bout converged fraction", (0.0, 1.0)),
    ]
    for ax, (field, label, xlim) in zip(axes_flat, metric_specs):
        values = _safe_metric_values(metrics, field)
        if values.size:
            ax.hist(values, bins=int(bins), alpha=0.72, color="#2ca02c")
        else:
            ax.text(0.5, 0.5, "No values", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(label)
        ax.set_xlabel(label)
        ax.set_ylabel("Bout count")
        ax.grid(alpha=0.25)
        if xlim is not None:
            ax.set_xlim(*xlim)

    n_bouts = int(len(metrics))
    fig.suptitle(
        f"Bout eye-gaze summaries ({source_speed_level}, {n_bouts} bouts)",
        fontsize=14,
    )
    fig.tight_layout()
    return _png_bytes_from_figure(fig, dpi=150)


def _plot_bout_movement_summary(
    *,
    metrics: np.ndarray,
    source_speed_level: str,
    bins: int,
) -> bytes:
    fig, axes = plt.subplots(2, 3, figsize=(17, 8))
    axes_flat = axes.ravel()
    metric_specs = [
        ("detector_duration_s", "Detector duration (s)", None),
        ("physical_active_duration_s", "Physical active duration (s)", None),
        ("physical_active_duration_s_interpolated", "Physical active duration interpolated (s)", None),
        ("physical_active_path_length_mm", "Physical active path length (mm)", None),
        ("physical_active_mean_speed_mm_s", "Physical active mean speed (mm/s)", None),
        ("physical_active_peak_speed_mm_s", "Physical active peak speed (mm/s)", None),
    ]
    for ax, (field, label, xlim) in zip(axes_flat, metric_specs):
        values = _safe_metric_values(metrics, field)
        if values.size:
            ax.hist(values, bins=int(bins), alpha=0.72, color="#1f77b4")
        else:
            ax.text(0.5, 0.5, "No values", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(label)
        ax.set_xlabel(label)
        ax.set_ylabel("Bout count")
        ax.grid(alpha=0.25)
        if xlim is not None:
            ax.set_xlim(*xlim)

    n_bouts = int(len(metrics))
    fig.suptitle(
        f"Bout physical movement summaries ({source_speed_level}, {n_bouts} bouts)",
        fontsize=14,
    )
    fig.tight_layout()
    return _png_bytes_from_figure(fig, dpi=150)


_BOUT_HEADING_ARTIFACT_FIELDS = (
    "bout_id",
    "source_start_frame",
    "source_end_frame",
    "source_core_start_frame",
    "source_core_end_frame",
    "source_core_start_time_s_interpolated",
    "source_core_end_time_s_interpolated",
    "source_core_duration_s_interpolated",
    "source_core_start_time_interpolated_valid",
    "source_core_end_time_interpolated_valid",
    "source_peak_frame",
    "source_peak_time_s",
    "source_peak_signal_value_mm_s",
    "source_peak_prominence_mm_s",
    "source_peak_width_s",
    "source_peak_width_height_mm_s",
    "source_peak_left_width_frame_interpolated",
    "source_peak_right_width_frame_interpolated",
    "source_peak_left_width_time_s",
    "source_peak_right_width_time_s",
    "source_peak_boundary_mode_bytes",
    "source_peak_shape_split_policy_bytes",
    "pre_epoch_start_frame",
    "pre_epoch_end_frame",
    "post_epoch_start_frame",
    "post_epoch_end_frame",
    "net_delta_heading_deg",
    "abs_net_delta_heading_deg",
    "pre_position_mean_x_mm",
    "pre_position_mean_y_mm",
    "post_position_mean_x_mm",
    "post_position_mean_y_mm",
    "interbout_epoch_displacement_mm",
    "pre_position_mean_x_px",
    "pre_position_mean_y_px",
    "post_position_mean_x_px",
    "post_position_mean_y_px",
    "interbout_epoch_displacement_px",
    "within_heading_range_deg",
    "within_heading_peak_to_peak_deg",
    "within_heading_path_deg",
    "within_heading_std_deg",
    "within_heading_zero_crossings",
    "within_angular_velocity_mean_deg_s",
    "within_angular_speed_mean_deg_s",
    "within_angular_speed_max_deg_s",
    "within_angular_velocity_std_deg_s",
    "within_angular_velocity_valid",
    "within_angular_velocity_transition_count",
)

_BOUT_EYE_GAZE_ARTIFACT_FIELDS = (
    "bout_id",
    "source_start_frame",
    "source_end_frame",
    "source_core_start_frame",
    "source_core_end_frame",
    "pre_epoch_start_frame",
    "pre_epoch_end_frame",
    "post_epoch_start_frame",
    "post_epoch_end_frame",
    "within_epoch_start_frame",
    "within_epoch_end_frame",
    "pre_left_gaze_mean_deg",
    "pre_right_gaze_mean_deg",
    "pre_vergence_gaze_mean_deg",
    "pre_vergence_gaze_signed_mean_deg",
    "pre_vergence_gaze_std_deg",
    "pre_vergence_gaze_valid_fraction",
    "pre_converged_fraction",
    "post_left_gaze_mean_deg",
    "post_right_gaze_mean_deg",
    "post_vergence_gaze_mean_deg",
    "post_vergence_gaze_signed_mean_deg",
    "post_vergence_gaze_std_deg",
    "post_vergence_gaze_valid_fraction",
    "post_converged_fraction",
    "within_bout_left_gaze_mean_deg",
    "within_bout_right_gaze_mean_deg",
    "within_bout_vergence_gaze_mean_deg",
    "within_bout_vergence_gaze_signed_mean_deg",
    "within_bout_vergence_gaze_max_deg",
    "within_bout_vergence_gaze_range_deg",
    "within_bout_vergence_gaze_std_deg",
    "within_bout_vergence_gaze_valid_fraction",
    "within_bout_converged_fraction",
    "pre_eye_window_valid",
    "post_eye_window_valid",
    "within_eye_window_valid",
)

_BOUT_MOVEMENT_ARTIFACT_FIELDS = (
    "bout_id",
    "source_start_frame",
    "source_end_frame",
    "source_core_start_frame",
    "source_core_end_frame",
    "detector_duration_s",
    "detector_observed_duration_s",
    "detector_core_duration_s",
    "physical_active_start_frame",
    "physical_active_end_frame",
    "physical_active_start_time_s",
    "physical_active_end_time_s",
    "physical_active_duration_s",
    "physical_active_observed_duration_s",
    "physical_active_start_time_s_interpolated",
    "physical_active_end_time_s_interpolated",
    "physical_active_duration_s_interpolated",
    "physical_active_start_time_interpolated_valid",
    "physical_active_end_time_interpolated_valid",
    "physical_active_valid_transition_fraction",
    "physical_active_path_length_mm",
    "physical_active_path_length_px",
    "physical_active_mean_speed_mm_s",
    "physical_active_peak_speed_mm_s",
    "physical_active_valid",
)


def _source_table_path_for_level(*, run_name: str, layout: str, level: str) -> str:
    run_path = f"analysis/bout_kinematics_runs/{run_name}"
    if layout == LAYOUT_COMPACT_TABULAR_V2:
        if level == MOVEMENT_LEVEL:
            return f"{run_path}/{COMPACT_MOVEMENT_TABLE}"
        if level == EYE_GAZE_LEVEL:
            return f"{run_path}/{COMPACT_EYE_GAZE_TABLE}"
        return f"{run_path}/{COMPACT_HEADING_TABLE}"
    return f"{run_path}/{level}/per_bout_metrics"


def _source_filter_for_level(*, layout: str, level: str) -> dict[str, str]:
    if layout != LAYOUT_COMPACT_TABULAR_V2:
        return {}
    if level == MOVEMENT_LEVEL:
        return {"table": COMPACT_MOVEMENT_TABLE, "analysis_level_bytes": MOVEMENT_LEVEL}
    if level == EYE_GAZE_LEVEL:
        return {"table": COMPACT_EYE_GAZE_TABLE, "analysis_level_bytes": EYE_GAZE_LEVEL}
    return {"table": COMPACT_HEADING_TABLE, "heading_level_bytes": level}


def _build_bout_kinematics_source_metadata(
    *,
    run_name: str,
    layout: str,
    levels_to_fields: Mapping[str, Sequence[str]],
) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    run_path = f"analysis/bout_kinematics_runs/{run_name}"
    source_paths: dict[str, str] = {"run": run_path}
    source_filters: dict[str, dict[str, str]] = {}

    for level, fields in levels_to_fields.items():
        table_path = _source_table_path_for_level(run_name=run_name, layout=layout, level=level)
        source_paths[f"{level}.per_bout_metrics"] = table_path
        if layout == LAYOUT_COMPACT_TABULAR_V2:
            table_name = table_path.rsplit("/", 1)[-1]
            source_paths.setdefault(table_name, table_path)
        for field in fields:
            source_paths[f"{level}.{field}"] = f"{table_path}/{field}"
        source_filter = _source_filter_for_level(layout=layout, level=level)
        if source_filter:
            source_filters[level] = source_filter

    return source_paths, source_filters


def _build_bout_kinematics_interactive_spec(
    *,
    run_name: str,
    source_refs: Mapping[str, Any],
    parameters: Mapping[str, Any],
    heading_levels: Sequence[str],
    default_heading_level: str,
    layout: str,
    bins: int,
) -> dict[str, Any]:
    source_paths, source_filters = _build_bout_kinematics_source_metadata(
        run_name=run_name,
        layout=layout,
        levels_to_fields={str(level): _BOUT_HEADING_ARTIFACT_FIELDS for level in heading_levels},
    )

    return {
        "schema_id": BOUT_KINEMATICS_PLOT_SPEC_SCHEMA_ID,
        "title": "Bout heading kinematics",
        "run_name": run_name,
        "renderer": BOUT_KINEMATICS_PLOT_RENDERER,
        "layout": layout,
        "source_refs": dict(source_refs),
        "source_paths": source_paths,
        "source_filters": source_filters,
        "parameters": dict(parameters),
        "default_heading_level": default_heading_level,
        "heading_levels": list(heading_levels),
        "panels": [
            {
                "id": "net_heading_change_histograms",
                "kind": "facet_histogram",
                "heading_levels": list(heading_levels),
                "metrics": [
                    "net_delta_heading_deg",
                    "abs_net_delta_heading_deg",
                ],
                "x_ranges_deg": {
                    "net_delta_heading_deg": [-180.0, 180.0],
                    "abs_net_delta_heading_deg": [0.0, 180.0],
                },
                "bins": int(bins),
            },
            {
                "id": "within_bout_heading_histograms",
                "kind": "facet_histogram",
                "heading_levels": list(heading_levels),
                "metrics": [
                    "within_heading_range_deg",
                    "within_heading_peak_to_peak_deg",
                    "within_heading_path_deg",
                    "within_heading_std_deg",
                ],
                "x_axis_policy": "independent_positive_degrees",
                "bins": int(bins),
            },
            {
                "id": "per_bout_heading_change",
                "kind": "scatter",
                "x": "bout_id",
                "y": "net_delta_heading_deg",
                "heading_levels": list(heading_levels),
            },
            {
                "id": "within_bout_angular_velocity_histograms",
                "kind": "facet_histogram",
                "heading_levels": list(heading_levels),
                "metrics": [
                    "within_angular_velocity_mean_deg_s",
                    "within_angular_speed_mean_deg_s",
                    "within_angular_speed_max_deg_s",
                    "within_angular_velocity_std_deg_s",
                ],
                "x_axis_policy": "independent_degrees_per_second",
                "bins": int(bins),
            },
        ],
    }


def _build_bout_eye_gaze_interactive_spec(
    *,
    run_name: str,
    source_refs: Mapping[str, Any],
    parameters: Mapping[str, Any],
    layout: str,
    bins: int,
) -> dict[str, Any]:
    source_paths, source_filters = _build_bout_kinematics_source_metadata(
        run_name=run_name,
        layout=layout,
        levels_to_fields={EYE_GAZE_LEVEL: _BOUT_EYE_GAZE_ARTIFACT_FIELDS},
    )

    return {
        "schema_id": BOUT_EYE_GAZE_PLOT_SPEC_SCHEMA_ID,
        "title": "Bout eye-gaze summaries",
        "run_name": run_name,
        "renderer": BOUT_KINEMATICS_PLOT_RENDERER,
        "layout": layout,
        "source_refs": dict(source_refs),
        "source_paths": source_paths,
        "source_filters": source_filters,
        "parameters": dict(parameters),
        "analysis_level": EYE_GAZE_LEVEL,
        "panels": [
            {
                "id": "bout_eye_gaze_histograms",
                "kind": "facet_histogram",
                "metrics": [
                    "pre_vergence_gaze_mean_deg",
                    "post_vergence_gaze_mean_deg",
                    "within_bout_vergence_gaze_mean_deg",
                    "within_bout_vergence_gaze_max_deg",
                    "within_bout_vergence_gaze_range_deg",
                    "within_bout_converged_fraction",
                ],
                "bins": int(bins),
            },
            {
                "id": "bout_eye_gaze_validity",
                "kind": "valid_fraction_summary",
                "metrics": [
                    "pre_vergence_gaze_valid_fraction",
                    "post_vergence_gaze_valid_fraction",
                    "within_bout_vergence_gaze_valid_fraction",
                ],
            },
        ],
    }


def _build_bout_movement_interactive_spec(
    *,
    run_name: str,
    source_refs: Mapping[str, Any],
    parameters: Mapping[str, Any],
    layout: str,
    bins: int,
) -> dict[str, Any]:
    source_paths, source_filters = _build_bout_kinematics_source_metadata(
        run_name=run_name,
        layout=layout,
        levels_to_fields={MOVEMENT_LEVEL: _BOUT_MOVEMENT_ARTIFACT_FIELDS},
    )

    return {
        "schema_id": BOUT_MOVEMENT_PLOT_SPEC_SCHEMA_ID,
        "title": "Bout physical movement summaries",
        "run_name": run_name,
        "renderer": BOUT_KINEMATICS_PLOT_RENDERER,
        "layout": layout,
        "source_refs": dict(source_refs),
        "source_paths": source_paths,
        "source_filters": source_filters,
        "parameters": dict(parameters),
        "analysis_level": MOVEMENT_LEVEL,
        "panels": [
            {
                "id": "bout_physical_movement_histograms",
                "kind": "facet_histogram",
                "metrics": [
                    "detector_duration_s",
                    "physical_active_duration_s",
                    "physical_active_duration_s_interpolated",
                    "physical_active_path_length_mm",
                    "physical_active_mean_speed_mm_s",
                    "physical_active_peak_speed_mm_s",
                ],
                "bins": int(bins),
            },
            {
                "id": "physical_active_validity",
                "kind": "validity_summary",
                "metrics": [
                    "physical_active_valid",
                    "physical_active_valid_transition_fraction",
                ],
            },
        ],
    }


def write_bout_movement_visualization_artifacts(
    *,
    zarr_path: Path,
    run_group: zarr.Group,
    run_name: str,
    movement_metrics: np.ndarray,
    source_refs: Mapping[str, Any],
    parameters: Mapping[str, Any],
    source_speed_level: str,
    layout: str,
    bins: int,
    artifact_dpi: int,
    command: Optional[str],
) -> None:
    png_artifact_name = f"{BOUT_MOVEMENT_PNG_PREFIX}_track_{int(source_refs['source_track_id'])}_png"
    spec_artifact_name = f"{BOUT_MOVEMENT_PNG_PREFIX}_track_{int(source_refs['source_track_id'])}_interactive"
    source_paths, source_filters = _build_bout_kinematics_source_metadata(
        run_name=run_name,
        layout=layout,
        levels_to_fields={MOVEMENT_LEVEL: _BOUT_MOVEMENT_ARTIFACT_FIELDS},
    )
    source_runs = {
        "bout_kinematics": run_name,
        "track_kinematics": source_refs.get("source_track_kinematics_run"),
        "swim_bout": source_refs.get("source_swim_bout_run"),
        "swim_bout_speed_level": source_refs.get("source_swim_bout_speed_level"),
    }
    plot_parameters = {
        "bins": int(bins),
        "artifact_dpi": int(artifact_dpi),
        "layout": layout,
        "analysis_level": MOVEMENT_LEVEL,
        "physical_active": parameters.get("physical_active", {}),
    }
    signature = _artifact_signature(
        {
            "schema_id": BOUT_MOVEMENT_PLOT_SPEC_SCHEMA_ID,
            "run_name": run_name,
            "source_refs": source_refs,
            "parameters": plot_parameters,
        }
    )
    created_at_utc = datetime.now(timezone.utc).isoformat()
    env_info = get_environment_info(disk_path=str(zarr_path), capture_env_vars=False)
    provenance = build_stage_provenance(
        stage="bout_movement_visualization",
        created_at_utc=created_at_utc,
        parameters={
            **plot_parameters,
            "plot_schema_id": BOUT_MOVEMENT_PLOT_SPEC_SCHEMA_ID,
            "renderer": BOUT_KINEMATICS_PLOT_RENDERER,
        },
        inputs={
            "zarr_path": str(zarr_path),
            "source_refs": dict(source_refs),
            "source_paths": source_paths,
            "source_filters": source_filters,
            "source_runs": source_runs,
        },
        command=command,
        version=BOUT_KINEMATICS_PLOT_RENDERER,
        git=get_git_info(),
        environment=env_info.get("environment"),
        platform=env_info.get("platform"),
        artifacts={
            "png_artifact": f"visualizations/{png_artifact_name}",
            "interactive_artifact": f"visualizations/{spec_artifact_name}",
            "artifact_signature": signature,
        },
    )
    png_bytes = _plot_bout_movement_summary(
        metrics=movement_metrics,
        source_speed_level=source_speed_level,
        bins=bins,
    )
    write_png_visualization_artifact(
        run_group,
        png_artifact_name,
        png_bytes,
        description="Bout physical movement summary PNG",
        created_by="fisheye.analysis.bout_kinematics",
        artifact_signature=signature,
        created_at_utc=created_at_utc,
        source_paths=source_paths,
        source_runs=source_runs,
        parameters=plot_parameters,
        extra_attrs={
            "plot_schema_id": BOUT_MOVEMENT_PLOT_SPEC_SCHEMA_ID,
            "source_filters": source_filters,
            "provenance": provenance,
        },
    )
    spec = _build_bout_movement_interactive_spec(
        run_name=run_name,
        source_refs=source_refs,
        parameters=parameters,
        layout=layout,
        bins=bins,
    )
    write_interactive_plot_spec_artifact(
        run_group,
        spec_artifact_name,
        spec,
        description="Bout physical movement interactive plot spec",
        created_by="fisheye.analysis.bout_kinematics",
        renderer=BOUT_KINEMATICS_PLOT_RENDERER,
        artifact_signature=signature,
        created_at_utc=created_at_utc,
        snapshot_artifact=png_artifact_name,
        source_paths=source_paths,
        source_runs=source_runs,
        parameters=plot_parameters,
        extra_attrs={
            "plot_schema_id": BOUT_MOVEMENT_PLOT_SPEC_SCHEMA_ID,
            "source_filters": source_filters,
            "provenance": provenance,
        },
    )


def write_bout_eye_gaze_visualization_artifacts(
    *,
    zarr_path: Path,
    run_group: zarr.Group,
    run_name: str,
    eye_gaze_metrics: np.ndarray,
    source_refs: Mapping[str, Any],
    parameters: Mapping[str, Any],
    source_speed_level: str,
    layout: str,
    bins: int,
    artifact_dpi: int,
    command: Optional[str],
) -> None:
    png_artifact_name = f"{BOUT_EYE_GAZE_PNG_PREFIX}_track_{int(source_refs['source_track_id'])}_png"
    spec_artifact_name = f"{BOUT_EYE_GAZE_PNG_PREFIX}_track_{int(source_refs['source_track_id'])}_interactive"
    source_paths, source_filters = _build_bout_kinematics_source_metadata(
        run_name=run_name,
        layout=layout,
        levels_to_fields={EYE_GAZE_LEVEL: _BOUT_EYE_GAZE_ARTIFACT_FIELDS},
    )
    source_runs = {
        "bout_kinematics": run_name,
        "track_kinematics": source_refs.get("source_track_kinematics_run"),
        "swim_bout": source_refs.get("source_swim_bout_run"),
        "swim_bout_speed_level": source_refs.get("source_swim_bout_speed_level"),
        "eye_angle": source_refs.get("source_eye_angle_run"),
    }
    plot_parameters = {
        "bins": int(bins),
        "artifact_dpi": int(artifact_dpi),
        "layout": layout,
        "analysis_level": EYE_GAZE_LEVEL,
        "pre_post_mode": parameters.get("pre_post_mode"),
        "eye_gaze": parameters.get("eye_gaze", {}),
    }
    signature = _artifact_signature(
        {
            "schema_id": BOUT_EYE_GAZE_PLOT_SPEC_SCHEMA_ID,
            "run_name": run_name,
            "source_refs": source_refs,
            "parameters": plot_parameters,
        }
    )
    created_at_utc = datetime.now(timezone.utc).isoformat()
    env_info = get_environment_info(disk_path=str(zarr_path), capture_env_vars=False)
    provenance = build_stage_provenance(
        stage="bout_eye_gaze_visualization",
        created_at_utc=created_at_utc,
        parameters={
            **plot_parameters,
            "plot_schema_id": BOUT_EYE_GAZE_PLOT_SPEC_SCHEMA_ID,
            "renderer": BOUT_KINEMATICS_PLOT_RENDERER,
        },
        inputs={
            "zarr_path": str(zarr_path),
            "source_refs": dict(source_refs),
            "source_paths": source_paths,
            "source_filters": source_filters,
            "source_runs": source_runs,
        },
        command=command,
        version=BOUT_KINEMATICS_PLOT_RENDERER,
        git=get_git_info(),
        environment=env_info.get("environment"),
        platform=env_info.get("platform"),
        artifacts={
            "png_artifact": f"visualizations/{png_artifact_name}",
            "interactive_artifact": f"visualizations/{spec_artifact_name}",
            "artifact_signature": signature,
        },
    )
    png_bytes = _plot_bout_eye_gaze_summary(
        metrics=eye_gaze_metrics,
        source_speed_level=source_speed_level,
        bins=bins,
    )
    write_png_visualization_artifact(
        run_group,
        png_artifact_name,
        png_bytes,
        description="Bout eye-gaze summary PNG",
        created_by="fisheye.analysis.bout_kinematics",
        artifact_signature=signature,
        created_at_utc=created_at_utc,
        source_paths=source_paths,
        source_runs=source_runs,
        parameters=plot_parameters,
        extra_attrs={
            "plot_schema_id": BOUT_EYE_GAZE_PLOT_SPEC_SCHEMA_ID,
            "source_filters": source_filters,
            "provenance": provenance,
        },
    )
    spec = _build_bout_eye_gaze_interactive_spec(
        run_name=run_name,
        source_refs=source_refs,
        parameters=parameters,
        layout=layout,
        bins=bins,
    )
    write_interactive_plot_spec_artifact(
        run_group,
        spec_artifact_name,
        spec,
        description="Bout eye-gaze interactive plot spec",
        created_by="fisheye.analysis.bout_kinematics",
        renderer=BOUT_KINEMATICS_PLOT_RENDERER,
        artifact_signature=signature,
        created_at_utc=created_at_utc,
        snapshot_artifact=png_artifact_name,
        source_paths=source_paths,
        source_runs=source_runs,
        parameters=plot_parameters,
        extra_attrs={
            "plot_schema_id": BOUT_EYE_GAZE_PLOT_SPEC_SCHEMA_ID,
            "source_filters": source_filters,
            "provenance": provenance,
        },
    )


def write_bout_kinematics_visualization_artifacts(
    *,
    zarr_path: Path,
    run_group: zarr.Group,
    run_name: str,
    metrics_by_level: Mapping[str, np.ndarray],
    movement_metrics: np.ndarray,
    eye_gaze_metrics: Optional[np.ndarray],
    source_refs: Mapping[str, Any],
    parameters: Mapping[str, Any],
    heading_levels: Sequence[str],
    default_heading_level: str,
    source_speed_level: str,
    layout: str,
    bins: int,
    artifact_dpi: int,
    command: Optional[str],
) -> None:
    png_artifact_name = f"{BOUT_KINEMATICS_PNG_PREFIX}_track_{int(source_refs['source_track_id'])}_png"
    spec_artifact_name = f"{BOUT_KINEMATICS_PNG_PREFIX}_track_{int(source_refs['source_track_id'])}_interactive"
    write_bout_movement_visualization_artifacts(
        zarr_path=zarr_path,
        run_group=run_group,
        run_name=run_name,
        movement_metrics=movement_metrics,
        source_refs=source_refs,
        parameters=parameters,
        source_speed_level=source_speed_level,
        layout=layout,
        bins=int(bins),
        artifact_dpi=int(artifact_dpi),
        command=command,
    )
    source_paths, source_filters = _build_bout_kinematics_source_metadata(
        run_name=run_name,
        layout=layout,
        levels_to_fields={str(level): _BOUT_HEADING_ARTIFACT_FIELDS for level in heading_levels},
    )
    source_runs = {
        "bout_kinematics": run_name,
        "track_kinematics": source_refs.get("source_track_kinematics_run"),
        "swim_bout": source_refs.get("source_swim_bout_run"),
        "swim_bout_speed_level": source_refs.get("source_swim_bout_speed_level"),
    }
    plot_parameters = {
        "bins": int(bins),
        "artifact_dpi": int(artifact_dpi),
        "layout": layout,
        "heading_levels": list(heading_levels),
        "default_heading_level": default_heading_level,
        "pre_post_mode": parameters.get("pre_post_mode"),
    }
    signature = _artifact_signature(
        {
            "schema_id": BOUT_KINEMATICS_PLOT_SPEC_SCHEMA_ID,
            "run_name": run_name,
            "source_refs": source_refs,
            "parameters": plot_parameters,
        }
    )
    created_at_utc = datetime.now(timezone.utc).isoformat()
    env_info = get_environment_info(disk_path=str(zarr_path), capture_env_vars=False)
    provenance = build_stage_provenance(
        stage="bout_kinematics_visualization",
        created_at_utc=created_at_utc,
        parameters={
            **plot_parameters,
            "plot_schema_id": BOUT_KINEMATICS_PLOT_SPEC_SCHEMA_ID,
            "renderer": BOUT_KINEMATICS_PLOT_RENDERER,
        },
        inputs={
            "zarr_path": str(zarr_path),
            "source_refs": dict(source_refs),
            "source_paths": source_paths,
            "source_filters": source_filters,
            "source_runs": source_runs,
        },
        command=command,
        version=BOUT_KINEMATICS_PLOT_RENDERER,
        git=get_git_info(),
        environment=env_info.get("environment"),
        platform=env_info.get("platform"),
        artifacts={
            "png_artifact": f"visualizations/{png_artifact_name}",
            "interactive_artifact": f"visualizations/{spec_artifact_name}",
            "artifact_signature": signature,
        },
    )
    png_bytes = _plot_bout_kinematics_summary(
        metrics_by_level=metrics_by_level,
        default_heading_level=default_heading_level,
        source_speed_level=source_speed_level,
        bins=bins,
    )
    write_png_visualization_artifact(
        run_group,
        png_artifact_name,
        png_bytes,
        description="Bout heading kinematics summary PNG",
        created_by="fisheye.analysis.bout_kinematics",
        artifact_signature=signature,
        created_at_utc=created_at_utc,
        source_paths=source_paths,
        source_runs=source_runs,
        parameters=plot_parameters,
        extra_attrs={
            "plot_schema_id": BOUT_KINEMATICS_PLOT_SPEC_SCHEMA_ID,
            "source_filters": source_filters,
            "provenance": provenance,
        },
    )
    spec = _build_bout_kinematics_interactive_spec(
        run_name=run_name,
        source_refs=source_refs,
        parameters=parameters,
        heading_levels=heading_levels,
        default_heading_level=default_heading_level,
        layout=layout,
        bins=bins,
    )
    write_interactive_plot_spec_artifact(
        run_group,
        spec_artifact_name,
        spec,
        description="Bout heading kinematics interactive plot spec",
        created_by="fisheye.analysis.bout_kinematics",
        renderer=BOUT_KINEMATICS_PLOT_RENDERER,
        artifact_signature=signature,
        created_at_utc=created_at_utc,
        snapshot_artifact=png_artifact_name,
        source_paths=source_paths,
        source_runs=source_runs,
        parameters=plot_parameters,
        extra_attrs={
            "plot_schema_id": BOUT_KINEMATICS_PLOT_SPEC_SCHEMA_ID,
            "source_filters": source_filters,
            "provenance": provenance,
        },
    )
    if eye_gaze_metrics is not None:
        write_bout_eye_gaze_visualization_artifacts(
            zarr_path=zarr_path,
            run_group=run_group,
            run_name=run_name,
            eye_gaze_metrics=eye_gaze_metrics,
            source_refs=source_refs,
            parameters=parameters,
            source_speed_level=source_speed_level,
            layout=layout,
            bins=int(bins),
            artifact_dpi=int(artifact_dpi),
            command=command,
        )


def compute_and_save_bout_kinematics(
    zarr_path: Path | str,
    *,
    run_name: Optional[str] = None,
    track_kinematics_run: str = "latest",
    track_scope: str = "offline",
    track_id: int = 0,
    swim_bout_run: str = "latest",
    speed_level: str = "filtered",
    heading_levels: Sequence[str] = ("heading_smoothed", "heading_raw"),
    default_heading_level: str = "heading_smoothed",
    pre_post_mode: str = "fixed_window",
    pre_window_s: float = 0.05,
    post_window_s: float = 0.05,
    within_window: str = "bout_start_end",
    physical_active_signal_level: str = "filtered",
    physical_active_threshold_mm_s: float = 0.01,
    physical_active_boundary_constraint: str = "search_with_margin",
    physical_active_boundary_margin_s: float = 0.05,
    zero_crossing_derivative_threshold_deg_s: float = 0.0,
    dominant_frequency: bool = False,
    dominant_frequency_min_samples: int = 8,
    dominant_frequency_detrend: bool = True,
    include_eye_gaze: bool = False,
    eye_angle_run: str = "latest",
    eye_angle_family: str = "gaze",
    eye_validity_min_fraction: float = 1.0,
    vergence_threshold_deg: Optional[float] = None,
    write_visualizations: bool = False,
    visualization_bins: int = 40,
    visualization_dpi: int = 150,
    layout: str = BOUT_KINEMATICS_LAYOUT_DEFAULT,
    overwrite: bool = False,
    command: Optional[str] = None,
) -> str:
    """Compute and persist linked per-bout heading metrics."""

    if within_window not in WITHIN_WINDOWS:
        expected = ", ".join(WITHIN_WINDOWS)
        raise ValueError(f"Unsupported within_window {within_window!r}; expected one of: {expected}")
    if pre_post_mode not in PRE_POST_MODES:
        expected = ", ".join(PRE_POST_MODES)
        raise ValueError(f"Unsupported pre_post_mode {pre_post_mode!r}; expected one of: {expected}")
    if not 0.0 <= float(eye_validity_min_fraction) <= 1.0:
        raise ValueError("eye_validity_min_fraction must be between 0 and 1.")
    if str(eye_angle_family).strip() not in EYE_ANGLE_FAMILIES:
        expected = ", ".join(EYE_ANGLE_FAMILIES)
        raise ValueError(f"Unsupported eye_angle_family {eye_angle_family!r}; expected one of: {expected}")
    if physical_active_boundary_constraint not in PHYSICAL_ACTIVE_BOUNDARY_CONSTRAINTS:
        expected = ", ".join(PHYSICAL_ACTIVE_BOUNDARY_CONSTRAINTS)
        raise ValueError(
            f"Unsupported physical_active_boundary_constraint {physical_active_boundary_constraint!r}; "
            f"expected one of: {expected}"
        )
    if layout not in BOUT_KINEMATICS_LAYOUTS:
        expected = ", ".join(BOUT_KINEMATICS_LAYOUTS)
        raise ValueError(f"Unsupported bout-kinematics layout {layout!r}; expected one of: {expected}")
    if float(physical_active_threshold_mm_s) < 0:
        raise ValueError("physical_active_threshold_mm_s must be >= 0.")
    if float(physical_active_boundary_margin_s) < 0:
        raise ValueError("physical_active_boundary_margin_s must be >= 0.")

    default_heading_level = normalize_heading_level(default_heading_level)
    normalized_heading_levels = tuple(dict.fromkeys(normalize_heading_level(level) for level in heading_levels))
    if default_heading_level not in normalized_heading_levels:
        normalized_heading_levels = (default_heading_level, *normalized_heading_levels)
    physical_active_level = normalize_speed_level(physical_active_signal_level)
    if physical_active_level not in PHYSICAL_ACTIVE_SPEED_LEVELS:
        expected = ", ".join(level.removeprefix("speed_") for level in PHYSICAL_ACTIVE_SPEED_LEVELS)
        raise ValueError(
            f"Unsupported physical_active_signal_level {physical_active_signal_level!r}; "
            f"expected one of: {expected}"
        )
    physical_active_suffix = _speed_level_suffix(physical_active_level)

    zarr_path = Path(zarr_path)
    root = open_zarr_root(zarr_path, mode="r+")
    track_run_group, track_run_name, track_run_path, resolved_scope = _resolve_track_run(
        root,
        track_kinematics_run,
        track_scope=track_scope,
    )
    tracks = track_run_group.get("tracks")
    if tracks is None or f"id_{int(track_id)}" not in tracks:
        raise ValueError(f"Track id_{track_id} not found in {track_run_path}.")
    track_group = tracks[f"id_{int(track_id)}"]

    frames = np.asarray(track_group["frame_indices"][:], dtype=np.int64)
    if "time_seconds" in track_group:
        times = np.asarray(track_group["time_seconds"][:], dtype=np.float64)
    else:
        fps_for_time = float(track_run_group.attrs.get("fps", 0.0))
        times = frames.astype(np.float64) / fps_for_time if fps_for_time > 0 else np.arange(frames.size)
    fps = float(track_run_group.attrs.get("fps", 0.0))
    if fps <= 0:
        raise ValueError(f"Track kinematics run {track_run_path} has invalid fps={fps!r}.")
    positions_mm = None
    positions_px = None
    physical_speed_mm = None
    physical_path_distance_mm = None
    physical_path_distance_px = None
    delta_seconds = None
    source_position_arrays: dict[str, str] = {}
    source_movement_arrays: dict[str, str] = {}
    source_validity_arrays: dict[str, str] = {}
    transition_valid = None
    sample_valid = None
    if "positions_mm" in track_group:
        positions_mm = np.asarray(track_group["positions_mm"][:], dtype=np.float64)
        if positions_mm.shape != (frames.shape[0], 2):
            raise ValueError(
                f"positions_mm shape {positions_mm.shape} does not match expected {(frames.shape[0], 2)}."
            )
        source_position_arrays["positions_mm"] = f"{track_run_path}/tracks/id_{int(track_id)}/positions_mm"
    if "positions_px" in track_group:
        positions_px = np.asarray(track_group["positions_px"][:], dtype=np.float64)
        if positions_px.shape != (frames.shape[0], 2):
            raise ValueError(
                f"positions_px shape {positions_px.shape} does not match expected {(frames.shape[0], 2)}."
            )
        source_position_arrays["positions_px"] = f"{track_run_path}/tracks/id_{int(track_id)}/positions_px"
    physical_speed_array = f"speed_{physical_active_suffix}_mm"
    if physical_speed_array not in track_group:
        raise ValueError(
            f"Physical active speed source {physical_speed_array!r} not found in "
            f"{track_run_path}/tracks/id_{track_id}."
        )
    physical_speed_mm = np.asarray(track_group[physical_speed_array][:], dtype=np.float64)
    if physical_speed_mm.shape[0] != frames.shape[0]:
        raise ValueError(
            f"Physical active speed source {physical_speed_array!r} length "
            f"{physical_speed_mm.shape[0]} does not match frames length {frames.shape[0]}."
        )
    source_movement_arrays["physical_active_speed"] = (
        f"{track_run_path}/tracks/id_{int(track_id)}/{physical_speed_array}"
    )
    path_distance_mm_array = f"frame_path_distance_{physical_active_suffix}_mm"
    path_distance_px_array = f"frame_path_distance_{physical_active_suffix}_px"
    if path_distance_mm_array in track_group:
        physical_path_distance_mm = np.asarray(track_group[path_distance_mm_array][:], dtype=np.float64)
        if physical_path_distance_mm.shape[0] != frames.shape[0]:
            raise ValueError(
                f"{path_distance_mm_array!r} length {physical_path_distance_mm.shape[0]} "
                f"does not match frames length {frames.shape[0]}."
            )
        source_movement_arrays["physical_active_path_distance_mm"] = (
            f"{track_run_path}/tracks/id_{int(track_id)}/{path_distance_mm_array}"
        )
    if path_distance_px_array in track_group:
        physical_path_distance_px = np.asarray(track_group[path_distance_px_array][:], dtype=np.float64)
        if physical_path_distance_px.shape[0] != frames.shape[0]:
            raise ValueError(
                f"{path_distance_px_array!r} length {physical_path_distance_px.shape[0]} "
                f"does not match frames length {frames.shape[0]}."
            )
        source_movement_arrays["physical_active_path_distance_px"] = (
            f"{track_run_path}/tracks/id_{int(track_id)}/{path_distance_px_array}"
        )
    if "delta_seconds" in track_group:
        delta_seconds = np.asarray(track_group["delta_seconds"][:], dtype=np.float64)
        if delta_seconds.shape[0] != frames.shape[0]:
            raise ValueError(
                f"delta_seconds length {delta_seconds.shape[0]} does not match frames length {frames.shape[0]}."
            )
        source_validity_arrays["delta_seconds"] = f"{track_run_path}/tracks/id_{int(track_id)}/delta_seconds"
    if "transition_valid" in track_group:
        transition_valid = np.asarray(track_group["transition_valid"][:], dtype=bool)
        if transition_valid.shape[0] != frames.shape[0]:
            raise ValueError(
                f"transition_valid length {transition_valid.shape[0]} does not match frames length {frames.shape[0]}."
            )
        source_validity_arrays["transition_valid"] = f"{track_run_path}/tracks/id_{int(track_id)}/transition_valid"
    if "sample_valid" in track_group:
        sample_valid = np.asarray(track_group["sample_valid"][:], dtype=bool)
        if sample_valid.shape[0] != frames.shape[0]:
            raise ValueError(
                f"sample_valid length {sample_valid.shape[0]} does not match frames length {frames.shape[0]}."
            )
        source_validity_arrays["sample_valid"] = f"{track_run_path}/tracks/id_{int(track_id)}/sample_valid"

    pre_window_frames = max(1, int(round(float(pre_window_s) * fps)))
    post_window_frames = max(1, int(round(float(post_window_s) * fps)))
    physical_active_boundary_margin_frames = max(
        0,
        int(math.ceil(float(physical_active_boundary_margin_s) * fps - 1e-9)),
    )
    source_heading_arrays = {
        heading_level: f"{track_run_path}/tracks/id_{int(track_id)}/{HEADING_LEVEL_TO_ARRAY[heading_level]}"
        for heading_level in normalized_heading_levels
    }

    swim_payload = load_swim_bout_tables(
        root,
        run_name=swim_bout_run,
        speed_level=speed_level,
    )
    swim_run_name = swim_payload.run_name
    source_speed_level = swim_payload.signal.speed_level
    swim_level_path = swim_payload.level_path
    swim_run_attrs = swim_payload.run_attrs
    source_track_id = swim_run_attrs.get("track_id")
    if source_track_id is not None and int(source_track_id) != int(track_id):
        raise ValueError(
            f"Swim-bout run {swim_run_name!r} was derived from track_id={source_track_id}, "
            f"not requested track_id={track_id}."
        )

    source_track_run = swim_run_attrs.get("source_track_kinematics_run")
    if source_track_run is not None and str(source_track_run).strip("/") not in {
        track_run_name,
        f"{resolved_scope}/{track_run_name}",
        track_run_path,
    }:
        raise ValueError(
            f"Swim-bout run {swim_run_name!r} source_track_kinematics_run={source_track_run!r} "
            f"does not match selected {track_run_path!r}."
        )

    bouts = swim_payload.bouts
    peak_events: Optional[np.ndarray] = None
    loaded_peak_events = swim_payload.peak_events
    if len(loaded_peak_events):
        if len(loaded_peak_events) == len(bouts) and _records_align_by_bout_id(bouts, loaded_peak_events):
            peak_events = loaded_peak_events

    if "analysis" not in root:
        analysis = root.create_group("analysis")
    else:
        analysis = root["analysis"]
    parent = require_runs_parent(analysis, "bout_kinematics_runs")

    if run_name is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_name = f"bout_kinematics_{timestamp}"
    elif run_name in parent:
        if not overwrite:
            raise ValueError(
                f"Bout kinematics run {run_name!r} already exists. Use --overwrite or a different name."
            )
        del parent[run_name]

    run_group = parent.create_group(run_name)
    mark_run_started(run_group, run_name=run_name, stage="bout_kinematics")
    run_group.attrs["status"] = "running"
    created_at_utc = datetime.now(timezone.utc).isoformat()
    source_refs = {
        "zarr_path": str(zarr_path),
        "source_track_kinematics_run": track_run_name,
        "source_track_kinematics_scope": resolved_scope,
        "source_track_kinematics_path": track_run_path,
        "source_track_kinematics_track_path": f"{track_run_path}/tracks/id_{int(track_id)}",
        "source_swim_bout_run": swim_run_name,
        "source_swim_bout_speed_level": source_speed_level,
        "source_swim_bout_path": swim_level_path,
        "source_swim_bout_candidate_id": int(swim_payload.candidate.candidate_id),
        "source_swim_bout_signal_id": int(swim_payload.signal.signal_id),
        "source_swim_bout_signal_role": swim_payload.signal.role,
        "source_track_id": int(track_id),
        "source_heading_arrays": source_heading_arrays,
        "source_position_arrays": source_position_arrays,
        "source_movement_arrays": source_movement_arrays,
        "source_validity_arrays": source_validity_arrays,
    }
    if peak_events is not None:
        source_refs["source_peak_events_path"] = f"{swim_level_path}/peak_events"
    eye_series: Optional[dict[str, np.ndarray]] = None
    if include_eye_gaze:
        eye_series, eye_source_refs = _load_eye_gaze_frame_series(
            root,
            eye_angle_run=eye_angle_run,
            eye_angle_family=eye_angle_family,
            frames=frames,
        )
        source_refs.update(eye_source_refs)
    source_bout_field_names = list(bouts.dtype.names or [])
    source_interpolated_threshold_fields = [
        field
        for field in (
            "core_start_time_s_interpolated",
            "core_end_time_s_interpolated",
            "core_duration_s_interpolated",
            "core_start_time_interpolated_valid",
            "core_end_time_interpolated_valid",
        )
        if field in source_bout_field_names
    ]
    source_peak_event_fields = list(peak_events.dtype.names if peak_events is not None else [])
    parameters = {
        "layout": layout,
        "default_heading_level": default_heading_level,
        "heading_levels": list(normalized_heading_levels),
        "pre_post_mode": pre_post_mode,
        "pre_window_s": float(pre_window_s),
        "post_window_s": float(post_window_s),
        "resolved_pre_window_frames": int(pre_window_frames),
        "resolved_post_window_frames": int(post_window_frames),
        "within_window": within_window,
        "heading_units": "degrees",
        "heading_unwrap_policy": "numpy.unwrap_contiguous_window",
        "physical_active": {
            "enabled": True,
            "boundary_policy": PHYSICAL_ACTIVE_BOUNDARY_POLICY,
            "boundary_constraint": physical_active_boundary_constraint,
            "boundary_margin_s": float(physical_active_boundary_margin_s),
            "resolved_boundary_margin_frames": int(physical_active_boundary_margin_frames),
            "threshold_mm_s": float(physical_active_threshold_mm_s),
            "measurement_signal_level": physical_active_level,
            "measurement_signal_array": physical_speed_array,
        },
        "source_interpolated_threshold_fields": source_interpolated_threshold_fields,
        "source_peak_event_fields": source_peak_event_fields,
        "zero_crossing_derivative_threshold_deg_s": float(zero_crossing_derivative_threshold_deg_s),
        "dominant_frequency": {
            "enabled": bool(dominant_frequency),
            "min_samples": int(dominant_frequency_min_samples),
            "method": "rfft_peak",
            "detrend": bool(dominant_frequency_detrend),
        },
        "eye_gaze": {
            "enabled": bool(include_eye_gaze),
            "eye_angle_run": str(eye_angle_run),
            "eye_angle_family": str(eye_angle_family),
            "eye_validity_min_fraction": float(eye_validity_min_fraction),
            "vergence_threshold_deg": (
                None if vergence_threshold_deg is None else float(vergence_threshold_deg)
            ),
        },
    }
    run_group.attrs["schema_id"] = SCHEMA_ID
    run_group.attrs["schema_version"] = SCHEMA_VERSION
    run_group.attrs["method"] = METHOD
    run_group.attrs["method_version"] = METHOD_VERSION
    run_group.attrs["layout"] = layout
    run_group.attrs["row_axis"] = "swim_bout_rows"
    run_group.attrs["source_refs"] = source_refs
    run_group.attrs["parameters"] = parameters
    run_group.attrs["source_track_id"] = int(track_id)
    run_group.attrs["source_swim_bout_run"] = swim_run_name
    run_group.attrs["source_swim_bout_speed_level"] = source_speed_level
    run_group.attrs["source_track_kinematics_run"] = track_run_name
    run_group.attrs["default_heading_level"] = default_heading_level

    movement_metrics = _build_metrics_for_movement(
        bouts=bouts,
        frames=frames,
        times=times,
        physical_speed_mm=physical_speed_mm,
        fps=fps,
        threshold_mm_s=float(physical_active_threshold_mm_s),
        boundary_constraint=physical_active_boundary_constraint,
        boundary_margin_frames=int(physical_active_boundary_margin_frames),
        boundary_margin_s=float(physical_active_boundary_margin_s),
        delta_seconds=delta_seconds,
        transition_valid=transition_valid,
        sample_valid=sample_valid,
        path_distance_mm=physical_path_distance_mm,
        path_distance_px=physical_path_distance_px,
    )
    movement_attrs = {
        "schema_id": f"{SCHEMA_ID}.movement.per_bout_metrics",
        "schema_version": SCHEMA_VERSION,
        "analysis_level": MOVEMENT_LEVEL,
        "source_bout_count": int(len(bouts)),
        "source_bout_field_names": source_bout_field_names,
        "physical_active_boundary_policy": PHYSICAL_ACTIVE_BOUNDARY_POLICY,
        "physical_active_boundary_constraint": physical_active_boundary_constraint,
        "physical_active_boundary_margin_s": float(physical_active_boundary_margin_s),
        "physical_active_boundary_margin_frames": int(physical_active_boundary_margin_frames),
        "physical_active_threshold_mm_s": float(physical_active_threshold_mm_s),
        "physical_active_signal_level": physical_active_level,
        "physical_active_signal_array": physical_speed_array,
        "source_movement_arrays": source_movement_arrays,
        "source_validity_arrays": source_validity_arrays,
    }
    if layout == LAYOUT_HIERARCHICAL_V1:
        movement_group = run_group.create_group(MOVEMENT_LEVEL)
        movement_group.attrs["analysis_level"] = MOVEMENT_LEVEL
        movement_group.attrs["source_swim_bout_path"] = swim_level_path
        movement_group.attrs["physical_active_boundary_policy"] = PHYSICAL_ACTIVE_BOUNDARY_POLICY
        movement_group.attrs["physical_active_boundary_constraint"] = physical_active_boundary_constraint
        movement_group.attrs["physical_active_boundary_margin_s"] = float(physical_active_boundary_margin_s)
        movement_group.attrs["physical_active_boundary_margin_frames"] = int(physical_active_boundary_margin_frames)
        movement_group.attrs["physical_active_threshold_mm_s"] = float(physical_active_threshold_mm_s)
        movement_group.attrs["physical_active_signal_level"] = physical_active_level
        movement_group.attrs["physical_active_signal_array"] = physical_speed_array
        write_columnar_dataset(
            movement_group,
            "per_bout_metrics",
            movement_metrics,
            attrs=movement_attrs,
        )

    written_levels: list[str] = []
    metrics_by_level: dict[str, np.ndarray] = {}
    for heading_level in normalized_heading_levels:
        array_name = HEADING_LEVEL_TO_ARRAY[heading_level]
        if array_name not in track_group:
            raise ValueError(f"Heading source array {array_name!r} not found in {track_run_path}/tracks/id_{track_id}.")
        headings = np.asarray(track_group[array_name][:], dtype=np.float64)
        if headings.shape[0] != frames.shape[0]:
            raise ValueError(
                f"Heading source {array_name!r} length {headings.shape[0]} does not match frames length {frames.shape[0]}."
            )
        metrics = _build_metrics_for_heading(
            bouts=bouts,
            peak_events=peak_events,
            frames=frames,
            times=times,
            headings=headings,
            transition_valid=transition_valid,
            sample_valid=sample_valid,
            positions_mm=positions_mm,
            positions_px=positions_px,
            fps=fps,
            pre_post_mode=pre_post_mode,
            pre_window_frames=pre_window_frames,
            post_window_frames=post_window_frames,
            within_window=within_window,
            derivative_threshold_deg_s=zero_crossing_derivative_threshold_deg_s,
            dominant_frequency_enabled=dominant_frequency,
            dominant_frequency_min_samples=dominant_frequency_min_samples,
            dominant_frequency_detrend=dominant_frequency_detrend,
        )
        if layout == LAYOUT_HIERARCHICAL_V1:
            level_group = run_group.create_group(heading_level)
            level_group.attrs["heading_source_array"] = array_name
            level_group.attrs["is_default_heading_level"] = heading_level == default_heading_level
            level_group.attrs["source_swim_bout_path"] = swim_level_path
            write_columnar_dataset(
                level_group,
                "per_bout_metrics",
                metrics,
                attrs={
                    "schema_id": f"{SCHEMA_ID}.per_bout_metrics",
                    "schema_version": SCHEMA_VERSION,
                    "heading_level": heading_level,
                    "heading_source_array": array_name,
                    "source_bout_count": int(len(bouts)),
                    "source_bout_field_names": source_bout_field_names,
                    "source_interpolated_threshold_fields": source_interpolated_threshold_fields,
                    "source_peak_event_fields": source_peak_event_fields,
                },
            )
        written_levels.append(heading_level)
        metrics_by_level[heading_level] = metrics

    written_analysis_levels = [MOVEMENT_LEVEL, *written_levels]
    eye_gaze_metrics: Optional[np.ndarray] = None
    eye_gaze_attrs: Optional[dict[str, object]] = None
    if include_eye_gaze:
        if eye_series is None:
            raise ValueError("include_eye_gaze=True did not resolve an eye-angle source.")
        eye_gaze_metrics = _build_metrics_for_eye_gaze(
            bouts=bouts,
            frames=frames,
            eye_series=eye_series,
            pre_post_mode=pre_post_mode,
            pre_window_frames=pre_window_frames,
            post_window_frames=post_window_frames,
            within_window=within_window,
            eye_validity_min_fraction=float(eye_validity_min_fraction),
            vergence_threshold_deg=vergence_threshold_deg,
        )
        eye_gaze_attrs = {
            "schema_id": f"{SCHEMA_ID}.eye_gaze.per_bout_metrics",
            "schema_version": SCHEMA_VERSION,
            "analysis_level": EYE_GAZE_LEVEL,
            "eye_angle_family": str(eye_angle_family),
            "eye_angle_run": source_refs["source_eye_angle_run"],
            "eye_angle_path": source_refs["source_eye_angle_path"],
            "source_bout_count": int(len(bouts)),
            "source_bout_field_names": source_bout_field_names,
            "source_interpolated_threshold_fields": source_interpolated_threshold_fields,
            "eye_validity_min_fraction": float(eye_validity_min_fraction),
            "vergence_threshold_deg": (
                None if vergence_threshold_deg is None else float(vergence_threshold_deg)
            ),
        }
        if layout == LAYOUT_HIERARCHICAL_V1:
            eye_group = run_group.create_group(EYE_GAZE_LEVEL)
            eye_group.attrs["eye_angle_family"] = str(eye_angle_family)
            eye_group.attrs["eye_angle_run"] = source_refs["source_eye_angle_run"]
            eye_group.attrs["eye_angle_path"] = source_refs["source_eye_angle_path"]
            eye_group.attrs["source_swim_bout_path"] = swim_level_path
            write_columnar_dataset(
                eye_group,
                "per_bout_metrics",
                eye_gaze_metrics,
                attrs=eye_gaze_attrs,
            )
        written_analysis_levels.append(EYE_GAZE_LEVEL)

    run_group.attrs["heading_levels"] = written_levels
    run_group.attrs["analysis_levels"] = written_analysis_levels

    if layout == LAYOUT_COMPACT_TABULAR_V2:
        _write_compact_bout_kinematics_tables(
            run_group,
            movement_metrics=movement_metrics,
            movement_attrs=movement_attrs,
            metrics_by_level=metrics_by_level,
            heading_levels=written_levels,
            default_heading_level=default_heading_level,
            heading_table_attrs={
                "schema_id": f"{SCHEMA_ID}.per_bout_metrics",
                "schema_version": SCHEMA_VERSION,
                "source_bout_count": int(len(bouts)),
                "source_bout_field_names": source_bout_field_names,
                "source_interpolated_threshold_fields": source_interpolated_threshold_fields,
                "source_peak_event_fields": source_peak_event_fields,
                "source_heading_arrays": source_heading_arrays,
            },
            eye_gaze_metrics=eye_gaze_metrics,
            eye_gaze_attrs=eye_gaze_attrs,
        )

    git_info = get_git_info()
    env_info = get_environment_info(disk_path=str(zarr_path), capture_env_vars=False)
    provenance = build_stage_provenance(
        stage="bout_kinematics",
        created_at_utc=created_at_utc,
        parameters=parameters,
        inputs=source_refs,
        command=command,
        version=METHOD_VERSION,
        git=git_info,
        environment=env_info.get("environment"),
        platform=env_info.get("platform"),
        artifacts={
            "run_path": f"analysis/bout_kinematics_runs/{run_name}",
            "layout": layout,
            "heading_levels": written_levels,
            "analysis_levels": written_analysis_levels,
            "tables": (
                [COMPACT_LEVEL_INDEX, COMPACT_MOVEMENT_TABLE, COMPACT_HEADING_TABLE]
                + ([COMPACT_EYE_GAZE_TABLE] if eye_gaze_metrics is not None else [])
                if layout == LAYOUT_COMPACT_TABULAR_V2
                else ["movement/per_bout_metrics", *[f"{level}/per_bout_metrics" for level in written_levels]]
                + (["eye_gaze/per_bout_metrics"] if eye_gaze_metrics is not None else [])
            ),
        },
    )
    write_stage_provenance(run_group, provenance)
    write_best_effort_run_lineage_attrs(run_group, run_family="bout_kinematics_run")

    if write_visualizations:
        try:
            write_bout_kinematics_visualization_artifacts(
                zarr_path=zarr_path,
                run_group=run_group,
                run_name=str(run_name),
                metrics_by_level=metrics_by_level,
                movement_metrics=movement_metrics,
                eye_gaze_metrics=eye_gaze_metrics,
                source_refs=source_refs,
                parameters=parameters,
                heading_levels=written_levels,
                default_heading_level=default_heading_level,
                source_speed_level=source_speed_level,
                layout=layout,
                bins=int(visualization_bins),
                artifact_dpi=int(visualization_dpi),
                command=command,
            )
        except Exception as exc:
            run_group.attrs["status"] = "failed"
            run_group.attrs["failure_stage"] = "bout_kinematics_visualization"
            run_group.attrs["failure_reason"] = f"{type(exc).__name__}: {exc}"
            mark_run_failed(run_group, error=f"{type(exc).__name__}: {exc}")
            raise

    run_group.attrs["status"] = "complete"
    mark_run_complete(run_group, parent_group=parent, run_name=run_name)

    return str(run_name)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Compute per-bout heading kinematics.")
    parser.add_argument("zarr_path", type=Path, help="Path to the Palette Zarr archive.")
    parser.add_argument("--run-name", type=str, default=None, help="Output bout-kinematics run name.")
    parser.add_argument("--overwrite", action="store_true", help="Replace --run-name if it exists.")
    parser.add_argument("--track-kinematics-run", type=str, default="latest", help="Track kinematics run name/path.")
    parser.add_argument("--track-scope", type=str, default="offline", help="Track kinematics scope for bare run names.")
    parser.add_argument("--track-id", type=int, default=0, help="Track ID to analyze.")
    parser.add_argument("--swim-bout-run", type=str, default="latest", help="Source swim-bout run name.")
    parser.add_argument("--speed-level", type=str, default="filtered", help="Source swim-bout speed level.")
    parser.add_argument(
        "--heading-level",
        action="append",
        dest="heading_levels",
        default=None,
        help="Heading level to compute. Repeatable. Defaults to smoothed and raw.",
    )
    parser.add_argument("--default-heading-level", type=str, default="heading_smoothed")
    parser.add_argument(
        "--pre-post-mode",
        choices=PRE_POST_MODES,
        default="fixed_window",
        help="How to resolve pre/post measurement epochs.",
    )
    parser.add_argument("--pre-window-s", type=float, default=0.05)
    parser.add_argument("--post-window-s", type=float, default=0.05)
    parser.add_argument("--within-window", choices=WITHIN_WINDOWS, default="bout_start_end")
    parser.add_argument(
        "--physical-active-signal-level",
        choices=tuple(level.removeprefix("speed_") for level in PHYSICAL_ACTIVE_SPEED_LEVELS),
        default="filtered",
        help="Physical speed source for active-duration measurement (default: filtered).",
    )
    parser.add_argument(
        "--physical-active-threshold-mm-s",
        type=float,
        default=0.01,
        help="Threshold on the physical speed source for active-duration measurement (default: 0.01).",
    )
    parser.add_argument(
        "--physical-active-boundary-constraint",
        choices=PHYSICAL_ACTIVE_BOUNDARY_CONSTRAINTS,
        default="search_with_margin",
        help="How physical-active boundaries may relate to detector boundaries.",
    )
    parser.add_argument(
        "--physical-active-boundary-margin-s",
        type=float,
        default=0.05,
        help="Search margin around detector boundaries when using search_with_margin (default: 0.05).",
    )
    parser.add_argument("--zero-crossing-derivative-threshold-deg-s", type=float, default=0.0)
    parser.add_argument("--dominant-frequency", action="store_true", help="Compute optional dominant frequency.")
    parser.add_argument("--dominant-frequency-min-samples", type=int, default=8)
    parser.add_argument(
        "--dominant-frequency-no-detrend",
        action="store_true",
        help="Disable linear detrending before frequency estimation.",
    )
    parser.add_argument(
        "--include-eye-gaze",
        action="store_true",
        help="Also compute per-bout eye-gaze summaries from an eye-angle v2 run.",
    )
    parser.add_argument("--eye-angle-run", type=str, default="latest")
    parser.add_argument("--eye-angle-family", choices=EYE_ANGLE_FAMILIES, default="gaze")
    parser.add_argument("--eye-validity-min-fraction", type=float, default=1.0)
    parser.add_argument(
        "--vergence-threshold-deg",
        type=float,
        default=None,
        help="Optional vergence threshold used for converged-fraction summaries.",
    )
    parser.add_argument(
        "--write-zarr-artifacts",
        action="store_true",
        help="Write PNG and interactive visualization artifacts under the bout-kinematics run.",
    )
    parser.add_argument(
        "--layout",
        choices=BOUT_KINEMATICS_LAYOUTS,
        default=BOUT_KINEMATICS_LAYOUT_DEFAULT,
        help=(
            "Physical Zarr layout to write. Default: "
            f"{BOUT_KINEMATICS_LAYOUT_DEFAULT}. Use hierarchical_v1 for legacy/debug compatibility."
        ),
    )
    parser.add_argument("--visualization-bins", type=int, default=40)
    parser.add_argument("--visualization-dpi", type=int, default=150)
    args = parser.parse_args(argv)

    compute_and_save_bout_kinematics(
        zarr_path=args.zarr_path,
        run_name=args.run_name,
        track_kinematics_run=args.track_kinematics_run,
        track_scope=args.track_scope,
        track_id=args.track_id,
        swim_bout_run=args.swim_bout_run,
        speed_level=args.speed_level,
        heading_levels=tuple(args.heading_levels) if args.heading_levels else ("heading_smoothed", "heading_raw"),
        default_heading_level=args.default_heading_level,
        pre_post_mode=args.pre_post_mode,
        pre_window_s=args.pre_window_s,
        post_window_s=args.post_window_s,
        within_window=args.within_window,
        physical_active_signal_level=args.physical_active_signal_level,
        physical_active_threshold_mm_s=args.physical_active_threshold_mm_s,
        physical_active_boundary_constraint=args.physical_active_boundary_constraint,
        physical_active_boundary_margin_s=args.physical_active_boundary_margin_s,
        zero_crossing_derivative_threshold_deg_s=args.zero_crossing_derivative_threshold_deg_s,
        dominant_frequency=args.dominant_frequency,
        dominant_frequency_min_samples=args.dominant_frequency_min_samples,
        dominant_frequency_detrend=not args.dominant_frequency_no_detrend,
        include_eye_gaze=args.include_eye_gaze,
        eye_angle_run=args.eye_angle_run,
        eye_angle_family=args.eye_angle_family,
        eye_validity_min_fraction=args.eye_validity_min_fraction,
        vergence_threshold_deg=args.vergence_threshold_deg,
        write_visualizations=args.write_zarr_artifacts,
        visualization_bins=args.visualization_bins,
        visualization_dpi=args.visualization_dpi,
        layout=args.layout,
        overwrite=args.overwrite,
        command=" ".join(sys.argv if argv is None else [sys.argv[0], *argv]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
