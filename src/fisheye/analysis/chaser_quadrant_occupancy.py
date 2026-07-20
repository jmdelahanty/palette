"""Compute chaser-relative quadrant occupancy for one or more chasers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import zarr  # noqa: E402

from fisheye.analysis.chaser_distance_runs import (
    ChaserDistanceWindow,
    _bytes_array,
    _write_array,
)
from fisheye.analysis.chaser_distance_io import (
    ChaserDistanceReadSnapshot,
    load_chaser_distance_run,
    reject_unsealed_chaser_derived_publication,
)
from fisheye.analysis.chaser_behavior import (
    BEHAVIOR_CLASS_LABELS,
    canonical_behavior_label,
    resolve_configured_chaser_behaviors,
)
from fisheye.analysis.chaser_profiles import (
    ChaserProtocolProfile,
    load_chaser_protocol_profile,
    resolve_protocol_payload_path,
)
from fisheye.shared.json_safety import decode_null_terminated_text, json_attr_safe
from fisheye.shared.plot_artifacts import write_interactive_plot_spec_artifact, write_png_visualization_artifact
from fisheye.shared.run_lineage_fingerprint import build_run_lineage_payload, write_run_lineage_attrs
from fisheye.shared.system_metadata import get_git_info

SCHEMA_ID = "palette.chaser.quadrant_occupancy.v1"
SCHEMA_VERSION = 1
METHOD = "chaser_relative_quadrant_occupancy"
METHOD_VERSION = "1"
COMPONENT_PARENT_NAME = "chaser_quadrant_occupancy"
DEFAULT_COMPONENT_NAME = "chaser_relative_pre_post_v1"
OVERVIEW_PNG_ARTIFACT_NAME = "chaser_quadrant_occupancy_overview_png"
INTERACTIVE_ARTIFACT_NAME = "chaser_quadrant_occupancy_interactive"
INTERACTIVE_RENDERER = "palette-chaser-quadrant-occupancy-v1"
INTERACTIVE_SPEC_SCHEMA_ID = "palette.chaser.quadrant_occupancy.interactive_spec.v1"
PHASE_LABELS = ("pre_static", "post_static")
SOURCE_WINDOW_LABELS = ("pre_event", "post_event")
QUADRANT_LABELS = ("top_left", "top_right", "bottom_left", "bottom_right")


@dataclass(frozen=True)
class ChaserQuadrantRole:
    chaser_index: int
    behavior_class: str
    behavior_class_id: int
    raw_color_rgba: tuple[float, float, float, float]
    raw_color_hex: str
    enable_chase: bool
    enable_random_movement: bool
    behavior_mode: int | None
    start_position_preset: str
    end_position_preset: str


@dataclass(frozen=True)
class ChaserQuadrantPhase:
    phase_index: int
    phase_label: str
    source_window_label: str
    source_start_frame: int
    source_end_frame: int
    effective_start_frame: int
    effective_end_frame: int
    settle_excluded_frame_count: int


@dataclass(frozen=True)
class ChaserQuadrantOccupancyResult:
    zarr_path: str
    recording_id: str
    component_name: str
    chaser_distance_run_name: str
    chaser_distance_run_path: str
    source_stimulus_run: str | None
    source_stimulus_path: str | None
    source_stimulus_epoch_run: str | None
    source_stimulus_epoch_path: str | None
    fps: float
    total_frames: int
    pixels_per_mm_projector: float
    coordinate_frame: str
    coordinate_origin: str
    quadrant_bounds_source: str | None
    quadrant_width_px: float
    quadrant_height_px: float
    dropout_warning_fraction: float
    dropout_exclusion_fraction: float | None
    static_chaser_drift_warning_mm: float
    chasers: tuple[ChaserQuadrantRole, ...]
    phases: tuple[ChaserQuadrantPhase, ...]
    chaser_phase_x_px: np.ndarray
    chaser_phase_y_px: np.ndarray
    chaser_phase_x_mm: np.ndarray
    chaser_phase_y_mm: np.ndarray
    chaser_quadrant_code: np.ndarray
    chaser_position_sample_count: np.ndarray
    chaser_max_drift_mm: np.ndarray
    chaser_median_drift_mm: np.ndarray
    median_distance_mm: np.ndarray
    mean_distance_mm: np.ndarray
    occupancy_fraction: np.ndarray
    occupancy_fraction_of_epoch: np.ndarray
    valid_frame_count: np.ndarray
    distance_valid_frame_count: np.ndarray
    total_frame_count: np.ndarray
    missing_frame_count: np.ndarray
    tracking_dropout_fraction: np.ndarray
    endpoint_status: str
    qc_warnings: tuple[str, ...]
    summary: dict[str, Any]
    diagnostics: dict[str, Any]
    protocol_profile_id: str | None = None
    protocol_profile_version: int | None = None
    protocol_profile_sha256: str | None = None
    post_settle_duration_s: float = 0.0


def _open_root(zarr_path: Path, *, mode: str) -> zarr.Group:
    return zarr.open_group(str(zarr_path), mode=mode, use_consolidated=False)


def _get_group_by_path(root: zarr.Group, path: str | None) -> zarr.Group | None:
    normalized = "/".join(part for part in str(path or "").strip("/").split("/") if part)
    if not normalized:
        return root
    current: Any = root
    for part in normalized.split("/"):
        try:
            current = current[part]
        except Exception:
            return None
        if not isinstance(current, zarr.Group):
            return None
    return current


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _unit_to_u8(value: Any) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    return int(round(max(0.0, min(1.0, number)) * 255.0))


def _rgba_to_hex(rgba: tuple[float, float, float, float]) -> str:
    return "#{:02x}{:02x}{:02x}".format(_unit_to_u8(rgba[0]), _unit_to_u8(rgba[1]), _unit_to_u8(rgba[2]))


def _canonical_window_label(label: Any) -> str:
    text = str(label or "").strip().lower()
    if text.startswith("pre"):
        return "pre_event"
    if text.startswith("post"):
        return "post_event"
    if text.startswith(("train", "training")):
        return "training_event"
    return text


def _decode_text_column(data: np.ndarray) -> list[str]:
    values = np.asarray(data)
    if values.ndim == 2 and values.dtype.kind in ("u", "i"):
        return [decode_null_terminated_text(row).strip() for row in values]
    return [decode_null_terminated_text(value).strip() for value in values.reshape(-1)]


def _read_windows(run_group: zarr.Group, *, fps: float) -> tuple[ChaserDistanceWindow, ...]:
    summary = run_group.get("epoch_summary")
    if summary is None:
        return ()
    required = ("window_id", "label_bytes", "start_frame", "end_frame")
    if any(name not in summary for name in required):
        return ()
    ids = np.asarray(summary["window_id"][:], dtype=np.int32).reshape(-1)
    labels = _decode_text_column(np.asarray(summary["label_bytes"][:]))
    starts = np.asarray(summary["start_frame"][:], dtype=np.int64).reshape(-1)
    ends = np.asarray(summary["end_frame"][:], dtype=np.int64).reshape(-1)
    n = min(ids.shape[0], len(labels), starts.shape[0], ends.shape[0])
    safe_fps = float(fps) if np.isfinite(fps) and fps > 0 else 1.0
    return tuple(
        ChaserDistanceWindow(
            window_id=int(ids[i]),
            label=str(labels[i]),
            start_frame=int(starts[i]),
            end_frame=int(ends[i]),
            start_time_s=float(starts[i]) / safe_fps,
            end_time_s=(float(ends[i]) + 1.0) / safe_fps,
            duration_s=max(0.0, (float(ends[i]) - float(starts[i]) + 1.0) / safe_fps),
        )
        for i in range(n)
    )


def _resolve_chaser_distance_run(
    root: zarr.Group,
    run_name: str,
) -> tuple[ChaserDistanceReadSnapshot, str, str]:
    snapshot = load_chaser_distance_run(
        root,
        run_name=str(run_name).strip() or "latest",
    )
    return snapshot, snapshot.run_name, snapshot.run_path


def _stimulus_group_from_run(root: zarr.Group, run_group: zarr.Group) -> tuple[zarr.Group | None, str | None, str | None]:
    source_path = run_group.attrs.get("source_stimulus_path")
    source_run = run_group.attrs.get("source_stimulus_run")
    source_group = _get_group_by_path(root, str(source_path or ""))
    if source_group is not None and isinstance(source_path, str) and source_path.strip():
        return source_group, str(source_run or "").strip() or None, source_path.strip()
    if source_run:
        candidate = f"analysis/stimulus_runs/{source_run}"
        candidate_group = _get_group_by_path(root, candidate)
        if candidate_group is not None:
            return candidate_group, str(source_run), candidate
    return None, str(source_run or "").strip() or None, str(source_path or "").strip() or None


def _protocol_payload_from_stimulus(stim_group: zarr.Group | None) -> dict[str, Any]:
    if stim_group is None:
        raise ValueError(
            "Cannot resolve stimulus run for chaser quadrant occupancy role mapping."
        )
    raw = stim_group.attrs.get("protocol_json")
    if not raw:
        raise ValueError(
            "Stimulus run lacks protocol_json required for chaser quadrant occupancy role mapping."
        )
    payload = json.loads(str(raw))
    if not isinstance(payload, dict):
        raise ValueError("Stimulus protocol_json did not decode to an object.")
    return payload


def resolve_chaser_roles_from_protocol_payload(
    payload: Mapping[str, Any],
) -> tuple[ChaserQuadrantRole, ...]:
    """Return canonical behavior roles from a variable-length chaser payload."""

    configured = resolve_configured_chaser_behaviors(payload)
    if not configured:
        raise ValueError(
            "chaser quadrant occupancy requires at least one configured chaser"
        )
    roles = [
        ChaserQuadrantRole(
            chaser_index=behavior.chaser_index,
            behavior_class=behavior.behavior_class,
            behavior_class_id=behavior.behavior_class_id,
            raw_color_rgba=behavior.raw_color_rgba,
            raw_color_hex=_rgba_to_hex(behavior.raw_color_rgba),
            enable_chase=behavior.enable_chase,
            enable_random_movement=behavior.enable_random_movement,
            behavior_mode=behavior.behavior_mode,
            start_position_preset=behavior.start_position_preset,
            end_position_preset=behavior.end_position_preset,
        )
        for behavior in configured
    ]
    return tuple(roles)


def _read_chaser_roles_from_distance_run(
    run_group: zarr.Group,
    *,
    protocol_payload: Mapping[str, Any],
) -> tuple[ChaserQuadrantRole, ...]:
    if "chasers" not in run_group:
        raise ValueError("chaser-distance run lacks the canonical chasers group")
    group = run_group["chasers"]
    required = ("chaser_index", "behavior_class_id", "behavior_class_label_bytes")
    missing = [name for name in required if name not in group]
    if missing:
        raise ValueError(
            "chaser-distance run lacks canonical chaser role arrays: " f"{missing}"
        )
    indices = np.asarray(group["chaser_index"][:], dtype=np.int64).reshape(-1)
    class_ids = np.asarray(group["behavior_class_id"][:], dtype=np.int16).reshape(-1)
    labels = _decode_text_column(np.asarray(group["behavior_class_label_bytes"][:]))
    if class_ids.shape[0] != indices.shape[0] or len(labels) != indices.shape[0]:
        raise ValueError(
            "canonical chaser identity and role arrays have different lengths"
        )
    stored_colors = (
        np.asarray(group["raw_color_rgba"][:], dtype=np.float64)
        if "raw_color_rgba" in group
        else np.full((indices.shape[0], 4), np.nan, dtype=np.float64)
    )
    if stored_colors.shape != (indices.shape[0], 4):
        raise ValueError("chasers/raw_color_rgba must have shape (chaser, rgba)")
    if "chaser_role_intervals" in run_group:
        intervals = run_group["chaser_role_intervals"]
        interval_required = (
            "chaser_index",
            "behavior_class_id",
            "start_frame",
            "end_frame",
        )
        interval_missing = [name for name in interval_required if name not in intervals]
        if interval_missing:
            raise ValueError(
                "chaser role interval contract is incomplete: " f"{interval_missing}"
            )
        interval_indices = np.asarray(
            intervals["chaser_index"][:], dtype=np.int64
        ).reshape(-1)
        interval_class_ids = np.asarray(
            intervals["behavior_class_id"][:], dtype=np.int16
        ).reshape(-1)
        interval_starts = np.asarray(
            intervals["start_frame"][:], dtype=np.int64
        ).reshape(-1)
        interval_ends = np.asarray(intervals["end_frame"][:], dtype=np.int64).reshape(
            -1
        )
        total_frames = int(run_group.attrs.get("total_frames") or 0)
        for chaser_index, class_id in zip(indices, class_ids):
            rows = np.flatnonzero(interval_indices == int(chaser_index))
            whole_recording = (
                rows.size == 1
                and int(interval_starts[rows[0]]) == 0
                and int(interval_ends[rows[0]]) in {-1, max(0, total_frames - 1)}
                and int(interval_class_ids[rows[0]]) == int(class_id)
            )
            if not whole_recording:
                raise ValueError(
                    "chaser quadrant occupancy v1 requires one stable whole-recording "
                    f"role interval per chaser; chaser_index={int(chaser_index)}"
                )
    try:
        configured = {
            int(value.chaser_index): value
            for value in resolve_configured_chaser_behaviors(protocol_payload)
        }
    except ValueError:
        configured = {}
    roles: list[ChaserQuadrantRole] = []
    for column, chaser_index in enumerate(indices):
        class_id = int(class_ids[column])
        role = canonical_behavior_label(labels[column])
        expected_role = BEHAVIOR_CLASS_LABELS.get(class_id)
        if expected_role is None or role != expected_role:
            raise ValueError(
                "chaser-distance behavior class id/label mismatch: "
                f"chaser_index={int(chaser_index)}, id={class_id}, label={role!r}"
            )
        source = configured.get(int(chaser_index))
        rgba_values = stored_colors[column]
        if not np.isfinite(rgba_values).all() and source is not None:
            rgba_values = np.asarray(source.raw_color_rgba, dtype=np.float64)
        if not np.isfinite(rgba_values).all():
            rgba_values = np.asarray((0.39, 0.45, 0.55, 1.0), dtype=np.float64)
        rgba = tuple(float(value) for value in rgba_values)
        roles.append(
            ChaserQuadrantRole(
                chaser_index=int(chaser_index),
                behavior_class=role,
                behavior_class_id=class_id,
                raw_color_rgba=rgba,  # type: ignore[arg-type]
                raw_color_hex=_rgba_to_hex(rgba),  # type: ignore[arg-type]
                enable_chase=bool(source.enable_chase) if source is not None else False,
                enable_random_movement=(
                    bool(source.enable_random_movement) if source is not None else False
                ),
                behavior_mode=source.behavior_mode if source is not None else None,
                start_position_preset=(
                    source.start_position_preset if source is not None else ""
                ),
                end_position_preset=(
                    source.end_position_preset if source is not None else ""
                ),
            )
        )
    if not roles:
        raise ValueError("chaser quadrant occupancy requires at least one chaser")
    return tuple(roles)


def _post_settle_duration_s_from_profile(
    profile: ChaserProtocolProfile | None,
    payload: Mapping[str, Any],
) -> float:
    if profile is None:
        return 0.0
    if profile.role_resolver_id != "configured_chaser_behavior_flags":
        raise ValueError(
            "unsupported chaser role resolver for quadrant occupancy: "
            f"{profile.role_resolver_id!r}"
        )
    parameters = profile.analysis_parameters.get("chaser_quadrant_occupancy", {})
    fallback = max(
        0.0,
        _safe_float(parameters.get("post_settle_duration_fallback_s"), 0.0),
    )
    source_path = str(parameters.get("post_settle_duration_source") or "").strip()
    if not source_path:
        return fallback
    value = resolve_protocol_payload_path(payload, source_path)
    return max(0.0, _safe_float(value, fallback))


def _quadrant_bounds_from_stimulus(stim_group: zarr.Group | None) -> tuple[float, float, str | None]:
    arena_name: str | None = None
    if stim_group is not None:
        arena = _get_group_by_path(stim_group, "stimulus_coordinates/arena_1")
        if arena is None:
            coords = stim_group.get("stimulus_coordinates")
            if isinstance(coords, zarr.Group):
                if "arena_1" in coords:
                    arena = coords["arena_1"]
                    arena_name = "arena_1"
                else:
                    keys_fn = getattr(coords, "group_keys", None)
                    keys = sorted(str(key) for key in keys_fn()) if callable(keys_fn) else []
                    arena_keys = [key for key in keys if key.startswith("arena_")]
                    if len(arena_keys) == 1:
                        arena_name = arena_keys[0]
                        arena = coords[arena_name]
        else:
            arena_name = "arena_1"
    else:
        arena = None
    if arena is not None:
        width = _safe_float(arena.attrs.get("texture_width_px"))
        height = _safe_float(arena.attrs.get("texture_height_px"))
        if np.isfinite(width) and width > 0 and np.isfinite(height) and height > 0:
            return (
                float(width),
                float(height),
                f"analysis/stimulus_runs/*/stimulus_coordinates/{arena_name or 'arena_1'}",
            )
    raise ValueError("Unable to resolve arena quadrant bounds from stimulus_coordinates/arena_*.")


def quadrant_code_for_xy(x: float, y: float, *, width_px: float, height_px: float) -> int:
    if not (math.isfinite(float(x)) and math.isfinite(float(y))):
        return -1
    if x < 0 or y < 0 or x >= float(width_px) or y >= float(height_px):
        return -1
    right = float(x) >= (float(width_px) / 2.0)
    bottom = float(y) >= (float(height_px) / 2.0)
    if not right and not bottom:
        return 0
    if right and not bottom:
        return 1
    if not right and bottom:
        return 2
    return 3


def _fish_in_quadrant(fish_xy: np.ndarray, quadrant_code: int, *, width_px: float, height_px: float) -> np.ndarray:
    codes = np.asarray(
        [
            quadrant_code_for_xy(float(x), float(y), width_px=width_px, height_px=height_px)
            for x, y in np.asarray(fish_xy, dtype=np.float64)
        ],
        dtype=np.int16,
    )
    return codes == int(quadrant_code)


def resolve_effective_phase_windows(
    windows: Sequence[ChaserDistanceWindow],
    *,
    fps: float,
    post_settle_duration_s: float,
) -> tuple[ChaserQuadrantPhase, ...]:
    by_label = {_canonical_window_label(window.label): window for window in windows}
    missing = [label for label in SOURCE_WINDOW_LABELS if label not in by_label]
    if missing:
        raise ValueError(
            f"Missing chaser quadrant occupancy source window(s): {missing}"
        )
    safe_fps = float(fps) if np.isfinite(fps) and fps > 0 else 1.0
    post_trim = int(math.ceil(max(0.0, float(post_settle_duration_s)) * safe_fps))
    pre = by_label["pre_event"]
    post = by_label["post_event"]
    phases = (
        ChaserQuadrantPhase(
            phase_index=0,
            phase_label="pre_static",
            source_window_label=pre.label,
            source_start_frame=int(pre.start_frame),
            source_end_frame=int(pre.end_frame),
            effective_start_frame=int(pre.start_frame),
            effective_end_frame=int(pre.end_frame),
            settle_excluded_frame_count=0,
        ),
        ChaserQuadrantPhase(
            phase_index=1,
            phase_label="post_static",
            source_window_label=post.label,
            source_start_frame=int(post.start_frame),
            source_end_frame=int(post.end_frame),
            effective_start_frame=int(post.start_frame) + post_trim,
            effective_end_frame=int(post.end_frame),
            settle_excluded_frame_count=post_trim,
        ),
    )
    if phases[1].effective_start_frame > phases[1].effective_end_frame:
        raise ValueError(
            "Post-settle trimming removed the full post window: "
            f"start={phases[1].effective_start_frame}, end={phases[1].effective_end_frame}."
        )
    return phases


def _phase_slice(phase: ChaserQuadrantPhase, total_frames: int) -> slice:
    start = max(0, int(phase.effective_start_frame))
    end = min(int(total_frames) - 1, int(phase.effective_end_frame))
    if end < start:
        return slice(0, 0)
    return slice(start, end + 1)


def _summary_value(
    array: np.ndarray, phase_index: int, chaser_index: int
) -> float | None:
    value = float(array[int(phase_index), int(chaser_index)])
    return value if math.isfinite(value) else None


def _compute_endpoint_arrays(
    *,
    chasers: Sequence[ChaserQuadrantRole],
    phases: Sequence[ChaserQuadrantPhase],
    chaser_indices: np.ndarray,
    fish_xy: np.ndarray,
    chaser_xy: np.ndarray,
    fish_valid: np.ndarray,
    chaser_valid: np.ndarray,
    distance_mm: np.ndarray,
    width_px: float,
    height_px: float,
    pixels_per_mm: float,
    dropout_warning_fraction: float,
    static_chaser_drift_warning_mm: float,
) -> tuple[dict[str, np.ndarray], tuple[str, ...], dict[str, Any]]:
    n_phases = len(phases)
    n_chasers = len(chasers)
    total_frames = int(fish_xy.shape[0])
    by_chaser_index = {
        int(value): idx
        for idx, value in enumerate(np.asarray(chaser_indices).reshape(-1))
    }
    chaser_columns: list[int] = []
    for obj in chasers:
        if obj.chaser_index not in by_chaser_index:
            raise ValueError(
                f"Chaser index {obj.chaser_index} not present in chaser indices {list(by_chaser_index)}."
            )
        chaser_columns.append(int(by_chaser_index[obj.chaser_index]))

    shape = (n_phases, n_chasers)
    chaser_x_px = np.full(shape, np.nan, dtype=np.float32)
    chaser_y_px = np.full(shape, np.nan, dtype=np.float32)
    chaser_x_mm = np.full(shape, np.nan, dtype=np.float32)
    chaser_y_mm = np.full(shape, np.nan, dtype=np.float32)
    chaser_quadrant_code = np.full(shape, -1, dtype=np.int16)
    chaser_sample_count = np.zeros(shape, dtype=np.int64)
    chaser_max_drift_mm = np.full(shape, np.nan, dtype=np.float32)
    chaser_median_drift_mm = np.full(shape, np.nan, dtype=np.float32)
    median_distance = np.full(shape, np.nan, dtype=np.float32)
    mean_distance = np.full(shape, np.nan, dtype=np.float32)
    occupancy_fraction = np.full(shape, np.nan, dtype=np.float32)
    occupancy_fraction_epoch = np.full(shape, np.nan, dtype=np.float32)
    valid_frame_count = np.zeros(shape, dtype=np.int64)
    distance_valid_count = np.zeros(shape, dtype=np.int64)
    total_frame_count = np.zeros(shape, dtype=np.int64)
    missing_frame_count = np.zeros(shape, dtype=np.int64)
    dropout_fraction = np.full(shape, np.nan, dtype=np.float32)
    warnings: list[str] = []

    for p_idx, phase in enumerate(phases):
        slc = _phase_slice(phase, total_frames)
        phase_len = max(0, int(slc.stop - slc.start))
        phase_fish_xy = fish_xy[slc]
        phase_fish_valid = np.asarray(fish_valid[slc], dtype=bool) & np.isfinite(
            phase_fish_xy
        ).all(axis=1)
        for o_idx, col_idx in enumerate(chaser_columns):
            total_frame_count[p_idx, o_idx] = int(phase_len)
            valid_frame_count[p_idx, o_idx] = int(np.count_nonzero(phase_fish_valid))
            missing_frame_count[p_idx, o_idx] = int(max(0, phase_len - valid_frame_count[p_idx, o_idx]))
            dropout = (
                float(missing_frame_count[p_idx, o_idx]) / float(phase_len)
                if phase_len > 0
                else math.nan
            )
            dropout_fraction[p_idx, o_idx] = dropout
            if math.isfinite(dropout) and dropout > float(dropout_warning_fraction):
                warnings.append(
                    f"{phase.phase_label}:{chasers[o_idx].behavior_class}:tracking_dropout_fraction>{dropout_warning_fraction:g}"
                )

            phase_chaser_xy = chaser_xy[slc, col_idx, :]
            phase_chaser_valid = np.asarray(
                chaser_valid[slc, col_idx], dtype=bool
            ) & np.isfinite(phase_chaser_xy).all(axis=1)
            chaser_sample_count[p_idx, o_idx] = int(
                np.count_nonzero(phase_chaser_valid)
            )
            if chaser_sample_count[p_idx, o_idx] > 0:
                samples = phase_chaser_xy[phase_chaser_valid].astype(np.float64)
                center = np.nanmedian(samples, axis=0)
                chaser_x_px[p_idx, o_idx] = float(center[0])
                chaser_y_px[p_idx, o_idx] = float(center[1])
                chaser_x_mm[p_idx, o_idx] = float(center[0]) / float(pixels_per_mm)
                chaser_y_mm[p_idx, o_idx] = float(center[1]) / float(pixels_per_mm)
                drift_px = np.linalg.norm(samples - center.reshape(1, 2), axis=1)
                drift_mm = drift_px / float(pixels_per_mm)
                chaser_max_drift_mm[p_idx, o_idx] = float(np.nanmax(drift_mm))
                chaser_median_drift_mm[p_idx, o_idx] = float(np.nanmedian(drift_mm))
                if chaser_max_drift_mm[p_idx, o_idx] > float(
                    static_chaser_drift_warning_mm
                ):
                    warnings.append(
                        f"{phase.phase_label}:{chasers[o_idx].behavior_class}:chaser_max_drift_mm>{static_chaser_drift_warning_mm:g}"
                    )
                chaser_quadrant_code[p_idx, o_idx] = quadrant_code_for_xy(
                    float(center[0]),
                    float(center[1]),
                    width_px=float(width_px),
                    height_px=float(height_px),
                )

            phase_distance = distance_mm[slc, col_idx]
            distance_valid = phase_fish_valid & phase_chaser_valid & np.isfinite(phase_distance)
            distance_valid_count[p_idx, o_idx] = int(np.count_nonzero(distance_valid))
            if distance_valid_count[p_idx, o_idx] > 0:
                values = np.asarray(phase_distance[distance_valid], dtype=np.float64)
                median_distance[p_idx, o_idx] = float(np.nanmedian(values))
                mean_distance[p_idx, o_idx] = float(np.nanmean(values))

            q_code = int(chaser_quadrant_code[p_idx, o_idx])
            if q_code >= 0 and valid_frame_count[p_idx, o_idx] > 0:
                in_quad = _fish_in_quadrant(
                    phase_fish_xy,
                    q_code,
                    width_px=float(width_px),
                    height_px=float(height_px),
                )
                in_quad_valid = in_quad & phase_fish_valid
                occupancy_fraction[p_idx, o_idx] = float(np.count_nonzero(in_quad_valid)) / float(valid_frame_count[p_idx, o_idx])
                occupancy_fraction_epoch[p_idx, o_idx] = (
                    float(np.count_nonzero(in_quad_valid)) / float(phase_len)
                    if phase_len > 0
                    else math.nan
                )

    arrays = {
        "chaser_x_px": chaser_x_px,
        "chaser_y_px": chaser_y_px,
        "chaser_x_mm": chaser_x_mm,
        "chaser_y_mm": chaser_y_mm,
        "chaser_quadrant_code": chaser_quadrant_code,
        "chaser_position_sample_count": chaser_sample_count,
        "chaser_max_drift_mm": chaser_max_drift_mm,
        "chaser_median_drift_mm": chaser_median_drift_mm,
        "median_distance_mm": median_distance,
        "mean_distance_mm": mean_distance,
        "occupancy_fraction": occupancy_fraction,
        "occupancy_fraction_of_epoch": occupancy_fraction_epoch,
        "valid_frame_count": valid_frame_count,
        "distance_valid_frame_count": distance_valid_count,
        "total_frame_count": total_frame_count,
        "missing_frame_count": missing_frame_count,
        "tracking_dropout_fraction": dropout_fraction,
    }
    diagnostics = {
        "chaser_columns": chaser_columns,
        "warning_count": len(warnings),
    }
    return arrays, tuple(warnings), diagnostics


def _build_summary(
    *,
    recording_id: str,
    chasers: Sequence[ChaserQuadrantRole],
    arrays: Mapping[str, np.ndarray],
    phases: Sequence[ChaserQuadrantPhase],
) -> dict[str, Any]:
    median_distance = arrays["median_distance_mm"]
    occupancy = arrays["occupancy_fraction"]
    dropout = arrays["tracking_dropout_fraction"]
    valid_counts = arrays["valid_frame_count"]
    q_codes = arrays["chaser_quadrant_code"]

    def q_label(phase_idx: int, chaser_idx: int) -> str | None:
        code = int(q_codes[phase_idx, chaser_idx])
        if 0 <= code < len(QUADRANT_LABELS):
            return QUADRANT_LABELS[code]
        return None

    def delta(values: Sequence[float | None]) -> float | None:
        if len(values) < 2 or values[0] is None or values[-1] is None:
            return None
        return float(values[-1]) - float(values[0])

    per_chaser: list[dict[str, Any]] = []
    role_members: dict[str, list[int]] = {}
    for chaser_pos, chaser in enumerate(chasers):
        distances = [
            _summary_value(median_distance, phase_pos, chaser_pos)
            for phase_pos in range(len(phases))
        ]
        occupancies = [
            _summary_value(occupancy, phase_pos, chaser_pos)
            for phase_pos in range(len(phases))
        ]
        dropouts = [
            _summary_value(dropout, phase_pos, chaser_pos)
            for phase_pos in range(len(phases))
        ]
        per_chaser.append(
            {
                "chaser_index": int(chaser.chaser_index),
                "behavior_class_id": int(chaser.behavior_class_id),
                "behavior_class": chaser.behavior_class,
                "raw_color_hex": chaser.raw_color_hex,
                "phase_values": [
                    {
                        "phase_label": phase.phase_label,
                        "median_distance_mm": distances[phase_pos],
                        "occupancy_fraction": occupancies[phase_pos],
                        "tracking_dropout_fraction": dropouts[phase_pos],
                        "quadrant": q_label(phase_pos, chaser_pos),
                    }
                    for phase_pos, phase in enumerate(phases)
                ],
                "first_to_last_delta_median_distance_mm": delta(distances),
                "first_to_last_delta_occupancy_fraction": delta(occupancies),
            }
        )
        role_members.setdefault(chaser.behavior_class, []).append(chaser_pos)

    def role_mean(
        source: np.ndarray, phase_pos: int, members: Sequence[int]
    ) -> float | None:
        values = np.asarray(
            [source[phase_pos, member] for member in members], dtype=np.float64
        )
        finite = values[np.isfinite(values)]
        return float(np.mean(finite)) if finite.size else None

    per_role: list[dict[str, Any]] = []
    for role, members in sorted(role_members.items()):
        distance_values = [
            role_mean(median_distance, phase_pos, members)
            for phase_pos in range(len(phases))
        ]
        occupancy_values = [
            role_mean(occupancy, phase_pos, members) for phase_pos in range(len(phases))
        ]
        per_role.append(
            {
                "behavior_class": role,
                "chaser_count": len(members),
                "phase_values": [
                    {
                        "phase_label": phase.phase_label,
                        "mean_chaser_median_distance_mm": distance_values[phase_pos],
                        "mean_chaser_occupancy_fraction": occupancy_values[phase_pos],
                    }
                    for phase_pos, phase in enumerate(phases)
                ],
                "first_to_last_delta_median_distance_mm": delta(distance_values),
                "first_to_last_delta_occupancy_fraction": delta(occupancy_values),
            }
        )

    return {
        "fish_id": "0",
        "recording_id": recording_id,
        "dpf": None,
        "chaser_count": len(chasers),
        "phase_labels": [phase.phase_label for phase in phases],
        "valid_frame_count_by_phase": {
            phase.phase_label: int(np.nanmax(valid_counts[phase_pos, :]))
            for phase_pos, phase in enumerate(phases)
        },
        "per_chaser": per_chaser,
        "per_role": per_role,
        "pairwise_role_contrast_policy": "not_computed_at_recording_level",
    }


def build_chaser_quadrant_occupancy_result(
    zarr_path: Path,
    *,
    chaser_distance_run: str = "latest",
    component_name: str = DEFAULT_COMPONENT_NAME,
    dropout_warning_fraction: float = 0.20,
    dropout_exclusion_fraction: float | None = None,
    static_chaser_drift_warning_mm: float = 1.0,
    protocol_profile: str | Path | None = None,
) -> ChaserQuadrantOccupancyResult:
    root = _open_root(zarr_path, mode="r")
    distance, distance_run_name, distance_run_path = _resolve_chaser_distance_run(
        root,
        chaser_distance_run,
    )
    # This endpoint assigns scientific meaning to aggressive/inert role labels,
    # colors, and role intervals.  Those protocol-derived arrays are present in
    # the current writer but are not yet protected by the canonical publication
    # seal.  Do not silently recover them from mutable protocol_json.
    distance.require_behavior_authority()
    run_group = root[distance_run_path]
    coordinate_frame = str(run_group.attrs.get("coordinate_frame") or "")
    coordinate_origin = str(run_group.attrs.get("coordinate_origin") or "")
    if coordinate_frame != "arena_relative_canvas_px":
        raise ValueError(
            f"Chaser quadrant occupancy requires coordinate_frame='arena_relative_canvas_px'; got {coordinate_frame!r}."
        )
    if coordinate_origin != "top_left_of_active_arena":
        raise ValueError(
            f"Chaser quadrant occupancy requires coordinate_origin='top_left_of_active_arena'; got {coordinate_origin!r}."
        )
    stim_group, source_stimulus_run, source_stimulus_path = _stimulus_group_from_run(
        root, run_group
    )
    payload = _protocol_payload_from_stimulus(stim_group)
    chasers = _read_chaser_roles_from_distance_run(
        run_group,
        protocol_payload=payload,
    )
    loaded_profile = (
        load_chaser_protocol_profile(protocol_profile)
        if protocol_profile is not None
        else None
    )
    post_settle_s = _post_settle_duration_s_from_profile(loaded_profile, payload)
    width_px, height_px, bounds_source = _quadrant_bounds_from_stimulus(stim_group)

    fps = _safe_float(run_group.attrs.get("fps"), 1.0)
    total_frames_attr = int(run_group.attrs.get("total_frames", 0) or 0)
    pixels_per_mm = _safe_float(run_group.attrs.get("pixels_per_mm_projector"))
    if not np.isfinite(pixels_per_mm) or pixels_per_mm <= 0:
        raise ValueError(
            "Chaser-distance run lacks a positive pixels_per_mm_projector attr."
        )
    if (
        "frames" not in run_group
        or "positions" not in run_group
        or "distances" not in run_group
    ):
        raise ValueError(
            "Chaser-distance run is missing frames, positions, or distances group."
        )
    positions = run_group["positions"]
    distances = run_group["distances"]
    required_positions = ("fish_centroid_arena_xy", "chaser_arena_xy", "fish_valid", "chaser_valid")
    missing = [name for name in required_positions if name not in positions]
    missing += [name for name in ("distance_mm",) if name not in distances]
    if missing:
        raise ValueError(
            f"Chaser-distance run missing required chaser quadrant occupancy array(s): {missing}"
        )

    fish_xy = np.asarray(positions["fish_centroid_arena_xy"][:], dtype=np.float32)
    chaser_xy = np.asarray(positions["chaser_arena_xy"][:], dtype=np.float32)
    fish_valid = np.asarray(positions["fish_valid"][:], dtype=bool)
    chaser_valid = np.asarray(positions["chaser_valid"][:], dtype=bool)
    distance_mm = np.asarray(distances["distance_mm"][:], dtype=np.float32)
    chaser_indices = np.asarray(run_group["chasers"]["chaser_index"][:], dtype=np.int64)
    total_frames = int(fish_xy.shape[0])
    if total_frames_attr > 0 and total_frames_attr != total_frames:
        raise ValueError(
            "Chaser-distance run frame-axis mismatch: "
            f"attrs total_frames={total_frames_attr}, fish_centroid_arena_xy length={total_frames}."
        )
    expected = {
        "positions/chaser_arena_xy": chaser_xy.shape[:1],
        "positions/fish_valid": fish_valid.shape[:1],
        "positions/chaser_valid": chaser_valid.shape[:1],
        "distances/distance_mm": distance_mm.shape[:1],
    }
    mismatched = {name: shape for name, shape in expected.items() if shape != (total_frames,)}
    if mismatched:
        raise ValueError(f"Chaser-distance run arrays disagree on camera-frame axis: {mismatched}")
    if chaser_xy.ndim != 3 or chaser_xy.shape[2] != 2:
        raise ValueError("positions/chaser_arena_xy must have shape (frame, chaser, xy).")
    if distance_mm.ndim != 2 or distance_mm.shape[1] != chaser_xy.shape[1]:
        raise ValueError("distances/distance_mm must have shape (frame, chaser).")
    if chaser_valid.ndim != 2 or chaser_valid.shape[1] != chaser_xy.shape[1]:
        raise ValueError("positions/chaser_valid must have shape (frame, chaser).")
    if chaser_indices.shape[0] != chaser_xy.shape[1]:
        raise ValueError("chasers/chaser_index length does not match chaser position columns.")

    windows = _read_windows(run_group, fps=fps)
    phases = resolve_effective_phase_windows(windows, fps=fps, post_settle_duration_s=post_settle_s)
    arrays, qc_warnings, diagnostics = _compute_endpoint_arrays(
        chasers=chasers,
        phases=phases,
        chaser_indices=chaser_indices,
        fish_xy=fish_xy,
        chaser_xy=chaser_xy,
        fish_valid=fish_valid,
        chaser_valid=chaser_valid,
        distance_mm=distance_mm,
        width_px=float(width_px),
        height_px=float(height_px),
        pixels_per_mm=float(pixels_per_mm),
        dropout_warning_fraction=float(dropout_warning_fraction),
        static_chaser_drift_warning_mm=float(static_chaser_drift_warning_mm),
    )
    if dropout_exclusion_fraction is not None:
        dropout = arrays["tracking_dropout_fraction"]
        if np.any(np.isfinite(dropout) & (dropout > float(dropout_exclusion_fraction))):
            qc_warnings = tuple([*qc_warnings, f"tracking_dropout_fraction>{float(dropout_exclusion_fraction):g}"])
    summary = _build_summary(
        recording_id=str(
            run_group.attrs.get("recording_id")
            or root.attrs.get("recording_id")
            or Path(zarr_path).stem
        ),
        chasers=chasers,
        arrays=arrays,
        phases=phases,
    )
    endpoint_status = "computed"
    diagnostics.update(
        {
            "post_settle_duration_s": float(post_settle_s),
            "dropout_exclusion_fraction": dropout_exclusion_fraction,
            "qc_warning_count": len(qc_warnings),
        }
    )

    return ChaserQuadrantOccupancyResult(
        zarr_path=str(zarr_path),
        recording_id=str(summary["recording_id"]),
        component_name=str(component_name),
        chaser_distance_run_name=distance_run_name,
        chaser_distance_run_path=distance_run_path,
        source_stimulus_run=source_stimulus_run,
        source_stimulus_path=source_stimulus_path,
        source_stimulus_epoch_run=run_group.attrs.get("source_stimulus_epoch_run"),
        source_stimulus_epoch_path=run_group.attrs.get("source_stimulus_epoch_path"),
        fps=float(fps),
        total_frames=int(total_frames),
        pixels_per_mm_projector=float(pixels_per_mm),
        coordinate_frame=coordinate_frame,
        coordinate_origin=coordinate_origin,
        quadrant_bounds_source=bounds_source,
        quadrant_width_px=float(width_px),
        quadrant_height_px=float(height_px),
        dropout_warning_fraction=float(dropout_warning_fraction),
        dropout_exclusion_fraction=dropout_exclusion_fraction,
        static_chaser_drift_warning_mm=float(static_chaser_drift_warning_mm),
        chasers=chasers,
        phases=phases,
        chaser_phase_x_px=arrays["chaser_x_px"],
        chaser_phase_y_px=arrays["chaser_y_px"],
        chaser_phase_x_mm=arrays["chaser_x_mm"],
        chaser_phase_y_mm=arrays["chaser_y_mm"],
        chaser_quadrant_code=arrays["chaser_quadrant_code"],
        chaser_position_sample_count=arrays["chaser_position_sample_count"],
        chaser_max_drift_mm=arrays["chaser_max_drift_mm"],
        chaser_median_drift_mm=arrays["chaser_median_drift_mm"],
        median_distance_mm=arrays["median_distance_mm"],
        mean_distance_mm=arrays["mean_distance_mm"],
        occupancy_fraction=arrays["occupancy_fraction"],
        occupancy_fraction_of_epoch=arrays["occupancy_fraction_of_epoch"],
        valid_frame_count=arrays["valid_frame_count"],
        distance_valid_frame_count=arrays["distance_valid_frame_count"],
        total_frame_count=arrays["total_frame_count"],
        missing_frame_count=arrays["missing_frame_count"],
        tracking_dropout_fraction=arrays["tracking_dropout_fraction"],
        endpoint_status=endpoint_status,
        qc_warnings=tuple(qc_warnings),
        summary=summary,
        diagnostics=diagnostics,
        protocol_profile_id=(
            loaded_profile.profile_id if loaded_profile is not None else None
        ),
        protocol_profile_version=(
            loaded_profile.profile_version if loaded_profile is not None else None
        ),
        protocol_profile_sha256=(
            loaded_profile.sha256 if loaded_profile is not None else None
        ),
        post_settle_duration_s=float(post_settle_s),
    )


def render_chaser_quadrant_occupancy_png(
    result: ChaserQuadrantOccupancyResult,
    *,
    fish_xy: np.ndarray,
    fish_valid: np.ndarray,
    dpi: int = 150,
    max_points_per_phase: int = 2500,
) -> bytes:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.0), constrained_layout=True)
    axes_list = list(axes.ravel()) if hasattr(axes, "ravel") else [axes]
    colors = {obj.chaser_index: obj.raw_color_hex for obj in result.chasers}
    markers = ("o", "s", "^", "D", "P", "X", "v", "<", ">")
    for phase_idx, (ax, phase) in enumerate(zip(axes_list, result.phases)):
        slc = _phase_slice(phase, result.total_frames)
        xy = np.asarray(fish_xy[slc], dtype=np.float64)
        valid = np.asarray(fish_valid[slc], dtype=bool) & np.isfinite(xy).all(axis=1)
        xy_valid = xy[valid]
        if xy_valid.shape[0] > 0:
            step = max(1, int(math.ceil(xy_valid.shape[0] / max(1, int(max_points_per_phase)))))
            sampled = xy_valid[::step]
            ax.plot(sampled[:, 0], sampled[:, 1], color="#334155", linewidth=0.7, alpha=0.45)
            ax.scatter(sampled[:, 0], sampled[:, 1], s=4, color="#0f766e", alpha=0.35)
        for obj_idx, obj in enumerate(result.chasers):
            q_code = int(result.chaser_quadrant_code[phase_idx, obj_idx])
            if 0 <= q_code < 4:
                half_w = result.quadrant_width_px / 2.0
                half_h = result.quadrant_height_px / 2.0
                x0 = half_w if q_code in {1, 3} else 0.0
                y0 = half_h if q_code in {2, 3} else 0.0
                rect = plt.Rectangle(
                    (x0, y0),
                    half_w,
                    half_h,
                    facecolor=colors.get(obj.chaser_index, "#64748b"),
                    alpha=0.08,
                    edgecolor=colors.get(obj.chaser_index, "#64748b"),
                    linewidth=1.1,
                    linestyle="--",
                )
                ax.add_patch(rect)
            x = float(result.chaser_phase_x_px[phase_idx, obj_idx])
            y = float(result.chaser_phase_y_px[phase_idx, obj_idx])
            if math.isfinite(x) and math.isfinite(y):
                ax.scatter(
                    [x],
                    [y],
                    s=85,
                    marker=markers[obj_idx % len(markers)],
                    color=colors.get(obj.chaser_index, "#64748b"),
                    edgecolor="white",
                    linewidth=1.2,
                    label=(
                        f"chaser {obj.chaser_index}: {obj.behavior_class} "
                        f"({obj.raw_color_hex})"
                    ),
                    zorder=5,
                )
        ax.set_title(phase.phase_label.replace("_", " "))
        ax.set_xlim(0, result.quadrant_width_px)
        ax.set_ylim(result.quadrant_height_px, 0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("arena x (px)")
        ax.set_ylabel("arena y (px)")
        ax.grid(alpha=0.18)
    handles, labels = axes_list[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(f"Chaser quadrant occupancy: {result.recording_id}", fontsize=12)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=int(dpi))
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def _interactive_spec(
    result: ChaserQuadrantOccupancyResult, component_path: str
) -> dict[str, Any]:
    return {
        "schema_id": INTERACTIVE_SPEC_SCHEMA_ID,
        "schema_version": 1,
        "renderer": INTERACTIVE_RENDERER,
        "recording_id": result.recording_id,
        "component_name": result.component_name,
        "component_path": component_path,
        "source_paths": {
            "component": component_path,
            "summary": f"{component_path}/summary",
            "per_chaser_phase": f"{component_path}/per_chaser_phase",
            "chaser_phase": f"{component_path}/chaser_phase",
            "phases": f"{component_path}/phases",
            "chasers": f"{component_path}/chasers",
            "fish_centroid_arena_xy": f"{result.chaser_distance_run_path}/positions/fish_centroid_arena_xy",
            "fish_valid": f"{result.chaser_distance_run_path}/positions/fish_valid",
        },
        "summary": result.summary,
        "qc_warnings": list(result.qc_warnings),
    }


def write_chaser_quadrant_occupancy_component(
    zarr_path: Path,
    result: ChaserQuadrantOccupancyResult,
    *,
    overwrite: bool = False,
    write_png: bool = True,
    write_interactive_spec: bool = True,
    mirror_run_level_interactive_spec: bool = True,
) -> str:
    root = _open_root(zarr_path, mode="a")
    reject_unsealed_chaser_derived_publication(
        root,
        run_name=result.chaser_distance_run_name,
        run_path=result.chaser_distance_run_path,
        relative_path=f"{COMPONENT_PARENT_NAME}/{result.component_name}",
    )
    run_group = root[result.chaser_distance_run_path]
    parent = run_group.require_group(COMPONENT_PARENT_NAME)
    component_name = result.component_name
    if component_name in parent:
        if not overwrite:
            raise ValueError(
                f"Chaser quadrant occupancy component already exists: {result.chaser_distance_run_path}/{COMPONENT_PARENT_NAME}/{component_name}"
            )
        del parent[component_name]
    component = parent.create_group(component_name)
    component_path = f"{result.chaser_distance_run_path}/{COMPONENT_PARENT_NAME}/{component_name}"

    chasers = component.require_group("chasers")
    _write_array(
        chasers,
        "chaser_index",
        np.asarray([obj.chaser_index for obj in result.chasers], dtype=np.int16),
    )
    _write_array(
        chasers,
        "behavior_class_id",
        np.asarray([obj.behavior_class_id for obj in result.chasers], dtype=np.int8),
    )
    _write_array(
        chasers,
        "behavior_class_label_bytes",
        _bytes_array([obj.behavior_class for obj in result.chasers], width=32),
    )
    _write_array(
        chasers,
        "raw_color_rgba",
        np.asarray([obj.raw_color_rgba for obj in result.chasers], dtype=np.float32),
    )
    _write_array(
        chasers,
        "raw_color_hex_bytes",
        _bytes_array([obj.raw_color_hex for obj in result.chasers], width=16),
    )
    _write_array(
        chasers,
        "enable_chase",
        np.asarray([obj.enable_chase for obj in result.chasers], dtype=bool),
    )
    _write_array(
        chasers,
        "enable_random_movement",
        np.asarray([obj.enable_random_movement for obj in result.chasers], dtype=bool),
    )
    _write_array(
        chasers,
        "behavior_mode",
        np.asarray(
            [
                obj.behavior_mode if obj.behavior_mode is not None else -1
                for obj in result.chasers
            ],
            dtype=np.int16,
        ),
    )
    _write_array(
        chasers,
        "start_position_preset_bytes",
        _bytes_array([obj.start_position_preset for obj in result.chasers], width=48),
    )
    _write_array(
        chasers,
        "end_position_preset_bytes",
        _bytes_array([obj.end_position_preset for obj in result.chasers], width=48),
    )
    chasers.attrs.update(
        {
            "row_axis": "chasers",
            "behavior_class_vocabulary": {
                "0": "unknown",
                "1": "aggressive",
                "2": "random_non_chasing",
                "3": "inert",
            },
        }
    )

    phases = component.require_group("phases")
    _write_array(
        phases,
        "phase_index",
        np.asarray([phase.phase_index for phase in result.phases], dtype=np.int16),
    )
    _write_array(
        phases,
        "phase_label_bytes",
        _bytes_array([phase.phase_label for phase in result.phases], width=48),
    )
    _write_array(
        phases,
        "source_window_label_bytes",
        _bytes_array([phase.source_window_label for phase in result.phases], width=48),
    )
    _write_array(
        phases,
        "source_start_frame",
        np.asarray(
            [phase.source_start_frame for phase in result.phases], dtype=np.int64
        ),
    )
    _write_array(
        phases,
        "source_end_frame",
        np.asarray([phase.source_end_frame for phase in result.phases], dtype=np.int64),
    )
    _write_array(
        phases,
        "effective_start_frame",
        np.asarray(
            [phase.effective_start_frame for phase in result.phases], dtype=np.int64
        ),
    )
    _write_array(
        phases,
        "effective_end_frame",
        np.asarray(
            [phase.effective_end_frame for phase in result.phases], dtype=np.int64
        ),
    )
    _write_array(
        phases,
        "settle_excluded_frame_count",
        np.asarray(
            [phase.settle_excluded_frame_count for phase in result.phases],
            dtype=np.int64,
        ),
    )
    phases.attrs.update({"row_axis": "chaser_quadrant_phases"})

    chaser_phase = component.require_group("chaser_phase")
    _write_array(chaser_phase, "chaser_x_px", result.chaser_phase_x_px)
    _write_array(chaser_phase, "chaser_y_px", result.chaser_phase_y_px)
    _write_array(chaser_phase, "chaser_x_mm", result.chaser_phase_x_mm)
    _write_array(chaser_phase, "chaser_y_mm", result.chaser_phase_y_mm)
    _write_array(chaser_phase, "chaser_quadrant_code", result.chaser_quadrant_code)
    _write_array(
        chaser_phase,
        "chaser_quadrant_label_bytes",
        _bytes_array(list(QUADRANT_LABELS), width=32),
    )
    _write_array(
        chaser_phase,
        "chaser_position_sample_count",
        result.chaser_position_sample_count,
    )
    _write_array(chaser_phase, "chaser_max_drift_mm", result.chaser_max_drift_mm)
    _write_array(chaser_phase, "chaser_median_drift_mm", result.chaser_median_drift_mm)
    chaser_phase.attrs.update(
        {
            "axis_order": ["phase", "chaser"],
            "quadrant_code_labels": {
                str(index): label for index, label in enumerate(QUADRANT_LABELS)
            },
            "coordinate_frame": result.coordinate_frame,
            "coordinate_origin": result.coordinate_origin,
        }
    )

    per_chaser = component.require_group("per_chaser_phase")
    _write_array(per_chaser, "median_distance_mm", result.median_distance_mm)
    _write_array(per_chaser, "mean_distance_mm", result.mean_distance_mm)
    _write_array(per_chaser, "occupancy_fraction", result.occupancy_fraction)
    _write_array(
        per_chaser, "occupancy_fraction_of_epoch", result.occupancy_fraction_of_epoch
    )
    _write_array(per_chaser, "valid_frame_count", result.valid_frame_count)
    _write_array(
        per_chaser, "distance_valid_frame_count", result.distance_valid_frame_count
    )
    _write_array(per_chaser, "total_frame_count", result.total_frame_count)
    _write_array(per_chaser, "missing_frame_count", result.missing_frame_count)
    _write_array(
        per_chaser, "tracking_dropout_fraction", result.tracking_dropout_fraction
    )
    per_chaser.attrs.update(
        {
            "axis_order": ["phase", "chaser"],
            "occupancy_fraction_denominator": "valid fish frames in effective phase",
            "occupancy_fraction_of_epoch_denominator": "all frames in effective phase",
        }
    )

    summary_group = component.require_group("summary")
    for key, value in result.summary.items():
        if isinstance(value, str) or value is None:
            _write_array(
                summary_group,
                f"{key}_bytes",
                _bytes_array(["" if value is None else str(value)], width=128),
            )
        elif isinstance(value, (list, dict)):
            _write_array(
                summary_group,
                f"{key}_json_bytes",
                _bytes_array(
                    [json.dumps(json_attr_safe(value), sort_keys=True)], width=65536
                ),
            )
        elif isinstance(value, int):
            _write_array(summary_group, key, np.asarray([value], dtype=np.int64))
        else:
            _write_array(summary_group, key, np.asarray([np.nan if value is None else float(value)], dtype=np.float32))
    _write_array(summary_group, "endpoint_status_bytes", _bytes_array([result.endpoint_status], width=48))
    _write_array(summary_group, "diagnostics_json_bytes", _bytes_array([json.dumps(json_attr_safe(result.diagnostics), sort_keys=True)], width=4096))
    _write_array(summary_group, "qc_warnings_json_bytes", _bytes_array([json.dumps(list(result.qc_warnings), sort_keys=True)], width=4096))
    summary_group.attrs.update({"row_axis": "fish_recording", "summary": json_attr_safe(result.summary)})

    git = get_git_info(Path(__file__).resolve().parents[3])
    source_refs = {
        "source_chaser_distance_run": result.chaser_distance_run_name,
        "source_chaser_distance_path": result.chaser_distance_run_path,
        "source_stimulus_run": result.source_stimulus_run,
        "source_stimulus_path": result.source_stimulus_path,
        "source_stimulus_epoch_run": result.source_stimulus_epoch_run,
        "source_stimulus_epoch_path": result.source_stimulus_epoch_path,
    }
    parameters = {
        "post_settle_policy": "protocol_profile_parameter_or_zero",
        "post_settle_duration_s": result.post_settle_duration_s,
        "protocol_profile_id": result.protocol_profile_id,
        "protocol_profile_version": result.protocol_profile_version,
        "protocol_profile_sha256": result.protocol_profile_sha256,
        "dropout_warning_fraction": result.dropout_warning_fraction,
        "dropout_exclusion_fraction": result.dropout_exclusion_fraction,
        "static_chaser_drift_warning_mm": result.static_chaser_drift_warning_mm,
        "quadrant_bounds_source": result.quadrant_bounds_source,
    }
    attrs = {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "method_version": METHOD_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "component_name": component_name,
        "recording_id": result.recording_id,
        "row_axis": "fish_recording",
        "status": result.endpoint_status,
        "source_refs": source_refs,
        "parameters": parameters,
        "summary": result.summary,
        "qc_warnings": list(result.qc_warnings),
        "diagnostics": result.diagnostics,
        "coordinate_frame": result.coordinate_frame,
        "coordinate_origin": result.coordinate_origin,
        "x_axis_direction": "right",
        "y_axis_direction": "down",
        "quadrant_bounds_source": result.quadrant_bounds_source,
        "quadrant_width_px": result.quadrant_width_px,
        "quadrant_height_px": result.quadrant_height_px,
        "pixels_per_mm_projector": result.pixels_per_mm_projector,
        "git_commit": git.get("commit_hash"),
        "git_branch": git.get("branch"),
        "git_dirty": git.get("is_dirty"),
        "provenance": {
            "stage": "chaser_quadrant_occupancy",
            "created_by": "fisheye.analysis.chaser_quadrant_occupancy",
            "inputs": source_refs,
            "parameters": parameters,
        },
    }
    component.attrs.update(json_attr_safe(attrs))
    lineage_payload = build_run_lineage_payload(
        run_family=f"{result.chaser_distance_run_path}/{COMPONENT_PARENT_NAME}",
        analysis_schema={"schema_id": SCHEMA_ID, "schema_version": SCHEMA_VERSION, "row_axis": "fish_recording"},
        method=METHOD,
        method_version=METHOD_VERSION,
        source_refs=source_refs,
        parameters=parameters,
        code={"git_commit": git.get("commit_hash"), "git_dirty": git.get("is_dirty")},
    )
    write_run_lineage_attrs(component, lineage_payload, fingerprint_status="best_effort", overwrite=True)
    parent.attrs["latest"] = component_name
    parent.attrs["latest_complete"] = component_name

    if write_png or write_interactive_spec:
        fish_xy = np.asarray(run_group["positions"]["fish_centroid_arena_xy"][:], dtype=np.float32)
        fish_valid = np.asarray(run_group["positions"]["fish_valid"][:], dtype=bool)
    if write_png:
        png = render_chaser_quadrant_occupancy_png(
            result, fish_xy=fish_xy, fish_valid=fish_valid
        )
        write_png_visualization_artifact(
            component,
            OVERVIEW_PNG_ARTIFACT_NAME,
            png,
            description="Per-chaser quadrant occupancy across configured analysis phases.",
            created_by="fisheye.analysis.chaser_quadrant_occupancy",
            role="analysis_summary",
            source_paths={
                **source_refs,
                "component": component_path,
                "fish_centroid_arena_xy": f"{result.chaser_distance_run_path}/positions/fish_centroid_arena_xy",
                "fish_valid": f"{result.chaser_distance_run_path}/positions/fish_valid",
            },
            source_runs={"chaser_distance_run": result.chaser_distance_run_name},
            parameters=parameters,
            extra_attrs={
                "chaser_quadrant_occupancy_schema_id": SCHEMA_ID,
                "component_path": component_path,
                "canonical_artifact": True,
            },
            overwrite=True,
        )
    if write_interactive_spec:
        spec = _interactive_spec(result, component_path)
        write_interactive_plot_spec_artifact(
            component,
            INTERACTIVE_ARTIFACT_NAME,
            spec,
            description="chaser Chaser quadrant occupancy interactive plot spec.",
            created_by="fisheye.analysis.chaser_quadrant_occupancy",
            renderer=INTERACTIVE_RENDERER,
            artifact_signature=None,
            snapshot_artifact=OVERVIEW_PNG_ARTIFACT_NAME,
            source_paths=spec["source_paths"],
            source_runs={"chaser_distance_run": result.chaser_distance_run_name},
            parameters=parameters,
            extra_attrs={
                "plot_schema_id": INTERACTIVE_SPEC_SCHEMA_ID,
                "component_path": component_path,
                "summary": json_attr_safe(result.summary),
                "canonical_artifact": True,
            },
            overwrite=True,
        )
        if mirror_run_level_interactive_spec:
            mirror_spec = {
                **spec,
                "canonical_artifact_path": (
                    f"{component_path}/visualizations/{INTERACTIVE_ARTIFACT_NAME}"
                ),
                "mirror_artifact_path": (
                    f"{result.chaser_distance_run_path}/visualizations/{INTERACTIVE_ARTIFACT_NAME}"
                ),
            }
            write_interactive_plot_spec_artifact(
                run_group,
                INTERACTIVE_ARTIFACT_NAME,
                mirror_spec,
                description="chaser Chaser quadrant occupancy interactive plot spec discovery mirror.",
                created_by="fisheye.analysis.chaser_quadrant_occupancy",
                renderer=INTERACTIVE_RENDERER,
                artifact_signature=None,
                snapshot_artifact=None,
                source_paths=spec["source_paths"],
                source_runs={"chaser_distance_run": result.chaser_distance_run_name},
                parameters=parameters,
                extra_attrs={
                    "plot_schema_id": INTERACTIVE_SPEC_SCHEMA_ID,
                    "component_path": component_path,
                    "canonical_artifact_path": f"{component_path}/visualizations/{INTERACTIVE_ARTIFACT_NAME}",
                    "canonical_artifact": False,
                    "summary": json_attr_safe(result.summary),
                },
                overwrite=True,
            )
    return component_path


def _result_payload(
    result: ChaserQuadrantOccupancyResult, *, applied_path: str | None
) -> dict[str, Any]:
    return {
        "schema_id": SCHEMA_ID,
        "zarr_path": result.zarr_path,
        "recording_id": result.recording_id,
        "component_name": result.component_name,
        "applied_path": applied_path,
        "chaser_distance_run": result.chaser_distance_run_name,
        "endpoint_status": result.endpoint_status,
        "qc_warnings": list(result.qc_warnings),
        "summary": result.summary,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Analysis zarr archive.")
    parser.add_argument("--chaser-distance-run", default="latest")
    parser.add_argument("--component-name", default=DEFAULT_COMPONENT_NAME)
    parser.add_argument("--dropout-warning-fraction", type=float, default=0.20)
    parser.add_argument("--dropout-exclusion-fraction", type=float)
    parser.add_argument("--static-chaser-drift-warning-mm", type=float, default=1.0)
    parser.add_argument(
        "--protocol-profile",
        type=Path,
        help="Versioned protocol adapter profile supplying source-specific parameters.",
    )
    parser.add_argument(
        "--apply", action="store_true", help="Write the endpoint component."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite an existing component."
    )
    parser.add_argument(
        "--no-png", action="store_true", help="Skip PNG overview artifact."
    )
    parser.add_argument(
        "--no-interactive-spec",
        action="store_true",
        help="Skip interactive plot spec artifact.",
    )
    parser.add_argument(
        "--no-run-level-interactive-spec",
        action="store_true",
        help="Do not mirror the component-local interactive spec at the chaser-distance run visualizations level.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON payload.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = build_chaser_quadrant_occupancy_result(
        Path(args.zarr_path),
        chaser_distance_run=str(args.chaser_distance_run),
        component_name=str(args.component_name),
        dropout_warning_fraction=float(args.dropout_warning_fraction),
        dropout_exclusion_fraction=args.dropout_exclusion_fraction,
        static_chaser_drift_warning_mm=float(args.static_chaser_drift_warning_mm),
        protocol_profile=args.protocol_profile,
    )
    applied_path = None
    if args.apply:
        applied_path = write_chaser_quadrant_occupancy_component(
            Path(args.zarr_path),
            result,
            overwrite=bool(args.overwrite),
            write_png=not bool(args.no_png),
            write_interactive_spec=not bool(args.no_interactive_spec),
            mirror_run_level_interactive_spec=not bool(args.no_run_level_interactive_spec),
        )
    payload = _result_payload(result, applied_path=applied_path)
    if args.json:
        print(json.dumps(json_attr_safe(payload), indent=2, sort_keys=True))
    else:
        print(f"recording_id\t{result.recording_id}")
        print(f"component_name\t{result.component_name}")
        print(f"chaser_distance_run\t{result.chaser_distance_run_name}")
        print(f"endpoint_status\t{result.endpoint_status}")
        print(f"qc_warning_count\t{len(result.qc_warnings)}")
        if applied_path:
            print(f"applied_path\t{applied_path}")
        else:
            print("dry_run\ttrue")
            print("pass --apply to write the endpoint component")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
