"""GoodCopBadCop chaser-centric escape/freeze diagnostic canary."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import zarr  # noqa: E402

from fisheye.analysis.chaser_distance_runs import _bytes_array, _write_array
from fisheye.analysis.chaser_state_interpolator import load_structured_dataset
from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.plot_artifacts import write_png_visualization_artifact
from fisheye.shared.run_lineage_fingerprint import build_run_lineage_payload, write_run_lineage_attrs
from fisheye.shared.zarr_run_completion import resolve_authoritative_run_name
from fisheye.utils.system import get_git_info


SCHEMA_ID = "palette.goodcopbadcop.chaser_escape_freeze_canary.v1"
SCHEMA_VERSION = 1
METHOD = "goodcopbadcop_chaser_centric_escape_freeze_canary"
METHOD_VERSION = "1"
COMPONENT_PARENT_NAME = "chaser_escape_freeze"
DEFAULT_COMPONENT_NAME = "canary_chaser0_escape_freeze_v1"
PER_TRIAL_DIAGNOSTIC_PNG_ARTIFACT_NAME = "escape_freeze_per_trial_diagnostic_png"
FISH_CENTERED_DIAGNOSTIC_PNG_ARTIFACT_NAME = "escape_freeze_fish_centered_diagnostic_png"
SPEED_DISPLACEMENT_SCATTER_PNG_ARTIFACT_NAME = "escape_freeze_speed_displacement_scatter_png"
RESPONSE_CLASS_BAR_PNG_ARTIFACT_NAME = "escape_freeze_response_class_bar_png"
TRIAL_OUTCOME_TIMELINE_PNG_ARTIFACT_NAME = "escape_freeze_trial_outcome_timeline_png"
FISH_CENTERED_POLAR_APPROACH_PNG_ARTIFACT_NAME = "escape_freeze_fish_centered_polar_approach_png"
FISH_CENTERED_POLAR_DENSITY_PNG_ARTIFACT_NAME = "escape_freeze_fish_centered_polar_density_png"
DEFAULT_CHASER_INDEX = 0
DEFAULT_RESPONSE_WINDOW_S = 0.6
DEFAULT_FREEZE_WINDOW_S = 2.0
DEFAULT_BASELINE_WINDOW_S = 1.0
DEFAULT_TRIGGER_RADIUS_MM = 5.0
DEFAULT_TRIGGER_RADIUS_SOURCE = "default_fallback_5mm"
DEFAULT_LOW_SPEED_THRESHOLD_MM_S = 1.0
DEFAULT_HEADING_MIN_SPEED_MM_S = 0.5
DEFAULT_ESCAPE_PATH_THRESHOLD_MM = 40.0


@dataclass(frozen=True)
class EscapeFreezeCanaryResult:
    zarr_path: str
    recording_id: str
    component_name: str
    chaser_distance_run_name: str
    chaser_distance_run_path: str
    source_stimulus_run: str | None
    source_stimulus_path: str | None
    fps: float
    total_frames: int
    pixels_per_mm_projector: float
    chaser_index: int
    response_window_s: float
    freeze_window_s: float
    baseline_window_s: float
    trigger_radius_mm: float
    trigger_radius_source: str
    trigger_radius_override: bool
    escape_path_threshold_mm: float
    low_speed_threshold_mm_s: float
    heading_min_speed_mm_s: float
    trial_table: np.ndarray
    metric_table: np.ndarray
    trajectory_table: np.ndarray
    summary: dict[str, Any]
    diagnostics: dict[str, Any]
    warnings: tuple[str, ...]


def _open_root(zarr_path: Path, *, mode: str) -> zarr.Group:
    return zarr.open_group(str(zarr_path), mode=mode, use_consolidated=False)


def _attr_text(attrs: Any, *keys: str) -> str | None:
    for key in keys:
        value = attrs.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _resolve_chaser_distance_run(root: zarr.Group, run_name: str | None) -> tuple[zarr.Group, str, str]:
    parent = root.get("analysis/chaser_distance_runs")
    if parent is None:
        raise ValueError("Archive has no analysis/chaser_distance_runs group.")
    resolved = str(run_name or "latest").strip()
    if not resolved or resolved == "latest":
        resolved = resolve_authoritative_run_name(parent) or str(parent.attrs.get("latest") or "").strip()
    if not resolved or resolved not in parent:
        raise ValueError("No usable chaser-distance run found; pass --chaser-distance-run.")
    return parent[resolved], resolved, f"analysis/chaser_distance_runs/{resolved}"


def _source_stimulus_path(run_group: zarr.Group) -> tuple[str | None, str | None]:
    source_run = _attr_text(run_group.attrs, "source_stimulus_run")
    source_path = _attr_text(run_group.attrs, "source_stimulus_path")
    if source_run and not source_path:
        source_path = f"analysis/stimulus_runs/{source_run}"
    return source_run, source_path


def _field_name(array: np.ndarray, *candidates: str, required: bool = True) -> str | None:
    names = array.dtype.names or ()
    for candidate in candidates:
        if candidate in names:
            return candidate
    if required:
        raise ValueError(f"Structured array missing expected field: {candidates}")
    return None


def _load_chaser_states(root: zarr.Group, source_stimulus_path: str | None) -> np.ndarray:
    if not source_stimulus_path:
        raise ValueError("Chaser-distance run lacks source_stimulus_path/source_stimulus_run.")
    stim_group = root.get(source_stimulus_path)
    if stim_group is None:
        raise ValueError(f"Source stimulus run not found: {source_stimulus_path}")
    tracking = stim_group.get("tracking_data")
    if tracking is None or "chaser_states" not in tracking:
        raise ValueError(f"Source stimulus run lacks tracking_data/chaser_states: {source_stimulus_path}")
    chaser_states, _attrs = load_structured_dataset(tracking, "chaser_states")
    return chaser_states


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _positive_float(value: Any) -> float | None:
    out = _safe_float(value)
    return out if math.isfinite(out) and out > 0 else None


def _read_scalar_attr(group: zarr.Group, name: str, default: float) -> float:
    return _safe_float(group.attrs.get(name), default)


def _chaser_column(run_group: zarr.Group, chaser_index: int) -> int:
    if "chasers" not in run_group or "chaser_index" not in run_group["chasers"]:
        raise ValueError("Chaser-distance run lacks chasers/chaser_index.")
    indices = np.asarray(run_group["chasers/chaser_index"][:], dtype=np.int64).reshape(-1)
    matches = np.flatnonzero(indices == int(chaser_index))
    if matches.size == 0:
        raise ValueError(f"Chaser index {chaser_index} not present in chaser-distance run; available={indices.tolist()}")
    return int(matches[0])


def _dense_controller_state(
    chaser_states: np.ndarray,
    *,
    stimulus_frame_nums: np.ndarray,
    chaser_index: int,
) -> dict[str, np.ndarray]:
    dtype_names = chaser_states.dtype.names or ()
    stim_field = _field_name(chaser_states, "stimulus_frame_num", "frame_number", "stim_frame_num")
    index_field = _field_name(chaser_states, "chaser_index", required=False)
    active_field = _field_name(chaser_states, "chase_sequence_active", "is_chasing")
    trial_id_field = _field_name(chaser_states, "chase_trial_id", required=False)

    stim_to_row = {
        int(stim): int(row)
        for row, stim in enumerate(np.asarray(stimulus_frame_nums, dtype=np.int64).reshape(-1))
        if int(stim) >= 0
    }
    n = int(stimulus_frame_nums.shape[0])
    active = np.zeros(n, dtype=bool)
    trial_id = np.zeros(n, dtype=np.int64)
    is_chasing = np.zeros(n, dtype=bool)
    trial_state = np.full(n, -1, dtype=np.int16)
    loom_phase = np.full(n, -1, dtype=np.int16)
    loom_mode = np.full(n, -1, dtype=np.int16)
    chase_speed_mm_s = np.full(n, np.nan, dtype=np.float32)
    distance_to_target_mm = np.full(n, np.nan, dtype=np.float32)

    for record in chaser_states:
        if index_field is not None and int(record[index_field]) != int(chaser_index):
            continue
        frame_row = stim_to_row.get(int(record[stim_field]))
        if frame_row is None:
            continue
        is_active = bool(record[active_field])
        active[frame_row] = is_active
        is_chasing[frame_row] = bool(record["is_chasing"]) if "is_chasing" in dtype_names else is_active
        if trial_id_field is not None:
            trial_id[frame_row] = _safe_int(record[trial_id_field], 0)
        if "trial_state" in dtype_names:
            trial_state[frame_row] = _safe_int(record["trial_state"], -1)
        if "loom_phase" in dtype_names:
            loom_phase[frame_row] = _safe_int(record["loom_phase"], -1)
        if "loom_mode" in dtype_names:
            loom_mode[frame_row] = _safe_int(record["loom_mode"], -1)
        if "chase_speed_mm_per_s" in dtype_names:
            chase_speed_mm_s[frame_row] = _safe_float(record["chase_speed_mm_per_s"])
        if "distance_to_target_mm" in dtype_names:
            distance_to_target_mm[frame_row] = _safe_float(record["distance_to_target_mm"])

    return {
        "active": active,
        "trial_id": trial_id,
        "is_chasing": is_chasing,
        "trial_state": trial_state,
        "loom_phase": loom_phase,
        "loom_mode": loom_mode,
        "chase_speed_mm_s": chase_speed_mm_s,
        "distance_to_target_mm": distance_to_target_mm,
        "matched_state_count": np.asarray([int(np.count_nonzero((trial_state >= 0) | active))], dtype=np.int64),
    }


def _contiguous_true_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool).reshape(-1)
    if values.size == 0:
        return []
    padded = np.concatenate([[False], values, [False]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(start), int(stop - 1)) for start, stop in zip(changes[0::2], changes[1::2])]


def _mode_positive(values: np.ndarray, fallback: int) -> int:
    data = np.asarray(values, dtype=np.int64).reshape(-1)
    data = data[data > 0]
    if data.size == 0:
        return int(fallback)
    uniq, counts = np.unique(data, return_counts=True)
    return int(uniq[int(np.argmax(counts))])


def _median_positive(values: np.ndarray) -> float | None:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    data = data[np.isfinite(data) & (data > 0)]
    if data.size == 0:
        return None
    return float(np.median(data))


def _trigger_radius_from_chaser_states(chaser_states: np.ndarray, *, chaser_index: int) -> float | None:
    names = chaser_states.dtype.names or ()
    if "initial_distance_mm" not in names:
        return None

    rows = chaser_states
    if "chaser_index" in names:
        rows = rows[np.asarray(rows["chaser_index"], dtype=np.int64) == int(chaser_index)]
    if "chase_sequence_active" in names:
        active_rows = rows[np.asarray(rows["chase_sequence_active"], dtype=bool)]
        if active_rows.size:
            rows = active_rows
    return _median_positive(np.asarray(rows["initial_distance_mm"], dtype=np.float64))


def _coerce_protocol_json(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, Mapping) else None
    return None


def _trigger_radius_from_protocol_json(protocol: Mapping[str, Any] | None, *, chaser_index: int) -> float | None:
    if not protocol:
        return None
    steps = protocol.get("steps")
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
        return None
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        params = step.get("parameters")
        if not isinstance(params, Mapping):
            continue
        chasers = params.get("chasers")
        if not isinstance(chasers, Sequence) or isinstance(chasers, (str, bytes)):
            continue
        for idx, chaser in enumerate(chasers):
            if not isinstance(chaser, Mapping):
                continue
            configured_index = _safe_int(chaser.get("chaser_index", idx), idx)
            if configured_index != int(chaser_index):
                continue
            radius = _positive_float(chaser.get("initial_distance_mm"))
            if radius is not None:
                return radius
    return None


def _resolve_trigger_radius(
    root: zarr.Group,
    *,
    source_stimulus_path: str | None,
    chaser_states: np.ndarray,
    chaser_index: int,
    trigger_radius_mm: float | None,
) -> tuple[float, str, bool, tuple[str, ...]]:
    override = _positive_float(trigger_radius_mm)
    if override is not None:
        return override, "cli_override", True, ()

    from_states = _trigger_radius_from_chaser_states(chaser_states, chaser_index=chaser_index)
    if from_states is not None:
        return from_states, "chaser_states.initial_distance_mm", False, ()

    if source_stimulus_path:
        stim_group = root.get(source_stimulus_path)
        if stim_group is not None:
            from_protocol = _trigger_radius_from_protocol_json(
                _coerce_protocol_json(stim_group.attrs.get("protocol_json")),
                chaser_index=chaser_index,
            )
            if from_protocol is not None:
                return from_protocol, "protocol_json.chasers.initial_distance_mm", False, ()

    return (
        float(DEFAULT_TRIGGER_RADIUS_MM),
        DEFAULT_TRIGGER_RADIUS_SOURCE,
        False,
        ("trigger_radius_metadata_absent_using_default_fallback",),
    )


def _select_trial_trigger(
    distance_mm: np.ndarray,
    frames: np.ndarray,
    *,
    trigger_radius_mm: float,
) -> tuple[int, int, str]:
    frame_ids = np.asarray(frames, dtype=np.int64).reshape(-1)
    if frame_ids.size == 0:
        return -1, -1, "missing_trial_frames"

    start = int(frame_ids[0])
    distances = np.asarray(distance_mm, dtype=np.float64).reshape(-1)
    in_bounds = frame_ids[(frame_ids >= 0) & (frame_ids < distances.shape[0])]
    if in_bounds.size == 0:
        return start, -1, "bout_onset_no_valid_distance"

    finite = np.isfinite(distances[in_bounds])
    if not np.any(finite):
        return start, -1, "bout_onset_no_valid_distance"

    finite_frames = in_bounds[finite]
    finite_distances = distances[finite_frames]
    first_finite_frame = int(finite_frames[0])
    if float(finite_distances[0]) <= float(trigger_radius_mm):
        source = "bout_onset_already_inside_radius" if first_finite_frame == start else "first_valid_already_inside_radius"
        return first_finite_frame, first_finite_frame, source

    finite_prox = finite_distances <= float(trigger_radius_mm)
    if np.any(finite_prox):
        trigger_proximity = int(finite_frames[np.flatnonzero(finite_prox)[0]])
        return trigger_proximity, trigger_proximity, "proximity"
    return start, -1, "bout_onset_no_proximity"


def _compute_fish_speed_mm_s(fish_xy_px: np.ndarray, fish_valid: np.ndarray, ppm: float, fps: float) -> np.ndarray:
    fish = np.asarray(fish_xy_px, dtype=np.float64)
    valid = np.asarray(fish_valid, dtype=bool)
    speed = np.full(fish.shape[0], np.nan, dtype=np.float32)
    if fish.shape[0] < 2 or ppm <= 0 or fps <= 0:
        return speed
    delta = np.diff(fish, axis=0)
    finite = valid[1:] & valid[:-1] & np.isfinite(delta).all(axis=1)
    dist_mm = np.linalg.norm(delta, axis=1) / float(ppm)
    speed[1:][finite] = (dist_mm[finite] * float(fps)).astype(np.float32)
    return speed


def _heading_angles_from_chaser(
    chaser_xy_px: np.ndarray,
    chaser_valid: np.ndarray,
    *,
    ppm: float,
    fps: float,
    min_speed_mm_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    chaser = np.asarray(chaser_xy_px, dtype=np.float64)
    valid = np.asarray(chaser_valid, dtype=bool).reshape(-1)
    n = chaser.shape[0]
    heading = np.full(n, np.nan, dtype=np.float64)
    speed = np.full(n, np.nan, dtype=np.float64)
    raw_good = np.zeros(n, dtype=bool)
    if n < 2 or ppm <= 0 or fps <= 0:
        return heading.astype(np.float32), np.ones(n, dtype=bool), speed.astype(np.float32)

    delta_px = np.diff(chaser, axis=0)
    delta_math_mm = np.column_stack([delta_px[:, 0] / float(ppm), -delta_px[:, 1] / float(ppm)])
    step_speed = np.linalg.norm(delta_math_mm, axis=1) * float(fps)
    good = valid[1:] & valid[:-1] & np.isfinite(delta_math_mm).all(axis=1) & (step_speed >= float(min_speed_mm_s))
    heading[1:][good] = np.arctan2(delta_math_mm[good, 1], delta_math_mm[good, 0])
    speed[1:] = step_speed
    raw_good[1:] = good

    good_indices = np.flatnonzero(raw_good)
    held = np.ones(n, dtype=bool)
    if good_indices.size == 0:
        heading[:] = math.pi / 2.0
        return heading.astype(np.float32), held, speed.astype(np.float32)

    last = float(heading[good_indices[0]])
    heading[: good_indices[0] + 1] = last
    held[: good_indices[0] + 1] = True
    for idx in range(good_indices[0] + 1, n):
        if raw_good[idx]:
            last = float(heading[idx])
            held[idx] = False
        else:
            heading[idx] = last
            held[idx] = True
    return heading.astype(np.float32), held, speed.astype(np.float32)


def chaser_frame_transform(
    fish_xy_px: np.ndarray,
    chaser_xy_px: np.ndarray,
    heading_rad: np.ndarray,
    *,
    pixels_per_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return fish coordinates in a chaser-centric frame.

    The source arena frame is x-right/y-down. The returned frame is x-right/y-up
    and rotated so the chaser heading points along +y.
    """

    fish = np.asarray(fish_xy_px, dtype=np.float64)
    chaser = np.asarray(chaser_xy_px, dtype=np.float64)
    heading = np.asarray(heading_rad, dtype=np.float64).reshape(-1)
    if fish.shape != chaser.shape or fish.ndim != 2 or fish.shape[1] != 2:
        raise ValueError("fish_xy_px and chaser_xy_px must have matching shape (n, 2).")
    if heading.shape[0] != fish.shape[0]:
        raise ValueError("heading_rad must have one value per frame.")
    ppm = float(pixels_per_mm)
    if not math.isfinite(ppm) or ppm <= 0:
        raise ValueError("pixels_per_mm must be positive.")

    delta = np.column_stack([(fish[:, 0] - chaser[:, 0]) / ppm, -(fish[:, 1] - chaser[:, 1]) / ppm])
    alpha = (math.pi / 2.0) - heading
    cos_a = np.cos(alpha)
    sin_a = np.sin(alpha)
    x_prime = cos_a * delta[:, 0] - sin_a * delta[:, 1]
    y_prime = sin_a * delta[:, 0] + cos_a * delta[:, 1]
    radius = np.sqrt(x_prime * x_prime + y_prime * y_prime)
    bearing_deg = np.degrees(np.arctan2(x_prime, y_prime))
    return np.column_stack([x_prime, y_prime]).astype(np.float32), radius.astype(np.float32), bearing_deg.astype(np.float32)


def _finite_mean(values: np.ndarray) -> float:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = data[np.isfinite(data)]
    return float(np.mean(finite)) if finite.size else np.nan


def _finite_max(values: np.ndarray) -> float:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = data[np.isfinite(data)]
    return float(np.max(finite)) if finite.size else np.nan


def _longest_true_run_seconds(mask: np.ndarray, fps: float) -> float:
    segments = _contiguous_true_segments(np.asarray(mask, dtype=bool))
    if not segments or fps <= 0:
        return 0.0
    return float(max(end - start + 1 for start, end in segments)) / float(fps)


def _path_length_mm(fish_xy_px: np.ndarray, valid: np.ndarray, ppm: float) -> float:
    fish = np.asarray(fish_xy_px, dtype=np.float64)
    ok = np.asarray(valid, dtype=bool)
    if fish.shape[0] < 2 or ppm <= 0:
        return np.nan
    delta = np.diff(fish, axis=0)
    finite = ok[1:] & ok[:-1] & np.isfinite(delta).all(axis=1)
    if not np.any(finite):
        return np.nan
    return float(np.sum(np.linalg.norm(delta[finite], axis=1) / float(ppm)))


def _net_displacement_mm(fish_xy_px: np.ndarray, valid: np.ndarray, ppm: float) -> float:
    fish = np.asarray(fish_xy_px, dtype=np.float64)
    ok = np.asarray(valid, dtype=bool).reshape(-1)
    if fish.shape[0] == 0 or fish.shape[0] != ok.shape[0] or ppm <= 0:
        return np.nan
    finite = ok & np.isfinite(fish).all(axis=1)
    indices = np.flatnonzero(finite)
    if indices.size < 2:
        return np.nan
    return float(np.linalg.norm(fish[indices[-1]] - fish[indices[0]]) / float(ppm))


def _cumulative_abs_delta(values: np.ndarray, valid: np.ndarray | None = None) -> float:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    if valid is None:
        finite = np.isfinite(data)
    else:
        finite = np.asarray(valid, dtype=bool).reshape(-1) & np.isfinite(data)
    indices = np.flatnonzero(finite)
    if indices.size < 2:
        return np.nan
    consecutive = np.diff(indices) == 1
    deltas = np.diff(data[indices])[consecutive]
    return float(np.sum(np.abs(deltas))) if deltas.size else 0.0


def _finite_fraction_less_equal(values: np.ndarray, threshold: float, valid: np.ndarray | None = None) -> float:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    if valid is None:
        finite = np.isfinite(data)
    else:
        finite = np.asarray(valid, dtype=bool).reshape(-1) & np.isfinite(data)
    if not np.any(finite):
        return np.nan
    return float(np.mean(data[finite] <= float(threshold)))


def _classify_escape_attempt_by_path(path_length_mm: float, *, threshold_mm: float) -> tuple[bool, str]:
    path = _safe_float(path_length_mm)
    threshold = _safe_float(threshold_mm, DEFAULT_ESCAPE_PATH_THRESHOLD_MM)
    if not math.isfinite(path) or not math.isfinite(threshold) or threshold <= 0:
        return False, "unclassified"
    if path >= threshold:
        return True, "escape_attempt"
    return False, "not_escape"


def _empty_trial_table() -> np.ndarray:
    return np.zeros(0, dtype=_trial_dtype())


def _trial_dtype() -> np.dtype:
    return np.dtype(
        [
            ("trial_id", np.int64),
            ("trial_ordinal", np.int32),
            ("chaser_index", np.int16),
            ("start_frame", np.int64),
            ("end_frame", np.int64),
            ("start_time_s", np.float64),
            ("end_time_s", np.float64),
            ("duration_s", np.float64),
            ("trigger_frame", np.int64),
            ("trigger_frame_proximity", np.int64),
            ("trigger_frame_bout_onset", np.int64),
            ("trigger_source", "S64"),
            ("trigger_radius_mm", np.float64),
            ("trigger_radius_source", "S64"),
            ("trigger_distance_mm", np.float64),
            ("response_start_frame", np.int64),
            ("response_end_frame", np.int64),
            ("freeze_start_frame", np.int64),
            ("freeze_end_frame", np.int64),
            ("baseline_start_frame", np.int64),
            ("baseline_end_frame", np.int64),
            ("valid_response_fraction", np.float64),
            ("valid_freeze_fraction", np.float64),
            ("controller_active_frame_count", np.int64),
        ]
    )


def _metric_dtype() -> np.dtype:
    return np.dtype(
        [
            ("trial_id", np.int64),
            ("trial_ordinal", np.int32),
            ("baseline_mean_speed_mm_s", np.float64),
            ("response_mean_speed_mm_s", np.float64),
            ("response_max_windowed_speed_mm_s", np.float64),
            ("net_displacement_mm", np.float64),
            ("radial_excursion_mm", np.float64),
            ("path_length_mm", np.float64),
            ("escape_bearing_deg", np.float64),
            ("freeze_low_speed_fraction", np.float64),
            ("longest_low_speed_run_s", np.float64),
            ("full_trial_fish_path_length_mm", np.float64),
            ("full_trial_fish_net_displacement_mm", np.float64),
            ("full_trial_mean_speed_mm_s", np.float64),
            ("full_trial_max_speed_mm_s", np.float64),
            ("full_trial_low_speed_fraction", np.float64),
            ("full_trial_distance_cumulative_abs_delta_mm", np.float64),
            ("full_trial_distance_range_mm", np.float64),
            ("full_trial_near_contact_fraction", np.float64),
            ("escape_path_threshold_mm", np.float64),
            ("escape_attempt", np.bool_),
            ("candidate_response_class", "S32"),
            ("response_valid_frame_count", np.int64),
            ("freeze_valid_frame_count", np.int64),
            ("full_trial_valid_frame_count", np.int64),
            ("baseline_valid_speed_count", np.int64),
            ("tracking_dropout_response_fraction", np.float64),
            ("tracking_dropout_freeze_fraction", np.float64),
            ("tracking_dropout_full_trial_fraction", np.float64),
            ("classification_locked", np.bool_),
        ]
    )


def _trajectory_dtype() -> np.dtype:
    return np.dtype(
        [
            ("trial_id", np.int64),
            ("trial_ordinal", np.int32),
            ("relative_frame", np.int32),
            ("camera_frame_id", np.int64),
            ("time_from_trigger_s", np.float64),
            ("fish_x_chaser_frame_mm", np.float64),
            ("fish_y_chaser_frame_mm", np.float64),
            ("chaser_x_fish_centered_mm", np.float64),
            ("chaser_y_fish_centered_mm", np.float64),
            ("distance_mm", np.float64),
            ("bearing_deg", np.float64),
            ("fish_speed_mm_s", np.float64),
            ("chaser_heading_deg", np.float64),
            ("chaser_speed_mm_s", np.float64),
            ("heading_held", np.bool_),
            ("valid", np.bool_),
        ]
    )


def build_escape_freeze_canary_result(
    zarr_path: Path,
    *,
    chaser_distance_run: str | None = "latest",
    component_name: str = DEFAULT_COMPONENT_NAME,
    chaser_index: int = DEFAULT_CHASER_INDEX,
    response_window_s: float = DEFAULT_RESPONSE_WINDOW_S,
    freeze_window_s: float = DEFAULT_FREEZE_WINDOW_S,
    baseline_window_s: float = DEFAULT_BASELINE_WINDOW_S,
    trigger_radius_mm: float | None = None,
    escape_path_threshold_mm: float = DEFAULT_ESCAPE_PATH_THRESHOLD_MM,
    low_speed_threshold_mm_s: float = DEFAULT_LOW_SPEED_THRESHOLD_MM_S,
    heading_min_speed_mm_s: float = DEFAULT_HEADING_MIN_SPEED_MM_S,
) -> EscapeFreezeCanaryResult:
    root = _open_root(zarr_path, mode="r")
    run_group, run_name, run_path = _resolve_chaser_distance_run(root, chaser_distance_run)
    source_stimulus_run, source_stimulus_path = _source_stimulus_path(run_group)
    chaser_states = _load_chaser_states(root, source_stimulus_path)
    resolved_trigger_radius_mm, trigger_radius_source, trigger_radius_override, radius_warnings = _resolve_trigger_radius(
        root,
        source_stimulus_path=source_stimulus_path,
        chaser_states=chaser_states,
        chaser_index=int(chaser_index),
        trigger_radius_mm=trigger_radius_mm,
    )

    fps = _read_scalar_attr(run_group, "fps", 100.0)
    ppm = _read_scalar_attr(run_group, "pixels_per_mm_projector", math.nan)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Chaser-distance run lacks a valid fps attr.")
    if not math.isfinite(ppm) or ppm <= 0:
        raise ValueError("Chaser-distance run lacks a valid pixels_per_mm_projector attr.")
    total_frames = int(_read_scalar_attr(run_group, "total_frames", 0.0))
    fish_xy = np.asarray(run_group["positions/fish_centroid_arena_xy"][:], dtype=np.float32)
    fish_valid = np.asarray(run_group["positions/fish_valid"][:], dtype=bool)
    chaser_col = _chaser_column(run_group, int(chaser_index))
    chaser_xy = np.asarray(run_group["positions/chaser_arena_xy"][:, chaser_col, :], dtype=np.float32)
    chaser_valid = np.asarray(run_group["positions/chaser_valid"][:, chaser_col], dtype=bool)
    distance_mm = np.asarray(run_group["distances/distance_mm"][:, chaser_col], dtype=np.float32)
    camera_frame_ids = np.asarray(run_group["frames/camera_frame_id"][:], dtype=np.int64)
    stimulus_frame_nums = np.asarray(run_group["frames/stimulus_frame_num"][:], dtype=np.int64)
    if total_frames <= 0:
        total_frames = int(fish_xy.shape[0])
    if fish_xy.shape[0] != total_frames:
        total_frames = int(fish_xy.shape[0])

    controller = _dense_controller_state(
        chaser_states,
        stimulus_frame_nums=stimulus_frame_nums,
        chaser_index=int(chaser_index),
    )
    active = np.asarray(controller["active"], dtype=bool)
    trial_id_by_frame = np.asarray(controller["trial_id"], dtype=np.int64)

    fish_speed = _compute_fish_speed_mm_s(fish_xy, fish_valid, ppm, fps)
    heading_rad, heading_held, chaser_speed_from_xy = _heading_angles_from_chaser(
        chaser_xy,
        chaser_valid,
        ppm=ppm,
        fps=fps,
        min_speed_mm_s=float(heading_min_speed_mm_s),
    )
    chaser_frame_xy, chaser_frame_radius, chaser_frame_bearing = chaser_frame_transform(
        fish_xy,
        chaser_xy,
        heading_rad,
        pixels_per_mm=ppm,
    )

    response_frames = max(1, int(round(float(response_window_s) * float(fps))))
    freeze_frames = max(1, int(round(float(freeze_window_s) * float(fps))))
    baseline_frames = max(1, int(round(float(baseline_window_s) * float(fps))))

    segments = _contiguous_true_segments(active)
    trial_table = np.zeros(len(segments), dtype=_trial_dtype())
    metric_table = np.zeros(len(segments), dtype=_metric_dtype())
    for name in trial_table.dtype.names or ():
        if trial_table.dtype[name].kind == "f":
            trial_table[name] = np.nan
    for name in metric_table.dtype.names or ():
        if metric_table.dtype[name].kind == "f":
            metric_table[name] = np.nan

    trajectory_rows: list[np.ndarray] = []
    warnings: list[str] = list(radius_warnings)
    if not segments:
        warnings.append("no_active_chase_trials_for_selected_chaser")

    for row_idx, (start, end) in enumerate(segments):
        frames = np.arange(start, end + 1, dtype=np.int64)
        trial_id = _mode_positive(trial_id_by_frame[frames], fallback=row_idx + 1)
        trigger, trigger_proximity, trigger_source = _select_trial_trigger(
            distance_mm,
            frames,
            trigger_radius_mm=float(resolved_trigger_radius_mm),
        )
        trigger_dist = float(distance_mm[trigger]) if 0 <= trigger < distance_mm.shape[0] else np.nan

        response_start = int(trigger)
        response_end = min(int(end), int(trigger) + response_frames - 1)
        freeze_start = int(trigger)
        freeze_end = min(int(end), int(trigger) + freeze_frames - 1)
        baseline_start = max(0, int(trigger) - baseline_frames)
        baseline_end = max(-1, int(trigger) - 1)

        response_slice = np.arange(response_start, response_end + 1, dtype=np.int64) if response_end >= response_start else np.zeros(0, dtype=np.int64)
        freeze_slice = np.arange(freeze_start, freeze_end + 1, dtype=np.int64) if freeze_end >= freeze_start else np.zeros(0, dtype=np.int64)
        baseline_slice = np.arange(baseline_start, baseline_end + 1, dtype=np.int64) if baseline_end >= baseline_start else np.zeros(0, dtype=np.int64)

        valid_response = fish_valid[response_slice] & chaser_valid[response_slice] if response_slice.size else np.zeros(0, dtype=bool)
        valid_freeze = fish_valid[freeze_slice] & chaser_valid[freeze_slice] if freeze_slice.size else np.zeros(0, dtype=bool)
        valid_full_trial = fish_valid[frames] & chaser_valid[frames] & np.isfinite(distance_mm[frames]) if frames.size else np.zeros(0, dtype=bool)
        baseline_speed_values = fish_speed[baseline_slice] if baseline_slice.size else np.asarray([], dtype=np.float32)
        response_speed_values = fish_speed[response_slice] if response_slice.size else np.asarray([], dtype=np.float32)
        freeze_speed_values = fish_speed[freeze_slice] if freeze_slice.size else np.asarray([], dtype=np.float32)
        full_trial_speed_values = fish_speed[frames] if frames.size else np.asarray([], dtype=np.float32)
        full_trial_distance_values = distance_mm[frames] if frames.size else np.asarray([], dtype=np.float32)

        response_mean = _finite_mean(response_speed_values[valid_response] if valid_response.size else np.asarray([]))
        response_max = _finite_max(response_speed_values[valid_response] if valid_response.size else np.asarray([]))
        baseline_mean = _finite_mean(baseline_speed_values[np.isfinite(baseline_speed_values)])
        baseline_count = int(np.count_nonzero(np.isfinite(baseline_speed_values)))

        if response_slice.size and 0 <= trigger < chaser_frame_xy.shape[0]:
            radii = chaser_frame_radius[response_slice]
            finite_radius = np.isfinite(radii) & valid_response
            if np.any(finite_radius):
                local = int(np.nanargmax(np.where(finite_radius, radii, np.nan)))
                max_frame = int(response_slice[local])
                trigger_xy = chaser_frame_xy[trigger]
                max_xy = chaser_frame_xy[max_frame]
                net_vec = max_xy - trigger_xy
                net_displacement = float(np.linalg.norm(net_vec))
                radial_excursion = float(chaser_frame_radius[max_frame] - chaser_frame_radius[trigger])
                escape_bearing = float(np.degrees(np.arctan2(float(net_vec[0]), float(net_vec[1]))))
            else:
                net_displacement = np.nan
                radial_excursion = np.nan
                escape_bearing = np.nan
        else:
            net_displacement = np.nan
            radial_excursion = np.nan
            escape_bearing = np.nan

        path_length = _path_length_mm(fish_xy[response_slice], valid_response, ppm) if response_slice.size else np.nan
        low_speed = np.isfinite(freeze_speed_values) & (freeze_speed_values <= float(low_speed_threshold_mm_s))
        if valid_freeze.size:
            low_speed &= valid_freeze
        freeze_valid_count = int(np.count_nonzero(valid_freeze))
        response_valid_count = int(np.count_nonzero(valid_response))
        full_trial_valid_count = int(np.count_nonzero(valid_full_trial))
        low_fraction = float(np.count_nonzero(low_speed)) / float(freeze_valid_count) if freeze_valid_count > 0 else np.nan
        longest_low = _longest_true_run_seconds(low_speed, fps)
        full_trial_path_length = _path_length_mm(fish_xy[frames], valid_full_trial, ppm) if frames.size else np.nan
        full_trial_net_displacement = _net_displacement_mm(fish_xy[frames], valid_full_trial, ppm) if frames.size else np.nan
        full_trial_mean_speed = _finite_mean(
            full_trial_speed_values[valid_full_trial] if valid_full_trial.size else np.asarray([])
        )
        full_trial_max_speed = _finite_max(
            full_trial_speed_values[valid_full_trial] if valid_full_trial.size else np.asarray([])
        )
        full_trial_low_speed_fraction = _finite_fraction_less_equal(
            full_trial_speed_values,
            float(low_speed_threshold_mm_s),
            valid_full_trial,
        )
        full_trial_distance_cumulative_abs_delta = _cumulative_abs_delta(
            full_trial_distance_values,
            valid_full_trial,
        )
        finite_full_distance = np.asarray(full_trial_distance_values, dtype=np.float64)[valid_full_trial]
        finite_full_distance = finite_full_distance[np.isfinite(finite_full_distance)]
        full_trial_distance_range = (
            float(np.max(finite_full_distance) - np.min(finite_full_distance))
            if finite_full_distance.size
            else np.nan
        )
        full_trial_near_contact_fraction = _finite_fraction_less_equal(
            full_trial_distance_values,
            3.5,
            valid_full_trial,
        )
        escape_attempt, candidate_response_class = _classify_escape_attempt_by_path(
            full_trial_path_length,
            threshold_mm=float(escape_path_threshold_mm),
        )

        trial_table[row_idx] = (
            int(trial_id),
            int(row_idx + 1),
            int(chaser_index),
            int(start),
            int(end),
            float(start) / float(fps),
            float(end + 1) / float(fps),
            float(end - start + 1) / float(fps),
            int(trigger),
            int(trigger_proximity),
            int(start),
            trigger_source.encode("utf-8")[:63],
            float(resolved_trigger_radius_mm),
            str(trigger_radius_source).encode("utf-8")[:63],
            trigger_dist,
            int(response_start),
            int(response_end),
            int(freeze_start),
            int(freeze_end),
            int(baseline_start),
            int(baseline_end),
            float(response_valid_count) / float(response_slice.size) if response_slice.size else np.nan,
            float(freeze_valid_count) / float(freeze_slice.size) if freeze_slice.size else np.nan,
            int(end - start + 1),
        )
        metric_table[row_idx] = (
            int(trial_id),
            int(row_idx + 1),
            baseline_mean,
            response_mean,
            response_max,
            net_displacement,
            radial_excursion,
            path_length,
            escape_bearing,
            low_fraction,
            longest_low,
            full_trial_path_length,
            full_trial_net_displacement,
            full_trial_mean_speed,
            full_trial_max_speed,
            full_trial_low_speed_fraction,
            full_trial_distance_cumulative_abs_delta,
            full_trial_distance_range,
            full_trial_near_contact_fraction,
            float(escape_path_threshold_mm),
            bool(escape_attempt),
            str(candidate_response_class).encode("utf-8")[:31],
            int(response_valid_count),
            int(freeze_valid_count),
            int(full_trial_valid_count),
            int(baseline_count),
            1.0 - (float(response_valid_count) / float(response_slice.size)) if response_slice.size else np.nan,
            1.0 - (float(freeze_valid_count) / float(freeze_slice.size)) if freeze_slice.size else np.nan,
            1.0 - (float(full_trial_valid_count) / float(frames.size)) if frames.size else np.nan,
            False,
        )

        traj = np.zeros(frames.shape[0], dtype=_trajectory_dtype())
        for name in traj.dtype.names or ():
            if traj.dtype[name].kind == "f":
                traj[name] = np.nan
        traj["trial_id"] = int(trial_id)
        traj["trial_ordinal"] = int(row_idx + 1)
        traj["relative_frame"] = (frames - int(trigger)).astype(np.int32)
        traj["camera_frame_id"] = camera_frame_ids[frames]
        traj["time_from_trigger_s"] = (frames.astype(np.float64) - float(trigger)) / float(fps)
        traj["fish_x_chaser_frame_mm"] = chaser_frame_xy[frames, 0].astype(np.float64)
        traj["fish_y_chaser_frame_mm"] = chaser_frame_xy[frames, 1].astype(np.float64)
        traj["chaser_x_fish_centered_mm"] = ((chaser_xy[frames, 0] - fish_xy[frames, 0]) / float(ppm)).astype(np.float64)
        traj["chaser_y_fish_centered_mm"] = (-(chaser_xy[frames, 1] - fish_xy[frames, 1]) / float(ppm)).astype(np.float64)
        traj["distance_mm"] = distance_mm[frames].astype(np.float64)
        traj["bearing_deg"] = chaser_frame_bearing[frames].astype(np.float64)
        traj["fish_speed_mm_s"] = fish_speed[frames].astype(np.float64)
        traj["chaser_heading_deg"] = np.degrees(heading_rad[frames]).astype(np.float64)
        traj["chaser_speed_mm_s"] = chaser_speed_from_xy[frames].astype(np.float64)
        traj["heading_held"] = heading_held[frames]
        traj["valid"] = fish_valid[frames] & chaser_valid[frames] & np.isfinite(distance_mm[frames])
        trajectory_rows.append(traj)

    trajectory_table = (
        np.concatenate(trajectory_rows).astype(_trajectory_dtype(), copy=False)
        if trajectory_rows
        else np.zeros(0, dtype=_trajectory_dtype())
    )
    recording_id = _attr_text(root.attrs, "recording_id", "recording_name") or Path(zarr_path).stem
    escape_attempt_count = int(np.count_nonzero(metric_table["escape_attempt"])) if metric_table.size else 0
    classified_count = int(
        np.count_nonzero(
            np.asarray(
                [value.decode("utf-8", errors="ignore").strip("\x00") for value in metric_table["candidate_response_class"]],
                dtype=object,
            )
            != "unclassified"
        )
    ) if metric_table.size else 0
    summary = {
        "trial_count": int(trial_table.shape[0]),
        "chaser_index": int(chaser_index),
        "classification_locked": False,
        "candidate_labels_available": True,
        "candidate_classifier": "full_trial_fish_path_threshold",
        "escape_path_threshold_mm": float(escape_path_threshold_mm),
        "escape_attempt_count": escape_attempt_count,
        "not_escape_count": max(0, classified_count - escape_attempt_count),
        "escape_attempt_fraction": (float(escape_attempt_count) / float(classified_count)) if classified_count else np.nan,
        "response_window_s": float(response_window_s),
        "freeze_window_s": float(freeze_window_s),
        "baseline_window_s": float(baseline_window_s),
        "trigger_radius_mm": float(resolved_trigger_radius_mm),
        "trigger_radius_source": str(trigger_radius_source),
        "trigger_radius_override": bool(trigger_radius_override),
        "mean_response_speed_mm_s": _finite_mean(metric_table["response_mean_speed_mm_s"]) if metric_table.size else np.nan,
        "median_net_displacement_mm": float(np.nanmedian(metric_table["net_displacement_mm"])) if metric_table.size else np.nan,
        "median_radial_excursion_mm": float(np.nanmedian(metric_table["radial_excursion_mm"])) if metric_table.size else np.nan,
        "mean_freeze_low_speed_fraction": _finite_mean(metric_table["freeze_low_speed_fraction"]) if metric_table.size else np.nan,
        "benign_active_pursuit_available": False,
        "interpretation": "diagnostic_us_validation_only_active_pursuit_chaser0_current_cohort",
    }
    diagnostics = {
        "controller_active_frame_count": int(np.count_nonzero(active)),
        "matched_controller_state_count": int(controller["matched_state_count"][0]),
        "heading_held_frame_count": int(np.count_nonzero(heading_held)),
        "heading_min_speed_mm_s": float(heading_min_speed_mm_s),
        "escape_path_threshold_mm": float(escape_path_threshold_mm),
        "candidate_classifier": "full_trial_fish_path_threshold",
        "trigger_radius_source": str(trigger_radius_source),
        "trigger_radius_override": bool(trigger_radius_override),
        "measurement_ceiling": "100 FPS; no C-start kinematic signature or sub-30 ms latency claims; windowed speed only",
        "trial_definition": "contiguous chase_sequence_active true segments; chase_trial_id recorded as trial_id when available",
    }
    return EscapeFreezeCanaryResult(
        zarr_path=str(zarr_path),
        recording_id=str(recording_id),
        component_name=str(component_name),
        chaser_distance_run_name=run_name,
        chaser_distance_run_path=run_path,
        source_stimulus_run=source_stimulus_run,
        source_stimulus_path=source_stimulus_path,
        fps=float(fps),
        total_frames=int(total_frames),
        pixels_per_mm_projector=float(ppm),
        chaser_index=int(chaser_index),
        response_window_s=float(response_window_s),
        freeze_window_s=float(freeze_window_s),
        baseline_window_s=float(baseline_window_s),
        trigger_radius_mm=float(resolved_trigger_radius_mm),
        trigger_radius_source=str(trigger_radius_source),
        trigger_radius_override=bool(trigger_radius_override),
        escape_path_threshold_mm=float(escape_path_threshold_mm),
        low_speed_threshold_mm_s=float(low_speed_threshold_mm_s),
        heading_min_speed_mm_s=float(heading_min_speed_mm_s),
        trial_table=trial_table,
        metric_table=metric_table,
        trajectory_table=trajectory_table,
        summary=json_attr_safe(summary),
        diagnostics=json_attr_safe(diagnostics),
        warnings=tuple(warnings),
    )


def render_per_trial_diagnostic_png(result: EscapeFreezeCanaryResult, *, dpi: int = 150) -> bytes:
    trials = result.trial_table
    traj = result.trajectory_table
    n_trials = int(trials.shape[0])
    if n_trials == 0:
        fig, ax = plt.subplots(figsize=(7, 3), constrained_layout=True)
        ax.text(0.5, 0.5, "No active chase trials", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        fig, axes = plt.subplots(
            n_trials,
            2,
            figsize=(10.5, max(2.2, 1.8 * n_trials)),
            constrained_layout=True,
            squeeze=False,
        )
        finite_xy = traj[np.asarray(traj["valid"], dtype=bool)]
        if finite_xy.size:
            lim = float(np.nanpercentile(np.abs(np.concatenate([finite_xy["fish_x_chaser_frame_mm"], finite_xy["fish_y_chaser_frame_mm"]])), 98))
            lim = max(5.0, min(80.0, lim * 1.15))
        else:
            lim = 20.0
        rings = [5.0, 10.0, 20.0, 40.0]
        for row, trial in enumerate(trials):
            trial_id = int(trial["trial_id"])
            rows = traj[traj["trial_id"] == trial_id]
            class_label = ""
            if row < result.metric_table.shape[0] and "candidate_response_class" in (result.metric_table.dtype.names or ()):
                class_label = result.metric_table["candidate_response_class"][row].decode("utf-8", errors="ignore").strip("\x00")
            ax_xy = axes[row, 0]
            ax_d = axes[row, 1]
            for ring in rings:
                if ring <= lim:
                    ax_xy.add_patch(plt.Circle((0.0, 0.0), ring, fill=False, color="#cbd5e1", linewidth=0.7))
            ax_xy.scatter([0.0], [0.0], marker="^", s=60, color="#dc2626", label="chaser")
            valid = np.asarray(rows["valid"], dtype=bool)
            if rows.size:
                colors = rows["time_from_trigger_s"].astype(np.float64)
                ax_xy.plot(rows["fish_x_chaser_frame_mm"], rows["fish_y_chaser_frame_mm"], color="#2563eb", linewidth=1.1, alpha=0.85)
                if np.any(valid):
                    first = int(np.flatnonzero(valid)[0])
                    last = int(np.flatnonzero(valid)[-1])
                    ax_xy.scatter(
                        [rows["fish_x_chaser_frame_mm"][first]],
                        [rows["fish_y_chaser_frame_mm"][first]],
                        marker="o",
                        s=30,
                        facecolor="white",
                        edgecolor="#0f172a",
                        linewidth=1.0,
                    )
                    ax_xy.scatter(
                        [rows["fish_x_chaser_frame_mm"][last]],
                        [rows["fish_y_chaser_frame_mm"][last]],
                        marker="o",
                        s=32,
                        color="#0f172a",
                    )
                ax_d.plot(rows["time_from_trigger_s"], rows["distance_mm"], color="#334155", linewidth=1.2)
            ax_xy.axhline(0, color="#e2e8f0", linewidth=0.7)
            ax_xy.axvline(0, color="#e2e8f0", linewidth=0.7)
            ax_xy.set_xlim(-lim, lim)
            ax_xy.set_ylim(-lim, lim)
            ax_xy.set_aspect("equal", adjustable="box")
            title_suffix = f" | {class_label}" if class_label else ""
            ax_xy.set_title(f"trial {int(trial['trial_ordinal'])} id={trial_id}{title_suffix}")
            ax_xy.set_ylabel("chaser-frame y (mm)")
            if row == n_trials - 1:
                ax_xy.set_xlabel("chaser-frame x (mm)")
            ax_d.axvline(0.0, color="#dc2626", linestyle="--", linewidth=0.9)
            ax_d.axhline(float(result.trigger_radius_mm), color="#f97316", linestyle=":", linewidth=0.9)
            ax_d.set_ylabel("distance (mm)")
            if row == n_trials - 1:
                ax_d.set_xlabel("time from trigger (s)")
            ax_d.set_title("distance trace")
            ax_d.grid(axis="y", alpha=0.25)
        fig.suptitle(
            f"Chaser-centric escape/freeze canary: {result.recording_id}\n"
            "Candidate labels from full-trial fish path; chaser heading points up; active pursuit chaser 0 only",
            fontsize=11,
        )
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=int(dpi))
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def render_fish_centered_diagnostic_png(result: EscapeFreezeCanaryResult, *, dpi: int = 150) -> bytes:
    trials = result.trial_table
    traj = result.trajectory_table
    n_trials = int(trials.shape[0])
    required = {"chaser_x_fish_centered_mm", "chaser_y_fish_centered_mm"}
    if n_trials == 0 or not required.issubset(set(traj.dtype.names or ())):
        fig, ax = plt.subplots(figsize=(7, 3), constrained_layout=True)
        ax.text(0.5, 0.5, "No fish-centered trajectory samples", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        fig, axes = plt.subplots(
            n_trials,
            2,
            figsize=(10.5, max(2.2, 1.8 * n_trials)),
            constrained_layout=True,
            squeeze=False,
        )
        finite = traj[
            np.asarray(traj["valid"], dtype=bool)
            & np.isfinite(traj["chaser_x_fish_centered_mm"])
            & np.isfinite(traj["chaser_y_fish_centered_mm"])
        ]
        if finite.size:
            lim = float(
                np.nanpercentile(
                    np.abs(
                        np.concatenate(
                            [
                                finite["chaser_x_fish_centered_mm"],
                                finite["chaser_y_fish_centered_mm"],
                            ]
                        )
                    ),
                    98,
                )
            )
            lim = max(5.0, min(80.0, lim * 1.15))
        else:
            lim = 20.0
        rings = [3.0, 5.0, 10.0, 20.0, 40.0]
        for row, trial in enumerate(trials):
            trial_id = int(trial["trial_id"])
            rows = traj[traj["trial_id"] == trial_id]
            class_label = ""
            if row < result.metric_table.shape[0] and "candidate_response_class" in (result.metric_table.dtype.names or ()):
                class_label = result.metric_table["candidate_response_class"][row].decode("utf-8", errors="ignore").strip("\x00")
            ax_xy = axes[row, 0]
            ax_d = axes[row, 1]
            for ring in rings:
                if ring <= lim:
                    color = "#f97316" if math.isclose(ring, 3.0) else "#cbd5e1"
                    linewidth = 1.0 if math.isclose(ring, 3.0) else 0.7
                    ax_xy.add_patch(plt.Circle((0.0, 0.0), ring, fill=False, color=color, linewidth=linewidth))
            ax_xy.scatter([0.0], [0.0], marker="o", s=48, color="#0f172a", label="fish")
            valid = (
                np.asarray(rows["valid"], dtype=bool)
                & np.isfinite(rows["chaser_x_fish_centered_mm"])
                & np.isfinite(rows["chaser_y_fish_centered_mm"])
            )
            if rows.size:
                ax_xy.plot(
                    rows["chaser_x_fish_centered_mm"],
                    rows["chaser_y_fish_centered_mm"],
                    color="#dc2626",
                    linewidth=1.1,
                    alpha=0.85,
                )
                if np.any(valid):
                    valid_indices = np.flatnonzero(valid)
                    first = int(valid_indices[0])
                    last = int(valid_indices[-1])
                    trigger_candidates = np.flatnonzero(valid & (np.asarray(rows["relative_frame"], dtype=np.int32) == 0))
                    ax_xy.scatter(
                        [rows["chaser_x_fish_centered_mm"][first]],
                        [rows["chaser_y_fish_centered_mm"][first]],
                        marker="o",
                        s=30,
                        facecolor="white",
                        edgecolor="#0f172a",
                        linewidth=1.0,
                    )
                    if trigger_candidates.size:
                        trigger_idx = int(trigger_candidates[0])
                        ax_xy.scatter(
                            [rows["chaser_x_fish_centered_mm"][trigger_idx]],
                            [rows["chaser_y_fish_centered_mm"][trigger_idx]],
                            marker="x",
                            s=38,
                            color="#f97316",
                            linewidth=1.3,
                        )
                    ax_xy.scatter(
                        [rows["chaser_x_fish_centered_mm"][last]],
                        [rows["chaser_y_fish_centered_mm"][last]],
                        marker="o",
                        s=32,
                        color="#dc2626",
                    )
                ax_d.plot(rows["time_from_trigger_s"], rows["distance_mm"], color="#334155", linewidth=1.2)
            ax_xy.axhline(0, color="#e2e8f0", linewidth=0.7)
            ax_xy.axvline(0, color="#e2e8f0", linewidth=0.7)
            ax_xy.set_xlim(-lim, lim)
            ax_xy.set_ylim(-lim, lim)
            ax_xy.set_aspect("equal", adjustable="box")
            title_suffix = f" | {class_label}" if class_label else ""
            ax_xy.set_title(f"trial {int(trial['trial_ordinal'])} id={trial_id}{title_suffix}")
            ax_xy.set_ylabel("chaser y relative to fish (mm)")
            if row == n_trials - 1:
                ax_xy.set_xlabel("chaser x relative to fish (mm)")
            ax_d.axvline(0.0, color="#dc2626", linestyle="--", linewidth=0.9)
            ax_d.axhline(3.0, color="#f97316", linestyle=":", linewidth=0.9)
            ax_d.axhline(float(result.trigger_radius_mm), color="#94a3b8", linestyle=":", linewidth=0.8)
            ax_d.set_ylabel("distance (mm)")
            if row == n_trials - 1:
                ax_d.set_xlabel("time from trigger (s)")
            ax_d.set_title("distance trace")
            ax_d.grid(axis="y", alpha=0.25)
        fig.suptitle(
            f"Fish-centered chaser diagnostic: {result.recording_id}\n"
            "Fish at origin; +x/right is 0 deg, +y/up is 90 deg; no fish-heading rotation",
            fontsize=11,
        )
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=int(dpi))
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def render_speed_displacement_scatter_png(result: EscapeFreezeCanaryResult, *, dpi: int = 150) -> bytes:
    metrics = result.metric_table
    fig, ax = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)
    if metrics.size == 0:
        ax.text(0.5, 0.5, "No active chase trials", ha="center", va="center", transform=ax.transAxes)
    else:
        x = metrics["full_trial_fish_path_length_mm"].astype(np.float64)
        y = metrics["full_trial_mean_speed_mm_s"].astype(np.float64)
        classes = np.asarray(metrics["escape_attempt"], dtype=bool)
        colors = np.where(classes, "#dc2626", "#64748b")
        ax.scatter(x, y, c=colors, s=68, edgecolor="#0f172a", linewidth=0.5)
        ax.axvline(
            float(result.escape_path_threshold_mm),
            color="#dc2626",
            linestyle="--",
            linewidth=1.0,
            label=f"escape threshold ({result.escape_path_threshold_mm:g} mm)",
        )
        for row in metrics:
            if np.isfinite(row["full_trial_fish_path_length_mm"]) and np.isfinite(row["full_trial_mean_speed_mm_s"]):
                ax.text(
                    float(row["full_trial_fish_path_length_mm"]),
                    float(row["full_trial_mean_speed_mm_s"]),
                    f"{int(row['trial_ordinal'])}",
                    fontsize=7,
                    ha="center",
                    va="center",
                    color="white",
                )
        ax.legend(loc="best")
    ax.set_xlabel("full-trial fish path length (mm)")
    ax.set_ylabel("full-trial mean fish speed (mm/s)")
    ax.set_title("Candidate escape labels from full-trial movement")
    ax.grid(alpha=0.25)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=int(dpi))
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def render_response_class_bar_png(result: EscapeFreezeCanaryResult, *, dpi: int = 150) -> bytes:
    metrics = result.metric_table
    fig, ax = plt.subplots(figsize=(6.6, 4.2), constrained_layout=True)
    labels = ["escape_attempt", "not_escape / freeze_candidate"]
    if metrics.size == 0 or "escape_attempt" not in (metrics.dtype.names or ()):
        counts = [0, 0]
    else:
        escape_count = int(np.count_nonzero(np.asarray(metrics["escape_attempt"], dtype=bool)))
        total = int(metrics.shape[0])
        counts = [escape_count, max(0, total - escape_count)]
    total_count = int(sum(counts))
    bars = ax.bar(
        labels,
        counts,
        color=["#dc2626", "#64748b"],
        edgecolor="#0f172a",
        linewidth=0.8,
    )
    ymax = max(1.0, float(max(counts) if counts else 0) * 1.25)
    ax.set_ylim(0, ymax)
    for bar, count in zip(bars, counts):
        pct = (100.0 * float(count) / float(total_count)) if total_count else 0.0
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + ymax * 0.03,
            f"{count} ({pct:.0f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylabel("Trial count")
    ax.set_title(
        "Candidate escape/no-escape trial labels\n"
        f"path threshold >= {result.escape_path_threshold_mm:g} mm; classifier not cohort-locked"
    )
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=0)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=int(dpi))
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def render_trial_outcome_timeline_png(result: EscapeFreezeCanaryResult, *, dpi: int = 150) -> bytes:
    trials = result.trial_table
    metrics = result.metric_table
    fig, ax = plt.subplots(figsize=(9.0, 3.8), constrained_layout=True)
    if trials.size == 0 or metrics.size == 0:
        ax.text(0.5, 0.5, "No active chase trials", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        n = min(int(trials.shape[0]), int(metrics.shape[0]))
        start_s = trials["start_time_s"][:n].astype(np.float64)
        end_s = trials["end_time_s"][:n].astype(np.float64)
        trigger_s = trials["trigger_frame"][:n].astype(np.float64) / float(result.fps if result.fps > 0 else 1.0)
        ordinal = trials["trial_ordinal"][:n].astype(np.int32)
        escape = np.asarray(metrics["escape_attempt"][:n], dtype=bool)
        y = np.ones(n, dtype=np.float64)

        for idx in range(n):
            ax.hlines(
                y=y[idx],
                xmin=float(start_s[idx]),
                xmax=float(end_s[idx]),
                color="#cbd5e1",
                linewidth=5.0,
                alpha=0.85,
                zorder=1,
            )

        if np.any(~escape):
            ax.scatter(
                trigger_s[~escape],
                y[~escape],
                marker="o",
                s=86,
                color="#64748b",
                edgecolor="#0f172a",
                linewidth=0.6,
                label="not_escape / freeze_candidate",
                zorder=3,
            )
        if np.any(escape):
            ax.scatter(
                trigger_s[escape],
                y[escape],
                marker="^",
                s=110,
                color="#dc2626",
                edgecolor="#0f172a",
                linewidth=0.6,
                label="escape_attempt",
                zorder=4,
            )

        for idx in range(n):
            ax.text(
                float(trigger_s[idx]),
                1.08 if idx % 2 == 0 else 0.92,
                str(int(ordinal[idx])),
                ha="center",
                va="center",
                fontsize=8,
                color="#0f172a",
            )

        xmin = float(np.nanmin(start_s)) if np.isfinite(start_s).any() else 0.0
        xmax = float(np.nanmax(end_s)) if np.isfinite(end_s).any() else 1.0
        pad = max(1.0, (xmax - xmin) * 0.04)
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(0.72, 1.28)
        ax.set_yticks([])
        ax.set_xlabel("Time in recording (s)")
        ax.set_title(
            "Trial outcome timeline\n"
            f"path threshold >= {result.escape_path_threshold_mm:g} mm; labels are candidate, not cohort-locked"
        )
        ax.grid(axis="x", alpha=0.25)
        ax.legend(loc="upper right")
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=int(dpi))
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def _fish_centered_polar_samples(result: EscapeFreezeCanaryResult) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    traj = result.trajectory_table
    required = {"chaser_x_fish_centered_mm", "chaser_y_fish_centered_mm", "trial_ordinal", "valid"}
    if not required.issubset(set(traj.dtype.names or ())) or traj.size == 0:
        return np.asarray([]), np.asarray([]), np.asarray([])
    x = np.asarray(traj["chaser_x_fish_centered_mm"], dtype=np.float64)
    y = np.asarray(traj["chaser_y_fish_centered_mm"], dtype=np.float64)
    valid = np.asarray(traj["valid"], dtype=bool) & np.isfinite(x) & np.isfinite(y)
    if not np.any(valid):
        return np.asarray([]), np.asarray([]), np.asarray([])
    theta = np.mod(np.arctan2(y[valid], x[valid]), 2.0 * math.pi)
    radius = np.sqrt((x[valid] * x[valid]) + (y[valid] * y[valid]))
    ordinal = np.asarray(traj["trial_ordinal"], dtype=np.float64)[valid]
    finite = np.isfinite(theta) & np.isfinite(radius) & np.isfinite(ordinal)
    return theta[finite], radius[finite], ordinal[finite]


def _polar_rmax(radius: np.ndarray, *, minimum: float = 20.0) -> float:
    values = np.asarray(radius, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float(minimum)
    high = float(np.nanpercentile(values, 99.0))
    high = max(float(minimum), high, DEFAULT_TRIGGER_RADIUS_MM)
    return float(math.ceil(high / 5.0) * 5.0)


def _polar_ring_values(rmax: float, trigger_radius_mm: float) -> list[float]:
    candidates = [3.0, 5.0, 10.0, float(trigger_radius_mm), 40.0]
    values: list[float] = []
    for value in candidates:
        if math.isfinite(value) and value > 0 and value <= float(rmax) and not any(math.isclose(value, old) for old in values):
            values.append(float(value))
    return values


def _decorate_fish_centered_polar_axis(ax: Any, *, rmax: float, trigger_radius_mm: float) -> None:
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_thetagrids([0, 90, 180, 270], labels=["0 right", "90 up", "180 left", "270 down"])
    ax.set_ylim(0.0, float(rmax))
    rings = _polar_ring_values(float(rmax), float(trigger_radius_mm))
    if rings:
        ax.set_rgrids(rings, labels=[f"{value:g} mm" for value in rings], angle=35)
    theta = np.linspace(0.0, 2.0 * math.pi, 360)
    if 3.0 <= float(rmax):
        ax.plot(theta, np.full_like(theta, 3.0), color="#f97316", linewidth=1.2, alpha=0.95)
    if math.isfinite(float(trigger_radius_mm)) and 0 < float(trigger_radius_mm) <= float(rmax):
        ax.plot(theta, np.full_like(theta, float(trigger_radius_mm)), color="#0ea5e9", linewidth=1.1, alpha=0.9)


def render_fish_centered_polar_approach_png(result: EscapeFreezeCanaryResult, *, dpi: int = 150) -> bytes:
    theta, radius, ordinal = _fish_centered_polar_samples(result)
    fig, ax = plt.subplots(figsize=(7.0, 6.4), subplot_kw={"projection": "polar"}, constrained_layout=True)
    if theta.size == 0:
        ax.text(0.5, 0.5, "No fish-centered polar samples", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        rmax = _polar_rmax(radius, minimum=max(20.0, float(result.trigger_radius_mm)))
        scatter = ax.scatter(
            theta,
            radius,
            c=ordinal,
            cmap="viridis",
            s=7,
            alpha=0.22,
            linewidths=0,
        )
        _decorate_fish_centered_polar_axis(ax, rmax=rmax, trigger_radius_mm=float(result.trigger_radius_mm))
        cbar = fig.colorbar(scatter, ax=ax, pad=0.10, shrink=0.82)
        cbar.set_label("trial ordinal")
        ax.set_title(
            "Fish-centered chaser approach frames\n"
            "all active-pursuit frames pooled; 0 deg = right, 90 deg = up",
            va="bottom",
        )
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=int(dpi))
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def render_fish_centered_polar_density_png(result: EscapeFreezeCanaryResult, *, dpi: int = 150) -> bytes:
    theta, radius, _ordinal = _fish_centered_polar_samples(result)
    fig, ax = plt.subplots(figsize=(7.0, 6.4), subplot_kw={"projection": "polar"}, constrained_layout=True)
    if theta.size == 0:
        ax.text(0.5, 0.5, "No fish-centered polar samples", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        rmax = _polar_rmax(radius, minimum=max(20.0, float(result.trigger_radius_mm)))
        theta_edges = np.linspace(0.0, 2.0 * math.pi, 73)
        radial_step = 1.0 if rmax <= 30.0 else 2.0
        radial_edges = np.arange(0.0, float(rmax) + radial_step, radial_step)
        if radial_edges.size < 2:
            radial_edges = np.asarray([0.0, float(rmax)], dtype=np.float64)
        hist, theta_edges, radial_edges = np.histogram2d(
            theta,
            radius,
            bins=(theta_edges, radial_edges),
        )
        mesh = ax.pcolormesh(
            theta_edges,
            radial_edges,
            np.log1p(hist.T),
            cmap="magma",
            shading="auto",
        )
        _decorate_fish_centered_polar_axis(ax, rmax=rmax, trigger_radius_mm=float(result.trigger_radius_mm))
        cbar = fig.colorbar(mesh, ax=ax, pad=0.10, shrink=0.82)
        cbar.set_label("log1p(frame count)")
        ax.set_title(
            "Fish-centered chaser approach density\n"
            "polar-binned active-pursuit frames; 0 deg = right, 90 deg = up",
            va="bottom",
        )
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=int(dpi))
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def _write_structured_table(group: zarr.Group, table: np.ndarray) -> None:
    for name in table.dtype.names or ():
        _write_array(group, name, np.asarray(table[name]))


def write_escape_freeze_canary_component(
    zarr_path: Path,
    result: EscapeFreezeCanaryResult,
    *,
    overwrite: bool = False,
    write_png: bool = True,
) -> str:
    root = _open_root(zarr_path, mode="a")
    run_group = root[result.chaser_distance_run_path]
    parent = run_group.require_group(COMPONENT_PARENT_NAME)
    component_name = result.component_name
    if component_name in parent:
        if not overwrite:
            raise ValueError(
                f"Escape/freeze canary component already exists: "
                f"{result.chaser_distance_run_path}/{COMPONENT_PARENT_NAME}/{component_name}"
            )
        del parent[component_name]
    component = parent.create_group(component_name)
    component_path = f"{result.chaser_distance_run_path}/{COMPONENT_PARENT_NAME}/{component_name}"

    config = component.require_group("config")
    config.attrs.update(
        json_attr_safe(
            {
                "chaser_index": int(result.chaser_index),
                "response_window_s": float(result.response_window_s),
                "freeze_window_s": float(result.freeze_window_s),
                "baseline_window_s": float(result.baseline_window_s),
                "trigger_radius_mm": float(result.trigger_radius_mm),
                "trigger_radius_source": str(result.trigger_radius_source),
                "trigger_radius_override": bool(result.trigger_radius_override),
                "escape_path_threshold_mm": float(result.escape_path_threshold_mm),
                "low_speed_threshold_mm_s": float(result.low_speed_threshold_mm_s),
                "heading_min_speed_mm_s": float(result.heading_min_speed_mm_s),
                "classification_locked": False,
            }
        )
    )

    trials = component.require_group("trials")
    _write_structured_table(trials, result.trial_table)
    trials.attrs.update({"row_axis": "active_pursuit_trials"})

    metrics = component.require_group("trial_metrics")
    _write_structured_table(metrics, result.metric_table)
    metrics.attrs.update(
        {
            "row_axis": "active_pursuit_trials",
            "classification_locked": False,
            "max_windowed_speed_caveat": "diagnostic only at 100 FPS; classification should use windowed mean/displacement",
        }
    )

    trajectories = component.require_group("trial_trajectories")
    _write_structured_table(trajectories, result.trajectory_table)
    trajectories.attrs.update(
        json_attr_safe(
            {
                "row_axis": "trial_frame_samples",
                "coordinate_frame": "chaser_centric_mm",
                "coordinate_frame_scope": "default for fish_x_chaser_frame_mm, fish_y_chaser_frame_mm, and bearing_deg",
                "x_axis_direction": "right_relative_to_chaser_heading",
                "y_axis_direction": "chaser_heading_forward",
                "chaser_centric_columns": ["fish_x_chaser_frame_mm", "fish_y_chaser_frame_mm", "bearing_deg"],
                "chaser_centric_coordinate_frame": "chaser_centric_mm",
                "chaser_centric_x_axis_direction": "right_relative_to_chaser_heading",
                "chaser_centric_y_axis_direction": "chaser_heading_forward",
                "chaser_centric_bearing_deg_convention": "0=chaser_heading_forward; positive=right_relative_to_chaser_heading",
                "fish_centered_columns": ["chaser_x_fish_centered_mm", "chaser_y_fish_centered_mm"],
                "fish_centered_coordinate_frame": "fish_centered_world_mm",
                "fish_centered_x_axis_direction": "right",
                "fish_centered_y_axis_direction": "up",
                "fish_centered_angle_convention": "0 deg=right; 90 deg=up; no fish-heading rotation",
                "column_coordinate_frames": {
                    "fish_x_chaser_frame_mm": "chaser_centric_mm",
                    "fish_y_chaser_frame_mm": "chaser_centric_mm",
                    "bearing_deg": "chaser_centric_mm_angle",
                    "chaser_x_fish_centered_mm": "fish_centered_world_mm",
                    "chaser_y_fish_centered_mm": "fish_centered_world_mm",
                },
            }
        )
    )

    summary_group = component.require_group("summary")
    _write_array(summary_group, "summary_json_bytes", _bytes_array([json.dumps(json_attr_safe(result.summary), sort_keys=True)], width=4096))
    _write_array(summary_group, "diagnostics_json_bytes", _bytes_array([json.dumps(json_attr_safe(result.diagnostics), sort_keys=True)], width=4096))
    _write_array(summary_group, "warnings_json_bytes", _bytes_array([json.dumps(list(result.warnings), sort_keys=True)], width=4096))
    summary_group.attrs.update({"row_axis": "fish_recording", "summary": json_attr_safe(result.summary)})

    git = get_git_info(Path(__file__).resolve().parents[3])
    source_refs = {
        "chaser_distance_run": result.chaser_distance_run_name,
        "chaser_distance_path": result.chaser_distance_run_path,
        "source_stimulus_run": result.source_stimulus_run,
        "source_stimulus_path": result.source_stimulus_path,
    }
    parameters = {
        "chaser_index": int(result.chaser_index),
        "response_window_s": float(result.response_window_s),
        "freeze_window_s": float(result.freeze_window_s),
        "baseline_window_s": float(result.baseline_window_s),
        "trigger_radius_mm": float(result.trigger_radius_mm),
        "trigger_radius_source": str(result.trigger_radius_source),
        "trigger_radius_override": bool(result.trigger_radius_override),
        "escape_path_threshold_mm": float(result.escape_path_threshold_mm),
        "low_speed_threshold_mm_s": float(result.low_speed_threshold_mm_s),
        "heading_min_speed_mm_s": float(result.heading_min_speed_mm_s),
    }
    attrs = {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "method_version": METHOD_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "component_name": component_name,
        "recording_id": result.recording_id,
        "status": "diagnostic_canary",
        "classification_locked": False,
        "row_axis": "active_pursuit_trials",
        "source_refs": source_refs,
        "parameters": parameters,
        "summary": result.summary,
        "diagnostics": result.diagnostics,
        "warnings": list(result.warnings),
        "fps": float(result.fps),
        "total_frames": int(result.total_frames),
        "pixels_per_mm_projector": float(result.pixels_per_mm_projector),
        "measurement_ceiling": "100 FPS; no C-start kinematic signature or sub-30 ms latency claims; windowed speed only",
        "interpretation_scope": "US validation and within-aggressive trend diagnostic; not active-pursuit aggressive-vs-benign specificity",
        "git_commit": git.get("commit_hash"),
        "git_branch": git.get("branch"),
        "git_dirty": git.get("is_dirty"),
        "provenance": {
            "stage": "chaser_escape_freeze_canary",
            "created_by": "fisheye.analysis.chaser_escape_freeze",
            "inputs": source_refs,
            "parameters": parameters,
        },
    }
    component.attrs.update(json_attr_safe(attrs))
    lineage_payload = build_run_lineage_payload(
        run_family=f"{result.chaser_distance_run_path}/{COMPONENT_PARENT_NAME}",
        analysis_schema={"schema_id": SCHEMA_ID, "schema_version": SCHEMA_VERSION, "row_axis": "active_pursuit_trials"},
        method=METHOD,
        method_version=METHOD_VERSION,
        source_refs=source_refs,
        parameters=parameters,
        code={"git_commit": git.get("commit_hash"), "git_dirty": git.get("is_dirty")},
    )
    write_run_lineage_attrs(component, lineage_payload, fingerprint_status="best_effort", overwrite=True)
    parent.attrs["latest"] = component_name
    parent.attrs["latest_complete"] = component_name

    if write_png:
        source_paths = {
            **source_refs,
            "component": component_path,
            "trials": f"{component_path}/trials",
            "trial_metrics": f"{component_path}/trial_metrics",
            "trial_trajectories": f"{component_path}/trial_trajectories",
        }
        for artifact_name, renderer, description in (
            (
                PER_TRIAL_DIAGNOSTIC_PNG_ARTIFACT_NAME,
                render_per_trial_diagnostic_png,
                "GoodCopBadCop chaser-centric per-trial escape/freeze diagnostic.",
            ),
            (
                FISH_CENTERED_DIAGNOSTIC_PNG_ARTIFACT_NAME,
                render_fish_centered_diagnostic_png,
                "GoodCopBadCop fish-centered per-trial chaser-position diagnostic.",
            ),
            (
                RESPONSE_CLASS_BAR_PNG_ARTIFACT_NAME,
                render_response_class_bar_png,
                "GoodCopBadCop candidate escape/no-escape response-class summary.",
            ),
            (
                TRIAL_OUTCOME_TIMELINE_PNG_ARTIFACT_NAME,
                render_trial_outcome_timeline_png,
                "GoodCopBadCop candidate escape/no-escape trial outcome timeline.",
            ),
            (
                FISH_CENTERED_POLAR_APPROACH_PNG_ARTIFACT_NAME,
                render_fish_centered_polar_approach_png,
                "GoodCopBadCop fish-centered polar chaser-approach scatter.",
            ),
            (
                FISH_CENTERED_POLAR_DENSITY_PNG_ARTIFACT_NAME,
                render_fish_centered_polar_density_png,
                "GoodCopBadCop fish-centered polar chaser-approach density.",
            ),
            (
                SPEED_DISPLACEMENT_SCATTER_PNG_ARTIFACT_NAME,
                render_speed_displacement_scatter_png,
                "GoodCopBadCop escape/freeze speed-vs-displacement diagnostic scatter.",
            ),
        ):
            write_png_visualization_artifact(
                component,
                artifact_name,
                renderer(result),
                description=description,
                created_by="fisheye.analysis.chaser_escape_freeze",
                role="analysis_diagnostic",
                source_paths=source_paths,
                source_runs={"chaser_distance_run": result.chaser_distance_run_name},
                parameters=parameters,
                extra_attrs={
                    "chaser_escape_freeze_schema_id": SCHEMA_ID,
                    "component_path": component_path,
                    "canonical_artifact": True,
                },
                overwrite=True,
            )
    return component_path


def _result_payload(result: EscapeFreezeCanaryResult, *, applied_path: str | None) -> dict[str, Any]:
    return {
        "schema_id": SCHEMA_ID,
        "zarr_path": result.zarr_path,
        "recording_id": result.recording_id,
        "component_name": result.component_name,
        "applied_path": applied_path,
        "chaser_distance_run": result.chaser_distance_run_name,
        "source_stimulus_run": result.source_stimulus_run,
        "summary": json_attr_safe(result.summary),
        "diagnostics": json_attr_safe(result.diagnostics),
        "warnings": list(result.warnings),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Analysis zarr archive.")
    parser.add_argument("--chaser-distance-run", default="latest", help="Chaser-distance run name, or latest.")
    parser.add_argument("--component-name", default=DEFAULT_COMPONENT_NAME)
    parser.add_argument("--chaser-index", type=int, default=DEFAULT_CHASER_INDEX)
    parser.add_argument("--response-window-s", type=float, default=DEFAULT_RESPONSE_WINDOW_S)
    parser.add_argument("--freeze-window-s", type=float, default=DEFAULT_FREEZE_WINDOW_S)
    parser.add_argument("--baseline-window-s", type=float, default=DEFAULT_BASELINE_WINDOW_S)
    parser.add_argument(
        "--trigger-radius-mm",
        type=float,
        default=None,
        help=(
            "Override trigger radius in mm. If omitted, resolve from chaser_states.initial_distance_mm, "
            "then protocol_json, then the 5 mm fallback."
        ),
    )
    parser.add_argument(
        "--escape-path-threshold-mm",
        type=float,
        default=DEFAULT_ESCAPE_PATH_THRESHOLD_MM,
        help=(
            "Candidate escape-attempt threshold on full-trial fish path length in mm. "
            "This labels canary trials but does not mark the classifier as cohort-locked."
        ),
    )
    parser.add_argument("--low-speed-threshold-mm-s", type=float, default=DEFAULT_LOW_SPEED_THRESHOLD_MM_S)
    parser.add_argument("--heading-min-speed-mm-s", type=float, default=DEFAULT_HEADING_MIN_SPEED_MM_S)
    parser.add_argument("--apply", action="store_true", help="Write the component into the zarr.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing component with the same name.")
    parser.add_argument("--no-png", action="store_true", help="Skip static PNG artifact generation.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = build_escape_freeze_canary_result(
        args.zarr_path,
        chaser_distance_run=args.chaser_distance_run,
        component_name=args.component_name,
        chaser_index=args.chaser_index,
        response_window_s=args.response_window_s,
        freeze_window_s=args.freeze_window_s,
        baseline_window_s=args.baseline_window_s,
        trigger_radius_mm=args.trigger_radius_mm,
        escape_path_threshold_mm=args.escape_path_threshold_mm,
        low_speed_threshold_mm_s=args.low_speed_threshold_mm_s,
        heading_min_speed_mm_s=args.heading_min_speed_mm_s,
    )
    applied_path = None
    if args.apply:
        applied_path = write_escape_freeze_canary_component(
            args.zarr_path,
            result,
            overwrite=args.overwrite,
            write_png=not args.no_png,
        )
    print(json.dumps(_result_payload(result, applied_path=applied_path), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
