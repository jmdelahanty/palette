"""
Stimulus response analysis — per-step behavioral metrics across stimulus types.

Consumes identity-resolved track data from ``track_kinematics_runs`` and
stimulus metadata from ``stimulus_runs`` to produce per-step movement
summaries, bout statistics, and (for grating steps) heading alignment metrics.

See ``docs/stimulus_response_run_design.md`` for the full storage layout
and metric definitions.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import zarr
from rich.console import Console

from fisheye.shared.citrus_enums import (
    EVENT_LOOM_AUTO_REPEAT,
    EVENT_LOOM_MANUAL_START,
    EVENT_STEP_START,
    EVENT_STEP_END,
    STIMULUS_MODE,
    load_event_types,
    load_stimulus_modes,
)
from fisheye.shared.coordinate_transform import (
    load_calibration_transform,
    projector_px_to_mm,
    projector_to_camera_mm,
    resolve_concentric_center_mm,
    visual_angle_deg,
)
from fisheye.shared.json_safety import (
    decode_null_terminated_text,
    json_attr_safe,
    json_attr_safe_mapping,
)
from fisheye.shared.stage_provenance import (
    build_stage_provenance,
    write_stage_provenance,
)
from fisheye.shared.run_lineage_fingerprint import write_best_effort_run_lineage_attrs
from fisheye.shared.zarr_helpers import resolve_zarr_run
from fisheye.shared.zarr_run_completion import mark_run_complete, mark_run_started, require_runs_parent
from fisheye.shared.system_metadata import get_git_info
from fisheye.shared.zarr_io import open_zarr_root

from fisheye.analysis.swim_bout_io import load_default_swim_bout_tables
from fisheye.analysis.chaser_state_interpolator import write_columnar_dataset
from fisheye.analysis.track_kinematics_io import (
    list_track_ids,
    load_track_kinematics_track,
    resolve_track_kinematics_run,
)
from fisheye.analysis.stimulus_response_grating import (
    _MOVING_GRATING,
    _grating_direction_vector,
    compute_grating_per_fish,
    compute_grating_per_frame,
    compute_grating_time_series,
    resolve_grating_direction,
    resolve_grating_speed_mm_s,
)
from fisheye.analysis.stimulus_response_concentric_omr import (
    CONCENTRIC_RADIAL_OMR_DEFAULT_EARLY_RESPONSE_WINDOWS_S,
    CONCENTRIC_RADIAL_OMR_DEFAULT_WINDOW_LENGTHS_S,
    CONCENTRIC_RADIAL_OMR_METHOD_VERSION,
    ConcentricRadialOMRStepData,
    compute_step_concentric_radial_omr_metrics,
)
# OMR metrics are implemented in a dedicated module, but re-exported here so
# existing callers can keep importing from fisheye.analysis.stimulus_response.
from fisheye.analysis.stimulus_response_omr import (
    OMR_DEFAULT_EARLY_RESPONSE_WINDOWS_S,
    OMR_DEFAULT_WINDOW_LENGTHS_S,
    OMR_METHOD_VERSION,
    OMRStepData,
    _resolve_omr_arena_geometry_mm,
    compute_global_omr_metrics,
    compute_step_omr_metrics,
)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class DenseTrack:
    """Frame-aligned dense arrays for one fish."""

    fish_id: int
    speed_mm: np.ndarray              # float32[n_frames] — zeros where no detection
    heading_deg: np.ndarray           # float32[n_frames]
    positions_mm: np.ndarray          # float32[n_frames, 2]
    angular_velocity: np.ndarray      # float32[n_frames]
    time_seconds: np.ndarray          # float32[n_frames]
    valid: np.ndarray                 # bool[n_frames] — True where detection exists
    detection_source: np.ndarray      # int8[n_frames] — 0=real, 1=interpolated, -1=gap
    frame_path_distance_smoothed_mm: Optional[np.ndarray] = None  # float32[n_frames], path increment from previous frame
    cumulative_path_distance_mm: Optional[np.ndarray] = None       # float32[n_frames], forward-filled through gaps


@dataclass
class ProtocolStep:
    """Parsed protocol step from stimulus events."""

    index: int
    name: str
    stimulus_mode: str        # string name, e.g. "MOVING_GRATING"
    stimulus_mode_id: int     # integer enum
    start_frame: int          # camera frame (inclusive)
    end_frame: int            # camera frame (exclusive)
    duration_s: float
    stimulus_params: Dict[str, Any] = field(default_factory=dict)


def flatten_stimulus_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Return the canonical Citrus protocol parameter dict for a step.

    Modern Citrus protocol JSON stores stimulus-specific fields under
    ``stimulus_params["parameters"]``. Synthetic unit tests may still pass a
    direct parameter dict, but nested Citrus parameters are the production
    source of truth.
    """

    if not isinstance(params, dict):
        return {}

    nested = params.get("parameters")
    if isinstance(nested, dict):
        return dict(nested)
    return dict(params)


_json_safe_attr_value = json_attr_safe
_json_safe_attrs = json_attr_safe_mapping


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------


def _snapshot_upstream_lineage(kin_group) -> Dict[str, Any]:
    """Read the track_kinematics run's upstream provenance for embedding."""
    lineage: Dict[str, Any] = {}
    attrs = kin_group.attrs if hasattr(kin_group, "attrs") else {}

    # Direct lineage attrs written by track_kinematics.
    for key in (
        "source_tracking_run",
        "source_arena_assignment_run",
        "method",
        "fps",
        "pixel_to_mm",
        "coordinate_space",
    ):
        val = attrs.get(key)
        if val is not None:
            lineage[key] = val

    # The inputs dict (if present) has the full upstream chain.
    inputs = attrs.get("inputs")
    if isinstance(inputs, dict):
        for key in (
            "detection_run",
            "detection_path",
            "detection_variant",
            "source_detect_run",
            "keypoint_run",
            "keypoint_variant",
            "base_keypoint_run",
            "crop_run",
        ):
            val = inputs.get(key)
            if val is not None:
                lineage[key] = val

    # Provenance contract info if available.
    prov = attrs.get("provenance")
    if isinstance(prov, dict):
        contract = prov.get("contract")
        if isinstance(contract, dict):
            lineage["provenance_contract"] = contract
        git = prov.get("git")
        if isinstance(git, dict):
            lineage["kinematics_git_commit"] = git.get("commit")

    return lineage


def load_track_data(
    root: zarr.Group,
    *,
    kinematics_type: str = "offline",
    kinematics_run: Optional[str] = None,
    console: Optional[Console] = None,
) -> Tuple[List[DenseTrack], str, int, Dict[str, Any]]:
    """Load track kinematics and expand sparse arrays to dense frame-aligned representation.

    Returns
    -------
    tracks : list[DenseTrack]
        One entry per fish, sorted by fish_id.
    run_name : str
        Resolved kinematics run name.
    n_frames : int
        Total frames (max frame_index + 1 across all tracks).
    upstream_lineage : dict
        Snapshot of the kinematics run's upstream provenance (detection run,
        keypoint run, crop run, tracking run, etc.).
    """
    console = console or Console()

    kin_group, run_name, run_path = resolve_track_kinematics_run(
        root,
        run_name=kinematics_run or "latest",
        scope=kinematics_type,
    )

    fps = float(kin_group.attrs.get("fps", 30.0))

    track_ids = list_track_ids(kin_group)
    if not track_ids:
        raise ValueError(f"No tracks found in {run_path}/")

    # First pass: determine total frame span and validate inputs.
    max_frame = 0
    sparse_tracks = []
    for fish_id in track_ids:
        track = load_track_kinematics_track(
            root,
            run_name=run_name,
            scope=kinematics_type,
            track_id=int(fish_id),
            required_speed_levels=("smoothed",),
        )
        sparse_tracks.append(track)
        if track.frame_indices.shape[0] > 0:
            max_frame = max(max_frame, int(track.frame_indices[-1]))

    n_frames = max_frame + 1

    # Second pass: expand sparse → dense.
    tracks: List[DenseTrack] = []
    for track in sparse_tracks:
        fish_id = int(track.track_id)
        frame_indices = track.frame_indices.astype(np.int64, copy=False)
        n_samples = frame_indices.shape[0]

        # Allocate dense arrays.  Gap frames default to zero/False/-1.
        speed_mm = np.zeros(n_frames, dtype=np.float32)
        heading_deg = np.zeros(n_frames, dtype=np.float32)
        pos_mm = np.zeros((n_frames, 2), dtype=np.float32)
        ang_vel = np.zeros(n_frames, dtype=np.float32)
        time_s = np.zeros(n_frames, dtype=np.float32)
        valid = np.zeros(n_frames, dtype=bool)
        det_src = np.full(n_frames, -1, dtype=np.int8)  # -1 = no detection
        frame_path_distance_mm: Optional[np.ndarray] = None
        cumulative_path_distance_mm: Optional[np.ndarray] = None

        if n_samples > 0:
            if track.heading_degrees is None:
                raise ValueError(f"{track.track_path} is missing required array 'heading_degrees'")
            if track.positions_mm is None:
                raise ValueError(f"{track.track_path} is missing required array 'positions_mm'")
            speed_mm[frame_indices] = track.speed_mm_by_level["smoothed"].astype(np.float32)
            heading_deg[frame_indices] = track.heading_degrees.astype(np.float32)
            pos_mm[frame_indices] = track.positions_mm.astype(np.float32)
            if track.angular_velocity_deg_s is not None:
                ang_vel[frame_indices] = track.angular_velocity_deg_s.astype(np.float32)
            if track.time_seconds is not None:
                time_s[frame_indices] = track.time_seconds.astype(np.float32)
            elif fps > 0:
                time_s[frame_indices] = frame_indices.astype(np.float32) / np.float32(fps)
            valid[frame_indices] = True
            if track.detection_source is not None:
                det_src[frame_indices] = track.detection_source.astype(np.int8)
            else:
                det_src[frame_indices] = 0
            if "smoothed" in track.frame_path_distance_mm_by_level:
                frame_path_distance_mm = np.zeros(n_frames, dtype=np.float32)
                frame_path_distance_mm[frame_indices] = track.frame_path_distance_mm_by_level["smoothed"].astype(np.float32)
            if track.cumulative_path_distance_mm is not None:
                sparse_cumulative = track.cumulative_path_distance_mm.astype(np.float32)
                cumulative_path_distance_mm = np.zeros(n_frames, dtype=np.float32)
                fill_positions = np.searchsorted(frame_indices, np.arange(n_frames), side="right") - 1
                has_previous = fill_positions >= 0
                cumulative_path_distance_mm[has_previous] = sparse_cumulative[fill_positions[has_previous]]

        tracks.append(DenseTrack(
            fish_id=fish_id,
            speed_mm=speed_mm,
            heading_deg=heading_deg,
            positions_mm=pos_mm,
            angular_velocity=ang_vel,
            time_seconds=time_s,
            valid=valid,
            detection_source=det_src,
            frame_path_distance_smoothed_mm=frame_path_distance_mm,
            cumulative_path_distance_mm=cumulative_path_distance_mm,
        ))

    upstream_lineage = _snapshot_upstream_lineage(kin_group)

    console.print(
        f"  Loaded {len(tracks)} track(s) from {run_path}/ "
        f"({n_frames} frames)"
    )
    return tracks, run_name, n_frames, upstream_lineage


# ---------------------------------------------------------------------------
# Stimulus event parsing
# ---------------------------------------------------------------------------


def _decode_text_value(value: Any) -> str:
    """Decode zarr string scalars, bytes, or fixed-width uint8 rows."""
    return decode_null_terminated_text(value)


def _decode_text_array(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 2 and arr.dtype.kind in ("u", "i"):
        return np.array([_decode_text_value(row) for row in arr], dtype=object)
    return np.array([_decode_text_value(value) for value in arr], dtype=object)


def _event_column(events_group: zarr.Group, *names: str) -> Optional[np.ndarray]:
    for name in names:
        if name in events_group:
            return np.asarray(events_group[name][:])
    return None


def _load_event_columns(
    root: zarr.Group,
    events_group: zarr.Group,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read legacy or modern stimulus event columns.

    Legacy Palette stimulus runs store explicit ``event_name``, ``step_index``,
    and ``stimulus_mode`` columns. Modern Citrus imports store
    ``event_type_id``, ``current_step_index``, and ``stimulus_mode_id`` plus
    enum tables. Return normalized event names, step indices, stimulus mode
    IDs, and camera frames.
    """
    event_names_raw = _event_column(events_group, "event_name")
    if event_names_raw is not None:
        event_names = _decode_text_array(event_names_raw)
    else:
        event_type_ids = _event_column(events_group, "event_type_id", "event_type")
        if event_type_ids is None:
            raise ValueError("Stimulus events lack event_name and event_type_id columns")
        event_type_map = load_event_types(root)
        event_names = np.array(
            [event_type_map.get(int(event_id), f"UNKNOWN_{int(event_id)}") for event_id in event_type_ids],
            dtype=object,
        )

    step_indices = _event_column(events_group, "step_index", "current_step_index")
    if step_indices is None:
        raise ValueError("Stimulus events lack step_index/current_step_index column")

    stimulus_modes = _event_column(events_group, "stimulus_mode", "stimulus_mode_id")
    if stimulus_modes is None:
        stimulus_modes = np.zeros(len(event_names), dtype=np.int32)

    camera_frames = _event_column(events_group, "camera_frame_id", "camera_frame_num")
    if camera_frames is None:
        raise ValueError("Stimulus events lack camera_frame_id column")

    lengths = {
        len(event_names),
        len(step_indices),
        len(stimulus_modes),
        len(camera_frames),
    }
    if len(lengths) != 1:
        raise ValueError("Stimulus event columns have inconsistent lengths")

    return (
        event_names,
        np.asarray(step_indices, dtype=np.int32),
        np.asarray(stimulus_modes, dtype=np.int32),
        np.asarray(camera_frames, dtype=np.int64),
    )


def _attr_int(attrs: Mapping[str, Any], *names: str, default: int = 0) -> int:
    for name in names:
        if name not in attrs:
            continue
        value = attrs.get(name)
        if isinstance(value, np.generic):
            value = value.item()
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return default


def _attr_float(attrs: Mapping[str, Any], *names: str, default: float = 0.0) -> float:
    for name in names:
        if name not in attrs:
            continue
        value = attrs.get(name)
        if isinstance(value, np.generic):
            value = value.item()
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return default


def _decode_protocol_params_json(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="ignore")
    if isinstance(raw, np.generic):
        raw = raw.item()
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
        return payload if isinstance(payload, dict) else {}
    return raw if isinstance(raw, dict) else {}


def _parse_canonical_stimulus_steps(stim_group: zarr.Group) -> List[ProtocolStep]:
    """Parse import-materialized ``stimulus_runs/<run>/steps`` metadata."""

    if "steps" not in stim_group:
        return []
    steps_group = stim_group["steps"]
    step_names = sorted(
        (name for name in steps_group.group_keys() if re.fullmatch(r"step_-?\d+", name)),
        key=lambda name: int(name.split("_", 1)[1]),
    )
    steps: List[ProtocolStep] = []
    for step_name in step_names:
        group = steps_group[step_name]
        attrs = group.attrs
        index = _attr_int(attrs, "step_index", default=int(step_name.split("_", 1)[1]))
        mode_id = _attr_int(attrs, "stimulus_mode_id", default=0)
        mode_name = str(attrs.get("stimulus_mode", f"UNKNOWN_{mode_id}"))
        start_frame = _attr_int(attrs, "start_camera_frame", "start_frame", default=0)
        end_frame = _attr_int(attrs, "end_camera_frame", "end_frame", default=start_frame + 1)
        duration_s = _attr_float(attrs, "duration_s", default=0.0)
        stimulus_params = _decode_protocol_params_json(attrs.get("raw_protocol_params_json"))
        for mode_group_name in ("moving_grating", "concentric_grating", "looming_dot"):
            if mode_group_name in group:
                stimulus_params[mode_group_name] = dict(group[mode_group_name].attrs)
        steps.append(ProtocolStep(
            index=index,
            name=str(attrs.get("step_name", step_name)),
            stimulus_mode=mode_name,
            stimulus_mode_id=mode_id,
            start_frame=start_frame,
            end_frame=end_frame,
            duration_s=duration_s,
            stimulus_params=stimulus_params,
        ))
    return steps


def parse_protocol_steps(
    root: zarr.Group,
    *,
    stimulus_run: Optional[str] = None,
    fps: float = 30.0,
    console: Optional[Console] = None,
) -> Tuple[List[ProtocolStep], str, Optional[Dict[str, Any]]]:
    """Extract protocol steps from stimulus run events.

    Returns
    -------
    steps : list[ProtocolStep]
        Ordered by step_index.
    run_name : str
        Resolved stimulus run name.
    protocol : dict or None
        Parsed protocol JSON if available.
    """
    console = console or Console()

    stim_group, run_name = resolve_zarr_run(
        root, "analysis/stimulus_runs", run_name=stimulus_run,
    )

    # Parse protocol JSON for stimulus params per step.
    protocol: Optional[Dict[str, Any]] = None
    proto_raw = stim_group.attrs.get("protocol_json")
    if proto_raw:
        if isinstance(proto_raw, bytes):
            proto_raw = proto_raw.decode("utf-8")
        try:
            protocol = json.loads(proto_raw)
        except (json.JSONDecodeError, TypeError):
            pass

    canonical_steps = _parse_canonical_stimulus_steps(stim_group)
    if canonical_steps:
        console.print(
            f"  Parsed {len(canonical_steps)} canonical protocol step(s) "
            f"from stimulus_runs/{run_name}/steps/"
        )
        return canonical_steps, run_name, protocol

    # Read events (columnar format).
    events_group = stim_group["events"]
    event_names, step_indices, stimulus_modes, camera_frames = _load_event_columns(root, events_group)

    # Pair STEP_START / STEP_END events by step_index.
    starts: Dict[int, int] = {}   # step_index → camera_frame
    ends: Dict[int, int] = {}
    modes: Dict[int, int] = {}

    for i in range(len(event_names)):
        name = str(event_names[i]).strip()
        si = int(step_indices[i])
        cf = int(camera_frames[i])
        mode = int(stimulus_modes[i])

        if name == EVENT_STEP_START:
            starts[si] = cf
            modes[si] = mode
        elif name == EVENT_STEP_END:
            ends[si] = cf

    if not starts:
        raise ValueError(
            f"No STEP_START events found in stimulus_runs/{run_name}/events/"
        )

    # Build step list.
    protocol_steps_list: Optional[list] = None
    if protocol:
        protocol_steps_list = protocol.get("steps") or protocol.get("protocol_steps")

    # Load stimulus mode enum from zarr (source of truth), fallback to hardcoded.
    stimulus_modes_map = load_stimulus_modes(root)

    steps: List[ProtocolStep] = []
    for si in sorted(starts):
        start_frame = starts[si]
        end_frame = ends.get(si, start_frame + 1)
        mode_id = modes.get(si, 0)
        mode_name = stimulus_modes_map.get(mode_id, f"UNKNOWN_{mode_id}")

        duration_s = (end_frame - start_frame) / fps if fps > 0 else 0.0

        # Extract step name and stimulus params from protocol JSON.
        step_name = f"step_{si}"
        stimulus_params: Dict[str, Any] = {}
        if protocol_steps_list and 0 <= si < len(protocol_steps_list):
            pstep = protocol_steps_list[si]
            step_name = pstep.get("name", pstep.get("step_name", step_name))
            stimulus_params = {
                k: v for k, v in pstep.items()
                if k not in ("name", "step_name", "step_index")
            }

        steps.append(ProtocolStep(
            index=si,
            name=step_name,
            stimulus_mode=mode_name,
            stimulus_mode_id=mode_id,
            start_frame=start_frame,
            end_frame=end_frame,
            duration_s=duration_s,
            stimulus_params=stimulus_params,
        ))

    console.print(
        f"  Parsed {len(steps)} protocol step(s) from stimulus_runs/{run_name}/"
    )
    return steps, run_name, protocol


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def _distance_for_window(track: DenseTrack, start_frame: int, end_frame: int) -> float:
    """Return gap-aware distance for the half-open frame interval [start, end).

    ``track_kinematics`` stores frame path-distance at the destination frame:
    sample ``f`` represents movement from ``f - 1`` to ``f``. For a step
    starting at frame ``s``, the first in-step path increment is therefore at
    ``s + 1``.
    """
    n_frames = int(track.valid.shape[0])
    start = max(0, min(int(start_frame), n_frames))
    end = max(start, min(int(end_frame), n_frames))
    if end - start <= 1:
        return 0.0

    first_path_frame = start + 1

    if track.frame_path_distance_smoothed_mm is not None:
        values = track.frame_path_distance_smoothed_mm[first_path_frame:end]
        return float(np.nansum(values))

    if track.cumulative_path_distance_mm is not None:
        cumulative = track.cumulative_path_distance_mm
        delta = float(cumulative[end - 1] - cumulative[start])
        return max(0.0, delta) if np.isfinite(delta) else 0.0

    # Backward-compatible fallback for tests or older track_kinematics runs:
    # only count adjacent valid dense frames, never jumps across missing frames.
    current_frames = np.arange(first_path_frame, end, dtype=np.int64)
    if current_frames.size == 0:
        return 0.0
    adjacent_valid = track.valid[current_frames] & track.valid[current_frames - 1]
    if not np.any(adjacent_valid):
        return 0.0
    frames = current_frames[adjacent_valid]
    deltas = track.positions_mm[frames] - track.positions_mm[frames - 1]
    return float(np.sum(np.linalg.norm(deltas, axis=1)))


def compute_global_metrics(
    tracks: Sequence[DenseTrack],
    fps: float,
    moving_threshold: float,
) -> Dict[str, np.ndarray]:
    """Recording-wide per-fish movement summary."""
    n_fish = len(tracks)
    fish_id = np.array([t.fish_id for t in tracks], dtype=np.int32)
    total_distance = np.zeros(n_fish, dtype=np.float32)
    mean_speed = np.zeros(n_fish, dtype=np.float32)
    total_active = np.zeros(n_fish, dtype=np.float32)
    fraction_moving = np.zeros(n_fish, dtype=np.float32)

    for i, t in enumerate(tracks):
        v = t.valid
        n_valid = int(v.sum())
        if n_valid == 0:
            continue
        speeds = t.speed_mm[v]
        total_distance[i] = _distance_for_window(t, 0, t.valid.shape[0])
        mean_speed[i] = float(np.mean(speeds))
        moving_mask = speeds > moving_threshold
        fraction_moving[i] = float(moving_mask.sum()) / n_valid
        total_active[i] = float(moving_mask.sum()) / fps if fps > 0 else 0.0

    return {
        "fish_id": fish_id,
        "total_distance_mm": total_distance,
        "mean_speed_mm_s": mean_speed,
        "total_active_s": total_active,
        "fraction_moving": fraction_moving,
    }


def compute_step_base_metrics(
    tracks: Sequence[DenseTrack],
    step: ProtocolStep,
    fps: float,
    moving_threshold: float,
) -> Dict[str, np.ndarray]:
    """Per-fish base movement metrics for one protocol step."""
    n_fish = len(tracks)
    sf, ef = step.start_frame, step.end_frame

    fish_id = np.array([t.fish_id for t in tracks], dtype=np.int32)
    total_distance = np.zeros(n_fish, dtype=np.float32)
    mean_speed = np.zeros(n_fish, dtype=np.float32)
    median_speed = np.zeros(n_fish, dtype=np.float32)
    max_speed = np.zeros(n_fish, dtype=np.float32)
    frac_moving = np.zeros(n_fish, dtype=np.float32)
    coverage = np.zeros(n_fish, dtype=np.float32)

    step_len = max(ef - sf, 1)

    for i, t in enumerate(tracks):
        v = t.valid[sf:ef]
        n_valid = int(v.sum())
        coverage[i] = float(n_valid) / step_len

        if n_valid == 0:
            continue

        speeds = t.speed_mm[sf:ef][v]
        total_distance[i] = _distance_for_window(t, sf, ef)
        mean_speed[i] = float(np.mean(speeds))
        median_speed[i] = float(np.median(speeds))
        max_speed[i] = float(np.max(speeds))
        moving = speeds > moving_threshold
        frac_moving[i] = float(moving.sum()) / n_valid

    return {
        "fish_id": fish_id,
        "total_distance_mm": total_distance,
        "mean_speed_mm_s": mean_speed,
        "median_speed_mm_s": median_speed,
        "max_speed_mm_s": max_speed,
        "fraction_moving": frac_moving,
        "coverage": coverage,
    }


# ---------------------------------------------------------------------------
# Bout loading and metrics
# ---------------------------------------------------------------------------


@dataclass
class BoutEntry:
    """One swim bout from detect_bouts_multi_level."""

    fish_id: int
    bout_id: int
    start_frame: int
    end_frame: int
    duration_s: float
    mean_speed: float
    peak_physical_speed: float


def _read_first_bout_column(
    bouts: np.ndarray,
    names: Sequence[str],
    *,
    dtype: np.dtype | type = np.float64,
) -> np.ndarray:
    """Read the first available bout column from current or legacy schemas."""

    field_names = bouts.dtype.names or ()
    for name in names:
        if name in field_names:
            return np.asarray(bouts[name]).astype(dtype)
    expected = ", ".join(names)
    raise ValueError(f"Bouts group is missing expected column; tried: {expected}")


def load_bout_data(
    root: zarr.Group,
    *,
    bout_run: Optional[str] = None,
    console: Optional[Console] = None,
) -> Tuple[Dict[int, List[BoutEntry]], str]:
    """Load swim bouts from the selected logical swim-bout candidate.

    Reads through ``swim_bout_io`` so callers do not depend on the physical
    ``analysis/swim_bout_runs/<run>/<speed_level>`` layout.

    Returns
    -------
    bouts_by_fish : dict[int, list[BoutEntry]]
        Bouts keyed by fish_id (track_id).
    run_name : str
        Resolved bout run name.
    """
    console = console or Console()

    payload = load_default_swim_bout_tables(root, run_name=bout_run or "latest")
    run_name = payload.run_name
    speed_level = payload.signal.speed_level
    track_id = payload.candidate.track_id
    if track_id is None:
        track_id = 0

    bouts = payload.bouts
    field_names = bouts.dtype.names or ()
    bout_ids = np.asarray(bouts["bout_id"]) if "bout_id" in field_names else np.array([], dtype=np.int32)
    n_bouts = len(bout_ids)

    if n_bouts == 0:
        return {}, run_name

    start_frames = np.asarray(bouts["start_frame"]).astype(np.int64)
    end_frames = np.asarray(bouts["end_frame"]).astype(np.int64)
    durations = np.asarray(bouts["duration_s"]).astype(np.float64)
    mean_speeds = _read_first_bout_column(
        bouts,
        ("mean_speed_mm_s", "mean_speed"),
    )
    peak_physical_speeds = _read_first_bout_column(
        bouts,
        ("peak_physical_speed_mm_s", "peak_speed_mm_s", "peak_speed"),
    )

    bouts_by_fish: Dict[int, List[BoutEntry]] = {}
    bout_list = bouts_by_fish.setdefault(track_id, [])
    for j in range(n_bouts):
        bout_list.append(BoutEntry(
            fish_id=track_id,
            bout_id=int(bout_ids[j]),
            start_frame=int(start_frames[j]),
            end_frame=int(end_frames[j]),
            duration_s=float(durations[j]),
            mean_speed=float(mean_speeds[j]),
            peak_physical_speed=float(peak_physical_speeds[j]),
        ))

    console.print(
        f"  Loaded {n_bouts} bout(s) from swim_bout_runs/{run_name} "
        f"(logical signal {speed_level})"
    )
    return bouts_by_fish, run_name


def compute_step_bout_metrics(
    bouts_by_fish: Dict[int, List[BoutEntry]],
    fish_ids: Sequence[int],
    step: ProtocolStep,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Per-fish and per-bout metrics for one protocol step.

    Returns
    -------
    per_fish : dict
        num_bouts, mean_bout_duration_s, mean_interbout_interval_s per fish.
    per_bout : dict
        fish_id, bout_id, start_frame, end_frame, duration_s,
        mean_speed_mm_s, peak_physical_speed_mm_s for all bouts in this step.
    """
    sf, ef = step.start_frame, step.end_frame
    n_fish = len(fish_ids)

    num_bouts = np.zeros(n_fish, dtype=np.int32)
    mean_bout_dur = np.zeros(n_fish, dtype=np.float32)
    mean_ibi = np.zeros(n_fish, dtype=np.float32)

    # Collect all bouts in this step across all fish.
    all_fish_id: List[int] = []
    all_bout_id: List[int] = []
    all_start: List[int] = []
    all_end: List[int] = []
    all_dur: List[float] = []
    all_mean_spd: List[float] = []
    all_peak_spd: List[float] = []

    for i, fid in enumerate(fish_ids):
        fish_bouts = bouts_by_fish.get(fid, [])
        # Filter bouts overlapping with this step's frame range.
        step_bouts = [
            b for b in fish_bouts
            if b.start_frame < ef and b.end_frame >= sf
        ]
        step_bouts.sort(key=lambda b: b.start_frame)

        num_bouts[i] = len(step_bouts)
        if step_bouts:
            durations = [b.duration_s for b in step_bouts]
            mean_bout_dur[i] = float(np.mean(durations))

            if len(step_bouts) >= 2:
                ibis = [
                    (step_bouts[k + 1].start_frame - step_bouts[k].end_frame)
                    for k in range(len(step_bouts) - 1)
                ]
                # Convert frame gaps to seconds using step duration / frames.
                step_frames = max(ef - sf, 1)
                frame_to_s = step.duration_s / step_frames if step_frames > 0 else 0.0
                mean_ibi[i] = float(np.mean(ibis)) * frame_to_s

        for b in step_bouts:
            all_fish_id.append(fid)
            all_bout_id.append(b.bout_id)
            all_start.append(b.start_frame)
            all_end.append(b.end_frame)
            all_dur.append(b.duration_s)
            all_mean_spd.append(b.mean_speed)
            all_peak_spd.append(b.peak_physical_speed)

    per_fish = {
        "num_bouts": num_bouts,
        "mean_bout_duration_s": mean_bout_dur,
        "mean_interbout_interval_s": mean_ibi,
    }
    per_bout = {
        "fish_id": np.array(all_fish_id, dtype=np.int32),
        "bout_id": np.array(all_bout_id, dtype=np.int32),
        "start_frame": np.array(all_start, dtype=np.int64),
        "end_frame": np.array(all_end, dtype=np.int64),
        "duration_s": np.array(all_dur, dtype=np.float32),
        "mean_speed_mm_s": np.array(all_mean_spd, dtype=np.float32),
        "peak_physical_speed_mm_s": np.array(all_peak_spd, dtype=np.float32),
    }
    return per_fish, per_bout


# ---------------------------------------------------------------------------
# Concentric grating metrics
# ---------------------------------------------------------------------------

_CONCENTRIC_GRATING = "CONCENTRIC_GRATING"
_LOOMING_DOT = "LOOMING_DOT"


def compute_concentric_per_frame(
    tracks: Sequence[DenseTrack],
    step: ProtocolStep,
    center_mm: Tuple[float, float],
    fps: float,
) -> Dict[str, np.ndarray]:
    """Per-frame radial decomposition for one CONCENTRIC_GRATING step."""
    sf, ef = step.start_frame, step.end_frame
    n_step = max(ef - sf, 1)
    n_fish = len(tracks)

    frame_indices = np.arange(sf, ef, dtype=np.int64)
    valid = np.zeros((n_fish, n_step), dtype=bool)
    det_src = np.full((n_fish, n_step), -1, dtype=np.int8)
    dist_center = np.zeros((n_fish, n_step), dtype=np.float32)
    radial_heading = np.zeros((n_fish, n_step), dtype=np.float32)
    radial_speed = np.zeros((n_fish, n_step), dtype=np.float32)
    tangential_speed = np.zeros((n_fish, n_step), dtype=np.float32)

    cx, cy = center_mm

    for i, t in enumerate(tracks):
        pos = t.positions_mm[sf:ef]          # (n_step, 2)
        heading_deg = t.heading_deg[sf:ef]
        speed = t.speed_mm[sf:ef]
        v = t.valid[sf:ef]

        valid[i] = v
        det_src[i] = t.detection_source[sf:ef]

        # Vector from fish to center.
        dx = cx - pos[:, 0]
        dy = cy - pos[:, 1]
        dist = np.sqrt(dx**2 + dy**2)
        dist_center[i] = dist

        # Angle from fish heading to the direction toward center.
        angle_to_center_rad = np.arctan2(dy, dx)
        heading_rad = np.deg2rad(heading_deg)
        radial_diff = heading_rad - angle_to_center_rad
        # Wrap to [-pi, pi].
        radial_diff = (radial_diff + np.pi) % (2 * np.pi) - np.pi
        radial_heading[i] = np.rad2deg(radial_diff)

        # Decompose speed into radial and tangential.
        radial_speed[i] = speed * np.cos(radial_diff)   # positive = toward center
        tangential_speed[i] = speed * np.sin(radial_diff)

        # Zero out invalid frames.
        inv = ~v
        dist_center[i, inv] = 0.0
        radial_heading[i, inv] = 0.0
        radial_speed[i, inv] = 0.0
        tangential_speed[i, inv] = 0.0

    return {
        "frame_indices": frame_indices,
        "valid": valid,
        "detection_source": det_src,
        "distance_to_center_mm": dist_center,
        "radial_heading_angle_deg": radial_heading,
        "radial_speed_mm_s": radial_speed,
        "tangential_speed_mm_s": tangential_speed,
    }


def compute_concentric_per_fish(
    per_frame: Dict[str, np.ndarray],
    tracks: Sequence[DenseTrack],
    step: ProtocolStep,
    fps: float,
    center_threshold_mm: float = 2.0,
) -> Dict[str, np.ndarray]:
    """Per-fish centering summary for one CONCENTRIC_GRATING step."""
    n_fish = len(tracks)
    sf, ef = step.start_frame, step.end_frame

    dist = per_frame["distance_to_center_mm"]
    r_speed = per_frame["radial_speed_mm_s"]
    r_heading = per_frame["radial_heading_angle_deg"]
    pf_valid = per_frame["valid"]

    mean_dist = np.zeros(n_fish, dtype=np.float32)
    initial_dist = np.zeros(n_fish, dtype=np.float32)
    final_dist = np.zeros(n_fish, dtype=np.float32)
    min_dist = np.zeros(n_fish, dtype=np.float32)
    net_radial = np.zeros(n_fish, dtype=np.float32)
    frac_approach = np.zeros(n_fish, dtype=np.float32)
    mean_radial_cos = np.zeros(n_fish, dtype=np.float32)
    time_to_center = np.full(n_fish, np.nan, dtype=np.float32)
    frac_near = np.zeros(n_fish, dtype=np.float32)
    mean_r_speed = np.zeros(n_fish, dtype=np.float32)
    mean_t_speed = np.zeros(n_fish, dtype=np.float32)

    t_speed = per_frame["tangential_speed_mm_s"]

    for i, t in enumerate(tracks):
        v = pf_valid[i]
        n_valid = int(v.sum())
        if n_valid == 0:
            continue

        d_v = dist[i][v]
        rs_v = r_speed[i][v]
        rh_v = r_heading[i][v]
        ts_v = t_speed[i][v]

        mean_dist[i] = float(np.mean(d_v))
        min_dist[i] = float(np.min(d_v))
        mean_r_speed[i] = float(np.mean(rs_v))
        mean_t_speed[i] = float(np.mean(np.abs(ts_v)))

        # Initial and final distance (first/last valid frame).
        initial_dist[i] = float(d_v[0])
        final_dist[i] = float(d_v[-1])
        net_radial[i] = float(d_v[-1] - d_v[0])  # negative = moved toward center

        # Fraction approaching (radial_speed > 0 means toward center in our convention).
        frac_approach[i] = float((rs_v > 0).sum()) / n_valid

        # Mean radial heading cosine.
        rh_rad = np.deg2rad(rh_v)
        mean_radial_cos[i] = float(np.mean(np.cos(rh_rad)))

        # Fraction near center.
        frac_near[i] = float((d_v < center_threshold_mm).sum()) / n_valid

        # Time to center: first frame where distance < threshold.
        full_dist = dist[i]
        full_valid = pf_valid[i]
        for f_idx in range(full_dist.shape[0]):
            if full_valid[f_idx] and full_dist[f_idx] < center_threshold_mm:
                time_to_center[i] = float(f_idx) / fps if fps > 0 else 0.0
                break

    return {
        "mean_distance_to_center_mm": mean_dist,
        "initial_distance_to_center_mm": initial_dist,
        "final_distance_to_center_mm": final_dist,
        "min_distance_to_center_mm": min_dist,
        "net_radial_displacement_mm": net_radial,
        "fraction_approaching": frac_approach,
        "mean_radial_heading_cos": mean_radial_cos,
        "time_to_center_s": time_to_center,
        "fraction_near_center": frac_near,
        "mean_radial_speed_mm_s": mean_r_speed,
        "mean_tangential_speed_mm_s": mean_t_speed,
    }


def compute_concentric_time_series(
    per_frame: Dict[str, np.ndarray],
    tracks: Sequence[DenseTrack],
    step: ProtocolStep,
    fps: float,
    bin_size_s: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Binned temporal dynamics for one CONCENTRIC_GRATING step."""
    sf, ef = step.start_frame, step.end_frame
    n_step = max(ef - sf, 1)
    n_fish = len(tracks)

    bin_size_frames = max(1, int(bin_size_s * fps))
    n_bins = max(1, (n_step + bin_size_frames - 1) // bin_size_frames)

    bin_center_s = np.zeros(n_bins, dtype=np.float32)
    dist_binned = np.zeros((n_fish, n_bins), dtype=np.float32)
    rspeed_binned = np.zeros((n_fish, n_bins), dtype=np.float32)
    rcos_binned = np.zeros((n_fish, n_bins), dtype=np.float32)
    fappr_binned = np.zeros((n_fish, n_bins), dtype=np.float32)

    pf_dist = per_frame["distance_to_center_mm"]
    pf_rspeed = per_frame["radial_speed_mm_s"]
    pf_rheading = per_frame["radial_heading_angle_deg"]
    pf_valid = per_frame["valid"]

    for b in range(n_bins):
        bs = b * bin_size_frames
        be = min(bs + bin_size_frames, n_step)
        bin_center_s[b] = ((bs + be) / 2.0) / fps if fps > 0 else 0.0

        for i, t in enumerate(tracks):
            v = pf_valid[i, bs:be]
            n_v = int(v.sum())
            if n_v == 0:
                continue
            dist_binned[i, b] = float(np.mean(pf_dist[i, bs:be][v]))
            rspeed_binned[i, b] = float(np.mean(pf_rspeed[i, bs:be][v]))
            rh_rad = np.deg2rad(pf_rheading[i, bs:be][v])
            rcos_binned[i, b] = float(np.mean(np.cos(rh_rad)))
            fappr_binned[i, b] = float((pf_rspeed[i, bs:be][v] > 0).sum()) / n_v

    return {
        "bin_center_s": bin_center_s,
        "distance_to_center_mm": dist_binned,
        "radial_speed_mm_s": rspeed_binned,
        "radial_heading_cos": rcos_binned,
        "fraction_approaching": fappr_binned,
    }


# ---------------------------------------------------------------------------
# Looming dot metrics
# ---------------------------------------------------------------------------


def _load_loom_onset_events(
    root: zarr.Group,
    stimulus_run: str,
    steps: Sequence[ProtocolStep],
) -> Dict[int, List[int]]:
    """Read loom onset events and group by step.

    The first onset for each loom step is the step start frame.
    Subsequent onsets come from LOOM_AUTO_REPEAT_TRIGGER or LOOM_MANUAL_START.
    """
    stim_group = root[f"analysis/stimulus_runs/{stimulus_run}"]
    events_group = stim_group["events"]

    event_names, step_indices, _, camera_frames = _load_event_columns(root, events_group)

    loom_step_indices = {
        s.index for s in steps if s.stimulus_mode == _LOOMING_DOT
    }

    onsets: Dict[int, List[int]] = {}
    for si in loom_step_indices:
        step = next(s for s in steps if s.index == si)
        onsets[si] = [step.start_frame]

    for i in range(len(event_names)):
        name = str(event_names[i]).strip()
        si = int(step_indices[i])
        cf = int(camera_frames[i])
        if si not in loom_step_indices:
            continue
        if name in (EVENT_LOOM_AUTO_REPEAT, EVENT_LOOM_MANUAL_START):
            onsets.setdefault(si, []).append(cf)

    for si in onsets:
        onsets[si] = sorted(set(onsets[si]))

    return onsets


def reconstruct_loom_trials(
    step: ProtocolStep,
    onset_frames: List[int],
    fps: float,
) -> List[LoomTrial]:
    """Build LoomTrial objects from onset frames and step parameters.

    Each trial's offset = onset + loom_duration_sec * fps, clamped to step end.
    """
    params = flatten_stimulus_params(step.stimulus_params)
    loom_duration_s = float(params.get("loom_duration_sec", 0.0))
    if loom_duration_s <= 0 or fps <= 0:
        return []

    loom_duration_frames = int(round(loom_duration_s * fps))

    trials: List[LoomTrial] = []
    for idx, onset in enumerate(onset_frames):
        if onset >= step.end_frame:
            continue
        offset = min(onset + loom_duration_frames, step.end_frame)
        trials.append(LoomTrial(
            trial_index=idx,
            onset_frame=onset,
            offset_frame=offset,
            duration_s=(offset - onset) / fps,
        ))
    return trials


def resolve_loom_center_mm(
    step: ProtocolStep,
    calibration: Dict[str, Any],
) -> Optional[Tuple[float, float]]:
    """Resolve the looming dot center in camera-space mm.

    For target_side=0, uses texture center (179, 179) via homography.
    Falls back to arena center if homography is unavailable.
    """
    params = flatten_stimulus_params(step.stimulus_params)

    # Pre-computed mm coordinates.
    cx_mm = params.get("loom_center_x_mm")
    cy_mm = params.get("loom_center_y_mm")
    if cx_mm is not None and cy_mm is not None:
        return (float(cx_mm), float(cy_mm))

    target_side = int(params.get("target_side", 0))

    if target_side == 0:
        # Center of 358×358 texture → homography → camera mm.
        if (calibration["homography"] is not None
                and calibration["pixel_to_mm"] is not None):
            texture_center = np.array([179.0, 179.0])
            pt_mm = projector_to_camera_mm(
                texture_center,
                calibration["homography"],
                calibration["pixel_to_mm"],
            )
            return (float(pt_mm[0]), float(pt_mm[1]))

    # Fallback: arena center.
    if (calibration["arena_center_px"] is not None
            and calibration["pixel_to_mm"] is not None):
        cx_cam, cy_cam = calibration["arena_center_px"]
        return (cx_cam * calibration["pixel_to_mm"],
                cy_cam * calibration["pixel_to_mm"])

    return None


def compute_loom_per_frame(
    tracks: Sequence[DenseTrack],
    step: ProtocolStep,
    trials: List[LoomTrial],
    center_mm: Tuple[float, float],
    fps: float,
    pixels_per_mm_projector: Optional[float],
    z_eff_mm: Optional[float],
) -> Dict[str, np.ndarray]:
    """Per-frame loom annotation and distance metrics for one LOOMING_DOT step."""
    sf, ef = step.start_frame, step.end_frame
    n_step = max(ef - sf, 1)
    n_fish = len(tracks)

    frame_indices = np.arange(sf, ef, dtype=np.int64)
    valid = np.zeros((n_fish, n_step), dtype=bool)
    det_src = np.full((n_fish, n_step), -1, dtype=np.int8)
    loom_active = np.zeros(n_step, dtype=bool)
    trial_idx = np.full(n_step, -1, dtype=np.int32)
    loom_radius_px = np.zeros(n_step, dtype=np.float32)
    vis_angle = np.zeros(n_step, dtype=np.float32)
    distance_to_loom = np.zeros((n_fish, n_step), dtype=np.float32)

    params = flatten_stimulus_params(step.stimulus_params)
    start_radius = float(params.get("start_radius_px", 0.0))
    end_radius = float(params.get("end_radius_px", 0.0))
    loom_duration_s = float(params.get("loom_duration_sec", 1.0))

    cx, cy = center_mm

    # Mark loom-active frames and reconstruct radius.
    for trial in trials:
        t_start = max(trial.onset_frame - sf, 0)
        t_end = min(trial.offset_frame - sf, n_step)
        for f in range(t_start, t_end):
            loom_active[f] = True
            trial_idx[f] = trial.trial_index
            t_within = (f - t_start) / fps if fps > 0 else 0.0
            frac = min(t_within / loom_duration_s, 1.0) if loom_duration_s > 0 else 0.0
            loom_radius_px[f] = start_radius + (end_radius - start_radius) * frac

    # Visual angle from radius.
    if pixels_per_mm_projector is not None and pixels_per_mm_projector > 0:
        radius_mm = loom_radius_px / pixels_per_mm_projector
        if z_eff_mm is not None and z_eff_mm > 0:
            vis_angle = visual_angle_deg(radius_mm, z_eff_mm).astype(np.float32)

    # Per-fish distance to loom center.
    for i, t in enumerate(tracks):
        pos = t.positions_mm[sf:ef]
        v = t.valid[sf:ef]
        valid[i] = v
        det_src[i] = t.detection_source[sf:ef]

        dx = pos[:, 0] - cx
        dy = pos[:, 1] - cy
        distance_to_loom[i] = np.sqrt(dx**2 + dy**2)
        distance_to_loom[i, ~v] = 0.0

    return {
        "frame_indices": frame_indices,
        "valid": valid,
        "detection_source": det_src,
        "loom_active": loom_active,
        "trial_index": trial_idx,
        "loom_radius_px": loom_radius_px,
        "visual_angle_deg": vis_angle,
        "distance_to_loom_mm": distance_to_loom,
    }


def compute_loom_per_trial_per_fish(
    tracks: Sequence[DenseTrack],
    step: ProtocolStep,
    trials: List[LoomTrial],
    center_mm: Tuple[float, float],
    per_frame: Dict[str, np.ndarray],
    fps: float,
    escape_speed_threshold_mm_s: float = 30.0,
    escape_window_s: float = 5.0,
) -> Dict[str, np.ndarray]:
    """Per-trial per-fish escape metrics.  Returns arrays shaped [n_fish, n_trials]."""
    sf = step.start_frame
    n_fish = len(tracks)
    n_trials = len(trials)

    escaped = np.zeros((n_fish, n_trials), dtype=bool)
    escape_latency_s = np.full((n_fish, n_trials), np.nan, dtype=np.float32)
    escape_latency_frames = np.full((n_fish, n_trials), -1, dtype=np.int32)
    peak_escape_speed = np.zeros((n_fish, n_trials), dtype=np.float32)
    distance_at_escape = np.full((n_fish, n_trials), np.nan, dtype=np.float32)
    vis_angle_at_escape = np.full((n_fish, n_trials), np.nan, dtype=np.float32)
    escape_heading = np.full((n_fish, n_trials), np.nan, dtype=np.float32)

    escape_window_frames = int(round(escape_window_s * fps)) if fps > 0 else 0

    pf_vis_angle = per_frame["visual_angle_deg"]
    pf_distance = per_frame["distance_to_loom_mm"]

    for j, trial in enumerate(trials):
        win_start = trial.onset_frame
        win_end = min(trial.onset_frame + escape_window_frames, step.end_frame)

        for i, t in enumerate(tracks):
            speed_win = t.speed_mm[win_start:win_end]
            valid_win = t.valid[win_start:win_end]

            above = valid_win & (speed_win >= escape_speed_threshold_mm_s)
            hit_indices = np.where(above)[0]
            if len(hit_indices) == 0:
                continue

            esc_offset = int(hit_indices[0])
            esc_frame = win_start + esc_offset

            escaped[i, j] = True
            escape_latency_frames[i, j] = esc_offset
            escape_latency_s[i, j] = float(esc_offset) / fps if fps > 0 else 0.0

            # Peak speed in a 0.5s burst window after escape initiation.
            burst_end = min(esc_frame + max(1, int(0.5 * fps)), step.end_frame)
            burst_speed = t.speed_mm[esc_frame:burst_end]
            burst_valid = t.valid[esc_frame:burst_end]
            if burst_valid.any():
                peak_escape_speed[i, j] = float(np.max(burst_speed[burst_valid]))

            # Distance and visual angle at escape frame.
            pf_idx = esc_frame - sf
            if 0 <= pf_idx < pf_distance.shape[1]:
                distance_at_escape[i, j] = float(pf_distance[i, pf_idx])
            if 0 <= pf_idx < pf_vis_angle.shape[0]:
                vis_angle_at_escape[i, j] = float(pf_vis_angle[pf_idx])

            if t.valid[esc_frame]:
                escape_heading[i, j] = float(t.heading_deg[esc_frame])

    return {
        "escaped": escaped,
        "escape_latency_s": escape_latency_s,
        "escape_latency_frames": escape_latency_frames,
        "peak_escape_speed_mm_s": peak_escape_speed,
        "distance_at_escape_mm": distance_at_escape,
        "visual_angle_at_escape_deg": vis_angle_at_escape,
        "escape_heading_deg": escape_heading,
    }


def compute_loom_per_fish(
    per_trial_per_fish: Dict[str, np.ndarray],
    n_trials: int,
) -> Dict[str, np.ndarray]:
    """Per-fish summary across all loom trials.  Returns arrays shaped [n_fish]."""
    esc = per_trial_per_fish["escaped"]
    lat = per_trial_per_fish["escape_latency_s"]
    peak_spd = per_trial_per_fish["peak_escape_speed_mm_s"]
    dist_esc = per_trial_per_fish["distance_at_escape_mm"]
    vis_esc = per_trial_per_fish["visual_angle_at_escape_deg"]

    n_fish = esc.shape[0]

    n_escapes = np.sum(esc, axis=1).astype(np.int32)
    escape_prob = (n_escapes.astype(np.float32) / n_trials) if n_trials > 0 else np.zeros(n_fish, dtype=np.float32)

    mean_lat = np.full(n_fish, np.nan, dtype=np.float32)
    median_lat = np.full(n_fish, np.nan, dtype=np.float32)
    mean_peak = np.zeros(n_fish, dtype=np.float32)
    mean_dist = np.full(n_fish, np.nan, dtype=np.float32)
    mean_vis = np.full(n_fish, np.nan, dtype=np.float32)
    hab_idx = np.full(n_fish, np.nan, dtype=np.float32)

    for i in range(n_fish):
        mask = esc[i]
        n_esc = int(mask.sum())
        if n_esc == 0:
            continue

        e_lat = lat[i][mask]
        mean_lat[i] = float(np.nanmean(e_lat))
        median_lat[i] = float(np.nanmedian(e_lat))
        mean_peak[i] = float(np.mean(peak_spd[i][mask]))

        d = dist_esc[i][mask]
        if not np.all(np.isnan(d)):
            mean_dist[i] = float(np.nanmean(d))
        v = vis_esc[i][mask]
        if not np.all(np.isnan(v)):
            mean_vis[i] = float(np.nanmean(v))

        # Habituation: linear slope of latency vs trial index.
        if n_esc >= 2:
            trial_idx = np.where(mask)[0].astype(np.float64)
            e_lat_f = e_lat.astype(np.float64)
            finite = np.isfinite(e_lat_f)
            if finite.sum() >= 2:
                x = trial_idx[finite]
                y = e_lat_f[finite]
                xc = x - x.mean()
                yc = y - y.mean()
                var_x = np.dot(xc, xc)
                if var_x > 1e-12:
                    hab_idx[i] = float(np.dot(xc, yc) / var_x)

    return {
        "n_escape_responses": n_escapes,
        "escape_probability": escape_prob,
        "mean_escape_latency_s": mean_lat,
        "median_escape_latency_s": median_lat,
        "mean_peak_escape_speed_mm_s": mean_peak,
        "mean_distance_at_escape_mm": mean_dist,
        "mean_visual_angle_at_escape_deg": mean_vis,
        "habituation_index": hab_idx,
    }


def compute_loom_time_series(
    tracks: Sequence[DenseTrack],
    step: ProtocolStep,
    trials: List[LoomTrial],
    per_frame: Dict[str, np.ndarray],
    fps: float,
    pre_onset_s: float = 1.0,
    post_onset_s: float = 5.0,
    bin_size_s: float = 0.1,
) -> Dict[str, np.ndarray]:
    """Trial-aligned, trial-averaged PSTH for one LOOMING_DOT step."""
    sf = step.start_frame
    n_fish = len(tracks)

    total_window_s = pre_onset_s + post_onset_s
    n_bins = max(1, int(round(total_window_s / bin_size_s)))
    bin_size_frames = max(1, int(round(bin_size_s * fps)))

    trial_time_s = np.array(
        [-pre_onset_s + (b + 0.5) * bin_size_s for b in range(n_bins)],
        dtype=np.float32,
    )

    pre_onset_frames = int(round(pre_onset_s * fps))
    pf_distance = per_frame["distance_to_loom_mm"]
    pf_valid = per_frame["valid"]

    speed_sum = np.zeros((n_fish, n_bins), dtype=np.float64)
    dist_sum = np.zeros((n_fish, n_bins), dtype=np.float64)
    count = np.zeros((n_fish, n_bins), dtype=np.int32)

    for trial in trials:
        win_start = trial.onset_frame - pre_onset_frames
        for b in range(n_bins):
            abs_start = max(0, win_start + b * bin_size_frames)
            abs_end = min(win_start + (b + 1) * bin_size_frames, step.end_frame)
            if abs_start >= abs_end:
                continue
            for i, t in enumerate(tracks):
                v = t.valid[abs_start:abs_end]
                n_v = int(v.sum())
                if n_v == 0:
                    continue
                speed_sum[i, b] += float(np.sum(t.speed_mm[abs_start:abs_end][v]))
                pf_s = abs_start - sf
                pf_e = abs_end - sf
                if 0 <= pf_s and pf_e <= pf_distance.shape[1]:
                    pf_v = pf_valid[i, pf_s:pf_e]
                    n_pf = int(pf_v.sum())
                    if n_pf > 0:
                        dist_sum[i, b] += float(np.sum(pf_distance[i, pf_s:pf_e][pf_v]))
                count[i, b] += n_v

    mean_speed = np.zeros((n_fish, n_bins), dtype=np.float32)
    mean_dist = np.zeros((n_fish, n_bins), dtype=np.float32)
    nz = count > 0
    mean_speed[nz] = (speed_sum[nz] / count[nz]).astype(np.float32)
    mean_dist[nz] = (dist_sum[nz] / count[nz]).astype(np.float32)

    return {
        "trial_time_s": trial_time_s,
        "mean_speed_mm_s": mean_speed,
        "mean_distance_to_loom_mm": mean_dist,
    }


# ---------------------------------------------------------------------------
# Zarr output
# ---------------------------------------------------------------------------


@dataclass
class ConcentricStepData:
    """Concentric grating metric outputs for one CONCENTRIC_GRATING step."""

    per_frame: Dict[str, np.ndarray]
    per_fish: Dict[str, np.ndarray]
    time_series: Dict[str, np.ndarray]
    radial_omr: Optional[ConcentricRadialOMRStepData] = None


@dataclass
class GratingStepData:
    """Grating metric outputs for one MOVING_GRATING step."""

    per_frame: Dict[str, np.ndarray]
    per_fish: Dict[str, np.ndarray]
    time_series: Dict[str, np.ndarray]
    omr: Optional[OMRStepData] = None


@dataclass
class LoomTrial:
    """One loom presentation within a LOOMING_DOT step."""

    trial_index: int
    onset_frame: int      # camera frame where expansion starts
    offset_frame: int     # camera frame where expansion ends (exclusive)
    duration_s: float


@dataclass
class LoomStepData:
    """Loom metric outputs for one LOOMING_DOT step."""

    trials: List[LoomTrial]
    per_frame: Dict[str, np.ndarray]
    per_trial_per_fish: Dict[str, np.ndarray]
    per_fish: Dict[str, np.ndarray]
    time_series: Dict[str, np.ndarray]


STIMULUS_RESPONSE_LAYOUT_HIERARCHICAL_V1 = "hierarchical_v1"
STIMULUS_RESPONSE_LAYOUT_COMPACT_V2 = "compact_tabular_v2"
STIMULUS_RESPONSE_LAYOUT_DEFAULT = STIMULUS_RESPONSE_LAYOUT_COMPACT_V2
STIMULUS_RESPONSE_SCHEMA_ID = "palette.stimulus_response"
STIMULUS_RESPONSE_SCHEMA_VERSION = 2


def _scalar_for_record(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _scalar_for_record(value.item())
        return _json_safe_attr_value(value.tolist())
    return _json_safe_attr_value(value)


def _mapping_to_rows(
    mapping: Mapping[str, np.ndarray],
    *,
    constants: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    constants = constants or {}
    arrays: dict[str, np.ndarray] = {}
    n_rows: int | None = None
    for name, values in mapping.items():
        arr = np.asarray(values)
        if arr.ndim != 1:
            continue
        if n_rows is None:
            n_rows = int(arr.shape[0])
        if int(arr.shape[0]) != n_rows:
            continue
        arrays[str(name)] = arr
    if n_rows is None:
        return []
    rows: list[dict[str, Any]] = []
    for idx in range(n_rows):
        row = {str(key): _scalar_for_record(value) for key, value in constants.items()}
        for name, arr in arrays.items():
            row[name] = _scalar_for_record(arr[idx])
        rows.append(row)
    return rows


def _dtype_for_values(values: Sequence[Any]) -> np.dtype:
    non_null = [value for value in values if value is not None]
    if not non_null:
        return np.dtype("float64")
    if any(isinstance(value, (str, bytes, bytearray)) for value in non_null):
        max_len = max(
            len(
                (
                    bytes(value)
                    if isinstance(value, (bytes, bytearray))
                    else str(value).encode("utf-8")
                )
            )
            for value in non_null
        )
        width = max(1, min(512, 2 ** max_len.bit_length()))
        return np.dtype(f"S{width}")
    if any(isinstance(value, float) for value in non_null):
        return np.dtype("float64")
    if all(isinstance(value, (bool, np.bool_)) for value in non_null):
        return np.dtype("bool")
    if all(isinstance(value, (int, np.integer, bool, np.bool_)) for value in non_null):
        return np.dtype("int64")
    return np.dtype("S512")


def _coerce_record_value(value: Any, dtype: np.dtype) -> Any:
    if value is None:
        if dtype.kind == "f":
            return np.nan
        if dtype.kind in {"i", "u"}:
            return 0
        if dtype.kind == "b":
            return False
        return b""
    if dtype.kind == "S":
        if isinstance(value, (bytes, bytearray)):
            return bytes(value)
        return str(value).encode("utf-8")
    return value


def _rows_to_structured(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    field_names: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            key_str = str(key)
            if key_str not in seen:
                seen.add(key_str)
                field_names.append(key_str)
    dtype = [
        (name, _dtype_for_values([row.get(name) for row in rows]))
        for name in field_names
    ]
    out = np.zeros(len(rows), dtype=dtype)
    for row_idx, row in enumerate(rows):
        for name, field_dtype in dtype:
            out[name][row_idx] = _coerce_record_value(row.get(name), np.dtype(field_dtype))
    return out


def _write_rows_table(
    parent: zarr.Group,
    name: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    attrs: Mapping[str, Any] | None = None,
) -> bool:
    if not rows:
        return False
    safe_attrs = _json_safe_attrs(attrs or {})
    data = _rows_to_structured(rows)
    write_columnar_dataset(parent, name, data, attrs=safe_attrs)
    return True


def _step_constants(step: ProtocolStep) -> dict[str, Any]:
    return {
        "step_index": int(step.index),
        "step_name": step.name,
        "stimulus_mode": step.stimulus_mode,
        "stimulus_mode_id": int(step.stimulus_mode_id),
        "start_frame": int(step.start_frame),
        "end_frame": int(step.end_frame),
        "duration_s": float(step.duration_s),
    }


def _step_attr_record(step: ProtocolStep) -> dict[str, Any]:
    record = _step_constants(step)
    record["stimulus_params"] = step.stimulus_params
    return record


def _write_stimulus_response_compact_v2(
    run_group: zarr.Group,
    *,
    global_metrics: Mapping[str, np.ndarray],
    steps: Sequence[ProtocolStep],
    step_metrics: Sequence[Mapping[str, np.ndarray]],
    frame_annotations: Optional[Mapping[str, np.ndarray]],
    step_bout_metrics: Optional[Sequence[Tuple[Mapping[str, np.ndarray], Mapping[str, np.ndarray]]]],
    step_grating_data: Optional[Mapping[int, GratingStepData]],
    step_concentric_data: Optional[Mapping[int, ConcentricStepData]],
    step_loom_data: Optional[Mapping[int, LoomStepData]],
    global_omr_metrics: Optional[Mapping[str, np.ndarray]],
) -> None:
    """Write the compact-tabular-v2 physical layout.

    This first compact slice intentionally omits high-volume per-frame and
    time-series tables. Those remain a later performance/object-count decision.
    """

    table_names: list[str] = []
    omitted_tables = [
        "grating_per_frame",
        "grating_time_series",
        "concentric_per_frame",
        "concentric_time_series",
        "concentric_radial_omr_per_frame",
        "looming_per_frame",
        "looming_time_series",
    ]

    def write_table(name: str, rows: Sequence[Mapping[str, Any]], attrs: Mapping[str, Any] | None = None) -> None:
        if _write_rows_table(run_group, name, rows, attrs=attrs):
            table_names.append(name)

    step_rows = [
        {
            **_step_constants(step),
            "stimulus_params_json": json.dumps(
                _json_safe_attr_value(step.stimulus_params),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
        }
        for step in steps
    ]
    write_table("step_index", step_rows, {"table_role": "step_index"})

    write_table(
        "global_per_fish",
        _mapping_to_rows(global_metrics),
        {"table_role": "global_per_fish"},
    )
    if global_omr_metrics is not None:
        write_table(
            "global_omr_per_fish",
            _mapping_to_rows(global_omr_metrics),
            {
                "table_role": "global_omr_per_fish",
                "method_version": OMR_METHOD_VERSION,
                "scope": "aggregate_across_moving_grating_steps",
            },
        )
    if frame_annotations is not None:
        write_table(
            "frame_annotations",
            _mapping_to_rows(frame_annotations),
            {"table_role": "frame_annotations"},
        )

    step_per_fish_rows: list[dict[str, Any]] = []
    step_per_bout_rows: list[dict[str, Any]] = []
    grating_per_fish_rows: list[dict[str, Any]] = []
    moving_omr_per_fish_rows: list[dict[str, Any]] = []
    moving_omr_per_bout_rows: list[dict[str, Any]] = []
    moving_omr_windows_rows: list[dict[str, Any]] = []
    moving_omr_early_rows: list[dict[str, Any]] = []
    moving_omr_attrs: list[dict[str, Any]] = []
    concentric_per_fish_rows: list[dict[str, Any]] = []
    radial_omr_per_fish_rows: list[dict[str, Any]] = []
    radial_omr_per_bout_rows: list[dict[str, Any]] = []
    radial_omr_windows_rows: list[dict[str, Any]] = []
    radial_omr_early_rows: list[dict[str, Any]] = []
    radial_omr_attrs: list[dict[str, Any]] = []
    looming_trials_rows: list[dict[str, Any]] = []
    looming_per_trial_per_fish_rows: list[dict[str, Any]] = []
    looming_per_fish_rows: list[dict[str, Any]] = []
    looming_attrs: list[dict[str, Any]] = []

    for idx, (step, metrics) in enumerate(zip(steps, step_metrics)):
        constants = _step_constants(step)
        per_fish_metrics = dict(metrics)
        if step_bout_metrics is not None and idx < len(step_bout_metrics):
            bout_per_fish, bout_per_bout = step_bout_metrics[idx]
            per_fish_metrics.update(bout_per_fish)
            step_per_bout_rows.extend(_mapping_to_rows(bout_per_bout, constants=constants))
        step_per_fish_rows.extend(_mapping_to_rows(per_fish_metrics, constants=constants))

        if step_grating_data is not None and step.index in step_grating_data:
            gd = step_grating_data[step.index]
            grating_constants = {**constants, "stimulus_family": "moving_grating"}
            grating_per_fish_rows.extend(_mapping_to_rows(gd.per_fish, constants=grating_constants))
            if gd.omr is not None:
                omr_constants = {
                    **constants,
                    "stimulus_family": "moving_grating",
                    "metric_family": "moving_grating_omr",
                }
                moving_omr_attrs.append({"step_index": step.index, "attrs": gd.omr.attrs})
                moving_omr_per_fish_rows.extend(_mapping_to_rows(gd.omr.per_fish, constants=omr_constants))
                moving_omr_per_bout_rows.extend(_mapping_to_rows(gd.omr.per_bout, constants=omr_constants))
                moving_omr_windows_rows.extend(_mapping_to_rows(gd.omr.windows, constants=omr_constants))
                moving_omr_early_rows.extend(_mapping_to_rows(gd.omr.early_windows, constants=omr_constants))

        if step_concentric_data is not None and step.index in step_concentric_data:
            cd = step_concentric_data[step.index]
            conc_constants = {**constants, "stimulus_family": "concentric_grating"}
            concentric_per_fish_rows.extend(_mapping_to_rows(cd.per_fish, constants=conc_constants))
            if cd.radial_omr is not None:
                radial_constants = {
                    **constants,
                    "stimulus_family": "concentric_grating",
                    "metric_family": "concentric_radial_omr",
                }
                radial_omr_attrs.append({"step_index": step.index, "attrs": cd.radial_omr.attrs})
                radial_omr_per_fish_rows.extend(_mapping_to_rows(cd.radial_omr.per_fish, constants=radial_constants))
                radial_omr_per_bout_rows.extend(_mapping_to_rows(cd.radial_omr.per_bout, constants=radial_constants))
                radial_omr_windows_rows.extend(_mapping_to_rows(cd.radial_omr.windows, constants=radial_constants))
                radial_omr_early_rows.extend(_mapping_to_rows(cd.radial_omr.early_windows, constants=radial_constants))

        if step_loom_data is not None and step.index in step_loom_data:
            ld = step_loom_data[step.index]
            loom_constants = {**constants, "stimulus_family": "looming", "metric_family": "looming"}
            looming_attrs.append({
                "step_index": step.index,
                "attrs": {
                    "n_trials": len(ld.trials),
                },
            })
            for trial in ld.trials:
                looming_trials_rows.append({
                    **loom_constants,
                    "trial_index": int(trial.trial_index),
                    "onset_frame": int(trial.onset_frame),
                    "offset_frame": int(trial.offset_frame),
                    "trial_duration_s": float(trial.duration_s),
                })
            looming_per_trial_per_fish_rows.extend(
                _mapping_to_rows(ld.per_trial_per_fish, constants=loom_constants)
            )
            looming_per_fish_rows.extend(_mapping_to_rows(ld.per_fish, constants=loom_constants))

    write_table("step_per_fish", step_per_fish_rows, {"table_role": "step_per_fish"})
    write_table("step_per_bout", step_per_bout_rows, {"table_role": "step_per_bout"})
    write_table("grating_per_fish", grating_per_fish_rows, {"table_role": "grating_per_fish"})
    write_table("moving_grating_omr_per_fish", moving_omr_per_fish_rows, {"table_role": "moving_grating_omr_per_fish"})
    write_table("moving_grating_omr_per_bout", moving_omr_per_bout_rows, {"table_role": "moving_grating_omr_per_bout"})
    write_table("moving_grating_omr_windows", moving_omr_windows_rows, {"table_role": "moving_grating_omr_windows"})
    write_table("moving_grating_omr_early_windows", moving_omr_early_rows, {"table_role": "moving_grating_omr_early_windows"})
    write_table("concentric_per_fish", concentric_per_fish_rows, {"table_role": "concentric_per_fish"})
    write_table("concentric_radial_omr_per_fish", radial_omr_per_fish_rows, {"table_role": "concentric_radial_omr_per_fish"})
    write_table("concentric_radial_omr_per_bout", radial_omr_per_bout_rows, {"table_role": "concentric_radial_omr_per_bout"})
    write_table("concentric_radial_omr_windows", radial_omr_windows_rows, {"table_role": "concentric_radial_omr_windows"})
    write_table("concentric_radial_omr_early_windows", radial_omr_early_rows, {"table_role": "concentric_radial_omr_early_windows"})
    write_table("looming_trials", looming_trials_rows, {"table_role": "looming_trials"})
    write_table("looming_per_trial_per_fish", looming_per_trial_per_fish_rows, {"table_role": "looming_per_trial_per_fish"})
    write_table("looming_per_fish", looming_per_fish_rows, {"table_role": "looming_per_fish"})

    run_group.attrs.update(_json_safe_attrs({
        "compact_table_names": table_names,
        "compact_omitted_tables": omitted_tables,
        "step_attrs": [_step_attr_record(step) for step in steps],
        "moving_grating_omr_attrs": moving_omr_attrs,
        "concentric_radial_omr_attrs": radial_omr_attrs,
        "looming_attrs": looming_attrs,
    }))


def build_frame_annotations(
    steps: Sequence[ProtocolStep],
    n_frames: int,
) -> Dict[str, np.ndarray]:
    """Build recording-wide per-frame annotation arrays.

    Returns step_index and stimulus_mode_id for every frame in the recording.
    Frames not covered by any step get -1.
    """
    step_index = np.full(n_frames, -1, dtype=np.int32)
    stimulus_mode_id = np.full(n_frames, -1, dtype=np.int32)

    for step in steps:
        sf = max(0, step.start_frame)
        ef = min(step.end_frame, n_frames)
        if sf < ef:
            step_index[sf:ef] = step.index
            stimulus_mode_id[sf:ef] = step.stimulus_mode_id

    return {
        "step_index": step_index,
        "stimulus_mode_id": stimulus_mode_id,
    }


def write_stimulus_response_run(
    root: zarr.Group,
    *,
    global_metrics: Dict[str, np.ndarray],
    steps: List[ProtocolStep],
    step_metrics: List[Dict[str, np.ndarray]],
    frame_annotations: Optional[Dict[str, np.ndarray]] = None,
    step_bout_metrics: Optional[List[Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]]] = None,
    step_grating_data: Optional[Dict[int, GratingStepData]] = None,
    step_concentric_data: Optional[Dict[int, ConcentricStepData]] = None,
    step_loom_data: Optional[Dict[int, LoomStepData]] = None,
    global_omr_metrics: Optional[Dict[str, np.ndarray]] = None,
    source_kinematics_run: str,
    source_kinematics_type: str,
    source_stimulus_run: str,
    source_bout_run: Optional[str] = None,
    upstream_lineage: Optional[Dict[str, Any]] = None,
    parameters: Dict[str, Any],
    run_name: Optional[str] = None,
    overwrite: bool = False,
    layout: str = STIMULUS_RESPONSE_LAYOUT_DEFAULT,
    console: Optional[Console] = None,
) -> str:
    """Write stimulus response run to zarr."""
    console = console or Console()
    if layout not in {STIMULUS_RESPONSE_LAYOUT_HIERARCHICAL_V1, STIMULUS_RESPONSE_LAYOUT_COMPACT_V2}:
        raise ValueError(
            "Unsupported stimulus_response layout "
            f"'{layout}'. Expected {STIMULUS_RESPONSE_LAYOUT_HIERARCHICAL_V1} "
            f"or {STIMULUS_RESPONSE_LAYOUT_COMPACT_V2}."
        )

    analysis = root.require_group("analysis")
    parent = require_runs_parent(analysis, "stimulus_response_runs")

    if run_name is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_name = f"stimulus_response_{ts}"

    if run_name in parent and not overwrite:
        raise ValueError(f"Stimulus response run '{run_name}' already exists.")
    if run_name in parent:
        del parent[run_name]

    run_group = parent.create_group(run_name)
    mark_run_started(run_group, run_name=run_name, stage="stimulus_response")

    # Provenance.
    fish_ids = global_metrics["fish_id"].tolist()
    inputs_dict: Dict[str, Any] = {
        "source_track_kinematics_run": (
            f"analysis/track_kinematics_runs/{source_kinematics_type}/"
            f"{source_kinematics_run}"
        ),
        "source_stimulus_run": (
            f"analysis/stimulus_runs/{source_stimulus_run}"
        ),
    }
    if source_bout_run is not None:
        inputs_dict["source_bout_run"] = (
            f"analysis/swim_bout_runs/{source_bout_run}"
        )

    # Archive-level identity: which recording produced this analysis?
    archive_identity: Dict[str, Any] = {}
    root_attrs = root.attrs if hasattr(root, "attrs") else {}
    for key in ("source_video_path", "source_video", "session_uuid"):
        val = root_attrs.get(key)
        if val is not None:
            archive_identity[key] = str(val) if isinstance(val, bytes) else val
    # Check analysis group for session_uuid if not at root.
    if "session_uuid" not in archive_identity:
        analysis_attrs = root.get("analysis", {})
        if hasattr(analysis_attrs, "attrs"):
            val = analysis_attrs.attrs.get("session_uuid")
            if val is not None:
                archive_identity["session_uuid"] = str(val) if isinstance(val, bytes) else val
    if archive_identity:
        inputs_dict["archive"] = archive_identity
    if upstream_lineage:
        inputs_dict["upstream_lineage"] = upstream_lineage

    safe_parameters = _json_safe_attr_value(parameters)
    safe_inputs = _json_safe_attr_value(inputs_dict)
    provenance = build_stage_provenance(
        stage="stimulus_response",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        parameters=safe_parameters,
        inputs=safe_inputs,
        git=get_git_info(),
        command=" ".join(sys.argv),
    )
    write_stage_provenance(run_group, provenance)

    run_attrs: Dict[str, Any] = {
        "schema_id": STIMULUS_RESPONSE_SCHEMA_ID,
        "schema_version": STIMULUS_RESPONSE_SCHEMA_VERSION,
        "layout": layout,
        "source_track_kinematics_run": source_kinematics_run,
        "source_track_kinematics_type": source_kinematics_type,
        "source_stimulus_run": source_stimulus_run,
        "n_steps": len(steps),
        "n_fish": len(fish_ids),
        "fish_ids": fish_ids,
    }
    if source_bout_run is not None:
        run_attrs["source_bout_run"] = source_bout_run
    if archive_identity:
        run_attrs["archive_identity"] = archive_identity
    run_group.attrs.update(_json_safe_attrs(run_attrs))
    write_best_effort_run_lineage_attrs(run_group, run_family="stimulus_response_run")

    if layout == STIMULUS_RESPONSE_LAYOUT_COMPACT_V2:
        _write_stimulus_response_compact_v2(
            run_group,
            global_metrics=global_metrics,
            steps=steps,
            step_metrics=step_metrics,
            frame_annotations=frame_annotations,
            step_bout_metrics=step_bout_metrics,
            step_grating_data=step_grating_data,
            step_concentric_data=step_concentric_data,
            step_loom_data=step_loom_data,
            global_omr_metrics=global_omr_metrics,
        )
        console.print(
            f"  Wrote stimulus_response_runs/{run_name}/ "
            f"({len(steps)} steps, {len(fish_ids)} fish, layout {layout})"
        )
        mark_run_complete(run_group, parent_group=parent, run_name=run_name)
        return run_name

    # Global group.
    global_group = run_group.create_group("global")
    for name, arr in global_metrics.items():
        global_group.create_array(name, data=arr, overwrite=True)
    if global_omr_metrics is not None:
        omr_group = global_group.create_group("omr")
        omr_group.attrs.update(_json_safe_attrs({
            "method_version": OMR_METHOD_VERSION,
            "scope": "aggregate_across_moving_grating_steps",
        }))
        omr_per_fish = omr_group.create_group("per_fish")
        for name, arr in global_omr_metrics.items():
            omr_per_fish.create_array(name, data=arr, overwrite=True)

    # Frame annotations (recording-wide stimulus context).
    if frame_annotations is not None:
        frames_group = run_group.create_group("frames")
        for name, arr in frame_annotations.items():
            frames_group.create_array(name, data=arr, overwrite=True)

    # Per-step groups.
    steps_parent = run_group.create_group("steps")
    for idx, (step, metrics) in enumerate(zip(steps, step_metrics)):
        step_group = steps_parent.create_group(f"step_{step.index}")
        step_group.attrs.update(_json_safe_attrs({
            "step_index": step.index,
            "step_name": step.name,
            "stimulus_mode": step.stimulus_mode,
            "stimulus_mode_id": step.stimulus_mode_id,
            "start_frame": step.start_frame,
            "end_frame": step.end_frame,
            "duration_s": step.duration_s,
            "stimulus_params": step.stimulus_params,
        }))

        per_fish = step_group.create_group("per_fish")
        for name, arr in metrics.items():
            per_fish.create_array(name, data=arr, overwrite=True)

        # Bout metrics (optional).
        if step_bout_metrics is not None and idx < len(step_bout_metrics):
            bout_per_fish, bout_per_bout = step_bout_metrics[idx]
            for name, arr in bout_per_fish.items():
                per_fish.create_array(name, data=arr, overwrite=True)
            if bout_per_bout["fish_id"].size > 0:
                per_bout_group = step_group.create_group("per_bout")
                for name, arr in bout_per_bout.items():
                    per_bout_group.create_array(name, data=arr, overwrite=True)

        # Grating metrics (MOVING_GRATING steps only).
        if step_grating_data is not None and step.index in step_grating_data:
            gd = step_grating_data[step.index]
            grating_group = step_group.create_group("grating")

            pf_group = grating_group.create_group("per_frame")
            for name, arr in gd.per_frame.items():
                pf_group.create_array(name, data=arr, overwrite=True)

            gpf_group = grating_group.create_group("per_fish")
            for name, arr in gd.per_fish.items():
                gpf_group.create_array(name, data=arr, overwrite=True)

            ts_group = grating_group.create_group("time_series")
            for name, arr in gd.time_series.items():
                ts_group.create_array(name, data=arr, overwrite=True)

            if gd.omr is not None:
                omr_group = grating_group.create_group("omr")
                omr_group.attrs.update(_json_safe_attrs(gd.omr.attrs))

                omr_per_fish = omr_group.create_group("per_fish")
                for name, arr in gd.omr.per_fish.items():
                    omr_per_fish.create_array(name, data=arr, overwrite=True)

                omr_per_bout = omr_group.create_group("per_bout")
                for name, arr in gd.omr.per_bout.items():
                    omr_per_bout.create_array(name, data=arr, overwrite=True)

                omr_windows = omr_group.create_group("windows")
                for name, arr in gd.omr.windows.items():
                    omr_windows.create_array(name, data=arr, overwrite=True)

                omr_early_windows = omr_group.create_group("early_windows")
                for name, arr in gd.omr.early_windows.items():
                    omr_early_windows.create_array(name, data=arr, overwrite=True)

        # Concentric grating metrics (CONCENTRIC_GRATING steps only).
        if step_concentric_data is not None and step.index in step_concentric_data:
            cd = step_concentric_data[step.index]
            conc_group = step_group.create_group("concentric_grating")

            cpf_group = conc_group.create_group("per_frame")
            for name, arr in cd.per_frame.items():
                cpf_group.create_array(name, data=arr, overwrite=True)

            cpfish_group = conc_group.create_group("per_fish")
            for name, arr in cd.per_fish.items():
                cpfish_group.create_array(name, data=arr, overwrite=True)

            cts_group = conc_group.create_group("time_series")
            for name, arr in cd.time_series.items():
                cts_group.create_array(name, data=arr, overwrite=True)

            if cd.radial_omr is not None:
                radial_group = conc_group.create_group("radial_omr")
                radial_group.attrs.update(_json_safe_attrs(cd.radial_omr.attrs))

                radial_per_frame = radial_group.create_group("per_frame")
                for name, arr in cd.radial_omr.per_frame.items():
                    radial_per_frame.create_array(name, data=arr, overwrite=True)

                radial_per_fish = radial_group.create_group("per_fish")
                for name, arr in cd.radial_omr.per_fish.items():
                    radial_per_fish.create_array(name, data=arr, overwrite=True)

                radial_per_bout = radial_group.create_group("per_bout")
                for name, arr in cd.radial_omr.per_bout.items():
                    radial_per_bout.create_array(name, data=arr, overwrite=True)

                radial_windows = radial_group.create_group("windows")
                for name, arr in cd.radial_omr.windows.items():
                    radial_windows.create_array(name, data=arr, overwrite=True)

                radial_early_windows = radial_group.create_group("early_windows")
                for name, arr in cd.radial_omr.early_windows.items():
                    radial_early_windows.create_array(name, data=arr, overwrite=True)

        # Looming dot metrics (LOOMING_DOT steps only).
        if step_loom_data is not None and step.index in step_loom_data:
            ld = step_loom_data[step.index]
            loom_group = step_group.create_group("looming")
            loom_group.attrs.update(_json_safe_attrs({
                "n_trials": len(ld.trials),
                "escape_speed_threshold_mm_s": parameters.get("escape_speed_threshold_mm_s"),
                "escape_window_s": parameters.get("escape_window_s"),
            }))

            # Trial timing.
            trials_group = loom_group.create_group("trials")
            trials_group.create_array(
                "onset_frame",
                data=np.array([t.onset_frame for t in ld.trials], dtype=np.int64),
                overwrite=True,
            )
            trials_group.create_array(
                "offset_frame",
                data=np.array([t.offset_frame for t in ld.trials], dtype=np.int64),
                overwrite=True,
            )
            trials_group.create_array(
                "duration_s",
                data=np.array([t.duration_s for t in ld.trials], dtype=np.float32),
                overwrite=True,
            )

            lpf_group = loom_group.create_group("per_frame")
            for name, arr in ld.per_frame.items():
                lpf_group.create_array(name, data=arr, overwrite=True)

            ltpf_group = loom_group.create_group("per_trial_per_fish")
            for name, arr in ld.per_trial_per_fish.items():
                ltpf_group.create_array(name, data=arr, overwrite=True)

            lpfish_group = loom_group.create_group("per_fish")
            for name, arr in ld.per_fish.items():
                lpfish_group.create_array(name, data=arr, overwrite=True)

            lts_group = loom_group.create_group("time_series")
            for name, arr in ld.time_series.items():
                lts_group.create_array(name, data=arr, overwrite=True)

    console.print(
        f"  Wrote stimulus_response_runs/{run_name}/ "
        f"({len(steps)} steps, {len(fish_ids)} fish)"
    )
    mark_run_complete(run_group, parent_group=parent, run_name=run_name)
    return run_name


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-step stimulus response metrics.",
    )
    parser.add_argument("zarr_path", help="Path to the analysis Zarr archive.")
    parser.add_argument(
        "--track-kinematics-type",
        default="offline",
        choices=("online", "offline"),
        help="Track kinematics type (default: offline).",
    )
    parser.add_argument(
        "--track-kinematics-run",
        default=None,
        help="Track kinematics run name (default: latest).",
    )
    parser.add_argument(
        "--stimulus-run",
        default=None,
        help="Stimulus run name (default: latest).",
    )
    parser.add_argument(
        "--moving-threshold-mm-s",
        type=float,
        default=2.0,
        help="Speed threshold for 'moving' classification in mm/s (default: 2.0).",
    )
    parser.add_argument(
        "--bout-run",
        default=None,
        help="Swim bout run name (default: latest if available, omit to skip bouts).",
    )
    parser.add_argument(
        "--no-bouts",
        action="store_true",
        help="Skip bout integration even if bout data is available.",
    )
    parser.add_argument(
        "--camera-to-projector-offset-deg",
        type=float,
        default=0.0,
        help="Angular offset from camera to projector space in degrees (default: 0.0).",
    )
    parser.add_argument(
        "--bin-size-s",
        type=float,
        default=1.0,
        help="Time bin size for grating temporal dynamics in seconds (default: 1.0).",
    )
    parser.add_argument(
        "--follow-threshold",
        type=float,
        default=0.5,
        help="alignment_cos threshold for sustained following detection (default: 0.5).",
    )
    parser.add_argument(
        "--follow-window-s",
        type=float,
        default=1.0,
        help="Window duration for sustained following detection in seconds (default: 1.0).",
    )
    parser.add_argument(
        "--no-omr",
        action="store_true",
        help=(
            "Skip OMR responsiveness metrics for MOVING_GRATING and "
            "CONCENTRIC_GRATING steps."
        ),
    )
    parser.add_argument(
        "--omr-projection-deadzone",
        type=float,
        default=0.0,
        help="Deadzone for per-bout OMR projection scores (default: 0.0).",
    )
    parser.add_argument(
        "--omr-projection-speed-deadzone-mm-s",
        type=float,
        default=0.0,
        help="Deadzone for time-weighted OMR projected speed in mm/s (default: 0.0).",
    )
    parser.add_argument(
        "--omr-window-s",
        type=float,
        action="append",
        default=None,
        help=(
            "Non-overlapping OMR window length in seconds; may be repeated "
            "(default: 10, 30, 60 plus full step)."
        ),
    )
    parser.add_argument(
        "--omr-early-window-s",
        type=float,
        action="append",
        default=None,
        help=(
            "Early-response OMR window length in seconds from each grating onset; "
            "may be repeated (default: 5, 10)."
        ),
    )
    parser.add_argument(
        "--center-threshold-mm",
        type=float,
        default=2.0,
        help="Distance threshold for 'near center' in concentric grating analysis (default: 2.0).",
    )
    parser.add_argument(
        "--concentric-radial-singularity-epsilon-mm",
        type=float,
        default=0.5,
        help=(
            "Minimum radius from the concentric-grating center required for "
            "radial/tangential OMR metrics (default: 0.5 mm)."
        ),
    )
    # Looming dot parameters.
    parser.add_argument(
        "--escape-speed-threshold-mm-s",
        type=float,
        default=30.0,
        help="Speed threshold for escape detection in mm/s (default: 30.0, per Fernandes et al. 2021).",
    )
    parser.add_argument(
        "--escape-window-s",
        type=float,
        default=5.0,
        help="Max time after loom onset to look for escape response in seconds (default: 5.0).",
    )
    parser.add_argument(
        "--loom-pre-onset-s",
        type=float,
        default=1.0,
        help="Pre-stimulus window for loom PSTH in seconds (default: 1.0).",
    )
    parser.add_argument(
        "--loom-post-onset-s",
        type=float,
        default=5.0,
        help="Post-stimulus window for loom PSTH in seconds (default: 5.0).",
    )
    parser.add_argument(
        "--loom-bin-size-s",
        type=float,
        default=0.1,
        help="Bin size for loom PSTH in seconds (default: 0.1).",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Output run name (default: auto-timestamped).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing stimulus_response run with the same name.",
    )
    parser.add_argument(
        "--layout",
        choices=(STIMULUS_RESPONSE_LAYOUT_HIERARCHICAL_V1, STIMULUS_RESPONSE_LAYOUT_COMPACT_V2),
        default=STIMULUS_RESPONSE_LAYOUT_DEFAULT,
        help=(
            "Physical Zarr layout to write. compact_tabular_v2 is the default "
            "and intentionally omits high-volume per-frame/time-series compact tables."
        ),
    )
    parser.add_argument(
        "--write-zarr-artifacts",
        action="store_true",
        help=(
            "Persist review visualization artifacts under the stimulus_response "
            "run's visualizations group."
        ),
    )
    parser.add_argument(
        "--artifact-dpi",
        type=int,
        default=150,
        help="DPI for PNG artifacts written with --write-zarr-artifacts (default: 150).",
    )
    args = parser.parse_args(argv)

    console = Console()
    console.print(f"\n[bold]Stimulus Response Analysis[/bold]")
    console.print(f"  Archive: {args.zarr_path}")

    root = open_zarr_root(args.zarr_path, mode="a")

    # Load inputs.
    tracks, kin_run, n_frames, upstream_lineage = load_track_data(
        root,
        kinematics_type=args.track_kinematics_type,
        kinematics_run=args.track_kinematics_run,
        console=console,
    )

    fps_attr = root.get("analysis", {})
    kin_parent = f"analysis/track_kinematics_runs/{args.track_kinematics_type}"
    kin_group, _ = resolve_zarr_run(root, kin_parent, run_name=kin_run)
    fps = float(kin_group.attrs.get("fps", 30.0))

    steps, stim_run, protocol = parse_protocol_steps(
        root,
        stimulus_run=args.stimulus_run,
        fps=fps,
        console=console,
    )

    # Compute metrics.
    console.print("  Computing global metrics...")
    global_metrics = compute_global_metrics(
        tracks, fps, args.moving_threshold_mm_s,
    )

    # Load bout data (optional).
    bouts_by_fish: Optional[Dict[int, List[BoutEntry]]] = None
    bout_run_name: Optional[str] = None

    if not args.no_bouts:
        try:
            bouts_by_fish, bout_run_name = load_bout_data(
                root, bout_run=args.bout_run, console=console,
            )
        except (KeyError, ValueError) as exc:
            console.print(f"  [yellow]Bout data not available: {exc}[/yellow]")

    # Resolve concentric centers before parallel dispatch (needs zarr access).
    concentric_centers: Dict[int, Tuple[float, float]] = {}
    concentric_center_sources: Dict[int, str] = {}
    for step in steps:
        if step.stimulus_mode == _CONCENTRIC_GRATING:
            cg_attrs = (
                step.stimulus_params.get("concentric_grating", {})
                if isinstance(step.stimulus_params, dict) else {}
            )
            if not isinstance(cg_attrs, dict):
                cg_attrs = {}
            center_params = flatten_stimulus_params(step.stimulus_params)
            center_params.update(cg_attrs)
            center = resolve_concentric_center_mm(
                root,
                center_params,
                stimulus_run=stim_run,
            )
            if center is not None:
                concentric_centers[step.index] = center
                concentric_center_sources[step.index] = str(
                    cg_attrs.get("center_mm_source")
                    or cg_attrs.get("center_source")
                    or "resolve_concentric_center_mm"
                )
            else:
                console.print(
                    f"  [yellow]Warning: could not resolve center for step {step.index}; "
                    f"skipping concentric metrics.[/yellow]"
                )

    concentric_arena_radius_mm: Optional[float] = None
    if concentric_centers:
        _center_unused, arena_extent, _source_unused = _resolve_omr_arena_geometry_mm(
            root,
            stim_run,
            np.array([1.0, 0.0], dtype=np.float64),
        )
        concentric_arena_radius_mm = arena_extent

    # Resolve loom calibration and events before parallel dispatch.
    loom_onset_events: Dict[int, List[int]] = {}
    loom_centers: Dict[int, Tuple[float, float]] = {}
    loom_cal: Dict[str, Any] = {}
    has_loom_steps = any(s.stimulus_mode == _LOOMING_DOT for s in steps)
    if has_loom_steps:
        cal = load_calibration_transform(root, stimulus_run=stim_run)
        loom_cal = {
            "pixels_per_mm_projector": cal.get("pixels_per_mm_projector"),
            "z_eff_mm": cal.get("z_eff_mm"),
        }
        loom_onset_events = _load_loom_onset_events(root, stim_run, steps)
        for step in steps:
            if step.stimulus_mode == _LOOMING_DOT:
                center = resolve_loom_center_mm(step, cal)
                if center is not None:
                    loom_centers[step.index] = center
                else:
                    console.print(
                        f"  [yellow]Warning: could not resolve loom center for step "
                        f"{step.index}; skipping loom metrics.[/yellow]"
                    )
        if loom_cal.get("z_eff_mm") is None:
            console.print(
                "  [yellow]Warning: z_eff_mm not in calibration; "
                "visual angle will be zeros.[/yellow]"
            )

    # --- Parallel per-step computation via Dask ---

    fish_ids = [t.fish_id for t in tracks]

    def _compute_one_step(step: ProtocolStep) -> Dict[str, Any]:
        """Compute all metrics for one step (runs in a Dask worker)."""
        result: Dict[str, Any] = {
            "step_index": step.index,
            "base": compute_step_base_metrics(
                tracks, step, fps, args.moving_threshold_mm_s,
            ),
        }

        # Bout metrics.
        if bouts_by_fish is not None:
            result["bout"] = compute_step_bout_metrics(bouts_by_fish, fish_ids, step)

        # Grating metrics.
        if step.stimulus_mode == _MOVING_GRATING:
            grating_dir = resolve_grating_direction(step, args.camera_to_projector_offset_deg)
            grating_speed = resolve_grating_speed_mm_s(step)
            pf = compute_grating_per_frame(tracks, step, grating_dir, fps)
            gpf = compute_grating_per_fish(
                pf, tracks, step, fps,
                grating_dir_deg=grating_dir,
                grating_speed_mm_s=grating_speed,
                follow_threshold=args.follow_threshold,
                follow_window_s=args.follow_window_s,
            )
            ts = compute_grating_time_series(
                pf, tracks, step, fps,
                bin_size_s=args.bin_size_s,
                grating_speed_mm_s=grating_speed,
            )
            omr = None
            if not args.no_omr:
                arena_center_mm, arena_axis_extent_mm, arena_geometry_source = (
                    _resolve_omr_arena_geometry_mm(
                        root,
                        stim_run,
                        _grating_direction_vector(grating_dir),
                    )
                )
                omr = compute_step_omr_metrics(
                    tracks,
                    step,
                    grating_dir,
                    fps,
                    moving_threshold_mm_s=args.moving_threshold_mm_s,
                    bouts_by_fish=bouts_by_fish,
                    projection_deadzone=args.omr_projection_deadzone,
                    projection_speed_deadzone_mm_s=args.omr_projection_speed_deadzone_mm_s,
                    window_lengths_s=(
                        args.omr_window_s
                        if args.omr_window_s is not None
                        else OMR_DEFAULT_WINDOW_LENGTHS_S
                    ),
                    early_window_lengths_s=(
                        args.omr_early_window_s
                        if args.omr_early_window_s is not None
                        else OMR_DEFAULT_EARLY_RESPONSE_WINDOWS_S
                    ),
                    arena_center_mm=arena_center_mm,
                    arena_axis_extent_mm=arena_axis_extent_mm,
                    arena_geometry_source=arena_geometry_source,
                )
            result["grating"] = GratingStepData(
                per_frame=pf, per_fish=gpf, time_series=ts, omr=omr,
            )

        # Concentric grating metrics.
        if step.stimulus_mode == _CONCENTRIC_GRATING and step.index in concentric_centers:
            center = concentric_centers[step.index]
            cpf = compute_concentric_per_frame(tracks, step, center, fps)
            cpfish = compute_concentric_per_fish(
                cpf, tracks, step, fps,
                center_threshold_mm=args.center_threshold_mm,
            )
            cts = compute_concentric_time_series(
                cpf, tracks, step, fps,
                bin_size_s=args.bin_size_s,
            )
            radial_omr = None
            if not args.no_omr:
                radial_omr = compute_step_concentric_radial_omr_metrics(
                    tracks,
                    step,
                    center,
                    fps,
                    moving_threshold_mm_s=args.moving_threshold_mm_s,
                    bouts_by_fish=bouts_by_fish,
                    projection_deadzone=args.omr_projection_deadzone,
                    projection_speed_deadzone_mm_s=args.omr_projection_speed_deadzone_mm_s,
                    window_lengths_s=(
                        args.omr_window_s
                        if args.omr_window_s is not None
                        else CONCENTRIC_RADIAL_OMR_DEFAULT_WINDOW_LENGTHS_S
                    ),
                    early_window_lengths_s=(
                        args.omr_early_window_s
                        if args.omr_early_window_s is not None
                        else CONCENTRIC_RADIAL_OMR_DEFAULT_EARLY_RESPONSE_WINDOWS_S
                    ),
                    radial_singularity_epsilon_mm=args.concentric_radial_singularity_epsilon_mm,
                    arena_radius_mm=concentric_arena_radius_mm,
                    center_source=concentric_center_sources.get(
                        step.index,
                        "resolve_concentric_center_mm",
                    ),
                )
            result["concentric"] = ConcentricStepData(
                per_frame=cpf,
                per_fish=cpfish,
                time_series=cts,
                radial_omr=radial_omr,
            )

        # Looming dot metrics.
        if (step.stimulus_mode == _LOOMING_DOT
                and step.index in loom_onset_events
                and step.index in loom_centers):
            trials = reconstruct_loom_trials(
                step, loom_onset_events[step.index], fps,
            )
            if trials:
                center = loom_centers[step.index]
                lpf = compute_loom_per_frame(
                    tracks, step, trials, center, fps,
                    loom_cal.get("pixels_per_mm_projector"),
                    loom_cal.get("z_eff_mm"),
                )
                ltpf = compute_loom_per_trial_per_fish(
                    tracks, step, trials, center, lpf, fps,
                    escape_speed_threshold_mm_s=args.escape_speed_threshold_mm_s,
                    escape_window_s=args.escape_window_s,
                )
                lpfish = compute_loom_per_fish(ltpf, len(trials))
                lts = compute_loom_time_series(
                    tracks, step, trials, lpf, fps,
                    pre_onset_s=args.loom_pre_onset_s,
                    post_onset_s=args.loom_post_onset_s,
                    bin_size_s=args.loom_bin_size_s,
                )
                result["loom"] = LoomStepData(
                    trials=trials, per_frame=lpf,
                    per_trial_per_fish=ltpf, per_fish=lpfish,
                    time_series=lts,
                )

        return result

    import dask
    from dask import delayed
    from dask.diagnostics import ProgressBar

    console.print(f"  Computing metrics for {len(steps)} step(s) in parallel...")
    delayed_results = [delayed(_compute_one_step)(step) for step in steps]
    with ProgressBar():
        computed_results = dask.compute(*delayed_results, scheduler="threads")

    # Unpack results into the structures write_stimulus_response_run expects.
    step_metrics: List[Dict[str, np.ndarray]] = []
    step_bout_metrics: Optional[List[Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]]] = None
    step_grating_data: Optional[Dict[int, GratingStepData]] = None
    step_concentric_data: Optional[Dict[int, ConcentricStepData]] = None
    step_loom_data: Optional[Dict[int, LoomStepData]] = None

    if bouts_by_fish is not None:
        step_bout_metrics = []

    for r in computed_results:
        step_metrics.append(r["base"])
        if step_bout_metrics is not None and "bout" in r:
            step_bout_metrics.append(r["bout"])
        elif step_bout_metrics is not None:
            # No bouts for this step — append empty.
            n_fish = len(fish_ids)
            step_bout_metrics.append((
                {"num_bouts": np.zeros(n_fish, dtype=np.int32),
                 "mean_bout_duration_s": np.zeros(n_fish, dtype=np.float32),
                 "mean_interbout_interval_s": np.zeros(n_fish, dtype=np.float32)},
                {"fish_id": np.array([], dtype=np.int32),
                 "bout_id": np.array([], dtype=np.int32),
                 "start_frame": np.array([], dtype=np.int64),
                 "end_frame": np.array([], dtype=np.int64),
                 "duration_s": np.array([], dtype=np.float32),
                 "mean_speed_mm_s": np.array([], dtype=np.float32),
                 "peak_physical_speed_mm_s": np.array([], dtype=np.float32)},
            ))
        if "grating" in r:
            if step_grating_data is None:
                step_grating_data = {}
            step_grating_data[r["step_index"]] = r["grating"]
        if "concentric" in r:
            if step_concentric_data is None:
                step_concentric_data = {}
            step_concentric_data[r["step_index"]] = r["concentric"]
        if "loom" in r:
            if step_loom_data is None:
                step_loom_data = {}
            step_loom_data[r["step_index"]] = r["loom"]

    global_omr_metrics: Optional[Dict[str, np.ndarray]] = None
    if step_grating_data is not None and not args.no_omr:
        omr_steps = [
            data.omr for data in step_grating_data.values()
            if data.omr is not None
        ]
        if omr_steps:
            global_omr_metrics = compute_global_omr_metrics(fish_ids, omr_steps)

    # Write output.
    parameters = {
        "layout": args.layout,
        "moving_threshold_mm_s": args.moving_threshold_mm_s,
        "fps": fps,
        "n_frames": n_frames,
        "camera_to_projector_offset_deg": args.camera_to_projector_offset_deg,
        "bin_size_s": args.bin_size_s,
        "follow_threshold": args.follow_threshold,
        "follow_window_s": args.follow_window_s,
        "omr_enabled": not args.no_omr,
        "omr_method_version": OMR_METHOD_VERSION if not args.no_omr else None,
        "concentric_radial_omr_method_version": (
            CONCENTRIC_RADIAL_OMR_METHOD_VERSION if not args.no_omr else None
        ),
        "omr_projection_deadzone": args.omr_projection_deadzone,
        "omr_projection_speed_deadzone_mm_s": args.omr_projection_speed_deadzone_mm_s,
        "omr_window_s": (
            args.omr_window_s
            if args.omr_window_s is not None
            else list(OMR_DEFAULT_WINDOW_LENGTHS_S)
        ),
        "omr_early_window_s": (
            args.omr_early_window_s
            if args.omr_early_window_s is not None
            else list(OMR_DEFAULT_EARLY_RESPONSE_WINDOWS_S)
        ),
        "center_threshold_mm": args.center_threshold_mm,
        "concentric_radial_singularity_epsilon_mm": args.concentric_radial_singularity_epsilon_mm,
        "escape_speed_threshold_mm_s": args.escape_speed_threshold_mm_s,
        "escape_window_s": args.escape_window_s,
        "loom_pre_onset_s": args.loom_pre_onset_s,
        "loom_post_onset_s": args.loom_post_onset_s,
        "loom_bin_size_s": args.loom_bin_size_s,
    }

    # Build frame annotations.
    frame_annotations = build_frame_annotations(steps, n_frames)

    run_name = write_stimulus_response_run(
        root,
        global_metrics=global_metrics,
        steps=steps,
        step_metrics=step_metrics,
        frame_annotations=frame_annotations,
        step_bout_metrics=step_bout_metrics,
        step_grating_data=step_grating_data,
        step_concentric_data=step_concentric_data,
        step_loom_data=step_loom_data,
        global_omr_metrics=global_omr_metrics,
        source_kinematics_run=kin_run,
        source_kinematics_type=args.track_kinematics_type,
        source_stimulus_run=stim_run,
        source_bout_run=bout_run_name,
        upstream_lineage=upstream_lineage,
        parameters=parameters,
        run_name=args.run_name,
        overwrite=args.overwrite,
        layout=args.layout,
        console=console,
    )

    if args.write_zarr_artifacts:
        from fisheye.analysis.plot_stimulus_response_omr import (
            write_omr_summary_visualization,
        )

        try:
            write_omr_summary_visualization(
                root,
                run_name=run_name,
                zarr_path=Path(args.zarr_path),
                artifact_dpi=args.artifact_dpi,
                command=" ".join(sys.argv),
                console=console,
            )
        except ValueError as exc:
            console.print(f"  [yellow]OMR visualization skipped: {exc}[/yellow]")

    console.print(f"\n[bold green]Done.[/bold green] Run: {run_name}")


if __name__ == "__main__":
    main()
