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
from fisheye.shared.stage_provenance import (
    build_stage_provenance,
    write_stage_provenance,
)
from fisheye.shared.zarr.analysis_stage_arrays import validate_track_inputs
from fisheye.shared.zarr_helpers import resolve_zarr_run
from fisheye.utils.system import get_git_info
from fisheye.utils.zarr_io import open_zarr_root


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

    parent_path = f"analysis/track_kinematics_runs/{kinematics_type}"
    kin_group, run_name = resolve_zarr_run(
        root, parent_path, run_name=kinematics_run,
    )

    fps = float(kin_group.attrs.get("fps", 30.0))

    tracks_parent = kin_group["tracks"]
    track_names = sorted(
        (name for name in tracks_parent.group_keys()
         if re.fullmatch(r"id_\d+", name)),
        key=lambda n: int(n.split("_")[1]),
    )
    if not track_names:
        raise ValueError(
            f"No track subgroups (id_*) found in {parent_path}/{run_name}/tracks/"
        )

    # First pass: determine total frame span and validate inputs.
    max_frame = 0
    for name in track_names:
        tg = tracks_parent[name]
        result = validate_track_inputs(tg, label=f"{parent_path}/{run_name}/tracks/{name}")
        if not result.valid:
            raise ValueError(
                f"Track {name} failed input validation:\n"
                + "\n".join(f"  - {e}" for e in result.errors)
            )
        fi = tg["frame_indices"]
        if fi.shape[0] > 0:
            max_frame = max(max_frame, int(fi[-1]))

    n_frames = max_frame + 1

    # Second pass: expand sparse → dense.
    tracks: List[DenseTrack] = []
    for name in track_names:
        fish_id = int(name.split("_")[1])
        tg = tracks_parent[name]

        frame_indices = tg["frame_indices"][:].astype(np.int64)
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
            speed_mm[frame_indices] = tg["speed_smoothed_mm"][:].astype(np.float32)
            heading_deg[frame_indices] = tg["heading_degrees"][:].astype(np.float32)
            pos_mm[frame_indices] = tg["positions_mm"][:].astype(np.float32)
            ang_vel[frame_indices] = tg["angular_velocity_deg_s"][:].astype(np.float32)
            time_s[frame_indices] = tg["time_seconds"][:].astype(np.float32)
            valid[frame_indices] = True
            det_src[frame_indices] = tg["detection_source"][:].astype(np.int8)
            if "frame_path_distance_smoothed_mm" in tg:
                frame_path_distance_mm = np.zeros(n_frames, dtype=np.float32)
                frame_path_distance_mm[frame_indices] = tg["frame_path_distance_smoothed_mm"][:].astype(np.float32)
            if "cumulative_path_distance_mm" in tg:
                sparse_cumulative = tg["cumulative_path_distance_mm"][:].astype(np.float32)
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
        f"  Loaded {len(tracks)} track(s) from {parent_path}/{run_name}/ "
        f"({n_frames} frames)"
    )
    return tracks, run_name, n_frames, upstream_lineage


# ---------------------------------------------------------------------------
# Stimulus event parsing
# ---------------------------------------------------------------------------


def _decode_text_value(value: Any) -> str:
    """Decode zarr string scalars, bytes, or fixed-width uint8 rows."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").rstrip("\x00")
    if isinstance(value, str):
        return value.rstrip("\x00")

    arr = np.asarray(value)
    if arr.dtype.kind in ("u", "i") and arr.ndim >= 1:
        payload = bytes(int(item) for item in arr.ravel() if int(item) != 0)
        return payload.decode("utf-8", errors="replace").rstrip("\x00")
    if arr.dtype.kind == "S":
        return bytes(arr.tobytes()).decode("utf-8", errors="replace").rstrip("\x00")
    return str(value).rstrip("\x00")


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
    bouts_group: zarr.Group,
    names: Sequence[str],
    *,
    dtype: np.dtype | type = np.float64,
) -> np.ndarray:
    """Read the first available bout column from current or legacy schemas."""

    for name in names:
        if name in bouts_group:
            return bouts_group[name][:].astype(dtype)
    expected = ", ".join(names)
    raise ValueError(f"Bouts group is missing expected column; tried: {expected}")


def load_bout_data(
    root: zarr.Group,
    *,
    bout_run: Optional[str] = None,
    console: Optional[Console] = None,
) -> Tuple[Dict[int, List[BoutEntry]], str]:
    """Load swim bouts from detect_bouts_multi_level output.

    Reads from the ``default_level`` speed subgroup (typically speed_smoothed).

    Returns
    -------
    bouts_by_fish : dict[int, list[BoutEntry]]
        Bouts keyed by fish_id (track_id).
    run_name : str
        Resolved bout run name.
    """
    console = console or Console()

    bout_group, run_name = resolve_zarr_run(
        root, "analysis/swim_bout_runs", run_name=bout_run,
    )

    default_level = str(bout_group.attrs.get("default_level", "speed_smoothed"))
    track_id = int(bout_group.attrs.get("track_id", 0))

    if default_level not in bout_group:
        raise ValueError(
            f"Bout run '{run_name}' missing expected level group '{default_level}'"
        )

    level_group = bout_group[default_level]
    if "bouts" not in level_group:
        console.print(f"  [yellow]No bouts group in {default_level}; returning empty.[/yellow]")
        return {}, run_name

    bouts_group = level_group["bouts"]

    # Read columnar arrays.
    bout_ids = bouts_group["bout_id"][:] if "bout_id" in bouts_group else np.array([], dtype=np.int32)
    n_bouts = len(bout_ids)

    if n_bouts == 0:
        return {}, run_name

    start_frames = bouts_group["start_frame"][:].astype(np.int64)
    end_frames = bouts_group["end_frame"][:].astype(np.int64)
    durations = bouts_group["duration_s"][:].astype(np.float64)
    mean_speeds = _read_first_bout_column(
        bouts_group,
        ("mean_speed_mm_s", "mean_speed"),
    )
    peak_physical_speeds = _read_first_bout_column(
        bouts_group,
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
        f"  Loaded {n_bouts} bout(s) from swim_bout_runs/{run_name}/{default_level}/"
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
# Grating metrics
# ---------------------------------------------------------------------------

_MOVING_GRATING = "MOVING_GRATING"  # Local alias for readability.
OMR_METHOD_VERSION = "stimulus_response_omr_v3"
OMR_DEFAULT_WINDOW_LENGTHS_S = (10.0, 30.0, 60.0)
OMR_DEFAULT_EARLY_RESPONSE_WINDOWS_S = (5.0, 10.0)


def resolve_grating_direction(
    step: ProtocolStep,
    offset_deg: float = 0.0,
) -> float:
    """Extract grating drift direction in camera-space degrees.

    Reads ``orientation_degrees`` (or fallbacks) from step stimulus_params
    and applies the configured projector-to-camera angular correction. The
    returned direction is wrapped to ``[0, 360)`` for stable persisted labels.
    """
    params = flatten_stimulus_params(step.stimulus_params)
    direction = 0.0
    for key in ("orientation_degrees", "angle_degrees", "grating_orientation"):
        if key in params and params[key] is not None:
            direction = params[key]
            break
    return (float(direction) + float(offset_deg)) % 360.0


def resolve_grating_speed_mm_s(step: ProtocolStep) -> float:
    """Resolve moving-grating speed from canonical protocol params."""

    params = flatten_stimulus_params(step.stimulus_params)
    for key in ("grating_speed_mm_s", "speed_mm_per_sec", "speed_mm_s"):
        if key in params and params[key] is not None:
            return float(params[key])
    return 0.0


def _wrap_angle(angle_deg: np.ndarray) -> np.ndarray:
    """Wrap angles to [-180, +180]."""
    return ((angle_deg + 180.0) % 360.0) - 180.0


def compute_grating_per_frame(
    tracks: Sequence[DenseTrack],
    step: ProtocolStep,
    grating_dir_deg: float,
    fps: float,
) -> Dict[str, np.ndarray]:
    """Per-frame grating alignment metrics for one MOVING_GRATING step.

    Returns arrays shaped (n_fish, n_step_frames).
    """
    sf, ef = step.start_frame, step.end_frame
    n_step = max(ef - sf, 1)
    n_fish = len(tracks)

    frame_indices = np.arange(sf, ef, dtype=np.int64)
    valid = np.zeros((n_fish, n_step), dtype=bool)
    det_src = np.full((n_fish, n_step), -1, dtype=np.int8)
    alignment_angle = np.zeros((n_fish, n_step), dtype=np.float32)
    alignment_cos = np.zeros((n_fish, n_step), dtype=np.float32)
    speed_along = np.zeros((n_fish, n_step), dtype=np.float32)
    ang_vel = np.zeros((n_fish, n_step), dtype=np.float32)

    grating_dir_rad = np.deg2rad(grating_dir_deg)

    for i, t in enumerate(tracks):
        heading_step = t.heading_deg[sf:ef]
        speed_step = t.speed_mm[sf:ef]
        valid_step = (
            t.valid[sf:ef]
            & np.isfinite(heading_step)
            & np.isfinite(speed_step)
            & np.isfinite(t.angular_velocity[sf:ef])
        )
        angvel_step = t.angular_velocity[sf:ef]

        valid[i] = valid_step
        det_src[i] = t.detection_source[sf:ef]

        # Alignment angle: heading - grating direction, wrapped to [-180, 180].
        raw_diff = heading_step - grating_dir_deg
        alignment_angle[i] = _wrap_angle(raw_diff)
        alignment_cos[i] = np.cos(np.deg2rad(alignment_angle[i]))

        # Speed projected onto grating direction.
        heading_rad = np.deg2rad(heading_step)
        speed_along[i] = speed_step * np.cos(heading_rad - grating_dir_rad)

        ang_vel[i] = angvel_step

        # Zero out invalid frames.
        inv = ~valid_step
        alignment_angle[i, inv] = 0.0
        alignment_cos[i, inv] = 0.0
        speed_along[i, inv] = 0.0
        ang_vel[i, inv] = 0.0

    return {
        "frame_indices": frame_indices,
        "valid": valid,
        "detection_source": det_src,
        "alignment_angle_deg": alignment_angle,
        "alignment_cos": alignment_cos,
        "speed_along_grating_mm_s": speed_along,
        "angular_velocity_deg_s": ang_vel,
    }


def compute_grating_per_fish(
    per_frame: Dict[str, np.ndarray],
    tracks: Sequence[DenseTrack],
    step: ProtocolStep,
    fps: float,
    grating_speed_mm_s: float = 0.0,
    follow_threshold: float = 0.5,
    follow_window_s: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Per-fish summary grating metrics for one step."""
    sf, ef = step.start_frame, step.end_frame
    n_fish = len(tracks)

    a_cos = per_frame["alignment_cos"]        # (n_fish, n_step)
    a_angle = per_frame["alignment_angle_deg"]
    spd_along = per_frame["speed_along_grating_mm_s"]
    ang_vel = per_frame["angular_velocity_deg_s"]

    mean_acos = np.zeros(n_fish, dtype=np.float32)
    rvl = np.zeros(n_fish, dtype=np.float32)
    frac_follow = np.zeros(n_fish, dtype=np.float32)
    frac_oppose = np.zeros(n_fish, dtype=np.float32)
    frac_perp = np.zeros(n_fish, dtype=np.float32)
    spd_wt_align = np.zeros(n_fish, dtype=np.float32)
    opt_gain = np.zeros(n_fish, dtype=np.float32)
    drift_along = np.zeros(n_fish, dtype=np.float32)
    drift_perp = np.zeros(n_fish, dtype=np.float32)
    latency = np.full(n_fish, np.nan, dtype=np.float32)

    grating_dir_deg = resolve_grating_direction(step)
    grating_dir_rad = np.deg2rad(grating_dir_deg)
    along_vec = np.array([np.cos(grating_dir_rad), np.sin(grating_dir_rad)], dtype=np.float32)
    perp_vec = np.array([-np.sin(grating_dir_rad), np.cos(grating_dir_rad)], dtype=np.float32)

    follow_window_frames = max(1, int(follow_window_s * fps))

    for i, t in enumerate(tracks):
        v = t.valid[sf:ef]
        n_valid = int(v.sum())
        if n_valid == 0:
            continue

        cos_v = a_cos[i][v]
        angle_rad = np.deg2rad(a_angle[i][v])
        spd_v = t.speed_mm[sf:ef][v]
        spd_along_v = spd_along[i][v]

        # Heading alignment summary.
        mean_acos[i] = float(np.mean(cos_v))

        # Resultant vector length (circular statistics).
        mean_vec = np.mean(np.exp(1j * angle_rad))
        rvl[i] = float(np.abs(mean_vec))

        # Fraction following / opposing / perpendicular.
        frac_follow[i] = float((cos_v > 0).sum()) / n_valid
        frac_oppose[i] = float((cos_v < 0).sum()) / n_valid
        frac_perp[i] = float((np.abs(cos_v) < 0.25).sum()) / n_valid

        # Speed-weighted alignment.
        total_spd = float(np.sum(spd_v))
        if total_spd > 0:
            spd_wt_align[i] = float(np.sum(spd_v * cos_v)) / total_spd

        # Optomotor gain.
        if grating_speed_mm_s > 0:
            opt_gain[i] = float(np.mean(spd_along_v)) / grating_speed_mm_s

        # Positional drift.
        pos_step = t.positions_mm[sf:ef]
        valid_pos = pos_step[v]
        if valid_pos.shape[0] >= 2:
            displacement = valid_pos[-1] - valid_pos[0]
            drift_along[i] = float(np.dot(displacement, along_vec))
            drift_perp[i] = float(np.dot(displacement, perp_vec))

        # Latency to follow: first sustained window where mean(cos) > threshold.
        full_cos = a_cos[i]  # includes invalid frames (zeros)
        full_valid = v
        if follow_window_frames <= full_cos.shape[0]:
            for start in range(full_cos.shape[0] - follow_window_frames + 1):
                window = slice(start, start + follow_window_frames)
                w_valid = full_valid[window]
                n_wv = int(w_valid.sum())
                if n_wv < follow_window_frames * 0.5:
                    continue
                w_cos = full_cos[window][w_valid]
                if float(np.mean(w_cos)) > follow_threshold:
                    latency[i] = float(start) / fps if fps > 0 else 0.0
                    break

    return {
        "mean_alignment_cos": mean_acos,
        "resultant_vector_length": rvl,
        "fraction_following": frac_follow,
        "fraction_opposing": frac_oppose,
        "fraction_perpendicular": frac_perp,
        "speed_weighted_alignment": spd_wt_align,
        "optomotor_gain": opt_gain,
        "drift_along_grating_mm": drift_along,
        "drift_perp_grating_mm": drift_perp,
        "latency_to_follow_s": latency,
    }


def compute_grating_time_series(
    per_frame: Dict[str, np.ndarray],
    tracks: Sequence[DenseTrack],
    step: ProtocolStep,
    fps: float,
    bin_size_s: float = 1.0,
    grating_speed_mm_s: float = 0.0,
) -> Dict[str, np.ndarray]:
    """Binned temporal dynamics for one MOVING_GRATING step."""
    sf, ef = step.start_frame, step.end_frame
    n_step = max(ef - sf, 1)
    n_fish = len(tracks)

    bin_size_frames = max(1, int(bin_size_s * fps))
    n_bins = max(1, (n_step + bin_size_frames - 1) // bin_size_frames)

    bin_center_s = np.zeros(n_bins, dtype=np.float32)
    acos_binned = np.zeros((n_fish, n_bins), dtype=np.float32)
    speed_binned = np.zeros((n_fish, n_bins), dtype=np.float32)
    follow_binned = np.zeros((n_fish, n_bins), dtype=np.float32)
    gain_binned = np.zeros((n_fish, n_bins), dtype=np.float32)

    a_cos = per_frame["alignment_cos"]
    spd_along = per_frame["speed_along_grating_mm_s"]

    for b in range(n_bins):
        bs = b * bin_size_frames
        be = min(bs + bin_size_frames, n_step)
        bin_center_s[b] = ((bs + be) / 2.0) / fps if fps > 0 else 0.0

        for i, t in enumerate(tracks):
            v = t.valid[sf + bs:sf + be]
            n_v = int(v.sum())
            if n_v == 0:
                continue
            cos_bin = a_cos[i, bs:be][v]
            spd_bin = t.speed_mm[sf + bs:sf + be][v]
            spd_along_bin = spd_along[i, bs:be][v]

            acos_binned[i, b] = float(np.mean(cos_bin))
            speed_binned[i, b] = float(np.mean(spd_bin))
            follow_binned[i, b] = float((cos_bin > 0).sum()) / n_v
            if grating_speed_mm_s > 0:
                gain_binned[i, b] = float(np.mean(spd_along_bin)) / grating_speed_mm_s

    return {
        "bin_center_s": bin_center_s,
        "alignment_cos": acos_binned,
        "speed_mm_s": speed_binned,
        "fraction_following": follow_binned,
        "optomotor_gain": gain_binned,
    }


def _grating_direction_vector(direction_deg: float) -> np.ndarray:
    """Return a unit vector in camera/mm coordinates for grating drift."""

    rad = np.deg2rad(direction_deg)
    return np.array([np.cos(rad), np.sin(rad)], dtype=np.float64)


def _finite_or_nan(value: float) -> float:
    value = float(value)
    return value if np.isfinite(value) else float("nan")


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0 or not np.isfinite(denominator):
        return float("nan")
    return _finite_or_nan(numerator / denominator)


def _json_safe_attr_value(value: Any) -> Any:
    """Return a strict-JSON-safe value for Zarr attrs/provenance."""

    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, np.ndarray):
        return _json_safe_attr_value(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_safe_attr_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_attr_value(item) for item in value]
    return value


def _json_safe_attrs(attrs: Mapping[str, Any]) -> Dict[str, Any]:
    return {str(key): _json_safe_attr_value(value) for key, value in attrs.items()}


def _first_finite_float(*values: Any) -> Optional[float]:
    for value in values:
        if value is None:
            continue
        if isinstance(value, np.generic):
            value = value.item()
        try:
            out = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(out):
            return out
    return None


def _group_attrs(group: Any) -> Dict[str, Any]:
    if group is None or not hasattr(group, "attrs"):
        return {}
    try:
        return dict(group.attrs)
    except Exception:
        return {}


def _get_child_group(group: Any, key: str) -> Any:
    if group is None:
        return None
    getter = getattr(group, "get", None)
    if callable(getter):
        try:
            child = getter(key)
            if child is not None:
                return child
        except Exception:
            pass
    try:
        return group[key]
    except Exception:
        return None


def _iter_child_group_names(group: Any) -> List[str]:
    if group is None:
        return []
    keys_fn = getattr(group, "group_keys", None)
    if callable(keys_fn):
        try:
            return [str(key) for key in keys_fn()]
        except Exception:
            pass
    keys_fn = getattr(group, "keys", None)
    if callable(keys_fn):
        names: List[str] = []
        try:
            for key in keys_fn():
                child = _get_child_group(group, str(key))
                if isinstance(child, zarr.Group):
                    names.append(str(key))
        except Exception:
            return names
        return names
    return []


def _calibration_attr_sources(root: Any, stimulus_run: Optional[str]) -> List[Tuple[str, Dict[str, Any]]]:
    sources: List[Tuple[str, Dict[str, Any]]] = []
    analysis = _get_child_group(root, "analysis")
    analysis_cal = _get_child_group(analysis, "calibration")
    sources.append(("analysis/calibration", _group_attrs(analysis_cal)))

    root_cal = _get_child_group(root, "calibration")
    sources.append(("calibration", _group_attrs(root_cal)))

    stim_parent = _get_child_group(analysis, "stimulus_runs")
    stim_name = stimulus_run
    if not stim_name and hasattr(stim_parent, "attrs"):
        latest = stim_parent.attrs.get("latest")
        stim_name = str(latest) if latest is not None else None
    stim_group = _get_child_group(stim_parent, stim_name) if stim_name else None
    stim_cal = _get_child_group(stim_group, "calibration")
    for camera_id in _iter_child_group_names(stim_cal):
        sources.append((
            f"analysis/stimulus_runs/{stim_name}/calibration/{camera_id}",
            _group_attrs(_get_child_group(stim_cal, camera_id)),
        ))
    return [(name, attrs) for name, attrs in sources if attrs]


def _resolve_omr_arena_geometry_mm(
    root: Any,
    stimulus_run: Optional[str],
    direction_xy: np.ndarray,
) -> Tuple[Optional[Tuple[float, float]], Optional[float], str]:
    """Resolve arena center and half-extent along the grating axis in mm."""

    cal = load_calibration_transform(root, stimulus_run=stimulus_run)
    pixel_to_mm = _first_finite_float(cal.get("pixel_to_mm"))
    projector_pixels_per_mm = _first_finite_float(cal.get("pixels_per_mm_projector"))
    sources = _calibration_attr_sources(root, stimulus_run)

    center_mm: Optional[Tuple[float, float]] = None
    center_source = ""
    if cal.get("arena_center_px") is not None and pixel_to_mm is not None:
        cx, cy = cal["arena_center_px"]
        center_mm = (float(cx) * pixel_to_mm, float(cy) * pixel_to_mm)
        center_source = "arena_center_px_camera"
    for source_name, attrs in sources:
        if center_mm is None:
            cx = _first_finite_float(
                attrs.get("arena_center_x_mm"),
                attrs.get("experimental_area_center_x_mm"),
            )
            cy = _first_finite_float(
                attrs.get("arena_center_y_mm"),
                attrs.get("experimental_area_center_y_mm"),
            )
            if cx is not None and cy is not None:
                center_mm = (cx, cy)
                center_source = f"{source_name}:center_mm"
        if center_mm is None:
            cx = _first_finite_float(attrs.get("experimental_area_center_x_px"))
            cy = _first_finite_float(attrs.get("experimental_area_center_y_px"))
            pixels_per_mm = _first_finite_float(
                attrs.get("pixels_per_mm_projector"),
                projector_pixels_per_mm,
            )
            if cx is not None and cy is not None and pixels_per_mm is not None and pixels_per_mm > 0:
                # Citrus experimental-area fields are in stimulus/projector
                # pixels, matching local arena millimetres after projector
                # scale conversion. Do not convert them with camera pixel_to_mm.
                center_mm = (cx / pixels_per_mm, cy / pixels_per_mm)
                center_source = f"{source_name}:experimental_area_center_projector_px"
        if center_mm is None:
            cx = _first_finite_float(attrs.get("arena_center_x_px"))
            cy = _first_finite_float(attrs.get("arena_center_y_px"))
            if cx is not None and cy is not None and pixel_to_mm is not None:
                center_mm = (cx * pixel_to_mm, cy * pixel_to_mm)
                center_source = f"{source_name}:arena_center_camera_px"
        if center_mm is None:
            w_mm = _first_finite_float(attrs.get("sub_arena_width_mm"))
            h_mm = _first_finite_float(attrs.get("sub_arena_height_mm"))
            if w_mm is not None and h_mm is not None and w_mm > 0 and h_mm > 0:
                center_mm = (0.5 * w_mm, 0.5 * h_mm)
                center_source = f"{source_name}:sub_arena_size_mm"
        if center_mm is None:
            w_px = _first_finite_float(attrs.get("sub_arena_width_px"))
            h_px = _first_finite_float(attrs.get("sub_arena_height_px"))
            pixels_per_mm = _first_finite_float(
                attrs.get("pixels_per_mm_projector"),
                projector_pixels_per_mm,
            )
            if w_px is not None and h_px is not None and pixels_per_mm is not None and pixels_per_mm > 0:
                center_mm = (0.5 * w_px / pixels_per_mm, 0.5 * h_px / pixels_per_mm)
                center_source = f"{source_name}:sub_arena_size_projector_px"

    extent_mm: Optional[float] = None
    extent_source = ""
    for source_name, attrs in sources:
        radius_mm = _first_finite_float(
            attrs.get("arena_radius_mm"),
            attrs.get("experimental_area_radius_mm"),
        )
        if radius_mm is not None and radius_mm > 0:
            extent_mm = radius_mm
            extent_source = f"{source_name}:radius_mm"
            break

        radius_px = _first_finite_float(attrs.get("experimental_area_radius_px"))
        pixels_per_mm = _first_finite_float(
            attrs.get("pixels_per_mm_projector"),
            projector_pixels_per_mm,
        )
        if radius_px is not None and radius_px > 0 and pixels_per_mm is not None and pixels_per_mm > 0:
            extent_mm = radius_px / pixels_per_mm
            extent_source = f"{source_name}:experimental_area_radius_projector_px"
            break

        radius_px = _first_finite_float(attrs.get("arena_radius_px"))
        if radius_px is not None and radius_px > 0 and pixel_to_mm is not None:
            extent_mm = radius_px * pixel_to_mm
            extent_source = f"{source_name}:arena_radius_camera_px"
            break

        width_mm = _first_finite_float(
            attrs.get("arena_width_mm"),
            attrs.get("experimental_area_width_mm"),
            attrs.get("sub_arena_width_mm"),
        )
        height_mm = _first_finite_float(
            attrs.get("arena_height_mm"),
            attrs.get("experimental_area_height_mm"),
            attrs.get("sub_arena_height_mm"),
        )
        if width_mm is None or height_mm is None:
            width_px = _first_finite_float(
                attrs.get("arena_width_px"),
                attrs.get("experimental_area_width_px"),
                attrs.get("sub_arena_width_px"),
            )
            height_px = _first_finite_float(
                attrs.get("arena_height_px"),
                attrs.get("experimental_area_height_px"),
                attrs.get("sub_arena_height_px"),
            )
            if width_px is not None and height_px is not None and pixel_to_mm is not None:
                width_mm = width_px * pixel_to_mm
                height_mm = height_px * pixel_to_mm
        if width_mm is not None and height_mm is not None and width_mm > 0 and height_mm > 0:
            extent_mm = 0.5 * (
                abs(float(direction_xy[0])) * width_mm
                + abs(float(direction_xy[1])) * height_mm
            )
            extent_source = f"{source_name}:axis_projected_rectangle"
            break

    if center_mm is None and extent_mm is None:
        return None, None, "unavailable"

    parts = []
    if center_mm is not None:
        parts.append(center_source or "center")
    if extent_mm is not None:
        parts.append(extent_source or "extent")
    return center_mm, extent_mm, ";".join(parts)


def _position_axis_metrics(
    track: DenseTrack,
    start_frame: int,
    end_frame: int,
    direction_xy: np.ndarray,
    arena_center_mm: Optional[Tuple[float, float]],
    arena_axis_extent_mm: Optional[float],
) -> Dict[str, float]:
    """Project fish occupancy onto the stimulus axis for one step/window."""

    keys = {
        "start_position_axis_mm": float("nan"),
        "end_position_axis_mm": float("nan"),
        "mean_position_axis_mm": float("nan"),
        "start_position_axis_norm": float("nan"),
        "end_position_axis_norm": float("nan"),
        "mean_position_axis_norm": float("nan"),
        "fraction_time_correct_side": float("nan"),
        "available_forward_space_at_start_mm": float("nan"),
        "available_backward_space_at_start_mm": float("nan"),
        "available_forward_space_at_start_norm": float("nan"),
        "available_backward_space_at_start_norm": float("nan"),
        "opportunity_normalized_parallel_displacement": float("nan"),
    }
    if arena_center_mm is None:
        return keys

    start = max(int(start_frame), 0)
    end = min(int(end_frame), int(track.valid.shape[0]))
    if end <= start:
        return keys

    frame_valid = (
        track.valid[start:end]
        & np.isfinite(track.positions_mm[start:end]).all(axis=1)
    )
    if not np.any(frame_valid):
        return keys

    positions = track.positions_mm[start:end][frame_valid].astype(np.float64)
    center = np.asarray(arena_center_mm, dtype=np.float64)
    axis = (positions - center) @ direction_xy.astype(np.float64)
    keys["start_position_axis_mm"] = float(axis[0])
    keys["end_position_axis_mm"] = float(axis[-1])
    keys["mean_position_axis_mm"] = float(np.mean(axis))
    keys["fraction_time_correct_side"] = float(np.mean(axis > 0.0))

    extent = float(arena_axis_extent_mm) if arena_axis_extent_mm is not None else float("nan")
    if np.isfinite(extent) and extent > 0.0:
        start_norm = keys["start_position_axis_mm"] / extent
        end_norm = keys["end_position_axis_mm"] / extent
        mean_norm = keys["mean_position_axis_mm"] / extent
        keys["start_position_axis_norm"] = _finite_or_nan(start_norm)
        keys["end_position_axis_norm"] = _finite_or_nan(end_norm)
        keys["mean_position_axis_norm"] = _finite_or_nan(mean_norm)
        keys["available_forward_space_at_start_mm"] = _finite_or_nan(extent - keys["start_position_axis_mm"])
        keys["available_backward_space_at_start_mm"] = _finite_or_nan(extent + keys["start_position_axis_mm"])
        keys["available_forward_space_at_start_norm"] = _finite_or_nan(1.0 - start_norm)
        keys["available_backward_space_at_start_norm"] = _finite_or_nan(1.0 + start_norm)
        parallel = keys["end_position_axis_mm"] - keys["start_position_axis_mm"]
        denom = (
            keys["available_forward_space_at_start_mm"]
            if parallel >= 0.0 else keys["available_backward_space_at_start_mm"]
        )
        keys["opportunity_normalized_parallel_displacement"] = _safe_ratio(parallel, denom)
    return keys


def _valid_transition_components(
    track: DenseTrack,
    start_frame: int,
    end_frame: int,
    direction_xy: np.ndarray,
    fps: float,
) -> Dict[str, np.ndarray]:
    """Frame-to-frame physical displacement components for one window.

    Returned arrays are indexed by the current frame in each transition. A
    transition from frame ``t-1`` to ``t`` is included only when both frames are
    valid. This prevents OMR displacement from silently crossing tracking gaps.
    """

    start = max(int(start_frame), 0)
    end = min(int(end_frame), int(track.valid.shape[0]))
    current_frames = np.arange(max(start + 1, 1), end, dtype=np.int64)
    if current_frames.size == 0:
        empty_float = np.array([], dtype=np.float64)
        return {
            "frames": current_frames,
            "dx": np.empty((0, 2), dtype=np.float64),
            "path": empty_float,
            "parallel": empty_float,
            "dt": empty_float,
            "speed": empty_float,
        }

    valid = (
        track.valid[current_frames]
        & track.valid[current_frames - 1]
        & np.isfinite(track.positions_mm[current_frames]).all(axis=1)
        & np.isfinite(track.positions_mm[current_frames - 1]).all(axis=1)
    )
    frames = current_frames[valid]
    if frames.size == 0:
        empty_float = np.array([], dtype=np.float64)
        return {
            "frames": frames,
            "dx": np.empty((0, 2), dtype=np.float64),
            "path": empty_float,
            "parallel": empty_float,
            "dt": empty_float,
            "speed": empty_float,
        }

    dx = (
        track.positions_mm[frames].astype(np.float64)
        - track.positions_mm[frames - 1].astype(np.float64)
    )
    path = np.linalg.norm(dx, axis=1)
    parallel = dx @ direction_xy.astype(np.float64)

    dt = (
        track.time_seconds[frames].astype(np.float64)
        - track.time_seconds[frames - 1].astype(np.float64)
    )
    fallback_dt = 1.0 / fps if fps > 0 else 0.0
    dt[~np.isfinite(dt) | (dt <= 0.0)] = fallback_dt
    speed = track.speed_mm[frames].astype(np.float64)

    return {
        "frames": frames,
        "dx": dx,
        "path": path,
        "parallel": parallel,
        "dt": dt,
        "speed": speed,
    }


def _omr_summary_for_window(
    track: DenseTrack,
    start_frame: int,
    end_frame: int,
    direction_xy: np.ndarray,
    fps: float,
    moving_threshold_mm_s: float,
    projection_speed_deadzone_mm_s: float,
    arena_center_mm: Optional[Tuple[float, float]] = None,
    arena_axis_extent_mm: Optional[float] = None,
) -> Dict[str, float | int]:
    """Compute physical OMR summary metrics for one fish/window."""

    components = _valid_transition_components(
        track, start_frame, end_frame, direction_xy, fps,
    )
    parallel = components["parallel"]
    path = components["path"]
    dt = components["dt"]
    speed = components["speed"]

    total_parallel = float(np.sum(parallel)) if parallel.size else 0.0
    total_path = float(np.sum(path)) if path.size else 0.0
    valid_transition_count = int(parallel.size)

    frames_possible = max(min(int(end_frame), track.valid.shape[0]) - max(int(start_frame), 0) - 1, 0)
    coverage = (
        float(valid_transition_count) / float(frames_possible)
        if frames_possible > 0 else 0.0
    )

    if valid_transition_count > 0:
        valid_frames = np.flatnonzero(track.valid[max(int(start_frame), 0):min(int(end_frame), track.valid.shape[0])])
    else:
        valid_frames = np.array([], dtype=np.int64)
    if valid_frames.size >= 2:
        offset = max(int(start_frame), 0)
        first_frame = int(valid_frames[0]) + offset
        last_frame = int(valid_frames[-1]) + offset
        net_dx = (
            track.positions_mm[last_frame].astype(np.float64)
            - track.positions_mm[first_frame].astype(np.float64)
        )
        net_displacement = float(np.linalg.norm(net_dx))
        net_parallel = float(net_dx @ direction_xy)
    else:
        net_displacement = 0.0
        net_parallel = 0.0

    moving = speed >= float(moving_threshold_mm_s)
    deadzone = float(projection_speed_deadzone_mm_s) * dt
    correct = moving & (parallel > deadzone)
    opposing = moving & (parallel < -deadzone)
    correct_s = float(np.sum(dt[correct])) if dt.size else 0.0
    opposing_s = float(np.sum(dt[opposing])) if dt.size else 0.0
    classified_s = correct_s + opposing_s

    if valid_transition_count == 0:
        quality_flag = 1
    elif total_path <= 0.0:
        quality_flag = 2
    else:
        quality_flag = 0

    result: Dict[str, float | int] = {
        "omr_path_index": _safe_ratio(total_parallel, total_path),
        "omr_net_direction_index": _safe_ratio(net_parallel, net_displacement),
        "parallel_displacement_mm": total_parallel,
        "net_displacement_mm": net_displacement,
        "path_length_mm": total_path,
        "valid_transition_count": valid_transition_count,
        "coverage_fraction": coverage,
        "time_fraction_correct_classified": _safe_ratio(correct_s, classified_s),
        "time_choice_index": _safe_ratio(correct_s - opposing_s, classified_s),
        "time_correct_s": correct_s,
        "time_opposing_s": opposing_s,
        "time_classified_s": classified_s,
        "quality_flag": quality_flag,
    }
    result.update(_position_axis_metrics(
        track,
        start_frame,
        end_frame,
        direction_xy,
        arena_center_mm,
        arena_axis_extent_mm,
    ))
    return result


def _bout_omr_score_for_bounds(
    track: DenseTrack,
    bout: BoutEntry,
    start_frame: int,
    end_frame: int,
    direction_xy: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Return per-bout OMR score and displacement/path components in bounds."""

    start = max(int(bout.start_frame), int(start_frame), 0)
    end = min(int(bout.end_frame), int(end_frame) - 1, track.valid.shape[0] - 1)
    if end <= start or not (track.valid[start] and track.valid[end]):
        return float("nan"), float("nan"), float("nan"), float("nan")

    displacement_xy = (
        track.positions_mm[end].astype(np.float64)
        - track.positions_mm[start].astype(np.float64)
    )
    bout_displacement = float(np.linalg.norm(displacement_xy))
    parallel = float(displacement_xy @ direction_xy)
    score = _safe_ratio(parallel, bout_displacement)
    path = _distance_for_window(track, start, end + 1)
    return score, parallel, bout_displacement, path


def _bout_omr_score(
    track: DenseTrack,
    bout: BoutEntry,
    step: ProtocolStep,
    direction_xy: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Return per-bout OMR score and displacement/path components."""

    return _bout_omr_score_for_bounds(
        track,
        bout,
        step.start_frame,
        step.end_frame,
        direction_xy,
    )


def _bout_omr_label(score: float, projection_deadzone: float) -> Tuple[int, int]:
    """Classify one per-bout OMR score into aligned/opposing/ambiguous."""

    if not np.isfinite(score):
        return 0, 1
    if score > projection_deadzone:
        return 1, 0
    if score < -projection_deadzone:
        return -1, 0
    return 0, 0


def _weighted_bout_omr_summary(
    labels: Sequence[int],
    parallel_displacements_mm: Sequence[float],
    bout_displacements_mm: Sequence[float],
    bout_path_lengths_mm: Sequence[float],
) -> Dict[str, float]:
    """Summarize bout-direction evidence weighted by physical movement."""

    label_arr = np.asarray(labels, dtype=np.int8)
    parallel = np.asarray(parallel_displacements_mm, dtype=np.float64)
    displacement = np.asarray(bout_displacements_mm, dtype=np.float64)
    path = np.asarray(bout_path_lengths_mm, dtype=np.float64)

    finite_path = np.isfinite(parallel) & np.isfinite(path) & (path > 0.0)
    finite_displacement = np.isfinite(displacement) & (displacement > 0.0)
    aligned = label_arr > 0
    opposing = label_arr < 0

    total_parallel = float(np.sum(parallel[finite_path])) if finite_path.size else 0.0
    total_path = float(np.sum(path[finite_path])) if finite_path.size else 0.0
    total_displacement = (
        float(np.sum(displacement[finite_displacement])) if finite_displacement.size else 0.0
    )
    aligned_path = float(np.sum(path[finite_path & aligned])) if finite_path.size else 0.0
    opposing_path = float(np.sum(path[finite_path & opposing])) if finite_path.size else 0.0
    aligned_displacement = (
        float(np.sum(displacement[finite_displacement & aligned]))
        if finite_displacement.size else 0.0
    )
    opposing_displacement = (
        float(np.sum(displacement[finite_displacement & opposing]))
        if finite_displacement.size else 0.0
    )
    classifiable_path = aligned_path + opposing_path
    classifiable_displacement = aligned_displacement + opposing_displacement

    return {
        "bout_path_index": _safe_ratio(total_parallel, total_path),
        "bout_parallel_displacement_sum_mm": total_parallel,
        "bout_path_length_sum_mm": total_path,
        "bout_displacement_sum_mm": total_displacement,
        "bout_classified_path_length_sum_mm": classifiable_path,
        "bout_classified_displacement_sum_mm": classifiable_displacement,
        "bout_fraction_correct_weighted_by_path": _safe_ratio(aligned_path, classifiable_path),
        "bout_fraction_correct_weighted_by_displacement": _safe_ratio(
            aligned_displacement,
            classifiable_displacement,
        ),
        "bout_classifiable_path_fraction": _safe_ratio(classifiable_path, total_path),
        "bout_classifiable_displacement_fraction": _safe_ratio(
            classifiable_displacement,
            total_displacement,
        ),
    }


def compute_step_omr_metrics(
    tracks: Sequence[DenseTrack],
    step: ProtocolStep,
    grating_dir_deg: float,
    fps: float,
    *,
    moving_threshold_mm_s: float,
    bouts_by_fish: Optional[Dict[int, List[BoutEntry]]] = None,
    projection_deadzone: float = 0.0,
    projection_speed_deadzone_mm_s: float = 0.0,
    window_lengths_s: Sequence[float] = OMR_DEFAULT_WINDOW_LENGTHS_S,
    early_window_lengths_s: Sequence[float] = OMR_DEFAULT_EARLY_RESPONSE_WINDOWS_S,
    position_anchor: str = "positions_mm",
    arena_center_mm: Optional[Tuple[float, float]] = None,
    arena_axis_extent_mm: Optional[float] = None,
    arena_geometry_source: str = "unavailable",
) -> "OMRStepData":
    """Compute OMR responsiveness metrics for one static MOVING_GRATING step."""

    direction_xy = _grating_direction_vector(grating_dir_deg)
    fish_ids = np.array([t.fish_id for t in tracks], dtype=np.int32)
    n_fish = len(tracks)

    per_fish: Dict[str, np.ndarray] = {
        "fish_id": fish_ids,
        "omr_path_index": np.full(n_fish, np.nan, dtype=np.float32),
        "omr_net_direction_index": np.full(n_fish, np.nan, dtype=np.float32),
        "parallel_displacement_mm": np.zeros(n_fish, dtype=np.float32),
        "net_displacement_mm": np.zeros(n_fish, dtype=np.float32),
        "path_length_mm": np.zeros(n_fish, dtype=np.float32),
        "valid_transition_count": np.zeros(n_fish, dtype=np.int32),
        "coverage_fraction": np.zeros(n_fish, dtype=np.float32),
        "bout_fraction_correct_classified": np.full(n_fish, np.nan, dtype=np.float32),
        "bout_fraction_correct_all": np.full(n_fish, np.nan, dtype=np.float32),
        "bout_choice_index": np.full(n_fish, np.nan, dtype=np.float32),
        "bout_path_index": np.full(n_fish, np.nan, dtype=np.float32),
        "bout_fraction_correct_weighted_by_path": np.full(n_fish, np.nan, dtype=np.float32),
        "bout_fraction_correct_weighted_by_displacement": np.full(n_fish, np.nan, dtype=np.float32),
        "bout_parallel_displacement_sum_mm": np.zeros(n_fish, dtype=np.float32),
        "bout_path_length_sum_mm": np.zeros(n_fish, dtype=np.float32),
        "bout_displacement_sum_mm": np.zeros(n_fish, dtype=np.float32),
        "bout_classified_path_length_sum_mm": np.zeros(n_fish, dtype=np.float32),
        "bout_classified_displacement_sum_mm": np.zeros(n_fish, dtype=np.float32),
        "bout_classifiable_path_fraction": np.full(n_fish, np.nan, dtype=np.float32),
        "bout_classifiable_displacement_fraction": np.full(n_fish, np.nan, dtype=np.float32),
        "bout_count_total": np.zeros(n_fish, dtype=np.int32),
        "bout_count_correct": np.zeros(n_fish, dtype=np.int32),
        "bout_count_opposing": np.zeros(n_fish, dtype=np.int32),
        "bout_count_ambiguous": np.zeros(n_fish, dtype=np.int32),
        "time_fraction_correct_classified": np.full(n_fish, np.nan, dtype=np.float32),
        "time_choice_index": np.full(n_fish, np.nan, dtype=np.float32),
        "time_correct_s": np.zeros(n_fish, dtype=np.float32),
        "time_opposing_s": np.zeros(n_fish, dtype=np.float32),
        "time_classified_s": np.zeros(n_fish, dtype=np.float32),
        "start_position_axis_mm": np.full(n_fish, np.nan, dtype=np.float32),
        "end_position_axis_mm": np.full(n_fish, np.nan, dtype=np.float32),
        "mean_position_axis_mm": np.full(n_fish, np.nan, dtype=np.float32),
        "start_position_axis_norm": np.full(n_fish, np.nan, dtype=np.float32),
        "end_position_axis_norm": np.full(n_fish, np.nan, dtype=np.float32),
        "mean_position_axis_norm": np.full(n_fish, np.nan, dtype=np.float32),
        "fraction_time_correct_side": np.full(n_fish, np.nan, dtype=np.float32),
        "available_forward_space_at_start_mm": np.full(n_fish, np.nan, dtype=np.float32),
        "available_backward_space_at_start_mm": np.full(n_fish, np.nan, dtype=np.float32),
        "available_forward_space_at_start_norm": np.full(n_fish, np.nan, dtype=np.float32),
        "available_backward_space_at_start_norm": np.full(n_fish, np.nan, dtype=np.float32),
        "opportunity_normalized_parallel_displacement": np.full(n_fish, np.nan, dtype=np.float32),
        "first_aligned_bout_id": np.full(n_fish, -1, dtype=np.int32),
        "first_aligned_bout_start_frame": np.full(n_fish, -1, dtype=np.int64),
        "first_aligned_bout_latency_s": np.full(n_fish, np.nan, dtype=np.float32),
        "first_aligned_bout_score": np.full(n_fish, np.nan, dtype=np.float32),
        "first_opposing_bout_id": np.full(n_fish, -1, dtype=np.int32),
        "first_opposing_bout_start_frame": np.full(n_fish, -1, dtype=np.int64),
        "first_opposing_bout_latency_s": np.full(n_fish, np.nan, dtype=np.float32),
        "first_opposing_bout_score": np.full(n_fish, np.nan, dtype=np.float32),
        "first_classified_bout_id": np.full(n_fish, -1, dtype=np.int32),
        "first_classified_bout_start_frame": np.full(n_fish, -1, dtype=np.int64),
        "first_classified_bout_latency_s": np.full(n_fish, np.nan, dtype=np.float32),
        "first_classified_bout_score": np.full(n_fish, np.nan, dtype=np.float32),
        "quality_flag": np.zeros(n_fish, dtype=np.int8),
    }

    all_fish_id: List[int] = []
    all_bout_id: List[int] = []
    all_start: List[int] = []
    all_end: List[int] = []
    all_score: List[float] = []
    all_parallel: List[float] = []
    all_displacement: List[float] = []
    all_path: List[float] = []
    all_label: List[int] = []
    all_quality: List[int] = []

    for i, track in enumerate(tracks):
        summary = _omr_summary_for_window(
            track,
            step.start_frame,
            step.end_frame,
            direction_xy,
            fps,
            moving_threshold_mm_s,
            projection_speed_deadzone_mm_s,
            arena_center_mm,
            arena_axis_extent_mm,
        )
        for key, value in summary.items():
            if key in per_fish:
                per_fish[key][i] = value

        bouts = []
        if bouts_by_fish is not None:
            bouts = [
                b for b in bouts_by_fish.get(track.fish_id, [])
                if b.start_frame < step.end_frame and b.end_frame >= step.start_frame
            ]
        correct = opposing = ambiguous = 0
        track_labels: List[int] = []
        track_parallel: List[float] = []
        track_displacement: List[float] = []
        track_path: List[float] = []
        for bout in bouts:
            score, parallel, displacement, path = _bout_omr_score(
                track, bout, step, direction_xy,
            )
            label, quality = _bout_omr_label(score, projection_deadzone)
            if label > 0:
                correct += 1
            elif label < 0:
                opposing += 1
            else:
                ambiguous += 1
            track_labels.append(label)
            track_parallel.append(parallel)
            track_displacement.append(displacement)
            track_path.append(path)

            all_fish_id.append(track.fish_id)
            all_bout_id.append(bout.bout_id)
            all_start.append(bout.start_frame)
            all_end.append(bout.end_frame)
            all_score.append(score)
            all_parallel.append(parallel)
            all_displacement.append(displacement)
            all_path.append(path)
            all_label.append(label)
            all_quality.append(quality)

            if label != 0:
                latency_start = max(int(bout.start_frame), int(step.start_frame))
                latency_s = (
                    (latency_start - int(step.start_frame)) / fps
                    if fps > 0 else float("nan")
                )
                if per_fish["first_classified_bout_id"][i] < 0:
                    per_fish["first_classified_bout_id"][i] = int(bout.bout_id)
                    per_fish["first_classified_bout_start_frame"][i] = latency_start
                    per_fish["first_classified_bout_latency_s"][i] = latency_s
                    per_fish["first_classified_bout_score"][i] = score
                if label > 0 and per_fish["first_aligned_bout_id"][i] < 0:
                    per_fish["first_aligned_bout_id"][i] = int(bout.bout_id)
                    per_fish["first_aligned_bout_start_frame"][i] = latency_start
                    per_fish["first_aligned_bout_latency_s"][i] = latency_s
                    per_fish["first_aligned_bout_score"][i] = score
                if label < 0 and per_fish["first_opposing_bout_id"][i] < 0:
                    per_fish["first_opposing_bout_id"][i] = int(bout.bout_id)
                    per_fish["first_opposing_bout_start_frame"][i] = latency_start
                    per_fish["first_opposing_bout_latency_s"][i] = latency_s
                    per_fish["first_opposing_bout_score"][i] = score

        total = correct + opposing + ambiguous
        classified = correct + opposing
        per_fish["bout_count_total"][i] = total
        per_fish["bout_count_correct"][i] = correct
        per_fish["bout_count_opposing"][i] = opposing
        per_fish["bout_count_ambiguous"][i] = ambiguous
        per_fish["bout_fraction_correct_classified"][i] = _safe_ratio(correct, classified)
        per_fish["bout_fraction_correct_all"][i] = _safe_ratio(correct, total)
        per_fish["bout_choice_index"][i] = _safe_ratio(correct - opposing, classified)
        weighted = _weighted_bout_omr_summary(
            track_labels,
            track_parallel,
            track_displacement,
            track_path,
        )
        for key, value in weighted.items():
            per_fish[key][i] = value

    per_bout = {
        "fish_id": np.array(all_fish_id, dtype=np.int32),
        "bout_id": np.array(all_bout_id, dtype=np.int32),
        "start_frame": np.array(all_start, dtype=np.int64),
        "end_frame": np.array(all_end, dtype=np.int64),
        "per_bout_omr_score": np.array(all_score, dtype=np.float32),
        "parallel_displacement_mm": np.array(all_parallel, dtype=np.float32),
        "bout_displacement_mm": np.array(all_displacement, dtype=np.float32),
        "bout_path_length_mm": np.array(all_path, dtype=np.float32),
        "correct_label": np.array(all_label, dtype=np.int8),
        "quality_flag": np.array(all_quality, dtype=np.int8),
    }

    windows = _compute_omr_windows(
        tracks,
        step,
        direction_xy,
        fps,
        moving_threshold_mm_s=moving_threshold_mm_s,
        projection_speed_deadzone_mm_s=projection_speed_deadzone_mm_s,
        window_lengths_s=window_lengths_s,
        bouts_by_fish=bouts_by_fish,
        arena_center_mm=arena_center_mm,
        arena_axis_extent_mm=arena_axis_extent_mm,
    )

    early_windows = _compute_omr_early_windows(
        tracks,
        step,
        direction_xy,
        fps,
        moving_threshold_mm_s=moving_threshold_mm_s,
        projection_deadzone=projection_deadzone,
        projection_speed_deadzone_mm_s=projection_speed_deadzone_mm_s,
        early_window_lengths_s=early_window_lengths_s,
        bouts_by_fish=bouts_by_fish,
        arena_center_mm=arena_center_mm,
        arena_axis_extent_mm=arena_axis_extent_mm,
    )

    attrs = {
        "method_version": OMR_METHOD_VERSION,
        "stimulus_direction_source": "static_step_params",
        "stimulus_direction_deg": float(grating_dir_deg),
        "detector_estimator_policy": "bout_boundaries_from_detector_physical_metrics_from_positions",
        "position_source_array": "positions_mm",
        "position_anchor": position_anchor,
        "speed_source_array": "speed_smoothed_mm",
        "arena_center_mm": list(arena_center_mm) if arena_center_mm is not None else None,
        "arena_axis_extent_mm": (
            float(arena_axis_extent_mm)
            if arena_axis_extent_mm is not None and np.isfinite(arena_axis_extent_mm)
            else None
        ),
        "arena_geometry_source": arena_geometry_source,
        "arena_position_axis_definition": (
            "dot(position_mm - arena_center_mm, stimulus_direction_xy); "
            "normalized values divide by arena_axis_extent_mm"
        ),
        "projection_deadzone": float(projection_deadzone),
        "projection_speed_deadzone_mm_s": float(projection_speed_deadzone_mm_s),
        "moving_threshold_mm_s": float(moving_threshold_mm_s),
        "early_response_window_lengths_s": [float(v) for v in early_window_lengths_s],
        "weighted_bout_metric_policy": (
            "bout_path_index includes all finite bout path; weighted correct fractions "
            "include only aligned/opposing classifiable bouts"
        ),
        "quality_flag_codes": {
            "0": "ok",
            "1": "no_valid_transitions_or_invalid_bout",
            "2": "no_movement",
        },
    }
    return OMRStepData(
        per_fish=per_fish,
        per_bout=per_bout,
        windows=windows,
        early_windows=early_windows,
        attrs=attrs,
    )


def _compute_omr_windows(
    tracks: Sequence[DenseTrack],
    step: ProtocolStep,
    direction_xy: np.ndarray,
    fps: float,
    *,
    moving_threshold_mm_s: float,
    projection_speed_deadzone_mm_s: float,
    window_lengths_s: Sequence[float],
    bouts_by_fish: Optional[Dict[int, List[BoutEntry]]] = None,
    arena_center_mm: Optional[Tuple[float, float]] = None,
    arena_axis_extent_mm: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """Compute non-overlapping windowed OMR metrics for a grating step."""

    full_length_s = float(step.duration_s)
    requested_lengths = [
        float(v) for v in window_lengths_s
        if float(v) > 0.0 and (full_length_s <= 0.0 or float(v) < full_length_s)
    ]
    if full_length_s > 0.0 and not any(abs(v - full_length_s) < 1e-6 for v in requested_lengths):
        requested_lengths.append(full_length_s)

    window_id: List[int] = []
    fish_id: List[int] = []
    start_frame: List[int] = []
    end_frame: List[int] = []
    start_time_s: List[float] = []
    end_time_s: List[float] = []
    window_length_s_out: List[float] = []
    omr_path_index: List[float] = []
    time_choice_index: List[float] = []
    coverage_fraction: List[float] = []
    mean_position_axis_norm: List[float] = []
    fraction_time_correct_side: List[float] = []
    n_bouts: List[int] = []
    quality_flag: List[int] = []

    wid = 0
    for window_length_s in requested_lengths:
        window_frames = max(1, int(round(window_length_s * fps))) if fps > 0 else max(1, step.end_frame - step.start_frame)
        cursor = int(step.start_frame)
        while cursor < int(step.end_frame):
            w_start = cursor
            w_end = min(cursor + window_frames, int(step.end_frame))
            actual_len_s = (w_end - w_start) / fps if fps > 0 else 0.0
            for track in tracks:
                summary = _omr_summary_for_window(
                    track,
                    w_start,
                    w_end,
                    direction_xy,
                    fps,
                    moving_threshold_mm_s,
                    projection_speed_deadzone_mm_s,
                    arena_center_mm,
                    arena_axis_extent_mm,
                )
                bouts = []
                if bouts_by_fish is not None:
                    bouts = [
                        b for b in bouts_by_fish.get(track.fish_id, [])
                        if b.start_frame < w_end and b.end_frame >= w_start
                    ]
                window_id.append(wid)
                fish_id.append(track.fish_id)
                start_frame.append(w_start)
                end_frame.append(w_end)
                start_time_s.append((w_start - step.start_frame) / fps if fps > 0 else 0.0)
                end_time_s.append((w_end - step.start_frame) / fps if fps > 0 else 0.0)
                window_length_s_out.append(actual_len_s)
                omr_path_index.append(float(summary["omr_path_index"]))
                time_choice_index.append(float(summary["time_choice_index"]))
                coverage_fraction.append(float(summary["coverage_fraction"]))
                mean_position_axis_norm.append(float(summary["mean_position_axis_norm"]))
                fraction_time_correct_side.append(float(summary["fraction_time_correct_side"]))
                n_bouts.append(len(bouts))
                quality_flag.append(int(summary["quality_flag"]))
            wid += 1
            cursor = w_end

    return {
        "window_id": np.array(window_id, dtype=np.int32),
        "fish_id": np.array(fish_id, dtype=np.int32),
        "start_frame": np.array(start_frame, dtype=np.int64),
        "end_frame": np.array(end_frame, dtype=np.int64),
        "start_time_s": np.array(start_time_s, dtype=np.float32),
        "end_time_s": np.array(end_time_s, dtype=np.float32),
        "window_length_s": np.array(window_length_s_out, dtype=np.float32),
        "omr_path_index": np.array(omr_path_index, dtype=np.float32),
        "time_choice_index": np.array(time_choice_index, dtype=np.float32),
        "coverage_fraction": np.array(coverage_fraction, dtype=np.float32),
        "mean_position_axis_norm": np.array(mean_position_axis_norm, dtype=np.float32),
        "fraction_time_correct_side": np.array(fraction_time_correct_side, dtype=np.float32),
        "n_bouts": np.array(n_bouts, dtype=np.int32),
        "quality_flag": np.array(quality_flag, dtype=np.int8),
    }


def _compute_omr_early_windows(
    tracks: Sequence[DenseTrack],
    step: ProtocolStep,
    direction_xy: np.ndarray,
    fps: float,
    *,
    moving_threshold_mm_s: float,
    projection_deadzone: float,
    projection_speed_deadzone_mm_s: float,
    early_window_lengths_s: Sequence[float],
    bouts_by_fish: Optional[Dict[int, List[BoutEntry]]] = None,
    arena_center_mm: Optional[Tuple[float, float]] = None,
    arena_axis_extent_mm: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """Compute fixed-from-onset early OMR summaries for each grating step."""

    requested_lengths = sorted({float(v) for v in early_window_lengths_s if float(v) > 0.0})

    window_id: List[int] = []
    fish_id: List[int] = []
    start_frame: List[int] = []
    end_frame: List[int] = []
    window_length_s_out: List[float] = []
    actual_window_length_s: List[float] = []
    omr_path_index: List[float] = []
    omr_net_direction_index: List[float] = []
    parallel_displacement_mm: List[float] = []
    net_displacement_mm: List[float] = []
    path_length_mm: List[float] = []
    time_fraction_correct_classified: List[float] = []
    time_choice_index: List[float] = []
    coverage_fraction: List[float] = []
    start_position_axis_norm: List[float] = []
    end_position_axis_norm: List[float] = []
    mean_position_axis_norm: List[float] = []
    fraction_time_correct_side: List[float] = []
    n_bouts: List[int] = []
    n_aligned_bouts: List[int] = []
    n_opposing_bouts: List[int] = []
    n_ambiguous_bouts: List[int] = []
    bout_path_index: List[float] = []
    bout_fraction_correct_weighted_by_path: List[float] = []
    bout_fraction_correct_weighted_by_displacement: List[float] = []
    quality_flag: List[int] = []

    for wid, window_length_s in enumerate(requested_lengths):
        window_frames = (
            max(1, int(math.ceil(window_length_s * fps)))
            if fps > 0 else max(1, int(step.end_frame) - int(step.start_frame))
        )
        w_start = int(step.start_frame)
        w_end = min(w_start + window_frames, int(step.end_frame))
        actual_len_s = (w_end - w_start) / fps if fps > 0 else 0.0

        for track in tracks:
            summary = _omr_summary_for_window(
                track,
                w_start,
                w_end,
                direction_xy,
                fps,
                moving_threshold_mm_s,
                projection_speed_deadzone_mm_s,
                arena_center_mm,
                arena_axis_extent_mm,
            )

            bouts = []
            if bouts_by_fish is not None:
                bouts = [
                    b for b in bouts_by_fish.get(track.fish_id, [])
                    if b.start_frame < w_end and b.end_frame >= w_start
                ]

            labels: List[int] = []
            parallels: List[float] = []
            displacements: List[float] = []
            paths: List[float] = []
            aligned_count = opposing_count = ambiguous_count = 0
            for bout in bouts:
                score, parallel, displacement, path = _bout_omr_score_for_bounds(
                    track,
                    bout,
                    w_start,
                    w_end,
                    direction_xy,
                )
                label, _quality = _bout_omr_label(score, projection_deadzone)
                if label > 0:
                    aligned_count += 1
                elif label < 0:
                    opposing_count += 1
                else:
                    ambiguous_count += 1
                labels.append(label)
                parallels.append(parallel)
                displacements.append(displacement)
                paths.append(path)

            weighted = _weighted_bout_omr_summary(labels, parallels, displacements, paths)

            window_id.append(wid)
            fish_id.append(track.fish_id)
            start_frame.append(w_start)
            end_frame.append(w_end)
            window_length_s_out.append(window_length_s)
            actual_window_length_s.append(actual_len_s)
            omr_path_index.append(float(summary["omr_path_index"]))
            omr_net_direction_index.append(float(summary["omr_net_direction_index"]))
            parallel_displacement_mm.append(float(summary["parallel_displacement_mm"]))
            net_displacement_mm.append(float(summary["net_displacement_mm"]))
            path_length_mm.append(float(summary["path_length_mm"]))
            time_fraction_correct_classified.append(float(summary["time_fraction_correct_classified"]))
            time_choice_index.append(float(summary["time_choice_index"]))
            coverage_fraction.append(float(summary["coverage_fraction"]))
            start_position_axis_norm.append(float(summary["start_position_axis_norm"]))
            end_position_axis_norm.append(float(summary["end_position_axis_norm"]))
            mean_position_axis_norm.append(float(summary["mean_position_axis_norm"]))
            fraction_time_correct_side.append(float(summary["fraction_time_correct_side"]))
            n_bouts.append(len(bouts))
            n_aligned_bouts.append(aligned_count)
            n_opposing_bouts.append(opposing_count)
            n_ambiguous_bouts.append(ambiguous_count)
            bout_path_index.append(float(weighted["bout_path_index"]))
            bout_fraction_correct_weighted_by_path.append(float(weighted["bout_fraction_correct_weighted_by_path"]))
            bout_fraction_correct_weighted_by_displacement.append(
                float(weighted["bout_fraction_correct_weighted_by_displacement"])
            )
            quality_flag.append(int(summary["quality_flag"]))

    return {
        "window_id": np.array(window_id, dtype=np.int32),
        "fish_id": np.array(fish_id, dtype=np.int32),
        "start_frame": np.array(start_frame, dtype=np.int64),
        "end_frame": np.array(end_frame, dtype=np.int64),
        "window_length_s": np.array(window_length_s_out, dtype=np.float32),
        "actual_window_length_s": np.array(actual_window_length_s, dtype=np.float32),
        "omr_path_index": np.array(omr_path_index, dtype=np.float32),
        "omr_net_direction_index": np.array(omr_net_direction_index, dtype=np.float32),
        "parallel_displacement_mm": np.array(parallel_displacement_mm, dtype=np.float32),
        "net_displacement_mm": np.array(net_displacement_mm, dtype=np.float32),
        "path_length_mm": np.array(path_length_mm, dtype=np.float32),
        "time_fraction_correct_classified": np.array(time_fraction_correct_classified, dtype=np.float32),
        "time_choice_index": np.array(time_choice_index, dtype=np.float32),
        "coverage_fraction": np.array(coverage_fraction, dtype=np.float32),
        "start_position_axis_norm": np.array(start_position_axis_norm, dtype=np.float32),
        "end_position_axis_norm": np.array(end_position_axis_norm, dtype=np.float32),
        "mean_position_axis_norm": np.array(mean_position_axis_norm, dtype=np.float32),
        "fraction_time_correct_side": np.array(fraction_time_correct_side, dtype=np.float32),
        "n_bouts": np.array(n_bouts, dtype=np.int32),
        "n_aligned_bouts": np.array(n_aligned_bouts, dtype=np.int32),
        "n_opposing_bouts": np.array(n_opposing_bouts, dtype=np.int32),
        "n_ambiguous_bouts": np.array(n_ambiguous_bouts, dtype=np.int32),
        "bout_path_index": np.array(bout_path_index, dtype=np.float32),
        "bout_fraction_correct_weighted_by_path": np.array(
            bout_fraction_correct_weighted_by_path,
            dtype=np.float32,
        ),
        "bout_fraction_correct_weighted_by_displacement": np.array(
            bout_fraction_correct_weighted_by_displacement,
            dtype=np.float32,
        ),
        "quality_flag": np.array(quality_flag, dtype=np.int8),
    }


def compute_global_omr_metrics(
    fish_ids: Sequence[int],
    step_omr_data: Sequence["OMRStepData"],
) -> Dict[str, np.ndarray]:
    """Aggregate OMR metrics across all eligible moving-grating steps."""

    n_fish = len(fish_ids)
    fish_id_arr = np.array(list(fish_ids), dtype=np.int32)
    eligible_step_count = np.zeros(n_fish, dtype=np.int32)
    eligible_window_count = np.zeros(n_fish, dtype=np.int32)
    omr_path_sum = np.zeros(n_fish, dtype=np.float64)
    omr_path_count = np.zeros(n_fish, dtype=np.int32)
    total_parallel = np.zeros(n_fish, dtype=np.float64)
    total_path = np.zeros(n_fish, dtype=np.float64)
    total_bouts = np.zeros(n_fish, dtype=np.int32)
    total_correct = np.zeros(n_fish, dtype=np.int32)
    total_opposing = np.zeros(n_fish, dtype=np.int32)
    total_ambiguous = np.zeros(n_fish, dtype=np.int32)
    total_bout_parallel = np.zeros(n_fish, dtype=np.float64)
    total_bout_path = np.zeros(n_fish, dtype=np.float64)
    total_bout_displacement = np.zeros(n_fish, dtype=np.float64)
    total_bout_classified_path = np.zeros(n_fish, dtype=np.float64)
    total_bout_classified_displacement = np.zeros(n_fish, dtype=np.float64)
    total_bout_weighted_path_correct_numerator = np.zeros(n_fish, dtype=np.float64)
    total_bout_weighted_displacement_correct_numerator = np.zeros(n_fish, dtype=np.float64)
    total_time_correct = np.zeros(n_fish, dtype=np.float64)
    total_time_opposing = np.zeros(n_fish, dtype=np.float64)
    coverage_sum = np.zeros(n_fish, dtype=np.float64)
    coverage_count = np.zeros(n_fish, dtype=np.int32)
    correct_side_sum = np.zeros(n_fish, dtype=np.float64)
    correct_side_count = np.zeros(n_fish, dtype=np.int32)
    start_axis_norm_sum = np.zeros(n_fish, dtype=np.float64)
    start_axis_norm_count = np.zeros(n_fish, dtype=np.int32)
    end_axis_norm_sum = np.zeros(n_fish, dtype=np.float64)
    end_axis_norm_count = np.zeros(n_fish, dtype=np.int32)
    mean_axis_norm_sum = np.zeros(n_fish, dtype=np.float64)
    mean_axis_norm_count = np.zeros(n_fish, dtype=np.int32)
    min_first_aligned_latency = np.full(n_fish, np.inf, dtype=np.float64)

    fish_to_idx = {int(fid): i for i, fid in enumerate(fish_id_arr)}
    for omr in step_omr_data:
        pf = omr.per_fish
        for row, fid_raw in enumerate(pf["fish_id"]):
            idx = fish_to_idx.get(int(fid_raw))
            if idx is None:
                continue
            eligible_step_count[idx] += 1
            path_index = float(pf["omr_path_index"][row])
            if np.isfinite(path_index):
                omr_path_sum[idx] += path_index
                omr_path_count[idx] += 1
            total_parallel[idx] += float(pf["parallel_displacement_mm"][row])
            total_path[idx] += float(pf["path_length_mm"][row])
            total_bouts[idx] += int(pf["bout_count_total"][row])
            total_correct[idx] += int(pf["bout_count_correct"][row])
            total_opposing[idx] += int(pf["bout_count_opposing"][row])
            total_ambiguous[idx] += int(pf["bout_count_ambiguous"][row])
            bout_parallel = float(pf.get("bout_parallel_displacement_sum_mm", np.zeros_like(pf["fish_id"]))[row])
            bout_path = float(pf.get("bout_path_length_sum_mm", np.zeros_like(pf["fish_id"]))[row])
            bout_displacement = float(pf.get("bout_displacement_sum_mm", np.zeros_like(pf["fish_id"]))[row])
            bout_classified_path = float(
                pf.get("bout_classified_path_length_sum_mm", np.zeros_like(pf["fish_id"]))[row]
            )
            bout_classified_displacement = float(
                pf.get("bout_classified_displacement_sum_mm", np.zeros_like(pf["fish_id"]))[row]
            )
            weighted_path_fraction = float(
                pf.get("bout_fraction_correct_weighted_by_path", np.full_like(pf["fish_id"], np.nan, dtype=np.float32))[row]
            )
            weighted_displacement_fraction = float(
                pf.get(
                    "bout_fraction_correct_weighted_by_displacement",
                    np.full_like(pf["fish_id"], np.nan, dtype=np.float32),
                )[row]
            )
            total_bout_parallel[idx] += bout_parallel
            total_bout_path[idx] += bout_path
            total_bout_displacement[idx] += bout_displacement
            total_bout_classified_path[idx] += bout_classified_path
            total_bout_classified_displacement[idx] += bout_classified_displacement
            if np.isfinite(weighted_path_fraction):
                total_bout_weighted_path_correct_numerator[idx] += (
                    weighted_path_fraction * bout_classified_path
                )
            if np.isfinite(weighted_displacement_fraction):
                total_bout_weighted_displacement_correct_numerator[idx] += (
                    weighted_displacement_fraction * bout_classified_displacement
                )
            total_time_correct[idx] += float(pf["time_correct_s"][row])
            total_time_opposing[idx] += float(pf["time_opposing_s"][row])
            coverage = float(pf["coverage_fraction"][row])
            if np.isfinite(coverage):
                coverage_sum[idx] += coverage
                coverage_count[idx] += 1
            correct_side = float(pf["fraction_time_correct_side"][row])
            if np.isfinite(correct_side):
                correct_side_sum[idx] += correct_side
                correct_side_count[idx] += 1
            start_axis = float(pf["start_position_axis_norm"][row])
            if np.isfinite(start_axis):
                start_axis_norm_sum[idx] += start_axis
                start_axis_norm_count[idx] += 1
            end_axis = float(pf["end_position_axis_norm"][row])
            if np.isfinite(end_axis):
                end_axis_norm_sum[idx] += end_axis
                end_axis_norm_count[idx] += 1
            mean_axis = float(pf["mean_position_axis_norm"][row])
            if np.isfinite(mean_axis):
                mean_axis_norm_sum[idx] += mean_axis
                mean_axis_norm_count[idx] += 1
            aligned_latency = float(pf["first_aligned_bout_latency_s"][row])
            if np.isfinite(aligned_latency):
                min_first_aligned_latency[idx] = min(min_first_aligned_latency[idx], aligned_latency)

        if "fish_id" in omr.windows:
            for fid_raw in omr.windows["fish_id"]:
                idx = fish_to_idx.get(int(fid_raw))
                if idx is not None:
                    eligible_window_count[idx] += 1

    classified_bouts = total_correct + total_opposing
    classified_time = total_time_correct + total_time_opposing

    omr_path_index_mean = np.full(n_fish, np.nan, dtype=np.float32)
    omr_path_index_weighted = np.full(n_fish, np.nan, dtype=np.float32)
    bout_fraction = np.full(n_fish, np.nan, dtype=np.float32)
    bout_choice = np.full(n_fish, np.nan, dtype=np.float32)
    bout_path_index = np.full(n_fish, np.nan, dtype=np.float32)
    bout_fraction_weighted_by_path = np.full(n_fish, np.nan, dtype=np.float32)
    bout_fraction_weighted_by_displacement = np.full(n_fish, np.nan, dtype=np.float32)
    time_choice = np.full(n_fish, np.nan, dtype=np.float32)
    coverage_fraction = np.full(n_fish, np.nan, dtype=np.float32)
    mean_fraction_time_correct_side = np.full(n_fish, np.nan, dtype=np.float32)
    mean_start_position_axis_norm = np.full(n_fish, np.nan, dtype=np.float32)
    mean_end_position_axis_norm = np.full(n_fish, np.nan, dtype=np.float32)
    mean_mean_position_axis_norm = np.full(n_fish, np.nan, dtype=np.float32)
    first_aligned_bout_latency_s_min = np.full(n_fish, np.nan, dtype=np.float32)
    quality_flag = np.zeros(n_fish, dtype=np.int8)

    for i in range(n_fish):
        omr_path_index_mean[i] = _safe_ratio(omr_path_sum[i], float(omr_path_count[i]))
        omr_path_index_weighted[i] = _safe_ratio(total_parallel[i], total_path[i])
        bout_fraction[i] = _safe_ratio(float(total_correct[i]), float(classified_bouts[i]))
        bout_choice[i] = _safe_ratio(float(total_correct[i] - total_opposing[i]), float(classified_bouts[i]))
        bout_path_index[i] = _safe_ratio(total_bout_parallel[i], total_bout_path[i])
        bout_fraction_weighted_by_path[i] = _safe_ratio(
            total_bout_weighted_path_correct_numerator[i],
            total_bout_classified_path[i],
        )
        bout_fraction_weighted_by_displacement[i] = _safe_ratio(
            total_bout_weighted_displacement_correct_numerator[i],
            total_bout_classified_displacement[i],
        )
        time_choice[i] = _safe_ratio(total_time_correct[i] - total_time_opposing[i], classified_time[i])
        coverage_fraction[i] = _safe_ratio(coverage_sum[i], float(coverage_count[i]))
        mean_fraction_time_correct_side[i] = _safe_ratio(
            correct_side_sum[i], float(correct_side_count[i]),
        )
        mean_start_position_axis_norm[i] = _safe_ratio(
            start_axis_norm_sum[i], float(start_axis_norm_count[i]),
        )
        mean_end_position_axis_norm[i] = _safe_ratio(
            end_axis_norm_sum[i], float(end_axis_norm_count[i]),
        )
        mean_mean_position_axis_norm[i] = _safe_ratio(
            mean_axis_norm_sum[i], float(mean_axis_norm_count[i]),
        )
        if np.isfinite(min_first_aligned_latency[i]):
            first_aligned_bout_latency_s_min[i] = min_first_aligned_latency[i]
        if eligible_step_count[i] == 0:
            quality_flag[i] = 1
        elif total_path[i] <= 0.0:
            quality_flag[i] = 2

    return {
        "fish_id": fish_id_arr,
        "eligible_step_count": eligible_step_count,
        "eligible_window_count": eligible_window_count,
        "omr_path_index_mean": omr_path_index_mean,
        "omr_path_index_weighted_by_path": omr_path_index_weighted,
        "bout_fraction_correct_classified": bout_fraction,
        "bout_choice_index": bout_choice,
        "bout_path_index": bout_path_index,
        "bout_fraction_correct_weighted_by_path": bout_fraction_weighted_by_path,
        "bout_fraction_correct_weighted_by_displacement": bout_fraction_weighted_by_displacement,
        "time_choice_index": time_choice,
        "mean_fraction_time_correct_side": mean_fraction_time_correct_side,
        "mean_start_position_axis_norm": mean_start_position_axis_norm,
        "mean_end_position_axis_norm": mean_end_position_axis_norm,
        "mean_mean_position_axis_norm": mean_mean_position_axis_norm,
        "first_aligned_bout_latency_s_min": first_aligned_bout_latency_s_min,
        "total_path_length_mm": total_path.astype(np.float32),
        "total_parallel_displacement_mm": total_parallel.astype(np.float32),
        "total_bouts": total_bouts,
        "total_bout_correct": total_correct,
        "total_bout_opposing": total_opposing,
        "total_bout_ambiguous": total_ambiguous,
        "total_bout_parallel_displacement_mm": total_bout_parallel.astype(np.float32),
        "total_bout_path_length_mm": total_bout_path.astype(np.float32),
        "total_bout_displacement_mm": total_bout_displacement.astype(np.float32),
        "coverage_fraction": coverage_fraction,
        "quality_flag": quality_flag,
    }


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


@dataclass
class OMRStepData:
    """OMR responsiveness outputs for one MOVING_GRATING step."""

    per_fish: Dict[str, np.ndarray]
    per_bout: Dict[str, np.ndarray]
    windows: Dict[str, np.ndarray]
    early_windows: Dict[str, np.ndarray]
    attrs: Dict[str, Any]


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
    console: Optional[Console] = None,
) -> str:
    """Write stimulus response run to zarr."""
    console = console or Console()

    analysis = root.require_group("analysis")
    parent = analysis.require_group("stimulus_response_runs")

    if run_name is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_name = f"stimulus_response_{ts}"

    if run_name in parent and not overwrite:
        raise ValueError(f"Stimulus response run '{run_name}' already exists.")
    if run_name in parent:
        del parent[run_name]

    run_group = parent.create_group(run_name)
    parent.attrs["latest"] = run_name

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
        help="Skip OMR responsiveness metrics for MOVING_GRATING steps.",
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
    for step in steps:
        if step.stimulus_mode == _CONCENTRIC_GRATING:
            center = resolve_concentric_center_mm(
                root,
                flatten_stimulus_params(step.stimulus_params),
                stimulus_run=stim_run,
            )
            if center is not None:
                concentric_centers[step.index] = center
            else:
                console.print(
                    f"  [yellow]Warning: could not resolve center for step {step.index}; "
                    f"skipping concentric metrics.[/yellow]"
                )

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
            result["concentric"] = ConcentricStepData(per_frame=cpf, per_fish=cpfish, time_series=cts)

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
        "moving_threshold_mm_s": args.moving_threshold_mm_s,
        "fps": fps,
        "n_frames": n_frames,
        "camera_to_projector_offset_deg": args.camera_to_projector_offset_deg,
        "bin_size_s": args.bin_size_s,
        "follow_threshold": args.follow_threshold,
        "follow_window_s": args.follow_window_s,
        "omr_enabled": not args.no_omr,
        "omr_method_version": OMR_METHOD_VERSION if not args.no_omr else None,
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
