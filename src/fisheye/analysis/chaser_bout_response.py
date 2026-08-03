"""Bout-level object-relative kinematics, with wall and virtual-object controls.

What the fish does near a static object is an *event*, and events are rare: on a 600 s
epoch the fish comes within 12 mm of a given object only 4-10 times. Any statistic built
on approach events is descriptive for a single fish. Bouts are 30-50x denser -- 100-400 of
them start within 15 mm of an object per epoch -- so the bout is the unit with power.

For every bout this module records, per reference object:

    distance_at_onset_mm      how far the object was when the bout began
    bearing_at_onset_deg      where the object was relative to the fish's heading
    turn_deg                  the heading change the bout executed
    turn_toward               did that turn rotate the fish toward the object's side
    delta_distance_mm         did the bout carry the fish away from the object

and aggregates them against distance, alongside a frame-level *circling index*
(|tangential velocity| / speed), which separates orbiting around an object from
approaching or retreating from it.

THE CONFOUND THIS MODULE EXISTS TO CONTROL
------------------------------------------
The fish wall-follows (~35% of frames in the perimeter band) and the objects sit close to
the wall. A wall-following fish traces an arc that automatically sweeps around any
wall-adjacent object, manufacturing angular sweep, tangential velocity, and a turn-toward
bias with no object involvement at all. Every "orbiting" signature can be produced this way.

So each real object gets a set of **virtual objects**: its own position rotated about the
arena centre. A virtual object has, by construction, exactly the same distance-from-centre
and the same wall proximity as the real one at every instant -- it simply is not there. If
the circling signature appears around the virtual objects too, it is the wall.

`circling_excess_vs_virtual` and `turn_bias_excess_vs_virtual` are the object-specific
quantities: the real object's value minus the mean over its virtual twins. Everything else
is a diagnostic on the way there. Wall-excluded twins of every aggregate are also written.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
import json
import math
from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import zarr  # noqa: E402

from fisheye.analysis.chaser_distance_runs import _bytes_array, _write_array
from fisheye.analysis.swim_bout_io import (
    SwimBoutIOError,
    resolve_swim_bout_run_name,
)
from fisheye.analysis.chaser_distance_io import (
    reject_unsealed_chaser_derived_publication,
)
from fisheye.analysis.chaser_radial_occupancy import (
    ChaserRadialEpoch,
    _apply_settle_trim,
    _normalize_float_array,
    _open_root,
    _protocol_position_transition_s,
    _read_epochs,
    _resolve_arena_geometry,
    _resolve_chaser_distance_run,
    _wall_mask,
)
from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.plot_artifacts import write_interactive_plot_spec_artifact, write_png_visualization_artifact
from fisheye.shared.run_lineage_fingerprint import build_run_lineage_payload, write_run_lineage_attrs
from fisheye.shared.system_metadata import get_git_info
from fisheye.shared.zarr_run_completion import resolve_authoritative_run_name


SCHEMA_ID = "palette.chaser_bout_response.v1"
SCHEMA_VERSION = 1
METHOD = "chaser_relative_bout_kinematics_with_virtual_controls"
METHOD_VERSION = "1"
COMPONENT_PARENT_NAME = "chaser_bout_response"
DEFAULT_COMPONENT_NAME = "chaser_bout_response_v1"

CIRCLING_PNG_ARTIFACT_NAME = "chaser_bout_response_circling_png"
TURN_BIAS_PNG_ARTIFACT_NAME = "chaser_bout_response_turn_bias_png"
BOUT_KINEMATICS_PNG_ARTIFACT_NAME = "chaser_bout_response_kinematics_png"
INTERACTIVE_ARTIFACT_NAME = "chaser_bout_response_interactive"
INTERACTIVE_RENDERER = "palette-chaser-bout-response-v1"
INTERACTIVE_SPEC_SCHEMA_ID = "palette.chaser_bout_response.interactive_spec.v1"

DEFAULT_DISTANCE_BIN_EDGES_MM = (0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 25.0, 30.0, 40.0, 60.0)
DEFAULT_NEAR_DISTANCE_MM = 15.0
# Rotations of each object's own position about the arena centre. 90 deg is deliberately
# absent from the default: with objects on opposite corners it lands a "virtual" reference
# straight on top of the other real object.
DEFAULT_VIRTUAL_ROTATIONS_DEG = (60.0, 120.0, 180.0, 240.0, 300.0)
DEFAULT_MIN_VIRTUAL_SEPARATION_MM = 8.0
DEFAULT_PERIMETER_BAND_MM = 5.0
DEFAULT_MOVING_SPEED_THRESHOLD_MM_S = 2.0
DEFAULT_VISIT_ENTER_MM = 15.0
DEFAULT_VISIT_EXIT_MM = 20.0
# Below this many independent visits, a bout-level correlation is pseudoreplicated: the bouts
# are clustered inside a handful of excursions and the effective n is the visit count, not the
# bout count. Warn rather than let someone compute a p-value off n=128 bouts from 4 visits.
DEFAULT_MIN_VISITS_FOR_INFERENCE = 10
DEFAULT_MIN_BIN_BOUTS = 20
DEFAULT_MIN_BIN_FRAMES = 20
DEFAULT_MOTION_SPREAD_THRESHOLD_MM = 1.0

REFERENCE_KIND_OBJECT = "object"
REFERENCE_KIND_VIRTUAL = "virtual"
REFERENCE_KIND_DISH_CENTER = "dish_center"


@dataclass(frozen=True)
class BoutReference:
    label: str
    kind: str
    chaser_index: int       # -1 for dish_center
    parent_chaser_index: int  # for virtual: the object it was rotated from; else -1
    rotation_deg: float     # for virtual; else nan


@dataclass(frozen=True)
class ChaserBoutResponseResult:
    zarr_path: str
    recording_id: str
    component_name: str
    chaser_distance_run_name: str
    chaser_distance_run_path: str
    source_swim_bout_run: str
    source_swim_bout_path: str
    source_egocentric_component: str
    fps: float
    total_frames: int
    pixels_per_mm_projector: float
    geometry_status: str
    arena_center_x_px: float | None
    arena_center_y_px: float | None
    arena_radius_px: float | None
    perimeter_band_mm: float
    near_distance_mm: float
    moving_speed_threshold_mm_s: float
    min_bin_bouts: int
    min_bin_frames: int
    visit_enter_mm: float
    visit_exit_mm: float
    min_visits_for_inference: int
    settle_trim_s: float
    distance_bin_edges_mm: np.ndarray
    distance_bin_centers_mm: np.ndarray
    epochs: tuple[ChaserRadialEpoch, ...]
    references: tuple[BoutReference, ...]
    # per-bout (n_bouts,)
    bout_id: np.ndarray
    bout_epoch_index: np.ndarray
    bout_start_frame: np.ndarray
    bout_end_frame: np.ndarray
    bout_turn_deg: np.ndarray
    bout_peak_speed_mm_s: np.ndarray
    bout_mean_speed_mm_s: np.ndarray
    bout_duration_s: np.ndarray
    bout_path_length_mm: np.ndarray
    bout_net_displacement_mm: np.ndarray
    bout_tortuosity: np.ndarray
    bout_wall_at_onset: np.ndarray
    bout_valid: np.ndarray
    # per-bout per-reference (n_bouts, n_refs)
    bout_distance_at_onset_mm: np.ndarray
    bout_bearing_at_onset_deg: np.ndarray
    bout_delta_distance_mm: np.ndarray
    bout_turn_toward: np.ndarray
    bout_visit_id: np.ndarray
    bout_predicted_miss_onset_mm: np.ndarray
    bout_predicted_miss_end_mm: np.ndarray
    bout_delta_predicted_miss_mm: np.ndarray
    bout_approaching_at_onset: np.ndarray
    # binned (epoch, reference, distance_bin)
    bout_count: np.ndarray
    bout_rate_per_min: np.ndarray
    time_in_bin_s: np.ndarray
    median_peak_speed_mm_s: np.ndarray
    median_path_length_mm: np.ndarray
    mean_abs_turn_deg: np.ndarray
    turn_toward_fraction: np.ndarray
    turn_bias_r: np.ndarray
    mean_delta_distance_mm: np.ndarray
    mean_delta_predicted_miss_mm: np.ndarray
    fraction_bouts_widening_miss: np.ndarray
    approach_bout_count: np.ndarray
    circling_index: np.ndarray
    radial_velocity_mm_s: np.ndarray
    tangential_speed_mm_s: np.ndarray
    frame_count: np.ndarray
    # wall-excluded twins
    bout_count_wall_excluded: np.ndarray
    turn_toward_fraction_wall_excluded: np.ndarray
    turn_bias_r_wall_excluded: np.ndarray
    circling_index_wall_excluded: np.ndarray
    frame_count_wall_excluded: np.ndarray
    # per (epoch, reference) near-band scalars
    near_turn_bias_r: np.ndarray
    near_turn_toward_fraction: np.ndarray
    near_circling_index: np.ndarray
    near_bout_count: np.ndarray
    near_visit_count: np.ndarray
    near_delta_predicted_miss_mm: np.ndarray
    near_fraction_widening_miss: np.ndarray
    # per (epoch, chaser) object-specific excess over its virtual twins
    circling_excess_vs_virtual: np.ndarray
    turn_bias_excess_vs_virtual: np.ndarray
    steering_excess_vs_virtual: np.ndarray
    steering_excess_by_band: np.ndarray
    circling_excess_by_band: np.ndarray
    status: str
    qc_warnings: tuple[str, ...]
    summary: dict[str, Any]
    diagnostics: dict[str, Any]


def _wrap_deg(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) + 180.0) % 360.0 - 180.0


def bearing_deg(rel_xy: np.ndarray, heading_deg: np.ndarray) -> np.ndarray:
    """Egocentric bearing to a reference, in the heading's frame.

    rel_xy: (frame, ref, 2) vector fish -> reference, in y-DOWN arena coordinates.
    heading_deg: (frame,) CCW-from-+x in a y-UP frame (the track-kinematics convention).

    The y is negated to bring the object vector into the heading's frame. Skipping that flip
    silently produces a plausible-looking but wrong bearing, and flips the sign of the turn
    bias -- see test_bearing_matches_canonical_egocentric_bearing.
    """

    rel = np.asarray(rel_xy, dtype=np.float64)
    angle = np.degrees(np.arctan2(-rel[:, :, 1], rel[:, :, 0]))
    return _wrap_deg(angle - np.asarray(heading_deg, dtype=np.float64).reshape(-1)[:, None])


def _resolve_swim_bout_run(root: zarr.Group, run_name: str) -> tuple[zarr.Group, str, str]:
    parent = root.get("analysis/swim_bout_runs")
    if parent is None:
        raise ValueError("Archive has no analysis/swim_bout_runs group; run bout detection first.")
    try:
        resolved = resolve_swim_bout_run_name(root, run_name=run_name)
    except SwimBoutIOError as exc:
        raise ValueError(
            "No stable complete selector-eligible swim-bout run was resolved; "
            "pass --swim-bout-run only for an exact current run."
        ) from exc
    return parent[resolved], resolved, f"analysis/swim_bout_runs/{resolved}"


def _select_bout_row_mask(
    bout_start: np.ndarray,
    bout_end: np.ndarray,
    *,
    total_frames: int,
    signal_id: np.ndarray | None,
    default_signal_id: Any,
) -> tuple[np.ndarray, int, str]:
    """Select valid bout rows from one run's default signal level."""

    keep = (bout_start >= 0) & (bout_end < total_frames) & (bout_end >= bout_start)
    bout_level_note = "single_level_table_no_signal_id"
    source_signal_id = -1
    if signal_id is not None:
        signal_id = np.asarray(signal_id, dtype=np.int64).reshape(-1)
        if default_signal_id is None:
            source_signal_id = int(np.min(signal_id))
            bout_level_note = (
                "multi_level_table_no_default_signal_id_attr_fell_back_to_"
                f"{source_signal_id}"
            )
        else:
            source_signal_id = int(default_signal_id)
            bout_level_note = (
                "multi_level_table_filtered_to_default_signal_id_"
                f"{source_signal_id}"
            )
        keep &= signal_id == source_signal_id
    return keep, source_signal_id, bout_level_note


def _resolve_heading(run_group: zarr.Group, *, total_frames: int) -> tuple[np.ndarray, np.ndarray, str]:
    """Per-camera-frame fish heading, taken from the egocentric-bearing component.

    That component already resolves heading onto the camera-frame axis of this distance run,
    which is exactly the join this module needs; re-deriving it from track kinematics would
    duplicate the alignment logic and risk drifting from it.
    """

    parent = run_group.get("egocentric_bearing")
    if parent is None:
        raise ValueError(
            "chaser_bout_response requires an egocentric_bearing component on the chaser-distance "
            "run (it supplies fish heading on the camera-frame axis). Run "
            "fisheye.analysis.chaser_egocentric_bearing first."
        )
    resolved = resolve_authoritative_run_name(parent, legacy_default=False)
    if not resolved or resolved not in parent:
        raise ValueError(
            "No stable complete selector-eligible egocentric_bearing component found."
        )
    component = parent[resolved]
    frames = component.get("frames")
    if frames is None or "fish_heading_deg" not in frames or "fish_heading_valid" not in frames:
        raise ValueError("egocentric_bearing component lacks frames/fish_heading_deg.")
    heading = np.asarray(frames["fish_heading_deg"][:], dtype=np.float64).reshape(-1)
    valid = np.asarray(frames["fish_heading_valid"][:], dtype=bool).reshape(-1)
    if heading.shape[0] != int(total_frames):
        raise ValueError(
            f"egocentric_bearing heading length {heading.shape[0]} does not match distance-run "
            f"total_frames {total_frames}."
        )
    return heading, valid, resolved


def _build_references(
    *,
    chaser_indices: np.ndarray,
    chaser_labels: Sequence[str],
    chaser_xy_px: np.ndarray,
    center_xy_px: tuple[float, float],
    rotations_deg: Sequence[float],
    min_separation_mm: float,
    pixels_per_mm: float,
) -> tuple[tuple[BoutReference, ...], np.ndarray, list[str]]:
    """Real objects, their rotated virtual twins, and the dish centre.

    Returns the reference metadata and per-frame reference positions (frame, ref, xy) in px.
    A virtual twin that lands on top of a real object is dropped -- with objects on opposite
    corners a 90 deg rotation does exactly that.
    """

    n_frames = int(chaser_xy_px.shape[0])
    n_chasers = int(chaser_indices.shape[0])
    cx, cy = float(center_xy_px[0]), float(center_xy_px[1])
    min_sep_px = max(0.0, float(min_separation_mm)) * float(pixels_per_mm)

    references: list[BoutReference] = []
    positions: list[np.ndarray] = []
    notes: list[str] = []

    for c_idx in range(n_chasers):
        label = chaser_labels[c_idx] if c_idx < len(chaser_labels) and chaser_labels[c_idx] else f"chaser{int(chaser_indices[c_idx])}"
        references.append(
            BoutReference(
                label=label,
                kind=REFERENCE_KIND_OBJECT,
                chaser_index=int(chaser_indices[c_idx]),
                parent_chaser_index=-1,
                rotation_deg=math.nan,
            )
        )
        positions.append(np.asarray(chaser_xy_px[:, c_idx, :], dtype=np.float64))

    real_positions = [np.asarray(chaser_xy_px[:, c, :], dtype=np.float64) for c in range(n_chasers)]

    for c_idx in range(n_chasers):
        base = real_positions[c_idx]
        rel = base - np.asarray([cx, cy], dtype=np.float64)
        for angle in rotations_deg:
            theta = math.radians(float(angle))
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            rotated = np.stack(
                [rel[:, 0] * cos_t - rel[:, 1] * sin_t, rel[:, 0] * sin_t + rel[:, 1] * cos_t], axis=1
            ) + np.asarray([cx, cy], dtype=np.float64)
            collision = False
            for other in range(n_chasers):
                gap = np.hypot(rotated[:, 0] - real_positions[other][:, 0], rotated[:, 1] - real_positions[other][:, 1])
                finite = gap[np.isfinite(gap)]
                if finite.size and float(np.nanmedian(finite)) < min_sep_px:
                    collision = True
                    break
            label = f"virtual_{references[c_idx].label}_{int(round(float(angle)))}"
            if collision:
                notes.append(f"virtual_reference_dropped_on_real_object:{label}")
                continue
            references.append(
                BoutReference(
                    label=label,
                    kind=REFERENCE_KIND_VIRTUAL,
                    chaser_index=-1,
                    parent_chaser_index=int(chaser_indices[c_idx]),
                    rotation_deg=float(angle),
                )
            )
            positions.append(rotated)

    references.append(
        BoutReference(
            label=REFERENCE_KIND_DISH_CENTER,
            kind=REFERENCE_KIND_DISH_CENTER,
            chaser_index=-1,
            parent_chaser_index=-1,
            rotation_deg=math.nan,
        )
    )
    positions.append(np.tile(np.asarray([cx, cy], dtype=np.float64), (n_frames, 1)))

    stacked = np.stack(positions, axis=1)  # (frame, ref, xy)
    return tuple(references), stacked, notes


def _segment_visits(
    distance: np.ndarray,
    usable: np.ndarray,
    *,
    enter_mm: float,
    exit_mm: float,
) -> np.ndarray:
    """Label contiguous near-object excursions with a visit id (-1 outside any visit).

    Hysteresis: a visit opens when the fish comes within enter_mm and closes when it leaves
    past exit_mm. Bouts inside one visit are NOT independent samples of the fish's policy --
    they are one approach, subsampled. This id is what cluster-aware inference must resample.
    """

    d = np.asarray(distance, dtype=np.float64)
    ok = np.asarray(usable, dtype=bool)
    out = np.full(d.shape[0], -1, dtype=np.int32)
    in_visit = False
    visit_id = -1
    for i in range(d.shape[0]):
        if not ok[i] or not math.isfinite(d[i]):
            continue
        if not in_visit:
            if d[i] < float(enter_mm):
                in_visit = True
                visit_id += 1
                out[i] = visit_id
            continue
        if d[i] > float(exit_mm):
            in_visit = False
        else:
            out[i] = visit_id
    return out


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if int(ok.sum()) < 3:
        return math.nan
    a, b = a[ok], b[ok]
    if float(np.std(a)) <= 0 or float(np.std(b)) <= 0:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def _compute(
    *,
    epochs: Sequence[ChaserRadialEpoch],
    references: Sequence[BoutReference],
    ref_xy_px: np.ndarray,
    fish_xy_px: np.ndarray,
    fish_valid: np.ndarray,
    heading_deg: np.ndarray,
    heading_valid: np.ndarray,
    bout_start: np.ndarray,
    bout_end: np.ndarray,
    bout_peak_speed: np.ndarray,
    bout_mean_speed: np.ndarray,
    bout_duration_s: np.ndarray,
    bout_path_length: np.ndarray,
    bout_net_displacement: np.ndarray,
    bout_ids: np.ndarray,
    fps: float,
    pixels_per_mm: float,
    geometry,
    perimeter_band_mm: float,
    distance_bin_edges_mm: np.ndarray,
    near_distance_mm: float,
    moving_speed_threshold_mm_s: float,
    min_bin_bouts: int,
    min_bin_frames: int,
    visit_enter_mm: float,
    visit_exit_mm: float,
    min_visits_for_inference: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any], tuple[str, ...]]:
    n_epochs = len(epochs)
    n_refs = len(references)
    edges = np.asarray(distance_bin_edges_mm, dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    n_bins = int(centers.shape[0])
    safe_fps = float(fps) if math.isfinite(float(fps)) and float(fps) > 0 else 1.0
    ppm = float(pixels_per_mm)
    warnings: list[str] = []

    fish_mm = np.asarray(fish_xy_px, dtype=np.float64) / ppm
    ref_mm = np.asarray(ref_xy_px, dtype=np.float64) / ppm
    wall = _wall_mask(
        np.asarray(fish_xy_px, dtype=np.float64),
        geometry=geometry,
        perimeter_band_mm=float(perimeter_band_mm),
        pixels_per_mm=ppm,
    )

    # ---- per-frame distance/bearing to every reference ----
    rel = ref_mm - fish_mm[:, None, :]                       # (frame, ref, xy)
    dist = np.hypot(rel[:, :, 0], rel[:, :, 1])              # (frame, ref)
    bearing = bearing_deg(rel, heading_deg)                  # (frame, ref)

    # ---- assign bouts to epochs ----
    n_bouts = int(bout_start.shape[0])
    epoch_index = np.full(n_bouts, -1, dtype=np.int16)
    for e_idx, epoch in enumerate(epochs):
        sel = (bout_start >= epoch.start_frame) & (bout_start <= epoch.end_frame)
        epoch_index[sel] = e_idx

    onset_ok = fish_valid[bout_start] & heading_valid[bout_start]
    end_ok = fish_valid[bout_end] & heading_valid[bout_end]
    bout_valid = onset_ok & end_ok & (epoch_index >= 0)

    turn = np.full(n_bouts, np.nan, dtype=np.float64)
    turn[end_ok & onset_ok] = _wrap_deg(heading_deg[bout_end] - heading_deg[bout_start])[end_ok & onset_ok]

    d_onset = np.where(onset_ok[:, None], dist[bout_start, :], np.nan)
    b_onset = np.where(onset_ok[:, None], bearing[bout_start, :], np.nan)
    d_end = np.where(end_ok[:, None], dist[bout_end, :], np.nan)
    b_end = np.where(end_ok[:, None], bearing[bout_end, :], np.nan)
    delta_distance = d_end - d_onset

    # --- steering: what did this bout do to the fish's AIM? -------------------------------
    # Predicted miss distance b = r*sin(bearing): "if I keep going straight from here, by how
    # much do I pass the object?" Evaluated before and after the bout, its change is the only
    # clean read on whether the fish steers around the object.
    #
    # The trajectory-level version of this question is unusable: b <= r by construction, and
    # conditioning on approaches that DID get close forces b to converge on the CPA (at the
    # closest point b == r identically). Per bout there is no such conditioning -- we simply
    # ask what each bout did, whether or not the fish ever ends up close.
    predicted_miss_onset = d_onset * np.abs(np.sin(np.radians(b_onset)))
    predicted_miss_end = d_end * np.abs(np.sin(np.radians(b_end)))
    delta_predicted_miss = predicted_miss_end - predicted_miss_onset
    # Only meaningful while the object is ahead: with it behind, "predicted miss" is not a
    # thing the fish can steer.
    approaching_at_onset = np.isfinite(b_onset) & (np.abs(b_onset) < 90.0)
    delta_predicted_miss = np.where(approaching_at_onset, delta_predicted_miss, np.nan)
    # Same-signed bearing and turn == the bout rotated the fish toward the object's side.
    turn_toward = np.sign(turn)[:, None] == np.sign(b_onset)
    turn_toward &= np.isfinite(turn)[:, None] & np.isfinite(b_onset)

    net = np.asarray(bout_net_displacement, dtype=np.float64)
    tortuosity = np.divide(
        np.asarray(bout_path_length, dtype=np.float64), net,
        out=np.full(n_bouts, np.nan), where=net > 1e-6,
    )
    bout_wall = np.zeros(n_bouts, dtype=bool)
    bout_wall[onset_ok] = wall[bout_start[onset_ok]]

    # ---- frame-level radial / tangential decomposition ----
    v_fish = np.full(fish_mm.shape, np.nan, dtype=np.float64)
    v_fish[:-1] = (fish_mm[1:] - fish_mm[:-1]) * safe_fps
    pair = np.zeros(fish_mm.shape[0], dtype=bool)
    pair[:-1] = fish_valid[:-1] & fish_valid[1:]
    speed = np.hypot(v_fish[:, 0], v_fish[:, 1])

    shape3 = (n_epochs, n_refs, n_bins)
    shape2 = (n_epochs, n_refs)
    bout_count = np.zeros(shape3, dtype=np.int64)
    bout_count_we = np.zeros(shape3, dtype=np.int64)
    bout_rate = np.full(shape3, np.nan, dtype=np.float32)
    time_in_bin = np.zeros(shape3, dtype=np.float32)
    med_peak = np.full(shape3, np.nan, dtype=np.float32)
    med_path = np.full(shape3, np.nan, dtype=np.float32)
    mean_abs_turn = np.full(shape3, np.nan, dtype=np.float32)
    toward_frac = np.full(shape3, np.nan, dtype=np.float32)
    toward_frac_we = np.full(shape3, np.nan, dtype=np.float32)
    bias_r = np.full(shape3, np.nan, dtype=np.float32)
    bias_r_we = np.full(shape3, np.nan, dtype=np.float32)
    mean_delta = np.full(shape3, np.nan, dtype=np.float32)
    mean_delta_miss = np.full(shape3, np.nan, dtype=np.float32)
    frac_widen = np.full(shape3, np.nan, dtype=np.float32)
    approach_bout_count = np.zeros(shape3, dtype=np.int64)
    circling = np.full(shape3, np.nan, dtype=np.float32)
    circling_we = np.full(shape3, np.nan, dtype=np.float32)
    radial_v = np.full(shape3, np.nan, dtype=np.float32)
    tangential_v = np.full(shape3, np.nan, dtype=np.float32)
    frame_count = np.zeros(shape3, dtype=np.int64)
    frame_count_we = np.zeros(shape3, dtype=np.int64)

    # Per-frame visit labels per reference, then the visit each bout's onset falls in.
    visit_frame = np.full((fish_mm.shape[0], n_refs), -1, dtype=np.int32)
    for r_idx in range(n_refs):
        visit_frame[:, r_idx] = _segment_visits(
            dist[:, r_idx], fish_valid & np.isfinite(dist[:, r_idx]),
            enter_mm=float(visit_enter_mm), exit_mm=float(visit_exit_mm),
        )
    bout_visit_id = np.where(onset_ok[:, None], visit_frame[bout_start, :], -1).astype(np.int32)
    near_visit_count = np.zeros(shape2, dtype=np.int64)

    near_delta_miss = np.full(shape2, np.nan, dtype=np.float32)
    near_frac_widen = np.full(shape2, np.nan, dtype=np.float32)
    near_bias = np.full(shape2, np.nan, dtype=np.float32)
    near_toward = np.full(shape2, np.nan, dtype=np.float32)
    near_circ = np.full(shape2, np.nan, dtype=np.float32)
    near_count = np.zeros(shape2, dtype=np.int64)

    for e_idx, epoch in enumerate(epochs):
        frames = np.arange(epoch.start_frame, epoch.end_frame + 1)
        bmask_epoch = (epoch_index == e_idx) & bout_valid

        for r_idx in range(n_refs):
            # ---- frame-level ----
            f_ok = pair[frames] & np.isfinite(dist[frames, r_idx]) & (speed[frames] > float(moving_speed_threshold_mm_s))
            f_all = fish_valid[frames] & np.isfinite(dist[frames, r_idx])
            fr = frames[f_ok]
            if fr.size:
                # Radial/tangential decomposition is frame-agnostic in magnitude (both the
                # velocity and the object vector are y-down here), so no flip is needed; the
                # circling index uses |v_t| and is sign-free.
                rel_f = ref_mm[fr, r_idx, :] - fish_mm[fr]
                norm = np.maximum(np.hypot(rel_f[:, 0], rel_f[:, 1]), 1e-9)
                u = rel_f / norm[:, None]
                t_hat = np.stack([-u[:, 1], u[:, 0]], axis=1)
                vr = np.sum(v_fish[fr] * u, axis=1)
                vt = np.sum(v_fish[fr] * t_hat, axis=1)
                sp = speed[fr]
                ci = np.divide(np.abs(vt), sp, out=np.full(sp.shape, np.nan), where=sp > 1e-9)
                fb = np.digitize(dist[fr, r_idx], edges) - 1
                fw = wall[fr]
                for b in range(n_bins):
                    sel = fb == b
                    n_sel = int(sel.sum())
                    frame_count[e_idx, r_idx, b] = n_sel
                    if n_sel >= int(min_bin_frames):
                        circling[e_idx, r_idx, b] = float(np.nanmean(ci[sel]))
                        radial_v[e_idx, r_idx, b] = float(np.nanmean(vr[sel]))
                        tangential_v[e_idx, r_idx, b] = float(np.nanmean(np.abs(vt[sel])))
                    sel_we = sel & ~fw
                    n_we = int(sel_we.sum())
                    frame_count_we[e_idx, r_idx, b] = n_we
                    if n_we >= int(min_bin_frames):
                        circling_we[e_idx, r_idx, b] = float(np.nanmean(ci[sel_we]))

            # occupancy time per bin (all valid frames, not just moving) -> bout rate denominator
            fa = frames[f_all]
            if fa.size:
                ab = np.digitize(dist[fa, r_idx], edges) - 1
                for b in range(n_bins):
                    time_in_bin[e_idx, r_idx, b] = float(np.count_nonzero(ab == b)) / safe_fps

            # ---- bout-level ----
            d0 = d_onset[:, r_idx]
            valid_b = bmask_epoch & np.isfinite(d0) & np.isfinite(turn)
            if not np.any(valid_b):
                continue
            bb = np.digitize(d0[valid_b], edges) - 1
            idx_b = np.where(valid_b)[0]
            for b in range(n_bins):
                sel = idx_b[bb == b]
                bout_count[e_idx, r_idx, b] = int(sel.size)
                seconds = float(time_in_bin[e_idx, r_idx, b])
                if seconds > 0:
                    bout_rate[e_idx, r_idx, b] = float(sel.size) / (seconds / 60.0)
                sel_we = sel[~bout_wall[sel]]
                bout_count_we[e_idx, r_idx, b] = int(sel_we.size)
                if sel.size >= int(min_bin_bouts):
                    med_peak[e_idx, r_idx, b] = float(np.nanmedian(bout_peak_speed[sel]))
                    med_path[e_idx, r_idx, b] = float(np.nanmedian(bout_path_length[sel]))
                    mean_abs_turn[e_idx, r_idx, b] = float(np.nanmean(np.abs(turn[sel])))
                    toward_frac[e_idx, r_idx, b] = float(np.nanmean(turn_toward[sel, r_idx]))
                    bias_r[e_idx, r_idx, b] = _pearson_r(b_onset[sel, r_idx], turn[sel])
                    mean_delta[e_idx, r_idx, b] = float(np.nanmean(delta_distance[sel, r_idx]))
                sel_ap = sel[approaching_at_onset[sel, r_idx]]
                approach_bout_count[e_idx, r_idx, b] = int(sel_ap.size)
                if sel_ap.size >= int(min_bin_bouts):
                    dm = delta_predicted_miss[sel_ap, r_idx]
                    dm = dm[np.isfinite(dm)]
                    if dm.size:
                        mean_delta_miss[e_idx, r_idx, b] = float(np.mean(dm))
                        frac_widen[e_idx, r_idx, b] = float(np.mean(dm > 0))
                if sel_we.size >= int(min_bin_bouts):
                    toward_frac_we[e_idx, r_idx, b] = float(np.nanmean(turn_toward[sel_we, r_idx]))
                    bias_r_we[e_idx, r_idx, b] = _pearson_r(b_onset[sel_we, r_idx], turn[sel_we])

            # ---- near-band scalars (pooled, not a mean of bin means) ----
            near_sel = idx_b[d0[idx_b] < float(near_distance_mm)]
            near_count[e_idx, r_idx] = int(near_sel.size)
            visits = bout_visit_id[near_sel, r_idx]
            near_visit_count[e_idx, r_idx] = int(np.unique(visits[visits >= 0]).size)
            if near_sel.size >= int(min_bin_bouts):
                near_bias[e_idx, r_idx] = _pearson_r(b_onset[near_sel, r_idx], turn[near_sel])
                near_toward[e_idx, r_idx] = float(np.nanmean(turn_toward[near_sel, r_idx]))
            near_ap = near_sel[approaching_at_onset[near_sel, r_idx]]
            dm_near = delta_predicted_miss[near_ap, r_idx]
            dm_near = dm_near[np.isfinite(dm_near)]
            if dm_near.size >= int(min_bin_bouts):
                near_delta_miss[e_idx, r_idx] = float(np.mean(dm_near))
                near_frac_widen[e_idx, r_idx] = float(np.mean(dm_near > 0))
            near_bins = centers < float(near_distance_mm)
            w = frame_count[e_idx, r_idx, near_bins].astype(np.float64)
            v = circling[e_idx, r_idx, near_bins].astype(np.float64)
            ok = np.isfinite(v) & (w > 0)
            if np.any(ok) and float(w[ok].sum()) >= float(min_bin_frames):
                near_circ[e_idx, r_idx] = float(np.sum(v[ok] * w[ok]) / np.sum(w[ok]))

    # ---- object-specific excess over the object's own virtual twins ----
    object_refs = [i for i, ref in enumerate(references) if ref.kind == REFERENCE_KIND_OBJECT]
    n_objects = len(object_refs)
    circ_excess = np.full((n_epochs, n_objects), np.nan, dtype=np.float32)
    bias_excess = np.full((n_epochs, n_objects), np.nan, dtype=np.float32)
    steer_excess = np.full((n_epochs, n_objects), np.nan, dtype=np.float32)
    # The dose-response. A single near-band scalar averages over bins where nothing happens and
    # dilutes the signal (on the cohort it halved it). The profile is the primary output: a real
    # object-driven response is localized and decays with distance; an artifact is flat.
    steer_excess_band = np.full((n_epochs, n_objects, n_bins), np.nan, dtype=np.float32)
    circ_excess_band = np.full((n_epochs, n_objects, n_bins), np.nan, dtype=np.float32)
    for o_pos, r_idx in enumerate(object_refs):
        chaser_id = references[r_idx].chaser_index
        twins = [
            i for i, ref in enumerate(references)
            if ref.kind == REFERENCE_KIND_VIRTUAL and ref.parent_chaser_index == chaser_id
        ]
        if not twins:
            warnings.append(f"no_virtual_controls_for_chaser{chaser_id}")
            continue
        for e_idx in range(n_epochs):
            base_c = float(near_circ[e_idx, r_idx])
            twin_c = np.asarray([near_circ[e_idx, t] for t in twins], dtype=np.float64)
            twin_c = twin_c[np.isfinite(twin_c)]
            if math.isfinite(base_c) and twin_c.size:
                circ_excess[e_idx, o_pos] = float(base_c - float(np.mean(twin_c)))
            base_b = float(near_bias[e_idx, r_idx])
            twin_b = np.asarray([near_bias[e_idx, t] for t in twins], dtype=np.float64)
            twin_b = twin_b[np.isfinite(twin_b)]
            if math.isfinite(base_b) and twin_b.size:
                bias_excess[e_idx, o_pos] = float(base_b - float(np.mean(twin_b)))
            base_s = float(near_delta_miss[e_idx, r_idx])
            twin_s = np.asarray([near_delta_miss[e_idx, t] for t in twins], dtype=np.float64)
            twin_s = twin_s[np.isfinite(twin_s)]
            if math.isfinite(base_s) and twin_s.size:
                steer_excess[e_idx, o_pos] = float(base_s - float(np.mean(twin_s)))
            with np.errstate(invalid="ignore"):
                twin_band_s = np.nanmean(np.asarray([mean_delta_miss[e_idx, t, :] for t in twins],
                                                    dtype=np.float64), axis=0)
                twin_band_c = np.nanmean(np.asarray([circling[e_idx, t, :] for t in twins],
                                                    dtype=np.float64), axis=0)
            steer_excess_band[e_idx, o_pos, :] = (
                mean_delta_miss[e_idx, r_idx, :].astype(np.float64) - twin_band_s
            ).astype(np.float32)
            circ_excess_band[e_idx, o_pos, :] = (
                circling[e_idx, r_idx, :].astype(np.float64) - twin_band_c
            ).astype(np.float32)

    for e_idx, epoch in enumerate(epochs):
        for o_pos, r_idx in enumerate(object_refs):
            label = references[r_idx].label
            if int(near_count[e_idx, r_idx]) < int(min_bin_bouts):
                warnings.append(f"few_near_bouts:{epoch.label}:{label}")
                continue
            n_visits = int(near_visit_count[e_idx, r_idx])
            if n_visits < int(min_visits_for_inference):
                warnings.append(
                    f"pseudoreplicated:{epoch.label}:{label}:"
                    f"{int(near_count[e_idx, r_idx])}bouts_from_{n_visits}visits"
                )

    arrays = {
        "bout_id": np.asarray(bout_ids, dtype=np.int64),
        "bout_epoch_index": epoch_index,
        "bout_start_frame": np.asarray(bout_start, dtype=np.int64),
        "bout_end_frame": np.asarray(bout_end, dtype=np.int64),
        "bout_turn_deg": turn.astype(np.float32),
        "bout_peak_speed_mm_s": np.asarray(bout_peak_speed, dtype=np.float32),
        "bout_mean_speed_mm_s": np.asarray(bout_mean_speed, dtype=np.float32),
        "bout_duration_s": np.asarray(bout_duration_s, dtype=np.float32),
        "bout_path_length_mm": np.asarray(bout_path_length, dtype=np.float32),
        "bout_net_displacement_mm": np.asarray(bout_net_displacement, dtype=np.float32),
        "bout_tortuosity": tortuosity.astype(np.float32),
        "bout_wall_at_onset": bout_wall,
        "bout_valid": bout_valid,
        "bout_distance_at_onset_mm": d_onset.astype(np.float32),
        "bout_bearing_at_onset_deg": b_onset.astype(np.float32),
        "bout_delta_distance_mm": delta_distance.astype(np.float32),
        "bout_predicted_miss_onset_mm": predicted_miss_onset.astype(np.float32),
        "bout_predicted_miss_end_mm": predicted_miss_end.astype(np.float32),
        "bout_delta_predicted_miss_mm": delta_predicted_miss.astype(np.float32),
        "bout_approaching_at_onset": approaching_at_onset,
        "bout_turn_toward": turn_toward,
        "bout_count": bout_count,
        "bout_rate_per_min": bout_rate,
        "time_in_bin_s": time_in_bin,
        "median_peak_speed_mm_s": med_peak,
        "median_path_length_mm": med_path,
        "mean_abs_turn_deg": mean_abs_turn,
        "turn_toward_fraction": toward_frac,
        "turn_bias_r": bias_r,
        "mean_delta_distance_mm": mean_delta,
        "mean_delta_predicted_miss_mm": mean_delta_miss,
        "fraction_bouts_widening_miss": frac_widen,
        "approach_bout_count": approach_bout_count,
        "circling_index": circling,
        "radial_velocity_mm_s": radial_v,
        "tangential_speed_mm_s": tangential_v,
        "frame_count": frame_count,
        "bout_count_wall_excluded": bout_count_we,
        "turn_toward_fraction_wall_excluded": toward_frac_we,
        "turn_bias_r_wall_excluded": bias_r_we,
        "circling_index_wall_excluded": circling_we,
        "frame_count_wall_excluded": frame_count_we,
        "near_delta_predicted_miss_mm": near_delta_miss,
        "near_fraction_widening_miss": near_frac_widen,
        "near_turn_bias_r": near_bias,
        "near_turn_toward_fraction": near_toward,
        "near_circling_index": near_circ,
        "near_bout_count": near_count,
        "near_visit_count": near_visit_count,
        "bout_visit_id": bout_visit_id,
        "circling_excess_vs_virtual": circ_excess,
        "turn_bias_excess_vs_virtual": bias_excess,
        "steering_excess_vs_virtual": steer_excess,
        "steering_excess_by_band": steer_excess_band,
        "circling_excess_by_band": circ_excess_band,
    }
    diagnostics = {
        "bearing_convention": (
            "wrap(atan2(-dy, dx) - heading); 0 deg = reference dead ahead, positive = reference to the "
            "fish's left (CCW). The y is negated because arena positions are y-down while heading is "
            "CCW-from-+x in a y-up frame; this reproduces chaser_egocentric_bearing.bearing_deg exactly."
        ),
        "turn_convention": "wrap(heading[bout_end] - heading[bout_start]); same angular sense as bearing",
        "turn_toward_definition": (
            "sign(turn) == sign(bearing_at_onset): the bout rotated the fish toward the reference's "
            "side. Sustained turning toward a reference at close range is the signature of ORBITING "
            "it, not of approaching it -- an arc around a point requires centripetal turning."
        ),
        "circling_index_definition": (
            "mean |tangential velocity| / speed over moving frames; 1.0 = motion purely around the "
            "reference, 0.0 = motion purely toward or away from it"
        ),
        "virtual_control_definition": (
            "each object's own position rotated about the arena centre, so a virtual reference has "
            "identical distance-from-centre and wall proximity at every instant but no object. If a "
            "signature survives around virtual references it is wall-following, not the object."
        ),
        "excess_definition": "object value minus the mean over that object's virtual twins; this is the object-specific quantity",
        "predicted_miss_definition": (
            "b = distance * |sin(bearing)|: the miss distance the fish would achieve if it kept going "
            "straight from here. delta_predicted_miss is its change across a bout -- positive means the "
            "bout steered the fish to pass WIDER, i.e. active avoidance steering. Restricted to bouts "
            "beginning with the object ahead (|bearing| < 90 deg)."
        ),
        "why_per_bout": (
            "the trajectory-level version of this question is unusable: b <= r by construction, and "
            "conditioning on approaches that got close forces b to converge on the CPA (at the closest "
            "point b == r identically, and the bearing there is always 90 deg). Per bout there is no "
            "such conditioning."
        ),
        "wall_confound_note": (
            "the fish wall-follows and the objects sit near the wall; a wall-following arc sweeps "
            "around a wall-adjacent object for free. Always read *_excess_vs_virtual and the "
            "wall_excluded twins, never the raw object value alone."
        ),
        "min_bin_bouts": int(min_bin_bouts),
        "min_bin_frames": int(min_bin_frames),
        "visit_definition": (
            f"contiguous excursion within {float(visit_enter_mm):g} mm of the reference, closing when the "
            f"fish leaves past {float(visit_exit_mm):g} mm"
        ),
        "pseudoreplication_note": (
            "near_bout_count is NOT the effective sample size. Bouts inside one visit are one approach "
            "subsampled, not independent draws. Use near_visit_count as the effective n and resample "
            "bout_visit_id (a cluster bootstrap) for any inference; a raw p-value on the bout count "
            "will be wildly anticonservative."
        ),
        "qc_warning_count": len(warnings),
    }
    return arrays, diagnostics, tuple(dict.fromkeys(warnings))


def build_chaser_bout_response_result(
    zarr_path: Path,
    *,
    chaser_distance_run: str = "latest",
    swim_bout_run: str = "latest",
    component_name: str = DEFAULT_COMPONENT_NAME,
    distance_bin_edges_mm: Sequence[float] = DEFAULT_DISTANCE_BIN_EDGES_MM,
    near_distance_mm: float = DEFAULT_NEAR_DISTANCE_MM,
    virtual_rotations_deg: Sequence[float] = DEFAULT_VIRTUAL_ROTATIONS_DEG,
    min_virtual_separation_mm: float = DEFAULT_MIN_VIRTUAL_SEPARATION_MM,
    perimeter_band_mm: float = DEFAULT_PERIMETER_BAND_MM,
    moving_speed_threshold_mm_s: float = DEFAULT_MOVING_SPEED_THRESHOLD_MM_S,
    min_bin_bouts: int = DEFAULT_MIN_BIN_BOUTS,
    min_bin_frames: int = DEFAULT_MIN_BIN_FRAMES,
    visit_enter_mm: float = DEFAULT_VISIT_ENTER_MM,
    visit_exit_mm: float = DEFAULT_VISIT_EXIT_MM,
    min_visits_for_inference: int = DEFAULT_MIN_VISITS_FOR_INFERENCE,
    settle_trim_s: float | None = None,
    motion_spread_threshold_mm: float = DEFAULT_MOTION_SPREAD_THRESHOLD_MM,
) -> ChaserBoutResponseResult:
    root = _open_root(zarr_path, mode="r")
    distance, run_name, run_path = _resolve_chaser_distance_run(
        root,
        chaser_distance_run,
    )
    distance.require_derived_surface_authority("egocentric_bearing")

    if distance.coordinate_space_id != "arena_relative_canvas_px":
        raise ValueError(
            "chaser_bout_response requires typed "
            "space_id='arena_relative_canvas_px'."
        )
    pixels_per_mm = float(distance.pixels_per_mm_projector)

    fish_xy = np.asarray(distance.fish_centroid_arena_xy, dtype=np.float64)
    chaser_xy = np.asarray(distance.chaser_arena_xy, dtype=np.float64)
    fish_valid = np.asarray(distance.fish_valid, dtype=bool)
    chaser_valid = np.asarray(distance.chaser_valid, dtype=bool)
    chaser_indices = np.asarray(distance.chaser_index, dtype=np.int64).reshape(-1)
    chaser_labels: tuple[str, ...] = ()
    total_frames = int(fish_xy.shape[0])
    if total_frames != distance.total_frames:
        raise ValueError("Typed chaser-distance frame extent is inconsistent.")

    geometry = _resolve_arena_geometry(
        root,
        distance,
        pixels_per_mm=float(pixels_per_mm),
    )
    if geometry.shape != "circle" or geometry.center_x_px is None or geometry.center_y_px is None:
        raise ValueError(
            "chaser_bout_response requires a circular arena geometry: the virtual controls are "
            "rotations about the arena centre, which is undefined otherwise."
        )
    fps = float(distance.fps)

    epochs = _read_epochs(distance, total_frames=total_frames)
    resolved_settle = (
        _protocol_position_transition_s(root, distance)
        if settle_trim_s is None
        else max(0.0, float(settle_trim_s))
    )
    epochs = _apply_settle_trim(
        epochs,
        chaser_xy=chaser_xy,
        chaser_valid=chaser_valid,
        fps=float(fps),
        pixels_per_mm=float(pixels_per_mm),
        settle_trim_s=float(resolved_settle),
        motion_spread_threshold_mm=float(motion_spread_threshold_mm),
    )

    heading, heading_valid, ego_name = _resolve_heading(
        root[run_path],
        total_frames=total_frames,
    )

    bout_group, bout_name, bout_path = _resolve_swim_bout_run(root, swim_bout_run)
    table = bout_group.get("tables/bouts") if "tables" in bout_group else None
    if table is None:
        raise ValueError(f"Swim-bout run {bout_name} lacks tables/bouts.")
    bout_start = np.asarray(table["start_frame"][:], dtype=np.int64).reshape(-1)
    bout_end = np.asarray(table["end_frame"][:], dtype=np.int64).reshape(-1)
    # A multi-level bout table (detect_bouts_multi_level) concatenates bouts from EVERY speed
    # level -- raw, filtered, smoothed, averaged, exponential -- in one table, tagged by
    # signal_id. Ingesting all of them counts each physical bout up to five times (and mixes
    # jittery raw peaks with smoothed ones). Select the run's default level, which it documents
    # as the one downstream consumers should use. A single-level table has no signal_id column
    # and is kept whole.
    source_level_name = str(bout_group.attrs.get("default_level") or "")
    signal_id = (
        np.asarray(table["signal_id"][:], dtype=np.int64).reshape(-1)
        if "signal_id" in table
        else None
    )
    keep, source_signal_id, bout_level_note = _select_bout_row_mask(
        bout_start,
        bout_end,
        total_frames=total_frames,
        signal_id=signal_id,
        default_signal_id=bout_group.attrs.get("default_signal_id"),
    )
    if not np.all(keep):
        bout_start, bout_end = bout_start[keep], bout_end[keep]
    take = lambda name, default=np.nan: (  # noqa: E731
        np.asarray(table[name][:], dtype=np.float64).reshape(-1)[keep]
        if name in table
        else np.full(int(keep.sum()), default, dtype=np.float64)
    )
    bout_ids = (
        np.asarray(table["bout_id"][:], dtype=np.int64).reshape(-1)[keep]
        if "bout_id" in table
        else np.arange(int(keep.sum()), dtype=np.int64)
    )

    references, ref_xy, ref_notes = _build_references(
        chaser_indices=chaser_indices,
        chaser_labels=chaser_labels,
        chaser_xy_px=chaser_xy,
        center_xy_px=(float(geometry.center_x_px), float(geometry.center_y_px)),
        rotations_deg=list(virtual_rotations_deg),
        min_separation_mm=float(min_virtual_separation_mm),
        pixels_per_mm=float(pixels_per_mm),
    )

    edges = _normalize_float_array(distance_bin_edges_mm, name="distance_bin_edges_mm")
    if edges.shape[0] < 2:
        raise ValueError("distance_bin_edges_mm must contain at least two increasing values.")
    centers = ((edges[:-1] + edges[1:]) / 2.0).astype(np.float32)

    arrays, diagnostics, qc = _compute(
        epochs=epochs,
        references=references,
        ref_xy_px=ref_xy,
        fish_xy_px=fish_xy,
        fish_valid=fish_valid,
        heading_deg=heading,
        heading_valid=heading_valid,
        bout_start=bout_start,
        bout_end=bout_end,
        bout_peak_speed=take("peak_physical_speed_mm_s"),
        bout_mean_speed=take("mean_speed_mm_s"),
        bout_duration_s=take("duration_s"),
        bout_path_length=take("path_length_mm"),
        bout_net_displacement=take("net_displacement_mm"),
        bout_ids=bout_ids,
        fps=float(fps),
        pixels_per_mm=float(pixels_per_mm),
        geometry=geometry,
        perimeter_band_mm=float(perimeter_band_mm),
        distance_bin_edges_mm=edges,
        near_distance_mm=float(near_distance_mm),
        moving_speed_threshold_mm_s=float(moving_speed_threshold_mm_s),
        min_bin_bouts=int(min_bin_bouts),
        min_bin_frames=int(min_bin_frames),
        visit_enter_mm=float(visit_enter_mm),
        visit_exit_mm=float(visit_exit_mm),
        min_visits_for_inference=int(min_visits_for_inference),
    )
    qc = tuple(dict.fromkeys(list(qc) + ref_notes))
    diagnostics["virtual_reference_notes"] = ref_notes
    diagnostics["reference_labels"] = [ref.label for ref in references]
    diagnostics["source_swim_bout_signal_id"] = int(source_signal_id)
    diagnostics["source_swim_bout_level"] = source_level_name
    diagnostics["bout_level_selection"] = bout_level_note
    diagnostics["bouts_ingested"] = int(keep.sum())
    qc = tuple(qc) + (bout_level_note,)

    recording_id = distance.recording_id
    object_refs = [ref for ref in references if ref.kind == REFERENCE_KIND_OBJECT]
    summary: dict[str, Any] = {"recording_id": recording_id, "n_references": len(references)}
    for e_idx, epoch in enumerate(epochs):
        for o_pos, ref in enumerate(object_refs):
            stem = f"{epoch.label}_{ref.label}"
            for key, name in (
                ("circling_excess_vs_virtual", "circling_excess"),
                ("turn_bias_excess_vs_virtual", "turn_bias_excess"),
                ("steering_excess_vs_virtual", "steering_excess"),
            ):
                value = float(arrays[key][e_idx, o_pos])
                summary[f"{name}_{stem}"] = value if math.isfinite(value) else None
            r_idx = references.index(ref)
            for key, name in (
                ("near_circling_index", "near_circling"),
                ("near_turn_bias_r", "near_turn_bias_r"),
                ("near_turn_toward_fraction", "near_turn_toward"),
                ("near_delta_predicted_miss_mm", "near_delta_predicted_miss"),
                ("near_fraction_widening_miss", "near_frac_widening_miss"),
            ):
                value = float(arrays[key][e_idx, r_idx])
                summary[f"{name}_{stem}"] = value if math.isfinite(value) else None
            summary[f"near_bout_count_{stem}"] = int(arrays["near_bout_count"][e_idx, r_idx])
            summary[f"near_visit_count_{stem}"] = int(arrays["near_visit_count"][e_idx, r_idx])

    return ChaserBoutResponseResult(
        zarr_path=str(zarr_path),
        recording_id=recording_id,
        component_name=str(component_name),
        chaser_distance_run_name=run_name,
        chaser_distance_run_path=run_path,
        source_swim_bout_run=bout_name,
        source_swim_bout_path=bout_path,
        source_egocentric_component=ego_name,
        fps=float(fps),
        total_frames=total_frames,
        pixels_per_mm_projector=float(pixels_per_mm),
        geometry_status=geometry.status,
        arena_center_x_px=geometry.center_x_px,
        arena_center_y_px=geometry.center_y_px,
        arena_radius_px=geometry.radius_px,
        perimeter_band_mm=float(perimeter_band_mm),
        near_distance_mm=float(near_distance_mm),
        moving_speed_threshold_mm_s=float(moving_speed_threshold_mm_s),
        min_bin_bouts=int(min_bin_bouts),
        min_bin_frames=int(min_bin_frames),
        visit_enter_mm=float(visit_enter_mm),
        visit_exit_mm=float(visit_exit_mm),
        min_visits_for_inference=int(min_visits_for_inference),
        settle_trim_s=float(resolved_settle),
        distance_bin_edges_mm=edges.astype(np.float32),
        distance_bin_centers_mm=centers,
        epochs=epochs,
        references=references,
        status="computed",
        qc_warnings=qc,
        summary=summary,
        diagnostics=diagnostics,
        **{k: v for k, v in arrays.items()},
    )


def _object_indices(result: ChaserBoutResponseResult) -> list[int]:
    return [i for i, ref in enumerate(result.references) if ref.kind == REFERENCE_KIND_OBJECT]


def _virtual_indices(result: ChaserBoutResponseResult, chaser_index: int) -> list[int]:
    return [
        i for i, ref in enumerate(result.references)
        if ref.kind == REFERENCE_KIND_VIRTUAL and ref.parent_chaser_index == chaser_index
    ]


def _curve_panels(result: ChaserBoutResponseResult, array: np.ndarray, ylabel: str, title: str, *, hline: float | None) -> bytes:
    n = len(result.epochs)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 4.6), sharey=True, constrained_layout=True)
    axes_list = list(np.atleast_1d(axes).ravel())
    for e_idx, (ax, epoch) in enumerate(zip(axes_list, result.epochs)):
        for r_idx in _object_indices(result):
            ref = result.references[r_idx]
            curve = array[e_idx, r_idx, :]
            if np.isfinite(curve).any():
                ax.plot(result.distance_bin_centers_mm, curve, marker="o", markersize=4.0,
                        linewidth=1.8, label=f"{ref.label} (object)", zorder=3)
            twins = _virtual_indices(result, ref.chaser_index)
            stack = np.asarray([array[e_idx, t, :] for t in twins], dtype=np.float64)
            if stack.size and np.isfinite(stack).any():
                mean = np.nanmean(stack, axis=0)
                lo, hi = np.nanmin(stack, axis=0), np.nanmax(stack, axis=0)
                ax.fill_between(result.distance_bin_centers_mm, lo, hi, alpha=0.18, lw=0, zorder=1)
                ax.plot(result.distance_bin_centers_mm, mean, linestyle="--", linewidth=1.2,
                        alpha=0.85, label=f"{ref.label}: virtual controls (wall null)", zorder=2)
        if hline is not None:
            ax.axhline(hline, color="#334155", linestyle=":", linewidth=0.9)
        ax.axvline(result.near_distance_mm, color="#94a3b8", linestyle=":", linewidth=0.9)
        ax.set_title(epoch.label.replace("_", " "))
        ax.set_xlabel("distance to reference at bout onset (mm)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=6.5)
    axes_list[0].set_ylabel(ylabel)
    fig.suptitle(title, fontsize=11)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return buf.getvalue()


def render_circling_png(result: ChaserBoutResponseResult) -> bytes:
    return _curve_panels(
        result, result.circling_index,
        ylabel="circling index  |v_tangential| / speed\n[1 = pure orbit, 0 = pure approach/retreat]",
        title=f"Circling around the object vs around a rotated virtual twin: {result.recording_id}",
        hline=None,
    )


def render_turn_bias_png(result: ChaserBoutResponseResult) -> bytes:
    return _curve_panels(
        result, result.turn_bias_r,
        ylabel="turn bias  r(bearing at onset, turn)\n[+ = bouts turn toward the object = orbiting]",
        title=f"Bout turn bias vs the wall null: {result.recording_id}",
        hline=0.0,
    )


def render_bout_kinematics_png(result: ChaserBoutResponseResult) -> bytes:
    n = len(result.epochs)
    fig, axes = plt.subplots(2, n, figsize=(5.0 * n, 7.6), sharex=True, constrained_layout=True, squeeze=False)
    for e_idx, epoch in enumerate(result.epochs):
        top, bottom = axes[0][e_idx], axes[1][e_idx]
        for r_idx in _object_indices(result):
            ref = result.references[r_idx]
            top.plot(result.distance_bin_centers_mm, result.median_peak_speed_mm_s[e_idx, r_idx, :],
                     marker="o", markersize=4.0, linewidth=1.6, label=ref.label)
            bottom.plot(result.distance_bin_centers_mm, result.bout_rate_per_min[e_idx, r_idx, :],
                        marker="o", markersize=4.0, linewidth=1.6, label=ref.label)
        top.set_title(epoch.label.replace("_", " "))
        for ax in (top, bottom):
            ax.axvline(result.near_distance_mm, color="#94a3b8", linestyle=":", linewidth=0.9)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7)
        bottom.set_xlabel("distance to object at bout onset (mm)")
    axes[0][0].set_ylabel("median bout peak speed (mm/s)")
    axes[1][0].set_ylabel("bout rate (per min in that distance band)")
    fig.suptitle(f"Bout kinematics vs distance to object: {result.recording_id}", fontsize=11)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return buf.getvalue()


def _source_refs(result: ChaserBoutResponseResult) -> dict[str, Any]:
    return {
        "source_chaser_distance_run": result.chaser_distance_run_name,
        "source_chaser_distance_path": result.chaser_distance_run_path,
        "source_swim_bout_run": result.source_swim_bout_run,
        "source_swim_bout_path": result.source_swim_bout_path,
        "source_egocentric_bearing_component": result.source_egocentric_component,
    }


def _parameters(result: ChaserBoutResponseResult) -> dict[str, Any]:
    return {
        "distance_bin_edges_mm": result.distance_bin_edges_mm,
        "near_distance_mm": result.near_distance_mm,
        "perimeter_band_mm": result.perimeter_band_mm,
        "moving_speed_threshold_mm_s": result.moving_speed_threshold_mm_s,
        "min_bin_bouts": result.min_bin_bouts,
        "min_bin_frames": result.min_bin_frames,
        "visit_enter_mm": result.visit_enter_mm,
        "visit_exit_mm": result.visit_exit_mm,
        "settle_trim_s": result.settle_trim_s,
        "virtual_rotations_deg": [ref.rotation_deg for ref in result.references if ref.kind == REFERENCE_KIND_VIRTUAL],
        "virtual_control_definition": result.diagnostics.get("virtual_control_definition"),
    }


def write_chaser_bout_response_component(
    zarr_path: Path,
    result: ChaserBoutResponseResult,
    *,
    overwrite: bool = False,
    write_png: bool = True,
    write_interactive_spec: bool = True,
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
    if result.component_name in parent:
        if not overwrite:
            raise ValueError(f"Chaser bout-response component already exists: {result.component_name}")
        del parent[result.component_name]
    component = parent.create_group(result.component_name)
    component_path = f"{result.chaser_distance_run_path}/{COMPONENT_PARENT_NAME}/{result.component_name}"

    config = component.require_group("config")
    _write_array(config, "distance_bin_edges_mm", result.distance_bin_edges_mm)
    _write_array(config, "distance_bin_centers_mm", result.distance_bin_centers_mm)
    config.attrs.update(
        {
            "near_distance_mm": float(result.near_distance_mm),
            "visit_enter_mm": float(result.visit_enter_mm),
            "visit_exit_mm": float(result.visit_exit_mm),
            "min_visits_for_inference": int(result.min_visits_for_inference),
            "perimeter_band_mm": float(result.perimeter_band_mm),
            "moving_speed_threshold_mm_s": float(result.moving_speed_threshold_mm_s),
            "min_bin_bouts": int(result.min_bin_bouts),
            "min_bin_frames": int(result.min_bin_frames),
            "settle_trim_s": float(result.settle_trim_s),
        }
    )

    epochs = component.require_group("epochs")
    _write_array(epochs, "label_bytes", _bytes_array([e.label for e in result.epochs], width=96))
    _write_array(epochs, "start_frame", np.asarray([e.start_frame for e in result.epochs], dtype=np.int64))
    _write_array(epochs, "end_frame", np.asarray([e.end_frame for e in result.epochs], dtype=np.int64))
    epochs.attrs.update({"row_axis": "epoch"})

    refs = component.require_group("references")
    _write_array(refs, "label_bytes", _bytes_array([r.label for r in result.references], width=64))
    _write_array(refs, "kind_bytes", _bytes_array([r.kind for r in result.references], width=16))
    _write_array(refs, "chaser_index", np.asarray([r.chaser_index for r in result.references], dtype=np.int16))
    _write_array(refs, "parent_chaser_index", np.asarray([r.parent_chaser_index for r in result.references], dtype=np.int16))
    _write_array(refs, "rotation_deg", np.asarray([r.rotation_deg for r in result.references], dtype=np.float32))
    refs.attrs.update(
        {
            "row_axis": "reference",
            "kinds": {"object": "the real stimulus object", "virtual": "the object's position rotated about the arena centre -- the wall null", "dish_center": "fixed arena centre"},
            "virtual_control_definition": result.diagnostics.get("virtual_control_definition"),
        }
    )

    bouts = component.require_group("bouts")
    for name in (
        "bout_id", "bout_epoch_index", "bout_start_frame", "bout_end_frame", "bout_turn_deg",
        "bout_peak_speed_mm_s", "bout_mean_speed_mm_s", "bout_duration_s", "bout_path_length_mm",
        "bout_net_displacement_mm", "bout_tortuosity", "bout_wall_at_onset", "bout_valid",
    ):
        _write_array(bouts, name.removeprefix("bout_") if name != "bout_id" else name, getattr(result, name))
    bouts.attrs.update({"row_axis": "bout", "turn_convention": result.diagnostics.get("turn_convention")})

    per_ref = component.require_group("bouts_per_reference")
    for name in ("bout_distance_at_onset_mm", "bout_bearing_at_onset_deg", "bout_delta_distance_mm",
                 "bout_turn_toward", "bout_visit_id", "bout_predicted_miss_onset_mm",
                 "bout_predicted_miss_end_mm", "bout_delta_predicted_miss_mm", "bout_approaching_at_onset"):
        _write_array(per_ref, name.removeprefix("bout_"), getattr(result, name))
    per_ref.attrs.update(
        {
            "axis_order": ["bout", "reference"],
            "bearing_convention": result.diagnostics.get("bearing_convention"),
            "turn_toward_definition": result.diagnostics.get("turn_toward_definition"),
            "visit_definition": result.diagnostics.get("visit_definition"),
            "pseudoreplication_note": result.diagnostics.get("pseudoreplication_note"),
        }
    )

    binned = component.require_group("binned")
    for name in (
        "bout_count", "bout_rate_per_min", "time_in_bin_s", "median_peak_speed_mm_s",
        "median_path_length_mm", "mean_abs_turn_deg", "turn_toward_fraction", "turn_bias_r",
        "mean_delta_distance_mm", "mean_delta_predicted_miss_mm", "fraction_bouts_widening_miss",
        "approach_bout_count", "circling_index", "radial_velocity_mm_s", "tangential_speed_mm_s",
        "frame_count", "bout_count_wall_excluded", "turn_toward_fraction_wall_excluded",
        "turn_bias_r_wall_excluded", "circling_index_wall_excluded", "frame_count_wall_excluded",
    ):
        _write_array(binned, name, getattr(result, name))
    binned.attrs.update(
        {
            "axis_order": ["epoch", "reference", "distance_bin"],
            "circling_index_definition": result.diagnostics.get("circling_index_definition"),
            "wall_confound_note": result.diagnostics.get("wall_confound_note"),
        }
    )

    controls = component.require_group("object_vs_virtual")
    for name in ("near_turn_bias_r", "near_turn_toward_fraction", "near_circling_index", "near_bout_count",
                 "near_visit_count", "near_delta_predicted_miss_mm", "near_fraction_widening_miss"):
        _write_array(controls, name, getattr(result, name))
    for name in ("circling_excess_vs_virtual", "turn_bias_excess_vs_virtual", "steering_excess_vs_virtual",
                 "steering_excess_by_band", "circling_excess_by_band"):
        _write_array(controls, name, getattr(result, name))
    controls.attrs.update(
        {
            "near_axis_order": ["epoch", "reference"],
            "excess_axis_order": ["epoch", "object"],
            "excess_by_band_axis_order": ["epoch", "object", "distance_bin"],
            "dose_response_note": (
                "steering_excess_by_band is the PRIMARY read. A real object-driven response is "
                "localized and decays with distance; an artifact is flat. On the GoodCopBadCop "
                "cohort the effect peaks in an 8-16 mm shell (+0.68 mm post, p=0.0004) and is gone "
                "by 35 mm. The single near-band scalar averages in bins where nothing happens and "
                "halved the signal."
            ),
            "excess_definition": result.diagnostics.get("excess_definition"),
            "predicted_miss_definition": result.diagnostics.get("predicted_miss_definition"),
            "why_per_bout": result.diagnostics.get("why_per_bout"),
            "wall_confound_note": result.diagnostics.get("wall_confound_note"),
            "pseudoreplication_note": result.diagnostics.get("pseudoreplication_note"),
            "effective_sample_size": "near_visit_count, NOT near_bout_count",
        }
    )

    summary_group = component.require_group("summary")
    for key, value in result.summary.items():
        if isinstance(value, str) or value is None:
            _write_array(summary_group, f"{key}_bytes", _bytes_array(["" if value is None else str(value)], width=128))
        elif isinstance(value, bool):
            _write_array(summary_group, key, np.asarray([int(value)], dtype=np.int8))
        elif isinstance(value, int):
            _write_array(summary_group, key, np.asarray([value], dtype=np.int64))
        else:
            _write_array(summary_group, key, np.asarray([float(value)], dtype=np.float32))
    _write_array(summary_group, "diagnostics_json_bytes",
                 _bytes_array([json.dumps(json_attr_safe(result.diagnostics), sort_keys=True)], width=8192))
    _write_array(summary_group, "qc_warnings_json_bytes",
                 _bytes_array([json.dumps(list(result.qc_warnings), sort_keys=True)], width=4096))
    summary_group.attrs.update({"row_axis": "fish_recording", "summary": json_attr_safe(result.summary)})

    git = get_git_info(Path(__file__).resolve().parents[3])
    source_refs = _source_refs(result)
    parameters = _parameters(result)
    component.attrs.update(json_attr_safe({
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "method_version": METHOD_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "component_name": result.component_name,
        "recording_id": result.recording_id,
        "row_axis": "fish_recording",
        "status": result.status,
        "source_refs": source_refs,
        "parameters": parameters,
        "summary": result.summary,
        "qc_warnings": list(result.qc_warnings),
        "diagnostics": result.diagnostics,
        "geometry_status": result.geometry_status,
        "pixels_per_mm_projector": result.pixels_per_mm_projector,
        "git_commit": git.get("commit_hash"),
        "git_branch": git.get("branch"),
        "git_dirty": git.get("is_dirty"),
        "provenance": {
            "stage": "chaser_bout_response",
            "created_by": "fisheye.analysis.chaser_bout_response",
            "inputs": source_refs,
            "parameters": parameters,
        },
    }))
    write_run_lineage_attrs(
        component,
        build_run_lineage_payload(
            run_family=f"{result.chaser_distance_run_path}/{COMPONENT_PARENT_NAME}",
            analysis_schema={"schema_id": SCHEMA_ID, "schema_version": SCHEMA_VERSION, "row_axis": "fish_recording"},
            method=METHOD,
            method_version=METHOD_VERSION,
            source_refs=source_refs,
            parameters=parameters,
            code={"git_commit": git.get("commit_hash"), "git_dirty": git.get("is_dirty")},
        ),
        fingerprint_status="best_effort",
        overwrite=True,
    )
    parent.attrs["latest"] = result.component_name
    parent.attrs["latest_complete"] = result.component_name

    if write_png:
        for artifact_name, renderer, description in (
            (CIRCLING_PNG_ARTIFACT_NAME, render_circling_png,
             "Circling index around the object vs its rotated virtual twins (the wall null)."),
            (TURN_BIAS_PNG_ARTIFACT_NAME, render_turn_bias_png,
             "Bout turn bias toward the object vs the wall null."),
            (BOUT_KINEMATICS_PNG_ARTIFACT_NAME, render_bout_kinematics_png,
             "Bout peak speed and bout rate vs distance to the object."),
        ):
            write_png_visualization_artifact(
                component, artifact_name, renderer(result),
                description=description,
                created_by="fisheye.analysis.chaser_bout_response",
                role="analysis_summary",
                source_paths={**source_refs, "component": component_path, "binned": f"{component_path}/binned"},
                source_runs={"chaser_distance_run": result.chaser_distance_run_name,
                             "swim_bout_run": result.source_swim_bout_run},
                parameters=parameters,
                extra_attrs={"chaser_bout_response_schema_id": SCHEMA_ID, "component_path": component_path,
                             "canonical_artifact": True},
                overwrite=True,
            )

    if write_interactive_spec:
        spec = {
            "schema_id": INTERACTIVE_SPEC_SCHEMA_ID,
            "schema_version": 1,
            "renderer": INTERACTIVE_RENDERER,
            "recording_id": result.recording_id,
            "component_name": result.component_name,
            "component_path": component_path,
            "source_paths": {
                "component": component_path,
                "binned": f"{component_path}/binned",
                "references": f"{component_path}/references",
                "object_vs_virtual": f"{component_path}/object_vs_virtual",
                "bouts": f"{component_path}/bouts",
                "bouts_per_reference": f"{component_path}/bouts_per_reference",
                "summary": f"{component_path}/summary",
            },
            "summary": result.summary,
            "parameters": parameters,
            "qc_warnings": list(result.qc_warnings),
        }
        write_interactive_plot_spec_artifact(
            component, INTERACTIVE_ARTIFACT_NAME, spec,
            description="Chaser bout-response interactive plot spec.",
            created_by="fisheye.analysis.chaser_bout_response",
            renderer=INTERACTIVE_RENDERER,
            artifact_signature=None,
            snapshot_artifact=CIRCLING_PNG_ARTIFACT_NAME,
            source_paths=spec["source_paths"],
            source_runs={"chaser_distance_run": result.chaser_distance_run_name},
            parameters=parameters,
            extra_attrs={"plot_schema_id": INTERACTIVE_SPEC_SCHEMA_ID, "component_path": component_path,
                         "summary": json_attr_safe(result.summary), "canonical_artifact": True},
            overwrite=True,
        )
    return component_path


def _parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bout-level object-relative kinematics with wall and virtual-object controls.")
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument("--chaser-distance-run", default="latest")
    parser.add_argument("--swim-bout-run", default="latest")
    parser.add_argument("--component-name", default=DEFAULT_COMPONENT_NAME)
    parser.add_argument("--distance-bin-edges-mm", default=",".join(f"{v:g}" for v in DEFAULT_DISTANCE_BIN_EDGES_MM))
    parser.add_argument("--near-distance-mm", type=float, default=DEFAULT_NEAR_DISTANCE_MM)
    parser.add_argument("--virtual-rotations-deg", default=",".join(f"{v:g}" for v in DEFAULT_VIRTUAL_ROTATIONS_DEG))
    parser.add_argument("--min-virtual-separation-mm", type=float, default=DEFAULT_MIN_VIRTUAL_SEPARATION_MM)
    parser.add_argument("--perimeter-band-mm", type=float, default=DEFAULT_PERIMETER_BAND_MM)
    parser.add_argument("--moving-speed-threshold-mm-s", type=float, default=DEFAULT_MOVING_SPEED_THRESHOLD_MM_S)
    parser.add_argument("--min-bin-bouts", type=int, default=DEFAULT_MIN_BIN_BOUTS)
    parser.add_argument("--min-bin-frames", type=int, default=DEFAULT_MIN_BIN_FRAMES)
    parser.add_argument("--visit-enter-mm", type=float, default=DEFAULT_VISIT_ENTER_MM)
    parser.add_argument("--visit-exit-mm", type=float, default=DEFAULT_VISIT_EXIT_MM)
    parser.add_argument("--min-visits-for-inference", type=int, default=DEFAULT_MIN_VISITS_FOR_INFERENCE)
    parser.add_argument("--settle-trim-s", type=float, default=None)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-png", action="store_true")
    parser.add_argument("--no-interactive-spec", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = build_chaser_bout_response_result(
        Path(args.zarr_path),
        chaser_distance_run=str(args.chaser_distance_run),
        swim_bout_run=str(args.swim_bout_run),
        component_name=str(args.component_name),
        distance_bin_edges_mm=_parse_float_list(str(args.distance_bin_edges_mm)),
        near_distance_mm=float(args.near_distance_mm),
        virtual_rotations_deg=_parse_float_list(str(args.virtual_rotations_deg)),
        min_virtual_separation_mm=float(args.min_virtual_separation_mm),
        perimeter_band_mm=float(args.perimeter_band_mm),
        moving_speed_threshold_mm_s=float(args.moving_speed_threshold_mm_s),
        min_bin_bouts=int(args.min_bin_bouts),
        min_bin_frames=int(args.min_bin_frames),
        visit_enter_mm=float(args.visit_enter_mm),
        visit_exit_mm=float(args.visit_exit_mm),
        min_visits_for_inference=int(args.min_visits_for_inference),
        settle_trim_s=None if args.settle_trim_s is None else float(args.settle_trim_s),
    )
    applied = None
    if args.apply:
        applied = write_chaser_bout_response_component(
            Path(args.zarr_path), result,
            overwrite=bool(args.overwrite),
            write_png=not bool(args.no_png),
            write_interactive_spec=not bool(args.no_interactive_spec),
        )
    if args.json:
        print(json.dumps(json_attr_safe({
            "schema_id": SCHEMA_ID, "recording_id": result.recording_id,
            "applied_path": applied, "qc_warnings": list(result.qc_warnings), "summary": result.summary,
        }), indent=2, sort_keys=True))
    else:
        print(f"recording_id\t{result.recording_id}")
        print(f"swim_bout_run\t{result.source_swim_bout_run}")
        print(f"references\t{len(result.references)} "
              f"({sum(1 for r in result.references if r.kind == REFERENCE_KIND_VIRTUAL)} virtual controls)")
        objects = _object_indices(result)
        for e_idx, epoch in enumerate(result.epochs):
            for o_pos, r_idx in enumerate(objects):
                ref = result.references[r_idx]
                print(
                    f"object\t{epoch.label}\t{ref.label}\t"
                    f"n_near_bouts={int(result.near_bout_count[e_idx, r_idx])}\t"
                    f"n_visits={int(result.near_visit_count[e_idx, r_idx])}\t"
                    f"circling={float(result.near_circling_index[e_idx, r_idx]):.3f}\t"
                    f"turn_bias_r={float(result.near_turn_bias_r[e_idx, r_idx]):+.3f}\t"
                    f"steer_dmiss={float(result.near_delta_predicted_miss_mm[e_idx, r_idx]):+.2f}mm\t"
                    f"| vs virtual: circling_excess={float(result.circling_excess_vs_virtual[e_idx, o_pos]):+.3f}\t"
                    f"turn_bias_excess={float(result.turn_bias_excess_vs_virtual[e_idx, o_pos]):+.3f}\t"
                    f"steering_excess={float(result.steering_excess_vs_virtual[e_idx, o_pos]):+.2f}mm"
                )
        for w in result.qc_warnings:
            print(f"qc_warning\t{w}")
        if applied:
            print(f"applied_path\t{applied}")
        else:
            print("dry_run\ttrue\npass --apply to write the component")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
