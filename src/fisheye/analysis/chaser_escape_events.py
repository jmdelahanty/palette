"""Escape events: rate, proximity trigger, and whether the chaser reels the fish back in.

An escape is a *rare, high-amplitude event*, not a shift in the bout distribution. On a
600 s epoch a fish makes ~10,000 bouts and perhaps 15 of them are escapes. That ratio is
what makes escapes easy to miss:

    Clustering the bout table (k-means, K=4, 322k bouts across the cohort) reported the
    escape-like cluster rising from 0.167 to 0.198 during the chase -- p=0.057, a null.
    Escapes are 1.45% of bouts. K-means minimizes within-cluster variance and will never
    spend a centroid on 1.5% of the mass, so the escapes were absorbed into a generic
    high-angle-turn cluster whose *rate* barely moves. A 7.8x effect diluted to nothing.

    Thresholding peak speed found it immediately.

So this module does not cluster. It picks the fast bouts out of the existing
`chaser_bout_response` bout table, and asks three questions about them:

    1. RATE      -- how often, per minute of *validly tracked* time, in each epoch?
    2. TRIGGER   -- how far away was the object when the escape fired, versus an
                    ordinary bout?
    3. PURSUIT   -- aligned on escape onset, what happens to the distance? Does the fish
                    gain ground, and does the chaser take it back?

THE CONTROL THAT MAKES (3) MEAN ANYTHING
----------------------------------------
A fast bout is not an escape. To show the escape is *directed away from the chaser* you
need the same fish making the same bout against the same object when that object is NOT
chasing -- which is exactly the post epoch, where the object sits still. Every trace here
is therefore computed per epoch: the chase epoch and the static epochs are the contrast,
and they are within-fish and within-object.

The virtual references inherited from `chaser_bout_response` (the object's position rotated
about the arena centre) give a second, within-epoch control: the fish cannot flee something
that is not there, so a virtual trace that mirrors the object trace means the "escape" is
wall geometry.

TWO STATISTICAL TRAPS, BOTH OF WHICH WE FELL INTO
-------------------------------------------------
*Pooling events.* 2,097 escape events from 32 fish are not 2,097 independent samples. The
per-event tables here exist so a caller can aggregate to one number per fish FIRST. The
`pursuit/` group is already reduced to a per-recording median. Do not pool `events/` across
recordings and t-test it.

*Net-change statistics.* Escape (+8 mm) and recapture (-5 mm) have opposite signs and very
nearly cancel: the net distance change from onset to +4 s is +1.6 mm, p=0.55 -- a null that
means nothing, because the statistic cannot see either effect. `pursuit/` therefore stores
`gain_mm` and `recapture_mm` *separately*, and `net_mm` only as a diagnostic. Read the
decomposition.

MEASUREMENT CEILING
-------------------
At 100 fps a larval C-start (~15-20 ms) spans 1.5-2 frames, so a centroid-differenced peak
speed systematically UNDER-reads true escape velocity. Nothing in this cohort exceeds
~300 mm/s while the literature reports 100-500 mm/s. Absolute `peak_speed_mm_s` values here
are not comparable to high-speed-imaging work. The *contrasts* (epoch, reference) are
unaffected -- the same under-reading applies to every condition. `threshold_sweep/` exists
so this can be checked rather than assumed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
import math
from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import zarr  # noqa: E402

from fisheye.analysis.chaser_bout_response import (
    REFERENCE_KIND_DISH_CENTER,
    REFERENCE_KIND_OBJECT,
    REFERENCE_KIND_VIRTUAL,
    BoutReference,
)
from fisheye.analysis.chaser_distance_runs import _bytes_array, _write_array
from fisheye.analysis.chaser_escape_freeze import (
    _controller_trial_segments,
    _dense_controller_state,
    _load_chaser_states,
    _resolve_trigger_radius,
    _select_trial_trigger,
    _source_stimulus_path,
)
from fisheye.analysis.chaser_radial_occupancy import (
    ChaserRadialEpoch,
    _decode_text_column,
    _open_root,
    _resolve_arena_geometry,
    _resolve_chaser_distance_run,
    _safe_float,
)
from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.plot_artifacts import write_png_visualization_artifact
from fisheye.shared.run_lineage_fingerprint import build_run_lineage_payload, write_run_lineage_attrs
from fisheye.shared.system_metadata import get_git_info
from fisheye.shared.zarr_run_completion import resolve_authoritative_run_name


SCHEMA_ID = "palette.chaser_escape_events.v3"
SCHEMA_VERSION = 3
METHOD = "peak_speed_escape_events_with_turn_tier_pursuit_decomposition_and_trial_habituation"
METHOD_VERSION = "3"
COMPONENT_PARENT_NAME = "chaser_escape_events"
DEFAULT_COMPONENT_NAME = "chaser_escape_events_v1"

BOUT_RESPONSE_PARENT = "chaser_bout_response"

DEFAULT_PEAK_SPEED_THRESHOLD_MM_S = 100.0
DEFAULT_HIGH_TURN_THRESHOLD_DEG = 45.0    # a fast bout with |turn| >= this is a "turn" escape
DEFAULT_THRESHOLD_SWEEP_MM_S = (60.0, 100.0, 150.0, 200.0, 300.0)
DEFAULT_PRE_WINDOW_S = 1.0
DEFAULT_POST_WINDOW_S = 5.0
DEFAULT_GAIN_WINDOW_S = 0.5
DEFAULT_RECAPTURE_WINDOW_S = 4.0
DEFAULT_ONSET_BASELINE_FRAMES = 4
DEFAULT_MAX_WINDOW_DROPOUT_FRACTION = 0.25
DEFAULT_MIN_EVENTS_FOR_TRACE = 3
MIN_TRIALS_FOR_HABITUATION_SLOPE = 4


@dataclass(frozen=True)
class ChaserEscapeEventsResult:
    zarr_path: str
    recording_id: str
    component_name: str
    chaser_distance_run_name: str
    chaser_distance_run_path: str
    source_bout_response_component: str
    source_bout_response_path: str
    fps: float
    total_frames: int
    pixels_per_mm_projector: float
    peak_speed_threshold_mm_s: float
    high_turn_threshold_deg: float
    threshold_sweep_mm_s: np.ndarray
    pre_window_s: float
    post_window_s: float
    gain_window_s: float
    recapture_window_s: float
    max_window_dropout_fraction: float
    min_events_for_trace: int
    epochs: tuple[ChaserRadialEpoch, ...]
    references: tuple[BoutReference, ...]
    # per escape event (n_events,)
    event_bout_id: np.ndarray
    event_epoch_index: np.ndarray
    event_start_frame: np.ndarray
    event_peak_speed_mm_s: np.ndarray
    event_turn_deg: np.ndarray
    event_is_high_turn: np.ndarray          # (n_events,) fast AND |turn| >= high_turn_threshold_deg
    event_trace_usable: np.ndarray          # (n_events,) window in-bounds and low dropout
    # per escape event, per reference (n_events, n_refs)
    event_distance_at_onset_mm: np.ndarray
    event_gain_mm: np.ndarray               # d(+gain_window) - d(onset)
    event_recapture_mm: np.ndarray          # d(+gain_window) - d(+recapture_window); >0 == ground lost
    event_net_mm: np.ndarray                # d(+recapture_window) - d(onset)
    # per epoch (n_epochs,)
    escape_count: np.ndarray
    high_turn_escape_count: np.ndarray
    ordinary_bout_count: np.ndarray
    epoch_duration_s: np.ndarray
    valid_duration_s: np.ndarray
    tracking_dropout_fraction: np.ndarray
    escape_rate_per_min: np.ndarray
    escape_rate_per_valid_min: np.ndarray
    high_turn_escape_rate_per_valid_min: np.ndarray
    high_turn_fraction: np.ndarray          # high-turn escapes / all escapes, per epoch
    escape_bout_fraction: np.ndarray
    # per (epoch, reference)
    escape_onset_distance_mm: np.ndarray    # median over escape events
    ordinary_onset_distance_mm: np.ndarray  # median over non-escape bouts
    proximity_shift_mm: np.ndarray          # escape - ordinary; negative == escapes fire closer
    pursuit_gain_mm: np.ndarray             # median over events
    pursuit_recapture_mm: np.ndarray
    pursuit_net_mm: np.ndarray
    pursuit_event_count: np.ndarray
    # per (epoch, reference, time)
    trace_time_s: np.ndarray                # (n_time,)
    trace_delta_distance_mm: np.ndarray     # median, baseline-subtracted at onset
    trace_event_count: np.ndarray
    # per (threshold, epoch)
    sweep_escape_count: np.ndarray
    sweep_rate_per_valid_min: np.ndarray
    # per chase trial (n_trials,) -- empty when the stimulus run has no chase_trial_id
    trial_id: np.ndarray
    trial_ordinal: np.ndarray
    trial_start_frame: np.ndarray
    trial_end_frame: np.ndarray
    trial_trigger_frame: np.ndarray
    trial_trigger_distance_mm: np.ndarray
    trial_escape_count: np.ndarray
    trial_any_escape: np.ndarray
    trial_valid_s: np.ndarray
    trial_dropout_fraction: np.ndarray
    trial_escape_rate_per_valid_s: np.ndarray
    trial_wall_distance_at_trigger_mm: np.ndarray
    trial_first_escape_latency_s: np.ndarray
    trial_segmentation_source: str
    habituation_slope_per_trial: float      # within-fish OLS slope of escape rate on ordinal
    habituation_slope_any_escape: float
    status: str
    qc_warnings: tuple[str, ...]
    summary: dict[str, Any]
    diagnostics: dict[str, Any]


# ----------------------------------------------------------------------------------------
# Reading the parent bout-response component
# ----------------------------------------------------------------------------------------


def _resolve_bout_response(run_group: zarr.Group, run_path: str, name: str) -> tuple[zarr.Group, str, str]:
    parent = run_group.get(BOUT_RESPONSE_PARENT)
    if parent is None or not len(list(parent.keys())):
        raise ValueError(
            "chaser_escape_events requires a materialized chaser_bout_response component "
            "(it supplies the bout table and the virtual references). Run it first."
        )
    resolved = str(name or "latest").strip()
    if not resolved or resolved == "latest":
        resolved = resolve_authoritative_run_name(parent) or str(parent.attrs.get("latest") or "").strip()
        if not resolved:
            resolved = sorted(parent.keys())[-1]
    if resolved not in parent:
        raise ValueError(f"chaser_bout_response component not found: {resolved}")
    return parent[resolved], resolved, f"{run_path}/{BOUT_RESPONSE_PARENT}/{resolved}"


def _read_references(component: zarr.Group) -> tuple[BoutReference, ...]:
    refs = component["references"]
    labels = _decode_text_column(np.asarray(refs["label_bytes"][:]))
    kinds = _decode_text_column(np.asarray(refs["kind_bytes"][:]))
    chaser_index = np.asarray(refs["chaser_index"][:], dtype=np.int64).reshape(-1)
    parent_index = np.asarray(refs["parent_chaser_index"][:], dtype=np.int64).reshape(-1)
    rotation = np.asarray(refs["rotation_deg"][:], dtype=np.float64).reshape(-1)
    return tuple(
        BoutReference(
            label=str(labels[i]),
            kind=str(kinds[i]),
            chaser_index=int(chaser_index[i]),
            parent_chaser_index=int(parent_index[i]),
            rotation_deg=float(rotation[i]),
        )
        for i in range(len(labels))
    )


def _read_epochs_from_component(component: zarr.Group) -> tuple[ChaserRadialEpoch, ...]:
    """The bout component's epochs are already settle-trimmed. Inherit them verbatim.

    Re-deriving them here would risk a silent divergence in the trim, which would put the
    escape rates on a different denominator than every other component in the run.
    """

    group = component["epochs"]
    labels = _decode_text_column(np.asarray(group["label_bytes"][:]))
    start = np.asarray(group["start_frame"][:], dtype=np.int64).reshape(-1)
    end = np.asarray(group["end_frame"][:], dtype=np.int64).reshape(-1)
    return tuple(
        ChaserRadialEpoch(
            window_id=int(i),
            label=str(labels[i]),
            start_frame=int(start[i]),
            end_frame=int(end[i]),
            frame_count=int(end[i] - start[i]),
            source_start_frame=int(start[i]),
            settle_excluded_frame_count=0,
            static_configuration=False,
        )
        for i in range(len(labels))
    )


def _reference_positions_px(
    references: Sequence[BoutReference],
    *,
    chaser_indices: np.ndarray,
    chaser_xy_px: np.ndarray,
    center_xy_px: tuple[float, float],
) -> np.ndarray:
    """Rebuild per-frame reference positions from the parent component's reference table.

    Reconstructed from the stored rotation angles rather than recomputed from this module's
    own defaults, so the references are identical to the ones the bout table was scored
    against by construction -- not merely by agreeing parameter values.
    """

    n_frames = int(chaser_xy_px.shape[0])
    cx, cy = float(center_xy_px[0]), float(center_xy_px[1])
    center = np.asarray([cx, cy], dtype=np.float64)
    col_of = {int(v): i for i, v in enumerate(np.asarray(chaser_indices, dtype=np.int64).reshape(-1))}

    out = np.full((n_frames, len(references), 2), np.nan, dtype=np.float64)
    for r, ref in enumerate(references):
        if ref.kind == REFERENCE_KIND_OBJECT:
            col = col_of.get(int(ref.chaser_index))
            if col is None:
                raise ValueError(f"Reference {ref.label!r} names chaser {ref.chaser_index}, absent from the run.")
            out[:, r, :] = np.asarray(chaser_xy_px[:, col, :], dtype=np.float64)
        elif ref.kind == REFERENCE_KIND_VIRTUAL:
            col = col_of.get(int(ref.parent_chaser_index))
            if col is None:
                raise ValueError(f"Virtual reference {ref.label!r} has no parent chaser in the run.")
            base = np.asarray(chaser_xy_px[:, col, :], dtype=np.float64) - center
            theta = math.radians(float(ref.rotation_deg))
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            out[:, r, 0] = base[:, 0] * cos_t - base[:, 1] * sin_t + cx
            out[:, r, 1] = base[:, 0] * sin_t + base[:, 1] * cos_t + cy
        elif ref.kind == REFERENCE_KIND_DISH_CENTER:
            out[:, r, :] = center
        else:
            raise ValueError(f"Unknown reference kind: {ref.kind!r}")
    return out


# ----------------------------------------------------------------------------------------
# Core
# ----------------------------------------------------------------------------------------


def _nan_median(values: np.ndarray) -> float:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    data = data[np.isfinite(data)]
    return float(np.median(data)) if data.size else math.nan


def build_chaser_escape_events_result(
    zarr_path: Path,
    *,
    chaser_distance_run: str = "latest",
    bout_response_component: str = "latest",
    component_name: str = DEFAULT_COMPONENT_NAME,
    peak_speed_threshold_mm_s: float = DEFAULT_PEAK_SPEED_THRESHOLD_MM_S,
    high_turn_threshold_deg: float = DEFAULT_HIGH_TURN_THRESHOLD_DEG,
    threshold_sweep_mm_s: Sequence[float] = DEFAULT_THRESHOLD_SWEEP_MM_S,
    pre_window_s: float = DEFAULT_PRE_WINDOW_S,
    post_window_s: float = DEFAULT_POST_WINDOW_S,
    gain_window_s: float = DEFAULT_GAIN_WINDOW_S,
    recapture_window_s: float = DEFAULT_RECAPTURE_WINDOW_S,
    onset_baseline_frames: int = DEFAULT_ONSET_BASELINE_FRAMES,
    max_window_dropout_fraction: float = DEFAULT_MAX_WINDOW_DROPOUT_FRACTION,
    min_events_for_trace: int = DEFAULT_MIN_EVENTS_FOR_TRACE,
) -> ChaserEscapeEventsResult:
    root = _open_root(zarr_path, mode="r")
    run_group, run_name, run_path = _resolve_chaser_distance_run(root, chaser_distance_run)

    pixels_per_mm = _safe_float(run_group.attrs.get("pixels_per_mm_projector"))
    if not math.isfinite(pixels_per_mm) or pixels_per_mm <= 0:
        raise ValueError("Chaser-distance run lacks a positive pixels_per_mm_projector attr.")
    fps = _safe_float(run_group.attrs.get("fps"), math.nan)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Chaser-distance run lacks a positive fps attr.")

    bout_component, bout_name, bout_path = _resolve_bout_response(run_group, run_path, bout_response_component)
    references = _read_references(bout_component)
    epochs = _read_epochs_from_component(bout_component)

    positions = run_group["positions"]
    fish_xy = np.asarray(positions["fish_centroid_arena_xy"][:], dtype=np.float64)
    fish_valid = np.asarray(positions["fish_valid"][:], dtype=bool)
    chaser_xy = np.asarray(positions["chaser_arena_xy"][:], dtype=np.float64)
    chaser_indices = np.asarray(run_group["chasers/chaser_index"][:], dtype=np.int64).reshape(-1)
    total_frames = int(fish_xy.shape[0])

    geometry = _resolve_arena_geometry(root, run_group, pixels_per_mm=float(pixels_per_mm))
    if geometry.shape != "circle" or geometry.center_x_px is None or geometry.center_y_px is None:
        raise ValueError(
            "chaser_escape_events requires a circular arena geometry: the virtual references "
            "it inherits are rotations about the arena centre."
        )
    ref_xy = _reference_positions_px(
        references,
        chaser_indices=chaser_indices,
        chaser_xy_px=chaser_xy,
        center_xy_px=(float(geometry.center_x_px), float(geometry.center_y_px)),
    )
    # (frame, ref) distance in mm. NaN wherever the fish is untracked -- an untracked frame
    # must not silently contribute a zero to a trace.
    delta = ref_xy - fish_xy[:, None, :]
    distance = np.linalg.norm(delta, axis=2) / float(pixels_per_mm)
    distance[~fish_valid, :] = np.nan

    bouts = bout_component["bouts"]
    bout_id = np.asarray(bouts["bout_id"][:], dtype=np.int64).reshape(-1)
    bout_epoch = np.asarray(bouts["epoch_index"][:], dtype=np.int64).reshape(-1)
    bout_start = np.asarray(bouts["start_frame"][:], dtype=np.int64).reshape(-1)
    bout_peak = np.asarray(bouts["peak_speed_mm_s"][:], dtype=np.float64).reshape(-1)
    bout_turn = np.asarray(bouts["turn_deg"][:], dtype=np.float64).reshape(-1)
    bout_valid = np.asarray(bouts["valid"][:], dtype=bool).reshape(-1)
    onset_distance_all = np.asarray(
        bout_component["bouts_per_reference/distance_at_onset_mm"][:], dtype=np.float64
    )

    usable = bout_valid & np.isfinite(bout_peak)
    is_escape = usable & (bout_peak > float(peak_speed_threshold_mm_s))
    is_ordinary = usable & ~is_escape
    # An escape is classified by SPEED alone; the turn tier is an added sub-classification that
    # never changes which bouts are escapes. A "turn" escape is fast AND high-angle (C-start-like
    # reorientation); a "dash" escape is fast but nearly straight (a forward sprint). Turn is
    # ~4x enriched in escapes over ordinary bouts, and the high-turn tier is if anything MORE
    # chase-specific than the velocity cut alone.
    is_high_turn = is_escape & np.isfinite(bout_turn) & (np.abs(bout_turn) >= float(high_turn_threshold_deg))

    n_epochs = len(epochs)
    n_refs = len(references)
    qc: list[str] = []

    # ---- per-epoch rates, on validly-tracked time ---------------------------------------
    # Wall-clock rate is the wrong denominator when a third of the cohort drops >10% of the
    # chase epoch: it would report a fish that vanished for 80 s as having escaped less.
    escape_count = np.zeros(n_epochs, dtype=np.int64)
    high_turn_count = np.zeros(n_epochs, dtype=np.int64)
    ordinary_count = np.zeros(n_epochs, dtype=np.int64)
    epoch_duration = np.zeros(n_epochs, dtype=np.float64)
    valid_duration = np.zeros(n_epochs, dtype=np.float64)
    dropout = np.full(n_epochs, np.nan, dtype=np.float64)
    for e, epoch in enumerate(epochs):
        lo, hi = int(epoch.start_frame), int(epoch.end_frame)
        span = max(0, hi - lo)
        epoch_duration[e] = float(span) / float(fps)
        n_valid = int(np.count_nonzero(fish_valid[lo:hi])) if span else 0
        valid_duration[e] = float(n_valid) / float(fps)
        dropout[e] = 1.0 - (float(n_valid) / float(span)) if span else math.nan
        escape_count[e] = int(np.count_nonzero(is_escape & (bout_epoch == e)))
        high_turn_count[e] = int(np.count_nonzero(is_high_turn & (bout_epoch == e)))
        ordinary_count[e] = int(np.count_nonzero(is_ordinary & (bout_epoch == e)))

    with np.errstate(divide="ignore", invalid="ignore"):
        rate_per_min = np.where(epoch_duration > 0, escape_count / epoch_duration * 60.0, np.nan)
        rate_per_valid_min = np.where(valid_duration > 0, escape_count / valid_duration * 60.0, np.nan)
        high_turn_rate_per_valid_min = np.where(
            valid_duration > 0, high_turn_count / valid_duration * 60.0, np.nan
        )
        high_turn_fraction = np.where(escape_count > 0, high_turn_count / np.maximum(escape_count, 1), np.nan)
        total_bouts = escape_count + ordinary_count
        escape_fraction = np.where(total_bouts > 0, escape_count / np.maximum(total_bouts, 1), np.nan)

    # ---- the events ---------------------------------------------------------------------
    ev = np.flatnonzero(is_escape)
    n_events = int(ev.size)
    if n_events == 0:
        qc.append(f"no_bout_exceeds_peak_speed_threshold_{peak_speed_threshold_mm_s:g}mm_s")

    pre_f = max(1, int(round(float(pre_window_s) * float(fps))))
    post_f = max(1, int(round(float(post_window_s) * float(fps))))
    n_time = pre_f + post_f
    trace_time = (np.arange(-pre_f, post_f, dtype=np.float64)) / float(fps)
    gain_idx = int(np.argmin(np.abs(trace_time - float(gain_window_s))))
    recap_idx = int(np.argmin(np.abs(trace_time - float(recapture_window_s))))

    event_traces = np.full((n_events, n_refs, n_time), np.nan, dtype=np.float64)
    trace_usable = np.zeros(n_events, dtype=bool)
    baseline_n = max(1, int(onset_baseline_frames))

    # The window must lie wholly inside the event's OWN epoch. Between epochs the objects are
    # repositioned, so a trace that runs past the boundary would splice a teleport into the
    # distance curve and read as a huge spurious recapture.
    epoch_bounds = [(int(e.start_frame), int(e.end_frame)) for e in epochs]

    for k, b in enumerate(ev):
        f0 = int(bout_start[b])
        lo, hi = f0 - pre_f, f0 + post_f
        if lo < 0 or hi > total_frames:
            continue
        e_idx = int(bout_epoch[b])
        if not (0 <= e_idx < n_epochs):
            continue
        e_lo, e_hi = epoch_bounds[e_idx]
        if lo < e_lo or hi > e_hi:
            continue
        window_valid = fish_valid[lo:hi]
        if float(np.mean(~window_valid)) > float(max_window_dropout_fraction):
            continue
        seg = distance[lo:hi, :]                       # (time, ref)
        base = np.nanmedian(seg[max(0, pre_f - baseline_n + 1) : pre_f + 1, :], axis=0)  # (ref,)
        if not np.any(np.isfinite(base)):
            continue
        event_traces[k] = (seg - base[None, :]).T      # (ref, time)
        trace_usable[k] = True

    dropped = n_events - int(np.count_nonzero(trace_usable))
    if n_events and dropped:
        qc.append(f"escape_events_without_a_clean_window:{dropped}_of_{n_events}")
        # Events are still counted in the rate; they just cannot carry a trace. Silently
        # dropping them from BOTH would understate the escape rate near epoch edges.

    event_gain = event_traces[:, :, gain_idx] if n_events else np.zeros((0, n_refs))
    event_at_recap = event_traces[:, :, recap_idx] if n_events else np.zeros((0, n_refs))
    event_recapture = event_gain - event_at_recap     # >0 == the fish lost ground back
    event_net = event_at_recap

    event_onset_distance = onset_distance_all[ev] if n_events else np.zeros((0, n_refs))

    # ---- per (epoch, reference) reductions ----------------------------------------------
    esc_onset = np.full((n_epochs, n_refs), np.nan, dtype=np.float64)
    ord_onset = np.full((n_epochs, n_refs), np.nan, dtype=np.float64)
    gain = np.full((n_epochs, n_refs), np.nan, dtype=np.float64)
    recapture = np.full((n_epochs, n_refs), np.nan, dtype=np.float64)
    net = np.full((n_epochs, n_refs), np.nan, dtype=np.float64)
    pursuit_n = np.zeros((n_epochs, n_refs), dtype=np.int64)
    trace = np.full((n_epochs, n_refs, n_time), np.nan, dtype=np.float64)
    trace_n = np.zeros((n_epochs, n_refs, n_time), dtype=np.int64)

    ev_epoch = bout_epoch[ev] if n_events else np.zeros(0, dtype=np.int64)
    for e in range(n_epochs):
        in_epoch = ev_epoch == e
        ord_rows = np.flatnonzero(is_ordinary & (bout_epoch == e))
        for r in range(n_refs):
            if ord_rows.size:
                ord_onset[e, r] = _nan_median(onset_distance_all[ord_rows, r])
            if np.any(in_epoch):
                esc_onset[e, r] = _nan_median(event_onset_distance[in_epoch, r])
            rows = np.flatnonzero(in_epoch & trace_usable)
            if rows.size < int(min_events_for_trace):
                continue
            block = event_traces[rows, r, :]           # (event, time)
            counts = np.count_nonzero(np.isfinite(block), axis=0)
            with np.errstate(invalid="ignore"):
                med = np.nanmedian(block, axis=0)
            med[counts < int(min_events_for_trace)] = np.nan
            trace[e, r, :] = med
            trace_n[e, r, :] = counts
            gain[e, r] = _nan_median(event_gain[rows, r])
            recapture[e, r] = _nan_median(event_recapture[rows, r])
            net[e, r] = _nan_median(event_net[rows, r])
            pursuit_n[e, r] = int(rows.size)

    proximity_shift = esc_onset - ord_onset

    # ---- threshold sweep -----------------------------------------------------------------
    sweep = np.asarray(list(threshold_sweep_mm_s), dtype=np.float64)
    sweep_count = np.zeros((sweep.size, n_epochs), dtype=np.int64)
    sweep_rate = np.full((sweep.size, n_epochs), np.nan, dtype=np.float64)
    for t, thr in enumerate(sweep):
        hit = usable & (bout_peak > float(thr))
        for e in range(n_epochs):
            c = int(np.count_nonzero(hit & (bout_epoch == e)))
            sweep_count[t, e] = c
            if valid_duration[e] > 0:
                sweep_rate[t, e] = float(c) / float(valid_duration[e]) * 60.0

    # ---- trial-locked: does the escape response habituate across chase trials? ------------
    #
    # The chase epoch is not 180 s of continuous chasing: it is ~12 experimenter-initiated
    # trials of ~5 s. Escape collapses after trial 1-2 while freezing rises -- an active-to-
    # passive defence switch. It is only visible per trial; averaged over the epoch it is
    # invisible.
    #
    # WALL DISTANCE IS RECORDED HERE ON PURPOSE. The fish moves to the wall after trial 1 and
    # stays, and at the wall it rarely escapes -- so "habituation" and "cornered against the
    # wall with nowhere to flee" are confounded by construction. Two checks rule the trap out
    # (see the contract), but wall position is DOWNSTREAM of the chase: it is a mediator of
    # the response, not a nuisance covariate. Regressing it out would be conditioning on a
    # post-treatment variable. It is stored so the question can be asked, not silently
    # "controlled for".
    trial_id = np.zeros(0, dtype=np.int64)
    trial_ordinal = np.zeros(0, dtype=np.int32)
    trial_start = np.zeros(0, dtype=np.int64)
    trial_end = np.zeros(0, dtype=np.int64)
    trial_trigger = np.zeros(0, dtype=np.int64)
    trial_trigger_dist = np.zeros(0, dtype=np.float64)
    trial_escapes = np.zeros(0, dtype=np.int64)
    trial_any = np.zeros(0, dtype=bool)
    trial_valid_s = np.zeros(0, dtype=np.float64)
    trial_dropout = np.zeros(0, dtype=np.float64)
    trial_rate = np.zeros(0, dtype=np.float64)
    trial_wall = np.zeros(0, dtype=np.float64)
    trial_latency = np.zeros(0, dtype=np.float64)
    seg_source = "unavailable"
    hab_slope = math.nan
    hab_slope_any = math.nan

    try:
        _stim_run, stim_path = _source_stimulus_path(run_group)
        chaser_states = _load_chaser_states(root, stim_path)
        stim_frames = np.asarray(run_group["frames/stimulus_frame_num"][:], dtype=np.int64)
        # Trials belong to the chaser that actually chases. Objects that never move have no
        # trials, so pick the chaser the controller marks active rather than assuming index 0.
        active_index = int(chaser_indices[0])
        for cand in np.asarray(chaser_indices, dtype=np.int64).reshape(-1):
            probe = _dense_controller_state(chaser_states, stimulus_frame_nums=stim_frames,
                                            chaser_index=int(cand))
            if int(np.count_nonzero(probe["active"])) > 0:
                active_index = int(cand)
                break
        controller = _dense_controller_state(chaser_states, stimulus_frame_nums=stim_frames,
                                             chaser_index=active_index)
        segments, seg_source = _controller_trial_segments(controller["active"], controller["trial_id"])
    except Exception as exc:                      # noqa: BLE001 -- a missing stimulus run is not fatal
        segments = []
        qc.append(f"trial_segmentation_unavailable:{type(exc).__name__}")

    if segments:
        radius_mm, _rsrc, _rov, _rw = _resolve_trigger_radius(
            root, source_stimulus_path=stim_path, chaser_states=chaser_states,
            chaser_index=active_index, trigger_radius_mm=None,
        )
        obj_col = {int(v): i for i, v in enumerate(chaser_indices)}.get(active_index, 0)
        obj_ref = next(
            (i for i, rr in enumerate(references)
             if rr.kind == REFERENCE_KIND_OBJECT and rr.chaser_index == active_index),
            None,
        )
        dist_to_chaser = distance[:, obj_ref] if obj_ref is not None else np.full(total_frames, np.nan)
        radius_mm_arena = float(geometry.radius_px) / float(pixels_per_mm)
        fish_radius_mm = np.hypot(
            fish_xy[:, 0] - float(geometry.center_x_px), fish_xy[:, 1] - float(geometry.center_y_px)
        ) / float(pixels_per_mm)
        wall_mm = radius_mm_arena - fish_radius_mm
        wall_mm[~fish_valid] = np.nan

        esc_frames = bout_start[ev] if n_events else np.zeros(0, dtype=np.int64)
        k = len(segments)
        trial_id = np.zeros(k, dtype=np.int64)
        trial_ordinal = np.zeros(k, dtype=np.int32)
        trial_start = np.zeros(k, dtype=np.int64)
        trial_end = np.zeros(k, dtype=np.int64)
        trial_trigger = np.zeros(k, dtype=np.int64)
        trial_trigger_dist = np.full(k, np.nan)
        trial_escapes = np.zeros(k, dtype=np.int64)
        trial_any = np.zeros(k, dtype=bool)
        trial_valid_s = np.zeros(k, dtype=np.float64)
        trial_dropout = np.full(k, np.nan)
        trial_rate = np.full(k, np.nan)
        trial_wall = np.full(k, np.nan)
        trial_latency = np.full(k, np.nan)

        for i, (start, end, tid) in enumerate(segments):
            lo, hi = int(start), int(end) + 1
            span = max(1, hi - lo)
            trig, _prox, _src = _select_trial_trigger(dist_to_chaser, np.arange(lo, hi, dtype=np.int64),
                                                      trigger_radius_mm=float(radius_mm))
            n_valid_t = int(np.count_nonzero(fish_valid[lo:hi]))
            inside = esc_frames[(esc_frames >= lo) & (esc_frames < hi)] if n_events else np.zeros(0, np.int64)

            trial_id[i] = int(tid)
            trial_ordinal[i] = int(i + 1)
            trial_start[i] = lo
            trial_end[i] = hi - 1
            trial_trigger[i] = int(trig)
            if 0 <= trig < total_frames:
                trial_trigger_dist[i] = float(dist_to_chaser[trig])
                trial_wall[i] = float(wall_mm[trig])
            trial_escapes[i] = int(inside.size)
            trial_any[i] = bool(inside.size)
            trial_valid_s[i] = float(n_valid_t) / float(fps)
            trial_dropout[i] = 1.0 - (float(n_valid_t) / float(span))
            # Escapes per *validly tracked* second: a trial that lost half its frames must not
            # read as a trial where the fish escaped half as much.
            if trial_valid_s[i] > 0.5:
                trial_rate[i] = float(inside.size) / float(trial_valid_s[i])
            if inside.size and 0 <= trig < total_frames:
                trial_latency[i] = float(int(inside.min()) - int(trig)) / float(fps)

        ordinals = trial_ordinal.astype(np.float64)
        for target, slot in ((trial_rate, "rate"), (trial_any.astype(np.float64), "any")):
            fit = np.isfinite(target)
            if int(np.count_nonzero(fit)) >= MIN_TRIALS_FOR_HABITUATION_SLOPE and float(np.std(ordinals[fit])) > 0:
                slope = float(np.polyfit(ordinals[fit], target[fit], 1)[0])
                if slot == "rate":
                    hab_slope = slope
                else:
                    hab_slope_any = slope

    # ---- QC ------------------------------------------------------------------------------
    finite_peak = bout_peak[usable]
    max_peak = float(np.max(finite_peak)) if finite_peak.size else math.nan
    if math.isfinite(max_peak) and max_peak < float(peak_speed_threshold_mm_s):
        qc.append(f"peak_speed_never_reaches_threshold:max_{max_peak:.1f}mm_s")
    for e, epoch in enumerate(epochs):
        if math.isfinite(dropout[e]) and dropout[e] > 0.10:
            qc.append(f"high_tracking_dropout:{epoch.label}={dropout[e]:.3f}")

    object_refs = [i for i, r in enumerate(references) if r.kind == REFERENCE_KIND_OBJECT]
    labels = [e.label for e in epochs]
    summary: dict[str, Any] = {
        "peak_speed_threshold_mm_s": float(peak_speed_threshold_mm_s),
        "high_turn_threshold_deg": float(high_turn_threshold_deg),
        "escape_event_count": int(n_events),
        "high_turn_escape_event_count": int(np.count_nonzero(is_high_turn)),
        "escape_event_count_with_clean_window": int(np.count_nonzero(trace_usable)),
        "max_peak_speed_mm_s": max_peak,
    }
    for e, label in enumerate(labels):
        summary[f"escape_rate_per_valid_min_{label}"] = float(rate_per_valid_min[e])
        summary[f"escape_count_{label}"] = int(escape_count[e])
        summary[f"high_turn_escape_rate_per_valid_min_{label}"] = float(high_turn_rate_per_valid_min[e])
        summary[f"high_turn_escape_count_{label}"] = int(high_turn_count[e])
        summary[f"high_turn_fraction_{label}"] = float(high_turn_fraction[e])
        summary[f"tracking_dropout_{label}"] = float(dropout[e])
    for e, label in enumerate(labels):
        for r in object_refs:
            ref = references[r]
            summary[f"proximity_shift_mm_{label}_{ref.label}"] = float(proximity_shift[e, r])
            summary[f"pursuit_gain_mm_{label}_{ref.label}"] = float(gain[e, r])
            summary[f"pursuit_recapture_mm_{label}_{ref.label}"] = float(recapture[e, r])
            summary[f"pursuit_event_count_{label}_{ref.label}"] = int(pursuit_n[e, r])

    if len(trial_ordinal):
        early = trial_ordinal <= 2
        late = trial_ordinal >= 5
        summary["trial_count"] = int(len(trial_ordinal))
        summary["trial_segmentation_source"] = str(seg_source)
        summary["habituation_slope_escapes_per_valid_s_per_trial"] = float(hab_slope)
        summary["habituation_slope_any_escape_per_trial"] = float(hab_slope_any)
        summary["escape_rate_per_valid_s_trials_1_2"] = _nan_median(trial_rate[early])
        summary["escape_rate_per_valid_s_trials_5plus"] = _nan_median(trial_rate[late])
        summary["p_any_escape_trials_1_2"] = float(np.mean(trial_any[early])) if early.any() else math.nan
        summary["p_any_escape_trials_5plus"] = float(np.mean(trial_any[late])) if late.any() else math.nan
        summary["wall_distance_at_trigger_mm_trials_1_2"] = _nan_median(trial_wall[early])
        summary["wall_distance_at_trigger_mm_trials_5plus"] = _nan_median(trial_wall[late])
        summary["median_first_escape_latency_s"] = _nan_median(trial_latency)

    diagnostics = {
        "escape_definition": (
            f"a valid bout whose peak_speed_mm_s exceeds {peak_speed_threshold_mm_s:g} mm/s. Not a "
            "cluster: escapes are ~1.5% of bouts and k-means will not allocate a centroid to them."
        ),
        "turn_tier": (
            f"an escape is 'high_turn' (a C-start-like reorientation) when its |turn_deg| also "
            f">= {high_turn_threshold_deg:g} deg; otherwise it is a straight forward 'dash'. The tier "
            "NEVER changes which bouts are escapes -- escape is peak-speed only. Turn is ~4x enriched "
            "in escapes vs ordinary bouts, and the high-turn tier is if anything more chase-specific. "
            "At 100 fps turn_deg is a net heading change, not a resolved C-start bend, so this is a "
            "coarse split (forward dash vs turn-away), not a kinematic C-start classifier."
        ),
        "rate_denominator": (
            "valid_duration_s = validly-tracked frames / fps. Wall-clock rate is also written, but "
            "escape_rate_per_valid_min is the one to use: detector recall collapses on frozen fish, "
            "so an epoch with 40% dropout would otherwise report an artificially low escape rate."
        ),
        "trace_definition": (
            "distance to each reference, aligned on escape bout onset and baseline-subtracted at "
            f"onset (median of the last {baseline_n} frames up to and including onset). Positive == "
            "further from the reference than when the escape fired."
        ),
        "pursuit_decomposition": (
            f"gain_mm = trace at +{gain_window_s:g}s (did the escape gain ground). "
            f"recapture_mm = gain - trace at +{recapture_window_s:g}s (did the chaser take it back; "
            ">0 means ground lost). net_mm is their difference and is a DIAGNOSTIC ONLY -- gain and "
            "recapture have opposite signs and nearly cancel, so a net-change test is underpowered "
            "by construction and will return a spurious null."
        ),
        "static_object_control": (
            "the same object in a non-chasing epoch (pre/post). A fast bout there produces no "
            "distance gain, which is what distinguishes an escape from a merely fast bout. Compare "
            "gain_mm across epochs for the same reference."
        ),
        "virtual_control": (
            "references inherited verbatim from the parent chaser_bout_response component and "
            "reconstructed from its stored rotation angles. A fish cannot flee something that is "
            "not there, so an object trace matched by its virtual twins is wall geometry."
        ),
        "measurement_ceiling": (
            "100 fps: a larval C-start spans 1.5-2 frames, so centroid-differenced peak_speed_mm_s "
            "UNDER-reads true escape velocity and is not comparable to high-speed-imaging values. "
            "Contrasts across epoch/reference are unaffected; threshold_sweep/ lets this be checked."
        ),
        "pseudoreplication_note": (
            "events/ holds one row per escape, and one fish contributes many. Reduce to a per-fish "
            "value before any cross-recording test. pursuit/ and traces/ are already per-recording "
            "medians and are the safe things to aggregate."
        ),
        "trial_definition": (
            "chase trials from the controller's chase_trial_id (see chaser_escape_freeze, whose "
            "segmentation this reuses): ~12 experimenter-initiated trials of ~5 s inside the 180 s "
            f"chase epoch. Segmentation source: {seg_source}."
        ),
        "habituation": (
            "escape collapses after trial 1-2 while freezing rises: an active-to-passive defence "
            "switch inside ~10-15 s of cumulative chase. Averaged over the epoch it is invisible. "
            "trial_escape_rate_per_valid_s is the readout; trial_escape_count is NOT, because "
            "trials differ in how much of the fish was tracked."
        ),
        "wall_confound": (
            "trial_wall_distance_at_trigger_mm is stored because habituation is CONFOUNDED with it: "
            "the fish moves to the wall after trial 1 (9.6 mm -> ~2 mm) and at the wall it seldom "
            "escapes (P=0.10 vs 0.60 off the wall). Two checks rule out a geometric trap -- (a) with "
            "no chaser at all, being at the wall does not suppress fast bouts (p=0.40); (b) on trial "
            "1, fish that start AT the wall escape just as often as fish that do not (0.82 vs 0.75, "
            "p=0.68). So the wall does not prevent escape. But wall position is DOWNSTREAM of the "
            "chase -- a mediator, not a nuisance covariate -- so do not regress it out; that is "
            "conditioning on a post-treatment variable."
        ),
        "source_bout_response_component": str(bout_name),
    }

    recording_id = str(root.attrs.get("recording_id") or root.attrs.get("recording_name") or Path(zarr_path).stem)
    status = "ok" if n_events else "no_escape_events"

    return ChaserEscapeEventsResult(
        zarr_path=str(zarr_path),
        recording_id=recording_id,
        component_name=str(component_name),
        chaser_distance_run_name=run_name,
        chaser_distance_run_path=run_path,
        source_bout_response_component=str(bout_name),
        source_bout_response_path=str(bout_path),
        fps=float(fps),
        total_frames=int(total_frames),
        pixels_per_mm_projector=float(pixels_per_mm),
        peak_speed_threshold_mm_s=float(peak_speed_threshold_mm_s),
        high_turn_threshold_deg=float(high_turn_threshold_deg),
        threshold_sweep_mm_s=sweep.astype(np.float32),
        pre_window_s=float(pre_window_s),
        post_window_s=float(post_window_s),
        gain_window_s=float(gain_window_s),
        recapture_window_s=float(recapture_window_s),
        max_window_dropout_fraction=float(max_window_dropout_fraction),
        min_events_for_trace=int(min_events_for_trace),
        epochs=epochs,
        references=references,
        event_bout_id=bout_id[ev].astype(np.int64),
        event_epoch_index=ev_epoch.astype(np.int16),
        event_start_frame=bout_start[ev].astype(np.int64),
        event_peak_speed_mm_s=bout_peak[ev].astype(np.float32),
        event_turn_deg=bout_turn[ev].astype(np.float32),
        event_is_high_turn=is_high_turn[ev] if n_events else np.zeros(0, dtype=bool),
        event_trace_usable=trace_usable,
        event_distance_at_onset_mm=event_onset_distance.astype(np.float32),
        event_gain_mm=event_gain.astype(np.float32),
        event_recapture_mm=event_recapture.astype(np.float32),
        event_net_mm=event_net.astype(np.float32),
        escape_count=escape_count,
        high_turn_escape_count=high_turn_count,
        ordinary_bout_count=ordinary_count,
        epoch_duration_s=epoch_duration.astype(np.float32),
        valid_duration_s=valid_duration.astype(np.float32),
        tracking_dropout_fraction=dropout.astype(np.float32),
        escape_rate_per_min=rate_per_min.astype(np.float32),
        escape_rate_per_valid_min=rate_per_valid_min.astype(np.float32),
        high_turn_escape_rate_per_valid_min=high_turn_rate_per_valid_min.astype(np.float32),
        high_turn_fraction=high_turn_fraction.astype(np.float32),
        escape_bout_fraction=escape_fraction.astype(np.float32),
        escape_onset_distance_mm=esc_onset.astype(np.float32),
        ordinary_onset_distance_mm=ord_onset.astype(np.float32),
        proximity_shift_mm=proximity_shift.astype(np.float32),
        pursuit_gain_mm=gain.astype(np.float32),
        pursuit_recapture_mm=recapture.astype(np.float32),
        pursuit_net_mm=net.astype(np.float32),
        pursuit_event_count=pursuit_n,
        trace_time_s=trace_time.astype(np.float32),
        trace_delta_distance_mm=trace.astype(np.float32),
        trace_event_count=trace_n,
        sweep_escape_count=sweep_count,
        sweep_rate_per_valid_min=sweep_rate.astype(np.float32),
        trial_id=trial_id,
        trial_ordinal=trial_ordinal,
        trial_start_frame=trial_start,
        trial_end_frame=trial_end,
        trial_trigger_frame=trial_trigger,
        trial_trigger_distance_mm=trial_trigger_dist.astype(np.float32),
        trial_escape_count=trial_escapes,
        trial_any_escape=trial_any,
        trial_valid_s=trial_valid_s.astype(np.float32),
        trial_dropout_fraction=trial_dropout.astype(np.float32),
        trial_escape_rate_per_valid_s=trial_rate.astype(np.float32),
        trial_wall_distance_at_trigger_mm=trial_wall.astype(np.float32),
        trial_first_escape_latency_s=trial_latency.astype(np.float32),
        trial_segmentation_source=str(seg_source),
        habituation_slope_per_trial=float(hab_slope),
        habituation_slope_any_escape=float(hab_slope_any),
        status=status,
        qc_warnings=tuple(qc),
        summary=json_attr_safe(summary),
        diagnostics=json_attr_safe(diagnostics),
    )


# ----------------------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------------------


def render_escape_events_png(result: ChaserEscapeEventsResult, *, dpi: int = 150) -> bytes:
    epochs = [e.label for e in result.epochs]
    objects = [i for i, r in enumerate(result.references) if r.kind == REFERENCE_KIND_OBJECT]
    fig, axes = plt.subplots(1, 2 + len(objects), figsize=(5.0 + 4.2 * len(objects), 4.0), constrained_layout=True)
    axes = np.atleast_1d(axes)

    ax = axes[0]
    x = range(len(epochs))
    total = np.asarray(result.escape_rate_per_valid_min, dtype=float)
    hi = np.asarray(result.high_turn_escape_rate_per_valid_min, dtype=float)
    ax.bar(x, total, color="#fca5a5", alpha=0.9, label="dash (fast, straight)")
    ax.bar(x, hi, color="#b91c1c", alpha=0.95,
           label=f"turn (|turn| ≥ {result.high_turn_threshold_deg:g}°)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(epochs, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("escapes / min of tracked time")
    ax.set_title(f"escape rate  (peak > {result.peak_speed_threshold_mm_s:g} mm/s)", fontsize=9)
    ax.legend(fontsize=6, loc="upper left")

    ax = axes[1]
    for r in objects:
        ref = result.references[r]
        ax.plot(range(len(epochs)), result.proximity_shift_mm[:, r], marker="o", label=ref.label)
    ax.axhline(0.0, color="#64748b", linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(epochs)))
    ax.set_xticklabels(epochs, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("escape onset - ordinary onset (mm)")
    ax.set_title("proximity trigger\n(negative = escapes fire closer)", fontsize=9)
    ax.legend(fontsize=7)

    t = result.trace_time_s
    for k, r in enumerate(objects):
        ax = axes[2 + k]
        ref = result.references[r]
        twins = [
            i for i, rr in enumerate(result.references)
            if rr.kind == REFERENCE_KIND_VIRTUAL and rr.parent_chaser_index == ref.chaser_index
        ]
        for e, label in enumerate(epochs):
            if not np.any(np.isfinite(result.trace_delta_distance_mm[e, r])):
                continue
            ax.plot(t, result.trace_delta_distance_mm[e, r], linewidth=1.6, label=f"{label} (object)")
            if twins:
                with np.errstate(invalid="ignore"):
                    twin = np.nanmedian(result.trace_delta_distance_mm[e, twins, :], axis=0)
                ax.plot(t, twin, linewidth=0.9, linestyle=":", alpha=0.7, label=f"{label} (virtual)")
        ax.axhline(0.0, color="#64748b", linewidth=0.8, linestyle="--")
        ax.axvline(0.0, color="#0f172a", linewidth=0.8)
        ax.set_xlabel("time from escape onset (s)")
        ax.set_ylabel("Δ distance from onset (mm)")
        ax.set_title(f"pursuit: {ref.label}\ngain then recapture", fontsize=9)
        ax.legend(fontsize=6)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    return buf.getvalue()


# ----------------------------------------------------------------------------------------
# Write
# ----------------------------------------------------------------------------------------


def write_chaser_escape_events_component(
    zarr_path: Path,
    result: ChaserEscapeEventsResult,
    *,
    overwrite: bool = False,
    write_png: bool = True,
) -> str:
    root = _open_root(zarr_path, mode="a")
    run_group = root[result.chaser_distance_run_path]
    parent = run_group.require_group(COMPONENT_PARENT_NAME)
    if result.component_name in parent:
        if not overwrite:
            raise ValueError(f"Chaser escape-events component already exists: {result.component_name}")
        del parent[result.component_name]
    component = parent.create_group(result.component_name)
    component_path = f"{result.chaser_distance_run_path}/{COMPONENT_PARENT_NAME}/{result.component_name}"

    config = component.require_group("config")
    _write_array(config, "threshold_sweep_mm_s", result.threshold_sweep_mm_s)
    config.attrs.update(
        {
            "peak_speed_threshold_mm_s": float(result.peak_speed_threshold_mm_s),
            "high_turn_threshold_deg": float(result.high_turn_threshold_deg),
            "pre_window_s": float(result.pre_window_s),
            "post_window_s": float(result.post_window_s),
            "gain_window_s": float(result.gain_window_s),
            "recapture_window_s": float(result.recapture_window_s),
            "max_window_dropout_fraction": float(result.max_window_dropout_fraction),
            "min_events_for_trace": int(result.min_events_for_trace),
        }
    )

    epochs = component.require_group("epochs")
    _write_array(epochs, "label_bytes", _bytes_array([e.label for e in result.epochs], width=96))
    _write_array(epochs, "start_frame", np.asarray([e.start_frame for e in result.epochs], dtype=np.int64))
    _write_array(epochs, "end_frame", np.asarray([e.end_frame for e in result.epochs], dtype=np.int64))
    epochs.attrs.update({"row_axis": "epoch", "inherited_from": result.source_bout_response_path})

    refs = component.require_group("references")
    _write_array(refs, "label_bytes", _bytes_array([r.label for r in result.references], width=64))
    _write_array(refs, "kind_bytes", _bytes_array([r.kind for r in result.references], width=16))
    _write_array(refs, "chaser_index", np.asarray([r.chaser_index for r in result.references], dtype=np.int16))
    _write_array(refs, "parent_chaser_index", np.asarray([r.parent_chaser_index for r in result.references], dtype=np.int16))
    _write_array(refs, "rotation_deg", np.asarray([r.rotation_deg for r in result.references], dtype=np.float32))
    refs.attrs.update({"row_axis": "reference", "inherited_from": result.source_bout_response_path})

    events = component.require_group("events")
    for name in ("event_bout_id", "event_epoch_index", "event_start_frame", "event_peak_speed_mm_s",
                 "event_turn_deg", "event_is_high_turn", "event_trace_usable"):
        _write_array(events, name.removeprefix("event_"), getattr(result, name))
    for name in ("event_distance_at_onset_mm", "event_gain_mm", "event_recapture_mm", "event_net_mm"):
        _write_array(events, name.removeprefix("event_"), getattr(result, name))
    events.attrs.update(
        {
            "row_axis": "escape_event",
            "per_reference_axis_order": ["escape_event", "reference"],
            "escape_definition": result.diagnostics.get("escape_definition"),
            "turn_tier": result.diagnostics.get("turn_tier"),
            "pseudoreplication_note": result.diagnostics.get("pseudoreplication_note"),
        }
    )

    rates = component.require_group("rates")
    for name in ("escape_count", "high_turn_escape_count", "ordinary_bout_count", "epoch_duration_s",
                 "valid_duration_s", "tracking_dropout_fraction", "escape_rate_per_min",
                 "escape_rate_per_valid_min", "high_turn_escape_rate_per_valid_min",
                 "high_turn_fraction", "escape_bout_fraction"):
        _write_array(rates, name, getattr(result, name))
    rates.attrs.update({
        "row_axis": "epoch",
        "rate_denominator": result.diagnostics.get("rate_denominator"),
        "turn_tier": result.diagnostics.get("turn_tier"),
        "high_turn_threshold_deg": float(result.high_turn_threshold_deg),
    })

    trigger = component.require_group("trigger")
    for name in ("escape_onset_distance_mm", "ordinary_onset_distance_mm", "proximity_shift_mm"):
        _write_array(trigger, name, getattr(result, name))
    trigger.attrs.update(
        {
            "axis_order": ["epoch", "reference"],
            "proximity_shift_mm": "escape onset - ordinary onset; negative == escapes fire closer to the reference",
        }
    )

    pursuit = component.require_group("pursuit")
    for name in ("pursuit_gain_mm", "pursuit_recapture_mm", "pursuit_net_mm", "pursuit_event_count"):
        _write_array(pursuit, name.removeprefix("pursuit_"), getattr(result, name))
    pursuit.attrs.update(
        {
            "axis_order": ["epoch", "reference"],
            "pursuit_decomposition": result.diagnostics.get("pursuit_decomposition"),
            "static_object_control": result.diagnostics.get("static_object_control"),
            "virtual_control": result.diagnostics.get("virtual_control"),
        }
    )

    traces = component.require_group("traces")
    _write_array(traces, "time_s", result.trace_time_s)
    _write_array(traces, "delta_distance_mm", result.trace_delta_distance_mm)
    _write_array(traces, "event_count", result.trace_event_count)
    traces.attrs.update(
        {
            "axis_order": ["epoch", "reference", "time"],
            "trace_definition": result.diagnostics.get("trace_definition"),
        }
    )

    trials = component.require_group("trials")
    for name in ("trial_id", "trial_ordinal", "trial_start_frame", "trial_end_frame",
                 "trial_trigger_frame", "trial_trigger_distance_mm", "trial_escape_count",
                 "trial_any_escape", "trial_valid_s", "trial_dropout_fraction",
                 "trial_escape_rate_per_valid_s", "trial_wall_distance_at_trigger_mm",
                 "trial_first_escape_latency_s"):
        _write_array(trials, name.removeprefix("trial_") if name != "trial_id" else name,
                     getattr(result, name))
    trials.attrs.update(
        {
            "row_axis": "chase_trial",
            "segmentation_source": result.trial_segmentation_source,
            "habituation_slope_escapes_per_valid_s_per_trial": float(result.habituation_slope_per_trial),
            "habituation_slope_any_escape_per_trial": float(result.habituation_slope_any_escape),
            "trial_definition": result.diagnostics.get("trial_definition"),
            "habituation": result.diagnostics.get("habituation"),
            "wall_confound": result.diagnostics.get("wall_confound"),
        }
    )

    sweep = component.require_group("threshold_sweep")
    _write_array(sweep, "threshold_mm_s", result.threshold_sweep_mm_s)
    _write_array(sweep, "escape_count", result.sweep_escape_count)
    _write_array(sweep, "rate_per_valid_min", result.sweep_rate_per_valid_min)
    sweep.attrs.update(
        {
            "axis_order": ["threshold", "epoch"],
            "measurement_ceiling": result.diagnostics.get("measurement_ceiling"),
        }
    )

    component.attrs.update(
        {
            "schema_id": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "method": METHOD,
            "method_version": METHOD_VERSION,
            "recording_id": result.recording_id,
            "status": result.status,
            "qc_warnings": list(result.qc_warnings),
            "fps": float(result.fps),
            "total_frames": int(result.total_frames),
            "pixels_per_mm_projector": float(result.pixels_per_mm_projector),
            "source_chaser_distance_run": result.chaser_distance_run_name,
            "source_chaser_distance_path": result.chaser_distance_run_path,
            "source_bout_response_component": result.source_bout_response_component,
            "source_bout_response_path": result.source_bout_response_path,
            "summary": result.summary,
            "diagnostics": result.diagnostics,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "git": get_git_info(),
        }
    )
    write_run_lineage_attrs(
        component,
        build_run_lineage_payload(
            run_family=f"{result.chaser_distance_run_path}/{COMPONENT_PARENT_NAME}",
            analysis_schema={"schema_id": SCHEMA_ID, "schema_version": SCHEMA_VERSION, "row_axis": "fish_recording"},
            method=METHOD,
            method_version=METHOD_VERSION,
            source_refs={
                "source_chaser_distance_run": result.chaser_distance_run_name,
                "source_chaser_distance_path": result.chaser_distance_run_path,
                "source_bout_response_component": result.source_bout_response_component,
                "source_bout_response_path": result.source_bout_response_path,
            },
            parameters={
                "peak_speed_threshold_mm_s": float(result.peak_speed_threshold_mm_s),
                "high_turn_threshold_deg": float(result.high_turn_threshold_deg),
                "threshold_sweep_mm_s": [float(v) for v in np.asarray(result.threshold_sweep_mm_s)],
                "pre_window_s": float(result.pre_window_s),
                "post_window_s": float(result.post_window_s),
                "gain_window_s": float(result.gain_window_s),
                "recapture_window_s": float(result.recapture_window_s),
                "max_window_dropout_fraction": float(result.max_window_dropout_fraction),
                "min_events_for_trace": int(result.min_events_for_trace),
            },
            code={"git_commit": get_git_info().get("commit_hash"), "git_dirty": get_git_info().get("is_dirty")},
        ),
        fingerprint_status="best_effort",
        overwrite=True,
    )
    parent.attrs["latest"] = result.component_name
    parent.attrs["latest_complete"] = result.component_name

    if write_png:
        write_png_visualization_artifact(
            run_group,
            name="chaser_escape_events_png",
            payload=render_escape_events_png(result),
            source_component=component_path,
        )
    return component_path


# ----------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Escape events: rate, proximity trigger, pursuit decomposition.")
    p.add_argument("zarr_path", type=Path)
    p.add_argument("--chaser-distance-run", default="latest")
    p.add_argument("--bout-response-component", default="latest")
    p.add_argument("--component-name", default=DEFAULT_COMPONENT_NAME)
    p.add_argument("--peak-speed-threshold-mm-s", type=float, default=DEFAULT_PEAK_SPEED_THRESHOLD_MM_S)
    p.add_argument("--high-turn-threshold-deg", type=float, default=DEFAULT_HIGH_TURN_THRESHOLD_DEG)
    p.add_argument("--pre-window-s", type=float, default=DEFAULT_PRE_WINDOW_S)
    p.add_argument("--post-window-s", type=float, default=DEFAULT_POST_WINDOW_S)
    p.add_argument("--gain-window-s", type=float, default=DEFAULT_GAIN_WINDOW_S)
    p.add_argument("--recapture-window-s", type=float, default=DEFAULT_RECAPTURE_WINDOW_S)
    p.add_argument("--min-events-for-trace", type=int, default=DEFAULT_MIN_EVENTS_FOR_TRACE)
    p.add_argument("--apply", action="store_true", help="write the component (default is a dry run)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-png", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = build_chaser_escape_events_result(
        args.zarr_path,
        chaser_distance_run=args.chaser_distance_run,
        bout_response_component=args.bout_response_component,
        component_name=args.component_name,
        peak_speed_threshold_mm_s=args.peak_speed_threshold_mm_s,
        high_turn_threshold_deg=args.high_turn_threshold_deg,
        pre_window_s=args.pre_window_s,
        post_window_s=args.post_window_s,
        gain_window_s=args.gain_window_s,
        recapture_window_s=args.recapture_window_s,
        min_events_for_trace=args.min_events_for_trace,
    )
    print(f"{result.recording_id}: {result.summary['escape_event_count']} escape events "
          f"(> {result.peak_speed_threshold_mm_s:g} mm/s), status={result.status}")
    for e, epoch in enumerate(result.epochs):
        print(f"   {epoch.label:<20} {int(result.escape_count[e]):>5} events  "
              f"{float(result.escape_rate_per_valid_min[e]):>7.2f}/valid-min  "
              f"dropout {float(result.tracking_dropout_fraction[e]):.3f}")
    for w in result.qc_warnings:
        print(f"   QC: {w}")
    if args.apply:
        path = write_chaser_escape_events_component(
            args.zarr_path, result, overwrite=args.overwrite, write_png=not args.no_png
        )
        print(f"wrote {path}")
    else:
        print("dry run; pass --apply to write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
