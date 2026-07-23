"""Freeze and escape-gain curves: the fish's own policy as a function of chaser distance.

Occupancy histograms cannot answer "what does the fish do when the chaser is close?".
Distance is a *joint* quantity -- it changes when the fish moves and when the chaser moves
-- and in a pursuit assay the chaser is actively driving it. Worse, the chaser's controller
clamps the dot at its own radius, which puts a hard floor in the distance distribution that
looks exactly like a behavioral keep-out zone but is not one.

This module measures the fish's *policy* instead, by decomposing the rate of change of
distance onto the fish->chaser axis:

    fish_radial_velocity      fish's velocity along that axis  (+ = toward the chaser)
    chaser_radial_velocity    chaser's velocity along it       (+ = toward the fish)

and by conditioning behavior on distance:

    immobile_fraction              P(fish is frozen | distance)      -- the FREEZE CURVE
    fish_radial_velocity_moving    E[v_r | fish is swimming, distance] -- the ESCAPE GAIN

A fish with a distinct close-range routine shows a regime change in these curves that no
occupancy statistic can establish.

Tracking caveat, which this module treats as a first-class output rather than a footnote:
a freezing fish is the hardest thing for a detector to hold on to, so the frames that
matter most are exactly the frames most likely to be missing. Dropout is therefore
reported per epoch, with gap structure, and a high-dropout epoch is flagged -- an
immobile_fraction computed on surviving frames is a LOWER BOUND on the true freeze rate.
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
from fisheye.analysis.chaser_distance_io import (
    ChaserDistanceReadSnapshot,
    reject_unsealed_chaser_derived_publication,
)
from fisheye.analysis.track_kinematics_io import load_track_kinematics_track
from fisheye.analysis.chaser_radial_occupancy import (
    ChaserRadialEpoch,
    _apply_settle_trim,
    _get_group_by_path,
    _normalize_float_array,
    _open_root,
    _protocol_position_transition_s,
    _read_epochs,
    _resolve_chaser_distance_run,
    _safe_float,
)
from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.plot_artifacts import write_interactive_plot_spec_artifact, write_png_visualization_artifact
from fisheye.shared.run_lineage_fingerprint import build_run_lineage_payload, write_run_lineage_attrs
from fisheye.shared.system_metadata import get_git_info


SCHEMA_ID = "palette.chaser_response_regimes.v1"
SCHEMA_VERSION = 1
METHOD = "chaser_relative_freeze_and_escape_gain"
METHOD_VERSION = "1"
COMPONENT_PARENT_NAME = "chaser_response_regimes"
DEFAULT_COMPONENT_NAME = "chaser_response_regimes_v1"

FREEZE_CURVE_PNG_ARTIFACT_NAME = "chaser_response_freeze_curve_png"
ESCAPE_GAIN_PNG_ARTIFACT_NAME = "chaser_response_escape_gain_png"
TRACKING_QC_PNG_ARTIFACT_NAME = "chaser_response_tracking_qc_png"
INTERACTIVE_ARTIFACT_NAME = "chaser_response_regimes_interactive"
INTERACTIVE_RENDERER = "palette-chaser-response-regimes-v1"
INTERACTIVE_SPEC_SCHEMA_ID = "palette.chaser_response_regimes.interactive_spec.v1"

# Matches cra_near_field so freeze rates are comparable across components.
DEFAULT_IMMOBILITY_SPEED_THRESHOLD_MM_S = 1.0
# Above this the fish is genuinely swimming; the escape gain is conditioned on it so that
# a mean radial velocity is not dominated by frozen frames sitting at ~0.
DEFAULT_MOVING_SPEED_THRESHOLD_MM_S = 2.0
DEFAULT_DISTANCE_BIN_EDGES_MM = (0.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 20.0, 30.0, 40.0, 60.0)
DEFAULT_NEAR_DISTANCE_MM = 5.0
DEFAULT_FAR_DISTANCE_MM = 20.0
DEFAULT_DROPOUT_WARN_FRACTION = 0.10
DEFAULT_LONG_GAP_WARN_S = 5.0
# The chaser controller clamps the dot when its edge reaches the fish, so the minimum
# attainable distance is the dot radius. Within this tolerance of it, say so.
DEFAULT_CLAMP_TOLERANCE_MM = 0.25
# A band mean over a handful of frames is noise wearing the costume of an effect. The fish
# is often almost never near a given object, so require real support before reporting one.
DEFAULT_MIN_BAND_FRAMES = 50
# Same problem one level down: a per-bin curve point over a handful of frames swings wildly
# and reads as structure. Points below this are NaN; frame_count still carries the support.
DEFAULT_MIN_BIN_FRAMES = 20
IMMOBILITY_SIGNAL_MODES = (
    "verified_track_motion",
    "raw_centroid_explicit",
)


@dataclass(frozen=True)
class TrackingQC:
    epoch_label: str
    chaser_index: int
    total_frames: int
    analyzable_pairs: int
    dropout_fraction: float
    gap_count: int
    longest_gap_frames: int
    longest_gap_s: float
    dropped_frames: int


@dataclass(frozen=True)
class ChaserResponseRegimesResult:
    zarr_path: str
    recording_id: str
    component_name: str
    chaser_distance_run_name: str
    chaser_distance_run_path: str
    source_stimulus_run: str | None
    source_stimulus_path: str | None
    source_track_motion_authority: dict[str, Any] | None
    fps: float
    total_frames: int
    pixels_per_mm_projector: float
    coordinate_frame: str
    coordinate_origin: str
    immobility_speed_threshold_mm_s: float
    moving_speed_threshold_mm_s: float
    near_distance_mm: float
    far_distance_mm: float
    settle_trim_s: float
    distance_bin_edges_mm: np.ndarray
    distance_bin_centers_mm: np.ndarray
    epochs: tuple[ChaserRadialEpoch, ...]
    chaser_indices: np.ndarray
    chaser_behavior_labels: tuple[str, ...]
    chaser_radius_mm: np.ndarray
    # (epoch, chaser, distance_bin)
    frame_count: np.ndarray
    immobile_fraction: np.ndarray
    moving_fraction: np.ndarray
    mean_speed_mm_s: np.ndarray
    median_speed_mm_s: np.ndarray
    fish_radial_velocity_mm_s: np.ndarray
    fish_radial_velocity_moving_mm_s: np.ndarray
    fraction_moving_away: np.ndarray
    chaser_radial_velocity_mm_s: np.ndarray
    fish_separation_rate_mm_s: np.ndarray
    chaser_separation_rate_mm_s: np.ndarray
    net_separation_rate_mm_s: np.ndarray
    moving_frame_count: np.ndarray
    # (epoch, chaser)
    min_distance_mm: np.ndarray
    distance_floor_is_clamp: np.ndarray
    freeze_index: np.ndarray
    immobile_fraction_near: np.ndarray
    immobile_fraction_far: np.ndarray
    escape_gain_near_mm_s: np.ndarray
    escape_gain_far_mm_s: np.ndarray
    approach_fraction: np.ndarray
    tracking_qc: tuple[TrackingQC, ...]
    status: str
    qc_warnings: tuple[str, ...]
    summary: dict[str, Any]
    diagnostics: dict[str, Any]


def _protocol_chaser_radii_mm(
    root: zarr.Group,
    distance: ChaserDistanceReadSnapshot,
    *,
    n_chasers: int,
) -> np.ndarray:
    """Per-chaser dot radius from the protocol; the distance floor lives here, not in the fish."""

    out = np.full(int(n_chasers), np.nan, dtype=np.float32)
    source_path = str(distance.source_stimulus_path or "").strip()
    stim_group = _get_group_by_path(root, source_path) if source_path else None
    if stim_group is None:
        return out
    raw = stim_group.attrs.get("protocol_json")
    if raw is None:
        return out
    try:
        payload = json.loads(raw) if isinstance(raw, str) else raw
        chasers = payload["steps"][0]["parameters"]["chasers"]
    except (TypeError, ValueError, KeyError, IndexError):
        return out
    for idx in range(min(int(n_chasers), len(chasers))):
        entry = chasers[idx]
        if not isinstance(entry, dict):
            continue
        radius = _safe_float(entry.get("radius_mm"))
        if math.isfinite(radius) and radius > 0:
            out[idx] = np.float32(radius)
    return out


def _gap_structure(valid: np.ndarray) -> tuple[int, int, int]:
    """(gap_count, longest_gap_frames, dropped_frames) over a boolean validity mask."""

    invalid = (~np.asarray(valid, dtype=bool)).astype(np.int8)
    if invalid.size == 0:
        return 0, 0, 0
    padded = np.concatenate(([0], invalid, [0]))
    deltas = np.diff(padded)
    starts = np.where(deltas == 1)[0]
    ends = np.where(deltas == -1)[0]
    lengths = ends - starts
    if lengths.size == 0:
        return 0, 0, 0
    return int(lengths.size), int(lengths.max()), int(lengths.sum())


def _weighted_nanmean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if finite.size else math.nan


def _band_mean(
    curve: np.ndarray,
    counts: np.ndarray,
    centers: np.ndarray,
    *,
    lo: float,
    hi: float,
    min_frames: float,
) -> float:
    """Count-weighted mean of a per-bin curve over the distance band [lo, hi).

    Returns NaN when the band holds fewer than min_frames samples: the fish is frequently
    almost never near a given object, and a band mean over a dozen frames reads as a large
    effect while being pure noise.
    """

    mask = (centers >= float(lo)) & (centers < float(hi))
    values = np.asarray(curve, dtype=np.float64)[mask]
    weights = np.asarray(counts, dtype=np.float64)[mask]
    usable = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(usable):
        return math.nan
    total = float(np.sum(weights[usable]))
    if total < float(min_frames):
        return math.nan
    return float(np.sum(values[usable] * weights[usable]) / total)


def _compute_regime_arrays(
    *,
    epochs: Sequence[ChaserRadialEpoch],
    chaser_indices: np.ndarray,
    fish_xy: np.ndarray,
    chaser_xy: np.ndarray,
    fish_valid: np.ndarray,
    chaser_valid: np.ndarray,
    distance_mm: np.ndarray,
    fps: float,
    pixels_per_mm: float,
    distance_bin_edges_mm: np.ndarray,
    immobility_speed_threshold_mm_s: float,
    moving_speed_threshold_mm_s: float,
    near_distance_mm: float,
    far_distance_mm: float,
    chaser_radius_mm: np.ndarray,
    dropout_warn_fraction: float,
    long_gap_warn_s: float,
    clamp_tolerance_mm: float,
    min_band_frames: float,
    min_bin_frames: float,
    immobility_speed: Optional[np.ndarray] = None,
) -> tuple[dict[str, np.ndarray], list[TrackingQC], dict[str, Any], tuple[str, ...]]:
    n_epochs = len(epochs)
    n_chasers = int(chaser_indices.shape[0])
    edges = np.asarray(distance_bin_edges_mm, dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    n_bins = int(centers.shape[0])
    safe_fps = float(fps) if math.isfinite(float(fps)) and float(fps) > 0 else 1.0

    shape3 = (n_epochs, n_chasers, n_bins)
    shape2 = (n_epochs, n_chasers)
    frame_count = np.zeros(shape3, dtype=np.int64)
    moving_count = np.zeros(shape3, dtype=np.int64)
    immobile_fraction = np.full(shape3, np.nan, dtype=np.float32)
    moving_fraction = np.full(shape3, np.nan, dtype=np.float32)
    mean_speed = np.full(shape3, np.nan, dtype=np.float32)
    median_speed = np.full(shape3, np.nan, dtype=np.float32)
    fish_vr = np.full(shape3, np.nan, dtype=np.float32)
    fish_vr_moving = np.full(shape3, np.nan, dtype=np.float32)
    fraction_away = np.full(shape3, np.nan, dtype=np.float32)
    chaser_vr = np.full(shape3, np.nan, dtype=np.float32)
    fish_sep = np.full(shape3, np.nan, dtype=np.float32)
    chaser_sep = np.full(shape3, np.nan, dtype=np.float32)
    net_sep = np.full(shape3, np.nan, dtype=np.float32)

    min_distance = np.full(shape2, np.nan, dtype=np.float32)
    floor_is_clamp = np.zeros(shape2, dtype=bool)
    freeze_index = np.full(shape2, np.nan, dtype=np.float32)
    immobile_near = np.full(shape2, np.nan, dtype=np.float32)
    immobile_far = np.full(shape2, np.nan, dtype=np.float32)
    escape_near = np.full(shape2, np.nan, dtype=np.float32)
    escape_far = np.full(shape2, np.nan, dtype=np.float32)
    approach_fraction = np.full(shape2, np.nan, dtype=np.float32)

    tracking: list[TrackingQC] = []
    warnings: list[str] = []

    for e_idx, epoch in enumerate(epochs):
        slc = slice(epoch.start_frame, epoch.end_frame + 1)
        f_xy = np.asarray(fish_xy[slc], dtype=np.float64) / float(pixels_per_mm)
        f_ok = np.asarray(fish_valid[slc], dtype=bool) & np.isfinite(f_xy).all(axis=1)
        n_frames = int(f_xy.shape[0])

        for c_idx in range(n_chasers):
            chaser_id = int(chaser_indices[c_idx])
            c_xy = np.asarray(chaser_xy[slc, c_idx, :], dtype=np.float64) / float(pixels_per_mm)
            c_ok = np.asarray(chaser_valid[slc, c_idx], dtype=bool) & np.isfinite(c_xy).all(axis=1)
            joint = f_ok & c_ok

            gap_count, longest_gap, dropped = _gap_structure(joint)
            dropout = float(dropped) / float(n_frames) if n_frames else math.nan

            if n_frames < 2:
                tracking.append(
                    TrackingQC(epoch.label, chaser_id, n_frames, 0, dropout, gap_count, longest_gap,
                               longest_gap / safe_fps, dropped)
                )
                continue

            # Velocity needs the fish AND chaser resolved on both sides of the step, so the
            # rate is a real measurement rather than an interpolation across a gap.
            pair = joint[:-1] & joint[1:]
            v_fish = (f_xy[1:] - f_xy[:-1]) * safe_fps
            v_chaser = (c_xy[1:] - c_xy[:-1]) * safe_fps
            rel = c_xy[:-1] - f_xy[:-1]
            dist = np.asarray(distance_mm[slc, c_idx], dtype=np.float64)[:-1]
            norm = np.hypot(rel[:, 0], rel[:, 1])
            unit = rel / np.maximum(norm, 1e-9)[:, None]

            vr_fish = np.sum(v_fish * unit, axis=1)      # + = fish toward chaser
            vr_chaser = -np.sum(v_chaser * unit, axis=1)  # + = chaser toward fish
            speed = np.hypot(v_fish[:, 0], v_fish[:, 1])
            # Immobile/moving are decided on the smoothed signal when available -- raw centroid
            # jitter (~1.6 mm/s floor) straddles the 1 mm/s threshold. Computed speed still feeds
            # the reported mean/median. Canonical track speeds are transition metrics anchored
            # to the destination acquisition frame, so transition t->t+1 uses sample t+1.
            im_speed = (
                np.asarray(immobility_speed[slc], dtype=np.float64)[1:]
                if immobility_speed is not None else speed
            )

            usable = pair & np.isfinite(dist) & np.isfinite(vr_fish) & np.isfinite(vr_chaser)
            if immobility_speed is not None:
                usable = usable & np.isfinite(im_speed)
            n_pairs = int(np.count_nonzero(usable))
            tracking.append(
                TrackingQC(
                    epoch_label=epoch.label,
                    chaser_index=chaser_id,
                    total_frames=n_frames,
                    analyzable_pairs=n_pairs,
                    dropout_fraction=dropout,
                    gap_count=gap_count,
                    longest_gap_frames=longest_gap,
                    longest_gap_s=float(longest_gap) / safe_fps,
                    dropped_frames=dropped,
                )
            )
            if math.isfinite(dropout) and dropout > float(dropout_warn_fraction):
                warnings.append(f"high_tracking_dropout:{epoch.label}:chaser{chaser_id}")
            if float(longest_gap) / safe_fps > float(long_gap_warn_s):
                warnings.append(f"long_tracking_gap:{epoch.label}:chaser{chaser_id}")
            if n_pairs == 0:
                continue

            d_use = dist[usable]
            min_distance[e_idx, c_idx] = float(np.min(d_use))
            radius = float(chaser_radius_mm[c_idx]) if c_idx < chaser_radius_mm.shape[0] else math.nan
            if math.isfinite(radius) and abs(float(min_distance[e_idx, c_idx]) - radius) <= float(clamp_tolerance_mm):
                floor_is_clamp[e_idx, c_idx] = True
                warnings.append(f"distance_floor_at_chaser_radius:{epoch.label}:chaser{chaser_id}")

            bin_idx = np.digitize(d_use, edges) - 1
            vr_f = vr_fish[usable]
            vr_c = vr_chaser[usable]
            sp = speed[usable]
            im_sp = im_speed[usable]
            immobile = im_sp < float(immobility_speed_threshold_mm_s)
            moving = im_sp > float(moving_speed_threshold_mm_s)

            if np.any(moving):
                approach_fraction[e_idx, c_idx] = float(np.mean(vr_f[moving] > 0))

            for b in range(n_bins):
                sel = bin_idx == b
                n_sel = int(np.count_nonzero(sel))
                if n_sel == 0:
                    continue
                frame_count[e_idx, c_idx, b] = n_sel
                immobile_fraction[e_idx, c_idx, b] = float(np.mean(immobile[sel]))
                moving_fraction[e_idx, c_idx, b] = float(np.mean(moving[sel]))
                mean_speed[e_idx, c_idx, b] = float(np.mean(sp[sel]))
                median_speed[e_idx, c_idx, b] = float(np.median(sp[sel]))
                fish_vr[e_idx, c_idx, b] = float(np.mean(vr_f[sel]))
                chaser_vr[e_idx, c_idx, b] = float(np.mean(vr_c[sel]))
                # Contributions to d(distance)/dt: moving toward shrinks the gap.
                fish_sep[e_idx, c_idx, b] = float(-np.mean(vr_f[sel]))
                chaser_sep[e_idx, c_idx, b] = float(-np.mean(vr_c[sel]))
                net_sep[e_idx, c_idx, b] = float(-np.mean(vr_f[sel]) - np.mean(vr_c[sel]))
                sel_moving = sel & moving
                n_moving = int(np.count_nonzero(sel_moving))
                moving_count[e_idx, c_idx, b] = n_moving
                if n_moving:
                    fish_vr_moving[e_idx, c_idx, b] = float(np.mean(vr_f[sel_moving]))
                    fraction_away[e_idx, c_idx, b] = float(np.mean(vr_f[sel_moving] < 0))

            # Suppress under-supported bins before anything downstream reads them.
            thin = frame_count[e_idx, c_idx, :] < int(min_bin_frames)
            for arr in (
                immobile_fraction, moving_fraction, mean_speed, median_speed,
                fish_vr, chaser_vr, fish_sep, chaser_sep, net_sep,
            ):
                arr[e_idx, c_idx, thin] = np.nan
            thin_moving = moving_count[e_idx, c_idx, :] < int(min_bin_frames)
            for arr in (fish_vr_moving, fraction_away):
                arr[e_idx, c_idx, thin_moving] = np.nan

            immobile_near[e_idx, c_idx] = _band_mean(
                immobile_fraction[e_idx, c_idx, :], frame_count[e_idx, c_idx, :], centers,
                lo=0.0, hi=float(near_distance_mm), min_frames=float(min_band_frames),
            )
            immobile_far[e_idx, c_idx] = _band_mean(
                immobile_fraction[e_idx, c_idx, :], frame_count[e_idx, c_idx, :], centers,
                lo=float(far_distance_mm), hi=float(edges[-1]), min_frames=float(min_band_frames),
            )
            if math.isfinite(immobile_near[e_idx, c_idx]) and math.isfinite(immobile_far[e_idx, c_idx]):
                freeze_index[e_idx, c_idx] = float(immobile_near[e_idx, c_idx] - immobile_far[e_idx, c_idx])
            escape_near[e_idx, c_idx] = _band_mean(
                fish_vr_moving[e_idx, c_idx, :], moving_count[e_idx, c_idx, :], centers,
                lo=0.0, hi=float(near_distance_mm), min_frames=float(min_band_frames),
            )
            escape_far[e_idx, c_idx] = _band_mean(
                fish_vr_moving[e_idx, c_idx, :], moving_count[e_idx, c_idx, :], centers,
                lo=float(far_distance_mm), hi=float(edges[-1]), min_frames=float(min_band_frames),
            )

    arrays = {
        "frame_count": frame_count,
        "moving_frame_count": moving_count,
        "immobile_fraction": immobile_fraction,
        "moving_fraction": moving_fraction,
        "mean_speed_mm_s": mean_speed,
        "median_speed_mm_s": median_speed,
        "fish_radial_velocity_mm_s": fish_vr,
        "fish_radial_velocity_moving_mm_s": fish_vr_moving,
        "fraction_moving_away": fraction_away,
        "chaser_radial_velocity_mm_s": chaser_vr,
        "fish_separation_rate_mm_s": fish_sep,
        "chaser_separation_rate_mm_s": chaser_sep,
        "net_separation_rate_mm_s": net_sep,
        "min_distance_mm": min_distance,
        "distance_floor_is_clamp": floor_is_clamp,
        "freeze_index": freeze_index,
        "immobile_fraction_near": immobile_near,
        "immobile_fraction_far": immobile_far,
        "escape_gain_near_mm_s": escape_near,
        "escape_gain_far_mm_s": escape_far,
        "approach_fraction": approach_fraction,
    }
    diagnostics = {
        "radial_velocity_sign_convention": "positive = toward the other agent",
        "separation_rate_definition": "contribution to d(distance)/dt; positive = opens the gap",
        "escape_gain_definition": (
            "mean fish radial velocity conditioned on the fish actually swimming "
            "(speed > moving_speed_threshold_mm_s); unconditioned means are dominated by frozen frames"
        ),
        "freeze_index_definition": "immobile_fraction_near - immobile_fraction_far; positive = freezes more when the chaser is close",
        "distance_floor_policy": (
            "the chaser controller clamps the dot when its edge reaches the fish, so the minimum "
            "attainable distance is the dot radius; distance_floor_is_clamp marks where the observed "
            "floor matches it, and that hole is stimulus geometry, not behavior"
        ),
        "dropout_caveat": (
            "a freezing fish is the hardest target for the detector, so immobile_fraction computed on "
            "surviving frames is a LOWER BOUND on the true freeze rate; read it next to tracking_qc"
        ),
        "velocity_estimator": "adjacent-frame centroid difference, both agents resolved on both sides of the step",
        "immobility_speed_threshold_mm_s": float(immobility_speed_threshold_mm_s),
        "moving_speed_threshold_mm_s": float(moving_speed_threshold_mm_s),
        "min_band_frames": float(min_band_frames),
        "min_bin_frames": float(min_bin_frames),
        "bin_undersampling_policy": "per-bin curve values are NaN where frame_count < min_bin_frames; frame_count is persisted raw so the support is visible",
        "band_undersampling_policy": "near/far band scalars are NaN when the band holds fewer than min_band_frames samples",
        "qc_warning_count": len(warnings),
    }
    return arrays, tracking, diagnostics, tuple(dict.fromkeys(warnings))


def _build_summary(
    *,
    recording_id: str,
    epochs: Sequence[ChaserRadialEpoch],
    chaser_indices: np.ndarray,
    chaser_behavior_labels: Sequence[str],
    arrays: dict[str, np.ndarray],
    tracking: Sequence[TrackingQC],
) -> dict[str, Any]:
    summary: dict[str, Any] = {"recording_id": recording_id}
    qc_by_key = {(t.epoch_label, t.chaser_index): t for t in tracking}
    for e_idx, epoch in enumerate(epochs):
        for c_idx in range(int(chaser_indices.shape[0])):
            chaser_id = int(chaser_indices[c_idx])
            label = chaser_behavior_labels[c_idx] if c_idx < len(chaser_behavior_labels) else ""
            stem = f"{epoch.label}_{label}" if label else f"{epoch.label}_chaser{chaser_id}"
            for key, name in (
                ("freeze_index", "freeze_index"),
                ("immobile_fraction_near", "immobile_near"),
                ("immobile_fraction_far", "immobile_far"),
                ("escape_gain_near_mm_s", "escape_gain_near"),
                ("escape_gain_far_mm_s", "escape_gain_far"),
                ("approach_fraction", "approach_fraction"),
                ("min_distance_mm", "min_distance_mm"),
            ):
                value = float(arrays[key][e_idx, c_idx])
                summary[f"{name}_{stem}"] = value if math.isfinite(value) else None
            summary[f"distance_floor_is_clamp_{stem}"] = bool(arrays["distance_floor_is_clamp"][e_idx, c_idx])
            qc = qc_by_key.get((epoch.label, chaser_id))
            if qc is not None:
                summary[f"tracking_dropout_{stem}"] = (
                    float(qc.dropout_fraction) if math.isfinite(qc.dropout_fraction) else None
                )
                summary[f"longest_gap_s_{stem}"] = float(qc.longest_gap_s)
                summary[f"analyzable_pairs_{stem}"] = int(qc.analyzable_pairs)
    return summary


def _load_smoothed_immobility_speed(
    root: zarr.Group,
    total_frames: int,
) -> tuple[np.ndarray, str, dict[str, Any]]:
    """Dense per-frame smoothed physical speed (mm/s) from the offline track_kinematics run,
    aligned to the chaser-distance frame axis, for the immobile/moving classification.

    Thresholding the RAW centroid speed is unreliable: its jitter noise floor (~1.6 mm/s)
    straddles the 1 mm/s immobility threshold, so "immobile fraction" partly measures tracking
    noise rather than stillness. ``speed_smoothed_mm`` separates bouts from between-bout stillness
    cleanly (deadbanded to 0 between bouts). This normal scientific path requires a
    freshly verified canonical track-motion publication. Missing or invalid authority
    is an error; raw-centroid analysis is available only through the caller's explicit
    ``raw_centroid_explicit`` mode.
    """
    frame_count = int(total_frames)
    track = load_track_kinematics_track(
        root,
        run_name="latest",
        scope="offline",
        track_id=0,
        required_speed_levels=("smoothed",),
    )
    if track.source_acquisition_frame_index is None:
        raise ValueError(
            f"Verified track motion {track.track_path} has no acquisition-frame identity."
        )
    if track.sample_valid is None or track.transition_valid is None:
        raise ValueError(
            f"Verified track motion {track.track_path} lacks sample/transition validity."
        )
    frames = np.asarray(
        track.source_acquisition_frame_index,
        dtype=np.int64,
    ).reshape(-1)
    speed = np.asarray(
        track.speed_mm_by_level["smoothed"],
        dtype=np.float64,
    ).reshape(-1)
    sample_valid = np.asarray(track.sample_valid, dtype=bool).reshape(-1)
    transition_valid = np.asarray(track.transition_valid, dtype=bool).reshape(-1)
    if not (
        frames.shape == speed.shape == sample_valid.shape == transition_valid.shape
    ):
        raise ValueError(
            f"Verified track motion {track.track_path} has inconsistent frame, "
            "speed, and validity lengths."
        )
    if np.any(frames < 0) or np.any(frames >= frame_count):
        raise ValueError(
            f"Verified acquisition-frame identities in {track.track_path} exceed "
            f"the chaser-distance extent [0, {frame_count})."
        )
    if np.unique(frames).shape[0] != frames.shape[0]:
        raise ValueError(
            f"Verified track motion {track.track_path} repeats acquisition-frame identities."
        )
    dense = np.full(frame_count, np.nan, dtype=np.float64)
    usable = sample_valid & transition_valid & np.isfinite(speed)
    dense[frames[usable]] = speed[usable]
    authority = track.authority_record()
    return (
        dense,
        "track_motion.movement/speed/smoothed/mm",
        authority,
    )


def build_chaser_response_regimes_result(
    zarr_path: Path,
    *,
    chaser_distance_run: str = "latest",
    component_name: str = DEFAULT_COMPONENT_NAME,
    distance_bin_edges_mm: Sequence[float] = DEFAULT_DISTANCE_BIN_EDGES_MM,
    immobility_speed_threshold_mm_s: float = DEFAULT_IMMOBILITY_SPEED_THRESHOLD_MM_S,
    moving_speed_threshold_mm_s: float = DEFAULT_MOVING_SPEED_THRESHOLD_MM_S,
    near_distance_mm: float = DEFAULT_NEAR_DISTANCE_MM,
    far_distance_mm: float = DEFAULT_FAR_DISTANCE_MM,
    dropout_warn_fraction: float = DEFAULT_DROPOUT_WARN_FRACTION,
    long_gap_warn_s: float = DEFAULT_LONG_GAP_WARN_S,
    clamp_tolerance_mm: float = DEFAULT_CLAMP_TOLERANCE_MM,
    min_band_frames: float = DEFAULT_MIN_BAND_FRAMES,
    min_bin_frames: float = DEFAULT_MIN_BIN_FRAMES,
    settle_trim_s: float | None = None,
    motion_spread_threshold_mm: float = 1.0,
    immobility_signal_mode: str = "verified_track_motion",
) -> ChaserResponseRegimesResult:
    if float(moving_speed_threshold_mm_s) < float(immobility_speed_threshold_mm_s):
        raise ValueError("moving_speed_threshold_mm_s must be >= immobility_speed_threshold_mm_s.")
    if not float(far_distance_mm) > float(near_distance_mm):
        raise ValueError("far_distance_mm must be greater than near_distance_mm.")
    normalized_immobility_mode = str(immobility_signal_mode).strip()
    if normalized_immobility_mode not in IMMOBILITY_SIGNAL_MODES:
        raise ValueError(
            f"Unsupported immobility_signal_mode {immobility_signal_mode!r}; "
            f"expected one of: {', '.join(IMMOBILITY_SIGNAL_MODES)}."
        )

    root = _open_root(zarr_path, mode="r")
    distance, run_name, run_path = _resolve_chaser_distance_run(
        root,
        chaser_distance_run,
    )
    distance.require_stimulus_protocol_authority(
        "chaser radii and position-transition timing"
    )

    coordinate_frame = distance.coordinate_space_id
    coordinate_origin = distance.coordinate_origin
    if coordinate_frame != "arena_relative_canvas_px":
        raise ValueError(
            "Chaser response regimes requires typed "
            f"space_id='arena_relative_canvas_px'; got {coordinate_frame!r}."
        )
    pixels_per_mm = float(distance.pixels_per_mm_projector)

    fish_xy = np.asarray(distance.fish_centroid_arena_xy, dtype=np.float32)
    chaser_xy = np.asarray(distance.chaser_arena_xy, dtype=np.float32)
    fish_valid = np.asarray(distance.fish_valid, dtype=bool)
    chaser_valid = np.asarray(distance.chaser_valid, dtype=bool)
    distance_mm = np.asarray(distance.distance_mm, dtype=np.float32)
    chaser_indices = np.asarray(distance.chaser_index, dtype=np.int64).reshape(-1)
    behavior_labels: tuple[str, ...] = ()

    total_frames = int(fish_xy.shape[0])
    if total_frames != distance.total_frames:
        raise ValueError("Typed chaser-distance frame extent is inconsistent.")
    if chaser_xy.ndim != 3 or chaser_xy.shape[0] != total_frames or chaser_xy.shape[2] != 2:
        raise ValueError("positions/chaser_arena_xy must have shape (frame, chaser, xy).")
    if distance_mm.shape != (total_frames, chaser_xy.shape[1]):
        raise ValueError("distances/distance_mm must have shape (frame, chaser).")
    if chaser_indices.shape[0] != distance_mm.shape[1]:
        raise ValueError("chasers/chaser_index length does not match distance columns.")

    fps = float(distance.fps)
    epochs = _read_epochs(distance, total_frames=total_frames)
    resolved_settle_trim_s = (
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
        settle_trim_s=float(resolved_settle_trim_s),
        motion_spread_threshold_mm=float(motion_spread_threshold_mm),
    )

    edges = _normalize_float_array(distance_bin_edges_mm, name="distance_bin_edges_mm")
    if edges.shape[0] < 2 or not np.all(np.diff(edges.astype(np.float64)) > 0):
        raise ValueError("distance_bin_edges_mm must contain at least two increasing values.")
    centers = ((edges[:-1] + edges[1:]) / 2.0).astype(np.float32)
    radii = _protocol_chaser_radii_mm(
        root,
        distance,
        n_chasers=int(chaser_indices.shape[0]),
    )

    source_track_motion_authority: dict[str, Any] | None
    if normalized_immobility_mode == "verified_track_motion":
        (
            immobility_speed,
            immobility_signal_source,
            source_track_motion_authority,
        ) = _load_smoothed_immobility_speed(root, total_frames)
    else:
        immobility_speed = None
        immobility_signal_source = "raw_centroid_explicit"
        source_track_motion_authority = None

    arrays, tracking, diagnostics, qc_warnings = _compute_regime_arrays(
        epochs=epochs,
        chaser_indices=chaser_indices,
        fish_xy=fish_xy,
        chaser_xy=chaser_xy,
        fish_valid=fish_valid,
        chaser_valid=chaser_valid,
        distance_mm=distance_mm,
        fps=float(fps),
        pixels_per_mm=float(pixels_per_mm),
        distance_bin_edges_mm=edges,
        immobility_speed_threshold_mm_s=float(immobility_speed_threshold_mm_s),
        moving_speed_threshold_mm_s=float(moving_speed_threshold_mm_s),
        near_distance_mm=float(near_distance_mm),
        far_distance_mm=float(far_distance_mm),
        chaser_radius_mm=radii,
        dropout_warn_fraction=float(dropout_warn_fraction),
        long_gap_warn_s=float(long_gap_warn_s),
        clamp_tolerance_mm=float(clamp_tolerance_mm),
        min_band_frames=float(min_band_frames),
        min_bin_frames=float(min_bin_frames),
        immobility_speed=immobility_speed,
    )
    diagnostics = dict(diagnostics)
    diagnostics["immobility_signal_source"] = immobility_signal_source
    diagnostics["immobility_signal_mode"] = normalized_immobility_mode
    if normalized_immobility_mode == "raw_centroid_explicit":
        qc_warnings = tuple(qc_warnings) + (
            "immobility_signal_explicit_raw_centroid",
        )

    recording_id = distance.recording_id
    summary = _build_summary(
        recording_id=recording_id,
        epochs=epochs,
        chaser_indices=chaser_indices,
        chaser_behavior_labels=behavior_labels,
        arrays=arrays,
        tracking=tracking,
    )
    diagnostics["chaser_radius_mm"] = [None if not math.isfinite(float(v)) else float(v) for v in radii]

    return ChaserResponseRegimesResult(
        zarr_path=str(zarr_path),
        recording_id=recording_id,
        component_name=str(component_name),
        chaser_distance_run_name=run_name,
        chaser_distance_run_path=run_path,
        source_stimulus_run=distance.source_stimulus_run,
        source_stimulus_path=distance.source_stimulus_path,
        source_track_motion_authority=source_track_motion_authority,
        fps=float(fps),
        total_frames=int(total_frames),
        pixels_per_mm_projector=float(pixels_per_mm),
        coordinate_frame=coordinate_frame,
        coordinate_origin=coordinate_origin,
        immobility_speed_threshold_mm_s=float(immobility_speed_threshold_mm_s),
        moving_speed_threshold_mm_s=float(moving_speed_threshold_mm_s),
        near_distance_mm=float(near_distance_mm),
        far_distance_mm=float(far_distance_mm),
        settle_trim_s=float(resolved_settle_trim_s),
        distance_bin_edges_mm=edges.astype(np.float32),
        distance_bin_centers_mm=centers,
        epochs=epochs,
        chaser_indices=chaser_indices.astype(np.int16),
        chaser_behavior_labels=behavior_labels,
        chaser_radius_mm=radii,
        frame_count=arrays["frame_count"],
        immobile_fraction=arrays["immobile_fraction"],
        moving_fraction=arrays["moving_fraction"],
        mean_speed_mm_s=arrays["mean_speed_mm_s"],
        median_speed_mm_s=arrays["median_speed_mm_s"],
        fish_radial_velocity_mm_s=arrays["fish_radial_velocity_mm_s"],
        fish_radial_velocity_moving_mm_s=arrays["fish_radial_velocity_moving_mm_s"],
        fraction_moving_away=arrays["fraction_moving_away"],
        chaser_radial_velocity_mm_s=arrays["chaser_radial_velocity_mm_s"],
        fish_separation_rate_mm_s=arrays["fish_separation_rate_mm_s"],
        chaser_separation_rate_mm_s=arrays["chaser_separation_rate_mm_s"],
        net_separation_rate_mm_s=arrays["net_separation_rate_mm_s"],
        moving_frame_count=arrays["moving_frame_count"],
        min_distance_mm=arrays["min_distance_mm"],
        distance_floor_is_clamp=arrays["distance_floor_is_clamp"],
        freeze_index=arrays["freeze_index"],
        immobile_fraction_near=arrays["immobile_fraction_near"],
        immobile_fraction_far=arrays["immobile_fraction_far"],
        escape_gain_near_mm_s=arrays["escape_gain_near_mm_s"],
        escape_gain_far_mm_s=arrays["escape_gain_far_mm_s"],
        approach_fraction=arrays["approach_fraction"],
        tracking_qc=tuple(tracking),
        status="computed",
        qc_warnings=qc_warnings,
        summary=summary,
        diagnostics=diagnostics,
    )


def _chaser_label(result: ChaserResponseRegimesResult, c_idx: int) -> str:
    index = int(result.chaser_indices[c_idx])
    if c_idx < len(result.chaser_behavior_labels) and result.chaser_behavior_labels[c_idx]:
        return f"chaser {index} ({result.chaser_behavior_labels[c_idx]})"
    return f"chaser {index}"


def _clamp_marker(ax, result: ChaserResponseRegimesResult, c_idx: int) -> None:
    radius = float(result.chaser_radius_mm[c_idx]) if c_idx < result.chaser_radius_mm.shape[0] else math.nan
    if math.isfinite(radius) and radius > 0:
        ax.axvspan(0.0, radius, color="#94a3b8", alpha=0.25, lw=0)


def render_freeze_curve_png(result: ChaserResponseRegimesResult, *, dpi: int = 150) -> bytes:
    n = len(result.epochs)
    fig, axes = plt.subplots(1, n, figsize=(4.8 * n, 4.4), sharey=True, constrained_layout=True)
    axes_list = list(np.atleast_1d(axes).ravel())
    for e_idx, (ax, epoch) in enumerate(zip(axes_list, result.epochs)):
        for c_idx in range(int(result.chaser_indices.shape[0])):
            curve = result.immobile_fraction[e_idx, c_idx, :]
            if not np.isfinite(curve).any():
                continue
            _clamp_marker(ax, result, c_idx)
            ax.plot(
                result.distance_bin_centers_mm, curve, marker="o", markersize=3.5, linewidth=1.4,
                label=_chaser_label(result, c_idx),
            )
        ax.set_title(epoch.label.replace("_", " "))
        ax.set_xlabel("distance to chaser (mm)")
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    axes_list[0].set_ylabel(f"P(frozen | distance)   [speed < {result.immobility_speed_threshold_mm_s:g} mm/s]")
    fig.suptitle(
        f"Freeze curve: {result.recording_id}   (shaded = dot radius, distance floor is the clamp)", fontsize=11
    )
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=int(dpi))
    plt.close(fig)
    return buf.getvalue()


def render_escape_gain_png(result: ChaserResponseRegimesResult, *, dpi: int = 150) -> bytes:
    n = len(result.epochs)
    fig, axes = plt.subplots(2, n, figsize=(4.8 * n, 7.4), sharex=True, constrained_layout=True, squeeze=False)
    for e_idx, epoch in enumerate(result.epochs):
        top = axes[0][e_idx]
        bottom = axes[1][e_idx]
        for c_idx in range(int(result.chaser_indices.shape[0])):
            gain = result.fish_radial_velocity_moving_mm_s[e_idx, c_idx, :]
            if np.isfinite(gain).any():
                _clamp_marker(top, result, c_idx)
                top.plot(
                    result.distance_bin_centers_mm, gain, marker="o", markersize=3.5, linewidth=1.4,
                    label=_chaser_label(result, c_idx),
                )
            fish_sep = result.fish_separation_rate_mm_s[e_idx, c_idx, :]
            chaser_sep = result.chaser_separation_rate_mm_s[e_idx, c_idx, :]
            if np.isfinite(fish_sep).any():
                bottom.plot(result.distance_bin_centers_mm, fish_sep, marker="o", markersize=3.0,
                            linewidth=1.3, label=f"fish ({_chaser_label(result, c_idx)})")
                bottom.plot(result.distance_bin_centers_mm, chaser_sep, marker="s", markersize=3.0,
                            linewidth=1.3, linestyle="--", label=f"chaser ({_chaser_label(result, c_idx)})")
        top.axhline(0.0, color="#334155", linestyle="--", linewidth=0.9)
        top.set_title(epoch.label.replace("_", " "))
        top.grid(alpha=0.2)
        top.legend(fontsize=7)
        bottom.axhline(0.0, color="#334155", linestyle="--", linewidth=0.9)
        bottom.set_xlabel("distance to chaser (mm)")
        bottom.grid(alpha=0.2)
        bottom.legend(fontsize=6)
    axes[0][0].set_ylabel("escape gain: fish v_r when swimming (mm/s)\n[negative = away from the dot]")
    axes[1][0].set_ylabel("contribution to d(distance)/dt (mm/s)\n[positive = opens the gap]")
    fig.suptitle(f"Escape gain and who moves: {result.recording_id}", fontsize=11)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=int(dpi))
    plt.close(fig)
    return buf.getvalue()


def render_tracking_qc_png(result: ChaserResponseRegimesResult, *, dpi: int = 150) -> bytes:
    fig, ax = plt.subplots(figsize=(8.0, 4.4), constrained_layout=True)
    labels = [f"{qc.epoch_label}\nchaser {qc.chaser_index}" for qc in result.tracking_qc]
    dropout = [100.0 * qc.dropout_fraction if math.isfinite(qc.dropout_fraction) else 0.0 for qc in result.tracking_qc]
    gaps = [qc.longest_gap_s for qc in result.tracking_qc]
    x = np.arange(len(labels), dtype=np.float64)
    colors = ["#dc2626" if value > 10.0 else "#2563eb" for value in dropout]
    ax.bar(x, dropout, color=colors, width=0.6)
    for xi, (value, gap) in enumerate(zip(dropout, gaps)):
        if value > 0.5:
            ax.text(xi, value + 1.0, f"{value:.1f}%\nmax gap {gap:.0f}s", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("fish tracking dropout (% of epoch frames)")
    ax.axhline(10.0, color="#dc2626", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.grid(alpha=0.2, axis="y")
    ax.set_title(
        "Tracking dropout by epoch\nfreezing fish are the hardest to detect, so freeze rates here are LOWER BOUNDS",
        fontsize=10,
    )
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=int(dpi))
    plt.close(fig)
    return buf.getvalue()


def _source_refs(result: ChaserResponseRegimesResult) -> dict[str, Any]:
    return {
        "source_chaser_distance_run": result.chaser_distance_run_name,
        "source_chaser_distance_path": result.chaser_distance_run_path,
        "source_stimulus_run": result.source_stimulus_run,
        "source_stimulus_path": result.source_stimulus_path,
        "source_track_motion_authority": result.source_track_motion_authority,
    }


def _parameters(result: ChaserResponseRegimesResult) -> dict[str, Any]:
    return {
        "distance_bin_edges_mm": result.distance_bin_edges_mm,
        "immobility_speed_threshold_mm_s": result.immobility_speed_threshold_mm_s,
        "moving_speed_threshold_mm_s": result.moving_speed_threshold_mm_s,
        "near_distance_mm": result.near_distance_mm,
        "far_distance_mm": result.far_distance_mm,
        "min_band_frames": result.diagnostics.get("min_band_frames"),
        "min_bin_frames": result.diagnostics.get("min_bin_frames"),
        "settle_trim_s": result.settle_trim_s,
        "chaser_radius_mm": result.chaser_radius_mm,
        "velocity_estimator": result.diagnostics.get("velocity_estimator"),
        "immobility_signal_mode": result.diagnostics.get(
            "immobility_signal_mode"
        ),
        "immobility_signal_source": result.diagnostics.get(
            "immobility_signal_source"
        ),
    }


def _interactive_spec(result: ChaserResponseRegimesResult, component_path: str) -> dict[str, Any]:
    return {
        "schema_id": INTERACTIVE_SPEC_SCHEMA_ID,
        "schema_version": 1,
        "renderer": INTERACTIVE_RENDERER,
        "recording_id": result.recording_id,
        "component_name": result.component_name,
        "component_path": component_path,
        "source_paths": {
            "component": component_path,
            "config": f"{component_path}/config",
            "epochs": f"{component_path}/epochs",
            "chasers": f"{component_path}/chasers",
            "regimes": f"{component_path}/regimes",
            "per_epoch_chaser": f"{component_path}/per_epoch_chaser",
            "tracking_qc": f"{component_path}/tracking_qc",
            "summary": f"{component_path}/summary",
        },
        "summary": result.summary,
        "parameters": _parameters(result),
        "qc_warnings": list(result.qc_warnings),
    }


def write_chaser_response_regimes_component(
    zarr_path: Path,
    result: ChaserResponseRegimesResult,
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
    component_name = result.component_name
    if component_name in parent:
        if not overwrite:
            raise ValueError(
                f"Chaser response regimes component already exists: "
                f"{result.chaser_distance_run_path}/{COMPONENT_PARENT_NAME}/{component_name}"
            )
        del parent[component_name]
    component = parent.create_group(component_name)
    component_path = f"{result.chaser_distance_run_path}/{COMPONENT_PARENT_NAME}/{component_name}"

    config = component.require_group("config")
    _write_array(config, "distance_bin_edges_mm", result.distance_bin_edges_mm)
    _write_array(config, "distance_bin_centers_mm", result.distance_bin_centers_mm)
    config.attrs.update(
        {
            "immobility_speed_threshold_mm_s": float(result.immobility_speed_threshold_mm_s),
            "moving_speed_threshold_mm_s": float(result.moving_speed_threshold_mm_s),
            "near_distance_mm": float(result.near_distance_mm),
            "far_distance_mm": float(result.far_distance_mm),
            "settle_trim_s": float(result.settle_trim_s),
            "velocity_estimator": result.diagnostics.get("velocity_estimator"),
        }
    )

    epochs = component.require_group("epochs")
    _write_array(epochs, "window_id", np.asarray([e.window_id for e in result.epochs], dtype=np.int32))
    _write_array(epochs, "label_bytes", _bytes_array([e.label for e in result.epochs], width=96))
    _write_array(epochs, "start_frame", np.asarray([e.start_frame for e in result.epochs], dtype=np.int64))
    _write_array(epochs, "end_frame", np.asarray([e.end_frame for e in result.epochs], dtype=np.int64))
    _write_array(epochs, "source_start_frame", np.asarray([e.source_start_frame for e in result.epochs], dtype=np.int64))
    _write_array(
        epochs,
        "settle_excluded_frame_count",
        np.asarray([e.settle_excluded_frame_count for e in result.epochs], dtype=np.int64),
    )
    epochs.attrs.update({"row_axis": "epoch", "settle_trim_s": float(result.settle_trim_s)})

    chasers = component.require_group("chasers")
    _write_array(chasers, "chaser_index", result.chaser_indices)
    _write_array(chasers, "chaser_radius_mm", result.chaser_radius_mm)
    if result.chaser_behavior_labels:
        _write_array(chasers, "behavior_class_label_bytes", _bytes_array(result.chaser_behavior_labels, width=32))
    chasers.attrs.update(
        {
            "row_axis": "chaser",
            "chaser_radius_mm_note": (
                "dot radius from the protocol; the controller clamps the dot when its edge reaches the "
                "fish, so this is the minimum attainable distance -- the hole below it is geometry"
            ),
        }
    )

    regimes = component.require_group("regimes")
    for name in (
        "frame_count",
        "moving_frame_count",
        "immobile_fraction",
        "moving_fraction",
        "mean_speed_mm_s",
        "median_speed_mm_s",
        "fish_radial_velocity_mm_s",
        "fish_radial_velocity_moving_mm_s",
        "fraction_moving_away",
        "chaser_radial_velocity_mm_s",
        "fish_separation_rate_mm_s",
        "chaser_separation_rate_mm_s",
        "net_separation_rate_mm_s",
    ):
        _write_array(regimes, name, getattr(result, name))
    regimes.attrs.update(
        {
            "axis_order": ["epoch", "chaser", "distance_bin"],
            "radial_velocity_sign_convention": result.diagnostics.get("radial_velocity_sign_convention"),
            "separation_rate_definition": result.diagnostics.get("separation_rate_definition"),
            "escape_gain_definition": result.diagnostics.get("escape_gain_definition"),
            "dropout_caveat": result.diagnostics.get("dropout_caveat"),
        }
    )

    per_epoch = component.require_group("per_epoch_chaser")
    for name in (
        "min_distance_mm",
        "distance_floor_is_clamp",
        "freeze_index",
        "immobile_fraction_near",
        "immobile_fraction_far",
        "escape_gain_near_mm_s",
        "escape_gain_far_mm_s",
        "approach_fraction",
    ):
        _write_array(per_epoch, name, getattr(result, name))
    per_epoch.attrs.update(
        {
            "axis_order": ["epoch", "chaser"],
            "freeze_index_definition": result.diagnostics.get("freeze_index_definition"),
            "distance_floor_policy": result.diagnostics.get("distance_floor_policy"),
            "approach_fraction_definition": (
                "fraction of swimming frames whose radial velocity is toward the chaser; "
                "a fish with no approach routine sits well below 0.5"
            ),
        }
    )

    qc = component.require_group("tracking_qc")
    _write_array(qc, "epoch_label_bytes", _bytes_array([t.epoch_label for t in result.tracking_qc], width=96))
    _write_array(qc, "chaser_index", np.asarray([t.chaser_index for t in result.tracking_qc], dtype=np.int16))
    _write_array(qc, "total_frames", np.asarray([t.total_frames for t in result.tracking_qc], dtype=np.int64))
    _write_array(qc, "analyzable_pairs", np.asarray([t.analyzable_pairs for t in result.tracking_qc], dtype=np.int64))
    _write_array(qc, "dropped_frames", np.asarray([t.dropped_frames for t in result.tracking_qc], dtype=np.int64))
    _write_array(
        qc, "dropout_fraction", np.asarray([t.dropout_fraction for t in result.tracking_qc], dtype=np.float32)
    )
    _write_array(qc, "gap_count", np.asarray([t.gap_count for t in result.tracking_qc], dtype=np.int64))
    _write_array(
        qc, "longest_gap_frames", np.asarray([t.longest_gap_frames for t in result.tracking_qc], dtype=np.int64)
    )
    _write_array(qc, "longest_gap_s", np.asarray([t.longest_gap_s for t in result.tracking_qc], dtype=np.float32))
    qc.attrs.update(
        {
            "row_axis": "epoch_chaser",
            "purpose": (
                "a freezing fish is the hardest target for the detector, so the frames that matter most "
                "are the ones most likely to be missing; immobile_fraction is a LOWER BOUND wherever "
                "dropout_fraction is material"
            ),
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
    _write_array(summary_group, "status_bytes", _bytes_array([result.status], width=48))
    _write_array(
        summary_group,
        "diagnostics_json_bytes",
        _bytes_array([json.dumps(json_attr_safe(result.diagnostics), sort_keys=True)], width=4096),
    )
    _write_array(
        summary_group,
        "qc_warnings_json_bytes",
        _bytes_array([json.dumps(list(result.qc_warnings), sort_keys=True)], width=4096),
    )
    summary_group.attrs.update({"row_axis": "fish_recording", "summary": json_attr_safe(result.summary)})

    git = get_git_info(Path(__file__).resolve().parents[3])
    source_refs = _source_refs(result)
    parameters = _parameters(result)
    attrs = {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "method_version": METHOD_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "component_name": component_name,
        "recording_id": result.recording_id,
        "row_axis": "fish_recording",
        "status": result.status,
        "source_refs": source_refs,
        "parameters": parameters,
        "summary": result.summary,
        "qc_warnings": list(result.qc_warnings),
        "diagnostics": result.diagnostics,
        "coordinate_frame": result.coordinate_frame,
        "coordinate_origin": result.coordinate_origin,
        "pixels_per_mm_projector": result.pixels_per_mm_projector,
        "git_commit": git.get("commit_hash"),
        "git_branch": git.get("branch"),
        "git_dirty": git.get("is_dirty"),
        "provenance": {
            "stage": "chaser_response_regimes",
            "created_by": "fisheye.analysis.chaser_response_regimes",
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

    if write_png:
        source_paths = {
            **source_refs,
            "component": component_path,
            "regimes": f"{component_path}/regimes",
            "tracking_qc": f"{component_path}/tracking_qc",
        }
        for artifact_name, renderer, description in (
            (FREEZE_CURVE_PNG_ARTIFACT_NAME, render_freeze_curve_png, "P(frozen | distance to chaser)."),
            (
                ESCAPE_GAIN_PNG_ARTIFACT_NAME,
                render_escape_gain_png,
                "Fish escape gain and the fish/chaser decomposition of d(distance)/dt.",
            ),
            (
                TRACKING_QC_PNG_ARTIFACT_NAME,
                render_tracking_qc_png,
                "Fish tracking dropout by epoch; freeze rates are lower bounds where this is high.",
            ),
        ):
            write_png_visualization_artifact(
                component,
                artifact_name,
                renderer(result),
                description=description,
                created_by="fisheye.analysis.chaser_response_regimes",
                role="analysis_summary",
                source_paths=source_paths,
                source_runs={"chaser_distance_run": result.chaser_distance_run_name},
                parameters=parameters,
                extra_attrs={
                    "chaser_response_regimes_schema_id": SCHEMA_ID,
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
            description="Chaser response regimes interactive plot spec.",
            created_by="fisheye.analysis.chaser_response_regimes",
            renderer=INTERACTIVE_RENDERER,
            artifact_signature=None,
            snapshot_artifact=FREEZE_CURVE_PNG_ARTIFACT_NAME,
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
    return component_path


def _result_payload(result: ChaserResponseRegimesResult, *, applied_path: str | None) -> dict[str, Any]:
    return {
        "schema_id": SCHEMA_ID,
        "zarr_path": result.zarr_path,
        "recording_id": result.recording_id,
        "component_name": result.component_name,
        "applied_path": applied_path,
        "chaser_distance_run": result.chaser_distance_run_name,
        "status": result.status,
        "qc_warnings": list(result.qc_warnings),
        "summary": result.summary,
    }


def _parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze and escape-gain curves vs distance to the chaser.")
    parser.add_argument("zarr_path", type=Path, help="Analysis zarr archive.")
    parser.add_argument("--chaser-distance-run", default="latest")
    parser.add_argument("--component-name", default=DEFAULT_COMPONENT_NAME)
    parser.add_argument(
        "--distance-bin-edges-mm", default=",".join(f"{v:g}" for v in DEFAULT_DISTANCE_BIN_EDGES_MM)
    )
    parser.add_argument("--immobility-speed-threshold-mm-s", type=float, default=DEFAULT_IMMOBILITY_SPEED_THRESHOLD_MM_S)
    parser.add_argument("--moving-speed-threshold-mm-s", type=float, default=DEFAULT_MOVING_SPEED_THRESHOLD_MM_S)
    parser.add_argument("--near-distance-mm", type=float, default=DEFAULT_NEAR_DISTANCE_MM)
    parser.add_argument("--far-distance-mm", type=float, default=DEFAULT_FAR_DISTANCE_MM)
    parser.add_argument("--dropout-warn-fraction", type=float, default=DEFAULT_DROPOUT_WARN_FRACTION)
    parser.add_argument("--long-gap-warn-s", type=float, default=DEFAULT_LONG_GAP_WARN_S)
    parser.add_argument("--min-band-frames", type=float, default=DEFAULT_MIN_BAND_FRAMES)
    parser.add_argument("--min-bin-frames", type=float, default=DEFAULT_MIN_BIN_FRAMES)
    parser.add_argument("--settle-trim-s", type=float, default=None)
    parser.add_argument(
        "--immobility-signal-mode",
        choices=IMMOBILITY_SIGNAL_MODES,
        default="verified_track_motion",
        help=(
            "Use sealed smoothed track motion (default), or explicitly request "
            "the raw centroid transition estimator."
        ),
    )
    parser.add_argument("--apply", action="store_true", help="Write the response-regimes component.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-png", action="store_true")
    parser.add_argument("--no-interactive-spec", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = build_chaser_response_regimes_result(
        Path(args.zarr_path),
        chaser_distance_run=str(args.chaser_distance_run),
        component_name=str(args.component_name),
        distance_bin_edges_mm=_parse_float_list(str(args.distance_bin_edges_mm)),
        immobility_speed_threshold_mm_s=float(args.immobility_speed_threshold_mm_s),
        moving_speed_threshold_mm_s=float(args.moving_speed_threshold_mm_s),
        near_distance_mm=float(args.near_distance_mm),
        far_distance_mm=float(args.far_distance_mm),
        dropout_warn_fraction=float(args.dropout_warn_fraction),
        long_gap_warn_s=float(args.long_gap_warn_s),
        min_band_frames=float(args.min_band_frames),
        min_bin_frames=float(args.min_bin_frames),
        settle_trim_s=None if args.settle_trim_s is None else float(args.settle_trim_s),
        immobility_signal_mode=str(args.immobility_signal_mode),
    )
    applied_path = None
    if args.apply:
        applied_path = write_chaser_response_regimes_component(
            Path(args.zarr_path),
            result,
            overwrite=bool(args.overwrite),
            write_png=not bool(args.no_png),
            write_interactive_spec=not bool(args.no_interactive_spec),
        )
    payload = _result_payload(result, applied_path=applied_path)
    if args.json:
        print(json.dumps(json_attr_safe(payload), indent=2, sort_keys=True))
    else:
        print(f"recording_id\t{result.recording_id}")
        print(f"chaser_distance_run\t{result.chaser_distance_run_name}")
        print(f"status\t{result.status}")
        for e_idx, epoch in enumerate(result.epochs):
            for c_idx in range(int(result.chaser_indices.shape[0])):
                near = float(result.immobile_fraction_near[e_idx, c_idx])
                far = float(result.immobile_fraction_far[e_idx, c_idx])
                fi = float(result.freeze_index[e_idx, c_idx])
                gain_near = float(result.escape_gain_near_mm_s[e_idx, c_idx])
                gain_far = float(result.escape_gain_far_mm_s[e_idx, c_idx])
                approach = float(result.approach_fraction[e_idx, c_idx])
                print(
                    f"regime\t{epoch.label}\tchaser{int(result.chaser_indices[c_idx])}\t"
                    f"frozen_near={near:.3f}\tfrozen_far={far:.3f}\tfreeze_index={fi:+.3f}\t"
                    f"escape_gain_near={gain_near:+.2f}\tescape_gain_far={gain_far:+.2f}\t"
                    f"approach_frac={approach:.3f}"
                )
        for qc in result.tracking_qc:
            if qc.dropout_fraction > 0.01:
                print(
                    f"tracking\t{qc.epoch_label}\tchaser{qc.chaser_index}\t"
                    f"dropout={100*qc.dropout_fraction:.1f}%\tlongest_gap={qc.longest_gap_s:.1f}s\t"
                    f"analyzable_pairs={qc.analyzable_pairs}"
                )
        for warning in result.qc_warnings:
            print(f"qc_warning\t{warning}")
        if applied_path:
            print(f"applied_path\t{applied_path}")
        else:
            print("dry_run\ttrue")
            print("pass --apply to write the response-regimes component")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
