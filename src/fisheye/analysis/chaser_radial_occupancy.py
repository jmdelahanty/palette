"""Compute area-normalized radial ("ring") occupancy around the moving chaser.

The chaser-distance run already stores a 1-D histogram of fish-to-chaser distance
(``epoch_distributions``), but that histogram is normalized by bin *width*, not by
the *area* of the annulus each bin represents. An annulus at radius r has area
~2*pi*r*dr, so even a fish moving uniformly at random produces a histogram that
rises with r and then falls once the ring is clipped by the arena wall. Reading a
peak out of that histogram measures arena geometry, not behavior.

This module computes the corrected quantity: for every frame it takes the annulus
around the chaser's *actual position on that frame*, clips it to the arena, and
accumulates the available area. That gives, per radial bin:

    expected_area_mm2      sum over frames of the ring's available area
    expected_fraction      the geometric null -- where a uniform fish would land
    occupancy_density      observed fish-frames per mm^2 of available area
    selection_index        observed_fraction / expected_fraction (1.0 == chance)

Unlike ``cra_near_field`` -- which handles the *static* objects during the
pre/post phases and re-uses a frozen object position per phase -- this component
tracks the chaser frame by frame, so it is valid during the active chase epoch.

Interpretation warning: when the chaser is *pursuing* the fish its position is
closed-loop on the fish, so the uniform-fish null is a geometric correction only,
not a behavioral null. A high selection index during pursuit largely reflects the
chaser's controller succeeding, not a spatial preference of the fish. The
per-epoch ``chaser_is_moving`` flag and the ``closed_loop_null`` QC warning mark
where this caveat applies.
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
    load_chaser_distance_run,
)
from fisheye.analysis.chaser_component_writer import (
    require_chaser_component_staging_capability,
    sealed_chaser_component_writer,
)
from fisheye.analysis.chaser_near_field_occupancy import _rectangle_annulus_area_mm2
from fisheye.shared.arena_geometry import (
    ArenaGeometry,
    out_of_bounds_notes,
    resolve_arena_geometry as _resolve_shared_arena_geometry,
)
from fisheye.shared.json_safety import decode_null_terminated_text, json_attr_safe
from fisheye.shared.plot_artifacts import write_interactive_plot_spec_artifact, write_png_visualization_artifact
from fisheye.shared.run_lineage_fingerprint import build_run_lineage_payload, write_run_lineage_attrs
from fisheye.shared.system_metadata import get_git_info


SCHEMA_ID = "palette.chaser_radial_occupancy.v1"
SCHEMA_VERSION = 1
METHOD = "chaser_relative_radial_occupancy"
METHOD_VERSION = "1"
COMPONENT_PARENT_NAME = "chaser_radial_occupancy"
DEFAULT_COMPONENT_NAME = "chaser_radial_occupancy_v1"

RADIAL_DENSITY_PNG_ARTIFACT_NAME = "chaser_radial_occupancy_density_png"
SELECTION_INDEX_PNG_ARTIFACT_NAME = "chaser_radial_occupancy_selection_index_png"
INTERACTIVE_ARTIFACT_NAME = "chaser_radial_occupancy_interactive"
INTERACTIVE_RENDERER = "palette-chaser-radial-occupancy-v1"
INTERACTIVE_SPEC_SCHEMA_ID = "palette.chaser_radial_occupancy.interactive_spec.v1"

DEFAULT_RADIAL_BIN_WIDTH_MM = 2.0
DEFAULT_R_ZONE_MM = 5.0
DEFAULT_CDF_THRESHOLDS_MM = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0)
DEFAULT_PERIMETER_BAND_MM = 5.0
DEFAULT_MOTION_SPREAD_THRESHOLD_MM = 1.0
DEFAULT_AREA_CACHE_STEP_MM = 1.0
# Rings close to the maximum distance the arena can produce have almost no available
# area, so the selection index there is a ratio of two near-zero numbers and explodes on
# a handful of frames. Suppress a ring's ratio unless the null expects at least this many
# samples in it (the usual expected-count floor for a chi-square-style comparison).
DEFAULT_MIN_EXPECTED_COUNT = 5.0
# None -> take the settle window from the protocol's position_transition_duration_s.
DEFAULT_SETTLE_TRIM_S: float | None = None
CONTROL_REFERENCE_LABEL = "dish_center"


@dataclass(frozen=True)
class ChaserRadialEpoch:
    window_id: int
    label: str
    # start_frame is the *effective* start: the settle window is trimmed off the front of
    # epochs whose chaser configuration is static once the objects have finished moving
    # into place. source_start_frame is the untrimmed boundary from the distance run.
    start_frame: int
    end_frame: int
    frame_count: int
    source_start_frame: int
    settle_excluded_frame_count: int
    static_configuration: bool


@dataclass(frozen=True)
class ChaserRadialOccupancyResult:
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
    geometry: ArenaGeometry
    r_zone_mm: float
    perimeter_band_mm: float
    motion_spread_threshold_mm: float
    area_cache_step_mm: float
    min_expected_count: float
    settle_trim_s: float
    radial_bin_edges_mm: np.ndarray
    radial_bin_centers_mm: np.ndarray
    cdf_thresholds_mm: np.ndarray
    epochs: tuple[ChaserRadialEpoch, ...]
    chaser_indices: np.ndarray
    chaser_behavior_labels: tuple[str, ...]
    # (epoch, chaser, bin)
    radial_count: np.ndarray
    radial_observed_fraction: np.ndarray
    radial_expected_area_mm2: np.ndarray
    radial_expected_fraction: np.ndarray
    radial_occupancy_density_per_mm2: np.ndarray
    radial_selection_index: np.ndarray
    radial_count_wall_excluded: np.ndarray
    radial_observed_fraction_wall_excluded: np.ndarray
    radial_expected_area_wall_excluded_mm2: np.ndarray
    radial_expected_fraction_wall_excluded: np.ndarray
    radial_occupancy_density_wall_excluded_per_mm2: np.ndarray
    radial_selection_index_wall_excluded: np.ndarray
    # (epoch, chaser, threshold)
    cdf_observed_fraction: np.ndarray
    cdf_expected_fraction: np.ndarray
    cdf_enrichment: np.ndarray
    # (epoch, chaser)
    valid_frame_count: np.ndarray
    wall_excluded_frame_count: np.ndarray
    near_zone_observed_fraction: np.ndarray
    near_zone_expected_fraction: np.ndarray
    near_zone_enrichment: np.ndarray
    near_zone_dwell_s: np.ndarray
    chaser_position_spread_mm: np.ndarray
    chaser_path_length_mm: np.ndarray
    chaser_mean_distance_to_center_mm: np.ndarray
    chaser_is_moving: np.ndarray
    # (epoch, bin) -- static dish-center control
    control_radial_count: np.ndarray
    control_radial_observed_fraction: np.ndarray
    control_radial_expected_fraction: np.ndarray
    control_radial_selection_index: np.ndarray
    status: str
    qc_warnings: tuple[str, ...]
    summary: dict[str, Any]
    diagnostics: dict[str, Any]


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


def _optional_float(value: Any) -> float | None:
    out = _safe_float(value)
    return out if math.isfinite(out) else None


def _decode_text_column(data: np.ndarray) -> list[str]:
    values = np.asarray(data)
    if values.ndim == 2 and values.dtype.kind in ("u", "i"):
        return [decode_null_terminated_text(row).strip() for row in values]
    return [decode_null_terminated_text(value).strip() for value in values.reshape(-1)]


def _resolve_chaser_distance_run(
    root: zarr.Group,
    run_name: str,
) -> tuple[ChaserDistanceReadSnapshot, str, str]:
    snapshot = load_chaser_distance_run(
        root,
        run_name=str(run_name).strip() or "latest",
    )
    return snapshot, snapshot.run_name, snapshot.run_path


def _resolve_arena_geometry(
    root: zarr.Group,
    distance: ChaserDistanceReadSnapshot,
    *,
    pixels_per_mm: float,
) -> ArenaGeometry:
    """Prefer the fitted dish mask over the projector's nominal experimental_area circle.

    See fisheye.shared.arena_geometry: they are different circles, and the nominal one is 3 mm
    off-centre and 2.4 mm small, which puts a wall-hugging fish "outside the arena".
    """

    geometry, _notes = _resolve_shared_arena_geometry(
        root,
        root[distance.run_path],
        pixels_per_mm=float(pixels_per_mm),
    )
    return geometry


def resolve_arena_geometry_with_notes(
    root: zarr.Group,
    distance: ChaserDistanceReadSnapshot,
    *,
    pixels_per_mm: float,
) -> tuple[ArenaGeometry, list[str]]:
    # Arena-geometry authority has not yet moved behind its own detached
    # snapshot.  Resolve the exact already-verified child only; do not perform a
    # second run selection here.  The remaining geometry lineage hardening is a
    # separate fail-closed work item.
    return _resolve_shared_arena_geometry(
        root,
        root[distance.run_path],
        pixels_per_mm=float(pixels_per_mm),
    )


def _read_epochs(
    distance: ChaserDistanceReadSnapshot,
    *,
    total_frames: int,
) -> tuple[ChaserRadialEpoch, ...]:
    if distance.epoch_start_frame.size == 0:
        return (
            ChaserRadialEpoch(
                window_id=-1,
                label="all_frames",
                start_frame=0,
                end_frame=max(0, int(total_frames) - 1),
                frame_count=int(total_frames),
                source_start_frame=0,
                settle_excluded_frame_count=0,
                static_configuration=False,
            ),
        )
    starts = np.asarray(distance.epoch_start_frame, dtype=np.int64).reshape(-1)
    ends = np.asarray(distance.epoch_end_frame, dtype=np.int64).reshape(-1)
    ids = np.asarray(distance.epoch_window_id, dtype=np.int64).reshape(-1)
    labels = list(distance.epoch_labels)
    epochs: list[ChaserRadialEpoch] = []
    for idx in range(starts.shape[0]):
        start = max(0, int(starts[idx]))
        end = min(int(total_frames) - 1, int(ends[idx]))
        if end < start:
            continue
        epochs.append(
            ChaserRadialEpoch(
                window_id=int(ids[idx]) if idx < ids.shape[0] else idx,
                label=str(labels[idx]).strip() if idx < len(labels) else f"window_{idx}",
                start_frame=start,
                end_frame=end,
                frame_count=end - start + 1,
                source_start_frame=start,
                settle_excluded_frame_count=0,
                static_configuration=False,
            )
        )
    if not epochs:
        raise ValueError("Chaser-distance run epoch_summary contains no usable windows.")
    return tuple(epochs)


def _protocol_position_transition_s(
    root: zarr.Group,
    distance: ChaserDistanceReadSnapshot,
) -> float:
    """The scripted duration objects take to move into their next static configuration."""

    source_path = str(distance.source_stimulus_path or "").strip()
    stim_group = _get_group_by_path(root, source_path) if source_path else None
    if stim_group is None:
        return 0.0
    raw = stim_group.attrs.get("protocol_json")
    if raw is None:
        return 0.0
    try:
        payload = json.loads(raw) if isinstance(raw, str) else raw
        steps = payload["steps"]
        params = steps[0]["parameters"]
    except (TypeError, ValueError, KeyError, IndexError):
        return 0.0
    return max(0.0, _safe_float(params.get("position_transition_duration_s"), 0.0))


def _chaser_position_spread_mm(
    chaser_xy: np.ndarray,
    chaser_valid: np.ndarray,
    *,
    pixels_per_mm: float,
) -> float:
    """RMS spread of a chaser's position (mm) over the given frames."""

    xy = np.asarray(chaser_xy, dtype=np.float64)
    valid = np.asarray(chaser_valid, dtype=bool) & np.isfinite(xy).all(axis=1)
    if not np.any(valid):
        return math.nan
    positions = xy[valid] / float(pixels_per_mm)
    centroid = positions.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((positions - centroid) ** 2, axis=1))))


def _apply_settle_trim(
    epochs: Sequence[ChaserRadialEpoch],
    *,
    chaser_xy: np.ndarray,
    chaser_valid: np.ndarray,
    fps: float,
    pixels_per_mm: float,
    settle_trim_s: float,
    motion_spread_threshold_mm: float,
) -> tuple[ChaserRadialEpoch, ...]:
    """Trim the object-repositioning window off the front of static-configuration epochs.

    At a phase boundary the objects travel to their next position over
    ``position_transition_duration_s``. Those in-transit frames belong to neither phase --
    ``cra_primary_endpoint`` already excludes them via its effective phase windows, and this
    keeps the same convention.

    The trim is applied only to epochs whose chasers are *static once settled*. An epoch
    where a chaser keeps moving after the settle window is a dynamic-stimulus epoch (the
    chase itself), and there is nothing there to settle -- trimming it would silently
    discard real stimulus frames.
    """

    safe_fps = float(fps) if math.isfinite(float(fps)) and float(fps) > 0 else 1.0
    settle_frames = int(math.ceil(max(0.0, float(settle_trim_s)) * safe_fps))
    if settle_frames <= 0:
        return tuple(epochs)

    trimmed: list[ChaserRadialEpoch] = []
    for epoch in epochs:
        candidate_start = epoch.start_frame + settle_frames
        if candidate_start > epoch.end_frame:
            trimmed.append(epoch)
            continue
        settled = slice(candidate_start, epoch.end_frame + 1)
        spreads = [
            _chaser_position_spread_mm(
                chaser_xy[settled, c_idx, :], chaser_valid[settled, c_idx], pixels_per_mm=pixels_per_mm
            )
            for c_idx in range(int(chaser_xy.shape[1]))
        ]
        finite = [s for s in spreads if math.isfinite(s)]
        static_configuration = bool(finite) and all(s <= float(motion_spread_threshold_mm) for s in finite)
        if not static_configuration:
            trimmed.append(epoch)
            continue
        trimmed.append(
            ChaserRadialEpoch(
                window_id=epoch.window_id,
                label=epoch.label,
                start_frame=candidate_start,
                end_frame=epoch.end_frame,
                frame_count=epoch.end_frame - candidate_start + 1,
                source_start_frame=epoch.source_start_frame,
                settle_excluded_frame_count=settle_frames,
                static_configuration=True,
            )
        )
    return tuple(trimmed)


def _disc_overlap_area_mm2(radius_mm: float, center_distance_mm: np.ndarray, arena_radius_mm: float) -> np.ndarray:
    """Area of the intersection of disc(radius_mm) -- centred at ``center_distance_mm``
    from the arena centre -- with the arena disc(arena_radius_mm).

    Closed-form circle-circle intersection, vectorised over frames. This replaces the
    grid quadrature used for the static objects in cra_near_field, which would be
    prohibitive here: it meshes the whole arena per call, and the moving chaser needs
    one call per frame (>1e5) rather than one per object-phase.
    """

    r = float(radius_mm)
    d = np.asarray(center_distance_mm, dtype=np.float64).reshape(-1)
    R = float(arena_radius_mm)
    out = np.zeros(d.shape, dtype=np.float64)
    if r <= 0.0 or R <= 0.0:
        return out
    contained = d <= (R - r)  # ring disc entirely inside the arena
    out[contained] = math.pi * r * r
    engulfing = d <= (r - R)  # arena entirely inside the ring disc
    out[engulfing] = math.pi * R * R
    lens = ~contained & ~engulfing & (d < (R + r))
    if np.any(lens):
        dl = d[lens]
        term_r = r * r * np.arccos(np.clip((dl**2 + r * r - R * R) / (2.0 * dl * r), -1.0, 1.0))
        term_R = R * R * np.arccos(np.clip((dl**2 + R * R - r * r) / (2.0 * dl * R), -1.0, 1.0))
        term_tri = 0.5 * np.sqrt(
            np.clip((-dl + r + R) * (dl + r - R) * (dl - r + R) * (dl + r + R), 0.0, None)
        )
        out[lens] = term_r + term_R - term_tri
    return out


def _accumulate_ring_area_circle_mm2(
    *,
    bin_edges_mm: np.ndarray,
    chaser_center_distance_mm: np.ndarray,
    arena_radius_mm: float,
) -> np.ndarray:
    """Available area of each ring, summed over frames. Units mm^2 (frame-weighted)."""

    edges = np.asarray(bin_edges_mm, dtype=np.float64).reshape(-1)
    cumulative = np.asarray(
        [_disc_overlap_area_mm2(edge, chaser_center_distance_mm, arena_radius_mm).sum() for edge in edges],
        dtype=np.float64,
    )
    return np.clip(np.diff(cumulative), 0.0, None)


def _accumulate_ring_area_rectangle_mm2(
    *,
    bin_edges_mm: np.ndarray,
    chaser_x_px: np.ndarray,
    chaser_y_px: np.ndarray,
    geometry: ArenaGeometry,
    pixels_per_mm: float,
    exclude_perimeter_band_mm: float,
    cache_step_mm: float,
) -> np.ndarray:
    """Rectangular-arena fallback: quantise chaser positions onto a cache grid so the
    grid quadrature runs once per occupied cell rather than once per frame."""

    edges = np.asarray(bin_edges_mm, dtype=np.float64).reshape(-1)
    total = np.zeros(edges.shape[0] - 1, dtype=np.float64)
    step_px = max(0.1, float(cache_step_mm)) * float(pixels_per_mm)
    cells = np.stack(
        [np.round(np.asarray(chaser_x_px, dtype=np.float64) / step_px), np.round(np.asarray(chaser_y_px, dtype=np.float64) / step_px)],
        axis=1,
    )
    unique_cells, counts = np.unique(cells, axis=0, return_counts=True)
    for cell, count in zip(unique_cells, counts):
        area = _rectangle_annulus_area_mm2(
            object_x_px=float(cell[0]) * step_px,
            object_y_px=float(cell[1]) * step_px,
            width_px=float(geometry.width_px),
            height_px=float(geometry.height_px),
            pixels_per_mm=float(pixels_per_mm),
            bin_edges_mm=edges,
            exclude_perimeter_band_mm=float(exclude_perimeter_band_mm),
        )
        total += np.nan_to_num(np.asarray(area, dtype=np.float64), nan=0.0) * float(count)
    return total


def _ring_areas_mm2(
    *,
    bin_edges_mm: np.ndarray,
    chaser_xy_px: np.ndarray,
    geometry: ArenaGeometry,
    pixels_per_mm: float,
    exclude_perimeter_band_mm: float,
    cache_step_mm: float,
) -> np.ndarray:
    if geometry.shape == "circle" and geometry.center_x_px is not None and geometry.center_y_px is not None:
        center_x_mm = float(geometry.center_x_px) / float(pixels_per_mm)
        center_y_mm = float(geometry.center_y_px) / float(pixels_per_mm)
        radius_mm = float(geometry.radius_px or 0.0) / float(pixels_per_mm)
        effective_radius_mm = max(0.0, radius_mm - max(0.0, float(exclude_perimeter_band_mm)))
        chaser_mm = np.asarray(chaser_xy_px, dtype=np.float64) / float(pixels_per_mm)
        center_distance = np.hypot(chaser_mm[:, 0] - center_x_mm, chaser_mm[:, 1] - center_y_mm)
        return _accumulate_ring_area_circle_mm2(
            bin_edges_mm=bin_edges_mm,
            chaser_center_distance_mm=center_distance,
            arena_radius_mm=effective_radius_mm,
        )
    chaser = np.asarray(chaser_xy_px, dtype=np.float64)
    return _accumulate_ring_area_rectangle_mm2(
        bin_edges_mm=bin_edges_mm,
        chaser_x_px=chaser[:, 0],
        chaser_y_px=chaser[:, 1],
        geometry=geometry,
        pixels_per_mm=float(pixels_per_mm),
        exclude_perimeter_band_mm=float(exclude_perimeter_band_mm),
        cache_step_mm=float(cache_step_mm),
    )


def _wall_mask(
    fish_xy_px: np.ndarray,
    *,
    geometry: ArenaGeometry,
    perimeter_band_mm: float,
    pixels_per_mm: float,
) -> np.ndarray:
    """True where the fish sits inside the perimeter band (i.e. hugging the wall)."""

    xy = np.asarray(fish_xy_px, dtype=np.float64)
    band_px = max(0.0, float(perimeter_band_mm)) * float(pixels_per_mm)
    if geometry.shape == "circle" and geometry.center_x_px is not None and geometry.center_y_px is not None:
        radial = np.hypot(xy[:, 0] - float(geometry.center_x_px), xy[:, 1] - float(geometry.center_y_px))
        return radial >= max(0.0, float(geometry.radius_px or 0.0) - band_px)
    return (
        (xy[:, 0] <= band_px)
        | (xy[:, 1] <= band_px)
        | (xy[:, 0] >= float(geometry.width_px) - band_px)
        | (xy[:, 1] >= float(geometry.height_px) - band_px)
    )


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    out = np.full(num.shape, np.nan, dtype=np.float64)
    usable = np.isfinite(num) & np.isfinite(den) & (den > 0)
    out[usable] = num[usable] / den[usable]
    return out


def _selection_index(
    observed_fraction: np.ndarray,
    expected_fraction: np.ndarray,
    *,
    sample_count: int,
    min_expected_count: float,
) -> tuple[np.ndarray, int]:
    """observed / expected, suppressed where the null expects too few samples to support
    a stable ratio. Returns the index and the number of rings suppressed."""

    ratio = _safe_ratio(observed_fraction, expected_fraction)
    expected_count = np.asarray(expected_fraction, dtype=np.float64) * float(sample_count)
    underpowered = np.isfinite(ratio) & (expected_count < float(min_expected_count))
    ratio[underpowered] = np.nan
    return ratio, int(np.count_nonzero(underpowered))


def _default_radial_bin_edges(*, geometry: ArenaGeometry, pixels_per_mm: float, bin_width_mm: float) -> np.ndarray:
    """Bins spanning the maximum distance the arena can physically produce."""

    width = float(bin_width_mm)
    if geometry.shape == "circle" and geometry.radius_px:
        max_distance_mm = 2.0 * float(geometry.radius_px) / float(pixels_per_mm)
    else:
        max_distance_mm = math.hypot(float(geometry.width_px), float(geometry.height_px)) / float(pixels_per_mm)
    n_bins = max(1, int(math.ceil(max_distance_mm / width)))
    return (np.arange(0, n_bins + 1, dtype=np.float64) * width).astype(np.float32)


def _normalize_float_array(values: Sequence[float] | np.ndarray, *, name: str, positive: bool = False) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if positive:
        array = array[array > 0]
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one finite value.")
    return np.unique(array.astype(np.float32))


def _compute_radial_arrays(
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
    geometry: ArenaGeometry,
    radial_bin_edges_mm: np.ndarray,
    cdf_thresholds_mm: np.ndarray,
    r_zone_mm: float,
    perimeter_band_mm: float,
    motion_spread_threshold_mm: float,
    area_cache_step_mm: float,
    min_expected_count: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any], tuple[str, ...]]:
    n_epochs = len(epochs)
    n_chasers = int(chaser_indices.shape[0])
    n_bins = int(radial_bin_edges_mm.shape[0] - 1)
    n_thresholds = int(cdf_thresholds_mm.shape[0])
    edges = np.asarray(radial_bin_edges_mm, dtype=np.float64)
    safe_fps = float(fps) if math.isfinite(float(fps)) and float(fps) > 0 else 1.0

    shape2 = (n_epochs, n_chasers)
    shape3 = (n_epochs, n_chasers, n_bins)
    shape_t = (n_epochs, n_chasers, n_thresholds)

    radial_count = np.zeros(shape3, dtype=np.int64)
    radial_observed = np.full(shape3, np.nan, dtype=np.float32)
    radial_area = np.full(shape3, np.nan, dtype=np.float32)
    radial_expected = np.full(shape3, np.nan, dtype=np.float32)
    radial_density = np.full(shape3, np.nan, dtype=np.float32)
    radial_selection = np.full(shape3, np.nan, dtype=np.float32)
    radial_count_we = np.zeros(shape3, dtype=np.int64)
    radial_observed_we = np.full(shape3, np.nan, dtype=np.float32)
    radial_area_we = np.full(shape3, np.nan, dtype=np.float32)
    radial_expected_we = np.full(shape3, np.nan, dtype=np.float32)
    radial_density_we = np.full(shape3, np.nan, dtype=np.float32)
    radial_selection_we = np.full(shape3, np.nan, dtype=np.float32)
    cdf_observed = np.full(shape_t, np.nan, dtype=np.float32)
    cdf_expected = np.full(shape_t, np.nan, dtype=np.float32)
    cdf_enrichment = np.full(shape_t, np.nan, dtype=np.float32)
    valid_count = np.zeros(shape2, dtype=np.int64)
    wall_excluded_count = np.zeros(shape2, dtype=np.int64)
    near_observed = np.full(shape2, np.nan, dtype=np.float32)
    near_expected = np.full(shape2, np.nan, dtype=np.float32)
    near_enrichment = np.full(shape2, np.nan, dtype=np.float32)
    near_dwell = np.full(shape2, np.nan, dtype=np.float32)
    chaser_spread = np.full(shape2, np.nan, dtype=np.float32)
    chaser_path = np.full(shape2, np.nan, dtype=np.float32)
    chaser_center_dist = np.full(shape2, np.nan, dtype=np.float32)
    chaser_moving = np.zeros(shape2, dtype=bool)
    control_count = np.zeros((n_epochs, n_bins), dtype=np.int64)
    control_observed = np.full((n_epochs, n_bins), np.nan, dtype=np.float32)
    control_expected = np.full((n_epochs, n_bins), np.nan, dtype=np.float32)
    control_selection = np.full((n_epochs, n_bins), np.nan, dtype=np.float32)

    warnings: list[str] = []
    # Warn on a *shape* fallback (a rectangle approximation), not on the status string --
    # "dish_mask" is the good case and must not read as a problem.
    if geometry.shape != "circle":
        warnings.append(f"arena_geometry_not_circular:{geometry.status}")

    # The dish-centre control uses a fixed reference point, so its available area is
    # identical on every frame -- compute the per-frame ring area once and scale.
    if geometry.shape == "circle" and geometry.center_x_px is not None and geometry.center_y_px is not None:
        control_xy_px = np.asarray([[float(geometry.center_x_px), float(geometry.center_y_px)]], dtype=np.float64)
    else:
        control_xy_px = np.asarray([[float(geometry.width_px) / 2.0, float(geometry.height_px) / 2.0]], dtype=np.float64)
    control_ring_area = _ring_areas_mm2(
        bin_edges_mm=edges,
        chaser_xy_px=control_xy_px,
        geometry=geometry,
        pixels_per_mm=float(pixels_per_mm),
        exclude_perimeter_band_mm=0.0,
        cache_step_mm=float(area_cache_step_mm),
    )
    control_expected_fraction = _safe_ratio(control_ring_area, np.full(control_ring_area.shape, control_ring_area.sum()))

    for e_idx, epoch in enumerate(epochs):
        slc = slice(epoch.start_frame, epoch.end_frame + 1)
        epoch_fish_xy = np.asarray(fish_xy[slc], dtype=np.float64)
        epoch_fish_valid = np.asarray(fish_valid[slc], dtype=bool) & np.isfinite(epoch_fish_xy).all(axis=1)
        epoch_wall = _wall_mask(
            epoch_fish_xy,
            geometry=geometry,
            perimeter_band_mm=float(perimeter_band_mm),
            pixels_per_mm=float(pixels_per_mm),
        )

        # Static dish-centre control: fish distance to the arena centre, same rings.
        control_distance = (
            np.hypot(
                epoch_fish_xy[:, 0] - float(control_xy_px[0, 0]),
                epoch_fish_xy[:, 1] - float(control_xy_px[0, 1]),
            )
            / float(pixels_per_mm)
        )
        control_values = control_distance[epoch_fish_valid & np.isfinite(control_distance)]
        if control_values.size:
            counts, _ = np.histogram(control_values, bins=edges)
            control_count[e_idx, :] = counts.astype(np.int64)
            observed = counts.astype(np.float64) / float(control_values.size)
            control_observed[e_idx, :] = observed.astype(np.float32)
            control_expected[e_idx, :] = control_expected_fraction.astype(np.float32)
            selection, _suppressed = _selection_index(
                observed,
                control_expected_fraction,
                sample_count=int(control_values.size),
                min_expected_count=float(min_expected_count),
            )
            control_selection[e_idx, :] = selection.astype(np.float32)

        for c_idx in range(n_chasers):
            epoch_chaser_xy = np.asarray(chaser_xy[slc, c_idx, :], dtype=np.float64)
            epoch_chaser_valid = np.asarray(chaser_valid[slc, c_idx], dtype=bool) & np.isfinite(epoch_chaser_xy).all(axis=1)
            epoch_distance = np.asarray(distance_mm[slc, c_idx], dtype=np.float64)
            usable = epoch_fish_valid & epoch_chaser_valid & np.isfinite(epoch_distance)
            valid_count[e_idx, c_idx] = int(np.count_nonzero(usable))
            if valid_count[e_idx, c_idx] == 0:
                warnings.append(f"no_valid_frames:{epoch.label}:chaser{int(chaser_indices[c_idx])}")
                continue

            chaser_mm = epoch_chaser_xy[usable] / float(pixels_per_mm)
            centroid = chaser_mm.mean(axis=0)
            spread = float(np.sqrt(np.mean(np.sum((chaser_mm - centroid) ** 2, axis=1))))
            chaser_spread[e_idx, c_idx] = spread
            chaser_path[e_idx, c_idx] = float(np.sum(np.hypot(*np.diff(chaser_mm, axis=0).T))) if chaser_mm.shape[0] > 1 else 0.0
            chaser_moving[e_idx, c_idx] = spread > float(motion_spread_threshold_mm)
            if geometry.shape == "circle" and geometry.center_x_px is not None and geometry.center_y_px is not None:
                chaser_center_dist[e_idx, c_idx] = float(
                    np.mean(
                        np.hypot(
                            chaser_mm[:, 0] - float(geometry.center_x_px) / float(pixels_per_mm),
                            chaser_mm[:, 1] - float(geometry.center_y_px) / float(pixels_per_mm),
                        )
                    )
                )
            if chaser_moving[e_idx, c_idx]:
                warnings.append(f"closed_loop_null:{epoch.label}:chaser{int(chaser_indices[c_idx])}")

            # --- full-arena rings ---
            values = epoch_distance[usable]
            ring_area = _ring_areas_mm2(
                bin_edges_mm=edges,
                chaser_xy_px=epoch_chaser_xy[usable],
                geometry=geometry,
                pixels_per_mm=float(pixels_per_mm),
                exclude_perimeter_band_mm=0.0,
                cache_step_mm=float(area_cache_step_mm),
            )
            counts, _ = np.histogram(values, bins=edges)
            observed = counts.astype(np.float64) / float(values.size)
            expected = _safe_ratio(ring_area, np.full(ring_area.shape, ring_area.sum()))
            radial_count[e_idx, c_idx, :] = counts.astype(np.int64)
            radial_observed[e_idx, c_idx, :] = observed.astype(np.float32)
            radial_area[e_idx, c_idx, :] = ring_area.astype(np.float32)
            radial_expected[e_idx, c_idx, :] = expected.astype(np.float32)
            radial_density[e_idx, c_idx, :] = _safe_ratio(counts.astype(np.float64), ring_area).astype(np.float32)
            selection, suppressed = _selection_index(
                observed,
                expected,
                sample_count=int(values.size),
                min_expected_count=float(min_expected_count),
            )
            radial_selection[e_idx, c_idx, :] = selection.astype(np.float32)
            if suppressed:
                warnings.append(f"low_expected_count_rings:{epoch.label}:chaser{int(chaser_indices[c_idx])}")

            # --- wall-excluded rings ---
            usable_we = usable & ~epoch_wall
            wall_excluded_count[e_idx, c_idx] = int(np.count_nonzero(usable_we))
            if wall_excluded_count[e_idx, c_idx] > 0:
                values_we = epoch_distance[usable_we]
                # The available region is the arena core; the frame set is unchanged, so
                # observed and expected refer to the same population of frames.
                ring_area_we = _ring_areas_mm2(
                    bin_edges_mm=edges,
                    chaser_xy_px=epoch_chaser_xy[usable],
                    geometry=geometry,
                    pixels_per_mm=float(pixels_per_mm),
                    exclude_perimeter_band_mm=float(perimeter_band_mm),
                    cache_step_mm=float(area_cache_step_mm),
                )
                counts_we, _ = np.histogram(values_we, bins=edges)
                observed_we = counts_we.astype(np.float64) / float(values_we.size)
                expected_we = _safe_ratio(ring_area_we, np.full(ring_area_we.shape, ring_area_we.sum()))
                radial_count_we[e_idx, c_idx, :] = counts_we.astype(np.int64)
                radial_observed_we[e_idx, c_idx, :] = observed_we.astype(np.float32)
                radial_area_we[e_idx, c_idx, :] = ring_area_we.astype(np.float32)
                radial_expected_we[e_idx, c_idx, :] = expected_we.astype(np.float32)
                radial_density_we[e_idx, c_idx, :] = _safe_ratio(counts_we.astype(np.float64), ring_area_we).astype(np.float32)
                selection_we, _suppressed_we = _selection_index(
                    observed_we,
                    expected_we,
                    sample_count=int(values_we.size),
                    min_expected_count=float(min_expected_count),
                )
                radial_selection_we[e_idx, c_idx, :] = selection_we.astype(np.float32)

            # --- CDF ladder, observed vs geometric null ---
            for t_idx, threshold in enumerate(cdf_thresholds_mm):
                below = edges[:-1] < float(threshold)
                observed_frac = float(np.count_nonzero(values <= float(threshold))) / float(values.size)
                expected_frac = float(np.nansum(expected[below]))
                cdf_observed[e_idx, c_idx, t_idx] = observed_frac
                cdf_expected[e_idx, c_idx, t_idx] = expected_frac
                cdf_enrichment[e_idx, c_idx, t_idx] = (
                    observed_frac / expected_frac if expected_frac > 0 else np.nan
                )

            # --- near zone ---
            near_count = int(np.count_nonzero(values <= float(r_zone_mm)))
            near_ring_area = _ring_areas_mm2(
                bin_edges_mm=np.asarray([0.0, float(r_zone_mm)], dtype=np.float64),
                chaser_xy_px=epoch_chaser_xy[usable],
                geometry=geometry,
                pixels_per_mm=float(pixels_per_mm),
                exclude_perimeter_band_mm=0.0,
                cache_step_mm=float(area_cache_step_mm),
            )
            total_area = float(ring_area.sum())
            near_observed[e_idx, c_idx] = float(near_count) / float(values.size)
            near_expected[e_idx, c_idx] = (
                float(near_ring_area[0]) / total_area if near_ring_area.size and total_area > 0 else np.nan
            )
            if math.isfinite(float(near_expected[e_idx, c_idx])) and float(near_expected[e_idx, c_idx]) > 0:
                near_enrichment[e_idx, c_idx] = float(near_observed[e_idx, c_idx]) / float(near_expected[e_idx, c_idx])
            near_dwell[e_idx, c_idx] = float(near_count) / safe_fps

    arrays = {
        "radial_count": radial_count,
        "radial_observed_fraction": radial_observed,
        "radial_expected_area_mm2": radial_area,
        "radial_expected_fraction": radial_expected,
        "radial_occupancy_density_per_mm2": radial_density,
        "radial_selection_index": radial_selection,
        "radial_count_wall_excluded": radial_count_we,
        "radial_observed_fraction_wall_excluded": radial_observed_we,
        "radial_expected_area_wall_excluded_mm2": radial_area_we,
        "radial_expected_fraction_wall_excluded": radial_expected_we,
        "radial_occupancy_density_wall_excluded_per_mm2": radial_density_we,
        "radial_selection_index_wall_excluded": radial_selection_we,
        "cdf_observed_fraction": cdf_observed,
        "cdf_expected_fraction": cdf_expected,
        "cdf_enrichment": cdf_enrichment,
        "valid_frame_count": valid_count,
        "wall_excluded_frame_count": wall_excluded_count,
        "near_zone_observed_fraction": near_observed,
        "near_zone_expected_fraction": near_expected,
        "near_zone_enrichment": near_enrichment,
        "near_zone_dwell_s": near_dwell,
        "chaser_position_spread_mm": chaser_spread,
        "chaser_path_length_mm": chaser_path,
        "chaser_mean_distance_to_center_mm": chaser_center_dist,
        "chaser_is_moving": chaser_moving,
        "control_radial_count": control_count,
        "control_radial_observed_fraction": control_observed,
        "control_radial_expected_fraction": control_expected,
        "control_radial_selection_index": control_selection,
    }
    diagnostics = {
        "geometry_status": geometry.status,
        "arena_geometry_source": geometry.source,
        "arena_shape": geometry.shape,
        "arena_center_x_px": geometry.center_x_px,
        "arena_center_y_px": geometry.center_y_px,
        "arena_radius_px": geometry.radius_px,
        "area_method": "closed_form_circle_intersection" if geometry.shape == "circle" else "grid_quadrature_cached_by_chaser_cell",
        "area_cache_step_mm": float(area_cache_step_mm),
        "null_model": (
            "fish uniform over available arena area, conditioned on the chaser position "
            "actually observed on each frame"
        ),
        "closed_loop_caveat": (
            "when chaser_is_moving is true the chaser is pursuing the fish, so its position is "
            "not independent of the fish; the expected_fraction is a geometric correction only, "
            "not a behavioral null"
        ),
        "wall_excluded_policy": "exclude fish samples and available area within perimeter_band_mm of the arena wall",
        "control_reference": CONTROL_REFERENCE_LABEL,
        "motion_spread_threshold_mm": float(motion_spread_threshold_mm),
        "min_expected_count": float(min_expected_count),
        "low_expected_count_policy": (
            "selection_index is NaN in rings where the geometric null expects fewer than "
            "min_expected_count samples; counts and areas are still persisted"
        ),
        "qc_warning_count": len(warnings),
    }
    return arrays, diagnostics, tuple(dict.fromkeys(warnings))


def _build_summary(
    *,
    recording_id: str,
    epochs: Sequence[ChaserRadialEpoch],
    chaser_indices: np.ndarray,
    chaser_behavior_labels: Sequence[str],
    near_zone_observed_fraction: np.ndarray,
    near_zone_expected_fraction: np.ndarray,
    near_zone_enrichment: np.ndarray,
    chaser_is_moving: np.ndarray,
    chaser_position_spread_mm: np.ndarray,
    valid_frame_count: np.ndarray,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "recording_id": recording_id,
        "n_epochs": len(epochs),
        "n_chasers": int(chaser_indices.shape[0]),
    }
    for e_idx, epoch in enumerate(epochs):
        for c_idx in range(int(chaser_indices.shape[0])):
            label = chaser_behavior_labels[c_idx] if c_idx < len(chaser_behavior_labels) else ""
            stem = f"{epoch.label}_chaser{int(chaser_indices[c_idx])}"
            if label:
                stem = f"{epoch.label}_{label}"
            observed = float(near_zone_observed_fraction[e_idx, c_idx])
            expected = float(near_zone_expected_fraction[e_idx, c_idx])
            enrichment = float(near_zone_enrichment[e_idx, c_idx])
            summary[f"nearzone_observed_{stem}"] = observed if math.isfinite(observed) else None
            summary[f"nearzone_expected_{stem}"] = expected if math.isfinite(expected) else None
            summary[f"nearzone_enrichment_{stem}"] = enrichment if math.isfinite(enrichment) else None
            summary[f"chaser_is_moving_{stem}"] = bool(chaser_is_moving[e_idx, c_idx])
            spread = float(chaser_position_spread_mm[e_idx, c_idx])
            summary[f"chaser_position_spread_mm_{stem}"] = spread if math.isfinite(spread) else None
            summary[f"n_valid_frames_{stem}"] = int(valid_frame_count[e_idx, c_idx])
    return summary


def build_chaser_radial_occupancy_result(
    zarr_path: Path,
    *,
    chaser_distance_run: str = "latest",
    component_name: str = DEFAULT_COMPONENT_NAME,
    radial_bin_width_mm: float = DEFAULT_RADIAL_BIN_WIDTH_MM,
    radial_bin_edges_mm: Sequence[float] | None = None,
    cdf_thresholds_mm: Sequence[float] = DEFAULT_CDF_THRESHOLDS_MM,
    r_zone_mm: float = DEFAULT_R_ZONE_MM,
    perimeter_band_mm: float = DEFAULT_PERIMETER_BAND_MM,
    motion_spread_threshold_mm: float = DEFAULT_MOTION_SPREAD_THRESHOLD_MM,
    area_cache_step_mm: float = DEFAULT_AREA_CACHE_STEP_MM,
    min_expected_count: float = DEFAULT_MIN_EXPECTED_COUNT,
    settle_trim_s: float | None = DEFAULT_SETTLE_TRIM_S,
) -> ChaserRadialOccupancyResult:
    if float(r_zone_mm) <= 0:
        raise ValueError("r_zone_mm must be positive.")
    if float(radial_bin_width_mm) <= 0:
        raise ValueError("radial_bin_width_mm must be positive.")

    root = _open_root(zarr_path, mode="r")
    distance, run_name, run_path = _resolve_chaser_distance_run(
        root,
        chaser_distance_run,
    )
    distance.require_arena_geometry_authority()

    coordinate_frame = distance.coordinate_space_id
    coordinate_origin = distance.coordinate_origin
    if coordinate_frame != "arena_relative_canvas_px":
        raise ValueError(
            "Chaser radial occupancy requires typed "
            f"space_id='arena_relative_canvas_px'; got {coordinate_frame!r}."
        )
    if coordinate_origin != "arena_top_left":
        raise ValueError(
            "Chaser radial occupancy requires typed "
            f"origin='arena_top_left'; got {coordinate_origin!r}."
        )
    pixels_per_mm = float(distance.pixels_per_mm_projector)

    fish_xy = np.asarray(distance.fish_centroid_arena_xy, dtype=np.float32)
    chaser_xy = np.asarray(distance.chaser_arena_xy, dtype=np.float32)
    fish_valid = np.asarray(distance.fish_valid, dtype=bool)
    chaser_valid = np.asarray(distance.chaser_valid, dtype=bool)
    distance_mm = np.asarray(distance.distance_mm, dtype=np.float32)
    chaser_indices = np.asarray(distance.chaser_index, dtype=np.int64).reshape(-1)
    # Protocol-derived roles are intentionally unavailable until their own
    # semantic authority is sealed.  Radial computation is identity-based and
    # remains valid without assigning role labels.
    behavior_labels: tuple[str, ...] = ()

    total_frames = int(fish_xy.shape[0])
    if distance.total_frames != total_frames:
        raise ValueError(
            "Chaser-distance typed frame-axis mismatch: "
            f"authority total_frames={distance.total_frames}, "
            f"fish_centroid_arena_xy length={total_frames}."
        )
    if fish_xy.ndim != 2 or fish_xy.shape[1] != 2:
        raise ValueError("positions/fish_centroid_arena_xy must have shape (frame, xy).")
    if chaser_xy.ndim != 3 or chaser_xy.shape[0] != total_frames or chaser_xy.shape[2] != 2:
        raise ValueError("positions/chaser_arena_xy must have shape (frame, chaser, xy).")
    if distance_mm.ndim != 2 or distance_mm.shape != (total_frames, chaser_xy.shape[1]):
        raise ValueError("distances/distance_mm must have shape (frame, chaser) matching chaser_arena_xy.")
    if chaser_valid.shape != distance_mm.shape:
        raise ValueError("positions/chaser_valid shape must match distances/distance_mm.")
    if chaser_indices.shape[0] != distance_mm.shape[1]:
        raise ValueError("chasers/chaser_index length does not match distance columns.")

    geometry, geometry_notes = resolve_arena_geometry_with_notes(
        root,
        distance,
        pixels_per_mm=float(pixels_per_mm),
    )
    geometry_notes = geometry_notes + out_of_bounds_notes(fish_xy, fish_valid, geometry, label="fish")
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

    edges = (
        _default_radial_bin_edges(
            geometry=geometry, pixels_per_mm=float(pixels_per_mm), bin_width_mm=float(radial_bin_width_mm)
        )
        if radial_bin_edges_mm is None
        else _normalize_float_array(radial_bin_edges_mm, name="radial_bin_edges_mm")
    )
    if edges.shape[0] < 2 or not np.all(np.diff(edges.astype(np.float64)) > 0):
        raise ValueError("radial_bin_edges_mm must contain at least two increasing values.")
    centers = ((edges[:-1] + edges[1:]) / 2.0).astype(np.float32)
    thresholds = _normalize_float_array(cdf_thresholds_mm, name="cdf_thresholds_mm", positive=True)

    arrays, diagnostics, qc_warnings = _compute_radial_arrays(
        epochs=epochs,
        chaser_indices=chaser_indices,
        fish_xy=fish_xy,
        chaser_xy=chaser_xy,
        fish_valid=fish_valid,
        chaser_valid=chaser_valid,
        distance_mm=distance_mm,
        fps=float(fps),
        pixels_per_mm=float(pixels_per_mm),
        geometry=geometry,
        radial_bin_edges_mm=edges,
        cdf_thresholds_mm=thresholds,
        r_zone_mm=float(r_zone_mm),
        perimeter_band_mm=float(perimeter_band_mm),
        motion_spread_threshold_mm=float(motion_spread_threshold_mm),
        area_cache_step_mm=float(area_cache_step_mm),
        min_expected_count=float(min_expected_count),
    )

    qc_warnings = tuple(dict.fromkeys(list(qc_warnings) + list(geometry_notes)))
    recording_id = distance.recording_id
    summary = _build_summary(
        recording_id=recording_id,
        epochs=epochs,
        chaser_indices=chaser_indices,
        chaser_behavior_labels=behavior_labels,
        near_zone_observed_fraction=arrays["near_zone_observed_fraction"],
        near_zone_expected_fraction=arrays["near_zone_expected_fraction"],
        near_zone_enrichment=arrays["near_zone_enrichment"],
        chaser_is_moving=arrays["chaser_is_moving"],
        chaser_position_spread_mm=arrays["chaser_position_spread_mm"],
        valid_frame_count=arrays["valid_frame_count"],
    )

    return ChaserRadialOccupancyResult(
        zarr_path=str(zarr_path),
        recording_id=recording_id,
        component_name=str(component_name),
        chaser_distance_run_name=run_name,
        chaser_distance_run_path=run_path,
        source_stimulus_run=distance.source_stimulus_run,
        source_stimulus_path=distance.source_stimulus_path,
        source_stimulus_epoch_run=distance.source_stimulus_epoch_run,
        source_stimulus_epoch_path=distance.source_stimulus_epoch_path,
        fps=float(fps),
        total_frames=int(total_frames),
        pixels_per_mm_projector=float(pixels_per_mm),
        coordinate_frame=coordinate_frame,
        coordinate_origin=coordinate_origin,
        geometry=geometry,
        r_zone_mm=float(r_zone_mm),
        perimeter_band_mm=float(perimeter_band_mm),
        motion_spread_threshold_mm=float(motion_spread_threshold_mm),
        area_cache_step_mm=float(area_cache_step_mm),
        min_expected_count=float(min_expected_count),
        settle_trim_s=float(resolved_settle_trim_s),
        radial_bin_edges_mm=edges.astype(np.float32),
        radial_bin_centers_mm=centers,
        cdf_thresholds_mm=thresholds.astype(np.float32),
        epochs=epochs,
        chaser_indices=chaser_indices.astype(np.int16),
        chaser_behavior_labels=behavior_labels,
        radial_count=arrays["radial_count"],
        radial_observed_fraction=arrays["radial_observed_fraction"],
        radial_expected_area_mm2=arrays["radial_expected_area_mm2"],
        radial_expected_fraction=arrays["radial_expected_fraction"],
        radial_occupancy_density_per_mm2=arrays["radial_occupancy_density_per_mm2"],
        radial_selection_index=arrays["radial_selection_index"],
        radial_count_wall_excluded=arrays["radial_count_wall_excluded"],
        radial_observed_fraction_wall_excluded=arrays["radial_observed_fraction_wall_excluded"],
        radial_expected_area_wall_excluded_mm2=arrays["radial_expected_area_wall_excluded_mm2"],
        radial_expected_fraction_wall_excluded=arrays["radial_expected_fraction_wall_excluded"],
        radial_occupancy_density_wall_excluded_per_mm2=arrays["radial_occupancy_density_wall_excluded_per_mm2"],
        radial_selection_index_wall_excluded=arrays["radial_selection_index_wall_excluded"],
        cdf_observed_fraction=arrays["cdf_observed_fraction"],
        cdf_expected_fraction=arrays["cdf_expected_fraction"],
        cdf_enrichment=arrays["cdf_enrichment"],
        valid_frame_count=arrays["valid_frame_count"],
        wall_excluded_frame_count=arrays["wall_excluded_frame_count"],
        near_zone_observed_fraction=arrays["near_zone_observed_fraction"],
        near_zone_expected_fraction=arrays["near_zone_expected_fraction"],
        near_zone_enrichment=arrays["near_zone_enrichment"],
        near_zone_dwell_s=arrays["near_zone_dwell_s"],
        chaser_position_spread_mm=arrays["chaser_position_spread_mm"],
        chaser_path_length_mm=arrays["chaser_path_length_mm"],
        chaser_mean_distance_to_center_mm=arrays["chaser_mean_distance_to_center_mm"],
        chaser_is_moving=arrays["chaser_is_moving"],
        control_radial_count=arrays["control_radial_count"],
        control_radial_observed_fraction=arrays["control_radial_observed_fraction"],
        control_radial_expected_fraction=arrays["control_radial_expected_fraction"],
        control_radial_selection_index=arrays["control_radial_selection_index"],
        status="computed",
        qc_warnings=qc_warnings,
        summary=summary,
        diagnostics=diagnostics,
    )


def _chaser_label(result: ChaserRadialOccupancyResult, c_idx: int) -> str:
    index = int(result.chaser_indices[c_idx])
    if c_idx < len(result.chaser_behavior_labels) and result.chaser_behavior_labels[c_idx]:
        return f"chaser {index} ({result.chaser_behavior_labels[c_idx]})"
    return f"chaser {index}"


def render_chaser_radial_occupancy_density_png(result: ChaserRadialOccupancyResult, *, dpi: int = 150) -> bytes:
    n = len(result.epochs)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 4.4), sharey=True, constrained_layout=True)
    axes_list = list(np.atleast_1d(axes).ravel())
    for e_idx, (ax, epoch) in enumerate(zip(axes_list, result.epochs)):
        for c_idx in range(int(result.chaser_indices.shape[0])):
            observed = result.radial_observed_fraction[e_idx, c_idx, :]
            if not np.isfinite(observed).any():
                continue
            ax.plot(
                result.radial_bin_centers_mm,
                observed,
                marker="o",
                markersize=3.0,
                linewidth=1.3,
                label=f"{_chaser_label(result, c_idx)} observed",
            )
            ax.plot(
                result.radial_bin_centers_mm,
                result.radial_expected_fraction[e_idx, c_idx, :],
                linestyle="--",
                linewidth=1.1,
                alpha=0.7,
                label=f"{_chaser_label(result, c_idx)} geometric null",
            )
        ax.axvline(result.r_zone_mm, color="#334155", linestyle=":", linewidth=0.9, alpha=0.8)
        ax.set_title(epoch.label.replace("_", " "))
        ax.set_xlabel("distance to chaser (mm)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    axes_list[0].set_ylabel("fraction of valid frames per ring")
    fig.suptitle(f"Chaser radial occupancy vs geometric null: {result.recording_id}", fontsize=12)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=int(dpi))
    plt.close(fig)
    return buf.getvalue()


def render_chaser_radial_occupancy_selection_index_png(result: ChaserRadialOccupancyResult, *, dpi: int = 150) -> bytes:
    n = len(result.epochs)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 4.4), sharey=True, constrained_layout=True)
    axes_list = list(np.atleast_1d(axes).ravel())
    for e_idx, (ax, epoch) in enumerate(zip(axes_list, result.epochs)):
        for c_idx in range(int(result.chaser_indices.shape[0])):
            selection = result.radial_selection_index[e_idx, c_idx, :]
            if not np.isfinite(selection).any():
                continue
            moving = bool(result.chaser_is_moving[e_idx, c_idx])
            ax.plot(
                result.radial_bin_centers_mm,
                selection,
                marker="o",
                markersize=3.0,
                linewidth=1.3,
                label=f"{_chaser_label(result, c_idx)}{' [pursuing]' if moving else ''}",
            )
        ax.axhline(1.0, color="#334155", linestyle="--", linewidth=0.9, alpha=0.8)
        ax.axvline(result.r_zone_mm, color="#334155", linestyle=":", linewidth=0.9, alpha=0.6)
        ax.set_yscale("log")
        ax.set_title(epoch.label.replace("_", " "))
        ax.set_xlabel("distance to chaser (mm)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    axes_list[0].set_ylabel("selection index (observed / geometric null)")
    fig.suptitle(f"Chaser radial selection index: {result.recording_id}", fontsize=12)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=int(dpi))
    plt.close(fig)
    return buf.getvalue()


def _source_refs(result: ChaserRadialOccupancyResult) -> dict[str, Any]:
    return {
        "source_chaser_distance_run": result.chaser_distance_run_name,
        "source_chaser_distance_path": result.chaser_distance_run_path,
        "source_stimulus_run": result.source_stimulus_run,
        "source_stimulus_path": result.source_stimulus_path,
        "source_stimulus_epoch_run": result.source_stimulus_epoch_run,
        "source_stimulus_epoch_path": result.source_stimulus_epoch_path,
    }


def _parameters(result: ChaserRadialOccupancyResult) -> dict[str, Any]:
    return {
        "r_zone_mm": result.r_zone_mm,
        "perimeter_band_mm": result.perimeter_band_mm,
        "motion_spread_threshold_mm": result.motion_spread_threshold_mm,
        "area_cache_step_mm": result.area_cache_step_mm,
        "min_expected_count": result.min_expected_count,
        "settle_trim_s": result.settle_trim_s,
        "settle_policy": "trim_position_transition_duration_s_from_static_configuration_epochs",
        "radial_bin_edges_mm": result.radial_bin_edges_mm,
        "cdf_thresholds_mm": result.cdf_thresholds_mm,
        "geometry_mode": result.geometry.status,
        "arena_geometry_source": result.geometry.source,
        "arena_shape": result.geometry.shape,
        "arena_center_x_px": result.geometry.center_x_px,
        "arena_center_y_px": result.geometry.center_y_px,
        "arena_radius_px": result.geometry.radius_px,
        "area_method": result.diagnostics.get("area_method"),
        "null_model": result.diagnostics.get("null_model"),
    }


def _interactive_spec(result: ChaserRadialOccupancyResult, component_path: str) -> dict[str, Any]:
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
            "radial_occupancy": f"{component_path}/radial_occupancy",
            "distance_cdf": f"{component_path}/distance_cdf",
            "control_reference": f"{component_path}/control_reference",
            "per_epoch_chaser": f"{component_path}/per_epoch_chaser",
            "summary": f"{component_path}/summary",
        },
        "summary": result.summary,
        "parameters": _parameters(result),
        "qc_warnings": list(result.qc_warnings),
    }


@sealed_chaser_component_writer(
    component_family=COMPONENT_PARENT_NAME,
    semantic_schema_id=SCHEMA_ID,
    semantic_schema_version=SCHEMA_VERSION,
    method_id=METHOD,
    method_version=METHOD_VERSION,
)
def write_chaser_radial_occupancy_component(
    zarr_path: Path,
    result: ChaserRadialOccupancyResult,
    *,
    overwrite: bool = False,
    write_png: bool = True,
    write_interactive_spec: bool = True,
    _chaser_component_staging_capability: object | None = None,
) -> str:
    require_chaser_component_staging_capability(
        _chaser_component_staging_capability
    )
    root = _open_root(zarr_path, mode="a")
    run_group = root[result.chaser_distance_run_path]
    parent = run_group.require_group(COMPONENT_PARENT_NAME)
    component_name = result.component_name
    if component_name in parent:
        raise RuntimeError(
            "Private chaser component staging archive contains a same-name child."
        )
    component = parent.create_group(component_name)
    component_path = f"{result.chaser_distance_run_path}/{COMPONENT_PARENT_NAME}/{component_name}"

    config = component.require_group("config")
    _write_array(config, "radial_bin_edges_mm", result.radial_bin_edges_mm)
    _write_array(config, "radial_bin_centers_mm", result.radial_bin_centers_mm)
    _write_array(config, "cdf_thresholds_mm", result.cdf_thresholds_mm)
    config.attrs.update(
        {
            "r_zone_mm": float(result.r_zone_mm),
            "perimeter_band_mm": float(result.perimeter_band_mm),
            "motion_spread_threshold_mm": float(result.motion_spread_threshold_mm),
            "area_cache_step_mm": float(result.area_cache_step_mm),
            "min_expected_count": float(result.min_expected_count),
            "settle_trim_s": float(result.settle_trim_s),
            "geometry_status": result.geometry.status,
            "arena_geometry_source": result.geometry.source,
            "arena_shape": result.geometry.shape,
            "arena_center_x_px": result.geometry.center_x_px,
            "arena_center_y_px": result.geometry.center_y_px,
            "arena_radius_px": result.geometry.radius_px,
        }
    )

    epochs = component.require_group("epochs")
    _write_array(epochs, "window_id", np.asarray([e.window_id for e in result.epochs], dtype=np.int32))
    _write_array(epochs, "label_bytes", _bytes_array([e.label for e in result.epochs], width=96))
    _write_array(epochs, "start_frame", np.asarray([e.start_frame for e in result.epochs], dtype=np.int64))
    _write_array(epochs, "end_frame", np.asarray([e.end_frame for e in result.epochs], dtype=np.int64))
    _write_array(epochs, "frame_count", np.asarray([e.frame_count for e in result.epochs], dtype=np.int64))
    _write_array(epochs, "source_start_frame", np.asarray([e.source_start_frame for e in result.epochs], dtype=np.int64))
    _write_array(
        epochs,
        "settle_excluded_frame_count",
        np.asarray([e.settle_excluded_frame_count for e in result.epochs], dtype=np.int64),
    )
    _write_array(epochs, "static_configuration", np.asarray([e.static_configuration for e in result.epochs], dtype=bool))
    epochs.attrs.update(
        {
            "row_axis": "epoch",
            "settle_trim_s": float(result.settle_trim_s),
            "settle_policy": (
                "start_frame is source_start_frame plus the settle window, applied only to epochs whose "
                "chasers are static once settled; dynamic-stimulus epochs are left untrimmed"
            ),
        }
    )

    chasers = component.require_group("chasers")
    _write_array(chasers, "chaser_index", result.chaser_indices)
    if result.chaser_behavior_labels:
        _write_array(chasers, "behavior_class_label_bytes", _bytes_array(result.chaser_behavior_labels, width=32))
    chasers.attrs.update({"row_axis": "chaser"})

    radial = component.require_group("radial_occupancy")
    _write_array(radial, "radial_count", result.radial_count)
    _write_array(radial, "radial_observed_fraction", result.radial_observed_fraction)
    _write_array(radial, "radial_expected_area_mm2", result.radial_expected_area_mm2)
    _write_array(radial, "radial_expected_fraction", result.radial_expected_fraction)
    _write_array(radial, "radial_occupancy_density_per_mm2", result.radial_occupancy_density_per_mm2)
    _write_array(radial, "radial_selection_index", result.radial_selection_index)
    _write_array(radial, "radial_count_wall_excluded", result.radial_count_wall_excluded)
    _write_array(radial, "radial_observed_fraction_wall_excluded", result.radial_observed_fraction_wall_excluded)
    _write_array(radial, "radial_expected_area_wall_excluded_mm2", result.radial_expected_area_wall_excluded_mm2)
    _write_array(radial, "radial_expected_fraction_wall_excluded", result.radial_expected_fraction_wall_excluded)
    _write_array(
        radial,
        "radial_occupancy_density_wall_excluded_per_mm2",
        result.radial_occupancy_density_wall_excluded_per_mm2,
    )
    _write_array(radial, "radial_selection_index_wall_excluded", result.radial_selection_index_wall_excluded)
    radial.attrs.update(
        {
            "axis_order": ["epoch", "chaser", "radial_bin"],
            "geometry_status": result.geometry.status,
            "area_method": result.diagnostics.get("area_method"),
            "expected_fraction_definition": (
                "share of frame-weighted available arena area falling in this ring, given the "
                "chaser position observed on each valid frame"
            ),
            "selection_index_definition": (
                "radial_observed_fraction / radial_expected_fraction; 1.0 == chance. NaN where the "
                "null expects fewer than min_expected_count samples in the ring"
            ),
            "closed_loop_caveat": result.diagnostics.get("closed_loop_caveat"),
            "wall_excluded_policy": result.diagnostics.get("wall_excluded_policy"),
        }
    )

    cdf = component.require_group("distance_cdf")
    _write_array(cdf, "cdf_observed_fraction", result.cdf_observed_fraction)
    _write_array(cdf, "cdf_expected_fraction", result.cdf_expected_fraction)
    _write_array(cdf, "cdf_enrichment", result.cdf_enrichment)
    cdf.attrs.update(
        {
            "axis_order": ["epoch", "chaser", "threshold"],
            "threshold_unit": "mm",
            "cdf_enrichment_definition": "cdf_observed_fraction / cdf_expected_fraction",
        }
    )

    per_epoch = component.require_group("per_epoch_chaser")
    _write_array(per_epoch, "valid_frame_count", result.valid_frame_count)
    _write_array(per_epoch, "wall_excluded_frame_count", result.wall_excluded_frame_count)
    _write_array(per_epoch, "near_zone_observed_fraction", result.near_zone_observed_fraction)
    _write_array(per_epoch, "near_zone_expected_fraction", result.near_zone_expected_fraction)
    _write_array(per_epoch, "near_zone_enrichment", result.near_zone_enrichment)
    _write_array(per_epoch, "near_zone_dwell_s", result.near_zone_dwell_s)
    _write_array(per_epoch, "chaser_position_spread_mm", result.chaser_position_spread_mm)
    _write_array(per_epoch, "chaser_path_length_mm", result.chaser_path_length_mm)
    _write_array(per_epoch, "chaser_mean_distance_to_center_mm", result.chaser_mean_distance_to_center_mm)
    _write_array(per_epoch, "chaser_is_moving", result.chaser_is_moving)
    per_epoch.attrs.update(
        {
            "axis_order": ["epoch", "chaser"],
            "near_zone_enrichment_definition": "near_zone_observed_fraction / near_zone_expected_fraction",
            "chaser_is_moving_definition": (
                "RMS spread of the chaser position within the epoch exceeds motion_spread_threshold_mm; "
                "when true the chaser is pursuing and the geometric null is not a behavioral null"
            ),
        }
    )

    controls = component.require_group("control_reference")
    _write_array(controls, "reference_label_bytes", _bytes_array([CONTROL_REFERENCE_LABEL], width=48))
    _write_array(controls, "radial_count", result.control_radial_count)
    _write_array(controls, "radial_observed_fraction", result.control_radial_observed_fraction)
    _write_array(controls, "radial_expected_fraction", result.control_radial_expected_fraction)
    _write_array(controls, "radial_selection_index", result.control_radial_selection_index)
    controls.attrs.update(
        {
            "axis_order": ["epoch", "radial_bin"],
            "purpose": (
                "rings around a fixed dish-centre reference; isolates thigmotaxis and global "
                "spatial bias from genuine chaser-relative structure"
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
            _write_array(summary_group, key, np.asarray([np.nan if value is None else float(value)], dtype=np.float32))
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
        "x_axis_direction": "right",
        "y_axis_direction": "down",
        "pixels_per_mm_projector": result.pixels_per_mm_projector,
        "geometry_status": result.geometry.status,
        "arena_geometry_source": result.geometry.source,
        "arena_shape": result.geometry.shape,
        "arena_center_x_px": result.geometry.center_x_px,
        "arena_center_y_px": result.geometry.center_y_px,
        "arena_radius_px": result.geometry.radius_px,
        "git_commit": git.get("commit_hash"),
        "git_branch": git.get("branch"),
        "git_dirty": git.get("is_dirty"),
        "provenance": {
            "stage": "chaser_radial_occupancy",
            "created_by": "fisheye.analysis.chaser_radial_occupancy",
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
    if write_png:
        source_paths = {
            **source_refs,
            "component": component_path,
            "radial_occupancy": f"{component_path}/radial_occupancy",
            "distance_cdf": f"{component_path}/distance_cdf",
            "control_reference": f"{component_path}/control_reference",
        }
        for artifact_name, renderer, description in (
            (
                RADIAL_DENSITY_PNG_ARTIFACT_NAME,
                render_chaser_radial_occupancy_density_png,
                "Chaser-relative radial occupancy vs geometric null.",
            ),
            (
                SELECTION_INDEX_PNG_ARTIFACT_NAME,
                render_chaser_radial_occupancy_selection_index_png,
                "Chaser-relative radial selection index (observed / geometric null).",
            ),
        ):
            write_png_visualization_artifact(
                component,
                artifact_name,
                renderer(result),
                description=description,
                created_by="fisheye.analysis.chaser_radial_occupancy",
                role="analysis_summary",
                source_paths=source_paths,
                source_runs={"chaser_distance_run": result.chaser_distance_run_name},
                parameters=parameters,
                extra_attrs={
                    "chaser_radial_occupancy_schema_id": SCHEMA_ID,
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
            description="Chaser radial occupancy interactive plot spec.",
            created_by="fisheye.analysis.chaser_radial_occupancy",
            renderer=INTERACTIVE_RENDERER,
            artifact_signature=None,
            snapshot_artifact=SELECTION_INDEX_PNG_ARTIFACT_NAME,
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


def _result_payload(result: ChaserRadialOccupancyResult, *, applied_path: str | None) -> dict[str, Any]:
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
    parser = argparse.ArgumentParser(description="Compute area-normalized radial occupancy around the moving chaser.")
    parser.add_argument("zarr_path", type=Path, help="Analysis zarr archive.")
    parser.add_argument("--chaser-distance-run", default="latest")
    parser.add_argument("--component-name", default=DEFAULT_COMPONENT_NAME)
    parser.add_argument("--radial-bin-width-mm", type=float, default=DEFAULT_RADIAL_BIN_WIDTH_MM)
    parser.add_argument("--radial-bin-edges-mm", default=None, help="Comma-separated edges. Defaults to arena-spanning bins.")
    parser.add_argument("--cdf-thresholds-mm", default=",".join(f"{value:g}" for value in DEFAULT_CDF_THRESHOLDS_MM))
    parser.add_argument("--r-zone-mm", type=float, default=DEFAULT_R_ZONE_MM)
    parser.add_argument("--perimeter-band-mm", type=float, default=DEFAULT_PERIMETER_BAND_MM)
    parser.add_argument("--motion-spread-threshold-mm", type=float, default=DEFAULT_MOTION_SPREAD_THRESHOLD_MM)
    parser.add_argument("--area-cache-step-mm", type=float, default=DEFAULT_AREA_CACHE_STEP_MM)
    parser.add_argument(
        "--min-expected-count",
        type=float,
        default=DEFAULT_MIN_EXPECTED_COUNT,
        help="Suppress the selection index in rings where the null expects fewer than this many samples.",
    )
    parser.add_argument(
        "--settle-trim-s",
        type=float,
        default=None,
        help="Object-repositioning window trimmed from static-configuration epochs. "
             "Defaults to the protocol position_transition_duration_s.",
    )
    parser.add_argument("--apply", action="store_true", help="Write the radial-occupancy component.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Deprecated compatibility flag; immutable published names are never replaced.",
    )
    parser.add_argument("--no-png", action="store_true", help="Skip PNG artifacts.")
    parser.add_argument("--no-interactive-spec", action="store_true", help="Skip interactive plot spec artifact.")
    parser.add_argument("--json", action="store_true", help="Print JSON payload.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = build_chaser_radial_occupancy_result(
        Path(args.zarr_path),
        chaser_distance_run=str(args.chaser_distance_run),
        component_name=str(args.component_name),
        radial_bin_width_mm=float(args.radial_bin_width_mm),
        radial_bin_edges_mm=None if args.radial_bin_edges_mm is None else _parse_float_list(str(args.radial_bin_edges_mm)),
        cdf_thresholds_mm=_parse_float_list(str(args.cdf_thresholds_mm)),
        r_zone_mm=float(args.r_zone_mm),
        perimeter_band_mm=float(args.perimeter_band_mm),
        motion_spread_threshold_mm=float(args.motion_spread_threshold_mm),
        area_cache_step_mm=float(args.area_cache_step_mm),
        min_expected_count=float(args.min_expected_count),
        settle_trim_s=None if args.settle_trim_s is None else float(args.settle_trim_s),
    )
    applied_path = None
    if args.apply:
        applied_path = write_chaser_radial_occupancy_component(
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
        print(f"component_name\t{result.component_name}")
        print(f"chaser_distance_run\t{result.chaser_distance_run_name}")
        print(f"geometry_status\t{result.geometry.status}")
        print(f"status\t{result.status}")
        print(f"qc_warning_count\t{len(result.qc_warnings)}")
        for e_idx, epoch in enumerate(result.epochs):
            for c_idx in range(int(result.chaser_indices.shape[0])):
                moving = "moving" if bool(result.chaser_is_moving[e_idx, c_idx]) else "static"
                observed = float(result.near_zone_observed_fraction[e_idx, c_idx])
                expected = float(result.near_zone_expected_fraction[e_idx, c_idx])
                enrichment = float(result.near_zone_enrichment[e_idx, c_idx])
                print(
                    f"nearzone\t{epoch.label}\tchaser{int(result.chaser_indices[c_idx])}\t{moving}\t"
                    f"obs={observed:.4f}\tnull={expected:.4f}\tenrichment={enrichment:.2f}x"
                )
        if applied_path:
            print(f"applied_path\t{applied_path}")
        else:
            print("dry_run\ttrue")
            print("pass --apply to write the radial-occupancy component")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
