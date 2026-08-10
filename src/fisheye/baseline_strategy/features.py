"""Pure feature builders for baseline behavioral strategies."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .contracts import (
    BASELINE_EXPLORATION_EPISODES_TABLE,
    BASELINE_STRATEGY_FEATURES_TABLE,
    METHOD,
    METHOD_VERSION,
    SCHEMA_ID,
    SCHEMA_VERSION,
    StrategyFeatureConfig,
)


def _float(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _mean(values: Iterable[object]) -> float | None:
    finite = [value for item in values if (value := _float(item)) is not None]
    return float(np.mean(finite)) if finite else None


def _median(values: Iterable[object]) -> float | None:
    finite = [value for item in values if (value := _float(item)) is not None]
    return float(np.median(finite)) if finite else None


def _safe_ratio(numerator: object, denominator: object) -> float | None:
    num = _float(numerator)
    den = _float(denominator)
    if num is None or den is None or den == 0:
        return None
    return num / den


def expected_wall_fraction(
    arena_radius_mm: object, wall_band_mm: object
) -> float | None:
    """Uniform-area expectation for a wall band in a circular arena."""

    radius = _float(arena_radius_mm)
    width = _float(wall_band_mm)
    if radius is None or width is None or radius <= 0 or width < 0:
        return None
    inner = max(0.0, radius - min(width, radius))
    return 1.0 - (inner / radius) ** 2


def _identity(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "recording_id": summary.get("recording_id"),
        "acquisition_batch_id": summary.get("acquisition_batch_id"),
        "subject_id": summary.get("subject_id"),
        "track_id": _int(summary.get("track_id")),
        "baseline_window_id": _int(summary.get("baseline_window_id")),
        "baseline_window_label": summary.get("baseline_window_label"),
        "source_export_run_id": (
            summary.get("source_export_run_id") or summary.get("export_run_id")
        ),
        "zarr_path": summary.get("zarr_path"),
    }


def _common_row(table_name: str, summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "table_name": table_name,
        "method": METHOD,
        "method_version": METHOD_VERSION,
        **_identity(summary),
    }


def _sorted_samples(sample_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for source in sample_rows:
        row = dict(source)
        time_s = _float(row.get("relative_time_s"))
        x = _float(row.get("x_arena_mm"))
        y = _float(row.get("y_arena_mm"))
        valid = bool(row.get("position_valid", True)) and bool(
            row.get("sample_valid", True)
        )
        if time_s is None or x is None or y is None or not valid:
            continue
        row["relative_time_s"] = time_s
        row["x_arena_mm"] = x
        row["y_arena_mm"] = y
        rows.append(row)
    rows.sort(
        key=lambda item: (item["relative_time_s"], _int(item.get("source_frame")) or -1)
    )
    return rows


def _sample_active(row: Mapping[str, Any], config: StrategyFeatureConfig) -> bool:
    speed = _float(row.get("speed_mm_s"))
    return speed is not None and speed >= config.active_speed_mm_s


def _sample_boundary_distance(row: Mapping[str, Any], radius: float) -> float | None:
    declared = _float(row.get("distance_to_arena_boundary_mm"))
    if declared is not None:
        return max(0.0, declared)
    x = _float(row.get("x_arena_mm"))
    y = _float(row.get("y_arena_mm"))
    if x is None or y is None:
        return None
    return max(0.0, radius - math.hypot(x, y))


def _sample_wall(
    row: Mapping[str, Any], radius: float, wall_band: float | None
) -> bool | None:
    distance = _sample_boundary_distance(row, radius)
    if distance is not None and wall_band is not None:
        return distance <= wall_band
    declared = row.get("wall")
    return bool(declared) if declared is not None else None


def _grid_cell(x: float, y: float, radius: float, size: int) -> tuple[int, int] | None:
    if radius <= 0 or x * x + y * y > radius * radius:
        return None
    x_index = min(size - 1, max(0, int(math.floor(((x / (2 * radius)) + 0.5) * size))))
    y_index = min(size - 1, max(0, int(math.floor(((y / (2 * radius)) + 0.5) * size))))
    return x_index, y_index


@lru_cache(maxsize=32)
def _accessible_cell_weights(
    size: int, subdivisions: int = 24
) -> dict[tuple[int, int], float]:
    """Approximate each grid cell's accessible fraction in a unit circle."""

    cells: dict[tuple[int, int], float] = {}
    for y_index in range(size):
        for x_index in range(size):
            inside = 0
            for sub_y in range(subdivisions):
                y = -1.0 + 2.0 * (y_index + (sub_y + 0.5) / subdivisions) / size
                for sub_x in range(subdivisions):
                    x = -1.0 + 2.0 * (x_index + (sub_x + 0.5) / subdivisions) / size
                    inside += int(x * x + y * y <= 1.0)
            fraction = inside / float(subdivisions * subdivisions)
            if fraction > 0:
                cells[(x_index, y_index)] = fraction
    return cells


def _occupancy_features(
    rows: Sequence[Mapping[str, Any]], radius: float, size: int
) -> dict[str, Any]:
    accessible_weights = _accessible_cell_weights(size)
    accessible = set(accessible_weights)
    counts = {cell: 0 for cell in accessible}
    ordered_cells: list[tuple[int, int]] = []
    ordered_times: list[float] = []
    for row in rows:
        cell = _grid_cell(
            float(row["x_arena_mm"]), float(row["y_arena_mm"]), radius, size
        )
        if cell not in counts:
            continue
        counts[cell] += 1
        ordered_cells.append(cell)
        ordered_times.append(float(row["relative_time_s"]))
    total = sum(counts.values())
    if total == 0 or not accessible:
        return {
            "occupancy_accessible_cell_count": len(accessible),
            "occupancy_visited_cell_count": 0,
            "occupancy_visited_cell_fraction": None,
            "occupancy_coverage_fraction": None,
            "occupancy_entropy_accessible_normalized": None,
            "occupancy_js_divergence_uniform": None,
            "occupancy_max_cell_fraction": None,
            "latency_to_half_final_coverage_s": None,
        }
    probabilities = np.asarray(
        [counts[cell] / total for cell in sorted(accessible)], dtype=float
    )
    positive = probabilities[probabilities > 0]
    entropy = -float(np.sum(positive * np.log(positive)))
    normalized_entropy = (
        entropy / math.log(len(accessible)) if len(accessible) > 1 else 0.0
    )
    ordered_accessible = sorted(accessible)
    area_weights = np.asarray(
        [accessible_weights[cell] for cell in ordered_accessible], dtype=float
    )
    uniform = area_weights / float(np.sum(area_weights))
    midpoint = 0.5 * (probabilities + uniform)
    kl_p = float(
        np.sum(
            probabilities[probabilities > 0]
            * np.log(probabilities[probabilities > 0] / midpoint[probabilities > 0])
        )
    )
    kl_q = float(np.sum(uniform * np.log(uniform / midpoint)))
    js_divergence = 0.5 * (kl_p + kl_q) / math.log(2.0)
    visited_count = sum(count > 0 for count in counts.values())
    visited_area = sum(
        accessible_weights[cell] for cell, count in counts.items() if count > 0
    )
    half_target = max(1, int(math.ceil(visited_count / 2.0)))
    seen: set[tuple[int, int]] = set()
    latency = None
    first_time = ordered_times[0] if ordered_times else 0.0
    for cell, time_s in zip(ordered_cells, ordered_times):
        seen.add(cell)
        if len(seen) >= half_target:
            latency = time_s - first_time
            break
    return {
        "occupancy_accessible_cell_count": len(accessible),
        "occupancy_visited_cell_count": visited_count,
        "occupancy_visited_cell_fraction": visited_count / len(accessible),
        "occupancy_coverage_fraction": visited_area / float(np.sum(area_weights)),
        "occupancy_entropy_accessible_normalized": normalized_entropy,
        "occupancy_js_divergence_uniform": js_divergence,
        "occupancy_uniform_reference": "circular_accessible_cell_area_subsample_v1",
        "occupancy_max_cell_fraction": float(np.max(probabilities)),
        "latency_to_half_final_coverage_s": latency,
    }


def _dominant_dwell_features(
    rows: Sequence[Mapping[str, Any]], radius: float, config: StrategyFeatureConfig
) -> tuple[dict[str, Any], tuple[int, int] | None]:
    dwell_rows = [row for row in rows if not _sample_active(row, config)] or list(rows)
    cells = [
        _grid_cell(
            float(row["x_arena_mm"]),
            float(row["y_arena_mm"]),
            radius,
            config.dwell_grid_size,
        )
        for row in dwell_rows
    ]
    cells = [cell for cell in cells if cell is not None]
    if not cells:
        return {
            "dominant_dwell_cell_fraction": None,
            "dominant_to_second_dwell_ratio": None,
            "dominant_dwell_visit_count": None,
            "dominant_dwell_center_distance_norm": None,
        }, None
    counts: dict[tuple[int, int], int] = {}
    for cell in cells:
        counts[cell] = counts.get(cell, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    dominant, dominant_count = ranked[0]
    second_count = ranked[1][1] if len(ranked) > 1 else 0
    visits = 0
    previous = None
    for cell in cells:
        if cell == dominant and previous != dominant:
            visits += 1
        previous = cell
    cell_width = 2 * radius / config.dwell_grid_size
    center_x = -radius + (dominant[0] + 0.5) * cell_width
    center_y = -radius + (dominant[1] + 0.5) * cell_width
    return {
        "dominant_dwell_cell_fraction": dominant_count / len(cells),
        "dominant_to_second_dwell_ratio": dominant_count / max(1, second_count),
        "dominant_dwell_visit_count": visits,
        "dominant_dwell_center_distance_norm": math.hypot(center_x, center_y) / radius,
    }, dominant


def derive_exploration_episodes(
    summary_row: Mapping[str, Any],
    sample_rows: Sequence[Mapping[str, Any]],
    *,
    config: StrategyFeatureConfig | None = None,
    dominant_dwell_cell: tuple[int, int] | None = None,
) -> list[dict[str, Any]]:
    """Derive active progression episodes from portable baseline samples."""

    config = config or StrategyFeatureConfig()
    config.validate()
    rows = _sorted_samples(sample_rows)
    radius = _float(summary_row.get("arena_radius_mm"))
    wall_band = _float(summary_row.get("wall_band_mm"))
    if radius is None or radius <= 0 or len(rows) < 2:
        return []
    active = [_sample_active(row, config) for row in rows]
    segments: list[tuple[int, int]] = []
    start = None
    for index, is_active in enumerate(active):
        if is_active and start is None:
            start = index
        gap_break = (
            start is not None
            and index > start
            and float(rows[index]["relative_time_s"])
            - float(rows[index - 1]["relative_time_s"])
            > config.max_episode_gap_s
        )
        if start is not None and (not is_active or gap_break):
            end = index - 1
            if gap_break and is_active:
                segments.append((start, end))
                start = index
            else:
                segments.append((start, end))
                start = None
    if start is not None:
        segments.append((start, len(rows) - 1))

    output: list[dict[str, Any]] = []
    for start_index, end_index in segments:
        points = rows[start_index : end_index + 1]
        start_s = float(points[0]["relative_time_s"])
        end_s = float(points[-1]["relative_time_s"])
        duration = max(0.0, end_s - start_s)
        if duration < config.min_episode_duration_s:
            continue
        xy = np.asarray(
            [[float(row["x_arena_mm"]), float(row["y_arena_mm"])] for row in points],
            dtype=float,
        )
        differences = np.diff(xy, axis=0)
        lengths = np.sqrt(np.sum(differences**2, axis=1))
        path_length = float(np.sum(lengths))
        displacement = float(np.linalg.norm(xy[-1] - xy[0]))
        center_distance = np.sqrt(np.sum(xy**2, axis=1))
        wall_flags = [
            bool(wall_value)
            for row in points
            if (wall_value := _sample_wall(row, radius, wall_band)) is not None
        ]
        tangential = []
        for point, delta, length in zip(xy[:-1], differences, lengths):
            radius_at_point = float(np.linalg.norm(point))
            if length <= 0 or radius_at_point <= 0:
                continue
            tangential.append(
                abs(float(point[0] * delta[1] - point[1] * delta[0]))
                / (radius_at_point * length)
            )
        origin_index = max(0, start_index - 1)
        destination_index = min(len(rows) - 1, end_index + 1)
        origin_cell = _grid_cell(
            float(rows[origin_index]["x_arena_mm"]),
            float(rows[origin_index]["y_arena_mm"]),
            radius,
            config.dwell_grid_size,
        )
        destination_cell = _grid_cell(
            float(rows[destination_index]["x_arena_mm"]),
            float(rows[destination_index]["y_arena_mm"]),
            radius,
            config.dwell_grid_size,
        )
        mean_tangential = float(np.mean(tangential)) if tangential else None
        output.append(
            {
                **_common_row(BASELINE_EXPLORATION_EPISODES_TABLE, summary_row),
                "episode_id": len(output),
                "episode_start_s": start_s,
                "episode_end_s": end_s,
                "episode_duration_s": duration,
                "sample_count": len(points),
                "path_length_mm": path_length,
                "net_displacement_mm": displacement,
                "tortuosity": path_length / displacement if displacement > 0 else None,
                "minimum_center_distance_mm": float(np.min(center_distance)),
                "maximum_inward_excursion_mm": radius - float(np.min(center_distance)),
                "wall_sample_fraction": float(np.mean(wall_flags))
                if wall_flags
                else None,
                "mean_tangential_alignment": mean_tangential,
                "wall_following": bool(
                    mean_tangential is not None
                    and mean_tangential >= config.wall_following_tangential_threshold
                    and bool(wall_flags)
                    and float(np.mean(wall_flags)) >= 0.5
                ),
                "origin_dominant_dwell_zone": (
                    dominant_dwell_cell is not None
                    and origin_cell == dominant_dwell_cell
                ),
                "destination_dominant_dwell_zone": (
                    dominant_dwell_cell is not None
                    and destination_cell == dominant_dwell_cell
                ),
                "returned_to_dominant_dwell_zone": (
                    dominant_dwell_cell is not None
                    and origin_cell == dominant_dwell_cell
                    and destination_cell == dominant_dwell_cell
                ),
                "path_length_method": "portable_sample_xy_chord_sum",
                "boundary_distance_method": (
                    points[0].get("boundary_distance_method")
                    or summary_row.get("boundary_distance_method")
                    or "circle_radius_minus_center_distance_v1"
                ),
            }
        )
    return output


def _temporal_features(
    time_bin_rows: Sequence[Mapping[str, Any]], radius: float | None, fraction: float
) -> dict[str, Any]:
    rows = sorted(
        (dict(row) for row in time_bin_rows),
        key=lambda row: _int(row.get("time_bin_index")) or 0,
    )
    if not rows:
        return {"time_bin_count": 0}
    edge_count = max(1, int(math.ceil(len(rows) * fraction)))
    output: dict[str, Any] = {"time_bin_count": len(rows)}
    metrics = {
        "wall_fraction": "wall_fraction",
        "mean_speed_mm_s": "mean_speed_mm_s",
        "distance_travelled_mm": "distance_travelled_mm",
        "mean_center_distance_mm": "mean_center_distance_mm",
        "bout_count": "bout_count",
    }
    for output_name, source_name in metrics.items():
        early = _mean(row.get(source_name) for row in rows[:edge_count])
        late = _mean(row.get(source_name) for row in rows[-edge_count:])
        if (
            radius is not None
            and radius > 0
            and source_name == "mean_center_distance_mm"
        ):
            early = early / radius if early is not None else None
            late = late / radius if late is not None else None
            output_name = "center_distance_norm"
        output[f"{output_name}_early"] = early
        output[f"{output_name}_late"] = late
        output[f"{output_name}_delta_late_minus_early"] = (
            late - early if early is not None and late is not None else None
        )
        x_values = []
        y_values = []
        for row in rows:
            value = _float(row.get(source_name))
            start = _float(row.get("relative_start_s"))
            end = _float(row.get("relative_end_s"))
            if value is None or start is None or end is None:
                continue
            if (
                radius is not None
                and radius > 0
                and source_name == "mean_center_distance_mm"
            ):
                value /= radius
            x_values.append(0.5 * (start + end))
            y_values.append(value)
        if len(x_values) >= 2 and max(x_values) > min(x_values):
            normalized_x = (np.asarray(x_values) - min(x_values)) / (
                max(x_values) - min(x_values)
            )
            output[f"{output_name}_slope_per_baseline"] = float(
                np.polyfit(normalized_x, np.asarray(y_values), 1)[0]
            )
        else:
            output[f"{output_name}_slope_per_baseline"] = None
    return output


def derive_baseline_strategy_features(
    summary_row: Mapping[str, Any],
    time_bin_rows: Sequence[Mapping[str, Any]] = (),
    sample_rows: Sequence[Mapping[str, Any]] = (),
    *,
    config: StrategyFeatureConfig | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build one factor-ready recording feature row plus progression episodes."""

    config = config or StrategyFeatureConfig()
    config.validate()
    radius = _float(summary_row.get("arena_radius_mm"))
    wall_band = _float(summary_row.get("wall_band_mm"))
    wall_fraction = _float(summary_row.get("wall_fraction"))
    geometry_type = str(summary_row.get("experimental_area_geometry_type") or "circle")
    expected = _float(summary_row.get("expected_uniform_wall_fraction"))
    if expected is None and geometry_type == "circle":
        expected = expected_wall_fraction(radius, wall_band)
    row = {
        **_common_row(BASELINE_STRATEGY_FEATURES_TABLE, summary_row),
        "sample_features_available": False,
        "time_bin_features_available": bool(time_bin_rows),
        "duration_s": _float(summary_row.get("duration_s")),
        "tracking_dropout_fraction": _float(
            summary_row.get("tracking_dropout_fraction")
        ),
        "mean_speed_mm_s": _float(summary_row.get("mean_speed_mm_s")),
        "median_speed_mm_s": _float(summary_row.get("median_speed_mm_s")),
        "p95_speed_mm_s": _float(summary_row.get("p95_speed_mm_s")),
        "total_path_mm": _float(summary_row.get("total_path_mm")),
        "path_per_min_mm": _safe_ratio(
            summary_row.get("total_path_mm"),
            (_float(summary_row.get("duration_s")) or 0.0) / 60.0,
        ),
        "bout_count": _int(summary_row.get("bout_count")),
        "bout_rate_per_min": _float(summary_row.get("bout_rate_per_min")),
        "wall_fraction": wall_fraction,
        "wall_band_mm": wall_band,
        "experimental_area_geometry_type": geometry_type,
        "boundary_distance_method": (
            summary_row.get("boundary_distance_method")
            or "circle_radius_minus_center_distance_v1"
        ),
        "wall_fraction_denominator": (
            summary_row.get("wall_fraction_denominator") or "valid_position_frames"
        ),
        "active_wall_fraction_denominator": "active_valid_portable_samples",
        "expected_uniform_wall_fraction": expected,
        "wall_enrichment_ratio": _safe_ratio(wall_fraction, expected),
        "wall_enrichment_log2": (
            math.log2(wall_fraction / expected)
            if wall_fraction is not None
            and expected is not None
            and wall_fraction > 0
            and expected > 0
            else None
        ),
        "mean_center_distance_norm": _float(
            summary_row.get("mean_center_distance_norm")
        ),
        "source_spatial_entropy_normalized": _float(
            summary_row.get("spatial_entropy_normalized")
        ),
        "source_quadrant_entropy_normalized": _float(
            summary_row.get("quadrant_entropy_normalized")
        ),
        "source_spatial_max_cell_fraction": _float(
            summary_row.get("spatial_max_cell_fraction")
        ),
        "source_quadrant_max_fraction": _float(
            summary_row.get("quadrant_max_fraction")
        ),
        "active_speed_threshold_mm_s": config.active_speed_mm_s,
        "spatial_grid_size": config.spatial_grid_size,
        "dwell_grid_size": config.dwell_grid_size,
    }
    row.update(_temporal_features(time_bin_rows, radius, config.early_late_fraction))

    samples = _sorted_samples(sample_rows)
    episodes: list[dict[str, Any]] = []
    if radius is not None and radius > 0 and len(samples) >= config.min_sample_count:
        row["sample_features_available"] = True
        active_flags = np.asarray(
            [_sample_active(sample, config) for sample in samples], dtype=bool
        )
        row["valid_sample_count"] = len(samples)
        row["active_sample_fraction"] = float(np.mean(active_flags))
        direct_boundary_distance = any(
            _float(sample.get("distance_to_arena_boundary_mm")) is not None
            for sample in samples
        )
        row["boundary_distance_sample_source"] = (
            "exported_distance_to_arena_boundary_mm"
            if direct_boundary_distance
            else "derived_from_circular_radius_and_position"
        )
        row["active_wall_fraction"] = _mean(
            wall_value
            for sample, active in zip(samples, active_flags)
            if active
            and (wall_value := _sample_wall(sample, radius, wall_band)) is not None
        )
        row.update(_occupancy_features(samples, radius, config.spatial_grid_size))
        dwell_features, dominant_cell = _dominant_dwell_features(
            samples, radius, config
        )
        row.update(dwell_features)
        episodes = derive_exploration_episodes(
            summary_row,
            samples,
            config=config,
            dominant_dwell_cell=dominant_cell,
        )
        row["exploration_episode_count"] = len(episodes)
        row["wall_following_episode_fraction"] = _mean(
            episode.get("wall_following") for episode in episodes
        )
        outbound = [
            episode for episode in episodes if episode["origin_dominant_dwell_zone"]
        ]
        row["dominant_dwell_excursion_count"] = len(outbound)
        row["dominant_dwell_return_fraction"] = (
            float(
                np.mean(
                    [episode["returned_to_dominant_dwell_zone"] for episode in outbound]
                )
            )
            if outbound
            else None
        )
        row["median_episode_path_length_mm"] = (
            float(np.median([episode["path_length_mm"] for episode in episodes]))
            if episodes
            else None
        )
        row["median_episode_tortuosity"] = _median(
            episode.get("tortuosity") for episode in episodes
        )
    else:
        row.update(
            {
                "valid_sample_count": len(samples),
                "active_sample_fraction": None,
                "active_wall_fraction": None,
                "boundary_distance_sample_source": "unavailable",
                "exploration_episode_count": None,
                "wall_following_episode_fraction": None,
                "dominant_dwell_excursion_count": None,
                "dominant_dwell_return_fraction": None,
            }
        )

    dropout = row["tracking_dropout_fraction"]
    valid_fraction = 1.0 - min(1.0, max(0.0, dropout if dropout is not None else 0.0))
    if radius is None or radius <= 0:
        status, reason = "invalid", "missing_positive_arena_radius_mm"
    elif valid_fraction < config.min_valid_position_fraction:
        status, reason = "invalid", "tracking_coverage_below_threshold"
    else:
        status, reason = "complete", None
    row["feature_status"] = status
    row["feature_reason"] = reason
    return row, episodes


__all__ = [
    "derive_baseline_strategy_features",
    "derive_exploration_episodes",
    "expected_wall_fraction",
]
