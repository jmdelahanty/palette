"""Logical readers for ``analysis/track_kinematics_runs``.

The current writer stores one subgroup per track under
``tracks/id_<track_id>`` and also writes a newer grouped
``movement/speed/<level>`` surface alongside legacy flat arrays. This module is
the resolver boundary for readers: callers should ask for logical track tables
and speed levels here instead of hard-coding physical paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import zarr


TRACK_KINEMATICS_LAYOUT_HIERARCHICAL_V1 = "hierarchical_v1"
TRACK_KINEMATICS_SPEED_LEVELS = ("raw", "filtered", "smoothed", "averaged")
TRACK_KINEMATICS_SOURCE_SPEED_LEVELS = {
    "raw": "speed_raw",
    "filtered": "speed_filtered",
    "smoothed": "speed_smoothed",
    "averaged": "speed_averaged",
}
TRACK_KINEMATICS_GROUPED_SPEED_LAYOUT = "movement/speed/<level>"


@dataclass(frozen=True)
class TrackKinematicsTrackTables:
    """Logical arrays for one track-kinematics track."""

    run_name: str
    scope: str
    run_path: str
    track_id: int
    track_path: str
    run_attrs: Mapping[str, Any]
    track_attrs: Mapping[str, Any]
    frame_indices: np.ndarray
    speed_mm_by_level: Mapping[str, np.ndarray]
    speed_px_by_level: Mapping[str, np.ndarray]
    frame_path_distance_mm_by_level: Mapping[str, np.ndarray]
    frame_path_distance_px_by_level: Mapping[str, np.ndarray]
    acceleration_mm_by_level: Mapping[str, np.ndarray]
    acceleration_px_by_level: Mapping[str, np.ndarray]
    smoothed_acceleration_mm_by_level: Mapping[str, np.ndarray]
    smoothed_acceleration_px_by_level: Mapping[str, np.ndarray]
    delta_seconds: Optional[np.ndarray]
    transition_valid: Optional[np.ndarray]
    sample_valid: Optional[np.ndarray]
    time_seconds: Optional[np.ndarray]
    heading_degrees: Optional[np.ndarray]
    smoothed_heading_degrees: Optional[np.ndarray]
    delta_heading_degrees: Optional[np.ndarray]
    delta_heading_smoothed_degrees: Optional[np.ndarray]
    angular_velocity_deg_s: Optional[np.ndarray]
    angular_velocity_smoothed_deg_s: Optional[np.ndarray]
    angular_speed_raw_deg_s: Optional[np.ndarray]
    angular_speed_smoothed_deg_s: Optional[np.ndarray]
    detection_source: Optional[np.ndarray]
    positions_mm: Optional[np.ndarray]
    positions_px: Optional[np.ndarray]
    cumulative_path_distance_mm: Optional[np.ndarray]
    cumulative_path_distance_px: Optional[np.ndarray]

    def speed_level_dict(self) -> dict[str, Optional[np.ndarray]]:
        """Return the legacy speed-dict shape expected by bout detectors."""

        out: dict[str, Optional[np.ndarray]] = {"frames": self.frame_indices}
        for level in TRACK_KINEMATICS_SPEED_LEVELS:
            source_level = TRACK_KINEMATICS_SOURCE_SPEED_LEVELS[level]
            out[f"{source_level}_mm"] = self.speed_mm_by_level.get(level)
            out[f"{source_level}_px"] = self.speed_px_by_level.get(level)
            out[f"frame_path_distance_{level}_mm"] = self.frame_path_distance_mm_by_level.get(level)
            out[f"frame_path_distance_{level}_px"] = self.frame_path_distance_px_by_level.get(level)
        out["delta_seconds"] = self.delta_seconds
        out["transition_valid"] = self.transition_valid
        out["sample_valid"] = self.sample_valid
        return out


def _group_attrs(group: zarr.Group) -> dict[str, Any]:
    return dict(group.attrs.asdict() if hasattr(group.attrs, "asdict") else dict(group.attrs))


def _optional_array(group: zarr.Group, name: str) -> Optional[np.ndarray]:
    if name not in group:
        return None
    return np.asarray(group[name][:])


def _require_array(group: zarr.Group, name: str, *, label: str) -> np.ndarray:
    values = _optional_array(group, name)
    if values is None:
        raise ValueError(f"{label} is missing required array '{name}'")
    return values


def _resolve_track_kinematics_parent(root: zarr.Group) -> zarr.Group:
    try:
        return root["analysis"]["track_kinematics_runs"]
    except Exception as exc:
        raise ValueError("No analysis/track_kinematics_runs group found") from exc


def resolve_track_kinematics_run(
    root: zarr.Group,
    *,
    run_name: str = "latest",
    scope: str = "offline",
) -> tuple[zarr.Group, str, str]:
    """Resolve a track-kinematics run group and its canonical path."""

    parent = _resolve_track_kinematics_parent(root)
    if scope not in parent:
        raise ValueError(f"No {scope!r} track_kinematics_runs group found")
    scope_group = parent[scope]
    resolved_name = run_name
    if run_name == "latest":
        resolved_name = scope_group.attrs.get("latest")
        if not resolved_name:
            raise ValueError(f"No latest {scope!r} track_kinematics run found")
    resolved_name = str(resolved_name)
    if resolved_name not in scope_group:
        raise ValueError(f"Track kinematics run {resolved_name!r} not found under {scope!r}")
    run_path = f"analysis/track_kinematics_runs/{scope}/{resolved_name}"
    return scope_group[resolved_name], resolved_name, run_path


def list_track_ids(run_group: zarr.Group) -> list[int]:
    """Return available track ids from a current hierarchical run."""

    if "track_ids" in run_group:
        return [int(value) for value in np.asarray(run_group["track_ids"][:]).tolist()]
    if "tracks" not in run_group:
        return []
    ids: list[int] = []
    for key in run_group["tracks"].group_keys():
        key_str = str(key)
        if key_str.startswith("id_"):
            try:
                ids.append(int(key_str.split("_", 1)[1]))
            except ValueError:
                continue
    return sorted(ids)


def _load_speed_level(
    track_group: zarr.Group,
    *,
    level: str,
) -> tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    """Load speed and path-distance arrays for one logical level.

    Prefer the grouped v2 ``movement/speed/<level>`` surface. Fall back to the
    legacy flat arrays to keep historical runs readable.
    """

    source_level = TRACK_KINEMATICS_SOURCE_SPEED_LEVELS[level]
    grouped = None
    if "movement" in track_group:
        movement = track_group["movement"]
        if "speed" in movement and level in movement["speed"]:
            grouped = movement["speed"][level]

    if grouped is not None:
        speed_mm = _optional_array(grouped, "mm")
        speed_px = _optional_array(grouped, "px")
        path_mm = _optional_array(grouped, "frame_path_distance_mm")
        path_px = _optional_array(grouped, "frame_path_distance_px")
        accel_mm = _optional_array(grouped, "acceleration_mm")
        accel_px = _optional_array(grouped, "acceleration_px")
        smooth_accel_mm = _optional_array(grouped, "smoothed_acceleration_mm")
        smooth_accel_px = _optional_array(grouped, "smoothed_acceleration_px")
    else:
        speed_mm = speed_px = path_mm = path_px = None
        accel_mm = accel_px = smooth_accel_mm = smooth_accel_px = None

    if speed_mm is None:
        speed_mm = _optional_array(track_group, f"{source_level}_mm")
    if speed_px is None:
        speed_px = _optional_array(track_group, f"{source_level}_px")
    if path_mm is None:
        path_mm = _optional_array(track_group, f"frame_path_distance_{level}_mm")
    if path_px is None:
        path_px = _optional_array(track_group, f"frame_path_distance_{level}_px")
    if level == "smoothed":
        if accel_mm is None:
            accel_mm = _optional_array(track_group, "acceleration_mm")
        if accel_px is None:
            accel_px = _optional_array(track_group, "acceleration_px")
        if smooth_accel_mm is None:
            smooth_accel_mm = _optional_array(track_group, "smoothed_acceleration_mm")
        if smooth_accel_px is None:
            smooth_accel_px = _optional_array(track_group, "smoothed_acceleration_px")
    return speed_mm, speed_px, path_mm, path_px, accel_mm, accel_px, smooth_accel_mm, smooth_accel_px


def load_track_kinematics_track(
    root: zarr.Group,
    *,
    run_name: str = "latest",
    scope: str = "offline",
    track_id: int = 0,
    required_speed_levels: Iterable[str] = TRACK_KINEMATICS_SPEED_LEVELS,
) -> TrackKinematicsTrackTables:
    """Load logical arrays for one track from ``analysis/track_kinematics_runs``."""

    requested_speed_levels = tuple(
        dict.fromkeys(str(level).strip() for level in required_speed_levels)
    )
    unsupported = tuple(
        level
        for level in requested_speed_levels
        if level not in TRACK_KINEMATICS_SPEED_LEVELS
    )
    if unsupported:
        supported = ", ".join(TRACK_KINEMATICS_SPEED_LEVELS)
        raise ValueError(
            "Unsupported physical track speed level(s): "
            f"{', '.join(unsupported)}. Expected a subset of: {supported}."
        )

    run_group, resolved_name, run_path = resolve_track_kinematics_run(
        root,
        run_name=run_name,
        scope=scope,
    )
    if "tracks" not in run_group:
        raise ValueError(f"Track kinematics run {resolved_name!r} has no tracks group")
    track_key = f"id_{int(track_id)}"
    tracks = run_group["tracks"]
    if track_key not in tracks:
        available = list_track_ids(run_group)
        raise ValueError(
            f"Track {track_key} not found in track kinematics run {resolved_name!r}; "
            f"available track ids: {available}"
        )

    track_group = tracks[track_key]
    track_path = f"{run_path}/tracks/{track_key}"
    label = track_path
    frame_indices = _require_array(track_group, "frame_indices", label=label).astype(np.int64, copy=False)

    speed_mm_by_level: dict[str, np.ndarray] = {}
    speed_px_by_level: dict[str, np.ndarray] = {}
    frame_path_distance_mm_by_level: dict[str, np.ndarray] = {}
    frame_path_distance_px_by_level: dict[str, np.ndarray] = {}
    acceleration_mm_by_level: dict[str, np.ndarray] = {}
    acceleration_px_by_level: dict[str, np.ndarray] = {}
    smoothed_acceleration_mm_by_level: dict[str, np.ndarray] = {}
    smoothed_acceleration_px_by_level: dict[str, np.ndarray] = {}
    for level in TRACK_KINEMATICS_SPEED_LEVELS:
        (
            speed_mm,
            speed_px,
            path_mm,
            path_px,
            accel_mm,
            accel_px,
            smooth_accel_mm,
            smooth_accel_px,
        ) = _load_speed_level(track_group, level=level)
        if speed_mm is not None:
            speed_mm_by_level[level] = speed_mm
        if speed_px is not None:
            speed_px_by_level[level] = speed_px
        if path_mm is not None:
            frame_path_distance_mm_by_level[level] = path_mm
        if path_px is not None:
            frame_path_distance_px_by_level[level] = path_px
        if accel_mm is not None:
            acceleration_mm_by_level[level] = accel_mm
        if accel_px is not None:
            acceleration_px_by_level[level] = accel_px
        if smooth_accel_mm is not None:
            smoothed_acceleration_mm_by_level[level] = smooth_accel_mm
        if smooth_accel_px is not None:
            smoothed_acceleration_px_by_level[level] = smooth_accel_px

    for required in requested_speed_levels:
        if required not in speed_mm_by_level:
            source_level = TRACK_KINEMATICS_SOURCE_SPEED_LEVELS[required]
            raise ValueError(f"{label} is missing required speed level '{source_level}_mm'")

    return TrackKinematicsTrackTables(
        run_name=resolved_name,
        scope=scope,
        run_path=run_path,
        track_id=int(track_id),
        track_path=track_path,
        run_attrs=_group_attrs(run_group),
        track_attrs=_group_attrs(track_group),
        frame_indices=frame_indices,
        speed_mm_by_level=speed_mm_by_level,
        speed_px_by_level=speed_px_by_level,
        frame_path_distance_mm_by_level=frame_path_distance_mm_by_level,
        frame_path_distance_px_by_level=frame_path_distance_px_by_level,
        acceleration_mm_by_level=acceleration_mm_by_level,
        acceleration_px_by_level=acceleration_px_by_level,
        smoothed_acceleration_mm_by_level=smoothed_acceleration_mm_by_level,
        smoothed_acceleration_px_by_level=smoothed_acceleration_px_by_level,
        delta_seconds=_optional_array(track_group, "delta_seconds"),
        transition_valid=_optional_array(track_group, "transition_valid"),
        sample_valid=_optional_array(track_group, "sample_valid"),
        time_seconds=_optional_array(track_group, "time_seconds"),
        heading_degrees=_optional_array(track_group, "heading_degrees"),
        smoothed_heading_degrees=_optional_array(track_group, "smoothed_heading_degrees"),
        delta_heading_degrees=_optional_array(track_group, "delta_heading_degrees"),
        delta_heading_smoothed_degrees=_optional_array(track_group, "delta_heading_smoothed_degrees"),
        angular_velocity_deg_s=_optional_array(track_group, "angular_velocity_deg_s"),
        angular_velocity_smoothed_deg_s=_optional_array(track_group, "angular_velocity_smoothed_deg_s"),
        angular_speed_raw_deg_s=_optional_array(track_group, "angular_speed_raw_deg_s"),
        angular_speed_smoothed_deg_s=_optional_array(track_group, "angular_speed_smoothed_deg_s"),
        detection_source=_optional_array(track_group, "detection_source"),
        positions_mm=_optional_array(track_group, "positions_mm"),
        positions_px=_optional_array(track_group, "positions_px"),
        cumulative_path_distance_mm=_optional_array(track_group, "cumulative_path_distance_mm"),
        cumulative_path_distance_px=_optional_array(track_group, "cumulative_path_distance_px"),
    )
