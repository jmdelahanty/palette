"""Core-behavior component with deferred Zarr projections and Polars queries.

Polars cannot scan Zarr directly.  This component therefore keeps the Zarr
source unopened until an analysis is selected, reads only that analysis' array
projection (and selected row interval), then exposes the in-memory projection
as a ``polars.LazyFrame``.  Exported Parquet data uses ``scan_parquet`` and is
lazy from storage through collection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import polars as pl

from fisheye.analysis.chaser_distance_io import (
    ChaserDistanceReadError,
    load_chaser_distance_run,
)
from fisheye.analysis.swim_bout_io import (
    SwimBoutIOError,
    discover_swim_bout_candidates,
    load_swim_bout_events,
    structured_records_to_dicts,
)
from fisheye.analysis.tail_kinematics_io import (
    TAIL_SCALAR_SERIES,
    catalog_tail_kinematics_run,
    discover_tail_kinematics_run_options,
    load_tail_kinematics_window,
)
from fisheye.analytics_exports.baseline import is_baseline_label
from fisheye.analysis_workflows.validated_recording_behavior_source import (
    ValidatedCapabilityUnavailableError,
    ValidatedRecordingBehaviorSource,
    ValidatedRecordingBehaviorSourceError,
)
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.utils.view_zarr_visualization import load_png_artifact_bytes
from fisheye.visualization.eye_angle_timeseries import (
    catalog_eye_angle_timeseries_data,
    discover_eye_angle_run_options,
    load_eye_angle_timeseries_window,
)

from .registry import InteractiveSpecOption
from .common import png_bytes_to_markdown_image

TRACK_KINEMATICS_RENDERER = "palette-track-kinematics-summary-v1"

SPEED_SERIES_PREFIXES = ("speed_", "acceleration_", "smoothed_acceleration_")
HEADING_SERIES_TOKENS = ("heading", "angular_velocity", "angular_speed")
PATH_SERIES_TOKENS = ("cumulative_path_distance", "frame_path_distance")
DISTANCE_TRAVELED_LEVELS = (
    ("mm", "cumulative_path_distance_mm", "frame_path_distance_smoothed_mm"),
    ("px", "cumulative_path_distance_px", "frame_path_distance_smoothed_px"),
)


@dataclass(frozen=True)
class BaselineInterval:
    label: str
    start_s: float
    stop_s: float


@dataclass(frozen=True)
class CoreBehaviorProjection:
    analysis_id: str
    frame: pl.LazyFrame
    columns: tuple[str, ...]
    source_paths: tuple[str, ...]
    start_s: float
    stop_s: float
    row_count: int
    load_duration_ms: float
    note: str
    related_frames: Mapping[str, pl.LazyFrame] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CoreBehaviorOption:
    """Selectable core source independent of optional visualization artifacts."""

    zarr_path: Path
    run_path: str
    run_name: str
    label: str
    track_id: int
    source_paths: Mapping[str, str]
    attrs: Mapping[str, Any]
    interactive_option: InteractiveSpecOption | None = None
    validated_bundle_path: str | None = None
    validated_bundle_sha256: str | None = None


def scan_export_parquet(
    paths: str | Path | Sequence[str | Path],
    *,
    columns: Sequence[str] | None = None,
) -> pl.LazyFrame:
    """Return a true lazy scan for an exported Parquet table or partition."""

    if isinstance(paths, (str, Path)):
        source: str | list[str] = str(paths)
    else:
        source = [str(path) for path in paths]
    lazy = pl.scan_parquet(source)
    return lazy.select(list(columns)) if columns is not None else lazy


def is_core_behavior_option(option: InteractiveSpecOption | CoreBehaviorOption) -> bool:
    return (
        isinstance(option, CoreBehaviorOption)
        or option.renderer == TRACK_KINEMATICS_RENDERER
    )


def _normal_path(value: object) -> str:
    return "/".join(part for part in str(value or "").strip("/").split("/") if part)


def _source_paths(option: InteractiveSpecOption | CoreBehaviorOption) -> dict[str, str]:
    if isinstance(option, CoreBehaviorOption):
        return dict(option.source_paths)
    raw = option.spec.get("source_paths")
    if not isinstance(raw, Mapping):
        return {}
    return {
        str(key): _normal_path(value)
        for key, value in raw.items()
        if _normal_path(value)
    }


def _core_option_from_spec(option: InteractiveSpecOption) -> CoreBehaviorOption:
    source_paths = _source_paths(option)
    # Immutable visualization snapshots own the spec artifact, but the core
    # behavior rowset remains the exact track run named by the spec's source
    # paths. Do not mistake the random render snapshot name for track lineage.
    source_run_path = source_paths.get("run") or _normal_path(option.run_path)
    return CoreBehaviorOption(
        zarr_path=option.zarr_path,
        run_path=source_run_path,
        run_name=option.run_name or source_run_path.split("/")[-1],
        label=option.label,
        track_id=int(option.spec.get("track_id") or 0),
        source_paths=source_paths,
        attrs=option.attrs,
        interactive_option=option,
    )


def validated_core_behavior_option(
    source: ValidatedRecordingBehaviorSource,
    *,
    validate_current_source: bool = True,
) -> CoreBehaviorOption:
    """Describe the exact provider-motion track selected by one bundle."""

    if type(validate_current_source) is not bool:
        raise TypeError("validate_current_source must be the exact boolean")
    if validate_current_source:
        catalog = source.provider_motion_catalog()
        run_path = catalog.run_path
        track_id = catalog.track_id
        manifest_sha256 = catalog.manifest_sha256
        verification_digest = catalog.verification_digest
        source_paths: dict[str, str] = {"run": run_path}
        aliases = {"source_acquisition_frame_index": "frame_indices"}
        for path in catalog.sample_array_paths:
            source_paths[aliases.get(path, path)] = f"{run_path}/{path}"
    else:
        capability = source.require_capability(
            "provider_motion",
            expected_binding_scope="source_bindings",
        )
        binding = capability.binding.get("source")
        if not isinstance(binding, Mapping):
            raise ValidatedRecordingBehaviorSourceError(
                "Provider-motion bundle binding lacks its exact source."
            )
        run_path = binding.get("run_path")
        track_id = binding.get("track_id")
        manifest_sha256 = binding.get("manifest_sha256")
        verification_digest = binding.get("verification_digest")
        if (
            type(run_path) is not str
            or not run_path
            or type(track_id) is not int
            or track_id < 0
            or type(manifest_sha256) is not str
            or type(verification_digest) is not str
        ):
            raise ValidatedRecordingBehaviorSourceError(
                "Provider-motion bundle binding is invalid."
            )
        source_paths = {"run": run_path}
    return CoreBehaviorOption(
        zarr_path=source.analysis_zarr,
        run_path=run_path,
        run_name=run_path.rsplit("/", 1)[-1],
        label=(
            f"validated bundle | track {track_id} | "
            f"{source.bundle_sha256[:12]}"
        ),
        track_id=track_id,
        source_paths=MappingProxyType(source_paths),
        attrs=MappingProxyType(
            {
                "source_mode": "validated_recording_behavior_bundle_v1",
                "bundle_path": str(source.bundle_path),
                "bundle_sha256": source.bundle_sha256,
                "provider_motion_manifest_sha256": manifest_sha256,
                "provider_motion_verification_digest": verification_digest,
                "current_source_validation": (
                    "provider_boundary"
                    if validate_current_source
                    else "deferred_until_provider_selection"
                ),
            }
        ),
        validated_bundle_path=str(source.bundle_path),
        validated_bundle_sha256=source.bundle_sha256,
    )


def _track_source_paths_from_group(
    run_path: str,
    run_group: Any,
    track_id: int,
) -> dict[str, str]:
    track_path = f"{run_path}/tracks/id_{int(track_id)}"
    track_group = run_group[f"tracks/id_{int(track_id)}"]
    paths: dict[str, str] = {"run": run_path, "track": track_path}
    flat_names = (
        "time_seconds",
        "frame_indices",
        "positions_px",
        "positions_mm",
        "speed_raw_px",
        "speed_raw_mm",
        "speed_filtered_px",
        "speed_filtered_mm",
        "speed_smoothed_px",
        "speed_smoothed_mm",
        "speed_averaged_px",
        "speed_averaged_mm",
        "acceleration_px",
        "acceleration_mm",
        "smoothed_heading_degrees",
        "smoothed_acceleration_px",
        "smoothed_acceleration_mm",
        "delta_heading_degrees",
        "angular_velocity_deg_s",
        "angular_velocity_raw_deg_s",
        "angular_speed_raw_deg_s",
        "delta_heading_smoothed_degrees",
        "angular_velocity_smoothed_deg_s",
        "angular_speed_smoothed_deg_s",
        "delta_frames",
        "delta_seconds",
        "frame_path_distance_raw_px",
        "frame_path_distance_filtered_px",
        "frame_path_distance_smoothed_px",
        "cumulative_path_distance_px",
        "frame_path_distance_raw_mm",
        "frame_path_distance_filtered_mm",
        "frame_path_distance_smoothed_mm",
        "cumulative_path_distance_mm",
        "sample_valid",
        "sample_reason_code",
        "transition_valid",
        "transition_reason_code",
    )
    for name in flat_names:
        if name in track_group:
            paths[name] = f"{track_path}/{name}"
    movement = track_group.get("movement")
    speed_parent = movement.get("speed") if movement is not None else None
    if speed_parent is not None:
        for level in ("raw", "filtered", "smoothed", "averaged"):
            if level not in speed_parent:
                continue
            level_group = speed_parent[level]
            for unit in ("px", "mm"):
                if unit in level_group:
                    paths.setdefault(
                        f"speed_{level}_{unit}",
                        f"{track_path}/movement/speed/{level}/{unit}",
                    )
    path_parent = movement.get("frame_path_distance") if movement is not None else None
    if path_parent is not None:
        for level in ("raw", "filtered", "smoothed"):
            if level not in path_parent:
                continue
            level_group = path_parent[level]
            for unit in ("px", "mm"):
                if unit in level_group:
                    paths.setdefault(
                        f"frame_path_distance_{level}_{unit}",
                        f"{track_path}/movement/frame_path_distance/{level}/{unit}",
                    )
    return paths


def _track_run_groups(
    parent: Any, parent_path: str, *, depth: int = 0
) -> list[tuple[str, Any]]:
    if parent is None or depth > 2:
        return []
    if "tracks" in parent:
        return [(parent_path, parent)]
    rows: list[tuple[str, Any]] = []
    group_keys = getattr(parent, "group_keys", None)
    if not callable(group_keys):
        return rows
    for name in group_keys():
        child = parent[name]
        rows.extend(_track_run_groups(child, f"{parent_path}/{name}", depth=depth + 1))
    return rows


def discover_core_behavior_options(
    zarr_path: Path | str,
    interactive_options: Sequence[InteractiveSpecOption] = (),
    *,
    legacy_eye_angle_compatibility: bool = False,
) -> list[CoreBehaviorOption]:
    """Discover canonical core runs even when no visualization spec was persisted."""

    if type(legacy_eye_angle_compatibility) is not bool:
        raise TypeError("legacy_eye_angle_compatibility must be an exact bool")

    archive = Path(zarr_path)
    options = [
        _core_option_from_spec(option)
        for option in interactive_options
        if is_core_behavior_option(option)
    ]
    seen = {(option.run_path, option.track_id) for option in options}
    root = open_zarr_root(archive, mode="r")
    parent_path = "analysis/track_kinematics_runs"
    parent = root.get(parent_path)
    latest_paths: set[str] = set()
    if parent is not None:
        for key in ("latest_complete", "latest"):
            value = str(parent.attrs.get(key) or "").strip("/")
            if value:
                latest_paths.add(f"{parent_path}/{value}")
        for run_path, run_group in _track_run_groups(parent, parent_path):
            status = str(
                run_group.attrs.get("palette_run_completion_status") or ""
            ).lower()
            if status and status != "complete":
                continue
            tracks = run_group.get("tracks")
            if tracks is None:
                continue
            for track_name in sorted(tracks.group_keys()):
                if not str(track_name).startswith("id_"):
                    continue
                try:
                    track_id = int(str(track_name).split("_", 1)[1])
                except ValueError:
                    continue
                if (run_path, track_id) in seen:
                    continue
                run_name = run_path.split("/")[-1]
                is_latest = run_path in latest_paths
                options.append(
                    CoreBehaviorOption(
                        zarr_path=archive,
                        run_path=run_path,
                        run_name=run_name,
                        label=(
                            f"{run_name} | track {track_id}"
                            f"{' | latest' if is_latest else ''} | canonical arrays"
                        ),
                        track_id=track_id,
                        source_paths=_track_source_paths_from_group(
                            run_path, run_group, track_id
                        ),
                        attrs=dict(run_group.attrs),
                    )
                )
                seen.add((run_path, track_id))
    if not options:
        eye_options = discover_eye_angle_run_options(
            archive,
            legacy_compatibility=legacy_eye_angle_compatibility,
        )
        if eye_options:
            eye = eye_options[0]
            options.append(
                CoreBehaviorOption(
                    zarr_path=archive,
                    run_path=eye.run_path,
                    run_name=eye.run_name,
                    label=f"{eye.label} | eye-angle capability",
                    track_id=0,
                    source_paths={},
                    attrs=eye.attrs,
                )
            )
        else:
            tail_options = discover_tail_kinematics_run_options(root)
            if tail_options:
                tail = tail_options[0]
                options.append(
                    CoreBehaviorOption(
                        zarr_path=archive,
                        run_path=tail.run_path,
                        run_name=tail.run_name,
                        label=f"{tail.label} | tail-kinematics capability",
                        track_id=0,
                        source_paths={},
                        attrs=tail.attrs,
                    )
                )
    return sorted(
        options,
        key=lambda item: (
            0 if "latest" in item.label else 1,
            0 if item.interactive_option is not None else 1,
            item.run_path,
            item.track_id,
        ),
    )


def _finite_bounds(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 0.0
    return float(finite[0]), float(finite[-1])


def _structured_lazy(records: np.ndarray) -> pl.LazyFrame:
    rows = structured_records_to_dicts(np.asarray(records))
    return pl.from_dicts(rows).lazy() if rows else pl.DataFrame().lazy()


def _plotly_columns(
    frame: pl.DataFrame,
    columns: Iterable[str],
) -> dict[str, np.ndarray]:
    """Expose bounded Polars columns to Plotly without converting to pandas."""

    return {
        name: frame.get_column(name).to_numpy()
        for name in columns
        if name in frame.columns
    }


def _is_physical_speed_column(name: str) -> bool:
    return (
        name.startswith("speed_")
        and (name.endswith("_mm") or name.endswith("_px"))
        and "acceleration" not in name
        and "frame_path_distance" not in name
    )


def _first_finite_metric_column(
    frame: pl.DataFrame,
    candidates: Sequence[str],
) -> str | None:
    """Choose the preferred persisted metric that contains usable values."""

    for name in candidates:
        if name not in frame.columns:
            continue
        values = frame.get_column(name).cast(pl.Float64, strict=False).to_numpy()
        if np.isfinite(values).any():
            return name
    return None


def _swim_bout_distribution_specs(
    frame: pl.DataFrame,
) -> tuple[tuple[str, str, str], ...]:
    """Resolve compatible bout metrics in calibrated-to-pixel preference order."""

    metric_families = (
        (
            ("duration_s", "bout_duration_s"),
            "Swim-bout duration distribution",
            "Duration (s)",
        ),
        (
            (
                "path_length_mm",
                "distance_mm",
                "distance",
                "path_length_bl",
                "distance_bl",
                "path_length_px",
                "distance_px",
            ),
            "Swim-bout distance distribution",
            None,
        ),
        (
            (
                "mean_speed_mm_s",
                "mean_speed",
                "mean_speed_bl_s",
                "mean_speed_px_s",
            ),
            "Swim-bout mean-speed distribution",
            None,
        ),
    )
    unit_labels = {
        "path_length_mm": "Distance (mm)",
        "distance_mm": "Distance (mm)",
        "distance": "Distance (mm)",
        "path_length_bl": "Distance (body lengths)",
        "distance_bl": "Distance (body lengths)",
        "path_length_px": "Distance (px)",
        "distance_px": "Distance (px)",
        "mean_speed_mm_s": "Mean speed (mm/s)",
        "mean_speed": "Mean speed (mm/s)",
        "mean_speed_bl_s": "Mean speed (body lengths/s)",
        "mean_speed_px_s": "Mean speed (px/s)",
    }
    specs: list[tuple[str, str, str]] = []
    for candidates, title, fixed_label in metric_families:
        column = _first_finite_metric_column(frame, candidates)
        if column is not None:
            specs.append((column, title, fixed_label or unit_labels[column]))
    return tuple(specs)


def _eye_provenance_summary(attrs: Mapping[str, Any]) -> dict[str, Any]:
    scalar_keys = (
        "schema_id",
        "schema_version",
        "layout",
        "method",
        "method_version",
        "row_axis",
        "fps",
        "preferred_angle_family",
        "preferred_eye_axis",
        "body_frame_estimator",
        "body_frame_angle_convention",
        "body_frame_coordinate_space",
        "angle_sign_convention",
        "angle_zero",
        "axis_ambiguity_resolution",
        "gaze_angle_source",
        "angle_smoothing_algorithm",
        "angle_smoothing_window_frames",
        "source_subject_shape_run",
        "source_refined_subject_masks_run",
        "source_keypoint_run",
        "source_detection_success_path",
        "valid_detection_fraction",
        "valid_frame_fraction",
        "lineage_hash",
        "palette_run_completion_status",
        "palette_run_completed_at_utc",
    )
    summary = {key: attrs.get(key) for key in scalar_keys if key in attrs}
    for key in (
        "eye_angle_algorithm_contract",
        "eye_angle_source_contracts",
        "eye_angle_timing_summary",
        "physical_storage_layout",
    ):
        value = attrs.get(key)
        if isinstance(value, Mapping):
            summary[key] = dict(value)
    return summary


def _tail_provenance_summary(
    attrs: Mapping[str, Any],
    source_shape_attrs: Mapping[str, Any],
) -> dict[str, Any]:
    scalar_keys = (
        "schema_id",
        "schema_version",
        "method",
        "method_version",
        "row_axis",
        "source_subject_shape_run",
        "source_subject_shape_path",
        "source_refined_subject_masks_run",
        "source_tail_geometry_kind",
        "body_frame_convention",
        "body_frame_source",
        "tail_angle_reference_axis",
        "tail_angle_positive_direction",
        "tail_angle_units_primary",
        "tail_sample_domain",
        "tail_angle_sample_count",
        "source_geometry_tail_sample_count",
        "curvature_source",
        "frame_index_source",
        "materialization_mode",
        "compute_kernel",
        "execution_backend",
        "worker_count_effective",
        "effective_block_rows",
        "effective_output_shard_rows",
        "palette_run_completion_status",
        "palette_run_completed_at_utc",
    )
    summary = {key: attrs.get(key) for key in scalar_keys if key in attrs}
    for key in ("source_refs", "provenance", "physical_storage_layout"):
        value = attrs.get(key)
        if isinstance(value, Mapping):
            summary[key] = dict(value)
    source_summary_keys = (
        "schema_id",
        "schema_version",
        "method",
        "method_version",
        "tail_sample_count",
        "spline_method",
        "spline_smoothing",
        "spline_degree",
        "body_frame_schema_id",
        "source_refined_subject_masks_run",
        "source_refined_keypoints_run",
        "palette_run_completion_status",
    )
    source_summary = {
        key: source_shape_attrs.get(key)
        for key in source_summary_keys
        if key in source_shape_attrs
    }
    if source_summary:
        summary["source_subject_shape_contract"] = source_summary
    return summary


def _run_png_artifacts(root: Any, run_path: str) -> tuple[dict[str, Any], ...]:
    try:
        visualizations = root[f"{run_path}/visualizations"]
    except Exception:
        return ()
    rows: list[dict[str, Any]] = []
    child_names = sorted(
        set(visualizations.group_keys()) | set(visualizations.array_keys())
    )
    for name in child_names:
        artifact_path = f"{run_path}/visualizations/{name}"
        node = visualizations[name]
        attrs = getattr(node, "attrs", {})
        media_type = str(attrs.get("media_type") or attrs.get("mime") or "")
        if media_type != "image/png" and not str(name).lower().endswith("png"):
            continue
        try:
            resolved_path, payload = load_png_artifact_bytes(root, artifact_path)
        except Exception as exc:
            rows.append({"path": artifact_path, "error": str(exc)})
            continue
        rows.append(
            {
                "path": resolved_path,
                "media_type": "image/png",
                "bytes": payload,
                "description": attrs.get("description"),
            }
        )
    return tuple(rows)


class CoreBehaviorSource:
    """Metadata-first, read-only source for canonical core analysis runs."""

    def __init__(
        self,
        zarr_path: Path | str,
        option: InteractiveSpecOption | CoreBehaviorOption,
        *,
        legacy_eye_angle_compatibility: bool = False,
        legacy_swim_bout_compatibility: bool = False,
    ):
        if not is_core_behavior_option(option):
            raise ValueError("Not a core-behavior source")
        self.zarr_path = Path(zarr_path)
        self.option = (
            option
            if isinstance(option, CoreBehaviorOption)
            else _core_option_from_spec(option)
        )
        self.source_paths = dict(self.option.source_paths)
        self.track_id = int(self.option.track_id)
        if type(legacy_eye_angle_compatibility) is not bool:
            raise TypeError("legacy_eye_angle_compatibility must be an exact bool")
        if type(legacy_swim_bout_compatibility) is not bool:
            raise TypeError("legacy_swim_bout_compatibility must be an exact bool")
        self.legacy_eye_angle_compatibility = legacy_eye_angle_compatibility
        self.legacy_swim_bout_compatibility = legacy_swim_bout_compatibility
        self._time_seconds_cache: np.ndarray | None = None
        self._swim_bout_selection_cache: tuple[Any, Any] | None = None
        self._swim_bout_events_cache: Any = None
        self._available_analysis_ids_cache: tuple[str, ...] | None = None
        self._eye_catalog_cache: dict[str, Any] = {}
        self._tail_catalog_cache: dict[str, Any] = {}

    @property
    def available_series(self) -> tuple[str, ...]:
        excluded = {
            "run",
            "track",
            "time_seconds",
            "frame_indices",
            "positions_mm",
            "positions_px",
            "sample_valid",
            "sample_reason_code",
            "transition_valid",
            "transition_reason_code",
            "position_source_valid",
            "linear_sample_valid",
            "linear_sample_reason_code",
            "angular_sample_valid",
            "angular_sample_reason_code",
            "body_frame_source_valid",
        }
        return tuple(sorted(key for key in self.source_paths if key not in excluded))

    def series_for(self, analysis_id: str) -> tuple[str, ...]:
        if analysis_id == "speed":
            return tuple(
                key
                for key in self.available_series
                if key.startswith(SPEED_SERIES_PREFIXES)
            )
        if analysis_id == "distance_traveled":
            return tuple(
                key
                for key in self.available_series
                if any(token in key for token in PATH_SERIES_TOKENS)
                or key in {"delta_frames", "delta_seconds"}
            )
        if analysis_id == "heading":
            return tuple(
                key
                for key in self.available_series
                if any(token in key for token in HEADING_SERIES_TOKENS)
            )
        return ()

    def default_series_for(self, analysis_id: str) -> tuple[str, ...]:
        available = set(self.series_for(analysis_id))
        if analysis_id == "speed":
            preferred = (
                "speed_smoothed_mm",
                "speed_filtered_mm",
                "speed_raw_mm",
                "speed_averaged_mm",
                "speed_smoothed_px",
                "speed_filtered_px",
                "speed_raw_px",
                "speed_averaged_px",
            )
            return tuple(name for name in preferred if name in available)[:1]
        if analysis_id == "heading":
            preferred = (
                "smoothed_heading_degrees",
                "angular_speed_smoothed_deg_s",
                "angular_velocity_smoothed_deg_s",
                "delta_heading_smoothed_degrees",
            )
            return tuple(name for name in preferred if name in available)[:2]
        return ()

    def available_analysis_ids(self) -> tuple[str, ...]:
        """Detect analysis capabilities from array and lineage metadata."""

        if self._available_analysis_ids_cache is not None:
            return self._available_analysis_ids_cache

        available: list[str] = []
        if self.series_for("speed"):
            available.append("speed")
        if self._distance_traveled_level() is not None:
            available.append("distance_traveled")
        if self.series_for("heading"):
            available.append("heading")
        if "positions_mm" in self.source_paths or "positions_px" in self.source_paths:
            available.append("position")
        try:
            if self._swim_bout_selection() is not None:
                available.append("swim_bouts")
        except Exception:
            pass
        try:
            if discover_eye_angle_run_options(
                self.zarr_path,
                legacy_compatibility=self.legacy_eye_angle_compatibility,
            ):
                available.append("eye_angles")
        except Exception:
            pass
        try:
            if discover_tail_kinematics_run_options(self._root()):
                available.append("tail_kinematics")
        except Exception:
            pass
        try:
            if self.baseline_interval() is not None and (
                "positions_mm" in self.source_paths
                or "positions_px" in self.source_paths
            ):
                available.append("baseline")
        except Exception:
            pass
        self._available_analysis_ids_cache = tuple(available)
        return self._available_analysis_ids_cache

    def eye_angle_options(self) -> tuple[Any, ...]:
        return tuple(
            discover_eye_angle_run_options(
                self.zarr_path,
                legacy_compatibility=self.legacy_eye_angle_compatibility,
            )
        )

    def eye_angle_catalog(self, run_name: str | None = None) -> Any:
        key = str(run_name or "latest")
        if key not in self._eye_catalog_cache:
            self._eye_catalog_cache[key] = catalog_eye_angle_timeseries_data(
                self.zarr_path,
                run_name=run_name,
                prefer_frame=True,
                legacy_compatibility=self.legacy_eye_angle_compatibility,
            )
        return self._eye_catalog_cache[key]

    def eye_representations_for(self, run_name: str | None = None) -> tuple[str, ...]:
        catalog = self.eye_angle_catalog(run_name)
        preferred_order = (
            "eye_frame",
            "gaze",
            "nasal_gaze",
            "major",
            "centroid",
            "legacy",
            "legacy_minor",
            "other",
        )
        present = set(catalog.channel_representations.values())
        return tuple(name for name in preferred_order if name in present) + tuple(
            sorted(present.difference(preferred_order))
        )

    def eye_series_for(
        self,
        run_name: str | None = None,
        representation: str | None = None,
    ) -> tuple[str, ...]:
        catalog = self.eye_angle_catalog(run_name)
        return tuple(
            name
            for name in catalog.angle_channels
            if representation is None
            or catalog.channel_representations.get(name) == str(representation)
        )

    def default_eye_series_for(
        self,
        run_name: str | None = None,
        representation: str | None = None,
    ) -> tuple[str, ...]:
        available = set(self.eye_series_for(run_name, representation))
        preferences = {
            "eye_frame": (
                "left_eye_angle_deg_smoothed",
                "right_eye_angle_deg_smoothed",
                "vergence_eye_angle_deg_smoothed",
                "left_eye_angle_deg",
                "right_eye_angle_deg",
                "vergence_eye_angle_deg",
            ),
            "gaze": (
                "left_gaze_signed_deg_smoothed",
                "right_gaze_signed_deg_smoothed",
                "vergence_gaze_signed_deg_smoothed",
                "left_gaze_signed_deg",
                "right_gaze_signed_deg",
                "vergence_gaze_signed_deg",
            ),
            "nasal_gaze": (
                "left_nasal_gaze_deg_smoothed",
                "right_nasal_gaze_deg_smoothed",
                "mean_eye_vergence_gaze_deg_smoothed",
                "left_nasal_gaze_deg",
                "right_nasal_gaze_deg",
                "mean_eye_vergence_gaze_deg",
            ),
            "major": (
                "left_major_signed_deg_smoothed",
                "right_major_signed_deg_smoothed",
                "vergence_major_signed_deg_smoothed",
            ),
            "centroid": (
                "left_centroid_deg_smoothed",
                "right_centroid_deg_smoothed",
                "vergence_centroid_deg_smoothed",
            ),
        }
        selected = [
            name
            for name in preferences.get(str(representation), ())
            if name in available
        ]
        if len(selected) < 3:
            selected.extend(name for name in sorted(available) if name not in selected)
        return tuple(selected[:3])

    def eye_time_bounds(self, run_name: str | None = None) -> tuple[float, float]:
        catalog = self.eye_angle_catalog(run_name)
        return float(catalog.time_start_s), float(catalog.time_stop_s)

    def tail_kinematics_options(self) -> tuple[Any, ...]:
        return tuple(discover_tail_kinematics_run_options(self._root()))

    def tail_kinematics_catalog(self, run_name: str | None = None) -> Any:
        key = str(run_name or "latest")
        if key not in self._tail_catalog_cache:
            self._tail_catalog_cache[key] = catalog_tail_kinematics_run(
                self._root(),
                run_name=run_name,
            )
        return self._tail_catalog_cache[key]

    def tail_scalar_series_for(self, run_name: str | None = None) -> tuple[str, ...]:
        available = set(self.tail_kinematics_catalog(run_name).scalar_series)
        return tuple(name for name in TAIL_SCALAR_SERIES if name in available)

    def default_tail_scalar_series_for(
        self,
        run_name: str | None = None,
    ) -> tuple[str, ...]:
        available = set(self.tail_scalar_series_for(run_name))
        preferred = (
            "tail_tip_angle_deg",
            "tail_tip_lateral_deflection_px",
            "tail_angle_rms_deg",
        )
        return tuple(name for name in preferred if name in available)

    def tail_time_bounds(self, run_name: str | None = None) -> tuple[float, float]:
        catalog = self.tail_kinematics_catalog(run_name)
        return float(catalog.time_start_s), float(catalog.time_stop_s)

    def _swim_bout_selection(self) -> tuple[Any, Any] | None:
        if self._swim_bout_selection_cache is not None:
            return self._swim_bout_selection_cache
        if "track" not in self.source_paths:
            return None
        root = self._root()
        track_run_name = _normal_path(self.option.run_path).split("/")[-1]
        candidates = discover_swim_bout_candidates(
            root,
            track_run_name=track_run_name,
            track_id=self.track_id,
            include_bout_counts=False,
            legacy_compatibility=self.legacy_swim_bout_compatibility,
        )
        if not candidates:
            return None
        candidate = candidates[0]
        signal = next(
            (item for item in candidate.signals if item.is_default),
            candidate.signals[0],
        )
        self._swim_bout_selection_cache = (candidate, signal)
        return self._swim_bout_selection_cache

    def _swim_bout_events(self) -> Any:
        if self._swim_bout_events_cache is not None:
            return self._swim_bout_events_cache
        selection = self._swim_bout_selection()
        if selection is None:
            raise SwimBoutIOError("No lineage-compatible swim-bout run is available")
        candidate, signal = selection
        self._swim_bout_events_cache = load_swim_bout_events(
            self._root(),
            candidate=candidate,
            signal=signal,
            legacy_compatibility=self.legacy_swim_bout_compatibility,
        )
        return self._swim_bout_events_cache

    def _root(self):
        return open_zarr_root(self.zarr_path, mode="r")

    def _array(self, root: Any, key: str):
        path = self.source_paths.get(key)
        if not path:
            return None
        try:
            return root[path]
        except Exception:
            return None

    def _projection_metadata(
        self,
        loaded_paths: Sequence[str],
    ) -> Mapping[str, Any]:
        del loaded_paths
        return MappingProxyType({})

    def _semantic_epoch_metadata(self) -> tuple[Mapping[str, Any], ...]:
        """Return no inferred epochs for an unvalidated source."""

        return ()

    def _distance_traveled_level(self) -> tuple[str, str, str] | None:
        for unit, cumulative_key, increment_key in DISTANCE_TRAVELED_LEVELS:
            if {
                cumulative_key,
                increment_key,
                "delta_frames",
                "delta_seconds",
                "transition_valid",
                "transition_reason_code",
            }.issubset(self.source_paths):
                return unit, cumulative_key, increment_key
        return None

    def time_bounds(self) -> tuple[float, float]:
        if self._time_seconds_cache is not None:
            return _finite_bounds(self._time_seconds_cache)
        root = self._root()
        array = self._array(root, "time_seconds")
        if array is None or int(array.shape[0]) == 0:
            return 0.0, 0.0
        first = float(np.asarray(array[0]).reshape(-1)[0])
        last = float(np.asarray(array[int(array.shape[0]) - 1]).reshape(-1)[0])
        if not np.isfinite(first) or not np.isfinite(last):
            return _finite_bounds(np.asarray(array[:], dtype=np.float64))
        return (min(first, last), max(first, last))

    def _row_projection(
        self,
        root: Any,
        *,
        start_s: float | None,
        stop_s: float | None,
    ) -> tuple[np.ndarray, slice]:
        time_array = self._array(root, "time_seconds")
        if time_array is None:
            raise ValueError("Track-kinematics spec does not resolve time_seconds")
        # Only the coordinate is materialized to resolve a chunk-friendly
        # contiguous slice. Payload arrays are read with that slice below.
        if self._time_seconds_cache is None:
            self._time_seconds_cache = np.asarray(
                time_array[:], dtype=np.float64
            ).reshape(-1)
        times = self._time_seconds_cache
        if times.size == 0:
            return times, slice(0, 0)
        lo = float(start_s) if start_s is not None else float(times[0])
        hi = float(stop_s) if stop_s is not None else float(times[-1])
        if hi < lo:
            lo, hi = hi, lo
        finite_monotonic = np.isfinite(times).all() and bool(
            np.all(np.diff(times) >= 0)
        )
        if finite_monotonic:
            start_index = int(np.searchsorted(times, lo, side="left"))
            stop_index = int(np.searchsorted(times, hi, side="right"))
        else:
            selected = np.flatnonzero(
                np.isfinite(times) & (times >= lo) & (times <= hi)
            )
            if selected.size == 0:
                return times[:0], slice(0, 0)
            start_index = int(selected[0])
            stop_index = int(selected[-1]) + 1
        selected_slice = slice(start_index, stop_index)
        return times[selected_slice], selected_slice

    def project_timeseries(
        self,
        analysis_id: str,
        *,
        start_s: float | None = None,
        stop_s: float | None = None,
        series_keys: Sequence[str] | None = None,
    ) -> CoreBehaviorProjection:
        started = time.perf_counter()
        root = self._root()
        times, row_slice = self._row_projection(root, start_s=start_s, stop_s=stop_s)
        columns: dict[str, Any] = {"time_s": times}
        loaded_paths: list[str] = [self.source_paths["time_seconds"]]
        frame_array = self._array(root, "frame_indices")
        if frame_array is not None:
            frame_values = np.asarray(frame_array[row_slice], dtype=np.int64).reshape(
                -1
            )
            if frame_values.shape[0] == times.shape[0]:
                columns["frame_index"] = frame_values
                loaded_paths.append(self.source_paths["frame_indices"])
        for key in (
            tuple(series_keys)
            if series_keys is not None
            else self.series_for(analysis_id)
        ):
            array = self._array(root, key)
            if array is None:
                continue
            values = np.asarray(array[row_slice])
            if values.ndim != 1 or values.shape[0] != times.shape[0]:
                continue
            columns[key] = values.astype(np.float64, copy=False)
            loaded_paths.append(self.source_paths[key])
        validity_keys = {
            "speed": ("linear_sample_valid", "linear_sample_reason_code"),
            "heading": ("angular_sample_valid", "angular_sample_reason_code"),
            "distance_traveled": (
                "transition_valid",
                "transition_reason_code",
            ),
        }.get(analysis_id, ())
        for key in validity_keys:
            array = self._array(root, key)
            if array is None:
                continue
            values = np.asarray(array[row_slice]).reshape(-1)
            if values.shape[0] != times.shape[0]:
                continue
            columns[key] = values
            loaded_paths.append(self.source_paths[key])
        frame = pl.DataFrame(columns)
        bounds = _finite_bounds(times)
        return CoreBehaviorProjection(
            analysis_id=analysis_id,
            frame=frame.lazy(),
            columns=tuple(frame.columns),
            source_paths=tuple(loaded_paths),
            start_s=bounds[0],
            stop_s=bounds[1],
            row_count=frame.height,
            load_duration_ms=(time.perf_counter() - started) * 1000.0,
            note="Deferred Zarr projection; Polars transformations are lazy after array read.",
            metadata=self._projection_metadata(tuple(dict.fromkeys(loaded_paths))),
        )

    def project_distance_traveled(
        self,
        *,
        start_s: float | None = None,
        stop_s: float | None = None,
    ) -> CoreBehaviorProjection:
        """Project persisted cumulative path with explicit gap coverage.

        This is a view over the track-kinematics payload, not a new scientific
        computation.  Per-second rows sum the persisted smoothed increments;
        invalid transitions remain explicit and never contribute zero-valued
        distance as if the animal had been observed stationary.
        """

        level = self._distance_traveled_level()
        if level is None:
            raise ValueError(
                "Track-kinematics source lacks one complete distance-traveled level"
            )
        unit, cumulative_key, increment_key = level
        projected = self.project_timeseries(
            "distance_traveled",
            start_s=start_s,
            stop_s=stop_s,
            series_keys=(
                cumulative_key,
                increment_key,
                "delta_frames",
                "delta_seconds",
            ),
        )
        candidate = pl.col("delta_frames").cast(pl.Int64, strict=True) > 0
        valid = (
            candidate
            & pl.col("transition_valid").fill_null(False)
            & pl.col(increment_key).cast(pl.Float64, strict=True).is_finite()
            & (pl.col("delta_seconds").cast(pl.Float64, strict=True) > 0.0)
        )
        per_second = (
            projected.frame.with_columns(
                pl.col("time_s").floor().cast(pl.Int64).alias("second_index")
            )
            .group_by("second_index")
            .agg(
                pl.when(valid)
                .then(pl.col(increment_key).cast(pl.Float64, strict=True))
                .otherwise(0.0)
                .sum()
                .alias(f"distance_{unit}"),
                pl.when(valid)
                .then(pl.col("delta_seconds").cast(pl.Float64, strict=True))
                .otherwise(0.0)
                .sum()
                .alias("observed_duration_s"),
                candidate.cast(pl.Int64).sum().alias("candidate_transition_count"),
                valid.cast(pl.Int64).sum().alias("valid_transition_count"),
                (candidate & ~valid)
                .cast(pl.Int64)
                .sum()
                .alias("invalid_transition_count"),
            )
            .with_columns(
                pl.when(pl.col("candidate_transition_count") > 0)
                .then(
                    pl.col("valid_transition_count")
                    / pl.col("candidate_transition_count")
                )
                .otherwise(None)
                .alias("valid_transition_fraction"),
                pl.when(pl.col("observed_duration_s") > 0.0)
                .then(pl.col(f"distance_{unit}") / pl.col("observed_duration_s"))
                .otherwise(None)
                .alias(f"observed_speed_{unit}_s"),
            )
            .sort("second_index")
        )
        metadata = {
            **dict(projected.metadata),
            "distance_traveled": {
                "unit": unit,
                "cumulative_array": cumulative_key,
                "increment_array": increment_key,
                "increment_level": "smoothed",
                "transition_anchor": "destination_sample",
                "window_distance_operation": "sum_valid_persisted_smoothed_increments",
                "invalid_transition_policy": "excluded_and_reported_not_zero_distance",
                "per_second_operation": "floor_time_s_then_sum_valid_increments",
            },
            "semantic_epochs": self._semantic_epoch_metadata(),
        }
        return CoreBehaviorProjection(
            analysis_id="distance_traveled",
            frame=projected.frame,
            columns=projected.columns,
            source_paths=projected.source_paths,
            start_s=projected.start_s,
            stop_s=projected.stop_s,
            row_count=projected.row_count,
            load_duration_ms=projected.load_duration_ms,
            note=(
                "Observed cumulative smoothed path from persisted provider-motion "
                "arrays; tracking gaps remain explicit invalid evidence."
            ),
            related_frames=MappingProxyType({"per_second": per_second}),
            metadata=_freeze_core_metadata(metadata),
        )

    def project_positions(
        self,
        *,
        start_s: float | None = None,
        stop_s: float | None = None,
        analysis_id: str = "position",
    ) -> CoreBehaviorProjection:
        started = time.perf_counter()
        root = self._root()
        times, row_slice = self._row_projection(root, start_s=start_s, stop_s=stop_s)
        position_key = (
            "positions_mm" if "positions_mm" in self.source_paths else "positions_px"
        )
        position_array = self._array(root, position_key)
        if position_array is None:
            raise ValueError("Track-kinematics spec has no projected position array")
        positions = np.asarray(position_array[row_slice], dtype=np.float64)
        if (
            positions.ndim != 2
            or positions.shape[1] < 2
            or positions.shape[0] != times.shape[0]
        ):
            raise ValueError("Projected position array must have shape (time, >=2)")
        unit = "mm" if position_key.endswith("_mm") else "px"
        columns: dict[str, Any] = {
            "time_s": times,
            "x": positions[:, 0],
            "y": positions[:, 1],
            "unit": np.full(times.shape[0], unit, dtype=object),
        }
        loaded_paths = [
            self.source_paths["time_seconds"],
            self.source_paths[position_key],
        ]
        frame_array = self._array(root, "frame_indices")
        if frame_array is not None:
            frame_values = np.asarray(frame_array[row_slice], dtype=np.int64).reshape(
                -1
            )
            if frame_values.shape[0] == times.shape[0]:
                columns["frame_index"] = frame_values
                loaded_paths.append(self.source_paths["frame_indices"])
        validity_array = self._array(root, "position_source_valid")
        if validity_array is not None:
            validity = np.asarray(validity_array[row_slice], dtype=bool).reshape(-1)
            if validity.shape[0] == times.shape[0]:
                columns["position_source_valid"] = validity
                loaded_paths.append(self.source_paths["position_source_valid"])
        frame = pl.DataFrame(columns)
        bounds = _finite_bounds(times)
        return CoreBehaviorProjection(
            analysis_id=analysis_id,
            frame=frame.lazy(),
            columns=tuple(frame.columns),
            source_paths=tuple(loaded_paths),
            start_s=bounds[0],
            stop_s=bounds[1],
            row_count=frame.height,
            load_duration_ms=(time.perf_counter() - started) * 1000.0,
            note="Deferred position projection from Zarr; Polars transformations are lazy after array read.",
            metadata=self._projection_metadata(tuple(dict.fromkeys(loaded_paths))),
        )

    def project_swim_bouts(
        self,
        *,
        start_s: float | None = None,
        stop_s: float | None = None,
    ) -> CoreBehaviorProjection:
        started = time.perf_counter()
        events = self._swim_bout_events()
        lazy = _structured_lazy(events.bouts)
        schema = lazy.collect_schema()
        aliases: list[pl.Expr] = []
        if "start_time_s" in schema and "start_s" not in schema:
            aliases.append(pl.col("start_time_s").alias("start_s"))
        if "end_time_s" in schema and "end_s" not in schema:
            aliases.append(pl.col("end_time_s").alias("end_s"))
        if aliases:
            lazy = lazy.with_columns(aliases)
            schema = lazy.collect_schema()
        if {"start_s", "end_s"}.issubset(schema):
            if start_s is not None:
                lazy = lazy.filter(pl.col("end_s") >= float(start_s))
            if stop_s is not None:
                lazy = lazy.filter(pl.col("start_s") <= float(stop_s))
        bounds_columns = [
            name for name in ("start_s", "end_s", "peak_time_s") if name in schema
        ]
        if bounds_columns:
            bounds_row = (
                lazy.select(
                    pl.min_horizontal(*[pl.col(name) for name in bounds_columns])
                    .min()
                    .alias("start"),
                    pl.max_horizontal(*[pl.col(name) for name in bounds_columns])
                    .max()
                    .alias("stop"),
                )
                .collect()
                .row(0)
            )
            bounds = (float(bounds_row[0] or 0.0), float(bounds_row[1] or 0.0))
        else:
            bounds = (0.0, 0.0)
        row_count = int(lazy.select(pl.len()).collect().item())
        signal_prefix = str(events.signal.speed_level or "").strip()
        speed_candidates = [
            name
            for name in (
                f"{signal_prefix}_mm",
                f"{signal_prefix}_px",
                "speed_smoothed_mm",
                "speed_filtered_mm",
                "speed_raw_mm",
                "speed_averaged_mm",
                "speed_smoothed_px",
                "speed_filtered_px",
                "speed_raw_px",
                "speed_averaged_px",
            )
            if name in self.source_paths and _is_physical_speed_column(name)
        ]
        selected_speed = speed_candidates[0] if speed_candidates else None
        speed = self.project_timeseries(
            "speed",
            start_s=start_s,
            stop_s=stop_s,
            series_keys=(selected_speed,) if selected_speed is not None else (),
        )
        related_frames = (
            {"speed_trace": speed.frame.select(["time_s", selected_speed])}
            if selected_speed is not None
            else {}
        )
        return CoreBehaviorProjection(
            analysis_id="swim_bouts",
            frame=lazy,
            columns=tuple(schema.names()),
            source_paths=tuple(dict.fromkeys((events.level_path, *speed.source_paths))),
            start_s=bounds[0],
            stop_s=bounds[1],
            row_count=row_count,
            load_duration_ms=(time.perf_counter() - started) * 1000.0,
            note=(
                f"Persisted `{events.signal.speed_level}` bout segmentation from `{events.run_name}`; "
                "the lineage-compatible speed trace and downstream Polars queries are read-only."
            ),
            related_frames=related_frames,
        )

    def project_eye_angles(
        self,
        *,
        run_name: str | None = None,
        representation: str | None = None,
        start_s: float | None = None,
        stop_s: float | None = None,
        series_keys: Sequence[str] | None = None,
    ) -> CoreBehaviorProjection:
        started = time.perf_counter()
        catalog = self.eye_angle_catalog(run_name)
        selected_series = (
            tuple(series_keys)
            if series_keys is not None
            else self.default_eye_series_for(
                catalog.run_name,
                representation,
            )
        )
        if not selected_series:
            raise ValueError(
                f"No selectable eye-angle series are available for representation {representation!r}"
            )
        payload = load_eye_angle_timeseries_window(
            self.zarr_path,
            run_name=catalog.run_name,
            prefer_frame=True,
            start_s=start_s,
            stop_s=stop_s,
            series_names=selected_series,
            legacy_compatibility=self.legacy_eye_angle_compatibility,
        )
        frame = payload.dataframe
        bounds = (
            _finite_bounds(frame["time_s"].to_numpy())
            if "time_s" in frame.columns
            else (0.0, 0.0)
        )
        qa_summary: dict[str, Any] = {}
        for qa_name in (
            "valid_frame",
            "valid_left",
            "valid_right",
            "major_axis_marginal",
        ):
            if qa_name not in frame.columns:
                continue
            values = frame.get_column(qa_name).cast(pl.Boolean, strict=False)
            qa_summary[f"{qa_name}_fraction"] = float(values.mean() or 0.0)
        root = self._root()
        pngs = _run_png_artifacts(root, payload.run_path)
        return CoreBehaviorProjection(
            analysis_id="eye_angles",
            frame=frame.lazy(),
            columns=tuple(frame.columns),
            source_paths=tuple(dict.fromkeys(payload.source_paths.values())),
            start_s=bounds[0],
            stop_s=bounds[1],
            row_count=frame.height,
            load_duration_ms=(time.perf_counter() - started) * 1000.0,
            note=(
                "Bounded, column-selective eye-angle projection from persisted arrays; "
                "downstream Polars transformations are lazy."
            ),
            metadata={
                "eye_run_name": payload.run_name,
                "eye_run_path": payload.run_path,
                "row_axis": payload.row_axis,
                "representation": representation,
                "selected_series": selected_series,
                "qa_summary": qa_summary,
                "provenance": _eye_provenance_summary(payload.attrs),
                "persisted_pngs": pngs,
            },
        )

    def project_tail_kinematics(
        self,
        *,
        run_name: str | None = None,
        start_s: float | None = None,
        stop_s: float | None = None,
        scalar_series: Sequence[str] | None = None,
    ) -> CoreBehaviorProjection:
        started = time.perf_counter()
        catalog = self.tail_kinematics_catalog(run_name)
        selected_scalars = (
            tuple(scalar_series)
            if scalar_series is not None
            else self.default_tail_scalar_series_for(catalog.run_name)
        )
        root = self._root()
        payload = load_tail_kinematics_window(
            root,
            run_name=catalog.run_name,
            start_s=start_s,
            stop_s=stop_s,
            scalar_series=selected_scalars,
            include_native_angles=True,
            include_dense_curvature=True,
            max_rows=10_000,
        )
        columns: dict[str, Any] = {
            "time_s": payload.time_seconds,
            "frame_index": payload.frame_indices,
            "valid": payload.valid,
        }
        angle_columns = tuple(
            f"tail_angle_{index:02d}_deg" for index in range(payload.angle_deg.shape[1])
        )
        for index, name in enumerate(angle_columns):
            columns[name] = payload.angle_deg[:, index]
        curvature_columns = tuple(
            f"tail_curvature_{index:02d}_px_inv"
            for index in range(payload.dense_curvature_px_inv.shape[1])
        )
        for index, name in enumerate(curvature_columns):
            columns[name] = payload.dense_curvature_px_inv[:, index]
        for name, values in payload.scalar_series.items():
            columns[name] = values
        frame = pl.DataFrame(columns)

        related_frames: dict[str, pl.LazyFrame] = {}
        companion_notes: list[str] = []
        if payload.catalog.source_shape_run_path is None:
            companion_notes.append(
                "The exact source subject-shape curvature surface is unavailable or does not "
                "satisfy the canonical row-alignment contract."
            )
        try:
            bouts = _structured_lazy(self._swim_bout_events().bouts)
            schema = bouts.collect_schema()
            aliases: list[pl.Expr] = []
            if "start_time_s" in schema and "start_s" not in schema:
                aliases.append(pl.col("start_time_s").alias("start_s"))
            if "end_time_s" in schema and "end_s" not in schema:
                aliases.append(pl.col("end_time_s").alias("end_s"))
            if aliases:
                bouts = bouts.with_columns(aliases)
            schema = bouts.collect_schema()
            if {"start_s", "end_s"}.issubset(schema):
                if frame.height:
                    bouts = bouts.filter(
                        (pl.col("end_s") >= float(frame["time_s"][0]))
                        & (pl.col("start_s") <= float(frame["time_s"][-1]))
                    )
                related_frames["bout_intervals"] = bouts
            else:
                companion_notes.append("Persisted bouts lack start/end time columns.")
        except Exception as exc:
            companion_notes.append(f"No lineage-compatible bout overlay: {exc}")

        try:
            frame_path = self.source_paths.get("frame_indices")
            position_key = (
                "positions_mm"
                if "positions_mm" in self.source_paths
                else "positions_px"
            )
            position_path = self.source_paths.get(position_key)
            if frame.height and frame_path and position_path:
                track_frames = root[frame_path]

                def _search(value: int, *, right: bool) -> int:
                    lo = 0
                    hi = int(track_frames.shape[0])
                    while lo < hi:
                        middle = (lo + hi) // 2
                        current = int(np.asarray(track_frames[middle]).reshape(-1)[0])
                        if current < value or (right and current == value):
                            lo = middle + 1
                        else:
                            hi = middle
                    return lo

                first_frame = int(frame["frame_index"][0])
                last_frame = int(frame["frame_index"][-1])
                row_slice = slice(
                    _search(first_frame, right=False),
                    _search(last_frame, right=True),
                )
                projected_frames = np.asarray(
                    track_frames[row_slice], dtype=np.int64
                ).reshape(-1)
                projected_positions = np.asarray(
                    root[position_path][row_slice], dtype=np.float64
                )
                if (
                    projected_positions.ndim == 2
                    and projected_positions.shape[0] == projected_frames.shape[0]
                    and projected_positions.shape[1] >= 2
                ):
                    unit = "mm" if position_key.endswith("_mm") else "px"
                    related_frames["position_trace"] = pl.DataFrame(
                        {
                            "time_s": projected_frames.astype(np.float64)
                            / float(payload.catalog.fps),
                            "frame_index": projected_frames,
                            "x": projected_positions[:, 0],
                            "y": projected_positions[:, 1],
                            "unit": np.full(
                                projected_frames.shape[0], unit, dtype=object
                            ),
                        }
                    ).lazy()
            elif frame.height:
                companion_notes.append(
                    "No track-position coordinate is available for this source."
                )
        except Exception as exc:
            companion_notes.append(f"Position companion could not be loaded: {exc}")

        bounds = _finite_bounds(payload.time_seconds)
        pngs = _run_png_artifacts(root, payload.catalog.run_path)
        return CoreBehaviorProjection(
            analysis_id="tail_kinematics",
            frame=frame.lazy(),
            columns=tuple(frame.columns),
            source_paths=tuple(dict.fromkeys(payload.source_paths.values())),
            start_s=bounds[0],
            stop_s=bounds[1],
            row_count=frame.height,
            load_duration_ms=(time.perf_counter() - started) * 1000.0,
            note=(
                "Bounded framewise tail projection from the canonical tail run and its exact "
                "subject-shape source; no viewer-side interpolation or writeback is performed."
            ),
            related_frames=related_frames,
            metadata={
                "tail_run_name": payload.catalog.run_name,
                "tail_run_path": payload.catalog.run_path,
                "source_shape_run_name": payload.catalog.source_shape_run_name,
                "source_shape_run_path": payload.catalog.source_shape_run_path,
                "fps": payload.catalog.fps,
                "fps_source": payload.catalog.fps_source,
                "nyquist_hz": (
                    float(payload.catalog.fps) / 2.0
                    if payload.catalog.fps is not None
                    else None
                ),
                "angle_columns": angle_columns,
                "angle_sample_s": payload.catalog.angle_sample_s.tolist(),
                "curvature_columns": curvature_columns,
                "curvature_sample_s": (
                    payload.catalog.source_curvature_sample_s.tolist()
                ),
                "scalar_columns": selected_scalars,
                "companion_notes": tuple(companion_notes),
                "provenance": _tail_provenance_summary(
                    payload.catalog.attrs,
                    payload.catalog.source_shape_attrs,
                ),
                "persisted_pngs": pngs,
            },
        )

    def baseline_interval(self) -> BaselineInterval | None:
        """Resolve a pre period only from the sealed logical epoch table."""

        try:
            distance = load_chaser_distance_run(self._root(), run_name="latest")
        except ChaserDistanceReadError:
            return None
        count = min(
            len(distance.epoch_labels),
            int(distance.epoch_start_frame.size),
            int(distance.epoch_end_frame.size),
        )
        for index, label in enumerate(distance.epoch_labels[:count]):
            if is_baseline_label(label):
                return BaselineInterval(
                    label=label,
                    start_s=float(distance.epoch_start_frame[index]) / distance.fps,
                    stop_s=(float(distance.epoch_end_frame[index] + 1) / distance.fps),
                )
        return None


class ValidatedCoreBehaviorSource(CoreBehaviorSource):
    """Exact Core Behavior projections routed through one validated bundle.

    Schema v1 intentionally exposes only direct provider-motion frame
    surfaces.  It never invokes the older independent bout, eye, tail, or
    baseline discovery paths; those capabilities remain unavailable here until
    their exact normalized loaders are routed through the bundle.
    """

    def __init__(self, source: ValidatedRecordingBehaviorSource) -> None:
        if not isinstance(source, ValidatedRecordingBehaviorSource):
            raise TypeError(
                "ValidatedCoreBehaviorSource requires one validated bundle handle"
            )
        self.validated_behavior_source = source
        option = validated_core_behavior_option(source)
        catalog = source.provider_motion_catalog()
        self._provider_array_by_key = {
            key: path.removeprefix(f"{catalog.run_path}/")
            for key, path in option.source_paths.items()
            if key != "run" and path.startswith(f"{catalog.run_path}/")
        }
        super().__init__(source.analysis_zarr, option)

    @property
    def capability_states(self) -> Mapping[str, Mapping[str, Any]]:
        return self.validated_behavior_source.capability_states()

    def available_analysis_ids(self) -> tuple[str, ...]:
        if self._available_analysis_ids_cache is None:
            available: list[str] = []
            if self.series_for("speed"):
                available.append("speed")
            if self._distance_traveled_level() is not None:
                available.append("distance_traveled")
            if self.series_for("heading"):
                available.append("heading")
            if (
                "positions_mm" in self.source_paths
                or "positions_px" in self.source_paths
            ):
                available.append("position")
            self._available_analysis_ids_cache = tuple(available)
        return self._available_analysis_ids_cache

    def _root(self) -> "ValidatedCoreBehaviorSource":
        return self

    def _array(self, root: Any, key: str):
        del root
        path = self._provider_array_by_key.get(key)
        if path is None:
            return None
        projection = self.validated_behavior_source.provider_motion_track_projection(
            (path,)
        )
        return projection.arrays[path]

    def _projection_metadata(
        self,
        loaded_paths: Sequence[str],
    ) -> Mapping[str, Any]:
        catalog = self.validated_behavior_source.provider_motion_catalog()
        prefix = f"{catalog.run_path}/"
        consumed = tuple(
            path.removeprefix(prefix)
            for path in loaded_paths
            if path.startswith(prefix)
        )
        records = {
            path: {
                "source_path": f"{catalog.run_path}/{path}",
                "sha256": catalog.array_records[path]["sha256"],
                "dtype": catalog.array_records[path]["dtype"],
                "shape": tuple(catalog.array_records[path]["shape"]),
            }
            for path in consumed
        }
        return _freeze_core_metadata(
            {
                "source_mode": "validated_recording_behavior_bundle_v1",
                "bundle_path": str(self.validated_behavior_source.bundle_path),
                "bundle_sha256": self.validated_behavior_source.bundle_sha256,
                "recording_id": self.validated_behavior_source.recording_id,
                "provider_motion": {
                    "run_path": catalog.run_path,
                    "manifest_sha256": catalog.manifest_sha256,
                    "verification_digest": catalog.verification_digest,
                    "track_id": catalog.track_id,
                    "track_row_start": catalog.track_row_start,
                    "track_row_stop": catalog.track_row_stop,
                },
                "consumed_arrays": records,
                "track_partition_arrays": {
                    path: catalog.array_records[path]["sha256"]
                    for path in ("track_ids", "track_row_offsets")
                },
                "validation_policy": (
                    "manifest_digest_per_consumed_array_plus_exact_track_partition"
                ),
                "selector_resolution": False,
                "capability_states": self.capability_states,
            }
        )

    def _semantic_epoch_metadata(self) -> tuple[Mapping[str, Any], ...]:
        try:
            records = self.validated_behavior_source.semantic_epoch_records()
        except ValidatedCapabilityUnavailableError:
            return ()
        return tuple(
            MappingProxyType(
                {
                    "window_id": record.window_id,
                    "analysis_role": record.analysis_role,
                    "source_label": record.source_label,
                    "start_frame": record.start_frame,
                    "end_frame_exclusive": record.end_frame_exclusive,
                    "source_interval_sha256": record.source_interval_sha256,
                    "protocol_semantic_hash": record.protocol_semantic_hash,
                    "protocol_semantic_step_index": record.protocol_semantic_step_index,
                    "protocol_semantic_step_ref": record.protocol_semantic_step_ref,
                    "terminal_frame_excluded_pending_step_end_contract": (
                        record.terminal_frame_excluded_pending_step_end_contract
                    ),
                }
            )
            for record in records
        )

    def _unsupported_exact_route(self, capability: str) -> None:
        record = self.validated_behavior_source.capability_record(capability)
        raise ValidatedRecordingBehaviorSourceError(
            f"Exact Core Behavior route for {capability!r} is not implemented in "
            f"schema-v1 adapter slice (bundle state={record['state']!r}); "
            "independent selector discovery is prohibited."
        )

    def _swim_bout_selection(self) -> tuple[Any, Any] | None:
        return None

    def eye_angle_options(self) -> tuple[Any, ...]:
        return ()

    def eye_angle_catalog(self, run_name: str | None = None) -> Any:
        del run_name
        self._unsupported_exact_route("eye_angles")
        raise AssertionError("unreachable")

    def tail_kinematics_options(self) -> tuple[Any, ...]:
        return ()

    def tail_kinematics_catalog(self, run_name: str | None = None) -> Any:
        del run_name
        self._unsupported_exact_route("tail_kinematics")
        raise AssertionError("unreachable")

    def baseline_interval(self) -> BaselineInterval | None:
        return None

    def project_swim_bouts(self, **_kwargs: Any) -> CoreBehaviorProjection:
        self._unsupported_exact_route("canonical_swim_bouts")
        raise AssertionError("unreachable")

    def project_eye_angles(self, **_kwargs: Any) -> CoreBehaviorProjection:
        self._unsupported_exact_route("eye_angles")
        raise AssertionError("unreachable")

    def project_tail_kinematics(self, **_kwargs: Any) -> CoreBehaviorProjection:
        self._unsupported_exact_route("tail_kinematics")
        raise AssertionError("unreachable")


def _freeze_core_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_core_metadata(item) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_core_metadata(item) for item in value)
    return value


def load_core_behavior_projection(
    source: CoreBehaviorSource,
    analysis_id: str,
    *,
    start_s: float | None = None,
    stop_s: float | None = None,
    series_keys: Sequence[str] | None = None,
    eye_run_name: str | None = None,
    eye_representation: str | None = None,
    tail_run_name: str | None = None,
    tail_scalar_series: Sequence[str] | None = None,
) -> CoreBehaviorProjection:
    if analysis_id in {"speed", "heading"}:
        return source.project_timeseries(
            analysis_id,
            start_s=start_s,
            stop_s=stop_s,
            series_keys=series_keys,
        )
    if analysis_id == "distance_traveled":
        return source.project_distance_traveled(start_s=start_s, stop_s=stop_s)
    if analysis_id == "position":
        return source.project_positions(start_s=start_s, stop_s=stop_s)
    if analysis_id == "swim_bouts":
        return source.project_swim_bouts(start_s=start_s, stop_s=stop_s)
    if analysis_id == "eye_angles":
        return source.project_eye_angles(
            run_name=eye_run_name,
            representation=eye_representation,
            start_s=start_s,
            stop_s=stop_s,
            series_keys=series_keys,
        )
    if analysis_id == "tail_kinematics":
        return source.project_tail_kinematics(
            run_name=tail_run_name,
            start_s=start_s,
            stop_s=stop_s,
            scalar_series=tail_scalar_series,
        )
    if analysis_id == "baseline":
        interval = source.baseline_interval()
        if interval is None:
            raise ValueError(
                "No canonical pre-period window is available in this recording"
            )
        position = source.project_positions(
            start_s=interval.start_s,
            stop_s=interval.stop_s,
            analysis_id="baseline",
        )
        speed = source.project_timeseries(
            "speed",
            start_s=interval.start_s,
            stop_s=interval.stop_s,
        )
        speed_columns_available = [
            name for name in speed.columns if _is_physical_speed_column(name)
        ]
        preferred_speed_columns = (
            "speed_smoothed_mm",
            "speed_filtered_mm",
            "speed_raw_mm",
            "speed_averaged_mm",
            "speed_smoothed_px",
            "speed_filtered_px",
            "speed_raw_px",
            "speed_averaged_px",
        )
        speed_columns = [
            name for name in preferred_speed_columns if name in speed_columns_available
        ][:4]
        joined = position.frame
        if speed_columns:
            joined = joined.join(
                speed.frame.select(["time_s", *speed_columns]),
                on="time_s",
                how="left",
            )
        schema = joined.collect_schema()
        return CoreBehaviorProjection(
            analysis_id="baseline",
            frame=joined,
            columns=tuple(schema.names()),
            source_paths=tuple(
                dict.fromkeys((*position.source_paths, *speed.source_paths))
            ),
            start_s=position.start_s,
            stop_s=position.stop_s,
            row_count=position.row_count,
            load_duration_ms=position.load_duration_ms + speed.load_duration_ms,
            note=(
                f"Canonical pre-period `{interval.label}` resolved from persisted epoch metadata; "
                "trajectory and activity are descriptive viewer projections."
            ),
        )
    raise ValueError(f"Unsupported core-behavior analysis: {analysis_id!r}")


def collect_projection(
    projection: CoreBehaviorProjection,
    *,
    columns: Iterable[str] | None = None,
    start_s: float | None = None,
    stop_s: float | None = None,
) -> pl.DataFrame:
    """Collect a bounded view from a projected LazyFrame."""

    query = projection.frame
    schema = query.collect_schema()
    if "time_s" in schema and (start_s is not None or stop_s is not None):
        if start_s is not None:
            query = query.filter(pl.col("time_s") >= float(start_s))
        if stop_s is not None:
            query = query.filter(pl.col("time_s") <= float(stop_s))
    if columns is not None:
        selected = [name for name in columns if name in query.collect_schema()]
        query = query.select(selected)
    return query.collect()


def _decimate_for_display(
    frame: pl.DataFrame,
    *,
    trace_count: int,
    max_total_values: int = 60000,
) -> pl.DataFrame:
    """Bound serialized plotting payload without changing persisted data."""

    if frame.is_empty():
        return frame
    row_budget = max(1000, int(max_total_values) // max(1, int(trace_count)))
    if frame.height <= row_budget:
        return frame
    stride = max(1, int(np.ceil(frame.height / float(row_budget))))
    return (
        frame.lazy()
        .with_row_index("_display_row")
        .filter((pl.col("_display_row") % stride) == 0)
        .drop("_display_row")
        .collect()
    )


def _distance_epoch_windows(
    frame: pl.DataFrame,
    metadata: Mapping[str, Any],
) -> tuple[tuple[str, float, float], ...]:
    """Map exact semantic frame intervals onto the selected time projection."""

    if not {"frame_index", "time_s"}.issubset(frame.columns):
        return ()
    raw_epochs = metadata.get("semantic_epochs", ())
    if not isinstance(raw_epochs, (tuple, list)):
        return ()
    frames = frame["frame_index"].to_numpy().astype(np.int64, copy=False)
    times = frame["time_s"].to_numpy().astype(np.float64, copy=False)
    result: list[tuple[str, float, float]] = []
    for raw in raw_epochs:
        if not isinstance(raw, Mapping):
            continue
        try:
            start = int(raw["start_frame"])
            stop = int(raw["end_frame_exclusive"])
            label = str(raw["analysis_role"])
        except (KeyError, TypeError, ValueError):
            continue
        selected = np.flatnonzero(
            (frames >= start) & (frames < stop) & np.isfinite(times)
        )
        if selected.size:
            result.append(
                (label, float(times[int(selected[0])]), float(times[int(selected[-1])]))
            )
    return tuple(result)


def _add_distance_epoch_shading(
    figure: Any,
    windows: Sequence[tuple[str, float, float]],
) -> None:
    colors = {
        "chaser_pre": "rgba(76,120,168,0.10)",
        "chaser_training": "rgba(228,87,86,0.10)",
        "chaser_post": "rgba(84,162,75,0.10)",
    }
    for label, start_s, stop_s in windows:
        figure.add_vrect(
            x0=start_s,
            x1=stop_s,
            fillcolor=colors.get(label, "rgba(120,120,120,0.08)"),
            opacity=1.0,
            line_width=0,
            layer="below",
        )


def build_core_behavior_output(
    mo: Any,
    go: Any,
    px: Any,
    *,
    projection: CoreBehaviorProjection,
) -> Any:
    """Render one selected core analysis without activating sibling analyses."""

    frame = collect_projection(projection)
    header = mo.hstack(
        [
            mo.stat(label="Rows projected", value=f"{projection.row_count:,}"),
            mo.stat(label="Columns", value=f"{len(projection.columns):,}"),
            mo.stat(label="Zarr read ms", value=f"{projection.load_duration_ms:.1f}"),
        ]
    )
    source_note = mo.md(
        f"{projection.note} Source arrays: `{', '.join(projection.source_paths) or 'none'}`"
    )
    if projection.analysis_id == "distance_traveled":
        contract = projection.metadata.get("distance_traveled", {})
        if not isinstance(contract, Mapping):
            contract = {}
        unit = str(contract.get("unit") or "distance units")
        cumulative_key = str(contract.get("cumulative_array") or "")
        increment_key = str(contract.get("increment_array") or "")
        required = {
            cumulative_key,
            increment_key,
            "delta_frames",
            "delta_seconds",
            "transition_valid",
        }
        if not cumulative_key or not increment_key or not required.issubset(frame.columns):
            body = mo.md("The persisted distance-traveled contract is incomplete.")
        else:
            candidate = frame["delta_frames"].cast(pl.Int64).to_numpy() > 0
            transition_valid = frame["transition_valid"].cast(pl.Boolean).to_numpy()
            increments = frame[increment_key].cast(pl.Float64).to_numpy()
            deltas = frame["delta_seconds"].cast(pl.Float64).to_numpy()
            valid = (
                candidate
                & transition_valid
                & np.isfinite(increments)
                & np.isfinite(deltas)
                & (deltas > 0.0)
            )
            invalid = candidate & ~valid
            candidate_count = int(np.count_nonzero(candidate))
            valid_count = int(np.count_nonzero(valid))
            window_distance = float(np.sum(increments[valid], dtype=np.float64))
            cumulative = frame[cumulative_key].cast(pl.Float64).to_numpy()
            finite_cumulative = cumulative[np.isfinite(cumulative)]
            cumulative_end = (
                float(finite_cumulative[-1]) if finite_cumulative.size else float("nan")
            )
            coverage = (
                valid_count / candidate_count if candidate_count else float("nan")
            )
            summary = mo.hstack(
                [
                    mo.stat(
                        label=f"Observed distance in window ({unit})",
                        value=f"{window_distance:,.2f}",
                    ),
                    mo.stat(
                        label=f"Cumulative at window end ({unit})",
                        value=(
                            f"{cumulative_end:,.2f}"
                            if np.isfinite(cumulative_end)
                            else "unavailable"
                        ),
                    ),
                    mo.stat(
                        label="Valid transition coverage",
                        value=(f"{100.0 * coverage:.1f}%" if np.isfinite(coverage) else "n/a"),
                    ),
                    mo.stat(
                        label="Invalid transitions",
                        value=f"{int(np.count_nonzero(invalid)):,}",
                    ),
                ]
            )
            display = _decimate_for_display(frame, trace_count=2)
            cumulative_figure = go.Figure()
            cumulative_figure.add_trace(
                go.Scattergl(
                    x=display["time_s"],
                    y=display[cumulative_key],
                    mode="lines",
                    name="Observed cumulative path",
                    line=dict(color="#2563eb", width=2),
                )
            )
            invalid_frame = frame.filter(pl.Series("_invalid", invalid))
            if invalid_frame.height:
                invalid_display = _decimate_for_display(
                    invalid_frame,
                    trace_count=1,
                    max_total_values=12000,
                )
                cumulative_figure.add_trace(
                    go.Scattergl(
                        x=invalid_display["time_s"],
                        y=invalid_display[cumulative_key],
                        mode="markers",
                        name="Invalid transition evidence",
                        marker=dict(color="#dc2626", size=5, symbol="x"),
                    )
                )
            epoch_windows = _distance_epoch_windows(frame, projection.metadata)
            _add_distance_epoch_shading(cumulative_figure, epoch_windows)
            cumulative_figure.update_layout(
                title="Observed cumulative smoothed path",
                xaxis_title="Time (s)",
                yaxis_title=f"Cumulative distance ({unit})",
                height=470,
                margin=dict(l=65, r=30, t=65, b=55),
                template="plotly_white",
            )

            per_second_lazy = projection.related_frames.get("per_second")
            per_second = (
                per_second_lazy.collect()
                if per_second_lazy is not None
                else pl.DataFrame()
            )
            distance_column = f"distance_{unit}"
            per_second_figure = go.Figure()
            if per_second.height and distance_column in per_second.columns:
                per_second_figure.add_trace(
                    go.Bar(
                        x=per_second["second_index"],
                        y=per_second[distance_column],
                        name="Observed distance",
                        marker_color="#64748b",
                    )
                )
                per_second_figure.add_trace(
                    go.Scatter(
                        x=per_second["second_index"],
                        y=per_second["valid_transition_fraction"],
                        mode="lines",
                        name="Transition coverage",
                        yaxis="y2",
                        line=dict(color="#dc2626", width=1.5),
                    )
                )
            _add_distance_epoch_shading(per_second_figure, epoch_windows)
            per_second_figure.update_layout(
                title="Observed distance per second with coverage",
                xaxis_title="Session second",
                yaxis=dict(title=f"Distance ({unit})"),
                yaxis2=dict(
                    title="Valid transition fraction",
                    overlaying="y",
                    side="right",
                    range=[0.0, 1.02],
                ),
                barmode="overlay",
                height=420,
                margin=dict(l=65, r=65, t=65, b=55),
                template="plotly_white",
            )
            body = mo.vstack(
                [
                    summary,
                    cumulative_figure,
                    per_second_figure,
                    mo.md(
                        "Red crosses are candidate transitions that failed the persisted "
                        "motion validity contract. They are retained as missing coverage, "
                        "not interpreted as zero movement. Epoch shading is shown only "
                        "when exact bundle-sealed semantic frame intervals are available."
                    ),
                ]
            )
    elif projection.analysis_id in {"speed", "heading", "eye_angles"}:
        value_columns = [
            name
            for name in frame.columns
            if name
            not in {
                "time_s",
                "frame_index",
                "row_index",
                "linear_sample_reason_code",
                "angular_sample_reason_code",
            }
            and frame.schema[name].is_numeric()
            and frame.schema[name] != pl.Boolean
        ]
        display_frame = _decimate_for_display(frame, trace_count=len(value_columns))
        figure = go.Figure()
        for column in value_columns:
            figure.add_trace(
                go.Scattergl(
                    x=display_frame["time_s"],
                    y=display_frame[column],
                    mode="lines",
                    name=column,
                )
            )
        figure.update_layout(
            title=(
                "Speed and acceleration traces"
                if projection.analysis_id == "speed"
                else (
                    "Eye angles and convergence"
                    if projection.analysis_id == "eye_angles"
                    else "Heading and turning traces"
                )
            ),
            xaxis_title="Time (s)",
            yaxis_title="Value",
            height=500,
            margin=dict(l=55, r=25, t=60, b=50),
        )
        if not value_columns:
            body = mo.md("No compatible series are present in this run.")
        elif projection.analysis_id != "eye_angles":
            body = figure
        else:
            eye_pieces: list[Any] = [figure]
            qa_summary = projection.metadata.get("qa_summary", {})
            if isinstance(qa_summary, Mapping) and qa_summary:
                qa_stats = [
                    mo.stat(
                        label=str(name)
                        .removesuffix("_fraction")
                        .replace("_", " ")
                        .title(),
                        value=f"{100.0 * float(value):.1f}%",
                    )
                    for name, value in qa_summary.items()
                ]
                eye_pieces.extend(
                    [mo.md("### Eye-angle QA in selected window"), mo.hstack(qa_stats)]
                )
            provenance = projection.metadata.get("provenance", {})
            if isinstance(provenance, Mapping) and provenance:
                provenance_view = (
                    mo.tree(dict(provenance), label="Computation and source provenance")
                    if hasattr(mo, "tree")
                    else mo.md(f"Provenance: `{dict(provenance)}`")
                )
                eye_pieces.extend(
                    [mo.md("### Persisted computation contract"), provenance_view]
                )
            pngs = projection.metadata.get("persisted_pngs", ())
            eye_pieces.append(mo.md("### Persisted snapshots"))
            if pngs:
                for index, artifact in enumerate(pngs):
                    if artifact.get("bytes"):
                        eye_pieces.extend(
                            [
                                mo.md(f"`{artifact.get('path')}`"),
                                png_bytes_to_markdown_image(
                                    mo,
                                    artifact["bytes"],
                                    alt_text=f"eye-angle persisted snapshot {index + 1}",
                                ),
                            ]
                        )
                    else:
                        eye_pieces.append(
                            mo.md(
                                f"Persisted snapshot could not be loaded: "
                                f"`{artifact.get('path')}` — `{artifact.get('error')}`"
                            )
                        )
            else:
                eye_pieces.append(
                    mo.md(
                        "No analysis-owned eye-angle PNG is persisted for this run. "
                        "The interactive traces above are rendered from the canonical arrays."
                    )
                )
            body = mo.vstack(eye_pieces)
    elif projection.analysis_id in {"position", "baseline"}:
        if frame.height:
            display_frame = _decimate_for_display(
                frame,
                trace_count=3,
                max_total_values=75000,
            )
            figure = px.scatter(
                _plotly_columns(display_frame, ("x", "y", "time_s")),
                x="x",
                y="y",
                color="time_s",
                render_mode="webgl",
                title=(
                    "Pre-period trajectory (descriptive viewer projection)"
                    if projection.analysis_id == "baseline"
                    else "Trajectory"
                ),
                labels={"time_s": "Time (s)"},
            )
            figure.update_yaxes(scaleanchor="x", scaleratio=1)
            figure.update_layout(height=600)
            if projection.analysis_id == "baseline":
                speed_column = next(
                    (name for name in frame.columns if _is_physical_speed_column(name)),
                    None,
                )
                pieces: list[Any] = [figure]
                if speed_column is not None:
                    speed_fig = go.Figure(
                        go.Scattergl(
                            x=display_frame["time_s"],
                            y=display_frame[speed_column],
                            mode="lines",
                            name=speed_column,
                        )
                    )
                    speed_fig.update_layout(
                        title="Pre-period activity trace",
                        xaxis_title="Time (s)",
                        yaxis_title=speed_column,
                        height=360,
                    )
                    summary = projection.frame.select(
                        pl.col(speed_column)
                        .drop_nans()
                        .drop_nulls()
                        .count()
                        .alias("finite_speed_samples"),
                        pl.col(speed_column)
                        .drop_nans()
                        .drop_nulls()
                        .mean()
                        .alias("mean_speed"),
                        pl.col(speed_column)
                        .drop_nans()
                        .drop_nulls()
                        .median()
                        .alias("median_speed"),
                        pl.col(speed_column)
                        .drop_nans()
                        .drop_nulls()
                        .max()
                        .alias("max_speed"),
                    ).collect()
                    pieces.extend(
                        [
                            speed_fig,
                            mo.md("### Descriptive pre-period activity"),
                            mo.ui.table(summary),
                        ]
                    )
                body = mo.vstack(pieces)
            else:
                body = figure
        else:
            body = mo.md("No position rows fall inside the selected interval.")
    elif projection.analysis_id == "swim_bouts":
        pieces: list[Any] = []
        speed_lazy = projection.related_frames.get("speed_trace")
        speed_frame = speed_lazy.collect() if speed_lazy is not None else pl.DataFrame()
        if speed_frame.height > 50000:
            stride = max(1, int(np.ceil(speed_frame.height / 50000.0)))
            speed_frame = (
                speed_frame.lazy()
                .with_row_index("_row")
                .filter((pl.col("_row") % stride) == 0)
                .drop("_row")
                .collect()
            )
        speed_column = next(
            (name for name in speed_frame.columns if name != "time_s"), None
        )
        segmentation_figure = go.Figure()
        if speed_column is not None and speed_frame.height:
            segmentation_figure.add_trace(
                go.Scattergl(
                    x=speed_frame["time_s"],
                    y=speed_frame[speed_column],
                    mode="lines",
                    name=speed_column,
                    line=dict(color="#334155", width=1.25),
                )
            )
        if frame.height and {"start_s", "end_s"}.issubset(frame.columns):
            starts = frame["start_s"].to_numpy().astype(np.float64, copy=False)
            stops = frame["end_s"].to_numpy().astype(np.float64, copy=False)
            widths = stops - starts
            valid = np.isfinite(starts) & np.isfinite(stops) & (widths > 0)
            bout_column = next(
                (
                    name
                    for name in ("bout_id", "source_bout_id")
                    if name in frame.columns
                ),
                None,
            )
            bout_ids = (
                frame[bout_column].to_numpy()
                if bout_column is not None
                else np.arange(frame.height, dtype=np.int64)
            )
            if valid.any():
                segmentation_figure.add_trace(
                    go.Bar(
                        x=(starts[valid] + stops[valid]) / 2.0,
                        y=np.ones(int(np.count_nonzero(valid)), dtype=np.float64),
                        width=widths[valid],
                        base=np.zeros(int(np.count_nonzero(valid)), dtype=np.float64),
                        yaxis="y2",
                        name="Persisted swim bouts",
                        marker=dict(color="#f59e0b", line=dict(width=0)),
                        opacity=0.24,
                        customdata=np.column_stack(
                            [bout_ids[valid], starts[valid], stops[valid]]
                        ),
                        hovertemplate=(
                            "bout=%{customdata[0]}<br>"
                            "start=%{customdata[1]:.3f}s<br>"
                            "end=%{customdata[2]:.3f}s<extra></extra>"
                        ),
                    )
                )
        if segmentation_figure.data:
            segmentation_figure.update_layout(
                title="Speed trace with persisted swim-bout segmentation",
                xaxis_title="Time (s)",
                yaxis_title=speed_column or "Speed",
                yaxis2=dict(
                    overlaying="y",
                    side="right",
                    range=[0, 1],
                    showgrid=False,
                    showticklabels=False,
                    title="Bout intervals",
                ),
                barmode="overlay",
                height=480,
                margin=dict(l=55, r=55, t=60, b=50),
                legend=dict(
                    orientation="h", yanchor="top", y=-0.14, xanchor="left", x=0.0
                ),
            )
            pieces.append(segmentation_figure)
        distribution_specs = _swim_bout_distribution_specs(frame)
        if distribution_specs:
            distribution_figures = []
            for column, title, xaxis_title in distribution_specs:
                histogram = px.histogram(
                    _plotly_columns(frame, (column,)),
                    x=column,
                    nbins=40,
                    title=title,
                )
                histogram.update_layout(
                    xaxis_title=xaxis_title,
                    yaxis_title="Bout count",
                    height=360,
                    margin=dict(l=55, r=25, t=60, b=50),
                    showlegend=False,
                )
                distribution_figures.append(histogram)
            pieces.extend(
                [
                    mo.md("### Swim-bout distributions"),
                    mo.hstack(distribution_figures),
                ]
            )
        pieces.extend(
            [
                mo.md("### Persisted bout rows"),
                mo.ui.table(frame, selection=None, page_size=12),
            ]
        )
        body = mo.vstack(pieces)
    else:
        body = mo.md(f"Unsupported analysis `{projection.analysis_id}`")
    exact_provenance = (
        mo.vstack(
            [
                mo.md("### Validated source identity"),
                (
                    mo.tree(
                        dict(projection.metadata),
                        label="Bundle, provider-motion, and consumed-array bindings",
                    )
                    if hasattr(mo, "tree")
                    else mo.md(f"Source identity: `{dict(projection.metadata)}`")
                ),
            ]
        )
        if projection.metadata.get("source_mode")
        == "validated_recording_behavior_bundle_v1"
        else mo.md("")
    )
    return mo.vstack([header, source_note, body, exact_provenance])
