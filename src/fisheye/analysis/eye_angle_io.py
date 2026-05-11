"""Logical readers for Palette eye-angle analysis runs.

Eye-angle consumers should read through this module rather than hard-coding
physical Zarr paths. It exposes the same logical ROI/frame angle, QA, vector,
and support tables for hierarchical-v1 runs and compact-dense-v2 runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
import zarr

from ..shared.json_safety import decode_null_terminated_text
from ..shared.zarr_helpers import (
    first_array_length as _shared_first_array_length,
    first_array_length_in_group as _first_array_length_in_group,
    normalize_zarr_path as _normalize_path,
    read_zarr_array_mapping,
    safe_int as _safe_int,
    zarr_attrs_dict as _attrs_dict,
    zarr_child_group as _child_group,
    zarr_group_keys as _group_keys,
)


EYE_ANGLE_RUN_PARENT = "analysis/eye_angle_runs"
EYE_ANGLE_LAYOUT_HIERARCHICAL_V1 = "hierarchical_v1"
EYE_ANGLE_LAYOUT_COMPACT_DENSE_V2 = "compact_dense_v2"

EYE_ANGLE_TIMESERIES_COLUMNS: tuple[str, ...] = (
    "left_eye_angle_deg",
    "left_eye_angle_deg_smoothed",
    "right_eye_angle_deg",
    "right_eye_angle_deg_smoothed",
    "vergence_eye_angle_deg",
    "vergence_eye_angle_deg_smoothed",
    "left_major_signed_deg",
    "left_major_signed_deg_smoothed",
    "right_major_signed_deg",
    "right_major_signed_deg_smoothed",
    "vergence_major_signed_deg",
    "vergence_major_signed_deg_smoothed",
    "version_major_deg",
    "version_major_deg_smoothed",
    "left_gaze_signed_deg",
    "left_gaze_signed_deg_smoothed",
    "right_gaze_signed_deg",
    "right_gaze_signed_deg_smoothed",
    "left_minor_signed_deg",
    "left_minor_signed_deg_smoothed",
    "right_minor_signed_deg",
    "right_minor_signed_deg_smoothed",
    "left_nasal_gaze_deg",
    "left_nasal_gaze_deg_smoothed",
    "right_nasal_gaze_deg",
    "right_nasal_gaze_deg_smoothed",
    "mean_eye_vergence_gaze_deg",
    "mean_eye_vergence_gaze_deg_smoothed",
    "vergence_gaze_deg",
    "vergence_gaze_deg_smoothed",
    "vergence_gaze_signed_deg",
    "vergence_gaze_signed_deg_smoothed",
    "version_gaze_deg",
    "version_gaze_deg_smoothed",
    "left_centroid_deg",
    "left_centroid_deg_smoothed",
    "right_centroid_deg",
    "right_centroid_deg_smoothed",
    "vergence_centroid_deg",
    "vergence_centroid_deg_smoothed",
)

EYE_ANGLE_ROW_COUNT_COLUMNS: tuple[str, ...] = (
    "vergence_eye_angle_deg",
    "vergence_eye_angle_deg_smoothed",
    "left_eye_angle_deg",
    "left_eye_angle_deg_smoothed",
    "mean_eye_vergence_gaze_deg",
    "mean_eye_vergence_gaze_deg_smoothed",
    "left_gaze_signed_deg",
    "left_gaze_signed_deg_smoothed",
    "left_nasal_gaze_deg",
    "left_nasal_gaze_deg_smoothed",
    "left_minor_signed_deg",
    "left_deg",
)


class EyeAngleIOError(ValueError):
    """Raised when an eye-angle run cannot be resolved or loaded."""


@dataclass(frozen=True)
class EyeAngleRunOption:
    """One selectable eye-angle run."""

    run_name: str
    run_path: str
    label: str
    schema_version: Optional[int]
    preferred_angle_family: Optional[str]
    preferred_eye_axis: Optional[str]
    row_axis: Optional[str]
    n_rows: int
    is_latest: bool
    attrs: Mapping[str, Any]


@dataclass(frozen=True)
class EyeAngleRunTables:
    """Logical view over one eye-angle run."""

    run_name: str
    run_path: str
    attrs: Mapping[str, Any]
    roi: Mapping[str, np.ndarray]
    frame: Mapping[str, np.ndarray]
    qa_roi: Mapping[str, np.ndarray]
    qa_frame: Mapping[str, np.ndarray]
    support: Mapping[str, np.ndarray]
    source_paths: Mapping[str, str]

    @property
    def schema_version(self) -> int:
        return int(self.attrs.get("schema_version", 0) or 0)

    @property
    def row_axis(self) -> Optional[str]:
        value = self.attrs.get("row_axis")
        return str(value) if value is not None else None

    def angle_arrays(self, row_axis: str) -> Mapping[str, np.ndarray]:
        if row_axis == "frame":
            return self.frame
        if row_axis == "roi":
            return self.roi
        raise EyeAngleIOError(f"Unsupported eye-angle row axis {row_axis!r}; expected 'frame' or 'roi'.")

    def qa_arrays(self, row_axis: str) -> Mapping[str, np.ndarray]:
        if row_axis == "frame":
            return self.qa_frame
        if row_axis == "roi":
            return self.qa_roi
        raise EyeAngleIOError(f"Unsupported eye-angle QA row axis {row_axis!r}; expected 'frame' or 'roi'.")


def first_array_length(arrays: Mapping[str, np.ndarray], names: tuple[str, ...] = EYE_ANGLE_ROW_COUNT_COLUMNS) -> int:
    """Return the first non-scalar array length found among candidate names."""

    return _shared_first_array_length(arrays, names)


def optional_1d_array(arrays: Mapping[str, np.ndarray], name: str, *, length: Optional[int] = None) -> Optional[np.ndarray]:
    """Return a 1D array if present and length-compatible."""

    values = arrays.get(name)
    if values is None or values.ndim != 1:
        return None
    if length is not None and int(values.shape[0]) != int(length):
        return None
    return values


def _layout(attrs: Mapping[str, Any]) -> str:
    return str(attrs.get("layout") or attrs.get("storage_layout") or EYE_ANGLE_LAYOUT_HIERARCHICAL_V1)


def _logical_source_path(tables: "EyeAngleRunTables", logical_path: str) -> str:
    return tables.source_paths.get(logical_path, logical_path)


def _channel_names(index_group: Any | None, *, expected_count: int | None = None) -> list[str]:
    if index_group is None:
        return []

    attrs = _attrs_dict(index_group)
    for attr_name in ("channel_names", "names", "field_names"):
        raw_names = attrs.get(attr_name)
        if isinstance(raw_names, (list, tuple)):
            names = [str(decode_null_terminated_text(value)) for value in raw_names]
            if names:
                return names[:expected_count] if expected_count is not None else names

    for array_name in ("name", "channel_name", "field_name"):
        try:
            if array_name not in index_group:
                continue
            values = np.asarray(index_group[array_name][:])
        except Exception:
            continue
        if values.ndim >= 2 and values.dtype.kind in {"u", "i"}:
            iter_values = values.reshape(values.shape[0], -1)
        else:
            iter_values = values.reshape(-1)
        names = [str(decode_null_terminated_text(value)) for value in iter_values]
        names = [name for name in names if name]
        if names:
            return names[:expected_count] if expected_count is not None else names
    return []


def _channel_availability(
    index_group: Any | None,
    *,
    array_name: str,
    expected_count: int,
) -> np.ndarray:
    if index_group is None:
        return np.ones(int(expected_count), dtype=bool)
    try:
        if array_name not in index_group:
            return np.ones(int(expected_count), dtype=bool)
        values = np.asarray(index_group[array_name][:], dtype=bool).reshape(-1)
    except Exception:
        return np.ones(int(expected_count), dtype=bool)
    if int(values.shape[0]) != int(expected_count):
        raise EyeAngleIOError(
            f"Channel availability array {array_name!r} has {values.shape[0]} rows; "
            f"expected {expected_count}."
        )
    return values


def _dense_channel_mapping(
    run_group: zarr.Group,
    *,
    data_name: str,
    index_name: str,
    run_path: str,
    logical_prefix: str,
    available_name: str,
    source_paths: dict[str, str],
) -> dict[str, np.ndarray]:
    if data_name not in run_group:
        return {}
    data = np.asarray(run_group[data_name][:])
    if data.ndim != 2:
        raise EyeAngleIOError(f"{run_path}/{data_name} must be a 2D dense channel array.")
    index_group = _child_group(run_group, index_name)
    names = _channel_names(index_group, expected_count=int(data.shape[1]))
    if not names:
        raise EyeAngleIOError(f"{run_path}/{data_name} is missing channel names in {index_name}.")
    if len(names) != int(data.shape[1]):
        raise EyeAngleIOError(
            f"{run_path}/{data_name} has {data.shape[1]} channels but "
            f"{index_name} names {len(names)} channels."
        )
    available = _channel_availability(index_group, array_name=available_name, expected_count=int(data.shape[1]))

    arrays: dict[str, np.ndarray] = {}
    for channel_idx, name in enumerate(names):
        if not bool(available[channel_idx]):
            continue
        arrays[name] = np.asarray(data[:, channel_idx])
        source_paths[f"{run_path}/{logical_prefix}/{name}"] = f"{run_path}/{data_name}[:,{channel_idx}]"
    return arrays


def _dense_vector_mapping(
    run_group: zarr.Group,
    *,
    data_name: str,
    index_name: str,
    run_path: str,
    logical_prefix: str,
    available_name: str,
    source_paths: dict[str, str],
) -> dict[str, np.ndarray]:
    if data_name not in run_group:
        return {}
    data = np.asarray(run_group[data_name][:])
    if data.ndim != 3 or int(data.shape[2]) != 2:
        raise EyeAngleIOError(f"{run_path}/{data_name} must have shape (rows, channels, 2).")
    index_group = _child_group(run_group, index_name)
    names = _channel_names(index_group, expected_count=int(data.shape[1]))
    if not names:
        raise EyeAngleIOError(f"{run_path}/{data_name} is missing channel names in {index_name}.")
    if len(names) != int(data.shape[1]):
        raise EyeAngleIOError(
            f"{run_path}/{data_name} has {data.shape[1]} channels but "
            f"{index_name} names {len(names)} channels."
        )
    available = _channel_availability(index_group, array_name=available_name, expected_count=int(data.shape[1]))

    arrays: dict[str, np.ndarray] = {}
    for channel_idx, name in enumerate(names):
        if not bool(available[channel_idx]):
            continue
        arrays[name] = np.asarray(data[:, channel_idx, :])
        source_paths[f"{run_path}/{logical_prefix}/{name}"] = f"{run_path}/{data_name}[:,{channel_idx},:]"
    return arrays


def _compact_dense_tables(
    run_group: zarr.Group,
    *,
    resolved_run: str,
    run_path: str,
    attrs: Mapping[str, Any],
) -> "EyeAngleRunTables":
    source_paths: dict[str, str] = {
        "run": run_path,
        "angles/roi": f"{run_path}/roi_angles",
        "angles/frame": f"{run_path}/frame_angles",
        "qa/roi": f"{run_path}/roi_qa",
        "qa/frame": f"{run_path}/frame_qa",
        "support": f"{run_path}/support",
        "angle_channel_index": f"{run_path}/angle_channel_index",
        "qa_channel_index": f"{run_path}/qa_channel_index",
        "vector_channel_index": f"{run_path}/vector_channel_index",
    }
    roi = _dense_channel_mapping(
        run_group,
        data_name="roi_angles",
        index_name="angle_channel_index",
        run_path=run_path,
        logical_prefix="angles/roi",
        available_name="roi_available",
        source_paths=source_paths,
    )
    frame = _dense_channel_mapping(
        run_group,
        data_name="frame_angles",
        index_name="angle_channel_index",
        run_path=run_path,
        logical_prefix="angles/frame",
        available_name="frame_available",
        source_paths=source_paths,
    )
    roi.update(
        _dense_vector_mapping(
            run_group,
            data_name="roi_vectors",
            index_name="vector_channel_index",
            run_path=run_path,
            logical_prefix="angles/roi",
            available_name="roi_available",
            source_paths=source_paths,
        )
    )
    frame.update(
        _dense_vector_mapping(
            run_group,
            data_name="frame_vectors",
            index_name="vector_channel_index",
            run_path=run_path,
            logical_prefix="angles/frame",
            available_name="frame_available",
            source_paths=source_paths,
        )
    )
    if not roi and not frame:
        raise EyeAngleIOError(f"Eye-angle compact run {resolved_run!r} is missing roi_angles and frame_angles.")

    qa_roi = _dense_channel_mapping(
        run_group,
        data_name="roi_qa",
        index_name="qa_channel_index",
        run_path=run_path,
        logical_prefix="qa/roi",
        available_name="roi_available",
        source_paths=source_paths,
    )
    qa_frame = _dense_channel_mapping(
        run_group,
        data_name="frame_qa",
        index_name="qa_channel_index",
        run_path=run_path,
        logical_prefix="qa/frame",
        available_name="frame_available",
        source_paths=source_paths,
    )
    support = read_zarr_array_mapping(
        _child_group(run_group, "support"),
        physical_prefix=f"{run_path}/support",
        source_paths=source_paths,
    )

    return EyeAngleRunTables(
        run_name=resolved_run,
        run_path=run_path,
        attrs=attrs,
        roi=roi,
        frame=frame,
        qa_roi=qa_roi,
        qa_frame=qa_frame,
        support=support,
        source_paths=source_paths,
    )


def _eye_angle_option_label(
    *,
    run_name: str,
    schema_version: Optional[int],
    preferred_angle_family: Optional[str],
    preferred_eye_axis: Optional[str],
    row_axis: Optional[str],
    n_rows: int,
    is_latest: bool,
) -> str:
    pieces = [run_name]
    if schema_version is not None:
        pieces.append(f"schema v{schema_version}")
    if preferred_angle_family or preferred_eye_axis:
        pieces.append(f"{preferred_angle_family or 'unknown'} / {preferred_eye_axis or 'unknown axis'}")
    if row_axis:
        pieces.append(str(row_axis))
    pieces.append(f"{n_rows} rows")
    if is_latest:
        pieces.append("latest")
    return " | ".join(pieces)


def resolve_eye_angle_run(
    root: zarr.Group,
    run_name: str | None = None,
) -> tuple[zarr.Group, str, str]:
    """Resolve one ``analysis/eye_angle_runs`` child from a name, path, or latest."""

    parent = root.get(EYE_ANGLE_RUN_PARENT)
    if parent is None:
        raise EyeAngleIOError("No analysis/eye_angle_runs group found.")

    if run_name is None or str(run_name).strip().lower() in {"", "latest"}:
        latest = parent.attrs.get("latest")
        if isinstance(latest, str) and latest in parent:
            resolved = latest
        else:
            raise EyeAngleIOError("No latest eye-angle run is recorded.")
    else:
        normalized = _normalize_path(str(run_name))
        parts = normalized.split("/")
        if normalized.startswith(EYE_ANGLE_RUN_PARENT + "/") and len(parts) >= 3:
            resolved = parts[2]
        else:
            resolved = parts[-1]

    if not resolved or resolved not in parent:
        raise EyeAngleIOError(f"Eye-angle run {run_name!r} not found in analysis/eye_angle_runs.")
    run_path = f"{EYE_ANGLE_RUN_PARENT}/{resolved}"
    return parent[resolved], str(resolved), run_path


def load_eye_angle_run_tables(
    root: zarr.Group,
    *,
    run_name: str | None = None,
) -> EyeAngleRunTables:
    """Load logical angle, QA, and support arrays for one eye-angle run."""

    run_group, resolved_run, run_path = resolve_eye_angle_run(root, run_name)
    attrs = _attrs_dict(run_group)
    if _layout(attrs) == EYE_ANGLE_LAYOUT_COMPACT_DENSE_V2:
        return _compact_dense_tables(
            run_group,
            resolved_run=resolved_run,
            run_path=run_path,
            attrs=attrs,
        )

    source_paths: dict[str, str] = {
        "run": run_path,
        "angles/roi": f"{run_path}/angles/roi",
        "angles/frame": f"{run_path}/angles/frame",
        "qa/roi": f"{run_path}/qa/roi",
        "qa/frame": f"{run_path}/qa/frame",
        "support": f"{run_path}/support",
    }
    roi_group = _child_group(run_group, "angles/roi")
    frame_group = _child_group(run_group, "angles/frame")
    qa_roi_group = _child_group(run_group, "qa/roi")
    qa_frame_group = _child_group(run_group, "qa/frame")
    support_group = _child_group(run_group, "support")

    roi = read_zarr_array_mapping(
        roi_group,
        physical_prefix=f"{run_path}/angles/roi",
        source_paths=source_paths,
    )
    frame = read_zarr_array_mapping(
        frame_group,
        physical_prefix=f"{run_path}/angles/frame",
        source_paths=source_paths,
    )
    if not roi and not frame:
        raise EyeAngleIOError(f"Eye-angle run {resolved_run!r} is missing angles/roi and angles/frame arrays.")
    qa_roi = read_zarr_array_mapping(
        qa_roi_group,
        physical_prefix=f"{run_path}/qa/roi",
        source_paths=source_paths,
    )
    qa_frame = read_zarr_array_mapping(
        qa_frame_group,
        physical_prefix=f"{run_path}/qa/frame",
        source_paths=source_paths,
    )
    support = read_zarr_array_mapping(
        support_group,
        physical_prefix=f"{run_path}/support",
        source_paths=source_paths,
    )

    return EyeAngleRunTables(
        run_name=resolved_run,
        run_path=run_path,
        attrs=attrs,
        roi=roi,
        frame=frame,
        qa_roi=qa_roi,
        qa_frame=qa_frame,
        support=support,
        source_paths=source_paths,
    )


def _run_row_count(run_group: zarr.Group) -> int:
    attrs = _attrs_dict(run_group)
    if _layout(attrs) == EYE_ANGLE_LAYOUT_COMPACT_DENSE_V2:
        for name in ("roi_angles", "frame_angles"):
            try:
                if name in run_group:
                    shape = getattr(run_group[name], "shape", ())
                    if shape:
                        return int(shape[0])
            except Exception:
                continue
        return 0
    return _first_array_length_in_group(_child_group(run_group, "angles/roi"), EYE_ANGLE_ROW_COUNT_COLUMNS)


def discover_eye_angle_run_options(root: zarr.Group) -> list[EyeAngleRunOption]:
    """Return available eye-angle analysis runs from an open Zarr root."""

    parent = root.get(EYE_ANGLE_RUN_PARENT)
    if parent is None:
        return []

    latest = parent.attrs.get("latest")
    options: list[EyeAngleRunOption] = []
    for run_name in _group_keys(parent):
        try:
            run_group = parent[run_name]
        except Exception:
            continue
        n_rows = _run_row_count(run_group)
        if n_rows <= 0:
            continue
        attrs = _attrs_dict(run_group)
        schema_version = _safe_int(attrs.get("schema_version"))
        preferred_angle_family = attrs.get("preferred_angle_family")
        preferred_eye_axis = attrs.get("preferred_eye_axis")
        row_axis = attrs.get("row_axis")
        is_latest = str(latest) == str(run_name)
        preferred_angle_family_str = (
            str(preferred_angle_family) if preferred_angle_family is not None else None
        )
        preferred_eye_axis_str = str(preferred_eye_axis) if preferred_eye_axis is not None else None
        row_axis_str = str(row_axis) if row_axis is not None else None
        run_path = f"{EYE_ANGLE_RUN_PARENT}/{run_name}"
        options.append(
            EyeAngleRunOption(
                run_name=str(run_name),
                run_path=run_path,
                label=_eye_angle_option_label(
                    run_name=str(run_name),
                    schema_version=schema_version,
                    preferred_angle_family=preferred_angle_family_str,
                    preferred_eye_axis=preferred_eye_axis_str,
                    row_axis=row_axis_str,
                    n_rows=n_rows,
                    is_latest=is_latest,
                ),
                schema_version=schema_version,
                preferred_angle_family=preferred_angle_family_str,
                preferred_eye_axis=preferred_eye_axis_str,
                row_axis=row_axis_str,
                n_rows=n_rows,
                is_latest=is_latest,
                attrs=attrs,
            )
        )

    return sorted(options, key=lambda item: (not item.is_latest, item.run_name))


def aligned_frame_values(
    values: np.ndarray,
    frames: np.ndarray,
    *,
    dtype: np.dtype | type,
    source_path: str,
) -> np.ndarray:
    """Return frame-indexed values with bounds checks."""

    frame_indices = np.asarray(frames, dtype=np.int64)
    if frame_indices.size == 0:
        return np.asarray([], dtype=dtype)
    if np.any(frame_indices < 0):
        raise EyeAngleIOError(f"{source_path} cannot be aligned to negative frame indices.")
    if int(np.max(frame_indices)) >= int(values.shape[0]):
        raise EyeAngleIOError(
            f"{source_path} length {values.shape[0]} cannot cover requested frame "
            f"{int(np.max(frame_indices))}."
        )
    return np.asarray(values[frame_indices], dtype=dtype)


def load_eye_gaze_frame_series(
    root: zarr.Group,
    *,
    eye_angle_run: str,
    eye_angle_family: str,
    frames: np.ndarray,
    allowed_families: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load frame-aligned gaze series for bout-level eye summaries."""

    family = str(eye_angle_family).strip()
    if family not in allowed_families:
        expected = ", ".join(allowed_families)
        raise EyeAngleIOError(f"Unsupported eye_angle_family {eye_angle_family!r}; expected one of: {expected}")

    tables = load_eye_angle_run_tables(root, run_name=eye_angle_run)
    if tables.schema_version < 2:
        raise EyeAngleIOError(
            f"Eye-angle run {tables.run_name!r} has schema_version={tables.schema_version}; "
            "bout eye-gaze summaries require schema v2 frame-level gaze arrays."
        )
    if not tables.frame:
        raise EyeAngleIOError(f"Eye-angle run {tables.run_name!r} is missing angles/frame outputs.")

    required_arrays = {
        "left_gaze_deg": "left_gaze_deg",
        "right_gaze_deg": "right_gaze_deg",
        "vergence_gaze_deg": "vergence_gaze_deg",
    }
    series: dict[str, np.ndarray] = {}
    source_arrays: dict[str, str] = {}
    for key, array_name in required_arrays.items():
        values = tables.frame.get(array_name)
        if values is None:
            raise EyeAngleIOError(f"Eye-angle run {tables.run_name!r} is missing angles/frame/{array_name}.")
        source_path = f"{tables.run_path}/angles/frame/{array_name}"
        source_path = _logical_source_path(tables, source_path)
        series[key] = aligned_frame_values(
            values,
            frames,
            dtype=np.float64,
            source_path=source_path,
        )
        source_arrays[key] = source_path

    signed_name = "vergence_gaze_signed_deg"
    signed_values = tables.frame.get(signed_name)
    if signed_values is not None:
        source_path = f"{tables.run_path}/angles/frame/{signed_name}"
        source_path = _logical_source_path(tables, source_path)
        series[signed_name] = aligned_frame_values(
            signed_values,
            frames,
            dtype=np.float64,
            source_path=source_path,
        )
        source_arrays[signed_name] = source_path
    else:
        series[signed_name] = np.full(frames.shape[0], float("nan"), dtype=np.float64)

    valid = np.isfinite(series["left_gaze_deg"]) & np.isfinite(series["right_gaze_deg"])
    valid &= np.isfinite(series["vergence_gaze_deg"])
    valid_frame = tables.qa_frame.get("valid_frame")
    if valid_frame is not None:
        source_path = f"{tables.run_path}/qa/frame/valid_frame"
        source_path = _logical_source_path(tables, source_path)
        valid &= aligned_frame_values(
            valid_frame,
            frames,
            dtype=bool,
            source_path=source_path,
        )
        source_arrays["valid_frame"] = source_path
    series["valid_frame"] = np.asarray(valid, dtype=bool)

    source_refs = {
        "source_eye_angle_run": tables.run_name,
        "source_eye_angle_path": tables.run_path,
        "source_eye_angle_schema_version": tables.schema_version,
        "source_eye_angle_family": family,
        "source_eye_angle_arrays": source_arrays,
    }
    return series, source_refs


def frame_time_seconds(tables: EyeAngleRunTables, *, row_count: int) -> Optional[np.ndarray]:
    """Return frame-aligned timestamps when present and length-compatible."""

    return optional_1d_array(tables.support, "frame_time_seconds", length=row_count)


def roi_time_seconds(tables: EyeAngleRunTables, *, row_count: int) -> Optional[np.ndarray]:
    """Return ROI-aligned timestamps when present and length-compatible."""

    return optional_1d_array(tables.support, "time_seconds", length=row_count)


def roi_frame_indices(tables: EyeAngleRunTables, *, row_count: int) -> Optional[np.ndarray]:
    """Return ROI frame indices when present and length-compatible."""

    values = optional_1d_array(tables.support, "frame_indices", length=row_count)
    if values is None:
        return None
    return values.astype(np.int64, copy=False)


__all__ = [
    "EYE_ANGLE_LAYOUT_COMPACT_DENSE_V2",
    "EYE_ANGLE_LAYOUT_HIERARCHICAL_V1",
    "EYE_ANGLE_RUN_PARENT",
    "EYE_ANGLE_ROW_COUNT_COLUMNS",
    "EYE_ANGLE_TIMESERIES_COLUMNS",
    "EyeAngleIOError",
    "EyeAngleRunOption",
    "EyeAngleRunTables",
    "aligned_frame_values",
    "discover_eye_angle_run_options",
    "first_array_length",
    "frame_time_seconds",
    "load_eye_angle_run_tables",
    "load_eye_gaze_frame_series",
    "optional_1d_array",
    "resolve_eye_angle_run",
    "roi_frame_indices",
    "roi_time_seconds",
]
