"""Metadata-first, bounded readers for canonical tail-kinematics runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

from ..shared.zarr_helpers import (
    safe_int as _safe_int,
    zarr_attrs_dict as _attrs_dict,
    zarr_group_keys as _group_keys,
)


TAIL_KINEMATICS_RUN_PARENT = "analysis/tail_kinematics_runs"
SUBJECT_SHAPE_RUN_PARENT = "analysis/subject_shape_runs"

TAIL_SCALAR_SERIES: tuple[str, ...] = (
    "tail_tip_angle_deg",
    "tail_tip_lateral_deflection_px",
    "tail_angle_rms_deg",
    "max_abs_tail_angle_deg",
    "integrated_abs_tail_angle_rad",
    "max_abs_tail_curvature_px_inv",
    "integrated_abs_tail_curvature",
)


class TailKinematicsIOError(ValueError):
    """Raised when a tail-kinematics run cannot be projected safely."""


@dataclass(frozen=True)
class TailKinematicsRunOption:
    run_name: str
    run_path: str
    label: str
    schema_version: int | None
    method: str | None
    source_subject_shape_run: str | None
    row_count: int
    sample_count: int
    is_latest: bool
    attrs: Mapping[str, Any]


@dataclass(frozen=True)
class TailKinematicsCatalog:
    run_name: str
    run_path: str
    attrs: Mapping[str, Any]
    row_count: int
    frame_start: int
    frame_stop: int
    fps: float | None
    fps_source: str
    time_start_s: float
    time_stop_s: float
    angle_sample_s: np.ndarray
    scalar_series: tuple[str, ...]
    source_shape_run_name: str | None
    source_shape_run_path: str | None
    source_curvature_sample_s: np.ndarray
    source_curvature_sample_count: int
    source_shape_attrs: Mapping[str, Any]


@dataclass(frozen=True)
class TailKinematicsWindow:
    catalog: TailKinematicsCatalog
    frame_indices: np.ndarray
    time_seconds: np.ndarray
    valid: np.ndarray
    angle_deg: np.ndarray
    dense_curvature_px_inv: np.ndarray
    scalar_series: Mapping[str, np.ndarray]
    source_paths: Mapping[str, str]


def _normal_path(value: object) -> str:
    return "/".join(part for part in str(value or "").strip("/").split("/") if part)


def _array_handle(group: Any, name: str, *, path: str) -> Any:
    value = group.get(name)
    if value is None or not hasattr(value, "shape") or not hasattr(value, "__getitem__"):
        raise TailKinematicsIOError(f"{path}/{name} is missing or is not an array.")
    return value


def _positive_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) and result > 0 else None


def resolve_tail_kinematics_run(
    root: zarr.Group,
    run_name: str | None = None,
) -> tuple[zarr.Group, str, str]:
    parent = root.get(TAIL_KINEMATICS_RUN_PARENT)
    if parent is None:
        raise TailKinematicsIOError("No analysis/tail_kinematics_runs group found.")
    if run_name is None or str(run_name).strip().lower() in {"", "latest"}:
        latest = parent.attrs.get("latest_complete") or parent.attrs.get("latest")
        resolved = str(latest or "")
    else:
        normalized = _normal_path(run_name)
        resolved = normalized.split("/")[-1]
    if not resolved or resolved not in parent:
        raise TailKinematicsIOError(
            f"Tail-kinematics run {run_name!r} not found in {TAIL_KINEMATICS_RUN_PARENT}."
        )
    run_path = f"{TAIL_KINEMATICS_RUN_PARENT}/{resolved}"
    run_group = parent[resolved]
    if not isinstance(run_group, zarr.Group):
        raise TailKinematicsIOError(f"{run_path} is not a Zarr group.")
    return run_group, resolved, run_path


def _run_dimensions(run_group: zarr.Group, run_path: str) -> tuple[int, int]:
    angle = _array_handle(run_group, "tail_angle_deg", path=run_path)
    if len(angle.shape) != 2 or int(angle.shape[0]) <= 0 or int(angle.shape[1]) <= 1:
        raise TailKinematicsIOError(f"{run_path}/tail_angle_deg must have shape (rows, samples).")
    frame_index = _array_handle(run_group, "frame_index", path=run_path)
    if len(frame_index.shape) != 1 or int(frame_index.shape[0]) != int(angle.shape[0]):
        raise TailKinematicsIOError(f"{run_path}/frame_index does not align with tail_angle_deg.")
    return int(angle.shape[0]), int(angle.shape[1])


def discover_tail_kinematics_run_options(root: zarr.Group) -> list[TailKinematicsRunOption]:
    parent = root.get(TAIL_KINEMATICS_RUN_PARENT)
    if parent is None:
        return []
    latest = str(parent.attrs.get("latest_complete") or parent.attrs.get("latest") or "")
    options: list[TailKinematicsRunOption] = []
    for run_name in _group_keys(parent):
        try:
            run_group = parent[run_name]
            attrs = _attrs_dict(run_group)
            status = str(attrs.get("palette_run_completion_status") or "").lower()
            if status and status != "complete":
                continue
            row_count, sample_count = _run_dimensions(
                run_group, f"{TAIL_KINEMATICS_RUN_PARENT}/{run_name}"
            )
        except Exception:
            continue
        schema_version = _safe_int(attrs.get("schema_version"))
        method = str(attrs.get("method")) if attrs.get("method") is not None else None
        source_shape = attrs.get("source_subject_shape_run")
        source_shape_name = str(source_shape) if source_shape is not None else None
        is_latest = str(run_name) == latest
        label_parts = [str(run_name)]
        if schema_version is not None:
            label_parts.append(f"schema v{schema_version}")
        label_parts.extend([f"{row_count:,} rows", f"{sample_count} angle positions"])
        if is_latest:
            label_parts.append("latest")
        options.append(
            TailKinematicsRunOption(
                run_name=str(run_name),
                run_path=f"{TAIL_KINEMATICS_RUN_PARENT}/{run_name}",
                label=" | ".join(label_parts),
                schema_version=schema_version,
                method=method,
                source_subject_shape_run=source_shape_name,
                row_count=row_count,
                sample_count=sample_count,
                is_latest=is_latest,
                attrs=attrs,
            )
        )
    return sorted(options, key=lambda option: (not option.is_latest, option.run_name))


def _resolve_fps(root: zarr.Group, run_attrs: Mapping[str, Any]) -> tuple[float | None, str]:
    for key in ("fps", "frame_rate", "source_video_fps", "video_fps"):
        value = _positive_float(run_attrs.get(key))
        if value is not None:
            return value, f"tail_run.attrs.{key}"
    for key in ("fps", "frame_rate", "source_video_fps", "video_fps"):
        value = _positive_float(root.attrs.get(key))
        if value is not None:
            return value, f"root.attrs.{key}"
    return None, "unavailable"


def _subject_shape_catalog(
    root: zarr.Group,
    source_run_name: str | None,
    *,
    expected_rows: int,
) -> tuple[str | None, np.ndarray, Mapping[str, Any]]:
    if not source_run_name:
        return None, np.asarray([], dtype=np.float32), {}
    normalized = _normal_path(source_run_name)
    resolved_name = normalized.split("/")[-1]
    run_path = f"{SUBJECT_SHAPE_RUN_PARENT}/{resolved_name}"
    try:
        run_group = root[run_path]
        body = run_group["components/subject_body"]
        curvature = body["tail_curvature_px_inv"]
        sample_s = np.asarray(body["tail_sample_s"][:], dtype=np.float32).reshape(-1)
        sample_valid = body["tail_sample_valid"]
        frame_indices = run_group["row_index/frame_indices"]
    except Exception:
        return None, np.asarray([], dtype=np.float32), {}
    if (
        len(curvature.shape) != 2
        or int(curvature.shape[0]) != int(expected_rows)
        or int(curvature.shape[1]) != int(sample_s.shape[0])
        or len(sample_valid.shape) != 1
        or int(sample_valid.shape[0]) != int(expected_rows)
        or len(frame_indices.shape) != 1
        or int(frame_indices.shape[0]) != int(expected_rows)
    ):
        return None, np.asarray([], dtype=np.float32), {}
    return run_path, sample_s, _attrs_dict(run_group)


def catalog_tail_kinematics_run(
    root: zarr.Group,
    *,
    run_name: str | None = None,
) -> TailKinematicsCatalog:
    run_group, resolved_run, run_path = resolve_tail_kinematics_run(root, run_name)
    attrs = _attrs_dict(run_group)
    row_count, sample_count = _run_dimensions(run_group, run_path)
    frame_index = _array_handle(run_group, "frame_index", path=run_path)
    frame_start = int(np.asarray(frame_index[0]).reshape(-1)[0])
    frame_stop = int(np.asarray(frame_index[row_count - 1]).reshape(-1)[0])
    if frame_start < 0 or frame_stop < frame_start:
        raise TailKinematicsIOError(f"{run_path}/frame_index is not a valid increasing coordinate.")
    angle_sample_s = np.asarray(
        _array_handle(run_group, "tail_angle_sample_s", path=run_path)[:],
        dtype=np.float32,
    ).reshape(-1)
    if int(angle_sample_s.shape[0]) != sample_count:
        raise TailKinematicsIOError(f"{run_path}/tail_angle_sample_s has the wrong length.")
    fps, fps_source = _resolve_fps(root, attrs)
    if fps is None:
        time_start_s = float(frame_start)
        time_stop_s = float(frame_stop)
    else:
        time_start_s = float(frame_start) / fps
        time_stop_s = float(frame_stop) / fps

    source_shape_name_raw = attrs.get("source_subject_shape_run")
    source_shape_name = None
    if source_shape_name_raw is not None:
        normalized_source = _normal_path(source_shape_name_raw)
        source_shape_name = normalized_source.split("/")[-1] or None
    source_shape_path, curvature_sample_s, source_shape_attrs = _subject_shape_catalog(
        root,
        source_shape_name,
        expected_rows=row_count,
    )
    scalar_series = tuple(name for name in TAIL_SCALAR_SERIES if name in run_group)
    return TailKinematicsCatalog(
        run_name=resolved_run,
        run_path=run_path,
        attrs=attrs,
        row_count=row_count,
        frame_start=frame_start,
        frame_stop=frame_stop,
        fps=fps,
        fps_source=fps_source,
        time_start_s=time_start_s,
        time_stop_s=time_stop_s,
        angle_sample_s=angle_sample_s,
        scalar_series=scalar_series,
        source_shape_run_name=source_shape_name,
        source_shape_run_path=source_shape_path,
        source_curvature_sample_s=curvature_sample_s,
        source_curvature_sample_count=int(curvature_sample_s.shape[0]),
        source_shape_attrs=source_shape_attrs,
    )


def _searchsorted_array(array: Any, value: int, *, side: str) -> int:
    """Binary-search a monotonic Zarr coordinate without materializing it."""

    lo = 0
    hi = int(array.shape[0])
    while lo < hi:
        mid = (lo + hi) // 2
        current = int(np.asarray(array[mid]).reshape(-1)[0])
        move_right = current < int(value) or (side == "right" and current == int(value))
        if move_right:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _selected_row_slice(
    frame_index: Any,
    catalog: TailKinematicsCatalog,
    *,
    start_s: float | None,
    stop_s: float | None,
    max_rows: int,
) -> tuple[slice, int, int]:
    if catalog.fps is None:
        raise TailKinematicsIOError(
            "Tail time projection requires positive recording fps metadata; none was found."
        )
    lo_s = catalog.time_start_s if start_s is None else float(start_s)
    hi_s = catalog.time_stop_s if stop_s is None else float(stop_s)
    if hi_s < lo_s:
        lo_s, hi_s = hi_s, lo_s
    lo_s = max(catalog.time_start_s, lo_s)
    hi_s = min(catalog.time_stop_s, hi_s)
    if hi_s < lo_s:
        return slice(0, 0), 0, -1
    start_frame = max(catalog.frame_start, int(np.ceil(lo_s * catalog.fps - 1e-9)))
    stop_frame = min(catalog.frame_stop, int(np.floor(hi_s * catalog.fps + 1e-9)))
    start_row = _searchsorted_array(frame_index, start_frame, side="left")
    stop_row = _searchsorted_array(frame_index, stop_frame, side="right")
    selected_rows = max(0, stop_row - start_row)
    if selected_rows > int(max_rows):
        raise TailKinematicsIOError(
            f"Requested tail window spans {selected_rows:,} rows; the read-only viewer limit "
            f"is {int(max_rows):,}. Select a shorter interval."
        )
    return slice(start_row, stop_row), start_frame, stop_frame


def load_tail_kinematics_window(
    root: zarr.Group,
    *,
    run_name: str | None = None,
    start_s: float | None = None,
    stop_s: float | None = None,
    scalar_series: Sequence[str] = (),
    include_native_angles: bool = True,
    include_dense_curvature: bool = True,
    max_rows: int = 60_000,
) -> TailKinematicsWindow:
    catalog = catalog_tail_kinematics_run(root, run_name=run_name)
    run_group, _resolved_run, run_path = resolve_tail_kinematics_run(root, catalog.run_name)
    frame_array = _array_handle(run_group, "frame_index", path=run_path)
    row_slice, start_frame, stop_frame = _selected_row_slice(
        frame_array,
        catalog,
        start_s=start_s,
        stop_s=stop_s,
        max_rows=max_rows,
    )
    frames = np.asarray(frame_array[row_slice], dtype=np.int64).reshape(-1)
    in_window = (frames >= start_frame) & (frames <= stop_frame)
    frames = frames[in_window]
    times = frames.astype(np.float64) / float(catalog.fps)
    valid = np.asarray(
        _array_handle(run_group, "valid", path=run_path)[row_slice], dtype=bool
    ).reshape(-1)[in_window]
    source_paths: dict[str, str] = {
        "run": run_path,
        "frame_index": f"{run_path}/frame_index",
        "valid": f"{run_path}/valid",
        "angle_sample_s": f"{run_path}/tail_angle_sample_s",
    }

    if include_native_angles:
        angle_deg = np.asarray(
            _array_handle(run_group, "tail_angle_deg", path=run_path)[row_slice],
            dtype=np.float32,
        )[in_window]
        angle_deg[~valid] = np.nan
        source_paths["angle_deg"] = f"{run_path}/tail_angle_deg"
    else:
        angle_deg = np.empty((frames.shape[0], 0), dtype=np.float32)

    requested_scalars = tuple(dict.fromkeys(str(name) for name in scalar_series))
    missing = [name for name in requested_scalars if name not in catalog.scalar_series]
    if missing:
        raise TailKinematicsIOError(f"Unavailable tail scalar series: {', '.join(missing)}")
    scalars: dict[str, np.ndarray] = {}
    for name in requested_scalars:
        values = np.asarray(run_group[name][row_slice], dtype=np.float32).reshape(-1)[in_window]
        values[~valid] = np.nan
        scalars[name] = values
        source_paths[f"scalar/{name}"] = f"{run_path}/{name}"

    dense_curvature = np.empty((frames.shape[0], 0), dtype=np.float32)
    if include_dense_curvature and catalog.source_shape_run_path is not None:
        shape_group = root[catalog.source_shape_run_path]
        shape_frames = np.asarray(
            shape_group["row_index/frame_indices"][row_slice], dtype=np.int64
        ).reshape(-1)[in_window]
        if not np.array_equal(shape_frames, frames):
            raise TailKinematicsIOError(
                "Source subject-shape frame lineage does not align with the tail run."
            )
        dense_curvature = np.asarray(
            shape_group["components/subject_body/tail_curvature_px_inv"][row_slice],
            dtype=np.float32,
        )[in_window]
        shape_valid = np.asarray(
            shape_group["components/subject_body/tail_sample_valid"][row_slice],
            dtype=bool,
        ).reshape(-1)[in_window]
        dense_valid = valid & shape_valid
        dense_curvature[~dense_valid] = np.nan
        source_paths.update(
            {
                "dense_curvature": (
                    f"{catalog.source_shape_run_path}/components/subject_body/"
                    "tail_curvature_px_inv"
                ),
                "dense_curvature_sample_s": (
                    f"{catalog.source_shape_run_path}/components/subject_body/tail_sample_s"
                ),
                "source_shape_frame_indices": (
                    f"{catalog.source_shape_run_path}/row_index/frame_indices"
                ),
            }
        )

    return TailKinematicsWindow(
        catalog=catalog,
        frame_indices=frames,
        time_seconds=times,
        valid=valid,
        angle_deg=angle_deg,
        dense_curvature_px_inv=dense_curvature,
        scalar_series=scalars,
        source_paths=source_paths,
    )


__all__ = [
    "TAIL_KINEMATICS_RUN_PARENT",
    "TAIL_SCALAR_SERIES",
    "TailKinematicsCatalog",
    "TailKinematicsIOError",
    "TailKinematicsRunOption",
    "TailKinematicsWindow",
    "catalog_tail_kinematics_run",
    "discover_tail_kinematics_run_options",
    "load_tail_kinematics_window",
    "resolve_tail_kinematics_run",
]
