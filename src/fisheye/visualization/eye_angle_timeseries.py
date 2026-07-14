"""Pandas-free eye-angle projections for deployed recording explorers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import polars as pl

from fisheye.analysis.eye_angle_io import (
    EYE_ANGLE_TIMESERIES_COLUMNS,
    EyeAngleRunOption,
    discover_eye_angle_run_options as discover_eye_angle_run_options_from_root,
    first_array_length,
    frame_time_seconds,
    load_eye_angle_run_tables,
    optional_1d_array,
    roi_frame_indices,
    roi_time_seconds,
)
from fisheye.shared.zarr_io import open_zarr_root


@dataclass(frozen=True)
class EyeAngleTimeseriesData:
    zarr_path: Path
    run_name: str
    run_path: str
    row_axis: str
    attrs: Mapping[str, Any]
    dataframe: pl.DataFrame


def _safe_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def discover_eye_angle_run_options(zarr_path: Path | str) -> list[EyeAngleRunOption]:
    """Return available eye-angle analysis runs without importing pandas."""

    root = open_zarr_root(Path(zarr_path), mode="r")
    return discover_eye_angle_run_options_from_root(root)


def load_eye_angle_timeseries_data(
    zarr_path: Path | str,
    *,
    run_name: str | None = None,
    prefer_frame: bool = True,
) -> EyeAngleTimeseriesData:
    """Load one eye-angle run into a Polars frame for interactive plotting."""

    archive = Path(zarr_path)
    root = open_zarr_root(archive, mode="r")
    tables = load_eye_angle_run_tables(root, run_name=run_name)
    resolved_run = tables.run_name

    frame_rows = first_array_length(tables.frame) if tables.frame else 0
    if prefer_frame and tables.frame and frame_rows > 0:
        data_arrays = tables.frame
        row_axis = "frame"
        row_count = frame_rows
        qa_arrays = tables.qa_frame
        time_seconds = frame_time_seconds(tables, row_count=row_count)
        frame_indices = np.arange(row_count, dtype=np.int64)
    else:
        data_arrays = tables.roi
        row_axis = "roi"
        row_count = first_array_length(tables.roi)
        qa_arrays = tables.qa_roi
        time_seconds = roi_time_seconds(tables, row_count=row_count)
        frame_indices = roi_frame_indices(tables, row_count=row_count)

    if row_count <= 0:
        dataframe = pl.DataFrame(schema={"time_s": pl.Float64})
    else:
        if time_seconds is None:
            fps = _safe_float(tables.attrs.get("fps"))
            if frame_indices is not None and fps and fps > 0:
                time_seconds = frame_indices.astype(np.float64, copy=False) / fps
            elif fps and fps > 0:
                time_seconds = np.arange(row_count, dtype=np.float64) / fps
            else:
                time_seconds = np.arange(row_count, dtype=np.float64)
        columns: dict[str, np.ndarray] = {
            "time_s": np.asarray(time_seconds, dtype=np.float64)
        }
        if frame_indices is not None and len(frame_indices) == row_count:
            columns["frame_index"] = np.asarray(frame_indices, dtype=np.int64)
        else:
            columns["row_index"] = np.arange(row_count, dtype=np.int64)

        for column_name in EYE_ANGLE_TIMESERIES_COLUMNS:
            values = optional_1d_array(data_arrays, column_name, length=row_count)
            if values is not None:
                columns[column_name] = values.astype(np.float64, copy=False)

        for qa_name in ("valid_frame", "valid_left", "valid_right"):
            values = optional_1d_array(qa_arrays, qa_name, length=row_count)
            if values is not None:
                columns[qa_name] = values.astype(bool, copy=False)
        dataframe = pl.DataFrame(columns)

    return EyeAngleTimeseriesData(
        zarr_path=archive,
        run_name=resolved_run,
        run_path=tables.run_path,
        row_axis=row_axis,
        attrs=tables.attrs,
        dataframe=dataframe,
    )


__all__ = [
    "EyeAngleTimeseriesData",
    "discover_eye_angle_run_options",
    "load_eye_angle_timeseries_data",
]
