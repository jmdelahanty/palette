from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import zarr

from .schema import resolve_skeleton_identity_from_attrs
from fisheye.shared.runtime_config import runtime_config_dirs


@dataclass(frozen=True)
class DerivedMetricDefinition:
    name: str
    type: str
    from_label: str
    to_label: str
    units: str
    normalization: str
    description: str


@dataclass(frozen=True)
class KeypointMetricSchema:
    schema_name: str
    schema_version: str
    skeleton_id: str
    source_pose_schema: str
    metrics: tuple[DerivedMetricDefinition, ...]

    @property
    def metric_labels(self) -> list[str]:
        return [metric.name for metric in self.metrics]


@dataclass(frozen=True)
class DerivedMetricResults:
    values: np.ndarray
    values_norm: np.ndarray
    valid: np.ndarray


@dataclass(frozen=True)
class DerivedMetricStorage:
    schema: KeypointMetricSchema
    values: zarr.Array
    values_norm: zarr.Array
    valid: zarr.Array


def load_metric_schema(schema_path: Path) -> KeypointMetricSchema:
    with open(schema_path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    metrics = tuple(
        DerivedMetricDefinition(
            name=str(item["name"]),
            type=str(item["type"]),
            from_label=str(item["from_label"]),
            to_label=str(item["to_label"]),
            units=str(item.get("units", "px")),
            normalization=str(item.get("normalization", "none")),
            description=str(item.get("description", "")),
        )
        for item in raw.get("metrics", [])
    )
    return KeypointMetricSchema(
        schema_name=str(raw["schema_name"]),
        schema_version=str(raw["schema_version"]),
        skeleton_id=str(raw["skeleton_id"]),
        source_pose_schema=str(raw["source_pose_schema"]),
        metrics=metrics,
    )


def metric_schema_from_package(name: str, base_dir: Optional[Path] = None) -> KeypointMetricSchema:
    search_dirs: list[Path] = []
    if base_dir is not None:
        search_dirs.append(Path(base_dir))
    search_dirs.extend(runtime_config_dirs("keypoint_metric_schemas"))

    tried: list[Path] = []
    for directory in search_dirs:
        candidate = directory / f"{name}.json"
        tried.append(candidate)
        if candidate.exists():
            return load_metric_schema(candidate)
    tried_text = ", ".join(str(path) for path in tried)
    raise FileNotFoundError(f"Keypoint metric schema '{name}' not found. Tried: {tried_text}")


def resolve_metric_schema_for_group(group: zarr.Group, *, required: bool = False) -> KeypointMetricSchema | None:
    resolved = resolve_skeleton_identity_from_attrs(group.attrs)
    pose_schema_name = resolved.pose_schema_name
    skeleton_id = resolved.skeleton_id
    if pose_schema_name is None:
        return None
    try:
        schema = metric_schema_from_package(str(pose_schema_name))
    except FileNotFoundError:
        if required:
            raise
        return None
    if skeleton_id and str(skeleton_id) != str(schema.skeleton_id):
        raise ValueError(
            f"Metric schema skeleton_id mismatch: run has {skeleton_id!r}, "
            f"schema expects {schema.skeleton_id!r}."
        )
    return schema


def _keypoint_index_map(keypoint_labels: Sequence[str]) -> dict[str, int]:
    return {str(label): int(idx) for idx, label in enumerate(keypoint_labels)}


def validate_metric_schema_labels(schema: KeypointMetricSchema, *, keypoint_labels: Sequence[str]) -> None:
    label_map = _keypoint_index_map(keypoint_labels)
    missing = []
    for metric in schema.metrics:
        if metric.from_label not in label_map:
            missing.append(metric.from_label)
        if metric.to_label not in label_map:
            missing.append(metric.to_label)
    if missing:
        unique_missing = sorted(set(str(label) for label in missing))
        raise ValueError(f"Metric schema {schema.schema_name!r} references missing keypoint labels: {unique_missing}")


def compute_derived_metric_results(
    keypoints_roi: np.ndarray,
    *,
    keypoint_labels: Sequence[str],
    schema: KeypointMetricSchema,
    roi_diagonal: float | None,
) -> DerivedMetricResults:
    points = np.asarray(keypoints_roi, dtype=np.float64)
    squeeze = False
    if points.ndim == 2:
        points = points[None, :, :]
        squeeze = True
    if points.ndim != 3 or points.shape[2] != 2:
        raise ValueError(f"Expected keypoints with shape (N,K,2) or (K,2), got {points.shape}.")

    validate_metric_schema_labels(schema, keypoint_labels=keypoint_labels)
    label_map = _keypoint_index_map(keypoint_labels)
    n_rows = int(points.shape[0])
    n_metrics = int(len(schema.metrics))
    values = np.full((n_rows, n_metrics), np.nan, dtype=np.float32)
    values_norm = np.full((n_rows, n_metrics), np.nan, dtype=np.float32)
    valid = np.zeros((n_rows, n_metrics), dtype=bool)

    use_roi_diagonal = roi_diagonal is not None and np.isfinite(float(roi_diagonal)) and float(roi_diagonal) > 0
    roi_diag = float(roi_diagonal) if use_roi_diagonal else None

    for metric_idx, metric in enumerate(schema.metrics):
        from_idx = label_map[metric.from_label]
        to_idx = label_map[metric.to_label]
        src = points[:, from_idx, :]
        dst = points[:, to_idx, :]
        metric_valid = np.all(np.isfinite(src), axis=1) & np.all(np.isfinite(dst), axis=1)
        if not np.any(metric_valid):
            continue
        deltas = src - dst
        dist = np.sqrt(np.sum(np.square(deltas), axis=1, dtype=np.float64))
        dist[~metric_valid] = np.nan
        values[:, metric_idx] = dist.astype(np.float32, copy=False)
        valid[:, metric_idx] = metric_valid
        if metric.normalization == "roi_diagonal" and roi_diag is not None:
            values_norm[:, metric_idx] = (dist / roi_diag).astype(np.float32, copy=False)
        elif metric.normalization == "none":
            values_norm[:, metric_idx] = dist.astype(np.float32, copy=False)

    if squeeze:
        return DerivedMetricResults(values=values[0], values_norm=values_norm[0], valid=valid[0])
    return DerivedMetricResults(values=values, values_norm=values_norm, valid=valid)


def _coerce_chunk_len(value: object, row_count: int) -> int:
    try:
        chunk_len = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        chunk_len = 0
    if chunk_len <= 0:
        chunk_len = max(1, min(1024, int(row_count)))
    return int(chunk_len)


def ensure_derived_metric_storage(
    run_group: zarr.Group,
    *,
    schema: KeypointMetricSchema,
    row_count: int,
    chunk_len: int,
    roi_diagonal: float | None,
    overwrite: bool = False,
) -> DerivedMetricStorage:
    metric_count = int(len(schema.metrics))
    if metric_count <= 0:
        raise ValueError(f"Metric schema {schema.schema_name!r} defines no metrics.")

    resolved_chunk_len = _coerce_chunk_len(chunk_len, row_count)
    metric_chunks = (resolved_chunk_len, metric_count)
    arrays_present = all(name in run_group for name in ("derived_metric_values", "derived_metric_values_norm", "derived_metric_valid"))
    if arrays_present and not overwrite:
        values_arr = run_group["derived_metric_values"]
        values_norm_arr = run_group["derived_metric_values_norm"]
        valid_arr = run_group["derived_metric_valid"]
    else:
        if overwrite:
            for name in ("derived_metric_values", "derived_metric_values_norm", "derived_metric_valid"):
                if name in run_group:
                    del run_group[name]
        values_arr = run_group.create_array(
            "derived_metric_values",
            shape=(int(row_count), metric_count),
            chunks=metric_chunks,
            dtype="f4",
            fill_value=np.nan,
            overwrite=overwrite,
        )
        values_norm_arr = run_group.create_array(
            "derived_metric_values_norm",
            shape=(int(row_count), metric_count),
            chunks=metric_chunks,
            dtype="f4",
            fill_value=np.nan,
            overwrite=overwrite,
        )
        valid_arr = run_group.create_array(
            "derived_metric_valid",
            shape=(int(row_count), metric_count),
            chunks=metric_chunks,
            dtype="bool",
            fill_value=False,
            overwrite=overwrite,
        )

    run_group.attrs["derived_metric_schema_id"] = schema.schema_name
    run_group.attrs["derived_metric_schema_version"] = schema.schema_version
    run_group.attrs["derived_metric_labels"] = schema.metric_labels
    run_group.attrs["derived_metric_type"] = "named_keypoint_derivations"
    run_group.attrs["derived_metric_source"] = "keypoint_metric_schema"
    run_group.attrs["derived_metric_count"] = metric_count
    run_group.attrs["derived_metric_normalization"] = {
        "mode": "roi_diagonal",
        "roi_diagonal": float(roi_diagonal) if roi_diagonal is not None and np.isfinite(float(roi_diagonal)) else None,
    }
    run_group.attrs["derived_metric_definitions"] = [
        {
            "name": metric.name,
            "type": metric.type,
            "from_label": metric.from_label,
            "to_label": metric.to_label,
            "units": metric.units,
            "normalization": metric.normalization,
            "description": metric.description,
        }
        for metric in schema.metrics
    ]
    return DerivedMetricStorage(
        schema=schema,
        values=values_arr,
        values_norm=values_norm_arr,
        valid=valid_arr,
    )


def update_derived_metric_rows(
    storage: DerivedMetricStorage,
    *,
    row_indexer,
    keypoints_roi: np.ndarray,
    keypoint_labels: Sequence[str],
    roi_diagonal: float | None,
) -> None:
    result = compute_derived_metric_results(
        keypoints_roi,
        keypoint_labels=keypoint_labels,
        schema=storage.schema,
        roi_diagonal=roi_diagonal,
    )
    storage.values[row_indexer] = result.values
    storage.values_norm[row_indexer] = result.values_norm
    storage.valid[row_indexer] = result.valid
