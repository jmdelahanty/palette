"""Validate and conservatively backfill refined subject-mask contract fields."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import zarr

from fisheye.shared.detect_reason_codec import write_reason_columns
from fisheye.shared.mask_store import MaskStoreError, open_mask_store
from fisheye.shared.row_lineage import direct_source_crop_row_ids
from fisheye.shared.subject_mask_chunks import refined_subject_mask_metric_row_chunk
from fisheye.shared.zarr_io import open_zarr_root

REQUIRED_COMPONENTS = ("subject_body", "eye_left", "eye_right", "swim_bladder")
REQUIRED_RUN_ARRAYS = (
    "frame_indices",
    "source_crop_row_ids",
    "detection_source",
    "available_channels",
    "edit_applied",
)
REQUIRED_RUN_METRICS = ("mask_present", "area_px")
RECOMMENDED_GEOMETRY_METRICS = ("centroid_xy", "centroid_valid", "bbox_xyxy", "bbox_valid")
REQUIRED_COMPONENT_ARRAYS = ("reason_bytes", "mask_present", "area_px", "edit_applied")
REQUIRED_RUN_ATTRS = (
    "source_subject_mask_run",
    "source_crop_run",
    "label_schema_id",
    "mask_labels",
    "output_semantics",
    "refinement_semantics",
    "method",
    "created_at_utc",
    "duration_seconds",
    "refined_subject_mask_review_status",
    "component_review_statuses",
)
DIRECT_CROP_ROW_BACKFILL_CHECK_ARRAYS = (
    "frame_indices",
    "detection_indices",
    "source_refined_row_ids",
    "source_detect_row_index",
)


@dataclass(frozen=True)
class ContractIssue:
    severity: str
    code: str
    path: str
    message: str
    backfillable: bool = False

    def to_json(self) -> dict[str, object]:
        return {
            "severity": self.severity,
            "code": self.code,
            "path": self.path,
            "message": self.message,
            "backfillable": bool(self.backfillable),
        }


def _is_array(node: object) -> bool:
    return isinstance(node, zarr.Array) or (hasattr(node, "shape") and hasattr(node, "dtype"))


def _is_group(node: object) -> bool:
    return isinstance(node, zarr.Group) or (hasattr(node, "attrs") and hasattr(node, "keys") and not _is_array(node))


def _normalize_labels(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            return [part.strip() for part in text.split(",") if part.strip()]
        return _normalize_labels(decoded)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return []


def _as_mapping(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return {str(k): v for k, v in value.items()}
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(decoded, Mapping):
            return {str(k): v for k, v in decoded.items()}
    return {}


def _array_shape(group: zarr.Group, path: str) -> Optional[tuple[int, ...]]:
    node = group.get(path)
    if not _is_array(node):
        return None
    return tuple(int(dim) for dim in node.shape)


def _add_issue(
    issues: list[ContractIssue],
    *,
    severity: str,
    code: str,
    path: str,
    message: str,
    backfillable: bool = False,
) -> None:
    issues.append(
        ContractIssue(
            severity=severity,
            code=code,
            path=path,
            message=message,
            backfillable=backfillable,
        )
    )


def _metric_chunks(total_rows: int) -> tuple[int, ...]:
    return (refined_subject_mask_metric_row_chunk(total_rows),)


def _metric_chunks_2d(total_rows: int) -> tuple[int, int]:
    return (refined_subject_mask_metric_row_chunk(total_rows), 1)


def _metric_chunks_lastdim(total_rows: int, width: int) -> tuple[int, int, int]:
    return (refined_subject_mask_metric_row_chunk(total_rows), 1, int(width))


def _resolve_refined_run(root: zarr.Group, run_name: str) -> tuple[str, zarr.Group]:
    parent = root.get("refined_subject_masks_runs")
    if not _is_group(parent):
        raise ValueError("Missing refined_subject_masks_runs group.")
    run_name = str(run_name).strip("/")
    if run_name.startswith("refined_subject_masks_runs/"):
        run_name = run_name.split("/", 1)[1].strip("/")
    if run_name == "latest":
        latest = parent.attrs.get("latest")
        if not isinstance(latest, str) or not latest:
            raise ValueError('Missing refined_subject_masks_runs.attrs["latest"].')
        run_name = latest
    if run_name not in parent:
        raise ValueError(f"Missing refined_subject_masks_runs/{run_name}.")
    run = parent[run_name]
    if not _is_group(run):
        raise ValueError(f"refined_subject_masks_runs/{run_name} is not a group.")
    return str(run_name), run


def _compute_mask_geometry_chunk(masks: np.ndarray) -> dict[str, np.ndarray]:
    binary = np.asarray(masks, dtype=np.uint8) > 0
    if binary.ndim != 4:
        raise ValueError(f"Expected masks with shape (N,C,H,W), got {tuple(binary.shape)}")
    rows, channels = int(binary.shape[0]), int(binary.shape[1])
    area_px = binary.reshape(rows, channels, -1).sum(axis=2, dtype=np.int64).astype(np.float32)
    mask_present = area_px > 0
    centroid_xy = np.zeros((rows, channels, 2), dtype=np.float32)
    centroid_valid = np.zeros((rows, channels), dtype=bool)
    bbox_xyxy = np.zeros((rows, channels, 4), dtype=np.float32)
    bbox_valid = np.zeros((rows, channels), dtype=bool)

    for row_idx in range(rows):
        for channel_idx in range(channels):
            if not bool(mask_present[row_idx, channel_idx]):
                continue
            ys, xs = np.nonzero(binary[row_idx, channel_idx])
            centroid_xy[row_idx, channel_idx] = np.asarray([xs.mean(), ys.mean()], dtype=np.float32)
            centroid_valid[row_idx, channel_idx] = True
            bbox_xyxy[row_idx, channel_idx] = np.asarray(
                [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
                dtype=np.float32,
            )
            bbox_valid[row_idx, channel_idx] = True

    return {
        "mask_present": mask_present.astype(bool, copy=False),
        "area_px": area_px,
        "centroid_xy": centroid_xy,
        "centroid_valid": centroid_valid,
        "bbox_xyxy": bbox_xyxy,
        "bbox_valid": bbox_valid,
    }


def _validate_mask_store_decodable(mask_store: Any, *, run_path: str, issues: list[ContractIssue]) -> None:
    """Exercise compact stores through the same dense materialization API consumers use."""

    if getattr(mask_store, "storage_surface", "") != "mask_rle":
        return
    total_rows = int(mask_store.n_rows)
    channels = int(mask_store.n_channels)
    height, width = (int(value) for value in mask_store.shape_hw)
    chunk_rows = refined_subject_mask_metric_row_chunk(total_rows)
    try:
        for start in range(0, total_rows, chunk_rows):
            stop = min(total_rows, start + chunk_rows)
            decoded = mask_store.read_dense(rows=slice(start, stop))
            expected_shape = (int(stop - start), channels, height, width)
            if tuple(int(value) for value in decoded.shape) != expected_shape:
                _add_issue(
                    issues,
                    severity="error",
                    code="invalid_mask_store",
                    path=f"{run_path}/mask_rle",
                    message=(
                        "compact mask_rle decoded to unexpected shape "
                        f"{tuple(decoded.shape)} for rows {start}:{stop}, expected {expected_shape}."
                    ),
                )
                return
    except Exception as exc:
        _add_issue(
            issues,
            severity="error",
            code="invalid_mask_store",
            path=f"{run_path}/mask_rle",
            message=f"compact mask_rle failed dense materialization: {exc}",
        )


def _validate_source_crop_row_mapping(
    root: zarr.Group,
    run: zarr.Group,
    *,
    run_path: str,
    total_rows: Optional[int],
    issues: list[ContractIssue],
) -> None:
    if total_rows is None:
        return
    crop_run = run.attrs.get("source_crop_run")
    if not isinstance(crop_run, str) or not crop_run.strip():
        return
    crop_group = root.get(f"crop_runs/{crop_run}")
    if not _is_group(crop_group):
        _add_issue(
            issues,
            severity="error",
            code="missing_source_crop_run",
            path=f"crop_runs/{crop_run}",
            message=f"source_crop_run {crop_run!r} is not present in this archive.",
        )
        return
    crop_frames_node = crop_group.get("frame_indices")
    frame_node = run.get("frame_indices")
    crop_row_node = run.get("source_crop_row_ids")
    if not (_is_array(crop_frames_node) and _is_array(frame_node) and _is_array(crop_row_node)):
        return
    frame_indices = np.asarray(frame_node[:], dtype=np.int64).reshape(-1)
    crop_row_ids = np.asarray(crop_row_node[:], dtype=np.int64).reshape(-1)
    crop_frames = np.asarray(crop_frames_node[:], dtype=np.int64).reshape(-1)
    if int(frame_indices.shape[0]) != int(total_rows):
        _add_issue(
            issues,
            severity="error",
            code="row_count_mismatch",
            path=f"{run_path}/frame_indices",
            message=f"frame_indices length {frame_indices.shape[0]} != mask rows {total_rows}.",
        )
        return
    if int(crop_row_ids.shape[0]) != int(total_rows):
        _add_issue(
            issues,
            severity="error",
            code="row_count_mismatch",
            path=f"{run_path}/source_crop_row_ids",
            message=f"source_crop_row_ids length {crop_row_ids.shape[0]} != mask rows {total_rows}.",
        )
        return
    if crop_row_ids.size and (int(crop_row_ids.min()) < 0 or int(crop_row_ids.max()) >= int(crop_frames.shape[0])):
        _add_issue(
            issues,
            severity="error",
            code="source_crop_row_out_of_bounds",
            path=f"{run_path}/source_crop_row_ids",
            message="source_crop_row_ids contains rows outside crop_runs/<source_crop_run>/frame_indices.",
        )
        return
    if not np.array_equal(crop_frames[crop_row_ids], frame_indices):
        _add_issue(
            issues,
            severity="error",
            code="source_crop_frame_mismatch",
            path=f"{run_path}/source_crop_row_ids",
            message=(
                "crop_runs/<source_crop_run>/frame_indices[source_crop_row_ids] "
                "does not match refined frame_indices."
            ),
        )


def _direct_source_crop_row_ids_backfillable(
    root: zarr.Group,
    run: zarr.Group,
    *,
    total_rows: Optional[int],
) -> tuple[bool, str]:
    if total_rows is None:
        return False, "mask row count is unknown"
    crop_run = run.attrs.get("source_crop_run")
    if not isinstance(crop_run, str) or not crop_run.strip():
        return False, "source_crop_run attr is missing"
    crop_group = root.get(f"crop_runs/{crop_run}")
    if not _is_group(crop_group):
        return False, f"crop_runs/{crop_run} is missing"
    for array_name in DIRECT_CROP_ROW_BACKFILL_CHECK_ARRAYS:
        crop_node = crop_group.get(array_name)
        run_node = run.get(array_name)
        if not (_is_array(crop_node) and _is_array(run_node)):
            if array_name == "frame_indices":
                return False, "frame_indices is required on both run and crop for direct-row backfill"
            continue
        if tuple(int(dim) for dim in crop_node.shape) != (int(total_rows),):
            return False, f"crop {array_name} shape {tuple(crop_node.shape)} is not ({total_rows},)"
        if tuple(int(dim) for dim in run_node.shape) != (int(total_rows),):
            return False, f"run {array_name} shape {tuple(run_node.shape)} is not ({total_rows},)"
        if not np.array_equal(np.asarray(crop_node[:]), np.asarray(run_node[:])):
            return False, f"run {array_name} does not match crop {array_name}"
    return True, "run rows match source crop rows exactly"


def _backfill_direct_source_crop_row_ids(
    root: zarr.Group,
    run: zarr.Group,
    *,
    total_rows: Optional[int],
    summary: dict[str, object],
) -> bool:
    if "source_crop_row_ids" in run:
        return False
    ok, reason = _direct_source_crop_row_ids_backfillable(root, run, total_rows=total_rows)
    if not ok:
        raise ValueError(f"Cannot backfill source_crop_row_ids: {reason}.")
    data = direct_source_crop_row_ids(int(total_rows))
    run.create_array(
        "source_crop_row_ids",
        data=data,
        chunks=_metric_chunks(int(total_rows)),
        overwrite=True,
    )
    summary.setdefault("backfilled", []).append("source_crop_row_ids")
    summary["source_crop_row_ids_backfill_policy"] = "direct_row_identity_after_matching_crop_row_arrays"
    summary["source_crop_row_ids_backfill_reason"] = reason
    return True


def _ensure_array(
    group: zarr.Group,
    name: str,
    *,
    shape: tuple[int, ...],
    dtype: object,
    chunks: tuple[int, ...],
    fill_value: object = 0,
) -> zarr.Array:
    existing = group.get(name)
    if _is_array(existing):
        return existing
    return group.create_array(name, shape=shape, dtype=dtype, chunks=chunks, fill_value=fill_value, overwrite=True)


def _backfill_available_channels(run: zarr.Group, labels: Sequence[str], summary: dict[str, object]) -> bool:
    if "available_channels" in run:
        return False
    component_reviews = _as_mapping(run.attrs.get("component_review_statuses"))
    components_group = run.get("components")
    if not _is_group(components_group) and not component_reviews:
        raise ValueError("Cannot backfill available_channels without component groups or component_review_statuses.")
    available = []
    for label in labels:
        declared = False
        if _is_group(components_group) and label in components_group:
            declared = True
        if label in component_reviews:
            declared = True
        available.append(declared)
    if not any(available):
        raise ValueError("Cannot backfill available_channels because no declared component availability was found.")
    run.create_array("available_channels", data=np.asarray(available, dtype=bool), overwrite=True)
    summary.setdefault("backfilled", []).append("available_channels")
    return True


def _component_mask_array(component_group: zarr.Group) -> Optional[zarr.Array]:
    for name in ("masks_roi", "mask_roi", "mask"):
        node = component_group.get(name)
        if _is_array(node):
            return node
    return None


def _backfill_masks_roi_from_components(
    run: zarr.Group,
    labels: Sequence[str],
    available: np.ndarray,
    summary: dict[str, object],
) -> bool:
    if "masks_roi" in run:
        return False
    try:
        open_mask_store(run, source_path="refined_subject_masks_runs/<backfill>", prefer="dense")
        return False
    except (MaskStoreError, ValueError):
        pass
    components_group = run.get("components")
    if not _is_group(components_group):
        raise ValueError("Cannot backfill masks_roi without components/<label>/ mask arrays.")

    component_arrays: list[Optional[zarr.Array]] = []
    row_count: Optional[int] = None
    height: Optional[int] = None
    width: Optional[int] = None
    for idx, label in enumerate(labels):
        if idx >= int(available.shape[0]) or not bool(available[idx]):
            component_arrays.append(None)
            continue
        if label == "eyes_union":
            raise ValueError("Refusing to backfill refined LR masks from eyes_union.")
        if label not in components_group:
            raise ValueError(f"Cannot backfill masks_roi: components/{label} is missing.")
        arr = _component_mask_array(components_group[label])
        if arr is None:
            raise ValueError(f"Cannot backfill masks_roi: components/{label} has no mask array.")
        shape = tuple(int(dim) for dim in arr.shape)
        if len(shape) == 4 and shape[1] == 1:
            shape = (shape[0], shape[2], shape[3])
        if len(shape) != 3:
            raise ValueError(f"Cannot backfill masks_roi: components/{label} mask shape is {tuple(arr.shape)}.")
        if row_count is None:
            row_count, height, width = shape
        elif (row_count, height, width) != shape:
            raise ValueError("Cannot backfill masks_roi: component mask shapes do not match.")
        component_arrays.append(arr)

    if row_count is None or height is None or width is None:
        raise ValueError("Cannot backfill masks_roi: no available component mask arrays were found.")
    masks = run.create_array(
        "masks_roi",
        shape=(int(row_count), len(labels), int(height), int(width)),
        dtype=np.uint8,
        chunks=(min(16, int(row_count)), 1, int(height), int(width)),
        fill_value=0,
        overwrite=True,
    )
    for channel_idx, arr in enumerate(component_arrays):
        if arr is None:
            continue
        values = np.asarray(arr[:], dtype=np.uint8)
        if values.ndim == 4:
            values = values[:, 0]
        masks[:, channel_idx] = values
    summary.setdefault("backfilled", []).append("masks_roi")
    return True


def _backfill_run_metrics(run: zarr.Group, labels: Sequence[str], summary: dict[str, object]) -> bool:
    try:
        mask_store = open_mask_store(run, source_path="refined_subject_masks_runs/<backfill>", prefer="dense")
    except (MaskStoreError, ValueError):
        return False
    shape = tuple(int(dim) for dim in mask_store.shape)
    if len(shape) != 4:
        return False
    total_rows, channels = shape[0], shape[1]
    if labels and int(channels) != len(labels):
        raise ValueError(f"Cannot backfill metrics: mask channels {channels} != len(mask_labels) {len(labels)}.")
    metrics = run.require_group("metrics")
    targets = {
        "mask_present": ((total_rows, channels), bool, _metric_chunks_2d(total_rows), False),
        "area_px": ((total_rows, channels), np.float32, _metric_chunks_2d(total_rows), 0.0),
        "centroid_xy": ((total_rows, channels, 2), np.float32, _metric_chunks_lastdim(total_rows, 2), np.nan),
        "centroid_valid": ((total_rows, channels), bool, _metric_chunks_2d(total_rows), False),
        "bbox_xyxy": ((total_rows, channels, 4), np.float32, _metric_chunks_lastdim(total_rows, 4), 0.0),
        "bbox_valid": ((total_rows, channels), bool, _metric_chunks_2d(total_rows), False),
    }
    missing = [name for name in targets if name not in metrics]
    if not missing:
        return False

    arrays = {
        name: _ensure_array(metrics, name, shape=target_shape, dtype=dtype, chunks=chunks, fill_value=fill_value)
        for name, (target_shape, dtype, chunks, fill_value) in targets.items()
        if name in missing
    }
    chunk_rows = refined_subject_mask_metric_row_chunk(total_rows)
    for start in range(0, total_rows, chunk_rows):
        stop = min(total_rows, start + chunk_rows)
        computed = _compute_mask_geometry_chunk(mask_store.read_dense(rows=slice(start, stop)))
        for name, arr in arrays.items():
            arr[start:stop] = computed[name]
    summary.setdefault("backfilled", []).extend(f"metrics/{name}" for name in missing)
    return True


def _backfill_component_arrays(
    run: zarr.Group,
    labels: Sequence[str],
    available: np.ndarray,
    summary: dict[str, object],
) -> bool:
    try:
        mask_store = open_mask_store(run, source_path="refined_subject_masks_runs/<backfill>", prefer="dense")
    except (MaskStoreError, ValueError):
        return False
    masks_shape = tuple(int(dim) for dim in mask_store.shape)
    if len(masks_shape) != 4:
        return False
    if labels and int(masks_shape[1]) != len(labels):
        raise ValueError(
            f"Cannot backfill component arrays: mask channels {masks_shape[1]} != len(mask_labels) {len(labels)}."
        )
    total_rows = int(masks_shape[0])
    components = run.require_group("components")
    metrics = run.get("metrics")
    if not _is_group(metrics):
        return False

    wrote = False
    for channel_idx, label in enumerate(labels):
        if channel_idx >= int(available.shape[0]) or not bool(available[channel_idx]):
            continue
        component = components.require_group(label)
        if "mask_present" not in component and "mask_present" in metrics:
            component.create_array(
                "mask_present",
                data=np.asarray(metrics["mask_present"][:, channel_idx], dtype=bool),
                chunks=_metric_chunks(total_rows),
                overwrite=True,
            )
            summary.setdefault("backfilled", []).append(f"components/{label}/mask_present")
            wrote = True
        if "area_px" not in component and "area_px" in metrics:
            component.create_array(
                "area_px",
                data=np.asarray(metrics["area_px"][:, channel_idx], dtype=np.float32),
                chunks=_metric_chunks(total_rows),
                overwrite=True,
            )
            summary.setdefault("backfilled", []).append(f"components/{label}/area_px")
            wrote = True
        if "edit_applied" not in component and "edit_applied" in run:
            component.create_array(
                "edit_applied",
                data=np.asarray(run["edit_applied"][:, channel_idx], dtype=bool),
                chunks=_metric_chunks(total_rows),
                overwrite=True,
            )
            summary.setdefault("backfilled", []).append(f"components/{label}/edit_applied")
            wrote = True
        if "reason_bytes" not in component and "reason" in component:
            write_reason_columns(
                component,
                np.asarray(component["reason"][:], dtype=str),
                refined_subject_mask_metric_row_chunk(total_rows),
                include_reason_text=True,
                overwrite=True,
            )
            summary.setdefault("backfilled", []).append(f"components/{label}/reason_bytes")
            wrote = True
    return wrote


def backfill_refined_subject_mask_contract(
    zarr_path: Path | str,
    *,
    run_name: str = "latest",
) -> dict[str, object]:
    root = open_zarr_root(zarr_path, mode="a")
    resolved_run, run = _resolve_refined_run(root, run_name)
    summary: dict[str, object] = {"run_name": resolved_run, "backfilled": []}
    labels = _normalize_labels(run.attrs.get("mask_labels"))
    if not labels:
        raise ValueError("Cannot backfill without mask_labels.")
    try:
        mask_store = open_mask_store(run, source_path=f"refined_subject_masks_runs/{resolved_run}", prefer="dense")
        masks_shape = tuple(int(dim) for dim in mask_store.shape)
        total_rows = int(masks_shape[0]) if len(masks_shape) == 4 else None
    except (MaskStoreError, ValueError):
        total_rows = _array_shape(run, "masks_roi")
        total_rows = int(total_rows[0]) if total_rows is not None and len(total_rows) == 4 else None
    _backfill_direct_source_crop_row_ids(root, run, total_rows=total_rows, summary=summary)
    _backfill_available_channels(run, labels, summary)
    available_node = run.get("available_channels")
    if not _is_array(available_node):
        raise ValueError("available_channels is still missing after backfill attempt.")
    available = np.asarray(available_node[:], dtype=bool).reshape(-1)
    _backfill_masks_roi_from_components(run, labels, available, summary)
    _backfill_run_metrics(run, labels, summary)
    _backfill_component_arrays(run, labels, available, summary)
    summary["backfill_count"] = len(summary.get("backfilled", []))
    return summary


def validate_refined_subject_mask_contract(
    zarr_path: Path | str,
    *,
    run_name: str = "latest",
    required_components: Sequence[str] = REQUIRED_COMPONENTS,
) -> dict[str, object]:
    root = open_zarr_root(zarr_path, mode="r")
    issues: list[ContractIssue] = []
    try:
        resolved_run, run = _resolve_refined_run(root, run_name)
    except ValueError as exc:
        _add_issue(
            issues,
            severity="error",
            code="missing_run",
            path="refined_subject_masks_runs",
            message=str(exc),
        )
        return _summary(zarr_path, run_name, None, [], [], issues)

    run_path = f"refined_subject_masks_runs/{resolved_run}"
    labels = _normalize_labels(run.attrs.get("mask_labels"))
    mask_store = None
    for attr in REQUIRED_RUN_ATTRS:
        if attr not in run.attrs:
            _add_issue(
                issues,
                severity="error",
                code="missing_attr",
                path=f"{run_path}.attrs[{attr!r}]",
                message=f"Missing required run attr {attr!r}.",
            )
    if labels:
        if len(labels) != len(set(labels)):
            _add_issue(
                issues,
                severity="error",
                code="duplicate_mask_labels",
                path=f"{run_path}.attrs['mask_labels']",
                message=f"mask_labels contains duplicates: {labels!r}.",
            )
    else:
        _add_issue(
            issues,
            severity="error",
            code="missing_mask_labels",
            path=f"{run_path}.attrs['mask_labels']",
            message="mask_labels is missing or empty.",
        )

    masks_shape = None
    try:
        mask_store = open_mask_store(run, source_path=run_path, prefer="dense")
        masks_shape = tuple(int(dim) for dim in mask_store.shape)
        if len(masks_shape) != 4:
            _add_issue(
                issues,
                severity="error",
                code="shape_mismatch",
                path=f"{run_path}/mask_store",
                message=f"mask store must materialize as 4D (N,C,H,W), got {masks_shape}.",
            )
        elif labels and int(mask_store.n_channels) != len(labels):
            _add_issue(
                issues,
                severity="error",
                code="channel_label_mismatch",
                path=f"{run_path}/mask_store",
                message=f"mask store channel count {mask_store.n_channels} != len(mask_labels) {len(labels)}.",
            )
        if labels and tuple(mask_store.mask_labels) != tuple(labels):
            _add_issue(
                issues,
                severity="error",
                code="mask_store_label_mismatch",
                path=f"{run_path}/mask_store",
                message=f"mask store labels {list(mask_store.mask_labels)!r} != mask_labels {labels!r}.",
            )
        _validate_mask_store_decodable(mask_store, run_path=run_path, issues=issues)
    except (MaskStoreError, ValueError) as exc:
        _add_issue(
            issues,
            severity="error",
            code="missing_mask_store",
            path=f"{run_path}/mask_store",
            message=f"Run must provide dense masks_roi or compact mask_rle storage ({exc}).",
            backfillable=True,
        )

    if masks_shape is None:
        masks_shape = _array_shape(run, "masks_roi")
    total_rows = int(masks_shape[0]) if masks_shape is not None and len(masks_shape) == 4 else None
    channel_count = int(masks_shape[1]) if masks_shape is not None and len(masks_shape) == 4 else None

    for name in REQUIRED_RUN_ARRAYS:
        node = run.get(name)
        if not _is_array(node):
            backfillable = name == "available_channels"
            if name == "source_crop_row_ids":
                backfillable, _reason = _direct_source_crop_row_ids_backfillable(root, run, total_rows=total_rows)
            _add_issue(
                issues,
                severity="error",
                code="missing_array",
                path=f"{run_path}/{name}",
                message=f"Missing required array {name!r}.",
                backfillable=backfillable,
            )
            continue
        shape = tuple(int(dim) for dim in node.shape)
        if name == "detection_source" and masks_shape is not None and len(shape) == 1 and len(masks_shape) == 4:
            if int(shape[0]) != int(masks_shape[0]):
                _add_issue(
                    issues,
                    severity="error",
                    code="row_count_mismatch",
                    path=f"{run_path}/detection_source",
                    message=f"detection_source length {shape[0]} != mask rows {masks_shape[0]}.",
                )
        if name in {"frame_indices", "source_crop_row_ids"} and masks_shape is not None and len(shape) == 1 and len(masks_shape) == 4:
            if int(shape[0]) != int(masks_shape[0]):
                _add_issue(
                    issues,
                    severity="error",
                    code="row_count_mismatch",
                    path=f"{run_path}/{name}",
                    message=f"{name} length {shape[0]} != mask rows {masks_shape[0]}.",
                )

    _validate_source_crop_row_mapping(root, run, run_path=run_path, total_rows=total_rows, issues=issues)

    available = None
    available_node = run.get("available_channels")
    if _is_array(available_node):
        available = np.asarray(available_node[:], dtype=bool).reshape(-1)
        if labels and int(available.shape[0]) != len(labels):
            _add_issue(
                issues,
                severity="error",
                code="available_label_mismatch",
                path=f"{run_path}/available_channels",
                message=f"available_channels length {available.shape[0]} != len(mask_labels) {len(labels)}.",
            )
        if channel_count is not None and int(available.shape[0]) != channel_count:
            _add_issue(
                issues,
                severity="error",
                code="available_channel_mismatch",
                path=f"{run_path}/available_channels",
                message=f"available_channels length {available.shape[0]} != mask channels {channel_count}.",
            )

    edit_shape = _array_shape(run, "edit_applied")
    if edit_shape is not None and total_rows is not None and channel_count is not None:
        if edit_shape != (total_rows, channel_count):
            _add_issue(
                issues,
                severity="error",
                code="shape_mismatch",
                path=f"{run_path}/edit_applied",
                message=f"edit_applied shape {edit_shape} != {(total_rows, channel_count)}.",
            )

    metrics = run.get("metrics")
    if not _is_group(metrics):
        _add_issue(
            issues,
            severity="error",
            code="missing_group",
            path=f"{run_path}/metrics",
            message="Missing required metrics group.",
            backfillable=total_rows is not None and channel_count is not None,
        )
    else:
        expected_2d = (total_rows, channel_count) if total_rows is not None and channel_count is not None else None
        for metric in REQUIRED_RUN_METRICS:
            shape = _array_shape(metrics, metric)
            if shape is None:
                _add_issue(
                    issues,
                    severity="error",
                    code="missing_metric",
                    path=f"{run_path}/metrics/{metric}",
                    message=f"Missing required metrics/{metric}.",
                    backfillable=True,
                )
            elif expected_2d is not None and shape != expected_2d:
                _add_issue(
                    issues,
                    severity="error",
                    code="shape_mismatch",
                    path=f"{run_path}/metrics/{metric}",
                    message=f"metrics/{metric} shape {shape} != {expected_2d}.",
                )
        expected_geometry = {
            "centroid_xy": (total_rows, channel_count, 2) if expected_2d is not None else None,
            "centroid_valid": expected_2d,
            "bbox_xyxy": (total_rows, channel_count, 4) if expected_2d is not None else None,
            "bbox_valid": expected_2d,
        }
        for metric in RECOMMENDED_GEOMETRY_METRICS:
            shape = _array_shape(metrics, metric)
            if shape is None:
                _add_issue(
                    issues,
                    severity="warning",
                    code="missing_geometry_metric",
                    path=f"{run_path}/metrics/{metric}",
                    message=f"Missing recommended geometry metric metrics/{metric}.",
                    backfillable=True,
                )
            elif expected_geometry[metric] is not None and shape != expected_geometry[metric]:
                _add_issue(
                    issues,
                    severity="error",
                    code="shape_mismatch",
                    path=f"{run_path}/metrics/{metric}",
                    message=f"metrics/{metric} shape {shape} != {expected_geometry[metric]}.",
                )

    available_components: list[str] = []
    if labels and available is not None:
        available_components = [
            label for idx, label in enumerate(labels) if idx < int(available.shape[0]) and bool(available[idx])
        ]
        for component in required_components:
            if component not in labels:
                _add_issue(
                    issues,
                    severity="error",
                    code="missing_component_label",
                    path=f"{run_path}.attrs['mask_labels']",
                    message=f"Required component {component!r} is absent from mask_labels.",
                )
            elif not bool(available[labels.index(component)]):
                _add_issue(
                    issues,
                    severity="error",
                    code="component_unavailable",
                    path=f"{run_path}/available_channels",
                    message=f"Required component {component!r} is present in mask_labels but not available.",
                )

    review_statuses = _as_mapping(run.attrs.get("component_review_statuses"))
    for component in required_components:
        if component not in review_statuses:
            _add_issue(
                issues,
                severity="error",
                code="missing_component_review_status",
                path=f"{run_path}.attrs['component_review_statuses']",
                message=f"Missing component_review_statuses entry for {component!r}.",
            )

    components_group = run.get("components")
    if not _is_group(components_group):
        _add_issue(
            issues,
            severity="error",
            code="missing_group",
            path=f"{run_path}/components",
            message="Missing components group.",
            backfillable=False,
        )
    else:
        for component in available_components:
            component_path = f"{run_path}/components/{component}"
            if component not in components_group:
                _add_issue(
                    issues,
                    severity="error",
                    code="missing_component_group",
                    path=component_path,
                    message=f"Available component {component!r} has no component subgroup.",
                    backfillable=True,
                )
                continue
            component_group = components_group[component]
            provenance = component_group.get("provenance")
            if not _is_group(provenance):
                _add_issue(
                    issues,
                    severity="error",
                    code="missing_component_provenance",
                    path=f"{component_path}/provenance",
                    message=f"Available component {component!r} has no provenance subgroup.",
                )
            for arr_name in REQUIRED_COMPONENT_ARRAYS:
                shape = _array_shape(component_group, arr_name)
                if shape is None:
                    _add_issue(
                        issues,
                        severity="error",
                        code="missing_component_array",
                        path=f"{component_path}/{arr_name}",
                        message=f"Available component {component!r} missing {arr_name}.",
                        backfillable=arr_name in {"mask_present", "area_px", "edit_applied"}
                        or (arr_name == "reason_bytes" and "reason" in component_group),
                    )
                    continue
                if total_rows is not None:
                    if arr_name == "reason_bytes":
                        if len(shape) != 2 or shape[0] != total_rows:
                            _add_issue(
                                issues,
                                severity="error",
                                code="shape_mismatch",
                                path=f"{component_path}/{arr_name}",
                                message=f"{arr_name} shape {shape} is not (N,width) with N={total_rows}.",
                            )
                    elif shape != (total_rows,):
                        _add_issue(
                            issues,
                            severity="error",
                            code="shape_mismatch",
                            path=f"{component_path}/{arr_name}",
                            message=f"{arr_name} shape {shape} != {(total_rows,)}.",
                        )

    return _summary(zarr_path, run_name, resolved_run, labels, available_components, issues)


def _summary(
    zarr_path: Path | str,
    requested_run: str,
    resolved_run: Optional[str],
    labels: Sequence[str],
    available_components: Sequence[str],
    issues: Sequence[ContractIssue],
) -> dict[str, object]:
    errors = [issue for issue in issues if issue.severity == "error"]
    warnings = [issue for issue in issues if issue.severity == "warning"]
    return {
        "zarr_path": str(zarr_path),
        "requested_run": requested_run,
        "run_name": resolved_run,
        "valid": not errors,
        "backfill_needed": any(issue.backfillable for issue in issues),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "mask_labels": list(labels),
        "available_components": list(available_components),
        "errors": [issue.to_json() for issue in errors],
        "warnings": [issue.to_json() for issue in warnings],
    }


def _format_issue_lines(prefix: str, issues: Iterable[Mapping[str, object]]) -> list[str]:
    lines = []
    for issue in issues:
        lines.append(
            f"{prefix} {issue.get('code')}: {issue.get('path')}: {issue.get('message')}"
            + (" [backfillable]" if issue.get("backfillable") else "")
        )
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Palette analysis zarr archive.")
    parser.add_argument(
        "--run",
        default="latest",
        help='Refined subject-mask run name, path suffix, or "latest" (default).',
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Apply conservative backfills for fields derivable from existing masks/labels.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON.",
    )
    args = parser.parse_args(argv)

    backfill_summary: Optional[dict[str, object]] = None
    if args.backfill:
        backfill_summary = backfill_refined_subject_mask_contract(args.zarr_path, run_name=str(args.run))
    summary = validate_refined_subject_mask_contract(args.zarr_path, run_name=str(args.run))
    if backfill_summary is not None:
        summary["backfill"] = backfill_summary

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        status = "valid" if summary["valid"] else "invalid"
        print(f"Refined subject-mask contract: {status}")
        print(f"Archive: {summary['zarr_path']}")
        print(f"Run: {summary['run_name'] or summary['requested_run']}")
        print(f"Mask labels: {', '.join(summary['mask_labels']) if summary['mask_labels'] else 'none'}")
        print(
            "Available components: "
            + (", ".join(summary["available_components"]) if summary["available_components"] else "none")
        )
        if backfill_summary is not None:
            backfilled = backfill_summary.get("backfilled", [])
            print(f"Backfilled: {', '.join(backfilled) if backfilled else 'nothing'}")
        for line in _format_issue_lines("ERROR", summary["errors"]):
            print(line)
        for line in _format_issue_lines("WARN", summary["warnings"]):
            print(line)

    return 0 if bool(summary["valid"]) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
