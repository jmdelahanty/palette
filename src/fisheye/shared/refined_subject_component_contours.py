"""Component contour helpers for refined subject-mask runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Sequence

import cv2
import numpy as np
import zarr


COMPONENT_CONTOUR_SCHEMA_ID = "component_contours_v1"
COMPONENT_ROW_UPDATE_SCHEMA_ID = "refined_subject_component_row_updates_v1"
DEFAULT_CONTOUR_METHOD = "largest_external_contour"
DEFAULT_CONTOUR_METHOD_VERSION = 1
DEFAULT_BOUNDARY_POLICY = "external_only"
DEFAULT_CONTOUR_COORDINATE_SPACE = "roi_pixels"
ROW_UPDATE_TIMESTAMP_WIDTH = 40
ROW_UPDATE_REASON_WIDTH = 128


@dataclass(frozen=True)
class ComponentContourSummary:
    component: str
    status: str
    roi_count: int
    contour_count: int = 0
    point_count: int = 0
    reason: Optional[str] = None
    existing: bool = False


@dataclass(frozen=True)
class ComponentContourRowUpdateSummary:
    component: str
    row_index: int
    status: str
    contour_len: int = 0
    point_offset: int = -1
    row_revision: int = 0
    reason: Optional[str] = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def extract_largest_external_contour(
    mask: np.ndarray,
    *,
    min_points: int = 2,
) -> np.ndarray | None:
    """Return the largest external contour in ``(x, y)`` coordinates."""

    mask_u8 = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8, copy=False)
    if int(np.count_nonzero(mask_u8)) == 0:
        return None
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32, copy=False)
    if int(contour.shape[0]) < int(min_points):
        return None
    return contour


def component_contours_exist(component_group: zarr.Group, roi_count: int) -> bool:
    return summarize_existing_component_contours(component_group, component="", roi_count=roi_count) is not None


def summarize_existing_component_contours(
    component_group: zarr.Group,
    *,
    component: str,
    roi_count: int,
) -> ComponentContourSummary | None:
    contours = component_group.get("contours")
    if not isinstance(contours, zarr.Group):
        return None
    for name, expected_shape in (("ptr", (roi_count,)), ("len", (roi_count,))):
        array = contours.get(name)
        if array is None:
            return None
        try:
            if tuple(array.shape) != tuple(expected_shape):
                return None
        except Exception:
            return None
    points = contours.get("points_xy")
    if points is None:
        return None
    try:
        if len(tuple(points.shape)) != 2 or int(points.shape[1]) != 2:
            return None
    except Exception:
        return None
    try:
        lengths = np.asarray(contours["len"][:], dtype=np.int64)
        contour_count = int(np.count_nonzero(lengths > 0))
        point_count = int(np.sum(lengths[lengths > 0]))
    except Exception:
        contour_count = 0
        point_count = 0
    return ComponentContourSummary(
        component=str(component),
        status="existing",
        roi_count=int(roi_count),
        contour_count=contour_count,
        point_count=point_count,
        existing=True,
    )


def write_component_contours(
    component_group: zarr.Group,
    contours_by_row: Sequence[np.ndarray | None],
    *,
    chunk_rois: int,
    component: str,
    source_mask_run: str | None = None,
    source_mask_label_schema_id: str | None = None,
    method: str = DEFAULT_CONTOUR_METHOD,
    method_version: int = DEFAULT_CONTOUR_METHOD_VERSION,
    boundary_policy: str = DEFAULT_BOUNDARY_POLICY,
    coordinate_space: str = DEFAULT_CONTOUR_COORDINATE_SPACE,
    min_points: int = 2,
) -> ComponentContourSummary:
    """Write packed variable-length contour arrays into a component group."""

    total_rois = int(len(contours_by_row))
    ptr = np.full((total_rois,), -1, dtype=np.int64)
    length = np.zeros((total_rois,), dtype=np.int32)
    point_chunks: list[np.ndarray] = []
    offset = 0

    for row_idx, contour in enumerate(contours_by_row):
        if contour is None:
            continue
        points = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
        if int(points.shape[0]) < int(min_points):
            continue
        ptr[int(row_idx)] = np.int64(offset)
        length[int(row_idx)] = np.int32(points.shape[0])
        point_chunks.append(points)
        offset += int(points.shape[0])

    points_xy = (
        np.concatenate(point_chunks, axis=0).astype(np.float32, copy=False)
        if point_chunks
        else np.zeros((1, 2), dtype=np.float32)
    )
    contours_group = component_group.require_group("contours")
    contours_group.attrs.update(
        {
            "schema_id": COMPONENT_CONTOUR_SCHEMA_ID,
            "contour_schema_id": COMPONENT_CONTOUR_SCHEMA_ID,
            "coordinate_space": coordinate_space,
            "point_order": "xy",
            "source_component": str(component),
            "source_mask_run": str(source_mask_run or ""),
            "source_mask_label_schema_id": str(source_mask_label_schema_id or ""),
            "method": str(method),
            "method_version": int(method_version),
            "boundary_policy": str(boundary_policy),
            "min_points": int(min_points),
            "generated_at_utc": _utc_now(),
            "points_placeholder_when_empty": bool(not point_chunks),
            "cache_coverage": "full_indexed_rows",
        }
    )
    contours_group.create_array("ptr", data=ptr, chunks=(max(1, int(chunk_rois)),), overwrite=True)
    contours_group.create_array("len", data=length, chunks=(max(1, int(chunk_rois)),), overwrite=True)
    contours_group.create_array(
        "points_xy",
        data=points_xy,
        chunks=(max(1, min(4096, int(points_xy.shape[0]))), 2),
        overwrite=True,
    )
    return ComponentContourSummary(
        component=str(component),
        status="written",
        roi_count=total_rois,
        contour_count=int(np.count_nonzero(length > 0)),
        point_count=int(points_xy.shape[0]) if point_chunks else 0,
    )


def _label_index_map(refined_group: zarr.Group) -> dict[str, int]:
    labels_raw = refined_group.attrs.get("mask_labels")
    if not isinstance(labels_raw, (list, tuple)):
        return {}
    return {str(label): int(idx) for idx, label in enumerate(labels_raw)}


def _component_available(refined_group: zarr.Group, component_idx: int) -> bool:
    available_arr = refined_group.get("available_channels")
    if available_arr is None:
        return True
    available = np.asarray(available_arr[:], dtype=bool).reshape(-1)
    return int(component_idx) < int(available.shape[0]) and bool(available[int(component_idx)])


def _metric_chunk(roi_count: int, *, max_chunk: int = 256) -> int:
    return max(1, min(int(max_chunk), int(roi_count)))


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
    if existing is not None:
        if tuple(existing.shape) != tuple(shape):
            raise ValueError(f"{group.name}/{name} shape mismatch: expected {shape}, got {tuple(existing.shape)}")
        return existing
    return group.create_array(
        name,
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        fill_value=fill_value,
    )


def _write_ascii_row(array: zarr.Array, row_index: int, value: str, *, width: int) -> None:
    encoded = str(value).encode("utf-8", errors="replace")[: int(width)]
    row = np.zeros((int(width),), dtype=np.uint8)
    if encoded:
        row[: len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
    array[int(row_index), :] = row


def ensure_component_row_update_tracking(
    component_group: zarr.Group,
    *,
    roi_count: int,
    reason_width: int = ROW_UPDATE_REASON_WIDTH,
) -> tuple[zarr.Array, zarr.Array, zarr.Array]:
    """Ensure per-row component revision arrays used by row-local cache refresh."""

    chunk = _metric_chunk(roi_count)
    component_group.attrs["row_update_schema_id"] = COMPONENT_ROW_UPDATE_SCHEMA_ID
    component_group.attrs["row_update_schema_version"] = 1
    revision = _ensure_array(
        component_group,
        "row_revision",
        shape=(int(roi_count),),
        dtype=np.int64,
        chunks=(chunk,),
        fill_value=0,
    )
    updated_at = _ensure_array(
        component_group,
        "row_updated_at_utc_bytes",
        shape=(int(roi_count), ROW_UPDATE_TIMESTAMP_WIDTH),
        dtype=np.uint8,
        chunks=(chunk, ROW_UPDATE_TIMESTAMP_WIDTH),
        fill_value=0,
    )
    reason = _ensure_array(
        component_group,
        "row_update_reason_bytes",
        shape=(int(roi_count), int(reason_width)),
        dtype=np.uint8,
        chunks=(chunk, int(reason_width)),
        fill_value=0,
    )
    return revision, updated_at, reason


def _ensure_component_contour_arrays(
    component_group: zarr.Group,
    *,
    component: str,
    roi_count: int,
    chunk_rois: int,
    source_mask_run: str | None,
    source_mask_label_schema_id: str | None,
    method: str,
    method_version: int,
    boundary_policy: str,
    coordinate_space: str,
    min_points: int,
) -> zarr.Group:
    contours_group = component_group.require_group("contours")
    contours_group.attrs.update(
        {
            "schema_id": COMPONENT_CONTOUR_SCHEMA_ID,
            "contour_schema_id": COMPONENT_CONTOUR_SCHEMA_ID,
            "coordinate_space": coordinate_space,
            "point_order": "xy",
            "source_component": str(component),
            "source_mask_run": str(source_mask_run or contours_group.attrs.get("source_mask_run", "")),
            "source_mask_label_schema_id": str(
                source_mask_label_schema_id or contours_group.attrs.get("source_mask_label_schema_id", "")
            ),
            "method": str(method),
            "method_version": int(method_version),
            "boundary_policy": str(boundary_policy),
            "min_points": int(min_points),
        }
    )
    chunk = max(1, int(chunk_rois))
    _ensure_array(
        contours_group,
        "ptr",
        shape=(int(roi_count),),
        dtype=np.int64,
        chunks=(chunk,),
        fill_value=-1,
    )
    _ensure_array(
        contours_group,
        "len",
        shape=(int(roi_count),),
        dtype=np.int32,
        chunks=(chunk,),
        fill_value=0,
    )
    points = contours_group.get("points_xy")
    if points is None:
        contours_group.create_array(
            "points_xy",
            data=np.zeros((1, 2), dtype=np.float32),
            chunks=(1, 2),
            overwrite=True,
        )
        contours_group.attrs["points_placeholder_when_empty"] = True
        contours_group.attrs.setdefault("cache_coverage", "partial_row_updates")
    else:
        if len(tuple(points.shape)) != 2 or int(points.shape[1]) != 2:
            raise ValueError(f"{contours_group.name}/points_xy must have shape (M, 2), got {tuple(points.shape)}")
        contours_group.attrs.setdefault("cache_coverage", "partial_row_updates")
    return contours_group


def _append_points(points_array: zarr.Array, points: np.ndarray, *, placeholder: bool) -> int:
    points_xy = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    old_count = int(points_array.shape[0])
    append_offset = 0 if placeholder and old_count == 1 else old_count
    new_count = int(append_offset + points_xy.shape[0])
    resize = getattr(points_array, "resize", None)
    if not callable(resize):
        raise RuntimeError(f"{points_array.name} does not support resize; cannot append row-local contour points.")
    resize((new_count, 2))
    points_array[append_offset:new_count, :] = points_xy
    return append_offset


def write_component_contour_row(
    component_group: zarr.Group,
    *,
    row_index: int,
    mask: np.ndarray,
    roi_count: int,
    component: str,
    reason: str = "row_local_mask_cache_refresh",
    updated_at_utc: str | None = None,
    chunk_rois: int = 256,
    source_mask_run: str | None = None,
    source_mask_label_schema_id: str | None = None,
    method: str = DEFAULT_CONTOUR_METHOD,
    method_version: int = DEFAULT_CONTOUR_METHOD_VERSION,
    boundary_policy: str = DEFAULT_BOUNDARY_POLICY,
    coordinate_space: str = DEFAULT_CONTOUR_COORDINATE_SPACE,
    min_points: int = 2,
) -> ComponentContourRowUpdateSummary:
    """Append-update one component contour row and increment its row revision."""

    row_idx = int(row_index)
    if row_idx < 0 or row_idx >= int(roi_count):
        raise IndexError(f"row_index {row_idx} outside component contour row count {int(roi_count)}")
    now = str(updated_at_utc or _utc_now())
    contours_group = _ensure_component_contour_arrays(
        component_group,
        component=component,
        roi_count=int(roi_count),
        chunk_rois=int(chunk_rois),
        source_mask_run=source_mask_run,
        source_mask_label_schema_id=source_mask_label_schema_id,
        method=method,
        method_version=method_version,
        boundary_policy=boundary_policy,
        coordinate_space=coordinate_space,
        min_points=min_points,
    )
    revision_arr, updated_at_arr, reason_arr = ensure_component_row_update_tracking(
        component_group,
        roi_count=int(roi_count),
    )

    contour = extract_largest_external_contour(mask, min_points=min_points)
    if contour is None:
        contours_group["ptr"][row_idx] = np.int64(-1)
        contours_group["len"][row_idx] = np.int32(0)
        status = "missing_contour"
        point_offset = -1
        contour_len = 0
    else:
        points = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
        points_array = contours_group["points_xy"]
        lengths = np.asarray(contours_group["len"][:], dtype=np.int64)
        placeholder = bool(contours_group.attrs.get("points_placeholder_when_empty")) and int(
            np.count_nonzero(lengths > 0)
        ) == 0
        point_offset = _append_points(points_array, points, placeholder=placeholder)
        contours_group["ptr"][row_idx] = np.int64(point_offset)
        contours_group["len"][row_idx] = np.int32(points.shape[0])
        contours_group.attrs["points_placeholder_when_empty"] = False
        contour_len = int(points.shape[0])
        status = "written"

    previous_revision = int(np.asarray(revision_arr[row_idx], dtype=np.int64))
    row_revision = previous_revision + 1
    revision_arr[row_idx] = np.int64(row_revision)
    _write_ascii_row(updated_at_arr, row_idx, now, width=ROW_UPDATE_TIMESTAMP_WIDTH)
    _write_ascii_row(reason_arr, row_idx, reason, width=ROW_UPDATE_REASON_WIDTH)
    component_group.attrs["last_row_update_at_utc"] = now
    component_group.attrs["last_row_update_reason"] = str(reason)
    contours_group.attrs["last_row_local_update_at_utc"] = now
    contours_group.attrs["last_row_local_update_reason"] = str(reason)
    contours_group.attrs["last_row_local_update_row"] = row_idx
    contours_group.attrs["row_local_update_count"] = int(contours_group.attrs.get("row_local_update_count", 0)) + 1
    contours_group.attrs["orphaned_points_possible"] = True

    return ComponentContourRowUpdateSummary(
        component=str(component),
        row_index=row_idx,
        status=status,
        contour_len=contour_len,
        point_offset=point_offset,
        row_revision=row_revision,
        reason=str(reason),
    )


def refresh_component_contour_rows_from_masks(
    refined_group: zarr.Group,
    component: str,
    row_indices: Sequence[int],
    *,
    reason: str = "row_local_mask_cache_refresh",
    updated_at_utc: str | None = None,
    chunk_rois: int = 256,
    min_points: int = 2,
) -> list[ComponentContourRowUpdateSummary]:
    """Refresh packed contour rows for a refined subject-mask component."""

    label_map = _label_index_map(refined_group)
    component_name = str(component)
    if component_name not in label_map:
        raise KeyError(f"Component {component_name!r} is not present in refined mask labels.")
    masks_roi = refined_group.get("masks_roi")
    if masks_roi is None or len(tuple(masks_roi.shape)) != 4:
        raise ValueError("refined_group must contain masks_roi with shape (N,C,H,W).")
    roi_count = int(masks_roi.shape[0])
    component_idx = int(label_map[component_name])
    if component_idx < 0 or component_idx >= int(masks_roi.shape[1]):
        raise ValueError(
            f"Component {component_name!r} channel {component_idx} outside masks_roi channel count "
            f"{int(masks_roi.shape[1])}."
        )
    if not _component_available(refined_group, component_idx):
        raise ValueError(f"Component {component_name!r} is marked unavailable in available_channels.")

    component_group = refined_group.require_group("components").require_group(component_name)
    updated_at = str(updated_at_utc or _utc_now())
    source_label_schema = refined_group.attrs.get("label_schema_id")
    source_mask_run = str(refined_group.attrs.get("run_name") or refined_group.name.rstrip("/").split("/")[-1])
    summaries: list[ComponentContourRowUpdateSummary] = []
    for row_index in row_indices:
        row_idx = int(row_index)
        mask = np.asarray(masks_roi[row_idx, component_idx], dtype=np.uint8)
        summaries.append(
            write_component_contour_row(
                component_group,
                row_index=row_idx,
                mask=mask,
                roi_count=roi_count,
                component=component_name,
                reason=reason,
                updated_at_utc=updated_at,
                chunk_rois=chunk_rois,
                source_mask_run=source_mask_run,
                source_mask_label_schema_id=str(source_label_schema or ""),
                min_points=min_points,
            )
        )
    return summaries


def build_component_contours_from_masks(
    refined_group: zarr.Group,
    component: str,
    *,
    min_points: int = 2,
) -> tuple[list[np.ndarray | None], ComponentContourSummary]:
    """Extract largest external contours for one refined mask component."""

    label_map = _label_index_map(refined_group)
    if component not in label_map:
        return [], ComponentContourSummary(component=component, status="missing_label", roi_count=0, reason="label absent")
    masks_roi = refined_group.get("masks_roi")
    if masks_roi is None:
        return [], ComponentContourSummary(
            component=component,
            status="missing_masks_roi",
            roi_count=0,
            reason="masks_roi array missing",
        )
    if len(tuple(masks_roi.shape)) != 4:
        return [], ComponentContourSummary(
            component=component,
            status="shape_mismatch",
            roi_count=0,
            reason=f"masks_roi shape is {tuple(masks_roi.shape)}",
        )
    roi_count = int(masks_roi.shape[0])
    channel_count = int(masks_roi.shape[1])
    component_idx = int(label_map[component])
    if component_idx < 0 or component_idx >= channel_count:
        return [], ComponentContourSummary(
            component=component,
            status="shape_mismatch",
            roi_count=roi_count,
            reason=f"component channel {component_idx} outside masks_roi channel count {channel_count}",
        )
    if not _component_available(refined_group, component_idx):
        return [], ComponentContourSummary(
            component=component,
            status="unavailable",
            roi_count=roi_count,
            reason="available_channels marks component unavailable",
        )

    contours: list[np.ndarray | None] = []
    contour_count = 0
    point_count = 0
    for row_idx in range(roi_count):
        contour = extract_largest_external_contour(masks_roi[row_idx, component_idx], min_points=min_points)
        contours.append(contour)
        if contour is not None:
            contour_count += 1
            point_count += int(contour.shape[0])
    return contours, ComponentContourSummary(
        component=component,
        status="computed",
        roi_count=roi_count,
        contour_count=contour_count,
        point_count=point_count,
    )


def write_refined_subject_component_contours(
    refined_group: zarr.Group,
    *,
    components: Sequence[str],
    source_mask_run: str | None = None,
    chunk_rois: int = 256,
    min_points: int = 2,
    overwrite: bool = False,
) -> list[ComponentContourSummary]:
    """Write component contour caches for selected refined mask components."""

    masks_roi = refined_group.get("masks_roi")
    roi_count = int(masks_roi.shape[0]) if masks_roi is not None and len(tuple(masks_roi.shape)) >= 1 else 0
    source_label_schema = refined_group.attrs.get("label_schema_id")
    summaries: list[ComponentContourSummary] = []
    for component in components:
        component_name = str(component)
        components_group_existing = refined_group.get("components")
        component_group = (
            components_group_existing.get(component_name) if isinstance(components_group_existing, zarr.Group) else None
        )
        existing_summary = (
            summarize_existing_component_contours(component_group, component=component_name, roi_count=roi_count)
            if isinstance(component_group, zarr.Group)
            else None
        )
        if existing_summary is not None and not overwrite:
            summaries.append(existing_summary)
            continue
        contours, summary = build_component_contours_from_masks(
            refined_group,
            component_name,
            min_points=min_points,
        )
        if summary.status != "computed":
            summaries.append(summary)
            continue
        components_group = refined_group.require_group("components")
        component_group = components_group.require_group(component_name)
        written = write_component_contours(
            component_group,
            contours,
            chunk_rois=chunk_rois,
            component=component_name,
            source_mask_run=source_mask_run,
            source_mask_label_schema_id=str(source_label_schema or ""),
            min_points=min_points,
        )
        summaries.append(written)
    return summaries
