"""Canonical eye geometry materialization for refined subject-mask runs."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional, Sequence

import numpy as np
import zarr

from .mask_geometry import MASK_ELLIPSE_METHOD
from .mask_store import MaskStoreError, open_mask_store

EYE_COMPONENTS = ("eye_left", "eye_right")
EYE_GEOMETRY_SCHEMA_ID = "refined_subject_eye_geometry_v1"
EYE_PAIR_RELATION_SCHEMA_ID = "refined_subject_eye_pair_relation_v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _group_source_path(group: zarr.Group) -> str:
    return str(getattr(group, "name", "") or "")


def _label_index_map(group: zarr.Group) -> dict[str, int]:
    labels_raw = group.attrs.get("mask_labels")
    if not isinstance(labels_raw, (list, tuple)):
        return {}
    return {str(label): idx for idx, label in enumerate(labels_raw)}


def _component_available(group: zarr.Group, component_idx: int) -> bool:
    available_arr = group.get("available_channels")
    if available_arr is None:
        return True
    available = np.asarray(available_arr[:], dtype=bool).reshape(-1)
    return component_idx < int(available.shape[0]) and bool(available[component_idx])


def _eye_geometry_should_update(updated_components: Optional[Sequence[str]]) -> bool:
    if updated_components is None:
        return True
    updated = {str(name) for name in updated_components}
    return bool(updated.intersection(EYE_COMPONENTS))


def _write_array(group: zarr.Group, name: str, data: np.ndarray, *, chunks: tuple[int, ...]) -> None:
    group.create_array(name, data=data, chunks=chunks, overwrite=True)


def _measure_eye_mask(mask: np.ndarray):
    """Import mask ellipse measurement only when geometry is materialized."""
    from .mask_geometry import (
        DEFAULT_MIN_ELLIPSE_FOREGROUND_PIXELS,
        measure_mask_ellipse,
    )

    return measure_mask_ellipse(
        mask,
        min_foreground_pixels=DEFAULT_MIN_ELLIPSE_FOREGROUND_PIXELS,
    )


def _write_component_contours(*args, **kwargs):
    """Import contour writing only when geometry is materialized."""
    from .refined_subject_component_contours import write_component_contours

    return write_component_contours(*args, **kwargs)


def write_refined_subject_eye_geometry(
    refined_group: zarr.Group,
    *,
    updated_components: Optional[Sequence[str]] = None,
    write_component_contours: bool = True,
) -> dict[str, Any]:
    """Write canonical LR eye geometry/relation arrays into a refined subject run.

    Geometry is derived from the run-level physical mask store for ``eye_left``
    and ``eye_right``. This makes the refined subject-mask run the canonical
    geometry surface; legacy refined-eye runs can still be materialized from it
    as compatibility artifacts.
    """

    if not _eye_geometry_should_update(updated_components):
        return {"status": "skipped", "reason": "no_eye_components_updated"}

    label_map = _label_index_map(refined_group)
    missing = [name for name in EYE_COMPONENTS if name not in label_map]
    if missing:
        return {"status": "skipped", "reason": "missing_eye_components", "missing_components": missing}
    try:
        mask_store = open_mask_store(refined_group, source_path=_group_source_path(refined_group), prefer="dense")
    except MaskStoreError:
        return {"status": "skipped", "reason": "missing_masks_roi"}

    total_rois = int(mask_store.n_rows)
    chunk_rois = max(1, min(256, total_rois if total_rois > 0 else 1))
    ellipse_params = np.full((total_rois, 2, 5), np.nan, dtype=np.float32)
    ellipse_success = np.zeros((total_rois, 2), dtype=bool)
    separation_px = np.full((total_rois,), np.nan, dtype=np.float32)
    separation_valid = np.zeros((total_rois,), dtype=bool)
    contours: dict[str, list[np.ndarray | None]] = {name: [] for name in EYE_COMPONENTS}
    centroids = np.full((total_rois, 2, 2), np.nan, dtype=np.float32)

    for row_idx in range(total_rois):
        for eye_idx, component_name in enumerate(EYE_COMPONENTS):
            comp_idx = int(label_map[component_name])
            if not _component_available(refined_group, comp_idx):
                contours[component_name].append(None)
                continue
            mask = mask_store.read_dense(rows=row_idx, channels=comp_idx)[0, 0]
            success, ellipse, centroid, contour, _failure = _measure_eye_mask(mask)
            ellipse_params[row_idx, eye_idx] = np.asarray(ellipse, dtype=np.float32)
            ellipse_success[row_idx, eye_idx] = bool(success)
            centroids[row_idx, eye_idx] = np.asarray(centroid, dtype=np.float32)
            contours[component_name].append(contour)

        if bool(np.all(ellipse_success[row_idx])) and bool(np.all(np.isfinite(centroids[row_idx]))):
            separation_px[row_idx] = np.float32(np.linalg.norm(centroids[row_idx, 0] - centroids[row_idx, 1]))
            separation_valid[row_idx] = True

    components_parent = refined_group.require_group("components")
    for eye_idx, component_name in enumerate(EYE_COMPONENTS):
        component_group = components_parent.require_group(component_name)
        geometry_group = component_group.require_group("geometry")
        geometry_group.attrs["geometry_schema_id"] = EYE_GEOMETRY_SCHEMA_ID
        geometry_group.attrs["geometry_method"] = MASK_ELLIPSE_METHOD
        geometry_group.attrs["source_mask_component"] = component_name
        geometry_group.attrs["updated_at_utc"] = _utc_now()
        _write_array(
            geometry_group,
            "ellipse_params",
            ellipse_params[:, eye_idx, :],
            chunks=(chunk_rois, 5),
        )
        _write_array(
            geometry_group,
            "ellipse_success",
            ellipse_success[:, eye_idx],
            chunks=(chunk_rois,),
        )
        if write_component_contours:
            _write_component_contours(
                component_group,
                contours[component_name],
                chunk_rois=chunk_rois,
                component=component_name,
                source_mask_label_schema_id=str(refined_group.attrs.get("label_schema_id") or ""),
                min_points=1,
            )

    relation_metrics = refined_group.require_group("relations").require_group("eye_pair").require_group("metrics")
    relation_metrics.attrs["relation_schema_id"] = EYE_PAIR_RELATION_SCHEMA_ID
    relation_metrics.attrs["relation_components"] = list(EYE_COMPONENTS)
    relation_metrics.attrs["relation_method"] = "ellipse_centroid_distance"
    relation_metrics.attrs["updated_at_utc"] = _utc_now()
    _write_array(relation_metrics, "separation_px", separation_px, chunks=(chunk_rois,))
    _write_array(relation_metrics, "separation_valid", separation_valid, chunks=(chunk_rois,))

    refined_group.attrs["eye_geometry_schema_id"] = EYE_GEOMETRY_SCHEMA_ID
    refined_group.attrs["eye_geometry_updated_at_utc"] = _utc_now()
    refined_group.attrs["eye_geometry_status"] = "computed"
    if "eye_geometry_deferred_reason" in refined_group.attrs:
        del refined_group.attrs["eye_geometry_deferred_reason"]
    return {
        "status": "updated",
        "roi_count": total_rois,
        "components": list(EYE_COMPONENTS),
        "ellipse_success_count": int(np.count_nonzero(ellipse_success)),
        "pair_success_count": int(np.count_nonzero(separation_valid)),
    }
