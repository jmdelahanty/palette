from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import zarr
from zarr.core.dtype import VariableLengthUTF8

from .crop_geometry import bbox_img_xyxy_to_norm_cxcywh, bbox_norm_cxcywh_to_img_xyxy
from .detect_reason_codec import read_reason_labels, update_reason_rows, write_reason_columns
from .json_safety import json_attr_safe
from .stage_provenance import build_stage_provenance
from .type_conversions import as_int, clean_mapping, normalize_attr


REFINED_DETECT_STATUS_CODE_MAP: Dict[str, int] = {
    "present": 0,
    "missing": 1,
    "filtered_out": 2,
    "ambiguous": 3,
}
REFINED_SOURCE_KIND_CODE_MAP: Dict[str, int] = {
    "none": 0,
    "raw_detect": 1,
    "interpolated": 2,
    "manual": 3,
}
REFINED_SOURCE_DETECTION_DECISION_CODE_MAP: Dict[str, int] = {
    "accepted": 0,
    "filtered": 1,
    "duplicate": 2,
    "manual_clear": 3,
}
REFINED_REVIEW_STATE_CODE_MAP: Dict[str, int] = {
    "unknown": 0,
    "approved": 1,
    "pending": 2,
    "needs_review": 3,
    "rejected": 4,
}
REFINED_ARTIFACT_STATE_CODE_MAP: Dict[str, int] = {
    "not_generated": 0,
    "pending": 1,
    "missing": 2,
    "present": 3,
    "not_applicable": 4,
}
FIXED_WIDTH_BBOX_ARRAY_NAMES = frozenset({"bbox_img_xyxy", "bbox_norm_coords"})
DEFAULT_REFINED_DETECT_ROW_CHUNK = 65_536

CURATED_REFINED_REQUIRED_ARRAYS: Tuple[str, ...] = (
    "refined_row_ids",
    "frame_indices",
    "entity_ids",
    "bbox_img_xyxy",
    "bbox_norm_coords",
    "status_codes",
    "source_kind_codes",
    "review_state_codes",
    "keypoints_state_codes",
    "subject_mask_state_codes",
    "eye_mask_state_codes",
    "swim_bladder_state_codes",
)
CURATED_REFINED_INSTANCES_REQUIRED_ARRAYS: Tuple[str, ...] = (
    "refined_row_ids",
    "frame_indices",
    "frame_offsets",
    "bbox_img_xyxy",
    "bbox_norm_coords",
    "source_kind_codes",
    "manual_edit_flags",
    "source_detect_row_index",
    "frame_counts",
)
CURATED_REFINED_SOURCE_DETECTIONS_REQUIRED_ARRAYS: Tuple[str, ...] = (
    "source_detect_row_index",
    "frame_indices",
    "bbox_img_xyxy",
    "bbox_norm_coords",
    "decision_codes",
    "resolved_refined_row_id",
)
LEGACY_DENSE_CURATED_ROOT_ARRAYS: Tuple[str, ...] = CURATED_REFINED_REQUIRED_ARRAYS + (
    "manual_edit_flags",
    "source_detect_row_index",
    "source_sparse_row_index",
    "source_sparse_group_codes",
    "detection_source",
    "confidence_scores",
    "class_ids",
    "reason_bytes",
    "reason",
    "review_notes",
)
LEGACY_DENSE_CURATED_ROOT_ATTRS: Tuple[str, ...] = (
    "active_sparse_group",
    "active_sparse_group_kind",
    "active_sparse_group_path",
    "dense_projection_storage",
    "dense_axis_names",
    "artifact_state_code_map",
    "source_sparse_group_code_map",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


_json_safe_attr_value = json_attr_safe


def _delete_if_present(group: zarr.Group, name: str) -> None:
    if name in group:
        try:
            del group[name]
        except Exception:
            children = getattr(group, "_children", None)
            if isinstance(children, dict):
                children.pop(name, None)
            else:
                raise


def _is_group_like(node: Any) -> bool:
    return hasattr(node, "attrs") and (
        hasattr(node, "create_array") or hasattr(node, "require_group")
    )


def _open_named_child_group(parent: zarr.Group, name: str, *, mode: str) -> zarr.Group:
    if not (hasattr(parent, "store_path") and hasattr(parent, "path")):
        existing = parent.get(name)
        if existing is not None and _is_group_like(existing):
            return existing
        if mode == "a":
            return _get_or_create_child_group(parent, name)
        raise KeyError(name)
    child_path = f"{parent.path}/{name}" if getattr(parent, "path", "") else name
    try:
        return zarr.open_group(
            store=parent.store_path.store,
            path=child_path,
            mode=mode,
            use_consolidated=False,
        )
    except TypeError:
        return zarr.open_group(
            store=parent.store_path.store,
            path=child_path,
            mode=mode,
        )


def _get_child_group_if_present(group: zarr.Group, name: str) -> Optional[zarr.Group]:
    if hasattr(group, "store_path") and hasattr(group, "path"):
        try:
            return _open_named_child_group(group, name, mode="r")
        except Exception:
            return None
    existing = group.get(name)
    if existing is None:
        try:
            existing = group[name]
        except Exception:
            existing = None
    if existing is not None and _is_group_like(existing):
        return existing
    return None


def _get_or_create_child_group(group: zarr.Group, name: str) -> zarr.Group:
    if hasattr(group, "store_path") and hasattr(group, "path"):
        return _open_named_child_group(group, name, mode="a")

    existing = group.get(name)
    if existing is not None:
        if _is_group_like(existing):
            return existing
        _delete_if_present(group, name)
    try:
        return group.create_group(name)
    except Exception as exc:
        message = str(exc)
        if "A group exists in store" not in message and "already exists" not in message:
            raise
        existing = group.get(name)
        if existing is None:
            existing = group[name]
        if _is_group_like(existing):
            return existing
        raise


def _common_array_chunks(name: str, data: np.ndarray) -> tuple[int, ...] | None:
    arr = np.asarray(data)
    if name in FIXED_WIDTH_BBOX_ARRAY_NAMES and arr.ndim == 2 and arr.shape[1] == 4:
        return (max(1, min(int(arr.shape[0]), DEFAULT_REFINED_DETECT_ROW_CHUNK)), 4)
    return None


def _write_common_array(group: zarr.Group, name: str, data: np.ndarray) -> None:
    _delete_if_present(group, name)
    chunks = _common_array_chunks(name, data)
    if chunks is None:
        group.create_array(name, data=data, overwrite=True)
    else:
        group.create_array(name, data=data, chunks=chunks, overwrite=True)


def _write_string_array(
    group: zarr.Group,
    name: str,
    labels: np.ndarray,
    *,
    overwrite: bool = True,
) -> None:
    if overwrite:
        _delete_if_present(group, name)
    arr = group.create_array(
        name,
        shape=(int(labels.shape[0]),),
        chunks=(max(1, int(labels.shape[0])),),
        dtype=VariableLengthUTF8(),
        fill_value="",
        overwrite=overwrite,
    )
    arr[:] = np.asarray(labels, dtype=object)


def _delete_legacy_dense_curated_root(refined_run: zarr.Group) -> None:
    for name in LEGACY_DENSE_CURATED_ROOT_ARRAYS:
        _delete_if_present(refined_run, name)
    for name in LEGACY_DENSE_CURATED_ROOT_ATTRS:
        if name in refined_run.attrs:
            del refined_run.attrs[name]


def _normalize_group_path(path: str) -> str:
    value = str(path or "").strip().strip("/")
    if not value:
        raise ValueError("group path must be non-empty")
    if ".." in Path(value).parts:
        raise ValueError(f"group path must not contain '..': {path!r}")
    return value


def _resolve_group_path(root: zarr.Group, path: str, *, mode: str) -> zarr.Group:
    normalized = _normalize_group_path(path)
    if mode == "a":
        return root.require_group(normalized)
    return root[normalized]


def _resolve_refined_parent(
    root: zarr.Group,
    *,
    refined_family_path: Optional[str] = None,
) -> Tuple[zarr.Group, str]:
    if refined_family_path is not None:
        normalized = _normalize_group_path(refined_family_path)
        return _resolve_group_path(root, normalized, mode="a"), normalized
    refined_parent = root.get("refined_detect_runs")
    if refined_parent is not None:
        return refined_parent, "refined_detect_runs"
    refined_parent = root.get("refined_runs")
    if refined_parent is not None:
        return refined_parent, "refined_runs"
    raise ValueError("No refined_detect_runs group found.")


def has_curated_refined_detect_arrays(refined_run: zarr.Group) -> bool:
    return all(name in refined_run for name in CURATED_REFINED_REQUIRED_ARRAYS)


def has_sparse_curated_refined_detect_instances_arrays(refined_run: zarr.Group) -> bool:
    instances = _get_child_group_if_present(refined_run, "instances")
    if instances is None:
        return False
    return all(name in instances for name in CURATED_REFINED_INSTANCES_REQUIRED_ARRAYS)


def has_curated_refined_detect_surface(refined_run: zarr.Group) -> bool:
    return has_curated_refined_detect_arrays(refined_run) or has_sparse_curated_refined_detect_instances_arrays(
        refined_run
    )


def has_curated_refined_source_detections_projection(refined_run: zarr.Group) -> bool:
    source_detections = _get_child_group_if_present(refined_run, "source_detections")
    if source_detections is None:
        return False
    return all(name in source_detections for name in CURATED_REFINED_SOURCE_DETECTIONS_REQUIRED_ARRAYS)


def resolve_curated_refined_detect_run(
    root: zarr.Group,
    *,
    run_name: Optional[str] = None,
) -> Tuple[zarr.Group, str]:
    refined_parent, _ = _resolve_refined_parent(root)
    resolved_name = normalize_attr(run_name) or normalize_attr(refined_parent.attrs.get("latest"))
    if not resolved_name or resolved_name not in refined_parent:
        raise ValueError("No refined detect run available.")
    refined_run = _open_named_child_group(refined_parent, resolved_name, mode="r")
    if not has_curated_refined_detect_surface(refined_run):
        raise ValueError(f"Refined detect run '{resolved_name}' does not have a curated detect surface.")
    return refined_run, resolved_name


def _resolve_source_group(
    root: zarr.Group,
    refined_run: zarr.Group,
    refined_run_name: str,
    *,
    source_group: Optional[str],
    parent_name: str,
) -> Tuple[str, Optional[str], zarr.Group, str]:
    requested = normalize_attr(source_group)
    manual_label = normalize_attr(refined_run.attrs.get("manual_review_latest"))

    if requested == "raw":
        source_detect_run = normalize_attr(refined_run.attrs.get("source_detect_run"))
        detect_parent = root.get("detect_runs")
        if not source_detect_run or detect_parent is None or source_detect_run not in detect_parent:
            raise ValueError("Refined detect run does not resolve to a valid source_detect_run.")
        return "raw", None, detect_parent[source_detect_run], f"detect_runs/{source_detect_run}"

    if requested == "manual":
        if manual_label and manual_label in refined_run:
            requested = manual_label
        elif "manual" in refined_run:
            requested = "manual"
        else:
            raise ValueError("No manual subgroup available.")
    if requested is None:
        if manual_label and manual_label in refined_run:
            requested = manual_label
        elif "manual" in refined_run:
            requested = "manual"
        elif "interpolated" in refined_run:
            requested = "interpolated"
        elif "filtered" in refined_run:
            requested = "filtered"
    if requested is None or requested not in refined_run:
        raise ValueError("No compatible sparse refined subgroup available.")

    group_kind = "manual" if str(requested).startswith("manual") else str(requested)
    return group_kind, requested, refined_run[requested], f"{parent_name}/{refined_run_name}/{requested}"


def assign_refined_row_ids(
    existing_run: Optional[zarr.Group],
    *,
    frame_indices: np.ndarray,
    entity_ids: np.ndarray,
) -> np.ndarray:
    frame_indices = np.asarray(frame_indices, dtype=np.int32).reshape(-1)
    entity_ids = np.asarray(entity_ids, dtype=np.int32).reshape(-1)
    if frame_indices.shape != entity_ids.shape:
        raise ValueError("frame_indices and entity_ids must have the same shape.")

    seen: set[tuple[int, int]] = set()
    for frame_index, entity_id in zip(frame_indices.tolist(), entity_ids.tolist()):
        key = (int(frame_index), int(entity_id))
        if key in seen:
            raise ValueError(f"Duplicate curated row identity detected for frame/entity {key}.")
        seen.add(key)

    if existing_run is None or not has_curated_refined_detect_arrays(existing_run):
        return np.arange(frame_indices.shape[0], dtype=np.int64)

    previous_row_ids = np.asarray(existing_run["refined_row_ids"][:], dtype=np.int64).reshape(-1)
    previous_frame_indices = np.asarray(existing_run["frame_indices"][:], dtype=np.int32).reshape(-1)
    previous_entity_ids = np.asarray(existing_run["entity_ids"][:], dtype=np.int32).reshape(-1)
    mapping: dict[tuple[int, int], int] = {}
    for row_id, frame_index, entity_id in zip(
        previous_row_ids.tolist(),
        previous_frame_indices.tolist(),
        previous_entity_ids.tolist(),
    ):
        mapping[(int(frame_index), int(entity_id))] = int(row_id)

    next_id = max(mapping.values(), default=-1) + 1
    out = np.empty(frame_indices.shape[0], dtype=np.int64)
    for idx, (frame_index, entity_id) in enumerate(zip(frame_indices.tolist(), entity_ids.tolist())):
        key = (int(frame_index), int(entity_id))
        if key in mapping:
            out[idx] = mapping[key]
            continue
        out[idx] = next_id
        mapping[key] = next_id
        next_id += 1
    return out


def _status_labels_from_codes(status_codes: np.ndarray) -> np.ndarray:
    reverse = {value: key for key, value in REFINED_DETECT_STATUS_CODE_MAP.items()}
    return np.asarray([reverse.get(int(code), "unknown") for code in status_codes.tolist()], dtype=object)


def _source_labels_from_codes(source_kind_codes: np.ndarray) -> np.ndarray:
    reverse = {value: key for key, value in REFINED_SOURCE_KIND_CODE_MAP.items()}
    return np.asarray([reverse.get(int(code), "unknown") for code in source_kind_codes.tolist()], dtype=object)


def _local_entity_ids_from_frame_indices(frame_indices: np.ndarray) -> np.ndarray:
    frame_indices_arr = np.asarray(frame_indices, dtype=np.int32).reshape(-1)
    entity_ids = np.zeros(frame_indices_arr.shape[0], dtype=np.int32)
    current_frame: Optional[int] = None
    current_entity = -1
    for idx, frame_index in enumerate(frame_indices_arr.tolist()):
        if current_frame != int(frame_index):
            current_frame = int(frame_index)
            current_entity = 0
        else:
            current_entity += 1
        entity_ids[idx] = int(current_entity)
    return entity_ids


def _build_detection_source_from_source_kind_codes(source_kind_codes: np.ndarray) -> np.ndarray:
    source_kind_codes_arr = np.asarray(source_kind_codes, dtype=np.int8).reshape(-1)
    return np.where(
        source_kind_codes_arr == REFINED_SOURCE_KIND_CODE_MAP["interpolated"],
        1,
        0,
    ).astype(np.int8, copy=False)


def _decision_labels_from_codes(decision_codes: np.ndarray) -> np.ndarray:
    reverse = {value: key for key, value in REFINED_SOURCE_DETECTION_DECISION_CODE_MAP.items()}
    return np.asarray([reverse.get(int(code), "unknown") for code in decision_codes.tolist()], dtype=object)


def _review_state_code_for_run(refined_run: zarr.Group) -> int:
    review_status = refined_run.attrs.get("detect_review_status")
    review_state = "unknown"
    if isinstance(review_status, Mapping):
        review_state = normalize_attr(review_status.get("state")) or "unknown"
    return REFINED_REVIEW_STATE_CODE_MAP.get(review_state, REFINED_REVIEW_STATE_CODE_MAP["unknown"])


def build_refined_detect_summary(
    *,
    frame_indices: np.ndarray,
    entity_ids: np.ndarray,
    status_codes: np.ndarray,
    source_kind_codes: np.ndarray,
    manual_edit_flags: Optional[np.ndarray] = None,
) -> Dict[str, int]:
    status_labels = _status_labels_from_codes(np.asarray(status_codes, dtype=np.int8).reshape(-1))
    source_labels = _source_labels_from_codes(np.asarray(source_kind_codes, dtype=np.int8).reshape(-1))
    summary = {
        "total_rows": int(frame_indices.shape[0]),
        "entity_count": int(len(set(np.asarray(entity_ids, dtype=np.int32).tolist()))),
        "frame_count_covered": int(len(set(np.asarray(frame_indices, dtype=np.int32).tolist()))),
    }
    for label in ("present", "missing", "filtered_out", "ambiguous"):
        summary[f"rows_{label}"] = int(np.sum(status_labels == label))
    for label in ("none", "raw_detect", "interpolated", "manual"):
        summary[f"rows_{label}"] = int(np.sum(source_labels == label))
    if manual_edit_flags is not None:
        manual_arr = np.asarray(manual_edit_flags, dtype=bool).reshape(-1)
        if manual_arr.shape[0] != frame_indices.shape[0]:
            raise ValueError("manual_edit_flags length does not match frame_indices length.")
        summary["rows_manual_edited"] = int(np.sum(manual_arr))
    return summary


def _build_sparse_refined_detect_summary(
    refined_run: zarr.Group,
    *,
    total_frames: int,
) -> Dict[str, int]:
    instances = _get_child_group_if_present(refined_run, "instances")
    if instances is None:
        raise ValueError("Curated refined detect run is missing instances subgroup.")
    frame_indices = np.asarray(instances["frame_indices"][:], dtype=np.int32).reshape(-1)
    source_kind_codes = np.asarray(instances["source_kind_codes"][:], dtype=np.int8).reshape(-1)
    manual_edit_flags = np.asarray(instances["manual_edit_flags"][:], dtype=bool).reshape(-1)
    frame_counts = (
        np.asarray(instances["frame_counts"][:], dtype=np.int32).reshape(-1)
        if "frame_counts" in instances
        else np.bincount(frame_indices, minlength=total_frames).astype(np.int32, copy=False)
    )
    source_labels = _source_labels_from_codes(source_kind_codes)

    summary = {
        "total_rows": int(frame_indices.shape[0]),
        "sparse_instance_rows": int(frame_indices.shape[0]),
        "frame_count_covered": int(np.count_nonzero(frame_counts)),
        "rows_present": int(frame_indices.shape[0]),
        "rows_missing": int(max(total_frames - int(np.count_nonzero(frame_counts)), 0)),
        "rows_filtered_out": 0,
        "rows_ambiguous": int(np.count_nonzero(frame_counts > 1)),
        "frames_multi_instance": int(np.count_nonzero(frame_counts > 1)),
        "max_instances_per_frame": int(np.max(frame_counts)) if frame_counts.size else 0,
        "rows_manual_edited": int(np.sum(manual_edit_flags)),
    }
    for label in ("none", "raw_detect", "interpolated", "manual"):
        summary[f"rows_{label}"] = int(np.sum(source_labels == label))

    if has_curated_refined_source_detections_projection(refined_run):
        source_summary = build_source_detection_decision_summary(refined_run)
        summary["source_detection_candidates"] = int(source_summary.get("total_candidates", 0) or 0)
        for label in ("accepted", "filtered", "duplicate", "manual_clear"):
            summary[f"source_detection_{label}"] = int(source_summary.get(f"decision_{label}", 0) or 0)
        summary["rows_filtered_out"] = (
            summary["source_detection_filtered"]
            + summary["source_detection_duplicate"]
            + summary["source_detection_manual_clear"]
        )
    return summary


def build_curated_detection_source_array(
    refined_run: zarr.Group,
    *,
    present_only: bool = False,
) -> np.ndarray:
    if has_sparse_curated_refined_detect_instances_arrays(refined_run) and (
        present_only or not has_curated_refined_detect_arrays(refined_run)
    ):
        instances = _get_child_group_if_present(refined_run, "instances")
        if instances is None:
            raise ValueError("Curated refined detect run is missing instances subgroup.")
        detection_source = _build_detection_source_from_source_kind_codes(instances["source_kind_codes"][:])
        return detection_source
    if "detection_source" in refined_run:
        detection_source = np.asarray(refined_run["detection_source"][:], dtype=np.int8).reshape(-1)
    elif "source_sparse_group_codes" in refined_run:
        source_sparse_group_codes = np.asarray(
            refined_run["source_sparse_group_codes"][:],
            dtype=np.int8,
        ).reshape(-1)
        code_map = refined_run.attrs.get("source_sparse_group_code_map")
        interpolated_code = None
        if isinstance(code_map, Mapping):
            interpolated_raw = code_map.get("interpolated")
            if interpolated_raw is not None:
                interpolated_code = int(interpolated_raw)
        if interpolated_code is None:
            interpolated_code = 2
        detection_source = np.where(source_sparse_group_codes == interpolated_code, 1, 0).astype(
            np.int8,
            copy=False,
        )
    else:
        if "source_kind_codes" not in refined_run:
            raise ValueError("Curated refined detect root arrays are incomplete.")
        detection_source = _build_detection_source_from_source_kind_codes(refined_run["source_kind_codes"][:])
    if not present_only:
        return detection_source
    return detection_source[present_curated_row_mask(refined_run)]


def present_curated_row_mask(refined_run: zarr.Group) -> np.ndarray:
    if not has_curated_refined_detect_arrays(refined_run):
        if has_sparse_curated_refined_detect_instances_arrays(refined_run):
            instances = _get_child_group_if_present(refined_run, "instances")
            if instances is None:
                raise ValueError("Curated refined detect run is missing instances subgroup.")
            return np.ones(int(instances["frame_indices"].shape[0]), dtype=bool)
        raise ValueError("Curated refined detect surface is incomplete.")
    if "status_codes" not in refined_run or "bbox_norm_coords" not in refined_run:
        raise ValueError("Curated refined detect root arrays are incomplete.")
    status_codes = np.asarray(refined_run["status_codes"][:], dtype=np.int8).reshape(-1)
    bbox_norm = np.asarray(refined_run["bbox_norm_coords"][:], dtype=np.float64).reshape(-1, 4)
    return (
        status_codes == REFINED_DETECT_STATUS_CODE_MAP["present"]
    ) & np.all(np.isfinite(bbox_norm), axis=1)


def _extract_present_curated_rows_from_dense_root(refined_run: zarr.Group) -> Dict[str, np.ndarray]:
    mask = present_curated_row_mask(refined_run)
    payload: Dict[str, np.ndarray] = {
        "present_row_indices": np.flatnonzero(mask).astype(np.int32, copy=False),
        "frame_indices": np.asarray(refined_run["frame_indices"][:], dtype=np.int32).reshape(-1)[mask],
        "bbox_norm_coords": np.asarray(refined_run["bbox_norm_coords"][:], dtype=np.float64).reshape(-1, 4)[mask],
        "entity_ids": np.asarray(refined_run["entity_ids"][:], dtype=np.int32).reshape(-1)[mask],
        "refined_row_ids": np.asarray(refined_run["refined_row_ids"][:], dtype=np.int64).reshape(-1)[mask],
        "detection_source": build_curated_detection_source_array(refined_run, present_only=True),
    }
    optional_arrays = {
        "bbox_img_xyxy": ("float64", (-1, 4)),
        "confidence_scores": ("float32", (-1,)),
        "class_ids": ("int32", (-1,)),
        "manual_edit_flags": ("bool", (-1,)),
        "source_detect_row_index": ("int32", (-1,)),
        "reason": ("object", (-1,)),
    }
    for name in optional_arrays:
        arr = refined_run.get(name)
        if arr is not None:
            payload[name] = np.asarray(arr[:])[mask]
    reason_labels = read_reason_labels(refined_run)
    if reason_labels is not None:
        payload["reason"] = np.asarray(reason_labels, dtype=object)[mask]
    return payload


def _extract_present_curated_rows_from_instances(refined_run: zarr.Group) -> Dict[str, np.ndarray]:
    instances = _get_child_group_if_present(refined_run, "instances")
    if instances is None:
        raise ValueError("Curated refined detect run is missing instances subgroup.")
    row_count = int(instances["frame_indices"].shape[0])
    frame_indices = np.asarray(instances["frame_indices"][:], dtype=np.int32).reshape(-1)
    payload: Dict[str, np.ndarray] = {
        "present_row_indices": np.arange(row_count, dtype=np.int32),
        "frame_indices": frame_indices,
        "bbox_norm_coords": np.asarray(instances["bbox_norm_coords"][:], dtype=np.float64).reshape(-1, 4),
        "entity_ids": _local_entity_ids_from_frame_indices(frame_indices),
        "refined_row_ids": np.asarray(instances["refined_row_ids"][:], dtype=np.int64).reshape(-1),
        "detection_source": _build_detection_source_from_source_kind_codes(instances["source_kind_codes"][:]),
    }
    optional_arrays = (
        "bbox_img_xyxy",
        "confidence_scores",
        "class_ids",
        "manual_edit_flags",
        "source_detect_row_index",
        "review_notes",
    )
    for name in optional_arrays:
        arr = instances.get(name)
        if arr is not None:
            payload[name] = np.asarray(arr[:])
    reason_labels = read_reason_labels(instances)
    if reason_labels is not None:
        payload["reason"] = np.asarray(reason_labels, dtype=object)
    return payload


def extract_present_curated_rows(refined_run: zarr.Group) -> Dict[str, np.ndarray]:
    if has_sparse_curated_refined_detect_instances_arrays(refined_run):
        return _extract_present_curated_rows_from_instances(refined_run)
    if has_curated_refined_detect_arrays(refined_run):
        return _extract_present_curated_rows_from_dense_root(refined_run)
    raise ValueError("Curated refined detect surface is incomplete.")


def extract_source_detection_rows(
    refined_run: zarr.Group,
    *,
    decision_labels: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    if not has_curated_refined_source_detections_projection(refined_run):
        raise ValueError("Curated refined detect run is missing source_detections projection.")

    subgroup = _get_child_group_if_present(refined_run, "source_detections")
    if subgroup is None:
        raise ValueError("Curated refined detect run is missing source_detections projection.")
    decisions = np.asarray(subgroup["decision_codes"][:], dtype=np.int8).reshape(-1)
    decision_names = _decision_labels_from_codes(decisions)

    mask = np.ones(decisions.shape[0], dtype=bool)
    if decision_labels is not None:
        requested = {str(label).strip().lower() for label in decision_labels if str(label).strip()}
        if not requested:
            mask = np.zeros(decisions.shape[0], dtype=bool)
        else:
            mask = np.isin(decision_names.astype(str), list(sorted(requested)))

    payload: Dict[str, np.ndarray] = {
        "source_detect_row_index": np.asarray(subgroup["source_detect_row_index"][:], dtype=np.int32).reshape(-1)[mask],
        "frame_indices": np.asarray(subgroup["frame_indices"][:], dtype=np.int32).reshape(-1)[mask],
        "bbox_img_xyxy": np.asarray(subgroup["bbox_img_xyxy"][:], dtype=np.float64).reshape(-1, 4)[mask],
        "bbox_norm_coords": np.asarray(subgroup["bbox_norm_coords"][:], dtype=np.float64).reshape(-1, 4)[mask],
        "decision_codes": decisions[mask],
        "decision_labels": np.asarray(decision_names, dtype=object)[mask],
        "resolved_refined_row_id": np.asarray(subgroup["resolved_refined_row_id"][:], dtype=np.int64).reshape(-1)[mask],
    }
    for name, dtype in (
        ("confidence_scores", np.float32),
        ("class_ids", np.int32),
        ("review_notes", object),
    ):
        arr = subgroup.get(name)
        if arr is not None:
            payload[name] = np.asarray(arr[:], dtype=dtype).reshape(-1)[mask]
    reason_labels = read_reason_labels(subgroup)
    if reason_labels is not None:
        payload["reason"] = np.asarray(reason_labels, dtype=object)[mask]
    return payload


def build_source_detection_decision_summary(refined_run: zarr.Group) -> Dict[str, int]:
    if not has_curated_refined_source_detections_projection(refined_run):
        return {}
    payload = extract_source_detection_rows(refined_run)
    decision_labels = np.asarray(payload["decision_labels"], dtype=object).reshape(-1)
    summary = {"total_candidates": int(decision_labels.shape[0])}
    for label in ("accepted", "filtered", "duplicate", "manual_clear"):
        summary[f"decision_{label}"] = int(np.sum(decision_labels == label))
    return summary


def _resolve_bound_source_detect_group(
    root: zarr.Group,
    refined_run: zarr.Group,
) -> tuple[Optional[zarr.Group], Optional[str]]:
    source_detect_run = normalize_attr(refined_run.attrs.get("source_detect_run"))
    source_detect_path = normalize_attr(refined_run.attrs.get("source_detect_path"))
    if source_detect_path:
        try:
            return root[source_detect_path], source_detect_run
        except Exception:
            pass
    detect_parent = root.get("detect_runs")
    if detect_parent is None or not source_detect_run or source_detect_run not in detect_parent:
        return None, source_detect_run
    return detect_parent[source_detect_run], source_detect_run


def _shape_hw(shape: Any) -> Tuple[Optional[int], Optional[int]]:
    try:
        dims = tuple(int(dim) for dim in shape)
    except Exception:
        return None, None
    if len(dims) == 3:
        return dims[1], dims[2]
    if len(dims) == 4:
        return dims[1], dims[2]
    return None, None


def _positive_width_height(width: Any, height: Any) -> Tuple[Optional[int], Optional[int]]:
    w = as_int(width)
    h = as_int(height)
    if w is not None and h is not None and w > 0 and h > 0:
        return w, h
    return None, None


def _resolve_image_dimensions(
    root: zarr.Group,
    *,
    detect_group: Optional[zarr.Group] = None,
    refined_run: Optional[zarr.Group] = None,
) -> Tuple[Optional[int], Optional[int]]:
    """Resolve image-space dimensions for detection bbox conversion.

    Analysis Zarrs normally carry root ``width``/``height`` attrs. Sampled
    training Zarrs may only carry dimensions under ``raw_video`` and seeded
    detections may be normalized to ``raw_video/images_ds``. Prefer the bound
    detect run's frame-source shape when present so refined training labels stay
    in the same image space as the reviewed frames.
    """

    bound_detect_group = detect_group
    if bound_detect_group is None and refined_run is not None:
        bound_detect_group, _source_detect_run = _resolve_bound_source_detect_group(root, refined_run)

    if bound_detect_group is not None:
        source_shape = bound_detect_group.attrs.get("frame_source_shape")
        height, width = _shape_hw(source_shape)
        if width is not None and height is not None and width > 0 and height > 0:
            return width, height
        width, height = _positive_width_height(
            bound_detect_group.attrs.get("inference_width"),
            bound_detect_group.attrs.get("inference_height"),
        )
        if width is not None and height is not None:
            return width, height

    width, height = _positive_width_height(root.attrs.get("inference_width"), root.attrs.get("inference_height"))
    if width is not None and height is not None:
        return width, height

    width, height = _positive_width_height(root.attrs.get("width"), root.attrs.get("height"))
    if width is not None and height is not None:
        return width, height

    raw = root.get("raw_video")
    if raw is not None:
        width, height = _positive_width_height(raw.attrs.get("video_width"), raw.attrs.get("video_height"))
        if width is not None and height is not None:
            return width, height
        resolution = raw.attrs.get("original_resolution")
        if isinstance(resolution, (list, tuple)) and len(resolution) >= 2:
            height = as_int(resolution[0])
            width = as_int(resolution[1])
            if width is not None and height is not None and width > 0 and height > 0:
                return width, height
        for name in ("images_full", "images_ds_rgb", "images_ds"):
            if name in raw:
                height, width = _shape_hw(raw[name].shape)
                if width is not None and height is not None and width > 0 and height > 0:
                    return width, height

    return None, None


def _resolved_total_frames(root: zarr.Group, refined_run: zarr.Group) -> int:
    total_frames = as_int(root.attrs.get("total_frames"))
    if total_frames is None:
        total_frames = as_int(root.attrs.get("n_frames"))
    if total_frames is not None and total_frames >= 0:
        return int(total_frames)
    total_frames = as_int(refined_run.attrs.get("coverage_frames_total"))
    if total_frames is not None and total_frames >= 0:
        return int(total_frames)
    detect_group, _source_detect_run = _resolve_bound_source_detect_group(root, refined_run)
    if detect_group is not None:
        for name in ("frame_counts", "n_detections"):
            if name in detect_group:
                return int(detect_group[name].shape[0])
    raw = root.get("raw_video")
    if raw is not None:
        total_frames = as_int(raw.attrs.get("total_frames"))
        if total_frames is None:
            total_frames = as_int(raw.attrs.get("n_frames"))
        if total_frames is not None and total_frames >= 0:
            return int(total_frames)
    if has_sparse_curated_refined_detect_instances_arrays(refined_run):
        instances = _get_child_group_if_present(refined_run, "instances")
        if instances is not None and "frame_counts" in instances:
            return int(instances["frame_counts"].shape[0])
        frame_indices_arr = (
            np.asarray(instances["frame_indices"][:], dtype=np.int32).reshape(-1)
            if instances is not None
            else np.empty((0,), dtype=np.int32)
        )
        if has_curated_refined_source_detections_projection(refined_run):
            source_detections = _get_child_group_if_present(refined_run, "source_detections")
            if source_detections is not None:
                source_frames = np.asarray(source_detections["frame_indices"][:], dtype=np.int32).reshape(-1)
                if source_frames.size:
                    frame_indices_arr = np.concatenate([frame_indices_arr, source_frames])
        if frame_indices_arr.size == 0:
            return 0
        return int(np.max(frame_indices_arr)) + 1
    frame_indices_arr = np.asarray(refined_run["frame_indices"][:], dtype=np.int32).reshape(-1)
    if frame_indices_arr.size == 0:
        return 0
    return int(np.max(frame_indices_arr)) + 1


def _write_sparse_instances_projection(
    root: zarr.Group,
    refined_run: zarr.Group,
) -> None:
    subgroup = _get_or_create_child_group(refined_run, "instances")
    total_frames = _resolved_total_frames(root, refined_run)
    payload = _extract_present_curated_rows_from_dense_root(refined_run)

    refined_row_ids = np.asarray(payload["refined_row_ids"], dtype=np.int64).reshape(-1)
    frame_indices = np.asarray(payload["frame_indices"], dtype=np.int32).reshape(-1)
    sort_idx = np.lexsort((refined_row_ids, frame_indices)) if frame_indices.size else np.asarray([], dtype=np.int64)

    refined_row_ids = refined_row_ids[sort_idx]
    frame_indices = frame_indices[sort_idx]
    bbox_img_xyxy = np.asarray(payload.get("bbox_img_xyxy"), dtype=np.float64).reshape(-1, 4)[sort_idx]
    bbox_norm_coords = np.asarray(payload["bbox_norm_coords"], dtype=np.float64).reshape(-1, 4)[sort_idx]
    source_kind_codes = np.asarray(refined_run["source_kind_codes"][:], dtype=np.int8).reshape(-1)[payload["present_row_indices"]][sort_idx]
    manual_edit_flags = (
        np.asarray(payload.get("manual_edit_flags"), dtype=bool).reshape(-1)[sort_idx]
        if "manual_edit_flags" in payload
        else np.zeros(frame_indices.shape[0], dtype=bool)
    )
    source_detect_row_index = (
        np.asarray(payload.get("source_detect_row_index"), dtype=np.int32).reshape(-1)[sort_idx]
        if "source_detect_row_index" in payload
        else np.full(frame_indices.shape[0], -1, dtype=np.int32)
    )
    frame_counts = np.bincount(frame_indices, minlength=total_frames).astype(np.int32, copy=False)
    frame_offsets = np.zeros(total_frames + 1, dtype=np.int64)
    if total_frames > 0:
        frame_offsets[1:] = np.cumsum(frame_counts, dtype=np.int64)
    reason_labels = np.asarray(payload.get("reason", np.full(frame_indices.shape[0], "", dtype=object)), dtype=object).reshape(-1)
    if reason_labels.shape[0] != frame_indices.shape[0]:
        reason_labels = np.full(frame_indices.shape[0], "", dtype=object)

    for name, data in (
        ("refined_row_ids", refined_row_ids),
        ("frame_indices", frame_indices),
        ("frame_offsets", frame_offsets),
        ("bbox_img_xyxy", bbox_img_xyxy),
        ("bbox_norm_coords", bbox_norm_coords),
        ("source_kind_codes", source_kind_codes),
        ("manual_edit_flags", manual_edit_flags),
        ("source_detect_row_index", source_detect_row_index),
        ("frame_counts", frame_counts),
    ):
        _write_common_array(subgroup, name, data)
    write_reason_columns(
        subgroup,
        reason_labels,
        max(1, int(frame_indices.shape[0])),
        include_reason_text=True,
        overwrite=True,
    )

    if "confidence_scores" in payload:
        _write_common_array(
            subgroup,
            "confidence_scores",
            np.asarray(payload["confidence_scores"], dtype=np.float32).reshape(-1)[sort_idx],
        )
    else:
        _delete_if_present(subgroup, "confidence_scores")
    if "class_ids" in payload:
        _write_common_array(
            subgroup,
            "class_ids",
            np.asarray(payload["class_ids"], dtype=np.int32).reshape(-1)[sort_idx],
        )
    else:
        _delete_if_present(subgroup, "class_ids")
    if "review_notes" in refined_run:
        review_notes = np.asarray(refined_run["review_notes"][:], dtype=object).reshape(-1)[payload["present_row_indices"]][sort_idx]
        _write_string_array(subgroup, "review_notes", review_notes, overwrite=True)
    else:
        _delete_if_present(subgroup, "review_notes")

    subgroup.attrs["row_sort_order"] = ["frame_indices", "refined_row_ids"]
    subgroup.attrs["source_kind_code_map"] = dict(REFINED_SOURCE_KIND_CODE_MAP)


def _write_source_detections_projection(
    root: zarr.Group,
    refined_run: zarr.Group,
) -> None:
    subgroup = _get_or_create_child_group(refined_run, "source_detections")
    detect_group, source_detect_run = _resolve_bound_source_detect_group(root, refined_run)
    width, height = _resolve_image_dimensions(root, detect_group=detect_group, refined_run=refined_run)

    if detect_group is None or width is None or height is None or width <= 0 or height <= 0:
        empty_int32 = np.zeros((0,), dtype=np.int32)
        empty_int64 = np.zeros((0,), dtype=np.int64)
        empty_bbox = np.zeros((0, 4), dtype=np.float64)
        empty_reason = np.zeros((0,), dtype=object)
        for name, data in (
            ("source_detect_row_index", empty_int32),
            ("frame_indices", empty_int32),
            ("bbox_img_xyxy", empty_bbox),
            ("bbox_norm_coords", empty_bbox),
            ("decision_codes", np.zeros((0,), dtype=np.int8)),
            ("resolved_refined_row_id", empty_int64),
        ):
            _write_common_array(subgroup, name, data)
        write_reason_columns(subgroup, empty_reason, 1, include_reason_text=True, overwrite=True)
        for name in ("confidence_scores", "class_ids", "review_notes"):
            _delete_if_present(subgroup, name)
        subgroup.attrs["decision_code_map"] = dict(REFINED_SOURCE_DETECTION_DECISION_CODE_MAP)
        if source_detect_run:
            subgroup.attrs["source_detect_run"] = source_detect_run
        return

    raw_frame_indices = np.asarray(detect_group["frame_indices"][:], dtype=np.int32).reshape(-1)
    raw_bbox_norm = np.asarray(detect_group["bbox_norm_coords"][:], dtype=np.float64).reshape(-1, 4)
    row_count = int(raw_frame_indices.shape[0])
    raw_bbox_img = _bbox_norm_to_img_xyxy_with_missing(raw_bbox_norm, width=int(width), height=int(height))
    decision_labels = np.full(row_count, "filtered", dtype=object)
    resolved_refined_row_id = np.full(row_count, -1, dtype=np.int64)
    reason_labels = np.full(row_count, "filtered", dtype=object)

    dense_frame_indices = np.asarray(refined_run["frame_indices"][:], dtype=np.int32).reshape(-1)
    dense_status_labels = _status_labels_from_codes(np.asarray(refined_run["status_codes"][:], dtype=np.int8).reshape(-1))
    dense_source_row_index = (
        np.asarray(refined_run["source_detect_row_index"][:], dtype=np.int32).reshape(-1)
        if "source_detect_row_index" in refined_run
        else np.full(dense_frame_indices.shape[0], -1, dtype=np.int32)
    )
    dense_row_ids = np.asarray(refined_run["refined_row_ids"][:], dtype=np.int64).reshape(-1)
    dense_manual_edit_flags = (
        np.asarray(refined_run["manual_edit_flags"][:], dtype=bool).reshape(-1)
        if "manual_edit_flags" in refined_run
        else np.zeros(dense_frame_indices.shape[0], dtype=bool)
    )
    dense_reason_labels = read_reason_labels(refined_run)
    if dense_reason_labels is None:
        dense_reason_labels = np.full(dense_frame_indices.shape[0], "", dtype=object)

    for frame_idx, status_label, source_row_idx, row_id, manual_edit, reason_label in zip(
        dense_frame_indices.tolist(),
        dense_status_labels.tolist(),
        dense_source_row_index.tolist(),
        dense_row_ids.tolist(),
        dense_manual_edit_flags.tolist(),
        np.asarray(dense_reason_labels, dtype=object).tolist(),
    ):
        raw_idx = int(source_row_idx)
        if raw_idx < 0 or raw_idx >= row_count:
            continue
        if str(status_label) == "present":
            decision_labels[raw_idx] = "accepted"
            resolved_refined_row_id[raw_idx] = int(row_id)
            reason_labels[raw_idx] = str(reason_label) or "accepted"
        elif str(status_label) == "filtered_out" and bool(manual_edit):
            decision_labels[raw_idx] = "manual_clear"
            reason_labels[raw_idx] = str(reason_label) or "manual_clear"
        elif str(status_label) == "filtered_out":
            decision_labels[raw_idx] = "filtered"
            reason_labels[raw_idx] = str(reason_label) or "filtered"
        else:
            reason_labels[raw_idx] = str(reason_label) or "filtered"

    decision_codes = np.asarray(
        [REFINED_SOURCE_DETECTION_DECISION_CODE_MAP[str(label)] for label in decision_labels],
        dtype=np.int8,
    )
    _write_common_array(subgroup, "source_detect_row_index", np.arange(row_count, dtype=np.int32))
    _write_common_array(subgroup, "frame_indices", raw_frame_indices)
    _write_common_array(subgroup, "bbox_img_xyxy", raw_bbox_img)
    _write_common_array(subgroup, "bbox_norm_coords", raw_bbox_norm)
    _write_common_array(subgroup, "decision_codes", decision_codes)
    _write_common_array(subgroup, "resolved_refined_row_id", resolved_refined_row_id)
    write_reason_columns(
        subgroup,
        np.asarray(reason_labels, dtype=object),
        max(1, row_count),
        include_reason_text=True,
        overwrite=True,
    )

    if "scores" in detect_group:
        _write_common_array(
            subgroup,
            "confidence_scores",
            np.asarray(detect_group["scores"][:], dtype=np.float32).reshape(-1),
        )
    else:
        _delete_if_present(subgroup, "confidence_scores")
    if "class_ids" in detect_group:
        _write_common_array(
            subgroup,
            "class_ids",
            np.asarray(detect_group["class_ids"][:], dtype=np.int32).reshape(-1),
        )
    else:
        _delete_if_present(subgroup, "class_ids")
    _delete_if_present(subgroup, "review_notes")

    subgroup.attrs["decision_code_map"] = dict(REFINED_SOURCE_DETECTION_DECISION_CODE_MAP)
    subgroup.attrs["source_detect_run"] = source_detect_run or ""
    subgroup.attrs["decision_projection_policy"] = "explicit_raw_backlinks_only"


def _sync_sparse_refined_detect_views(
    root: zarr.Group,
    refined_run: zarr.Group,
) -> None:
    _write_sparse_instances_projection(root, refined_run)
    _write_source_detections_projection(root, refined_run)
    refined_run.attrs["source_detection_decision_code_map"] = dict(REFINED_SOURCE_DETECTION_DECISION_CODE_MAP)


def _bbox_norm_to_img_xyxy_with_missing(
    bbox_norm: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    bbox_norm = np.asarray(bbox_norm, dtype=np.float64).reshape(-1, 4)
    bbox_img = np.full((bbox_norm.shape[0], 4), np.nan, dtype=np.float64)
    valid_mask = np.all(np.isfinite(bbox_norm), axis=1)
    if np.any(valid_mask):
        bbox_img[valid_mask] = bbox_norm_cxcywh_to_img_xyxy(
            bbox_norm[valid_mask],
            width=width,
            height=height,
        )
    return bbox_img


def _normalize_source_row_index(values: np.ndarray, length: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.int32).reshape(-1)
    if arr.shape[0] != length:
        raise ValueError("source_detect_row_index length does not match frame_indices length.")
    return arr


def _refresh_curated_refined_detect_metadata(
    refined_run: zarr.Group,
    *,
    resolved_refined_run_name: str,
    zarr_path: Optional[Path],
    command: Optional[str],
    env_info: Optional[Mapping[str, Any]],
    source_context: Optional[Mapping[str, Any]],
) -> None:
    has_sparse_instances = has_sparse_curated_refined_detect_instances_arrays(refined_run)
    if has_sparse_instances:
        instances_group = _get_child_group_if_present(refined_run, "instances")
        if instances_group is None:
            raise ValueError("Curated refined detect run is missing instances subgroup.")
        frame_indices_arr = np.asarray(instances_group["frame_indices"][:], dtype=np.int32).reshape(-1)
        entity_ids_arr = _local_entity_ids_from_frame_indices(frame_indices_arr)
        source_kind_codes = np.asarray(instances_group["source_kind_codes"][:], dtype=np.int8).reshape(-1)
        manual_edit_flags_arr = np.asarray(instances_group["manual_edit_flags"][:], dtype=bool).reshape(-1)
        total_frames = (
            int(instances_group["frame_counts"].shape[0])
            if "frame_counts" in instances_group
            else (
                int(np.max(frame_indices_arr)) + 1
                if frame_indices_arr.size
                else 0
            )
        )
        summary_statistics = _build_sparse_refined_detect_summary(refined_run, total_frames=total_frames)
    else:
        frame_indices_arr = np.asarray(refined_run["frame_indices"][:], dtype=np.int32).reshape(-1)
        entity_ids_arr = np.asarray(refined_run["entity_ids"][:], dtype=np.int32).reshape(-1)
        status_codes = np.asarray(refined_run["status_codes"][:], dtype=np.int8).reshape(-1)
        source_kind_codes = np.asarray(refined_run["source_kind_codes"][:], dtype=np.int8).reshape(-1)
        manual_edit_flags_arr = (
            np.asarray(refined_run["manual_edit_flags"][:], dtype=bool).reshape(-1)
            if "manual_edit_flags" in refined_run
            else np.zeros(frame_indices_arr.shape[0], dtype=bool)
        )
        summary_statistics = build_refined_detect_summary(
            frame_indices=frame_indices_arr,
            entity_ids=entity_ids_arr,
            status_codes=status_codes,
            source_kind_codes=source_kind_codes,
            manual_edit_flags=manual_edit_flags_arr,
        )

    attr_updates = {
        "curated_row_storage": "sparse_instances_v1" if has_sparse_instances else "dense_frame_entity_v3",
        "curated_primary_surface": "instances" if has_sparse_instances else "dense_root",
        "refined_storage_semantics": "sparse_instances_v1"
        if has_sparse_instances
        else "dense_frame_entity_v3",
        "entity_assignment_policy": "local_instance_index_per_frame"
        if has_sparse_instances
        else "single_subject_default_entity0",
        "coordinate_space": "full_image_xyxy",
        "row_identity_policy": "stable_sparse_refined_row_id"
        if has_sparse_instances
        else "stable_frame_entity_row_id",
        "status_code_map": dict(REFINED_DETECT_STATUS_CODE_MAP),
        "source_kind_code_map": dict(REFINED_SOURCE_KIND_CODE_MAP),
        "review_state_code_map": dict(REFINED_REVIEW_STATE_CODE_MAP),
        "summary_statistics": summary_statistics,
    }
    for key, value in attr_updates.items():
        refined_run.attrs[str(key)] = _json_safe_attr_value(value)
    if has_sparse_instances:
        for stale_attr in ("dense_projection_storage", "dense_axis_names", "artifact_state_code_map"):
            if stale_attr in refined_run.attrs:
                del refined_run.attrs[stale_attr]

    source_context_clean = clean_mapping(dict(source_context or {}))
    if source_context_clean:
        refined_run.attrs["curated_source_context"] = _json_safe_attr_value(source_context_clean)
    elif "curated_source_context" in refined_run.attrs:
        del refined_run.attrs["curated_source_context"]

    created_at_utc = _utc_now()
    git_info = dict((env_info or {}).get("git") or {})
    platform_info = dict((env_info or {}).get("platform") or {})
    refined_run.attrs["curation_provenance"] = _json_safe_attr_value(
        build_stage_provenance(
        stage="refined_detect_curation",
        created_at_utc=created_at_utc,
        command=command,
        version=git_info.get("short_hash") or git_info.get("commit_hash"),
        git={
            "commit": git_info.get("commit_hash"),
            "short": git_info.get("short_hash"),
            "branch": git_info.get("branch"),
            "is_dirty": git_info.get("is_dirty"),
            "remote": git_info.get("remote_url"),
        },
        environment=(env_info or {}).get("environment"),
        platform={
            "hostname": platform_info.get("hostname"),
            "system": platform_info.get("system"),
            "machine": platform_info.get("machine"),
            "python_version": platform_info.get("python_version"),
        },
        parameters={
            "curated_row_storage": "sparse_instances_v1" if has_sparse_instances else "dense_frame_entity_v3",
            "entity_assignment_policy": "local_instance_index_per_frame"
            if has_sparse_instances
            else "single_subject_default_entity0",
            "status_labels": ["present"] if has_sparse_instances else sorted(set(_status_labels_from_codes(status_codes).tolist())),
            "source_kind_labels": sorted(set(_source_labels_from_codes(source_kind_codes).tolist())),
        },
        inputs=clean_mapping(
            {
                "zarr_path": str(zarr_path) if zarr_path is not None else None,
                "source_detect_run": normalize_attr(refined_run.attrs.get("source_detect_run")),
                "source_refined_detect_run": resolved_refined_run_name,
                **source_context_clean,
            }
        ),
    ))
    refined_run.attrs["curation_updated_at_utc"] = created_at_utc


def _normalize_row_indices(row_indices: np.ndarray, row_count: int) -> np.ndarray:
    arr = np.asarray(row_indices, dtype=np.int64).reshape(-1)
    if arr.size == 0:
        return arr
    if np.any(arr < 0) or np.any(arr >= row_count):
        raise ValueError("row_indices contain out-of-range values.")
    if len(set(arr.tolist())) != int(arr.shape[0]):
        raise ValueError("row_indices must be unique.")
    return arr


def _ensure_optional_numeric_array(
    refined_run: zarr.Group,
    *,
    name: str,
    row_count: int,
    dtype: Any,
    fill_value: Any,
) -> Any:
    arr = refined_run.get(name)
    if arr is not None:
        return arr
    values = np.full(row_count, fill_value, dtype=dtype)
    return refined_run.create_array(name, data=values, overwrite=True)


def _next_refined_row_id(existing_run: Optional[zarr.Group]) -> int:
    max_row_id = -1
    if existing_run is not None:
        if has_sparse_curated_refined_detect_instances_arrays(existing_run):
            instances = _get_child_group_if_present(existing_run, "instances")
            if instances is not None:
                row_ids = np.asarray(instances["refined_row_ids"][:], dtype=np.int64).reshape(-1)
            else:
                row_ids = np.zeros((0,), dtype=np.int64)
            if row_ids.size:
                max_row_id = max(max_row_id, int(np.max(row_ids)))
        if has_curated_refined_detect_arrays(existing_run):
            row_ids = np.asarray(existing_run["refined_row_ids"][:], dtype=np.int64).reshape(-1)
            if row_ids.size:
                max_row_id = max(max_row_id, int(np.max(row_ids)))
    return max_row_id + 1


def _assign_sparse_instance_row_ids(
    existing_run: Optional[zarr.Group],
    *,
    frame_indices: np.ndarray,
    source_detect_row_index: np.ndarray,
    refined_row_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    frame_indices_arr = np.asarray(frame_indices, dtype=np.int32).reshape(-1)
    source_detect_row_index_arr = np.asarray(source_detect_row_index, dtype=np.int32).reshape(-1)
    if frame_indices_arr.shape[0] != source_detect_row_index_arr.shape[0]:
        raise ValueError("source_detect_row_index length does not match instance frame_indices length.")

    assigned = (
        np.asarray(refined_row_ids, dtype=np.int64).reshape(-1).copy()
        if refined_row_ids is not None
        else np.full(frame_indices_arr.shape[0], -1, dtype=np.int64)
    )
    if assigned.shape[0] != frame_indices_arr.shape[0]:
        raise ValueError("refined_row_ids length does not match instance frame_indices length.")

    by_source: dict[int, int] = {}
    by_frame: dict[int, int] = {}
    if existing_run is not None:
        if has_sparse_curated_refined_detect_instances_arrays(existing_run):
            instances = _get_child_group_if_present(existing_run, "instances")
            if instances is None:
                instances = None
            else:
                old_row_ids = np.asarray(instances["refined_row_ids"][:], dtype=np.int64).reshape(-1)
                old_frames = np.asarray(instances["frame_indices"][:], dtype=np.int32).reshape(-1)
                old_source_rows = np.asarray(instances["source_detect_row_index"][:], dtype=np.int32).reshape(-1)
                unique_old_frames, old_counts = np.unique(old_frames, return_counts=True)
                unique_frame_map = {
                    int(frame): int(count)
                    for frame, count in zip(unique_old_frames.tolist(), old_counts.tolist())
                }
                for row_id, frame_index, source_row_index in zip(
                    old_row_ids.tolist(),
                    old_frames.tolist(),
                    old_source_rows.tolist(),
                ):
                    if int(source_row_index) >= 0 and int(source_row_index) not in by_source:
                        by_source[int(source_row_index)] = int(row_id)
                    if unique_frame_map.get(int(frame_index), 0) == 1 and int(frame_index) not in by_frame:
                        by_frame[int(frame_index)] = int(row_id)
        if has_curated_refined_detect_arrays(existing_run):
            old_row_ids = np.asarray(existing_run["refined_row_ids"][:], dtype=np.int64).reshape(-1)
            old_frames = np.asarray(existing_run["frame_indices"][:], dtype=np.int32).reshape(-1)
            old_entities = np.asarray(existing_run["entity_ids"][:], dtype=np.int32).reshape(-1)
            old_source_rows = (
                np.asarray(existing_run["source_detect_row_index"][:], dtype=np.int32).reshape(-1)
                if "source_detect_row_index" in existing_run
                else np.full(old_frames.shape[0], -1, dtype=np.int32)
            )
            for row_id, frame_index, entity_id, source_row_index in zip(
                old_row_ids.tolist(),
                old_frames.tolist(),
                old_entities.tolist(),
                old_source_rows.tolist(),
            ):
                if int(entity_id) != 0:
                    continue
                if int(source_row_index) >= 0 and int(source_row_index) not in by_source:
                    by_source[int(source_row_index)] = int(row_id)
                if int(frame_index) not in by_frame:
                    by_frame[int(frame_index)] = int(row_id)

    next_row_id = _next_refined_row_id(existing_run)
    new_frame_counts: dict[int, int] = {}
    for frame_index in frame_indices_arr.tolist():
        new_frame_counts[int(frame_index)] = new_frame_counts.get(int(frame_index), 0) + 1
    used_ids = {int(value) for value in assigned.tolist() if int(value) >= 0}
    use_frame_identity = all(int(count) == 1 for count in new_frame_counts.values())

    for idx, (frame_index, source_row_index) in enumerate(
        zip(frame_indices_arr.tolist(), source_detect_row_index_arr.tolist())
    ):
        if int(assigned[idx]) >= 0:
            continue
        candidate: Optional[int] = None
        if int(source_row_index) >= 0:
            candidate = by_source.get(int(source_row_index))
        if candidate is None and new_frame_counts.get(int(frame_index), 0) == 1:
            candidate = by_frame.get(int(frame_index))
        if candidate is None and use_frame_identity and int(frame_index) not in used_ids:
            candidate = int(frame_index)
        if candidate is not None and int(candidate) not in used_ids:
            assigned[idx] = int(candidate)
            used_ids.add(int(candidate))
            continue
        assigned[idx] = int(next_row_id)
        used_ids.add(int(next_row_id))
        next_row_id += 1
    return assigned.astype(np.int64, copy=False)


def _write_sparse_instances_arrays(
    refined_run: zarr.Group,
    *,
    width: int,
    height: int,
    total_frames: int,
    refined_row_ids: np.ndarray,
    frame_indices: np.ndarray,
    bbox_norm_coords: np.ndarray,
    source_kind_codes: np.ndarray,
    manual_edit_flags: np.ndarray,
    source_detect_row_index: np.ndarray,
    reason_labels: np.ndarray,
    confidence_scores: Optional[np.ndarray] = None,
    class_ids: Optional[np.ndarray] = None,
    review_notes: Optional[np.ndarray] = None,
) -> None:
    subgroup = _get_or_create_child_group(refined_run, "instances")
    refined_row_ids_arr = np.asarray(refined_row_ids, dtype=np.int64).reshape(-1)
    frame_indices_arr = np.asarray(frame_indices, dtype=np.int32).reshape(-1)
    bbox_norm_arr = np.asarray(bbox_norm_coords, dtype=np.float64).reshape(-1, 4)
    source_kind_codes_arr = np.asarray(source_kind_codes, dtype=np.int8).reshape(-1)
    manual_edit_flags_arr = np.asarray(manual_edit_flags, dtype=bool).reshape(-1)
    source_detect_row_index_arr = np.asarray(source_detect_row_index, dtype=np.int32).reshape(-1)
    reason_labels_arr = np.asarray(reason_labels, dtype=object).reshape(-1)
    row_count = int(frame_indices_arr.shape[0])
    if not (
        refined_row_ids_arr.shape[0]
        == frame_indices_arr.shape[0]
        == bbox_norm_arr.shape[0]
        == source_kind_codes_arr.shape[0]
        == manual_edit_flags_arr.shape[0]
        == source_detect_row_index_arr.shape[0]
        == reason_labels_arr.shape[0]
    ):
        raise ValueError("Sparse instance arrays must agree on row count.")

    sort_idx = (
        np.lexsort((refined_row_ids_arr, frame_indices_arr))
        if row_count
        else np.asarray([], dtype=np.int64)
    )
    refined_row_ids_arr = refined_row_ids_arr[sort_idx]
    frame_indices_arr = frame_indices_arr[sort_idx]
    bbox_norm_arr = bbox_norm_arr[sort_idx]
    source_kind_codes_arr = source_kind_codes_arr[sort_idx]
    manual_edit_flags_arr = manual_edit_flags_arr[sort_idx]
    source_detect_row_index_arr = source_detect_row_index_arr[sort_idx]
    reason_labels_arr = reason_labels_arr[sort_idx]
    bbox_img_xyxy = _bbox_norm_to_img_xyxy_with_missing(
        bbox_norm_arr,
        width=width,
        height=height,
    )
    frame_counts = np.bincount(frame_indices_arr, minlength=total_frames).astype(np.int32, copy=False)
    frame_offsets = np.zeros(total_frames + 1, dtype=np.int64)
    if total_frames > 0:
        frame_offsets[1:] = np.cumsum(frame_counts, dtype=np.int64)

    for name, data in (
        ("refined_row_ids", refined_row_ids_arr),
        ("frame_indices", frame_indices_arr),
        ("frame_offsets", frame_offsets),
        ("bbox_img_xyxy", bbox_img_xyxy),
        ("bbox_norm_coords", bbox_norm_arr),
        ("source_kind_codes", source_kind_codes_arr),
        ("manual_edit_flags", manual_edit_flags_arr),
        ("source_detect_row_index", source_detect_row_index_arr),
        ("frame_counts", frame_counts),
    ):
        _write_common_array(subgroup, name, data)
    write_reason_columns(
        subgroup,
        reason_labels_arr,
        max(1, row_count),
        include_reason_text=True,
        overwrite=True,
    )

    if confidence_scores is not None:
        confidence_scores_arr = np.asarray(confidence_scores, dtype=np.float32).reshape(-1)
        if confidence_scores_arr.shape[0] != row_count:
            raise ValueError("instance confidence_scores length does not match row count.")
        _write_common_array(subgroup, "confidence_scores", confidence_scores_arr[sort_idx])
    else:
        _delete_if_present(subgroup, "confidence_scores")
    if class_ids is not None:
        class_ids_arr = np.asarray(class_ids, dtype=np.int32).reshape(-1)
        if class_ids_arr.shape[0] != row_count:
            raise ValueError("instance class_ids length does not match row count.")
        _write_common_array(subgroup, "class_ids", class_ids_arr[sort_idx])
    else:
        _delete_if_present(subgroup, "class_ids")
    if review_notes is not None:
        review_notes_arr = np.asarray(review_notes, dtype=object).reshape(-1)
        if review_notes_arr.shape[0] != row_count:
            raise ValueError("instance review_notes length does not match row count.")
        _write_string_array(subgroup, "review_notes", review_notes_arr[sort_idx], overwrite=True)
    else:
        _delete_if_present(subgroup, "review_notes")

    subgroup.attrs["row_sort_order"] = ["frame_indices", "refined_row_ids"]
    subgroup.attrs["source_kind_code_map"] = dict(REFINED_SOURCE_KIND_CODE_MAP)


def _write_source_detections_arrays(
    refined_run: zarr.Group,
    *,
    width: int,
    height: int,
    source_detect_row_index: np.ndarray,
    frame_indices: np.ndarray,
    bbox_norm_coords: np.ndarray,
    decision_codes: np.ndarray,
    resolved_refined_row_id: np.ndarray,
    reason_labels: np.ndarray,
    confidence_scores: Optional[np.ndarray] = None,
    class_ids: Optional[np.ndarray] = None,
    review_notes: Optional[np.ndarray] = None,
) -> None:
    subgroup = _get_or_create_child_group(refined_run, "source_detections")
    source_detect_row_index_arr = np.asarray(source_detect_row_index, dtype=np.int32).reshape(-1)
    frame_indices_arr = np.asarray(frame_indices, dtype=np.int32).reshape(-1)
    bbox_norm_arr = np.asarray(bbox_norm_coords, dtype=np.float64).reshape(-1, 4)
    decision_codes_arr = np.asarray(decision_codes, dtype=np.int8).reshape(-1)
    resolved_refined_row_id_arr = np.asarray(resolved_refined_row_id, dtype=np.int64).reshape(-1)
    reason_labels_arr = np.asarray(reason_labels, dtype=object).reshape(-1)
    row_count = int(source_detect_row_index_arr.shape[0])
    if not (
        source_detect_row_index_arr.shape[0]
        == frame_indices_arr.shape[0]
        == bbox_norm_arr.shape[0]
        == decision_codes_arr.shape[0]
        == resolved_refined_row_id_arr.shape[0]
        == reason_labels_arr.shape[0]
    ):
        raise ValueError("source_detections arrays must agree on row count.")

    sort_idx = (
        np.argsort(source_detect_row_index_arr, kind="stable")
        if row_count
        else np.asarray([], dtype=np.int64)
    )
    source_detect_row_index_arr = source_detect_row_index_arr[sort_idx]
    frame_indices_arr = frame_indices_arr[sort_idx]
    bbox_norm_arr = bbox_norm_arr[sort_idx]
    decision_codes_arr = decision_codes_arr[sort_idx]
    resolved_refined_row_id_arr = resolved_refined_row_id_arr[sort_idx]
    reason_labels_arr = reason_labels_arr[sort_idx]
    bbox_img_xyxy = _bbox_norm_to_img_xyxy_with_missing(
        bbox_norm_arr,
        width=width,
        height=height,
    )

    for name, data in (
        ("source_detect_row_index", source_detect_row_index_arr),
        ("frame_indices", frame_indices_arr),
        ("bbox_img_xyxy", bbox_img_xyxy),
        ("bbox_norm_coords", bbox_norm_arr),
        ("decision_codes", decision_codes_arr),
        ("resolved_refined_row_id", resolved_refined_row_id_arr),
    ):
        _write_common_array(subgroup, name, data)
    write_reason_columns(
        subgroup,
        reason_labels_arr,
        max(1, row_count),
        include_reason_text=True,
        overwrite=True,
    )

    if confidence_scores is not None:
        confidence_scores_arr = np.asarray(confidence_scores, dtype=np.float32).reshape(-1)
        if confidence_scores_arr.shape[0] != row_count:
            raise ValueError("source_detections confidence_scores length does not match row count.")
        _write_common_array(subgroup, "confidence_scores", confidence_scores_arr[sort_idx])
    else:
        _delete_if_present(subgroup, "confidence_scores")
    if class_ids is not None:
        class_ids_arr = np.asarray(class_ids, dtype=np.int32).reshape(-1)
        if class_ids_arr.shape[0] != row_count:
            raise ValueError("source_detections class_ids length does not match row count.")
        _write_common_array(subgroup, "class_ids", class_ids_arr[sort_idx])
    else:
        _delete_if_present(subgroup, "class_ids")
    if review_notes is not None:
        review_notes_arr = np.asarray(review_notes, dtype=object).reshape(-1)
        if review_notes_arr.shape[0] != row_count:
            raise ValueError("source_detections review_notes length does not match row count.")
        _write_string_array(subgroup, "review_notes", review_notes_arr[sort_idx], overwrite=True)
    else:
        _delete_if_present(subgroup, "review_notes")

    subgroup.attrs["decision_code_map"] = dict(REFINED_SOURCE_DETECTION_DECISION_CODE_MAP)
    subgroup.attrs["source_detect_run"] = normalize_attr(refined_run.attrs.get("source_detect_run")) or ""
    subgroup.attrs["decision_projection_policy"] = "canonical_sparse_surface"


def _existing_dense_or_default(
    refined_run: zarr.Group,
    *,
    name: str,
    row_count: int,
    dtype: Any,
    fill_value: Any,
) -> np.ndarray:
    arr = refined_run.get(name)
    if arr is not None:
        values = np.asarray(arr[:], dtype=dtype).reshape(-1)
        if values.shape[0] == row_count:
            return values
    return np.full(row_count, fill_value, dtype=dtype)


def _write_dense_curated_root_arrays(
    refined_run: zarr.Group,
    *,
    width: int,
    height: int,
    refined_row_ids: np.ndarray,
    frame_indices: np.ndarray,
    entity_ids: np.ndarray,
    bbox_norm_coords: np.ndarray,
    status_codes: np.ndarray,
    source_kind_codes: np.ndarray,
    review_state_codes: np.ndarray,
    keypoints_state_codes: np.ndarray,
    subject_mask_state_codes: np.ndarray,
    eye_mask_state_codes: np.ndarray,
    swim_bladder_state_codes: np.ndarray,
    source_detect_row_index: np.ndarray,
    reason_labels: np.ndarray,
    manual_edit_flags: Optional[np.ndarray] = None,
    detection_source: Optional[np.ndarray] = None,
    confidence_scores: Optional[np.ndarray] = None,
    class_ids: Optional[np.ndarray] = None,
    review_notes: Optional[Sequence[str]] = None,
) -> None:
    refined_row_ids_arr = np.asarray(refined_row_ids, dtype=np.int64).reshape(-1)
    frame_indices_arr = np.asarray(frame_indices, dtype=np.int32).reshape(-1)
    entity_ids_arr = np.asarray(entity_ids, dtype=np.int32).reshape(-1)
    bbox_norm_arr = np.asarray(bbox_norm_coords, dtype=np.float64).reshape(-1, 4)
    status_codes_arr = np.asarray(status_codes, dtype=np.int8).reshape(-1)
    source_kind_codes_arr = np.asarray(source_kind_codes, dtype=np.int8).reshape(-1)
    review_state_codes_arr = np.asarray(review_state_codes, dtype=np.int8).reshape(-1)
    keypoints_state_codes_arr = np.asarray(keypoints_state_codes, dtype=np.int8).reshape(-1)
    subject_mask_state_codes_arr = np.asarray(subject_mask_state_codes, dtype=np.int8).reshape(-1)
    eye_mask_state_codes_arr = np.asarray(eye_mask_state_codes, dtype=np.int8).reshape(-1)
    swim_bladder_state_codes_arr = np.asarray(swim_bladder_state_codes, dtype=np.int8).reshape(-1)
    source_detect_row_index_arr = np.asarray(source_detect_row_index, dtype=np.int32).reshape(-1)
    reason_labels_arr = np.asarray(reason_labels, dtype=object).reshape(-1)
    row_count = int(frame_indices_arr.shape[0])
    if not (
        refined_row_ids_arr.shape[0]
        == frame_indices_arr.shape[0]
        == entity_ids_arr.shape[0]
        == bbox_norm_arr.shape[0]
        == status_codes_arr.shape[0]
        == source_kind_codes_arr.shape[0]
        == review_state_codes_arr.shape[0]
        == keypoints_state_codes_arr.shape[0]
        == subject_mask_state_codes_arr.shape[0]
        == eye_mask_state_codes_arr.shape[0]
        == swim_bladder_state_codes_arr.shape[0]
        == source_detect_row_index_arr.shape[0]
        == reason_labels_arr.shape[0]
    ):
        raise ValueError("Dense curated root arrays must agree on row count.")

    manual_edit_flags_arr = (
        np.asarray(manual_edit_flags, dtype=bool).reshape(-1)
        if manual_edit_flags is not None
        else np.zeros(row_count, dtype=bool)
    )
    if manual_edit_flags_arr.shape[0] != row_count:
        raise ValueError("manual_edit_flags length does not match dense root row count.")
    detection_source_arr = (
        np.asarray(detection_source, dtype=np.int8).reshape(-1)
        if detection_source is not None
        else None
    )
    if detection_source_arr is not None and detection_source_arr.shape[0] != row_count:
        raise ValueError("detection_source length does not match dense root row count.")
    confidence_scores_arr = (
        np.asarray(confidence_scores, dtype=np.float32).reshape(-1)
        if confidence_scores is not None
        else np.full(row_count, np.nan, dtype=np.float32)
    )
    if confidence_scores_arr.shape[0] != row_count:
        raise ValueError("confidence_scores length does not match dense root row count.")
    class_ids_arr = (
        np.asarray(class_ids, dtype=np.int32).reshape(-1)
        if class_ids is not None
        else np.full(row_count, -1, dtype=np.int32)
    )
    if class_ids_arr.shape[0] != row_count:
        raise ValueError("class_ids length does not match dense root row count.")

    bbox_img_xyxy = _bbox_norm_to_img_xyxy_with_missing(
        bbox_norm_arr,
        width=width,
        height=height,
    )

    for name in (
        "source_sparse_row_index",
        "source_sparse_group_codes",
        "active_sparse_group",
        "active_sparse_group_kind",
        "active_sparse_group_path",
    ):
        if name in refined_run:
            del refined_run[name]
        elif name in refined_run.attrs:
            del refined_run.attrs[name]

    for name, data in (
        ("refined_row_ids", refined_row_ids_arr),
        ("frame_indices", frame_indices_arr),
        ("entity_ids", entity_ids_arr),
        ("bbox_img_xyxy", bbox_img_xyxy),
        ("bbox_norm_coords", bbox_norm_arr),
        ("status_codes", status_codes_arr),
        ("source_kind_codes", source_kind_codes_arr),
        ("manual_edit_flags", manual_edit_flags_arr),
        ("source_detect_row_index", source_detect_row_index_arr),
        ("review_state_codes", review_state_codes_arr),
        ("keypoints_state_codes", keypoints_state_codes_arr),
        ("subject_mask_state_codes", subject_mask_state_codes_arr),
        ("eye_mask_state_codes", eye_mask_state_codes_arr),
        ("swim_bladder_state_codes", swim_bladder_state_codes_arr),
        ("confidence_scores", confidence_scores_arr),
        ("class_ids", class_ids_arr),
    ):
        _write_common_array(refined_run, name, data)
    if detection_source_arr is not None:
        _write_common_array(refined_run, "detection_source", detection_source_arr)
    elif "detection_source" in refined_run:
        del refined_run["detection_source"]

    write_reason_columns(
        refined_run,
        reason_labels_arr,
        max(1, row_count),
        include_reason_text=True,
        overwrite=True,
    )
    if review_notes is not None:
        review_notes_arr = np.asarray(review_notes, dtype=object).reshape(-1)
        if review_notes_arr.shape[0] != row_count:
            raise ValueError("review_notes length does not match dense root row count.")
        _write_string_array(refined_run, "review_notes", review_notes_arr, overwrite=True)
    elif "review_notes" in refined_run:
        del refined_run["review_notes"]


def _sync_dense_curated_refined_root_from_sparse_views(
    root: zarr.Group,
    refined_run: zarr.Group,
) -> None:
    width, height = _resolve_image_dimensions(root, refined_run=refined_run)
    if width is None or height is None or width <= 0 or height <= 0:
        raise ValueError("Root attrs must include positive width and height.")

    total_frames = _resolved_total_frames(root, refined_run)
    existing_run = refined_run if has_curated_refined_detect_arrays(refined_run) else None
    preserve_existing_layout = False
    if existing_run is not None:
        existing_frame_indices = np.asarray(existing_run["frame_indices"][:], dtype=np.int32).reshape(-1)
        existing_entity_ids = np.asarray(existing_run["entity_ids"][:], dtype=np.int32).reshape(-1)
        preserve_existing_layout = (
            existing_frame_indices.shape[0] == total_frames
            and existing_entity_ids.shape[0] == total_frames
            and np.all(existing_entity_ids == 0)
            and np.array_equal(np.sort(existing_frame_indices), np.arange(total_frames, dtype=np.int32))
        )
        if preserve_existing_layout:
            dense_frame_indices = existing_frame_indices.copy()
            dense_entity_ids = existing_entity_ids.copy()
        else:
            dense_frame_indices = np.arange(total_frames, dtype=np.int32)
            dense_entity_ids = np.zeros(total_frames, dtype=np.int32)
    else:
        dense_frame_indices = np.arange(total_frames, dtype=np.int32)
        dense_entity_ids = np.zeros(total_frames, dtype=np.int32)

    dense_refined_row_ids = assign_refined_row_ids(
        existing_run,
        frame_indices=dense_frame_indices,
        entity_ids=dense_entity_ids,
    )
    dense_row_by_frame = {int(frame): idx for idx, frame in enumerate(dense_frame_indices.tolist())}

    dense_bbox_norm = np.full((total_frames, 4), np.nan, dtype=np.float64)
    dense_status_labels = np.full(total_frames, "missing", dtype=object)
    dense_source_kind_labels = np.full(total_frames, "none", dtype=object)
    dense_reason_labels = np.full(total_frames, "missing_detection", dtype=object)
    dense_source_detect_row_index = np.full(total_frames, -1, dtype=np.int32)
    dense_manual_edit_flags = np.zeros(total_frames, dtype=bool)
    dense_confidence_scores = np.full(total_frames, np.nan, dtype=np.float32)
    dense_class_ids = np.full(total_frames, -1, dtype=np.int32)
    dense_review_notes = np.full(total_frames, "", dtype=object)

    instances = _get_child_group_if_present(refined_run, "instances")
    instances_multi_frames: set[int] = set()
    if instances is not None and has_sparse_curated_refined_detect_instances_arrays(refined_run):
        inst_frames = np.asarray(instances["frame_indices"][:], dtype=np.int32).reshape(-1)
        inst_bbox = np.asarray(instances["bbox_norm_coords"][:], dtype=np.float64).reshape(-1, 4)
        inst_source_kind_codes = np.asarray(instances["source_kind_codes"][:], dtype=np.int8).reshape(-1)
        inst_source_kind_labels = _source_labels_from_codes(inst_source_kind_codes)
        inst_source_detect_row_index = np.asarray(instances["source_detect_row_index"][:], dtype=np.int32).reshape(-1)
        inst_manual_edit_flags = np.asarray(instances["manual_edit_flags"][:], dtype=bool).reshape(-1)
        inst_reason_labels = read_reason_labels(instances)
        if inst_reason_labels is None:
            inst_reason_labels = np.full(inst_frames.shape[0], "", dtype=object)
        inst_confidence_scores = (
            np.asarray(instances["confidence_scores"][:], dtype=np.float32).reshape(-1)
            if "confidence_scores" in instances
            else np.full(inst_frames.shape[0], np.nan, dtype=np.float32)
        )
        inst_class_ids = (
            np.asarray(instances["class_ids"][:], dtype=np.int32).reshape(-1)
            if "class_ids" in instances
            else np.full(inst_frames.shape[0], -1, dtype=np.int32)
        )
        inst_review_notes = (
            np.asarray(instances["review_notes"][:], dtype=object).reshape(-1)
            if "review_notes" in instances
            else np.full(inst_frames.shape[0], "", dtype=object)
        )
        unique_frames, counts = np.unique(inst_frames, return_counts=True) if inst_frames.size else (np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32))
        instances_multi_frames = {
            int(frame)
            for frame, count in zip(unique_frames.tolist(), counts.tolist())
            if int(count) > 1
        }
        for frame_index, bbox_norm, source_kind_label, source_row_index, manual_edit_flag, reason_label, confidence_score, class_id, review_note in zip(
            inst_frames.tolist(),
            inst_bbox.tolist(),
            inst_source_kind_labels.tolist(),
            inst_source_detect_row_index.tolist(),
            inst_manual_edit_flags.tolist(),
            np.asarray(inst_reason_labels, dtype=object).tolist(),
            inst_confidence_scores.tolist(),
            inst_class_ids.tolist(),
            np.asarray(inst_review_notes, dtype=object).tolist(),
        ):
            frame = int(frame_index)
            dense_row_index = dense_row_by_frame.get(frame)
            if dense_row_index is None:
                continue
            if frame in instances_multi_frames:
                continue
            dense_bbox_norm[dense_row_index] = np.asarray(bbox_norm, dtype=np.float64)
            dense_status_labels[dense_row_index] = "present"
            dense_source_kind_labels[dense_row_index] = str(source_kind_label)
            dense_reason_labels[dense_row_index] = str(reason_label) or "present"
            dense_source_detect_row_index[dense_row_index] = int(source_row_index)
            dense_manual_edit_flags[dense_row_index] = bool(manual_edit_flag)
            dense_confidence_scores[dense_row_index] = np.float32(confidence_score)
            dense_class_ids[dense_row_index] = np.int32(class_id)
            dense_review_notes[dense_row_index] = str(review_note or "")

    source_context = refined_run.attrs.get("curated_source_context")
    ignore_source_only_frames = (
        isinstance(source_context, Mapping)
        and normalize_attr(source_context.get("materialized_from_group_kind")) not in (None, "", "raw")
    )

    source_detections = _get_child_group_if_present(refined_run, "source_detections")
    if source_detections is not None and has_curated_refined_source_detections_projection(refined_run):
        src_frames = np.asarray(source_detections["frame_indices"][:], dtype=np.int32).reshape(-1)
        src_source_rows = np.asarray(source_detections["source_detect_row_index"][:], dtype=np.int32).reshape(-1)
        src_bbox = np.asarray(source_detections["bbox_norm_coords"][:], dtype=np.float64).reshape(-1, 4)
        src_decision_codes = np.asarray(source_detections["decision_codes"][:], dtype=np.int8).reshape(-1)
        src_decision_labels = _decision_labels_from_codes(src_decision_codes)
        src_reason_labels = read_reason_labels(source_detections)
        if src_reason_labels is None:
            src_reason_labels = np.full(src_frames.shape[0], "", dtype=object)
        src_confidence_scores = (
            np.asarray(source_detections["confidence_scores"][:], dtype=np.float32).reshape(-1)
            if "confidence_scores" in source_detections
            else np.full(src_frames.shape[0], np.nan, dtype=np.float32)
        )
        src_class_ids = (
            np.asarray(source_detections["class_ids"][:], dtype=np.int32).reshape(-1)
            if "class_ids" in source_detections
            else np.full(src_frames.shape[0], -1, dtype=np.int32)
        )
        src_review_notes = (
            np.asarray(source_detections["review_notes"][:], dtype=object).reshape(-1)
            if "review_notes" in source_detections
            else np.full(src_frames.shape[0], "", dtype=object)
        )
        rows_by_frame: dict[int, list[int]] = {}
        for idx, frame in enumerate(src_frames.tolist()):
            rows_by_frame.setdefault(int(frame), []).append(int(idx))
        for frame, row_indices in rows_by_frame.items():
            dense_row_index = dense_row_by_frame.get(int(frame))
            if dense_row_index is None:
                continue
            if frame in instances_multi_frames:
                dense_status_labels[dense_row_index] = "ambiguous"
                dense_source_kind_labels[dense_row_index] = "none"
                dense_reason_labels[dense_row_index] = "multiple_instances"
                continue
            if dense_status_labels[dense_row_index] == "present":
                continue
            if ignore_source_only_frames:
                continue
            if len(row_indices) > 1:
                dense_status_labels[dense_row_index] = "ambiguous"
                dense_source_kind_labels[dense_row_index] = "none"
                dense_reason_labels[dense_row_index] = "multiple_candidates"
                dense_manual_edit_flags[dense_row_index] = any(
                    str(src_decision_labels[row_idx]) == "manual_clear" for row_idx in row_indices
                )
                continue
            row_idx = int(row_indices[0])
            decision_label = str(src_decision_labels[row_idx])
            reason_label = str(np.asarray(src_reason_labels, dtype=object)[row_idx]) or decision_label
            dense_bbox_norm[dense_row_index] = np.asarray(src_bbox[row_idx], dtype=np.float64)
            dense_source_detect_row_index[dense_row_index] = int(src_source_rows[row_idx])
            dense_confidence_scores[dense_row_index] = np.float32(src_confidence_scores[row_idx])
            dense_class_ids[dense_row_index] = np.int32(src_class_ids[row_idx])
            dense_review_notes[dense_row_index] = str(np.asarray(src_review_notes, dtype=object)[row_idx] or "")
            if decision_label == "manual_clear":
                dense_status_labels[dense_row_index] = "filtered_out"
                dense_source_kind_labels[dense_row_index] = "none"
                dense_reason_labels[dense_row_index] = reason_label
                dense_manual_edit_flags[dense_row_index] = True
            elif decision_label in {"filtered", "duplicate"}:
                dense_status_labels[dense_row_index] = "filtered_out"
                dense_source_kind_labels[dense_row_index] = "raw_detect"
                dense_reason_labels[dense_row_index] = reason_label
            elif decision_label == "accepted":
                dense_status_labels[dense_row_index] = "filtered_out"
                dense_source_kind_labels[dense_row_index] = "raw_detect"
                dense_reason_labels[dense_row_index] = reason_label

    review_state_code = _review_state_code_for_run(refined_run)
    review_state_codes = np.full(total_frames, review_state_code, dtype=np.int8)
    downstream_default = np.full(
        total_frames,
        REFINED_ARTIFACT_STATE_CODE_MAP["not_generated"],
        dtype=np.int8,
    )
    keypoints_state_codes = _existing_dense_or_default(
        refined_run,
        name="keypoints_state_codes",
        row_count=total_frames,
        dtype=np.int8,
        fill_value=REFINED_ARTIFACT_STATE_CODE_MAP["not_generated"],
    )
    subject_mask_state_codes = _existing_dense_or_default(
        refined_run,
        name="subject_mask_state_codes",
        row_count=total_frames,
        dtype=np.int8,
        fill_value=REFINED_ARTIFACT_STATE_CODE_MAP["not_generated"],
    )
    eye_mask_state_codes = _existing_dense_or_default(
        refined_run,
        name="eye_mask_state_codes",
        row_count=total_frames,
        dtype=np.int8,
        fill_value=REFINED_ARTIFACT_STATE_CODE_MAP["not_generated"],
    )
    swim_bladder_state_codes = _existing_dense_or_default(
        refined_run,
        name="swim_bladder_state_codes",
        row_count=total_frames,
        dtype=np.int8,
        fill_value=REFINED_ARTIFACT_STATE_CODE_MAP["not_generated"],
    )
    if keypoints_state_codes.shape[0] != total_frames:
        keypoints_state_codes = downstream_default.copy()
    if subject_mask_state_codes.shape[0] != total_frames:
        subject_mask_state_codes = downstream_default.copy()
    if eye_mask_state_codes.shape[0] != total_frames:
        eye_mask_state_codes = downstream_default.copy()
    if swim_bladder_state_codes.shape[0] != total_frames:
        swim_bladder_state_codes = downstream_default.copy()

    status_codes = np.asarray(
        [REFINED_DETECT_STATUS_CODE_MAP[str(label)] for label in dense_status_labels.tolist()],
        dtype=np.int8,
    )
    source_kind_codes = np.asarray(
        [REFINED_SOURCE_KIND_CODE_MAP[str(label)] for label in dense_source_kind_labels.tolist()],
        dtype=np.int8,
    )
    detection_source = _build_detection_source_from_source_kind_codes(source_kind_codes)
    review_notes = dense_review_notes if np.any(dense_review_notes != "") else None

    _write_dense_curated_root_arrays(
        refined_run,
        width=int(width),
        height=int(height),
        refined_row_ids=dense_refined_row_ids,
        frame_indices=dense_frame_indices,
        entity_ids=dense_entity_ids,
        bbox_norm_coords=dense_bbox_norm,
        status_codes=status_codes,
        source_kind_codes=source_kind_codes,
        review_state_codes=review_state_codes,
        keypoints_state_codes=keypoints_state_codes,
        subject_mask_state_codes=subject_mask_state_codes,
        eye_mask_state_codes=eye_mask_state_codes,
        swim_bladder_state_codes=swim_bladder_state_codes,
        source_detect_row_index=dense_source_detect_row_index,
        reason_labels=dense_reason_labels,
        manual_edit_flags=dense_manual_edit_flags,
        detection_source=detection_source,
        confidence_scores=dense_confidence_scores,
        class_ids=dense_class_ids,
        review_notes=review_notes,
    )


def write_curated_refined_detect_surfaces(
    root: zarr.Group,
    *,
    zarr_path: Optional[Path] = None,
    refined_family_path: Optional[str] = None,
    refined_run_name: Optional[str] = None,
    instance_frame_indices: np.ndarray,
    instance_bbox_norm_coords: np.ndarray,
    instance_source_kind_labels: Sequence[str],
    instance_reason_labels: Sequence[str],
    instance_source_detect_row_index: Optional[np.ndarray] = None,
    instance_manual_edit_flags: Optional[np.ndarray] = None,
    instance_confidence_scores: Optional[np.ndarray] = None,
    instance_class_ids: Optional[np.ndarray] = None,
    instance_review_notes: Optional[Sequence[str]] = None,
    instance_refined_row_ids: Optional[np.ndarray] = None,
    source_detection_source_detect_row_index: Optional[np.ndarray] = None,
    source_detection_frame_indices: Optional[np.ndarray] = None,
    source_detection_bbox_norm_coords: Optional[np.ndarray] = None,
    source_detection_decision_labels: Optional[Sequence[str]] = None,
    source_detection_reason_labels: Optional[Sequence[str]] = None,
    source_detection_confidence_scores: Optional[np.ndarray] = None,
    source_detection_class_ids: Optional[np.ndarray] = None,
    source_detection_review_notes: Optional[Sequence[str]] = None,
    command: Optional[str] = None,
    env_info: Optional[Mapping[str, Any]] = None,
    source_context: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    refined_parent, _ = _resolve_refined_parent(root, refined_family_path=refined_family_path)
    resolved_refined_run_name = normalize_attr(refined_run_name) or normalize_attr(refined_parent.attrs.get("latest"))
    if not resolved_refined_run_name or resolved_refined_run_name not in refined_parent:
        raise ValueError("No refined detect run available.")
    refined_run = _open_named_child_group(refined_parent, resolved_refined_run_name, mode="a")
    width, height = _resolve_image_dimensions(root, refined_run=refined_run)
    if width is None or height is None or width <= 0 or height <= 0:
        raise ValueError("Root attrs must include positive width and height.")
    total_frames = _resolved_total_frames(root, refined_run)

    instance_frame_indices_arr = np.asarray(instance_frame_indices, dtype=np.int32).reshape(-1)
    instance_bbox_norm_arr = np.asarray(instance_bbox_norm_coords, dtype=np.float64).reshape(-1, 4)
    instance_source_kind_labels_arr = np.asarray(instance_source_kind_labels, dtype=object).reshape(-1)
    instance_reason_labels_arr = np.asarray(instance_reason_labels, dtype=object).reshape(-1)
    instance_row_count = int(instance_frame_indices_arr.shape[0])
    if not (
        instance_bbox_norm_arr.shape[0]
        == instance_source_kind_labels_arr.shape[0]
        == instance_reason_labels_arr.shape[0]
        == instance_row_count
    ):
        raise ValueError("Instance arrays must agree on row count.")
    instance_source_detect_row_index_arr = (
        np.asarray(instance_source_detect_row_index, dtype=np.int32).reshape(-1)
        if instance_source_detect_row_index is not None
        else np.full(instance_row_count, -1, dtype=np.int32)
    )
    if instance_source_detect_row_index_arr.shape[0] != instance_row_count:
        raise ValueError("instance_source_detect_row_index length does not match instance row count.")
    instance_manual_edit_flags_arr = (
        np.asarray(instance_manual_edit_flags, dtype=bool).reshape(-1)
        if instance_manual_edit_flags is not None
        else np.zeros(instance_row_count, dtype=bool)
    )
    if instance_manual_edit_flags_arr.shape[0] != instance_row_count:
        raise ValueError("instance_manual_edit_flags length does not match instance row count.")
    instance_confidence_scores_arr = (
        np.asarray(instance_confidence_scores, dtype=np.float32).reshape(-1)
        if instance_confidence_scores is not None
        else None
    )
    if instance_confidence_scores_arr is not None and instance_confidence_scores_arr.shape[0] != instance_row_count:
        raise ValueError("instance_confidence_scores length does not match instance row count.")
    instance_class_ids_arr = (
        np.asarray(instance_class_ids, dtype=np.int32).reshape(-1)
        if instance_class_ids is not None
        else None
    )
    if instance_class_ids_arr is not None and instance_class_ids_arr.shape[0] != instance_row_count:
        raise ValueError("instance_class_ids length does not match instance row count.")
    instance_review_notes_arr = (
        np.asarray(instance_review_notes, dtype=object).reshape(-1)
        if instance_review_notes is not None
        else None
    )
    if instance_review_notes_arr is not None and instance_review_notes_arr.shape[0] != instance_row_count:
        raise ValueError("instance_review_notes length does not match instance row count.")
    instance_source_kind_codes = np.asarray(
        [REFINED_SOURCE_KIND_CODE_MAP[str(label)] for label in instance_source_kind_labels_arr.tolist()],
        dtype=np.int8,
    )
    instance_refined_row_ids_arr = _assign_sparse_instance_row_ids(
        refined_run,
        frame_indices=instance_frame_indices_arr,
        source_detect_row_index=instance_source_detect_row_index_arr,
        refined_row_ids=instance_refined_row_ids,
    )

    if source_detection_frame_indices is None and source_detection_bbox_norm_coords is None and source_detection_source_detect_row_index is None:
        source_row_count = 0
        source_detection_source_row_arr = np.empty((0,), dtype=np.int32)
        source_detection_frame_indices_arr = np.empty((0,), dtype=np.int32)
        source_detection_bbox_norm_arr = np.empty((0, 4), dtype=np.float64)
    else:
        if source_detection_source_detect_row_index is None or source_detection_frame_indices is None or source_detection_bbox_norm_coords is None:
            raise ValueError("source_detections require source_detect_row_index, frame_indices, and bbox_norm_coords.")
        source_detection_source_row_arr = np.asarray(source_detection_source_detect_row_index, dtype=np.int32).reshape(-1)
        source_detection_frame_indices_arr = np.asarray(source_detection_frame_indices, dtype=np.int32).reshape(-1)
        source_detection_bbox_norm_arr = np.asarray(source_detection_bbox_norm_coords, dtype=np.float64).reshape(-1, 4)
        source_row_count = int(source_detection_source_row_arr.shape[0])
        if not (
            source_detection_frame_indices_arr.shape[0]
            == source_detection_bbox_norm_arr.shape[0]
            == source_row_count
        ):
            raise ValueError("source_detections core arrays must agree on row count.")

    source_detection_decision_labels_arr = (
        np.asarray(source_detection_decision_labels, dtype=object).reshape(-1)
        if source_detection_decision_labels is not None
        else np.full(source_row_count, "filtered", dtype=object)
    )
    if source_detection_decision_labels_arr.shape[0] != source_row_count:
        raise ValueError("source_detection_decision_labels length does not match source row count.")
    source_detection_reason_labels_arr = (
        np.asarray(source_detection_reason_labels, dtype=object).reshape(-1)
        if source_detection_reason_labels is not None
        else np.asarray(source_detection_decision_labels_arr, dtype=object).copy()
    )
    if source_detection_reason_labels_arr.shape[0] != source_row_count:
        raise ValueError("source_detection_reason_labels length does not match source row count.")
    source_detection_confidence_scores_arr = (
        np.asarray(source_detection_confidence_scores, dtype=np.float32).reshape(-1)
        if source_detection_confidence_scores is not None
        else None
    )
    if source_detection_confidence_scores_arr is not None and source_detection_confidence_scores_arr.shape[0] != source_row_count:
        raise ValueError("source_detection_confidence_scores length does not match source row count.")
    source_detection_class_ids_arr = (
        np.asarray(source_detection_class_ids, dtype=np.int32).reshape(-1)
        if source_detection_class_ids is not None
        else None
    )
    if source_detection_class_ids_arr is not None and source_detection_class_ids_arr.shape[0] != source_row_count:
        raise ValueError("source_detection_class_ids length does not match source row count.")
    source_detection_review_notes_arr = (
        np.asarray(source_detection_review_notes, dtype=object).reshape(-1)
        if source_detection_review_notes is not None
        else None
    )
    if source_detection_review_notes_arr is not None and source_detection_review_notes_arr.shape[0] != source_row_count:
        raise ValueError("source_detection_review_notes length does not match source row count.")

    instance_row_id_by_source: dict[int, int] = {}
    for row_id, source_row_index in zip(
        instance_refined_row_ids_arr.tolist(),
        instance_source_detect_row_index_arr.tolist(),
    ):
        if int(source_row_index) >= 0 and int(source_row_index) not in instance_row_id_by_source:
            instance_row_id_by_source[int(source_row_index)] = int(row_id)

    normalized_decision_labels = np.asarray(source_detection_decision_labels_arr, dtype=object).copy()
    normalized_reason_labels = np.asarray(source_detection_reason_labels_arr, dtype=object).copy()
    resolved_refined_row_id = np.full(source_row_count, -1, dtype=np.int64)
    for idx, source_row_index in enumerate(source_detection_source_row_arr.tolist()):
        mapped_row_id = instance_row_id_by_source.get(int(source_row_index))
        if mapped_row_id is not None:
            normalized_decision_labels[idx] = "accepted"
            resolved_refined_row_id[idx] = int(mapped_row_id)
            if not str(normalized_reason_labels[idx]).strip():
                normalized_reason_labels[idx] = "accepted"
            continue
        if str(normalized_decision_labels[idx]) == "accepted":
            normalized_decision_labels[idx] = "filtered"
        if not str(normalized_reason_labels[idx]).strip():
            normalized_reason_labels[idx] = str(normalized_decision_labels[idx]) or "filtered"

    decision_codes = np.asarray(
        [REFINED_SOURCE_DETECTION_DECISION_CODE_MAP[str(label)] for label in normalized_decision_labels.tolist()],
        dtype=np.int8,
    )

    _write_sparse_instances_arrays(
        refined_run,
        width=int(width),
        height=int(height),
        total_frames=total_frames,
        refined_row_ids=instance_refined_row_ids_arr,
        frame_indices=instance_frame_indices_arr,
        bbox_norm_coords=instance_bbox_norm_arr,
        source_kind_codes=instance_source_kind_codes,
        manual_edit_flags=instance_manual_edit_flags_arr,
        source_detect_row_index=instance_source_detect_row_index_arr,
        reason_labels=instance_reason_labels_arr,
        confidence_scores=instance_confidence_scores_arr,
        class_ids=instance_class_ids_arr,
        review_notes=instance_review_notes_arr,
    )
    _write_source_detections_arrays(
        refined_run,
        width=int(width),
        height=int(height),
        source_detect_row_index=source_detection_source_row_arr,
        frame_indices=source_detection_frame_indices_arr,
        bbox_norm_coords=source_detection_bbox_norm_arr,
        decision_codes=decision_codes,
        resolved_refined_row_id=resolved_refined_row_id,
        reason_labels=normalized_reason_labels,
        confidence_scores=source_detection_confidence_scores_arr,
        class_ids=source_detection_class_ids_arr,
        review_notes=source_detection_review_notes_arr,
    )

    refined_run.attrs["refined_storage_semantics"] = "sparse_instances_v1"
    refined_run.attrs["curated_primary_surface"] = "instances"
    refined_run.attrs["source_detection_decision_code_map"] = dict(REFINED_SOURCE_DETECTION_DECISION_CODE_MAP)
    refined_run.attrs["source_kind_code_map"] = dict(REFINED_SOURCE_KIND_CODE_MAP)
    source_context_clean = clean_mapping(dict(source_context or {}))
    if source_context_clean:
        refined_run.attrs["curated_source_context"] = source_context_clean

    _delete_legacy_dense_curated_root(refined_run)
    _refresh_curated_refined_detect_metadata(
        refined_run,
        resolved_refined_run_name=resolved_refined_run_name,
        zarr_path=zarr_path,
        command=command,
        env_info=env_info,
        source_context=source_context,
    )
    return {
        "status": "ok",
        "zarr_path": str(zarr_path) if zarr_path is not None else None,
        "refined_detect_run": resolved_refined_run_name,
        "rows_instances": int(instance_row_count),
        "rows_source_detections": int(source_row_count),
        "rows_present": int(instance_row_count),
    }


def _load_single_slot_curated_edit_payload(
    root: zarr.Group,
    refined_run: zarr.Group,
) -> Dict[str, np.ndarray]:
    if has_sparse_curated_refined_detect_instances_arrays(refined_run):
        total_frames = _resolved_total_frames(root, refined_run)
        frame_indices = np.arange(int(total_frames), dtype=np.int32)
        entity_ids = np.zeros(int(total_frames), dtype=np.int32)
        bbox_norm = np.full((int(total_frames), 4), np.nan, dtype=np.float64)
        confidence_scores = np.full(int(total_frames), np.nan, dtype=np.float32)
        class_ids = np.full(int(total_frames), -1, dtype=np.int32)
        status_labels = np.full(int(total_frames), "missing", dtype=object)
        source_kind_labels = np.full(int(total_frames), "none", dtype=object)
        manual_edit_flags = np.zeros(int(total_frames), dtype=bool)
        reason_labels = np.full(int(total_frames), "missing_detection", dtype=object)
        source_detect_row_index = np.full(int(total_frames), -1, dtype=np.int32)
        review_notes = np.full(int(total_frames), "", dtype=object)

        if has_curated_refined_source_detections_projection(refined_run):
            source_detections = _get_child_group_if_present(refined_run, "source_detections")
            if source_detections is not None:
                src_frames = np.asarray(source_detections["frame_indices"][:], dtype=np.int32).reshape(-1)
                src_source_rows = np.asarray(
                    source_detections["source_detect_row_index"][:],
                    dtype=np.int32,
                ).reshape(-1)
                src_bbox = np.asarray(source_detections["bbox_norm_coords"][:], dtype=np.float64).reshape(-1, 4)
                src_decision_codes = np.asarray(source_detections["decision_codes"][:], dtype=np.int8).reshape(-1)
                src_decision_labels = _decision_labels_from_codes(src_decision_codes)
                src_reason_labels = read_reason_labels(source_detections)
                if src_reason_labels is None:
                    src_reason_labels = np.asarray(src_decision_labels, dtype=object)
                src_confidence_scores = (
                    np.asarray(source_detections["confidence_scores"][:], dtype=np.float32).reshape(-1)
                    if "confidence_scores" in source_detections
                    else np.full(src_frames.shape[0], np.nan, dtype=np.float32)
                )
                src_class_ids = (
                    np.asarray(source_detections["class_ids"][:], dtype=np.int32).reshape(-1)
                    if "class_ids" in source_detections
                    else np.full(src_frames.shape[0], -1, dtype=np.int32)
                )
                src_review_notes = (
                    np.asarray(source_detections["review_notes"][:], dtype=object).reshape(-1)
                    if "review_notes" in source_detections
                    else np.full(src_frames.shape[0], "", dtype=object)
                )
                rows_by_frame: dict[int, list[int]] = {}
                for idx, frame in enumerate(src_frames.tolist()):
                    rows_by_frame.setdefault(int(frame), []).append(int(idx))
                for frame, row_indices in rows_by_frame.items():
                    if frame < 0 or frame >= int(total_frames):
                        continue
                    if len(row_indices) > 1:
                        raise RuntimeError(
                            "Single-slot refined detect updates do not support multiple source detections "
                            f"for frame {frame}."
                        )
                    row_idx = int(row_indices[0])
                    decision_label = str(src_decision_labels[row_idx])
                    source_detect_row_index[frame] = int(src_source_rows[row_idx])
                    confidence_scores[frame] = np.float32(src_confidence_scores[row_idx])
                    class_ids[frame] = np.int32(src_class_ids[row_idx])
                    review_notes[frame] = str(np.asarray(src_review_notes, dtype=object)[row_idx] or "")
                    if decision_label == "manual_clear":
                        status_labels[frame] = "filtered_out"
                        manual_edit_flags[frame] = True
                        reason_labels[frame] = str(np.asarray(src_reason_labels, dtype=object)[row_idx]) or "manual_clear"
                    elif decision_label in {"filtered", "duplicate"}:
                        status_labels[frame] = "filtered_out"
                        source_kind_labels[frame] = "raw_detect"
                        reason_labels[frame] = str(np.asarray(src_reason_labels, dtype=object)[row_idx]) or decision_label
                    bbox_norm[frame] = np.asarray(src_bbox[row_idx], dtype=np.float64)

        instances = _get_child_group_if_present(refined_run, "instances")
        if instances is None:
            raise ValueError("Curated refined detect run is missing instances subgroup.")
        inst_frames = np.asarray(instances["frame_indices"][:], dtype=np.int32).reshape(-1)
        if inst_frames.size and np.any(np.diff(np.sort(inst_frames)) == 0):
            raise RuntimeError(
                "Single-slot refined detect updates do not support multi-instance sparse runs."
            )
        inst_bbox = np.asarray(instances["bbox_norm_coords"][:], dtype=np.float64).reshape(-1, 4)
        inst_source_kind = _source_labels_from_codes(
            np.asarray(instances["source_kind_codes"][:], dtype=np.int8).reshape(-1)
        )
        inst_source_rows = np.asarray(instances["source_detect_row_index"][:], dtype=np.int32).reshape(-1)
        inst_manual_flags = np.asarray(instances["manual_edit_flags"][:], dtype=bool).reshape(-1)
        inst_reason_labels = read_reason_labels(instances)
        if inst_reason_labels is None:
            inst_reason_labels = np.full(inst_frames.shape[0], "", dtype=object)
        inst_confidence_scores = (
            np.asarray(instances["confidence_scores"][:], dtype=np.float32).reshape(-1)
            if "confidence_scores" in instances
            else np.full(inst_frames.shape[0], np.nan, dtype=np.float32)
        )
        inst_class_ids = (
            np.asarray(instances["class_ids"][:], dtype=np.int32).reshape(-1)
            if "class_ids" in instances
            else np.full(inst_frames.shape[0], -1, dtype=np.int32)
        )
        inst_review_notes = (
            np.asarray(instances["review_notes"][:], dtype=object).reshape(-1)
            if "review_notes" in instances
            else np.full(inst_frames.shape[0], "", dtype=object)
        )
        for frame, row_bbox, row_source_kind, row_source_idx, row_manual_flag, row_reason, row_score, row_class, row_review_note in zip(
            inst_frames.tolist(),
            inst_bbox.tolist(),
            inst_source_kind.tolist(),
            inst_source_rows.tolist(),
            inst_manual_flags.tolist(),
            np.asarray(inst_reason_labels, dtype=object).tolist(),
            inst_confidence_scores.tolist(),
            inst_class_ids.tolist(),
            np.asarray(inst_review_notes, dtype=object).tolist(),
        ):
            frame_idx = int(frame)
            if frame_idx < 0 or frame_idx >= int(total_frames):
                continue
            bbox_norm[frame_idx] = np.asarray(row_bbox, dtype=np.float64)
            confidence_scores[frame_idx] = np.float32(row_score)
            class_ids[frame_idx] = np.int32(row_class)
            status_labels[frame_idx] = "present"
            source_kind_labels[frame_idx] = str(row_source_kind)
            manual_edit_flags[frame_idx] = bool(row_manual_flag)
            reason_labels[frame_idx] = str(row_reason) or "present"
            source_detect_row_index[frame_idx] = int(row_source_idx)
            review_notes[frame_idx] = str(row_review_note or "")

        return {
            "frame_indices": frame_indices,
            "entity_ids": entity_ids,
            "bbox_norm_coords": bbox_norm,
            "status_labels": status_labels,
            "source_kind_labels": source_kind_labels,
            "reason_labels": reason_labels,
            "source_detect_row_index": source_detect_row_index,
            "manual_edit_flags": manual_edit_flags,
            "detection_source": _build_detection_source_from_source_kind_codes(
                np.asarray(
                    [REFINED_SOURCE_KIND_CODE_MAP[str(label)] for label in source_kind_labels.tolist()],
                    dtype=np.int8,
                )
            ),
            "confidence_scores": confidence_scores,
            "class_ids": class_ids,
            "review_notes": review_notes,
        }

    if not has_curated_refined_detect_arrays(refined_run):
        raise ValueError("Refined detect run does not have a readable curated detect surface.")

    row_count = int(refined_run["frame_indices"].shape[0])
    payload: Dict[str, np.ndarray] = {
        "frame_indices": np.asarray(refined_run["frame_indices"][:], dtype=np.int32).reshape(-1),
        "entity_ids": np.asarray(refined_run["entity_ids"][:], dtype=np.int32).reshape(-1),
        "bbox_norm_coords": np.asarray(refined_run["bbox_norm_coords"][:], dtype=np.float64).reshape(-1, 4),
        "status_labels": _status_labels_from_codes(
            np.asarray(refined_run["status_codes"][:], dtype=np.int8).reshape(-1)
        ),
        "source_kind_labels": _source_labels_from_codes(
            np.asarray(refined_run["source_kind_codes"][:], dtype=np.int8).reshape(-1)
        ),
        "source_detect_row_index": (
            np.asarray(refined_run["source_detect_row_index"][:], dtype=np.int32).reshape(-1)
            if "source_detect_row_index" in refined_run
            else np.full(row_count, -1, dtype=np.int32)
        ),
        "manual_edit_flags": (
            np.asarray(refined_run["manual_edit_flags"][:], dtype=bool).reshape(-1)
            if "manual_edit_flags" in refined_run
            else np.zeros(row_count, dtype=bool)
        ),
        "detection_source": (
            np.asarray(refined_run["detection_source"][:], dtype=np.int8).reshape(-1)
            if "detection_source" in refined_run
            else np.zeros(row_count, dtype=np.int8)
        ),
        "confidence_scores": (
            np.asarray(refined_run["confidence_scores"][:], dtype=np.float32).reshape(-1)
            if "confidence_scores" in refined_run
            else np.full(row_count, np.nan, dtype=np.float32)
        ),
        "class_ids": (
            np.asarray(refined_run["class_ids"][:], dtype=np.int32).reshape(-1)
            if "class_ids" in refined_run
            else np.full(row_count, -1, dtype=np.int32)
        ),
        "review_notes": (
            np.asarray(refined_run["review_notes"][:], dtype=object).reshape(-1)
            if "review_notes" in refined_run
            else np.full(row_count, "", dtype=object)
        ),
    }
    reason_labels = read_reason_labels(refined_run)
    payload["reason_labels"] = (
        np.asarray(reason_labels, dtype=object).reshape(-1)
        if reason_labels is not None
        else np.full(row_count, "", dtype=object)
    )
    return payload

def update_curated_refined_detect_rows(
    root: zarr.Group,
    *,
    zarr_path: Optional[Path] = None,
    refined_run_name: Optional[str] = None,
    row_indices: np.ndarray,
    bbox_norm_coords: Optional[np.ndarray] = None,
    status_labels: Optional[Sequence[str]] = None,
    source_kind_labels: Optional[Sequence[str]] = None,
    reason_labels: Optional[Sequence[str]] = None,
    source_detect_row_index: Optional[np.ndarray] = None,
    manual_edit_flags: Optional[np.ndarray] = None,
    detection_source: Optional[np.ndarray] = None,
    confidence_scores: Optional[np.ndarray] = None,
    class_ids: Optional[np.ndarray] = None,
    command: Optional[str] = None,
    env_info: Optional[Mapping[str, Any]] = None,
    source_context: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    refined_parent, _ = _resolve_refined_parent(root)
    resolved_refined_run_name = normalize_attr(refined_run_name) or normalize_attr(refined_parent.attrs.get("latest"))
    if not resolved_refined_run_name or resolved_refined_run_name not in refined_parent:
        raise ValueError("No refined detect run available.")
    refined_run = _open_named_child_group(refined_parent, resolved_refined_run_name, mode="a")
    payload = _load_single_slot_curated_edit_payload(root, refined_run)
    row_count = int(np.asarray(payload["frame_indices"], dtype=np.int32).shape[0])
    row_indices_arr = _normalize_row_indices(row_indices, row_count)

    frame_indices_arr = np.asarray(payload["frame_indices"], dtype=np.int32).reshape(-1)
    entity_ids_arr = np.asarray(payload["entity_ids"], dtype=np.int32).reshape(-1)
    bbox_norm_arr = np.asarray(payload["bbox_norm_coords"], dtype=np.float64).reshape(-1, 4).copy()
    status_labels_arr = np.asarray(payload["status_labels"], dtype=object).reshape(-1).copy()
    source_kind_labels_arr = np.asarray(payload["source_kind_labels"], dtype=object).reshape(-1).copy()
    reason_labels_arr = np.asarray(payload["reason_labels"], dtype=object).reshape(-1).copy()
    source_detect_row_index_arr = np.asarray(payload["source_detect_row_index"], dtype=np.int32).reshape(-1).copy()
    manual_edit_flags_arr = np.asarray(payload["manual_edit_flags"], dtype=bool).reshape(-1).copy()
    detection_source_arr = np.asarray(payload["detection_source"], dtype=np.int8).reshape(-1).copy()
    confidence_scores_arr = np.asarray(payload["confidence_scores"], dtype=np.float32).reshape(-1).copy()
    class_ids_arr = np.asarray(payload["class_ids"], dtype=np.int32).reshape(-1).copy()
    review_notes_arr = np.asarray(payload["review_notes"], dtype=object).reshape(-1).copy()

    if bbox_norm_coords is not None:
        values = np.asarray(bbox_norm_coords, dtype=np.float64).reshape(-1, 4)
        if values.shape[0] != row_indices_arr.shape[0]:
            raise ValueError("bbox_norm_coords length does not match row_indices length.")
        bbox_norm_arr[row_indices_arr] = values
    if status_labels is not None:
        values = np.asarray(status_labels, dtype=object).reshape(-1)
        if values.shape[0] != row_indices_arr.shape[0]:
            raise ValueError("status_labels length does not match row_indices length.")
        status_labels_arr[row_indices_arr] = values
    if source_kind_labels is not None:
        values = np.asarray(source_kind_labels, dtype=object).reshape(-1)
        if values.shape[0] != row_indices_arr.shape[0]:
            raise ValueError("source_kind_labels length does not match row_indices length.")
        source_kind_labels_arr[row_indices_arr] = values
    if reason_labels is not None:
        values = np.asarray(reason_labels, dtype=object).reshape(-1)
        if values.shape[0] != row_indices_arr.shape[0]:
            raise ValueError("reason_labels length does not match row_indices length.")
        reason_labels_arr[row_indices_arr] = values
    if source_detect_row_index is not None:
        values = _normalize_source_row_index(source_detect_row_index, int(row_indices_arr.shape[0]))
        source_detect_row_index_arr[row_indices_arr] = values
    if manual_edit_flags is not None:
        values = np.asarray(manual_edit_flags, dtype=bool).reshape(-1)
        if values.shape[0] != row_indices_arr.shape[0]:
            raise ValueError("manual_edit_flags length does not match row_indices length.")
        manual_edit_flags_arr[row_indices_arr] = values
    if detection_source is not None:
        values = np.asarray(detection_source, dtype=np.int8).reshape(-1)
        if values.shape[0] != row_indices_arr.shape[0]:
            raise ValueError("detection_source length does not match row_indices length.")
        detection_source_arr[row_indices_arr] = values
    if confidence_scores is not None:
        values = np.asarray(confidence_scores, dtype=np.float32).reshape(-1)
        if values.shape[0] != row_indices_arr.shape[0]:
            raise ValueError("confidence_scores length does not match row_indices length.")
        confidence_scores_arr[row_indices_arr] = values
    if class_ids is not None:
        values = np.asarray(class_ids, dtype=np.int32).reshape(-1)
        if values.shape[0] != row_indices_arr.shape[0]:
            raise ValueError("class_ids length does not match row_indices length.")
        class_ids_arr[row_indices_arr] = values

    payload = write_curated_refined_detect_root(
        root,
        zarr_path=zarr_path,
        refined_run_name=resolved_refined_run_name,
        frame_indices=frame_indices_arr,
        entity_ids=entity_ids_arr,
        bbox_norm_coords=bbox_norm_arr,
        status_labels=status_labels_arr,
        source_kind_labels=source_kind_labels_arr,
        reason_labels=reason_labels_arr,
        source_detect_row_index=source_detect_row_index_arr,
        manual_edit_flags=manual_edit_flags_arr,
        detection_source=detection_source_arr,
        confidence_scores=confidence_scores_arr,
        class_ids=class_ids_arr,
        review_notes=review_notes_arr,
        command=command,
        env_info=env_info,
        source_context=source_context,
    )
    payload["rows_updated"] = int(row_indices_arr.shape[0])
    return payload


def write_curated_refined_detect_root(
    root: zarr.Group,
    *,
    zarr_path: Optional[Path] = None,
    refined_run_name: Optional[str] = None,
    frame_indices: np.ndarray,
    entity_ids: np.ndarray,
    bbox_norm_coords: np.ndarray,
    status_labels: Sequence[str],
    source_kind_labels: Sequence[str],
    reason_labels: Sequence[str],
    source_detect_row_index: np.ndarray,
    manual_edit_flags: Optional[np.ndarray] = None,
    detection_source: Optional[np.ndarray] = None,
    confidence_scores: Optional[np.ndarray] = None,
    class_ids: Optional[np.ndarray] = None,
    review_notes: Optional[Sequence[str]] = None,
    command: Optional[str] = None,
    env_info: Optional[Mapping[str, Any]] = None,
    source_context: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    refined_parent, _ = _resolve_refined_parent(root)
    resolved_refined_run_name = normalize_attr(refined_run_name) or normalize_attr(refined_parent.attrs.get("latest"))
    if not resolved_refined_run_name or resolved_refined_run_name not in refined_parent:
        raise ValueError("No refined detect run available.")
    refined_run = _open_named_child_group(refined_parent, resolved_refined_run_name, mode="a")

    frame_indices_arr = np.asarray(frame_indices, dtype=np.int32).reshape(-1)
    entity_ids_arr = np.asarray(entity_ids, dtype=np.int32).reshape(-1)
    bbox_norm_arr = np.asarray(bbox_norm_coords, dtype=np.float64).reshape(-1, 4)
    if frame_indices_arr.shape[0] != entity_ids_arr.shape[0] or frame_indices_arr.shape[0] != bbox_norm_arr.shape[0]:
        raise ValueError("Dense refined detect arrays must agree on row count.")

    status_codes = np.asarray(
        [REFINED_DETECT_STATUS_CODE_MAP[str(label)] for label in status_labels],
        dtype=np.int8,
    )
    source_kind_codes = np.asarray(
        [REFINED_SOURCE_KIND_CODE_MAP[str(label)] for label in source_kind_labels],
        dtype=np.int8,
    )
    if status_codes.shape[0] != frame_indices_arr.shape[0] or source_kind_codes.shape[0] != frame_indices_arr.shape[0]:
        raise ValueError("status/source arrays must match frame_indices length.")

    source_detect_row_index_arr = _normalize_source_row_index(
        source_detect_row_index,
        int(frame_indices_arr.shape[0]),
    )
    manual_edit_flags_arr = (
        np.asarray(manual_edit_flags, dtype=bool).reshape(-1)
        if manual_edit_flags is not None
        else np.zeros(frame_indices_arr.shape[0], dtype=bool)
    )
    if manual_edit_flags_arr.shape[0] != frame_indices_arr.shape[0]:
        raise ValueError("manual_edit_flags length does not match frame_indices length.")
    reason_labels_arr = np.asarray(reason_labels, dtype=object).reshape(-1)
    if reason_labels_arr.shape[0] != frame_indices_arr.shape[0]:
        raise ValueError("reason_labels length does not match frame_indices length.")
    confidence_scores_arr = (
        np.asarray(confidence_scores, dtype=np.float32).reshape(-1)
        if confidence_scores is not None
        else None
    )
    if confidence_scores_arr is not None and confidence_scores_arr.shape[0] != frame_indices_arr.shape[0]:
        raise ValueError("confidence_scores length does not match frame_indices length.")
    class_ids_arr = (
        np.asarray(class_ids, dtype=np.int32).reshape(-1)
        if class_ids is not None
        else None
    )
    if class_ids_arr is not None and class_ids_arr.shape[0] != frame_indices_arr.shape[0]:
        raise ValueError("class_ids length does not match frame_indices length.")
    review_notes_arr = (
        np.asarray(review_notes, dtype=object).reshape(-1)
        if review_notes is not None
        else None
    )
    if review_notes_arr is not None and review_notes_arr.shape[0] != frame_indices_arr.shape[0]:
        raise ValueError("review_notes length does not match frame_indices length.")

    present_mask = (
        status_codes == REFINED_DETECT_STATUS_CODE_MAP["present"]
    ) & np.all(np.isfinite(bbox_norm_arr), axis=1)
    instance_frame_indices = frame_indices_arr[present_mask]
    instance_bbox_norm_coords = bbox_norm_arr[present_mask]
    instance_source_kind_labels = np.asarray(source_kind_labels, dtype=object).reshape(-1)[present_mask]
    instance_reason_labels = reason_labels_arr[present_mask]
    instance_source_detect_row_index = source_detect_row_index_arr[present_mask]
    instance_manual_edit_flags = manual_edit_flags_arr[present_mask]
    instance_confidence_scores = (
        confidence_scores_arr[present_mask] if confidence_scores_arr is not None else None
    )
    instance_class_ids = class_ids_arr[present_mask] if class_ids_arr is not None else None
    instance_review_notes = review_notes_arr[present_mask] if review_notes_arr is not None else None

    detect_group, _ = _resolve_bound_source_detect_group(root, refined_run)
    if detect_group is not None:
        source_detection_frame_indices = np.asarray(detect_group["frame_indices"][:], dtype=np.int32).reshape(-1)
        source_detection_bbox_norm_coords = np.asarray(detect_group["bbox_norm_coords"][:], dtype=np.float64).reshape(-1, 4)
        source_detection_source_detect_row_index = np.arange(source_detection_frame_indices.shape[0], dtype=np.int32)
        source_detection_decision_labels = np.full(source_detection_frame_indices.shape[0], "filtered", dtype=object)
        source_detection_reason_labels = np.full(source_detection_frame_indices.shape[0], "filtered", dtype=object)
        for status_label, source_row_index_value, manual_flag, reason_label in zip(
            np.asarray(status_labels, dtype=object).reshape(-1).tolist(),
            source_detect_row_index_arr.tolist(),
            manual_edit_flags_arr.tolist(),
            reason_labels_arr.tolist(),
        ):
            raw_idx = int(source_row_index_value)
            if raw_idx < 0 or raw_idx >= int(source_detection_source_detect_row_index.shape[0]):
                continue
            if str(status_label) == "present":
                source_detection_decision_labels[raw_idx] = "accepted"
                source_detection_reason_labels[raw_idx] = str(reason_label) or "accepted"
            elif str(status_label) == "filtered_out" and bool(manual_flag):
                source_detection_decision_labels[raw_idx] = "manual_clear"
                source_detection_reason_labels[raw_idx] = str(reason_label) or "manual_clear"
            elif str(status_label) == "filtered_out":
                source_detection_decision_labels[raw_idx] = "filtered"
                source_detection_reason_labels[raw_idx] = str(reason_label) or "filtered"
        source_detection_confidence_scores = (
            np.asarray(detect_group["scores"][:], dtype=np.float32).reshape(-1)
            if "scores" in detect_group
            else None
        )
        source_detection_class_ids = (
            np.asarray(detect_group["class_ids"][:], dtype=np.int32).reshape(-1)
            if "class_ids" in detect_group
            else None
        )
    else:
        source_detection_frame_indices = np.empty((0,), dtype=np.int32)
        source_detection_bbox_norm_coords = np.empty((0, 4), dtype=np.float64)
        source_detection_source_detect_row_index = np.empty((0,), dtype=np.int32)
        source_detection_decision_labels = np.empty((0,), dtype=object)
        source_detection_reason_labels = np.empty((0,), dtype=object)
        source_detection_confidence_scores = None
        source_detection_class_ids = None

    payload = write_curated_refined_detect_surfaces(
        root,
        zarr_path=zarr_path,
        refined_run_name=resolved_refined_run_name,
        instance_frame_indices=instance_frame_indices,
        instance_bbox_norm_coords=instance_bbox_norm_coords,
        instance_source_kind_labels=instance_source_kind_labels,
        instance_reason_labels=instance_reason_labels,
        instance_source_detect_row_index=instance_source_detect_row_index,
        instance_manual_edit_flags=instance_manual_edit_flags,
        instance_confidence_scores=instance_confidence_scores,
        instance_class_ids=instance_class_ids,
        instance_review_notes=instance_review_notes,
        source_detection_source_detect_row_index=source_detection_source_detect_row_index,
        source_detection_frame_indices=source_detection_frame_indices,
        source_detection_bbox_norm_coords=source_detection_bbox_norm_coords,
        source_detection_decision_labels=source_detection_decision_labels,
        source_detection_reason_labels=source_detection_reason_labels,
        source_detection_confidence_scores=source_detection_confidence_scores,
        source_detection_class_ids=source_detection_class_ids,
        command=command,
        env_info=env_info,
        source_context=source_context,
    )
    payload["rows_materialized"] = int(frame_indices_arr.shape[0])
    return payload


def materialize_refined_detect_curation(
    root: zarr.Group,
    *,
    zarr_path: Optional[Path] = None,
    refined_run_name: Optional[str] = None,
    source_group: Optional[str] = None,
    command: Optional[str] = None,
    env_info: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    total_frames = as_int(root.attrs.get("total_frames"))
    if total_frames is None or total_frames <= 0:
        raise ValueError("Root attrs must include positive total_frames.")

    refined_parent, parent_name = _resolve_refined_parent(root)
    resolved_refined_run_name = normalize_attr(refined_run_name) or normalize_attr(refined_parent.attrs.get("latest"))
    if not resolved_refined_run_name or resolved_refined_run_name not in refined_parent:
        raise ValueError("No refined detect run available.")
    refined_run = _open_named_child_group(refined_parent, resolved_refined_run_name, mode="a")
    source_group_kind, actual_source_group, source_arrays, source_group_path = _resolve_source_group(
        root,
        refined_run,
        resolved_refined_run_name,
        source_group=source_group,
        parent_name=parent_name,
    )

    dense_frame_indices = np.arange(total_frames, dtype=np.int32)
    dense_entity_ids = np.zeros(total_frames, dtype=np.int32)
    dense_bbox = np.full((total_frames, 4), np.nan, dtype=np.float64)
    dense_scores = np.full(total_frames, np.nan, dtype=np.float32)
    dense_class_ids = np.full(total_frames, -1, dtype=np.int32)
    dense_reason = np.full(total_frames, "missing_detection", dtype=object)
    dense_status = np.full(total_frames, "missing", dtype=object)
    dense_source_kind = np.full(total_frames, "none", dtype=object)
    dense_source_row_index = np.full(total_frames, -1, dtype=np.int32)
    dense_detection_source = np.zeros(total_frames, dtype=np.int8)
    dense_manual_edit_flags = np.zeros(total_frames, dtype=bool)

    frame_indices = np.asarray(source_arrays["frame_indices"][:], dtype=np.int32).reshape(-1)
    bbox_norm = np.asarray(source_arrays["bbox_norm_coords"][:], dtype=np.float64).reshape(-1, 4)
    scores = (
        np.asarray(source_arrays["scores"][:], dtype=np.float32).reshape(-1)
        if "scores" in source_arrays
        else np.ones(frame_indices.shape[0], dtype=np.float32)
    )
    class_ids = (
        np.asarray(source_arrays["class_ids"][:], dtype=np.int32).reshape(-1)
        if "class_ids" in source_arrays
        else np.zeros(frame_indices.shape[0], dtype=np.int32)
    )
    reason_labels = read_reason_labels(source_arrays)
    if reason_labels is None:
        reason_labels = np.full(frame_indices.shape[0], source_group_kind, dtype=object)
    source_detection = (
        np.asarray(source_arrays["detection_source"][:], dtype=np.int8).reshape(-1)
        if "detection_source" in source_arrays
        else np.zeros(frame_indices.shape[0], dtype=np.int8)
    )

    frame_to_rows: dict[int, list[int]] = {}
    for idx, frame_index in enumerate(frame_indices.tolist()):
        frame_to_rows.setdefault(int(frame_index), []).append(int(idx))

    group_source_kind = {
        "raw": "raw_detect",
        "filtered": "raw_detect",
        "interpolated": "interpolated",
        "manual": "manual",
    }.get(source_group_kind, "none")

    for frame_index, row_indices in frame_to_rows.items():
        if len(row_indices) != 1:
            dense_status[frame_index] = "ambiguous"
            dense_reason[frame_index] = "multiple_candidates"
            continue
        row_idx = int(row_indices[0])
        dense_bbox[frame_index] = bbox_norm[row_idx]
        dense_scores[frame_index] = scores[row_idx]
        dense_class_ids[frame_index] = class_ids[row_idx]
        dense_reason[frame_index] = str(reason_labels[row_idx])
        dense_status[frame_index] = "present"
        dense_source_kind[frame_index] = group_source_kind
        dense_source_row_index[frame_index] = row_idx if source_group_kind == "raw" else -1
        dense_detection_source[frame_index] = int(source_detection[row_idx]) if source_group_kind == "interpolated" else 0
        dense_manual_edit_flags[frame_index] = source_group_kind == "manual"

    payload = write_curated_refined_detect_root(
        root,
        zarr_path=zarr_path,
        refined_run_name=resolved_refined_run_name,
        frame_indices=dense_frame_indices,
        entity_ids=dense_entity_ids,
        bbox_norm_coords=dense_bbox,
        status_labels=dense_status,
        source_kind_labels=dense_source_kind,
        reason_labels=dense_reason,
        source_detect_row_index=dense_source_row_index,
        manual_edit_flags=dense_manual_edit_flags,
        detection_source=dense_detection_source,
        confidence_scores=dense_scores,
        class_ids=dense_class_ids,
        command=command,
        env_info=env_info,
        source_context={
            "materialized_from_group": actual_source_group or "raw",
            "materialized_from_group_kind": source_group_kind,
            "materialized_from_path": source_group_path,
        },
    )
    payload["source_group"] = actual_source_group or "raw"
    payload["source_group_kind"] = source_group_kind
    return payload


__all__ = [
    "CURATED_REFINED_INSTANCES_REQUIRED_ARRAYS",
    "CURATED_REFINED_REQUIRED_ARRAYS",
    "CURATED_REFINED_SOURCE_DETECTIONS_REQUIRED_ARRAYS",
    "REFINED_ARTIFACT_STATE_CODE_MAP",
    "REFINED_DETECT_STATUS_CODE_MAP",
    "REFINED_REVIEW_STATE_CODE_MAP",
    "REFINED_SOURCE_DETECTION_DECISION_CODE_MAP",
    "REFINED_SOURCE_KIND_CODE_MAP",
    "assign_refined_row_ids",
    "build_curated_detection_source_array",
    "build_refined_detect_summary",
    "build_source_detection_decision_summary",
    "extract_present_curated_rows",
    "extract_source_detection_rows",
    "has_curated_refined_detect_arrays",
    "has_curated_refined_source_detections_projection",
    "has_curated_refined_detect_surface",
    "has_sparse_curated_refined_detect_instances_arrays",
    "materialize_refined_detect_curation",
    "present_curated_row_mask",
    "resolve_curated_refined_detect_run",
    "update_curated_refined_detect_rows",
    "write_curated_refined_detect_root",
    "write_curated_refined_detect_surfaces",
]
