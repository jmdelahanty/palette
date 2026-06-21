"""Eye/subject mask performance and quality row extractors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import zarr

from fisheye.shared.batch_logging import utc_now
from fisheye.shared.type_conversions import normalize_attr as _decode_attr
from fisheye.shared.zarr_run_completion import resolve_latest_complete_run_name


MAX_STORED_BYTE_ACCOUNTING_LOGICAL_BYTES = 1 << 30


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return None


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except Exception:
        return None


def _json_dumps(value: Any) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(value, sort_keys=True)


def _coerce_mapping(value: Any) -> Optional[Dict[str, Any]]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (bytes, bytearray)):
        text = value.decode("utf-8", "ignore").strip()
    elif isinstance(value, str):
        text = value.strip()
    else:
        return None
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except Exception:
        return None
    if isinstance(parsed, Mapping):
        return dict(parsed)
    return None


def _lookup_attr(attrs: Mapping[str, Any], provenance_inputs: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        value = attrs.get(name)
        if value is not None:
            return value
        value = provenance_inputs.get(name)
        if value is not None:
            return value
    return None


def _resolve_source_roi_pixel_contract(
    attrs: Mapping[str, Any],
    provenance_inputs: Mapping[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    contract = _coerce_mapping(_lookup_attr(attrs, provenance_inputs, "source_roi_pixel_contract"))
    contract_name = _decode_attr(_lookup_attr(attrs, provenance_inputs, "source_roi_pixel_contract_name"))
    if contract_name is None and contract is not None:
        contract_name = _decode_attr(contract.get("name"))
    return contract, contract_name


def _as_bool_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, np.integer)):
        return 1 if int(value) != 0 else 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return 1
        if normalized in {"0", "false", "no", "n", "off"}:
            return 0
    return None


def _open_child_group(parent: Any, key: str) -> Any:
    store = getattr(parent, "store", None)
    if store is None:
        return None
    parent_path = _decode_attr(getattr(parent, "path", None))
    child_path = f"{parent_path}/{key}" if parent_path else key
    try:
        return zarr.open_group(store=store, path=child_path, mode="r")
    except TypeError:
        try:
            return zarr.open_group(store, mode="r", path=child_path)
        except Exception:
            return None
    except Exception:
        return None


def _get_group(parent: Any, key: str) -> Any:
    child = None
    getter = getattr(parent, "get", None)
    if callable(getter):
        try:
            child = getter(key)
        except Exception:
            child = None
    if child is not None:
        return child
    try:
        child = parent[key]
    except Exception:
        child = None
    if child is not None:
        return child
    return _open_child_group(parent, key)


def _array_logical_bytes(array: Any) -> Optional[int]:
    if array is None:
        return None
    nbytes = getattr(array, "nbytes", None)
    if nbytes is not None:
        try:
            return int(nbytes() if callable(nbytes) else nbytes)
        except Exception:
            pass
    shape = getattr(array, "shape", None)
    dtype = getattr(array, "dtype", None)
    if shape is None or dtype is None:
        return None
    try:
        itemsize = int(np.dtype(dtype).itemsize)
        total = int(itemsize)
        for dim in shape:
            total *= int(dim)
        return int(total)
    except Exception:
        return None


def _array_stored_bytes(array: Any) -> Optional[int]:
    if array is None:
        return None
    logical_bytes = _array_logical_bytes(array)
    if (
        logical_bytes is not None
        and int(logical_bytes) > MAX_STORED_BYTE_ACCOUNTING_LOGICAL_BYTES
    ):
        # zarr.Array.nbytes_stored() can walk every physical chunk. For large
        # dense mask stores that turns registry refresh into a storage crawl and
        # blocks summary rows entirely. Logical bytes remain queryable; physical
        # byte accounting can be done by explicit diagnostics.
        return None
    for attr_name in ("nbytes_stored", "nbytes_stored_sync"):
        value = getattr(array, attr_name, None)
        if value is None:
            continue
        try:
            resolved = value() if callable(value) else value
        except Exception:
            continue
        if hasattr(resolved, "__await__"):
            continue
        try:
            return int(resolved)
        except Exception:
            continue
    return None


def _sum_optional_ints(values: Sequence[Optional[int]]) -> Optional[int]:
    present = [int(value) for value in values if value is not None]
    return int(sum(present)) if present else None


def _mask_rle_component_arrays(rle_group: Any) -> List[Any]:
    arrays: List[Any] = []
    components_group = _get_group(rle_group, "components") if rle_group is not None else None
    if components_group is None:
        return arrays
    for component_key in _run_names(components_group):
        component_group = _get_group(components_group, component_key)
        if component_group is None:
            continue
        for array_name in ("counts", "indptr", "present", "area_px", "bbox_xyxy"):
            array = _get_group(component_group, array_name)
            if array is not None:
                arrays.append(array)
    return arrays


def _format_ratio(numerator: Optional[int], denominator: Optional[int]) -> Optional[float]:
    if numerator is None or denominator is None:
        return None
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _run_names(parent: zarr.Group) -> List[str]:
    try:
        names = list(parent.group_keys())
    except Exception:
        names = [name for name in parent.keys() if isinstance(name, str)]
    return sorted(str(name) for name in names)


def _resolve_latest_group_name(parent: Optional[zarr.Group]) -> Optional[str]:
    if parent is None:
        return None
    return resolve_latest_complete_run_name(parent)


def _resolve_source_keypoints_run(attrs: Mapping[str, Any]) -> Optional[str]:
    value = _decode_attr(attrs.get("source_keypoints_run"))
    if value:
        return value
    return _decode_attr(attrs.get("source_keypoint_run"))


def _coerce_text_list(value: Any) -> List[str]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:
            pass
    if isinstance(value, (bytes, bytearray, str)):
        text = _decode_attr(value)
        return [text] if text else []
    if isinstance(value, Sequence):
        values: List[str] = []
        for item in value:
            text = _decode_attr(item)
            if text:
                values.append(text)
        return values
    return []


def _coerce_bool_list(value: Any) -> List[bool]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:
            pass
    if isinstance(value, np.ndarray):
        try:
            value = value.tolist()
        except Exception:
            return []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return [bool(item) for item in value]
    return []


def _extract_review_fields(
    review_status: Optional[Mapping[str, Any]],
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
    if not review_status:
        return None, None, None, None, None
    return (
        _decode_attr(review_status.get("state")) or _decode_attr(review_status.get("review_state")),
        _decode_attr(review_status.get("method")) or _decode_attr(review_status.get("review_method")),
        _decode_attr(review_status.get("intended_use")) or _decode_attr(review_status.get("review_intended_use")),
        _decode_attr(review_status.get("reviewer")) or _decode_attr(review_status.get("review_reviewer")),
        _decode_attr(review_status.get("timestamp_utc"))
        or _decode_attr(review_status.get("timestamp"))
        or _decode_attr(review_status.get("review_timestamp_utc"))
        or _decode_attr(review_status.get("reviewed_at_utc"))
        or _decode_attr(review_status.get("reviewed_at")),
    )


def _derive_review_lifecycle(
    *,
    review_state: Optional[str],
    stale_state: Optional[str] = None,
    stale_reason: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    stale_state_norm = str(stale_state).strip().lower() if stale_state else None
    if stale_state_norm == "stale":
        return "stale", stale_reason or "source_subject_mask_stale"
    review_state_norm = str(review_state).strip().lower() if review_state else None
    if review_state_norm in {"pending", "needs_review", "review"}:
        return "in_progress", review_state
    if review_state_norm in {"approved", "rejected"}:
        return review_state_norm, review_state
    return None, None


def _subject_component_family(component_name: Optional[str]) -> Optional[str]:
    name = _decode_attr(component_name)
    if not name:
        return None
    if name in {"eye_left", "eye_right", "eyes_union"}:
        return "eyes"
    return name


def _extract_subject_mask_total_rois(run_group: zarr.Group, attrs: Mapping[str, Any]) -> Optional[int]:
    total_rois = _as_int(attrs.get("total_rois"))
    if total_rois is not None:
        return total_rois
    for key in ("masks_roi", "frame_indices", "detection_indices"):
        arr = _get_group(run_group, key)
        shape = getattr(arr, "shape", None)
        if shape:
            try:
                return int(shape[0])
            except Exception:
                pass
    frame_counts = _get_group(run_group, "frame_counts")
    if frame_counts is not None:
        try:
            values = np.asarray(frame_counts[:])
            return int(values.sum())
        except Exception:
            return None
    rle_group = _get_group(run_group, "mask_rle")
    components_group = _get_group(rle_group, "components") if rle_group is not None else None
    if components_group is not None:
        try:
            for component_key in _run_names(components_group):
                component_group = _get_group(components_group, component_key)
                indptr = _get_group(component_group, "indptr") if component_group is not None else None
                if indptr is not None and getattr(indptr, "shape", None):
                    return max(0, int(indptr.shape[0]) - 1)
        except Exception:
            return None
    return None


def _extract_subject_mask_presence(
    run_group: zarr.Group,
    *,
    mask_labels: Sequence[str],
    total_rois: Optional[int],
) -> Tuple[Optional[int], Dict[str, int]]:
    counts_by_label = {str(label): 0 for label in mask_labels}
    any_present_rows: Optional[int] = None

    metrics = _get_group(run_group, "metrics")
    mask_present = _get_group(metrics, "mask_present") if metrics is not None else None
    if mask_present is not None:
        try:
            values = np.asarray(mask_present[:], dtype=bool)
            if values.ndim == 2 and values.shape[1] >= len(mask_labels):
                any_present_rows = int(values[:, : len(mask_labels)].any(axis=1).sum())
                for index, label in enumerate(mask_labels):
                    counts_by_label[str(label)] = int(values[:, index].sum())
                return any_present_rows, counts_by_label
        except Exception:
            pass

    masks_roi = _get_group(run_group, "masks_roi")
    if masks_roi is not None:
        try:
            values = np.asarray(masks_roi[:])
            if values.ndim == 4 and values.shape[1] >= len(mask_labels):
                present = values[:, : len(mask_labels), :, :].reshape(values.shape[0], len(mask_labels), -1).any(axis=2)
                any_present_rows = int(present.any(axis=1).sum())
                for index, label in enumerate(mask_labels):
                    counts_by_label[str(label)] = int(present[:, index].sum())
                return any_present_rows, counts_by_label
        except Exception:
            pass

    rle_group = _get_group(run_group, "mask_rle")
    components_group = _get_group(rle_group, "components") if rle_group is not None else None
    if components_group is not None:
        try:
            union_present: Optional[np.ndarray] = None
            for component_key in _run_names(components_group):
                component_group = _get_group(components_group, component_key)
                if component_group is None:
                    continue
                component_attrs = dict(getattr(component_group, "attrs", {}) or {})
                component_name = _decode_attr(component_attrs.get("component_name"))
                if not component_name:
                    raw_index = component_attrs.get("component_index")
                    index = _as_int(raw_index)
                    if index is None:
                        prefix = str(component_key).split("_", 1)[0]
                        index = int(prefix) if prefix.isdigit() else None
                    if index is not None and 0 <= index < len(mask_labels):
                        component_name = str(mask_labels[index])
                if component_name not in counts_by_label:
                    continue
                present = _get_group(component_group, "present")
                if present is None:
                    continue
                values = np.asarray(present[:], dtype=bool).reshape(-1)
                counts_by_label[str(component_name)] = int(np.count_nonzero(values))
                union_present = values.copy() if union_present is None else (union_present | values)
            if counts_by_label:
                any_present_rows = int(np.count_nonzero(union_present)) if union_present is not None else None
                return any_present_rows, counts_by_label
        except Exception:
            pass

    if total_rois is not None and total_rois >= 0:
        return 0, counts_by_label
    return None, counts_by_label


def _extract_mask_storage_fields(run_group: zarr.Group, attrs: Mapping[str, Any]) -> Dict[str, Any]:
    masks_roi = _get_group(run_group, "masks_roi")
    bitpacked_group = _get_group(run_group, "mask_bitpacked")
    rle_group = _get_group(run_group, "mask_rle")
    rle_attrs = dict(getattr(rle_group, "attrs", {}) or {}) if rle_group is not None else {}
    rle_arrays = _mask_rle_component_arrays(rle_group)

    store_encodings = _coerce_text_list(attrs.get("mask_store_encodings"))
    if not store_encodings:
        if masks_roi is not None:
            store_encodings.append("dense_uint8")
        if bitpacked_group is not None:
            store_encodings.append("bitpacked_binary_v1")
        if rle_group is not None:
            store_encodings.append("component_rle_v1")

    storage_encoding = _decode_attr(attrs.get("mask_storage_encoding"))
    if not storage_encoding and store_encodings:
        storage_encoding = "+".join(store_encodings)

    masks_roi_materialized = _as_bool_int(attrs.get("masks_roi_materialized"))
    if masks_roi_materialized is None:
        masks_roi_materialized = 1 if masks_roi is not None else 0
    mask_rle_materialized = _as_bool_int(attrs.get("mask_rle_materialized"))
    if mask_rle_materialized is None:
        mask_rle_materialized = 1 if rle_group is not None else 0
    stale_components = _coerce_text_list(attrs.get("mask_rle_stale_component_names"))

    return {
        "mask_storage_encoding": storage_encoding,
        "mask_store_encodings_json": _json_dumps(store_encodings) if store_encodings else None,
        "masks_roi_materialized": masks_roi_materialized,
        "mask_rle_materialized": mask_rle_materialized,
        "mask_rle_schema_id": _decode_attr(attrs.get("mask_rle_schema_id")) or _decode_attr(rle_attrs.get("schema_id")),
        "mask_rle_encoding": _decode_attr(attrs.get("mask_rle_encoding")) or _decode_attr(rle_attrs.get("mask_encoding")),
        "mask_rle_layout": _decode_attr(attrs.get("mask_rle_layout")) or _decode_attr(rle_attrs.get("layout")),
        "masks_roi_logical_bytes": _array_logical_bytes(masks_roi),
        "masks_roi_stored_bytes": _array_stored_bytes(masks_roi),
        "masks_roi_materialized_from": _decode_attr(attrs.get("masks_roi_materialized_from")),
        "masks_roi_materialized_at_utc": _decode_attr(attrs.get("masks_roi_materialized_at_utc")),
        "mask_rle_logical_bytes": _sum_optional_ints([_array_logical_bytes(array) for array in rle_arrays]),
        "mask_rle_stored_bytes": _sum_optional_ints([_array_stored_bytes(array) for array in rle_arrays]),
        "mask_rle_refreshed_at_utc": _decode_attr(attrs.get("mask_rle_refreshed_at_utc"))
        or _decode_attr(rle_attrs.get("refreshed_at_utc")),
        "mask_rle_refresh_row_count": _as_int(attrs.get("mask_rle_refresh_row_count")),
        "mask_rle_stale": _as_bool_int(attrs.get("mask_rle_stale")),
        "mask_rle_stale_reason": _decode_attr(attrs.get("mask_rle_stale_reason")),
        "mask_rle_stale_at_utc": _decode_attr(attrs.get("mask_rle_stale_at_utc")),
        "mask_rle_stale_component_names_json": _json_dumps(stale_components) if stale_components else None,
        "mask_rle_stale_row_count": _as_int(attrs.get("mask_rle_stale_row_count")),
        "mask_rle_stale_row_min": _as_int(attrs.get("mask_rle_stale_row_min")),
        "mask_rle_stale_row_max": _as_int(attrs.get("mask_rle_stale_row_max")),
    }


def _extract_subject_mask_performance_rows(
    root: zarr.Group,
    *,
    zarr_path: Path,
    recording_id: Optional[str],
    zarr_use: Optional[str],
) -> List[Dict[str, Any]]:
    try:
        zarr_mtime_ns = int(zarr_path.stat().st_mtime_ns)
    except Exception:
        zarr_mtime_ns = None
    updated_utc = utc_now()
    latest_subject_mask_run = _resolve_latest_group_name(root.get("subject_mask_runs"))

    rows: List[Dict[str, Any]] = []
    for stage_group in ("subject_mask_runs", "refined_subject_masks_runs"):
        parent = root.get(stage_group)
        if parent is None:
            continue
        for run_name in _run_names(parent):
            if run_name not in parent:
                continue
            run_group = parent[run_name]
            attrs = dict(run_group.attrs)
            provenance = _coerce_mapping(attrs.get("provenance")) or {}
            summary_statistics = _coerce_mapping(attrs.get("summary_statistics"))
            reason_counts = _coerce_mapping(attrs.get("reason_counts"))
            review_status = _coerce_mapping(
                attrs.get("refined_subject_mask_review_status")
                if stage_group == "refined_subject_masks_runs"
                else attrs.get("subject_mask_review_status")
            ) or _coerce_mapping(attrs.get("subject_mask_review_status"))
            component_review_statuses = _coerce_mapping(attrs.get("component_review_statuses")) or {}

            provenance_parameters = _coerce_mapping(provenance.get("parameters")) or {}
            provenance_inputs = _coerce_mapping(provenance.get("inputs")) or {}
            run_created_utc = (
                _decode_attr(attrs.get("created_at_utc"))
                or _decode_attr(attrs.get("created_utc"))
                or _decode_attr(attrs.get("timestamp_utc"))
                or _decode_attr(provenance.get("created_at_utc"))
            )
            method = (
                _decode_attr(attrs.get("method"))
                or _decode_attr(provenance_parameters.get("method"))
                or _decode_attr(provenance.get("method"))
            )
            if not method and stage_group == "refined_subject_masks_runs":
                method = "refine_subject_masks"

            review_state, review_method, review_intended_use, review_reviewer, review_timestamp_utc = (
                _extract_review_fields(review_status)
            )

            source_subject_mask_run = _decode_attr(attrs.get("source_subject_mask_run")) or _decode_attr(
                provenance_inputs.get("source_subject_mask_run")
            )
            source_subject_mask_method = _decode_attr(attrs.get("source_subject_mask_method")) or _decode_attr(
                provenance_inputs.get("source_subject_mask_method")
            )
            source_subject_mask_stale = _coerce_mapping(attrs.get("source_subject_mask_stale"))
            source_subject_mask_stale_state = (
                _decode_attr(source_subject_mask_stale.get("state")) if source_subject_mask_stale else None
            )
            source_subject_mask_stale_reason = (
                _decode_attr(source_subject_mask_stale.get("reason")) if source_subject_mask_stale else None
            )
            source_subject_mask_stale_timestamp_utc = (
                _decode_attr(source_subject_mask_stale.get("timestamp_utc"))
                or _decode_attr(source_subject_mask_stale.get("timestamp"))
                or _decode_attr(source_subject_mask_stale.get("stale_at_utc"))
                or _decode_attr(source_subject_mask_stale.get("stale_at"))
                if source_subject_mask_stale
                else None
            )
            run_semantics = _decode_attr(attrs.get("run_semantics")) or _decode_attr(
                provenance_parameters.get("run_semantics")
            )
            probability_semantics = _decode_attr(attrs.get("probability_semantics")) or _decode_attr(
                provenance_parameters.get("probability_semantics")
            )
            tuning_source = _decode_attr(attrs.get("tuning_source")) or _decode_attr(
                provenance_parameters.get("tuning_source")
            )
            tuning_timestamp = _decode_attr(attrs.get("tuning_timestamp")) or _decode_attr(
                provenance_parameters.get("tuning_timestamp")
            )
            source_background_run = _decode_attr(attrs.get("source_background_run")) or _decode_attr(
                provenance_inputs.get("source_background_run")
            )
            source_background_array = _decode_attr(attrs.get("source_background_array")) or _decode_attr(
                provenance_inputs.get("source_background_array")
            )
            source_dish_mask_array = _decode_attr(attrs.get("source_dish_mask_array")) or _decode_attr(
                provenance_inputs.get("source_dish_mask_array")
            )
            if (
                source_subject_mask_stale_state is None
                and stage_group == "refined_subject_masks_runs"
                and latest_subject_mask_run
                and source_subject_mask_run
                and source_subject_mask_run != latest_subject_mask_run
            ):
                source_subject_mask_stale_state = "stale"
                source_subject_mask_stale_reason = "latest_subject_mask_run_mismatch"

            lifecycle_state, lifecycle_reason = _derive_review_lifecycle(
                review_state=review_state,
                stale_state=source_subject_mask_stale_state,
                stale_reason=source_subject_mask_stale_reason,
            )

            mask_labels = _coerce_text_list(attrs.get("mask_labels"))
            available_flags: List[bool] = []
            available_channels = _get_group(run_group, "available_channels")
            if available_channels is not None:
                try:
                    available_flags = _coerce_bool_list(available_channels[:])
                except Exception:
                    available_flags = []
            if mask_labels:
                if not available_flags:
                    available_flags = [True] * len(mask_labels)
                if len(available_flags) < len(mask_labels):
                    available_flags.extend([False] * (len(mask_labels) - len(available_flags)))
                available_flags = available_flags[: len(mask_labels)]
            available_components = [label for label, flag in zip(mask_labels, available_flags) if flag]
            unavailable_components = [label for label, flag in zip(mask_labels, available_flags) if not flag]
            component_review_states = {
                str(label): state
                for label, payload in component_review_statuses.items()
                if (state := (_decode_attr((_coerce_mapping(payload) or {}).get("state")) or _decode_attr((_coerce_mapping(payload) or {}).get("review_state"))))
            }

            eye_component_mode: Optional[str] = None
            if "eye_left" in mask_labels or "eye_right" in mask_labels:
                eye_component_mode = "lr"
            elif "eyes_union" in mask_labels:
                eye_component_mode = "union"

            total_rois = _extract_subject_mask_total_rois(run_group, attrs)
            rows_with_any_mask, _ = _extract_subject_mask_presence(
                run_group,
                mask_labels=mask_labels,
                total_rois=total_rois,
            )
            coverage_percent = None
            if rows_with_any_mask is not None and total_rois is not None:
                ratio = _format_ratio(rows_with_any_mask, total_rois)
                coverage_percent = float(ratio) * 100.0 if ratio is not None else None

            duration_seconds = _as_float(attrs.get("duration_seconds"))
            rois_per_second = None
            if duration_seconds is not None and duration_seconds > 0 and total_rois is not None:
                rois_per_second = float(total_rois) / float(duration_seconds)
            source_roi_pixel_contract, source_roi_pixel_contract_name = _resolve_source_roi_pixel_contract(
                attrs,
                provenance_inputs,
            )
            source_roi_cache_used = _as_bool_int(
                _lookup_attr(attrs, provenance_inputs, "source_roi_cache_used", "roi_cache_used")
            )
            mask_storage_fields = _extract_mask_storage_fields(run_group, attrs)

            rows.append(
                {
                    "stage_group": stage_group,
                    "run_name": str(run_name),
                    "run_created_utc": run_created_utc,
                    "recording_id": recording_id,
                    "zarr_use": zarr_use,
                    "subject_mask_method": method,
                    "label_schema_id": _decode_attr(attrs.get("label_schema_id")),
                    "source_crop_run": _decode_attr(attrs.get("source_crop_run")) or _decode_attr(
                        provenance_inputs.get("source_crop_run")
                    ),
                    "source_crop_storage_mode": _decode_attr(
                        _lookup_attr(attrs, provenance_inputs, "source_crop_storage_mode")
                    ),
                    "source_crop_signature": _decode_attr(
                        _lookup_attr(attrs, provenance_inputs, "source_crop_signature")
                    ),
                    "source_crop_revision": _as_int(
                        _lookup_attr(attrs, provenance_inputs, "source_crop_revision")
                    ),
                    "source_roi_image_representation": _decode_attr(
                        _lookup_attr(attrs, provenance_inputs, "source_roi_image_representation")
                    ),
                    "source_roi_pixel_contract_name": source_roi_pixel_contract_name,
                    "source_roi_pixel_contract_json": _json_dumps(source_roi_pixel_contract),
                    "source_roi_read_mode": _decode_attr(
                        _lookup_attr(attrs, provenance_inputs, "source_roi_read_mode")
                    ),
                    "roi_cache_policy": _decode_attr(_lookup_attr(attrs, provenance_inputs, "roi_cache_policy")),
                    "source_roi_cache_used": source_roi_cache_used,
                    "source_roi_cache_backend": _decode_attr(
                        _lookup_attr(attrs, provenance_inputs, "source_roi_cache_backend", "roi_cache_backend")
                    ),
                    "source_roi_live_acceleration_effective": _decode_attr(
                        _lookup_attr(
                            attrs,
                            provenance_inputs,
                            "source_roi_live_acceleration_effective",
                            "roi_live_acceleration_effective",
                        )
                    ),
                    "source_roi_live_gpu_chunk_frames": _as_int(
                        _lookup_attr(
                            attrs,
                            provenance_inputs,
                            "source_roi_live_gpu_chunk_frames",
                            "roi_live_gpu_chunk_frames",
                        )
                    ),
                    **mask_storage_fields,
                    "source_keypoint_group": _decode_attr(attrs.get("source_keypoint_group")) or _decode_attr(
                        provenance_inputs.get("source_keypoint_group")
                    ),
                    "source_keypoints_run": _resolve_source_keypoints_run(attrs)
                    or _decode_attr(provenance_inputs.get("source_keypoints_run"))
                    or _decode_attr(provenance_inputs.get("source_keypoint_run")),
                    "source_subject_mask_run": source_subject_mask_run,
                    "source_subject_mask_method": source_subject_mask_method,
                    "run_semantics": run_semantics,
                    "probability_semantics": probability_semantics,
                    "source_background_run": source_background_run,
                    "source_background_array": source_background_array,
                    "source_dish_mask_array": source_dish_mask_array,
                    "tuning_source": tuning_source,
                    "tuning_timestamp": tuning_timestamp,
                    "total_rois": total_rois,
                    "rows_with_any_mask": rows_with_any_mask,
                    "coverage_percent": coverage_percent,
                    "duration_seconds": duration_seconds,
                    "rois_per_second": rois_per_second,
                    "available_component_count": len(available_components),
                    "available_components_json": _json_dumps(available_components),
                    "unavailable_components_json": _json_dumps(unavailable_components),
                    "component_review_states_json": _json_dumps(component_review_states),
                    "eye_component_mode": eye_component_mode,
                    "reason_counts_json": _json_dumps(reason_counts),
                    "summary_statistics_json": _json_dumps(summary_statistics),
                    "review_state": review_state,
                    "review_method": review_method,
                    "review_intended_use": review_intended_use,
                    "review_reviewer": review_reviewer,
                    "review_timestamp_utc": review_timestamp_utc,
                    "source_subject_mask_stale_state": source_subject_mask_stale_state,
                    "source_subject_mask_stale_reason": source_subject_mask_stale_reason,
                    "source_subject_mask_stale_timestamp_utc": source_subject_mask_stale_timestamp_utc,
                    "source_subject_mask_stale_json": _json_dumps(source_subject_mask_stale),
                    "lifecycle_state": lifecycle_state,
                    "lifecycle_reason": lifecycle_reason,
                    "zarr_mtime_ns": zarr_mtime_ns,
                    "updated_utc": updated_utc,
                }
            )

    return rows


def _extract_subject_mask_component_quality_rows(
    root: zarr.Group,
    *,
    zarr_path: Path,
    recording_id: Optional[str],
    zarr_use: Optional[str],
) -> List[Dict[str, Any]]:
    performance_rows = _extract_subject_mask_performance_rows(
        root,
        zarr_path=zarr_path,
        recording_id=recording_id,
        zarr_use=zarr_use,
    )
    rows: List[Dict[str, Any]] = []
    for performance in performance_rows:
        stage_group = str(performance.get("stage_group") or "")
        run_name = str(performance.get("run_name") or "")
        parent = root.get(stage_group)
        if parent is None or run_name not in parent:
            continue
        run_group = parent[run_name]
        attrs = dict(run_group.attrs)
        mask_labels = _coerce_text_list(attrs.get("mask_labels"))
        if not mask_labels:
            continue

        available_flags: List[bool] = []
        available_channels = _get_group(run_group, "available_channels")
        if available_channels is not None:
            try:
                available_flags = _coerce_bool_list(available_channels[:])
            except Exception:
                available_flags = []
        if not available_flags:
            available_flags = [True] * len(mask_labels)
        if len(available_flags) < len(mask_labels):
            available_flags.extend([False] * (len(mask_labels) - len(available_flags)))
        available_flags = available_flags[: len(mask_labels)]

        total_rois = _as_int(performance.get("total_rois"))
        _, counts_by_label = _extract_subject_mask_presence(
            run_group,
            mask_labels=mask_labels,
            total_rois=total_rois,
        )
        component_review_statuses = _coerce_mapping(attrs.get("component_review_statuses")) or {}
        stale_state = _decode_attr(performance.get("source_subject_mask_stale_state"))
        stale_reason = _decode_attr(performance.get("source_subject_mask_stale_reason"))

        for index, component_name in enumerate(mask_labels):
            available = bool(available_flags[index]) if index < len(available_flags) else False
            review_payload = _coerce_mapping(component_review_statuses.get(component_name))
            review_state, review_method, review_intended_use, review_reviewer, review_timestamp_utc = (
                _extract_review_fields(review_payload)
            )
            rows_with_component_mask = counts_by_label.get(component_name)
            rows_with_component_mask_rate = _format_ratio(rows_with_component_mask, total_rois)
            lifecycle_state: Optional[str]
            lifecycle_reason: Optional[str]
            if not available:
                lifecycle_state, lifecycle_reason = "na", "component_unavailable"
            else:
                lifecycle_state, lifecycle_reason = _derive_review_lifecycle(
                    review_state=review_state,
                    stale_state=stale_state,
                    stale_reason=stale_reason,
                )

            rows.append(
                {
                    "stage_group": stage_group,
                    "run_name": run_name,
                    "component_name": component_name,
                    "component_family": _subject_component_family(component_name),
                    "run_created_utc": performance.get("run_created_utc"),
                    "recording_id": performance.get("recording_id"),
                    "zarr_use": performance.get("zarr_use"),
                    "subject_mask_method": performance.get("subject_mask_method"),
                    "label_schema_id": performance.get("label_schema_id"),
                    "eye_component_mode": performance.get("eye_component_mode"),
                    "source_subject_mask_run": performance.get("source_subject_mask_run"),
                    "available": int(available),
                    "review_state": review_state,
                    "review_method": review_method,
                    "review_intended_use": review_intended_use,
                    "review_reviewer": review_reviewer,
                    "review_timestamp_utc": review_timestamp_utc,
                    "total_rois": total_rois,
                    "rows_with_component_mask": rows_with_component_mask,
                    "rows_with_component_mask_rate": rows_with_component_mask_rate,
                    "lifecycle_state": lifecycle_state,
                    "lifecycle_reason": lifecycle_reason,
                    "quality_updated_utc": performance.get("updated_utc") or utc_now(),
                    "zarr_mtime_ns": performance.get("zarr_mtime_ns"),
                }
            )
    return rows


def _extract_eye_mask_performance_rows(
    root: zarr.Group,
    *,
    zarr_path: Path,
    recording_id: Optional[str],
    zarr_use: Optional[str],
) -> List[Dict[str, Any]]:
    try:
        zarr_mtime_ns = int(zarr_path.stat().st_mtime_ns)
    except Exception:
        zarr_mtime_ns = None
    updated_utc = utc_now()

    rows: List[Dict[str, Any]] = []
    for stage_group in ("eye_masks_runs", "refined_eye_masks_runs"):
        parent = root.get(stage_group)
        if parent is None:
            continue
        for run_name in _run_names(parent):
            if run_name not in parent:
                continue
            run_group = parent[run_name]
            attrs = dict(run_group.attrs)
            provenance = _coerce_mapping(attrs.get("provenance")) or {}
            summary_statistics = _coerce_mapping(attrs.get("summary_statistics"))
            reason_counts = _coerce_mapping(attrs.get("reason_counts"))
            review_status = _coerce_mapping(attrs.get("eye_mask_review_status"))
            source_keypoint_stale = _coerce_mapping(attrs.get("source_keypoint_stale"))

            run_created_utc = (
                _decode_attr(attrs.get("created_at_utc"))
                or _decode_attr(attrs.get("created_utc"))
                or _decode_attr(attrs.get("timestamp_utc"))
                or _decode_attr(provenance.get("created_at_utc"))
            )
            method = _decode_attr(attrs.get("method")) or _decode_attr(provenance.get("method"))
            review_state = _decode_attr(review_status.get("state")) if review_status else None
            review_method = _decode_attr(review_status.get("method")) if review_status else None
            review_intended_use = _decode_attr(review_status.get("intended_use")) if review_status else None
            review_reviewer = _decode_attr(review_status.get("reviewer")) if review_status else None
            review_timestamp_utc = (
                _decode_attr(review_status.get("timestamp_utc"))
                or _decode_attr(review_status.get("timestamp"))
                or _decode_attr(review_status.get("reviewed_at_utc"))
                or _decode_attr(review_status.get("reviewed_at"))
                if review_status
                else None
            )
            source_keypoint_stale_state = (
                _decode_attr(source_keypoint_stale.get("state")) if source_keypoint_stale else None
            )
            source_keypoint_stale_reason = (
                _decode_attr(source_keypoint_stale.get("reason")) if source_keypoint_stale else None
            )
            source_keypoint_stale_timestamp_utc = (
                _decode_attr(source_keypoint_stale.get("timestamp_utc"))
                or _decode_attr(source_keypoint_stale.get("timestamp"))
                or _decode_attr(source_keypoint_stale.get("stale_at_utc"))
                or _decode_attr(source_keypoint_stale.get("stale_at"))
                if source_keypoint_stale
                else None
            )
            review_state_norm = str(review_state).strip().lower() if review_state else None
            source_stale_state_norm = (
                str(source_keypoint_stale_state).strip().lower() if source_keypoint_stale_state else None
            )
            lifecycle_state: Optional[str] = None
            lifecycle_reason: Optional[str] = None
            if source_stale_state_norm == "stale":
                lifecycle_state = "stale"
                lifecycle_reason = source_keypoint_stale_reason or "source_keypoint_stale"
            elif review_state_norm in {"pending", "needs_review", "review"}:
                lifecycle_state = "in_progress"
                lifecycle_reason = review_state
            elif review_state_norm in {"approved", "rejected"}:
                lifecycle_state = review_state_norm
                lifecycle_reason = review_state

            total_rois = _as_int(attrs.get("total_rois"))
            successful_eyes = _as_int(attrs.get("successful_eyes"))
            successful_roi_pairs = _as_int(attrs.get("successful_roi_pairs"))
            successful_roi_pair_rate = _as_float(attrs.get("successful_roi_pair_rate"))
            if successful_roi_pair_rate is None:
                successful_roi_pair_rate = _format_ratio(successful_roi_pairs, total_rois)
            duration_seconds = _as_float(attrs.get("duration_seconds"))
            rois_per_second = None
            if duration_seconds is not None and duration_seconds > 0 and total_rois is not None:
                rois_per_second = float(total_rois) / float(duration_seconds)
            inference_duration_seconds = _as_float(attrs.get("inference_duration_seconds"))
            inference_average_fps = _as_float(attrs.get("inference_average_fps"))
            if inference_average_fps is None and rois_per_second is not None:
                inference_average_fps = rois_per_second

            rows.append(
                {
                    "stage_group": stage_group,
                    "run_name": str(run_name),
                    "run_created_utc": run_created_utc,
                    "recording_id": recording_id,
                    "zarr_use": zarr_use,
                    "method": method,
                    "source_crop_run": _decode_attr(attrs.get("source_crop_run")),
                    "source_keypoint_group": _decode_attr(attrs.get("source_keypoint_group")),
                    "source_keypoints_run": _resolve_source_keypoints_run(attrs),
                    "source_eye_masks_run": _decode_attr(attrs.get("source_eye_masks_run")),
                    "source_eye_masks_method": _decode_attr(attrs.get("source_eye_masks_method")),
                    "total_rois": total_rois,
                    "successful_eyes": successful_eyes,
                    "successful_roi_pairs": successful_roi_pairs,
                    "successful_roi_pair_rate": successful_roi_pair_rate,
                    "duration_seconds": duration_seconds,
                    "rois_per_second": rois_per_second,
                    "inference_duration_seconds": inference_duration_seconds,
                    "inference_average_fps": inference_average_fps,
                    "reason_counts_json": _json_dumps(reason_counts),
                    "summary_statistics_json": _json_dumps(summary_statistics),
                    "review_state": review_state,
                    "review_method": review_method,
                    "review_intended_use": review_intended_use,
                    "review_reviewer": review_reviewer,
                    "review_timestamp_utc": review_timestamp_utc,
                    "source_keypoint_stale_state": source_keypoint_stale_state,
                    "source_keypoint_stale_reason": source_keypoint_stale_reason,
                    "source_keypoint_stale_timestamp_utc": source_keypoint_stale_timestamp_utc,
                    "source_keypoint_stale_json": _json_dumps(source_keypoint_stale),
                    "lifecycle_state": lifecycle_state,
                    "lifecycle_reason": lifecycle_reason,
                    "zarr_mtime_ns": zarr_mtime_ns,
                    "updated_utc": updated_utc,
                }
            )

    return rows


def _extract_eye_mask_quality_rows(
    root: zarr.Group,
    *,
    zarr_path: Path,
    recording_id: Optional[str],
    zarr_use: Optional[str],
) -> List[Dict[str, Any]]:
    performance_rows = _extract_eye_mask_performance_rows(
        root,
        zarr_path=zarr_path,
        recording_id=recording_id,
        zarr_use=zarr_use,
    )
    rows: List[Dict[str, Any]] = []
    for performance in performance_rows:
        # Quality parity tracks reviewed refined runs, mirroring detect/keypoint quality scope.
        if str(performance.get("stage_group") or "") != "refined_eye_masks_runs":
            continue
        rows.append(
            {
                "stage_group": str(performance.get("stage_group") or "refined_eye_masks_runs"),
                "run_name": str(performance.get("run_name") or ""),
                "run_created_utc": performance.get("run_created_utc"),
                "recording_id": performance.get("recording_id"),
                "zarr_use": performance.get("zarr_use"),
                "eye_mask_method": performance.get("method"),
                "source_crop_run": performance.get("source_crop_run"),
                "source_keypoint_group": performance.get("source_keypoint_group"),
                "source_keypoints_run": performance.get("source_keypoints_run"),
                "source_eye_masks_run": performance.get("source_eye_masks_run"),
                "source_eye_masks_method": performance.get("source_eye_masks_method"),
                "review_state": performance.get("review_state"),
                "review_method": performance.get("review_method"),
                "review_intended_use": performance.get("review_intended_use"),
                "review_reviewer": performance.get("review_reviewer"),
                "review_timestamp_utc": performance.get("review_timestamp_utc"),
                "total_rois": performance.get("total_rois"),
                "successful_eyes": performance.get("successful_eyes"),
                "successful_roi_pairs": performance.get("successful_roi_pairs"),
                "successful_roi_pair_rate": performance.get("successful_roi_pair_rate"),
                "source_keypoint_stale_state": performance.get("source_keypoint_stale_state"),
                "source_keypoint_stale_reason": performance.get("source_keypoint_stale_reason"),
                "source_keypoint_stale_timestamp_utc": performance.get("source_keypoint_stale_timestamp_utc"),
                "source_keypoint_stale_json": performance.get("source_keypoint_stale_json"),
                "lifecycle_state": performance.get("lifecycle_state"),
                "lifecycle_reason": performance.get("lifecycle_reason"),
                "quality_updated_utc": performance.get("updated_utc") or utc_now(),
                "zarr_mtime_ns": performance.get("zarr_mtime_ns"),
            }
        )
    return rows


__all__ = [
    "_extract_eye_mask_performance_rows",
    "_extract_eye_mask_quality_rows",
    "_extract_subject_mask_component_quality_rows",
    "_extract_subject_mask_performance_rows",
]
