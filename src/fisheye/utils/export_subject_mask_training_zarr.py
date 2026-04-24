#!/usr/bin/env python3
"""Export and validate merged subject-mask-training Zarr artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import zarr

from fisheye.registry.db import Registry
from fisheye.shared.batch_logging import utc_now
from fisheye.shared.frame_flags import resolve_row_identity_arrays
from fisheye.shared.type_conversions import normalize_attr as _as_text
from fisheye.utils.export_eye_mask_training_zarr import (
    _json_dict,
    _json_list,
    _make_split_indices,
    _resolve_source_dataset_id,
    _write_string_array,
)

TARGET_SCHEMAS: Dict[str, Tuple[str, ...]] = {
    "subject_v1_union": ("subject_body", "eyes_union", "swim_bladder"),
    "subject_v1_lr": ("subject_body", "eye_left", "eye_right", "swim_bladder"),
}
LABEL_SCHEMA_CHOICES = ("auto", "subject_v1_union", "subject_v1_lr")
SUBJECT_STAGE_CHOICES = ("subject_mask_runs", "refined_subject_masks_runs")
LABEL_ORIGIN_CODES = {
    "unknown": 0,
    "auto": 1,
    "manual_review": 2,
    "manual_training": 3,
    "interpolated": 4,
    "synthetic": 5,
}
SUPERVISION_MODE_CODES = {
    "no_supervision": 0,
    "dense": 1,
    "explicit_negative": 2,
    "box_only": 3,
}

_utc_now = utc_now


def _clean_slug(value: Optional[str], fallback: str) -> str:
    text = _as_text(value) or fallback
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in text)
    cleaned = cleaned.strip("._")
    return cleaned or fallback


def _collapse_source_attr(values: Sequence[str], *, mixed_value: str = "mixed") -> str:
    normalized = [str(value).strip() for value in values if str(value).strip()]
    if not normalized:
        return ""
    unique = list(dict.fromkeys(normalized))
    return unique[0] if len(unique) == 1 else mixed_value


def _normalize_input_format(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"gray", "grey", "grayscale"}:
        return "gray"
    if text in {"rgb", "color", "colour"}:
        return "rgb"
    return None


def _normalize_label_schema(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in TARGET_SCHEMAS:
        return text
    if text == "auto":
        return "auto"
    return None


def _normalize_subject_stage_group(value: Optional[str]) -> str:
    text = _as_text(value) or "subject_mask_runs"
    if text not in SUBJECT_STAGE_CHOICES:
        raise ValueError(
            f"Unsupported subject-mask source stage {text!r}. "
            f"Expected one of {sorted(SUBJECT_STAGE_CHOICES)}."
        )
    return text


def _normalize_bool_array(value: Any, *, expected_len: int) -> np.ndarray:
    if value is None:
        return np.ones((expected_len,), dtype=np.bool_)
    if isinstance(value, zarr.Array):
        arr = np.asarray(value[:], dtype=np.bool_)
    else:
        arr = np.asarray(value, dtype=np.bool_)
    arr = arr.reshape(-1)
    if int(arr.shape[0]) != int(expected_len):
        raise ValueError(
            f"available_channels length mismatch ({arr.shape[0]} != expected {expected_len})."
        )
    return arr.astype(np.bool_, copy=False)


def _normalize_mask_labels(value: Any) -> List[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [value]
    return []


def _resolve_source_run(
    root: zarr.Group,
    *,
    stage_group: str,
    run_name: Optional[str],
) -> Tuple[str, zarr.Group]:
    parent = root.get(stage_group)
    if not isinstance(parent, zarr.Group):
        raise ValueError(f"Missing required group '{stage_group}'.")
    if run_name:
        if run_name not in parent:
            raise ValueError(f"Subject-mask run '{run_name}' not found under {stage_group}.")
        return str(run_name), parent[str(run_name)]
    latest = _as_text(parent.attrs.get("latest"))
    if latest and latest in parent:
        return latest, parent[latest]
    names = (
        sorted(str(name) for name in parent.group_keys())
        if hasattr(parent, "group_keys")
        else sorted(parent.keys())
    )
    if not names:
        raise ValueError(f"No subject-mask runs found under {stage_group}.")
    return str(names[-1]), parent[str(names[-1])]


def _resolve_crop_run(
    root: zarr.Group,
    *,
    requested: Optional[str],
    source_group: zarr.Group,
) -> Tuple[str, zarr.Group]:
    crop_name = _as_text(requested) or _as_text(source_group.attrs.get("source_crop_run"))
    crop_parent = root.get("crop_runs")
    if not isinstance(crop_parent, zarr.Group):
        raise ValueError("Missing required group 'crop_runs'.")
    if crop_name is None:
        latest = _as_text(crop_parent.attrs.get("latest"))
        if latest and latest in crop_parent:
            crop_name = latest
    if crop_name is None:
        names = (
            sorted(str(name) for name in crop_parent.group_keys())
            if hasattr(crop_parent, "group_keys")
            else sorted(crop_parent.keys())
        )
        if not names:
            raise ValueError("No crop runs found under crop_runs.")
        crop_name = str(names[-1])
    if crop_name not in crop_parent:
        raise ValueError(f"Crop run '{crop_name}' not found under crop_runs.")
    return str(crop_name), crop_parent[str(crop_name)]


def _resolve_source_schema(source_group: zarr.Group, *, stage_group: str) -> Tuple[str, List[str], np.ndarray]:
    label_schema_id = _as_text(source_group.attrs.get("label_schema_id"))
    if label_schema_id not in TARGET_SCHEMAS:
        raise ValueError(
            f"Unsupported subject-mask source schema {label_schema_id!r}. Expected one of {sorted(TARGET_SCHEMAS)}."
        )
    mask_labels = _normalize_mask_labels(source_group.attrs.get("mask_labels"))
    expected = list(TARGET_SCHEMAS[label_schema_id])
    if mask_labels != expected:
        raise ValueError(
            f"{stage_group} mask_labels mismatch for schema {label_schema_id}: "
            f"{mask_labels!r} != {expected!r}."
        )
    available = _normalize_bool_array(
        source_group.get("available_channels"),
        expected_len=len(expected),
    )
    return label_schema_id, expected, available


def _resolve_target_schema(requested: str, *, source_schema_ids: Sequence[str]) -> str:
    if requested != "auto":
        if requested == "subject_v1_lr" and any(schema != "subject_v1_lr" for schema in source_schema_ids):
            raise ValueError(
                "subject_v1_lr export requires all source runs to already provide subject_v1_lr labels."
            )
        return requested
    if source_schema_ids and all(schema == "subject_v1_lr" for schema in source_schema_ids):
        return "subject_v1_lr"
    return "subject_v1_union"


def _infer_input_format(roi_images: zarr.Array, requested: Optional[str]) -> str:
    explicit = _normalize_input_format(requested)
    if explicit:
        return explicit
    if int(roi_images.ndim) == 4 and int(roi_images.shape[-1]) == 3:
        return "rgb"
    return "gray"


def _projection_mode(source_schema_id: str, target_schema_id: str) -> str:
    if source_schema_id == target_schema_id:
        return "schema_identity"
    if source_schema_id == "subject_v1_lr" and target_schema_id == "subject_v1_union":
        return "eyes_union_from_subject_lr"
    raise ValueError(
        f"Unsupported source/target schema projection: {source_schema_id} -> {target_schema_id}."
    )


def _session_id_for_export(out_zarr: Path, *, training_set_id: Optional[str]) -> str:
    base = training_set_id or out_zarr.stem
    slug = _clean_slug(base, fallback="subject_masks_training")
    return slug if slug.endswith("_merged") else f"{slug}_merged"


def _copy_optional_array(
    src_group: zarr.Group,
    name: str,
    *,
    fallback_name: Optional[str] = None,
    default: Optional[np.ndarray] = None,
) -> np.ndarray:
    if name in src_group:
        return np.asarray(src_group[name][:])
    if fallback_name and fallback_name in src_group:
        return np.asarray(src_group[fallback_name][:])
    if default is not None:
        return default
    raise ValueError(f"Missing required array '{name}'.")


def _codebook_attr(codebook: Dict[str, int]) -> Dict[str, int]:
    return {str(name): int(code) for name, code in codebook.items()}


def _normalize_codebook_attr(value: Any) -> Optional[Dict[str, int]]:
    if not isinstance(value, Mapping):
        return None
    normalized: Dict[str, int] = {}
    for key, raw_code in value.items():
        try:
            normalized[str(key)] = int(raw_code)
        except (TypeError, ValueError):
            return None
    return normalized


@dataclass(frozen=True)
class SubjectMergeSourceSpec:
    source_zarr: Path
    subject_run: Optional[str] = None
    crop_run: Optional[str] = None
    stage_group: str = "subject_mask_runs"


@dataclass(frozen=True)
class SubjectResolvedSource:
    source_zarr: Path
    dataset_id: str
    stage_group: str
    subject_run: str
    crop_run: str
    source_schema_id: str
    source_mask_labels: Tuple[str, ...]
    available_channels: np.ndarray
    projection_mode: str
    roi_images: zarr.Array
    bbox_norm_coords: zarr.Array
    crop_bbox_norm_coords: zarr.Array
    frame_indices: np.ndarray
    source_refined_row_ids: np.ndarray
    source_detect_row_index: np.ndarray
    detection_source: np.ndarray
    masks_roi: zarr.Array
    row_count: int
    input_format: str


def _resolve_source_spec(
    spec: SubjectMergeSourceSpec,
    *,
    registry_path: Optional[Path],
    target_schema_id: str,
    requested_input_format: Optional[str],
) -> SubjectResolvedSource:
    source_path = Path(spec.source_zarr).expanduser().resolve()
    root = zarr.open_group(str(source_path), mode="r")
    stage_group = _normalize_subject_stage_group(spec.stage_group)
    subject_run, subject_group = _resolve_source_run(
        root,
        stage_group=stage_group,
        run_name=spec.subject_run,
    )
    crop_run, crop_group = _resolve_crop_run(root, requested=spec.crop_run, source_group=subject_group)

    if "roi_images" not in crop_group:
        raise ValueError(f"{source_path.name}: crop_runs/{crop_run} missing roi_images.")
    if "bbox_norm_coords" not in crop_group:
        raise ValueError(f"{source_path.name}: crop_runs/{crop_run} missing bbox_norm_coords.")
    if "masks_roi" not in subject_group:
        raise ValueError(f"{source_path.name}: {stage_group}/{subject_run} missing masks_roi.")

    source_schema_id, source_labels, available_channels = _resolve_source_schema(
        subject_group,
        stage_group=stage_group,
    )
    projection_mode = _projection_mode(source_schema_id, target_schema_id)
    roi_images = crop_group["roi_images"]
    masks_roi = subject_group["masks_roi"]
    if int(roi_images.shape[0]) != int(masks_roi.shape[0]):
        raise ValueError(
            f"{source_path.name}: row mismatch between crop_runs/{crop_run}/roi_images "
            f"({roi_images.shape[0]}) and subject_mask_runs/{subject_run}/masks_roi ({masks_roi.shape[0]})."
        )
    bbox_norm = crop_group["bbox_norm_coords"]
    crop_bbox = crop_group["crop_bbox_norm_coords"] if "crop_bbox_norm_coords" in crop_group else bbox_norm
    frame_indices = _copy_optional_array(
        crop_group,
        "frame_indices",
        default=np.arange(int(roi_images.shape[0]), dtype=np.int64),
    ).astype(np.int64, copy=False)
    detection_source = _copy_optional_array(
        crop_group,
        "detection_source",
        default=np.zeros((int(roi_images.shape[0]),), dtype=np.int8),
    ).astype(np.int8, copy=False)
    source_refined_row_ids, source_detect_row_index = resolve_row_identity_arrays(
        subject_group,
        crop_group,
        total_rois=int(roi_images.shape[0]),
    )

    dataset_id = _resolve_source_dataset_id(
        source_root=root,
        source_path=source_path,
        registry_path=registry_path,
    )
    input_format = _infer_input_format(roi_images, requested_input_format)
    return SubjectResolvedSource(
        source_zarr=source_path,
        dataset_id=dataset_id,
        stage_group=stage_group,
        subject_run=subject_run,
        crop_run=crop_run,
        source_schema_id=source_schema_id,
        source_mask_labels=tuple(source_labels),
        available_channels=available_channels.astype(np.bool_, copy=False),
        projection_mode=projection_mode,
        roi_images=roi_images,
        bbox_norm_coords=bbox_norm,
        crop_bbox_norm_coords=crop_bbox,
        frame_indices=frame_indices,
        source_refined_row_ids=source_refined_row_ids,
        source_detect_row_index=source_detect_row_index,
        detection_source=detection_source,
        masks_roi=masks_roi,
        row_count=int(masks_roi.shape[0]),
        input_format=input_format,
    )


def _schema_label_index(label_schema_id: str) -> Dict[str, int]:
    return {label: idx for idx, label in enumerate(TARGET_SCHEMAS[label_schema_id])}


def _project_masks_and_validity(
    source_masks: np.ndarray,
    *,
    source_schema_id: str,
    source_available: np.ndarray,
    target_schema_id: str,
) -> Tuple[np.ndarray, np.ndarray]:
    total_rows = int(source_masks.shape[0])
    height = int(source_masks.shape[2])
    width = int(source_masks.shape[3])
    target_labels = TARGET_SCHEMAS[target_schema_id]
    target_masks = np.zeros((total_rows, len(target_labels), height, width), dtype=np.uint8)
    target_valid = np.zeros((total_rows, len(target_labels)), dtype=np.bool_)

    source_idx = _schema_label_index(source_schema_id)
    target_idx = _schema_label_index(target_schema_id)

    if source_schema_id == target_schema_id:
        for label in target_labels:
            s_idx = source_idx[label]
            t_idx = target_idx[label]
            target_masks[:, t_idx] = source_masks[:, s_idx].astype(np.uint8, copy=False)
            if bool(source_available[s_idx]):
                target_valid[:, t_idx] = True
        return target_masks, target_valid

    if source_schema_id == "subject_v1_lr" and target_schema_id == "subject_v1_union":
        body_src = source_idx["subject_body"]
        left_src = source_idx["eye_left"]
        right_src = source_idx["eye_right"]
        bladder_src = source_idx["swim_bladder"]

        target_masks[:, target_idx["subject_body"]] = source_masks[:, body_src].astype(np.uint8, copy=False)
        target_masks[:, target_idx["eyes_union"]] = np.maximum(
            source_masks[:, left_src],
            source_masks[:, right_src],
        ).astype(np.uint8, copy=False)
        target_masks[:, target_idx["swim_bladder"]] = source_masks[:, bladder_src].astype(np.uint8, copy=False)

        if bool(source_available[body_src]):
            target_valid[:, target_idx["subject_body"]] = True
        if bool(source_available[left_src]) and bool(source_available[right_src]):
            target_valid[:, target_idx["eyes_union"]] = True
        if bool(source_available[bladder_src]):
            target_valid[:, target_idx["swim_bladder"]] = True
        return target_masks, target_valid

    raise ValueError(f"Unsupported source/target schema projection: {source_schema_id} -> {target_schema_id}.")


def _summarize_channel_supervision(
    *,
    masks_roi: np.ndarray,
    target_valid_channels: np.ndarray,
    mask_labels: Sequence[str],
) -> Dict[str, Any]:
    supervised_row_counts: Dict[str, int] = {}
    positive_row_counts: Dict[str, int] = {}
    negative_row_counts: Dict[str, int] = {}
    unsupervised_row_counts: Dict[str, int] = {}
    available_labels: List[str] = []
    missing_labels: List[str] = []

    eye_labels = {"eyes_union", "eye_left", "eye_right"}
    for idx, label in enumerate(mask_labels):
        valid_mask = np.asarray(target_valid_channels[:, idx], dtype=np.bool_)
        positive_mask = np.sum(np.asarray(masks_roi[:, idx], dtype=np.uint8), axis=(1, 2)) > 0
        supervised_count = int(np.sum(valid_mask))
        positive_count = int(np.sum(valid_mask & positive_mask))
        supervised_row_counts[str(label)] = supervised_count
        positive_row_counts[str(label)] = positive_count
        negative_row_counts[str(label)] = int(supervised_count - positive_count)
        unsupervised_row_counts[str(label)] = int(target_valid_channels.shape[0] - supervised_count)
        if supervised_count > 0:
            available_labels.append(str(label))
        else:
            missing_labels.append(str(label))

    eye_supervised = any(supervised_row_counts.get(label, 0) > 0 for label in eye_labels)
    body_supervised = supervised_row_counts.get("subject_body", 0) > 0
    bladder_supervised = supervised_row_counts.get("swim_bladder", 0) > 0
    if eye_supervised and not body_supervised and not bladder_supervised:
        coverage_class = "eyes_only"
    elif all(count > 0 for count in supervised_row_counts.values()):
        coverage_class = "dense_all_components"
    else:
        coverage_class = "partial_subject_masks"

    return {
        "coverage_class": coverage_class,
        "contains_only_eye_masks": bool(coverage_class == "eyes_only"),
        "available_labels": available_labels,
        "missing_labels": missing_labels,
        "supervised_row_counts": supervised_row_counts,
        "positive_row_counts": positive_row_counts,
        "negative_row_counts": negative_row_counts,
        "unsupervised_row_counts": unsupervised_row_counts,
    }


def _merge_invocation_summary(
    existing: Optional[Dict[str, Any]],
    *,
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    payload = dict(existing or {})
    payload["task_type"] = "subject_masks"
    payload["task"] = "subject_masks"
    payload["subject_mask_training_summary"] = summary
    return payload


def _register_merged_dataset_in_registry(
    *,
    registry_path: Path,
    merged_zarr: Path,
    source_zarr_paths: Sequence[Path],
    set_id: Optional[str],
    set_name: Optional[str],
    training_summary: Dict[str, Any],
) -> Dict[str, Any]:
    registry = Registry(registry_path)
    try:
        merged_dataset_id = registry.scan_zarr(merged_zarr)
        if not merged_dataset_id:
            raise RuntimeError(f"Failed to register merged zarr: {merged_zarr}")

        source_dataset_ids: List[str] = []
        for source_path in source_zarr_paths:
            source_dataset_id = registry.scan_zarr(source_path)
            if source_dataset_id:
                source_dataset_ids.append(str(source_dataset_id))
        source_dataset_ids = sorted(set(source_dataset_ids))

        training_set_linked = False
        if set_id:
            existing = registry.conn.execute(
                "SELECT name, query_filter, dataset_ids_json, invocation_json FROM training_sets WHERE set_id = ?",
                (str(set_id),),
            ).fetchone()
            existing_ids = _json_list(existing["dataset_ids_json"]) if existing else []
            existing_query_filter = _json_dict(existing["query_filter"]) if existing else None
            existing_invocation = _json_dict(existing["invocation_json"]) if existing else None
            merged_ids = sorted(
                {
                    str(merged_dataset_id),
                    *(str(item) for item in source_dataset_ids if item),
                    *(str(item) for item in existing_ids if item),
                }
            )
            registry.upsert_training_set(
                set_id=str(set_id),
                name=set_name or (existing["name"] if existing else None),
                task_type="subject_masks",
                query_filter=existing_query_filter,
                dataset_ids=merged_ids,
                invocation=_merge_invocation_summary(existing_invocation, summary=training_summary),
            )
            training_set_linked = True

        registry.replace_dataset_lineage(
            child_dataset_id=str(merged_dataset_id),
            parent_dataset_ids=source_dataset_ids,
            relationship_type="training_merge_source",
            source_set_id=str(set_id) if set_id else None,
            metadata={"producer": "export_subject_mask_training_zarr"},
        )

        return {
            "registry_path": str(registry_path),
            "merged_dataset_id": str(merged_dataset_id),
            "source_dataset_ids": source_dataset_ids,
            "training_set_linked": bool(training_set_linked),
            "training_set_id": str(set_id) if set_id else None,
        }
    finally:
        registry.close()


def export_merged_subject_mask_training_zarr_from_sources(
    *,
    source_specs: Sequence[SubjectMergeSourceSpec],
    out_zarr: Path,
    subject_label_schema: str = "auto",
    input_format: Optional[str] = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
    split_seed: int = 123,
    run_name: Optional[str] = None,
    overwrite: bool = False,
    registry: Optional[Path] = None,
    training_set_id: Optional[str] = None,
    training_set_name: Optional[str] = None,
) -> Dict[str, Any]:
    if not source_specs:
        raise ValueError("At least one source spec is required.")
    normalized_schema = _normalize_label_schema(subject_label_schema)
    if normalized_schema is None:
        raise ValueError(
            f"Unsupported subject_label_schema '{subject_label_schema}'. "
            f"Expected one of {', '.join(LABEL_SCHEMA_CHOICES)}."
        )

    out_path = Path(out_zarr).expanduser().resolve()
    registry_path = Path(registry).expanduser().resolve() if registry is not None else None
    if out_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {out_path}")
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    source_roots: List[Tuple[Path, zarr.Group, str, List[str], np.ndarray]] = []
    for spec in source_specs:
        root = zarr.open_group(str(Path(spec.source_zarr).expanduser().resolve()), mode="r")
        stage_group = _normalize_subject_stage_group(spec.stage_group)
        _subject_run, subject_group = _resolve_source_run(
            root,
            stage_group=stage_group,
            run_name=spec.subject_run,
        )
        schema_id, mask_labels, available = _resolve_source_schema(
            subject_group,
            stage_group=stage_group,
        )
        source_roots.append((Path(spec.source_zarr).expanduser().resolve(), root, schema_id, mask_labels, available))

    target_schema_id = _resolve_target_schema(
        normalized_schema,
        source_schema_ids=[item[2] for item in source_roots],
    )
    resolved_sources = [
        _resolve_source_spec(
            spec,
            registry_path=registry_path,
            target_schema_id=target_schema_id,
            requested_input_format=input_format,
        )
        for spec in source_specs
    ]

    ref_source = resolved_sources[0]
    total_samples = int(sum(item.row_count for item in resolved_sources))
    if total_samples <= 0:
        raise ValueError("Selected subject-mask sources contain no rows.")

    for item in resolved_sources[1:]:
        if tuple(int(v) for v in item.roi_images.shape[1:]) != tuple(int(v) for v in ref_source.roi_images.shape[1:]):
            raise ValueError("All source roi_images arrays must share identical shape after the row axis.")
        if tuple(int(v) for v in item.masks_roi.shape[2:]) != tuple(int(v) for v in ref_source.masks_roi.shape[2:]):
            raise ValueError("All source masks_roi arrays must share identical HxW.")
        if item.input_format != ref_source.input_format:
            raise ValueError(
                f"Mixed input formats are not supported yet ({ref_source.input_format} vs {item.input_format})."
            )

    export_run_name = run_name or f"training_export_{_utc_now().replace(':', '').replace('-', '')}"
    merged_session_id = _session_id_for_export(out_path, training_set_id=training_set_id)
    target_labels = list(TARGET_SCHEMAS[target_schema_id])

    roi_shape = tuple(int(v) for v in ref_source.roi_images.shape[1:])
    bbox_shape = tuple(int(v) for v in ref_source.bbox_norm_coords.shape[1:])
    mask_shape = tuple(int(v) for v in ref_source.masks_roi.shape[2:])
    roi_dtype = ref_source.roi_images.dtype
    mask_dtype = np.dtype(np.uint8)
    train_idx, val_idx, test_idx = _make_split_indices(
        total_samples,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=split_seed,
    )

    root = zarr.open_group(str(out_path), mode="w")
    root.attrs.update(
        {
            "zarr_purpose": "training",
            "training_task": "subject_masks",
            "session_uuid": str(merged_session_id),
            "session_id": str(merged_session_id),
        }
    )

    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = export_run_name
    crop_group = crop_parent.create_group(export_run_name)
    vector_chunks = (max(1, min(8192, total_samples)),)
    bbox_chunks = (max(1, min(8192, total_samples)), max(1, bbox_shape[0] if bbox_shape else 4))

    roi_dest = crop_group.create_array(
        "roi_images",
        shape=(total_samples, *roi_shape),
        dtype=roi_dtype,
        chunks=(max(1, min(256, total_samples)), *roi_shape),
        overwrite=True,
    )
    bbox_dest = crop_group.create_array(
        "bbox_norm_coords",
        shape=(total_samples, *bbox_shape),
        dtype=ref_source.bbox_norm_coords.dtype,
        chunks=bbox_chunks,
        overwrite=True,
    )
    crop_bbox_dest = crop_group.create_array(
        "crop_bbox_norm_coords",
        shape=(total_samples, *bbox_shape),
        dtype=ref_source.crop_bbox_norm_coords.dtype,
        chunks=bbox_chunks,
        overwrite=True,
    )
    frame_indices_dest = crop_group.create_array(
        "frame_indices",
        shape=(total_samples,),
        dtype=np.int64,
        chunks=vector_chunks,
        overwrite=True,
    )
    detection_source_dest = crop_group.create_array(
        "detection_source",
        shape=(total_samples,),
        dtype=np.int8,
        chunks=vector_chunks,
        overwrite=True,
    )

    subject_parent = root.create_group("subject_mask_runs")
    subject_parent.attrs["latest"] = export_run_name
    subject_group = subject_parent.create_group(export_run_name)
    masks_dest = subject_group.create_array(
        "masks_roi",
        shape=(total_samples, len(target_labels), *mask_shape),
        dtype=mask_dtype,
        chunks=(max(1, min(256, total_samples)), len(target_labels), *mask_shape),
        overwrite=True,
    )
    target_valid_dest = subject_group.create_array(
        "target_valid_channels",
        shape=(total_samples, len(target_labels)),
        dtype=np.bool_,
        chunks=(max(1, min(8192, total_samples)), len(target_labels)),
        overwrite=True,
    )

    split_group = root.create_group("splits")
    split_group.create_array("train_indices", data=train_idx, overwrite=True)
    split_group.create_array("val_indices", data=val_idx, overwrite=True)
    split_group.create_array("test_indices", data=test_idx, overwrite=True)

    source_index = root.create_group("source_index")
    source_dataset_idx_dest = source_index.create_array(
        "source_dataset_idx",
        shape=(total_samples,),
        dtype=np.int32,
        chunks=vector_chunks,
        overwrite=True,
    )
    source_frame_idx_dest = source_index.create_array(
        "source_frame_idx",
        shape=(total_samples,),
        dtype=np.int64,
        chunks=vector_chunks,
        overwrite=True,
    )
    source_roi_idx_dest = source_index.create_array(
        "source_roi_idx",
        shape=(total_samples,),
        dtype=np.int64,
        chunks=vector_chunks,
        overwrite=True,
    )
    source_refined_row_ids_dest = source_index.create_array(
        "source_refined_row_ids",
        shape=(total_samples,),
        dtype=np.int64,
        chunks=vector_chunks,
        overwrite=True,
    )
    source_detect_row_index_dest = source_index.create_array(
        "source_detect_row_index",
        shape=(total_samples,),
        dtype=np.int32,
        chunks=vector_chunks,
        overwrite=True,
    )
    label_origin_dest = source_index.create_array(
        "label_origin_codes",
        shape=(total_samples, len(target_labels)),
        dtype=np.uint8,
        chunks=(max(1, min(8192, total_samples)), len(target_labels)),
        overwrite=True,
    )
    supervision_mode_dest = source_index.create_array(
        "supervision_mode_codes",
        shape=(total_samples, len(target_labels)),
        dtype=np.uint8,
        chunks=(max(1, min(8192, total_samples)), len(target_labels)),
        overwrite=True,
    )

    offset = 0
    source_stage_groups: List[str] = []
    source_run_names: List[str] = []
    source_schema_ids: List[str] = []
    source_projection_modes: List[str] = []
    source_dataset_ids: List[str] = []
    source_paths: List[str] = []
    source_crop_runs: List[str] = []

    for source_idx, source in enumerate(resolved_sources):
        count = int(source.row_count)
        row_slice = slice(offset, offset + count)
        roi_dest[row_slice] = source.roi_images[:]
        bbox_dest[row_slice] = source.bbox_norm_coords[:]
        crop_bbox_dest[row_slice] = source.crop_bbox_norm_coords[:]
        frame_indices_dest[row_slice] = np.arange(offset, offset + count, dtype=np.int64)
        detection_source_dest[row_slice] = source.detection_source
        source_dataset_idx_dest[row_slice] = int(source_idx)
        source_frame_idx_dest[row_slice] = source.frame_indices
        source_roi_idx_dest[row_slice] = np.arange(count, dtype=np.int64)
        source_refined_row_ids_dest[row_slice] = source.source_refined_row_ids
        source_detect_row_index_dest[row_slice] = source.source_detect_row_index

        source_masks = np.asarray(source.masks_roi[:], dtype=np.uint8)
        projected_masks, projected_valid = _project_masks_and_validity(
            source_masks,
            source_schema_id=source.source_schema_id,
            source_available=source.available_channels,
            target_schema_id=target_schema_id,
        )
        masks_dest[row_slice] = projected_masks
        target_valid_dest[row_slice] = projected_valid

        label_codes = np.zeros((count, len(target_labels)), dtype=np.uint8)
        supervision_codes = np.zeros((count, len(target_labels)), dtype=np.uint8)
        label_codes[projected_valid] = np.uint8(LABEL_ORIGIN_CODES["auto"])
        supervision_codes[projected_valid] = np.uint8(SUPERVISION_MODE_CODES["dense"])
        label_origin_dest[row_slice] = label_codes
        supervision_mode_dest[row_slice] = supervision_codes

        source_stage_groups.append(str(source.stage_group))
        source_run_names.append(str(source.subject_run))
        source_schema_ids.append(str(source.source_schema_id))
        source_projection_modes.append(str(source.projection_mode))
        source_dataset_ids.append(str(source.dataset_id))
        source_paths.append(str(source.source_zarr))
        source_crop_runs.append(str(source.crop_run))
        offset += count

    _write_string_array(source_index, "source_dataset_id", source_dataset_ids)
    _write_string_array(source_index, "source_zarr_path", source_paths)
    _write_string_array(source_index, "source_stage_group", source_stage_groups)
    _write_string_array(source_index, "source_run_name", source_run_names)
    _write_string_array(source_index, "source_label_schema_id", source_schema_ids)
    _write_string_array(source_index, "source_projection_mode", source_projection_modes)
    source_index.attrs.update(
        {
            "mapping_version": 1,
            "source_count": int(len(resolved_sources)),
            "label_origin_codebook": _codebook_attr(LABEL_ORIGIN_CODES),
            "supervision_mode_codebook": _codebook_attr(SUPERVISION_MODE_CODES),
        }
    )

    masks_final = np.asarray(masks_dest[:], dtype=np.uint8)
    valid_final = np.asarray(target_valid_dest[:], dtype=np.bool_)
    supervision_summary = _summarize_channel_supervision(
        masks_roi=masks_final,
        target_valid_channels=valid_final,
        mask_labels=target_labels,
    )
    source_subject_mask_run = _collapse_source_attr(source_run_names)
    source_crop_run = _collapse_source_attr(source_crop_runs)

    crop_group.attrs.update(
        {
            "source_count": int(len(resolved_sources)),
            "bbox_norm_coords_semantics": "source_crop_bbox_xywhn",
            "crop_bbox_norm_coords_semantics": "source_crop_stage_bbox_xywhn",
        }
    )
    subject_group.attrs.update(
        {
            "label_schema_id": target_schema_id,
            "mask_labels": target_labels,
            "allow_partial_supervision": True,
            "source_mask_stage": source_stage_groups[0] if len(set(source_stage_groups)) == 1 else "mixed",
            "source_subject_mask_run": source_subject_mask_run,
            "source_crop_run": source_crop_run,
            "valid_channel_counts": supervision_summary["supervised_row_counts"],
            "positive_channel_counts": supervision_summary["positive_row_counts"],
            "dense_channel_counts": supervision_summary["supervised_row_counts"],
            "explicit_negative_channel_counts": {
                label: 0 for label in target_labels
            },
            "projection_summary": {
                "target_label_schema_id": target_schema_id,
                "source_projection_modes": sorted(set(source_projection_modes)),
                "coverage_class": supervision_summary["coverage_class"],
            },
        }
    )

    training_summary = {
        "label_schema_id": target_schema_id,
        "mask_labels": target_labels,
        **supervision_summary,
    }
    training_export = {
        "tool": "fisheye.utils.export_subject_mask_training_zarr",
        "created_at_utc": _utc_now(),
        "input_format": ref_source.input_format,
        "label_schema_id": target_schema_id,
        "mask_labels": target_labels,
        "allow_partial_supervision": True,
        "source_stage": source_stage_groups[0] if len(set(source_stage_groups)) == 1 else "mixed",
        "source_count": int(len(resolved_sources)),
        "source_zarr_paths": source_paths,
        "source_stage_groups": list(source_stage_groups),
        "source_subject_mask_run": source_subject_mask_run,
        "source_subject_mask_runs": list(source_run_names),
        "source_crop_run": source_crop_run,
        "source_crop_runs": list(source_crop_runs),
        "split_seed": int(split_seed),
        "split": {
            "train_ratio": float(train_ratio),
            "val_ratio": float(val_ratio),
            "test_ratio": float(test_ratio),
            "seed": int(split_seed),
            "strategy": "global_random",
        },
        "source_label_schema_ids": sorted(set(source_schema_ids)),
        "source_projection_modes": sorted(set(source_projection_modes)),
        "channel_supervision_summary": training_summary,
        "training_set_id": _as_text(training_set_id),
    }
    root.attrs["training_export"] = training_export

    registry_summary: Optional[Dict[str, Any]] = None
    if registry_path is not None:
        registry_summary = _register_merged_dataset_in_registry(
            registry_path=registry_path,
            merged_zarr=out_path,
            source_zarr_paths=[source.source_zarr for source in resolved_sources],
            set_id=_as_text(training_set_id),
            set_name=_as_text(training_set_name),
            training_summary=training_summary,
        )
        training_export["registry_registration"] = registry_summary
        root.attrs["training_export"] = training_export

    return {
        "out_zarr": str(out_path),
        "run_name": export_run_name,
        "total_samples": total_samples,
        "channels": int(len(target_labels)),
        "label_schema_id": target_schema_id,
        "coverage_class": supervision_summary["coverage_class"],
        "contains_only_eye_masks": supervision_summary["contains_only_eye_masks"],
        "source_count": int(len(resolved_sources)),
        "registry_registration": registry_summary,
        "channel_supervision_summary": training_summary,
        "source_subject_mask_run": source_subject_mask_run or None,
        "source_crop_run": source_crop_run or None,
    }


def export_merged_subject_mask_training_zarr(
    source_zarr: Path | str,
    out_zarr: Path | str,
    *,
    subject_run: Optional[str] = None,
    crop_run: Optional[str] = None,
    source_stage_group: str = "subject_mask_runs",
    subject_label_schema: str = "auto",
    input_format: Optional[str] = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
    split_seed: int = 123,
    run_name: Optional[str] = None,
    overwrite: bool = False,
    registry: Optional[Path] = None,
    training_set_id: Optional[str] = None,
    training_set_name: Optional[str] = None,
) -> Dict[str, Any]:
    return export_merged_subject_mask_training_zarr_from_sources(
        source_specs=[
            SubjectMergeSourceSpec(
                source_zarr=Path(source_zarr),
                subject_run=subject_run,
                crop_run=crop_run,
                stage_group=source_stage_group,
            )
        ],
        out_zarr=Path(out_zarr),
        subject_label_schema=subject_label_schema,
        input_format=input_format,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        split_seed=split_seed,
        run_name=run_name,
        overwrite=overwrite,
        registry=registry,
        training_set_id=training_set_id,
        training_set_name=training_set_name,
    )


def validate_merged_subject_mask_training_zarr(
    zarr_path: Path | str,
    *,
    expected_input_format: Optional[str] = None,
    expected_total_samples: Optional[int] = None,
    expected_label_schema_id: Optional[str] = None,
) -> Dict[str, Any]:
    root = zarr.open_group(str(Path(zarr_path).expanduser().resolve()), mode="r")
    errors: List[str] = []

    if str(root.attrs.get("zarr_purpose", "")).strip().lower() != "training":
        errors.append("root attr zarr_purpose must be 'training'.")
    if str(root.attrs.get("training_task", "")).strip().lower() != "subject_masks":
        errors.append("root attr training_task must be 'subject_masks'.")
    for group_name in ("crop_runs", "subject_mask_runs", "splits", "source_index"):
        if group_name not in root:
            errors.append(f"missing group {group_name}.")
    if errors:
        raise ValueError("Merged subject-mask zarr validation failed:\n- " + "\n- ".join(errors))

    crop_parent = root["crop_runs"]
    mask_parent = root["subject_mask_runs"]
    crop_latest = _as_text(crop_parent.attrs.get("latest"))
    mask_latest = _as_text(mask_parent.attrs.get("latest"))
    if not crop_latest or crop_latest not in crop_parent:
        errors.append("crop_runs/latest missing or points to a non-existent run.")
    if not mask_latest or mask_latest not in mask_parent:
        errors.append("subject_mask_runs/latest missing or points to a non-existent run.")
    if errors:
        raise ValueError("Merged subject-mask zarr validation failed:\n- " + "\n- ".join(errors))

    crop = crop_parent[str(crop_latest)]
    masks = mask_parent[str(mask_latest)]
    required_crop_arrays = (
        "roi_images",
        "bbox_norm_coords",
        "crop_bbox_norm_coords",
        "frame_indices",
        "detection_source",
    )
    required_mask_arrays = ("masks_roi", "target_valid_channels")
    for name in required_crop_arrays:
        if name not in crop:
            errors.append(f"missing required array crop_runs/{crop_latest}/{name}.")
    for name in required_mask_arrays:
        if name not in masks:
            errors.append(f"missing required array subject_mask_runs/{mask_latest}/{name}.")
    source_index = root["source_index"]
    required_source_arrays = (
        "source_dataset_idx",
        "source_frame_idx",
        "source_roi_idx",
        "label_origin_codes",
        "supervision_mode_codes",
        "source_dataset_id",
        "source_zarr_path",
        "source_stage_group",
        "source_run_name",
        "source_label_schema_id",
        "source_projection_mode",
    )
    for name in required_source_arrays:
        if name not in source_index:
            errors.append(f"missing required array source_index/{name}.")
    label_origin_codebook = _normalize_codebook_attr(source_index.attrs.get("label_origin_codebook"))
    if label_origin_codebook != LABEL_ORIGIN_CODES:
        errors.append(
            "source_index attr label_origin_codebook missing or does not match the stable subject-mask codebook."
        )
    supervision_mode_codebook = _normalize_codebook_attr(source_index.attrs.get("supervision_mode_codebook"))
    if supervision_mode_codebook != SUPERVISION_MODE_CODES:
        errors.append(
            "source_index attr supervision_mode_codebook missing or does not match the stable subject-mask codebook."
        )
    if errors:
        raise ValueError("Merged subject-mask zarr validation failed:\n- " + "\n- ".join(errors))

    roi = np.asarray(crop["roi_images"][:])
    bbox = np.asarray(crop["bbox_norm_coords"][:])
    crop_bbox = np.asarray(crop["crop_bbox_norm_coords"][:])
    frame_indices = np.asarray(crop["frame_indices"][:])
    detection_source = np.asarray(crop["detection_source"][:])
    masks_roi = np.asarray(masks["masks_roi"][:], dtype=np.uint8)
    target_valid = np.asarray(masks["target_valid_channels"][:], dtype=np.bool_)
    label_origin = np.asarray(source_index["label_origin_codes"][:], dtype=np.uint8)
    supervision_mode = np.asarray(source_index["supervision_mode_codes"][:], dtype=np.uint8)
    source_dataset_idx = np.asarray(source_index["source_dataset_idx"][:], dtype=np.int64)
    source_refined_row_ids = (
        np.asarray(source_index["source_refined_row_ids"][:])
        if "source_refined_row_ids" in source_index
        else None
    )
    source_detect_row_index = (
        np.asarray(source_index["source_detect_row_index"][:])
        if "source_detect_row_index" in source_index
        else None
    )

    if roi.ndim < 3:
        errors.append(f"roi_images must have shape (N, ...), got {tuple(roi.shape)}.")
    total_samples = int(roi.shape[0]) if roi.ndim >= 1 else 0
    if bbox.ndim != 2 or int(bbox.shape[0]) != total_samples or int(bbox.shape[1]) != 4:
        errors.append(f"bbox_norm_coords must have shape (N,4), got {tuple(bbox.shape)}.")
    if crop_bbox.ndim != 2 or int(crop_bbox.shape[0]) != total_samples or int(crop_bbox.shape[1]) != 4:
        errors.append(f"crop_bbox_norm_coords must have shape (N,4), got {tuple(crop_bbox.shape)}.")
    if masks_roi.ndim != 4 or int(masks_roi.shape[0]) != total_samples:
        errors.append(f"masks_roi must have shape (N,C,H,W), got {tuple(masks_roi.shape)}.")
    if target_valid.ndim != 2 or int(target_valid.shape[0]) != total_samples:
        errors.append(f"target_valid_channels must have shape (N,C), got {tuple(target_valid.shape)}.")
    if (
        target_valid.ndim == 2
        and masks_roi.ndim == 4
        and int(target_valid.shape[1]) != int(masks_roi.shape[1])
    ):
        errors.append(
            "masks_roi.shape[1] must equal target_valid_channels.shape[1] "
            f"({masks_roi.shape[1]} != {target_valid.shape[1]})."
        )
    if label_origin.shape != target_valid.shape:
        errors.append("label_origin_codes shape must match target_valid_channels.")
    elif not set(np.unique(label_origin).astype(int).tolist()).issubset(set(LABEL_ORIGIN_CODES.values())):
        errors.append("label_origin_codes contains values outside label_origin_codebook.")
    if supervision_mode.shape != target_valid.shape:
        errors.append("supervision_mode_codes shape must match target_valid_channels.")
    elif not set(np.unique(supervision_mode).astype(int).tolist()).issubset(set(SUPERVISION_MODE_CODES.values())):
        errors.append("supervision_mode_codes contains values outside supervision_mode_codebook.")

    if frame_indices.ndim != 1 or int(frame_indices.shape[0]) != total_samples:
        errors.append(f"frame_indices must be 1D length N, got {tuple(frame_indices.shape)}.")
    elif not np.array_equal(frame_indices.astype(np.int64, copy=False), np.arange(total_samples, dtype=np.int64)):
        errors.append("frame_indices must be local 0..N-1 indexing.")

    if detection_source.ndim != 1 or int(detection_source.shape[0]) != total_samples:
        errors.append("detection_source must be 1D length N.")
    if source_dataset_idx.ndim != 1 or int(source_dataset_idx.shape[0]) != total_samples:
        errors.append("source_dataset_idx must be 1D length N.")
    for opt_name, opt_arr in (
        ("source_refined_row_ids", source_refined_row_ids),
        ("source_detect_row_index", source_detect_row_index),
    ):
        if opt_arr is None:
            continue
        if opt_arr.ndim != 1 or int(opt_arr.shape[0]) != total_samples:
            errors.append(f"{opt_name} must be 1D length N.")
        elif not np.issubdtype(opt_arr.dtype, np.integer):
            errors.append(f"{opt_name} must be integer dtype.")

    source_count = int(source_index["source_dataset_id"].shape[0])
    if source_dataset_idx.ndim == 1 and source_count > 0:
        if int(np.min(source_dataset_idx)) < 0 or int(np.max(source_dataset_idx)) >= source_count:
            errors.append("source_dataset_idx contains out-of-range values.")

    if np.any(~target_valid & (supervision_mode != 0)):
        errors.append("supervision_mode_codes must equal 0 where target_valid_channels is false.")
    if np.any(target_valid & (supervision_mode == 0)):
        errors.append("supervision_mode_codes must be non-zero where target_valid_channels is true.")

    training_export = root.attrs.get("training_export")
    if not isinstance(training_export, dict):
        errors.append("root attr training_export must be a dict.")
    else:
        label_schema_id = _as_text(training_export.get("label_schema_id"))
        if label_schema_id not in TARGET_SCHEMAS:
            errors.append("training_export.label_schema_id missing or invalid.")
        elif masks_roi.ndim == 4 and int(masks_roi.shape[1]) != int(len(TARGET_SCHEMAS[label_schema_id])):
            errors.append(
                "training_export.label_schema_id/masks_roi channel mismatch "
                f"({label_schema_id} => {len(TARGET_SCHEMAS[label_schema_id])} channels, got {masks_roi.shape[1]})."
            )
        mask_labels = training_export.get("mask_labels")
        if not isinstance(mask_labels, list) or len(mask_labels) != int(masks_roi.shape[1]):
            errors.append("training_export.mask_labels missing or invalid.")

        summary = training_export.get("channel_supervision_summary")
        if not isinstance(summary, dict):
            errors.append("training_export.channel_supervision_summary missing or invalid.")
        else:
            supervised_counts = summary.get("supervised_row_counts")
            if not isinstance(supervised_counts, dict):
                errors.append("channel_supervision_summary.supervised_row_counts missing or invalid.")
            else:
                for channel_idx, label in enumerate(mask_labels or []):
                    expected_count = int(np.sum(target_valid[:, channel_idx]))
                    actual_count = supervised_counts.get(str(label))
                    if actual_count is not None and int(actual_count) != expected_count:
                        errors.append(
                            f"supervised_row_counts mismatch for {label} ({actual_count} != {expected_count})."
                        )

    if expected_total_samples is not None and int(expected_total_samples) != total_samples:
        errors.append(f"total sample mismatch ({total_samples} != expected {int(expected_total_samples)}).")
    if expected_input_format is not None:
        expected_input = _normalize_input_format(expected_input_format)
        if expected_input == "rgb" and not (roi.ndim == 4 and int(roi.shape[-1]) == 3):
            errors.append("roi_images must be (N,H,W,3) for rgb input format.")
        if expected_input == "gray" and roi.ndim == 4 and int(roi.shape[-1]) == 3:
            errors.append("roi_images appears rgb but expected gray input format.")
    if expected_label_schema_id is not None:
        expected_schema = _normalize_label_schema(expected_label_schema_id)
        actual_schema = (
            _as_text(root.attrs.get("training_export", {}).get("label_schema_id"))
            if isinstance(root.attrs.get("training_export"), dict)
            else None
        )
        if expected_schema != actual_schema:
            errors.append(f"label schema mismatch ({actual_schema!r} != expected {expected_schema!r}).")

    splits = root["splits"]
    split_counts: Dict[str, int] = {}
    seen_indices: List[int] = []
    for name in ("train_indices", "val_indices", "test_indices"):
        arr = np.asarray(splits[name][:], dtype=np.int64)
        if arr.ndim != 1:
            errors.append(f"{name} must be 1D.")
        if arr.size > 0:
            if int(np.min(arr)) < 0 or int(np.max(arr)) >= total_samples:
                errors.append(f"{name} contains out-of-range indices.")
            if len(np.unique(arr)) != int(arr.shape[0]):
                errors.append(f"{name} contains duplicate indices.")
        split_counts[name.removesuffix("_indices")] = int(arr.shape[0])
        seen_indices.extend(int(v) for v in arr.tolist())
    if len(seen_indices) != len(set(seen_indices)):
        errors.append("split indices must be pairwise disjoint.")
    if len(seen_indices) != total_samples:
        errors.append("split indices must exactly cover all rows.")

    if errors:
        raise ValueError("Merged subject-mask zarr validation failed:\n- " + "\n- ".join(errors))

    summary = training_export.get("channel_supervision_summary") if isinstance(training_export, dict) else {}
    return {
        "total_samples": total_samples,
        "channels": int(masks_roi.shape[1]),
        "source_count": source_count,
        "split_counts": split_counts,
        "label_schema_id": (
            _as_text(training_export.get("label_schema_id"))
            if isinstance(training_export, dict)
            else None
        ),
        "coverage_class": summary.get("coverage_class") if isinstance(summary, dict) else None,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Export merged subject-mask training zarr.")
    parser.add_argument(
        "sources",
        nargs="+",
        type=Path,
        help="One or more source zarr paths containing subject-mask source runs.",
    )
    parser.add_argument("out_zarr", type=Path, help="Output merged training zarr path.")
    parser.add_argument("--subject-run", type=str, help="Optional source run name to export from each source.")
    parser.add_argument(
        "--source-stage-group",
        choices=list(SUBJECT_STAGE_CHOICES),
        default="subject_mask_runs",
        help="Source stage group to export from each source.",
    )
    parser.add_argument("--crop-run", type=str, help="Optional crop_runs/<run> to use from each source.")
    parser.add_argument(
        "--subject-label-schema",
        default="auto",
        choices=list(LABEL_SCHEMA_CHOICES),
        help="Target subject-mask label schema for the merged dataset.",
    )
    parser.add_argument("--input-format", choices=["gray", "rgb"], help="Optional explicit input format.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Training split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.0, help="Test split ratio.")
    parser.add_argument("--split-seed", type=int, default=123, help="Random seed for global split assignment.")
    parser.add_argument("--run-name", type=str, help="Optional merged run name.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output zarr.")
    parser.add_argument("--registry", type=Path, help="Optional registry SQLite path for merged-export registration.")
    parser.add_argument("--training-set-id", type=str, help="Optional training set identifier for registry linkage.")
    parser.add_argument("--training-set-name", type=str, help="Optional training set display name.")
    args = parser.parse_args(argv)

    source_specs = [
        SubjectMergeSourceSpec(
            source_zarr=source,
            subject_run=args.subject_run,
            crop_run=args.crop_run,
            stage_group=args.source_stage_group,
        )
        for source in args.sources
    ]
    summary = export_merged_subject_mask_training_zarr_from_sources(
        source_specs=source_specs,
        out_zarr=args.out_zarr,
        subject_label_schema=args.subject_label_schema,
        input_format=args.input_format,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_seed=args.split_seed,
        run_name=args.run_name,
        overwrite=bool(args.overwrite),
        registry=args.registry,
        training_set_id=args.training_set_id,
        training_set_name=args.training_set_name,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
