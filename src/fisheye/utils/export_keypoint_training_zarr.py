#!/usr/bin/env python3
"""Export keypoint-training Zarr datasets from a training manifest.

Merged mode is the initial supported mode:
- one merged Zarr for the full training set
- fixed split arrays (train/val/test)
- source-index traceability arrays
- merged config/manifest emission for train_keypoints
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml
import zarr
from zarr.core.dtype import VariableLengthUTF8

from fisheye.registry.db import Registry, RegistryPaths
from fisheye.pose.schema import (
    resolve_keypoint_labels_from_attrs,
    resolve_skeleton_identity_from_attrs,
)
from fisheye.shared.batch_logging import utc_now
from fisheye.shared.frame_flags import resolve_row_identity_arrays
from fisheye.shared.detect_reason_codec import read_reason_labels
from fisheye.shared.type_conversions import normalize_attr as _shared_as_text
from fisheye.utils import prepare_keypoint_training_from_registry as prepare_pose
from fisheye.utils.system import build_invocation_record

try:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        Progress,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
except Exception:  # pragma: no cover - rich is optional
    Console = None  # type: ignore
    Progress = None  # type: ignore
    TextColumn = None  # type: ignore
    BarColumn = None  # type: ignore
    TaskProgressColumn = None  # type: ignore
    TimeElapsedColumn = None  # type: ignore
    TimeRemainingColumn = None  # type: ignore


@dataclass
class PoseMergeSourceSpec:
    ordinal: int
    dataset_name: str
    dataset_id: str
    source_zarr: Path
    source_crop_run: str
    keypoint_run: str
    source_type_resolved: str
    source_sample_count: int
    sample_count: int
    selected_indices: np.ndarray
    row_gate_policy: str
    row_gate_refined_run: Optional[str]
    row_gate_selected: int
    row_gate_total: int
    row_gate_raw_success_true: int
    row_gate_usable_true: Optional[int]
    roi_path: str
    bbox_path: str
    keypoints_path: str
    success_path: str
    frame_indices_path: Optional[str]
    detection_source_path: Optional[str]
    roi_shape: Tuple[int, ...]
    roi_dtype: np.dtype
    roi_chunks: Optional[Tuple[int, ...]]
    roi_pixel_contract: Optional[Dict[str, Any]]
    roi_pixel_contract_name: Optional[str]
    keypoint_shape: Tuple[int, ...]
    keypoint_dtype: np.dtype
    keypoint_labels: List[str]
    skeleton_id: Optional[str]
    kpt_shape: Optional[Tuple[int, int]]
    skeleton_signature: str
    dish_design: Optional[str]
    canvas_name: Optional[str]
    rig_id: Optional[str]
    box_only_selected_mask: Optional[np.ndarray] = None
    row_gate_box_only_true: Optional[int] = None
    row_gate_box_only_selected: int = 0


@dataclass
class PoseMergeResult:
    run_name: str
    total_samples: int
    total_successful: int
    total_failed: int
    detection_source_counts: Dict[str, int]
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    source_specs: List[PoseMergeSourceSpec]
    input_format: str
    source_type: str
    keypoint_shape: Tuple[int, ...]
    keypoint_labels: List[str]
    skeleton_id: Optional[str]
    kpt_shape: Optional[Tuple[int, int]]
    skeleton_signature: str
    row_gate_policy: str
    row_gate_counts: Dict[str, int]
    keypoint_supervision_counts: Dict[str, int]


_utc_now = utc_now


def _sha256(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _normalize_manifest_stem(value: str) -> str:
    text = str(value).strip()
    while text.endswith(".manifest"):
        text = text[: -len(".manifest")]
    return text


def _ensure_suffix(value: str, suffix: str) -> str:
    text = str(value).strip()
    return text if text.endswith(suffix) else f"{text}{suffix}"


_as_text = _shared_as_text


BOX_ONLY_REASON_TAG = "fish_present_no_keypoints"


def _collapse_source_type_values(values: Sequence[Any], *, fallback: str) -> str:
    normalized = sorted(
        {
            str(value).strip().lower()
            for value in values
            if value is not None and str(value).strip()
        }
    )
    if not normalized:
        return str(fallback).strip().lower()
    if len(normalized) == 1:
        return normalized[0]
    return "mixed"


def _reason_has_tag(reason_value: object, tag: str) -> bool:
    reason_text = _as_text(reason_value)
    if reason_text is None:
        return False
    tag_norm = str(tag).strip().lower()
    if not tag_norm:
        return False
    tokens = [
        token.strip().lower()
        for token in reason_text.replace(",", "|").replace(";", "|").split("|")
        if token is not None
    ]
    return tag_norm in {token for token in tokens if token}


def _read_reason_labels_safe(group: zarr.Group) -> Optional[np.ndarray]:
    try:
        labels = read_reason_labels(group)
    except Exception:
        labels = None
    if labels is None:
        return None
    return np.asarray(labels, dtype=object)


def _resolve_roi_pixel_contract(crop_group: zarr.Group) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    raw_contract = crop_group.attrs.get("roi_pixel_contract")
    contract = dict(raw_contract) if isinstance(raw_contract, Mapping) else None
    contract_name = (
        _as_text(crop_group.attrs.get("roi_pixel_contract_name"))
        or (_as_text(contract.get("name")) if contract is not None else None)
    )
    return contract, contract_name


def _resolve_required_roi_pixel_contract_name(
    *,
    manifest_payload: Mapping[str, Any],
    dataset_payload: Mapping[str, Any],
) -> Optional[str]:
    return (
        _as_text(dataset_payload.get("required_roi_pixel_contract_name"))
        or _as_text(dataset_payload.get("roi_pixel_contract_name"))
        or _as_text(manifest_payload.get("required_roi_pixel_contract_name"))
    )


def _clean_slug(value: Optional[str], fallback: str) -> str:
    text = _as_text(value) or fallback
    cleaned = "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in text.strip())
    return cleaned or fallback


def _parse_split_spec(spec: str) -> Tuple[float, float, float]:
    parts = [part.strip() for part in str(spec).split("/") if part.strip()]
    if len(parts) not in {2, 3}:
        raise ValueError(f"Invalid --split '{spec}'. Use train/val or train/val/test (e.g. 0.8/0.2).")
    values = [float(part) for part in parts]
    if any(value < 0.0 for value in values):
        raise ValueError(f"Invalid --split '{spec}': ratios must be non-negative.")
    total = sum(values)
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(f"Invalid --split '{spec}': ratios must sum to 1.0 (got {total:.6f}).")
    if len(values) == 2:
        return values[0], values[1], 0.0
    return values[0], values[1], values[2]


def _compute_split_indices(
    total: int,
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if total < 0:
        raise ValueError(f"Invalid sample count: {total}")
    order = np.arange(total, dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(order)

    if total == 0:
        return order.copy(), order.copy(), order.copy()

    train_count = int(round(total * train_ratio))
    val_count = int(round(total * val_ratio))
    train_count = max(0, min(train_count, total))
    val_count = max(0, min(val_count, total - train_count))
    test_count = total - train_count - val_count
    if test_ratio <= 0.0:
        val_count = total - train_count
        test_count = 0

    train_idx = order[:train_count]
    val_idx = order[train_count : train_count + val_count]
    test_idx = order[train_count + val_count : train_count + val_count + test_count]
    return train_idx, val_idx, test_idx


def _normalize_chunks(chunks: Optional[Tuple[int, ...]], shape: Tuple[int, ...]) -> Tuple[int, ...]:
    if not shape:
        return ()
    if chunks is None:
        first = 32 if shape[0] >= 32 else max(1, shape[0])
        return (first, *shape[1:])
    normalized: List[int] = []
    for idx, dim in enumerate(shape):
        if idx < len(chunks) and chunks[idx]:
            value = int(chunks[idx])
        else:
            value = int(dim)
        value = max(1, min(value, int(dim)))
        normalized.append(value)
    return tuple(normalized)


def _copy_progress(total: int) -> Optional["Progress"]:
    if total <= 0 or Progress is None:
        return None
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.completed}/{task.total} samples"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=Console() if Console is not None else None,
    )


def _copy_rows(
    src_array: zarr.Array,
    dest_array: zarr.Array,
    dest_start: int,
    count: int,
    *,
    batch_size: int = 128,
    on_copied: Optional[Callable[[int], None]] = None,
) -> None:
    if count <= 0:
        return
    batch_size = max(1, int(batch_size))
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        dest_array[dest_start + start : dest_start + stop] = src_array[start:stop]
        if on_copied is not None:
            on_copied(stop - start)


def _copy_rows_indexed(
    src_array: zarr.Array,
    dest_array: zarr.Array,
    dest_start: int,
    indices: np.ndarray,
    *,
    batch_size: int = 128,
    on_copied: Optional[Callable[[int], None]] = None,
) -> None:
    if indices.size <= 0:
        return
    batch_size = max(1, int(batch_size))
    for start in range(0, int(indices.size), batch_size):
        stop = min(start + batch_size, int(indices.size))
        idx_batch = indices[start:stop]
        dest_array[dest_start + start : dest_start + stop] = src_array[idx_batch]
        if on_copied is not None:
            on_copied(stop - start)


def _take_crop_or_source_rows(
    values: np.ndarray,
    selected: np.ndarray,
    *,
    source_detect_row_index: np.ndarray,
    source_sample_count: int,
    source_zarr: Path,
    array_name: str,
) -> np.ndarray:
    if int(values.shape[0]) == int(source_sample_count):
        return values[selected]
    if int(source_detect_row_index.shape[0]) != int(source_sample_count):
        raise ValueError(
            f"{source_zarr}: source_detect_row_index length mismatch for {array_name} "
            f"({source_detect_row_index.shape[0]} != {source_sample_count})."
        )
    mapped = np.asarray(source_detect_row_index[selected], dtype=np.int64)
    if mapped.size and (int(mapped.min()) < 0 or int(mapped.max()) >= int(values.shape[0])):
        raise ValueError(
            f"{source_zarr}: source_detect_row_index points outside {array_name} "
            f"(max index {int(mapped.max())}, rows {int(values.shape[0])})."
        )
    return values[mapped]


def _write_string_array(group: zarr.Group, name: str, values: Sequence[str]) -> None:
    labels = np.asarray([str(value) for value in values], dtype=object)
    chunk_size = max(1, min(1024, int(labels.shape[0])))
    arr = group.create_array(
        name,
        shape=(int(labels.shape[0]),),
        dtype=VariableLengthUTF8(),
        fill_value="",
        chunks=(chunk_size,),
        overwrite=True,
    )
    arr[:] = labels


def _json_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(item) for item in raw if item]
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", "ignore")
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            payload = json.loads(text)
        except Exception:
            return []
        if isinstance(payload, list):
            return [str(item) for item in payload if item]
    return []


def _json_dict(raw: Any) -> Optional[Dict[str, Any]]:
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", "ignore")
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            payload = json.loads(text)
        except Exception:
            return None
        if isinstance(payload, dict):
            return payload
    return None


def _as_mapping(value: Any) -> Optional[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", "ignore")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            payload = json.loads(text)
        except Exception:
            return None
        if isinstance(payload, Mapping):
            return payload
    return None


def _normalize_kpt_shape(value: Any) -> Optional[Tuple[int, int]]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        k = int(value[0])
        d = int(value[1])
    except Exception:
        return None
    if k <= 0 or d <= 0:
        return None
    return (k, d)


def _format_kpt_shape(value: Optional[Tuple[int, int]]) -> str:
    if value is None:
        return "missing"
    return f"[{value[0]},{value[1]}]"


def _format_skeleton_signature(*, skeleton_id: Optional[str], kpt_shape: Optional[Tuple[int, int]]) -> str:
    return f"skeleton_id={skeleton_id or 'missing'}, kpt_shape={_format_kpt_shape(kpt_shape)}"


def _resolve_dataset_skeleton_identity(
    *,
    dataset_payload: Dict[str, Any],
    kp_group: zarr.Group,
    source_zarr: Path,
    keypoint_run: str,
    keypoint_count: int,
    manifest_skeleton_id: Optional[str],
    manifest_kpt_shape: Optional[Tuple[int, int]],
) -> Tuple[Optional[str], Tuple[int, int], str]:
    runtime_identity = resolve_skeleton_identity_from_attrs(
        dict(kp_group.attrs),
        keypoint_count=int(keypoint_count),
    )
    dataset_identity = resolve_skeleton_identity_from_attrs(dataset_payload)
    manifest_identity = resolve_skeleton_identity_from_attrs(
        {
            "skeleton_id": manifest_skeleton_id,
            "kpt_shape": list(manifest_kpt_shape) if manifest_kpt_shape is not None else None,
        }
    )

    skeleton_id = (
        runtime_identity.skeleton_id
        or dataset_identity.skeleton_id
        or manifest_identity.skeleton_id
    )
    runtime_kpt_shape = runtime_identity.kpt_shape
    preferred_kpt_shape = dataset_identity.kpt_shape or manifest_identity.kpt_shape

    if runtime_kpt_shape is not None and runtime_kpt_shape[0] != int(keypoint_count):
        raise ValueError(
            f"{source_zarr}: keypoint run '{keypoint_run}' resolved "
            f"{_format_skeleton_signature(skeleton_id=skeleton_id, kpt_shape=runtime_kpt_shape)} "
            f"but keypoints_roi has K={keypoint_count}."
        )
    if preferred_kpt_shape is not None and preferred_kpt_shape[0] != keypoint_count:
        raise ValueError(
            f"{source_zarr}: manifest/dataset pose_schema kpt_shape={list(preferred_kpt_shape)} "
            f"does not match keypoint run '{keypoint_run}' keypoints_roi K={keypoint_count}."
        )

    resolved_dims = None
    if preferred_kpt_shape is not None:
        resolved_dims = int(preferred_kpt_shape[1])
    elif runtime_kpt_shape is not None:
        resolved_dims = int(runtime_kpt_shape[1])
    if resolved_dims is None or resolved_dims <= 0:
        resolved_dims = 2
    resolved_kpt_shape = (int(keypoint_count), int(resolved_dims))

    signature = _format_skeleton_signature(skeleton_id=skeleton_id, kpt_shape=resolved_kpt_shape)
    return skeleton_id, resolved_kpt_shape, signature


def _resolve_dataset_keypoint_labels(
    *,
    manifest_payload: Dict[str, Any],
    dataset_payload: Dict[str, Any],
    annotation_group: zarr.Group,
    source_zarr: Path,
    keypoint_run: str,
    keypoint_count: int,
) -> List[str]:
    runtime_labels = list(
        resolve_keypoint_labels_from_attrs(
            dict(annotation_group.attrs),
            keypoint_count=int(keypoint_count),
        )
    )
    dataset_labels = list(
        resolve_keypoint_labels_from_attrs(
            dataset_payload,
            keypoint_count=int(keypoint_count),
        )
    )
    manifest_labels = list(
        resolve_keypoint_labels_from_attrs(
            {
                "keypoint_labels": manifest_payload.get("keypoint_labels"),
                "pose_schema": manifest_payload.get("pose_schema"),
            },
            keypoint_count=int(keypoint_count),
        )
    )

    selected = runtime_labels or dataset_labels or manifest_labels
    if not selected:
        raise ValueError(
            f"{source_zarr}: keypoint run '{keypoint_run}' is missing keypoint label metadata "
            "(expected keypoint_labels or pose_schema nodes on the run, dataset manifest row, or manifest pose_schema)."
        )

    for source_name, labels in (
        ("dataset", dataset_labels),
        ("manifest", manifest_labels),
    ):
        if labels and labels != selected:
            raise ValueError(
                f"{source_zarr}: {source_name} keypoint_labels {labels} do not match "
                f"resolved run labels {selected} for '{keypoint_run}'."
            )
    return [str(label) for label in selected]


def _extract_identity(dataset_payload: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    def _clean_text(value: Any) -> Optional[str]:
        text = _as_text(value)
        return text if text else None

    dish = _clean_text(dataset_payload.get("dish_design"))
    canvas = _clean_text(dataset_payload.get("canvas_name"))
    rig = _clean_text(dataset_payload.get("rig_id"))

    provenance = dataset_payload.get("provenance")
    if isinstance(provenance, dict):
        if not dish:
            arena = provenance.get("arena")
            if isinstance(arena, dict):
                dish = _clean_text(arena.get("dish_design"))
        rig_info = provenance.get("rig_info")
        if isinstance(rig_info, dict):
            if not canvas:
                canvas = _clean_text(rig_info.get("canvas_name"))
            if not rig:
                rig = _clean_text(rig_info.get("rig_id"))
    return dish, canvas, rig


def _resolve_row_gate_selection(
    *,
    root: zarr.Group,
    dataset_payload: Dict[str, Any],
    keypoint_run: str,
    sample_count: int,
    success_arr: zarr.Array,
    row_gate_policy: str,
    source_zarr: Path,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    policy = str(row_gate_policy).strip().lower()
    if policy not in {"auto", "refined_usable", "raw_success", "raw_success_plus_box_only"}:
        raise ValueError(f"Unsupported row gate policy '{row_gate_policy}'.")

    raw_success = np.asarray(success_arr[:], dtype=np.bool_)
    if int(raw_success.shape[0]) != int(sample_count):
        raise ValueError(
            f"{source_zarr}: detection_success/roi row mismatch "
            f"({raw_success.shape[0]} != {sample_count})."
        )

    refined_parent = root.get("refined_keypoints_runs") or root.get("keypoints_refined_runs")
    refined_candidates: List[Tuple[str, str, zarr.Group]] = []
    if refined_parent is not None:
        requested_refined = _as_text(
            dataset_payload.get("refined_keypoint_run")
            or dataset_payload.get("quality_registry_refined_run")
        )
        if requested_refined and requested_refined in refined_parent:
            refined_group = refined_parent[requested_refined]
            source_run = _as_text(
                refined_group.attrs.get("source_keypoints_run") or refined_group.attrs.get("source_keypoint_run")
            )
            if source_run == keypoint_run:
                ts = _as_text(refined_group.attrs.get("created_utc") or refined_group.attrs.get("timestamp_utc")) or ""
                refined_candidates.append((ts, requested_refined, refined_group))

        if not refined_candidates:
            for run_name in refined_parent.group_keys():
                refined_group = refined_parent[run_name]
                source_run = _as_text(
                    refined_group.attrs.get("source_keypoints_run") or refined_group.attrs.get("source_keypoint_run")
                )
                if source_run != keypoint_run:
                    continue
                ts = _as_text(refined_group.attrs.get("created_utc") or refined_group.attrs.get("timestamp_utc")) or ""
                refined_candidates.append((ts, str(run_name), refined_group))
            refined_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)

    selected_policy = "raw_success"
    selected_refined_run = None
    selected_mask = raw_success.copy()
    usable_true = None
    box_only_mask = np.zeros(int(sample_count), dtype=np.bool_)

    if policy in {"auto", "refined_usable"}:
        for _ts, refined_run, refined_group in refined_candidates:
            usable_arr = refined_group.get("usable_keypoints")
            if usable_arr is None:
                continue
            usable_mask = np.asarray(usable_arr[:], dtype=np.bool_)
            if int(usable_mask.shape[0]) != int(sample_count):
                raise ValueError(
                    f"{source_zarr}: refined run '{refined_run}' usable_keypoints length mismatch "
                    f"({usable_mask.shape[0]} != {sample_count})."
                )
            selected_policy = "refined_usable"
            selected_refined_run = refined_run
            selected_mask = usable_mask
            usable_true = int(np.sum(usable_mask))
            break

        if policy == "refined_usable" and selected_policy != "refined_usable":
            raise ValueError(
                f"{source_zarr}: row_gate_policy=refined_usable requested but no compatible "
                "refined usable_keypoints mask was found for the selected keypoint run."
            )

    if policy == "raw_success_plus_box_only":
        selected_policy = "raw_success_plus_box_only"
        for _ts, refined_run, refined_group in refined_candidates:
            reason_values = _read_reason_labels_safe(refined_group)
            if reason_values is None:
                continue
            if int(reason_values.shape[0]) != int(sample_count):
                raise ValueError(
                    f"{source_zarr}: refined run '{refined_run}' reason label length mismatch "
                    f"({reason_values.shape[0]} != {sample_count})."
                )
            tagged_mask = np.asarray(
                [_reason_has_tag(value, BOX_ONLY_REASON_TAG) for value in reason_values],
                dtype=np.bool_,
            )
            tagged_mask &= ~raw_success
            box_only_mask = tagged_mask
            selected_refined_run = refined_run
            break
        selected_mask = np.logical_or(raw_success, box_only_mask)

    selected_indices = np.where(selected_mask)[0].astype(np.int64, copy=False)
    selected_box_only_mask = box_only_mask[selected_indices].astype(np.bool_, copy=False)
    stats = {
        "policy": selected_policy,
        "refined_run": selected_refined_run,
        "selected": int(selected_indices.shape[0]),
        "total": int(sample_count),
        "raw_success_true": int(np.sum(raw_success)),
        "usable_true": usable_true,
        "box_only_true": int(np.sum(box_only_mask)),
        "box_only_selected": int(np.sum(selected_box_only_mask)),
    }
    return selected_indices, selected_box_only_mask, stats


def _discover_merge_sources(
    manifest_payload: Dict[str, Any],
    *,
    expected_input_format: str,
    row_gate_policy: str,
) -> Tuple[List[PoseMergeSourceSpec], Dict[str, Any]]:
    datasets = manifest_payload.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("Manifest has no datasets.")

    specs: List[PoseMergeSourceSpec] = []
    roi_shape: Optional[Tuple[int, ...]] = None
    roi_dtype: Optional[np.dtype] = None
    roi_chunks: Optional[Tuple[int, ...]] = None
    keypoint_shape: Optional[Tuple[int, ...]] = None
    keypoint_dtype: Optional[np.dtype] = None
    keypoint_labels: Optional[List[str]] = None
    keypoint_label_members: Dict[Tuple[str, ...], List[str]] = {}
    manifest_pose_schema = _as_mapping(manifest_payload.get("pose_schema"))
    manifest_skeleton_id = (
        _as_text(manifest_payload.get("skeleton_id"))
        or (_as_text(manifest_pose_schema.get("skeleton_id")) if manifest_pose_schema is not None else None)
    )
    manifest_kpt_shape = (
        _normalize_kpt_shape(manifest_payload.get("kpt_shape"))
        or (_normalize_kpt_shape(manifest_pose_schema.get("kpt_shape")) if manifest_pose_schema is not None else None)
    )
    selected_skeleton_identity: Optional[Tuple[Optional[str], Tuple[int, int]]] = None
    selected_skeleton_signature: Optional[str] = None
    skeleton_signature_members: Dict[str, List[str]] = {}

    for ordinal, dataset in enumerate(datasets):
        if not isinstance(dataset, dict):
            raise ValueError(f"datasets[{ordinal}] is not an object.")

        zarr_path_raw = dataset.get("zarr_path")
        if not zarr_path_raw:
            raise ValueError(f"datasets[{ordinal}] is missing zarr_path.")
        source_zarr = Path(str(zarr_path_raw))
        root = zarr.open_group(str(source_zarr), mode="r", use_consolidated=False)

        kp_parent = root.get("keypoints_runs")
        refined_parent_name: Optional[str] = None
        if "refined_keypoints_runs" in root:
            refined_parent_name = "refined_keypoints_runs"
        elif "keypoints_refined_runs" in root:
            refined_parent_name = "keypoints_refined_runs"

        keypoint_run = _as_text(dataset.get("keypoint_run_resolved") or dataset.get("keypoint_run"))
        if not keypoint_run:
            latest = _as_text(kp_parent.attrs.get("latest")) if kp_parent is not None else None
            if latest:
                keypoint_run = latest
        if not keypoint_run:
            raise ValueError(f"{source_zarr}: unable to resolve keypoint run.")
        kp_group = kp_parent[keypoint_run] if kp_parent is not None and keypoint_run in kp_parent else None

        annotation_group: Optional[zarr.Group] = kp_group
        annotation_run = keypoint_run
        annotation_parent_name = "keypoints_runs" if kp_group is not None else None
        keypoints_path: Optional[str] = None
        success_path: Optional[str] = None
        if kp_group is not None:
            keypoints_path = f"keypoints_runs/{keypoint_run}/keypoints_roi"
            success_path = f"keypoints_runs/{keypoint_run}/detection_success"
        else:
            keypoints_path = _as_text(dataset.get("keypoints_array_path"))
            success_path = _as_text(dataset.get("detection_success_path"))
            annotation_parent_name = _as_text(dataset.get("annotation_source_parent"))
            annotation_run = (
                _as_text(dataset.get("annotation_source_run"))
                or _as_text(dataset.get("refined_keypoint_run"))
                or annotation_run
            )
            if not keypoints_path or not success_path or not annotation_parent_name or not annotation_run:
                raise ValueError(
                    f"{source_zarr}: keypoint run '{keypoint_run}' not found in keypoints_runs and "
                    "manifest does not provide refined annotation paths."
                )
            group_path = f"{annotation_parent_name}/{annotation_run}"
            if group_path not in root:
                raise ValueError(f"{source_zarr}: annotation group '{group_path}' not found.")
            annotation_group = root[group_path]
        dataset_id = (
            _as_text(dataset.get("dataset_id"))
            or _as_text(dataset.get("session_uuid"))
            or f"{_clean_slug(str(dataset.get('name') or source_zarr.stem), 'dataset')}_{ordinal}"
        )

        source_crop_run = _as_text(dataset.get("source_crop_run"))
        if not source_crop_run and kp_group is not None:
            source_crop_run = _as_text(kp_group.attrs.get("source_crop_run"))
        if not source_crop_run and annotation_group is not None:
            source_crop_run = _as_text(annotation_group.attrs.get("source_crop_run"))
        if not source_crop_run:
            raise ValueError(
                f"{source_zarr}: keypoint run '{keypoint_run}' missing source_crop_run; "
                "refusing ambiguous crop fallback."
            )

        crop_parent = root.get("crop_runs")
        if crop_parent is None or source_crop_run not in crop_parent:
            raise ValueError(f"{source_zarr}: crop run '{source_crop_run}' not found in crop_runs.")
        crop_group = crop_parent[source_crop_run]
        roi_pixel_contract, roi_pixel_contract_name = _resolve_roi_pixel_contract(crop_group)
        required_roi_pixel_contract_name = _resolve_required_roi_pixel_contract_name(
            manifest_payload=manifest_payload,
            dataset_payload=dataset,
        )
        if required_roi_pixel_contract_name and roi_pixel_contract_name != required_roi_pixel_contract_name:
            raise ValueError(
                f"{source_zarr}: crop run '{source_crop_run}' ROI pixel contract mismatch "
                f"({roi_pixel_contract_name!r} != {required_roi_pixel_contract_name!r})."
            )

        if "roi_images" not in crop_group:
            raise ValueError(f"{source_zarr}: crop run '{source_crop_run}' missing roi_images.")
        if "bbox_norm_coords" not in crop_group:
            raise ValueError(f"{source_zarr}: crop run '{source_crop_run}' missing bbox_norm_coords.")
        if keypoints_path not in root:
            raise ValueError(f"{source_zarr}: keypoints array '{keypoints_path}' not found.")
        if success_path not in root:
            raise ValueError(f"{source_zarr}: success array '{success_path}' not found.")

        roi_arr = crop_group["roi_images"]
        bbox_arr = crop_group["bbox_norm_coords"]
        keypoints_arr = root[keypoints_path]
        success_arr = root[success_path]

        source_sample_count = int(roi_arr.shape[0])
        if int(bbox_arr.shape[0]) != source_sample_count and "source_detect_row_index" not in crop_group:
            raise ValueError(
                f"{source_zarr}: bbox/roi row mismatch ({bbox_arr.shape[0]} != {source_sample_count}) "
                "and crop run has no source_detect_row_index for row-map gather."
            )
        if int(keypoints_arr.shape[0]) != source_sample_count:
            raise ValueError(
                f"{source_zarr}: keypoints/roi row mismatch "
                f"({keypoints_arr.shape[0]} != {source_sample_count})."
            )
        if int(success_arr.shape[0]) != source_sample_count:
            raise ValueError(
                f"{source_zarr}: detection_success/roi row mismatch "
                f"({success_arr.shape[0]} != {source_sample_count})."
            )

        if len(keypoints_arr.shape) != 3 or int(keypoints_arr.shape[2]) != 2:
            raise ValueError(
                f"{source_zarr}: keypoints_roi must have shape (N, K, 2), got {tuple(keypoints_arr.shape)}."
            )

        dataset_input_format = str(dataset.get("input_format") or expected_input_format).strip().lower()
        if dataset_input_format != expected_input_format:
            raise ValueError(
                f"{source_zarr}: input_format mismatch ({dataset_input_format} != {expected_input_format})."
            )

        dataset_roi_shape = tuple(int(v) for v in roi_arr.shape[1:])
        dataset_roi_dtype = np.dtype(roi_arr.dtype)
        dataset_roi_chunks = tuple(int(v) for v in roi_arr.chunks) if roi_arr.chunks else None
        dataset_keypoint_shape = tuple(int(v) for v in keypoints_arr.shape[1:])
        dataset_keypoint_dtype = np.dtype(keypoints_arr.dtype)

        selected_indices, selected_box_only_mask, gate_stats = _resolve_row_gate_selection(
            root=root,
            dataset_payload=dataset,
            keypoint_run=keypoint_run,
            sample_count=source_sample_count,
            success_arr=success_arr,
            row_gate_policy=row_gate_policy,
            source_zarr=source_zarr,
        )

        if roi_shape is None:
            roi_shape = dataset_roi_shape
            roi_dtype = dataset_roi_dtype
            roi_chunks = dataset_roi_chunks
        else:
            if dataset_roi_shape != roi_shape:
                raise ValueError(f"{source_zarr}: roi shape mismatch {dataset_roi_shape} != {roi_shape}.")
            if dataset_roi_dtype != roi_dtype:
                raise ValueError(f"{source_zarr}: roi dtype mismatch {dataset_roi_dtype} != {roi_dtype}.")

        if annotation_group is None or annotation_parent_name is None or not keypoints_path or not success_path:
            raise ValueError(f"{source_zarr}: unable to resolve keypoint annotation source.")

        frame_indices_path = (
            f"crop_runs/{source_crop_run}/source_frame_indices"
            if "source_frame_indices" in crop_group
            else (f"crop_runs/{source_crop_run}/frame_indices" if "frame_indices" in crop_group else None)
        )
        detection_source_path = (
            f"crop_runs/{source_crop_run}/detection_source" if "detection_source" in crop_group else None
        )
        source_type_resolved = (
            _as_text(dataset.get("source_type_resolved"))
            or _as_text(crop_group.attrs.get("detection_source_type"))
            or _as_text(manifest_payload.get("source_type"))
            or prepare_pose.DEFAULT_KEYPOINT_SOURCE_TYPE
        )
        source_type_resolved = str(source_type_resolved).strip().lower()
        if source_type_resolved not in prepare_pose.ALLOWED_KEYPOINT_CROP_SOURCE_TYPES:
            raise ValueError(
                f"{source_zarr}: keypoint merged export requires crop lineage detection_source_type in "
                f"{sorted(prepare_pose.ALLOWED_KEYPOINT_CROP_SOURCE_TYPES)!r}, observed {source_type_resolved!r} "
                f"on crop run '{source_crop_run}'."
            )

        dish_design, canvas_name, rig_id = _extract_identity(dataset)

        if (
            str(gate_stats.get("policy") or "").strip().lower() == "refined_usable"
            and gate_stats.get("refined_run")
            and refined_parent_name is not None
        ):
            refined_run_name = str(gate_stats["refined_run"])
            refined_group = root[f"{refined_parent_name}/{refined_run_name}"]
            if "keypoints_roi" in refined_group:
                refined_keypoints_arr = refined_group["keypoints_roi"]
                if int(refined_keypoints_arr.shape[0]) != source_sample_count:
                    raise ValueError(
                        f"{source_zarr}: refined keypoints/roi row mismatch "
                        f"({refined_keypoints_arr.shape[0]} != {source_sample_count})."
                    )
                keypoints_path = f"{refined_parent_name}/{refined_run_name}/keypoints_roi"
                dataset_keypoint_shape = tuple(int(v) for v in refined_keypoints_arr.shape[1:])
                dataset_keypoint_dtype = np.dtype(refined_keypoints_arr.dtype)
                annotation_group = refined_group
                annotation_run = refined_run_name
                annotation_parent_name = refined_parent_name
            if "usable_keypoints" in refined_group:
                refined_success_arr = refined_group["usable_keypoints"]
                if int(refined_success_arr.shape[0]) != source_sample_count:
                    raise ValueError(
                        f"{source_zarr}: refined usable_keypoints/roi row mismatch "
                        f"({refined_success_arr.shape[0]} != {source_sample_count})."
                    )
                success_path = f"{refined_parent_name}/{refined_run_name}/usable_keypoints"
            elif "refined_success" in refined_group:
                refined_success_arr = refined_group["refined_success"]
                if int(refined_success_arr.shape[0]) != source_sample_count:
                    raise ValueError(
                        f"{source_zarr}: refined refined_success/roi row mismatch "
                        f"({refined_success_arr.shape[0]} != {source_sample_count})."
                    )
                success_path = f"{refined_parent_name}/{refined_run_name}/refined_success"

        if keypoint_shape is None:
            keypoint_shape = dataset_keypoint_shape
            keypoint_dtype = dataset_keypoint_dtype
        else:
            if dataset_keypoint_shape != keypoint_shape:
                raise ValueError(
                    f"{source_zarr}: keypoint shape mismatch {dataset_keypoint_shape} != {keypoint_shape}."
                )
            if dataset_keypoint_dtype != keypoint_dtype:
                raise ValueError(
                    f"{source_zarr}: keypoint dtype mismatch {dataset_keypoint_dtype} != {keypoint_dtype}."
                )

        dataset_skeleton_id, dataset_kpt_shape, dataset_signature = _resolve_dataset_skeleton_identity(
            dataset_payload=dataset,
            kp_group=annotation_group,
            source_zarr=source_zarr,
            keypoint_run=f"{annotation_parent_name}/{annotation_run}",
            keypoint_count=int(dataset_keypoint_shape[0]),
            manifest_skeleton_id=manifest_skeleton_id,
            manifest_kpt_shape=manifest_kpt_shape,
        )
        dataset_labels = _resolve_dataset_keypoint_labels(
            manifest_payload=manifest_payload,
            dataset_payload=dataset,
            annotation_group=annotation_group,
            source_zarr=source_zarr,
            keypoint_run=f"{annotation_parent_name}/{annotation_run}",
            keypoint_count=int(dataset_keypoint_shape[0]),
        )
        dataset_member = f"{dataset_id} (zarr={source_zarr}, keypoint_run={keypoint_run})"
        label_signature = tuple(str(label) for label in dataset_labels)
        keypoint_label_members.setdefault(label_signature, []).append(dataset_member)
        if keypoint_labels is None:
            keypoint_labels = list(dataset_labels)
        elif list(dataset_labels) != keypoint_labels:
            detail_lines = []
            for signature, members in sorted(keypoint_label_members.items()):
                detail_lines.append(f"- {list(signature)}: {', '.join(members)}")
            raise ValueError(
                "Mixed keypoint label sets detected while exporting keypoint training data. "
                "Expected one keypoint_labels ordering/signature but found:\n"
                + "\n".join(detail_lines)
            )
        skeleton_signature_members.setdefault(dataset_signature, []).append(dataset_member)
        candidate_identity = (dataset_skeleton_id, dataset_kpt_shape)
        if selected_skeleton_identity is None:
            selected_skeleton_identity = candidate_identity
            selected_skeleton_signature = dataset_signature
        elif candidate_identity != selected_skeleton_identity:
            detail_lines = []
            for signature, members in sorted(skeleton_signature_members.items()):
                detail_lines.append(f"- {signature}: {', '.join(members)}")
            raise ValueError(
                "Mixed skeleton identities detected while exporting keypoint training data. "
                "Expected one (skeleton_id, kpt_shape) signature but found:\n"
                + "\n".join(detail_lines)
            )

        specs.append(
            PoseMergeSourceSpec(
                ordinal=ordinal,
                dataset_name=str(dataset.get("name") or source_zarr.stem),
                dataset_id=dataset_id,
                source_zarr=source_zarr,
                source_crop_run=source_crop_run,
                keypoint_run=keypoint_run,
                source_type_resolved=str(source_type_resolved),
                source_sample_count=source_sample_count,
                sample_count=int(selected_indices.shape[0]),
                selected_indices=selected_indices,
                row_gate_policy=str(gate_stats["policy"]),
                row_gate_refined_run=(
                    str(gate_stats["refined_run"]) if gate_stats.get("refined_run") is not None else None
                ),
                row_gate_selected=int(gate_stats["selected"]),
                row_gate_total=int(gate_stats["total"]),
                row_gate_raw_success_true=int(gate_stats["raw_success_true"]),
                row_gate_usable_true=(
                    int(gate_stats["usable_true"]) if gate_stats.get("usable_true") is not None else None
                ),
                row_gate_box_only_true=int(gate_stats.get("box_only_true") or 0),
                row_gate_box_only_selected=int(gate_stats.get("box_only_selected") or 0),
                roi_path=f"crop_runs/{source_crop_run}/roi_images",
                bbox_path=f"crop_runs/{source_crop_run}/bbox_norm_coords",
                keypoints_path=keypoints_path,
                success_path=success_path,
                frame_indices_path=frame_indices_path,
                detection_source_path=detection_source_path,
                roi_shape=dataset_roi_shape,
                roi_dtype=dataset_roi_dtype,
                roi_chunks=dataset_roi_chunks,
                roi_pixel_contract=roi_pixel_contract,
                roi_pixel_contract_name=roi_pixel_contract_name,
                keypoint_shape=dataset_keypoint_shape,
                keypoint_dtype=dataset_keypoint_dtype,
                keypoint_labels=list(dataset_labels),
                skeleton_id=dataset_skeleton_id,
                kpt_shape=dataset_kpt_shape,
                skeleton_signature=dataset_signature,
                dish_design=dish_design,
                canvas_name=canvas_name,
                rig_id=rig_id,
                box_only_selected_mask=selected_box_only_mask,
            )
        )

    if roi_shape is None or roi_dtype is None or keypoint_shape is None or keypoint_dtype is None:
        raise ValueError("Failed to resolve merged keypoint layout from manifest datasets.")
    if selected_skeleton_identity is None:
        raise ValueError("Failed to resolve skeleton identity from manifest datasets.")
    if keypoint_labels is None:
        raise ValueError("Failed to resolve keypoint label metadata from manifest datasets.")

    return specs, {
        "roi_shape": roi_shape,
        "roi_dtype": roi_dtype,
        "roi_chunks": roi_chunks,
        "keypoint_shape": keypoint_shape,
        "keypoint_dtype": keypoint_dtype,
        "keypoint_labels": keypoint_labels,
        "skeleton_id": selected_skeleton_identity[0],
        "kpt_shape": selected_skeleton_identity[1],
        "skeleton_signature": selected_skeleton_signature
        or _format_skeleton_signature(
            skeleton_id=selected_skeleton_identity[0],
            kpt_shape=selected_skeleton_identity[1],
        ),
    }


def _ensure_writable_destination(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Destination exists: {path}")
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)


def _pose_bbox_from_keypoints_px(keypoints_px: np.ndarray, *, roi_h: int, roi_w: int) -> np.ndarray:
    """Compute pose-training bbox (xywhn) from keypoints in ROI pixel coordinates."""
    if keypoints_px.ndim != 3 or int(keypoints_px.shape[2]) != 2:
        raise ValueError(f"keypoints_px must have shape (N,K,2), got {tuple(keypoints_px.shape)}.")
    if roi_h <= 0 or roi_w <= 0:
        raise ValueError(f"Invalid ROI dimensions for bbox normalization: h={roi_h}, w={roi_w}.")

    keypoints_norm = keypoints_px.astype(np.float32, copy=True)
    keypoints_norm[..., 0] /= float(roi_w)
    keypoints_norm[..., 1] /= float(roi_h)
    keypoints_norm = np.clip(keypoints_norm, 0.0, 1.0)

    finite_mask = np.isfinite(keypoints_norm).all(axis=(1, 2))
    if not np.all(finite_mask):
        bad = int(np.sum(~finite_mask))
        raise ValueError(f"Cannot compute pose bbox: {bad} row(s) contain non-finite keypoints.")

    min_xy = np.nanmin(keypoints_norm, axis=1)
    max_xy = np.nanmax(keypoints_norm, axis=1)
    span = max_xy - min_xy
    margin = span * 0.5
    center = np.clip((min_xy + max_xy) / 2.0, 0.0, 1.0)
    bbox_wh = np.clip(span + margin, 1e-6, 1.0)
    return np.concatenate([center, bbox_wh], axis=1).astype(np.float32)


def _export_merged(
    *,
    manifest_payload: Dict[str, Any],
    manifest_path: Path,
    out_zarr: Path,
    merged_dataset_id: Optional[str],
    overwrite: bool,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    copy_batch_size: int,
    row_gate_policy: str,
    invocation: Dict[str, Any],
) -> PoseMergeResult:
    input_format = str(manifest_payload.get("input_format") or "gray").strip().lower()
    if input_format not in {"gray", "rgb"}:
        raise ValueError(f"Unsupported manifest.input_format '{input_format}'. Expected gray or rgb.")
    requested_source_type = str(
        manifest_payload.get("source_type_requested")
        or manifest_payload.get("source_type")
        or prepare_pose.DEFAULT_KEYPOINT_SOURCE_TYPE
    ).strip().lower()
    if requested_source_type not in prepare_pose.ALLOWED_KEYPOINT_CROP_SOURCE_TYPES:
        raise ValueError(
            "Keypoint merged export only supports reviewed refined/manual crop-source manifests; "
            f"observed source_type_requested={requested_source_type!r}."
        )

    source_specs, layout = _discover_merge_sources(
        manifest_payload,
        expected_input_format=input_format,
        row_gate_policy=row_gate_policy,
    )
    source_type = _collapse_source_type_values(
        [spec.source_type_resolved for spec in source_specs],
        fallback=requested_source_type,
    )
    resolved_source_types = {spec.source_type_resolved for spec in source_specs}
    unsupported_source_types = sorted(resolved_source_types - prepare_pose.ALLOWED_KEYPOINT_CROP_SOURCE_TYPES)
    if unsupported_source_types:
        raise ValueError(
            "Keypoint merged export only supports reviewed refined/manual crop-source datasets; "
            f"unsupported source_type values={unsupported_source_types!r}."
        )
    merged_skeleton_id = _as_text(layout.get("skeleton_id"))
    merged_kpt_shape = _normalize_kpt_shape(layout.get("kpt_shape"))
    merged_skeleton_signature = (
        _as_text(layout.get("skeleton_signature"))
        or _format_skeleton_signature(skeleton_id=merged_skeleton_id, kpt_shape=merged_kpt_shape)
    )
    total_samples = int(sum(spec.sample_count for spec in source_specs))
    row_gate_counts: Dict[str, int] = {}

    _ensure_writable_destination(out_zarr, overwrite=overwrite)
    out_root = zarr.open_group(str(out_zarr), mode="w")
    crop_parent = out_root.require_group("crop_runs")
    keypoint_parent = out_root.require_group("keypoints_runs")
    split_group = out_root.require_group("splits")
    source_index_group = out_root.require_group("source_index")

    run_name = f"merged_export_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    crop_parent.attrs["latest"] = run_name
    keypoint_parent.attrs["latest"] = run_name
    crop_group = crop_parent.require_group(run_name)
    keypoint_group = keypoint_parent.require_group(run_name)

    roi_shape = tuple(layout["roi_shape"])
    roi_dtype = np.dtype(layout["roi_dtype"])
    roi_chunks = _normalize_chunks(layout["roi_chunks"], (total_samples, *roi_shape))
    keypoint_shape = tuple(layout["keypoint_shape"])
    keypoint_dtype = np.dtype(layout["keypoint_dtype"])
    keypoint_labels = list(layout["keypoint_labels"])

    vector_chunks = (max(1, min(8192, total_samples)),)
    bbox_chunks = (max(1, min(8192, total_samples)), 4)

    roi_dest = crop_group.require_array(
        "roi_images",
        shape=(total_samples, *roi_shape),
        dtype=roi_dtype,
        chunks=roi_chunks,
        overwrite=True,
    )
    bbox_dest = crop_group.require_array(
        "bbox_norm_coords",
        shape=(total_samples, 4),
        dtype=np.float32,
        chunks=bbox_chunks,
        overwrite=True,
    )
    crop_bbox_dest = crop_group.require_array(
        "crop_bbox_norm_coords",
        shape=(total_samples, 4),
        dtype=np.float32,
        chunks=bbox_chunks,
        overwrite=True,
    )
    frame_idx_dest = crop_group.require_array(
        "frame_indices",
        shape=(total_samples,),
        dtype=np.int64,
        chunks=vector_chunks,
        overwrite=True,
    )
    det_source_dest = crop_group.require_array(
        "detection_source",
        shape=(total_samples,),
        dtype=np.int8,
        chunks=vector_chunks,
        overwrite=True,
    )
    keypoints_dest = keypoint_group.require_array(
        "keypoints_roi",
        shape=(total_samples, *keypoint_shape),
        dtype=keypoint_dtype,
        chunks=_normalize_chunks(None, (total_samples, *keypoint_shape)),
        overwrite=True,
    )
    success_dest = keypoint_group.require_array(
        "detection_success",
        shape=(total_samples,),
        dtype=np.bool_,
        chunks=vector_chunks,
        overwrite=True,
    )
    box_only_dest = keypoint_group.require_array(
        "keypoint_box_only",
        shape=(total_samples,),
        dtype=np.bool_,
        chunks=vector_chunks,
        overwrite=True,
    )
    src_dataset_idx_dest = source_index_group.require_array(
        "source_dataset_idx",
        shape=(total_samples,),
        dtype=np.int32,
        chunks=vector_chunks,
        overwrite=True,
    )
    src_frame_idx_dest = source_index_group.require_array(
        "source_frame_idx",
        shape=(total_samples,),
        dtype=np.int64,
        chunks=vector_chunks,
        overwrite=True,
    )
    src_roi_idx_dest = source_index_group.require_array(
        "source_roi_idx",
        shape=(total_samples,),
        dtype=np.int64,
        chunks=vector_chunks,
        overwrite=True,
    )
    src_refined_row_ids_dest = source_index_group.require_array(
        "source_refined_row_ids",
        shape=(total_samples,),
        dtype=np.int64,
        chunks=vector_chunks,
        overwrite=True,
    )
    src_detect_row_index_dest = source_index_group.require_array(
        "source_detect_row_index",
        shape=(total_samples,),
        dtype=np.int32,
        chunks=vector_chunks,
        overwrite=True,
    )

    copy_total = total_samples * 2  # roi + keypoints arrays
    copy_progress = _copy_progress(copy_total)
    copy_task_id: Optional[int] = None
    if copy_progress is not None:
        copy_progress.start()
        copy_task_id = copy_progress.add_task("Copying merged pose samples", total=copy_total)

    def _advance_copy(count: int) -> None:
        if copy_progress is not None and copy_task_id is not None:
            copy_progress.advance(copy_task_id, count)

    roi_h = int(roi_shape[0])
    roi_w = int(roi_shape[1])
    total_successful = 0
    keypoint_supervision_counts: Dict[str, int] = {"full": 0, "box_only": 0}
    detection_source_counts: Dict[str, int] = {}
    offset = 0
    try:
        for spec in source_specs:
            if copy_progress is not None and copy_task_id is not None:
                copy_progress.update(copy_task_id, description=f"Copying {spec.dataset_name}")

            root = zarr.open_group(str(spec.source_zarr), mode="r", use_consolidated=False)
            roi_src = root[spec.roi_path]
            crop_source_group = root[f"crop_runs/{spec.source_crop_run}"]
            selected = np.asarray(spec.selected_indices, dtype=np.int64)
            bbox_all = np.asarray(root[spec.bbox_path][:], dtype=np.float32)
            keypoints_all = np.asarray(root[spec.keypoints_path][:], dtype=keypoint_dtype)
            success_all = np.asarray(root[spec.success_path][:], dtype=np.bool_)
            annotation_group_path = spec.keypoints_path.rsplit("/", 1)[0]
            annotation_group = root[annotation_group_path]
            source_refined_row_ids, source_detect_row_index = resolve_row_identity_arrays(
                annotation_group,
                crop_source_group,
                total_rois=spec.source_sample_count,
            )
            keypoints = keypoints_all[selected]
            success = success_all[selected]
            crop_bbox = _take_crop_or_source_rows(
                bbox_all,
                selected,
                source_detect_row_index=source_detect_row_index,
                source_sample_count=spec.source_sample_count,
                source_zarr=spec.source_zarr,
                array_name=spec.bbox_path,
            )
            local_total = int(spec.sample_count)
            box_only_mask = (
                np.asarray(spec.box_only_selected_mask, dtype=np.bool_)
                if spec.box_only_selected_mask is not None
                else np.zeros(local_total, dtype=np.bool_)
            )
            if box_only_mask.shape[0] != local_total:
                raise ValueError(
                    f"{spec.source_zarr}: box_only mask length mismatch "
                    f"({box_only_mask.shape[0]} != {local_total})."
                )
            full_mask = ~box_only_mask

            keypoints_export = np.asarray(keypoints, dtype=keypoint_dtype).copy()
            if np.any(box_only_mask):
                # Placeholder finite coordinates; loader maps these rows to visibility=0.
                keypoints_export[box_only_mask] = 0

            pose_bbox = np.zeros((local_total, 4), dtype=np.float32)
            full_idx = np.where(full_mask)[0]
            if full_idx.size > 0:
                pose_bbox_full = _pose_bbox_from_keypoints_px(
                    keypoints[full_idx],
                    roi_h=roi_h,
                    roi_w=roi_w,
                )
                pose_bbox[full_idx] = pose_bbox_full
            box_only_idx = np.where(box_only_mask)[0]
            if box_only_idx.size > 0:
                crop_bbox_box_only = crop_bbox[box_only_idx]
                if not np.isfinite(crop_bbox_box_only).all():
                    raise ValueError(
                        f"{spec.source_zarr}: box-only rows include non-finite crop bbox values."
                    )
                pose_bbox[box_only_idx] = crop_bbox_box_only

            if spec.frame_indices_path:
                source_frame_idx = np.asarray(root[spec.frame_indices_path][:], dtype=np.int64)
            else:
                source_frame_idx = np.arange(spec.source_sample_count, dtype=np.int64)
            if source_frame_idx.shape[0] != int(spec.source_sample_count):
                raise ValueError(
                    f"{spec.source_zarr}: frame_indices length mismatch "
                    f"({source_frame_idx.shape[0]} != {spec.source_sample_count})."
                )
            source_frame_idx = source_frame_idx[selected]

            if spec.detection_source_path:
                detection_source = np.asarray(root[spec.detection_source_path][:], dtype=np.int8)
                detection_source = _take_crop_or_source_rows(
                    detection_source,
                    selected,
                    source_detect_row_index=source_detect_row_index,
                    source_sample_count=spec.source_sample_count,
                    source_zarr=spec.source_zarr,
                    array_name=spec.detection_source_path,
                    )
            else:
                detection_source = np.zeros(local_total, dtype=np.int8)

            _copy_rows_indexed(
                roi_src,
                roi_dest,
                offset,
                selected,
                batch_size=copy_batch_size,
                on_copied=_advance_copy,
            )
            keypoints_dest[offset : offset + local_total] = keypoints_export
            _advance_copy(local_total)
            # Canonical pose bbox for training semantics.
            bbox_dest[offset : offset + local_total] = pose_bbox
            # Crop-stage bbox provenance for audit/debug.
            crop_bbox_dest[offset : offset + local_total] = crop_bbox
            success_dest[offset : offset + local_total] = success
            box_only_dest[offset : offset + local_total] = box_only_mask
            frame_idx_dest[offset : offset + local_total] = np.arange(offset, offset + local_total, dtype=np.int64)
            det_source_dest[offset : offset + local_total] = detection_source
            src_dataset_idx_dest[offset : offset + local_total] = int(spec.ordinal)
            src_frame_idx_dest[offset : offset + local_total] = source_frame_idx
            src_roi_idx_dest[offset : offset + local_total] = selected
            src_refined_row_ids_dest[offset : offset + local_total] = source_refined_row_ids[selected]
            src_detect_row_index_dest[offset : offset + local_total] = source_detect_row_index[selected]

            total_successful += int(success.sum())
            keypoint_supervision_counts["box_only"] += int(np.sum(box_only_mask))
            keypoint_supervision_counts["full"] += int(np.sum(full_mask))
            uniques, counts = np.unique(detection_source, return_counts=True)
            for value, count in zip(uniques.tolist(), counts.tolist()):
                key = str(int(value))
                detection_source_counts[key] = detection_source_counts.get(key, 0) + int(count)
            row_gate_counts[spec.row_gate_policy] = row_gate_counts.get(spec.row_gate_policy, 0) + int(local_total)

            offset += local_total
    finally:
        if copy_progress is not None:
            copy_progress.stop()

    total_failed = int(total_samples - total_successful)
    train_idx, val_idx, test_idx = _compute_split_indices(
        total_samples,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    split_group.require_array(
        "train_indices",
        shape=train_idx.shape,
        dtype=train_idx.dtype,
        chunks=(max(1, min(8192, train_idx.shape[0] or 1)),),
        overwrite=True,
    )[:] = train_idx
    split_group.require_array(
        "val_indices",
        shape=val_idx.shape,
        dtype=val_idx.dtype,
        chunks=(max(1, min(8192, val_idx.shape[0] or 1)),),
        overwrite=True,
    )[:] = val_idx
    if test_idx.size > 0:
        split_group.require_array(
            "test_indices",
            shape=test_idx.shape,
            dtype=test_idx.dtype,
            chunks=(max(1, min(8192, test_idx.shape[0])),),
            overwrite=True,
        )[:] = test_idx

    _write_string_array(source_index_group, "source_dataset_id", [spec.dataset_id for spec in source_specs])
    _write_string_array(source_index_group, "source_zarr_path", [str(spec.source_zarr) for spec in source_specs])
    source_index_group.attrs.update({"mapping_version": 1, "source_count": int(len(source_specs))})

    crop_group.attrs.update(
        {
            "detection_source_type": source_type,
            "includes_interpolated": bool(np.sum(det_source_dest[:] != 0) > 0),
            "n_real_detections": int(np.sum(det_source_dest[:] == 0)),
            "n_interpolated_detections": int(np.sum(det_source_dest[:] != 0)),
            "source_count": int(len(source_specs)),
            "bbox_norm_coords_semantics": "pose_xywhn_with_box_only_crop_fallback",
            "crop_bbox_norm_coords_semantics": "crop_stage_provenance_xywhn",
        }
    )
    keypoint_group.attrs.update(
        {
            "method": "merged_export",
            "source_crop_run": run_name,
            "keypoint_labels": keypoint_labels,
            "skeleton_id": merged_skeleton_id,
            "kpt_shape": list(merged_kpt_shape) if merged_kpt_shape is not None else None,
            "skeleton_signature": merged_skeleton_signature,
            "keypoints_processed": int(total_samples),
            "success_rate": (float(total_successful) / float(total_samples)) if total_samples > 0 else 0.0,
            "row_gate_policy": (
                next(iter(row_gate_counts.keys())) if len(row_gate_counts) == 1 else "mixed"
            ),
            "row_gate_applied": True,
            "row_gate_counts": dict(row_gate_counts),
            "keypoint_supervision_counts": dict(keypoint_supervision_counts),
            "keypoint_box_only_semantics": {
                "false": "full_keypoint_supervision",
                "true": "box_only_no_keypoint_coordinates",
            },
        }
    )

    training_export = {
        "schema_version": "1.0.0",
        "task": "pose",
        "set_id": manifest_payload.get("set_id"),
        "set_name": manifest_payload.get("set_name"),
        "set_version": manifest_payload.get("set_version"),
        "source_type_requested": source_type,
        "input_format": input_format,
        "created_at_utc": _utc_now(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "query_filter": manifest_payload.get("query_filter"),
        "invocation": invocation,
        "source_dataset_ids": [spec.dataset_id for spec in source_specs],
        "source_zarr_paths": [str(spec.source_zarr) for spec in source_specs],
        "skeleton_id": merged_skeleton_id,
        "kpt_shape": list(merged_kpt_shape) if merged_kpt_shape is not None else None,
        "skeleton_signature": merged_skeleton_signature,
        "source_skeleton_signatures": [
            {
                "dataset_id": spec.dataset_id,
                "skeleton_id": spec.skeleton_id,
                "kpt_shape": list(spec.kpt_shape) if spec.kpt_shape is not None else None,
                "signature": spec.skeleton_signature,
                "zarr_path": str(spec.source_zarr),
            }
            for spec in source_specs
        ],
        "row_gate": {
            "requested_policy": row_gate_policy,
            "applied_policy": next(iter(row_gate_counts.keys())) if len(row_gate_counts) == 1 else "mixed",
            "per_policy_counts": dict(row_gate_counts),
        },
        "keypoint_supervision_counts": dict(keypoint_supervision_counts),
        "split": {
            "train_ratio": float(train_ratio),
            "val_ratio": float(val_ratio),
            "test_ratio": float(test_ratio) if test_ratio > 0.0 else None,
            "seed": int(seed),
            "strategy": "global_random",
        },
    }
    out_root.attrs.update(
        {
            "zarr_purpose": "training",
            "training_export": training_export,
            **(
                {
                    "session_uuid": str(merged_dataset_id),
                    "session_id": str(merged_dataset_id),
                }
                if merged_dataset_id
                else {}
            ),
        }
    )

    split_group.attrs.update(
        {
            "split_mode": "fixed_indices",
            "train_ratio": float(train_ratio),
            "val_ratio": float(val_ratio),
            "test_ratio": float(test_ratio) if test_ratio > 0.0 else None,
            "seed": int(seed),
            "strategy": "global_random",
            "created_at_utc": _utc_now(),
        }
    )

    return PoseMergeResult(
        run_name=run_name,
        total_samples=total_samples,
        total_successful=total_successful,
        total_failed=total_failed,
        detection_source_counts=detection_source_counts,
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        source_specs=source_specs,
        input_format=input_format,
        source_type=source_type,
        keypoint_shape=keypoint_shape,
        keypoint_labels=keypoint_labels,
        skeleton_id=merged_skeleton_id,
        kpt_shape=merged_kpt_shape,
        skeleton_signature=merged_skeleton_signature,
        row_gate_policy=next(iter(row_gate_counts.keys())) if len(row_gate_counts) == 1 else "mixed",
        row_gate_counts=row_gate_counts,
        keypoint_supervision_counts=dict(keypoint_supervision_counts),
    )


def validate_merged_keypoint_training_zarr(
    zarr_path: Path,
    *,
    expected_input_format: Optional[str] = None,
    expected_total_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Validate merged keypoint-training Zarr layout and trainer-facing invariants."""
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    errors: List[str] = []

    if str(root.attrs.get("zarr_purpose", "")).strip().lower() != "training":
        errors.append("root attr zarr_purpose must be 'training'.")
    for group_name in ("crop_runs", "keypoints_runs", "splits", "source_index"):
        if group_name not in root:
            errors.append(f"missing group {group_name}.")
    if errors:
        raise ValueError("Merged keypoint zarr validation failed:\n- " + "\n- ".join(errors))

    crop_parent = root["crop_runs"]
    keypoint_parent = root["keypoints_runs"]

    crop_latest = _as_text(crop_parent.attrs.get("latest"))
    keypoint_latest = _as_text(keypoint_parent.attrs.get("latest"))
    if not crop_latest or crop_latest not in crop_parent:
        errors.append("crop_runs/latest missing or points to a non-existent run.")
    if not keypoint_latest or keypoint_latest not in keypoint_parent:
        errors.append("keypoints_runs/latest missing or points to a non-existent run.")
    if errors:
        raise ValueError("Merged keypoint zarr validation failed:\n- " + "\n- ".join(errors))

    crop = crop_parent[str(crop_latest)]
    kp = keypoint_parent[str(keypoint_latest)]
    required_crop_arrays = (
        "roi_images",
        "bbox_norm_coords",
        "crop_bbox_norm_coords",
        "frame_indices",
        "detection_source",
    )
    for name in required_crop_arrays:
        if name not in crop:
            errors.append(f"missing required array crop_runs/{crop_latest}/{name}.")
    required_kp_arrays = ("keypoints_roi", "detection_success")
    for name in required_kp_arrays:
        if name not in kp:
            errors.append(f"missing required array keypoints_runs/{keypoint_latest}/{name}.")
    if errors:
        raise ValueError("Merged keypoint zarr validation failed:\n- " + "\n- ".join(errors))

    roi = np.asarray(crop["roi_images"][:])
    bbox = np.asarray(crop["bbox_norm_coords"][:])
    crop_bbox = np.asarray(crop["crop_bbox_norm_coords"][:])
    frame_indices = np.asarray(crop["frame_indices"][:])
    detection_source = np.asarray(crop["detection_source"][:])
    keypoints = np.asarray(kp["keypoints_roi"][:])
    detection_success = np.asarray(kp["detection_success"][:])
    keypoint_box_only = (
        np.asarray(kp["keypoint_box_only"][:], dtype=np.bool_)
        if "keypoint_box_only" in kp
        else np.zeros(int(keypoints.shape[0]), dtype=np.bool_)
    )

    if roi.ndim < 3:
        errors.append(f"roi_images must have shape (N, ...), got {tuple(roi.shape)}.")
    total_samples = int(roi.shape[0]) if roi.ndim >= 1 else 0

    training_export = root.attrs.get("training_export")
    train_input_format = expected_input_format
    if train_input_format is None:
        if isinstance(training_export, dict):
            train_input_format = _as_text(training_export.get("input_format"))
    if train_input_format:
        train_input_format = train_input_format.strip().lower()
        if train_input_format not in {"gray", "rgb"}:
            errors.append(f"Unsupported expected input format '{train_input_format}'.")
        elif train_input_format == "rgb":
            if roi.ndim != 4 or int(roi.shape[-1]) != 3:
                errors.append("roi_images must be (N,H,W,3) for rgb input format.")
        elif train_input_format == "gray":
            if roi.ndim == 4 and int(roi.shape[-1]) == 3:
                errors.append("roi_images appears rgb but expected gray input format.")

    if bbox.ndim != 2 or int(bbox.shape[1]) != 4:
        errors.append(f"bbox_norm_coords must have shape (N,4), got {tuple(bbox.shape)}.")
    if bbox.ndim == 2 and int(bbox.shape[0]) != total_samples:
        errors.append(f"bbox_norm_coords length mismatch ({bbox.shape[0]} != {total_samples}).")
    if crop_bbox.ndim != 2 or int(crop_bbox.shape[1]) != 4:
        errors.append(f"crop_bbox_norm_coords must have shape (N,4), got {tuple(crop_bbox.shape)}.")
    if crop_bbox.ndim == 2 and int(crop_bbox.shape[0]) != total_samples:
        errors.append(f"crop_bbox_norm_coords length mismatch ({crop_bbox.shape[0]} != {total_samples}).")

    if keypoints.ndim != 3 or int(keypoints.shape[2]) != 2:
        errors.append(f"keypoints_roi must have shape (N,K,2), got {tuple(keypoints.shape)}.")
    if keypoints.ndim == 3 and int(keypoints.shape[0]) != total_samples:
        errors.append(f"keypoints_roi length mismatch ({keypoints.shape[0]} != {total_samples}).")

    if detection_success.ndim != 1:
        errors.append(f"detection_success must be 1D, got ndim={detection_success.ndim}.")
    if detection_success.ndim == 1 and int(detection_success.shape[0]) != total_samples:
        errors.append(f"detection_success length mismatch ({detection_success.shape[0]} != {total_samples}).")
    if keypoint_box_only.ndim != 1:
        errors.append(f"keypoint_box_only must be 1D, got ndim={keypoint_box_only.ndim}.")
    if keypoint_box_only.ndim == 1 and int(keypoint_box_only.shape[0]) != total_samples:
        errors.append(f"keypoint_box_only length mismatch ({keypoint_box_only.shape[0]} != {total_samples}).")

    if frame_indices.ndim != 1:
        errors.append(f"frame_indices must be 1D, got ndim={frame_indices.ndim}.")
    if frame_indices.ndim == 1 and int(frame_indices.shape[0]) != total_samples:
        errors.append(f"frame_indices length mismatch ({frame_indices.shape[0]} != {total_samples}).")
    if frame_indices.ndim == 1 and np.issubdtype(frame_indices.dtype, np.integer):
        expected_local = np.arange(total_samples, dtype=np.int64)
        if not np.array_equal(frame_indices.astype(np.int64, copy=False), expected_local):
            errors.append("frame_indices must be local 0..N-1 indexing.")
    elif frame_indices.ndim == 1:
        errors.append(f"frame_indices must be integer dtype, got {frame_indices.dtype}.")

    if detection_source.ndim != 1:
        errors.append(f"detection_source must be 1D, got ndim={detection_source.ndim}.")
    if detection_source.ndim == 1 and int(detection_source.shape[0]) != total_samples:
        errors.append(f"detection_source length mismatch ({detection_source.shape[0]} != {total_samples}).")
    if detection_source.ndim == 1 and np.issubdtype(detection_source.dtype, np.integer):
        unique_codes = np.unique(detection_source.astype(np.int64, copy=False))
        invalid_codes = [int(code) for code in unique_codes.tolist() if int(code) not in (0, 1)]
        if invalid_codes:
            errors.append(f"detection_source contains invalid codes: {sorted(set(invalid_codes))}.")
    elif detection_source.ndim == 1:
        errors.append(f"detection_source must be integer dtype, got {detection_source.dtype}.")

    if expected_total_samples is not None and int(expected_total_samples) != total_samples:
        errors.append(f"total sample mismatch ({total_samples} != expected {int(expected_total_samples)}).")

    merged_skeleton_id: Optional[str] = None
    merged_kpt_shape: Optional[Tuple[int, int]] = None
    merged_skeleton_signature: Optional[str] = None
    if not isinstance(training_export, dict):
        errors.append("root attr training_export must be an object with skeleton identity metadata.")
    else:
        merged_skeleton_id = _as_text(training_export.get("skeleton_id"))
        merged_kpt_shape = _normalize_kpt_shape(training_export.get("kpt_shape"))
        merged_skeleton_signature = _format_skeleton_signature(
            skeleton_id=merged_skeleton_id,
            kpt_shape=merged_kpt_shape,
        )
        if merged_kpt_shape is None:
            errors.append("training_export.kpt_shape missing or invalid (expected [K,D]).")
        elif keypoints.ndim == 3 and merged_kpt_shape[0] != int(keypoints.shape[1]):
            errors.append(
                "training_export.kpt_shape/keypoints_roi mismatch "
                f"(kpt_shape={list(merged_kpt_shape)} vs keypoints_roi K={int(keypoints.shape[1])})."
            )

        raw_source_signatures = training_export.get("source_skeleton_signatures")
        if not isinstance(raw_source_signatures, list) or not raw_source_signatures:
            errors.append("training_export.source_skeleton_signatures missing or empty.")
        else:
            source_signature_members: Dict[str, List[str]] = {}
            source_identity: Optional[Tuple[Optional[str], Tuple[int, int]]] = None
            for idx, raw_entry in enumerate(raw_source_signatures):
                if not isinstance(raw_entry, dict):
                    errors.append(
                        f"training_export.source_skeleton_signatures[{idx}] must be an object."
                    )
                    continue
                entry_dataset_id = _as_text(raw_entry.get("dataset_id")) or f"entry[{idx}]"
                entry_skeleton_id = _as_text(raw_entry.get("skeleton_id"))
                entry_kpt_shape = _normalize_kpt_shape(raw_entry.get("kpt_shape"))
                if entry_kpt_shape is None:
                    errors.append(
                        f"training_export.source_skeleton_signatures[{idx}] has invalid kpt_shape "
                        f"for dataset '{entry_dataset_id}'."
                    )
                    continue
                entry_identity = (entry_skeleton_id, entry_kpt_shape)
                entry_signature = _format_skeleton_signature(
                    skeleton_id=entry_skeleton_id,
                    kpt_shape=entry_kpt_shape,
                )
                source_signature_members.setdefault(entry_signature, []).append(entry_dataset_id)
                if source_identity is None:
                    source_identity = entry_identity
                elif entry_identity != source_identity:
                    detail = "; ".join(
                        f"{signature}: {', '.join(members)}"
                        for signature, members in sorted(source_signature_members.items())
                    )
                    errors.append(
                        "Mixed skeleton identities detected in training_export.source_skeleton_signatures: "
                        + detail
                    )
                    break
            if source_identity is not None and merged_kpt_shape is not None:
                merged_identity = (merged_skeleton_id, merged_kpt_shape)
                if source_identity != merged_identity:
                    errors.append(
                        "training_export skeleton identity mismatch between merged metadata and source signatures "
                        f"(merged={_format_skeleton_signature(skeleton_id=merged_identity[0], kpt_shape=merged_identity[1])}, "
                        f"source={_format_skeleton_signature(skeleton_id=source_identity[0], kpt_shape=source_identity[1])})."
                    )

        supervision_counts = training_export.get("keypoint_supervision_counts")
        if isinstance(supervision_counts, dict):
            expected_full = int(np.sum(~keypoint_box_only))
            expected_box_only = int(np.sum(keypoint_box_only))
            if _as_text(supervision_counts.get("full")) is not None:
                try:
                    if int(supervision_counts.get("full")) != expected_full:
                        errors.append(
                            "training_export.keypoint_supervision_counts.full does not match keypoint_box_only."
                        )
                except Exception:
                    errors.append("training_export.keypoint_supervision_counts.full must be an integer.")
            if _as_text(supervision_counts.get("box_only")) is not None:
                try:
                    if int(supervision_counts.get("box_only")) != expected_box_only:
                        errors.append(
                            "training_export.keypoint_supervision_counts.box_only does not match keypoint_box_only."
                        )
                except Exception:
                    errors.append("training_export.keypoint_supervision_counts.box_only must be an integer.")

    split_arrays: Dict[str, np.ndarray] = {}
    for name in ("train_indices", "val_indices", "test_indices"):
        path = f"splits/{name}"
        if name == "test_indices" and path not in root:
            split_arrays[name] = np.empty(0, dtype=np.int64)
            continue
        if path not in root:
            errors.append(f"missing required array {path}.")
            continue
        arr = np.asarray(root[path][:])
        if arr.ndim != 1:
            errors.append(f"{path} must be 1D, got ndim={arr.ndim}.")
            continue
        if not np.issubdtype(arr.dtype, np.integer):
            errors.append(f"{path} must be integer dtype, got {arr.dtype}.")
            continue
        arr_i64 = arr.astype(np.int64, copy=False)
        if arr_i64.size > 0:
            min_idx = int(arr_i64.min())
            max_idx = int(arr_i64.max())
            if min_idx < 0 or max_idx >= total_samples:
                errors.append(
                    f"{path} indices out of bounds (min={min_idx}, max={max_idx}, total_samples={total_samples})."
                )
            if np.unique(arr_i64).size != arr_i64.size:
                errors.append(f"{path} contains duplicate indices.")
        split_arrays[name] = arr_i64

    train_idx = split_arrays.get("train_indices", np.empty(0, dtype=np.int64))
    val_idx = split_arrays.get("val_indices", np.empty(0, dtype=np.int64))
    test_idx = split_arrays.get("test_indices", np.empty(0, dtype=np.int64))
    if np.intersect1d(train_idx, val_idx).size > 0:
        errors.append("splits/train_indices overlaps with splits/val_indices.")
    if np.intersect1d(train_idx, test_idx).size > 0:
        errors.append("splits/train_indices overlaps with splits/test_indices.")
    if np.intersect1d(val_idx, test_idx).size > 0:
        errors.append("splits/val_indices overlaps with splits/test_indices.")
    combined = np.concatenate([train_idx, val_idx, test_idx]) if total_samples > 0 else np.empty(0, dtype=np.int64)
    if total_samples > 0:
        if combined.size != total_samples:
            errors.append(
                "split coverage mismatch "
                f"(train+val+test={combined.size} but total_samples={total_samples})."
            )
        elif np.unique(combined).size != total_samples:
            errors.append("split coverage must be exact and non-duplicated across split arrays.")

    src_dataset_idx_path = "source_index/source_dataset_idx"
    src_frame_idx_path = "source_index/source_frame_idx"
    src_dataset_id_path = "source_index/source_dataset_id"
    src_zarr_path_path = "source_index/source_zarr_path"
    for path in (src_dataset_idx_path, src_frame_idx_path, src_dataset_id_path, src_zarr_path_path):
        if path not in root:
            errors.append(f"missing required array {path}.")

    if not errors:
        source_dataset_idx = np.asarray(root[src_dataset_idx_path][:])
        source_frame_idx = np.asarray(root[src_frame_idx_path][:])
        source_dataset_id = np.asarray(root[src_dataset_id_path][:])
        source_zarr_path = np.asarray(root[src_zarr_path_path][:])

        if source_dataset_idx.ndim != 1 or source_dataset_idx.shape[0] != total_samples:
            errors.append(
                f"{src_dataset_idx_path} must be 1D length N ({total_samples}), got {source_dataset_idx.shape}."
            )
        if source_frame_idx.ndim != 1 or source_frame_idx.shape[0] != total_samples:
            errors.append(
                f"{src_frame_idx_path} must be 1D length N ({total_samples}), got {source_frame_idx.shape}."
            )
        if source_dataset_id.ndim != 1 or source_zarr_path.ndim != 1:
            errors.append("source_index/source_dataset_id and source_index/source_zarr_path must be 1D arrays.")
        elif source_dataset_id.shape[0] != source_zarr_path.shape[0]:
            errors.append(
                "source_index/source_dataset_id and source_index/source_zarr_path length mismatch "
                f"({source_dataset_id.shape[0]} != {source_zarr_path.shape[0]})."
            )
        elif total_samples > 0 and source_dataset_id.shape[0] == 0:
            errors.append("source index mapping arrays are empty but dataset has samples.")

        if source_dataset_idx.ndim == 1 and np.issubdtype(source_dataset_idx.dtype, np.integer):
            source_dataset_idx_i64 = source_dataset_idx.astype(np.int64, copy=False)
            if source_dataset_idx_i64.size > 0 and int(source_dataset_idx_i64.min()) < 0:
                errors.append(f"{src_dataset_idx_path} contains negative indices.")
            if source_dataset_id.ndim == 1 and source_dataset_id.shape[0] > 0 and source_dataset_idx_i64.size > 0:
                max_idx = int(source_dataset_idx_i64.max())
                if max_idx >= int(source_dataset_id.shape[0]):
                    errors.append(
                        f"{src_dataset_idx_path} has value {max_idx} outside mapping length {source_dataset_id.shape[0]}."
                    )
        else:
            errors.append(f"{src_dataset_idx_path} must be integer dtype.")

        if source_frame_idx.ndim == 1 and np.issubdtype(source_frame_idx.dtype, np.integer):
            source_frame_idx_i64 = source_frame_idx.astype(np.int64, copy=False)
            if source_frame_idx_i64.size > 0 and int(source_frame_idx_i64.min()) < 0:
                errors.append(f"{src_frame_idx_path} contains negative indices.")
        else:
            errors.append(f"{src_frame_idx_path} must be integer dtype.")

        for opt_path, require_nonnegative in (
            ("source_index/source_roi_idx", True),
            ("source_index/source_refined_row_ids", False),
            ("source_index/source_detect_row_index", False),
        ):
            if opt_path not in root:
                continue
            opt_arr = np.asarray(root[opt_path][:])
            if opt_arr.ndim != 1 or opt_arr.shape[0] != total_samples:
                errors.append(f"{opt_path} must be 1D length N ({total_samples}), got {opt_arr.shape}.")
                continue
            if not np.issubdtype(opt_arr.dtype, np.integer):
                errors.append(f"{opt_path} must be integer dtype, got {opt_arr.dtype}.")
                continue
            opt_i64 = opt_arr.astype(np.int64, copy=False)
            if require_nonnegative and opt_i64.size > 0 and int(opt_i64.min()) < 0:
                errors.append(f"{opt_path} contains negative indices.")

    if errors:
        raise ValueError("Merged keypoint zarr validation failed:\n- " + "\n- ".join(errors))

    success_count = int(np.asarray(kp["detection_success"][:], dtype=np.bool_).sum())
    return {
        "zarr_path": str(zarr_path),
        "run_name": str(crop_latest),
        "training_input_format": train_input_format,
        "skeleton_id": merged_skeleton_id,
        "kpt_shape": list(merged_kpt_shape) if merged_kpt_shape is not None else None,
        "skeleton_signature": merged_skeleton_signature,
        "total_samples": int(total_samples),
        "success_count": int(success_count),
        "failure_count": int(total_samples - success_count),
        "keypoint_supervision_counts": {
            "full": int(np.sum(~keypoint_box_only)),
            "box_only": int(np.sum(keypoint_box_only)),
        },
        "split_counts": {
            "train": int(train_idx.shape[0]),
            "val": int(val_idx.shape[0]),
            "test": int(test_idx.shape[0]),
        },
        "source_count": int(np.asarray(root[src_dataset_id_path][:]).shape[0]),
    }


def _register_merged_dataset_in_registry(
    *,
    registry_path: Path,
    merged_zarr: Path,
    set_id: Optional[str],
    set_name: Optional[str],
    source_dataset_ids: Sequence[str],
    query_filter: Optional[Dict[str, Any]],
    invocation: Optional[Dict[str, Any]],
) -> Tuple[str, bool]:
    """Register merged zarr and update training-set linkage.

    Returns (merged_dataset_id, training_set_link_updated).
    """
    registry = Registry(registry_path)
    try:
        dataset_id = registry.scan_zarr(merged_zarr)
        if not dataset_id:
            raise RuntimeError(f"Failed to register merged zarr: {merged_zarr}")

        linked = False
        if set_id:
            existing = registry.conn.execute(
                "SELECT name, query_filter, dataset_ids_json, invocation_json FROM training_sets WHERE set_id = ?",
                (set_id,),
            ).fetchone()
            existing_ids = _json_list(existing["dataset_ids_json"]) if existing else []
            merged_ids = sorted(
                {
                    *(str(item) for item in source_dataset_ids if item),
                    *existing_ids,
                    str(dataset_id),
                }
            )
            existing_query_filter = _json_dict(existing["query_filter"]) if existing else None
            existing_invocation = _json_dict(existing["invocation_json"]) if existing else None
            registry.upsert_training_set(
                set_id=set_id,
                name=set_name or (existing["name"] if existing else None),
                task_type="pose",
                query_filter=query_filter or existing_query_filter,
                dataset_ids=merged_ids,
                invocation=invocation or existing_invocation,
            )
            linked = True
        registry.replace_dataset_lineage(
            child_dataset_id=str(dataset_id),
            parent_dataset_ids=[str(item) for item in source_dataset_ids if item],
            relationship_type="training_merge_source",
            source_set_id=set_id,
            metadata={"producer": "export_keypoint_training_zarr"},
        )
        return str(dataset_id), linked
    finally:
        registry.close()


def _build_merged_manifest_payload(
    *,
    manifest_payload: Dict[str, Any],
    merged_zarr: Path,
    merged_dataset_id: str,
    merged_dataset_name: str,
    run_name: str,
    out_manifest: Path,
    out_config: Path,
    merge_result: PoseMergeResult,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, Any]:
    def _collapse(values: set[str], *, mixed_label: str) -> Optional[str]:
        if len(values) == 1:
            return next(iter(values))
        if len(values) > 1:
            return mixed_label
        return None

    source_datasets_payload: List[Dict[str, Any]] = []
    dish_values: set[str] = set()
    canvas_values: set[str] = set()
    rig_values: set[str] = set()
    roi_contract_names: set[str] = set()
    for spec in merge_result.source_specs:
        if spec.dish_design:
            dish_values.add(spec.dish_design)
        if spec.canvas_name:
            canvas_values.add(spec.canvas_name)
        if spec.rig_id:
            rig_values.add(spec.rig_id)
        if spec.roi_pixel_contract_name:
            roi_contract_names.add(spec.roi_pixel_contract_name)
        source_datasets_payload.append(
            {
                "name": spec.dataset_name,
                "dataset_id": spec.dataset_id,
                "zarr_path": str(spec.source_zarr),
                "sample_count": int(spec.sample_count),
                "source_sample_count": int(spec.source_sample_count),
                "row_gate_policy": spec.row_gate_policy,
                "row_gate_refined_run": spec.row_gate_refined_run,
                "row_gate_selected": int(spec.row_gate_selected),
                "row_gate_total": int(spec.row_gate_total),
                "row_gate_raw_success_true": int(spec.row_gate_raw_success_true),
                "row_gate_usable_true": spec.row_gate_usable_true,
                "row_gate_box_only_true": int(spec.row_gate_box_only_true or 0),
                "row_gate_box_only_selected": int(spec.row_gate_box_only_selected),
                "source_crop_run": spec.source_crop_run,
                "keypoint_run": spec.keypoint_run,
                "roi_pixel_contract": spec.roi_pixel_contract,
                "roi_pixel_contract_name": spec.roi_pixel_contract_name,
                "skeleton_id": spec.skeleton_id,
                "kpt_shape": list(spec.kpt_shape) if spec.kpt_shape is not None else None,
                "skeleton_signature": spec.skeleton_signature,
                "dish_design": spec.dish_design,
                "canvas_name": spec.canvas_name,
                "rig_id": spec.rig_id,
            }
        )

    merged_dish = _collapse(dish_values, mixed_label="mixed_dishes")
    merged_canvas = _collapse(canvas_values, mixed_label="mixed_canvas")
    merged_rig = _collapse(rig_values, mixed_label="mixed_rigs")
    merged_roi_contract_name = _collapse(roi_contract_names, mixed_label="mixed_roi_pixel_contracts")

    requested_source_type = (
        _as_text(manifest_payload.get("source_type_requested"))
        or _as_text(manifest_payload.get("source_type"))
        or merge_result.source_type
    )

    merged_dataset = {
        "name": merged_dataset_name,
        "zarr_path": str(merged_zarr),
        "dataset_id": merged_dataset_id,
        "session_uuid": None,
        "dish_design": merged_dish,
        "canvas_name": merged_canvas,
        "rig_id": merged_rig,
        "source_type_requested": requested_source_type,
        "source_type_resolved": merge_result.source_type,
        "source_crop_run": run_name,
        "input_format": merge_result.input_format,
        "roi_pixel_contract_name": merged_roi_contract_name,
        "keypoint_run_requested": "merged_export",
        "keypoint_run_resolved": run_name,
        "skeleton_id": merge_result.skeleton_id,
        "kpt_shape": list(merge_result.kpt_shape) if merge_result.kpt_shape is not None else None,
        "skeleton_signature": merge_result.skeleton_signature,
        "row_gate_policy": merge_result.row_gate_policy,
        "keypoint_supervision_counts": dict(merge_result.keypoint_supervision_counts),
        "keypoints_array_path": f"keypoints_runs/{run_name}/keypoints_roi",
        "detection_success_path": f"keypoints_runs/{run_name}/detection_success",
        "keypoints_total": int(merge_result.total_samples),
        "keypoints_successful": int(merge_result.total_successful),
        "keypoints_success_rate": (
            float(merge_result.total_successful) / float(merge_result.total_samples)
            if merge_result.total_samples > 0
            else 0.0
        ),
        "warnings": [],
    }

    payload = dict(manifest_payload)
    payload["datasets"] = [merged_dataset]
    payload["output_manifest_path"] = str(out_manifest)
    payload["output_config_path"] = str(out_config)
    payload["source_type_requested"] = requested_source_type
    payload["source_type"] = merge_result.source_type
    payload["roi_pixel_contract_name"] = merged_roi_contract_name
    payload["roi_pixel_contract_names"] = sorted(roi_contract_names)
    payload["dish_design"] = merged_dish
    payload["canvas_name"] = merged_canvas
    payload["rig_name"] = merged_rig
    payload["exported_at_utc"] = _utc_now()
    payload_pose_schema = payload.get("pose_schema")
    if isinstance(payload_pose_schema, dict):
        pose_schema_out = dict(payload_pose_schema)
    else:
        pose_schema_out = {}
    pose_schema_out["skeleton_id"] = merge_result.skeleton_id
    pose_schema_out["kpt_shape"] = list(merge_result.kpt_shape) if merge_result.kpt_shape is not None else None
    pose_schema_out["skeleton_signature"] = merge_result.skeleton_signature
    payload["pose_schema"] = pose_schema_out
    payload["merged_export"] = {
        "enabled": True,
        "zarr_path": str(merged_zarr),
        "run_name": run_name,
        "input_format": merge_result.input_format,
        "source_type": merge_result.source_type,
        "roi_pixel_contract_name": merged_roi_contract_name,
        "roi_pixel_contract_names": sorted(roi_contract_names),
        "skeleton_id": merge_result.skeleton_id,
        "kpt_shape": list(merge_result.kpt_shape) if merge_result.kpt_shape is not None else None,
        "skeleton_signature": merge_result.skeleton_signature,
        "row_gate_policy": merge_result.row_gate_policy,
        "row_gate_counts": dict(merge_result.row_gate_counts),
        "keypoint_supervision_counts": dict(merge_result.keypoint_supervision_counts),
        "keypoint_shape": list(merge_result.keypoint_shape),
        "keypoint_labels": list(merge_result.keypoint_labels),
        "split": {
            "train_ratio": float(train_ratio),
            "val_ratio": float(val_ratio),
            "test_ratio": float(test_ratio) if test_ratio > 0.0 else None,
            "seed": int(seed),
        },
        "counts": {
            "total_samples": int(merge_result.total_samples),
            "successful": int(merge_result.total_successful),
            "failed": int(merge_result.total_failed),
            "train": int(merge_result.train_indices.shape[0]),
            "val": int(merge_result.val_indices.shape[0]),
            "test": int(merge_result.test_indices.shape[0]),
            "source_count": int(len(merge_result.source_specs)),
            "row_gate_policy": merge_result.row_gate_policy,
            "row_gate_counts": dict(merge_result.row_gate_counts),
            "keypoint_supervision_counts": dict(merge_result.keypoint_supervision_counts),
        },
        "source_datasets": source_datasets_payload,
    }
    return payload


def _resolve_source_config_path(manifest_payload: Dict[str, Any]) -> Optional[Path]:
    output_config_path = _as_text(manifest_payload.get("output_config_path"))
    if output_config_path:
        candidate = Path(output_config_path)
        if candidate.exists():
            return candidate
    base_config_path = _as_text(manifest_payload.get("base_config_path"))
    if base_config_path:
        candidate = Path(base_config_path)
        if candidate.exists():
            return candidate
    return None


def _write_merged_config(
    *,
    source_config_path: Optional[Path],
    out_config: Path,
    merged_zarr: Path,
    dataset_name: str,
    source_type: str,
    input_format: str,
    keypoint_run: str,
    train_ratio: float,
    val_ratio: float,
    random_seed: int,
) -> None:
    cfg: Dict[str, Any] = {}
    if source_config_path and source_config_path.exists():
        loaded = yaml.safe_load(source_config_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            cfg = loaded

    cfg["datasets"] = {
        dataset_name: {
            "zarr_path": str(merged_zarr),
            "source_type": source_type,
            "input_format": input_format,
            "keypoint_run": keypoint_run,
            "split": {
                "train": float(train_ratio),
                "val": float(val_ratio),
            },
        }
    }
    cfg["random_seed"] = int(random_seed)
    out_config.parent.mkdir(parents=True, exist_ok=True)
    out_config.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def _write_merge_summary(
    *,
    summary_path: Path,
    out_zarr: Path,
    merge_result: PoseMergeResult,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> None:
    payload = {
        "created_at_utc": _utc_now(),
        "zarr_path": str(out_zarr),
        "input_format": merge_result.input_format,
        "source_type": merge_result.source_type,
        "skeleton_id": merge_result.skeleton_id,
        "kpt_shape": list(merge_result.kpt_shape) if merge_result.kpt_shape is not None else None,
        "skeleton_signature": merge_result.skeleton_signature,
        "row_gate_policy": merge_result.row_gate_policy,
        "row_gate_counts": dict(merge_result.row_gate_counts),
        "keypoint_supervision_counts": dict(merge_result.keypoint_supervision_counts),
        "keypoint_shape": list(merge_result.keypoint_shape),
        "keypoint_labels": list(merge_result.keypoint_labels),
        "counts": {
            "total_samples": int(merge_result.total_samples),
            "successful": int(merge_result.total_successful),
            "failed": int(merge_result.total_failed),
            "detection_source_counts": merge_result.detection_source_counts,
            "train": int(merge_result.train_indices.shape[0]),
            "val": int(merge_result.val_indices.shape[0]),
            "test": int(merge_result.test_indices.shape[0]),
        },
        "split": {
            "train_ratio": float(train_ratio),
            "val_ratio": float(val_ratio),
            "test_ratio": float(test_ratio) if test_ratio > 0.0 else None,
            "seed": int(seed),
        },
        "sources": [
            {
                "dataset_name": spec.dataset_name,
                "dataset_id": spec.dataset_id,
                "zarr_path": str(spec.source_zarr),
                "sample_count": int(spec.sample_count),
                "source_sample_count": int(spec.source_sample_count),
                "row_gate_policy": spec.row_gate_policy,
                "row_gate_refined_run": spec.row_gate_refined_run,
                "row_gate_selected": int(spec.row_gate_selected),
                "row_gate_total": int(spec.row_gate_total),
                "row_gate_raw_success_true": int(spec.row_gate_raw_success_true),
                "row_gate_usable_true": spec.row_gate_usable_true,
                "row_gate_box_only_true": int(spec.row_gate_box_only_true or 0),
                "row_gate_box_only_selected": int(spec.row_gate_box_only_selected),
                "source_crop_run": spec.source_crop_run,
                "keypoint_run": spec.keypoint_run,
                "skeleton_id": spec.skeleton_id,
                "kpt_shape": list(spec.kpt_shape) if spec.kpt_shape is not None else None,
                "skeleton_signature": spec.skeleton_signature,
                "dish_design": spec.dish_design,
                "canvas_name": spec.canvas_name,
                "rig_id": spec.rig_id,
            }
            for spec in merge_result.source_specs
        ],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _add_arg(cmd: List[str], flag: str, value: Any) -> None:
    if value is None:
        return
    cmd.extend([flag, str(value)])


def _load_keypoint_data_card_main() -> tuple[Optional[Callable[[Optional[List[str]]], int]], Optional[str]]:
    try:
        module = importlib.import_module("fisheye.utils.aggregate_keypoint_training_data_card")
    except Exception as exc:  # pragma: no cover - exercised when module is unavailable
        return None, f"{exc.__class__.__name__}: {exc}"
    main_fn = getattr(module, "main", None)
    if not callable(main_fn):
        return None, "module does not define callable main(argv)."
    return main_fn, None


def _run_keypoint_data_card_aggregation(*, cli: List[str], required: bool) -> int:
    main_fn, load_error = _load_keypoint_data_card_main()
    if main_fn is None:
        if required:
            raise SystemExit(
                "Keypoint training data-card aggregation requested, but "
                "fisheye.utils.aggregate_keypoint_training_data_card is unavailable "
                f"({load_error})."
            )
        print(
            "Skipping keypoint training data-card aggregation: "
            f"aggregator unavailable ({load_error})."
        )
        return 0
    try:
        result = main_fn(cli)
    except SystemExit as exc:  # pragma: no cover - defensive wrapper around argparse exits
        rc = int(exc.code) if isinstance(exc.code, int) else 1
        if required:
            return rc
        print(
            "Skipping keypoint training data-card aggregation: "
            f"aggregator exited with code {rc}."
        )
        return 0
    rc = int(result) if result is not None else 0
    if rc != 0 and not required:
        print(
            "Skipping keypoint training data-card aggregation: "
            f"aggregator returned code {rc}."
        )
        return 0
    return rc


def _build_data_card_cli(
    *,
    manifest_path: Path,
    merged_zarr: Path,
    registry_path: Path,
    args: argparse.Namespace,
) -> List[str]:
    card_cli: List[str] = [
        "--manifest",
        str(manifest_path),
        "--merged-zarr",
        str(merged_zarr),
        "--registry",
        str(registry_path),
        "--split",
        str(args.data_card_split),
    ]
    _add_arg(card_cli, "--output", args.data_card_output)
    if args.data_card_no_plots:
        card_cli.append("--no-plots")
    _add_arg(card_cli, "--plot-dir", args.data_card_plot_dir)
    _add_arg(card_cli, "--plot-prefix", args.data_card_plot_prefix)
    _add_arg(card_cli, "--plot-heatmap-bin-factor", args.data_card_plot_heatmap_bin_factor)
    if args.data_card_allow_profile_mtime_mismatch:
        card_cli.append("--allow-profile-mtime-mismatch")
    if args.data_card_allow_profile_fallback_scan:
        card_cli.append("--allow-profile-fallback-scan")
    if args.data_card_view:
        card_cli.append("--view")
    if args.data_card_force_plots:
        card_cli.append("--force")
    return card_cli


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Training manifest JSON.")
    parser.add_argument("--out-dir", type=Path, help="Output directory for merged artifacts.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merged mode (current supported mode).",
    )
    parser.add_argument(
        "--out-zarr",
        type=Path,
        help="Output path for merged Zarr.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="0.8/0.2",
        help="Split ratios: train/val or train/val/test.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for merged split generation.",
    )
    parser.add_argument(
        "--copy-batch-size",
        type=int,
        default=128,
        help="Row-copy batch size for merged export.",
    )
    parser.add_argument(
        "--row-gate-policy",
        choices=["auto", "refined_usable", "raw_success", "raw_success_plus_box_only"],
        default="auto",
        help=(
            "Row-level inclusion policy for merged export: "
            "auto prefers refined usable_keypoints when available, otherwise raw detection_success. "
            "raw_success_plus_box_only additionally keeps refined rows tagged fish_present_no_keypoints "
            "as box-only supervision."
        ),
    )
    parser.add_argument(
        "--registry",
        type=Path,
        help="Optional registry SQLite path. Registers merged dataset and updates training_set linkage.",
    )
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip merged-zarr post-export validation checks.",
    )
    aggregate_group = parser.add_mutually_exclusive_group()
    aggregate_group.add_argument(
        "--aggregate-training-data-card",
        dest="aggregate_training_data_card",
        action="store_true",
        help="Aggregate keypoint training data card after export/validation (default: enabled).",
    )
    aggregate_group.add_argument(
        "--no-aggregate-training-data-card",
        dest="aggregate_training_data_card",
        action="store_false",
        help="Disable automatic keypoint training data-card aggregation after export.",
    )
    parser.set_defaults(aggregate_training_data_card=True)
    parser.add_argument(
        "--data-card-output",
        type=Path,
        help="Optional output JSON path for keypoint training data-card aggregation.",
    )
    parser.add_argument(
        "--data-card-split",
        type=str,
        default="train",
        help="Split label for data-card selection metadata (default: train).",
    )
    parser.add_argument(
        "--data-card-no-plots",
        action="store_true",
        help="Skip plot PNG generation when aggregating the keypoint training data card.",
    )
    parser.add_argument(
        "--data-card-plot-dir",
        type=Path,
        help="Optional output directory for keypoint training data-card plots.",
    )
    parser.add_argument(
        "--data-card-plot-prefix",
        type=str,
        help="Optional filename prefix for keypoint training data-card plots.",
    )
    parser.add_argument(
        "--data-card-plot-heatmap-bin-factor",
        type=int,
        default=2,
        help="Coarsening factor for keypoint heatmap bins in data-card plots (default: 2).",
    )
    parser.add_argument(
        "--data-card-allow-profile-mtime-mismatch",
        action="store_true",
        help="Allow keypoint_data_profile_latest mtime mismatches during data-card aggregation.",
    )
    parser.add_argument(
        "--data-card-allow-profile-fallback-scan",
        action="store_true",
        help="Allow data-card aggregation to scan Zarr directly when profile rows are missing.",
    )
    parser.add_argument(
        "--data-card-view",
        action="store_true",
        help="Open keypoint data-card plots after aggregation.",
    )
    parser.add_argument(
        "--data-card-force-plots",
        action="store_true",
        help="Regenerate keypoint data-card plots even when files already exist.",
    )
    args = parser.parse_args(argv)

    if not args.merge:
        raise SystemExit("Only merged export mode is supported. Re-run with --merge.")

    manifest_payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest_payload, dict):
        raise ValueError(f"Manifest root must be an object: {args.manifest}")
    if not isinstance(manifest_payload.get("datasets"), list) or not manifest_payload.get("datasets"):
        raise ValueError(f"Manifest has no datasets: {args.manifest}")

    set_id_source = _as_text(manifest_payload.get("set_id")) or args.manifest.stem
    set_id = _normalize_manifest_stem(set_id_source)
    if not set_id:
        set_id = "training_set"
    out_dir = args.out_dir or (prepare_pose._resolve_dataset_root("pose") / set_id)
    out_manifest = out_dir / f"{set_id}.manifest.json"
    out_config = out_dir / f"{set_id}.yaml"
    merged_zarr = args.out_zarr or (out_dir / "zarr" / f"{set_id}_merged.zarr")
    merged_dataset_id = _ensure_suffix(set_id, "_merged")
    merged_dataset_name = _ensure_suffix(str(manifest_payload.get("set_name") or "merged"), "_merged")

    train_ratio, val_ratio, test_ratio = _parse_split_spec(args.split)
    if train_ratio + val_ratio <= 0.0:
        raise ValueError(f"Invalid --split '{args.split}': train+val must be > 0 for training config generation.")
    config_train_ratio = train_ratio / (train_ratio + val_ratio)
    config_val_ratio = val_ratio / (train_ratio + val_ratio)

    invocation = build_invocation_record(
        tool="fisheye.utils.export_keypoint_training_zarr",
        args=args,
    )

    print(f"Manifest: {args.manifest}")
    print(f"Merged output: {merged_zarr}")
    print(f"Datasets: {len(manifest_payload['datasets'])}")
    print(f"Split: train={train_ratio:.3f} val={val_ratio:.3f} test={test_ratio:.3f} seed={args.seed}")
    print(f"Copy batch size: {int(args.copy_batch_size)}")
    print(f"Row gate policy: {args.row_gate_policy}")
    print(f"Training input format: {manifest_payload.get('input_format', 'gray')}")
    for entry in manifest_payload["datasets"]:
        if isinstance(entry, dict):
            print(f"- {entry.get('name')}: {entry.get('zarr_path')}")

    if args.dry_run:
        _discover_merge_sources(
            manifest_payload,
            expected_input_format=str(manifest_payload.get("input_format") or "gray").strip().lower(),
            row_gate_policy=str(args.row_gate_policy),
        )
        print("Dry run: validation passed, no files written.")
        return 0

    merge_result = _export_merged(
        manifest_payload=manifest_payload,
        manifest_path=args.manifest,
        out_zarr=merged_zarr,
        merged_dataset_id=merged_dataset_id,
        overwrite=bool(args.overwrite),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=int(args.seed),
        copy_batch_size=int(args.copy_batch_size),
        row_gate_policy=str(args.row_gate_policy),
        invocation=invocation,
    )

    if not args.skip_validate:
        validation_summary = validate_merged_keypoint_training_zarr(
            merged_zarr,
            expected_input_format=merge_result.input_format,
            expected_total_samples=merge_result.total_samples,
        )
        split_counts = validation_summary["split_counts"]
        print(
            "Validated merged zarr: "
            f"samples={validation_summary['total_samples']} "
            f"train={split_counts['train']} "
            f"val={split_counts['val']} "
            f"test={split_counts['test']} "
            f"sources={validation_summary['source_count']}"
        )

    if args.registry:
        try:
            merged_dataset_id, linked = _register_merged_dataset_in_registry(
                registry_path=Path(args.registry),
                merged_zarr=merged_zarr,
                set_id=_as_text(manifest_payload.get("set_id")),
                set_name=_as_text(manifest_payload.get("set_name")),
                source_dataset_ids=[spec.dataset_id for spec in merge_result.source_specs],
                query_filter=manifest_payload.get("query_filter"),
                invocation=manifest_payload.get("invocation"),
            )
            print(f"Registered merged dataset in registry: {merged_dataset_id}")
            if linked and manifest_payload.get("set_id"):
                print(f"Updated training_set dataset_ids linkage: {manifest_payload['set_id']}")
        except Exception as exc:
            print(f"Warning: merged dataset registry update skipped: {exc}")

    summary_path = merged_zarr.with_suffix(".summary.json")
    _write_merge_summary(
        summary_path=summary_path,
        out_zarr=merged_zarr,
        merge_result=merge_result,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=int(args.seed),
    )

    merged_manifest_payload = _build_merged_manifest_payload(
        manifest_payload=manifest_payload,
        merged_zarr=merged_zarr,
        merged_dataset_id=merged_dataset_id,
        merged_dataset_name=merged_dataset_name,
        run_name=merge_result.run_name,
        out_manifest=out_manifest,
        out_config=out_config,
        merge_result=merge_result,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=int(args.seed),
    )
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(merged_manifest_payload, indent=2), encoding="utf-8")

    _write_merged_config(
        source_config_path=_resolve_source_config_path(manifest_payload),
        out_config=out_config,
        merged_zarr=merged_zarr,
        dataset_name=merged_dataset_name,
        source_type=merge_result.source_type,
        input_format=merge_result.input_format,
        keypoint_run=merge_result.run_name,
        train_ratio=config_train_ratio,
        val_ratio=config_val_ratio,
        random_seed=int(args.seed),
    )

    if args.aggregate_training_data_card:
        card_registry = Path(args.registry) if args.registry is not None else RegistryPaths.from_env(Path.cwd()).path
        card_cli = _build_data_card_cli(
            manifest_path=out_manifest,
            merged_zarr=merged_zarr,
            registry_path=card_registry,
            args=args,
        )
        card_rc = _run_keypoint_data_card_aggregation(
            cli=card_cli,
            required=True,
        )
        if card_rc != 0:
            return int(card_rc)

    print(f"\nWrote merged zarr: {merged_zarr}")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote manifest: {out_manifest}")
    print(f"Wrote config: {out_config}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
