#!/usr/bin/env python3
"""Export detection-training Zarr datasets from a training manifest.

Modes:
- Legacy copy mode (default): one exported Zarr per source dataset.
- Merged mode (`--merge`): one merged Zarr for the full training set.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml
import zarr
from zarr.core.dtype import VariableLengthUTF8

from fisheye.diagnostics.prepare_detect_training import DatasetManifest, TrainingManifest
from fisheye.registry.db import Registry
from fisheye.shared.detect_reason_codec import read_reason_labels
from fisheye.shared.refined_detect_curation import (
    REFINED_SOURCE_KIND_CODE_MAP,
    write_curated_refined_detect_surfaces,
)
from fisheye.shared.batch_logging import utc_now
from fisheye.shared.zarr_helpers import open_zarr_group_direct
from fisheye.shared.zarr_run_completion import (
    mark_run_complete,
    mark_run_started,
    require_runs_parent,
)
from fisheye.utils.system import build_invocation_record
from fisheye.utils.zarr_metadata import get_downsample_array_name, get_downsample_array_path

try:
    from rich.console import Console
    from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
except Exception:  # pragma: no cover - rich is optional
    Console = None  # type: ignore
    Progress = None  # type: ignore
    TextColumn = None  # type: ignore
    BarColumn = None  # type: ignore
    TaskProgressColumn = None  # type: ignore
    TimeElapsedColumn = None  # type: ignore
    TimeRemainingColumn = None  # type: ignore


@dataclass
class ExportPlan:
    dataset_name: str
    source_zarr: Path
    dest_zarr: Path
    bbox_path: str
    detection_source_path: Optional[str]
    input_format: str
    crop_run: Optional[str]
    includes_interpolated: bool
    detection_source_type: str


@dataclass
class MergeSourceSpec:
    ordinal: int
    dataset_name: str
    dataset_id: str
    source_zarr: Path
    bbox_path: str
    frame_indices_path: Optional[str]
    detection_source_path: Optional[str]
    detection_source_type: str
    sample_count: int
    has_gray: bool
    has_rgb: bool
    gray_shape: Optional[Tuple[int, ...]]
    rgb_shape: Optional[Tuple[int, ...]]
    gray_dtype: Optional[np.dtype]
    rgb_dtype: Optional[np.dtype]
    gray_chunks: Optional[Tuple[int, ...]]
    rgb_chunks: Optional[Tuple[int, ...]]
    fps: Optional[float]
    gray_pixel_contract_name: Optional[str] = None
    rgb_pixel_contract_name: Optional[str] = None
    gray_decode_backend: Optional[str] = None
    rgb_decode_backend: Optional[str] = None


@dataclass
class MergeResult:
    run_name: str
    total_samples: int
    total_real: int
    total_interpolated: int
    source_kind_counts: Dict[str, int]
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    source_specs: List[MergeSourceSpec]
    downsample_formats: List[str]
    training_input_format: str


_utc_now = utc_now


def _sha256(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _resolve_default_dataset_root() -> Path:
    env_override = os.getenv("PALETTE_TRAINING_DATASETS_ROOT")
    if env_override:
        return Path(env_override).expanduser().resolve()
    nvme_default = Path("/nvme1/training/datasets")
    if nvme_default.exists():
        return nvme_default
    return Path("datasets") / "detect"


def _chunk_slices(shape: Tuple[int, ...], chunks: Tuple[int, ...]) -> Iterable[Tuple[slice, ...]]:
    if not chunks:
        yield tuple(slice(0, size) for size in shape)
        return
    grid = [math.ceil(dim / chunk) for dim, chunk in zip(shape, chunks)]
    for index in np.ndindex(*grid):
        slices = []
        for dim_idx, chunk_idx in enumerate(index):
            start = chunk_idx * chunks[dim_idx]
            stop = min(start + chunks[dim_idx], shape[dim_idx])
            slices.append(slice(start, stop))
        yield tuple(slices)


def _copy_array(src: zarr.Array, dest_group: zarr.Group, name: str) -> zarr.Array:
    chunks = src.chunks if src.chunks is not None else None
    dest = dest_group.require_array(
        name,
        shape=src.shape,
        dtype=src.dtype,
        chunks=chunks,
        overwrite=True,
    )
    if chunks is None:
        dest[...] = src[...]
        return dest
    for slc in _chunk_slices(src.shape, chunks):
        dest[slc] = src[slc]
    return dest


def _copy_attrs(src: zarr.attrs.Attributes, dest: zarr.attrs.Attributes) -> None:
    dest.update(dict(src))


def _build_plan(manifest: TrainingManifest, out_dir: Path) -> List[ExportPlan]:
    plans: List[ExportPlan] = []
    for dataset in manifest.datasets:
        source_zarr = Path(dataset.zarr_path)
        dest_zarr = out_dir / "zarr" / f"{dataset.name}.zarr"
        plans.append(
            ExportPlan(
                dataset_name=dataset.name,
                source_zarr=source_zarr,
                dest_zarr=dest_zarr,
                bbox_path=dataset.bbox_array_path,
                detection_source_path=dataset.detection_source_path,
                input_format=dataset.input_format,
                crop_run=dataset.crop_run,
                includes_interpolated=dataset.includes_interpolated,
                detection_source_type=dataset.detection_source_type,
            )
        )
    return plans


def _export_dataset(plan: ExportPlan, *, overwrite: bool, run_name: str, manifest_path: Path) -> None:
    if plan.dest_zarr.exists():
        if not overwrite:
            raise FileExistsError(f"Destination exists: {plan.dest_zarr}")
        if plan.dest_zarr.is_dir():
            shutil.rmtree(plan.dest_zarr)
        else:
            plan.dest_zarr.unlink()

    plan.dest_zarr.parent.mkdir(parents=True, exist_ok=True)

    src_root = open_zarr_group_direct(plan.source_zarr, mode="r")
    dest_root = zarr.open_group(str(plan.dest_zarr), mode="w")

    raw_src = src_root["raw_video"]
    raw_dest = dest_root.require_group("raw_video")
    _copy_attrs(raw_src.attrs, raw_dest.attrs)
    array_name = get_downsample_array_name(src_root, format_hint=plan.input_format)
    if array_name is None:
        raise KeyError(f"Missing downsampled frames for format '{plan.input_format}' in {plan.source_zarr}")
    _copy_array(raw_src[array_name], raw_dest, array_name)

    crop_parent = require_runs_parent(dest_root, "crop_runs")
    crop_group = crop_parent.require_group(run_name)
    mark_run_started(crop_group, run_name=run_name, stage="crop")
    bbox_src = src_root[plan.bbox_path]
    _copy_array(bbox_src, crop_group, "bbox_norm_coords")
    total_samples = int(bbox_src.shape[0])

    frame_indices_path = None
    if plan.crop_run and "crop_runs" in src_root and plan.crop_run in src_root["crop_runs"]:
        if "frame_indices" in src_root["crop_runs"][plan.crop_run]:
            frame_indices_path = f"crop_runs/{plan.crop_run}/frame_indices"
    if frame_indices_path and frame_indices_path in src_root:
        _copy_array(src_root[frame_indices_path], crop_group, "frame_indices")

    if plan.crop_run and "crop_runs" in src_root and plan.crop_run in src_root["crop_runs"]:
        crop_src = src_root["crop_runs"][plan.crop_run]
        for name in ("detection_indices",):
            if name in crop_src and int(crop_src[name].shape[0]) == total_samples:
                _copy_array(crop_src[name], crop_group, name)

    source_refined_row_ids, source_detect_row_index = _resolve_source_row_identity(
        src_root,
        bbox_path=plan.bbox_path,
        crop_run=plan.crop_run,
        frame_indices_path=frame_indices_path,
        total_rows=total_samples,
    )
    crop_group.create_array(
        "source_refined_row_ids",
        data=source_refined_row_ids,
        chunks=(max(1, min(8192, total_samples)),),
        overwrite=True,
    )
    crop_group.create_array(
        "source_detect_row_index",
        data=source_detect_row_index,
        chunks=(max(1, min(8192, total_samples)),),
        overwrite=True,
    )
    crop_group.attrs["source_refined_row_ids_available"] = bool(np.any(source_refined_row_ids >= 0))
    crop_group.attrs["source_detect_row_index_available"] = bool(np.any(source_detect_row_index >= 0))

    if plan.detection_source_path and plan.detection_source_path in src_root:
        _copy_array(src_root[plan.detection_source_path], crop_group, "detection_source")
        detection_source = src_root[plan.detection_source_path][:]
        n_interpolated = int(np.sum(detection_source != 0))
        n_real = int(np.sum(detection_source == 0))
    else:
        n_real = int(bbox_src.shape[0])
        n_interpolated = 0

    crop_group.attrs.update(
        {
            "detection_source_type": plan.detection_source_type,
            "includes_interpolated": bool(plan.includes_interpolated),
            "n_real_detections": n_real,
            "n_interpolated_detections": n_interpolated,
            "source_bbox_path": plan.bbox_path,
            "source_detection_source_path": plan.detection_source_path,
            "source_crop_run": plan.crop_run,
            "source_zarr_path": str(plan.source_zarr),
        }
    )

    dest_root.attrs.update(
        {
            "zarr_purpose": "training",
            "training_export": {
                "created_at_utc": _utc_now(),
                "source_manifest": str(manifest_path),
                "source_manifest_sha256": _sha256(manifest_path),
                "source_zarr": str(plan.source_zarr),
                "source_bbox_path": plan.bbox_path,
                "source_frame_array_path": get_downsample_array_path(
                    src_root, format_hint=plan.input_format
                ),
            },
        }
    )
    mark_run_complete(crop_group, parent_group=crop_parent, run_name=run_name)


def _update_manifest_paths(
    manifest_payload: Dict[str, object],
    plans: List[ExportPlan],
    out_manifest: Path,
    out_config: Path,
    out_dir: Path,
) -> None:
    plan_map = {plan.dataset_name: plan for plan in plans}
    datasets = manifest_payload.get("datasets", [])
    for entry in datasets:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        plan = plan_map.get(str(name))
        if plan is None:
            continue
        entry["zarr_path"] = str(plan.dest_zarr)

    manifest_payload["output_manifest_path"] = str(out_manifest)
    manifest_payload["output_config_path"] = str(out_config)
    manifest_payload["exported_at_utc"] = _utc_now()
    manifest_payload["export_root"] = str(out_dir)


def _write_config(source_config: Path, out_config: Path, plans: List[ExportPlan]) -> None:
    cfg = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    datasets = cfg.get("datasets") or {}
    plan_map = {plan.dataset_name: plan for plan in plans}
    for name, dataset_cfg in datasets.items():
        plan = plan_map.get(str(name))
        if plan is None or not isinstance(dataset_cfg, dict):
            continue
        dataset_cfg["zarr_path"] = str(plan.dest_zarr)
    out_config.parent.mkdir(parents=True, exist_ok=True)
    out_config.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


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


def _normalize_manifest_stem(value: str) -> str:
    text = str(value).strip()
    while text.endswith(".manifest"):
        text = text[: -len(".manifest")]
    return text


def _ensure_suffix(value: str, suffix: str) -> str:
    text = str(value).strip()
    return text if text.endswith(suffix) else f"{text}{suffix}"


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


def _normalize_input_format(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"gray", "grey", "grayscale"}:
        return "gray"
    if text in {"rgb", "color", "colour"}:
        return "rgb"
    if text == "both":
        return "both"
    return None


_PIXEL_CONTRACT_NAME_ATTRS = (
    "pixel_contract_name",
    "raw_pixel_contract_name",
    "source_pixel_contract_name",
)
_DECODE_BACKEND_ATTRS = (
    "decode_backend",
    "decode_backend_effective",
    "source_decode_backend",
)


def _as_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _first_attr_text(attrs: Any, names: Sequence[str]) -> Optional[str]:
    for name in names:
        try:
            value = attrs.get(name)
        except Exception:
            value = None
        text = _as_text(value)
        if text:
            return text
    return None


def _raw_image_attr_text(raw_group: zarr.Group, array_name: str, names: Sequence[str]) -> Optional[str]:
    if array_name in raw_group:
        text = _first_attr_text(raw_group[array_name].attrs, names)
        if text:
            return text
    return _first_attr_text(raw_group.attrs, names)


def _collapse_metadata_values(
    values: Sequence[Optional[str]],
    *,
    mixed_label: str,
    partial_label: str,
) -> Optional[str]:
    observed = [str(value) for value in values if value]
    if not observed:
        return None
    if len(observed) != len(values):
        return partial_label
    unique = sorted(set(observed))
    if len(unique) == 1:
        return unique[0]
    if len(unique) > 1:
        return mixed_label
    return None


def _format_raw_metadata_summary(
    source_specs: Sequence[MergeSourceSpec],
    *,
    need_gray: bool,
    need_rgb: bool,
) -> dict[str, Any]:
    contract_names_by_format: dict[str, list[str]] = {}
    decode_backends_by_format: dict[str, list[str]] = {}
    if need_gray:
        contract_names_by_format["gray"] = sorted(
            {str(spec.gray_pixel_contract_name) for spec in source_specs if spec.gray_pixel_contract_name}
        )
        decode_backends_by_format["gray"] = sorted(
            {str(spec.gray_decode_backend) for spec in source_specs if spec.gray_decode_backend}
        )
    if need_rgb:
        contract_names_by_format["rgb"] = sorted(
            {str(spec.rgb_pixel_contract_name) for spec in source_specs if spec.rgb_pixel_contract_name}
        )
        decode_backends_by_format["rgb"] = sorted(
            {str(spec.rgb_decode_backend) for spec in source_specs if spec.rgb_decode_backend}
        )
    return {
        "pixel_contract_names_by_format": contract_names_by_format,
        "decode_backends_by_format": decode_backends_by_format,
        "gray_pixel_contract_name": _collapse_metadata_values(
            [spec.gray_pixel_contract_name for spec in source_specs],
            mixed_label="mixed_raw_video_pixel_contracts",
            partial_label="partial_raw_video_pixel_contracts",
        ),
        "rgb_pixel_contract_name": _collapse_metadata_values(
            [spec.rgb_pixel_contract_name for spec in source_specs],
            mixed_label="mixed_raw_video_pixel_contracts",
            partial_label="partial_raw_video_pixel_contracts",
        ),
        "gray_decode_backend": _collapse_metadata_values(
            [spec.gray_decode_backend for spec in source_specs],
            mixed_label="mixed_decode_backends",
            partial_label="partial_decode_backends",
        ),
        "rgb_decode_backend": _collapse_metadata_values(
            [spec.rgb_decode_backend for spec in source_specs],
            mixed_label="mixed_decode_backends",
            partial_label="partial_decode_backends",
        ),
    }


def validate_merged_training_zarr(
    zarr_path: Path,
    *,
    expected_input_format: Optional[str] = None,
    expected_total_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Validate current merged training Zarr layout and trainer-facing invariants."""
    root = zarr.open_group(str(zarr_path), mode="r")
    errors: List[str] = []

    if str(root.attrs.get("zarr_purpose", "")).strip().lower() != "training":
        errors.append("root attr zarr_purpose must be 'training'.")

    if "raw_video" not in root:
        errors.append("missing group raw_video.")
    if "refined_detect_runs" not in root:
        errors.append("missing group refined_detect_runs.")
    if "splits" not in root:
        errors.append("missing group splits.")
    if "source_index" not in root:
        errors.append("missing group source_index.")

    if errors:
        raise ValueError("Merged training zarr validation failed:\n- " + "\n- ".join(errors))

    raw = root["raw_video"]
    refined_parent = root["refined_detect_runs"]

    train_input_format = _normalize_input_format(expected_input_format)
    if train_input_format is None:
        training_export = root.attrs.get("training_export")
        if isinstance(training_export, dict):
            train_input_format = _normalize_input_format(training_export.get("input_format"))

    if train_input_format == "gray" and "images_ds" not in raw:
        errors.append("missing required array raw_video/images_ds for gray training input.")
    if train_input_format == "rgb" and "images_ds_rgb" not in raw:
        errors.append("missing required array raw_video/images_ds_rgb for rgb training input.")
    if "images_ds" not in raw and "images_ds_rgb" not in raw:
        errors.append("raw_video must include at least one of images_ds or images_ds_rgb.")

    latest_run = refined_parent.attrs.get("latest")
    if not latest_run or latest_run not in refined_parent:
        errors.append("refined_detect_runs/latest missing or points to a non-existent run.")
        raise ValueError("Merged training zarr validation failed:\n- " + "\n- ".join(errors))
    instances_path = f"refined_detect_runs/{latest_run}/instances"
    if instances_path not in root:
        errors.append(f"missing required group {instances_path}.")
        raise ValueError("Merged training zarr validation failed:\n- " + "\n- ".join(errors))

    bbox_path = f"{instances_path}/bbox_norm_coords"
    frame_idx_path = f"{instances_path}/frame_indices"
    source_kind_path = f"{instances_path}/source_kind_codes"
    manual_flags_path = f"{instances_path}/manual_edit_flags"
    frame_counts_path = f"{instances_path}/frame_counts"
    source_detect_path = f"{instances_path}/source_detect_row_index"
    for path in (bbox_path, frame_idx_path, source_kind_path, manual_flags_path, frame_counts_path, source_detect_path):
        if path not in root:
            errors.append(f"missing required array {path}.")

    if errors:
        raise ValueError("Merged training zarr validation failed:\n- " + "\n- ".join(errors))

    bbox = np.asarray(root[bbox_path][:])
    frame_indices = np.asarray(root[frame_idx_path][:])
    source_kind_codes = np.asarray(root[source_kind_path][:])
    manual_edit_flags = np.asarray(root[manual_flags_path][:])
    frame_counts = np.asarray(root[frame_counts_path][:])
    source_detect_row_index = np.asarray(root[source_detect_path][:])

    if bbox.ndim != 2 or int(bbox.shape[1]) != 4:
        errors.append(f"{bbox_path} must have shape (N, 4), got {tuple(int(v) for v in bbox.shape)}.")
    total_samples = int(bbox.shape[0]) if bbox.ndim >= 1 else 0

    if frame_indices.ndim != 1:
        errors.append(f"{frame_idx_path} must be 1D, got ndim={frame_indices.ndim}.")
    if source_kind_codes.ndim != 1:
        errors.append(f"{source_kind_path} must be 1D, got ndim={source_kind_codes.ndim}.")
    if manual_edit_flags.ndim != 1:
        errors.append(f"{manual_flags_path} must be 1D, got ndim={manual_edit_flags.ndim}.")
    if source_detect_row_index.ndim != 1:
        errors.append(f"{source_detect_path} must be 1D, got ndim={source_detect_row_index.ndim}.")
    if frame_counts.ndim != 1:
        errors.append(f"{frame_counts_path} must be 1D, got ndim={frame_counts.ndim}.")
    if frame_indices.ndim == 1 and frame_indices.shape[0] != total_samples:
        errors.append(f"{frame_idx_path} length mismatch ({frame_indices.shape[0]} != {total_samples}).")
    for path, arr in (
        (source_kind_path, source_kind_codes),
        (manual_flags_path, manual_edit_flags),
        (source_detect_path, source_detect_row_index),
    ):
        if arr.ndim == 1 and arr.shape[0] != total_samples:
            errors.append(f"{path} length mismatch ({arr.shape[0]} != {total_samples}).")

    if frame_indices.ndim == 1 and not np.issubdtype(frame_indices.dtype, np.integer):
        errors.append(f"{frame_idx_path} must be integer dtype, got {frame_indices.dtype}.")
    if source_kind_codes.ndim == 1 and not np.issubdtype(source_kind_codes.dtype, np.integer):
        errors.append(f"{source_kind_path} must be integer dtype, got {source_kind_codes.dtype}.")
    if source_detect_row_index.ndim == 1 and not np.issubdtype(source_detect_row_index.dtype, np.integer):
        errors.append(f"{source_detect_path} must be integer dtype, got {source_detect_row_index.dtype}.")
    if manual_edit_flags.ndim == 1 and not np.issubdtype(manual_edit_flags.dtype, np.bool_):
        errors.append(f"{manual_flags_path} must be bool dtype, got {manual_edit_flags.dtype}.")
    if frame_indices.ndim == 1 and frame_indices.shape[0] == total_samples:
        expected_local = np.arange(total_samples, dtype=np.int64)
        frame_indices_i64 = frame_indices.astype(np.int64, copy=False)
        if not np.array_equal(frame_indices_i64, expected_local):
            errors.append(f"{frame_idx_path} must be local 0..N-1 indexing.")
    if source_kind_codes.ndim == 1 and source_kind_codes.size > 0:
        source_codes = np.unique(source_kind_codes.astype(np.int64, copy=False))
        valid_codes = {int(value) for value in REFINED_SOURCE_KIND_CODE_MAP.values()}
        invalid_codes = [int(code) for code in source_codes.tolist() if int(code) not in valid_codes]
        if invalid_codes:
            errors.append(f"{source_kind_path} contains invalid codes: {sorted(set(invalid_codes))}.")
        interpolated_code = int(REFINED_SOURCE_KIND_CODE_MAP["interpolated"])
        interpolated_count = int(np.count_nonzero(source_kind_codes == interpolated_code))
        if interpolated_count:
            errors.append(
                f"{source_kind_path} contains {interpolated_count} interpolated rows; "
                "new merged exports must be interpolation-free."
            )
    if frame_counts.ndim == 1 and int(frame_counts.shape[0]) != total_samples:
        errors.append(f"{frame_counts_path} length must equal merged frame count ({frame_counts.shape[0]} != {total_samples}).")
    if frame_counts.ndim == 1 and frame_counts.shape[0] == total_samples:
        if not np.all(frame_counts.astype(np.int64, copy=False) == 1):
            errors.append(f"{frame_counts_path} must contain exactly one instance per merged training frame.")

    if expected_total_samples is not None and int(expected_total_samples) != total_samples:
        errors.append(f"total sample mismatch ({total_samples} != expected {int(expected_total_samples)}).")

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
            errors.append("split coverage must be exact and non-duplicated across all split arrays.")

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

        optional_source_vectors = (
            ("source_index/source_roi_idx", np.integer, True),
            ("source_index/source_refined_row_ids", np.integer, False),
            ("source_index/source_detect_row_index", np.integer, False),
        )
        for opt_path, dtype_kind, require_nonnegative in optional_source_vectors:
            if opt_path not in root:
                continue
            opt_arr = np.asarray(root[opt_path][:])
            if opt_arr.ndim != 1 or opt_arr.shape[0] != total_samples:
                errors.append(f"{opt_path} must be 1D length N ({total_samples}), got {opt_arr.shape}.")
                continue
            if not np.issubdtype(opt_arr.dtype, dtype_kind):
                errors.append(f"{opt_path} must be integer dtype, got {opt_arr.dtype}.")
                continue
            opt_i64 = opt_arr.astype(np.int64, copy=False)
            if require_nonnegative and opt_i64.size > 0 and int(opt_i64.min()) < 0:
                errors.append(f"{opt_path} contains negative indices.")

    if errors:
        raise ValueError("Merged training zarr validation failed:\n- " + "\n- ".join(errors))

    return {
        "zarr_path": str(zarr_path),
        "run_name": str(latest_run),
        "instances_path": instances_path,
        "training_input_format": train_input_format,
        "total_samples": int(total_samples),
        "split_counts": {
            "train": int(train_idx.shape[0]),
            "val": int(val_idx.shape[0]),
            "test": int(test_idx.shape[0]),
        },
        "source_count": int(np.asarray(root[src_dataset_id_path][:]).shape[0]),
    }


def _resolve_frame_indices_path(root: zarr.Group, dataset: DatasetManifest) -> Optional[str]:
    candidates: List[str] = []
    if dataset.crop_run:
        candidates.append(f"crop_runs/{dataset.crop_run}/frame_indices")
    bbox_path = dataset.bbox_array_path
    if bbox_path.endswith("/bbox_norm_coords"):
        candidates.append(f"{bbox_path.rsplit('/', 1)[0]}/frame_indices")
    if bbox_path.startswith("detect_runs/") and bbox_path.count("/") >= 2:
        run = bbox_path.split("/")[1]
        candidates.append(f"detect_runs/{run}/frame_indices")
    for candidate in candidates:
        if candidate in root:
            return candidate
    return None


def _resolve_detection_source_path(root: zarr.Group, dataset: DatasetManifest) -> Optional[str]:
    candidates: List[str] = []
    if dataset.crop_run:
        candidates.append(f"crop_runs/{dataset.crop_run}/detection_source")
    bbox_path = dataset.bbox_array_path
    if bbox_path.endswith("/bbox_norm_coords"):
        candidates.append(f"{bbox_path.rsplit('/', 1)[0]}/detection_source")
    if dataset.detection_source_path:
        source_path = dataset.detection_source_path.rstrip("/")
        candidates.append(source_path)
        if not source_path.endswith("/detection_source"):
            candidates.append(f"{source_path}/detection_source")
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate not in root:
            continue
        if isinstance(root[candidate], zarr.Array):
            return candidate
    return None


def _resolve_bbox_parent_group(root: zarr.Group, bbox_path: str) -> Optional[zarr.Group]:
    if "/" not in bbox_path:
        return None
    parent_path = bbox_path.rsplit("/", 1)[0]
    if parent_path not in root:
        return None
    group = root[parent_path]
    return group if isinstance(group, zarr.Group) else None


def _read_optional_source_vector(
    group: Optional[zarr.Group],
    name: str,
    *,
    expected_len: int,
    dtype: np.dtype | type,
) -> Optional[np.ndarray]:
    if group is None or name not in group:
        return None
    arr = np.asarray(group[name][:], dtype=dtype).reshape(-1)
    if int(arr.shape[0]) != int(expected_len):
        raise ValueError(f"{name} length mismatch ({arr.shape[0]} != {expected_len}).")
    return arr


def _source_kind_payload_for_merge(
    *,
    source_zarr: Path,
    spec: MergeSourceSpec,
    source_group: Optional[zarr.Group],
    detection_source: np.ndarray,
    row_count: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Dict[str, int]]:
    """Return canonical refined-instance payload vectors for one source dataset.

    New merged exports are canonical sparse refined-detect datasets. Legacy
    ``detection_source`` is still read only to reject old interpolated rows before
    they are copied into a new export.
    """

    reverse_kind_map = {int(value): str(key) for key, value in REFINED_SOURCE_KIND_CODE_MAP.items()}
    source_kind_codes = _read_optional_source_vector(
        source_group,
        "source_kind_codes",
        expected_len=row_count,
        dtype=np.int8,
    )
    if source_kind_codes is not None:
        invalid_codes = sorted(
            {
                int(code)
                for code in np.unique(source_kind_codes.astype(np.int64, copy=False)).tolist()
                if int(code) not in reverse_kind_map
            }
        )
        if invalid_codes:
            raise ValueError(f"{source_zarr}: source_kind_codes contains unknown codes: {invalid_codes}.")
        interpolated_code = int(REFINED_SOURCE_KIND_CODE_MAP["interpolated"])
        interpolated_count = int(np.count_nonzero(source_kind_codes == interpolated_code))
        if interpolated_count:
            raise ValueError(
                f"{source_zarr}: merged canonical detection export refuses {interpolated_count} "
                "legacy interpolated rows. Review/migrate/exclude them before exporting."
            )
        source_kind_labels = np.asarray(
            [reverse_kind_map[int(code)] for code in source_kind_codes.tolist()],
            dtype=object,
        )
    else:
        if detection_source.shape[0] != row_count:
            raise ValueError(
                f"{source_zarr}: detection_source length mismatch ({detection_source.shape[0]} != {row_count})."
            )
        interpolated_count = int(np.count_nonzero(detection_source != 0))
        if interpolated_count:
            raise ValueError(
                f"{source_zarr}: merged canonical detection export refuses {interpolated_count} "
                "legacy interpolated detection_source rows. Review/migrate/exclude them before exporting."
            )
        default_kind = "manual" if str(spec.detection_source_type).strip().lower() == "manual" else "raw_detect"
        source_kind_labels = np.full(row_count, default_kind, dtype=object)

    manual_edit_flags = _read_optional_source_vector(
        source_group,
        "manual_edit_flags",
        expected_len=row_count,
        dtype=bool,
    )
    if manual_edit_flags is None:
        manual_edit_flags = source_kind_labels == "manual"

    reason_labels = read_reason_labels(source_group) if source_group is not None else None
    if reason_labels is None or int(np.asarray(reason_labels).reshape(-1).shape[0]) != row_count:
        reason_labels = np.asarray(source_kind_labels, dtype=object)
    else:
        reason_labels = np.asarray(reason_labels, dtype=object).reshape(-1)

    confidence_scores = _read_optional_source_vector(
        source_group,
        "confidence_scores",
        expected_len=row_count,
        dtype=np.float32,
    )
    if confidence_scores is None:
        confidence_scores = _read_optional_source_vector(
            source_group,
            "scores",
            expected_len=row_count,
            dtype=np.float32,
        )
    class_ids = _read_optional_source_vector(
        source_group,
        "class_ids",
        expected_len=row_count,
        dtype=np.int32,
    )

    code_by_label = {str(key): int(value) for key, value in REFINED_SOURCE_KIND_CODE_MAP.items()}
    source_kind_counts: Dict[str, int] = {}
    for label in source_kind_labels.tolist():
        key = str(code_by_label[str(label)])
        source_kind_counts[key] = source_kind_counts.get(key, 0) + 1
    return (
        source_kind_labels,
        manual_edit_flags.astype(bool, copy=False),
        reason_labels,
        confidence_scores,
        class_ids,
        source_kind_counts,
    )


def _sanitize_dataset_id(dataset: DatasetManifest, ordinal: int) -> str:
    if dataset.dataset_id:
        return dataset.dataset_id
    if dataset.session_uuid:
        return dataset.session_uuid
    return f"{dataset.name}_{ordinal}"


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


def _copy_indexed_frames(
    src_array: zarr.Array,
    frame_indices: np.ndarray,
    dest_array: zarr.Array,
    dest_start: int,
    *,
    batch_size: int = 128,
    on_copied: Optional[Callable[[int], None]] = None,
) -> None:
    if frame_indices.size == 0:
        return
    batch_size = max(1, int(batch_size))
    total = int(frame_indices.shape[0])

    # Fast path: fully contiguous source slice can be copied in one operation.
    if total > 0:
        first = int(frame_indices[0])
        expected = np.arange(first, first + total, dtype=np.int64)
        if np.array_equal(frame_indices.astype(np.int64, copy=False), expected):
            dest_array[dest_start : dest_start + total] = src_array[first : first + total]
            if on_copied is not None:
                on_copied(total)
            return

    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        selection = frame_indices[start:stop]
        if selection.size == 0:
            continue

        # Secondary fast path: batch itself is contiguous.
        if selection.size > 1 and np.all(np.diff(selection.astype(np.int64, copy=False)) == 1):
            batch_first = int(selection[0])
            batch_last = int(selection[-1]) + 1
            batch = src_array[batch_first:batch_last, ...]
        else:
            try:
                batch = src_array.vindex[selection, ...]
            except Exception:
                batch = np.stack([src_array[int(idx)] for idx in selection], axis=0)
        dest_array[dest_start + start : dest_start + stop] = batch
        if on_copied is not None:
            on_copied(stop - start)


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


def _read_optional_vector(
    group: zarr.Group,
    names: Sequence[str],
    *,
    expected_len: int,
    dtype: np.dtype | type,
) -> Optional[np.ndarray]:
    for name in names:
        if name not in group:
            continue
        arr = np.asarray(group[name][:], dtype=dtype)
        if arr.ndim == 1 and int(arr.shape[0]) == int(expected_len):
            return arr
    return None


def _candidate_lineage_groups(
    root: zarr.Group,
    *,
    bbox_path: str,
    crop_run: Optional[str] = None,
    frame_indices_path: Optional[str] = None,
) -> List[zarr.Group]:
    paths: List[str] = []
    if crop_run:
        paths.append(f"crop_runs/{crop_run}")
    if frame_indices_path and "/" in frame_indices_path:
        paths.append(frame_indices_path.rsplit("/", 1)[0])
    if bbox_path and "/" in bbox_path:
        paths.append(bbox_path.rsplit("/", 1)[0])

    groups: List[zarr.Group] = []
    seen: set[str] = set()
    for path in paths:
        if not path or path in seen or path not in root:
            continue
        seen.add(path)
        group = root[path]
        if isinstance(group, zarr.Group):
            groups.append(group)
    return groups


def _resolve_source_row_identity(
    root: zarr.Group,
    *,
    bbox_path: str,
    total_rows: int,
    crop_run: Optional[str] = None,
    frame_indices_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    refined = None
    detect = None
    for group in _candidate_lineage_groups(
        root,
        bbox_path=bbox_path,
        crop_run=crop_run,
        frame_indices_path=frame_indices_path,
    ):
        if refined is None:
            refined = _read_optional_vector(
                group,
                ("source_refined_row_ids", "refined_row_ids"),
                expected_len=total_rows,
                dtype=np.int64,
            )
        if detect is None:
            detect = _read_optional_vector(
                group,
                ("source_detect_row_index",),
                expected_len=total_rows,
                dtype=np.int32,
            )
        if refined is not None and detect is not None:
            break

    if refined is None:
        refined = np.full((total_rows,), -1, dtype=np.int64)
    else:
        refined = refined.astype(np.int64, copy=False)
    if detect is None:
        detect = np.full((total_rows,), -1, dtype=np.int32)
    else:
        detect = detect.astype(np.int32, copy=False)
    return refined, detect


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
                task_type="detect",
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
            metadata={"producer": "export_detect_training_zarr"},
        )
        return str(dataset_id), linked
    finally:
        registry.close()


def _discover_merge_sources(
    manifest: TrainingManifest,
    *,
    need_gray: bool,
    need_rgb: bool,
    required_pixel_contract_name: Optional[str] = None,
) -> Tuple[List[MergeSourceSpec], Dict[str, Any]]:
    specs: List[MergeSourceSpec] = []
    gray_shape: Optional[Tuple[int, ...]] = None
    rgb_shape: Optional[Tuple[int, ...]] = None
    gray_dtype: Optional[np.dtype] = None
    rgb_dtype: Optional[np.dtype] = None
    gray_chunks: Optional[Tuple[int, ...]] = None
    rgb_chunks: Optional[Tuple[int, ...]] = None
    fps_values: List[float] = []

    for ordinal, dataset in enumerate(manifest.datasets):
        source_zarr = Path(dataset.zarr_path)
        root = open_zarr_group_direct(source_zarr, mode="r")
        if "raw_video" not in root:
            raise KeyError(f"{source_zarr}: missing raw_video group.")
        raw = root["raw_video"]

        has_gray = "images_ds" in raw
        has_rgb = "images_ds_rgb" in raw
        if need_gray and not has_gray:
            raise KeyError(f"{source_zarr}: required raw_video/images_ds is missing.")
        if need_rgb and not has_rgb:
            raise KeyError(f"{source_zarr}: required raw_video/images_ds_rgb is missing.")

        bbox = root[dataset.bbox_array_path]
        if len(bbox.shape) != 2 or int(bbox.shape[1]) != 4:
            raise ValueError(f"{source_zarr}: bbox array '{dataset.bbox_array_path}' must have shape (N, 4).")
        sample_count = int(bbox.shape[0])

        frame_indices_path = _resolve_frame_indices_path(root, dataset)
        detection_source_path = _resolve_detection_source_path(root, dataset)

        dataset_gray_shape = tuple(int(dim) for dim in raw["images_ds"].shape[1:]) if has_gray else None
        dataset_rgb_shape = tuple(int(dim) for dim in raw["images_ds_rgb"].shape[1:]) if has_rgb else None
        dataset_gray_dtype = np.dtype(raw["images_ds"].dtype) if has_gray else None
        dataset_rgb_dtype = np.dtype(raw["images_ds_rgb"].dtype) if has_rgb else None
        dataset_gray_chunks = tuple(int(c) for c in raw["images_ds"].chunks) if has_gray and raw["images_ds"].chunks else None
        dataset_rgb_chunks = (
            tuple(int(c) for c in raw["images_ds_rgb"].chunks)
            if has_rgb and raw["images_ds_rgb"].chunks
            else None
        )
        dataset_gray_pixel_contract_name = (
            _raw_image_attr_text(raw, "images_ds", _PIXEL_CONTRACT_NAME_ATTRS)
            if has_gray
            else None
        )
        dataset_rgb_pixel_contract_name = (
            _raw_image_attr_text(raw, "images_ds_rgb", _PIXEL_CONTRACT_NAME_ATTRS)
            if has_rgb
            else None
        )
        dataset_gray_decode_backend = (
            _raw_image_attr_text(raw, "images_ds", _DECODE_BACKEND_ATTRS)
            if has_gray
            else None
        )
        dataset_rgb_decode_backend = (
            _raw_image_attr_text(raw, "images_ds_rgb", _DECODE_BACKEND_ATTRS)
            if has_rgb
            else None
        )

        required_contract = _as_text(required_pixel_contract_name)
        if required_contract:
            required_checks: list[tuple[str, Optional[str]]] = []
            if need_gray:
                required_checks.append(("raw_video/images_ds", dataset_gray_pixel_contract_name))
            if need_rgb:
                required_checks.append(("raw_video/images_ds_rgb", dataset_rgb_pixel_contract_name))
            for surface_name, observed_name in required_checks:
                if observed_name != required_contract:
                    raise ValueError(
                        f"{source_zarr}: {surface_name} pixel contract mismatch "
                        f"({observed_name!r} != {required_contract!r})."
                    )

        if need_gray and dataset_gray_shape is not None:
            if gray_shape is None:
                gray_shape = dataset_gray_shape
                gray_dtype = dataset_gray_dtype
                gray_chunks = dataset_gray_chunks
            else:
                if dataset_gray_shape != gray_shape:
                    raise ValueError(
                        f"{source_zarr}: gray shape mismatch {dataset_gray_shape} != {gray_shape}."
                    )
                if dataset_gray_dtype != gray_dtype:
                    raise ValueError(
                        f"{source_zarr}: gray dtype mismatch {dataset_gray_dtype} != {gray_dtype}."
                    )
        if need_rgb and dataset_rgb_shape is not None:
            if rgb_shape is None:
                rgb_shape = dataset_rgb_shape
                rgb_dtype = dataset_rgb_dtype
                rgb_chunks = dataset_rgb_chunks
            else:
                if dataset_rgb_shape != rgb_shape:
                    raise ValueError(
                        f"{source_zarr}: rgb shape mismatch {dataset_rgb_shape} != {rgb_shape}."
                    )
                if dataset_rgb_dtype != rgb_dtype:
                    raise ValueError(
                        f"{source_zarr}: rgb dtype mismatch {dataset_rgb_dtype} != {rgb_dtype}."
                    )

        fps = raw.attrs.get("fps")
        if isinstance(fps, (int, float)):
            fps_values.append(float(fps))

        specs.append(
            MergeSourceSpec(
                ordinal=ordinal,
                dataset_name=dataset.name,
                dataset_id=_sanitize_dataset_id(dataset, ordinal),
                source_zarr=source_zarr,
                bbox_path=dataset.bbox_array_path,
                frame_indices_path=frame_indices_path,
                detection_source_path=detection_source_path,
                detection_source_type=dataset.detection_source_type,
                sample_count=sample_count,
                has_gray=has_gray,
                has_rgb=has_rgb,
                gray_shape=dataset_gray_shape,
                rgb_shape=dataset_rgb_shape,
                gray_dtype=dataset_gray_dtype,
                rgb_dtype=dataset_rgb_dtype,
                gray_chunks=dataset_gray_chunks,
                rgb_chunks=dataset_rgb_chunks,
                fps=float(fps) if isinstance(fps, (int, float)) else None,
                gray_pixel_contract_name=dataset_gray_pixel_contract_name,
                rgb_pixel_contract_name=dataset_rgb_pixel_contract_name,
                gray_decode_backend=dataset_gray_decode_backend,
                rgb_decode_backend=dataset_rgb_decode_backend,
            )
        )

    layout = {
        "gray_shape": gray_shape,
        "rgb_shape": rgb_shape,
        "gray_dtype": gray_dtype,
        "rgb_dtype": rgb_dtype,
        "gray_chunks": gray_chunks,
        "rgb_chunks": rgb_chunks,
        "fps_values": fps_values,
    }
    return specs, layout


def _ensure_writable_destination(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Destination exists: {path}")
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)


def _export_merged(
    *,
    manifest: TrainingManifest,
    manifest_path: Path,
    out_zarr: Path,
    merged_dataset_id: Optional[str],
    overwrite: bool,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    include_rgb: bool,
    copy_batch_size: int,
    invocation: Dict[str, Any],
    required_pixel_contract_name: Optional[str] = None,
) -> MergeResult:
    train_input_format = str(manifest.input_format).strip().lower()
    if train_input_format not in {"gray", "rgb"}:
        raise ValueError(f"Unsupported manifest.input_format '{manifest.input_format}'. Expected gray or rgb.")
    need_gray = train_input_format == "gray"
    need_rgb = train_input_format == "rgb" or include_rgb

    source_specs, layout = _discover_merge_sources(
        manifest,
        need_gray=need_gray,
        need_rgb=need_rgb,
        required_pixel_contract_name=required_pixel_contract_name,
    )
    total_samples = int(sum(spec.sample_count for spec in source_specs))

    _ensure_writable_destination(out_zarr, overwrite=overwrite)
    out_root = zarr.open_group(str(out_zarr), mode="w")
    raw_group = out_root.require_group("raw_video")
    refined_parent = require_runs_parent(out_root, "refined_detect_runs")
    source_index_group = out_root.require_group("source_index")
    split_group = out_root.require_group("splits")

    run_name = f"merged_export_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    refined_run = refined_parent.require_group(run_name)
    mark_run_started(refined_run, run_name=run_name, stage="refined_detect")

    vector_chunks = (max(1, min(8192, total_samples)),)

    merged_bbox_norm = np.empty((total_samples, 4), dtype=np.float32)
    merged_source_kind_labels = np.empty((total_samples,), dtype=object)
    merged_reason_labels = np.empty((total_samples,), dtype=object)
    merged_manual_edit_flags = np.zeros((total_samples,), dtype=bool)
    merged_source_detect_row_index = np.full((total_samples,), -1, dtype=np.int32)
    merged_confidence_scores: Optional[np.ndarray] = None
    merged_class_ids: Optional[np.ndarray] = None

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

    gray_dest = None
    rgb_dest = None
    if need_gray:
        gray_shape = tuple(layout["gray_shape"])
        gray_dtype = np.dtype(layout["gray_dtype"])
        gray_chunks = _normalize_chunks(layout["gray_chunks"], (total_samples, *gray_shape))
        gray_dest = raw_group.require_array(
            "images_ds",
            shape=(total_samples, *gray_shape),
            dtype=gray_dtype,
            chunks=gray_chunks,
            overwrite=True,
        )
        raw_metadata = _format_raw_metadata_summary(source_specs, need_gray=True, need_rgb=False)
        if raw_metadata["gray_pixel_contract_name"]:
            gray_dest.attrs["pixel_contract_name"] = raw_metadata["gray_pixel_contract_name"]
        if raw_metadata["gray_decode_backend"]:
            gray_dest.attrs["decode_backend"] = raw_metadata["gray_decode_backend"]
    if need_rgb:
        rgb_shape = tuple(layout["rgb_shape"])
        rgb_dtype = np.dtype(layout["rgb_dtype"])
        rgb_chunks = _normalize_chunks(layout["rgb_chunks"], (total_samples, *rgb_shape))
        rgb_dest = raw_group.require_array(
            "images_ds_rgb",
            shape=(total_samples, *rgb_shape),
            dtype=rgb_dtype,
            chunks=rgb_chunks,
            overwrite=True,
        )
        raw_metadata = _format_raw_metadata_summary(source_specs, need_gray=False, need_rgb=True)
        if raw_metadata["rgb_pixel_contract_name"]:
            rgb_dest.attrs["pixel_contract_name"] = raw_metadata["rgb_pixel_contract_name"]
        if raw_metadata["rgb_decode_backend"]:
            rgb_dest.attrs["decode_backend"] = raw_metadata["rgb_decode_backend"]

    resolution: Optional[List[int]] = None
    if train_input_format == "gray" and layout["gray_shape"] is not None:
        gray_shape = tuple(layout["gray_shape"])
        resolution = [int(gray_shape[0]), int(gray_shape[1])]
    elif layout["rgb_shape"] is not None:
        rgb_shape = tuple(layout["rgb_shape"])
        resolution = [int(rgb_shape[0]), int(rgb_shape[1])]
    if resolution is None:
        raise ValueError("Could not resolve merged training image resolution.")
    image_height, image_width = int(resolution[0]), int(resolution[1])
    out_root.attrs.update(
        {
            "total_frames": int(total_samples),
            "n_frames": int(total_samples),
            "height": image_height,
            "width": image_width,
            "inference_height": image_height,
            "inference_width": image_width,
        }
    )
    refined_run.attrs.update(
        {
            "coverage_frames_total": int(total_samples),
            "detection_source_type": "refined",
            "interpolation_enabled": False,
            "interpolation_policy": "forbidden_for_merged_training_export",
        }
    )

    total_real = 0
    total_interpolated = 0
    merged_source_kind_counts: Dict[str, int] = {}

    copy_passes = int(bool(need_gray)) + int(bool(need_rgb))
    copy_total = total_samples * copy_passes
    copy_progress = _copy_progress(copy_total)
    copy_task_id: Optional[int] = None
    if copy_progress is not None:
        copy_progress.start()
        copy_task_id = copy_progress.add_task("Copying merged samples", total=copy_total)

    def _advance_copy(count: int) -> None:
        if copy_progress is not None and copy_task_id is not None:
            copy_progress.advance(copy_task_id, count)

    offset = 0
    try:
        for spec in source_specs:
            if copy_progress is not None and copy_task_id is not None:
                copy_progress.update(copy_task_id, description=f"Copying {spec.dataset_name}")

            root = open_zarr_group_direct(spec.source_zarr, mode="r")
            bbox = np.asarray(root[spec.bbox_path][:], dtype=np.float32)
            local_total = int(bbox.shape[0])
            source_refined_row_ids, source_detect_row_index = _resolve_source_row_identity(
                root,
                bbox_path=spec.bbox_path,
                frame_indices_path=spec.frame_indices_path,
                total_rows=local_total,
            )

            if spec.frame_indices_path:
                src_frame_indices = np.asarray(root[spec.frame_indices_path][:], dtype=np.int64)
            else:
                src_frame_indices = np.arange(local_total, dtype=np.int64)
            if src_frame_indices.shape[0] != local_total:
                raise ValueError(
                    f"{spec.source_zarr}: frame_indices length mismatch "
                    f"({src_frame_indices.shape[0]} != {local_total})."
                )

            if spec.detection_source_path:
                detection_source = np.asarray(root[spec.detection_source_path][:], dtype=np.int8)
                if detection_source.shape[0] != local_total:
                    raise ValueError(
                        f"{spec.source_zarr}: detection_source length mismatch "
                        f"({detection_source.shape[0]} != {local_total})."
                    )
            else:
                detection_source = np.zeros(local_total, dtype=np.int8)

            source_group = _resolve_bbox_parent_group(root, spec.bbox_path)
            (
                source_kind_labels,
                manual_edit_flags,
                reason_labels,
                confidence_scores,
                class_ids,
                local_source_kind_counts,
            ) = _source_kind_payload_for_merge(
                source_zarr=spec.source_zarr,
                spec=spec,
                source_group=source_group,
                detection_source=detection_source,
                row_count=local_total,
            )

            if need_gray and gray_dest is not None:
                src_gray = root["raw_video/images_ds"]
                frame_limit = int(src_gray.shape[0])
                if src_frame_indices.size > 0:
                    min_idx = int(np.min(src_frame_indices))
                    max_idx = int(np.max(src_frame_indices))
                    if min_idx < 0 or max_idx >= frame_limit:
                        raise IndexError(
                            f"{spec.source_zarr}: frame_indices out of bounds for images_ds "
                            f"(min={min_idx}, max={max_idx}, frame_count={frame_limit})."
                        )
                _copy_indexed_frames(
                    src_gray,
                    src_frame_indices,
                    gray_dest,
                    offset,
                    batch_size=copy_batch_size,
                    on_copied=_advance_copy,
                )

            if need_rgb and rgb_dest is not None:
                src_rgb = root["raw_video/images_ds_rgb"]
                frame_limit = int(src_rgb.shape[0])
                if src_frame_indices.size > 0:
                    min_idx = int(np.min(src_frame_indices))
                    max_idx = int(np.max(src_frame_indices))
                    if min_idx < 0 or max_idx >= frame_limit:
                        raise IndexError(
                            f"{spec.source_zarr}: frame_indices out of bounds for images_ds_rgb "
                            f"(min={min_idx}, max={max_idx}, frame_count={frame_limit})."
                        )
                _copy_indexed_frames(
                    src_rgb,
                    src_frame_indices,
                    rgb_dest,
                    offset,
                    batch_size=copy_batch_size,
                    on_copied=_advance_copy,
                )

            merged_bbox_norm[offset : offset + local_total] = bbox
            merged_source_kind_labels[offset : offset + local_total] = source_kind_labels
            merged_reason_labels[offset : offset + local_total] = reason_labels
            merged_manual_edit_flags[offset : offset + local_total] = manual_edit_flags
            merged_source_detect_row_index[offset : offset + local_total] = source_detect_row_index
            if confidence_scores is not None and merged_confidence_scores is None:
                merged_confidence_scores = np.full((total_samples,), np.nan, dtype=np.float32)
            if merged_confidence_scores is not None and confidence_scores is not None:
                merged_confidence_scores[offset : offset + local_total] = confidence_scores
            if class_ids is not None and merged_class_ids is None:
                merged_class_ids = np.full((total_samples,), -1, dtype=np.int32)
            if merged_class_ids is not None and class_ids is not None:
                merged_class_ids[offset : offset + local_total] = class_ids

            src_dataset_idx_dest[offset : offset + local_total] = int(spec.ordinal)
            src_frame_idx_dest[offset : offset + local_total] = src_frame_indices
            src_roi_idx_dest[offset : offset + local_total] = np.arange(local_total, dtype=np.int64)
            src_refined_row_ids_dest[offset : offset + local_total] = source_refined_row_ids
            src_detect_row_index_dest[offset : offset + local_total] = source_detect_row_index

            for key, count in local_source_kind_counts.items():
                merged_source_kind_counts[key] = merged_source_kind_counts.get(key, 0) + int(count)
            total_real += int(local_total)

            offset += local_total
    finally:
        if copy_progress is not None:
            copy_progress.stop()

    write_curated_refined_detect_surfaces(
        out_root,
        zarr_path=out_zarr,
        refined_run_name=run_name,
        instance_frame_indices=np.arange(total_samples, dtype=np.int32),
        instance_bbox_norm_coords=merged_bbox_norm,
        instance_source_kind_labels=merged_source_kind_labels,
        instance_reason_labels=merged_reason_labels,
        instance_source_detect_row_index=merged_source_detect_row_index,
        instance_manual_edit_flags=merged_manual_edit_flags,
        instance_confidence_scores=merged_confidence_scores,
        instance_class_ids=merged_class_ids,
        command="fisheye.utils.export_detect_training_zarr --merge",
        source_context={
            "source_type_requested": str(manifest.source_type),
            "source_type_resolved": "refined",
            "source_count": int(len(source_specs)),
            "interpolation_policy": "forbidden_for_merged_training_export",
        },
    )

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

    downsample_formats: List[str] = []
    if need_gray:
        downsample_formats.append("gray")
    if need_rgb:
        downsample_formats.append("rgb")

    raw_attrs: Dict[str, Any] = {
        "downsample_formats": downsample_formats,
    }
    raw_metadata = _format_raw_metadata_summary(source_specs, need_gray=need_gray, need_rgb=need_rgb)
    selected_pixel_contract_name = (
        raw_metadata["gray_pixel_contract_name"]
        if train_input_format == "gray"
        else raw_metadata["rgb_pixel_contract_name"]
    )
    selected_decode_backend = (
        raw_metadata["gray_decode_backend"]
        if train_input_format == "gray"
        else raw_metadata["rgb_decode_backend"]
    )
    if selected_pixel_contract_name:
        raw_attrs["pixel_contract_name"] = selected_pixel_contract_name
    if selected_decode_backend:
        raw_attrs["decode_backend"] = selected_decode_backend
    if any(raw_metadata["pixel_contract_names_by_format"].values()):
        raw_attrs["pixel_contract_names_by_format"] = raw_metadata["pixel_contract_names_by_format"]
    if any(raw_metadata["decode_backends_by_format"].values()):
        raw_attrs["decode_backends_by_format"] = raw_metadata["decode_backends_by_format"]
    if resolution is not None:
        raw_attrs["downsampled_resolution"] = resolution
    fps_values = [float(value) for value in layout["fps_values"] if value is not None]
    if fps_values and np.isclose(max(fps_values), min(fps_values), atol=1e-6):
        raw_attrs["fps"] = float(fps_values[0])
    raw_group.attrs.update(raw_attrs)

    refined_run.attrs.update(
        {
            "detection_source_type": "refined",
            "source_type_requested": str(manifest.source_type),
            "source_type_resolved": "refined",
            "includes_interpolated": False,
            "interpolation_enabled": False,
            "interpolation_policy": "forbidden_for_merged_training_export",
            "n_real_detections": int(total_real),
            "n_interpolated_detections": 0,
            "source_count": int(len(source_specs)),
        }
    )

    _write_string_array(source_index_group, "source_dataset_id", [spec.dataset_id for spec in source_specs])
    _write_string_array(source_index_group, "source_zarr_path", [str(spec.source_zarr) for spec in source_specs])
    source_index_group.attrs.update(
        {
            "mapping_version": 1,
            "source_count": int(len(source_specs)),
        }
    )

    training_export = {
        "schema_version": "1.0.0",
        "task": str(manifest.task),
        "set_id": manifest.set_id,
        "set_name": manifest.set_name,
        "set_version": manifest.set_version,
        "source_type_requested": str(manifest.source_type),
        "source_type_resolved": "refined",
        "canonical_label_path": f"refined_detect_runs/{run_name}/instances",
        "interpolation_policy": "forbidden_for_merged_training_export",
        "input_format": "both" if (need_gray and need_rgb) else train_input_format,
        "include_rgb": bool(need_rgb),
        "created_at_utc": _utc_now(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "query_filter": manifest.query_filter,
        "invocation": invocation,
        "source_dataset_ids": [spec.dataset_id for spec in source_specs],
        "source_zarr_paths": [str(spec.source_zarr) for spec in source_specs],
        "required_pixel_contract_name": _as_text(required_pixel_contract_name),
        "pixel_contract_name": selected_pixel_contract_name,
        "pixel_contract_names_by_format": raw_metadata["pixel_contract_names_by_format"],
        "decode_backend": selected_decode_backend,
        "decode_backends_by_format": raw_metadata["decode_backends_by_format"],
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
    mark_run_complete(refined_run, parent_group=refined_parent, run_name=run_name)

    return MergeResult(
        run_name=run_name,
        total_samples=total_samples,
        total_real=total_real,
        total_interpolated=total_interpolated,
        source_kind_counts=merged_source_kind_counts,
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        source_specs=source_specs,
        downsample_formats=downsample_formats,
        training_input_format=train_input_format,
    )


def _build_merged_manifest_payload(
    *,
    manifest: TrainingManifest,
    manifest_payload: Dict[str, Any],
    merged_zarr: Path,
    merged_dataset_id: str,
    merged_dataset_name: str,
    run_name: str,
    out_manifest: Path,
    out_config: Path,
    merge_result: MergeResult,
    include_rgb: bool,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, Any]:
    def _clean_text(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _extract_identity(dataset_payload: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        dish = _clean_text(dataset_payload.get("dish_design"))
        canvas = _clean_text(dataset_payload.get("canvas_name"))
        rig = _clean_text(dataset_payload.get("rig_id"))
        provenance = dataset_payload.get("provenance")
        if isinstance(provenance, dict):
            if not dish:
                arena = provenance.get("arena")
                if isinstance(arena, dict):
                    dish = _clean_text(arena.get("dish_design"))
            if not canvas:
                rig_info = provenance.get("rig_info")
                if isinstance(rig_info, dict):
                    canvas = _clean_text(rig_info.get("canvas_name"))
            if not rig:
                rig_info = provenance.get("rig_info")
                if isinstance(rig_info, dict):
                    rig = _clean_text(rig_info.get("rig_id"))
        return dish, canvas, rig

    def _collapse(values: set[str], *, mixed_label: str) -> Optional[str]:
        if len(values) == 1:
            return next(iter(values))
        if len(values) > 1:
            return mixed_label
        return None

    payload = dict(manifest_payload)
    source_rows = payload.get("datasets")
    if not isinstance(source_rows, list):
        source_rows = []
    identity_by_path: Dict[str, Tuple[Optional[str], Optional[str], Optional[str]]] = {}
    for row in source_rows:
        if not isinstance(row, dict):
            continue
        zarr_path = _clean_text(row.get("zarr_path"))
        if not zarr_path:
            continue
        identity_by_path[zarr_path] = _extract_identity(row)

    source_datasets_payload: List[Dict[str, Any]] = []
    dish_values: set[str] = set()
    canvas_values: set[str] = set()
    rig_values: set[str] = set()
    for spec in merge_result.source_specs:
        dish, canvas, rig = identity_by_path.get(str(spec.source_zarr), (None, None, None))
        if dish:
            dish_values.add(dish)
        if canvas:
            canvas_values.add(canvas)
        if rig:
            rig_values.add(rig)
        source_datasets_payload.append(
            {
                "name": spec.dataset_name,
                "dataset_id": spec.dataset_id,
                "zarr_path": str(spec.source_zarr),
                "sample_count": int(spec.sample_count),
                "gray_pixel_contract_name": spec.gray_pixel_contract_name,
                "rgb_pixel_contract_name": spec.rgb_pixel_contract_name,
                "gray_decode_backend": spec.gray_decode_backend,
                "rgb_decode_backend": spec.rgb_decode_backend,
                "dish_design": dish,
                "canvas_name": canvas,
                "rig_id": rig,
            }
        )

    merged_dish = _collapse(dish_values, mixed_label="mixed_dishes")
    merged_canvas = _collapse(canvas_values, mixed_label="mixed_canvas")
    merged_rig = _collapse(rig_values, mixed_label="mixed_rigs")
    include_gray = merge_result.training_input_format == "gray"
    include_rgb = merge_result.training_input_format == "rgb" or bool(include_rgb)
    raw_metadata = _format_raw_metadata_summary(
        merge_result.source_specs,
        need_gray=include_gray,
        need_rgb=include_rgb,
    )
    selected_pixel_contract_name = (
        raw_metadata["gray_pixel_contract_name"]
        if merge_result.training_input_format == "gray"
        else raw_metadata["rgb_pixel_contract_name"]
    )

    if merge_result.training_input_format == "gray":
        spatial_shape = list(merge_result.source_specs[0].gray_shape or ())
    else:
        spatial_shape = list(merge_result.source_specs[0].rgb_shape or ())

    merged_dataset = {
        "name": merged_dataset_name,
        "zarr_path": str(merged_zarr),
        "dataset_id": merged_dataset_id,
        "session_uuid": None,
        "dish_design": merged_dish,
        "canvas_name": merged_canvas,
        "rig_id": merged_rig,
        "crop_run": None,
        "bbox_array_path": f"refined_detect_runs/{run_name}/instances/bbox_norm_coords",
        "detection_source_type": "refined",
        "detection_source_path": f"refined_detect_runs/{run_name}/instances",
        "source_type_requested": str(manifest.source_type),
        "source_type_resolved": "refined",
        "includes_interpolated": False,
        "interpolation_policy": "forbidden_for_merged_training_export",
        "input_format": merge_result.training_input_format,
        "pixel_contract_name": selected_pixel_contract_name,
        "images_ds_shape": spatial_shape,
        "total_bboxes": int(merge_result.total_samples),
        "invalid_bboxes": 0,
        "invalid_bbox_sample": [],
        "source_kind_counts": merge_result.source_kind_counts,
        "frame_indices_present": True,
        "detection_source_present": False,
        "provenance": None,
    }

    payload["datasets"] = [merged_dataset]
    payload["output_manifest_path"] = str(out_manifest)
    payload["output_config_path"] = str(out_config)
    payload["merged_export"] = {
        "enabled": True,
        "zarr_path": str(merged_zarr),
        "run_name": run_name,
        "downsample_formats": merge_result.downsample_formats,
        "training_input_format": merge_result.training_input_format,
        "include_rgb": bool(include_rgb),
        "pixel_contract_name": selected_pixel_contract_name,
        "pixel_contract_names_by_format": raw_metadata["pixel_contract_names_by_format"],
        "decode_backends_by_format": raw_metadata["decode_backends_by_format"],
        "canonical_label_path": f"refined_detect_runs/{run_name}/instances",
        "interpolation_policy": "forbidden_for_merged_training_export",
        "split": {
            "train_ratio": float(train_ratio),
            "val_ratio": float(val_ratio),
            "test_ratio": float(test_ratio) if test_ratio > 0.0 else None,
            "seed": int(seed),
        },
        "counts": {
            "total_samples": int(merge_result.total_samples),
            "real": int(merge_result.total_real),
            "interpolated": int(merge_result.total_interpolated),
            "train": int(merge_result.train_indices.shape[0]),
            "val": int(merge_result.val_indices.shape[0]),
            "test": int(merge_result.test_indices.shape[0]),
            "source_count": int(len(merge_result.source_specs)),
        },
        "source_datasets": source_datasets_payload,
    }
    payload["dish_design"] = merged_dish
    payload["canvas_name"] = merged_canvas
    payload["rig_name"] = merged_rig
    payload["exported_at_utc"] = _utc_now()
    return payload


def _write_merged_config(
    *,
    source_config_path: Optional[Path],
    out_config: Path,
    merged_zarr: Path,
    dataset_name: str,
    source_type: str,
    input_format: str,
    train_ratio: float,
    val_ratio: float,
    random_seed: int,
) -> List[Path]:
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
            "split": {
                "train": float(train_ratio),
                "val": float(val_ratio),
            },
        }
    }
    cfg["random_seed"] = int(random_seed)

    created_dummy_paths: List[Path] = []
    data_root = cfg.get("path")
    base_dir = Path(data_root) if data_root else out_config.parent
    for key in ("train", "val"):
        value = cfg.get(key)
        if not isinstance(value, str) or not value:
            continue
        target = Path(value)
        if not target.is_absolute():
            target = (base_dir / target).resolve()
        if target.exists():
            continue
        if target.suffix.lower() != ".txt" and "dummy" not in target.name:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("", encoding="utf-8")
        created_dummy_paths.append(target)

    out_config.parent.mkdir(parents=True, exist_ok=True)
    out_config.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return created_dummy_paths


def _write_merge_summary(
    *,
    summary_path: Path,
    out_zarr: Path,
    merge_result: MergeResult,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> None:
    include_gray = merge_result.training_input_format == "gray"
    include_rgb = "rgb" in merge_result.downsample_formats
    raw_metadata = _format_raw_metadata_summary(
        merge_result.source_specs,
        need_gray=include_gray,
        need_rgb=include_rgb,
    )
    payload = {
        "created_at_utc": _utc_now(),
        "zarr_path": str(out_zarr),
        "downsample_formats": merge_result.downsample_formats,
        "training_input_format": merge_result.training_input_format,
        "pixel_contract_names_by_format": raw_metadata["pixel_contract_names_by_format"],
        "decode_backends_by_format": raw_metadata["decode_backends_by_format"],
        "counts": {
            "total_samples": int(merge_result.total_samples),
            "real": int(merge_result.total_real),
            "interpolated": int(merge_result.total_interpolated),
            "source_kind_counts": merge_result.source_kind_counts,
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
            }
            for spec in merge_result.source_specs
        ],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve_source_config_path(manifest: TrainingManifest) -> Optional[Path]:
    if manifest.output_config_path:
        candidate = Path(manifest.output_config_path)
        if candidate.exists():
            return candidate
    if manifest.base_config_path:
        candidate = Path(manifest.base_config_path)
        if candidate.exists():
            return candidate
    return None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Training manifest JSON.")
    parser.add_argument("--out-dir", type=Path, help="Output directory for exported training Zarrs.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Write one merged training Zarr instead of one Zarr per dataset.",
    )
    parser.add_argument(
        "--out-zarr",
        type=Path,
        help="Output path for merged Zarr (used with --merge).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="0.8/0.2",
        help="Merged split ratios: train/val or train/val/test.",
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
        help="Frame-copy batch size for merged export (higher can improve throughput).",
    )
    parser.add_argument(
        "--include-rgb",
        action="store_true",
        help="In merged mode, also export raw_video/images_ds_rgb (requires RGB arrays in all sources).",
    )
    parser.add_argument(
        "--required-pixel-contract-name",
        help=(
            "Optional merged-export guard: require each copied raw-video image surface "
            "to carry this pixel_contract_name."
        ),
    )
    parser.add_argument(
        "--registry",
        type=Path,
        help="Optional registry SQLite path. In merged mode, registers merged dataset and updates training_set linkage.",
    )
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip merged-zarr post-export validation checks.",
    )
    args = parser.parse_args(argv)

    manifest_payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest = TrainingManifest.model_validate(manifest_payload)
    if not manifest.datasets:
        raise ValueError(f"Manifest has no datasets: {args.manifest}")

    set_id_source = str(manifest.set_id).strip() if manifest.set_id else args.manifest.stem
    set_id = _normalize_manifest_stem(set_id_source)
    if not set_id:
        set_id = "training_set"
    out_dir = args.out_dir or (_resolve_default_dataset_root() / set_id)
    out_manifest = out_dir / f"{set_id}.manifest.json"
    out_config = out_dir / f"{set_id}.yaml"
    invocation = build_invocation_record(
        tool="fisheye.utils.export_detect_training_zarr",
        args=args,
    )

    if args.merge:
        train_ratio, val_ratio, test_ratio = _parse_split_spec(args.split)
        merged_zarr = args.out_zarr or (out_dir / "zarr" / f"{set_id}_merged.zarr")
        merged_dataset_name = _ensure_suffix(str(manifest.set_name or "merged"), "_merged")
        merged_dataset_id_base = str(manifest.set_id).strip() if manifest.set_id else merged_zarr.stem
        merged_dataset_id_base = _normalize_manifest_stem(merged_dataset_id_base)
        merged_dataset_id = _ensure_suffix(merged_dataset_id_base, "_merged")
        if train_ratio + val_ratio <= 0.0:
            raise ValueError(
                f"Invalid --split '{args.split}': train+val must be > 0 for training config generation."
            )
        config_train_ratio = train_ratio / (train_ratio + val_ratio)
        config_val_ratio = val_ratio / (train_ratio + val_ratio)

        print(f"Manifest: {args.manifest}")
        print(f"Merged output: {merged_zarr}")
        print(f"Datasets: {len(manifest.datasets)}")
        print(f"Split: train={train_ratio:.3f} val={val_ratio:.3f} test={test_ratio:.3f} seed={args.seed}")
        print(f"Copy batch size: {int(args.copy_batch_size)}")
        print(f"Training input format: {manifest.input_format}")
        print(f"Also export RGB array: {bool(args.include_rgb)}")
        for dataset in manifest.datasets:
            print(f"- {dataset.name}: {dataset.zarr_path}")

        if args.dry_run:
            _discover_merge_sources(
                manifest,
                need_gray=str(manifest.input_format).strip().lower() == "gray",
                need_rgb=str(manifest.input_format).strip().lower() == "rgb" or bool(args.include_rgb),
                required_pixel_contract_name=args.required_pixel_contract_name,
            )
            print("Dry run: validation passed, no files written.")
            return 0

        merge_result = _export_merged(
            manifest=manifest,
            manifest_path=args.manifest,
            out_zarr=merged_zarr,
            merged_dataset_id=merged_dataset_id,
            overwrite=bool(args.overwrite),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=int(args.seed),
            include_rgb=bool(args.include_rgb),
            copy_batch_size=int(args.copy_batch_size),
            invocation=invocation,
            required_pixel_contract_name=args.required_pixel_contract_name,
        )

        if not args.skip_validate:
            validation_summary = validate_merged_training_zarr(
                merged_zarr,
                expected_input_format=merge_result.training_input_format,
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
                    set_id=manifest.set_id,
                    set_name=manifest.set_name,
                    source_dataset_ids=[spec.dataset_id for spec in merge_result.source_specs],
                    query_filter=manifest.query_filter,
                    invocation=manifest.invocation,
                )
                print(f"Registered merged dataset in registry: {merged_dataset_id}")
                if linked and manifest.set_id:
                    print(f"Updated training_set dataset_ids linkage: {manifest.set_id}")
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
            manifest=manifest,
            manifest_payload=manifest_payload,
            merged_zarr=merged_zarr,
            merged_dataset_id=merged_dataset_id,
            merged_dataset_name=merged_dataset_name,
            run_name=merge_result.run_name,
            out_manifest=out_manifest,
            out_config=out_config,
            merge_result=merge_result,
            include_rgb=bool(args.include_rgb),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=int(args.seed),
        )
        out_manifest.parent.mkdir(parents=True, exist_ok=True)
        out_manifest.write_text(json.dumps(merged_manifest_payload, indent=2), encoding="utf-8")

        source_config_path = _resolve_source_config_path(manifest)
        created_dummy_paths = _write_merged_config(
            source_config_path=source_config_path,
            out_config=out_config,
            merged_zarr=merged_zarr,
            dataset_name=merged_dataset_name,
            source_type="refined",
            input_format=merge_result.training_input_format,
            train_ratio=config_train_ratio,
            val_ratio=config_val_ratio,
            random_seed=int(args.seed),
        )
        for path in created_dummy_paths:
            print(f"Created dummy dataset file: {path}")

        export_log = out_dir / "export_invocation.json"
        export_log.write_text(json.dumps(invocation, indent=2), encoding="utf-8")

        print(f"\nWrote merged zarr: {merged_zarr}")
        print(f"Wrote summary: {summary_path}")
        print(f"Wrote manifest: {out_manifest}")
        print(f"Wrote config: {out_config}")
        print(f"Next: python -m fisheye.training.train_detection {out_config} --manifest {out_manifest}")
        return 0

    run_name = f"training_export_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    plans = _build_plan(manifest, out_dir)

    print(f"Manifest: {args.manifest}")
    print(f"Export root: {out_dir}")
    print(f"Datasets: {len(plans)}")
    for plan in plans:
        print(f"- {plan.dataset_name}: {plan.source_zarr} -> {plan.dest_zarr}")

    if args.dry_run:
        return 0

    for plan in plans:
        print(f"\nExporting {plan.dataset_name}...")
        _export_dataset(plan, overwrite=args.overwrite, run_name=run_name, manifest_path=args.manifest)

    _update_manifest_paths(manifest_payload, plans, out_manifest, out_config, out_dir)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")

    source_config_path = _resolve_source_config_path(manifest)
    if source_config_path and source_config_path.exists():
        _write_config(source_config_path, out_config, plans)
    else:
        print("Warning: source config not found; wrote manifest only.")

    export_log = out_dir / "export_invocation.json"
    export_log.write_text(json.dumps(invocation, indent=2), encoding="utf-8")

    print(f"\nWrote manifest: {out_manifest}")
    if out_config.exists():
        print(f"Wrote config: {out_config}")
        print(f"Next: python -m fisheye.training.train_detection {out_config} --manifest {out_manifest}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
