#!/usr/bin/env python3
"""Project legacy eye-mask runs into subject_mask_runs.

Default mode is dry-run. Use --apply to write a new subject_mask_runs/<run>
projection for each selected archive.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import zarr

from fisheye.shared.provenance_attrs import (
    build_source_keypoints_attrs,
    resolve_source_keypoints_run,
)
from fisheye.shared.subject_mask_chunks import subject_mask_storage_chunks
from fisheye.shared.subject_mask_component_provenance import write_subject_mask_component_provenance
from fisheye.shared.type_conversions import normalize_attr
from fisheye.shared.zarr_helpers import resolve_zarr_run
from fisheye.utils.zarr_io import open_zarr_root


TARGET_SCHEMAS: dict[str, tuple[str, ...]] = {
    "subject_v1_union": ("subject_body", "eyes_union", "swim_bladder"),
    "subject_v1_lr": ("subject_body", "eye_left", "eye_right", "swim_bladder"),
}
BACKFILL_SUBJECT_MASK_METHOD = "fisheye.utils.backfill_subject_mask_runs"
LEGACY_EYE_MASK_PROJECTION_SEMANTICS = "legacy_eye_mask_projection"
RUNTIME_EYE_MASK_PROJECTION_SEMANTICS = "eye_mask_runtime_projection"
AVAILABLE_EYE_ONLY_BY_SCHEMA: dict[str, np.ndarray] = {
    "subject_v1_union": np.asarray([False, True, False], dtype=bool),
    "subject_v1_lr": np.asarray([False, True, True, False], dtype=bool),
}
SOURCE_STAGE_CHOICES = ("auto", "prefer_refined", "eye_masks_runs", "refined_eye_masks_runs")
LABEL_SCHEMA_CHOICES = ("auto", "subject_v1_union", "subject_v1_lr")
SOURCE_KEYPOINT_GROUP_ATTR = "source_keypoint_group"


@dataclass(frozen=True)
class ResolvedSource:
    stage_group: str
    run_name: str
    run_group: zarr.Group


@dataclass(frozen=True)
class ProbabilitySource:
    array: Optional[zarr.Array]
    array_name: Optional[str]
    source_path: Optional[str]
    source_group: Optional[zarr.Group]


@dataclass(frozen=True)
class EyeLayout:
    kind: str
    left_idx: Optional[int] = None
    right_idx: Optional[int] = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _iter_zarr(roots: list[Path], recursive: bool) -> Iterable[Path]:
    seen: set[str] = set()
    for root in roots:
        root = root.expanduser()
        candidates: list[Path] = []
        if root.suffix == ".zarr" and (root.is_dir() or root.is_file()):
            candidates = [root]
        elif root.exists():
            if recursive:
                candidates = sorted(root.rglob("*.zarr"))
            else:
                candidates = sorted(root.glob("*.zarr")) + sorted(root.glob("*/zarr/*.zarr"))
        for candidate in candidates:
            try:
                key = str(candidate.resolve())
            except OSError:
                key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            yield candidate


def _infer_zarr_use(root: zarr.Group, zarr_path: Path) -> Optional[str]:
    for key in ("zarr_use", "zarr_purpose"):
        value = normalize_attr(root.attrs.get(key))
        if value in {"analysis", "training"}:
            return value
    name = zarr_path.name.lower()
    if name.endswith("_analysis.zarr"):
        return "analysis"
    if name.endswith("_training.zarr"):
        return "training"
    return None


def _resolve_source(
    root: zarr.Group,
    *,
    source_stage: str,
    source_run: Optional[str],
) -> ResolvedSource:
    errors: list[str] = []
    if source_stage == "auto":
        candidates = ("eye_masks_runs", "refined_eye_masks_runs")
    elif source_stage == "prefer_refined":
        candidates = ("refined_eye_masks_runs", "eye_masks_runs")
    else:
        candidates = (source_stage,)
    for stage_group in candidates:
        try:
            run_group, run_name = resolve_zarr_run(
                root,
                stage_group,
                source_run,
                fallback_to_latest=True,
                run_label="Subject-mask source run",
            )
            return ResolvedSource(stage_group=stage_group, run_name=run_name, run_group=run_group)
        except Exception as exc:
            errors.append(f"{stage_group}: {exc}")
            continue
    raise ValueError("; ".join(errors) if errors else "No usable eye-mask source run found.")


def _default_run_name(source: ResolvedSource) -> str:
    return f"subject_masks_from_{source.run_name}"


def _default_runtime_run_name(source_run: str) -> str:
    if source_run.startswith("eye_masks_"):
        return f"subject_masks_{source_run[len('eye_masks_'):]}"
    return f"subject_masks_from_{source_run}"


def _normalize_eye_label(value: Any) -> Optional[str]:
    text = normalize_attr(value)
    if text is None:
        return None
    lowered = text.strip().lower()
    if lowered in {"eye_left", "left", "left_eye"}:
        return "eye_left"
    if lowered in {"eye_right", "right", "right_eye"}:
        return "eye_right"
    if lowered in {"eye_union", "eyes_union", "union"}:
        return "eyes_union"
    return lowered


def _eye_labels_for_group(group: Optional[zarr.Group]) -> list[str]:
    if group is None:
        return []
    value = group.attrs.get("eye_labels")
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return []


def _resolve_eye_layout(group: Optional[zarr.Group], channel_count: int) -> EyeLayout:
    if channel_count == 1:
        return EyeLayout(kind="union")
    if channel_count != 2:
        raise ValueError(f"Expected 1 or 2 source channels, got {channel_count}.")

    normalized = [_normalize_eye_label(label) for label in _eye_labels_for_group(group)]
    left_idx: Optional[int] = None
    right_idx: Optional[int] = None
    for idx, label in enumerate(normalized[:channel_count]):
        if label == "eye_left":
            left_idx = idx
        elif label == "eye_right":
            right_idx = idx
    if left_idx is not None and right_idx is not None:
        return EyeLayout(kind="lr", left_idx=left_idx, right_idx=right_idx)
    return EyeLayout(kind="pair")


def _resolve_label_schema(
    requested_schema: str,
    *,
    source_layout: EyeLayout,
    source: ResolvedSource,
) -> str:
    if requested_schema == "auto":
        return "subject_v1_lr" if source_layout.kind == "lr" else "subject_v1_union"
    if requested_schema == "subject_v1_lr" and source_layout.kind != "lr":
        raise ValueError(
            f"{source.stage_group}/{source.run_name} does not provide anatomical left/right eye labels for "
            "subject_v1_lr projection."
        )
    if requested_schema not in TARGET_SCHEMAS:
        raise ValueError(f"Unsupported label_schema={requested_schema!r}.")
    return requested_schema


def _projection_mode_for_schema(label_schema: str, source_layout: EyeLayout) -> str:
    if label_schema == "subject_v1_lr":
        return "eye_lr_from_lr"
    if source_layout.kind == "union":
        return "eyes_union_from_union"
    if source_layout.kind == "lr":
        return "eyes_union_from_lr"
    return "eyes_union_from_pair"


def _component_source_channels_for_projection(
    *,
    label_schema: str,
    source_layout: EyeLayout,
) -> dict[str, list[str]]:
    if label_schema == "subject_v1_lr":
        if source_layout.kind != "lr":
            raise ValueError("subject_v1_lr projection requires anatomical left/right eye labels.")
        return {
            "eye_left": ["eye_left"],
            "eye_right": ["eye_right"],
        }
    if source_layout.kind == "union":
        return {"eyes_union": ["eyes_union"]}
    if source_layout.kind == "lr":
        return {"eyes_union": ["eye_left", "eye_right"]}
    return {"eyes_union": ["channel_0", "channel_1"]}


def _probability_source_usable_for_schema(
    label_schema: str,
    *,
    probability_source: ProbabilitySource,
) -> bool:
    if probability_source.array is None:
        return False
    channel_count = int(probability_source.array.shape[1])
    if label_schema == "subject_v1_union":
        return channel_count in {1, 2}
    return _resolve_eye_layout(probability_source.source_group, channel_count).kind == "lr"


def _normalize_encoding(value: Any) -> Optional[str]:
    text = normalize_attr(value)
    if text in {"unit_float", "linear_uint8_0_255"}:
        return text
    return None


def _default_encoding_for_dtype(dtype: np.dtype) -> str:
    if np.issubdtype(dtype, np.integer):
        return "linear_uint8_0_255"
    return "unit_float"


def _select_probability_source(root: zarr.Group, source: ResolvedSource) -> ProbabilitySource:
    direct_names = ("mask_probs_roi", "mask_probs_roi_refined")
    if source.stage_group == "refined_eye_masks_runs":
        direct_names = ("mask_probs_roi_refined", "mask_probs_roi")

    for name in direct_names:
        if name in source.run_group:
            return ProbabilitySource(
                array=source.run_group[name],
                array_name=name,
                source_path=f"{source.stage_group}/{source.run_name}/{name}",
                source_group=source.run_group,
            )

    if source.stage_group == "refined_eye_masks_runs":
        raw_run_name = normalize_attr(source.run_group.attrs.get("source_eye_masks_run"))
        raw_parent = root.get("eye_masks_runs")
        if raw_run_name and raw_parent is not None and raw_run_name in raw_parent:
            raw_group = raw_parent[raw_run_name]
            if "mask_probs_roi" in raw_group:
                return ProbabilitySource(
                    array=raw_group["mask_probs_roi"],
                    array_name="mask_probs_roi",
                    source_path=f"eye_masks_runs/{raw_run_name}/mask_probs_roi",
                    source_group=raw_group,
                )

    return ProbabilitySource(array=None, array_name=None, source_path=None, source_group=None)


def _resolve_lineage_array(
    root: zarr.Group,
    source: ResolvedSource,
    *,
    source_crop_run: Optional[str],
    array_name: str,
) -> Optional[zarr.Array]:
    if array_name in source.run_group:
        return source.run_group[array_name]
    if not source_crop_run:
        return None
    crop_parent = root.get("crop_runs")
    if crop_parent is None or source_crop_run not in crop_parent:
        return None
    crop_group = crop_parent[source_crop_run]
    if array_name in crop_group:
        return crop_group[array_name]
    return None


def _copy_array(group: zarr.Group, name: str, source: zarr.Array) -> None:
    data = source[:]
    kwargs: dict[str, object] = {
        "overwrite": True,
    }
    chunks = getattr(source, "chunks", None)
    if chunks is not None:
        kwargs["chunks"] = chunks
    group.create_array(name, data=data, **kwargs)


def _decode_probabilities(batch: np.ndarray, *, encoding: str) -> np.ndarray:
    source_dtype = batch.dtype
    decoded = batch.astype(np.float32, copy=False)
    if np.issubdtype(source_dtype, np.integer) and encoding == "linear_uint8_0_255":
        max_value = float(np.iinfo(source_dtype).max)
        if max_value > 1.0:
            decoded /= max_value
    decoded = np.nan_to_num(decoded, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(decoded, 0.0, 1.0, out=decoded)


def _project_union(batch: np.ndarray) -> np.ndarray:
    if batch.ndim != 4:
        raise ValueError(f"Expected 4D mask/probability batch, got shape {batch.shape}.")
    channel_count = int(batch.shape[1])
    if channel_count == 1:
        return batch
    if channel_count == 2:
        return np.maximum(batch[:, 0:1, :, :], batch[:, 1:2, :, :])
    raise ValueError(f"Expected 1 or 2 source mask channels, got {channel_count}.")


def _project_lr(batch: np.ndarray, layout: EyeLayout) -> np.ndarray:
    if batch.ndim != 4:
        raise ValueError(f"Expected 4D mask/probability batch, got shape {batch.shape}.")
    if batch.shape[1] != 2:
        raise ValueError(f"Expected 2-channel left/right source batch, got shape {batch.shape}.")
    if layout.kind != "lr" or layout.left_idx is None or layout.right_idx is None:
        raise ValueError("Source batch does not provide anatomical left/right eye labels.")
    return batch[:, [layout.left_idx, layout.right_idx], :, :]


def _target_eye_slice(label_schema: str) -> slice:
    if label_schema == "subject_v1_union":
        return slice(1, 2)
    if label_schema == "subject_v1_lr":
        return slice(1, 3)
    raise ValueError(f"Unsupported label schema {label_schema!r}.")


def _zero_template(shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    return np.zeros(shape, dtype=dtype)


def _target_prob_dtype(prob_source: ProbabilitySource) -> np.dtype:
    if prob_source.array is None:
        return np.dtype(np.float32)
    return np.dtype(prob_source.array.dtype)


def _target_mask_chunks(*, total_rows: int, height: int, width: int) -> tuple[int, int, int, int]:
    return subject_mask_storage_chunks(total_rows, height, width)


def backfill_subject_mask_run(
    zarr_path: Path | str,
    *,
    source_stage: str = "auto",
    source_run: Optional[str] = None,
    label_schema: str = "auto",
    run_name: Optional[str] = None,
    batch_size: int = 256,
    overwrite: bool = False,
    apply: bool = False,
    method: Optional[str] = None,
    run_semantics: str = LEGACY_EYE_MASK_PROJECTION_SEMANTICS,
) -> dict[str, Any]:
    if source_stage not in SOURCE_STAGE_CHOICES:
        raise ValueError(f"Unsupported source_stage={source_stage!r}.")
    if label_schema not in LABEL_SCHEMA_CHOICES:
        raise ValueError(f"Unsupported label_schema={label_schema!r}.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    resolved_method = normalize_attr(method) or BACKFILL_SUBJECT_MASK_METHOD
    resolved_run_semantics = normalize_attr(run_semantics)
    if not resolved_run_semantics:
        raise ValueError("run_semantics must be a non-empty string.")

    zarr_path = Path(zarr_path)
    root = open_zarr_root(zarr_path, mode="r+" if apply else "r")
    source = _resolve_source(root, source_stage=source_stage, source_run=source_run)
    prob_source = _select_probability_source(root, source)

    source_masks = source.run_group.get("masks_roi")
    if source_masks is None and prob_source.array is None:
        raise ValueError(
            f"{source.stage_group}/{source.run_name} provides neither masks_roi nor probability masks."
        )

    reference = source_masks if source_masks is not None else prob_source.array
    assert reference is not None
    if len(reference.shape) != 4:
        raise ValueError(
            f"{source.stage_group}/{source.run_name} expected 4D masks/probabilities, got {reference.shape}."
        )

    total_rows = int(reference.shape[0])
    source_channel_count = int(reference.shape[1])
    if source_channel_count not in {1, 2}:
        raise ValueError(
            f"{source.stage_group}/{source.run_name} expected 1 or 2 source channels, got {source_channel_count}."
        )
    source_layout = _resolve_eye_layout(source.run_group, source_channel_count)
    target_label_schema = _resolve_label_schema(label_schema, source_layout=source_layout, source=source)
    target_labels = TARGET_SCHEMAS[target_label_schema]
    available_channels = AVAILABLE_EYE_ONLY_BY_SCHEMA[target_label_schema]
    target_eye_slice = _target_eye_slice(target_label_schema)
    height = int(reference.shape[2])
    width = int(reference.shape[3])

    target_run = run_name or _default_run_name(source)
    parent = root.get("subject_mask_runs")
    existing = parent is not None and target_run in parent
    if existing and not overwrite:
        return {
            "zarr_path": str(zarr_path),
            "status": "exists",
            "target_run": target_run,
            "source_stage": source.stage_group,
            "source_run": source.run_name,
        }

    source_crop_run = normalize_attr(source.run_group.attrs.get("source_crop_run"))
    source_keypoints_run = normalize_attr(resolve_source_keypoints_run(source.run_group.attrs))
    source_keypoint_group = normalize_attr(source.run_group.attrs.get(SOURCE_KEYPOINT_GROUP_ATTR))
    use_probability_source = _probability_source_usable_for_schema(target_label_schema, probability_source=prob_source)
    if use_probability_source:
        probability_dtype = _target_prob_dtype(prob_source)
        probability_encoding = _normalize_encoding(
            prob_source.source_group.attrs.get("probabilities_encoding") if prob_source.source_group is not None else None
        )
        if probability_encoding is None:
            probability_encoding = _default_encoding_for_dtype(probability_dtype)
        prob_source_path = prob_source.source_path or f"{source.stage_group}/{source.run_name}/masks_roi"
    else:
        probability_dtype = np.dtype(np.float32)
        probability_encoding = "unit_float"
        prob_source_path = f"{source.stage_group}/{source.run_name}/masks_roi"

    projection_mode = _projection_mode_for_schema(target_label_schema, source_layout)
    summary = {
        "zarr_path": str(zarr_path),
        "status": "would_update" if not apply else "updated",
        "target_run": target_run,
        "source_stage": source.stage_group,
        "source_run": source.run_name,
        "source_rows": total_rows,
        "label_schema_id": target_label_schema,
        "projection_mode": projection_mode,
        "probability_source": prob_source_path,
        "method": resolved_method,
        "run_semantics": resolved_run_semantics,
    }
    if not apply:
        return summary

    start = time.perf_counter()
    subject_parent = root.require_group("subject_mask_runs")
    if existing and overwrite:
        del subject_parent[target_run]
    run_group = subject_parent.create_group(target_run)
    subject_parent.attrs["latest"] = target_run

    run_group.attrs.update(
        {
            "source_crop_run": source_crop_run,
            "label_schema_id": target_label_schema,
            "mask_labels": list(target_labels),
            "output_semantics": "multilabel",
            "overlap_policy": "independent_sigmoid",
            "method": resolved_method,
            "run_semantics": resolved_run_semantics,
            "source_mask_stage": source.stage_group,
            "projection_mode": projection_mode,
            "source_probability_path": prob_source_path,
            "probabilities_dtype": str(probability_dtype),
            "probabilities_encoding": probability_encoding,
            "created_at_utc": _utc_now(),
        }
    )
    if source.stage_group == "eye_masks_runs":
        run_group.attrs["source_eye_masks_run"] = source.run_name
    elif source.stage_group == "refined_eye_masks_runs":
        run_group.attrs["source_refined_eye_masks_run"] = source.run_name
        raw_eye_run = normalize_attr(source.run_group.attrs.get("source_eye_masks_run"))
        if raw_eye_run:
            run_group.attrs["source_eye_masks_run"] = raw_eye_run
    if source_keypoints_run:
        run_group.attrs.update(build_source_keypoints_attrs(source_keypoints_run))
    if source_keypoint_group:
        run_group.attrs[SOURCE_KEYPOINT_GROUP_ATTR] = source_keypoint_group
    source_method = str(source.run_group.attrs.get("method") or "unknown")
    source_label_schema_id = normalize_attr(source.run_group.attrs.get("label_schema_id"))
    source_created_at_utc = normalize_attr(source.run_group.attrs.get("created_at_utc"))
    component_source_channels = _component_source_channels_for_projection(
        label_schema=target_label_schema,
        source_layout=source_layout,
    )
    for component_name, source_channels in component_source_channels.items():
        write_subject_mask_component_provenance(
            run_group,
            component_name=component_name,
            source_stage=source.stage_group,
            source_run=source.run_name,
            source_method=source_method,
            source_channels=source_channels,
            source_label_schema_id=source_label_schema_id,
            projection_mode=projection_mode,
            source_created_at_utc=source_created_at_utc,
        )

    masks_chunks = _target_mask_chunks(total_rows=total_rows, height=height, width=width)
    probs_chunks = _target_mask_chunks(total_rows=total_rows, height=height, width=width)

    masks_out = run_group.create_array(
        "masks_roi",
        shape=(total_rows, len(target_labels), height, width),
        dtype=np.uint8,
        chunks=masks_chunks,
        fill_value=0,
        overwrite=True,
    )
    probs_out = run_group.create_array(
        "mask_probs_roi",
        shape=(total_rows, len(target_labels), height, width),
        dtype=probability_dtype,
        chunks=probs_chunks,
        fill_value=probability_dtype.type(0),
        overwrite=True,
    )
    run_group.create_array(
        "available_channels",
        data=available_channels,
        overwrite=True,
    )
    metrics_group = run_group.require_group("metrics")
    prob_max_out = metrics_group.create_array(
        "prob_max",
        shape=(total_rows, len(target_labels)),
        dtype=np.float32,
        chunks=(min(batch_size, total_rows), 1),
        fill_value=0.0,
        overwrite=True,
    )
    mask_present_out = metrics_group.create_array(
        "mask_present",
        shape=(total_rows, len(target_labels)),
        dtype=bool,
        chunks=(min(batch_size, total_rows), 1),
        fill_value=False,
        overwrite=True,
    )

    for name in ("frame_indices", "frame_counts", "detection_indices"):
        source_array = _resolve_lineage_array(root, source, source_crop_run=source_crop_run, array_name=name)
        if source_array is not None:
            _copy_array(run_group, name, source_array)

    detection_source_array = _resolve_lineage_array(root, source, source_crop_run=source_crop_run, array_name="detection_source")
    if detection_source_array is None:
        raise ValueError(
            f"Could not resolve detection_source for {source.stage_group}/{source.run_name} or source crop run."
        )
    _copy_array(run_group, "detection_source", detection_source_array)

    for start_idx in range(0, total_rows, batch_size):
        end_idx = min(total_rows, start_idx + batch_size)
        row_slice = slice(start_idx, end_idx)
        if source_masks is not None:
            source_mask_batch = np.asarray(source_masks[row_slice], dtype=np.uint8)
        else:
            assert prob_source.array is not None
            prob_batch = np.asarray(prob_source.array[row_slice])
            source_mask_batch = (_decode_probabilities(prob_batch, encoding=probability_encoding) >= 0.5).astype(
                np.uint8,
                copy=False,
            )

        if target_label_schema == "subject_v1_union":
            projected_binary = _project_union(source_mask_batch)
        else:
            projected_binary = _project_lr(source_mask_batch, source_layout)
        masks_out[row_slice, target_eye_slice, :, :] = projected_binary

        if use_probability_source and prob_source.array is not None:
            prob_batch_raw = np.asarray(prob_source.array[row_slice])
            if target_label_schema == "subject_v1_union":
                projected_prob_raw = _project_union(prob_batch_raw)
                projected_prob_semantic = _project_union(_decode_probabilities(prob_batch_raw, encoding=probability_encoding))
            else:
                projected_prob_raw = _project_lr(prob_batch_raw, _resolve_eye_layout(prob_source.source_group, prob_batch_raw.shape[1]))
                projected_prob_semantic = _project_lr(
                    _decode_probabilities(prob_batch_raw, encoding=probability_encoding),
                    _resolve_eye_layout(prob_source.source_group, prob_batch_raw.shape[1]),
                )
            probs_out[row_slice, target_eye_slice, :, :] = projected_prob_raw.astype(probability_dtype, copy=False)
        else:
            projected_prob_semantic = projected_binary.astype(np.float32, copy=False)
            probs_out[row_slice, target_eye_slice, :, :] = projected_prob_semantic.astype(probability_dtype, copy=False)

        prob_max_out[row_slice, target_eye_slice] = projected_prob_semantic.astype(np.float32, copy=False).max(
            axis=(2, 3),
            initial=0.0,
        )
        mask_present_out[row_slice, target_eye_slice] = projected_binary.any(axis=(2, 3))

    duration = float(time.perf_counter() - start)
    summary_statistics = {
        "rows": int(total_rows),
        "available_channels": [bool(v) for v in available_channels.tolist()],
        "label_schema_id": target_label_schema,
        "projection_mode": projection_mode,
        "source_stage": source.stage_group,
        "source_run": source.run_name,
        "probability_source": prob_source_path,
    }
    run_group.attrs["duration_seconds"] = duration
    run_group.attrs["summary_statistics"] = summary_statistics
    summary["duration_seconds"] = duration
    return summary


def project_eye_mask_run_to_subject_mask_run(
    zarr_path: Path | str,
    *,
    source_run: str,
    run_name: Optional[str] = None,
    label_schema: str = "subject_v1_union",
    batch_size: int = 256,
    overwrite: bool = True,
) -> dict[str, Any]:
    if not source_run:
        raise ValueError("source_run is required.")
    if label_schema != "subject_v1_union":
        raise ValueError("Runtime eye-to-subject projection currently requires label_schema='subject_v1_union'.")

    source_root = open_zarr_root(Path(zarr_path), mode="r")
    source = _resolve_source(source_root, source_stage="eye_masks_runs", source_run=source_run)
    source_method = normalize_attr(source.run_group.attrs.get("method")) or BACKFILL_SUBJECT_MASK_METHOD
    target_run_name = run_name or _default_runtime_run_name(source.run_name)
    return backfill_subject_mask_run(
        zarr_path,
        source_stage="eye_masks_runs",
        source_run=source.run_name,
        label_schema=label_schema,
        run_name=target_run_name,
        batch_size=batch_size,
        overwrite=overwrite,
        apply=True,
        method=source_method,
        run_semantics=RUNTIME_EYE_MASK_PROJECTION_SEMANTICS,
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Zarr paths or roots containing .zarr archives.")
    parser.add_argument("--recursive", action="store_true", help="Recursively search roots for .zarr archives.")
    parser.add_argument(
        "--zarr-use",
        choices=["analysis", "training", "any"],
        default="any",
        help="Archive scope filter (default: any).",
    )
    parser.add_argument(
        "--source-stage",
        choices=list(SOURCE_STAGE_CHOICES),
        default="auto",
        help=(
            "Legacy source stage to project "
            "(default: auto prefers eye_masks_runs; prefer_refined prefers "
            "refined_eye_masks_runs and falls back to eye_masks_runs)."
        ),
    )
    parser.add_argument("--source-run", type=str, default=None, help="Explicit source run name (default: latest).")
    parser.add_argument(
        "--label-schema",
        choices=list(LABEL_SCHEMA_CHOICES),
        default="auto",
        help="Target subject-mask label schema (default: auto preserves LR when the source is anatomical).",
    )
    parser.add_argument("--run-name", type=str, default=None, help="Explicit subject-mask run name to write.")
    parser.add_argument("--batch-size", type=int, default=256, help="Rows per copy batch (default: 256).")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing target run when --apply is set.")
    parser.add_argument("--apply", action="store_true", help="Write the projected subject-mask run.")
    args = parser.parse_args(argv)

    results: list[dict[str, Any]] = []
    exit_code = 0
    for zarr_path in _iter_zarr(list(args.paths), args.recursive):
        try:
            root = open_zarr_root(zarr_path, mode="r")
        except Exception as exc:
            print(f"ERROR {zarr_path}: failed to open archive ({exc})")
            exit_code = 1
            continue
        inferred_use = _infer_zarr_use(root, zarr_path)
        if args.zarr_use != "any" and inferred_use is not None and inferred_use != args.zarr_use:
            continue
        try:
            result = backfill_subject_mask_run(
                zarr_path,
                source_stage=str(args.source_stage),
                source_run=args.source_run,
                label_schema=str(args.label_schema),
                run_name=args.run_name,
                batch_size=int(args.batch_size),
                overwrite=bool(args.overwrite),
                apply=bool(args.apply),
            )
            results.append(result)
            print(
                f"{result['status'].upper()} {zarr_path} "
                f"source={result.get('source_stage')}/{result.get('source_run')} "
                f"target={result.get('target_run')}"
            )
        except Exception as exc:
            print(f"ERROR {zarr_path}: {exc}")
            exit_code = 1

    if not results:
        print("No matching zarr archives found.")
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
