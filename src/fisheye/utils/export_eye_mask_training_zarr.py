#!/usr/bin/env python3
"""Export and validate merged eye-mask-training Zarr artifacts.

This module provides:
- ``export_merged_eye_mask_training_zarr``: scaffold exporter for a single source Zarr.
- ``validate_merged_eye_mask_training_zarr``: contract validator for merged artifacts.
"""

from __future__ import annotations

import argparse
from hashlib import sha1
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import zarr
from zarr.core.dtype import VariableLengthUTF8

from fisheye.registry.db import Registry, resolve_dataset_id
from fisheye.shared.detect_reason_codec import (
    REASON_BYTES_ENCODING,
    read_reason_labels,
    write_reason_columns,
)

EYE_ROW_GATE_POLICIES = ("all_rows", "usable_only", "usable_plus_explicit_negatives")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_text(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="ignore")
    else:
        text = str(value)
    text = text.strip()
    return text or None


def _normalize_input_format(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"gray", "grey", "grayscale"}:
        return "gray"
    if text in {"rgb", "color", "colour"}:
        return "rgb"
    return None


def _normalize_label_mode(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"lr", "left_right", "left-right"}:
        return "lr"
    if text in {"union", "merged"}:
        return "union"
    return None


def _normalize_row_gate_policy(value: str) -> str:
    policy = str(value).strip().lower()
    if policy not in EYE_ROW_GATE_POLICIES:
        raise ValueError(
            f"Unsupported row_gate_policy '{value}'. "
            f"Expected one of: {', '.join(EYE_ROW_GATE_POLICIES)}."
        )
    return policy


def _normalize_explicit_negative_ratio(value: float) -> float:
    ratio = float(value)
    if not math.isfinite(ratio) or ratio < 0.0:
        raise ValueError(
            f"Invalid explicit_negative_ratio '{value}'. "
            "Expected a finite value >= 0.0."
        )
    return ratio


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


def _row_gate_seed(*, split_seed: int, source_dataset_id: str, source_path: Path) -> int:
    digest = sha1(f"{int(split_seed)}::{source_dataset_id}::{source_path}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _resolve_row_gate_selection(
    *,
    source_path: Path,
    source_dataset_id: str,
    ellipse_success: zarr.Array,
    reason_values: np.ndarray,
    row_gate_policy: str,
    explicit_negative_ratio: float,
    split_seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    policy = _normalize_row_gate_policy(row_gate_policy)
    ratio = _normalize_explicit_negative_ratio(explicit_negative_ratio)

    ellipse_success_arr = np.asarray(ellipse_success[:], dtype=np.bool_)
    if ellipse_success_arr.ndim == 1:
        pair_success = ellipse_success_arr.astype(np.bool_, copy=False)
    elif ellipse_success_arr.ndim == 2:
        pair_success = np.all(ellipse_success_arr, axis=1)
    else:
        raise ValueError(
            f"{source_path.name}: ellipse_success must be 1D or 2D for row gating, got {ellipse_success_arr.shape}."
        )
    total_rows = int(pair_success.shape[0])

    reason_arr = np.asarray(reason_values, dtype=object)
    if int(reason_arr.shape[0]) != total_rows:
        raise ValueError(
            f"{source_path.name}: reason label length mismatch for row gating "
            f"({reason_arr.shape[0]} != {total_rows})."
        )
    explicit_negative_mask = np.asarray(
        [_reason_has_tag(value, "fish_present_no_keypoints") for value in reason_arr],
        dtype=np.bool_,
    )
    explicit_negative_mask &= ~pair_success

    positive_indices = np.where(pair_success)[0].astype(np.int64, copy=False)
    explicit_negative_indices = np.where(explicit_negative_mask)[0].astype(np.int64, copy=False)

    selected_indices: np.ndarray
    selected_explicit_negative = 0
    if policy == "all_rows":
        selected_indices = np.arange(total_rows, dtype=np.int64)
        selected_explicit_negative = int(explicit_negative_indices.shape[0])
    elif policy == "usable_only":
        selected_indices = positive_indices
    else:
        max_negatives = int(math.floor(float(positive_indices.shape[0]) * ratio))
        if max_negatives <= 0 or explicit_negative_indices.size == 0:
            sampled_negatives = np.empty((0,), dtype=np.int64)
        elif explicit_negative_indices.size <= max_negatives:
            sampled_negatives = explicit_negative_indices
        else:
            rng = np.random.default_rng(
                _row_gate_seed(
                    split_seed=int(split_seed),
                    source_dataset_id=source_dataset_id,
                    source_path=source_path,
                )
            )
            sampled_negatives = np.sort(
                rng.choice(explicit_negative_indices, size=max_negatives, replace=False).astype(
                    np.int64,
                    copy=False,
                )
            )
        selected_explicit_negative = int(sampled_negatives.shape[0])
        if sampled_negatives.size > 0:
            selected_indices = np.sort(np.concatenate([positive_indices, sampled_negatives], axis=0))
        else:
            selected_indices = positive_indices

    stats = {
        "policy": policy,
        "total_rows": int(total_rows),
        "selected_rows": int(selected_indices.shape[0]),
        "pair_success_rows": int(np.sum(pair_success)),
        "explicit_negative_rows": int(explicit_negative_indices.shape[0]),
        "explicit_negative_selected_rows": int(selected_explicit_negative),
        "explicit_negative_ratio": float(ratio),
    }
    return selected_indices.astype(np.int64, copy=False), stats


def _clean_slug(value: Optional[str], fallback: str) -> str:
    text = _as_text(value) or fallback
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in text)
    cleaned = cleaned.strip("._")
    return cleaned or fallback


def _default_data_card_output_path(out_zarr: Path) -> Path:
    return out_zarr.with_suffix(".data_card.json")


def _default_data_card_plot_dir(card_json: Path) -> Path:
    return card_json.parent / f"{card_json.stem}.plots"


def _default_data_card_plot_prefix(*, run_name: str, card_json: Path) -> str:
    return _clean_slug(run_name, fallback=card_json.stem)


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


def _format_command(command: Sequence[str]) -> str:
    return " ".join(str(part) for part in command)


def _run_command_checked(
    *,
    command: Sequence[str],
    step_name: str,
    remediation: str,
) -> str:
    command_text = _format_command(command)
    try:
        completed = subprocess.run(
            [str(part) for part in command],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"{step_name} failed: command not found ({command_text}). "
            f"Remediation: {remediation}"
        ) from exc

    if int(completed.returncode) != 0:
        output = (completed.stderr or completed.stdout or "").strip()
        if output:
            output = " ".join(output.splitlines())
            if len(output) > 300:
                output = f"{output[:300]}..."
            details = f" (exit={int(completed.returncode)}; output={output})"
        else:
            details = f" (exit={int(completed.returncode)})"
        raise RuntimeError(
            f"{step_name} failed{details}. Command: {command_text}. "
            f"Remediation: {remediation}"
        )
    return command_text


def _resolve_source_dataset_id(
    *,
    source_root: zarr.Group,
    source_path: Path,
    registry_path: Optional[Path],
) -> str:
    """Resolve source dataset identity with detect/keypoint-aligned precedence."""
    explicit_dataset_id = _as_text(source_root.attrs.get("dataset_id"))
    if explicit_dataset_id:
        return str(explicit_dataset_id)

    dataset_id, _ = resolve_dataset_id(source_root, source_path)

    if registry_path is not None:
        registry = Registry(registry_path)
        try:
            registered_dataset_id = registry.scan_zarr(source_path)
            if registered_dataset_id:
                return str(registered_dataset_id)
        finally:
            registry.close()

    return str(dataset_id)


def _register_merged_dataset_in_registry(
    *,
    registry_path: Path,
    merged_zarr: Path,
    source_zarr_paths: Sequence[Path],
    set_id: Optional[str],
    set_name: Optional[str],
) -> Dict[str, Any]:
    """Register merged eye-mask export and update lineage/training-set linkage."""
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
                task_type=None,
                query_filter=existing_query_filter,
                dataset_ids=merged_ids,
                invocation=existing_invocation,
            )
            training_set_linked = True

        registry.replace_dataset_lineage(
            child_dataset_id=str(merged_dataset_id),
            parent_dataset_ids=source_dataset_ids,
            relationship_type="training_merge_source",
            source_set_id=str(set_id) if set_id else None,
            metadata={"producer": "export_eye_mask_training_zarr"},
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


def _iter_chunk_slices(shape: Tuple[int, ...], chunks: Tuple[int, ...]) -> Iterable[Tuple[slice, ...]]:
    if not chunks:
        yield tuple(slice(0, int(dim)) for dim in shape)
        return
    chunk_dims = []
    for axis, dim in enumerate(shape):
        chunk = chunks[axis] if axis < len(chunks) else chunks[-1]
        if int(chunk) <= 0:
            chunk = dim
        chunk_dims.append(int(chunk))
    grid = [int(math.ceil(int(dim) / int(chunk))) for dim, chunk in zip(shape, chunk_dims)]
    for idx in np.ndindex(*grid):
        slices: List[slice] = []
        for axis, chunk_idx in enumerate(idx):
            start = int(chunk_idx) * int(chunk_dims[axis])
            stop = min(start + int(chunk_dims[axis]), int(shape[axis]))
            slices.append(slice(start, stop))
        yield tuple(slices)


def _copy_array(src: zarr.Array, dest_group: zarr.Group, name: str) -> zarr.Array:
    chunks = src.chunks if src.chunks is not None else None
    dest = dest_group.create_array(
        name,
        shape=src.shape,
        dtype=src.dtype,
        chunks=chunks,
        overwrite=True,
    )
    if chunks is None:
        dest[...] = src[...]
        return dest
    for slc in _iter_chunk_slices(tuple(int(v) for v in src.shape), tuple(int(v) for v in chunks)):
        dest[slc] = src[slc]
    return dest


def _write_string_array(group: zarr.Group, name: str, values: Sequence[str]) -> zarr.Array:
    arr = group.create_array(
        name,
        shape=(int(len(values)),),
        dtype=VariableLengthUTF8(),
        chunks=(max(1, min(65536, int(len(values)) or 1)),),
        overwrite=True,
    )
    arr[:] = np.asarray([str(v) for v in values], dtype=object)
    return arr


def _resolve_run_name(root: zarr.Group, parent_name: str, explicit: Optional[str]) -> str:
    parent = root.get(parent_name)
    if not isinstance(parent, zarr.Group):
        raise ValueError(f"Missing required group '{parent_name}'.")
    if explicit:
        if explicit not in parent:
            raise ValueError(f"Run '{explicit}' not found under {parent_name}.")
        return str(explicit)
    latest = parent.attrs.get("latest")
    latest_text = _as_text(latest)
    if latest_text and latest_text in parent:
        return latest_text
    names = sorted(str(name) for name in parent.group_keys()) if hasattr(parent, "group_keys") else sorted(parent.keys())
    if not names:
        raise ValueError(f"No runs found under {parent_name}.")
    return str(names[-1])


def _resolve_eye_source(
    root: zarr.Group,
    *,
    eye_stage: str,
    eye_run: Optional[str],
) -> Tuple[str, str, zarr.Group]:
    if eye_stage not in {"auto", "eye_masks_runs", "refined_eye_masks_runs"}:
        raise ValueError(f"Unsupported eye_stage '{eye_stage}'.")

    stage_order = (
        ["refined_eye_masks_runs", "eye_masks_runs"]
        if eye_stage == "auto"
        else [eye_stage]
    )

    if eye_run:
        for stage in stage_order:
            parent = root.get(stage)
            if isinstance(parent, zarr.Group) and eye_run in parent:
                return stage, str(eye_run), parent[str(eye_run)]
        raise ValueError(f"Eye run '{eye_run}' not found in selected stage(s): {stage_order}.")

    for stage in stage_order:
        parent = root.get(stage)
        if not isinstance(parent, zarr.Group):
            continue
        latest = _as_text(parent.attrs.get("latest"))
        if latest and latest in parent:
            return stage, latest, parent[latest]
        names = sorted(str(name) for name in parent.group_keys()) if hasattr(parent, "group_keys") else sorted(parent.keys())
        if names:
            return stage, str(names[-1]), parent[str(names[-1])]
    raise ValueError(f"No eye-mask runs found in selected stage(s): {stage_order}.")


def _read_reason_labels_safe(group: zarr.Group) -> Optional[np.ndarray]:
    try:
        return read_reason_labels(group)
    except Exception:
        reason_arr = group.get("reason")
        if isinstance(reason_arr, zarr.Array):
            return np.asarray(reason_arr[:], dtype=object)
    return None


def _resolve_reason_labels(run_group: zarr.Group) -> Optional[np.ndarray]:
    metrics_group = run_group.get("metrics")
    if isinstance(metrics_group, zarr.Group):
        labels = _read_reason_labels_safe(metrics_group)
        if labels is not None:
            return labels

    labels = _read_reason_labels_safe(run_group)
    if labels is not None:
        return labels
    return None


def _resolve_mask_probs_name(run_group: zarr.Group) -> Optional[str]:
    for candidate in ("mask_probs_roi_refined", "mask_probs_roi"):
        if candidate in run_group:
            return candidate
    return None


def _normalized_split_ratios(
    train: float,
    val: float,
    test: float,
) -> Tuple[float, float, float]:
    train_v = max(0.0, float(train))
    val_v = max(0.0, float(val))
    test_v = max(0.0, float(test))
    total = train_v + val_v + test_v
    if total <= 0.0:
        raise ValueError("At least one split ratio must be > 0.")
    return train_v / total, val_v / total, test_v / total


def _make_split_indices(
    total_samples: int,
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = int(total_samples)
    if total <= 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty

    tr, vr, ter = _normalized_split_ratios(train_ratio, val_ratio, test_ratio)
    order = np.random.default_rng(int(seed)).permutation(total).astype(np.int64, copy=False)
    train_count = int(round(float(total) * tr))
    val_count = int(round(float(total) * vr))
    train_count = max(0, min(train_count, total))
    val_count = max(0, min(val_count, total - train_count))
    test_count = total - train_count - val_count
    if ter <= 0.0:
        val_count = total - train_count
        test_count = 0

    train_idx = order[:train_count]
    val_idx = order[train_count: train_count + val_count]
    test_idx = order[train_count + val_count: train_count + val_count + test_count]
    return train_idx, val_idx, test_idx


@dataclass
class EyeExportSelection:
    crop_run: str
    eye_stage: str
    eye_run: str
    total_samples: int
    channels: int
    mask_probs_name: Optional[str]


def _select_source_runs(
    source_root: zarr.Group,
    *,
    crop_run: Optional[str],
    eye_stage: str,
    eye_run: Optional[str],
) -> Tuple[EyeExportSelection, zarr.Group, zarr.Group]:
    stage_name, run_name, eye_group = _resolve_eye_source(
        source_root,
        eye_stage=eye_stage,
        eye_run=eye_run,
    )

    selected_crop = _as_text(crop_run)
    if selected_crop is None:
        selected_crop = _as_text(eye_group.attrs.get("source_crop_run"))
    if selected_crop is None:
        selected_crop = _resolve_run_name(source_root, "crop_runs", explicit=None)

    crop_parent = source_root.get("crop_runs")
    if not isinstance(crop_parent, zarr.Group) or selected_crop not in crop_parent:
        raise ValueError(f"Crop run '{selected_crop}' not found under crop_runs.")
    crop_group = crop_parent[str(selected_crop)]

    if "roi_images" not in crop_group:
        raise ValueError(f"crop_runs/{selected_crop} missing roi_images.")
    if "bbox_norm_coords" not in crop_group:
        raise ValueError(f"crop_runs/{selected_crop} missing bbox_norm_coords.")
    if "masks_roi" not in eye_group:
        raise ValueError(f"{stage_name}/{run_name} missing masks_roi.")
    if "ellipse_params" not in eye_group:
        raise ValueError(f"{stage_name}/{run_name} missing ellipse_params.")
    if "ellipse_success" not in eye_group:
        raise ValueError(f"{stage_name}/{run_name} missing ellipse_success.")

    roi_images = crop_group["roi_images"]
    masks_roi = eye_group["masks_roi"]
    if int(roi_images.shape[0]) != int(masks_roi.shape[0]):
        raise ValueError(
            f"Row mismatch: crop_runs/{selected_crop}/roi_images has {roi_images.shape[0]} rows "
            f"but {stage_name}/{run_name}/masks_roi has {masks_roi.shape[0]} rows."
        )

    selection = EyeExportSelection(
        crop_run=str(selected_crop),
        eye_stage=stage_name,
        eye_run=run_name,
        total_samples=int(masks_roi.shape[0]),
        channels=int(masks_roi.shape[1]) if masks_roi.ndim >= 2 else 0,
        mask_probs_name=_resolve_mask_probs_name(eye_group),
    )
    return selection, crop_group, eye_group


@dataclass(frozen=True)
class EyeMaskCardArtifacts:
    card_json: Path
    plot_dir: Path
    plot_prefix: str
    training_set_id: Optional[str]


def _resolve_card_artifacts(
    *,
    out_zarr: Path,
    run_name: str,
    training_set_id: Optional[str],
    data_card_output: Optional[Path],
    data_card_plot_dir: Optional[Path],
    data_card_plot_prefix: Optional[str],
) -> EyeMaskCardArtifacts:
    card_json = (
        Path(data_card_output).expanduser().resolve()
        if data_card_output is not None
        else _default_data_card_output_path(out_zarr)
    )
    plot_dir = (
        Path(data_card_plot_dir).expanduser().resolve()
        if data_card_plot_dir is not None
        else _default_data_card_plot_dir(card_json)
    )
    plot_prefix = (
        _clean_slug(_as_text(data_card_plot_prefix), fallback=card_json.stem)
        if data_card_plot_prefix is not None
        else _default_data_card_plot_prefix(run_name=run_name, card_json=card_json)
    )
    training_set_text = _as_text(training_set_id)
    resolved_training_set = (
        _clean_slug(training_set_text, fallback="training_set")
        if training_set_text is not None
        else None
    )
    return EyeMaskCardArtifacts(
        card_json=card_json,
        plot_dir=plot_dir,
        plot_prefix=plot_prefix,
        training_set_id=resolved_training_set,
    )


def _run_data_card_workflow(
    *,
    registry_path: Optional[Path],
    artifacts: EyeMaskCardArtifacts,
    no_plots: bool,
) -> Dict[str, Any]:
    if not artifacts.training_set_id:
        raise ValueError(
            "Eye-mask data-card aggregation requires an explicit training_set_id. "
            "Provide --training-set-id (or training_set_id=...) to run aggregation."
        )

    sync_cmd: List[str] = [
        "scripts/py",
        "-m",
        "fisheye.utils.sync_eye_mask_profile_registry",
        "--apply",
    ]
    if registry_path is not None:
        sync_cmd.extend(["--registry", str(registry_path)])
    sync_text = _run_command_checked(
        command=sync_cmd,
        step_name="Eye-mask profile sync",
        remediation=(
            "Run scripts/py -m fisheye.utils.sync_eye_mask_profile_registry --apply "
            "before retrying eye-mask data-card aggregation."
        ),
    )

    aggregate_cmd: List[str] = [
        "scripts/py",
        "-m",
        "fisheye.utils.aggregate_eye_mask_training_data_card",
        "--training-set",
        str(artifacts.training_set_id),
        "--output",
        str(artifacts.card_json),
    ]
    if registry_path is not None:
        aggregate_cmd.extend(["--registry", str(registry_path)])
    if no_plots:
        aggregate_cmd.append("--no-plots")
    else:
        aggregate_cmd.extend(
            [
                "--plot-dir",
                str(artifacts.plot_dir),
                "--plot-prefix",
                str(artifacts.plot_prefix),
            ]
        )
    aggregate_text = _run_command_checked(
        command=aggregate_cmd,
        step_name="Eye-mask data-card aggregation",
        remediation=(
            "Run scripts/py -m fisheye.utils.aggregate_eye_mask_training_data_card "
            "--training-set <set_id> (and optional --registry) after successful profile sync."
        ),
    )

    plot_text: Optional[str] = None
    if not no_plots:
        plot_cmd: List[str] = [
            "scripts/py",
            "-m",
            "fisheye.utils.plot_eye_mask_training_data_card",
            "--card-json",
            str(artifacts.card_json),
            "--outdir",
            str(artifacts.plot_dir),
            "--prefix",
            str(artifacts.plot_prefix),
        ]
        plot_text = _run_command_checked(
            command=plot_cmd,
            step_name="Eye-mask data-card plotting",
            remediation=(
                "Run scripts/py -m fisheye.utils.plot_eye_mask_training_data_card "
                f"--card-json {artifacts.card_json} --outdir {artifacts.plot_dir}."
            ),
        )

    return {
        "profile_sync_command": sync_text,
        "aggregate_command": aggregate_text,
        "plot_command": plot_text,
        "plots_generated": not bool(no_plots),
    }


@dataclass(frozen=True)
class EyeMergeSourceSpec:
    source_zarr: Path
    crop_run: Optional[str] = None
    eye_stage: str = "auto"
    eye_run: Optional[str] = None


@dataclass
class _ResolvedEyeMergeSource:
    source_path: Path
    source_dataset_id: str
    selection: EyeExportSelection
    crop_group: zarr.Group
    eye_group: zarr.Group
    roi_images: zarr.Array
    bbox_norm: zarr.Array
    crop_bbox: Optional[zarr.Array]
    masks_roi: zarr.Array
    ellipse_params: zarr.Array
    ellipse_success: zarr.Array
    mask_probs_name: Optional[str]
    mask_probs_src: Optional[zarr.Array]
    source_frame_idx: np.ndarray
    source_roi_idx: np.ndarray
    detection_source: np.ndarray
    eye_separation_data: np.ndarray
    reason_values: np.ndarray
    selected_indices: np.ndarray
    row_gate_policy: str
    row_gate_stats: Dict[str, Any]

    @property
    def total_samples(self) -> int:
        return int(self.selection.total_samples)

    @property
    def selected_samples(self) -> int:
        return int(self.selected_indices.shape[0])


def _collapse_source_attr(values: Sequence[str], *, mixed_value: str = "mixed") -> str:
    unique = sorted({str(value) for value in values if str(value)})
    if not unique:
        return ""
    if len(unique) == 1:
        return unique[0]
    return mixed_value


def _merge_chunks(chunks: Optional[Tuple[int, ...]], *, total_samples: int) -> Optional[Tuple[int, ...]]:
    if chunks is None:
        return None
    normalized: List[int] = []
    for idx, value in enumerate(chunks):
        try:
            chunk = int(value)
        except Exception:
            chunk = 0
        if chunk <= 0:
            chunk = max(1, int(total_samples))
        if idx == 0:
            chunk = max(1, min(chunk, max(1, int(total_samples))))
        normalized.append(chunk)
    return tuple(normalized) if normalized else None


def _read_row_selection(array_obj: zarr.Array, row_indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(row_indices, dtype=np.int64).reshape(-1)
    tail_shape = tuple(int(v) for v in array_obj.shape[1:])
    if indices.size == 0:
        return np.empty((0, *tail_shape), dtype=array_obj.dtype)

    if indices.size == int(array_obj.shape[0]) and np.array_equal(
        indices,
        np.arange(indices.size, dtype=np.int64),
    ):
        return np.asarray(array_obj[:])

    if indices.size == 1:
        idx = int(indices[0])
        return np.asarray(array_obj[idx:idx + 1, ...])

    starts: List[int] = []
    stops: List[int] = []
    run_start = int(indices[0])
    prev = run_start
    for raw_idx in indices[1:]:
        idx = int(raw_idx)
        if idx == prev + 1:
            prev = idx
            continue
        starts.append(run_start)
        stops.append(prev + 1)
        run_start = idx
        prev = idx
    starts.append(run_start)
    stops.append(prev + 1)

    blocks = [np.asarray(array_obj[start:stop, ...]) for start, stop in zip(starts, stops, strict=False)]
    if len(blocks) == 1:
        return blocks[0]
    return np.concatenate(blocks, axis=0)


def _validate_source_for_merge(
    source: _ResolvedEyeMergeSource,
    *,
    expected_input_format: str,
    expected_label_mode: str,
) -> None:
    roi_images = source.roi_images
    masks_roi = source.masks_roi
    ellipse_params = source.ellipse_params
    ellipse_success = source.ellipse_success
    bbox_norm = source.bbox_norm
    total_samples = int(source.total_samples)
    source_id = source.source_path.name

    if roi_images.ndim < 3:
        raise ValueError(f"{source_id}: roi_images must be (N,H,W) or (N,H,W,C), got {roi_images.shape}.")
    if int(roi_images.shape[0]) != total_samples:
        raise ValueError(
            f"{source_id}: roi_images length mismatch ({roi_images.shape[0]} != {total_samples})."
        )
    if expected_input_format == "rgb":
        if roi_images.ndim != 4 or int(roi_images.shape[-1]) != 3:
            raise ValueError(f"{source_id}: expected RGB roi_images with shape (N,H,W,3), got {roi_images.shape}.")
    elif roi_images.ndim == 4 and int(roi_images.shape[-1]) == 3:
        raise ValueError(f"{source_id}: roi_images appears RGB but input_format is gray.")

    if bbox_norm.ndim != 2 or int(bbox_norm.shape[1]) != 4:
        raise ValueError(f"{source_id}: bbox_norm_coords must be (N,4), got {bbox_norm.shape}.")
    if int(bbox_norm.shape[0]) != total_samples:
        raise ValueError(
            f"{source_id}: bbox_norm_coords length mismatch ({bbox_norm.shape[0]} != {total_samples})."
        )
    if isinstance(source.crop_bbox, zarr.Array):
        if source.crop_bbox.ndim != 2 or int(source.crop_bbox.shape[1]) != 4:
            raise ValueError(f"{source_id}: crop_bbox_norm_coords must be (N,4), got {source.crop_bbox.shape}.")
        if int(source.crop_bbox.shape[0]) != total_samples:
            raise ValueError(
                f"{source_id}: crop_bbox_norm_coords length mismatch ({source.crop_bbox.shape[0]} != {total_samples})."
            )

    if masks_roi.ndim != 4:
        raise ValueError(f"{source_id}: masks_roi must be (N,C,H,W), got {masks_roi.shape}.")
    if int(masks_roi.shape[0]) != total_samples:
        raise ValueError(f"{source_id}: masks_roi length mismatch ({masks_roi.shape[0]} != {total_samples}).")
    if expected_label_mode == "lr" and int(masks_roi.shape[1]) != 2:
        raise ValueError(f"{source_id}: label_mode=lr requires masks channel count 2, got {masks_roi.shape[1]}.")
    if expected_label_mode == "union" and int(masks_roi.shape[1]) != 1:
        raise ValueError(
            f"{source_id}: label_mode=union requires masks channel count 1, got {masks_roi.shape[1]}."
        )
    if int(masks_roi.shape[2]) != int(roi_images.shape[1]) or int(masks_roi.shape[3]) != int(roi_images.shape[2]):
        raise ValueError(
            f"{source_id}: masks_roi spatial shape {masks_roi.shape[2:]} does not match roi_images {roi_images.shape[1:3]}."
        )

    if ellipse_params.ndim != 3 or int(ellipse_params.shape[-1]) != 5:
        raise ValueError(f"{source_id}: ellipse_params must be (N,C,5), got {ellipse_params.shape}.")
    if ellipse_success.ndim != 2:
        raise ValueError(f"{source_id}: ellipse_success must be (N,C), got {ellipse_success.shape}.")
    if tuple(ellipse_params.shape[:2]) != tuple(ellipse_success.shape[:2]):
        raise ValueError(
            f"{source_id}: ellipse_params/ellipse_success channel mismatch "
            f"({ellipse_params.shape[:2]} != {ellipse_success.shape[:2]})."
        )
    if int(ellipse_params.shape[0]) != total_samples or int(ellipse_success.shape[0]) != total_samples:
        raise ValueError(f"{source_id}: ellipse arrays length mismatch with total_samples={total_samples}.")

    if int(source.source_frame_idx.shape[0]) != total_samples:
        raise ValueError(
            f"{source_id}: source frame-index length mismatch ({source.source_frame_idx.shape[0]} != {total_samples})."
        )
    if int(source.detection_source.shape[0]) != total_samples:
        raise ValueError(
            f"{source_id}: detection_source length mismatch ({source.detection_source.shape[0]} != {total_samples})."
        )
    if int(source.eye_separation_data.shape[0]) != total_samples:
        raise ValueError(
            f"{source_id}: eye_separation length mismatch ({source.eye_separation_data.shape[0]} != {total_samples})."
        )
    if int(source.reason_values.shape[0]) != total_samples:
        raise ValueError(
            f"{source_id}: reason label length mismatch ({source.reason_values.shape[0]} != {total_samples})."
        )
    if isinstance(source.mask_probs_src, zarr.Array) and source.mask_probs_name:
        if tuple(source.mask_probs_src.shape) != tuple(masks_roi.shape):
            raise ValueError(
                f"{source_id}: {source.mask_probs_name} shape {source.mask_probs_src.shape} "
                f"does not match masks_roi {masks_roi.shape}."
            )


def _validate_merge_schema_compatibility(
    reference: _ResolvedEyeMergeSource,
    candidate: _ResolvedEyeMergeSource,
) -> None:
    reference_id = reference.source_path.name
    candidate_id = candidate.source_path.name

    if tuple(candidate.roi_images.shape[1:]) != tuple(reference.roi_images.shape[1:]):
        raise ValueError(
            f"{candidate_id}: roi_images shape tail {candidate.roi_images.shape[1:]} "
            f"does not match {reference_id} {reference.roi_images.shape[1:]}."
        )
    if candidate.roi_images.dtype != reference.roi_images.dtype:
        raise ValueError(
            f"{candidate_id}: roi_images dtype {candidate.roi_images.dtype} "
            f"does not match {reference_id} {reference.roi_images.dtype}."
        )

    if tuple(candidate.bbox_norm.shape[1:]) != tuple(reference.bbox_norm.shape[1:]):
        raise ValueError(
            f"{candidate_id}: bbox_norm_coords shape tail {candidate.bbox_norm.shape[1:]} "
            f"does not match {reference_id} {reference.bbox_norm.shape[1:]}."
        )
    if candidate.bbox_norm.dtype != reference.bbox_norm.dtype:
        raise ValueError(
            f"{candidate_id}: bbox_norm_coords dtype {candidate.bbox_norm.dtype} "
            f"does not match {reference_id} {reference.bbox_norm.dtype}."
        )

    if tuple(candidate.masks_roi.shape[1:]) != tuple(reference.masks_roi.shape[1:]):
        raise ValueError(
            f"{candidate_id}: masks_roi shape tail {candidate.masks_roi.shape[1:]} "
            f"does not match {reference_id} {reference.masks_roi.shape[1:]}."
        )
    if candidate.masks_roi.dtype != reference.masks_roi.dtype:
        raise ValueError(
            f"{candidate_id}: masks_roi dtype {candidate.masks_roi.dtype} "
            f"does not match {reference_id} {reference.masks_roi.dtype}."
        )

    if tuple(candidate.ellipse_params.shape[1:]) != tuple(reference.ellipse_params.shape[1:]):
        raise ValueError(
            f"{candidate_id}: ellipse_params shape tail {candidate.ellipse_params.shape[1:]} "
            f"does not match {reference_id} {reference.ellipse_params.shape[1:]}."
        )
    if tuple(candidate.ellipse_success.shape[1:]) != tuple(reference.ellipse_success.shape[1:]):
        raise ValueError(
            f"{candidate_id}: ellipse_success shape tail {candidate.ellipse_success.shape[1:]} "
            f"does not match {reference_id} {reference.ellipse_success.shape[1:]}."
        )


def _resolve_merge_source(
    source_spec: EyeMergeSourceSpec,
    *,
    registry_path: Optional[Path],
    row_gate_policy: str,
    explicit_negative_ratio: float,
    split_seed: int,
) -> _ResolvedEyeMergeSource:
    source_path = Path(source_spec.source_zarr).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Source zarr does not exist: {source_path}")

    try:
        src_root = zarr.open_group(str(source_path), mode="r", use_consolidated=False)
    except TypeError:
        src_root = zarr.open_group(str(source_path), mode="r")

    selection, crop_group, eye_group = _select_source_runs(
        src_root,
        crop_run=source_spec.crop_run,
        eye_stage=source_spec.eye_stage,
        eye_run=source_spec.eye_run,
    )

    roi_images = crop_group["roi_images"]
    bbox_norm = crop_group["bbox_norm_coords"]
    crop_bbox = crop_group.get("crop_bbox_norm_coords")
    crop_frame_indices = crop_group.get("frame_indices")
    crop_detection_source = crop_group.get("detection_source")

    masks_roi = eye_group["masks_roi"]
    ellipse_params = eye_group["ellipse_params"]
    ellipse_success = eye_group["ellipse_success"]
    eye_separation = eye_group.get("eye_separation")
    reason_labels = _resolve_reason_labels(eye_group)
    mask_probs_name = selection.mask_probs_name
    mask_probs_src = eye_group[mask_probs_name] if mask_probs_name else None

    total_samples = int(selection.total_samples)
    local_frame_indices = np.arange(total_samples, dtype=np.int64)
    source_frame_idx = (
        np.asarray(crop_frame_indices[:], dtype=np.int64)
        if isinstance(crop_frame_indices, zarr.Array)
        else local_frame_indices.copy()
    )
    detection_source = (
        np.asarray(crop_detection_source[:], dtype=np.int8)
        if isinstance(crop_detection_source, zarr.Array)
        else np.zeros((total_samples,), dtype=np.int8)
    )
    if eye_separation is None:
        eye_separation_data = np.full((total_samples,), np.nan, dtype=np.float32)
    else:
        eye_separation_data = np.asarray(eye_separation[:], dtype=np.float32)

    if reason_labels is not None:
        reason_values = np.asarray(reason_labels, dtype=object)
        reason_values = np.asarray([str(v) if v is not None else "" for v in reason_values], dtype=object)
    else:
        reason_values = np.where(
            detection_source.astype(np.int8, copy=False) == 1,
            "interpolated",
            "clean",
        ).astype(object)

    source_dataset_id = _resolve_source_dataset_id(
        source_root=src_root,
        source_path=source_path,
        registry_path=registry_path,
    )
    selected_indices, row_gate_stats = _resolve_row_gate_selection(
        source_path=source_path,
        source_dataset_id=source_dataset_id,
        ellipse_success=ellipse_success,
        reason_values=reason_values,
        row_gate_policy=row_gate_policy,
        explicit_negative_ratio=explicit_negative_ratio,
        split_seed=split_seed,
    )
    return _ResolvedEyeMergeSource(
        source_path=source_path,
        source_dataset_id=source_dataset_id,
        selection=selection,
        crop_group=crop_group,
        eye_group=eye_group,
        roi_images=roi_images,
        bbox_norm=bbox_norm,
        crop_bbox=crop_bbox if isinstance(crop_bbox, zarr.Array) else None,
        masks_roi=masks_roi,
        ellipse_params=ellipse_params,
        ellipse_success=ellipse_success,
        mask_probs_name=mask_probs_name,
        mask_probs_src=mask_probs_src if isinstance(mask_probs_src, zarr.Array) else None,
        source_frame_idx=source_frame_idx,
        source_roi_idx=np.arange(total_samples, dtype=np.int64),
        detection_source=detection_source,
        eye_separation_data=eye_separation_data,
        reason_values=reason_values,
        selected_indices=selected_indices,
        row_gate_policy=str(row_gate_stats["policy"]),
        row_gate_stats=dict(row_gate_stats),
    )


def export_merged_eye_mask_training_zarr_from_sources(
    source_specs: Sequence[EyeMergeSourceSpec | Path | str],
    out_zarr: Path,
    *,
    run_name: str = "merged_export_smoke",
    input_format: str = "gray",
    label_mode: str = "lr",
    split_train: float = 0.8,
    split_val: float = 0.2,
    split_test: float = 0.0,
    split_seed: int = 42,
    row_gate_policy: str = "all_rows",
    explicit_negative_ratio: float = 0.25,
    overwrite: bool = False,
    validate: bool = True,
    registry: Optional[Path] = None,
    training_set_id: Optional[str] = None,
    training_set_name: Optional[str] = None,
    aggregate_training_data_card: bool = False,
    data_card_output: Optional[Path] = None,
    data_card_plot_dir: Optional[Path] = None,
    data_card_plot_prefix: Optional[str] = None,
    data_card_no_plots: bool = False,
) -> Dict[str, Any]:
    if not source_specs:
        raise ValueError("At least one source spec is required for merged eye-mask export.")

    normalized_source_specs: List[EyeMergeSourceSpec] = []
    for spec in source_specs:
        if isinstance(spec, EyeMergeSourceSpec):
            normalized_source_specs.append(spec)
            continue
        if isinstance(spec, (str, Path)):
            normalized_source_specs.append(
                EyeMergeSourceSpec(
                    source_zarr=Path(spec),
                    crop_run=None,
                    eye_stage="auto",
                    eye_run=None,
                )
            )
            continue
        raise TypeError(
            "source_specs items must be EyeMergeSourceSpec or path-like values. "
            f"Got {type(spec).__name__}."
        )

    out_path = Path(out_zarr).expanduser().resolve()
    normalized_input_format = _normalize_input_format(input_format)
    if normalized_input_format is None:
        raise ValueError(f"Unsupported input_format '{input_format}'. Expected gray or rgb.")
    normalized_label_mode = _normalize_label_mode(label_mode)
    if normalized_label_mode is None:
        raise ValueError(f"Unsupported label_mode '{label_mode}'. Expected lr or union.")
    normalized_row_gate_policy = _normalize_row_gate_policy(row_gate_policy)
    normalized_explicit_negative_ratio = _normalize_explicit_negative_ratio(explicit_negative_ratio)
    registry_path = Path(registry).expanduser().resolve() if registry is not None else None
    artifacts = _resolve_card_artifacts(
        out_zarr=out_path,
        run_name=run_name,
        training_set_id=training_set_id,
        data_card_output=data_card_output,
        data_card_plot_dir=data_card_plot_dir,
        data_card_plot_prefix=data_card_plot_prefix,
    )
    if aggregate_training_data_card and not artifacts.training_set_id:
        raise ValueError(
            "aggregate_training_data_card requires training_set_id. "
            "Pass --training-set-id (CLI) or training_set_id=... (API)."
        )

    if out_path.exists():
        if not overwrite:
            raise FileExistsError(f"Destination exists: {out_path}")
        if out_path.is_dir():
            shutil.rmtree(out_path)
        else:
            out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    resolved_sources: List[_ResolvedEyeMergeSource] = [
        _resolve_merge_source(
            spec,
            registry_path=registry_path,
            row_gate_policy=normalized_row_gate_policy,
            explicit_negative_ratio=normalized_explicit_negative_ratio,
            split_seed=int(split_seed),
        )
        for spec in normalized_source_specs
    ]
    if not resolved_sources:
        raise ValueError("No source data resolved for merged eye-mask export.")

    for source in resolved_sources:
        _validate_source_for_merge(
            source,
            expected_input_format=normalized_input_format,
            expected_label_mode=normalized_label_mode,
        )
    ref_source = resolved_sources[0]
    for source in resolved_sources[1:]:
        _validate_merge_schema_compatibility(ref_source, source)

    source_mask_prob_names = [source.mask_probs_name for source in resolved_sources]
    mask_probs_name = source_mask_prob_names[0]
    if not mask_probs_name or any(name != mask_probs_name for name in source_mask_prob_names):
        mask_probs_name = None
    if mask_probs_name and any(source.mask_probs_src is None for source in resolved_sources):
        mask_probs_name = None

    source_counts = [int(source.selected_samples) for source in resolved_sources]
    total_samples = int(sum(source_counts))
    if total_samples <= 0:
        raise ValueError(
            "Row gating selected zero samples across all sources. "
            "Relax row-gate settings or provide sources with usable eye-mask rows."
        )
    local_frame_indices = np.arange(total_samples, dtype=np.int64)
    detection_source = (
        np.concatenate(
            [
                np.asarray(source.detection_source[source.selected_indices], dtype=np.int8)
                for source in resolved_sources
                if source.selected_samples > 0
            ]
        ).astype(np.int8, copy=False)
        if total_samples > 0
        else np.empty((0,), dtype=np.int8)
    )
    source_frame_idx = (
        np.concatenate(
            [
                np.asarray(source.source_frame_idx[source.selected_indices], dtype=np.int64)
                for source in resolved_sources
                if source.selected_samples > 0
            ]
        ).astype(np.int64, copy=False)
        if total_samples > 0
        else np.empty((0,), dtype=np.int64)
    )
    source_roi_idx = (
        np.concatenate(
            [
                np.asarray(source.source_roi_idx[source.selected_indices], dtype=np.int64)
                for source in resolved_sources
                if source.selected_samples > 0
            ]
        ).astype(np.int64, copy=False)
        if total_samples > 0
        else np.empty((0,), dtype=np.int64)
    )
    eye_separation_data = (
        np.concatenate(
            [
                np.asarray(source.eye_separation_data[source.selected_indices], dtype=np.float32)
                for source in resolved_sources
                if source.selected_samples > 0
            ]
        ).astype(np.float32, copy=False)
        if total_samples > 0
        else np.empty((0,), dtype=np.float32)
    )
    reason_values = (
        np.concatenate(
            [
                np.asarray(source.reason_values[source.selected_indices], dtype=object)
                for source in resolved_sources
                if source.selected_samples > 0
            ]
        ).astype(object, copy=False)
        if total_samples > 0
        else np.empty((0,), dtype=object)
    )
    source_dataset_idx = np.empty((total_samples,), dtype=np.int32)
    offset = 0
    for idx, count in enumerate(source_counts):
        if count > 0:
            source_dataset_idx[offset: offset + count] = int(idx)
            offset += int(count)

    row_gate_policies = sorted({str(source.row_gate_policy) for source in resolved_sources})
    applied_row_gate_policy = row_gate_policies[0] if len(row_gate_policies) == 1 else "mixed"
    row_gate_total_rows = int(sum(int(source.row_gate_stats.get("total_rows", 0)) for source in resolved_sources))
    row_gate_selected_rows = int(sum(int(source.row_gate_stats.get("selected_rows", 0)) for source in resolved_sources))
    row_gate_pair_success_rows = int(
        sum(int(source.row_gate_stats.get("pair_success_rows", 0)) for source in resolved_sources)
    )
    row_gate_explicit_negative_rows = int(
        sum(int(source.row_gate_stats.get("explicit_negative_rows", 0)) for source in resolved_sources)
    )
    row_gate_explicit_negative_selected_rows = int(
        sum(int(source.row_gate_stats.get("explicit_negative_selected_rows", 0)) for source in resolved_sources)
    )
    row_gate_applied = bool(applied_row_gate_policy != "all_rows" or row_gate_selected_rows != row_gate_total_rows)

    source_stage = _collapse_source_attr([source.selection.eye_stage for source in resolved_sources])
    source_eye_run = _collapse_source_attr([source.selection.eye_run for source in resolved_sources])
    source_crop_run = _collapse_source_attr([source.selection.crop_run for source in resolved_sources])
    source_zarr_paths = [str(source.source_path) for source in resolved_sources]

    dst_root = zarr.open_group(str(out_path), mode="w")
    training_export_payload = {
        "tool": "fisheye.utils.export_eye_mask_training_zarr",
        "created_at_utc": _utc_now(),
        "input_format": normalized_input_format,
        "label_mode": normalized_label_mode,
        "source_stage": source_stage,
        "source_eye_run": source_eye_run,
        "source_crop_run": source_crop_run,
        "source_count": int(len(resolved_sources)),
        "source_zarr_paths": source_zarr_paths,
        "row_gate": {
            "requested_policy": normalized_row_gate_policy,
            "applied_policy": applied_row_gate_policy,
            "applied": bool(row_gate_applied),
            "explicit_negative_ratio": float(normalized_explicit_negative_ratio),
            "total_rows": int(row_gate_total_rows),
            "selected_rows": int(row_gate_selected_rows),
            "pair_success_rows": int(row_gate_pair_success_rows),
            "explicit_negative_rows": int(row_gate_explicit_negative_rows),
            "explicit_negative_selected_rows": int(row_gate_explicit_negative_selected_rows),
        },
        "split_seed": int(split_seed),
        "training_set_id": artifacts.training_set_id,
        "aggregate_training_data_card": bool(aggregate_training_data_card),
        "data_card_no_plots": bool(data_card_no_plots),
        "artifact_paths": {
            "data_card_json": str(artifacts.card_json),
            "data_card_plot_dir": str(artifacts.plot_dir),
            "data_card_plot_prefix": str(artifacts.plot_prefix),
        },
    }
    dst_root.attrs.update(
        {
            "zarr_purpose": "training",
            "training_task": "eye_masks",
            "training_export": training_export_payload,
        }
    )

    dst_crop_parent = dst_root.create_group("crop_runs")
    dst_crop_parent.attrs["latest"] = run_name
    dst_crop = dst_crop_parent.create_group(run_name)
    roi_shape = (total_samples, *tuple(int(v) for v in ref_source.roi_images.shape[1:]))
    roi_images_dest = dst_crop.create_array(
        "roi_images",
        shape=roi_shape,
        dtype=ref_source.roi_images.dtype,
        chunks=_merge_chunks(getattr(ref_source.roi_images, "chunks", None), total_samples=total_samples),
        overwrite=True,
    )
    bbox_shape = (total_samples, *tuple(int(v) for v in ref_source.bbox_norm.shape[1:]))
    bbox_dest = dst_crop.create_array(
        "bbox_norm_coords",
        shape=bbox_shape,
        dtype=ref_source.bbox_norm.dtype,
        chunks=_merge_chunks(getattr(ref_source.bbox_norm, "chunks", None), total_samples=total_samples),
        overwrite=True,
    )
    crop_bbox_dest = dst_crop.create_array(
        "crop_bbox_norm_coords",
        shape=bbox_shape,
        dtype=ref_source.bbox_norm.dtype,
        chunks=_merge_chunks(getattr(ref_source.bbox_norm, "chunks", None), total_samples=total_samples),
        overwrite=True,
    )

    dst_eye_parent = dst_root.create_group("eye_masks_runs")
    dst_eye_parent.attrs["latest"] = run_name
    dst_eye = dst_eye_parent.create_group(run_name)
    masks_shape = (total_samples, *tuple(int(v) for v in ref_source.masks_roi.shape[1:]))
    masks_dest = dst_eye.create_array(
        "masks_roi",
        shape=masks_shape,
        dtype=ref_source.masks_roi.dtype,
        chunks=_merge_chunks(getattr(ref_source.masks_roi, "chunks", None), total_samples=total_samples),
        overwrite=True,
    )
    ellipse_params_shape = (total_samples, *tuple(int(v) for v in ref_source.ellipse_params.shape[1:]))
    ellipse_params_dest = dst_eye.create_array(
        "ellipse_params",
        shape=ellipse_params_shape,
        dtype=ref_source.ellipse_params.dtype,
        chunks=_merge_chunks(getattr(ref_source.ellipse_params, "chunks", None), total_samples=total_samples),
        overwrite=True,
    )
    ellipse_success_shape = (total_samples, *tuple(int(v) for v in ref_source.ellipse_success.shape[1:]))
    ellipse_success_dest = dst_eye.create_array(
        "ellipse_success",
        shape=ellipse_success_shape,
        dtype=ref_source.ellipse_success.dtype,
        chunks=_merge_chunks(getattr(ref_source.ellipse_success, "chunks", None), total_samples=total_samples),
        overwrite=True,
    )
    mask_probs_dest: Optional[zarr.Array] = None
    if mask_probs_name:
        mask_probs_dest = dst_eye.create_array(
            mask_probs_name,
            shape=masks_shape,
            dtype=resolved_sources[0].mask_probs_src.dtype if resolved_sources[0].mask_probs_src is not None else np.float32,
            chunks=_merge_chunks(
                getattr(resolved_sources[0].mask_probs_src, "chunks", None)
                if resolved_sources[0].mask_probs_src is not None
                else None,
                total_samples=total_samples,
            ),
            overwrite=True,
        )

    offset = 0
    for source in resolved_sources:
        selected = np.asarray(source.selected_indices, dtype=np.int64)
        next_offset = offset + int(selected.shape[0])
        if next_offset > offset:
            roi_images_dest[offset:next_offset, ...] = _read_row_selection(source.roi_images, selected)
            bbox_dest[offset:next_offset, ...] = _read_row_selection(source.bbox_norm, selected)
            if isinstance(source.crop_bbox, zarr.Array):
                crop_bbox_dest[offset:next_offset, ...] = _read_row_selection(source.crop_bbox, selected)
            else:
                crop_bbox_dest[offset:next_offset, ...] = np.asarray(
                    _read_row_selection(source.bbox_norm, selected),
                    dtype=np.float32,
                )
            masks_dest[offset:next_offset, ...] = _read_row_selection(source.masks_roi, selected)
            ellipse_params_dest[offset:next_offset, ...] = _read_row_selection(source.ellipse_params, selected)
            ellipse_success_dest[offset:next_offset, ...] = _read_row_selection(source.ellipse_success, selected)
            if mask_probs_dest is not None and source.mask_probs_src is not None:
                mask_probs_dest[offset:next_offset, ...] = _read_row_selection(source.mask_probs_src, selected)
        offset = next_offset

    dst_crop.create_array(
        "frame_indices",
        data=local_frame_indices,
        chunks=(max(1, min(total_samples, 65536)),),
        overwrite=True,
    )
    dst_crop.create_array(
        "detection_source",
        data=detection_source,
        chunks=(max(1, min(total_samples, 65536)),),
        overwrite=True,
    )
    dst_crop.attrs.update(
        {
            "source_crop_run": source_crop_run,
            "source_zarr_paths": source_zarr_paths,
        }
    )
    if len(resolved_sources) == 1:
        dst_crop.attrs["source_zarr_path"] = source_zarr_paths[0]

    dst_eye.create_array(
        "eye_separation",
        data=eye_separation_data,
        chunks=(max(1, min(total_samples, 65536)),),
        overwrite=True,
    )
    dst_eye.create_array(
        "frame_indices",
        data=local_frame_indices,
        chunks=(max(1, min(total_samples, 65536)),),
        overwrite=True,
    )
    dst_eye.create_array(
        "detection_source",
        data=detection_source,
        chunks=(max(1, min(total_samples, 65536)),),
        overwrite=True,
    )
    write_reason_columns(
        dst_eye,
        reason_values,
        chunk_size=max(1, min(total_samples, 65536)),
        include_reason_text=True,
        overwrite=True,
    )

    for attr_name in (
        "method",
        "eye_labels",
        "min_eye_separation",
        "max_eye_separation",
        "config",
        "source_keypoints_run",
        "source_keypoint_run",
        "source_keypoint_group",
        "source_eye_masks_run",
        "source_eye_masks_method",
        "reason_counts",
    ):
        ref_value = ref_source.eye_group.attrs.get(attr_name)
        if ref_value is None:
            continue
        if all(source.eye_group.attrs.get(attr_name) == ref_value for source in resolved_sources):
            dst_eye.attrs[attr_name] = ref_value
    dst_eye.attrs.update(
        {
            "source_eye_stage": source_stage,
            "source_eye_run": source_eye_run,
            "source_crop_run": source_crop_run,
            "source_zarr_paths": source_zarr_paths,
            "label_mode": normalized_label_mode,
            "row_gate_policy": applied_row_gate_policy,
            "row_gate_requested_policy": normalized_row_gate_policy,
            "row_gate_applied": bool(row_gate_applied),
            "row_gate_explicit_negative_ratio": float(normalized_explicit_negative_ratio),
            "row_gate_counts": {
                "total_rows": int(row_gate_total_rows),
                "selected_rows": int(row_gate_selected_rows),
                "pair_success_rows": int(row_gate_pair_success_rows),
                "explicit_negative_rows": int(row_gate_explicit_negative_rows),
                "explicit_negative_selected_rows": int(row_gate_explicit_negative_selected_rows),
            },
        }
    )
    if len(resolved_sources) == 1:
        dst_eye.attrs["source_zarr_path"] = source_zarr_paths[0]

    train_idx, val_idx, test_idx = _make_split_indices(
        total_samples,
        train_ratio=float(split_train),
        val_ratio=float(split_val),
        test_ratio=float(split_test),
        seed=int(split_seed),
    )
    split_group = dst_root.create_group("splits")
    split_group.create_array(
        "train_indices",
        data=train_idx.astype(np.int64, copy=False),
        chunks=(max(1, min(train_idx.size or 1, 65536)),),
    )
    split_group.create_array(
        "val_indices",
        data=val_idx.astype(np.int64, copy=False),
        chunks=(max(1, min(val_idx.size or 1, 65536)),),
    )
    split_group.create_array(
        "test_indices",
        data=test_idx.astype(np.int64, copy=False),
        chunks=(max(1, min(test_idx.size or 1, 65536)),),
    )
    split_group.attrs.update(
        {
            "split_seed": int(split_seed),
            "split_ratios": {
                "train": float(split_train),
                "val": float(split_val),
                "test": float(split_test),
            },
        }
    )

    source_index = dst_root.create_group("source_index")
    source_index.create_array(
        "source_dataset_idx",
        data=source_dataset_idx,
        chunks=(max(1, min(total_samples, 65536)),),
        overwrite=True,
    )
    source_index.create_array(
        "source_frame_idx",
        data=source_frame_idx.astype(np.int64, copy=False),
        chunks=(max(1, min(total_samples, 65536)),),
        overwrite=True,
    )
    source_index.create_array(
        "source_roi_idx",
        data=source_roi_idx.astype(np.int64, copy=False),
        chunks=(max(1, min(total_samples, 65536)),),
        overwrite=True,
    )
    source_dataset_ids = [source.source_dataset_id for source in resolved_sources]
    _write_string_array(source_index, "source_dataset_id", source_dataset_ids)
    _write_string_array(source_index, "source_zarr_path", source_zarr_paths)
    source_index.attrs.update(
        {
            "mapping_version": 1,
            "source_count": int(len(resolved_sources)),
        }
    )

    summary: Dict[str, Any]
    if validate:
        summary = validate_merged_eye_mask_training_zarr(
            out_path,
            expected_input_format=normalized_input_format,
            expected_total_samples=total_samples,
            expected_label_mode=normalized_label_mode,
        )
    else:
        summary = {
            "zarr_path": str(out_path),
            "run_name": str(run_name),
            "total_samples": int(total_samples),
            "split_counts": {
                "train": int(train_idx.shape[0]),
                "val": int(val_idx.shape[0]),
                "test": int(test_idx.shape[0]),
            },
        }

    registry_summary: Optional[Dict[str, Any]] = None
    if registry_path is not None:
        try:
            registry_summary = _register_merged_dataset_in_registry(
                registry_path=registry_path,
                merged_zarr=out_path,
                source_zarr_paths=[source.source_path for source in resolved_sources],
                set_id=training_set_id,
                set_name=training_set_name,
            )
        except Exception as exc:
            raise RuntimeError(
                "Merged eye-mask export registry registration failed. "
                "Remediation: rerun with a valid --registry path, or omit --registry to skip registration."
            ) from exc
        training_export_payload["registry_registration"] = registry_summary

    workflow_summary: Optional[Dict[str, Any]] = None
    if aggregate_training_data_card:
        workflow_summary = _run_data_card_workflow(
            registry_path=registry_path,
            artifacts=artifacts,
            no_plots=bool(data_card_no_plots),
        )
        training_export_payload["data_card_workflow"] = workflow_summary
    else:
        training_export_payload["data_card_workflow"] = {
            "profile_sync_command": None,
            "aggregate_command": None,
            "plot_command": None,
            "plots_generated": False,
        }

    dst_root.attrs["training_export"] = training_export_payload

    summary.update(
        {
            "source_zarrs": source_zarr_paths,
            "source_dataset_ids": source_dataset_ids,
            "source_eye_stages": [source.selection.eye_stage for source in resolved_sources],
            "source_eye_runs": [source.selection.eye_run for source in resolved_sources],
            "source_crop_runs": [source.selection.crop_run for source in resolved_sources],
            "source_count": int(len(resolved_sources)),
            "row_gate": {
                "requested_policy": normalized_row_gate_policy,
                "applied_policy": applied_row_gate_policy,
                "applied": bool(row_gate_applied),
                "explicit_negative_ratio": float(normalized_explicit_negative_ratio),
                "total_rows": int(row_gate_total_rows),
                "selected_rows": int(row_gate_selected_rows),
                "pair_success_rows": int(row_gate_pair_success_rows),
                "explicit_negative_rows": int(row_gate_explicit_negative_rows),
                "explicit_negative_selected_rows": int(row_gate_explicit_negative_selected_rows),
            },
            "source_row_gate": [
                {
                    "source_dataset_id": source.source_dataset_id,
                    "source_zarr_path": str(source.source_path),
                    **dict(source.row_gate_stats),
                }
                for source in resolved_sources
            ],
            "training_set_id": artifacts.training_set_id,
            "artifact_paths": {
                "data_card_json": str(artifacts.card_json),
                "data_card_plot_dir": str(artifacts.plot_dir),
                "data_card_plot_prefix": str(artifacts.plot_prefix),
            },
            "aggregate_training_data_card": bool(aggregate_training_data_card),
            "data_card_no_plots": bool(data_card_no_plots),
            "registry_registration": registry_summary,
            "data_card_workflow": workflow_summary,
        }
    )
    if len(resolved_sources) == 1:
        summary["source_zarr"] = source_zarr_paths[0]
        summary["source_eye_stage"] = resolved_sources[0].selection.eye_stage
        summary["source_eye_run"] = resolved_sources[0].selection.eye_run
        summary["source_crop_run"] = resolved_sources[0].selection.crop_run
    return summary


def export_merged_eye_mask_training_zarr(
    source_zarr: Path,
    out_zarr: Path,
    *,
    crop_run: Optional[str] = None,
    eye_stage: str = "auto",
    eye_run: Optional[str] = None,
    run_name: str = "merged_export_smoke",
    input_format: str = "gray",
    label_mode: str = "lr",
    split_train: float = 0.8,
    split_val: float = 0.2,
    split_test: float = 0.0,
    split_seed: int = 42,
    row_gate_policy: str = "all_rows",
    explicit_negative_ratio: float = 0.25,
    overwrite: bool = False,
    validate: bool = True,
    registry: Optional[Path] = None,
    training_set_id: Optional[str] = None,
    training_set_name: Optional[str] = None,
    aggregate_training_data_card: bool = False,
    data_card_output: Optional[Path] = None,
    data_card_plot_dir: Optional[Path] = None,
    data_card_plot_prefix: Optional[str] = None,
    data_card_no_plots: bool = False,
) -> Dict[str, Any]:
    source_spec = EyeMergeSourceSpec(
        source_zarr=Path(source_zarr),
        crop_run=crop_run,
        eye_stage=eye_stage,
        eye_run=eye_run,
    )
    return export_merged_eye_mask_training_zarr_from_sources(
        [source_spec],
        out_zarr=out_zarr,
        run_name=run_name,
        input_format=input_format,
        label_mode=label_mode,
        split_train=float(split_train),
        split_val=float(split_val),
        split_test=float(split_test),
        split_seed=int(split_seed),
        row_gate_policy=str(row_gate_policy),
        explicit_negative_ratio=float(explicit_negative_ratio),
        overwrite=bool(overwrite),
        validate=bool(validate),
        registry=registry,
        training_set_id=training_set_id,
        training_set_name=training_set_name,
        aggregate_training_data_card=bool(aggregate_training_data_card),
        data_card_output=data_card_output,
        data_card_plot_dir=data_card_plot_dir,
        data_card_plot_prefix=data_card_plot_prefix,
        data_card_no_plots=bool(data_card_no_plots),
    )


def validate_merged_eye_mask_training_zarr(
    zarr_path: Path,
    *,
    expected_input_format: Optional[str] = None,
    expected_total_samples: Optional[int] = None,
    expected_label_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate merged eye-mask-training Zarr layout and trainer-facing invariants."""
    root = zarr.open_group(str(zarr_path), mode="r")
    errors: List[str] = []

    if str(root.attrs.get("zarr_purpose", "")).strip().lower() != "training":
        errors.append("root attr zarr_purpose must be 'training'.")
    training_task = str(root.attrs.get("training_task", "")).strip().lower()
    if training_task and training_task != "eye_masks":
        errors.append("root attr training_task must be 'eye_masks' when present.")

    for group_name in ("crop_runs", "eye_masks_runs", "splits", "source_index"):
        if group_name not in root:
            errors.append(f"missing group {group_name}.")
    if errors:
        raise ValueError("Merged eye-mask zarr validation failed:\n- " + "\n- ".join(errors))

    crop_parent = root["crop_runs"]
    eye_parent = root["eye_masks_runs"]
    crop_latest = _as_text(crop_parent.attrs.get("latest"))
    eye_latest = _as_text(eye_parent.attrs.get("latest"))
    if not crop_latest or crop_latest not in crop_parent:
        errors.append("crop_runs/latest missing or points to a non-existent run.")
    if not eye_latest or eye_latest not in eye_parent:
        errors.append("eye_masks_runs/latest missing or points to a non-existent run.")
    if errors:
        raise ValueError("Merged eye-mask zarr validation failed:\n- " + "\n- ".join(errors))

    crop = crop_parent[str(crop_latest)]
    eye = eye_parent[str(eye_latest)]

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

    required_eye_arrays = (
        "masks_roi",
        "ellipse_params",
        "ellipse_success",
        "eye_separation",
        "frame_indices",
        "detection_source",
    )
    for name in required_eye_arrays:
        if name not in eye:
            errors.append(f"missing required array eye_masks_runs/{eye_latest}/{name}.")
    if errors:
        raise ValueError("Merged eye-mask zarr validation failed:\n- " + "\n- ".join(errors))

    roi_images = np.asarray(crop["roi_images"][:])
    bbox_norm = np.asarray(crop["bbox_norm_coords"][:])
    crop_bbox = np.asarray(crop["crop_bbox_norm_coords"][:])
    crop_frame_indices = np.asarray(crop["frame_indices"][:])
    crop_detection_source = np.asarray(crop["detection_source"][:])

    masks_roi = np.asarray(eye["masks_roi"][:])
    ellipse_params = np.asarray(eye["ellipse_params"][:], dtype=np.float32)
    ellipse_success = np.asarray(eye["ellipse_success"][:], dtype=bool)
    eye_separation = np.asarray(eye["eye_separation"][:], dtype=np.float32)
    eye_frame_indices = np.asarray(eye["frame_indices"][:])
    eye_detection_source = np.asarray(eye["detection_source"][:])

    if roi_images.ndim < 3:
        errors.append(f"roi_images must have shape (N,H,W) or (N,H,W,C), got {tuple(roi_images.shape)}.")
    total_samples = int(roi_images.shape[0]) if roi_images.ndim >= 1 else 0
    if expected_total_samples is not None and int(expected_total_samples) != total_samples:
        errors.append(f"total sample mismatch ({total_samples} != expected {int(expected_total_samples)}).")

    input_format = _normalize_input_format(expected_input_format)
    if input_format is None:
        export_meta = root.attrs.get("training_export")
        if isinstance(export_meta, dict):
            input_format = _normalize_input_format(export_meta.get("input_format"))
    if input_format == "rgb":
        if roi_images.ndim != 4 or int(roi_images.shape[-1]) != 3:
            errors.append("roi_images must be (N,H,W,3) for rgb input format.")
    if input_format == "gray":
        if roi_images.ndim == 4 and int(roi_images.shape[-1]) == 3:
            errors.append("roi_images appears rgb but expected gray input format.")

    label_mode = _normalize_label_mode(expected_label_mode)
    if label_mode is None:
        export_meta = root.attrs.get("training_export")
        if isinstance(export_meta, dict):
            label_mode = _normalize_label_mode(export_meta.get("label_mode"))

    if bbox_norm.ndim != 2 or int(bbox_norm.shape[1]) != 4:
        errors.append(f"bbox_norm_coords must have shape (N,4), got {tuple(bbox_norm.shape)}.")
    if crop_bbox.ndim != 2 or int(crop_bbox.shape[1]) != 4:
        errors.append(f"crop_bbox_norm_coords must have shape (N,4), got {tuple(crop_bbox.shape)}.")
    if bbox_norm.ndim == 2 and int(bbox_norm.shape[0]) != total_samples:
        errors.append(f"bbox_norm_coords length mismatch ({bbox_norm.shape[0]} != {total_samples}).")
    if crop_bbox.ndim == 2 and int(crop_bbox.shape[0]) != total_samples:
        errors.append(f"crop_bbox_norm_coords length mismatch ({crop_bbox.shape[0]} != {total_samples}).")

    if masks_roi.ndim != 4:
        errors.append(f"masks_roi must have shape (N,C,H,W), got {tuple(masks_roi.shape)}.")
    else:
        if int(masks_roi.shape[0]) != total_samples:
            errors.append(f"masks_roi length mismatch ({masks_roi.shape[0]} != {total_samples}).")
        channels = int(masks_roi.shape[1])
        if channels < 1:
            errors.append("masks_roi must have at least 1 channel.")
        if label_mode == "lr" and channels != 2:
            errors.append(f"label_mode=lr requires masks_roi channel count 2, got {channels}.")
        if label_mode == "union" and channels != 1:
            errors.append(f"label_mode=union requires masks_roi channel count 1, got {channels}.")
        if roi_images.ndim >= 3:
            roi_h = int(roi_images.shape[1])
            roi_w = int(roi_images.shape[2])
            if int(masks_roi.shape[2]) != roi_h or int(masks_roi.shape[3]) != roi_w:
                errors.append(
                    "masks_roi spatial dims do not match roi_images "
                    f"(({masks_roi.shape[2]}, {masks_roi.shape[3]}) != ({roi_h}, {roi_w}))."
                )
        unique_vals = np.unique(masks_roi.astype(np.int8, copy=False))
        invalid_vals = [int(v) for v in unique_vals.tolist() if int(v) not in (0, 1)]
        if invalid_vals:
            errors.append(f"masks_roi contains non-binary values: {sorted(set(invalid_vals))}.")

    if ellipse_params.ndim != 3 or int(ellipse_params.shape[-1]) != 5:
        errors.append(f"ellipse_params must have shape (N,C,5), got {tuple(ellipse_params.shape)}.")
    if ellipse_success.ndim != 2:
        errors.append(f"ellipse_success must have shape (N,C), got {tuple(ellipse_success.shape)}.")
    if ellipse_params.ndim == 3 and int(ellipse_params.shape[0]) != total_samples:
        errors.append(f"ellipse_params length mismatch ({ellipse_params.shape[0]} != {total_samples}).")
    if ellipse_success.ndim == 2 and int(ellipse_success.shape[0]) != total_samples:
        errors.append(f"ellipse_success length mismatch ({ellipse_success.shape[0]} != {total_samples}).")
    if (
        ellipse_params.ndim == 3
        and ellipse_success.ndim == 2
        and tuple(ellipse_params.shape[:2]) != tuple(ellipse_success.shape[:2])
    ):
        errors.append(
            "ellipse_params and ellipse_success channel shapes differ "
            f"({tuple(ellipse_params.shape[:2])} != {tuple(ellipse_success.shape[:2])})."
        )

    if eye_separation.ndim != 1 or int(eye_separation.shape[0]) != total_samples:
        errors.append(f"eye_separation must be 1D length N ({total_samples}), got {tuple(eye_separation.shape)}.")

    for name, arr in (("crop frame_indices", crop_frame_indices), ("eye frame_indices", eye_frame_indices)):
        if arr.ndim != 1:
            errors.append(f"{name} must be 1D, got ndim={arr.ndim}.")
        elif int(arr.shape[0]) != total_samples:
            errors.append(f"{name} length mismatch ({arr.shape[0]} != {total_samples}).")
        elif not np.issubdtype(arr.dtype, np.integer):
            errors.append(f"{name} must be integer dtype, got {arr.dtype}.")
        else:
            expected_local = np.arange(total_samples, dtype=np.int64)
            if not np.array_equal(arr.astype(np.int64, copy=False), expected_local):
                errors.append(f"{name} must be local 0..N-1 indexing.")

    for name, arr in (("crop detection_source", crop_detection_source), ("eye detection_source", eye_detection_source)):
        if arr.ndim != 1:
            errors.append(f"{name} must be 1D, got ndim={arr.ndim}.")
        elif int(arr.shape[0]) != total_samples:
            errors.append(f"{name} length mismatch ({arr.shape[0]} != {total_samples}).")
        elif not np.issubdtype(arr.dtype, np.integer):
            errors.append(f"{name} must be integer dtype, got {arr.dtype}.")
        else:
            unique_codes = np.unique(arr.astype(np.int64, copy=False))
            invalid_codes = [int(code) for code in unique_codes.tolist() if int(code) not in (0, 1)]
            if invalid_codes:
                errors.append(f"{name} contains invalid codes: {sorted(set(invalid_codes))} (expected 0 or 1).")

    if crop_detection_source.ndim == 1 and eye_detection_source.ndim == 1:
        if crop_detection_source.shape == eye_detection_source.shape:
            if not np.array_equal(
                crop_detection_source.astype(np.int64, copy=False),
                eye_detection_source.astype(np.int64, copy=False),
            ):
                errors.append("eye detection_source must match crop detection_source.")

    reason_labels: Optional[np.ndarray] = None
    try:
        reason_labels = read_reason_labels(eye)
    except Exception as exc:
        errors.append(f"eye reason label decode failed: {exc}")
    if reason_labels is not None:
        reason_arr = np.asarray(reason_labels, dtype=object)
        if reason_arr.ndim != 1 or int(reason_arr.shape[0]) != total_samples:
            errors.append(
                f"eye reason labels must be 1D length N ({total_samples}), got {tuple(reason_arr.shape)}."
            )

    if "reason_bytes" in eye:
        reason_bytes = np.asarray(eye["reason_bytes"][:], dtype=np.uint8)
        if reason_bytes.ndim != 2 or int(reason_bytes.shape[0]) != total_samples:
            errors.append(
                "eye reason_bytes must be 2D with first dimension N "
                f"({total_samples}), got {tuple(reason_bytes.shape)}."
            )
        encoding = _as_text(eye.attrs.get("reason_encoding"))
        if encoding is not None and encoding != REASON_BYTES_ENCODING:
            errors.append(
                "eye reason_encoding must be "
                f"'{REASON_BYTES_ENCODING}', got '{encoding}'."
            )
        width_attr = eye.attrs.get("reason_bytes_width")
        if width_attr is not None and reason_bytes.ndim == 2:
            if int(width_attr) != int(reason_bytes.shape[1]):
                errors.append(
                    "eye reason_bytes_width attr does not match reason_bytes shape "
                    f"({width_attr} != {reason_bytes.shape[1]})."
                )
        null_term = eye.attrs.get("reason_bytes_null_terminated")
        if null_term is not None and bool(null_term) is not True:
            errors.append("eye reason_bytes_null_terminated attr must be true when present.")
        fallback = eye.attrs.get("reason_fallback_order")
        if fallback is not None:
            if isinstance(fallback, np.ndarray):
                fallback_list = [str(item) for item in fallback.tolist()]
            elif isinstance(fallback, (list, tuple)):
                fallback_list = [str(item) for item in fallback]
            else:
                fallback_list = [str(fallback)]
            if fallback_list != ["reason_bytes", "reason", "detection_source"]:
                errors.append(
                    "eye reason_fallback_order must be "
                    "['reason_bytes', 'reason', 'detection_source']."
                )

    probs_name_present = None
    for probs_name in ("mask_probs_roi_refined", "mask_probs_roi"):
        if probs_name in eye:
            probs_name_present = probs_name
            probs = np.asarray(eye[probs_name][:], dtype=np.float32)
            if probs.shape != masks_roi.shape:
                errors.append(
                    f"{probs_name} shape must match masks_roi "
                    f"({tuple(probs.shape)} != {tuple(masks_roi.shape)})."
                )
            elif not np.all(np.isfinite(probs)):
                errors.append(f"{probs_name} contains non-finite values.")
            else:
                min_val = float(np.min(probs)) if probs.size else 0.0
                max_val = float(np.max(probs)) if probs.size else 0.0
                if min_val < -1e-6 or max_val > 1.0 + 1e-6:
                    errors.append(f"{probs_name} values must be in [0,1], got min={min_val:.4f}, max={max_val:.4f}.")
            break

    if ellipse_params.ndim == 3 and ellipse_success.ndim == 2 and tuple(ellipse_params.shape[:2]) == tuple(ellipse_success.shape[:2]):
        success_mask = ellipse_success.astype(bool, copy=False)
        major = ellipse_params[:, :, 2]
        minor = ellipse_params[:, :, 3]
        success_major = major[success_mask]
        success_minor = minor[success_mask]
        if success_major.size > 0:
            if not np.all(np.isfinite(success_major)) or not np.all(np.isfinite(success_minor)):
                errors.append("Successful ellipse rows contain non-finite major/minor axes.")
            if np.any(success_major <= 0.0) or np.any(success_minor <= 0.0):
                errors.append("Successful ellipse rows must have positive major/minor axes.")
            if np.any(success_major < success_minor):
                errors.append("Successful ellipse rows must satisfy major >= minor.")

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

    source_count = 0
    if not errors:
        source_dataset_idx = np.asarray(root[src_dataset_idx_path][:])
        source_frame_idx = np.asarray(root[src_frame_idx_path][:])
        source_dataset_id = np.asarray(root[src_dataset_id_path][:])
        source_zarr_path = np.asarray(root[src_zarr_path_path][:])
        source_roi_idx = np.asarray(root["source_index/source_roi_idx"][:]) if "source_index/source_roi_idx" in root else None

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
        source_count = int(source_dataset_id.shape[0]) if source_dataset_id.ndim == 1 else 0

        if source_dataset_idx.ndim == 1 and np.issubdtype(source_dataset_idx.dtype, np.integer):
            source_dataset_idx_i64 = source_dataset_idx.astype(np.int64, copy=False)
            if source_dataset_idx_i64.size > 0 and int(source_dataset_idx_i64.min()) < 0:
                errors.append(f"{src_dataset_idx_path} contains negative indices.")
            if source_count > 0 and source_dataset_idx_i64.size > 0:
                max_idx = int(source_dataset_idx_i64.max())
                if max_idx >= source_count:
                    errors.append(
                        f"{src_dataset_idx_path} has value {max_idx} outside mapping length {source_count}."
                    )
        else:
            errors.append(f"{src_dataset_idx_path} must be integer dtype.")

        if source_frame_idx.ndim == 1 and np.issubdtype(source_frame_idx.dtype, np.integer):
            source_frame_idx_i64 = source_frame_idx.astype(np.int64, copy=False)
            if source_frame_idx_i64.size > 0 and int(source_frame_idx_i64.min()) < 0:
                errors.append(f"{src_frame_idx_path} contains negative indices.")
        else:
            errors.append(f"{src_frame_idx_path} must be integer dtype.")

        if source_roi_idx is not None:
            if source_roi_idx.ndim != 1 or source_roi_idx.shape[0] != total_samples:
                errors.append(
                    f"source_index/source_roi_idx must be 1D length N ({total_samples}), got {source_roi_idx.shape}."
                )
            elif not np.issubdtype(source_roi_idx.dtype, np.integer):
                errors.append(f"source_index/source_roi_idx must be integer dtype, got {source_roi_idx.dtype}.")
            else:
                source_roi_idx_i64 = source_roi_idx.astype(np.int64, copy=False)
                if source_roi_idx_i64.size > 0:
                    min_idx = int(source_roi_idx_i64.min())
                    if min_idx < 0:
                        errors.append(
                            "source_index/source_roi_idx must reference non-negative source-local row indices "
                            f"(min={min_idx})."
                        )

    if errors:
        raise ValueError("Merged eye-mask zarr validation failed:\n- " + "\n- ".join(errors))

    success_eyes = int(ellipse_success.sum())
    successful_roi_pairs = int(np.all(ellipse_success, axis=1).sum()) if ellipse_success.ndim == 2 else 0
    return {
        "zarr_path": str(zarr_path),
        "crop_run": str(crop_latest),
        "eye_run": str(eye_latest),
        "input_format": input_format,
        "label_mode": label_mode,
        "total_samples": int(total_samples),
        "channels": int(masks_roi.shape[1]),
        "success_eyes": success_eyes,
        "successful_roi_pairs": successful_roi_pairs,
        "split_counts": {
            "train": int(train_idx.shape[0]),
            "val": int(val_idx.shape[0]),
            "test": int(test_idx.shape[0]),
        },
        "source_count": int(source_count),
        "mask_probs_array": probs_name_present,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_zarr", type=Path, help="Source training/analysis zarr path.")
    parser.add_argument("out_zarr", type=Path, help="Output merged eye-mask-training zarr path.")
    parser.add_argument("--crop-run", help="Optional crop run override.")
    parser.add_argument(
        "--eye-stage",
        choices=["auto", "eye_masks_runs", "refined_eye_masks_runs"],
        default="auto",
        help="Eye-mask stage selector (default: auto prefers refined).",
    )
    parser.add_argument("--eye-run", help="Optional explicit eye-mask run name.")
    parser.add_argument("--run-name", default="merged_export_smoke", help="Merged run name inside output zarr.")
    parser.add_argument("--input-format", choices=["gray", "rgb"], default="gray")
    parser.add_argument("--label-mode", choices=["lr", "union"], default="lr")
    parser.add_argument("--split-train", type=float, default=0.8)
    parser.add_argument("--split-val", type=float, default=0.2)
    parser.add_argument("--split-test", type=float, default=0.0)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--row-gate-policy",
        choices=list(EYE_ROW_GATE_POLICIES),
        default="all_rows",
        help=(
            "Row inclusion policy for merged eye-mask export. "
            "usable_only keeps pair-success rows; usable_plus_explicit_negatives "
            "adds capped fish_present_no_keypoints negatives."
        ),
    )
    parser.add_argument(
        "--explicit-negative-ratio",
        type=float,
        default=0.25,
        help=(
            "Maximum explicit-negative rows per positive row when "
            "--row-gate-policy=usable_plus_explicit_negatives."
        ),
    )
    parser.add_argument("--registry", type=Path, help="Optional registry SQLite path for merged-export registration.")
    parser.add_argument(
        "--training-set-id",
        type=str,
        help="Optional training set identifier used for registry linkage and data-card aggregation.",
    )
    parser.add_argument(
        "--training-set-name",
        type=str,
        help="Optional training set display name when updating registry linkage.",
    )
    parser.add_argument(
        "--aggregate-training-data-card",
        action="store_true",
        help="Aggregate eye-mask data card after export (requires --training-set-id).",
    )
    parser.add_argument(
        "--no-aggregate-training-data-card",
        action="store_true",
        help="Compatibility flag; no automatic aggregation is enabled by default.",
    )
    parser.add_argument(
        "--data-card-output",
        type=Path,
        help="Deterministic output JSON path for eye-mask data-card aggregation.",
    )
    parser.add_argument(
        "--data-card-no-plots",
        action="store_true",
        help="Skip plot generation for eye-mask data-card aggregation.",
    )
    parser.add_argument(
        "--data-card-plot-dir",
        type=Path,
        help="Deterministic output directory for eye-mask data-card plots.",
    )
    parser.add_argument(
        "--data-card-plot-prefix",
        type=str,
        help="Deterministic filename prefix for eye-mask data-card plot artifacts.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing out_zarr.")
    parser.add_argument("--no-validate", action="store_true", help="Skip post-export validation.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.aggregate_training_data_card and args.no_aggregate_training_data_card:
        parser.error(
            "--aggregate-training-data-card cannot be combined with --no-aggregate-training-data-card."
        )
    should_aggregate = bool(args.aggregate_training_data_card)

    summary = export_merged_eye_mask_training_zarr(
        source_zarr=args.source_zarr,
        out_zarr=args.out_zarr,
        crop_run=args.crop_run,
        eye_stage=args.eye_stage,
        eye_run=args.eye_run,
        run_name=args.run_name,
        input_format=args.input_format,
        label_mode=args.label_mode,
        split_train=float(args.split_train),
        split_val=float(args.split_val),
        split_test=float(args.split_test),
        split_seed=int(args.split_seed),
        row_gate_policy=str(args.row_gate_policy),
        explicit_negative_ratio=float(args.explicit_negative_ratio),
        overwrite=bool(args.overwrite),
        validate=not bool(args.no_validate),
        registry=args.registry,
        training_set_id=args.training_set_id,
        training_set_name=args.training_set_name,
        aggregate_training_data_card=should_aggregate,
        data_card_output=args.data_card_output,
        data_card_plot_dir=args.data_card_plot_dir,
        data_card_plot_prefix=args.data_card_plot_prefix,
        data_card_no_plots=bool(args.data_card_no_plots),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
