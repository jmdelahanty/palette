#!/usr/bin/env python3
"""Query the registry and build a keypoint (pose) training manifest/config."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml
import zarr

from fisheye.registry.db import Registry, RegistryPaths
from fisheye.shared.type_conversions import normalize_attr as _shared_decode_attr
from fisheye.training.config import PoseConfig
from fisheye.shared.system_metadata import build_invocation_record
from fisheye.shared.zarr_metadata import get_downsample_array_path, get_downsample_shape

DEFAULT_KEYPOINT_SOURCE_TYPE = "refined"
ALLOWED_KEYPOINT_CROP_SOURCE_TYPES = frozenset({"refined", "manual"})
MISSING_CONTRACT_LABEL = "<missing>"


def _add_arg(argv: List[str], flag: str, value: Optional[object]) -> None:
    if value is None:
        return
    argv.extend([flag, str(value)])


def _sanitize_name(value: str) -> str:
    cleaned = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_", "."):
            cleaned.append(ch)
        else:
            cleaned.append("_")
    return "".join(cleaned)


def _build_training_set_id(set_name: str, set_version: int) -> str:
    return f"pose_{set_name}_v{set_version:03d}"


def _resolve_dataset_root(task: str) -> Path:
    env_override = os.getenv("PALETTE_TRAINING_DATASETS_ROOT")
    if env_override:
        return Path(env_override).expanduser().resolve()
    nvme_default = Path("/nvme1/training/datasets")
    if nvme_default.exists():
        return nvme_default
    return Path("datasets") / task


def _looks_like_training_artifact_path(zarr_path: Path) -> bool:
    normalized = str(zarr_path).replace("\\", "/").lower()
    stem = zarr_path.stem.lower()
    if "/training/datasets/" in normalized:
        return True
    if stem.endswith("_merged"):
        return True
    return False


def _next_version_from_dataset_root(
    *,
    set_name: str,
    task_prefix: str,
    dataset_root: Path,
) -> int:
    versions: List[int] = []
    name_re = re.compile(rf"^{re.escape(task_prefix)}_{re.escape(set_name)}_v(\d+)$")
    if dataset_root.exists():
        for entry in dataset_root.iterdir():
            match = name_re.match(entry.name)
            if match:
                versions.append(int(match.group(1)))
    return max(versions) + 1 if versions else 1


def _slug_component(value: Optional[str], *, fallback: str, max_len: int = 32) -> str:
    text = "" if value is None else str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    if not text:
        text = fallback
    return text[:max_len]


def _infer_row_value(
    rows: Sequence[Mapping[str, Any]],
    key: str,
    *,
    fallback: str,
    mixed: str,
) -> str:
    values_set: set[str] = set()
    for row in rows:
        try:
            raw = row[key]
        except Exception:
            continue
        if raw is None:
            continue
        text = str(raw).strip()
        if text:
            values_set.add(text)
    values = sorted(values_set)
    if not values:
        return fallback
    if len(values) == 1:
        return values[0]
    return mixed


def _source_type_counts(values: Sequence[Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        text = str(value).strip().lower() if value is not None else ""
        if not text:
            continue
        counts[text] = int(counts.get(text, 0) + 1)
    return dict(sorted(counts.items()))


def _parse_csv_values(values: Optional[Sequence[str]]) -> Tuple[str, ...]:
    if not values:
        return ()
    parsed: List[str] = []
    for value in values:
        for part in str(value).split(","):
            text = part.strip()
            if text:
                parsed.append(text)
    return tuple(dict.fromkeys(parsed))


def _contract_count_value(value: Any) -> str:
    text = _decode_attr(value)
    if text is None:
        return MISSING_CONTRACT_LABEL
    text = str(text).strip()
    return text if text else MISSING_CONTRACT_LABEL


def _contract_counts(rows: Sequence[Mapping[str, Any]], key: str) -> Dict[str, int]:
    return dict(sorted(Counter(_contract_count_value(row.get(key)) for row in rows).items()))


def _query_keypoint_performance_rows(
    registry: Registry,
    *,
    dataset_ids: Sequence[str],
) -> Dict[str, Dict[str, Mapping[str, Any]]]:
    ids = [str(dataset_id) for dataset_id in dict.fromkeys(dataset_ids) if str(dataset_id).strip()]
    rows_by_dataset: Dict[str, Dict[str, Mapping[str, Any]]] = {dataset_id: {} for dataset_id in ids}
    if not ids:
        return rows_by_dataset

    fields = [
        "dataset_id",
        "keypoint_run",
        "keypoint_created_utc",
        "keypoint_method",
        "source_crop_run",
        "source_crop_storage_mode",
        "source_crop_signature",
        "source_crop_revision",
        "source_roi_image_representation",
        "source_roi_pixel_contract_name",
        "source_roi_read_mode",
        "roi_cache_policy",
        "source_roi_cache_used",
        "source_roi_cache_backend",
        "source_roi_live_acceleration_effective",
        "source_roi_live_gpu_chunk_frames",
        "input_mode_requested",
        "input_mode_effective",
        "updated_utc",
    ]
    for start in range(0, len(ids), 500):
        chunk = ids[start : start + 500]
        placeholders = ",".join("?" for _ in chunk)
        sql = (
            f"SELECT {', '.join(fields)} "
            "FROM keypoint_performance "
            f"WHERE dataset_id IN ({placeholders}) "
            "ORDER BY dataset_id, COALESCE(keypoint_created_utc, updated_utc) DESC, keypoint_run DESC"
        )
        for row in registry.conn.execute(sql, chunk):
            dataset_id = str(row["dataset_id"])
            keypoint_run = _decode_attr(row["keypoint_run"])
            if not keypoint_run:
                continue
            rows_by_dataset.setdefault(dataset_id, {})
            rows_by_dataset[dataset_id][str(keypoint_run)] = dict(row)
    return rows_by_dataset


def _missing_keypoint_performance_row(
    *,
    dataset_id: str,
    keypoint_run: str,
) -> Dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "keypoint_run": keypoint_run,
        "keypoint_created_utc": None,
        "keypoint_method": None,
        "source_crop_run": None,
        "source_crop_storage_mode": None,
        "source_crop_signature": None,
        "source_crop_revision": None,
        "source_roi_image_representation": None,
        "source_roi_pixel_contract_name": None,
        "source_roi_read_mode": None,
        "roi_cache_policy": None,
        "source_roi_cache_used": None,
        "source_roi_cache_backend": None,
        "source_roi_live_acceleration_effective": None,
        "source_roi_live_gpu_chunk_frames": None,
        "input_mode_requested": None,
        "input_mode_effective": None,
        "updated_utc": None,
        "registry_row_missing": True,
    }


def _build_keypoint_contract_policy(
    datasets: Sequence[Mapping[str, Any]],
    *,
    compatible_contracts: Sequence[str],
) -> Dict[str, Any]:
    contract_counts = _contract_counts(datasets, "source_roi_pixel_contract_name")
    read_mode_counts = _contract_counts(datasets, "source_roi_read_mode")
    cache_backend_counts = _contract_counts(datasets, "source_roi_cache_backend")
    input_mode_counts = _contract_counts(datasets, "input_mode_effective")
    missing_count = int(contract_counts.get(MISSING_CONTRACT_LABEL, 0))
    explicit_contracts = sorted(
        contract for contract in contract_counts if contract != MISSING_CONTRACT_LABEL
    )
    compatible_set = set(compatible_contracts)

    if missing_count:
        missing_datasets = [
            str(dataset.get("dataset_id") or dataset.get("name") or dataset.get("zarr_path"))
            for dataset in datasets
            if _contract_count_value(dataset.get("source_roi_pixel_contract_name")) == MISSING_CONTRACT_LABEL
        ]
        raise ValueError(
            "Keypoint training selection contains missing keypoint ROI pixel contracts "
            f"({missing_count}/{len(datasets)} dataset(s)): {', '.join(missing_datasets[:10])}. "
            "Refresh keypoint_performance or regenerate the keypoint run with "
            "source_roi_pixel_contract_name before exporting."
        )

    if len(explicit_contracts) > 1:
        if not compatible_set:
            raise ValueError(
                "Mixed keypoint ROI pixel contracts detected: "
                f"{explicit_contracts}. Re-run with --compatible-keypoint-contract for each "
                "contract only after validating that this mix is acceptable for the export."
            )
        if not set(explicit_contracts) <= compatible_set:
            raise ValueError(
                "Mixed keypoint ROI pixel contracts are not covered by the requested compatibility group. "
                f"observed={explicit_contracts}, compatible={sorted(compatible_set)}"
            )
        status = "mixed_explicit_allowed"
    elif len(explicit_contracts) == 1:
        status = "single_contract"
    else:
        status = "empty"

    return {
        "schema_version": "palette.keypoint_roi_pixel_contract_policy.v1",
        "status": status,
        "strict_missing_contracts": True,
        "mixed_contracts_require_explicit_compatibility": True,
        "compatible_contracts": list(compatible_contracts),
        "explicit_contracts": explicit_contracts,
        "required_roi_pixel_contract_name": explicit_contracts[0] if len(explicit_contracts) == 1 else None,
        "contract_counts": contract_counts,
        "read_mode_counts": read_mode_counts,
        "cache_backend_counts": cache_backend_counts,
        "input_mode_counts": input_mode_counts,
    }


def _build_set_name_query_signature(args: argparse.Namespace, *, model_input: str) -> Dict[str, Any]:
    return {
        "dish_design": args.dish_design,
        "dish_design_like": args.dish_design_like,
        "fps_min": args.fps_min,
        "fps_max": args.fps_max,
        "exposure_min": args.exposure_min,
        "exposure_max": args.exposure_max,
        "frame_rate_min": args.frame_rate_min,
        "frame_rate_max": args.frame_rate_max,
        "gain_min": args.gain_min,
        "gain_max": args.gain_max,
        "video_codec": args.video_codec,
        "video_pix_fmt": args.video_pix_fmt,
        "format_encoder": args.format_encoder,
        "format_title": args.format_title,
        "format_comment": args.format_comment,
        "encoder_name": args.encoder_name,
        "encoder_codec": args.encoder_codec,
        "encoder_preset": args.encoder_preset,
        "encoder_tuning": args.encoder_tuning,
        "encoder_rc": args.encoder_rc,
        "compression": args.compression,
        "camera_model": args.camera_model,
        "camera_serial": args.camera_serial,
        "camera_id": args.camera_id,
        "rig_id": args.rig_id,
        "arena_id": args.arena_id,
        "path_contains": args.path_contains,
        "limit": args.limit,
        "source_type": DEFAULT_KEYPOINT_SOURCE_TYPE,
        "input_format": args.input_format,
        "model_input": model_input,
        "keypoint_run": args.keypoint_run,
    }


def _query_hash(args: argparse.Namespace, *, model_input: str) -> str:
    signature = _build_set_name_query_signature(args, model_input=model_input)
    canonical = json.dumps(signature, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:8]


def _default_set_name(
    args: argparse.Namespace,
    rows: Sequence[Mapping[str, Any]],
    *,
    model_input: str,
) -> str:
    if args.dish_design:
        dish_raw = args.dish_design
    elif args.dish_design_like:
        dish_raw = f"like_{args.dish_design_like}"
    else:
        dish_raw = _infer_row_value(rows, "dish_design", fallback="all_dishes", mixed="mixed_dishes")
    canvas_raw = _infer_row_value(rows, "canvas_name", fallback="unknown_canvas", mixed="mixed_canvas")
    query_hash = _query_hash(args, model_input=model_input)
    return "_".join(
        [
            _slug_component(dish_raw, fallback="all_dishes"),
            _slug_component(canvas_raw, fallback="unknown_canvas"),
            _slug_component(DEFAULT_KEYPOINT_SOURCE_TYPE, fallback="source"),
            _slug_component(args.input_format, fallback="input"),
            _slug_component(args.keypoint_run, fallback="keypoints"),
            query_hash,
        ]
    )


def _normalize_input_format(value: str) -> str:
    text = str(value).strip().lower()
    if text not in {"gray", "rgb"}:
        raise ValueError(f"Unsupported input format '{value}'. Expected gray or rgb.")
    return text


_decode_attr = _shared_decode_attr


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
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


def _row_get(row: Mapping[str, Any], key: str, default: Any = None) -> Any:
    try:
        return row[key]
    except Exception:
        return default


def _dataset_id_rank(row: Mapping[str, Any]) -> Tuple[int, str]:
    dataset_id = str(_row_get(row, "dataset_id", "") or "")
    # Prefer canonical dataset IDs over registry collision suffixes when both
    # entries point at the same physical zarr and both pass selection gates.
    suffix_rank = 1 if ":z" in dataset_id else 0
    return suffix_rank, dataset_id


def _resolved_zarr_path_key(row: Mapping[str, Any]) -> str:
    raw_path = Path(str(_row_get(row, "zarr_path", ""))).expanduser()
    try:
        return str(raw_path.resolve(strict=False))
    except Exception:
        return str(raw_path)


def _dedupe_rows_by_zarr_path(
    rows: Sequence[Mapping[str, Any]],
) -> Tuple[List[Mapping[str, Any]], List[Dict[str, str]]]:
    selected_by_path: Dict[str, Mapping[str, Any]] = {}
    order: List[str] = []
    exclusions: List[Dict[str, str]] = []
    for row in rows:
        key = _resolved_zarr_path_key(row)
        current = selected_by_path.get(key)
        if current is None:
            selected_by_path[key] = row
            order.append(key)
            continue
        candidates = sorted((current, row), key=_dataset_id_rank)
        keep = candidates[0]
        drop = candidates[1]
        selected_by_path[key] = keep
        exclusions.append(
            {
                "kept_dataset_id": str(_row_get(keep, "dataset_id", "") or ""),
                "dropped_dataset_id": str(_row_get(drop, "dataset_id", "") or ""),
                "zarr_path": key,
                "reason": "duplicate_zarr_path",
            }
        )
    return [selected_by_path[key] for key in order], exclusions


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
    kp_group: zarr.Group,
    keypoint_count: int,
    fallback_kpt_shape: Optional[Tuple[int, int]],
) -> Tuple[Optional[str], Tuple[int, int]]:
    pose_schema = _as_mapping(kp_group.attrs.get("pose_schema"))
    skeleton_id = _decode_attr(kp_group.attrs.get("skeleton_id"))
    if skeleton_id is None and pose_schema is not None:
        skeleton_id = _decode_attr(pose_schema.get("skeleton_id"))
    if skeleton_id is None and pose_schema is not None:
        pose_schema_name = _decode_attr(pose_schema.get("name"))
        if pose_schema_name:
            skeleton_id = f"pose_schema:{pose_schema_name}"

    runtime_kpt_shape = _normalize_kpt_shape(kp_group.attrs.get("kpt_shape"))
    if runtime_kpt_shape is None and pose_schema is not None:
        runtime_kpt_shape = _normalize_kpt_shape(pose_schema.get("kpt_shape"))
    if runtime_kpt_shape is not None and runtime_kpt_shape[0] != int(keypoint_count):
        raise ValueError(
            "kpt_shape/keypoints_roi mismatch for keypoint run metadata "
            f"(kpt_shape={list(runtime_kpt_shape)} vs keypoints_roi K={keypoint_count})."
        )

    resolved_dims = None
    if fallback_kpt_shape is not None:
        resolved_dims = int(fallback_kpt_shape[1])
    elif runtime_kpt_shape is not None:
        resolved_dims = int(runtime_kpt_shape[1])
    if resolved_dims is None or resolved_dims <= 0:
        resolved_dims = 2
    return skeleton_id, (int(keypoint_count), int(resolved_dims))


def _resolve_effective_annotation_source(
    root: zarr.Group,
    *,
    source_keypoint_run: str,
    source_keypoint_group: Optional[zarr.Group],
    sample_count: int,
    refined_quality: Mapping[str, Any],
) -> Dict[str, Any]:
    source = None
    if source_keypoint_group is not None:
        source = {
            "kind": "raw",
            "parent_name": "keypoints_runs",
            "run_name": str(source_keypoint_run),
            "group": source_keypoint_group,
            "keypoints_path": f"keypoints_runs/{source_keypoint_run}/keypoints_roi",
            "success_path": (
                f"keypoints_runs/{source_keypoint_run}/detection_success"
                if "detection_success" in source_keypoint_group
                else None
            ),
        }

    refined_run_name = _decode_attr(refined_quality.get("refined_keypoint_run"))
    refined_parent = _resolve_refined_parent(root)
    refined_parent_name = _resolve_refined_parent_name(root)
    if not refined_run_name or refined_parent is None or refined_parent_name is None:
        if source is None:
            raise ValueError(
                f"No raw keypoint group or refined keypoint annotation source found for '{source_keypoint_run}'."
            )
        return source
    if refined_run_name not in refined_parent:
        if source is None:
            raise ValueError(
                f"Refined keypoint run '{refined_run_name}' not found and raw keypoint group is absent "
                f"for '{source_keypoint_run}'."
            )
        return source

    refined_group = refined_parent[refined_run_name]
    refined_source_run = _decode_attr(
        refined_group.attrs.get("source_keypoints_run") or refined_group.attrs.get("source_keypoint_run")
    )
    if refined_source_run != source_keypoint_run:
        if source is None:
            raise ValueError(
                f"Refined keypoint run '{refined_run_name}' sources '{refined_source_run}', "
                f"not requested source '{source_keypoint_run}', and raw keypoint group is absent."
            )
        return source
    if "keypoints_roi" not in refined_group or "usable_keypoints" not in refined_group:
        if source is None:
            raise ValueError(
                f"Refined keypoint run '{refined_run_name}' is missing keypoints_roi/usable_keypoints "
                f"and raw keypoint group is absent for '{source_keypoint_run}'."
            )
        return source

    refined_keypoints = refined_group["keypoints_roi"]
    if int(refined_keypoints.shape[0]) != int(sample_count):
        raise ValueError(
            f"Refined keypoints row mismatch for '{refined_parent_name}/{refined_run_name}' "
            f"({refined_keypoints.shape[0]} != {sample_count})."
        )
    usable_arr = refined_group["usable_keypoints"]
    if int(usable_arr.shape[0]) != int(sample_count):
        raise ValueError(
            f"Refined usable_keypoints row mismatch for '{refined_parent_name}/{refined_run_name}' "
            f"({usable_arr.shape[0]} != {sample_count})."
        )

    return {
        "kind": "refined",
        "parent_name": refined_parent_name,
        "run_name": refined_run_name,
        "group": refined_group,
        "keypoints_path": f"{refined_parent_name}/{refined_run_name}/keypoints_roi",
        "success_path": f"{refined_parent_name}/{refined_run_name}/usable_keypoints",
    }


def _resolve_keypoint_labels(group: zarr.Group) -> Optional[List[str]]:
    raw_labels = group.attrs.get("keypoint_labels")
    if isinstance(raw_labels, (list, tuple)):
        labels = [str(item).strip() for item in raw_labels if str(item).strip()]
        if labels:
            return labels
    pose_schema = _as_mapping(group.attrs.get("pose_schema"))
    if pose_schema is not None:
        schema_labels = pose_schema.get("keypoint_labels")
        if isinstance(schema_labels, (list, tuple)):
            labels = [str(item).strip() for item in schema_labels if str(item).strip()]
            if labels:
                return labels
    return None


def _resolve_keypoint_skeleton(group: zarr.Group) -> Optional[List[List[int]]]:
    raw_skeleton = group.attrs.get("keypoint_skeleton")
    edges_source: Any = raw_skeleton
    if not isinstance(edges_source, (list, tuple)):
        pose_schema = _as_mapping(group.attrs.get("pose_schema"))
        if pose_schema is not None:
            edges_source = pose_schema.get("edges")
    if not isinstance(edges_source, (list, tuple)):
        return None

    edges: List[List[int]] = []
    for edge in edges_source:
        if not isinstance(edge, (list, tuple)) or len(edge) < 2:
            continue
        try:
            edges.append([int(edge[0]), int(edge[1])])
        except Exception:
            continue
    return edges or None


def _normalize_keypoint_run_selector(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = value.strip().lower()
    if text in {"", "latest"}:
        return None
    if text in {"latest_traditional", "latest:traditional", "traditional"}:
        return "latest_traditional"
    if text in {"latest_yolo", "latest:yolo", "yolo"}:
        return "latest_yolo"
    return value


def _parse_ts(value: Any) -> datetime:
    text = _decode_attr(value)
    if not text:
        return datetime.min
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        # Keypoint run timestamps are expected in ISO format.
        return datetime.fromisoformat(text)
    except Exception:
        return datetime.min


def _resolve_latest_by_method(parent: zarr.Group, method: Optional[str]) -> Optional[str]:
    candidates: List[Tuple[datetime, str]] = []
    for run_name in parent.group_keys():
        run_group = parent[run_name]
        run_method = _decode_attr(run_group.attrs.get("method"))
        if method and run_method != method:
            continue
        ts = _parse_ts(run_group.attrs.get("keypoints_timestamp_utc") or run_group.attrs.get("timestamp_utc"))
        candidates.append((ts, str(run_name)))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _resolve_keypoint_run(root: zarr.Group, requested: Optional[str]) -> Tuple[str, Optional[str]]:
    parent = root.get("keypoints_runs")
    if parent is None:
        raise ValueError("Missing 'keypoints_runs' group.")

    selector = _normalize_keypoint_run_selector(requested)
    if selector is None:
        latest = _decode_attr(parent.attrs.get("latest"))
        if not latest:
            raise ValueError("keypoints_runs group missing 'latest' attribute; pass --keypoint-run explicitly.")
        if latest not in parent:
            raise ValueError(f"keypoints_runs latest run '{latest}' not found in group keys.")
        return latest, "latest"

    if selector == "latest_traditional":
        run = _resolve_latest_by_method(parent, "traditional_pose")
        if run is None:
            raise ValueError("No keypoint run found for selector 'latest_traditional'.")
        return run, selector
    if selector == "latest_yolo":
        run = _resolve_latest_by_method(parent, "yolo_pose")
        if run is None:
            raise ValueError("No keypoint run found for selector 'latest_yolo'.")
        return run, selector

    if requested not in parent:
        raise ValueError(f"Requested keypoint run '{requested}' not found.")
    return str(requested), selector


def _imgsz_list(value: Any) -> List[int]:
    if isinstance(value, int):
        return [int(value), int(value)]
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return [int(value[0]), int(value[1])]
    raise ValueError(f"Unsupported imgsz value '{value}'. Expected int or [h,w].")


def _resolve_refined_parent(root: zarr.Group) -> Optional[zarr.Group]:
    refined_parent = root.get("refined_keypoints_runs")
    if refined_parent is not None:
        return refined_parent
    legacy_parent = root.get("keypoints_refined_runs")
    if legacy_parent is not None:
        return legacy_parent
    return None


def _resolve_refined_parent_name(root: zarr.Group) -> Optional[str]:
    if root.get("refined_keypoints_runs") is not None:
        return "refined_keypoints_runs"
    if root.get("keypoints_refined_runs") is not None:
        return "keypoints_refined_runs"
    return None


def _load_group_attrs_from_disk(group_path: Path) -> Dict[str, Any]:
    zarr_json = group_path / "zarr.json"
    attrs: Dict[str, Any] = {}
    if zarr_json.exists():
        try:
            data = json.loads(zarr_json.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        attrs_raw = data.get("attributes") if isinstance(data, Mapping) else None
        if isinstance(attrs_raw, Mapping):
            attrs = dict(attrs_raw)

    parent_zarr = group_path.parent / "zarr.json"
    if parent_zarr.exists():
        try:
            parent_data = json.loads(parent_zarr.read_text(encoding="utf-8"))
        except Exception:
            parent_data = {}
        meta = None
        if isinstance(parent_data, Mapping):
            consolidated = parent_data.get("consolidated_metadata")
            if isinstance(consolidated, Mapping):
                meta = consolidated.get("metadata")
        if isinstance(meta, Mapping):
            entry = meta.get(group_path.name)
            if isinstance(entry, Mapping):
                child_attrs = entry.get("attributes")
                if isinstance(child_attrs, Mapping):
                    for key, value in child_attrs.items():
                        attrs.setdefault(str(key), value)
    return attrs


def _canonicalize_mapping_payload(value: Mapping[str, Any]) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    except Exception:
        return str(sorted((str(k), str(v)) for k, v in value.items()))


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


def _resolve_review_status_sources(
    refined_group: zarr.Group,
    *,
    zarr_path: Optional[Path],
    refined_parent_name: Optional[str],
    refined_run_name: str,
) -> Dict[str, Any]:
    attrs_status = _coerce_mapping(refined_group.attrs.get("keypoint_review_status"))

    disk_checked = zarr_path is not None and refined_parent_name is not None
    disk_status: Optional[Dict[str, Any]] = None
    if disk_checked and zarr_path is not None and refined_parent_name is not None:
        on_disk_attrs = _load_group_attrs_from_disk(zarr_path / refined_parent_name / refined_run_name)
        disk_status = _coerce_mapping(on_disk_attrs.get("keypoint_review_status"))

    effective_status = attrs_status if attrs_status is not None else disk_status

    if not disk_checked:
        divergence = "disk_not_checked"
    elif attrs_status is not None and disk_status is not None:
        if _canonicalize_mapping_payload(attrs_status) == _canonicalize_mapping_payload(disk_status):
            divergence = "in_sync"
        else:
            divergence = "conflict"
    elif attrs_status is None and disk_status is not None:
        divergence = "attrs_missing_disk_present"
    elif attrs_status is not None and disk_status is None:
        divergence = "attrs_present_disk_missing"
    else:
        divergence = "absent_both"

    return {
        "attrs": attrs_status,
        "disk": disk_status,
        "effective": effective_status,
        "divergence": divergence,
    }


def _resolve_review_status_with_fallback(
    refined_group: zarr.Group,
    *,
    zarr_path: Optional[Path],
    refined_parent_name: Optional[str],
    refined_run_name: str,
) -> Optional[Dict[str, Any]]:
    sources = _resolve_review_status_sources(
        refined_group,
        zarr_path=zarr_path,
        refined_parent_name=refined_parent_name,
        refined_run_name=refined_run_name,
    )
    effective = sources.get("effective")
    if isinstance(effective, Mapping):
        return dict(effective)
    return None


def _resolve_refined_keypoint_quality(
    root: zarr.Group,
    source_keypoint_run: str,
    *,
    zarr_path: Optional[Path] = None,
) -> Dict[str, Any]:
    refined_parent = _resolve_refined_parent(root)
    if refined_parent is None:
        return {}
    refined_parent_name = _resolve_refined_parent_name(root)

    candidates: List[Tuple[datetime, str]] = []
    for run_name in refined_parent.group_keys():
        run_group = refined_parent[run_name]
        source_run = _decode_attr(
            run_group.attrs.get("source_keypoints_run") or run_group.attrs.get("source_keypoint_run")
        )
        if source_run != source_keypoint_run:
            continue
        ts = _parse_ts(run_group.attrs.get("created_utc") or run_group.attrs.get("timestamp_utc"))
        candidates.append((ts, str(run_name)))

    if not candidates:
        return {}
    candidates.sort(key=lambda item: item[0], reverse=True)
    refined_run_name = candidates[0][1]
    refined_group = refined_parent[refined_run_name]

    review_sources = _resolve_review_status_sources(
        refined_group,
        zarr_path=zarr_path,
        refined_parent_name=refined_parent_name,
        refined_run_name=refined_run_name,
    )
    review_status = review_sources.get("effective")
    if isinstance(review_status, Mapping):
        review_status = dict(review_status)
    else:
        review_status = None
    usable_keypoints_total: Optional[int] = None
    usable_keypoints_rate: Optional[float] = None

    if "usable_keypoints" in refined_group:
        usable_arr = refined_group["usable_keypoints"]
        total_rows = int(usable_arr.shape[0])
        usable_keypoints_total = int(usable_arr[:].sum())
        usable_keypoints_rate = _format_ratio(usable_keypoints_total, total_rows)

    summary_stats = refined_group.attrs.get("summary_statistics")
    if isinstance(summary_stats, Mapping):
        postprocess_stats = summary_stats.get("postprocess")
        if not isinstance(postprocess_stats, Mapping):
            postprocess_stats = None
        for candidate in (postprocess_stats, summary_stats):
            if not isinstance(candidate, Mapping):
                continue
            if usable_keypoints_total is None:
                usable_keypoints_total = _as_int(candidate.get("usable_keypoints"))
            if usable_keypoints_rate is None:
                numerator = _as_int(candidate.get("usable_keypoints"))
                denominator = _as_int(candidate.get("total_rois"))
                if numerator is not None and denominator is not None and denominator > 0:
                    usable_keypoints_rate = float(numerator) / float(denominator)

    return {
        "refined_keypoint_run": refined_run_name,
        "keypoint_review_status": review_status,
        "keypoint_review_status_divergence": str(review_sources.get("divergence")),
        "usable_keypoints_total": usable_keypoints_total,
        "usable_keypoints_rate": usable_keypoints_rate,
    }


def _resolve_reviewed_keypoint_run(
    root: zarr.Group,
    *,
    method_hint: Optional[str],
    required_state: Optional[str],
    required_intended_use: Optional[str],
    zarr_path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    refined_parent = _resolve_refined_parent(root)
    keypoints_parent = root.get("keypoints_runs")
    if refined_parent is None:
        return None
    refined_parent_name = _resolve_refined_parent_name(root)

    candidates: List[Tuple[datetime, str, str, Dict[str, Any]]] = []
    for refined_run_name in refined_parent.group_keys():
        refined_group = refined_parent[refined_run_name]
        source_keypoint_run = _decode_attr(
            refined_group.attrs.get("source_keypoints_run") or refined_group.attrs.get("source_keypoint_run")
        )
        if not source_keypoint_run:
            continue
        if method_hint is not None:
            if keypoints_parent is None or source_keypoint_run not in keypoints_parent:
                continue
            source_method = _decode_attr(keypoints_parent[source_keypoint_run].attrs.get("method"))
            if source_method != method_hint:
                continue
        review_sources = _resolve_review_status_sources(
            refined_group,
            zarr_path=zarr_path,
            refined_parent_name=refined_parent_name,
            refined_run_name=str(refined_run_name),
        )
        review_status = review_sources.get("effective")
        if isinstance(review_status, Mapping):
            review_status = dict(review_status)
        else:
            review_status = None
        if review_status is None:
            continue
        state = _decode_attr(review_status.get("state"))
        intended_use = _decode_attr(review_status.get("intended_use"))
        if required_state is not None and state != required_state:
            continue
        if required_intended_use is not None and intended_use != required_intended_use:
            continue
        ts = _parse_ts(refined_group.attrs.get("created_utc") or refined_group.attrs.get("timestamp_utc"))
        candidates.append((ts, str(refined_run_name), str(source_keypoint_run), review_status))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, refined_run_name, source_keypoint_run, review_status = candidates[0]
    return {
        "refined_keypoint_run": refined_run_name,
        "source_keypoint_run": source_keypoint_run,
        "keypoint_review_status": review_status,
        "keypoint_review_status_divergence": str(review_sources.get("divergence")),
    }


def _build_query_filter_payload(
    args: argparse.Namespace,
    *,
    model_input: str,
    set_name: Optional[str],
    set_version: Optional[int],
    selected_paths: Sequence[Path],
) -> Dict[str, Any]:
    return {
        "tool": "fisheye.utils.prepare_keypoint_training_from_registry",
        "task": "pose",
        "source_type": DEFAULT_KEYPOINT_SOURCE_TYPE,
        "input_format": args.input_format,
        "model_input": model_input,
        "keypoint_run": args.keypoint_run,
        "skeleton_id": args.skeleton_id,
        "min_usable_keypoints_rate": args.min_usable_keypoints_rate,
        "require_review_state": args.require_review_state,
        "require_review_intended_use": args.require_review_intended_use,
        "compatible_keypoint_contracts": list(args.compatible_keypoint_contracts),
        "base_config_path": str(args.base_config),
        "imgsz_override": int(args.imgsz) if args.imgsz is not None else None,
        "set_name": set_name,
        "set_version": set_version,
        "selected_zarr_paths": [str(path) for path in selected_paths],
    }


def _choose_dataset_name(seen: set[str], zarr_path: Path, ordinal: int) -> str:
    base = _sanitize_name(zarr_path.stem) or f"dataset_{ordinal}"
    candidate = base
    suffix = 2
    while candidate in seen:
        candidate = f"{base}_{suffix}"
        suffix += 1
    seen.add(candidate)
    return candidate


def _format_ratio(numerator: Optional[int], denominator: int) -> Optional[float]:
    if numerator is None:
        return None
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _quality_rate_matches(expected: Optional[float], observed: Optional[float], *, tol: float = 1e-9) -> bool:
    if expected is None and observed is None:
        return True
    if expected is None or observed is None:
        return False
    return abs(float(expected) - float(observed)) <= tol


def _print_summary(
    *,
    source_type: str,
    input_format: str,
    imgsz: List[int],
    set_id: Optional[str],
    datasets: Sequence[Dict[str, Any]],
) -> None:
    print("\nKeypoint Training Preflight")
    print("  Task: pose")
    print(f"  Source type: {source_type}")
    print(f"  Input format: {input_format}")
    print(f"  imgsz: {imgsz}")
    if set_id:
        print(f"  Set ID: {set_id}")
    for dataset in datasets:
        print(f"\nDataset: {dataset['name']}")
        print(f"  Zarr: {dataset['zarr_path']}")
        print(f"  Source crop run: {dataset.get('source_crop_run') or 'N/A'}")
        print(f"  Crop source type: {dataset.get('source_type_resolved')}")
        print(f"  ROI pixel contract: {dataset.get('source_roi_pixel_contract_name') or 'missing'}")
        print(f"  ROI read mode: {dataset.get('source_roi_read_mode') or 'missing'}")
        print(f"  Keypoint run: {dataset.get('keypoint_run_resolved')}")
        if dataset.get("refined_keypoint_run"):
            print(f"  Refined keypoint run: {dataset.get('refined_keypoint_run')}")
        print(f"  Keypoint rows: {dataset.get('keypoints_total')}")
        if dataset.get("keypoints_successful") is not None:
            print(f"  Successful keypoints: {dataset['keypoints_successful']}")
        if dataset.get("keypoints_success_rate") is not None:
            print(f"  Success rate: {dataset['keypoints_success_rate']:.3f}")
        if dataset.get("usable_keypoints_total") is not None:
            print(f"  Usable keypoints: {dataset['usable_keypoints_total']}")
        if dataset.get("usable_keypoints_rate") is not None:
            print(f"  Usable keypoint rate: {dataset['usable_keypoints_rate']:.3f}")
        review_status = dataset.get("keypoint_review_status")
        if isinstance(review_status, Mapping) and review_status:
            state = _decode_attr(review_status.get("state")) or "unknown"
            intended_use = _decode_attr(review_status.get("intended_use")) or "unspecified"
            print(f"  Review status: {state} ({intended_use})")
        for warning in dataset.get("warnings", []):
            print(f"  ⚠ {warning}")


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, help="Registry SQLite path.")
    parser.add_argument("--dish-design", type=str, help="Exact dish design match.")
    parser.add_argument("--dish-design-like", type=str, help="Substring match for dish design.")
    parser.add_argument("--fps-min", type=float)
    parser.add_argument("--fps-max", type=float)
    parser.add_argument("--exposure-min", type=float)
    parser.add_argument("--exposure-max", type=float)
    parser.add_argument("--frame-rate-min", type=float)
    parser.add_argument("--frame-rate-max", type=float)
    parser.add_argument("--gain-min", type=float)
    parser.add_argument("--gain-max", type=float)
    parser.add_argument("--video-codec", type=str)
    parser.add_argument("--video-pix-fmt", type=str)
    parser.add_argument("--format-encoder", type=str, help="Exact match on container encoder tag.")
    parser.add_argument("--format-title", type=str, help="Exact match on container title tag.")
    parser.add_argument("--format-comment", type=str, help="Exact match on container comment tag.")
    parser.add_argument("--encoder-name", type=str, help="Exact match on encoder name in comment.")
    parser.add_argument("--encoder-codec", type=str, help="Exact match on encoder codec in comment.")
    parser.add_argument("--encoder-preset", type=str, help="Exact match on encoder preset in comment.")
    parser.add_argument("--encoder-tuning", type=str, help="Exact match on encoder tuning in comment.")
    parser.add_argument("--encoder-rc", type=str, help="Exact match on encoder rate control in comment.")
    parser.add_argument("--compression", type=str)
    parser.add_argument("--camera-model", type=str)
    parser.add_argument("--camera-serial", type=str)
    parser.add_argument("--camera-id", type=str)
    parser.add_argument("--rig-id", type=str)
    parser.add_argument("--arena-id", type=str)
    parser.add_argument("--path-contains", type=str)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-file-list", type=Path, help="Write matched zarr paths to file.")

    parser.add_argument("--input-format", choices=["gray", "rgb"], default="gray")
    parser.add_argument(
        "--model-input",
        choices=["gray", "rgb"],
        help="Registry filter for required training input modality. Defaults to --input-format.",
    )
    parser.add_argument(
        "--keypoint-run",
        type=str,
        default="latest_traditional",
        help="Keypoint run selector (e.g. latest, latest_traditional, latest_yolo, or explicit run name).",
    )
    parser.add_argument(
        "--skeleton-id",
        type=str,
        help=(
            "Require the effective annotation source to resolve to this skeleton_id "
            "(for example pose_skel_traditional_v2)."
        ),
    )
    parser.add_argument("--base-config", type=Path, default=Path("configs/fisheye/pose_config.yaml"))
    parser.add_argument("--out-config", type=Path)
    parser.add_argument("--out-manifest", type=Path)
    parser.add_argument(
        "--set-name",
        type=str,
        help="Training set name. Auto-generated when omitted and --out-config is not provided.",
    )
    parser.add_argument("--set-version", type=int)
    parser.add_argument("--project", type=str)
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument(
        "--min-usable-keypoints-rate",
        type=float,
        help="Require refined usable_keypoints rate >= threshold (0-1).",
    )
    parser.add_argument(
        "--require-review-state",
        choices=["approved", "pending", "rejected", "needs_review"],
        help="Require refined keypoint review status state.",
    )
    parser.add_argument(
        "--require-review-intended-use",
        choices=["training", "full_recording"],
        help="Require refined keypoint review intended_use.",
    )
    parser.add_argument(
        "--allow-cross-method-review-fallback",
        action="store_true",
        help=(
            "When review-gated selection with latest_traditional/latest_yolo finds no reviewed run "
            "for the requested method, allow fallback to a reviewed run from a different method."
        ),
    )
    parser.add_argument(
        "--compatible-keypoint-contract",
        action="append",
        help=(
            "Allow a mixed explicit keypoint ROI pixel-contract export when all selected contracts are "
            "covered by this compatibility group. Repeatable and comma-separated values are accepted. "
            "Single-contract exports do not require this flag."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--register", action="store_true", help="Record training set metadata in the registry.")
    parser.add_argument(
        "--register-registry",
        type=Path,
        help="Registry path to use when --register is set (defaults to --registry).",
    )

    cli_argv = [str(token) for token in (list(argv) if argv is not None else list(sys.argv[1:]))]
    args = parser.parse_args(cli_argv)
    args.input_format = _normalize_input_format(args.input_format)
    args.compatible_keypoint_contracts = _parse_csv_values(args.compatible_keypoint_contract)
    if args.min_usable_keypoints_rate is not None and not (0.0 <= args.min_usable_keypoints_rate <= 1.0):
        raise ValueError("--min-usable-keypoints-rate must be between 0 and 1.")

    model_input = args.model_input or args.input_format
    if args.model_input and args.model_input != args.input_format:
        raise SystemExit("--model-input must match --input-format for keypoint training selection.")

    registry_path = args.registry or RegistryPaths.from_env(Path.cwd()).path
    registry = Registry(registry_path)
    rows = registry.query_datasets(
        dish_design=args.dish_design,
        dish_design_like=args.dish_design_like,
        fps_min=args.fps_min,
        fps_max=args.fps_max,
        exposure_min=args.exposure_min,
        exposure_max=args.exposure_max,
        frame_rate_min=args.frame_rate_min,
        frame_rate_max=args.frame_rate_max,
        gain_min=args.gain_min,
        gain_max=args.gain_max,
        video_codec=args.video_codec,
        video_pix_fmt=args.video_pix_fmt,
        format_encoder=args.format_encoder,
        format_title=args.format_title,
        format_comment=args.format_comment,
        encoder_name=args.encoder_name,
        encoder_codec=args.encoder_codec,
        encoder_preset=args.encoder_preset,
        encoder_tuning=args.encoder_tuning,
        encoder_rc=args.encoder_rc,
        compression_name=args.compression,
        camera_model=args.camera_model,
        camera_serial=args.camera_serial,
        camera_id=args.camera_id,
        rig_id=args.rig_id,
        arena_id=args.arena_id,
        model_input=model_input,
        path_contains=args.path_contains,
        limit=args.limit,
    )

    if not rows:
        registry.close()
        raise SystemExit("Registry query returned no datasets.")

    non_training_rows: List[Mapping[str, Any]] = []
    skipped_training_rows: List[Tuple[str, str, str]] = []
    for row in rows:
        purpose = _decode_attr(row["zarr_use"])
        zarr_path = Path(str(row["zarr_path"]))
        if str(purpose or "").lower() == "training" and _looks_like_training_artifact_path(zarr_path):
            skipped_training_rows.append(
                (
                    str(row["dataset_id"]),
                    str(zarr_path),
                    "zarr_use=training and path looks like merged/training artifact",
                )
            )
            continue
        non_training_rows.append(row)
    rows = non_training_rows
    if skipped_training_rows:
        print(
            f"Skipped {len(skipped_training_rows)} training-artifact dataset(s) "
            "(non-source artifacts) before keypoint selection."
        )
        for dataset_id, zarr_path, reason in skipped_training_rows[:20]:
            print(f"  - {dataset_id} [{reason}] {zarr_path}")
        if len(skipped_training_rows) > 20:
            print(f"  ... {len(skipped_training_rows) - 20} more skip(s) omitted.")
    if not rows:
        registry.close()
        raise SystemExit("No source datasets remain after prefiltering.")

    normalized_selector = _normalize_keypoint_run_selector(args.keypoint_run)
    selector_method_hint: Optional[str] = None
    if normalized_selector == "latest_traditional":
        selector_method_hint = "traditional_pose"
    elif normalized_selector == "latest_yolo":
        selector_method_hint = "yolo_pose"
    selector_is_dynamic = normalized_selector in {None, "latest_traditional", "latest_yolo"}
    quality_gate_active = (
        args.require_review_state is not None
        or args.require_review_intended_use is not None
        or args.min_usable_keypoints_rate is not None
    )
    selected_quality_rows_by_dataset: Dict[str, Mapping[str, Any]] = {}
    quality_exclusions: List[Dict[str, Any]] = []
    skeleton_exclusions: List[Dict[str, Any]] = []
    duplicate_zarr_path_exclusions: List[Dict[str, str]] = []
    if quality_gate_active:
        dataset_ids_all = [str(row["dataset_id"]) for row in rows if row["dataset_id"]]
        strict_method = None
        if selector_is_dynamic and selector_method_hint and not args.allow_cross_method_review_fallback:
            strict_method = selector_method_hint

        selected_quality_rows = registry.query_keypoint_quality_current(
            dataset_ids=dataset_ids_all,
            keypoint_method=strict_method,
            review_state=args.require_review_state,
            review_intended_use=args.require_review_intended_use,
            min_usable_keypoints_rate=args.min_usable_keypoints_rate,
        )
        selected_quality_rows_by_dataset = {
            str(row["dataset_id"]): dict(row) for row in selected_quality_rows if row["dataset_id"]
        }

        all_quality_rows = registry.query_keypoint_quality_current(dataset_ids=dataset_ids_all)
        all_quality_by_dataset: Dict[str, List[Mapping[str, Any]]] = {}
        for quality_row in all_quality_rows:
            dataset_id = str(quality_row["dataset_id"])
            all_quality_by_dataset.setdefault(dataset_id, []).append(dict(quality_row))
        method_quality_by_dataset: Dict[str, List[Mapping[str, Any]]] = {}
        if strict_method is not None:
            method_rows = registry.query_keypoint_quality_current(
                dataset_ids=dataset_ids_all,
                keypoint_method=strict_method,
            )
            for quality_row in method_rows:
                dataset_id = str(quality_row["dataset_id"])
                method_quality_by_dataset.setdefault(dataset_id, []).append(dict(quality_row))

        filtered_rows: List[Mapping[str, Any]] = []
        for row in rows:
            dataset_id = str(row["dataset_id"])
            if dataset_id in selected_quality_rows_by_dataset:
                filtered_rows.append(row)
                continue
            zarr_path = str(row["zarr_path"])
            candidate_rows = all_quality_by_dataset.get(dataset_id, [])
            if not candidate_rows:
                quality_exclusions.append(
                    {
                        "dataset_id": dataset_id,
                        "zarr_path": zarr_path,
                        "reason": "missing_quality_row",
                    }
                )
                continue
            if strict_method is not None and not method_quality_by_dataset.get(dataset_id):
                quality_exclusions.append(
                    {
                        "dataset_id": dataset_id,
                        "zarr_path": zarr_path,
                        "reason": f"no_quality_for_method:{strict_method}",
                    }
                )
                continue
            candidate = (
                method_quality_by_dataset.get(dataset_id, [candidate_rows[0]])[0]
                if strict_method is not None
                else candidate_rows[0]
            )
            review_state = _decode_attr(candidate.get("review_state"))
            review_use = _decode_attr(candidate.get("review_intended_use"))
            usable_rate = _as_float(candidate.get("usable_keypoints_rate"))
            if args.require_review_state is not None and review_state != args.require_review_state:
                quality_exclusions.append(
                    {
                        "dataset_id": dataset_id,
                        "zarr_path": zarr_path,
                        "reason": f"review_state_mismatch:{review_state or 'missing'}!={args.require_review_state}",
                    }
                )
            elif (
                args.require_review_intended_use is not None
                and review_use != args.require_review_intended_use
            ):
                quality_exclusions.append(
                    {
                        "dataset_id": dataset_id,
                        "zarr_path": zarr_path,
                        "reason": f"review_use_mismatch:{review_use or 'missing'}!={args.require_review_intended_use}",
                    }
                )
            elif args.min_usable_keypoints_rate is not None and usable_rate is None:
                quality_exclusions.append(
                    {
                        "dataset_id": dataset_id,
                        "zarr_path": zarr_path,
                        "reason": "missing_usable_keypoints_rate",
                    }
                )
            elif (
                args.min_usable_keypoints_rate is not None
                and usable_rate is not None
                and float(usable_rate) < float(args.min_usable_keypoints_rate)
            ):
                quality_exclusions.append(
                    {
                        "dataset_id": dataset_id,
                        "zarr_path": zarr_path,
                        "reason": f"usable_rate_below_threshold:{usable_rate:.6f}<{args.min_usable_keypoints_rate:.6f}",
                    }
                )
            else:
                quality_exclusions.append(
                    {
                        "dataset_id": dataset_id,
                        "zarr_path": zarr_path,
                        "reason": "excluded_by_quality_filters",
                    }
                )
        rows = filtered_rows

        if quality_exclusions:
            print(f"Keypoint quality SQL filter excluded {len(quality_exclusions)} dataset(s):")
            for exclusion in quality_exclusions[:20]:
                print(f"  - {exclusion['dataset_id']} [{exclusion['reason']}] {exclusion['zarr_path']}")
            if len(quality_exclusions) > 20:
                print(f"  ... {len(quality_exclusions) - 20} more exclusion(s) omitted.")

    rows, duplicate_zarr_path_exclusions = _dedupe_rows_by_zarr_path(rows)
    if duplicate_zarr_path_exclusions:
        print(
            f"Skipped {len(duplicate_zarr_path_exclusions)} duplicate zarr-path dataset row(s) "
            "before keypoint training selection."
        )
        for exclusion in duplicate_zarr_path_exclusions[:20]:
            print(
                "  - kept "
                f"{exclusion['kept_dataset_id']} dropped {exclusion['dropped_dataset_id']} "
                f"[{exclusion['reason']}] {exclusion['zarr_path']}"
            )
        if len(duplicate_zarr_path_exclusions) > 20:
            print(f"  ... {len(duplicate_zarr_path_exclusions) - 20} more duplicate(s) omitted.")

    keypoint_performance_by_dataset = _query_keypoint_performance_rows(
        registry,
        dataset_ids=[str(row["dataset_id"]) for row in rows if row["dataset_id"]],
    )

    registry.close()
    if not rows:
        raise SystemExit("No datasets remain after keypoint quality filtering and zarr-path deduplication.")

    zarr_paths = [Path(row["zarr_path"]) for row in rows]
    resolved_set_name: Optional[str] = None
    set_version: Optional[int] = None
    set_id: Optional[str] = None
    if args.out_config is None:
        if args.set_name:
            resolved_set_name = _sanitize_name(args.set_name)
        else:
            resolved_set_name = _default_set_name(args, rows, model_input=model_input)
            print(f"Auto set-name: {resolved_set_name}")
        dataset_root = _resolve_dataset_root("pose")
        if args.set_version is not None:
            if args.set_version < 1:
                raise ValueError("--set-version must be >= 1")
            set_version = args.set_version
        else:
            set_version = _next_version_from_dataset_root(
                set_name=resolved_set_name,
                task_prefix="pose",
                dataset_root=dataset_root,
            )
        set_id = _build_training_set_id(resolved_set_name, set_version)
        out_dir = dataset_root / set_id
        args.out_config = out_dir / f"{set_id}.yaml"
        if args.out_manifest is None:
            args.out_manifest = out_dir / f"{set_id}.manifest.json"
    else:
        # Preserve set identity metadata when wrapper passes explicit output paths.
        if args.set_name:
            resolved_set_name = _sanitize_name(args.set_name)
        if args.set_version is not None:
            if args.set_version < 1:
                raise ValueError("--set-version must be >= 1")
            set_version = int(args.set_version)
        if resolved_set_name is not None and set_version is not None:
            set_id = _build_training_set_id(resolved_set_name, set_version)

    if args.output_file_list:
        args.output_file_list.parent.mkdir(parents=True, exist_ok=True)
        args.output_file_list.write_text("\n".join(str(path) for path in zarr_paths) + "\n", encoding="utf-8")
        print(f"Wrote {len(zarr_paths)} paths to {args.output_file_list}")
    print(f"Registry query matched {len(zarr_paths)} dataset(s).")

    invocation_payload = build_invocation_record(
        tool="fisheye.utils.prepare_keypoint_training_from_registry",
        args=args,
        argv=cli_argv,
    )

    if not args.base_config.exists():
        raise FileNotFoundError(f"Base config not found: {args.base_config}")
    base_config = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    if not isinstance(base_config, dict):
        raise ValueError(f"Base config is not a mapping: {args.base_config}")
    base_config.setdefault("training_params", {})
    base_config_kpt_shape = _normalize_kpt_shape(base_config.get("kpt_shape"))
    if base_config.get("kpt_shape") is not None and base_config_kpt_shape is None:
        raise ValueError(
            f"Base config has invalid kpt_shape '{base_config.get('kpt_shape')}'. Expected [K,D] with positive ints."
        )

    seen_names: set[str] = set()
    dataset_entries: Dict[str, Dict[str, Any]] = {}
    manifest_datasets: List[Dict[str, Any]] = []
    dataset_ids: List[str] = []
    inferred_keypoint_labels: Optional[List[str]] = None
    inferred_keypoint_skeleton: Optional[List[List[int]]] = None
    selected_skeleton_identity: Optional[Tuple[Optional[str], Tuple[int, int]]] = None
    selected_skeleton_signature: Optional[str] = None
    skeleton_signature_members: Dict[str, List[str]] = {}

    for idx, row in enumerate(rows, start=1):
        zarr_path = Path(row["zarr_path"])
        try:
            root = zarr.open_group(str(zarr_path), mode="r", consolidated=False)
        except TypeError:
            root = zarr.open_group(str(zarr_path), mode="r")
        warnings: List[str] = []
        array_path = get_downsample_array_path(root, format_hint=args.input_format)
        if array_path is None:
            raise ValueError(
                f"{zarr_path.name}: downsample data for input_format '{args.input_format}' is missing."
            )
        ds_shape = get_downsample_shape(root, format_hint=args.input_format)
        if ds_shape is None:
            raise ValueError(f"{zarr_path.name}: unable to resolve downsample shape for '{args.input_format}'.")

        dataset_id_text = str(row["dataset_id"]) if row["dataset_id"] else ""
        quality_row = selected_quality_rows_by_dataset.get(dataset_id_text) if quality_gate_active else None
        method_hint = selector_method_hint
        keypoints_parent = root.get("keypoints_runs")
        keypoint_run_resolved: Optional[str] = None
        keypoint_selector: Optional[str] = None
        if quality_row is not None:
            quality_source_run = _decode_attr(quality_row["source_keypoint_run"])
            if quality_source_run is None:
                raise ValueError(
                    f"{zarr_path.name}: quality row missing source_keypoint_run for dataset_id '{dataset_id_text}'."
                )
            quality_method = _decode_attr(quality_row["keypoint_method"])
            if method_hint is not None and quality_method is not None and quality_method != method_hint:
                warnings.append(
                    f"Selector '{args.keypoint_run}' required method '{method_hint}', "
                    f"quality-selected run method is '{quality_method}' (cross-method fallback)."
                )
            keypoint_run_resolved = quality_source_run
            keypoint_selector = "quality"
        else:
            try:
                keypoint_run_resolved, keypoint_selector = _resolve_keypoint_run(root, args.keypoint_run)
            except ValueError:
                if not (
                    quality_gate_active
                    and selector_is_dynamic
                    and (args.require_review_state is not None or args.require_review_intended_use is not None)
                ):
                    raise
        if keypoint_run_resolved is None and quality_gate_active and selector_is_dynamic and (
            args.require_review_state is not None or args.require_review_intended_use is not None
        ):
            reviewed_choice = _resolve_reviewed_keypoint_run(
                root,
                method_hint=method_hint,
                required_state=args.require_review_state,
                required_intended_use=args.require_review_intended_use,
                zarr_path=zarr_path,
            )
            relaxed_method_hint = False
            if (
                reviewed_choice is None
                and method_hint is not None
                and args.allow_cross_method_review_fallback
            ):
                reviewed_choice = _resolve_reviewed_keypoint_run(
                    root,
                    method_hint=None,
                    required_state=args.require_review_state,
                    required_intended_use=args.require_review_intended_use,
                    zarr_path=zarr_path,
                )
                relaxed_method_hint = reviewed_choice is not None
            if reviewed_choice is None:
                constraint_parts: List[str] = []
                if args.require_review_state is not None:
                    constraint_parts.append(f"state={args.require_review_state!r}")
                if args.require_review_intended_use is not None:
                    constraint_parts.append(f"intended_use={args.require_review_intended_use!r}")
                constraint_text = ", ".join(constraint_parts) or "none"
                fallback_hint = ""
                if method_hint is not None and not args.allow_cross_method_review_fallback:
                    fallback_hint = (
                        " Re-run with --allow-cross-method-review-fallback to allow reviewed "
                        "runs from another keypoint method."
                    )
                raise ValueError(
                    f"{zarr_path.name}: no refined keypoint run satisfies review constraints ({constraint_text}) "
                    f"for selector '{args.keypoint_run}'.{fallback_hint}"
                )
            reviewed_source = str(reviewed_choice["source_keypoint_run"])
            if relaxed_method_hint:
                reviewed_method = (
                    _decode_attr(keypoints_parent[reviewed_source].attrs.get("method"))
                    if keypoints_parent is not None and reviewed_source in keypoints_parent
                    else "unknown"
                )
                warnings.append(
                    f"Selector '{args.keypoint_run}' found no reviewed run with method '{method_hint}'; "
                    f"using reviewed source keypoint run '{reviewed_source}' (method '{reviewed_method}')."
                )
            elif reviewed_source != keypoint_run_resolved:
                warnings.append(
                    f"Selector '{args.keypoint_run}' resolved to '{keypoint_run_resolved}', "
                    f"using reviewed source keypoint run '{reviewed_source}' instead."
                )
            if reviewed_source != keypoint_run_resolved:
                keypoint_run_resolved = reviewed_source
                keypoint_selector = (keypoint_selector or "latest") + "_reviewed"

        if keypoint_run_resolved is None:
            raise ValueError(f"{zarr_path.name}: unable to resolve keypoint run.")

        keypoint_perf_row = dict(
            keypoint_performance_by_dataset.get(dataset_id_text, {}).get(str(keypoint_run_resolved))
            or _missing_keypoint_performance_row(
                dataset_id=dataset_id_text,
                keypoint_run=str(keypoint_run_resolved),
            )
        )

        kp_group: Optional[zarr.Group] = (
            keypoints_parent[keypoint_run_resolved]
            if keypoints_parent is not None and keypoint_run_resolved in keypoints_parent
            else None
        )
        refined_quality = _resolve_refined_keypoint_quality(
            root,
            keypoint_run_resolved,
            zarr_path=zarr_path,
        )

        if kp_group is not None:
            if "keypoints_roi" not in kp_group:
                raise ValueError(
                    f"{zarr_path.name}: keypoint run '{keypoint_run_resolved}' missing keypoints_roi array."
                )
            keypoints_arr = kp_group["keypoints_roi"]
        else:
            refined_run_name = _decode_attr(refined_quality.get("refined_keypoint_run"))
            refined_parent = _resolve_refined_parent(root)
            if refined_parent is None or not refined_run_name or refined_run_name not in refined_parent:
                raise ValueError(
                    f"{zarr_path.name}: keypoint run '{keypoint_run_resolved}' not found in keypoints_runs "
                    "and no matching refined_keypoints_runs annotation source is available."
                )
            refined_group = refined_parent[refined_run_name]
            if "keypoints_roi" not in refined_group:
                raise ValueError(
                    f"{zarr_path.name}: refined keypoint run '{refined_run_name}' missing keypoints_roi array."
                )
            keypoints_arr = refined_group["keypoints_roi"]
            warnings.append(
                f"Using refined-only keypoint annotation source '{refined_run_name}' because "
                f"raw keypoint run '{keypoint_run_resolved}' is not present."
            )

        keypoint_shape = tuple(int(v) for v in keypoints_arr.shape)
        if len(keypoint_shape) < 2:
            raise ValueError(
                f"{zarr_path.name}: keypoints_roi must have at least 2 dims (N,K), got shape={keypoint_shape}."
            )
        keypoints_total = int(keypoint_shape[0])
        keypoint_count = int(keypoint_shape[1])
        crop_parent = root.get("crop_runs")
        if crop_parent is None:
            raise ValueError(f"{zarr_path.name}: missing crop_runs group.")
        source_attrs = kp_group.attrs if kp_group is not None else refined_group.attrs
        source_crop_run = _decode_attr(source_attrs.get("source_crop_run"))
        if not source_crop_run:
            raise ValueError(
                f"{zarr_path.name}: keypoint source '{keypoint_run_resolved}' missing source_crop_run. "
                "Refusing ambiguous fallback to crop_runs/latest."
            )
        if source_crop_run not in crop_parent:
            raise ValueError(
                f"{zarr_path.name}: source crop run '{source_crop_run}' not found in crop_runs for "
                f"keypoint run '{keypoint_run_resolved}'."
            )
        registry_source_crop_run = _decode_attr(keypoint_perf_row.get("source_crop_run"))
        if (
            not bool(keypoint_perf_row.get("registry_row_missing"))
            and registry_source_crop_run
            and registry_source_crop_run != source_crop_run
        ):
            raise ValueError(
                f"{zarr_path.name}: stale keypoint_performance row for keypoint run "
                f"'{keypoint_run_resolved}' (registry source_crop_run={registry_source_crop_run!r}, "
                f"zarr source_crop_run={source_crop_run!r}). Refresh keypoint_performance before export."
            )

        crop_group = crop_parent[source_crop_run]
        if "roi_images" not in crop_group:
            raise ValueError(f"{zarr_path.name}: crop run '{source_crop_run}' missing roi_images array.")
        roi_total = int(crop_group["roi_images"].shape[0])
        if roi_total != keypoints_total:
            raise ValueError(
                f"{zarr_path.name}: roi/keypoint row mismatch "
                f"(roi_images={roi_total}, keypoints_roi={keypoints_total})."
            )

        source_type_resolved = _decode_attr(crop_group.attrs.get("detection_source_type"))
        source_type_resolved = str(source_type_resolved).strip().lower() if source_type_resolved is not None else None
        if source_type_resolved not in ALLOWED_KEYPOINT_CROP_SOURCE_TYPES:
            raise ValueError(
                f"{zarr_path.name}: keypoint training requires crop lineage detection_source_type in "
                f"{sorted(ALLOWED_KEYPOINT_CROP_SOURCE_TYPES)!r}, observed {source_type_resolved or 'missing'!r} "
                f"on crop run '{source_crop_run}'."
            )

        keypoints_successful: Optional[int] = None
        keypoints_success_rate = _as_float(source_attrs.get("success_rate")) if kp_group is not None else None
        keypoints_processed = _as_int(source_attrs.get("keypoints_processed")) if kp_group is not None else None
        if keypoints_success_rate is not None:
            denominator = keypoints_processed if keypoints_processed is not None else keypoints_total
            keypoints_successful = int(round(keypoints_success_rate * float(denominator)))
        elif kp_group is not None and "detection_success" in kp_group:
            success_arr = kp_group["detection_success"]
            if success_arr.shape[0] != keypoints_total:
                raise ValueError(
                    f"{zarr_path.name}: detection_success row mismatch "
                    f"(detection_success={success_arr.shape[0]}, keypoints_roi={keypoints_total})."
                )
            # Reading one boolean vector is cheap and gives an exact success count.
            keypoints_successful = int(success_arr[:].sum())
            keypoints_success_rate = _format_ratio(keypoints_successful, keypoints_total)
        elif kp_group is None:
            for success_name in ("usable_keypoints", "refined_success", "source_success"):
                if success_name not in refined_group:
                    continue
                success_arr = refined_group[success_name]
                if success_arr.shape[0] != keypoints_total:
                    raise ValueError(
                        f"{zarr_path.name}: {success_name} row mismatch "
                        f"({success_name}={success_arr.shape[0]}, keypoints_roi={keypoints_total})."
                    )
                keypoints_successful = int(success_arr[:].sum())
                keypoints_success_rate = _format_ratio(keypoints_successful, keypoints_total)
                break

        annotation_source = _resolve_effective_annotation_source(
            root,
            source_keypoint_run=keypoint_run_resolved,
            source_keypoint_group=kp_group,
            sample_count=keypoints_total,
            refined_quality=refined_quality,
        )
        annotation_group = annotation_source["group"]
        annotation_keypoints_arr = annotation_group["keypoints_roi"]
        annotation_keypoint_shape = tuple(int(v) for v in annotation_keypoints_arr.shape)
        annotation_keypoint_count = int(annotation_keypoint_shape[1])

        dataset_skeleton_id, dataset_kpt_shape = _resolve_dataset_skeleton_identity(
            kp_group=annotation_group,
            keypoint_count=annotation_keypoint_count,
            fallback_kpt_shape=base_config_kpt_shape,
        )
        dataset_signature = _format_skeleton_signature(
            skeleton_id=dataset_skeleton_id,
            kpt_shape=dataset_kpt_shape,
        )
        dataset_label = str(row["dataset_id"] or zarr_path.name)
        if args.skeleton_id is not None and dataset_skeleton_id != args.skeleton_id:
            skeleton_exclusions.append(
                {
                    "dataset_id": dataset_id_text or dataset_label,
                    "zarr_path": str(zarr_path),
                    "reason": f"skeleton_id_mismatch:{dataset_skeleton_id or 'missing'}!={args.skeleton_id}",
                }
            )
            continue
        if inferred_keypoint_labels is None:
            labels = _resolve_keypoint_labels(annotation_group)
            if labels:
                inferred_keypoint_labels = labels
        if inferred_keypoint_skeleton is None:
            edges = _resolve_keypoint_skeleton(annotation_group)
            if edges:
                inferred_keypoint_skeleton = edges
        dataset_member = (
            f"{dataset_label} (zarr={zarr_path.name}, "
            f"annotation_source={annotation_source['parent_name']}/{annotation_source['run_name']})"
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
                "Mixed skeleton identities detected while selecting keypoint training datasets. "
                "Expected one (skeleton_id, kpt_shape) signature but found:\n"
                + "\n".join(detail_lines)
            )

        review_status = refined_quality.get("keypoint_review_status")
        review_state = None
        review_intended_use = None
        if isinstance(review_status, Mapping):
            review_state = _decode_attr(review_status.get("state"))
            review_intended_use = _decode_attr(review_status.get("intended_use"))

        if quality_row is not None:
            expected_refined_run = _decode_attr(quality_row["refined_run"])
            observed_refined_run = _decode_attr(refined_quality.get("refined_keypoint_run"))
            if expected_refined_run != observed_refined_run:
                raise ValueError(
                    f"{zarr_path.name}: stale keypoint_quality row: expected refined_run "
                    f"'{expected_refined_run}', observed '{observed_refined_run}'."
                )
            expected_state = _decode_attr(quality_row.get("review_state"))
            expected_use = _decode_attr(quality_row.get("review_intended_use"))
            if review_state != expected_state or review_intended_use != expected_use:
                raise ValueError(
                    f"{zarr_path.name}: review metadata divergence for refined run '{expected_refined_run}' "
                    f"(registry={expected_state}/{expected_use}, zarr={review_state}/{review_intended_use})."
                )
            expected_usable_total = _as_int(quality_row.get("usable_keypoints"))
            expected_total = _as_int(quality_row.get("total_keypoints"))
            expected_rate = _as_float(quality_row.get("usable_keypoints_rate"))
            observed_usable_total = _as_int(refined_quality.get("usable_keypoints_total"))
            observed_rate = _as_float(refined_quality.get("usable_keypoints_rate"))
            observed_total = keypoints_total
            if expected_usable_total != observed_usable_total:
                raise ValueError(
                    f"{zarr_path.name}: usable_keypoints divergence for refined run '{expected_refined_run}' "
                    f"(registry={expected_usable_total}, zarr={observed_usable_total})."
                )
            if expected_total is not None and expected_total != observed_total:
                raise ValueError(
                    f"{zarr_path.name}: total_keypoints divergence for refined run '{expected_refined_run}' "
                    f"(registry={expected_total}, zarr={observed_total})."
                )
            if not _quality_rate_matches(expected_rate, observed_rate):
                raise ValueError(
                    f"{zarr_path.name}: usable_keypoints_rate divergence for refined run '{expected_refined_run}' "
                    f"(registry={expected_rate}, zarr={observed_rate})."
                )
            expected_mtime_ns = _as_int(quality_row.get("zarr_mtime_ns"))
            if expected_mtime_ns is not None:
                try:
                    actual_mtime_ns = int(zarr_path.stat().st_mtime_ns)
                except Exception:
                    actual_mtime_ns = None
                if actual_mtime_ns is None or actual_mtime_ns != expected_mtime_ns:
                    raise ValueError(
                        f"{zarr_path.name}: keypoint_quality row is stale for filesystem mtime "
                        f"(registry={expected_mtime_ns}, actual={actual_mtime_ns})."
                    )

        if args.require_review_state is not None:
            if review_state is None:
                raise ValueError(
                    f"{zarr_path.name}: require-review-state set to '{args.require_review_state}' "
                    "but no keypoint review status is recorded for this keypoint run."
                )
            if review_state != args.require_review_state:
                raise ValueError(
                    f"{zarr_path.name}: review_state '{review_state}' does not match required "
                    f"'{args.require_review_state}'."
                )
        if args.require_review_intended_use is not None:
            if review_intended_use is None:
                raise ValueError(
                    f"{zarr_path.name}: require-review-intended-use set to "
                    f"'{args.require_review_intended_use}' but no intended_use is recorded."
                )
            if review_intended_use != args.require_review_intended_use:
                raise ValueError(
                    f"{zarr_path.name}: review intended_use '{review_intended_use}' does not match required "
                    f"'{args.require_review_intended_use}'."
                )

        usable_keypoints_total = _as_int(refined_quality.get("usable_keypoints_total"))
        usable_keypoints_rate = _as_float(refined_quality.get("usable_keypoints_rate"))
        if args.min_usable_keypoints_rate is not None:
            if usable_keypoints_rate is None:
                raise ValueError(
                    f"{zarr_path.name}: min-usable-keypoints-rate requested but usable keypoint metadata "
                    "is unavailable (run refine_keypoints/keypoint_review first)."
                )
            if float(usable_keypoints_rate) < float(args.min_usable_keypoints_rate):
                raise ValueError(
                    f"{zarr_path.name}: usable_keypoints_rate {usable_keypoints_rate:.3f} is below threshold "
                    f"{args.min_usable_keypoints_rate:.3f}."
                )

        dataset_name = _choose_dataset_name(seen_names, zarr_path, idx)
        dataset_entries[dataset_name] = {
            "zarr_path": str(zarr_path),
            "source_type": source_type_resolved,
            "input_format": args.input_format,
            "keypoint_run": keypoint_run_resolved,
        }
        manifest_datasets.append(
            {
                "name": dataset_name,
                "zarr_path": str(zarr_path),
                "dataset_id": row["dataset_id"],
                "session_uuid": row["session_uuid"],
                "rig_id": row["rig_id"],
                "dish_design": row["dish_design"],
                "canvas_name": row["canvas_name"],
                "source_type_requested": DEFAULT_KEYPOINT_SOURCE_TYPE,
                "source_type_resolved": source_type_resolved,
                "source_crop_run": source_crop_run,
                "source_crop_storage_mode": _decode_attr(keypoint_perf_row.get("source_crop_storage_mode")),
                "source_crop_signature": _decode_attr(keypoint_perf_row.get("source_crop_signature")),
                "source_crop_revision": _as_int(keypoint_perf_row.get("source_crop_revision")),
                "source_roi_image_representation": _decode_attr(
                    keypoint_perf_row.get("source_roi_image_representation")
                ),
                "source_roi_pixel_contract_name": _decode_attr(
                    keypoint_perf_row.get("source_roi_pixel_contract_name")
                ),
                "required_roi_pixel_contract_name": _decode_attr(
                    keypoint_perf_row.get("source_roi_pixel_contract_name")
                ),
                "source_roi_read_mode": _decode_attr(keypoint_perf_row.get("source_roi_read_mode")),
                "roi_cache_policy": _decode_attr(keypoint_perf_row.get("roi_cache_policy")),
                "source_roi_cache_used": keypoint_perf_row.get("source_roi_cache_used"),
                "source_roi_cache_backend": _decode_attr(keypoint_perf_row.get("source_roi_cache_backend")),
                "source_roi_live_acceleration_effective": _decode_attr(
                    keypoint_perf_row.get("source_roi_live_acceleration_effective")
                ),
                "source_roi_live_gpu_chunk_frames": _as_int(
                    keypoint_perf_row.get("source_roi_live_gpu_chunk_frames")
                ),
                "input_mode_requested": _decode_attr(keypoint_perf_row.get("input_mode_requested")),
                "input_mode_effective": _decode_attr(keypoint_perf_row.get("input_mode_effective")),
                "keypoint_performance_registry_used": not bool(keypoint_perf_row.get("registry_row_missing")),
                "keypoint_performance_registry_updated_utc": _decode_attr(keypoint_perf_row.get("updated_utc")),
                "input_format": args.input_format,
                "images_ds_shape": [int(ds_shape[0]), int(ds_shape[1])],
                "keypoint_run_requested": args.keypoint_run,
                "keypoint_run_selector": keypoint_selector,
                "keypoint_run_resolved": keypoint_run_resolved,
                "quality_registry_used": quality_row is not None,
                "quality_registry_refined_run": (
                    _decode_attr(quality_row.get("refined_run")) if quality_row is not None else None
                ),
                "quality_registry_keypoint_method": (
                    _decode_attr(quality_row.get("keypoint_method")) if quality_row is not None else None
                ),
                "cross_method_fallback_used": bool(
                    quality_row is not None
                    and method_hint is not None
                    and _decode_attr(quality_row.get("keypoint_method")) not in {None, method_hint}
                ),
                "annotation_source_kind": annotation_source["kind"],
                "annotation_source_parent": annotation_source["parent_name"],
                "annotation_source_run": annotation_source["run_name"],
                "refined_keypoint_run": refined_quality.get("refined_keypoint_run"),
                "keypoints_array_path": annotation_source["keypoints_path"],
                "detection_success_path": annotation_source["success_path"],
                "keypoints_total": keypoints_total,
                "keypoints_successful": keypoints_successful,
                "keypoints_success_rate": keypoints_success_rate,
                "skeleton_id": dataset_skeleton_id,
                "kpt_shape": list(dataset_kpt_shape),
                "skeleton_signature": dataset_signature,
                "usable_keypoints_total": usable_keypoints_total,
                "usable_keypoints_rate": usable_keypoints_rate,
                "keypoint_review_status": review_status,
                "warnings": warnings,
            }
        )
        if row["dataset_id"]:
            dataset_ids.append(str(row["dataset_id"]))

    if skeleton_exclusions:
        print(f"Skeleton filter excluded {len(skeleton_exclusions)} dataset(s):")
        for exclusion in skeleton_exclusions[:20]:
            print(f"  - {exclusion['dataset_id']} [{exclusion['reason']}] {exclusion['zarr_path']}")
        if len(skeleton_exclusions) > 20:
            print(f"  ... {len(skeleton_exclusions) - 20} more exclusion(s) omitted.")

    if not manifest_datasets:
        if args.skeleton_id is not None:
            raise SystemExit("No datasets remain after skeleton filtering.")
        raise SystemExit("No datasets remain after keypoint training selection.")

    keypoint_contract_policy = _build_keypoint_contract_policy(
        manifest_datasets,
        compatible_contracts=args.compatible_keypoint_contracts,
    )

    base_config["datasets"] = dataset_entries
    base_config["task"] = "pose"
    if args.imgsz is not None:
        base_config["training_params"]["imgsz"] = int(args.imgsz)
    if args.project:
        base_config["training_params"]["project"] = str(Path(args.project).expanduser().resolve())

    manifest_imgsz = _imgsz_list(base_config["training_params"]["imgsz"])
    pose_schema_kpt_shape = (
        list(selected_skeleton_identity[1])
        if selected_skeleton_identity is not None
        else (
            list(base_config_kpt_shape)
            if base_config_kpt_shape is not None
            else (
                list(base_config.get("kpt_shape"))
                if isinstance(base_config.get("kpt_shape"), (list, tuple))
                else None
            )
        )
    )
    pose_schema_skeleton_id = selected_skeleton_identity[0] if selected_skeleton_identity is not None else None
    pose_schema_signature = selected_skeleton_signature or _format_skeleton_signature(
        skeleton_id=pose_schema_skeleton_id,
        kpt_shape=_normalize_kpt_shape(pose_schema_kpt_shape),
    )
    if pose_schema_kpt_shape is not None:
        base_config["kpt_shape"] = list(pose_schema_kpt_shape)
    pose_schema_payload = {
        "skeleton_id": pose_schema_skeleton_id,
        "kpt_shape": pose_schema_kpt_shape,
        "keypoint_labels": inferred_keypoint_labels,
        "skeleton": inferred_keypoint_skeleton,
        "skeleton_signature": pose_schema_signature,
    }
    PoseConfig.model_validate(base_config)

    query_filter_payload = _build_query_filter_payload(
        args,
        model_input=model_input,
        set_name=resolved_set_name,
        set_version=set_version,
        selected_paths=zarr_paths,
    )
    planned_out_manifest: Optional[Path] = None
    if args.out_config is not None:
        planned_out_manifest = args.out_manifest if args.out_manifest is not None else args.out_config.with_suffix(".manifest.json")

    resolved_manifest_source_type_counts = _source_type_counts(
        [dataset.get("source_type_resolved") for dataset in manifest_datasets]
    )

    manifest_payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "pose",
        "source_type": DEFAULT_KEYPOINT_SOURCE_TYPE,
        "source_type_requested": DEFAULT_KEYPOINT_SOURCE_TYPE,
        "source_type_resolved_counts": resolved_manifest_source_type_counts,
        "input_format": args.input_format,
        "imgsz": manifest_imgsz,
        "datasets": manifest_datasets,
        "required_roi_pixel_contract_name": keypoint_contract_policy.get("required_roi_pixel_contract_name"),
        "keypoint_contract_policy": keypoint_contract_policy,
        "base_config_path": str(args.base_config),
        "output_config_path": str(args.out_config) if args.out_config else None,
        "output_manifest_path": str(planned_out_manifest) if planned_out_manifest else None,
        "project": args.project,
        "run_name": args.run_name,
        "registry_path": str(registry_path),
        "set_name": resolved_set_name,
        "set_version": set_version,
        "set_id": set_id,
        "query_filter": query_filter_payload,
        "pose_schema": pose_schema_payload,
        "quality_exclusions": [
            {
                "dataset_id": exclusion["dataset_id"],
                "zarr_path": exclusion["zarr_path"],
                "reason": exclusion["reason"],
            }
            for exclusion in [*quality_exclusions, *skeleton_exclusions]
        ],
        "duplicate_zarr_path_exclusions": duplicate_zarr_path_exclusions,
        "invocation": invocation_payload,
    }

    _print_summary(
        source_type=DEFAULT_KEYPOINT_SOURCE_TYPE,
        input_format=args.input_format,
        imgsz=manifest_imgsz,
        set_id=set_id,
        datasets=manifest_datasets,
    )

    config_yaml = yaml.safe_dump(base_config, sort_keys=False)
    manifest_json = json.dumps(manifest_payload, indent=2)

    if args.dry_run:
        print("\n--- Generated Config (YAML) ---")
        print(config_yaml.strip())
        print("\n--- Training Manifest (JSON) ---")
        print(manifest_json)
        return 0

    if args.out_config is None:
        raise ValueError("--out-config is required unless --dry-run is set.")
    out_manifest = planned_out_manifest if planned_out_manifest is not None else args.out_config.with_suffix(".manifest.json")
    args.out_config.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.out_config.write_text(config_yaml, encoding="utf-8")
    out_manifest.write_text(manifest_json, encoding="utf-8")
    print(f"\nWrote config: {args.out_config}")
    print(f"Wrote manifest: {out_manifest}")

    if args.register:
        if not set_id:
            print("Note: --register ignored because set_id is only defined in set-based mode (omit --out-config).")
        else:
            register_registry_path = args.register_registry or args.registry or RegistryPaths.from_env(Path.cwd()).path
            register_registry = Registry(register_registry_path)
            skeleton_id = register_registry.upsert_pose_skeleton_spec(
                kpt_shape=pose_schema_payload.get("kpt_shape"),
                keypoint_labels=pose_schema_payload.get("keypoint_labels"),
                edges=pose_schema_payload.get("skeleton"),
                name=resolved_set_name,
            )
            register_registry.upsert_training_set(
                set_id=set_id,
                name=resolved_set_name,
                task_type="pose",
                query_filter=query_filter_payload,
                dataset_ids=dataset_ids,
                skeleton_id=skeleton_id,
                invocation=invocation_payload,
            )
            register_registry.close()
            print(f"Recorded training set: {set_id}")

    cli: List[str] = ["python", "-m", "fisheye.training.train_pose", str(args.out_config)]
    _add_arg(cli, "--manifest", out_manifest)
    _add_arg(cli, "--run-name", args.run_name)
    _add_arg(cli, "--set-id", set_id)
    _add_arg(cli, "--registry", registry_path)
    print("Next: " + " ".join(str(token) for token in cli))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
