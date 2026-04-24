#!/usr/bin/env python3
"""Query the registry and prepare a subject-mask training manifest/config."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import yaml

from fisheye.registry.db import Registry, RegistryPaths
from fisheye.shared.type_conversions import normalize_attr as _shared_as_text
from fisheye.utils.system import build_invocation_record

_as_text = _shared_as_text

EXPORTABLE_STAGE_GROUPS = ("subject_mask_runs", "refined_subject_masks_runs")
TARGET_LABEL_SCHEMA = "subject_v1_union"
TARGET_LABELS = ["subject_body", "eyes_union", "swim_bladder"]
KNOWN_COMPONENTS = {"subject_body", "eyes_union", "eye_left", "eye_right", "swim_bladder"}


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
        raw = row.get(key)
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


def _build_training_set_id(set_name: str, set_version: int) -> str:
    return f"subject_mask_{set_name}_v{set_version:03d}"


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


def _manifest_path_for_config(config_path: Path) -> Path:
    return config_path.with_suffix(".manifest.json")


def _config_path_for_manifest(manifest_path: Path) -> Path:
    name = manifest_path.name
    suffix = ".manifest.json"
    if name.endswith(suffix):
        return manifest_path.with_name(f"{name[:-len(suffix)]}.yaml")
    return manifest_path.with_suffix(".yaml")


def _ensure_suffix(value: str, suffix: str) -> str:
    text = str(value).strip()
    if text.endswith(suffix):
        return text
    return f"{text}{suffix}"


def _parse_required_components(values: Optional[Sequence[str]]) -> List[str]:
    if not values:
        return []
    seen: set[str] = set()
    components: List[str] = []
    for raw in values:
        for token in str(raw).split(","):
            component = token.strip()
            if not component:
                continue
            if component not in KNOWN_COMPONENTS:
                raise SystemExit(
                    f"Unsupported component '{component}'. Expected one of {sorted(KNOWN_COMPONENTS)}."
                )
            if component not in seen:
                components.append(component)
                seen.add(component)
    return components


def _build_query_signature(args: argparse.Namespace, *, model_input: str) -> Dict[str, Any]:
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
        "zarr_use": args.zarr_use,
        "path_contains": args.path_contains,
        "limit": args.limit,
        "input_format": args.input_format,
        "model_input": model_input,
        "subject_run": args.subject_run,
        "crop_run": args.crop_run,
        "subject_mask_method": args.subject_mask_method,
        "source_label_schema": args.source_label_schema,
        "require_components": _parse_required_components(args.require_component),
        "require_review_state": args.require_review_state,
        "require_review_intended_use": args.require_review_intended_use,
        "min_component_mask_rate": args.min_component_mask_rate,
    }


def _query_hash(args: argparse.Namespace, *, model_input: str) -> str:
    signature = _build_query_signature(args, model_input=model_input)
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
    rig_raw = args.rig_id or _infer_row_value(rows, "rig_id", fallback="unknown_rig", mixed="mixed_rigs")
    query_hash = _query_hash(args, model_input=model_input)
    return "_".join(
        [
            _slug_component(dish_raw, fallback="all_dishes"),
            _slug_component(canvas_raw, fallback="unknown_canvas"),
            _slug_component(rig_raw, fallback="unknown_rig"),
            _slug_component(args.input_format, fallback="input"),
            _slug_component(TARGET_LABEL_SCHEMA, fallback="schema"),
            query_hash,
        ]
    )


def _resolve_merged_out_zarr(
    *,
    out_dir: Path,
    set_id: Optional[str],
    set_name: Optional[str],
    out_config: Path,
) -> Path:
    if set_id is not None:
        merged_stem_base = _sanitize_name(set_id)
    elif set_name is not None:
        merged_stem_base = _sanitize_name(set_name)
    else:
        merged_stem_base = _sanitize_name(out_config.stem) or "subject_mask_training"
    merged_stem = _ensure_suffix(merged_stem_base, "_merged")
    return out_dir / "zarr" / f"{merged_stem}.zarr"


def _resolve_merged_dataset_identity(
    *,
    set_id: Optional[str],
    set_name: Optional[str],
    merged_out_zarr: Path,
) -> tuple[str, str]:
    dataset_id_base = _sanitize_name(set_id or set_name or merged_out_zarr.stem) or "subject_mask_training"
    dataset_id = _ensure_suffix(dataset_id_base, "_merged")
    return dataset_id, dataset_id


def _choose_dataset_name(seen: set[str], zarr_path: Path, ordinal: int) -> str:
    base = _sanitize_name(zarr_path.stem) or f"dataset_{ordinal}"
    candidate = base
    suffix = 2
    while candidate in seen:
        candidate = f"{base}_{suffix}"
        suffix += 1
    seen.add(candidate)
    return candidate


def _placeholders(count: int) -> str:
    return ",".join("?" for _ in range(count))


def _query_component_rows(
    registry: Registry,
    *,
    dataset_ids: Sequence[str],
    table: str,
) -> List[Dict[str, Any]]:
    if not dataset_ids:
        return []
    placeholders = _placeholders(len(dataset_ids))
    sql = f"""
        SELECT *
        FROM {table}
        WHERE dataset_id IN ({placeholders})
        ORDER BY dataset_id, stage_group, run_name, component_name;
    """
    return [dict(row) for row in registry.conn.execute(sql, list(dataset_ids)).fetchall()]


def _query_subject_mask_performance_rows(
    registry: Registry,
    *,
    dataset_ids: Sequence[str],
) -> List[Dict[str, Any]]:
    if not dataset_ids:
        return []
    placeholders = _placeholders(len(dataset_ids))
    sql = f"""
        SELECT *
        FROM subject_mask_performance
        WHERE dataset_id IN ({placeholders})
        ORDER BY dataset_id, stage_group, run_name;
    """
    return [dict(row) for row in registry.conn.execute(sql, list(dataset_ids)).fetchall()]


def _row_timestamp(row: Mapping[str, Any]) -> str:
    for key in ("review_timestamp_utc", "run_created_utc", "quality_updated_utc"):
        text = _as_text(row.get(key))
        if text:
            return text
    return ""


@dataclass(frozen=True)
class SubjectSourceSelection:
    stage_group: str
    run_name: str
    crop_run: Optional[str]
    subject_mask_method: Optional[str]
    label_schema_id: Optional[str]
    eye_component_mode: Optional[str]
    total_rois: Optional[int]
    available_components: List[str]
    component_rows: List[Dict[str, Any]]
    latest_components: List[Dict[str, Any]]


def _source_exclusion_reason(
    *,
    component_rows: Sequence[Mapping[str, Any]],
    required_components: Sequence[str],
    require_review_state: Optional[str],
    require_review_intended_use: Optional[str],
    min_component_mask_rate: Optional[float],
    source_label_schema: Optional[str],
) -> Optional[str]:
    if not component_rows:
        return "missing_subject_mask_component_rows"
    label_schema = _as_text(component_rows[0].get("label_schema_id"))
    if source_label_schema is not None and label_schema != source_label_schema:
        return "source_label_schema_mismatch"
    if label_schema not in {"subject_v1_union", "subject_v1_lr"}:
        return "unsupported_source_label_schema"

    by_component = {str(row.get("component_name")): row for row in component_rows}
    available_rows = [
        row for row in component_rows if int(row.get("available") or 0) == 1
    ]
    if not available_rows:
        return "no_available_subject_mask_components"
    for component in required_components:
        row = by_component.get(component)
        if row is None or int(row.get("available") or 0) != 1:
            return f"missing_required_component:{component}"

    gated_rows = available_rows
    for row in gated_rows:
        if require_review_state is not None and _as_text(row.get("review_state")) != require_review_state:
            return "review_state_mismatch"
        if (
            require_review_intended_use is not None
            and _as_text(row.get("review_intended_use")) != require_review_intended_use
        ):
            return "review_intended_use_mismatch"
        if min_component_mask_rate is not None:
            rate = _as_float(row.get("rows_with_component_mask_rate"))
            if rate is None or rate < float(min_component_mask_rate):
                return "component_mask_rate_below_threshold"
    return None


def _select_subject_sources(
    *,
    rows: Sequence[Mapping[str, Any]],
    component_rows: Sequence[Mapping[str, Any]],
    performance_rows: Sequence[Mapping[str, Any]],
    latest_rows: Sequence[Mapping[str, Any]],
    required_components: Sequence[str],
    args: argparse.Namespace,
) -> tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    rows_by_dataset = {str(row["dataset_id"]): row for row in rows}
    performance_by_key = {
        (
            str(row.get("dataset_id")),
            str(row.get("stage_group")),
            str(row.get("run_name")),
        ): dict(row)
        for row in performance_rows
    }
    latest_by_dataset: Dict[str, List[Dict[str, Any]]] = {}
    for row in latest_rows:
        latest_by_dataset.setdefault(str(row.get("dataset_id")), []).append(dict(row))

    by_dataset_run: Dict[tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in component_rows:
        stage_group = _as_text(row.get("stage_group"))
        if stage_group not in EXPORTABLE_STAGE_GROUPS:
            continue
        dataset_id = _as_text(row.get("dataset_id"))
        run_name = _as_text(row.get("run_name"))
        if dataset_id is None or run_name is None:
            continue
        if args.subject_run is not None and run_name != str(args.subject_run):
            continue
        by_dataset_run.setdefault((dataset_id, stage_group, run_name), []).append(dict(row))

    candidates_by_dataset: Dict[str, List[SubjectSourceSelection]] = {}
    exclusions: List[Dict[str, str]] = []
    excluded_dataset_ids: set[str] = set()
    for (dataset_id, stage_group, run_name), rows_for_run in by_dataset_run.items():
        reason = _source_exclusion_reason(
            component_rows=rows_for_run,
            required_components=required_components,
            require_review_state=args.require_review_state,
            require_review_intended_use=args.require_review_intended_use,
            min_component_mask_rate=args.min_component_mask_rate,
            source_label_schema=args.source_label_schema,
        )
        if reason is not None:
            source_row = rows_by_dataset.get(dataset_id, {})
            exclusions.append(
                {
                    "dataset_id": dataset_id,
                    "zarr_path": str(source_row.get("zarr_path") or ""),
                    "stage_group": stage_group,
                    "run_name": run_name,
                    "reason": reason,
                }
            )
            excluded_dataset_ids.add(dataset_id)
            continue
        available_rows = [row for row in rows_for_run if int(row.get("available") or 0) == 1]
        available_components = sorted(str(row["component_name"]) for row in available_rows)
        first = rows_for_run[0]
        performance = performance_by_key.get((dataset_id, stage_group, run_name), {})
        selection = SubjectSourceSelection(
            stage_group=stage_group,
            run_name=run_name,
            crop_run=_as_text(performance.get("source_crop_run")),
            subject_mask_method=_as_text(first.get("subject_mask_method"))
            or _as_text(performance.get("subject_mask_method")),
            label_schema_id=_as_text(first.get("label_schema_id"))
            or _as_text(performance.get("label_schema_id")),
            eye_component_mode=_as_text(first.get("eye_component_mode"))
            or _as_text(performance.get("eye_component_mode")),
            total_rois=_as_int(first.get("total_rois")) or _as_int(performance.get("total_rois")),
            available_components=available_components,
            component_rows=rows_for_run,
            latest_components=latest_by_dataset.get(dataset_id, []),
        )
        candidates_by_dataset.setdefault(dataset_id, []).append(selection)

    selected_payload: List[Dict[str, Any]] = []
    seen_names: set[str] = set()
    for ordinal, row in enumerate(rows, start=1):
        dataset_id = str(row["dataset_id"])
        candidates = candidates_by_dataset.get(dataset_id, [])
        if not candidates:
            if dataset_id not in excluded_dataset_ids:
                exclusions.append(
                    {
                        "dataset_id": dataset_id,
                        "zarr_path": str(row.get("zarr_path") or ""),
                        "run_name": str(args.subject_run or ""),
                        "reason": "no_exportable_subject_mask_run",
                    }
                )
            continue

        latest_available = [
            item
            for item in latest_by_dataset.get(dataset_id, [])
            if int(item.get("available") or 0) == 1
            and _as_text(item.get("stage_group")) in EXPORTABLE_STAGE_GROUPS
        ]
        latest_available_components = {
            str(item.get("component_name"))
            for item in latest_available
            if _as_text(item.get("component_name")) is not None
        }

        def candidate_key(selection: SubjectSourceSelection) -> tuple[int, int, int, int, str, str]:
            required_count = sum(1 for component in required_components if component in selection.available_components)
            latest_match_count = sum(
                1
                for item in latest_available
                if _as_text(item.get("stage_group")) == selection.stage_group
                and _as_text(item.get("run_name")) == selection.run_name
            )
            coherent_latest = (
                bool(latest_available_components)
                and latest_match_count == len(latest_available_components)
            )
            stage_priority = 1 if selection.stage_group == "refined_subject_masks_runs" else 0
            latest_ts = max((_row_timestamp(component) for component in selection.component_rows), default="")
            return (
                int(coherent_latest),
                required_count,
                len(selection.available_components),
                latest_match_count,
                latest_ts,
                f"{stage_priority}:{selection.run_name}",
            )

        selected = sorted(candidates, key=candidate_key, reverse=True)[0]
        source_zarr = Path(str(row["zarr_path"]))
        latest_summary = [
            {
                "component_name": _as_text(latest.get("component_name")),
                "stage_group": _as_text(latest.get("stage_group")),
                "run_name": _as_text(latest.get("run_name")),
                "available": _as_int(latest.get("available")),
                "review_state": _as_text(latest.get("review_state")),
                "review_intended_use": _as_text(latest.get("review_intended_use")),
                "lifecycle_state": _as_text(latest.get("lifecycle_state")),
                "quality_stale": _as_int(latest.get("quality_stale")),
            }
            for latest in selected.latest_components
        ]
        non_exportable_latest = [
            item
            for item in latest_summary
            if item.get("stage_group") is not None and item.get("stage_group") not in EXPORTABLE_STAGE_GROUPS
        ]
        selected_latest_components = [
            item
            for item in latest_summary
            if item.get("stage_group") == selected.stage_group and item.get("run_name") == selected.run_name
        ]
        latest_exportable_runs = sorted(
            {
                (str(item.get("stage_group")), str(item.get("run_name")))
                for item in latest_summary
                if item.get("stage_group") in EXPORTABLE_STAGE_GROUPS and item.get("available") == 1
            }
        )
        canonical_latest_requires_assembly = (
            len(latest_exportable_runs) > 1
            and any(stage_group == "refined_subject_masks_runs" for stage_group, _run_name in latest_exportable_runs)
        )
        selected_payload.append(
            {
                "name": _choose_dataset_name(seen_names, source_zarr, ordinal),
                "dataset_id": dataset_id,
                "session_uuid": _as_text(row.get("session_uuid")),
                "zarr_path": str(source_zarr),
                "source_stage_group": selected.stage_group,
                "source_subject_mask_run": selected.run_name,
                "source_crop_run": args.crop_run or selected.crop_run,
                "source_subject_mask_method": selected.subject_mask_method,
                "label_schema_id": selected.label_schema_id,
                "eye_component_mode": selected.eye_component_mode,
                "total_samples": int(selected.total_rois or 0),
                "available_components": selected.available_components,
                "component_quality": [
                    {
                        "component_name": _as_text(component.get("component_name")),
                        "available": _as_int(component.get("available")),
                        "review_state": _as_text(component.get("review_state")),
                        "review_intended_use": _as_text(component.get("review_intended_use")),
                        "rows_with_component_mask_rate": _as_float(component.get("rows_with_component_mask_rate")),
                        "lifecycle_state": _as_text(component.get("lifecycle_state")),
                    }
                    for component in selected.component_rows
                ],
                "canonical_latest_components": latest_summary,
                "canonical_latest_selected_components": selected_latest_components,
                "canonical_latest_non_exportable_components": non_exportable_latest,
                "canonical_latest_requires_assembly": bool(canonical_latest_requires_assembly),
                "dish_design": _as_text(row.get("dish_design")),
                "canvas_name": _as_text(row.get("canvas_name")),
                "rig_id": _as_text(row.get("rig_id")),
            }
        )
    return selected_payload, exclusions


def _build_query_filter_payload(
    args: argparse.Namespace,
    *,
    model_input: str,
    set_name: Optional[str],
    set_version: Optional[int],
    set_id: Optional[str],
    selected_paths: Sequence[Path],
    required_components: Sequence[str],
) -> Dict[str, Any]:
    return {
        "tool": "fisheye.utils.prepare_subject_mask_training_from_registry",
        "task": "subject_masks",
        "input_format": args.input_format,
        "model_input": model_input,
        "subject_label_schema": TARGET_LABEL_SCHEMA,
        "source_stage_groups": list(EXPORTABLE_STAGE_GROUPS),
        "subject_run": args.subject_run,
        "crop_run": args.crop_run,
        "subject_mask_method": args.subject_mask_method,
        "source_label_schema": args.source_label_schema,
        "require_components": list(required_components),
        "require_review_state": args.require_review_state,
        "require_review_intended_use": args.require_review_intended_use,
        "min_component_mask_rate": args.min_component_mask_rate,
        "split_train": float(args.split_train),
        "split_val": float(args.split_val),
        "split_test": float(args.split_test),
        "split_seed": int(args.split_seed),
        "set_name": set_name,
        "set_version": set_version,
        "set_id": set_id,
        "selected_zarr_paths": [str(path) for path in selected_paths],
    }


def _build_training_config_payload(
    *,
    base_config_path: Path,
    datasets: Sequence[Mapping[str, Any]],
    split_seed: int,
    merged_run_name: str,
) -> Dict[str, Any]:
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")
    loaded = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Base config is not a mapping: {base_config_path}")

    config_payload: Dict[str, Any] = copy.deepcopy(loaded)
    dataset_template: Dict[str, Any] = {}
    existing_datasets = config_payload.get("datasets")
    if isinstance(existing_datasets, dict) and existing_datasets:
        first_value = next(iter(existing_datasets.values()))
        if isinstance(first_value, dict):
            dataset_template = copy.deepcopy(first_value)

    generated_datasets: Dict[str, Any] = {}
    for dataset in datasets:
        dataset_entry = copy.deepcopy(dataset_template)
        dataset_entry["zarr_path"] = str(dataset["out_zarr"])
        dataset_entry["crop_run"] = str(dataset["run_name"])
        dataset_entry["subject_mask_run"] = str(dataset["run_name"])
        generated_datasets[str(dataset["name"])] = dataset_entry
    config_payload["datasets"] = generated_datasets
    config_payload["names"] = list(TARGET_LABELS)
    config_payload["nc"] = len(TARGET_LABELS)
    config_payload["random_seed"] = int(split_seed)

    training_params = config_payload.get("training_params")
    if training_params is None:
        training_params = {}
        config_payload["training_params"] = training_params
    if not isinstance(training_params, dict):
        raise ValueError("Base config training_params must be a mapping.")
    training_params["label_schema_id"] = TARGET_LABEL_SCHEMA
    training_params["crop_run"] = str(merged_run_name)
    training_params["subject_masks_run"] = str(merged_run_name)
    return config_payload


def _print_summary(
    *,
    set_id: Optional[str],
    set_name: Optional[str],
    input_format: str,
    required_components: Sequence[str],
    selected_sources: Sequence[Mapping[str, Any]],
    merged_dataset: Mapping[str, Any],
) -> None:
    print("\nSubject-Mask Training Preflight")
    print("  Task: subject_masks")
    print(f"  Input format: {input_format}")
    print(f"  Target schema: {TARGET_LABEL_SCHEMA}")
    if required_components:
        print(f"  Required components: {', '.join(required_components)}")
    if set_name is not None:
        print(f"  Set name: {set_name}")
    if set_id is not None:
        print(f"  Set ID: {set_id}")
    print(f"  Selected sources: {len(selected_sources)}")
    print(f"  Merged run name: {merged_dataset['run_name']}")
    print(f"  Merged out zarr: {merged_dataset['out_zarr']}")
    for source in selected_sources:
        print(f"\nSource: {source['name']}")
        print(f"  Dataset ID: {source['dataset_id']}")
        print(f"  Zarr: {source['zarr_path']}")
        print(f"  Subject source: {source['source_stage_group']}/{source['source_subject_mask_run']}")
        print(f"  Available components: {', '.join(source['available_components'])}")
        non_exportable = source.get("canonical_latest_non_exportable_components") or []
        if non_exportable:
            components = ", ".join(
                f"{item.get('component_name')}={item.get('stage_group')}/{item.get('run_name')}"
                for item in non_exportable
            )
            print(f"  Canonical latest non-exportable components: {components}")


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
    parser.add_argument("--zarr-use", choices=["analysis", "training"], type=str)
    parser.add_argument("--path-contains", type=str)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-file-list", type=Path, help="Write matched source zarr paths to file.")

    parser.add_argument("--subject-run", type=str, help="Explicit subject_mask_runs/<run> to export.")
    parser.add_argument("--crop-run", type=str, help="Explicit source crop run name.")
    parser.add_argument("--subject-mask-method", type=str, help="Require a subject-mask method.")
    parser.add_argument(
        "--source-label-schema",
        choices=["subject_v1_union", "subject_v1_lr"],
        help="Require source subject_mask_runs label schema.",
    )
    parser.add_argument("--input-format", choices=["gray", "rgb"], default="gray")
    parser.add_argument("--model-input", choices=["gray", "rgb"], help="Registry downsample filter mode.")
    parser.add_argument("--split-train", type=float, default=0.8)
    parser.add_argument("--split-val", type=float, default=0.2)
    parser.add_argument("--split-test", type=float, default=0.0)
    parser.add_argument("--split-seed", type=int, default=123)
    parser.add_argument(
        "--require-component",
        action="append",
        help="Require one or more source components. Repeat or comma-separate values.",
    )
    parser.add_argument(
        "--require-review-state",
        choices=["approved", "pending", "rejected", "needs_review"],
        help="Require every available component in the selected source run to have this review state.",
    )
    parser.add_argument(
        "--require-review-intended-use",
        choices=["training", "full_recording"],
        help="Require every available component in the selected source run to have this intended use.",
    )
    parser.add_argument(
        "--min-component-mask-rate",
        type=float,
        help="Require every available component mask-present rate to meet this threshold.",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/fisheye/subject_mask_union_canary_20260406.yaml"),
        help="Base subject-mask training config used to generate the output YAML.",
    )
    parser.add_argument("--out-config", type=Path, help="Output training config YAML path.")
    parser.add_argument("--out-dir", type=Path, help="Output directory root for merged zarr exports.")
    parser.add_argument("--out-manifest", type=Path, help="Output manifest JSON path.")
    parser.add_argument("--out-commands", type=Path, help="Write next commands to this file.")
    parser.add_argument("--set-name", type=str, help="Training set name.")
    parser.add_argument("--set-version", type=int)
    parser.add_argument("--run-name-prefix", type=str, default="merged_subject_masks")
    parser.add_argument("--dry-run", action="store_true")

    cli_argv = [str(token) for token in (list(argv) if argv is not None else list(sys.argv[1:]))]
    args = parser.parse_args(cli_argv)
    required_components = _parse_required_components(args.require_component)
    if args.min_component_mask_rate is not None and not (0.0 <= args.min_component_mask_rate <= 1.0):
        raise ValueError("--min-component-mask-rate must be between 0 and 1.")
    if float(args.split_train) < 0.0 or float(args.split_val) < 0.0 or float(args.split_test) < 0.0:
        raise ValueError("Split ratios must be non-negative.")
    if (float(args.split_train) + float(args.split_val) + float(args.split_test)) <= 0.0:
        raise ValueError("At least one split ratio must be greater than zero.")

    model_input = args.model_input or args.input_format
    if args.model_input and args.model_input != args.input_format:
        raise SystemExit("--model-input must match --input-format for subject-mask training selection.")

    registry_path = args.registry or RegistryPaths.from_env(Path.cwd()).path
    registry = Registry(registry_path)
    rows_raw = registry.query_datasets(
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
        zarr_use=args.zarr_use,
        model_input=model_input,
        path_contains=args.path_contains,
        limit=args.limit,
    )
    rows: List[Mapping[str, Any]] = [dict(row) for row in rows_raw]
    if not rows:
        registry.close()
        raise SystemExit("Registry query returned no datasets.")

    source_rows: List[Mapping[str, Any]] = []
    skipped_training_rows: List[tuple[str, str, str]] = []
    for row in rows:
        purpose = _as_text(row.get("zarr_use"))
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
        source_rows.append(row)
    rows = source_rows
    if not rows:
        registry.close()
        raise SystemExit("No source datasets remain after prefiltering.")

    dataset_ids = [str(row["dataset_id"]) for row in rows if row.get("dataset_id")]
    component_rows = _query_component_rows(
        registry,
        dataset_ids=dataset_ids,
        table="subject_mask_component_quality_overview",
    )
    performance_rows = _query_subject_mask_performance_rows(
        registry,
        dataset_ids=dataset_ids,
    )
    if args.subject_mask_method is not None:
        component_rows = [
            row for row in component_rows if _as_text(row.get("subject_mask_method")) == args.subject_mask_method
        ]
    latest_rows = _query_component_rows(
        registry,
        dataset_ids=dataset_ids,
        table="subject_mask_component_quality_latest",
    )
    registry.close()

    selected_sources_payload, quality_exclusions = _select_subject_sources(
        rows=rows,
        component_rows=component_rows,
        performance_rows=performance_rows,
        latest_rows=latest_rows,
        required_components=required_components,
        args=args,
    )
    if not selected_sources_payload:
        raise SystemExit("No exportable subject-mask sources remain after component filtering.")

    resolved_set_name: Optional[str] = None
    set_version: Optional[int] = None
    set_id: Optional[str] = None
    if args.out_manifest is None and args.out_config is None and args.out_dir is None:
        if args.set_name:
            resolved_set_name = _sanitize_name(args.set_name)
        else:
            resolved_set_name = _default_set_name(args, rows, model_input=model_input)
            print(f"Auto set-name: {resolved_set_name}")
        dataset_root = _resolve_dataset_root("subject_masks")
        if args.set_version is not None:
            if int(args.set_version) < 1:
                raise ValueError("--set-version must be >= 1")
            set_version = int(args.set_version)
        else:
            set_version = _next_version_from_dataset_root(
                set_name=resolved_set_name,
                task_prefix="subject_mask",
                dataset_root=dataset_root,
            )
        set_id = _build_training_set_id(resolved_set_name, set_version)
        out_dir = args.out_dir if args.out_dir is not None else (dataset_root / set_id)
        out_config = out_dir / f"{set_id}.yaml"
        out_manifest = out_dir / f"{set_id}.manifest.json"
    else:
        if args.set_name:
            resolved_set_name = _sanitize_name(args.set_name)
        if args.set_version is not None:
            if int(args.set_version) < 1:
                raise ValueError("--set-version must be >= 1")
            set_version = int(args.set_version)
        if resolved_set_name is not None and set_version is not None:
            set_id = _build_training_set_id(resolved_set_name, set_version)
        if args.out_dir is not None:
            out_dir = args.out_dir
        elif set_id is not None:
            out_dir = _resolve_dataset_root("subject_masks") / set_id
        elif args.out_config is not None:
            out_dir = args.out_config.parent
        elif args.out_manifest is not None:
            out_dir = args.out_manifest.parent
        else:
            out_dir = _resolve_dataset_root("subject_masks")
        if args.out_config is not None:
            out_config = args.out_config
        elif args.out_manifest is not None:
            out_config = _config_path_for_manifest(args.out_manifest)
        elif set_id is not None:
            out_config = out_dir / f"{set_id}.yaml"
        else:
            out_config = out_dir / "subject_mask_training.yaml"
        out_manifest = args.out_manifest if args.out_manifest is not None else _manifest_path_for_config(out_config)

    zarr_paths = [Path(str(source["zarr_path"])) for source in selected_sources_payload]
    if args.output_file_list:
        args.output_file_list.parent.mkdir(parents=True, exist_ok=True)
        args.output_file_list.write_text("\n".join(str(path) for path in zarr_paths) + "\n", encoding="utf-8")
        print(f"Wrote {len(zarr_paths)} paths to {args.output_file_list}")
    print(f"Registry query matched {len(rows)} dataset(s); selected {len(selected_sources_payload)} source(s).")
    if skipped_training_rows:
        print(f"Skipped {len(skipped_training_rows)} training-artifact dataset(s).")

    merged_out_zarr = _resolve_merged_out_zarr(
        out_dir=out_dir,
        set_id=set_id,
        set_name=resolved_set_name,
        out_config=out_config,
    )
    merged_dataset_id, merged_dataset_name = _resolve_merged_dataset_identity(
        set_id=set_id,
        set_name=resolved_set_name,
        merged_out_zarr=merged_out_zarr,
    )
    merged_run_name = _sanitize_name(args.run_name_prefix) or "merged_subject_masks"

    command: List[str] = [
        "scripts/py",
        "-m",
        "fisheye.utils.run_subject_mask_training_pipeline",
        "--config",
        str(out_config),
        "--manifest",
        str(out_manifest),
        "--export-merged",
        "--merge-out-zarr",
        str(merged_out_zarr),
        "--merge-run-name",
        merged_run_name,
        "--subject-label-schema",
        TARGET_LABEL_SCHEMA,
        "--merge-split",
        f"{float(args.split_train)}/{float(args.split_val)}/{float(args.split_test)}",
        "--merge-seed",
        str(int(args.split_seed)),
        "--input-format",
        str(args.input_format),
    ]
    _add_arg(command, "--registry", registry_path)
    _add_arg(command, "--training-set-id", set_id)
    if resolved_set_name is not None and set_id is not None:
        _add_arg(command, "--training-set-name", resolved_set_name)
    command_lines = [" ".join(command)]

    total_selected_samples = sum(int(source.get("total_samples") or 0) for source in selected_sources_payload)
    selected_stage_groups = sorted(
        {
            str(source.get("source_stage_group"))
            for source in selected_sources_payload
            if source.get("source_stage_group")
        }
    )
    manifest_source_stage_group = selected_stage_groups[0] if len(selected_stage_groups) == 1 else "mixed"
    merged_dataset_payload: Dict[str, Any] = {
        "name": merged_dataset_name,
        "run_name": merged_run_name,
        "dataset_id": merged_dataset_id,
        "out_zarr": str(merged_out_zarr),
        "zarr_path": str(merged_out_zarr),
        "source_count": int(len(selected_sources_payload)),
        "total_samples": int(total_selected_samples),
        "input_format": args.input_format,
        "subject_label_schema": TARGET_LABEL_SCHEMA,
        "split": {
            "train": float(args.split_train),
            "val": float(args.split_val),
            "test": float(args.split_test),
            "seed": int(args.split_seed),
        },
        "export_command": command_lines[0],
        "export_status": "planned",
    }

    invocation_payload = build_invocation_record(
        tool="fisheye.utils.prepare_subject_mask_training_from_registry",
        args=args,
        argv=cli_argv,
    )
    query_filter_payload = _build_query_filter_payload(
        args,
        model_input=model_input,
        set_name=resolved_set_name,
        set_version=set_version,
        set_id=set_id,
        selected_paths=zarr_paths,
        required_components=required_components,
    )
    config_payload = _build_training_config_payload(
        base_config_path=args.base_config,
        datasets=[merged_dataset_payload],
        split_seed=int(args.split_seed),
        merged_run_name=merged_run_name,
    )
    merged_export_payload: Dict[str, Any] = {
        "dataset_name": merged_dataset_name,
        "dataset_id": merged_dataset_id,
        "run_name": merged_run_name,
        "zarr_path": str(merged_out_zarr),
        "source_count": int(len(selected_sources_payload)),
        "total_samples": int(total_selected_samples),
        "subject_label_schema": TARGET_LABEL_SCHEMA,
        "source_datasets": selected_sources_payload,
        "command": command_lines[0],
        "export_status": "planned",
    }
    manifest_payload: Dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "subject_masks",
        "registry_path": str(registry_path),
        "set_name": resolved_set_name,
        "set_version": set_version,
        "set_id": set_id,
        "input_format": args.input_format,
        "subject_label_schema": TARGET_LABEL_SCHEMA,
        "source_stage_group": manifest_source_stage_group,
        "source_stage_groups": selected_stage_groups,
        "subject_run": args.subject_run,
        "crop_run": args.crop_run,
        "base_config_path": str(args.base_config),
        "output_config_path": str(out_config),
        "output_root": str(out_dir),
        "output_manifest_path": str(out_manifest),
        "query_filter": query_filter_payload,
        "quality_exclusions": quality_exclusions,
        "selected_sources": selected_sources_payload,
        "datasets": [merged_dataset_payload],
        "merged_export": merged_export_payload,
        "commands": command_lines,
        "invocation": invocation_payload,
        "execution": {
            "mode": "dry-run" if args.dry_run else "planned",
            "planned": 1,
            "succeeded": 0,
            "failed": 0,
        },
    }

    _print_summary(
        set_id=set_id,
        set_name=resolved_set_name,
        input_format=args.input_format,
        required_components=required_components,
        selected_sources=selected_sources_payload,
        merged_dataset=merged_dataset_payload,
    )
    config_yaml = yaml.safe_dump(config_payload, sort_keys=False)
    manifest_json = json.dumps(manifest_payload, indent=2)

    if args.dry_run:
        print("\n--- Generated Config (YAML) ---")
        print(config_yaml.strip())
        print("\n--- Training Manifest (JSON) ---")
        print(manifest_json)
        print("\n--- Next Commands ---")
        for line in command_lines:
            print(line)
        return 0

    out_config.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_config.write_text(config_yaml, encoding="utf-8")
    out_manifest.write_text(manifest_json, encoding="utf-8")
    print(f"\nWrote config: {out_config}")
    print(f"Wrote manifest: {out_manifest}")

    if args.out_commands is not None:
        args.out_commands.parent.mkdir(parents=True, exist_ok=True)
        args.out_commands.write_text("\n".join(command_lines) + "\n", encoding="utf-8")
        print(f"Wrote commands: {args.out_commands}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
