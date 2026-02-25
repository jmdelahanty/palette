#!/usr/bin/env python3
"""Query the registry and prepare/export eye-mask training datasets."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import zarr
import yaml

from fisheye.registry.db import Registry, RegistryPaths
from fisheye.utils import export_eye_mask_training_zarr as export_eye
from fisheye.utils.system import build_invocation_record


def _as_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        text = value.decode("utf-8", "ignore").strip()
    else:
        text = str(value).strip()
    return text or None


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _parse_iso_ts(value: Any) -> datetime:
    text = _as_text(value)
    if text is None:
        return datetime.min.replace(tzinfo=timezone.utc)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except Exception:
        return datetime.min.replace(tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _add_arg(argv: List[str], flag: str, value: Optional[object]) -> None:
    if value is None:
        return
    argv.extend([flag, str(value)])


def _manifest_path_for_config(config_path: Path) -> Path:
    return config_path.with_suffix(".manifest.json")


def _config_path_for_manifest(manifest_path: Path) -> Path:
    name = manifest_path.name
    suffix = ".manifest.json"
    if name.endswith(suffix):
        return manifest_path.with_name(f"{name[:-len(suffix)]}.yaml")
    return manifest_path.with_suffix(".yaml")


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


def _resolve_dataset_root(task: str) -> Path:
    env_override = os.getenv("PALETTE_TRAINING_DATASETS_ROOT")
    if env_override:
        return Path(env_override).expanduser().resolve()
    nvme_default = Path("/nvme1/training/datasets")
    if nvme_default.exists():
        return nvme_default
    return Path("datasets") / task


def _build_training_set_id(set_name: str, set_version: int) -> str:
    return f"eye_mask_{set_name}_v{set_version:03d}"


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


def _looks_like_training_artifact_path(zarr_path: Path) -> bool:
    normalized = str(zarr_path).replace("\\", "/").lower()
    stem = zarr_path.stem.lower()
    if "/training/datasets/" in normalized:
        return True
    if stem.endswith("_merged"):
        return True
    return False


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
        "eye_stage": args.eye_stage,
        "eye_run": args.eye_run,
        "crop_run": args.crop_run,
        "input_format": args.input_format,
        "model_input": model_input,
        "label_mode": args.label_mode,
        "profile_stage_group": args.profile_stage_group,
        "eye_mask_method": args.eye_mask_method,
        "min_usable_rate": args.min_usable_rate,
        "require_review_state": args.require_review_state,
        "require_review_intended_use": args.require_review_intended_use,
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
    if args.rig_id:
        rig_raw = args.rig_id
    else:
        rig_raw = _infer_row_value(rows, "rig_id", fallback="unknown_rig", mixed="mixed_rigs")
    query_hash = _query_hash(args, model_input=model_input)
    return "_".join(
        [
            _slug_component(dish_raw, fallback="all_dishes"),
            _slug_component(canvas_raw, fallback="unknown_canvas"),
            _slug_component(rig_raw, fallback="unknown_rig"),
            _slug_component(args.eye_stage, fallback="eye_masks"),
            _slug_component(args.input_format, fallback="input"),
            _slug_component(args.label_mode, fallback="labels"),
            query_hash,
        ]
    )


def _default_profile_stage_group(args: argparse.Namespace) -> Optional[str]:
    if args.profile_stage_group:
        return str(args.profile_stage_group)
    if args.eye_stage in {"eye_masks_runs", "refined_eye_masks_runs"}:
        return str(args.eye_stage)
    return None


def _quality_filters_active(args: argparse.Namespace) -> bool:
    return any(
        [
            args.eye_mask_method is not None,
            args.min_usable_rate is not None,
            args.require_review_state is not None,
            args.require_review_intended_use is not None,
            _default_profile_stage_group(args) is not None,
        ]
    )


def _choose_profile_candidate(
    candidates: Sequence[Mapping[str, Any]],
    *,
    preferred_stages: Sequence[str],
    required_review_state: Optional[str],
    required_review_use: Optional[str],
) -> Optional[Mapping[str, Any]]:
    if not candidates:
        return None
    ordered = sorted(candidates, key=lambda row: _parse_iso_ts(row.get("profile_created_utc")), reverse=True)

    prioritized: List[Mapping[str, Any]] = []
    seen_ids: set[int] = set()
    for stage in preferred_stages:
        for row in ordered:
            if id(row) in seen_ids:
                continue
            if _as_text(row.get("stage_group")) == stage:
                prioritized.append(row)
                seen_ids.add(id(row))
    for row in ordered:
        if id(row) in seen_ids:
            continue
        prioritized.append(row)
        seen_ids.add(id(row))

    for row in prioritized:
        review_state = _as_text(row.get("review_state"))
        review_use = _as_text(row.get("review_intended_use"))
        if required_review_state is not None and review_state != required_review_state:
            continue
        if required_review_use is not None and review_use != required_review_use:
            continue
        return row
    return None


def _choose_dataset_name(seen: set[str], zarr_path: Path, ordinal: int) -> str:
    base = _sanitize_name(zarr_path.stem) or f"dataset_{ordinal}"
    candidate = base
    suffix = 2
    while candidate in seen:
        candidate = f"{base}_{suffix}"
        suffix += 1
    seen.add(candidate)
    return candidate


def _normalize_input_format(value: str) -> str:
    text = str(value).strip().lower()
    if text not in {"gray", "rgb"}:
        raise ValueError(f"Unsupported input format '{value}'. Expected gray or rgb.")
    return text


def _normalize_label_mode(value: str) -> str:
    text = str(value).strip().lower()
    if text not in {"lr", "union"}:
        raise ValueError(f"Unsupported label mode '{value}'. Expected lr or union.")
    return text


def _build_query_filter_payload(
    args: argparse.Namespace,
    *,
    model_input: str,
    set_name: Optional[str],
    set_version: Optional[int],
    set_id: Optional[str],
    selected_paths: Sequence[Path],
) -> Dict[str, Any]:
    return {
        "tool": "fisheye.utils.prepare_eye_mask_training_from_registry",
        "task": "eye_masks",
        "input_format": args.input_format,
        "label_mode": args.label_mode,
        "model_input": model_input,
        "eye_stage": args.eye_stage,
        "eye_run": args.eye_run,
        "crop_run": args.crop_run,
        "eye_mask_method": args.eye_mask_method,
        "profile_stage_group": _default_profile_stage_group(args),
        "min_usable_rate": args.min_usable_rate,
        "require_review_state": args.require_review_state,
        "require_review_intended_use": args.require_review_intended_use,
        "split_train": float(args.split_train),
        "split_val": float(args.split_val),
        "split_test": float(args.split_test),
        "split_seed": int(args.split_seed),
        "set_name": set_name,
        "set_version": set_version,
        "set_id": set_id,
        "selected_zarr_paths": [str(path) for path in selected_paths],
    }


@dataclass(frozen=True)
class EyeSourceSelection:
    crop_run: str
    eye_stage: str
    eye_run: str
    total_samples: int
    channels: int
    mask_probs_name: Optional[str]
    method: Optional[str]
    source_keypoints_run: Optional[str]
    source_keypoint_group: Optional[str]


def _inspect_source_selection(
    zarr_path: Path,
    *,
    crop_run: Optional[str],
    eye_stage: str,
    eye_run: Optional[str],
) -> EyeSourceSelection:
    try:
        root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    except TypeError:
        root = zarr.open_group(str(zarr_path), mode="r")
    selection, _crop_group, eye_group = export_eye._select_source_runs(
        root,
        crop_run=crop_run,
        eye_stage=eye_stage,
        eye_run=eye_run,
    )
    source_keypoints_run = _as_text(eye_group.attrs.get("source_keypoints_run"))
    if source_keypoints_run is None:
        source_keypoints_run = _as_text(eye_group.attrs.get("source_keypoint_run"))
    return EyeSourceSelection(
        crop_run=str(selection.crop_run),
        eye_stage=str(selection.eye_stage),
        eye_run=str(selection.eye_run),
        total_samples=int(selection.total_samples),
        channels=int(selection.channels),
        mask_probs_name=_as_text(selection.mask_probs_name),
        method=_as_text(eye_group.attrs.get("method")),
        source_keypoints_run=source_keypoints_run,
        source_keypoint_group=_as_text(eye_group.attrs.get("source_keypoint_group")),
    )


def _ensure_suffix(value: str, suffix: str) -> str:
    text = str(value).strip()
    if text.endswith(suffix):
        return text
    return f"{text}{suffix}"


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
        merged_stem_base = _sanitize_name(out_config.stem) or "eye_mask_training"
    merged_stem = _ensure_suffix(merged_stem_base, "_merged")
    return out_dir / "zarr" / f"{merged_stem}.zarr"


def _resolve_merged_dataset_identity(
    *,
    set_id: Optional[str],
    set_name: Optional[str],
    merged_out_zarr: Path,
) -> tuple[str, str]:
    dataset_id_base = _sanitize_name(set_id or set_name or merged_out_zarr.stem) or "eye_mask_training"
    dataset_id = _ensure_suffix(dataset_id_base, "_merged")
    dataset_name = dataset_id
    return dataset_id, dataset_name


def _print_summary(
    *,
    set_id: Optional[str],
    set_name: Optional[str],
    input_format: str,
    label_mode: str,
    selected_sources: Sequence[Mapping[str, Any]],
    merged_dataset: Mapping[str, Any],
) -> None:
    print("\nEye-Mask Training Preflight")
    print("  Task: eye_masks")
    print(f"  Input format: {input_format}")
    print(f"  Label mode: {label_mode}")
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
        print(f"  Eye source: {source['source_eye_stage']}/{source['source_eye_run']}")
        print(f"  Crop run: {source['source_crop_run']}")
        print(f"  Total samples: {source['total_samples']}")


def _build_training_config_payload(
    *,
    base_config_path: Path,
    datasets: Sequence[Mapping[str, Any]],
    label_mode: str,
    split_seed: int,
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
        dataset_entry["mask_run"] = str(dataset["run_name"])
        generated_datasets[str(dataset["name"])] = dataset_entry
    config_payload["datasets"] = generated_datasets

    config_payload["random_seed"] = int(split_seed)
    training_params = config_payload.get("training_params")
    if training_params is None:
        training_params = {}
        config_payload["training_params"] = training_params
    if not isinstance(training_params, dict):
        raise ValueError("Base config training_params must be a mapping.")
    training_params["label_mode"] = str(label_mode)
    return config_payload


def _execute_merged_export(
    *,
    selected_sources: List[Dict[str, Any]],
    merged_dataset: Dict[str, Any],
    input_format: str,
    label_mode: str,
    split_train: float,
    split_val: float,
    split_test: float,
    split_seed: int,
    overwrite: bool,
    registry_path: Optional[Path],
    set_id: Optional[str],
    set_name: Optional[str],
    aggregate_training_data_card: bool,
    data_card_no_plots: bool,
) -> Dict[str, int]:
    summary = {
        "planned": 1,
        "succeeded": 0,
        "failed": 0,
    }
    source_specs = [
        export_eye.EyeMergeSourceSpec(
            source_zarr=Path(str(source["zarr_path"])),
            crop_run=_as_text(source.get("source_crop_run")),
            eye_stage=_as_text(source.get("source_eye_stage")) or "auto",
            eye_run=_as_text(source.get("source_eye_run")),
        )
        for source in selected_sources
    ]
    out_zarr = Path(str(merged_dataset["out_zarr"]))
    try:
        result = export_eye.export_merged_eye_mask_training_zarr_from_sources(
            source_specs=source_specs,
            out_zarr=out_zarr,
            run_name=_as_text(merged_dataset.get("run_name")) or "merged_eye_masks",
            input_format=input_format,
            label_mode=label_mode,
            split_train=float(split_train),
            split_val=float(split_val),
            split_test=float(split_test),
            split_seed=int(split_seed),
            overwrite=bool(overwrite),
            validate=True,
            registry=registry_path,
            training_set_id=set_id,
            training_set_name=set_name if set_id is not None else None,
            aggregate_training_data_card=bool(aggregate_training_data_card),
            data_card_no_plots=bool(data_card_no_plots),
        )
    except Exception as exc:
        merged_dataset["export_status"] = "failed"
        merged_dataset["export_error"] = str(exc)
        summary["failed"] += 1
        print(f"export_error\t{merged_dataset['dataset_id']}\t{out_zarr}\t{exc}")
        return summary

    merged_dataset["export_status"] = "succeeded"
    merged_dataset["export_result"] = result
    summary["succeeded"] += 1
    print(f"exported\t{merged_dataset['dataset_id']}\t{out_zarr}\tsources={len(source_specs)}")
    return summary


def _reject_orchestration_flags(args: argparse.Namespace) -> None:
    legacy_flags: list[str] = []
    if args.overwrite:
        legacy_flags.append("--overwrite")
    if args.aggregate_training_data_card:
        legacy_flags.append("--aggregate-training-data-card")
    if args.data_card_no_plots:
        legacy_flags.append("--data-card-no-plots")
    if legacy_flags:
        flags_text = ", ".join(sorted(set(legacy_flags)))
        raise SystemExit(
            "prepare_eye_mask_training_from_registry is now prepare-only and does not run merge/export/aggregation. "
            f"Received orchestration flags: {flags_text}. "
            "Use: scripts/py -m fisheye.utils.run_eye_mask_training_pipeline ..."
        )


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

    parser.add_argument("--eye-stage", choices=["auto", "eye_masks_runs", "refined_eye_masks_runs"], default="auto")
    parser.add_argument("--eye-run", type=str, help="Explicit eye-mask run name.")
    parser.add_argument("--crop-run", type=str, help="Explicit source crop run name.")
    parser.add_argument("--input-format", choices=["gray", "rgb"], default="gray")
    parser.add_argument("--model-input", choices=["gray", "rgb"], help="Registry downsample filter mode.")
    parser.add_argument("--label-mode", choices=["lr", "union"], default="lr")
    parser.add_argument("--split-train", type=float, default=0.8)
    parser.add_argument("--split-val", type=float, default=0.2)
    parser.add_argument("--split-test", type=float, default=0.0)
    parser.add_argument("--split-seed", type=int, default=42)

    parser.add_argument(
        "--profile-stage-group",
        choices=["eye_masks_runs", "refined_eye_masks_runs"],
        type=str,
        help="Optional eye-mask profile stage filter.",
    )
    parser.add_argument("--eye-mask-method", type=str, help="Optional eye-mask method filter from profile rows.")
    parser.add_argument("--min-usable-rate", type=float, help="Require profile usable_rate >= threshold.")
    parser.add_argument(
        "--require-review-state",
        choices=["approved", "pending", "rejected", "needs_review"],
        help="Require profile review_state.",
    )
    parser.add_argument(
        "--require-review-intended-use",
        choices=["training", "full_recording"],
        help="Require profile review_intended_use.",
    )

    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/fisheye/eye_segmentation_config.yaml"),
        help="Base eye-mask training config used to generate the output YAML.",
    )
    parser.add_argument("--out-config", type=Path, help="Output training config YAML path.")
    parser.add_argument("--out-dir", type=Path, help="Output directory root for merged zarr exports.")
    parser.add_argument("--out-manifest", type=Path, help="Output manifest JSON path.")
    parser.add_argument("--out-commands", type=Path, help="Write export commands to this file.")
    parser.add_argument("--set-name", type=str, help="Training set name.")
    parser.add_argument("--set-version", type=int)
    parser.add_argument("--run-name-prefix", type=str, default="merged_eye_masks")
    parser.add_argument("--overwrite", action="store_true", help="Pass --overwrite to export calls.")
    parser.add_argument(
        "--aggregate-training-data-card",
        action="store_true",
        help="Run data-card aggregation during each export call.",
    )
    parser.add_argument(
        "--data-card-no-plots",
        action="store_true",
        help="Skip data-card plot generation during export calls.",
    )
    parser.add_argument("--dry-run", action="store_true")

    cli_argv = [str(token) for token in (list(argv) if argv is not None else list(sys.argv[1:]))]
    args = parser.parse_args(cli_argv)
    _reject_orchestration_flags(args)
    args.input_format = _normalize_input_format(args.input_format)
    args.label_mode = _normalize_label_mode(args.label_mode)

    if args.min_usable_rate is not None and not (0.0 <= args.min_usable_rate <= 1.0):
        raise ValueError("--min-usable-rate must be between 0 and 1.")
    if float(args.split_train) < 0.0 or float(args.split_val) < 0.0 or float(args.split_test) < 0.0:
        raise ValueError("Split ratios must be non-negative.")
    if (float(args.split_train) + float(args.split_val) + float(args.split_test)) <= 0.0:
        raise ValueError("At least one split ratio must be greater than zero.")

    model_input = args.model_input or args.input_format
    if args.model_input and args.model_input != args.input_format:
        raise SystemExit("--model-input must match --input-format for eye-mask training selection.")

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

    non_training_rows: List[Mapping[str, Any]] = []
    skipped_training_rows: List[tuple[str, str, str]] = []
    for row in rows:
        purpose = _as_text(row["zarr_use"])
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
            "(non-source artifacts) before eye-mask selection."
        )
        for dataset_id, zarr_path, reason in skipped_training_rows[:20]:
            print(f"  - {dataset_id} [{reason}] {zarr_path}")
        if len(skipped_training_rows) > 20:
            print(f"  ... {len(skipped_training_rows) - 20} more skip(s) omitted.")
    if not rows:
        registry.close()
        raise SystemExit("No source datasets remain after prefiltering.")

    selected_profile_rows: Dict[str, Mapping[str, Any]] = {}
    quality_exclusions: List[Dict[str, str]] = []
    if _quality_filters_active(args):
        dataset_ids = [str(row["dataset_id"]) for row in rows if row.get("dataset_id")]
        profile_rows = registry.query_eye_mask_data_profile_latest(
            dataset_ids=dataset_ids,
            stage_group=_default_profile_stage_group(args),
            eye_mask_method=args.eye_mask_method,
            min_usable_rate=args.min_usable_rate,
        )
        by_dataset: Dict[str, List[Mapping[str, Any]]] = {}
        for profile_row in profile_rows:
            profile_row_dict = dict(profile_row)
            dataset_id = _as_text(profile_row_dict.get("dataset_id"))
            if dataset_id:
                by_dataset.setdefault(dataset_id, []).append(profile_row_dict)

        if args.eye_stage == "refined_eye_masks_runs":
            preferred_stages = ["refined_eye_masks_runs", "eye_masks_runs"]
        elif args.eye_stage == "eye_masks_runs":
            preferred_stages = ["eye_masks_runs", "refined_eye_masks_runs"]
        else:
            preferred_stages = ["refined_eye_masks_runs", "eye_masks_runs"]

        filtered_rows: List[Mapping[str, Any]] = []
        for row in rows:
            dataset_id = str(row["dataset_id"])
            candidates = by_dataset.get(dataset_id, [])
            chosen = _choose_profile_candidate(
                candidates,
                preferred_stages=preferred_stages,
                required_review_state=args.require_review_state,
                required_review_use=args.require_review_intended_use,
            )
            if chosen is None:
                quality_exclusions.append(
                    {
                        "dataset_id": dataset_id,
                        "zarr_path": str(row["zarr_path"]),
                        "reason": "missing_or_mismatched_eye_mask_profile",
                    }
                )
                continue
            selected_profile_rows[dataset_id] = chosen
            filtered_rows.append(row)
        rows = filtered_rows

        if quality_exclusions:
            print(f"Eye-mask profile filter excluded {len(quality_exclusions)} dataset(s):")
            for exclusion in quality_exclusions[:20]:
                print(f"  - {exclusion['dataset_id']} [{exclusion['reason']}] {exclusion['zarr_path']}")
            if len(quality_exclusions) > 20:
                print(f"  ... {len(quality_exclusions) - 20} more exclusion(s) omitted.")

    registry.close()
    if not rows:
        raise SystemExit("No datasets remain after eye-mask profile filtering.")

    resolved_set_name: Optional[str] = None
    set_version: Optional[int] = None
    set_id: Optional[str] = None
    if args.out_manifest is None and args.out_config is None and args.out_dir is None:
        if args.set_name:
            resolved_set_name = _sanitize_name(args.set_name)
        else:
            resolved_set_name = _default_set_name(args, rows, model_input=model_input)
            print(f"Auto set-name: {resolved_set_name}")
        dataset_root = _resolve_dataset_root("eye_masks")
        if args.set_version is not None:
            if int(args.set_version) < 1:
                raise ValueError("--set-version must be >= 1")
            set_version = int(args.set_version)
        else:
            set_version = _next_version_from_dataset_root(
                set_name=resolved_set_name,
                task_prefix="eye_mask",
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
            out_dir = _resolve_dataset_root("eye_masks") / set_id
        elif args.out_config is not None:
            out_dir = args.out_config.parent
        elif args.out_manifest is not None:
            out_dir = args.out_manifest.parent
        else:
            out_dir = _resolve_dataset_root("eye_masks")
        if args.out_config is not None:
            out_config = args.out_config
        elif args.out_manifest is not None:
            out_config = _config_path_for_manifest(args.out_manifest)
        elif set_id is not None:
            out_config = out_dir / f"{set_id}.yaml"
        else:
            out_config = out_dir / "eye_mask_training.yaml"
        out_manifest = args.out_manifest if args.out_manifest is not None else _manifest_path_for_config(out_config)

    zarr_paths = [Path(str(row["zarr_path"])) for row in rows]
    if args.output_file_list:
        args.output_file_list.parent.mkdir(parents=True, exist_ok=True)
        args.output_file_list.write_text("\n".join(str(path) for path in zarr_paths) + "\n", encoding="utf-8")
        print(f"Wrote {len(zarr_paths)} paths to {args.output_file_list}")
    print(f"Registry query matched {len(zarr_paths)} dataset(s).")

    seen_names: set[str] = set()
    selected_sources_payload: List[Dict[str, Any]] = []
    total_selected_samples = 0
    for ordinal, row in enumerate(rows, start=1):
        source_zarr = Path(str(row["zarr_path"]))
        source_info = _inspect_source_selection(
            source_zarr,
            crop_run=args.crop_run,
            eye_stage=args.eye_stage,
            eye_run=args.eye_run,
        )
        dataset_name = _choose_dataset_name(seen_names, source_zarr, ordinal)
        dataset_id = str(row["dataset_id"])
        profile_row = selected_profile_rows.get(dataset_id)
        total_selected_samples += int(source_info.total_samples)
        selected_sources_payload.append(
            {
                "name": dataset_name,
                "dataset_id": dataset_id,
                "session_uuid": _as_text(row.get("session_uuid")),
                "zarr_path": str(source_zarr),
                "source_eye_stage": source_info.eye_stage,
                "source_eye_run": source_info.eye_run,
                "source_crop_run": source_info.crop_run,
                "source_eye_method": source_info.method,
                "source_keypoints_run": source_info.source_keypoints_run,
                "source_keypoint_group": source_info.source_keypoint_group,
                "total_samples": int(source_info.total_samples),
                "channels": int(source_info.channels),
                "mask_probs_name": source_info.mask_probs_name,
                "dish_design": _as_text(row.get("dish_design")),
                "canvas_name": _as_text(row.get("canvas_name")),
                "rig_id": _as_text(row.get("rig_id")),
                "eye_mask_profile": (
                    {
                        "profile_run": _as_text(profile_row.get("profile_run")),
                        "stage_group": _as_text(profile_row.get("stage_group")),
                        "eye_mask_method": _as_text(profile_row.get("eye_mask_method")),
                        "usable_rate": _as_float(profile_row.get("usable_rate")),
                        "review_state": _as_text(profile_row.get("review_state")),
                        "review_intended_use": _as_text(profile_row.get("review_intended_use")),
                        "profile_created_utc": _as_text(profile_row.get("profile_created_utc")),
                    }
                    if profile_row is not None
                    else None
                ),
            }
        )

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
    merged_run_name = _sanitize_name(args.run_name_prefix) or "merged_eye_masks"

    command_lines: List[str] = []
    if len(selected_sources_payload) == 1:
        source = selected_sources_payload[0]
        single_cmd: List[str] = [
            "scripts/py",
            "-m",
            "fisheye.utils.export_eye_mask_training_zarr",
            str(source["zarr_path"]),
            str(merged_out_zarr),
            "--eye-stage",
            str(source["source_eye_stage"]),
            "--eye-run",
            str(source["source_eye_run"]),
            "--crop-run",
            str(source["source_crop_run"]),
            "--run-name",
            str(merged_run_name),
            "--input-format",
            str(args.input_format),
            "--label-mode",
            str(args.label_mode),
            "--split-train",
            str(float(args.split_train)),
            "--split-val",
            str(float(args.split_val)),
            "--split-test",
            str(float(args.split_test)),
            "--split-seed",
            str(int(args.split_seed)),
        ]
        if registry_path is not None:
            single_cmd.extend(["--registry", str(registry_path)])
        if set_id is not None:
            single_cmd.extend(["--training-set-id", str(set_id)])
        if resolved_set_name is not None and set_id is not None:
            single_cmd.extend(["--training-set-name", str(resolved_set_name)])
        if bool(args.overwrite):
            single_cmd.append("--overwrite")
        if bool(args.aggregate_training_data_card):
            single_cmd.append("--aggregate-training-data-card")
        if bool(args.data_card_no_plots):
            single_cmd.append("--data-card-no-plots")
        command_lines.append(" ".join(single_cmd))
    else:
        command_lines.append(
            "# multi-source merged eye-mask export executed via API "
            f"(sources={len(selected_sources_payload)} out_zarr={merged_out_zarr})"
        )

    merged_dataset_payload: Dict[str, Any] = {
        "name": merged_dataset_name,
        "run_name": merged_run_name,
        "dataset_id": merged_dataset_id,
        "out_zarr": str(merged_out_zarr),
        "source_count": int(len(selected_sources_payload)),
        "total_samples": int(total_selected_samples),
        "input_format": args.input_format,
        "label_mode": args.label_mode,
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
        tool="fisheye.utils.prepare_eye_mask_training_from_registry",
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
    )
    config_payload = _build_training_config_payload(
        base_config_path=args.base_config,
        datasets=[merged_dataset_payload],
        label_mode=args.label_mode,
        split_seed=int(args.split_seed),
    )
    merged_export_payload: Dict[str, Any] = {
        "dataset_name": merged_dataset_name,
        "dataset_id": merged_dataset_id,
        "run_name": merged_run_name,
        "zarr_path": str(merged_out_zarr),
        "source_count": int(len(selected_sources_payload)),
        "total_samples": int(total_selected_samples),
        "source_datasets": selected_sources_payload,
        "command": command_lines[0],
        "export_status": "planned",
    }
    manifest_payload: Dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "eye_masks",
        "registry_path": str(registry_path),
        "set_name": resolved_set_name,
        "set_version": set_version,
        "set_id": set_id,
        "input_format": args.input_format,
        "label_mode": args.label_mode,
        "eye_stage": args.eye_stage,
        "eye_run": args.eye_run,
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
        label_mode=args.label_mode,
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

    manifest_json = json.dumps(manifest_payload, indent=2)

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
