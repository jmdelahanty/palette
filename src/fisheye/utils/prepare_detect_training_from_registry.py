#!/usr/bin/env python3
"""Query the registry and build a detection training manifest/config."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, List, Mapping, Optional, Sequence

import numpy as np
from fisheye.diagnostics import prepare_detect_training as pdt
from fisheye.registry.db import Registry, RegistryPaths


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


def _resolve_default_dataset_root() -> Path:
    env_override = os.getenv("PALETTE_TRAINING_DATASETS_ROOT")
    if env_override:
        return Path(env_override).expanduser().resolve()
    nvme_default = Path("/nvme1/training/datasets")
    if nvme_default.exists():
        return nvme_default
    return Path("datasets") / "detect"


def _looks_like_training_artifact_path(zarr_path: Path) -> bool:
    normalized = str(zarr_path).replace("\\", "/").lower()
    stem = zarr_path.stem.lower()
    if "/training/datasets/" in normalized:
        return True
    if stem.endswith("_merged"):
        return True
    return False


def _next_detect_version(set_name: str) -> int:
    versions: list[int] = []
    dataset_root = _resolve_default_dataset_root()
    dataset_re = re.compile(rf"^detect_{re.escape(set_name)}_v(\d+)$")
    if dataset_root.exists():
        for entry in dataset_root.iterdir():
            match = dataset_re.match(entry.name)
            if match:
                versions.append(int(match.group(1)))

    # Legacy v1 pathing under repo runs/ directories.
    config_dir = Path("runs") / "configs" / "detect"
    manifest_dir = Path("runs") / "manifests" / "detect"
    config_re = re.compile(rf"^{re.escape(set_name)}_v(\d+)\.ya?ml$")
    manifest_re = re.compile(rf"^{re.escape(set_name)}_v(\d+)\.manifest\.json$")
    if config_dir.exists():
        for path in config_dir.glob(f"{set_name}_v*.y*ml"):
            match = config_re.match(path.name)
            if match:
                versions.append(int(match.group(1)))
    if manifest_dir.exists():
        for path in manifest_dir.glob(f"{set_name}_v*.manifest.json"):
            match = manifest_re.match(path.name)
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


def _build_query_signature(args: argparse.Namespace, *, model_input: str) -> dict[str, Any]:
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
        "source_type": args.source_type,
        "input_format": args.input_format,
        "model_input": model_input,
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
    query_hash = _query_hash(args, model_input=model_input)
    return "_".join(
        [
            _slug_component(dish_raw, fallback="all_dishes"),
            _slug_component(canvas_raw, fallback="unknown_canvas"),
            _slug_component(args.source_type, fallback="source"),
            _slug_component(args.input_format, fallback="input"),
            query_hash,
        ]
    )


def _resolve_preflight_output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.out_config is not None:
        config_path = Path(args.out_config)
        manifest_path = Path(args.out_manifest) if args.out_manifest is not None else config_path.with_suffix(".manifest.json")
        return config_path, manifest_path
    if args.set_name is None:
        raise SystemExit("Unable to infer output paths: set-name/out-config unresolved.")
    safe_name = _sanitize_name(args.set_name)
    if args.set_version is None:
        args.set_version = _next_detect_version(safe_name)
    version = int(args.set_version)
    set_id = f"detect_{safe_name}_v{version:03d}"
    out_dir = _resolve_default_dataset_root() / set_id
    config_path = out_dir / f"{set_id}.yaml"
    manifest_path = Path(args.out_manifest) if args.out_manifest is not None else (out_dir / f"{set_id}.manifest.json")
    return config_path, manifest_path


def _reject_legacy_orchestration_flags(args: argparse.Namespace) -> None:
    legacy_flags: list[str] = []
    if args.train:
        legacy_flags.append("--train")
    if args.export_merged:
        legacy_flags.append("--export-merged")
    if args.export_onnx:
        legacy_flags.append("--export-onnx")
    if args.export_trt:
        legacy_flags.append("--export-trt")
    if args.onnx_opset is not None:
        legacy_flags.append("--onnx-opset")
    if args.onnx_simplify:
        legacy_flags.append("--onnx-simplify")
    if args.onnx_path:
        legacy_flags.append("--onnx-path")
    if args.nms_conf is not None:
        legacy_flags.append("--nms-conf")
    if args.nms_iou is not None:
        legacy_flags.append("--nms-iou")
    if args.nms_topk is not None:
        legacy_flags.append("--nms-topk")
    if args.trt_precision is not None:
        legacy_flags.append("--trt-precision")
    if args.trtexec:
        legacy_flags.append("--trtexec")
    if args.trt_cuda_graph:
        legacy_flags.append("--trt-cuda-graph")
    if args.trt_profiling:
        legacy_flags.append("--trt-profiling")
    if args.trt_verbose:
        legacy_flags.append("--trt-verbose")
    if args.merge_out_zarr is not None:
        legacy_flags.append("--merge-out-zarr")
    if args.merge_out_dir is not None:
        legacy_flags.append("--merge-out-dir")
    if args.merge_split is not None:
        legacy_flags.append("--merge-split")
    if args.merge_seed is not None:
        legacy_flags.append("--merge-seed")
    if args.merge_copy_batch_size is not None:
        legacy_flags.append("--merge-copy-batch-size")
    if args.merge_include_rgb:
        legacy_flags.append("--merge-include-rgb")
    if args.merge_overwrite:
        legacy_flags.append("--merge-overwrite")
    if legacy_flags:
        msg = ", ".join(sorted(set(legacy_flags)))
        raise SystemExit(
            "prepare_detect_training_from_registry is now prepare-only and does not run merge/train/export. "
            f"Received orchestration flags: {msg}. "
            "Use: python -m fisheye.utils.run_detect_training_pipeline ..."
        )


def _decode_attr(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        text = value.decode("utf-8", "ignore").strip()
    else:
        text = str(value).strip()
    return text or None


def _coerce_mapping(value: Any) -> Optional[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        return value
    return None


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _rate_matches(expected: Optional[float], observed: Optional[float], *, tol: float = 1e-9) -> bool:
    if expected is None and observed is None:
        return True
    if expected is None or observed is None:
        return False
    return abs(float(expected) - float(observed)) <= tol


def _resolve_detect_quality_from_zarr(
    zarr_path: Path,
    *,
    expected_refined_run: str,
) -> dict[str, Any]:
    import zarr  # local import to keep module import-light for CLI help usage

    try:
        root = zarr.open_group(str(zarr_path), mode="r", consolidated=False)
    except TypeError:
        root = zarr.open_group(str(zarr_path), mode="r")

    refined_parent = root.get("refined_detect_runs") or root.get("refined_runs")
    if refined_parent is None:
        raise ValueError(f"{zarr_path.name}: missing refined_detect_runs/refined_runs group.")
    if expected_refined_run not in refined_parent:
        raise ValueError(
            f"{zarr_path.name}: stale detect_quality row: refined run '{expected_refined_run}' is missing."
        )

    refined_group = refined_parent[expected_refined_run]
    review_status = _coerce_mapping(refined_group.attrs.get("detect_review_status"))
    review_state = _decode_attr(review_status.get("state")) if review_status else None
    review_use = _decode_attr(review_status.get("intended_use")) if review_status else None
    source_detect_run = _decode_attr(refined_group.attrs.get("source_detect_run"))

    resolved_group = _decode_attr(review_status.get("resolved_group")) if review_status else None
    if not resolved_group:
        manual_latest = _decode_attr(refined_group.attrs.get("manual_review_latest"))
        if manual_latest and manual_latest in refined_group:
            resolved_group = manual_latest
        elif "interpolated" in refined_group:
            resolved_group = "interpolated"
        elif "filtered" in refined_group:
            resolved_group = "filtered"
        else:
            resolved_group = "raw"

    resolved = refined_group.get(resolved_group) if resolved_group else None
    if resolved is None:
        raise ValueError(
            f"{zarr_path.name}: refined run '{expected_refined_run}' missing resolved group '{resolved_group}'."
        )

    total_detections: Optional[int] = None
    real_detections: Optional[int] = None
    interpolated_detections: Optional[int] = None
    interpolated_detections_rate: Optional[float] = None

    if "bbox_norm_coords" in resolved:
        total_detections = int(resolved["bbox_norm_coords"].shape[0])

    if "detection_source" in resolved:
        source_arr = np.asarray(resolved["detection_source"][:], dtype=np.int64)
        real_detections = int(np.sum(source_arr == 0))
        interpolated_detections = int(np.sum(source_arr != 0))
        total_detections = int(source_arr.shape[0])
    else:
        if total_detections is None:
            total_detections = _as_int(resolved.attrs.get("total_detections"))
        interpolated_detections = _as_int(resolved.attrs.get("interpolated_detections"))
        if interpolated_detections is None and _decode_attr(resolved_group) == "filtered":
            interpolated_detections = 0
        real_detections = _as_int(resolved.attrs.get("original_detections"))
        if real_detections is None and total_detections is not None and interpolated_detections is not None:
            real_detections = int(total_detections) - int(interpolated_detections)

    if total_detections is None and real_detections is not None and interpolated_detections is not None:
        total_detections = int(real_detections) + int(interpolated_detections)
    if total_detections is not None and interpolated_detections is not None and int(total_detections) > 0:
        interpolated_detections_rate = float(interpolated_detections) / float(total_detections)

    try:
        zarr_mtime_ns = int(zarr_path.stat().st_mtime_ns)
    except Exception:
        zarr_mtime_ns = None

    return {
        "source_detect_run": source_detect_run,
        "review_state": review_state,
        "review_intended_use": review_use,
        "review_resolved_group": resolved_group,
        "total_detections": total_detections,
        "real_detections": real_detections,
        "interpolated_detections": interpolated_detections,
        "interpolated_detections_rate": interpolated_detections_rate,
        "zarr_mtime_ns": zarr_mtime_ns,
    }


def main(argv: Optional[List[str]] = None) -> int:
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

    # prepare_detect_training passthroughs
    parser.add_argument("--source-type", choices=["detect", "filtered", "interpolated", "manual"], default="manual")
    parser.add_argument("--input-format", choices=["gray", "rgb"], default="gray")
    parser.add_argument(
        "--model-input",
        choices=["gray", "rgb"],
        help="Registry filter for required training input modality. Defaults to --input-format.",
    )
    parser.add_argument("--base-config", type=Path, default=Path("configs/fisheye/detect_config.yaml"))
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
    parser.add_argument("--provenance-policy", choices=["warn", "strict", "ignore"], default="warn")
    parser.add_argument("--metadata-json", type=Path)
    parser.add_argument("--allow-source-mismatch", action="store_true")
    parser.add_argument("--allow-unapproved", action="store_true")
    parser.add_argument("--no-prefer-manual", action="store_true")
    parser.add_argument(
        "--require-review-state",
        choices=["approved", "pending", "rejected", "needs_review"],
        help="Require refined detect review state via detect_quality_current.",
    )
    parser.add_argument(
        "--require-review-intended-use",
        choices=["training", "full_recording"],
        help="Require refined detect review intended_use via detect_quality_current.",
    )
    parser.add_argument(
        "--max-interpolated-detections-rate",
        type=float,
        help="Require interpolated_detections_rate <= threshold (0-1) via detect_quality_current.",
    )
    # Legacy orchestration flags are intentionally hidden and rejected.
    parser.add_argument("--export-onnx", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--export-trt", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--onnx-opset", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--onnx-simplify", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--onnx-path", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--nms-conf", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--nms-iou", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--nms-topk", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--trt-precision", choices=["fp16", "int8"], help=argparse.SUPPRESS)
    parser.add_argument("--trtexec", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--trt-cuda-graph", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--trt-profiling", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--trt-verbose", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--train", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--register", action="store_true", help="Register datasets in the registry.")
    parser.add_argument(
        "--register-registry",
        type=Path,
        help="Registry path to use when --register is set (defaults to --registry).",
    )
    parser.add_argument("--export-merged", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--merge-out-zarr", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--merge-out-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--merge-split", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--merge-seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--merge-copy-batch-size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--merge-include-rgb", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--merge-overwrite", action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args(argv)
    _reject_legacy_orchestration_flags(args)

    model_input = args.model_input or args.input_format
    if args.model_input and args.model_input != args.input_format:
        raise SystemExit("--model-input must match --input-format for detection training selection.")
    if args.max_interpolated_detections_rate is not None and not (0.0 <= args.max_interpolated_detections_rate <= 1.0):
        raise ValueError("--max-interpolated-detections-rate must be between 0 and 1.")

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

    non_training_rows: List[Mapping[str, Any]] = []
    skipped_training_rows: List[tuple[str, str, str]] = []
    for row in rows:
        purpose = _decode_attr(row["zarr_purpose"])
        zarr_path = Path(str(row["zarr_path"]))
        if str(purpose or "").lower() == "training" and _looks_like_training_artifact_path(zarr_path):
            skipped_training_rows.append(
                (
                    str(row["dataset_id"]),
                    str(zarr_path),
                    "zarr_purpose=training and path looks like merged/training artifact",
                )
            )
            continue
        non_training_rows.append(row)
    rows = non_training_rows
    if skipped_training_rows:
        print(
            f"Skipped {len(skipped_training_rows)} training-purpose dataset(s) "
            "(non-source artifacts) before detect selection."
        )
    if not rows:
        registry.close()
        raise SystemExit("No source datasets remain after prefiltering.")

    quality_gate_active = (
        args.require_review_state is not None
        or args.require_review_intended_use is not None
        or args.max_interpolated_detections_rate is not None
    )
    selected_quality_rows_by_dataset: dict[str, Mapping[str, Any]] = {}
    quality_exclusions: List[dict[str, Any]] = []
    if quality_gate_active:
        dataset_ids_all = [str(row["dataset_id"]) for row in rows if row["dataset_id"]]
        selected_quality_rows = registry.query_detect_quality_current(
            dataset_ids=dataset_ids_all,
            review_state=args.require_review_state,
            review_intended_use=args.require_review_intended_use,
            max_interpolated_detections_rate=args.max_interpolated_detections_rate,
        )
        selected_quality_rows_by_dataset = {
            str(row["dataset_id"]): dict(row) for row in selected_quality_rows if row["dataset_id"]
        }
        all_quality_rows = registry.query_detect_quality_current(dataset_ids=dataset_ids_all)
        all_quality_by_dataset: dict[str, List[Mapping[str, Any]]] = {}
        for quality_row in all_quality_rows:
            dataset_id = str(quality_row["dataset_id"])
            all_quality_by_dataset.setdefault(dataset_id, []).append(dict(quality_row))

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
                    {"dataset_id": dataset_id, "zarr_path": zarr_path, "reason": "missing_quality_row"}
                )
                continue
            candidate = candidate_rows[0]
            review_state = _decode_attr(candidate.get("review_state"))
            review_use = _decode_attr(candidate.get("review_intended_use"))
            interp_rate = _as_float(candidate.get("interpolated_detections_rate"))
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
                        "reason": (
                            f"review_use_mismatch:{review_use or 'missing'}"
                            f"!={args.require_review_intended_use}"
                        ),
                    }
                )
            elif args.max_interpolated_detections_rate is not None and interp_rate is None:
                quality_exclusions.append(
                    {
                        "dataset_id": dataset_id,
                        "zarr_path": zarr_path,
                        "reason": "missing_interpolated_detections_rate",
                    }
                )
            elif (
                args.max_interpolated_detections_rate is not None
                and interp_rate is not None
                and float(interp_rate) > float(args.max_interpolated_detections_rate)
            ):
                quality_exclusions.append(
                    {
                        "dataset_id": dataset_id,
                        "zarr_path": zarr_path,
                        "reason": (
                            "interpolated_rate_above_threshold:"
                            f"{interp_rate:.6f}>{args.max_interpolated_detections_rate:.6f}"
                        ),
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
            print(f"Detect quality SQL filter excluded {len(quality_exclusions)} dataset(s):")
            for exclusion in quality_exclusions[:20]:
                print(f"  - {exclusion['dataset_id']} [{exclusion['reason']}] {exclusion['zarr_path']}")
            if len(quality_exclusions) > 20:
                print(f"  ... {len(quality_exclusions) - 20} more exclusion(s) omitted.")

    registry.close()

    if not rows:
        raise SystemExit("No datasets remain after detect quality filtering.")

    zarr_paths = [Path(row["zarr_path"]) for row in rows]

    if quality_gate_active:
        for row in rows:
            dataset_id = str(row["dataset_id"])
            zarr_path = Path(str(row["zarr_path"]))
            quality_row = selected_quality_rows_by_dataset.get(dataset_id)
            if quality_row is None:
                raise ValueError(
                    f"{zarr_path.name}: missing detect_quality row after SQL selection for dataset_id '{dataset_id}'."
                )
            expected_refined_run = _decode_attr(quality_row["refined_run"])
            if expected_refined_run is None:
                raise ValueError(f"{zarr_path.name}: detect_quality row missing refined_run.")
            observed = _resolve_detect_quality_from_zarr(
                zarr_path,
                expected_refined_run=expected_refined_run,
            )
            expected_source_run = _decode_attr(quality_row.get("source_detect_run"))
            if observed["source_detect_run"] != expected_source_run:
                raise ValueError(
                    f"{zarr_path.name}: source_detect_run divergence for refined run '{expected_refined_run}' "
                    f"(registry={expected_source_run}, zarr={observed['source_detect_run']})."
                )
            expected_state = _decode_attr(quality_row.get("review_state"))
            expected_use = _decode_attr(quality_row.get("review_intended_use"))
            if observed["review_state"] != expected_state or observed["review_intended_use"] != expected_use:
                raise ValueError(
                    f"{zarr_path.name}: review metadata divergence for refined run '{expected_refined_run}' "
                    f"(registry={expected_state}/{expected_use}, "
                    f"zarr={observed['review_state']}/{observed['review_intended_use']})."
                )
            expected_resolved_group = _decode_attr(quality_row.get("review_resolved_group"))
            if (
                expected_resolved_group is not None
                and observed["review_resolved_group"] != expected_resolved_group
            ):
                raise ValueError(
                    f"{zarr_path.name}: resolved detect group divergence for refined run '{expected_refined_run}' "
                    f"(registry={expected_resolved_group}, zarr={observed['review_resolved_group']})."
                )

            for field in ("total_detections", "real_detections", "interpolated_detections"):
                expected_value = _as_int(quality_row.get(field))
                if expected_value is not None and _as_int(observed.get(field)) != expected_value:
                    raise ValueError(
                        f"{zarr_path.name}: {field} divergence for refined run '{expected_refined_run}' "
                        f"(registry={expected_value}, zarr={_as_int(observed.get(field))})."
                    )
            expected_interp_rate = _as_float(quality_row.get("interpolated_detections_rate"))
            observed_interp_rate = _as_float(observed.get("interpolated_detections_rate"))
            if not _rate_matches(expected_interp_rate, observed_interp_rate):
                raise ValueError(
                    f"{zarr_path.name}: interpolated_detections_rate divergence for refined run "
                    f"'{expected_refined_run}' (registry={expected_interp_rate}, zarr={observed_interp_rate})."
                )

            expected_mtime_ns = _as_int(quality_row.get("zarr_mtime_ns"))
            if expected_mtime_ns is not None and observed["zarr_mtime_ns"] != expected_mtime_ns:
                raise ValueError(
                    f"{zarr_path.name}: detect_quality row is stale for filesystem mtime "
                    f"(registry={expected_mtime_ns}, actual={observed['zarr_mtime_ns']})."
                )

    if args.set_name is None and args.out_config is None:
        args.set_name = _default_set_name(args, rows, model_input=model_input)
        print(f"Auto set-name: {args.set_name}")

    if args.out_config is None:
        args.out_config, inferred_manifest = _resolve_preflight_output_paths(args)
        print(f"Auto out-config: {args.out_config}")
        if args.out_manifest is None:
            args.out_manifest = inferred_manifest
            print(f"Auto out-manifest: {args.out_manifest}")
    elif args.out_manifest is None:
        args.out_manifest = Path(args.out_config).with_suffix(".manifest.json")
        print(f"Auto out-manifest: {args.out_manifest}")

    if args.output_file_list:
        args.output_file_list.parent.mkdir(parents=True, exist_ok=True)
        args.output_file_list.write_text(
            "\n".join([str(p) for p in zarr_paths]) + "\n", encoding="utf-8"
        )
        print(f"Wrote {len(zarr_paths)} paths to {args.output_file_list}")

    print(f"Registry query matched {len(zarr_paths)} dataset(s).")

    cli: List[str] = [str(p) for p in zarr_paths]
    _add_arg(cli, "--source-type", args.source_type)
    _add_arg(cli, "--input-format", args.input_format)
    _add_arg(cli, "--base-config", args.base_config)
    _add_arg(cli, "--out-config", args.out_config)
    _add_arg(cli, "--out-manifest", args.out_manifest)
    _add_arg(cli, "--set-name", args.set_name)
    _add_arg(cli, "--set-version", args.set_version)
    _add_arg(cli, "--project", args.project)
    _add_arg(cli, "--run-name", args.run_name)
    _add_arg(cli, "--imgsz", args.imgsz)
    _add_arg(cli, "--provenance-policy", args.provenance_policy)
    _add_arg(cli, "--metadata-json", args.metadata_json)
    if args.allow_source_mismatch:
        cli.append("--allow-source-mismatch")
    if args.allow_unapproved:
        cli.append("--allow-unapproved")
    if args.no_prefer_manual:
        cli.append("--no-prefer-manual")
    if args.dry_run:
        cli.append("--dry-run")
    if args.registry and not args.register:
        cli.extend(["--registry", str(args.registry)])
    if args.register:
        cli.append("--register")
        reg_path = args.register_registry or args.registry
        if reg_path:
            cli.extend(["--registry", str(reg_path)])

    pdt.main(cli)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
