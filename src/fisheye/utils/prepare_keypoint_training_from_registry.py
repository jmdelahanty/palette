#!/usr/bin/env python3
"""Query the registry and build a keypoint (pose) training manifest/config."""

from __future__ import annotations

import argparse
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
from fisheye.training.config import PoseConfig
from fisheye.utils.system import build_invocation_record
from fisheye.utils.zarr_metadata import get_downsample_array_path, get_downsample_shape


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
        "source_type": args.source_type,
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
            _slug_component(args.source_type, fallback="source"),
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


def _decode_attr(value: Any) -> Optional[str]:
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


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


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
        "source_type": args.source_type,
        "input_format": args.input_format,
        "model_input": model_input,
        "keypoint_run": args.keypoint_run,
        "allow_source_mismatch": bool(args.allow_source_mismatch),
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
        print(f"  Keypoint run: {dataset.get('keypoint_run_resolved')}")
        print(f"  Keypoint rows: {dataset.get('keypoints_total')}")
        if dataset.get("keypoints_successful") is not None:
            print(f"  Successful keypoints: {dataset['keypoints_successful']}")
        if dataset.get("keypoints_success_rate") is not None:
            print(f"  Success rate: {dataset['keypoints_success_rate']:.3f}")
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

    parser.add_argument("--source-type", choices=["detect", "filtered", "interpolated", "manual"], default="filtered")
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
    parser.add_argument("--allow-source-mismatch", action="store_true")
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
    registry.close()

    if not rows:
        raise SystemExit("Registry query returned no datasets.")

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
    elif args.set_name:
        print("Note: --set-name ignored because --out-config was provided.")

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

    seen_names: set[str] = set()
    dataset_entries: Dict[str, Dict[str, Any]] = {}
    manifest_datasets: List[Dict[str, Any]] = []
    dataset_ids: List[str] = []

    for idx, row in enumerate(rows, start=1):
        zarr_path = Path(row["zarr_path"])
        root = zarr.open_group(str(zarr_path), mode="r")
        array_path = get_downsample_array_path(root, format_hint=args.input_format)
        if array_path is None:
            raise ValueError(
                f"{zarr_path.name}: downsample data for input_format '{args.input_format}' is missing."
            )
        ds_shape = get_downsample_shape(root, format_hint=args.input_format)
        if ds_shape is None:
            raise ValueError(f"{zarr_path.name}: unable to resolve downsample shape for '{args.input_format}'.")

        keypoint_run_resolved, keypoint_selector = _resolve_keypoint_run(root, args.keypoint_run)
        kp_group = root["keypoints_runs"][keypoint_run_resolved]
        if "keypoints_roi" not in kp_group:
            raise ValueError(f"{zarr_path.name}: keypoint run '{keypoint_run_resolved}' missing keypoints_roi array.")
        keypoints_total = int(kp_group["keypoints_roi"].shape[0])

        crop_parent = root.get("crop_runs")
        if crop_parent is None:
            raise ValueError(f"{zarr_path.name}: missing crop_runs group.")
        latest_crop = _decode_attr(crop_parent.attrs.get("latest"))
        if not latest_crop:
            raise ValueError(f"{zarr_path.name}: crop_runs group missing 'latest' attribute.")

        warnings: List[str] = []
        source_crop_run = _decode_attr(kp_group.attrs.get("source_crop_run"))
        if source_crop_run and source_crop_run not in crop_parent:
            warnings.append(
                f"keypoint run source_crop_run '{source_crop_run}' not found in crop_runs; falling back to latest '{latest_crop}'."
            )
            source_crop_run = None
        if source_crop_run is None:
            source_crop_run = latest_crop
            warnings.append(
                "keypoint run missing source_crop_run; using crop_runs/latest."
            )
        if source_crop_run not in crop_parent:
            raise ValueError(f"{zarr_path.name}: source crop run '{source_crop_run}' not found in crop_runs.")

        crop_group = crop_parent[source_crop_run]
        if "roi_images" not in crop_group:
            raise ValueError(f"{zarr_path.name}: crop run '{source_crop_run}' missing roi_images array.")
        roi_total = int(crop_group["roi_images"].shape[0])
        if roi_total != keypoints_total:
            warnings.append(
                f"roi/keypoint row mismatch (roi_images={roi_total}, keypoints_roi={keypoints_total})."
            )

        source_type_resolved = _decode_attr(crop_group.attrs.get("detection_source_type")) or "detect"
        source_type_resolved = source_type_resolved.lower()
        if source_type_resolved != args.source_type:
            message = (
                f"{zarr_path.name}: requested source_type '{args.source_type}' but source crop run "
                f"'{source_crop_run}' reports '{source_type_resolved}'."
            )
            if not args.allow_source_mismatch:
                raise ValueError(f"{message} Re-run with --allow-source-mismatch to proceed.")
            warnings.append(message)

        keypoints_successful: Optional[int] = None
        keypoints_success_rate = _as_float(kp_group.attrs.get("success_rate"))
        keypoints_processed = _as_int(kp_group.attrs.get("keypoints_processed"))
        if keypoints_success_rate is not None:
            denominator = keypoints_processed if keypoints_processed is not None else keypoints_total
            keypoints_successful = int(round(keypoints_success_rate * float(denominator)))
        elif "detection_success" in kp_group:
            success_arr = kp_group["detection_success"]
            if success_arr.shape[0] != keypoints_total:
                warnings.append(
                    f"detection_success row mismatch (detection_success={success_arr.shape[0]}, keypoints_roi={keypoints_total})."
                )
            else:
                # Reading one boolean vector is cheap and gives an exact success count.
                keypoints_successful = int(success_arr[:].sum())
                keypoints_success_rate = _format_ratio(keypoints_successful, keypoints_total)

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
                "dish_design": row["dish_design"],
                "canvas_name": row["canvas_name"],
                "source_type_requested": args.source_type,
                "source_type_resolved": source_type_resolved,
                "source_crop_run": source_crop_run,
                "input_format": args.input_format,
                "images_ds_shape": [int(ds_shape[0]), int(ds_shape[1])],
                "keypoint_run_requested": args.keypoint_run,
                "keypoint_run_selector": keypoint_selector,
                "keypoint_run_resolved": keypoint_run_resolved,
                "keypoints_array_path": f"keypoints_runs/{keypoint_run_resolved}/keypoints_roi",
                "detection_success_path": (
                    f"keypoints_runs/{keypoint_run_resolved}/detection_success"
                    if "detection_success" in kp_group
                    else None
                ),
                "keypoints_total": keypoints_total,
                "keypoints_successful": keypoints_successful,
                "keypoints_success_rate": keypoints_success_rate,
                "warnings": warnings,
            }
        )
        if row["dataset_id"]:
            dataset_ids.append(str(row["dataset_id"]))

    base_config["datasets"] = dataset_entries
    base_config["task"] = "pose"
    base_config["allow_source_mismatch"] = bool(args.allow_source_mismatch)
    if args.imgsz is not None:
        base_config["training_params"]["imgsz"] = int(args.imgsz)
    if args.project:
        base_config["training_params"]["project"] = str(Path(args.project).expanduser().resolve())

    PoseConfig.model_validate(base_config)
    manifest_imgsz = _imgsz_list(base_config["training_params"]["imgsz"])

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

    manifest_payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "pose",
        "source_type": args.source_type,
        "input_format": args.input_format,
        "imgsz": manifest_imgsz,
        "datasets": manifest_datasets,
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
        "invocation": invocation_payload,
    }

    _print_summary(
        source_type=args.source_type,
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
            register_registry.upsert_training_set(
                set_id=set_id,
                name=resolved_set_name,
                query_filter=query_filter_payload,
                dataset_ids=dataset_ids,
                invocation=invocation_payload,
            )
            register_registry.close()
            print(f"Recorded training set: {set_id}")

    cli: List[str] = ["python", "-m", "fisheye.training.train_keypoints", str(args.out_config)]
    _add_arg(cli, "--manifest", out_manifest)
    _add_arg(cli, "--run-name", args.run_name)
    _add_arg(cli, "--set-id", set_id)
    _add_arg(cli, "--registry", registry_path)
    print("Next: " + " ".join(str(token) for token in cli))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
