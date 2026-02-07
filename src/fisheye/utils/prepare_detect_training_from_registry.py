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

    zarr_paths = [Path(row["zarr_path"]) for row in rows]
    if not zarr_paths:
        raise SystemExit("Registry query returned no datasets.")

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
