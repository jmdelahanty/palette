#!/usr/bin/env python3
"""Run detection pipeline from registry query through optional merge and train."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import List, Optional

from fisheye.diagnostics import prepare_detect_training as pdt
from fisheye.registry.db import Registry, RegistryPaths
from fisheye.utils import export_detect_training_zarr as export_zarr
from fisheye.utils import prepare_detect_training_from_registry as prepare_from_registry


def _add_arg(argv: List[str], flag: str, value: Optional[object]) -> None:
    if value is None:
        return
    argv.extend([flag, str(value)])


def _normalize_manifest_stem(value: str) -> str:
    text = str(value).strip()
    while text.endswith(".manifest"):
        text = text[: -len(".manifest")]
    return text


def _resolve_merged_training_paths(
    *,
    preflight_manifest_path: Path,
    merge_out_dir: Optional[Path],
) -> tuple[Path, Path]:
    payload = json.loads(preflight_manifest_path.read_text(encoding="utf-8"))
    set_id_source = str(payload.get("set_id")).strip() if payload.get("set_id") else preflight_manifest_path.stem
    set_id = _normalize_manifest_stem(set_id_source) or "training_set"
    out_dir = Path(merge_out_dir) if merge_out_dir is not None else (prepare_from_registry._resolve_default_dataset_root() / set_id)
    return out_dir / f"{set_id}.yaml", out_dir / f"{set_id}.manifest.json"


def _require_manifest_set_id(manifest_path: Path) -> str:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive path
        raise SystemExit(f"--train requires a readable manifest JSON with set_id: {manifest_path} ({exc})")
    raw = payload.get("set_id") if isinstance(payload, dict) else None
    set_id = str(raw).strip() if raw is not None else ""
    if not set_id:
        raise SystemExit(
            "--train requires manifest set_id to avoid unlinked runs. "
            "Use --set-name (recommended) so preflight writes set_id."
        )
    return set_id


def _run_training(
    *,
    config_path: Path,
    manifest_path: Path,
    set_id: str,
    registry_path: Path,
    run_name: Optional[str],
    project: Optional[str],
    allow_source_mismatch: bool,
    export_onnx: bool,
    export_trt: bool,
    onnx_opset: Optional[int],
    onnx_simplify: bool,
    onnx_path: Optional[str],
    nms_conf: Optional[float],
    nms_iou: Optional[float],
    nms_topk: Optional[int],
    trt_precision: Optional[str],
    trtexec: Optional[str],
    trt_cuda_graph: bool,
    trt_profiling: bool,
    trt_verbose: bool,
    profile: bool,
) -> int:
    if not config_path.exists():
        raise SystemExit(f"Training config not found: {config_path}")
    if not manifest_path.exists():
        raise SystemExit(f"Training manifest not found: {manifest_path}")

    cmd: List[str] = [
        sys.executable,
        "-m",
        "fisheye.training.train_detection",
        str(config_path),
        "--manifest",
        str(manifest_path),
        "--set-id",
        str(set_id),
        "--registry",
        str(registry_path),
    ]
    _add_arg(cmd, "--run-name", run_name)
    _add_arg(cmd, "--project", project)
    if allow_source_mismatch:
        cmd.append("--allow-source-mismatch")
    if export_onnx:
        cmd.append("--export-onnx")
    if export_trt:
        cmd.append("--export-trt")
    _add_arg(cmd, "--onnx-opset", onnx_opset)
    if onnx_simplify:
        cmd.append("--onnx-simplify")
    _add_arg(cmd, "--onnx-path", onnx_path)
    _add_arg(cmd, "--nms-conf", nms_conf)
    _add_arg(cmd, "--nms-iou", nms_iou)
    _add_arg(cmd, "--nms-topk", nms_topk)
    effective_trt_precision = trt_precision or ("fp16" if export_trt else None)
    _add_arg(cmd, "--trt-precision", effective_trt_precision)
    _add_arg(cmd, "--trtexec", trtexec)
    if trt_cuda_graph:
        cmd.append("--trt-cuda-graph")
    if trt_profiling:
        cmd.append("--trt-profiling")
    if trt_verbose:
        cmd.append("--trt-verbose")
    if profile:
        cmd.append("--profile")
    print("Launching training: " + " ".join(cmd))
    completed = subprocess.run(cmd, check=False)
    return int(completed.returncode)


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
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run fisheye.training.train_detection after successful prepare/export.",
    )
    parser.add_argument("--register", action="store_true", help="Register datasets in the registry.")
    parser.add_argument(
        "--register-registry",
        type=Path,
        help="Registry path to use when --register is set (defaults to --registry).",
    )
    parser.add_argument(
        "--export-merged",
        action="store_true",
        help="After preflight manifest generation, export one merged training Zarr.",
    )
    parser.add_argument(
        "--merge-out-zarr",
        type=Path,
        help="Output merged .zarr path (passed to export_detect_training_zarr --out-zarr).",
    )
    parser.add_argument(
        "--merge-out-dir",
        type=Path,
        help="Output directory for merged config/manifest/summary artifacts.",
    )
    parser.add_argument(
        "--merge-split",
        type=str,
        default="0.8/0.2",
        help="Split ratios for merged export (train/val or train/val/test).",
    )
    parser.add_argument(
        "--merge-seed",
        type=int,
        default=42,
        help="Split seed for merged export.",
    )
    parser.add_argument(
        "--merge-copy-batch-size",
        type=int,
        default=128,
        help="Frame-copy batch size for merged export (passed to export_detect_training_zarr --copy-batch-size).",
    )
    parser.add_argument(
        "--merge-include-rgb",
        action="store_true",
        help="Also export images_ds_rgb in merged mode (requires RGB arrays in all sources).",
    )
    parser.add_argument(
        "--merge-overwrite",
        action="store_true",
        help="Allow overwrite of an existing merged output Zarr.",
    )
    parser.add_argument("--export-onnx", action="store_true", help="Export ONNX after --train.")
    parser.add_argument("--export-trt", action="store_true", help="Export TensorRT after --train.")
    parser.add_argument("--onnx-opset", type=int, help="ONNX opset for export.")
    parser.add_argument("--onnx-simplify", action="store_true", help="Run ONNX simplification after export.")
    parser.add_argument("--onnx-path", type=str, help="Optional existing ONNX path to reuse.")
    parser.add_argument("--nms-conf", type=float, help="NMS confidence threshold for export.")
    parser.add_argument("--nms-iou", type=float, help="NMS IoU threshold for export.")
    parser.add_argument("--nms-topk", type=int, help="NMS top-k for export.")
    parser.add_argument(
        "--trt-precision",
        choices=["fp16", "int8"],
        help="TensorRT precision (default: fp16 when --export-trt is set).",
    )
    parser.add_argument("--trtexec", type=str, help="Optional trtexec binary path.")
    parser.add_argument("--trt-cuda-graph", action="store_true", help="Enable TensorRT CUDA graph.")
    parser.add_argument("--trt-profiling", action="store_true", help="Enable TensorRT profiling outputs.")
    parser.add_argument("--trt-verbose", action="store_true", help="Enable verbose TensorRT build logs.")
    parser.add_argument("--profile", action="store_true", help="Enable detect input-pipeline profiling during --train.")

    args = parser.parse_args(argv)
    if args.export_merged and args.dry_run:
        raise SystemExit("--export-merged cannot be combined with --dry-run (no manifest is written).")
    if args.train and args.dry_run:
        raise SystemExit("--train cannot be combined with --dry-run.")

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
        args.set_name = prepare_from_registry._default_set_name(args, rows, model_input=model_input)
        print(f"Auto set-name: {args.set_name}")

    if args.out_config is None:
        args.out_config, inferred_manifest = prepare_from_registry._resolve_preflight_output_paths(args)
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

    prepare_cli: List[str] = [str(p) for p in zarr_paths]
    _add_arg(prepare_cli, "--source-type", args.source_type)
    _add_arg(prepare_cli, "--input-format", args.input_format)
    _add_arg(prepare_cli, "--base-config", args.base_config)
    _add_arg(prepare_cli, "--out-config", args.out_config)
    _add_arg(prepare_cli, "--out-manifest", args.out_manifest)
    _add_arg(prepare_cli, "--set-name", args.set_name)
    _add_arg(prepare_cli, "--set-version", args.set_version)
    _add_arg(prepare_cli, "--project", args.project)
    _add_arg(prepare_cli, "--run-name", args.run_name)
    _add_arg(prepare_cli, "--imgsz", args.imgsz)
    _add_arg(prepare_cli, "--provenance-policy", args.provenance_policy)
    _add_arg(prepare_cli, "--metadata-json", args.metadata_json)
    if args.allow_source_mismatch:
        prepare_cli.append("--allow-source-mismatch")
    if args.allow_unapproved:
        prepare_cli.append("--allow-unapproved")
    if args.no_prefer_manual:
        prepare_cli.append("--no-prefer-manual")
    if args.dry_run:
        prepare_cli.append("--dry-run")
    if args.registry and not args.register:
        prepare_cli.extend(["--registry", str(args.registry)])
    if args.register:
        prepare_cli.append("--register")
        reg_path = args.register_registry or args.registry
        if reg_path:
            prepare_cli.extend(["--registry", str(reg_path)])

    pdt.main(prepare_cli)
    if args.export_merged:
        manifest_path = Path(args.out_manifest)
        if not manifest_path.exists():
            raise SystemExit(f"Expected manifest not found after preflight: {manifest_path}")
        export_cli: List[str] = [
            "--manifest",
            str(manifest_path),
            "--merge",
            "--split",
            str(args.merge_split),
            "--seed",
            str(args.merge_seed),
            "--copy-batch-size",
            str(args.merge_copy_batch_size),
        ]
        _add_arg(export_cli, "--registry", registry_path)
        _add_arg(export_cli, "--out-zarr", args.merge_out_zarr)
        _add_arg(export_cli, "--out-dir", args.merge_out_dir)
        if args.merge_include_rgb:
            export_cli.append("--include-rgb")
        if args.merge_overwrite:
            export_cli.append("--overwrite")
        export_zarr.main(export_cli)
        if args.train:
            merged_config, merged_manifest = _resolve_merged_training_paths(
                preflight_manifest_path=manifest_path,
                merge_out_dir=args.merge_out_dir,
            )
            effective_set_id = _require_manifest_set_id(merged_manifest)
            train_registry = args.register_registry or registry_path
            return _run_training(
                config_path=merged_config,
                manifest_path=merged_manifest,
                set_id=effective_set_id,
                registry_path=Path(train_registry),
                run_name=args.run_name,
                project=args.project,
                allow_source_mismatch=bool(args.allow_source_mismatch),
                export_onnx=bool(args.export_onnx),
                export_trt=bool(args.export_trt),
                onnx_opset=args.onnx_opset,
                onnx_simplify=bool(args.onnx_simplify),
                onnx_path=args.onnx_path,
                nms_conf=args.nms_conf,
                nms_iou=args.nms_iou,
                nms_topk=args.nms_topk,
                trt_precision=args.trt_precision,
                trtexec=args.trtexec,
                trt_cuda_graph=bool(args.trt_cuda_graph),
                trt_profiling=bool(args.trt_profiling),
                trt_verbose=bool(args.trt_verbose),
                profile=bool(args.profile),
            )
    elif args.train:
        train_config, train_manifest = prepare_from_registry._resolve_preflight_output_paths(args)
        effective_set_id = _require_manifest_set_id(train_manifest)
        train_registry = args.register_registry or registry_path
        return _run_training(
            config_path=train_config,
            manifest_path=train_manifest,
            set_id=effective_set_id,
            registry_path=Path(train_registry),
            run_name=args.run_name,
            project=args.project,
            allow_source_mismatch=bool(args.allow_source_mismatch),
            export_onnx=bool(args.export_onnx),
            export_trt=bool(args.export_trt),
            onnx_opset=args.onnx_opset,
            onnx_simplify=bool(args.onnx_simplify),
            onnx_path=args.onnx_path,
            nms_conf=args.nms_conf,
            nms_iou=args.nms_iou,
            nms_topk=args.nms_topk,
            trt_precision=args.trt_precision,
            trtexec=args.trtexec,
            trt_cuda_graph=bool(args.trt_cuda_graph),
            trt_profiling=bool(args.trt_profiling),
            trt_verbose=bool(args.trt_verbose),
            profile=bool(args.profile),
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
