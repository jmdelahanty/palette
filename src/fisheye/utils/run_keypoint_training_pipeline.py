#!/usr/bin/env python3
"""Run pose pipeline from registry query through optional train."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Callable, List, Optional

from fisheye.registry.db import Registry, RegistryPaths
from fisheye.utils import export_keypoint_training_zarr as export_zarr
from fisheye.utils import prepare_keypoint_training_from_registry as prepare_from_registry


def _add_arg(argv: List[str], flag: str, value: Optional[object]) -> None:
    if value is None:
        return
    argv.extend([flag, str(value)])


def _resolve_preflight_output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.out_config is not None:
        config_path = Path(args.out_config)
        manifest_path = Path(args.out_manifest) if args.out_manifest is not None else config_path.with_suffix(".manifest.json")
        return config_path, manifest_path

    if args.set_name is None:
        raise SystemExit("Unable to infer output paths: set-name/out-config unresolved.")

    safe_name = prepare_from_registry._sanitize_name(args.set_name)
    dataset_root = prepare_from_registry._resolve_dataset_root("pose")
    if args.set_version is None:
        args.set_version = prepare_from_registry._next_version_from_dataset_root(
            set_name=safe_name,
            task_prefix="pose",
            dataset_root=dataset_root,
        )
    version = int(args.set_version)
    if version < 1:
        raise SystemExit("--set-version must be >= 1")

    set_id = prepare_from_registry._build_training_set_id(safe_name, version)
    out_dir = dataset_root / set_id
    config_path = out_dir / f"{set_id}.yaml"
    manifest_path = Path(args.out_manifest) if args.out_manifest is not None else (out_dir / f"{set_id}.manifest.json")
    return config_path, manifest_path


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
    out_dir = (
        Path(merge_out_dir)
        if merge_out_dir is not None
        else (prepare_from_registry._resolve_dataset_root("pose") / set_id)
    )
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
    export_onnx: bool,
    onnx_opset: Optional[int],
    onnx_simplify: bool,
    onnx_path: Optional[str],
    export_trt: bool,
    trt_precision: Optional[str],
    trtexec: Optional[str],
    trt_cuda_graph: bool,
    trt_profiling: bool,
    trt_verbose: bool,
) -> int:
    if not config_path.exists():
        raise SystemExit(f"Training config not found: {config_path}")
    if not manifest_path.exists():
        raise SystemExit(f"Training manifest not found: {manifest_path}")

    cmd: List[str] = [
        sys.executable,
        "-m",
        "fisheye.training.train_pose",
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
    if export_onnx:
        cmd.append("--export-onnx")
    _add_arg(cmd, "--onnx-opset", onnx_opset)
    if onnx_simplify:
        cmd.append("--onnx-simplify")
    _add_arg(cmd, "--onnx-path", onnx_path)
    if export_trt:
        cmd.append("--export-trt")
    _add_arg(cmd, "--trt-precision", trt_precision)
    _add_arg(cmd, "--trtexec", trtexec)
    if trt_cuda_graph:
        cmd.append("--trt-cuda-graph")
    if trt_profiling:
        cmd.append("--trt-profiling")
    if trt_verbose:
        cmd.append("--trt-verbose")
    print("Launching training: " + " ".join(cmd))
    completed = subprocess.run(cmd, check=False)
    return int(completed.returncode)


def _load_keypoint_data_card_main() -> tuple[Optional[Callable[[Optional[List[str]]], int]], Optional[str]]:
    try:
        module = importlib.import_module("fisheye.utils.aggregate_keypoint_training_data_card")
    except Exception as exc:  # pragma: no cover - exercised when module is not yet present
        return None, f"{exc.__class__.__name__}: {exc}"
    main_fn = getattr(module, "main", None)
    if not callable(main_fn):
        return None, "module does not define callable main(argv)."
    return main_fn, None


def _run_keypoint_data_card_aggregation(*, cli: List[str], required: bool) -> int:
    main_fn, load_error = _load_keypoint_data_card_main()
    if main_fn is None:
        if required:
            raise SystemExit(
                "Keypoint training data-card aggregation requested, but "
                "fisheye.utils.aggregate_keypoint_training_data_card is unavailable "
                f"({load_error})."
            )
        print(
            "Skipping keypoint training data-card aggregation: "
            f"aggregator unavailable ({load_error})."
        )
        return 0
    try:
        result = main_fn(cli)
    except SystemExit as exc:  # pragma: no cover - defensive wrapper around argparse exits
        rc = int(exc.code) if isinstance(exc.code, int) else 1
        if required:
            return rc
        print(
            "Skipping keypoint training data-card aggregation: "
            f"aggregator exited with code {rc}."
        )
        return 0
    rc = int(result) if result is not None else 0
    if rc != 0 and not required:
        print(
            "Skipping keypoint training data-card aggregation: "
            f"aggregator returned code {rc}."
        )
        return 0
    return rc


def _build_data_card_cli(
    *,
    manifest_path: Path,
    registry_path: Path,
    args: argparse.Namespace,
) -> List[str]:
    card_cli: List[str] = [
        "--manifest",
        str(manifest_path),
        "--registry",
        str(registry_path),
        "--split",
        str(args.data_card_split),
    ]
    _add_arg(card_cli, "--output", args.data_card_output)
    if args.data_card_no_plots:
        card_cli.append("--no-plots")
    _add_arg(card_cli, "--plot-dir", args.data_card_plot_dir)
    _add_arg(card_cli, "--plot-prefix", args.data_card_plot_prefix)
    _add_arg(card_cli, "--plot-heatmap-bin-factor", args.data_card_plot_heatmap_bin_factor)
    if args.data_card_allow_profile_mtime_mismatch:
        card_cli.append("--allow-profile-mtime-mismatch")
    if args.data_card_allow_profile_fallback_scan:
        card_cli.append("--allow-profile-fallback-scan")
    if args.data_card_view:
        card_cli.append("--view")
    if args.data_card_force_plots:
        card_cli.append("--force")
    return card_cli


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

    parser.add_argument(
        "--source-type",
        choices=list(prepare_from_registry.KEYPOINT_SOURCE_TYPE_CHOICES),
        default=prepare_from_registry.DEFAULT_KEYPOINT_SOURCE_TYPE,
        help=(
            "Requested crop/detect source label for the training set "
            f"(default: {prepare_from_registry.DEFAULT_KEYPOINT_SOURCE_TYPE})."
        ),
    )
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
        help="Require the effective keypoint annotation source to match this skeleton_id.",
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
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run fisheye.training.train_pose after successful preflight.",
    )
    parser.add_argument(
        "--export-merged",
        action="store_true",
        help="After preflight manifest generation, export one merged keypoint training Zarr.",
    )
    parser.add_argument(
        "--merge-out-zarr",
        type=Path,
        help="Output merged .zarr path (passed to export_keypoint_training_zarr --out-zarr).",
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
        help="Row-copy batch size for merged export.",
    )
    parser.add_argument(
        "--merge-row-gate-policy",
        choices=["auto", "refined_usable", "raw_success", "raw_success_plus_box_only"],
        default="auto",
        help=(
            "Row inclusion policy for merged pose export. "
            "auto prefers refined usable_keypoints when present; "
            "raw_success_plus_box_only also includes fish_present_no_keypoints rows as box-only labels."
        ),
    )
    parser.add_argument(
        "--merge-overwrite",
        action="store_true",
        help="Allow overwrite of an existing merged output Zarr.",
    )
    parser.add_argument(
        "--aggregate-training-data-card",
        action="store_true",
        help="Aggregate keypoint training data card from the preflight manifest after successful prepare/export.",
    )
    parser.add_argument(
        "--no-aggregate-training-data-card",
        action="store_true",
        help="Disable automatic data-card aggregation when --export-merged is used.",
    )
    parser.add_argument(
        "--data-card-output",
        type=Path,
        help="Optional output JSON path for keypoint training data-card aggregation.",
    )
    parser.add_argument(
        "--data-card-split",
        type=str,
        default="train",
        help="Split label for data-card selection metadata (default: train).",
    )
    parser.add_argument(
        "--data-card-no-plots",
        action="store_true",
        help="Skip plot PNG generation when aggregating the keypoint training data card.",
    )
    parser.add_argument(
        "--data-card-plot-dir",
        type=Path,
        help="Optional output directory for keypoint training data-card plots.",
    )
    parser.add_argument(
        "--data-card-plot-prefix",
        type=str,
        help="Optional filename prefix for keypoint training data-card plots.",
    )
    parser.add_argument(
        "--data-card-plot-heatmap-bin-factor",
        type=int,
        default=2,
        help="Coarsening factor for keypoint heatmap bins in data-card plots (default: 2).",
    )
    parser.add_argument(
        "--data-card-allow-profile-mtime-mismatch",
        action="store_true",
        help=(
            "Allow keypoint_data_profile_latest mtime mismatches during keypoint "
            "training data-card aggregation."
        ),
    )
    parser.add_argument(
        "--data-card-allow-profile-fallback-scan",
        action="store_true",
        help=(
            "Allow keypoint data-card aggregation to scan Zarr directly when "
            "keypoint_data_profile_latest rows are missing."
        ),
    )
    parser.add_argument(
        "--data-card-view",
        action="store_true",
        help="Open keypoint data-card plots after aggregation (uses existing plots unless forced).",
    )
    parser.add_argument(
        "--data-card-force-plots",
        action="store_true",
        help="Regenerate keypoint data-card plots even when plot files already exist.",
    )
    parser.add_argument("--export-onnx", action="store_true", help="Export ONNX after --train.")
    parser.add_argument("--onnx-opset", type=int, default=13, help="ONNX opset for export.")
    parser.add_argument("--onnx-simplify", action="store_true", help="Run ONNX simplification after export.")
    parser.add_argument("--onnx-path", type=str, help="Optional existing ONNX path to reuse.")
    parser.add_argument("--export-trt", action="store_true", help="Export TensorRT after --train (implies ONNX).")
    parser.add_argument(
        "--trt-precision",
        choices=["fp16", "int8"],
        default="fp16",
        help="TensorRT precision (default: fp16 when --export-trt is set).",
    )
    parser.add_argument("--trtexec", type=str, help="Optional trtexec binary path.")
    parser.add_argument("--trt-cuda-graph", action="store_true", help="Enable TensorRT CUDA graph.")
    parser.add_argument("--trt-profiling", action="store_true", help="Enable TensorRT profiling outputs.")
    parser.add_argument("--trt-verbose", action="store_true", help="Enable verbose TensorRT build logs.")
    parser.add_argument("--register", action="store_true", help="Record training set metadata in the registry.")
    parser.add_argument(
        "--register-registry",
        type=Path,
        help="Registry path to use when --register is set (defaults to --registry).",
    )

    args = parser.parse_args(argv)
    if args.export_merged and args.dry_run:
        raise SystemExit("--export-merged cannot be combined with --dry-run (no manifest is written).")
    if args.train and args.dry_run:
        raise SystemExit("--train cannot be combined with --dry-run.")
    if args.aggregate_training_data_card and args.dry_run:
        raise SystemExit("--aggregate-training-data-card cannot be combined with --dry-run.")
    if args.register_registry and not args.register:
        raise SystemExit("--register-registry requires --register.")
    if args.aggregate_training_data_card and args.no_aggregate_training_data_card:
        raise SystemExit(
            "--aggregate-training-data-card cannot be combined with --no-aggregate-training-data-card."
        )
    if args.data_card_plot_heatmap_bin_factor < 1:
        parser.error("--data-card-plot-heatmap-bin-factor must be >= 1.")
    if args.data_card_view and args.data_card_no_plots:
        parser.error("--data-card-view cannot be combined with --data-card-no-plots.")
    if args.data_card_force_plots and args.data_card_no_plots:
        parser.error("--data-card-force-plots cannot be combined with --data-card-no-plots.")

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

    zarr_paths = [Path(row["zarr_path"]) for row in rows]
    if not zarr_paths:
        raise SystemExit("Registry query returned no datasets.")

    if args.set_name is None and args.out_config is None:
        args.set_name = prepare_from_registry._default_set_name(args, rows, model_input=model_input)
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
    preflight_manifest_path = Path(args.out_manifest)
    explicit_data_card_requested = bool(args.aggregate_training_data_card)
    auto_data_card_requested = bool(args.export_merged and not args.no_aggregate_training_data_card)
    should_aggregate_data_card = bool(explicit_data_card_requested or auto_data_card_requested)
    if should_aggregate_data_card and auto_data_card_requested and not explicit_data_card_requested:
        print(
            "Auto enabling keypoint training data-card aggregation for merged export. "
            "Use --no-aggregate-training-data-card to disable."
        )

    prepare_cli: List[str] = []
    _add_arg(prepare_cli, "--dish-design", args.dish_design)
    _add_arg(prepare_cli, "--dish-design-like", args.dish_design_like)
    _add_arg(prepare_cli, "--fps-min", args.fps_min)
    _add_arg(prepare_cli, "--fps-max", args.fps_max)
    _add_arg(prepare_cli, "--exposure-min", args.exposure_min)
    _add_arg(prepare_cli, "--exposure-max", args.exposure_max)
    _add_arg(prepare_cli, "--frame-rate-min", args.frame_rate_min)
    _add_arg(prepare_cli, "--frame-rate-max", args.frame_rate_max)
    _add_arg(prepare_cli, "--gain-min", args.gain_min)
    _add_arg(prepare_cli, "--gain-max", args.gain_max)
    _add_arg(prepare_cli, "--video-codec", args.video_codec)
    _add_arg(prepare_cli, "--video-pix-fmt", args.video_pix_fmt)
    _add_arg(prepare_cli, "--format-encoder", args.format_encoder)
    _add_arg(prepare_cli, "--format-title", args.format_title)
    _add_arg(prepare_cli, "--format-comment", args.format_comment)
    _add_arg(prepare_cli, "--encoder-name", args.encoder_name)
    _add_arg(prepare_cli, "--encoder-codec", args.encoder_codec)
    _add_arg(prepare_cli, "--encoder-preset", args.encoder_preset)
    _add_arg(prepare_cli, "--encoder-tuning", args.encoder_tuning)
    _add_arg(prepare_cli, "--encoder-rc", args.encoder_rc)
    _add_arg(prepare_cli, "--compression", args.compression)
    _add_arg(prepare_cli, "--camera-model", args.camera_model)
    _add_arg(prepare_cli, "--camera-serial", args.camera_serial)
    _add_arg(prepare_cli, "--camera-id", args.camera_id)
    _add_arg(prepare_cli, "--rig-id", args.rig_id)
    _add_arg(prepare_cli, "--arena-id", args.arena_id)
    _add_arg(prepare_cli, "--path-contains", args.path_contains)
    _add_arg(prepare_cli, "--limit", args.limit)
    _add_arg(prepare_cli, "--source-type", args.source_type)
    _add_arg(prepare_cli, "--input-format", args.input_format)
    _add_arg(prepare_cli, "--model-input", args.model_input)
    _add_arg(prepare_cli, "--keypoint-run", args.keypoint_run)
    _add_arg(prepare_cli, "--skeleton-id", args.skeleton_id)
    _add_arg(prepare_cli, "--base-config", args.base_config)
    _add_arg(prepare_cli, "--out-config", args.out_config)
    _add_arg(prepare_cli, "--out-manifest", args.out_manifest)
    _add_arg(prepare_cli, "--set-name", args.set_name)
    _add_arg(prepare_cli, "--set-version", args.set_version)
    _add_arg(prepare_cli, "--project", args.project)
    _add_arg(prepare_cli, "--run-name", args.run_name)
    _add_arg(prepare_cli, "--imgsz", args.imgsz)
    _add_arg(prepare_cli, "--min-usable-keypoints-rate", args.min_usable_keypoints_rate)
    _add_arg(prepare_cli, "--require-review-state", args.require_review_state)
    _add_arg(prepare_cli, "--require-review-intended-use", args.require_review_intended_use)
    if args.allow_cross_method_review_fallback:
        prepare_cli.append("--allow-cross-method-review-fallback")
    if args.dry_run:
        prepare_cli.append("--dry-run")
    if args.registry:
        prepare_cli.extend(["--registry", str(args.registry)])
    if args.register:
        prepare_cli.append("--register")
        _add_arg(prepare_cli, "--register-registry", args.register_registry)

    prepare_from_registry.main(prepare_cli)

    if args.export_merged:
        if not preflight_manifest_path.exists():
            raise SystemExit(f"Expected manifest not found after preflight: {preflight_manifest_path}")

        export_cli: List[str] = [
            "--manifest",
            str(preflight_manifest_path),
            "--merge",
            "--split",
            str(args.merge_split),
            "--seed",
            str(args.merge_seed),
            "--copy-batch-size",
            str(args.merge_copy_batch_size),
            "--row-gate-policy",
            str(args.merge_row_gate_policy),
        ]
        export_registry = args.register_registry or registry_path
        _add_arg(export_cli, "--registry", export_registry)
        _add_arg(export_cli, "--out-zarr", args.merge_out_zarr)
        _add_arg(export_cli, "--out-dir", args.merge_out_dir)
        # Pipeline controls data-card orchestration below; avoid duplicate aggregation in exporter.
        export_cli.append("--no-aggregate-training-data-card")
        if args.merge_overwrite:
            export_cli.append("--overwrite")
        export_zarr.main(export_cli)
        if should_aggregate_data_card:
            card_cli = _build_data_card_cli(
                manifest_path=preflight_manifest_path,
                registry_path=Path(registry_path),
                args=args,
            )
            card_rc = _run_keypoint_data_card_aggregation(
                cli=card_cli,
                required=explicit_data_card_requested,
            )
            if card_rc != 0:
                return int(card_rc)

        if args.train:
            merged_config, merged_manifest = _resolve_merged_training_paths(
                preflight_manifest_path=preflight_manifest_path,
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
                export_onnx=args.export_onnx,
                onnx_opset=args.onnx_opset,
                onnx_simplify=args.onnx_simplify,
                onnx_path=args.onnx_path,
                export_trt=args.export_trt,
                trt_precision=args.trt_precision,
                trtexec=args.trtexec,
                trt_cuda_graph=args.trt_cuda_graph,
                trt_profiling=args.trt_profiling,
                trt_verbose=args.trt_verbose,
            )
    else:
        if should_aggregate_data_card:
            if not preflight_manifest_path.exists():
                raise SystemExit(
                    f"--aggregate-training-data-card requires manifest output: {preflight_manifest_path}"
                )
            card_cli = _build_data_card_cli(
                manifest_path=preflight_manifest_path,
                registry_path=Path(registry_path),
                args=args,
            )
            card_rc = _run_keypoint_data_card_aggregation(
                cli=card_cli,
                required=explicit_data_card_requested,
            )
            if card_rc != 0:
                return int(card_rc)
        if not args.train:
            return 0

        train_config = Path(args.out_config)
        train_manifest = Path(args.out_manifest)
        effective_set_id = _require_manifest_set_id(train_manifest)
        train_registry = args.register_registry or registry_path
        return _run_training(
            config_path=train_config,
            manifest_path=train_manifest,
            set_id=effective_set_id,
            registry_path=Path(train_registry),
            run_name=args.run_name,
            project=args.project,
            export_onnx=args.export_onnx,
            onnx_opset=args.onnx_opset,
            onnx_simplify=args.onnx_simplify,
            onnx_path=args.onnx_path,
            export_trt=args.export_trt,
            trt_precision=args.trt_precision,
            trtexec=args.trtexec,
            trt_cuda_graph=args.trt_cuda_graph,
            trt_profiling=args.trt_profiling,
            trt_verbose=args.trt_verbose,
        )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
