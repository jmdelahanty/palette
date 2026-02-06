#!/usr/bin/env python3
"""Query the registry and build a detection training manifest/config."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from fisheye.diagnostics import prepare_detect_training as pdt
from fisheye.registry.db import Registry, RegistryPaths
from fisheye.utils import export_detect_training_zarr as export_zarr


def _add_arg(argv: List[str], flag: str, value: Optional[object]) -> None:
    if value is None:
        return
    argv.extend([flag, str(value)])


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
    parser.add_argument("--set-name", type=str)
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
        "--merge-include-rgb",
        action="store_true",
        help="Also export images_ds_rgb in merged mode (requires RGB arrays in all sources).",
    )
    parser.add_argument(
        "--merge-overwrite",
        action="store_true",
        help="Allow overwrite of an existing merged output Zarr.",
    )

    args = parser.parse_args(argv)
    if args.export_merged and args.dry_run:
        raise SystemExit("--export-merged cannot be combined with --dry-run (no manifest is written).")
    if args.export_merged and args.out_manifest is None:
        raise SystemExit("--export-merged requires --out-manifest so export step has a concrete manifest path.")

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
    if args.register:
        cli.append("--register")
        reg_path = args.register_registry or args.registry
        if reg_path:
            cli.extend(["--registry", str(reg_path)])

    pdt.main(cli)
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
        ]
        _add_arg(export_cli, "--registry", registry_path)
        _add_arg(export_cli, "--out-zarr", args.merge_out_zarr)
        _add_arg(export_cli, "--out-dir", args.merge_out_dir)
        if args.merge_include_rgb:
            export_cli.append("--include-rgb")
        if args.merge_overwrite:
            export_cli.append("--overwrite")
        export_zarr.main(export_cli)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
