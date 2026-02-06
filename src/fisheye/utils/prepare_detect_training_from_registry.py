#!/usr/bin/env python3
"""Query the registry and build a detection training manifest/config."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, List, Mapping, Optional, Sequence

from fisheye.diagnostics import prepare_detect_training as pdt
from fisheye.registry.db import Registry, RegistryPaths
from fisheye.utils import export_detect_training_zarr as export_zarr


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


def _normalize_manifest_stem(value: str) -> str:
    text = str(value).strip()
    while text.endswith(".manifest"):
        text = text[: -len(".manifest")]
    return text


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
    config_path = Path("runs") / "configs" / "detect" / f"{safe_name}_v{version:03d}.yaml"
    manifest_path = (
        Path(args.out_manifest)
        if args.out_manifest is not None
        else (Path("runs") / "manifests" / "detect" / f"{safe_name}_v{version:03d}.manifest.json")
    )
    return config_path, manifest_path


def _resolve_merged_training_paths(
    *,
    preflight_manifest_path: Path,
    merge_out_dir: Optional[Path],
) -> tuple[Path, Path]:
    payload = json.loads(preflight_manifest_path.read_text(encoding="utf-8"))
    set_id_source = str(payload.get("set_id")).strip() if payload.get("set_id") else preflight_manifest_path.stem
    set_id = _normalize_manifest_stem(set_id_source) or "training_set"
    out_dir = Path(merge_out_dir) if merge_out_dir is not None else (_resolve_default_dataset_root() / set_id)
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
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run fisheye.training.train_detection after successful preflight/export.",
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
        args.set_name = _default_set_name(args, rows, model_input=model_input)
        print(f"Auto set-name: {args.set_name}")

    if (args.export_merged or args.train) and args.out_config is None and args.set_name and args.set_version is None:
        args.set_version = _next_detect_version(_sanitize_name(args.set_name))

    if args.export_merged and args.out_manifest is None:
        if args.out_config is not None:
            args.out_manifest = args.out_config.with_suffix(".manifest.json")
        elif args.set_name is not None:
            safe_name = _sanitize_name(args.set_name)
            if args.set_version is not None:
                if args.set_version < 1:
                    raise SystemExit("--set-version must be >= 1.")
                set_version = int(args.set_version)
            else:
                set_version = _next_detect_version(safe_name)
                args.set_version = set_version
            args.out_manifest = (
                Path("runs") / "manifests" / "detect" / f"{safe_name}_v{set_version:03d}.manifest.json"
            )
        else:
            raise SystemExit("--export-merged requires either --set-name or --out-config to infer --out-manifest.")
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
            )
    elif args.train:
        train_config, train_manifest = _resolve_preflight_output_paths(args)
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
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
