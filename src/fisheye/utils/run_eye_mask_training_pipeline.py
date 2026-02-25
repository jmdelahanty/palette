#!/usr/bin/env python3
"""Run eye-mask pipeline from registry query through optional merge and train."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence

import yaml

from fisheye.registry.db import Registry, RegistryPaths
from fisheye.utils import aggregate_eye_mask_training_data_card as aggregate_data_card
from fisheye.utils import export_eye_mask_training_zarr as export_zarr
from fisheye.utils import prepare_eye_mask_training_from_registry as prepare_from_registry


def _add_arg(argv: List[str], flag: str, value: Optional[object]) -> None:
    if value is None:
        return
    argv.extend([flag, str(value)])


def _parse_split_spec(value: str) -> tuple[float, float, float]:
    text = str(value).strip()
    if not text:
        raise SystemExit("--merge-split must be non-empty (expected train/val or train/val/test).")
    parts = [part.strip() for part in text.split("/") if part.strip()]
    if len(parts) not in (2, 3):
        raise SystemExit(
            f"Invalid --merge-split '{value}'. Expected train/val or train/val/test ratios."
        )
    try:
        ratios = [float(part) for part in parts]
    except Exception as exc:  # pragma: no cover - defensive
        raise SystemExit(f"Invalid --merge-split '{value}': {exc}") from exc
    if any(ratio < 0.0 for ratio in ratios):
        raise SystemExit(f"Invalid --merge-split '{value}': ratios must be non-negative.")
    total = float(sum(ratios))
    if total <= 0.0:
        raise SystemExit(f"Invalid --merge-split '{value}': at least one ratio must be > 0.")
    if len(ratios) == 2:
        return float(ratios[0]), float(ratios[1]), 0.0
    return float(ratios[0]), float(ratios[1]), float(ratios[2])


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
        else (prepare_from_registry._resolve_dataset_root("eye_masks") / set_id)
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


def _resolve_preflight_output_paths(
    args: argparse.Namespace,
    *,
    rows: Sequence[Mapping[str, Any]],
    model_input: str,
) -> tuple[Path, Path]:
    # Match prepare wrapper path conventions so downstream stages can locate outputs.
    if args.out_manifest is None and args.out_config is None and args.out_dir is None:
        if args.set_name:
            resolved_set_name = prepare_from_registry._sanitize_name(args.set_name)
        else:
            resolved_set_name = prepare_from_registry._default_set_name(args, rows, model_input=model_input)
            args.set_name = resolved_set_name
            print(f"Auto set-name: {resolved_set_name}")
        dataset_root = prepare_from_registry._resolve_dataset_root("eye_masks")
        if args.set_version is None:
            args.set_version = prepare_from_registry._next_version_from_dataset_root(
                set_name=resolved_set_name,
                task_prefix="eye_mask",
                dataset_root=dataset_root,
            )
        version = int(args.set_version)
        if version < 1:
            raise SystemExit("--set-version must be >= 1")
        set_id = prepare_from_registry._build_training_set_id(resolved_set_name, version)
        out_dir = Path(args.out_dir) if args.out_dir is not None else (dataset_root / set_id)
        out_config = out_dir / f"{set_id}.yaml"
        out_manifest = out_dir / f"{set_id}.manifest.json"
        return out_config, out_manifest

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    elif args.out_config is not None:
        out_dir = Path(args.out_config).parent
    elif args.out_manifest is not None:
        out_dir = Path(args.out_manifest).parent
    else:
        out_dir = prepare_from_registry._resolve_dataset_root("eye_masks")

    if args.set_name:
        args.set_name = prepare_from_registry._sanitize_name(args.set_name)
    if args.set_version is not None and int(args.set_version) < 1:
        raise SystemExit("--set-version must be >= 1")

    out_config = (
        Path(args.out_config)
        if args.out_config is not None
        else (
            prepare_from_registry._config_path_for_manifest(Path(args.out_manifest))
            if args.out_manifest is not None
            else (out_dir / "eye_mask_training.yaml")
        )
    )
    out_manifest = (
        Path(args.out_manifest)
        if args.out_manifest is not None
        else prepare_from_registry._manifest_path_for_config(out_config)
    )
    return out_config, out_manifest


def _resolve_default_merged_out_zarr(*, manifest_payload: Mapping[str, Any], manifest_path: Path) -> Path:
    datasets = manifest_payload.get("datasets")
    if isinstance(datasets, list) and datasets:
        first = datasets[0]
        if isinstance(first, Mapping):
            raw = first.get("out_zarr")
            if raw:
                return Path(str(raw))
    set_id = str(manifest_payload.get("set_id")).strip() if manifest_payload.get("set_id") else ""
    if not set_id:
        set_id = _normalize_manifest_stem(manifest_path.stem) or "training_set"
    output_root = manifest_payload.get("output_root")
    if output_root:
        return Path(str(output_root)) / "zarr" / f"{set_id}_merged.zarr"
    return prepare_from_registry._resolve_dataset_root("eye_masks") / set_id / "zarr" / f"{set_id}_merged.zarr"


def _load_manifest_payload(manifest_path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Failed to read manifest JSON: {manifest_path} ({exc})") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"Manifest root must be an object: {manifest_path}")
    return payload


def _build_merge_source_specs(manifest_payload: Mapping[str, Any]) -> List[export_zarr.EyeMergeSourceSpec]:
    selected_sources = manifest_payload.get("selected_sources")
    if not isinstance(selected_sources, list) or not selected_sources:
        raise SystemExit("Preflight manifest missing selected_sources for merged export.")

    specs: List[export_zarr.EyeMergeSourceSpec] = []
    for entry in selected_sources:
        if not isinstance(entry, Mapping):
            continue
        zarr_path = entry.get("zarr_path")
        if not zarr_path:
            continue
        specs.append(
            export_zarr.EyeMergeSourceSpec(
                source_zarr=Path(str(zarr_path)),
                crop_run=prepare_from_registry._as_text(entry.get("source_crop_run")),
                eye_stage=prepare_from_registry._as_text(entry.get("source_eye_stage")) or "auto",
                eye_run=prepare_from_registry._as_text(entry.get("source_eye_run")),
            )
        )
    if not specs:
        raise SystemExit("No valid selected_sources rows with zarr_path in preflight manifest.")
    return specs


def _write_merged_outputs(
    *,
    preflight_config: Path,
    preflight_manifest: Path,
    merged_config: Path,
    merged_manifest: Path,
    merged_out_zarr: Path,
    merge_summary: Mapping[str, Any],
) -> None:
    manifest_payload = _load_manifest_payload(preflight_manifest)
    config_payload = yaml.safe_load(preflight_config.read_text(encoding="utf-8"))
    if not isinstance(config_payload, dict):
        raise SystemExit(f"Preflight config root must be a mapping: {preflight_config}")

    datasets = manifest_payload.get("datasets")
    if isinstance(datasets, list) and datasets:
        dataset0 = datasets[0]
        if isinstance(dataset0, dict):
            dataset0["out_zarr"] = str(merged_out_zarr)
            dataset0["export_status"] = "succeeded"
            dataset0["export_result"] = dict(merge_summary)
    merged_export = manifest_payload.get("merged_export")
    if isinstance(merged_export, dict):
        merged_export["zarr_path"] = str(merged_out_zarr)
        merged_export["export_status"] = "succeeded"
        merged_export["export_result"] = dict(merge_summary)

    execution = manifest_payload.get("execution")
    if not isinstance(execution, dict):
        execution = {}
        manifest_payload["execution"] = execution
    execution["mode"] = "apply"
    execution["planned"] = 1
    execution["succeeded"] = 1
    execution["failed"] = 0

    manifest_payload["output_root"] = str(merged_config.parent)
    manifest_payload["output_config_path"] = str(merged_config)
    manifest_payload["output_manifest_path"] = str(merged_manifest)

    generated_datasets = config_payload.get("datasets")
    if isinstance(generated_datasets, dict):
        for _, dataset_cfg in generated_datasets.items():
            if isinstance(dataset_cfg, dict):
                dataset_cfg["zarr_path"] = str(merged_out_zarr)

    merged_config.parent.mkdir(parents=True, exist_ok=True)
    merged_manifest.parent.mkdir(parents=True, exist_ok=True)
    merged_config.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")
    merged_manifest.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")


def _run_training(
    *,
    config_path: Path,
    manifest_path: Path,
    set_id: str,
    registry_path: Path,
    run_name: Optional[str],
    project: Optional[str],
) -> int:
    if not config_path.exists():
        raise SystemExit(f"Training config not found: {config_path}")
    if not manifest_path.exists():
        raise SystemExit(f"Training manifest not found: {manifest_path}")

    cmd: List[str] = [
        sys.executable,
        "-m",
        "fisheye.training.train_eye_masks",
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
    parser.add_argument("--out-commands", type=Path, help="Write planned export commands to this file.")
    parser.add_argument("--set-name", type=str, help="Training set name.")
    parser.add_argument("--set-version", type=int)
    parser.add_argument("--run-name-prefix", type=str, default="merged_eye_masks")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument(
        "--train",
        action="store_true",
        help="Run fisheye.training.train_eye_masks after successful prepare/export.",
    )
    parser.add_argument(
        "--export-merged",
        action="store_true",
        help="After preflight manifest generation, export one merged eye-mask training Zarr.",
    )
    parser.add_argument(
        "--build-dataset",
        action="store_true",
        help="Convenience flag that enables both --export-merged and --aggregate-training-data-card.",
    )
    parser.add_argument("--merge-out-zarr", type=Path, help="Output merged .zarr path.")
    parser.add_argument("--merge-out-dir", type=Path, help="Output directory for merged config/manifest artifacts.")
    parser.add_argument(
        "--merge-split",
        type=str,
        default="0.8/0.2",
        help="Split ratios for merged export (train/val or train/val/test).",
    )
    parser.add_argument("--merge-seed", type=int, default=42, help="Split seed for merged export.")
    parser.add_argument("--merge-overwrite", action="store_true", help="Allow overwrite of merged output zarr.")
    parser.add_argument(
        "--aggregate-training-data-card",
        action="store_true",
        help="Aggregate eye-mask training data card after preflight/export.",
    )
    parser.add_argument(
        "--no-aggregate-training-data-card",
        action="store_true",
        help="Disable automatic data-card aggregation when --export-merged is used.",
    )
    parser.add_argument("--data-card-output", type=Path, help="Optional output JSON path for data-card aggregation.")
    parser.add_argument(
        "--data-card-split",
        type=str,
        default="train",
        help="Split label for data-card selection metadata (default: train).",
    )
    parser.add_argument("--data-card-no-plots", action="store_true", help="Skip data-card plot generation.")
    parser.add_argument("--data-card-plot-dir", type=Path, help="Optional output directory for data-card plots.")
    parser.add_argument("--data-card-plot-prefix", type=str, help="Optional filename prefix for data-card plots.")
    parser.add_argument(
        "--data-card-plot-heatmap-bin-factor",
        type=int,
        default=2,
        help="Coarsening factor for heatmap bins in data-card plots (default: 2).",
    )
    parser.add_argument(
        "--data-card-view",
        action="store_true",
        help="Open eye-mask data-card plots after aggregation (uses existing plots unless forced).",
    )
    parser.add_argument(
        "--data-card-force-plots",
        action="store_true",
        help="Regenerate eye-mask data-card plots even when plot files already exist.",
    )
    parser.add_argument(
        "--data-card-allow-profile-mtime-mismatch",
        action="store_true",
        help="Allow stale eye_mask_data_profile_latest rows during aggregation.",
    )
    parser.add_argument(
        "--data-card-allow-profile-fallback-scan",
        action="store_true",
        help="Allow fallback scan when eye_mask_data_profile_latest rows are missing.",
    )
    parser.add_argument("--run-name", type=str, help="Optional training run name override.")
    parser.add_argument("--project", type=str, help="Optional training project override.")

    args = parser.parse_args(argv)
    if args.build_dataset:
        args.export_merged = True
        args.aggregate_training_data_card = True
    if args.build_dataset and args.dry_run:
        raise SystemExit("--build-dataset cannot be combined with --dry-run.")
    if args.export_merged and args.dry_run:
        raise SystemExit("--export-merged cannot be combined with --dry-run (no merged zarr is written).")
    if args.train and args.dry_run:
        raise SystemExit("--train cannot be combined with --dry-run.")
    if args.aggregate_training_data_card and args.dry_run:
        raise SystemExit("--aggregate-training-data-card cannot be combined with --dry-run.")
    if args.aggregate_training_data_card and args.no_aggregate_training_data_card:
        raise SystemExit(
            "--aggregate-training-data-card cannot be combined with --no-aggregate-training-data-card."
        )
    if args.data_card_plot_heatmap_bin_factor < 1:
        parser.error("--data-card-plot-heatmap-bin-factor must be >= 1.")
    if args.data_card_no_plots and args.data_card_plot_dir is not None:
        parser.error("--data-card-plot-dir cannot be combined with --data-card-no-plots.")
    if args.data_card_view and args.data_card_no_plots:
        parser.error("--data-card-view cannot be combined with --data-card-no-plots.")
    if args.data_card_force_plots and args.data_card_no_plots:
        parser.error("--data-card-force-plots cannot be combined with --data-card-no-plots.")

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
    non_training_rows: List[Mapping[str, Any]] = []
    for row in rows:
        purpose = prepare_from_registry._as_text(row.get("zarr_use"))
        zarr_path = Path(str(row["zarr_path"]))
        if str(purpose or "").lower() == "training" and prepare_from_registry._looks_like_training_artifact_path(zarr_path):
            continue
        non_training_rows.append(row)
    rows = non_training_rows

    if prepare_from_registry._quality_filters_active(args):
        dataset_ids = [str(row["dataset_id"]) for row in rows if row.get("dataset_id")]
        profile_rows = registry.query_eye_mask_data_profile_latest(
            dataset_ids=dataset_ids,
            stage_group=prepare_from_registry._default_profile_stage_group(args),
            eye_mask_method=args.eye_mask_method,
            min_usable_rate=args.min_usable_rate,
        )
        by_dataset: Dict[str, List[Mapping[str, Any]]] = {}
        for profile_row in profile_rows:
            profile_row_dict = dict(profile_row)
            dataset_id = prepare_from_registry._as_text(profile_row_dict.get("dataset_id"))
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
            chosen = prepare_from_registry._choose_profile_candidate(
                candidates,
                preferred_stages=preferred_stages,
                required_review_state=args.require_review_state,
                required_review_use=args.require_review_intended_use,
            )
            if chosen is not None:
                filtered_rows.append(row)
        rows = filtered_rows
    registry.close()
    if not rows:
        raise SystemExit("Registry query returned no datasets.")

    out_config, out_manifest = _resolve_preflight_output_paths(args, rows=rows, model_input=model_input)
    if args.out_manifest is None:
        print(f"Auto out-manifest: {out_manifest}")
    if args.out_config is None:
        print(f"Auto out-config: {out_config}")

    if args.output_file_list:
        args.output_file_list.parent.mkdir(parents=True, exist_ok=True)
        args.output_file_list.write_text(
            "\n".join([str(Path(row["zarr_path"])) for row in rows]) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {len(rows)} paths to {args.output_file_list}")

    preflight_manifest_path = Path(out_manifest)
    preflight_config_path = Path(out_config)
    print(f"Registry query matched {len(rows)} dataset(s).")
    print("Stage 1/4: prepare eye-mask preflight manifest/config")

    prepare_cli: List[str] = []
    _add_arg(prepare_cli, "--registry", registry_path)
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
    _add_arg(prepare_cli, "--zarr-use", args.zarr_use)
    _add_arg(prepare_cli, "--path-contains", args.path_contains)
    _add_arg(prepare_cli, "--limit", args.limit)
    _add_arg(prepare_cli, "--output-file-list", args.output_file_list)
    _add_arg(prepare_cli, "--eye-stage", args.eye_stage)
    _add_arg(prepare_cli, "--eye-run", args.eye_run)
    _add_arg(prepare_cli, "--crop-run", args.crop_run)
    _add_arg(prepare_cli, "--input-format", args.input_format)
    _add_arg(prepare_cli, "--model-input", args.model_input)
    _add_arg(prepare_cli, "--label-mode", args.label_mode)
    _add_arg(prepare_cli, "--split-train", args.split_train)
    _add_arg(prepare_cli, "--split-val", args.split_val)
    _add_arg(prepare_cli, "--split-test", args.split_test)
    _add_arg(prepare_cli, "--split-seed", args.split_seed)
    _add_arg(prepare_cli, "--profile-stage-group", args.profile_stage_group)
    _add_arg(prepare_cli, "--eye-mask-method", args.eye_mask_method)
    _add_arg(prepare_cli, "--min-usable-rate", args.min_usable_rate)
    _add_arg(prepare_cli, "--require-review-state", args.require_review_state)
    _add_arg(prepare_cli, "--require-review-intended-use", args.require_review_intended_use)
    _add_arg(prepare_cli, "--base-config", args.base_config)
    _add_arg(prepare_cli, "--out-config", out_config)
    _add_arg(prepare_cli, "--out-dir", args.out_dir)
    _add_arg(prepare_cli, "--out-manifest", out_manifest)
    _add_arg(prepare_cli, "--out-commands", args.out_commands)
    _add_arg(prepare_cli, "--set-name", args.set_name)
    _add_arg(prepare_cli, "--set-version", args.set_version)
    _add_arg(prepare_cli, "--run-name-prefix", args.run_name_prefix)
    if args.dry_run:
        prepare_cli.append("--dry-run")

    prepare_rc = int(prepare_from_registry.main(prepare_cli))
    if prepare_rc != 0:
        return int(prepare_rc)
    if args.dry_run:
        return 0

    explicit_data_card_requested = bool(args.aggregate_training_data_card)
    auto_data_card_requested = bool(args.export_merged and not args.no_aggregate_training_data_card)
    should_aggregate_data_card = bool(explicit_data_card_requested or auto_data_card_requested)
    if should_aggregate_data_card and auto_data_card_requested and not explicit_data_card_requested:
        print(
            "Auto enabling eye-mask training data-card aggregation for merged export. "
            "Use --no-aggregate-training-data-card to disable."
        )

    effective_config_path = preflight_config_path
    effective_manifest_path = preflight_manifest_path

    if args.export_merged:
        print("Stage 2/4: export merged eye-mask zarr")
        if not preflight_manifest_path.exists():
            raise SystemExit(f"Expected manifest not found after preflight: {preflight_manifest_path}")
        if not preflight_config_path.exists():
            raise SystemExit(f"Expected config not found after preflight: {preflight_config_path}")

        manifest_payload = _load_manifest_payload(preflight_manifest_path)
        source_specs = _build_merge_source_specs(manifest_payload)
        default_out_zarr = _resolve_default_merged_out_zarr(
            manifest_payload=manifest_payload,
            manifest_path=preflight_manifest_path,
        )
        if args.merge_out_zarr is not None:
            merged_out_zarr = Path(args.merge_out_zarr)
        elif args.merge_out_dir is not None:
            set_id = str(manifest_payload.get("set_id")).strip() if manifest_payload.get("set_id") else ""
            if not set_id:
                set_id = _normalize_manifest_stem(preflight_manifest_path.stem) or "training_set"
            merged_out_zarr = Path(args.merge_out_dir) / "zarr" / f"{set_id}_merged.zarr"
        else:
            merged_out_zarr = default_out_zarr

        train_ratio, val_ratio, test_ratio = _parse_split_spec(str(args.merge_split))
        run_name = "merged_eye_masks"
        datasets = manifest_payload.get("datasets")
        if isinstance(datasets, list) and datasets:
            first = datasets[0]
            if isinstance(first, Mapping):
                run_name = str(first.get("run_name") or run_name)

        merge_summary = export_zarr.export_merged_eye_mask_training_zarr_from_sources(
            source_specs=source_specs,
            out_zarr=Path(merged_out_zarr),
            run_name=run_name,
            input_format=str(manifest_payload.get("input_format") or args.input_format),
            label_mode=str(manifest_payload.get("label_mode") or args.label_mode),
            split_train=float(train_ratio),
            split_val=float(val_ratio),
            split_test=float(test_ratio),
            split_seed=int(args.merge_seed),
            overwrite=bool(args.merge_overwrite),
            validate=True,
            registry=Path(registry_path),
            training_set_id=str(manifest_payload.get("set_id")) if manifest_payload.get("set_id") else None,
            training_set_name=str(manifest_payload.get("set_name")) if manifest_payload.get("set_name") else None,
            aggregate_training_data_card=False,
            data_card_no_plots=bool(args.data_card_no_plots),
        )

        if args.merge_out_dir is not None:
            merged_config, merged_manifest = _resolve_merged_training_paths(
                preflight_manifest_path=preflight_manifest_path,
                merge_out_dir=args.merge_out_dir,
            )
            _write_merged_outputs(
                preflight_config=preflight_config_path,
                preflight_manifest=preflight_manifest_path,
                merged_config=merged_config,
                merged_manifest=merged_manifest,
                merged_out_zarr=Path(merged_out_zarr),
                merge_summary=merge_summary,
            )
            effective_config_path = merged_config
            effective_manifest_path = merged_manifest
        else:
            _write_merged_outputs(
                preflight_config=preflight_config_path,
                preflight_manifest=preflight_manifest_path,
                merged_config=preflight_config_path,
                merged_manifest=preflight_manifest_path,
                merged_out_zarr=Path(merged_out_zarr),
                merge_summary=merge_summary,
            )

        print(
            "Merged export complete: "
            f"zarr={merged_out_zarr} samples={merge_summary.get('total_samples')} "
            f"sources={merge_summary.get('source_count')}"
        )

    if should_aggregate_data_card:
        print("Stage 3/4: aggregate eye-mask training data card")
        if not effective_manifest_path.exists():
            raise SystemExit(
                f"--aggregate-training-data-card requires manifest output: {effective_manifest_path}"
            )
        card_cli: List[str] = [
            "--manifest",
            str(effective_manifest_path),
            "--registry",
            str(registry_path),
            "--split",
            str(args.data_card_split),
        ]
        _add_arg(card_cli, "--output", args.data_card_output)
        if args.data_card_allow_profile_mtime_mismatch:
            card_cli.append("--allow-profile-mtime-mismatch")
        if args.data_card_allow_profile_fallback_scan:
            card_cli.append("--allow-profile-fallback-scan")
        if args.data_card_no_plots:
            card_cli.append("--no-plots")
        _add_arg(card_cli, "--plot-dir", args.data_card_plot_dir)
        _add_arg(card_cli, "--plot-prefix", args.data_card_plot_prefix)
        _add_arg(card_cli, "--plot-heatmap-bin-factor", args.data_card_plot_heatmap_bin_factor)
        if args.data_card_view:
            card_cli.append("--view")
        if args.data_card_force_plots:
            card_cli.append("--force")
        card_rc = aggregate_data_card.main(card_cli)
        if card_rc != 0:
            return int(card_rc)

    if args.train:
        print("Stage 4/4: train eye-mask model")
        set_id = _require_manifest_set_id(effective_manifest_path)
        return _run_training(
            config_path=effective_config_path,
            manifest_path=effective_manifest_path,
            set_id=set_id,
            registry_path=Path(registry_path),
            run_name=args.run_name,
            project=args.project,
        )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
