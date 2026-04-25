#!/usr/bin/env python3
"""Run subject-mask merged export and optional U-Net training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence

import yaml

from fisheye.utils import export_subject_mask_training_zarr as export_zarr


def _add_arg(argv: List[str], flag: str, value: Optional[object]) -> None:
    if value is None:
        return
    argv.extend([flag, str(value)])


def _as_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


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
    return text or "subject_mask_training"


def _load_manifest_payload(manifest_path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Failed to read manifest JSON: {manifest_path} ({exc})") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"Manifest root must be an object: {manifest_path}")
    return payload


def _load_config_payload(config_path: Path) -> Dict[str, Any]:
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Failed to read config YAML: {config_path} ({exc})") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"Config root must be a mapping: {config_path}")
    return payload


def _require_manifest_set_id(
    *,
    manifest_payload: Mapping[str, Any],
    manifest_path: Path,
    training_set_id: Optional[str],
) -> str:
    set_id = _as_text(training_set_id) or _as_text(manifest_payload.get("set_id"))
    if not set_id:
        raise SystemExit(
            "--train requires --training-set-id or manifest set_id to avoid unlinked runs. "
            f"Manifest: {manifest_path}"
        )
    return set_id


def _source_entries(manifest_payload: Mapping[str, Any]) -> Sequence[Any]:
    selected = manifest_payload.get("selected_sources")
    if isinstance(selected, list) and selected:
        return selected
    datasets = manifest_payload.get("datasets")
    if isinstance(datasets, list) and datasets:
        return datasets
    return []


def _build_merge_source_specs(
    manifest_payload: Mapping[str, Any],
) -> List[export_zarr.SubjectMergeSourceSpec]:
    specs: List[export_zarr.SubjectMergeSourceSpec] = []
    for entry in _source_entries(manifest_payload):
        if not isinstance(entry, Mapping):
            continue
        zarr_path = (
            entry.get("zarr_path")
            or entry.get("source_zarr")
            or entry.get("source_zarr_path")
        )
        if not zarr_path:
            continue
        specs.append(
            export_zarr.SubjectMergeSourceSpec(
                source_zarr=Path(str(zarr_path)),
                subject_run=(
                    _as_text(entry.get("source_subject_run"))
                    or _as_text(entry.get("source_subject_mask_run"))
                    or _as_text(entry.get("subject_run"))
                    or _as_text(entry.get("subject_mask_run"))
                ),
                crop_run=(
                    _as_text(entry.get("source_crop_run"))
                    or _as_text(entry.get("crop_run"))
                ),
                stage_group=(
                    _as_text(entry.get("source_stage_group"))
                    or _as_text(entry.get("stage_group"))
                    or "subject_mask_runs"
                ),
                allow_unapproved_refined=_as_bool(entry.get("allow_unapproved_refined")),
            )
        )
    if not specs:
        raise SystemExit(
            "Manifest missing source rows with zarr_path/source_zarr. "
            "Use selected_sources for canonical source selection."
        )
    return specs


def _resolve_default_merged_out_zarr(
    *,
    manifest_payload: Mapping[str, Any],
    manifest_path: Path,
    training_set_id: Optional[str],
) -> Path:
    merged_export = manifest_payload.get("merged_export")
    if isinstance(merged_export, Mapping):
        raw = merged_export.get("zarr_path") or merged_export.get("out_zarr")
        if raw:
            return Path(str(raw))

    set_id = (
        _as_text(training_set_id)
        or _as_text(manifest_payload.get("set_id"))
        or _normalize_manifest_stem(manifest_path.stem)
    )
    output_root = manifest_payload.get("output_root")
    if output_root:
        return Path(str(output_root)) / "zarr" / f"{set_id}_merged.zarr"
    return manifest_path.parent / "zarr" / f"{set_id}_merged.zarr"


def _resolve_merge_run_name(
    *,
    manifest_payload: Mapping[str, Any],
    requested: Optional[str],
) -> str:
    if requested:
        return requested
    merged_export = manifest_payload.get("merged_export")
    if isinstance(merged_export, Mapping):
        run_name = _as_text(merged_export.get("run_name"))
        if run_name:
            return run_name
    datasets = manifest_payload.get("datasets")
    if isinstance(datasets, list) and datasets:
        first = datasets[0]
        if isinstance(first, Mapping):
            run_name = _as_text(first.get("run_name"))
            if run_name:
                return run_name
    return "merged_subject_masks"


def _resolve_output_path(
    *,
    input_path: Path,
    output_path: Optional[Path],
) -> Path:
    return Path(output_path) if output_path is not None else input_path


def _write_merged_outputs(
    *,
    config_path: Optional[Path],
    manifest_path: Path,
    out_config: Optional[Path],
    out_manifest: Optional[Path],
    merged_out_zarr: Path,
    merge_summary: Mapping[str, Any],
    merged_run_name: str,
) -> tuple[Optional[Path], Path]:
    manifest_payload = _load_manifest_payload(manifest_path)
    datasets = manifest_payload.get("datasets")
    if isinstance(datasets, list) and datasets:
        first = datasets[0]
        if isinstance(first, dict):
            first["zarr_path"] = str(merged_out_zarr)
            first["out_zarr"] = str(merged_out_zarr)
            first["run_name"] = merged_run_name
            first["export_status"] = "succeeded"
            first["export_result"] = dict(merge_summary)
    merged_export = manifest_payload.get("merged_export")
    if not isinstance(merged_export, dict):
        merged_export = {}
        manifest_payload["merged_export"] = merged_export
    merged_export["zarr_path"] = str(merged_out_zarr)
    merged_export["run_name"] = merged_run_name
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

    effective_manifest_path = _resolve_output_path(input_path=manifest_path, output_path=out_manifest)
    manifest_payload["output_manifest_path"] = str(effective_manifest_path)

    effective_config_path: Optional[Path] = None
    if config_path is not None:
        config_payload = _load_config_payload(config_path)
        generated_datasets = config_payload.get("datasets")
        if isinstance(generated_datasets, dict):
            for dataset_cfg in generated_datasets.values():
                if isinstance(dataset_cfg, dict):
                    dataset_cfg["zarr_path"] = str(merged_out_zarr)
                    dataset_cfg["crop_run"] = merged_run_name
                    dataset_cfg["subject_mask_run"] = merged_run_name
        training_params = config_payload.get("training_params")
        if isinstance(training_params, dict):
            training_params["crop_run"] = merged_run_name
            training_params["subject_masks_run"] = merged_run_name
            training_params["label_schema_id"] = "auto"
            config_payload.pop("names", None)
            config_payload.pop("nc", None)

        effective_config_path = _resolve_output_path(input_path=config_path, output_path=out_config)
        manifest_payload["output_config_path"] = str(effective_config_path)
        effective_config_path.parent.mkdir(parents=True, exist_ok=True)
        effective_config_path.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")

    effective_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    effective_manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    return effective_config_path, effective_manifest_path


def _run_training(
    *,
    config_path: Path,
    manifest_path: Optional[Path],
    set_id: Optional[str],
    registry_path: Optional[Path],
    run_name: Optional[str],
    output_dir: Optional[str],
    device: Optional[str],
    epochs: Optional[int],
    tb_logdir: Optional[str],
    no_progress: bool,
    no_compile: bool,
) -> int:
    if not config_path.exists():
        raise SystemExit(f"Training config not found: {config_path}")
    cmd: List[str] = [
        sys.executable,
        "-m",
        "fisheye.segmentation.train_unet_subject_masks",
        str(config_path),
    ]
    if manifest_path is not None:
        if not manifest_path.exists():
            raise SystemExit(f"Training manifest not found: {manifest_path}")
        _add_arg(cmd, "--manifest", manifest_path)
    _add_arg(cmd, "--set-id", set_id)
    _add_arg(cmd, "--registry", registry_path)
    _add_arg(cmd, "--run-name", run_name)
    _add_arg(cmd, "--output-dir", output_dir)
    _add_arg(cmd, "--device", device)
    _add_arg(cmd, "--epochs", epochs)
    _add_arg(cmd, "--tb-logdir", tb_logdir)
    if no_progress:
        cmd.append("--no-progress")
    if no_compile:
        cmd.append("--no-compile")
    print("Launching training: " + " ".join(cmd))
    completed = subprocess.run(cmd, check=False)
    return int(completed.returncode)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="SubjectMaskTrainingConfig YAML path.")
    parser.add_argument("--manifest", type=Path, help="Subject-mask training manifest JSON path.")
    parser.add_argument("--out-config", type=Path, help="Optional rewritten config path after merged export.")
    parser.add_argument("--out-manifest", type=Path, help="Optional rewritten manifest path after merged export.")
    parser.add_argument("--registry", type=Path, help="Optional registry SQLite path.")
    parser.add_argument("--training-set-id", type=str, help="Training set ID for export/train registry linkage.")
    parser.add_argument("--training-set-name", type=str, help="Training set display name for merged export.")
    parser.add_argument(
        "--export-merged",
        action="store_true",
        help="Export one merged subject-mask training Zarr from manifest selected_sources.",
    )
    parser.add_argument("--merge-out-zarr", type=Path, help="Output merged .zarr path.")
    parser.add_argument(
        "--merge-split",
        type=str,
        default="0.8/0.2",
        help="Split ratios for merged export (train/val or train/val/test).",
    )
    parser.add_argument("--merge-seed", type=int, default=123, help="Split seed for merged export.")
    parser.add_argument("--merge-run-name", type=str, help="Run name for crop_runs and subject_mask_runs.")
    parser.add_argument(
        "--subject-label-schema",
        choices=list(export_zarr.LABEL_SCHEMA_CHOICES),
        default="subject_v1_union",
        help="Target subject-mask label schema for the merged export.",
    )
    parser.add_argument("--input-format", choices=["gray", "rgb"], help="Optional explicit input format.")
    parser.add_argument("--merge-overwrite", action="store_true", help="Allow overwrite of merged output zarr.")
    parser.add_argument(
        "--allow-unapproved-refined",
        action="store_true",
        help="Allow draft/QA merged export from refined_subject_masks_runs components without approved review state.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run fisheye.segmentation.train_unet_subject_masks after optional export.",
    )
    parser.add_argument("--run-name", type=str, help="Optional training run name override.")
    parser.add_argument("--project", type=str, help="Optional training output directory override.")
    parser.add_argument("--device", type=str, help="Optional training device override.")
    parser.add_argument("--epochs", type=int, help="Optional training epoch override.")
    parser.add_argument("--tb-logdir", type=str, help="Optional TensorBoard log directory for training.")
    parser.add_argument("--no-progress", action="store_true", help="Disable rich progress bars in training.")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile in training.")
    args = parser.parse_args(argv)

    if args.export_merged and args.manifest is None:
        raise SystemExit("--export-merged requires --manifest.")
    if args.train and args.config is None and args.out_config is None:
        raise SystemExit("--train requires --config or --out-config.")
    if args.out_config is not None and args.config is None:
        raise SystemExit("--out-config requires --config.")
    if args.out_manifest is not None and args.manifest is None:
        raise SystemExit("--out-manifest requires --manifest.")
    if args.epochs is not None and int(args.epochs) <= 0:
        parser.error("--epochs must be > 0.")

    effective_config_path: Optional[Path] = Path(args.config) if args.config is not None else None
    effective_manifest_path: Optional[Path] = Path(args.manifest) if args.manifest is not None else None

    manifest_payload: Dict[str, Any] = {}
    if args.manifest is not None:
        manifest_payload = _load_manifest_payload(Path(args.manifest))

    if args.export_merged:
        assert effective_manifest_path is not None
        print("Stage 1/2: export merged subject-mask zarr")
        source_specs = _build_merge_source_specs(manifest_payload)
        merged_out_zarr = (
            Path(args.merge_out_zarr)
            if args.merge_out_zarr is not None
            else _resolve_default_merged_out_zarr(
                manifest_payload=manifest_payload,
                manifest_path=effective_manifest_path,
                training_set_id=args.training_set_id,
            )
        )
        train_ratio, val_ratio, test_ratio = _parse_split_spec(str(args.merge_split))
        merged_run_name = _resolve_merge_run_name(
            manifest_payload=manifest_payload,
            requested=args.merge_run_name,
        )
        set_id = _as_text(args.training_set_id) or _as_text(manifest_payload.get("set_id"))
        set_name = _as_text(args.training_set_name) or _as_text(manifest_payload.get("set_name"))
        merge_summary = export_zarr.export_merged_subject_mask_training_zarr_from_sources(
            source_specs=source_specs,
            out_zarr=merged_out_zarr,
            subject_label_schema=str(args.subject_label_schema),
            input_format=args.input_format,
            train_ratio=float(train_ratio),
            val_ratio=float(val_ratio),
            test_ratio=float(test_ratio),
            split_seed=int(args.merge_seed),
            run_name=merged_run_name,
            overwrite=bool(args.merge_overwrite),
            registry=args.registry,
            training_set_id=set_id,
            training_set_name=set_name,
            allow_unapproved_refined=bool(args.allow_unapproved_refined),
        )
        effective_config_path, effective_manifest_path = _write_merged_outputs(
            config_path=effective_config_path,
            manifest_path=effective_manifest_path,
            out_config=args.out_config,
            out_manifest=args.out_manifest,
            merged_out_zarr=Path(merged_out_zarr),
            merge_summary=merge_summary,
            merged_run_name=merged_run_name,
        )
        print(
            "Merged export complete: "
            f"zarr={merged_out_zarr} samples={merge_summary.get('total_samples')} "
            f"sources={merge_summary.get('source_count')}"
        )
    else:
        if args.out_config is not None:
            effective_config_path = Path(args.out_config)
        if args.out_manifest is not None:
            effective_manifest_path = Path(args.out_manifest)

    if args.train:
        print("Stage 2/2: train subject-mask model")
        set_id: Optional[str] = args.training_set_id
        if effective_manifest_path is not None:
            effective_manifest_payload = _load_manifest_payload(effective_manifest_path)
            set_id = _require_manifest_set_id(
                manifest_payload=effective_manifest_payload,
                manifest_path=effective_manifest_path,
                training_set_id=args.training_set_id,
            )
        if effective_config_path is None:
            raise SystemExit("--train requires a config path.")
        return _run_training(
            config_path=effective_config_path,
            manifest_path=effective_manifest_path,
            set_id=set_id,
            registry_path=args.registry,
            run_name=args.run_name,
            output_dir=args.project,
            device=args.device,
            epochs=args.epochs,
            tb_logdir=args.tb_logdir,
            no_progress=bool(args.no_progress),
            no_compile=bool(args.no_compile),
        )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
