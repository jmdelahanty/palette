#!/usr/bin/env python3
"""Resolve an eye-mask model from registry, run inference, and persist resolution provenance."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import zarr

from fisheye.registry.db import Registry, RegistryPaths
from fisheye.utils.model_resolution_provenance import build_model_resolution_payload
from fisheye.utils.resolve_detect_model import Candidate, TargetProfile
from fisheye.utils.resolve_detect_model import _load_candidates, _load_target_profile, _resolve_recording_id


def _resolve_output(recording_dir: Path, explicit_output: Optional[Path]) -> Path:
    if explicit_output is not None:
        return explicit_output.expanduser().resolve()
    return (recording_dir / "zarr" / f"{recording_dir.name}_analysis.zarr").resolve()


def _pick_best_candidate(candidates: list[Candidate], *, require_unique: bool) -> Candidate:
    if not candidates:
        raise SystemExit("No eye-mask model candidates found.")
    best = candidates[0]
    if require_unique and len(candidates) > 1:
        if abs(candidates[0].weighted_score - candidates[1].weighted_score) < 1e-12:
            raise SystemExit("Top candidate score tied; rerun with --set-id to choose deterministically.")
    return best


def _resolution_payload(
    *,
    args: argparse.Namespace,
    argv: Optional[list[str]],
    recording_dir: Path,
    output_path: Path,
    registry_path: Path,
    recording_id: str,
    target: TargetProfile,
    selected: Candidate,
    candidates: list[Candidate],
    top_k: int,
    method: str,
) -> dict[str, Any]:
    target_payload = asdict(target)

    def _candidate_payload(item: Candidate) -> dict[str, Any]:
        return {
            "run_id": item.run_id,
            "set_id": item.set_id,
            "model_path": item.model_path,
            "score": item.weighted_score,
            "created_utc": item.created_utc,
            "status": item.status,
            "dataset_count": item.dataset_count,
            "feature_match_counts": item.feature_match_counts,
            "feature_weights_used": item.feature_weights_used,
        }

    selected_payload = _candidate_payload(selected)
    candidate_payloads = [_candidate_payload(item) for item in candidates[: max(0, int(top_k))]]

    return build_model_resolution_payload(
        tool="fisheye.utils.run_eye_masks_with_registry_model",
        args=args,
        argv=argv,
        task="eye_masks",
        method=method,
        registry_path=registry_path,
        recording_id=recording_id,
        target=target_payload,
        selected=selected_payload,
        candidates=candidate_payloads,
        parameters={
            "set_id_filter": args.set_id,
            "require_unique": bool(args.require_unique),
            "top_k": int(args.top_k),
            "include_non_success": bool(args.include_non_success),
            "dry_run": bool(args.dry_run),
            "method": args.method,
            "run_name": args.run_name,
            "crop_run": args.crop_run,
            "keypoints_run": args.keypoints_run,
            "batch_size": args.batch_size,
            "device": args.device,
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "max_det": args.max_det,
            "mask_threshold": args.mask_threshold,
            "adaptive_scale": args.adaptive_scale,
            "adaptive_cap": args.adaptive_cap,
            "no_retina_masks": bool(args.no_retina_masks),
            "proto_upsample_factor": args.proto_upsample_factor,
            "legacy_masks": bool(args.legacy_masks),
            "verbose": bool(args.verbose),
            "label_mode": args.label_mode,
            "write_binary_masks": bool(args.write_binary_masks),
            "no_use_crop": bool(args.no_use_crop),
        },
        inputs={
            "recording_dir": str(recording_dir),
            "output_zarr": str(output_path),
            "recording_id": recording_id,
            "target": target_payload,
        },
        artifacts={
            "selected_model": selected_payload,
            "candidate_models": candidate_payloads,
            "output_zarr": str(output_path),
        },
    )


def _write_model_resolution_provenance(
    *,
    zarr_path: Path,
    run_name: str,
    payload: dict[str, Any],
) -> None:
    root = zarr.open_group(str(zarr_path), mode="r+")
    eye_parent = root.get("eye_masks_runs")
    if eye_parent is None or run_name not in eye_parent:
        raise RuntimeError(f"eye-mask run not found for provenance annotation: eye_masks_runs/{run_name}")

    eye_group = eye_parent[run_name]
    selected = payload.get("selected", {}) if isinstance(payload.get("selected"), dict) else {}
    attrs = dict(eye_group.attrs)
    attrs["model_resolution_mode"] = "registry"
    attrs["model_resolution_task"] = "eye_masks"
    attrs["model_resolution_method"] = payload.get("method")
    attrs["model_resolution_registry_path"] = payload.get("registry_path")
    attrs["model_resolution_recording_id"] = payload.get("recording_id")
    attrs["model_resolution_selected_run_id"] = selected.get("run_id")
    attrs["model_resolution_selected_set_id"] = selected.get("set_id")
    attrs["model_resolution_selected_model_path"] = selected.get("model_path")
    attrs["model_resolution_selected_score"] = selected.get("score")
    attrs["model_resolution_selected_created_utc"] = selected.get("created_utc")
    attrs["model_resolution_resolved_at_utc"] = payload.get("resolved_at_utc")
    attrs["model_resolution_candidates_json"] = json.dumps(payload.get("candidates", []), sort_keys=True)

    provenance = attrs.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {}
    provenance["model_resolution"] = payload
    attrs["provenance"] = provenance
    eye_group.attrs.put(attrs)


def _latest_eye_masks_run(zarr_path: Path) -> Optional[str]:
    root = zarr.open_group(str(zarr_path), mode="r")
    parent = root.get("eye_masks_runs")
    if parent is None:
        return None
    latest = parent.attrs.get("latest")
    if latest is None:
        return None
    latest_name = str(latest)
    if latest_name not in parent:
        return None
    return latest_name


def _segment_eye_masks_yolo(**kwargs: Any) -> str:
    from fisheye.segmentation.eye_segmentation_yolo import segment_eye_masks_yolo

    return segment_eye_masks_yolo(**kwargs)


def _infer_unet_eye_masks(argv: list[str]) -> None:
    from fisheye.segmentation.infer_unet_eye_masks import main as infer_unet_eye_masks_main

    infer_unet_eye_masks_main(argv)


def _run_yolo(output_path: Path, model_path: str, args: argparse.Namespace) -> str:
    run_name = _segment_eye_masks_yolo(
        zarr_path=str(output_path),
        model_path=model_path,
        run_name=args.run_name,
        crop_run=args.crop_run,
        keypoints_run=args.keypoints_run,
        batch_size=args.batch_size,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        mask_threshold=args.mask_threshold,
        adaptive_scale=args.adaptive_scale,
        adaptive_cap=args.adaptive_cap,
        use_retina_masks=(not bool(args.no_retina_masks)),
        proto_upsample_factor=args.proto_upsample_factor,
        legacy_masks=bool(args.legacy_masks),
        verbose=bool(args.verbose),
        console=None,
    )
    if not run_name:
        raise RuntimeError("YOLO eye-mask inference did not produce a run name.")
    return run_name


def _run_unet(output_path: Path, model_path: str, args: argparse.Namespace) -> str:
    unet_argv: list[str] = [str(output_path), str(model_path)]
    if args.run_name:
        unet_argv.extend(["--run-name", str(args.run_name)])
    if args.crop_run:
        unet_argv.extend(["--crop-run", str(args.crop_run)])
    if args.keypoints_run:
        unet_argv.extend(["--keypoints-run", str(args.keypoints_run)])
    if args.batch_size is not None:
        unet_argv.extend(["--batch-size", str(args.batch_size)])
    if args.device:
        unet_argv.extend(["--device", str(args.device)])
    if args.label_mode:
        unet_argv.extend(["--label-mode", str(args.label_mode)])
    if args.write_binary_masks:
        unet_argv.append("--write-binary-masks")
    if not args.no_use_crop:
        unet_argv.append("--use-crop")
    _infer_unet_eye_masks(unet_argv)
    run_name = _latest_eye_masks_run(output_path)
    if not run_name:
        raise RuntimeError("U-Net eye-mask inference completed but no latest eye_masks run found.")
    return run_name


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording-dir", type=Path, required=True, help="Recording directory to process.")
    parser.add_argument("--output", type=Path, help="Optional explicit output zarr path.")
    parser.add_argument("--registry", type=Path, help="Optional registry sqlite path.")
    parser.add_argument("--set-id", type=str, help="Optional set filter during model resolution.")
    parser.add_argument(
        "--method",
        choices=("yolo", "unet"),
        default="yolo",
        help="Model-based eye-mask method to run after resolution.",
    )
    parser.add_argument("--require-unique", action="store_true", help="Fail if top scores tie.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of candidates to store in provenance.")
    parser.add_argument("--include-non-success", action="store_true", help="Include non-success training runs.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve model only; do not run eye-mask inference.")

    parser.add_argument("--run-name", type=str, default=None, help="Optional explicit eye_masks run name.")
    parser.add_argument("--crop-run", type=str, default=None, help="Optional explicit crop run name.")
    parser.add_argument("--keypoints-run", type=str, default=None, help="Optional explicit keypoints run name.")
    parser.add_argument("--batch-size", type=int, default=128, help="Inference batch size override.")
    parser.add_argument("--device", type=str, default=None, help="Optional torch device override.")

    parser.add_argument("--imgsz", type=int, default=None, help="YOLO-only image size override.")
    parser.add_argument("--conf", type=float, default=0.05, help="YOLO-only confidence threshold override.")
    parser.add_argument("--iou", type=float, default=0.5, help="YOLO-only IoU threshold override.")
    parser.add_argument("--max-det", type=int, default=2, help="YOLO-only max detections override.")
    parser.add_argument("--mask-threshold", type=float, default=0.05, help="YOLO-only mask threshold override.")
    parser.add_argument("--adaptive-scale", type=float, default=0.6, help="YOLO-only adaptive scale override.")
    parser.add_argument("--adaptive-cap", type=float, default=0.6, help="YOLO-only adaptive cap override.")
    parser.add_argument("--no-retina-masks", action="store_true", help="YOLO-only: disable retina masks.")
    parser.add_argument("--proto-upsample-factor", type=int, default=2, help="YOLO-only proto upsample factor.")
    parser.add_argument("--legacy-masks", action="store_true", help="YOLO-only legacy mask conversion path.")
    parser.add_argument("--verbose", action="store_true", help="YOLO-only verbose Ultralytics output.")

    parser.add_argument("--label-mode", choices=("union", "lr"), default=None, help="U-Net-only label-mode override.")
    parser.add_argument("--write-binary-masks", action="store_true", help="U-Net-only: write thresholded masks.")
    parser.add_argument("--no-use-crop", action="store_true", help="U-Net-only: do not pass --use-crop.")
    parser.add_argument("--json", action="store_true", help="Print resolved payload JSON.")
    args = parser.parse_args(argv)

    recording_dir = args.recording_dir.expanduser().resolve()
    registry_path = (args.registry or RegistryPaths.from_env(Path.cwd()).path).expanduser().resolve()
    output_path = _resolve_output(recording_dir, args.output)

    registry = Registry(registry_path)
    try:
        recording_id = _resolve_recording_id(
            registry,
            recording_id=None,
            recording_dir=recording_dir,
        )
        target = _load_target_profile(registry, recording_id)
        candidates = _load_candidates(
            registry,
            target=target,
            task="eye_masks",
            set_id_filter=args.set_id,
            include_non_success=bool(args.include_non_success),
        )
    finally:
        registry.close()

    best = _pick_best_candidate(candidates, require_unique=bool(args.require_unique))
    payload = _resolution_payload(
        args=args,
        argv=argv,
        recording_dir=recording_dir,
        output_path=output_path,
        registry_path=registry_path,
        recording_id=recording_id,
        target=target,
        selected=best,
        candidates=candidates,
        top_k=int(args.top_k),
        method=str(args.method),
    )

    if args.json or args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
    if args.dry_run:
        return 0

    if args.method == "unet":
        run_name = _run_unet(output_path, best.model_path, args)
    else:
        run_name = _run_yolo(output_path, best.model_path, args)

    _write_model_resolution_provenance(
        zarr_path=output_path,
        run_name=run_name,
        payload=payload,
    )
    print("Model resolution provenance written")
    print(f"  output_zarr: {output_path}")
    print(f"  eye_masks_run: {run_name}")
    print(f"  selected_model: {best.model_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
