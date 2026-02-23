"""CLI wrapper for YOLO eye-mask inference on FishEye Zarr crops."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import zarr

from rich.console import Console

from ..registry.db import Registry, RegistryPaths, resolve_dataset_id
from ..registry.status_ledger import upsert_recording_step_status
from ..segmentation.eye_segmentation_yolo import segment_eye_masks_yolo

_STATUS_SOURCE = "runtime_predict_eye_masks"


def _as_optional_text(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_mapping(value: object) -> Optional[Dict[str, object]]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", "ignore")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return dict(parsed)
    return None


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean_details(payload: Dict[str, object]) -> Dict[str, object]:
    return {key: value for key, value in payload.items() if value is not None}


def _open_root(zarr_path: Path) -> zarr.Group:
    try:
        return zarr.open(str(zarr_path), mode="r", use_consolidated=False)
    except TypeError:
        try:
            return zarr.open(str(zarr_path), mode="r", consolidated=False)
        except TypeError:
            return zarr.open(str(zarr_path), mode="r")


def _zarr_mtime_ns(zarr_path: Path) -> Optional[int]:
    try:
        return int(zarr_path.stat().st_mtime_ns)
    except OSError:
        return None


def _classify_failure(exc: Exception) -> Tuple[str, str]:
    message = str(exc).lower()
    if "missing crop_runs" in message or "no crop run available" in message:
        return "missing", "crop_missing"
    if "no rois available" in message:
        return "missing", "no_rois"
    if "zarr path not found" in message:
        return "missing", "zarr_missing"
    if "already exists" in message:
        return "error", "run_exists"
    if "model path not found" in message:
        return "error", "model_missing"
    return "error", "runtime_error"


def _write_eye_masks_status(
    *,
    zarr_path: str,
    status: str,
    reason: str,
    run_name: Optional[str],
    requested_crop_run: Optional[str],
    method_hint: Optional[str],
    error_text: Optional[str] = None,
    console: Optional[Console] = None,
) -> None:
    zarr_file = Path(zarr_path).expanduser()
    if not zarr_file.exists():
        return

    try:
        root = _open_root(zarr_file)
    except Exception as exc:
        if console is not None:
            console.print(f"[yellow]Status ledger write skipped: unable to open Zarr ({exc}).[/yellow]")
        return

    dataset_id, session_uuid = resolve_dataset_id(root, zarr_file)
    root_attrs = getattr(root, "attrs", {})
    recording_id = _as_optional_text(session_uuid) or _as_optional_text(root_attrs.get("recording_id"))

    run_attrs: Dict[str, object] = {}
    eye_parent = root.get("eye_masks_runs") if hasattr(root, "get") else None
    if run_name and eye_parent is not None and run_name in eye_parent:
        run_group = eye_parent[run_name]
        run_attrs = dict(getattr(run_group, "attrs", {}))

    method = _as_optional_text(run_attrs.get("method")) or _as_optional_text(method_hint)
    pair_rate = _safe_float(run_attrs.get("successful_roi_pair_rate"))
    coverage_pct = None
    if pair_rate is not None:
        coverage_pct = float(pair_rate) * 100.0 if pair_rate <= 1.0 else float(pair_rate)

    details = _clean_details(
        {
            "reason": reason,
            "source_crop_run": _as_optional_text(run_attrs.get("source_crop_run")),
            "source_keypoints_run": _as_optional_text(
                run_attrs.get("source_keypoints_run") or run_attrs.get("source_keypoint_run")
            ),
            "source_keypoint_group": _as_optional_text(run_attrs.get("source_keypoint_group")),
            "successful_roi_pairs": run_attrs.get("successful_roi_pairs"),
            "total_rois": run_attrs.get("total_rois"),
            "successful_roi_pair_rate": pair_rate,
            "probability_stats": _coerce_mapping(run_attrs.get("probability_stats")),
            "ellipse_soft_available": run_attrs.get("ellipse_soft_available"),
            "requested_crop_run": _as_optional_text(requested_crop_run),
            "error": _as_optional_text(error_text),
        }
    )
    review_status = _coerce_mapping(run_attrs.get("eye_mask_review_status"))

    try:
        registry_path = RegistryPaths.from_env(Path.cwd()).path
        registry = Registry(registry_path)
        registry.upsert_dataset(
            dataset_id,
            session_uuid=session_uuid,
            zarr_path=zarr_file,
            recording_id=recording_id,
            zarr_purpose=_as_optional_text(root_attrs.get("zarr_purpose")),
            zarr_use=_as_optional_text(root_attrs.get("zarr_use")),
        )
        upsert_recording_step_status(
            registry,
            dataset_id=dataset_id,
            recording_id=recording_id,
            step_name="eye_masks",
            status=status,
            run_name=run_name,
            method=method,
            coverage_pct=coverage_pct,
            review_status_json=review_status,
            details_json=details,
            source=_STATUS_SOURCE,
            zarr_mtime_ns=_zarr_mtime_ns(zarr_file),
        )
    except Exception as exc:
        if console is not None:
            console.print(f"[yellow]Status ledger write failed for eye_masks: {exc}[/yellow]")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run YOLO eye segmentation on FishEye Zarr crops",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m fisheye.inference.predict_eye_masks --model weights/best.pt --zarr video.zarr
  python -m fisheye.inference.predict_eye_masks --model best.pt --zarr video.zarr --batch-size 128
""",
    )
    parser.add_argument("--model", required=True, help="Path to trained YOLO segmentation model (.pt)")
    parser.add_argument("--zarr", required=True, help="Path to FishEye Zarr archive")
    parser.add_argument("--run-name", help="Optional custom run name inside eye_masks_runs")
    parser.add_argument("--crop-run", help="Optional crop run name to use (defaults to latest)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for inference")
    parser.add_argument("--device", default=None, help="Torch device string (e.g. cuda:0, cpu)")
    parser.add_argument("--imgsz", type=int, default=None, help="Override inference image size")
    parser.add_argument("--conf", type=float, default=0.05, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for NMS")
    parser.add_argument("--max-det", type=int, default=2, help="Max detections per ROI")
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.05,
        help="Global floor for adaptive binarization (final threshold = max(floor, min(adaptive_cap, adaptive_scale * pmax)))",
    )
    parser.add_argument(
        "--adaptive-scale",
        type=float,
        default=0.6,
        help="Multiplier applied to each ROI's peak probability for adaptive thresholding",
    )
    parser.add_argument(
        "--adaptive-cap",
        type=float,
        default=0.6,
        help="Upper cap applied to the adaptive threshold (set <=1.0)",
    )
    parser.add_argument(
        "--proto-upsample-factor",
        type=int,
        default=2,
        help="Optional upsampling factor applied to YOLO mask prototypes prior to reconstruction",
    )
    parser.add_argument(
        "--no-retina-masks",
        dest="use_retina_masks",
        action="store_false",
        help="Disable Ultralytics retina mask outputs (use prototype reconstruction instead)",
    )
    parser.add_argument(
        "--legacy-masks",
        action="store_true",
        help="Disable retina/soft-mask hooks and use only Ultralytics binary masks (compatibility mode)",
    )
    parser.set_defaults(use_retina_masks=True, legacy_masks=False)
    parser.add_argument("--verbose", action="store_true", help="Enable verbose Ultralytics output")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    console = Console()
    try:
        run_name = segment_eye_masks_yolo(
            zarr_path=args.zarr,
            model_path=args.model,
            run_name=args.run_name,
            crop_run=args.crop_run,
            batch_size=args.batch_size,
            device=args.device,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            mask_threshold=args.mask_threshold,
            adaptive_scale=args.adaptive_scale,
            adaptive_cap=args.adaptive_cap,
            use_retina_masks=args.use_retina_masks if not args.legacy_masks else False,
            proto_upsample_factor=args.proto_upsample_factor,
            legacy_masks=args.legacy_masks,
            verbose=args.verbose,
            console=console,
        )
    except Exception as exc:
        status, reason = _classify_failure(exc)
        _write_eye_masks_status(
            zarr_path=args.zarr,
            status=status,
            reason=reason,
            run_name=None,
            requested_crop_run=args.crop_run,
            method_hint="yolo_eye_segmentation",
            error_text=str(exc),
            console=console,
        )
        raise

    if run_name:
        _write_eye_masks_status(
            zarr_path=args.zarr,
            status="ok",
            reason="present",
            run_name=run_name,
            requested_crop_run=args.crop_run,
            method_hint="yolo_eye_segmentation",
            console=console,
        )
    else:
        _write_eye_masks_status(
            zarr_path=args.zarr,
            status="missing",
            reason="no_rois",
            run_name=None,
            requested_crop_run=args.crop_run,
            method_hint="yolo_eye_segmentation",
            console=console,
        )


if __name__ == "__main__":
    main()
