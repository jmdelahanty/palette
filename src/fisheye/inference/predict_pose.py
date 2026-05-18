"""CLI wrapper for YOLO pose inference on FishEye Zarr crops."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import zarr
from rich.console import Console

from ..detection.detect_keypoints_yolo import detect_keypoints_yolo
from ..registry.db import Registry, RegistryPaths
from ..shared.registry_stage_complete import DatasetMetadata, emit_stage_completion
from ..shared.type_conversions import as_float, normalize_attr

_KEYPOINT_STEP_NAME = "keypoints"
_STATUS_SOURCE = "runtime_predict_pose"


@dataclass(frozen=True)
class _StatusContext:
    registry_path: Path
    dataset_id: str
    recording_id: Optional[str]
    zarr_path: Path


def _as_float(value: object) -> Optional[float]:
    return as_float(value)


def _resolve_status_context(zarr_path: str) -> Optional[_StatusContext]:
    resolved_zarr = Path(zarr_path).expanduser().resolve()
    registry_path = RegistryPaths.from_env(Path.cwd()).path.expanduser().resolve()
    if not registry_path.exists():
        return None

    registry = Registry(registry_path)
    try:
        path_hash = hashlib.sha256(str(resolved_zarr).encode("utf-8")).hexdigest()
        row = registry.conn.execute(
            """
            SELECT dataset_id, recording_id
            FROM datasets
            WHERE path_hash = ?
            ORDER BY COALESCE(last_seen_utc, '') DESC, dataset_id
            LIMIT 1;
            """,
            (path_hash,),
        ).fetchone()

        if row is None:
            path_variants = tuple(
                dict.fromkeys(
                    [
                        str(resolved_zarr),
                        str(Path(zarr_path).expanduser()),
                        str(Path(zarr_path)),
                    ]
                )
            )
            for candidate in path_variants:
                row = registry.conn.execute(
                    """
                    SELECT dataset_id, recording_id
                    FROM datasets
                    WHERE zarr_path = ?
                    ORDER BY COALESCE(last_seen_utc, '') DESC, dataset_id
                    LIMIT 1;
                    """,
                    (candidate,),
                ).fetchone()
                if row is not None:
                    break

        if row is None:
            row = registry.conn.execute(
                """
                SELECT dataset_id, recording_id
                FROM datasets
                WHERE lower(COALESCE(zarr_path, '')) = lower(?)
                ORDER BY COALESCE(last_seen_utc, '') DESC, dataset_id
                LIMIT 1;
                """,
                (str(resolved_zarr),),
            ).fetchone()

        if row is None:
            return None

        dataset_id = normalize_attr(row["dataset_id"])
        if dataset_id is None:
            return None
        recording_id = normalize_attr(row["recording_id"])
        return _StatusContext(
            registry_path=registry_path,
            dataset_id=dataset_id,
            recording_id=recording_id,
            zarr_path=resolved_zarr,
        )
    finally:
        registry.close()


def _extract_success_coverage_pct(summary: Mapping[str, object]) -> Optional[float]:
    pct = _as_float(summary.get("success_rate_percent"))
    if pct is not None:
        return pct
    successful = _as_float(summary.get("successful_detections"))
    total = _as_float(summary.get("total_rois"))
    if successful is None or total is None or total <= 0:
        return None
    return successful / total * 100.0


def _collect_keypoint_run_payload(
    zarr_path: Path,
    run_name: str,
) -> Tuple[Optional[str], Optional[float], Dict[str, object]]:
    method: Optional[str] = None
    coverage_pct: Optional[float] = None
    details: Dict[str, object] = {
        "reason": "completed",
        "run_group": "keypoints_runs",
    }

    try:
        root = zarr.open_group(str(zarr_path), mode="r")
    except Exception as exc:
        details["metadata_error"] = str(exc)
        return method, coverage_pct, details

    keypoint_parent = root.get("keypoints_runs")
    if keypoint_parent is None or run_name not in keypoint_parent:
        details["run_present"] = False
        return method, coverage_pct, details

    run_group = keypoint_parent[run_name]
    details["run_present"] = True
    method = normalize_attr(run_group.attrs.get("method"))

    summary = run_group.attrs.get("summary_statistics")
    if isinstance(summary, Mapping):
        summary_payload: Dict[str, object] = {}
        for key in (
            "total_rois",
            "successful_detections",
            "failed_detections",
            "success_rate_percent",
            "mean_confidence",
        ):
            value = _as_float(summary.get(key))
            if value is not None:
                summary_payload[key] = value
        if summary_payload:
            details["summary_statistics"] = summary_payload
        coverage_pct = _extract_success_coverage_pct(summary)

    if coverage_pct is None:
        coverage_pct = _as_float(run_group.attrs.get("success_rate"))
        if coverage_pct is not None and coverage_pct <= 1.0:
            coverage_pct = coverage_pct * 100.0

    for key in (
        "source_crop_run",
        "source_detect_run",
        "source_refined_run",
        "model_name",
        "device",
        "model_resolution_selected_run_id",
        "model_resolution_selected_set_id",
    ):
        text = normalize_attr(run_group.attrs.get(key))
        if text is not None:
            details[key] = text

    return method, coverage_pct, details


def _collect_no_roi_payload(zarr_path: Path) -> Dict[str, object]:
    details: Dict[str, object] = {
        "reason": "no_rois",
        "run_group": "keypoints_runs",
    }
    try:
        root = zarr.open_group(str(zarr_path), mode="r")
    except Exception:
        return details
    crop_parent = root.get("crop_runs")
    if crop_parent is None:
        return details
    latest_crop = normalize_attr(crop_parent.attrs.get("latest"))
    if latest_crop is not None:
        details["source_crop_run"] = latest_crop
    return details


def _classify_keypoint_failure(exc: Exception) -> Tuple[str, str]:
    if isinstance(exc, FileNotFoundError):
        return "error", "file_missing"
    message = str(exc).lower()
    if "missing crop_runs" in message or "no crop run found" in message:
        return "absent", "crop_missing"
    if "no rois found in crop run" in message:
        return "missing", "no_rois"
    return "error", "inference_failed"


def _emit_keypoint_status(
    *,
    context: Optional[_StatusContext],
    status: str,
    run_name: Optional[str],
    method: Optional[str],
    coverage_pct: Optional[float],
    details: Dict[str, object],
    console: Console,
) -> None:
    if context is None:
        return

    registry = Registry(context.registry_path)
    try:
        metadata = DatasetMetadata(
            dataset_id=context.dataset_id,
            session_uuid=None,
            recording_id=context.recording_id,
            zarr_use=None,
            zarr_purpose=None,
            source_layout=None,
            source_frame_index_path=None,
            source_recording_frame_index_path=None,
            source_frame_index_schema=None,
        )
        emit_stage_completion(
            None,
            context.zarr_path,
            step_name=_KEYPOINT_STEP_NAME,
            status=status,
            source=_STATUS_SOURCE,
            run_name=run_name,
            method=method,
            coverage_pct=coverage_pct,
            details_json=details,
            console=console,
            warning_label=_KEYPOINT_STEP_NAME,
            registry=registry,
            auto_registry_from_env=False,
            invalidate_on_ok=True,
            trigger_run_name=run_name,
            metadata=metadata,
            upsert_dataset_row=False,
        )
    finally:
        registry.close()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run YOLO pose inference on FishEye Zarr crops",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m fisheye.inference.predict_pose --model weights/best.pt --zarr video.zarr
  python -m fisheye.inference.predict_pose --model best.pt --zarr video.zarr --batch-size 128
""",
    )
    parser.add_argument("--model", required=True, help="Path to trained YOLO pose model (.pt)")
    parser.add_argument("--zarr", required=True, help="Path to FishEye Zarr archive")
    parser.add_argument("--run-name", help="Optional custom run name inside keypoints_runs")
    parser.add_argument("--crop-run", help="Optional crop run name to use (defaults to latest)")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for inference")
    parser.add_argument("--device", default=None, help="Torch device string (e.g. cuda:0, cpu)")
    parser.add_argument("--imgsz", type=int, default=None, help="Override inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for NMS")
    parser.add_argument("--max-det", type=int, default=1, help="Max detections per ROI")
    parser.add_argument(
        "--roi-cache-policy",
        choices=("never", "auto", "always"),
        default="auto",
        help="Temporary ROI cache policy for geometry-only crop runs (default: auto).",
    )
    parser.add_argument(
        "--roi-cache-dir",
        type=Path,
        default=None,
        help="Optional scratch directory for temporary ROI caches.",
    )
    parser.add_argument(
        "--roi-live-acceleration",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="Live ROI read acceleration for geometry-only crop runs (default: auto).",
    )
    parser.add_argument(
        "--roi-live-gpu-chunk-frames",
        type=int,
        default=32,
        help="Frame batch size for GPU-accelerated live ROI reads (default: 32).",
    )
    parser.add_argument(
        "--profile-timings",
        action="store_true",
        help="Collect per-stage timing diagnostics and store them in the output run attrs.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose Ultralytics output")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    console = Console()
    zarr_path = Path(args.zarr).expanduser().resolve()
    status_context = _resolve_status_context(str(zarr_path))

    try:
        run_name = detect_keypoints_yolo(
            zarr_path=str(zarr_path),
            model_path=args.model,
            run_name=args.run_name,
            crop_run=args.crop_run,
            batch_size=args.batch_size,
            device=args.device,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            roi_cache_policy=args.roi_cache_policy,
            roi_cache_dir=args.roi_cache_dir,
            roi_live_acceleration=args.roi_live_acceleration,
            roi_live_gpu_chunk_frames=args.roi_live_gpu_chunk_frames,
            profile_timings=args.profile_timings,
            verbose=args.verbose,
            registry=(status_context.registry_path if status_context is not None else None),
            console=console,
        )
    except Exception as exc:
        status, reason = _classify_keypoint_failure(exc)
        _emit_keypoint_status(
            context=status_context,
            status=status,
            run_name=None,
            method="yolo_pose",
            coverage_pct=None,
            details={
                "reason": reason,
                "error_type": exc.__class__.__name__,
                "error": str(exc),
            },
            console=console,
        )
        raise

    if run_name:
        return

    _emit_keypoint_status(
        context=status_context,
        status="missing",
        run_name=None,
        method="yolo_pose",
        coverage_pct=None,
        details=_collect_no_roi_payload(zarr_path),
        console=console,
    )


if __name__ == "__main__":
    main()
