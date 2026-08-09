"""YOLO detection inference helper.

Wraps :func:`fisheye.detection.detect_yolo.detect_yolo` to match the
`fisheye.inference` CLI layout so both detection and pose inference live in
one namespace.
"""

from __future__ import annotations

import argparse
from json import JSONDecodeError, loads as json_loads
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import zarr
from rich.console import Console

from ..detection.candidate_builder import build_detection_candidate
from ..shared.detection_candidate import (
    DEFAULT_DETECT_FRAME_SHARD_ROWS,
    DEFAULT_DETECT_ROW_SHARD_ROWS,
)
from ..registry.stage_complete import emit_stage_completion
from ..shared.zarr_run_completion import resolve_authoritative_run_name


def _normalize_text(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", "ignore")
    text = str(value).strip()
    return text or None


def _coerce_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_mapping(value: object) -> Optional[dict[str, Any]]:
    if isinstance(value, dict):
        return value
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", "ignore")
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        payload = json_loads(text)
    except JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _extract_detect_coverage_pct(detect_group: object) -> Optional[float]:
    if detect_group is None or not hasattr(detect_group, "attrs"):
        return None

    summary = _parse_mapping(detect_group.attrs.get("summary_statistics"))  # type: ignore[attr-defined]
    if summary is not None:
        coverage = _coerce_float(summary.get("percent_frames_with_detections"))
        if coverage is not None:
            return coverage

    if not hasattr(detect_group, "get"):
        return None
    for key in ("frame_counts", "n_detections"):
        try:
            counts_ds = detect_group.get(key)  # type: ignore[attr-defined]
        except Exception:
            counts_ds = None
        if counts_ds is None:
            continue
        try:
            counts = np.asarray(counts_ds[:])
        except Exception:
            continue
        if counts.size == 0:
            continue
        present = int(np.sum(counts > 0))
        return (present / float(counts.shape[0])) * 100.0

    return None


def _extract_detect_quality_pointer(detect_group: object) -> tuple[Optional[str], Optional[str], Optional[float]]:
    if detect_group is None or not hasattr(detect_group, "get"):
        return None, None, None
    try:
        quality_parent = detect_group.get("quality_reports")  # type: ignore[attr-defined]
    except Exception:
        quality_parent = None
    if quality_parent is None or not hasattr(quality_parent, "attrs"):
        return None, None, None

    quality_run = _normalize_text(resolve_authoritative_run_name(quality_parent))
    if not quality_run or quality_run not in quality_parent:
        return quality_run, None, None

    quality_group = quality_parent[quality_run]
    if not hasattr(quality_group, "attrs"):
        return quality_run, None, None

    quality_score = _parse_mapping(quality_group.attrs.get("quality_score"))
    if quality_score is None:
        return quality_run, None, None
    quality_grade = _normalize_text(quality_score.get("grade"))
    quality_overall = _coerce_float(quality_score.get("overall_score"))
    return quality_run, quality_grade, quality_overall


def _write_detect_step_status(
    *,
    output_zarr: Path,
    run_name: Optional[str],
    status: str,
    reason: str,
    video_path: Optional[Path],
    console: Optional[Console],
    error: Optional[BaseException] = None,
) -> None:
    output_zarr_path = output_zarr.expanduser()
    if not output_zarr_path.exists():
        return

    try:
        root = zarr.open_group(str(output_zarr_path), mode="r", use_consolidated=False)

        detect_group = None
        resolved_run_name = _normalize_text(run_name)
        detect_parent = root.get("detect_runs")
        if detect_parent is not None:
            if resolved_run_name and resolved_run_name in detect_parent:
                detect_group = detect_parent[resolved_run_name]
            else:
                latest_run = (
                    _normalize_text(resolve_authoritative_run_name(detect_parent))
                    if hasattr(detect_parent, "attrs")
                    else None
                )
                if latest_run and latest_run in detect_parent:
                    detect_group = detect_parent[latest_run]
                    resolved_run_name = resolved_run_name or latest_run

        method = None
        coverage_pct = None
        quality_run = None
        quality_grade = None
        quality_score = None
        if detect_group is not None and hasattr(detect_group, "attrs"):
            method = _normalize_text(detect_group.attrs.get("detection_method")) or _normalize_text(  # type: ignore[attr-defined]
                detect_group.attrs.get("method")  # type: ignore[attr-defined]
            )
            coverage_pct = _extract_detect_coverage_pct(detect_group)
            quality_run, quality_grade, quality_score = _extract_detect_quality_pointer(detect_group)

        detect_group_path = f"detect_runs/{resolved_run_name}" if resolved_run_name else None
        details: dict[str, Any] = {
            "reason": reason,
            "detect_group_path": detect_group_path,
            "coverage_summary_path": f"{detect_group_path}.attrs.summary_statistics" if detect_group_path else None,
            "quality_reports_path": f"{detect_group_path}/quality_reports" if detect_group_path else None,
            "detect_quality_run": quality_run,
            "detect_quality_path": (
                f"{detect_group_path}/quality_reports/{quality_run}"
                if detect_group_path and quality_run
                else None
            ),
            "detect_quality_grade": quality_grade,
            "detect_quality_score": quality_score,
            "video_path": str(video_path) if video_path is not None else None,
            "error": str(error) if error is not None else None,
        }
        details = {key: value for key, value in details.items() if value is not None}

        emit_stage_completion(
            root,
            output_zarr_path,
            step_name="detect",
            status=status,
            source="runtime_predict_detections",
            run_name=resolved_run_name,
            method=method,
            coverage_pct=coverage_pct,
            review_status_json=None,
            details_json=details,
            console=console,
            warning_label="detect",
            auto_registry_from_env=True,
            require_env_registry_exists=False,
            invalidate_on_ok=True,
            trigger_run_name=resolved_run_name,
            upsert_dataset_row=False,
        )
    except Exception as exc:
        if console is not None:
            console.print(
                "[yellow]Warning:[/yellow] Failed to update recording step status "
                f"for detect: {exc}"
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run YOLO detection inference and write a detect_runs entry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Minimal (uses config defaults)
  python -m fisheye.inference.predict_detections --video video.mp4

  # Specify model and output Zarr
  python -m fisheye.inference.predict_detections --video video.mp4 --model runs/detect/best.pt --output detections.zarr

  # Override thresholds and batch size
  python -m fisheye.inference.predict_detections --video video.mp4 --model best.pt --conf 0.35 --batch-size 64

  # Force CPU inference
  python -m fisheye.inference.predict_detections --video video.mp4 --cpu
""",
    )
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--model", dest="model_path", help="YOLO detection model (.pt)")
    parser.add_argument("--output", dest="output_zarr", help="Output Zarr path (optional)")
    parser.add_argument("--config", dest="config_path", help="YAML config with defaults")
    parser.add_argument("--conf", dest="conf_threshold", type=float, help="Confidence threshold override")
    parser.add_argument("--iou", dest="iou_threshold", type=float, help="IoU threshold override")
    parser.add_argument("--max-det", dest="max_det", type=int, help="Max detections per frame override")
    parser.add_argument("--batch-size", type=int, help="Batch size override")
    detect_storage_group = parser.add_mutually_exclusive_group()
    detect_storage_group.add_argument(
        "--detect-row-shard-rows",
        type=int,
        default=DEFAULT_DETECT_ROW_SHARD_ROWS,
        help=(
            "Requested outer rows for indexed-sharded detection arrays "
            f"(default: {DEFAULT_DETECT_ROW_SHARD_ROWS})."
        ),
    )
    detect_storage_group.add_argument(
        "--no-detect-sharding",
        action="store_const",
        dest="detect_row_shard_rows",
        const=None,
        help="Use ordinary chunks for YOLO detection outputs.",
    )
    parser.add_argument(
        "--detect-frame-shard-rows",
        type=int,
        default=DEFAULT_DETECT_FRAME_SHARD_ROWS,
        help="Outer row count for frame-count arrays when detection sharding is enabled.",
    )
    parser.add_argument(
        "--resize-dims",
        nargs="+",
        type=int,
        help="Canonical inference size override [h w] (or one value for square); mapped to YOLO imgsz.",
    )
    parser.add_argument(
        "--imgsz",
        nargs="+",
        type=int,
        help="Legacy alias for YOLO inference size; normalized into --resize-dims.",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    parser.add_argument("--gpu", action="store_true", help="Force GPU inference")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    console = Console()

    if args.cpu and args.gpu:
        raise ValueError("--cpu and --gpu are mutually exclusive")
    use_gpu: Optional[bool]
    if args.cpu:
        use_gpu = False
    elif args.gpu:
        use_gpu = True
    else:
        use_gpu = None
    output_zarr_path = (
        Path(args.output_zarr).expanduser()
        if args.output_zarr
        else video_path.parent / f"{video_path.stem}_detections.zarr"
    )

    try:
        run_name_raw = build_detection_candidate(
            video_path=str(video_path),
            model_path=args.model_path,
            output_zarr=args.output_zarr,
            config_path=args.config_path,
            conf_threshold=args.conf_threshold,
            iou_threshold=args.iou_threshold,
            max_det=args.max_det,
            batch_size=args.batch_size,
            resize_dims=args.resize_dims,
            imgsz=args.imgsz,
            console=console,
            use_gpu=use_gpu,
            detect_row_shard_rows=args.detect_row_shard_rows,
            detect_frame_shard_rows=args.detect_frame_shard_rows,
        )
    except Exception as exc:
        _write_detect_step_status(
            output_zarr=output_zarr_path,
            run_name=None,
            status="error",
            reason="detect_inference_failed",
            video_path=video_path,
            console=console,
            error=exc,
        )
        raise

    run_name = _normalize_text(run_name_raw)
    _write_detect_step_status(
        output_zarr=output_zarr_path,
        run_name=run_name,
        status="ok",
        reason="detect_run_completed",
        video_path=video_path,
        console=console,
    )


if __name__ == "__main__":
    main()
