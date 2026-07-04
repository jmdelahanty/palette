"""Import acquisition-time crop-recorder boxes as a normal detect run."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Optional, Sequence

import numpy as np
import zarr

from fisheye.diagnostics.compare_realtime_offline_detections import (
    infer_recording_dir_from_zarr,
    load_crop_meta_realtime_detection_rows,
    resolve_crop_meta_path,
)
from fisheye.shared.stage_provenance import build_stage_provenance, write_stage_provenance
from fisheye.shared.zarr_run_completion import mark_run_complete, mark_run_failed, mark_run_started, require_runs_parent
from fisheye.shared.system_metadata import get_environment_info, get_git_info


SCHEMA_ID = "palette.acquisition_detections_import.v1"
DEFAULT_RUN_PREFIX = "detect_acquisition_crop_meta"


@dataclass(frozen=True)
class AcquisitionDetectImportResult:
    zarr_path: str
    recording_dir: str
    crop_meta_path: str
    run_name: str
    run_path: str
    total_frames: int
    total_detections: int
    frames_with_detections: int
    blank_frame_count: int
    no_detection_frame_count: int
    source_width: int
    source_height: int
    applied: bool


def utc_run_name(prefix: str = DEFAULT_RUN_PREFIX) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    return f"{prefix}_{stamp}"


def _open_root(zarr_path: Path, *, mode: str) -> zarr.Group:
    return zarr.open_group(str(zarr_path), mode=mode, use_consolidated=False)


def _attr_int(attrs: Any, *keys: str) -> Optional[int]:
    for key in keys:
        value = attrs.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except Exception:
            continue
    return None


def _load_recording_manifest(recording_dir: Path) -> dict[str, Any]:
    manifest_path = recording_dir / "recording_manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _manifest_stream(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    video_streams = manifest.get("video_streams")
    if not isinstance(video_streams, dict):
        return {}
    streams = video_streams.get("streams")
    if not isinstance(streams, dict):
        return {}
    stream = streams.get(name)
    return stream if isinstance(stream, dict) else {}


def _stream_int(stream: dict[str, Any], *keys: str) -> Optional[int]:
    for key in keys:
        value = stream.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except Exception:
            continue
    return None


def resolve_source_dimensions(
    root: zarr.Group,
    *,
    recording_dir: Path,
    source_width: Optional[int],
    source_height: Optional[int],
) -> tuple[int, int]:
    width = source_width or _attr_int(
        root.attrs,
        "source_video_width",
        "source_full_width",
        "video_width",
        "width",
        "palette_video_width",
    )
    height = source_height or _attr_int(
        root.attrs,
        "source_video_height",
        "source_full_height",
        "video_height",
        "height",
        "palette_video_height",
    )

    raw_video = root.get("raw_video")
    if raw_video is not None:
        width = width or _attr_int(raw_video.attrs, "source_video_width", "source_full_width", "width", "video_width")
        height = height or _attr_int(raw_video.attrs, "source_video_height", "source_full_height", "height", "video_height")

    manifest = _load_recording_manifest(recording_dir)
    full_stream = _manifest_stream(manifest, "full")
    width = width or _stream_int(full_stream, "width", "source_width")
    height = height or _stream_int(full_stream, "height", "source_height")

    if width is None or height is None or width <= 0 or height <= 0:
        raise ValueError(
            "Could not resolve source full-frame dimensions; pass --source-width and --source-height."
        )
    return int(width), int(height)


def _resolve_total_frames(root: zarr.Group, crop_frame_indices: np.ndarray) -> int:
    total = _attr_int(root.attrs, "total_frames", "n_frames", "source_video_total_frames")
    raw_video = root.get("raw_video")
    if raw_video is not None:
        total = total or _attr_int(raw_video.attrs, "total_frames", "n_frames", "source_video_total_frames")
    crop_total = int(np.max(crop_frame_indices)) + 1 if crop_frame_indices.size else 0
    return max(int(total or 0), crop_total)


def _bbox_img_xyxy_to_norm_cxcywh(bbox_img_xyxy: np.ndarray, *, width: int, height: int) -> np.ndarray:
    bbox = np.asarray(bbox_img_xyxy, dtype=np.float64).reshape(-1, 4)
    x1 = bbox[:, 0]
    y1 = bbox[:, 1]
    x2 = bbox[:, 2]
    y2 = bbox[:, 3]
    out = np.column_stack(
        [
            ((x1 + x2) * 0.5) / float(width),
            ((y1 + y2) * 0.5) / float(height),
            (x2 - x1) / float(width),
            (y2 - y1) / float(height),
        ]
    )
    return out.astype(np.float32, copy=False)


def _chunks_1d(n: int) -> tuple[int]:
    return (max(1, min(int(n), 65_536)),)


def _chunks_2d(n: int, width: int) -> tuple[int, int]:
    return (max(1, min(int(n), 8192)), int(width))


def _delete_if_present(group: zarr.Group, name: str) -> None:
    if name in group:
        del group[name]


def _create_array(group: zarr.Group, name: str, data: np.ndarray) -> None:
    _delete_if_present(group, name)
    arr = np.asarray(data)
    chunks = _chunks_1d(arr.shape[0]) if arr.ndim == 1 else _chunks_2d(arr.shape[0], arr.shape[1])
    group.create_array(name, data=arr, chunks=chunks, overwrite=True)


def _manifest_crop_attrs(recording_dir: Path) -> dict[str, Any]:
    manifest = _load_recording_manifest(recording_dir)
    crop_stream = _manifest_stream(manifest, "crop")
    return {
        "selection_policy": crop_stream.get("selection_policy"),
        "blank_frame_policy": crop_stream.get("blank_frame_policy"),
        "crop_video": crop_stream.get("video"),
        "crop_metadata": crop_stream.get("metadata"),
        "source_geometry_coordinate_space": crop_stream.get("source_geometry_coordinate_space"),
        "video_pixel_coordinate_space": crop_stream.get("video_pixel_coordinate_space"),
    }


def import_acquisition_detections_to_detect_run(
    zarr_path: Path,
    *,
    recording_dir: Optional[Path] = None,
    crop_meta_path: Optional[Path] = None,
    run_name: Optional[str] = None,
    source_width: Optional[int] = None,
    source_height: Optional[int] = None,
    class_id: int = 0,
    overwrite: bool = False,
    apply: bool = False,
) -> AcquisitionDetectImportResult:
    zarr_path = Path(zarr_path)
    resolved_recording_dir = Path(recording_dir) if recording_dir is not None else infer_recording_dir_from_zarr(zarr_path)
    resolved_crop_meta = resolve_crop_meta_path(
        zarr_path,
        recording_dir=resolved_recording_dir,
        crop_meta_path=crop_meta_path,
        required=True,
    )
    if resolved_crop_meta is None:
        raise ValueError("No crop metadata resolved.")

    detections, crop_rows = load_crop_meta_realtime_detection_rows(resolved_crop_meta)
    root = _open_root(zarr_path, mode="a" if apply else "r")
    width, height = resolve_source_dimensions(
        root,
        recording_dir=resolved_recording_dir,
        source_width=source_width,
        source_height=source_height,
    )
    total_frames = _resolve_total_frames(root, crop_rows.frame_indices)

    frame_indices = detections.frame_indices.astype(np.int32, copy=False)
    bbox_img_xyxy = detections.bbox_img_xyxy.astype(np.float64, copy=False)
    bbox_norm = _bbox_img_xyxy_to_norm_cxcywh(bbox_img_xyxy, width=width, height=height)
    scores = detections.confidence.astype(np.float32, copy=False)
    class_ids = np.full(frame_indices.shape, int(class_id), dtype=np.int32)
    frame_counts = np.bincount(frame_indices.astype(np.int64, copy=False), minlength=total_frames).astype(np.int32)
    source_crop_xywh = crop_rows.crop_xywh[np.searchsorted(crop_rows.frame_indices, detections.frame_indices)].astype(
        np.float32,
        copy=False,
    )
    source_crop_meta_row_indices = detections.row_indices.astype(np.int64, copy=False)
    source_recording_frame_ids = detections.frame_indices.astype(np.int64, copy=False) + 1
    resolved_run_name = run_name or utc_run_name()
    run_path = f"detect_runs/{resolved_run_name}"

    if not apply:
        return AcquisitionDetectImportResult(
            zarr_path=str(zarr_path),
            recording_dir=str(resolved_recording_dir),
            crop_meta_path=str(resolved_crop_meta),
            run_name=resolved_run_name,
            run_path=run_path,
            total_frames=int(total_frames),
            total_detections=int(frame_indices.shape[0]),
            frames_with_detections=int(np.count_nonzero(frame_counts)),
            blank_frame_count=int(np.count_nonzero(crop_rows.blank_frame)),
            no_detection_frame_count=int(crop_rows.frame_indices.shape[0] - np.count_nonzero(crop_rows.has_detection)),
            source_width=int(width),
            source_height=int(height),
            applied=False,
        )

    parent = require_runs_parent(root, "detect_runs")
    if resolved_run_name in parent:
        if not overwrite:
            raise ValueError(f"Detect run already exists: {run_path}")
        del parent[resolved_run_name]
    run = parent.create_group(resolved_run_name)
    mark_run_started(run, run_name=resolved_run_name, stage="detect")

    try:
        _create_array(run, "frame_indices", frame_indices)
        _create_array(run, "bbox_norm_coords", bbox_norm)
        _create_array(run, "bbox_img_xyxy", bbox_img_xyxy)
        _create_array(run, "scores", scores)
        _create_array(run, "class_ids", class_ids)
        _create_array(run, "frame_counts", frame_counts)
        _create_array(run, "n_detections", frame_counts)
        _create_array(run, "source_crop_xywh", source_crop_xywh)
        _create_array(run, "source_crop_meta_row_indices", source_crop_meta_row_indices)
        _create_array(run, "source_recording_frame_ids", source_recording_frame_ids)

        frames_with_detections = int(np.count_nonzero(frame_counts))
        blank_count = int(np.count_nonzero(crop_rows.blank_frame))
        no_detection_count = int(crop_rows.frame_indices.shape[0] - np.count_nonzero(crop_rows.has_detection))
        now = datetime.now(timezone.utc).isoformat()
        manifest_attrs = _manifest_crop_attrs(resolved_recording_dir)
        stats = {
            "total_detections": int(frame_indices.shape[0]),
            "frames_with_detections": frames_with_detections,
            "percent_frames_with_detections": float(frames_with_detections / total_frames * 100.0)
            if total_frames
            else 0.0,
            "frames_with_zero_detections": int(total_frames - frames_with_detections),
            "frames_with_multiple_detections": int(np.count_nonzero(frame_counts > 1)),
            "mean_detections_per_frame": float(frame_indices.shape[0] / total_frames) if total_frames else 0.0,
            "mean_confidence": float(np.nanmean(scores)) if scores.size else 0.0,
            "min_confidence": float(np.nanmin(scores)) if scores.size else 0.0,
            "max_confidence": float(np.nanmax(scores)) if scores.size else 0.0,
            "crop_meta_row_count": int(crop_rows.frame_indices.shape[0]),
            "crop_meta_blank_frame_count": blank_count,
            "crop_meta_no_detection_frame_count": no_detection_count,
        }
        run.attrs.update(
            {
                "schema_id": SCHEMA_ID,
                "detect_timestamp_utc": now,
                "detection_method": "acquisition_runtime_import",
                "detection_source": "external_crop_recorder_crop_meta",
                "model_type": "runtime_acquisition_detection",
                "source_video_width": int(width),
                "source_video_height": int(height),
                "source_full_width": int(width),
                "source_full_height": int(height),
                "bbox_img_xyxy_coordinate_space": "source_image_xyxy",
                "bbox_img_xyxy_reference_width": int(width),
                "bbox_img_xyxy_reference_height": int(height),
                "source_artifact_path": str(resolved_crop_meta),
                "source_recording_dir": str(resolved_recording_dir),
                "source_frame_clock": "recording_frame_id",
                "source_frame_index_transform": "palette_frame_index = recording_frame_id - 1",
                "source_crop_xywh_coordinate_space": "source_image_xywh",
                "parameters": {
                    "class_id": int(class_id),
                    "selection_policy": manifest_attrs.get("selection_policy"),
                    "blank_frame_policy": manifest_attrs.get("blank_frame_policy"),
                    "source_width": int(width),
                    "source_height": int(height),
                },
                "summary_statistics": stats,
                "acquisition_crop_stream": manifest_attrs,
            }
        )
        git_info = get_git_info(Path(__file__).resolve().parents[3])
        env_info = get_environment_info(include_all_packages=False, disk_path=str(zarr_path), collect_ip=False)
        provenance = build_stage_provenance(
            stage="detect",
            command=" ".join(sys.argv),
            created_at_utc=now,
            version=git_info.get("short_hash") or git_info.get("commit_hash"),
            git=git_info,
            environment=env_info.get("environment"),
            platform=env_info.get("platform"),
            parameters=dict(run.attrs.get("parameters") or {}),
            inputs={
                "source": "external_crop_recorder_crop_meta",
                "crop_meta_path": str(resolved_crop_meta),
                "recording_dir": str(resolved_recording_dir),
            },
            artifacts={
                "run_path": run_path,
                "source_crop_xywh": "source_crop_xywh",
                "source_crop_meta_row_indices": "source_crop_meta_row_indices",
            },
        )
        write_stage_provenance(run, provenance)
        mark_run_complete(run, parent_group=parent, run_name=resolved_run_name)
    except Exception as exc:
        mark_run_failed(run, error=str(exc))
        raise

    return AcquisitionDetectImportResult(
        zarr_path=str(zarr_path),
        recording_dir=str(resolved_recording_dir),
        crop_meta_path=str(resolved_crop_meta),
        run_name=resolved_run_name,
        run_path=run_path,
        total_frames=int(total_frames),
        total_detections=int(frame_indices.shape[0]),
        frames_with_detections=int(np.count_nonzero(frame_counts)),
        blank_frame_count=int(np.count_nonzero(crop_rows.blank_frame)),
        no_detection_frame_count=int(crop_rows.frame_indices.shape[0] - np.count_nonzero(crop_rows.has_detection)),
        source_width=int(width),
        source_height=int(height),
        applied=True,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Analysis zarr archive.")
    parser.add_argument("--recording-dir", type=Path, help="Recording root used to resolve recording_manifest.json.")
    parser.add_argument("--crop-meta", type=Path, help="Explicit external crop-recorder *_crop_meta.csv path.")
    parser.add_argument("--run-name", help="Detect run name (default: timestamped acquisition import run).")
    parser.add_argument("--source-width", type=int, help="Full-frame source width for normalized bbox conversion.")
    parser.add_argument("--source-height", type=int, help="Full-frame source height for normalized bbox conversion.")
    parser.add_argument("--class-id", type=int, default=0, help="Class id assigned to imported acquisition boxes.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing detect run with the same name.")
    parser.add_argument("--apply", action="store_true", help="Write detect_runs/<run>; otherwise only print the plan.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    result = import_acquisition_detections_to_detect_run(
        args.zarr_path,
        recording_dir=args.recording_dir,
        crop_meta_path=args.crop_meta,
        run_name=args.run_name,
        source_width=args.source_width,
        source_height=args.source_height,
        class_id=int(args.class_id),
        overwrite=bool(args.overwrite),
        apply=bool(args.apply),
    )
    payload = asdict(result)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"run_path: {result.run_path}")
        print(f"crop_meta: {result.crop_meta_path}")
        print(f"source_shape: {result.source_width}x{result.source_height}")
        print(
            "detections: "
            f"{result.total_detections} over {result.frames_with_detections}/{result.total_frames} frames"
        )
        print(
            "acquisition_missing: "
            f"blank={result.blank_frame_count} no_detection={result.no_detection_frame_count}"
        )
        if not result.applied:
            print("dry_run: pass --apply to write detect_runs/<run>")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
