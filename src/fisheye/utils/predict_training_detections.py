#!/usr/bin/env python3
"""Seed detection predictions on a Palette training Zarr.

This utility is intentionally narrower than the production video inference
path. It reads sampled frames already stored in ``raw_video`` and writes a
normal ``detect_runs/<run>`` group that downstream review/training tools can use
as initial labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import zarr

from fisheye.registry.db import Registry, RegistryPaths
from fisheye.shared.stage_provenance import build_stage_provenance, write_stage_provenance
from fisheye.shared.zarr_run_completion import mark_run_complete, mark_run_started, require_runs_parent
from fisheye.utils.system import get_environment_info, get_git_info


@dataclass(frozen=True)
class ModelInputSpec:
    artifact_kind: str
    run_id: str
    set_id: Optional[str]
    task_type: str
    artifact_path: str
    input_shape: Optional[list[int]]
    input_layout: Optional[str]
    input_channels: Optional[int]
    img_h: Optional[int]
    img_w: Optional[int]
    max_batch: Optional[int]
    dynamic_shapes: Optional[bool]
    input_dtype: Optional[str]
    input_color_space: Optional[str]
    input_shape_source: Optional[str]
    input_shape_status: Optional[str]
    artifact_precision: Optional[str] = None


@dataclass(frozen=True)
class FrameSourceSelection:
    path: str
    shape: tuple[int, ...]
    n_frames: int
    height: int
    width: int
    channels: int
    matches_model_shape: bool
    needs_gray_to_rgb: bool
    reason: str


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _open_zarr_group(path: Path, *, mode: str) -> Any:
    try:
        return zarr.open_group(str(path), mode=mode, use_consolidated=False)
    except TypeError:  # pragma: no cover - compatibility with older zarr API
        return zarr.open_group(str(path), mode=mode)


def _parse_shape(raw: Any) -> Optional[list[int]]:
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        values = raw
    else:
        try:
            values = json.loads(str(raw))
        except Exception:
            return None
    if not isinstance(values, list):
        return None
    parsed: list[int] = []
    for item in values:
        try:
            parsed.append(int(item))
        except (TypeError, ValueError):
            return None
    return parsed


def _int_or_none(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bool_or_none(value: Any) -> Optional[bool]:
    if value is None or value == "":
        return None
    return bool(int(value)) if isinstance(value, (int, np.integer)) else bool(value)


def _row_to_spec(row: Any) -> ModelInputSpec:
    return ModelInputSpec(
        artifact_kind=str(row["artifact_kind"]),
        run_id=str(row["run_id"]),
        set_id=str(row["set_id"]) if row["set_id"] is not None else None,
        task_type=str(row["task_type"]),
        artifact_path=str(row["artifact_path"]),
        artifact_precision=str(row["artifact_precision"]) if row["artifact_precision"] is not None else None,
        input_shape=_parse_shape(row["input_shape"]),
        input_layout=str(row["input_layout"]) if row["input_layout"] is not None else None,
        input_channels=_int_or_none(row["input_channels"]),
        img_h=_int_or_none(row["img_h"]),
        img_w=_int_or_none(row["img_w"]),
        max_batch=_int_or_none(row["max_batch"]),
        dynamic_shapes=_bool_or_none(row["dynamic_shapes"]),
        input_dtype=str(row["input_dtype"]) if row["input_dtype"] is not None else None,
        input_color_space=str(row["input_color_space"]) if row["input_color_space"] is not None else None,
        input_shape_source=str(row["input_shape_source"]) if row["input_shape_source"] is not None else None,
        input_shape_status=str(row["input_shape_status"]) if row["input_shape_status"] is not None else None,
    )


def resolve_model_input_spec(
    registry_path: Path,
    *,
    model_run_id: Optional[str],
    model_path: Optional[Path],
    set_id: Optional[str],
    artifact_kind: str,
) -> ModelInputSpec:
    """Resolve the detection model artifact and normalized input shape."""

    registry = Registry(registry_path)
    try:
        where = [
            "task_type = 'detect'",
            "artifact_kind = ?",
            "artifact_path IS NOT NULL",
        ]
        params: list[Any] = [artifact_kind]
        if model_run_id:
            where.append("run_id = ?")
            params.append(str(model_run_id))
        if set_id:
            where.append("set_id = ?")
            params.append(str(set_id))
        if model_path is not None:
            path_text = str(model_path.expanduser())
            resolved_text = str(model_path.expanduser().resolve()) if model_path.expanduser().exists() else path_text
            where.append("(artifact_path = ? OR artifact_path = ?)")
            params.extend([path_text, resolved_text])

        sql = f"""
            SELECT
                artifact_kind,
                run_id,
                set_id,
                task_type,
                artifact_path,
                artifact_sha256,
                artifact_precision,
                input_shape,
                input_layout,
                input_channels,
                img_h,
                img_w,
                max_batch,
                dynamic_shapes,
                input_dtype,
                input_color_space,
                input_shape_source,
                input_shape_status,
                created_utc
            FROM model_input_shapes
            WHERE {" AND ".join(where)}
            ORDER BY
                CASE input_shape_status
                    WHEN 'explicit' THEN 0
                    WHEN 'inferred_from_imgsz' THEN 1
                    WHEN 'export_backfill' THEN 2
                    WHEN 'conflict' THEN 3
                    ELSE 4
                END,
                COALESCE(created_utc, '') DESC,
                run_id DESC
            LIMIT 1;
        """
        row = registry.conn.execute(sql, params).fetchone()
    finally:
        registry.close()

    if row is None:
        raise ValueError(
            "No detection model input shape row matched the requested filters. "
            "Pass --model-run-id, --model-path, or --set-id for a registered detect model."
        )
    spec = _row_to_spec(row)
    if spec.img_h is None or spec.img_w is None:
        raise ValueError(
            f"Resolved model row lacks usable img_h/img_w: run_id={spec.run_id}, "
            f"status={spec.input_shape_status}."
        )
    artifact_path = Path(spec.artifact_path).expanduser()
    if not artifact_path.exists():
        raise FileNotFoundError(f"Resolved model artifact does not exist: {artifact_path}")
    return spec


def _frame_array_info(raw_video: Any, key: str) -> Optional[tuple[tuple[int, ...], int, int, int, int]]:
    if key not in raw_video:
        return None
    shape = tuple(int(dim) for dim in raw_video[key].shape)
    if len(shape) == 3:
        n, h, w = shape
        channels = 1
    elif len(shape) == 4:
        n, h, w, channels = shape
    else:
        return None
    return shape, n, h, w, channels


def select_frame_source(root: Any, spec: ModelInputSpec) -> FrameSourceSelection:
    raw_video = root.get("raw_video")
    if raw_video is None:
        raise ValueError("Training Zarr is missing raw_video group.")
    if spec.img_h is None or spec.img_w is None:
        raise ValueError("Model input spec must include img_h/img_w.")

    candidates = [
        ("images_ds_rgb", "downsampled RGB frames"),
        ("images_ds", "downsampled grayscale frames"),
        ("images_full", "full-resolution grayscale frames"),
    ]
    infos: dict[str, tuple[tuple[int, ...], int, int, int, int]] = {}
    for key, _label in candidates:
        info = _frame_array_info(raw_video, key)
        if info is not None:
            infos[key] = info

    for key in ("images_ds_rgb", "images_ds"):
        info = infos.get(key)
        if info is None:
            continue
        shape, n, h, w, channels = info
        if h == int(spec.img_h) and w == int(spec.img_w):
            return FrameSourceSelection(
                path=f"raw_video/{key}",
                shape=shape,
                n_frames=n,
                height=h,
                width=w,
                channels=channels,
                matches_model_shape=True,
                needs_gray_to_rgb=(channels == 1 and (spec.input_channels or 3) == 3),
                reason="sampled_array_matches_model_shape",
            )

    for key in ("images_full", "images_ds_rgb", "images_ds"):
        info = infos.get(key)
        if info is None:
            continue
        shape, n, h, w, channels = info
        return FrameSourceSelection(
            path=f"raw_video/{key}",
            shape=shape,
            n_frames=n,
            height=h,
            width=w,
            channels=channels,
            matches_model_shape=(h == int(spec.img_h) and w == int(spec.img_w)),
            needs_gray_to_rgb=(channels == 1 and (spec.input_channels or 3) == 3),
            reason="fallback_to_available_frame_array",
        )

    raise ValueError("Training Zarr raw_video group has no supported frame arrays.")


def _to_model_images(frames: np.ndarray, *, needs_gray_to_rgb: bool) -> list[np.ndarray]:
    arr = np.asarray(frames)
    if arr.ndim == 3 and needs_gray_to_rgb:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    elif arr.ndim == 4 and arr.shape[-1] == 1 and needs_gray_to_rgb:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.ndim == 3:
        arr = arr[..., None]
    return [np.asarray(frame) for frame in arr]


def _extract_result_arrays(
    results: Sequence[Any],
    *,
    frame_indices: Sequence[int],
    frame_height: int,
    frame_width: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    output: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for result, frame_idx in zip(results, frame_indices):
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.detach().cpu().numpy()
        if xyxy.size == 0:
            continue
        scores = boxes.conf.detach().cpu().numpy().astype(np.float32, copy=False)
        cls_tensor = getattr(boxes, "cls", None)
        if cls_tensor is None:
            class_ids = np.zeros((xyxy.shape[0],), dtype=np.int32)
        else:
            class_ids = cls_tensor.detach().cpu().numpy().astype(np.int32, copy=False)
        cx = (xyxy[:, 0] + xyxy[:, 2]) * 0.5 / float(frame_width)
        cy = (xyxy[:, 1] + xyxy[:, 3]) * 0.5 / float(frame_height)
        ww = (xyxy[:, 2] - xyxy[:, 0]) / float(frame_width)
        hh = (xyxy[:, 3] - xyxy[:, 1]) / float(frame_height)
        bbox_norm = np.column_stack((cx, cy, ww, hh)).astype(np.float64, copy=False)
        indices = np.full((xyxy.shape[0],), int(frame_idx), dtype=np.int32)
        output.append((indices, bbox_norm, scores, class_ids))
    return output


def run_training_zarr_prediction(
    *,
    zarr_path: Path,
    spec: ModelInputSpec,
    run_name: str,
    batch_size: int,
    conf: float,
    iou: float,
    max_det: int,
    cpu: bool,
    overwrite: bool,
    argv: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Run YOLO predictions from raw_video arrays and persist a detect run."""

    from ultralytics import YOLO

    root = _open_zarr_group(zarr_path, mode="r+")
    selection = select_frame_source(root, spec)
    raw_video = root["raw_video"]
    frame_array = raw_video[selection.path.split("/", 1)[1]]

    detect_parent = require_runs_parent(root, "detect_runs")
    if run_name in detect_parent and not overwrite:
        raise ValueError(f"detect run already exists: detect_runs/{run_name}; pass --overwrite to replace it.")
    if run_name in detect_parent and overwrite:
        del detect_parent[run_name]
    detect_group = detect_parent.require_group(run_name)
    mark_run_started(detect_group, run_name=run_name, stage="detect")

    model = YOLO(str(Path(spec.artifact_path).expanduser()))
    if cpu:
        model.to("cpu")

    predict_kwargs = {
        "conf": float(conf),
        "iou": float(iou),
        "max_det": int(max_det),
        "verbose": False,
        "device": "cpu" if cpu else None,
        "imgsz": [int(spec.img_h), int(spec.img_w)],
    }
    predict_kwargs = {key: value for key, value in predict_kwargs.items() if value is not None}

    detection_chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    n_frames = int(selection.n_frames)
    for start in range(0, n_frames, int(batch_size)):
        end = min(start + int(batch_size), n_frames)
        frames = np.asarray(frame_array[start:end])
        images = _to_model_images(frames, needs_gray_to_rgb=selection.needs_gray_to_rgb)
        results = model.predict(images, **predict_kwargs)
        detection_chunks.extend(
            _extract_result_arrays(
                results,
                frame_indices=list(range(start, end)),
                frame_height=selection.height,
                frame_width=selection.width,
            )
        )

    if detection_chunks:
        frame_indices = np.concatenate([chunk[0] for chunk in detection_chunks])
        bbox_norm = np.concatenate([chunk[1] for chunk in detection_chunks])
        scores = np.concatenate([chunk[2] for chunk in detection_chunks])
        class_ids = np.concatenate([chunk[3] for chunk in detection_chunks]).astype(np.int32, copy=False)
    else:
        frame_indices = np.empty((0,), dtype=np.int32)
        bbox_norm = np.empty((0, 4), dtype=np.float64)
        scores = np.empty((0,), dtype=np.float32)
        class_ids = np.empty((0,), dtype=np.int32)

    frame_counts = np.bincount(frame_indices, minlength=n_frames).astype(np.int32, copy=False)
    det_chunk = max(1, min(max(1, int(frame_indices.size)), 16384))
    count_chunk = max(1, min(n_frames, 16384))
    detect_group.create_array("frame_indices", data=frame_indices, chunks=(det_chunk,), overwrite=True)
    detect_group.create_array("bbox_norm_coords", data=bbox_norm, chunks=(det_chunk, 4), overwrite=True)
    detect_group.create_array("scores", data=scores, chunks=(det_chunk,), overwrite=True)
    detect_group.create_array("class_ids", data=class_ids, chunks=(det_chunk,), overwrite=True)
    detect_group.create_array("n_detections", data=frame_counts, chunks=(count_chunk,), overwrite=True)
    detect_group.create_array("frame_counts", data=frame_counts, chunks=(count_chunk,), overwrite=True)

    original_frame_indices = raw_video.get("original_frame_indices")
    if original_frame_indices is not None and frame_indices.size:
        source_frame_indices = np.asarray(original_frame_indices[:], dtype=np.int64)[frame_indices]
        detect_group.create_array(
            "source_frame_indices",
            data=source_frame_indices,
            chunks=(det_chunk,),
            overwrite=True,
        )

    frames_with_detections = int(np.sum(frame_counts > 0))
    total_detections = int(frame_indices.size)
    summary_statistics = {
        "total_detections": total_detections,
        "frames_with_detections": frames_with_detections,
        "percent_frames_with_detections": float(frames_with_detections / n_frames * 100.0) if n_frames else 0.0,
        "frames_with_zero_detections": int(np.sum(frame_counts == 0)),
        "frames_with_multiple_detections": int(np.sum(frame_counts > 1)),
        "mean_detections_per_frame": float(total_detections / n_frames) if n_frames else 0.0,
        "mean_confidence": float(np.mean(scores)) if scores.size else 0.0,
        "min_confidence": float(np.min(scores)) if scores.size else 0.0,
        "max_confidence": float(np.max(scores)) if scores.size else 0.0,
    }

    now = datetime.now(timezone.utc).isoformat()
    git_info = get_git_info()
    env_info = get_environment_info(include_all_packages=False, disk_path=str(zarr_path), collect_ip=False)
    parameters = {
        "conf_threshold": float(conf),
        "iou_threshold": float(iou),
        "max_det": int(max_det),
        "batch_size": int(batch_size),
        "imgsz": [int(spec.img_h), int(spec.img_w)],
        "cpu": bool(cpu),
    }
    frame_source_payload = asdict(selection)
    model_payload = asdict(spec)
    detect_group.attrs.update(
        {
            "detect_timestamp_utc": now,
            "detection_method": "yolo",
            "detection_source": "training_zarr_raw_video",
            "model_type": "yolo_object_detection",
            "model_path": str(Path(spec.artifact_path).expanduser()),
            "model_name": Path(spec.artifact_path).name,
            "model_registry_run_id": spec.run_id,
            "model_registry_set_id": spec.set_id,
            "model_registry_artifact_kind": spec.artifact_kind,
            "model_input_shape": spec.input_shape,
            "model_input_shape_status": spec.input_shape_status,
            "model_input_shape_source": spec.input_shape_source,
            "frame_source_path": selection.path,
            "frame_source_shape": list(selection.shape),
            "frame_source_matches_model_shape": bool(selection.matches_model_shape),
            "source_frame_index_space": "training_zarr_row_index",
            "source_original_frame_indices_available": original_frame_indices is not None,
            "parameters": parameters,
            "summary_statistics": summary_statistics,
            "git_commit": git_info.get("commit_hash", "unknown"),
            "git_branch": git_info.get("branch", "unknown"),
            "hostname": env_info["platform"].get("hostname"),
        }
    )

    provenance = build_stage_provenance(
        stage="detect",
        command=" ".join(str(item) for item in (argv or sys.argv)),
        created_at_utc=now,
        version=git_info.get("short_hash") or git_info.get("commit_hash"),
        git={
            "commit": git_info.get("commit_hash"),
            "short": git_info.get("short_hash"),
            "branch": git_info.get("branch"),
            "is_dirty": git_info.get("is_dirty"),
            "remote": git_info.get("remote_url"),
        },
        environment=env_info.get("environment"),
        platform={
            "hostname": env_info["platform"].get("hostname"),
            "system": env_info["platform"].get("system"),
            "release": env_info["platform"].get("release"),
            "python_version": env_info["platform"].get("python_version"),
            "machine": env_info["platform"].get("machine"),
        },
        parameters=parameters,
        inputs={
            "zarr_path": str(zarr_path),
            "frame_source": frame_source_payload,
            "model": model_payload,
        },
        artifacts={
            "detect_run": f"detect_runs/{run_name}",
            "frame_indices": f"detect_runs/{run_name}/frame_indices",
            "bbox_norm_coords": f"detect_runs/{run_name}/bbox_norm_coords",
        },
    )
    write_stage_provenance(detect_group, provenance)

    mark_run_complete(detect_group, parent_group=detect_parent, run_name=run_name)
    return {
        "ok": True,
        "zarr_path": str(zarr_path),
        "detect_run": run_name,
        "frame_source": frame_source_payload,
        "model": model_payload,
        "summary_statistics": summary_statistics,
    }


def _default_run_name() -> str:
    return "detect_training_seed_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Training Zarr to seed with detect predictions.")
    parser.add_argument("--registry", type=Path, help="Palette registry SQLite path.")
    parser.add_argument("--model-run-id", help="Training/export run_id to use.")
    parser.add_argument("--model-path", type=Path, help="Registered model artifact path to use.")
    parser.add_argument("--set-id", help="Training set id filter when selecting a model.")
    parser.add_argument(
        "--artifact-kind",
        choices=("training", "onnx", "tensorrt"),
        default="training",
        help="Model artifact kind to resolve from model_input_shapes.",
    )
    parser.add_argument("--run-name", default=None, help="Output detect run name.")
    parser.add_argument("--conf", type=float, default=0.40, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold.")
    parser.add_argument("--max-det", type=int, default=20, help="Max detections per frame.")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing detect run of the same name.")
    parser.add_argument("--apply", action="store_true", help="Write predictions. Without this, only print the plan.")
    parser.add_argument("--json", action="store_true", help="Emit JSON summary.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    zarr_path = args.zarr_path.expanduser().resolve()
    if not zarr_path.exists():
        raise SystemExit(f"Training Zarr does not exist: {zarr_path}")
    registry_path = (args.registry or RegistryPaths.from_env(Path.cwd()).path).expanduser().resolve()
    spec = resolve_model_input_spec(
        registry_path,
        model_run_id=args.model_run_id,
        model_path=args.model_path,
        set_id=args.set_id,
        artifact_kind=args.artifact_kind,
    )
    root = _open_zarr_group(zarr_path, mode="r")
    selection = select_frame_source(root, spec)
    run_name = args.run_name or _default_run_name()
    plan = {
        "ok": True,
        "mode": "apply" if args.apply else "dry_run",
        "zarr_path": str(zarr_path),
        "registry_path": str(registry_path),
        "run_name": run_name,
        "model": asdict(spec),
        "frame_source": asdict(selection),
        "parameters": {
            "conf": float(args.conf),
            "iou": float(args.iou),
            "max_det": int(args.max_det),
            "batch_size": int(args.batch_size),
            "cpu": bool(args.cpu),
        },
    }
    if not args.apply:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    result = run_training_zarr_prediction(
        zarr_path=zarr_path,
        spec=spec,
        run_name=run_name,
        batch_size=int(args.batch_size),
        conf=float(args.conf),
        iou=float(args.iou),
        max_det=int(args.max_det),
        cpu=bool(args.cpu),
        overwrite=bool(args.overwrite),
        argv=list(argv) if argv is not None else sys.argv,
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        stats = result["summary_statistics"]
        print(
            f"Wrote detect_runs/{run_name}: {stats['total_detections']} detections, "
            f"{stats['percent_frames_with_detections']:.1f}% frame coverage "
            f"from {selection.path}."
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
