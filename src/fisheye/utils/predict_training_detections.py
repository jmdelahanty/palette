#!/usr/bin/env python3
"""Persist model predictions from a Palette training Zarr as an unbound artifact.

This utility is intentionally narrower than the production video inference
path. It reads sampled frames already stored in ``raw_video`` and writes an
immutable, selector-free ``detection_artifact_runs/<run>``. Training/model-frame
geometry is useful as numeric evidence, but it is not a canonical source-camera
detection surface and cannot be selected as an ordinary ``detect_runs`` run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import zarr

from fisheye.registry.db import Registry, RegistryPaths
from fisheye.shared.detection_producer_lifecycle import (
    DETECTION_ARTIFACT_RUN_FAMILY,
    DetectionProducerAttempt,
    UNBOUND_ARTIFACT_RUN_BINDING_KEY,
    build_unbound_artifact_run_binding,
    publish_artifact_payload_inventory_seal,
    publish_empty_artifact_observation_proof,
    stamp_unbound_artifact_numeric_semantics,
)
from fisheye.shared.run_provenance import build_run_provenance_from_stage_record
from fisheye.shared.stage_provenance import build_stage_provenance, write_stage_provenance
from fisheye.shared.system_metadata import get_environment_info, get_git_info

ARTIFACT_FRAME_SOURCE_LINEAGE_ATTR = "artifact_frame_source_lineage"
ARTIFACT_FRAME_SOURCE_LINEAGE_SCHEMA = (
    "palette.training_detection_artifact_frame_source_lineage.v1"
)


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


def _canonical_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(_json_dumps(value).encode("utf-8")).hexdigest()


def _array_payload_sha256(values: np.ndarray) -> str:
    array = np.asarray(values)
    if array.dtype.hasobject:
        raise ValueError("Artifact lineage arrays cannot use object dtype.")
    digest = hashlib.sha256()
    digest.update(b"palette.ndarray_payload.v1\x00")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\x00")
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def _normalize_run_name(run_name: str) -> str:
    if type(run_name) is not str:
        raise ValueError("run_name must be an exact string.")
    normalized = run_name.strip()
    if not normalized or "/" in normalized or normalized in {".", ".."}:
        raise ValueError("run_name must normalize to one nonempty path segment.")
    return normalized


def _open_zarr_group(path: Path, *, mode: str) -> Any:
    try:
        return zarr.open_group(str(path), mode=mode, use_consolidated=False)
    except TypeError:  # pragma: no cover - compatibility with older zarr API
        return zarr.open_group(str(path), mode=mode)


def _close_store(node: Any) -> None:
    close = getattr(getattr(node, "store", None), "close", None)
    if callable(close):
        close()


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


def _load_original_frame_mapping(
    raw_video: Any,
    *,
    n_frames: int,
) -> tuple[Optional[np.ndarray], dict[str, Any]]:
    node = raw_video.get("original_frame_indices")
    if node is None:
        return None, {
            "status": "absent",
            "meaning": "training_frame_rows_have_no_persisted_source_frame_mapping",
        }
    raw = np.asarray(node[:])
    if raw.ndim != 1 or raw.shape != (n_frames,):
        raise ValueError(
            "raw_video/original_frame_indices must have one value for every "
            "selected training frame row."
        )
    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("raw_video/original_frame_indices must use integer dtype.")
    if np.any(raw < 0):
        raise ValueError("raw_video/original_frame_indices cannot contain negatives.")
    if raw.size > 1 and np.any(np.diff(raw.astype(object)) <= 0):
        raise ValueError(
            "raw_video/original_frame_indices must be strictly increasing."
        )
    if raw.size and int(np.max(raw)) > np.iinfo(np.int64).max:
        raise ValueError(
            "raw_video/original_frame_indices cannot be represented as int64."
        )
    normalized = raw.astype(np.int64, copy=False)
    return normalized, {
        "status": "present_unbound_source_evidence",
        "array_path": "raw_video/original_frame_indices",
        "shape": list(raw.shape),
        "source_dtype": raw.dtype.str,
        "source_payload_sha256": _array_payload_sha256(raw),
        "persisted_artifact_dtype": normalized.dtype.str,
        "persisted_artifact_payload_sha256": _array_payload_sha256(normalized),
        "direction": "training_frame_row_to_recording_source_frame_index",
        "canonical_identity_status": "not_claimed",
    }


def _build_frame_source_lineage(
    *,
    selection: FrameSourceSelection,
    frame_array: Any,
    spec: ModelInputSpec,
    original_frame_mapping: dict[str, Any],
) -> dict[str, Any]:
    chunks = getattr(frame_array, "chunks", None)
    return {
        "schema_id": ARTIFACT_FRAME_SOURCE_LINEAGE_SCHEMA,
        "status": "unbound_artifact_provenance_only",
        "selected_array_path": selection.path,
        "selected_array_shape": list(selection.shape),
        "selected_array_dtype": np.dtype(frame_array.dtype).str,
        "selected_array_chunks": (
            [int(value) for value in chunks] if chunks is not None else None
        ),
        "frame_row_count": int(selection.n_frames),
        "frame_source_extent": {
            "width": int(selection.width),
            "height": int(selection.height),
            "channels": int(selection.channels),
            "units": "pixels",
            "extent_basis": "selected_training_frame_array_shape",
        },
        "selection": asdict(selection),
        "model_request": {
            "registry_run_id": spec.run_id,
            "registry_set_id": spec.set_id,
            "artifact_kind": spec.artifact_kind,
            "requested_imgsz": [int(spec.img_h), int(spec.img_w)],
            "selected_frame_matches_requested_imgsz": bool(
                selection.matches_model_shape
            ),
            "preprocessing_implementation": "ultralytics.YOLO.predict",
            "exact_preprocessing_transform_persisted": False,
        },
        "prediction_result_basis": (
            "ultralytics_xyxy_rescaled_to_selected_training_frame_array_extent"
        ),
        "pixel_content_binding_status": "not_content_hashed_unbound_artifact",
        "original_frame_mapping": original_frame_mapping,
        "source_camera_overlay_suitability": (
            "unsupported_without_canonical_frame_and_crop_lineage"
        ),
    }


def _stamp_prediction_array_semantics(
    run: Any,
    *,
    selection: FrameSourceSelection,
    frame_source_lineage_sha256: str,
    original_frame_mapping: dict[str, Any],
) -> None:
    profiles = {
        "artifact_row_id": "training.artifact_row_id.v1",
        "frame_indices": "training.frame_indices.v1",
        "bbox_norm_coords": "training.bbox_norm_cxcywh.v1",
        "scores": "training.scores.v1",
        "class_ids": "training.class_ids.v1",
        "frame_counts": "training.frame_counts.v1",
        "n_detections": "training.n_detections.v1",
    }
    if "source_frame_indices" in run:
        profiles["source_frame_indices"] = "training.source_frame_indices.v1"
    if set(profiles) != set(run.keys()):
        raise ValueError(
            "Training detection semantic inventory does not match live arrays."
        )
    source_mapping_sha256 = (
        str(original_frame_mapping["source_payload_sha256"])
        if "source_frame_indices" in run
        else None
    )
    for name, profile_id in profiles.items():
        stamp_unbound_artifact_numeric_semantics(
            run[name],
            semantic_profile_id=profile_id,
            reference_node_path=selection.path,
            reference_width=int(selection.width),
            reference_height=int(selection.height),
            source_frame_count=int(selection.n_frames),
            source_sha256=frame_source_lineage_sha256,
            source_mapping_sha256=(
                source_mapping_sha256
                if name == "source_frame_indices"
                else None
            ),
        )


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
    """Run YOLO predictions and persist an immutable unbound artifact."""

    if overwrite:
        raise ValueError(
            "Training detection artifacts are immutable; --overwrite is unsupported. "
            "Choose a new --run-name."
        )
    normalized_run_name = _normalize_run_name(run_name)
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive exact integer.")

    root = _open_zarr_group(zarr_path, mode="r+")
    attempt: Optional[DetectionProducerAttempt] = None
    try:
        selection = select_frame_source(root, spec)
        n_frames = int(selection.n_frames)
        if n_frames <= 0:
            raise ValueError(
                "Training detection prediction requires at least one source frame."
            )
        if n_frames - 1 > np.iinfo(np.int32).max:
            raise ValueError(
                "Training frame rows cannot be represented by int32 frame_indices."
            )
        raw_video = root["raw_video"]
        frame_array = raw_video[selection.path.split("/", 1)[1]]
        existing_parent = root.get(DETECTION_ARTIFACT_RUN_FAMILY)
        if existing_parent is not None and normalized_run_name in existing_parent:
            raise ValueError(
                "Detection artifact already exists: "
                f"{DETECTION_ARTIFACT_RUN_FAMILY}/{normalized_run_name}; choose "
                "a new immutable run name."
            )

        original_frame_indices, original_frame_mapping = (
            _load_original_frame_mapping(raw_video, n_frames=n_frames)
        )

        from ultralytics import YOLO

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
        predict_kwargs = {
            key: value for key, value in predict_kwargs.items() if value is not None
        }

        detection_chunks: list[
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = []
        processed_frame_count = 0
        for start in range(0, n_frames, batch_size):
            end = min(start + batch_size, n_frames)
            frames = np.asarray(frame_array[start:end])
            images = _to_model_images(
                frames,
                needs_gray_to_rgb=selection.needs_gray_to_rgb,
            )
            results = model.predict(images, **predict_kwargs)
            if len(results) != len(images):
                raise RuntimeError(
                    "YOLO returned a result count that does not match the exact "
                    "training-frame batch; refusing a partial artifact."
                )
            detection_chunks.extend(
                _extract_result_arrays(
                    results,
                    frame_indices=list(range(start, end)),
                    frame_height=selection.height,
                    frame_width=selection.width,
                )
            )
            processed_frame_count += len(images)
        if processed_frame_count != n_frames:
            raise RuntimeError(
                "Training prediction did not validate the complete frame domain."
            )

        if detection_chunks:
            frame_indices = np.concatenate(
                [chunk[0] for chunk in detection_chunks]
            )
            bbox_norm = np.concatenate([chunk[1] for chunk in detection_chunks])
            scores = np.concatenate([chunk[2] for chunk in detection_chunks])
            class_ids = np.concatenate(
                [chunk[3] for chunk in detection_chunks]
            ).astype(np.int32, copy=False)
        else:
            frame_indices = np.empty((0,), dtype=np.int32)
            bbox_norm = np.empty((0, 4), dtype=np.float64)
            scores = np.empty((0,), dtype=np.float32)
            class_ids = np.empty((0,), dtype=np.int32)

        total_detections = int(frame_indices.size)
        artifact_row_id = np.arange(total_detections, dtype=np.uint64)
        frame_counts = np.bincount(frame_indices, minlength=n_frames).astype(
            np.int32,
            copy=False,
        )
        source_frame_indices = (
            original_frame_indices[frame_indices.astype(np.int64, copy=False)]
            if original_frame_indices is not None
            else None
        )
        semantic_manifest_id = (
            "training_detection_with_source_mapping.v1"
            if source_frame_indices is not None
            else "training_detection_without_source_mapping.v1"
        )
        frame_source_lineage = _build_frame_source_lineage(
            selection=selection,
            frame_array=frame_array,
            spec=spec,
            original_frame_mapping=original_frame_mapping,
        )
        frame_source_lineage[UNBOUND_ARTIFACT_RUN_BINDING_KEY] = (
            build_unbound_artifact_run_binding(
                manifest_id=semantic_manifest_id,
                reference_node_path=selection.path,
                reference_width=int(selection.width),
                reference_height=int(selection.height),
                source_frame_count=int(selection.n_frames),
                source_mapping_sha256=(
                    str(original_frame_mapping["source_payload_sha256"])
                    if source_frame_indices is not None
                    else None
                ),
            )
        )
        frame_source_lineage_sha256 = _canonical_sha256(frame_source_lineage)

        frames_with_detections = int(np.sum(frame_counts > 0))
        summary_statistics = {
            "total_detections": total_detections,
            "frames_with_detections": frames_with_detections,
            "percent_frames_with_detections": float(
                frames_with_detections / n_frames * 100.0
            ),
            "frames_with_zero_detections": int(np.sum(frame_counts == 0)),
            "frames_with_multiple_detections": int(np.sum(frame_counts > 1)),
            "mean_detections_per_frame": float(total_detections / n_frames),
            "mean_confidence": float(np.mean(scores)) if scores.size else 0.0,
            "min_confidence": float(np.min(scores)) if scores.size else 0.0,
            "max_confidence": float(np.max(scores)) if scores.size else 0.0,
        }

        now = datetime.now(timezone.utc).isoformat()
        git_info = get_git_info()
        env_info = get_environment_info(
            include_all_packages=False,
            disk_path=str(zarr_path),
            collect_ip=False,
        )
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

        attempt = DetectionProducerAttempt.begin_unbound_artifact(
            root,
            run_name=normalized_run_name,
            semantic_manifest_id=semantic_manifest_id,
            stage="training_detection_artifact",
            strict_integrity_required=True,
        )
        run = attempt.run
        normalized_run_name = attempt.run_name
        det_chunk = max(1, min(max(1, total_detections), 16384))
        count_chunk = max(1, min(n_frames, 16384))
        run.create_array(
            "artifact_row_id",
            data=artifact_row_id,
            chunks=(det_chunk,),
        )
        run.create_array(
            "frame_indices",
            data=frame_indices,
            chunks=(det_chunk,),
        )
        run.create_array(
            "bbox_norm_coords",
            data=bbox_norm,
            chunks=(det_chunk, 4),
        )
        run.create_array("scores", data=scores, chunks=(det_chunk,))
        run.create_array("class_ids", data=class_ids, chunks=(det_chunk,))
        run.create_array(
            "n_detections",
            data=frame_counts,
            chunks=(count_chunk,),
        )
        run.create_array(
            "frame_counts",
            data=frame_counts,
            chunks=(count_chunk,),
        )
        if source_frame_indices is not None:
            run.create_array(
                "source_frame_indices",
                data=source_frame_indices,
                chunks=(det_chunk,),
            )

        run.attrs.update(
            {
                "schema_id": "palette.training_detection_artifact.v1",
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
                "frame_source_matches_model_shape": bool(
                    selection.matches_model_shape
                ),
                "source_original_frame_indices_available": (
                    source_frame_indices is not None
                ),
                "artifact_row_identity": "dense_run_local_uint64_v1",
                ARTIFACT_FRAME_SOURCE_LINEAGE_ATTR: frame_source_lineage,
                f"{ARTIFACT_FRAME_SOURCE_LINEAGE_ATTR}_sha256": (
                    frame_source_lineage_sha256
                ),
                "parameters": parameters,
                "summary_statistics": summary_statistics,
                "git_commit": git_info.get("commit_hash", "unknown"),
                "git_branch": git_info.get("branch", "unknown"),
                "hostname": env_info["platform"].get("hostname"),
            }
        )

        row_array_names = [
            "artifact_row_id",
            "frame_indices",
            "bbox_norm_coords",
            "scores",
            "class_ids",
        ]
        if source_frame_indices is not None:
            row_array_names.append("source_frame_indices")
        _stamp_prediction_array_semantics(
            run,
            selection=selection,
            frame_source_lineage_sha256=frame_source_lineage_sha256,
            original_frame_mapping=original_frame_mapping,
        )
        if total_detections == 0:
            publish_empty_artifact_observation_proof(
                run,
                source_frame_count=n_frames,
                row_array_names=tuple(row_array_names),
                full_domain_evidence={
                    "coverage_status": "full_source_domain_validated",
                    "source_frame_count": n_frames,
                    "processed_frame_count": processed_frame_count,
                    "processed_frame_domain": {
                        "start": 0,
                        "stop_exclusive": n_frames,
                        "step": 1,
                    },
                    "inference_result_cardinality_validated": True,
                    "frame_source_lineage_sha256": frame_source_lineage_sha256,
                    "model_registry_run_id": spec.run_id,
                    "prediction_parameters": parameters,
                },
            )
        payload_inventory = publish_artifact_payload_inventory_seal(
            run,
            source_frame_count=n_frames,
        )
        payload_inventory_sha256 = run.attrs[
            "artifact_payload_inventory_seal_sha256"
        ]

        artifact_path = (
            f"{DETECTION_ARTIFACT_RUN_FAMILY}/{normalized_run_name}"
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
                "frame_source_lineage_sha256": frame_source_lineage_sha256,
                "model": model_payload,
            },
            artifacts={
                "detection_artifact_run": artifact_path,
                "artifact_row_id": f"{artifact_path}/artifact_row_id",
                "frame_indices": f"{artifact_path}/frame_indices",
                "bbox_norm_coords": f"{artifact_path}/bbox_norm_coords",
                "artifact_payload_inventory_seal_sha256": (
                    payload_inventory_sha256
                ),
                "artifact_payload_row_count": payload_inventory["row_count"],
            },
        )
        write_stage_provenance(run, provenance)
        attempt.complete(
            run_provenance=build_run_provenance_from_stage_record(provenance),
        )
    except BaseException as exc:
        try:
            if attempt is not None:
                attempt.fail(exc)
        finally:
            _close_store(root)
        raise

    _close_store(root)
    return {
        "ok": True,
        "zarr_path": str(zarr_path),
        "artifact_run": normalized_run_name,
        "artifact_path": (
            f"{DETECTION_ARTIFACT_RUN_FAMILY}/{normalized_run_name}"
        ),
        "output_parent": DETECTION_ARTIFACT_RUN_FAMILY,
        "frame_source": frame_source_payload,
        "model": model_payload,
        "summary_statistics": summary_statistics,
    }


def _default_run_name() -> str:
    return "detect_training_seed_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "zarr_path",
        type=Path,
        help="Training Zarr on which to compute an unbound detection artifact.",
    )
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
    parser.add_argument("--run-name", default=None, help="Output artifact run name.")
    parser.add_argument("--conf", type=float, default=0.40, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold.")
    parser.add_argument("--max-det", type=int, default=20, help="Max detections per frame.")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Unsupported for immutable artifacts; supplying it fails closed.",
    )
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
    try:
        selection = select_frame_source(root, spec)
    finally:
        _close_store(root)
    if args.overwrite:
        raise SystemExit(
            "Training detection artifacts are immutable; --overwrite is unsupported. "
            "Choose a new --run-name."
        )
    run_name = _normalize_run_name(args.run_name or _default_run_name())
    artifact_path = f"{DETECTION_ARTIFACT_RUN_FAMILY}/{run_name}"
    plan = {
        "ok": True,
        "mode": "apply" if args.apply else "dry_run",
        "zarr_path": str(zarr_path),
        "registry_path": str(registry_path),
        "run_name": run_name,
        "output_parent": DETECTION_ARTIFACT_RUN_FAMILY,
        "artifact_path": artifact_path,
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
            f"Wrote {artifact_path}: {stats['total_detections']} detections, "
            f"{stats['percent_frames_with_detections']:.1f}% frame coverage "
            f"from {selection.path}."
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
