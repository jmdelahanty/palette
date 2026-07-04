"""Create a sampled training Zarr from an Orange-style clipped recording."""

from __future__ import annotations

from fisheye.shared.json_safety import write_json_atomic as _write_json
from fisheye.shared.batch_logging import utc_now as _utc_now
import argparse
import json
import os
import shutil
import socket
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import zarr

from fisheye.utils.create_clipped_analysis_zarr import (
    DISH_MASK_POLICY_SCHEMA_VERSION,
    _dish_mask_policy_attrs,
)
from fisheye.utils.system import get_environment_summary, get_git_info, get_platform_info


MODULE_NAME = "fisheye.utils.create_clipped_training_zarr"
TRAINING_SCHEMA_VERSION = "palette.clipped_training_zarr.v1"
SOURCE_FRAME_INDEX_SCHEMA_VERSION = "palette.training_source_frame_index.v1"

RUN_GROUPS = (
    "background_runs",
    "detect_runs",
    "refined_detect_runs",
    "crop_runs",
    "keypoints_runs",
    "refined_keypoints_runs",
    "eye_masks_runs",
    "refined_eye_masks_runs",
    "subject_mask_runs",
    "refined_subject_masks_runs",
    "refined_online_runs",
    "tracking_runs",
    "arena_assignment_runs",
)

METADATA_ROOT_ATTR_KEYS = (
    "recording_type",
    "recording_subtype",
    "behavior_mode",
    "artifact_schema_id",
    "dish_design",
    "rig_id",
    "arena_id",
    "camera_id",
    "canvas_name",
    "protocol_name",
    "protocol_name_from_definition",
    "experiment_context_status",
    "experiment_context_source",
    "experiment_context_status_detail",
    "genotype",
    "dpf_at_acquisition",
)


@dataclass(frozen=True)
class ClippedTrainingOptions:
    frame_step: int = 5000
    camera_serial: str | None = None
    max_frames: int | None = None
    target_count: int | None = None
    write_images_ds: bool = True
    downsample_size: tuple[int, int] = (640, 640)
    chunk_size: int = 16
    overwrite: bool = False
    dry_run: bool = False
    write_manifest: bool = False
    require_dish_mask: bool = False
    copy_analysis_metadata_from: Path | None = None
    copy_existing_detections_from: Path | None = None


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _default_output_zarr(recording_dir: Path, camera_serial: str | None) -> Path:
    suffix = f"_cam{camera_serial}" if camera_serial else ""
    return recording_dir / "zarr" / f"{recording_dir.name}{suffix}_clipped_training.zarr"


def _resolve_recording_frame_index(recording_dir: Path) -> tuple[Path, Mapping[str, Any]]:
    manifest_path = recording_dir / "recording_frame_index_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"recording_frame_index_manifest.json not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError(f"recording_frame_index_manifest.json is not an object: {manifest_path}")
    if manifest.get("status") != "ok":
        raise ValueError(f"recording frame index manifest is not ok: {manifest_path}")
    raw_path = manifest.get("recording_frame_index_path") or "recording_frame_index.parquet"
    index_path = Path(str(raw_path)).expanduser()
    if not index_path.is_absolute():
        index_path = recording_dir / index_path
    index_path = index_path.resolve()
    if not index_path.exists():
        raise FileNotFoundError(f"recording_frame_index.parquet not found: {index_path}")
    return index_path, manifest


def _load_selected_source_rows(
    frame_index_path: Path,
    *,
    camera_serial: str | None,
    frame_step: int,
    max_frames: int | None,
    target_count: int | None,
) -> tuple[pa.Table, str]:
    if frame_step <= 0:
        raise ValueError("--frame-step must be a positive integer")
    table = pq.read_table(frame_index_path).combine_chunks()
    if table.num_rows == 0:
        raise ValueError(f"recording frame index has zero rows: {frame_index_path}")
    if "camera_serial" not in table.column_names:
        raise ValueError(f"recording frame index missing camera_serial column: {frame_index_path}")
    camera_values = sorted({str(value.as_py()) for value in table["camera_serial"]})
    effective_camera = camera_serial
    if effective_camera is None:
        if len(camera_values) != 1:
            raise ValueError(
                "Multiple cameras are present; pass --camera-serial. "
                f"camera_serials={camera_values}"
            )
        effective_camera = camera_values[0]
    table = table.filter(pc.equal(table["camera_serial"], str(effective_camera)))
    if table.num_rows == 0:
        raise ValueError(f"No rows found for camera_serial={effective_camera!r}")
    table = table.sort_by([("parent_frame_index", "ascending"), ("clip_index", "ascending")])
    parent = np.asarray(table["parent_frame_index"].to_numpy(zero_copy_only=False), dtype=np.int64)
    candidate_indices = np.flatnonzero((parent % int(frame_step)) == 0)
    if candidate_indices.size == 0:
        raise ValueError(f"No source rows selected by frame_step={frame_step}")
    if target_count is not None:
        if target_count <= 0:
            raise ValueError("--target-count must be positive when provided")
        if target_count < candidate_indices.size:
            positions = np.linspace(0, candidate_indices.size - 1, int(target_count), dtype=np.int64)
            candidate_indices = candidate_indices[positions]
    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError("--max-frames must be positive when provided")
        candidate_indices = candidate_indices[: int(max_frames)]
    selected = table.take(pa.array(candidate_indices, type=pa.int64()))
    sample_index = pa.array(np.arange(selected.num_rows, dtype=np.int64), type=pa.int64())
    selected = selected.add_column(0, "sample_index", sample_index)
    return selected, str(effective_camera)


def _append_source_training_columns(
    table: pa.Table,
    *,
    frame_index_path: Path,
    frame_step: int,
    sample_plan_id: str,
) -> pa.Table:
    row_count = table.num_rows
    return (
        table.append_column(
            "source_recording_frame_index_path",
            pa.array([str(frame_index_path)] * row_count, type=pa.string()),
        )
        .append_column("frame_step", pa.array([int(frame_step)] * row_count, type=pa.int64()))
        .append_column("sample_plan_id", pa.array([sample_plan_id] * row_count, type=pa.string()))
    )


def _rows_by_video(table: pa.Table) -> dict[str, list[dict[str, Any]]]:
    rows = table.to_pylist()
    by_video: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_video.setdefault(str(row["video_path"]), []).append(row)
    for group in by_video.values():
        group.sort(key=lambda item: int(item["clip_local_frame_index"]))
    return by_video


def _contiguous_sample_runs(rows: Sequence[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    sorted_rows = sorted(rows, key=lambda row: int(row["sample_index"]))
    runs: list[list[Mapping[str, Any]]] = []
    current: list[Mapping[str, Any]] = []
    previous_sample_index: int | None = None
    for row in sorted_rows:
        sample_index = int(row["sample_index"])
        if previous_sample_index is None or sample_index == previous_sample_index + 1:
            current.append(row)
        else:
            runs.append(current)
            current = [row]
        previous_sample_index = sample_index
    if current:
        runs.append(current)
    return runs


def _decode_clip_frames(video_path: Path, frame_indices: Sequence[int]) -> dict[int, np.ndarray]:
    """Decode selected frames as uint8 grayscale arrays.

    The function is intentionally small so tests can monkeypatch it and so a
    future PyNvVideoCodec backend can replace only this layer.
    """
    import cv2

    requested = sorted({int(index) for index in frame_indices})
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open clip video: {video_path}")
    decoded: dict[int, np.ndarray] = {}
    try:
        for frame_index in requested:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"Could not decode frame {frame_index} from {video_path}")
            if frame.ndim == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            decoded[frame_index] = np.asarray(gray, dtype=np.uint8)
    finally:
        cap.release()
    return decoded


def _resize_gray(frame: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    import cv2

    target_h, target_w = int(size_hw[0]), int(size_hw[1])
    return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA).astype(np.uint8, copy=False)


def _first_frame_shape(table: pa.Table) -> tuple[int, int]:
    first = table.slice(0, 1).to_pylist()[0]
    decoded = _decode_clip_frames(Path(str(first["video_path"])), [int(first["clip_local_frame_index"])])
    frame = decoded[int(first["clip_local_frame_index"])]
    if frame.ndim != 2:
        raise ValueError(f"Decoded frame must be grayscale HxW, got shape={frame.shape}")
    return int(frame.shape[0]), int(frame.shape[1])


def _create_array_codecs() -> tuple[zarr.codecs.BytesCodec, list[zarr.codecs.BloscCodec]]:
    return (
        zarr.codecs.BytesCodec(),
        [zarr.codecs.BloscCodec(cname="lz4", clevel=1, shuffle="bitshuffle")],
    )


def _create_training_zarr(
    zarr_path: Path,
    *,
    selected: pa.Table,
    frame_shape: tuple[int, int],
    options: ClippedTrainingOptions,
    recording_dir: Path,
    frame_index_path: Path,
    frame_manifest: Mapping[str, Any],
    camera_serial: str,
    source_metadata_zarr: Path | None,
    command_line_args: Sequence[str] | None,
) -> zarr.Group:
    if zarr_path.exists():
        if not options.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing training Zarr: {zarr_path}")
        shutil.rmtree(zarr_path)
    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(zarr_path), mode="w", zarr_format=3)
    env_summary = get_environment_summary()
    height, width = frame_shape
    root.attrs.update(
        {
            "schema_version": "3.0.0",
            "zarr_format": 3,
            "zarr_purpose": "training",
            "training_schema_version": TRAINING_SCHEMA_VERSION,
            "created_at": _utc_now(),
            "created_by": MODULE_NAME,
            "recording_id": frame_manifest.get("recording_id") or recording_dir.name,
            "session_id": frame_manifest.get("session_id") or frame_manifest.get("recording_id") or recording_dir.name,
            "recording_name": recording_dir.name,
            "recording_path": str(recording_dir),
            "source_layout": "rolling_clips",
            "camera_serial": str(camera_serial),
            "has_raw_video": True,
            "width": int(width),
            "height": int(height),
            "total_frames": int(selected.num_rows),
            "source_recording_frame_index_path": str(frame_index_path),
            "source_frame_index_path": "source_frame_index.parquet",
            "source_frame_index_schema": SOURCE_FRAME_INDEX_SCHEMA_VERSION,
            "raw_video_storage": "materialized_training_samples",
            "git_info": get_git_info(),
            "platform_info": get_platform_info(collect_ip=False, disk_path=str(zarr_path)),
            "software_versions": env_summary.get("key_packages", {}),
            "environment": {
                "type": env_summary.get("environment_type", "unknown"),
                "name": env_summary.get("environment_name", "none"),
                "python_version": env_summary.get("python_version"),
            },
        }
    )
    if command_line_args is not None:
        root.attrs["command_line_args"] = list(command_line_args)

    if source_metadata_zarr is not None and source_metadata_zarr.exists():
        source_root = zarr.open_group(str(source_metadata_zarr), mode="r")
        for key in METADATA_ROOT_ATTR_KEYS:
            if key in source_root.attrs:
                root.attrs[key] = source_root.attrs[key]

    pipeline_params = root.require_group("pipeline_params")
    pipeline_params.attrs["clipped_training_import"] = {
        "frame_step": int(options.frame_step),
        "max_frames": options.max_frames,
        "target_count": options.target_count,
        "write_images_ds": bool(options.write_images_ds),
        "downsample_size": list(options.downsample_size),
        "chunk_size": int(options.chunk_size),
    }

    serializer, compressors = _create_array_codecs()
    raw = root.require_group("raw_video")
    raw.attrs.update(
        {
            "import_method": MODULE_NAME,
            "import_mode": "sampled_clipped_recording",
            "import_purpose": "training_data",
            "source_layout": "rolling_clips",
            "frame_step": int(options.frame_step),
            "imported_frame_count": int(selected.num_rows),
            "original_frame_indices_semantics": "parent_frame_index",
            "source_frame_index_path": "source_frame_index.parquet",
            "original_resolution": [int(height), int(width)],
            "has_full_resolution": True,
            "has_downsampled": bool(options.write_images_ds),
        }
    )
    raw.create_array(
        "images_full",
        shape=(int(selected.num_rows), int(height), int(width)),
        chunks=(max(1, int(options.chunk_size)), int(height), int(width)),
        dtype=np.uint8,
        fill_value=0,
        serializer=serializer,
        compressors=compressors,
    )
    if options.write_images_ds:
        ds_h, ds_w = int(options.downsample_size[0]), int(options.downsample_size[1])
        raw.create_array(
            "images_ds",
            shape=(int(selected.num_rows), ds_h, ds_w),
            chunks=(max(1, int(options.chunk_size)), ds_h, ds_w),
            dtype=np.uint8,
            fill_value=0,
            serializer=serializer,
            compressors=compressors,
        )
        raw["images_ds"].attrs.update({"format": "gray", "resolution": [ds_h, ds_w]})
    raw.create_array(
        "original_frame_indices",
        data=np.asarray(selected["parent_frame_index"].to_numpy(zero_copy_only=False), dtype=np.int64),
        chunks=(min(1000, max(1, int(selected.num_rows))),),
        serializer=serializer,
        compressors=[],
    )
    raw.create_array(
        "timestamps",
        data=np.asarray(
            [np.nan if value is None else float(value) for value in selected["timestamp"].to_pylist()],
            dtype=np.float64,
        ),
        chunks=(min(1000, max(1, int(selected.num_rows))),),
        serializer=serializer,
        compressors=[],
    )

    for group_name in RUN_GROUPS:
        group = root.require_group(group_name)
        group.attrs["latest"] = None

    root.require_group("analysis")
    analysis_metadata = root.require_group("analysis_metadata")
    if source_metadata_zarr is not None and source_metadata_zarr.exists():
        source_root = zarr.open_group(str(source_metadata_zarr), mode="r")
        source_analysis = source_root.get("analysis_metadata")
        if source_analysis is not None:
            for key, value in dict(source_analysis.attrs).items():
                analysis_metadata.attrs[key] = value
            analysis_metadata.attrs["analysis_metadata_source_zarr"] = str(source_metadata_zarr)
    analysis_metadata.attrs.update(_dish_mask_policy_attrs([str(camera_serial)]))
    analysis_metadata.attrs["dish_mask_policy_schema"] = DISH_MASK_POLICY_SCHEMA_VERSION
    analysis_metadata.attrs["recording_camera_frame_shape"] = [int(height), int(width)]
    if options.require_dish_mask and "dish_mask" not in analysis_metadata.attrs:
        raise ValueError(
            "--require-dish-mask was set, but no dish_mask was copied into analysis_metadata. "
            "Pass --copy-existing-detections-from or tune/copy a mask first."
        )
    root.require_group("calibration")
    return root


def _copy_compatible_detection_groups(
    *,
    source_zarr: Path,
    dest_zarr: Path,
    expected_parent_frame_indices: np.ndarray,
    frame_shape: tuple[int, int],
) -> dict[str, Any]:
    source_root = zarr.open_group(str(source_zarr), mode="r")
    raw = source_root.get("raw_video")
    if raw is None or "original_frame_indices" not in raw:
        raise ValueError(
            f"Cannot copy detections from {source_zarr}: raw_video/original_frame_indices is missing."
        )
    source_indices = np.asarray(raw["original_frame_indices"][:], dtype=np.int64)
    expected = np.asarray(expected_parent_frame_indices, dtype=np.int64)
    if source_indices.shape != expected.shape or not np.array_equal(source_indices, expected):
        raise ValueError(
            "Cannot copy existing detections: source raw_video/original_frame_indices does not exactly "
            f"match selected parent_frame_index values. source_shape={source_indices.shape} "
            f"expected_shape={expected.shape}"
        )
    if "images_full" in raw:
        source_shape = tuple(int(v) for v in raw["images_full"].shape[1:3])
        if source_shape != tuple(frame_shape):
            raise ValueError(
                "Cannot copy existing detections: source images_full shape does not match clipped frames. "
                f"source={source_shape} expected={tuple(frame_shape)}"
            )

    copied: list[str] = []
    skipped: list[str] = []
    for group_name in ("detect_runs", "refined_detect_runs"):
        source_group_path = source_zarr / group_name
        if not source_group_path.exists():
            skipped.append(group_name)
            continue
        dest_group_path = dest_zarr / group_name
        if dest_group_path.exists():
            shutil.rmtree(dest_group_path)
        shutil.copytree(source_group_path, dest_group_path)
        copied.append(group_name)

    root = zarr.open_group(str(dest_zarr), mode="r+")
    root.attrs["copied_detection_runs_from"] = str(source_zarr)
    root.attrs["copied_detection_runs_mapping_policy"] = "exact_original_frame_indices_match"
    return {
        "status": "ok",
        "source_zarr": str(source_zarr),
        "copied_groups": copied,
        "skipped_groups": skipped,
        "mapping_policy": "exact_original_frame_indices_match",
    }


def _preflight_existing_detection_source(
    *,
    source_zarr: Path,
    expected_parent_frame_indices: np.ndarray,
    require_dish_mask: bool,
) -> dict[str, Any]:
    source_root = zarr.open_group(str(source_zarr), mode="r")
    raw = source_root.get("raw_video")
    if raw is None or "original_frame_indices" not in raw:
        raise ValueError(
            f"Cannot copy detections from {source_zarr}: raw_video/original_frame_indices is missing."
        )
    source_indices = np.asarray(raw["original_frame_indices"][:], dtype=np.int64)
    expected = np.asarray(expected_parent_frame_indices, dtype=np.int64)
    compatible = source_indices.shape == expected.shape and np.array_equal(source_indices, expected)
    if not compatible:
        raise ValueError(
            "Cannot copy existing detections: source raw_video/original_frame_indices does not exactly "
            f"match selected parent_frame_index values. source_shape={source_indices.shape} "
            f"expected_shape={expected.shape}"
        )
    analysis = source_root.get("analysis_metadata")
    has_dish_mask = bool(analysis is not None and "dish_mask" in analysis.attrs)
    if require_dish_mask and not has_dish_mask:
        raise ValueError(
            f"--require-dish-mask was set, but source analysis_metadata has no dish_mask: {source_zarr}"
        )
    detect = source_root.get("detect_runs")
    refined = source_root.get("refined_detect_runs")
    return {
        "status": "ok",
        "source_zarr": str(source_zarr),
        "mapping_policy": "exact_original_frame_indices_match",
        "frame_count": int(expected.shape[0]),
        "has_dish_mask": has_dish_mask,
        "detect_latest": detect.attrs.get("latest") if detect is not None else None,
        "refined_detect_latest": refined.attrs.get("latest") if refined is not None else None,
    }


def _preflight_analysis_metadata_source(
    *,
    source_zarr: Path,
    require_dish_mask: bool,
) -> dict[str, Any]:
    source_root = zarr.open_group(str(source_zarr), mode="r")
    analysis = source_root.get("analysis_metadata")
    has_analysis_metadata = analysis is not None
    has_dish_mask = bool(analysis is not None and "dish_mask" in analysis.attrs)
    if require_dish_mask and not has_dish_mask:
        raise ValueError(
            f"--require-dish-mask was set, but source analysis_metadata has no dish_mask: {source_zarr}"
        )
    return {
        "status": "ok",
        "source_zarr": str(source_zarr),
        "has_analysis_metadata": has_analysis_metadata,
        "has_dish_mask": has_dish_mask,
    }


def _write_frames(
    root: zarr.Group,
    selected: pa.Table,
    *,
    frame_shape: tuple[int, int],
    options: ClippedTrainingOptions,
) -> dict[str, Any]:
    raw = root["raw_video"]
    images_full = raw["images_full"]
    images_ds = raw.get("images_ds")
    expected_h, expected_w = frame_shape
    by_video = _rows_by_video(selected)
    decode_seconds = 0.0
    write_seconds = 0.0
    clips_written = 0
    frames_written = 0
    zarr_write_batches = 0
    for video_text, rows in sorted(by_video.items()):
        local_indices = [int(row["clip_local_frame_index"]) for row in rows]
        start = time.perf_counter()
        decoded = _decode_clip_frames(Path(video_text), local_indices)
        decode_seconds += time.perf_counter() - start
        clips_written += 1
        for run_rows in _contiguous_sample_runs(rows):
            sample_start = int(run_rows[0]["sample_index"])
            sample_stop = int(run_rows[-1]["sample_index"]) + 1
            frames: list[np.ndarray] = []
            for row in run_rows:
                local_index = int(row["clip_local_frame_index"])
                frame = decoded[local_index]
                if frame.shape != (expected_h, expected_w):
                    raise ValueError(
                        f"Decoded frame shape mismatch for {video_text} frame {local_index}: "
                        f"{frame.shape} != {(expected_h, expected_w)}"
                    )
                frames.append(frame)
            batch = np.stack(frames, axis=0)
            start = time.perf_counter()
            images_full[sample_start:sample_stop] = batch
            if images_ds is not None:
                ds_batch = np.stack(
                    [_resize_gray(frame, options.downsample_size) for frame in frames],
                    axis=0,
                )
                images_ds[sample_start:sample_stop] = ds_batch
            write_seconds += time.perf_counter() - start
            frames_written += len(frames)
            zarr_write_batches += 1
    return {
        "clips_written": clips_written,
        "frames_written": frames_written,
        "decode_seconds": decode_seconds,
        "zarr_write_seconds": write_seconds,
        "zarr_write_batches": zarr_write_batches,
    }


def create_clipped_training_zarr(
    recording_dir: str | Path,
    *,
    output_zarr: str | Path | None = None,
    options: ClippedTrainingOptions | None = None,
    command_line_args: Sequence[str] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    opts = options or ClippedTrainingOptions()
    command_args = list(command_line_args) if command_line_args is not None else None
    recording_path = Path(recording_dir).expanduser().resolve()
    if not recording_path.exists():
        raise FileNotFoundError(f"Recording folder not found: {recording_path}")
    frame_index_path, frame_manifest = _resolve_recording_frame_index(recording_path)
    selected, camera_serial = _load_selected_source_rows(
        frame_index_path,
        camera_serial=opts.camera_serial,
        frame_step=opts.frame_step,
        max_frames=opts.max_frames,
        target_count=opts.target_count,
    )
    sample_plan_id = f"clipped_training_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    source_table = _append_source_training_columns(
        selected,
        frame_index_path=frame_index_path,
        frame_step=opts.frame_step,
        sample_plan_id=sample_plan_id,
    )
    zarr_path = Path(output_zarr).expanduser().resolve() if output_zarr else _default_output_zarr(recording_path, camera_serial)
    metadata_source_zarr = opts.copy_analysis_metadata_from or opts.copy_existing_detections_from
    if opts.require_dish_mask and metadata_source_zarr is None:
        raise ValueError(
            "--require-dish-mask requires --copy-analysis-metadata-from or "
            "--copy-existing-detections-from."
        )
    manifest_path = zarr_path.parent / f"{zarr_path.name}_clipped_training_manifest.json"
    source_parquet_path = zarr_path / "source_frame_index.parquet"
    clip_ids = sorted({str(value.as_py()) for value in source_table["clip_id"]})
    parent_values = np.asarray(source_table["parent_frame_index"].to_numpy(zero_copy_only=False), dtype=np.int64)
    planned = {
        "status": "ok",
        "generated_by": MODULE_NAME,
        "schema_version": TRAINING_SCHEMA_VERSION,
        "dry_run": bool(opts.dry_run),
        "recording_folder": str(recording_path),
        "output_zarr": str(zarr_path),
        "manifest_path": str(manifest_path) if opts.write_manifest else None,
        "source_frame_index_path": str(source_parquet_path),
        "recording_frame_index_path": str(frame_index_path),
        "camera_serial": camera_serial,
        "frame_step": int(opts.frame_step),
        "selected_frame_count": int(source_table.num_rows),
        "selected_parent_frame_min": int(parent_values.min()),
        "selected_parent_frame_max": int(parent_values.max()),
        "clip_count": len(clip_ids),
        "clip_ids": clip_ids,
        "copy_existing_detections_from": str(opts.copy_existing_detections_from)
        if opts.copy_existing_detections_from is not None
        else None,
        "copy_analysis_metadata_from": str(metadata_source_zarr) if metadata_source_zarr is not None else None,
        "command_line_args": command_args,
    }
    if metadata_source_zarr is not None:
        planned["analysis_metadata_preflight"] = _preflight_analysis_metadata_source(
            source_zarr=metadata_source_zarr.expanduser().resolve(),
            require_dish_mask=bool(opts.require_dish_mask),
        )
    if opts.copy_existing_detections_from is not None:
        planned["copy_existing_detections_preflight"] = _preflight_existing_detection_source(
            source_zarr=opts.copy_existing_detections_from.expanduser().resolve(),
            expected_parent_frame_indices=parent_values,
            require_dish_mask=bool(opts.require_dish_mask),
        )
    if opts.dry_run:
        return {**planned, "wrote_zarr": False, "wrote_manifest": False}

    shape_start = time.perf_counter()
    frame_shape = _first_frame_shape(source_table)
    shape_seconds = time.perf_counter() - shape_start
    root = _create_training_zarr(
        zarr_path,
        selected=source_table,
        frame_shape=frame_shape,
        options=opts,
        recording_dir=recording_path,
        frame_index_path=frame_index_path,
        frame_manifest=frame_manifest,
        camera_serial=camera_serial,
        source_metadata_zarr=metadata_source_zarr,
        command_line_args=command_args,
    )
    write_stats = _write_frames(root, source_table, frame_shape=frame_shape, options=opts)
    pq.write_table(source_table, source_parquet_path)

    copy_result: dict[str, Any] | None = None
    if opts.copy_existing_detections_from is not None:
        copy_result = _copy_compatible_detection_groups(
            source_zarr=opts.copy_existing_detections_from.expanduser().resolve(),
            dest_zarr=zarr_path,
            expected_parent_frame_indices=parent_values,
            frame_shape=frame_shape,
        )

    result = {
        **planned,
        "dry_run": False,
        "wrote_zarr": True,
        "wrote_manifest": bool(opts.write_manifest),
        "created_at_utc": _utc_now(),
        "host": socket.gethostname(),
        "pid": int(os.getpid()),
        "frame_shape": list(frame_shape),
        "source_frame_index_schema": SOURCE_FRAME_INDEX_SCHEMA_VERSION,
        "decode_shape_probe_seconds": shape_seconds,
        "timing": {
            **write_stats,
            "total_seconds": time.perf_counter() - started,
        },
        "dish_mask_policy": _dish_mask_policy_attrs([camera_serial]),
        "copied_detection_runs": copy_result,
    }
    if opts.write_manifest:
        _write_json(manifest_path, result, overwrite=opts.overwrite)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording_dir", type=Path)
    parser.add_argument("--output-zarr", type=Path)
    parser.add_argument("--camera-serial")
    parser.add_argument("--frame-step", type=int, default=5000)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--target-count", type=int)
    parser.add_argument("--downsample-size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"), default=(640, 640))
    parser.add_argument("--no-images-ds", action="store_true", help="Do not write raw_video/images_ds.")
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--copy-existing-detections-from", type=Path)
    parser.add_argument(
        "--copy-analysis-metadata-from",
        type=Path,
        help="Copy analysis_metadata attrs such as dish_mask without copying detection run groups.",
    )
    parser.add_argument("--require-dish-mask", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--apply", action="store_true", help="Write the clipped training Zarr. Default is dry-run.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def _print_summary(result: Mapping[str, Any]) -> None:
    print(f"status: {result.get('status')}")
    print(f"recording_folder: {result.get('recording_folder')}")
    print(f"output_zarr: {result.get('output_zarr')}")
    print(f"camera_serial: {result.get('camera_serial')}")
    print(f"frame_step: {result.get('frame_step')}")
    print(f"selected_frame_count: {result.get('selected_frame_count')}")
    print(f"clip_count: {result.get('clip_count')}")
    print(f"source_frame_index_path: {result.get('source_frame_index_path')}")
    print(f"wrote_zarr: {result.get('wrote_zarr')}")
    print(f"wrote_manifest: {result.get('wrote_manifest')}")
    copied = result.get("copied_detection_runs")
    if isinstance(copied, Mapping):
        print(f"copied_detection_groups: {copied.get('copied_groups')}")
    timing = result.get("timing")
    if isinstance(timing, Mapping):
        print(
            "timing: "
            f"decode={float(timing.get('decode_seconds', 0.0)):.3f}s "
            f"write={float(timing.get('zarr_write_seconds', 0.0)):.3f}s "
            f"total={float(timing.get('total_seconds', 0.0)):.3f}s"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.apply and args.dry_run:
        raise SystemExit("--apply and --dry-run are mutually exclusive")
    options = ClippedTrainingOptions(
        frame_step=int(args.frame_step),
        camera_serial=args.camera_serial,
        max_frames=args.max_frames,
        target_count=args.target_count,
        write_images_ds=not bool(args.no_images_ds),
        downsample_size=(int(args.downsample_size[0]), int(args.downsample_size[1])),
        chunk_size=int(args.chunk_size),
        overwrite=bool(args.overwrite),
        dry_run=(not bool(args.apply)) or bool(args.dry_run),
        write_manifest=bool(args.write_manifest),
        require_dish_mask=bool(args.require_dish_mask),
        copy_analysis_metadata_from=args.copy_analysis_metadata_from,
        copy_existing_detections_from=args.copy_existing_detections_from,
    )
    result = create_clipped_training_zarr(
        args.recording_dir,
        output_zarr=args.output_zarr,
        options=options,
        command_line_args=list(argv) if argv is not None else sys.argv[1:],
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True, default=_json_default))
    else:
        _print_summary(result)
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
