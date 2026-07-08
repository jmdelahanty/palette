"""Flat ROI caches from finalized clipped refined-detect collections."""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import zarr

from fisheye.shared import flat_roi_cache as flat_cache_mod
from fisheye.shared.crop_geometry import (
    DEFAULT_PREFERRED_CROP_POLICY_NAME,
    compute_centered_roi_mapping,
    infer_preferred_roi_size,
    normalize_roi_size,
)
from fisheye.shared.roi_pixel_contract import orange_mono_pynvvc_luma_pixel_contract
from fisheye.utils.resolve_clipped_refined_detect_collection import build_collection_frame_map


CLIPPED_COLLECTION_ROW_INDEX_SCHEMA = "palette_clipped_collection_flat_roi_cache_rows_v1"
CLIPPED_COLLECTION_CACHE_BUILDER_SCHEMA = "palette_clipped_collection_flat_roi_cache_builder_v1"


def build_clipped_collection_flat_roi_cache(
    *,
    zarr_path: str | Path,
    collection_id: str | None = None,
    recording_frame_index: str | Path | None = None,
    clip_ids: Sequence[str] | None = None,
    work_unit_ids: Sequence[str] | None = None,
    output_dir: str | Path | None = None,
    manifest_path: str | Path | None = None,
    roi_size: Sequence[int] | None = None,
    limit_rows: int | None = None,
    overwrite: bool = False,
    compute_sha256: bool = False,
    gpu_chunk_frames: int = 32,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    progress_interval_seconds: float = 30.0,
    progress_every_batches: int = 0,
) -> dict[str, Any]:
    """Build a flat binary ROI cache from a finalized clipped collection.

    The output uses the same flat-bin payload contract as ``build_flat_roi_cache``
    but the source is a finalized clipped refined-detect collection instead of a
    single ``crop_runs/<run>`` group. A sidecar Parquet row index records the
    per-row clip, frame, and refined-detection lineage.
    """

    archive_path = Path(zarr_path).expanduser().resolve()
    if not archive_path.exists():
        raise FileNotFoundError(f"Zarr path not found: {archive_path}")
    if output_dir is None and manifest_path is None:
        raise ValueError("Provide either output_dir or manifest_path.")

    root = _open_root(archive_path)
    resolved_roi_size = _resolve_roi_size(root, roi_size)
    video_shape = _resolve_video_shape(root)
    collection_summary, frame_map = build_collection_frame_map(
        archive_path,
        collection_id=collection_id,
        recording_frame_index=recording_frame_index,
    )
    selection = _build_selection(clip_ids=clip_ids, work_unit_ids=work_unit_ids)
    collection_summary = _collection_summary_for_selection(collection_summary, selection)
    row_index = _build_row_index(
        root=root,
        frame_map=frame_map,
        selected_runs=collection_summary["selected_runs"],
        roi_size=resolved_roi_size,
        video_shape=video_shape,
        limit_rows=limit_rows,
    )
    total_rois = int(row_index.num_rows)
    cache_key = _build_collection_cache_key(
        archive_path=archive_path,
        collection_summary=collection_summary,
        roi_size=resolved_roi_size,
        row_index=row_index,
    )
    resolved_manifest_path = _resolve_manifest_path(
        archive_path=archive_path,
        collection_id=str(collection_summary["collection_id"]),
        cache_key=cache_key,
        output_dir=output_dir,
        manifest_path=manifest_path,
    )
    resolved_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    bin_path = resolved_manifest_path.with_suffix(".bin")
    row_index_path = resolved_manifest_path.with_suffix(".rows.parquet")

    if resolved_manifest_path.exists() and not overwrite:
        manifest = flat_cache_mod.load_flat_roi_cache_manifest(resolved_manifest_path)
        flat_cache_mod._validate_manifest_against_expected(
            manifest,
            expected_archive_path=archive_path,
            expected_crop_run=None,
            expected_shape=(total_rois, resolved_roi_size[0], resolved_roi_size[1]),
        )
        if not row_index_path.exists():
            raise FileNotFoundError(f"Existing manifest row index is missing: {row_index_path}")
        manifest.setdefault("manifest_path", str(resolved_manifest_path))
        flat_cache_mod._emit_progress(
            progress_callback,
            "reuse_existing",
            manifest_path=str(resolved_manifest_path),
            bin_path=str(bin_path),
            row_index_path=str(row_index_path),
            source=_progress_source_payload(collection_summary, total_rois, resolved_roi_size),
        )
        return manifest

    manifest_tmp = flat_cache_mod._temporary_path(resolved_manifest_path, suffix=".tmp.json")
    bin_tmp = flat_cache_mod._temporary_path(bin_path, suffix=".tmp.bin")
    row_index_tmp = flat_cache_mod._temporary_path(row_index_path, suffix=".tmp.parquet")
    total_bytes = int(total_rois) * int(resolved_roi_size[0]) * int(resolved_roi_size[1])
    started = time.perf_counter()
    flat_cache_mod._emit_progress(
        progress_callback,
        "start",
        manifest_path=str(resolved_manifest_path),
        bin_path=str(bin_path),
        row_index_path=str(row_index_path),
        total_rois=total_rois,
        total_bytes=total_bytes,
        gpu_chunk_frames=max(1, int(gpu_chunk_frames)),
        source=_progress_source_payload(collection_summary, total_rois, resolved_roi_size),
    )

    try:
        pq.write_table(row_index, row_index_tmp)
        timing = _write_collection_payload_pynvvc_luma(
            row_index=row_index,
            bin_tmp=bin_tmp,
            total_bytes=total_bytes,
            roi_size=resolved_roi_size,
            video_shape=video_shape,
            gpu_chunk_frames=max(1, int(gpu_chunk_frames)),
            progress_callback=progress_callback,
            progress_interval_seconds=float(progress_interval_seconds),
            progress_every_batches=int(progress_every_batches),
            started=started,
        )
        sha_started = time.perf_counter()
        payload_sha256 = flat_cache_mod._sha256_file(bin_tmp) if compute_sha256 else None
        timing["sha256_seconds_total"] += float(time.perf_counter() - sha_started)
    except Exception:
        for path in (bin_tmp, row_index_tmp):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise

    os.replace(bin_tmp, bin_path)
    os.replace(row_index_tmp, row_index_path)
    duration_seconds = float(time.perf_counter() - started)
    timing_summary = flat_cache_mod._timing_summary(
        timing=timing,
        total_rois=total_rois,
        total_bytes=total_bytes,
        duration_seconds=duration_seconds,
    )
    manifest = _build_manifest(
        archive_path=archive_path,
        collection_summary=collection_summary,
        manifest_path=resolved_manifest_path,
        bin_path=bin_path,
        row_index_path=row_index_path,
        cache_key=cache_key,
        roi_size=resolved_roi_size,
        total_rois=total_rois,
        total_bytes=total_bytes,
        payload_sha256=payload_sha256,
        duration_seconds=duration_seconds,
        timing_summary=timing_summary,
        limit_rows=limit_rows,
        selection=selection,
    )
    manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(manifest_tmp, resolved_manifest_path)
    flat_cache_mod._emit_progress(
        progress_callback,
        "complete",
        manifest_path=str(resolved_manifest_path),
        bin_path=str(bin_path),
        row_index_path=str(row_index_path),
        decode_backend_effective="pynvvc_luma",
        source=_progress_source_payload(collection_summary, total_rois, resolved_roi_size),
        progress=flat_cache_mod._progress_totals(
            rows_written=int(timing["rows"]),
            bytes_written=int(timing["bytes"]),
            total_rois=total_rois,
            total_bytes=total_bytes,
            elapsed_seconds=duration_seconds,
            timing=timing,
        ),
        timing=timing_summary,
    )
    return manifest


def _open_root(path: Path) -> zarr.Group:
    try:
        return zarr.open_group(str(path), mode="r", use_consolidated=False)
    except TypeError:
        return zarr.open_group(str(path), mode="r")


def _resolve_roi_size(root: zarr.Group, value: Sequence[int] | None) -> tuple[int, int]:
    if value is not None:
        return normalize_roi_size(value)
    return infer_preferred_roi_size(root)


def _resolve_video_shape(root: zarr.Group) -> tuple[int, int] | None:
    height = root.attrs.get("height")
    width = root.attrs.get("width")
    if height is None or width is None:
        return None
    height_i = int(height)
    width_i = int(width)
    if height_i <= 0 or width_i <= 0:
        raise ValueError(f"Invalid root video dimensions: width={width_i}, height={height_i}")
    return height_i, width_i


def _normalize_selection_values(values: Sequence[str] | None) -> tuple[str, ...]:
    if not values:
        return ()
    normalized: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in normalized:
            normalized.append(text)
    return tuple(normalized)


def _build_selection(
    *,
    clip_ids: Sequence[str] | None,
    work_unit_ids: Sequence[str] | None,
) -> dict[str, tuple[str, ...]]:
    return {
        "clip_ids": _normalize_selection_values(clip_ids),
        "work_unit_ids": _normalize_selection_values(work_unit_ids),
    }


def _filter_selected_runs(
    selected_runs: Sequence[Mapping[str, Any]],
    *,
    clip_ids: Sequence[str] | None = None,
    work_unit_ids: Sequence[str] | None = None,
) -> list[Mapping[str, Any]]:
    clip_filter = set(_normalize_selection_values(clip_ids))
    work_unit_filter = set(_normalize_selection_values(work_unit_ids))
    if not clip_filter and not work_unit_filter:
        return list(selected_runs)

    filtered: list[Mapping[str, Any]] = []
    for selected in selected_runs:
        clip_id = str(selected.get("clip_id") or "").strip()
        work_unit_id = str(selected.get("work_unit_id") or "").strip()
        if clip_filter and clip_id not in clip_filter:
            continue
        if work_unit_filter and work_unit_id not in work_unit_filter:
            continue
        filtered.append(selected)

    if not filtered:
        details = {
            "clip_ids": sorted(clip_filter),
            "work_unit_ids": sorted(work_unit_filter),
            "available_clip_ids": sorted({str(row.get("clip_id") or "") for row in selected_runs if row.get("clip_id")}),
            "available_work_unit_ids": sorted(
                {str(row.get("work_unit_id") or "") for row in selected_runs if row.get("work_unit_id")}
            ),
        }
        raise ValueError(f"Selected clipped collection filter matched no runs: {details}")
    return filtered


def _collection_summary_for_selection(
    collection_summary: Mapping[str, Any],
    selection: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    selected_runs = _filter_selected_runs(
        collection_summary.get("selected_runs", []),
        clip_ids=selection.get("clip_ids"),
        work_unit_ids=selection.get("work_unit_ids"),
    )
    summary = dict(collection_summary)
    summary["selected_runs"] = selected_runs
    summary["selected_run_count"] = len(selected_runs)
    summary["selection"] = {
        "clip_ids": list(selection.get("clip_ids") or ()),
        "work_unit_ids": list(selection.get("work_unit_ids") or ()),
        "filtered": bool(selection.get("clip_ids") or selection.get("work_unit_ids")),
    }
    return summary


def _build_row_index(
    *,
    root: zarr.Group,
    frame_map: pa.Table,
    selected_runs: Sequence[Mapping[str, Any]],
    roi_size: tuple[int, int],
    video_shape: tuple[int, int] | None,
    limit_rows: int | None,
) -> pa.Table:
    row_payloads: list[dict[str, Any]] = []
    remaining = None if limit_rows is None else max(0, int(limit_rows))
    frame_lookup = _frame_lookup_by_pair(frame_map)
    width = int(video_shape[1]) if video_shape is not None else None
    height = int(video_shape[0]) if video_shape is not None else None

    for selected in selected_runs:
        if remaining is not None and remaining <= 0:
            break
        refined_path = str(selected.get("refined_group_path") or "").strip().strip("/")
        if not refined_path:
            raise ValueError(f"Selected run is missing refined_group_path: {selected}")
        refined_group = root[refined_path]
        instances = refined_group["instances"] if "instances" in refined_group else refined_group
        frame_indices = np.asarray(instances["frame_indices"][:], dtype=np.int64).reshape(-1)
        bbox_norm = np.asarray(instances["bbox_norm_coords"][:], dtype=np.float64).reshape(-1, 4)
        row_count = int(frame_indices.shape[0])
        if bbox_norm.shape[0] != row_count:
            raise ValueError(f"{refined_path}: bbox_norm_coords and frame_indices length mismatch")
        if remaining is not None:
            row_count = min(row_count, remaining)
            frame_indices = frame_indices[:row_count]
            bbox_norm = bbox_norm[:row_count]
            remaining -= row_count
        if row_count == 0:
            continue

        if "bbox_img_xyxy" in instances:
            bbox_img_xyxy = np.asarray(instances["bbox_img_xyxy"][:], dtype=np.float64).reshape(-1, 4)[:row_count]
        else:
            if width is None or height is None:
                raise ValueError(
                    f"{refined_path}: bbox_img_xyxy is missing and root attrs do not provide width/height "
                    "for bbox_norm_coords conversion"
                )
            bbox_img_xyxy = _bbox_norm_cxcywh_to_img_xyxy(bbox_norm, width=width, height=height)
        roi_offsets, roi_sizes = compute_centered_roi_mapping(bbox_img_xyxy, roi_size=roi_size)
        pair_lookup = frame_lookup.get(
            (str(selected.get("camera_serial") or ""), str(selected.get("clip_id") or "")),
            {},
        )
        clip_mapping = _lookup_frame_map_rows(frame_indices, pair_lookup)
        refined_row_ids = _read_optional_array(instances, "refined_row_ids", np.int64, row_count, default=-1)
        source_detect_rows = _read_optional_array(
            instances,
            "source_detect_row_index",
            np.int64,
            row_count,
            default=-1,
        )
        source_kind_codes = _read_optional_array(instances, "source_kind_codes", np.int8, row_count, default=0)
        manual_edit_flags = _read_optional_array(instances, "manual_edit_flags", bool, row_count, default=False)
        confidence_scores = _read_optional_array(
            instances,
            "confidence_scores",
            np.float32,
            row_count,
            default=np.nan,
        )
        class_ids = _read_optional_array(instances, "class_ids", np.int32, row_count, default=-1)
        source = selected.get("source") if isinstance(selected.get("source"), Mapping) else {}
        video_path = str(source.get("video_path") or "")
        if not video_path:
            raise ValueError(f"Selected run is missing source.video_path: {selected}")

        row_payloads.append(
            {
                "work_unit_id": _repeat_str(selected.get("work_unit_id"), row_count),
                "camera_serial": _repeat_str(selected.get("camera_serial"), row_count),
                "clip_id": _repeat_str(selected.get("clip_id"), row_count),
                "clip_index": np.full(row_count, int(selected.get("clip_index") or 0), dtype=np.int32),
                "clip_local_frame_index": frame_indices.astype(np.int64, copy=False),
                "recording_frame_id": clip_mapping["recording_frame_id"],
                "parent_frame_index": clip_mapping["parent_frame_index"],
                "timestamp": clip_mapping["timestamp"],
                "timestamp_sys": clip_mapping["timestamp_sys"],
                "refined_group_path": _repeat_str(refined_path, row_count),
                "refined_detect_run": _repeat_str(selected.get("refined_detect_run"), row_count),
                "refined_instance_row_index": np.arange(row_count, dtype=np.int64),
                "refined_row_id": refined_row_ids.astype(np.int64, copy=False),
                "source_detect_row_index": source_detect_rows.astype(np.int64, copy=False),
                "source_kind_code": source_kind_codes.astype(np.int8, copy=False),
                "manual_edit_flag": manual_edit_flags.astype(bool, copy=False),
                "confidence_score": confidence_scores.astype(np.float32, copy=False),
                "class_id": class_ids.astype(np.int32, copy=False),
                "bbox_norm_cx": bbox_norm[:, 0].astype(np.float32, copy=False),
                "bbox_norm_cy": bbox_norm[:, 1].astype(np.float32, copy=False),
                "bbox_norm_w": bbox_norm[:, 2].astype(np.float32, copy=False),
                "bbox_norm_h": bbox_norm[:, 3].astype(np.float32, copy=False),
                "roi_x": roi_offsets[:, 0].astype(np.int32, copy=False),
                "roi_y": roi_offsets[:, 1].astype(np.int32, copy=False),
                "roi_w": roi_sizes[:, 0].astype(np.int32, copy=False),
                "roi_h": roi_sizes[:, 1].astype(np.int32, copy=False),
                "video_path": _repeat_str(video_path, row_count),
                "metadata_path": _repeat_str(source.get("metadata_path"), row_count),
                "keyframe_path": _repeat_str(source.get("keyframe_path"), row_count),
            }
        )

    if not row_payloads:
        return _row_table_from_columns({"roi_row_index": np.empty((0,), dtype=np.int64)})

    columns: dict[str, Any] = {}
    for name in row_payloads[0]:
        values = [payload[name] for payload in row_payloads]
        if isinstance(values[0], np.ndarray):
            columns[name] = np.concatenate(values)
        else:
            merged: list[Any] = []
            for item in values:
                merged.extend(item)
            columns[name] = merged
    total_rows = int(len(columns["clip_local_frame_index"]))
    columns = {"roi_row_index": np.arange(total_rows, dtype=np.int64), **columns}
    return _row_table_from_columns(columns)


def _bbox_norm_cxcywh_to_img_xyxy(
    bbox_norm_coords: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    bbox = np.asarray(bbox_norm_coords, dtype=np.float64).reshape(-1, 4)
    cx = bbox[:, 0] * float(width)
    cy = bbox[:, 1] * float(height)
    bw = bbox[:, 2] * float(width)
    bh = bbox[:, 3] * float(height)
    return np.stack((cx - bw * 0.5, cy - bh * 0.5, cx + bw * 0.5, cy + bh * 0.5), axis=1)


def _read_optional_array(
    group: zarr.Group,
    name: str,
    dtype: Any,
    row_count: int,
    *,
    default: Any,
) -> np.ndarray:
    if name not in group:
        return np.full(row_count, default, dtype=dtype)
    values = np.asarray(group[name][:], dtype=dtype).reshape(-1)
    if values.shape[0] < row_count:
        raise ValueError(f"{group.path}/{name} has {values.shape[0]} rows, expected at least {row_count}")
    return values[:row_count]


def _repeat_str(value: Any, count: int) -> list[str]:
    text = "" if value is None else str(value)
    return [text] * int(count)


def _frame_lookup_by_pair(frame_map: pa.Table) -> dict[tuple[str, str], dict[int, Mapping[str, Any]]]:
    if frame_map.num_rows == 0:
        return {}
    table = frame_map.combine_chunks()
    camera_values = pc.cast(table["camera_serial"], "string").to_pylist()
    clip_values = pc.cast(table["clip_id"], "string").to_pylist()
    local_values = table["clip_local_frame_index"].to_pylist()
    recording_values = table["recording_frame_id"].to_pylist()
    parent_values = (
        table["parent_frame_index"].to_pylist()
        if "parent_frame_index" in table.column_names
        else [None] * table.num_rows
    )
    timestamp_values = (
        table["timestamp"].to_pylist() if "timestamp" in table.column_names else [None] * table.num_rows
    )
    timestamp_sys_values = (
        table["timestamp_sys"].to_pylist() if "timestamp_sys" in table.column_names else [None] * table.num_rows
    )
    out: dict[tuple[str, str], dict[int, Mapping[str, Any]]] = {}
    for camera, clip, local, recording, parent, timestamp, timestamp_sys in zip(
        camera_values,
        clip_values,
        local_values,
        recording_values,
        parent_values,
        timestamp_values,
        timestamp_sys_values,
    ):
        key = (str(camera), str(clip))
        out.setdefault(key, {})[int(local)] = {
            "recording_frame_id": int(recording),
            "parent_frame_index": int(parent) if parent is not None else int(recording) - 1,
            "timestamp": None if timestamp is None else float(timestamp),
            "timestamp_sys": None if timestamp_sys is None else float(timestamp_sys),
        }
    return out


def _lookup_frame_map_rows(
    frame_indices: np.ndarray,
    pair_lookup: Mapping[int, Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    row_count = int(frame_indices.shape[0])
    recording = np.full(row_count, -1, dtype=np.int64)
    parent = np.full(row_count, -1, dtype=np.int64)
    timestamp = np.full(row_count, np.nan, dtype=np.float64)
    timestamp_sys = np.full(row_count, np.nan, dtype=np.float64)
    for idx, frame_idx in enumerate(frame_indices.tolist()):
        row = pair_lookup.get(int(frame_idx))
        if row is None:
            continue
        recording[idx] = int(row["recording_frame_id"])
        parent[idx] = int(row["parent_frame_index"])
        if row.get("timestamp") is not None:
            timestamp[idx] = float(row["timestamp"])
        if row.get("timestamp_sys") is not None:
            timestamp_sys[idx] = float(row["timestamp_sys"])
    if np.any(recording < 0):
        missing = int(np.count_nonzero(recording < 0))
        raise ValueError(f"recording_frame_index is missing {missing} ROI frame mapping row(s)")
    return {
        "recording_frame_id": recording,
        "parent_frame_index": parent,
        "timestamp": timestamp,
        "timestamp_sys": timestamp_sys,
    }


def _row_table_from_columns(columns: Mapping[str, Any]) -> pa.Table:
    return pa.table({name: value for name, value in columns.items()})


def _write_collection_payload_pynvvc_luma(
    *,
    row_index: pa.Table,
    bin_tmp: Path,
    total_bytes: int,
    roi_size: tuple[int, int],
    video_shape: tuple[int, int] | None,
    gpu_chunk_frames: int,
    progress_callback: Callable[[dict[str, Any]], None] | None,
    progress_interval_seconds: float,
    progress_every_batches: int,
    started: float,
) -> dict[str, Any]:
    total_rois = int(row_index.num_rows)
    timing = flat_cache_mod._new_timing()
    last_progress = started
    row_stride = int(roi_size[0]) * int(roi_size[1])
    if total_rois == 0:
        with bin_tmp.open("wb") as handle:
            handle.truncate(0)
        return timing

    table = row_index.combine_chunks()
    roi_ids_all = np.asarray(table["roi_row_index"].to_numpy(), dtype=np.int64)
    frames_all = np.asarray(table["clip_local_frame_index"].to_numpy(), dtype=np.int64)
    coords_full = np.stack(
        (
            np.asarray(table["roi_x"].to_numpy(), dtype=np.int32),
            np.asarray(table["roi_y"].to_numpy(), dtype=np.int32),
        ),
        axis=1,
    )
    video_paths = pc.cast(table["video_path"], "string").to_pylist()
    unique_video_paths = list(dict.fromkeys(str(path) for path in video_paths))
    batch_index = 0
    rows_written = np.zeros(total_rois, dtype=bool)

    with bin_tmp.open("w+b") as handle:
        handle.truncate(int(total_bytes))
        for video_path in unique_video_paths:
            video_row_indices = np.flatnonzero(np.asarray([path == video_path for path in video_paths], dtype=bool))
            if video_row_indices.size == 0:
                continue
            frame_to_roi: dict[int, list[int]] = {}
            for row_idx in video_row_indices.tolist():
                frame_to_roi.setdefault(int(frames_all[row_idx]), []).append(int(roi_ids_all[row_idx]))
            max_frame = int(max(frame_to_roi)) if frame_to_roi else -1
            reader = flat_cache_mod._open_pynvvc_luma_reader(Path(video_path))
            try:
                source_height = int(reader.source_height)
                source_width = int(reader.source_width)
                if video_shape is not None and (source_height, source_width) != (
                    int(video_shape[0]),
                    int(video_shape[1]),
                ):
                    expected_h, expected_w = int(video_shape[0]), int(video_shape[1])
                    raise ValueError(
                        "PyNvVideoCodec source dimensions do not match zarr metadata: "
                        f"decoder={source_width}x{source_height}, metadata={expected_w}x{expected_h}."
                    )
                batch_index, last_progress = flat_cache_mod._write_pynvvc_luma_streamed_rois(
                    reader=reader,
                    frame_to_roi=frame_to_roi,
                    roi_coordinates_full=coords_full,
                    roi_shape=roi_size,
                    video_shape=(source_height, source_width),
                    handle=handle,
                    row_stride=row_stride,
                    rows_written_mask=rows_written,
                    total_rois=total_rois,
                    total_bytes=total_bytes,
                    timing=timing,
                    batch_index=batch_index,
                    last_progress=last_progress,
                    started=started,
                    progress_callback=progress_callback,
                    progress_interval_seconds=progress_interval_seconds,
                    progress_every_batches=progress_every_batches,
                    staging_row_capacity=max(1024, int(gpu_chunk_frames)),
                    progress_context={"video_path": str(video_path)},
                )
            finally:
                reader.close()

    written = int(rows_written.sum())
    if written != total_rois:
        raise RuntimeError(f"PyNvVideoCodec clipped collection cache wrote {written}/{total_rois} rows")
    return timing


def _build_collection_cache_key(
    *,
    archive_path: Path,
    collection_summary: Mapping[str, Any],
    roi_size: tuple[int, int],
    row_index: pa.Table,
) -> str:
    payload = {
        "schema": CLIPPED_COLLECTION_CACHE_BUILDER_SCHEMA,
        "archive_path": str(archive_path),
        "collection_id": collection_summary.get("collection_id"),
        "collection_path": collection_summary.get("collection_path"),
        "recording_frame_index": collection_summary.get("recording_frame_index"),
        "selected_run_count": collection_summary.get("selected_run_count"),
        "selection": collection_summary.get("selection"),
        "roi_size": list(roi_size),
        "row_count": int(row_index.num_rows),
        "row_index_preview": row_index.slice(0, min(10, row_index.num_rows)).to_pylist(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _resolve_manifest_path(
    *,
    archive_path: Path,
    collection_id: str,
    cache_key: str,
    output_dir: str | Path | None,
    manifest_path: str | Path | None,
) -> Path:
    if manifest_path is not None:
        return Path(manifest_path).expanduser().resolve()
    assert output_dir is not None
    safe_archive = flat_cache_mod._safe_component(archive_path.stem)
    safe_collection = flat_cache_mod._safe_component(collection_id)
    return (
        Path(output_dir).expanduser().resolve()
        / f"{safe_archive}__{safe_collection}__{cache_key[:12]}.flat_roi_cache.json"
    )


def _build_manifest(
    *,
    archive_path: Path,
    collection_summary: Mapping[str, Any],
    manifest_path: Path,
    bin_path: Path,
    row_index_path: Path,
    cache_key: str,
    roi_size: tuple[int, int],
    total_rois: int,
    total_bytes: int,
    payload_sha256: str | None,
    duration_seconds: float,
    timing_summary: Mapping[str, Any],
    limit_rows: int | None,
    selection: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    roi_h, roi_w = int(roi_size[0]), int(roi_size[1])
    pixel_contract = orange_mono_pynvvc_luma_pixel_contract()
    row_columns = [
        "roi_row_index",
        "camera_serial",
        "clip_id",
        "clip_local_frame_index",
        "recording_frame_id",
        "refined_row_id",
        "source_detect_row_index",
        "bbox_norm_cx",
        "bbox_norm_cy",
        "bbox_norm_w",
        "bbox_norm_h",
        "roi_x",
        "roi_y",
        "video_path",
    ]
    return {
        "schema": flat_cache_mod.FLAT_ROI_CACHE_SCHEMA,
        "layout": flat_cache_mod.FLAT_ROI_CACHE_LAYOUT,
        "cache_complete": True,
        "cache_key": cache_key,
        "manifest_path": str(manifest_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "archive_path": str(archive_path),
            "source_kind": "finalized_clipped_refined_detect_collection",
            "collection_id": collection_summary.get("collection_id"),
            "collection_path": collection_summary.get("collection_path"),
            "recording_frame_index": collection_summary.get("recording_frame_index"),
            "selected_run_count": collection_summary.get("selected_run_count"),
            "clip_count": len({row.get("clip_id") for row in collection_summary.get("selected_runs", [])}),
            "selection": {
                "clip_ids": list(selection.get("clip_ids") or ()),
                "work_unit_ids": list(selection.get("work_unit_ids") or ()),
                "filtered": bool(selection.get("clip_ids") or selection.get("work_unit_ids")),
            },
            "crop_policy": DEFAULT_PREFERRED_CROP_POLICY_NAME,
            "crop_storage_mode": "geometry_only_collection_derived",
            "frame_source_kind": "clip_source_video_path",
        },
        "row_index": {
            "schema": CLIPPED_COLLECTION_ROW_INDEX_SCHEMA,
            "path": flat_cache_mod._relative_or_absolute(row_index_path, base=manifest_path.parent),
            "row_count": int(total_rois),
            "columns": row_columns,
        },
        "array": {
            "bin_path": flat_cache_mod._relative_or_absolute(bin_path, base=manifest_path.parent),
            "dtype": "uint8",
            "shape": [int(total_rois), roi_h, roi_w],
            "order": "C",
            "row_stride_bytes": roi_h * roi_w,
            "total_bytes": int(total_bytes),
            "sha256": payload_sha256,
        },
        "builder": {
            "schema": CLIPPED_COLLECTION_CACHE_BUILDER_SCHEMA,
            "duration_seconds": float(duration_seconds),
            "decode_backend_requested": "pynvvc_luma",
            "decode_backend_effective": "pynvvc_luma",
            "pixel_contract": pixel_contract,
            "pixel_contract_name": pixel_contract.get("name"),
            "timing": dict(timing_summary),
            "limit_rows": None if limit_rows is None else int(limit_rows),
            "format_note": (
                "flat_bin_v1 stores ROI rows contiguously as raw uint8 bytes. "
                "This collection variant records row lineage in row_index.path."
            ),
        },
    }


def _progress_source_payload(
    collection_summary: Mapping[str, Any],
    total_rois: int,
    roi_size: tuple[int, int],
) -> dict[str, Any]:
    return {
        "source_kind": "finalized_clipped_refined_detect_collection",
        "collection_id": str(collection_summary.get("collection_id") or ""),
        "selected_run_count": int(collection_summary.get("selected_run_count") or 0),
        "total_rois": int(total_rois),
        "roi_shape": [int(roi_size[0]), int(roi_size[1])],
        "decode_backend": "pynvvc_luma",
    }
