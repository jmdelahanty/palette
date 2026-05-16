"""Compare stored training ROI crops against PyNvVideoCodec luma crops.

This diagnostic is intended for validating whether the fast sequential
PyNvVideoCodec luma path produces the same model-facing ROI pixels as existing
training Zarr crop images. For training Zarrs, crop frame indices are often
local sampled-frame indices; when ``raw_video/original_frame_indices`` is
available, the diagnostic maps those rows back to original MP4 frame numbers
before decoding.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import zarr

from fisheye.shared.flat_roi_cache import _crop_pynvvc_luma_frame
from fisheye.shared.roi_pixel_contract import (
    crop_run_pixel_contract,
    flat_cache_pixel_contract_for_backend,
    normalize_pixel_contract,
)


SOURCE_FRAME_INDEX_MODES = ("auto", "direct", "original_frame_indices")


def _open_pynvvc_luma_reader(video_path: Path) -> Any:
    from fisheye.shared.flat_roi_cache import _open_pynvvc_luma_reader as open_reader

    return open_reader(video_path)


def _parse_rows(value: str | None) -> list[int] | None:
    if value is None or not value.strip():
        return None
    return [int(part) for part in value.replace(",", " ").split()]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _valid_attr_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"unknown", "none", "null"}:
        return None
    return text


def _first_attr_text(*values: Any) -> str | None:
    for value in values:
        text = _valid_attr_text(value)
        if text:
            return text
    return None


def _first_positive_int(*values: Any) -> int | None:
    for value in values:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


def _resolve_crop_run(root: Any, crop_run: str | None) -> str:
    crop_parent = root.get("crop_runs")
    if crop_parent is None:
        raise KeyError("Zarr archive is missing crop_runs.")
    if crop_run:
        if crop_run not in crop_parent:
            raise KeyError(f"Crop run '{crop_run}' not found.")
        return crop_run
    for attr_name in ("latest_materialized", "latest", "latest_any"):
        candidate = crop_parent.attrs.get(attr_name)
        if candidate and str(candidate) in crop_parent:
            return str(candidate)
    names = sorted(str(name) for name in crop_parent.group_keys())
    if len(names) == 1:
        return names[0]
    raise ValueError("Unable to resolve crop run; pass --crop-run.")


def _resolve_video_path(root: Any, crop_group: Any) -> Path:
    metadata = root.attrs.get("source_video_metadata")
    metadata_path = metadata.get("source_path") if isinstance(metadata, Mapping) else None
    text = _first_attr_text(
        crop_group.attrs.get("source_video_path"),
        crop_group.attrs.get("video_source_path"),
        root.attrs.get("source_video_path"),
        root.attrs.get("video_source_path"),
        root.attrs.get("source_path"),
        metadata_path,
    )
    if not text:
        raise ValueError("Unable to resolve source video path from crop/root attrs.")
    return Path(text).expanduser()


def _resolve_video_shape(root: Any, crop_group: Any) -> tuple[int, int]:
    metadata = root.attrs.get("source_video_metadata")
    metadata_height = metadata.get("height") if isinstance(metadata, Mapping) else None
    metadata_width = metadata.get("width") if isinstance(metadata, Mapping) else None
    height = _first_positive_int(crop_group.attrs.get("height"), root.attrs.get("height"), metadata_height)
    width = _first_positive_int(crop_group.attrs.get("width"), root.attrs.get("width"), metadata_width)
    if height is None or width is None:
        raise ValueError("Unable to resolve source video dimensions from crop/root attrs.")
    return int(height), int(width)


def _resolve_roi_shape(crop_group: Any) -> tuple[int, int]:
    roi_size = crop_group.attrs.get("roi_size")
    if isinstance(roi_size, (list, tuple)) and len(roi_size) == 2:
        return int(roi_size[0]), int(roi_size[1])
    if "roi_images" in crop_group and len(crop_group["roi_images"].shape) >= 3:
        return int(crop_group["roi_images"].shape[1]), int(crop_group["roi_images"].shape[2])
    raise ValueError("Unable to resolve ROI size from crop attrs or roi_images shape.")


def _load_original_frame_indices(root: Any) -> np.ndarray | None:
    raw_video = root.get("raw_video")
    if raw_video is None or "original_frame_indices" not in raw_video:
        return None
    return np.asarray(raw_video["original_frame_indices"][:], dtype=np.int64)


def _should_use_original_frame_indices(
    *,
    root: Any,
    crop_frame_indices: np.ndarray,
    original_frame_indices: np.ndarray | None,
    mode: str,
) -> bool:
    if mode == "direct":
        return False
    if mode == "original_frame_indices":
        if original_frame_indices is None:
            raise ValueError("--source-frame-index-mode original_frame_indices requested, but no mapping exists.")
        return True
    if original_frame_indices is None or crop_frame_indices.size == 0:
        return False
    max_local = int(np.max(crop_frame_indices))
    if max_local >= int(original_frame_indices.shape[0]):
        return False
    purpose = str(root.attrs.get("zarr_purpose") or "").strip().lower()
    source_total = _first_positive_int(root.attrs.get("source_video_total_frames"))
    if purpose == "training":
        return True
    if source_total is not None and source_total != int(original_frame_indices.shape[0]):
        return True
    return False


def _map_source_frame_indices(
    *,
    root: Any,
    crop_frame_indices: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    original_frame_indices = _load_original_frame_indices(root)
    use_original = _should_use_original_frame_indices(
        root=root,
        crop_frame_indices=crop_frame_indices,
        original_frame_indices=original_frame_indices,
        mode=mode,
    )
    if not use_original:
        return np.asarray(crop_frame_indices, dtype=np.int64), {
            "mode": "direct",
            "original_frame_indices_available": original_frame_indices is not None,
            "original_frame_indices_length": (
                int(original_frame_indices.shape[0]) if original_frame_indices is not None else None
            ),
        }

    assert original_frame_indices is not None
    local = np.asarray(crop_frame_indices, dtype=np.int64)
    bad = np.flatnonzero((local < 0) | (local >= int(original_frame_indices.shape[0])))
    if bad.size:
        example = int(bad[0])
        raise IndexError(
            "Crop frame index is outside raw_video/original_frame_indices: "
            f"row={example}, frame_index={int(local[example])}, "
            f"mapping_length={int(original_frame_indices.shape[0])}."
        )
    mapped = original_frame_indices[local]
    return np.asarray(mapped, dtype=np.int64), {
        "mode": "original_frame_indices",
        "original_frame_indices_available": True,
        "original_frame_indices_length": int(original_frame_indices.shape[0]),
    }


def _boundary_rows(
    *,
    roi_coordinates_full: np.ndarray,
    roi_shape: tuple[int, int],
    video_shape: tuple[int, int],
    limit: int,
) -> list[int]:
    if limit <= 0 or roi_coordinates_full.size == 0:
        return []
    roi_h, roi_w = int(roi_shape[0]), int(roi_shape[1])
    frame_h, frame_w = int(video_shape[0]), int(video_shape[1])
    coords = np.asarray(roi_coordinates_full, dtype=np.int64)
    x1 = coords[:, 0]
    y1 = coords[:, 1]
    outside = np.flatnonzero((x1 < 0) | (y1 < 0) | (x1 + roi_w > frame_w) | (y1 + roi_h > frame_h))
    return [int(v) for v in outside[: int(limit)]]


def _sample_rows(
    *,
    total_rois: int,
    sample_count: int,
    seed: int,
    explicit_rows: Sequence[int] | None,
    boundary_rows: Sequence[int],
    all_rows: bool,
) -> list[int]:
    if all_rows:
        rows = list(range(int(total_rois)))
    elif explicit_rows is not None:
        rows = [int(v) for v in explicit_rows]
    else:
        rows = [0, total_rois // 2, total_rois - 1] if total_rois > 0 else []
        rows.extend(int(v) for v in boundary_rows)
        remaining = max(0, int(sample_count) - len(set(rows)))
        if remaining > 0 and total_rois > 0:
            rng = np.random.default_rng(int(seed))
            random_rows = rng.choice(total_rois, size=min(remaining, total_rois), replace=False)
            rows.extend(int(v) for v in random_rows.tolist())

    unique = sorted(set(rows))
    for row in unique:
        if row < 0 or row >= total_rois:
            raise IndexError(f"ROI row {row} out of range for total_rois={total_rois}.")
    return unique


def _read_roi_rows(roi_images: Any, rows: Sequence[int]) -> np.ndarray:
    if len(roi_images.shape) != 3:
        raise ValueError(
            f"Expected crop_runs/<run>/roi_images to be 3D [roi, height, width], got {roi_images.shape}."
        )
    if not rows:
        shape = roi_images.shape
        return np.empty((0, int(shape[1]), int(shape[2])), dtype=np.uint8)
    return np.stack([np.asarray(roi_images[int(row)], dtype=np.uint8) for row in rows], axis=0)


def _safe_percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, q))


def _diff_summary(
    *,
    rows: Sequence[int],
    reference: np.ndarray,
    candidate: np.ndarray,
    top_mismatches: int,
) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        raise ValueError(f"Shape mismatch: reference={reference.shape}, candidate={candidate.shape}")
    if reference.size == 0:
        return {
            "rows_compared": 0,
            "pixels_compared": 0,
            "exact_rows": 0,
            "mismatched_rows": 0,
            "byte_equal": True,
            "max_abs_diff": 0,
            "mean_abs_diff": 0.0,
            "p95_abs_diff": 0.0,
            "top_mismatches": [],
        }

    diff = np.abs(reference.astype(np.int16) - candidate.astype(np.int16))
    row_max = diff.reshape(diff.shape[0], -1).max(axis=1)
    row_mean = diff.reshape(diff.shape[0], -1).mean(axis=1)
    exact_mask = row_max == 0
    mismatch_order = np.argsort(-row_max)
    top: list[dict[str, Any]] = []
    for local_idx in mismatch_order[: max(0, int(top_mismatches))]:
        local = int(local_idx)
        if row_max[local] == 0:
            break
        top.append(
            {
                "row": int(rows[local]),
                "max_abs_diff": int(row_max[local]),
                "mean_abs_diff": float(row_mean[local]),
            }
        )

    return {
        "rows_compared": int(reference.shape[0]),
        "pixels_compared": int(diff.size),
        "exact_rows": int(exact_mask.sum()),
        "mismatched_rows": int((~exact_mask).sum()),
        "byte_equal": bool(np.array_equal(reference, candidate)),
        "max_abs_diff": int(diff.max(initial=0)),
        "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
        "p95_abs_diff": _safe_percentile(diff.reshape(-1), 95.0),
        "top_mismatches": top,
    }


def _frame_to_selected_rows(
    *,
    rows: Sequence[int],
    source_frame_indices: np.ndarray,
) -> dict[int, list[int]]:
    mapping: dict[int, list[int]] = {}
    for row in rows:
        frame_idx = int(source_frame_indices[int(row)])
        mapping.setdefault(frame_idx, []).append(int(row))
    return mapping


def _decode_pynvvc_luma_rows(
    *,
    video_path: Path,
    rows: Sequence[int],
    source_frame_indices: np.ndarray,
    roi_coordinates_full: np.ndarray,
    roi_shape: tuple[int, int],
    video_shape: tuple[int, int],
    decode_chunk_frames: int,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    import torch

    frame_to_rows = _frame_to_selected_rows(rows=rows, source_frame_indices=source_frame_indices)
    if not frame_to_rows:
        return {}, {
            "video_open_seconds": 0.0,
            "decode_seconds": 0.0,
            "crop_seconds": 0.0,
            "contiguous_seconds": 0.0,
            "decoded_frames": 0,
            "frames_with_selected_rows": 0,
            "max_source_frame_decoded": None,
        }

    open_started = time.perf_counter()
    reader = _open_pynvvc_luma_reader(video_path)
    video_open_seconds = time.perf_counter() - open_started
    try:
        source_height = int(reader.source_height)
        source_width = int(reader.source_width)
        expected_height, expected_width = int(video_shape[0]), int(video_shape[1])
        if (source_height, source_width) != (expected_height, expected_width):
            raise ValueError(
                "PyNvVideoCodec dimensions do not match Zarr metadata: "
                f"decoder={source_width}x{source_height}, metadata={expected_width}x{expected_height}."
            )

        max_frame = int(max(frame_to_rows))
        decode_chunk = max(1, int(decode_chunk_frames))
        decoded_frame_index = 0
        decoded_frames = 0
        decode_seconds = 0.0
        crop_seconds = 0.0
        contiguous_seconds = 0.0
        produced: dict[int, np.ndarray] = {}

        while decoded_frame_index <= max_frame:
            count = min(decode_chunk, max_frame - decoded_frame_index + 1)
            started = time.perf_counter()
            frames = reader.decode_next(count)
            decode_seconds += time.perf_counter() - started
            if not frames:
                break
            decoded_frames += int(len(frames))

            for offset, frame_tensor in enumerate(frames):
                frame_idx = decoded_frame_index + offset
                roi_rows = frame_to_rows.get(frame_idx)
                if not roi_rows:
                    continue
                started = time.perf_counter()
                crops = _crop_pynvvc_luma_frame(
                    frame_tensor,
                    roi_ids=roi_rows,
                    roi_coordinates_full=roi_coordinates_full,
                    roi_shape=roi_shape,
                    video_shape=video_shape,
                )
                if torch.cuda.is_available() and getattr(crops, "is_cuda", False):
                    torch.cuda.synchronize()
                crop_seconds += time.perf_counter() - started

                started = time.perf_counter()
                crops_cpu = np.ascontiguousarray(crops.cpu().numpy(), dtype=np.uint8)
                contiguous_seconds += time.perf_counter() - started
                for local_idx, row in enumerate(roi_rows):
                    produced[int(row)] = np.ascontiguousarray(crops_cpu[local_idx], dtype=np.uint8)

            decoded_frame_index += int(len(frames))

        return produced, {
            "video_open_seconds": float(video_open_seconds),
            "decode_seconds": float(decode_seconds),
            "crop_seconds": float(crop_seconds),
            "contiguous_seconds": float(contiguous_seconds),
            "decoded_frames": int(decoded_frames),
            "frames_with_selected_rows": int(len(frame_to_rows)),
            "max_source_frame_decoded": int(decoded_frame_index - 1) if decoded_frames else None,
        }
    finally:
        reader.close()


def _stored_crop_pixel_contract(crop_group: Any) -> dict[str, Any]:
    explicit = normalize_pixel_contract(
        crop_group.attrs.get("roi_pixel_contract") or crop_group.attrs.get("crop_pixel_contract")
    )
    if explicit is not None:
        return explicit
    parameters = crop_group.attrs.get("parameters")
    parameter_acceleration = parameters.get("acceleration") if isinstance(parameters, Mapping) else None
    return crop_run_pixel_contract(
        crop_storage_mode=str(crop_group.attrs.get("crop_storage_mode") or "materialized"),
        video_source_type=str(crop_group.attrs.get("video_source_type") or ""),
        acceleration=str(crop_group.attrs.get("acceleration") or parameter_acceleration or ""),
    )


def check_training_crop_pynvvc_pixel_parity(
    *,
    zarr_path: str | Path,
    crop_run: str | None = None,
    rows: Sequence[int] | None = None,
    all_rows: bool = False,
    sample_count: int = 32,
    seed: int = 0,
    boundary_sample_count: int = 4,
    source_frame_index_mode: str = "auto",
    decode_chunk_frames: int = 32,
    max_abs_diff: int = 0,
    max_mean_abs_diff: float = 0.0,
    max_p95_abs_diff: float = 0.0,
    top_mismatches: int = 10,
) -> dict[str, Any]:
    if source_frame_index_mode not in SOURCE_FRAME_INDEX_MODES:
        raise ValueError(f"Unsupported source_frame_index_mode: {source_frame_index_mode}")

    archive_path = Path(zarr_path).expanduser().resolve()
    started_total = time.perf_counter()
    root = zarr.open_group(str(archive_path), mode="r", use_consolidated=False)
    resolved_crop_run = _resolve_crop_run(root, crop_run)
    crop_group = root["crop_runs"][resolved_crop_run]
    if "roi_images" not in crop_group:
        raise ValueError(f"crop_runs/{resolved_crop_run} is missing roi_images.")
    if "frame_indices" not in crop_group:
        raise ValueError(f"crop_runs/{resolved_crop_run} is missing frame_indices.")
    if "roi_coordinates_full" not in crop_group:
        raise ValueError(f"crop_runs/{resolved_crop_run} is missing roi_coordinates_full.")

    video_path = _resolve_video_path(root, crop_group)
    video_shape = _resolve_video_shape(root, crop_group)
    roi_shape = _resolve_roi_shape(crop_group)
    roi_images = crop_group["roi_images"]
    total_rois = int(roi_images.shape[0])
    frame_indices = np.asarray(crop_group["frame_indices"][:], dtype=np.int64)
    roi_coordinates_full = np.asarray(crop_group["roi_coordinates_full"][:], dtype=np.int32)
    if int(frame_indices.shape[0]) != total_rois:
        raise ValueError(
            f"frame_indices length {frame_indices.shape[0]} does not match roi_images rows {total_rois}."
        )
    if int(roi_coordinates_full.shape[0]) != total_rois:
        raise ValueError(
            "roi_coordinates_full length "
            f"{roi_coordinates_full.shape[0]} does not match roi_images rows {total_rois}."
        )

    source_frame_indices, frame_mapping_payload = _map_source_frame_indices(
        root=root,
        crop_frame_indices=frame_indices,
        mode=source_frame_index_mode,
    )
    selected_rows = _sample_rows(
        total_rois=total_rois,
        sample_count=sample_count,
        seed=seed,
        explicit_rows=rows,
        boundary_rows=_boundary_rows(
            roi_coordinates_full=roi_coordinates_full,
            roi_shape=roi_shape,
            video_shape=video_shape,
            limit=boundary_sample_count,
        ),
        all_rows=all_rows,
    )

    reference_started = time.perf_counter()
    reference = _read_roi_rows(roi_images, selected_rows)
    reference_read_seconds = time.perf_counter() - reference_started

    produced, decode_timing = _decode_pynvvc_luma_rows(
        video_path=video_path,
        rows=selected_rows,
        source_frame_indices=source_frame_indices,
        roi_coordinates_full=roi_coordinates_full,
        roi_shape=roi_shape,
        video_shape=video_shape,
        decode_chunk_frames=decode_chunk_frames,
    )
    present_rows = [row for row in selected_rows if row in produced]
    missing_rows = [row for row in selected_rows if row not in produced]
    candidate = (
        np.stack([produced[row] for row in present_rows], axis=0)
        if present_rows
        else np.empty((0, int(roi_shape[0]), int(roi_shape[1])), dtype=np.uint8)
    )
    present_reference = reference[[selected_rows.index(row) for row in present_rows]] if present_rows else reference[:0]
    diff = _diff_summary(
        rows=present_rows,
        reference=present_reference,
        candidate=candidate,
        top_mismatches=top_mismatches,
    )

    thresholds = {
        "max_abs_diff": int(max_abs_diff),
        "max_mean_abs_diff": float(max_mean_abs_diff),
        "max_p95_abs_diff": float(max_p95_abs_diff),
    }
    failures: list[str] = []
    if missing_rows:
        failures.append(f"PyNvVideoCodec did not produce {len(missing_rows)} selected rows.")
    if int(diff["max_abs_diff"]) > int(max_abs_diff):
        failures.append(f"max_abs_diff {diff['max_abs_diff']} exceeds threshold {int(max_abs_diff)}")
    if float(diff["mean_abs_diff"]) > float(max_mean_abs_diff):
        failures.append(
            f"mean_abs_diff {float(diff['mean_abs_diff']):.6f} exceeds threshold "
            f"{float(max_mean_abs_diff):.6f}"
        )
    if float(diff["p95_abs_diff"]) > float(max_p95_abs_diff):
        failures.append(
            f"p95_abs_diff {float(diff['p95_abs_diff']):.6f} exceeds threshold "
            f"{float(max_p95_abs_diff):.6f}"
        )

    source_frames_selected = [int(source_frame_indices[row]) for row in selected_rows]
    local_frames_selected = [int(frame_indices[row]) for row in selected_rows]
    duration_seconds = time.perf_counter() - started_total
    return {
        "status": "ok" if not failures else "fail",
        "failures": failures,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "hostname": socket.gethostname(),
            "LSB_JOBID": os.environ.get("LSB_JOBID"),
            "LSB_QUEUE": os.environ.get("LSB_QUEUE"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "inputs": {
            "zarr_path": str(archive_path),
            "crop_run": resolved_crop_run,
            "video_path": str(video_path),
            "all_rows": bool(all_rows),
            "sample_count_requested": int(sample_count),
            "sample_seed": int(seed),
            "explicit_rows": [int(v) for v in rows] if rows is not None else None,
            "boundary_sample_count": int(boundary_sample_count),
            "source_frame_index_mode_requested": str(source_frame_index_mode),
            "decode_chunk_frames": int(decode_chunk_frames),
        },
        "source": {
            "total_rois": int(total_rois),
            "roi_shape": [int(roi_shape[0]), int(roi_shape[1])],
            "roi_images_shape": [int(v) for v in roi_images.shape],
            "video_shape": [int(video_shape[0]), int(video_shape[1])],
            "stored_crop_pixel_contract": _json_safe(_stored_crop_pixel_contract(crop_group)),
            "pynvvc_pixel_contract": _json_safe(flat_cache_pixel_contract_for_backend("pynvvc_luma")),
        },
        "frame_index_mapping": {
            **frame_mapping_payload,
            "selected_local_frame_min": int(min(local_frames_selected)) if local_frames_selected else None,
            "selected_local_frame_max": int(max(local_frames_selected)) if local_frames_selected else None,
            "selected_source_frame_min": int(min(source_frames_selected)) if source_frames_selected else None,
            "selected_source_frame_max": int(max(source_frames_selected)) if source_frames_selected else None,
            "selected_source_frame_count": int(len(set(source_frames_selected))),
        },
        "rows": selected_rows,
        "missing_pynv_rows": missing_rows[: max(0, int(top_mismatches))],
        "thresholds": thresholds,
        "diff": diff,
        "timing": {
            "total_seconds": float(duration_seconds),
            "reference_read_seconds": float(reference_read_seconds),
            **decode_timing,
            "reference_rows_per_second": (
                float(len(selected_rows) / reference_read_seconds)
                if reference_read_seconds > 0
                else None
            ),
            "decoded_frames_per_second": (
                float(decode_timing["decoded_frames"] / decode_timing["decode_seconds"])
                if float(decode_timing["decode_seconds"]) > 0
                else None
            ),
        },
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Per-recording training Zarr archive.")
    parser.add_argument("--crop-run", default=None, help="Crop run override; defaults to crop_runs latest.")
    parser.add_argument("--rows", default=None, help="Comma/space separated explicit ROI rows to compare.")
    parser.add_argument("--all-rows", action="store_true", help="Compare every ROI row.")
    parser.add_argument("--sample-count", type=_non_negative_int, default=32)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--boundary-sample-count", type=_non_negative_int, default=4)
    parser.add_argument(
        "--source-frame-index-mode",
        choices=SOURCE_FRAME_INDEX_MODES,
        default="auto",
        help=(
            "How crop frame_indices map to source-video frames. Auto uses "
            "raw_video/original_frame_indices for training Zarrs when available."
        ),
    )
    parser.add_argument("--decode-chunk-frames", type=_positive_int, default=32)
    parser.add_argument("--max-abs-diff", type=int, default=0)
    parser.add_argument("--max-mean-abs-diff", type=float, default=0.0)
    parser.add_argument("--max-p95-abs-diff", type=float, default=0.0)
    parser.add_argument("--top-mismatches", type=_non_negative_int, default=10)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--json", action="store_true", help="Print full JSON report.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    payload = check_training_crop_pynvvc_pixel_parity(
        zarr_path=args.zarr_path,
        crop_run=args.crop_run,
        rows=_parse_rows(args.rows),
        all_rows=args.all_rows,
        sample_count=args.sample_count,
        seed=args.sample_seed,
        boundary_sample_count=args.boundary_sample_count,
        source_frame_index_mode=args.source_frame_index_mode,
        decode_chunk_frames=args.decode_chunk_frames,
        max_abs_diff=args.max_abs_diff,
        max_mean_abs_diff=args.max_mean_abs_diff,
        max_p95_abs_diff=args.max_p95_abs_diff,
        top_mismatches=args.top_mismatches,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    if args.json:
        print(text)
    else:
        diff = payload["diff"]
        timing = payload["timing"]
        mapping = payload["frame_index_mapping"]
        print(f"status: {payload['status']}")
        print(f"crop_run: {payload['inputs']['crop_run']}")
        print(f"frame_index_mapping: {mapping['mode']}")
        print(
            "selected_source_frames: "
            f"{mapping['selected_source_frame_min']}..{mapping['selected_source_frame_max']} "
            f"({mapping['selected_source_frame_count']} unique)"
        )
        print(f"rows_compared: {diff['rows_compared']}")
        print(f"byte_equal: {diff['byte_equal']}")
        print(f"max_abs_diff: {diff['max_abs_diff']}")
        print(f"mean_abs_diff: {float(diff['mean_abs_diff']):.6f}")
        print(f"p95_abs_diff: {float(diff['p95_abs_diff']):.6f}")
        print(
            "timing: "
            f"open={float(timing['video_open_seconds']):.3f}s "
            f"decode={float(timing['decode_seconds']):.3f}s "
            f"crop={float(timing['crop_seconds']):.3f}s "
            f"decoded_frames={timing['decoded_frames']}"
        )
        if payload["failures"]:
            print("failures:")
            for failure in payload["failures"]:
                print(f"- {failure}")
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
