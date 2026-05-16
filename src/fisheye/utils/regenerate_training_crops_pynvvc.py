"""Regenerate training crop ROI pixels from Orange mono videos via PyNvVideoCodec.

This utility creates a new materialized ``crop_runs/<target>`` group with the
same row geometry as an existing crop run, but rewrites ``roi_images`` from the
source MP4 using the PyNvVideoCodec NV12 Y/luma plane. It is intended for the
training crop-representation migration and does not change ``crop_runs/latest``
unless explicitly requested.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

from fisheye.shared.crop_roi_layout import (
    DEFAULT_CANONICAL_CROP_ROI_CHUNK_LEN,
    build_canonical_crop_roi_layout,
    build_crop_roi_create_kwargs,
    crop_roi_layout_attrs,
)
from fisheye.shared.flat_roi_cache import _crop_pynvvc_luma_frame
from fisheye.shared.roi_pixel_contract import (
    ORANGE_MONO_PYNVVC_LUMA_CONTRACT_NAME,
    orange_mono_pynvvc_luma_pixel_contract,
)


SOURCE_FRAME_INDEX_MODES = ("auto", "direct", "original_frame_indices")
MODULE_NAME = "fisheye.utils.regenerate_training_crops_pynvvc"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    raise ValueError("Unable to resolve crop run; pass --source-crop-run.")


def _default_target_run(source_crop_run: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    return f"{source_crop_run}_pynvvc_luma_{stamp}"


def _resolve_video_path(root: Any, crop_group: Any, explicit: str | Path | None) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser()
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
        raise ValueError("Unable to resolve source video path; pass --video-path.")
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
    purpose = str(root.attrs.get("zarr_purpose") or root.attrs.get("zarr_use") or "").strip().lower()
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
    if mode not in SOURCE_FRAME_INDEX_MODES:
        raise ValueError(f"Unsupported source frame index mode: {mode}")
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
        row = int(bad[0])
        raise IndexError(
            "Crop frame index is outside raw_video/original_frame_indices: "
            f"row={row}, frame_index={int(local[row])}, "
            f"mapping_length={int(original_frame_indices.shape[0])}."
        )
    mapped = original_frame_indices[local]
    return np.asarray(mapped, dtype=np.int64), {
        "mode": "original_frame_indices",
        "original_frame_indices_available": True,
        "original_frame_indices_length": int(original_frame_indices.shape[0]),
    }


def _frame_to_roi_indices(source_frame_indices: np.ndarray) -> dict[int, list[int]]:
    mapping: dict[int, list[int]] = {}
    for roi_idx, frame_idx in enumerate(np.asarray(source_frame_indices, dtype=np.int64).reshape(-1)):
        mapping.setdefault(int(frame_idx), []).append(int(roi_idx))
    return mapping


def _open_pynvvc_luma_reader(video_path: Path) -> Any:
    from fisheye.shared.pynvvc_luma_rgb import PynvvcLumaRgbReader

    return PynvvcLumaRgbReader(video_path, start_frame=0, gpu_id=0)


def _copy_source_array(source_group: Any, target_group: Any, name: str) -> None:
    source = source_group[name]
    data = np.asarray(source[:])
    chunks = getattr(source, "chunks", None)
    kwargs: dict[str, Any] = {"data": data, "overwrite": True}
    if chunks is not None:
        kwargs["chunks"] = chunks
    target_group.create_array(name, **kwargs)


def _copy_crop_arrays(source_group: Any, target_group: Any) -> list[str]:
    copied: list[str] = []
    for name in sorted(str(k) for k in source_group.array_keys()):
        if name == "roi_images":
            continue
        _copy_source_array(source_group, target_group, name)
        copied.append(name)
    return copied


def _set_latest_pointers(crop_parent: Any, target_run: str) -> None:
    crop_parent.attrs["latest"] = target_run
    crop_parent.attrs["latest_materialized"] = target_run
    crop_parent.attrs["latest_any"] = target_run


def regenerate_training_crops_pynvvc(
    *,
    zarr_path: str | Path,
    source_crop_run: str | None = None,
    target_crop_run: str | None = None,
    video_path: str | Path | None = None,
    source_frame_index_mode: str = "auto",
    decode_chunk_frames: int = 32,
    roi_chunk_len: int = DEFAULT_CANONICAL_CROP_ROI_CHUNK_LEN,
    overwrite: bool = False,
    set_latest: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    archive_path = Path(zarr_path).expanduser().resolve()
    started = time.perf_counter()
    root = zarr.open_group(str(archive_path), mode="a", use_consolidated=False)
    crop_parent = root.get("crop_runs")
    if crop_parent is None:
        raise KeyError("Zarr archive is missing crop_runs.")
    resolved_source_crop = _resolve_crop_run(root, source_crop_run)
    source_group = crop_parent[resolved_source_crop]
    if "frame_indices" not in source_group:
        raise ValueError(f"crop_runs/{resolved_source_crop} is missing frame_indices.")
    if "roi_coordinates_full" not in source_group:
        raise ValueError(f"crop_runs/{resolved_source_crop} is missing roi_coordinates_full.")

    resolved_target_crop = target_crop_run or _default_target_run(resolved_source_crop)
    if resolved_target_crop in crop_parent and not overwrite:
        raise FileExistsError(
            f"Target crop run already exists: crop_runs/{resolved_target_crop}. "
            "Pass --overwrite to replace it."
        )

    frame_indices = np.asarray(source_group["frame_indices"][:], dtype=np.int64).reshape(-1)
    roi_coordinates_full = np.asarray(source_group["roi_coordinates_full"][:], dtype=np.int32)
    roi_shape = _resolve_roi_shape(source_group)
    total_rois = int(frame_indices.shape[0])
    if int(roi_coordinates_full.shape[0]) != total_rois:
        raise ValueError(
            "roi_coordinates_full length "
            f"{roi_coordinates_full.shape[0]} does not match frame_indices rows {total_rois}."
        )
    if "roi_images" in source_group and int(source_group["roi_images"].shape[0]) != total_rois:
        raise ValueError(
            f"source roi_images rows {source_group['roi_images'].shape[0]} "
            f"do not match frame_indices rows {total_rois}."
        )

    resolved_video_path = _resolve_video_path(root, source_group, video_path)
    video_shape = _resolve_video_shape(root, source_group)
    source_frame_indices, frame_mapping = _map_source_frame_indices(
        root=root,
        crop_frame_indices=frame_indices,
        mode=source_frame_index_mode,
    )
    frame_to_rows = _frame_to_roi_indices(source_frame_indices)
    max_frame = int(max(frame_to_rows)) if frame_to_rows else -1
    contract = orange_mono_pynvvc_luma_pixel_contract()
    layout = build_canonical_crop_roi_layout(
        total_rois=total_rois,
        preferred_chunk_len=int(roi_chunk_len),
        roi_storage="compressed",
    )

    plan: dict[str, Any] = {
        "status": "dry_run" if dry_run else "planned",
        "zarr_path": str(archive_path),
        "source_crop_run": str(resolved_source_crop),
        "target_crop_run": str(resolved_target_crop),
        "video_path": str(resolved_video_path),
        "video_shape": [int(video_shape[0]), int(video_shape[1])],
        "roi_shape": [int(roi_shape[0]), int(roi_shape[1])],
        "total_rois": int(total_rois),
        "source_frame_index_mapping": frame_mapping,
        "source_frame_min": int(source_frame_indices.min()) if source_frame_indices.size else None,
        "source_frame_max": int(source_frame_indices.max()) if source_frame_indices.size else None,
        "decode_chunk_frames": int(decode_chunk_frames),
        "roi_chunk_len": int(layout.roi_chunk_len),
        "pixel_contract": contract,
        "set_latest": bool(set_latest),
    }
    if dry_run:
        return plan

    target_group = crop_parent.create_group(resolved_target_crop, overwrite=bool(overwrite))
    target_group.attrs.update(dict(source_group.attrs))
    target_group.attrs.update(
        {
            "status": "running",
            "created_at_utc": _utc_now(),
            "generated_by": MODULE_NAME,
            "source_crop_run": str(resolved_source_crop),
            "source_crop_path": f"crop_runs/{resolved_source_crop}",
            "crop_storage_mode": "materialized",
            "roi_size": [int(roi_shape[0]), int(roi_shape[1])],
            "source_video_path": str(resolved_video_path),
            "height": int(video_shape[0]),
            "width": int(video_shape[1]),
            "decode_backend": "pynvvc_luma",
            "crop_pixel_migration_version": "training_orange_mono_pynvvc_luma_v1",
            "roi_image_representation": contract.get("image_representation"),
            "roi_pixel_contract": contract,
            "source_frame_index_mapping": frame_mapping,
            "source_frame_index_mode_requested": str(source_frame_index_mode),
            "source_frame_indices_min": int(source_frame_indices.min()) if source_frame_indices.size else None,
            "source_frame_indices_max": int(source_frame_indices.max()) if source_frame_indices.size else None,
        }
    )
    target_group.attrs.update(crop_roi_layout_attrs(layout))

    copied_arrays = _copy_crop_arrays(source_group, target_group)
    target_group.create_array(
        "source_frame_indices",
        data=np.asarray(source_frame_indices, dtype=np.int64),
        chunks=(max(1, min(4096, total_rois)),),
        overwrite=True,
    )
    roi_images = target_group.create_array(
        "roi_images",
        **build_crop_roi_create_kwargs(
            total_rois=total_rois,
            roi_sz=roi_shape,
            layout=layout,
            overwrite=True,
        ),
    )

    timing: dict[str, Any] = {
        "video_open_seconds": 0.0,
        "decode_seconds": 0.0,
        "crop_seconds": 0.0,
        "contiguous_seconds": 0.0,
        "write_seconds": 0.0,
        "decoded_frames": 0,
        "skipped_frames": 0,
        "frames_with_rois": int(len(frame_to_rows)),
        "rows_written": 0,
    }
    rows_written_mask = np.zeros(total_rois, dtype=bool)
    open_started = time.perf_counter()
    reader: Any | None = None
    try:
        reader = _open_pynvvc_luma_reader(resolved_video_path)
        timing["video_open_seconds"] = float(time.perf_counter() - open_started)
        source_height = int(reader.source_height)
        source_width = int(reader.source_width)
        expected_height, expected_width = int(video_shape[0]), int(video_shape[1])
        if (source_height, source_width) != (expected_height, expected_width):
            raise ValueError(
                "PyNvVideoCodec dimensions do not match Zarr metadata: "
                f"decoder={source_width}x{source_height}, metadata={expected_width}x{expected_height}."
            )

        decoded_frame_index = 0
        decode_chunk = max(1, int(decode_chunk_frames))
        while decoded_frame_index <= max_frame:
            count = min(decode_chunk, max_frame - decoded_frame_index + 1)
            decode_started = time.perf_counter()
            frames = reader.decode_next(count)
            timing["decode_seconds"] += float(time.perf_counter() - decode_started)
            if not frames:
                break
            timing["decoded_frames"] += int(len(frames))

            for frame_offset, frame_tensor in enumerate(frames):
                frame_idx = decoded_frame_index + frame_offset
                rows = frame_to_rows.get(frame_idx)
                if not rows:
                    timing["skipped_frames"] += 1
                    continue
                crop_started = time.perf_counter()
                crops = _crop_pynvvc_luma_frame(
                    frame_tensor,
                    roi_ids=rows,
                    roi_coordinates_full=roi_coordinates_full,
                    roi_shape=roi_shape,
                    video_shape=video_shape,
                )
                try:
                    import torch

                    if torch.cuda.is_available() and getattr(crops, "is_cuda", False):
                        torch.cuda.synchronize()
                except Exception:
                    pass
                timing["crop_seconds"] += float(time.perf_counter() - crop_started)

                contiguous_started = time.perf_counter()
                crops_cpu = np.ascontiguousarray(crops.cpu().numpy(), dtype=np.uint8)
                timing["contiguous_seconds"] += float(time.perf_counter() - contiguous_started)

                write_started = time.perf_counter()
                for local_idx, row in enumerate(rows):
                    roi_images[int(row)] = crops_cpu[int(local_idx)]
                    rows_written_mask[int(row)] = True
                timing["write_seconds"] += float(time.perf_counter() - write_started)
                timing["rows_written"] = int(rows_written_mask.sum())

            decoded_frame_index += int(len(frames))

        rows_written = int(rows_written_mask.sum())
        if rows_written != total_rois:
            missing = int(total_rois - rows_written)
            raise RuntimeError(
                f"PyNvVideoCodec crop regeneration wrote {rows_written}/{total_rois} rows; "
                f"{missing} rows were not produced before decoder EOF."
            )

        duration = float(time.perf_counter() - started)
        timing["total_seconds"] = duration
        timing["decode_fps"] = (
            float(timing["decoded_frames"]) / float(timing["decode_seconds"])
            if float(timing["decode_seconds"]) > 0
            else None
        )
        timing["rows_per_second"] = float(total_rois) / duration if duration > 0 else None
        target_group.attrs["status"] = "completed"
        target_group.attrs["completed_at_utc"] = _utc_now()
        target_group.attrs["duration_seconds"] = duration
        target_group.attrs["summary_statistics"] = {
            "total_rois_cropped": int(total_rois),
            "roi_size": [int(roi_shape[0]), int(roi_shape[1])],
            "roi_pixels_materialized": True,
            "source_crop_run": str(resolved_source_crop),
            "pixel_contract_name": ORANGE_MONO_PYNVVC_LUMA_CONTRACT_NAME,
        }
        target_group.attrs["timing"] = _json_safe(timing)
        if set_latest:
            _set_latest_pointers(crop_parent, resolved_target_crop)

        return {
            **plan,
            "status": "ok",
            "copied_arrays": copied_arrays,
            "timing": _json_safe(timing),
            "host": socket.gethostname(),
            "pid": int(os.getpid()),
        }
    except Exception as exc:
        target_group.attrs["status"] = "failed"
        target_group.attrs["failed_at_utc"] = _utc_now()
        target_group.attrs["error_message"] = str(exc)
        raise
    finally:
        if reader is not None:
            reader.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Regenerate a training crop run's roi_images from PyNvVideoCodec luma."
    )
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument("--source-crop-run", help="Existing crop_runs/<run> to copy geometry from.")
    parser.add_argument("--target-crop-run", help="New crop_runs/<run> name to create.")
    parser.add_argument("--video-path", type=Path, help="Source MP4 path. Overrides zarr metadata.")
    parser.add_argument(
        "--source-frame-index-mode",
        choices=SOURCE_FRAME_INDEX_MODES,
        default="auto",
        help="How crop frame_indices map to source-video frame numbers.",
    )
    parser.add_argument("--decode-chunk-frames", type=int, default=32)
    parser.add_argument("--roi-chunk-len", type=int, default=DEFAULT_CANONICAL_CROP_ROI_CHUNK_LEN)
    parser.add_argument("--overwrite", action="store_true", help="Replace target crop run if it already exists.")
    parser.add_argument(
        "--set-latest",
        action="store_true",
        help="Update crop_runs/latest, latest_materialized, and latest_any to the new run.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Resolve inputs and print a plan without writing.")
    parser.add_argument("--output-json", type=Path, help="Write the report JSON to this path.")
    parser.add_argument("--json", action="store_true", help="Print the full JSON report.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = regenerate_training_crops_pynvvc(
        zarr_path=args.zarr_path,
        source_crop_run=args.source_crop_run,
        target_crop_run=args.target_crop_run,
        video_path=args.video_path,
        source_frame_index_mode=args.source_frame_index_mode,
        decode_chunk_frames=args.decode_chunk_frames,
        roi_chunk_len=args.roi_chunk_len,
        overwrite=args.overwrite,
        set_latest=args.set_latest,
        dry_run=args.dry_run,
    )
    text = json.dumps(_json_safe(report), indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    if args.json or args.output_json is None:
        print(text)
    else:
        print(
            f"status: {report['status']}\n"
            f"source_crop_run: {report['source_crop_run']}\n"
            f"target_crop_run: {report['target_crop_run']}\n"
            f"total_rois: {report['total_rois']}\n"
            f"pixel_contract: {ORANGE_MONO_PYNVVC_LUMA_CONTRACT_NAME}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
