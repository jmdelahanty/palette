#!/usr/bin/env python3
"""Create sampled training Zarrs from source video using PyNvVideoCodec luma."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import zarr

from fisheye.shared.pynvvc_luma_rgb import PynvvcLumaRgbReader
from fisheye.shared.roi_pixel_contract import (
    ORANGE_MONO_PYNVVC_LUMA_CONTRACT_NAME,
    orange_mono_pynvvc_luma_pixel_contract,
)


PYNVVC_LUMA_DECODE_BACKEND = "pynvvc_luma"
RAW_VIDEO_PIXEL_CONTRACT_SCHEMA = "palette_raw_video_pixel_contract_v1"


@dataclass(frozen=True)
class SampledTrainingImportResult:
    zarr_path: Path
    source_video_path: Path
    imported_frame_count: int
    source_frame_count: int
    frame_step: int
    skip_tail_frames: int
    original_resolution: tuple[int, int]
    downsampled_resolution: tuple[int, int] | None
    decode_backend: str
    duration_s: float


def _compute_frame_indices(total_frames: int, frame_step: int, *, skip_tail_frames: int) -> list[int]:
    if total_frames <= 0:
        raise ValueError(f"source_frame_count must be positive, got {total_frames}")
    if frame_step < 1:
        raise ValueError(f"frame_step must be >= 1, got {frame_step}")
    if skip_tail_frames < 0:
        raise ValueError(f"skip_tail_frames must be >= 0, got {skip_tail_frames}")
    effective_total = int(total_frames) - int(skip_tail_frames)
    if effective_total <= 0:
        raise ValueError(
            f"source_frame_count ({total_frames}) must be greater than skip_tail_frames ({skip_tail_frames})"
        )
    return list(range(0, effective_total, int(frame_step)))


def _load_import_config(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    with config_path.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}
    return payload if isinstance(payload, dict) else {}


def _downsample_config(config: dict[str, Any]) -> tuple[bool, tuple[int, int], str, bool, int]:
    import_cfg = config.get("import") if isinstance(config.get("import"), dict) else {}
    down_cfg = import_cfg.get("downsampled") if isinstance(import_cfg.get("downsampled"), dict) else {}
    resolutions = str(import_cfg.get("resolutions", "both"))
    create_downsampled = resolutions in {"both", "downsampled"}
    raw_size = down_cfg.get("size", [640, 640])
    if not isinstance(raw_size, (list, tuple)) or len(raw_size) != 2:
        raw_size = [640, 640]
    target_hw = (int(raw_size[0]), int(raw_size[1]))
    method = str(down_cfg.get("method", "area"))
    preserve_aspect = bool(down_cfg.get("preserve_aspect", False))
    chunk_size = int(down_cfg.get("chunk_size", import_cfg.get("chunk_size", 32)))
    return create_downsampled, target_hw, method, preserve_aspect, max(1, chunk_size)


def _chunk_size(config: dict[str, Any]) -> int:
    import_cfg = config.get("import") if isinstance(config.get("import"), dict) else {}
    return max(1, int(import_cfg.get("chunk_size", 32)))


def _compute_letterbox_dims(
    source_h: int,
    source_w: int,
    target_h: int,
    target_w: int,
) -> tuple[int, int, int, int, int, int]:
    scale = min(target_h / source_h, target_w / source_w)
    resized_h = max(1, min(target_h, int(round(source_h * scale))))
    resized_w = max(1, min(target_w, int(round(source_w * scale))))
    pad_top = (target_h - resized_h) // 2
    pad_bottom = target_h - resized_h - pad_top
    pad_left = (target_w - resized_w) // 2
    pad_right = target_w - resized_w - pad_left
    return resized_h, resized_w, pad_top, pad_bottom, pad_left, pad_right


def _resize_luma(
    luma_hw: torch.Tensor,
    *,
    target_hw: tuple[int, int],
    method: str,
    preserve_aspect: bool,
) -> torch.Tensor:
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    mode = str(method)
    align_corners = False if mode in {"bilinear", "bicubic"} else None
    batch = luma_hw.unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
    if preserve_aspect:
        source_h, source_w = int(luma_hw.shape[-2]), int(luma_hw.shape[-1])
        resized_h, resized_w, pad_top, pad_bottom, pad_left, pad_right = _compute_letterbox_dims(
            source_h,
            source_w,
            target_h,
            target_w,
        )
        resized = F.interpolate(batch, size=(resized_h, resized_w), mode=mode, align_corners=align_corners)
        if any((pad_top, pad_bottom, pad_left, pad_right)):
            resized = F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0)
    else:
        resized = F.interpolate(batch, size=(target_h, target_w), mode=mode, align_corners=align_corners)
    return resized.squeeze(0).squeeze(0).clamp(0, 255).to(dtype=torch.uint8).contiguous()


def _raw_video_pixel_contract() -> dict[str, Any]:
    roi_contract = orange_mono_pynvvc_luma_pixel_contract()
    return {
        "schema": RAW_VIDEO_PIXEL_CONTRACT_SCHEMA,
        "name": ORANGE_MONO_PYNVVC_LUMA_CONTRACT_NAME,
        "image_representation": "uint8_luma_frame_v1",
        "shape": "[frame, height, width]",
        "dtype": "uint8",
        "order": "C",
        "source_frame_representation": roi_contract.get("source_frame_representation"),
        "color_conversion": roi_contract.get("color_conversion"),
        "production_status": "canonical_sampled_training_candidate",
    }


def _json_attr(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _safe_read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _manifest_recording_attrs(recording_dir: Path | None) -> dict[str, Any]:
    if recording_dir is None:
        return {}
    manifest = _safe_read_json(recording_dir / "recording_manifest.json")
    if not manifest:
        return {}

    keys = (
        "recording_id",
        "session_uuid",
        "session_id",
        "recording_name",
        "recording_type",
        "recording_subtype",
        "behavior_mode",
        "artifact_schema_id",
        "session_start_iso8601_utc",
        "rig_id",
        "arena_id",
        "camera_id",
        "canvas_name",
        "protocol_name",
        "protocol_name_from_definition",
        "dish_design",
    )
    attrs: dict[str, Any] = {}
    for key in keys:
        value = manifest.get(key)
        if value is None:
            continue
        text = str(value).strip() if not isinstance(value, (dict, list)) else ""
        if text:
            attrs[key] = value
    attrs["recording_manifest_path"] = str((recording_dir / "recording_manifest.json").resolve())
    return attrs


def _safe_replace_temp(temp_path: Path, output_path: Path, *, overwrite: bool) -> None:
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output Zarr already exists: {output_path}")
        shutil.rmtree(output_path)
    temp_path.rename(output_path)


def _write_attrs(
    root: Any,
    raw: Any,
    *,
    source_video_path: Path,
    source_frame_count: int,
    frame_indices: list[int],
    frame_step: int,
    skip_tail_frames: int,
    original_hw: tuple[int, int],
    downsampled_hw: tuple[int, int] | None,
    downsample_method: str,
    downsample_preserve_aspect: bool,
    camera_id: str | None,
    recording_dir: Path | None,
    h5_path: Path | None,
    gpu_id: int,
    created_at_utc: str,
    duration_s: float | None = None,
) -> None:
    raw_contract = _raw_video_pixel_contract()
    roi_contract = orange_mono_pynvvc_luma_pixel_contract()
    root_attrs = {
        "zarr_purpose": "training",
        "zarr_use": "training",
        "created_at_utc": created_at_utc,
        "source_video_path": str(source_video_path),
        "source_video_width": int(original_hw[1]),
        "source_video_height": int(original_hw[0]),
        "video_width": int(original_hw[1]),
        "video_height": int(original_hw[0]),
    }
    root_attrs.update(_manifest_recording_attrs(recording_dir))
    root.attrs.update(root_attrs)
    raw_attrs: dict[str, Any] = {
        "import_method": "pynvvc_luma_sampled_training",
        "import_stage": "complete",
        "import_mode": "sampled",
        "import_purpose": "training_data",
        "device": f"cuda:{int(gpu_id)}",
        "decode_backend": PYNVVC_LUMA_DECODE_BACKEND,
        "decode_backend_family": "PyNvVideoCodec",
        "decode_contract_status": "pynvvc_luma_canonical_candidate",
        "source_decode_surface": "nv12_y_plane_uint8",
        "pixel_contract_name": ORANGE_MONO_PYNVVC_LUMA_CONTRACT_NAME,
        "pixel_contract": _json_attr(raw_contract),
        "roi_pixel_contract_name": ORANGE_MONO_PYNVVC_LUMA_CONTRACT_NAME,
        "roi_pixel_contract": _json_attr(roi_contract),
        "color_range": "tv",
        "color_space": "bt709_or_source_unspecified_monochrome_luma",
        "color_matrix": "source_encoded_nv12_y_plane",
        "color_transfer": "source_encoded",
        "color_primaries": "source_encoded",
        "stored_luma_transform": "raw_nv12_y_plane_no_rgb_reconstruction",
        "stored_luma_color_range": "tv_limited_range_y_plane",
        "source_video": source_video_path.name,
        "source_path": str(source_video_path.resolve()),
        "source_video_path": str(source_video_path),
        "source_video_width": int(original_hw[1]),
        "source_video_height": int(original_hw[0]),
        "video_width": int(original_hw[1]),
        "video_height": int(original_hw[0]),
        "source_frame_count": int(source_frame_count),
        "original_video_length": int(source_frame_count),
        "effective_video_length": int(source_frame_count) - int(skip_tail_frames),
        "skip_tail_frames": int(skip_tail_frames),
        "frame_step": int(frame_step),
        "total_frames": len(frame_indices),
        "imported_frame_count": len(frame_indices),
        "original_resolution": [int(original_hw[0]), int(original_hw[1])],
        "has_full_resolution": True,
        "has_downsampled": downsampled_hw is not None,
        "import_timestamp": created_at_utc,
    }
    if downsampled_hw is not None:
        raw_attrs.update(
            {
                "downsampled_resolution": [int(downsampled_hw[0]), int(downsampled_hw[1])],
                "downsample_method": str(downsample_method),
                "downsample_preserve_aspect": bool(downsample_preserve_aspect),
                "downsample_formats": ["gray"],
                "downsampled_shapes": {"images_ds": [int(downsampled_hw[0]), int(downsampled_hw[1])]},
            }
        )
    if camera_id:
        raw_attrs["camera_id"] = str(camera_id)
    if recording_dir is not None:
        raw_attrs["recording_dir"] = str(recording_dir)
    if h5_path is not None:
        raw_attrs["source_h5_path"] = str(h5_path)
    if duration_s is not None:
        raw_attrs["import_duration_seconds"] = float(duration_s)
    raw.attrs.update(raw_attrs)


def import_sampled_training_pynvvc(
    *,
    video_path: Path,
    zarr_path: Path,
    source_frame_count: int,
    frame_step: int,
    skip_tail_frames: int = 0,
    config_path: Path | None = None,
    overwrite: bool = False,
    camera_id: str | None = None,
    recording_dir: Path | None = None,
    h5_path: Path | None = None,
    gpu_id: int = 0,
    require_cuda: bool = True,
    reader_factory: Callable[..., Any] = PynvvcLumaRgbReader,
) -> SampledTrainingImportResult:
    """Write a sampled training Zarr using sequential PyNvVC luma decode.

    The final path is created by renaming a sibling temp Zarr only after the
    decode and writes complete. This avoids leaving a partially-created final
    training Zarr on GPU/decode failures.
    """

    video_path = video_path.expanduser().resolve()
    zarr_path = zarr_path.expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Source video not found: {video_path}")
    if zarr_path.exists() and not overwrite:
        raise FileExistsError(f"Output Zarr already exists: {zarr_path}")
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "PyNvVC sampled training import requires CUDA-enabled torch; "
            "current environment reports torch.cuda.is_available() == False."
        )

    frame_indices = _compute_frame_indices(
        int(source_frame_count),
        int(frame_step),
        skip_tail_frames=int(skip_tail_frames),
    )
    if not frame_indices:
        raise ValueError("No sampled frames selected")

    # Fail fast on PyNvVC/NVDEC before creating any Zarr paths.
    reader = reader_factory(video_path, start_frame=0, gpu_id=int(gpu_id))
    source_h = int(reader.source_height)
    source_w = int(reader.source_width)
    original_hw = (source_h, source_w)
    config = _load_import_config(config_path)
    full_chunk_size = _chunk_size(config)
    write_batch_size = full_chunk_size
    create_downsampled, down_hw, down_method, down_preserve, down_chunk_size = _downsample_config(config)
    downsampled_hw = down_hw if create_downsampled else None

    temp_path = zarr_path.with_name(f"{zarr_path.name}.__pynvvc_tmp_{os.getpid()}")
    if temp_path.exists():
        shutil.rmtree(temp_path)
    if zarr_path.exists() and overwrite:
        # Do not remove the final output until the temp output has been built.
        pass

    start = time.perf_counter()
    created_at = datetime.now(timezone.utc).isoformat()
    selected = set(frame_indices)
    next_max = max(frame_indices)
    out_row = 0
    batch_start = 0
    full_batch: list[np.ndarray] = []
    ds_batch: list[np.ndarray] = []

    try:
        root = zarr.open_group(str(temp_path), mode="w", zarr_format=3)
        raw = root.create_group("raw_video", overwrite=True)
        images_full = raw.create_array(
            "images_full",
            shape=(len(frame_indices), source_h, source_w),
            chunks=(min(full_chunk_size, len(frame_indices)), source_h, source_w),
            dtype="uint8",
            fill_value=0,
            compressors=[],
            overwrite=True,
        )
        images_full.attrs.update({"format": "gray", "resolution": [source_h, source_w]})
        images_ds = None
        if downsampled_hw is not None:
            ds_h, ds_w = downsampled_hw
            images_ds = raw.create_array(
                "images_ds",
                shape=(len(frame_indices), ds_h, ds_w),
                chunks=(min(down_chunk_size, len(frame_indices)), ds_h, ds_w),
                dtype="uint8",
                fill_value=0,
                compressors=[],
                overwrite=True,
            )
            images_ds.attrs.update({"format": "gray", "resolution": [ds_h, ds_w]})
        raw.create_array(
            "original_frame_indices",
            data=np.asarray(frame_indices, dtype=np.int32),
            chunks=(min(1000, len(frame_indices)),),
            overwrite=True,
        )
        raw.create_array(
            "timestamps",
            shape=(len(frame_indices),),
            chunks=(min(1000, len(frame_indices)),),
            dtype="float64",
            fill_value=float("nan"),
            compressors=[],
            overwrite=True,
        )
        _write_attrs(
            root,
            raw,
            source_video_path=video_path,
            source_frame_count=int(source_frame_count),
            frame_indices=frame_indices,
            frame_step=int(frame_step),
            skip_tail_frames=int(skip_tail_frames),
            original_hw=original_hw,
            downsampled_hw=downsampled_hw,
            downsample_method=down_method,
            downsample_preserve_aspect=down_preserve,
            camera_id=camera_id,
            recording_dir=recording_dir,
            h5_path=h5_path,
            gpu_id=int(gpu_id),
            created_at_utc=created_at,
        )

        def flush() -> None:
            nonlocal batch_start, full_batch, ds_batch
            if not full_batch:
                return
            stop = batch_start + len(full_batch)
            images_full[batch_start:stop] = np.stack(full_batch, axis=0)
            if images_ds is not None:
                images_ds[batch_start:stop] = np.stack(ds_batch, axis=0)
            batch_start = stop
            full_batch = []
            ds_batch = []

        with torch.no_grad():
            for frame_idx, frame in enumerate(reader.iter_frames()):
                if frame_idx > next_max:
                    break
                if frame_idx not in selected:
                    continue
                luma = frame[:source_h, :source_w].contiguous()
                full_batch.append(luma.to("cpu").numpy().copy())
                if images_ds is not None:
                    ds = _resize_luma(
                        luma,
                        target_hw=downsampled_hw or (source_h, source_w),
                        method=down_method,
                        preserve_aspect=down_preserve,
                    )
                    ds_batch.append(ds.to("cpu").numpy().copy())
                out_row += 1
                if len(full_batch) >= write_batch_size:
                    flush()
            flush()
        if out_row != len(frame_indices):
            raise RuntimeError(
                f"Decoded {out_row} selected frames, expected {len(frame_indices)}. "
                f"Video ended before frame {next_max}."
            )
        duration = time.perf_counter() - start
        _write_attrs(
            root,
            raw,
            source_video_path=video_path,
            source_frame_count=int(source_frame_count),
            frame_indices=frame_indices,
            frame_step=int(frame_step),
            skip_tail_frames=int(skip_tail_frames),
            original_hw=original_hw,
            downsampled_hw=downsampled_hw,
            downsample_method=down_method,
            downsample_preserve_aspect=down_preserve,
            camera_id=camera_id,
            recording_dir=recording_dir,
            h5_path=h5_path,
            gpu_id=int(gpu_id),
            created_at_utc=created_at,
            duration_s=duration,
        )
        reader.close()
        _safe_replace_temp(temp_path, zarr_path, overwrite=overwrite)
        return SampledTrainingImportResult(
            zarr_path=zarr_path,
            source_video_path=video_path,
            imported_frame_count=len(frame_indices),
            source_frame_count=int(source_frame_count),
            frame_step=int(frame_step),
            skip_tail_frames=int(skip_tail_frames),
            original_resolution=original_hw,
            downsampled_resolution=downsampled_hw,
            decode_backend=PYNVVC_LUMA_DECODE_BACKEND,
            duration_s=duration,
        )
    except Exception:
        try:
            reader.close()
        finally:
            if temp_path.exists():
                shutil.rmtree(temp_path, ignore_errors=True)
        raise


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import a sampled training Zarr using PyNvVC luma decode.")
    parser.add_argument("video_path", type=Path)
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument("--source-frame-count", type=int, required=True)
    parser.add_argument("--frame-step", type=int, required=True)
    parser.add_argument("--skip-tail-frames", type=int, default=0)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--camera-id")
    parser.add_argument("--recording-dir", type=Path)
    parser.add_argument("--h5-path", type=Path)
    parser.add_argument("--gpu-id", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    result = import_sampled_training_pynvvc(
        video_path=args.video_path,
        zarr_path=args.zarr_path,
        source_frame_count=int(args.source_frame_count),
        frame_step=int(args.frame_step),
        skip_tail_frames=int(args.skip_tail_frames),
        config_path=args.config,
        overwrite=bool(args.overwrite),
        camera_id=args.camera_id,
        recording_dir=args.recording_dir,
        h5_path=args.h5_path,
        gpu_id=int(args.gpu_id),
    )
    print(json.dumps({
        "zarr_path": str(result.zarr_path),
        "source_video_path": str(result.source_video_path),
        "imported_frame_count": result.imported_frame_count,
        "source_frame_count": result.source_frame_count,
        "frame_step": result.frame_step,
        "skip_tail_frames": result.skip_tail_frames,
        "original_resolution": list(result.original_resolution),
        "downsampled_resolution": list(result.downsampled_resolution) if result.downsampled_resolution else None,
        "decode_backend": result.decode_backend,
        "duration_s": result.duration_s,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
