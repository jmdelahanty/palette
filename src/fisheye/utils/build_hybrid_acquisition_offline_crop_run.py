"""Build reviewed hybrid crop runs backed by crop video plus offline ROI cache.

The output crop run is geometry-only.  The production route joins the canonical
raw acquisition crop ledger to reviewed detections by recording frame.  Live
detection identities and boxes remain evidence only; the reviewed refined rowset
is the sole analysis-instance and bbox authority.  An explicit legacy route
retains the historical instance-key join for already-published compatibility
artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import zarr

from fisheye.shared.acquisition_crop_stream_ledger import (
    ACQUISITION_CROP_LEDGER_RUNS_GROUP,
    validate_current_acquisition_crop_stream_ledger,
)
from fisheye.shared.batch_logging import utc_now as _utc_now
from fisheye.shared.composite_crop import assert_crop_run_unreferenced
from fisheye.shared.crop_defaults import DEFAULT_ZEBRAFISH_CROP_SIZE_PX
from fisheye.shared.crop_geometry import (
    compute_centered_roi_mapping,
    resolve_full_frame_shape,
)
from fisheye.shared.flat_roi_cache import FLAT_ROI_CACHE_LAYOUT, FLAT_ROI_CACHE_SCHEMA
from fisheye.shared.flat_roi_cache import _crop_pynvvc_luma_frame  # noqa: PLC2701
from fisheye.shared.hybrid_crop_provider import (
    HYBRID_CROP_RUN_SCHEMA_ID,
    build_hybrid_crop_provider_identity,
)
from fisheye.shared.refined_detect_curation import (
    extract_present_curated_rows,
    resolve_curated_refined_detect_run,
)
from fisheye.shared.roi_pixel_contract import (
    SOURCE_PIXELS_RAW_CAMERA_VIDEO,
    orange_mono_pynvvc_luma_hybrid_pixel_contract,
    orange_mono_pynvvc_luma_pixel_contract,
)
from fisheye.shared.run_provenance import build_run_provenance_from_stage_record
from fisheye.shared.stage_provenance import (
    build_stage_provenance,
    write_stage_provenance,
)
from fisheye.shared.zarr.chunk_profiles import (
    create_geometry_preload_array,
    stamp_geometry_preload_attrs,
)
from fisheye.shared.zarr.detection_schema import derive_canonical_detection_geometry
from fisheye.shared.zarr.refined_detection_manifest import (
    REFINED_DETECTION_AUTHORITY_PROVENANCE_ATTRIBUTE,
    REFINED_DETECTION_AUTHORITY_RUN_ATTRIBUTE,
    validate_refined_detection_authority_provenance,
)
from fisheye.shared.zarr.refined_detection_crop_source import (
    bind_refined_detection_crop_source,
)
from fisheye.shared.zarr.refined_detection_manifest import (
    REFINED_DETECTION_RUN_MANIFEST_ATTRIBUTE,
)
from fisheye.shared.zarr_helpers import consolidate_metadata_capture_expected_warnings
from fisheye.shared.zarr_run_completion import (
    mark_run_complete,
    mark_run_failed,
    mark_run_started,
)
from fisheye.shared.system_metadata import get_environment_info, get_git_info

SCHEMA_ID = HYBRID_CROP_RUN_SCHEMA_ID
DEFAULT_RUN_PREFIX = "crop_hybrid_acquisition_offline"
DECODE_MODES = ("auto", "indexed", "sequential")
ACQUISITION_SOURCE_MODES = ("canonical_ledger", "legacy_crop_run")
PRODUCTION_ROUTING_POLICY_ID = "goodbatbadbat_crop_pixel_routing_v1"
PRODUCTION_CROP_POLICY_ID = "zebrafish_crop_384_v1"
PRODUCTION_CONTEXT_MARGIN_PX = 0.0
SOURCE_PIXEL_KIND_CODE_MAP = {
    "acquisition_crop_video": 0,
    "offline_full_frame_supplemental_flat_cache": 1,
}
CROP_STATE_CODE_MAP = {
    "reviewed_acquisition_crop_reused": 0,
    "reviewed_supplemental_crop_materialized": 1,
}
DETECTION_SOURCE_CODE_MAP = {
    "reviewed_refined_detection": 1,
}
ROUTING_REASON_CODE_MAP = {
    "acquisition_crop_selected": 0,
    "blank_acquisition_crop": 1,
    "acquisition_no_detection": 2,
    "acquisition_crop_missing": 3,
    "canonical_roi_not_contained": 4,
    "frame_identity_mismatch": 5,
    "coordinate_or_extent_mismatch": 6,
    "legacy_instance_key_not_matched": 7,
}


def _utc_run_name(prefix: str = DEFAULT_RUN_PREFIX) -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        _json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _resolve_canonical_crop_ledger(
    root: zarr.Group,
    *,
    expected_record_sha256: str | None,
) -> tuple[zarr.Group, str, str]:
    publication = validate_current_acquisition_crop_stream_ledger(
        root,
        expected_record_sha256=expected_record_sha256,
    )
    stream = root["analysis/acquisition_video_streams/streams/crop"]
    run = stream[ACQUISITION_CROP_LEDGER_RUNS_GROUP][publication.run_name]
    return run, publication.run_name, publication.record_sha256


def _ledger_crop_video_path(ledger_group: zarr.Group) -> Path:
    raw = str(ledger_group.attrs.get("source_video_path") or "").strip()
    if not raw:
        raise ValueError("Canonical acquisition crop ledger lacks source_video_path.")
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Canonical acquisition crop video not found: {path}")
    return path


def _ledger_roi_shape(ledger_group: zarr.Group) -> tuple[int, int]:
    contract = ledger_group.attrs.get("source_stream_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("Canonical acquisition crop ledger lacks source_stream_contract.")
    try:
        width = int(contract.get("width"))
        height = int(contract.get("height"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Canonical crop stream width/height are not integers.") from exc
    if width <= 0 or height <= 0:
        raise ValueError("Canonical crop stream width/height must be positive.")
    return height, width


def _bound_stat_media_identity(
    *,
    path: Path,
    attrs: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    strategy = str(attrs.get("source_video_fingerprint_strategy") or "").strip()
    fingerprint = str(attrs.get("source_video_fingerprint") or "").strip()
    try:
        declared_size = int(attrs.get("source_video_size_bytes"))
        declared_mtime = int(attrs.get("source_video_mtime_ns"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} lacks its declared stat media identity.") from exc
    if strategy != "stat_v1" or len(fingerprint) != 64:
        raise ValueError(f"{label} lacks a valid stat_v1 media fingerprint.")
    observed = path.stat()
    if (
        int(observed.st_size) != declared_size
        or int(observed.st_mtime_ns) != declared_mtime
    ):
        raise ValueError(f"{label} changed after its declared media fingerprint.")
    return {
        "path": str(path),
        "strategy": strategy,
        "fingerprint": fingerprint,
        "size_bytes": declared_size,
        "mtime_ns": declared_mtime,
    }


def _create_array(group: zarr.Group, name: str, data: np.ndarray) -> None:
    if name in group:
        del group[name]
    create_geometry_preload_array(group, name, data=np.asarray(data), overwrite=True)


def _resolve_crop_run(
    root: zarr.Group, crop_run: str | None
) -> tuple[zarr.Group, zarr.Group, str]:
    parent = root.get("crop_runs")
    if parent is None:
        raise ValueError("Zarr archive is missing crop_runs.")
    if crop_run:
        if crop_run not in parent:
            raise ValueError(f"crop_runs/{crop_run} not found.")
        return parent, parent[crop_run], str(crop_run)
    for attr_name in ("latest_any", "latest", "latest_materialized"):
        candidate = parent.attrs.get(attr_name)
        if candidate and str(candidate) in parent:
            return parent, parent[str(candidate)], str(candidate)
    raise ValueError("No crop run specified and crop_runs has no latest pointer.")


def _resolve_recording_dir(zarr_path: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    if zarr_path.parent.name == "zarr":
        return zarr_path.parent.parent.resolve()
    for parent in zarr_path.parents:
        if (parent / "recording_manifest.json").exists():
            return parent.resolve()
    return zarr_path.parent.resolve()


def _resolve_source_video_path(
    *,
    root: zarr.Group,
    recording_dir: Path,
    explicit: Path | None,
) -> Path:
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Source video path is not a file: {path}")
        return path

    cams_dir = recording_dir / "cams"
    if cams_dir.is_dir():
        mp4s = sorted(cams_dir.glob("*.mp4"))
        if len(mp4s) == 1:
            return mp4s[0].resolve()
        if len(mp4s) > 1:
            raise ValueError(
                f"Multiple camera videos found under {cams_dir}; pass --source-video-path."
            )

    for attr_name in ("source_video_path", "video_path", "source_video"):
        value = root.attrs.get(attr_name)
        if not value:
            continue
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = recording_dir / path
        if path.is_file():
            return path.resolve()

    raise FileNotFoundError(
        "Unable to resolve full camera source video. Pass --source-video-path or ensure cams/*.mp4 exists."
    )


def _resolve_roi_shape(crop_group: zarr.Group) -> tuple[int, int]:
    roi_size = crop_group.attrs.get("roi_size")
    if isinstance(roi_size, (list, tuple)) and len(roi_size) == 2:
        return int(roi_size[0]), int(roi_size[1])
    if "roi_sizes_full" in crop_group:
        sizes = np.asarray(crop_group["roi_sizes_full"][:], dtype=np.int32).reshape(
            -1, 2
        )
        valid = np.logical_and(sizes[:, 0] > 0, sizes[:, 1] > 0)
        unique = np.unique(sizes[valid], axis=0)
        if unique.shape[0] == 1:
            width, height = int(unique[0, 0]), int(unique[0, 1])
            return height, width
    raise ValueError("Unable to resolve fixed ROI shape from acquisition crop run.")


def _read_array_or_default(
    group: zarr.Group,
    name: str,
    *,
    rows: int,
    shape_suffix: tuple[int, ...] = (),
    dtype: Any,
    fill: Any,
) -> np.ndarray:
    arr = group.get(name)
    if arr is not None:
        return np.asarray(arr[:], dtype=dtype)
    shape = (int(rows), *shape_suffix)
    return np.full(shape, fill, dtype=dtype)


def _bbox_roi_xyxy(
    bbox_img_xyxy: np.ndarray, roi_coordinates_full: np.ndarray
) -> np.ndarray:
    bbox = np.asarray(bbox_img_xyxy, dtype=np.float64).reshape(-1, 4)
    offsets = np.asarray(roi_coordinates_full, dtype=np.float64).reshape(-1, 2)
    out = bbox.copy()
    out[:, [0, 2]] -= offsets[:, 0:1]
    out[:, [1, 3]] -= offsets[:, 1:2]
    return out


def _bbox_crop_norm_xywh(
    bbox_roi_xyxy: np.ndarray, roi_shape: tuple[int, int]
) -> np.ndarray:
    roi_h, roi_w = int(roi_shape[0]), int(roi_shape[1])
    bbox = np.asarray(bbox_roi_xyxy, dtype=np.float64).reshape(-1, 4)
    out = np.empty_like(bbox, dtype=np.float64)
    out[:, 0] = ((bbox[:, 0] + bbox[:, 2]) * 0.5) / float(roi_w)
    out[:, 1] = ((bbox[:, 1] + bbox[:, 3]) * 0.5) / float(roi_h)
    out[:, 2] = (bbox[:, 2] - bbox[:, 0]) / float(roi_w)
    out[:, 3] = (bbox[:, 3] - bbox[:, 1]) / float(roi_h)
    return out


def _require_unique_instance_keys(
    group: zarr.Group,
    *,
    rows: int,
    label: str,
) -> np.ndarray:
    node = group.get("instance_key")
    if node is None:
        raise ValueError(
            f"{label} requires stable instance_key values for reviewed hybrid routing."
        )
    keys = np.asarray(node[:])
    if keys.shape != (int(rows),) or np.dtype(keys.dtype) != np.dtype(np.uint64):
        raise ValueError(
            f"{label} instance_key must have exact uint64 shape ({int(rows)},)."
        )
    if np.unique(keys).shape[0] != int(rows):
        raise ValueError(f"{label} instance_key values must be unique.")
    return np.asarray(keys, dtype=np.uint64)


def _require_approved_refined_input(
    root: zarr.Group,
    refined_group: zarr.Group,
    *,
    run_name: str,
) -> str:
    review = refined_group.attrs.get("detect_review_status")
    if isinstance(review, Mapping):
        state = str(review.get("state") or "").strip().lower()
        intended_use = str(review.get("intended_use") or "").strip().lower()
        if state == "approved" and intended_use in {
            "analysis",
            "training",
            "analysis_and_training",
        }:
            return "run_local_approved_review"

    parent = root.get("refined_detect_runs")
    if parent is not None:
        selected = str(
            parent.attrs.get(REFINED_DETECTION_AUTHORITY_RUN_ATTRIBUTE) or ""
        ).strip()
        authority = parent.attrs.get(
            REFINED_DETECTION_AUTHORITY_PROVENANCE_ATTRIBUTE
        )
        if selected == str(run_name) and isinstance(authority, Mapping):
            errors = validate_refined_detection_authority_provenance(authority)
            intended_use = (
                str(authority.get("payload", {}).get("intended_use") or "")
                .strip()
                .lower()
                if isinstance(authority.get("payload"), Mapping)
                else ""
            )
            if not errors and intended_use in {"analysis", "analysis_and_training"}:
                return "approved_authoritative_refined_v1"

    raise ValueError(
        "Reviewed hybrid crop publication requires an approved refined-detection "
        "input authorized for analysis or training."
    )


def _resolve_refined_input(
    *,
    archive_path: Path,
    root: zarr.Group,
    run_name: str | None,
) -> tuple[zarr.Group, str, str, Mapping[str, np.ndarray], str | None, str | None]:
    """Resolve legacy curated or strict finalized refined authority."""

    parent = root.get("refined_detect_runs")
    explicit = str(run_name or "").strip()
    candidate = parent.get(explicit) if parent is not None and explicit else None
    strict_manifest = (
        candidate.attrs.get(REFINED_DETECTION_RUN_MANIFEST_ATTRIBUTE)
        if candidate is not None
        else None
    )
    if isinstance(strict_manifest, Mapping):
        bound = bind_refined_detection_crop_source(
            archive_path,
            run_id=explicit,
            allow_selector_ineligible_benchmark=True,
        )
        attrs = bound.run_group.attrs
        gate = attrs.get("registered_detection_gate")
        if (
            attrs.get("finalized_recording_authority") is not True
            or attrs.get("immutable_snapshot") is not True
            or attrs.get("status") != "complete"
        ):
            raise ValueError(
                "Strict refined crop input must be a complete finalized immutable "
                "recording authority."
            )
        if (
            attrs.get("registered_detection_gate_requirement") != "required"
            or not isinstance(gate, Mapping)
            or gate.get("applied") is not True
            or gate.get("status") != "applied"
            or gate.get("ordered_instance_key_coverage_exact") is not True
        ):
            raise ValueError(
                "Strict refined crop input lacks exact required dish-gate consumption."
            )
        instances = bound.instances_group
        frames = np.asarray(instances["frame_indices"][:], dtype=np.int64)
        acquisition_frames = np.asarray(
            instances["source_acquisition_frame_index"][:], dtype=np.int64
        )
        if not np.array_equal(frames, acquisition_frames):
            raise ValueError(
                "Strict full-recording refined rows disagree with acquisition-frame "
                "identity."
            )
        payload = {
            "frame_indices": frames,
            "bbox_norm_coords": np.asarray(instances["bbox_norm_coords"][:]),
            "bbox_img_xyxy": np.asarray(instances["bbox_img_xyxy"][:]),
            "instance_key": np.asarray(instances["instance_key"][:]),
            "refined_row_ids": np.asarray(instances["refined_row_ids"][:]),
            "source_detect_row_index": np.asarray(
                instances["source_detect_row_index"][:]
            ),
        }
        return (
            bound.run_group,
            bound.run_id,
            "explicit_finalized_recording_authority_v1",
            payload,
            str(bound.manifest["payload_digest"]),
            str(bound.logical_content_digest),
        )

    refined_group, resolved_run = resolve_curated_refined_detect_run(
        root, run_name=run_name
    )
    selection_mode = _require_approved_refined_input(
        root, refined_group, run_name=resolved_run
    )
    return (
        refined_group,
        resolved_run,
        selection_mode,
        extract_present_curated_rows(refined_group),
        None,
        None,
    )


def _reviewed_box_is_contained_by_roi(
    bbox_img_xyxy: np.ndarray,
    roi_coordinates_full: np.ndarray,
    roi_sizes_full: np.ndarray,
) -> np.ndarray:
    bbox = np.asarray(bbox_img_xyxy, dtype=np.float64).reshape(-1, 4)
    origins = np.asarray(roi_coordinates_full, dtype=np.int64).reshape(-1, 2)
    sizes = np.asarray(roi_sizes_full, dtype=np.int64).reshape(-1, 2)
    return np.logical_and.reduce(
        (
            np.isfinite(bbox).all(axis=1),
            sizes[:, 0] > 0,
            sizes[:, 1] > 0,
            bbox[:, 0] >= origins[:, 0],
            bbox[:, 1] >= origins[:, 1],
            bbox[:, 2] <= origins[:, 0] + sizes[:, 0],
            bbox[:, 3] <= origins[:, 1] + sizes[:, 1],
        )
    )


def _open_indexed_decoder(video_path: Path) -> Any:
    try:
        import PyNvVideoCodec as nvc  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            f"PyNvVideoCodec import failed; cannot use indexed decode: {exc}"
        ) from exc
    return nvc.SimpleDecoder(
        str(video_path),
        gpu_id=0,
        use_device_memory=True,
        output_color_type=nvc.OutputColorType.NATIVE,
    )


def _open_sequential_reader(video_path: Path) -> Any:
    from fisheye.shared.pynvvc_luma_rgb import PynvvcLumaRgbReader

    return PynvvcLumaRgbReader(video_path, start_frame=0, gpu_id=0)


def _decoder_dimensions(decoder: Any) -> tuple[int, int]:
    if hasattr(decoder, "source_height") and hasattr(decoder, "source_width"):
        return int(decoder.source_height), int(decoder.source_width)
    if hasattr(decoder, "get_stream_metadata"):
        metadata = decoder.get_stream_metadata()
        return int(metadata.height), int(metadata.width)
    raise ValueError("Unable to resolve PyNvVideoCodec source dimensions.")


def _close_decoder(decoder: Any | None) -> None:
    if decoder is None:
        return
    try:
        if hasattr(decoder, "close"):
            decoder.close()
        elif hasattr(decoder, "stop"):
            decoder.stop()
        elif hasattr(decoder, "end"):
            decoder.end()
    except Exception:
        pass


def _choose_decode_mode(requested: str, frames: np.ndarray) -> str:
    if requested not in DECODE_MODES:
        raise ValueError(f"Unsupported decode mode: {requested}")
    if requested != "auto":
        return requested
    if frames.size == 0:
        return "indexed"
    span = int(frames.max()) + 1
    return "indexed" if (frames.size / float(max(1, span))) < 0.5 else "sequential"


def _write_cache_rows(
    *,
    handle: Any,
    frame_tensor: Any,
    rows: Sequence[int],
    roi_coordinates_full: np.ndarray,
    roi_shape: tuple[int, int],
    video_shape: tuple[int, int],
    row_stride: int,
    rows_written_mask: np.ndarray,
) -> None:
    crops = _crop_pynvvc_luma_frame(
        frame_tensor,
        roi_ids=rows,
        roi_coordinates_full=roi_coordinates_full,
        roi_shape=roi_shape,
        video_shape=video_shape,
    )
    crops_cpu = np.ascontiguousarray(crops.cpu().numpy(), dtype=np.uint8)
    for local_idx, row in enumerate(rows):
        row_int = int(row)
        handle.seek(row_int * row_stride)
        handle.write(crops_cpu[int(local_idx)].tobytes(order="C"))
        rows_written_mask[row_int] = True


def _write_supplemental_cache_indexed(
    *,
    video_path: Path,
    manifest_path: Path,
    bin_path: Path,
    frame_to_rows: Mapping[int, list[int]],
    roi_coordinates_full: np.ndarray,
    roi_shape: tuple[int, int],
    video_shape: tuple[int, int],
    decode_chunk_frames: int,
) -> dict[str, Any]:
    import torch

    row_stride = int(roi_shape[0]) * int(roi_shape[1])
    rows_written_mask = np.zeros(int(roi_coordinates_full.shape[0]), dtype=bool)
    timing = {
        "decode_seconds": 0.0,
        "crop_write_seconds": 0.0,
        "indexed_batches": 0,
        "decoded_frames": 0,
    }
    decoder = None
    started = time.perf_counter()
    try:
        decoder = _open_indexed_decoder(video_path)
        source_height, source_width = _decoder_dimensions(decoder)
        if (source_height, source_width) != tuple(video_shape):
            raise ValueError(
                "PyNvVideoCodec dimensions do not match metadata: "
                f"decoder={source_width}x{source_height}, metadata={video_shape[1]}x{video_shape[0]}."
            )
        with bin_path.open("w+b") as handle:
            handle.truncate(int(row_stride * roi_coordinates_full.shape[0]))
            requested = sorted(int(frame_idx) for frame_idx in frame_to_rows)
            for start in range(0, len(requested), max(1, int(decode_chunk_frames))):
                frame_batch = requested[
                    start : start + max(1, int(decode_chunk_frames))
                ]
                decode_started = time.perf_counter()
                decoded_frames = decoder.get_batch_frames_by_index(frame_batch)
                timing["decode_seconds"] += float(time.perf_counter() - decode_started)
                timing["indexed_batches"] += 1
                timing["decoded_frames"] += int(len(decoded_frames))
                if len(decoded_frames) != len(frame_batch):
                    raise RuntimeError(
                        "PyNvVideoCodec indexed decode returned "
                        f"{len(decoded_frames)} frame(s) for {len(frame_batch)} requested index/indices."
                    )
                for frame_idx, frame in zip(frame_batch, decoded_frames):
                    write_started = time.perf_counter()
                    _write_cache_rows(
                        handle=handle,
                        frame_tensor=torch.from_dlpack(frame),
                        rows=frame_to_rows[int(frame_idx)],
                        roi_coordinates_full=roi_coordinates_full,
                        roi_shape=roi_shape,
                        video_shape=video_shape,
                        row_stride=row_stride,
                        rows_written_mask=rows_written_mask,
                    )
                    timing["crop_write_seconds"] += float(
                        time.perf_counter() - write_started
                    )
        if int(rows_written_mask.sum()) != int(roi_coordinates_full.shape[0]):
            raise RuntimeError(
                "Indexed supplemental cache write missed "
                f"{int(roi_coordinates_full.shape[0]) - int(rows_written_mask.sum())} rows."
            )
        timing["total_seconds"] = float(time.perf_counter() - started)
        return {"decode_mode_effective": "indexed", "timing": timing}
    finally:
        _close_decoder(decoder)


def _write_supplemental_cache_sequential(
    *,
    video_path: Path,
    bin_path: Path,
    frame_to_rows: Mapping[int, list[int]],
    roi_coordinates_full: np.ndarray,
    roi_shape: tuple[int, int],
    video_shape: tuple[int, int],
) -> dict[str, Any]:
    row_stride = int(roi_shape[0]) * int(roi_shape[1])
    rows_written_mask = np.zeros(int(roi_coordinates_full.shape[0]), dtype=bool)
    timing = {
        "decode_seconds": 0.0,
        "crop_write_seconds": 0.0,
        "decoded_frames": 0,
        "skipped_frames": 0,
    }
    reader = None
    started = time.perf_counter()
    try:
        reader = _open_sequential_reader(video_path)
        source_height, source_width = _decoder_dimensions(reader)
        if (source_height, source_width) != tuple(video_shape):
            raise ValueError(
                "PyNvVideoCodec dimensions do not match metadata: "
                f"decoder={source_width}x{source_height}, metadata={video_shape[1]}x{video_shape[0]}."
            )
        max_frame = int(max(frame_to_rows)) if frame_to_rows else -1
        with bin_path.open("w+b") as handle:
            handle.truncate(int(row_stride * roi_coordinates_full.shape[0]))
            frame_iter = reader.iter_frames()
            frame_idx = 0
            while frame_idx <= max_frame:
                decode_started = time.perf_counter()
                try:
                    frame_tensor = next(frame_iter)
                except StopIteration:
                    break
                timing["decode_seconds"] += float(time.perf_counter() - decode_started)
                timing["decoded_frames"] += 1
                rows = frame_to_rows.get(frame_idx)
                if rows:
                    write_started = time.perf_counter()
                    _write_cache_rows(
                        handle=handle,
                        frame_tensor=frame_tensor,
                        rows=rows,
                        roi_coordinates_full=roi_coordinates_full,
                        roi_shape=roi_shape,
                        video_shape=video_shape,
                        row_stride=row_stride,
                        rows_written_mask=rows_written_mask,
                    )
                    timing["crop_write_seconds"] += float(
                        time.perf_counter() - write_started
                    )
                else:
                    timing["skipped_frames"] += 1
                frame_idx += 1
        if int(rows_written_mask.sum()) != int(roi_coordinates_full.shape[0]):
            raise RuntimeError(
                "Sequential supplemental cache write missed "
                f"{int(roi_coordinates_full.shape[0]) - int(rows_written_mask.sum())} rows."
            )
        timing["total_seconds"] = float(time.perf_counter() - started)
        return {"decode_mode_effective": "sequential", "timing": timing}
    finally:
        _close_decoder(reader)


def _write_supplemental_cache(
    *,
    zarr_path: Path,
    run_name: str,
    video_path: Path,
    manifest_path: Path,
    frame_indices: np.ndarray,
    roi_coordinates_full: np.ndarray,
    roi_shape: tuple[int, int],
    video_shape: tuple[int, int],
    decode_mode: str,
    decode_chunk_frames: int,
    overwrite: bool,
) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    bin_path = manifest_path.with_suffix(".bin")
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f"Supplemental ROI cache manifest exists: {manifest_path}"
        )
    if bin_path.exists() and not overwrite:
        raise FileExistsError(f"Supplemental ROI cache payload exists: {bin_path}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    frame_to_rows: dict[int, list[int]] = {}
    for row_idx, frame_idx in enumerate(
        np.asarray(frame_indices, dtype=np.int64).reshape(-1)
    ):
        frame_to_rows.setdefault(int(frame_idx), []).append(int(row_idx))

    effective_requested = _choose_decode_mode(
        str(decode_mode), np.unique(frame_indices.astype(np.int64, copy=False))
    )
    try:
        if effective_requested == "indexed":
            cache_report = _write_supplemental_cache_indexed(
                video_path=video_path,
                manifest_path=manifest_path,
                bin_path=bin_path,
                frame_to_rows=frame_to_rows,
                roi_coordinates_full=roi_coordinates_full,
                roi_shape=roi_shape,
                video_shape=video_shape,
                decode_chunk_frames=decode_chunk_frames,
            )
        else:
            cache_report = _write_supplemental_cache_sequential(
                video_path=video_path,
                bin_path=bin_path,
                frame_to_rows=frame_to_rows,
                roi_coordinates_full=roi_coordinates_full,
                roi_shape=roi_shape,
                video_shape=video_shape,
            )
    except Exception as exc:
        if decode_mode == "indexed":
            raise
        if effective_requested == "indexed":
            cache_report = _write_supplemental_cache_sequential(
                video_path=video_path,
                bin_path=bin_path,
                frame_to_rows=frame_to_rows,
                roi_coordinates_full=roi_coordinates_full,
                roi_shape=roi_shape,
                video_shape=video_shape,
            )
            cache_report["indexed_fallback_error"] = f"{exc.__class__.__name__}: {exc}"
        else:
            raise

    total_bytes = (
        int(roi_coordinates_full.shape[0]) * int(roi_shape[0]) * int(roi_shape[1])
    )
    pixel_contract = orange_mono_pynvvc_luma_pixel_contract(
        source_pixels=SOURCE_PIXELS_RAW_CAMERA_VIDEO,
    )
    manifest = {
        "schema": FLAT_ROI_CACHE_SCHEMA,
        "layout": FLAT_ROI_CACHE_LAYOUT,
        "cache_complete": True,
        "cache_key": f"{run_name}:offline_supplement:{roi_coordinates_full.shape[0]}",
        "manifest_path": str(manifest_path),
        "created_at_utc": _utc_now(),
        "source": {
            "archive_path": str(zarr_path),
            "crop_run_name": str(run_name),
            "source_crop_storage_mode": "geometry_only",
            "frame_source_kind": "source_video_path",
            "frame_source_path": str(video_path),
            "frame_source_identity": {
                "source_video_path": str(video_path),
                "frame_index_min": (
                    int(frame_indices.min()) if frame_indices.size else None
                ),
                "frame_index_max": (
                    int(frame_indices.max()) if frame_indices.size else None
                ),
                "row_count": int(frame_indices.shape[0]),
            },
        },
        "array": {
            "bin_path": bin_path.name,
            "dtype": "uint8",
            "shape": [
                int(roi_coordinates_full.shape[0]),
                int(roi_shape[0]),
                int(roi_shape[1]),
            ],
            "order": "C",
            "row_stride_bytes": int(roi_shape[0]) * int(roi_shape[1]),
            "total_bytes": int(total_bytes),
            "sha256": None,
        },
        "builder": {
            "module": __name__,
            "decode_backend_requested": "pynvvc_luma",
            "decode_backend_effective": "pynvvc_luma",
            "decode_mode_requested": str(decode_mode),
            "decode_mode_effective": cache_report["decode_mode_effective"],
            "decode_chunk_frames": int(decode_chunk_frames),
            "pixel_contract": pixel_contract,
            "pixel_contract_name": pixel_contract.get("name"),
            "timing": cache_report["timing"],
        },
    }
    if "indexed_fallback_error" in cache_report:
        manifest["builder"]["indexed_fallback_error"] = cache_report[
            "indexed_fallback_error"
        ]
    tmp_path = manifest_path.with_suffix(".tmp.json")
    tmp_path.write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_path, manifest_path)
    return manifest


def _default_manifest_path(recording_dir: Path, zarr_path: Path, run_name: str) -> Path:
    zarr_stem = zarr_path.name.removesuffix(".zarr")
    return (
        recording_dir
        / "derived"
        / "roi_cache"
        / run_name
        / f"{zarr_stem}__{run_name}.supplemental.flat_roi_cache.json"
    )


def _prepare_canonical_ledger_hybrid_payload(
    *,
    root: zarr.Group,
    ledger_group: zarr.Group,
    refined_payload: Mapping[str, np.ndarray],
    frame_width: int,
    frame_height: int,
    roi_shape: tuple[int, int],
    context_margin_px: float = PRODUCTION_CONTEXT_MARGIN_PX,
) -> dict[str, np.ndarray | dict[str, int]]:
    """Route reviewed rows through the frame-complete acquisition ledger.

    This join deliberately ignores the live detector's observation identity and
    geometry.  The acquisition row is selected only by the recording frame;
    refined ``instance_key`` and bbox values remain authoritative.
    """

    if tuple(int(value) for value in roi_shape) != (
        DEFAULT_ZEBRAFISH_CROP_SIZE_PX,
        DEFAULT_ZEBRAFISH_CROP_SIZE_PX,
    ):
        raise ValueError(
            f"{PRODUCTION_CROP_POLICY_ID} requires exact "
            f"{DEFAULT_ZEBRAFISH_CROP_SIZE_PX}x{DEFAULT_ZEBRAFISH_CROP_SIZE_PX} "
            f"acquisition crops; observed {roi_shape[1]}x{roi_shape[0]}."
        )
    margin = float(context_margin_px)
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("context_margin_px must be finite and non-negative.")

    ledger_frames = np.asarray(
        ledger_group["source_recording_frame_indices"][:], dtype=np.int64
    ).reshape(-1)
    ledger_rows = int(ledger_frames.shape[0])
    if np.unique(ledger_frames).shape[0] != ledger_rows:
        raise ValueError("Canonical acquisition crop ledger frame indices are duplicated.")
    ledger_frame_ids = np.asarray(
        ledger_group["source_recording_frame_ids"][:], dtype=np.int64
    ).reshape(-1)
    ledger_meta_rows = np.asarray(
        ledger_group["source_crop_meta_row_indices"][:], dtype=np.int64
    ).reshape(-1)
    ledger_video_frames = np.asarray(
        ledger_group["source_crop_video_frame_indices"][:], dtype=np.int64
    ).reshape(-1)
    ledger_local_frames = np.asarray(
        ledger_group["source_crop_local_frame_ids"][:], dtype=np.int64
    ).reshape(-1)
    ledger_has_detection = np.asarray(
        ledger_group["has_detection"][:], dtype=bool
    ).reshape(-1)
    ledger_blank = np.asarray(ledger_group["blank_frame"][:], dtype=bool).reshape(-1)
    ledger_crop_valid = np.asarray(
        ledger_group["crop_rect_valid"][:], dtype=bool
    ).reshape(-1)
    ledger_crop_xywh = np.asarray(
        ledger_group["source_crop_xywh"][:], dtype=np.float64
    ).reshape(-1, 4)
    row_aligned = {
        "source_recording_frame_ids": ledger_frame_ids,
        "source_crop_meta_row_indices": ledger_meta_rows,
        "source_crop_video_frame_indices": ledger_video_frames,
        "source_crop_local_frame_ids": ledger_local_frames,
        "has_detection": ledger_has_detection,
        "blank_frame": ledger_blank,
        "crop_rect_valid": ledger_crop_valid,
        "source_crop_xywh": ledger_crop_xywh,
    }
    bad_lengths = {
        name: int(values.shape[0])
        for name, values in row_aligned.items()
        if int(values.shape[0]) != ledger_rows
    }
    if bad_lengths:
        raise ValueError(
            "Canonical acquisition crop ledger arrays are not row aligned: "
            f"{bad_lengths}."
        )

    refined_frames = np.asarray(
        refined_payload["frame_indices"], dtype=np.int64
    ).reshape(-1)
    refined_rows = int(refined_frames.shape[0])
    refined_keys = np.asarray(refined_payload.get("instance_key"))
    if refined_keys.shape != (refined_rows,) or refined_keys.dtype != np.dtype(np.uint64):
        raise ValueError(
            "Refined detection instance_key must have exact uint64 shape "
            f"({refined_rows},)."
        )
    if np.unique(refined_keys).shape[0] != refined_rows:
        raise ValueError("Refined detection instance_key values must be unique.")
    refined_keys = np.asarray(refined_keys, dtype=np.uint64)
    canonical_order = np.lexsort((refined_keys, refined_frames))
    if not np.array_equal(canonical_order, np.arange(refined_rows, dtype=np.int64)):
        raise ValueError(
            "Reviewed refined rows must already be ordered by frame_indices and "
            "instance_key; routing does not reorder scientific authority."
        )

    refined_bbox_norm = np.asarray(
        refined_payload["bbox_norm_coords"], dtype=np.float32
    ).reshape(-1, 4)
    if refined_bbox_norm.shape != (refined_rows, 4):
        raise ValueError(
            f"Refined bbox_norm_coords must have exact shape ({refined_rows}, 4)."
        )
    refined_bbox_img, _ = derive_canonical_detection_geometry(
        refined_bbox_norm,
        source_width=int(frame_width),
        source_height=int(frame_height),
    )
    refined_bbox_img = np.asarray(refined_bbox_img, dtype=np.float32)
    valid_bbox = np.logical_and(
        np.isfinite(refined_bbox_img).all(axis=1),
        np.logical_and(
            refined_bbox_img[:, 2] > refined_bbox_img[:, 0],
            refined_bbox_img[:, 3] > refined_bbox_img[:, 1],
        ),
    )
    if not np.all(valid_bbox):
        bad = np.flatnonzero(~valid_bbox)[:10]
        raise ValueError(
            "Reviewed refined detections must have finite positive boxes; bad rows "
            f"include {bad.tolist()}."
        )

    row_by_frame = {int(frame): row for row, frame in enumerate(ledger_frames.tolist())}
    matched_rows = np.asarray(
        [row_by_frame.get(int(frame), -1) for frame in refined_frames], dtype=np.int64
    )
    has_frame_match = matched_rows >= 0
    frame_identity_agrees = np.zeros(refined_rows, dtype=bool)
    if np.any(has_frame_match):
        matched = matched_rows[has_frame_match]
        frame_identity_agrees[has_frame_match] = np.logical_and(
            ledger_frames[matched] == refined_frames[has_frame_match],
            ledger_frame_ids[matched] == refined_frames[has_frame_match] + 1,
        )

    matched_crop_xywh = np.full((refined_rows, 4), np.nan, dtype=np.float64)
    matched_has_detection = np.zeros(refined_rows, dtype=bool)
    matched_blank = np.ones(refined_rows, dtype=bool)
    matched_crop_valid = np.zeros(refined_rows, dtype=bool)
    if np.any(has_frame_match):
        matched = matched_rows[has_frame_match]
        matched_crop_xywh[has_frame_match] = ledger_crop_xywh[matched]
        matched_has_detection[has_frame_match] = ledger_has_detection[matched]
        matched_blank[has_frame_match] = ledger_blank[matched]
        matched_crop_valid[has_frame_match] = ledger_crop_valid[matched]

    rounded_crop_xywh = np.rint(matched_crop_xywh)
    integral_crop_geometry = np.logical_and(
        np.isfinite(matched_crop_xywh).all(axis=1),
        np.all(np.isclose(matched_crop_xywh, rounded_crop_xywh, atol=0.0), axis=1),
    )
    expected_w = int(roi_shape[1])
    expected_h = int(roi_shape[0])
    exact_extent = np.logical_and(
        matched_crop_xywh[:, 2] == float(expected_w),
        matched_crop_xywh[:, 3] == float(expected_h),
    )
    in_native_bounds = np.logical_and.reduce(
        (
            matched_crop_xywh[:, 0] >= 0.0,
            matched_crop_xywh[:, 1] >= 0.0,
            matched_crop_xywh[:, 0] + matched_crop_xywh[:, 2] <= float(frame_width),
            matched_crop_xywh[:, 1] + matched_crop_xywh[:, 3] <= float(frame_height),
        )
    )
    valid_placement = np.logical_and.reduce(
        (matched_crop_valid, integral_crop_geometry, exact_extent, in_native_bounds)
    )

    containment_margins = np.full((refined_rows, 4), np.nan, dtype=np.float32)
    if np.any(has_frame_match):
        crop = matched_crop_xywh[has_frame_match]
        bbox = refined_bbox_img[has_frame_match]
        containment_margins[has_frame_match] = np.column_stack(
            (
                bbox[:, 0] - crop[:, 0],
                bbox[:, 1] - crop[:, 1],
                crop[:, 0] + crop[:, 2] - bbox[:, 2],
                crop[:, 1] + crop[:, 3] - bbox[:, 3],
            )
        ).astype(np.float32, copy=False)
    contains_context = np.logical_and(
        np.isfinite(containment_margins).all(axis=1),
        np.all(containment_margins >= margin, axis=1),
    )

    video_mask = np.logical_and.reduce(
        (
            has_frame_match,
            frame_identity_agrees,
            matched_has_detection,
            ~matched_blank,
            valid_placement,
            contains_context,
        )
    )
    supplemental_mask = ~video_mask
    reason_codes = np.full(
        refined_rows,
        ROUTING_REASON_CODE_MAP["acquisition_crop_selected"],
        dtype=np.int8,
    )
    reason_codes[~has_frame_match] = ROUTING_REASON_CODE_MAP[
        "acquisition_crop_missing"
    ]
    reason_codes[np.logical_and(has_frame_match, ~frame_identity_agrees)] = (
        ROUTING_REASON_CODE_MAP["frame_identity_mismatch"]
    )
    eligible_prefix = np.logical_and(has_frame_match, frame_identity_agrees)
    reason_codes[np.logical_and(eligible_prefix, matched_blank)] = ROUTING_REASON_CODE_MAP[
        "blank_acquisition_crop"
    ]
    reason_codes[
        np.logical_and.reduce((eligible_prefix, ~matched_blank, ~matched_has_detection))
    ] = ROUTING_REASON_CODE_MAP["acquisition_no_detection"]
    reason_codes[
        np.logical_and.reduce(
            (eligible_prefix, ~matched_blank, matched_has_detection, ~valid_placement)
        )
    ] = ROUTING_REASON_CODE_MAP["coordinate_or_extent_mismatch"]
    reason_codes[
        np.logical_and.reduce(
            (
                eligible_prefix,
                ~matched_blank,
                matched_has_detection,
                valid_placement,
                ~contains_context,
            )
        )
    ] = ROUTING_REASON_CODE_MAP["canonical_roi_not_contained"]

    supplemental_origins, supplemental_sizes = compute_centered_roi_mapping(
        refined_bbox_img[supplemental_mask], roi_size=roi_shape
    )
    roi_origins = np.zeros((refined_rows, 2), dtype=np.int32)
    roi_sizes = np.tile(
        np.asarray([[expected_w, expected_h]], dtype=np.int32), (refined_rows, 1)
    )
    if np.any(video_mask):
        roi_origins[video_mask] = rounded_crop_xywh[video_mask, :2].astype(np.int32)
    roi_origins[supplemental_mask] = supplemental_origins
    roi_sizes[supplemental_mask] = supplemental_sizes
    bbox_roi = _bbox_roi_xyxy(refined_bbox_img, roi_origins)
    bbox_crop_norm = _bbox_crop_norm_xywh(bbox_roi, roi_shape)

    def _matched_values(name: str, *, dtype: Any, fill: Any) -> np.ndarray:
        source = np.asarray(row_aligned[name], dtype=dtype).reshape(-1)
        result = np.full(refined_rows, fill, dtype=dtype)
        result[has_frame_match] = source[matched_rows[has_frame_match]]
        return result

    refined_row_ids = np.asarray(
        refined_payload.get("refined_row_ids", np.arange(refined_rows)), dtype=np.int64
    ).reshape(-1)
    source_detect_rows = np.asarray(
        refined_payload.get("source_detect_row_index", np.full(refined_rows, -1)),
        dtype=np.int64,
    ).reshape(-1)
    if refined_row_ids.shape != (refined_rows,) or source_detect_rows.shape != (
        refined_rows,
    ):
        raise ValueError("Refined row lineage arrays do not match the reviewed row count.")

    provider_crop_xywh = np.column_stack((roi_origins, roi_sizes)).astype(np.float64)
    combined: dict[str, np.ndarray | dict[str, int]] = {
        "instance_key": refined_keys,
        "frame_indices": refined_frames,
        "source_frame_indices": refined_frames,
        "source_acquisition_frame_index": refined_frames,
        "source_recording_frame_ids": np.where(
            has_frame_match,
            _matched_values("source_recording_frame_ids", dtype=np.int64, fill=-1),
            refined_frames + 1,
        ).astype(np.int64),
        "source_crop_meta_row_indices": _matched_values(
            "source_crop_meta_row_indices", dtype=np.int64, fill=-1
        ),
        "source_acquisition_crop_row_indices": matched_rows,
        "source_crop_video_frame_indices": _matched_values(
            "source_crop_video_frame_indices", dtype=np.int64, fill=-1
        ),
        "source_crop_local_frame_ids": _matched_values(
            "source_crop_local_frame_ids", dtype=np.int64, fill=-1
        ),
        "source_acquisition_crop_xywh": matched_crop_xywh,
        "source_crop_xywh": provider_crop_xywh,
        "roi_coordinates_full": roi_origins,
        "roi_sizes_full": roi_sizes,
        "bbox_img_xyxy": refined_bbox_img,
        "bbox_norm_coords": refined_bbox_norm,
        "bbox_roi_xyxy": bbox_roi.astype(np.float32),
        "bbox_crop_norm_coords": bbox_crop_norm.astype(np.float32),
        "containment_margins_px": containment_margins,
        "detection_success": np.ones(refined_rows, dtype=bool),
        "detection_source": np.full(
            refined_rows,
            DETECTION_SOURCE_CODE_MAP["reviewed_refined_detection"],
            dtype=np.int8,
        ),
        "source_pixel_kind_codes": np.where(
            video_mask,
            SOURCE_PIXEL_KIND_CODE_MAP["acquisition_crop_video"],
            SOURCE_PIXEL_KIND_CODE_MAP[
                "offline_full_frame_supplemental_flat_cache"
            ],
        ).astype(np.int8),
        "crop_state_codes": np.where(
            video_mask,
            CROP_STATE_CODE_MAP["reviewed_acquisition_crop_reused"],
            CROP_STATE_CODE_MAP["reviewed_supplemental_crop_materialized"],
        ).astype(np.int8),
        "routing_reason_codes": reason_codes,
        "source_refined_row_ids": refined_row_ids,
        "source_detect_row_index": source_detect_rows,
    }
    supplemental_rows = np.full(refined_rows, -1, dtype=np.int64)
    supplemental_rows[supplemental_mask] = np.arange(
        int(np.count_nonzero(supplemental_mask)), dtype=np.int64
    )
    combined["supplemental_cache_row_indices"] = supplemental_rows

    total_frames = max(
        int(root.attrs.get("total_frames") or 0),
        int(refined_frames.max()) + 1 if refined_frames.size else 0,
    )
    frame_counts = np.zeros(total_frames, dtype=np.int32)
    if total_frames:
        np.add.at(frame_counts, refined_frames, 1)
    combined["frame_counts"] = frame_counts
    frame_offsets = np.zeros(total_frames + 1, dtype=np.int64)
    frame_offsets[1:] = np.cumsum(frame_counts, dtype=np.int64)
    combined["frame_row_offsets"] = frame_offsets
    combined["detection_indices"] = np.arange(refined_rows, dtype=np.int64)
    reverse_reasons = {value: key for key, value in ROUTING_REASON_CODE_MAP.items()}
    reason_histogram = {
        reverse_reasons[int(code)]: int(np.count_nonzero(reason_codes == code))
        for code in sorted(set(reason_codes.tolist()))
    }
    combined["summary"] = {
        "acquisition_ledger_rows_available": ledger_rows,
        "reviewed_refined_rows": refined_rows,
        "acquisition_video_rows_reused": int(np.count_nonzero(video_mask)),
        "supplemental_rows_materialized": int(np.count_nonzero(supplemental_mask)),
        "matched_acquisition_ledger_rows": int(np.count_nonzero(has_frame_match)),
        "total_rows": refined_rows,
        "total_frames": total_frames,
        **{f"routing_reason_{name}": count for name, count in reason_histogram.items()},
    }
    return combined


def _prepare_hybrid_payload(
    *,
    root: zarr.Group,
    acquisition_group: zarr.Group,
    refined_payload: Mapping[str, np.ndarray],
    frame_width: int,
    frame_height: int,
    roi_shape: tuple[int, int],
) -> dict[str, np.ndarray | dict[str, int]]:
    online_frame_indices = np.asarray(
        acquisition_group["frame_indices"][:], dtype=np.int64
    ).reshape(-1)
    online_rows = int(online_frame_indices.shape[0])
    online_instance_keys = _require_unique_instance_keys(
        acquisition_group,
        rows=online_rows,
        label="Acquisition crop run",
    )

    refined_frame_indices = np.asarray(
        refined_payload["frame_indices"], dtype=np.int64
    ).reshape(-1)
    refined_rows = int(refined_frame_indices.shape[0])
    if "instance_key" not in refined_payload:
        raise ValueError(
            "Refined detection authority requires instance_key for reviewed hybrid routing."
        )
    refined_instance_keys = np.asarray(refined_payload["instance_key"])
    if refined_instance_keys.shape != (refined_rows,) or np.dtype(
        refined_instance_keys.dtype
    ) != np.dtype(np.uint64):
        raise ValueError(
            "Refined detection instance_key must have exact uint64 shape "
            f"({refined_rows},)."
        )
    if np.unique(refined_instance_keys).shape[0] != refined_rows:
        raise ValueError("Refined detection instance_key values must be unique.")
    refined_instance_keys = np.asarray(refined_instance_keys, dtype=np.uint64)

    refined_bbox_norm = np.asarray(
        refined_payload["bbox_norm_coords"], dtype=np.float32
    ).reshape(-1, 4)
    if refined_bbox_norm.shape != (refined_rows, 4):
        raise ValueError(
            "Refined bbox_norm_coords must have exact shape "
            f"({refined_rows}, 4)."
        )
    refined_bbox_img, _refined_centers = derive_canonical_detection_geometry(
        refined_bbox_norm,
        source_width=int(frame_width),
        source_height=int(frame_height),
    )
    refined_bbox_img = np.asarray(refined_bbox_img, dtype=np.float32)
    finite_bbox = np.isfinite(refined_bbox_img).all(axis=1)
    positive_bbox = np.logical_and(
        refined_bbox_img[:, 2] > refined_bbox_img[:, 0],
        refined_bbox_img[:, 3] > refined_bbox_img[:, 1],
    )
    if not np.all(np.logical_and(finite_bbox, positive_bbox)):
        bad = np.flatnonzero(~np.logical_and(finite_bbox, positive_bbox))[:10]
        raise ValueError(
            "Reviewed refined detections must have finite positive boxes; bad rows "
            f"include {bad.tolist()}."
        )
    online_roi_coordinates = np.asarray(
        acquisition_group["roi_coordinates_full"][:], dtype=np.int32
    ).reshape(-1, 2)
    online_roi_sizes = np.asarray(
        acquisition_group["roi_sizes_full"][:], dtype=np.int32
    ).reshape(-1, 2)
    if (
        online_roi_coordinates.shape[0] != online_rows
        or online_roi_sizes.shape[0] != online_rows
    ):
        raise ValueError(
            "Acquisition crop geometry row counts do not match frame_indices."
        )
    expected_size = np.asarray([int(roi_shape[1]), int(roi_shape[0])], dtype=np.int32)
    if online_rows and np.any(online_roi_sizes != expected_size):
        raise ValueError(
            "Acquisition crop rows must all match the fixed hybrid ROI shape."
        )

    online_row_by_key = {
        int(key): int(row) for row, key in enumerate(online_instance_keys.tolist())
    }
    matched_online_rows = np.asarray(
        [online_row_by_key.get(int(key), -1) for key in refined_instance_keys],
        dtype=np.int64,
    )
    has_identity_match = matched_online_rows >= 0
    matched_frames_agree = np.zeros(refined_rows, dtype=bool)
    if np.any(has_identity_match):
        matched_frames_agree[has_identity_match] = (
            online_frame_indices[matched_online_rows[has_identity_match]]
            == refined_frame_indices[has_identity_match]
        )
    if np.any(np.logical_and(has_identity_match, ~matched_frames_agree)):
        bad = np.flatnonzero(np.logical_and(has_identity_match, ~matched_frames_agree))[
            :10
        ]
        raise ValueError(
            "Stable instance_key matched acquisition and refined rows on different frames; "
            f"bad refined rows include {bad.tolist()}."
        )

    matched_origins = np.zeros((refined_rows, 2), dtype=np.int32)
    matched_sizes = np.zeros((refined_rows, 2), dtype=np.int32)
    if np.any(has_identity_match):
        matched_origins[has_identity_match] = online_roi_coordinates[
            matched_online_rows[has_identity_match]
        ]
        matched_sizes[has_identity_match] = online_roi_sizes[
            matched_online_rows[has_identity_match]
        ]
    contains_reviewed_bbox = np.zeros(refined_rows, dtype=bool)
    if np.any(has_identity_match):
        contains_reviewed_bbox[has_identity_match] = _reviewed_box_is_contained_by_roi(
            refined_bbox_img[has_identity_match],
            matched_origins[has_identity_match],
            matched_sizes[has_identity_match],
        )
    video_mask = np.logical_and(has_identity_match, contains_reviewed_bbox)
    supplemental_mask = ~video_mask

    supplemental_roi_coordinates, supplemental_roi_sizes = compute_centered_roi_mapping(
        refined_bbox_img[supplemental_mask],
        roi_size=roi_shape,
    )
    roi_coordinates = np.array(matched_origins, copy=True)
    roi_sizes = np.array(matched_sizes, copy=True)
    roi_coordinates[supplemental_mask] = supplemental_roi_coordinates
    roi_sizes[supplemental_mask] = supplemental_roi_sizes
    bbox_roi = _bbox_roi_xyxy(refined_bbox_img, roi_coordinates)
    bbox_crop_norm = _bbox_crop_norm_xywh(bbox_roi, roi_shape)

    def _online_values(name: str, *, dtype: Any, fill: Any) -> np.ndarray:
        source = _read_array_or_default(
            acquisition_group,
            name,
            rows=online_rows,
            dtype=dtype,
            fill=fill,
        ).reshape(-1)
        output = np.full(refined_rows, fill, dtype=dtype)
        output[video_mask] = source[matched_online_rows[video_mask]]
        return output

    source_crop_xywh = np.column_stack([roi_coordinates, roi_sizes]).astype(
        np.float64, copy=False
    )
    video_crop_xywh = _read_array_or_default(
        acquisition_group,
        "source_crop_xywh",
        rows=online_rows,
        shape_suffix=(4,),
        dtype=np.float64,
        fill=np.nan,
    ).reshape(-1, 4)
    if np.any(video_mask):
        source_crop_xywh[video_mask] = video_crop_xywh[matched_online_rows[video_mask]]
    source_acquisition_crop_xywh = np.full(
        (refined_rows, 4), np.nan, dtype=np.float64
    )
    if np.any(video_mask):
        source_acquisition_crop_xywh[video_mask] = video_crop_xywh[
            matched_online_rows[video_mask]
        ]

    refined_row_ids = np.asarray(
        refined_payload.get("refined_row_ids", np.arange(refined_rows)),
        dtype=np.int64,
    ).reshape(-1)
    if refined_row_ids.shape != (refined_rows,):
        raise ValueError("Refined row identities do not match the reviewed row count.")
    source_detect_rows = np.asarray(
        refined_payload.get("source_detect_row_index", np.full(refined_rows, -1)),
        dtype=np.int64,
    ).reshape(-1)
    if source_detect_rows.shape != (refined_rows,):
        raise ValueError("Refined source-detection lineage does not match row count.")

    combined = {
        "instance_key": refined_instance_keys,
        "frame_indices": refined_frame_indices,
        "source_frame_indices": refined_frame_indices,
        "source_acquisition_frame_index": refined_frame_indices,
        "source_recording_frame_ids": np.where(
            video_mask,
            _online_values("source_recording_frame_ids", dtype=np.int64, fill=-1),
            refined_frame_indices + 1,
        ).astype(np.int64, copy=False),
        "source_crop_meta_row_indices": _online_values(
            "source_crop_meta_row_indices", dtype=np.int64, fill=-1
        ),
        "source_acquisition_crop_row_indices": np.where(
            video_mask,
            matched_online_rows,
            -1,
        ).astype(np.int64, copy=False),
        "source_crop_video_frame_indices": _online_values(
            "source_crop_video_frame_indices", dtype=np.int64, fill=-1
        ),
        "source_crop_local_frame_ids": _online_values(
            "source_crop_local_frame_ids", dtype=np.int64, fill=-1
        ),
        "source_acquisition_crop_xywh": source_acquisition_crop_xywh,
        "source_crop_xywh": source_crop_xywh,
        "roi_coordinates_full": roi_coordinates.astype(np.int32, copy=False),
        "roi_sizes_full": roi_sizes.astype(np.int32, copy=False),
        "bbox_img_xyxy": refined_bbox_img.astype(np.float32, copy=False),
        "bbox_norm_coords": refined_bbox_norm.astype(np.float32, copy=False),
        "bbox_roi_xyxy": bbox_roi.astype(np.float32, copy=False),
        "bbox_crop_norm_coords": bbox_crop_norm.astype(np.float32, copy=False),
        "detection_success": np.ones(refined_rows, dtype=bool),
        "detection_source": np.full(
            refined_rows,
            DETECTION_SOURCE_CODE_MAP["reviewed_refined_detection"],
            dtype=np.int8,
        ),
        "source_pixel_kind_codes": np.where(
            video_mask,
            SOURCE_PIXEL_KIND_CODE_MAP["acquisition_crop_video"],
            SOURCE_PIXEL_KIND_CODE_MAP["offline_full_frame_supplemental_flat_cache"],
        ).astype(np.int8, copy=False),
        "crop_state_codes": np.where(
            video_mask,
            CROP_STATE_CODE_MAP["reviewed_acquisition_crop_reused"],
            CROP_STATE_CODE_MAP["reviewed_supplemental_crop_materialized"],
        ).astype(np.int8, copy=False),
        "source_refined_row_ids": refined_row_ids,
        "source_detect_row_index": source_detect_rows,
    }
    order = np.lexsort((refined_instance_keys, refined_frame_indices))
    for name, arr in list(combined.items()):
        combined[name] = np.asarray(arr)[order]

    final_supplemental_mask = (
        np.asarray(combined["source_pixel_kind_codes"], dtype=np.int8)
        == SOURCE_PIXEL_KIND_CODE_MAP["offline_full_frame_supplemental_flat_cache"]
    )
    supplemental_cache_rows = np.full(refined_rows, -1, dtype=np.int64)
    supplemental_cache_rows[final_supplemental_mask] = np.arange(
        int(np.count_nonzero(final_supplemental_mask)), dtype=np.int64
    )
    combined["supplemental_cache_row_indices"] = supplemental_cache_rows

    total_frames = max(
        int(root.attrs.get("total_frames") or 0),
        (
            int(np.asarray(combined["frame_indices"]).max()) + 1
            if combined["frame_indices"].size
            else 0
        ),
    )
    frame_counts = np.zeros(total_frames, dtype=np.int32)
    if total_frames:
        np.add.at(
            frame_counts, np.asarray(combined["frame_indices"], dtype=np.int64), 1
        )
    combined["frame_counts"] = frame_counts
    frame_row_offsets = np.zeros(total_frames + 1, dtype=np.int64)
    frame_row_offsets[1:] = np.cumsum(frame_counts, dtype=np.int64)
    combined["frame_row_offsets"] = frame_row_offsets
    combined["detection_indices"] = np.arange(
        int(combined["frame_indices"].shape[0]), dtype=np.int64
    )
    reused_online_rows = set(
        int(value) for value in matched_online_rows[video_mask].tolist()
    )
    combined["summary"] = {
        "online_rows_available": int(online_rows),
        "reviewed_refined_rows": int(refined_rows),
        "acquisition_video_rows_reused": int(np.count_nonzero(video_mask)),
        "supplemental_rows_materialized": int(np.count_nonzero(supplemental_mask)),
        "supplemental_unmatched_instance_key": int(
            np.count_nonzero(~has_identity_match)
        ),
        "supplemental_reviewed_bbox_outside_acquisition_roi": int(
            np.count_nonzero(
                np.logical_and(has_identity_match, ~contains_reviewed_bbox)
            )
        ),
        "acquisition_rows_retired": int(online_rows - len(reused_online_rows)),
        "total_rows": int(refined_rows),
        "total_frames": int(total_frames),
    }
    return combined


def build_hybrid_acquisition_offline_crop_run(
    zarr_path: str | Path,
    *,
    acquisition_source_mode: str = "canonical_ledger",
    acquisition_ledger_record_sha256: str | None = None,
    acquisition_crop_run: str | None = None,
    refined_detect_run: str | None = None,
    run_name: str | None = None,
    recording_dir: str | Path | None = None,
    source_video_path: str | Path | None = None,
    supplemental_manifest_path: str | Path | None = None,
    decode_mode: str = "auto",
    decode_chunk_frames: int = 1,
    context_margin_px: float = PRODUCTION_CONTEXT_MARGIN_PX,
    overwrite: bool = False,
    set_latest_any: bool = False,
    apply: bool = False,
) -> dict[str, Any]:
    if acquisition_source_mode not in ACQUISITION_SOURCE_MODES:
        raise ValueError(
            f"acquisition_source_mode must be one of {ACQUISITION_SOURCE_MODES}."
        )
    if acquisition_source_mode == "canonical_ledger" and acquisition_crop_run:
        raise ValueError(
            "--acquisition-crop-run is valid only with the explicit "
            "legacy_crop_run acquisition source mode."
        )
    if (
        acquisition_source_mode == "legacy_crop_run"
        and acquisition_ledger_record_sha256 is not None
    ):
        raise ValueError(
            "An acquisition ledger digest cannot be used with legacy_crop_run mode."
        )
    archive_path = Path(zarr_path).expanduser().resolve()
    resolved_recording_dir = _resolve_recording_dir(
        archive_path,
        Path(recording_dir) if recording_dir is not None else None,
    )
    resolved_run_name = run_name or _utc_run_name()
    manifest_path = (
        Path(supplemental_manifest_path).expanduser()
        if supplemental_manifest_path is not None
        else _default_manifest_path(
            resolved_recording_dir, archive_path, resolved_run_name
        )
    )

    root = zarr.open_group(
        str(archive_path), mode="a" if apply else "r", use_consolidated=False
    )
    if acquisition_source_mode == "canonical_ledger":
        (
            acquisition_group,
            resolved_acquisition_run,
            resolved_acquisition_record_sha256,
        ) = _resolve_canonical_crop_ledger(
            root,
            expected_record_sha256=acquisition_ledger_record_sha256,
        )
        crop_parent = root.require_group("crop_runs")
        acquisition_crop_video_path = _ledger_crop_video_path(acquisition_group)
        roi_shape = _ledger_roi_shape(acquisition_group)
    else:
        crop_parent, acquisition_group, resolved_acquisition_run = _resolve_crop_run(
            root, acquisition_crop_run
        )
        resolved_acquisition_record_sha256 = None
        raw_crop_video = str(
            acquisition_group.attrs.get("source_crop_video_path")
            or acquisition_group.attrs.get("source_video_path")
            or ""
        ).strip()
        if not raw_crop_video:
            raise ValueError("Legacy acquisition crop run lacks its crop-video path.")
        acquisition_crop_video_path = Path(raw_crop_video).expanduser().resolve()
        if not acquisition_crop_video_path.is_file():
            raise FileNotFoundError(
                f"Legacy acquisition crop video not found: {acquisition_crop_video_path}"
            )
        roi_shape = _resolve_roi_shape(acquisition_group)
    (
        refined_group,
        resolved_refined_run,
        refined_selection_mode,
        refined_payload,
        refined_manifest_digest,
        refined_logical_content_digest,
    ) = _resolve_refined_input(
        archive_path=archive_path,
        root=root,
        run_name=refined_detect_run,
    )
    frame_height, frame_width = resolve_full_frame_shape(root)
    full_video_path = _resolve_source_video_path(
        root=root,
        recording_dir=resolved_recording_dir,
        explicit=Path(source_video_path) if source_video_path is not None else None,
    )
    if acquisition_source_mode == "canonical_ledger":
        full_video_identity = _bound_stat_media_identity(
            path=full_video_path,
            attrs=root.attrs,
            label="Full-frame source video",
        )
        crop_video_identity = _bound_stat_media_identity(
            path=acquisition_crop_video_path,
            attrs=acquisition_group.attrs,
            label="Acquisition crop video",
        )
        payload = _prepare_canonical_ledger_hybrid_payload(
            root=root,
            ledger_group=acquisition_group,
            refined_payload=refined_payload,
            frame_width=frame_width,
            frame_height=frame_height,
            roi_shape=roi_shape,
            context_margin_px=float(context_margin_px),
        )
        routing_policy_id = PRODUCTION_ROUTING_POLICY_ID
        crop_policy_id = PRODUCTION_CROP_POLICY_ID
    else:
        full_video_identity = {"path": str(full_video_path), "strategy": "legacy_unbound"}
        crop_video_identity = {
            "path": str(acquisition_crop_video_path),
            "strategy": "legacy_unbound",
        }
        payload = _prepare_hybrid_payload(
            root=root,
            acquisition_group=acquisition_group,
            refined_payload=refined_payload,
            frame_width=frame_width,
            frame_height=frame_height,
            roi_shape=roi_shape,
        )
        legacy_pixel_kinds = np.asarray(
            payload["source_pixel_kind_codes"], dtype=np.int8
        )
        payload["routing_reason_codes"] = np.where(
            legacy_pixel_kinds
            == SOURCE_PIXEL_KIND_CODE_MAP["acquisition_crop_video"],
            ROUTING_REASON_CODE_MAP["acquisition_crop_selected"],
            ROUTING_REASON_CODE_MAP["legacy_instance_key_not_matched"],
        ).astype(np.int8)
        routing_policy_id = "legacy_instance_key_match_then_bbox_containment_v1"
        crop_policy_id = "legacy_acquisition_crop_extent"
    summary = dict(payload.pop("summary"))  # type: ignore[arg-type]
    offline_rows = np.flatnonzero(
        np.asarray(payload["source_pixel_kind_codes"], dtype=np.int8)
        == SOURCE_PIXEL_KIND_CODE_MAP["offline_full_frame_supplemental_flat_cache"]
    )
    hybrid_pixel_contract = orange_mono_pynvvc_luma_hybrid_pixel_contract()
    provider_rowset_record = {
        "schema_id": "palette.roi_pixel_provider_record.v1",
        "schema_version": 1,
        "crop_run": str(resolved_run_name),
        "acquisition_source_mode": acquisition_source_mode,
        "acquisition_source_run": str(resolved_acquisition_run),
        "acquisition_ledger_record_sha256": resolved_acquisition_record_sha256,
        "refined_detect_run": str(resolved_refined_run),
        "refined_selection_mode": refined_selection_mode,
        "refined_manifest_digest": refined_manifest_digest,
        "refined_logical_content_digest": refined_logical_content_digest,
        "full_frame_media_identity": full_video_identity,
        "acquisition_crop_media_identity": crop_video_identity,
        "routing_policy_id": routing_policy_id,
        "crop_policy_id": crop_policy_id,
        "context_margin_px": float(context_margin_px),
        "roi_shape": [int(roi_shape[0]), int(roi_shape[1])],
        "frame_shape": [int(frame_height), int(frame_width)],
        "row_count": int(np.asarray(payload["instance_key"]).shape[0]),
        "rowset_array_sha256": {
            name: _array_sha256(np.asarray(payload[name]))
            for name in (
                "instance_key",
                "frame_indices",
                "source_refined_row_ids",
                "source_acquisition_crop_row_indices",
                "source_acquisition_crop_xywh",
                "source_crop_video_frame_indices",
                "source_crop_xywh",
                "source_pixel_kind_codes",
                "routing_reason_codes",
                "roi_coordinates_full",
                "roi_sizes_full",
                "bbox_norm_coords",
                "crop_state_codes",
                "supplemental_cache_row_indices",
            )
        },
    }
    provider_record_sha256 = _canonical_json_sha256(provider_rowset_record)
    provider_identity = build_hybrid_crop_provider_identity(
        payload,
        provider_record_sha256=provider_record_sha256,
        routing_policy_id=routing_policy_id,
        crop_policy_id=crop_policy_id,
        pixel_contract=hybrid_pixel_contract,
    )
    payload["source_row_signature"] = provider_identity.row_signatures.signatures

    plan = {
        "status": "dry_run" if not apply else "planned",
        "zarr_path": str(archive_path),
        "recording_dir": str(resolved_recording_dir),
        "source_video_path": str(full_video_path),
        "source_crop_video_path": str(acquisition_crop_video_path),
        "acquisition_source_mode": acquisition_source_mode,
        "acquisition_crop_run": str(resolved_acquisition_run),
        "acquisition_ledger_record_sha256": resolved_acquisition_record_sha256,
        "refined_detect_run": str(resolved_refined_run),
        "refined_selection_mode": refined_selection_mode,
        "refined_manifest_digest": refined_manifest_digest,
        "refined_logical_content_digest": refined_logical_content_digest,
        "target_crop_run": str(resolved_run_name),
        "supplemental_manifest_path": str(manifest_path),
        "roi_shape": [int(roi_shape[0]), int(roi_shape[1])],
        "frame_shape": [int(frame_height), int(frame_width)],
        "decode_mode_requested": str(decode_mode),
        "decode_chunk_frames": int(decode_chunk_frames),
        "routing_policy_id": routing_policy_id,
        "crop_policy_id": crop_policy_id,
        "context_margin_px": float(context_margin_px),
        "provider_record_sha256": provider_record_sha256,
        "source_row_signature_spec_digest": (
            provider_identity.row_signatures.spec.spec_digest
        ),
        "source_pixel_fingerprint": provider_identity.source_pixel_fingerprint,
        "source_rowset_fingerprint": provider_identity.source_rowset_fingerprint,
        "crop_signature": dict(provider_identity.crop_signature),
        "set_latest_any": bool(set_latest_any),
        "summary": summary,
    }
    if not apply:
        return plan

    if resolved_run_name in crop_parent:
        if not overwrite:
            raise FileExistsError(f"crop_runs/{resolved_run_name} already exists.")
        assert_crop_run_unreferenced(crop_parent, resolved_run_name)
        del crop_parent[resolved_run_name]

    supplemental_manifest: dict[str, Any] | None = None
    if offline_rows.size:
        supplemental_manifest = _write_supplemental_cache(
            zarr_path=archive_path,
            run_name=resolved_run_name,
            video_path=full_video_path,
            manifest_path=manifest_path,
            frame_indices=np.asarray(payload["frame_indices"], dtype=np.int64)[
                offline_rows
            ],
            roi_coordinates_full=np.asarray(
                payload["roi_coordinates_full"], dtype=np.int32
            )[offline_rows],
            roi_shape=roi_shape,
            video_shape=(int(frame_height), int(frame_width)),
            decode_mode=str(decode_mode),
            decode_chunk_frames=int(decode_chunk_frames),
            overwrite=bool(overwrite),
        )

    group = crop_parent.create_group(resolved_run_name)
    mark_run_started(group, run_name=resolved_run_name, stage="crop")
    started = time.perf_counter()
    try:
        array_names = [
            "instance_key",
            "frame_indices",
            "source_frame_indices",
            "source_acquisition_frame_index",
            "source_recording_frame_ids",
            "source_crop_meta_row_indices",
            "source_acquisition_crop_row_indices",
            "source_crop_video_frame_indices",
            "source_crop_local_frame_ids",
            "source_acquisition_crop_xywh",
            "source_crop_xywh",
            "roi_coordinates_full",
            "roi_sizes_full",
            "bbox_img_xyxy",
            "bbox_norm_coords",
            "bbox_roi_xyxy",
            "bbox_crop_norm_coords",
            "containment_margins_px",
            "detection_success",
            "detection_source",
            "source_pixel_kind_codes",
            "crop_state_codes",
            "routing_reason_codes",
            "supplemental_cache_row_indices",
            "source_refined_row_ids",
            "source_detect_row_index",
            "frame_counts",
            "frame_row_offsets",
            "detection_indices",
            "source_row_signature",
        ]
        for name in array_names:
            if name not in payload:
                continue
            _create_array(group, name, np.asarray(payload[name]))
        stamp_geometry_preload_attrs(group)

        now = _utc_now()
        attrs = {
            "schema_id": SCHEMA_ID,
            "crop_storage_mode": "geometry_only",
            "source_pixels": "hybrid_acquisition_crop_video_offline_supplement",
            "roi_pixel_provider": "hybrid_acquisition_crop_video_offline_supplement",
            "source_type": "hybrid_acquisition_crop_video_offline_supplement",
            "roi_size": [int(roi_shape[0]), int(roi_shape[1])],
            "roi_pixel_contract_name": hybrid_pixel_contract["name"],
            "roi_pixel_contract": hybrid_pixel_contract,
            "decode_backend": "pynvvc_luma",
            "decode_mode_requested": str(decode_mode),
            "source_video_path": str(full_video_path),
            "source_crop_video_path": str(acquisition_crop_video_path),
            "source_full_frame_media_identity": full_video_identity,
            "source_acquisition_crop_media_identity": crop_video_identity,
            "source_acquisition_mode": acquisition_source_mode,
            "source_acquisition_crop_run": (
                str(resolved_acquisition_run)
                if acquisition_source_mode == "legacy_crop_run"
                else None
            ),
            "source_acquisition_crop_ledger_run": (
                str(resolved_acquisition_run)
                if acquisition_source_mode == "canonical_ledger"
                else None
            ),
            "source_acquisition_crop_ledger_record_sha256": (
                resolved_acquisition_record_sha256
            ),
            "source_refined_detect_run": str(resolved_refined_run),
            "source_refined_run_id": str(resolved_refined_run),
            "source_refined_manifest_digest": refined_manifest_digest,
            "source_refined_logical_content_digest": (
                refined_logical_content_digest
            ),
            "source_refined_detection_selection_mode": refined_selection_mode,
            "source_registered_detection_gate_requirement": refined_group.attrs.get(
                "registered_detection_gate_requirement"
            ),
            "source_registered_detection_gate": refined_group.attrs.get(
                "registered_detection_gate"
            ),
            "supplemental_roi_cache_manifest": (
                str(manifest_path) if supplemental_manifest else None
            ),
            "source_pixel_kind_code_map": SOURCE_PIXEL_KIND_CODE_MAP,
            "crop_state_code_map": CROP_STATE_CODE_MAP,
            "routing_reason_code_map": ROUTING_REASON_CODE_MAP,
            "detection_source_code_map": DETECTION_SOURCE_CODE_MAP,
            "source_crop_video_frame_indices_semantics": "zero_based_frame_index_in_acquisition_crop_video_when_a_frame-ledger_row_exists_or_-1_when_missing",
            "source_acquisition_crop_row_indices_semantics": "row_index_in_source_acquisition_ledger_or_legacy_crop_run_or_-1_when_missing",
            "supplemental_cache_row_indices_semantics": "row_index_in_supplemental_flat_roi_cache_or_-1_for_acquisition_video_rows",
            "roi_coordinates_full_coordinate_space": "source_image_xy",
            "bbox_norm_coords_semantics": "bbox_xywh_normalized_to_full_frame",
            "bbox_img_xyxy_semantics": "bbox_xyxy_full_frame_pixels",
            "bbox_authority": "reviewed_refined_detection",
            "rowset_authority": "complete_reviewed_refined_detection_instances",
            "row_identity": "instance_key",
            "source_pixel_routing_policy": routing_policy_id,
            "crop_policy_id": crop_policy_id,
            "crop_policy_edge_handling": "translation_only_centered_roi_with_zero_fill_outside_native_extent",
            "routing_context_margin_px": float(context_margin_px),
            "routing_containment_boundary": "inclusive",
            "routing_decode_validation": "runtime_fail_closed_and_separate_canary_sample_v1",
            "provider_record": provider_rowset_record,
            "provider_record_sha256": provider_record_sha256,
            "provider_manifest_kind": "crop_run_attrs_and_row_aligned_arrays_v1",
            "stage_selector_eligible": False,
            "registry_activation": "deferred",
            "summary_statistics": summary,
            "created_at_utc": now,
            "status": "completed",
            "completed_at_utc": now,
            "duration_seconds": float(time.perf_counter() - started),
            **provider_identity.attrs(),
        }
        if supplemental_manifest is not None:
            attrs["supplemental_roi_cache_manifest_payload"] = supplemental_manifest
        group.attrs.update(attrs)

        git_info = get_git_info(Path(__file__).resolve().parents[3])
        env_info = get_environment_info(
            include_all_packages=False, disk_path=str(archive_path), collect_ip=False
        )
        provenance = build_stage_provenance(
            stage="crop",
            command=" ".join(sys.argv),
            created_at_utc=now,
            version=git_info.get("short_hash") or git_info.get("commit_hash"),
            git=git_info,
            environment=env_info.get("environment"),
            platform=env_info.get("platform"),
            parameters={
                "run_name": resolved_run_name,
                "decode_mode": str(decode_mode),
                "acquisition_source_mode": acquisition_source_mode,
                "routing_policy_id": routing_policy_id,
                "crop_policy_id": crop_policy_id,
                "context_margin_px": float(context_margin_px),
                "set_latest_any": bool(set_latest_any),
            },
            inputs={
                "zarr_path": str(archive_path),
                "source_video_path": str(full_video_path),
                "source_acquisition_run": str(resolved_acquisition_run),
                "source_acquisition_ledger_record_sha256": (
                    resolved_acquisition_record_sha256
                ),
                "source_refined_detect_run": str(resolved_refined_run),
            },
            artifacts={
                "run_path": f"crop_runs/{resolved_run_name}",
                "supplemental_roi_cache_manifest": (
                    str(manifest_path) if supplemental_manifest else None
                ),
            },
        )
        write_stage_provenance(group, provenance)
        mark_run_complete(
            group,
            parent_group=crop_parent,
            run_name=resolved_run_name,
            run_provenance=build_run_provenance_from_stage_record(provenance),
        )
        if set_latest_any:
            crop_parent.attrs["latest_any"] = resolved_run_name
            crop_parent.attrs["latest_hybrid_acquisition_offline"] = resolved_run_name
        consolidation = consolidate_metadata_capture_expected_warnings(archive_path)
        consolidated_root = zarr.open_group(
            str(archive_path), mode="r", use_consolidated=True
        )
        if (
            "crop_runs" not in consolidated_root
            or resolved_run_name not in consolidated_root["crop_runs"]
        ):
            raise RuntimeError(
                "Consolidated metadata does not expose the completed hybrid crop run."
            )
        consolidated_run = consolidated_root["crop_runs"][resolved_run_name]
        if (
            consolidated_run.attrs.get("provider_record_sha256")
            != provider_record_sha256
        ):
            raise RuntimeError(
                "Consolidated metadata exposes a stale hybrid provider record."
            )
        if (
            consolidated_run.attrs.get("source_rowset_fingerprint")
            != provider_identity.source_rowset_fingerprint
        ):
            raise RuntimeError(
                "Consolidated metadata exposes a stale hybrid signed rowset."
            )
        return {
            **plan,
            "status": "ok",
            "applied": True,
            "host": socket.gethostname(),
            "pid": int(os.getpid()),
            "supplemental_cache": supplemental_manifest,
            "metadata_consolidation": consolidation,
        }
    except Exception as exc:
        mark_run_failed(
            group,
            parent_group=crop_parent,
            run_name=resolved_run_name,
            error=str(exc),
        )
        try:
            consolidate_metadata_capture_expected_warnings(archive_path)
        except Exception:
            pass
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a hybrid crop run that uses acquisition crop-video rows plus a "
            "supplemental flat cache. The default production route joins the "
            "canonical acquisition ledger by recording frame."
        )
    )
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument(
        "--acquisition-source-mode",
        choices=ACQUISITION_SOURCE_MODES,
        default="canonical_ledger",
        help=(
            "Use the pointer-selected canonical raw ledger (default), or explicitly "
            "request the historical instance-key crop-run compatibility route."
        ),
    )
    parser.add_argument(
        "--acquisition-ledger-record-sha256",
        help="Optional exact expected canonical ledger record digest.",
    )
    parser.add_argument(
        "--acquisition-crop-run",
        help=(
            "Existing acquisition crop-video crop run; valid only with "
            "--acquisition-source-mode legacy_crop_run."
        ),
    )
    parser.add_argument(
        "--refined-detect-run",
        help="Refined detect run to recover offline-only frames.",
    )
    parser.add_argument("--run-name", help="Output crop_runs/<run> name.")
    parser.add_argument(
        "--recording-dir",
        type=Path,
        help="Recording directory used to resolve cams/*.mp4.",
    )
    parser.add_argument(
        "--source-video-path",
        type=Path,
        help="Full camera source MP4 for supplemental rows.",
    )
    parser.add_argument(
        "--supplemental-manifest-path",
        type=Path,
        help="Output flat-cache manifest path.",
    )
    parser.add_argument("--decode-mode", choices=DECODE_MODES, default="auto")
    parser.add_argument("--decode-chunk-frames", type=int, default=1)
    parser.add_argument(
        "--context-margin-px",
        type=float,
        default=PRODUCTION_CONTEXT_MARGIN_PX,
        help=(
            "Required inclusive reviewed-bbox margin inside an acquisition crop. "
            "The frozen v1 policy default is 0 pixels."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--set-latest-any", action="store_true")
    parser.add_argument(
        "--apply", action="store_true", help="Write the cache and crop run."
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = build_hybrid_acquisition_offline_crop_run(
        args.zarr_path,
        acquisition_source_mode=args.acquisition_source_mode,
        acquisition_ledger_record_sha256=args.acquisition_ledger_record_sha256,
        acquisition_crop_run=args.acquisition_crop_run,
        refined_detect_run=args.refined_detect_run,
        run_name=args.run_name,
        recording_dir=args.recording_dir,
        source_video_path=args.source_video_path,
        supplemental_manifest_path=args.supplemental_manifest_path,
        decode_mode=args.decode_mode,
        decode_chunk_frames=args.decode_chunk_frames,
        context_margin_px=args.context_margin_px,
        overwrite=args.overwrite,
        set_latest_any=args.set_latest_any,
        apply=args.apply,
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
            f"target_crop_run: {report['target_crop_run']}\n"
            f"supplemental_manifest_path: {report['supplemental_manifest_path']}\n"
            "supplemental_rows_materialized: "
            f"{report['summary']['supplemental_rows_materialized']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
