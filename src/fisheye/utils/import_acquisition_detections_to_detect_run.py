"""Import acquisition crop-recorder boxes as an unbound detection artifact.

The external CSV does not yet have a coordinate-lineage record supported by
the canonical detection loader.  This importer therefore preserves its useful
geometry without publishing or selecting a normal ``detect_runs`` child.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
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
from fisheye.shared.run_provenance import build_run_provenance_from_stage_record
from fisheye.shared.detection_producer_lifecycle import (
    DETECTION_ARTIFACT_RUN_FAMILY,
    DetectionProducerAttempt,
    UNBOUND_ARTIFACT_RUN_BINDING_KEY,
    UNBOUND_DETECTION_ARTIFACT_COORDINATE_CONTRACT,
    build_unbound_artifact_run_binding,
    publish_artifact_payload_inventory_seal,
    publish_empty_artifact_observation_proof,
    stamp_unbound_artifact_numeric_semantics,
)
from fisheye.shared.stage_provenance import build_stage_provenance, write_stage_provenance
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
    output_parent: str
    coordinate_contract: str
    stage_selector_eligible: bool
    applied: bool


def utc_run_name(prefix: str = DEFAULT_RUN_PREFIX) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    return f"{prefix}_{stamp}"


def _open_root(zarr_path: Path, *, mode: str) -> zarr.Group:
    return zarr.open_group(str(zarr_path), mode=mode, use_consolidated=False)


def _close_store(node: Any) -> None:
    close = getattr(getattr(node, "store", None), "close", None)
    if callable(close):
        close()


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
        raise ValueError(
            f"External detection import requires recording_manifest.json: {manifest_path}"
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Recording manifest is not valid JSON: {manifest_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Recording manifest must be a JSON object.")
    return payload


def _manifest_stream(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    video_streams = manifest.get("video_streams")
    if not isinstance(video_streams, dict):
        return {}
    streams = video_streams.get("streams")
    if not isinstance(streams, dict):
        return {}
    stream = streams.get(name)
    return stream if isinstance(stream, dict) else {}


def _legacy_stream_int(stream: dict[str, Any], *keys: str) -> Optional[int]:
    """Compatibility coercion used only by the legacy crop-run builder."""

    for key in keys:
        value = stream.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except Exception:
            continue
    return None


def _strict_stream_int(
    stream: dict[str, Any],
    key: str,
    *,
    label: str,
    positive: bool = True,
) -> int:
    value = stream.get(key)
    if type(value) is not int:
        raise ValueError(
            f"Recording manifest {label} must be an exact JSON integer."
        )
    if positive and value <= 0:
        raise ValueError(f"Recording manifest {label} must be positive.")
    return value


def resolve_source_dimensions(
    root: zarr.Group,
    *,
    recording_dir: Path,
    source_width: Optional[int],
    source_height: Optional[int],
) -> tuple[int, int]:
    # Legacy helper retained for the acquisition-crop builder.  The detection
    # artifact importer below deliberately does not call it.
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
        width = width or _attr_int(
            raw_video.attrs,
            "source_video_width",
            "source_full_width",
            "width",
            "video_width",
        )
        height = height or _attr_int(
            raw_video.attrs,
            "source_video_height",
            "source_full_height",
            "height",
            "video_height",
        )
    if width is None or height is None:
        manifest = _load_recording_manifest(recording_dir)
        full_stream = _manifest_stream(manifest, "full")
        width = width or _legacy_stream_int(
            full_stream,
            "width",
            "source_width",
        )
        height = height or _legacy_stream_int(
            full_stream,
            "height",
            "source_height",
        )
    if width is None or height is None or width <= 0 or height <= 0:
        raise ValueError(
            "Could not resolve source full-frame dimensions; pass --source-width "
            "and --source-height."
        )
    return int(width), int(height)


@dataclass(frozen=True)
class _ExternalCropManifestAuthority:
    recording_id: str
    camera_id: str
    width: int
    height: int
    total_frames: int
    crop_frame_count: int
    crop_metadata_ref: str
    crop_stream: dict[str, Any]


@dataclass(frozen=True)
class _RawCropMetaAuthority:
    recording_frame_ids: np.ndarray
    has_detection: np.ndarray
    blank_frame: np.ndarray
    crop_xywh: np.ndarray
    detection_xywh: np.ndarray
    detection_confidence: np.ndarray


def _required_text(value: Any, *, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(
            f"Recording manifest must declare {label} as an exact unpadded "
            "nonempty string."
        )
    return value


def _resolve_external_crop_manifest_authority(
    root: zarr.Group,
    *,
    recording_dir: Path,
    crop_meta_path: Path,
    source_width: Optional[int],
    source_height: Optional[int],
) -> _ExternalCropManifestAuthority:
    manifest = _load_recording_manifest(recording_dir)
    streams = manifest.get("video_streams")
    if not isinstance(streams, dict) or streams.get("schema_id") != (
        "orange_runtime_video_streams_v1"
    ):
        raise ValueError(
            "Recording manifest lacks orange_runtime_video_streams_v1 authority."
        )
    if streams.get("frame_clock") != "recording_frame_id":
        raise ValueError("Video-stream authority must use recording_frame_id.")
    full_stream = _manifest_stream(manifest, "full")
    crop_stream = _manifest_stream(manifest, "crop")
    width = _strict_stream_int(
        full_stream,
        "width",
        label="full stream width",
    )
    height = _strict_stream_int(
        full_stream,
        "height",
        label="full stream height",
    )
    for label, supplied, expected in (
        ("source_width", source_width, width),
        ("source_height", source_height, height),
    ):
        if supplied is not None and type(supplied) is not int:
            raise ValueError(f"Explicit {label} must be an exact integer assertion.")
        if supplied is not None and supplied != expected:
            raise ValueError(
                f"Explicit {label}={supplied} disagrees with recording-manifest "
                f"authority {expected}."
            )
    total_frames = _strict_stream_int(
        full_stream,
        "frame_count",
        label="full stream frame_count",
    )
    crop_frame_count = _strict_stream_int(
        crop_stream,
        "frame_count",
        label="crop stream frame_count",
    )
    if crop_frame_count != total_frames:
        raise ValueError(
            "Recording manifest crop stream frame_count must exactly equal the "
            "full stream frame_count."
        )
    recording_id = _required_text(manifest.get("recording_id"), label="recording_id")
    full_camera = _required_text(full_stream.get("camera_id"), label="full camera_id")
    crop_camera = _required_text(crop_stream.get("camera_id"), label="crop camera_id")
    root_camera = root.attrs.get("camera_id")
    root_recording = root.attrs.get("recording_id")
    if crop_camera != full_camera:
        raise ValueError("Full and crop streams declare different camera_id values.")
    if root_camera is not None and (
        type(root_camera) is not str or root_camera != full_camera
    ):
        raise ValueError("Zarr camera_id disagrees with recording-manifest authority.")
    if root_recording is not None and (
        type(root_recording) is not str or root_recording != recording_id
    ):
        raise ValueError("Zarr recording_id disagrees with recording-manifest authority.")
    for label, stream in (("full", full_stream), ("crop", crop_stream)):
        if stream.get("frame_clock") != "recording_frame_id":
            raise ValueError(
                f"Recording manifest {label} stream must use recording_frame_id."
            )
    if full_stream.get("coordinate_space") != "full_frame_pixels":
        raise ValueError(
            "Full-stream coordinate_space must explicitly be full_frame_pixels."
        )
    if crop_stream.get("source_geometry_coordinate_space") != "full_frame_pixels":
        raise ValueError(
            "Crop metadata source geometry must explicitly be full_frame_pixels."
        )
    if crop_stream.get("video_pixel_coordinate_space") != "crop_frame_pixels":
        raise ValueError(
            "Crop video pixels must explicitly be crop_frame_pixels."
        )
    required_columns = {
        "crop_x",
        "crop_y",
        "crop_w",
        "crop_h",
        "detection_x",
        "detection_y",
        "detection_w",
        "detection_h",
    }
    columns = crop_stream.get("geometry_columns")
    if not isinstance(columns, list) or not required_columns.issubset(
        {str(value) for value in columns}
    ):
        raise ValueError(
            "Crop stream must explicitly declare all crop/detection geometry columns."
        )
    metadata_ref = _required_text(
        crop_stream.get("metadata"), label="crop metadata path"
    )
    declared_path = Path(metadata_ref)
    if not declared_path.is_absolute():
        declared_path = recording_dir / declared_path
    if declared_path.resolve() != crop_meta_path.resolve():
        raise ValueError(
            "Resolved crop metadata does not match the exact recording-manifest path."
        )
    return _ExternalCropManifestAuthority(
        recording_id=recording_id,
        camera_id=full_camera,
        width=width,
        height=height,
        total_frames=int(total_frames),
        crop_frame_count=int(crop_frame_count),
        crop_metadata_ref=metadata_ref,
        crop_stream=dict(crop_stream),
    )


def _load_raw_crop_meta_authority(
    crop_meta_path: Path,
    *,
    expected_count: int,
) -> _RawCropMetaAuthority:
    """Validate raw CSV identity and selection flags before normalization."""

    raw_ids: list[int] = []
    raw_has_detection: list[bool] = []
    raw_blank_frame: list[bool] = []
    raw_crop_xywh: list[tuple[float, float, float, float]] = []
    raw_detection_xywh: list[tuple[float, float, float, float]] = []
    raw_detection_confidence: list[float] = []

    def exact_float_token(
        row: dict[str, Any],
        field_name: str,
        *,
        row_index: int,
        allow_blank: bool,
    ) -> float:
        raw_value = row.get(field_name)
        if type(raw_value) is not str or raw_value != raw_value.strip():
            raise ValueError(
                f"Crop metadata {field_name} must be an exact unpadded numeric "
                f"token; row {row_index} is {raw_value!r}."
            )
        if raw_value == "":
            if allow_blank:
                return float("nan")
            raise ValueError(
                f"Crop metadata {field_name} cannot be blank at row {row_index}."
            )
        try:
            return float(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"Crop metadata {field_name} is not numeric at row {row_index}: "
                f"{raw_value!r}."
            ) from exc

    with crop_meta_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "recording_frame_id",
            "has_detection",
            "blank_frame",
            "detection_confidence",
            "crop_x",
            "crop_y",
            "crop_w",
            "crop_h",
            "detection_x",
            "detection_y",
            "detection_w",
            "detection_h",
        }
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(
                "Crop metadata is missing raw identity/selection columns: "
                f"{missing!r}."
            )
        for row_index, row in enumerate(reader):
            raw_value = row.get("recording_frame_id")
            if (
                type(raw_value) is not str
                or not raw_value.isascii()
                or not raw_value.isdecimal()
            ):
                raise ValueError(
                    "Crop metadata recording_frame_id values must be exact "
                    f"positive base-10 integers; row {row_index} is {raw_value!r}."
                )
            value = int(raw_value)
            if value <= 0 or raw_value != str(value):
                raise ValueError(
                    "Crop metadata recording_frame_id values must use the exact "
                    "canonical positive base-10 token; "
                    f"row {row_index} is {raw_value!r}."
                )
            raw_ids.append(value)
            for field_name, output in (
                ("has_detection", raw_has_detection),
                ("blank_frame", raw_blank_frame),
            ):
                raw_flag = row.get(field_name)
                if type(raw_flag) is not str or raw_flag not in {"0", "1"}:
                    raise ValueError(
                        f"Crop metadata {field_name} must be exactly integer 0 or 1; "
                        f"row {row_index} is {raw_flag!r}."
                    )
                output.append(raw_flag == "1")
            crop_values = tuple(
                exact_float_token(
                    row,
                    field_name,
                    row_index=row_index,
                    allow_blank=False,
                )
                for field_name in ("crop_x", "crop_y", "crop_w", "crop_h")
            )
            if not np.isfinite(crop_values).all():
                raise ValueError(
                    f"Crop metadata crop placement must be finite at row {row_index}."
                )
            raw_crop_xywh.append(crop_values)
            detection_values = tuple(
                exact_float_token(
                    row,
                    field_name,
                    row_index=row_index,
                    allow_blank=True,
                )
                for field_name in (
                    "detection_x",
                    "detection_y",
                    "detection_w",
                    "detection_h",
                )
            )
            confidence = exact_float_token(
                row,
                "detection_confidence",
                row_index=row_index,
                allow_blank=True,
            )
            if (
                raw_has_detection[-1]
                and not raw_blank_frame[-1]
                and (
                    not np.isfinite(detection_values).all()
                    or not np.isfinite(confidence)
                )
            ):
                raise ValueError(
                    "Every raw eligible detection row must contain finite detection "
                    f"geometry and confidence; row {row_index} would become false-empty."
                )
            raw_detection_xywh.append(detection_values)
            raw_detection_confidence.append(confidence)

    expected = list(range(1, int(expected_count) + 1))
    if len(raw_ids) != len(set(raw_ids)):
        raise ValueError("Crop metadata recording_frame_id values are duplicated.")
    if len(raw_ids) != int(expected_count):
        raise ValueError(
            "Crop metadata recording_frame_id domain is incomplete: expected "
            f"{expected_count} ordered rows, found {len(raw_ids)}."
        )
    if raw_ids != expected:
        raise ValueError(
            "Crop metadata recording_frame_id values are permuted or outside the "
            "exact ordered 1..frame_count domain."
        )
    return _RawCropMetaAuthority(
        recording_frame_ids=np.asarray(raw_ids, dtype=np.int64),
        has_detection=np.asarray(raw_has_detection, dtype=bool),
        blank_frame=np.asarray(raw_blank_frame, dtype=bool),
        crop_xywh=np.asarray(raw_crop_xywh, dtype=np.float64).reshape(-1, 4),
        detection_xywh=np.asarray(raw_detection_xywh, dtype=np.float64).reshape(-1, 4),
        detection_confidence=np.asarray(
            raw_detection_confidence,
            dtype=np.float64,
        ),
    )


def _validate_external_rows(
    *,
    authority: _ExternalCropManifestAuthority,
    raw_crop_meta: _RawCropMetaAuthority,
    detections: Any,
    crop_rows: Any,
) -> None:
    expected_frames = np.arange(authority.total_frames, dtype=np.int64)
    raw_ids = np.asarray(
        raw_crop_meta.recording_frame_ids,
        dtype=np.int64,
    ).reshape(-1)
    if not np.array_equal(raw_ids, expected_frames + 1):
        raise ValueError(
            "Raw recording_frame_id values do not match the exact manifest domain."
        )
    crop_frames = np.asarray(crop_rows.frame_indices)
    if (
        crop_frames.dtype != np.dtype("int64")
        or crop_frames.shape != expected_frames.shape
        or not np.array_equal(crop_frames, expected_frames)
    ):
        raise ValueError(
            "Crop metadata must provide exact int64 zero-based recording_frame_id "
            "rows for every declared full-stream frame without coercion."
        )
    crop_row_indices = np.asarray(crop_rows.row_indices)
    if (
        crop_row_indices.dtype != np.dtype("int64")
        or crop_row_indices.shape != expected_frames.shape
        or not np.array_equal(crop_row_indices, expected_frames)
    ):
        raise ValueError(
            "Shared crop metadata loader changed the exact int64 source row identity."
        )
    crop_xywh = np.asarray(crop_rows.crop_xywh)
    if (
        crop_xywh.dtype != np.dtype("float64")
        or crop_xywh.shape != (authority.total_frames, 4)
        or not np.isfinite(crop_xywh).all()
    ):
        raise ValueError(
            "Every crop_xywh row must remain exact float64, finite, and row-aligned."
        )
    if not np.array_equal(crop_xywh, raw_crop_meta.crop_xywh):
        raise ValueError(
            "Shared crop metadata loader changed exact raw float64 crop placement."
        )
    has_detection = np.asarray(crop_rows.has_detection)
    blank_frame = np.asarray(crop_rows.blank_frame)
    if (
        has_detection.shape != (authority.total_frames,)
        or blank_frame.shape != (authority.total_frames,)
        or has_detection.dtype != np.dtype(bool)
        or blank_frame.dtype != np.dtype(bool)
    ):
        raise ValueError(
            "Crop metadata has_detection and blank_frame must be exact boolean "
            "vectors aligned to the manifest frame domain."
        )
    if not np.array_equal(has_detection, raw_crop_meta.has_detection) or not (
        np.array_equal(blank_frame, raw_crop_meta.blank_frame)
    ):
        raise ValueError(
            "Shared crop metadata loader changed exact raw has_detection or "
            "blank_frame selection flags."
        )
    zero_crop_sentinel = np.all(crop_xywh == 0.0, axis=1)
    sentinel_allowed = blank_frame & ~has_detection
    if np.any(zero_crop_sentinel & ~sentinel_allowed):
        raise ValueError(
            "An all-zero crop_xywh sentinel is permitted only on an explicit "
            "blank no-detection row."
        )
    non_sentinel = ~zero_crop_sentinel
    crop_rectangles = crop_xywh[non_sentinel]
    if crop_rectangles.shape[0] and (
        np.any(crop_rectangles[:, 0] < 0.0)
        or np.any(crop_rectangles[:, 1] < 0.0)
        or np.any(crop_rectangles[:, 2] <= 0.0)
        or np.any(crop_rectangles[:, 3] <= 0.0)
        or np.any(
            crop_rectangles[:, 0] + crop_rectangles[:, 2]
            > float(authority.width)
        )
        or np.any(
            crop_rectangles[:, 1] + crop_rectangles[:, 3]
            > float(authority.height)
        )
    ):
        raise ValueError(
            "Every non-sentinel crop_xywh row must have a nonnegative origin, "
            "positive extent, and fit inside the exact manifest full-frame dimensions."
        )
    expected_detection_frames = np.flatnonzero(
        raw_crop_meta.has_detection & ~raw_crop_meta.blank_frame
    ).astype(np.int64, copy=False)
    raw_detection_frames = np.asarray(detections.frame_indices)
    if (
        raw_detection_frames.ndim != 1
        or raw_detection_frames.dtype != np.dtype("int64")
        or raw_detection_frames.shape != expected_detection_frames.shape
        or not np.array_equal(raw_detection_frames, expected_detection_frames)
    ):
        raise ValueError(
            "Detection frame_indices must exactly equal "
            "flatnonzero(raw has_detection & ~raw blank_frame), including dtype, "
            "value, order, and cardinality. Every raw eligible row must survive "
            "geometry loading, and no ineligible row may be added."
        )
    detection_frames = raw_detection_frames.astype(np.int64, copy=False)
    if np.any(detection_frames < 0) or np.any(
        detection_frames >= authority.total_frames
    ):
        raise ValueError("Detection rows fall outside the declared acquisition frame domain.")
    if detection_frames.size > 1 and np.any(np.diff(detection_frames) <= 0):
        raise ValueError(
            "Detection frame_indices must be strictly increasing and unique."
        )
    raw_detection_row_indices = np.asarray(detections.row_indices)
    if (
        raw_detection_row_indices.ndim != 1
        or raw_detection_row_indices.dtype != np.dtype("int64")
    ):
        raise ValueError(
            "Detection row_indices must be an exact int64 rank-1 array."
        )
    detection_row_indices = raw_detection_row_indices
    if detection_row_indices.shape != detection_frames.shape:
        raise ValueError(
            "Detection row_indices count must match detection frame_indices."
        )
    if np.any(detection_row_indices < 0) or np.any(
        detection_row_indices >= authority.total_frames
    ):
        raise ValueError(
            "Detection row_indices fall outside the exact crop metadata row domain."
        )
    if not np.array_equal(detection_row_indices, detection_frames):
        raise ValueError(
            "Detection row_indices must exactly equal frame_indices under the "
            "ordered one-row-per-recording-frame contract."
        )
    if detection_row_indices.size and np.any(
        zero_crop_sentinel[detection_row_indices]
    ):
        raise ValueError(
            "An all-zero blank-row crop sentinel cannot be persisted as detection geometry."
        )
    bbox = np.asarray(detections.bbox_img_xyxy)
    if (
        bbox.dtype != np.dtype("float64")
        or bbox.shape != (detection_frames.shape[0], 4)
        or not np.isfinite(bbox).all()
    ):
        raise ValueError(
            "External detection boxes must remain exact float64, finite, and row-aligned."
        )
    centers = np.asarray(detections.centers_xy)
    if (
        centers.dtype != np.dtype("float64")
        or centers.shape != (detection_frames.shape[0], 2)
        or not np.isfinite(centers).all()
    ):
        raise ValueError(
            "External detection centers must remain exact float64, finite, and row-aligned."
        )
    confidence = np.asarray(detections.confidence)
    if (
        confidence.dtype != np.dtype("float64")
        or confidence.shape != detection_frames.shape
        or not np.isfinite(confidence).all()
    ):
        raise ValueError(
            "External detection confidence must remain exact float64, finite, and "
            "row-aligned."
        )
    selected_xywh = raw_crop_meta.detection_xywh[expected_detection_frames]
    expected_bbox = np.column_stack(
        (
            selected_xywh[:, 0],
            selected_xywh[:, 1],
            selected_xywh[:, 0] + selected_xywh[:, 2],
            selected_xywh[:, 1] + selected_xywh[:, 3],
        )
    ).astype(np.float64, copy=False)
    expected_centers = np.column_stack(
        (
            selected_xywh[:, 0] + selected_xywh[:, 2] * 0.5,
            selected_xywh[:, 1] + selected_xywh[:, 3] * 0.5,
        )
    ).astype(np.float64, copy=False)
    expected_confidence = raw_crop_meta.detection_confidence[
        expected_detection_frames
    ]
    if not np.array_equal(bbox, expected_bbox):
        raise ValueError(
            "Shared crop metadata loader changed exact raw float64 detection boxes."
        )
    if not np.array_equal(centers, expected_centers):
        raise ValueError(
            "Shared crop metadata loader changed exact raw float64 detection centers."
        )
    if not np.array_equal(confidence, expected_confidence):
        raise ValueError(
            "Shared crop metadata loader changed exact raw float64 confidence."
        )
    if bbox.shape[0] and (
        np.any(bbox[:, 2] <= bbox[:, 0])
        or np.any(bbox[:, 3] <= bbox[:, 1])
        or np.any(bbox[:, 0] < 0.0)
        or np.any(bbox[:, 1] < 0.0)
        or np.any(bbox[:, 2] > float(authority.width))
        or np.any(bbox[:, 3] > float(authority.height))
    ):
        raise ValueError(
            "External detection boxes must have positive extents inside the exact "
            "manifest-declared full-frame dimensions."
        )


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_mapping_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _recording_frame_ids_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype=np.int64)
    digest = hashlib.sha256()
    digest.update(b"palette.recording_frame_ids.int64_le.v1\x00")
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.astype("<i8", copy=False).tobytes())
    return digest.hexdigest()


def _raw_selection_sha256(raw_crop_meta: _RawCropMetaAuthority) -> str:
    digest = hashlib.sha256()
    digest.update(b"palette.external_crop_raw_selection.v1\x00")
    for values, dtype in (
        (raw_crop_meta.recording_frame_ids, "<i8"),
        (raw_crop_meta.has_detection, "u1"),
        (raw_crop_meta.blank_frame, "u1"),
    ):
        array = np.ascontiguousarray(values).astype(dtype, copy=False)
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _stamp_import_artifact_semantics(
    run: zarr.Group,
    *,
    reference_node_path: str,
    reference_width: int,
    reference_height: int,
    source_frame_count: int,
    source_lineage_sha256: str,
    source_mapping_sha256: str,
) -> None:
    profiles = {
        "artifact_row_id": "import.artifact_row_id.v1",
        "frame_indices": "import.frame_indices.v1",
        "bbox_norm_coords": "import.bbox_norm_cxcywh.v1",
        "bbox_img_xyxy": "import.bbox_img_xyxy.v1",
        "centers_img_xy": "import.centers_img_xy.v1",
        "scores": "import.scores.v1",
        "class_ids": "import.class_ids.v1",
        "frame_counts": "import.frame_counts.v1",
        "n_detections": "import.n_detections.v1",
        "source_crop_xywh": "import.source_crop_xywh.v1",
        "source_crop_meta_row_indices": (
            "import.source_crop_meta_row_indices.v1"
        ),
        "source_recording_frame_ids": "import.source_recording_frame_ids.v1",
    }
    if set(run.keys()) != set(profiles):
        raise ValueError(
            "Acquisition-import artifact semantic inventory does not match live arrays."
        )
    for name, profile_id in profiles.items():
        stamp_unbound_artifact_numeric_semantics(
            run[name],
            semantic_profile_id=profile_id,
            reference_node_path=reference_node_path,
            reference_width=reference_width,
            reference_height=reference_height,
            source_frame_count=source_frame_count,
            source_sha256=source_lineage_sha256,
            source_mapping_sha256=(
                source_mapping_sha256 if name == "frame_indices" else None
            ),
        )


def _require_file_sha256(path: Path, *, expected: str, label: str) -> None:
    observed = _file_sha256(path)
    if observed != expected:
        raise ValueError(
            f"{label} changed while the detection artifact was being prepared; "
            "refusing to persist lineage for an unstable source."
        )


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
    artifact_only: bool = False,
) -> AcquisitionDetectImportResult:
    zarr_path = Path(zarr_path)
    if type(class_id) is not int or not (
        np.iinfo(np.int32).min <= class_id <= np.iinfo(np.int32).max
    ):
        raise ValueError("class_id must be an exact int32-representable integer.")
    if run_name is None:
        resolved_run_name = utc_run_name()
    else:
        if type(run_name) is not str:
            raise ValueError("run_name must be an exact string when provided.")
        resolved_run_name = run_name.strip()
        if (
            not resolved_run_name
            or "/" in resolved_run_name
            or resolved_run_name in {".", ".."}
        ):
            raise ValueError("run_name must normalize to one nonempty path segment.")
    if apply and not artifact_only:
        raise ValueError(
            "Acquisition crop-metadata geometry cannot publish a selectable "
            "canonical detect run. Pass artifact_only=True (CLI: --artifact-only) "
            "to write an explicit nonselector detection artifact."
        )
    if overwrite:
        raise ValueError(
            "Fail-closed detection publication does not overwrite immutable runs; "
            "choose a new --run-name."
        )
    resolved_recording_dir = Path(recording_dir) if recording_dir is not None else infer_recording_dir_from_zarr(zarr_path)
    resolved_crop_meta = resolve_crop_meta_path(
        zarr_path,
        recording_dir=resolved_recording_dir,
        crop_meta_path=crop_meta_path,
        required=True,
    )
    if resolved_crop_meta is None:
        raise ValueError("No crop metadata resolved.")

    manifest_path = resolved_recording_dir / "recording_manifest.json"
    crop_meta_sha256 = _file_sha256(resolved_crop_meta)
    manifest_sha256 = _file_sha256(manifest_path)
    root = _open_root(zarr_path, mode="r")
    try:
        authority = _resolve_external_crop_manifest_authority(
            root,
            recording_dir=resolved_recording_dir,
            crop_meta_path=resolved_crop_meta,
            source_width=source_width,
            source_height=source_height,
        )
    finally:
        _close_store(root)
    raw_crop_meta = _load_raw_crop_meta_authority(
        resolved_crop_meta,
        expected_count=authority.total_frames,
    )
    detections, crop_rows = load_crop_meta_realtime_detection_rows(
        resolved_crop_meta
    )
    _validate_external_rows(
        authority=authority,
        raw_crop_meta=raw_crop_meta,
        detections=detections,
        crop_rows=crop_rows,
    )
    _require_file_sha256(
        resolved_crop_meta,
        expected=crop_meta_sha256,
        label="Crop metadata",
    )
    _require_file_sha256(
        manifest_path,
        expected=manifest_sha256,
        label="Recording manifest",
    )
    width, height = authority.width, authority.height
    total_frames = authority.total_frames
    if total_frames - 1 > np.iinfo(np.int32).max:
        raise ValueError(
            "External acquisition frame domain cannot be represented exactly by "
            "the persisted int32 frame_indices array."
        )

    frame_indices = detections.frame_indices.astype(np.int32, copy=False)
    bbox_img_xyxy = detections.bbox_img_xyxy.astype(np.float64, copy=False)
    centers_img_xy = np.column_stack(
        (
            (bbox_img_xyxy[:, 0] + bbox_img_xyxy[:, 2]) * 0.5,
            (bbox_img_xyxy[:, 1] + bbox_img_xyxy[:, 3]) * 0.5,
        )
    ).astype(np.float64, copy=False)
    bbox_norm = _bbox_img_xyxy_to_norm_cxcywh(bbox_img_xyxy, width=width, height=height)
    scores = detections.confidence.astype(np.float32, copy=False)
    class_ids = np.full(frame_indices.shape, class_id, dtype=np.int32)
    frame_counts = np.bincount(frame_indices.astype(np.int64, copy=False), minlength=total_frames).astype(np.int32)
    source_crop_xywh = crop_rows.crop_xywh[np.searchsorted(crop_rows.frame_indices, detections.frame_indices)].astype(
        np.float64,
        copy=False,
    )
    source_crop_meta_row_indices = detections.row_indices.astype(np.int64, copy=False)
    source_recording_frame_ids = detections.frame_indices.astype(np.int64, copy=False) + 1
    artifact_row_id = np.arange(frame_indices.shape[0], dtype=np.uint64)
    recording_frame_ids_sha256 = _recording_frame_ids_sha256(
        raw_crop_meta.recording_frame_ids
    )
    manifest_full_stream_ref = (
        f"{manifest_path}#/video_streams/streams/full"
    )
    external_source_frame_evidence = {
        "schema_id": "palette.external_detection_source_frame_evidence.v1",
        "status": "unbound_source_evidence",
        "manifest_recording_id": authority.recording_id,
        "manifest_camera_id": authority.camera_id,
        "manifest_full_stream_ref": manifest_full_stream_ref,
        "reference_width": int(width),
        "reference_height": int(height),
        "frame_clock": "recording_frame_id",
        "source_domain": "ordered_positive_recording_frame_id",
        "target_domain": "zero_based_palette_frame_index",
        "direction": "recording_frame_id_to_palette_frame_index",
        "operation": "palette_frame_index = recording_frame_id - 1",
        "frame_count": authority.total_frames,
        "recording_frame_ids_sha256": recording_frame_ids_sha256,
        "crop_meta_sha256": crop_meta_sha256,
        "recording_manifest_sha256": manifest_sha256,
        UNBOUND_ARTIFACT_RUN_BINDING_KEY: build_unbound_artifact_run_binding(
            manifest_id="acquisition_detection_import.v1",
            reference_node_path=manifest_full_stream_ref,
            reference_width=int(width),
            reference_height=int(height),
            source_frame_count=int(total_frames),
            source_mapping_sha256=recording_frame_ids_sha256,
        ),
    }
    external_source_frame_evidence_sha256 = _canonical_mapping_sha256(
        external_source_frame_evidence
    )
    run_path = f"{DETECTION_ARTIFACT_RUN_FAMILY}/{resolved_run_name}"

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
            output_parent=DETECTION_ARTIFACT_RUN_FAMILY,
            coordinate_contract=UNBOUND_DETECTION_ARTIFACT_COORDINATE_CONTRACT,
            stage_selector_eligible=False,
            applied=False,
        )

    root = _open_root(zarr_path, mode="a")
    attempt: DetectionProducerAttempt | None = None
    try:
        attempt = DetectionProducerAttempt.begin_unbound_artifact(
            root,
            run_name=resolved_run_name,
            semantic_manifest_id="acquisition_detection_import.v1",
            strict_integrity_required=True,
        )
        run = attempt.run
        _create_array(run, "artifact_row_id", artifact_row_id)
        _create_array(run, "frame_indices", frame_indices)
        _create_array(run, "bbox_norm_coords", bbox_norm)
        _create_array(run, "bbox_img_xyxy", bbox_img_xyxy)
        _create_array(run, "centers_img_xy", centers_img_xy)
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
        manifest_attrs = dict(authority.crop_stream)
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
                "source_geometry_status": "unbound_external_artifact",
                "source_geometry_reason": (
                    "external crop metadata lacks a canonical persisted lineage "
                    "record supported by the detection coordinate loader"
                ),
                "source_artifact_path": str(resolved_crop_meta),
                "source_artifact_sha256": crop_meta_sha256,
                "source_manifest_path": str(manifest_path),
                "source_manifest_sha256": manifest_sha256,
                "source_recording_dir": str(resolved_recording_dir),
                "source_frame_clock": "recording_frame_id",
                "external_source_frame_evidence": external_source_frame_evidence,
                "external_source_frame_evidence_sha256": (
                    external_source_frame_evidence_sha256
                ),
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
        _stamp_import_artifact_semantics(
            run,
            reference_node_path=manifest_full_stream_ref,
            reference_width=int(width),
            reference_height=int(height),
            source_frame_count=int(total_frames),
            source_lineage_sha256=external_source_frame_evidence_sha256,
            source_mapping_sha256=recording_frame_ids_sha256,
        )
        if frame_indices.shape[0] == 0:
            publish_empty_artifact_observation_proof(
                run,
                source_frame_count=int(total_frames),
                row_array_names=(
                    "artifact_row_id",
                    "frame_indices",
                    "bbox_norm_coords",
                    "bbox_img_xyxy",
                    "centers_img_xy",
                    "scores",
                    "class_ids",
                    "source_crop_xywh",
                    "source_crop_meta_row_indices",
                    "source_recording_frame_ids",
                ),
                full_domain_evidence={
                    "coverage_status": "full_source_domain_validated",
                    "source_frame_count": int(total_frames),
                    "selection_rule": (
                        "flatnonzero(raw_has_detection_and_not_raw_blank_frame)"
                    ),
                    "raw_selection_sha256": _raw_selection_sha256(raw_crop_meta),
                    "eligible_observation_count": 0,
                    "crop_meta_sha256": crop_meta_sha256,
                    "recording_manifest_sha256": manifest_sha256,
                    "external_source_frame_evidence_sha256": (
                        external_source_frame_evidence_sha256
                    ),
                },
            )
        artifact_payload_seal = publish_artifact_payload_inventory_seal(
            run,
            source_frame_count=int(total_frames),
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
                "crop_meta_sha256": crop_meta_sha256,
                "recording_manifest_path": str(manifest_path),
                "recording_manifest_sha256": manifest_sha256,
                "external_source_frame_evidence_sha256": (
                    external_source_frame_evidence_sha256
                ),
                "recording_dir": str(resolved_recording_dir),
            },
            artifacts={
                "run_path": run_path,
                "source_crop_xywh": "source_crop_xywh",
                "source_crop_meta_row_indices": "source_crop_meta_row_indices",
                "artifact_payload_inventory_seal_sha256": run.attrs[
                    "artifact_payload_inventory_seal_sha256"
                ],
                "artifact_row_count": artifact_payload_seal["row_count"],
            },
        )
        write_stage_provenance(run, provenance)
        _require_file_sha256(
            resolved_crop_meta,
            expected=crop_meta_sha256,
            label="Crop metadata",
        )
        _require_file_sha256(
            manifest_path,
            expected=manifest_sha256,
            label="Recording manifest",
        )
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
        output_parent=DETECTION_ARTIFACT_RUN_FAMILY,
        coordinate_contract=UNBOUND_DETECTION_ARTIFACT_COORDINATE_CONTRACT,
        stage_selector_eligible=False,
        applied=True,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Analysis zarr archive.")
    parser.add_argument("--recording-dir", type=Path, help="Recording root used to resolve recording_manifest.json.")
    parser.add_argument("--crop-meta", type=Path, help="Explicit external crop-recorder *_crop_meta.csv path.")
    parser.add_argument("--run-name", help="Detect run name (default: timestamped acquisition import run).")
    parser.add_argument(
        "--source-width",
        type=int,
        help="Optional assertion that must match the manifest full-stream width.",
    )
    parser.add_argument(
        "--source-height",
        type=int,
        help="Optional assertion that must match the manifest full-stream height.",
    )
    parser.add_argument("--class-id", type=int, default=0, help="Class id assigned to imported acquisition boxes.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rejected by the immutable fail-closed publisher; choose a new run name.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Write the planned output; requires --artifact-only. Otherwise only "
            "print the plan."
        ),
    )
    parser.add_argument(
        "--artifact-only",
        action="store_true",
        help=(
            "Explicitly permit an unbound nonselector detection_artifact_runs output."
        ),
    )
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
        artifact_only=bool(args.artifact_only),
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
            print(
                "dry_run: pass --apply --artifact-only to write "
                "detection_artifact_runs/<run>"
            )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
