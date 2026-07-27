#!/usr/bin/env python3
"""Compare two exact arena-geometry candidates against one raw detect run.

This is a diagnostic-only surface.  It never selects a candidate, mutates the
analysis Zarr, filters detections, or advances a registry status.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import socket
import subprocess
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from fisheye.analysis_workflows.materializers.arena_geometry_candidates import (
    ACQUISITION_CANDIDATE_KIND,
    PALETTE_CANDIDATE_KIND,
    validate_arena_geometry_candidate_record,
)
from fisheye.shared.json_safety import strict_json_dumps
from fisheye.shared.zarr_io import open_zarr_root


SCHEMA_ID = "palette.diagnostics.arena_geometry_detection_gate_audit"
SCHEMA_VERSION = 1
AUDIT_METHOD = "exact_native_centroid_dual_circle_inclusive_v1"
REVIEW_SELECTION_METHOD = "temporal_quantiles_per_exclusive_category_v1"
BOUNDARY_SENTINEL_SELECTION_METHOD = "temporal_bins_boundary_nearest_sentinels_v1"
PALETTE_COLOR_BGR = (255, 255, 0)
ACQUISITION_COLOR_BGR = (0, 165, 255)
DETECTION_COLOR_BGR = (255, 0, 255)


@dataclass(frozen=True)
class Circle:
    center_x_px: float
    center_y_px: float
    radius_px: float


@dataclass(frozen=True)
class CandidateBinding:
    run_name: str
    candidate_id: str
    candidate_kind: str
    candidate_record_sha256: str
    circle: Circle
    pixel_frame_record_ref: str
    pixel_frame_record_sha256: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload_sha256(value: Any) -> str:
    return hashlib.sha256(strict_json_dumps(value).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def signed_circle_distance(centers_xy: np.ndarray, circle: Circle) -> np.ndarray:
    """Return positive-inside signed distance from points to a circle gate."""

    centers = np.asarray(centers_xy, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("centers_xy must have shape [row, 2]")
    if not np.all(np.isfinite(centers)):
        raise ValueError("centers_xy contains non-finite coordinates")
    values = (circle.center_x_px, circle.center_y_px, circle.radius_px)
    if not all(math.isfinite(value) for value in values) or circle.radius_px <= 0:
        raise ValueError("circle must be finite with a positive radius")
    return circle.radius_px - np.hypot(
        centers[:, 0] - circle.center_x_px,
        centers[:, 1] - circle.center_y_px,
    )


def classify_gate_results(
    palette_signed_distance_px: np.ndarray,
    acquisition_signed_distance_px: np.ndarray,
) -> np.ndarray:
    """Classify inclusive gate results without rounding either distance."""

    palette = np.asarray(palette_signed_distance_px, dtype=np.float64)
    acquisition = np.asarray(acquisition_signed_distance_px, dtype=np.float64)
    if palette.shape != acquisition.shape or palette.ndim != 1:
        raise ValueError("signed-distance arrays must be equal one-dimensional shapes")
    if not np.all(np.isfinite(palette)) or not np.all(np.isfinite(acquisition)):
        raise ValueError("signed-distance arrays must be finite")
    palette_inside = palette >= 0.0
    acquisition_inside = acquisition >= 0.0
    result = np.full(palette.shape, "both_outside", dtype="<U16")
    result[palette_inside & acquisition_inside] = "both_inside"
    result[palette_inside & ~acquisition_inside] = "palette_only"
    result[~palette_inside & acquisition_inside] = "acquisition_only"
    return result


def select_review_rows(
    *,
    categories: np.ndarray,
    frame_indices: np.ndarray,
    max_per_category: int,
) -> np.ndarray:
    """Select deterministic temporal quantiles from both disagreement classes."""

    labels = np.asarray(categories)
    frames = np.asarray(frame_indices, dtype=np.int64)
    if labels.ndim != 1 or frames.shape != labels.shape:
        raise ValueError("categories and frame_indices must be equal vectors")
    if max_per_category <= 0:
        raise ValueError("max_per_category must be positive")
    selected: list[int] = []
    for label in ("palette_only", "acquisition_only"):
        rows = np.flatnonzero(labels == label)
        if len(rows) == 0:
            continue
        rows = rows[np.lexsort((rows, frames[rows]))]
        if len(rows) <= max_per_category:
            chosen = rows
        else:
            positions = np.rint(np.linspace(0, len(rows) - 1, max_per_category)).astype(
                np.int64
            )
            chosen = rows[positions]
        selected.extend(int(row) for row in chosen)
    return np.asarray(
        sorted(set(selected), key=lambda row: (int(frames[row]), row)),
        dtype=np.int64,
    )


def select_boundary_sentinel_rows(
    *,
    frame_indices: np.ndarray,
    palette_signed_distance_px: np.ndarray,
    acquisition_signed_distance_px: np.ndarray,
    max_rows: int,
) -> np.ndarray:
    """Select the boundary-nearest row within each temporal partition."""

    frames = np.asarray(frame_indices, dtype=np.int64)
    palette = np.asarray(palette_signed_distance_px, dtype=np.float64)
    acquisition = np.asarray(acquisition_signed_distance_px, dtype=np.float64)
    if (
        frames.ndim != 1
        or palette.shape != frames.shape
        or acquisition.shape != frames.shape
    ):
        raise ValueError("frames and signed-distance inputs must be equal vectors")
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    if not np.all(np.isfinite(palette)) or not np.all(np.isfinite(acquisition)):
        raise ValueError("signed-distance inputs must be finite")
    if len(frames) == 0:
        return np.empty(0, dtype=np.int64)
    ordered = np.lexsort((np.arange(len(frames)), frames))
    selected: list[int] = []
    for partition in np.array_split(ordered, min(max_rows, len(ordered))):
        proximity = np.minimum(
            np.abs(palette[partition]), np.abs(acquisition[partition])
        )
        selected.append(int(partition[int(np.argmin(proximity))]))
    return np.asarray(
        sorted(selected, key=lambda row: (int(frames[row]), row)), dtype=np.int64
    )


def _candidate_binding(
    root: Any,
    *,
    run_name: str,
    expected_kind: str,
) -> CandidateBinding:
    path = f"analysis/arena_geometry_runs/{run_name}"
    try:
        group = root[path]
    except KeyError as exc:
        raise RuntimeError(f"arena-geometry candidate is missing: {path}") from exc
    attrs = dict(group.attrs)
    if attrs.get("palette_run_completion_status") != "complete":
        raise RuntimeError(f"candidate {run_name!r} is not complete")
    if attrs.get("stage_selector_eligible") is not True:
        raise RuntimeError(f"candidate {run_name!r} is not selector-eligible")
    if (
        attrs.get("operational_selection_status") != "not_selected"
        or attrs.get("detection_gate_applied") is not False
    ):
        raise RuntimeError(
            f"candidate {run_name!r} is not an unselected audit-only candidate"
        )
    record = attrs.get("candidate_record")
    if not isinstance(record, Mapping):
        raise RuntimeError(f"candidate {run_name!r} lacks candidate_record")
    validate_arena_geometry_candidate_record(record)
    if record.get("candidate_kind") != expected_kind:
        raise RuntimeError(
            f"candidate {run_name!r} has kind {record.get('candidate_kind')!r}, "
            f"expected {expected_kind!r}"
        )
    record_sha256 = _payload_sha256(record)
    if attrs.get("candidate_record_sha256") != record_sha256:
        raise RuntimeError(f"candidate {run_name!r} record digest is invalid")
    gate = record["valid_detection_region"]
    geometry = gate["geometry"]
    center = geometry["center_px"]
    coordinate = record["coordinate_binding"]
    return CandidateBinding(
        run_name=run_name,
        candidate_id=str(attrs["candidate_id"]),
        candidate_kind=expected_kind,
        candidate_record_sha256=record_sha256,
        circle=Circle(
            center_x_px=float(center["x"]),
            center_y_px=float(center["y"]),
            radius_px=float(geometry["radius_px"]),
        ),
        pixel_frame_record_ref=str(coordinate["pixel_frame_record_ref"]),
        pixel_frame_record_sha256=str(coordinate["pixel_frame_record_sha256"]),
    )


def _require_coordinate_descriptor(
    array: Any,
    *,
    geometry_type: str,
    expected_width: int,
    expected_height: int,
    pixel_frame_record_ref: str,
    pixel_frame_record_sha256: str,
) -> None:
    descriptor = array.attrs.get("coordinate_descriptor")
    if not isinstance(descriptor, Mapping):
        raise RuntimeError(f"{array.path} lacks a coordinate descriptor")
    extent = descriptor.get("reference_extent")
    frame = descriptor.get("frame_record")
    if not isinstance(extent, Mapping) or not isinstance(frame, Mapping):
        raise RuntimeError(f"{array.path} has an incomplete coordinate descriptor")
    expected = {
        "schema_id": "palette.coordinate_descriptor",
        "schema_version": 2,
        "space_id": "source_camera_image_px",
        "geometry_type": geometry_type,
        "profile_id": "source_camera_image_px.top_left_y_down.v1",
        "origin": "top_left",
    }
    for name, value in expected.items():
        if descriptor.get(name) != value:
            raise RuntimeError(f"{array.path} coordinate descriptor {name} mismatch")
    if (
        int(extent.get("width", 0)) != expected_width
        or int(extent.get("height", 0)) != expected_height
        or frame.get("record_ref") != pixel_frame_record_ref
        or frame.get("record_sha256") != pixel_frame_record_sha256
    ):
        raise RuntimeError(
            f"{array.path} is not bound to the candidates' source-camera frame"
        )


def _read_detection_inputs(
    root: Any,
    *,
    detect_run_name: str,
    candidate: CandidateBinding,
    expected_width: int,
    expected_height: int,
    expected_frame_count: int,
) -> dict[str, Any]:
    path = f"detect_runs/{detect_run_name}"
    try:
        group = root[path]
    except KeyError as exc:
        raise RuntimeError(f"raw detect run is missing: {path}") from exc
    attrs = dict(group.attrs)
    if attrs.get("palette_run_completion_status") != "complete":
        raise RuntimeError(f"detect run {detect_run_name!r} is not complete")
    required = (
        "centers_img_xy",
        "bbox_img_xyxy",
        "frame_indices",
        "instance_key",
        "scores",
    )
    missing = [name for name in required if name not in group]
    if missing:
        raise RuntimeError(f"detect run lacks required arrays: {missing}")
    centers_node = group["centers_img_xy"]
    bbox_node = group["bbox_img_xyxy"]
    _require_coordinate_descriptor(
        centers_node,
        geometry_type="point_xy",
        expected_width=expected_width,
        expected_height=expected_height,
        pixel_frame_record_ref=candidate.pixel_frame_record_ref,
        pixel_frame_record_sha256=candidate.pixel_frame_record_sha256,
    )
    # Boxes use the pixel-edge authority while centers use the continuous
    # authority.  Their native extent is still required to match.
    bbox_descriptor = bbox_node.attrs.get("coordinate_descriptor")
    if not isinstance(bbox_descriptor, Mapping):
        raise RuntimeError("bbox_img_xyxy lacks a coordinate descriptor")
    bbox_extent = bbox_descriptor.get("reference_extent")
    if (
        bbox_descriptor.get("space_id") != "source_camera_image_px"
        or bbox_descriptor.get("geometry_type") != "bbox_xyxy"
        or not isinstance(bbox_extent, Mapping)
        or int(bbox_extent.get("width", 0)) != expected_width
        or int(bbox_extent.get("height", 0)) != expected_height
    ):
        raise RuntimeError("bbox_img_xyxy is not native source-camera geometry")

    arrays = {name: np.asarray(group[name][:]) for name in required}
    row_count = int(arrays["centers_img_xy"].shape[0])
    expected_shapes = {
        "centers_img_xy": (row_count, 2),
        "bbox_img_xyxy": (row_count, 4),
        "frame_indices": (row_count,),
        "instance_key": (row_count,),
        "scores": (row_count,),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise RuntimeError(f"{path}/{name} shape mismatch: {arrays[name].shape}")
    centers = arrays["centers_img_xy"].astype(np.float64, copy=False)
    bbox = arrays["bbox_img_xyxy"].astype(np.float64, copy=False)
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(bbox)):
        raise RuntimeError("detect geometry contains non-finite values")
    derived_centers = np.column_stack(
        ((bbox[:, 0] + bbox[:, 2]) / 2.0, (bbox[:, 1] + bbox[:, 3]) / 2.0)
    )
    if not np.allclose(centers, derived_centers, rtol=0.0, atol=1e-9):
        raise RuntimeError("centers_img_xy disagrees with bbox_img_xyxy")
    frame_indices = arrays["frame_indices"].astype(np.int64, copy=False)
    if row_count and (
        int(frame_indices.min()) < 0 or int(frame_indices.max()) >= expected_frame_count
    ):
        raise RuntimeError("detection frame indices escape the source frame domain")
    instance_keys = arrays["instance_key"].astype(np.uint64, copy=False)
    if len(np.unique(instance_keys)) != row_count:
        raise RuntimeError("detect run instance_key values are not unique")
    if not np.all(np.isfinite(arrays["scores"])):
        raise RuntimeError("detection scores contain non-finite values")
    return {
        "group_attrs": attrs,
        "centers": centers,
        "bbox": bbox,
        "frame_indices": frame_indices,
        "instance_key": instance_keys,
        "scores": arrays["scores"].astype(np.float64, copy=False),
        "row_count": row_count,
    }


def _row_payload(
    row: int,
    *,
    detections: Mapping[str, Any],
    categories: np.ndarray,
    palette_distance: np.ndarray,
    acquisition_distance: np.ndarray,
) -> dict[str, Any]:
    return {
        "row_index": int(row),
        "instance_key": str(int(detections["instance_key"][row])),
        "frame_index": int(detections["frame_indices"][row]),
        "score": float(detections["scores"][row]),
        "centroid_native_px": [float(value) for value in detections["centers"][row]],
        "bbox_native_xyxy": [float(value) for value in detections["bbox"][row]],
        "palette_signed_distance_px": float(palette_distance[row]),
        "acquisition_signed_distance_px": float(acquisition_distance[row]),
        "category": str(categories[row]),
    }


def _write_disagreements_csv(
    path: Path,
    *,
    rows: np.ndarray,
    detections: Mapping[str, Any],
    categories: np.ndarray,
    palette_distance: np.ndarray,
    acquisition_distance: np.ndarray,
) -> None:
    fields = (
        "row_index",
        "instance_key",
        "frame_index",
        "score",
        "centroid_x_native_px",
        "centroid_y_native_px",
        "bbox_x_min_native_px",
        "bbox_y_min_native_px",
        "bbox_x_max_native_px",
        "bbox_y_max_native_px",
        "palette_signed_distance_px",
        "acquisition_signed_distance_px",
        "category",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            index = int(row)
            center = detections["centers"][index]
            bbox = detections["bbox"][index]
            writer.writerow(
                {
                    "row_index": index,
                    "instance_key": str(int(detections["instance_key"][index])),
                    "frame_index": int(detections["frame_indices"][index]),
                    "score": repr(float(detections["scores"][index])),
                    "centroid_x_native_px": repr(float(center[0])),
                    "centroid_y_native_px": repr(float(center[1])),
                    "bbox_x_min_native_px": repr(float(bbox[0])),
                    "bbox_y_min_native_px": repr(float(bbox[1])),
                    "bbox_x_max_native_px": repr(float(bbox[2])),
                    "bbox_y_max_native_px": repr(float(bbox[3])),
                    "palette_signed_distance_px": repr(float(palette_distance[index])),
                    "acquisition_signed_distance_px": repr(
                        float(acquisition_distance[index])
                    ),
                    "category": str(categories[index]),
                }
            )


def _decode_one_frame_from_preceding_keyframe(
    *,
    demuxer: Any,
    decoder: Any,
    target_frame_index: int,
    materialize_frame: Any,
    max_packets_per_seek: int = 256,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Seek to a GOP keyframe and prove the exact requested display frame."""

    seek_timestamp = int(demuxer.TimestampFromFrame(int(target_frame_index)))
    demuxer.Seek(seek_timestamp)
    keyframe_packet_pts: int | None = None
    previous_packet_pts: int | None = None
    target_packet_number: int | None = None
    pending: deque[tuple[int, int]] = deque()
    for packet_count in range(1, max_packets_per_seek + 1):
        packet = demuxer.Demux()
        if int(packet.bsl) <= 0:
            raise RuntimeError("PyNvVideoCodec seek reached an empty packet")
        packet_pts = int(packet.pts)
        if previous_packet_pts is not None and packet_pts <= previous_packet_pts:
            raise RuntimeError(
                "Exact-frame seek does not support reordered/nonmonotonic packet PTS"
            )
        previous_packet_pts = packet_pts
        if packet_count == 1:
            if int(packet.key) != 1:
                raise RuntimeError("PyNvVideoCodec seek did not land on a keyframe")
            keyframe_packet_pts = packet_pts
        relation = int(demuxer.isSeekDone(packet_pts, target_frame_index))
        if relation not in {-1, 0, 1}:
            raise RuntimeError(
                f"PyNvVideoCodec returned invalid seek relation {relation}"
            )
        if relation == 0:
            if target_packet_number is not None:
                raise RuntimeError("PyNvVideoCodec reported the target packet twice")
            target_packet_number = packet_count
        elif relation > 0 and target_packet_number is None:
            raise RuntimeError(
                f"PyNvVideoCodec seek passed target frame {target_frame_index}"
            )
        pending.append((relation, packet_pts))
        decoded = list(decoder.Decode(packet))
        if len(decoded) > len(pending):
            raise RuntimeError(
                "PyNvVideoCodec produced more display frames than submitted packets"
            )
        for frame in decoded:
            output_relation, output_pts = pending.popleft()
            if output_relation == 0:
                if target_packet_number is None:
                    raise RuntimeError("target display frame preceded its packet")
                return materialize_frame(frame), {
                    "target_frame_index": int(target_frame_index),
                    "seek_timestamp": seek_timestamp,
                    "keyframe_packet_pts": keyframe_packet_pts,
                    "target_packet_pts": output_pts,
                    "target_packet_number": target_packet_number,
                    "packets_submitted_through_target_output": packet_count,
                    "packets_after_target_for_decoder_latency": (
                        packet_count - target_packet_number
                    ),
                    "exact_frame_proof": (
                        "demuxer_isSeekDone_exact_monotonic_pts_ordered_display_queue"
                    ),
                }
    raise RuntimeError(
        f"PyNvVideoCodec seek exceeded {max_packets_per_seek} packets for "
        f"frame {target_frame_index}"
    )


def decode_selected_luma_pynvvc(
    video_path: Path,
    *,
    frame_indices: Sequence[int],
    expected_shape_hw: tuple[int, int],
    gpu_id: int,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Decode exact frames from their preceding GOP keyframes."""

    requested = sorted(set(int(value) for value in frame_indices))
    if not requested:
        return {}, {"backend": "not_run_no_disagreements", "requested_frame_count": 0}
    if requested[0] < 0:
        raise ValueError("frame indices must be nonnegative")
    try:
        import PyNvVideoCodec as nvc  # type: ignore
        import torch
    except Exception as exc:  # pragma: no cover - cluster environment dependent
        raise RuntimeError(
            f"PyNvVideoCodec CUDA decode dependencies are unavailable: {exc}"
        ) from exc
    demuxer = nvc.CreateDemuxer(filename=str(video_path))
    source_height = int(demuxer.Height())
    source_width = int(demuxer.Width())
    if (source_height, source_width) != expected_shape_hw:
        raise RuntimeError("video dimensions disagree with the candidate authority")
    frames: dict[int, np.ndarray] = {}
    seeks: list[dict[str, Any]] = []
    started = time.perf_counter()

    def materialize(frame: Any) -> np.ndarray:
        tensor = torch.from_dlpack(frame)
        result = (
            tensor[:source_height, :]
            .contiguous()
            .cpu()
            .numpy()
            .astype(np.uint8, copy=True)
        )
        del tensor
        return result

    for target in requested:
        decoder = nvc.CreateDecoder(
            gpuid=int(gpu_id), codec=demuxer.GetNvCodecId(), usedevicememory=True
        )
        frame, seek = _decode_one_frame_from_preceding_keyframe(
            demuxer=demuxer,
            decoder=decoder,
            target_frame_index=target,
            materialize_frame=materialize,
        )
        if frame.shape != (source_height, source_width):
            raise RuntimeError(
                f"decoded frame {target} has unexpected shape {frame.shape}"
            )
        frames[target] = frame
        seeks.append(seek)
        del decoder
    packet_counts = [
        int(item["packets_submitted_through_target_output"]) for item in seeks
    ]
    return frames, {
        "backend": "pynvvc_luma_gop_keyframe_seek",
        "gpu_id": int(gpu_id),
        "requested_frame_count": len(requested),
        "decoded_packet_count_total": sum(packet_counts),
        "decoded_packet_count_max_per_seek": max(packet_counts),
        "seek_count": len(seeks),
        "seeks": seeks,
        "elapsed_seconds": time.perf_counter() - started,
        "demuxer_frame_rate": float(demuxer.FrameRate()),
        "codec": str(demuxer.GetNvCodecId()),
    }


def _draw_circle(
    image: np.ndarray,
    circle: Circle,
    *,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    cv2.circle(
        image,
        (round(circle.center_x_px), round(circle.center_y_px)),
        round(circle.radius_px),
        color,
        thickness,
        cv2.LINE_AA,
    )


def _render_review_panel(
    luma: np.ndarray,
    *,
    row: Mapping[str, Any],
    palette: Circle,
    acquisition: Circle,
) -> np.ndarray:
    full = cv2.cvtColor(np.asarray(luma), cv2.COLOR_GRAY2BGR)
    thickness = max(4, round(max(full.shape[:2]) / 900))
    _draw_circle(full, palette, color=PALETTE_COLOR_BGR, thickness=thickness)
    _draw_circle(full, acquisition, color=ACQUISITION_COLOR_BGR, thickness=thickness)
    bbox = [round(float(value)) for value in row["bbox_native_xyxy"]]
    center = [round(float(value)) for value in row["centroid_native_px"]]
    cv2.rectangle(
        full, (bbox[0], bbox[1]), (bbox[2], bbox[3]), DETECTION_COLOR_BGR, thickness
    )
    cv2.drawMarker(
        full,
        (center[0], center[1]),
        DETECTION_COLOR_BGR,
        cv2.MARKER_CROSS,
        40,
        thickness,
    )

    panel_width = 560
    full_view = cv2.resize(
        full, (panel_width, panel_width), interpolation=cv2.INTER_AREA
    )
    half_crop = 420
    x0 = max(0, center[0] - half_crop)
    y0 = max(0, center[1] - half_crop)
    x1 = min(full.shape[1], center[0] + half_crop)
    y1 = min(full.shape[0], center[1] + half_crop)
    crop = full[y0:y1, x0:x1]
    crop_view = cv2.resize(
        crop, (panel_width, panel_width), interpolation=cv2.INTER_AREA
    )
    header_height = 105
    panel = np.full(
        (panel_width + header_height, panel_width * 2, 3), 20, dtype=np.uint8
    )
    panel[header_height:, :panel_width] = full_view
    panel[header_height:, panel_width:] = crop_view
    lines = (
        f"{row['category']}  frame={row['frame_index']}  row={row['row_index']}  score={row['score']:.4f}",
        f"Palette signed={row['palette_signed_distance_px']:+.2f}px  acquisition={row['acquisition_signed_distance_px']:+.2f}px",
        "cyan=Palette visible top rim  orange=acquisition valid gate  magenta=detection",
    )
    for index, line in enumerate(lines):
        cv2.putText(
            panel,
            line,
            (16, 28 + 31 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
    return panel


def _render_review_artifacts(
    output_dir: Path,
    *,
    selected_rows: Sequence[Mapping[str, Any]],
    frames: Mapping[int, np.ndarray],
    palette: Circle,
    acquisition: Circle,
) -> dict[str, Any]:
    panel_dir = output_dir / "review_panels"
    panel_dir.mkdir()
    panels: list[np.ndarray] = []
    files: list[dict[str, Any]] = []
    for index, row in enumerate(selected_rows):
        frame_index = int(row["frame_index"])
        panel = _render_review_panel(
            frames[frame_index],
            row=row,
            palette=palette,
            acquisition=acquisition,
        )
        path = panel_dir / f"{index:02d}_{row['category']}_frame_{frame_index:09d}.png"
        if not cv2.imwrite(str(path), panel):
            raise RuntimeError(f"failed to write review panel: {path}")
        panels.append(panel)
        files.append(
            {
                "path": str(path.relative_to(output_dir)),
                "sha256": _sha256_file(path),
                "row_index": int(row["row_index"]),
                "frame_index": frame_index,
                "category": str(row["category"]),
            }
        )
    if panels:
        columns = 2
        rows = math.ceil(len(panels) / columns)
        blank = np.full_like(panels[0], 20)
        montage_rows = []
        for row_index in range(rows):
            cells = panels[row_index * columns : (row_index + 1) * columns]
            cells += [blank] * (columns - len(cells))
            montage_rows.append(np.hstack(cells))
        montage = np.vstack(montage_rows)
        montage_path = output_dir / "detection_gate_disagreement_montage.png"
        if not cv2.imwrite(str(montage_path), montage):
            raise RuntimeError(f"failed to write review montage: {montage_path}")
        montage_record: Mapping[str, Any] | None = {
            "path": montage_path.name,
            "sha256": _sha256_file(montage_path),
            "shape": [int(value) for value in montage.shape],
        }
    else:
        montage_record = None
    return {"panels": files, "montage": montage_record}


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_audit(args: argparse.Namespace) -> Path:
    source_zarr = Path(args.zarr).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing existing output directory: {output_dir}")
    if not source_zarr.is_dir():
        raise FileNotFoundError(source_zarr)
    root = open_zarr_root(source_zarr, mode="r")
    root_attrs = dict(root.attrs)
    width = int(root_attrs.get("source_video_width", 0))
    height = int(root_attrs.get("source_video_height", 0))
    frame_count = int(root_attrs.get("total_frames", 0))
    if width <= 0 or height <= 0 or frame_count <= 0:
        raise RuntimeError("analysis Zarr lacks valid native video geometry")

    palette = _candidate_binding(
        root,
        run_name=args.palette_candidate_run,
        expected_kind=PALETTE_CANDIDATE_KIND,
    )
    acquisition = _candidate_binding(
        root,
        run_name=args.acquisition_candidate_run,
        expected_kind=ACQUISITION_CANDIDATE_KIND,
    )
    if (
        palette.pixel_frame_record_ref != acquisition.pixel_frame_record_ref
        or palette.pixel_frame_record_sha256 != acquisition.pixel_frame_record_sha256
    ):
        raise RuntimeError("geometry candidates do not share one pixel-frame authority")
    detections = _read_detection_inputs(
        root,
        detect_run_name=args.detect_run,
        candidate=palette,
        expected_width=width,
        expected_height=height,
        expected_frame_count=frame_count,
    )
    palette_distance = signed_circle_distance(detections["centers"], palette.circle)
    acquisition_distance = signed_circle_distance(
        detections["centers"], acquisition.circle
    )
    categories = classify_gate_results(palette_distance, acquisition_distance)
    exclusive_rows = np.flatnonzero(
        (categories == "palette_only") | (categories == "acquisition_only")
    )
    review_indices = select_review_rows(
        categories=categories,
        frame_indices=detections["frame_indices"],
        max_per_category=args.max_review_samples_per_category,
    )
    if len(review_indices):
        review_selection_method = REVIEW_SELECTION_METHOD
        review_selection_reason = "exclusive_gate_disagreements_present"
    else:
        review_indices = select_boundary_sentinel_rows(
            frame_indices=detections["frame_indices"],
            palette_signed_distance_px=palette_distance,
            acquisition_signed_distance_px=acquisition_distance,
            max_rows=args.max_review_samples_per_category,
        )
        review_selection_method = BOUNDARY_SENTINEL_SELECTION_METHOD
        review_selection_reason = (
            "no_exclusive_disagreements_boundary_nearest_sentinels"
        )
    review_rows = [
        _row_payload(
            int(row),
            detections=detections,
            categories=categories,
            palette_distance=palette_distance,
            acquisition_distance=acquisition_distance,
        )
        for row in review_indices
    ]

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp.", dir=output_dir.parent)
    )
    try:
        _write_disagreements_csv(
            temporary / "detection_gate_disagreements.csv",
            rows=exclusive_rows,
            detections=detections,
            categories=categories,
            palette_distance=palette_distance,
            acquisition_distance=acquisition_distance,
        )
        review_plan = {
            "schema_id": f"{SCHEMA_ID}.review_plan",
            "schema_version": SCHEMA_VERSION,
            "selection_method": review_selection_method,
            "selection_reason": review_selection_reason,
            "max_per_category": int(args.max_review_samples_per_category),
            "rows": review_rows,
        }
        _atomic_json(temporary / "review_plan.json", review_plan)

        decode: Mapping[str, Any] = {
            "backend": "not_requested",
            "requested_frame_count": 0,
        }
        review_artifacts: Mapping[str, Any] = {"panels": [], "montage": None}
        video_record: Mapping[str, Any] | None = None
        if args.video is not None:
            video_path = Path(args.video).expanduser().resolve()
            if not video_path.is_file():
                raise FileNotFoundError(video_path)
            source_metadata = root_attrs.get("source_video_metadata")
            if not isinstance(source_metadata, Mapping):
                raise RuntimeError("analysis Zarr lacks source_video_metadata")
            fingerprint = source_metadata.get("file_fingerprint")
            if not isinstance(fingerprint, Mapping):
                raise RuntimeError("analysis Zarr lacks source video fingerprint")
            if video_path.name != source_metadata.get(
                "source_video"
            ) or video_path.stat().st_size != int(fingerprint.get("size_bytes", -1)):
                raise RuntimeError(
                    "video does not match the analysis Zarr source identity"
                )
            frames, decode = decode_selected_luma_pynvvc(
                video_path,
                frame_indices=[row["frame_index"] for row in review_rows],
                expected_shape_hw=(height, width),
                gpu_id=args.gpu_id,
            )
            review_artifacts = _render_review_artifacts(
                temporary,
                selected_rows=review_rows,
                frames=frames,
                palette=palette.circle,
                acquisition=acquisition.circle,
            )
            video_record = {
                "path": str(video_path),
                "filename": video_path.name,
                "size_bytes": video_path.stat().st_size,
            }

        counts = {
            label: int(np.count_nonzero(categories == label))
            for label in (
                "both_inside",
                "both_outside",
                "palette_only",
                "acquisition_only",
            )
        }
        report = {
            "schema_id": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "status": "complete_review_required",
            "created_at_utc": _utc_now(),
            "audit_method": AUDIT_METHOD,
            "source_zarr": str(source_zarr),
            "detect_run": {
                "run_name": args.detect_run,
                "row_count": int(detections["row_count"]),
                "row_identity_contract_sha256": detections["group_attrs"].get(
                    "row_identity_contract_sha256"
                ),
                "detection_gate_applied": False,
            },
            "palette_candidate": {
                **palette.__dict__,
                "circle": palette.circle.__dict__,
            },
            "acquisition_candidate": {
                **acquisition.__dict__,
                "circle": acquisition.circle.__dict__,
            },
            "coordinate_contract": {
                "space_id": "source_camera_image_px",
                "profile_id": "source_camera_image_px.top_left_y_down.v1",
                "pixel_convention": "continuous",
                "width_px": width,
                "height_px": height,
                "pixel_frame_record_ref": palette.pixel_frame_record_ref,
                "pixel_frame_record_sha256": palette.pixel_frame_record_sha256,
                "centroid_gate_boundary": "inclusive",
                "signed_distance_definition": "radius_px-hypot(x-cx,y-cy)",
            },
            "counts": {
                **counts,
                "exclusive_disagreements": int(len(exclusive_rows)),
                "total": int(detections["row_count"]),
            },
            "artifacts": {
                "disagreements_csv": {
                    "path": "detection_gate_disagreements.csv",
                    "sha256": _sha256_file(
                        temporary / "detection_gate_disagreements.csv"
                    ),
                    "row_count": int(len(exclusive_rows)),
                },
                "review_plan": {
                    "path": "review_plan.json",
                    "sha256": _sha256_file(temporary / "review_plan.json"),
                    "row_count": len(review_rows),
                },
                "review_images": review_artifacts,
            },
            "video": video_record,
            "decode": decode,
            "environment": {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "lsf_job_id": os.environ.get("LSB_JOBID"),
                "palette_git_commit": _git_commit(),
            },
            "decision": {
                "operational_candidate_selected": False,
                "detections_filtered_or_mutated": False,
                "registry_updated": False,
                "next_gate": (
                    "human_review_of_detection_disagreement_montage"
                    if len(exclusive_rows)
                    else "human_review_of_boundary_sentinel_montage"
                ),
            },
        }
        _atomic_json(temporary / "audit_report.json", report)
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_dir / "audit_report.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr", type=Path, required=True)
    parser.add_argument("--palette-candidate-run", required=True)
    parser.add_argument("--acquisition-candidate-run", required=True)
    parser.add_argument("--detect-run", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--video",
        type=Path,
        help="Optional exact source video; enables PyNvVC review rendering.",
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--max-review-samples-per-category", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_audit(args)
    print(
        json.dumps({"status": "complete", "audit_report": str(report)}, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
