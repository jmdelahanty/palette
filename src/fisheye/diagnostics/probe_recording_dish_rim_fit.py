#!/usr/bin/env python3
"""Create blind temporal-composite diagnostics for a recording's dish rim.

This command is deliberately not a publication surface.  It does not open an
analysis Zarr, update the registry, select a mask, or gate detections.  The
Palette fit is frozen before an optional acquisition observation is opened for
reveal-only comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import socket
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from fisheye.shared.pynvvc_exact_seek import (
    decode_one_frame_from_preceding_keyframe,
)

SCHEMA_ID = "palette.diagnostics.recording_dish_rim_probe"
SCHEMA_VERSION = 1
FIT_METHOD = "temporal_median_keyframe_only_multicandidate_radial_edge_circle_v2"
TARGET_FEATURE = "dish_inner_rim_water_side_edge"
TARGET_PLANE = "dish_top_rim"
WINDOW_NAMES = ("early", "middle", "late")
WINDOW_FRACTIONS = (0.10, 0.50, 0.90)
CLIPPED_INDEX_SCHEMA_ID = "palette.orange_external_ipc_recording_clip_index.v1"


@dataclass(frozen=True)
class CircleCandidate:
    candidate_id: str
    center_x_px: float
    center_y_px: float
    radius_px: float
    angular_support_fraction: float
    radial_residual_px: float
    median_radial_gradient: float
    evidence_score: float

    def to_json(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "geometry": {
                "type": "circle",
                "center_px": {"x": self.center_x_px, "y": self.center_y_px},
                "radius_px": self.radius_px,
            },
            "coordinate_space": "camera_native_pixels",
            "observed_feature_classification": "unclassified_concentric_rim_edge",
            "angular_support_fraction": self.angular_support_fraction,
            "radial_residual_px": self.radial_residual_px,
            "median_radial_gradient": self.median_radial_gradient,
            "evidence_score": self.evidence_score,
        }


@dataclass(frozen=True)
class CircleFit:
    center_x_px: float
    center_y_px: float
    radius_px: float
    angular_support_fraction: float
    median_radial_gradient: float
    candidate_count: int
    radial_residual_px: float = 0.0
    selected_candidate_id: str | None = None
    selection_reason: str = "median_consensus_v1"
    frozen_candidates: tuple[CircleCandidate, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "geometry": {
                "type": "circle",
                "center_px": {"x": self.center_x_px, "y": self.center_y_px},
                "radius_px": self.radius_px,
            },
            "coordinate_space": "camera_native_pixels",
            "target_feature": TARGET_FEATURE,
            "intended_target_feature": TARGET_FEATURE,
            "observed_feature_classification": "unclassified_concentric_rim_edge",
            "target_plane": TARGET_PLANE,
            "angular_support_fraction": self.angular_support_fraction,
            "radial_residual_px": self.radial_residual_px,
            "median_radial_gradient": self.median_radial_gradient,
            "candidate_count": self.candidate_count,
            "selected_candidate_id": self.selected_candidate_id,
            "selection_reason": self.selection_reason,
            "frozen_candidates": [
                candidate.to_json() for candidate in self.frozen_candidates
            ],
        }


@dataclass(frozen=True)
class WindowSpec:
    name: str
    fraction: float
    center_frame: int
    frame_indices: tuple[int, ...]


@dataclass(frozen=True)
class ClipVideoSource:
    clip_id: str
    clip_index: int
    camera_serial: str
    video_path: Path
    keyframe_path: Path
    first_recording_frame_id: int
    last_recording_frame_id: int
    frame_count: int
    fps: float
    keyframe_frames: tuple[int, ...]


@dataclass(frozen=True)
class ClippedRecordingSource:
    recording_dir: Path
    clip_index_path: Path
    recording_id: str
    session_id: str
    camera_serial: str
    first_recording_frame_id: int
    last_recording_frame_id: int
    frame_count: int
    fps: float
    width: int
    height: int
    clips: tuple[ClipVideoSource, ...]


@dataclass(frozen=True)
class ClippedFrameRef:
    clip_id: str
    clip_index: int
    video_path: Path
    keyframe_path: Path
    clip_local_frame_index: int
    recording_frame_id: int


@dataclass(frozen=True)
class ClippedWindowSpec:
    name: str
    fraction: float
    center_recording_frame_id: int
    frames: tuple[ClippedFrameRef, ...]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(tmp, path)


def load_declared_keyframes(
    path: Path,
    *,
    expected_frame_count: int,
    expected_fps: float,
) -> tuple[int, ...]:
    """Load and validate the encoder-declared keyframe frame numbers."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("keyframe summary must contain a JSON object")
    if int(payload.get("total_frames", 0)) != int(expected_frame_count):
        raise ValueError("keyframe summary frame count disagrees with source summary")
    declared_fps = float(payload.get("fps", 0.0))
    if not math.isclose(declared_fps, float(expected_fps), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("keyframe summary fps disagrees with source summary")
    raw = payload.get("keyframe_frames")
    if not isinstance(raw, list) or not raw:
        raise ValueError("keyframe summary has no keyframe_frames")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in raw):
        raise ValueError("keyframe frame numbers must be exact integers")
    frames = tuple(int(value) for value in raw)
    if tuple(sorted(set(frames))) != frames:
        raise ValueError(
            "keyframe frame numbers must be strictly increasing and unique"
        )
    if frames[0] < 0 or frames[-1] >= int(expected_frame_count):
        raise ValueError("keyframe frame numbers escape the source frame domain")
    return frames


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {label} JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _resolve_recording_file(
    recording_dir: Path,
    value: Any,
    *,
    field: str,
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"clipped index row has no {field}")
    raw = Path(value)
    candidate = raw if raw.is_absolute() else recording_dir / raw
    resolved = candidate.expanduser().resolve()
    if resolved != recording_dir and recording_dir not in resolved.parents:
        raise ValueError(f"clipped index {field} escapes recording directory: {value}")
    if not resolved.is_file():
        raise FileNotFoundError(f"clipped index {field} does not exist: {resolved}")
    return resolved


def _recording_camera_properties(
    recording_dir: Path,
    *,
    recording_id: str,
    session_id: str,
    camera_serial: str,
) -> tuple[int, int, float]:
    snapshot_path = (
        recording_dir / "raw" / "recording_geometry_bundle" / "recording_snapshot.json"
    )
    snapshot = _load_json_object(snapshot_path, label="recording geometry snapshot")
    snapshot_recording_id = str(snapshot.get("recording_id") or "")
    expected_recording_ids = {recording_id, session_id}
    if snapshot_recording_id not in expected_recording_ids:
        raise ValueError(
            "recording geometry snapshot recording_id is inconsistent: "
            f"expected_one_of={sorted(expected_recording_ids)!r}, "
            f"observed={snapshot_recording_id!r}"
        )
    runtime_by_camera = snapshot.get("camera_runtime")
    camera = (
        runtime_by_camera.get(camera_serial)
        if isinstance(runtime_by_camera, Mapping)
        else None
    )
    if not isinstance(camera, Mapping):
        raise ValueError(
            f"camera {camera_serial} is absent from recording geometry snapshot: {snapshot_path}"
        )
    coordinate_frame = camera.get("coordinate_frame")
    image_shape = (
        coordinate_frame.get("image_shape")
        if isinstance(coordinate_frame, Mapping)
        else None
    )
    runtime = camera.get("runtime")
    height = int(
        (image_shape.get("height") if isinstance(image_shape, Mapping) else 0)
        or (runtime.get("height") if isinstance(runtime, Mapping) else 0)
        or camera.get("height")
        or 0
    )
    width = int(
        (image_shape.get("width") if isinstance(image_shape, Mapping) else 0)
        or (runtime.get("width") if isinstance(runtime, Mapping) else 0)
        or camera.get("width")
        or 0
    )
    frame_rate = float(
        (runtime.get("frame_rate") if isinstance(runtime, Mapping) else 0.0) or 0.0
    )
    if height <= 0 or width <= 0:
        raise ValueError(
            f"camera {camera_serial} has no positive image shape in {snapshot_path}"
        )
    if not math.isfinite(frame_rate) or frame_rate <= 0:
        raise ValueError(
            f"camera {camera_serial} has no positive frame rate in {snapshot_path}"
        )
    return height, width, frame_rate


def load_clipped_recording_source(
    recording_dir: str | Path,
) -> ClippedRecordingSource:
    """Load and fail-closed validate one camera's rolling-clip source."""

    root = Path(recording_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"clipped recording directory not found: {root}")
    index_path = root / "recording_clip_index.json"
    index = _load_json_object(index_path, label="recording clip index")
    raw_rows = index.get("rows")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError(f"recording clip index has no rows: {index_path}")
    if any(not isinstance(row, Mapping) for row in raw_rows):
        raise ValueError(
            f"recording clip index contains a non-object row: {index_path}"
        )
    rows = [dict(row) for row in raw_rows]
    if int(index.get("row_count") or -1) != len(rows):
        raise ValueError(
            f"recording clip index row_count is inconsistent: {index_path}"
        )
    if str(index.get("schema_id") or "") != CLIPPED_INDEX_SCHEMA_ID:
        raise ValueError(f"unsupported recording clip index schema: {index_path}")
    if int(index.get("schema_version") or 0) != 1:
        raise ValueError(f"unsupported recording clip index version: {index_path}")
    if str(index.get("source_layout") or "") != "rolling_clips":
        raise ValueError(f"recording clip index is not rolling_clips: {index_path}")

    cameras = sorted({str(row.get("camera_serial") or "") for row in rows})
    if len(cameras) != 1 or not cameras[0].isdigit():
        raise ValueError(
            f"clipped dish-rim fitting requires exactly one camera stream; observed {cameras}"
        )
    camera_serial = cameras[0]
    declared_cameras = index.get("cameras")
    if declared_cameras is not None and [str(value) for value in declared_cameras] != [
        camera_serial
    ]:
        raise ValueError(
            f"recording clip index camera declaration is inconsistent: {index_path}"
        )
    recording_id = str(index.get("recording_id") or root.name)
    if recording_id != root.name:
        raise ValueError(
            f"recording clip index recording_id {recording_id!r} does not match {root.name!r}"
        )
    session_id = str(index.get("session_id") or recording_id)
    height, width, recording_fps = _recording_camera_properties(
        root,
        recording_id=recording_id,
        session_id=session_id,
        camera_serial=camera_serial,
    )

    clips: list[ClipVideoSource] = []
    seen_clip_ids: set[str] = set()
    seen_clip_indices: set[int] = set()
    fps_values: set[float] = set()
    previous_last: int | None = None
    for row in sorted(rows, key=lambda item: int(item.get("clip_index") or 0)):
        clip_id = str(row.get("clip_id") or "")
        clip_index = int(row.get("clip_index") or 0)
        if not re.fullmatch(r"clip_[0-9]{6}", clip_id):
            raise ValueError(f"invalid clip_id in {index_path}: {clip_id!r}")
        if clip_id != f"clip_{clip_index:06d}":
            raise ValueError(
                f"clip_id and clip_index disagree in {index_path}: "
                f"{clip_id!r}, {clip_index}"
            )
        if str(row.get("camera_serial") or "") != camera_serial:
            raise ValueError(f"clip camera disagrees with recording: {clip_id}")
        if str(row.get("recording_id") or "") != recording_id:
            raise ValueError(f"clip recording_id disagrees with recording: {clip_id}")
        if str(row.get("session_id") or "") != session_id:
            raise ValueError(f"clip session_id disagrees with recording: {clip_id}")
        if clip_id in seen_clip_ids or clip_index in seen_clip_indices:
            raise ValueError(f"duplicate clip identity in {index_path}: {clip_id}")
        seen_clip_ids.add(clip_id)
        seen_clip_indices.add(clip_index)
        if row.get("status") not in (None, "completed"):
            raise ValueError(f"clip is not completed: {clip_id}")
        if row.get("recording_frame_id_gaps") not in (None, 0, "0", [], {}):
            raise ValueError(f"clip reports recording-frame gaps: {clip_id}")

        first = int(row.get("first_recording_frame_id") or 0)
        last = int(row.get("last_recording_frame_id") or 0)
        frame_count = int(row.get("frame_count") or 0)
        if first <= 0 or last < first or frame_count != last - first + 1:
            raise ValueError(f"clip has an invalid dense frame range: {clip_id}")
        if previous_last is not None and first != previous_last + 1:
            raise ValueError(
                f"recording frame ranges are not continuous before {clip_id}: "
                f"previous_last={previous_last}, first={first}"
            )
        previous_last = last
        video_path = _resolve_recording_file(
            root,
            row.get("video_path") or row.get("video"),
            field=f"{clip_id}.video_path",
        )
        keyframe_path = _resolve_recording_file(
            root,
            row.get("keyframe_path") or row.get("keyframes"),
            field=f"{clip_id}.keyframe_path",
        )
        keyframe_payload = _load_json_object(
            keyframe_path, label=f"{clip_id} keyframe summary"
        )
        fps = float(keyframe_payload.get("fps") or 0.0)
        keyframes = load_declared_keyframes(
            keyframe_path,
            expected_frame_count=frame_count,
            expected_fps=recording_fps,
        )
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"clip has an invalid frame rate: {clip_id}")
        fps_values.add(fps)
        clips.append(
            ClipVideoSource(
                clip_id=clip_id,
                clip_index=clip_index,
                camera_serial=camera_serial,
                video_path=video_path,
                keyframe_path=keyframe_path,
                first_recording_frame_id=first,
                last_recording_frame_id=last,
                frame_count=frame_count,
                fps=fps,
                keyframe_frames=keyframes,
            )
        )

    declared_clip_count = int(index.get("clip_count") or -1)
    if declared_clip_count != len(clips):
        raise ValueError(
            f"recording clip index clip_count is inconsistent: {index_path}"
        )
    if len(fps_values) != 1:
        raise ValueError(
            f"clipped recording has inconsistent frame rates: {sorted(fps_values)}"
        )
    if not math.isclose(
        next(iter(fps_values)), recording_fps, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError(
            "clipped recording frame rate disagrees with recording geometry"
        )
    first = clips[0].first_recording_frame_id
    last = clips[-1].last_recording_frame_id
    frame_count = sum(clip.frame_count for clip in clips)
    if frame_count != last - first + 1:
        raise ValueError("clipped recording frame ranges are not globally dense")
    return ClippedRecordingSource(
        recording_dir=root,
        clip_index_path=index_path,
        recording_id=recording_id,
        session_id=session_id,
        camera_serial=camera_serial,
        first_recording_frame_id=first,
        last_recording_frame_id=last,
        frame_count=frame_count,
        fps=recording_fps,
        width=width,
        height=height,
        clips=tuple(clips),
    )


def build_keyframe_window_specs(
    *,
    frame_count: int,
    fps: float,
    keyframe_frames: Sequence[int],
    max_keyframes_per_window: int = 21,
    span_seconds: float = 5.0,
    fractions: Sequence[float] = WINDOW_FRACTIONS,
) -> tuple[WindowSpec, ...]:
    """Select only declared keyframes in early, middle, and late windows."""

    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if max_keyframes_per_window < 3:
        raise ValueError("max_keyframes_per_window must be at least three")
    if not math.isfinite(span_seconds) or span_seconds <= 0:
        raise ValueError("span_seconds must be finite and positive")
    if len(fractions) != len(WINDOW_NAMES):
        raise ValueError(f"exactly {len(WINDOW_NAMES)} window fractions are required")

    available = np.asarray(
        tuple(int(value) for value in keyframe_frames), dtype=np.int64
    )
    if available.ndim != 1 or len(available) == 0:
        raise ValueError("keyframe_frames must be a nonempty vector")
    if (
        np.any(available[1:] <= available[:-1])
        or int(available[0]) < 0
        or int(available[-1]) >= frame_count
    ):
        raise ValueError("keyframe_frames must be ordered, unique, and in bounds")

    half_span = 0.5 * span_seconds * fps
    specs: list[WindowSpec] = []
    occupied: set[int] = set()
    for name, fraction in zip(WINDOW_NAMES, fractions, strict=True):
        if not math.isfinite(float(fraction)) or not 0.0 < float(fraction) < 1.0:
            raise ValueError("window fractions must lie strictly between zero and one")
        center = int(round(float(fraction) * (frame_count - 1)))
        within = available[
            (available >= math.ceil(center - half_span))
            & (available <= math.floor(center + half_span))
        ]
        if len(within) < 3:
            raise ValueError(f"{name} window contains fewer than three keyframes")
        count = min(int(max_keyframes_per_window), len(within))
        if count % 2 == 0:
            count -= 1
        positions = np.rint(np.linspace(0, len(within) - 1, count)).astype(np.int64)
        selected = tuple(int(value) for value in within[positions])
        if len(set(selected)) != len(selected):
            raise RuntimeError(f"{name} keyframe sampling produced duplicates")
        overlap = occupied.intersection(selected)
        if overlap:
            raise ValueError(
                f"temporal keyframe windows overlap at frame {min(overlap)}"
            )
        occupied.update(selected)
        specs.append(WindowSpec(name, float(fraction), center, selected))
    return tuple(specs)


def build_clipped_keyframe_window_specs(
    source: ClippedRecordingSource,
    *,
    max_keyframes_per_window: int = 21,
    span_seconds: float = 5.0,
    fractions: Sequence[float] = WINDOW_FRACTIONS,
) -> tuple[ClippedWindowSpec, ...]:
    """Select declared keyframes over one continuous clipped-recording clock."""

    if max_keyframes_per_window < 3:
        raise ValueError("max_keyframes_per_window must be at least three")
    if not math.isfinite(span_seconds) or span_seconds <= 0:
        raise ValueError("span_seconds must be finite and positive")
    if len(fractions) != len(WINDOW_NAMES):
        raise ValueError(f"exactly {len(WINDOW_NAMES)} window fractions are required")

    half_span = 0.5 * span_seconds * source.fps
    occupied: set[tuple[str, int]] = set()
    specs: list[ClippedWindowSpec] = []
    for name, fraction in zip(WINDOW_NAMES, fractions, strict=True):
        if not math.isfinite(float(fraction)) or not 0.0 < float(fraction) < 1.0:
            raise ValueError("window fractions must lie strictly between zero and one")
        center = source.first_recording_frame_id + int(
            round(float(fraction) * (source.frame_count - 1))
        )
        lower = math.ceil(center - half_span)
        upper = math.floor(center + half_span)
        available: list[ClippedFrameRef] = []
        for clip in source.clips:
            if (
                clip.last_recording_frame_id < lower
                or clip.first_recording_frame_id > upper
            ):
                continue
            for local_index in clip.keyframe_frames:
                recording_frame_id = clip.first_recording_frame_id + local_index
                if lower <= recording_frame_id <= upper:
                    available.append(
                        ClippedFrameRef(
                            clip_id=clip.clip_id,
                            clip_index=clip.clip_index,
                            video_path=clip.video_path,
                            keyframe_path=clip.keyframe_path,
                            clip_local_frame_index=local_index,
                            recording_frame_id=recording_frame_id,
                        )
                    )
        available.sort(key=lambda item: item.recording_frame_id)
        if len(available) < 3:
            raise ValueError(f"{name} window contains fewer than three keyframes")
        count = min(int(max_keyframes_per_window), len(available))
        if count % 2 == 0:
            count -= 1
        positions = np.rint(np.linspace(0, len(available) - 1, count)).astype(np.int64)
        selected = tuple(available[int(position)] for position in positions)
        identities = {(item.clip_id, item.clip_local_frame_index) for item in selected}
        if len(identities) != len(selected):
            raise RuntimeError(f"{name} clipped keyframe sampling produced duplicates")
        overlap = occupied.intersection(identities)
        if overlap:
            clip_id, local_index = sorted(overlap)[0]
            raise ValueError(
                f"temporal keyframe windows overlap at {clip_id} frame {local_index}"
            )
        occupied.update(identities)
        specs.append(
            ClippedWindowSpec(
                name=name,
                fraction=float(fraction),
                center_recording_frame_id=center,
                frames=selected,
            )
        )
    return tuple(specs)


def temporal_median(frames: np.ndarray) -> np.ndarray:
    """Reduce a uint8 ``[sample, y, x]`` stack without preserving the stack."""

    stack = np.asarray(frames)
    if stack.ndim != 3 or stack.shape[0] < 3:
        raise ValueError(
            "frames must have shape [sample, y, x] with at least three samples"
        )
    if stack.dtype != np.uint8:
        raise ValueError("frames must be uint8 luma")
    return np.median(stack, axis=0, overwrite_input=True).astype(np.uint8)


def _fit_circle_least_squares(points_xy: np.ndarray) -> tuple[float, float, float]:
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError("at least three xy points are required")
    x = points[:, 0]
    y = points[:, 1]
    matrix = np.column_stack((2.0 * x, 2.0 * y, np.ones_like(x)))
    rhs = x * x + y * y
    cx, cy, constant = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
    radius_sq = constant + cx * cx + cy * cy
    if not np.isfinite(radius_sq) or radius_sq <= 0:
        raise RuntimeError("circle least-squares fit produced an invalid radius")
    return float(cx), float(cy), float(math.sqrt(radius_sq))


def _gradient_magnitude(image: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(np.asarray(image, dtype=np.uint8), (0, 0), 1.5)
    gx = cv2.Scharr(blurred, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(blurred, cv2.CV_32F, 0, 1)
    return cv2.magnitude(gx, gy)


def _radial_evidence(
    gradient: np.ndarray,
    circle: tuple[float, float, float],
    *,
    radial_band_px: float,
    angle_count: int = 1440,
) -> tuple[np.ndarray, np.ndarray]:
    cx, cy, radius = circle
    angles = np.linspace(
        0.0, 2.0 * np.pi, angle_count, endpoint=False, dtype=np.float32
    )
    offsets = np.linspace(-radial_band_px, radial_band_px, 2 * int(radial_band_px) + 1)
    radii = radius + offsets[:, None]
    map_x = (cx + radii * np.cos(angles)[None, :]).astype(np.float32)
    map_y = (cy + radii * np.sin(angles)[None, :]).astype(np.float32)
    sampled = cv2.remap(
        gradient,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    best_rows = np.argmax(sampled, axis=0)
    columns = np.arange(angle_count)
    peaks = sampled[best_rows, columns]
    peak_radii = radius + offsets[best_rows]
    points = np.column_stack(
        (cx + peak_radii * np.cos(angles), cy + peak_radii * np.sin(angles))
    )
    return points.astype(np.float64), peaks.astype(np.float64)


def _refine_and_score_circle(
    gradient: np.ndarray,
    circle: tuple[float, float, float],
    *,
    radial_band_px: float,
) -> tuple[tuple[float, float, float], float, float, float]:
    refined = circle
    peaks = np.empty(0, dtype=np.float64)
    for _ in range(2):
        points, peaks = _radial_evidence(
            gradient, refined, radial_band_px=radial_band_px
        )
        positive = peaks[peaks > 0]
        if len(positive) < 24:
            break
        cutoff = max(float(np.percentile(positive, 35.0)), 1.0)
        keep = peaks >= cutoff
        if int(np.count_nonzero(keep)) < 24:
            break
        refined = _fit_circle_least_squares(points[keep])

    positive = peaks[peaks > 0]
    if len(positive) == 0:
        return refined, 0.0, 0.0, float(radial_band_px)
    support_cutoff = max(float(np.percentile(gradient, 85.0)) * 0.35, 1.0)
    support = float(np.mean(peaks >= support_cutoff))
    median = float(np.median(positive))
    points, peaks = _radial_evidence(
        gradient,
        refined,
        radial_band_px=radial_band_px,
    )
    supported = peaks >= support_cutoff
    if int(np.count_nonzero(supported)) < 3:
        residual = float(radial_band_px)
    else:
        distances = np.hypot(
            points[supported, 0] - refined[0],
            points[supported, 1] - refined[1],
        )
        residual = float(np.median(np.abs(distances - refined[2])))
    return refined, support, median, residual


def score_fixed_circle_edge_support(
    image: np.ndarray,
    circle: tuple[float, float, float],
    *,
    radial_band_px: float = 4.0,
) -> dict[str, Any]:
    """Measure image support around a frozen circle without refining its geometry."""

    gradient = _gradient_magnitude(image)
    points, peaks = _radial_evidence(
        gradient,
        circle,
        radial_band_px=radial_band_px,
    )
    positive = peaks[peaks > 0]
    cutoff = max(float(np.percentile(gradient, 85.0)) * 0.35, 1.0)
    supported = peaks >= cutoff
    distances = np.hypot(points[:, 0] - circle[0], points[:, 1] - circle[1])
    offsets = distances - circle[2]
    return {
        "status": "measured",
        "method": "fixed_circle_radial_gradient_support_v1",
        "geometry_frozen": True,
        "radial_band_px": float(radial_band_px),
        "angular_sample_count": int(len(peaks)),
        "angular_edge_support_fraction": float(np.mean(supported)),
        "median_radial_gradient": (
            float(np.median(positive)) if len(positive) else 0.0
        ),
        "median_absolute_radial_offset_px": (
            float(np.median(np.abs(offsets[supported])))
            if int(np.count_nonzero(supported))
            else float(radial_band_px)
        ),
        "signed_median_radial_offset_px": (
            float(np.median(offsets[supported]))
            if int(np.count_nonzero(supported))
            else 0.0
        ),
    }


def _deduplicate_candidates(
    candidates: Sequence[tuple[float, float, float]], *, tolerance_px: float
) -> list[tuple[float, float, float]]:
    kept: list[tuple[float, float, float]] = []
    for candidate in candidates:
        if any(math.dist(candidate, prior) <= tolerance_px for prior in kept):
            continue
        kept.append(candidate)
    return kept


def fit_dish_circle(
    composite: np.ndarray,
    *,
    coarse_max_dimension_px: int = 2048,
) -> tuple[CircleFit, np.ndarray]:
    """Fit a provisional circle without acquisition geometry as an input."""

    image = np.asarray(composite, dtype=np.uint8)
    if image.ndim != 2:
        raise ValueError("composite must be a 2D uint8 image")
    height, width = image.shape
    if min(height, width) < 128:
        raise ValueError("composite is too small for a dish-rim fit")
    if coarse_max_dimension_px < 256:
        raise ValueError("coarse_max_dimension_px must be at least 256")

    scale = min(1.0, float(coarse_max_dimension_px) / float(max(height, width)))
    coarse_width = max(1, int(round(width * scale)))
    coarse_height = max(1, int(round(height * scale)))
    scale_x = coarse_width / width
    scale_y = coarse_height / height
    if not math.isclose(scale_x, scale_y, rel_tol=0.0, abs_tol=5e-4):
        raise ValueError(
            "coarse resize is not sufficiently isotropic for circle fitting"
        )
    coarse = cv2.resize(
        image, (coarse_width, coarse_height), interpolation=cv2.INTER_AREA
    )
    blurred = cv2.GaussianBlur(coarse, (0, 0), 2.0)
    min_dimension = min(coarse.shape)
    raw_candidates: list[tuple[float, float, float]] = []
    for param2 in (64.0, 48.0, 36.0, 28.0):
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.25,
            minDist=0.35 * min_dimension,
            param1=120.0,
            param2=param2,
            minRadius=int(round(0.35 * min_dimension)),
            maxRadius=int(round(0.505 * min_dimension)),
        )
        if circles is not None:
            raw_candidates.extend(
                tuple(float(value) for value in row) for row in circles[0]
            )
    candidates = _deduplicate_candidates(raw_candidates, tolerance_px=8.0)
    if not candidates:
        raise RuntimeError("no coarse dish-circle candidates were found")

    coarse_gradient = _gradient_magnitude(coarse)
    ranked: list[tuple[float, tuple[float, float, float], float, float, float]] = []
    for candidate in candidates:
        refined, support, median, residual = _refine_and_score_circle(
            coarse_gradient, candidate, radial_band_px=10.0
        )
        score = median * (0.25 + support)
        ranked.append((score, refined, support, median, residual))
    ranked.sort(key=lambda item: item[0], reverse=True)

    inverse_scale = 1.0 / ((scale_x + scale_y) * 0.5)
    full_gradient = _gradient_magnitude(image)
    full_candidates: list[CircleCandidate] = []
    for index, (_score, coarse_circle, _support, _median, _residual) in enumerate(
        ranked
    ):
        full_circle = (
            coarse_circle[0] / scale_x,
            coarse_circle[1] / scale_y,
            coarse_circle[2] * inverse_scale,
        )
        refined, support, median, residual = _refine_and_score_circle(
            full_gradient,
            full_circle,
            radial_band_px=max(12.0, 18.0 * inverse_scale),
        )
        cx, cy, radius = refined
        if (
            not all(math.isfinite(value) for value in refined)
            or radius <= 0
            or not (
                -0.05 * width <= cx <= 1.05 * width
                and -0.05 * height <= cy <= 1.05 * height
            )
        ):
            continue
        full_candidates.append(
            CircleCandidate(
                candidate_id=f"candidate_{index:03d}",
                center_x_px=cx,
                center_y_px=cy,
                radius_px=radius,
                angular_support_fraction=support,
                radial_residual_px=residual,
                median_radial_gradient=median,
                evidence_score=median * (0.25 + support),
            )
        )
    if not full_candidates:
        raise RuntimeError("dish-circle refinement produced no valid candidates")
    full_candidates.sort(key=lambda item: item.evidence_score, reverse=True)
    selected = full_candidates[0]
    fit = CircleFit(
        selected.center_x_px,
        selected.center_y_px,
        selected.radius_px,
        selected.angular_support_fraction,
        selected.median_radial_gradient,
        len(full_candidates),
        radial_residual_px=selected.radial_residual_px,
        selected_candidate_id=selected.candidate_id,
        selection_reason="highest_frozen_radial_evidence_score_v1",
        frozen_candidates=tuple(full_candidates),
    )
    edge = np.clip(
        full_gradient / max(float(np.percentile(full_gradient, 99.5)), 1.0) * 255.0,
        0,
        255,
    )
    return fit, edge.astype(np.uint8)


def consensus_circle(fits: Sequence[CircleFit]) -> CircleFit:
    if len(fits) < 1:
        raise ValueError("at least one fit is required")
    return CircleFit(
        center_x_px=float(np.median([fit.center_x_px for fit in fits])),
        center_y_px=float(np.median([fit.center_y_px for fit in fits])),
        radius_px=float(np.median([fit.radius_px for fit in fits])),
        angular_support_fraction=float(
            np.median([fit.angular_support_fraction for fit in fits])
        ),
        median_radial_gradient=float(
            np.median([fit.median_radial_gradient for fit in fits])
        ),
        candidate_count=sum(fit.candidate_count for fit in fits),
        radial_residual_px=float(np.median([fit.radial_residual_px for fit in fits])),
        selected_candidate_id=None,
        selection_reason="median_of_window_selected_candidates_v1",
    )


def _draw_circle(
    image: np.ndarray,
    circle: tuple[float, float, float],
    *,
    color: tuple[int, int, int],
    label: str,
) -> np.ndarray:
    output = cv2.cvtColor(np.asarray(image, dtype=np.uint8), cv2.COLOR_GRAY2BGR)
    cx, cy, radius = circle
    thickness = max(2, int(round(max(output.shape[:2]) / 1500)))
    cv2.circle(
        output, (round(cx), round(cy)), round(radius), color, thickness, cv2.LINE_AA
    )
    cv2.drawMarker(
        output,
        (round(cx), round(cy)),
        color,
        markerType=cv2.MARKER_CROSS,
        markerSize=24,
        thickness=thickness,
    )
    cv2.putText(
        output,
        label,
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        thickness,
        cv2.LINE_AA,
    )
    return output


def _write_png(path: Path, image: np.ndarray) -> str:
    if not cv2.imwrite(str(path), np.asarray(image)):
        raise RuntimeError(f"failed to write PNG: {path}")
    return _sha256_file(path)


def _load_summary(path: Path) -> tuple[int, float, int, int, str]:
    payload = json.loads(path.read_text())
    frame_count = int(
        payload.get("frames_received") or payload["merged_output"]["packets_written"]
    )
    fps = float(payload["fps"])
    geometry = payload["video_metadata"]["geometry"]
    width = int(geometry["source_width"])
    height = int(geometry["source_height"])
    serial = str(payload["video_metadata"]["camera_serial"])
    if frame_count <= 0 or fps <= 0 or width <= 0 or height <= 0 or not serial:
        raise ValueError("external summary contains invalid source metadata")
    return frame_count, fps, width, height, serial


def decode_keyframe_window_medians_pynvvc(
    video_path: Path,
    specs: Sequence[WindowSpec],
    *,
    expected_shape_hw: tuple[int, int],
    gpu_id: int,
) -> tuple[dict[str, np.ndarray], dict[str, str], dict[str, Any]]:
    """Decode only the exact declared keyframes selected for each window."""

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
        raise RuntimeError(
            "video dimensions disagree with the external summary: "
            f"video={(source_height, source_width)} summary={expected_shape_hw}"
        )
    medians: dict[str, np.ndarray] = {}
    frame_hashes: dict[str, str] = {}
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

    for spec in specs:
        stack = np.empty(
            (len(spec.frame_indices), source_height, source_width), dtype=np.uint8
        )
        hasher = hashlib.sha256()
        for row, target in enumerate(spec.frame_indices):
            decoder = nvc.CreateDecoder(
                gpuid=int(gpu_id),
                codec=demuxer.GetNvCodecId(),
                usedevicememory=True,
            )
            frame, proof = decode_one_frame_from_preceding_keyframe(
                demuxer=demuxer,
                decoder=decoder,
                target_frame_index=target,
                materialize_frame=materialize,
            )
            if proof["target_packet_number"] != 1:
                raise RuntimeError(
                    f"declared keyframe {target} did not resolve as the first seek packet"
                )
            if frame.shape != (source_height, source_width):
                raise RuntimeError(
                    f"decoded keyframe {target} has unexpected shape {frame.shape}"
                )
            stack[row] = frame
            frame_bytes = frame.tobytes(order="C")
            frame_sha256 = _sha256_bytes(frame_bytes)
            hasher.update(frame_bytes)
            seeks.append(
                {
                    "window": spec.name,
                    "decoded_frame_sha256": frame_sha256,
                    **proof,
                }
            )
            del decoder, frame
        medians[spec.name] = temporal_median(stack)
        frame_hashes[spec.name] = hasher.hexdigest()

    packet_counts = [
        int(item["packets_submitted_through_target_output"]) for item in seeks
    ]
    metadata = {
        "backend": "pynvvc_luma_declared_keyframes_only",
        "gpu_id": int(gpu_id),
        "requested_frame_count": len(seeks),
        "seek_count": len(seeks),
        "decoded_packet_count_total": sum(packet_counts),
        "decoded_packet_count_max_per_seek": max(packet_counts),
        "seeks": seeks,
        "elapsed_seconds": time.perf_counter() - started,
        "demuxer_frame_rate": float(demuxer.FrameRate()),
        "codec": str(demuxer.GetNvCodecId()),
    }
    return medians, frame_hashes, metadata


def decode_clipped_keyframe_window_medians_pynvvc(
    specs: Sequence[ClippedWindowSpec],
    *,
    expected_shape_hw: tuple[int, int],
    expected_fps: float,
    gpu_id: int,
) -> tuple[dict[str, np.ndarray], dict[str, str], dict[str, Any]]:
    """Decode recording-wide window samples from their owning clip videos."""

    try:
        import PyNvVideoCodec as nvc  # type: ignore
        import torch
    except Exception as exc:  # pragma: no cover - cluster environment dependent
        raise RuntimeError(
            f"PyNvVideoCodec CUDA decode dependencies are unavailable: {exc}"
        ) from exc

    source_height, source_width = expected_shape_hw
    demuxers: dict[Path, Any] = {}
    source_metadata: dict[Path, dict[str, Any]] = {}

    def demuxer_for(path: Path) -> Any:
        demuxer = demuxers.get(path)
        if demuxer is not None:
            return demuxer
        demuxer = nvc.CreateDemuxer(filename=str(path))
        observed_shape = (int(demuxer.Height()), int(demuxer.Width()))
        if observed_shape != expected_shape_hw:
            raise RuntimeError(
                "clip video dimensions disagree with recording geometry: "
                f"video={path}, observed={observed_shape}, expected={expected_shape_hw}"
            )
        observed_fps = float(demuxer.FrameRate())
        if not math.isclose(observed_fps, expected_fps, rel_tol=0.0, abs_tol=1e-9):
            raise RuntimeError(
                "clip video frame rate disagrees with keyframe metadata: "
                f"video={path}, observed={observed_fps}, expected={expected_fps}"
            )
        demuxers[path] = demuxer
        source_metadata[path] = {
            "video_path": str(path),
            "frame_rate": observed_fps,
            "codec": str(demuxer.GetNvCodecId()),
            "height": observed_shape[0],
            "width": observed_shape[1],
        }
        return demuxer

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

    medians: dict[str, np.ndarray] = {}
    frame_hashes: dict[str, str] = {}
    seeks: list[dict[str, Any]] = []
    started = time.perf_counter()
    for spec in specs:
        stack = np.empty(
            (len(spec.frames), source_height, source_width), dtype=np.uint8
        )
        hasher = hashlib.sha256()
        for row, reference in enumerate(spec.frames):
            demuxer = demuxer_for(reference.video_path)
            decoder = nvc.CreateDecoder(
                gpuid=int(gpu_id),
                codec=demuxer.GetNvCodecId(),
                usedevicememory=True,
            )
            frame, proof = decode_one_frame_from_preceding_keyframe(
                demuxer=demuxer,
                decoder=decoder,
                target_frame_index=reference.clip_local_frame_index,
                materialize_frame=materialize,
            )
            if proof["target_packet_number"] != 1:
                raise RuntimeError(
                    f"declared keyframe {reference.clip_id}:"
                    f"{reference.clip_local_frame_index} did not resolve as the first seek packet"
                )
            if frame.shape != (source_height, source_width):
                raise RuntimeError(
                    f"decoded keyframe has unexpected shape {frame.shape}: {reference}"
                )
            stack[row] = frame
            frame_bytes = frame.tobytes(order="C")
            frame_sha256 = _sha256_bytes(frame_bytes)
            hasher.update(frame_bytes)
            seeks.append(
                {
                    **proof,
                    "window": spec.name,
                    "clip_id": reference.clip_id,
                    "clip_index": reference.clip_index,
                    "video_path": str(reference.video_path),
                    "keyframe_path": str(reference.keyframe_path),
                    "clip_local_frame_index": reference.clip_local_frame_index,
                    "recording_frame_id": reference.recording_frame_id,
                    "decoded_frame_sha256": frame_sha256,
                }
            )
            del decoder, frame
        medians[spec.name] = temporal_median(stack)
        frame_hashes[spec.name] = hasher.hexdigest()

    packet_counts = [
        int(item["packets_submitted_through_target_output"]) for item in seeks
    ]
    metadata = {
        "backend": "pynvvc_luma_clipped_declared_keyframes_only",
        "gpu_id": int(gpu_id),
        "requested_frame_count": len(seeks),
        "seek_count": len(seeks),
        "decoded_packet_count_total": sum(packet_counts),
        "decoded_packet_count_max_per_seek": max(packet_counts),
        "seeks": seeks,
        "elapsed_seconds": time.perf_counter() - started,
        "source_video_count": len(demuxers),
        "source_videos": [source_metadata[path] for path in sorted(source_metadata)],
    }
    return medians, frame_hashes, metadata


def _circle_tuple(payload: Mapping[str, Any]) -> tuple[float, float, float]:
    geometry = payload["geometry"]
    center = geometry["center_px"]
    result = (float(center["x"]), float(center["y"]), float(geometry["radius_px"]))
    if not all(math.isfinite(value) for value in result) or result[2] <= 0:
        raise ValueError("circle geometry is invalid")
    return result


def render_acquisition_reveal(
    *,
    output_dir: Path,
    observation_path: Path,
    fit_report_path: Path,
    composites: Mapping[str, np.ndarray],
) -> Path:
    """Render comparison files after the independent fit report is immutable."""

    fit_report_bytes = fit_report_path.read_bytes()
    fit_report = json.loads(fit_report_bytes)
    observation_bytes = observation_path.read_bytes()
    observation = json.loads(observation_bytes)
    acquisition = _circle_tuple(observation["accepted_inner_rim_boundary"])
    expected_shape = (
        int(observation["camera"]["height"]),
        int(observation["camera"]["width"]),
    )
    reveal_files: dict[str, dict[str, Any]] = {}
    support_by_window: dict[str, dict[str, Any]] = {}
    for name in WINDOW_NAMES:
        image = composites[name]
        if image.shape != expected_shape:
            raise ValueError(
                f"acquisition observation shape {expected_shape} disagrees with {name} composite {image.shape}"
            )
        palette = _circle_tuple(fit_report["windows"][name]["fit"])
        overlay = _draw_circle(
            image,
            palette,
            color=(255, 255, 0),
            label="Palette blind fit (cyan)",
        )
        cv2.circle(
            overlay,
            (round(acquisition[0]), round(acquisition[1])),
            round(acquisition[2]),
            (0, 165, 255),
            max(2, int(round(max(image.shape) / 1500))),
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            "Acquisition accepted inner rim (orange)",
            (30, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 165, 255),
            max(2, int(round(max(image.shape) / 1500))),
            cv2.LINE_AA,
        )
        path = output_dir / f"{name}_acquisition_reveal.png"
        digest = _write_png(path, overlay)
        reveal_files[name] = {
            "path": path.name,
            "sha256": digest,
            "delta_center_x_px": palette[0] - acquisition[0],
            "delta_center_y_px": palette[1] - acquisition[1],
            "delta_radius_px": palette[2] - acquisition[2],
        }
        support_by_window[name] = score_fixed_circle_edge_support(
            image,
            acquisition,
        )

    support_values = [
        support_by_window[name]["angular_edge_support_fraction"]
        for name in WINDOW_NAMES
    ]
    residual_values = [
        support_by_window[name]["median_absolute_radial_offset_px"]
        for name in WINDOW_NAMES
    ]
    gradient_values = [
        support_by_window[name]["median_radial_gradient"] for name in WINDOW_NAMES
    ]
    acquisition_support = {
        "status": "measured",
        "method": "fixed_circle_radial_gradient_support_v1",
        "fit_frozen_before_measurement": True,
        "coordinate_space": "camera_native_pixels",
        "geometry": observation["accepted_inner_rim_boundary"]["geometry"],
        "source_observation_sha256": _sha256_bytes(observation_bytes),
        "windows": support_by_window,
        "median_angular_edge_support_fraction": float(np.median(support_values)),
        "minimum_angular_edge_support_fraction": float(min(support_values)),
        "median_absolute_radial_offset_px": float(np.median(residual_values)),
        "median_radial_gradient": float(np.median(gradient_values)),
    }

    reveal = {
        "schema_id": f"{SCHEMA_ID}.acquisition_reveal",
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "fit_report": {
            "path": fit_report_path.name,
            "sha256": _sha256_bytes(fit_report_bytes),
        },
        "acquisition_observation": {
            "path": str(observation_path),
            "sha256": _sha256_bytes(observation_bytes),
            "artifact_id": observation.get("artifact_id"),
            "accepted_inner_rim_boundary": observation["accepted_inner_rim_boundary"],
        },
        "files": reveal_files,
        "acquisition_boundary_edge_support": acquisition_support,
        "purpose": "visual_reveal_only_after_blind_palette_fit_was_frozen",
        "prohibitions": [
            "not_a_mask_selection",
            "not_a_detection_gate",
            "not_a_zarr_or_registry_publication",
        ],
    }
    path = output_dir / "acquisition_reveal.json"
    _atomic_json(path, reveal)
    return path


def write_review_package(output_dir: Path, *, acquisition_revealed: bool) -> Path:
    """Create one bounded montage and receipt for the mandatory review barrier."""

    source_suffix = "acquisition_reveal" if acquisition_revealed else "palette_fit"
    source_paths = [output_dir / f"{name}_{source_suffix}.png" for name in WINDOW_NAMES]
    images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in source_paths]
    if any(image is None for image in images):
        missing = [
            str(path) for path, image in zip(source_paths, images) if image is None
        ]
        raise RuntimeError(f"review montage inputs are unreadable: {missing}")
    typed_images = [np.asarray(image) for image in images]
    max_height = 1200
    resized: list[np.ndarray] = []
    for image in typed_images:
        scale = min(1.0, max_height / float(image.shape[0]))
        width = max(1, int(round(image.shape[1] * scale)))
        height = max(1, int(round(image.shape[0] * scale)))
        resized.append(
            cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            if scale < 1.0
            else image
        )
    target_height = min(image.shape[0] for image in resized)
    normalized = [
        (
            image[:target_height]
            if image.shape[0] == target_height
            else cv2.resize(
                image,
                (round(image.shape[1] * target_height / image.shape[0]), target_height),
                interpolation=cv2.INTER_AREA,
            )
        )
        for image in resized
    ]
    montage = np.hstack(normalized)
    montage_path = output_dir / "dish_rim_review_montage.png"
    montage_sha256 = _write_png(montage_path, montage)
    fit_report = output_dir / "fit_report.json"
    receipt = {
        "schema_id": f"{SCHEMA_ID}.review_package",
        "schema_version": SCHEMA_VERSION,
        "status": "awaiting_explicit_human_review",
        "created_at_utc": _utc_now(),
        "fit_report": {
            "path": fit_report.name,
            "sha256": _sha256_file(fit_report),
        },
        "montage": {
            "path": montage_path.name,
            "sha256": montage_sha256,
            "shape": [int(value) for value in montage.shape],
            "source": source_suffix,
        },
        "source_panels": [
            {"path": path.name, "sha256": _sha256_file(path)} for path in source_paths
        ],
        "human_review_required_before": [
            "palette_candidate_publication",
            "candidate_comparison",
            "operational_geometry_selection",
            "detection_gating",
        ],
    }
    receipt_path = output_dir / "review_package.json"
    _atomic_json(receipt_path, receipt)
    return receipt_path


def _clipped_source_report(
    source: ClippedRecordingSource,
    specs: Sequence[ClippedWindowSpec],
) -> dict[str, Any]:
    geometry_snapshot_path = (
        source.recording_dir
        / "raw"
        / "recording_geometry_bundle"
        / "recording_snapshot.json"
    )
    sampled_by_clip: dict[str, ClipVideoSource] = {}
    clips_by_id = {clip.clip_id: clip for clip in source.clips}
    for spec in specs:
        for frame in spec.frames:
            sampled_by_clip[frame.clip_id] = clips_by_id[frame.clip_id]
    sampled_clips = []
    for clip_id in sorted(sampled_by_clip):
        clip = sampled_by_clip[clip_id]
        video_stat = clip.video_path.stat()
        sampled_clips.append(
            {
                "clip_id": clip.clip_id,
                "clip_index": clip.clip_index,
                "first_recording_frame_id": clip.first_recording_frame_id,
                "last_recording_frame_id": clip.last_recording_frame_id,
                "frame_count": clip.frame_count,
                "video_path": str(clip.video_path),
                "video_size_bytes": video_stat.st_size,
                "video_mtime_ns": video_stat.st_mtime_ns,
                "keyframe_summary_path": str(clip.keyframe_path),
                "keyframe_summary_sha256": _sha256_file(clip.keyframe_path),
                "declared_keyframe_count": len(clip.keyframe_frames),
            }
        )
    return {
        "mode": "clipped_recording",
        "recording_dir": str(source.recording_dir),
        "recording_id": source.recording_id,
        "session_id": source.session_id,
        "recording_clip_index_path": str(source.clip_index_path),
        "recording_clip_index_sha256": _sha256_file(source.clip_index_path),
        "recording_geometry_snapshot_path": str(geometry_snapshot_path),
        "recording_geometry_snapshot_sha256": _sha256_file(geometry_snapshot_path),
        "camera_serial": source.camera_serial,
        "clip_count": len(source.clips),
        "sampled_clip_count": len(sampled_clips),
        "sampled_clips": sampled_clips,
        "first_recording_frame_id": source.first_recording_frame_id,
        "last_recording_frame_id": source.last_recording_frame_id,
        "frame_count": source.frame_count,
        "fps": source.fps,
        "image_shape_px": {"height": source.height, "width": source.width},
        "pixel_contract": "orange.camera.mono8.full_frame.v1",
        "source_binding": (
            "recording_clip_index plus per-clip declared keyframe summaries and "
            "decoded-frame hashes"
        ),
    }


def _window_sampling_report(
    spec: WindowSpec | ClippedWindowSpec,
    *,
    decode: Mapping[str, Any],
) -> dict[str, Any]:
    decoded = [item for item in decode["seeks"] if item["window"] == spec.name]
    if isinstance(spec, WindowSpec):
        return {
            "fraction": spec.fraction,
            "center_frame": spec.center_frame,
            "frame_indices": list(spec.frame_indices),
            "decoded_frames": [
                {
                    "frame_index": int(item["target_frame_index"]),
                    "decoded_frame_sha256": item["decoded_frame_sha256"],
                }
                for item in decoded
            ],
        }
    return {
        "fraction": spec.fraction,
        "center_recording_frame_id": spec.center_recording_frame_id,
        "recording_frame_ids": [frame.recording_frame_id for frame in spec.frames],
        "sampled_clip_ids": sorted({frame.clip_id for frame in spec.frames}),
        "decoded_frames": [
            {
                "clip_id": item["clip_id"],
                "clip_index": item["clip_index"],
                "video_path": item["video_path"],
                "keyframe_path": item["keyframe_path"],
                "clip_local_frame_index": item["clip_local_frame_index"],
                "recording_frame_id": item["recording_frame_id"],
                "decoded_frame_sha256": item["decoded_frame_sha256"],
            }
            for item in decoded
        ],
    }


def _validate_probe_source_args(args: argparse.Namespace) -> str:
    has_video = args.video is not None
    has_recording = args.recording_dir is not None
    if has_video == has_recording:
        raise ValueError("choose exactly one source mode: --video or --recording-dir")
    if has_recording:
        if args.summary is not None or args.keyframes is not None:
            raise ValueError(
                "--summary and --keyframes belong to --video mode; clipped mode "
                "discovers them from recording_clip_index.json"
            )
        return "clipped_recording"
    if args.summary is None or args.keyframes is None:
        raise ValueError("--video mode requires both --summary and --keyframes")
    return "single_video"


def run_probe(args: argparse.Namespace) -> Path:
    source_mode = _validate_probe_source_args(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing existing output directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp.", dir=output_dir.parent)
    )

    try:
        if source_mode == "single_video":
            video_path = Path(args.video).expanduser().resolve()
            summary_path = Path(args.summary).expanduser().resolve()
            keyframe_path = Path(args.keyframes).expanduser().resolve()
            if not video_path.is_file():
                raise FileNotFoundError(video_path)
            if not summary_path.is_file():
                raise FileNotFoundError(summary_path)
            if not keyframe_path.is_file():
                raise FileNotFoundError(keyframe_path)
            frame_count, fps, width, height, camera_serial = _load_summary(summary_path)
            declared_keyframes = load_declared_keyframes(
                keyframe_path,
                expected_frame_count=frame_count,
                expected_fps=fps,
            )
            specs: tuple[WindowSpec, ...] | tuple[ClippedWindowSpec, ...] = (
                build_keyframe_window_specs(
                    frame_count=frame_count,
                    fps=fps,
                    keyframe_frames=declared_keyframes,
                    max_keyframes_per_window=args.max_keyframes_per_window,
                    span_seconds=args.span_seconds,
                )
            )
            composites, frame_hashes, decode = decode_keyframe_window_medians_pynvvc(
                video_path,
                specs,
                expected_shape_hw=(height, width),
                gpu_id=args.gpu_id,
            )
            source_report = {
                "mode": "single_video",
                "video_path": str(video_path),
                "video_size_bytes": video_path.stat().st_size,
                "video_mtime_ns": video_path.stat().st_mtime_ns,
                "video_sha256": _sha256_file(video_path),
                "summary_path": str(summary_path),
                "summary_sha256": _sha256_file(summary_path),
                "keyframe_summary_path": str(keyframe_path),
                "keyframe_summary_sha256": _sha256_file(keyframe_path),
                "declared_keyframe_count": len(declared_keyframes),
                "camera_serial": camera_serial,
                "frame_count": frame_count,
                "fps": fps,
                "image_shape_px": {"height": height, "width": width},
                "pixel_contract": "orange.camera.mono8.full_frame.v1",
            }
            sampling_policy = "declared_keyframes_only"
        else:
            clipped_source = load_clipped_recording_source(args.recording_dir)
            specs = build_clipped_keyframe_window_specs(
                clipped_source,
                max_keyframes_per_window=args.max_keyframes_per_window,
                span_seconds=args.span_seconds,
            )
            composites, frame_hashes, decode = (
                decode_clipped_keyframe_window_medians_pynvvc(
                    specs,
                    expected_shape_hw=(clipped_source.height, clipped_source.width),
                    expected_fps=clipped_source.fps,
                    gpu_id=args.gpu_id,
                )
            )
            source_report = _clipped_source_report(clipped_source, specs)
            sampling_policy = (
                "recording_clip_index_declared_keyframes_on_continuous_recording_clock"
            )

        windows: dict[str, Any] = {}
        fits: list[CircleFit] = []
        for spec in specs:
            composite = composites[spec.name]
            fit, edge = fit_dish_circle(
                composite, coarse_max_dimension_px=args.coarse_max_dimension_px
            )
            fits.append(fit)
            composite_path = temporary / f"{spec.name}_temporal_median.png"
            overlay_path = temporary / f"{spec.name}_palette_fit.png"
            edge_path = temporary / f"{spec.name}_edge_evidence.png"
            files = {
                "temporal_median": {
                    "path": composite_path.name,
                    "sha256": _write_png(composite_path, composite),
                },
                "palette_fit": {
                    "path": overlay_path.name,
                    "sha256": _write_png(
                        overlay_path,
                        _draw_circle(
                            composite,
                            (fit.center_x_px, fit.center_y_px, fit.radius_px),
                            color=(255, 255, 0),
                            label=f"Palette blind fit: {spec.name}",
                        ),
                    ),
                },
                "edge_evidence": {
                    "path": edge_path.name,
                    "sha256": _write_png(edge_path, edge),
                },
            }
            windows[spec.name] = {
                **_window_sampling_report(spec, decode=decode),
                "decoded_luma_sequence_sha256": frame_hashes[spec.name],
                "composite_pixel_sha256": _sha256_bytes(composite.tobytes(order="C")),
                "fit": fit.to_json(),
                "files": files,
            }

        consensus = consensus_circle(fits)
        temporal_stability = {
            "center_x_range": float(
                max(fit.center_x_px for fit in fits)
                - min(fit.center_x_px for fit in fits)
            ),
            "center_y_range": float(
                max(fit.center_y_px for fit in fits)
                - min(fit.center_y_px for fit in fits)
            ),
            "radius_range": float(
                max(fit.radius_px for fit in fits) - min(fit.radius_px for fit in fits)
            ),
        }
        report = {
            "schema_id": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "status": "provisional_visual_review_required",
            "created_at_utc": _utc_now(),
            "fit_frozen_before_acquisition_reveal": True,
            "fit_method": FIT_METHOD,
            "target_feature": TARGET_FEATURE,
            "target_plane": TARGET_PLANE,
            "source": source_report,
            "parameters": {
                "sampling_policy": sampling_policy,
                "max_keyframes_per_window": args.max_keyframes_per_window,
                "actual_keyframes_per_window": {
                    spec.name: (
                        len(spec.frame_indices)
                        if isinstance(spec, WindowSpec)
                        else len(spec.frames)
                    )
                    for spec in specs
                },
                "span_seconds_per_window": args.span_seconds,
                "window_fractions": list(WINDOW_FRACTIONS),
                "coarse_max_dimension_px": args.coarse_max_dimension_px,
                "acquisition_geometry_available_to_fitter": False,
            },
            "decode": decode,
            "windows": windows,
            "consensus_fit": consensus.to_json(),
            "temporal_stability_px": temporal_stability,
            "fit_evidence_contract": {
                "all_window_candidates_frozen": True,
                "candidate_geometry_revealed_to_acquisition_fit": False,
                "candidate_feature_classification": (
                    "unclassified_concentric_rim_edge"
                ),
                "selection_scope": "window_consensus_for_review_not_operational_selection",
            },
            "environment": {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "lsf_job_id": os.environ.get("LSB_JOBID"),
                "numpy_version": np.__version__,
                "opencv_version": cv2.__version__,
            },
            "prohibitions": [
                "not_a_mask_selection",
                "not_a_detection_gate",
                "not_a_zarr_or_registry_publication",
                "do_not_advance_without_visual_review",
            ],
        }
        fit_report_path = temporary / "fit_report.json"
        _atomic_json(fit_report_path, report)

        acquisition_revealed = args.acquisition_observation is not None
        if acquisition_revealed:
            render_acquisition_reveal(
                output_dir=temporary,
                observation_path=Path(args.acquisition_observation)
                .expanduser()
                .resolve(),
                fit_report_path=fit_report_path,
                composites=composites,
            )
        write_review_package(
            temporary,
            acquisition_revealed=acquisition_revealed,
        )
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_dir / "fit_report.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate blind early/middle/late dish-rim fit diagnostics."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path)
    source.add_argument(
        "--recording-dir",
        type=Path,
        help=(
            "One-camera rolling-clips recording. Video and keyframe paths are "
            "resolved from recording_clip_index.json."
        ),
    )
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--keyframes", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--acquisition-observation",
        type=Path,
        help="Optional reveal-only observation; opened after fit_report.json is frozen.",
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--max-keyframes-per-window",
        "--sample-count",
        dest="max_keyframes_per_window",
        type=int,
        default=21,
        help="Maximum odd number of declared keyframes used in each window.",
    )
    parser.add_argument("--span-seconds", type=float, default=5.0)
    parser.add_argument("--coarse-max-dimension-px", type=int, default=2048)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_probe_source_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    report = run_probe(args)
    print(json.dumps({"status": "complete", "fit_report": str(report)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
