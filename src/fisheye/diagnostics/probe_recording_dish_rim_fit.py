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
import socket
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


SCHEMA_ID = "palette.diagnostics.recording_dish_rim_probe"
SCHEMA_VERSION = 1
FIT_METHOD = "temporal_median_multicandidate_radial_edge_circle_v1"
TARGET_FEATURE = "dish_inner_rim_water_side_edge"
TARGET_PLANE = "dish_top_rim"
WINDOW_NAMES = ("early", "middle", "late")
WINDOW_FRACTIONS = (0.10, 0.50, 0.90)


@dataclass(frozen=True)
class CircleFit:
    center_x_px: float
    center_y_px: float
    radius_px: float
    angular_support_fraction: float
    median_radial_gradient: float
    candidate_count: int

    def to_json(self) -> dict[str, Any]:
        return {
            "geometry": {
                "type": "circle",
                "center_px": {"x": self.center_x_px, "y": self.center_y_px},
                "radius_px": self.radius_px,
            },
            "coordinate_space": "camera_native_pixels",
            "target_feature": TARGET_FEATURE,
            "target_plane": TARGET_PLANE,
            "angular_support_fraction": self.angular_support_fraction,
            "median_radial_gradient": self.median_radial_gradient,
            "candidate_count": self.candidate_count,
        }


@dataclass(frozen=True)
class WindowSpec:
    name: str
    fraction: float
    center_frame: int
    frame_indices: tuple[int, ...]


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


def build_window_specs(
    *,
    frame_count: int,
    fps: float,
    sample_count: int = 61,
    span_seconds: float = 5.0,
    fractions: Sequence[float] = WINDOW_FRACTIONS,
) -> tuple[WindowSpec, ...]:
    """Return exact, bounded frame indices for early/middle/late windows."""

    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if sample_count < 3 or sample_count % 2 != 1:
        raise ValueError("sample_count must be an odd integer >= 3")
    if not math.isfinite(span_seconds) or span_seconds <= 0:
        raise ValueError("span_seconds must be finite and positive")
    if len(fractions) != len(WINDOW_NAMES):
        raise ValueError(f"exactly {len(WINDOW_NAMES)} window fractions are required")

    half_span = 0.5 * span_seconds * fps
    specs: list[WindowSpec] = []
    occupied: set[int] = set()
    for name, fraction in zip(WINDOW_NAMES, fractions, strict=True):
        if not math.isfinite(float(fraction)) or not 0.0 < float(fraction) < 1.0:
            raise ValueError("window fractions must lie strictly between zero and one")
        center = int(round(float(fraction) * (frame_count - 1)))
        raw = np.linspace(center - half_span, center + half_span, sample_count)
        indices = np.rint(raw).astype(np.int64)
        indices = np.clip(indices, 0, frame_count - 1)
        unique = tuple(int(value) for value in np.unique(indices))
        if len(unique) != sample_count:
            raise ValueError(
                f"recording is too short for {sample_count} unique samples in {name} window"
            )
        overlap = occupied.intersection(unique)
        if overlap:
            raise ValueError(f"temporal windows overlap at frame {min(overlap)}")
        occupied.update(unique)
        specs.append(WindowSpec(name, float(fraction), center, unique))
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
) -> tuple[tuple[float, float, float], float, float]:
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
        return refined, 0.0, 0.0
    support_cutoff = max(float(np.percentile(gradient, 85.0)) * 0.35, 1.0)
    support = float(np.mean(peaks >= support_cutoff))
    median = float(np.median(positive))
    return refined, support, median


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
    ranked: list[tuple[float, tuple[float, float, float], float, float]] = []
    for candidate in candidates:
        refined, support, median = _refine_and_score_circle(
            coarse_gradient, candidate, radial_band_px=10.0
        )
        score = median * (0.25 + support)
        ranked.append((score, refined, support, median))
    ranked.sort(key=lambda item: item[0], reverse=True)
    _, coarse_circle, _, _ = ranked[0]

    inverse_scale = 1.0 / ((scale_x + scale_y) * 0.5)
    full_circle = (
        coarse_circle[0] / scale_x,
        coarse_circle[1] / scale_y,
        coarse_circle[2] * inverse_scale,
    )
    full_gradient = _gradient_magnitude(image)
    refined, support, median = _refine_and_score_circle(
        full_gradient, full_circle, radial_band_px=max(12.0, 18.0 * inverse_scale)
    )
    cx, cy, radius = refined
    if not all(math.isfinite(value) for value in refined) or radius <= 0:
        raise RuntimeError("dish-circle refinement produced invalid geometry")
    if not (
        -0.05 * width <= cx <= 1.05 * width and -0.05 * height <= cy <= 1.05 * height
    ):
        raise RuntimeError("dish-circle refinement placed the center outside the image")
    fit = CircleFit(cx, cy, radius, support, median, len(candidates))
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


def decode_window_medians_pynvvc(
    video_path: Path,
    specs: Sequence[WindowSpec],
    *,
    expected_shape_hw: tuple[int, int],
    gpu_id: int,
) -> tuple[dict[str, np.ndarray], dict[str, str], dict[str, Any]]:
    """Decode all windows in one presentation-order PyNvVC pass."""

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
    decoder = nvc.CreateDecoder(
        gpuid=int(gpu_id), codec=demuxer.GetNvCodecId(), usedevicememory=True
    )

    frame_to_target: dict[int, tuple[str, int]] = {}
    buffers: dict[str, np.ndarray] = {}
    hashers: dict[str, Any] = {}
    specs_by_name = {spec.name: spec for spec in specs}
    for spec in specs:
        hashers[spec.name] = hashlib.sha256()
        for row, frame_idx in enumerate(spec.frame_indices):
            frame_to_target[frame_idx] = (spec.name, row)

    max_requested = max(frame_to_target)
    copied = 0
    decoded_count = 0
    medians: dict[str, np.ndarray] = {}
    started = time.perf_counter()
    for packet in demuxer:
        for frame in decoder.Decode(packet):
            target = frame_to_target.get(decoded_count)
            if target is not None:
                name, row = target
                if name not in buffers:
                    buffers[name] = np.empty(
                        (
                            len(specs_by_name[name].frame_indices),
                            source_height,
                            source_width,
                        ),
                        dtype=np.uint8,
                    )
                tensor = torch.from_dlpack(frame)
                luma = (
                    tensor[:source_height, :]
                    .contiguous()
                    .cpu()
                    .numpy()
                    .astype(np.uint8, copy=False)
                )
                buffers[name][row] = luma
                hashers[name].update(luma.tobytes(order="C"))
                copied += 1
                del luma, tensor
                if row == len(specs_by_name[name].frame_indices) - 1:
                    medians[name] = temporal_median(buffers.pop(name))
            decoded_count += 1
            if decoded_count > max_requested:
                break
        if decoded_count > max_requested:
            break
    expected = sum(len(spec.frame_indices) for spec in specs)
    if copied != expected:
        raise RuntimeError(f"decoded {copied}/{expected} requested frames")

    frame_hashes: dict[str, str] = {}
    for spec in specs:
        if spec.name not in medians:
            raise RuntimeError(f"window {spec.name!r} did not complete")
        frame_hashes[spec.name] = hashers[spec.name].hexdigest()
    metadata = {
        "backend": "pynvvc_luma_sequential",
        "gpu_id": int(gpu_id),
        "decoded_frame_count_through_last_window": decoded_count,
        "requested_frame_count": expected,
        "elapsed_seconds": time.perf_counter() - started,
        "demuxer_frame_rate": float(demuxer.FrameRate()),
        "codec": str(demuxer.GetNvCodecId()),
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


def run_probe(args: argparse.Namespace) -> Path:
    video_path = Path(args.video).expanduser().resolve()
    summary_path = Path(args.summary).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    if output_dir.exists():
        raise FileExistsError(f"refusing existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)

    frame_count, fps, width, height, camera_serial = _load_summary(summary_path)
    specs = build_window_specs(
        frame_count=frame_count,
        fps=fps,
        sample_count=args.sample_count,
        span_seconds=args.span_seconds,
    )
    composites, frame_hashes, decode = decode_window_medians_pynvvc(
        video_path,
        specs,
        expected_shape_hw=(height, width),
        gpu_id=args.gpu_id,
    )

    windows: dict[str, Any] = {}
    fits: list[CircleFit] = []
    for spec in specs:
        composite = composites[spec.name]
        fit, edge = fit_dish_circle(
            composite, coarse_max_dimension_px=args.coarse_max_dimension_px
        )
        fits.append(fit)
        composite_path = output_dir / f"{spec.name}_temporal_median.png"
        overlay_path = output_dir / f"{spec.name}_palette_fit.png"
        edge_path = output_dir / f"{spec.name}_edge_evidence.png"
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
            "fraction": spec.fraction,
            "center_frame": spec.center_frame,
            "frame_indices": list(spec.frame_indices),
            "decoded_luma_sequence_sha256": frame_hashes[spec.name],
            "composite_pixel_sha256": _sha256_bytes(composite.tobytes(order="C")),
            "fit": fit.to_json(),
            "files": files,
        }

    consensus = consensus_circle(fits)
    report = {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "status": "provisional_visual_review_required",
        "created_at_utc": _utc_now(),
        "fit_frozen_before_acquisition_reveal": True,
        "fit_method": FIT_METHOD,
        "target_feature": TARGET_FEATURE,
        "target_plane": TARGET_PLANE,
        "source": {
            "video_path": str(video_path),
            "video_size_bytes": video_path.stat().st_size,
            "video_mtime_ns": video_path.stat().st_mtime_ns,
            "summary_path": str(summary_path),
            "summary_sha256": _sha256_file(summary_path),
            "camera_serial": camera_serial,
            "frame_count": frame_count,
            "fps": fps,
            "image_shape_px": {"height": height, "width": width},
            "pixel_contract": "orange.camera.mono8.full_frame.v1",
        },
        "parameters": {
            "sample_count_per_window": args.sample_count,
            "span_seconds_per_window": args.span_seconds,
            "window_fractions": list(WINDOW_FRACTIONS),
            "coarse_max_dimension_px": args.coarse_max_dimension_px,
            "acquisition_geometry_available_to_fitter": False,
        },
        "decode": decode,
        "windows": windows,
        "consensus_fit": consensus.to_json(),
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "lsf_job_id": os.environ.get("LSB_JOBID"),
        },
        "prohibitions": [
            "not_a_mask_selection",
            "not_a_detection_gate",
            "not_a_zarr_or_registry_publication",
            "do_not_advance_without_visual_review",
        ],
    }
    fit_report_path = output_dir / "fit_report.json"
    _atomic_json(fit_report_path, report)

    if args.acquisition_observation:
        render_acquisition_reveal(
            output_dir=output_dir,
            observation_path=Path(args.acquisition_observation).expanduser().resolve(),
            fit_report_path=fit_report_path,
            composites=composites,
        )
    return fit_report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate blind early/middle/late dish-rim fit diagnostics."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--acquisition-observation",
        type=Path,
        help="Optional reveal-only observation; opened after fit_report.json is frozen.",
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=61)
    parser.add_argument("--span-seconds", type=float, default=5.0)
    parser.add_argument("--coarse-max-dimension-px", type=int, default=2048)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_probe(args)
    print(json.dumps({"status": "complete", "fit_report": str(report)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
