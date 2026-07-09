from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from _common import ensure_output_dir, load_config, resolve_roi_rect, roi_rect_corners
from map_pixel_band_contributions import _bandpass_matrix, _draw_polygon, _load_roi_pixel_traces
from visualize_roi_intensity_diagnostics import (
    _interpolate_short_gaps_all_segments,
    _mask_from_npz,
    _zscore,
)


def _read_frame(capture: Any, *, frame_index: int, next_expected: int | None) -> tuple[bool, np.ndarray | None, int]:
    if next_expected is None or int(frame_index) != int(next_expected):
        capture.set(1, int(frame_index))
    ok, frame = capture.read()
    return bool(ok), frame, int(frame_index) + 1


def _clip_bounds_from_roi(
    roi_polygon: np.ndarray,
    *,
    shape_hw: tuple[int, int],
    pad: int,
) -> tuple[int, int, int, int]:
    points = np.asarray(roi_polygon, dtype=np.float64)
    x0 = max(0, int(math.floor(float(np.nanmin(points[:, 0])))) - int(pad))
    x1 = min(shape_hw[1], int(math.ceil(float(np.nanmax(points[:, 0])))) + int(pad) + 1)
    y0 = max(0, int(math.floor(float(np.nanmin(points[:, 1])))) - int(pad))
    y1 = min(shape_hw[0], int(math.ceil(float(np.nanmax(points[:, 1])))) + int(pad) + 1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"ROI crop is empty: {(x0, y0, x1, y1)}")
    return x0, y0, x1, y1


def _heat_colors(values: np.ndarray, *, clip_value: float) -> np.ndarray:
    clipped = np.clip(values.astype(np.float64) / max(float(clip_value), np.finfo(float).eps), -1.0, 1.0)
    colors = np.full((values.size, 3), 255, dtype=np.float64)
    positive = clipped > 0
    negative = clipped < 0
    colors[positive, 0] = 255.0 * (1.0 - clipped[positive])
    colors[positive, 1] = 255.0 * (1.0 - clipped[positive])
    colors[positive, 2] = 255.0
    magnitude = -clipped[negative]
    colors[negative, 0] = 255.0
    colors[negative, 1] = 255.0 * (1.0 - magnitude)
    colors[negative, 2] = 255.0 * (1.0 - magnitude)
    return np.clip(colors, 0, 255).astype(np.uint8)


def _overlay_darkening(
    frame: np.ndarray,
    *,
    pixel_x: np.ndarray,
    pixel_y: np.ndarray,
    values: np.ndarray,
    clip_value: float,
    alpha: float,
) -> np.ndarray:
    out = frame.copy()
    valid = np.isfinite(values)
    if int(np.count_nonzero(valid)) == 0:
        return out
    x = pixel_x[valid]
    y = pixel_y[valid]
    colors = _heat_colors(values[valid], clip_value=clip_value)
    base = out[y, x, :3].astype(np.float64)
    heat = colors.astype(np.float64)
    strength = np.clip(np.abs(values[valid]) / max(float(clip_value), np.finfo(float).eps), 0.15, 1.0)
    local_alpha = float(alpha) * strength[:, None]
    out[y, x, :3] = np.clip(base * (1.0 - local_alpha) + heat * local_alpha, 0, 255).astype(np.uint8)
    return out


def _resize_to_square(image: np.ndarray, *, size: int) -> np.ndarray:
    import cv2

    return cv2.resize(image[:, :, :3], (int(size), int(size)), interpolation=cv2.INTER_NEAREST)


def _draw_text(
    image: np.ndarray,
    text: str,
    *,
    origin: tuple[int, int],
    scale: float = 0.45,
    color: tuple[int, int, int] = (245, 245, 245),
    thickness: int = 1,
) -> None:
    import cv2

    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        float(scale),
        color,
        int(thickness),
        cv2.LINE_AA,
    )


def _trace_panel(
    *,
    width: int,
    height: int,
    frame_index: np.ndarray,
    current_row: int,
    fps: float,
    raw_mean: np.ndarray,
    roi_darkening: np.ndarray,
    trace_seconds: float,
    band_label: str,
) -> np.ndarray:
    import cv2

    panel = np.full((int(height), int(width), 3), 245, dtype=np.uint8)
    left = 56
    right = int(width) - 18
    top = 22
    mid = int(height) // 2
    bottom = int(height) - 26
    cv2.rectangle(panel, (left, top), (right, bottom), (210, 210, 210), 1)

    current_frame = int(frame_index[current_row])
    half_frames = int(round(float(trace_seconds) * float(fps) / 2.0))
    lo_frame = current_frame - half_frames
    hi_frame = current_frame + half_frames
    selected = np.flatnonzero((frame_index >= lo_frame) & (frame_index <= hi_frame))
    if selected.size < 2:
        return panel

    x_values = frame_index[selected].astype(np.float64)
    x_min = float(x_values[0])
    x_max = float(x_values[-1])
    x_span = max(1.0, x_max - x_min)
    xs = left + np.round((x_values - x_min) / x_span * float(right - left)).astype(np.int32)

    dark = roi_darkening[selected].astype(np.float64)
    finite_dark = dark[np.isfinite(dark)]
    if finite_dark.size:
        ylim = max(1.0, float(np.nanpercentile(np.abs(finite_dark), 98)))
        ys = mid - np.round(np.clip(dark / ylim, -1.0, 1.0) * float(mid - top - 8)).astype(np.int32)
        for a, b, y0, y1 in zip(xs[:-1], xs[1:], ys[:-1], ys[1:]):
            if np.isfinite([y0, y1]).all():
                cv2.line(panel, (int(a), int(y0)), (int(b), int(y1)), (50, 50, 180), 1, cv2.LINE_AA)

    raw = raw_mean[selected].astype(np.float64)
    finite_raw = raw[np.isfinite(raw)]
    if finite_raw.size:
        r_min = float(np.nanpercentile(finite_raw, 2))
        r_max = float(np.nanpercentile(finite_raw, 98))
        if r_max <= r_min:
            r_max = r_min + 1.0
        ys = bottom - np.round(np.clip((raw - r_min) / (r_max - r_min), 0.0, 1.0) * float(bottom - mid - 12)).astype(np.int32)
        for a, b, y0, y1 in zip(xs[:-1], xs[1:], ys[:-1], ys[1:]):
            if np.isfinite([y0, y1]).all():
                cv2.line(panel, (int(a), int(y0)), (int(b), int(y1)), (30, 90, 130), 1, cv2.LINE_AA)

    current_x = left + int(round((float(current_frame) - x_min) / x_span * float(right - left)))
    cv2.line(panel, (current_x, top), (current_x, bottom), (20, 20, 20), 1, cv2.LINE_AA)
    cv2.line(panel, (left, mid), (right, mid), (230, 230, 230), 1, cv2.LINE_AA)
    _draw_text(panel, f"red trace: heartbeat-band darkening ({band_label})", origin=(left, 15), color=(45, 45, 45))
    _draw_text(panel, "brown trace: raw masked-ROI mean", origin=(left, height - 8), color=(70, 70, 70))
    return panel


def _prepare_signal(
    *,
    traces: np.ndarray,
    valid: np.ndarray,
    max_gap: int,
    fps: float,
    band_min_hz: float,
    band_max_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, list[tuple[int, int]]]:
    interpolated, finite_rows, interpolated_rows, segments = _interpolate_short_gaps_all_segments(
        traces=traces,
        valid=valid,
        max_gap=max_gap,
    )
    bandpassed = np.full(interpolated.shape, math.nan, dtype=np.float64)
    for start, stop in segments:
        bandpassed[start:stop] = _bandpass_matrix(
            interpolated[start:stop],
            fps=float(fps),
            band_min_hz=float(band_min_hz),
            band_max_hz=float(band_max_hz),
        )
    analysis_rows = np.isfinite(bandpassed).all(axis=1)
    pixel_z = np.full_like(bandpassed, math.nan, dtype=np.float64)
    pixel_z[analysis_rows] = _zscore(bandpassed[analysis_rows], axis=0)
    roi_signal = np.full(bandpassed.shape[0], math.nan, dtype=np.float64)
    roi_signal[analysis_rows] = np.mean(bandpassed[analysis_rows], axis=1)
    roi_darkening = np.full_like(roi_signal, math.nan)
    roi_darkening[analysis_rows] = -_zscore(roi_signal[analysis_rows])
    return pixel_z, roi_darkening, finite_rows, int(interpolated_rows), segments


def main() -> None:
    import cv2

    parser = argparse.ArgumentParser(description="Render a video of heartbeat-band ROI intensity fluctuations.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.example.toml"))
    parser.add_argument("--video", type=Path, required=True, help="Stabilized video to sample and render.")
    parser.add_argument("--roi-json", type=Path, default=None, help="ROI JSON written by draw_roi.py.")
    parser.add_argument("--roi", type=str, default=None, help="Stabilized ROI rectangle x,y,width,height.")
    parser.add_argument("--status-csv", type=Path, default=None, help="Optional stabilized-video status CSV.")
    parser.add_argument("--mask-npz", type=Path, default=None, help="Optional pixel-band NPZ whose roi_mask defines pixels to sample.")
    parser.add_argument("--frame-start", type=int, default=30000)
    parser.add_argument("--frame-count", type=int, default=3000)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=float, default=100.0)
    parser.add_argument("--playback-fps", type=float, default=30.0)
    parser.add_argument("--band-min-hz", type=float, default=1.5)
    parser.add_argument("--band-max-hz", type=float, default=3.0)
    parser.add_argument(
        "--min-roi-mean-intensity",
        type=float,
        default=1.0,
        help="Reject all-black acquisition-dropout frames when sampled ROI mean is at or below this threshold.",
    )
    parser.add_argument("--max-interpolated-gap-samples", type=int, default=5)
    parser.add_argument("--overlay-clip-z", type=float, default=2.5)
    parser.add_argument("--overlay-alpha", type=float, default=0.95)
    parser.add_argument("--trace-seconds", type=float, default=8.0)
    parser.add_argument("--panel-size", type=int, default=512)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("playgrounds/heartrate_stabilization/outputs/roi_fluctuation_video.mp4"),
    )
    parser.add_argument("--summary-json", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    roi_rect = resolve_roi_rect(config, roi=args.roi, roi_json=args.roi_json)
    roi_polygon = roi_rect_corners(roi_rect)

    sample_mask = None
    if args.mask_npz is not None:
        from map_pixel_band_contributions import _video_shape

        sample_mask = _mask_from_npz(args.mask_npz, shape_hw=_video_shape(args.video))

    loaded = _load_roi_pixel_traces(
        video_path=args.video,
        roi_polygon=roi_polygon,
        status_csv=args.status_csv,
        frame_start=max(0, int(args.frame_start)),
        frame_count=max(0, int(args.frame_count)),
        stride=max(1, int(args.stride)),
        sample_mask=sample_mask,
        min_roi_mean_intensity=args.min_roi_mean_intensity,
    )
    effective_fps = float(args.fps) / max(1, int(args.stride))
    pixel_z, roi_darkening, finite_rows, interpolated_rows, segments = _prepare_signal(
        traces=loaded["traces"],
        valid=loaded["valid"],
        max_gap=max(0, int(args.max_interpolated_gap_samples)),
        fps=effective_fps,
        band_min_hz=float(args.band_min_hz),
        band_max_hz=float(args.band_max_hz),
    )
    raw_mean = np.full(loaded["traces"].shape[0], math.nan, dtype=np.float64)
    raw_rows = loaded["valid"] & np.isfinite(loaded["traces"]).all(axis=1)
    raw_mean[raw_rows] = np.mean(loaded["traces"][raw_rows], axis=1)

    output = args.output
    ensure_output_dir(output.parent)
    panel_size = int(args.panel_size)
    trace_height = 180
    output_width = panel_size * 2
    output_height = panel_size + trace_height
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.playback_fps),
        (output_width, output_height),
    )
    if not writer.isOpened():
        raise ValueError(f"Could not open output writer: {output}")

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        writer.release()
        raise ValueError(f"Could not open video: {args.video}")

    frame_index = loaded["frame_indices"]
    pixel_x = loaded["roi_x"]
    pixel_y = loaded["roi_y"]
    x0, y0, x1, y1 = _clip_bounds_from_roi(roi_polygon, shape_hw=loaded["roi_mask"].shape, pad=8)
    band_label = f"{float(args.band_min_hz):g}-{float(args.band_max_hz):g} Hz"

    rendered = 0
    invalid_rendered = 0
    next_expected: int | None = None
    try:
        for row, frame_number in enumerate(frame_index.tolist()):
            ok, frame, next_expected = _read_frame(capture, frame_index=int(frame_number), next_expected=next_expected)
            if not ok or frame is None:
                invalid_rendered += 1
                frame = np.zeros((loaded["height"], loaded["width"], 3), dtype=np.uint8)
            values = np.full(pixel_x.shape[0], math.nan, dtype=np.float64)
            if row < pixel_z.shape[0] and np.isfinite(pixel_z[row]).any():
                values = -pixel_z[row]
            overlay = _overlay_darkening(
                frame[:, :, :3],
                pixel_x=pixel_x,
                pixel_y=pixel_y,
                values=values,
                clip_value=float(args.overlay_clip_z),
                alpha=float(args.overlay_alpha),
            )
            overlay = _draw_polygon(overlay, roi_polygon, color=(0, 255, 255))
            full_panel = _resize_to_square(overlay, size=panel_size)

            zoom = overlay[y0:y1, x0:x1].copy()
            zoom = _draw_polygon(zoom, roi_polygon - np.asarray([x0, y0], dtype=np.float64), color=(0, 255, 255))
            zoom_panel = _resize_to_square(zoom, size=panel_size)

            valid_label = "valid" if bool(loaded["valid"][row]) and np.isfinite(values).any() else "invalid/gap"
            time_s = (int(frame_number) - int(frame_index[0])) / effective_fps
            _draw_text(
                full_panel,
                f"frame {int(frame_number)}  +{time_s:.2f}s  {valid_label}",
                origin=(12, 24),
                scale=0.55,
                color=(255, 255, 255),
                thickness=1,
            )
            _draw_text(
                full_panel,
                "red=darker, blue=brighter after temporal band-pass",
                origin=(12, panel_size - 16),
                scale=0.5,
                color=(255, 255, 255),
            )
            _draw_text(
                zoom_panel,
                f"ROI zoom, {band_label}",
                origin=(12, 24),
                scale=0.55,
                color=(255, 255, 255),
            )
            if np.isfinite(roi_darkening[row]):
                _draw_text(
                    zoom_panel,
                    f"ROI darkening z={roi_darkening[row]:+.2f}",
                    origin=(12, panel_size - 16),
                    scale=0.55,
                    color=(255, 255, 255),
                )

            trace = _trace_panel(
                width=output_width,
                height=trace_height,
                frame_index=frame_index,
                current_row=row,
                fps=effective_fps,
                raw_mean=raw_mean,
                roi_darkening=roi_darkening,
                trace_seconds=float(args.trace_seconds),
                band_label=band_label,
            )
            top = np.concatenate([full_panel, zoom_panel], axis=1)
            writer.write(np.concatenate([top, trace], axis=0))
            rendered += 1
    finally:
        capture.release()
        writer.release()

    summary = {
        "source_video": str(args.video),
        "status_csv": str(args.status_csv) if args.status_csv is not None else None,
        "roi_json": str(args.roi_json) if args.roi_json is not None else None,
        "mask_npz": str(args.mask_npz) if args.mask_npz is not None else None,
        "output_video": str(output),
        "frame_start": int(args.frame_start),
        "frame_count": int(args.frame_count),
        "stride": int(args.stride),
        "fps": effective_fps,
        "playback_fps": float(args.playback_fps),
        "band_hz": [float(args.band_min_hz), float(args.band_max_hz)],
        "filter": "third-order Butterworth zero-phase temporal band-pass via scipy.signal.sosfiltfilt",
        "display": "red means darker-than-baseline heartbeat-band fluctuation; blue means brighter",
        "rendered_frames": int(rendered),
        "invalid_rendered_frames": int(invalid_rendered),
        "loaded_frames": int(loaded["traces"].shape[0]),
        "valid_loaded_frames": int(np.count_nonzero(loaded["valid"])),
        "low_intensity_frame_count": int(loaded["low_intensity_frame_count"]),
        "min_roi_mean_intensity": loaded["min_roi_mean_intensity"],
        "roi_pixel_count": int(pixel_x.size),
        "interpolated_rows": int(interpolated_rows),
        "valid_segments": [
            {
                "start_loaded_row": int(start),
                "stop_loaded_row": int(stop),
                "start_frame_index": int(frame_index[start]),
                "stop_frame_index_inclusive": int(frame_index[stop - 1]),
                "length": int(stop - start),
            }
            for start, stop in segments
        ],
    }
    summary_path = args.summary_json or output.with_suffix(".summary.json")
    ensure_output_dir(summary_path.parent)
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"output_video: {output}")
    print(f"summary_json: {summary_path}")
    print(f"rendered_frames: {rendered}")
    print(f"roi_pixel_count: {pixel_x.size}")
    print(f"valid_loaded_frames: {summary['valid_loaded_frames']}")
    print(f"low_intensity_frame_count: {summary['low_intensity_frame_count']}")


if __name__ == "__main__":
    main()
