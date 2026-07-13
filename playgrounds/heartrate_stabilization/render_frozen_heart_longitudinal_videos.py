from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any
import warnings

import numpy as np

from analyze_frozen_heart_masks_longitudinal import _read_mask
from diagnose_frozen_mask_longitudinal_tracking import _mask_at_pixels
from extract_reliable_local_rostral_heartrate import load_dataset
from render_dynamic_heart_phase import (
    _blend_values,
    _draw_contour,
    _draw_text,
    _pixel_geometry,
    _resize_nearest,
    _scatter,
    _support_raster,
)


def _frequency_bgr(frequency_hz: float, *, low: float = 2.0, high: float = 4.0) -> tuple[int, int, int]:
    from matplotlib import colormaps

    normalized = float(np.clip((float(frequency_hz) - low) / (high - low), 0.0, 1.0))
    rgb = np.asarray(colormaps["viridis"](normalized)[:3]) * 255.0
    return tuple(int(round(value)) for value in rgb[::-1])


def _read_windows(path: Path, mask_name: str) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["mask"] == mask_name]
    output: list[dict[str, Any]] = []
    for row in rows:
        output.append(
            {
                "window_index": int(row["window_index"]),
                "frame_start": int(row["window_frame_start"]),
                "frame_stop": int(row["window_frame_stop_inclusive"]),
                "mid_s": float(row["window_mid_s"]),
                "valid_fraction": float(row["valid_frame_fraction"]),
                "status": str(row["status"]),
                "rate": float(row["candidate_cycles_per_min"] or "nan"),
                "frequency": float(row["candidate_frequency_hz"] or "nan"),
                "latent": float(row["latent_score"] or "nan"),
            }
        )
    return output


def _timeline_base(
    windows: list[dict[str, Any]],
    *,
    width: int,
    height: int,
    total_s: float,
) -> np.ndarray:
    import cv2

    panel = np.full((height, width, 3), 250, dtype=np.uint8)
    left, right, top, bottom = 54, 14, 24, 24
    plot_w = width - left - right
    plot_h = height - top - bottom
    cv2.rectangle(panel, (left, top), (left + plot_w, top + plot_h), (175, 175, 175), 1)
    for rate in (120, 180, 240):
        y = top + int(round((240.0 - rate) / 120.0 * plot_h))
        cv2.line(panel, (left, y), (left + plot_w, y), (220, 220, 220), 1)
        _draw_text(panel, str(rate), (5, y + 4), scale=0.34, color=(80, 80, 80))
    points: list[tuple[int, int, float] | None] = []
    for window in windows:
        if window["status"] != "ok" or not np.isfinite(window["rate"]):
            points.append(None)
            continue
        x = left + int(round(float(window["mid_s"]) / max(total_s, 1e-9) * plot_w))
        y = top + int(round((240.0 - float(window["rate"])) / 120.0 * plot_h))
        points.append((x, int(np.clip(y, top, top + plot_h)), float(window["frequency"])))
    previous: tuple[int, int, float] | None = None
    for point in points:
        if point is None:
            previous = None
            continue
        xy = (point[0], point[1])
        color = _frequency_bgr(point[2])
        if previous is not None:
            cv2.line(panel, (previous[0], previous[1]), xy, color, 2, cv2.LINE_AA)
        cv2.circle(panel, xy, 2, color, -1, cv2.LINE_AA)
        previous = point
    _draw_text(panel, "compact-mask candidate cycles/min", (left, 16), scale=0.40)
    _draw_text(panel, "0", (left - 2, height - 5), scale=0.32, color=(80, 80, 80))
    _draw_text(
        panel,
        f"{total_s / 60.0:.1f} min",
        (width - 70, height - 5),
        scale=0.32,
        color=(80, 80, 80),
    )
    return panel


def _window_for_frame(
    frame_index: int,
    windows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    for window in windows:
        if int(window["frame_start"]) <= frame_index <= int(window["frame_stop"]):
            return window
    return None


def _blend_mask(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    color: tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    output = np.asarray(image, dtype=np.uint8).copy()
    selected = np.asarray(mask, dtype=bool)
    if np.any(selected):
        target = np.asarray(color, dtype=np.float64)
        output[selected] = np.clip(
            output[selected].astype(np.float64) * (1.0 - float(alpha))
            + target * float(alpha),
            0,
            255,
        ).astype(np.uint8)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render frame-locked clean and longitudinal-data frozen-heart videos."
    )
    parser.add_argument("--dataset-npz", type=Path, required=True)
    parser.add_argument("--longitudinal-csv", type=Path, required=True)
    parser.add_argument("--phase-npz", type=Path, required=True)
    parser.add_argument("--original-mask-npz", type=Path, required=True)
    parser.add_argument("--original-mask-key", default="heart_support_mask")
    parser.add_argument("--consensus-mask-npz", type=Path, required=True)
    parser.add_argument("--consensus-mask-key", default="consensus_mask")
    parser.add_argument("--frame-stride", type=int, default=3)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-count", type=int)
    parser.add_argument("--playback-fps", type=float, default=30.0)
    parser.add_argument("--panel-size", type=int, default=640)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    if int(args.frame_stride) < 1 or float(args.playback_fps) <= 0.0:
        raise ValueError("frame stride and playback fps must be positive")
    if int(args.panel_size) < 320:
        raise ValueError("panel size must be at least 320")
    dataset = load_dataset(args.dataset_npz)
    original = _read_mask(args.original_mask_npz, args.original_mask_key)
    consensus = _read_mask(args.consensus_mask_npz, args.consensus_mask_key)
    intersection = original & consensus
    windows = _read_windows(args.longitudinal_csv, "intersection_8")
    geometry = _pixel_geometry(dataset, margin_px=1)
    original_raster = _support_raster(_mask_at_pixels(original, dataset.pixel_xy), geometry)
    consensus_raster = _support_raster(_mask_at_pixels(consensus, dataset.pixel_xy), geometry)
    intersection_raster = _support_raster(
        _mask_at_pixels(intersection, dataset.pixel_xy), geometry
    )
    with np.load(args.phase_npz, allow_pickle=False) as phase_data:
        phase_frame_indices = np.asarray(phase_data["frame_indices"], dtype=np.int64)
        phase_pixel_indices = np.asarray(phase_data["pixel_indices"], dtype=np.int64)
        phase_rad = np.asarray(phase_data["phase_rad"], dtype=np.float32)
        phase_alpha = np.asarray(phase_data["phase_alpha"], dtype=np.float32)
        phase_valid = np.asarray(phase_data["phase_valid"], dtype=bool)
        phase_mask_name = (
            str(np.asarray(phase_data["phase_mask_name"]).item())
            if "phase_mask_name" in phase_data
            else "intersection_8"
        )
        frequency_source_mask = (
            str(np.asarray(phase_data["frequency_source_mask"]).item())
            if "frequency_source_mask" in phase_data
            else "intersection_8"
        )
    if not np.array_equal(phase_frame_indices, np.asarray(dataset.frame_indices, dtype=np.int64)):
        raise ValueError("phase cache frame indices do not match the dataset")
    expected_phase_shape = (dataset.frame_count, phase_pixel_indices.size)
    if phase_rad.shape != expected_phase_shape or phase_alpha.shape != expected_phase_shape:
        raise ValueError("phase cache arrays do not match dataset frame/pixel dimensions")
    if np.any(phase_pixel_indices < 0) or np.any(phase_pixel_indices >= dataset.pixel_count):
        raise ValueError("phase cache pixel indices are outside the dataset")
    phase_support = np.zeros(dataset.pixel_count, dtype=bool)
    phase_support[phase_pixel_indices] = True
    phase_support_raster = _support_raster(phase_support, geometry)
    traces = np.asarray(dataset.traces, dtype=np.float64)
    finite = traces[np.isfinite(traces)]
    low, high = np.quantile(finite, [0.02, 0.98])
    if not high > low:
        high = low + 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median_values = np.nanmedian(traces, axis=0)
    median_values = np.where(np.isfinite(median_values), median_values, 0.5 * (low + high))

    import cv2

    def large_mask(mask: np.ndarray) -> np.ndarray:
        return cv2.resize(
            np.asarray(mask, dtype=np.uint8),
            (int(args.panel_size), int(args.panel_size)),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    original_large = large_mask(original_raster)
    consensus_large = large_mask(consensus_raster)
    intersection_large = large_mask(intersection_raster)

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    clean_path = output_prefix.with_suffix(".mask_overlay.mp4")
    data_path = output_prefix.with_suffix(".data_overlay.mp4")
    summary_path = output_prefix.with_suffix(".video_overlays.summary.json")
    panel_size = int(args.panel_size)
    clean_header = 54
    data_header = 94
    timeline_h = 136
    clean_size = (panel_size, panel_size + clean_header)
    data_size = (panel_size, panel_size + data_header + timeline_h)
    clean_writer = cv2.VideoWriter(
        str(clean_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.playback_fps),
        clean_size,
    )
    data_writer = cv2.VideoWriter(
        str(data_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.playback_fps),
        data_size,
    )
    if not clean_writer.isOpened() or not data_writer.isOpened():
        raise RuntimeError("could not open longitudinal video writers")
    total_s = float(dataset.timestamps_s[-1] - dataset.timestamps_s[0])
    timeline = _timeline_base(windows, width=panel_size, height=timeline_h, total_s=total_s)
    frequency_gradient = np.linspace(2.0, 4.0, panel_size - 130, dtype=np.float64)
    frequency_gradient_bar = np.repeat(
        np.asarray([_frequency_bgr(value) for value in frequency_gradient], dtype=np.uint8)[
            None, :, :
        ],
        9,
        axis=0,
    )
    rendered = 0
    frame_start = int(args.frame_start)
    frame_stop = (
        dataset.frame_count
        if args.frame_count is None
        else min(dataset.frame_count, frame_start + int(args.frame_count))
    )
    if not (0 <= frame_start < frame_stop <= dataset.frame_count):
        raise ValueError("invalid render frame range")
    for row_index in range(frame_start, frame_stop, int(args.frame_stride)):
        values = traces[row_index].copy()
        missing = ~np.isfinite(values)
        values[missing] = median_values[missing]
        raw = _scatter(values, geometry)
        gray = np.clip((raw - low) / (high - low), 0.0, 1.0)
        base = np.asarray(np.nan_to_num(gray, nan=0.5) * 255.0, dtype=np.uint8)
        base_bgr = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

        clean_roi = base_bgr.copy()
        clean_roi = _resize_nearest(clean_roi, panel_size)
        _draw_contour(clean_roi, original_large, color=(20, 185, 230), thickness=2)
        _draw_contour(clean_roi, consensus_large, color=(220, 120, 20), thickness=2)
        _draw_contour(clean_roi, intersection_large, color=(40, 185, 60), thickness=2)
        clean_canvas = np.full((clean_size[1], clean_size[0], 3), 255, dtype=np.uint8)
        clean_canvas[clean_header:, :] = clean_roi
        frame_index = int(dataset.frame_indices[row_index])
        relative_s = float(dataset.timestamps_s[row_index] - dataset.timestamps_s[0])
        _draw_text(clean_canvas, "Frozen anatomical masks on local stabilized ROI", (12, 22), scale=0.52)
        _draw_text(
            clean_canvas,
            "gold=original 38 | blue=consensus 9 | green=intersection 8",
            (12, 44),
            scale=0.38,
            color=(70, 70, 70),
        )
        _draw_text(
            clean_canvas,
            f"frame {frame_index} | t={relative_s / 60.0:.2f} min",
            (12, clean_header + panel_size - 14),
            scale=0.48,
            color=(245, 245, 245),
        )
        clean_writer.write(clean_canvas)

        window = _window_for_frame(frame_index, windows)
        scorable = bool(window is not None and window["status"] == "ok")
        data_roi = _blend_mask(
            base_bgr,
            phase_support_raster,
            color=(150, 150, 150),
            alpha=0.28 if scorable else 0.20,
        )
        pixel_phase = np.full(dataset.pixel_count, np.nan, dtype=np.float64)
        pixel_alpha = np.zeros(dataset.pixel_count, dtype=np.float64)
        pixel_phase[phase_pixel_indices] = phase_rad[row_index]
        pixel_alpha[phase_pixel_indices] = phase_alpha[row_index]
        data_roi = _blend_values(
            data_roi,
            _scatter(pixel_phase, geometry),
            _scatter(pixel_alpha, geometry, fill=0.0),
            cmap_name="twilight_shifted",
            value_min=-np.pi,
            value_max=np.pi,
        )
        outline_color = (
            _frequency_bgr(float(window["frequency"])) if scorable else (130, 130, 130)
        )
        data_roi = _resize_nearest(data_roi, panel_size)
        _draw_contour(data_roi, original_large, color=(20, 185, 230), thickness=2)
        _draw_contour(data_roi, consensus_large, color=(220, 120, 20), thickness=2)
        _draw_contour(data_roi, intersection_large, color=outline_color, thickness=3)
        data_canvas = np.full((data_size[1], data_size[0], 3), 255, dtype=np.uint8)
        data_canvas[data_header : data_header + panel_size, :] = data_roi
        if scorable:
            phase_display = phase_mask_name.replace("_", "")
            frequency_display = (
                "core" if frequency_source_mask == "intersection_8" else frequency_source_mask
            )
            headline = (
                f"candidate {window['rate']:.0f} cycles/min | {window['frequency']:.2f} Hz | "
                f"latent {window['latent']:.2f}"
            )
            phase_status = "phase shown" if bool(phase_valid[row_index]) else "no phase at this frame"
            detail = (
                f"phase {phase_display} @ {frequency_display} frequency | "
                f"valid {window['valid_fraction']:.1%} | {phase_status}"
            )
        else:
            headline = "no reliable window estimate"
            detail = "insufficient valid local tracking; no interpolation"
        _draw_text(data_canvas, headline, (12, 28), scale=0.54)
        _draw_text(data_canvas, detail, (12, 56), scale=0.42, color=(70, 70, 70))
        data_canvas[68:77, 64 : panel_size - 66] = frequency_gradient_bar
        _draw_text(data_canvas, "2 Hz", (12, 77), scale=0.30, color=(70, 70, 70))
        _draw_text(data_canvas, "4 Hz", (panel_size - 58, 77), scale=0.30, color=(70, 70, 70))
        _draw_text(
            data_canvas,
            f"frame {frame_index} | t={relative_s / 60.0:.2f} min",
            (12, data_header + panel_size - 14),
            scale=0.48,
            color=(245, 245, 245),
        )
        timeline_frame = timeline.copy()
        marker_x = 54 + int(round(relative_s / max(total_s, 1e-9) * (panel_size - 54 - 14)))
        cv2.line(timeline_frame, (marker_x, 24), (marker_x, timeline_h - 24), (30, 30, 220), 2)
        data_canvas[data_header + panel_size :, :] = timeline_frame
        data_writer.write(data_canvas)
        rendered += 1
        if rendered % 2500 == 0:
            print(f"rendered_frames: {rendered}", flush=True)
    clean_writer.release()
    data_writer.release()
    timestamp_diffs = np.diff(np.asarray(dataset.timestamps_s, dtype=np.float64))
    source_fps = 1.0 / float(np.median(timestamp_diffs))
    summary = {
        "dataset_npz": str(args.dataset_npz),
        "longitudinal_csv": str(args.longitudinal_csv),
        "phase_npz": str(args.phase_npz),
        "source_frame_count": dataset.frame_count,
        "render_frame_start": frame_start,
        "render_frame_stop_exclusive": frame_stop,
        "rendered_frame_count": rendered,
        "frame_stride": int(args.frame_stride),
        "playback_fps": float(args.playback_fps),
        "source_fps": source_fps,
        "playback_speed_ratio": float(args.frame_stride) * float(args.playback_fps) / source_fps,
        "source_duration_s": total_s,
        "rendered_duration_s": float(rendered) / float(args.playback_fps),
        "unscorable_policy": "show no estimate and preserve a timeline gap",
        "phase_mask": phase_mask_name,
        "frequency_source_mask": frequency_source_mask,
        "phase_encoding": f"{phase_mask_name}_fill_uses_twilight_shifted_cyclic_hue",
        "phase_opacity": "crossfit_loading_and_relative_amplitude",
        "frequency_encoding": "intersection_outline_and_timeline_use_fixed_viridis_2_to_4_hz",
        "caveat": "Candidate cycles per minute are descriptive frozen-mask oscillation estimates, not validated heart rate.",
        "outputs": {"mask_overlay_mp4": str(clean_path), "data_overlay_mp4": str(data_path)},
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"mask_overlay_mp4: {clean_path}")
    print(f"data_overlay_mp4: {data_path}")
    print(f"summary_json: {summary_path}")


if __name__ == "__main__":
    main()
