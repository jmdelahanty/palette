from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from _common import ensure_output_dir, load_config, resolve_roi_rect, roi_rect_corners
from localized_periodic_signal_probe import (
    _extract_events,
    _json_ready,
    _load_roi_stack,
    _nuisance_regressors,
    _prepare_stack,
    _residualize,
    _video_shape,
)
from map_pixel_band_contributions import _draw_polygon
from visualize_roi_intensity_diagnostics import _mask_from_npz


def _load_selected_mask(path: Path) -> dict[str, Any]:
    with np.load(path) as data:
        selected = np.asarray(data["selected_mask"], dtype=bool)
        candidate = np.asarray(data["candidate_mask"], dtype=bool)
        bbox = tuple(int(value) for value in np.asarray(data["bbox_xyxy"]).reshape(-1).tolist())
    if selected.shape != candidate.shape:
        raise ValueError(f"{path} selected_mask shape {selected.shape} differs from candidate_mask {candidate.shape}")
    if len(bbox) != 4:
        raise ValueError(f"{path} bbox_xyxy must have four values.")
    return {"selected_mask": selected, "candidate_mask": candidate, "bbox_xyxy": bbox}


def _read_frame(capture: Any, *, frame_index: int, next_expected: int | None) -> tuple[bool, np.ndarray | None, int]:
    import cv2

    if next_expected is None or int(frame_index) != int(next_expected):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = capture.read()
    return bool(ok), frame, int(frame_index) + 1


def _event_rate_trace(
    *,
    event_peaks: np.ndarray,
    frame_count: int,
    fps: float,
    window_seconds: float,
) -> np.ndarray:
    event_times = np.asarray(event_peaks, dtype=np.float64) / float(fps)
    frame_times = np.arange(int(frame_count), dtype=np.float64) / float(fps)
    half_window = float(window_seconds) / 2.0
    rates = np.full(frame_times.shape, math.nan, dtype=np.float64)
    for idx, time_s in enumerate(frame_times.tolist()):
        inside = event_times[(event_times >= time_s - half_window) & (event_times <= time_s + half_window)]
        if inside.size >= 3:
            intervals = np.diff(inside)
            intervals = intervals[intervals > np.finfo(float).eps]
            if intervals.size:
                rates[idx] = 60.0 / float(np.median(intervals))
    return rates


def _rate_color(rate: float, *, rate_min: float, rate_max: float) -> tuple[int, int, int]:
    import cv2

    if not np.isfinite(rate):
        return (170, 170, 170)
    scaled = np.clip((float(rate) - float(rate_min)) / max(float(rate_max) - float(rate_min), 1e-6), 0.0, 1.0)
    # TURBO is blue/green/yellow/red and gives a readable scalar rate cue.
    color = cv2.applyColorMap(np.asarray([[int(round(scaled * 255.0))]], dtype=np.uint8), cv2.COLORMAP_TURBO)[0, 0]
    return int(color[0]), int(color[1]), int(color[2])


def _draw_text(
    image: np.ndarray,
    text: str,
    *,
    origin: tuple[int, int],
    scale: float = 0.52,
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


def _overlay_mask(
    crop: np.ndarray,
    *,
    candidate_mask: np.ndarray,
    selected_mask: np.ndarray,
    color_bgr: tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    import cv2

    out = crop[:, :, :3].copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    base = out.astype(np.float64)
    color = np.asarray(color_bgr, dtype=np.float64)
    selected = np.asarray(selected_mask, dtype=bool)
    candidate = np.asarray(candidate_mask, dtype=bool)
    base[selected] = np.clip(base[selected] * (1.0 - float(alpha)) + color * float(alpha), 0, 255)
    out = base.astype(np.uint8)
    if int(np.count_nonzero(candidate)):
        cv2.drawContours(
            out,
            cv2.findContours(candidate.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
            -1,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    if int(np.count_nonzero(selected)):
        cv2.drawContours(
            out,
            cv2.findContours(selected.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
            -1,
            color_bgr,
            1,
            cv2.LINE_AA,
        )
    return out


def _event_active(*, row: int, event_peaks: np.ndarray, highlight_frames: int) -> bool:
    if event_peaks.size == 0:
        return False
    nearest = int(np.min(np.abs(event_peaks.astype(np.int64) - int(row))))
    return nearest <= int(highlight_frames)


def _resize(image: np.ndarray, *, width: int, height: int) -> np.ndarray:
    import cv2

    return cv2.resize(image[:, :, :3], (int(width), int(height)), interpolation=cv2.INTER_NEAREST)


def _plot_series(
    panel: np.ndarray,
    *,
    x: np.ndarray,
    y: np.ndarray,
    current_index: int,
    rect: tuple[int, int, int, int],
    color: tuple[int, int, int],
    y_label: str,
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    import cv2

    x0, y0, x1, y1 = rect
    cv2.rectangle(panel, (x0, y0), (x1, y1), (210, 210, 210), 1)
    finite = np.isfinite(y)
    if int(np.count_nonzero(finite)) >= 2:
        yy = y.astype(np.float64)
        if y_min is None:
            y_min = float(np.nanpercentile(yy[finite], 2))
        if y_max is None:
            y_max = float(np.nanpercentile(yy[finite], 98))
        if not np.isfinite(y_min) or not np.isfinite(y_max) or y_max <= y_min:
            y_min = float(np.nanmin(yy[finite])) - 1.0
            y_max = float(np.nanmax(yy[finite])) + 1.0
        x_span = max(float(x[-1] - x[0]), 1.0)
        xs = x0 + np.round((x - float(x[0])) / x_span * float(x1 - x0)).astype(np.int32)
        ys = y1 - np.round(np.clip((yy - y_min) / (y_max - y_min), 0.0, 1.0) * float(y1 - y0)).astype(np.int32)
        for idx in range(xs.size - 1):
            if finite[idx] and finite[idx + 1]:
                cv2.line(panel, (int(xs[idx]), int(ys[idx])), (int(xs[idx + 1]), int(ys[idx + 1])), color, 1, cv2.LINE_AA)
    current_x = x0 + int(round((float(current_index) / max(1, len(x) - 1)) * float(x1 - x0)))
    cv2.line(panel, (current_x, y0), (current_x, y1), (30, 30, 30), 1, cv2.LINE_AA)
    _draw_text(panel, y_label, origin=(x0, max(14, y0 - 6)), scale=0.42, color=(40, 40, 40))


def _trace_panel(
    *,
    width: int,
    height: int,
    current_index: int,
    fps: float,
    filtered: np.ndarray,
    event_rate: np.ndarray,
    event_peaks: np.ndarray,
    rate_min: float,
    rate_max: float,
    trace_seconds: float,
) -> np.ndarray:
    import cv2

    panel = np.full((int(height), int(width), 3), 246, dtype=np.uint8)
    half = max(2, int(round(float(trace_seconds) * float(fps) / 2.0)))
    start = max(0, int(current_index) - half)
    stop = min(filtered.size, int(current_index) + half + 1)
    if stop - start < 3:
        return panel
    x = np.arange(stop - start, dtype=np.float64)
    top_rect = (58, 24, int(width) - 22, int(height) // 2 - 12)
    bottom_rect = (58, int(height) // 2 + 26, int(width) - 22, int(height) - 26)
    local_current = int(current_index) - start
    local_filtered = filtered[start:stop]
    finite = local_filtered[np.isfinite(local_filtered)]
    ylim = max(1.0, float(np.nanpercentile(np.abs(finite), 98))) if finite.size else 1.0
    _plot_series(
        panel,
        x=x,
        y=local_filtered,
        current_index=local_current,
        rect=top_rect,
        color=(55, 55, 160),
        y_label="band-passed fixed-mask trace",
        y_min=-ylim,
        y_max=ylim,
    )
    local_events = event_peaks[(event_peaks >= start) & (event_peaks < stop)] - start
    x0, y0, x1, y1 = top_rect
    for event in local_events.tolist():
        event_x = x0 + int(round((float(event) / max(1, stop - start - 1)) * float(x1 - x0)))
        cv2.drawMarker(panel, (event_x, y0 + 10), (30, 30, 30), markerType=cv2.MARKER_TRIANGLE_DOWN, markerSize=7, thickness=1)
    _plot_series(
        panel,
        x=x,
        y=event_rate[start:stop],
        current_index=local_current,
        rect=bottom_rect,
        color=(30, 120, 70),
        y_label="rolling event rate (/min)",
        y_min=float(rate_min),
        y_max=float(rate_max),
    )
    return panel


def _write_trace_csv(
    path: Path,
    *,
    frame_indices: np.ndarray,
    fps: float,
    filtered: np.ndarray,
    event_rate: np.ndarray,
) -> None:
    ensure_output_dir(path.parent)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame_index", "time_s", "filtered_trace", "rolling_event_rate_per_min"])
        writer.writeheader()
        for idx in range(frame_indices.size):
            writer.writerow(
                {
                    "frame_index": int(frame_indices[idx]),
                    "time_s": float(idx / float(fps)),
                    "filtered_trace": float(filtered[idx]) if np.isfinite(filtered[idx]) else math.nan,
                    "rolling_event_rate_per_min": float(event_rate[idx]) if np.isfinite(event_rate[idx]) else math.nan,
                }
            )


def main() -> None:
    import cv2

    parser = argparse.ArgumentParser(description="Render a fixed-mask localized periodic-signal diagnostic video.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.example.toml"))
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--roi-json", type=Path, default=None)
    parser.add_argument("--roi", type=str, default=None)
    parser.add_argument("--status-csv", type=Path, default=None)
    parser.add_argument("--mask-npz", type=Path, default=None, help="Usable-pixel mask NPZ.")
    parser.add_argument("--selected-mask-npz", type=Path, required=True, help="Fixed selected-mask NPZ from the probe.")
    parser.add_argument("--frame-start", type=int, default=30000)
    parser.add_argument("--frame-count", type=int, default=3000)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--fps", type=float, default=100.0)
    parser.add_argument("--playback-fps", type=float, default=50.0)
    parser.add_argument("--band-min-hz", type=float, default=1.5)
    parser.add_argument("--band-max-hz", type=float, default=3.5)
    parser.add_argument("--pad-px", type=int, default=4)
    parser.add_argument("--min-roi-mean-intensity", type=float, default=1.0)
    parser.add_argument("--duplicate-mean-tol", type=float, default=1e-6)
    parser.add_argument("--max-interpolated-gap-samples", type=int, default=5)
    parser.add_argument("--min-segment-samples", type=int, default=512)
    parser.add_argument("--event-prominence-scale", type=float, default=0.5)
    parser.add_argument("--rate-window-seconds", type=float, default=8.0)
    parser.add_argument("--rate-min", type=float, default=90.0)
    parser.add_argument("--rate-max", type=float, default=210.0)
    parser.add_argument("--event-highlight-seconds", type=float, default=0.08)
    parser.add_argument("--trace-seconds", type=float, default=10.0)
    parser.add_argument("--panel-width", type=int, default=560)
    parser.add_argument("--panel-height", type=int, default=420)
    parser.add_argument("--overlay-alpha", type=float, default=0.85)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("outputs") / "fixed_mask_event_video.mp4",
    )
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--trace-csv", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    roi_rect = resolve_roi_rect(config, roi=args.roi, roi_json=args.roi_json)
    roi_polygon = roi_rect_corners(roi_rect)
    sample_mask = _mask_from_npz(args.mask_npz, shape_hw=_video_shape(args.video)) if args.mask_npz is not None else None
    fixed = _load_selected_mask(args.selected_mask_npz)

    loaded = _load_roi_stack(
        video_path=args.video,
        roi_polygon=roi_polygon,
        status_csv=args.status_csv,
        frame_start=max(0, int(args.frame_start)),
        frame_count=max(0, int(args.frame_count)),
        stride=max(1, int(args.stride)),
        sample_mask=sample_mask,
        min_roi_mean_intensity=float(args.min_roi_mean_intensity),
        pad_px=max(0, int(args.pad_px)),
        duplicate_mean_tol=float(args.duplicate_mean_tol),
    )
    if tuple(loaded["bbox_xyxy"]) != tuple(fixed["bbox_xyxy"]):
        raise ValueError(f"Loaded bbox {loaded['bbox_xyxy']} does not match fixed-mask bbox {fixed['bbox_xyxy']}")
    stack, finite_valid, segment, interpolated = _prepare_stack(
        loaded["stack"],
        loaded["valid"],
        max_interpolated_gap=max(0, int(args.max_interpolated_gap_samples)),
        min_segment_samples=max(32, int(args.min_segment_samples)),
    )
    segment_start, segment_stop = segment
    frame_indices = loaded["frame_indices"][segment_start:segment_stop]
    effective_fps = float(args.fps) / max(1, int(args.stride))
    candidate_mask = np.asarray(loaded["candidate_crop_mask"], dtype=bool)
    selected_mask = np.asarray(fixed["selected_mask"], dtype=bool) & candidate_mask
    if int(np.count_nonzero(selected_mask)) == 0:
        raise ValueError("Fixed selected mask has no pixels inside this clip's candidate mask.")

    mask_flat = candidate_mask.reshape(-1)
    traces = stack.reshape(stack.shape[0], -1)[:, mask_flat].T.astype(np.float64)
    motion_pred, scalar, mean_image = _nuisance_regressors(stack, candidate_mask)
    residual = _residualize(traces, motion_pred, scalar)
    selected_idx = np.flatnonzero(selected_mask.reshape(-1)[mask_flat])
    selected_trace = np.median(residual[selected_idx], axis=0)
    events = _extract_events(
        selected_trace,
        fps=effective_fps,
        band_min_hz=float(args.band_min_hz),
        band_max_hz=float(args.band_max_hz),
        prominence_scale=float(args.event_prominence_scale),
    )
    filtered = np.asarray(events["filtered"], dtype=np.float64)
    event_peaks = np.asarray(events["peaks"], dtype=np.int64)
    event_rate = _event_rate_trace(
        event_peaks=event_peaks,
        frame_count=filtered.size,
        fps=effective_fps,
        window_seconds=float(args.rate_window_seconds),
    )

    output = args.output
    ensure_output_dir(output.parent)
    panel_width = int(args.panel_width)
    panel_height = int(args.panel_height)
    trace_height = 190
    visual_height = panel_height * 2
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.playback_fps),
        (panel_width * 2, visual_height + trace_height),
    )
    if not writer.isOpened():
        raise ValueError(f"Could not open output writer: {output}")

    x0, y0, x1, y1 = tuple(int(value) for value in loaded["bbox_xyxy"])
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        writer.release()
        raise ValueError(f"Could not open video: {args.video}")

    rendered = 0
    next_expected: int | None = None
    highlight_frames = max(1, int(round(float(args.event_highlight_seconds) * effective_fps)))
    try:
        for row, frame_number in enumerate(frame_indices.tolist()):
            ok, frame, next_expected = _read_frame(capture, frame_index=int(frame_number), next_expected=next_expected)
            if not ok or frame is None:
                frame = np.zeros((loaded["video_shape_hw"][0], loaded["video_shape_hw"][1], 3), dtype=np.uint8)
            crop = frame[y0:y1, x0:x1, :3]
            rate = float(event_rate[row]) if row < event_rate.size else math.nan
            color = _rate_color(rate, rate_min=float(args.rate_min), rate_max=float(args.rate_max))
            rate_crop = _overlay_mask(
                crop,
                candidate_mask=candidate_mask,
                selected_mask=selected_mask,
                color_bgr=color,
                alpha=float(args.overlay_alpha),
            )
            rate_panel = _resize(rate_crop, width=panel_width, height=panel_height)
            _draw_text(rate_panel, f"fixed mask by rolling rate  frame {int(frame_number)}", origin=(14, 26))
            _draw_text(
                rate_panel,
                f"rolling rate: {rate:.1f}/min" if np.isfinite(rate) else "rolling rate: n/a",
                origin=(14, panel_height - 18),
                color=(255, 255, 255),
            )

            clean_panel = _resize(crop, width=panel_width, height=panel_height)
            _draw_text(clean_panel, "ROI zoom, no mask overlay", origin=(14, 26), color=(255, 255, 255))

            active = _event_active(row=row, event_peaks=event_peaks, highlight_frames=highlight_frames)
            event_color = (30, 30, 255) if active else (145, 145, 145)
            event_crop = _overlay_mask(
                crop,
                candidate_mask=candidate_mask,
                selected_mask=selected_mask,
                color_bgr=event_color,
                alpha=0.95 if active else 0.65,
            )
            event_panel = _resize(event_crop, width=panel_width, height=panel_height)
            _draw_text(event_panel, f"event state: {'yes' if active else 'no'}", origin=(14, 26), color=(255, 255, 255))
            _draw_text(
                event_panel,
                f"highlight window: +/-{float(args.event_highlight_seconds):.2f}s",
                origin=(14, panel_height - 18),
                color=(255, 255, 255),
            )

            context = frame[:, :, :3].copy()
            context = _draw_polygon(context, roi_polygon, color=(0, 255, 255))
            context[y0:y1, x0:x1] = _overlay_mask(
                context[y0:y1, x0:x1],
                candidate_mask=candidate_mask,
                selected_mask=selected_mask,
                color_bgr=color,
                alpha=float(args.overlay_alpha),
            )
            context_panel = _resize(context, width=panel_width, height=panel_height)
            time_s = float(row) / effective_fps
            _draw_text(context_panel, f"+{time_s:.2f}s  {float(args.band_min_hz):g}-{float(args.band_max_hz):g} Hz", origin=(14, 26))
            _draw_text(context_panel, "color = rolling event rate", origin=(14, panel_height - 18))

            trace = _trace_panel(
                width=panel_width * 2,
                height=trace_height,
                current_index=row,
                fps=effective_fps,
                filtered=filtered,
                event_rate=event_rate,
                event_peaks=event_peaks,
                rate_min=float(args.rate_min),
                rate_max=float(args.rate_max),
                trace_seconds=float(args.trace_seconds),
            )
            top = np.concatenate([context_panel, clean_panel], axis=1)
            middle = np.concatenate([rate_panel, event_panel], axis=1)
            writer.write(np.concatenate([top, middle, trace], axis=0))
            rendered += 1
    finally:
        capture.release()
        writer.release()

    trace_csv = args.trace_csv or output.with_suffix(".trace.csv")
    _write_trace_csv(trace_csv, frame_indices=frame_indices, fps=effective_fps, filtered=filtered, event_rate=event_rate)

    rates = np.asarray(events["event_rate_per_min"], dtype=np.float64)
    finite_rates = rates[np.isfinite(rates)]
    rolling_finite = event_rate[np.isfinite(event_rate)]
    summary = {
        "source_video": str(args.video),
        "selected_mask_npz": str(args.selected_mask_npz),
        "output_video": str(output),
        "trace_csv": str(trace_csv),
        "frame_start": int(args.frame_start),
        "frame_count_requested": int(args.frame_count),
        "stride": int(args.stride),
        "fps": float(effective_fps),
        "playback_fps": float(args.playback_fps),
        "analysis_frame_indices": [int(frame_indices[0]), int(frame_indices[-1])],
        "analysis_frame_count": int(frame_indices.size),
        "band_hz": [float(args.band_min_hz), float(args.band_max_hz)],
        "selected_pixel_count": int(selected_idx.size),
        "event_count": int(event_peaks.size),
        "event_rate_per_min_median": float(np.median(finite_rates)) if finite_rates.size else math.nan,
        "rolling_rate_per_min_median": float(np.median(rolling_finite)) if rolling_finite.size else math.nan,
        "rolling_rate_window_seconds": float(args.rate_window_seconds),
        "event_highlight_seconds": float(args.event_highlight_seconds),
        "rendered_frames": int(rendered),
        "interpolated_short_gap_frame_count": int(interpolated),
    }
    summary_path = args.summary_json or output.with_suffix(".summary.json")
    ensure_output_dir(summary_path.parent)
    with summary_path.open("w") as handle:
        json.dump(_json_ready(summary), handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"output_video: {output}")
    print(f"summary_json: {summary_path}")
    print(f"trace_csv: {trace_csv}")
    print(f"rendered_frames: {rendered}")
    print(f"event_count: {event_peaks.size}")
    if finite_rates.size:
        print(f"event_rate_per_min_median: {np.median(finite_rates):.3f}")


if __name__ == "__main__":
    main()
