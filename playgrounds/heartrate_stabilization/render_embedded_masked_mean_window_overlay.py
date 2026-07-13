from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from analyze_embedded_crossfit_mask import temporal_half_partitions
from analyze_embedded_positive_control import (
    _load_roi,
    _reference_at_times,
    read_numeric_xlsx_row,
)
from evaluate_embedded_rate_window_sweep import evaluate_window
from render_embedded_crossfit_mask_overlay import crossfit_model_indices
from render_embedded_positive_control_overlay import (
    _color_pixel_change,
    _ffmpeg_process,
    _polyline_points,
    _text,
)


_CANVAS_SIZE = (1152, 648)
_SOURCE_ORIGIN = (16, 44)
_RAW_ROI_ORIGIN = (590, 44)
_STATE_ROI_ORIGIN = (830, 44)
_ROI_SCALE = 10
_FOLD_COLORS = ((50, 70, 245), (235, 130, 45))


def smooth_finite_runs(values: np.ndarray, *, sigma_samples: float) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    result = np.full_like(source, np.nan)
    finite = np.flatnonzero(np.isfinite(source))
    if not finite.size:
        return result
    runs = np.split(finite, np.flatnonzero(np.diff(finite) > 1) + 1)
    for run in runs:
        if float(sigma_samples) > 0.0 and run.size >= 2:
            result[run] = gaussian_filter1d(
                source[run], sigma=float(sigma_samples), mode="nearest"
            )
        else:
            result[run] = source[run]
    return result


def nearest_ridge_value(
    ridge_time_s: np.ndarray,
    ridge_hz: np.ndarray,
    *,
    time_s: float,
    maximum_distance_s: float,
) -> float:
    times = np.asarray(ridge_time_s, dtype=np.float64)
    values = np.asarray(ridge_hz, dtype=np.float64)
    if not times.size or times.shape != values.shape:
        return float("nan")
    index = int(np.argmin(np.abs(times - float(time_s))))
    if abs(float(times[index]) - float(time_s)) > float(maximum_distance_s):
        return float("nan")
    return float(values[index])


def _draw_mask_outline(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    origin: tuple[int, int],
    scale: int,
    color: tuple[int, int, int],
) -> None:
    contours, _ = cv2.findContours(
        np.asarray(mask, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    ox, oy = origin
    shifted = [contour * int(scale) + np.asarray([[[ox, oy]]], dtype=np.int32) for contour in contours]
    if shifted:
        cv2.drawContours(image, shifted, -1, color, 1 if scale > 1 else 2, cv2.LINE_AA)


def _draw_rate_graph(
    canvas: np.ndarray,
    *,
    reference_bpm: np.ndarray,
    ridge_time_s: np.ndarray,
    ridge_bpm: np.ndarray,
    event_rate_bpm: np.ndarray | None,
    frame_index: int,
    fps: float,
    window_seconds: float,
    stimulus_s: tuple[float, float] | None,
    supplied_bouts_s: list[tuple[float, float]],
    detected_spans_s: list[tuple[float, float]],
    event_display_smoothed: bool,
) -> None:
    rect = (590, 350, 1130, 505)
    x0, y0, x1, y1 = rect
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (92, 98, 102), 1)
    duration = max((reference_bpm.size - 1) / fps, 1e-9)
    shade = canvas.copy()
    if stimulus_s is not None:
        left = int(round(x0 + stimulus_s[0] / duration * (x1 - x0)))
        right = int(round(x0 + stimulus_s[1] / duration * (x1 - x0)))
        cv2.rectangle(shade, (left, y0), (right, y1), (145, 70, 145), -1)
    for start, stop in supplied_bouts_s:
        left = int(round(x0 + start / duration * (x1 - x0)))
        right = int(round(x0 + stop / duration * (x1 - x0)))
        cv2.rectangle(shade, (left, y0), (right, y1), (45, 45, 190), -1)
    for start, stop in detected_spans_s:
        left = int(round(x0 + start / duration * (x1 - x0)))
        right = int(round(x0 + stop / duration * (x1 - x0)))
        cv2.rectangle(shade, (left, y0), (right, y1), (180, 120, 35), -1)
    cv2.addWeighted(shade, 0.18, canvas, 0.82, 0.0, canvas)
    for bpm in (100.0, 120.0, 140.0, 160.0):
        y = int(round(y1 - (bpm - 90.0) / 80.0 * (y1 - y0)))
        cv2.line(canvas, (x0, y), (x1, y), (48, 53, 56), 1)
        _text(canvas, str(int(bpm)), (x0 + 4, y - 3), scale=0.34, color=(145, 150, 153))
    reference_points = _polyline_points(reference_bpm, rect=rect, y_range=(90.0, 170.0))
    cv2.polylines(canvas, [reference_points], False, (220, 220, 220), 2, cv2.LINE_AA)
    ridge_x = x0 + np.asarray(ridge_time_s) / duration * (x1 - x0)
    ridge_y = y1 - np.clip((np.asarray(ridge_bpm) - 90.0) / 80.0, 0.0, 1.0) * (y1 - y0)
    points = np.column_stack([ridge_x, ridge_y]).round().astype(np.int32)
    split_at = np.flatnonzero(np.diff(ridge_time_s) > 1.5) + 1
    for run in np.split(np.arange(points.shape[0]), split_at):
        if run.size >= 2:
            cv2.polylines(canvas, [points[run]], False, (0, 180, 255), 2, cv2.LINE_AA)
        for point in points[run]:
            cv2.circle(canvas, tuple(point), 2, (0, 180, 255), -1, cv2.LINE_AA)
    if event_rate_bpm is not None:
        finite = np.flatnonzero(np.isfinite(event_rate_bpm))
        event_points = _polyline_points(
            np.nan_to_num(event_rate_bpm, nan=90.0),
            rect=rect,
            y_range=(90.0, 170.0),
        )
        runs = np.split(finite, np.flatnonzero(np.diff(finite) > 1) + 1) if finite.size else []
        for run in runs:
            if run.size >= 2:
                cv2.polylines(canvas, [event_points[run]], False, (255, 160, 40), 1, cv2.LINE_AA)
    cursor_x = int(round(x0 + frame_index / max(reference_bpm.size - 1, 1) * (x1 - x0)))
    cv2.line(canvas, (cursor_x, y0), (cursor_x, y1), (255, 220, 30), 2)
    _text(
        canvas,
        (
            f"rate: reference / {window_seconds:g} s ridge / peak interval"
            + (" (display-smoothed)" if event_display_smoothed else "")
        ),
        (x0, y0 - 10),
        scale=0.42,
    )


def _draw_waveform(
    canvas: np.ndarray,
    *,
    trace: np.ndarray,
    frame_index: int,
    fps: float,
    window_seconds: float,
) -> None:
    x0, y0, x1, y1 = (590, 550, 1130, 620)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (92, 98, 102), 1)
    half = int(round(fps))
    start = max(0, frame_index - half)
    stop = min(trace.size, frame_index + half + 1)
    values = np.full(2 * half + 1, np.nan, dtype=np.float64)
    insert = half - (frame_index - start)
    values[insert : insert + stop - start] = trace[start:stop]
    finite_trace = trace[np.isfinite(trace)]
    scale = max(float(np.quantile(np.abs(finite_trace), 0.99)), 1e-9)
    x = np.linspace(x0, x1, values.size)
    y = (y0 + y1) / 2.0 - np.clip(values / scale, -1.0, 1.0) * (y1 - y0) * 0.44
    finite = np.flatnonzero(np.isfinite(values))
    runs = np.split(finite, np.flatnonzero(np.diff(finite) > 1) + 1) if finite.size else []
    for run in runs:
        if run.size >= 2:
            points = np.column_stack([x[run], y[run]]).round().astype(np.int32)
            cv2.polylines(canvas, [points], False, (0, 180, 255), 2, cv2.LINE_AA)
    center_x = int(round((x0 + x1) / 2.0))
    cv2.line(canvas, (center_x, y0), (center_x, y1), (255, 220, 30), 1)
    _text(
        canvas,
        f"held-out equal-mask mean: {window_seconds:g} second waveform",
        (x0, y0 - 9),
        scale=0.44,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a held-out embedded top-camera masked-mean rate window."
    )
    parser.add_argument("--roi-json", type=Path, required=True)
    parser.add_argument("--mask-summary", type=Path, required=True)
    parser.add_argument("--projection-arrays", type=Path, required=True)
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--trial-number", type=int, default=1)
    parser.add_argument("--method", default="masked_equal_mean")
    parser.add_argument("--reference-fps", type=float, default=100.0)
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--step-seconds", type=float, default=1.0)
    parser.add_argument("--output-fps", type=float, default=50.0)
    parser.add_argument("--ffmpeg", type=Path, default=Path("/usr/bin/ffmpeg"))
    parser.add_argument("--ffmpeg-preset", default="veryfast")
    parser.add_argument("--event-display-smoothing-s", type=float, default=0.0)
    parser.add_argument("--brady-summary", type=Path)
    parser.add_argument("--source-start-s", type=float, default=0.0)
    parser.add_argument("--source-stop-s", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads(args.mask_summary.read_text())
    video, roi_xywh = _load_roi(args.roi_json)
    arrays = np.load(args.projection_arrays)
    trace_key = f"trace_{args.method}"
    if trace_key not in arrays:
        raise KeyError(f"projection arrays do not contain {trace_key}")
    trace = np.asarray(arrays[trace_key], dtype=np.float64)
    event_rate_key = f"event_rate_hz_{args.method}"
    event_rate_bpm = (
        np.asarray(arrays[event_rate_key], dtype=np.float64) * 60.0
        if event_rate_key in arrays
        else None
    )
    masks = np.asarray(arrays["fold_masks"], dtype=bool)
    x0, y0, roi_width, roi_height = roi_xywh
    if masks.shape != (2, roi_height, roi_width):
        raise ValueError("fold mask shape does not match the ROI")
    if Path(summary["source_video"]).resolve() != video.resolve():
        raise ValueError("mask summary belongs to a different video")

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"could not open source video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    source_height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    source_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    if source_frames != trace.size:
        capture.release()
        raise ValueError("projection trace length does not match source video")
    if source_width > 544 or source_height > 512:
        capture.release()
        raise ValueError("source video exceeds native display panel")

    partitions = temporal_half_partitions(
        trace.size,
        fps=fps,
        guard_seconds=float(summary["guard_seconds"]),
    )
    model_indices = crossfit_model_indices(partitions, frame_count=trace.size)
    reference_hz = read_numeric_xlsx_row(
        args.workbook,
        sheet_name="heart_rate_trace",
        row_number=int(args.trial_number) + 1,
    )
    band_hz = tuple(float(value) for value in summary["band_hz"])
    sweep_summary, ridge_time_s, ridge_hz, _ridge_reference = evaluate_window(
        trace,
        reference_hz,
        fps=fps,
        reference_fps=float(args.reference_fps),
        window_seconds=float(args.window_seconds),
        step_seconds=float(args.step_seconds),
        band_hz=band_hz,
    )
    frame_time = np.arange(trace.size, dtype=np.float64) / fps
    reference_bpm = _reference_at_times(
        reference_hz, frame_time, float(args.reference_fps)
    ) * 60.0
    state_scale = max(float(np.quantile(np.abs(trace[np.isfinite(trace)]), 0.99)), 1e-9)
    displayed_event_rate_bpm = (
        smooth_finite_runs(
            event_rate_bpm,
            sigma_samples=float(args.event_display_smoothing_s) * fps,
        )
        if event_rate_bpm is not None
        else None
    )
    stimulus_s: tuple[float, float] | None = None
    supplied_bouts_s: list[tuple[float, float]] = []
    detected_spans_s: list[tuple[float, float]] = []
    if args.brady_summary is not None:
        brady = json.loads(args.brady_summary.read_text())
        stimulus_s = tuple(float(value) for value in brady["stimulus_s"])
        supplied_bouts_s = [
            (float(values[0]), float(values[1]))
            for values in brady["stimulus_brady_bouts_s"]
        ]
        detected_spans_s = [
            (float(values[0]), float(values[1]))
            for values in brady["top_detected_response_spans"]
        ]

    render_start = max(0, int(round(float(args.source_start_s) * fps)))
    render_stop = (
        trace.size
        if args.source_stop_s is None
        else min(trace.size, int(round(float(args.source_stop_s) * fps)))
    )
    if render_stop <= render_start:
        capture.release()
        raise ValueError("source render interval is empty")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.rendering{output.suffix}")
    process = _ffmpeg_process(
        ffmpeg=args.ffmpeg,
        output=temporary,
        output_fps=float(args.output_fps),
        preset=str(args.ffmpeg_preset),
    )
    if process.stdin is None:
        capture.release()
        raise RuntimeError("FFmpeg stdin pipe was not created")
    capture.set(cv2.CAP_PROP_POS_FRAMES, render_start)
    try:
        for frame_index in range(render_start, render_stop):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"could not decode source frame {frame_index}")
            canvas = np.full((_CANVAS_SIZE[1], _CANVAS_SIZE[0], 3), 25, dtype=np.uint8)
            sx, sy = _SOURCE_ORIGIN
            canvas[sy : sy + source_height, sx : sx + source_width] = frame
            fold_index = int(model_indices[frame_index])
            outline_color = _FOLD_COLORS[fold_index] if fold_index >= 0 else (130, 130, 130)
            cv2.rectangle(
                canvas,
                (sx + x0, sy + y0),
                (sx + x0 + roi_width - 1, sy + y0 + roi_height - 1),
                outline_color,
                2,
            )
            if fold_index >= 0:
                _draw_mask_outline(
                    canvas,
                    masks[fold_index],
                    origin=(sx + x0, sy + y0),
                    scale=1,
                    color=outline_color,
                )

            roi = frame[y0 : y0 + roi_height, x0 : x0 + roi_width]
            raw_zoom = cv2.resize(
                roi,
                (roi_width * _ROI_SCALE, roi_height * _ROI_SCALE),
                interpolation=cv2.INTER_NEAREST,
            )
            state = np.full((roi_height, roi_width, 3), 42, dtype=np.uint8)
            if fold_index >= 0 and np.isfinite(trace[frame_index]):
                color = _color_pixel_change(
                    np.asarray([trace[frame_index]]), scale=state_scale
                )[0]
                state[masks[fold_index]] = color
            state_zoom = cv2.resize(
                state,
                (roi_width * _ROI_SCALE, roi_height * _ROI_SCALE),
                interpolation=cv2.INTER_NEAREST,
            )
            rx, ry = _RAW_ROI_ORIGIN
            tx, ty = _STATE_ROI_ORIGIN
            canvas[ry : ry + raw_zoom.shape[0], rx : rx + raw_zoom.shape[1]] = raw_zoom
            canvas[ty : ty + state_zoom.shape[0], tx : tx + state_zoom.shape[1]] = state_zoom
            if fold_index >= 0:
                _draw_mask_outline(
                    canvas,
                    masks[fold_index],
                    origin=(tx, ty),
                    scale=_ROI_SCALE,
                    color=outline_color,
                )

            _text(canvas, "embedded top camera", (16, 27), scale=0.52)
            _text(canvas, "raw ROI", (rx, 27), scale=0.44)
            _text(canvas, "uniform masked mean", (tx, 27), scale=0.44)
            _draw_rate_graph(
                canvas,
                reference_bpm=reference_bpm,
                ridge_time_s=ridge_time_s,
                ridge_bpm=ridge_hz * 60.0,
                event_rate_bpm=displayed_event_rate_bpm,
                frame_index=frame_index,
                fps=fps,
                window_seconds=float(args.window_seconds),
                stimulus_s=stimulus_s,
                supplied_bouts_s=supplied_bouts_s,
                detected_spans_s=detected_spans_s,
                event_display_smoothed=float(args.event_display_smoothing_s) > 0.0,
            )
            _draw_waveform(
                canvas,
                trace=trace,
                frame_index=frame_index,
                fps=fps,
                window_seconds=float(args.window_seconds),
            )
            time_s = frame_index / fps
            current_hz = nearest_ridge_value(
                ridge_time_s,
                ridge_hz,
                time_s=time_s,
                maximum_distance_s=0.5 * float(args.step_seconds) + 1e-9,
            )
            fold_text = f"fold {fold_index + 1} held out" if fold_index >= 0 else "midpoint guard"
            rate_text = f"{current_hz * 60.0:.1f} bpm candidate" if np.isfinite(current_hz) else "no window estimate"
            event_bpm = event_rate_bpm[frame_index] if event_rate_bpm is not None else float("nan")
            event_text = f"peak interval {event_bpm:.1f} bpm" if np.isfinite(event_bpm) else "no peak interval"
            _text(canvas, f"source {frame_index:04d}/{trace.size-1:04d}  t={time_s:06.3f} s", (16, 580), scale=0.48)
            _text(canvas, fold_text, (16, 608), scale=0.45, color=outline_color)
            _text(canvas, rate_text, (235, 608), scale=0.45, color=(0, 180, 255))
            _text(canvas, event_text, (440, 608), scale=0.42, color=(255, 160, 40))
            _text(canvas, "window rate, not beat timing", (590, 640), scale=0.38, color=(170, 175, 178))
            process.stdin.write(canvas.tobytes())
            if (frame_index - render_start) % 2000 == 0:
                print(f"rendered {frame_index-render_start}/{render_stop-render_start-1}", flush=True)
        process.stdin.close()
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"FFmpeg exited with status {return_code}")
        temporary.replace(output)
    except Exception:
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
        process.kill()
        process.wait()
        raise
    finally:
        capture.release()

    metadata = {
        "analysis_status": "descriptive_heldout_window_rate_visualization",
        "source_video": str(video.resolve()),
        "roi_json": str(args.roi_json.resolve()),
        "mask_summary": str(args.mask_summary.resolve()),
        "projection_arrays": str(args.projection_arrays.resolve()),
        "method": str(args.method),
        "window_seconds": float(args.window_seconds),
        "step_seconds": float(args.step_seconds),
        "ridge_summary": sweep_summary,
        "source_frames": int(trace.size),
        "render_frame_start": render_start,
        "render_frame_stop_exclusive": render_stop,
        "rendered_frames": render_stop - render_start,
        "source_fps": fps,
        "output_fps": float(args.output_fps),
        "playback_slowdown": fps / float(args.output_fps),
        "state_display": "uniform signed held-out equal-mask mean within the active fold mask",
        "event_display_smoothing_seconds": float(args.event_display_smoothing_s),
        "event_display_smoothing_scope": "visualization only; metrics and numeric status use raw peak intervals",
        "interpretation": "window-level candidate rate; no beat timing claim",
        "brady_summary": str(args.brady_summary.resolve()) if args.brady_summary is not None else None,
        "output_video": str(output),
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
