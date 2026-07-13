from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from analyze_embedded_crossfit_mask import (
    _bandpass,
    _fit_scale,
    temporal_half_partitions,
)
from analyze_embedded_positive_control import (
    _extract_video,
    _load_roi,
    _reference_at_times,
    read_numeric_xlsx_row,
)
from render_embedded_positive_control_overlay import (
    _color_pixel_change,
    _ffmpeg_process,
    _polyline_points,
    _text,
)


_CANVAS_SIZE = (1152, 648)
_SOURCE_ORIGIN = (16, 44)
_RAW_ROI_ORIGIN = (590, 44)
_COLOR_ROI_ORIGIN = (830, 44)
_ROI_SCALE = 10


def crossfit_model_indices(
    partitions: tuple[tuple[np.ndarray, np.ndarray], ...],
    *,
    frame_count: int,
) -> np.ndarray:
    model = np.full(int(frame_count), -1, dtype=np.int16)
    for fold_index, (_discovery, confirmation) in enumerate(partitions):
        rows = np.asarray(confirmation, dtype=bool)
        if rows.shape != (frame_count,):
            raise ValueError("cross-fit confirmation rows have the wrong shape")
        if np.any(model[rows] >= 0):
            raise ValueError("cross-fit confirmation rows overlap")
        model[rows] = int(fold_index)
    return model


def _draw_mask_outline(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    color: tuple[int, int, int],
    scale: int,
) -> None:
    binary = np.asarray(mask, dtype=np.uint8)
    contours, _hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    scaled = [np.asarray(contour, dtype=np.int32) * int(scale) for contour in contours]
    if scaled:
        cv2.drawContours(image, scaled, -1, color, 1, cv2.LINE_AA)


def _draw_rate_graph(
    canvas: np.ndarray,
    *,
    reference_bpm: np.ndarray,
    ridge_time_s: np.ndarray,
    ridge_bpm: np.ndarray,
    event_rate_bpm: np.ndarray,
    frame_index: int,
    fps: float,
) -> None:
    rect = (590, 350, 1130, 505)
    x0, y0, x1, y1 = rect
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (92, 98, 102), 1)
    for bpm in (100.0, 120.0, 140.0, 160.0):
        y = int(round(y1 - (bpm - 90.0) / 80.0 * (y1 - y0)))
        cv2.line(canvas, (x0, y), (x1, y), (48, 53, 56), 1)
        _text(canvas, f"{int(bpm)}", (x0 + 4, y - 3), scale=0.34, color=(145, 150, 153))
    reference_points = _polyline_points(reference_bpm, rect=rect, y_range=(90.0, 170.0))
    cv2.polylines(canvas, [reference_points], False, (220, 220, 220), 2, cv2.LINE_AA)

    duration = max((reference_bpm.size - 1) / float(fps), 1e-9)
    ridge_x = x0 + np.asarray(ridge_time_s, dtype=np.float64) / duration * (x1 - x0)
    ridge_y = y1 - np.clip((np.asarray(ridge_bpm) - 90.0) / 80.0, 0.0, 1.0) * (y1 - y0)
    ridge_points = np.column_stack([ridge_x, ridge_y]).round().astype(np.int32)
    split_at = np.flatnonzero(np.diff(np.asarray(ridge_time_s)) > 1.5) + 1
    for run in np.split(np.arange(ridge_points.shape[0]), split_at):
        if run.size >= 2:
            cv2.polylines(canvas, [ridge_points[run]], False, (255, 150, 20), 2, cv2.LINE_AA)
        for point in ridge_points[run]:
            cv2.circle(canvas, tuple(point), 2, (255, 150, 20), -1, cv2.LINE_AA)
    finite_events = np.flatnonzero(np.isfinite(event_rate_bpm))
    if finite_events.size:
        event_points = _polyline_points(
            np.nan_to_num(event_rate_bpm, nan=90.0),
            rect=rect,
            y_range=(90.0, 170.0),
        )
        event_runs = np.split(
            finite_events,
            np.flatnonzero(np.diff(finite_events) > 1) + 1,
        )
        for run in event_runs:
            if run.size >= 2:
                cv2.polylines(
                    canvas,
                    [event_points[run]],
                    False,
                    (0, 150, 255),
                    1,
                    cv2.LINE_AA,
                )
    cursor_x = int(round(x0 + frame_index / max(reference_bpm.size - 1, 1) * (x1 - x0)))
    cv2.line(canvas, (cursor_x, y0), (cursor_x, y1), (255, 220, 30), 2)
    _text(
        canvas,
        "rate: side reference / held-out ridge / event interval",
        (x0, y0 - 10),
        scale=0.42,
    )


def _draw_waveform(
    canvas: np.ndarray,
    *,
    signal: np.ndarray,
    frame_index: int,
    fps: float,
) -> None:
    x0, y0, x1, y1 = (590, 550, 1130, 620)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (92, 98, 102), 1)
    half = int(round(1.5 * fps))
    start = max(0, frame_index - half)
    stop = min(signal.size, frame_index + half + 1)
    values = np.full(2 * half + 1, np.nan, dtype=np.float64)
    insert = half - (frame_index - start)
    values[insert : insert + stop - start] = signal[start:stop]
    finite_signal = np.asarray(signal)[np.isfinite(signal)]
    scale = max(float(np.quantile(np.abs(finite_signal), 0.99)), 1e-9)
    x = np.linspace(x0, x1, values.size)
    y = (y0 + y1) / 2.0 - np.clip(values / scale, -1.0, 1.0) * (y1 - y0) * 0.45
    finite = np.flatnonzero(np.isfinite(values))
    runs = np.split(finite, np.flatnonzero(np.diff(finite) > 1) + 1) if finite.size else []
    for run in runs:
        if run.size >= 2:
            points = np.column_stack([x[run], y[run]]).round().astype(np.int32)
            cv2.polylines(canvas, [points], False, (0, 150, 255), 2, cv2.LINE_AA)
    center_x = int(round((x0 + x1) / 2.0))
    cv2.line(canvas, (center_x, y0), (center_x, y1), (255, 220, 30), 1)
    _text(canvas, "cross-fit held-out waveform: 3 second window", (x0, y0 - 9), scale=0.44)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the embedded top-camera cross-fit heart-mask diagnostic."
    )
    parser.add_argument("--roi-json", type=Path, required=True)
    parser.add_argument("--analysis-summary", type=Path, required=True)
    parser.add_argument("--analysis-arrays", type=Path, required=True)
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--trial-number", type=int, default=1)
    parser.add_argument("--reference-fps", type=float, default=100.0)
    parser.add_argument("--output-fps", type=float, default=50.0)
    parser.add_argument("--ffmpeg", type=Path, default=Path("/usr/bin/ffmpeg"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads(args.analysis_summary.read_text())
    arrays = np.load(args.analysis_arrays)
    video, roi_xywh = _load_roi(args.roi_json)
    pixels, _, duplicates, _, source_fps, frame_shape = _extract_video(
        video, roi_xywh, ring_margin=8
    )
    if Path(summary["source_video"]).resolve() != video.resolve():
        raise ValueError("analysis summary belongs to a different source video")
    if tuple(summary["roi_xywh"]) != roi_xywh:
        raise ValueError("analysis summary ROI does not match roi-json")
    if int(summary["source_frames"]) != pixels.shape[0] or not np.isclose(
        float(summary["source_fps"]), source_fps
    ):
        raise ValueError("analysis summary timebase does not match decoded video")

    masks = np.asarray(arrays["fold_masks"], dtype=bool)
    loadings = np.asarray(arrays["fold_loadings"], dtype=np.float64).reshape(2, -1)
    ridge_time_s = np.asarray(arrays["ridge_time_s"], dtype=np.float64)
    ridge_bpm = np.asarray(arrays["ridge_hz"], dtype=np.float64) * 60.0
    event_rate_bpm = np.asarray(arrays["event_rate_trace_hz"], dtype=np.float64) * 60.0
    height, width = roi_xywh[3], roi_xywh[2]
    if masks.shape != (2, height, width) or loadings.shape != (2, height * width):
        raise ValueError("analysis mask/loading shapes do not match ROI")

    partitions = temporal_half_partitions(
        pixels.shape[0],
        fps=source_fps,
        guard_seconds=float(summary["guard_seconds"]),
    )
    model_indices = crossfit_model_indices(partitions, frame_count=pixels.shape[0])
    crossfit_pixels = np.full_like(pixels, np.nan, dtype=np.float64)
    crossfit_trace = np.full(pixels.shape[0], np.nan, dtype=np.float64)
    band_hz = tuple(float(value) for value in summary["band_hz"])
    for fold_index, (discovery_rows, confirmation_rows) in enumerate(partitions):
        standardized, _scale, _usable = _fit_scale(pixels, discovery_rows)
        filtered = _bandpass(standardized, fps=source_fps, band_hz=band_hz)
        crossfit_pixels[confirmation_rows] = filtered[confirmation_rows]
        crossfit_trace[confirmation_rows] = filtered[confirmation_rows] @ loadings[fold_index]
    finite_selected = np.zeros_like(crossfit_pixels, dtype=bool)
    for fold_index in range(2):
        rows = model_indices == fold_index
        finite_selected[np.ix_(rows, masks[fold_index].reshape(-1))] = True
    color_scale = max(
        float(np.quantile(np.abs(crossfit_pixels[finite_selected]), 0.995)),
        1e-9,
    )

    reference_hz = read_numeric_xlsx_row(
        args.workbook,
        sheet_name="heart_rate_trace",
        row_number=int(args.trial_number) + 1,
    )
    frame_time = np.arange(pixels.shape[0], dtype=np.float64) / source_fps
    reference_bpm = _reference_at_times(
        reference_hz, frame_time, float(args.reference_fps)
    ) * 60.0

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.rendering{output.suffix}")
    process = _ffmpeg_process(
        ffmpeg=args.ffmpeg,
        output=temporary,
        output_fps=float(args.output_fps),
    )
    if process.stdin is None:
        raise RuntimeError("FFmpeg stdin pipe was not created")
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        process.kill()
        raise ValueError(f"could not reopen video: {video}")
    source_height, source_width = frame_shape
    x0, y0, roi_width, roi_height = roi_xywh
    fold_colors = ((40, 60, 255), (255, 100, 40))
    try:
        for frame_index in range(pixels.shape[0]):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"could not decode source frame {frame_index}")
            canvas = np.full((_CANVAS_SIZE[1], _CANVAS_SIZE[0], 3), 25, dtype=np.uint8)
            sx, sy = _SOURCE_ORIGIN
            canvas[sy : sy + source_height, sx : sx + source_width] = frame
            fold_index = int(model_indices[frame_index])
            outline_color = fold_colors[fold_index] if fold_index >= 0 else (130, 130, 130)
            cv2.rectangle(
                canvas,
                (sx + x0, sy + y0),
                (sx + x0 + roi_width - 1, sy + y0 + roi_height - 1),
                outline_color,
                2,
                cv2.LINE_AA,
            )

            raw_roi = frame[y0 : y0 + roi_height, x0 : x0 + roi_width]
            raw_zoom = cv2.resize(
                raw_roi,
                (roi_width * _ROI_SCALE, roi_height * _ROI_SCALE),
                interpolation=cv2.INTER_NEAREST,
            )
            dynamic = np.full((roi_height, roi_width, 3), 38, dtype=np.uint8)
            if fold_index >= 0:
                active_mask = masks[fold_index]
                values = crossfit_pixels[frame_index].reshape(roi_height, roi_width)
                mapped = _color_pixel_change(values, scale=color_scale)
                dynamic[active_mask] = mapped[active_mask]
                _draw_mask_outline(
                    raw_zoom,
                    active_mask,
                    color=outline_color,
                    scale=_ROI_SCALE,
                )
            dynamic_zoom = cv2.resize(
                dynamic,
                (roi_width * _ROI_SCALE, roi_height * _ROI_SCALE),
                interpolation=cv2.INTER_NEAREST,
            )
            rx, ry = _RAW_ROI_ORIGIN
            cx, cy = _COLOR_ROI_ORIGIN
            canvas[ry : ry + raw_zoom.shape[0], rx : rx + raw_zoom.shape[1]] = raw_zoom
            canvas[cy : cy + dynamic_zoom.shape[0], cx : cx + dynamic_zoom.shape[1]] = dynamic_zoom
            cv2.rectangle(canvas, (rx, ry), (rx + raw_zoom.shape[1], ry + raw_zoom.shape[0]), (100, 105, 108), 1)
            cv2.rectangle(canvas, (cx, cy), (cx + dynamic_zoom.shape[1], cy + dynamic_zoom.shape[0]), (100, 105, 108), 1)

            _text(canvas, "embedded top camera, native pixels", (16, 27), scale=0.52)
            _text(canvas, "raw ROI + held-out mask", (rx, 27), scale=0.42)
            _text(canvas, "selected band change", (cx, 27), scale=0.42)
            _draw_rate_graph(
                canvas,
                reference_bpm=reference_bpm,
                ridge_time_s=ridge_time_s,
                ridge_bpm=ridge_bpm,
                event_rate_bpm=event_rate_bpm,
                frame_index=frame_index,
                fps=source_fps,
            )
            _draw_waveform(
                canvas,
                signal=crossfit_trace,
                frame_index=frame_index,
                fps=source_fps,
            )
            _text(
                canvas,
                f"source {frame_index:04d}/{pixels.shape[0] - 1:04d}   t={frame_index / source_fps:06.3f} s",
                (16, 577),
                scale=0.50,
            )
            if fold_index >= 0:
                learned_half = "late" if fold_index == 1 else "early"
                _text(
                    canvas,
                    f"active fold {fold_index}: mask learned on {learned_half} half, {int(np.count_nonzero(masks[fold_index]))} pixels",
                    (16, 607),
                    scale=0.45,
                    color=outline_color,
                )
            else:
                _text(canvas, "guard interval: analysis intentionally blank", (16, 607), scale=0.45, color=(150, 150, 150))
            _text(canvas, f"side reference {reference_bpm[frame_index]:.1f} bpm", (16, 634), scale=0.45)
            if np.isfinite(event_rate_bpm[frame_index]):
                _text(
                    canvas,
                    f"event interval {event_rate_bpm[frame_index]:.1f} bpm",
                    (260, 634),
                    scale=0.45,
                    color=(0, 170, 255),
                )
            if duplicates[frame_index]:
                _text(canvas, "duplicate ROI frame", (440, 634), scale=0.38, color=(80, 180, 255))
            process.stdin.write(canvas.tobytes())
            if frame_index % 1000 == 0:
                print(f"rendered {frame_index}/{pixels.shape[0] - 1}", flush=True)
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
        "analysis_status": "exploratory_crossfit_mask_visualization",
        "source_video": str(video.resolve()),
        "roi_json": str(args.roi_json.resolve()),
        "analysis_summary": str(args.analysis_summary.resolve()),
        "analysis_arrays": str(args.analysis_arrays.resolve()),
        "workbook": str(args.workbook.resolve()),
        "source_frames": int(pixels.shape[0]),
        "source_fps": float(source_fps),
        "output_fps": float(args.output_fps),
        "playback_slowdown": float(source_fps / float(args.output_fps)),
        "band_hz": list(band_hz),
        "fold_mask_pixel_counts": [int(np.count_nonzero(mask)) for mask in masks],
        "event_count": int(np.asarray(arrays["event_frame_indices"]).size),
        "valid_event_interval_count": int(np.asarray(arrays["event_interval_hz"]).size),
        "guard_frame_count": int(np.count_nonzero(model_indices < 0)),
        "pixel_change_color_scale_995": color_scale,
        "output_video": str(output),
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
