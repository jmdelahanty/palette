from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from analyze_embedded_positive_control import (
    _event_train,
    _reference_at_times,
    read_numeric_xlsx_row,
)
from analyze_embedded_side_chambers import chamber_masks_from_json
from render_embedded_positive_control_overlay import (
    _color_pixel_change,
    _event_rate_trace,
    _ffmpeg_process,
    _polyline_points,
    _text,
)


_CANVAS_SIZE = (1152, 648)
_SOURCE_ORIGIN = (16, 44)
_RAW_ORIGIN = (590, 44)
_STATE_ORIGIN = (830, 44)
_COLORS = {
    "chamber_a": (65, 75, 245),
    "chamber_b": (235, 135, 45),
    "chamber_union": (190, 80, 190),
}


def chamber_crop_bounds(
    masks: dict[str, np.ndarray],
    *,
    margin_px: int,
) -> tuple[int, int, int, int]:
    union = np.zeros_like(next(iter(masks.values())), dtype=bool)
    for mask in masks.values():
        union |= np.asarray(mask, dtype=bool)
    y, x = np.nonzero(union)
    if x.size == 0:
        raise ValueError("chamber masks are empty")
    height, width = union.shape
    x0 = max(0, int(np.min(x)) - int(margin_px))
    y0 = max(0, int(np.min(y)) - int(margin_px))
    x1 = min(width, int(np.max(x)) + int(margin_px) + 1)
    y1 = min(height, int(np.max(y)) + int(margin_px) + 1)
    return x0, y0, x1, y1


def _robust_scale(values: np.ndarray) -> float:
    source = np.asarray(values, dtype=np.float64)
    center = float(np.median(source))
    return max(1.4826 * float(np.median(np.abs(source - center))), 1e-9)


def _draw_rate_graph(
    canvas: np.ndarray,
    *,
    reference_bpm: np.ndarray,
    event_rates_bpm: dict[str, np.ndarray],
    frame_index: int,
) -> None:
    rect = (590, 315, 1130, 465)
    x0, y0, x1, y1 = rect
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (92, 98, 102), 1)
    for bpm in (100.0, 120.0, 140.0, 160.0):
        y = int(round(y1 - (bpm - 90.0) / 80.0 * (y1 - y0)))
        cv2.line(canvas, (x0, y), (x1, y), (48, 53, 56), 1)
        _text(canvas, str(int(bpm)), (x0 + 4, y - 3), scale=0.32, color=(145, 150, 153))
    reference_points = _polyline_points(reference_bpm, rect=rect, y_range=(90.0, 170.0))
    cv2.polylines(canvas, [reference_points], False, (220, 220, 220), 2, cv2.LINE_AA)
    for label, rate in event_rates_bpm.items():
        finite = np.flatnonzero(np.isfinite(rate))
        points = _polyline_points(np.nan_to_num(rate, nan=90.0), rect=rect, y_range=(90.0, 170.0))
        runs = np.split(finite, np.flatnonzero(np.diff(finite) > 1) + 1) if finite.size else []
        for run in runs:
            if run.size >= 2:
                cv2.polylines(canvas, [points[run]], False, _COLORS[label], 1, cv2.LINE_AA)
    cursor_x = int(round(x0 + frame_index / max(reference_bpm.size - 1, 1) * (x1 - x0)))
    cv2.line(canvas, (cursor_x, y0), (cursor_x, y1), (255, 220, 30), 2)
    _text(canvas, "rate: reference / chamber A / chamber B / union", (x0, y0 - 10), scale=0.42)


def _draw_waveforms(
    canvas: np.ndarray,
    *,
    traces: dict[str, np.ndarray],
    frame_index: int,
    fps: float,
) -> None:
    x0, y0, x1, y1 = (590, 515, 1130, 610)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (92, 98, 102), 1)
    half = int(round(1.5 * fps))
    start = max(0, frame_index - half)
    stop = min(next(iter(traces.values())).size, frame_index + half + 1)
    x = np.linspace(x0, x1, 2 * half + 1)
    for label in ("chamber_a", "chamber_b"):
        values = np.full(2 * half + 1, np.nan)
        insert = half - (frame_index - start)
        values[insert : insert + stop - start] = traces[label][start:stop]
        values /= _robust_scale(traces[label])
        y = (y0 + y1) / 2.0 - np.clip(values, -2.0, 2.0) / 2.0 * (y1 - y0) * 0.44
        finite = np.flatnonzero(np.isfinite(y))
        if finite.size >= 2:
            points = np.column_stack([x[finite], y[finite]]).round().astype(np.int32)
            cv2.polylines(canvas, [points], False, _COLORS[label], 2, cv2.LINE_AA)
    center_x = int(round((x0 + x1) / 2.0))
    cv2.line(canvas, (center_x, y0), (center_x, y1), (255, 220, 30), 1)
    _text(canvas, "bandpassed polygon means: chamber A / chamber B", (x0, y0 - 9), scale=0.42)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render fixed side-camera chamber means and events.")
    parser.add_argument("--chambers-json", type=Path, required=True)
    parser.add_argument("--analysis-arrays", type=Path, required=True)
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--trial-number", type=int, default=1)
    parser.add_argument("--reference-fps", type=float, default=100.0)
    parser.add_argument("--band-hz", type=float, nargs=2, default=(1.5, 4.0))
    parser.add_argument("--output-fps", type=float, default=25.0)
    parser.add_argument("--ffmpeg", type=Path, default=Path("/usr/bin/ffmpeg"))
    parser.add_argument("--ffmpeg-preset", default="medium")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    video, masks, _annotation_frame = chamber_masks_from_json(args.chambers_json)
    arrays = np.load(args.analysis_arrays)
    traces = {
        "chamber_a": np.asarray(arrays["filtered_chamber_a_base"], dtype=np.float64),
        "chamber_b": np.asarray(arrays["filtered_chamber_b_base"], dtype=np.float64),
        "chamber_union": np.asarray(arrays["filtered_chamber_union_base"], dtype=np.float64),
    }
    frame_count = traces["chamber_a"].size
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {video}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    source_height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if (source_height, source_width) != next(iter(masks.values())).shape:
        capture.release()
        raise ValueError("chamber mask shape does not match source video")
    if source_width > 544 or source_height > 512:
        capture.release()
        raise ValueError("source video exceeds native display panel")

    band_hz = (float(args.band_hz[0]), float(args.band_hz[1]))
    peaks = {}
    event_rates_bpm = {}
    for label, trace in traces.items():
        label_peaks, _polarity, _cv = _event_train(trace, fps=source_fps, max_hz=band_hz[1])
        peaks[label] = label_peaks
        event_rates_bpm[label] = _event_rate_trace(
            label_peaks, frame_count=frame_count, fps=source_fps
        ) * 60.0
    reference_hz = read_numeric_xlsx_row(
        args.workbook,
        sheet_name="heart_rate_trace",
        row_number=int(args.trial_number) + 1,
    )
    times = np.arange(frame_count, dtype=np.float64) / source_fps
    reference_bpm = _reference_at_times(reference_hz, times, float(args.reference_fps)) * 60.0

    crop_x0, crop_y0, crop_x1, crop_y1 = chamber_crop_bounds(masks, margin_px=8)
    crop_width, crop_height = crop_x1 - crop_x0, crop_y1 - crop_y0
    scale = max(1, min(2, 220 // crop_width, 250 // crop_height))
    state_scales = {label: _robust_scale(trace) for label, trace in traces.items()}

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
    try:
        for frame_index in range(frame_count):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"could not decode source frame {frame_index}")
            canvas = np.full((_CANVAS_SIZE[1], _CANVAS_SIZE[0], 3), 25, dtype=np.uint8)
            sx, sy = _SOURCE_ORIGIN
            canvas[sy : sy + source_height, sx : sx + source_width] = frame
            for label in ("chamber_a", "chamber_b"):
                contour_mask = masks[label].astype(np.uint8)
                contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                shifted = [contour + np.asarray([[[sx, sy]]], dtype=np.int32) for contour in contours]
                cv2.drawContours(canvas, shifted, -1, _COLORS[label], 2, cv2.LINE_AA)

            crop = frame[crop_y0:crop_y1, crop_x0:crop_x1]
            raw_zoom = cv2.resize(crop, (crop_width * scale, crop_height * scale), interpolation=cv2.INTER_NEAREST)
            state = np.full((crop_height, crop_width, 3), 38, dtype=np.uint8)
            for label in ("chamber_a", "chamber_b"):
                local_mask = masks[label][crop_y0:crop_y1, crop_x0:crop_x1]
                color = _color_pixel_change(
                    np.asarray([traces[label][frame_index]]), scale=state_scales[label]
                )[0]
                state[local_mask] = color
            state_zoom = cv2.resize(state, (crop_width * scale, crop_height * scale), interpolation=cv2.INTER_NEAREST)
            rx, ry = _RAW_ORIGIN
            tx, ty = _STATE_ORIGIN
            canvas[ry : ry + raw_zoom.shape[0], rx : rx + raw_zoom.shape[1]] = raw_zoom
            canvas[ty : ty + state_zoom.shape[0], tx : tx + state_zoom.shape[1]] = state_zoom
            cv2.rectangle(canvas, (rx, ry), (rx + raw_zoom.shape[1], ry + raw_zoom.shape[0]), (100, 105, 108), 1)
            cv2.rectangle(canvas, (tx, ty), (tx + state_zoom.shape[1], ty + state_zoom.shape[0]), (100, 105, 108), 1)

            _text(canvas, "side camera, native pixels", (16, 27), scale=0.52)
            _text(canvas, "raw chamber crop", (rx, 27), scale=0.44)
            _text(canvas, "polygon mean state", (tx, 27), scale=0.44)
            _draw_rate_graph(canvas, reference_bpm=reference_bpm, event_rates_bpm=event_rates_bpm, frame_index=frame_index)
            _draw_waveforms(canvas, traces=traces, frame_index=frame_index, fps=source_fps)
            _text(canvas, f"source {frame_index:04d}/{frame_count-1:04d}   t={frame_index/source_fps:06.3f} s", (16, 580), scale=0.50)
            _text(canvas, f"reference {reference_bpm[frame_index]:.1f} bpm", (16, 608), scale=0.45)
            status_x = 230
            for label in ("chamber_a", "chamber_b"):
                rate = event_rates_bpm[label][frame_index]
                text = f"{label[-1].upper()} {rate:.1f}" if np.isfinite(rate) else f"{label[-1].upper()} pending"
                _text(canvas, text, (status_x, 608), scale=0.43, color=_COLORS[label])
                status_x += 125
            process.stdin.write(canvas.tobytes())
            if frame_index % 1000 == 0:
                print(f"rendered {frame_index}/{frame_count-1}", flush=True)
        process.stdin.close()
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"FFmpeg exited with status {return_code}")
        temporary.replace(output)
    except Exception:
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
        process.kill(); process.wait()
        raise
    finally:
        capture.release()

    metadata = {
        "analysis_status": "exploratory_fixed_chamber_mean_visualization",
        "source_video": str(video.resolve()),
        "chambers_json": str(args.chambers_json.resolve()),
        "analysis_arrays": str(args.analysis_arrays.resolve()),
        "workbook": str(args.workbook.resolve()),
        "source_frames": frame_count,
        "source_fps": source_fps,
        "output_fps": float(args.output_fps),
        "ffmpeg_preset": str(args.ffmpeg_preset),
        "playback_slowdown": source_fps / float(args.output_fps),
        "band_hz": list(band_hz),
        "event_counts": {label: int(value.size) for label, value in peaks.items()},
        "state_display": "uniform signed bandpassed mean within each fixed polygon",
        "output_video": str(output),
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
