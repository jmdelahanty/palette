from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
from scipy.signal import detrend

from analyze_embedded_positive_control import (
    _bandpass,
    _event_train,
    _extract_video,
    _load_roi,
    _reference_at_times,
    build_candidate_signals,
    read_numeric_xlsx_row,
)


_CANVAS_SIZE = (1152, 648)
_SOURCE_ORIGIN = (16, 44)
_RAW_ROI_ORIGIN = (590, 44)
_COLOR_ROI_ORIGIN = (870, 44)
_ROI_PANEL_SIZE = (264, 280)
_MAX_ROI_SCALE = 10


def _text(
    image: np.ndarray,
    value: str,
    origin: tuple[int, int],
    *,
    scale: float = 0.48,
    color: tuple[int, int, int] = (235, 235, 235),
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        value,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _event_rate_trace(peaks: np.ndarray, *, frame_count: int, fps: float) -> np.ndarray:
    rates = np.full(frame_count, np.nan, dtype=np.float64)
    if peaks.size < 2:
        return rates
    intervals = np.diff(peaks)
    for index, interval in enumerate(intervals):
        if interval <= 0:
            continue
        start = int(peaks[index])
        stop = int(peaks[index + 1])
        rates[start:stop] = fps / float(interval)
    return rates


def _color_pixel_change(values: np.ndarray, *, scale: float) -> np.ndarray:
    normalized = np.clip(np.asarray(values, dtype=np.float64) / scale, -1.0, 1.0)
    color = np.full((*normalized.shape, 3), 42.0, dtype=np.float64)
    positive = np.maximum(normalized, 0.0)
    negative = np.maximum(-normalized, 0.0)
    color[..., 2] += 213.0 * positive
    color[..., 1] += 32.0 * np.abs(normalized)
    color[..., 0] += 213.0 * negative
    return np.clip(color, 0.0, 255.0).astype(np.uint8)


def _roi_display_scale(roi_width: int, roi_height: int) -> int:
    panel_width, panel_height = _ROI_PANEL_SIZE
    if roi_width < 1 or roi_height < 1:
        raise ValueError("ROI dimensions must be positive")
    return max(
        1,
        min(
            _MAX_ROI_SCALE,
            panel_width // int(roi_width),
            panel_height // int(roi_height),
        ),
    )


def _polyline_points(
    values: np.ndarray,
    *,
    rect: tuple[int, int, int, int],
    y_range: tuple[float, float],
) -> np.ndarray:
    x0, y0, x1, y1 = rect
    count = values.size
    x = np.linspace(x0, x1, count)
    low, high = y_range
    y = y1 - np.clip((values - low) / max(high - low, 1e-9), 0.0, 1.0) * (y1 - y0)
    return np.column_stack([x, y]).round().astype(np.int32)


def _draw_rate_graph(
    canvas: np.ndarray,
    *,
    reference_bpm: np.ndarray,
    estimated_bpm: np.ndarray,
    frame_index: int,
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
    finite = np.isfinite(estimated_bpm)
    if np.count_nonzero(finite) >= 2:
        all_points = _polyline_points(
            np.nan_to_num(estimated_bpm, nan=90.0), rect=rect, y_range=(90.0, 170.0)
        )
        runs = np.split(np.flatnonzero(finite), np.flatnonzero(np.diff(np.flatnonzero(finite)) > 1) + 1)
        for run in runs:
            if run.size >= 2:
                cv2.polylines(canvas, [all_points[run]], False, (0, 150, 255), 1, cv2.LINE_AA)
    cursor_x = int(round(x0 + frame_index / max(reference_bpm.size - 1, 1) * (x1 - x0)))
    cv2.line(canvas, (cursor_x, y0), (cursor_x, y1), (255, 220, 30), 2)
    _text(canvas, "rate: supplied reference / extracted events", (x0, y0 - 10), scale=0.44)


def _draw_waveform(
    canvas: np.ndarray,
    *,
    signal: np.ndarray,
    frame_index: int,
    fps: float,
    polarity: int,
    signal_label: str,
) -> None:
    rect = (590, 550, 1130, 620)
    x0, y0, x1, y1 = rect
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (92, 98, 102), 1)
    half_window = int(round(1.5 * fps))
    start = max(0, frame_index - half_window)
    stop = min(signal.size, frame_index + half_window + 1)
    window = polarity * signal[start:stop]
    scale = max(float(np.quantile(np.abs(signal), 0.99)), 1e-9)
    values = np.full(2 * half_window + 1, np.nan, dtype=np.float64)
    insert = half_window - (frame_index - start)
    values[insert : insert + window.size] = window
    x = np.linspace(x0, x1, values.size)
    y = (y0 + y1) / 2.0 - np.clip(values / scale, -1.0, 1.0) * (y1 - y0) * 0.45
    finite = np.isfinite(y)
    points = np.column_stack([x[finite], y[finite]]).round().astype(np.int32)
    if points.shape[0] >= 2:
        cv2.polylines(canvas, [points], False, (0, 150, 255), 2, cv2.LINE_AA)
    center_x = int(round((x0 + x1) / 2.0))
    cv2.line(canvas, (center_x, y0), (center_x, y1), (255, 220, 30), 1)
    _text(canvas, f"{signal_label} waveform: 3 second window", (x0, y0 - 9), scale=0.44)


def _ffmpeg_process(
    *,
    ffmpeg: Path,
    output: Path,
    output_fps: float,
    crf: int = 18,
    preset: str = "medium",
) -> subprocess.Popen[bytes]:
    width, height = _CANVAS_SIZE
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        f"{output_fps:.8g}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        str(preset),
        "-crf",
        str(int(crf)),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a slow-motion inspection overlay for an embedded-fish heart ROI.")
    parser.add_argument("--roi-json", type=Path, required=True)
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--trial-number", type=int, default=1)
    parser.add_argument("--reference-fps", type=float, default=100.0)
    parser.add_argument("--band-hz", type=float, nargs=2, default=(1.5, 4.0))
    parser.add_argument(
        "--signal",
        choices=("roi_mean", "roi_ring_log_ratio", "band_pca1"),
        default="band_pca1",
    )
    parser.add_argument("--output-fps", type=float, default=50.0)
    parser.add_argument("--ffmpeg", type=Path, default=Path("/usr/bin/ffmpeg"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    video, roi_xywh = _load_roi(args.roi_json)
    pixels, ring_mean, duplicates, _, source_fps, frame_shape = _extract_video(
        video, roi_xywh, ring_margin=8
    )
    source_height, source_width = frame_shape
    if source_width > 544 or source_height > 512:
        raise ValueError(
            f"source frame {frame_shape} exceeds the native-pixel panel limit (512, 544)"
        )
    reference_hz = read_numeric_xlsx_row(
        args.workbook,
        sheet_name="heart_rate_trace",
        row_number=int(args.trial_number) + 1,
    )
    band_hz = (float(args.band_hz[0]), float(args.band_hz[1]))
    candidates, _ = build_candidate_signals(
        pixels,
        ring_mean,
        fps=source_fps,
        band_hz=band_hz,
    )
    signal = candidates[str(args.signal)]
    peaks, polarity, _ = _event_train(signal, fps=source_fps, max_hz=band_hz[1])
    event_rate_hz = _event_rate_trace(peaks, frame_count=signal.size, fps=source_fps)
    frame_times = np.arange(signal.size, dtype=np.float64) / source_fps
    reference_per_frame = _reference_at_times(
        reference_hz, frame_times, float(args.reference_fps)
    )

    raw_residual = detrend(pixels, axis=0, type="linear")
    filtered_pixels = _bandpass(raw_residual, fps=source_fps, band_hz=band_hz)
    color_scale = max(float(np.quantile(np.abs(filtered_pixels), 0.995)), 1e-6)
    phase_scale = max(float(np.quantile(np.abs(signal), 0.99)), 1e-9)
    roi_height, roi_width = roi_xywh[3], roi_xywh[2]
    roi_scale = _roi_display_scale(roi_width, roi_height)
    reference_bpm = reference_per_frame * 60.0
    estimated_bpm = event_rate_hz * 60.0

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
    x, y, width, height = roi_xywh
    canvas_width, canvas_height = _CANVAS_SIZE
    try:
        for frame_index in range(signal.size):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"could not decode source frame {frame_index}")
            canvas = np.full((canvas_height, canvas_width, 3), 25, dtype=np.uint8)
            source_x, source_y = _SOURCE_ORIGIN
            canvas[
                source_y : source_y + source_height,
                source_x : source_x + source_width,
            ] = frame
            phase = float(
                np.clip(polarity * signal[frame_index] / phase_scale, -1.0, 1.0)
            )
            box_color = (40, 70, 255) if phase >= 0 else (255, 100, 40)
            cv2.rectangle(
                canvas,
                (source_x + x, source_y + y),
                (source_x + x + width - 1, source_y + y + height - 1),
                box_color,
                2,
                cv2.LINE_AA,
            )
            roi_frame = frame[y : y + height, x : x + width]
            raw_zoom = cv2.resize(
                roi_frame,
                (roi_width * roi_scale, roi_height * roi_scale),
                interpolation=cv2.INTER_NEAREST,
            )
            color_small = _color_pixel_change(
                filtered_pixels[frame_index].reshape(roi_height, roi_width),
                scale=color_scale,
            )
            color_zoom = cv2.resize(
                color_small,
                (roi_width * roi_scale, roi_height * roi_scale),
                interpolation=cv2.INTER_NEAREST,
            )
            rx, ry = _RAW_ROI_ORIGIN
            cx, cy = _COLOR_ROI_ORIGIN
            canvas[ry : ry + raw_zoom.shape[0], rx : rx + raw_zoom.shape[1]] = raw_zoom
            canvas[cy : cy + color_zoom.shape[0], cx : cx + color_zoom.shape[1]] = color_zoom
            cv2.rectangle(canvas, (rx, ry), (rx + raw_zoom.shape[1], ry + raw_zoom.shape[0]), (100, 105, 108), 1)
            cv2.rectangle(canvas, (cx, cy), (cx + color_zoom.shape[1], cy + color_zoom.shape[0]), (100, 105, 108), 1)

            _text(canvas, "source camera, native pixels", (16, 27), scale=0.55)
            _text(canvas, f"raw ROI, {roi_scale}x nearest", (rx, 27), scale=0.48)
            _text(canvas, "bandpassed pixel change", (cx, 27), scale=0.48)
            _draw_rate_graph(
                canvas,
                reference_bpm=reference_bpm,
                estimated_bpm=estimated_bpm,
                frame_index=frame_index,
            )
            _draw_waveform(
                canvas,
                signal=signal,
                frame_index=frame_index,
                fps=source_fps,
                polarity=polarity,
                signal_label=str(args.signal),
            )
            _text(
                canvas,
                f"source {frame_index:04d}/{signal.size - 1:04d}   t={frame_index / source_fps:06.3f} s",
                (16, 577),
                scale=0.52,
            )
            _text(
                canvas,
                f"reference {reference_bpm[frame_index]:6.1f} bpm",
                (16, 605),
                scale=0.52,
                color=(220, 220, 220),
            )
            estimate_text = (
                f"event interval {estimated_bpm[frame_index]:6.1f} bpm"
                if np.isfinite(estimated_bpm[frame_index])
                else "event interval pending"
            )
            _text(canvas, estimate_text, (260, 605), scale=0.52, color=(0, 170, 255))
            if duplicates[frame_index]:
                _text(canvas, "duplicate ROI frame", (16, 633), scale=0.45, color=(80, 180, 255))
            process.stdin.write(canvas.tobytes())
            if frame_index % 1000 == 0:
                print(f"rendered {frame_index}/{signal.size - 1}", flush=True)
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
        "source_video": str(video.resolve()),
        "roi_json": str(args.roi_json.resolve()),
        "workbook": str(args.workbook.resolve()),
        "trial_number": int(args.trial_number),
        "roi_xywh": list(roi_xywh),
        "source_fps": source_fps,
        "output_fps": float(args.output_fps),
        "playback_slowdown": source_fps / float(args.output_fps),
        "source_frames": int(signal.size),
        "event_count": int(peaks.size),
        "event_polarity": int(polarity),
        "signal": str(args.signal),
        "band_hz": list(band_hz),
        "roi_display_scale": int(roi_scale),
        "pixel_change_color_scale_995": color_scale,
        "output_video": str(output),
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
