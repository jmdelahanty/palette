from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import butter, detrend, sosfiltfilt

from analyze_segmented_cache_pca import (
    _common_valid_segments,
    _interpolate_segment,
    _read_mask_membership,
    _sample_rate,
    _segmented_events,
    _segmented_pca,
)
from extract_reliable_local_rostral_heartrate import load_dataset
from fisheye.analysis.local_rostral_heartrate import HeartrateConfig, build_risk_surfaces
from render_dynamic_heart_phase import _pixel_geometry, _scatter
from render_embedded_positive_control_overlay import (
    _color_pixel_change,
    _ffmpeg_process,
    _text,
)


_CANVAS_SIZE = (1152, 648)
_SOURCE_ORIGIN = (16, 44)
_SOURCE_SCALE = 2
_RAW_ORIGIN = (590, 44)
_COLOR_ORIGIN = (870, 44)
_LOCAL_SCALE = 12


def _raw_canonical_image(values: np.ndarray, geometry: dict[str, object]) -> np.ndarray:
    raster = _scatter(values, geometry)
    output = np.full((*raster.shape, 3), 38, dtype=np.uint8)
    finite = np.isfinite(raster)
    gray = np.nan_to_num(np.clip(raster, 0.0, 255.0), nan=0.0).astype(np.uint8)
    output[finite] = np.repeat(gray[..., None], 3, axis=2)[finite]
    return output


def _analysis_core_rows(
    segments: list[np.ndarray],
    timestamps_s: np.ndarray,
    *,
    edge_seconds: float,
    frame_count: int,
) -> np.ndarray:
    valid = np.zeros(frame_count, dtype=bool)
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    for rows in segments:
        local = timestamps[rows]
        keep = (
            (local >= local[0] + float(edge_seconds))
            & (local <= local[-1] - float(edge_seconds))
        )
        valid[rows[keep]] = True
    return valid


def _event_rate_trace(
    event_times_s: np.ndarray,
    timestamps_s: np.ndarray,
    *,
    band_hz: tuple[float, float],
) -> np.ndarray:
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    rate = np.full(timestamps.size, np.nan, dtype=np.float64)
    events = np.asarray(event_times_s, dtype=np.float64)
    for first, second in zip(events[:-1], events[1:]):
        interval = float(second - first)
        if not 1.0 / band_hz[1] <= interval <= 1.0 / band_hz[0]:
            continue
        inside = (timestamps >= first) & (timestamps < second)
        rate[inside] = 60.0 / interval
    return rate


def _window_candidate_trace(
    path: Path,
    timestamps_s: np.ndarray,
) -> np.ndarray:
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    relative = timestamps - float(timestamps[0])
    candidate = np.full(timestamps.size, np.nan, dtype=np.float64)
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"window CSV has no rows: {path}")
    previous_stop = -np.inf
    for row in rows:
        start = float(row["window_start_s"])
        stop = float(row["window_stop_s"])
        if start < previous_stop - 1e-9 or stop <= start:
            raise ValueError(f"window CSV has invalid or overlapping bounds: {path}")
        previous_stop = stop
        if row["status"] != "ok":
            continue
        value = float(row["candidate_cycles_per_min"])
        candidate[(relative >= start) & (relative < stop)] = value
    return candidate


def _bandpassed_raw_pixels(
    traces: np.ndarray,
    segments: list[np.ndarray],
    selected: np.ndarray,
    *,
    fps: float,
    band_hz: tuple[float, float],
) -> np.ndarray:
    source = np.asarray(traces, dtype=np.float64)[:, selected]
    output = np.full(source.shape, np.nan, dtype=np.float64)
    sos = butter(3, band_hz, btype="bandpass", fs=fps, output="sos")
    for rows in segments:
        filled = _interpolate_segment(source[rows])
        output[rows] = sosfiltfilt(sos, detrend(filled, axis=0, type="linear"), axis=0)
    return output


def _draw_source_overlay(
    canvas: np.ndarray,
    frame: np.ndarray,
    source_xy: np.ndarray,
    pixel_change: np.ndarray,
    *,
    color_scale: float,
    analysis_valid: bool,
) -> None:
    enlarged = cv2.resize(
        frame,
        (frame.shape[1] * _SOURCE_SCALE, frame.shape[0] * _SOURCE_SCALE),
        interpolation=cv2.INTER_NEAREST,
    )
    ox, oy = _SOURCE_ORIGIN
    canvas[oy : oy + enlarged.shape[0], ox : ox + enlarged.shape[1]] = enlarged
    finite = np.isfinite(source_xy).all(axis=1)
    points = source_xy[finite]
    if points.shape[0] >= 3:
        hull = cv2.convexHull(
            np.round(points * _SOURCE_SCALE + np.asarray([ox, oy])).astype(np.int32)
        )
        cv2.polylines(canvas, [hull], True, (0, 225, 255), 1, cv2.LINE_AA)
    if not analysis_valid:
        return
    color = _color_pixel_change(pixel_change.reshape(1, -1), scale=color_scale)[0]
    for point, value_color in zip(source_xy, color):
        if not np.isfinite(point).all():
            continue
        location = (
            int(round(ox + point[0] * _SOURCE_SCALE)),
            int(round(oy + point[1] * _SOURCE_SCALE)),
        )
        cv2.circle(
            canvas,
            location,
            2,
            tuple(int(item) for item in value_color),
            -1,
            cv2.LINE_AA,
        )


def _timeline_y(value_bpm: float, *, height: int) -> int:
    return int(
        round(
            (height - 1)
            - np.clip((float(value_bpm) - 80.0) / 160.0, 0.0, 1.0)
            * (height - 1)
        )
    )


def _timeline_panel(
    *,
    event_rate_bpm: np.ndarray,
    analysis_valid: np.ndarray,
    candidate_bpm: np.ndarray,
) -> np.ndarray:
    width, height = 541, 141
    panel = np.full((height, width, 3), 25, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (92, 98, 102), 1)
    for bpm in (100.0, 140.0, 180.0, 220.0):
        y = _timeline_y(bpm, height=height)
        cv2.line(panel, (0, y), (width - 1, y), (48, 53, 56), 1)
        _text(panel, f"{int(bpm)}", (4, y - 3), scale=0.32, color=(145, 150, 153))
    count = event_rate_bpm.size
    x = np.linspace(0, width - 1, count)
    for index in range(1, count):
        if (
            analysis_valid[index - 1]
            and analysis_valid[index]
            and np.isfinite(candidate_bpm[index - 1])
            and np.isfinite(candidate_bpm[index])
        ):
            cv2.line(
                panel,
                (int(round(x[index - 1])), _timeline_y(candidate_bpm[index - 1], height=height)),
                (int(round(x[index])), _timeline_y(candidate_bpm[index], height=height)),
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
        if np.isfinite(event_rate_bpm[index - 1]) and np.isfinite(event_rate_bpm[index]):
            cv2.line(
                panel,
                (int(round(x[index - 1])), _timeline_y(event_rate_bpm[index - 1], height=height)),
                (int(round(x[index])), _timeline_y(event_rate_bpm[index], height=height)),
                (0, 150, 255),
                1,
                cv2.LINE_AA,
            )
    return panel


def _draw_timeline(
    canvas: np.ndarray,
    panel: np.ndarray,
    *,
    frame_index: int,
    frame_count: int,
) -> None:
    x0, y0 = 590, 315
    height, width = panel.shape[:2]
    canvas[y0 : y0 + height, x0 : x0 + width] = panel
    cursor_x = int(round(x0 + frame_index / max(frame_count - 1, 1) * (width - 1)))
    cv2.line(canvas, (cursor_x, y0), (cursor_x, y0 + height - 1), (255, 220, 30), 2)
    _text(canvas, "candidate frequency / within-segment event intervals", (x0, y0 - 10), scale=0.42)


def _draw_waveform(
    canvas: np.ndarray,
    *,
    signal: np.ndarray,
    frame_index: int,
    fps: float,
    polarity: int,
) -> None:
    rect = (590, 510, 1130, 600)
    x0, y0, x1, y1 = rect
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (92, 98, 102), 1)
    half = int(round(1.5 * fps))
    start = max(0, frame_index - half)
    stop = min(signal.size, frame_index + half + 1)
    scale = max(float(np.nanquantile(np.abs(signal), 0.99)), 1e-9)
    x = np.linspace(x0, x1, 2 * half + 1)
    values = np.full(2 * half + 1, np.nan, dtype=np.float64)
    insert = half - (frame_index - start)
    values[insert : insert + stop - start] = polarity * signal[start:stop]
    y = (y0 + y1) / 2.0 - np.clip(values / scale, -1.0, 1.0) * (y1 - y0) * 0.45
    for index in range(1, values.size):
        if np.isfinite(values[index - 1]) and np.isfinite(values[index]):
            cv2.line(
                canvas,
                (int(round(x[index - 1])), int(round(y[index - 1]))),
                (int(round(x[index])), int(round(y[index]))),
                (0, 150, 255),
                2,
                cv2.LINE_AA,
            )
    center_x = int(round((x0 + x1) / 2.0))
    cv2.line(canvas, (center_x, y0), (center_x, y1), (255, 220, 30), 1)
    _text(canvas, "segmented PCA waveform: 3 second window", (x0, y0 - 9), scale=0.42)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the simple no-fold PCA on a moving-fish local-coordinate cache.")
    parser.add_argument("--dataset-npz", type=Path, required=True)
    parser.add_argument("--analysis-mask-npz", type=Path, required=True)
    parser.add_argument("--analysis-mask-key", default="heart_support_mask")
    parser.add_argument("--band-hz", type=float, nargs=2, default=(1.5, 4.0))
    parser.add_argument("--candidate-frequency-hz", type=float)
    parser.add_argument("--window-csv", type=Path)
    parser.add_argument("--minimum-segment-seconds", type=float, default=2.0)
    parser.add_argument("--max-interpolated-gap-seconds", type=float, default=0.02)
    parser.add_argument("--filter-edge-seconds", type=float, default=0.75)
    parser.add_argument("--output-fps", type=float, default=25.0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--ffmpeg", type=Path, default=Path("/usr/bin/ffmpeg"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.candidate_frequency_hz is None and args.window_csv is None:
        parser.error("provide --candidate-frequency-hz or --window-csv")
    if args.candidate_frequency_hz is not None and args.window_csv is not None:
        parser.error("provide only one of --candidate-frequency-hz or --window-csv")
    if int(args.frame_stride) < 1:
        parser.error("--frame-stride must be positive")
    if not 0 <= int(args.crf) <= 51:
        parser.error("--crf must be between 0 and 51")

    dataset = load_dataset(args.dataset_npz)
    metadata = dict(dataset.metadata)
    video = Path(str(metadata["crop_video"]))
    band_hz = (float(args.band_hz[0]), float(args.band_hz[1]))
    fps = _sample_rate(dataset.timestamps_s)
    config = HeartrateConfig(band_min_hz=band_hz[0], band_max_hz=band_hz[1]).validated()
    risks = build_risk_surfaces(dataset, config)
    membership = _read_mask_membership(
        args.analysis_mask_npz,
        args.analysis_mask_key,
        dataset.pixel_xy,
    )
    selected = np.asarray(risks.eligible, dtype=bool) & membership
    segments, interpolated = _common_valid_segments(
        dataset.timestamps_s,
        dataset.frame_valid,
        dataset.pixel_valid,
        selected,
        min_seconds=float(args.minimum_segment_seconds),
        max_interpolated_gap_seconds=float(args.max_interpolated_gap_seconds),
    )
    scores, _, _, explained = _segmented_pca(
        np.asarray(dataset.traces)[:, selected],
        segments,
        fps=fps,
        band_hz=band_hz,
    )
    polarity, event_times, intervals, interval_cv = _segmented_events(
        scores,
        segments,
        dataset.timestamps_s,
        fps=fps,
        band_hz=band_hz,
        edge_seconds=float(args.filter_edge_seconds),
    )
    signal = np.full(dataset.frame_count, np.nan, dtype=np.float64)
    for rows, score in zip(segments, scores):
        signal[rows] = score
    analysis_valid = _analysis_core_rows(
        segments,
        dataset.timestamps_s,
        edge_seconds=float(args.filter_edge_seconds),
        frame_count=dataset.frame_count,
    )
    event_rate = _event_rate_trace(event_times, dataset.timestamps_s, band_hz=band_hz)
    if args.window_csv is not None:
        candidate_bpm = _window_candidate_trace(args.window_csv, dataset.timestamps_s)
    else:
        candidate_bpm = np.full(
            dataset.frame_count,
            60.0 * float(args.candidate_frequency_hz),
            dtype=np.float64,
        )
    display_valid = analysis_valid & np.isfinite(candidate_bpm)
    event_rate[~display_valid] = np.nan
    display_signal = signal.copy()
    display_signal[~display_valid] = np.nan
    filtered_pixels = _bandpassed_raw_pixels(
        dataset.traces,
        segments,
        selected,
        fps=fps,
        band_hz=band_hz,
    )
    color_scale = max(
        float(np.nanquantile(np.abs(filtered_pixels[display_valid]), 0.995)),
        1e-6,
    )
    geometry = _pixel_geometry(dataset, margin_px=1)
    if not np.all(membership[selected]):
        raise RuntimeError("selected mask membership contract failed")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.rendering{output.suffix}")
    process = _ffmpeg_process(
        ffmpeg=args.ffmpeg,
        output=temporary,
        output_fps=float(args.output_fps),
        crf=int(args.crf),
    )
    if process.stdin is None:
        raise RuntimeError("FFmpeg stdin pipe was not created")
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        process.kill()
        raise ValueError(f"could not open source crop video: {video}")
    first_source_frame = int(dataset.frame_indices[0])
    capture.set(cv2.CAP_PROP_POS_FRAMES, first_source_frame)
    roi_height, roi_width = tuple(geometry["shape_hw"])
    raw_width, raw_height = roi_width * _LOCAL_SCALE, roi_height * _LOCAL_SCALE
    phase_scale = max(float(np.nanquantile(np.abs(display_signal), 0.99)), 1e-9)
    timeline_panel = _timeline_panel(
        event_rate_bpm=event_rate,
        analysis_valid=display_valid,
        candidate_bpm=candidate_bpm,
    )
    output_frame_count = 0
    try:
        for row in range(dataset.frame_count):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"could not decode source frame {first_source_frame + row}")
            if row % int(args.frame_stride) != 0:
                continue
            canvas = np.full((_CANVAS_SIZE[1], _CANVAS_SIZE[0], 3), 25, dtype=np.uint8)
            selected_source = np.asarray(dataset.source_xy[row], dtype=np.float64)[selected]
            change = filtered_pixels[row]
            _draw_source_overlay(
                canvas,
                frame,
                selected_source,
                np.nan_to_num(change, nan=0.0),
                color_scale=color_scale,
                analysis_valid=bool(display_valid[row]),
            )

            raw_small = _raw_canonical_image(np.asarray(dataset.traces[row]), geometry)
            raw_zoom = cv2.resize(
                raw_small,
                (raw_width, raw_height),
                interpolation=cv2.INTER_NEAREST,
            )
            color_values = np.full(dataset.pixel_count, np.nan, dtype=np.float64)
            if display_valid[row]:
                color_values[selected] = change
            color_raster = _scatter(color_values, geometry)
            color_small = np.full((*color_raster.shape, 3), 38, dtype=np.uint8)
            finite = np.isfinite(color_raster)
            if np.any(finite):
                mapped = _color_pixel_change(
                    np.nan_to_num(color_raster, nan=0.0),
                    scale=color_scale,
                )
                color_small[finite] = mapped[finite]
            color_zoom = cv2.resize(
                color_small,
                (raw_width, raw_height),
                interpolation=cv2.INTER_NEAREST,
            )
            rx, ry = _RAW_ORIGIN
            cx, cy = _COLOR_ORIGIN
            canvas[ry : ry + raw_height, rx : rx + raw_width] = raw_zoom
            canvas[cy : cy + raw_height, cx : cx + raw_width] = color_zoom
            cv2.rectangle(canvas, (rx, ry), (rx + raw_width, ry + raw_height), (100, 105, 108), 1)
            cv2.rectangle(canvas, (cx, cy), (cx + raw_width, cy + raw_height), (100, 105, 108), 1)

            phase = (
                float(np.clip(polarity * signal[row] / phase_scale, -1.0, 1.0))
                if np.isfinite(display_signal[row]) and display_valid[row]
                else float("nan")
            )
            _text(canvas, "source crop, 2x nearest", (16, 27), scale=0.52)
            _text(canvas, "raw canonical ROI", (rx, 27), scale=0.46)
            _text(canvas, "38-pixel band change", (cx, 27), scale=0.46)
            _draw_timeline(
                canvas,
                timeline_panel,
                frame_index=row,
                frame_count=dataset.frame_count,
            )
            _draw_waveform(
                canvas,
                signal=display_signal,
                frame_index=row,
                fps=fps,
                polarity=polarity,
            )
            relative_time = float(dataset.timestamps_s[row] - dataset.timestamps_s[0])
            _text(
                canvas,
                f"source frame {int(dataset.frame_indices[row])}   t={relative_time:06.3f} s",
                (16, 582),
                scale=0.50,
            )
            status = "analysis-valid" if display_valid[row] else "no analysis color"
            _text(
                canvas,
                status,
                (16, 611),
                scale=0.48,
                color=(230, 230, 230) if display_valid[row] else (150, 150, 150),
            )
            candidate_text = (
                f"candidate {candidate_bpm[row]:.0f} cycles/min"
                if np.isfinite(candidate_bpm[row])
                else "no reliable window candidate"
            )
            _text(
                canvas,
                candidate_text,
                (190, 611),
                scale=0.48,
                color=(210, 210, 210),
            )
            if np.isfinite(event_rate[row]):
                _text(
                    canvas,
                    f"event interval {event_rate[row]:.1f}/min",
                    (16, 637),
                    scale=0.48,
                    color=(0, 170, 255),
                )
            if np.isfinite(phase):
                phase_color = (40, 70, 255) if phase >= 0 else (255, 100, 40)
                cv2.circle(canvas, (548, 628), 8, phase_color, -1, cv2.LINE_AA)
            process.stdin.write(canvas.tobytes())
            output_frame_count += 1
            if row % 5000 == 0:
                print(f"rendered {row}/{dataset.frame_count - 1}", flush=True)
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

    metadata_out = {
        "analysis_status": "exploratory_no_fold_same_cache_pca_visualization",
        "dataset_npz": str(args.dataset_npz.resolve()),
        "source_video": str(video.resolve()),
        "source_frame_start": first_source_frame,
        "source_frames": dataset.frame_count,
        "source_fps": fps,
        "output_fps": float(args.output_fps),
        "frame_stride": int(args.frame_stride),
        "output_frames": output_frame_count,
        "playback_slowdown": fps / (float(args.output_fps) * int(args.frame_stride)),
        "analysis_mask_npz": str(args.analysis_mask_npz.resolve()),
        "analysis_mask_key": str(args.analysis_mask_key),
        "selected_pixels": int(np.count_nonzero(selected)),
        "candidate_frequency_hz": (
            float(args.candidate_frequency_hz)
            if args.candidate_frequency_hz is not None
            else None
        ),
        "window_csv": (
            str(args.window_csv.resolve()) if args.window_csv is not None else None
        ),
        "segment_count": len(segments),
        "analysis_valid_frames": int(np.count_nonzero(analysis_valid)),
        "display_valid_frames": int(np.count_nonzero(display_valid)),
        "bounded_interpolated_gap_frames": int(np.count_nonzero(interpolated)),
        "event_count": int(event_times.size),
        "within_segment_interval_count": int(intervals.size),
        "interval_cv": interval_cv,
        "event_polarity": polarity,
        "pca_explained_variance_fraction": explained,
        "pixel_change_color_scale_995": color_scale,
        "folds_used": False,
        "crf": int(args.crf),
        "output_video": str(output),
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata_out, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metadata_out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
