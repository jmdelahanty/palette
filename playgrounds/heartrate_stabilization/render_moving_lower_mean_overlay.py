from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from analyze_segmented_cache_pca import (
    _common_valid_segments,
    _read_mask_membership,
    _sample_rate,
)
from extract_reliable_local_rostral_heartrate import load_dataset
from fisheye.analysis.local_rostral_heartrate import HeartrateConfig, build_risk_surfaces
from render_dynamic_heart_phase import _pixel_geometry, _scatter
from render_embedded_positive_control_overlay import (
    _color_pixel_change,
    _ffmpeg_process,
    _text,
)
from render_segmented_cache_pca_overlay import (
    _analysis_core_rows,
    _event_rate_trace,
    _raw_canonical_image,
)


_CANVAS_SIZE = (1152, 648)
_SOURCE_ORIGIN = (16, 44)
_SOURCE_SCALE = 2
_RAW_ORIGIN = (590, 44)
_COLOR_ORIGIN = (870, 44)
_LOCAL_SCALE = 12


def _window_candidate_trace(
    path: Path,
    timestamps_s: np.ndarray,
    *,
    method: str,
    window_seconds: float,
) -> np.ndarray:
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    relative = timestamps - float(timestamps[0])
    candidate = np.full(timestamps.size, np.nan, dtype=np.float64)
    with Path(path).open(newline="") as handle:
        matching = [
            row
            for row in csv.DictReader(handle)
            if row["method"] == method
            and np.isclose(float(row["window_seconds"]), float(window_seconds))
        ]
    if not matching:
        raise ValueError(
            f"window CSV has no {method!r} rows for {window_seconds:g} seconds: {path}"
        )
    previous_stop = -np.inf
    for row in matching:
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


def _timeline_y(value_bpm: float, *, height: int) -> int:
    return int(
        round(
            (height - 1)
            - np.clip((float(value_bpm) - 100.0) / 140.0, 0.0, 1.0)
            * (height - 1)
        )
    )


def _timeline_panel(
    *,
    event_rate_bpm: np.ndarray,
    analysis_valid: np.ndarray,
    candidate_4s_bpm: np.ndarray,
    candidate_8s_bpm: np.ndarray,
) -> np.ndarray:
    width, height = 541, 141
    panel = np.full((height, width, 3), 25, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (92, 98, 102), 1)
    for bpm in (120.0, 150.0, 180.0, 210.0, 240.0):
        y = _timeline_y(bpm, height=height)
        cv2.line(panel, (0, y), (width - 1, y), (48, 53, 56), 1)
        _text(panel, f"{int(bpm)}", (4, y - 3), scale=0.32, color=(145, 150, 153))
    x = np.linspace(0, width - 1, event_rate_bpm.size)
    series = (
        (candidate_8s_bpm, (80, 215, 120), 2),
        (candidate_4s_bpm, (225, 225, 225), 1),
        (event_rate_bpm, (0, 150, 255), 1),
    )
    for values, color, thickness in series:
        for index in range(1, values.size):
            if not (
                analysis_valid[index - 1]
                and analysis_valid[index]
                and np.isfinite(values[index - 1])
                and np.isfinite(values[index])
            ):
                continue
            cv2.line(
                panel,
                (int(round(x[index - 1])), _timeline_y(values[index - 1], height=height)),
                (int(round(x[index])), _timeline_y(values[index], height=height)),
                color,
                thickness,
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
    _text(canvas, "lower mean: 4 s white, 8 s green, intervals orange", (x0, y0 - 10), scale=0.42)


def _draw_waveform(
    canvas: np.ndarray,
    *,
    signal: np.ndarray,
    frame_index: int,
    fps: float,
    polarity: int,
) -> None:
    x0, y0, x1, y1 = (590, 510, 1130, 600)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (92, 98, 102), 1)
    half = int(round(1.5 * fps))
    start = max(0, frame_index - half)
    stop = min(signal.size, frame_index + half + 1)
    finite = np.abs(signal[np.isfinite(signal)])
    scale = max(float(np.quantile(finite, 0.99)) if finite.size else 0.0, 1e-9)
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
    _text(canvas, "20-pixel lower equal-mean waveform: 3 second window", (x0, y0 - 9), scale=0.42)


def _draw_source_overlay(
    canvas: np.ndarray,
    frame: np.ndarray,
    source_xy: np.ndarray,
    *,
    scalar_change: float,
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
    color = _color_pixel_change(
        np.asarray([[scalar_change]], dtype=np.float64), scale=color_scale
    )[0, 0]
    for point in source_xy:
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
            tuple(int(item) for item in color),
            -1,
            cv2.LINE_AA,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the saved lower frozen-mask equal-mean oscillator over a moving-fish cache."
    )
    parser.add_argument("--dataset-npz", type=Path, required=True)
    parser.add_argument("--full-mask-npz", type=Path, required=True)
    parser.add_argument("--full-mask-key", default="original_38_mask")
    parser.add_argument("--lower-mask-npz", type=Path, required=True)
    parser.add_argument("--lower-mask-key", default="lower_mask")
    parser.add_argument("--comparison-arrays", type=Path, required=True)
    parser.add_argument("--comparison-summary", type=Path, required=True)
    parser.add_argument("--window-csv", type=Path, required=True)
    parser.add_argument("--method", default="lower_equal_mean")
    parser.add_argument("--band-hz", type=float, nargs=2, default=(2.0, 4.0))
    parser.add_argument("--minimum-segment-seconds", type=float, default=2.0)
    parser.add_argument("--max-interpolated-gap-seconds", type=float, default=0.02)
    parser.add_argument("--filter-edge-seconds", type=float, default=0.75)
    parser.add_argument("--output-fps", type=float, default=25.0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--ffmpeg-preset", default="medium")
    parser.add_argument("--ffmpeg", type=Path, default=Path("/usr/bin/ffmpeg"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if int(args.frame_stride) < 1:
        parser.error("--frame-stride must be positive")
    if not 0 <= int(args.crf) <= 51:
        parser.error("--crf must be between 0 and 51")

    dataset = load_dataset(args.dataset_npz)
    metadata = dict(dataset.metadata)
    video = Path(str(metadata["crop_video"]))
    fps = _sample_rate(dataset.timestamps_s)
    band_hz = (float(args.band_hz[0]), float(args.band_hz[1]))
    config = HeartrateConfig(band_min_hz=band_hz[0], band_max_hz=band_hz[1]).validated()
    eligible = np.asarray(build_risk_surfaces(dataset, config).eligible, dtype=bool)
    full_membership = _read_mask_membership(
        args.full_mask_npz, args.full_mask_key, dataset.pixel_xy
    )
    lower_membership = _read_mask_membership(
        args.lower_mask_npz, args.lower_mask_key, dataset.pixel_xy
    )
    full_selected = eligible & full_membership
    lower_selected = eligible & lower_membership
    if np.any(lower_selected & ~full_selected):
        raise ValueError("lower mask is not a subset of the frozen full mask")

    segments, interpolated = _common_valid_segments(
        dataset.timestamps_s,
        dataset.frame_valid,
        dataset.pixel_valid,
        full_selected,
        min_seconds=float(args.minimum_segment_seconds),
        max_interpolated_gap_seconds=float(args.max_interpolated_gap_seconds),
    )
    analysis_valid = _analysis_core_rows(
        segments,
        dataset.timestamps_s,
        edge_seconds=float(args.filter_edge_seconds),
        frame_count=dataset.frame_count,
    )

    with np.load(args.comparison_arrays) as arrays:
        saved_lower = np.asarray(arrays["lower_membership"], dtype=bool)
        signal = np.asarray(arrays[f"trace_{args.method}"], dtype=np.float64)
        event_times = np.asarray(arrays[f"event_times_s_{args.method}"], dtype=np.float64)
    if not np.array_equal(saved_lower, lower_selected):
        raise ValueError("saved comparison lower membership differs from the requested lower mask")
    if signal.shape != (dataset.frame_count,):
        raise ValueError("saved lower trace length differs from the dataset frame count")
    summary = json.loads(args.comparison_summary.read_text())
    method_summary = summary["summaries"][args.method]
    polarity = int(method_summary["event_polarity"])
    expected_count = int(summary["pixel_counts"]["lower"])
    if int(np.count_nonzero(lower_selected)) != expected_count:
        raise ValueError("lower-mask pixel count differs from the saved comparison summary")

    candidate_4s = _window_candidate_trace(
        args.window_csv,
        dataset.timestamps_s,
        method=args.method,
        window_seconds=4.0,
    )
    candidate_8s = _window_candidate_trace(
        args.window_csv,
        dataset.timestamps_s,
        method=args.method,
        window_seconds=8.0,
    )
    event_rate = _event_rate_trace(event_times, dataset.timestamps_s, band_hz=band_hz)
    event_rate[~analysis_valid] = np.nan
    display_signal = signal.copy()
    display_signal[~analysis_valid] = np.nan
    finite_signal = np.abs(display_signal[np.isfinite(display_signal)])
    color_scale = max(
        float(np.quantile(finite_signal, 0.995)) if finite_signal.size else 0.0,
        1e-9,
    )
    timeline = _timeline_panel(
        event_rate_bpm=event_rate,
        analysis_valid=analysis_valid,
        candidate_4s_bpm=candidate_4s,
        candidate_8s_bpm=candidate_8s,
    )
    geometry = _pixel_geometry(dataset, margin_px=1)

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.rendering{output.suffix}")
    process = _ffmpeg_process(
        ffmpeg=args.ffmpeg,
        output=temporary,
        output_fps=float(args.output_fps),
        crf=int(args.crf),
        preset=str(args.ffmpeg_preset),
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
    local_width, local_height = roi_width * _LOCAL_SCALE, roi_height * _LOCAL_SCALE
    output_frame_count = 0
    try:
        for row in range(dataset.frame_count):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"could not decode source frame {first_source_frame + row}")
            if row % int(args.frame_stride) != 0:
                continue
            valid = bool(analysis_valid[row] and np.isfinite(signal[row]))
            canvas = np.full((_CANVAS_SIZE[1], _CANVAS_SIZE[0], 3), 25, dtype=np.uint8)
            lower_source = np.asarray(dataset.source_xy[row], dtype=np.float64)[lower_selected]
            _draw_source_overlay(
                canvas,
                frame,
                lower_source,
                scalar_change=float(np.nan_to_num(signal[row], nan=0.0)),
                color_scale=color_scale,
                analysis_valid=valid,
            )

            raw_small = _raw_canonical_image(np.asarray(dataset.traces[row]), geometry)
            raw_zoom = cv2.resize(
                raw_small, (local_width, local_height), interpolation=cv2.INTER_NEAREST
            )
            color_values = np.full(dataset.pixel_count, np.nan, dtype=np.float64)
            if valid:
                color_values[lower_selected] = signal[row]
            color_raster = _scatter(color_values, geometry)
            color_small = np.full((*color_raster.shape, 3), 38, dtype=np.uint8)
            finite = np.isfinite(color_raster)
            if np.any(finite):
                mapped = _color_pixel_change(
                    np.nan_to_num(color_raster, nan=0.0), scale=color_scale
                )
                color_small[finite] = mapped[finite]
            color_zoom = cv2.resize(
                color_small, (local_width, local_height), interpolation=cv2.INTER_NEAREST
            )
            rx, ry = _RAW_ORIGIN
            cx, cy = _COLOR_ORIGIN
            canvas[ry : ry + local_height, rx : rx + local_width] = raw_zoom
            canvas[cy : cy + local_height, cx : cx + local_width] = color_zoom
            cv2.rectangle(canvas, (rx, ry), (rx + local_width, ry + local_height), (100, 105, 108), 1)
            cv2.rectangle(canvas, (cx, cy), (cx + local_width, cy + local_height), (100, 105, 108), 1)

            _text(canvas, "source crop: lower 20-pixel mask", (16, 27), scale=0.52)
            _text(canvas, "raw canonical ROI", (rx, 27), scale=0.46)
            _text(canvas, "uniform lower mean", (cx, 27), scale=0.46)
            _draw_timeline(canvas, timeline, frame_index=row, frame_count=dataset.frame_count)
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
                f"source frame {int(dataset.frame_indices[row])}   t={relative_time:07.3f} s",
                (16, 582),
                scale=0.50,
            )
            _text(
                canvas,
                "analysis-valid" if valid else "invalid/gap edge: color hidden",
                (16, 611),
                scale=0.48,
                color=(230, 230, 230) if valid else (150, 150, 150),
            )
            if np.isfinite(candidate_4s[row]):
                _text(canvas, f"4 s {candidate_4s[row]:.0f}/min", (270, 611), scale=0.46)
            if np.isfinite(candidate_8s[row]):
                _text(
                    canvas,
                    f"8 s {candidate_8s[row]:.0f}/min",
                    (405, 611),
                    scale=0.46,
                    color=(80, 215, 120),
                )
            if np.isfinite(event_rate[row]):
                _text(
                    canvas,
                    f"raw interval {event_rate[row]:.1f}/min (not validated beats)",
                    (16, 637),
                    scale=0.46,
                    color=(0, 170, 255),
                )
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
        "analysis_status": "exploratory_moving_fish_lower_equal_mean_visualization",
        "interpretation": "Candidate oscillator visualization; not validated heart rate or beat timing.",
        "dataset_npz": str(args.dataset_npz.resolve()),
        "source_video": str(video.resolve()),
        "source_frame_start": first_source_frame,
        "source_frames": dataset.frame_count,
        "source_fps": fps,
        "output_fps": float(args.output_fps),
        "frame_stride": int(args.frame_stride),
        "output_frames": output_frame_count,
        "playback_speed_relative_to_source": (
            float(args.output_fps) * int(args.frame_stride) / fps
        ),
        "full_mask_npz": str(args.full_mask_npz.resolve()),
        "full_mask_key": args.full_mask_key,
        "lower_mask_npz": str(args.lower_mask_npz.resolve()),
        "lower_mask_key": args.lower_mask_key,
        "full_mask_pixels": int(np.count_nonzero(full_selected)),
        "lower_mask_pixels": int(np.count_nonzero(lower_selected)),
        "projection": args.method,
        "comparison_arrays": str(args.comparison_arrays.resolve()),
        "comparison_summary": str(args.comparison_summary.resolve()),
        "window_csv": str(args.window_csv.resolve()),
        "band_hz": list(band_hz),
        "segment_count": len(segments),
        "analysis_valid_frames": int(np.count_nonzero(analysis_valid)),
        "bounded_interpolated_gap_frames": int(np.count_nonzero(interpolated)),
        "event_count": int(event_times.size),
        "event_polarity": polarity,
        "signal_color_scale_995": color_scale,
        "uniform_color_within_lower_mask": True,
        "crf": int(args.crf),
        "ffmpeg_preset": str(args.ffmpeg_preset),
        "output_video": str(output),
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata_out, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metadata_out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
