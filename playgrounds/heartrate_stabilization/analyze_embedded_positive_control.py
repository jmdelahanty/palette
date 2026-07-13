from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import zipfile
import xml.etree.ElementTree as ET

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, detrend, find_peaks, sosfiltfilt, welch
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits


_SHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


@dataclass(frozen=True)
class CandidateSummary:
    name: str
    peak_hz: float
    peak_bpm: float
    spectral_peak_to_median: float
    ridge_correlation: float
    ridge_mae_bpm: float
    event_count: int
    event_rate_mae_bpm: float
    event_interval_cv: float
    event_polarity: int


def _column_index(cell_reference: str) -> int:
    letters = re.match(r"[A-Z]+", str(cell_reference))
    if letters is None:
        raise ValueError(f"invalid XLSX cell reference: {cell_reference}")
    index = 0
    for letter in letters.group(0):
        index = index * 26 + ord(letter) - ord("A") + 1
    return index - 1


def _worksheet_path(archive: zipfile.ZipFile, sheet_name: str) -> str:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    sheet = next(
        (
            item
            for item in workbook.findall(f".//{{{_SHEET_NS}}}sheet")
            if item.get("name") == sheet_name
        ),
        None,
    )
    if sheet is None:
        raise ValueError(f"XLSX has no sheet named {sheet_name!r}")
    relationship_id = sheet.get(f"{{{_REL_NS}}}id")
    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    relationship = next(
        (
            item
            for item in relationships.findall(f"{{{_PACKAGE_REL_NS}}}Relationship")
            if item.get("Id") == relationship_id
        ),
        None,
    )
    if relationship is None:
        raise ValueError(f"could not resolve worksheet relationship {relationship_id}")
    target = str(relationship.get("Target"))
    return target.lstrip("/") if target.startswith("xl/") else f"xl/{target}"


def read_numeric_xlsx_row(
    workbook_path: Path,
    *,
    sheet_name: str,
    row_number: int,
) -> np.ndarray:
    with zipfile.ZipFile(workbook_path) as archive:
        worksheet = ET.fromstring(archive.read(_worksheet_path(archive, sheet_name)))
    row = next(
        (
            item
            for item in worksheet.findall(f".//{{{_SHEET_NS}}}row")
            if int(item.get("r", "-1")) == int(row_number)
        ),
        None,
    )
    if row is None:
        raise ValueError(f"sheet {sheet_name!r} has no row {row_number}")
    cells = row.findall(f"{{{_SHEET_NS}}}c")
    if not cells:
        raise ValueError(f"sheet {sheet_name!r} row {row_number} is empty")
    width = max(_column_index(str(cell.get("r"))) for cell in cells) + 1
    values = np.full(width, np.nan, dtype=np.float64)
    for cell in cells:
        value = cell.find(f"{{{_SHEET_NS}}}v")
        if value is not None and value.text is not None:
            values[_column_index(str(cell.get("r")))] = float(value.text)
    if not np.isfinite(values).all():
        raise ValueError(f"sheet {sheet_name!r} row {row_number} is not fully numeric")
    return values


def _load_roi(path: Path) -> tuple[Path, tuple[int, int, int, int]]:
    payload = json.loads(path.read_text())
    if payload.get("schema_id") != "palette.playground.fixed_video_roi.v1":
        raise ValueError("unsupported ROI JSON schema")
    roi = tuple(int(value) for value in payload["roi_xywh"])
    if len(roi) != 4:
        raise ValueError("roi_xywh must have four values")
    return Path(payload["source_video"]), roi


def _extract_video(
    video: Path,
    roi_xywh: tuple[int, int, int, int],
    *,
    ring_margin: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, tuple[int, int]]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    expected_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    frame_width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    frame_height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    x, y, width, height = roi_xywh
    if x < 0 or y < 0 or x + width > frame_width or y + height > frame_height:
        capture.release()
        raise ValueError(f"ROI {roi_xywh} exceeds video frame bounds")
    outer_x0 = max(0, x - ring_margin)
    outer_y0 = max(0, y - ring_margin)
    outer_x1 = min(frame_width, x + width + ring_margin)
    outer_y1 = min(frame_height, y + height + ring_margin)
    ring_mask = np.ones((outer_y1 - outer_y0, outer_x1 - outer_x0), dtype=bool)
    ring_mask[y - outer_y0 : y + height - outer_y0, x - outer_x0 : x + width - outer_x0] = False

    pixels: list[np.ndarray] = []
    ring_means: list[float] = []
    duplicate: list[bool] = []
    first_frame: np.ndarray | None = None
    previous: np.ndarray | None = None
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if first_frame is None:
            first_frame = frame.copy()
        current = gray[y : y + height, x : x + width].reshape(-1).copy()
        outer = gray[outer_y0:outer_y1, outer_x0:outer_x1]
        pixels.append(current)
        ring_means.append(float(np.mean(outer[ring_mask])))
        duplicate.append(previous is not None and np.array_equal(current, previous))
        previous = current
    capture.release()
    if first_frame is None or len(pixels) < 2:
        raise ValueError(f"video yielded too few frames: {video}")
    if expected_count > 0 and abs(len(pixels) - expected_count) > 1:
        raise ValueError(f"decoded {len(pixels)} frames but container reports {expected_count}")
    frame_indices = np.arange(len(pixels), dtype=np.int64)
    return (
        np.asarray(pixels, dtype=np.float64),
        np.asarray(ring_means, dtype=np.float64),
        np.asarray(duplicate, dtype=bool),
        first_frame,
        fps,
        (frame_height, frame_width),
    )


def _bandpass(values: np.ndarray, *, fps: float, band_hz: tuple[float, float]) -> np.ndarray:
    sos = butter(3, band_hz, btype="bandpass", fs=fps, output="sos")
    return sosfiltfilt(sos, np.asarray(values, dtype=np.float64), axis=0)


def _robust_standardize_pixels(pixels: np.ndarray) -> np.ndarray:
    values = detrend(np.asarray(pixels, dtype=np.float64), axis=0, type="linear")
    median = np.median(values, axis=0)
    scale = 1.4826 * np.median(np.abs(values - median), axis=0)
    usable = scale > 1e-6
    if np.count_nonzero(usable) < 3:
        raise ValueError("fewer than three varying pixels in ROI")
    return (values[:, usable] - median[usable]) / scale[usable]


def build_candidate_signals(
    pixels: np.ndarray,
    ring_mean: np.ndarray,
    *,
    fps: float,
    band_hz: tuple[float, float],
) -> tuple[dict[str, np.ndarray], float]:
    roi_mean = np.mean(pixels, axis=1)
    log_ratio = np.log(np.maximum(roi_mean, 1.0)) - np.log(np.maximum(ring_mean, 1.0))
    standardized = _robust_standardize_pixels(pixels)
    filtered_pixels = _bandpass(standardized, fps=fps, band_hz=band_hz)
    pca = PCA(n_components=1, svd_solver="randomized", random_state=0)
    with threadpool_limits(limits=1):
        band_pca = pca.fit_transform(filtered_pixels).reshape(-1)
    candidates = {
        "roi_mean": _bandpass(detrend(roi_mean), fps=fps, band_hz=band_hz),
        "roi_ring_log_ratio": _bandpass(
            detrend(log_ratio), fps=fps, band_hz=band_hz
        ),
        "band_pca1": band_pca,
    }
    return candidates, float(pca.explained_variance_ratio_[0])


def _spectrum(
    signal: np.ndarray,
    *,
    fps: float,
    band_hz: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, float, float]:
    nperseg = min(len(signal), max(256, int(round(8.0 * fps))))
    frequencies, power = welch(signal, fs=fps, nperseg=nperseg)
    target = (frequencies >= band_hz[0]) & (frequencies <= band_hz[1])
    target_indices = np.flatnonzero(target)
    peak_index = int(target_indices[np.argmax(power[target])])
    peak_hz = float(frequencies[peak_index])
    ratio = float(power[peak_index] / max(np.median(power[target]), np.finfo(float).tiny))
    return frequencies, power, peak_hz, ratio


def _ridge(
    signal: np.ndarray,
    *,
    fps: float,
    band_hz: tuple[float, float],
    window_seconds: float,
    step_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    window = int(round(window_seconds * fps))
    step = int(round(step_seconds * fps))
    nfft = max(8192, 1 << (window - 1).bit_length())
    centers: list[float] = []
    peaks: list[float] = []
    for start in range(0, len(signal) - window + 1, step):
        frequencies, power = welch(
            signal[start : start + window], fs=fps, nperseg=window, nfft=nfft
        )
        target = (frequencies >= band_hz[0]) & (frequencies <= band_hz[1])
        peaks.append(float(frequencies[target][np.argmax(power[target])]))
        centers.append((start + (window - 1) / 2.0) / fps)
    return np.asarray(centers), np.asarray(peaks)


def _event_train(signal: np.ndarray, *, fps: float, max_hz: float) -> tuple[np.ndarray, int, float]:
    standardized = signal / max(1.4826 * np.median(np.abs(signal - np.median(signal))), 1e-9)
    distance = max(1, int(np.floor(fps / max_hz)))
    options: list[tuple[float, np.ndarray, int, float]] = []
    for polarity in (1, -1):
        peaks, properties = find_peaks(
            polarity * standardized,
            distance=distance,
            prominence=0.5,
        )
        intervals = np.diff(peaks) / fps
        plausible = intervals[(intervals >= 1.0 / max_hz) & (intervals <= 1.0 / 1.5)]
        cv = float(np.std(plausible) / np.mean(plausible)) if plausible.size >= 2 else np.inf
        prominence = float(np.median(properties["prominences"])) if peaks.size else 0.0
        options.append((cv - 0.01 * prominence, peaks, polarity, cv))
    _, peaks, polarity, cv = min(options, key=lambda item: item[0])
    return peaks, polarity, cv


def _reference_at_times(reference_hz: np.ndarray, times: np.ndarray, reference_fps: float) -> np.ndarray:
    reference_time = np.arange(reference_hz.size, dtype=np.float64) / reference_fps
    return np.interp(times, reference_time, reference_hz)


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or np.std(left) <= 0 or np.std(right) <= 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _offset_controls(
    roi_xywh: tuple[int, int, int, int],
    *,
    frame_shape: tuple[int, int],
    offset_pixels: int,
) -> dict[str, tuple[int, int, int, int]]:
    x, y, width, height = roi_xywh
    candidates = {
        "offset_left": (x - offset_pixels, y, width, height),
        "offset_right": (x + offset_pixels, y, width, height),
        "offset_anterior": (x, y - offset_pixels, width, height),
        "offset_posterior": (x, y + offset_pixels, width, height),
    }
    frame_height, frame_width = frame_shape
    return {
        name: roi
        for name, roi in candidates.items()
        if roi[0] >= 0
        and roi[1] >= 0
        and roi[0] + roi[2] <= frame_width
        and roi[1] + roi[3] <= frame_height
    }


def analyze_candidate(
    name: str,
    signal: np.ndarray,
    *,
    fps: float,
    band_hz: tuple[float, float],
    reference_hz: np.ndarray,
    reference_fps: float,
) -> tuple[CandidateSummary, dict[str, np.ndarray]]:
    frequencies, power, peak_hz, ratio = _spectrum(signal, fps=fps, band_hz=band_hz)
    ridge_time, ridge_hz = _ridge(
        signal,
        fps=fps,
        band_hz=band_hz,
        window_seconds=8.0,
        step_seconds=1.0,
    )
    ridge_reference = _reference_at_times(reference_hz, ridge_time, reference_fps)
    peaks, polarity, interval_cv = _event_train(signal, fps=fps, max_hz=band_hz[1])
    event_time = peaks[1:].astype(np.float64) / fps
    event_hz = fps / np.diff(peaks) if peaks.size >= 2 else np.empty(0)
    event_reference = _reference_at_times(reference_hz, event_time, reference_fps)
    summary = CandidateSummary(
        name=name,
        peak_hz=peak_hz,
        peak_bpm=60.0 * peak_hz,
        spectral_peak_to_median=ratio,
        ridge_correlation=_safe_correlation(ridge_hz, ridge_reference),
        ridge_mae_bpm=float(60.0 * np.mean(np.abs(ridge_hz - ridge_reference))),
        event_count=int(peaks.size),
        event_rate_mae_bpm=(
            float(60.0 * np.mean(np.abs(event_hz - event_reference)))
            if event_hz.size
            else float("nan")
        ),
        event_interval_cv=interval_cv,
        event_polarity=polarity,
    )
    return summary, {
        "frequency_hz": frequencies,
        "power": power,
        "ridge_time_s": ridge_time,
        "ridge_hz": ridge_hz,
        "event_time_s": event_time,
        "event_hz": event_hz,
    }


def _plot(
    output: Path,
    *,
    frame: np.ndarray,
    roi_xywh: tuple[int, int, int, int],
    fps: float,
    candidates: dict[str, np.ndarray],
    details: dict[str, dict[str, np.ndarray]],
    reference_hz: np.ndarray,
    reference_fps: float,
    band_hz: tuple[float, float],
) -> None:
    colors = {"roi_mean": "#0072B2", "roi_ring_log_ratio": "#009E73", "band_pca1": "#D55E00"}
    figure, axes = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
    preview = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    x, y, width, height = roi_xywh
    axes[0, 0].imshow(preview)
    axes[0, 0].add_patch(plt.Rectangle((x, y), width, height, fill=False, color="red", linewidth=2))
    axes[0, 0].set_title("Trial 1 top-camera ROI")
    axes[0, 0].axis("off")

    reference_time = np.arange(reference_hz.size) / reference_fps
    axes[0, 1].plot(reference_time, 60.0 * reference_hz, color="#333333", linewidth=1.5)
    axes[0, 1].set(title="Supplied rate reference", xlabel="Time (s)", ylabel="Rate (bpm)")

    for name, detail in details.items():
        target = (detail["frequency_hz"] >= band_hz[0]) & (detail["frequency_hz"] <= band_hz[1])
        power = detail["power"][target]
        axes[1, 0].plot(
            detail["frequency_hz"][target] * 60.0,
            power / max(float(np.max(power)), np.finfo(float).tiny),
            label=name,
            color=colors[name],
        )
        axes[1, 1].plot(
            detail["ridge_time_s"],
            detail["ridge_hz"] * 60.0,
            label=name,
            color=colors[name],
        )
    axes[1, 0].axvline(60.0 * float(np.median(reference_hz)), color="#333333", linestyle="--", label="reference median")
    axes[1, 0].set(title="Normalized Welch spectra", xlabel="Frequency (bpm)", ylabel="Relative power")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].plot(reference_time, 60.0 * reference_hz, color="#333333", linewidth=2, label="reference")
    axes[1, 1].set(title="Reference-blind 8 s spectral ridge", xlabel="Time (s)", ylabel="Rate (bpm)")
    axes[1, 1].legend(fontsize=8)

    segment = slice(0, min(len(next(iter(candidates.values()))), int(round(8.0 * fps))))
    time = np.arange(len(next(iter(candidates.values())))) / fps
    for name, signal in candidates.items():
        scale = max(float(np.std(signal)), np.finfo(float).tiny)
        axes[2, 0].plot(time[segment], signal[segment] / scale, label=name, color=colors[name])
        detail = details[name]
        axes[2, 1].scatter(
            detail["event_time_s"],
            detail["event_hz"] * 60.0,
            s=10,
            alpha=0.65,
            label=name,
            color=colors[name],
        )
    axes[2, 0].set(title="First 8 s, bandpassed", xlabel="Time (s)", ylabel="Standard deviations")
    axes[2, 0].legend(fontsize=8)
    axes[2, 1].plot(reference_time, 60.0 * reference_hz, color="#333333", linewidth=1.5, label="reference")
    axes[2, 1].set(title="Peak-interval event rates", xlabel="Time (s)", ylabel="Rate (bpm)")
    axes[2, 1].legend(fontsize=8)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze an embedded-fish top-camera ROI against a supplied heart-rate trace.")
    parser.add_argument("--roi-json", type=Path, required=True)
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--trial-number", type=int, default=1)
    parser.add_argument("--reference-fps", type=float, default=100.0)
    parser.add_argument("--band-hz", type=float, nargs=2, default=(1.5, 4.0))
    parser.add_argument("--ring-margin", type=int, default=8)
    parser.add_argument("--control-offset-pixels", type=int, default=40)
    parser.add_argument("--skip-spatial-controls", action="store_true")
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    video, roi_xywh = _load_roi(args.roi_json)
    pixels, ring_mean, duplicates, first_frame, fps, frame_shape = _extract_video(
        video, roi_xywh, ring_margin=int(args.ring_margin)
    )
    reference_hz = read_numeric_xlsx_row(
        args.workbook,
        sheet_name="heart_rate_trace",
        row_number=int(args.trial_number) + 1,
    )
    video_duration = pixels.shape[0] / fps
    reference_duration = reference_hz.size / float(args.reference_fps)
    duration_difference = abs(video_duration - reference_duration)
    duration_tolerance = 1.0 / float(args.reference_fps) + 1.0 / fps
    if duration_difference > duration_tolerance + 1e-9:
        raise ValueError(
            f"video/reference durations differ: {video_duration:.6f} versus {reference_duration:.6f} s"
        )
    band_hz = (float(args.band_hz[0]), float(args.band_hz[1]))
    candidates, pca_variance = build_candidate_signals(
        pixels, ring_mean, fps=fps, band_hz=band_hz
    )
    summaries: list[CandidateSummary] = []
    details: dict[str, dict[str, np.ndarray]] = {}
    for name, signal in candidates.items():
        summary, detail = analyze_candidate(
            name,
            signal,
            fps=fps,
            band_hz=band_hz,
            reference_hz=reference_hz,
            reference_fps=float(args.reference_fps),
        )
        summaries.append(summary)
        details[name] = detail

    control_summaries: list[dict[str, object]] = []
    if not args.skip_spatial_controls:
        controls = _offset_controls(
            roi_xywh,
            frame_shape=frame_shape,
            offset_pixels=int(args.control_offset_pixels),
        )
        for name, control_roi in controls.items():
            control_pixels, control_ring, _, _, control_fps, _ = _extract_video(
                video,
                control_roi,
                ring_margin=int(args.ring_margin),
            )
            control_candidates, control_variance = build_candidate_signals(
                control_pixels,
                control_ring,
                fps=control_fps,
                band_hz=band_hz,
            )
            control_summary, _ = analyze_candidate(
                name,
                control_candidates["band_pca1"],
                fps=control_fps,
                band_hz=band_hz,
                reference_hz=reference_hz,
                reference_fps=float(args.reference_fps),
            )
            control_summaries.append(
                {
                    **asdict(control_summary),
                    "roi_xywh": list(control_roi),
                    "band_pca1_explained_variance_fraction": control_variance,
                }
            )

    prefix = args.output_prefix.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = prefix.with_suffix(".summary.json")
    csv_path = prefix.with_suffix(".summary.csv")
    plot_path = prefix.with_suffix(".diagnostic.png")
    payload = {
        "analysis_status": "descriptive_positive_control_not_formal_validation",
        "source_video": str(video.resolve()),
        "roi_json": str(args.roi_json.resolve()),
        "workbook": str(args.workbook.resolve()),
        "trial_number": int(args.trial_number),
        "roi_xywh": list(roi_xywh),
        "frame_shape_hw": list(frame_shape),
        "decoded_frames": int(pixels.shape[0]),
        "video_fps": fps,
        "reference_samples": int(reference_hz.size),
        "reference_fps": float(args.reference_fps),
        "video_duration_s": float(video_duration),
        "reference_duration_s": float(reference_duration),
        "duration_difference_s": float(duration_difference),
        "duration_tolerance_s": float(duration_tolerance),
        "reference_median_bpm": float(60.0 * np.median(reference_hz)),
        "reference_range_bpm": [float(60.0 * np.min(reference_hz)), float(60.0 * np.max(reference_hz))],
        "exact_roi_duplicate_frame_fraction": float(np.mean(duplicates[1:])),
        "band_hz": list(band_hz),
        "band_pca1_explained_variance_fraction": pca_variance,
        "candidate_selection": "none; all three pre-specified candidates are reported",
        "candidates": [asdict(summary) for summary in summaries],
        "spatial_control_method": (
            "same-size boxes shifted from the target; each control independently "
            "fits the same reference-blind band PCA"
        ),
        "spatial_controls": control_summaries,
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(summaries[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(summary) for summary in summaries)
    _plot(
        plot_path,
        frame=first_frame,
        roi_xywh=roi_xywh,
        fps=fps,
        candidates=candidates,
        details=details,
        reference_hz=reference_hz,
        reference_fps=float(args.reference_fps),
        band_hz=band_hz,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {summary_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
