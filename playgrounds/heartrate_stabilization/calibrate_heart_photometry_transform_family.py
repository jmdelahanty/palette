from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import scipy

import compare_heart_photometry_transforms as comparison
from analyze_frozen_heart_masks_longitudinal import _read_mask, _window_dataset
from diagnose_frozen_mask_longitudinal_tracking import _mask_at_pixels
from extract_reliable_local_rostral_heartrate import load_dataset
from fisheye.analysis.heart_photometry_nulls import (
    PhotometryFamilyEvaluation,
    compute_photometry_family_null_batch,
    evaluate_photometry_family,
    familywise_p_values,
    higher_quantile,
    load_photometry_null_batch,
    plus_one_p_value,
    write_photometry_null_batch,
)
from fisheye.analysis.local_rostral_heartrate import (
    LocalCoordinateDataset,
    alternating_block_partitions,
    autocorrelation_surrogate_shift_diagnostics,
)


_INTERPRETATION = (
    "exploratory_conditional_transform_family_null_not_full_pipeline_null_or_cardiac_validation"
)
_ANALYSIS_SCHEMA = "palette.heart_photometry_transform_family_calibration.v2"


@dataclass(frozen=True)
class FrozenWindow:
    window_index: int
    start: int
    stop: int
    source_frame_start: int
    source_frame_stop_inclusive: int
    frequency_hz: float


@dataclass(frozen=True)
class FrozenFamilyScorer:
    windows: tuple[FrozenWindow, ...]
    candidate_names: tuple[str, ...]
    target_pixels: np.ndarray
    upper_pixels: np.ndarray
    lower_pixels: np.ndarray
    reference_pixels: np.ndarray
    control_pixels: np.ndarray
    sg_windows: tuple[int, ...]
    lag_frames: tuple[int, ...]
    gaussian_sigma_px: float
    frequency_min_hz: float
    frequency_max_hz: float
    frequency_step_hz: float
    block_seconds: float
    guard_seconds: float
    min_block_seconds: float
    min_block_valid_fraction: float
    max_interpolated_gap_seconds: float
    nuisance_ridge: float
    outer_discovery_parity: int
    min_discovery_windows: int
    min_discovery_spectral_ratio: float
    min_discovery_control_ratio: float
    min_confirmation_windows: int
    min_confirmation_scorable_fraction: float
    include_matched_projection: bool

    def __call__(self, dataset: LocalCoordinateDataset) -> PhotometryFamilyEvaluation:
        shape = (len(self.windows), len(self.candidate_names))
        spectral = np.full(shape, np.nan, dtype=np.float64)
        control = np.full(shape, np.nan, dtype=np.float64)
        scorable = np.zeros(shape, dtype=bool)
        for window_position, window in enumerate(self.windows):
            local = _window_dataset(dataset, int(window.start), int(window.stop))
            try:
                partitions = alternating_block_partitions(
                    local.timestamps_s,
                    block_seconds=float(self.block_seconds),
                    guard_seconds=float(self.guard_seconds),
                )
                candidates = comparison._candidate_traces(
                    local,
                    target=self.target_pixels,
                    upper=self.upper_pixels,
                    lower=self.lower_pixels,
                    reference=self.reference_pixels,
                    control=self.control_pixels,
                    sg_windows=self.sg_windows,
                    lag_frames=self.lag_frames,
                    gaussian_sigma_px=float(self.gaussian_sigma_px),
                )
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                continue
            if self.include_matched_projection:
                try:
                    candidates["crossfit_matched_spatial_projection"] = (
                        comparison._matched_projection_trace_set(
                            local,
                            target=self.target_pixels,
                            control=self.control_pixels,
                            partitions=partitions,
                            frequency_hz=float(window.frequency_hz),
                        )
                    )
                except (RuntimeError, ValueError, np.linalg.LinAlgError):
                    pass
            for candidate_position, name in enumerate(self.candidate_names):
                raw = candidates.get(name)
                if raw is None:
                    continue
                try:
                    traces = (
                        raw
                        if name == "crossfit_matched_spatial_projection"
                        else comparison._crossfit_trace_set(
                            raw,
                            local,
                            partitions,
                            ridge=float(self.nuisance_ridge),
                        )
                    )
                    metrics = comparison._measure_window(
                        local,
                        traces,
                        frequency_hz=float(window.frequency_hz),
                        frequency_min_hz=float(self.frequency_min_hz),
                        frequency_max_hz=float(self.frequency_max_hz),
                        frequency_step_hz=float(self.frequency_step_hz),
                        block_seconds=float(self.block_seconds),
                        min_block_seconds=float(self.min_block_seconds),
                        min_valid_fraction=float(self.min_block_valid_fraction),
                        max_interpolated_gap_seconds=float(
                            self.max_interpolated_gap_seconds
                        ),
                    )
                except (RuntimeError, ValueError, np.linalg.LinAlgError):
                    continue
                if int(metrics.block_count) < 2:
                    continue
                spectral[window_position, candidate_position] = float(
                    metrics.spectral_ratio
                )
                control[window_position, candidate_position] = float(
                    metrics.control_ratio
                )
                scorable[window_position, candidate_position] = bool(
                    np.isfinite(metrics.spectral_ratio)
                    and np.isfinite(metrics.control_ratio)
                )
        window_indices = np.asarray(
            [window.window_index for window in self.windows], dtype=np.int32
        )
        return evaluate_photometry_family(
            candidate_names=self.candidate_names,
            window_indices=window_indices,
            discovery_windows=(
                window_indices % 2 == int(self.outer_discovery_parity)
            ),
            spectral_ratios=spectral,
            control_ratios=control,
            scorable=scorable,
            min_discovery_windows=int(self.min_discovery_windows),
            min_discovery_spectral_ratio=float(
                self.min_discovery_spectral_ratio
            ),
            min_discovery_control_ratio=float(self.min_discovery_control_ratio),
            min_confirmation_windows=int(self.min_confirmation_windows),
            min_confirmation_scorable_fraction=float(
                self.min_confirmation_scorable_fraction
            ),
        )


def _parse_ints(value: str | None) -> tuple[int, ...]:
    if value is None or not value.strip():
        return ()
    return tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _file_identity(path: Path) -> dict[str, Any]:
    stat = Path(path).stat()
    return {
        "path": str(Path(path).resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _analysis_identity(
    *,
    dataset_path: Path,
    longitudinal_path: Path,
    masks: Mapping[str, np.ndarray],
    scorer: FrozenFamilyScorer,
    seed: int,
    spatial_block_px: int,
    min_shift_seconds: float,
    max_gap_factor: float,
) -> str:
    code_paths = (
        Path(__file__),
        Path(comparison.__file__),
        Path(__file__).with_name("analyze_frozen_heart_masks_longitudinal.py"),
        Path(__file__).with_name("diagnose_frozen_mask_longitudinal_tracking.py"),
        Path(__file__).with_name("extract_reliable_local_rostral_heartrate.py"),
        Path(__file__).resolve().parents[2]
        / "src/fisheye/analysis/heart_photometry_nulls.py",
        Path(__file__).resolve().parents[2]
        / "src/fisheye/analysis/heart_photometry_transforms.py",
        Path(__file__).resolve().parents[2]
        / "src/fisheye/analysis/heart_photometry_projection.py",
        Path(__file__).resolve().parents[2]
        / "src/fisheye/analysis/local_rostral_heartrate.py",
    )
    payload = {
        "schema": _ANALYSIS_SCHEMA,
        "dataset": _file_identity(dataset_path),
        "longitudinal": _file_identity(longitudinal_path),
        "masks": {
            name: hashlib.sha256(
                np.asarray(mask, dtype=np.uint8).tobytes()
            ).hexdigest()
            for name, mask in masks.items()
        },
        "windows": [
            {
                "index": int(window.window_index),
                "start": int(window.start),
                "stop": int(window.stop),
                "frequency_hz": float(window.frequency_hz),
            }
            for window in scorer.windows
        ],
        "family": {
            "candidate_names": list(scorer.candidate_names),
            "sg_windows": list(scorer.sg_windows),
            "lag_frames": list(scorer.lag_frames),
            "gaussian_sigma_px": float(scorer.gaussian_sigma_px),
            "frequency_min_hz": float(scorer.frequency_min_hz),
            "frequency_max_hz": float(scorer.frequency_max_hz),
            "frequency_step_hz": float(scorer.frequency_step_hz),
            "block_seconds": float(scorer.block_seconds),
            "guard_seconds": float(scorer.guard_seconds),
            "min_block_seconds": float(scorer.min_block_seconds),
            "min_block_valid_fraction": float(
                scorer.min_block_valid_fraction
            ),
            "max_interpolated_gap_seconds": float(
                scorer.max_interpolated_gap_seconds
            ),
            "nuisance_ridge": float(scorer.nuisance_ridge),
            "outer_discovery_parity": int(scorer.outer_discovery_parity),
            "min_discovery_windows": int(scorer.min_discovery_windows),
            "min_discovery_spectral_ratio": float(
                scorer.min_discovery_spectral_ratio
            ),
            "min_discovery_control_ratio": float(
                scorer.min_discovery_control_ratio
            ),
            "min_confirmation_windows": int(scorer.min_confirmation_windows),
            "min_confirmation_scorable_fraction": float(
                scorer.min_confirmation_scorable_fraction
            ),
        },
        "surrogate": {
            "seed": int(seed),
            "spatial_block_px": int(spatial_block_px),
            "requested_min_shift_seconds": float(min_shift_seconds),
            "max_gap_factor": float(max_gap_factor),
        },
        "code_sha256": {
            str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in code_paths
        },
        "runtime_versions": {
            "numpy": str(np.__version__),
            "scipy": str(scipy.__version__),
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _candidate_names(
    *,
    sg_windows: Sequence[int],
    lag_frames: Sequence[int],
    gaussian_sigma_px: float,
    include_matched_projection: bool,
) -> tuple[str, ...]:
    names = (
        "baseline_mean_intensity",
        "robust_huber_intensity",
        "reference_log_ratio",
        "reference_fractional_difference",
        f"masked_gaussian_huber_sigma{float(gaussian_sigma_px):g}",
        "regional_spatial_std",
        *(f"huber_savgol_derivative_w{value}" for value in sg_windows),
        f"gaussian_savgol_derivative_w7_sigma{float(gaussian_sigma_px):g}",
        *(f"huber_normalized_signed_lag{value}" for value in lag_frames),
    )
    if include_matched_projection:
        return (*names, "crossfit_matched_spatial_projection")
    return names


def _load_windows(
    path: Path,
    dataset: LocalCoordinateDataset,
    *,
    frequency_source_mask: str,
    selected_indices: set[int] | None,
    max_windows: int | None,
    frame_count: int | None,
) -> tuple[FrozenWindow, ...]:
    rows = comparison._read_source_windows(path, frequency_source_mask)
    if selected_indices is not None:
        rows = [row for row in rows if int(row["window_index"]) in selected_indices]
    rows = [row for row in rows if row["status"] == "ok"]
    if max_windows is not None:
        rows = rows[: int(max_windows)]
    frame_indices = np.asarray(dataset.frame_indices, dtype=np.int64)
    windows: list[FrozenWindow] = []
    for row in rows:
        source_start = int(row["window_frame_start"])
        source_stop = int(row["window_frame_stop_inclusive"])
        start = int(np.searchsorted(frame_indices, source_start, side="left"))
        stop = int(np.searchsorted(frame_indices, source_stop, side="right"))
        if frame_count is not None:
            stop = min(stop, start + int(frame_count))
        if stop - start < 16:
            continue
        frequency = float(row["candidate_frequency_hz"] or "nan")
        if not np.isfinite(frequency):
            continue
        windows.append(
            FrozenWindow(
                window_index=int(row["window_index"]),
                start=start,
                stop=stop,
                source_frame_start=source_start,
                source_frame_stop_inclusive=source_stop,
                frequency_hz=frequency,
            )
        )
    if not windows:
        raise ValueError("no scorable frozen-frequency windows remain")
    return tuple(windows)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate the complete Mono8 photometry transform family with "
            "autocorrelation-preserving spatial-block surrogates."
        )
    )
    parser.add_argument("--dataset-npz", type=Path, required=True)
    parser.add_argument("--longitudinal-csv", type=Path, required=True)
    parser.add_argument("--original-mask-npz", type=Path, required=True)
    parser.add_argument("--original-mask-key", default="heart_support_mask")
    parser.add_argument("--consensus-mask-npz", type=Path, required=True)
    parser.add_argument("--consensus-mask-key", default="consensus_mask")
    parser.add_argument("--regions-npz", type=Path, required=True)
    parser.add_argument("--upper-key", default="upper_mask")
    parser.add_argument("--lower-key", default="lower_mask")
    parser.add_argument("--reference-mask-npz", type=Path)
    parser.add_argument("--reference-mask-key", default="reference_mask")
    parser.add_argument("--control-mask-npz", type=Path)
    parser.add_argument("--control-mask-key", default="control_mask")
    parser.add_argument("--frequency-source-mask", default="intersection_8")
    parser.add_argument("--frequency-min-hz", type=float, default=2.0)
    parser.add_argument("--frequency-max-hz", type=float, default=4.0)
    parser.add_argument("--frequency-step-hz", type=float, default=0.05)
    parser.add_argument("--block-seconds", type=float, default=4.0)
    parser.add_argument("--guard-seconds", type=float, default=0.25)
    parser.add_argument("--min-block-seconds", type=float, default=2.0)
    parser.add_argument("--min-block-valid-fraction", type=float, default=0.7)
    parser.add_argument("--max-interpolated-gap-seconds", type=float, default=0.02)
    parser.add_argument("--nuisance-ridge", type=float, default=1e-6)
    parser.add_argument("--sg-windows", default="5,7,11")
    parser.add_argument("--lag-frames", default="8,12,16")
    parser.add_argument("--gaussian-sigma-px", type=float, default=0.8)
    parser.add_argument("--outer-discovery-parity", type=int, choices=(0, 1), default=0)
    parser.add_argument("--min-discovery-windows", type=int, default=3)
    parser.add_argument("--min-discovery-spectral-ratio", type=float, default=1.5)
    parser.add_argument("--min-discovery-control-ratio", type=float, default=1.1)
    parser.add_argument("--min-confirmation-windows", type=int, default=3)
    parser.add_argument(
        "--min-confirmation-scorable-fraction", type=float, default=0.5
    )
    parser.add_argument("--surrogate-count", type=int, default=199)
    parser.add_argument("--surrogate-batch-size", type=int, default=10)
    parser.add_argument("--surrogate-workers", type=int, default=2)
    parser.add_argument("--surrogate-batch-dir", type=Path)
    parser.add_argument("--surrogate-spatial-block-px", type=int, default=2)
    parser.add_argument("--surrogate-min-shift-seconds", type=float, default=1.0)
    parser.add_argument("--surrogate-max-gap-factor", type=float, default=1.75)
    parser.add_argument("--seed", type=int, default=811)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--window-indices")
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--frame-count", type=int)
    parser.add_argument("--skip-matched-projection", action="store_true")
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    if (
        int(args.surrogate_count) < 1
        or int(args.surrogate_batch_size) < 1
        or int(args.surrogate_workers) < 1
    ):
        raise ValueError("surrogate count, batch size, and workers must be positive")
    if not 0.0 < float(args.alpha) < 1.0:
        raise ValueError("alpha must be between zero and one")
    if int(args.min_confirmation_windows) < 1:
        raise ValueError("min-confirmation-windows must be positive")
    if not 0.0 < float(args.min_confirmation_scorable_fraction) <= 1.0:
        raise ValueError("min-confirmation-scorable-fraction must be in (0, 1]")
    sg_windows = _parse_ints(args.sg_windows)
    lag_frames = _parse_ints(args.lag_frames)
    if any(value < 3 or value % 2 == 0 for value in sg_windows):
        raise ValueError("Savitzky-Golay windows must be odd and at least 3")
    if any(value < 2 or value % 2 for value in lag_frames):
        raise ValueError("centered lag differences require positive even lags")
    if args.frame_count is not None and int(args.frame_count) < max(
        64, max(lag_frames, default=0) + 16
    ):
        raise ValueError("frame-count is too short for the configured transform family")

    dataset = load_dataset(args.dataset_npz)
    original_image = _read_mask(args.original_mask_npz, args.original_mask_key)
    consensus_image = _read_mask(args.consensus_mask_npz, args.consensus_mask_key)
    upper_stored = _read_mask(args.regions_npz, args.upper_key)
    lower_stored = _read_mask(args.regions_npz, args.lower_key)
    target = comparison._validate_pixel_mask(
        "original target", _mask_at_pixels(original_image, dataset.pixel_xy)
    )
    consensus = comparison._validate_pixel_mask(
        "consensus", _mask_at_pixels(consensus_image, dataset.pixel_xy)
    )
    intersection = comparison._validate_pixel_mask(
        "original/consensus intersection", target & consensus
    )
    upper = comparison._validate_pixel_mask(
        "upper",
        comparison._stored_mask_at_pixels(upper_stored, dataset, name="upper")
        & target,
    )
    lower = comparison._validate_pixel_mask(
        "lower",
        comparison._stored_mask_at_pixels(lower_stored, dataset, name="lower")
        & target,
    )
    if np.any(upper & lower) or not np.array_equal(upper | lower, target):
        raise ValueError("upper/lower masks must form a disjoint target partition")

    automatic_reference, automatic_control = comparison._auto_reference_and_control_masks(
        original_image
    )
    reference_image = (
        _read_mask(args.reference_mask_npz, args.reference_mask_key)
        if args.reference_mask_npz is not None
        else automatic_reference
    )
    control_image = (
        _read_mask(args.control_mask_npz, args.control_mask_key)
        if args.control_mask_npz is not None
        else automatic_control
    )
    reference = comparison._validate_pixel_mask(
        "reference", _mask_at_pixels(reference_image, dataset.pixel_xy)
    )
    control = comparison._validate_pixel_mask(
        "control", _mask_at_pixels(control_image, dataset.pixel_xy)
    )
    if np.any(reference & target) or np.any(control & target):
        raise ValueError("reference/control masks cannot overlap the target")

    selected = set(_parse_ints(args.window_indices)) if args.window_indices else None
    windows = _load_windows(
        args.longitudinal_csv,
        dataset,
        frequency_source_mask=str(args.frequency_source_mask),
        selected_indices=selected,
        max_windows=args.max_windows,
        frame_count=args.frame_count,
    )
    include_matched = not bool(args.skip_matched_projection)
    names = _candidate_names(
        sg_windows=sg_windows,
        lag_frames=lag_frames,
        gaussian_sigma_px=float(args.gaussian_sigma_px),
        include_matched_projection=include_matched,
    )
    scorer = FrozenFamilyScorer(
        windows=windows,
        candidate_names=names,
        target_pixels=target,
        upper_pixels=upper,
        lower_pixels=lower,
        reference_pixels=reference,
        control_pixels=control,
        sg_windows=sg_windows,
        lag_frames=lag_frames,
        gaussian_sigma_px=float(args.gaussian_sigma_px),
        frequency_min_hz=float(args.frequency_min_hz),
        frequency_max_hz=float(args.frequency_max_hz),
        frequency_step_hz=float(args.frequency_step_hz),
        block_seconds=float(args.block_seconds),
        guard_seconds=float(args.guard_seconds),
        min_block_seconds=float(args.min_block_seconds),
        min_block_valid_fraction=float(args.min_block_valid_fraction),
        max_interpolated_gap_seconds=float(args.max_interpolated_gap_seconds),
        nuisance_ridge=float(args.nuisance_ridge),
        outer_discovery_parity=int(args.outer_discovery_parity),
        min_discovery_windows=int(args.min_discovery_windows),
        min_discovery_spectral_ratio=float(args.min_discovery_spectral_ratio),
        min_discovery_control_ratio=float(args.min_discovery_control_ratio),
        min_confirmation_windows=int(args.min_confirmation_windows),
        min_confirmation_scorable_fraction=float(
            args.min_confirmation_scorable_fraction
        ),
        include_matched_projection=include_matched,
    )
    identity = _analysis_identity(
        dataset_path=args.dataset_npz,
        longitudinal_path=args.longitudinal_csv,
        masks={
            "target": target,
            "consensus": consensus,
            "intersection": intersection,
            "upper": upper,
            "lower": lower,
            "reference": reference,
            "control": control,
        },
        scorer=scorer,
        seed=int(args.seed),
        spatial_block_px=int(args.surrogate_spatial_block_px),
        min_shift_seconds=float(args.surrogate_min_shift_seconds),
        max_gap_factor=float(args.surrogate_max_gap_factor),
    )
    observed_started = time.perf_counter()
    observed = scorer(dataset)
    observed_elapsed = time.perf_counter() - observed_started

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    batch_dir = (
        Path(args.surrogate_batch_dir)
        if args.surrogate_batch_dir is not None
        else output_prefix.parent
        / f"{output_prefix.name}.photometry_family_null.surrogate_batches"
    )
    count = int(args.surrogate_count)
    null_maximum = np.full(count, np.nan, dtype=np.float64)
    null_confirmation = np.full(count, np.nan, dtype=np.float64)
    null_selected = np.full(count, -2, dtype=np.int32)
    null_confirmation_count = np.full(count, -1, dtype=np.int16)
    null_confirmation_fraction = np.full(count, np.nan, dtype=np.float64)
    null_confirmation_gate = np.zeros(count, dtype=bool)
    null_best_window = np.full(count, -2, dtype=np.int32)
    null_best_candidate = np.full(count, -2, dtype=np.int32)
    active = np.zeros(dataset.frame_count, dtype=bool)
    for window in windows:
        active[window.start : window.stop] = True
    active &= np.asarray(dataset.frame_valid, dtype=bool)
    shift_diagnostics = autocorrelation_surrogate_shift_diagnostics(
        dataset,
        active,
        min_shift_seconds=float(args.surrogate_min_shift_seconds),
        max_gap_factor=float(args.surrogate_max_gap_factor),
    )
    null_started = time.perf_counter()
    computed_count = 0
    for batch_start in range(0, count, int(args.surrogate_batch_size)):
        batch_stop = min(batch_start + int(args.surrogate_batch_size), count)
        indices = np.arange(batch_start, batch_stop, dtype=np.int64)
        batch_path = batch_dir / (
            f"{identity[:16]}.surrogates_{batch_start:06d}_{batch_stop:06d}.npz"
        )
        batch = load_photometry_null_batch(
            batch_path,
            identity=identity,
            expected_indices=indices,
        )
        if batch is None:
            batch = compute_photometry_family_null_batch(
                dataset,
                active,
                surrogate_indices=indices,
                seed=int(args.seed),
                scorer=scorer,
                spatial_block_px=int(args.surrogate_spatial_block_px),
                min_shift_seconds=float(args.surrogate_min_shift_seconds),
                max_gap_factor=float(args.surrogate_max_gap_factor),
                workers=int(args.surrogate_workers),
            )
            write_photometry_null_batch(batch_path, identity=identity, batch=batch)
            computed_count += int(indices.size)
        null_maximum[indices] = batch.maximum_cell_statistics
        null_confirmation[indices] = batch.selected_confirmation_statistics
        null_selected[indices] = batch.selected_candidate_indices
        null_confirmation_count[indices] = (
            batch.selected_confirmation_window_counts
        )
        null_confirmation_fraction[indices] = (
            batch.selected_confirmation_scorable_fractions
        )
        null_confirmation_gate[indices] = batch.selected_confirmation_gate_passed
        null_best_window[indices] = batch.maximum_window_indices
        null_best_candidate[indices] = batch.maximum_candidate_indices
        print(
            f"surrogates {batch_stop}/{count} computed_now={computed_count} "
            f"no_candidate={np.count_nonzero(null_selected[:batch_stop] == -1)}",
            flush=True,
        )
    null_elapsed = time.perf_counter() - null_started
    if np.any(np.isnan(null_maximum)) or np.any(np.isnan(null_confirmation)):
        raise RuntimeError("merged surrogate arrays contain NaN")

    alpha = float(args.alpha)
    maximum_threshold = higher_quantile(null_maximum, alpha=alpha)
    maximum_p = plus_one_p_value(observed.maximum_cell_statistic, null_maximum)
    cell_p = familywise_p_values(observed.cell_statistics, null_maximum)
    cell_significant = (
        (cell_p <= alpha)
        & (observed.cell_statistics > maximum_threshold)
        & observed.scorable
    )
    adaptive_threshold = higher_quantile(null_confirmation, alpha=alpha)
    adaptive_p = plus_one_p_value(
        observed.selected_confirmation_statistic, null_confirmation
    )
    adaptive_confirmed = bool(
        observed.selected_candidate_index >= 0
        and observed.selected_confirmation_gate_passed
        and adaptive_p <= alpha
        and observed.selected_confirmation_statistic > adaptive_threshold
    )

    cell_rows: list[dict[str, Any]] = []
    for wi, window in enumerate(windows):
        for ci, name in enumerate(names):
            cell_rows.append(
                {
                    "window_index": int(window.window_index),
                    "outer_role": (
                        "discovery"
                        if window.window_index % 2 == int(args.outer_discovery_parity)
                        else "confirmation"
                    ),
                    "frozen_frequency_hz": float(window.frequency_hz),
                    "candidate": name,
                    "scorable": bool(observed.scorable[wi, ci]),
                    "spectral_ratio": float(observed.spectral_ratios[wi, ci]),
                    "control_ratio": float(observed.control_ratios[wi, ci]),
                    "combined_log2_statistic": float(observed.cell_statistics[wi, ci]),
                    "maximum_familywise_p_value": float(cell_p[wi, ci]),
                    "maximum_familywise_significant": bool(cell_significant[wi, ci]),
                }
            )
    surrogate_rows = [
        {
            "surrogate_index": index,
            "maximum_cell_statistic": float(null_maximum[index]),
            "selected_confirmation_statistic": float(null_confirmation[index]),
            "selected_candidate_index": int(null_selected[index]),
            "selected_candidate": (
                names[int(null_selected[index])] if int(null_selected[index]) >= 0 else ""
            ),
            "selected_confirmation_window_count": int(
                null_confirmation_count[index]
            ),
            "selected_confirmation_scorable_fraction": float(
                null_confirmation_fraction[index]
            ),
            "selected_confirmation_gate_passed": bool(
                null_confirmation_gate[index]
            ),
            "maximum_window_index": int(null_best_window[index]),
            "maximum_candidate_index": int(null_best_candidate[index]),
            "maximum_candidate": (
                names[int(null_best_candidate[index])]
                if int(null_best_candidate[index]) >= 0
                else ""
            ),
        }
        for index in range(count)
    ]
    cells_path = output_prefix.parent / (
        f"{output_prefix.name}.photometry_family_null.cells.csv"
    )
    surrogates_path = output_prefix.parent / (
        f"{output_prefix.name}.photometry_family_null.surrogates.csv"
    )
    arrays_path = output_prefix.parent / (
        f"{output_prefix.name}.photometry_family_null.arrays.npz"
    )
    summary_path = output_prefix.parent / (
        f"{output_prefix.name}.photometry_family_null.summary.json"
    )
    _write_csv(cells_path, cell_rows)
    _write_csv(surrogates_path, surrogate_rows)
    np.savez_compressed(
        arrays_path,
        schema=np.asarray(_ANALYSIS_SCHEMA),
        interpretation=np.asarray(_INTERPRETATION),
        identity=np.asarray(identity),
        candidate_names=np.asarray(names),
        window_indices=np.asarray(
            [window.window_index for window in windows], dtype=np.int32
        ),
        discovery_windows=np.asarray(observed.discovery_windows, dtype=bool),
        frozen_frequencies_hz=np.asarray(
            [window.frequency_hz for window in windows], dtype=np.float32
        ),
        observed_spectral_ratios=observed.spectral_ratios.astype(np.float32),
        observed_control_ratios=observed.control_ratios.astype(np.float32),
        observed_scorable=observed.scorable,
        observed_cell_statistics=observed.cell_statistics.astype(np.float32),
        observed_cell_familywise_p_values=cell_p.astype(np.float32),
        observed_cell_familywise_significant=cell_significant,
        observed_discovery_selection_scores=(
            observed.discovery_selection_scores.astype(np.float32)
        ),
        observed_selected_candidate_index=np.asarray(
            observed.selected_candidate_index, dtype=np.int32
        ),
        observed_selected_confirmation_window_count=np.asarray(
            observed.selected_confirmation_window_count, dtype=np.int16
        ),
        observed_total_confirmation_window_count=np.asarray(
            observed.total_confirmation_window_count, dtype=np.int16
        ),
        observed_selected_confirmation_scorable_fraction=np.asarray(
            observed.selected_confirmation_scorable_fraction, dtype=np.float32
        ),
        observed_selected_confirmation_gate_passed=np.asarray(
            observed.selected_confirmation_gate_passed, dtype=bool
        ),
        observed_selected_confirmation_statistic=np.asarray(
            observed.selected_confirmation_statistic, dtype=np.float32
        ),
        observed_maximum_cell_statistic=np.asarray(
            observed.maximum_cell_statistic, dtype=np.float32
        ),
        null_maximum_cell_statistics=null_maximum.astype(np.float32),
        null_selected_confirmation_statistics=null_confirmation.astype(np.float32),
        null_selected_candidate_indices=null_selected,
        null_selected_confirmation_window_counts=null_confirmation_count,
        null_selected_confirmation_scorable_fractions=(
            null_confirmation_fraction.astype(np.float32)
        ),
        null_selected_confirmation_gate_passed=null_confirmation_gate,
        null_maximum_window_indices=null_best_window,
        null_maximum_candidate_indices=null_best_candidate,
    )
    selected_name = (
        names[observed.selected_candidate_index]
        if observed.selected_candidate_index >= 0
        else None
    )
    strict_detected = bool(
        maximum_p <= alpha
        and observed.maximum_cell_statistic > maximum_threshold
    )
    summary = {
        "schema": _ANALYSIS_SCHEMA,
        "interpretation": _INTERPRETATION,
        "identity": identity,
        "inference_limit": (
            "THIS IS NOT A FULL-PIPELINE NULL. The per-window frequencies were "
            "adaptively discovered earlier from the observed recording and are held fixed "
            "here, as are the anatomical masks and regional/control definitions. This null "
            f"only calibrates the downstream {len(names)}-transform search conditional on "
            "those earlier "
            "choices. Its p-values are exploratory conditional diagnostics, not overall "
            "false-positive probabilities, cardiac-identity evidence, or event validation."
        ),
        "null_scope": {
            "full_pipeline_null": False,
            "adaptive_frequency_discovery_rerun": False,
            "mask_discovery_rerun": False,
            "regional_split_and_controls_rerun": False,
            "downstream_transform_family_rerun": True,
        },
        "sources": {
            "dataset_npz": str(args.dataset_npz),
            "longitudinal_csv": str(args.longitudinal_csv),
            "original_mask_npz": str(args.original_mask_npz),
            "consensus_mask_npz": str(args.consensus_mask_npz),
            "regions_npz": str(args.regions_npz),
        },
        "family": {
            "candidate_count": len(names),
            "candidate_names": list(names),
            "window_count": len(windows),
            "discovery_window_count": int(np.count_nonzero(observed.discovery_windows)),
            "confirmation_window_count": int(np.count_nonzero(~observed.discovery_windows)),
            "frequency_policy": (
                "fixed per-window values adaptively discovered earlier from observed data; "
                "frequency discovery is not represented in this null"
            ),
            "allows_no_candidate": True,
            "minimum_scorable_confirmation_windows": int(
                args.min_confirmation_windows
            ),
            "minimum_scorable_confirmation_fraction": float(
                args.min_confirmation_scorable_fraction
            ),
        },
        "surrogates": {
            "count": count,
            "seed": int(args.seed),
            "spatial_block_px": int(args.surrogate_spatial_block_px),
            "requested_minimum_shift_seconds": float(
                args.surrogate_min_shift_seconds
            ),
            "effective_shift_diagnostics": shift_diagnostics,
            "maximum_gap_factor": float(args.surrogate_max_gap_factor),
            "random_stream": "SeedSequence([seed, global_surrogate_index])",
            "missingness_policy": (
                "per-pixel validity is circularly shifted with its intensity samples; "
                "frame validity and nuisance measurements stay at observed times"
            ),
            "workers": int(args.surrogate_workers),
            "batch_size": int(args.surrogate_batch_size),
            "computed_this_run": computed_count,
            "no_candidate_count": int(np.count_nonzero(null_selected == -1)),
            "no_candidate_fraction": float(np.mean(null_selected == -1)),
        },
        "strict_maximum_test": {
            "scope": "maximum across every scorable transform and relevant window",
            "combined_statistic": "log2(target_to_sideband) + log2(target_to_control)",
            "alpha": alpha,
            "observed_maximum": float(observed.maximum_cell_statistic),
            "observed_best_window_index": int(observed.maximum_window_index),
            "observed_best_candidate": (
                names[observed.maximum_candidate_index]
                if observed.maximum_candidate_index >= 0
                else None
            ),
            "null_threshold": maximum_threshold,
            "familywise_p_value": maximum_p,
            "detected": strict_detected,
            "significant_cell_count": int(np.count_nonzero(cell_significant)),
        },
        "adaptive_discovery_confirmation_test": {
            "selection": (
                "choose one transform from discovery-window median spectral/control "
                "ratios after frozen gates; evaluate only that transform on confirmation windows"
            ),
            "selected_candidate": selected_name,
            "no_candidate_selected": selected_name is None,
            "minimum_scorable_confirmation_windows": int(
                args.min_confirmation_windows
            ),
            "minimum_scorable_confirmation_fraction": float(
                args.min_confirmation_scorable_fraction
            ),
            "observed_scorable_confirmation_window_count": int(
                observed.selected_confirmation_window_count
            ),
            "observed_total_confirmation_window_count": int(
                observed.total_confirmation_window_count
            ),
            "observed_scorable_confirmation_fraction": float(
                observed.selected_confirmation_scorable_fraction
            ),
            "observed_confirmation_gate_passed": bool(
                observed.selected_confirmation_gate_passed
            ),
            "observed_confirmation_statistic": float(
                observed.selected_confirmation_statistic
            ),
            "null_threshold": adaptive_threshold,
            "familywise_p_value": adaptive_p,
            "confirmed": adaptive_confirmed,
        },
        "runtime": {
            "observed_seconds": observed_elapsed,
            "null_seconds_this_invocation": null_elapsed,
            "mean_seconds_per_computed_surrogate": (
                null_elapsed / computed_count if computed_count else None
            ),
            "estimated_seconds_for_199_at_current_effective_rate": (
                null_elapsed / computed_count * 199 if computed_count else None
            ),
        },
        "outputs": {
            "cells_csv": str(cells_path),
            "surrogates_csv": str(surrogates_path),
            "arrays_npz": str(arrays_path),
            "summary_json": str(summary_path),
            "surrogate_batch_dir": str(batch_dir),
        },
    }
    summary_path.write_text(json.dumps(_json_value(summary), indent=2, sort_keys=True) + "\n")
    print(json.dumps(_json_value(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
