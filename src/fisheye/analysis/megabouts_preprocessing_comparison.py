"""Compare Palette classifier tensors with Megabouts-preprocessed tensors.

This module is intentionally read-only. It answers one narrow question:
given the same Palette swim-bout windows, how different are the classifier
inputs if we run Megabouts' own preprocessing first?
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import zarr

from fisheye.analysis.megabouts_classifier import (
    _git_commit_for_repo,
    _load_megabouts_runtime,
    _resolve_megabouts_repo,
    classify_megabouts_input_pack,
)
from fisheye.analysis.megabouts_classifier_inputs import (
    DEFAULT_ALIGN_TRAJ_TO_ONSET,
    DEFAULT_BOUT_DURATION_S,
    DEFAULT_HEADING_SOURCE,
    DEFAULT_MAX_CONSECUTIVE_INVALID_FRAMES,
    DEFAULT_MIN_TAIL_VALID_FRACTION,
    DEFAULT_MIN_TRAJ_VALID_FRACTION,
    DEFAULT_TRAJ_REFERENCE_INDEX,
    MEGABOUTS_TAIL_SEGMENT_COUNT,
    MegaboutsClassifierInputPack,
    _align_traj_array_to_reference,
    _frame_to_index,
    _json_safe,
    _load_track_arrays,
    _max_consecutive_false,
    _require_array,
    _resolve_tail_posture_view_run,
    _resolve_track_run,
    build_megabouts_classifier_input_pack,
    summarize_input_pack,
)
from fisheye.utils.zarr_io import open_zarr_root

COMPARISON_METHOD = "palette_megabouts_preprocessing_comparison"
COMPARISON_METHOD_VERSION = 1
MEGABOUTS_PREPROCESSED_MODE = "megabouts_preprocessed_full_timeseries"


@dataclass(frozen=True)
class MegaboutsPreprocessingRuntime:
    """Resolved optional Megabouts preprocessing objects."""

    tail_preprocessing_class: object
    tail_preprocessing_config_class: object
    traj_preprocessing_class: object
    traj_preprocessing_config_class: object
    package_version: str
    package_path: str
    source_repo: Optional[str]
    git_commit: Optional[str]


def _load_megabouts_preprocessing_runtime(
    megabouts_repo: Optional[str | Path] = None,
) -> MegaboutsPreprocessingRuntime:
    repo_path = _resolve_megabouts_repo(megabouts_repo)
    if repo_path is not None and str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

    try:
        import megabouts
        from megabouts.config.preprocessing_config import (
            TailPreprocessingConfig,
            TrajPreprocessingConfig,
        )
        from megabouts.preprocessing import TailPreprocessing, TrajPreprocessing
    except Exception as exc:  # pragma: no cover - depends on optional external package
        raise RuntimeError(
            "Megabouts is required for preprocessing comparison but is not importable. "
            "Install/configure Megabouts preprocessing dependencies outside Palette, "
            f"or pass --megabouts-repo. Import failed with: {exc}"
        ) from exc

    package_path = str(Path(getattr(megabouts, "__file__", "") or "").resolve())
    return MegaboutsPreprocessingRuntime(
        tail_preprocessing_class=TailPreprocessing,
        tail_preprocessing_config_class=TailPreprocessingConfig,
        traj_preprocessing_class=TrajPreprocessing,
        traj_preprocessing_config_class=TrajPreprocessingConfig,
        package_version=str(getattr(megabouts, "__version__", "unknown")),
        package_path=package_path,
        source_repo=None if repo_path is None else str(repo_path),
        git_commit=_git_commit_for_repo(repo_path),
    )


def _runtime_attrs(runtime: MegaboutsPreprocessingRuntime) -> dict[str, object]:
    return {
        "megabouts_package_version": runtime.package_version,
        "megabouts_package_path": runtime.package_path,
        "megabouts_source_repo": runtime.source_repo,
        "megabouts_git_commit": runtime.git_commit,
    }


def _dense_frame_axis(*frame_arrays: np.ndarray) -> np.ndarray:
    finite_frames: list[np.ndarray] = []
    for frames in frame_arrays:
        arr = np.asarray(frames, dtype=np.int64).reshape(-1)
        if arr.size:
            finite_frames.append(arr)
    if not finite_frames:
        return np.asarray([], dtype=np.int64)
    merged = np.concatenate(finite_frames)
    start = int(np.min(merged))
    end = int(np.max(merged))
    return np.arange(start, end + 1, dtype=np.int64)


def _window_extract(
    values_by_frame: np.ndarray,
    valid_by_frame: np.ndarray,
    frame_axis: np.ndarray,
    window_start_frame: np.ndarray,
    window_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract fixed windows from a dense frame-indexed series.

    Returns arrays in Megabouts classifier layout: ``(n_bouts, channels, window)``.
    """

    values = np.asarray(values_by_frame, dtype=np.float32)
    valid = np.asarray(valid_by_frame, dtype=bool)
    if values.ndim != 2:
        raise ValueError(f"values_by_frame must be 2D, got shape {values.shape}.")
    if valid.shape != (values.shape[0],):
        raise ValueError(f"valid_by_frame shape {valid.shape} does not match values shape {values.shape}.")
    frames = np.asarray(frame_axis, dtype=np.int64)
    if frames.size != values.shape[0]:
        raise ValueError(f"frame_axis length {frames.size} does not match values rows {values.shape[0]}.")

    n_bouts = int(np.asarray(window_start_frame).shape[0])
    n_channels = int(values.shape[1])
    out = np.full((n_bouts, n_channels, int(window_frames)), np.nan, dtype=np.float32)
    out_valid = np.zeros((n_bouts, int(window_frames)), dtype=bool)
    if frames.size == 0:
        return out, out_valid

    origin = int(frames[0])
    for bout_idx, start in enumerate(np.asarray(window_start_frame, dtype=np.int64).tolist()):
        for sample_idx, frame in enumerate(range(int(start), int(start) + int(window_frames))):
            row = int(frame) - origin
            if 0 <= row < values.shape[0] and bool(valid[row]):
                row_values = values[row]
                if np.all(np.isfinite(row_values)):
                    out[bout_idx, :, sample_idx] = row_values
                    out_valid[bout_idx, sample_idx] = True
    return out, out_valid


def _build_dense_tail_df(
    root: zarr.Group,
    pack: MegaboutsClassifierInputPack,
) -> tuple[object, np.ndarray, np.ndarray]:
    import pandas as pd

    posture = root[pack.source_refs["tail_posture_view_run"]]
    posture_frames = np.asarray(_require_array(posture, "frame_index")[:], dtype=np.int64)
    posture_valid = np.asarray(_require_array(posture, "valid")[:], dtype=bool)
    tail_angle = np.asarray(_require_array(posture, "tail_angle_rad")[:], dtype=np.float32)
    if tail_angle.ndim != 2 or tail_angle.shape[1] != MEGABOUTS_TAIL_SEGMENT_COUNT:
        raise ValueError(
            "Tail posture view must provide "
            f"{MEGABOUTS_TAIL_SEGMENT_COUNT} tail-angle channels, got {tail_angle.shape}."
        )

    frame_axis = _dense_frame_axis(
        posture_frames,
        np.asarray(pack.window_start_frame, dtype=np.int64),
        np.asarray(pack.window_end_frame, dtype=np.int64),
    )
    dense = np.full((frame_axis.size, MEGABOUTS_TAIL_SEGMENT_COUNT), np.nan, dtype=np.float32)
    if frame_axis.size:
        lookup = _frame_to_index(frame_axis)
        for row_idx, frame in enumerate(posture_frames.tolist()):
            dense_idx = lookup.get(int(frame))
            if dense_idx is None or not bool(posture_valid[row_idx]):
                continue
            values = tail_angle[row_idx]
            if np.all(np.isfinite(values)):
                dense[dense_idx, :] = values
    columns = [f"angle_{idx}" for idx in range(MEGABOUTS_TAIL_SEGMENT_COUNT)]
    return pd.DataFrame(dense, columns=columns), frame_axis, np.all(np.isfinite(dense), axis=1)


def _build_dense_traj_df(
    root: zarr.Group,
    pack: MegaboutsClassifierInputPack,
) -> tuple[object, np.ndarray, np.ndarray]:
    import pandas as pd

    track_run_path = str(pack.source_refs["track_kinematics_run"])
    track_scope = track_run_path.split("/")[-2]
    track_run_name = track_run_path.split("/")[-1]
    track_run, _, resolved_path, _ = _resolve_track_run(
        root,
        f"{track_scope}/{track_run_name}",
        track_scope=track_scope,
    )
    track_id = int(pack.parameters.get("track_id", 0))
    heading_source = str(pack.parameters.get("heading_source", DEFAULT_HEADING_SOURCE))
    _, track_arrays, _, _ = _load_track_arrays(
        track_run,
        resolved_path,
        track_id=track_id,
        heading_source=heading_source,
    )

    track_frames = np.asarray(track_arrays["frame_indices"], dtype=np.int64)
    frame_axis = _dense_frame_axis(
        track_frames,
        np.asarray(pack.window_start_frame, dtype=np.int64),
        np.asarray(pack.window_end_frame, dtype=np.int64),
    )
    dense = np.full((frame_axis.size, 3), np.nan, dtype=np.float32)
    if frame_axis.size:
        lookup = _frame_to_index(frame_axis)
        positions = np.asarray(track_arrays["positions_mm"], dtype=np.float32)
        heading = np.asarray(track_arrays["heading"], dtype=np.float32)
        sample_valid = np.asarray(track_arrays["sample_valid"], dtype=bool)
        for row_idx, frame in enumerate(track_frames.tolist()):
            dense_idx = lookup.get(int(frame))
            if dense_idx is None or not bool(sample_valid[row_idx]):
                continue
            values = np.asarray([positions[row_idx, 0], positions[row_idx, 1], heading[row_idx]], dtype=np.float32)
            if np.all(np.isfinite(values)):
                dense[dense_idx, :] = values
    return pd.DataFrame(dense, columns=["x", "y", "yaw"]), frame_axis, np.all(np.isfinite(dense), axis=1)


def _valid_fraction(valid: np.ndarray) -> np.ndarray:
    if valid.shape[0] == 0:
        return np.asarray([], dtype=np.float32)
    return np.mean(np.asarray(valid, dtype=bool), axis=1).astype(np.float32, copy=False)


def _failure_reasons_for_windows(
    tail_fraction: np.ndarray,
    traj_fraction: np.ndarray,
    max_tail_invalid: np.ndarray,
    max_traj_invalid: np.ndarray,
    traj_reference_valid: np.ndarray,
    valid_bout: np.ndarray,
    *,
    min_tail_valid_fraction: float,
    min_traj_valid_fraction: float,
    max_consecutive_invalid_frames: int,
) -> np.ndarray:
    reasons = np.full((valid_bout.shape[0],), "ok", dtype=object)
    for idx in range(valid_bout.shape[0]):
        if bool(valid_bout[idx]):
            continue
        failures: list[str] = []
        if float(tail_fraction[idx]) < float(min_tail_valid_fraction):
            failures.append("tail_valid_fraction_below_threshold")
        if float(traj_fraction[idx]) < float(min_traj_valid_fraction):
            failures.append("traj_valid_fraction_below_threshold")
        if int(max_tail_invalid[idx]) > int(max_consecutive_invalid_frames):
            failures.append("tail_consecutive_invalid_exceeds_threshold")
        if int(max_traj_invalid[idx]) > int(max_consecutive_invalid_frames):
            failures.append("traj_consecutive_invalid_exceeds_threshold")
        if not bool(traj_reference_valid[idx]):
            failures.append("traj_reference_invalid")
        reasons[idx] = "|".join(failures) if failures else "invalid"
    return reasons


def _rebuild_preprocessed_pack(
    source_pack: MegaboutsClassifierInputPack,
    *,
    tail_array: np.ndarray,
    traj_array: np.ndarray,
    tail_valid: np.ndarray,
    traj_valid: np.ndarray,
    traj_reference_valid: np.ndarray,
    runtime: MegaboutsPreprocessingRuntime,
) -> MegaboutsClassifierInputPack:
    min_tail = float(source_pack.parameters.get("min_tail_valid_fraction", DEFAULT_MIN_TAIL_VALID_FRACTION))
    min_traj = float(source_pack.parameters.get("min_traj_valid_fraction", DEFAULT_MIN_TRAJ_VALID_FRACTION))
    max_invalid = int(
        source_pack.parameters.get("max_consecutive_invalid_frames", DEFAULT_MAX_CONSECUTIVE_INVALID_FRAMES)
    )
    tail_fraction = _valid_fraction(tail_valid)
    traj_fraction = _valid_fraction(traj_valid)
    max_tail_invalid = np.asarray([_max_consecutive_false(row) for row in tail_valid], dtype=np.int32)
    max_traj_invalid = np.asarray([_max_consecutive_false(row) for row in traj_valid], dtype=np.int32)
    valid_bout = (
        (tail_fraction >= min_tail)
        & (traj_fraction >= min_traj)
        & (max_tail_invalid <= max_invalid)
        & (max_traj_invalid <= max_invalid)
        & np.asarray(traj_reference_valid, dtype=bool)
    )
    reasons = _failure_reasons_for_windows(
        tail_fraction,
        traj_fraction,
        max_tail_invalid,
        max_traj_invalid,
        np.asarray(traj_reference_valid, dtype=bool),
        valid_bout,
        min_tail_valid_fraction=min_tail,
        min_traj_valid_fraction=min_traj,
        max_consecutive_invalid_frames=max_invalid,
    )
    parameters = {
        **dict(source_pack.parameters),
        "adapter_method": COMPARISON_METHOD,
        "adapter_method_version": COMPARISON_METHOD_VERSION,
        "classifier_input_mode": MEGABOUTS_PREPROCESSED_MODE,
        "megabouts_preprocessing": True,
        "megabouts_tail_trace": "TailPreprocessingResult.angle_smooth",
        "megabouts_traj_trace": "TrajPreprocessingResult.x_smooth/y_smooth/yaw_smooth",
        "calls_megabouts": True,
        "calls_megabouts_preprocessing": True,
        "calls_megabouts_classifier": False,
        **_runtime_attrs(runtime),
    }
    source_refs = {
        **dict(source_pack.source_refs),
        "megabouts_tail_preprocessor": "megabouts.preprocessing.TailPreprocessing",
        "megabouts_traj_preprocessor": "megabouts.preprocessing.TrajPreprocessing",
    }
    return replace(
        source_pack,
        tail_array=np.asarray(tail_array, dtype=np.float32),
        traj_array=np.asarray(traj_array, dtype=np.float32),
        tail_valid=np.asarray(tail_valid, dtype=bool),
        traj_valid=np.asarray(traj_valid, dtype=bool),
        traj_reference_valid=np.asarray(traj_reference_valid, dtype=bool),
        tail_valid_fraction=tail_fraction,
        traj_valid_fraction=traj_fraction,
        max_consecutive_tail_invalid=max_tail_invalid,
        max_consecutive_traj_invalid=max_traj_invalid,
        valid_bout=valid_bout.astype(bool, copy=False),
        failure_reason=reasons,
        parameters=parameters,
        source_refs=source_refs,
    )


def build_megabouts_preprocessed_input_pack(
    root: zarr.Group,
    *,
    source_pack: Optional[MegaboutsClassifierInputPack] = None,
    runtime: Optional[MegaboutsPreprocessingRuntime] = None,
    megabouts_repo: Optional[str | Path] = None,
    tail_posture_view_run: str = "latest",
    track_kinematics_run: str = "latest",
    track_scope: str = "offline",
    track_id: int = 0,
    swim_bout_run: str = "latest",
    speed_level: str = "default",
    heading_source: str = DEFAULT_HEADING_SOURCE,
    bout_duration_s: float = DEFAULT_BOUT_DURATION_S,
    bout_duration_frames: Optional[int] = None,
    min_tail_valid_fraction: float = DEFAULT_MIN_TAIL_VALID_FRACTION,
    min_traj_valid_fraction: float = DEFAULT_MIN_TRAJ_VALID_FRACTION,
    max_consecutive_invalid_frames: int = DEFAULT_MAX_CONSECUTIVE_INVALID_FRAMES,
    align_traj_to_onset: bool = DEFAULT_ALIGN_TRAJ_TO_ONSET,
    traj_reference_index: int = DEFAULT_TRAJ_REFERENCE_INDEX,
) -> MegaboutsClassifierInputPack:
    """Build Megabouts-preprocessed classifier tensors over Palette bout windows."""

    pack = source_pack
    if pack is None:
        pack = build_megabouts_classifier_input_pack(
            root,
            tail_posture_view_run=tail_posture_view_run,
            track_kinematics_run=track_kinematics_run,
            track_scope=track_scope,
            track_id=int(track_id),
            swim_bout_run=swim_bout_run,
            speed_level=speed_level,
            heading_source=heading_source,
            bout_duration_s=float(bout_duration_s),
            bout_duration_frames=bout_duration_frames,
            min_tail_valid_fraction=float(min_tail_valid_fraction),
            min_traj_valid_fraction=float(min_traj_valid_fraction),
            max_consecutive_invalid_frames=int(max_consecutive_invalid_frames),
            align_traj_to_onset=bool(align_traj_to_onset),
            traj_reference_index=int(traj_reference_index),
        )

    resolved_runtime = runtime if runtime is not None else _load_megabouts_preprocessing_runtime(megabouts_repo)
    fps = float(pack.parameters.get("fps", 0.0))
    rounded_fps = int(round(fps))
    if rounded_fps <= 0 or not math.isclose(fps, float(rounded_fps), rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"Megabouts preprocessing requires integer fps; Palette resolved fps={fps!r}.")

    tail_df, tail_frames, _ = _build_dense_tail_df(root, pack)
    tail_cfg = resolved_runtime.tail_preprocessing_config_class(fps=rounded_fps)
    tail_result = resolved_runtime.tail_preprocessing_class(tail_cfg).preprocess_tail_df(tail_df)
    tail_values = np.asarray(tail_result.angle_smooth, dtype=np.float32)
    tail_no_tracking = np.asarray(tail_result.no_tracking, dtype=bool)
    tail_series_valid = (~tail_no_tracking) & np.all(np.isfinite(tail_values), axis=1)
    tail_array, tail_valid = _window_extract(
        tail_values,
        tail_series_valid,
        tail_frames,
        pack.window_start_frame,
        int(pack.tail_array.shape[2]),
    )

    traj_df, traj_frames, _ = _build_dense_traj_df(root, pack)
    traj_cfg = resolved_runtime.traj_preprocessing_config_class(fps=rounded_fps)
    traj_result = resolved_runtime.traj_preprocessing_class(traj_cfg).preprocess_traj_df(traj_df)
    traj_values = np.stack(
        [
            np.asarray(traj_result.x_smooth, dtype=np.float32),
            np.asarray(traj_result.y_smooth, dtype=np.float32),
            np.asarray(traj_result.yaw_smooth, dtype=np.float32),
        ],
        axis=1,
    )
    traj_no_tracking = np.asarray(traj_result.no_tracking, dtype=bool)
    traj_series_valid = (~traj_no_tracking) & np.all(np.isfinite(traj_values), axis=1)
    traj_array, traj_valid = _window_extract(
        traj_values,
        traj_series_valid,
        traj_frames,
        pack.window_start_frame,
        int(pack.traj_array.shape[2]),
    )

    if bool(pack.parameters.get("traj_alignment") == "onset_translation_rotation"):
        traj_array, traj_reference_valid = _align_traj_array_to_reference(
            traj_array,
            traj_valid,
            reference_index=int(pack.parameters.get("traj_reference_index", DEFAULT_TRAJ_REFERENCE_INDEX)),
        )
    else:
        ref = int(pack.parameters.get("traj_reference_index", DEFAULT_TRAJ_REFERENCE_INDEX))
        if ref < 0 or ref >= int(pack.traj_array.shape[2]):
            raise ValueError(f"trajectory reference index {ref} is outside window length {pack.traj_array.shape[2]}.")
        traj_reference_valid = traj_valid[:, ref] & np.all(np.isfinite(traj_array[:, :, ref]), axis=1)

    return _rebuild_preprocessed_pack(
        pack,
        tail_array=tail_array,
        traj_array=traj_array,
        tail_valid=tail_valid,
        traj_valid=traj_valid,
        traj_reference_valid=traj_reference_valid,
        runtime=resolved_runtime,
    )


def _wrap_signed_radians(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) + math.pi) % (2.0 * math.pi) - math.pi


def _array_stats(
    a: np.ndarray,
    b: np.ndarray,
    mask: np.ndarray,
    *,
    angular_radians: bool = False,
) -> dict[str, object]:
    values_a = np.asarray(a, dtype=np.float64)
    values_b = np.asarray(b, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(values_a) & np.isfinite(values_b)
    n = int(np.count_nonzero(valid))
    if n == 0:
        return {
            "n": 0,
            "rmse": math.nan,
            "mean_abs": math.nan,
            "median_abs": math.nan,
            "max_abs": math.nan,
            "corr": math.nan,
        }
    if angular_radians:
        delta = _wrap_signed_radians(values_b[valid] - values_a[valid])
        corr_a = np.unwrap(values_a[valid])
        corr_b = np.unwrap(values_a[valid] + delta)
    else:
        delta = values_b[valid] - values_a[valid]
        corr_a = values_a[valid]
        corr_b = values_b[valid]
    if n >= 2 and float(np.std(corr_a)) > 0.0 and float(np.std(corr_b)) > 0.0:
        corr = float(np.corrcoef(corr_a, corr_b)[0, 1])
    else:
        corr = math.nan
    return {
        "n": n,
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "median_abs": float(np.median(np.abs(delta))),
        "max_abs": float(np.max(np.abs(delta))),
        "corr": corr,
    }


def _compare_window_arrays(
    palette_array: np.ndarray,
    megabouts_array: np.ndarray,
    palette_valid: np.ndarray,
    megabouts_valid: np.ndarray,
    *,
    channel_names: Sequence[str],
    angular_channel_names: Sequence[str] = (),
    common_bout_mask: np.ndarray,
) -> dict[str, object]:
    sample_mask = (
        np.asarray(common_bout_mask, dtype=bool)[:, None, None]
        & np.asarray(palette_valid, dtype=bool)[:, None, :]
        & np.asarray(megabouts_valid, dtype=bool)[:, None, :]
    )
    angular_names = {str(name) for name in angular_channel_names}
    linear_channel_idxs = [
        idx for idx, name in enumerate(channel_names) if str(name) not in angular_names
    ]
    if linear_channel_idxs:
        overall = _array_stats(
            np.asarray(palette_array)[:, linear_channel_idxs, :],
            np.asarray(megabouts_array)[:, linear_channel_idxs, :],
            sample_mask,
        )
    else:
        overall = _array_stats(palette_array, megabouts_array, sample_mask)
    per_channel: dict[str, object] = {}
    for channel_idx, name in enumerate(channel_names):
        per_channel[str(name)] = _array_stats(
            np.asarray(palette_array)[:, channel_idx, :],
            np.asarray(megabouts_array)[:, channel_idx, :],
            sample_mask[:, 0, :],
            angular_radians=str(name) in angular_names,
        )
    return {
        "overall": overall,
        "per_channel": per_channel,
    }


def _classification_agreement(
    palette_pack: MegaboutsClassifierInputPack,
    megabouts_pack: MegaboutsClassifierInputPack,
    *,
    exclude_cs: bool,
    device: str,
    megabouts_repo: Optional[str | Path],
) -> dict[str, object]:
    runtime = _load_megabouts_runtime(megabouts_repo)
    palette_result = classify_megabouts_input_pack(
        palette_pack,
        exclude_cs=exclude_cs,
        device=device,
        runtime=runtime,
    )
    megabouts_result = classify_megabouts_input_pack(
        megabouts_pack,
        exclude_cs=exclude_cs,
        device=device,
        runtime=runtime,
    )
    n_bouts = int(palette_pack.source_bout_id.shape[0])
    palette_cat = np.full((n_bouts,), -1, dtype=np.int32)
    megabouts_cat = np.full((n_bouts,), -1, dtype=np.int32)
    palette_prob = np.full((n_bouts,), np.nan, dtype=np.float32)
    megabouts_prob = np.full((n_bouts,), np.nan, dtype=np.float32)
    palette_cat[palette_result.classified_indices] = palette_result.classif_results["cat"]
    megabouts_cat[megabouts_result.classified_indices] = megabouts_result.classif_results["cat"]
    palette_prob[palette_result.classified_indices] = palette_result.classif_results["proba"]
    megabouts_prob[megabouts_result.classified_indices] = megabouts_result.classif_results["proba"]
    common = np.asarray(palette_pack.valid_bout, dtype=bool) & np.asarray(megabouts_pack.valid_bout, dtype=bool)
    common_classified = common & (palette_cat >= 0) & (megabouts_cat >= 0)
    n_common = int(np.count_nonzero(common_classified))
    agreement = (
        float(np.mean(palette_cat[common_classified] == megabouts_cat[common_classified]))
        if n_common
        else math.nan
    )
    prob_delta = megabouts_prob[common_classified] - palette_prob[common_classified]
    return {
        "calls_megabouts_classifier": True,
        "palette_classified_count": int(palette_result.classified_indices.size),
        "megabouts_preprocessed_classified_count": int(megabouts_result.classified_indices.size),
        "common_classified_count": n_common,
        "category_agreement_fraction": agreement,
        "mean_probability_delta": float(np.nanmean(prob_delta)) if prob_delta.size else math.nan,
        "mean_abs_probability_delta": float(np.nanmean(np.abs(prob_delta))) if prob_delta.size else math.nan,
    }


def compare_megabouts_preprocessing_with_palette(
    root: zarr.Group,
    *,
    megabouts_repo: Optional[str | Path] = None,
    classify: bool = False,
    exclude_cs: bool = False,
    device: str = "auto",
    runtime: Optional[MegaboutsPreprocessingRuntime] = None,
    tail_posture_view_run: str = "latest",
    track_kinematics_run: str = "latest",
    track_scope: str = "offline",
    track_id: int = 0,
    swim_bout_run: str = "latest",
    speed_level: str = "default",
    heading_source: str = DEFAULT_HEADING_SOURCE,
    bout_duration_s: float = DEFAULT_BOUT_DURATION_S,
    bout_duration_frames: Optional[int] = None,
    min_tail_valid_fraction: float = DEFAULT_MIN_TAIL_VALID_FRACTION,
    min_traj_valid_fraction: float = DEFAULT_MIN_TRAJ_VALID_FRACTION,
    max_consecutive_invalid_frames: int = DEFAULT_MAX_CONSECUTIVE_INVALID_FRAMES,
    align_traj_to_onset: bool = DEFAULT_ALIGN_TRAJ_TO_ONSET,
    traj_reference_index: int = DEFAULT_TRAJ_REFERENCE_INDEX,
) -> dict[str, object]:
    """Compare Palette-prepared and Megabouts-preprocessed classifier inputs."""

    palette_pack = build_megabouts_classifier_input_pack(
        root,
        tail_posture_view_run=tail_posture_view_run,
        track_kinematics_run=track_kinematics_run,
        track_scope=track_scope,
        track_id=int(track_id),
        swim_bout_run=swim_bout_run,
        speed_level=speed_level,
        heading_source=heading_source,
        bout_duration_s=float(bout_duration_s),
        bout_duration_frames=bout_duration_frames,
        min_tail_valid_fraction=float(min_tail_valid_fraction),
        min_traj_valid_fraction=float(min_traj_valid_fraction),
        max_consecutive_invalid_frames=int(max_consecutive_invalid_frames),
        align_traj_to_onset=bool(align_traj_to_onset),
        traj_reference_index=int(traj_reference_index),
    )
    megabouts_pack = build_megabouts_preprocessed_input_pack(
        root,
        source_pack=palette_pack,
        runtime=runtime,
        megabouts_repo=megabouts_repo,
    )
    common_bout_mask = np.asarray(palette_pack.valid_bout, dtype=bool) & np.asarray(
        megabouts_pack.valid_bout, dtype=bool
    )
    tail_comparison = _compare_window_arrays(
        palette_pack.tail_array,
        megabouts_pack.tail_array,
        palette_pack.tail_valid,
        megabouts_pack.tail_valid,
        channel_names=[f"angle_{idx}" for idx in range(int(palette_pack.tail_array.shape[1]))],
        common_bout_mask=common_bout_mask,
    )
    traj_comparison = _compare_window_arrays(
        palette_pack.traj_array,
        megabouts_pack.traj_array,
        palette_pack.traj_valid,
        megabouts_pack.traj_valid,
        channel_names=["x_mm_onset_aligned", "y_mm_onset_aligned", "yaw_rad_onset_aligned"],
        angular_channel_names=["yaw_rad_onset_aligned"],
        common_bout_mask=common_bout_mask,
    )

    report: dict[str, object] = {
        "status": "ok",
        "method": COMPARISON_METHOD,
        "method_version": COMPARISON_METHOD_VERSION,
        "mutates_archive": False,
        "calls_megabouts_preprocessing": True,
        "calls_megabouts_classifier": bool(classify),
        "comparison_scope": "same_palette_swim_bout_windows",
        "palette_input_summary": summarize_input_pack(palette_pack),
        "megabouts_preprocessed_input_summary": summarize_input_pack(megabouts_pack),
        "source_bout_count": int(palette_pack.source_bout_id.shape[0]),
        "palette_valid_bout_count": int(np.count_nonzero(palette_pack.valid_bout)),
        "megabouts_preprocessed_valid_bout_count": int(np.count_nonzero(megabouts_pack.valid_bout)),
        "common_valid_bout_count": int(np.count_nonzero(common_bout_mask)),
        "tail_validity_disagreement_count": int(np.count_nonzero(palette_pack.tail_valid != megabouts_pack.tail_valid)),
        "traj_validity_disagreement_count": int(np.count_nonzero(palette_pack.traj_valid != megabouts_pack.traj_valid)),
        "tail_angle_comparison_rad": tail_comparison,
        "trajectory_comparison": traj_comparison,
        "notes": [
            "Palette input uses persisted tail_angle_rad and track positions/headings sampled into fixed windows.",
            "Megabouts-preprocessed input runs TailPreprocessing.angle_smooth and TrajPreprocessing x/y/yaw_smooth over full dense time series before sampling the same windows.",
            "This report is input comparison only unless --classify is provided.",
        ],
    }
    if classify:
        report["classification_comparison"] = _classification_agreement(
            palette_pack,
            megabouts_pack,
            exclude_cs=bool(exclude_cs),
            device=str(device),
            megabouts_repo=megabouts_repo,
        )
    return dict(_json_safe(report))


def compare_megabouts_preprocessing_with_palette_zarr(
    zarr_path: str | Path,
    **kwargs: object,
) -> dict[str, object]:
    root = open_zarr_root(zarr_path, mode="r")
    return compare_megabouts_preprocessing_with_palette(root, **kwargs)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare existing Palette Megabouts classifier inputs with inputs produced by "
            "running Megabouts preprocessing over the same Palette bout windows. Read-only."
        )
    )
    parser.add_argument("zarr_path", type=Path, help="Palette zarr archive.")
    parser.add_argument("--megabouts-repo", default=None, help="Optional local Megabouts checkout.")
    parser.add_argument("--tail-posture-view-run", default="latest", help="analysis/tail_posture_view_runs/<run>.")
    parser.add_argument("--track-kinematics-run", default="latest", help="analysis/track_kinematics_runs run.")
    parser.add_argument("--track-scope", default="offline", help="Track kinematics scope for non-path run names.")
    parser.add_argument("--track-id", type=int, default=0, help="Track id to use.")
    parser.add_argument("--swim-bout-run", default="latest", help="analysis/swim_bout_runs/<run>.")
    parser.add_argument("--speed-level", default="default", help="Swim-bout speed level or 'default'.")
    parser.add_argument("--heading-source", default=DEFAULT_HEADING_SOURCE, help="Track heading array in radians.")
    parser.add_argument("--bout-duration-s", type=float, default=DEFAULT_BOUT_DURATION_S)
    parser.add_argument("--bout-duration-frames", type=int, default=None)
    parser.add_argument("--min-tail-valid-fraction", type=float, default=DEFAULT_MIN_TAIL_VALID_FRACTION)
    parser.add_argument("--min-traj-valid-fraction", type=float, default=DEFAULT_MIN_TRAJ_VALID_FRACTION)
    parser.add_argument("--max-consecutive-invalid-frames", type=int, default=DEFAULT_MAX_CONSECUTIVE_INVALID_FRAMES)
    parser.add_argument(
        "--no-align-traj-to-onset",
        action="store_true",
        help="Disable Megabouts-style onset-frame translation/rotation for trajectory windows.",
    )
    parser.add_argument("--traj-reference-index", type=int, default=DEFAULT_TRAJ_REFERENCE_INDEX)
    parser.add_argument("--classify", action="store_true", help="Also compare Megabouts classifier outputs.")
    parser.add_argument("--exclude-cs", action="store_true", help="Pass exclude_CS=True to Megabouts classifier.")
    parser.add_argument("--device", default="auto", help="Megabouts classifier device when --classify is set.")
    parser.add_argument("--json", action="store_true", help="Emit compact JSON.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    summary = compare_megabouts_preprocessing_with_palette_zarr(
        args.zarr_path,
        megabouts_repo=args.megabouts_repo,
        classify=bool(args.classify),
        exclude_cs=bool(args.exclude_cs),
        device=str(args.device),
        tail_posture_view_run=args.tail_posture_view_run,
        track_kinematics_run=args.track_kinematics_run,
        track_scope=args.track_scope,
        track_id=int(args.track_id),
        swim_bout_run=args.swim_bout_run,
        speed_level=args.speed_level,
        heading_source=args.heading_source,
        bout_duration_s=float(args.bout_duration_s),
        bout_duration_frames=args.bout_duration_frames,
        min_tail_valid_fraction=float(args.min_tail_valid_fraction),
        min_traj_valid_fraction=float(args.min_traj_valid_fraction),
        max_consecutive_invalid_frames=int(args.max_consecutive_invalid_frames),
        align_traj_to_onset=not bool(args.no_align_traj_to_onset),
        traj_reference_index=int(args.traj_reference_index),
    )
    print(json.dumps(summary, indent=None if args.json else 2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
