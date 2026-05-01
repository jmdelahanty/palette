"""Dry-run Megabouts classifier input construction from Palette Zarr runs."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import zarr

from fisheye.analysis.chaser_state_interpolator import load_structured_dataset
from fisheye.analysis.detect_bouts_multi_level import normalize_speed_level
from fisheye.utils.zarr_io import open_zarr_root

DEFAULT_TAIL_POSTURE_VIEW_FAMILY = "megabouts_compatible"
DEFAULT_BOUT_DURATION_S = 0.2
DEFAULT_MIN_TAIL_VALID_FRACTION = 0.90
DEFAULT_MIN_TRAJ_VALID_FRACTION = 0.90
DEFAULT_MAX_CONSECUTIVE_INVALID_FRAMES = 2
DEFAULT_HEADING_SOURCE = "smoothed_heading_radians"
MEGABOUTS_TAIL_SEGMENT_COUNT = 10
MEGABOUTS_MAX_CLASSIFIER_WINDOW_FRAMES = 140


@dataclass(frozen=True)
class MegaboutsClassifierInputPack:
    """Fixed-window classifier tensors plus validity summaries.

    This object is intentionally in-memory. It is a dry-run readiness surface
    for later optional Megabouts execution, not a stored derived-analysis run.
    """

    tail_array: np.ndarray
    traj_array: np.ndarray
    tail_valid: np.ndarray
    traj_valid: np.ndarray
    source_bout_id: np.ndarray
    source_start_frame: np.ndarray
    source_end_frame: np.ndarray
    window_start_frame: np.ndarray
    window_end_frame: np.ndarray
    tail_valid_fraction: np.ndarray
    traj_valid_fraction: np.ndarray
    max_consecutive_tail_invalid: np.ndarray
    max_consecutive_traj_invalid: np.ndarray
    valid_bout: np.ndarray
    failure_reason: np.ndarray
    source_refs: dict[str, str]
    parameters: dict[str, object]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _require_group(parent: zarr.Group, name: str) -> zarr.Group:
    group = parent.get(name)
    if not isinstance(group, zarr.Group):
        raise ValueError(f"Missing required group: {parent.name}/{name}")
    return group


def _require_array(group: zarr.Group, name: str) -> zarr.Array:
    arr = group.get(name)
    if arr is None:
        raise ValueError(f"Missing required array: {group.name}/{name}")
    return arr


def _duration_seconds_to_frames(duration_s: float, fps: float) -> int:
    duration = float(duration_s)
    rate = float(fps)
    if duration <= 0.0:
        raise ValueError(f"bout_duration_s must be > 0, got {duration_s!r}.")
    if rate <= 0.0:
        raise ValueError(f"fps must be > 0, got {fps!r}.")
    return max(1, int(math.ceil(duration * rate - 1e-9)))


def _resolve_tail_posture_view_run(
    root: zarr.Group,
    run_name: str,
    *,
    view_family: str = DEFAULT_TAIL_POSTURE_VIEW_FAMILY,
) -> tuple[zarr.Group, str, str]:
    analysis = _require_group(root, "analysis")
    parent = _require_group(analysis, "tail_posture_view_runs")
    spec = str(run_name or "latest").strip().strip("/")
    parts = spec.split("/")
    if spec.startswith("analysis/tail_posture_view_runs/") and len(parts) >= 3:
        resolved = parts[2]
    elif spec == "latest":
        resolved = parent.attrs.get(f"latest_{view_family}") or parent.attrs.get("latest")
    else:
        resolved = spec
    if not resolved or resolved not in parent:
        raise ValueError(f"Tail posture view run {run_name!r} not found.")
    path = f"analysis/tail_posture_view_runs/{resolved}"
    return parent[resolved], str(resolved), path


def _resolve_track_run(
    root: zarr.Group,
    run_name: str,
    *,
    track_scope: str,
) -> tuple[zarr.Group, str, str, str]:
    analysis = _require_group(root, "analysis")
    parent = _require_group(analysis, "track_kinematics_runs")
    spec = str(run_name or "latest").strip().strip("/")
    parts = spec.split("/")
    if spec.startswith("analysis/track_kinematics_runs/") and len(parts) >= 4:
        scope = parts[2]
        resolved = parts[3]
    elif len(parts) == 2 and parts[0] in parent:
        scope = parts[0]
        resolved = parts[1]
    else:
        scope = str(track_scope)
        if scope not in parent:
            raise ValueError(f"Track kinematics scope {scope!r} not found.")
        resolved = parent[scope].attrs.get("latest") if spec == "latest" else spec
    if not resolved or scope not in parent or resolved not in parent[scope]:
        raise ValueError(f"Track kinematics run {run_name!r} not found under scope {scope!r}.")
    path = f"analysis/track_kinematics_runs/{scope}/{resolved}"
    return parent[scope][resolved], str(resolved), path, str(scope)


def _resolve_swim_bout_level(
    root: zarr.Group,
    run_name: str,
    *,
    speed_level: str,
) -> tuple[zarr.Group, zarr.Group, str, str, str]:
    analysis = _require_group(root, "analysis")
    parent = _require_group(analysis, "swim_bout_runs")
    spec = str(run_name or "latest").strip().strip("/")
    if spec.startswith("analysis/swim_bout_runs/"):
        parts = spec.split("/")
        resolved = parts[2] if len(parts) >= 3 else ""
    else:
        resolved = parent.attrs.get("latest") if spec == "latest" else spec
    if not resolved or resolved not in parent:
        raise ValueError(f"Swim-bout run {run_name!r} not found.")

    run = parent[resolved]
    level_spec = str(speed_level or "default").strip()
    if level_spec in {"default", "latest"}:
        level = str(run.attrs.get("default_level") or "speed_filtered")
    else:
        level = normalize_speed_level(level_spec)
    if level not in run:
        raise ValueError(f"Speed level {level!r} not found in swim-bout run {resolved!r}.")
    path = f"analysis/swim_bout_runs/{resolved}/{level}"
    return run, run[level], str(resolved), str(level), path


def _frame_to_index(frames: np.ndarray) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for idx, frame in enumerate(np.asarray(frames, dtype=np.int64).tolist()):
        mapping.setdefault(int(frame), int(idx))
    return mapping


def _max_consecutive_false(mask: np.ndarray) -> int:
    max_run = 0
    current = 0
    for value in np.asarray(mask, dtype=bool).tolist():
        if value:
            current = 0
        else:
            current += 1
            max_run = max(max_run, current)
    return int(max_run)


def _decode_reason_value(value: Any) -> str:
    if isinstance(value, bytes):
        return value.split(b"\x00", 1)[0].decode("utf-8", "replace")
    if isinstance(value, str):
        return value
    arr = np.asarray(value)
    if arr.ndim == 0:
        item = arr.item()
        if isinstance(item, bytes):
            return item.split(b"\x00", 1)[0].decode("utf-8", "replace")
        return str(item)
    if arr.dtype.kind in {"S", "U", "O"}:
        return str(arr.reshape(-1)[0])
    return bytes(arr.astype(np.uint8, copy=False).tolist()).split(b"\x00", 1)[0].decode("utf-8", "replace")


def _load_reason_array(group: zarr.Group, names: Sequence[str]) -> Optional[np.ndarray]:
    for name in names:
        if name in group:
            return np.asarray(group[name][:])
    return None


def _resolve_start_frames(bouts: np.ndarray, fps: float) -> np.ndarray:
    names = bouts.dtype.names or ()
    if "start_frame" in names:
        return np.asarray(bouts["start_frame"], dtype=np.int64)
    if "start_time_s" in names:
        return np.rint(np.asarray(bouts["start_time_s"], dtype=np.float64) * float(fps)).astype(np.int64)
    raise ValueError("Bouts table must include start_frame or start_time_s.")


def _resolve_end_frames(bouts: np.ndarray, fps: float) -> np.ndarray:
    names = bouts.dtype.names or ()
    if "end_frame" in names:
        return np.asarray(bouts["end_frame"], dtype=np.int64)
    if "end_time_s" in names:
        return np.rint(np.asarray(bouts["end_time_s"], dtype=np.float64) * float(fps)).astype(np.int64)
    return np.full((bouts.shape[0],), -1, dtype=np.int64)


def _resolve_bout_ids(bouts: np.ndarray) -> np.ndarray:
    names = bouts.dtype.names or ()
    if "bout_id" in names:
        return np.asarray(bouts["bout_id"], dtype=np.int64)
    return np.arange(int(bouts.shape[0]), dtype=np.int64)


def _load_track_arrays(
    track_run: zarr.Group,
    track_path: str,
    *,
    track_id: int,
    heading_source: str,
) -> tuple[zarr.Group, dict[str, np.ndarray], dict[str, str], float]:
    tracks = _require_group(track_run, "tracks")
    track_name = f"id_{int(track_id)}"
    if track_name not in tracks:
        raise ValueError(f"Track {track_name!r} not found in {track_path}/tracks.")
    track = tracks[track_name]
    track_path_full = f"{track_path}/tracks/{track_name}"
    frames = np.asarray(_require_array(track, "frame_indices")[:], dtype=np.int64)
    positions_mm = np.asarray(_require_array(track, "positions_mm")[:], dtype=np.float32)
    heading = np.asarray(_require_array(track, heading_source)[:], dtype=np.float32)
    if positions_mm.shape != (frames.shape[0], 2):
        raise ValueError(f"positions_mm shape {positions_mm.shape} does not match frames length {frames.shape[0]}.")
    if heading.shape[0] != frames.shape[0]:
        raise ValueError(f"{heading_source} length {heading.shape[0]} does not match frames length {frames.shape[0]}.")
    if "sample_valid" in track:
        sample_valid = np.asarray(track["sample_valid"][:], dtype=bool)
    else:
        sample_valid = np.ones((frames.shape[0],), dtype=bool)
    if sample_valid.shape[0] != frames.shape[0]:
        raise ValueError(f"sample_valid length {sample_valid.shape[0]} does not match frames length {frames.shape[0]}.")
    fps = float(track_run.attrs.get("fps", 0.0))
    arrays = {
        "frame_indices": frames,
        "positions_mm": positions_mm,
        "heading": heading,
        "sample_valid": sample_valid,
    }
    refs = {
        "track_group": track_path_full,
        "track_frame_indices": f"{track_path_full}/frame_indices",
        "positions_mm": f"{track_path_full}/positions_mm",
        "heading": f"{track_path_full}/{heading_source}",
        "sample_valid": f"{track_path_full}/sample_valid" if "sample_valid" in track else "implicit_all_true",
    }
    return track, arrays, refs, fps


def build_megabouts_classifier_input_pack(
    root: zarr.Group,
    *,
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
) -> MegaboutsClassifierInputPack:
    """Build fixed-window arrays for optional Megabouts classification.

    The returned tensors follow the classifier-facing convention documented in
    ``docs/megabouts_direct_integration_design.md``:

    - ``tail_array`` has shape ``(n_bouts, 10, window_frames)``.
    - ``traj_array`` has shape ``(n_bouts, 3, window_frames)`` with channels
      ``x_mm``, ``y_mm``, and selected heading/yaw in radians.
    """

    posture, posture_name, posture_path = _resolve_tail_posture_view_run(root, tail_posture_view_run)
    track_run, track_name, track_path, resolved_scope = _resolve_track_run(
        root,
        track_kinematics_run,
        track_scope=track_scope,
    )
    _, bout_level, bout_run_name, resolved_level, bout_level_path = _resolve_swim_bout_level(
        root,
        swim_bout_run,
        speed_level=speed_level,
    )

    track_group, track_arrays, track_refs, fps = _load_track_arrays(
        track_run,
        track_path,
        track_id=int(track_id),
        heading_source=str(heading_source),
    )
    if fps <= 0.0:
        fps = float(bout_level.attrs.get("fps", 0.0))
    if fps <= 0.0:
        raise ValueError("Unable to resolve positive fps from track run or swim-bout level attrs.")
    window_frames = (
        int(bout_duration_frames)
        if bout_duration_frames is not None
        else _duration_seconds_to_frames(float(bout_duration_s), fps)
    )
    if window_frames <= 0:
        raise ValueError(f"bout_duration_frames must be > 0, got {bout_duration_frames!r}.")
    if window_frames > MEGABOUTS_MAX_CLASSIFIER_WINDOW_FRAMES:
        raise ValueError(
            "Megabouts classifier input windows are capped at "
            f"{MEGABOUTS_MAX_CLASSIFIER_WINDOW_FRAMES} frames, got {window_frames}."
        )

    posture_frames = np.asarray(_require_array(posture, "frame_index")[:], dtype=np.int64)
    posture_valid = np.asarray(_require_array(posture, "valid")[:], dtype=bool)
    tail_angle = np.asarray(_require_array(posture, "tail_angle_rad")[:], dtype=np.float32)
    if tail_angle.ndim != 2:
        raise ValueError(f"tail_angle_rad must be 2D, got shape {tail_angle.shape}.")
    if posture_frames.shape[0] != tail_angle.shape[0] or posture_valid.shape[0] != tail_angle.shape[0]:
        raise ValueError("Tail posture frame_index, valid, and tail_angle_rad row counts must match.")
    angle_count = int(tail_angle.shape[1])
    if angle_count != MEGABOUTS_TAIL_SEGMENT_COUNT:
        raise ValueError(
            "Megabouts classifier input requires "
            f"{MEGABOUTS_TAIL_SEGMENT_COUNT} tail-angle channels, got {angle_count}."
        )

    bouts, _ = load_structured_dataset(bout_level, "bouts")
    start_frames = _resolve_start_frames(bouts, fps)
    end_frames = _resolve_end_frames(bouts, fps)
    bout_ids = _resolve_bout_ids(bouts)
    n_bouts = int(start_frames.shape[0])

    tail_array = np.full((n_bouts, angle_count, window_frames), np.nan, dtype=np.float32)
    traj_array = np.full((n_bouts, 3, window_frames), np.nan, dtype=np.float32)
    tail_valid = np.zeros((n_bouts, window_frames), dtype=bool)
    traj_valid = np.zeros((n_bouts, window_frames), dtype=bool)
    window_start = np.asarray(start_frames, dtype=np.int64)
    window_end = window_start + int(window_frames) - 1

    posture_lookup = _frame_to_index(posture_frames)
    track_lookup = _frame_to_index(track_arrays["frame_indices"])

    for bout_idx, start in enumerate(window_start.tolist()):
        frames = np.arange(int(start), int(start) + int(window_frames), dtype=np.int64)
        for sample_idx, frame in enumerate(frames.tolist()):
            posture_idx = posture_lookup.get(int(frame))
            if posture_idx is not None and bool(posture_valid[posture_idx]):
                values = tail_angle[posture_idx]
                if np.all(np.isfinite(values)):
                    tail_array[bout_idx, :, sample_idx] = values
                    tail_valid[bout_idx, sample_idx] = True

            track_idx = track_lookup.get(int(frame))
            if track_idx is not None and bool(track_arrays["sample_valid"][track_idx]):
                x_mm, y_mm = track_arrays["positions_mm"][track_idx]
                yaw = track_arrays["heading"][track_idx]
                values = np.asarray([x_mm, y_mm, yaw], dtype=np.float32)
                if np.all(np.isfinite(values)):
                    traj_array[bout_idx, :, sample_idx] = values
                    traj_valid[bout_idx, sample_idx] = True

    tail_fraction = np.mean(tail_valid, axis=1).astype(np.float32, copy=False) if n_bouts else np.asarray([], dtype=np.float32)
    traj_fraction = np.mean(traj_valid, axis=1).astype(np.float32, copy=False) if n_bouts else np.asarray([], dtype=np.float32)
    max_tail_invalid = np.asarray([_max_consecutive_false(row) for row in tail_valid], dtype=np.int32)
    max_traj_invalid = np.asarray([_max_consecutive_false(row) for row in traj_valid], dtype=np.int32)

    valid_bout = (
        (tail_fraction >= float(min_tail_valid_fraction))
        & (traj_fraction >= float(min_traj_valid_fraction))
        & (max_tail_invalid <= int(max_consecutive_invalid_frames))
        & (max_traj_invalid <= int(max_consecutive_invalid_frames))
    )
    reasons = np.full((n_bouts,), "ok", dtype=object)
    for idx in range(n_bouts):
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
        reasons[idx] = "|".join(failures) if failures else "invalid"

    source_refs = {
        "tail_posture_view_run": posture_path,
        "tail_angle_rad": f"{posture_path}/tail_angle_rad",
        "tail_valid": f"{posture_path}/valid",
        "track_kinematics_run": track_path,
        **track_refs,
        "swim_bout_level": bout_level_path,
        "bouts": f"{bout_level_path}/bouts",
    }
    parameters = {
        "adapter_method": "palette_megabouts_classifier_input_dry_run",
        "source_mode": "palette_bouts",
        "tail_posture_view_run": posture_name,
        "track_kinematics_scope": resolved_scope,
        "track_kinematics_run": track_name,
        "track_id": int(track_id),
        "swim_bout_run": bout_run_name,
        "speed_level": resolved_level,
        "heading_source": str(heading_source),
        "fps": float(fps),
        "bout_duration_s": float(window_frames) / float(fps),
        "bout_duration_frames": int(window_frames),
        "window_policy": "start_frame_fixed_duration",
        "tail_array_shape": list(tail_array.shape),
        "traj_array_shape": list(traj_array.shape),
        "min_tail_valid_fraction": float(min_tail_valid_fraction),
        "min_traj_valid_fraction": float(min_traj_valid_fraction),
        "max_consecutive_invalid_frames": int(max_consecutive_invalid_frames),
        "mutates_archive": False,
        "calls_megabouts": False,
    }
    return MegaboutsClassifierInputPack(
        tail_array=tail_array,
        traj_array=traj_array,
        tail_valid=tail_valid,
        traj_valid=traj_valid,
        source_bout_id=bout_ids,
        source_start_frame=start_frames,
        source_end_frame=end_frames,
        window_start_frame=window_start,
        window_end_frame=window_end,
        tail_valid_fraction=tail_fraction,
        traj_valid_fraction=traj_fraction,
        max_consecutive_tail_invalid=max_tail_invalid,
        max_consecutive_traj_invalid=max_traj_invalid,
        valid_bout=valid_bout.astype(bool),
        failure_reason=reasons,
        source_refs=source_refs,
        parameters=parameters,
    )


def summarize_input_pack(pack: MegaboutsClassifierInputPack) -> dict[str, object]:
    """Return a JSON-safe summary of a dry-run classifier input pack."""

    reason_counts: dict[str, int] = {}
    for reason in np.asarray(pack.failure_reason, dtype=object).tolist():
        key = str(reason or "")
        reason_counts[key] = int(reason_counts.get(key, 0) + 1)
    n_bouts = int(pack.valid_bout.shape[0])
    valid_count = int(np.count_nonzero(pack.valid_bout))
    summary = {
        "status": "ok",
        "mutates_archive": False,
        "calls_megabouts": False,
        "n_bouts": n_bouts,
        "valid_bout_count": valid_count,
        "invalid_bout_count": int(n_bouts - valid_count),
        "tail_array_shape": list(pack.tail_array.shape),
        "traj_array_shape": list(pack.traj_array.shape),
        "tail_valid_fraction_min": float(np.nanmin(pack.tail_valid_fraction)) if n_bouts else math.nan,
        "tail_valid_fraction_mean": float(np.nanmean(pack.tail_valid_fraction)) if n_bouts else math.nan,
        "traj_valid_fraction_min": float(np.nanmin(pack.traj_valid_fraction)) if n_bouts else math.nan,
        "traj_valid_fraction_mean": float(np.nanmean(pack.traj_valid_fraction)) if n_bouts else math.nan,
        "failure_reason_counts": reason_counts,
        "parameters": pack.parameters,
        "source_refs": pack.source_refs,
    }
    return dict(_json_safe(summary))


def diagnose_input_pack_invalid_windows(
    root: zarr.Group,
    pack: MegaboutsClassifierInputPack,
    *,
    max_examples: int = 12,
) -> dict[str, object]:
    """Explain invalid classifier windows without mutating the archive."""

    posture = root[pack.source_refs["tail_posture_view_run"]]
    posture_frames = np.asarray(_require_array(posture, "frame_index")[:], dtype=np.int64)
    posture_valid = np.asarray(_require_array(posture, "valid")[:], dtype=bool)
    tail_angle = np.asarray(_require_array(posture, "tail_angle_rad")[:], dtype=np.float32)
    posture_reasons = _load_reason_array(posture, ("failure_reason_bytes", "reason_bytes"))
    posture_lookup = _frame_to_index(posture_frames)

    track = root[pack.source_refs["track_group"]]
    track_frames = np.asarray(_require_array(track, "frame_indices")[:], dtype=np.int64)
    positions_mm = np.asarray(_require_array(track, "positions_mm")[:], dtype=np.float32)
    heading_path = pack.source_refs["heading"].split("/")[-1]
    heading = np.asarray(_require_array(track, heading_path)[:], dtype=np.float32)
    if "sample_valid" in track:
        track_valid = np.asarray(track["sample_valid"][:], dtype=bool)
    else:
        track_valid = np.ones((track_frames.shape[0],), dtype=bool)
    track_reasons = _load_reason_array(track, ("failure_reason_bytes", "reason_bytes"))
    track_lookup = _frame_to_index(track_frames)

    invalid_idxs = np.flatnonzero(~pack.valid_bout)
    failure_reason_counts = Counter(str(pack.failure_reason[idx]) for idx in invalid_idxs.tolist())
    tail_issue_counts: Counter[str] = Counter()
    traj_issue_counts: Counter[str] = Counter()
    posture_reason_counts: Counter[str] = Counter()
    track_reason_counts: Counter[str] = Counter()
    examples: list[dict[str, object]] = []

    for bout_idx in invalid_idxs.tolist():
        window_frames = list(range(int(pack.window_start_frame[bout_idx]), int(pack.window_end_frame[bout_idx]) + 1))
        missing_posture_frames: list[int] = []
        invalid_posture_frames: list[int] = []
        nonfinite_tail_frames: list[int] = []
        missing_track_frames: list[int] = []
        invalid_track_frames: list[int] = []
        nonfinite_traj_frames: list[int] = []
        valid_tail_frames: list[int] = []
        valid_traj_frames: list[int] = []

        for frame in window_frames:
            posture_idx = posture_lookup.get(int(frame))
            if posture_idx is None:
                missing_posture_frames.append(int(frame))
                tail_issue_counts["missing_posture_frame"] += 1
            elif not bool(posture_valid[posture_idx]):
                invalid_posture_frames.append(int(frame))
                tail_issue_counts["posture_valid_false"] += 1
                if posture_reasons is not None:
                    posture_reason_counts[_decode_reason_value(posture_reasons[posture_idx]) or "<empty>"] += 1
            elif not np.all(np.isfinite(tail_angle[posture_idx])):
                nonfinite_tail_frames.append(int(frame))
                tail_issue_counts["nonfinite_tail_angle"] += 1
            else:
                valid_tail_frames.append(int(frame))

            track_idx = track_lookup.get(int(frame))
            if track_idx is None:
                missing_track_frames.append(int(frame))
                traj_issue_counts["missing_track_frame"] += 1
            elif not bool(track_valid[track_idx]):
                invalid_track_frames.append(int(frame))
                traj_issue_counts["track_sample_valid_false"] += 1
                if track_reasons is not None:
                    track_reason_counts[_decode_reason_value(track_reasons[track_idx]) or "<empty>"] += 1
            else:
                values = np.asarray([positions_mm[track_idx, 0], positions_mm[track_idx, 1], heading[track_idx]])
                if not np.all(np.isfinite(values)):
                    nonfinite_traj_frames.append(int(frame))
                    traj_issue_counts["nonfinite_traj"] += 1
                else:
                    valid_traj_frames.append(int(frame))

        if len(examples) < int(max_examples):
            examples.append(
                {
                    "bout_index": int(bout_idx),
                    "source_bout_id": int(pack.source_bout_id[bout_idx]),
                    "failure_reason": str(pack.failure_reason[bout_idx]),
                    "source_window": [
                        int(pack.source_start_frame[bout_idx]),
                        int(pack.source_end_frame[bout_idx]),
                    ],
                    "classifier_window": [
                        int(pack.window_start_frame[bout_idx]),
                        int(pack.window_end_frame[bout_idx]),
                    ],
                    "tail_valid_fraction": float(pack.tail_valid_fraction[bout_idx]),
                    "traj_valid_fraction": float(pack.traj_valid_fraction[bout_idx]),
                    "max_consecutive_tail_invalid": int(pack.max_consecutive_tail_invalid[bout_idx]),
                    "max_consecutive_traj_invalid": int(pack.max_consecutive_traj_invalid[bout_idx]),
                    "missing_posture_frames": missing_posture_frames,
                    "invalid_posture_frames": invalid_posture_frames,
                    "nonfinite_tail_frames": nonfinite_tail_frames,
                    "missing_track_frames": missing_track_frames,
                    "invalid_track_frames": invalid_track_frames,
                    "nonfinite_traj_frames": nonfinite_traj_frames,
                    "valid_tail_frames": valid_tail_frames,
                    "valid_traj_frames": valid_traj_frames,
                }
            )

    invalid_tail_fraction = pack.tail_valid_fraction[invalid_idxs]
    invalid_traj_fraction = pack.traj_valid_fraction[invalid_idxs]
    result = {
        "status": "ok",
        "diagnostic": "megabouts_classifier_invalid_windows",
        "mutates_archive": False,
        "calls_megabouts": False,
        "n_bouts": int(pack.valid_bout.shape[0]),
        "valid_bout_count": int(np.count_nonzero(pack.valid_bout)),
        "invalid_bout_count": int(invalid_idxs.shape[0]),
        "failure_reason_counts": dict(failure_reason_counts),
        "tail_frame_issue_counts_across_invalid_windows": dict(tail_issue_counts),
        "traj_frame_issue_counts_across_invalid_windows": dict(traj_issue_counts),
        "posture_failure_reason_counts_across_invalid_frames": dict(posture_reason_counts),
        "track_failure_reason_counts_across_invalid_frames": dict(track_reason_counts),
        "invalid_tail_fraction_quantiles": {
            "min": float(np.min(invalid_tail_fraction)) if invalid_tail_fraction.size else math.nan,
            "p25": float(np.quantile(invalid_tail_fraction, 0.25)) if invalid_tail_fraction.size else math.nan,
            "median": float(np.median(invalid_tail_fraction)) if invalid_tail_fraction.size else math.nan,
            "p75": float(np.quantile(invalid_tail_fraction, 0.75)) if invalid_tail_fraction.size else math.nan,
            "max": float(np.max(invalid_tail_fraction)) if invalid_tail_fraction.size else math.nan,
        },
        "invalid_traj_fraction_quantiles": {
            "min": float(np.min(invalid_traj_fraction)) if invalid_traj_fraction.size else math.nan,
            "p25": float(np.quantile(invalid_traj_fraction, 0.25)) if invalid_traj_fraction.size else math.nan,
            "median": float(np.median(invalid_traj_fraction)) if invalid_traj_fraction.size else math.nan,
            "p75": float(np.quantile(invalid_traj_fraction, 0.75)) if invalid_traj_fraction.size else math.nan,
            "max": float(np.max(invalid_traj_fraction)) if invalid_traj_fraction.size else math.nan,
        },
        "source_refs": pack.source_refs,
        "parameters": pack.parameters,
        "examples": examples,
    }
    return dict(_json_safe(result))


def build_megabouts_classifier_input_pack_from_zarr(
    zarr_path: str | Path,
    **kwargs: object,
) -> MegaboutsClassifierInputPack:
    root = open_zarr_root(zarr_path, mode="r")
    return build_megabouts_classifier_input_pack(root, **kwargs)


def dry_run_megabouts_classifier_inputs(
    zarr_path: str | Path,
    **kwargs: object,
) -> dict[str, object]:
    pack = build_megabouts_classifier_input_pack_from_zarr(zarr_path, **kwargs)
    return summarize_input_pack(pack)


def diagnose_megabouts_classifier_invalid_windows(
    zarr_path: str | Path,
    *,
    max_examples: int = 12,
    **kwargs: object,
) -> dict[str, object]:
    root = open_zarr_root(zarr_path, mode="r")
    pack = build_megabouts_classifier_input_pack(root, **kwargs)
    return diagnose_input_pack_invalid_windows(root, pack, max_examples=max_examples)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dry-run Megabouts classifier input arrays from Palette runs without calling Megabouts."
    )
    parser.add_argument("zarr_path", type=Path, help="Palette zarr archive.")
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
        "--diagnose-invalid-windows",
        action="store_true",
        help="Emit detailed invalid-window cause report instead of the compact dry-run summary.",
    )
    parser.add_argument("--max-examples", type=int, default=12, help="Maximum invalid-window examples in diagnostics.")
    parser.add_argument("--json", action="store_true", help="Emit compact JSON.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    kwargs = {
        "tail_posture_view_run": args.tail_posture_view_run,
        "track_kinematics_run": args.track_kinematics_run,
        "track_scope": args.track_scope,
        "track_id": int(args.track_id),
        "swim_bout_run": args.swim_bout_run,
        "speed_level": args.speed_level,
        "heading_source": args.heading_source,
        "bout_duration_s": float(args.bout_duration_s),
        "bout_duration_frames": args.bout_duration_frames,
        "min_tail_valid_fraction": float(args.min_tail_valid_fraction),
        "min_traj_valid_fraction": float(args.min_traj_valid_fraction),
        "max_consecutive_invalid_frames": int(args.max_consecutive_invalid_frames),
    }
    if args.diagnose_invalid_windows:
        summary = diagnose_megabouts_classifier_invalid_windows(
            args.zarr_path,
            max_examples=int(args.max_examples),
            **kwargs,
        )
    else:
        summary = dry_run_megabouts_classifier_inputs(args.zarr_path, **kwargs)
    print(json.dumps(summary, indent=None if args.json else 2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
