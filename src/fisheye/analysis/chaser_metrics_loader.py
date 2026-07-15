"""Utilities to present online and offline chaser metrics through a common API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import zarr

from fisheye.shared.zarr.columnar import load_structured_dataset


PathLike = Union[str, Path]


@dataclass
class ChaserMetricsBundle:
    """Normalized chaser metrics aligned by camera frame.

    Attributes
    ----------
    camera_frame_ids:
        Sorted camera-frame identifiers spanning the stimulus run.
    stimulus_frame_nums:
        Stimulus frame numbers aligned to ``camera_frame_ids``.
    timestamp_ns:
        Timestamps (nanoseconds) for each camera frame when available; -1 when unknown.
    trial_state:
        Trial state enum per frame (-1 when unavailable).
    metadata_mask:
        Boolean mask indicating whether the underlying metadata row is original (True)
        or interpolated/synthesized (False). ``None`` if no mask was stored.
    online:
        Dictionary of online chaser fields (arrays aligned to ``camera_frame_ids``).
    offline:
        Dictionary of offline metrics (arrays aligned to ``camera_frame_ids``). Vector
        quantities use shape ``(n_frames, 2)`` and scalars ``(n_frames,)``. Includes a
        ``"has_offline"`` boolean mask.
    provenance:
        Provenance details such as the stimulus run, metrics run, keypoint source, and
        chaser index.
    """

    camera_frame_ids: np.ndarray
    stimulus_frame_nums: np.ndarray
    timestamp_ns: np.ndarray
    trial_state: np.ndarray
    metadata_mask: Optional[np.ndarray]
    online: Dict[str, np.ndarray]
    offline: Dict[str, np.ndarray]
    provenance: Dict[str, object]


def load_chaser_metrics(
    zarr_path: PathLike,
    *,
    stimulus_run: Optional[str] = None,
    metrics_run: Optional[str] = None,
    chaser_index: int = 0,
) -> ChaserMetricsBundle:
    """Load online and offline chaser metrics in a unified layout."""

    root = zarr.open(str(zarr_path), mode="r")
    analysis_group = root.get("analysis")
    if analysis_group is None:
        raise ValueError("Zarr store does not contain analysis group.")

    stimulus_parent = analysis_group.get("stimulus_runs")
    if stimulus_parent is None:
        raise ValueError("Zarr store does not contain analysis/stimulus_runs.")
    stim_run = stimulus_run or _resolve_latest(stimulus_parent)
    if stim_run is None:
        raise ValueError("Stimulus run not provided and no 'latest' attribute present.")
    if stim_run not in stimulus_parent:
        raise ValueError(f"Stimulus run '{stim_run}' not found under analysis/stimulus_runs.")

    stim_group = stimulus_parent[stim_run]
    camera_meta, stimulus_field, timestamp_field, stim_to_camera = _build_camera_metadata_map(stim_group)
    metadata_mask = _load_metadata_mask(stim_group, len(camera_meta))

    frames_sorted = np.array(sorted(camera_meta.keys()), dtype=np.int64)
    frame_to_index = {frame: idx for idx, frame in enumerate(frames_sorted)}

    stimulus_frames = np.array(
        [int(camera_meta[frame][stimulus_field]) for frame in frames_sorted], dtype=np.int64
    )
    if timestamp_field is not None:
        timestamps = np.array(
            [int(camera_meta[frame][timestamp_field]) for frame in frames_sorted], dtype=np.int64
        )
    else:
        timestamps = np.full(frames_sorted.shape, -1, dtype=np.int64)

    trial_state = np.full(frames_sorted.shape, -1, dtype=np.int16)
    online_fields = _extract_online_fields(
        stim_group=stim_group,
        frames_sorted=frames_sorted,
        frame_to_index=frame_to_index,
        trial_state=trial_state,
        chaser_index=chaser_index,
        stim_to_camera=stim_to_camera,
    )

    metrics_parent = analysis_group.get("chaser_fish_metrics")
    metrics_run_resolved: Optional[str]
    if metrics_parent is None:
        metrics_run_resolved = metrics_run
    elif metrics_run is not None:
        metrics_run_resolved = metrics_run
    else:
        metrics_run_resolved = _resolve_latest(metrics_parent)

    offline_fields: Dict[str, np.ndarray]
    provenance = {
        "stimulus_run": stim_run,
        "metrics_run": metrics_run_resolved,
        "chaser_index": int(chaser_index),
    }

    if metrics_parent is not None and metrics_run_resolved and metrics_run_resolved in metrics_parent:
        metrics_group = metrics_parent[metrics_run_resolved]
        offline_fields = _extract_offline_fields(
            metrics_group=metrics_group,
            frames_sorted=frames_sorted,
            frame_to_index=frame_to_index,
        )
        provenance.update(
            {
                "source_stimulus_run": metrics_group.attrs.get("source_stimulus_run"),
                "source_keypoints_run": metrics_group.attrs.get("source_keypoints_run"),
            }
        )
    else:
        offline_fields = _empty_offline_fields(frames_sorted.shape[0])

    return ChaserMetricsBundle(
        camera_frame_ids=frames_sorted,
        stimulus_frame_nums=stimulus_frames,
        timestamp_ns=timestamps,
        trial_state=trial_state,
        metadata_mask=metadata_mask,
        online=online_fields,
        offline=offline_fields,
        provenance=provenance,
    )


def _resolve_latest(group: zarr.Group) -> Optional[str]:
    latest = group.attrs.get("latest")
    if isinstance(latest, bytes):
        latest = latest.decode("utf-8", "ignore")
    if isinstance(latest, str) and latest in group:
        return latest
    return None


def _load_metadata_mask(stim_group: zarr.Group, length: int) -> Optional[np.ndarray]:
    for mask_name in ("camera_aligned_interpolation_mask", "interpolation_mask"):
        if mask_name not in stim_group:
            continue
        mask = np.asarray(stim_group[mask_name], dtype=bool)
        if mask.shape[0] == length:
            return mask
    return None


def _build_camera_metadata_map(
    stim_group: zarr.Group,
) -> Tuple[Dict[int, np.void], str, Optional[str], Dict[int, int]]:
    meta_group = stim_group.require_group("video_metadata")
    base_dataset_name = "frame_metadata"
    if base_dataset_name in meta_group:
        base_metadata, _ = load_structured_dataset(meta_group, base_dataset_name)
    else:
        base_metadata = None

    aligned_dataset_name = "camera_aligned_frame_metadata"
    if aligned_dataset_name in meta_group:
        aligned_metadata, _ = load_structured_dataset(meta_group, aligned_dataset_name)
    else:
        aligned_metadata = base_metadata

    if aligned_metadata is None:
        raise ValueError("Stimulus run missing video_metadata/frame_metadata dataset.")
    if base_metadata is None:
        base_metadata = aligned_metadata

    aligned_stim_field = _resolve_struct_field(
        aligned_metadata,
        "stimulus_frame_num",
        "frame_number",
        "stim_frame_num",
    )
    aligned_camera_field = _resolve_struct_field(
        aligned_metadata,
        "triggering_camera_frame_id",
        "camera_frame_id",
    )
    timestamp_field = _maybe_resolve_struct_field(
        aligned_metadata,
        "timestamp_ns",
        "timestamp_ns_session",
        "relative_timestamp_ns",
    )

    camera_to_record: Dict[int, np.void] = {}
    stim_to_camera: Dict[int, int] = {}
    for record in aligned_metadata:
        camera_id = int(record[aligned_camera_field])
        stimulation_id = int(record[aligned_stim_field])
        camera_to_record[camera_id] = record
        stim_to_camera[stimulation_id] = camera_id

    base_stim_field = _resolve_struct_field(
        base_metadata,
        "stimulus_frame_num",
        "frame_number",
        "stim_frame_num",
    )
    base_camera_field = _resolve_struct_field(
        base_metadata,
        "triggering_camera_frame_id",
        "camera_frame_id",
    )
    for record in base_metadata:
        stimulus_id = int(record[base_stim_field])
        if stimulus_id not in stim_to_camera:
            stim_to_camera[stimulus_id] = int(record[base_camera_field])

    return camera_to_record, base_stim_field, timestamp_field, stim_to_camera


def _resolve_struct_field(array: np.ndarray, *candidates: str) -> str:
    names = array.dtype.names or ()
    for candidate in candidates:
        if candidate in names:
            return candidate
    raise ValueError(f"Structured array missing expected field (tried {', '.join(candidates)})")


def _maybe_resolve_struct_field(array: np.ndarray, *candidates: str) -> Optional[str]:
    names = array.dtype.names or ()
    for candidate in candidates:
        if candidate in names:
            return candidate
    return None


def _extract_online_fields(
    *,
    stim_group: zarr.Group,
    frames_sorted: np.ndarray,
    frame_to_index: Dict[int, int],
    trial_state: np.ndarray,
    chaser_index: int,
    stim_to_camera: Dict[int, int],
) -> Dict[str, np.ndarray]:
    tracking_group = stim_group.require_group("tracking_data")
    if "chaser_states" not in tracking_group:
        raise ValueError(
            "Stimulus run tracking_data group lacks chaser_states dataset."
        )

    chaser_states, _ = load_structured_dataset(tracking_group, "chaser_states")
    dtype_names = chaser_states.dtype.names or ()

    frame_field = _resolve_optional_field(chaser_states, "frame_number", "stimulus_frame_num")
    camera_field = _resolve_optional_field(
        chaser_states,
        "camera_frame_id",
        "triggering_camera_frame_id",
    )

    if camera_field is not None:
        camera_frames = np.asarray(chaser_states[camera_field], dtype=np.int64)
    else:
        if frame_field is None:
            raise ValueError(
                "chaser_states lacks both camera_frame_id and stimulus_frame_num fields"
            )
        frame_numbers = np.asarray(chaser_states[frame_field], dtype=np.int64)
        camera_frames = np.array(
            [stim_to_camera.get(int(stim), -1) for stim in frame_numbers],
            dtype=np.int64,
        )

    if "chaser_index" in dtype_names:
        mask = chaser_states["chaser_index"] == chaser_index
    else:
        mask = np.ones(chaser_states.shape[0], dtype=bool)

    filtered_states = chaser_states[mask]
    filtered_camera_frames = camera_frames[mask]

    trial_state_field = "trial_state" if "trial_state" in dtype_names else None

    numeric_fields = [
        "chaser_pos_x",
        "chaser_pos_y",
        "target_pos_x",
        "target_pos_y",
        "distance_to_target_px",
        "distance_to_target_mm",
        "visual_angle_deg",
        "tau_ms",
    ]

    present_fields = [field for field in numeric_fields if field in dtype_names]
    online_arrays = {
        field: np.full(frames_sorted.shape, np.nan, dtype=np.float64)
        for field in present_fields
    }

    for record, camera_frame in zip(filtered_states, filtered_camera_frames):
        if camera_frame not in frame_to_index:
            continue
        idx = frame_to_index[camera_frame]
        if trial_state_field is not None:
            try:
                trial_state[idx] = int(record[trial_state_field])
            except Exception:
                trial_state[idx] = int(record[trial_state_field].item())
        for field in present_fields:
            try:
                online_arrays[field][idx] = float(record[field])
            except Exception:
                online_arrays[field][idx] = float(np.asarray(record[field]))

    return online_arrays


def _resolve_optional_field(array: np.ndarray, *candidates: str) -> Optional[str]:
    names = array.dtype.names or ()
    for candidate in candidates:
        if candidate in names:
            return candidate
    return None


def _extract_offline_fields(
    *,
    metrics_group: zarr.Group,
    frames_sorted: np.ndarray,
    frame_to_index: Dict[int, int],
) -> Dict[str, np.ndarray]:
    frame_indices = np.asarray(metrics_group["frame_indices"], dtype=np.int64)
    valid_mask = np.asarray(metrics_group["valid_mask"], dtype=bool)

    first_valid_index: Dict[int, int] = {}
    for idx, (frame, is_valid) in enumerate(zip(frame_indices, valid_mask)):
        if not is_valid or frame < 0:
            continue
        if frame not in first_valid_index:
            first_valid_index[frame] = idx

    offline_scalars = {
        "distance_px": np.asarray(metrics_group["distance_px"], dtype=np.float64),
        "distance_mm": np.asarray(metrics_group["distance_mm"], dtype=np.float64),
        "angle_unsigned_deg": np.asarray(metrics_group["angle_unsigned_deg"], dtype=np.float64),
        "angle_signed_deg": np.asarray(metrics_group["angle_signed_deg"], dtype=np.float64),
    }
    if "heading_deg" in metrics_group:
        offline_scalars["heading_deg"] = np.asarray(metrics_group["heading_deg"], dtype=np.float64)
    offline_vectors = {
        "fish_centroid_px": np.asarray(metrics_group["fish_centroid_px"], dtype=np.float64),
        "chaser_position_px": np.asarray(metrics_group["chaser_position_px"], dtype=np.float64),
    }

    n_frames = frames_sorted.shape[0]
    output_scalars = {
        name: np.full(n_frames, np.nan, dtype=np.float64) for name in offline_scalars
    }
    output_vectors = {
        name: np.full((n_frames, 2), np.nan, dtype=np.float64) for name in offline_vectors
    }
    has_offline = np.zeros(n_frames, dtype=bool)

    for frame, roi_idx in first_valid_index.items():
        if frame not in frame_to_index:
            continue
        idx = frame_to_index[frame]
        has_offline[idx] = True
        for name, data in offline_scalars.items():
            output_scalars[name][idx] = float(data[roi_idx])
        for name, data in offline_vectors.items():
            output_vectors[name][idx] = data[roi_idx]

    offline_fields: Dict[str, np.ndarray] = {
        **output_scalars,
        **output_vectors,
        "has_offline": has_offline,
    }
    return offline_fields


def _empty_offline_fields(length: int) -> Dict[str, np.ndarray]:
    offline_fields = {
        "distance_px": np.full(length, np.nan, dtype=np.float64),
        "distance_mm": np.full(length, np.nan, dtype=np.float64),
        "angle_unsigned_deg": np.full(length, np.nan, dtype=np.float64),
        "angle_signed_deg": np.full(length, np.nan, dtype=np.float64),
        "fish_centroid_px": np.full((length, 2), np.nan, dtype=np.float64),
        "chaser_position_px": np.full((length, 2), np.nan, dtype=np.float64),
        "has_offline": np.zeros(length, dtype=bool),
    }
    return offline_fields
