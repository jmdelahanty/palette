"""Dry-run Megabouts classifier input construction from Palette Zarr runs."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import zarr

from fisheye.analysis.swim_bout_io import load_swim_bout_tables
from fisheye.analysis.track_kinematics_io import load_track_kinematics_track
from fisheye.shared.coordinate_identity import (
    TRACK_SAMPLE_SOURCE_INSTANCE_KEY_DTYPE,
)
from fisheye.shared.json_safety import decode_null_terminated_text, json_attr_safe
from fisheye.shared.tail_coordinate_publication import (
    BoundTailCoordinatePublication,
    load_tail_posture_coordinate_publication,
)
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    is_run_complete_in_parent,
    is_run_selector_eligible,
)

DEFAULT_TAIL_POSTURE_VIEW_FAMILY = "megabouts_compatible"
DEFAULT_BOUT_DURATION_S = 0.2
DEFAULT_MIN_TAIL_VALID_FRACTION = 0.90
DEFAULT_MIN_TRAJ_VALID_FRACTION = 0.90
DEFAULT_MAX_CONSECUTIVE_INVALID_FRAMES = 2
DEFAULT_HEADING_SOURCE = "smoothed_heading_radians"
DEFAULT_ALIGN_TRAJ_TO_ONSET = True
DEFAULT_TRAJ_REFERENCE_INDEX = 0
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
    traj_reference_valid: np.ndarray
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


_json_safe = json_attr_safe


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
) -> tuple[zarr.Group, str, str, BoundTailCoordinatePublication]:
    analysis = _require_group(root, "analysis")
    parent = _require_group(analysis, "tail_posture_view_runs")
    spec = str(run_name or "latest").strip()
    if spec != spec.strip("/") or "//" in spec:
        raise ValueError(
            f"Tail posture view run selector {run_name!r} is not an exact canonical path."
        )
    prefix = "analysis/tail_posture_view_runs/"
    family = str(view_family).strip()
    if not family or "/" in family:
        raise ValueError("Tail posture view_family must be one non-empty selector token.")

    def load_exact(name: str) -> tuple[
        zarr.Group,
        str,
        str,
        BoundTailCoordinatePublication,
    ]:
        if not name or "/" in name or name not in parent:
            raise ValueError(f"Tail posture view run {run_name!r} not found.")
        child = parent[name]
        if not isinstance(child, zarr.Group):
            raise ValueError(f"{prefix}{name} is not a Zarr group.")
        if str(child.attrs.get("view_family") or "") != family:
            raise ValueError(
                f"Tail posture selector for family {family!r} names "
                f"a run from family {child.attrs.get('view_family')!r}."
            )
        if not is_run_selector_eligible(child) or not is_run_complete_in_parent(
            parent,
            child,
            legacy_default=False,
        ):
            raise ValueError(f"Tail posture view run {name!r} is not complete and selector-eligible.")
        path = f"{prefix}{name}"
        publication = load_tail_posture_coordinate_publication(root, path)
        return child, name, path, publication

    if spec.startswith(prefix):
        resolved = spec[len(prefix) :]
        if not resolved or "/" in resolved:
            raise ValueError(f"Tail posture view run path {run_name!r} is invalid.")
        return load_exact(resolved)
    if spec != "latest":
        return load_exact(spec)

    family_selector = f"latest_{family}"
    preferred_raw = parent.attrs.get(family_selector)
    preferred = str(preferred_raw) if preferred_raw is not None else ""
    if not preferred:
        raise ValueError(
            f"Tail posture family selector {family_selector!r} is missing."
        )
    if preferred != preferred.strip() or "/" in preferred or preferred not in parent:
        raise ValueError(
            f"Tail posture family selector {family_selector!r} is invalid."
        )
    child = parent[preferred]
    if not isinstance(child, zarr.Group):
        raise ValueError(
            f"Tail posture family selector {family_selector!r} does not name a group."
        )
    if str(child.attrs.get("view_family") or "") != family:
        raise ValueError(
            f"Tail posture family selector {family_selector!r} names the wrong family."
        )
    if not is_run_selector_eligible(child) or not is_run_complete_in_parent(
        parent,
        child,
        legacy_default=False,
    ):
        raise ValueError(
            f"Tail posture family selector {family_selector!r} names a run that "
            "is not complete and selector-eligible; canonical readers do not "
            "guess a prior run."
        )
    return load_exact(preferred)


def _require_unchanged_tail_posture_publication(
    root: zarr.Group,
    run_path: str,
    *,
    expected_manifest_ref: str,
    expected_manifest_sha256: str,
    expected_source_run_path: str,
    expected_source_manifest_ref: str,
    expected_source_manifest_sha256: str,
    error_message: str,
) -> None:
    fresh = load_tail_posture_coordinate_publication(root, run_path)
    if (
        fresh.run_path != run_path
        or fresh.manifest.record_ref != expected_manifest_ref
        or fresh.manifest.record_sha256 != expected_manifest_sha256
        or fresh.source.run_path != expected_source_run_path
        or fresh.source.manifest.record_ref != expected_source_manifest_ref
        or fresh.source.manifest.record_sha256 != expected_source_manifest_sha256
    ):
        raise ValueError(error_message)


def _require_input_pack_tail_posture_publication(
    root: zarr.Group,
    source_refs: Mapping[str, str],
    *,
    error_message: str,
) -> None:
    required_refs = (
        "tail_posture_view_run",
        "tail_posture_publication_manifest_ref",
        "tail_posture_publication_manifest_sha256",
        "tail_posture_source_subject_shape_run",
        "tail_posture_source_subject_shape_publication_manifest_ref",
        "tail_posture_source_subject_shape_publication_manifest_sha256",
    )
    missing = [name for name in required_refs if not source_refs.get(name)]
    if missing:
        raise ValueError(
            "Megabouts input pack is missing required tail-posture publication "
            f"references: {', '.join(missing)}."
        )
    _require_unchanged_tail_posture_publication(
        root,
        source_refs["tail_posture_view_run"],
        expected_manifest_ref=source_refs["tail_posture_publication_manifest_ref"],
        expected_manifest_sha256=source_refs[
            "tail_posture_publication_manifest_sha256"
        ],
        expected_source_run_path=source_refs[
            "tail_posture_source_subject_shape_run"
        ],
        expected_source_manifest_ref=source_refs[
            "tail_posture_source_subject_shape_publication_manifest_ref"
        ],
        expected_source_manifest_sha256=source_refs[
            "tail_posture_source_subject_shape_publication_manifest_sha256"
        ],
        error_message=error_message,
    )


def _resolve_track_run(
    root: zarr.Group,
    run_name: str,
    *,
    track_scope: str,
) -> tuple[zarr.Group, str, str, str]:
    analysis = _require_group(root, "analysis")
    parent = _require_group(analysis, "track_kinematics_runs")
    spec = str(run_name or "latest").strip()
    if spec != spec.strip("/") or "//" in spec:
        raise ValueError(
            f"Track kinematics run selector {run_name!r} is not an exact canonical path."
        )
    parts = spec.split("/")
    if parts[:2] == ["analysis", "track_kinematics_runs"]:
        if len(parts) != 4 or not parts[2] or not parts[3]:
            raise ValueError(
                f"Track kinematics run path {run_name!r} must exactly equal "
                "analysis/track_kinematics_runs/<scope>/<run>."
            )
        scope = parts[2]
        resolved = parts[3]
    elif len(parts) == 2:
        scope = parts[0]
        resolved = parts[1]
        if not scope or not resolved:
            raise ValueError(
                f"Track kinematics run selector {run_name!r} is invalid."
            )
    elif len(parts) > 1:
        raise ValueError(
            f"Track kinematics run selector {run_name!r} must be a bare run, "
            "<scope>/<run>, or an exact canonical path."
        )
    else:
        scope = str(track_scope).strip()
        if not scope or "/" in scope:
            raise ValueError(f"Track kinematics scope {track_scope!r} is invalid.")
        if scope not in parent or not isinstance(parent[scope], zarr.Group):
            raise ValueError(f"Track kinematics scope {scope!r} not found.")
        resolved = parent[scope].attrs.get("latest") if spec == "latest" else spec
    if scope not in parent or not isinstance(parent[scope], zarr.Group):
        raise ValueError(f"Track kinematics scope {scope!r} not found.")
    if (
        not isinstance(resolved, str)
        or not resolved
        or "/" in resolved
        or resolved not in parent[scope]
    ):
        raise ValueError(f"Track kinematics run {run_name!r} not found under scope {scope!r}.")
    if not isinstance(parent[scope][resolved], zarr.Group):
        raise ValueError(
            f"Track kinematics run {run_name!r} does not name a Zarr group."
        )
    path = f"analysis/track_kinematics_runs/{scope}/{resolved}"
    return parent[scope][resolved], str(resolved), path, str(scope)


def _unique_frame_to_index(frames: np.ndarray, *, label: str) -> dict[int, int]:
    values = np.asarray(frames, dtype=np.int64)
    if values.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional, got {values.shape}.")
    mapping: dict[int, int] = {}
    for idx, frame in enumerate(values.tolist()):
        value = int(frame)
        if value in mapping:
            raise ValueError(
                f"{label} contains duplicate acquisition frame {value}; "
                "track-row selection would be ambiguous."
            )
        mapping[value] = int(idx)
    return mapping


def _frame_to_index(frames: np.ndarray) -> dict[int, int]:
    """Compatibility helper for dense, necessarily unique frame axes."""

    return _unique_frame_to_index(frames, label="dense frame axis")


def _require_posture_instance_keys(
    posture: zarr.Group,
    *,
    row_count: int,
) -> np.ndarray:
    values = np.asarray(_require_array(posture, "instance_key")[:])
    if values.dtype != np.dtype("uint64") or values.shape != (int(row_count),):
        raise ValueError(
            f"{posture.name}/instance_key must be canonical uint64 shape "
            f"({int(row_count)},), got dtype={values.dtype}, shape={values.shape}."
        )
    if np.unique(values).size != values.size:
        raise ValueError(
            f"{posture.name}/instance_key must contain unique observation identities."
        )
    return values


def _require_track_source_instance_keys(
    values: object,
    *,
    row_count: int,
    track_path: str,
) -> np.ndarray:
    if values is None:
        raise ValueError(
            f"{track_path} has no verified source_instance_key identity lineage."
        )
    keys = np.asarray(values)
    if (
        keys.dtype != TRACK_SAMPLE_SOURCE_INSTANCE_KEY_DTYPE
        or keys.shape != (int(row_count),)
    ):
        raise ValueError(
            f"{track_path}/source_instance_key must use canonical nullable "
            f"dtype {TRACK_SAMPLE_SOURCE_INSTANCE_KEY_DTYPE} and shape "
            f"({int(row_count)},), got dtype={keys.dtype}, shape={keys.shape}."
        )
    if np.any(~keys["valid"] & (keys["instance_key"] != 0)):
        raise ValueError(
            f"{track_path}/source_instance_key contains noncanonical null values."
        )
    valid_keys = keys["instance_key"][keys["valid"]]
    if np.unique(valid_keys).size != valid_keys.size:
        raise ValueError(
            f"{track_path}/source_instance_key contains duplicate valid identities."
        )
    return keys


def _join_posture_rows_to_track_sources(
    *,
    posture_instance_keys: np.ndarray,
    posture_frames: np.ndarray,
    track_source_instance_keys: np.ndarray,
    track_frames: np.ndarray,
) -> np.ndarray:
    """Bind each track row to one posture row through observation identity.

    Missing observations remain ``-1`` and therefore invalidate only that tail
    sample. A matching identity with a different acquisition frame indicates
    corrupt lineage and aborts the entire input pack.
    """

    posture_lookup = {
        int(instance_key): int(row_index)
        for row_index, instance_key in enumerate(posture_instance_keys.tolist())
    }
    joined = np.full((int(track_frames.shape[0]),), -1, dtype=np.int64)
    for track_index in range(int(track_frames.shape[0])):
        if not bool(track_source_instance_keys["valid"][track_index]):
            continue
        instance_key = int(
            track_source_instance_keys["instance_key"][track_index]
        )
        posture_index = posture_lookup.get(instance_key)
        if posture_index is None:
            continue
        posture_frame = int(posture_frames[posture_index])
        track_frame = int(track_frames[track_index])
        if posture_frame != track_frame:
            raise ValueError(
                "Tail-posture/track source_instance_key acquisition-frame "
                f"mismatch for instance_key={instance_key}: posture={posture_frame}, "
                f"track={track_frame}."
            )
        joined[track_index] = posture_index
    return joined


def _require_swim_bout_track_lineage(
    swim_payload: object,
    track_tables: object,
) -> None:
    candidate = swim_payload.candidate
    source_run = candidate.source_track_kinematics_run
    expected_run = str(track_tables.run_name)
    if not source_run:
        raise ValueError(
            "Selected swim-bout candidate lacks source_track_kinematics_run."
        )
    if str(source_run) != expected_run:
        raise ValueError(
            "Selected swim-bout candidate source_track_kinematics_run mismatch: "
            f"bout={source_run!r}, track={expected_run!r}."
        )

    source_track_id = candidate.track_id
    expected_track_id = int(track_tables.track_id)
    if source_track_id is None:
        raise ValueError("Selected swim-bout candidate lacks track_id lineage.")
    if int(source_track_id) != expected_track_id:
        raise ValueError(
            "Selected swim-bout candidate track_id mismatch: "
            f"bout={int(source_track_id)}, track={expected_track_id}."
        )

    run_attrs = swim_payload.run_attrs
    raw_digest = run_attrs.get("source_track_motion_manifest_sha256")
    source_digest = str(raw_digest).strip() if raw_digest is not None else ""
    expected_raw_digest = track_tables.motion_manifest_sha256
    expected_digest = (
        str(expected_raw_digest).strip()
        if expected_raw_digest is not None
        else ""
    )
    if not source_digest:
        raise ValueError(
            "Selected swim-bout run lacks source_track_motion_manifest_sha256."
        )
    if not expected_digest:
        raise ValueError("Verified track input lacks a motion-manifest digest.")
    if source_digest != expected_digest:
        raise ValueError(
            "Selected swim-bout source_track_motion_manifest_sha256 mismatch: "
            f"bout={source_digest!r}, track={expected_digest!r}."
        )


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


def _align_traj_array_to_reference(
    traj_array: np.ndarray,
    traj_valid: np.ndarray,
    *,
    reference_index: int = DEFAULT_TRAJ_REFERENCE_INDEX,
) -> tuple[np.ndarray, np.ndarray]:
    """Translate and rotate trajectory windows into a per-bout onset frame.

    Megabouts' full-tracking classifier expects trajectory channels extracted
    with its segmentation helper's default alignment: x/y are relative to the
    reference sample and rotated by the negative reference heading, while
    heading is expressed relative to that same sample.
    """

    traj = np.asarray(traj_array, dtype=np.float32).copy()
    valid = np.asarray(traj_valid, dtype=bool)
    if traj.ndim != 3 or traj.shape[1] != 3:
        raise ValueError(f"traj_array must have shape (n_bouts, 3, window), got {traj.shape}.")
    if valid.shape != (traj.shape[0], traj.shape[2]):
        raise ValueError(f"traj_valid shape {valid.shape} does not match traj_array shape {traj.shape}.")
    ref = int(reference_index)
    if ref < 0 or ref >= traj.shape[2]:
        raise ValueError(f"trajectory reference index {ref} is outside window length {traj.shape[2]}.")

    reference_valid = valid[:, ref] & np.all(np.isfinite(traj[:, :, ref]), axis=1)
    for bout_idx in np.flatnonzero(reference_valid).tolist():
        x0 = float(traj[bout_idx, 0, ref])
        y0 = float(traj[bout_idx, 1, ref])
        theta0 = float(traj[bout_idx, 2, ref])
        dx = traj[bout_idx, 0, :] - x0
        dy = traj[bout_idx, 1, :] - y0
        cos_t = math.cos(theta0)
        sin_t = math.sin(theta0)
        traj[bout_idx, 0, :] = cos_t * dx + sin_t * dy
        traj[bout_idx, 1, :] = -sin_t * dx + cos_t * dy
        traj[bout_idx, 2, :] = traj[bout_idx, 2, :] - theta0

    return traj, reference_valid.astype(bool, copy=False)


def _decode_reason_value(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_, str)):
        return decode_null_terminated_text(value)
    arr = np.asarray(value)
    if arr.ndim == 0:
        item = arr.item()
        return decode_null_terminated_text(item)
    if arr.dtype.kind in {"S", "U", "O"}:
        return str(arr.reshape(-1)[0])
    return decode_null_terminated_text(arr.astype(np.uint8, copy=False))


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
    root: zarr.Group,
    *,
    run_name: str,
    track_scope: str,
    track_id: int,
    heading_source: str,
) -> tuple[object, dict[str, np.ndarray], dict[str, str], float]:
    tables = load_track_kinematics_track(
        root,
        run_name=run_name,
        scope=track_scope,
        track_id=int(track_id),
        required_speed_levels=(),
    )
    frames = np.asarray(tables.source_acquisition_frame_index, dtype=np.int64)
    if tables.positions_mm is None:
        raise ValueError(
            f"{tables.track_path} has no verified physical positions_mm surface."
        )
    positions_mm = np.asarray(tables.positions_mm, dtype=np.float32)
    headings = {
        "heading_degrees": tables.heading_degrees,
        "heading_radians": tables.heading_radians,
        "smoothed_heading_degrees": tables.smoothed_heading_degrees,
        "smoothed_heading_radians": tables.smoothed_heading_radians,
    }
    if heading_source not in headings or headings[heading_source] is None:
        raise ValueError(
            f"Unsupported or unavailable verified heading source {heading_source!r}."
        )
    heading = np.asarray(headings[heading_source], dtype=np.float32)
    if positions_mm.shape != (frames.shape[0], 2):
        raise ValueError(f"positions_mm shape {positions_mm.shape} does not match frames length {frames.shape[0]}.")
    if heading.shape[0] != frames.shape[0]:
        raise ValueError(f"{heading_source} length {heading.shape[0]} does not match frames length {frames.shape[0]}.")
    if tables.sample_valid is None:
        raise ValueError(f"{tables.track_path} has no verified sample_valid surface.")
    sample_valid = np.asarray(tables.sample_valid, dtype=bool)
    if sample_valid.shape[0] != frames.shape[0]:
        raise ValueError(f"sample_valid length {sample_valid.shape[0]} does not match frames length {frames.shape[0]}.")
    source_instance_key = _require_track_source_instance_keys(
        tables.source_instance_key,
        row_count=int(frames.shape[0]),
        track_path=str(tables.track_path),
    )
    fps = float(tables.run_attrs.get("fps", 0.0))
    arrays = {
        "frame_indices": frames,
        "source_instance_key": source_instance_key,
        "positions_mm": positions_mm,
        "heading": heading,
        "sample_valid": sample_valid,
    }
    refs = {
        "track_group": tables.track_path,
        "track_frame_indices": (
            f"{tables.track_path}/source_acquisition_frame_index"
        ),
        "track_source_instance_key": f"{tables.track_path}/source_instance_key",
        "positions_mm": f"{tables.track_path}/positions_mm",
        "positions_mm_coordinate_descriptor_sha256": str(
            tables.positions_mm_descriptor_sha256
        ),
        "heading": f"{tables.track_path}/{heading_source}",
        "sample_valid": f"{tables.track_path}/sample_valid",
        "track_motion_manifest_sha256": str(tables.motion_manifest_sha256),
    }
    return tables, arrays, refs, fps


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
    align_traj_to_onset: bool = DEFAULT_ALIGN_TRAJ_TO_ONSET,
    traj_reference_index: int = DEFAULT_TRAJ_REFERENCE_INDEX,
) -> MegaboutsClassifierInputPack:
    """Build fixed-window arrays for optional Megabouts classification.

    The returned tensors follow the classifier-facing convention documented in
    ``docs/megabouts_direct_integration_design.md``:

    - ``tail_array`` has shape ``(n_bouts, 10, window_frames)``.
    - ``traj_array`` has shape ``(n_bouts, 3, window_frames)`` with channels
      ``x_mm``, ``y_mm``, and selected heading/yaw in radians. By default,
      these channels are translated/rotated into the onset body frame to match
      Megabouts' classifier-facing trajectory extraction.
    """

    (
        posture,
        posture_name,
        posture_path,
        posture_publication,
    ) = _resolve_tail_posture_view_run(root, tail_posture_view_run)
    posture_manifest_ref = posture_publication.manifest.record_ref
    posture_manifest_sha256 = posture_publication.manifest.record_sha256
    posture_source_run_path = posture_publication.source.run_path
    posture_source_manifest_ref = posture_publication.source.manifest.record_ref
    posture_source_manifest_sha256 = (
        posture_publication.source.manifest.record_sha256
    )
    _track_run, track_name, track_path, resolved_scope = _resolve_track_run(
        root,
        track_kinematics_run,
        track_scope=track_scope,
    )
    requested_speed_level = None if str(speed_level or "default").strip() in {"default", "latest"} else speed_level
    swim_payload = load_swim_bout_tables(
        root,
        run_name=swim_bout_run,
        speed_level=requested_speed_level,
    )
    bout_run_name = swim_payload.run_name
    resolved_level = swim_payload.signal.speed_level
    bout_level_path = swim_payload.level_path

    track_tables, track_arrays, track_refs, fps = _load_track_arrays(
        root,
        run_name=track_name,
        track_scope=resolved_scope,
        track_id=int(track_id),
        heading_source=str(heading_source),
    )
    _require_swim_bout_track_lineage(swim_payload, track_tables)
    if fps <= 0.0:
        fps = float(swim_payload.signal_attrs.get("fps") or swim_payload.run_attrs.get("fps", 0.0))
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

    posture_frames = np.asarray(
        _require_array(posture, "source_acquisition_frame_index")[:],
        dtype=np.int64,
    )
    posture_valid = np.asarray(_require_array(posture, "valid")[:], dtype=bool)
    tail_angle = np.asarray(_require_array(posture, "tail_angle_rad")[:], dtype=np.float32)
    if tail_angle.ndim != 2:
        raise ValueError(f"tail_angle_rad must be 2D, got shape {tail_angle.shape}.")
    if posture_frames.shape[0] != tail_angle.shape[0] or posture_valid.shape[0] != tail_angle.shape[0]:
        raise ValueError(
            "Tail posture source_acquisition_frame_index, valid, and "
            "tail_angle_rad row counts must match."
        )
    posture_instance_keys = _require_posture_instance_keys(
        posture,
        row_count=int(tail_angle.shape[0]),
    )
    angle_count = int(tail_angle.shape[1])
    if angle_count != MEGABOUTS_TAIL_SEGMENT_COUNT:
        raise ValueError(
            "Megabouts classifier input requires "
            f"{MEGABOUTS_TAIL_SEGMENT_COUNT} tail-angle channels, got {angle_count}."
        )

    bouts = swim_payload.bouts
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

    track_frames = track_arrays["frame_indices"]
    track_lookup = _unique_frame_to_index(
        track_frames,
        label=f"{track_refs['track_group']}/source_acquisition_frame_index",
    )
    posture_row_by_track_index = _join_posture_rows_to_track_sources(
        posture_instance_keys=posture_instance_keys,
        posture_frames=posture_frames,
        track_source_instance_keys=track_arrays["source_instance_key"],
        track_frames=track_frames,
    )

    for bout_idx, start in enumerate(window_start.tolist()):
        frames = np.arange(int(start), int(start) + int(window_frames), dtype=np.int64)
        for sample_idx, frame in enumerate(frames.tolist()):
            track_idx = track_lookup.get(int(frame))
            posture_idx = (
                int(posture_row_by_track_index[track_idx])
                if track_idx is not None
                else -1
            )
            if posture_idx >= 0 and bool(posture_valid[posture_idx]):
                values = tail_angle[posture_idx]
                if np.all(np.isfinite(values)):
                    tail_array[bout_idx, :, sample_idx] = values
                    tail_valid[bout_idx, sample_idx] = True

            if track_idx is not None and bool(track_arrays["sample_valid"][track_idx]):
                x_mm, y_mm = track_arrays["positions_mm"][track_idx]
                yaw = track_arrays["heading"][track_idx]
                values = np.asarray([x_mm, y_mm, yaw], dtype=np.float32)
                if np.all(np.isfinite(values)):
                    traj_array[bout_idx, :, sample_idx] = values
                    traj_valid[bout_idx, sample_idx] = True

    if bool(align_traj_to_onset):
        traj_array, traj_reference_valid = _align_traj_array_to_reference(
            traj_array,
            traj_valid,
            reference_index=int(traj_reference_index),
        )
    else:
        ref = int(traj_reference_index)
        if ref < 0 or ref >= int(window_frames):
            raise ValueError(f"trajectory reference index {ref} is outside window length {window_frames}.")
        traj_reference_valid = (
            traj_valid[:, ref] & np.all(np.isfinite(traj_array[:, :, ref]), axis=1)
            if n_bouts
            else np.asarray([], dtype=bool)
        )

    tail_fraction = np.mean(tail_valid, axis=1).astype(np.float32, copy=False) if n_bouts else np.asarray([], dtype=np.float32)
    traj_fraction = np.mean(traj_valid, axis=1).astype(np.float32, copy=False) if n_bouts else np.asarray([], dtype=np.float32)
    max_tail_invalid = np.asarray([_max_consecutive_false(row) for row in tail_valid], dtype=np.int32)
    max_traj_invalid = np.asarray([_max_consecutive_false(row) for row in traj_valid], dtype=np.int32)

    valid_bout = (
        (tail_fraction >= float(min_tail_valid_fraction))
        & (traj_fraction >= float(min_traj_valid_fraction))
        & (max_tail_invalid <= int(max_consecutive_invalid_frames))
        & (max_traj_invalid <= int(max_consecutive_invalid_frames))
        & traj_reference_valid
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
        if not bool(traj_reference_valid[idx]):
            failures.append("traj_reference_invalid")
        reasons[idx] = "|".join(failures) if failures else "invalid"

    source_refs = {
        "tail_posture_view_run": posture_path,
        "tail_frame_indices": f"{posture_path}/source_acquisition_frame_index",
        "tail_instance_key": f"{posture_path}/instance_key",
        "tail_angle_rad": f"{posture_path}/tail_angle_rad",
        "tail_valid": f"{posture_path}/valid",
        "tail_posture_publication_manifest_ref": posture_manifest_ref,
        "tail_posture_publication_manifest_sha256": posture_manifest_sha256,
        "tail_posture_source_subject_shape_run": posture_source_run_path,
        "tail_posture_source_subject_shape_publication_manifest_ref": (
            posture_source_manifest_ref
        ),
        "tail_posture_source_subject_shape_publication_manifest_sha256": (
            posture_source_manifest_sha256
        ),
        "track_kinematics_run": track_path,
        **track_refs,
        "swim_bout_run": swim_payload.run_path,
        "swim_bout_level": bout_level_path,
        "bouts": f"{bout_level_path}/bouts",
        "swim_bout_source_track_motion_manifest_sha256": str(
            swim_payload.run_attrs["source_track_motion_manifest_sha256"]
        ),
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
        "swim_bout_candidate_id": int(swim_payload.candidate.candidate_id),
        "swim_bout_signal_id": int(swim_payload.signal.signal_id),
        "heading_source": str(heading_source),
        "fps": float(fps),
        "bout_duration_s": float(window_frames) / float(fps),
        "bout_duration_frames": int(window_frames),
        "window_policy": "start_frame_fixed_duration",
        "tail_track_join_policy": (
            "posture_instance_key_to_track_source_instance_key_then_exact_"
            "acquisition_frame_v1"
        ),
        "traj_alignment": "onset_translation_rotation" if bool(align_traj_to_onset) else "none",
        "traj_reference_index": int(traj_reference_index),
        "requires_traj_reference_valid": True,
        "tail_array_shape": list(tail_array.shape),
        "traj_array_shape": list(traj_array.shape),
        "min_tail_valid_fraction": float(min_tail_valid_fraction),
        "min_traj_valid_fraction": float(min_traj_valid_fraction),
        "max_consecutive_invalid_frames": int(max_consecutive_invalid_frames),
        "mutates_archive": False,
        "calls_megabouts": False,
    }
    pack = MegaboutsClassifierInputPack(
        tail_array=tail_array,
        traj_array=traj_array,
        tail_valid=tail_valid,
        traj_valid=traj_valid,
        traj_reference_valid=traj_reference_valid,
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
    _require_unchanged_tail_posture_publication(
        root,
        posture_path,
        expected_manifest_ref=posture_manifest_ref,
        expected_manifest_sha256=posture_manifest_sha256,
        expected_source_run_path=posture_source_run_path,
        expected_source_manifest_ref=posture_source_manifest_ref,
        expected_source_manifest_sha256=posture_source_manifest_sha256,
        error_message=(
            "Tail posture or its source subject-shape publication changed while "
            "Megabouts inputs were copied."
        ),
    )
    return pack


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
        "traj_alignment": pack.parameters.get("traj_alignment"),
        "traj_reference_index": pack.parameters.get("traj_reference_index"),
        "traj_reference_valid_count": int(np.count_nonzero(pack.traj_reference_valid)),
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

    _require_input_pack_tail_posture_publication(
        root,
        pack.source_refs,
        error_message=(
            "Megabouts diagnostic source tail-posture publication changed since "
            "the input pack was built."
        ),
    )
    posture = root[pack.source_refs["tail_posture_view_run"]]
    posture_frames = np.asarray(
        _require_array(posture, "source_acquisition_frame_index")[:],
        dtype=np.int64,
    )
    posture_valid = np.asarray(_require_array(posture, "valid")[:], dtype=bool)
    tail_angle = np.asarray(_require_array(posture, "tail_angle_rad")[:], dtype=np.float32)
    posture_reasons = _load_reason_array(posture, ("failure_reason_bytes", "reason_bytes"))
    posture_instance_keys = _require_posture_instance_keys(
        posture,
        row_count=int(tail_angle.shape[0]),
    )

    track_tables = load_track_kinematics_track(
        root,
        run_name=str(pack.parameters["track_kinematics_run"]),
        scope=str(pack.parameters["track_kinematics_scope"]),
        track_id=int(pack.parameters["track_id"]),
        required_speed_levels=(),
    )
    if track_tables.motion_manifest_sha256 != pack.source_refs.get(
        "track_motion_manifest_sha256"
    ):
        raise ValueError(
            "Megabouts diagnostic source track-motion manifest changed since "
            "the input pack was built."
        )
    track_frames = np.asarray(
        track_tables.source_acquisition_frame_index,
        dtype=np.int64,
    )
    if track_tables.positions_mm is None:
        raise ValueError("Verified track input has no physical positions_mm.")
    positions_mm = np.asarray(track_tables.positions_mm, dtype=np.float32)
    heading_path = pack.source_refs["heading"].split("/")[-1]
    heading_sources = {
        "heading_degrees": track_tables.heading_degrees,
        "heading_radians": track_tables.heading_radians,
        "smoothed_heading_degrees": track_tables.smoothed_heading_degrees,
        "smoothed_heading_radians": track_tables.smoothed_heading_radians,
    }
    if heading_path not in heading_sources or heading_sources[heading_path] is None:
        raise ValueError(f"Verified heading source {heading_path!r} is unavailable.")
    heading = np.asarray(heading_sources[heading_path], dtype=np.float32)
    if track_tables.sample_valid is None:
        raise ValueError("Verified track input has no sample_valid surface.")
    track_valid = np.asarray(track_tables.sample_valid, dtype=bool)
    track_source_instance_keys = _require_track_source_instance_keys(
        track_tables.source_instance_key,
        row_count=int(track_frames.shape[0]),
        track_path=str(track_tables.track_path),
    )
    track_reasons = None
    if track_tables.sample_reason_code is not None:
        raw_reason_codes = track_tables.track_attrs.get("sample_reason_codes")
        if not isinstance(raw_reason_codes, Mapping):
            raise ValueError(
                "Verified track input has sample_reason_code values but no "
                "controlled sample_reason_codes codebook."
            )
        reason_codebook: dict[int, str] = {}
        for raw_code, raw_label in raw_reason_codes.items():
            try:
                code = int(raw_code)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Verified track sample_reason_codes contains a non-integer key."
                ) from exc
            label = str(raw_label).strip()
            if not label:
                raise ValueError(
                    f"Verified track sample_reason_codes[{code}] has an empty label."
                )
            reason_codebook[code] = label
        decoded_reasons: list[str] = []
        for raw_code in np.asarray(track_tables.sample_reason_code).tolist():
            code = int(raw_code)
            if code not in reason_codebook:
                raise ValueError(
                    "Verified track sample_reason_code contains an unknown "
                    f"controlled value: {code}."
                )
            decoded_reasons.append(reason_codebook[code])
        track_reasons = np.asarray(decoded_reasons, dtype=object)
    track_lookup = _unique_frame_to_index(
        track_frames,
        label=f"{track_tables.track_path}/source_acquisition_frame_index",
    )
    posture_row_by_track_index = _join_posture_rows_to_track_sources(
        posture_instance_keys=posture_instance_keys,
        posture_frames=posture_frames,
        track_source_instance_keys=track_source_instance_keys,
        track_frames=track_frames,
    )

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
            track_idx = track_lookup.get(int(frame))
            posture_idx = (
                int(posture_row_by_track_index[track_idx])
                if track_idx is not None
                else -1
            )
            if track_idx is None:
                missing_posture_frames.append(int(frame))
                tail_issue_counts["missing_track_row_for_posture_join"] += 1
            elif not bool(track_source_instance_keys["valid"][track_idx]):
                missing_posture_frames.append(int(frame))
                tail_issue_counts["track_source_instance_key_null"] += 1
            elif posture_idx < 0:
                missing_posture_frames.append(int(frame))
                tail_issue_counts["missing_posture_instance_key"] += 1
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
    safe_result = dict(_json_safe(result))
    _require_input_pack_tail_posture_publication(
        root,
        pack.source_refs,
        error_message=(
            "Tail posture or its source subject-shape publication changed while "
            "Megabouts diagnostics were copied."
        ),
    )
    return safe_result


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
        "--no-align-traj-to-onset",
        action="store_true",
        help="Disable Megabouts-style onset-frame translation/rotation for trajectory windows.",
    )
    parser.add_argument("--traj-reference-index", type=int, default=DEFAULT_TRAJ_REFERENCE_INDEX)
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
        "align_traj_to_onset": not bool(args.no_align_traj_to_onset),
        "traj_reference_index": int(args.traj_reference_index),
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
