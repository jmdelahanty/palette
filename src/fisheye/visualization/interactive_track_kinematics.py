"""Load track-kinematics interactive visualization specs from Zarr."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd
import zarr

from fisheye.analysis.bout_kinematics import resolve_bout_kinematics_tables
from fisheye.analysis.track_kinematics_io import (
    TrackKinematicsTrackTables,
    load_track_kinematics_track,
)
from fisheye.analysis.eye_angle_io import (
    EYE_ANGLE_TIMESERIES_COLUMNS,
    EyeAngleRunOption,
    discover_eye_angle_run_options as discover_eye_angle_run_options_from_root,
    first_array_length,
    frame_time_seconds,
    load_eye_angle_run_tables,
    optional_1d_array,
    roi_frame_indices,
    roi_time_seconds,
)
from fisheye.analysis.swim_bout_io import (
    SwimBoutIOError,
    discover_swim_bout_candidates as discover_swim_bout_candidates_resolved,
    load_swim_bout_tables,
)
from fisheye.shared.zarr_io import open_zarr_root


TRACK_KINEMATICS_SPEC_SCHEMA_ID = "palette.plot_spec.track_kinematics_summary.v1"
DEFAULT_INTERACTIVE_ARTIFACT = "track_kinematics_summary_track_0_interactive"
TRACK_KINEMATICS_VISUALIZATION_RUNS_PATH = (
    "analysis/track_kinematics_visualization_runs"
)


@dataclass(frozen=True)
class TrackKinematicsInteractiveData:
    zarr_path: Path
    run_path: str
    artifact_name: str
    spec: Mapping[str, Any]
    attrs: Mapping[str, Any]
    source_paths: Mapping[str, str]
    time_seconds: np.ndarray
    frame_indices: Optional[np.ndarray]
    series: dict[str, np.ndarray]
    positions: Optional[np.ndarray]
    position_unit: str
    swim_bouts: np.ndarray
    swim_bout_records: np.ndarray
    inter_bout_intervals: np.ndarray
    swim_bout_label: Optional[str]
    swim_bout_source: Optional[str]
    validity_spans: np.ndarray
    validity_labels: np.ndarray
    validity_source: Optional[str]


@dataclass(frozen=True)
class EyeAngleTimeseriesData:
    zarr_path: Path
    run_name: str
    run_path: str
    row_axis: str
    attrs: Mapping[str, Any]
    dataframe: pd.DataFrame


@dataclass(frozen=True)
class TrackKinematicsRunOption:
    run_name: str
    run_path: str
    run_scope: str
    artifact_name: str
    track_id: int
    label: str
    is_latest: bool
    attrs: Mapping[str, Any]


@dataclass(frozen=True)
class SwimBoutRunOption:
    run_name: str
    label: str
    layout: str
    candidate_id: int
    signal_id: int
    signal_role: str
    default_level: str
    speed_level: str
    source_track_kinematics_run: Optional[str]
    track_id: Optional[int]
    detection_method: str
    threshold_mm: Optional[float]
    exponential_tau_s: Optional[float]
    exponential_source_level: Optional[str]
    n_bouts_by_level: Mapping[str, int]
    is_latest: bool


@dataclass(frozen=True)
class BoutKinematicsRunOption:
    run_name: str
    label: str
    default_heading_level: str
    heading_levels: tuple[str, ...]
    pre_post_mode: Optional[str]
    source_track_kinematics_run: Optional[str]
    source_track_id: Optional[int]
    source_swim_bout_run: Optional[str]
    source_swim_bout_speed_level: Optional[str]
    n_rows_by_level: Mapping[str, int]
    is_latest: bool


@dataclass(frozen=True)
class BoutClassificationRunOption:
    run_name: str
    label: str
    classifier_family: str
    classifier_name: str
    classifier_version: Optional[str]
    source_swim_bout_run: Optional[str]
    source_swim_bout_speed_level: Optional[str]
    source_bout_count: int
    classified_bout_count: int
    skipped_bout_count: int
    is_latest: bool
    attrs: Mapping[str, Any]


@dataclass(frozen=True)
class _SwimBoutPayload:
    spans: np.ndarray
    records: np.ndarray
    inter_bout_intervals: np.ndarray
    series: Mapping[str, np.ndarray]
    label: Optional[str]
    source: Optional[str]


@dataclass(frozen=True)
class _BoutKinematicsPayload:
    records_by_level: dict[str, np.ndarray]
    label: Optional[str]
    source: Optional[str]


def _normalize_path(path: str) -> str:
    return "/".join(part for part in str(path).strip("/").split("/") if part)


def _join_path(*parts: str) -> str:
    return "/".join(_normalize_path(part) for part in parts if _normalize_path(part))


def _json_from_uint8_array(array: zarr.Array) -> Mapping[str, Any]:
    payload = np.asarray(array[:], dtype=np.uint8).tobytes()
    digest = hashlib.sha256(payload).hexdigest()
    if array.attrs.get("content_sha256") != digest:
        raise ValueError("interactive spec_json payload failed digest validation")
    parsed = json.loads(payload.decode("utf-8"))
    if not isinstance(parsed, Mapping):
        raise ValueError("interactive spec payload must be a JSON object")
    return parsed


def _as_str_mapping(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): str(item) for key, item in value.items()}


def _group_keys(group: object) -> list[str]:
    keys_fn = getattr(group, "group_keys", None)
    if not callable(keys_fn):
        return []
    try:
        return sorted(str(key) for key in keys_fn())
    except Exception:
        return []


def _track_id_from_interactive_artifact_name(artifact_name: str) -> int:
    prefix = "track_kinematics_summary_track_"
    suffix = "_interactive"
    if not artifact_name.startswith(prefix) or not artifact_name.endswith(suffix):
        raise ValueError(
            "Track interactive artifact names must encode their track id as "
            f"{prefix}<nonnegative integer>{suffix}."
        )
    track_id = _safe_int(artifact_name[len(prefix) : -len(suffix)])
    if track_id is None or track_id < 0:
        raise ValueError("Track interactive artifact name has an invalid track id.")
    return int(track_id)


def _resolve_interactive_artifact(
    root: zarr.Group,
    *,
    run_path: str,
    artifact_name: str,
) -> zarr.Group:
    parts = _normalize_path(run_path).split("/")
    if (
        len(parts) != 4
        or parts[:2] != ["analysis", "track_kinematics_runs"]
        or parts[2] not in {"online", "offline"}
    ):
        raise ValueError(
            "Interactive track-motion reads require the canonical source run "
            "path analysis/track_kinematics_runs/<scope>/<run>."
        )
    track_id = _track_id_from_interactive_artifact_name(artifact_name)
    visualization_parent_path = _join_path(
        TRACK_KINEMATICS_VISUALIZATION_RUNS_PATH,
        parts[2],
        parts[3],
        "tracks",
        f"id_{track_id}",
    )
    try:
        visualization_parent = root[visualization_parent_path]
    except Exception as exc:
        raise ValueError(
            f"No sibling visualization runs found for {run_path}."
        ) from exc
    render_name = visualization_parent.attrs.get("latest_complete")
    if (
        not isinstance(render_name, str)
        or render_name not in visualization_parent
    ):
        raise ValueError(
            f"No selected complete visualization run found for {run_path}."
        )
    render = visualization_parent[render_name]
    if (
        render.attrs.get("palette_run_completion_status") != "complete"
        or render.attrs.get("stage_selector_eligible") is not True
    ):
        raise ValueError(
            f"Selected visualization run {render.path!r} is not complete and eligible."
        )
    artifact_path = _join_path(render.path, "visualizations", artifact_name)
    try:
        artifact = root[artifact_path]
    except Exception as exc:
        raise ValueError(f"Interactive artifact not found: {artifact_path}") from exc
    if not isinstance(artifact, zarr.Group) and not hasattr(artifact, "group_keys"):
        raise ValueError(f"Interactive artifact is not a group: {artifact_path}")
    if "spec_json" not in artifact:
        raise ValueError(f"Interactive artifact missing spec_json: {artifact_path}")
    return artifact


def _try_resolve_interactive_artifact(
    root: zarr.Group,
    *,
    run_path: str,
    artifact_name: str,
) -> Optional[zarr.Group]:
    try:
        return _resolve_interactive_artifact(root, run_path=run_path, artifact_name=artifact_name)
    except ValueError:
        return None


def _load_verified_track_for_spec(
    root: zarr.Group,
    *,
    run_path: str,
    artifact: zarr.Group,
    spec: Mapping[str, Any],
) -> TrackKinematicsTrackTables:
    parts = _normalize_path(run_path).split("/")
    if (
        len(parts) != 4
        or parts[:2] != ["analysis", "track_kinematics_runs"]
        or parts[2] not in {"online", "offline"}
    ):
        raise ValueError("Track interactive spec source run path is not canonical.")
    track_id = _safe_int(spec.get("track_id"))
    if track_id is None:
        raise ValueError("Track interactive spec omits an exact track_id.")
    tables = load_track_kinematics_track(
        root,
        run_name=parts[3],
        scope=parts[2],
        track_id=track_id,
        required_speed_levels=("raw", "smoothed"),
    )
    authority = tables.authority_record()
    if (
        artifact.attrs.get("content_sha256")
        != artifact["spec_json"].attrs.get("content_sha256")
    ):
        raise ValueError("Interactive artifact and spec_json digests disagree.")
    if (
        spec.get("track_motion_authority") != authority
        or artifact.attrs.get("track_motion_authority") != authority
    ):
        raise ValueError(
            "Interactive artifact does not bind the freshly verified "
            "track-motion authority."
        )
    from fisheye.analysis.plot_track_kinematics import _track_source_paths

    expected_paths = _track_source_paths(
        f"{parts[2]}/{parts[3]}",
        tables,
    )
    if _as_str_mapping(spec.get("source_paths")) != expected_paths:
        raise ValueError(
            "Interactive artifact source paths differ from the manifest-authorized "
            "track-motion surfaces."
        )
    return tables


def _load_array(root: zarr.Group, source_paths: Mapping[str, str], key: str) -> Optional[np.ndarray]:
    path = source_paths.get(key)
    if not path:
        return None
    try:
        return np.asarray(root[path][:])
    except Exception:
        return None


def _load_track_array(
    root: zarr.Group,
    source_paths: Mapping[str, str],
    *,
    run_path: str,
    track_id: int,
    key: str,
) -> Optional[np.ndarray]:
    values = _load_array(root, source_paths, key)
    if values is not None:
        return values
    movement_path = _movement_speed_path_for_track_key(run_path, track_id, key)
    if movement_path is not None:
        try:
            return np.asarray(root[movement_path][:])
        except Exception:
            pass
    direct_path = _join_path(run_path, "tracks", f"id_{int(track_id)}", key)
    try:
        return np.asarray(root[direct_path][:])
    except Exception:
        return None


def _movement_speed_path_for_track_key(
    run_path: str,
    track_id: int,
    key: str,
) -> Optional[str]:
    """Return the v2 movement/speed path for a legacy logical series key."""

    level_by_prefix = {
        "speed_raw": "raw",
        "speed_filtered": "filtered",
        "speed_smoothed": "smoothed",
        "speed_averaged": "averaged",
    }
    for prefix, level in level_by_prefix.items():
        if key == f"{prefix}_px":
            return _join_path(run_path, "tracks", f"id_{int(track_id)}", "movement", "speed", level, "px")
        if key == f"{prefix}_mm":
            return _join_path(run_path, "tracks", f"id_{int(track_id)}", "movement", "speed", level, "mm")
        for suffix in (
            "acceleration_px",
            "acceleration_mm",
            "smoothed_acceleration_px",
            "smoothed_acceleration_mm",
            "frame_path_distance_px",
            "frame_path_distance_mm",
        ):
            if key == f"{prefix}_{suffix}":
                return _join_path(
                    run_path,
                    "tracks",
                    f"id_{int(track_id)}",
                    "movement",
                    "speed",
                    level,
                    suffix,
                )

    if key in {"acceleration_px", "acceleration_mm", "smoothed_acceleration_px", "smoothed_acceleration_mm"}:
        return _join_path(run_path, "tracks", f"id_{int(track_id)}", "movement", "speed", "smoothed", key)

    return None


def _load_track_attrs(
    root: zarr.Group,
    *,
    run_path: str,
    track_id: int,
) -> Mapping[str, Any]:
    direct_path = _join_path(run_path, "tracks", f"id_{int(track_id)}")
    try:
        track_group = root[direct_path]
    except Exception:
        return {}
    attrs = getattr(track_group, "attrs", {})
    return dict(attrs)


def _empty_spans() -> np.ndarray:
    return np.empty((0, 2), dtype=np.float64)


def _empty_labels() -> np.ndarray:
    return np.empty((0,), dtype=object)


def _empty_records() -> np.ndarray:
    return np.zeros(0, dtype=[])


def _empty_swim_bout_payload() -> _SwimBoutPayload:
    return _SwimBoutPayload(
        spans=_empty_spans(),
        records=_empty_records(),
        inter_bout_intervals=_empty_records(),
        series={},
        label=None,
        source=None,
    )


def _speed_level_key(speed_level: Optional[str], default: str = "speed_smoothed") -> str:
    level = str(speed_level or default).strip()
    level_map = {
        "raw": "speed_raw",
        "filtered": "speed_filtered",
        "smoothed": "speed_smoothed",
        "averaged": "speed_averaged",
        "exp": "speed_exponential",
        "exponential": "speed_exponential",
    }
    return level_map.get(level, level if level.startswith("speed_") else f"speed_{level}")


def _heading_level_key(heading_level: Optional[str], default: str = "heading_smoothed") -> str:
    level = str(heading_level or default).strip()
    level_map = {
        "raw": "heading_raw",
        "smoothed": "heading_smoothed",
    }
    return level_map.get(level, level if level.startswith("heading_") else f"heading_{level}")


def _spans_from_start_stop(starts: np.ndarray, stops: np.ndarray) -> np.ndarray:
    rows: list[tuple[float, float]] = []
    for start, stop in zip(starts, stops):
        if not (np.isfinite(start) and np.isfinite(stop)):
            continue
        if float(stop) < float(start):
            continue
        rows.append((float(start), float(stop)))
    if not rows:
        return _empty_spans()
    return np.asarray(rows, dtype=np.float64)


def _reason_name(
    mapping: object,
    code: object,
    *,
    prefix: str,
) -> str:
    if isinstance(mapping, Mapping):
        key = str(int(code)) if _safe_int(code) is not None else str(code)
        value = mapping.get(key)
        if value is not None:
            return str(value)
    parsed = _safe_int(code)
    return f"{prefix}_{parsed}" if parsed is not None else prefix


def _row_interval(time_seconds: np.ndarray, index: int) -> tuple[float, float] | None:
    if index < 0 or index >= time_seconds.shape[0]:
        return None
    center = float(time_seconds[index])
    if not np.isfinite(center):
        return None
    if time_seconds.shape[0] == 1:
        return (center, center)
    if index == 0:
        left = center
    else:
        left = (float(time_seconds[index - 1]) + center) / 2.0
    if index + 1 >= time_seconds.shape[0]:
        right = center
    else:
        right = (center + float(time_seconds[index + 1])) / 2.0
    if not (np.isfinite(left) and np.isfinite(right)):
        return None
    if right < left:
        left, right = right, left
    if right == left and time_seconds.shape[0] > 1:
        if index > 0:
            left = min(float(time_seconds[index - 1]), center)
        if index + 1 < time_seconds.shape[0]:
            right = max(center, float(time_seconds[index + 1]))
    if right <= left:
        return None
    return (float(left), float(right))


def _load_validity_spans(
    root: zarr.Group,
    *,
    run_path: str,
    track_id: int,
    source_paths: Mapping[str, str],
    time_seconds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Optional[str]]:
    track_attrs = _load_track_attrs(root, run_path=run_path, track_id=track_id)
    transition_valid = _load_track_array(
        root,
        source_paths,
        run_path=run_path,
        track_id=track_id,
        key="transition_valid",
    )
    transition_reason = _load_track_array(
        root,
        source_paths,
        run_path=run_path,
        track_id=track_id,
        key="transition_reason_code",
    )
    sample_valid = _load_track_array(
        root,
        source_paths,
        run_path=run_path,
        track_id=track_id,
        key="sample_valid",
    )
    sample_reason = _load_track_array(
        root,
        source_paths,
        run_path=run_path,
        track_id=track_id,
        key="sample_reason_code",
    )

    rows: list[tuple[float, float, str]] = []
    n_time = int(time_seconds.shape[0])
    transition_map = track_attrs.get("transition_reason_codes", {})
    sample_map = track_attrs.get("sample_reason_codes", {})

    if transition_valid is not None and transition_valid.shape[0] == n_time:
        transition_valid = np.asarray(transition_valid, dtype=bool)
        if transition_reason is not None and transition_reason.shape[0] == n_time:
            transition_reason_values = np.asarray(transition_reason)
        else:
            transition_reason_values = np.full(n_time, -1, dtype=np.int16)
        for index in range(1, n_time):
            if bool(transition_valid[index]):
                continue
            start = float(time_seconds[index - 1])
            stop = float(time_seconds[index])
            if not (np.isfinite(start) and np.isfinite(stop)):
                continue
            if stop < start:
                start, stop = stop, start
            if stop <= start:
                continue
            reason = _reason_name(
                transition_map,
                transition_reason_values[index],
                prefix="transition",
            )
            if reason == "first_sample":
                continue
            rows.append((start, stop, f"transition:{reason}"))

    if sample_valid is not None and sample_valid.shape[0] == n_time:
        sample_valid = np.asarray(sample_valid, dtype=bool)
        if sample_reason is not None and sample_reason.shape[0] == n_time:
            sample_reason_values = np.asarray(sample_reason)
        else:
            sample_reason_values = np.full(n_time, -1, dtype=np.int16)
        for index, valid in enumerate(sample_valid):
            if bool(valid):
                continue
            interval = _row_interval(time_seconds, index)
            if interval is None:
                continue
            reason = _reason_name(
                sample_map,
                sample_reason_values[index],
                prefix="sample",
            )
            rows.append((interval[0], interval[1], f"sample:{reason}"))

    if not rows:
        return _empty_spans(), _empty_labels(), None
    spans = np.asarray([(start, stop) for start, stop, _label in rows], dtype=np.float64)
    labels = np.asarray([label for _start, _stop, label in rows], dtype=object)
    return spans, labels, "track_validity"


def _spans_from_structured_bouts(bouts: np.ndarray, fps: Optional[float]) -> np.ndarray:
    if bouts.size == 0:
        return _empty_spans()
    names = bouts.dtype.names or ()
    if "start_time_s" in names and "end_time_s" in names:
        return _spans_from_start_stop(bouts["start_time_s"], bouts["end_time_s"])
    if "start_frame" in names and "end_frame" in names and fps:
        return _spans_from_start_stop(
            np.asarray(bouts["start_frame"], dtype=np.float64) / float(fps),
            np.asarray(bouts["end_frame"], dtype=np.float64) / float(fps),
        )
    return _empty_spans()


def _load_structured_or_empty(group: zarr.Group, name: str) -> np.ndarray:
    from fisheye.shared.zarr.columnar import load_structured_dataset

    try:
        records, _attrs = load_structured_dataset(group, name)
    except Exception:
        return _empty_records()
    return np.asarray(records)


def _decode_structured_value(value: object) -> object:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _records_match_bout_ids(records: np.ndarray, peak_events: np.ndarray) -> bool:
    record_names = records.dtype.names or ()
    peak_names = peak_events.dtype.names or ()
    if "bout_id" not in record_names or "bout_id" not in peak_names:
        return True
    try:
        return bool(np.array_equal(records["bout_id"], peak_events["bout_id"]))
    except Exception:
        return False


def _augment_bouts_with_peak_events(
    records: np.ndarray,
    peak_events: np.ndarray,
    *,
    fps: Optional[float],
) -> np.ndarray:
    """Append aligned peak-event review columns to bout records."""

    if records.size == 0 or peak_events.size == 0:
        return records
    if records.dtype.names is None or peak_events.dtype.names is None:
        return records
    if records.shape[0] != peak_events.shape[0] or not _records_match_bout_ids(records, peak_events):
        return records

    record_names = set(records.dtype.names or ())
    extra_fields: list[tuple[str, object, np.ndarray]] = []

    numeric_fields = (
        "peak_frame",
        "peak_time_s",
        "peak_signal_value_mm_s",
        "peak_prominence_mm_s",
        "peak_width_samples",
        "peak_width_s",
        "peak_width_height_mm_s",
        "left_ips",
        "right_ips",
        "left_width_frame_interpolated",
        "right_width_frame_interpolated",
        "left_base_frame",
        "right_base_frame",
        "left_base_signal_value_mm_s",
        "right_base_signal_value_mm_s",
    )
    for field in numeric_fields:
        if field not in peak_events.dtype.names:
            continue
        output_name = f"peak_event_{field}"
        if output_name in record_names:
            continue
        values = np.asarray(peak_events[field])
        extra_fields.append((output_name, values.dtype, values))

    for field in ("boundary_mode", "shape_split_policy"):
        if field not in peak_events.dtype.names:
            continue
        output_name = f"peak_event_{field}"
        if output_name in record_names:
            continue
        values = np.asarray([_decode_structured_value(value) for value in peak_events[field]], dtype=object)
        extra_fields.append((output_name, object, values))

    if fps is not None and fps > 0:
        for source_name, output_name in (
            ("left_width_frame_interpolated", "peak_event_left_width_time_s"),
            ("right_width_frame_interpolated", "peak_event_right_width_time_s"),
        ):
            if source_name not in peak_events.dtype.names or output_name in record_names:
                continue
            values = np.asarray(peak_events[source_name], dtype=np.float64) / float(fps)
            extra_fields.append((output_name, np.float64, values))

    if not extra_fields:
        return records

    dtype = list(records.dtype.descr)
    dtype.extend((name, dtype_spec) for name, dtype_spec, _values in extra_fields)
    augmented = np.empty(records.shape, dtype=np.dtype(dtype))
    for name in records.dtype.names or ():
        augmented[name] = records[name]
    for name, _dtype_spec, values in extra_fields:
        augmented[name] = values
    return augmented


def _structured_to_dataframe(records: np.ndarray) -> pd.DataFrame:
    if records.size == 0 or records.dtype.names is None:
        return pd.DataFrame()
    return pd.DataFrame.from_records(records)


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _safe_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _format_number(value: object) -> str:
    parsed = _safe_float(value)
    if parsed is None:
        return "?"
    if abs(parsed - round(parsed)) < 1e-9:
        return str(int(round(parsed)))
    return f"{parsed:g}"


def _track_option_label(
    *,
    scope: str,
    run_name: str,
    run_group: zarr.Group,
    track_id: int,
    is_latest: bool,
) -> str:
    label = f"{scope}/{run_name}"
    high = run_group.attrs.get("hysteresis_high_px")
    low = run_group.attrs.get("hysteresis_low_px")
    smooth = run_group.attrs.get("smoothing_seconds")
    pieces: list[str] = []
    if high is not None or low is not None:
        pieces.append(f"hyst {_format_number(high)}/{_format_number(low)} px")
    if smooth is not None:
        pieces.append(f"smooth {_format_number(smooth)} s")
    pieces.append(f"track {track_id}")
    if is_latest:
        pieces.append("latest")
    return f"{label} | " + " | ".join(pieces)


def _columnar_row_count(group: zarr.Group, dataset_name: str) -> int:
    if dataset_name not in group:
        return 0
    node = group[dataset_name]
    if isinstance(node, zarr.Array):
        return int(node.shape[0])
    if isinstance(node, zarr.Group):
        fields = list(node.attrs.get("field_names", []))
        if fields and fields[0] in node:
            return int(node[fields[0]].shape[0])
    return 0


def _swim_option_label(
    *,
    run_name: str,
    speed_level: str,
    is_default_level: bool,
    detection_method: str,
    threshold_mm: Optional[float],
    exponential_tau_s: Optional[float] = None,
    exponential_source_level: Optional[str] = None,
    count: int,
    is_latest: bool,
) -> str:
    pieces = [run_name, speed_level, f"{count} bouts"]
    if threshold_mm is not None:
        pieces.append(f"threshold {threshold_mm:g} mm/s")
    if speed_level == "exponential" and exponential_tau_s is not None:
        source = (exponential_source_level or "speed_filtered").replace("speed_", "")
        pieces.append(f"exp tau {exponential_tau_s:g} s from {source}")
    if detection_method:
        pieces.append(f"method {detection_method}")
    if is_default_level:
        pieces.append("default")
    if is_latest:
        pieces.append("latest")
    return " | ".join(pieces)


def _bout_classification_option_label(
    *,
    run_name: str,
    classifier_family: str,
    classifier_name: str,
    classifier_version: Optional[str],
    source_bout_count: int,
    classified_bout_count: int,
    skipped_bout_count: int,
    speed_level: Optional[str],
    is_latest: bool,
) -> str:
    pieces = [run_name]
    classifier = classifier_family
    if classifier_name:
        classifier = f"{classifier_family}/{classifier_name}" if classifier_family else classifier_name
    if classifier_version:
        classifier = f"{classifier} v{classifier_version}"
    if classifier:
        pieces.append(classifier)
    pieces.append(f"{classified_bout_count}/{source_bout_count} classified")
    if skipped_bout_count:
        pieces.append(f"{skipped_bout_count} skipped")
    if speed_level:
        pieces.append(f"source {speed_level.replace('speed_', '')}")
    if is_latest:
        pieces.append("latest")
    return " | ".join(pieces)


def _classification_source_from_attrs(
    attrs: Mapping[str, Any],
) -> tuple[Optional[str], Optional[str]]:
    parameters = attrs.get("parameters")
    parameters = dict(parameters) if isinstance(parameters, Mapping) else {}
    source_refs = attrs.get("source_refs")
    source_refs = dict(source_refs) if isinstance(source_refs, Mapping) else {}

    source_run = parameters.get("swim_bout_run")
    source_speed = parameters.get("speed_level")
    swim_bout_level = source_refs.get("swim_bout_level") or source_refs.get("bouts")
    if (source_run is None or source_speed is None) and isinstance(swim_bout_level, str):
        parts = _normalize_path(swim_bout_level).split("/")
        try:
            index = parts.index("swim_bout_runs")
        except ValueError:
            index = -1
        if index >= 0 and len(parts) > index + 1 and source_run is None:
            source_run = parts[index + 1]
        if index >= 0 and len(parts) > index + 2 and source_speed is None:
            source_speed = parts[index + 2]

    source_speed_key = _speed_level_key(source_speed) if source_speed is not None else None
    return (
        str(source_run) if source_run is not None else None,
        source_speed_key,
    )


def _classification_matches_swim_bout(
    attrs: Mapping[str, Any],
    *,
    swim_bout_run: str,
    speed_level: str,
) -> bool:
    source_run, source_speed = _classification_source_from_attrs(attrs)
    if source_run is not None and str(source_run) != str(swim_bout_run):
        return False
    speed_level_key = _speed_level_key(speed_level)
    if source_speed is not None and source_speed != speed_level_key:
        return False
    return source_run is not None or source_speed is not None


def discover_eye_angle_run_options(
    zarr_path: Path | str,
    *,
    legacy_compatibility: bool = False,
) -> list[EyeAngleRunOption]:
    """Return available eye-angle analysis runs."""

    root = open_zarr_root(Path(zarr_path), mode="r")
    return discover_eye_angle_run_options_from_root(
        root,
        legacy_compatibility=legacy_compatibility,
    )


def load_eye_angle_timeseries_data(
    zarr_path: Path | str,
    *,
    run_name: Optional[str] = None,
    prefer_frame: bool = True,
    legacy_compatibility: bool = False,
) -> EyeAngleTimeseriesData:
    """Load eye-angle arrays into a time-indexed dataframe for interactive plotting."""

    archive = Path(zarr_path)
    root = open_zarr_root(archive, mode="r")
    tables = load_eye_angle_run_tables(
        root,
        run_name=run_name,
        legacy_compatibility=legacy_compatibility,
    )
    resolved_run = tables.run_name

    frame_rows = (
        first_array_length(tables.frame)
        if tables.frame
        else 0
    )
    if prefer_frame and tables.frame and frame_rows > 0:
        data_arrays = tables.frame
        row_axis = "frame"
        row_count = frame_rows
        qa_arrays = tables.qa_frame
        time_seconds = frame_time_seconds(tables, row_count=row_count)
        frame_indices = np.arange(row_count, dtype=np.int64)
    else:
        data_arrays = tables.roi
        row_axis = "roi"
        row_count = first_array_length(tables.roi)
        qa_arrays = tables.qa_roi
        time_seconds = roi_time_seconds(tables, row_count=row_count)
        frame_indices = roi_frame_indices(tables, row_count=row_count)

    if row_count <= 0:
        dataframe = pd.DataFrame(columns=["time_s"])
    else:
        if time_seconds is None:
            fps = _safe_float(tables.attrs.get("fps"))
            if frame_indices is not None and fps and fps > 0:
                time_seconds = frame_indices.astype(np.float64, copy=False) / float(fps)
            elif fps and fps > 0:
                time_seconds = np.arange(row_count, dtype=np.float64) / float(fps)
            else:
                time_seconds = np.arange(row_count, dtype=np.float64)
        dataframe = pd.DataFrame({"time_s": np.asarray(time_seconds, dtype=np.float64)})
        if frame_indices is not None and len(frame_indices) == row_count:
            dataframe["frame_index"] = np.asarray(frame_indices, dtype=np.int64)
        else:
            dataframe["row_index"] = np.arange(row_count, dtype=np.int64)

        for column_name in EYE_ANGLE_TIMESERIES_COLUMNS:
            values = optional_1d_array(data_arrays, column_name, length=row_count)
            if values is None:
                continue
            dataframe[column_name] = values.astype(np.float64, copy=False)

        for qa_name in ("valid_frame", "valid_left", "valid_right"):
            values = optional_1d_array(qa_arrays, qa_name, length=row_count)
            if values is not None:
                dataframe[qa_name] = values.astype(bool, copy=False)

    return EyeAngleTimeseriesData(
        zarr_path=archive,
        run_name=resolved_run,
        run_path=tables.run_path,
        row_axis=row_axis,
        attrs=tables.attrs,
        dataframe=dataframe,
    )


def discover_track_kinematics_run_options(
    zarr_path: Path | str,
    *,
    artifact_name: str = DEFAULT_INTERACTIVE_ARTIFACT,
) -> list[TrackKinematicsRunOption]:
    """Return track-kinematics runs that have a persisted interactive plot spec."""

    root = open_zarr_root(Path(zarr_path), mode="r")
    parent = root.get("analysis/track_kinematics_runs")
    if parent is None:
        return []

    options: list[TrackKinematicsRunOption] = []
    for scope in _group_keys(parent):
        scope_group = parent[scope]
        latest = scope_group.attrs.get("latest")
        for run_name in _group_keys(scope_group):
            run_path = _join_path("analysis/track_kinematics_runs", scope, run_name)
            artifact = _try_resolve_interactive_artifact(
                root,
                run_path=run_path,
                artifact_name=artifact_name,
            )
            if artifact is None:
                continue
            spec = _json_from_uint8_array(artifact["spec_json"])
            if spec.get("schema_id") != TRACK_KINEMATICS_SPEC_SCHEMA_ID:
                continue
            track_id = _safe_int(spec.get("track_id"))
            if track_id is None:
                continue
            try:
                _load_verified_track_for_spec(
                    root,
                    run_path=run_path,
                    artifact=artifact,
                    spec=spec,
                )
            except ValueError:
                continue
            run_group = scope_group[run_name]
            is_latest = str(latest) == str(run_name)
            options.append(
                TrackKinematicsRunOption(
                    run_name=run_name,
                    run_path=run_path,
                    run_scope=scope,
                    artifact_name=artifact_name,
                    track_id=track_id,
                    label=_track_option_label(
                        scope=scope,
                        run_name=run_name,
                        run_group=run_group,
                        track_id=track_id,
                        is_latest=is_latest,
                    ),
                    is_latest=is_latest,
                    attrs=dict(run_group.attrs),
                )
            )

    return sorted(options, key=lambda item: (not item.is_latest, item.run_scope, item.run_name))


def discover_swim_bout_run_options(
    zarr_path: Path | str,
    *,
    track_run_path: str,
    track_id: int,
) -> list[SwimBoutRunOption]:
    """Return swim-bout runs derived from the selected track-kinematics run."""

    root = open_zarr_root(Path(zarr_path), mode="r")
    track_run_name = _normalize_path(track_run_path).split("/")[-1]
    candidates = discover_swim_bout_candidates_resolved(
        root,
        track_run_name=track_run_name,
        track_id=track_id,
    )
    options: list[SwimBoutRunOption] = []
    for candidate in candidates:
        default_level = candidate.default_speed_level or ""
        if not default_level:
            continue

        n_bouts_by_level = {
            signal.speed_level: signal.n_bouts
            for signal in candidate.signals
        }
        method = candidate.detection_method
        layout = str(candidate.attrs.get("layout") or "hierarchical_v1")
        threshold = _safe_float(candidate.attrs.get("threshold_mm"))
        exponential_tau_s = _safe_float(candidate.attrs.get("exponential_tau_s"))
        exponential_source_level = candidate.attrs.get("exponential_source_level")
        exponential_source_level = (
            str(exponential_source_level) if exponential_source_level is not None else None
        )
        ordered_signals = [
            signal for signal in candidate.signals if signal.speed_level == default_level
        ] + [
            signal for signal in candidate.signals if signal.speed_level != default_level
        ]
        for signal in ordered_signals:
            level = signal.speed_level
            if level not in n_bouts_by_level:
                continue
            signal_source = exponential_source_level or signal.source_level
            speed_level = level.replace("speed_", "", 1)
            options.append(
                SwimBoutRunOption(
                    run_name=candidate.run_name,
                    label=_swim_option_label(
                        run_name=candidate.run_name,
                        speed_level=speed_level,
                        is_default_level=level == default_level,
                        detection_method=method,
                        threshold_mm=threshold,
                        exponential_tau_s=exponential_tau_s,
                        exponential_source_level=signal_source,
                        count=n_bouts_by_level[level],
                        is_latest=candidate.is_latest,
                    ),
                    layout=layout,
                    candidate_id=int(candidate.candidate_id),
                    signal_id=int(signal.signal_id),
                    signal_role=signal.role,
                    default_level=default_level,
                    speed_level=speed_level,
                    source_track_kinematics_run=candidate.source_track_kinematics_run,
                    track_id=candidate.track_id,
                    detection_method=method,
                    threshold_mm=threshold,
                    exponential_tau_s=exponential_tau_s,
                    exponential_source_level=signal_source,
                    n_bouts_by_level=n_bouts_by_level,
                    is_latest=candidate.is_latest,
                )
            )

    level_order = {"filtered": 0, "exponential": 1, "smoothed": 2, "raw": 3, "averaged": 4}
    return sorted(
        options,
        key=lambda item: (
            not item.is_latest,
            item.run_name,
            item.speed_level != item.default_level.replace("speed_", "", 1),
            level_order.get(item.speed_level, 99),
        ),
    )


def discover_bout_kinematics_run_options(
    zarr_path: Path | str,
    *,
    track_run_path: str,
    track_id: int,
    swim_bout_run: Optional[str],
    speed_level: Optional[str],
) -> list[BoutKinematicsRunOption]:
    """Return bout-kinematics runs derived from the selected bout candidate."""

    if swim_bout_run is None or not speed_level:
        return []

    root = open_zarr_root(Path(zarr_path), mode="r")
    parent = root.get("analysis/bout_kinematics_runs")
    if parent is None:
        return []

    track_run_name = _normalize_path(track_run_path).split("/")[-1]
    speed_level_key = _speed_level_key(speed_level)
    latest = parent.attrs.get("latest")
    options: list[BoutKinematicsRunOption] = []
    for run_name in _group_keys(parent):
        run_group = parent[run_name]
        source_refs = run_group.attrs.get("source_refs")
        source_refs = dict(source_refs) if isinstance(source_refs, Mapping) else {}
        source_track_run = (
            source_refs.get("source_track_kinematics_run")
            or run_group.attrs.get("source_track_kinematics_run")
        )
        if source_track_run is None or not _run_names_match(source_track_run, track_run_name):
            continue
        source_track_id_raw = (
            source_refs["source_track_id"]
            if "source_track_id" in source_refs
            else run_group.attrs.get("source_track_id")
        )
        source_track_id = _safe_int(source_track_id_raw)
        if source_track_id is None or source_track_id != int(track_id):
            continue
        source_swim = source_refs.get("source_swim_bout_run") or run_group.attrs.get("source_swim_bout_run")
        if str(source_swim) != str(swim_bout_run):
            continue
        source_speed = _speed_level_key(
            source_refs.get("source_swim_bout_speed_level")
            or run_group.attrs.get("source_swim_bout_speed_level")
        )
        if source_speed != speed_level_key:
            continue

        heading_levels = tuple(str(level) for level in run_group.attrs.get("heading_levels", []))
        if not heading_levels:
            heading_levels = tuple(level for level in ("heading_smoothed", "heading_raw") if level in run_group)
        default_heading = _heading_level_key(run_group.attrs.get("default_heading_level", "heading_smoothed"))
        parameters = run_group.attrs.get("parameters")
        parameters = dict(parameters) if isinstance(parameters, Mapping) else {}
        pre_post_mode = parameters.get("pre_post_mode")
        try:
            records_by_level, _level_attrs, _table_attrs = resolve_bout_kinematics_tables(run_group)
        except Exception:
            records_by_level = {}
        n_rows_by_level = {
            str(level): int(len(records))
            for level, records in records_by_level.items()
        }
        has_eye_gaze = n_rows_by_level.get("eye_gaze", 0) > 0
        is_latest = str(latest) == str(run_name)
        label_parts = [
            run_name,
            f"pre/post {pre_post_mode}" if pre_post_mode else "pre/post unknown",
            f"headings {', '.join(level.replace('heading_', '') for level in heading_levels)}",
            f"source {speed_level_key.replace('speed_', '')}",
        ]
        if n_rows_by_level.get("movement", 0) > 0:
            label_parts.append("movement")
        if has_eye_gaze:
            label_parts.append("eye gaze")
        if is_latest:
            label_parts.append("latest")
        options.append(
            BoutKinematicsRunOption(
                run_name=run_name,
                label=" | ".join(label_parts),
                default_heading_level=default_heading,
                heading_levels=heading_levels,
                pre_post_mode=str(pre_post_mode) if pre_post_mode is not None else None,
                source_track_kinematics_run=str(source_track_run),
                source_track_id=source_track_id,
                source_swim_bout_run=str(source_swim),
                source_swim_bout_speed_level=speed_level_key,
                n_rows_by_level=n_rows_by_level,
                is_latest=is_latest,
            )
        )

    return sorted(options, key=lambda item: (not item.is_latest, item.run_name))


def discover_bout_classification_run_options(
    zarr_path: Path | str,
    *,
    swim_bout_run: Optional[str],
    speed_level: Optional[str],
) -> list[BoutClassificationRunOption]:
    """Return classification runs derived from the selected swim-bout candidate."""

    if swim_bout_run is None or not speed_level:
        return []

    root = open_zarr_root(Path(zarr_path), mode="r")
    parent = root.get("analysis/bout_classification_runs")
    if parent is None:
        return []

    latest = parent.attrs.get("latest")
    options: list[BoutClassificationRunOption] = []
    speed_level_key = _speed_level_key(speed_level)
    for run_name in _group_keys(parent):
        run_group = parent[run_name]
        attrs = dict(run_group.attrs)
        if not _classification_matches_swim_bout(
            attrs,
            swim_bout_run=str(swim_bout_run),
            speed_level=speed_level_key,
        ):
            continue

        source_run, source_speed = _classification_source_from_attrs(attrs)
        source_count = _safe_int(attrs.get("source_bout_count"))
        if source_count is None:
            source_count = _columnar_row_count(run_group, "per_bout")
        classified_count = _safe_int(attrs.get("classified_bout_count"))
        if classified_count is None and "per_bout" in run_group:
            try:
                from fisheye.analysis.bout_classification_runs import load_bout_classification_table

                records = load_bout_classification_table(run_group)
                classified_count = int(np.count_nonzero(np.asarray(records["classified"], dtype=bool)))
            except Exception:
                classified_count = 0
        if classified_count is None:
            classified_count = 0
        skipped_count = max(0, int(source_count) - int(classified_count))
        classifier_family = str(attrs.get("classifier_family", "unknown"))
        classifier_name = str(attrs.get("classifier_name", "unknown"))
        classifier_version = attrs.get("classifier_version")
        classifier_version_str = str(classifier_version) if classifier_version is not None else None
        is_latest = str(latest) == str(run_name)
        options.append(
            BoutClassificationRunOption(
                run_name=run_name,
                label=_bout_classification_option_label(
                    run_name=run_name,
                    classifier_family=classifier_family,
                    classifier_name=classifier_name,
                    classifier_version=classifier_version_str,
                    source_bout_count=int(source_count),
                    classified_bout_count=int(classified_count),
                    skipped_bout_count=int(skipped_count),
                    speed_level=source_speed,
                    is_latest=is_latest,
                ),
                classifier_family=classifier_family,
                classifier_name=classifier_name,
                classifier_version=classifier_version_str,
                source_swim_bout_run=source_run,
                source_swim_bout_speed_level=source_speed,
                source_bout_count=int(source_count),
                classified_bout_count=int(classified_count),
                skipped_bout_count=int(skipped_count),
                is_latest=is_latest,
                attrs=attrs,
            )
        )

    return sorted(options, key=lambda item: (not item.is_latest, item.run_name))


def _run_names_match(source_run: object, spec_run: object) -> bool:
    source = _normalize_path(str(source_run))
    spec = _normalize_path(str(spec_run))
    return source == spec or source.endswith(f"/{spec}") or spec.endswith(f"/{source}")


def _load_global_swim_bout_payload(
    root: zarr.Group,
    *,
    spec: Mapping[str, Any],
    requested_run: Optional[str],
    speed_level: Optional[str],
) -> _SwimBoutPayload:
    run_name = requested_run
    if isinstance(run_name, str) and run_name.lower() == "none":
        return _empty_swim_bout_payload()

    try:
        swim_payload = load_swim_bout_tables(
            root,
            run_name=run_name or "latest",
            speed_level=speed_level,
        )
    except SwimBoutIOError:
        return _empty_swim_bout_payload()

    candidate = swim_payload.candidate
    source_run = candidate.source_track_kinematics_run
    if source_run is not None and not _run_names_match(source_run, spec.get("run_name")):
        return _empty_swim_bout_payload()
    source_track_id = candidate.track_id
    spec_track_id = spec.get("track_id")
    if source_track_id is not None and spec_track_id is not None:
        try:
            if int(source_track_id) != int(spec_track_id):
                return _empty_swim_bout_payload()
        except (TypeError, ValueError):
            return _empty_swim_bout_payload()

    records = swim_payload.bouts
    peak_events = swim_payload.peak_events
    fps_value = swim_payload.run_attrs.get("fps")
    fps = None
    try:
        fps = float(fps_value) if fps_value is not None else None
    except (TypeError, ValueError):
        fps = None
    records = _augment_bouts_with_peak_events(records, peak_events, fps=fps)
    spans = _spans_from_structured_bouts(records, fps)
    intervals = swim_payload.inter_bout_intervals
    series = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in swim_payload.series.items()
        if np.asarray(values).ndim == 1
    }
    method = candidate.detection_method
    label = f"{candidate.run_name} ({swim_payload.signal.speed_level}) ({method})"
    return _SwimBoutPayload(
        spans=spans,
        records=records,
        inter_bout_intervals=intervals,
        series=series,
        label=label,
        source=swim_payload.level_path,
    )


def _load_swim_bout_payload(
    root: zarr.Group,
    spec: Mapping[str, Any],
    *,
    requested_run: Optional[str],
    speed_level: Optional[str],
) -> _SwimBoutPayload:
    overlays = spec.get("overlays")
    swim_overlay = overlays.get("swim_bouts") if isinstance(overlays, Mapping) else {}
    if not isinstance(swim_overlay, Mapping):
        swim_overlay = {}

    explicit_request = requested_run is not None
    run_selector = requested_run if explicit_request else swim_overlay.get("requested_run")
    if isinstance(run_selector, str) and run_selector.lower() == "none":
        return _empty_swim_bout_payload()

    overlay_enabled = bool(swim_overlay.get("enabled", False))
    if not explicit_request and not overlay_enabled:
        return _empty_swim_bout_payload()

    level_selector = speed_level if speed_level is not None else swim_overlay.get("speed_level")
    if not isinstance(run_selector, str):
        run_selector = None
    if not isinstance(level_selector, str):
        level_selector = None

    return _load_global_swim_bout_payload(
        root,
        spec=spec,
        requested_run=run_selector,
        speed_level=level_selector,
    )


def _collect_series(
    root: zarr.Group,
    source_paths: Mapping[str, str],
    *,
    run_path: str,
    track_id: int,
) -> dict[str, np.ndarray]:
    preferred_keys = (
        "speed_raw_mm",
        "speed_filtered_mm",
        "speed_smoothed_mm",
        "speed_averaged_mm",
        "speed_raw_px",
        "speed_filtered_px",
        "speed_smoothed_px",
        "speed_averaged_px",
        "acceleration_mm",
        "acceleration_px",
        "smoothed_acceleration_mm",
        "smoothed_acceleration_px",
        "speed_raw_acceleration_mm",
        "speed_raw_acceleration_px",
        "speed_raw_smoothed_acceleration_mm",
        "speed_raw_smoothed_acceleration_px",
        "speed_filtered_acceleration_mm",
        "speed_filtered_acceleration_px",
        "speed_filtered_smoothed_acceleration_mm",
        "speed_filtered_smoothed_acceleration_px",
        "speed_smoothed_acceleration_mm",
        "speed_smoothed_acceleration_px",
        "speed_smoothed_smoothed_acceleration_mm",
        "speed_smoothed_smoothed_acceleration_px",
        "speed_averaged_acceleration_mm",
        "speed_averaged_acceleration_px",
        "speed_averaged_smoothed_acceleration_mm",
        "speed_averaged_smoothed_acceleration_px",
        "speed_raw_frame_path_distance_mm",
        "speed_raw_frame_path_distance_px",
        "speed_filtered_frame_path_distance_mm",
        "speed_filtered_frame_path_distance_px",
        "speed_smoothed_frame_path_distance_mm",
        "speed_smoothed_frame_path_distance_px",
        "delta_heading_degrees",
        "angular_velocity_deg_s",
        "angular_velocity_raw_deg_s",
        "angular_speed_raw_deg_s",
        "delta_heading_smoothed_degrees",
        "angular_velocity_smoothed_deg_s",
        "angular_speed_smoothed_deg_s",
        "smoothed_heading_degrees",
        "cumulative_path_distance_mm",
        "cumulative_path_distance_px",
        "distance_to_target_mm",
        "distance_to_target_px",
        "distance_to_target_smoothed_mm",
        "distance_to_target_smoothed_px",
        "distance_to_target_interpolated_mm",
        "distance_to_target_interpolated_px",
    )
    series: dict[str, np.ndarray] = {}
    for key in preferred_keys:
        values = _load_array(root, source_paths, key)
        if values is None:
            values = _load_track_array(
                root,
                source_paths,
                run_path=run_path,
                track_id=track_id,
                key=key,
            )
        if values is None:
            continue
        if values.ndim == 1:
            series[key] = values.astype(np.float64, copy=False)
    return series


def _collect_verified_track_series(
    tables: TrackKinematicsTrackTables,
) -> dict[str, np.ndarray]:
    """Build UI series only from arrays copied through the strict reader."""

    series: dict[str, np.ndarray] = {}
    prefix_by_level = {
        "raw": "speed_raw",
        "filtered": "speed_filtered",
        "smoothed": "speed_smoothed",
        "averaged": "speed_averaged",
    }
    for level, prefix in prefix_by_level.items():
        for unit, speeds, accelerations, smoothed_accelerations, distances in (
            (
                "px",
                tables.speed_px_by_level,
                tables.acceleration_px_by_level,
                tables.smoothed_acceleration_px_by_level,
                tables.frame_path_distance_px_by_level,
            ),
            (
                "mm",
                tables.speed_mm_by_level,
                tables.acceleration_mm_by_level,
                tables.smoothed_acceleration_mm_by_level,
                tables.frame_path_distance_mm_by_level,
            ),
        ):
            if level in speeds:
                series[f"{prefix}_{unit}"] = np.asarray(
                    speeds[level], dtype=np.float64
                )
            if level in accelerations:
                key = f"acceleration_{unit}"
                series[f"{prefix}_{key}"] = np.asarray(
                    accelerations[level], dtype=np.float64
                )
                if level == "smoothed":
                    series[key] = series[f"{prefix}_{key}"]
            if level in smoothed_accelerations:
                key = f"smoothed_acceleration_{unit}"
                series[f"{prefix}_{key}"] = np.asarray(
                    smoothed_accelerations[level], dtype=np.float64
                )
                if level == "smoothed":
                    series[key] = series[f"{prefix}_{key}"]
            if level in distances:
                series[f"{prefix}_frame_path_distance_{unit}"] = np.asarray(
                    distances[level], dtype=np.float64
                )

    optional = {
        "delta_heading_degrees": tables.delta_heading_degrees,
        "angular_velocity_deg_s": tables.angular_velocity_deg_s,
        "angular_speed_raw_deg_s": tables.angular_speed_raw_deg_s,
        "delta_heading_smoothed_degrees": (
            tables.delta_heading_smoothed_degrees
        ),
        "angular_velocity_smoothed_deg_s": (
            tables.angular_velocity_smoothed_deg_s
        ),
        "angular_speed_smoothed_deg_s": tables.angular_speed_smoothed_deg_s,
        "smoothed_heading_degrees": tables.smoothed_heading_degrees,
        "cumulative_path_distance_mm": tables.cumulative_path_distance_mm,
        "cumulative_path_distance_px": tables.cumulative_path_distance_px,
    }
    for name, values in optional.items():
        if values is not None:
            series[name] = np.asarray(values, dtype=np.float64)
    return series


def _verified_validity_spans(
    tables: TrackKinematicsTrackTables,
    time_seconds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Optional[str]]:
    rows: list[tuple[float, float, str]] = []
    n_time = int(time_seconds.shape[0])
    transition_valid = tables.transition_valid
    transition_reason = tables.transition_reason_code
    sample_valid = tables.sample_valid
    sample_reason = tables.sample_reason_code
    transition_map = tables.track_attrs.get("transition_reason_codes", {})
    sample_map = tables.track_attrs.get("sample_reason_codes", {})

    if transition_valid is None or transition_reason is None:
        raise ValueError("Verified track tables omit transition validity semantics.")
    if sample_valid is None or sample_reason is None:
        raise ValueError("Verified track tables omit sample validity semantics.")
    if (
        np.asarray(transition_valid).shape != (n_time,)
        or np.asarray(transition_reason).shape != (n_time,)
        or np.asarray(sample_valid).shape != (n_time,)
        or np.asarray(sample_reason).shape != (n_time,)
    ):
        raise ValueError("Verified track validity arrays disagree with the time axis.")

    for index in range(1, n_time):
        if bool(transition_valid[index]):
            continue
        start = float(time_seconds[index - 1])
        stop = float(time_seconds[index])
        if not (np.isfinite(start) and np.isfinite(stop)):
            continue
        if stop < start:
            start, stop = stop, start
        if stop <= start:
            continue
        reason = _reason_name(
            transition_map,
            transition_reason[index],
            prefix="transition",
        )
        if reason != "first_sample":
            rows.append((start, stop, f"transition:{reason}"))

    for index, valid in enumerate(np.asarray(sample_valid, dtype=bool)):
        if bool(valid):
            continue
        interval = _row_interval(time_seconds, index)
        if interval is None:
            continue
        reason = _reason_name(
            sample_map,
            sample_reason[index],
            prefix="sample",
        )
        rows.append((interval[0], interval[1], f"sample:{reason}"))

    if not rows:
        return _empty_spans(), _empty_labels(), None
    return (
        np.asarray([(start, stop) for start, stop, _ in rows], dtype=np.float64),
        np.asarray([label for _, _, label in rows], dtype=object),
        "track_validity",
    )


def load_track_kinematics_interactive_data(
    zarr_path: Path | str,
    *,
    run_path: str,
    artifact_name: str = DEFAULT_INTERACTIVE_ARTIFACT,
    swim_bout_run: Optional[str] = None,
    speed_level: Optional[str] = None,
) -> TrackKinematicsInteractiveData:
    """Load a persisted track-kinematics interactive spec and source arrays."""

    archive = Path(zarr_path)
    root = open_zarr_root(archive, mode="r")
    artifact = _resolve_interactive_artifact(root, run_path=run_path, artifact_name=artifact_name)
    spec = _json_from_uint8_array(artifact["spec_json"])
    schema_id = spec.get("schema_id")
    if schema_id != TRACK_KINEMATICS_SPEC_SCHEMA_ID:
        raise ValueError(
            f"Unsupported interactive spec schema: {schema_id!r}; "
            f"expected {TRACK_KINEMATICS_SPEC_SCHEMA_ID!r}"
        )

    source_paths = _as_str_mapping(spec.get("source_paths"))
    tables = _load_verified_track_for_spec(
        root,
        run_path=run_path,
        artifact=artifact,
        spec=spec,
    )
    track_id = int(tables.track_id)
    if tables.time_seconds is None:
        raise ValueError("Verified track-motion publication omits time_seconds.")
    time_seconds = np.asarray(tables.time_seconds, dtype=np.float64)
    if tables.source_acquisition_frame_index is None:
        raise ValueError(
            "Verified track-motion publication omits acquisition-frame identity."
        )
    frame_indices = np.asarray(
        tables.source_acquisition_frame_index,
        dtype=np.int64,
    )

    from fisheye.analysis.plot_track_kinematics import pick_units

    position_unit, position_x, position_y = pick_units(tables)
    positions = np.column_stack([position_x, position_y]).astype(
        np.float64,
        copy=False,
    )
    if tables.sample_valid is None:
        raise ValueError("Verified track-motion publication omits sample validity.")
    sample_valid = np.asarray(tables.sample_valid, dtype=bool)
    if sample_valid.shape != (positions.shape[0],):
        raise ValueError("Track positions and sample validity disagree.")
    positions = np.array(positions, copy=True)
    positions[~sample_valid] = np.nan

    swim_bout_payload = _load_swim_bout_payload(
        root,
        spec,
        requested_run=swim_bout_run,
        speed_level=speed_level,
    )
    validity_spans, validity_labels, validity_source = _verified_validity_spans(
        tables,
        time_seconds,
    )
    series = _collect_verified_track_series(tables)
    series.update(swim_bout_payload.series)

    parts = _normalize_path(run_path).split("/")
    current = load_track_kinematics_track(
        root,
        run_name=parts[3],
        scope=parts[2],
        track_id=track_id,
        required_speed_levels=("raw", "smoothed"),
    )
    if current.authority_record() != tables.authority_record():
        raise ValueError(
            "Track-motion authority changed while loading interactive data."
        )

    return TrackKinematicsInteractiveData(
        zarr_path=archive,
        run_path=_normalize_path(run_path),
        artifact_name=artifact_name,
        spec=spec,
        attrs=dict(artifact.attrs),
        source_paths=source_paths,
        time_seconds=time_seconds,
        frame_indices=frame_indices,
        series=series,
        positions=positions,
        position_unit=position_unit,
        swim_bouts=swim_bout_payload.spans,
        swim_bout_records=swim_bout_payload.records,
        inter_bout_intervals=swim_bout_payload.inter_bout_intervals,
        swim_bout_label=swim_bout_payload.label,
        swim_bout_source=swim_bout_payload.source,
        validity_spans=validity_spans,
        validity_labels=validity_labels,
        validity_source=validity_source,
    )


def to_timeseries_dataframe(data: TrackKinematicsInteractiveData) -> pd.DataFrame:
    frame: dict[str, Any] = {"time_s": data.time_seconds}
    if data.frame_indices is not None and data.frame_indices.shape[0] == data.time_seconds.shape[0]:
        frame["frame_index"] = data.frame_indices
    for name, values in data.series.items():
        if values.shape[0] == data.time_seconds.shape[0]:
            frame[name] = values
    return pd.DataFrame(frame)


def to_position_dataframe(data: TrackKinematicsInteractiveData) -> pd.DataFrame:
    if data.positions is None or data.positions.ndim != 2 or data.positions.shape[1] < 2:
        return pd.DataFrame(columns=["time_s", "frame_index", "x", "y", "unit"])
    count = min(data.positions.shape[0], data.time_seconds.shape[0])
    frame: dict[str, Any] = {
        "time_s": data.time_seconds[:count],
    }
    if data.frame_indices is not None and data.frame_indices.shape[0] >= count:
        frame["frame_index"] = data.frame_indices[:count]
    frame.update(
        {
            "x": data.positions[:count, 0],
            "y": data.positions[:count, 1],
            "unit": data.position_unit,
        }
    )
    return pd.DataFrame(frame)


def to_swim_bout_dataframe(data: TrackKinematicsInteractiveData) -> pd.DataFrame:
    if data.swim_bout_records.size and data.swim_bout_records.dtype.names is not None:
        frame = _structured_to_dataframe(data.swim_bout_records)
        if "start_time_s" in frame and "start_s" not in frame:
            frame["start_s"] = frame["start_time_s"]
        if "end_time_s" in frame and "end_s" not in frame:
            frame["end_s"] = frame["end_time_s"]
        if "duration_s" not in frame and {"start_s", "end_s"}.issubset(frame.columns):
            frame["duration_s"] = frame["end_s"] - frame["start_s"]
        return frame
    if data.swim_bouts.size == 0:
        return pd.DataFrame(columns=["start_s", "end_s", "duration_s"])
    starts = data.swim_bouts[:, 0]
    ends = data.swim_bouts[:, 1]
    return pd.DataFrame(
        {
            "start_s": starts,
            "end_s": ends,
            "duration_s": ends - starts,
        }
    )


def to_inter_bout_interval_dataframe(data: TrackKinematicsInteractiveData) -> pd.DataFrame:
    return _structured_to_dataframe(data.inter_bout_intervals)


def load_bout_kinematics_records(
    zarr_path: Path | str,
    *,
    run_name: str,
    heading_level: Optional[str] = None,
) -> tuple[dict[str, np.ndarray], Mapping[str, Any]]:
    """Load per-bout kinematics records from one analysis run."""

    root = open_zarr_root(Path(zarr_path), mode="r")
    path = _join_path("analysis/bout_kinematics_runs", run_name)
    try:
        run_group = root[path]
    except Exception as exc:
        raise ValueError(f"Bout kinematics run not found: {path}") from exc

    records_by_level, _level_attrs, _table_attrs = resolve_bout_kinematics_tables(
        run_group,
        heading_level=heading_level,
    )
    return records_by_level, dict(run_group.attrs)


def bout_kinematics_records_to_dataframe(records_by_level: Mapping[str, np.ndarray]) -> pd.DataFrame:
    frames = []
    for level, records in records_by_level.items():
        frame = _structured_to_dataframe(records)
        if frame.empty:
            continue
        frame.insert(0, "heading_level", level)
        frame.insert(1, "analysis_level", level)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_bout_classification_records(
    zarr_path: Path | str,
    *,
    run_name: str,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    """Load per-bout classification records from one analysis run."""

    from fisheye.analysis.bout_classification_runs import (
        load_bout_classification_table,
        resolve_bout_classification_run,
    )

    root = open_zarr_root(Path(zarr_path), mode="r")
    run_group, _resolved_name, _run_path = resolve_bout_classification_run(root, run_name)
    return load_bout_classification_table(run_group), dict(run_group.attrs)


def bout_classification_records_to_dataframe(records: np.ndarray) -> pd.DataFrame:
    """Convert a bout-classification structured table to a plotting dataframe."""

    frame = _structured_to_dataframe(records)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "source_bout_id",
                "category_label",
                "failure_reason",
                "classified",
                "valid",
                "probability",
            ]
        )
    if "category_label_bytes" in frame:
        frame["category_label"] = [
            _decode_structured_value(value).rstrip("\x00")
            for value in frame["category_label_bytes"]
        ]
    if "failure_reason_bytes" in frame:
        frame["failure_reason"] = [
            _decode_structured_value(value).rstrip("\x00")
            for value in frame["failure_reason_bytes"]
        ]
    return frame


def to_validity_span_dataframe(data: TrackKinematicsInteractiveData) -> pd.DataFrame:
    if data.validity_spans.size == 0:
        return pd.DataFrame(columns=["start_s", "end_s", "duration_s", "reason"])
    starts = data.validity_spans[:, 0]
    ends = data.validity_spans[:, 1]
    return pd.DataFrame(
        {
            "start_s": starts,
            "end_s": ends,
            "duration_s": ends - starts,
            "reason": data.validity_labels,
        }
    )
