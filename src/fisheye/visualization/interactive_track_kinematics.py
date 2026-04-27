"""Load track-kinematics interactive visualization specs from Zarr."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd
import zarr

from fisheye.utils.zarr_io import open_zarr_root


TRACK_KINEMATICS_SPEC_SCHEMA_ID = "palette.plot_spec.track_kinematics_summary.v1"
DEFAULT_INTERACTIVE_ARTIFACT = "track_kinematics_summary_track_0_interactive"


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
    default_level: str
    speed_level: str
    source_track_kinematics_run: Optional[str]
    track_id: Optional[int]
    detection_method: str
    threshold_mm: Optional[float]
    n_bouts_by_level: Mapping[str, int]
    is_latest: bool


@dataclass(frozen=True)
class BoutKinematicsRunOption:
    run_name: str
    label: str
    default_heading_level: str
    heading_levels: tuple[str, ...]
    source_track_kinematics_run: Optional[str]
    source_track_id: Optional[int]
    source_swim_bout_run: Optional[str]
    source_swim_bout_speed_level: Optional[str]
    is_latest: bool


@dataclass(frozen=True)
class _SwimBoutPayload:
    spans: np.ndarray
    records: np.ndarray
    inter_bout_intervals: np.ndarray
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
    payload = np.asarray(array[:], dtype=np.uint8).tobytes().decode("utf-8")
    parsed = json.loads(payload)
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


def _resolve_interactive_artifact(
    root: zarr.Group,
    *,
    run_path: str,
    artifact_name: str,
) -> zarr.Group:
    artifact_path = _join_path(run_path, "visualizations", artifact_name)
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
    direct_path = _join_path(run_path, "tracks", f"id_{int(track_id)}", key)
    try:
        return np.asarray(root[direct_path][:])
    except Exception:
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


def _load_bout_spans_from_group(group: zarr.Group, *, fps: Optional[float] = None) -> np.ndarray:
    if "bouts" in group:
        from fisheye.analysis.chaser_state_interpolator import load_structured_dataset

        bouts, _attrs = load_structured_dataset(group, "bouts")
        return _spans_from_structured_bouts(bouts, fps)
    if "start_time_s" in group and "end_time_s" in group:
        return _spans_from_start_stop(
            np.asarray(group["start_time_s"][:], dtype=np.float64),
            np.asarray(group["end_time_s"][:], dtype=np.float64),
        )
    if "start_frame" in group and "end_frame" in group and fps:
        return _spans_from_start_stop(
            np.asarray(group["start_frame"][:], dtype=np.float64) / float(fps),
            np.asarray(group["end_frame"][:], dtype=np.float64) / float(fps),
        )
    return _empty_spans()


def _load_structured_or_empty(group: zarr.Group, name: str) -> np.ndarray:
    from fisheye.analysis.chaser_state_interpolator import load_structured_dataset

    try:
        records, _attrs = load_structured_dataset(group, name)
    except Exception:
        return _empty_records()
    return np.asarray(records)


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


def _bout_count(level_group: zarr.Group) -> int:
    if "bouts" not in level_group:
        return 0
    node = level_group["bouts"]
    if isinstance(node, zarr.Array):
        return int(node.shape[0])
    if isinstance(node, zarr.Group):
        fields = list(node.attrs.get("field_names", []))
        for field in fields:
            if field in node:
                return int(node[field].shape[0])
    return 0


def _swim_option_label(
    *,
    run_name: str,
    speed_level: str,
    is_default_level: bool,
    detection_method: str,
    threshold_mm: Optional[float],
    count: int,
    is_latest: bool,
) -> str:
    pieces = [run_name, speed_level, f"{count} bouts"]
    if threshold_mm is not None:
        pieces.append(f"threshold {threshold_mm:g} mm/s")
    if detection_method:
        pieces.append(f"method {detection_method}")
    if is_default_level:
        pieces.append("default")
    if is_latest:
        pieces.append("latest")
    return " | ".join(pieces)


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
                track_id = 0
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
    parent = root.get("analysis/swim_bout_runs")
    if parent is None:
        return []

    track_run_name = _normalize_path(track_run_path).split("/")[-1]
    latest = parent.attrs.get("latest")
    speed_levels = ("speed_filtered", "speed_smoothed", "speed_raw", "speed_averaged")
    options: list[SwimBoutRunOption] = []
    for run_name in _group_keys(parent):
        run_group = parent[run_name]
        source_run = run_group.attrs.get("source_track_kinematics_run")
        if source_run is None or not _run_names_match(source_run, track_run_name):
            continue
        source_track_id = _safe_int(run_group.attrs.get("track_id"))
        if source_track_id is None or source_track_id != int(track_id):
            continue

        default_level = _speed_level_key(str(run_group.attrs.get("default_level", "speed_smoothed")))
        if default_level not in run_group:
            default_level = "speed_smoothed" if "speed_smoothed" in run_group else ""
        if not default_level:
            continue

        n_bouts_by_level = {
            level: _bout_count(run_group[level])
            for level in speed_levels
            if level in run_group
        }
        method = str(run_group.attrs.get("detection_method", "unknown"))
        threshold = _safe_float(run_group.attrs.get("threshold_mm"))
        is_latest = str(latest) == str(run_name)
        ordered_levels = [default_level] + [
            level for level in speed_levels if level != default_level
        ]
        for level in ordered_levels:
            if level not in n_bouts_by_level:
                continue
            speed_level = level.replace("speed_", "", 1)
            options.append(
                SwimBoutRunOption(
                    run_name=run_name,
                    label=_swim_option_label(
                        run_name=run_name,
                        speed_level=speed_level,
                        is_default_level=level == default_level,
                        detection_method=method,
                        threshold_mm=threshold,
                        count=n_bouts_by_level[level],
                        is_latest=is_latest,
                    ),
                    default_level=default_level,
                    speed_level=speed_level,
                    source_track_kinematics_run=str(source_run) if source_run is not None else None,
                    track_id=source_track_id,
                    detection_method=method,
                    threshold_mm=threshold,
                    n_bouts_by_level=n_bouts_by_level,
                    is_latest=is_latest,
                )
            )

    level_order = {"filtered": 0, "smoothed": 1, "raw": 2, "averaged": 3}
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
        source_track_id = _safe_int(source_refs.get("source_track_id") or run_group.attrs.get("source_track_id"))
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
        is_latest = str(latest) == str(run_name)
        label_parts = [
            run_name,
            f"headings {', '.join(level.replace('heading_', '') for level in heading_levels)}",
            f"source {speed_level_key.replace('speed_', '')}",
        ]
        if is_latest:
            label_parts.append("latest")
        options.append(
            BoutKinematicsRunOption(
                run_name=run_name,
                label=" | ".join(label_parts),
                default_heading_level=default_heading,
                heading_levels=heading_levels,
                source_track_kinematics_run=str(source_track_run),
                source_track_id=source_track_id,
                source_swim_bout_run=str(source_swim),
                source_swim_bout_speed_level=speed_level_key,
                is_latest=is_latest,
            )
        )

    return sorted(options, key=lambda item: (not item.is_latest, item.run_name))


def _run_names_match(source_run: object, spec_run: object) -> bool:
    source = _normalize_path(str(source_run))
    spec = _normalize_path(str(spec_run))
    return source == spec or source.endswith(f"/{spec}") or spec.endswith(f"/{source}")


def _swim_bout_run_matches_track(run_group: zarr.Group, spec: Mapping[str, Any]) -> bool:
    source_run = run_group.attrs.get("source_track_kinematics_run")
    if source_run is not None and not _run_names_match(source_run, spec.get("run_name")):
        return False

    source_track_id = run_group.attrs.get("track_id")
    spec_track_id = spec.get("track_id")
    if source_track_id is not None and spec_track_id is not None:
        try:
            if int(source_track_id) != int(spec_track_id):
                return False
        except (TypeError, ValueError):
            return False

    return True


def _load_global_swim_bout_payload(
    root: zarr.Group,
    *,
    spec: Mapping[str, Any],
    requested_run: Optional[str],
    speed_level: Optional[str],
) -> _SwimBoutPayload:
    swim_parent = root.get("analysis/swim_bout_runs")
    if swim_parent is None:
        return _empty_swim_bout_payload()

    run_name = requested_run
    if not run_name or run_name.lower() == "latest":
        latest = swim_parent.attrs.get("latest")
        if isinstance(latest, str) and latest:
            run_name = latest
    if not run_name or run_name.lower() == "none" or run_name not in swim_parent:
        return _empty_swim_bout_payload()

    run_group = swim_parent[run_name]
    if not _swim_bout_run_matches_track(run_group, spec):
        return _empty_swim_bout_payload()

    speed_levels = ("speed_raw", "speed_filtered", "speed_smoothed", "speed_averaged")
    is_hierarchical = all(level in run_group for level in speed_levels)
    method = run_group.attrs.get("detection_method", "unknown")

    if is_hierarchical:
        default_level = str(run_group.attrs.get("default_level", "speed_smoothed"))
        level_key = _speed_level_key(speed_level, default_level)
        if level_key not in run_group and default_level in run_group:
            level_key = default_level
        if level_key not in run_group:
            return _empty_swim_bout_payload()
        level_group = run_group[level_key]
        records = _load_structured_or_empty(level_group, "bouts")
        spans = _load_bout_spans_from_group(
            level_group,
            fps=float(run_group.attrs["fps"]) if "fps" in run_group.attrs else None,
        )
        intervals = _load_structured_or_empty(level_group, "inter_bout_intervals")
        label = f"{run_name} ({level_key}) ({method})"
        return _SwimBoutPayload(
            spans=spans,
            records=records,
            inter_bout_intervals=intervals,
            label=label,
            source="analysis_swim_bout_runs",
        )

    records = _load_structured_or_empty(run_group, "bouts")
    spans = _load_bout_spans_from_group(
        run_group,
        fps=float(run_group.attrs["fps"]) if "fps" in run_group.attrs else None,
    )
    intervals = _load_structured_or_empty(run_group, "inter_bout_intervals")
    label = f"{run_name} ({method})"
    return _SwimBoutPayload(
        spans=spans,
        records=records,
        inter_bout_intervals=intervals,
        label=label,
        source="analysis_swim_bout_runs",
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


def _collect_series(root: zarr.Group, source_paths: Mapping[str, str]) -> dict[str, np.ndarray]:
    preferred_keys = (
        "speed_raw_mm",
        "speed_filtered_mm",
        "speed_smoothed_mm",
        "speed_averaged_mm",
        "speed_raw_px",
        "speed_filtered_px",
        "speed_smoothed_px",
        "speed_averaged_px",
        "smoothed_acceleration_mm",
        "smoothed_acceleration_px",
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
            continue
        if values.ndim == 1:
            series[key] = values.astype(np.float64, copy=False)
    return series


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
    track_id = _safe_int(spec.get("track_id"))
    if track_id is None:
        track_id = 0
    time_seconds = _load_array(root, source_paths, "time_seconds")
    if time_seconds is None:
        raise ValueError("Interactive spec does not resolve a time_seconds source array")
    time_seconds = time_seconds.astype(np.float64, copy=False)

    frame_indices = _load_array(root, source_paths, "frame_indices")
    if frame_indices is not None:
        frame_indices = frame_indices.astype(np.int64, copy=False)

    positions = _load_array(root, source_paths, "positions_mm")
    position_unit = "mm"
    if positions is None:
        positions = _load_array(root, source_paths, "positions_px")
        position_unit = "px"
    if positions is not None:
        positions = positions.astype(np.float64, copy=False)

    swim_bout_payload = _load_swim_bout_payload(
        root,
        spec,
        requested_run=swim_bout_run,
        speed_level=speed_level,
    )
    validity_spans, validity_labels, validity_source = _load_validity_spans(
        root,
        run_path=_normalize_path(run_path),
        track_id=track_id,
        source_paths=source_paths,
        time_seconds=time_seconds,
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
        series=_collect_series(root, source_paths),
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
        return pd.DataFrame(columns=["time_s", "x", "y", "unit"])
    count = min(data.positions.shape[0], data.time_seconds.shape[0])
    return pd.DataFrame(
        {
            "time_s": data.time_seconds[:count],
            "x": data.positions[:count, 0],
            "y": data.positions[:count, 1],
            "unit": data.position_unit,
        }
    )


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

    if heading_level is not None:
        levels = (_heading_level_key(heading_level),)
    else:
        levels = tuple(str(level) for level in run_group.attrs.get("heading_levels", []))
        if not levels:
            levels = tuple(level for level in ("heading_smoothed", "heading_raw") if level in run_group)

    records_by_level: dict[str, np.ndarray] = {}
    for level in levels:
        if level not in run_group:
            continue
        metrics = _load_structured_or_empty(run_group[level], "per_bout_metrics")
        records_by_level[level] = metrics
    return records_by_level, dict(run_group.attrs)


def bout_kinematics_records_to_dataframe(records_by_level: Mapping[str, np.ndarray]) -> pd.DataFrame:
    frames = []
    for level, records in records_by_level.items():
        frame = _structured_to_dataframe(records)
        if frame.empty:
            continue
        frame.insert(0, "heading_level", level)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


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
