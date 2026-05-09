"""Logical readers for Palette swim-bout runs.

This module provides a compatibility layer over the current hierarchical
``analysis/swim_bout_runs/<run>/<speed_level>`` layout. The public objects are
shaped to match the planned compact v2 schema so downstream code can stop
depending on physical path names before the writer changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import zarr

from fisheye.analysis.chaser_state_interpolator import load_structured_dataset


SPEED_LEVEL_ORDER: tuple[str, ...] = (
    "speed_filtered",
    "speed_exponential",
    "speed_smoothed",
    "speed_raw",
    "speed_averaged",
)

SPEED_LEVEL_ALIASES: Mapping[str, str] = {
    "raw": "speed_raw",
    "speed_raw": "speed_raw",
    "filtered": "speed_filtered",
    "speed_filtered": "speed_filtered",
    "smoothed": "speed_smoothed",
    "speed_smoothed": "speed_smoothed",
    "averaged": "speed_averaged",
    "average": "speed_averaged",
    "speed_averaged": "speed_averaged",
    "exp": "speed_exponential",
    "exponential": "speed_exponential",
    "speed_exp": "speed_exponential",
    "speed_exponential": "speed_exponential",
}


class SwimBoutIOError(ValueError):
    """Raised when a swim-bout run cannot be resolved or loaded."""


@dataclass(frozen=True)
class SwimBoutSignalVariant:
    """One selectable speed/detector signal variant inside a swim-bout run."""

    run_name: str
    signal_id: int
    speed_level: str
    signal_name: str
    role: str
    source_level: Optional[str]
    is_default: bool
    n_bouts: int
    attrs: Mapping[str, Any]


@dataclass(frozen=True)
class SwimBoutCandidate:
    """One logical swim-bout candidate.

    In the current v1 layout this maps to one run group. In compact v2 this can
    map to a row in ``indexes/candidates``.
    """

    run_name: str
    candidate_id: int
    candidate_name: str
    run_path: str
    is_latest: bool
    source_track_kinematics_run: Optional[str]
    track_id: Optional[int]
    detection_method: str
    default_signal_id: Optional[int]
    default_speed_level: Optional[str]
    signals: tuple[SwimBoutSignalVariant, ...]
    attrs: Mapping[str, Any]


@dataclass(frozen=True)
class SwimBoutTables:
    """Normalized table payload for one candidate/signal selection."""

    run_name: str
    run_path: str
    level_path: str
    candidate: SwimBoutCandidate
    signal: SwimBoutSignalVariant
    bouts: np.ndarray
    peak_events: np.ndarray
    inter_bout_intervals: np.ndarray
    global_metrics: np.ndarray
    bout_points: np.ndarray
    series: Mapping[str, np.ndarray]
    run_attrs: Mapping[str, Any]
    signal_attrs: Mapping[str, Any]


def normalize_speed_level(value: object, default: str | None = None) -> str:
    """Return a canonical ``speed_*`` level name."""

    if value is None:
        if default is None:
            raise SwimBoutIOError("No speed level specified.")
        return normalize_speed_level(default)
    text = str(value).strip()
    if not text:
        if default is None:
            raise SwimBoutIOError("Empty speed level specified.")
        return normalize_speed_level(default)
    key = text.lower()
    return SPEED_LEVEL_ALIASES.get(key, f"speed_{key}" if not key.startswith("speed_") else key)


def structured_records_to_dicts(records: np.ndarray) -> list[dict[str, Any]]:
    """Convert a structured array to row dictionaries with JSON/Parquet-safe scalars."""

    if records.dtype.names is None:
        return []
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.append({name: _scalar_value(record[name]) for name in records.dtype.names})
    return rows


def discover_swim_bout_candidates(
    root: zarr.Group,
    *,
    track_run_name: object | None = None,
    track_id: int | None = None,
) -> list[SwimBoutCandidate]:
    """Discover logical swim-bout candidates under ``analysis/swim_bout_runs``.

    This first implementation supports the current v1 hierarchical layout. The
    returned dataclasses are intentionally shaped so compact v2 can be added
    without changing callers.
    """

    parent = _get_child(root, "analysis/swim_bout_runs")
    if parent is None:
        return []

    latest = parent.attrs.get("latest")
    group_names = _group_names(parent)
    fallback_latest = group_names[-1] if group_names else None
    latest_name = str(latest) if isinstance(latest, str) and latest else fallback_latest
    candidates: list[SwimBoutCandidate] = []
    for run_name in group_names:
        run_group = parent[run_name]
        attrs = _attrs_dict(run_group)
        if not _matches_track(attrs, track_run_name=track_run_name, track_id=track_id):
            continue
        candidate = _candidate_from_v1_group(
            run_group,
            run_name=str(run_name),
            is_latest=str(latest_name) == str(run_name),
        )
        if candidate.signals:
            candidates.append(candidate)

    return sorted(candidates, key=lambda item: (not item.is_latest, item.run_name))


def load_default_swim_bout_tables(
    root: zarr.Group,
    *,
    run_name: str | None = "latest",
) -> SwimBoutTables:
    """Load the default signal tables for a swim-bout run."""

    candidate = resolve_swim_bout_candidate(root, run_name=run_name)
    default_signal = _default_signal(candidate)
    return load_swim_bout_tables(
        root,
        run_name=candidate.run_name,
        candidate_id=candidate.candidate_id,
        signal_id=default_signal.signal_id,
    )


def resolve_swim_bout_candidate(
    root: zarr.Group,
    *,
    run_name: str | None = "latest",
) -> SwimBoutCandidate:
    """Resolve one logical swim-bout candidate by run selector."""

    parent = _require_child(root, "analysis/swim_bout_runs")
    resolved_name = _resolve_run_name(parent, run_name)
    candidate = _candidate_from_v1_group(
        parent[resolved_name],
        run_name=resolved_name,
        is_latest=str(parent.attrs.get("latest")) == str(resolved_name),
    )
    if not candidate.signals:
        raise SwimBoutIOError(f"Swim-bout run {resolved_name!r} has no readable speed levels.")
    return candidate


def load_swim_bout_tables(
    root: zarr.Group,
    *,
    run_name: str | None = "latest",
    candidate_id: int | None = None,
    signal_id: int | None = None,
    speed_level: str | None = None,
) -> SwimBoutTables:
    """Load normalized tables for one candidate/signal selection."""

    parent = _require_child(root, "analysis/swim_bout_runs")
    resolved_name = _resolve_run_name(parent, run_name)
    run_group = parent[resolved_name]
    candidate = _candidate_from_v1_group(
        run_group,
        run_name=resolved_name,
        is_latest=str(parent.attrs.get("latest")) == str(resolved_name),
    )
    if candidate_id is not None and int(candidate_id) != candidate.candidate_id:
        raise SwimBoutIOError(
            f"Candidate id {candidate_id!r} not found in v1 swim-bout run {resolved_name!r}."
        )
    signal = _resolve_signal(candidate, signal_id=signal_id, speed_level=speed_level)
    level_group = _require_child(run_group, signal.speed_level) if signal.speed_level else run_group

    bouts = _load_structured_or_empty(level_group, "bouts", required=True)
    peak_events = _load_structured_or_empty(level_group, "peak_events")
    intervals = _load_structured_or_empty(level_group, "inter_bout_intervals")
    global_metrics = _load_structured_or_empty(level_group, "global_metrics")
    bout_points = _load_structured_or_empty(level_group, "bout_points")
    series = _load_signal_series(level_group)
    run_path = f"analysis/swim_bout_runs/{resolved_name}"
    level_path = f"{run_path}/{signal.speed_level}" if signal.speed_level else run_path
    return SwimBoutTables(
        run_name=resolved_name,
        run_path=run_path,
        level_path=level_path,
        candidate=candidate,
        signal=signal,
        bouts=bouts,
        peak_events=peak_events,
        inter_bout_intervals=intervals,
        global_metrics=global_metrics,
        bout_points=bout_points,
        series=series,
        run_attrs=_attrs_dict(run_group),
        signal_attrs=_attrs_dict(level_group),
    )


def _candidate_from_v1_group(
    run_group: zarr.Group,
    *,
    run_name: str,
    is_latest: bool,
) -> SwimBoutCandidate:
    attrs = _attrs_dict(run_group)
    default_level = _default_level_for_run(run_group)
    signals = _signals_from_v1_group(run_group, run_name=run_name, default_level=default_level)
    default_signal_id = None
    if default_level is not None:
        for signal in signals:
            if signal.speed_level == default_level:
                default_signal_id = signal.signal_id
                break
    if default_signal_id is None and len(signals) == 1 and signals[0].speed_level == "":
        default_signal_id = signals[0].signal_id
    return SwimBoutCandidate(
        run_name=run_name,
        candidate_id=0,
        candidate_name=run_name,
        run_path=f"analysis/swim_bout_runs/{run_name}",
        is_latest=is_latest,
        source_track_kinematics_run=_optional_str(attrs.get("source_track_kinematics_run")),
        track_id=_safe_int(attrs.get("track_id")),
        detection_method=str(attrs.get("detection_method", "unknown")),
        default_signal_id=default_signal_id,
        default_speed_level=default_level,
        signals=signals,
        attrs=attrs,
    )


def _signals_from_v1_group(
    run_group: zarr.Group,
    *,
    run_name: str,
    default_level: str | None,
) -> tuple[SwimBoutSignalVariant, ...]:
    signals: list[SwimBoutSignalVariant] = []
    for level in SPEED_LEVEL_ORDER:
        child = _get_child(run_group, level)
        if child is None:
            continue
        attrs = _attrs_dict(child)
        signal_id = len(signals)
        transform_type = str(attrs.get("detection_signal_transform_type", "identity"))
        role = "detector_response" if level == "speed_exponential" or transform_type != "identity" else "physical_estimator"
        source_level = _optional_str(
            attrs.get("detection_signal_source_level")
            or attrs.get("path_distance_source_level")
            or attrs.get("movement_metric_source_level")
        )
        signals.append(
            SwimBoutSignalVariant(
                run_name=run_name,
                signal_id=signal_id,
                speed_level=level,
                signal_name=level.replace("speed_", "", 1),
                role=role,
                source_level=source_level,
                is_default=level == default_level,
                n_bouts=_bout_count(child),
                attrs=attrs,
            )
        )
    if not signals and "bouts" in run_group:
        attrs = _attrs_dict(run_group)
        signals.append(
            SwimBoutSignalVariant(
                run_name=run_name,
                signal_id=0,
                speed_level="",
                signal_name="legacy",
                role="physical_estimator",
                source_level=None,
                is_default=True,
                n_bouts=_bout_count(run_group),
                attrs=attrs,
            )
        )
    return tuple(signals)


def _resolve_signal(
    candidate: SwimBoutCandidate,
    *,
    signal_id: int | None,
    speed_level: str | None,
) -> SwimBoutSignalVariant:
    if signal_id is not None:
        for signal in candidate.signals:
            if signal.signal_id == int(signal_id):
                return signal
        raise SwimBoutIOError(
            f"Signal id {signal_id!r} not found in swim-bout run {candidate.run_name!r}."
        )
    if speed_level is not None:
        level = normalize_speed_level(speed_level)
        for signal in candidate.signals:
            if signal.speed_level == level:
                return signal
        if len(candidate.signals) == 1 and candidate.signals[0].speed_level == "":
            return candidate.signals[0]
        raise SwimBoutIOError(
            f"Speed level {level!r} not found in swim-bout run {candidate.run_name!r}."
        )
    return _default_signal(candidate)


def _default_signal(candidate: SwimBoutCandidate) -> SwimBoutSignalVariant:
    if candidate.default_signal_id is not None:
        for signal in candidate.signals:
            if signal.signal_id == candidate.default_signal_id:
                return signal
    if candidate.signals:
        return candidate.signals[0]
    raise SwimBoutIOError(f"Swim-bout run {candidate.run_name!r} has no signals.")


def _default_level_for_run(run_group: zarr.Group) -> str | None:
    raw = run_group.attrs.get("default_level", "speed_smoothed")
    try:
        default = normalize_speed_level(raw)
    except SwimBoutIOError:
        default = "speed_smoothed"
    if default in run_group:
        return default
    for level in SPEED_LEVEL_ORDER:
        if level in run_group:
            return level
    if "bouts" in run_group:
        return ""
    return None


def _load_structured_or_empty(
    group: zarr.Group,
    name: str,
    *,
    required: bool = False,
) -> np.ndarray:
    node = _get_child(group, name)
    if node is None:
        if required:
            raise SwimBoutIOError(f"Missing required swim-bout table {name!r}.")
        return np.zeros(0, dtype=[])
    try:
        return load_structured_dataset(group, name)[0]
    except Exception as exc:
        records = _read_simple_column_group_or_empty(node)
        if records.dtype.names:
            return records
        if required:
            raise SwimBoutIOError(f"Unable to read required swim-bout table {name!r}.") from exc
        return np.zeros(0, dtype=[])


def _read_simple_column_group_or_empty(group: Any) -> np.ndarray:
    """Read a legacy group of aligned 1D arrays into a structured array."""

    arrays: dict[str, np.ndarray] = {}
    n_rows: int | None = None
    try:
        names = sorted(str(name) for name in group.keys())
    except Exception:
        return np.zeros(0, dtype=[])
    for name in names:
        try:
            child = group[name]
            if hasattr(child, "keys"):
                continue
            arr = np.asarray(child[:])
        except Exception:
            continue
        if arr.ndim != 1:
            continue
        if n_rows is None:
            n_rows = int(arr.shape[0])
        if int(arr.shape[0]) != n_rows:
            continue
        arrays[name] = arr
    if not arrays or n_rows is None:
        return np.zeros(0, dtype=[])

    dtype = [(name, arr.dtype) for name, arr in arrays.items()]
    records = np.empty(n_rows, dtype=dtype)
    for name, arr in arrays.items():
        records[name] = arr
    return records


def _load_signal_series(group: zarr.Group) -> dict[str, np.ndarray]:
    series: dict[str, np.ndarray] = {}
    for name in ("detection_signal_mm_s", "speed_exponential_mm", "frame_indices"):
        if name not in group:
            continue
        try:
            series[name] = np.asarray(group[name][:])
        except Exception:
            continue
    return series


def _bout_count(level_group: zarr.Group) -> int:
    attr_count = _safe_int(level_group.attrs.get("n_bouts"))
    if attr_count is not None:
        return attr_count
    bouts = _get_child(level_group, "bouts")
    if bouts is None:
        return 0
    shape = getattr(bouts, "shape", None)
    if shape is not None:
        try:
            return int(shape[0])
        except Exception:
            pass
    field_names = list(getattr(bouts, "attrs", {}).get("field_names", []))
    for name in field_names:
        if name not in bouts:
            continue
        try:
            return int(bouts[name].shape[0])
        except Exception:
            continue
    return 0


def _get_child(group: Any, path: str) -> Any | None:
    current = group
    for part in str(path).strip("/").split("/"):
        if not part:
            continue
        try:
            if part not in current:
                return None
            current = current[part]
        except Exception:
            return None
    return current


def _require_child(group: Any, path: str) -> Any:
    child = _get_child(group, path)
    if child is None:
        raise SwimBoutIOError(f"Missing Zarr group: {path}")
    return child


def _resolve_run_name(parent: zarr.Group, run_name: str | None) -> str:
    requested = "latest" if run_name is None else str(run_name).strip().strip("/")
    if requested.startswith("analysis/swim_bout_runs/"):
        requested = requested.split("/", 2)[2].split("/", 1)[0]
    if requested in ("", "latest"):
        latest = parent.attrs.get("latest")
        if isinstance(latest, str) and latest:
            requested = latest
        else:
            names = _group_names(parent)
            if not names:
                raise SwimBoutIOError("No swim-bout runs available.")
            requested = names[-1]
    if requested not in parent:
        raise SwimBoutIOError(f"Swim-bout run {requested!r} not found.")
    return requested


def _group_names(group: Any) -> list[str]:
    try:
        names = group.group_keys()
    except Exception:
        try:
            names = [name for name in group.keys() if hasattr(group[name], "keys")]
        except Exception:
            return []
    return sorted(str(name) for name in names)


def _attrs_dict(group: Any) -> dict[str, Any]:
    try:
        return {str(key): _scalar_value(value) for key, value in dict(group.attrs).items()}
    except Exception:
        return {}


def _matches_track(
    attrs: Mapping[str, Any],
    *,
    track_run_name: object | None,
    track_id: int | None,
) -> bool:
    if track_run_name is not None:
        source_run = attrs.get("source_track_kinematics_run")
        if source_run is None or not _run_names_match(source_run, track_run_name):
            return False
    if track_id is not None:
        source_track_id = _safe_int(attrs.get("track_id"))
        if source_track_id is None or source_track_id != int(track_id):
            return False
    return True


def _run_names_match(source_run: object, requested_run: object) -> bool:
    source = _normalize_path(str(source_run))
    requested = _normalize_path(str(requested_run))
    return source == requested or source.endswith(f"/{requested}") or requested.endswith(f"/{source}")


def _normalize_path(path: str) -> str:
    return "/".join(part for part in str(path).strip("/").split("/") if part)


def _optional_str(value: object) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def _safe_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _scalar_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _scalar_value(value.item())
    if isinstance(value, bytes):
        return value.rstrip(b"\x00").decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _scalar_value(value.item())
        return [_scalar_value(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _scalar_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scalar_value(item) for item in value]
    return value
