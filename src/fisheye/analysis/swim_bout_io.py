"""Logical readers for Palette swim-bout runs.

This module provides a compatibility layer over historical hierarchical and
current compact-tabular layouts. Compact detector traces may use a versioned
same-Zarr reference to the authoritative track-kinematics frame axis; callers
do not need to depend on that physical storage choice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import zarr

from fisheye.shared.zarr.columnar import load_structured_dataset
from fisheye.analysis.swim_bout_frame_axis import (
    SwimBoutFrameAxisError,
    resolve_swim_bout_frame_axis,
)
from fisheye.shared.json_safety import decode_null_terminated_text
from fisheye.shared.zarr_helpers import (
    normalize_zarr_path as _normalize_path,
    zarr_group_keys,
)


COMPACT_V2_LAYOUT = "compact_tabular_v2"
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
    inter_bout_interval_histogram: np.ndarray
    global_metrics: np.ndarray
    trials: np.ndarray
    bout_points: np.ndarray
    series: Mapping[str, np.ndarray]
    run_attrs: Mapping[str, Any]
    signal_attrs: Mapping[str, Any]


@dataclass(frozen=True)
class SwimBoutEvents:
    """Projected event-only payload for interactive timelines."""

    run_name: str
    run_path: str
    level_path: str
    candidate: SwimBoutCandidate
    signal: SwimBoutSignalVariant
    bouts: np.ndarray
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
    include_bout_counts: bool = True,
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
        for candidate in _candidates_from_run_group(
            run_group,
            run_name=str(run_name),
            is_latest=str(latest_name) == str(run_name),
            include_bout_counts=include_bout_counts,
        ):
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
    candidate = _default_candidate_from_run_group(
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
    is_latest = str(parent.attrs.get("latest")) == str(resolved_name)
    if _is_compact_v2_group(run_group):
        return _load_compact_v2_tables(
            root,
            run_group,
            run_name=resolved_name,
            is_latest=is_latest,
            candidate_id=candidate_id,
            signal_id=signal_id,
            speed_level=speed_level,
        )
    candidate = _candidate_from_v1_group(run_group, run_name=resolved_name, is_latest=is_latest)
    if candidate_id is not None and int(candidate_id) != candidate.candidate_id:
        raise SwimBoutIOError(
            f"Candidate id {candidate_id!r} not found in v1 swim-bout run {resolved_name!r}."
        )
    signal = _resolve_signal(candidate, signal_id=signal_id, speed_level=speed_level)
    level_group = _require_child(run_group, signal.speed_level) if signal.speed_level else run_group

    bouts = _load_structured_or_empty(level_group, "bouts", required=True)
    peak_events = _load_structured_or_empty(level_group, "peak_events")
    intervals = _load_structured_or_empty(level_group, "inter_bout_intervals")
    interval_histogram = _load_structured_or_empty(level_group, "inter_bout_interval_histogram")
    global_metrics = _load_structured_or_empty(level_group, "global_metrics")
    trials = _load_structured_or_empty(level_group, "trials")
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
        inter_bout_interval_histogram=interval_histogram,
        global_metrics=global_metrics,
        trials=trials,
        bout_points=bout_points,
        series=series,
        run_attrs=_attrs_dict(run_group),
        signal_attrs=_attrs_dict(level_group),
    )


def load_swim_bout_events(
    root: zarr.Group,
    *,
    candidate: SwimBoutCandidate,
    signal: SwimBoutSignalVariant,
) -> SwimBoutEvents:
    """Load only the selected persisted bout-event rows.

    Unlike :func:`load_swim_bout_tables`, this projection does not read peak
    events, intervals, histograms, summary metrics, point tables, trials, or
    detector-signal arrays.
    """

    parent = _require_child(root, "analysis/swim_bout_runs")
    run_group = _require_child(parent, candidate.run_name)
    if _is_compact_v2_group(run_group):
        tables = _require_child(run_group, "tables")
        bouts = _filter_records(
            _load_structured_or_empty(tables, "bouts", required=True),
            candidate_id=candidate.candidate_id,
            signal_id=signal.signal_id,
        )
        indexes = _require_child(run_group, "indexes")
        signal_rows = _load_structured_or_empty(indexes, "signal_variants")
        signal_row = _row_by_int_field(signal_rows, "signal_id", signal.signal_id)
        level_path = (
            f"analysis/swim_bout_runs/{candidate.run_name}/tables/bouts"
            f"?candidate_id={candidate.candidate_id}&signal_id={signal.signal_id}"
        )
        signal_attrs = _record_to_dict(signal_row) if signal_row is not None else signal.attrs
    else:
        level_group = _require_child(run_group, signal.speed_level) if signal.speed_level else run_group
        bouts = _load_structured_or_empty(level_group, "bouts", required=True)
        level_path = (
            f"analysis/swim_bout_runs/{candidate.run_name}/{signal.speed_level}"
            if signal.speed_level
            else f"analysis/swim_bout_runs/{candidate.run_name}"
        )
        signal_attrs = _attrs_dict(level_group)
    return SwimBoutEvents(
        run_name=candidate.run_name,
        run_path=f"analysis/swim_bout_runs/{candidate.run_name}",
        level_path=level_path,
        candidate=candidate,
        signal=signal,
        bouts=bouts,
        run_attrs=_attrs_dict(run_group),
        signal_attrs=signal_attrs,
    )


def _is_compact_v2_group(run_group: zarr.Group) -> bool:
    attrs = _attrs_dict(run_group)
    return attrs.get("layout") == COMPACT_V2_LAYOUT or _get_child(run_group, "indexes/candidates") is not None


def _candidates_from_run_group(
    run_group: zarr.Group,
    *,
    run_name: str,
    is_latest: bool,
    include_bout_counts: bool = True,
) -> list[SwimBoutCandidate]:
    if _is_compact_v2_group(run_group):
        return _candidates_from_compact_v2_group(
            run_group,
            run_name=run_name,
            is_latest=is_latest,
            include_bout_counts=include_bout_counts,
        )
    return [_candidate_from_v1_group(run_group, run_name=run_name, is_latest=is_latest)]


def _default_candidate_from_run_group(
    run_group: zarr.Group,
    *,
    run_name: str,
    is_latest: bool,
) -> SwimBoutCandidate:
    candidates = _candidates_from_run_group(run_group, run_name=run_name, is_latest=is_latest)
    if not candidates:
        raise SwimBoutIOError(f"Swim-bout run {run_name!r} has no candidates.")
    default_candidate_id = _safe_int(_attrs_dict(run_group).get("default_candidate_id"))
    if default_candidate_id is not None:
        for candidate in candidates:
            if candidate.candidate_id == default_candidate_id:
                return candidate
    for candidate in candidates:
        if candidate.attrs.get("is_default") is True:
            return candidate
    return candidates[0]


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


def _candidates_from_compact_v2_group(
    run_group: zarr.Group,
    *,
    run_name: str,
    is_latest: bool,
    include_bout_counts: bool = True,
) -> list[SwimBoutCandidate]:
    candidate_rows = _load_structured_or_empty(_require_child(run_group, "indexes"), "candidates")
    if candidate_rows.size == 0:
        return []
    signal_rows = _load_structured_or_empty(_require_child(run_group, "indexes"), "signal_variants")
    bouts = (
        _load_structured_or_empty(_require_child(run_group, "tables"), "bouts")
        if include_bout_counts
        else np.zeros(0, dtype=[])
    )
    run_attrs = _attrs_dict(run_group)
    default_signal_id = _safe_int(run_attrs.get("default_signal_id"))
    default_speed_level = _signal_row_speed_level(
        _row_by_int_field(signal_rows, "signal_id", default_signal_id)
    ) if default_signal_id is not None else None
    candidates: list[SwimBoutCandidate] = []
    for row in candidate_rows:
        row_attrs = _record_to_dict(row)
        candidate_id = _safe_int(row_attrs.get("candidate_id"))
        if candidate_id is None:
            continue
        signals = _signals_from_compact_v2_rows(
            signal_rows,
            run_name=run_name,
            default_signal_id=default_signal_id,
            n_bouts_by_signal=_count_records_by_signal(bouts, candidate_id=candidate_id),
        )
        detection_method = str(row_attrs.get("detection_method") or run_attrs.get("detection_method", "unknown"))
        candidates.append(
            SwimBoutCandidate(
                run_name=run_name,
                candidate_id=candidate_id,
                candidate_name=str(row_attrs.get("candidate_name") or f"candidate_{candidate_id}"),
                run_path=f"analysis/swim_bout_runs/{run_name}",
                is_latest=is_latest,
                source_track_kinematics_run=_optional_str(run_attrs.get("source_track_kinematics_run")),
                track_id=_safe_int(run_attrs.get("track_id")),
                detection_method=detection_method,
                default_signal_id=default_signal_id,
                default_speed_level=default_speed_level,
                signals=signals,
                attrs={**run_attrs, **row_attrs},
            )
        )
    return candidates


def _signals_from_compact_v2_rows(
    signal_rows: np.ndarray,
    *,
    run_name: str,
    default_signal_id: int | None,
    n_bouts_by_signal: Mapping[int, int] | None = None,
) -> tuple[SwimBoutSignalVariant, ...]:
    signals: list[SwimBoutSignalVariant] = []
    for row in signal_rows:
        attrs = _record_to_dict(row)
        signal_id = _safe_int(attrs.get("signal_id"))
        if signal_id is None:
            continue
        speed_level = _signal_row_speed_level(row)
        signal_name = str(attrs.get("signal_name") or speed_level.replace("speed_", "", 1))
        signals.append(
            SwimBoutSignalVariant(
                run_name=run_name,
                signal_id=signal_id,
                speed_level=speed_level,
                signal_name=signal_name,
                role=str(attrs.get("role") or "physical_estimator"),
                source_level=_optional_str(attrs.get("source_level")),
                is_default=default_signal_id == signal_id,
                n_bouts=int((n_bouts_by_signal or {}).get(signal_id, 0)),
                attrs=attrs,
            )
        )
    return tuple(signals)


def _load_compact_v2_tables(
    root: zarr.Group,
    run_group: zarr.Group,
    *,
    run_name: str,
    is_latest: bool,
    candidate_id: int | None,
    signal_id: int | None,
    speed_level: str | None,
) -> SwimBoutTables:
    candidate = _resolve_compact_candidate(
        run_group,
        run_name=run_name,
        is_latest=is_latest,
        candidate_id=candidate_id,
    )
    signal = _resolve_signal(candidate, signal_id=signal_id, speed_level=speed_level)
    indexes = _require_child(run_group, "indexes")
    tables = _require_child(run_group, "tables")
    raw_bouts = _load_structured_or_empty(tables, "bouts", required=True)
    bouts = _filter_records(raw_bouts, candidate_id=candidate.candidate_id, signal_id=signal.signal_id)
    peak_events = _filter_records(
        _load_structured_or_empty(tables, "peak_events"),
        candidate_id=candidate.candidate_id,
        signal_id=signal.signal_id,
    )
    intervals = _filter_records(
        _load_structured_or_empty(tables, "inter_bout_intervals"),
        candidate_id=candidate.candidate_id,
        signal_id=signal.signal_id,
    )
    summary_metrics = _filter_records(
        _load_structured_or_empty(tables, "summary_metrics"),
        candidate_id=candidate.candidate_id,
        signal_id=signal.signal_id,
    )
    histograms = _filter_records(
        _load_structured_or_empty(tables, "histograms"),
        candidate_id=candidate.candidate_id,
        signal_id=signal.signal_id,
    )
    bout_points = _filter_records(
        _load_structured_or_empty(tables, "bout_points"),
        candidate_id=candidate.candidate_id,
        signal_id=signal.signal_id,
    )
    signal_rows = _load_structured_or_empty(indexes, "signal_variants")
    signal_row = _row_by_int_field(signal_rows, "signal_id", signal.signal_id)
    return SwimBoutTables(
        run_name=run_name,
        run_path=f"analysis/swim_bout_runs/{run_name}",
        level_path=f"analysis/swim_bout_runs/{run_name}/tables/bouts?candidate_id={candidate.candidate_id}&signal_id={signal.signal_id}",
        candidate=candidate,
        signal=signal,
        bouts=bouts,
        peak_events=peak_events,
        inter_bout_intervals=intervals,
        inter_bout_interval_histogram=_histogram_rows_to_legacy(histograms),
        global_metrics=_summary_metrics_to_legacy(summary_metrics),
        trials=np.zeros(0, dtype=[]),
        bout_points=bout_points,
        series=_load_compact_signal_series(root, run_group, signal=signal),
        run_attrs=_attrs_dict(run_group),
        signal_attrs=_record_to_dict(signal_row) if signal_row is not None else signal.attrs,
    )


def _resolve_compact_candidate(
    run_group: zarr.Group,
    *,
    run_name: str,
    is_latest: bool,
    candidate_id: int | None,
) -> SwimBoutCandidate:
    candidates = _candidates_from_compact_v2_group(run_group, run_name=run_name, is_latest=is_latest)
    if not candidates:
        raise SwimBoutIOError(f"Swim-bout run {run_name!r} has no compact candidates.")
    if candidate_id is not None:
        for candidate in candidates:
            if candidate.candidate_id == int(candidate_id):
                return candidate
        raise SwimBoutIOError(f"Candidate id {candidate_id!r} not found in compact swim-bout run {run_name!r}.")
    return _default_candidate_from_run_group(run_group, run_name=run_name, is_latest=is_latest)


def _signal_row_speed_level(row: np.void | None) -> str:
    if row is None:
        return ""
    attrs = _record_to_dict(row)
    raw = attrs.get("speed_level") or attrs.get("signal_name")
    if raw is None:
        return ""
    try:
        return normalize_speed_level(raw)
    except SwimBoutIOError:
        text = str(raw)
        return text if text.startswith("speed_") else f"speed_{text}"


def _row_by_int_field(records: np.ndarray, field: str, value: int | None) -> np.void | None:
    if value is None or records.dtype.names is None or field not in records.dtype.names:
        return None
    for row in records:
        if _safe_int(row[field]) == int(value):
            return row
    return None


def _record_to_dict(record: np.void | None) -> dict[str, Any]:
    if record is None or getattr(record, "dtype", None) is None or record.dtype.names is None:
        return {}
    return {name: _scalar_value(record[name]) for name in record.dtype.names}


def _filter_records(
    records: np.ndarray,
    *,
    candidate_id: int | None = None,
    signal_id: int | None = None,
) -> np.ndarray:
    if records.dtype.names is None or records.size == 0:
        return records
    mask = np.ones(records.shape[0], dtype=bool)
    if candidate_id is not None and "candidate_id" in records.dtype.names:
        mask &= np.asarray(records["candidate_id"], dtype=np.int64) == int(candidate_id)
    if signal_id is not None and "signal_id" in records.dtype.names:
        mask &= np.asarray(records["signal_id"], dtype=np.int64) == int(signal_id)
    return records[mask]


def _count_records_by_signal(records: np.ndarray, *, candidate_id: int | None = None) -> dict[int, int]:
    if records.dtype.names is None or "signal_id" not in records.dtype.names:
        return {}
    filtered = _filter_records(records, candidate_id=candidate_id)
    counts: dict[int, int] = {}
    for value in filtered["signal_id"]:
        signal_id = _safe_int(value)
        if signal_id is None:
            continue
        counts[signal_id] = counts.get(signal_id, 0) + 1
    return counts


def _summary_metrics_to_legacy(records: np.ndarray) -> np.ndarray:
    if records.dtype.names is None or records.size == 0:
        return np.zeros(0, dtype=[])
    required = {"metric_name", "value"}
    if not required.issubset(set(records.dtype.names)):
        return records
    metric_names = [_scalar_value(value) for value in records["metric_name"]]
    dtype = [(str(name), "f8") for name in metric_names]
    result = np.zeros(1, dtype=dtype)
    for name, value in zip(metric_names, records["value"]):
        result[str(name)][0] = float(value)
    return result


def _histogram_rows_to_legacy(records: np.ndarray) -> np.ndarray:
    if records.dtype.names is None or records.size == 0:
        return np.zeros(0, dtype=[])
    names = set(records.dtype.names)
    if {"bin_left_edge_s", "bin_right_edge_s", "count"}.issubset(names):
        return records
    if not {"bin_left", "bin_right", "count"}.issubset(names):
        return records
    result = np.zeros(
        records.shape[0],
        dtype=[
            ("bin_left_edge_s", "f8"),
            ("bin_right_edge_s", "f8"),
            ("count", "i8"),
        ],
    )
    result["bin_left_edge_s"] = np.asarray(records["bin_left"], dtype=np.float64)
    result["bin_right_edge_s"] = np.asarray(records["bin_right"], dtype=np.float64)
    result["count"] = np.asarray(records["count"], dtype=np.int64)
    return result


def _load_compact_signal_series(
    root: zarr.Group,
    run_group: zarr.Group,
    *,
    signal: SwimBoutSignalVariant,
) -> dict[str, np.ndarray]:
    series: dict[str, np.ndarray] = {}
    signals_group = _get_child(run_group, "signals")
    if signals_group is None:
        return series
    detector_signal = _get_child(signals_group, "detector_signal_mm_s")
    if detector_signal is not None:
        signal_ids_node = _get_child(signals_group, "detector_signal_signal_ids")
        if signal_ids_node is not None:
            try:
                signal_ids = np.asarray(signal_ids_node[:], dtype=np.int64)
            except Exception:
                signal_ids = np.zeros(0, dtype=np.int64)
        else:
            signal_ids = np.asarray([signal.signal_id], dtype=np.int64)
        matches = np.flatnonzero(signal_ids == int(signal.signal_id))
        if matches.size:
            try:
                if int(detector_signal.ndim) >= 2:
                    # Compact v2 is signal-major. Select the logical row before
                    # reading so time-axis-sharded arrays only fetch the chosen
                    # detector signal.
                    values = np.asarray(detector_signal[int(matches[0]), :])
                else:
                    values = np.asarray(detector_signal[:])
            except Exception:
                values = np.zeros(0, dtype=np.float32)
            if values.size:
                series["detection_signal_mm_s"] = values
                if signal.speed_level == "speed_exponential":
                    series["speed_exponential_mm"] = values
    expected_length = None
    if "detection_signal_mm_s" in series:
        expected_length = int(series["detection_signal_mm_s"].size)
    try:
        frame_indices = resolve_swim_bout_frame_axis(
            root,
            run_group,
            expected_length=expected_length,
        )
    except SwimBoutFrameAxisError as exc:
        raise SwimBoutIOError(
            f"Cannot resolve frame axis for swim-bout run {signal.run_name!r}: {exc}"
        ) from exc
    if frame_indices is not None:
        series["frame_indices"] = frame_indices
    return series


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
    return zarr_group_keys(group)


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
    if isinstance(value, (bytes, np.bytes_)):
        return decode_null_terminated_text(value)
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _scalar_value(value.item())
        return [_scalar_value(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _scalar_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scalar_value(item) for item in value]
    return value
