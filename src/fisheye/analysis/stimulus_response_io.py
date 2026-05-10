"""Logical readers for stimulus-response analysis runs.

The current writer stores stimulus-response output as a hierarchical step tree.
This module is the compatibility boundary for readers: consumers should ask for
logical tables here rather than walking physical paths directly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import zarr

from fisheye.analysis.chaser_state_interpolator import read_columnar_dataset


ArrayMapping = Mapping[str, np.ndarray]


@dataclass(frozen=True)
class StimulusResponseMetricTables:
    """Metric tables for one stimulus-response metric family."""

    attrs: Mapping[str, Any] = field(default_factory=dict)
    per_frame: ArrayMapping = field(default_factory=dict)
    per_fish: ArrayMapping = field(default_factory=dict)
    per_bout: ArrayMapping = field(default_factory=dict)
    windows: ArrayMapping = field(default_factory=dict)
    early_windows: ArrayMapping = field(default_factory=dict)
    time_series: ArrayMapping = field(default_factory=dict)
    trials: ArrayMapping = field(default_factory=dict)
    per_trial_per_fish: ArrayMapping = field(default_factory=dict)


@dataclass(frozen=True)
class StimulusResponseStepTables:
    """Logical tables for one protocol step in a stimulus-response run."""

    step_key: str
    step_index: int
    step_name: str
    stimulus_mode: str
    stimulus_mode_id: int | None
    start_frame: int | None
    end_frame: int | None
    duration_s: float | None
    attrs: Mapping[str, Any]
    per_fish: ArrayMapping = field(default_factory=dict)
    per_bout: ArrayMapping = field(default_factory=dict)
    grating_per_frame: ArrayMapping = field(default_factory=dict)
    grating_per_fish: ArrayMapping = field(default_factory=dict)
    grating_time_series: ArrayMapping = field(default_factory=dict)
    moving_grating_omr: StimulusResponseMetricTables | None = None
    concentric_per_frame: ArrayMapping = field(default_factory=dict)
    concentric_per_fish: ArrayMapping = field(default_factory=dict)
    concentric_time_series: ArrayMapping = field(default_factory=dict)
    concentric_radial_omr: StimulusResponseMetricTables | None = None
    looming: StimulusResponseMetricTables | None = None


@dataclass(frozen=True)
class StimulusResponseTables:
    """Logical view over a stimulus-response run."""

    layout: str
    attrs: Mapping[str, Any]
    global_per_fish: ArrayMapping = field(default_factory=dict)
    global_omr_per_fish: ArrayMapping = field(default_factory=dict)
    frame_annotations: ArrayMapping = field(default_factory=dict)
    steps: tuple[StimulusResponseStepTables, ...] = ()


def _attrs_dict(group: Any) -> dict[str, Any]:
    attrs = getattr(group, "attrs", {})
    try:
        return {str(key): value for key, value in attrs.items()}
    except Exception:
        return {}


def _keys(group: Any) -> list[str]:
    try:
        return [str(key) for key in group.keys()]
    except Exception:
        return []


def _has_child(group: Any, name: str) -> bool:
    try:
        return name in group
    except Exception:
        return False


def _child_group(group: Any, name: str) -> Any | None:
    if not _has_child(group, name):
        return None
    try:
        child = group[name]
    except Exception:
        return None
    return child if hasattr(child, "keys") else None


def read_array_mapping(group: Any | None) -> dict[str, np.ndarray]:
    """Read direct child arrays from a Zarr group into a name-to-array mapping."""

    if group is None:
        return {}
    mapping: dict[str, np.ndarray] = {}
    for name in _keys(group):
        try:
            value = group[name]
        except Exception:
            continue
        if hasattr(value, "shape"):
            try:
                mapping[str(name)] = np.asarray(value[:])
            except Exception:
                continue
    return mapping


def _read_child_array_mapping(group: Any | None, child_name: str) -> dict[str, np.ndarray]:
    return read_array_mapping(_child_group(group, child_name))


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _decode_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).rstrip(b"\x00").decode("utf-8", errors="ignore")
    return value


def _read_columnar_mapping(group: Any, name: str) -> dict[str, np.ndarray]:
    child = _child_group(group, name)
    if child is None:
        return {}
    try:
        table = read_columnar_dataset(child)
    except Exception:
        return read_array_mapping(child)
    return {str(field): np.asarray(table[field]) for field in table.dtype.names or ()}


def _table_records(group: Any, name: str) -> np.ndarray | None:
    child = _child_group(group, name)
    if child is None:
        return None
    try:
        return read_columnar_dataset(child)
    except Exception:
        return None


def _filter_mapping_by_step(mapping: Mapping[str, np.ndarray], step_index: int) -> dict[str, np.ndarray]:
    if not mapping or "step_index" not in mapping:
        return {}
    step_values = np.asarray(mapping["step_index"])
    mask = step_values.astype(np.int64, copy=False) == int(step_index)
    if not np.any(mask):
        return {}
    excluded = {
        "step_index",
        "step_name",
        "stimulus_mode",
        "stimulus_mode_id",
        "start_frame",
        "end_frame",
        "duration_s",
        "stimulus_family",
        "metric_family",
    }
    out: dict[str, np.ndarray] = {}
    for name, values in mapping.items():
        if name in excluded:
            continue
        arr = np.asarray(values)
        if arr.ndim >= 1 and arr.shape[0] == mask.shape[0]:
            out[str(name)] = arr[mask]
    return out


def _attrs_by_step(attrs: Mapping[str, Any], name: str) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    raw = attrs.get(name, [])
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        step_index = _safe_int(item.get("step_index"))
        payload = item.get("attrs", {})
        if step_index is None or not isinstance(payload, Mapping):
            continue
        out[int(step_index)] = {str(key): value for key, value in payload.items()}
    return out


def _step_sort_key(name: str) -> tuple[int, str]:
    match = re.fullmatch(r"step_(\d+)", str(name))
    if match:
        return int(match.group(1)), str(name)
    return 10**12, str(name)


def _metric_tables(group: Any | None) -> StimulusResponseMetricTables | None:
    if group is None:
        return None
    return StimulusResponseMetricTables(
        attrs=_attrs_dict(group),
        per_frame=_read_child_array_mapping(group, "per_frame"),
        per_fish=_read_child_array_mapping(group, "per_fish"),
        per_bout=_read_child_array_mapping(group, "per_bout"),
        windows=_read_child_array_mapping(group, "windows"),
        early_windows=_read_child_array_mapping(group, "early_windows"),
        time_series=_read_child_array_mapping(group, "time_series"),
        trials=_read_child_array_mapping(group, "trials"),
        per_trial_per_fish=_read_child_array_mapping(group, "per_trial_per_fish"),
    )


def _resolve_hierarchical_v1(run_group: zarr.Group, *, layout: str) -> StimulusResponseTables:
    steps: list[StimulusResponseStepTables] = []
    steps_group = _child_group(run_group, "steps")
    if steps_group is not None:
        for step_key in sorted(_keys(steps_group), key=_step_sort_key):
            step_group = _child_group(steps_group, step_key)
            if step_group is None:
                continue
            attrs = _attrs_dict(step_group)
            grating_group = _child_group(step_group, "grating")
            concentric_group = _child_group(step_group, "concentric_grating")
            looming_group = _child_group(step_group, "looming")
            step_index = _safe_int(attrs.get("step_index"))
            if step_index is None:
                step_index = _step_sort_key(step_key)[0]
            steps.append(
                StimulusResponseStepTables(
                    step_key=str(step_key),
                    step_index=int(step_index),
                    step_name=str(attrs.get("step_name", step_key)),
                    stimulus_mode=str(attrs.get("stimulus_mode", "")),
                    stimulus_mode_id=_safe_int(attrs.get("stimulus_mode_id")),
                    start_frame=_safe_int(attrs.get("start_frame", attrs.get("start_camera_frame"))),
                    end_frame=_safe_int(attrs.get("end_frame", attrs.get("end_camera_frame"))),
                    duration_s=_safe_float(attrs.get("duration_s")),
                    attrs=attrs,
                    per_fish=_read_child_array_mapping(step_group, "per_fish"),
                    per_bout=_read_child_array_mapping(step_group, "per_bout"),
                    grating_per_frame=_read_child_array_mapping(grating_group, "per_frame"),
                    grating_per_fish=_read_child_array_mapping(grating_group, "per_fish"),
                    grating_time_series=_read_child_array_mapping(grating_group, "time_series"),
                    moving_grating_omr=_metric_tables(_child_group(grating_group, "omr")),
                    concentric_per_frame=_read_child_array_mapping(concentric_group, "per_frame"),
                    concentric_per_fish=_read_child_array_mapping(concentric_group, "per_fish"),
                    concentric_time_series=_read_child_array_mapping(concentric_group, "time_series"),
                    concentric_radial_omr=_metric_tables(_child_group(concentric_group, "radial_omr")),
                    looming=_metric_tables(looming_group),
                )
            )

    global_group = _child_group(run_group, "global")
    global_omr = _child_group(global_group, "omr")
    return StimulusResponseTables(
        layout=layout,
        attrs=_attrs_dict(run_group),
        global_per_fish=read_array_mapping(global_group),
        global_omr_per_fish=_read_child_array_mapping(global_omr, "per_fish"),
        frame_annotations=_read_child_array_mapping(run_group, "frames"),
        steps=tuple(steps),
    )


def _step_row_value(record: np.void, name: str, default: Any = None) -> Any:
    names = record.dtype.names or ()
    if name not in names:
        return default
    return _decode_scalar(record[name])


def _resolve_compact_tabular_v2(run_group: zarr.Group, *, layout: str) -> StimulusResponseTables:
    attrs = _attrs_dict(run_group)
    step_records = _table_records(run_group, "step_index")
    if step_records is None:
        step_records = np.zeros(0, dtype=[])

    step_per_fish = _read_columnar_mapping(run_group, "step_per_fish")
    step_per_bout = _read_columnar_mapping(run_group, "step_per_bout")
    grating_per_fish = _read_columnar_mapping(run_group, "grating_per_fish")
    moving_omr_per_fish = _read_columnar_mapping(run_group, "moving_grating_omr_per_fish")
    moving_omr_per_bout = _read_columnar_mapping(run_group, "moving_grating_omr_per_bout")
    moving_omr_windows = _read_columnar_mapping(run_group, "moving_grating_omr_windows")
    moving_omr_early = _read_columnar_mapping(run_group, "moving_grating_omr_early_windows")
    concentric_per_fish = _read_columnar_mapping(run_group, "concentric_per_fish")
    radial_omr_per_fish = _read_columnar_mapping(run_group, "concentric_radial_omr_per_fish")
    radial_omr_per_bout = _read_columnar_mapping(run_group, "concentric_radial_omr_per_bout")
    radial_omr_windows = _read_columnar_mapping(run_group, "concentric_radial_omr_windows")
    radial_omr_early = _read_columnar_mapping(run_group, "concentric_radial_omr_early_windows")
    looming_trials = _read_columnar_mapping(run_group, "looming_trials")
    looming_per_trial_per_fish = _read_columnar_mapping(run_group, "looming_per_trial_per_fish")
    looming_per_fish = _read_columnar_mapping(run_group, "looming_per_fish")

    moving_attrs = _attrs_by_step(attrs, "moving_grating_omr_attrs")
    radial_attrs = _attrs_by_step(attrs, "concentric_radial_omr_attrs")
    looming_attrs = _attrs_by_step(attrs, "looming_attrs")

    steps: list[StimulusResponseStepTables] = []
    for idx, record in enumerate(step_records):
        step_index = _safe_int(_step_row_value(record, "step_index", idx))
        if step_index is None:
            step_index = idx
        step_attrs = {
            "step_index": step_index,
            "step_name": _step_row_value(record, "step_name", f"step_{step_index}"),
            "stimulus_mode": _step_row_value(record, "stimulus_mode", ""),
            "stimulus_mode_id": _safe_int(_step_row_value(record, "stimulus_mode_id")),
            "start_frame": _safe_int(_step_row_value(record, "start_frame")),
            "end_frame": _safe_int(_step_row_value(record, "end_frame")),
            "duration_s": _safe_float(_step_row_value(record, "duration_s")),
        }
        if "stimulus_params_json" in (record.dtype.names or ()):
            step_attrs["stimulus_params_json"] = _step_row_value(record, "stimulus_params_json", "")

        moving_pf = _filter_mapping_by_step(moving_omr_per_fish, int(step_index))
        moving_pb = _filter_mapping_by_step(moving_omr_per_bout, int(step_index))
        moving_win = _filter_mapping_by_step(moving_omr_windows, int(step_index))
        moving_early = _filter_mapping_by_step(moving_omr_early, int(step_index))
        moving_omr = None
        if moving_pf or moving_pb or moving_win or moving_early or int(step_index) in moving_attrs:
            moving_omr = StimulusResponseMetricTables(
                attrs=moving_attrs.get(int(step_index), {}),
                per_fish=moving_pf,
                per_bout=moving_pb,
                windows=moving_win,
                early_windows=moving_early,
            )

        radial_pf = _filter_mapping_by_step(radial_omr_per_fish, int(step_index))
        radial_pb = _filter_mapping_by_step(radial_omr_per_bout, int(step_index))
        radial_win = _filter_mapping_by_step(radial_omr_windows, int(step_index))
        radial_early_step = _filter_mapping_by_step(radial_omr_early, int(step_index))
        radial_omr = None
        if radial_pf or radial_pb or radial_win or radial_early_step or int(step_index) in radial_attrs:
            radial_omr = StimulusResponseMetricTables(
                attrs=radial_attrs.get(int(step_index), {}),
                per_fish=radial_pf,
                per_bout=radial_pb,
                windows=radial_win,
                early_windows=radial_early_step,
            )

        loom_trials = _filter_mapping_by_step(looming_trials, int(step_index))
        loom_per_trial = _filter_mapping_by_step(looming_per_trial_per_fish, int(step_index))
        loom_pf = _filter_mapping_by_step(looming_per_fish, int(step_index))
        looming = None
        if loom_trials or loom_per_trial or loom_pf or int(step_index) in looming_attrs:
            looming = StimulusResponseMetricTables(
                attrs=looming_attrs.get(int(step_index), {}),
                trials=loom_trials,
                per_trial_per_fish=loom_per_trial,
                per_fish=loom_pf,
            )

        steps.append(
            StimulusResponseStepTables(
                step_key=f"step_{step_index}",
                step_index=int(step_index),
                step_name=str(step_attrs["step_name"]),
                stimulus_mode=str(step_attrs["stimulus_mode"]),
                stimulus_mode_id=_safe_int(step_attrs.get("stimulus_mode_id")),
                start_frame=_safe_int(step_attrs.get("start_frame")),
                end_frame=_safe_int(step_attrs.get("end_frame")),
                duration_s=_safe_float(step_attrs.get("duration_s")),
                attrs=step_attrs,
                per_fish=_filter_mapping_by_step(step_per_fish, int(step_index)),
                per_bout=_filter_mapping_by_step(step_per_bout, int(step_index)),
                grating_per_fish=_filter_mapping_by_step(grating_per_fish, int(step_index)),
                moving_grating_omr=moving_omr,
                concentric_per_fish=_filter_mapping_by_step(concentric_per_fish, int(step_index)),
                concentric_radial_omr=radial_omr,
                looming=looming,
            )
        )

    return StimulusResponseTables(
        layout=layout,
        attrs=attrs,
        global_per_fish=_read_columnar_mapping(run_group, "global_per_fish"),
        global_omr_per_fish=_read_columnar_mapping(run_group, "global_omr_per_fish"),
        frame_annotations=_read_columnar_mapping(run_group, "frame_annotations"),
        steps=tuple(sorted(steps, key=lambda step: step.step_index)),
    )


def resolve_stimulus_response_tables(run_group: zarr.Group) -> StimulusResponseTables:
    """Return logical stimulus-response tables for a run group.

    Current production runs use the hierarchical-v1 physical layout. Future
    compact-tabular-v2 runs should be adapted here so downstream consumers do
    not need physical-layout branches.
    """

    attrs = _attrs_dict(run_group)
    layout = str(attrs.get("layout") or attrs.get("storage_layout") or "hierarchical_v1")
    if layout == "compact_tabular_v2":
        return _resolve_compact_tabular_v2(run_group, layout=layout)
    if layout in {"hierarchical_v1", "legacy", ""} or _has_child(run_group, "steps"):
        return _resolve_hierarchical_v1(run_group, layout=layout or "hierarchical_v1")
    raise ValueError(f"Unsupported stimulus_response layout: {layout}")


def moving_grating_omr_steps(
    run_group: zarr.Group,
) -> tuple[StimulusResponseStepTables, ...]:
    """Return steps that contain moving-grating OMR tables."""

    tables = resolve_stimulus_response_tables(run_group)
    return tuple(step for step in tables.steps if step.moving_grating_omr is not None)


def concentric_radial_omr_steps(
    run_group: zarr.Group,
) -> tuple[StimulusResponseStepTables, ...]:
    """Return steps that contain concentric radial-OMR tables."""

    tables = resolve_stimulus_response_tables(run_group)
    return tuple(step for step in tables.steps if step.concentric_radial_omr is not None)
