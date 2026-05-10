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


def resolve_stimulus_response_tables(run_group: zarr.Group) -> StimulusResponseTables:
    """Return logical stimulus-response tables for a run group.

    Current production runs use the hierarchical-v1 physical layout. Future
    compact-tabular-v2 runs should be adapted here so downstream consumers do
    not need physical-layout branches.
    """

    attrs = _attrs_dict(run_group)
    layout = str(attrs.get("layout") or attrs.get("storage_layout") or "hierarchical_v1")
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
