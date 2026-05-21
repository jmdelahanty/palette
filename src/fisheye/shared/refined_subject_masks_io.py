"""Logical readers for ``refined_subject_masks_runs``.

Refined subject-mask runs are editable, dense mask artifacts. Unlike most
analysis readers, this module intentionally does not materialize ``masks_roi``
as a NumPy array by default; callers get the array handle and can slice only the
rows/channels they need. Small run metadata, metrics, row lineage, component
QC, and component geometry arrays are exposed as logical tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import zarr

from .zarr_helpers import (
    normalize_zarr_path as _normalize_path,
    read_zarr_array_mapping,
    zarr_attrs_dict as _attrs_dict,
    zarr_child_group as _child_group,
    zarr_group_keys as _group_keys,
)
from .zarr_run_completion import resolve_latest_complete_run_name


REFINED_SUBJECT_MASKS_RUN_PARENT = "refined_subject_masks_runs"
REFINED_SUBJECT_MASKS_ROW_LINEAGE_ARRAYS: tuple[str, ...] = (
    "frame_indices",
    "frame_counts",
    "detection_indices",
    "source_refined_row_ids",
    "source_detect_row_index",
)
REFINED_SUBJECT_MASKS_SMALL_RUN_ARRAYS: tuple[str, ...] = (
    "available_channels",
    "edit_applied",
    "detection_source",
    "reason_bytes",
    "reason",
    *REFINED_SUBJECT_MASKS_ROW_LINEAGE_ARRAYS,
)
REFINED_SUBJECT_MASKS_RUN_METRIC_ARRAYS: tuple[str, ...] = (
    "mask_present",
    "area_px",
    "centroid_xy",
    "centroid_valid",
    "bbox_xyxy",
    "bbox_valid",
)
REFINED_SUBJECT_COMPONENT_SMALL_ARRAYS: tuple[str, ...] = (
    "row_revision",
    "source_row_fingerprint",
    "manual_override",
    "source_row_stale",
    "reason_bytes",
    "reason",
)


class RefinedSubjectMasksIOError(ValueError):
    """Raised when a refined subject-mask run cannot be resolved or loaded."""


@dataclass(frozen=True)
class RefinedSubjectMasksRunOption:
    """One selectable refined subject-mask run."""

    run_name: str
    run_path: str
    label: str
    method: Optional[str]
    label_schema_id: Optional[str]
    mask_labels: tuple[str, ...]
    available_components: tuple[str, ...]
    n_rows: int
    channel_count: int
    roi_shape: tuple[int, int]
    is_latest: bool
    attrs: Mapping[str, Any]


@dataclass(frozen=True)
class RefinedSubjectComponentTables:
    """Small logical tables for one refined subject component."""

    component_name: str
    component_path: str
    component_index: int
    available: bool
    attrs: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]
    metrics: Mapping[str, np.ndarray]
    qc: Mapping[str, np.ndarray]
    geometry: Mapping[str, np.ndarray]
    finalization_metrics: Mapping[str, np.ndarray]
    source_paths: Mapping[str, str]

    def require_array(self, name: str) -> np.ndarray:
        values = self.arrays.get(name)
        if values is None:
            raise RefinedSubjectMasksIOError(f"{self.component_path} is missing required array {name!r}.")
        return values


@dataclass(frozen=True)
class RefinedSubjectRelationTables:
    """Small logical tables for one refined subject-mask relation."""

    relation_name: str
    relation_path: str
    attrs: Mapping[str, Any]
    metrics: Mapping[str, np.ndarray]
    source_paths: Mapping[str, str]


@dataclass(frozen=True)
class RefinedSubjectMasksRunTables:
    """Logical view over one refined subject-mask run."""

    run_name: str
    run_path: str
    attrs: Mapping[str, Any]
    mask_labels: tuple[str, ...]
    label_to_index: Mapping[str, int]
    available_channels: np.ndarray
    masks_roi: Any | None
    run_arrays: Mapping[str, np.ndarray]
    metrics: Mapping[str, np.ndarray]
    components: Mapping[str, RefinedSubjectComponentTables]
    relations: Mapping[str, RefinedSubjectRelationTables]
    source_paths: Mapping[str, str]

    @property
    def n_rows(self) -> int:
        if self.masks_roi is not None:
            return int(self.masks_roi.shape[0])
        for name in REFINED_SUBJECT_MASKS_ROW_LINEAGE_ARRAYS:
            values = self.run_arrays.get(name)
            if values is not None and values.shape:
                return int(values.shape[0])
        for values in self.metrics.values():
            if values.shape:
                return int(values.shape[0])
        for name, values in self.run_arrays.items():
            if name == "available_channels":
                continue
            if values.shape:
                return int(values.shape[0])
        return 0

    @property
    def channel_count(self) -> int:
        if self.masks_roi is not None and len(tuple(self.masks_roi.shape)) >= 2:
            return int(self.masks_roi.shape[1])
        return int(self.available_channels.shape[0])

    def component_index(self, component_name: str) -> int:
        value = self.label_to_index.get(str(component_name))
        if value is None:
            raise RefinedSubjectMasksIOError(
                f"Component {component_name!r} not present in refined run {self.run_name!r}."
            )
        return int(value)

    def component_available(self, component_name: str) -> bool:
        idx = self.component_index(component_name)
        return idx < int(self.available_channels.shape[0]) and bool(self.available_channels[idx])

    def resolve_components(self, components: Optional[Sequence[str]] = None) -> tuple[tuple[str, int], ...]:
        requested = [str(value) for value in components] if components else list(self.mask_labels)
        resolved: list[tuple[str, int]] = []
        seen: set[str] = set()
        for component_name in requested:
            if component_name in seen:
                continue
            idx = self.component_index(component_name)
            if idx < int(self.available_channels.shape[0]) and bool(self.available_channels[idx]):
                resolved.append((component_name, idx))
                seen.add(component_name)
        if not resolved:
            raise RefinedSubjectMasksIOError(
                f"No available refined subject-mask components selected in run {self.run_name!r}."
            )
        return tuple(resolved)

    def require_masks_roi(self) -> Any:
        if self.masks_roi is None:
            raise RefinedSubjectMasksIOError(f"{self.run_path} was loaded without masks_roi.")
        return self.masks_roi


def _mask_labels(group: Any) -> tuple[str, ...]:
    labels_raw = _attrs_dict(group).get("mask_labels")
    if not isinstance(labels_raw, (list, tuple)) or not labels_raw:
        raise RefinedSubjectMasksIOError("Refined subject-mask run is missing usable mask_labels attr.")
    return tuple(str(value) for value in labels_raw)


def _available_channels(group: Any, *, channel_count: int) -> np.ndarray:
    available = None
    try:
        available = group.get("available_channels")
    except Exception:
        available = None
    if available is None:
        return np.ones((int(channel_count),), dtype=bool)
    return np.asarray(available[:], dtype=bool).reshape(-1)


def _run_n_rows(run_group: Any) -> int:
    masks = run_group.get("masks_roi") if hasattr(run_group, "get") else None
    if masks is not None and hasattr(masks, "shape") and len(tuple(masks.shape)) >= 1:
        return int(masks.shape[0])
    for name in REFINED_SUBJECT_MASKS_SMALL_RUN_ARRAYS:
        arr = run_group.get(name) if hasattr(run_group, "get") else None
        if arr is not None and hasattr(arr, "shape") and tuple(arr.shape):
            return int(arr.shape[0])
    return 0


def _roi_shape(run_group: Any) -> tuple[int, int]:
    masks = run_group.get("masks_roi") if hasattr(run_group, "get") else None
    if masks is not None and hasattr(masks, "shape") and len(tuple(masks.shape)) >= 4:
        return (int(masks.shape[2]), int(masks.shape[3]))
    return (0, 0)


def _option_label(
    *,
    run_name: str,
    method: Optional[str],
    label_schema_id: Optional[str],
    available_components: tuple[str, ...],
    n_rows: int,
    is_latest: bool,
) -> str:
    pieces = [run_name]
    if method:
        pieces.append(str(method))
    if label_schema_id:
        pieces.append(str(label_schema_id))
    pieces.append(f"{len(available_components)} available components")
    pieces.append(f"{n_rows} rows")
    if is_latest:
        pieces.append("latest")
    return " | ".join(pieces)


def resolve_refined_subject_masks_run(
    root: zarr.Group,
    run_name: str | None = None,
) -> tuple[zarr.Group, str, str]:
    """Resolve one ``refined_subject_masks_runs`` child from a name, path, or latest."""

    parent = root.get(REFINED_SUBJECT_MASKS_RUN_PARENT)
    if parent is None:
        raise RefinedSubjectMasksIOError("No refined_subject_masks_runs group found.")

    if run_name is None or str(run_name).strip().lower() in {"", "latest"}:
        latest = resolve_latest_complete_run_name(parent, legacy_default=True)
        if latest and latest in parent:
            resolved = latest
        else:
            keys = _group_keys(parent)
            if not keys:
                raise RefinedSubjectMasksIOError("refined_subject_masks_runs has no runs.")
            resolved = keys[-1]
    else:
        normalized = _normalize_path(str(run_name))
        parts = normalized.split("/")
        if normalized.startswith(REFINED_SUBJECT_MASKS_RUN_PARENT + "/") and len(parts) >= 2:
            resolved = parts[1]
        else:
            resolved = parts[-1]

    if not resolved or resolved not in parent:
        raise RefinedSubjectMasksIOError(f"refined_subject_masks_runs/{run_name!s} not found.")
    run_path = f"{REFINED_SUBJECT_MASKS_RUN_PARENT}/{resolved}"
    run_group = parent[resolved]
    if not isinstance(run_group, zarr.Group):
        raise RefinedSubjectMasksIOError(f"{run_path} is not a Zarr group.")
    return run_group, str(resolved), run_path


def load_refined_subject_masks_run_tables(
    root: zarr.Group,
    *,
    run_name: str | None = None,
    component_names: Optional[Sequence[str]] = None,
    include_masks_roi: bool = True,
    include_metrics: bool = True,
    include_components: bool = True,
    include_relations: bool = True,
    run_array_names: Sequence[str] = REFINED_SUBJECT_MASKS_SMALL_RUN_ARRAYS,
    component_array_names: Sequence[str] = REFINED_SUBJECT_COMPONENT_SMALL_ARRAYS,
) -> RefinedSubjectMasksRunTables:
    """Load one refined subject-mask run as logical tables.

    ``masks_roi`` remains an array handle. The small arrays requested by
    ``run_array_names`` and ``component_array_names`` are materialized.
    """

    run_group, resolved_run, run_path = resolve_refined_subject_masks_run(root, run_name)
    labels = _mask_labels(run_group)
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    masks_roi = run_group.get("masks_roi") if include_masks_roi else None
    channel_count = int(masks_roi.shape[1]) if masks_roi is not None and len(tuple(masks_roi.shape)) >= 2 else len(labels)
    available_channels = _available_channels(run_group, channel_count=channel_count)
    source_paths: dict[str, str] = {"run": run_path}

    run_arrays = read_zarr_array_mapping(
        run_group,
        physical_prefix=run_path,
        logical_prefix="run",
        source_paths=source_paths,
        array_names=tuple(name for name in run_array_names if name != "masks_roi"),
    )
    if masks_roi is not None:
        source_paths["masks_roi"] = f"{run_path}/masks_roi"

    metrics = (
        read_zarr_array_mapping(
            _child_group(run_group, "metrics"),
            physical_prefix=f"{run_path}/metrics",
            logical_prefix="metrics",
            source_paths=source_paths,
            array_names=REFINED_SUBJECT_MASKS_RUN_METRIC_ARRAYS,
        )
        if include_metrics
        else {}
    )

    requested_components = tuple(str(value) for value in component_names) if component_names is not None else labels
    components_parent = _child_group(run_group, "components")
    components: dict[str, RefinedSubjectComponentTables] = {}
    if include_components:
        for component_name in requested_components:
            component_index = label_to_index.get(component_name)
            if component_index is None:
                if component_names is not None:
                    raise RefinedSubjectMasksIOError(
                        f"Component {component_name!r} not present in refined run {resolved_run!r}."
                    )
                continue
            component_group = _child_group(components_parent, component_name)
            if component_group is None:
                if component_names is not None:
                    raise RefinedSubjectMasksIOError(f"{run_path}/components/{component_name} not found.")
                continue
            component_path = f"{run_path}/components/{component_name}"
            component_source_paths: dict[str, str] = {}
            arrays = read_zarr_array_mapping(
                component_group,
                physical_prefix=component_path,
                logical_prefix=f"components/{component_name}",
                source_paths=source_paths,
                array_names=component_array_names,
            )
            component_metrics = read_zarr_array_mapping(
                _child_group(component_group, "metrics"),
                physical_prefix=f"{component_path}/metrics",
                logical_prefix=f"components/{component_name}/metrics",
                source_paths=source_paths,
            )
            qc = read_zarr_array_mapping(
                _child_group(component_group, "qc"),
                physical_prefix=f"{component_path}/qc",
                logical_prefix=f"components/{component_name}/qc",
                source_paths=source_paths,
            )
            geometry = read_zarr_array_mapping(
                _child_group(component_group, "geometry"),
                physical_prefix=f"{component_path}/geometry",
                logical_prefix=f"components/{component_name}/geometry",
                source_paths=source_paths,
            )
            finalization_metrics = read_zarr_array_mapping(
                _child_group(component_group, "finalization_metrics"),
                physical_prefix=f"{component_path}/finalization_metrics",
                logical_prefix=f"components/{component_name}/finalization_metrics",
                source_paths=source_paths,
            )
            for logical_path, physical_path in source_paths.items():
                if logical_path.startswith(f"components/{component_name}/"):
                    component_source_paths[
                        logical_path.removeprefix(f"components/{component_name}/")
                    ] = physical_path
            components[component_name] = RefinedSubjectComponentTables(
                component_name=component_name,
                component_path=component_path,
                component_index=int(component_index),
                available=(
                    int(component_index) < int(available_channels.shape[0])
                    and bool(available_channels[int(component_index)])
                ),
                attrs=_attrs_dict(component_group),
                arrays=arrays,
                metrics=component_metrics,
                qc=qc,
                geometry=geometry,
                finalization_metrics=finalization_metrics,
                source_paths=component_source_paths,
            )

    relations_parent = _child_group(run_group, "relations")
    relations: dict[str, RefinedSubjectRelationTables] = {}
    if include_relations:
        for relation_name in _group_keys(relations_parent):
            relation_group = _child_group(relations_parent, relation_name)
            if relation_group is None:
                continue
            relation_path = f"{run_path}/relations/{relation_name}"
            relation_source_paths: dict[str, str] = {}
            relation_metrics = read_zarr_array_mapping(
                _child_group(relation_group, "metrics"),
                physical_prefix=f"{relation_path}/metrics",
                logical_prefix=f"relations/{relation_name}/metrics",
                source_paths=source_paths,
            )
            for logical_path, physical_path in source_paths.items():
                if logical_path.startswith(f"relations/{relation_name}/"):
                    relation_source_paths[
                        logical_path.removeprefix(f"relations/{relation_name}/")
                    ] = physical_path
            relations[relation_name] = RefinedSubjectRelationTables(
                relation_name=relation_name,
                relation_path=relation_path,
                attrs=_attrs_dict(relation_group),
                metrics=relation_metrics,
                source_paths=relation_source_paths,
            )

    if masks_roi is None and not run_arrays and not metrics:
        raise RefinedSubjectMasksIOError(f"Refined subject-mask run {resolved_run!r} has no readable arrays.")

    return RefinedSubjectMasksRunTables(
        run_name=resolved_run,
        run_path=run_path,
        attrs=_attrs_dict(run_group),
        mask_labels=labels,
        label_to_index=label_to_index,
        available_channels=available_channels,
        masks_roi=masks_roi,
        run_arrays=run_arrays,
        metrics=metrics,
        components=components,
        relations=relations,
        source_paths=source_paths,
    )


def discover_refined_subject_masks_run_options(root: zarr.Group) -> list[RefinedSubjectMasksRunOption]:
    """Return available refined subject-mask runs from an open Zarr root."""

    parent = root.get(REFINED_SUBJECT_MASKS_RUN_PARENT)
    if parent is None:
        return []
    latest = resolve_latest_complete_run_name(parent, legacy_default=True)
    options: list[RefinedSubjectMasksRunOption] = []
    for run_name in _group_keys(parent):
        try:
            run_group = parent[run_name]
        except Exception:
            continue
        if not isinstance(run_group, zarr.Group):
            continue
        try:
            labels = _mask_labels(run_group)
        except RefinedSubjectMasksIOError:
            continue
        masks = run_group.get("masks_roi")
        channel_count = int(masks.shape[1]) if masks is not None and len(tuple(masks.shape)) >= 2 else len(labels)
        available = _available_channels(run_group, channel_count=channel_count)
        available_components = tuple(
            label for idx, label in enumerate(labels) if idx < int(available.shape[0]) and bool(available[idx])
        )
        attrs = _attrs_dict(run_group)
        n_rows = _run_n_rows(run_group)
        if n_rows <= 0:
            continue
        method = attrs.get("method")
        label_schema_id = attrs.get("label_schema_id")
        run_path = f"{REFINED_SUBJECT_MASKS_RUN_PARENT}/{run_name}"
        is_latest = str(latest) == str(run_name)
        options.append(
            RefinedSubjectMasksRunOption(
                run_name=str(run_name),
                run_path=run_path,
                label=_option_label(
                    run_name=str(run_name),
                    method=str(method) if method is not None else None,
                    label_schema_id=str(label_schema_id) if label_schema_id is not None else None,
                    available_components=available_components,
                    n_rows=n_rows,
                    is_latest=is_latest,
                ),
                method=str(method) if method is not None else None,
                label_schema_id=str(label_schema_id) if label_schema_id is not None else None,
                mask_labels=labels,
                available_components=available_components,
                n_rows=n_rows,
                channel_count=channel_count,
                roi_shape=_roi_shape(run_group),
                is_latest=is_latest,
                attrs=attrs,
            )
        )
    return sorted(options, key=lambda item: (not item.is_latest, item.run_name))


__all__ = [
    "REFINED_SUBJECT_COMPONENT_SMALL_ARRAYS",
    "REFINED_SUBJECT_MASKS_ROW_LINEAGE_ARRAYS",
    "REFINED_SUBJECT_MASKS_RUN_METRIC_ARRAYS",
    "REFINED_SUBJECT_MASKS_RUN_PARENT",
    "REFINED_SUBJECT_MASKS_SMALL_RUN_ARRAYS",
    "RefinedSubjectComponentTables",
    "RefinedSubjectMasksIOError",
    "RefinedSubjectMasksRunOption",
    "RefinedSubjectMasksRunTables",
    "RefinedSubjectRelationTables",
    "discover_refined_subject_masks_run_options",
    "load_refined_subject_masks_run_tables",
    "resolve_refined_subject_masks_run",
]
