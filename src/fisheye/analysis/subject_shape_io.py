"""Logical readers for ``analysis/subject_shape_runs``.

Subject-shape runs currently store geometry under component-specific groups
such as ``components/subject_body`` plus run-level ``body_frame`` and
``relations`` groups. This module is the read boundary for those physical
paths so consumers can request logical component/body-frame tables before any
future compact layout work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import zarr

from ..shared.zarr_helpers import (
    first_array_length as _shared_first_array_length,
    first_array_length_in_group as _first_array_length_in_group,
    normalize_zarr_path as _normalize_path,
    read_zarr_array_mapping,
    safe_int as _safe_int,
    zarr_attrs_dict as _attrs_dict,
    zarr_child_group as _child_group,
    zarr_group_keys as _group_keys,
)


SUBJECT_SHAPE_RUN_PARENT = "analysis/subject_shape_runs"

SUBJECT_SHAPE_ROW_COUNT_COLUMNS: tuple[str, ...] = (
    "tail_sample_valid",
    "bspline_valid",
    "centerline_valid",
    "mask_present",
    "area_px",
    "centroid_valid",
)
BODY_FRAME_ROW_COUNT_COLUMNS: tuple[str, ...] = (
    "valid",
    "origin_xy",
    "forward_axis_xy",
    "heading_deg",
)


class SubjectShapeIOError(ValueError):
    """Raised when a subject-shape run cannot be resolved or loaded."""


@dataclass(frozen=True)
class SubjectShapeRunOption:
    """One selectable subject-shape run."""

    run_name: str
    run_path: str
    label: str
    schema_version: Optional[int]
    method: Optional[str]
    row_axis: Optional[str]
    component_names: tuple[str, ...]
    relation_names: tuple[str, ...]
    n_rows: int
    is_latest: bool
    attrs: Mapping[str, Any]


@dataclass(frozen=True)
class SubjectShapeComponentTables:
    """Logical arrays for one subject-shape component."""

    component_name: str
    component_path: str
    attrs: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]
    source_paths: Mapping[str, str]

    def require_array(self, name: str) -> np.ndarray:
        values = self.arrays.get(name)
        if values is None:
            raise SubjectShapeIOError(f"{self.component_path} is missing required array {name!r}.")
        return values


@dataclass(frozen=True)
class SubjectShapeRelationTables:
    """Logical arrays for one subject-shape relation."""

    relation_name: str
    relation_path: str
    attrs: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]
    source_paths: Mapping[str, str]

    def require_array(self, name: str) -> np.ndarray:
        values = self.arrays.get(name)
        if values is None:
            raise SubjectShapeIOError(f"{self.relation_path} is missing required array {name!r}.")
        return values


@dataclass(frozen=True)
class SubjectShapeRunTables:
    """Logical view over one subject-shape run."""

    run_name: str
    run_path: str
    attrs: Mapping[str, Any]
    components: Mapping[str, SubjectShapeComponentTables]
    relations: Mapping[str, SubjectShapeRelationTables]
    body_frame: Mapping[str, np.ndarray]
    body_frame_attrs: Mapping[str, Any]
    row_index: Mapping[str, np.ndarray]
    source_refined_subject_masks: Mapping[str, np.ndarray]
    source_paths: Mapping[str, str]

    @property
    def schema_version(self) -> int:
        return int(self.attrs.get("schema_version", 0) or 0)

    @property
    def row_axis(self) -> Optional[str]:
        value = self.attrs.get("row_axis")
        return str(value) if value is not None else None

    @property
    def component_names(self) -> tuple[str, ...]:
        return tuple(self.components.keys())

    @property
    def relation_names(self) -> tuple[str, ...]:
        return tuple(self.relations.keys())

    def require_component(self, name: str) -> SubjectShapeComponentTables:
        component = self.components.get(str(name))
        if component is None:
            available = ", ".join(self.components.keys()) or "<none>"
            raise SubjectShapeIOError(
                f"Subject-shape run {self.run_name!r} is missing component {name!r}; "
                f"available components: {available}."
            )
        return component

    def require_body_frame_array(self, name: str) -> np.ndarray:
        values = self.body_frame.get(name)
        if values is None:
            raise SubjectShapeIOError(f"{self.run_path}/body_frame is missing required array {name!r}.")
        return values


def first_array_length(
    arrays: Mapping[str, np.ndarray],
    names: tuple[str, ...] = SUBJECT_SHAPE_ROW_COUNT_COLUMNS,
) -> int:
    """Return the first non-scalar array length found among candidate names."""

    return _shared_first_array_length(arrays, names)


def _run_component_names(run_group: Any) -> tuple[str, ...]:
    attrs = _attrs_dict(run_group)
    attr_names = attrs.get("component_names")
    if isinstance(attr_names, (list, tuple)):
        names = tuple(str(value) for value in attr_names)
        if names:
            return names
    components_group = _child_group(run_group, "components")
    return tuple(_group_keys(components_group)) if components_group is not None else ()


def _run_relation_names(run_group: Any) -> tuple[str, ...]:
    attrs = _attrs_dict(run_group)
    attr_names = attrs.get("relation_names")
    if isinstance(attr_names, (list, tuple)):
        names = tuple(str(value) for value in attr_names)
        if names:
            return names
    relations_group = _child_group(run_group, "relations")
    return tuple(_group_keys(relations_group)) if relations_group is not None else ()


def _subject_shape_option_label(
    *,
    run_name: str,
    schema_version: Optional[int],
    method: Optional[str],
    component_names: tuple[str, ...],
    n_rows: int,
    is_latest: bool,
) -> str:
    pieces = [run_name]
    if schema_version is not None:
        pieces.append(f"schema v{schema_version}")
    if method:
        pieces.append(str(method))
    if component_names:
        pieces.append(f"{len(component_names)} components")
    pieces.append(f"{n_rows} rows")
    if is_latest:
        pieces.append("latest")
    return " | ".join(pieces)


def resolve_subject_shape_run(
    root: zarr.Group,
    run_name: str | None = None,
) -> tuple[zarr.Group, str, str]:
    """Resolve one ``analysis/subject_shape_runs`` child from a name, path, or latest."""

    parent = root.get(SUBJECT_SHAPE_RUN_PARENT)
    if parent is None:
        raise SubjectShapeIOError("No analysis/subject_shape_runs group found.")

    if run_name is None or str(run_name).strip().lower() in {"", "latest"}:
        latest = parent.attrs.get("latest")
        if isinstance(latest, str) and latest in parent:
            resolved = latest
        else:
            raise SubjectShapeIOError("No latest subject-shape run is recorded.")
    else:
        normalized = _normalize_path(str(run_name))
        parts = normalized.split("/")
        if normalized.startswith(SUBJECT_SHAPE_RUN_PARENT + "/") and len(parts) >= 3:
            resolved = parts[2]
        else:
            resolved = parts[-1]

    if not resolved or resolved not in parent:
        raise SubjectShapeIOError(f"Subject-shape run {run_name!r} not found in analysis/subject_shape_runs.")
    run_path = f"{SUBJECT_SHAPE_RUN_PARENT}/{resolved}"
    run_group = parent[resolved]
    if not isinstance(run_group, zarr.Group):
        raise SubjectShapeIOError(f"{run_path} is not a Zarr group.")
    return run_group, str(resolved), run_path


def load_subject_shape_run_tables(
    root: zarr.Group,
    *,
    run_name: str | None = None,
    component_names: Optional[Sequence[str]] = None,
    relation_names: Optional[Sequence[str]] = None,
    component_array_names: Optional[Mapping[str, Sequence[str]]] = None,
    relation_array_names: Optional[Mapping[str, Sequence[str]]] = None,
    include_body_frame: bool = True,
    include_row_index: bool = True,
    include_source_refined_subject_masks: bool = True,
) -> SubjectShapeRunTables:
    """Load logical component, relation, body-frame, and lineage arrays."""

    run_group, resolved_run, run_path = resolve_subject_shape_run(root, run_name)
    source_paths: dict[str, str] = {"run": run_path}
    requested_components = tuple(str(value) for value in component_names) if component_names is not None else None
    requested_relations = tuple(str(value) for value in relation_names) if relation_names is not None else None

    components_group = _child_group(run_group, "components")
    component_tables: dict[str, SubjectShapeComponentTables] = {}
    component_iter = requested_components if requested_components is not None else _run_component_names(run_group)
    for component_name in component_iter:
        component_group = _child_group(components_group, component_name)
        if component_group is None:
            if requested_components is not None:
                raise SubjectShapeIOError(f"{run_path}/components/{component_name} not found.")
            continue
        component_path = f"{run_path}/components/{component_name}"
        component_source_paths: dict[str, str] = {}
        arrays = read_zarr_array_mapping(
            component_group,
            physical_prefix=component_path,
            logical_prefix=f"components/{component_name}",
            source_paths=source_paths,
            array_names=(
                component_array_names.get(str(component_name)) if component_array_names is not None else None
            ),
        )
        for logical_path, physical_path in source_paths.items():
            if logical_path.startswith(f"components/{component_name}/"):
                component_source_paths[logical_path.rsplit("/", 1)[-1]] = physical_path
        component_tables[str(component_name)] = SubjectShapeComponentTables(
            component_name=str(component_name),
            component_path=component_path,
            attrs=_attrs_dict(component_group),
            arrays=arrays,
            source_paths=component_source_paths,
        )

    relations_group = _child_group(run_group, "relations")
    relation_tables: dict[str, SubjectShapeRelationTables] = {}
    relation_iter = requested_relations if requested_relations is not None else _run_relation_names(run_group)
    for relation_name in relation_iter:
        relation_group = _child_group(relations_group, relation_name)
        if relation_group is None:
            if requested_relations is not None:
                raise SubjectShapeIOError(f"{run_path}/relations/{relation_name} not found.")
            continue
        relation_path = f"{run_path}/relations/{relation_name}"
        relation_source_paths: dict[str, str] = {}
        arrays = read_zarr_array_mapping(
            relation_group,
            physical_prefix=relation_path,
            logical_prefix=f"relations/{relation_name}",
            source_paths=source_paths,
            array_names=relation_array_names.get(str(relation_name)) if relation_array_names is not None else None,
        )
        for logical_path, physical_path in source_paths.items():
            if logical_path.startswith(f"relations/{relation_name}/"):
                relation_source_paths[logical_path.rsplit("/", 1)[-1]] = physical_path
        relation_tables[str(relation_name)] = SubjectShapeRelationTables(
            relation_name=str(relation_name),
            relation_path=relation_path,
            attrs=_attrs_dict(relation_group),
            arrays=arrays,
            source_paths=relation_source_paths,
        )

    body_frame_group = _child_group(run_group, "body_frame") if include_body_frame else None
    body_frame = read_zarr_array_mapping(
        body_frame_group,
        physical_prefix=f"{run_path}/body_frame",
        logical_prefix="body_frame",
        source_paths=source_paths,
    )

    row_index_group = _child_group(run_group, "row_index") if include_row_index else None
    row_index = read_zarr_array_mapping(
        row_index_group,
        physical_prefix=f"{run_path}/row_index",
        logical_prefix="row_index",
        source_paths=source_paths,
    )

    source_revision_group = (
        _child_group(run_group, "source_refined_subject_masks") if include_source_refined_subject_masks else None
    )
    source_refined_subject_masks = read_zarr_array_mapping(
        source_revision_group,
        physical_prefix=f"{run_path}/source_refined_subject_masks",
        logical_prefix="source_refined_subject_masks",
        source_paths=source_paths,
    )

    if not component_tables and not body_frame and not relation_tables:
        raise SubjectShapeIOError(f"Subject-shape run {resolved_run!r} has no readable component/body-frame arrays.")

    return SubjectShapeRunTables(
        run_name=resolved_run,
        run_path=run_path,
        attrs=_attrs_dict(run_group),
        components=component_tables,
        relations=relation_tables,
        body_frame=body_frame,
        body_frame_attrs=_attrs_dict(body_frame_group) if body_frame_group is not None else {},
        row_index=row_index,
        source_refined_subject_masks=source_refined_subject_masks,
        source_paths=source_paths,
    )


def discover_subject_shape_run_options(root: zarr.Group) -> list[SubjectShapeRunOption]:
    """Return available subject-shape analysis runs from an open Zarr root."""

    parent = root.get(SUBJECT_SHAPE_RUN_PARENT)
    if parent is None:
        return []

    latest = parent.attrs.get("latest")
    options: list[SubjectShapeRunOption] = []
    for run_name in _group_keys(parent):
        try:
            run_group = parent[run_name]
        except Exception:
            continue
        if not isinstance(run_group, zarr.Group):
            continue
        attrs = _attrs_dict(run_group)
        component_names = _run_component_names(run_group)
        relation_names = _run_relation_names(run_group)
        body_frame_group = _child_group(run_group, "body_frame")
        n_rows = _first_array_length_in_group(body_frame_group, BODY_FRAME_ROW_COUNT_COLUMNS)
        if n_rows <= 0 and component_names:
            component_group = _child_group(run_group, f"components/{component_names[0]}")
            n_rows = _first_array_length_in_group(component_group, SUBJECT_SHAPE_ROW_COUNT_COLUMNS)
        if n_rows <= 0:
            continue
        schema_version = _safe_int(attrs.get("schema_version"))
        method = attrs.get("method")
        row_axis = attrs.get("row_axis")
        method_str = str(method) if method is not None else None
        row_axis_str = str(row_axis) if row_axis is not None else None
        is_latest = str(latest) == str(run_name)
        run_path = f"{SUBJECT_SHAPE_RUN_PARENT}/{run_name}"
        options.append(
            SubjectShapeRunOption(
                run_name=str(run_name),
                run_path=run_path,
                label=_subject_shape_option_label(
                    run_name=str(run_name),
                    schema_version=schema_version,
                    method=method_str,
                    component_names=component_names,
                    n_rows=n_rows,
                    is_latest=is_latest,
                ),
                schema_version=schema_version,
                method=method_str,
                row_axis=row_axis_str,
                component_names=component_names,
                relation_names=relation_names,
                n_rows=n_rows,
                is_latest=is_latest,
                attrs=attrs,
            )
        )

    return sorted(options, key=lambda item: (not item.is_latest, item.run_name))


def load_subject_body_component(
    root: zarr.Group,
    *,
    run_name: str | None = None,
    include_body_frame: bool = False,
    array_names: Optional[Sequence[str]] = None,
) -> tuple[SubjectShapeRunTables, SubjectShapeComponentTables]:
    """Load the subject-body component from one subject-shape run."""

    tables = load_subject_shape_run_tables(
        root,
        run_name=run_name,
        component_names=("subject_body",),
        relation_names=(),
        component_array_names={"subject_body": tuple(array_names)} if array_names is not None else None,
        include_body_frame=include_body_frame,
        include_row_index=True,
        include_source_refined_subject_masks=True,
    )
    return tables, tables.require_component("subject_body")


__all__ = [
    "BODY_FRAME_ROW_COUNT_COLUMNS",
    "SUBJECT_SHAPE_ROW_COUNT_COLUMNS",
    "SUBJECT_SHAPE_RUN_PARENT",
    "SubjectShapeComponentTables",
    "SubjectShapeIOError",
    "SubjectShapeRelationTables",
    "SubjectShapeRunOption",
    "SubjectShapeRunTables",
    "discover_subject_shape_run_options",
    "first_array_length",
    "load_subject_body_component",
    "load_subject_shape_run_tables",
    "resolve_subject_shape_run",
]
