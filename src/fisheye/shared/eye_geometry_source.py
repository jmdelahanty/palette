"""Resolve eye-geometry arrays across subject-shape and compatibility sources."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import zarr

from .provenance_attrs import resolve_source_keypoints_run
from .refined_subject_eye_geometry import EYE_COMPONENTS
from .zarr_run_completion import resolve_latest_complete_run_name


EYE_GEOMETRY_STAGE_REFINED_SUBJECT = "refined_subject_masks_runs"
EYE_GEOMETRY_STAGE_REFINED_EYE = "refined_eye_masks_runs"
EYE_GEOMETRY_STAGE_SUBJECT_SHAPE = "analysis/subject_shape_runs"


def _normalize_text(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _group_get(group: Any, path: str) -> Any:
    try:
        value = group.get(path)
    except Exception:
        value = None
    if value is not None:
        return value
    try:
        return group[path]
    except Exception:
        return None


def _contains(group: Any, path: str) -> bool:
    return _group_get(group, path) is not None


def _group_names(parent: Any) -> list[str]:
    if parent is None:
        return []
    try:
        names = parent.group_keys()
    except Exception:
        try:
            names = parent.keys()
        except Exception:
            return []
    return sorted(str(name) for name in names)


def _resolve_run_group(
    root: zarr.Group,
    parent_name: str,
    run_name: Optional[str],
    *,
    fallback_to_latest: bool,
) -> tuple[Optional[str], Optional[zarr.Group]]:
    parent = _group_get(root, parent_name)
    if parent is None:
        return None, None

    requested = _normalize_text(run_name)
    if requested == "latest":
        requested = None
    if requested:
        group = _group_get(parent, requested)
        return (requested, group) if group is not None else (requested, None)

    if fallback_to_latest:
        latest = _normalize_text(resolve_latest_complete_run_name(parent, legacy_default=True))
        if latest:
            group = _group_get(parent, latest)
            if group is not None:
                return latest, group

    names = _group_names(parent)
    if names:
        name = names[-1]
        return name, _group_get(parent, name)
    return None, None


def _label_index_map(group: zarr.Group) -> dict[str, int]:
    labels_raw = group.attrs.get("mask_labels")
    if not isinstance(labels_raw, (list, tuple)):
        return {}
    return {str(label): idx for idx, label in enumerate(labels_raw)}


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(int(v) for v in getattr(value, "shape", ()))


def _select_channel_indices(channel_key: object, channel_count: int) -> list[int]:
    if isinstance(channel_key, (int, np.integer)):
        idx = int(channel_key)
        if idx < 0:
            idx += channel_count
        return [idx]
    if isinstance(channel_key, slice):
        return list(range(channel_count))[channel_key]
    if isinstance(channel_key, np.ndarray):
        return [int(v) for v in channel_key.reshape(-1).tolist()]
    if isinstance(channel_key, Sequence) and not isinstance(channel_key, (str, bytes, bytearray)):
        return [int(v) for v in channel_key]
    raise TypeError(f"Unsupported channel index type: {type(channel_key).__name__}")


def _is_scalar_row_key(row_key: object) -> bool:
    return isinstance(row_key, (int, np.integer))


def _stack_component_values(values: list[np.ndarray], *, row_scalar: bool) -> np.ndarray:
    if not values:
        return np.asarray(values)
    axis = 0 if row_scalar else 1
    return np.stack(values, axis=axis)


class StackedComponentArray:
    """Array-like view that exposes split component arrays as ``(N, C, ...)``."""

    def __init__(self, components: Sequence[Any]):
        if not components:
            raise ValueError("At least one component array is required.")
        shapes = [_shape(component) for component in components]
        if any(shape != shapes[0] for shape in shapes):
            raise ValueError(f"Component shapes do not match: {shapes}")
        self._components = tuple(components)
        self._component_shape = shapes[0]
        self.shape = (
            int(self._component_shape[0]),
            len(self._components),
            *tuple(int(v) for v in self._component_shape[1:]),
        )
        self.ndim = len(self.shape)
        self.dtype = getattr(self._components[0], "dtype", np.asarray(self._components[0][:]).dtype)
        chunks = getattr(self._components[0], "chunks", None)
        self.chunks = None if chunks is None else (chunks[0], len(self._components), *tuple(chunks[1:]))

    def __array__(self, dtype=None) -> np.ndarray:
        values = np.asarray(self[:])
        return values.astype(dtype, copy=False) if dtype is not None else values

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            values = [np.asarray(component[key]) for component in self._components]
            return _stack_component_values(values, row_scalar=_is_scalar_row_key(key))

        if not key:
            return self[:]
        row_key = key[0]
        if len(key) == 1:
            values = [np.asarray(component[row_key]) for component in self._components]
            return _stack_component_values(values, row_scalar=_is_scalar_row_key(row_key))

        channel_key = key[1]
        tail_key = tuple(key[2:])
        channel_indices = _select_channel_indices(channel_key, len(self._components))
        component_key = (row_key, *tail_key)
        values = [np.asarray(self._components[idx][component_key]) for idx in channel_indices]
        if isinstance(channel_key, (int, np.integer)):
            return np.asarray(values[0])
        return _stack_component_values(values, row_scalar=_is_scalar_row_key(row_key))


class ChannelSelectionArray:
    """Array-like view selecting semantic component channels from ``masks_roi``."""

    def __init__(self, source: Any, channel_indices: Sequence[int]):
        if not channel_indices:
            raise ValueError("At least one channel index is required.")
        self._source = source
        self._channel_indices = tuple(int(v) for v in channel_indices)
        source_shape = _shape(source)
        if len(source_shape) < 2:
            raise ValueError(f"Source array must include a channel axis, got {source_shape}.")
        self._component_shape = (int(source_shape[0]), *tuple(int(v) for v in source_shape[2:]))
        self.shape = (int(source_shape[0]), len(self._channel_indices), *tuple(int(v) for v in source_shape[2:]))
        self.ndim = len(self.shape)
        self.dtype = getattr(source, "dtype", np.asarray(source[:1]).dtype)
        chunks = getattr(source, "chunks", None)
        self.chunks = None if chunks is None else (chunks[0], len(self._channel_indices), *tuple(chunks[2:]))

    def __array__(self, dtype=None) -> np.ndarray:
        values = np.asarray(self[:])
        return values.astype(dtype, copy=False) if dtype is not None else values

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            values = [np.asarray(self._source[(key, idx)]) for idx in self._channel_indices]
            return _stack_component_values(values, row_scalar=_is_scalar_row_key(key))

        if not key:
            return self[:]
        row_key = key[0]
        if len(key) == 1:
            values = [np.asarray(self._source[(row_key, idx)]) for idx in self._channel_indices]
            return _stack_component_values(values, row_scalar=_is_scalar_row_key(row_key))

        channel_key = key[1]
        tail_key = tuple(key[2:])
        selected = _select_channel_indices(channel_key, len(self._channel_indices))
        source_key_tail = (row_key, *tail_key)
        values = [
            np.asarray(self._source[(source_key_tail[0], self._channel_indices[idx], *source_key_tail[1:])])
            for idx in selected
        ]
        if isinstance(channel_key, (int, np.integer)):
            return np.asarray(values[0])
        return _stack_component_values(values, row_scalar=_is_scalar_row_key(row_key))


@dataclass
class EyeGeometrySource:
    stage_group: str
    run_name: str
    group_path: str
    group: zarr.Group
    masks_roi: Optional[Any]
    ellipse_params: Any
    ellipse_success: Any
    eye_separation: Optional[Any]
    lineage_attrs: Mapping[str, object]
    source_refined_eye_run: Optional[str] = None
    source_refined_subject_run: Optional[str] = None
    source_subject_shape_run: Optional[str] = None


def _has_subject_eye_geometry(group: zarr.Group) -> bool:
    label_map = _label_index_map(group)
    if any(component not in label_map for component in EYE_COMPONENTS):
        return False
    masks = _group_get(group, "masks_roi")
    if masks is None or len(_shape(masks)) != 4:
        return False
    channel_count = int(_shape(masks)[1])
    if any(int(label_map[component]) >= channel_count for component in EYE_COMPONENTS):
        return False
    for component in EYE_COMPONENTS:
        if not _contains(group, f"components/{component}/geometry/ellipse_params"):
            return False
        if not _contains(group, f"components/{component}/geometry/ellipse_success"):
            return False
    return _contains(group, "relations/eye_pair/metrics/separation_px")


def _has_subject_shape_eye_geometry(group: zarr.Group) -> bool:
    arrays: list[Any] = []
    for component in EYE_COMPONENTS:
        params = _group_get(group, f"components/{component}/ellipse_params")
        success = _group_get(group, f"components/{component}/ellipse_success")
        if params is None or success is None:
            return False
        param_shape = _shape(params)
        success_shape = _shape(success)
        if len(param_shape) < 2 or len(success_shape) != 1 or param_shape[0] != success_shape[0]:
            return False
        arrays.extend([params, success])
    if _shape(arrays[0]) != _shape(arrays[2]) or _shape(arrays[1]) != _shape(arrays[3]):
        return False
    separation = _group_get(group, "relations/eye_pair/separation_px")
    return separation is not None and _shape(separation)[:1] == _shape(arrays[1])[:1]


def _find_subject_run_for_refined_eye(root: zarr.Group, refined_eye_run: str) -> tuple[Optional[str], Optional[zarr.Group]]:
    parent = _group_get(root, EYE_GEOMETRY_STAGE_REFINED_SUBJECT)
    if parent is None:
        return None, None

    candidates: list[tuple[int, str, zarr.Group]] = []
    for idx, name in enumerate(_group_names(parent)):
        group = _group_get(parent, name)
        if group is None or not _has_subject_eye_geometry(group):
            continue
        attrs = group.attrs
        score = 0
        if _normalize_text(attrs.get("compat_refined_eye_masks_run")) == refined_eye_run:
            score = 3
        elif _normalize_text(attrs.get("source_refined_eye_masks_run")) == refined_eye_run:
            score = 2
        else:
            for component in EYE_COMPONENTS:
                provenance = _group_get(group, f"components/{component}/provenance")
                if provenance is not None and _normalize_text(provenance.attrs.get("source_run")) == refined_eye_run:
                    score = max(score, 1)
        if score:
            candidates.append((score, name, group))

    if not candidates:
        return None, None
    candidates.sort(key=lambda item: (item[0], item[1]))
    _score, name, group = candidates[-1]
    return name, group


def _find_latest_subject_eye_geometry(root: zarr.Group) -> tuple[Optional[str], Optional[zarr.Group]]:
    parent = _group_get(root, EYE_GEOMETRY_STAGE_REFINED_SUBJECT)
    if parent is None:
        return None, None

    latest = _normalize_text(resolve_latest_complete_run_name(parent, legacy_default=True))
    if latest:
        latest_group = _group_get(parent, latest)
        if latest_group is not None and _has_subject_eye_geometry(latest_group):
            return latest, latest_group

    for name in reversed(_group_names(parent)):
        group = _group_get(parent, name)
        if group is not None and _has_subject_eye_geometry(group):
            return name, group
    return None, None


def _find_latest_subject_shape_eye_geometry(root: zarr.Group) -> tuple[Optional[str], Optional[zarr.Group]]:
    parent = _group_get(root, EYE_GEOMETRY_STAGE_SUBJECT_SHAPE)
    if parent is None:
        return None, None

    latest = _normalize_text(resolve_latest_complete_run_name(parent, legacy_default=True))
    if latest:
        latest_group = _group_get(parent, latest)
        if latest_group is not None and _has_subject_shape_eye_geometry(latest_group):
            return latest, latest_group

    for name in reversed(_group_names(parent)):
        group = _group_get(parent, name)
        if group is not None and _has_subject_shape_eye_geometry(group):
            return name, group
    return None, None


def _merge_lineage_attrs(primary: zarr.Group, fallback: Optional[zarr.Group] = None) -> dict[str, object]:
    attrs = dict(primary.attrs)
    if fallback is not None:
        fallback_attrs = dict(fallback.attrs)
        for key in ("source_keypoints_run", "source_keypoint_run", "source_keypoint_group"):
            if attrs.get(key) is None and fallback_attrs.get(key) is not None:
                attrs[key] = fallback_attrs[key]
    return attrs


def _build_subject_source(
    run_name: str,
    group: zarr.Group,
    *,
    refined_eye_group: Optional[zarr.Group] = None,
    source_refined_eye_run: Optional[str] = None,
) -> EyeGeometrySource:
    if not _has_subject_eye_geometry(group):
        raise ValueError(f"{EYE_GEOMETRY_STAGE_REFINED_SUBJECT}/{run_name} missing canonical eye geometry.")
    label_map = _label_index_map(group)
    masks = ChannelSelectionArray(group["masks_roi"], [label_map["eye_left"], label_map["eye_right"]])
    ellipse_params = StackedComponentArray(
        [
            group["components/eye_left/geometry/ellipse_params"],
            group["components/eye_right/geometry/ellipse_params"],
        ]
    )
    ellipse_success = StackedComponentArray(
        [
            group["components/eye_left/geometry/ellipse_success"],
            group["components/eye_right/geometry/ellipse_success"],
        ]
    )
    source_refined_eye_run = source_refined_eye_run or _normalize_text(group.attrs.get("source_refined_eye_masks_run"))
    return EyeGeometrySource(
        stage_group=EYE_GEOMETRY_STAGE_REFINED_SUBJECT,
        run_name=run_name,
        group_path=f"{EYE_GEOMETRY_STAGE_REFINED_SUBJECT}/{run_name}",
        group=group,
        masks_roi=masks,
        ellipse_params=ellipse_params,
        ellipse_success=ellipse_success,
        eye_separation=group["relations/eye_pair/metrics/separation_px"],
        lineage_attrs=_merge_lineage_attrs(group, refined_eye_group),
        source_refined_eye_run=source_refined_eye_run,
        source_refined_subject_run=run_name,
    )


def _build_subject_shape_source(run_name: str, group: zarr.Group) -> EyeGeometrySource:
    if not _has_subject_shape_eye_geometry(group):
        raise ValueError(f"{EYE_GEOMETRY_STAGE_SUBJECT_SHAPE}/{run_name} missing analysis eye geometry.")
    ellipse_params = StackedComponentArray(
        [
            group["components/eye_left/ellipse_params"],
            group["components/eye_right/ellipse_params"],
        ]
    )
    ellipse_success = StackedComponentArray(
        [
            group["components/eye_left/ellipse_success"],
            group["components/eye_right/ellipse_success"],
        ]
    )
    return EyeGeometrySource(
        stage_group=EYE_GEOMETRY_STAGE_SUBJECT_SHAPE,
        run_name=run_name,
        group_path=f"{EYE_GEOMETRY_STAGE_SUBJECT_SHAPE}/{run_name}",
        group=group,
        masks_roi=None,
        ellipse_params=ellipse_params,
        ellipse_success=ellipse_success,
        eye_separation=group["relations/eye_pair/separation_px"],
        lineage_attrs=dict(group.attrs),
        source_refined_eye_run=_normalize_text(group.attrs.get("source_refined_eye_masks_run")),
        source_refined_subject_run=_normalize_text(group.attrs.get("source_refined_subject_masks_run")),
        source_subject_shape_run=run_name,
    )


def _build_refined_eye_source(run_name: str, group: zarr.Group) -> EyeGeometrySource:
    for dataset in ("masks_roi", "ellipse_params", "ellipse_success"):
        if dataset not in group:
            raise ValueError(f"{EYE_GEOMETRY_STAGE_REFINED_EYE}/{run_name} missing {dataset}.")
    return EyeGeometrySource(
        stage_group=EYE_GEOMETRY_STAGE_REFINED_EYE,
        run_name=run_name,
        group_path=f"{EYE_GEOMETRY_STAGE_REFINED_EYE}/{run_name}",
        group=group,
        masks_roi=group["masks_roi"],
        ellipse_params=group["ellipse_params"],
        ellipse_success=group["ellipse_success"],
        eye_separation=group.get("eye_separation"),
        lineage_attrs=dict(group.attrs),
        source_refined_eye_run=run_name,
        source_refined_subject_run=_normalize_text(group.attrs.get("source_refined_subject_masks_run")),
    )


def resolve_eye_geometry_source(
    root: zarr.Group,
    *,
    subject_shape_run: Optional[str] = None,
    refined_subject_run: Optional[str] = None,
    refined_eye_run: Optional[str] = None,
    prefer_subject_shape: bool = False,
    prefer_subject: bool = True,
) -> EyeGeometrySource:
    """Resolve the active eye geometry source.

    Callers that only need analysis-facing geometry can opt into
    ``analysis/subject_shape_runs/<run>`` preference. Callers that need mask
    pixels should keep the default refined-subject/refined-eye behavior.
    """

    if subject_shape_run:
        run_name, group = _resolve_run_group(
            root,
            EYE_GEOMETRY_STAGE_SUBJECT_SHAPE,
            subject_shape_run,
            fallback_to_latest=True,
        )
        if run_name is None or group is None:
            raise ValueError(f"Subject-shape run not found: {subject_shape_run}.")
        return _build_subject_shape_source(run_name, group)

    if refined_subject_run:
        run_name, group = _resolve_run_group(
            root,
            EYE_GEOMETRY_STAGE_REFINED_SUBJECT,
            refined_subject_run,
            fallback_to_latest=True,
        )
        if run_name is None or group is None:
            raise ValueError(f"Refined subject-mask run not found: {refined_subject_run}.")
        refined_eye_group: Optional[zarr.Group] = None
        source_refined_eye_run = _normalize_text(group.attrs.get("source_refined_eye_masks_run"))
        if source_refined_eye_run:
            _eye_name, refined_eye_group = _resolve_run_group(
                root,
                EYE_GEOMETRY_STAGE_REFINED_EYE,
                source_refined_eye_run,
                fallback_to_latest=False,
            )
        return _build_subject_source(
            run_name,
            group,
            refined_eye_group=refined_eye_group,
            source_refined_eye_run=source_refined_eye_run,
        )

    if refined_eye_run:
        eye_name, eye_group = _resolve_run_group(
            root,
            EYE_GEOMETRY_STAGE_REFINED_EYE,
            refined_eye_run,
            fallback_to_latest=True,
        )
        if eye_name is None or eye_group is None:
            raise ValueError(f"Refined eye-mask run not found: {refined_eye_run}.")
        if prefer_subject:
            subject_name = _normalize_text(eye_group.attrs.get("source_refined_subject_masks_run"))
            subject_group = None
            if subject_name:
                _subject_name, subject_group = _resolve_run_group(
                    root,
                    EYE_GEOMETRY_STAGE_REFINED_SUBJECT,
                    subject_name,
                    fallback_to_latest=False,
                )
            if subject_group is None:
                subject_name, subject_group = _find_subject_run_for_refined_eye(root, eye_name)
            if subject_name is not None and subject_group is not None:
                return _build_subject_source(
                    subject_name,
                    subject_group,
                    refined_eye_group=eye_group,
                    source_refined_eye_run=eye_name,
                )
        return _build_refined_eye_source(eye_name, eye_group)

    if prefer_subject_shape:
        shape_name, shape_group = _find_latest_subject_shape_eye_geometry(root)
        if shape_name is not None and shape_group is not None:
            return _build_subject_shape_source(shape_name, shape_group)

    if prefer_subject:
        subject_name, subject_group = _find_latest_subject_eye_geometry(root)
        if subject_name is not None and subject_group is not None:
            source_refined_eye_run = _normalize_text(subject_group.attrs.get("source_refined_eye_masks_run"))
            refined_eye_group = None
            if source_refined_eye_run:
                _eye_name, refined_eye_group = _resolve_run_group(
                    root,
                    EYE_GEOMETRY_STAGE_REFINED_EYE,
                    source_refined_eye_run,
                    fallback_to_latest=False,
                )
            return _build_subject_source(
                subject_name,
                subject_group,
                refined_eye_group=refined_eye_group,
                source_refined_eye_run=source_refined_eye_run,
            )

    eye_name, eye_group = _resolve_run_group(
        root,
        EYE_GEOMETRY_STAGE_REFINED_EYE,
        None,
        fallback_to_latest=True,
    )
    if eye_name is None or eye_group is None:
        raise ValueError("No canonical refined-subject eye geometry or refined-eye mask run found.")
    return _build_refined_eye_source(eye_name, eye_group)


def resolve_source_keypoints_run_for_eye_geometry(source: EyeGeometrySource) -> Optional[str]:
    return resolve_source_keypoints_run(source.lineage_attrs)


__all__ = [
    "ChannelSelectionArray",
    "EYE_GEOMETRY_STAGE_REFINED_EYE",
    "EYE_GEOMETRY_STAGE_REFINED_SUBJECT",
    "EYE_GEOMETRY_STAGE_SUBJECT_SHAPE",
    "EyeGeometrySource",
    "StackedComponentArray",
    "resolve_eye_geometry_source",
    "resolve_source_keypoints_run_for_eye_geometry",
]
