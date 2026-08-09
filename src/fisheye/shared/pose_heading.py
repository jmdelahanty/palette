"""Shared helpers for metadata-driven keypoint heading evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .pose_schema import PoseSchema, canonicalize_keypoint_label


@dataclass(frozen=True)
class HeadingComputationResolution:
    source: Optional[str]
    spec: Optional[dict[str, object]]


def _coerce_mapping(value: object) -> Optional[dict[str, object]]:
    if isinstance(value, Mapping):
        return dict(value)
    return None


def _extract_pose_schema_metadata(pose_schema: object) -> Optional[dict[str, object]]:
    if isinstance(pose_schema, PoseSchema):
        return _coerce_mapping(pose_schema.metadata)

    pose_schema_map = _coerce_mapping(pose_schema)
    if pose_schema_map is None:
        return None
    metadata = pose_schema_map.get("metadata")
    return _coerce_mapping(metadata)


def resolve_heading_computation(
    *,
    pose_schema: object = None,
    heading_computation_override: object = None,
    heading_computation: object = None,
) -> HeadingComputationResolution:
    override = _coerce_mapping(heading_computation_override)
    if override is not None:
        return HeadingComputationResolution(
            source="heading_computation_override",
            spec=override,
        )

    metadata = _extract_pose_schema_metadata(pose_schema)
    if metadata is not None:
        schema_spec = _coerce_mapping(metadata.get("heading_computation"))
        if schema_spec is not None:
            return HeadingComputationResolution(
                source="pose_schema.metadata.heading_computation",
                spec=schema_spec,
            )

    legacy = _coerce_mapping(heading_computation)
    if legacy is not None:
        return HeadingComputationResolution(
            source="heading_computation",
            spec=legacy,
        )

    return HeadingComputationResolution(source=None, spec=None)


def resolve_heading_computation_from_attrs(
    attrs: Mapping[str, object],
) -> HeadingComputationResolution:
    return resolve_heading_computation(
        pose_schema=attrs.get("pose_schema"),
        heading_computation_override=attrs.get("heading_computation_override"),
        heading_computation=attrs.get("heading_computation"),
    )


def dependent_keypoints(spec: Mapping[str, object] | None) -> tuple[str, ...]:
    spec_map = _coerce_mapping(spec)
    if spec_map is None:
        return ()
    raw = spec_map.get("dependent_keypoints")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return ()
    labels: list[str] = []
    for value in raw:
        text = str(value).strip()
        if text:
            labels.append(text)
    return tuple(labels)


def _label_index_map(labels: Sequence[str]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for idx, label in enumerate(labels):
        text = canonicalize_keypoint_label(label)
        if text:
            mapping[text] = idx
    return mapping


def _extract_point(points: np.ndarray, idx: int) -> np.ndarray:
    point = np.asarray(points[idx], dtype=np.float64).reshape(-1)
    if point.size < 2:
        raise ValueError(f"Expected keypoint at index {idx} to expose at least 2 coordinates.")
    return point[:2]


def evaluate_point_expression(
    expr: Mapping[str, object] | None,
    *,
    labels: Sequence[str],
    points: np.ndarray,
    strict: bool = False,
) -> np.ndarray:
    expr_map = _coerce_mapping(expr)
    if expr_map is None:
        if strict:
            raise ValueError("Heading point expression is missing or invalid.")
        return np.full(2, np.nan, dtype=np.float64)

    op = str(expr_map.get("op", "")).strip()
    label_to_idx = _label_index_map(labels)

    try:
        if op == "keypoint":
            label = canonicalize_keypoint_label(expr_map.get("label"))
            if not label:
                raise ValueError("Heading keypoint expression is missing 'label'.")
            if label not in label_to_idx:
                raise ValueError(f"Heading keypoint '{label}' not found in labels.")
            return _extract_point(points, label_to_idx[label])

        if op == "midpoint":
            raw_labels = expr_map.get("labels")
            if not isinstance(raw_labels, Sequence) or isinstance(raw_labels, (str, bytes)):
                raise ValueError("Heading midpoint expression must define 'labels'.")
            midpoint_labels = [
                label
                for label in (canonicalize_keypoint_label(value) for value in raw_labels)
                if label is not None
            ]
            if len(midpoint_labels) != 2:
                raise ValueError(
                    "Heading midpoint expression currently requires exactly 2 labels."
                )
            pts = np.stack(
                [_extract_point(points, label_to_idx[label]) for label in midpoint_labels],
                axis=0,
            )
            return np.mean(pts, axis=0)

        raise ValueError(f"Unsupported heading point expression op '{op}'.")
    except Exception:
        if strict:
            raise
        return np.full(2, np.nan, dtype=np.float64)


def compute_heading_from_spec(
    spec: Mapping[str, object] | None,
    *,
    labels: Sequence[str],
    points: np.ndarray,
    strict: bool = False,
) -> float:
    spec_map = _coerce_mapping(spec)
    if spec_map is None:
        return float("nan")

    if spec_map.get("enabled") is False:
        return float("nan")

    try:
        start = evaluate_point_expression(
            _coerce_mapping(spec_map.get("direction_from")),
            labels=labels,
            points=points,
            strict=True,
        )
        end = evaluate_point_expression(
            _coerce_mapping(spec_map.get("direction_to")),
            labels=labels,
            points=points,
            strict=True,
        )
        head_vec = end - start
        if not np.all(np.isfinite(head_vec)) or np.linalg.norm(head_vec) == 0:
            return float("nan")
        return float(np.rad2deg(np.arctan2(-head_vec[1], head_vec[0])))
    except Exception:
        if strict:
            raise
        return float("nan")


def compute_heading_origin_from_spec(
    spec: Mapping[str, object] | None,
    *,
    labels: Sequence[str],
    points: np.ndarray,
    strict: bool = False,
) -> Optional[np.ndarray]:
    spec_map = _coerce_mapping(spec)
    if spec_map is None or spec_map.get("enabled") is False:
        return None
    try:
        origin = evaluate_point_expression(
            _coerce_mapping(spec_map.get("origin")),
            labels=labels,
            points=points,
            strict=True,
        )
    except Exception:
        if strict:
            raise
        return None
    return origin if np.all(np.isfinite(origin)) else None


def compute_heading_from_attrs(
    attrs: Mapping[str, object],
    *,
    labels: Sequence[str],
    points: np.ndarray,
    strict: bool = False,
) -> float:
    resolved = resolve_heading_computation_from_attrs(attrs)
    return compute_heading_from_spec(
        resolved.spec,
        labels=labels,
        points=points,
        strict=strict,
    )


def compute_heading_origin_from_attrs(
    attrs: Mapping[str, object],
    *,
    labels: Sequence[str],
    points: np.ndarray,
    strict: bool = False,
) -> Optional[np.ndarray]:
    resolved = resolve_heading_computation_from_attrs(attrs)
    return compute_heading_origin_from_spec(
        resolved.spec,
        labels=labels,
        points=points,
        strict=strict,
    )
