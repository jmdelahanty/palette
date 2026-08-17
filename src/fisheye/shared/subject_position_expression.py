"""Canonical point expressions and pure subject-position evaluators.

This module deliberately stops at the pure contract/evaluator boundary.  It
does not open Zarr stores, resolve ``latest`` pointers, inspect a schema, or
choose a source modality.  Callers supply the exact row-aligned arrays and
role/array bindings that the selected immutable source contract authorizes.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
from types import MappingProxyType
from typing import Any, Final, Mapping, Sequence

import numpy as np

from fisheye.shared.subject_position_types import (
    CANONICAL_FLOAT32_QNAN_BITS,
    OBSERVATION_POSITION_ROW_AXIS,
    POSITION_FAILURE_REASON_CODES,
    POSITION_FAILURE_REASON_PRECEDENCE,
    PositionEvaluationResult,
    SOURCE_CAMERA_POSITION_PROFILE_ID,
    canonical_float32_nan,
    empty_position_xy,
)
from fisheye.shared.coordinate_surface_contract import SOURCE_CAMERA_BBOX_XYXY
from fisheye.shared.zarr.manifest_digest import (
    canonical_json_bytes,
    canonical_json_sha256,
)

POINT_EXPRESSION_SCHEMA_ID: Final = "palette.subject_position_point_expression"
POINT_EXPRESSION_SCHEMA_VERSION: Final = 1
ESTIMATOR_PROFILE_SCHEMA_ID: Final = "palette.subject_position_estimator_profile"
ESTIMATOR_PROFILE_SCHEMA_VERSION: Final = 1
UPSTREAM_AUTHORITY_VALIDITY_POLICY_ID: Final = "upstream_authority_required.v1"
SOURCE_CAMERA_BBOX_SURFACE_ID: Final = SOURCE_CAMERA_BBOX_XYXY.surface_id

KEYPOINT_ANATOMICAL_TRIAD_MEAN_ESTIMATOR_ID: Final = "keypoint_anatomical_triad_mean.v1"
MASK_COMPONENT_ANATOMICAL_TRIAD_MEAN_ESTIMATOR_ID: Final = (
    "mask_component_anatomical_triad_mean.v1"
)
DETECTION_BBOX_CENTROID_ESTIMATOR_ID: Final = "detection_bbox_centroid.v1"
SUBJECT_BODY_MASK_CENTROID_ESTIMATOR_ID: Final = "subject_body_mask_centroid.v1"

_ESTIMATOR_PROFILE_IDS: Final = frozenset(
    {
        DETECTION_BBOX_CENTROID_ESTIMATOR_ID,
        KEYPOINT_ANATOMICAL_TRIAD_MEAN_ESTIMATOR_ID,
        MASK_COMPONENT_ANATOMICAL_TRIAD_MEAN_ESTIMATOR_ID,
        SUBJECT_BODY_MASK_CENTROID_ESTIMATOR_ID,
    }
)
_EXPRESSION_OPS: Final = frozenset(
    {"keypoint", "component_centroid", "bbox_centroid", "midpoint", "mean_points"}
)
_REASON = POSITION_FAILURE_REASON_CODES
_REASON_PRECEDENCE_CODES: Final = tuple(
    int(_REASON[tag]) for tag in POSITION_FAILURE_REASON_PRECEDENCE
)
ESTIMATOR_VALIDITY_POLICY_SCHEMA_ID: Final = (
    "palette.subject_position_estimator_validity_policy"
)
ESTIMATOR_VALIDITY_POLICY_SCHEMA_VERSION: Final = 1


def _validity_policy_record() -> dict[str, Any]:
    return {
        "schema_id": ESTIMATOR_VALIDITY_POLICY_SCHEMA_ID,
        "schema_version": ESTIMATOR_VALIDITY_POLICY_SCHEMA_VERSION,
        "policy_id": UPSTREAM_AUTHORITY_VALIDITY_POLICY_ID,
        "required_anchor_policy": "all_required",
        "confidence_policy": "upstream_confidence_valid_only",
        "missing_binding_policy": "structural_failure",
        "invalid_row_policy": "preserve_row_with_canonical_paired_qnan",
        "fallback": "none",
        "primary_reason_precedence": [
            {"tag": tag, "code": int(_REASON[tag])}
            for tag in POSITION_FAILURE_REASON_PRECEDENCE
        ],
    }


class _FrozenJsonDict(dict[str, Any]):
    """JSON-serializable mapping whose recursively frozen values cannot mutate."""

    @staticmethod
    def _immutable(*_args: object, **_kwargs: object) -> None:
        raise TypeError("Built-in estimator profile records are immutable.")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _FrozenJsonDict(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _require_exact_text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be one nonempty, trimmed string.")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{name} contains a control character.")
    return value


def _require_exact_mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object.")
    if any(type(key) is not str for key in value):
        raise ValueError(f"{name} field names must be strings.")
    return value


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    name: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        detail: list[str] = []
        if missing:
            detail.append(f"missing {missing}")
        if extra:
            detail.append(f"unknown {extra}")
        raise ValueError(f"{name} has an invalid field set ({'; '.join(detail)}).")


def _canonical_sort_key(value: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(value)


def _canonicalize_point_expression(value: object) -> dict[str, Any]:
    record = _require_exact_mapping(value, name="point expression")
    if "op" not in record:
        raise ValueError("Point expression must declare op.")
    op = _require_exact_text(record["op"], name="point expression op")
    if op not in _EXPRESSION_OPS:
        raise ValueError(f"Unknown point-expression operation {op!r}.")

    if op == "keypoint":
        _require_exact_fields(record, {"op", "role"}, name="keypoint expression")
        return {"op": op, "role": _require_exact_text(record["role"], name="role")}

    if op == "component_centroid":
        _require_exact_fields(
            record, {"op", "role"}, name="component-centroid expression"
        )
        return {
            "op": op,
            "role": _require_exact_text(record["role"], name="role"),
        }

    if op == "bbox_centroid":
        _require_exact_fields(
            record, {"op", "array_ref"}, name="bounding-box expression"
        )
        return {
            "op": op,
            "array_ref": _require_exact_text(
                record["array_ref"], name="bbox array_ref"
            ),
        }

    if op == "midpoint":
        _require_exact_fields(record, {"op", "point_a", "point_b"}, name="midpoint")
        point_a = _canonicalize_point_expression(record["point_a"])
        point_b = _canonicalize_point_expression(record["point_b"])
        if point_a == point_b:
            raise ValueError("Midpoint operands must be distinct.")
        ordered = sorted((point_a, point_b), key=_canonical_sort_key)
        return {"op": op, "point_a": ordered[0], "point_b": ordered[1]}

    _require_exact_fields(
        record, {"op", "points", "weighting"}, name="mean-points expression"
    )
    if type(record["points"]) not in (list, tuple):
        raise ValueError("mean_points points must be an array.")
    raw_points = record["points"]
    if len(raw_points) < 2:
        raise ValueError("mean_points requires at least two point expressions.")
    if record["weighting"] != "equal_per_point":
        raise ValueError("Only equal_per_point weighting is supported in v1.")
    points = [_canonicalize_point_expression(point) for point in raw_points]
    if len({canonical_json_bytes(point) for point in points}) != len(points):
        raise ValueError("mean_points cannot contain duplicate point expressions.")
    points.sort(key=_canonical_sort_key)
    return {"op": op, "points": points, "weighting": "equal_per_point"}


def canonicalize_point_expression(value: object) -> dict[str, Any]:
    """Validate and return the strict canonical v1 expression record."""

    return _canonicalize_point_expression(value)


normalize_point_expression = canonicalize_point_expression


def point_expression_envelope(value: object) -> dict[str, Any]:
    """Return the schema-bound canonical v1 expression envelope."""

    return {
        "schema_id": POINT_EXPRESSION_SCHEMA_ID,
        "schema_version": POINT_EXPRESSION_SCHEMA_VERSION,
        "expression": canonicalize_point_expression(value),
    }


def point_expression_digest(value: object) -> str:
    """Digest the schema-bound canonical expression envelope."""

    return canonical_json_sha256(point_expression_envelope(value))


expression_digest = point_expression_digest


def _reject_duplicate_json_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field {key!r}.")
        result[key] = value
    return result


def parse_point_expression_json(value: str | bytes | bytearray) -> dict[str, Any]:
    """Parse JSON with duplicate-field rejection and canonicalize the record."""

    try:
        parsed = json.loads(value, object_pairs_hook=_reject_duplicate_json_fields)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid point-expression JSON: {exc}.") from exc
    return canonicalize_point_expression(parsed)


@dataclass(frozen=True)
class PointArrayBinding:
    """One explicitly supplied row-aligned point source.

    ``values`` is ``[N, 2]`` in the already-authorized source-camera pixel
    coordinates.  ``valid`` and ``confidence_valid`` are upstream authority
    decisions, not values inferred from another modality.
    """

    values: Any
    valid: Any
    confidence: Any | None = None
    confidence_valid: Any | None = None


PointSourceBinding = PointArrayBinding


@dataclass(frozen=True)
class ComponentSourceBinding:
    """One explicit source-camera component-centroid source.

    Raster masks are intentionally not accepted by this pure evaluator.  A
    mask producer must first publish source-camera centroids under its own
    exact ROI-to-source transform and binary-mask contract.
    """

    centroids: Any
    valid: Any
    confidence: Any | None = None
    confidence_valid: Any | None = None


MaskComponentBinding = ComponentSourceBinding


@dataclass(frozen=True)
class BoundingBoxSourceBinding:
    """One authoritative half-open ``[N, 4]`` XYXY source and validity."""

    xyxy: Any
    valid: Any
    confidence: Any | None = None
    confidence_valid: Any | None = None


BBoxSourceBinding = BoundingBoxSourceBinding


@dataclass(frozen=True)
class PointExpressionBindings:
    """Explicit source bindings consumed by :func:`evaluate_point_expression`."""

    keypoints: Mapping[str, Any] = field(default_factory=dict)
    components: Mapping[str, Any] = field(default_factory=dict)
    bboxes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("keypoints", "components", "bboxes"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise ValueError(f"{name} bindings must be a mapping.")
            if any(type(key) is not str for key in value):
                raise ValueError(f"{name} binding keys must be strings.")


SourceBindings = PointExpressionBindings
SubjectPositionSourceBindings = PointExpressionBindings


def _as_numeric_array(value: Any, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array.") from exc
    if array.dtype.hasobject or np.issubdtype(array.dtype, np.complexfloating):
        raise ValueError(f"{name} must be a real numeric array.")
    if array.dtype == np.dtype(bool) or not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be a numeric array.")
    return array


def _as_bool_rows(value: Any, row_count: int, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.dtype(bool) or array.shape != (row_count,):
        raise ValueError(f"{name} must be bool[{row_count}].")
    return np.asarray(array, dtype=bool)


def _as_confidence_rows(value: Any, row_count: int, *, name: str) -> np.ndarray:
    array = _as_numeric_array(value, name=name)
    if array.shape != (row_count,):
        raise ValueError(f"{name} must have shape ({row_count},).")
    return np.asarray(array, dtype=np.float64)


def _set_context_row_count(current: int | None, observed: int, *, name: str) -> int:
    if current is not None and current != observed:
        raise ValueError(
            f"Row-aligned source mismatch for {name}: expected {current}, got {observed}."
        )
    return observed if current is None else current


@dataclass
class _EvaluationContext:
    row_count: int | None


@dataclass
class _EvaluatedExpression:
    values64: np.ndarray
    valid: np.ndarray
    reasons: np.ndarray
    support_values64: np.ndarray
    support_valid: np.ndarray
    support_reasons: np.ndarray
    support_confidence64: np.ndarray | None


def _binding_mapping(value: Any, *, name: str, allowed: set[str]) -> Mapping[str, Any]:
    if isinstance(
        value, (PointArrayBinding, ComponentSourceBinding, BoundingBoxSourceBinding)
    ):
        return value.__dict__
    mapping = _require_exact_mapping(value, name=name)
    if not set(mapping).issubset(allowed):
        raise ValueError(f"{name} has unknown binding fields.")
    return mapping


def _confidence_state(
    mapping: Mapping[str, Any],
    row_count: int,
    *,
    name: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    confidence = mapping.get("confidence")
    confidence_valid = mapping.get("confidence_valid")
    low = np.zeros(row_count, dtype=bool)
    confidence64: np.ndarray | None = None
    if confidence is not None:
        confidence64 = _as_confidence_rows(
            confidence, row_count, name=f"{name}.confidence"
        )
        if not np.isfinite(confidence64).all():
            raise ValueError(
                f"{name}.confidence must be finite when diagnostic confidence is supplied."
            )
    if confidence_valid is not None:
        confidence_valid_array = _as_bool_rows(
            confidence_valid, row_count, name=f"{name}.confidence_valid"
        )
        low |= ~confidence_valid_array
    return low, confidence64


def _support_confidence(
    confidence64: np.ndarray | None, *, row_count: int
) -> np.ndarray | None:
    if confidence64 is None:
        return None
    return confidence64.reshape(row_count, 1)


def _make_leaf(
    values64: np.ndarray,
    valid: np.ndarray,
    reasons: np.ndarray,
    confidence64: np.ndarray | None,
) -> _EvaluatedExpression:
    support_values = np.asarray(values64, dtype=np.float64).reshape(-1, 1, 2)
    support_valid = np.asarray(valid, dtype=bool).reshape(-1, 1)
    support_reasons = np.asarray(reasons, dtype=np.uint16).reshape(-1, 1)
    support_confidence = _support_confidence(confidence64, row_count=values64.shape[0])
    return _EvaluatedExpression(
        values64=np.asarray(values64, dtype=np.float64),
        valid=support_valid[:, 0],
        reasons=support_reasons[:, 0],
        support_values64=support_values,
        support_valid=support_valid,
        support_reasons=support_reasons,
        support_confidence64=support_confidence,
    )


def _point_binding(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, (PointArrayBinding, Mapping)):
        raise ValueError(
            f"{name} requires an explicit binding with values and upstream "
            "valid bool[N]; bare point arrays are not accepted."
        )
    mapping = _binding_mapping(
        value,
        name=name,
        allowed={"values", "valid", "confidence", "confidence_valid"},
    )
    if "values" not in mapping:
        raise ValueError(f"{name} must declare values.")
    if "valid" not in mapping or mapping.get("valid") is None:
        raise ValueError(
            f"{name}.valid is missing; keypoint leaves require explicit "
            "upstream valid bool[N]."
        )
    return mapping


def _evaluate_point_leaf(
    binding: Any,
    context: _EvaluationContext,
    *,
    name: str,
    rejected_reason: int,
) -> _EvaluatedExpression:
    mapping = _point_binding(binding, name=name)
    values = _as_numeric_array(mapping["values"], name=f"{name}.values")
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"{name}.values must have shape [N, 2].")
    row_count = int(values.shape[0])
    context.row_count = _set_context_row_count(context.row_count, row_count, name=name)
    values64 = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values64).all(axis=1)
    valid = finite.copy()
    reasons = np.zeros(row_count, dtype=np.uint16)
    reasons[~finite] = np.uint16(_REASON["nonfinite_source_geometry"])
    source_valid = _as_bool_rows(mapping["valid"], row_count, name=f"{name}.valid")
    rejected = ~source_valid & finite
    valid &= source_valid
    reasons[rejected] = np.uint16(rejected_reason)
    low_confidence, confidence64 = _confidence_state(mapping, row_count, name=name)
    low = low_confidence & finite & valid
    valid &= ~low_confidence
    reasons[low] = np.uint16(_REASON["required_anchor_low_confidence"])
    return _make_leaf(values64, valid, reasons, confidence64)


def _component_binding(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, (ComponentSourceBinding, Mapping)):
        raise ValueError(
            f"{name} requires an explicit source-camera centroids binding; "
            "bare component arrays and masks are not accepted."
        )
    if isinstance(value, Mapping) and "masks" in value:
        raise ValueError(
            f"{name} cannot consume masks; bind canonical source-camera centroids."
        )
    mapping = _binding_mapping(
        value,
        name=name,
        allowed={
            "centroids",
            "valid",
            "confidence",
            "confidence_valid",
        },
    )
    if "centroids" not in mapping or mapping.get("centroids") is None:
        raise ValueError(f"{name} must declare source-camera centroids.")
    if "valid" not in mapping or mapping.get("valid") is None:
        raise ValueError(f"{name} must declare upstream centroid validity.")
    return mapping


def _evaluate_component_leaf(
    binding: Any,
    context: _EvaluationContext,
    *,
    name: str,
) -> _EvaluatedExpression:
    mapping = _component_binding(binding, name=name)
    centroids = _as_numeric_array(mapping["centroids"], name=f"{name}.centroids")
    if centroids.ndim != 2 or centroids.shape[1] != 2:
        raise ValueError(f"{name}.centroids must have shape [N, 2].")
    row_count = int(centroids.shape[0])
    context.row_count = _set_context_row_count(context.row_count, row_count, name=name)
    values64 = np.asarray(centroids, dtype=np.float64)
    finite = np.isfinite(values64).all(axis=1)
    source_valid = _as_bool_rows(mapping["valid"], row_count, name=f"{name}.valid")
    reasons = np.zeros(row_count, dtype=np.uint16)
    reasons[~source_valid] = np.uint16(_REASON["empty_mask_component"])
    invalid_nonfinite = ~finite & source_valid
    reasons[invalid_nonfinite] = np.uint16(_REASON["nonfinite_source_geometry"])
    low_confidence, confidence64 = _confidence_state(mapping, row_count, name=name)
    low = low_confidence & source_valid & finite
    valid = source_valid & finite & ~low_confidence
    reasons[low] = np.uint16(_REASON["required_anchor_low_confidence"])
    return _make_leaf(values64, valid, reasons, confidence64)


def _bbox_binding(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, (BoundingBoxSourceBinding, Mapping)):
        raise ValueError(
            f"{name} requires an explicit binding with xyxy and upstream "
            "valid bool[N]; bare bbox arrays are not accepted."
        )
    mapping = _binding_mapping(
        value,
        name=name,
        allowed={"xyxy", "valid", "confidence", "confidence_valid"},
    )
    if "xyxy" not in mapping:
        raise ValueError(f"{name} must declare xyxy.")
    if "valid" not in mapping or mapping.get("valid") is None:
        raise ValueError(
            f"{name}.valid is missing; bbox leaves require explicit upstream "
            "valid bool[N]."
        )
    return mapping


def _evaluate_bbox_leaf(
    binding: Any,
    context: _EvaluationContext,
    *,
    name: str,
) -> _EvaluatedExpression:
    mapping = _bbox_binding(binding, name=name)
    boxes = _as_numeric_array(mapping["xyxy"], name=f"{name}.xyxy")
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"{name}.xyxy must have shape [N, 4].")
    row_count = int(boxes.shape[0])
    context.row_count = _set_context_row_count(context.row_count, row_count, name=name)
    boxes64 = np.asarray(boxes, dtype=np.float64)
    finite = np.isfinite(boxes64).all(axis=1)
    nondegenerate = (boxes64[:, 2] > boxes64[:, 0]) & (boxes64[:, 3] > boxes64[:, 1])
    valid = finite & nondegenerate
    reasons = np.zeros(row_count, dtype=np.uint16)
    reasons[~finite] = np.uint16(_REASON["nonfinite_source_geometry"])
    degenerate = finite & ~nondegenerate
    reasons[degenerate] = np.uint16(_REASON["degenerate_source_geometry"])
    source_valid = _as_bool_rows(mapping["valid"], row_count, name=f"{name}.valid")
    rejected = ~source_valid & finite & nondegenerate
    valid &= source_valid
    reasons[rejected] = np.uint16(_REASON["source_observation_rejected"])
    low_confidence, confidence64 = _confidence_state(mapping, row_count, name=name)
    low = low_confidence & finite & nondegenerate & valid
    valid &= ~low_confidence
    reasons[low] = np.uint16(_REASON["required_anchor_low_confidence"])
    values64 = np.empty((row_count, 2), dtype=np.float64)
    values64[:, 0] = (boxes64[:, 0] + boxes64[:, 2]) / 2.0
    values64[:, 1] = (boxes64[:, 1] + boxes64[:, 3]) / 2.0
    return _make_leaf(values64, valid, reasons, confidence64)


def _combine_support(
    children: Sequence[_EvaluatedExpression],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    values = np.concatenate([child.support_values64 for child in children], axis=1)
    valid = np.concatenate([child.support_valid for child in children], axis=1)
    reasons = np.concatenate([child.support_reasons for child in children], axis=1)
    if all(child.support_confidence64 is not None for child in children):
        confidence = np.concatenate(
            [
                child.support_confidence64
                for child in children
                if child.support_confidence64 is not None
            ],
            axis=1,
        )
    else:
        confidence = None
    return values, valid, reasons, confidence


def _primary_reason(row_reasons: Sequence[np.ndarray]) -> np.ndarray:
    if not row_reasons:
        raise ValueError("At least one child reason array is required.")
    stack = np.stack(row_reasons, axis=1).astype(np.uint16, copy=False)
    result = np.zeros(stack.shape[0], dtype=np.uint16)
    for code in _REASON_PRECEDENCE_CODES:
        present = np.any(stack == code, axis=1) & (result == 0)
        result[present] = np.uint16(code)
    return result


def _evaluate_expression(
    expression: Mapping[str, Any],
    bindings: PointExpressionBindings,
    context: _EvaluationContext,
) -> _EvaluatedExpression:
    op = expression["op"]
    if op == "keypoint":
        role = expression["role"]
        if role not in bindings.keypoints:
            raise ValueError(f"No explicit keypoint binding for role {role!r}.")
        return _evaluate_point_leaf(
            bindings.keypoints[role],
            context,
            name=f"keypoint[{role}]",
            rejected_reason=_REASON["required_anchor_invalid"],
        )
    if op == "component_centroid":
        role = expression["role"]
        if role not in bindings.components:
            raise ValueError(f"No explicit component binding for role {role!r}.")
        return _evaluate_component_leaf(
            bindings.components[role], context, name=f"component[{role}]"
        )
    if op == "bbox_centroid":
        array_ref = expression["array_ref"]
        if array_ref not in bindings.bboxes:
            raise ValueError(f"No explicit bbox binding for array_ref {array_ref!r}.")
        return _evaluate_bbox_leaf(
            bindings.bboxes[array_ref], context, name=f"bbox[{array_ref}]"
        )

    if op == "midpoint":
        children = [
            _evaluate_expression(expression["point_a"], bindings, context),
            _evaluate_expression(expression["point_b"], bindings, context),
        ]
        denominator = 2.0
    else:
        children = [
            _evaluate_expression(point, bindings, context)
            for point in expression["points"]
        ]
        denominator = float(len(children))
    values = np.zeros_like(children[0].values64, dtype=np.float64)
    for child in children:
        values += np.where(child.valid[:, None], child.values64, 0.0)
    values /= denominator
    valid = np.logical_and.reduce([child.valid for child in children])
    reasons = _primary_reason([child.reasons for child in children])
    support_values, support_valid, support_reasons, support_confidence = (
        _combine_support(children)
    )
    return _EvaluatedExpression(
        values64=values,
        valid=valid,
        reasons=reasons,
        support_values64=support_values,
        support_valid=support_valid,
        support_reasons=support_reasons,
        support_confidence64=support_confidence,
    )


def _coerce_bindings(
    source_bindings: PointExpressionBindings | Mapping[str, Any] | None,
    *,
    keypoints: Mapping[str, Any] | None,
    components: Mapping[str, Any] | None,
    bboxes: Mapping[str, Any] | None,
) -> PointExpressionBindings:
    supplied = [
        source_bindings is not None,
        keypoints is not None,
        components is not None,
        bboxes is not None,
    ]
    if source_bindings is not None and any(supplied[1:]):
        raise ValueError("Use source_bindings or named source categories, not both.")
    if source_bindings is None:
        return PointExpressionBindings(
            keypoints={} if keypoints is None else keypoints,
            components={} if components is None else components,
            bboxes={} if bboxes is None else bboxes,
        )
    if isinstance(source_bindings, PointExpressionBindings):
        return source_bindings
    mapping = _require_exact_mapping(source_bindings, name="source_bindings")
    _require_exact_fields(
        mapping,
        {"keypoints", "components", "bboxes"},
        name="source_bindings",
    )
    return PointExpressionBindings(
        keypoints=mapping["keypoints"],
        components=mapping["components"],
        bboxes=mapping["bboxes"],
    )


def _canonicalize_output(
    evaluation: _EvaluatedExpression,
) -> PositionEvaluationResult:
    valid = np.asarray(evaluation.valid, dtype=bool)
    reasons = np.asarray(evaluation.reasons, dtype=np.uint16)
    if valid.shape != (evaluation.values64.shape[0],):
        raise ValueError("Evaluator produced an invalid row-coverage shape.")
    if np.any(valid & (reasons != 0)) or np.any(~valid & (reasons == 0)):
        raise ValueError("Evaluator produced inconsistent validity/reason state.")
    if np.any(valid & ~np.isfinite(evaluation.values64).all(axis=1)):
        raise ValueError("Evaluator produced nonfinite coordinates for a valid row.")
    values32 = np.asarray(evaluation.values64, dtype=np.float32)
    if np.any(valid & ~np.isfinite(values32).all(axis=1)):
        raise ValueError("Evaluator result overflows float32 publication.")
    invalid = ~valid
    values32[invalid] = canonical_float32_nan()
    values32.view(np.uint32)[invalid, :] = CANONICAL_FLOAT32_QNAN_BITS

    support32 = np.asarray(evaluation.support_values64, dtype=np.float32)
    support_invalid = ~np.asarray(evaluation.support_valid, dtype=bool)
    if support32.ndim != 3 or support32.shape[2] != 2:
        raise ValueError("Evaluator produced invalid support geometry shape.")
    if np.any(~support_invalid & ~np.isfinite(support32).all(axis=2)):
        raise ValueError("Evaluator produced nonfinite valid support geometry.")
    support32[support_invalid] = canonical_float32_nan()
    support32.view(np.uint32)[support_invalid, :] = CANONICAL_FLOAT32_QNAN_BITS
    support_confidence = evaluation.support_confidence64
    if support_confidence is not None:
        if not np.isfinite(support_confidence).all():
            raise ValueError("Evaluator produced nonfinite diagnostic confidence.")
        support_confidence32 = np.asarray(support_confidence, dtype=np.float32)
        if not np.isfinite(support_confidence32).all():
            raise ValueError("Diagnostic confidence overflows float32 publication.")
    else:
        support_confidence32 = None
    return PositionEvaluationResult(
        position_xy=values32,
        valid=valid,
        failure_reason_codes=reasons,
        source_points_xy=support32,
        source_points_valid=np.asarray(evaluation.support_valid, dtype=bool),
        source_point_reason_codes=np.asarray(
            evaluation.support_reasons, dtype=np.uint16
        ),
        source_point_confidence=support_confidence32,
    )


def evaluate_point_expression(
    expression: Mapping[str, Any],
    source_bindings: PointExpressionBindings | Mapping[str, Any] | None = None,
    *,
    keypoints: Mapping[str, Any] | None = None,
    components: Mapping[str, Any] | None = None,
    bboxes: Mapping[str, Any] | None = None,
    row_count: int | None = None,
) -> PositionEvaluationResult:
    """Evaluate one canonical expression over explicit row-aligned sources.

    The evaluator never discovers sources or resolves roles.  The expression's
    role is looked up exactly in the corresponding explicit mapping.  A
    structural missing binding raises ``ValueError``; ordinary row failures
    retain their row with a canonical paired NaN and a controlled reason code.
    """

    canonical = canonicalize_point_expression(expression)
    resolved_bindings = _coerce_bindings(
        source_bindings,
        keypoints=keypoints,
        components=components,
        bboxes=bboxes,
    )
    if row_count is not None:
        if type(row_count) is not int or row_count < 0:
            raise ValueError("row_count must be a non-negative exact integer.")
    context = _EvaluationContext(row_count=row_count)
    evaluation = _evaluate_expression(canonical, resolved_bindings, context)
    if context.row_count is None:
        raise ValueError("Expression evaluation requires at least one source array.")
    if context.row_count == 0:
        # Exercise the storage helper's exact empty shape/NaN contract as part
        # of the pure evaluator boundary.
        evaluation.values64 = np.empty((0, 2), dtype=np.float64)
        result = _canonicalize_output(evaluation)
        result.position_xy[...] = empty_position_xy(0)
        return result
    return _canonicalize_output(evaluation)


evaluate_expression = evaluate_point_expression


def _profile_record(
    *,
    estimator_id: str,
    source_modality: str,
    expression: Mapping[str, Any],
    anatomy_profile_id: str | None,
) -> dict[str, Any]:
    return {
        "schema_id": ESTIMATOR_PROFILE_SCHEMA_ID,
        "schema_version": ESTIMATOR_PROFILE_SCHEMA_VERSION,
        "estimator_id": estimator_id,
        "estimator_version": 1,
        "source_modality": source_modality,
        "anatomy_profile_id": anatomy_profile_id,
        "expression": canonicalize_point_expression(expression),
        "coordinate_profile_id": SOURCE_CAMERA_POSITION_PROFILE_ID,
        "row_axis": OBSERVATION_POSITION_ROW_AXIS,
        "validity_policy_id": UPSTREAM_AUTHORITY_VALIDITY_POLICY_ID,
        "validity_policy": _validity_policy_record(),
        "fallback": "none",
    }


_DETECTION_BBOX_CENTROID_PROFILE_RECORD = _profile_record(
    estimator_id=DETECTION_BBOX_CENTROID_ESTIMATOR_ID,
    source_modality="detection",
    expression={"op": "bbox_centroid", "array_ref": "bbox_img_xyxy"},
    anatomy_profile_id=None,
)
_KEYPOINT_ANATOMICAL_TRIAD_MEAN_PROFILE_RECORD = _profile_record(
    estimator_id=KEYPOINT_ANATOMICAL_TRIAD_MEAN_ESTIMATOR_ID,
    source_modality="keypoint",
    expression={
        "op": "mean_points",
        "points": [
            {"op": "keypoint", "role": "swim_bladder"},
            {"op": "keypoint", "role": "eye_left"},
            {"op": "keypoint", "role": "eye_right"},
        ],
        "weighting": "equal_per_point",
    },
    anatomy_profile_id="zebrafish_larva_anatomy.v1",
)
_MASK_COMPONENT_ANATOMICAL_TRIAD_MEAN_PROFILE_RECORD = _profile_record(
    estimator_id=MASK_COMPONENT_ANATOMICAL_TRIAD_MEAN_ESTIMATOR_ID,
    source_modality="subject_mask",
    expression={
        "op": "mean_points",
        "points": [
            {"op": "component_centroid", "role": "swim_bladder"},
            {"op": "component_centroid", "role": "eye_left"},
            {"op": "component_centroid", "role": "eye_right"},
        ],
        "weighting": "equal_per_point",
    },
    anatomy_profile_id="zebrafish_larva_anatomy.v1",
)
_SUBJECT_BODY_MASK_CENTROID_PROFILE_RECORD = _profile_record(
    estimator_id=SUBJECT_BODY_MASK_CENTROID_ESTIMATOR_ID,
    source_modality="subject_mask",
    expression={"op": "component_centroid", "role": "subject_body"},
    anatomy_profile_id="zebrafish_larva_anatomy.v1",
)

_BUILTIN_ESTIMATOR_PROFILE_JSON: Final[Mapping[str, bytes]] = MappingProxyType(
    {
        str(profile["estimator_id"]): canonical_json_bytes(profile)
        for profile in (
            _DETECTION_BBOX_CENTROID_PROFILE_RECORD,
            _KEYPOINT_ANATOMICAL_TRIAD_MEAN_PROFILE_RECORD,
            _MASK_COMPONENT_ANATOMICAL_TRIAD_MEAN_PROFILE_RECORD,
            _SUBJECT_BODY_MASK_CENTROID_PROFILE_RECORD,
        )
    }
)


def _load_builtin_estimator_profile(estimator_id: str) -> dict[str, Any]:
    return json.loads(_BUILTIN_ESTIMATOR_PROFILE_JSON[estimator_id])


DETECTION_BBOX_CENTROID_PROFILE: Final[Mapping[str, Any]] = _freeze_json(
    _load_builtin_estimator_profile(DETECTION_BBOX_CENTROID_ESTIMATOR_ID)
)
KEYPOINT_ANATOMICAL_TRIAD_MEAN_PROFILE: Final[Mapping[str, Any]] = _freeze_json(
    _load_builtin_estimator_profile(KEYPOINT_ANATOMICAL_TRIAD_MEAN_ESTIMATOR_ID)
)
MASK_COMPONENT_ANATOMICAL_TRIAD_MEAN_PROFILE: Final[Mapping[str, Any]] = _freeze_json(
    _load_builtin_estimator_profile(MASK_COMPONENT_ANATOMICAL_TRIAD_MEAN_ESTIMATOR_ID)
)
SUBJECT_BODY_MASK_CENTROID_PROFILE: Final[Mapping[str, Any]] = _freeze_json(
    _load_builtin_estimator_profile(SUBJECT_BODY_MASK_CENTROID_ESTIMATOR_ID)
)

ESTIMATOR_PROFILE_RECORDS: Final[Mapping[str, Mapping[str, Any]]] = MappingProxyType(
    {
        str(profile["estimator_id"]): profile
        for profile in (
            DETECTION_BBOX_CENTROID_PROFILE,
            KEYPOINT_ANATOMICAL_TRIAD_MEAN_PROFILE,
            MASK_COMPONENT_ANATOMICAL_TRIAD_MEAN_PROFILE,
            SUBJECT_BODY_MASK_CENTROID_PROFILE,
        )
    }
)


def canonicalize_estimator_profile(value: object) -> dict[str, Any]:
    """Validate one exact v1 profile record and return its canonical form."""

    record = _require_exact_mapping(value, name="estimator profile")
    expected_fields = {
        "schema_id",
        "schema_version",
        "estimator_id",
        "estimator_version",
        "source_modality",
        "anatomy_profile_id",
        "expression",
        "coordinate_profile_id",
        "row_axis",
        "validity_policy_id",
        "validity_policy",
        "fallback",
    }
    _require_exact_fields(record, expected_fields, name="estimator profile")
    if record["schema_id"] != ESTIMATOR_PROFILE_SCHEMA_ID:
        raise ValueError("Unknown estimator profile schema ID.")
    if type(record["schema_version"]) is not int or record["schema_version"] != 1:
        raise ValueError("Only estimator profile schema version 1 is supported.")
    estimator_id = _require_exact_text(record["estimator_id"], name="estimator_id")
    if estimator_id not in _ESTIMATOR_PROFILE_IDS:
        raise ValueError(f"Unknown estimator profile {estimator_id!r}.")
    if type(record["estimator_version"]) is not int or record["estimator_version"] != 1:
        raise ValueError("Only estimator profile version 1 is supported.")
    source_modality = _require_exact_text(
        record["source_modality"], name="source_modality"
    )
    anatomy_id = record["anatomy_profile_id"]
    if anatomy_id is not None:
        anatomy_id = _require_exact_text(anatomy_id, name="anatomy_profile_id")
    validity_policy = _require_exact_mapping(
        record["validity_policy"], name="validity_policy"
    )
    if canonical_json_bytes(validity_policy) != canonical_json_bytes(
        _validity_policy_record()
    ):
        raise ValueError("Estimator validity policy or reason precedence is invalid.")
    canonical = {
        "schema_id": ESTIMATOR_PROFILE_SCHEMA_ID,
        "schema_version": 1,
        "estimator_id": estimator_id,
        "estimator_version": 1,
        "source_modality": source_modality,
        "anatomy_profile_id": anatomy_id,
        "expression": canonicalize_point_expression(record["expression"]),
        "coordinate_profile_id": _require_exact_text(
            record["coordinate_profile_id"], name="coordinate_profile_id"
        ),
        "row_axis": _require_exact_text(record["row_axis"], name="row_axis"),
        "validity_policy_id": _require_exact_text(
            record["validity_policy_id"], name="validity_policy_id"
        ),
        "validity_policy": _validity_policy_record(),
        "fallback": _require_exact_text(record["fallback"], name="fallback"),
    }
    expected = _load_builtin_estimator_profile(estimator_id)
    if canonical != expected:
        raise ValueError(
            f"Estimator profile {estimator_id!r} does not match its v1 record."
        )
    return deepcopy(canonical)


validate_estimator_profile = canonicalize_estimator_profile


def estimator_profile_digest(value: object) -> str:
    """Digest an exact canonical estimator-profile record."""

    return canonical_json_sha256(canonicalize_estimator_profile(value))


def get_estimator_profile(estimator_id: str) -> dict[str, Any]:
    """Return a detached exact profile record by ID; no default is selected."""

    estimator_id = _require_exact_text(estimator_id, name="estimator_id")
    if estimator_id not in ESTIMATOR_PROFILE_RECORDS:
        raise ValueError(f"Unknown estimator profile {estimator_id!r}.")
    return _load_builtin_estimator_profile(estimator_id)


def evaluate_estimator_profile(
    profile: str | Mapping[str, Any],
    source_bindings: PointExpressionBindings | Mapping[str, Any],
    *,
    row_count: int | None = None,
) -> PositionEvaluationResult:
    """Evaluate a validated estimator profile using only explicit bindings."""

    resolved = (
        get_estimator_profile(profile)
        if isinstance(profile, str)
        else canonicalize_estimator_profile(profile)
    )
    return evaluate_point_expression(
        resolved["expression"],
        source_bindings,
        row_count=row_count,
    )


evaluate_profile = evaluate_estimator_profile


__all__ = [
    "BBoxSourceBinding",
    "BoundingBoxSourceBinding",
    "ComponentSourceBinding",
    "DETECTION_BBOX_CENTROID_ESTIMATOR_ID",
    "DETECTION_BBOX_CENTROID_PROFILE",
    "ESTIMATOR_PROFILE_RECORDS",
    "ESTIMATOR_PROFILE_SCHEMA_ID",
    "ESTIMATOR_PROFILE_SCHEMA_VERSION",
    "ESTIMATOR_VALIDITY_POLICY_SCHEMA_ID",
    "ESTIMATOR_VALIDITY_POLICY_SCHEMA_VERSION",
    "KEYPOINT_ANATOMICAL_TRIAD_MEAN_ESTIMATOR_ID",
    "KEYPOINT_ANATOMICAL_TRIAD_MEAN_PROFILE",
    "MASK_COMPONENT_ANATOMICAL_TRIAD_MEAN_ESTIMATOR_ID",
    "MASK_COMPONENT_ANATOMICAL_TRIAD_MEAN_PROFILE",
    "MaskComponentBinding",
    "POINT_EXPRESSION_SCHEMA_ID",
    "POINT_EXPRESSION_SCHEMA_VERSION",
    "PointArrayBinding",
    "PointExpressionBindings",
    "PointSourceBinding",
    "SUBJECT_BODY_MASK_CENTROID_ESTIMATOR_ID",
    "SUBJECT_BODY_MASK_CENTROID_PROFILE",
    "SourceBindings",
    "SOURCE_CAMERA_BBOX_SURFACE_ID",
    "SubjectPositionSourceBindings",
    "UPSTREAM_AUTHORITY_VALIDITY_POLICY_ID",
    "canonicalize_estimator_profile",
    "canonicalize_point_expression",
    "estimator_profile_digest",
    "evaluate_estimator_profile",
    "evaluate_expression",
    "evaluate_point_expression",
    "evaluate_profile",
    "expression_digest",
    "get_estimator_profile",
    "normalize_point_expression",
    "parse_point_expression_json",
    "point_expression_envelope",
    "point_expression_digest",
    "validate_estimator_profile",
]
