"""Historical-v1 directed-homography compatibility and migration support.

Future transform publication uses :mod:`fisheye.shared.directed_transform_v2`
with sealed typed endpoints and transform authorities.  The readers and
application helpers here remain available for historical archives.  V1
mutation/build helpers are intentionally absent from ``__all__`` and are used
only by explicit compatibility or migration code.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping

import numpy as np

from fisheye.shared.coordinate_descriptor import (
    COORDINATE_SPACE_IDS,
    NORMALIZED_SPACE_IDS,
    PIXEL_SPACE_IDS,
    REFERENCE_UNITS,
)


DIRECTED_TRANSFORM_SCHEMA_ID = "palette.directed_transform"
DIRECTED_TRANSFORM_SCHEMA_VERSION = 1
DIRECTED_TRANSFORM_KIND = "homography"
DIRECTED_TRANSFORM_DIRECTION = "source_to_target"
DIRECTED_TRANSFORM_ATTR = "directed_transform"
DIRECTED_TRANSFORM_DIGEST_SUFFIX = "_sha256"
DIRECTED_TRANSFORM_CANONICALIZATION = "canonical_json_sort_keys_v1"
HOMOGRAPHY_MATRIX_CANONICALIZATION = "float64_little_endian_c_order_v1"
CAMERA_BOUND_SPACE_IDS = frozenset(
    {"source_camera_image_px", "source_camera_normalized_xy"}
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "transform_id",
        "kind",
        "from_space_id",
        "to_space_id",
        "direction",
        "source_reference_extent",
        "target_reference_extent",
        "calibration_ref",
        "matrix_sha256",
    }
)
_OPTIONAL_FIELDS = frozenset({"camera_id", "source_transform"})
_REFERENCE_FIELDS = frozenset({"width", "height", "units", "authority"})
_SOURCE_TRANSFORM_FIELDS = frozenset(
    {"relationship", "transform_id", "transform_sha256"}
)


class DirectedTransformError(ValueError):
    """Raised when transform metadata, its matrix, or an application is unsafe."""


@dataclass(frozen=True)
class TransformReferenceExtent:
    """Exact authoritative extent for one side of a directed transform."""

    width: int | float
    height: int | float
    units: str
    authority: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "units": self.units,
            "authority": self.authority,
        }


@dataclass(frozen=True)
class SourceTransformReference:
    """Canonical reference to the transform from which this one was derived."""

    relationship: str
    transform_id: str
    transform_sha256: str

    def to_dict(self) -> dict[str, str]:
        return {
            "relationship": self.relationship,
            "transform_id": self.transform_id,
            "transform_sha256": self.transform_sha256,
        }


@dataclass(frozen=True)
class DirectedHomography:
    """Canonical metadata for a separately persisted 3x3 homography matrix."""

    transform_id: str
    from_space_id: str
    to_space_id: str
    source_reference_extent: TransformReferenceExtent
    target_reference_extent: TransformReferenceExtent
    calibration_ref: str
    matrix_sha256: str
    camera_id: str | None = None
    source_transform: SourceTransformReference | None = None

    @property
    def kind(self) -> str:
        return DIRECTED_TRANSFORM_KIND

    @property
    def direction(self) -> str:
        return DIRECTED_TRANSFORM_DIRECTION

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_id": DIRECTED_TRANSFORM_SCHEMA_ID,
            "schema_version": DIRECTED_TRANSFORM_SCHEMA_VERSION,
            "transform_id": self.transform_id,
            "kind": self.kind,
            "from_space_id": self.from_space_id,
            "to_space_id": self.to_space_id,
            "direction": self.direction,
            "source_reference_extent": self.source_reference_extent.to_dict(),
            "target_reference_extent": self.target_reference_extent.to_dict(),
            "calibration_ref": self.calibration_ref,
            "matrix_sha256": self.matrix_sha256,
        }
        if self.camera_id is not None:
            payload["camera_id"] = self.camera_id
        if self.source_transform is not None:
            payload["source_transform"] = self.source_transform.to_dict()
        return payload

    def canonical_json(self) -> str:
        return canonical_directed_transform_json(self)

    def digest(self) -> str:
        return directed_transform_digest(self)


@dataclass(frozen=True)
class BoundDirectedHomography:
    """One exact persisted matrix bound to validated metadata and its array path."""

    array_path: str
    matrix: np.ndarray
    transform: DirectedHomography
    transform_sha256: str

    @property
    def matrix_sha256(self) -> str:
        return self.transform.matrix_sha256


def _required_text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise DirectedTransformError(f"{field_name} must be a non-empty string.")
    if value != value.strip():
        raise DirectedTransformError(
            f"{field_name} must not have surrounding whitespace."
        )
    return value


def _sha256(value: Any, *, field_name: str) -> str:
    text = _required_text(value, field_name=field_name)
    if _SHA256_RE.fullmatch(text) is None:
        raise DirectedTransformError(
            f"{field_name} must be a lowercase 64-character SHA-256 digest."
        )
    return text


def _positive_dimension(value: Any, *, field_name: str) -> int | float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise DirectedTransformError(f"{field_name} must be a positive number.")
    normalized = value.item() if isinstance(value, np.generic) else value
    if not math.isfinite(float(normalized)) or float(normalized) <= 0:
        raise DirectedTransformError(f"{field_name} must be a positive finite number.")
    return normalized


def _parse_reference_extent(
    value: Any,
    *,
    field_name: str,
    space_id: str,
) -> TransformReferenceExtent:
    if isinstance(value, TransformReferenceExtent):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        raise DirectedTransformError(f"{field_name} must be a mapping.")
    fields = frozenset(value)
    if fields != _REFERENCE_FIELDS:
        missing = sorted(_REFERENCE_FIELDS - fields)
        unknown = sorted(fields - _REFERENCE_FIELDS)
        raise DirectedTransformError(
            f"{field_name} fields are invalid; missing={missing}, unknown={unknown}."
        )
    width = _positive_dimension(value["width"], field_name=f"{field_name}.width")
    height = _positive_dimension(value["height"], field_name=f"{field_name}.height")
    units = _required_text(value["units"], field_name=f"{field_name}.units")
    if units not in REFERENCE_UNITS or units == "not_applicable":
        raise DirectedTransformError(
            f"{field_name}.units must describe a concrete 2-D extent."
        )
    authority = _required_text(
        value["authority"],
        field_name=f"{field_name}.authority",
    )
    if space_id in PIXEL_SPACE_IDS | NORMALIZED_SPACE_IDS and units != "px":
        raise DirectedTransformError(
            f"{field_name}.units must be 'px' for coordinate space {space_id!r}."
        )
    if space_id == "physical_mm" and units != "mm":
        raise DirectedTransformError(
            f"{field_name}.units must be 'mm' for coordinate space 'physical_mm'."
        )
    return TransformReferenceExtent(
        width=width,
        height=height,
        units=units,
        authority=authority,
    )


def _parse_source_transform(value: Any) -> SourceTransformReference:
    if isinstance(value, SourceTransformReference):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        raise DirectedTransformError("source_transform must be a mapping.")
    fields = frozenset(value)
    if fields != _SOURCE_TRANSFORM_FIELDS:
        missing = sorted(_SOURCE_TRANSFORM_FIELDS - fields)
        unknown = sorted(fields - _SOURCE_TRANSFORM_FIELDS)
        raise DirectedTransformError(
            "source_transform fields are invalid; "
            f"missing={missing}, unknown={unknown}."
        )
    relationship = _required_text(
        value["relationship"],
        field_name="source_transform.relationship",
    )
    if relationship != "inverse_of":
        raise DirectedTransformError(
            "source_transform.relationship must be 'inverse_of'."
        )
    return SourceTransformReference(
        relationship=relationship,
        transform_id=_required_text(
            value["transform_id"],
            field_name="source_transform.transform_id",
        ),
        transform_sha256=_sha256(
            value["transform_sha256"],
            field_name="source_transform.transform_sha256",
        ),
    )


def _load_json_without_duplicate_keys(value: str) -> Any:
    """Parse transform JSON without last-key-wins ambiguity at any depth."""

    def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for name, item in pairs:
            if name in payload:
                raise DirectedTransformError(
                    f"Directed-transform JSON contains duplicate key {name!r}."
                )
            payload[name] = item
        return payload

    try:
        return json.loads(value, object_pairs_hook=reject_duplicate_pairs)
    except json.JSONDecodeError as exc:
        raise DirectedTransformError(
            f"Directed-transform JSON is invalid: {exc.msg}."
        ) from exc


def _payload_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, DirectedHomography):
        return value.to_dict()
    if isinstance(value, (bytes, bytearray)):
        try:
            value = bytes(value).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DirectedTransformError(
                "Directed-transform bytes must be UTF-8."
            ) from exc
    if isinstance(value, str):
        value = _load_json_without_duplicate_keys(value)
    if not isinstance(value, Mapping):
        raise DirectedTransformError(
            "Directed-transform metadata must be a mapping or JSON object."
        )
    return value


def parse_directed_homography(value: Any) -> DirectedHomography:
    """Strictly parse canonical metadata without loading or interpreting a matrix."""

    payload = _payload_mapping(value)
    fields = frozenset(payload)
    allowed = _REQUIRED_FIELDS | _OPTIONAL_FIELDS
    missing = sorted(_REQUIRED_FIELDS - fields)
    unknown = sorted(fields - allowed)
    if missing or unknown:
        raise DirectedTransformError(
            f"Directed-transform fields are invalid; missing={missing}, unknown={unknown}."
        )
    if payload["schema_id"] != DIRECTED_TRANSFORM_SCHEMA_ID:
        raise DirectedTransformError("Unsupported directed-transform schema_id.")
    if payload["schema_version"] != DIRECTED_TRANSFORM_SCHEMA_VERSION:
        raise DirectedTransformError("Unsupported directed-transform schema_version.")
    if payload["kind"] != DIRECTED_TRANSFORM_KIND:
        raise DirectedTransformError("Directed transform kind must be 'homography'.")
    if payload["direction"] != DIRECTED_TRANSFORM_DIRECTION:
        raise DirectedTransformError(
            "Directed transform direction must be 'source_to_target'."
        )

    from_space_id = _required_text(
        payload["from_space_id"],
        field_name="from_space_id",
    )
    to_space_id = _required_text(
        payload["to_space_id"],
        field_name="to_space_id",
    )
    for field_name, space_id in (
        ("from_space_id", from_space_id),
        ("to_space_id", to_space_id),
    ):
        if space_id not in COORDINATE_SPACE_IDS:
            raise DirectedTransformError(
                f"{field_name} has unsupported coordinate space {space_id!r}."
            )

    camera_id = None
    if "camera_id" in payload:
        camera_id = _required_text(payload["camera_id"], field_name="camera_id")
    if (
        from_space_id in CAMERA_BOUND_SPACE_IDS
        or to_space_id in CAMERA_BOUND_SPACE_IDS
    ) and camera_id is None:
        raise DirectedTransformError(
            "camera_id is required when a directed transform has a source-camera side."
        )
    source_transform = None
    if "source_transform" in payload:
        source_transform = _parse_source_transform(payload["source_transform"])

    return DirectedHomography(
        transform_id=_required_text(payload["transform_id"], field_name="transform_id"),
        from_space_id=from_space_id,
        to_space_id=to_space_id,
        source_reference_extent=_parse_reference_extent(
            payload["source_reference_extent"],
            field_name="source_reference_extent",
            space_id=from_space_id,
        ),
        target_reference_extent=_parse_reference_extent(
            payload["target_reference_extent"],
            field_name="target_reference_extent",
            space_id=to_space_id,
        ),
        calibration_ref=_canonical_node_path(
            payload["calibration_ref"],
            field_name="calibration_ref",
        ),
        matrix_sha256=_sha256(
            payload["matrix_sha256"],
            field_name="matrix_sha256",
        ),
        camera_id=camera_id,
        source_transform=source_transform,
    )


def serialize_directed_homography(value: Any) -> dict[str, Any]:
    """Return a strictly validated JSON-compatible metadata mapping."""

    return parse_directed_homography(value).to_dict()


def canonical_directed_transform_json(value: Any) -> str:
    """Serialize metadata deterministically for persistence and hashing."""

    return json.dumps(
        serialize_directed_homography(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def directed_transform_digest(value: Any) -> str:
    """Return the SHA-256 digest of canonical transform metadata."""

    canonical = canonical_directed_transform_json(value).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def directed_homography_attrs(
    value: Any,
    *,
    attr_name: str = DIRECTED_TRANSFORM_ATTR,
) -> dict[str, Any]:
    """Return the canonical metadata attr and its independent content digest."""

    name = _required_text(attr_name, field_name="attr_name")
    transform = parse_directed_homography(value)
    return {
        name: transform.to_dict(),
        f"{name}{DIRECTED_TRANSFORM_DIGEST_SUFFIX}": transform.digest(),
    }


def load_directed_homography_attrs(
    attrs: Mapping[str, Any],
    *,
    attr_name: str = DIRECTED_TRANSFORM_ATTR,
) -> DirectedHomography:
    """Load transform attrs and reject missing, malformed, or stale metadata."""

    if not isinstance(attrs, Mapping):
        raise DirectedTransformError("Transform attrs must be a mapping.")
    name = _required_text(attr_name, field_name="attr_name")
    digest_name = f"{name}{DIRECTED_TRANSFORM_DIGEST_SUFFIX}"
    if name not in attrs:
        raise DirectedTransformError(f"Directed-transform attr {name!r} is missing.")
    if digest_name not in attrs:
        raise DirectedTransformError(
            f"Directed-transform digest attr {digest_name!r} is missing."
        )
    transform = parse_directed_homography(attrs[name])
    stored_digest = _sha256(
        attrs[digest_name],
        field_name=digest_name,
    )
    if stored_digest != transform.digest():
        raise DirectedTransformError(
            "Stored directed-transform digest does not match canonical metadata."
        )
    return transform


def validate_homography_matrix(matrix: Any) -> np.ndarray:
    """Return a float64 3x3 finite, nonsingular matrix or raise."""

    try:
        values = np.asarray(matrix, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DirectedTransformError(
            "Homography matrix must contain numeric values."
        ) from exc
    if values.shape != (3, 3):
        raise DirectedTransformError("Homography matrix must have shape (3, 3).")
    if not np.isfinite(values).all():
        raise DirectedTransformError("Homography matrix must contain only finite values.")
    if int(np.linalg.matrix_rank(values)) != 3:
        raise DirectedTransformError("Homography matrix must be nonsingular.")
    return np.array(values, dtype=np.float64, order="C", copy=True)


def homography_matrix_sha256(matrix: Any) -> str:
    """Digest the validated matrix using canonical little-endian float64 bytes."""

    values = validate_homography_matrix(matrix).astype("<f8", copy=False)
    prefix = f"{HOMOGRAPHY_MATRIX_CANONICALIZATION}\0".encode("ascii")
    return hashlib.sha256(prefix + values.tobytes(order="C")).hexdigest()


def _canonical_node_path(value: Any, *, field_name: str) -> str:
    text = _required_text(value, field_name=field_name)
    normalized = text.strip("/")
    parts = normalized.split("/")
    if (
        text != normalized
        or not normalized
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise DirectedTransformError(
            f"{field_name} must be a canonical archive-relative path."
        )
    return normalized


def _resolve_array_path(node: Any, *, array_path: str | None) -> str:
    explicit = (
        _canonical_node_path(array_path, field_name="array_path")
        if array_path is not None
        else None
    )
    raw_node_path = getattr(node, "path", None)
    persisted = (
        _canonical_node_path(raw_node_path, field_name="node.path")
        if raw_node_path is not None and str(raw_node_path).strip("/")
        else None
    )
    if explicit is None and persisted is None:
        raise DirectedTransformError(
            "An exact array_path is required when the node does not expose one."
        )
    if explicit is not None and persisted is not None and explicit != persisted:
        raise DirectedTransformError(
            f"Array path mismatch: requested {explicit!r}, node exposes {persisted!r}."
        )
    if explicit is not None:
        return explicit
    assert persisted is not None
    return persisted


def _read_homography_array(node: Any) -> np.ndarray:
    try:
        raw = node[:]
    except Exception as exc:
        raise DirectedTransformError("Unable to read persisted homography matrix.") from exc
    return validate_homography_matrix(raw)


def _readonly_matrix(matrix: Any) -> np.ndarray:
    values = validate_homography_matrix(matrix)
    values.setflags(write=False)
    return values


def stamp_directed_homography(
    node: Any,
    value: Any,
    *,
    array_path: str | None = None,
    attr_name: str = DIRECTED_TRANSFORM_ATTR,
) -> BoundDirectedHomography:
    """Validate an existing matrix array and stamp its canonical transform attrs."""

    attrs = getattr(node, "attrs", None)
    if attrs is None or not hasattr(attrs, "update"):
        raise TypeError("node must expose a mutable attrs mapping.")
    path = _resolve_array_path(node, array_path=array_path)
    matrix = _read_homography_array(node)
    transform = parse_directed_homography(value)
    if homography_matrix_sha256(matrix) != transform.matrix_sha256:
        raise DirectedTransformError(
            "Homography matrix digest does not match directed-transform metadata."
        )
    stamped = directed_homography_attrs(transform, attr_name=attr_name)
    attrs.update(stamped)
    digest = transform.digest()
    return BoundDirectedHomography(
        array_path=path,
        matrix=_readonly_matrix(matrix),
        transform=transform,
        transform_sha256=digest,
    )


def load_bound_directed_homography(
    node: Any,
    *,
    array_path: str | None = None,
    expected_from_space_id: str,
    expected_to_space_id: str,
    expected_camera_id: str | None,
    expected_source_reference_extent: TransformReferenceExtent | Mapping[str, Any],
    expected_target_reference_extent: TransformReferenceExtent | Mapping[str, Any],
    expected_transform_sha256: str,
    expected_calibration_ref: str,
    attr_name: str = DIRECTED_TRANSFORM_ATTR,
) -> BoundDirectedHomography:
    """Load one exact matrix/metadata binding and validate every caller expectation."""

    path = _resolve_array_path(node, array_path=array_path)
    attrs = getattr(node, "attrs", None)
    if not isinstance(attrs, Mapping):
        raise DirectedTransformError("Homography array must expose an attrs mapping.")
    transform = load_directed_homography_attrs(attrs, attr_name=attr_name)

    expected_from = _required_text(
        expected_from_space_id,
        field_name="expected_from_space_id",
    )
    expected_to = _required_text(
        expected_to_space_id,
        field_name="expected_to_space_id",
    )
    if (
        transform.from_space_id != expected_from
        or transform.to_space_id != expected_to
    ):
        raise DirectedTransformError(
            "Transform direction mismatch: "
            f"metadata maps {transform.from_space_id!r} to {transform.to_space_id!r}, "
            f"loader expected {expected_from!r} to {expected_to!r}."
        )

    normalized_camera_id = (
        _required_text(expected_camera_id, field_name="expected_camera_id")
        if expected_camera_id is not None
        else None
    )
    if transform.camera_id != normalized_camera_id:
        raise DirectedTransformError(
            "Transform camera mismatch: "
            f"metadata names {transform.camera_id!r}, loader expected "
            f"{normalized_camera_id!r}."
        )

    expected_source = _application_extent(
        expected_source_reference_extent,
        field_name="expected_source_reference_extent",
        space_id=expected_from,
    )
    expected_target = _application_extent(
        expected_target_reference_extent,
        field_name="expected_target_reference_extent",
        space_id=expected_to,
    )
    if transform.source_reference_extent != expected_source:
        raise DirectedTransformError("Source reference extent mismatch.")
    if transform.target_reference_extent != expected_target:
        raise DirectedTransformError("Target reference extent mismatch.")

    digest = transform.digest()
    expected_digest = _sha256(
        expected_transform_sha256,
        field_name="expected_transform_sha256",
    )
    if digest != expected_digest:
        raise DirectedTransformError(
            "Directed-transform digest does not match the selected pointer."
        )
    calibration_ref = _canonical_node_path(
        expected_calibration_ref,
        field_name="expected_calibration_ref",
    )
    if transform.calibration_ref != calibration_ref:
        raise DirectedTransformError(
            "Transform calibration_ref does not match the selected calibration group."
        )

    matrix = _read_homography_array(node)
    if homography_matrix_sha256(matrix) != transform.matrix_sha256:
        raise DirectedTransformError(
            "Homography matrix digest does not match directed-transform metadata."
        )
    return BoundDirectedHomography(
        array_path=path,
        matrix=_readonly_matrix(matrix),
        transform=transform,
        transform_sha256=digest,
    )


def build_directed_homography(
    *,
    transform_id: str,
    matrix: Any,
    from_space_id: str,
    to_space_id: str,
    source_reference_extent: TransformReferenceExtent | Mapping[str, Any],
    target_reference_extent: TransformReferenceExtent | Mapping[str, Any],
    calibration_ref: str,
    camera_id: str | None = None,
    source_transform: SourceTransformReference | Mapping[str, Any] | None = None,
) -> DirectedHomography:
    """Build strictly validated metadata bound to the exact provided matrix."""

    payload: dict[str, Any] = {
        "schema_id": DIRECTED_TRANSFORM_SCHEMA_ID,
        "schema_version": DIRECTED_TRANSFORM_SCHEMA_VERSION,
        "transform_id": transform_id,
        "kind": DIRECTED_TRANSFORM_KIND,
        "from_space_id": from_space_id,
        "to_space_id": to_space_id,
        "direction": DIRECTED_TRANSFORM_DIRECTION,
        "source_reference_extent": (
            source_reference_extent.to_dict()
            if isinstance(source_reference_extent, TransformReferenceExtent)
            else source_reference_extent
        ),
        "target_reference_extent": (
            target_reference_extent.to_dict()
            if isinstance(target_reference_extent, TransformReferenceExtent)
            else target_reference_extent
        ),
        "calibration_ref": calibration_ref,
        "matrix_sha256": homography_matrix_sha256(matrix),
    }
    if camera_id is not None:
        payload["camera_id"] = camera_id
    if source_transform is not None:
        payload["source_transform"] = (
            source_transform.to_dict()
            if isinstance(source_transform, SourceTransformReference)
            else source_transform
        )
    return parse_directed_homography(payload)


def _application_extent(
    value: TransformReferenceExtent | Mapping[str, Any] | None,
    *,
    field_name: str,
    space_id: str,
) -> TransformReferenceExtent:
    if value is None:
        raise DirectedTransformError(
            f"{field_name} is required to validate transform reference dimensions."
        )
    return _parse_reference_extent(
        value,
        field_name=field_name,
        space_id=space_id,
    )


def apply_directed_homography(
    points_xy: Any,
    matrix: Any,
    transform: Any,
    *,
    from_space_id: str,
    to_space_id: str,
    source_reference_extent: TransformReferenceExtent | Mapping[str, Any] | None,
    target_reference_extent: TransformReferenceExtent | Mapping[str, Any] | None,
    w_epsilon: float = 1e-12,
) -> np.ndarray:
    """Apply a homography only when direction, extents, and matrix digest match."""

    metadata = parse_directed_homography(transform)
    expected_from = _required_text(from_space_id, field_name="from_space_id")
    expected_to = _required_text(to_space_id, field_name="to_space_id")
    if metadata.from_space_id != expected_from or metadata.to_space_id != expected_to:
        raise DirectedTransformError(
            "Transform direction mismatch: "
            f"metadata maps {metadata.from_space_id!r} to {metadata.to_space_id!r}, "
            f"application requested {expected_from!r} to {expected_to!r}."
        )

    expected_source = _application_extent(
        source_reference_extent,
        field_name="source_reference_extent",
        space_id=expected_from,
    )
    expected_target = _application_extent(
        target_reference_extent,
        field_name="target_reference_extent",
        space_id=expected_to,
    )
    if expected_source != metadata.source_reference_extent:
        raise DirectedTransformError("Source reference extent mismatch.")
    if expected_target != metadata.target_reference_extent:
        raise DirectedTransformError("Target reference extent mismatch.")

    values = validate_homography_matrix(matrix)
    if homography_matrix_sha256(values) != metadata.matrix_sha256:
        raise DirectedTransformError(
            "Homography matrix digest does not match directed-transform metadata."
        )
    if not math.isfinite(float(w_epsilon)) or float(w_epsilon) <= 0:
        raise DirectedTransformError("w_epsilon must be positive and finite.")

    try:
        points = np.asarray(points_xy, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DirectedTransformError("Points must contain numeric XY values.") from exc
    if points.ndim < 1 or points.shape[-1] != 2:
        raise DirectedTransformError("Points must have shape (..., 2).")
    if not np.isfinite(points).all():
        raise DirectedTransformError("Points must contain only finite values.")

    original_shape = points.shape
    flat = points.reshape(-1, 2)
    homogeneous = np.column_stack(
        (flat, np.ones(flat.shape[0], dtype=np.float64))
    )
    projected = (values @ homogeneous.T).T
    w = projected[:, 2]
    if np.any(np.abs(w) <= float(w_epsilon)):
        raise DirectedTransformError(
            "Homography application produced near-zero homogeneous w."
        )
    result = projected[:, :2] / w[:, None]
    if not np.isfinite(result).all():
        raise DirectedTransformError("Homography application produced non-finite output.")
    return result.reshape(original_shape)


def invert_directed_homography(
    transform: Any,
    matrix: Any,
    *,
    transform_id: str,
) -> tuple[DirectedHomography, np.ndarray]:
    """Return an explicit inverse with swapped spaces/extents and source lineage."""

    source = parse_directed_homography(transform)
    source_matrix = validate_homography_matrix(matrix)
    if homography_matrix_sha256(source_matrix) != source.matrix_sha256:
        raise DirectedTransformError(
            "Homography matrix digest does not match directed-transform metadata."
        )
    try:
        inverse_matrix = np.linalg.inv(source_matrix)
    except np.linalg.LinAlgError as exc:  # Defensive after rank validation.
        raise DirectedTransformError("Homography matrix is not invertible.") from exc
    inverse_matrix = validate_homography_matrix(inverse_matrix)
    source_ref = SourceTransformReference(
        relationship="inverse_of",
        transform_id=source.transform_id,
        transform_sha256=source.digest(),
    )
    inverse = build_directed_homography(
        transform_id=transform_id,
        matrix=inverse_matrix,
        from_space_id=source.to_space_id,
        to_space_id=source.from_space_id,
        source_reference_extent=source.target_reference_extent,
        target_reference_extent=source.source_reference_extent,
        calibration_ref=source.calibration_ref,
        camera_id=source.camera_id,
        source_transform=source_ref,
    )
    return inverse, inverse_matrix


__all__ = [
    "DIRECTED_TRANSFORM_SCHEMA_ID",
    "DIRECTED_TRANSFORM_SCHEMA_VERSION",
    "DIRECTED_TRANSFORM_KIND",
    "DIRECTED_TRANSFORM_DIRECTION",
    "DIRECTED_TRANSFORM_ATTR",
    "DIRECTED_TRANSFORM_DIGEST_SUFFIX",
    "DIRECTED_TRANSFORM_CANONICALIZATION",
    "HOMOGRAPHY_MATRIX_CANONICALIZATION",
    "CAMERA_BOUND_SPACE_IDS",
    "DirectedTransformError",
    "TransformReferenceExtent",
    "SourceTransformReference",
    "DirectedHomography",
    "BoundDirectedHomography",
    "parse_directed_homography",
    "serialize_directed_homography",
    "canonical_directed_transform_json",
    "directed_transform_digest",
    "load_directed_homography_attrs",
    "validate_homography_matrix",
    "homography_matrix_sha256",
    "load_bound_directed_homography",
    "apply_directed_homography",
]
