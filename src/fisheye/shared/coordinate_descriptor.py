"""Compact, versioned coordinate semantics for persisted array surfaces.

The descriptor in this module intentionally answers only the questions needed
to interpret one array: which coordinate space it uses, how its components are
ordered, which reference extent defines that space, and how rows are
identified.  Potentially large or mutable crop, calibration, transform, and
source provenance records remain separate artifacts and are linked through
``lineage_refs`` and ``transform_refs``.

Historical labels such as ``camera`` and ``texture`` are not canonical space
identifiers.  They can be resolved only through :func:`resolve_legacy_space_id`
when the caller supplies explicit dimensions, authority, and evidence.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

from fisheye.shared.coordinate_identity import (
    ROW_IDENTITY_CONTRACT_ATTR,
    RowIdentityContract,
    RowIdentityContractError,
    parse_row_identity_contract,
)


COORDINATE_DESCRIPTOR_ATTR = "coordinate_descriptor"
COORDINATE_DESCRIPTOR_DIGEST_SUFFIX = "_sha256"
COORDINATE_DESCRIPTOR_SCHEMA_ID = "palette.coordinate_descriptor"
COORDINATE_DESCRIPTOR_SCHEMA_VERSION = 1
COORDINATE_DESCRIPTOR_CANONICALIZATION = "canonical_json_sort_keys_v1"

COORDINATE_SPACE_IDS = frozenset(
    {
        "source_camera_image_px",
        "source_camera_normalized_xy",
        "detector_model_input_px",
        "detector_normalized_xy",
        "roi_local_px",
        "stimulus_texture_px",
        "stimulus_canvas_px",
        "projector_px",
        "arena_relative_canvas_px",
        "physical_mm",
        "fish_anatomical_body_frame",
    }
)

PIXEL_SPACE_IDS = frozenset(
    {
        "source_camera_image_px",
        "detector_model_input_px",
        "roi_local_px",
        "stimulus_texture_px",
        "stimulus_canvas_px",
        "projector_px",
        "arena_relative_canvas_px",
    }
)
NORMALIZED_SPACE_IDS = frozenset(
    {"source_camera_normalized_xy", "detector_normalized_xy"}
)

GEOMETRY_TYPES = frozenset(
    {
        "point_xy",
        "points_xy",
        "bbox_xyxy",
        "bbox_xywh",
        "bbox_cxcywh",
        "raster_yx",
        "polyline_xy",
        "polygon_xy",
        "vector_xy",
        "vector_sequence_xy",
        "line_segment_xyxy",
        "circle_cxcy_r",
        "ellipse_cxcy_wh_angle",
        "distance",
        "coordinate_component",
    }
)

_GEOMETRY_COMPONENTS: dict[str, tuple[str, ...]] = {
    "point_xy": ("x", "y"),
    "points_xy": ("x", "y"),
    "bbox_xyxy": ("x_min", "y_min", "x_max", "y_max"),
    "bbox_xywh": ("x", "y", "width", "height"),
    "bbox_cxcywh": ("center_x", "center_y", "width", "height"),
    "raster_yx": ("y", "x"),
    "polyline_xy": ("x", "y"),
    "polygon_xy": ("x", "y"),
    "vector_xy": ("x", "y"),
    "vector_sequence_xy": ("x", "y"),
    "line_segment_xyxy": ("x0", "y0", "x1", "y1"),
    "circle_cxcy_r": ("center_x", "center_y", "radius"),
    "ellipse_cxcy_wh_angle": (
        "center_x",
        "center_y",
        "width",
        "height",
        "angle",
    ),
    "distance": ("distance",),
}

COMPONENT_UNITS = frozenset(
    {"px", "normalized", "mm", "unitless", "deg", "rad"}
)
REFERENCE_UNITS = frozenset({"px", "mm", "not_applicable"})
ORIGINS = frozenset(
    {
        "top_left",
        "arena_top_left",
        "projector_top_left",
        "physical_frame_origin",
        "body_frame_origin",
        "row_varying_reference",
        "not_applicable",
    }
)
POSITIVE_DIRECTIONS = frozenset(
    {
        "right",
        "left",
        "up",
        "down",
        "anterior",
        "posterior",
        "anatomical_left",
        "anatomical_right",
        "increasing",
        "decreasing",
        "not_applicable",
    }
)
PIXEL_CONVENTIONS = frozenset(
    {
        "pixel_center",
        "pixel_edge",
        "pixel_edge_half_open",
        "continuous",
        "not_applicable",
    }
)
ROW_IDENTITY_MODES = frozenset(
    {
        "instance_key",
        "source_crop_row_ids",
        "frame_indices",
        "track_frame_indices",
        "sample_indices",
        "explicit_array",
        "not_applicable",
    }
)
SOURCE_CAMERA_OVERLAY_STATUSES = frozenset(
    {"direct", "requires_transform", "not_suitable", "unknown"}
)

_FORBIDDEN_PRESENTATION_SPACE_TOKENS = (
    "viewport",
    "display",
    "renderer",
    "render_",
    "screen",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class CoordinateIssue:
    """One stable, machine-classifiable descriptor validation issue."""

    code: str
    path: str
    message: str


class CoordinateDescriptorError(ValueError):
    """Raised when coordinate metadata is missing, ambiguous, or invalid."""

    def __init__(self, issues: Sequence[CoordinateIssue]):
        normalized = tuple(issues)
        if not normalized:
            normalized = (
                CoordinateIssue(
                    code="descriptor_invalid",
                    path="$",
                    message="Coordinate descriptor is invalid.",
                ),
            )
        self.issues = normalized
        super().__init__(
            "; ".join(
                f"{issue.code} at {issue.path}: {issue.message}"
                for issue in self.issues
            )
        )


@dataclass(frozen=True)
class PositiveDirections:
    x: str
    y: str

    def to_dict(self) -> dict[str, str]:
        return {"x": self.x, "y": self.y}


@dataclass(frozen=True)
class ReferenceExtent:
    """The authoritative 2-D extent against which coordinates are defined."""

    width: int | float | None
    height: int | float | None
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
class RowIdentityReference:
    """Reference to the array that identifies the descriptor's leading rows.

    Relative references are resolved from the descriptor owner's parent group
    when the owner is an array, and from the owner itself when it is a group.
    This keeps sibling row arrays such as ``frame_indices`` addressable with the
    same compact reference at every array surface.
    """

    mode: str
    array_ref: str | None

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "array_ref": self.array_ref}


@dataclass(frozen=True)
class CoordinateRecordRef:
    """Compact reference to a separate lineage or directed-transform record."""

    ref: str
    sha256: str | None = None

    def to_dict(self) -> dict[str, str]:
        payload = {"ref": self.ref}
        if self.sha256 is not None:
            payload["sha256"] = self.sha256
        return payload


@dataclass(frozen=True)
class CoordinateDescriptor:
    """Canonical interpretation contract for one persisted array surface."""

    space_id: str
    geometry_type: str
    components: tuple[str, ...]
    component_units: tuple[str, ...]
    origin: str
    positive_directions: PositiveDirections
    reference_extent: ReferenceExtent
    pixel_convention: str
    row_identity: RowIdentityReference
    source_camera_overlay: str
    legacy_space_label: str | None = None
    physical_frame: str | None = None
    lineage_refs: tuple[CoordinateRecordRef, ...] = ()
    transform_refs: tuple[CoordinateRecordRef, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_id": COORDINATE_DESCRIPTOR_SCHEMA_ID,
            "schema_version": COORDINATE_DESCRIPTOR_SCHEMA_VERSION,
            "space_id": self.space_id,
            "geometry_type": self.geometry_type,
            "components": list(self.components),
            "component_units": list(self.component_units),
            "origin": self.origin,
            "positive_directions": self.positive_directions.to_dict(),
            "reference_extent": self.reference_extent.to_dict(),
            "pixel_convention": self.pixel_convention,
            "row_identity": self.row_identity.to_dict(),
            "source_camera_overlay": self.source_camera_overlay,
        }
        if self.legacy_space_label is not None:
            payload["legacy_space_label"] = self.legacy_space_label
        if self.physical_frame is not None:
            payload["physical_frame"] = self.physical_frame
        if self.lineage_refs:
            payload["lineage_refs"] = [item.to_dict() for item in self.lineage_refs]
        if self.transform_refs:
            payload["transform_refs"] = [item.to_dict() for item in self.transform_refs]
        return payload

    def canonical_json(self) -> str:
        return canonical_coordinate_descriptor_json(self)

    def digest(self) -> str:
        return coordinate_descriptor_digest(self)

    def to_attrs(
        self,
        *,
        attr_name: str = COORDINATE_DESCRIPTOR_ATTR,
    ) -> dict[str, Any]:
        """Return historical-v1 attrs; retained only for explicit legacy objects."""

        return _historical_coordinate_descriptor_v1_attrs(
            self,
            attr_name=attr_name,
        )


@dataclass(frozen=True)
class LegacySpaceContext:
    """Explicit evidence required to interpret one historical space label."""

    canonical_space_id: str
    reference_width: int | float
    reference_height: int | float
    reference_units: str
    reference_authority: str
    evidence_refs: tuple[CoordinateRecordRef, ...]


_REQUIRED_DESCRIPTOR_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "space_id",
        "geometry_type",
        "components",
        "component_units",
        "origin",
        "positive_directions",
        "reference_extent",
        "pixel_convention",
        "row_identity",
        "source_camera_overlay",
    }
)
_OPTIONAL_DESCRIPTOR_FIELDS = frozenset(
    {"legacy_space_label", "physical_frame", "lineage_refs", "transform_refs"}
)


def _issue(code: str, path: str, message: str) -> CoordinateIssue:
    return CoordinateIssue(code=code, path=path, message=message)


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _exact_json_equal(left: Any, right: Any) -> bool:
    """Compare persisted JSON without bool/int or int/float coercion."""

    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return (
            type(left) is type(right)
            and set(left) == set(right)
            and all(_exact_json_equal(left[name], right[name]) for name in left)
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _exact_json_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return type(left) is type(right) and left == right


def _payload_mapping(value: Any) -> tuple[Mapping[str, Any] | None, list[CoordinateIssue]]:
    if isinstance(value, CoordinateDescriptor):
        return value.to_dict(), []
    if isinstance(value, (bytes, bytearray)):
        try:
            value = bytes(value).decode("utf-8")
        except UnicodeDecodeError:
            return None, [
                _issue("descriptor_json_invalid", "$", "Descriptor bytes are not UTF-8.")
            ]
    if isinstance(value, str):
        def reject_duplicate_pairs(
            pairs: list[tuple[str, Any]],
        ) -> dict[str, Any]:
            payload: dict[str, Any] = {}
            for name, item in pairs:
                if name in payload:
                    raise ValueError(f"duplicate JSON key {name!r}")
                payload[name] = item
            return payload

        try:
            value = json.loads(value, object_pairs_hook=reject_duplicate_pairs)
        except (json.JSONDecodeError, ValueError) as exc:
            return None, [
                _issue(
                    "descriptor_json_invalid",
                    "$",
                    f"Descriptor JSON is invalid: {exc}.",
                )
            ]
    if not isinstance(value, Mapping):
        return None, [
            _issue("descriptor_not_mapping", "$", "Descriptor must be a mapping or JSON object.")
        ]
    return value, []


def _required_text(
    payload: Mapping[str, Any],
    name: str,
    issues: list[CoordinateIssue],
    *,
    path: str | None = None,
) -> str | None:
    value = payload.get(name)
    issue_path = path or f"$.{name}"
    if not isinstance(value, str) or not value.strip():
        issues.append(_issue("invalid_text", issue_path, "Value must be a non-empty string."))
        return None
    if value != value.strip():
        issues.append(_issue("noncanonical_text", issue_path, "Value must not have surrounding whitespace."))
        return None
    return value


def _optional_text(
    payload: Mapping[str, Any],
    name: str,
    issues: list[CoordinateIssue],
) -> str | None:
    if name not in payload:
        return None
    value = payload[name]
    if not isinstance(value, str) or not value.strip():
        issues.append(_issue("invalid_text", f"$.{name}", "Value must be a non-empty string."))
        return None
    if value != value.strip():
        issues.append(_issue("noncanonical_text", f"$.{name}", "Value must not have surrounding whitespace."))
        return None
    return value


def _dimension(
    value: Any,
    *,
    path: str,
    issues: list[CoordinateIssue],
) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        issues.append(_issue("reference_extent_invalid", path, "Dimension must be a positive finite number or null."))
        return None
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0:
        issues.append(_issue("reference_extent_invalid", path, "Dimension must be positive and finite."))
        return None
    return int(numeric) if numeric.is_integer() else numeric


def _parse_positive_directions(
    value: Any,
    issues: list[CoordinateIssue],
) -> PositiveDirections | None:
    path = "$.positive_directions"
    if not isinstance(value, Mapping):
        issues.append(_issue("positive_directions_invalid", path, "Value must be a mapping with x and y."))
        return None
    expected = {"x", "y"}
    for name in sorted(expected - set(value)):
        issues.append(_issue("missing_field", f"{path}.{name}", "Required field is missing."))
    for name in sorted(set(value) - expected):
        issues.append(_issue("unknown_field", f"{path}.{name}", "Field is not part of schema version 1."))
    x = value.get("x")
    y = value.get("y")
    for name, direction in (("x", x), ("y", y)):
        if direction not in POSITIVE_DIRECTIONS:
            issues.append(
                _issue(
                    "positive_direction_unsupported",
                    f"{path}.{name}",
                    f"Unsupported positive direction {direction!r}.",
                )
            )
    if x not in POSITIVE_DIRECTIONS or y not in POSITIVE_DIRECTIONS:
        return None
    return PositiveDirections(x=str(x), y=str(y))


def _parse_reference_extent(
    value: Any,
    issues: list[CoordinateIssue],
) -> ReferenceExtent | None:
    path = "$.reference_extent"
    if not isinstance(value, Mapping):
        issues.append(_issue("reference_extent_invalid", path, "Value must be a mapping."))
        return None
    expected = {"width", "height", "units", "authority"}
    for name in sorted(expected - set(value)):
        issues.append(_issue("missing_field", f"{path}.{name}", "Required field is missing."))
    for name in sorted(set(value) - expected):
        issues.append(_issue("unknown_field", f"{path}.{name}", "Field is not part of schema version 1."))
    width = _dimension(value.get("width"), path=f"{path}.width", issues=issues)
    height = _dimension(value.get("height"), path=f"{path}.height", issues=issues)
    if (value.get("width") is None) != (value.get("height") is None):
        issues.append(
            _issue(
                "reference_extent_pair_required",
                path,
                "width and height must either both be numbers or both be null.",
            )
        )
    units = value.get("units")
    if units not in REFERENCE_UNITS:
        issues.append(_issue("reference_units_unsupported", f"{path}.units", f"Unsupported reference units {units!r}."))
    authority = value.get("authority")
    if not isinstance(authority, str) or not authority.strip() or authority != authority.strip():
        issues.append(_issue("reference_authority_invalid", f"{path}.authority", "Authority must be a non-empty trimmed reference."))
    if units not in REFERENCE_UNITS or not isinstance(authority, str) or not authority.strip() or authority != authority.strip():
        return None
    return ReferenceExtent(width=width, height=height, units=str(units), authority=authority)


def _parse_row_identity(
    value: Any,
    issues: list[CoordinateIssue],
) -> RowIdentityReference | None:
    path = "$.row_identity"
    if not isinstance(value, Mapping):
        issues.append(_issue("row_identity_invalid", path, "Value must be a mapping."))
        return None
    expected = {"mode", "array_ref"}
    for name in sorted(expected - set(value)):
        issues.append(_issue("missing_field", f"{path}.{name}", "Required field is missing."))
    for name in sorted(set(value) - expected):
        issues.append(_issue("unknown_field", f"{path}.{name}", "Field is not part of schema version 1."))
    mode = value.get("mode")
    if mode not in ROW_IDENTITY_MODES:
        issues.append(_issue("row_identity_mode_unsupported", f"{path}.mode", f"Unsupported row identity mode {mode!r}."))
    array_ref = value.get("array_ref")
    if mode == "not_applicable":
        if array_ref is not None:
            issues.append(_issue("row_identity_ref_forbidden", f"{path}.array_ref", "not_applicable row identity cannot name an array."))
    elif not isinstance(array_ref, str) or not array_ref.strip() or array_ref != array_ref.strip():
        issues.append(_issue("row_identity_ref_required", f"{path}.array_ref", "A non-empty trimmed array reference is required."))
    if mode not in ROW_IDENTITY_MODES:
        return None
    if mode == "not_applicable":
        return RowIdentityReference(mode=str(mode), array_ref=None)
    if not isinstance(array_ref, str) or not array_ref.strip() or array_ref != array_ref.strip():
        return None
    return RowIdentityReference(mode=str(mode), array_ref=array_ref)


def _parse_record_refs(
    value: Any,
    *,
    field_name: str,
    issues: list[CoordinateIssue],
) -> tuple[CoordinateRecordRef, ...]:
    path = f"$.{field_name}"
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        issues.append(_issue("record_refs_invalid", path, "Value must be a list of record references."))
        return ()
    parsed: list[CoordinateRecordRef] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        if not isinstance(item, Mapping):
            issues.append(_issue("record_ref_invalid", item_path, "Record reference must be a mapping."))
            continue
        expected = {"ref", "sha256"}
        if "ref" not in item:
            issues.append(_issue("missing_field", f"{item_path}.ref", "Required field is missing."))
        for name in sorted(set(item) - expected):
            issues.append(_issue("unknown_field", f"{item_path}.{name}", "Field is not part of a record reference."))
        ref = item.get("ref")
        if not isinstance(ref, str) or not ref.strip() or ref != ref.strip():
            issues.append(_issue("record_ref_invalid", f"{item_path}.ref", "Reference must be a non-empty trimmed string."))
            continue
        if ref in seen:
            issues.append(_issue("record_ref_duplicate", f"{item_path}.ref", f"Reference {ref!r} is duplicated."))
            continue
        seen.add(ref)
        raw_digest = item.get("sha256")
        digest: str | None
        if raw_digest is None:
            issues.append(
                _issue(
                    "record_digest_required",
                    f"{item_path}.sha256",
                    "Historical evidence is readable only when bound to a lowercase SHA-256 digest.",
                )
            )
            continue
        elif not isinstance(raw_digest, str) or _SHA256_RE.fullmatch(raw_digest) is None:
            issues.append(_issue("record_digest_invalid", f"{item_path}.sha256", "Digest must be 64 lowercase hexadecimal characters."))
            continue
        else:
            digest = raw_digest
        parsed.append(CoordinateRecordRef(ref=ref, sha256=digest))
    return tuple(parsed)


def _parse_descriptor(value: Any) -> CoordinateDescriptor:
    payload, initial_issues = _payload_mapping(value)
    issues = list(initial_issues)
    if payload is None:
        raise CoordinateDescriptorError(issues)

    for name in sorted(_REQUIRED_DESCRIPTOR_FIELDS - set(payload)):
        issues.append(_issue("missing_field", f"$.{name}", "Required field is missing."))
    for name in sorted(set(payload) - _REQUIRED_DESCRIPTOR_FIELDS - _OPTIONAL_DESCRIPTOR_FIELDS):
        issues.append(_issue("unknown_field", f"$.{name}", "Field is not part of schema version 1."))

    if payload.get("schema_id") != COORDINATE_DESCRIPTOR_SCHEMA_ID:
        issues.append(_issue("schema_id_unsupported", "$.schema_id", f"Expected {COORDINATE_DESCRIPTOR_SCHEMA_ID!r}."))
    version = payload.get("schema_version")
    if type(version) is not int or version != COORDINATE_DESCRIPTOR_SCHEMA_VERSION:
        issues.append(_issue("schema_version_unsupported", "$.schema_version", f"Expected version {COORDINATE_DESCRIPTOR_SCHEMA_VERSION}."))

    space_id = _required_text(payload, "space_id", issues)
    if space_id is not None:
        lowered = space_id.lower()
        if any(token in lowered for token in _FORBIDDEN_PRESENTATION_SPACE_TOKENS):
            issues.append(_issue("presentation_space_forbidden", "$.space_id", "Renderer viewport/display coordinates must remain ephemeral."))
        elif space_id not in COORDINATE_SPACE_IDS:
            issues.append(_issue("space_id_unsupported", "$.space_id", f"Unsupported coordinate space {space_id!r}."))

    geometry_type = _required_text(payload, "geometry_type", issues)
    if geometry_type is not None and geometry_type not in GEOMETRY_TYPES:
        issues.append(_issue("geometry_type_unsupported", "$.geometry_type", f"Unsupported geometry type {geometry_type!r}."))

    raw_components = payload.get("components")
    components: tuple[str, ...] = ()
    if not isinstance(raw_components, (list, tuple)) or not raw_components:
        issues.append(_issue("components_invalid", "$.components", "components must be a non-empty list."))
    else:
        component_values: list[str] = []
        for index, component in enumerate(raw_components):
            if not isinstance(component, str) or not component.strip() or component != component.strip():
                issues.append(_issue("component_invalid", f"$.components[{index}]", "Component must be a non-empty trimmed string."))
            else:
                component_values.append(component)
        components = tuple(component_values)
        if len(set(components)) != len(components):
            issues.append(_issue("component_duplicate", "$.components", "Component names must be unique."))

    raw_units = payload.get("component_units")
    component_units: tuple[str, ...] = ()
    if not isinstance(raw_units, (list, tuple)) or not raw_units:
        issues.append(_issue("component_units_invalid", "$.component_units", "component_units must be a non-empty list."))
    else:
        unit_values: list[str] = []
        for index, unit in enumerate(raw_units):
            if unit not in COMPONENT_UNITS:
                issues.append(_issue("component_unit_unsupported", f"$.component_units[{index}]", f"Unsupported component unit {unit!r}."))
            else:
                unit_values.append(str(unit))
        component_units = tuple(unit_values)
    if isinstance(raw_components, (list, tuple)) and isinstance(raw_units, (list, tuple)) and len(raw_components) != len(raw_units):
        issues.append(_issue("component_unit_count_mismatch", "$.component_units", "One component unit is required for every component."))

    if geometry_type in _GEOMETRY_COMPONENTS and components:
        expected_components = _GEOMETRY_COMPONENTS[geometry_type]
        if components != expected_components:
            issues.append(_issue("geometry_components_mismatch", "$.components", f"{geometry_type} requires components {list(expected_components)!r}."))
    elif geometry_type == "coordinate_component" and len(components) != 1:
        issues.append(_issue("geometry_components_mismatch", "$.components", "coordinate_component requires exactly one component."))

    origin = _required_text(payload, "origin", issues)
    if origin is not None and origin not in ORIGINS:
        issues.append(_issue("origin_unsupported", "$.origin", f"Unsupported origin {origin!r}."))
    directions = _parse_positive_directions(payload.get("positive_directions"), issues)
    extent = _parse_reference_extent(payload.get("reference_extent"), issues)

    pixel_convention = _required_text(payload, "pixel_convention", issues)
    if pixel_convention is not None and pixel_convention not in PIXEL_CONVENTIONS:
        issues.append(_issue("pixel_convention_unsupported", "$.pixel_convention", f"Unsupported pixel convention {pixel_convention!r}."))
    row_identity = _parse_row_identity(payload.get("row_identity"), issues)

    overlay = _required_text(payload, "source_camera_overlay", issues)
    if overlay is not None and overlay not in SOURCE_CAMERA_OVERLAY_STATUSES:
        issues.append(_issue("overlay_status_unsupported", "$.source_camera_overlay", f"Unsupported overlay status {overlay!r}."))
    if space_id not in (None, "source_camera_image_px") and overlay == "direct":
        issues.append(_issue("overlay_status_inconsistent", "$.source_camera_overlay", "Only source_camera_image_px is directly overlayable."))

    legacy_label = _optional_text(payload, "legacy_space_label", issues)
    if legacy_label is not None and legacy_label not in {"camera", "texture"}:
        issues.append(_issue("legacy_label_unsupported", "$.legacy_space_label", "Only historical camera and texture labels are recognized."))
    physical_frame = _optional_text(payload, "physical_frame", issues)
    if space_id == "physical_mm" and physical_frame is None:
        issues.append(_issue("physical_frame_required", "$.physical_frame", "physical_mm requires an explicit physical frame identifier."))

    lineage_refs = _parse_record_refs(payload.get("lineage_refs"), field_name="lineage_refs", issues=issues)
    transform_refs = _parse_record_refs(payload.get("transform_refs"), field_name="transform_refs", issues=issues)

    if legacy_label is not None:
        legacy_space = {
            "camera": "source_camera_image_px",
            "texture": "stimulus_texture_px",
        }.get(legacy_label)
        if legacy_space is not None and space_id != legacy_space:
            issues.append(
                _issue(
                    "legacy_label_space_mismatch",
                    "$.legacy_space_label",
                    f"Historical label {legacy_label!r} is incompatible with {space_id!r}.",
                )
            )
        if not lineage_refs:
            issues.append(
                _issue(
                    "legacy_label_evidence_required",
                    "$.lineage_refs",
                    "Preserving a legacy space label requires an explicit lineage evidence record.",
                )
            )

    if space_id in PIXEL_SPACE_IDS | NORMALIZED_SPACE_IDS and extent is not None:
        if extent.width is None or extent.height is None:
            issues.append(_issue("reference_extent_required", "$.reference_extent", f"{space_id} requires positive width and height."))
        if extent.units != "px":
            issues.append(_issue("reference_units_inconsistent", "$.reference_extent.units", f"{space_id} reference dimensions must use px."))
    if space_id == "physical_mm" and extent is not None and extent.units not in {"mm", "not_applicable"}:
        issues.append(_issue("reference_units_inconsistent", "$.reference_extent.units", "physical_mm reference extent must use mm or not_applicable."))

    if issues:
        raise CoordinateDescriptorError(issues)
    assert space_id is not None
    assert geometry_type is not None
    assert origin is not None
    assert directions is not None
    assert extent is not None
    assert pixel_convention is not None
    assert row_identity is not None
    assert overlay is not None
    return CoordinateDescriptor(
        space_id=space_id,
        geometry_type=geometry_type,
        components=components,
        component_units=component_units,
        origin=origin,
        positive_directions=directions,
        reference_extent=extent,
        pixel_convention=pixel_convention,
        row_identity=row_identity,
        source_camera_overlay=overlay,
        legacy_space_label=legacy_label,
        physical_frame=physical_frame,
        lineage_refs=lineage_refs,
        transform_refs=transform_refs,
    )


def parse_coordinate_descriptor(value: Any) -> CoordinateDescriptor:
    """Strictly parse a descriptor mapping or JSON representation."""

    return _parse_descriptor(value)


def validate_coordinate_descriptor(value: Any) -> tuple[CoordinateIssue, ...]:
    """Return stable structured issues instead of raising on invalid metadata."""

    try:
        _parse_descriptor(value)
    except CoordinateDescriptorError as exc:
        return exc.issues
    return ()


def _build_historical_coordinate_descriptor_v1(
    *,
    space_id: str,
    geometry_type: str,
    components: Sequence[str],
    component_units: Sequence[str],
    origin: str,
    positive_x: str,
    positive_y: str,
    reference_width: int | float | None,
    reference_height: int | float | None,
    reference_units: str,
    reference_authority: str,
    pixel_convention: str,
    row_identity_mode: str,
    row_identity_array_ref: str | None,
    source_camera_overlay: str,
    legacy_space_label: str | None = None,
    physical_frame: str | None = None,
    lineage_refs: Sequence[CoordinateRecordRef] = (),
    transform_refs: Sequence[CoordinateRecordRef] = (),
) -> CoordinateDescriptor:
    """Build one historical-v1 descriptor for explicit migration use."""

    descriptor = CoordinateDescriptor(
        space_id=space_id,
        geometry_type=geometry_type,
        components=tuple(components),
        component_units=tuple(component_units),
        origin=origin,
        positive_directions=PositiveDirections(x=positive_x, y=positive_y),
        reference_extent=ReferenceExtent(
            width=reference_width,
            height=reference_height,
            units=reference_units,
            authority=reference_authority,
        ),
        pixel_convention=pixel_convention,
        row_identity=RowIdentityReference(
            mode=row_identity_mode,
            array_ref=row_identity_array_ref,
        ),
        source_camera_overlay=source_camera_overlay,
        legacy_space_label=legacy_space_label,
        physical_frame=physical_frame,
        lineage_refs=tuple(lineage_refs),
        transform_refs=tuple(transform_refs),
    )
    return parse_coordinate_descriptor(descriptor)


def canonical_coordinate_descriptor_json(value: Any) -> str:
    """Return deterministic JSON after strict schema validation."""

    return _canonical_json(parse_coordinate_descriptor(value).to_dict())


def coordinate_descriptor_digest(value: Any) -> str:
    """Return the SHA-256 digest of canonical descriptor JSON."""

    canonical = canonical_coordinate_descriptor_json(value)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _historical_coordinate_descriptor_v1_attrs(
    value: Any,
    *,
    attr_name: str = COORDINATE_DESCRIPTOR_ATTR,
) -> dict[str, Any]:
    """Return historical-v1 attrs for explicit migration use."""

    name = str(attr_name).strip()
    if not name:
        raise ValueError("attr_name must be non-empty.")
    descriptor = parse_coordinate_descriptor(value)
    return {
        name: descriptor.to_dict(),
        f"{name}{COORDINATE_DESCRIPTOR_DIGEST_SUFFIX}": descriptor.digest(),
    }


def _stamp_historical_coordinate_descriptor_v1(
    node: Any,
    value: Any,
    *,
    attr_name: str = COORDINATE_DESCRIPTOR_ATTR,
) -> CoordinateDescriptor:
    """Stamp historical-v1 attrs for explicit migration/compatibility use."""

    attrs = getattr(node, "attrs", None)
    if attrs is None:
        raise TypeError("node must expose an attrs mapping.")
    descriptor = parse_coordinate_descriptor(value)
    snapshot = copy.deepcopy(dict(attrs))
    try:
        attrs.update(
            _historical_coordinate_descriptor_v1_attrs(
                descriptor,
                attr_name=attr_name,
            )
        )
    except Exception as exc:
        try:
            for name in tuple(attrs.keys()):
                if name not in snapshot:
                    del attrs[name]
            attrs.update(copy.deepcopy(snapshot))
            if dict(attrs) != snapshot:
                raise RuntimeError("restored attrs differ from the pre-call snapshot")
        except Exception as rollback_exc:
            raise CoordinateDescriptorError(
                (
                    _issue(
                        "historical_descriptor_stamp_rollback_incomplete",
                        "$",
                        f"Historical descriptor stamp failed and rollback was incomplete: {rollback_exc}.",
                    ),
                )
            ) from exc
        raise CoordinateDescriptorError(
            (
                _issue(
                    "historical_descriptor_stamp_failed",
                    "$",
                    "Historical descriptor stamp failed; attrs were restored exactly.",
                ),
            )
        ) from exc
    return descriptor


def load_coordinate_descriptor_attrs(
    attrs: Mapping[str, Any],
    *,
    attr_name: str = COORDINATE_DESCRIPTOR_ATTR,
) -> CoordinateDescriptor:
    """Load descriptor attrs and fail closed if their digest is absent or stale."""

    digest_name = f"{attr_name}{COORDINATE_DESCRIPTOR_DIGEST_SUFFIX}"
    issues: list[CoordinateIssue] = []
    if attr_name not in attrs:
        issues.append(_issue("descriptor_attr_missing", f"$.{attr_name}", "Descriptor attr is missing."))
    if digest_name not in attrs:
        issues.append(_issue("descriptor_digest_missing", f"$.{digest_name}", "Descriptor digest attr is missing."))
    if issues:
        raise CoordinateDescriptorError(issues)
    descriptor = parse_coordinate_descriptor(attrs[attr_name])
    stored_digest = attrs[digest_name]
    if not isinstance(stored_digest, str) or _SHA256_RE.fullmatch(stored_digest) is None:
        raise CoordinateDescriptorError(
            [_issue("descriptor_digest_invalid", f"$.{digest_name}", "Digest must be 64 lowercase hexadecimal characters.")]
        )
    actual_digest = descriptor.digest()
    if stored_digest != actual_digest:
        raise CoordinateDescriptorError(
            [_issue("descriptor_digest_mismatch", f"$.{digest_name}", "Stored digest does not match canonical descriptor content.")]
        )
    return descriptor


def resolve_legacy_space_id(
    legacy_label: str,
    *,
    context: LegacySpaceContext | None,
) -> str:
    """Resolve ``camera``/``texture`` only from explicit authoritative evidence."""

    label = str(legacy_label).strip().lower()
    expected_spaces = {
        "camera": "source_camera_image_px",
        "texture": "stimulus_texture_px",
    }
    if label not in expected_spaces:
        raise CoordinateDescriptorError(
            [_issue("legacy_label_unsupported", "$.legacy_space_label", f"Unsupported historical label {legacy_label!r}.")]
        )
    if context is None:
        raise CoordinateDescriptorError(
            [_issue("legacy_context_missing", "$.legacy_context", f"Resolving {label!r} requires explicit context.")]
        )

    issues: list[CoordinateIssue] = []
    expected_space = expected_spaces[label]
    if context.canonical_space_id != expected_space:
        issues.append(_issue("legacy_context_space_mismatch", "$.legacy_context.canonical_space_id", f"{label!r} requires explicit canonical space {expected_space!r}."))
    for name, value in (
        ("reference_width", context.reference_width),
        ("reference_height", context.reference_height),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0:
            issues.append(_issue("legacy_context_extent_invalid", f"$.legacy_context.{name}", "A positive finite dimension is required."))
    if context.reference_units != "px":
        issues.append(_issue("legacy_context_units_invalid", "$.legacy_context.reference_units", "Historical camera/texture extents must be explicitly measured in px."))
    if not isinstance(context.reference_authority, str) or not context.reference_authority.strip() or context.reference_authority != context.reference_authority.strip():
        issues.append(_issue("legacy_context_authority_missing", "$.legacy_context.reference_authority", "A non-empty reference authority is required."))
    if not context.evidence_refs:
        issues.append(_issue("legacy_context_evidence_missing", "$.legacy_context.evidence_refs", "At least one explicit evidence record is required."))
    else:
        _parse_record_refs(
            [item.to_dict() for item in context.evidence_refs],
            field_name="legacy_context.evidence_refs",
            issues=issues,
        )
    if issues:
        raise CoordinateDescriptorError(issues)
    return expected_space


# ---------------------------------------------------------------------------
# Canonical future-write schema (version 2)
# ---------------------------------------------------------------------------

# Version 1 above is retained only so historical archives and the writers being
# migrated in this branch remain explicitly readable.  Version 2 has separate
# entry points and types: neither parser dispatches to, upgrades, or guesses the
# other version.
CANONICAL_COORDINATE_DESCRIPTOR_SCHEMA_VERSION = 2
CANONICAL_COORDINATE_DESCRIPTOR_CANONICALIZATION = "canonical_json_sort_keys_v2"
CANONICAL_OVERLAY_CHAIN_DIRECTION = "descriptor_to_source_camera_image"

CANONICAL_OVERLAY_DIRECT = "direct"
CANONICAL_OVERLAY_REQUIRES_TRANSFORM = "requires_transform"
CANONICAL_OVERLAY_NOT_SUITABLE = "not_suitable"
CANONICAL_OVERLAY_STATUSES = frozenset(
    {
        CANONICAL_OVERLAY_DIRECT,
        CANONICAL_OVERLAY_REQUIRES_TRANSFORM,
        CANONICAL_OVERLAY_NOT_SUITABLE,
    }
)

PIXEL_FRAME_AUTHORITY_RECORD_KIND = "pixel_frame_authority"
PHYSICAL_FRAME_CALIBRATION_RECORD_KIND = "physical_frame_calibration"
FISH_BODY_FRAME_RECORD_KIND = "fish_anatomical_body_frame"
CANONICAL_FRAME_RECORD_KINDS = frozenset(
    {
        PIXEL_FRAME_AUTHORITY_RECORD_KIND,
        PHYSICAL_FRAME_CALIBRATION_RECORD_KIND,
        FISH_BODY_FRAME_RECORD_KIND,
    }
)

CANONICAL_COLLECTION_AXIS_ROLES = frozenset(
    {
        "subject_component",
        "keypoint",
        "tail_segment",
        "chaser",
    }
)

_CANONICAL_RECORD_ATTR_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_CANONICAL_RECORD_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_.:+-]+$")
_CANONICAL_ATTR_SELECTOR_RE = re.compile(
    r"^attrs\[(?P<width>[a-z][a-z0-9_]*),(?P<height>[a-z][a-z0-9_]*)\]$"
)
_CANONICAL_AUTHORITY_SELECTORS = frozenset({"shape[-2:]", "record"})


@dataclass(frozen=True)
class DigestBoundCoordinateRecordRef:
    """One canonical archive record and the digest of its exact content."""

    record_ref: str
    record_sha256: str

    def to_dict(self) -> dict[str, str]:
        return {
            "record_ref": self.record_ref,
            "record_sha256": self.record_sha256,
        }


@dataclass(frozen=True)
class CanonicalReferenceAuthority:
    """Digest-bound authority for an extent and its controlled resolver."""

    record_ref: str
    record_sha256: str
    selector: str

    @property
    def record(self) -> DigestBoundCoordinateRecordRef:
        return DigestBoundCoordinateRecordRef(
            record_ref=self.record_ref,
            record_sha256=self.record_sha256,
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "record_ref": self.record_ref,
            "record_sha256": self.record_sha256,
            "selector": self.selector,
        }


@dataclass(frozen=True)
class CanonicalReferenceExtent:
    width: int | float | None
    height: int | float | None
    units: str
    authority: CanonicalReferenceAuthority

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "units": self.units,
            "authority": self.authority.to_dict(),
        }


@dataclass(frozen=True)
class CanonicalRowIdentityRef:
    """Compact link to one external ``palette.row_identity_contract`` record."""

    record_ref: str
    record_sha256: str

    def to_dict(self) -> dict[str, str]:
        return {
            "record_ref": self.record_ref,
            "record_sha256": self.record_sha256,
        }


@dataclass(frozen=True)
class CanonicalSourceCameraOverlay:
    """Whether and how the native coordinates can reach source-camera pixels."""

    status: str
    transform_refs: tuple[DigestBoundCoordinateRecordRef, ...] = ()
    chain_direction: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"status": self.status}
        if self.transform_refs:
            payload["chain_direction"] = self.chain_direction
            payload["transform_refs"] = [
                item.to_dict() for item in self.transform_refs
            ]
        return payload


@dataclass(frozen=True)
class CanonicalFrameRecord:
    """Exact physical or anatomical frame record required by local frames."""

    kind: str
    record_ref: str
    record_sha256: str

    @property
    def record(self) -> DigestBoundCoordinateRecordRef:
        return DigestBoundCoordinateRecordRef(
            record_ref=self.record_ref,
            record_sha256=self.record_sha256,
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "record_ref": self.record_ref,
            "record_sha256": self.record_sha256,
        }


@dataclass(frozen=True)
class CanonicalCollectionAxis:
    """One controlled non-coordinate axis on a coordinate-bearing array.

    The axis is deliberately distinct from ``components``: those fields name
    the coordinate tuple itself, while this record names a collection of
    scientific surfaces that share one coordinate frame.  Version 2 admits
    only controlled, named collection roles and binds their exact labels via a
    persisted, digest-bound authority record.
    """

    axis: int
    role: str
    cardinality: int
    label_authority: DigestBoundCoordinateRecordRef

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "role": self.role,
            "cardinality": self.cardinality,
            "label_authority": self.label_authority.to_dict(),
        }


@dataclass(frozen=True)
class CanonicalCoordinateProfile:
    """Controlled field combination for one scientifically defined space."""

    profile_id: str
    space_id: str
    origin: str
    positive_x: str
    positive_y: str
    coordinate_unit: str
    reference_units: str
    extent_mode: str
    pixel_conventions: frozenset[str]
    overlay_statuses: frozenset[str]
    frame_record_kind: str | None = None
    geometry_types: frozenset[str] = GEOMETRY_TYPES
    publication_status: str = "available"


def _canonical_profile(
    profile_id: str,
    space_id: str,
    *,
    origin: str,
    positive_x: str,
    positive_y: str,
    coordinate_unit: str,
    reference_units: str,
    extent_mode: str = "required",
    pixel_conventions: frozenset[str] = frozenset(
        {"pixel_center", "pixel_edge_half_open", "continuous"}
    ),
    overlay_statuses: frozenset[str] = frozenset(
        {CANONICAL_OVERLAY_REQUIRES_TRANSFORM, CANONICAL_OVERLAY_NOT_SUITABLE}
    ),
    frame_record_kind: str | None = None,
    geometry_types: frozenset[str] = GEOMETRY_TYPES,
    publication_status: str = "available",
) -> CanonicalCoordinateProfile:
    return CanonicalCoordinateProfile(
        profile_id=profile_id,
        space_id=space_id,
        origin=origin,
        positive_x=positive_x,
        positive_y=positive_y,
        coordinate_unit=coordinate_unit,
        reference_units=reference_units,
        extent_mode=extent_mode,
        pixel_conventions=pixel_conventions,
        overlay_statuses=overlay_statuses,
        frame_record_kind=frame_record_kind,
        geometry_types=geometry_types,
        publication_status=publication_status,
    )


CANONICAL_COORDINATE_PROFILES: Mapping[str, CanonicalCoordinateProfile] = {
    item.profile_id: item
    for item in (
        _canonical_profile(
            "source_camera_image_px.top_left_y_down.v1",
            "source_camera_image_px",
            origin="top_left",
            positive_x="right",
            positive_y="down",
            coordinate_unit="px",
            reference_units="px",
            overlay_statuses=frozenset({CANONICAL_OVERLAY_DIRECT}),
            frame_record_kind=PIXEL_FRAME_AUTHORITY_RECORD_KIND,
        ),
        _canonical_profile(
            "source_camera_image_px.unit_vector_y_down.v1",
            "source_camera_image_px",
            origin="not_applicable",
            positive_x="right",
            positive_y="down",
            coordinate_unit="unitless",
            reference_units="px",
            pixel_conventions=frozenset({"not_applicable"}),
            overlay_statuses=frozenset({CANONICAL_OVERLAY_NOT_SUITABLE}),
            frame_record_kind=PIXEL_FRAME_AUTHORITY_RECORD_KIND,
            geometry_types=frozenset({"vector_xy", "vector_sequence_xy"}),
        ),
        _canonical_profile(
            "source_camera_image_px.displacement_vector_y_down.v1",
            "source_camera_image_px",
            origin="not_applicable",
            positive_x="right",
            positive_y="down",
            coordinate_unit="px",
            reference_units="px",
            pixel_conventions=frozenset({"not_applicable"}),
            overlay_statuses=frozenset({CANONICAL_OVERLAY_NOT_SUITABLE}),
            frame_record_kind=PIXEL_FRAME_AUTHORITY_RECORD_KIND,
            geometry_types=frozenset({"vector_xy"}),
        ),
        _canonical_profile(
            "source_camera_normalized_xy.top_left_y_down.v1",
            "source_camera_normalized_xy",
            origin="top_left",
            positive_x="right",
            positive_y="down",
            coordinate_unit="normalized",
            reference_units="px",
            pixel_conventions=frozenset({"continuous"}),
            frame_record_kind=PIXEL_FRAME_AUTHORITY_RECORD_KIND,
        ),
        _canonical_profile(
            "detector_model_input_px.top_left_y_down.v1",
            "detector_model_input_px",
            origin="top_left",
            positive_x="right",
            positive_y="down",
            coordinate_unit="px",
            reference_units="px",
            frame_record_kind=PIXEL_FRAME_AUTHORITY_RECORD_KIND,
        ),
        _canonical_profile(
            "detector_normalized_xy.top_left_y_down.v1",
            "detector_normalized_xy",
            origin="top_left",
            positive_x="right",
            positive_y="down",
            coordinate_unit="normalized",
            reference_units="px",
            pixel_conventions=frozenset({"continuous"}),
            frame_record_kind=PIXEL_FRAME_AUTHORITY_RECORD_KIND,
        ),
        _canonical_profile(
            "roi_local_px.top_left_y_down.v1",
            "roi_local_px",
            origin="top_left",
            positive_x="right",
            positive_y="down",
            coordinate_unit="px",
            reference_units="px",
            frame_record_kind=PIXEL_FRAME_AUTHORITY_RECORD_KIND,
        ),
        _canonical_profile(
            "stimulus_texture_px.top_left_y_down.v1",
            "stimulus_texture_px",
            origin="top_left",
            positive_x="right",
            positive_y="down",
            coordinate_unit="px",
            reference_units="px",
            frame_record_kind=PIXEL_FRAME_AUTHORITY_RECORD_KIND,
            publication_status="unavailable_missing_typed_lineage",
        ),
        _canonical_profile(
            "stimulus_canvas_px.top_left_y_down.v1",
            "stimulus_canvas_px",
            origin="top_left",
            positive_x="right",
            positive_y="down",
            coordinate_unit="px",
            reference_units="px",
            frame_record_kind=PIXEL_FRAME_AUTHORITY_RECORD_KIND,
        ),
        _canonical_profile(
            "projector_px.top_left_y_down.v1",
            "projector_px",
            origin="projector_top_left",
            positive_x="right",
            positive_y="down",
            coordinate_unit="px",
            reference_units="px",
            frame_record_kind=PIXEL_FRAME_AUTHORITY_RECORD_KIND,
            publication_status="unavailable_missing_typed_lineage",
        ),
        _canonical_profile(
            "arena_relative_canvas_px.top_left_y_down.v1",
            "arena_relative_canvas_px",
            origin="arena_top_left",
            positive_x="right",
            positive_y="down",
            coordinate_unit="px",
            reference_units="px",
            frame_record_kind=PIXEL_FRAME_AUTHORITY_RECORD_KIND,
        ),
        _canonical_profile(
            "physical_mm.source_camera_y_down.v1",
            "physical_mm",
            origin="physical_frame_origin",
            positive_x="right",
            positive_y="down",
            coordinate_unit="mm",
            reference_units="mm",
            pixel_conventions=frozenset({"not_applicable"}),
            frame_record_kind=PHYSICAL_FRAME_CALIBRATION_RECORD_KIND,
        ),
        _canonical_profile(
            "physical_mm.arena_y_down.v1",
            "physical_mm",
            origin="physical_frame_origin",
            positive_x="right",
            positive_y="down",
            coordinate_unit="mm",
            reference_units="mm",
            pixel_conventions=frozenset({"not_applicable"}),
            frame_record_kind=PHYSICAL_FRAME_CALIBRATION_RECORD_KIND,
            publication_status="unavailable_missing_direction_labelled_transform",
        ),
        _canonical_profile(
            "physical_mm.cartesian_y_up.v1",
            "physical_mm",
            origin="physical_frame_origin",
            positive_x="right",
            positive_y="up",
            coordinate_unit="mm",
            reference_units="not_applicable",
            extent_mode="record",
            pixel_conventions=frozenset({"not_applicable"}),
            frame_record_kind=PHYSICAL_FRAME_CALIBRATION_RECORD_KIND,
            publication_status="unavailable_missing_direction_labelled_transform",
        ),
        _canonical_profile(
            "fish_anatomical_body_frame.px_anterior_left.v1",
            "fish_anatomical_body_frame",
            origin="body_frame_origin",
            positive_x="anterior",
            positive_y="anatomical_left",
            coordinate_unit="px",
            reference_units="not_applicable",
            extent_mode="record",
            pixel_conventions=frozenset({"not_applicable"}),
            frame_record_kind=FISH_BODY_FRAME_RECORD_KIND,
        ),
        _canonical_profile(
            "fish_anatomical_body_frame.mm_anterior_left.v1",
            "fish_anatomical_body_frame",
            origin="body_frame_origin",
            positive_x="anterior",
            positive_y="anatomical_left",
            coordinate_unit="mm",
            reference_units="not_applicable",
            extent_mode="record",
            pixel_conventions=frozenset({"not_applicable"}),
            frame_record_kind=FISH_BODY_FRAME_RECORD_KIND,
        ),
        _canonical_profile(
            "fish_anatomical_body_frame.unit_vector.v1",
            "fish_anatomical_body_frame",
            origin="body_frame_origin",
            positive_x="anterior",
            positive_y="anatomical_left",
            coordinate_unit="unitless",
            reference_units="not_applicable",
            extent_mode="record",
            pixel_conventions=frozenset({"not_applicable"}),
            frame_record_kind=FISH_BODY_FRAME_RECORD_KIND,
            geometry_types=frozenset({"vector_xy", "coordinate_component"}),
        ),
    )
}


@dataclass(frozen=True)
class CanonicalCoordinateDescriptor:
    """Strict, compact version-2 contract for one persisted geometry surface."""

    profile_id: str
    space_id: str
    geometry_type: str
    components: tuple[str, ...]
    component_units: tuple[str, ...]
    origin: str
    positive_directions: PositiveDirections
    reference_extent: CanonicalReferenceExtent
    pixel_convention: str
    row_identity: CanonicalRowIdentityRef
    source_camera_overlay: CanonicalSourceCameraOverlay
    lineage_refs: tuple[DigestBoundCoordinateRecordRef, ...]
    collection_axis: CanonicalCollectionAxis | None = None
    frame_record: CanonicalFrameRecord | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_id": COORDINATE_DESCRIPTOR_SCHEMA_ID,
            "schema_version": CANONICAL_COORDINATE_DESCRIPTOR_SCHEMA_VERSION,
            "profile_id": self.profile_id,
            "space_id": self.space_id,
            "geometry_type": self.geometry_type,
            "components": list(self.components),
            "component_units": list(self.component_units),
            "origin": self.origin,
            "positive_directions": self.positive_directions.to_dict(),
            "reference_extent": self.reference_extent.to_dict(),
            "pixel_convention": self.pixel_convention,
            "row_identity": self.row_identity.to_dict(),
            "source_camera_overlay": self.source_camera_overlay.to_dict(),
            "lineage_refs": [item.to_dict() for item in self.lineage_refs],
        }
        if self.frame_record is not None:
            payload["frame_record"] = self.frame_record.to_dict()
        if self.collection_axis is not None:
            payload["collection_axis"] = self.collection_axis.to_dict()
        return payload

    def canonical_json(self) -> str:
        return canonical_coordinate_descriptor_v2_json(self)

    def digest(self) -> str:
        return canonical_coordinate_descriptor_v2_digest(self)

    def to_attrs(
        self,
        *,
        attr_name: str = COORDINATE_DESCRIPTOR_ATTR,
    ) -> dict[str, Any]:
        return canonical_coordinate_descriptor_v2_attrs(
            self,
            attr_name=attr_name,
        )


_CANONICAL_V2_REQUIRED_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "profile_id",
        "space_id",
        "geometry_type",
        "components",
        "component_units",
        "origin",
        "positive_directions",
        "reference_extent",
        "pixel_convention",
        "row_identity",
        "source_camera_overlay",
        "lineage_refs",
    }
)
_CANONICAL_V2_OPTIONAL_FIELDS = frozenset({"frame_record", "collection_axis"})


def _canonical_archive_record_ref(
    value: Any,
    *,
    path: str,
    issues: list[CoordinateIssue],
) -> str | None:
    if not isinstance(value, str) or not value or value != value.strip():
        issues.append(
            _issue(
                "record_ref_invalid",
                path,
                "Record ref must be a non-empty canonical archive-root path.",
            )
        )
        return None
    if value.count("@") > 1:
        issues.append(_issue("record_ref_invalid", path, "Record ref has multiple attr selectors."))
        return None
    node_path, separator, attr_name = value.partition("@")
    if (
        not node_path.startswith("/")
        or node_path == "/"
        or node_path.endswith("/")
        or "//" in node_path
    ):
        issues.append(_issue("record_ref_invalid", path, "Record ref must be an archive-root path such as /analysis/run."))
        return None
    segments = node_path[1:].split("/")
    if any(
        segment in {"", ".", ".."}
        or _CANONICAL_RECORD_SEGMENT_RE.fullmatch(segment) is None
        for segment in segments
    ):
        issues.append(_issue("record_ref_invalid", path, "Record ref contains a noncanonical path segment."))
        return None
    if separator and _CANONICAL_RECORD_ATTR_RE.fullmatch(attr_name) is None:
        issues.append(_issue("record_ref_invalid", path, "Record attr selector is not canonical."))
        return None
    return value


def _parse_digest_bound_record_ref(
    value: Any,
    *,
    path: str,
    issues: list[CoordinateIssue],
) -> DigestBoundCoordinateRecordRef | None:
    start = len(issues)
    expected = {"record_ref", "record_sha256"}
    if not isinstance(value, Mapping):
        issues.append(_issue("record_ref_invalid", path, "Digest-bound record ref must be a mapping."))
        return None
    for name in sorted(expected - set(value)):
        issues.append(_issue("missing_field", f"{path}.{name}", "Required field is missing."))
    for name in sorted(set(value) - expected):
        issues.append(_issue("unknown_field", f"{path}.{name}", "Field is not part of a digest-bound record ref."))
    record_ref = _canonical_archive_record_ref(
        value.get("record_ref"),
        path=f"{path}.record_ref",
        issues=issues,
    )
    digest = value.get("record_sha256")
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        issues.append(_issue("record_digest_required", f"{path}.record_sha256", "A lowercase SHA-256 digest is required."))
    if len(issues) != start or record_ref is None or not isinstance(digest, str):
        return None
    return DigestBoundCoordinateRecordRef(record_ref=record_ref, record_sha256=digest)


def _parse_canonical_reference_extent(
    value: Any,
    issues: list[CoordinateIssue],
) -> CanonicalReferenceExtent | None:
    path = "$.reference_extent"
    start = len(issues)
    expected = {"width", "height", "units", "authority"}
    if not isinstance(value, Mapping):
        issues.append(_issue("reference_extent_invalid", path, "Reference extent must be a mapping."))
        return None
    for name in sorted(expected - set(value)):
        issues.append(_issue("missing_field", f"{path}.{name}", "Required field is missing."))
    for name in sorted(set(value) - expected):
        issues.append(_issue("unknown_field", f"{path}.{name}", "Field is not part of the canonical extent."))
    width = _dimension(value.get("width"), path=f"{path}.width", issues=issues)
    height = _dimension(value.get("height"), path=f"{path}.height", issues=issues)
    if (value.get("width") is None) != (value.get("height") is None):
        issues.append(_issue("reference_extent_pair_required", path, "width and height must both be numbers or both be null."))
    units = value.get("units")
    if units not in REFERENCE_UNITS:
        issues.append(_issue("reference_units_unsupported", f"{path}.units", f"Unsupported units {units!r}."))
    authority_raw = value.get("authority")
    authority: CanonicalReferenceAuthority | None = None
    authority_path = f"{path}.authority"
    if not isinstance(authority_raw, Mapping):
        issues.append(_issue("reference_authority_invalid", authority_path, "Authority must be a digest-bound mapping."))
    else:
        authority_expected = {"record_ref", "record_sha256", "selector"}
        for name in sorted(authority_expected - set(authority_raw)):
            issues.append(_issue("missing_field", f"{authority_path}.{name}", "Required field is missing."))
        for name in sorted(set(authority_raw) - authority_expected):
            issues.append(_issue("unknown_field", f"{authority_path}.{name}", "Field is not part of a reference authority."))
        record = _parse_digest_bound_record_ref(
            {
                "record_ref": authority_raw.get("record_ref"),
                "record_sha256": authority_raw.get("record_sha256"),
            },
            path=authority_path,
            issues=issues,
        )
        selector = authority_raw.get("selector")
        if not isinstance(selector, str) or (
            selector not in _CANONICAL_AUTHORITY_SELECTORS
            and _CANONICAL_ATTR_SELECTOR_RE.fullmatch(selector) is None
        ):
            issues.append(_issue("reference_authority_selector_unsupported", f"{authority_path}.selector", "Selector must be shape[-2:], attrs[width,height], or record."))
        if record is not None and isinstance(selector, str):
            authority = CanonicalReferenceAuthority(
                record_ref=record.record_ref,
                record_sha256=record.record_sha256,
                selector=selector,
            )
    if len(issues) != start or authority is None or units not in REFERENCE_UNITS:
        return None
    return CanonicalReferenceExtent(
        width=width,
        height=height,
        units=str(units),
        authority=authority,
    )


def _parse_canonical_row_identity(
    value: Any,
    issues: list[CoordinateIssue],
) -> CanonicalRowIdentityRef | None:
    path = "$.row_identity"
    record = _parse_digest_bound_record_ref(value, path=path, issues=issues)
    if record is None:
        return None
    if not record.record_ref.endswith(f"@{ROW_IDENTITY_CONTRACT_ATTR}"):
        issues.append(_issue("row_identity_record_ref_invalid", f"{path}.record_ref", f"Row identity must reference @{ROW_IDENTITY_CONTRACT_ATTR}."))
        return None
    return CanonicalRowIdentityRef(
        record_ref=record.record_ref,
        record_sha256=record.record_sha256,
    )


def _parse_canonical_record_refs(
    value: Any,
    *,
    path: str,
    issues: list[CoordinateIssue],
    require_nonempty: bool,
) -> tuple[DigestBoundCoordinateRecordRef, ...]:
    if not isinstance(value, list) or (require_nonempty and not value):
        issues.append(_issue("record_refs_invalid", path, "Record refs must be a non-empty list." if require_nonempty else "Record refs must be a list."))
        return ()
    parsed: list[DigestBoundCoordinateRecordRef] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        record = _parse_digest_bound_record_ref(
            item,
            path=f"{path}[{index}]",
            issues=issues,
        )
        if record is None:
            continue
        if record.record_ref in seen:
            issues.append(_issue("record_ref_duplicate", f"{path}[{index}].record_ref", "Record ref is duplicated."))
            continue
        seen.add(record.record_ref)
        parsed.append(record)
    return tuple(parsed)


def _parse_canonical_overlay(
    value: Any,
    issues: list[CoordinateIssue],
) -> CanonicalSourceCameraOverlay | None:
    path = "$.source_camera_overlay"
    if not isinstance(value, Mapping):
        issues.append(_issue("overlay_invalid", path, "Overlay contract must be a mapping."))
        return None
    status = value.get("status")
    if status not in CANONICAL_OVERLAY_STATUSES:
        issues.append(_issue("overlay_status_unsupported", f"{path}.status", f"Unsupported status {status!r}."))
        return None
    if status == CANONICAL_OVERLAY_REQUIRES_TRANSFORM:
        expected = {"status", "chain_direction", "transform_refs"}
        for name in sorted(expected - set(value)):
            issues.append(_issue("missing_field", f"{path}.{name}", "Required field is missing."))
        for name in sorted(set(value) - expected):
            issues.append(_issue("unknown_field", f"{path}.{name}", "Field is not part of a transform-required overlay."))
        chain_direction = value.get("chain_direction")
        if chain_direction != CANONICAL_OVERLAY_CHAIN_DIRECTION:
            issues.append(_issue("overlay_chain_direction_invalid", f"{path}.chain_direction", f"Expected {CANONICAL_OVERLAY_CHAIN_DIRECTION!r}."))
        refs = _parse_canonical_record_refs(
            value.get("transform_refs"),
            path=f"{path}.transform_refs",
            issues=issues,
            require_nonempty=True,
        )
        if chain_direction != CANONICAL_OVERLAY_CHAIN_DIRECTION or not refs:
            return None
        return CanonicalSourceCameraOverlay(
            status=str(status),
            transform_refs=refs,
            chain_direction=str(chain_direction),
        )
    expected = {"status"}
    for name in sorted(set(value) - expected):
        issues.append(_issue("unknown_field", f"{path}.{name}", "Direct/not-suitable overlay cannot carry transform state."))
    if set(value) != expected:
        return None
    return CanonicalSourceCameraOverlay(status=str(status))


def _parse_canonical_frame_record(
    value: Any,
    issues: list[CoordinateIssue],
) -> CanonicalFrameRecord | None:
    path = "$.frame_record"
    if not isinstance(value, Mapping):
        issues.append(_issue("frame_record_invalid", path, "Frame record must be a mapping."))
        return None
    expected = {"kind", "record_ref", "record_sha256"}
    for name in sorted(expected - set(value)):
        issues.append(_issue("missing_field", f"{path}.{name}", "Required field is missing."))
    for name in sorted(set(value) - expected):
        issues.append(_issue("unknown_field", f"{path}.{name}", "Field is not part of a frame record."))
    kind = value.get("kind")
    if kind not in CANONICAL_FRAME_RECORD_KINDS:
        issues.append(_issue("frame_record_kind_unsupported", f"{path}.kind", f"Unsupported frame kind {kind!r}."))
    record = _parse_digest_bound_record_ref(
        {
            "record_ref": value.get("record_ref"),
            "record_sha256": value.get("record_sha256"),
        },
        path=path,
        issues=issues,
    )
    if kind not in CANONICAL_FRAME_RECORD_KINDS or record is None:
        return None
    return CanonicalFrameRecord(
        kind=str(kind),
        record_ref=record.record_ref,
        record_sha256=record.record_sha256,
    )


def _parse_canonical_collection_axis(
    value: Any,
    issues: list[CoordinateIssue],
) -> CanonicalCollectionAxis | None:
    path = "$.collection_axis"
    start = len(issues)
    expected = {"axis", "role", "cardinality", "label_authority"}
    if not isinstance(value, Mapping):
        issues.append(
            _issue(
                "collection_axis_invalid",
                path,
                "Collection axis must be a mapping.",
            )
        )
        return None
    for name in sorted(expected - set(value)):
        issues.append(
            _issue("missing_field", f"{path}.{name}", "Required field is missing.")
        )
    for name in sorted(set(value) - expected):
        issues.append(
            _issue(
                "unknown_field",
                f"{path}.{name}",
                "Field is not part of the canonical collection-axis record.",
            )
        )

    axis = value.get("axis")
    if type(axis) is not int or axis != 1:
        issues.append(
            _issue(
                "collection_axis_index_unsupported",
                f"{path}.axis",
                "A controlled collection axis must be exact physical axis 1; axis 0 is row identity and arbitrary unnamed axes are forbidden.",
            )
        )
    role = value.get("role")
    if role not in CANONICAL_COLLECTION_AXIS_ROLES:
        issues.append(
            _issue(
                "collection_axis_role_unsupported",
                f"{path}.role",
                f"Unsupported collection-axis role {role!r}.",
            )
        )
    cardinality = value.get("cardinality")
    if type(cardinality) is not int or cardinality <= 0:
        issues.append(
            _issue(
                "collection_axis_cardinality_invalid",
                f"{path}.cardinality",
                "Collection-axis cardinality must be an exact positive integer.",
            )
        )
    authority = _parse_digest_bound_record_ref(
        value.get("label_authority"),
        path=f"{path}.label_authority",
        issues=issues,
    )
    if len(issues) != start or authority is None:
        return None
    assert type(axis) is int
    assert isinstance(role, str)
    assert type(cardinality) is int
    return CanonicalCollectionAxis(
        axis=axis,
        role=role,
        cardinality=cardinality,
        label_authority=authority,
    )


def _canonical_v2_payload_mapping(
    value: Any,
) -> tuple[Mapping[str, Any] | None, list[CoordinateIssue]]:
    if isinstance(value, CanonicalCoordinateDescriptor):
        return value.to_dict(), []
    if isinstance(value, CoordinateDescriptor):
        return value.to_dict(), []
    if isinstance(value, (bytes, bytearray)):
        try:
            value = bytes(value).decode("utf-8")
        except UnicodeDecodeError:
            return None, [_issue("descriptor_json_invalid", "$", "Descriptor bytes are not UTF-8.")]
    if isinstance(value, str):
        def reject_duplicate_pairs(
            pairs: list[tuple[str, Any]],
        ) -> dict[str, Any]:
            payload: dict[str, Any] = {}
            for name, item in pairs:
                if name in payload:
                    raise ValueError(f"duplicate JSON key {name!r}")
                payload[name] = item
            return payload

        try:
            value = json.loads(value, object_pairs_hook=reject_duplicate_pairs)
        except (json.JSONDecodeError, ValueError) as exc:
            return None, [_issue("descriptor_json_invalid", "$", f"Descriptor JSON is invalid: {exc}.")]
    if not isinstance(value, Mapping):
        return None, [_issue("descriptor_not_mapping", "$", "Descriptor must be a mapping or JSON object.")]
    return value, []


def _parse_canonical_components(
    payload: Mapping[str, Any],
    *,
    geometry_type: str | None,
    issues: list[CoordinateIssue],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    raw_components = payload.get("components")
    components: tuple[str, ...] = ()
    if not isinstance(raw_components, list) or not raw_components:
        issues.append(_issue("components_invalid", "$.components", "components must be a non-empty list."))
    else:
        values: list[str] = []
        for index, component in enumerate(raw_components):
            if not isinstance(component, str) or not component or component != component.strip():
                issues.append(_issue("component_invalid", f"$.components[{index}]", "Component must be a non-empty canonical string."))
            else:
                values.append(component)
        components = tuple(values)
        if len(set(components)) != len(components):
            issues.append(_issue("component_duplicate", "$.components", "Components must be unique."))

    raw_units = payload.get("component_units")
    units: tuple[str, ...] = ()
    if not isinstance(raw_units, list) or not raw_units:
        issues.append(_issue("component_units_invalid", "$.component_units", "component_units must be a non-empty list."))
    else:
        values = []
        for index, unit in enumerate(raw_units):
            if unit not in COMPONENT_UNITS:
                issues.append(_issue("component_unit_unsupported", f"$.component_units[{index}]", f"Unsupported unit {unit!r}."))
            else:
                values.append(str(unit))
        units = tuple(values)
    if isinstance(raw_components, list) and isinstance(raw_units, list) and len(raw_components) != len(raw_units):
        issues.append(_issue("component_unit_count_mismatch", "$.component_units", "One unit is required for every component."))
    if geometry_type in _GEOMETRY_COMPONENTS and components:
        expected = _GEOMETRY_COMPONENTS[geometry_type]
        if components != expected:
            issues.append(_issue("geometry_components_mismatch", "$.components", f"{geometry_type} requires {list(expected)!r}."))
    elif geometry_type == "coordinate_component" and len(components) != 1:
        issues.append(_issue("geometry_components_mismatch", "$.components", "coordinate_component requires one component."))
    return components, units


def _canonical_profile_issues(
    descriptor: CanonicalCoordinateDescriptor,
) -> tuple[CoordinateIssue, ...]:
    issues: list[CoordinateIssue] = []
    profile = CANONICAL_COORDINATE_PROFILES.get(descriptor.profile_id)
    if profile is None:
        return (
            _issue("profile_id_unsupported", "$.profile_id", f"Unsupported profile {descriptor.profile_id!r}."),
        )
    fixed = (
        ("space_id", descriptor.space_id, profile.space_id),
        ("origin", descriptor.origin, profile.origin),
        ("positive_directions.x", descriptor.positive_directions.x, profile.positive_x),
        ("positive_directions.y", descriptor.positive_directions.y, profile.positive_y),
        ("reference_extent.units", descriptor.reference_extent.units, profile.reference_units),
    )
    for path, actual, expected in fixed:
        if actual != expected:
            issues.append(_issue("profile_field_mismatch", f"$.{path}", f"Profile requires {expected!r}, found {actual!r}."))
    if descriptor.geometry_type not in profile.geometry_types:
        issues.append(_issue("profile_geometry_unsupported", "$.geometry_type", f"Profile does not support {descriptor.geometry_type!r}."))
    if descriptor.pixel_convention not in profile.pixel_conventions:
        issues.append(_issue("profile_pixel_convention_mismatch", "$.pixel_convention", f"Profile does not support {descriptor.pixel_convention!r}."))
    for index, (component, unit) in enumerate(
        zip(descriptor.components, descriptor.component_units, strict=True)
    ):
        expected_unit = profile.coordinate_unit
        if component == "angle":
            if unit not in {"deg", "rad"}:
                issues.append(_issue("profile_component_unit_mismatch", f"$.component_units[{index}]", "Angle must use deg or rad."))
        elif unit != expected_unit:
            issues.append(_issue("profile_component_unit_mismatch", f"$.component_units[{index}]", f"Component {component!r} must use {expected_unit!r}."))

    extent = descriptor.reference_extent
    if extent.units == "px":
        for name, value in (("width", extent.width), ("height", extent.height)):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                issues.append(
                    _issue(
                        "pixel_reference_extent_not_integer",
                        f"$.reference_extent.{name}",
                        "Pixel reference extents must be exact positive integers.",
                    )
                )
    if profile.extent_mode == "required":
        if extent.width is None or extent.height is None:
            issues.append(_issue("reference_extent_required", "$.reference_extent", "Profile requires positive width and height."))
        if (
            extent.authority.selector == "record"
            and profile.frame_record_kind is None
        ):
            issues.append(_issue("reference_authority_selector_mismatch", "$.reference_extent.authority.selector", "A concrete extent requires shape or attrs authority."))
        if (
            profile.frame_record_kind is not None
            and extent.authority.selector != "record"
        ):
            issues.append(
                _issue(
                    "reference_authority_selector_mismatch",
                    "$.reference_extent.authority.selector",
                    "A typed coordinate-frame extent must be resolved from its exact frame record.",
                )
            )
    else:
        if extent.width is not None or extent.height is not None:
            issues.append(_issue("reference_extent_forbidden", "$.reference_extent", "Unbounded frame profile requires null width and height."))
        if extent.authority.selector != "record":
            issues.append(_issue("reference_authority_selector_mismatch", "$.reference_extent.authority.selector", "Unbounded frame authority must select the frame record."))

    if descriptor.geometry_type == "raster_yx" and descriptor.pixel_convention != "pixel_center":
        issues.append(_issue("geometry_pixel_convention_mismatch", "$.pixel_convention", "raster_yx requires pixel_center."))
    if descriptor.geometry_type.startswith("bbox_") and descriptor.pixel_convention not in {"continuous", "pixel_edge_half_open"}:
        issues.append(_issue("geometry_pixel_convention_mismatch", "$.pixel_convention", "Bounding boxes require continuous or half-open edge coordinates."))
    if descriptor.space_id in NORMALIZED_SPACE_IDS and descriptor.pixel_convention != "continuous":
        issues.append(_issue("geometry_pixel_convention_mismatch", "$.pixel_convention", "Normalized coordinates are continuous."))

    overlay = descriptor.source_camera_overlay
    if overlay.status not in profile.overlay_statuses:
        issues.append(_issue("profile_overlay_mismatch", "$.source_camera_overlay.status", f"Profile does not permit {overlay.status!r}."))
    if overlay.status == CANONICAL_OVERLAY_DIRECT and descriptor.space_id != "source_camera_image_px":
        issues.append(_issue("overlay_direct_space_mismatch", "$.source_camera_overlay.status", "Only source-camera image pixels are directly overlayable."))

    lineage_by_ref = {item.record_ref: item for item in descriptor.lineage_refs}
    authority = extent.authority
    authority_lineage = lineage_by_ref.get(authority.record_ref)
    if authority_lineage is None or authority_lineage.record_sha256 != authority.record_sha256:
        issues.append(_issue("reference_authority_lineage_missing", "$.lineage_refs", "Exact reference authority and digest must occur in lineage_refs."))

    collection_axis = descriptor.collection_axis
    if collection_axis is not None:
        label_authority = collection_axis.label_authority
        label_lineage = lineage_by_ref.get(label_authority.record_ref)
        if (
            label_lineage is None
            or label_lineage.record_sha256 != label_authority.record_sha256
        ):
            issues.append(
                _issue(
                    "collection_axis_authority_lineage_missing",
                    "$.lineage_refs",
                    "The exact collection-label authority and digest must occur in lineage_refs.",
                )
            )

    if profile.frame_record_kind is None:
        if descriptor.frame_record is not None:
            issues.append(_issue("frame_record_forbidden", "$.frame_record", "This profile does not use a local frame record."))
    else:
        frame = descriptor.frame_record
        if frame is None:
            issues.append(_issue("frame_record_required", "$.frame_record", f"Profile requires {profile.frame_record_kind!r}."))
        else:
            if frame.kind != profile.frame_record_kind:
                issues.append(_issue("frame_record_kind_mismatch", "$.frame_record.kind", f"Profile requires {profile.frame_record_kind!r}."))
            frame_lineage = lineage_by_ref.get(frame.record_ref)
            if frame_lineage is None or frame_lineage.record_sha256 != frame.record_sha256:
                issues.append(_issue("frame_record_lineage_missing", "$.lineage_refs", "Exact frame record and digest must occur in lineage_refs."))
            if (
                frame.record_ref != authority.record_ref
                or frame.record_sha256 != authority.record_sha256
            ):
                issues.append(_issue("frame_authority_mismatch", "$.frame_record", "Frame record must be the exact reference authority."))
    return tuple(issues)


def _parse_canonical_coordinate_descriptor_v2(
    value: Any,
) -> CanonicalCoordinateDescriptor:
    payload, initial = _canonical_v2_payload_mapping(value)
    issues = list(initial)
    if payload is None:
        raise CoordinateDescriptorError(issues)
    for name in sorted(_CANONICAL_V2_REQUIRED_FIELDS - set(payload)):
        issues.append(_issue("missing_field", f"$.{name}", "Required field is missing."))
    for name in sorted(set(payload) - _CANONICAL_V2_REQUIRED_FIELDS - _CANONICAL_V2_OPTIONAL_FIELDS):
        issues.append(_issue("unknown_field", f"$.{name}", "Field is not part of canonical schema version 2."))
    if payload.get("schema_id") != COORDINATE_DESCRIPTOR_SCHEMA_ID:
        issues.append(_issue("schema_id_unsupported", "$.schema_id", f"Expected {COORDINATE_DESCRIPTOR_SCHEMA_ID!r}."))
    version = payload.get("schema_version")
    if (
        type(version) is not int
        or version != CANONICAL_COORDINATE_DESCRIPTOR_SCHEMA_VERSION
    ):
        issues.append(_issue("canonical_schema_version_required", "$.schema_version", "Canonical future descriptors require schema version 2."))

    profile_id = _required_text(payload, "profile_id", issues)
    profile = CANONICAL_COORDINATE_PROFILES.get(profile_id) if profile_id is not None else None
    if profile_id is not None and profile is None:
        issues.append(_issue("profile_id_unsupported", "$.profile_id", f"Unsupported profile {profile_id!r}."))
    space_id = _required_text(payload, "space_id", issues)
    if space_id is not None:
        lowered = space_id.lower()
        if any(token in lowered for token in _FORBIDDEN_PRESENTATION_SPACE_TOKENS):
            issues.append(_issue("presentation_space_forbidden", "$.space_id", "Presentation viewport/display coordinates are ephemeral."))
        elif space_id not in COORDINATE_SPACE_IDS:
            issues.append(_issue("space_id_unsupported", "$.space_id", f"Unsupported canonical space {space_id!r}."))
    geometry_type = _required_text(payload, "geometry_type", issues)
    if geometry_type is not None and geometry_type not in GEOMETRY_TYPES:
        issues.append(_issue("geometry_type_unsupported", "$.geometry_type", f"Unsupported geometry {geometry_type!r}."))
    components, component_units = _parse_canonical_components(
        payload,
        geometry_type=geometry_type,
        issues=issues,
    )
    origin = _required_text(payload, "origin", issues)
    if origin is not None and origin not in ORIGINS:
        issues.append(_issue("origin_unsupported", "$.origin", f"Unsupported origin {origin!r}."))
    directions = _parse_positive_directions(payload.get("positive_directions"), issues)
    extent = _parse_canonical_reference_extent(payload.get("reference_extent"), issues)
    pixel_convention = _required_text(payload, "pixel_convention", issues)
    if pixel_convention is not None and pixel_convention not in PIXEL_CONVENTIONS:
        issues.append(_issue("pixel_convention_unsupported", "$.pixel_convention", f"Unsupported convention {pixel_convention!r}."))
    row_identity = _parse_canonical_row_identity(payload.get("row_identity"), issues)
    overlay = _parse_canonical_overlay(payload.get("source_camera_overlay"), issues)
    lineage_refs = _parse_canonical_record_refs(
        payload.get("lineage_refs"),
        path="$.lineage_refs",
        issues=issues,
        require_nonempty=True,
    )
    frame_record = None
    if "frame_record" in payload:
        frame_record = _parse_canonical_frame_record(payload["frame_record"], issues)
    collection_axis = None
    if "collection_axis" in payload:
        collection_axis = _parse_canonical_collection_axis(
            payload["collection_axis"],
            issues,
        )

    if issues:
        raise CoordinateDescriptorError(issues)
    assert profile_id is not None and profile is not None
    assert space_id is not None
    assert geometry_type is not None
    assert origin is not None
    assert directions is not None
    assert extent is not None
    assert pixel_convention is not None
    assert row_identity is not None
    assert overlay is not None
    descriptor = CanonicalCoordinateDescriptor(
        profile_id=profile_id,
        space_id=space_id,
        geometry_type=geometry_type,
        components=components,
        component_units=component_units,
        origin=origin,
        positive_directions=directions,
        reference_extent=extent,
        pixel_convention=pixel_convention,
        row_identity=row_identity,
        source_camera_overlay=overlay,
        lineage_refs=lineage_refs,
        collection_axis=collection_axis,
        frame_record=frame_record,
    )
    profile_issues = _canonical_profile_issues(descriptor)
    if profile_issues:
        raise CoordinateDescriptorError(profile_issues)
    return descriptor


def parse_canonical_coordinate_descriptor(
    value: Any,
) -> CanonicalCoordinateDescriptor:
    """Parse canonical schema v2 only; never dispatch to historical v1."""

    return _parse_canonical_coordinate_descriptor_v2(value)


def validate_canonical_coordinate_descriptor(
    value: Any,
) -> tuple[CoordinateIssue, ...]:
    try:
        _parse_canonical_coordinate_descriptor_v2(value)
    except CoordinateDescriptorError as exc:
        return exc.issues
    return ()


def _parsed_row_identity_contract(
    value: RowIdentityContract | Mapping[str, Any],
) -> RowIdentityContract:
    try:
        return parse_row_identity_contract(value)
    except RowIdentityContractError as exc:
        raise CoordinateDescriptorError(
            tuple(
                _issue(
                    f"row_identity_{item.code}",
                    f"$.row_identity_contract{item.path[1:]}",
                    item.message,
                )
                for item in exc.issues
            )
        ) from exc


def _validated_digest_bound_record(
    value: DigestBoundCoordinateRecordRef,
    *,
    path: str,
) -> DigestBoundCoordinateRecordRef:
    issues: list[CoordinateIssue] = []
    parsed = _parse_digest_bound_record_ref(
        value.to_dict() if isinstance(value, DigestBoundCoordinateRecordRef) else value,
        path=path,
        issues=issues,
    )
    if issues or parsed is None:
        raise CoordinateDescriptorError(issues)
    return parsed


def build_canonical_coordinate_descriptor(
    *,
    profile_id: str,
    geometry_type: str,
    components: Sequence[str],
    component_units: Sequence[str],
    reference_width: int | float | None,
    reference_height: int | float | None,
    reference_authority: DigestBoundCoordinateRecordRef,
    reference_selector: str,
    pixel_convention: str,
    row_identity_contract: RowIdentityContract | Mapping[str, Any],
    row_identity_record_ref: str,
    source_camera_overlay_status: str,
    overlay_transform_refs: Sequence[DigestBoundCoordinateRecordRef] = (),
    lineage_refs: Sequence[DigestBoundCoordinateRecordRef] = (),
    collection_axis: CanonicalCollectionAxis | None = None,
    frame_record: CanonicalFrameRecord | None = None,
) -> CanonicalCoordinateDescriptor:
    """Build one strict future descriptor from controlled profiles and records."""

    profile = CANONICAL_COORDINATE_PROFILES.get(profile_id)
    if profile is None:
        raise CoordinateDescriptorError(
            (_issue("profile_id_unsupported", "$.profile_id", f"Unsupported profile {profile_id!r}."),)
        )
    if profile.publication_status != "available":
        raise CoordinateDescriptorError(
            (
                _issue(
                    "profile_publication_unavailable",
                    "$.profile_id",
                    "This profile is reserved until an exact direction-labelled frame and transform authority exists.",
                ),
            )
        )
    authority = _validated_digest_bound_record(
        reference_authority,
        path="$.reference_extent.authority",
    )
    contract = _parsed_row_identity_contract(row_identity_contract)
    row_ref_issues: list[CoordinateIssue] = []
    canonical_row_ref = _canonical_archive_record_ref(
        row_identity_record_ref,
        path="$.row_identity.record_ref",
        issues=row_ref_issues,
    )
    if row_ref_issues or canonical_row_ref is None:
        raise CoordinateDescriptorError(row_ref_issues)

    combined_lineage: list[DigestBoundCoordinateRecordRef] = []
    seen_lineage: dict[str, str] = {}

    def add_lineage(record: DigestBoundCoordinateRecordRef, *, path: str) -> None:
        parsed = _validated_digest_bound_record(record, path=path)
        prior = seen_lineage.get(parsed.record_ref)
        if prior is not None:
            if prior != parsed.record_sha256:
                raise CoordinateDescriptorError(
                    (_issue("record_ref_digest_conflict", path, f"Record {parsed.record_ref!r} has conflicting digests."),)
                )
            return
        seen_lineage[parsed.record_ref] = parsed.record_sha256
        combined_lineage.append(parsed)

    add_lineage(authority, path="$.reference_extent.authority")
    if frame_record is not None:
        add_lineage(frame_record.record, path="$.frame_record")
    parsed_collection_axis: CanonicalCollectionAxis | None = None
    if collection_axis is not None:
        collection_issues: list[CoordinateIssue] = []
        parsed_collection_axis = _parse_canonical_collection_axis(
            (
                collection_axis.to_dict()
                if isinstance(collection_axis, CanonicalCollectionAxis)
                else collection_axis
            ),
            collection_issues,
        )
        if collection_issues or parsed_collection_axis is None:
            raise CoordinateDescriptorError(collection_issues)
        add_lineage(
            parsed_collection_axis.label_authority,
            path="$.collection_axis.label_authority",
        )
    for index, item in enumerate(lineage_refs):
        add_lineage(item, path=f"$.lineage_refs[{index}]")

    transform_refs = tuple(
        _validated_digest_bound_record(item, path=f"$.source_camera_overlay.transform_refs[{index}]")
        for index, item in enumerate(overlay_transform_refs)
    )
    overlay = CanonicalSourceCameraOverlay(
        status=source_camera_overlay_status,
        transform_refs=transform_refs,
        chain_direction=(
            CANONICAL_OVERLAY_CHAIN_DIRECTION
            if source_camera_overlay_status == CANONICAL_OVERLAY_REQUIRES_TRANSFORM
            else None
        ),
    )
    descriptor = CanonicalCoordinateDescriptor(
        profile_id=profile.profile_id,
        space_id=profile.space_id,
        geometry_type=geometry_type,
        components=tuple(components),
        component_units=tuple(component_units),
        origin=profile.origin,
        positive_directions=PositiveDirections(
            x=profile.positive_x,
            y=profile.positive_y,
        ),
        reference_extent=CanonicalReferenceExtent(
            width=reference_width,
            height=reference_height,
            units=profile.reference_units,
            authority=CanonicalReferenceAuthority(
                record_ref=authority.record_ref,
                record_sha256=authority.record_sha256,
                selector=reference_selector,
            ),
        ),
        pixel_convention=pixel_convention,
        row_identity=CanonicalRowIdentityRef(
            record_ref=canonical_row_ref,
            record_sha256=contract.digest(),
        ),
        source_camera_overlay=overlay,
        lineage_refs=tuple(combined_lineage),
        collection_axis=parsed_collection_axis,
        frame_record=frame_record,
    )
    return parse_canonical_coordinate_descriptor(descriptor)


def canonical_coordinate_descriptor_v2_json(value: Any) -> str:
    """Return deterministic JSON for a validated canonical v2 descriptor."""

    return _canonical_json(parse_canonical_coordinate_descriptor(value).to_dict())


def canonical_coordinate_descriptor_v2_digest(value: Any) -> str:
    return hashlib.sha256(
        canonical_coordinate_descriptor_v2_json(value).encode("utf-8")
    ).hexdigest()


def canonical_coordinate_descriptor_v2_attrs(
    value: Any,
    *,
    attr_name: str = COORDINATE_DESCRIPTOR_ATTR,
) -> dict[str, Any]:
    name = str(attr_name).strip()
    if not name:
        raise ValueError("attr_name must be non-empty.")
    descriptor = parse_canonical_coordinate_descriptor(value)
    return {
        name: descriptor.to_dict(),
        f"{name}{COORDINATE_DESCRIPTOR_DIGEST_SUFFIX}": descriptor.digest(),
    }


def verify_canonical_coordinate_descriptor_identity(
    value: Any,
    *,
    row_identity_contract: RowIdentityContract | Mapping[str, Any],
    expected_row_identity_record_ref: str,
    owner_shape: Sequence[int],
) -> CanonicalCoordinateDescriptor:
    """Bind descriptor metadata to the exact external identity record and rows."""

    descriptor = parse_canonical_coordinate_descriptor(value)
    contract = _parsed_row_identity_contract(row_identity_contract)
    issues: list[CoordinateIssue] = []
    expected_ref = _canonical_archive_record_ref(
        expected_row_identity_record_ref,
        path="$.row_identity.record_ref",
        issues=issues,
    )
    if expected_ref is not None and descriptor.row_identity.record_ref != expected_ref:
        issues.append(_issue("row_identity_record_ref_mismatch", "$.row_identity.record_ref", "Descriptor does not reference the resolved identity record."))
    if descriptor.row_identity.record_sha256 != contract.digest():
        issues.append(_issue("row_identity_record_digest_mismatch", "$.row_identity.record_sha256", "Descriptor digest does not match the resolved identity contract."))
    shape: tuple[int, ...] = ()
    if not isinstance(owner_shape, (tuple, list)):
        issues.append(_issue("owner_shape_invalid", "$.owner_shape", "Owner shape must be an integer sequence."))
    else:
        parsed_shape: list[int] = []
        for index, item in enumerate(owner_shape):
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                issues.append(
                    _issue(
                        "owner_shape_invalid",
                        f"$.owner_shape[{index}]",
                        "Owner dimensions must be exact nonnegative integers.",
                    )
                )
            else:
                parsed_shape.append(item)
        if len(parsed_shape) == len(owner_shape):
            shape = tuple(parsed_shape)
    if not shape:
        issues.append(_issue("owner_row_axis_missing", "$.owner_shape", "Canonical coordinate arrays require leading row axis 0."))
    elif shape[0] != contract.leading_dimension:
        issues.append(_issue("row_identity_count_mismatch", "$.owner_shape[0]", "Coordinate row count disagrees with the identity contract."))
    if shape:
        issues.extend(_canonical_geometry_shape_issues(descriptor, shape))
    if issues:
        raise CoordinateDescriptorError(issues)
    return descriptor


def _canonical_geometry_shape_issues(
    descriptor: CanonicalCoordinateDescriptor,
    shape: tuple[int, ...],
) -> tuple[CoordinateIssue, ...]:
    """Validate the physical array layout implied by one geometry type."""

    geometry = descriptor.geometry_type
    issues: list[CoordinateIssue] = []

    def exact_rank_and_components(rank: int, count: int) -> None:
        if len(shape) != rank or (len(shape) == rank and shape[-1] != count):
            issues.append(
                _issue(
                    "geometry_owner_shape_mismatch",
                    "$.owner_shape",
                    f"{geometry} requires physical shape (N, {count}).",
                )
            )

    collection = descriptor.collection_axis
    if collection is not None:
        if collection.axis >= len(shape):
            issues.append(
                _issue(
                    "collection_axis_out_of_bounds",
                    "$.collection_axis.axis",
                    "Collection axis is outside the physical owner shape.",
                )
            )
            return tuple(issues)
        if shape[collection.axis] != collection.cardinality:
            issues.append(
                _issue(
                    "collection_axis_cardinality_mismatch",
                    "$.owner_shape",
                    "Physical collection-axis cardinality disagrees with the digest-bound label authority.",
                )
            )
        expected_shape: tuple[int | None, ...] | None = None
        if geometry == "raster_yx":
            expected_shape = (None, collection.cardinality, None, None)
            if len(shape) == 4 and (shape[2] <= 0 or shape[3] <= 0):
                issues.append(
                    _issue(
                        "geometry_owner_shape_mismatch",
                        "$.owner_shape",
                        "A collected raster_yx requires positive H and W dimensions.",
                    )
                )
        elif geometry == "point_xy":
            expected_shape = (None, collection.cardinality, 2)
        elif geometry in {"bbox_xyxy", "bbox_xywh", "bbox_cxcywh"}:
            expected_shape = (None, collection.cardinality, 4)
        else:
            issues.append(
                _issue(
                    "collection_axis_geometry_unsupported",
                    "$.geometry_type",
                    "A controlled collection axis supports raster_yx, point_xy, "
                    "and bbox geometry.",
                )
            )
        if expected_shape is not None and (
            len(shape) != len(expected_shape)
            or any(
                expected is not None and actual != expected
                for actual, expected in zip(shape, expected_shape, strict=False)
            )
        ):
            issues.append(
                _issue(
                    "geometry_owner_shape_mismatch",
                    "$.owner_shape",
                    f"Collected {geometry} has an invalid physical layout for axis {collection.axis} and cardinality {collection.cardinality}.",
                )
            )
        return tuple(issues)

    if geometry in {"point_xy", "vector_xy"}:
        exact_rank_and_components(2, 2)
    elif geometry == "points_xy":
        if len(shape) != 3 or shape[-1] != 2 or shape[1] <= 0:
            issues.append(
                _issue(
                    "geometry_owner_shape_mismatch",
                    "$.owner_shape",
                    "points_xy requires physical shape (N, P, 2) with P > 0; use point_xy for one point per row.",
                )
            )
    elif geometry == "vector_sequence_xy":
        if len(shape) != 3 or shape[-1] != 2 or shape[1] <= 0:
            issues.append(
                _issue(
                    "geometry_owner_shape_mismatch",
                    "$.owner_shape",
                    "vector_sequence_xy requires physical shape (N, K, 2) with K > 0; use vector_xy for one vector per row.",
                )
            )
    elif geometry in {"bbox_xyxy", "bbox_xywh", "bbox_cxcywh", "line_segment_xyxy"}:
        exact_rank_and_components(2, 4)
    elif geometry == "circle_cxcy_r":
        exact_rank_and_components(2, 3)
    elif geometry == "ellipse_cxcy_wh_angle":
        exact_rank_and_components(2, 5)
    elif geometry in {"polyline_xy", "polygon_xy"}:
        if len(shape) != 3 or shape[-1] != 2 or shape[1] <= 0:
            issues.append(
                _issue(
                    "geometry_owner_shape_mismatch",
                    "$.owner_shape",
                    f"{geometry} requires physical shape (N, P, 2) with P > 0.",
                )
            )
    elif geometry == "raster_yx":
        if len(shape) != 3 or shape[1] <= 0 or shape[2] <= 0:
            issues.append(
                _issue(
                    "geometry_owner_shape_mismatch",
                    "$.owner_shape",
                    "raster_yx requires physical shape (N, H, W) with H,W > 0.",
                )
            )
    elif geometry in {"distance", "coordinate_component"}:
        if len(shape) != 1:
            issues.append(
                _issue(
                    "geometry_owner_shape_mismatch",
                    "$.owner_shape",
                    f"{geometry} requires physical shape (N,).",
                )
            )
    return tuple(issues)


def load_canonical_coordinate_descriptor_attrs(
    attrs: Mapping[str, Any],
    *,
    row_identity_contract: RowIdentityContract | Mapping[str, Any],
    expected_row_identity_record_ref: str,
    owner_shape: Sequence[int],
    attr_name: str = COORDINATE_DESCRIPTOR_ATTR,
) -> CanonicalCoordinateDescriptor:
    digest_name = f"{attr_name}{COORDINATE_DESCRIPTOR_DIGEST_SUFFIX}"
    issues: list[CoordinateIssue] = []
    if attr_name not in attrs:
        issues.append(_issue("descriptor_attr_missing", f"$.{attr_name}", "Descriptor attr is missing."))
    if digest_name not in attrs:
        issues.append(_issue("descriptor_digest_missing", f"$.{digest_name}", "Descriptor digest is missing."))
    if issues:
        raise CoordinateDescriptorError(issues)
    raw_descriptor = attrs[attr_name]
    descriptor = parse_canonical_coordinate_descriptor(raw_descriptor)
    if not isinstance(raw_descriptor, Mapping) or not _exact_json_equal(
        raw_descriptor,
        descriptor.to_dict(),
    ):
        raise CoordinateDescriptorError(
            (
                _issue(
                    "descriptor_persisted_form_noncanonical",
                    f"$.{attr_name}",
                    "Persisted canonical descriptor must exactly equal its canonical JSON mapping without numeric or container coercion.",
                ),
            )
        )
    stored = attrs[digest_name]
    if not isinstance(stored, str) or _SHA256_RE.fullmatch(stored) is None:
        raise CoordinateDescriptorError(
            (_issue("descriptor_digest_invalid", f"$.{digest_name}", "Descriptor digest must be lowercase SHA-256."),)
        )
    if stored != descriptor.digest():
        raise CoordinateDescriptorError(
            (_issue("descriptor_digest_mismatch", f"$.{digest_name}", "Descriptor digest does not match canonical content."),)
        )
    return verify_canonical_coordinate_descriptor_identity(
        descriptor,
        row_identity_contract=row_identity_contract,
        expected_row_identity_record_ref=expected_row_identity_record_ref,
        owner_shape=owner_shape,
    )


# Explicit historical-v1 names.  They intentionally remain separate from all
# canonical-v2 entry points and never upgrade a payload.
HistoricalCoordinateDescriptorV1 = CoordinateDescriptor
HistoricalCoordinateRecordRefV1 = CoordinateRecordRef


def parse_historical_coordinate_descriptor_v1(value: Any) -> CoordinateDescriptor:
    return parse_coordinate_descriptor(value)


def validate_historical_coordinate_descriptor_v1(
    value: Any,
) -> tuple[CoordinateIssue, ...]:
    return validate_coordinate_descriptor(value)


def build_historical_coordinate_descriptor_v1(**kwargs: Any) -> CoordinateDescriptor:
    return _build_historical_coordinate_descriptor_v1(**kwargs)


def historical_coordinate_descriptor_v1_json(value: Any) -> str:
    return canonical_coordinate_descriptor_json(value)


def historical_coordinate_descriptor_v1_digest(value: Any) -> str:
    return coordinate_descriptor_digest(value)


def historical_coordinate_descriptor_v1_attrs(
    value: Any,
    *,
    attr_name: str = COORDINATE_DESCRIPTOR_ATTR,
) -> dict[str, Any]:
    return _historical_coordinate_descriptor_v1_attrs(value, attr_name=attr_name)


def stamp_historical_coordinate_descriptor_v1(
    node: Any,
    value: Any,
    *,
    attr_name: str = COORDINATE_DESCRIPTOR_ATTR,
) -> CoordinateDescriptor:
    return _stamp_historical_coordinate_descriptor_v1(
        node,
        value,
        attr_name=attr_name,
    )


def load_historical_coordinate_descriptor_v1_attrs(
    attrs: Mapping[str, Any],
    *,
    attr_name: str = COORDINATE_DESCRIPTOR_ATTR,
) -> CoordinateDescriptor:
    return load_coordinate_descriptor_attrs(attrs, attr_name=attr_name)


__all__ = [
    "COORDINATE_DESCRIPTOR_ATTR",
    "COORDINATE_DESCRIPTOR_DIGEST_SUFFIX",
    "COORDINATE_DESCRIPTOR_SCHEMA_ID",
    "COORDINATE_DESCRIPTOR_SCHEMA_VERSION",
    "COORDINATE_DESCRIPTOR_CANONICALIZATION",
    "COORDINATE_SPACE_IDS",
    "PIXEL_SPACE_IDS",
    "NORMALIZED_SPACE_IDS",
    "GEOMETRY_TYPES",
    "COMPONENT_UNITS",
    "REFERENCE_UNITS",
    "ORIGINS",
    "POSITIVE_DIRECTIONS",
    "PIXEL_CONVENTIONS",
    "ROW_IDENTITY_MODES",
    "SOURCE_CAMERA_OVERLAY_STATUSES",
    "CoordinateIssue",
    "CoordinateDescriptorError",
    "PositiveDirections",
    "ReferenceExtent",
    "RowIdentityReference",
    "CoordinateRecordRef",
    "CoordinateDescriptor",
    "LegacySpaceContext",
    "parse_coordinate_descriptor",
    "validate_coordinate_descriptor",
    "canonical_coordinate_descriptor_json",
    "coordinate_descriptor_digest",
    "load_coordinate_descriptor_attrs",
    "resolve_legacy_space_id",
    "CANONICAL_COORDINATE_DESCRIPTOR_SCHEMA_VERSION",
    "CANONICAL_COORDINATE_DESCRIPTOR_CANONICALIZATION",
    "CANONICAL_COORDINATE_PROFILES",
    "CANONICAL_OVERLAY_CHAIN_DIRECTION",
    "CANONICAL_OVERLAY_DIRECT",
    "CANONICAL_OVERLAY_REQUIRES_TRANSFORM",
    "CANONICAL_OVERLAY_NOT_SUITABLE",
    "CANONICAL_OVERLAY_STATUSES",
    "PIXEL_FRAME_AUTHORITY_RECORD_KIND",
    "PHYSICAL_FRAME_CALIBRATION_RECORD_KIND",
    "FISH_BODY_FRAME_RECORD_KIND",
    "CANONICAL_FRAME_RECORD_KINDS",
    "CANONICAL_COLLECTION_AXIS_ROLES",
    "DigestBoundCoordinateRecordRef",
    "CanonicalReferenceAuthority",
    "CanonicalReferenceExtent",
    "CanonicalRowIdentityRef",
    "CanonicalSourceCameraOverlay",
    "CanonicalFrameRecord",
    "CanonicalCollectionAxis",
    "CanonicalCoordinateProfile",
    "CanonicalCoordinateDescriptor",
    "parse_canonical_coordinate_descriptor",
    "validate_canonical_coordinate_descriptor",
    "build_canonical_coordinate_descriptor",
    "canonical_coordinate_descriptor_v2_json",
    "canonical_coordinate_descriptor_v2_digest",
    "canonical_coordinate_descriptor_v2_attrs",
    "verify_canonical_coordinate_descriptor_identity",
    "load_canonical_coordinate_descriptor_attrs",
    "HistoricalCoordinateDescriptorV1",
    "HistoricalCoordinateRecordRefV1",
    "parse_historical_coordinate_descriptor_v1",
    "validate_historical_coordinate_descriptor_v1",
    "build_historical_coordinate_descriptor_v1",
    "historical_coordinate_descriptor_v1_json",
    "historical_coordinate_descriptor_v1_digest",
    "historical_coordinate_descriptor_v1_attrs",
    "stamp_historical_coordinate_descriptor_v1",
    "load_historical_coordinate_descriptor_v1_attrs",
]
