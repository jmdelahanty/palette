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

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence


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
    """Reference to the array that identifies the descriptor's leading rows."""

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
        return coordinate_descriptor_attrs(self, attr_name=attr_name)


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
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            return None, [
                _issue(
                    "descriptor_json_invalid",
                    "$",
                    f"Descriptor JSON is invalid: {exc.msg}.",
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
            digest = None
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
    if isinstance(version, bool) or version != COORDINATE_DESCRIPTOR_SCHEMA_VERSION:
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


def build_coordinate_descriptor(
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
    """Build and validate one canonical descriptor from typed arguments."""

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


def coordinate_descriptor_attrs(
    value: Any,
    *,
    attr_name: str = COORDINATE_DESCRIPTOR_ATTR,
) -> dict[str, Any]:
    """Return compact JSON-safe attrs for one descriptor and its digest."""

    name = str(attr_name).strip()
    if not name:
        raise ValueError("attr_name must be non-empty.")
    descriptor = parse_coordinate_descriptor(value)
    return {
        name: descriptor.to_dict(),
        f"{name}{COORDINATE_DESCRIPTOR_DIGEST_SUFFIX}": descriptor.digest(),
    }


def stamp_coordinate_descriptor(
    node: Any,
    value: Any,
    *,
    attr_name: str = COORDINATE_DESCRIPTOR_ATTR,
) -> CoordinateDescriptor:
    """Validate and stamp descriptor attrs on a Zarr-like node."""

    attrs = getattr(node, "attrs", None)
    if attrs is None:
        raise TypeError("node must expose an attrs mapping.")
    descriptor = parse_coordinate_descriptor(value)
    attrs.update(coordinate_descriptor_attrs(descriptor, attr_name=attr_name))
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
    "build_coordinate_descriptor",
    "canonical_coordinate_descriptor_json",
    "coordinate_descriptor_digest",
    "coordinate_descriptor_attrs",
    "stamp_coordinate_descriptor",
    "load_coordinate_descriptor_attrs",
    "resolve_legacy_space_id",
]
