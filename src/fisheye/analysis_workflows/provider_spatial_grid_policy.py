"""Immutable arena-millimetre grid authority for provider spatial analytics.

The grid in this module is a scientific coordinate authority, not a display
choice.  Its extent is derived from a declared circular physical rim and an
exact source-camera scale.  Observed provider positions are deliberately not
accepted as an input: changing the observed data must never change the bins.

The GoodBatBadBat recommendation is one-millimetre bins.  The one-millimetre
rule is explicit and versioned, while the outer extent is rounded outward to
the smallest symmetric multiple of that width that covers the declared rim.
For the current arena-2 canary this yields approximately ``[-41, 41]`` mm,
because the reviewed radius is about 40.935 mm.  The older nominal ``[-40,40]``
extent is therefore not used.

The selected Palette canary boundary is intentionally represented as
``visible_dish_top_rim_edge``.  That is a reviewed arena boundary, not an
operator-confirmed ``physical_inner_rim`` and not the outward forgiving
bounding-box centroid gate.  These meanings remain distinct in the policy
record; the grid uses the exact selected reviewed circle only because it is the
declared geometry authority for that recording.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Final, Mapping

import numpy as np

from fisheye.analysis.provider_occupancy_v2 import OccupancyGrid


ARENA_MM_GRID_POLICY_SCHEMA_ID: Final = "palette.arena_mm_grid_policy"
ARENA_MM_GRID_POLICY_SCHEMA_VERSION: Final = 1
ARENA_MM_GRID_SOURCE_BINDING_SCHEMA_ID: Final = (
    "palette.arena_mm_grid_source_binding"
)
ARENA_MM_GRID_SOURCE_BINDING_SCHEMA_VERSION: Final = 2
ARENA_MM_GRID_SOURCE_BINDING_SUPPORTED_SCHEMA_VERSIONS: Final = frozenset({1, 2})

GOODBATBADBAT_GRID_POLICY_ID: Final = "goodbatbadbat_arena_mm_grid_v1"
GOODBATBADBAT_BIN_WIDTH_MM: Final = 1.0
BIN_WIDTH_RULE_ID: Final = "declared_1mm_round_outward_symmetric_extent_v1"
EXTENT_RULE_ID: Final = "circular_rim_radius_round_outward_symmetric_square_v1"
GRID_COORDINATE_SPACE_ID: Final = "arena_centered_mm_y_down_v1"
GEOMETRY_COORDINATE_SPACE_ID: Final = "camera_native_pixels_y_down_v1"
PHYSICAL_RIM_BOUNDARY_ROLE: Final = "physical_inner_rim"
REVIEWED_TOP_RIM_BOUNDARY_ROLE: Final = "visible_dish_top_rim_edge"
REJECTED_DETECTION_GATE_BOUNDARY_ROLE: Final = (
    "bounding_box_centroid_detection_gating"
)
EDGE_POLICY_ID: Final = "left_closed_right_open_final_outer_edge_inclusive_v1"

_SHA256_LENGTH = hashlib.sha256().digest_size * 2
_MUTABLE_ALIASES = frozenset(
    {
        "active",
        "authoritative",
        "current",
        "default",
        "fallback",
        "latest",
        "latest_complete",
        "none",
        "null",
        "selected",
        "stale",
        "unknown",
    }
)


class ArenaMMGridPolicyError(ValueError):
    """Raised when a grid authority is incomplete, stale, or unsafe."""


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ArenaMMGridPolicyError("Grid authority is not strict JSON.") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _array_payload_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _require_identity(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(character.isspace() for character in value)
        or value.lower() in _MUTABLE_ALIASES
        or "/" in value
        or "@" in value
    ):
        raise ArenaMMGridPolicyError(
            f"{field} must be one exact immutable identity, not a selector."
        )
    return value


def _require_reference(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(character.isspace() for character in value)
        or value.lower() in _MUTABLE_ALIASES
    ):
        raise ArenaMMGridPolicyError(
            f"{field} must be one exact immutable record reference."
        )
    return value


def _require_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ArenaMMGridPolicyError(f"{field} must be a lowercase SHA-256 digest.")
    return value


def _finite_positive(value: object, *, field: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ArenaMMGridPolicyError(f"{field} must be finite and positive.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ArenaMMGridPolicyError(f"{field} must be finite and positive.") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ArenaMMGridPolicyError(f"{field} must be finite and positive.")
    return result


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ArenaMMGridPolicyError(f"{field} must be finite and numeric.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ArenaMMGridPolicyError(f"{field} must be finite and numeric.") from exc
    if not math.isfinite(result):
        raise ArenaMMGridPolicyError(f"{field} must be finite and numeric.")
    return result


def _record_with_digest(payload: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    """Validate a digest-bearing record and return a detached canonical copy."""

    if not isinstance(payload, Mapping):
        raise ArenaMMGridPolicyError(f"{field} must be a mapping.")
    record = dict(payload)
    supplied = _require_sha256(record.pop("record_sha256", None), field=f"{field}.record_sha256")
    expected = _digest(record)
    if supplied != expected:
        raise ArenaMMGridPolicyError(f"{field} has a stale record_sha256.")
    record["record_sha256"] = supplied
    return json.loads(_canonical_json(record))


def _readonly_edges(values: Any, *, field: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.dtype.kind not in "iuf":
        raise ArenaMMGridPolicyError(f"{field} must be a one-dimensional numeric array.")
    result = np.asarray(raw, dtype=np.float64).copy()
    if result.size < 2 or not np.isfinite(result).all() or not np.all(np.diff(result) > 0.0):
        raise ArenaMMGridPolicyError(f"{field} must be finite and strictly increasing.")
    result.setflags(write=False)
    return result


def _symmetric_edges(*, radius_mm: float, bin_width_mm: float) -> tuple[np.ndarray, float]:
    """Return float64 edges whose symmetric outer edge covers ``radius_mm``."""

    bins_to_outer_edge = max(1, math.ceil(radius_mm / bin_width_mm))
    outer = float(bins_to_outer_edge) * bin_width_mm
    # Guard the rare case where the division/rounding boundary loses one ULP.
    while outer < radius_mm:
        bins_to_outer_edge += 1
        outer = float(bins_to_outer_edge) * bin_width_mm
    values = (
        np.arange(-bins_to_outer_edge, bins_to_outer_edge + 1, dtype=np.float64)
        * np.float64(bin_width_mm)
    )
    values = np.asarray(values, dtype=np.float64)
    values.setflags(write=False)
    if values[0] > -radius_mm or values[-1] < radius_mm:
        raise ArenaMMGridPolicyError("Derived grid does not cover the declared rim.")
    if not np.array_equal(values, -values[::-1]):
        raise ArenaMMGridPolicyError("Derived grid edges are not exactly symmetric.")
    return values, outer


@dataclass(frozen=True)
class CircularArenaGeometryAuthority:
    """Digest-bound reviewed circular boundary in native camera pixels.

    ``boundary_role`` and ``observed_feature`` are retained verbatim as
    scientific semantics.  A visible top-rim review is not rewritten as a
    physical inner-rim observation.
    """

    geometry_id: str
    coordinate_authority_id: str
    center_x_px: float
    center_y_px: float
    radius_px: float
    record_ref: str = "inline_circular_arena_geometry"
    boundary_role: str = PHYSICAL_RIM_BOUNDARY_ROLE
    observed_feature: str | None = None
    coordinate_space: str = GEOMETRY_COORDINATE_SPACE_ID
    record_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "geometry_id", _require_identity(self.geometry_id, field="geometry_id"))
        object.__setattr__(
            self,
            "coordinate_authority_id",
            _require_identity(self.coordinate_authority_id, field="coordinate_authority_id"),
        )
        object.__setattr__(self, "record_ref", _require_reference(self.record_ref, field="geometry.record_ref"))
        if self.boundary_role not in {
            PHYSICAL_RIM_BOUNDARY_ROLE,
            REVIEWED_TOP_RIM_BOUNDARY_ROLE,
        }:
            raise ArenaMMGridPolicyError(
                "Grid extent must be derived from a declared physical inner rim or "
                "an explicitly reviewed visible top-rim boundary; the outward "
                "detection gate is not a grid geometry authority."
            )
        observed_feature = self.observed_feature
        if observed_feature is None:
            observed_feature = (
                "dish_inner_rim_water_side_edge"
                if self.boundary_role == PHYSICAL_RIM_BOUNDARY_ROLE
                else REVIEWED_TOP_RIM_BOUNDARY_ROLE
            )
        if not isinstance(observed_feature, str) or not observed_feature.strip():
            raise ArenaMMGridPolicyError("observed_feature must be explicit text.")
        if self.boundary_role == REVIEWED_TOP_RIM_BOUNDARY_ROLE and observed_feature != REVIEWED_TOP_RIM_BOUNDARY_ROLE:
            raise ArenaMMGridPolicyError(
                "visible_dish_top_rim_edge must retain its exact observed_feature."
            )
        if self.boundary_role == PHYSICAL_RIM_BOUNDARY_ROLE and observed_feature == REVIEWED_TOP_RIM_BOUNDARY_ROLE:
            raise ArenaMMGridPolicyError(
                "A visible top-rim observation cannot be relabeled as physical_inner_rim."
            )
        object.__setattr__(self, "observed_feature", observed_feature)
        if self.coordinate_space != GEOMETRY_COORDINATE_SPACE_ID:
            raise ArenaMMGridPolicyError(
                "Circular geometry must be declared in native camera pixels; pixel-grid "
                "or arena-mm inputs cannot be used as a scale source."
            )
        object.__setattr__(self, "center_x_px", _finite_number(self.center_x_px, field="center_x_px"))
        object.__setattr__(self, "center_y_px", _finite_number(self.center_y_px, field="center_y_px"))
        object.__setattr__(self, "radius_px", _finite_positive(self.radius_px, field="radius_px"))
        expected = _digest(self.payload())
        if self.record_sha256 is not None and _require_sha256(self.record_sha256, field="geometry.record_sha256") != expected:
            raise ArenaMMGridPolicyError("geometry has a stale record_sha256.")
        object.__setattr__(self, "record_sha256", expected)

    def payload(self) -> dict[str, Any]:
        return {
            "schema_id": "palette.circular_arena_geometry_authority",
            "schema_version": 1,
            "geometry_id": self.geometry_id,
            "coordinate_authority_id": self.coordinate_authority_id,
            "coordinate_space": self.coordinate_space,
            "boundary_role": self.boundary_role,
            "observed_feature": self.observed_feature,
            "center_x_px": self.center_x_px,
            "center_y_px": self.center_y_px,
            "radius_px": self.radius_px,
            "record_ref": self.record_ref,
        }

    def as_record(self) -> dict[str, Any]:
        return {**self.payload(), "record_sha256": self.record_sha256}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "CircularArenaGeometryAuthority":
        normalized = _record_with_digest(record, field="geometry")
        if normalized.get("schema_id") != "palette.circular_arena_geometry_authority" or normalized.get("schema_version") != 1:
            raise ArenaMMGridPolicyError("Unsupported circular geometry authority schema.")
        expected = {
            "geometry_id": normalized.get("geometry_id"),
            "coordinate_authority_id": normalized.get("coordinate_authority_id"),
            "coordinate_space": normalized.get("coordinate_space"),
            "boundary_role": normalized.get("boundary_role"),
            "observed_feature": normalized.get("observed_feature"),
            "center_x_px": normalized.get("center_x_px"),
            "center_y_px": normalized.get("center_y_px"),
            "radius_px": normalized.get("radius_px"),
            "record_ref": normalized.get("record_ref", "inline_circular_arena_geometry"),
            "record_sha256": normalized["record_sha256"],
        }
        return cls(**expected)


@dataclass(frozen=True)
class PhysicalScaleAuthority:
    """Digest-bound source-camera physical scale."""

    scale_id: str
    coordinate_authority_id: str
    mm_per_pixel: float
    record_ref: str = "inline_source_camera_physical_scale"
    coordinate_space: str = GEOMETRY_COORDINATE_SPACE_ID
    record_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "scale_id", _require_identity(self.scale_id, field="scale_id"))
        object.__setattr__(
            self,
            "coordinate_authority_id",
            _require_identity(self.coordinate_authority_id, field="scale.coordinate_authority_id"),
        )
        object.__setattr__(self, "record_ref", _require_reference(self.record_ref, field="scale.record_ref"))
        if self.coordinate_space != GEOMETRY_COORDINATE_SPACE_ID:
            raise ArenaMMGridPolicyError("Physical scale must be bound to native camera pixels.")
        object.__setattr__(self, "mm_per_pixel", _finite_positive(self.mm_per_pixel, field="mm_per_pixel"))
        expected = _digest(self.payload())
        if self.record_sha256 is not None and _require_sha256(self.record_sha256, field="scale.record_sha256") != expected:
            raise ArenaMMGridPolicyError("scale has a stale record_sha256.")
        object.__setattr__(self, "record_sha256", expected)

    def payload(self) -> dict[str, Any]:
        return {
            "schema_id": "palette.source_camera_physical_scale_authority",
            "schema_version": 1,
            "scale_id": self.scale_id,
            "coordinate_authority_id": self.coordinate_authority_id,
            "coordinate_space": self.coordinate_space,
            "units": "mm_per_pixel",
            "mm_per_pixel": self.mm_per_pixel,
            "record_ref": self.record_ref,
        }

    def as_record(self) -> dict[str, Any]:
        return {**self.payload(), "record_sha256": self.record_sha256}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "PhysicalScaleAuthority":
        normalized = _record_with_digest(record, field="scale")
        return cls(
            scale_id=normalized.get("scale_id"),
            coordinate_authority_id=normalized.get("coordinate_authority_id"),
            coordinate_space=normalized.get("coordinate_space"),
            mm_per_pixel=normalized.get("mm_per_pixel"),
            record_ref=normalized.get("record_ref", "inline_source_camera_physical_scale"),
            record_sha256=normalized["record_sha256"],
        )


@dataclass(frozen=True)
class SelectionAuthority:
    """Exact selected-geometry/stimulus selection identity used by the grid."""

    selection_id: str
    recording_id: str
    record_sha256: str
    record_ref: str = "inline_arena_geometry_selection"
    record_payload: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "selection_id", _require_identity(self.selection_id, field="selection_id"))
        object.__setattr__(self, "recording_id", _require_identity(self.recording_id, field="recording_id"))
        object.__setattr__(self, "record_sha256", _require_sha256(self.record_sha256, field="selection.record_sha256"))
        object.__setattr__(self, "record_ref", _require_reference(self.record_ref, field="selection.record_ref"))
        if self.record_payload is not None:
            payload = json.loads(_canonical_json(dict(self.record_payload)))
            if _digest(payload) != self.record_sha256:
                raise ArenaMMGridPolicyError("selection has a stale record_sha256.")
            object.__setattr__(self, "record_payload", payload)

    def as_record(self) -> dict[str, Any]:
        return {
            "schema_id": "palette.arena_geometry_selection_authority",
            "schema_version": 1,
            "selection_id": self.selection_id,
            "recording_id": self.recording_id,
            "record_ref": self.record_ref,
            "record_sha256": self.record_sha256,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "SelectionAuthority":
        if not isinstance(record, Mapping):
            raise ArenaMMGridPolicyError("selection must be a mapping.")
        return cls(
            selection_id=record.get("selection_id"),
            recording_id=record.get("recording_id"),
            record_sha256=record.get("record_sha256"),
            record_ref=record.get("record_ref", "inline_arena_geometry_selection"),
            record_payload=record.get("record_payload"),
        )

    @classmethod
    def from_payload(
        cls,
        *,
        selection_id: str,
        recording_id: str,
        payload: Mapping[str, Any],
        record_ref: str = "inline_arena_geometry_selection",
    ) -> "SelectionAuthority":
        normalized = json.loads(_canonical_json(dict(payload)))
        return cls(
            selection_id=selection_id,
            recording_id=recording_id,
            record_sha256=_digest(normalized),
            record_ref=record_ref,
            record_payload=normalized,
        )


@dataclass(frozen=True)
class ArenaMMGridPolicy:
    """Immutable, digest-bound fixed grid derived from declared authorities."""

    policy_id: str
    recording_id: str
    geometry: CircularArenaGeometryAuthority
    scale: PhysicalScaleAuthority
    selection: SelectionAuthority
    bin_width_mm: float
    x_edges: np.ndarray
    y_edges: np.ndarray
    extent_radius_mm: float
    rim_radius_mm: float
    bin_width_rule_id: str = BIN_WIDTH_RULE_ID
    extent_rule_id: str = EXTENT_RULE_ID
    grid_coordinate_space: str = GRID_COORDINATE_SPACE_ID
    edge_policy_id: str = EDGE_POLICY_ID
    record_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _require_identity(self.policy_id, field="policy_id"))
        object.__setattr__(self, "recording_id", _require_identity(self.recording_id, field="recording_id"))
        if self.recording_id != self.selection.recording_id:
            raise ArenaMMGridPolicyError("Grid recording_id disagrees with selection authority.")
        if self.bin_width_rule_id != BIN_WIDTH_RULE_ID or self.extent_rule_id != EXTENT_RULE_ID:
            raise ArenaMMGridPolicyError("Unsupported arena-mm grid policy rule.")
        if self.grid_coordinate_space != GRID_COORDINATE_SPACE_ID:
            raise ArenaMMGridPolicyError("Grid coordinate space must be arena-centred millimetres.")
        if self.edge_policy_id != EDGE_POLICY_ID:
            raise ArenaMMGridPolicyError("Unsupported occupancy edge policy.")
        width = _finite_positive(self.bin_width_mm, field="bin_width_mm")
        rim = _finite_positive(self.rim_radius_mm, field="rim_radius_mm")
        extent = _finite_positive(self.extent_radius_mm, field="extent_radius_mm")
        if extent < rim:
            raise ArenaMMGridPolicyError("Grid extent does not cover the declared physical rim.")
        x_edges = _readonly_edges(self.x_edges, field="x_edges")
        y_edges = _readonly_edges(self.y_edges, field="y_edges")
        if not np.array_equal(x_edges, y_edges) or not np.array_equal(x_edges, -x_edges[::-1]):
            raise ArenaMMGridPolicyError("Arena-mm grid edges must be identical and symmetric.")
        if not math.isclose(float(np.diff(x_edges).min()), width, rel_tol=0.0, abs_tol=0.0):
            raise ArenaMMGridPolicyError("Grid edges do not use the declared bin width.")
        if x_edges[0] > -extent or x_edges[-1] < extent:
            raise ArenaMMGridPolicyError("Grid edges do not cover the declared extent.")
        object.__setattr__(self, "bin_width_mm", width)
        object.__setattr__(self, "rim_radius_mm", rim)
        object.__setattr__(self, "extent_radius_mm", extent)
        object.__setattr__(self, "x_edges", x_edges)
        object.__setattr__(self, "y_edges", y_edges)
        expected = _digest(self.payload())
        if self.record_sha256 is not None and _require_sha256(self.record_sha256, field="policy.record_sha256") != expected:
            raise ArenaMMGridPolicyError("Grid policy has a stale record_sha256.")
        object.__setattr__(self, "record_sha256", expected)

    @property
    def schema_id(self) -> str:
        return ARENA_MM_GRID_POLICY_SCHEMA_ID

    @property
    def schema_version(self) -> int:
        return ARENA_MM_GRID_POLICY_SCHEMA_VERSION

    def payload(self) -> dict[str, Any]:
        return {
            "schema_id": ARENA_MM_GRID_POLICY_SCHEMA_ID,
            "schema_version": ARENA_MM_GRID_POLICY_SCHEMA_VERSION,
            "policy_id": self.policy_id,
            "recording_id": self.recording_id,
            "grid_coordinate_space": self.grid_coordinate_space,
            "edge_policy_id": self.edge_policy_id,
            "bin_width_rule_id": self.bin_width_rule_id,
            "bin_width_mm": self.bin_width_mm,
            "extent_rule_id": self.extent_rule_id,
            "rim_radius_mm": self.rim_radius_mm,
            "extent_radius_mm": self.extent_radius_mm,
            "x_edges_float64": self.x_edges.tolist(),
            "y_edges_float64": self.y_edges.tolist(),
            "geometry": self.geometry.as_record(),
            "scale": self.scale.as_record(),
            "selection": self.selection.as_record(),
        }

    def as_record(self) -> dict[str, Any]:
        return {**self.payload(), "record_sha256": self.record_sha256}

    @property
    def policy_digest(self) -> str:
        return str(self.record_sha256)

    def to_occupancy_grid(self) -> OccupancyGrid:
        return OccupancyGrid(
            x_edges=self.x_edges,
            y_edges=self.y_edges,
            edge_policy_id=self.edge_policy_id,
        )

    def source_binding_authority_record(self) -> dict[str, Any]:
        payload = {
            "schema_id": ARENA_MM_GRID_SOURCE_BINDING_SCHEMA_ID,
            "schema_version": ARENA_MM_GRID_SOURCE_BINDING_SCHEMA_VERSION,
            "recording_id": self.recording_id,
            "grid_policy_id": self.policy_id,
            "grid_policy_record_sha256": self.policy_digest,
            "grid_coordinate_space": self.grid_coordinate_space,
            "geometry": {
                "geometry_id": self.geometry.geometry_id,
                "record_ref": self.geometry.record_ref,
                "record_sha256": self.geometry.record_sha256,
                "coordinate_authority_id": self.geometry.coordinate_authority_id,
                "coordinate_space": self.geometry.coordinate_space,
                "boundary_role": self.geometry.boundary_role,
                "observed_feature": self.geometry.observed_feature,
            },
            "scale": {
                "scale_id": self.scale.scale_id,
                "record_ref": self.scale.record_ref,
                "record_sha256": self.scale.record_sha256,
                "coordinate_space": self.scale.coordinate_space,
                "mm_per_pixel": self.scale.mm_per_pixel,
            },
            "selection": {
                "selection_id": self.selection.selection_id,
                "record_ref": self.selection.record_ref,
                "record_sha256": self.selection.record_sha256,
            },
            "edge_policy_id": self.edge_policy_id,
            "x_edges_sha256": _array_payload_sha256(self.x_edges),
            "y_edges_sha256": _array_payload_sha256(self.y_edges),
            "bin_width_rule_id": self.bin_width_rule_id,
            "bin_width_mm": self.bin_width_mm,
            "extent_rule_id": self.extent_rule_id,
        }
        return {**payload, "record_sha256": _digest(payload)}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "ArenaMMGridPolicy":
        normalized = _record_with_digest(record, field="grid policy")
        geometry = CircularArenaGeometryAuthority.from_record(normalized.get("geometry"))
        scale = PhysicalScaleAuthority.from_record(normalized.get("scale"))
        selection = SelectionAuthority.from_record(normalized.get("selection"))
        return cls(
            policy_id=normalized.get("policy_id"),
            recording_id=normalized.get("recording_id"),
            geometry=geometry,
            scale=scale,
            selection=selection,
            bin_width_mm=normalized.get("bin_width_mm"),
            x_edges=normalized.get("x_edges_float64"),
            y_edges=normalized.get("y_edges_float64"),
            extent_radius_mm=normalized.get("extent_radius_mm"),
            rim_radius_mm=normalized.get("rim_radius_mm"),
            bin_width_rule_id=normalized.get("bin_width_rule_id"),
            extent_rule_id=normalized.get("extent_rule_id"),
            grid_coordinate_space=normalized.get("grid_coordinate_space"),
            edge_policy_id=normalized.get("edge_policy_id"),
            record_sha256=normalized["record_sha256"],
        )


def build_arena_mm_grid_policy(
    *,
    recording_id: str,
    geometry: CircularArenaGeometryAuthority,
    scale: PhysicalScaleAuthority,
    selection: SelectionAuthority,
    bin_width_mm: float = GOODBATBADBAT_BIN_WIDTH_MM,
    policy_id: str = GOODBATBADBAT_GRID_POLICY_ID,
    observed_positions: object = None,
) -> ArenaMMGridPolicy:
    """Build one fixed arena-mm grid from declared authorities.

    ``observed_positions`` is present only to make accidental data-dependent
    use fail loudly; any non-``None`` value is rejected.
    """

    if observed_positions is not None:
        raise ArenaMMGridPolicyError(
            "Grid extent is authority-derived; observed positions are forbidden."
        )
    if type(geometry) is not CircularArenaGeometryAuthority:
        raise ArenaMMGridPolicyError("geometry must be a verified circular geometry authority.")
    if type(scale) is not PhysicalScaleAuthority:
        raise ArenaMMGridPolicyError("scale must be a verified physical scale authority.")
    if type(selection) is not SelectionAuthority:
        raise ArenaMMGridPolicyError("selection must be a verified selection authority.")
    recording = _require_identity(recording_id, field="recording_id")
    if recording != selection.recording_id:
        raise ArenaMMGridPolicyError("recording_id disagrees with selection authority.")
    if geometry.coordinate_authority_id != scale.coordinate_authority_id:
        raise ArenaMMGridPolicyError("geometry and scale use different coordinate authorities.")
    width = _finite_positive(bin_width_mm, field="bin_width_mm")
    rim_radius_mm = geometry.radius_px * scale.mm_per_pixel
    edges, extent_radius_mm = _symmetric_edges(
        radius_mm=rim_radius_mm,
        bin_width_mm=width,
    )
    return ArenaMMGridPolicy(
        policy_id=policy_id,
        recording_id=recording,
        geometry=geometry,
        scale=scale,
        selection=selection,
        bin_width_mm=width,
        x_edges=edges,
        y_edges=edges.copy(),
        extent_radius_mm=extent_radius_mm,
        rim_radius_mm=rim_radius_mm,
    )


def validate_source_binding_authority_record(record: Mapping[str, Any]) -> None:
    """Validate an exact source-binding record without opening production data."""

    normalized = _record_with_digest(record, field="grid source binding")
    common_keys = {
        "schema_id",
        "schema_version",
        "recording_id",
        "grid_policy_id",
        "grid_policy_record_sha256",
        "grid_coordinate_space",
        "geometry",
        "scale",
        "selection",
        "edge_policy_id",
        "bin_width_rule_id",
        "bin_width_mm",
        "extent_rule_id",
        "record_sha256",
    }
    expected_keys = set(common_keys)
    if normalized.get("schema_version") == 2:
        expected_keys.update({"x_edges_sha256", "y_edges_sha256"})
    if set(normalized) != expected_keys:
        raise ArenaMMGridPolicyError("Grid source binding has unexpected fields.")
    if (
        normalized["schema_id"] != ARENA_MM_GRID_SOURCE_BINDING_SCHEMA_ID
        or normalized["schema_version"]
        not in ARENA_MM_GRID_SOURCE_BINDING_SUPPORTED_SCHEMA_VERSIONS
    ):
        raise ArenaMMGridPolicyError("Unsupported grid source-binding schema.")
    if normalized["schema_version"] == 2:
        _require_sha256(normalized["x_edges_sha256"], field="binding.x_edges_sha256")
        _require_sha256(normalized["y_edges_sha256"], field="binding.y_edges_sha256")
    _require_identity(normalized["recording_id"], field="binding.recording_id")
    _require_identity(normalized["grid_policy_id"], field="binding.grid_policy_id")
    _require_sha256(normalized["grid_policy_record_sha256"], field="binding.grid_policy_record_sha256")
    if normalized["grid_coordinate_space"] != GRID_COORDINATE_SPACE_ID:
        raise ArenaMMGridPolicyError("Grid source binding is not arena-mm.")
    if normalized["edge_policy_id"] != EDGE_POLICY_ID or normalized["bin_width_rule_id"] != BIN_WIDTH_RULE_ID or normalized["extent_rule_id"] != EXTENT_RULE_ID:
        raise ArenaMMGridPolicyError("Grid source binding uses an unsupported policy.")
    _finite_positive(normalized["bin_width_mm"], field="binding.bin_width_mm")
    geometry = normalized["geometry"]
    scale = normalized["scale"]
    selection = normalized["selection"]
    if not all(isinstance(value, Mapping) for value in (geometry, scale, selection)):
        raise ArenaMMGridPolicyError("binding geometry, scale, and selection must be mappings.")
    if set(geometry) != {
        "geometry_id",
        "record_ref",
        "record_sha256",
        "coordinate_authority_id",
        "coordinate_space",
        "boundary_role",
        "observed_feature",
    }:
        raise ArenaMMGridPolicyError("binding.geometry has unexpected fields.")
    if set(scale) != {
        "scale_id",
        "record_ref",
        "record_sha256",
        "coordinate_space",
        "mm_per_pixel",
    }:
        raise ArenaMMGridPolicyError("binding.scale has unexpected fields.")
    if set(selection) != {"selection_id", "record_ref", "record_sha256"}:
        raise ArenaMMGridPolicyError("binding.selection has unexpected fields.")
    _require_identity(geometry["geometry_id"], field="binding.geometry.geometry_id")
    _require_identity(
        geometry["coordinate_authority_id"],
        field="binding.geometry.coordinate_authority_id",
    )
    _require_reference(geometry["record_ref"], field="binding.geometry.record_ref")
    _require_sha256(geometry["record_sha256"], field="binding.geometry.record_sha256")
    if geometry["coordinate_space"] != GEOMETRY_COORDINATE_SPACE_ID:
        raise ArenaMMGridPolicyError("binding.geometry is not native camera pixels.")
    if geometry["boundary_role"] not in {
        PHYSICAL_RIM_BOUNDARY_ROLE,
        REVIEWED_TOP_RIM_BOUNDARY_ROLE,
    }:
        raise ArenaMMGridPolicyError("binding.geometry uses a forbidden boundary role.")
    if geometry["boundary_role"] == REVIEWED_TOP_RIM_BOUNDARY_ROLE and geometry["observed_feature"] != REVIEWED_TOP_RIM_BOUNDARY_ROLE:
        raise ArenaMMGridPolicyError("binding.geometry relabels the reviewed top rim.")
    _require_identity(scale["scale_id"], field="binding.scale.scale_id")
    _require_reference(scale["record_ref"], field="binding.scale.record_ref")
    _require_sha256(scale["record_sha256"], field="binding.scale.record_sha256")
    if scale["coordinate_space"] != GEOMETRY_COORDINATE_SPACE_ID:
        raise ArenaMMGridPolicyError("binding.scale is not native camera pixels.")
    _finite_positive(scale["mm_per_pixel"], field="binding.scale.mm_per_pixel")
    _require_identity(selection["selection_id"], field="binding.selection.selection_id")
    _require_reference(selection["record_ref"], field="binding.selection.record_ref")
    _require_sha256(selection["record_sha256"], field="binding.selection.record_sha256")


__all__ = [
    "ARENA_MM_GRID_POLICY_SCHEMA_ID",
    "ARENA_MM_GRID_POLICY_SCHEMA_VERSION",
    "ARENA_MM_GRID_SOURCE_BINDING_SCHEMA_ID",
    "ARENA_MM_GRID_SOURCE_BINDING_SCHEMA_VERSION",
    "ARENA_MM_GRID_SOURCE_BINDING_SUPPORTED_SCHEMA_VERSIONS",
    "BIN_WIDTH_RULE_ID",
    "CircularArenaGeometryAuthority",
    "EXTENT_RULE_ID",
    "GEOMETRY_COORDINATE_SPACE_ID",
    "GOODBATBADBAT_BIN_WIDTH_MM",
    "GOODBATBADBAT_GRID_POLICY_ID",
    "GRID_COORDINATE_SPACE_ID",
    "PHYSICAL_RIM_BOUNDARY_ROLE",
    "REJECTED_DETECTION_GATE_BOUNDARY_ROLE",
    "REVIEWED_TOP_RIM_BOUNDARY_ROLE",
    "ArenaMMGridPolicy",
    "ArenaMMGridPolicyError",
    "PhysicalScaleAuthority",
    "SelectionAuthority",
    "build_arena_mm_grid_policy",
    "validate_source_binding_authority_record",
]
