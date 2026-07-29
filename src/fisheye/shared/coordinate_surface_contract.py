"""Shared semantic templates for persisted coordinate-bearing arrays.

The canonical coordinate descriptor remains the per-array persisted authority:
it binds concrete extents, row identity, lineage records, and directed
transforms.  This module owns the smaller schema-level vocabulary that writers,
readers, and storage manifests can share before those run-specific values are
known.

In particular, ``source_camera_image_px`` means continuous source-camera pixel
coordinates unless a surface explicitly declares a different pixel
convention.  Integer extraction indices and pixel extents are deliberately
separate surface kinds; they are not silently treated as display points.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from fisheye.shared.coordinate_descriptor import (
    CANONICAL_COORDINATE_PROFILES,
    CANONICAL_OVERLAY_DIRECT,
    CANONICAL_OVERLAY_REQUIRES_TRANSFORM,
)


COORDINATE_SURFACE_CONTRACT_SCHEMA_ID = "palette.coordinate_surface_contract"
COORDINATE_SURFACE_CONTRACT_SCHEMA_VERSION = 1

SOURCE_CAMERA_PROFILE_ID = "source_camera_image_px.top_left_y_down.v1"
SOURCE_CAMERA_NORMALIZED_PROFILE_ID = "source_camera_normalized_xy.top_left_y_down.v1"
ROI_LOCAL_PROFILE_ID = "roi_local_px.top_left_y_down.v1"

SOURCE_CAMERA_POINT_PIXEL_CONVENTION = "continuous"
SOURCE_CAMERA_BBOX_PIXEL_CONVENTION = "pixel_edge_half_open"
ROI_POINT_PIXEL_CONVENTION = "continuous"
ROI_BBOX_PIXEL_CONVENTION = "pixel_edge_half_open"

DIRECT_PRESENTATION_MAPPING = "direct_source_camera_continuous_pixels"
NORMALIZED_PRESENTATION_MAPPING = "scale_by_source_camera_extent"
ROI_PRESENTATION_MAPPING = "rowwise_roi_to_source_camera_transform"
NON_POSITIONAL_MAPPING = "not_positional_geometry"

SOURCE_CAMERA_REFERENCE_EXTENT = "source_camera_frame"
ROI_REFERENCE_EXTENT = "row_roi_frame"


@dataclass(frozen=True)
class CoordinateSurfaceContract:
    """Run-independent coordinate semantics for one persisted array surface."""

    surface_id: str
    domain_id: str
    geometry_type: str
    components: tuple[str, ...]
    component_units: tuple[str, ...]
    pixel_convention: str
    reference_extent_role: str
    source_camera_mapping: str
    descriptor_profile_id: str | None
    descriptor_overlay_status: str | None

    def __post_init__(self) -> None:
        if not self.surface_id.strip():
            raise ValueError("surface_id cannot be empty.")
        if not self.domain_id.strip():
            raise ValueError("domain_id cannot be empty.")
        if not self.geometry_type.strip():
            raise ValueError("geometry_type cannot be empty.")
        if not self.components or len(self.components) != len(self.component_units):
            raise ValueError(
                "Every coordinate surface component requires one exact unit."
            )
        if not self.reference_extent_role.strip():
            raise ValueError("reference_extent_role cannot be empty.")
        if self.source_camera_mapping not in {
            DIRECT_PRESENTATION_MAPPING,
            NORMALIZED_PRESENTATION_MAPPING,
            ROI_PRESENTATION_MAPPING,
            NON_POSITIONAL_MAPPING,
        }:
            raise ValueError(
                f"Unsupported source-camera mapping {self.source_camera_mapping!r}."
            )

        if self.descriptor_profile_id is None:
            if self.descriptor_overlay_status is not None:
                raise ValueError(
                    "Non-descriptor surfaces cannot claim an overlay status."
                )
            return

        profile = CANONICAL_COORDINATE_PROFILES.get(self.descriptor_profile_id)
        if profile is None:
            raise ValueError(
                f"Unknown canonical coordinate profile {self.descriptor_profile_id!r}."
            )
        if profile.publication_status != "available":
            raise ValueError(
                f"Coordinate profile {self.descriptor_profile_id!r} is not publishable."
            )
        if profile.space_id != self.domain_id:
            raise ValueError(
                "Coordinate surface domain differs from its canonical profile."
            )
        if self.geometry_type not in profile.geometry_types:
            raise ValueError(
                f"Profile {self.descriptor_profile_id!r} does not support "
                f"{self.geometry_type!r}."
            )
        if self.pixel_convention not in profile.pixel_conventions:
            raise ValueError(
                f"Profile {self.descriptor_profile_id!r} does not support pixel "
                f"convention {self.pixel_convention!r}."
            )
        if self.descriptor_overlay_status not in profile.overlay_statuses:
            raise ValueError(
                f"Profile {self.descriptor_profile_id!r} does not support overlay "
                f"status {self.descriptor_overlay_status!r}."
            )
        for component, unit in zip(
            self.components,
            self.component_units,
            strict=True,
        ):
            if component != "angle" and unit != profile.coordinate_unit:
                raise ValueError(
                    f"Component {component!r} requires {profile.coordinate_unit!r} "
                    f"under profile {self.descriptor_profile_id!r}."
                )

    @property
    def has_canonical_descriptor(self) -> bool:
        return self.descriptor_profile_id is not None

    def descriptor_kwargs(self) -> dict[str, object]:
        """Return the invariant arguments for the canonical descriptor builder."""

        if self.descriptor_profile_id is None or self.descriptor_overlay_status is None:
            raise ValueError(
                f"Surface {self.surface_id!r} is a typed storage measurement, not "
                "a canonical coordinate-descriptor surface."
            )
        return {
            "profile_id": self.descriptor_profile_id,
            "geometry_type": self.geometry_type,
            "components": self.components,
            "component_units": self.component_units,
            "pixel_convention": self.pixel_convention,
            "source_camera_overlay_status": self.descriptor_overlay_status,
        }

    def as_manifest(self) -> dict[str, object]:
        return {
            "schema_id": COORDINATE_SURFACE_CONTRACT_SCHEMA_ID,
            "schema_version": COORDINATE_SURFACE_CONTRACT_SCHEMA_VERSION,
            "surface_id": self.surface_id,
            "domain_id": self.domain_id,
            "geometry_type": self.geometry_type,
            "components": list(self.components),
            "component_units": list(self.component_units),
            "pixel_convention": self.pixel_convention,
            "reference_extent_role": self.reference_extent_role,
            "source_camera_mapping": self.source_camera_mapping,
            "descriptor_profile_id": self.descriptor_profile_id,
            "descriptor_overlay_status": self.descriptor_overlay_status,
        }


SOURCE_CAMERA_NORMALIZED_BBOX_CXCYWH = CoordinateSurfaceContract(
    surface_id="source_camera_normalized_bbox_cxcywh_v1",
    domain_id="source_camera_normalized_xy",
    geometry_type="bbox_cxcywh",
    components=("center_x", "center_y", "width", "height"),
    component_units=("normalized",) * 4,
    pixel_convention="continuous",
    reference_extent_role=SOURCE_CAMERA_REFERENCE_EXTENT,
    source_camera_mapping=NORMALIZED_PRESENTATION_MAPPING,
    descriptor_profile_id=SOURCE_CAMERA_NORMALIZED_PROFILE_ID,
    descriptor_overlay_status=CANONICAL_OVERLAY_REQUIRES_TRANSFORM,
)

SOURCE_CAMERA_NORMALIZED_BBOX_XYXY = CoordinateSurfaceContract(
    surface_id="source_camera_normalized_bbox_xyxy_v1",
    domain_id="source_camera_normalized_xy",
    geometry_type="bbox_xyxy",
    components=("x_min", "y_min", "x_max", "y_max"),
    component_units=("normalized",) * 4,
    pixel_convention="continuous",
    reference_extent_role=SOURCE_CAMERA_REFERENCE_EXTENT,
    source_camera_mapping=NORMALIZED_PRESENTATION_MAPPING,
    descriptor_profile_id=SOURCE_CAMERA_NORMALIZED_PROFILE_ID,
    descriptor_overlay_status=CANONICAL_OVERLAY_REQUIRES_TRANSFORM,
)

SOURCE_CAMERA_BBOX_XYXY = CoordinateSurfaceContract(
    surface_id="source_camera_bbox_xyxy_v1",
    domain_id="source_camera_image_px",
    geometry_type="bbox_xyxy",
    components=("x_min", "y_min", "x_max", "y_max"),
    component_units=("px",) * 4,
    pixel_convention=SOURCE_CAMERA_BBOX_PIXEL_CONVENTION,
    reference_extent_role=SOURCE_CAMERA_REFERENCE_EXTENT,
    source_camera_mapping=DIRECT_PRESENTATION_MAPPING,
    descriptor_profile_id=SOURCE_CAMERA_PROFILE_ID,
    descriptor_overlay_status=CANONICAL_OVERLAY_DIRECT,
)

SOURCE_CAMERA_POINT_XY = CoordinateSurfaceContract(
    surface_id="source_camera_point_xy_v1",
    domain_id="source_camera_image_px",
    geometry_type="point_xy",
    components=("x", "y"),
    component_units=("px", "px"),
    pixel_convention=SOURCE_CAMERA_POINT_PIXEL_CONVENTION,
    reference_extent_role=SOURCE_CAMERA_REFERENCE_EXTENT,
    source_camera_mapping=DIRECT_PRESENTATION_MAPPING,
    descriptor_profile_id=SOURCE_CAMERA_PROFILE_ID,
    descriptor_overlay_status=CANONICAL_OVERLAY_DIRECT,
)

SOURCE_CAMERA_NORMALIZED_POINT_XY = CoordinateSurfaceContract(
    surface_id="source_camera_normalized_point_xy_v1",
    domain_id="source_camera_normalized_xy",
    geometry_type="point_xy",
    components=("x", "y"),
    component_units=("normalized", "normalized"),
    pixel_convention="continuous",
    reference_extent_role=SOURCE_CAMERA_REFERENCE_EXTENT,
    source_camera_mapping=NORMALIZED_PRESENTATION_MAPPING,
    descriptor_profile_id=SOURCE_CAMERA_NORMALIZED_PROFILE_ID,
    descriptor_overlay_status=CANONICAL_OVERLAY_REQUIRES_TRANSFORM,
)

SOURCE_CAMERA_CROP_XYWH = CoordinateSurfaceContract(
    surface_id="source_camera_crop_xywh_v1",
    domain_id="source_camera_image_px",
    geometry_type="bbox_xywh",
    components=("x", "y", "width", "height"),
    component_units=("px",) * 4,
    pixel_convention=SOURCE_CAMERA_BBOX_PIXEL_CONVENTION,
    reference_extent_role=SOURCE_CAMERA_REFERENCE_EXTENT,
    source_camera_mapping=DIRECT_PRESENTATION_MAPPING,
    descriptor_profile_id=SOURCE_CAMERA_PROFILE_ID,
    descriptor_overlay_status=CANONICAL_OVERLAY_DIRECT,
)

SOURCE_CAMERA_EXTRACTION_ORIGIN_XY = CoordinateSurfaceContract(
    surface_id="source_camera_extraction_origin_xy_v1",
    domain_id="source_camera_image_px",
    geometry_type="point_xy",
    components=("x", "y"),
    component_units=("px", "px"),
    pixel_convention=SOURCE_CAMERA_POINT_PIXEL_CONVENTION,
    reference_extent_role=SOURCE_CAMERA_REFERENCE_EXTENT,
    source_camera_mapping=DIRECT_PRESENTATION_MAPPING,
    descriptor_profile_id=SOURCE_CAMERA_PROFILE_ID,
    descriptor_overlay_status=CANONICAL_OVERLAY_DIRECT,
)

SOURCE_CAMERA_EXTRACTION_EXTENT_WH = CoordinateSurfaceContract(
    surface_id="source_camera_extraction_extent_wh_v1",
    domain_id="source_camera_image_px",
    geometry_type="extent_wh",
    components=("width", "height"),
    component_units=("px", "px"),
    pixel_convention="not_applicable",
    reference_extent_role=SOURCE_CAMERA_REFERENCE_EXTENT,
    source_camera_mapping=NON_POSITIONAL_MAPPING,
    descriptor_profile_id=None,
    descriptor_overlay_status=None,
)

ROI_BBOX_XYXY = CoordinateSurfaceContract(
    surface_id="roi_bbox_xyxy_v1",
    domain_id="roi_local_px",
    geometry_type="bbox_xyxy",
    components=("x_min", "y_min", "x_max", "y_max"),
    component_units=("px",) * 4,
    pixel_convention=ROI_BBOX_PIXEL_CONVENTION,
    reference_extent_role=ROI_REFERENCE_EXTENT,
    source_camera_mapping=ROI_PRESENTATION_MAPPING,
    descriptor_profile_id=ROI_LOCAL_PROFILE_ID,
    descriptor_overlay_status=CANONICAL_OVERLAY_REQUIRES_TRANSFORM,
)

ROI_POINT_XY = CoordinateSurfaceContract(
    surface_id="roi_point_xy_v1",
    domain_id="roi_local_px",
    geometry_type="point_xy",
    components=("x", "y"),
    component_units=("px", "px"),
    pixel_convention=ROI_POINT_PIXEL_CONVENTION,
    reference_extent_role=ROI_REFERENCE_EXTENT,
    source_camera_mapping=ROI_PRESENTATION_MAPPING,
    descriptor_profile_id=ROI_LOCAL_PROFILE_ID,
    descriptor_overlay_status=CANONICAL_OVERLAY_REQUIRES_TRANSFORM,
)

ROI_POINTS_XY = CoordinateSurfaceContract(
    surface_id="roi_points_xy_v1",
    domain_id="roi_local_px",
    geometry_type="points_xy",
    components=("x", "y"),
    component_units=("px", "px"),
    pixel_convention=ROI_POINT_PIXEL_CONVENTION,
    reference_extent_role=ROI_REFERENCE_EXTENT,
    source_camera_mapping=ROI_PRESENTATION_MAPPING,
    descriptor_profile_id=ROI_LOCAL_PROFILE_ID,
    descriptor_overlay_status=CANONICAL_OVERLAY_REQUIRES_TRANSFORM,
)

ROI_RASTER_YX = CoordinateSurfaceContract(
    surface_id="roi_raster_yx_v1",
    domain_id="roi_local_px",
    geometry_type="raster_yx",
    components=("y", "x"),
    component_units=("px", "px"),
    pixel_convention="pixel_center",
    reference_extent_role=ROI_REFERENCE_EXTENT,
    source_camera_mapping=ROI_PRESENTATION_MAPPING,
    descriptor_profile_id=ROI_LOCAL_PROFILE_ID,
    descriptor_overlay_status=CANONICAL_OVERLAY_REQUIRES_TRANSFORM,
)


COORDINATE_SURFACE_CONTRACTS: Mapping[str, CoordinateSurfaceContract] = {
    surface.surface_id: surface
    for surface in (
        SOURCE_CAMERA_NORMALIZED_BBOX_CXCYWH,
        SOURCE_CAMERA_NORMALIZED_BBOX_XYXY,
        SOURCE_CAMERA_BBOX_XYXY,
        SOURCE_CAMERA_POINT_XY,
        SOURCE_CAMERA_NORMALIZED_POINT_XY,
        SOURCE_CAMERA_CROP_XYWH,
        SOURCE_CAMERA_EXTRACTION_ORIGIN_XY,
        SOURCE_CAMERA_EXTRACTION_EXTENT_WH,
        ROI_BBOX_XYXY,
        ROI_POINT_XY,
        ROI_POINTS_XY,
        ROI_RASTER_YX,
    )
}


def coordinate_surface_contract(surface_id: str) -> CoordinateSurfaceContract:
    try:
        return COORDINATE_SURFACE_CONTRACTS[str(surface_id)]
    except KeyError as exc:
        raise KeyError(f"Unknown coordinate surface contract {surface_id!r}.") from exc


__all__ = [
    "COORDINATE_SURFACE_CONTRACT_SCHEMA_ID",
    "COORDINATE_SURFACE_CONTRACT_SCHEMA_VERSION",
    "COORDINATE_SURFACE_CONTRACTS",
    "CoordinateSurfaceContract",
    "DIRECT_PRESENTATION_MAPPING",
    "NON_POSITIONAL_MAPPING",
    "NORMALIZED_PRESENTATION_MAPPING",
    "ROI_BBOX_PIXEL_CONVENTION",
    "ROI_BBOX_XYXY",
    "ROI_LOCAL_PROFILE_ID",
    "ROI_POINT_PIXEL_CONVENTION",
    "ROI_POINT_XY",
    "ROI_POINTS_XY",
    "ROI_PRESENTATION_MAPPING",
    "ROI_RASTER_YX",
    "SOURCE_CAMERA_BBOX_PIXEL_CONVENTION",
    "SOURCE_CAMERA_BBOX_XYXY",
    "SOURCE_CAMERA_CROP_XYWH",
    "SOURCE_CAMERA_EXTRACTION_EXTENT_WH",
    "SOURCE_CAMERA_EXTRACTION_ORIGIN_XY",
    "SOURCE_CAMERA_NORMALIZED_BBOX_CXCYWH",
    "SOURCE_CAMERA_NORMALIZED_BBOX_XYXY",
    "SOURCE_CAMERA_NORMALIZED_POINT_XY",
    "SOURCE_CAMERA_NORMALIZED_PROFILE_ID",
    "SOURCE_CAMERA_POINT_PIXEL_CONVENTION",
    "SOURCE_CAMERA_POINT_XY",
    "SOURCE_CAMERA_PROFILE_ID",
    "coordinate_surface_contract",
]
