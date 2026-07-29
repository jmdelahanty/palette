from __future__ import annotations

from dataclasses import replace

import pytest

from fisheye.shared.coordinate_surface_contract import (
    DIRECT_PRESENTATION_MAPPING,
    NORMALIZED_PRESENTATION_MAPPING,
    ROI_BBOX_XYXY,
    ROI_PRESENTATION_MAPPING,
    SOURCE_CAMERA_BBOX_XYXY,
    SOURCE_CAMERA_EXTRACTION_EXTENT_WH,
    SOURCE_CAMERA_NORMALIZED_BBOX_CXCYWH,
    SOURCE_CAMERA_POINT_XY,
)
from fisheye.shared.zarr.array_contracts import (
    BODY_FRAME_ARRAY_CONTRACTS,
    CORE_ARRAY_CONTRACTS,
    CROP_ARRAY_CONTRACTS,
    DETECTION_ARRAY_CONTRACTS,
    DETECTION_BBOX_NORM_COORDS_V1,
    KEYPOINTS_IMG_V1,
    KEYPOINT_ARRAY_CONTRACTS,
    REFINED_KEYPOINT_ARRAY_CONTRACTS,
    REFINED_DETECTION_ARRAY_CONTRACTS,
)
from fisheye.shared.zarr.coordinate_contracts import (
    ARRAY_COORDINATE_BINDINGS,
    AUTHORITY,
    array_coordinate_binding,
    array_coordinate_catalog_manifest,
    validate_array_coordinate_bindings,
)
from fisheye.shared.zarr.crop_schema import CROP_GEOMETRY_SCHEMA_V1
from fisheye.shared.zarr.detection_schema import CANONICAL_DETECTION_SCHEMA_V1
from fisheye.shared.zarr.refined_detection_schema import (
    REFINED_DETECTION_SCHEMA_V1,
)


def test_every_coordinate_bearing_core_array_has_one_exact_binding() -> None:
    assert validate_array_coordinate_bindings(CORE_ARRAY_CONTRACTS) == ()
    expected = {
        contract.key
        for contract in CORE_ARRAY_CONTRACTS.contracts
        if contract.coordinate_space is not None
    }
    assert set(ARRAY_COORDINATE_BINDINGS) == expected


@pytest.mark.parametrize(
    "catalog",
    (
        DETECTION_ARRAY_CONTRACTS,
        REFINED_DETECTION_ARRAY_CONTRACTS,
        CROP_ARRAY_CONTRACTS,
        KEYPOINT_ARRAY_CONTRACTS,
        REFINED_KEYPOINT_ARRAY_CONTRACTS,
        BODY_FRAME_ARRAY_CONTRACTS,
    ),
)
def test_stage_coordinate_catalogs_are_complete(catalog: object) -> None:
    assert validate_array_coordinate_bindings(catalog) == ()
    manifest = array_coordinate_catalog_manifest(catalog)
    assert manifest["schema_id"] == "palette.array_coordinate_catalog"
    assert manifest["schema_version"] == 1
    assert manifest["bindings"]
    assert manifest["surfaces"]


def test_stage_accessors_do_not_change_existing_schema_manifests() -> None:
    assert (
        CANONICAL_DETECTION_SCHEMA_V1.coordinate_contract_manifest()
        == array_coordinate_catalog_manifest(DETECTION_ARRAY_CONTRACTS)
    )
    assert (
        REFINED_DETECTION_SCHEMA_V1.coordinate_contract_manifest()
        == array_coordinate_catalog_manifest(REFINED_DETECTION_ARRAY_CONTRACTS)
    )
    assert (
        CROP_GEOMETRY_SCHEMA_V1.coordinate_contract_manifest()
        == array_coordinate_catalog_manifest(CROP_ARRAY_CONTRACTS)
    )


def test_detection_authority_and_pixel_projection_are_distinct() -> None:
    normalized = array_coordinate_binding(DETECTION_BBOX_NORM_COORDS_V1)
    assert normalized.semantic_role == AUTHORITY
    assert normalized.surface is SOURCE_CAMERA_NORMALIZED_BBOX_CXCYWH
    assert normalized.surface.source_camera_mapping == NORMALIZED_PRESENTATION_MAPPING
    assert normalized.surface.geometry_type == "bbox_cxcywh"
    assert normalized.surface.component_units == ("normalized",) * 4

    pixel = SOURCE_CAMERA_BBOX_XYXY
    assert pixel.source_camera_mapping == DIRECT_PRESENTATION_MAPPING
    assert pixel.geometry_type == "bbox_xyxy"
    assert pixel.pixel_convention == "pixel_edge_half_open"


def test_source_camera_pixels_mean_continuous_geometry_not_integer_indices() -> None:
    binding = array_coordinate_binding(KEYPOINTS_IMG_V1)
    assert binding.surface is SOURCE_CAMERA_POINT_XY
    assert binding.surface.pixel_convention == "continuous"
    assert binding.surface.domain_id == "source_camera_image_px"
    assert binding.surface.source_camera_mapping == DIRECT_PRESENTATION_MAPPING


def test_roi_geometry_requires_rowwise_transform_to_source_camera() -> None:
    assert ROI_BBOX_XYXY.domain_id == "roi_local_px"
    assert ROI_BBOX_XYXY.source_camera_mapping == ROI_PRESENTATION_MAPPING
    assert ROI_BBOX_XYXY.pixel_convention == "pixel_edge_half_open"


def test_integer_crop_extent_is_not_falsely_published_as_a_point_descriptor() -> None:
    extent = SOURCE_CAMERA_EXTRACTION_EXTENT_WH
    assert extent.geometry_type == "extent_wh"
    assert extent.source_camera_mapping == "not_positional_geometry"
    assert extent.has_canonical_descriptor is False
    with pytest.raises(ValueError, match="typed storage measurement"):
        extent.descriptor_kwargs()


def test_surface_contract_rejects_profile_domain_drift() -> None:
    with pytest.raises(ValueError, match="domain differs"):
        replace(SOURCE_CAMERA_POINT_XY, domain_id="roi_local_px")


def test_catalog_manifest_is_deterministic_and_deduplicates_surfaces() -> None:
    first = array_coordinate_catalog_manifest(CORE_ARRAY_CONTRACTS)
    second = array_coordinate_catalog_manifest(CORE_ARRAY_CONTRACTS)
    assert first == second
    surface_ids = [item["surface_id"] for item in first["surfaces"]]
    assert surface_ids == sorted(set(surface_ids))
    binding_keys = [
        (item["array_contract_id"], item["array_contract_version"])
        for item in first["bindings"]
    ]
    assert binding_keys == sorted(binding_keys)
