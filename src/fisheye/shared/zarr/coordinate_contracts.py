"""Bind logical array contracts to shared coordinate-surface semantics.

``ArrayContract.coordinate_space`` is retained for compatibility with existing
manifests.  It is intentionally not the future authority: labels such as
``image_pixel`` and ``roi_pixel`` omit geometry type and pixel convention.
This catalog makes those details exact while preserving every existing array
contract ID, version, dtype, and shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from fisheye.shared.coordinate_surface_contract import (
    COORDINATE_SURFACE_CONTRACTS,
    CoordinateSurfaceContract,
    ROI_BBOX_XYXY,
    ROI_POINT_XY,
    ROI_POINTS_XY,
    ROI_RASTER_YX,
    SOURCE_CAMERA_BBOX_XYXY,
    SOURCE_CAMERA_CROP_XYWH,
    SOURCE_CAMERA_EXTRACTION_EXTENT_WH,
    SOURCE_CAMERA_EXTRACTION_ORIGIN_XY,
    SOURCE_CAMERA_NORMALIZED_BBOX_CXCYWH,
    SOURCE_CAMERA_NORMALIZED_POINT_XY,
    SOURCE_CAMERA_POINT_XY,
)
from fisheye.shared.zarr.array_contracts import (
    BODY_FRAME_ORIGIN_XY_V1,
    CONTOUR_POINTS_XY_V1,
    CROP_BBOX_ROI_XYXY_V1,
    CROP_ROI_COORDINATES_FULL_V1,
    CROP_ROI_SIZES_FULL_V1,
    CROP_SOURCE_CROP_XYWH_V1,
    DENSE_SUBJECT_MASKS_ROI_V1,
    DETECTION_BBOX_IMG_XYXY_V1,
    DETECTION_BBOX_NORM_COORDS_V1,
    DETECTION_CENTERS_IMG_XY_V1,
    KEYPOINTS_IMG_V1,
    KEYPOINTS_IMG_V2,
    KEYPOINTS_NORM_V1,
    KEYPOINTS_ROI_V1,
    KEYPOINTS_ROI_V2,
    KEYPOINT_POSE_BBOX_XYXY_IMG_V1,
    KEYPOINT_POSE_BBOX_XYXY_ROI_V1,
    REFINED_SOURCE_BBOX_IMG_XYXY_V1,
    REFINED_SOURCE_BBOX_NORM_COORDS_V1,
    REFINED_SOURCE_CENTERS_IMG_XY_V1,
    ArrayContract,
    ArrayContractCatalog,
)


ARRAY_COORDINATE_BINDING_SCHEMA_ID = "palette.array_coordinate_binding"
ARRAY_COORDINATE_BINDING_SCHEMA_VERSION = 1
ARRAY_COORDINATE_CATALOG_SCHEMA_ID = "palette.array_coordinate_catalog"
ARRAY_COORDINATE_CATALOG_SCHEMA_VERSION = 1

AUTHORITY = "authoritative_numeric_surface"
DERIVED = "exact_derived_numeric_surface"
PLACEMENT = "crop_placement_surface"
SAMPLED = "sampled_spatial_surface"


@dataclass(frozen=True)
class ArrayCoordinateBinding:
    """One exact array-contract-to-coordinate-surface assignment."""

    array_contract_id: str
    array_contract_version: int
    surface_id: str
    semantic_role: str
    legacy_coordinate_space: str

    def __post_init__(self) -> None:
        if not self.array_contract_id.strip():
            raise ValueError("array_contract_id cannot be empty.")
        if (
            type(self.array_contract_version) is not int
            or self.array_contract_version <= 0
        ):
            raise ValueError("array_contract_version must be a positive exact integer.")
        if self.surface_id not in COORDINATE_SURFACE_CONTRACTS:
            raise ValueError(f"Unknown coordinate surface {self.surface_id!r}.")
        if self.semantic_role not in {AUTHORITY, DERIVED, PLACEMENT, SAMPLED}:
            raise ValueError(
                f"Unsupported coordinate semantic role {self.semantic_role!r}."
            )
        if not self.legacy_coordinate_space.strip():
            raise ValueError("legacy_coordinate_space cannot be empty.")

    @property
    def key(self) -> tuple[str, int]:
        return self.array_contract_id, self.array_contract_version

    @property
    def surface(self) -> CoordinateSurfaceContract:
        return COORDINATE_SURFACE_CONTRACTS[self.surface_id]

    def as_manifest(self) -> dict[str, object]:
        return {
            "schema_id": ARRAY_COORDINATE_BINDING_SCHEMA_ID,
            "schema_version": ARRAY_COORDINATE_BINDING_SCHEMA_VERSION,
            "array_contract_id": self.array_contract_id,
            "array_contract_version": self.array_contract_version,
            "surface_id": self.surface_id,
            "semantic_role": self.semantic_role,
            "legacy_coordinate_space": self.legacy_coordinate_space,
        }


def _binding(
    contract: ArrayContract,
    surface: CoordinateSurfaceContract,
    *,
    semantic_role: str,
) -> ArrayCoordinateBinding:
    if contract.coordinate_space is None:
        raise ValueError(
            f"Coordinate binding target {contract.schema_id!r} lacks a legacy "
            "coordinate_space annotation."
        )
    return ArrayCoordinateBinding(
        array_contract_id=contract.schema_id,
        array_contract_version=contract.schema_version,
        surface_id=surface.surface_id,
        semantic_role=semantic_role,
        legacy_coordinate_space=contract.coordinate_space,
    )


_BINDINGS = (
    _binding(
        DETECTION_BBOX_NORM_COORDS_V1,
        SOURCE_CAMERA_NORMALIZED_BBOX_CXCYWH,
        semantic_role=AUTHORITY,
    ),
    _binding(
        DETECTION_BBOX_IMG_XYXY_V1,
        SOURCE_CAMERA_BBOX_XYXY,
        semantic_role=DERIVED,
    ),
    _binding(
        DETECTION_CENTERS_IMG_XY_V1,
        SOURCE_CAMERA_POINT_XY,
        semantic_role=DERIVED,
    ),
    _binding(
        REFINED_SOURCE_BBOX_NORM_COORDS_V1,
        SOURCE_CAMERA_NORMALIZED_BBOX_CXCYWH,
        semantic_role=AUTHORITY,
    ),
    _binding(
        REFINED_SOURCE_BBOX_IMG_XYXY_V1,
        SOURCE_CAMERA_BBOX_XYXY,
        semantic_role=DERIVED,
    ),
    _binding(
        REFINED_SOURCE_CENTERS_IMG_XY_V1,
        SOURCE_CAMERA_POINT_XY,
        semantic_role=DERIVED,
    ),
    _binding(
        CROP_ROI_COORDINATES_FULL_V1,
        SOURCE_CAMERA_EXTRACTION_ORIGIN_XY,
        semantic_role=PLACEMENT,
    ),
    _binding(
        CROP_ROI_SIZES_FULL_V1,
        SOURCE_CAMERA_EXTRACTION_EXTENT_WH,
        semantic_role=PLACEMENT,
    ),
    _binding(
        CROP_SOURCE_CROP_XYWH_V1,
        SOURCE_CAMERA_CROP_XYWH,
        semantic_role=PLACEMENT,
    ),
    _binding(
        CROP_BBOX_ROI_XYXY_V1,
        ROI_BBOX_XYXY,
        semantic_role=DERIVED,
    ),
    _binding(KEYPOINTS_ROI_V1, ROI_POINT_XY, semantic_role=AUTHORITY),
    _binding(KEYPOINTS_IMG_V1, SOURCE_CAMERA_POINT_XY, semantic_role=DERIVED),
    _binding(
        KEYPOINTS_NORM_V1,
        SOURCE_CAMERA_NORMALIZED_POINT_XY,
        semantic_role=DERIVED,
    ),
    _binding(KEYPOINTS_ROI_V2, ROI_POINT_XY, semantic_role=AUTHORITY),
    _binding(KEYPOINTS_IMG_V2, SOURCE_CAMERA_POINT_XY, semantic_role=DERIVED),
    _binding(
        KEYPOINT_POSE_BBOX_XYXY_ROI_V1,
        ROI_BBOX_XYXY,
        semantic_role=AUTHORITY,
    ),
    _binding(
        KEYPOINT_POSE_BBOX_XYXY_IMG_V1,
        SOURCE_CAMERA_BBOX_XYXY,
        semantic_role=DERIVED,
    ),
    _binding(BODY_FRAME_ORIGIN_XY_V1, SOURCE_CAMERA_POINT_XY, semantic_role=DERIVED),
    _binding(DENSE_SUBJECT_MASKS_ROI_V1, ROI_RASTER_YX, semantic_role=AUTHORITY),
    _binding(CONTOUR_POINTS_XY_V1, ROI_POINTS_XY, semantic_role=SAMPLED),
)

ARRAY_COORDINATE_BINDINGS: Mapping[tuple[str, int], ArrayCoordinateBinding] = {
    binding.key: binding for binding in _BINDINGS
}
if len(ARRAY_COORDINATE_BINDINGS) != len(_BINDINGS):  # pragma: no cover
    raise RuntimeError("Array coordinate bindings contain duplicate contract keys.")


def array_coordinate_binding(
    contract: ArrayContract | tuple[str, int],
) -> ArrayCoordinateBinding:
    key = contract.key if isinstance(contract, ArrayContract) else contract
    try:
        return ARRAY_COORDINATE_BINDINGS[(str(key[0]), int(key[1]))]
    except KeyError as exc:
        raise KeyError(f"No coordinate binding for array contract {key!r}.") from exc


def validate_array_coordinate_bindings(
    contracts: ArrayContractCatalog | Iterable[ArrayContract],
) -> tuple[str, ...]:
    """Return coverage and compatibility errors for one logical catalog."""

    values = (
        contracts.contracts
        if isinstance(contracts, ArrayContractCatalog)
        else tuple(contracts)
    )
    by_key = {contract.key: contract for contract in values}
    errors: list[str] = []
    for contract in values:
        binding = ARRAY_COORDINATE_BINDINGS.get(contract.key)
        if contract.coordinate_space is None:
            if binding is not None:
                errors.append(
                    f"{contract.schema_id}@{contract.schema_version} has a coordinate "
                    "binding but no legacy coordinate_space annotation"
                )
            continue
        if binding is None:
            errors.append(
                f"{contract.schema_id}@{contract.schema_version} has coordinate_space "
                f"{contract.coordinate_space!r} but no exact coordinate binding"
            )
            continue
        if binding.legacy_coordinate_space != contract.coordinate_space:
            errors.append(
                f"{contract.schema_id}@{contract.schema_version} legacy coordinate "
                f"space differs: {binding.legacy_coordinate_space!r} != "
                f"{contract.coordinate_space!r}"
            )

    for key, binding in ARRAY_COORDINATE_BINDINGS.items():
        contract = by_key.get(key)
        if contract is None:
            continue
        if contract.coordinate_space != binding.legacy_coordinate_space:
            errors.append(
                f"{binding.array_contract_id}@{binding.array_contract_version} binding "
                "does not match its logical array contract"
            )
    return tuple(errors)


def require_array_coordinate_bindings(
    contracts: ArrayContractCatalog | Iterable[ArrayContract],
) -> None:
    errors = validate_array_coordinate_bindings(contracts)
    if errors:
        raise ValueError("Invalid array coordinate catalog: " + "; ".join(errors))


def array_coordinate_catalog_manifest(
    contracts: ArrayContractCatalog | Iterable[ArrayContract],
) -> dict[str, object]:
    """Return the exact relevant bindings and deduplicated surface templates."""

    values = (
        contracts.contracts
        if isinstance(contracts, ArrayContractCatalog)
        else tuple(contracts)
    )
    require_array_coordinate_bindings(values)
    bindings = [
        ARRAY_COORDINATE_BINDINGS[contract.key]
        for contract in values
        if contract.coordinate_space is not None
    ]
    bindings.sort(key=lambda item: item.key)
    surface_ids = sorted({binding.surface_id for binding in bindings})
    return {
        "schema_id": ARRAY_COORDINATE_CATALOG_SCHEMA_ID,
        "schema_version": ARRAY_COORDINATE_CATALOG_SCHEMA_VERSION,
        "bindings": [binding.as_manifest() for binding in bindings],
        "surfaces": [
            COORDINATE_SURFACE_CONTRACTS[surface_id].as_manifest()
            for surface_id in surface_ids
        ],
    }


__all__ = [
    "ARRAY_COORDINATE_BINDING_SCHEMA_ID",
    "ARRAY_COORDINATE_BINDING_SCHEMA_VERSION",
    "ARRAY_COORDINATE_BINDINGS",
    "ARRAY_COORDINATE_CATALOG_SCHEMA_ID",
    "ARRAY_COORDINATE_CATALOG_SCHEMA_VERSION",
    "AUTHORITY",
    "ArrayCoordinateBinding",
    "DERIVED",
    "PLACEMENT",
    "SAMPLED",
    "array_coordinate_binding",
    "array_coordinate_catalog_manifest",
    "require_array_coordinate_bindings",
    "validate_array_coordinate_bindings",
]
