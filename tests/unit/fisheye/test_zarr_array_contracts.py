from __future__ import annotations

import json

import numpy as np
import pytest

from fisheye.shared.zarr.array_contracts import (
    CORE_ARRAY_CONTRACTS,
    DENSE_SUBJECT_MASKS_ROI_V1,
    FRAME_COUNTS_V1,
    KEYPOINTS_IMG_V1,
    REFINED_DETECTION_REFINED_ROW_IDS_V1,
    REFINED_SOURCE_FRAME_ROW_OFFSETS_V1,
    UTF8,
    ArrayContract,
    ArrayContractBinding,
    ArrayContractCatalog,
)
from fisheye.shared.zarr.storage_intent import AccessPattern, WriteMode
from fisheye.shared.zarr.storage_planner import plan_storage
from fisheye.shared.zarr.storage_profiles import PUBLISHED_HTTP_V1


def test_core_contract_catalog_is_versioned_and_json_safe() -> None:
    resolved = CORE_ARRAY_CONTRACTS.resolve("palette.array.keypoints_img", 1)

    assert resolved is KEYPOINTS_IMG_V1
    manifest = CORE_ARRAY_CONTRACTS.as_manifest()
    assert json.loads(json.dumps(manifest)) == manifest
    assert manifest["schema_id"] == "palette.array_contract_catalog"
    assert len(manifest["contracts"]) == 53
    assert CORE_ARRAY_CONTRACTS.resolve(
        "palette.array.crop.roi_sizes_full",
        1,
    ).dtype.dtype_id == "int32"
    assert CORE_ARRAY_CONTRACTS.resolve(
        "palette.array.refined_detection.refined_row_ids",
        1,
    ) is REFINED_DETECTION_REFINED_ROW_IDS_V1
    assert CORE_ARRAY_CONTRACTS.resolve(
        "palette.array.refined_detection.source.frame_row_offsets",
        1,
    ) is REFINED_SOURCE_FRAME_ROW_OFFSETS_V1


def test_keypoint_contract_requires_exact_dtype_and_fixed_xy_axis() -> None:
    class Observed:
        shape = (1_000, 5, 2)
        dtype = np.dtype(np.float64)

    assert KEYPOINTS_IMG_V1.validate_observation(Observed()) == ()

    Observed.dtype = np.dtype(np.float32)
    errors = KEYPOINTS_IMG_V1.validate_observation(Observed())
    assert errors == ("dtype mismatch: expected float64, got float32",)

    Observed.dtype = np.dtype(np.float64)
    Observed.shape = (1_000, 5, 3)
    errors = KEYPOINTS_IMG_V1.validate_observation(Observed())
    assert "axis 2 (xy) expected 2, got 3" in errors


def test_symbolic_dimension_bindings_validate_cross_array_sizes() -> None:
    assert FRAME_COUNTS_V1.validate_shape(
        (100_000,), dimensions={"n_frames": 100_000}
    ) == ()
    assert FRAME_COUNTS_V1.validate_shape(
        (99_999,), dimensions={"n_frames": 100_000}
    ) == (
        "axis 0 (camera_frame) expected symbolic dimension n_frames=100000, "
        "got 99999",
    )


def test_contract_produces_exact_dtype_storage_intent() -> None:
    intent = KEYPOINTS_IMG_V1.storage_intent(
        name="keypoints_img",
        shape=(1_000_000, 5, 2),
        dimensions={"n_rois": 1_000_000, "n_keypoints": 5},
        access=AccessPattern.WINDOWED,
        write_mode=WriteMode.IMMUTABLE,
    )
    plan = plan_storage(intent, PUBLISHED_HTTP_V1)

    assert intent.dtype == np.dtype(np.float64)
    assert plan.logical_dtype == "float64"
    assert plan.logical_schema_id == "palette.array.keypoints_img"
    assert plan.logical_schema_version == 1
    assert plan.chunk_shape == (16_384, 5, 2)


def test_dense_mask_contract_can_declare_component_access_unit() -> None:
    intent = DENSE_SUBJECT_MASKS_ROI_V1.storage_intent(
        name="masks_roi",
        shape=(1_000_000, 4, 512, 512),
        dimensions={
            "n_rois": 1_000_000,
            "n_channels": 4,
            "H": 512,
            "W": 512,
        },
        access=AccessPattern.PER_ROW,
        write_mode=WriteMode.IMMUTABLE,
        access_unit_shape=(1, 1, 512, 512),
    )
    plan = plan_storage(intent, PUBLISHED_HTTP_V1)

    assert plan.logical_dtype == "uint8"
    assert plan.chunk_shape == (4, 1, 512, 512)
    assert plan.shard_shape == (1_024, 1, 512, 512)


def test_binding_maps_archive_path_to_contract_identity() -> None:
    binding = ArrayContractBinding(
        path="keypoints_runs/run_1/keypoints_img",
        contract_id=KEYPOINTS_IMG_V1.schema_id,
        contract_version=KEYPOINTS_IMG_V1.schema_version,
        required=True,
    )

    assert binding.as_manifest() == {
        "path": "keypoints_runs/run_1/keypoints_img",
        "contract_id": "palette.array.keypoints_img",
        "contract_version": 1,
        "required": True,
    }


def test_catalog_rejects_duplicate_contract_versions() -> None:
    with pytest.raises(ValueError, match="Duplicate array contract"):
        ArrayContractCatalog((FRAME_COUNTS_V1, FRAME_COUNTS_V1))


def test_variable_utf8_contract_is_exact_but_requires_size_for_planning() -> None:
    class VariableLengthUTF8:
        pass

    assert UTF8.matches(VariableLengthUTF8())

    contract = ArrayContract(
        schema_id="palette.array.review_note",
        schema_version=1,
        dtype=UTF8,
        shape_template=("n_rows",),
        axis_names=("row",),
        description="Review note text.",
    )
    with pytest.raises(ValueError, match="logical_itemsize_bytes"):
        contract.storage_intent(
            shape=(1_000,),
            access=AccessPattern.WINDOWED,
            write_mode=WriteMode.IMMUTABLE,
        )

    intent = contract.storage_intent(
        shape=(1_000,),
        access=AccessPattern.WINDOWED,
        write_mode=WriteMode.IMMUTABLE,
        logical_itemsize_bytes=64,
    )
    assert intent.logical_nbytes == 64_000
