from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.cluster.flat_roi_cache import plan_flat_roi_cache_binding
from fisheye.cluster.keypoints.common import (
    FlatRoiCacheBinding,
    validate_keypoint_input_dag,
)
from fisheye.shared.zarr.crop_consumer import (
    CROP_RUN_REFERENCE_LEGACY_PROFILE,
    CROP_RUN_REFERENCE_SIGNED_PROFILE,
    CROP_RUN_REFERENCE_STRICT_PROFILE,
    CROP_RUN_REFERENCE_UNVERSIONED_LEGACY_PROFILE,
    authoritative_crop_roi_pixel_contract,
    build_crop_run_reference,
    strict_crop_fixed_roi_shape,
    strict_crop_required_roi_pixel_contract,
    strict_crop_row_source_signature_spec,
    strict_crop_source_frame_shape,
    validate_crop_run_reference,
)
from fisheye.shared.roi_pixel_contract import (
    orange_mono_pynvvc_luma_hybrid_pixel_contract,
    orange_mono_pynvvc_luma_pixel_contract,
)
from fisheye.shared.zarr.crop_schema import (
    CropGeometryPolicy,
    CropPaddingMode,
    CropSizeMode,
)
from fisheye.shared.zarr.crop_shadow import publish_refined_crop_geometry_shadow
from tests.unit.fisheye.test_crop_shadow import _pixel, _policy, _refined_source


class _LegacyCrop:
    attrs = {"crop_signature": {"revision": 4}, "crop_revision": 4}


class _UnversionedLegacyCrop:
    attrs = {}


class _AcquisitionCropVideo:
    attrs = {
        "crop_signature": {"source": "acquisition_crop_video"},
        "crop_revision": 1,
        "source_pixels": "acquisition_crop_video",
        "roi_pixel_provider": "acquisition_crop_video",
        "roi_pixel_contract": orange_mono_pynvvc_luma_pixel_contract(),
    }


class _HybridAcquisitionCropVideo:
    attrs = {
        "crop_signature": {"source": "hybrid_acquisition_crop_video"},
        "crop_revision": 1,
        "source_pixels": "hybrid_acquisition_crop_video_offline_supplement",
        "roi_pixel_provider": ("hybrid_acquisition_crop_video_offline_supplement"),
        "roi_pixel_contract": orange_mono_pynvvc_luma_hybrid_pixel_contract(),
    }


def _strict_crop(tmp_path: Path, *, mode: str = "r"):
    output = tmp_path / "strict_crop.zarr"
    publish_refined_crop_geometry_shadow(
        _refined_source(tmp_path),
        policy=_policy(),
        pixel_authority=_pixel(),
        destination=output,
        run_id="strict_crop",
        shadow_root=tmp_path,
        coordinate_catalog=True,
    )
    return zarr.open_group(
        str(output / "crop_runs" / "strict_crop"),
        mode=mode,
        use_consolidated=False,
    )


def test_strict_reference_uses_manifest_and_logical_digests(tmp_path: Path) -> None:
    crop = _strict_crop(tmp_path)

    reference = build_crop_run_reference(crop, run_id="strict_crop")

    assert reference["profile"] == CROP_RUN_REFERENCE_STRICT_PROFILE
    assert reference["run_manifest_digest"] == crop.attrs["run_manifest"]["payload_digest"]
    assert reference["logical_content_digest"] == crop.attrs["run_manifest"]["payload"]["logical_content"]["digest"]
    assert validate_crop_run_reference(reference) == reference
    assert strict_crop_fixed_roi_shape(crop, run_id="strict_crop") == (8, 8)
    assert strict_crop_source_frame_shape(crop, run_id="strict_crop") == (80, 100)
    row_signature = strict_crop_row_source_signature_spec(
        crop,
        run_id="strict_crop",
    )
    assert row_signature is not None
    assert row_signature.stage == "crop_geometry"
    pixel_contract = strict_crop_required_roi_pixel_contract(
        crop,
        run_id="strict_crop",
    )
    assert pixel_contract is not None
    assert pixel_contract["name"] == "orange_mono_pynvvc_luma_uint8_v1"
    assert pixel_contract["source_pixels"] == "raw_camera_video"


def test_legacy_reference_is_explicit_compatibility_mode() -> None:
    reference = build_crop_run_reference(_LegacyCrop(), run_id="legacy_crop")

    assert reference == {
        "schema_id": "palette.crop_geometry.run_reference",
        "schema_version": 1,
        "profile": CROP_RUN_REFERENCE_LEGACY_PROFILE,
        "run_id": "legacy_crop",
        "crop_signature": {"revision": 4},
        "crop_revision": 4,
    }
    assert validate_crop_run_reference(reference) == reference


def test_unversioned_legacy_reference_requires_explicit_compatibility_opt_in() -> None:
    with pytest.raises(ValueError, match="immutable run_manifest"):
        build_crop_run_reference(_UnversionedLegacyCrop(), run_id="old_crop")

    reference = build_crop_run_reference(
        _UnversionedLegacyCrop(),
        run_id="old_crop",
        allow_unversioned_legacy=True,
    )
    assert reference["profile"] == CROP_RUN_REFERENCE_UNVERSIONED_LEGACY_PROFILE
    assert validate_crop_run_reference(reference) == reference


def test_acquisition_crop_video_is_a_current_authoritative_pixel_source() -> None:
    contract = authoritative_crop_roi_pixel_contract(
        _AcquisitionCropVideo(),
        run_id="acquisition_crop",
    )

    assert contract is not None
    assert contract["name"] == "orange_mono_pynvvc_luma_uint8_v1"
    assert contract["source_pixels"] == "acquisition_crop_video"
    reference = build_crop_run_reference(
        _AcquisitionCropVideo(),
        run_id="acquisition_crop",
    )
    assert reference["profile"] == CROP_RUN_REFERENCE_SIGNED_PROFILE
    assert validate_crop_run_reference(reference) == reference


def test_acquisition_crop_video_rejects_an_ambiguous_decode_contract() -> None:
    crop = _AcquisitionCropVideo()
    crop.attrs = dict(crop.attrs)
    crop.attrs["roi_pixel_contract"] = {"name": "decoder_default"}

    with pytest.raises(ValueError, match="Acquisition crop-video run"):
        authoritative_crop_roi_pixel_contract(
            crop,
            run_id="acquisition_crop",
        )


def test_hybrid_crop_video_is_an_explicit_current_mixed_source() -> None:
    contract = authoritative_crop_roi_pixel_contract(
        _HybridAcquisitionCropVideo(),
        run_id="hybrid_crop",
    )

    assert contract is not None
    assert contract["name"] == "orange_mono_pynvvc_luma_hybrid_uint8_v1"
    assert contract["source_pixel_routing_array"] == "source_pixel_kind_codes"
    assert contract["source_pixel_kind_map"] == {
        "acquisition_crop_video": 0,
        "offline_full_frame_supplemental_flat_cache": 1,
    }
    assert contract["source_pixel_contracts"]["acquisition_crop_video"][
        "source_pixels"
    ] == "acquisition_crop_video"
    assert contract["source_pixel_contracts"][
        "offline_full_frame_supplemental_flat_cache"
    ]["source_pixels"] == "raw_camera_video"


def test_invalid_strict_manifest_never_downgrades_to_legacy(tmp_path: Path) -> None:
    crop = _strict_crop(tmp_path, mode="a")
    manifest = copy.deepcopy(crop.attrs["run_manifest"])
    manifest["payload_digest"] = "0" * 64
    crop.attrs["run_manifest"] = manifest
    crop.attrs["crop_signature"] = "fallback-forbidden"
    crop.attrs["crop_revision"] = 1

    with pytest.raises(ValueError, match="payload_digest mismatch"):
        build_crop_run_reference(crop, run_id="strict_crop")


def test_flat_cache_planner_consumes_strict_reference_without_legacy_attrs(
    tmp_path: Path,
) -> None:
    crop = _strict_crop(tmp_path)
    archive = Path(str(crop.store.root)).parent.parent

    binding = plan_flat_roi_cache_binding(
        analysis_zarr=archive,
        crop_run="strict_crop",
        manifest_path=tmp_path / "cache" / "manifest.json",
        producer_job_key="cache:strict_crop",
        min_roi_size=1,
    )

    assert binding.crop_signature is None
    assert binding.crop_revision is None
    assert binding.crop_run_reference is not None
    assert binding.crop_run_reference["profile"] == CROP_RUN_REFERENCE_STRICT_PROFILE
    assert binding.shape == (4, 8, 8)
    assert binding.pixel_contract is not None
    assert binding.pixel_contract["name"] == "orange_mono_pynvvc_luma_uint8_v1"
    assert binding.pixel_contract["source_pixels"] == "raw_camera_video"


def test_keypoint_dag_rejects_wrong_pixel_source_for_strict_crop(
    tmp_path: Path,
) -> None:
    crop = _strict_crop(tmp_path)
    archive = Path(str(crop.store.root)).parent.parent
    reference = build_crop_run_reference(crop, run_id="strict_crop")
    binding = FlatRoiCacheBinding(
        manifest_path=tmp_path / "wrong-cache.json",
        manifest_sha256="a" * 64,
        payload_path=tmp_path / "wrong-cache.bin",
        crop_run="strict_crop",
        cache_key="wrong-source",
        crop_signature=None,
        crop_revision=None,
        shape=(4, 8, 8),
        total_bytes=4 * 8 * 8,
        payload_sha256=None,
        crop_run_reference=reference,
        pixel_contract=orange_mono_pynvvc_luma_pixel_contract(),
    )

    with pytest.raises(ValueError, match="pixel_contract"):
        validate_keypoint_input_dag(
            analysis_zarr=archive,
            cache=binding,
            min_roi_size=1,
        )


def test_dense_flat_cache_planner_rejects_variable_per_row_crop_shape(
    tmp_path: Path,
) -> None:
    output = tmp_path / "variable_crop.zarr"
    publish_refined_crop_geometry_shadow(
        _refined_source(tmp_path),
        policy=CropGeometryPolicy(
            purpose="variable_inspection",
            size_mode=CropSizeMode.VARIABLE_PER_ROW,
            padding_mode=CropPaddingMode.ZERO_OUTSIDE_SOURCE_FRAME,
        ),
        pixel_authority=_pixel(),
        destination=output,
        run_id="variable_crop",
        roi_sizes_full=np.asarray(
            [[8, 8], [12, 8], [10, 14], [6, 9]],
            dtype=np.int32,
        ),
        shadow_root=tmp_path,
    )

    with pytest.raises(ValueError, match="fixed ROI size"):
        plan_flat_roi_cache_binding(
            analysis_zarr=output,
            crop_run="variable_crop",
            manifest_path=tmp_path / "cache" / "variable.json",
            producer_job_key="cache:variable",
            min_roi_size=1,
        )
