from __future__ import annotations

import copy
from pathlib import Path

import pytest
import zarr

from fisheye.cluster.flat_roi_cache import plan_flat_roi_cache_binding
from fisheye.shared.zarr.crop_consumer import (
    CROP_RUN_REFERENCE_LEGACY_PROFILE,
    CROP_RUN_REFERENCE_STRICT_PROFILE,
    CROP_RUN_REFERENCE_UNVERSIONED_LEGACY_PROFILE,
    build_crop_run_reference,
    strict_crop_fixed_roi_shape,
    strict_crop_source_frame_shape,
    validate_crop_run_reference,
)
from fisheye.shared.zarr.crop_shadow import publish_refined_crop_geometry_shadow
from tests.unit.fisheye.test_crop_shadow import _pixel, _policy, _refined_source


class _LegacyCrop:
    attrs = {"crop_signature": {"revision": 4}, "crop_revision": 4}


class _UnversionedLegacyCrop:
    attrs = {}


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
    assert reference["run_manifest_digest"] == crop.attrs["run_manifest"][
        "payload_digest"
    ]
    assert reference["logical_content_digest"] == crop.attrs["run_manifest"][
        "payload"
    ]["logical_content"]["digest"]
    assert validate_crop_run_reference(reference) == reference
    assert strict_crop_fixed_roi_shape(crop, run_id="strict_crop") == (8, 8)
    assert strict_crop_source_frame_shape(crop, run_id="strict_crop") == (80, 100)


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
