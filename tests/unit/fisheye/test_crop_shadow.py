from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.shared.instance_keys import mint_detection_instance_keys
from fisheye.shared.zarr.crop_manifest import CropPixelAuthority
from fisheye.shared.zarr.crop_manifest import (
    CROP_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION,
)
from fisheye.shared.zarr.crop_schema import (
    CropGeometryPolicy,
    CropPaddingMode,
    CropSizeMode,
)
from fisheye.shared.zarr.crop_shadow import (
    open_persisted_crop_geometry_publication,
    prepare_crop_geometry_from_refined_source,
    publish_refined_crop_geometry_shadow,
    require_safe_crop_geometry_shadow_destination,
    validate_crop_geometry_shadow_publication,
)
from fisheye.shared.zarr.detection_schema import (
    CanonicalDetectionDimensions,
    derive_canonical_detection_geometry,
)
from fisheye.shared.zarr.refined_detection_crop_source import (
    bind_refined_detection_crop_source,
)
from fisheye.shared.zarr.refined_detection_manifest import (
    RefinedDetectionSnapshotLineage,
    RefinedDetectionSourceIdentity,
)
from fisheye.shared.zarr.refined_detection_snapshot import (
    publish_selector_ineligible_refined_detection_snapshot,
)
from fisheye.shared.zarr.refined_detection_transition import (
    build_accept_all_refined_detection_root,
)


RECORDING_IDENTITY = "crop_shadow_multi_subject"
REFINED_RUN_ID = "refined_crop_source"


def _refined_source(tmp_path: Path):
    dimensions = CanonicalDetectionDimensions(
        n_frames=4,
        n_instances=4,
        source_width=100,
        source_height=80,
    )
    frames = np.asarray([0, 0, 2, 3], dtype=np.int32)
    boxes = np.asarray(
        [
            [0.20, 0.20, 0.10, 0.10],
            [0.70, 0.20, 0.10, 0.10],
            [0.50, 0.70, 0.20, 0.10],
            [0.25, 0.75, 0.10, 0.15],
        ],
        dtype=np.float32,
    )
    classes = np.asarray([1, 2, 1, 2], dtype=np.int32)
    bbox_img, centers = derive_canonical_detection_geometry(
        boxes,
        source_width=dimensions.source_width,
        source_height=dimensions.source_height,
    )
    canonical = {
        "instances/frame_indices": frames,
        "instances/source_acquisition_frame_index": frames.astype(np.int64),
        "instances/instance_key": mint_detection_instance_keys(
            recording_identity=RECORDING_IDENTITY,
            frame_indices=frames,
            bbox_norm_coords=boxes,
            class_ids=classes,
        ),
        "instances/bbox_norm_coords": boxes,
        "instances/bbox_img_xyxy": bbox_img,
        "instances/centers_img_xy": centers,
        "instances/scores": np.asarray([0.9, 0.8, 0.7, 0.6], dtype=np.float32),
        "instances/class_ids": classes,
        "instances/frame_row_offsets": np.asarray(
            [0, 2, 2, 3, 4],
            dtype=np.int64,
        ),
    }
    transition = build_accept_all_refined_detection_root(
        canonical,
        dimensions=dimensions,
        recording_identity=RECORDING_IDENTITY,
    )
    source_root = tmp_path / "refined"
    publication = publish_selector_ineligible_refined_detection_snapshot(
        dimensions=transition.dimensions,
        arrays=transition.arrays,
        instance_reason_codes=transition.instance_reason_codes,
        source_reason_codes=transition.source_reason_codes,
        destination=source_root / "source.zarr",
        run_id=REFINED_RUN_ID,
        lineage=RefinedDetectionSnapshotLineage(
            lineage_id="11111111-1111-4111-8111-111111111111",
            snapshot_id="22222222-2222-4222-8222-222222222222",
            recording_identity=RECORDING_IDENTITY,
            next_refined_row_id=4,
        ),
        source=RefinedDetectionSourceIdentity(
            run_id="detect_source",
            run_manifest_digest="a" * 64,
            logical_content_digest="b" * 64,
        ),
        created_by="test",
        publication_kind="crop_shadow_source",
        safe_root=source_root,
    )
    return bind_refined_detection_crop_source(
        publication.output_path,
        run_id=REFINED_RUN_ID,
        allow_selector_ineligible_benchmark=True,
    )


def _pixel() -> CropPixelAuthority:
    return CropPixelAuthority(
        authority_id=(
            "sleepyfish_camera_video_v1"
            "#decode=orange_mono_pynvvc_luma_uint8_v1"
        ),
        authority_manifest_digest="c" * 64,
        recording_identity=RECORDING_IDENTITY,
        camera_identity="cam2010095",
        n_frames=4,
        source_width=100,
        source_height=80,
    )


def _policy() -> CropGeometryPolicy:
    return CropGeometryPolicy(
        purpose="subject_analysis",
        size_mode=CropSizeMode.FIXED_PER_RUN,
        fixed_size_wh=(8, 8),
        padding_mode=CropPaddingMode.ZERO_OUTSIDE_SOURCE_FRAME,
    )


def test_real_shadow_is_geometry_only_consolidated_and_selector_ineligible(
    tmp_path: Path,
) -> None:
    source = _refined_source(tmp_path)
    shadow_root = tmp_path / "crop_shadows"
    publication = publish_refined_crop_geometry_shadow(
        source,
        policy=_policy(),
        pixel_authority=_pixel(),
        destination=shadow_root / "crop.zarr",
        run_id="crop_geometry_shadow",
        shadow_root=shadow_root,
        coordinate_catalog=True,
    )

    assert validate_crop_geometry_shadow_publication(publication) == ()
    assert publication.manifest["schema_version"] == (
        CROP_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION
    )
    assert "coordinate_contract" in publication.manifest["payload"]
    assert publication.receipt["production_state_changes"] == []
    assert publication.receipt["selector_eligible"] is False
    assert len(publication.arrays) == 13
    assert "roi_images" not in publication.arrays
    np.testing.assert_array_equal(
        publication.arrays["frame_row_offsets"][:],
        [0, 2, 2, 3, 4],
    )
    np.testing.assert_array_equal(
        publication.arrays["source_refined_row_ids"][:],
        [0, 1, 2, 3],
    )
    np.testing.assert_array_equal(
        publication.arrays["roi_sizes_full"][:],
        np.full((4, 2), 8, dtype=np.int32),
    )

    root = zarr.open_group(
        str(publication.output_path),
        mode="r",
        use_consolidated=False,
    )
    family = root["crop_runs"]
    run = family["crop_geometry_shadow"]
    assert "latest" not in family.attrs
    assert run.attrs["status"] == "complete"
    assert run.attrs["stage_selector_eligible"] is False
    assert run.attrs["run_manifest"]["payload_digest"] == (
        publication.manifest["payload_digest"]
    )
    assert publication.output_path.joinpath("zarr.json").is_file()
    assert (
        publication.output_path
        / "crop_runs"
        / "crop_geometry_shadow"
        / "zarr.json"
    ).is_file()

    rebound = open_persisted_crop_geometry_publication(
        publication.output_path,
        run_id="crop_geometry_shadow",
        source_refined_archive=source.archive_path,
    )
    assert rebound.output_path == publication.output_path
    assert rebound.source_manifest == source.manifest
    assert rebound.receipt["persisted_archive_path"] == str(
        publication.output_path
    )


def test_variable_size_preparation_preserves_refined_rows_without_pixels(
    tmp_path: Path,
) -> None:
    source = _refined_source(tmp_path)
    policy = CropGeometryPolicy(
        purpose="inspection",
        size_mode=CropSizeMode.VARIABLE_PER_ROW,
        padding_mode=CropPaddingMode.ZERO_OUTSIDE_SOURCE_FRAME,
    )
    sizes = np.asarray([[8, 8], [12, 8], [10, 14], [6, 9]], dtype=np.int32)

    prepared = prepare_crop_geometry_from_refined_source(
        source,
        policy=policy,
        pixel_authority=_pixel(),
        roi_sizes_full=sizes,
    )

    np.testing.assert_array_equal(prepared.arrays["roi_sizes_full"], sizes)
    np.testing.assert_array_equal(
        prepared.arrays["source_crop_xywh"][:, 2:].astype(np.int32),
        sizes,
    )
    assert "roi_images" not in prepared.arrays
    assert prepared.arrays["frame_indices"].dtype == np.dtype(np.int64)


def test_shadow_validation_detects_post_publication_geometry_tampering(
    tmp_path: Path,
) -> None:
    source = _refined_source(tmp_path)
    shadow_root = tmp_path / "crop_shadows"
    publication = publish_refined_crop_geometry_shadow(
        source,
        policy=_policy(),
        pixel_authority=_pixel(),
        destination=shadow_root / "tamper.zarr",
        run_id="crop_geometry_shadow",
        shadow_root=shadow_root,
    )
    coordinates = publication.arrays["roi_coordinates_full"]
    changed = np.asarray(coordinates[0], dtype=np.int32)
    changed[0] += 1
    coordinates[0] = changed

    errors = validate_crop_geometry_shadow_publication(publication)
    assert any(
        "logical array validation failed" in error
        or "differs from decoded arrays" in error
        for error in errors
    )


def test_pixel_authority_and_destination_guards_fail_closed(tmp_path: Path) -> None:
    source = _refined_source(tmp_path)
    wrong_pixel = CropPixelAuthority(
        authority_id="wrong_recording_video",
        authority_manifest_digest="d" * 64,
        recording_identity="another_recording",
        camera_identity="cam2010095",
        n_frames=4,
        source_width=100,
        source_height=80,
    )
    with pytest.raises(ValueError, match="different recordings"):
        prepare_crop_geometry_from_refined_source(
            source,
            policy=_policy(),
            pixel_authority=wrong_pixel,
        )

    with pytest.raises(ValueError, match="must be a child"):
        require_safe_crop_geometry_shadow_destination(
            tmp_path / "outside.zarr",
            shadow_root=tmp_path / "safe",
        )
