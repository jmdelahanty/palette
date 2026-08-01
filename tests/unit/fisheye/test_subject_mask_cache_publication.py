from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import zarr

from fisheye.diagnostics.publish_subject_mask_sampled_contour_canary import (
    publish_canary,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.subject_mask_cache_publication import (
    SUBJECT_MASK_CACHE_FAMILY,
    SUBJECT_MASK_CACHE_RUN_MANIFEST_ATTRIBUTE,
    publish_selector_ineligible_subject_mask_sampled_contours,
    validate_persisted_subject_mask_cache_publication,
    validate_subject_mask_cache_run_manifest,
)
from fisheye.shared.zarr.subject_mask_cache_storage import (
    plan_subject_mask_sampled_contour_storage,
)
from fisheye.shared.zarr.subject_mask_core_publication import (
    publish_selector_ineligible_subject_mask_core_snapshot,
)
from fisheye.shared.zarr.subject_mask_schema import (
    SubjectMaskComponentRegistry,
    SubjectMaskDimensions,
    derive_subject_mask_frame_row_offsets,
    derive_subject_mask_metrics,
)


def _components() -> SubjectMaskComponentRegistry:
    return SubjectMaskComponentRegistry(
        ("subject_body", "eye_left", "eye_right", "swim_bladder")
    )


def _refined_source(tmp_path: Path):
    masks = np.zeros((4, 4, 16, 16), dtype=np.uint8)
    masks[:, 0, 2:14, 2:14] = 1
    masks[:, 1, 4:7, 4:7] = 1
    masks[:, 2, 4:7, 10:13] = 1
    masks[:, 3, 10:13, 7:10] = 1
    masks[1, 2] = 0
    metrics = derive_subject_mask_metrics(masks)
    frames = np.asarray([0, 0, 2, 3], dtype=np.int64)
    arrays = {
        "source_crop_row_ids": np.arange(4, dtype=np.int64),
        "instance_key": np.asarray([101, 102, 201, 301], dtype=np.uint64),
        "source_acquisition_frame_index": frames,
        "frame_row_offsets": derive_subject_mask_frame_row_offsets(frames, n_frames=4),
        "source_crop_xywh": np.asarray(
            [[0, 0, 16, 16], [1, 0, 16, 16], [0, 1, 16, 16], [1, 1, 16, 16]],
            dtype=np.float32,
        ),
        "available_channels": np.ones((4,), dtype=bool),
        "masks_roi": masks,
        **{f"metrics/{name}": values for name, values in metrics.items()},
    }
    crop = {
        "instance_key": arrays["instance_key"],
        "source_acquisition_frame_index": frames,
        "source_crop_xywh": arrays["source_crop_xywh"],
    }
    return publish_selector_ineligible_subject_mask_core_snapshot(
        arrays,
        source_crop_arrays=crop,
        source_manifest={
            "schema_id": "palette.subject_mask.cache_test_source",
            "schema_version": 1,
            "run_id": "source_001",
        },
        n_frames=4,
        components=_components(),
        destination=tmp_path / "refined.zarr",
        run_id="refined_001",
        kind="refined_dense_core",
        source_run_path="refined_subject_mask_shard_runs/source_001",
        created_by="pytest",
    )


def test_sampled_contour_storage_is_byte_derived_and_access_aware() -> None:
    dimensions = SubjectMaskDimensions(
        n_frames=1_188_000,
        n_rois=1_169_010,
        n_channels=4,
        roi_height=512,
        roi_width=512,
    )
    plans = plan_subject_mask_sampled_contour_storage(
        dimensions, components=_components()
    ).by_path()

    assert plans[
        "components/subject_body/sampled_contours/points_xy"
    ].plan.chunk_shape == (128, 128, 2)
    assert plans["components/eye_left/sampled_contours/points_xy"].plan.chunk_shape == (
        256,
        64,
        2,
    )
    assert plans[
        "components/swim_bladder/sampled_contours/points_xy"
    ].plan.chunk_shape == (512, 32, 2)
    assert plans["components/subject_body/sampled_contours/valid"].plan.chunk_shape == (
        131_072,
    )
    assert plans[
        "components/subject_body/sampled_contours/source_point_count"
    ].plan.chunk_shape == (32_768,)
    assert all(entry.plan.shard_shape is not None for entry in plans.values())
    assert all(entry.plan.shard_nbytes <= 8 * 1024 * 1024 for entry in plans.values())


def test_sampled_contours_publish_as_fresh_selector_ineligible_cache(
    tmp_path: Path,
) -> None:
    source = _refined_source(tmp_path)
    publication = publish_selector_ineligible_subject_mask_sampled_contours(
        refined_snapshot_root=source.output_path,
        refined_run_id=source.run_id,
        destination=tmp_path / "cache.zarr",
        cache_run_id="cache_001",
        source_compute_block_bytes=2 * 4 * 16 * 16,
        created_by="pytest",
    )

    assert publication.run_id == "cache_001"
    assert (
        validate_persisted_subject_mask_cache_publication(
            publication.output_path,
            run_id=publication.run_id,
            source_manifest=source.manifest,
        )
        == ()
    )
    root = zarr.open_group(
        str(publication.output_path), mode="r", use_consolidated=False
    )
    run = root[f"{SUBJECT_MASK_CACHE_FAMILY}/cache_001"]
    assert run.attrs["stage_selector_eligible"] is False
    assert run.attrs["palette_run_completion_status"] == "complete"
    body = run["components/subject_body/sampled_contours/points_xy"][:]
    right = run["components/eye_right/sampled_contours"]
    assert body.shape == (4, 128, 2)
    assert np.isfinite(body).all()
    assert right["valid"][:].tolist() == [True, False, True, True]
    assert np.isnan(right["points_xy"][1]).all()
    manifest = run.attrs[SUBJECT_MASK_CACHE_RUN_MANIFEST_ATTRIBUTE]
    assert len(manifest["payload"]["cache_extension"]["receipts"]) == 4
    assert manifest["payload"]["write_receipt"]["full_dense_equivalence"] is True


def test_sampled_contour_manifest_rejects_recomputed_nested_tampering(
    tmp_path: Path,
) -> None:
    source = _refined_source(tmp_path)
    publication = publish_selector_ineligible_subject_mask_sampled_contours(
        refined_snapshot_root=source.output_path,
        refined_run_id=source.run_id,
        destination=tmp_path / "cache.zarr",
        cache_run_id="cache_001",
        source_compute_block_bytes=2 * 4 * 16 * 16,
        created_by="pytest",
    )
    tampered = copy.deepcopy(publication.manifest)
    tampered["payload"]["storage_plan"]["arrays"][0]["plan"]["chunk_shape"][0] += 1
    tampered["payload_digest"] = canonical_json_sha256(tampered["payload"])
    assert any(
        "storage plan differs" in error
        for error in validate_subject_mask_cache_run_manifest(
            tampered, source_manifest=source.manifest
        )
    )

    tampered = copy.deepcopy(publication.manifest)
    receipt = tampered["payload"]["cache_extension"]["receipts"][0]
    receipt["payload"]["logical_content_digest"] = "00" * 32
    receipt["payload_digest"] = canonical_json_sha256(receipt["payload"])
    extension = tampered["payload"]["cache_extension"]
    extension["receipts_digest"] = canonical_json_sha256(extension["receipts"])
    tampered["payload_digest"] = canonical_json_sha256(tampered["payload"])
    assert any(
        "receipt content differs" in error
        for error in validate_subject_mask_cache_run_manifest(
            tampered, source_manifest=source.manifest
        )
    )


def test_full_duration_canary_harness_stages_locally_and_publishes_atomically(
    tmp_path: Path,
) -> None:
    source = _refined_source(tmp_path)
    source_metadata_before = (source.output_path / "zarr.json").read_bytes()
    destination = tmp_path / ".palette_benchmarks" / "contour_canary"

    result = publish_canary(
        source_analysis_zarr=source.output_path,
        source_refined_run=source.run_id,
        destination=destination,
        scratch_root=tmp_path / "node_local_scratch",
        canary_id="canary_001",
        cache_run_id="cache_001",
        source_compute_block_bytes=2 * 4 * 16 * 16,
        compute_workers=2,
        palette_commit="0123456789abcdef",
    )

    assert source_metadata_before == (source.output_path / "zarr.json").read_bytes()
    assert destination.is_dir()
    assert not any(
        "publish_tmp" in child.name for child in destination.parent.iterdir()
    )
    canary_manifest = json.loads(
        (destination / "canary_manifest.json").read_text(encoding="utf-8")
    )
    publication_receipt = json.loads(
        (destination / "publication_receipt.json").read_text(encoding="utf-8")
    )
    assert canary_manifest == result["canary_manifest"]
    assert publication_receipt == result["publication_receipt"]
    assert canary_manifest["execution"]["compute_and_zarr_write"] == ("node_local_only")
    assert canary_manifest["execution"]["compute_workers"] == 2
    assert publication_receipt["canary_manifest_digest"] == canonical_json_sha256(
        canary_manifest
    )
    assert (
        validate_persisted_subject_mask_cache_publication(
            destination / "cache.zarr",
            run_id="cache_001",
            source_manifest=source.manifest,
        )
        == ()
    )
