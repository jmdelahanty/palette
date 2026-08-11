from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
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
from fisheye.shared.refined_subject_component_contours import (
    write_sampled_component_contour_arrays,
)
from fisheye.shared.zarr.subject_mask_sampled_contour_worker_receipt import (
    build_subject_mask_sampled_contour_worker_assembly,
    build_subject_mask_sampled_contour_worker_receipt,
    sampled_contour_worker_arrays,
    validate_subject_mask_sampled_contour_worker_receipt,
)
from fisheye.shared.zarr.subject_mask_validation_receipt import (
    subject_mask_array_unit_document,
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


def _sampled_worker(
    tmp_path: Path,
    *,
    name: str,
    masks: np.ndarray,
    global_start_row: int,
) -> tuple[zarr.Group, dict[str, object], dict[str, object]]:
    archive = tmp_path / f"{name}.zarr"
    root = zarr.open_group(str(archive), mode="w", zarr_format=3)
    run = root.create_group("refined_subject_masks_runs").create_group(name)
    run.attrs.update(
        {
            "mask_labels": list(_components().labels),
            "sampled_component_contours_status": "computed",
            "derived_mask_caches_stale": False,
            "contours_stale": False,
        }
    )
    run.create_array("masks_roi", data=np.asarray(masks, dtype=np.uint8))
    for component_index, component in enumerate(_components().labels):
        sample_count = {
            "subject_body": 128,
            "eye_left": 64,
            "eye_right": 64,
            "swim_bladder": 32,
        }[component]
        rows = int(masks.shape[0])
        points = np.empty((rows, sample_count, 2), dtype=np.float32)
        points[..., 0] = np.arange(sample_count, dtype=np.float32)
        points[..., 1] = np.float32(component_index)
        write_sampled_component_contour_arrays(
            run.require_group("components").require_group(component),
            points_xy=points,
            valid=np.ones((rows,), dtype=bool),
            source_point_count=np.full((rows,), sample_count, dtype=np.int32),
            component=component,
            source_mask_run=name,
            row_chunk=2,
        )
    dense_document = subject_mask_array_unit_document(
        {"masks_roi": run["masks_roi"]}, ("masks_roi",), unit_rows=2
    )
    worker_payload = {
        "stage_kind": "refined_subject_mask",
        "run_path": f"refined_subject_masks_runs/{name}",
        "arrays": dense_document,
    }
    worker_receipt = {
        "schema_id": "palette.subject_mask.worker_semantic_receipt",
        "schema_version": 1,
        "digest_algorithm": "sha256_canonical_json_v1",
        "payload_digest": canonical_json_sha256(worker_payload),
        "payload": worker_payload,
    }
    receipt = build_subject_mask_sampled_contour_worker_receipt(
        run,
        global_start_row=global_start_row,
        worker_receipt=worker_receipt,
        producer_commit="a" * 40,
        unit_rows=2,
    )
    return run, worker_receipt, receipt


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


def test_worker_sampled_contours_assemble_without_dense_regeneration(
    tmp_path: Path,
) -> None:
    source = _refined_source(tmp_path)
    source_run = zarr.open_group(
        str(source.output_path), mode="r", use_consolidated=False
    )[f"refined_subject_masks_runs/{source.run_id}"]
    masks = np.asarray(source_run["masks_roi"][:], dtype=np.uint8)
    workers = [
        _sampled_worker(
            tmp_path,
            name="refined_worker_0",
            masks=masks[:2],
            global_start_row=0,
        ),
        _sampled_worker(
            tmp_path,
            name="refined_worker_1",
            masks=masks[2:],
            global_start_row=2,
        ),
    ]
    evidence = {
        "workers": [
            {
                "global_row_interval": {
                    "start_row": index * 2,
                    "stop_row": index * 2 + 2,
                },
                "run_path": receipt["payload"]["source_run_path"],
                "worker_receipt_payload_digest": worker_receipt["payload_digest"],
            }
            for index, (_run, worker_receipt, receipt) in enumerate(workers)
        ]
    }
    receipts = [item[2] for item in workers]
    assembly = build_subject_mask_sampled_contour_worker_assembly(
        receipts,
        source_producer_evidence=evidence,
        n_rois=4,
        components=_components(),
        producer_commit="a" * 40,
    )
    worker_arrays = [
        sampled_contour_worker_arrays(run, components=_components())
        for run, _worker_receipt, _receipt in workers
    ]
    precomputed = {
        path: np.concatenate([arrays[path][:] for arrays in worker_arrays], axis=0)
        for path in worker_arrays[0]
    }

    publication = publish_selector_ineligible_subject_mask_sampled_contours(
        refined_snapshot_root=source.output_path,
        refined_run_id=source.run_id,
        destination=tmp_path / "assembled_cache.zarr",
        cache_run_id="cache_assembled",
        precomputed_arrays=precomputed,
        worker_assembly=assembly,
        created_by="pytest",
    )

    write_receipt = publication.manifest["payload"]["write_receipt"]
    assert publication.manifest["schema_version"] == 2
    assert write_receipt["source_mode"] == "receipt_bound_worker_arrays"
    assert write_receipt["effective_compute_workers"] == 0
    assert "bounded_dense_contour_generation" not in publication.phase_seconds
    assert "receipt_bound_worker_contour_assembly" in publication.phase_seconds
    result = zarr.open_group(
        str(publication.output_path), mode="r", use_consolidated=False
    )[f"{SUBJECT_MASK_CACHE_FAMILY}/{publication.run_id}"]
    for path, values in precomputed.items():
        assert np.array_equal(result[path][:], values, equal_nan=True)


def test_worker_sampled_contour_receipt_rejects_stale_or_changed_values(
    tmp_path: Path,
) -> None:
    masks = np.ones((2, 4, 8, 8), dtype=np.uint8)
    run, worker_receipt, receipt = _sampled_worker(
        tmp_path,
        name="refined_worker",
        masks=masks,
        global_start_row=0,
    )
    assert (
        validate_subject_mask_sampled_contour_worker_receipt(
            receipt, run=run, worker_receipt=worker_receipt
        )["payload_digest"]
        == receipt["payload_digest"]
    )

    points = run["components/subject_body/sampled_contours/points_xy"]
    changed = np.asarray(points[0])
    changed[0, 0] += 1
    points[0] = changed
    with pytest.raises(ValueError, match="logical values differ"):
        validate_subject_mask_sampled_contour_worker_receipt(
            receipt, run=run, worker_receipt=worker_receipt
        )

    run.attrs["contours_stale"] = True
    with pytest.raises(ValueError, match="contour cache is stale"):
        build_subject_mask_sampled_contour_worker_receipt(
            run,
            global_start_row=0,
            worker_receipt=worker_receipt,
            producer_commit="a" * 40,
        )


def test_default_worker_profile_forbids_full_ragged_contours(tmp_path: Path) -> None:
    run, worker_receipt, _receipt = _sampled_worker(
        tmp_path,
        name="refined_worker",
        masks=np.ones((2, 4, 8, 8), dtype=np.uint8),
        global_start_row=0,
    )
    run["components/subject_body"].create_group("contours")
    with pytest.raises(ValueError, match="full ragged contours are forbidden"):
        build_subject_mask_sampled_contour_worker_receipt(
            run,
            global_start_row=0,
            worker_receipt=worker_receipt,
            producer_commit="a" * 40,
        )


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
