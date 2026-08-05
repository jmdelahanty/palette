from __future__ import annotations

from io import BytesIO
import hashlib
import json
from pathlib import Path
import tarfile
import time
from uuid import NAMESPACE_URL, uuid5

import numpy as np
import pytest
import zarr

from fisheye.shared.subject_mask_chunks import refined_subject_mask_storage_chunks
from fisheye.shared.subject_mask_attempt import (
    build_subject_mask_attempt,
    build_subject_mask_scientific_identity,
)
from fisheye.shared.subject_mask_worker_receipt import (
    REFINED_SUBJECT_MASK_WORKER_OUTPUT_PATHS,
    build_subject_mask_worker_semantic_receipt,
)
from fisheye.shared.detect_reason_codec import read_reason_labels, write_reason_columns
from fisheye.shared.refined_subject_mask_encoded_chunks import (
    prepare_global_mask_chunk_grid,
)
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    mark_run_complete,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_bytes
from fisheye.shared.zarr.subject_mask_validation_receipt import (
    subject_mask_array_unit_document,
)
from fisheye.shared.zarr.subject_mask_schema import derive_subject_mask_metrics
from fisheye.utils.convert_refined_subject_mask_clip_package_v2 import convert_package
from fisheye.utils.finalize_subject_mask_clip_package import PACKAGE_SCHEMA_ID
from fisheye.utils.import_refined_subject_mask_clip_packages import (
    ClipPackage,
    _build_row_chunk_copy_plan,
    import_refined_subject_mask_clip_packages,
)
from fisheye.utils.repair_refined_subject_mask_frame_counts import repair_frame_counts
from fisheye.utils.validate_encoded_subject_mask_import_canary import validate_canary


def _write_target_crop_contract(
    target_zarr: Path,
    *,
    crop_run: str = "crop_proxy_collection",
    recording_frame_count: int = 2048,
) -> None:
    root = zarr.open_group(str(target_zarr), mode="a", use_consolidated=False)
    root.attrs["recording_frame_index_row_count"] = int(recording_frame_count)
    crop = root.require_group("crop_runs").require_group(crop_run)
    crop.attrs["recording_frame_index_row_count"] = int(recording_frame_count)


def _write_package(
    tmp_path: Path,
    *,
    package_name: str,
    run_name: str,
    crop_row_ids: list[int],
    source_crop_run: str = "crop_proxy_collection",
    labels: list[str] | None = None,
    body_reason_labels: list[str] | None = None,
    mask_row_chunk: int | None = None,
    frame_indices: list[int] | None = None,
    production_proof: bool = False,
    roi_shape: tuple[int, int] = (2, 3),
) -> Path:
    labels = labels or ["subject_body", "eye_left"]
    package_root = tmp_path / f"{package_name}_src"
    run_path = package_root / "refined_subject_masks_runs" / run_name
    run = zarr.open_group(str(run_path), mode="w")
    run.attrs["mask_labels"] = labels
    run.attrs["label_schema_id"] = "subject_v1_lr"
    run.attrs["source_crop_run"] = source_crop_run
    run.attrs["summary_statistics"] = {"rows_total": len(crop_row_ids)}
    run.attrs["clip_package_host"] = f"{package_name}_host"
    run.attrs["clip_package_lsb_jobid"] = f"{package_name}_job"
    run.attrs["source_roi_cache_used"] = True
    run.attrs["source_roi_cache_path"] = f"/tmp/{package_name}.flat_roi_cache.json"
    run.attrs["source_roi_cache_canonical_path"] = (
        f"/nrs/{package_name}.flat_roi_cache.json"
    )
    run.attrs["source_roi_cache_key"] = f"{package_name}_cache_key"
    run.attrs["source_subject_mask_shard_runs"] = [f"subject_masks_{run_name}"]
    run.attrs["source_subject_mask_shard_run_paths"] = [
        f"subject_mask_shard_runs/subject_masks_{run_name}"
    ]
    run.attrs["source_subject_mask_shard_crop_runs"] = [f"crop_{run_name}"]

    row_count = len(crop_row_ids)
    roi_height, roi_width = (int(roi_shape[0]), int(roi_shape[1]))
    masks = np.zeros((row_count, len(labels), roi_height, roi_width), dtype=np.uint8)
    for row_idx, crop_row_id in enumerate(crop_row_ids):
        if production_proof:
            masks[
                row_idx,
                :,
                row_idx % roi_height,
                int(crop_row_id) % roi_width,
            ] = 1
        else:
            masks[row_idx, :, :, :] = np.uint8(crop_row_id)
    run.create_array(
        "masks_roi",
        data=masks,
        chunks=(mask_row_chunk or max(1, row_count), 1, 2, 3),
        overwrite=True,
    )
    run.create_array(
        "source_crop_row_ids",
        data=np.asarray(crop_row_ids, dtype=np.int64),
        overwrite=True,
    )
    frame_values = np.asarray(
        (
            frame_indices
            if frame_indices is not None
            else (np.asarray(crop_row_ids, dtype=np.int64) + 1000)
        ),
        dtype=np.int64,
    )
    if int(frame_values.shape[0]) != row_count:
        raise ValueError("frame_indices must have one value per crop row")
    run.create_array("frame_indices", data=frame_values, overwrite=True)
    clip_frame_counts = np.bincount(
        frame_values,
        minlength=(int(frame_values.max()) + 1 if frame_values.size else 1),
    ).astype(np.int32)
    run.create_array(
        "frame_counts",
        data=clip_frame_counts,
        chunks=(min(1024, clip_frame_counts.size),),
    )
    run.create_array(
        "available_channels",
        data=np.ones((len(labels),), dtype=bool),
        overwrite=True,
    )

    metrics = run.require_group("metrics")
    metric_values = np.stack(
        [
            np.asarray(crop_row_ids, dtype=np.float32),
            np.asarray(crop_row_ids, dtype=np.float32) + 0.5,
        ],
        axis=1,
    )
    metrics.create_array(
        "area_px",
        data=metric_values[:, : len(labels)],
        chunks=(max(1, row_count), len(labels)),
    )

    components = run.require_group("components")
    body = components.require_group("subject_body")
    body.create_array(
        "manual_override",
        data=np.zeros((row_count,), dtype=bool),
        chunks=(max(1, row_count),),
    )
    if body_reason_labels is not None:
        write_reason_columns(
            body,
            np.asarray(body_reason_labels, dtype=object),
            chunk_size=max(1, row_count),
            overwrite=True,
        )
    contours = body.require_group("contours")
    ptr = np.arange(row_count, dtype=np.int64)
    length = np.ones((row_count,), dtype=np.int32)
    points_xy = np.asarray(
        [
            [float(crop_row_id), float(crop_row_id) + 0.25]
            for crop_row_id in crop_row_ids
        ],
        dtype=np.float32,
    )
    contours.create_array("ptr", data=ptr, chunks=(max(1, row_count),))
    contours.create_array("len", data=length, chunks=(max(1, row_count),))
    contours.create_array("points_xy", data=points_xy, chunks=(max(1, row_count), 2))

    worker_proof = None
    if production_proof:
        for metric_name, metric_values in derive_subject_mask_metrics(masks).items():
            metrics.create_array(
                metric_name,
                data=metric_values,
                chunks=(max(1, row_count), *metric_values.shape[1:]),
                overwrite=True,
            )
        run.attrs["stage_selector_eligible"] = False
        run_path_text = f"refined_subject_masks_runs/{run_name}"
        science = build_subject_mask_scientific_identity(
            stage_kind="refined_subject_mask",
            model={"artifact": "pytest_refiner"},
            crop={"run_id": source_crop_run},
            pixels={"digest": hashlib.sha256(package_name.encode()).hexdigest()},
            row_identity={"source_crop_row_ids": list(crop_row_ids)},
            inference_contract={"components": list(labels)},
            schema_version=1,
        )
        attempt = build_subject_mask_attempt(
            scientific_identity=science,
            run_path=run_path_text,
            attempt_id=str(uuid5(NAMESPACE_URL, f"pytest:{run_path_text}")),
        )
        arrays = {path: run[path] for path in REFINED_SUBJECT_MASK_WORKER_OUTPUT_PATHS}
        receipt = build_subject_mask_worker_semantic_receipt(
            stage_kind="refined_subject_mask",
            run_path=run_path_text,
            scientific_identity=science,
            attempt=attempt,
            scope={"package_name": package_name},
            row_count=row_count,
            array_document=subject_mask_array_unit_document(
                arrays,
                REFINED_SUBJECT_MASK_WORKER_OUTPUT_PATHS,
                unit_rows=max(1, min(2, row_count)),
            ),
            required_paths=REFINED_SUBJECT_MASK_WORKER_OUTPUT_PATHS,
            roi_aligned_paths=tuple(
                path
                for path in REFINED_SUBJECT_MASK_WORKER_OUTPUT_PATHS
                if path != "available_channels"
            ),
        )
        receipt_bytes = canonical_json_bytes(receipt)
        receipt_relative = f"{run_path_text}/worker_semantic_receipt.json"
        receipt_path = package_root / receipt_relative
        receipt_path.write_bytes(receipt_bytes)
        binding = {
            "schema_id": receipt["schema_id"],
            "schema_version": receipt["schema_version"],
            "payload_digest": receipt["payload_digest"],
            "relative_path": receipt_relative,
            "document_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
            "storage": "strict_json_sidecar_v1",
        }
        run.attrs.update(
            {
                "subject_mask_scientific_identity": science,
                "subject_mask_attempt": attempt,
                "subject_mask_worker_semantic_receipt_binding": binding,
            }
        )
        worker_proof = {
            "scientific_identity_digest": science["digest"],
            "attempt_id": attempt["payload"]["attempt_id"],
            "attempt_payload_digest": attempt["payload_digest"],
            "semantic_receipt_payload_digest": receipt["payload_digest"],
            "semantic_receipt_document_sha256": binding["document_sha256"],
            "semantic_receipt_relative_path": receipt_relative,
        }

    # Clip packages are immutable handoff artifacts.  Even legacy-compatible
    # test packages must carry both the completed run envelope and the sealed
    # package envelope before the importer may inspect their payload.
    mark_run_complete(run, run_name=run_name)

    package_path = tmp_path / f"{package_name}.tar.gz"
    manifest = {
        "schema_id": PACKAGE_SCHEMA_ID,
        "created_at_utc": "2026-07-08T00:00:00+00:00",
        "package_completion_status": "complete",
        "run_group_path": f"refined_subject_masks_runs/{run_name}",
        "summary": {"rows_total": row_count},
    }
    if worker_proof is not None:
        manifest["worker_proof"] = worker_proof
    manifest_bytes = (json.dumps(manifest, sort_keys=True) + "\n").encode("utf-8")
    with tarfile.open(package_path, "w:gz") as tar:
        tar.add(run_path, arcname=f"refined_subject_masks_runs/{run_name}")
        info = tarfile.TarInfo("package.json")
        info.size = len(manifest_bytes)
        info.mtime = int(time.time())
        tar.addfile(info, BytesIO(manifest_bytes))
    return package_path


def test_row_chunk_copy_plan_splits_misaligned_clip_boundary_once() -> None:
    package_a_path = Path("/tmp/clip_a.tar.gz")
    package_b_path = Path("/tmp/clip_b.tar.gz")
    package_a = ClipPackage(
        package_path=package_a_path,
        extract_dir=Path("/tmp/clip_a"),
        manifest={},
        run_name="clip_a",
        group=None,  # type: ignore[arg-type]
        source_crop_row_ids=np.arange(6, dtype=np.int64),
    )
    package_b = ClipPackage(
        package_path=package_b_path,
        extract_dir=Path("/tmp/clip_b"),
        manifest={},
        run_name="clip_b",
        group=None,  # type: ignore[arg-type]
        source_crop_row_ids=np.arange(6, 10, dtype=np.int64),
    )
    row_maps = {
        package_a_path: (np.arange(6, dtype=np.int64), np.arange(6, dtype=np.int64)),
        package_b_path: (
            np.arange(4, dtype=np.int64),
            np.arange(6, 10, dtype=np.int64),
        ),
    }

    plan = _build_row_chunk_copy_plan(
        [package_a, package_b],
        row_maps=row_maps,
        total_rows=10,
        row_chunk=4,
    )

    assert [(chunk.dst_start, chunk.dst_stop) for chunk in plan] == [
        (0, 4),
        (4, 8),
        (8, 10),
    ]
    assert [
        [
            (run.package_path, run.src_start, run.src_stop, run.dst_start, run.dst_stop)
            for run in chunk.runs
        ]
        for chunk in plan
    ] == [
        [(package_a_path, 0, 4, 0, 4)],
        [
            (package_a_path, 4, 6, 4, 6),
            (package_b_path, 0, 2, 6, 8),
        ],
        [(package_b_path, 2, 4, 8, 10)],
    ]


def test_encoded_v2_packages_copy_complete_chunks_and_decode_only_boundary(
    tmp_path: Path,
) -> None:
    target_zarr = tmp_path / "target.zarr"
    root = zarr.open_group(str(target_zarr), mode="w")
    crop = root.require_group("crop_runs").create_group("crop_proxy_collection")
    crop.create_array(
        "source_crop_row_ids",
        data=np.arange(10, 20, dtype=np.int64),
        chunks=(4,),
        overwrite=True,
    )
    _write_target_crop_contract(target_zarr)
    grid_manifest = tmp_path / "global_mask_grid.json"
    prepare_global_mask_chunk_grid(
        zarr_path=target_zarr,
        crop_run="crop_proxy_collection",
        output_manifest=grid_manifest,
        mask_labels=("subject_body", "eye_left"),
        mask_height=2,
        mask_width=3,
        dense_mask_row_chunk=4,
    )
    package_a_v1 = _write_package(
        tmp_path,
        package_name="clip_a_v1",
        run_name="refined_clip_a",
        crop_row_ids=[10, 11, 12, 13, 14, 15],
        mask_row_chunk=4,
    )
    package_b_v1 = _write_package(
        tmp_path,
        package_name="clip_b_v1",
        run_name="refined_clip_b",
        crop_row_ids=[16, 17, 18, 19],
        mask_row_chunk=4,
    )
    import_refined_subject_mask_clip_packages(
        zarr_path=target_zarr,
        package_paths=[package_a_v1, package_b_v1],
        output_run="refined_collection_v1_baseline",
        expected_target_crop_run="crop_proxy_collection",
        array_copy_workers=2,
    )
    package_a_v2 = tmp_path / "clip_a_v2.tar.gz"
    package_b_v2 = tmp_path / "clip_b_v2.tar.gz"
    convert_package(
        source_package=package_a_v1,
        output_package=package_a_v2,
        grid_manifest=grid_manifest,
        copy_workers=2,
    )
    convert_package(
        source_package=package_b_v1,
        output_package=package_b_v2,
        grid_manifest=grid_manifest,
        copy_workers=2,
    )

    result = import_refined_subject_mask_clip_packages(
        zarr_path=target_zarr,
        package_paths=[package_a_v2, package_b_v2],
        output_run="refined_collection_v2",
        expected_target_crop_run="crop_proxy_collection",
        array_copy_workers=2,
        encoded_copy_workers=2,
    )

    publication = result["encoded_mask_publication"]
    assert publication["strategy"] == "encoded_global_chunk_copy_v1"
    assert publication["row_chunk_count"] == 3
    assert publication["direct_row_chunk_count"] == 2
    assert publication["boundary_row_chunk_count"] == 1
    run = zarr.open_group(str(target_zarr), mode="r", use_consolidated=False)[
        "refined_subject_masks_runs"
    ]["refined_collection_v2"]
    baseline = zarr.open_group(str(target_zarr), mode="r", use_consolidated=False)[
        "refined_subject_masks_runs"
    ]["refined_collection_v1_baseline"]
    assert run.attrs["masks_roi_publication_strategy"] == "encoded_global_chunk_copy_v1"
    np.testing.assert_array_equal(
        run["source_crop_row_ids"][:],
        np.arange(10, 20, dtype=np.int64),
    )
    np.testing.assert_array_equal(
        run["masks_roi"][:, 0, 0, 0],
        np.arange(10, 20, dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        run["masks_roi"][:, 1, 1, 2],
        np.arange(10, 20, dtype=np.uint8),
    )
    np.testing.assert_array_equal(run["masks_roi"][:], baseline["masks_roi"][:])
    np.testing.assert_array_equal(run["frame_indices"][:], baseline["frame_indices"][:])
    np.testing.assert_allclose(
        run["metrics"]["area_px"][:], baseline["metrics"]["area_px"][:]
    )
    canary = validate_canary(
        zarr_path=target_zarr,
        baseline_run="refined_collection_v1_baseline",
        encoded_run="refined_collection_v2",
        sample_row_chunks=3,
    )
    assert canary["status"] == "ok"
    assert canary["boundary_row_chunk_indices"] == [1]
    with pytest.raises(ValueError, match="Refusing to overwrite complete"):
        import_refined_subject_mask_clip_packages(
            zarr_path=target_zarr,
            package_paths=[package_a_v2, package_b_v2],
            output_run="refined_collection_v2",
            expected_target_crop_run="crop_proxy_collection",
            overwrite=True,
        )


def test_import_refined_subject_mask_clip_packages_merges_rows_and_contours(
    tmp_path: Path,
) -> None:
    target_zarr = tmp_path / "target.zarr"
    zarr.open_group(str(target_zarr), mode="w")
    _write_target_crop_contract(target_zarr, recording_frame_count=32_768)
    package_a = _write_package(
        tmp_path,
        package_name="clip_a",
        run_name="refined_clip_a",
        crop_row_ids=[10, 12],
    )
    package_b = _write_package(
        tmp_path,
        package_name="clip_b",
        run_name="refined_clip_b",
        crop_row_ids=[11],
    )

    result = import_refined_subject_mask_clip_packages(
        zarr_path=target_zarr,
        package_paths=[package_a, package_b],
        output_run="refined_collection",
        expected_target_crop_run="crop_proxy_collection",
    )

    assert result["status"] == "ok"
    assert result["row_count"] == 3

    root = zarr.open_group(str(target_zarr), mode="r", use_consolidated=False)
    parent = root["refined_subject_masks_runs"]
    run = parent["refined_collection"]
    assert parent.attrs["latest_complete"] == "refined_collection"
    assert parent.attrs["latest"] == "refined_collection"
    assert (
        parent.attrs["refined_subject_mask_review_status_latest"]
        == "refined_collection"
    )
    assert run.attrs[RUN_COMPLETION_STATUS_ATTR] == "complete"
    assert run.attrs["mask_storage_authority"] == "masks_roi"
    assert run.attrs["editable_mask_surface"] == "masks_roi"
    assert run.attrs["masks_roi_materialized"] is True
    assert run.attrs["mask_bitpacked_materialized"] is False
    assert run.attrs["mask_rle_materialized"] is False
    assert run.attrs["derived_mask_caches_stale"] is False
    assert run.attrs["mask_store_contract_encodings"] == ["dense_uint8_v1"]

    np.testing.assert_array_equal(
        run["source_crop_row_ids"][:], np.asarray([10, 11, 12], dtype=np.int64)
    )
    np.testing.assert_array_equal(
        run["frame_indices"][:], np.asarray([1010, 1011, 1012], dtype=np.int64)
    )
    expected_frame_counts = np.zeros((32_768,), dtype=np.int32)
    expected_frame_counts[[1010, 1011, 1012]] = 1
    np.testing.assert_array_equal(run["frame_counts"][:], expected_frame_counts)
    assert run["frame_counts"].attrs["frame_counts_generation"] == (
        "bincount_of_assembled_frame_indices_v1"
    )
    assert run["frame_counts"].attrs["palette_physical_layout"] == "indexed_sharding_v1"
    assert tuple(
        int(value) for value in run["masks_roi"].chunks
    ) == refined_subject_mask_storage_chunks(3, 2, 3)
    assert run["masks_roi"].shape == (3, 2, 2, 3)
    np.testing.assert_array_equal(
        run["masks_roi"][:, 0, 0, 0], np.asarray([10, 11, 12], dtype=np.uint8)
    )
    np.testing.assert_allclose(
        run["metrics"]["area_px"][:, 0],
        np.asarray([10.0, 11.0, 12.0], dtype=np.float32),
    )

    contours = run["components"]["subject_body"]["contours"]
    np.testing.assert_array_equal(
        contours["ptr"][:], np.asarray([0, 1, 2], dtype=np.int64)
    )
    np.testing.assert_array_equal(contours["len"][:], np.ones((3,), dtype=np.int32))
    np.testing.assert_allclose(
        contours["points_xy"][:],
        np.asarray([[10.0, 10.25], [11.0, 11.25], [12.0, 12.25]], dtype=np.float32),
    )
    assert run.attrs["component_contours_status"] == "computed"
    assert run.attrs["component_contours_components"] == ["subject_body"]
    assert "clip_package_host" not in run.attrs
    assert "clip_package_lsb_jobid" not in run.attrs
    assert "source_roi_cache_path" not in run.attrs
    assert "source_roi_cache_canonical_path" not in run.attrs
    assert "source_roi_cache_key" not in run.attrs
    assert run.attrs["clip_package_hosts"] == ["clip_a_host", "clip_b_host"]
    assert run.attrs["clip_package_lsb_jobids"] == ["clip_a_job", "clip_b_job"]
    assert run.attrs["source_roi_cache_used"] is True
    assert run.attrs["source_roi_cache_package_count"] == 2
    assert run.attrs["source_roi_cache_paths"] == [
        "/tmp/clip_a.flat_roi_cache.json",
        "/tmp/clip_b.flat_roi_cache.json",
    ]
    assert run.attrs["source_roi_cache_keys"] == [
        "clip_a_cache_key",
        "clip_b_cache_key",
    ]
    assert run.attrs["source_subject_mask_shard_runs"] == [
        "subject_masks_refined_clip_a",
        "subject_masks_refined_clip_b",
    ]


def test_production_import_aggregates_two_clip_receipts_and_stays_inactive(
    tmp_path: Path,
) -> None:
    target_zarr = tmp_path / "production_target.zarr"
    root = zarr.open_group(str(target_zarr), mode="w", zarr_format=3)
    root.attrs.update(
        {
            "recording_id": "recording_001",
            "recording_frame_index_row_count": 4,
        }
    )
    crop = root.require_group("crop_runs").create_group("crop_proxy_collection")
    crop.create_array(
        "instance_key",
        data=np.asarray([101, 102, 201, 301], dtype=np.uint64),
    )
    crop.create_array(
        "source_acquisition_frame_index",
        data=np.asarray([0, 0, 2, 3], dtype=np.int64),
    )
    crop.create_array(
        "source_crop_xywh",
        data=np.asarray(
            [[0, 0, 8, 8], [1, 0, 8, 8], [0, 1, 8, 8], [1, 1, 8, 8]],
            dtype=np.float32,
        ),
    )
    package_a = _write_package(
        tmp_path,
        package_name="proof_clip_a",
        run_name="refined_proof_a",
        crop_row_ids=[0, 1],
        frame_indices=[0, 0],
        production_proof=True,
    )
    package_b = _write_package(
        tmp_path,
        package_name="proof_clip_b",
        run_name="refined_proof_b",
        crop_row_ids=[2, 3],
        frame_indices=[2, 3],
        production_proof=True,
    )

    result = import_refined_subject_mask_clip_packages(
        zarr_path=target_zarr,
        package_paths=(package_b, package_a),
        output_run="refined_recording_draft",
        expected_target_crop_run="crop_proxy_collection",
        require_production_proof=True,
        array_copy_workers=2,
    )

    assert result["status"] == "ok"
    assert result["package_count"] == 2
    assert result["selector_eligible"] is False
    assert result["source_manifest_payload_digest"]
    assert result["source_validation_receipt_payload_digest"]
    reopened = zarr.open_group(str(target_zarr), mode="r", use_consolidated=False)
    parent = reopened["refined_subject_masks_runs"]
    run = parent["refined_recording_draft"]
    assert run.attrs["stage_selector_eligible"] is False
    assert parent.attrs.get("latest") != "refined_recording_draft"
    assert parent.attrs.get("latest_complete") != "refined_recording_draft"
    assert (
        parent.attrs.get("refined_subject_mask_review_status_latest")
        != "refined_recording_draft"
    )
    binding = run.attrs["subject_mask_recording_source_receipt_binding"]
    receipt_path = target_zarr / str(binding["relative_path"])
    assert receipt_path.is_file()
    assert (
        hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        == binding["document_sha256"]
    )
    assert (
        run.attrs["run_manifest"]["payload_digest"]
        == result["source_manifest_payload_digest"]
    )


def test_import_refined_subject_mask_clip_packages_rejects_duplicate_source_crop_rows(
    tmp_path: Path,
) -> None:
    target_zarr = tmp_path / "target.zarr"
    zarr.open_group(str(target_zarr), mode="w")
    package_a = _write_package(
        tmp_path,
        package_name="clip_a",
        run_name="refined_clip_a",
        crop_row_ids=[10, 11],
    )
    package_b = _write_package(
        tmp_path,
        package_name="clip_b",
        run_name="refined_clip_b",
        crop_row_ids=[11, 12],
    )

    with pytest.raises(ValueError, match="Duplicate source_crop_row_ids"):
        import_refined_subject_mask_clip_packages(
            zarr_path=target_zarr,
            package_paths=[package_a, package_b],
            output_run="refined_collection",
        )


def test_import_refined_subject_mask_clip_packages_rejects_mask_label_mismatch(
    tmp_path: Path,
) -> None:
    target_zarr = tmp_path / "target.zarr"
    zarr.open_group(str(target_zarr), mode="w")
    package_a = _write_package(
        tmp_path,
        package_name="clip_a",
        run_name="refined_clip_a",
        crop_row_ids=[10],
    )
    package_b = _write_package(
        tmp_path,
        package_name="clip_b",
        run_name="refined_clip_b",
        crop_row_ids=[11],
        labels=["subject_body", "eye_right"],
    )

    with pytest.raises(ValueError, match="mask_labels"):
        import_refined_subject_mask_clip_packages(
            zarr_path=target_zarr,
            package_paths=[package_a, package_b],
            output_run="refined_collection",
        )


def test_import_refined_subject_mask_clip_packages_requires_recording_frame_authority(
    tmp_path: Path,
) -> None:
    target_zarr = tmp_path / "target.zarr"
    root = zarr.open_group(str(target_zarr), mode="w")
    root.require_group("crop_runs").create_group("crop_proxy_collection")
    package = _write_package(
        tmp_path,
        package_name="clip_a",
        run_name="refined_clip_a",
        crop_row_ids=[10],
    )

    with pytest.raises(
        ValueError, match="Cannot establish the recording frame universe"
    ):
        import_refined_subject_mask_clip_packages(
            zarr_path=target_zarr,
            package_paths=[package],
            output_run="refined_collection",
            expected_target_crop_run="crop_proxy_collection",
        )


def test_import_refined_subject_mask_clip_packages_regenerates_reason_bytes(
    tmp_path: Path,
) -> None:
    target_zarr = tmp_path / "target.zarr"
    zarr.open_group(str(target_zarr), mode="w")
    _write_target_crop_contract(target_zarr)
    package_a = _write_package(
        tmp_path,
        package_name="clip_a",
        run_name="refined_clip_a",
        crop_row_ids=[10],
        body_reason_labels=["short"],
    )
    package_b = _write_package(
        tmp_path,
        package_name="clip_b",
        run_name="refined_clip_b",
        crop_row_ids=[11],
        body_reason_labels=[
            "different_reason_label_that_requires_a_wider_reason_bytes_matrix"
        ],
    )

    import_refined_subject_mask_clip_packages(
        zarr_path=target_zarr,
        package_paths=[package_a, package_b],
        output_run="refined_collection",
        array_copy_workers=2,
    )

    root = zarr.open_group(str(target_zarr), mode="r", use_consolidated=False)
    body = root["refined_subject_masks_runs"]["refined_collection"]["components"][
        "subject_body"
    ]
    assert body["reason_bytes"].shape[0] == 2
    assert body["reason_bytes"].shape[1] > 64
    assert read_reason_labels(body).tolist() == [
        "short",
        "different_reason_label_that_requires_a_wider_reason_bytes_matrix",
    ]


def test_repair_refined_subject_mask_frame_counts_uses_atomic_swap(
    tmp_path: Path,
) -> None:
    target_zarr = tmp_path / "target.zarr"
    root = zarr.open_group(str(target_zarr), mode="w")
    root.attrs["recording_frame_index_row_count"] = 6
    root.require_group("crop_runs").create_group("crop_proxy_collection")
    parent = root.require_group("refined_subject_masks_runs")
    run = parent.create_group("refined_collection")
    run.attrs["palette_run_completion_status"] = "complete"
    run.attrs["source_crop_run"] = "crop_proxy_collection"
    run.create_array("frame_indices", data=np.asarray([0, 2, 2, 5], dtype=np.int64))
    run.create_array("frame_counts", data=np.asarray([1, 0, 2], dtype=np.int32))

    dry_run = repair_frame_counts(
        zarr_path=target_zarr,
        run_name="refined_collection",
        execute=False,
    )
    assert dry_run["status"] == "would_repair"
    assert run["frame_counts"].shape == (3,)

    receipt = tmp_path / "repair.json"
    result = repair_frame_counts(
        zarr_path=target_zarr,
        run_name="refined_collection",
        execute=True,
        receipt_path=receipt,
    )

    repaired = zarr.open_group(str(target_zarr), mode="r", use_consolidated=False)[
        "refined_subject_masks_runs"
    ]["refined_collection"]
    assert result["status"] == "repaired"
    assert receipt.is_file()
    assert repaired.attrs["palette_run_completion_status"] == "complete"
    np.testing.assert_array_equal(
        repaired["frame_counts"][:],
        np.asarray([1, 0, 2, 0, 0, 1], dtype=np.int32),
    )
    assert not any(
        str(name).startswith("frame_counts_repair_") for name in repaired.keys()
    )
