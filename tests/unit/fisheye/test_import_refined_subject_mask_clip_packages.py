from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import tarfile
import time

import numpy as np
import pytest
import zarr

from fisheye.shared.subject_mask_chunks import refined_subject_mask_storage_chunks
from fisheye.shared.detect_reason_codec import read_reason_labels, write_reason_columns
from fisheye.shared.refined_subject_mask_encoded_chunks import prepare_global_mask_chunk_grid
from fisheye.shared.zarr_run_completion import RUN_COMPLETION_STATUS_ATTR
from fisheye.utils.convert_refined_subject_mask_clip_package_v2 import convert_package
from fisheye.utils.finalize_subject_mask_clip_package import PACKAGE_SCHEMA_ID
from fisheye.utils.import_refined_subject_mask_clip_packages import (
    ClipPackage,
    _build_row_chunk_copy_plan,
    import_refined_subject_mask_clip_packages,
)
from fisheye.utils.validate_encoded_subject_mask_import_canary import validate_canary


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
    run.attrs["source_roi_cache_canonical_path"] = f"/nrs/{package_name}.flat_roi_cache.json"
    run.attrs["source_roi_cache_key"] = f"{package_name}_cache_key"
    run.attrs["source_subject_mask_shard_runs"] = [f"subject_masks_{run_name}"]
    run.attrs["source_subject_mask_shard_run_paths"] = [f"subject_mask_shard_runs/subject_masks_{run_name}"]
    run.attrs["source_subject_mask_shard_crop_runs"] = [f"crop_{run_name}"]

    row_count = len(crop_row_ids)
    masks = np.zeros((row_count, len(labels), 2, 3), dtype=np.uint8)
    for row_idx, crop_row_id in enumerate(crop_row_ids):
        masks[row_idx, :, :, :] = np.uint8(crop_row_id)
    run.create_array(
        "masks_roi",
        data=masks,
        chunks=(mask_row_chunk or max(1, row_count), 1, 2, 3),
        overwrite=True,
    )
    run.create_array("source_crop_row_ids", data=np.asarray(crop_row_ids, dtype=np.int64), overwrite=True)
    run.create_array("frame_indices", data=np.asarray(crop_row_ids, dtype=np.int64) + 1000, overwrite=True)
    run.create_array("available_channels", data=np.ones((row_count, len(labels)), dtype=bool), overwrite=True)

    metrics = run.require_group("metrics")
    metric_values = np.stack(
        [
            np.asarray(crop_row_ids, dtype=np.float32),
            np.asarray(crop_row_ids, dtype=np.float32) + 0.5,
        ],
        axis=1,
    )
    metrics.create_array("area_px", data=metric_values[:, : len(labels)], chunks=(max(1, row_count), len(labels)))

    components = run.require_group("components")
    body = components.require_group("subject_body")
    body.create_array("manual_override", data=np.zeros((row_count,), dtype=bool), chunks=(max(1, row_count),))
    if body_reason_labels is not None:
        write_reason_columns(
            body,
            np.asarray(body_reason_labels, dtype=object),
            chunk_size=max(1, row_count),
            include_reason_text=True,
            overwrite=True,
        )
    contours = body.require_group("contours")
    ptr = np.arange(row_count, dtype=np.int64)
    length = np.ones((row_count,), dtype=np.int32)
    points_xy = np.asarray([[float(crop_row_id), float(crop_row_id) + 0.25] for crop_row_id in crop_row_ids], dtype=np.float32)
    contours.create_array("ptr", data=ptr, chunks=(max(1, row_count),))
    contours.create_array("len", data=length, chunks=(max(1, row_count),))
    contours.create_array("points_xy", data=points_xy, chunks=(max(1, row_count), 2))

    package_path = tmp_path / f"{package_name}.tar.gz"
    manifest = {
        "schema_id": PACKAGE_SCHEMA_ID,
        "created_at_utc": "2026-07-08T00:00:00+00:00",
        "run_group_path": f"refined_subject_masks_runs/{run_name}",
        "summary": {"rows_total": row_count},
    }
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
        package_b_path: (np.arange(4, dtype=np.int64), np.arange(6, 10, dtype=np.int64)),
    }

    plan = _build_row_chunk_copy_plan(
        [package_a, package_b],
        row_maps=row_maps,
        total_rows=10,
        row_chunk=4,
    )

    assert [(chunk.dst_start, chunk.dst_stop) for chunk in plan] == [(0, 4), (4, 8), (8, 10)]
    assert [
        [(run.package_path, run.src_start, run.src_stop, run.dst_start, run.dst_stop) for run in chunk.runs]
        for chunk in plan
    ] == [
        [(package_a_path, 0, 4, 0, 4)],
        [
            (package_a_path, 4, 6, 4, 6),
            (package_b_path, 0, 2, 6, 8),
        ],
        [(package_b_path, 2, 4, 8, 10)],
    ]


def test_encoded_v2_packages_copy_complete_chunks_and_decode_only_boundary(tmp_path: Path) -> None:
    target_zarr = tmp_path / "target.zarr"
    root = zarr.open_group(str(target_zarr), mode="w")
    crop = root.require_group("crop_runs").create_group("crop_proxy_collection")
    crop.create_array(
        "source_crop_row_ids",
        data=np.arange(10, 20, dtype=np.int64),
        chunks=(4,),
        overwrite=True,
    )
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
    np.testing.assert_allclose(run["metrics"]["area_px"][:], baseline["metrics"]["area_px"][:])
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


def test_import_refined_subject_mask_clip_packages_merges_rows_and_contours(tmp_path: Path) -> None:
    target_zarr = tmp_path / "target.zarr"
    zarr.open_group(str(target_zarr), mode="w")
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
    assert parent.attrs["refined_subject_mask_review_status_latest"] == "refined_collection"
    assert run.attrs[RUN_COMPLETION_STATUS_ATTR] == "complete"
    assert run.attrs["mask_storage_authority"] == "masks_roi"
    assert run.attrs["editable_mask_surface"] == "masks_roi"
    assert run.attrs["masks_roi_materialized"] is True
    assert run.attrs["mask_bitpacked_materialized"] is False
    assert run.attrs["mask_rle_materialized"] is False
    assert run.attrs["derived_mask_caches_stale"] is False
    assert run.attrs["mask_store_contract_encodings"] == ["dense_uint8_v1"]

    np.testing.assert_array_equal(run["source_crop_row_ids"][:], np.asarray([10, 11, 12], dtype=np.int64))
    np.testing.assert_array_equal(run["frame_indices"][:], np.asarray([1010, 1011, 1012], dtype=np.int64))
    assert tuple(int(value) for value in run["masks_roi"].chunks) == refined_subject_mask_storage_chunks(3, 2, 3)
    assert run["masks_roi"].shape == (3, 2, 2, 3)
    np.testing.assert_array_equal(run["masks_roi"][:, 0, 0, 0], np.asarray([10, 11, 12], dtype=np.uint8))
    np.testing.assert_allclose(run["metrics"]["area_px"][:, 0], np.asarray([10.0, 11.0, 12.0], dtype=np.float32))

    contours = run["components"]["subject_body"]["contours"]
    np.testing.assert_array_equal(contours["ptr"][:], np.asarray([0, 1, 2], dtype=np.int64))
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
    assert run.attrs["source_roi_cache_keys"] == ["clip_a_cache_key", "clip_b_cache_key"]
    assert run.attrs["source_subject_mask_shard_runs"] == [
        "subject_masks_refined_clip_a",
        "subject_masks_refined_clip_b",
    ]


def test_import_refined_subject_mask_clip_packages_rejects_duplicate_source_crop_rows(tmp_path: Path) -> None:
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


def test_import_refined_subject_mask_clip_packages_rejects_mask_label_mismatch(tmp_path: Path) -> None:
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


def test_import_refined_subject_mask_clip_packages_regenerates_reason_bytes(tmp_path: Path) -> None:
    target_zarr = tmp_path / "target.zarr"
    zarr.open_group(str(target_zarr), mode="w")
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
        body_reason_labels=["different_reason_label_that_requires_a_wider_reason_bytes_matrix"],
    )

    import_refined_subject_mask_clip_packages(
        zarr_path=target_zarr,
        package_paths=[package_a, package_b],
        output_run="refined_collection",
        array_copy_workers=2,
    )

    root = zarr.open_group(str(target_zarr), mode="r", use_consolidated=False)
    body = root["refined_subject_masks_runs"]["refined_collection"]["components"]["subject_body"]
    assert body["reason_bytes"].shape[0] == 2
    assert body["reason_bytes"].shape[1] > 64
    assert read_reason_labels(body).tolist() == [
        "short",
        "different_reason_label_that_requires_a_wider_reason_bytes_matrix",
    ]
