from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.refinement import refine_subject_masks as mod
from fisheye.shared.detect_reason_codec import read_reason_labels
from fisheye.shared.mask_store import write_component_rle_mask_store_from_dense
from fisheye.tune import refined_subject_mask_review as review_mod

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")


def _build_subject_review_archive(zarr_path: Path) -> None:
    root = zarr.open_group(str(zarr_path), mode="w")

    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "crop_001"
    crop = crop_parent.create_group("crop_001")
    roi_images = np.zeros((2, 8, 8), dtype=np.uint8)
    roi_images[0, 1:7, 1:7] = 70
    roi_images[1, 2:6, 2:6] = 120
    crop.create_array("roi_images", data=roi_images)

    subject_parent = root.create_group("subject_mask_runs")
    subject_parent.attrs["latest"] = "subject_masks_001"
    subject = subject_parent.create_group("subject_masks_001")
    subject.attrs["source_crop_run"] = "crop_001"
    subject.attrs["method"] = "subject_mask_threshold_lr_v1"
    subject.attrs["mask_labels"] = ["subject_body", "eye_left", "eye_right", "swim_bladder"]
    subject.attrs["label_schema_id"] = "subject_v1_lr"
    subject.attrs["source_keypoint_group"] = "refined_keypoints_runs"
    subject.attrs["source_keypoints_run"] = "refined_kp_001"
    subject.create_array("detection_source", data=np.zeros((2,), dtype=np.int8))
    subject.create_array("frame_indices", data=np.asarray([10, 11], dtype=np.int32))
    subject.create_array("detection_indices", data=np.asarray([0, 1], dtype=np.int32))
    subject.create_array("frame_counts", data=np.asarray([1, 1], dtype=np.int32))
    subject.create_array("available_channels", data=np.asarray([True, True, True, False], dtype=np.bool_))

    masks = np.zeros((2, 4, 8, 8), dtype=np.uint8)
    masks[0, 0, 1:7, 1:7] = 1
    masks[1, 0, 2:6, 2:6] = 1
    subject.create_array("masks_roi", data=masks)

    metrics = subject.create_group("metrics")
    metrics.create_array(
        "mask_present",
        data=np.asarray(
            [
                [True, False, False, False],
                [True, False, False, False],
            ],
            dtype=np.bool_,
        ),
    )


def test_refine_subject_masks_batches_driver_writeback_and_records_scheduler_attrs(tmp_path: Path) -> None:
    zarr_path = tmp_path / "subject_review.zarr"
    _build_subject_review_archive(zarr_path)

    root = zarr.open_group(str(zarr_path), mode="a")
    _source, refined = review_mod.prepare_refined_subject_run(
        root,
        subject_run="subject_masks_001",
        refined_run="refined_subject_masks_001",
        components=("subject_body", "swim_bladder"),
    )
    write_component_rle_mask_store_from_dense(
        refined.group,
        refined.group["masks_roi"],
        component_names=refined.component_names,
        encode_row_chunk_size=1,
    )

    edited0 = np.asarray(refined.group["masks_roi"][0], dtype=np.uint8)
    edited0[0, 1:7, 1:7] = 0
    refined.group["masks_roi"][0] = edited0

    summary = mod.refine_subject_masks(
        zarr_path,
        refined_run="refined_subject_masks_001",
        components=("subject_body",),
        roi_indices=[0, 1],
        chunk_size=1,
        scheduler="single-threaded",
    )

    assert summary["status"] == "updated"
    assert summary["refined_run"] == "refined_subject_masks_001"
    assert summary["source_subject_mask_run"] == "subject_masks_001"
    assert summary["component_names"] == ["subject_body"]
    assert summary["roi_indices"] == [0, 1]
    assert summary["roi_count"] == 2
    assert summary["chunk_count"] == 2
    assert summary["changed_roi_count"] == 1
    assert summary["noop_roi_count"] == 1
    assert summary["scheduler"] == "single-threaded"
    assert summary["chunk_size"] == 1
    assert summary["updated_at_utc"]

    run = root["refined_subject_masks_runs"]["refined_subject_masks_001"]
    assert run.attrs["dask_scheduler"] == "single-threaded"
    assert run.attrs["dask_num_workers"] is None
    assert int(run.attrs["dask_chunk_size"]) == 1
    assert bool(run.attrs["mask_rle_stale"]) is True
    assert run.attrs["mask_rle_stale_reason"] == "batch_refined_subject_mask_edit"
    assert run.attrs["mask_rle_stale_component_names"] == ["subject_body"]
    assert run.attrs["mask_rle_stale_row_count"] == 2
    assert run.attrs["mask_rle_stale_row_min"] == 0
    assert run.attrs["mask_rle_stale_row_max"] == 1

    body_group = run["components/subject_body"]
    swim_group = run["components/swim_bladder"]
    body_provenance = body_group["provenance"]
    swim_provenance = swim_group["provenance"]
    body_reasons = read_reason_labels(body_group)
    swim_reasons = read_reason_labels(swim_group)
    assert body_reasons is not None
    assert swim_reasons is not None
    assert body_reasons.tolist() == ["manual_correction", "copied_from_source"]
    assert swim_reasons.tolist() == ["clean", "clean"]
    np.testing.assert_array_equal(
        np.asarray(run["edit_applied"][:], dtype=bool),
        np.asarray(
            [
                [True, False],
                [False, False],
            ],
            dtype=bool,
        ),
    )
    assert body_provenance.attrs["last_update_stage"] == review_mod.REFINED_SUBJECT_STAGE_NAME
    assert body_provenance.attrs["last_update_mode"] == "batch"
    assert body_provenance.attrs["last_update_method"] == "fisheye.refinement.refine_subject_masks"
    assert body_provenance.attrs["updated_at_utc"] == run.attrs["updated_at_utc"]
    assert swim_provenance.attrs["last_update_mode"] == "create"


def test_refine_subject_masks_cli_emits_json_summary(tmp_path: Path, capsys) -> None:
    zarr_path = tmp_path / "subject_review.zarr"
    _build_subject_review_archive(zarr_path)

    root = zarr.open_group(str(zarr_path), mode="a")
    _source, refined = review_mod.prepare_refined_subject_run(
        root,
        subject_run="subject_masks_001",
        refined_run="refined_subject_masks_001",
        components=("subject_body", "swim_bladder"),
    )

    edited0 = np.asarray(refined.group["masks_roi"][0], dtype=np.uint8)
    edited0[0, 1:7, 1:7] = 0
    refined.group["masks_roi"][0] = edited0

    rc = mod.main(
        [
            str(zarr_path),
            "--refined-run",
            "refined_subject_masks_001",
            "--components",
            "subject_body",
            "--roi-indices",
            "0,1",
            "--chunk-size",
            "1",
            "--scheduler",
            "single-threaded",
            "--json",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["status"] == "updated"
    assert payload["component_names"] == ["subject_body"]
    assert payload["roi_indices"] == [0, 1]
    assert payload["changed_roi_count"] == 1
    assert payload["noop_roi_count"] == 1


def test_refine_subject_masks_cli_dry_run_supports_aliases_and_ranges(tmp_path: Path, capsys) -> None:
    zarr_path = tmp_path / "subject_review.zarr"
    _build_subject_review_archive(zarr_path)

    rc = mod.main(
        [
            str(zarr_path),
            "--source-run",
            "subject_masks_001",
            "--run-name",
            "refined_subject_masks_preview",
            "--component",
            "subject_body",
            "--roi-index",
            "0",
            "--roi-indices",
            "1-1",
            "--scheduler",
            "single_thread",
            "--dry-run",
            "--json",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["status"] == "planned"
    assert payload["source_subject_mask_run"] == "subject_masks_001"
    assert payload["refined_run"] == "refined_subject_masks_preview"
    assert payload["refined_run_selection"] == "explicit"
    assert payload["refined_run_exists"] is False
    assert payload["would_create_refined_run"] is True
    assert payload["mutates_archive"] is False
    assert payload["component_names"] == ["subject_body"]
    assert payload["available_component_names"] == ["subject_body", "eye_left", "eye_right", "swim_bladder"]
    assert payload["roi_indices"] == [0, 1]
    assert payload["roi_count"] == 2
    assert payload["total_roi_count"] == 2
    assert payload["scheduler"] == "single-threaded"

    root = zarr.open_group(str(zarr_path), mode="r")
    assert "refined_subject_masks_runs" not in root


def test_refine_subject_masks_cli_dry_run_reports_existing_target(tmp_path: Path, capsys) -> None:
    zarr_path = tmp_path / "subject_review.zarr"
    _build_subject_review_archive(zarr_path)

    root = zarr.open_group(str(zarr_path), mode="a")
    review_mod.prepare_refined_subject_run(
        root,
        subject_run="subject_masks_001",
        refined_run="refined_subject_masks_001",
        components=("subject_body", "swim_bladder"),
    )
    updated_before = str(root["refined_subject_masks_runs"]["refined_subject_masks_001"].attrs.get("updated_at_utc") or "")

    rc = mod.main(
        [
            str(zarr_path),
            "--run-name",
            "refined_subject_masks_001",
            "--dry-run",
            "--json",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["status"] == "planned"
    assert payload["refined_run"] == "refined_subject_masks_001"
    assert payload["refined_run_selection"] == "explicit"
    assert payload["refined_run_exists"] is True
    assert payload["would_create_refined_run"] is False
    assert payload["available_component_names"] == ["subject_body", "swim_bladder"]
    assert payload["roi_indices"] == [0, 1]

    root_after = zarr.open_group(str(zarr_path), mode="r")
    updated_after = str(root_after["refined_subject_masks_runs"]["refined_subject_masks_001"].attrs.get("updated_at_utc") or "")
    assert updated_after == updated_before


def test_refine_subject_masks_dry_run_inspects_compact_existing_target_without_materializing_dense(
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "subject_review_compact.zarr"
    _build_subject_review_archive(zarr_path)

    root = zarr.open_group(str(zarr_path), mode="a")
    _source, refined = review_mod.prepare_refined_subject_run(
        root,
        subject_run="subject_masks_001",
        refined_run="refined_subject_masks_001",
        components=("subject_body", "swim_bladder"),
    )
    run = refined.group
    dense = np.asarray(run["masks_roi"][:], dtype=np.uint8)
    del run["masks_roi"]
    write_component_rle_mask_store_from_dense(
        run,
        dense,
        component_names=list(refined.component_names),
        encode_row_chunk_size=1,
    )
    assert "masks_roi" not in run

    summary = mod.refine_subject_masks(
        zarr_path,
        refined_run="refined_subject_masks_001",
        components=("subject_body",),
        roi_indices=[0, 1],
        chunk_size=1,
        scheduler="single-threaded",
        dry_run=True,
    )

    assert summary["status"] == "planned"
    assert summary["refined_run_exists"] is True
    assert summary["would_create_refined_run"] is False
    assert summary["mutates_archive"] is False
    assert summary["component_names"] == ["subject_body"]
    assert summary["roi_indices"] == [0, 1]
    assert summary["total_roi_count"] == 2
    assert summary["refined_mask_store_encoding"] == "component_rle_v1"
    assert summary["refined_mask_storage_surface"] == "mask_rle"
    assert summary["refined_mask_store_path"] == "refined_subject_masks_runs/refined_subject_masks_001/mask_rle"

    root_after = zarr.open_group(str(zarr_path), mode="r")
    run_after = root_after["refined_subject_masks_runs/refined_subject_masks_001"]
    assert "masks_roi" not in run_after
    assert "mask_rle" in run_after


def test_parse_scheduler_arg_rejects_invalid_value() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="Invalid scheduler"):
        mod._parse_scheduler_arg("bogus")
