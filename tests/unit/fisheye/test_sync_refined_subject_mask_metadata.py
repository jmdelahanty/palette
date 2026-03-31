from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import zarr

from fisheye.shared.detect_reason_codec import read_reason_labels
from fisheye.tune import refined_subject_mask_review as review_mod
from fisheye.utils import sync_refined_subject_mask_metadata as cli_mod

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


def test_sync_refined_subject_mask_metadata_updates_touched_component(tmp_path: Path) -> None:
    zarr_path = tmp_path / "subject_review.zarr"
    _build_subject_review_archive(zarr_path)

    root = zarr.open_group(str(zarr_path), mode="a")
    _source, refined = review_mod.prepare_refined_subject_run(
        root,
        subject_run="subject_masks_001",
        refined_run="refined_subject_masks_001",
        components=("subject_body", "swim_bladder"),
    )

    edited = np.asarray(refined.group["masks_roi"][0], dtype=np.uint8)
    edited[0, 1:7, 1:7] = 0
    refined.group["masks_roi"][0] = edited

    summary = review_mod.sync_refined_subject_mask_metadata(
        zarr_path,
        refined_run="refined_subject_masks_001",
        component_name="subject_body",
        roi_indices=[0],
    )

    assert summary["status"] == "updated"
    assert summary["component_name"] == "subject_body"
    assert summary["source_subject_mask_run"] == "subject_masks_001"
    assert summary["roi_indices"] == [0]
    assert summary["roi_count"] == 1
    assert summary["changed_roi_count"] == 1
    assert summary["noop_roi_count"] == 0
    assert summary["updated_at_utc"]

    run = root["refined_subject_masks_runs"]["refined_subject_masks_001"]
    np.testing.assert_array_equal(
        np.asarray(run["metrics/mask_present"][0], dtype=bool),
        np.asarray([False, False], dtype=bool),
    )
    np.testing.assert_allclose(
        np.asarray(run["metrics/area_px"][0], dtype=np.float32),
        np.asarray([0.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(run["metrics/centroid_xy"][0], dtype=np.float32),
        np.asarray([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(run["metrics/centroid_valid"][0], dtype=bool),
        np.asarray([False, False], dtype=bool),
    )
    np.testing.assert_allclose(
        np.asarray(run["metrics/bbox_xyxy"][0], dtype=np.float32),
        np.asarray([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(run["metrics/bbox_valid"][0], dtype=bool),
        np.asarray([False, False], dtype=bool),
    )
    np.testing.assert_array_equal(
        np.asarray(run["edit_applied"][0], dtype=bool),
        np.asarray([True, False], dtype=bool),
    )

    body_group = run["components/subject_body"]
    swim_group = run["components/swim_bladder"]
    assert bool(np.asarray(body_group["edit_applied"][0], dtype=bool)) is True
    assert bool(np.asarray(swim_group["edit_applied"][0], dtype=bool)) is False
    body_reasons = read_reason_labels(body_group)
    swim_reasons = read_reason_labels(swim_group)
    assert body_reasons is not None
    assert swim_reasons is not None
    assert body_reasons[0] == "manual_correction"
    assert swim_reasons[0] == "clean"
    assert int(np.asarray(body_group["metrics/component_count"][0], dtype=np.int32)) == 0
    assert float(np.asarray(body_group["metrics/largest_component_fraction"][0], dtype=np.float32)) == 0.0
    assert int(np.asarray(body_group["metrics/hole_count"][0], dtype=np.int32)) == 0
    assert float(np.asarray(body_group["metrics/hole_area_fraction"][0], dtype=np.float32)) == 0.0
    assert float(np.asarray(body_group["metrics/sigma_noise"][0], dtype=np.float32)) == 0.0
    assert float(np.asarray(body_group["metrics/curvature_var"][0], dtype=np.float32)) == 0.0
    assert float(np.asarray(body_group["metrics/ipr"][0], dtype=np.float32)) == 0.0
    assert float(np.asarray(body_group["metrics/solidity"][0], dtype=np.float32)) == 0.0


def test_sync_refined_subject_mask_metadata_batches_rows_and_tracks_noops(tmp_path: Path) -> None:
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

    edited1 = np.asarray(refined.group["masks_roi"][1], dtype=np.uint8)
    refined.group["masks_roi"][1] = edited1

    summary = review_mod.sync_refined_subject_mask_metadata(
        zarr_path,
        refined_run="refined_subject_masks_001",
        component_name="subject_body",
        roi_indices=[0, 1],
    )

    assert summary["status"] == "updated"
    assert summary["roi_indices"] == [0, 1]
    assert summary["roi_count"] == 2
    assert summary["changed_roi_count"] == 1
    assert summary["noop_roi_count"] == 1

    run = root["refined_subject_masks_runs"]["refined_subject_masks_001"]
    body_group = run["components/subject_body"]
    body_reasons = read_reason_labels(body_group)
    assert body_reasons is not None
    assert body_reasons.tolist() == ["manual_correction", "copied_from_source"]


def test_sync_refined_subject_mask_metadata_cli_emits_json_summary(tmp_path: Path, capsys) -> None:
    zarr_path = tmp_path / "subject_review.zarr"
    _build_subject_review_archive(zarr_path)

    root = zarr.open_group(str(zarr_path), mode="a")
    _source, refined = review_mod.prepare_refined_subject_run(
        root,
        subject_run="subject_masks_001",
        refined_run="refined_subject_masks_001",
        components=("subject_body", "swim_bladder"),
    )

    edited = np.asarray(refined.group["masks_roi"][0], dtype=np.uint8)
    edited[0, 1:7, 1:7] = 0
    refined.group["masks_roi"][0] = edited

    rc = cli_mod.main(
        [
            "--zarr-path",
            str(zarr_path),
            "--refined-run",
            "refined_subject_masks_001",
            "--component-name",
            "subject_body",
            "--dataset",
            "labels/refined_subject_masks/refined_subject_masks_001/subject_body",
            "--roi-indices",
            "0,0",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["status"] == "updated"
    assert payload["dataset"] == "labels/refined_subject_masks/refined_subject_masks_001/subject_body"
    assert payload["roi_indices"] == [0]
    assert payload["changed_roi_count"] == 1
    assert payload["noop_roi_count"] == 0
