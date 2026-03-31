from __future__ import annotations

import os

import numpy as np
import zarr

from fisheye.tune import refined_subject_mask_review as review_mod
from fisheye.visualization import subject_mask_inspector as mod

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")


def _build_inspector_root():
    root = zarr.group()

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
    subject.create_array("detection_source", data=np.zeros((2,), dtype=np.int8))
    subject.create_array("frame_indices", data=np.asarray([10, 11], dtype=np.int32))
    subject.create_array("detection_indices", data=np.asarray([0, 1], dtype=np.int32))
    subject.create_array("frame_counts", data=np.asarray([1, 1], dtype=np.int32))
    subject.create_array("available_channels", data=np.asarray([True, False, False, True], dtype=np.bool_))

    masks = np.zeros((2, 4, 8, 8), dtype=np.uint8)
    masks[0, 0, 1:7, 1:7] = 1
    masks[1, 0, 2:6, 2:6] = 1
    masks[0, 3, 4:6, 4:6] = 1
    subject.create_array("masks_roi", data=masks)

    metrics = subject.create_group("metrics")
    mask_present, area_px = review_mod._compute_mask_metrics(masks)
    geometry = review_mod._compute_geometry_metrics(masks)
    metrics.create_array("mask_present", data=mask_present)
    metrics.create_array("area_px", data=area_px)
    metrics.create_array("centroid_xy", data=geometry["centroid_xy"])
    metrics.create_array("centroid_valid", data=geometry["centroid_valid"])
    metrics.create_array("bbox_xyxy", data=geometry["bbox_xyxy"])
    metrics.create_array("bbox_valid", data=geometry["bbox_valid"])

    _source, refined = review_mod.prepare_refined_subject_run(
        root,
        subject_run="subject_masks_001",
        refined_run="refined_subject_masks_001",
        components=("subject_body", "swim_bladder"),
    )

    edited = np.asarray(refined.group["masks_roi"][0], dtype=np.uint8)
    edited[0, 1:7, 1:7] = 1
    edited[0, 3:5, 3:5] = 0
    review_mod.save_refined_subject_roi(
        source=_source,
        refined=refined,
        roi_idx=0,
        edited_masks=edited,
    )
    return root


def test_load_runs_infers_matching_latest_refined_run() -> None:
    root = _build_inspector_root()

    subject, refined = mod._load_runs(root, subject_run="subject_masks_001", refined_run=None)

    assert subject is not None
    assert refined is not None
    assert subject.run_name == "subject_masks_001"
    assert refined.run_name == "refined_subject_masks_001"
    assert refined.source_subject_mask_run == "subject_masks_001"


def test_stage_summary_lines_include_raw_and_refined_metrics() -> None:
    root = _build_inspector_root()
    subject, refined = mod._load_runs(root, subject_run="subject_masks_001", refined_run="refined_subject_masks_001")

    assert subject is not None
    assert refined is not None

    raw_lines = mod._stage_summary_lines(subject, "subject_body", 0)
    refined_lines = mod._stage_summary_lines(refined, "subject_body", 0)

    assert raw_lines[0] == "subject_mask_runs/subject_masks_001"
    assert any("area_px=36.0" in line for line in raw_lines)
    assert any("bbox=[1.0,1.0,6.0,6.0]" in line for line in raw_lines)
    assert refined_lines[0] == "refined_subject_masks_runs/refined_subject_masks_001"
    assert any("review=" in line for line in refined_lines)
    assert any("sigma_noise=" in line for line in refined_lines)
    assert any("ipr=" in line for line in refined_lines)
    assert any("solidity=" in line for line in refined_lines)


def test_component_geometry_reads_run_level_bbox_and_centroid() -> None:
    root = _build_inspector_root()
    subject, _refined = mod._load_runs(root, subject_run="subject_masks_001", refined_run="refined_subject_masks_001")

    assert subject is not None
    centroid_xy, centroid_valid, bbox_xyxy, bbox_valid = mod._component_geometry(subject, "subject_body", 0)

    np.testing.assert_allclose(centroid_xy, np.asarray([3.5, 3.5], dtype=np.float32))
    assert centroid_valid is True
    np.testing.assert_allclose(bbox_xyxy, np.asarray([1.0, 1.0, 6.0, 6.0], dtype=np.float32))
    assert bbox_valid is True


def test_component_flag_reasons_and_flagged_indices_use_refined_qc() -> None:
    root = _build_inspector_root()
    _subject, refined = mod._load_runs(root, subject_run="subject_masks_001", refined_run="refined_subject_masks_001")

    assert refined is not None
    reasons = mod._component_flag_reasons(
        refined,
        "subject_body",
        0,
        thresholds=mod.InspectorThresholds(),
    )
    flagged = mod._flagged_roi_indices(
        refined,
        "subject_body",
        thresholds=mod.InspectorThresholds(),
    )

    assert "hole_count>0" in reasons
    assert 0 in flagged


def test_component_names_union_available_components() -> None:
    root = _build_inspector_root()
    subject, refined = mod._load_runs(root, subject_run="subject_masks_001", refined_run="refined_subject_masks_001")

    component_names = mod._component_names(subject, refined)

    assert component_names == ("subject_body", "swim_bladder")
