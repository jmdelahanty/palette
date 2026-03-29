from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import numpy as np
import zarr

from fisheye.shared.detect_reason_codec import read_reason_labels
from fisheye.tune import refined_subject_mask_review as mod


def _build_subject_review_root():
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
    masks[0, 1, 2:3, 2:3] = 1
    masks[0, 2, 2:3, 4:5] = 1
    subject.create_array("masks_roi", data=masks)

    metrics = subject.create_group("metrics")
    metrics.create_array(
        "mask_present",
        data=np.asarray(
            [
                [True, True, True, False],
                [True, False, False, False],
            ],
            dtype=np.bool_,
        ),
    )

    return root


def test_prepare_refined_subject_run_creates_body_swim_editor_run(monkeypatch) -> None:
    root = _build_subject_review_root()
    monkeypatch.setattr(
        mod,
        "get_git_info",
        lambda repo_path=None: {  # noqa: ARG005
            "commit_hash": "b" * 40,
            "short_hash": "bbbbbbbb",
            "branch": "main",
            "is_dirty": False,
            "remote_url": "git@example.com:palette.git",
        },
    )
    monkeypatch.setattr(
        mod,
        "get_environment_info",
        lambda **kwargs: {  # noqa: ARG005
            "environment": {"python": "3.12"},
            "platform": {
                "hostname": "review-host",
                "system": "Linux",
                "release": "6.8",
                "python_version": "3.12.0",
                "machine": "x86_64",
            },
        },
    )
    monkeypatch.setattr(
        mod.sys,
        "argv",
        ["scripts/py", "-m", "fisheye.tune.refined_subject_mask_review"],
    )

    source, refined = mod.prepare_refined_subject_run(
        root,
        subject_run="subject_masks_001",
        refined_run="refined_subject_masks_001",
        components=("subject_body", "swim_bladder"),
    )

    assert source.run_name == "subject_masks_001"
    assert refined.run_name == "refined_subject_masks_001"
    assert refined.component_names == ("subject_body", "swim_bladder")
    assert root["refined_subject_masks_runs"].attrs["latest"] == "refined_subject_masks_001"

    run = refined.group
    assert run.attrs["source_subject_mask_run"] == "subject_masks_001"
    assert run.attrs["source_crop_run"] == "crop_001"
    assert run.attrs["source_keypoint_group"] == "refined_keypoints_runs"
    assert run.attrs["source_keypoints_run"] == "refined_kp_001"
    assert run.attrs["source_keypoint_run"] == "refined_kp_001"
    assert run.attrs["label_schema_id"] == "refined_subject_v1_body_swim"
    assert run.attrs["git_commit"] == "b" * 40

    masks = np.asarray(run["masks_roi"][:], dtype=np.uint8)
    np.testing.assert_array_equal(masks[:, 0], np.asarray(source.masks_roi[:, 0], dtype=np.uint8))
    assert np.count_nonzero(masks[:, 1]) == 0

    available = np.asarray(run["available_channels"][:], dtype=bool)
    np.testing.assert_array_equal(available, np.asarray([True, True], dtype=bool))
    edit_applied = np.asarray(run["edit_applied"][:], dtype=bool)
    assert not edit_applied.any()

    metrics = run["metrics"]
    mask_present = np.asarray(metrics["mask_present"][:], dtype=bool)
    area_px = np.asarray(metrics["area_px"][:], dtype=np.float32)
    np.testing.assert_array_equal(mask_present[:, 0], np.asarray([True, True], dtype=bool))
    np.testing.assert_array_equal(mask_present[:, 1], np.asarray([False, False], dtype=bool))
    np.testing.assert_allclose(area_px[:, 0], np.asarray([36.0, 16.0], dtype=np.float32))
    np.testing.assert_allclose(area_px[:, 1], np.asarray([0.0, 0.0], dtype=np.float32))

    body_reasons = read_reason_labels(run["components/subject_body"])
    swim_reasons = read_reason_labels(run["components/swim_bladder"])
    assert body_reasons is not None
    assert swim_reasons is not None
    assert body_reasons.tolist() == ["copied_from_source", "copied_from_source"]
    assert swim_reasons.tolist() == ["clean", "clean"]

    provenance = run.attrs["provenance"]
    assert provenance["stage"] == "refine_subject_masks"
    assert provenance["command"] == "scripts/py -m fisheye.tune.refined_subject_mask_review"
    assert provenance["git"]["commit"] == "b" * 40
    assert provenance["inputs"]["source_subject_mask_run"] == "subject_masks_001"
    assert provenance["inputs"]["source_keypoints_run"] == "refined_kp_001"
    assert provenance["inputs"]["source_keypoint_group"] == "refined_keypoints_runs"


def test_save_refined_subject_roi_updates_edit_applied_metrics_and_reasons() -> None:
    root = _build_subject_review_root()
    source, refined = mod.prepare_refined_subject_run(
        root,
        subject_run="subject_masks_001",
        refined_run="refined_subject_masks_001",
        components=("subject_body", "swim_bladder"),
    )

    edited = np.asarray(refined.group["masks_roi"][0], dtype=np.uint8)
    edited[0, 1:7, 1:7] = 0
    edited[1, 4:6, 4:6] = 1

    mod.save_refined_subject_roi(
        source=source,
        refined=refined,
        roi_idx=0,
        edited_masks=edited,
    )

    run = refined.group
    saved = np.asarray(run["masks_roi"][0], dtype=np.uint8)
    np.testing.assert_array_equal(saved, edited)

    edit_applied = np.asarray(run["edit_applied"][0], dtype=bool)
    np.testing.assert_array_equal(edit_applied, np.asarray([True, True], dtype=bool))

    mask_present = np.asarray(run["metrics/mask_present"][0], dtype=bool)
    area_px = np.asarray(run["metrics/area_px"][0], dtype=np.float32)
    np.testing.assert_array_equal(mask_present, np.asarray([False, True], dtype=bool))
    np.testing.assert_allclose(area_px, np.asarray([0.0, 4.0], dtype=np.float32))

    body_group = run["components/subject_body"]
    swim_group = run["components/swim_bladder"]
    assert bool(np.asarray(body_group["edit_applied"][0], dtype=bool)) is True
    assert bool(np.asarray(swim_group["edit_applied"][0], dtype=bool)) is True
    body_reasons = read_reason_labels(body_group)
    swim_reasons = read_reason_labels(swim_group)
    assert body_reasons is not None
    assert swim_reasons is not None
    assert body_reasons[0] == "manual_correction"
    assert swim_reasons[0] == "manual_correction"


def test_apply_component_review_status_aggregates_run_review_state() -> None:
    root = _build_subject_review_root()
    _source, refined = mod.prepare_refined_subject_run(
        root,
        subject_run="subject_masks_001",
        refined_run="refined_subject_masks_001",
        components=("subject_body", "swim_bladder"),
    )

    body_payload, run_payload = mod.apply_component_review_status(
        refined.parent,
        refined.run_name,
        refined.group,
        component_name="subject_body",
        state="approved",
        method="manual",
        intended_use="training",
        reviewer="tester",
        notes="body looks good",
    )
    assert body_payload["state"] == "approved"
    assert run_payload["state"] == "pending"

    swim_payload, run_payload = mod.apply_component_review_status(
        refined.parent,
        refined.run_name,
        refined.group,
        component_name="swim_bladder",
        state="needs_review",
        method="manual",
        intended_use="training",
        reviewer="tester",
        notes="swim bladder not done",
    )
    assert swim_payload["state"] == "needs_review"
    assert run_payload["state"] == "needs_review"

    _swim_payload, run_payload = mod.apply_component_review_status(
        refined.parent,
        refined.run_name,
        refined.group,
        component_name="swim_bladder",
        state="approved",
        method="manual",
        intended_use="training",
        reviewer="tester",
        notes="done",
    )
    assert run_payload["state"] == "approved"
    assert refined.parent.attrs["refined_subject_mask_review_status_latest"] == refined.run_name
