from __future__ import annotations

import numpy as np
import pytest
import zarr

from fisheye.refinement import finalize_subject_masks as mod
from fisheye.shared.detect_reason_codec import read_reason_labels
from fisheye.tune import refined_subject_mask_review as review_mod


def _patch_refined_subject_provenance(monkeypatch) -> None:
    monkeypatch.setattr(
        review_mod,
        "get_git_info",
        lambda repo_path=None: {  # noqa: ARG005
            "commit_hash": "c" * 40,
            "short_hash": "cccccccc",
            "branch": "main",
            "is_dirty": False,
            "remote_url": "git@example.com:palette.git",
        },
    )
    monkeypatch.setattr(
        review_mod,
        "get_environment_info",
        lambda **kwargs: {  # noqa: ARG005
            "environment": {"python": "3.12"},
            "platform": {
                "hostname": "finalize-host",
                "system": "Linux",
                "release": "6.8",
                "python_version": "3.12.0",
                "machine": "x86_64",
            },
        },
    )


def _build_probability_root() -> zarr.Group:
    root = zarr.group()
    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "crop_001"
    crop = crop_parent.create_group("crop_001")
    crop.attrs["crop_storage_mode"] = "materialized"
    crop.attrs["crop_signature"] = {"signature_version": 2, "crop_revision": 7}
    crop.attrs["crop_revision"] = 7
    crop.attrs["detect_review_status_ref"] = "refined_detect_runs/refined_detect_001/review_status"
    crop.create_array("roi_images", data=np.zeros((2, 10, 10), dtype=np.uint8), overwrite=True)

    kp_parent = root.create_group("refined_keypoints_runs")
    kp_parent.attrs["latest"] = "refined_kp_001"
    kp = kp_parent.create_group("refined_kp_001")
    kp.attrs["keypoint_labels"] = ["swim_bladder", "eye_left", "eye_right"]
    kp.create_array(
        "keypoints_roi",
        data=np.asarray(
            [
                [[5.0, 6.0], [2.0, 2.0], [7.0, 2.0]],
                [[5.0, 6.0], [2.0, 4.0], [7.0, 4.0]],
            ],
            dtype=np.float32,
        ),
        overwrite=True,
    )
    kp.create_array("detection_success", data=np.asarray([True, True], dtype=bool), overwrite=True)

    parent = root.create_group("subject_mask_runs")
    parent.attrs["latest"] = "subject_probs_001"
    run = parent.create_group("subject_probs_001")
    run.attrs.update(
        {
            "source_crop_run": "crop_001",
            "source_crop_storage_mode": "materialized",
            "source_crop_signature": {"signature_version": 2, "crop_revision": 7},
            "source_crop_revision": 7,
            "source_detect_review_status_ref": "refined_detect_runs/refined_detect_001/review_status",
            "method": "unet_subject_masks_v1",
            "mask_labels": ["subject_body", "eyes_union", "swim_bladder"],
            "label_schema_id": "subject_v1_union_eyes",
            "source_keypoints_run": "refined_kp_001",
            "source_keypoint_run": "refined_kp_001",
            "source_keypoint_group": "refined_keypoints_runs",
            "created_at_utc": "2026-04-01T00:00:00+00:00",
            "probabilities_encoding": "linear_uint8_0_255",
            "mask_probability_threshold": 0.5,
        }
    )
    run.create_array("detection_source", data=np.asarray([0, 0], dtype=np.int8), overwrite=True)
    run.create_array("frame_indices", data=np.asarray([10, 11], dtype=np.int32), overwrite=True)
    run.create_array("detection_indices", data=np.asarray([0, 1], dtype=np.int32), overwrite=True)
    run.create_array("frame_counts", data=np.asarray([1, 1], dtype=np.int32), overwrite=True)
    run.create_array("available_channels", data=np.asarray([True, True, True], dtype=bool), overwrite=True)

    probs = np.zeros((2, 3, 10, 10), dtype=np.uint8)
    probs[:, 0, 2:9, 2:9] = 255
    probs[0, 0, 4:6, 4:6] = 0
    probs[0, 0, 0, 0] = 255
    probs[0, 1, 1:4, 1:4] = 255
    probs[0, 1, 1:4, 6:9] = 255
    probs[1, 1, 3:6, 1:4] = 255
    probs[1, 1, 3:6, 6:9] = 255
    probs[:, 2, 5:8, 4:7] = 255
    probs[0, 2, 6, 5] = 0
    run.create_array("mask_probs_roi", data=probs, overwrite=True)
    return root


def test_finalize_subject_mask_run_creates_refined_candidates_from_probabilities(monkeypatch) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    root = _build_probability_root()

    summary = mod.finalize_subject_mask_run(
        root,
        subject_run="subject_probs_001",
        refined_run="refined_subject_masks_smart_001",
    )

    assert summary["status"] == "updated"
    assert summary["component_names"] == ["subject_body", "eye_left", "eye_right", "swim_bladder"]
    assert summary["review_counts"]["subject_body"]["needs_review"] >= 1

    run = root["refined_subject_masks_runs"]["refined_subject_masks_smart_001"]
    assert run.attrs["method"] == "smart_finalize_subject_masks_v1"
    assert run.attrs["finalization_semantics"] == "smart_probability_to_refined_candidate"
    assert run.attrs["refined_subject_mask_review_status"]["state"] == "pending"
    assert run.attrs["component_review_statuses"]["subject_body"]["state"] == "pending"
    assert run.attrs["summary_statistics"]["rows_total"] == 2

    labels = list(run.attrs["mask_labels"])
    body_idx = labels.index("subject_body")
    eye_left_idx = labels.index("eye_left")
    eye_right_idx = labels.index("eye_right")
    swim_idx = labels.index("swim_bladder")
    masks = np.asarray(run["masks_roi"][:], dtype=np.uint8)
    assert masks[0, body_idx, 4, 4] == 1
    assert masks[0, body_idx, 0, 0] == 0
    assert np.count_nonzero(masks[:, eye_left_idx]) > 0
    assert np.count_nonzero(masks[:, eye_right_idx]) > 0
    assert masks[0, swim_idx, 6, 5] == 1

    body_reasons = read_reason_labels(run["components/subject_body"])
    assert body_reasons is not None
    assert "cleanup_thresholded_probability" in str(body_reasons[0])
    assert "cleanup_closed_gaps" in str(body_reasons[0])
    assert "needs_review" in str(body_reasons[0])

    eye_left_reasons = read_reason_labels(run["components/eye_left"])
    assert eye_left_reasons is not None
    assert "assigned_from_eyes_union" in str(eye_left_reasons[0])
    assert "split_by_keypoint" in str(eye_left_reasons[0])

    provenance = run["components/subject_body/provenance"].attrs
    assert provenance["finalization_method"] == "smart_finalize_subject_masks_v1"
    assert provenance["source_binary_derivation"] == "smart_finalize(mask_probs_roi)"
    assert provenance["source_probability_path"] == "subject_mask_runs/subject_probs_001/mask_probs_roi"
    assert provenance["source_probability_threshold"] == pytest.approx(0.5)
    assert "finalization_metrics" in run["components/subject_body"]
    metrics = run["components/subject_body/finalization_metrics"]
    assert metrics.attrs["schema_id"] == "refined_subject_component_finalization_metrics_v1"
    assert np.asarray(metrics["quality_code"][:], dtype=np.int16).shape == (2,)
    assert "relations" in run


def test_finalize_subject_mask_run_dry_run_and_overwrite_guard(monkeypatch) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    root = _build_probability_root()

    dry = mod.finalize_subject_mask_run(
        root,
        subject_run="subject_probs_001",
        refined_run="refined_subject_masks_smart_001",
        dry_run=True,
    )

    assert dry["status"] == "planned"
    assert dry["mutates_archive"] is False
    assert "refined_subject_masks_runs" not in root

    mod.finalize_subject_mask_run(
        root,
        subject_run="subject_probs_001",
        refined_run="refined_subject_masks_smart_001",
    )
    with pytest.raises(ValueError, match="already exists"):
        mod.finalize_subject_mask_run(
            root,
            subject_run="subject_probs_001",
            refined_run="refined_subject_masks_smart_001",
        )
