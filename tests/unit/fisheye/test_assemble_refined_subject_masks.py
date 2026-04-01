from __future__ import annotations

import os
import numpy as np
import zarr

from fisheye.refinement import assemble_refined_subject_masks as assemble_mod
from fisheye.refinement import refine_subject_masks as batch_mod
from fisheye.shared.detect_reason_codec import read_reason_labels
from fisheye.tune import refined_subject_mask_review as review_mod

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")


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
                "hostname": "assemble-host",
                "system": "Linux",
                "release": "6.8",
                "python_version": "3.12.0",
                "machine": "x86_64",
            },
        },
    )


def _create_subject_run(
    root: zarr.Group,
    *,
    run_name: str,
    method: str,
    mask_labels: list[str],
    available_channels: np.ndarray,
    masks: np.ndarray,
    source_crop_run: str = "crop_001",
    source_keypoints_run: str = "refined_kp_001",
    source_keypoint_group: str = "refined_keypoints_runs",
) -> zarr.Group:
    parent = root.require_group("subject_mask_runs")
    parent.attrs["latest"] = run_name
    run = parent.create_group(run_name)
    run.attrs.update(
        {
            "source_crop_run": source_crop_run,
            "method": method,
            "mask_labels": list(mask_labels),
            "label_schema_id": "subject_v1_lr" if mask_labels == ["subject_body", "eye_left", "eye_right", "swim_bladder"] else "subject_v1_custom",
            "source_keypoints_run": source_keypoints_run,
            "source_keypoint_run": source_keypoints_run,
            "source_keypoint_group": source_keypoint_group,
            "created_at_utc": "2026-04-01T00:00:00+00:00",
        }
    )
    run.create_array("detection_source", data=np.asarray([0, 0], dtype=np.int8), overwrite=True)
    run.create_array("frame_indices", data=np.asarray([10, 11], dtype=np.int32), overwrite=True)
    run.create_array("detection_indices", data=np.asarray([0, 1], dtype=np.int32), overwrite=True)
    run.create_array("frame_counts", data=np.asarray([1, 1], dtype=np.int32), overwrite=True)
    run.create_array("available_channels", data=np.asarray(available_channels, dtype=bool), overwrite=True)
    run.create_array("masks_roi", data=np.asarray(masks, dtype=np.uint8), overwrite=True)
    return run


def _build_assembly_root() -> zarr.Group:
    root = zarr.group()
    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "crop_001"
    crop = crop_parent.create_group("crop_001")
    crop.create_array("roi_images", data=np.zeros((2, 8, 8), dtype=np.uint8), overwrite=True)

    body_masks = np.zeros((2, 1, 8, 8), dtype=np.uint8)
    body_masks[0, 0, 1:7, 1:7] = 1
    body_masks[1, 0, 2:6, 2:6] = 1
    _create_subject_run(
        root,
        run_name="body_run_001",
        method="sam_subject_body_v1",
        mask_labels=["subject_body"],
        available_channels=np.asarray([True], dtype=bool),
        masks=body_masks,
    )

    eye_masks = np.zeros((2, 2, 8, 8), dtype=np.uint8)
    eye_masks[0, 0, 2:3, 2:3] = 1
    eye_masks[0, 1, 2:3, 5:6] = 1
    eye_masks[1, 0, 3:4, 2:3] = 1
    eye_masks[1, 1, 3:4, 5:6] = 1
    _create_subject_run(
        root,
        run_name="eye_run_001",
        method="refined_eye_projection_v1",
        mask_labels=["eye_left", "eye_right"],
        available_channels=np.asarray([True, True], dtype=bool),
        masks=eye_masks,
    )

    swim_masks = np.zeros((2, 1, 8, 8), dtype=np.uint8)
    swim_masks[0, 0, 4:6, 4:6] = 1
    swim_masks[1, 0, 3:5, 3:5] = 1
    _create_subject_run(
        root,
        run_name="swim_run_001",
        method="traditional_swim_bladder_inference",
        mask_labels=["swim_bladder"],
        available_channels=np.asarray([True], dtype=bool),
        masks=swim_masks,
    )
    return root


def test_assemble_refined_subject_run_creates_finalized_mixed_source_run(monkeypatch) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    root = _build_assembly_root()

    summary = assemble_mod.assemble_refined_subject_run(
        root,
        body_run="body_run_001",
        eye_run="eye_run_001",
        swim_run="swim_run_001",
        refined_run="refined_subject_masks_mixed_001",
    )

    assert summary["status"] == "updated"
    assert summary["component_names"] == ["subject_body", "eye_left", "eye_right", "swim_bladder"]
    assert summary["source_subject_mask_runs"] == {
        "subject_body": "body_run_001",
        "eye_left": "eye_run_001",
        "eye_right": "eye_run_001",
        "swim_bladder": "swim_run_001",
    }

    run = root["refined_subject_masks_runs"]["refined_subject_masks_mixed_001"]
    assert run.attrs["label_schema_id"] == "subject_v1_lr"
    assert run.attrs["method"] == assemble_mod.ASSEMBLE_REFINED_SUBJECT_METHOD
    assert run.attrs["assembly_semantics"] == "multi_source_component_seed"
    assert run.attrs["source_subject_mask_run"] == "body_run_001"
    assert run.attrs["source_body_subject_mask_run"] == "body_run_001"
    assert run.attrs["source_eye_subject_mask_run"] == "eye_run_001"
    assert run.attrs["source_swim_subject_mask_run"] == "swim_run_001"
    np.testing.assert_array_equal(
        np.asarray(run["available_channels"][:], dtype=bool),
        np.asarray([True, True, True, True], dtype=bool),
    )
    np.testing.assert_array_equal(
        np.asarray(run["edit_applied"][:], dtype=bool),
        np.zeros((2, 4), dtype=bool),
    )
    np.testing.assert_array_equal(
        np.asarray(run["metrics/mask_present"][:], dtype=bool),
        np.asarray(
            [
                [True, True, True, True],
                [True, True, True, True],
            ],
            dtype=bool,
        ),
    )

    body_reasons = read_reason_labels(run["components/subject_body"])
    eye_left_reasons = read_reason_labels(run["components/eye_left"])
    swim_reasons = read_reason_labels(run["components/swim_bladder"])
    assert body_reasons is not None
    assert eye_left_reasons is not None
    assert swim_reasons is not None
    assert body_reasons.tolist() == ["copied_from_source", "copied_from_source"]
    assert eye_left_reasons.tolist() == ["copied_from_source", "copied_from_source"]
    assert swim_reasons.tolist() == ["copied_from_source", "copied_from_source"]

    assert run["components/subject_body/provenance"].attrs["source_run"] == "body_run_001"
    assert run["components/eye_left/provenance"].attrs["source_run"] == "eye_run_001"
    assert run["components/eye_right/provenance"].attrs["source_run"] == "eye_run_001"
    assert run["components/swim_bladder/provenance"].attrs["source_run"] == "swim_run_001"
    assert run["components/swim_bladder/provenance"].attrs["last_update_mode"] == "create"


def test_refine_subject_masks_uses_component_provenance_for_assembled_run(monkeypatch) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    root = _build_assembly_root()
    monkeypatch.setattr(assemble_mod, "open_zarr_root", lambda *args, **kwargs: root)  # noqa: ARG005
    monkeypatch.setattr(batch_mod, "open_zarr_root", lambda *args, **kwargs: root)  # noqa: ARG005

    assemble_mod.assemble_refined_subject_masks(
        "in-memory.zarr",
        body_run="body_run_001",
        swim_run="swim_run_001",
        refined_run="refined_subject_masks_body_swim_001",
    )

    run = root["refined_subject_masks_runs"]["refined_subject_masks_body_swim_001"]
    edited_row = np.asarray(run["masks_roi"][0], dtype=np.uint8)
    edited_row[1, :, :] = 0
    run["masks_roi"][0] = edited_row

    summary = batch_mod.refine_subject_masks(
        "in-memory.zarr",
        refined_run="refined_subject_masks_body_swim_001",
        components=("swim_bladder",),
        roi_indices=[0, 1],
        chunk_size=1,
        scheduler="single-threaded",
    )

    assert summary["status"] == "updated"
    assert summary["source_subject_mask_run"] == "body_run_001"
    assert summary["changed_roi_count"] == 1
    assert summary["noop_roi_count"] == 1

    run = root["refined_subject_masks_runs"]["refined_subject_masks_body_swim_001"]
    np.testing.assert_array_equal(
        np.asarray(run["edit_applied"][:], dtype=bool),
        np.asarray(
            [
                [False, True],
                [False, False],
            ],
            dtype=bool,
        ),
    )
    swim_reasons = read_reason_labels(run["components/swim_bladder"])
    assert swim_reasons is not None
    assert swim_reasons.tolist() == ["manual_correction", "copied_from_source"]
