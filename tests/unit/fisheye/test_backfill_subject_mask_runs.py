from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from fisheye.shared.zarr.stage_arrays import SUBJECT_MASKS_SPEC, validate_run
from fisheye.utils import backfill_subject_mask_runs as mod


def _create_crop_run(root: zarr.Group, run_name: str = "crop_001") -> None:
    crop_parent = root.require_group("crop_runs")
    crop_parent.attrs["latest"] = run_name
    crop = crop_parent.create_group(run_name)
    crop.create_array("frame_indices", data=np.asarray([0, 0], dtype=np.int32), overwrite=True)
    crop.create_array("frame_counts", data=np.asarray([2], dtype=np.int32), overwrite=True)
    crop.create_array("detection_indices", data=np.asarray([5, 6], dtype=np.int32), overwrite=True)
    crop.create_array("detection_source", data=np.asarray([0, 1], dtype=np.int8), overwrite=True)


def _create_eye_run(
    root: zarr.Group,
    *,
    run_name: str = "eye_masks_001",
    eye_labels: tuple[str, ...] | None = None,
) -> zarr.Group:
    eye_parent = root.require_group("eye_masks_runs")
    eye_parent.attrs["latest"] = run_name
    eye = eye_parent.create_group(run_name)
    eye.attrs["source_crop_run"] = "crop_001"
    eye.attrs["source_keypoints_run"] = "refined_keypoints_001"
    eye.attrs["source_keypoint_group"] = "refined_keypoints_runs"
    eye.attrs["probabilities_encoding"] = "linear_uint8_0_255"
    if eye_labels is not None:
        eye.attrs["eye_labels"] = list(eye_labels)
    masks = np.asarray(
        [
            [
                [[1, 0], [0, 0]],
                [[0, 1], [0, 0]],
            ],
            [
                [[0, 0], [0, 0]],
                [[0, 0], [1, 0]],
            ],
        ],
        dtype=np.uint8,
    )
    probs = np.asarray(
        [
            [
                [[255, 0], [0, 0]],
                [[0, 128], [0, 0]],
            ],
            [
                [[0, 0], [0, 0]],
                [[0, 0], [64, 0]],
            ],
        ],
        dtype=np.uint8,
    )
    eye.create_array("masks_roi", data=masks, overwrite=True)
    eye.create_array("mask_probs_roi", data=probs, overwrite=True)
    eye.create_array("detection_source", data=np.asarray([0, 1], dtype=np.int8), overwrite=True)
    return eye


def _create_refined_eye_run(root: zarr.Group, *, run_name: str = "refined_eye_masks_001") -> zarr.Group:
    refined_parent = root.require_group("refined_eye_masks_runs")
    refined_parent.attrs["latest"] = run_name
    refined = refined_parent.create_group(run_name)
    refined.attrs["source_crop_run"] = "crop_001"
    refined.attrs["source_eye_masks_run"] = "eye_masks_001"
    refined.attrs["source_keypoints_run"] = "refined_keypoints_001"
    refined.attrs["source_keypoint_group"] = "refined_keypoints_runs"
    refined.attrs["eye_labels"] = ["eye_left", "eye_right"]
    refined_masks = np.asarray(
        [
            [
                [[1, 0], [0, 0]],
                [[0, 0], [0, 0]],
            ],
            [
                [[0, 0], [0, 0]],
                [[0, 1], [0, 0]],
            ],
        ],
        dtype=np.uint8,
    )
    refined.create_array("masks_roi", data=refined_masks, overwrite=True)
    refined.create_array("detection_source", data=np.asarray([0, 1], dtype=np.int8), overwrite=True)
    return refined


def test_backfill_subject_mask_run_projects_raw_eye_masks(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    _create_crop_run(root)
    _create_eye_run(root)

    summary = mod.backfill_subject_mask_run(zarr_path, apply=True)

    assert summary["status"] == "updated"
    assert summary["source_stage"] == "eye_masks_runs"
    subject_parent = root["subject_mask_runs"]
    assert subject_parent.attrs["latest"] == "subject_masks_from_eye_masks_001"
    run = subject_parent["subject_masks_from_eye_masks_001"]

    masks = np.asarray(run["masks_roi"][:], dtype=np.uint8)
    expected_masks = np.asarray(
        [
            [
                [[0, 0], [0, 0]],
                [[1, 1], [0, 0]],
                [[0, 0], [0, 0]],
            ],
            [
                [[0, 0], [0, 0]],
                [[0, 0], [1, 0]],
                [[0, 0], [0, 0]],
            ],
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(masks, expected_masks)
    assert tuple(run["masks_roi"].chunks) == (2, 1, 2, 2)
    assert run["masks_roi"].fill_value == 0

    probs = np.asarray(run["mask_probs_roi"][:], dtype=np.uint8)
    expected_probs = np.asarray(
        [
            [
                [[0, 0], [0, 0]],
                [[255, 128], [0, 0]],
                [[0, 0], [0, 0]],
            ],
            [
                [[0, 0], [0, 0]],
                [[0, 0], [64, 0]],
                [[0, 0], [0, 0]],
            ],
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(probs, expected_probs)
    assert tuple(run["mask_probs_roi"].chunks) == (2, 1, 2, 2)
    assert run["mask_probs_roi"].fill_value == 0

    np.testing.assert_array_equal(np.asarray(run["available_channels"][:], dtype=bool), np.asarray([False, True, False]))
    np.testing.assert_array_equal(np.asarray(run["frame_indices"][:], dtype=np.int32), np.asarray([0, 0], dtype=np.int32))
    np.testing.assert_array_equal(
        np.asarray(run["detection_indices"][:], dtype=np.int32),
        np.asarray([5, 6], dtype=np.int32),
    )

    prob_max = np.asarray(run["metrics/prob_max"][:], dtype=np.float32)
    np.testing.assert_allclose(
        prob_max,
        np.asarray(
            [
                [0.0, 1.0, 0.0],
                [0.0, 64.0 / 255.0, 0.0],
            ],
            dtype=np.float32,
        ),
        atol=1e-6,
    )
    mask_present = np.asarray(run["metrics/mask_present"][:], dtype=bool)
    np.testing.assert_array_equal(
        mask_present,
        np.asarray(
            [
                [False, True, False],
                [False, True, False],
            ],
            dtype=bool,
        ),
    )

    assert run.attrs["label_schema_id"] == "subject_v1_union"
    assert run.attrs["projection_mode"] == "eyes_union_from_pair"
    assert run.attrs["source_eye_masks_run"] == "eye_masks_001"
    assert run.attrs["source_probability_path"] == "eye_masks_runs/eye_masks_001/mask_probs_roi"
    assert run.attrs["probabilities_encoding"] == "linear_uint8_0_255"

    validation = validate_run(run, SUBJECT_MASKS_SPEC)
    assert validation.valid, validation.errors


def test_backfill_subject_mask_run_projects_refined_eye_masks_to_lr_schema(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    _create_crop_run(root)
    _create_eye_run(root)
    _create_refined_eye_run(root)

    summary = mod.backfill_subject_mask_run(
        zarr_path,
        source_stage="refined_eye_masks_runs",
        apply=True,
    )

    assert summary["status"] == "updated"
    run = root["subject_mask_runs/subject_masks_from_refined_eye_masks_001"]
    masks = np.asarray(run["masks_roi"][:], dtype=np.uint8)
    expected_masks = np.asarray(
        [
            [
                [[0, 0], [0, 0]],
                [[1, 0], [0, 0]],
                [[0, 0], [0, 0]],
                [[0, 0], [0, 0]],
            ],
            [
                [[0, 0], [0, 0]],
                [[0, 0], [0, 0]],
                [[0, 1], [0, 0]],
                [[0, 0], [0, 0]],
            ],
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(masks, expected_masks)
    assert tuple(run["masks_roi"].chunks) == (2, 1, 2, 2)
    assert run["masks_roi"].fill_value == 0

    probs = np.asarray(run["mask_probs_roi"][:], dtype=np.float32)
    np.testing.assert_array_equal(
        probs,
        expected_masks.astype(np.float32),
    )
    assert tuple(run["mask_probs_roi"].chunks) == (2, 1, 2, 2)
    assert run["mask_probs_roi"].fill_value == 0.0
    np.testing.assert_array_equal(
        np.asarray(run["available_channels"][:], dtype=bool),
        np.asarray([False, True, True, False]),
    )
    np.testing.assert_array_equal(
        np.asarray(run["metrics/mask_present"][:], dtype=bool),
        np.asarray(
            [
                [False, True, False, False],
                [False, False, True, False],
            ],
            dtype=bool,
        ),
    )
    assert run.attrs["source_refined_eye_masks_run"] == "refined_eye_masks_001"
    assert run.attrs["source_eye_masks_run"] == "eye_masks_001"
    assert run.attrs["label_schema_id"] == "subject_v1_lr"
    assert run.attrs["projection_mode"] == "eye_lr_from_lr"
    assert run.attrs["source_probability_path"] == "refined_eye_masks_runs/refined_eye_masks_001/masks_roi"
    assert run.attrs["probabilities_encoding"] == "unit_float"
    validation = validate_run(run, SUBJECT_MASKS_SPEC)
    assert validation.valid, validation.errors


def test_backfill_subject_mask_run_can_preserve_raw_lr_when_eye_labels_are_anatomical(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    _create_crop_run(root)
    _create_eye_run(root, eye_labels=("eye_left", "eye_right"))

    summary = mod.backfill_subject_mask_run(zarr_path, apply=True)

    assert summary["status"] == "updated"
    run = root["subject_mask_runs/subject_masks_from_eye_masks_001"]
    masks = np.asarray(run["masks_roi"][:], dtype=np.uint8)
    np.testing.assert_array_equal(
        masks,
        np.asarray(
            [
                [
                    [[0, 0], [0, 0]],
                    [[1, 0], [0, 0]],
                    [[0, 1], [0, 0]],
                    [[0, 0], [0, 0]],
                ],
                [
                    [[0, 0], [0, 0]],
                    [[0, 0], [0, 0]],
                    [[0, 0], [1, 0]],
                    [[0, 0], [0, 0]],
                ],
            ],
            dtype=np.uint8,
        ),
    )
    probs = np.asarray(run["mask_probs_roi"][:], dtype=np.uint8)
    np.testing.assert_array_equal(
        probs,
        np.asarray(
            [
                [
                    [[0, 0], [0, 0]],
                    [[255, 0], [0, 0]],
                    [[0, 128], [0, 0]],
                    [[0, 0], [0, 0]],
                ],
                [
                    [[0, 0], [0, 0]],
                    [[0, 0], [0, 0]],
                    [[0, 0], [64, 0]],
                    [[0, 0], [0, 0]],
                ],
            ],
            dtype=np.uint8,
        ),
    )
    assert run.attrs["label_schema_id"] == "subject_v1_lr"
    assert run.attrs["source_probability_path"] == "eye_masks_runs/eye_masks_001/mask_probs_roi"


def test_backfill_subject_mask_run_rejects_lr_projection_from_unlabeled_pair_source(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    _create_crop_run(root)
    _create_eye_run(root)

    try:
        mod.backfill_subject_mask_run(zarr_path, label_schema="subject_v1_lr", apply=False)
    except ValueError as exc:
        assert "does not provide anatomical left/right eye labels" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected subject_v1_lr projection to fail for unlabeled raw eye-mask runs.")


def test_backfill_subject_mask_run_dry_run_does_not_write(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    _create_crop_run(root)
    _create_eye_run(root)
    _create_refined_eye_run(root)

    summary = mod.backfill_subject_mask_run(zarr_path, apply=False)

    assert summary["status"] == "would_update"
    assert summary["source_stage"] == "eye_masks_runs"
    assert "subject_mask_runs" not in root
