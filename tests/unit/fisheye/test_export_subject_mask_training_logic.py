"""Pure logic tests for subject-mask training export helpers."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.utils import export_subject_mask_training_zarr as mod


def test_project_masks_and_validity_collapses_lr_to_union() -> None:
    source_masks = np.zeros((2, 4, 8, 8), dtype=np.uint8)
    source_masks[:, 1, 2:4, 2:4] = 1
    source_masks[:, 2, 2:4, 5:7] = 1
    source_available = np.array([False, True, True, False], dtype=np.bool_)

    projected_masks, projected_valid = mod._project_masks_and_validity(  # noqa: SLF001
        source_masks,
        source_schema_id="subject_v1_lr",
        source_available=source_available,
        target_schema_id="subject_v1_union",
    )

    assert projected_masks.shape == (2, 3, 8, 8)
    assert projected_valid.shape == (2, 3)
    assert projected_valid[:, 0].tolist() == [False, False]
    assert projected_valid[:, 1].tolist() == [True, True]
    assert projected_valid[:, 2].tolist() == [False, False]
    assert int(np.sum(projected_masks[:, 1])) > 0


def test_summarize_channel_supervision_classifies_eyes_only() -> None:
    masks = np.zeros((3, 4, 8, 8), dtype=np.uint8)
    masks[:, 1, 2:4, 2:4] = 1
    masks[:, 2, 2:4, 5:7] = 1
    target_valid = np.array(
        [
            [False, True, True, False],
            [False, True, True, False],
            [False, True, True, False],
        ],
        dtype=np.bool_,
    )

    summary = mod._summarize_channel_supervision(  # noqa: SLF001
        masks_roi=masks,
        target_valid_channels=target_valid,
        mask_labels=["subject_body", "eye_left", "eye_right", "swim_bladder"],
    )

    assert summary["coverage_class"] == "eyes_only"
    assert summary["contains_only_eye_masks"] is True
    assert summary["supervised_row_counts"] == {
        "subject_body": 0,
        "eye_left": 3,
        "eye_right": 3,
        "swim_bladder": 0,
    }
