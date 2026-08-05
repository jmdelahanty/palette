from __future__ import annotations

import numpy as np

from fisheye.shared import observation_coordinate_publication as publication
from fisheye.shared.sampled_training_detection_selection import (
    SELECTION_REASON_ACCEPTED,
    SELECTION_REASON_MISSING,
    SELECTION_REASON_MULTIPLE,
    SELECTION_REASON_OUTSIDE_TARGET_CROP,
    SELECTION_REASON_WEAK_OR_BAD_MATCH,
    select_strong_single_detections,
)


def _norm_xyxy(boxes: np.ndarray, *, width: int, height: int) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float64)
    result = np.empty_like(boxes)
    result[:, 0] = (boxes[:, 0] + boxes[:, 2]) / (2.0 * width)
    result[:, 1] = (boxes[:, 1] + boxes[:, 3]) / (2.0 * height)
    result[:, 2] = (boxes[:, 2] - boxes[:, 0]) / width
    result[:, 3] = (boxes[:, 3] - boxes[:, 1]) / height
    return result


def test_strong_single_selection_excludes_ambiguous_and_bad_rows() -> None:
    proposal = np.tile(
        np.asarray([[40.0, 40.0, 60.0, 60.0]], dtype=np.float32),
        (5, 1),
    )
    detection_frames = np.asarray([0, 2, 2, 3, 4], dtype=np.int64)
    detection_boxes = np.asarray(
        [
            [41.0, 41.0, 59.0, 59.0],
            [40.0, 40.0, 60.0, 60.0],
            [42.0, 42.0, 58.0, 58.0],
            [40.0, 40.0, 60.0, 60.0],
            [20.0, 40.0, 60.0, 60.0],
        ],
        dtype=np.float64,
    )

    selection = select_strong_single_detections(
        crop_source_acquisition_frame_index=np.arange(5, dtype=np.int64),
        proposal_bbox_img_xyxy=proposal,
        target_roi_top_left_xy=np.tile(
            np.asarray([[25, 25]], dtype=np.int32),
            (5, 1),
        ),
        target_roi_size=(50, 50),
        detection_source_acquisition_frame_index=detection_frames,
        detection_bbox_norm_coords=_norm_xyxy(
            detection_boxes,
            width=100,
            height=100,
        ),
        detection_scores=np.asarray(
            [0.8, 0.9, 0.85, 0.3, 0.9],
            dtype=np.float32,
        ),
        source_width=100,
        source_height=100,
        minimum_score=0.4,
        minimum_proposal_iou=0.5,
    )

    np.testing.assert_array_equal(selection.candidate_count, [1, 0, 2, 1, 1])
    np.testing.assert_array_equal(
        selection.reason_code,
        [
            SELECTION_REASON_ACCEPTED,
            SELECTION_REASON_MISSING,
            SELECTION_REASON_MULTIPLE,
            SELECTION_REASON_WEAK_OR_BAD_MATCH,
            SELECTION_REASON_OUTSIDE_TARGET_CROP,
        ],
    )
    np.testing.assert_array_equal(selection.accepted_crop_row_indices, [0])
    np.testing.assert_array_equal(selection.accepted_detection_row_indices, [0])
    assert selection.reason_counts() == {
        "accepted_strong_single": 1,
        "missing_detection": 1,
        "multiple_candidates": 1,
        "weak_or_bad_match": 1,
        "outside_target_crop": 1,
    }


def test_policy_v2_rejects_a_padded_target_crop_but_v1_remains_loadable() -> None:
    kwargs = {
        "crop_source_acquisition_frame_index": np.asarray([0], dtype=np.int64),
        "proposal_bbox_img_xyxy": np.asarray(
            [[10.0, 40.0, 20.0, 50.0]],
            dtype=np.float32,
        ),
        "target_roi_top_left_xy": np.asarray([[-1, 25]], dtype=np.int32),
        "target_roi_size": (50, 50),
        "detection_source_acquisition_frame_index": np.asarray(
            [0],
            dtype=np.int64,
        ),
        "detection_bbox_norm_coords": _norm_xyxy(
            np.asarray([[10.0, 40.0, 20.0, 50.0]], dtype=np.float64),
            width=100,
            height=100,
        ),
        "detection_scores": np.asarray([0.9], dtype=np.float32),
        "source_width": 100,
        "source_height": 100,
        "minimum_score": 0.4,
        "minimum_proposal_iou": 0.5,
    }

    current = select_strong_single_detections(**kwargs)
    historical = select_strong_single_detections(
        **kwargs,
        policy_schema_version=1,
    )

    np.testing.assert_array_equal(
        current.reason_code,
        [SELECTION_REASON_OUTSIDE_TARGET_CROP],
    )
    assert current.accepted_row_count == 0
    np.testing.assert_array_equal(
        historical.reason_code,
        [SELECTION_REASON_ACCEPTED],
    )
    assert historical.accepted_row_count == 1


def test_crop_source_dispatches_sampled_and_ordinary_detection_families(
    monkeypatch,
) -> None:
    calls: list[tuple[str, object, str]] = []
    root = object()
    sampled = object()
    ordinary = object()

    def load_sampled(root_node, path):
        calls.append(("sampled", root_node, path))
        return sampled

    def load_ordinary(root_node, path):
        calls.append(("ordinary", root_node, path))
        return ordinary

    monkeypatch.setattr(
        publication,
        "load_persisted_sampled_training_detection_geometry",
        load_sampled,
    )
    monkeypatch.setattr(
        publication,
        "load_persisted_detection_observation_geometry",
        load_ordinary,
    )

    assert (
        publication._load_persisted_crop_source_observation_geometry(
            root,
            "sampled_detection_runs/strong_single",
        )
        is sampled
    )
    assert (
        publication._load_persisted_crop_source_observation_geometry(
            root,
            "detect_runs/canonical",
        )
        is ordinary
    )
    assert calls == [
        ("sampled", root, "sampled_detection_runs/strong_single"),
        ("ordinary", root, "detect_runs/canonical"),
    ]
