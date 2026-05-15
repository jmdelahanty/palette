from __future__ import annotations

import pytest
import torch

from fisheye.diagnostics import compare_detect_decode_backend_predictions as mod


class _Boxes:
    def __init__(self) -> None:
        self.xyxy = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
        self.conf = torch.tensor([0.9])
        self.cls = torch.tensor([1.0])

    def __len__(self) -> int:
        return 1


class _Prediction:
    boxes = _Boxes()


def test_prediction_rows_normalizes_xyxy_boxes() -> None:
    rows = mod._prediction_rows(  # noqa: SLF001
        [_Prediction()],
        frames=[5],
        inference_height=100,
        inference_width=200,
    )

    assert rows[0]["frame"] == 5
    assert rows[0]["count"] == 1
    assert rows[0]["boxes_norm"] == [[0.05, 0.2, 0.15, 0.4]]
    assert rows[0]["scores"][0] == pytest.approx(0.9)
    assert rows[0]["class_ids"] == [1]


def test_compare_rows_reports_mismatches_and_diffs() -> None:
    rows_a = [
        {"frame": 0, "count": 1, "boxes_norm": [[0.1, 0.2, 0.3, 0.4]], "scores": [0.9], "class_ids": [0]},
        {"frame": 1, "count": 1, "boxes_norm": [[0.5, 0.2, 0.3, 0.4]], "scores": [0.8], "class_ids": [0]},
    ]
    rows_b = [
        {"frame": 0, "count": 1, "boxes_norm": [[0.2, 0.2, 0.3, 0.4]], "scores": [0.7], "class_ids": [0]},
        {"frame": 1, "count": 0, "boxes_norm": [], "scores": [], "class_ids": []},
    ]

    result = mod._compare_rows(rows_a, rows_b)  # noqa: SLF001

    assert result["detections_a"] == 2
    assert result["detections_b"] == 1
    assert result["count_mismatch_frames"] == 1
    assert result["first_count_mismatch_frames"] == [1]
    assert result["bbox_abs_diff_max"] == pytest.approx(0.1)
    assert result["score_abs_diff_max"] == pytest.approx(0.2)
