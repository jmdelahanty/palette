from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from fisheye.tune import eye_mask_review as mod


class _FakeArray:
    def __init__(self, data: np.ndarray) -> None:
        self._data = np.asarray(data)
        self.shape = self._data.shape

    def __getitem__(self, item: object) -> np.ndarray:
        return self._data[item]


class _FakeGroup(dict):
    def __init__(self, *args: Any, attrs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.attrs = attrs or {}

    def get(self, key: str, default: Any = None) -> Any:
        return super().get(key, default)


def _refined_group(ellipse_success: np.ndarray, total_rois: int) -> _FakeGroup:
    return _FakeGroup(
        {
            "ellipse_success": _FakeArray(np.asarray(ellipse_success, dtype=bool)),
            "eye_separation": _FakeArray(np.ones((total_rois,), dtype=np.float32)),
            "masks_roi": _FakeArray(np.ones((total_rois, 2, 8, 8), dtype=np.uint8)),
        },
        attrs={"source_crop_run": "crop_001"},
    )


def test_manual_review_start_roi_prefers_frame_flag_file(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording.zarr"
    refined = _refined_group(
        np.asarray(
            [
                [False, False],
                [True, True],
                [True, True],
                [True, True],
            ],
            dtype=bool,
        ),
        total_rois=4,
    )
    crop = _FakeGroup(
        {
            "roi_images": _FakeArray(np.zeros((4, 8, 8), dtype=np.uint8)),
            "frame_indices": _FakeArray(np.asarray([10, 11, 12, 13], dtype=np.int64)),
        }
    )
    root = _FakeGroup({"crop_runs": _FakeGroup({"crop_001": crop})})
    flag_path = tmp_path / "coverage_failures.json"
    flag_path.write_text(
        json.dumps({str(zarr_path): [{"frame_idx": 12, "roi_idx": 2}]}),
        encoding="utf-8",
    )

    start_roi = mod._resolve_manual_review_start_roi(
        root=root,
        refined=refined,
        zarr_path=str(zarr_path),
        crop_run="crop_001",
        frame_flag_file=str(flag_path),
    )

    assert start_roi == 2


def test_manual_review_start_roi_falls_back_to_refined_eye_failure(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording.zarr"
    refined = _refined_group(
        np.asarray(
            [
                [True, True],
                [False, True],
                [True, True],
            ],
            dtype=bool,
        ),
        total_rois=3,
    )
    root = _FakeGroup()

    start_roi = mod._resolve_manual_review_start_roi(
        root=root,
        refined=refined,
        zarr_path=str(zarr_path),
        crop_run=None,
        frame_flag_file=None,
    )

    assert start_roi == 1
