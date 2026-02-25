from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fisheye.tune import eye_mask_failure_review as mod


def test_collect_flagged_roi_indices_merges_roi_and_frame_entries(tmp_path: Path) -> None:
    flag_path = tmp_path / "eye_mask_frame_flags.json"
    zarr_path = "/tmp/recording_training.zarr"
    flag_path.write_text(
        json.dumps(
            {
                zarr_path: [
                    {"frame_idx": 10, "roi_idx": 1},
                    {"frame_idx": 11, "roi_idx": None},
                    10,
                ]
            }
        ),
        encoding="utf-8",
    )

    frame_indices = np.array([9, 10, 10, 11], dtype=np.int64)
    out = mod._collect_flagged_roi_indices(
        flag_path=flag_path,
        zarr_path=zarr_path,
        total_rois=4,
        frame_indices=frame_indices,
    )

    np.testing.assert_array_equal(out, np.array([1, 2, 3], dtype=np.int32))


def test_collect_flagged_roi_indices_matches_resolved_zarr_key(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_training.zarr"
    resolved_key = str(zarr_path.resolve())
    flag_path = tmp_path / "eye_mask_frame_flags.json"
    flag_path.write_text(
        json.dumps(
            {
                resolved_key: [
                    {"frame_idx": 5, "roi_idx": 0},
                ]
            }
        ),
        encoding="utf-8",
    )

    out = mod._collect_flagged_roi_indices(
        flag_path=flag_path,
        zarr_path=str(zarr_path),
        total_rois=3,
        frame_indices=np.array([5, 6, 7], dtype=np.int64),
    )

    np.testing.assert_array_equal(out, np.array([0], dtype=np.int32))


def test_next_review_position_wraps_when_current_removed_from_end() -> None:
    failures = np.array([100, 200], dtype=np.int32)
    out = mod._next_review_position(
        failures,
        previous_roi_idx=300,
        previous_pos=2,
    )
    assert out == 0


def test_next_review_position_keeps_forward_order_when_current_removed() -> None:
    failures = np.array([10, 30, 40], dtype=np.int32)
    out = mod._next_review_position(
        failures,
        previous_roi_idx=20,
        previous_pos=1,
    )
    assert out == 1


def test_load_failure_indices_includes_small_area_pairs(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeArray:
        def __init__(self, data: np.ndarray) -> None:
            self._data = data

        def __getitem__(self, item):
            if isinstance(item, slice):
                return self._data[item]
            if item == slice(None):
                return self._data
            return self._data[item]

    class _FakeGroup(dict):
        def __init__(self, *args, attrs: dict[str, object] | None = None, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.attrs = attrs or {}

        def get(self, key: str, default=None):
            return super().get(key, default)

    monkeypatch.setattr(mod.zarr, "Group", _FakeGroup)

    metrics = _FakeGroup(
        {
            "area_refined": _FakeArray(
                np.array(
                    [
                        [40.0, 100.0],  # left eye below default threshold (50 px)
                        [80.0, 90.0],
                    ],
                    dtype=np.float32,
                )
            )
        }
    )
    refined = _FakeGroup(
        {
            "ellipse_success": _FakeArray(
                np.array(
                    [
                        [True, True],
                        [True, True],
                    ],
                    dtype=bool,
                )
            ),
            "eye_separation": _FakeArray(np.array([10.0, 10.0], dtype=np.float32)),
            "metrics": metrics,
        },
        attrs={},
    )

    failures = mod._load_failure_indices(refined, min_sep=None, max_sep=None)
    np.testing.assert_array_equal(failures, np.array([0], dtype=np.int32))
