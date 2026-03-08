from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fisheye.visualization.visualize_eye_masks import (
    EyeMaskViewer,
    _append_flagged_frame,
    _is_refined_variant,
    _load_frame_flags,
    parse_args,
)


def test_append_flagged_frame_dedupes_and_sorts(tmp_path: Path) -> None:
    flag_path = tmp_path / "eye_mask_frame_flags.json"
    zarr_path = "/tmp/recording_training.zarr"

    _append_flagged_frame(flag_path, zarr_path, frame_idx=20, roi_idx=5)
    _append_flagged_frame(flag_path, zarr_path, frame_idx=10, roi_idx=None)
    _append_flagged_frame(flag_path, zarr_path, frame_idx=20, roi_idx=5)
    _append_flagged_frame(flag_path, zarr_path, frame_idx=20, roi_idx=1)

    payload = json.loads(flag_path.read_text(encoding="utf-8"))
    assert payload[zarr_path] == [
        {"frame_idx": 10, "roi_idx": None},
        {"frame_idx": 20, "roi_idx": 1},
        {"frame_idx": 20, "roi_idx": 5},
    ]


def test_load_frame_flags_accepts_legacy_frame_lists(tmp_path: Path) -> None:
    flag_path = tmp_path / "legacy_frame_flags.json"
    flag_path.write_text(
        json.dumps(
            {
                "/tmp/a.zarr": [9, "bad", 1],
                "/tmp/b.zarr": [{"frame_idx": 5, "roi_idx": 2}, {"frame_idx": "x"}],
            }
        ),
        encoding="utf-8",
    )

    parsed = _load_frame_flags(flag_path)
    assert parsed["/tmp/a.zarr"] == [
        {"frame_idx": 9, "roi_idx": None},
        {"frame_idx": 1, "roi_idx": None},
    ]
    assert parsed["/tmp/b.zarr"] == [{"frame_idx": 5, "roi_idx": 2}]


def test_is_refined_variant_detects_refined_group_path() -> None:
    assert _is_refined_variant("refined_eye_masks_runs/refined_eye_masks_001")
    assert not _is_refined_variant("eye_masks_runs/eye_masks_001")


def test_parse_args_sets_default_frame_flag_file() -> None:
    args = parse_args(["/tmp/example.zarr"])
    assert args.frame_flag_file == "eye_mask_frame_flags.json"


def test_ellipse_curve_returns_expected_points_for_axis_aligned_case() -> None:
    curve = EyeMaskViewer._ellipse_curve(
        cx=10.0,
        cy=20.0,
        major=8.0,
        minor=4.0,
        orientation_deg=0.0,
        num_points=32,
    )
    assert curve.shape == (32, 2)
    # t=0 should lie at positive major-axis endpoint.
    assert curve[0, 0] == 14.0
    assert curve[0, 1] == 20.0


def test_ellipse_curve_returns_empty_for_invalid_params() -> None:
    curve = EyeMaskViewer._ellipse_curve(
        cx=np.nan,
        cy=0.0,
        major=8.0,
        minor=4.0,
        orientation_deg=0.0,
    )
    assert curve.shape == (0, 2)


class _FakeCropSource:
    def __init__(self, rois: np.ndarray) -> None:
        self._rois = np.asarray(rois)
        self.total_rois = int(self._rois.shape[0])

    def read_slice(self, start: int, end: int) -> np.ndarray:
        return np.asarray(self._rois[start:end])


class _MaskVariantStub:
    def __init__(self, masks: np.ndarray) -> None:
        self.name = "stub"
        self.group_path = "eye_masks_runs/stub"
        self.masks = np.asarray(masks, dtype=np.uint8)
        self.mask_probs = None
        self.ellipse_params = None
        self.ellipse_success = None
        self.eye_labels = ["mask"]
        self.display_names = ["Mask"]
        self.channel_colors = [np.array([1.0, 0.0, 0.0], dtype=np.float32)]
        self.channel_hex = ["#ff0000"]
        self.is_refined = False
        self.unrefined_note = None
        self.summary_lines = []

    @property
    def channel_count(self) -> int:
        return int(self.masks.shape[1])


def test_eye_mask_viewer_reads_roi_from_crop_source() -> None:
    rois = np.arange(2 * 4 * 4, dtype=np.uint8).reshape(2, 4, 4)
    crop_source = _FakeCropSource(rois)
    masks = np.zeros((2, 1, 4, 4), dtype=np.uint8)
    masks[1, 0, 1:3, 1:3] = 1
    success = np.array([True, True], dtype=bool)
    keypoints = np.array(
        [
            [[1.0, 1.0], [2.0, 2.0]],
            [[0.5, 0.5], [3.0, 3.0]],
        ],
        dtype=np.float32,
    )
    viewer = EyeMaskViewer(
        root=None,  # type: ignore[arg-type]
        variants=[_MaskVariantStub(masks)],
        crop_source=crop_source,  # type: ignore[arg-type]
        success_flags=success,
        keypoints=keypoints,
        keypoint_labels=["left", "right"],
    )

    _overlay, summary, _axes, _ellipse, _probs, base, mask_list, _variant, _kp, _valid = (
        viewer.make_overlay(1, 0)
    )

    assert base.shape == (4, 4)
    assert np.isclose(base.max(), 1.0)
    assert np.isclose(base.min(), 0.0)
    assert mask_list[0].sum() == 4
    assert "ROI 2/2" in summary
