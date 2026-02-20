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
