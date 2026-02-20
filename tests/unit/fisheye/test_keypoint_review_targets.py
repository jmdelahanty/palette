from __future__ import annotations

import json
from pathlib import Path

from fisheye.tune.keypoint_review import _parse_targets_arg


def test_parse_targets_arg_reads_frame_and_roi_entries_from_json_mapping(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_training.zarr"
    flag_file = tmp_path / "keypoint_frame_flags.json"
    flag_file.write_text(
        json.dumps(
            {
                str(zarr_path): [
                    {"frame_idx": 123, "roi_idx": 5},
                    {"frame_idx": 124},
                    {"roi_idx": 7},
                    130,
                ]
            }
        ),
        encoding="utf-8",
    )

    frames, roi_indices = _parse_targets_arg(str(flag_file), str(zarr_path))
    assert frames == [123, 124, 130]
    assert roi_indices == [5, 7]


def test_parse_targets_arg_plain_csv_returns_frames_only() -> None:
    frames, roi_indices = _parse_targets_arg("10, 11  12", "/tmp/recording.zarr")
    assert frames == [10, 11, 12]
    assert roi_indices is None

