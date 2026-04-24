from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import zarr

from fisheye.tune import keypoint_failure_review as failure_mod
from fisheye.tune import keypoint_review as mod
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


def test_parse_targets_arg_manual_exact_resolves_stable_ids_without_frame_overselect(
    tmp_path: Path,
) -> None:
    class _FakeArray:
        def __init__(self, data):
            self._data = data
            self.shape = data.shape

        def __getitem__(self, item):
            return self._data[item]

    class _FakeGroup(dict):
        def __init__(self, *args, attrs=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.attrs = attrs or {}

        def get(self, key, default=None):
            return super().get(key, default)

    zarr_path = tmp_path / "recording_training.zarr"
    flag_file = tmp_path / "keypoint_frame_flags.json"
    flag_file.write_text(
        json.dumps(
            {
                str(zarr_path): [
                    {
                        "frame_idx": 5,
                        "roi_idx": 0,
                        "source_refined_row_id": 102,
                    },
                    {"frame_idx": 7},
                ]
            }
        ),
        encoding="utf-8",
    )
    crop = _FakeGroup(
        {
            "frame_indices": _FakeArray(np.array([5, 5, 6], dtype=np.int64)),
            "source_refined_row_ids": _FakeArray(np.array([100, 101, 102], dtype=np.int64)),
        }
    )
    root = _FakeGroup({"crop_runs": _FakeGroup({"crop_001": crop}, attrs={"latest": "crop_001"})})

    frames, roi_indices = _parse_targets_arg(
        str(flag_file),
        str(zarr_path),
        manual_exact=True,
        root=root,
        crop_run="crop_001",
    )

    assert frames == [7]
    assert roi_indices == [2]


def test_parse_targets_arg_plain_csv_returns_frames_only() -> None:
    frames, roi_indices = _parse_targets_arg("10, 11  12", "/tmp/recording.zarr")
    assert frames == [10, 11, 12]
    assert roi_indices is None


def test_run_manual_review_uses_open_zarr_root_for_new_refined_runs(tmp_path: Path, monkeypatch) -> None:
    zarr_path = tmp_path / "recording_training.zarr"
    root = zarr.open_group(store=zarr_path, mode="w")
    parent = root.create_group("refined_keypoints_runs")
    parent.attrs["latest"] = "refined_keypoints_traditional_v2_seed_001"
    parent.create_group("refined_keypoints_traditional_v2_seed_001")

    opened: list[tuple[str, str]] = []

    def _open(path: str | Path, mode: str = "r", **_kwargs):
        opened.append((str(path), mode))
        return zarr.open_group(store=Path(path), mode=mode)

    monkeypatch.setattr(mod, "open_zarr_root", _open)
    monkeypatch.setattr(failure_mod, "launch_review", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        mod,
        "_update_postprocess_summary",
        lambda refined, print_summary=True, **_kwargs: {"ok": True},
    )

    result = mod.run_manual_review(
        str(zarr_path),
        refined_run="refined_keypoints_traditional_v2_seed_001",
        review_intended_use="training",
        reviewer="tester",
    )

    assert result == {"ok": True}
    assert opened == [
        (str(zarr_path), "a"),
        (str(zarr_path), "a"),
    ]
