from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fisheye.diagnostics import check_eye_mask_keypoint_coverage as mod


class _FakeGroup(dict):
    def __init__(self, *args: Any, attrs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.attrs = attrs or {}

    def get(self, key: str, default: Any = None) -> Any:
        return super().get(key, default)

    def group_keys(self) -> list[str]:
        keys: list[str] = []
        for key, value in self.items():
            if isinstance(value, _FakeGroup):
                keys.append(str(key))
        return keys


def _build_demo_root(
    *,
    fail_row_indices: tuple[int, ...] = (),
    include_eye_stage: bool = True,
) -> _FakeGroup:
    crop = _FakeGroup(
        {
            "frame_indices": np.array([100, 101, 102], dtype=np.int64),
        }
    )
    crop_parent = _FakeGroup({"crop_001": crop}, attrs={"latest": "crop_001"})

    keypoints_roi = np.array(
        [
            [[5.0, 5.0], [2.0, 2.0], [4.0, 2.0]],
            [[6.0, 5.0], [2.0, 2.0], [4.0, 2.0]],
            [[7.0, 5.0], [3.0, 3.0], [3.0, 3.0]],  # invalid (left/right identical)
        ],
        dtype=np.float32,
    )
    kp = _FakeGroup(
        {
            "keypoints_roi": keypoints_roi,
            "detection_success": np.array([True, True, True], dtype=bool),
        }
    )
    kp_parent = _FakeGroup({"kp_001": kp}, attrs={"latest": "kp_001"})

    root = _FakeGroup(
        {
            "crop_runs": crop_parent,
            "refined_keypoints_runs": kp_parent,
        }
    )

    if not include_eye_stage:
        return root

    eye = _FakeGroup(
        attrs={
            "source_crop_run": "crop_001",
            "source_keypoint_group": "refined_keypoints_runs",
            "source_keypoints_run": "kp_001",
        }
    )

    ellipse_success = np.array(
        [
            [True, True],
            [True, True],
            [False, False],
        ],
        dtype=bool,
    )
    for idx in fail_row_indices:
        ellipse_success[int(idx)] = np.array([True, False], dtype=bool)
    eye["ellipse_success"] = ellipse_success
    eye_parent = _FakeGroup(
        {
            "refined_eye_masks_001": eye,
        },
        attrs={"latest": "refined_eye_masks_001"},
    )
    root["refined_eye_masks_runs"] = eye_parent
    return root


def test_compute_keypoint_valid_mask_respects_success_and_geometry() -> None:
    keypoints = np.array(
        [
            [[0.0, 0.0], [1.0, 1.0], [3.0, 1.0]],   # valid
            [[0.0, 0.0], [1.0, 1.0], [3.0, 1.0]],   # success false
            [[0.0, 0.0], [2.0, 2.0], [2.0, 2.0]],   # identical eyes
            [[0.0, 0.0], [np.nan, 1.0], [3.0, 1.0]],  # NaN
        ],
        dtype=np.float32,
    )
    success = np.array([True, False, True, True], dtype=bool)

    valid = mod._compute_keypoint_valid_mask(keypoints, success)
    assert valid.tolist() == [True, False, False, False]


def test_analyze_root_reports_fail_for_missing_eye_pair(tmp_path: Path) -> None:
    zarr_path = tmp_path / "demo_training.zarr"
    root = _build_demo_root(fail_row_indices=(1,))

    report = mod._analyze_root(
        root=root,
        zarr_path=zarr_path,
        stage="auto",
        eye_run=None,
        keypoint_group=None,
        keypoint_run=None,
        sample_limit=3,
    )

    assert report.status == "fail"
    assert report.total_rois == 3
    assert report.keypoint_valid_rows == 2
    assert report.rows_with_two_eye_labels == 1
    assert report.rows_missing_two_eye_labels == 1
    assert report.sample_missing == [{"roi_idx": 1, "frame_idx": 101}]


def test_analyze_root_passes_when_all_keypoint_valid_rows_have_two_labels(tmp_path: Path) -> None:
    zarr_path = tmp_path / "demo_training.zarr"
    root = _build_demo_root(fail_row_indices=())

    report = mod._analyze_root(
        root=root,
        zarr_path=zarr_path,
        stage="auto",
        eye_run=None,
        keypoint_group=None,
        keypoint_run=None,
        sample_limit=3,
    )

    assert report.status == "pass"
    assert report.keypoint_valid_rows == 2
    assert report.rows_missing_two_eye_labels == 0


def test_analyze_root_missing_when_eye_stage_absent(tmp_path: Path) -> None:
    zarr_path = tmp_path / "missing_eye_training.zarr"
    root = _build_demo_root(include_eye_stage=False)

    report = mod._analyze_root(
        root=root,
        zarr_path=zarr_path,
        stage="auto",
        eye_run=None,
        keypoint_group=None,
        keypoint_run=None,
        sample_limit=3,
    )

    assert report.status == "missing"
    assert "No eye-mask runs found" in str(report.reason)


def test_run_appends_failing_zarrs_to_review_list(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fail_zarr = (tmp_path / "fail_training.zarr").resolve()
    pass_zarr = (tmp_path / "pass_training.zarr").resolve()
    roots = {
        str(fail_zarr): _build_demo_root(fail_row_indices=(1,)),
        str(pass_zarr): _build_demo_root(fail_row_indices=()),
    }
    monkeypatch.setattr(mod, "_collect_zarr_paths", lambda *_args, **_kwargs: [fail_zarr, pass_zarr])
    monkeypatch.setattr(mod.zarr, "open_group", lambda path, **_kwargs: roots[str(Path(path).resolve())])

    review_list = tmp_path / "eye_mask_manual_list.txt"
    review_list.write_text("# existing\n", encoding="utf-8")

    rc = mod.main(
        [
            str(tmp_path),
            "--recursive",
            "--append-review-list",
            str(review_list),
            "--no-log",
        ]
    )
    assert rc == 0

    lines = review_list.read_text(encoding="utf-8").splitlines()
    assert str(fail_zarr.resolve()) in lines
    assert str(pass_zarr.resolve()) not in lines

    # Second run should not duplicate entries.
    rc_again = mod.main(
        [
            str(tmp_path),
            "--recursive",
            "--append-review-list",
            str(review_list),
            "--no-log",
        ]
    )
    assert rc_again == 0
    lines_again = review_list.read_text(encoding="utf-8").splitlines()
    assert lines_again.count(str(fail_zarr.resolve())) == 1


def test_run_respects_zarr_use_filter_for_training_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fail_analysis = (tmp_path / "bad_analysis.zarr").resolve()
    pass_training = (tmp_path / "ok_training.zarr").resolve()
    roots = {
        str(fail_analysis): _build_demo_root(fail_row_indices=(1,)),
        str(pass_training): _build_demo_root(fail_row_indices=()),
    }
    monkeypatch.setattr(
        mod,
        "_collect_zarr_paths",
        lambda *_args, **_kwargs: [fail_analysis, pass_training],
    )
    monkeypatch.setattr(mod.zarr, "open_group", lambda path, **_kwargs: roots[str(Path(path).resolve())])

    rc = mod.main(
        [
            str(tmp_path),
            "--recursive",
            "--zarr-use",
            "training",
            "--strict",
            "--no-log",
        ]
    )
    assert rc == 0
