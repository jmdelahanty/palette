from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from fisheye.diagnostics import check_eye_mask_lineage as mod


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


def _build_root(*, mismatch_detection_indices: bool = False) -> _FakeGroup:
    frame_indices = np.array([100, 101, 102], dtype=np.int32)
    detection_indices = np.array([0, 1, 2], dtype=np.int32)
    if mismatch_detection_indices:
        detection_indices = np.array([0, 99, 2], dtype=np.int32)
    frame_counts = np.array([1, 1, 1], dtype=np.int32)

    crop = _FakeGroup(
        {
            "frame_indices": np.array([100, 101, 102], dtype=np.int32),
            "detection_indices": np.array([0, 1, 2], dtype=np.int32),
            "frame_counts": frame_counts,
        }
    )
    eye_run = _FakeGroup(
        {
            "masks_roi": np.zeros((3, 2, 4, 4), dtype=np.uint8),
            "frame_indices": frame_indices,
            "detection_indices": detection_indices,
            "frame_counts": frame_counts,
        },
        attrs={"source_crop_run": "crop_001"},
    )
    eye_parent = _FakeGroup({"eye_masks_001": eye_run}, attrs={"latest": "eye_masks_001"})
    crop_parent = _FakeGroup({"crop_001": crop}, attrs={"latest": "crop_001"})
    return _FakeGroup({"crop_runs": crop_parent, "eye_masks_runs": eye_parent})


def test_analyze_run_group_passes_when_arrays_match_crop() -> None:
    root = _build_root(mismatch_detection_indices=False)
    run_group = root["eye_masks_runs"]["eye_masks_001"]

    report = mod._analyze_run_group(
        root=root,
        stage="eye_masks_runs",
        run_name="eye_masks_001",
        run_group=run_group,
    )

    assert report.has_issues is False
    assert report.total_rois == 3
    assert report.mismatched_arrays == []


def test_analyze_run_group_reports_mismatch_sample() -> None:
    root = _build_root(mismatch_detection_indices=True)
    run_group = root["eye_masks_runs"]["eye_masks_001"]

    report = mod._analyze_run_group(
        root=root,
        stage="eye_masks_runs",
        run_name="eye_masks_001",
        run_group=run_group,
    )

    assert report.has_issues is True
    assert report.mismatched_arrays == ["detection_indices"]
    assert any("detection_indices: idx 1" in sample for sample in report.mismatch_samples)


def test_resolve_run_names_prefers_latest_unless_all_runs() -> None:
    parent = _FakeGroup(
        {
            "eye_masks_001": _FakeGroup(),
            "eye_masks_002": _FakeGroup(),
        },
        attrs={"latest": "eye_masks_002"},
    )
    assert mod._resolve_run_names(parent, explicit_run=None, all_runs=False) == ["eye_masks_002"]
    assert mod._resolve_run_names(parent, explicit_run=None, all_runs=True) == [
        "eye_masks_001",
        "eye_masks_002",
    ]


def test_run_strict_exit_code_tracks_lineage_mismatches(monkeypatch) -> None:
    args = mod.build_parser().parse_args(
        [
            "dummy.zarr",
            "--stage",
            "eye_masks_runs",
            "--strict",
            "--no-log",
        ]
    )

    monkeypatch.setattr(mod.zarr, "open_group", lambda *_args, **_kwargs: _build_root(mismatch_detection_indices=False))
    assert mod.run(args) == 0

    monkeypatch.setattr(mod.zarr, "open_group", lambda *_args, **_kwargs: _build_root(mismatch_detection_indices=True))
    assert mod.run(args) == 1


def test_run_batch_strict_counts_failures_across_zarrs(monkeypatch) -> None:
    paths = [Path("/tmp/a_training.zarr"), Path("/tmp/b_training.zarr")]
    args = mod.build_parser().parse_args(
        [
            "/tmp",
            "--recursive",
            "--stage",
            "eye_masks_runs",
            "--strict",
            "--no-log",
        ]
    )

    monkeypatch.setattr(mod, "_collect_zarr_paths", lambda *_args, **_kwargs: paths)

    def _open_group(path: str, *_args, **_kwargs):
        if "a_training.zarr" in str(path):
            return _build_root(mismatch_detection_indices=False)
        return _build_root(mismatch_detection_indices=True)

    monkeypatch.setattr(mod.zarr, "open_group", _open_group)
    assert mod.run(args) == 1


def test_run_batch_zarr_use_filter_skips_nonmatching(monkeypatch) -> None:
    paths = [Path("/tmp/a_analysis.zarr"), Path("/tmp/b_training.zarr")]
    args = mod.build_parser().parse_args(
        [
            "/tmp",
            "--recursive",
            "--stage",
            "eye_masks_runs",
            "--strict",
            "--zarr-use",
            "training",
            "--no-log",
        ]
    )

    monkeypatch.setattr(mod, "_collect_zarr_paths", lambda *_args, **_kwargs: paths)

    def _open_group(path: str, *_args, **_kwargs):
        if "a_analysis.zarr" in str(path):
            return _build_root(mismatch_detection_indices=True)
        return _build_root(mismatch_detection_indices=False)

    monkeypatch.setattr(mod.zarr, "open_group", _open_group)
    assert mod.run(args) == 0
