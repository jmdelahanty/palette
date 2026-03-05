from __future__ import annotations

from pathlib import Path

import zarr

from fisheye.tune.keypoint_failure_review import (
    _build_manual_reason,
    _resolve_full_frame_dimensions,
    _resolve_review_intended_use,
)


def test_resolve_full_frame_dimensions_from_root_attrs_when_images_full_missing(tmp_path: Path) -> None:
    zarr_path = tmp_path / "analysis.zarr"
    root = zarr.open_group(store=zarr_path, mode="w")
    root.attrs["width"] = 4512
    root.attrs["height"] = 4512
    root.create_group("raw_video")

    full_h, full_w = _resolve_full_frame_dimensions(root)
    assert full_h == 4512
    assert full_w == 4512


def test_resolve_full_frame_dimensions_from_images_ds_when_attrs_missing(tmp_path: Path) -> None:
    zarr_path = tmp_path / "analysis_ds_only.zarr"
    root = zarr.open_group(store=zarr_path, mode="w")
    raw = root.create_group("raw_video")
    raw.create_array("images_ds", shape=(3, 720, 1280), dtype="u1")

    full_h, full_w = _resolve_full_frame_dimensions(root)
    assert full_h == 720
    assert full_w == 1280


def test_build_manual_reason_is_canonical_and_idempotent() -> None:
    first = _build_manual_reason("manual_correction|geometry_issue", geom_ok=False)
    second = _build_manual_reason(first, geom_ok=False)
    assert first == "manual_correction|geometry_issue"
    assert second == first


def test_resolve_review_intended_use_prefers_existing_status(tmp_path: Path) -> None:
    zarr_path = tmp_path / "analysis_with_existing_review.zarr"
    root = zarr.open_group(store=zarr_path, mode="w")
    root.attrs["zarr_use"] = "analysis"
    refined = root.create_group("refined_keypoints_runs").create_group("refined_1")
    refined.attrs["keypoint_review_status"] = {
        "state": "approved",
        "method": "algorithmic",
        "intended_use": "full_recording",
    }

    resolved = _resolve_review_intended_use(
        requested=None,
        refined=refined,
        root=root,
        zarr_path=str(zarr_path),
    )
    assert resolved == "full_recording"


def test_resolve_review_intended_use_uses_training_for_training_zarr(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_training.zarr"
    root = zarr.open_group(store=zarr_path, mode="w")
    root.attrs["zarr_use"] = "training"
    refined = root.create_group("refined_keypoints_runs").create_group("refined_1")

    resolved = _resolve_review_intended_use(
        requested=None,
        refined=refined,
        root=root,
        zarr_path=str(zarr_path),
    )
    assert resolved == "training"
