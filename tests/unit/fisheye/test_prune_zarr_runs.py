from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.utils import prune_zarr_runs as mod


def _make_crop_lineage(root: zarr.Group, run_name: str = "crop_001") -> None:
    crop_parent = root.require_group("crop_runs")
    crop = crop_parent.create_group(run_name)
    crop.create_array("frame_indices", data=np.array([10], dtype=np.int32))
    crop.create_array("detection_indices", data=np.array([0], dtype=np.int32))
    crop.create_array("frame_counts", data=np.array([1], dtype=np.int32))
    crop_parent.attrs["latest"] = run_name


def _make_eye_run(
    parent: zarr.Group,
    run_name: str,
    *,
    source_crop_run: str | None,
    with_lineage_arrays: bool,
) -> None:
    run = parent.create_group(run_name)
    run.create_array("masks_roi", data=np.zeros((1, 2, 4, 4), dtype=np.uint8))
    if source_crop_run is not None:
        run.attrs["source_crop_run"] = source_crop_run
    if with_lineage_arrays:
        run.create_array("frame_indices", data=np.array([10], dtype=np.int32))
        run.create_array("detection_indices", data=np.array([0], dtype=np.int32))
        run.create_array("frame_counts", data=np.array([1], dtype=np.int32))


def _make_generic_run(parent: zarr.Group, run_name: str) -> None:
    run = parent.create_group(run_name)
    run.create_array("values", data=np.array([1], dtype=np.int32))


def test_iter_zarr_accepts_explicit_zarr_directory(tmp_path: Path) -> None:
    zarr_dir = tmp_path / "recording_training.zarr"
    zarr_dir.mkdir(parents=True, exist_ok=True)
    out = list(mod._iter_zarr([zarr_dir], recursive=False))
    assert out == [zarr_dir]


def test_build_lineage_failure_plan_prunes_nonlatest_failures(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_training.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    _make_crop_lineage(root)

    eye_parent = root.create_group("eye_masks_runs")
    _make_eye_run(eye_parent, "eye_masks_old_fail", source_crop_run="crop_001", with_lineage_arrays=False)
    _make_eye_run(eye_parent, "eye_masks_latest_pass", source_crop_run="crop_001", with_lineage_arrays=True)
    eye_parent.attrs["latest"] = "eye_masks_latest_pass"

    refined_parent = root.create_group("refined_eye_masks_runs")
    _make_eye_run(refined_parent, "refined_old_fail", source_crop_run="crop_001", with_lineage_arrays=False)
    _make_eye_run(refined_parent, "refined_latest_fail", source_crop_run=None, with_lineage_arrays=True)
    refined_parent.attrs["latest"] = "refined_latest_fail"

    deletions, skips = mod._build_lineage_failure_plan(
        root,
        ["eye_masks_runs", "refined_eye_masks_runs"],
    )

    assert deletions["eye_masks_runs"] == ["eye_masks_old_fail"]
    assert deletions["refined_eye_masks_runs"] == ["refined_old_fail"]
    assert "refined_eye_masks_runs" in skips
    assert "not pruning latest" in skips["refined_eye_masks_runs"]


def test_main_rejects_lineage_stage_without_lineage_mode(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="--lineage-stage requires --lineage-failures-only."):
        mod.main([str(tmp_path), "--lineage-stage", "eye_masks_runs"])


def test_resolve_selected_parents_defaults_to_all() -> None:
    selected = mod._resolve_selected_parents(None)
    assert selected == mod.RUN_PARENTS


def test_resolve_selected_parents_rejects_unknown_value() -> None:
    with pytest.raises(SystemExit, match="Unknown run parent"):
        mod._resolve_selected_parents("keypoints_runs,unknown_runs")


def test_build_plan_respects_selected_parents(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")

    keypoint_parent = root.create_group("keypoints_runs")
    _make_generic_run(keypoint_parent, "keypoints_old")
    _make_generic_run(keypoint_parent, "keypoints_latest")
    keypoint_parent.attrs["latest"] = "keypoints_latest"

    detect_parent = root.create_group("detect_runs")
    _make_generic_run(detect_parent, "detect_old")
    _make_generic_run(detect_parent, "detect_latest")
    detect_parent.attrs["latest"] = "detect_latest"

    deletions, _ = mod._build_plan(root, ["keypoints_runs"])
    assert deletions == {"keypoints_runs": ["keypoints_old"]}


def test_main_rejects_parents_with_lineage_mode(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="--parents cannot be used with --lineage-failures-only."):
        mod.main([str(tmp_path), "--lineage-failures-only", "--parents", "keypoints_runs"])
