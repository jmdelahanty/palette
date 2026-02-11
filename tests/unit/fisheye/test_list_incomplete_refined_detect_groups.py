from __future__ import annotations

import numpy as np
import zarr

from fisheye.utils.list_incomplete_refined_detect_groups import _collect_issues


def _make_archive(base, name: str):
    zarr_path = base / name / "zarr" / f"{name}_analysis.zarr"
    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(zarr_path), mode="w")
    root.attrs["zarr_purpose"] = "analysis"
    return zarr_path, root


def _write_complete_subgroup(run_group: zarr.Group, subgroup_name: str) -> None:
    group = run_group.create_group(subgroup_name)
    group.create_array(
        "frame_indices",
        data=np.array([0, 1], dtype=np.int32),
        overwrite=True,
    )
    group.create_array(
        "bbox_norm_coords",
        data=np.array([[0.4, 0.4, 0.1, 0.1], [0.5, 0.5, 0.1, 0.1]], dtype=np.float32),
        overwrite=True,
    )


def test_collect_issues_flags_missing_required_arrays(tmp_path) -> None:
    zarr_path, root = _make_archive(tmp_path, "rec_bad")
    parent = root.create_group("refined_detect_runs")
    run = parent.create_group("refined_detect_1")
    parent.attrs["latest"] = "refined_detect_1"
    bad = run.create_group("interpolated")
    bad.create_array(
        "bbox_norm_coords",
        data=np.array([[0.4, 0.4, 0.1, 0.1]], dtype=np.float32),
        overwrite=True,
    )

    rows = _collect_issues([tmp_path], recursive=True, zarr_use_filter="analysis", latest_only=True)
    assert len(rows) == 1
    row = rows[0]
    assert row.zarr_path == str(zarr_path)
    assert row.refined_run == "refined_detect_1"
    assert row.subgroup == "interpolated"
    assert row.issue == "missing_required_arrays"
    assert row.missing_arrays == "frame_indices"


def test_collect_issues_latest_only_vs_all_runs(tmp_path) -> None:
    _zarr_path, root = _make_archive(tmp_path, "rec_multi")
    parent = root.create_group("refined_detect_runs")

    old_run = parent.create_group("refined_detect_old")
    bad = old_run.create_group("filtered")
    bad.create_array(
        "bbox_norm_coords",
        data=np.array([[0.4, 0.4, 0.1, 0.1]], dtype=np.float32),
        overwrite=True,
    )

    new_run = parent.create_group("refined_detect_new")
    _write_complete_subgroup(new_run, "filtered")
    parent.attrs["latest"] = "refined_detect_new"

    latest_only_rows = _collect_issues(
        [tmp_path],
        recursive=True,
        zarr_use_filter="analysis",
        latest_only=True,
    )
    assert latest_only_rows == []

    all_run_rows = _collect_issues(
        [tmp_path],
        recursive=True,
        zarr_use_filter="analysis",
        latest_only=False,
    )
    assert len(all_run_rows) == 1
    assert all_run_rows[0].refined_run == "refined_detect_old"
    assert all_run_rows[0].issue == "missing_required_arrays"
