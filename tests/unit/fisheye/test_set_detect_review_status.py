from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.utils import set_detect_review_status as mod


def _make_zarr(path: Path) -> Path:
    root = zarr.open_group(store=path, mode="w")
    parent = root.create_group("refined_detect_runs")
    parent.attrs["latest"] = "refined_1"
    run = parent.create_group("refined_1")
    group = run.create_group("interpolated")
    group.create_array("frame_indices", data=np.asarray([0, 1], dtype=np.int32))
    group.create_array(
        "bbox_norm_coords",
        data=np.asarray(
            [
                [0.5, 0.5, 0.2, 0.2],
                [0.4, 0.4, 0.1, 0.1],
            ],
            dtype=np.float64,
        ),
    )
    group.create_array("frame_counts", data=np.asarray([1, 1], dtype=np.int32))
    return path


def test_set_detect_review_status_approved_sets_authoritative_without_legacy_pointer(
    tmp_path: Path,
) -> None:
    zarr_path = _make_zarr(tmp_path / "rec.zarr")
    rc = mod.main(
        [
            str(zarr_path),
            "--state",
            "approved",
            "--method",
            "manual",
            "--intended-use",
            "training",
            "--reviewer",
            "operator1",
        ]
    )
    assert rc == 0

    root = zarr.open_group(store=zarr_path, mode="r")
    parent = root["refined_detect_runs"]
    status = dict(parent["refined_1"].attrs["detect_review_status"])
    assert status["state"] == "approved"
    assert status["resolved_group"] == "interpolated"
    assert status["authoritative_approval"]["status"] == "ok"
    assert parent.attrs["authoritative_run"] == "refined_1"
    assert "detect_review_status_latest" not in parent.attrs


def test_set_detect_review_status_non_approved_skips_authoritative_approval(tmp_path: Path) -> None:
    zarr_path = _make_zarr(tmp_path / "rec.zarr")
    rc = mod.main([str(zarr_path), "--state", "pending"])
    assert rc == 0

    root = zarr.open_group(store=zarr_path, mode="r")
    parent = root["refined_detect_runs"]
    status = dict(parent["refined_1"].attrs["detect_review_status"])
    assert status["state"] == "pending"
    assert "authoritative_approval" not in status
    assert "authoritative_run" not in parent.attrs
    assert "detect_review_status_latest" not in parent.attrs


def test_set_detect_review_status_approved_is_fail_closed_when_run_is_incomplete(
    tmp_path: Path,
) -> None:
    zarr_path = _make_zarr(tmp_path / "strict.zarr")
    root = zarr.open_group(store=zarr_path, mode="a")
    # Strict completion epoch: unmarked runs are not legacy-complete, so the
    # authoritative approval must block and detect_review_status must not be
    # written.
    root["refined_detect_runs"].attrs["palette_completion_epoch"] = 1

    with pytest.raises(RuntimeError, match="could not set authoritative refined detect run"):
        mod.main([str(zarr_path), "--state", "approved", "--reviewer", "operator2"])

    reopened = zarr.open_group(store=zarr_path, mode="r")
    parent = reopened["refined_detect_runs"]
    assert "detect_review_status" not in parent["refined_1"].attrs
    assert "authoritative_run" not in parent.attrs
    assert "detect_review_status_latest" not in parent.attrs


def test_set_detect_review_status_rejects_removed_no_latest_flag(tmp_path: Path) -> None:
    zarr_path = _make_zarr(tmp_path / "rec.zarr")
    with pytest.raises(SystemExit):
        mod.main([str(zarr_path), "--no-latest"])
