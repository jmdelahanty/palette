from __future__ import annotations

from pathlib import Path

import zarr

from fisheye.utils import set_keypoint_review_status as mod


def _make_zarr(path: Path) -> Path:
    root = zarr.open_group(store=path, mode="w")
    parent = root.create_group("refined_keypoints_runs")
    parent.attrs["latest"] = "refined_1"
    run = parent.create_group("refined_1")
    run.attrs["source_keypoints_run"] = "kp_run_001"
    return path


def test_set_keypoint_review_status_writes_timestamp_utc_and_latest(tmp_path: Path) -> None:
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
    parent = root["refined_keypoints_runs"]
    run = parent["refined_1"]
    status = dict(run.attrs["keypoint_review_status"])

    assert status["state"] == "approved"
    assert status["method"] == "manual"
    assert status["intended_use"] == "training"
    assert status["reviewer"] == "operator1"
    assert "timestamp_utc" in status
    assert status["timestamp"] == status["timestamp_utc"]
    assert parent.attrs["keypoint_review_status_latest"] == "refined_1"
    assert "keypoint_review_signature" in run.attrs


def test_set_keypoint_review_status_preserves_existing_signature(tmp_path: Path) -> None:
    zarr_path = _make_zarr(tmp_path / "rec.zarr")
    root = zarr.open_group(store=zarr_path, mode="a")
    existing_signature = {
        "signature_version": 1,
        "source_keypoints_run": "preexisting",
        "parameters_hash": "abc123",
    }
    root["refined_keypoints_runs"]["refined_1"].attrs["keypoint_signature"] = existing_signature

    rc = mod.main([str(zarr_path), "--no-latest"])
    assert rc == 0

    root = zarr.open_group(store=zarr_path, mode="r")
    parent = root["refined_keypoints_runs"]
    run = parent["refined_1"]

    assert "keypoint_review_status_latest" not in parent.attrs
    assert dict(run.attrs["keypoint_signature"]) == existing_signature
    assert dict(run.attrs["keypoint_review_signature"]) == existing_signature
