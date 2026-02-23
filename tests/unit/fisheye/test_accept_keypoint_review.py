from __future__ import annotations

import json
from pathlib import Path

import zarr

from fisheye.utils import accept_keypoint_review as mod


def _make_zarr(path: Path) -> Path:
    root = zarr.open_group(store=path, mode="w")
    parent = root.create_group("refined_keypoints_runs")
    parent.attrs["latest"] = "refined_1"
    parent.create_group("refined_1")
    return path


def test_accept_keypoint_review_writes_status_and_latest(tmp_path: Path) -> None:
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
    assert "timestamp_utc" in status
    assert status["timestamp"] == status["timestamp_utc"]
    assert parent.attrs["keypoint_review_status_latest"] == "refined_1"
    assert "keypoint_review_signature" in run.attrs


def test_accept_keypoint_review_dry_run_does_not_write(tmp_path: Path) -> None:
    zarr_path = _make_zarr(tmp_path / "rec.zarr")
    rc = mod.main([str(zarr_path), "--dry-run", "--json"])
    assert rc == 0

    root = zarr.open_group(store=zarr_path, mode="r")
    run = root["refined_keypoints_runs"]["refined_1"]
    assert "keypoint_review_status" not in run.attrs


def test_accept_keypoint_review_strict_requires_reviewer_for_approved(tmp_path: Path) -> None:
    zarr_path = _make_zarr(tmp_path / "rec.zarr")
    rc = mod.main(
        [
            str(zarr_path),
            "--strict",
            "--state",
            "approved",
            "--intended-use",
            "training",
            "--json",
        ]
    )
    assert rc == 1


def test_accept_keypoint_review_json_output_contains_expected_fields(tmp_path: Path, capsys) -> None:
    zarr_path = _make_zarr(tmp_path / "rec.zarr")
    rc = mod.main(
        [
            str(zarr_path),
            "--state",
            "approved",
            "--intended-use",
            "full_recording",
            "--reviewer",
            "operator2",
            "--json",
        ]
    )
    assert rc == 0
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["refined_run"] == "refined_1"
    assert payload["state"] == "approved"
    assert payload["intended_use"] == "full_recording"
