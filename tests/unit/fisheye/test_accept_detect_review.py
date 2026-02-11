from __future__ import annotations

import json
from pathlib import Path

import zarr

from fisheye.utils import accept_detect_review as mod


def _make_zarr(path: Path, *, with_group: str = "interpolated") -> Path:
    root = zarr.open_group(store=path, mode="w")
    parent = root.create_group("refined_detect_runs")
    parent.attrs["latest"] = "refined_1"
    run = parent.create_group("refined_1")
    if with_group:
        run.create_group(with_group)
    return path


def test_accept_detect_review_writes_status_and_latest(tmp_path: Path) -> None:
    zarr_path = _make_zarr(tmp_path / "rec.zarr", with_group="interpolated")
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
    run = parent["refined_1"]
    status = dict(run.attrs["detect_review_status"])
    assert status["state"] == "approved"
    assert status["method"] == "manual"
    assert status["intended_use"] == "training"
    assert status["resolved_group"] == "interpolated"
    assert parent.attrs["detect_review_status_latest"] == "refined_1"


def test_accept_detect_review_dry_run_does_not_write(tmp_path: Path) -> None:
    zarr_path = _make_zarr(tmp_path / "rec.zarr", with_group="interpolated")
    rc = mod.main([str(zarr_path), "--dry-run", "--json"])
    assert rc == 0

    root = zarr.open_group(store=zarr_path, mode="r")
    run = root["refined_detect_runs"]["refined_1"]
    assert "detect_review_status" not in run.attrs


def test_accept_detect_review_strict_requires_reviewer_for_approved(tmp_path: Path) -> None:
    zarr_path = _make_zarr(tmp_path / "rec.zarr", with_group="interpolated")
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


def test_accept_detect_review_target_group_missing_fails(tmp_path: Path) -> None:
    zarr_path = _make_zarr(tmp_path / "rec.zarr", with_group="interpolated")
    rc = mod.main([str(zarr_path), "--target-group", "manual", "--json"])
    assert rc == 1


def test_accept_detect_review_json_output_contains_expected_fields(tmp_path: Path, capsys) -> None:
    zarr_path = _make_zarr(tmp_path / "rec.zarr", with_group="interpolated")
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
    assert payload["resolved_group"] == "interpolated"
    assert payload["state"] == "approved"
    assert payload["intended_use"] == "full_recording"
