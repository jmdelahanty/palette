from __future__ import annotations

from pathlib import Path

import pytest
import zarr

from fisheye.utils import resolve_eye_mask_stale as mod


def _make_training_zarr(path: Path, *, keypoint_run: str = "refined_keypoints_001") -> None:
    root = zarr.open_group(str(path), mode="w")
    root.attrs["zarr_use"] = "training"

    kp_parent = root.create_group("refined_keypoints_runs")
    kp_parent.create_group(keypoint_run)
    kp_parent.attrs["latest"] = keypoint_run

    subject_parent = root.create_group("refined_subject_masks_runs")
    subject_run = subject_parent.create_group("refined_subject_masks_001")
    subject_parent.attrs["latest"] = "refined_subject_masks_001"
    subject_run.attrs.update(
        {
            "source_keypoints_run": keypoint_run,
            "source_keypoint_group": "refined_keypoints_runs",
            "source_subject_mask_stale": {
                "state": "stale",
                "reason": "keypoint_manual_correction",
                "timestamp_utc": "2026-02-12T00:00:00+00:00",
            },
        }
    )


def test_main_dry_run_reports_would_resolve(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    zarr_path = tmp_path / "rec_training.zarr"
    _make_training_zarr(zarr_path)

    rc = mod.main([str(zarr_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "would_resolve=1" in out
    assert "Dry run summary" in out

    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    payload = dict(root["refined_subject_masks_runs"]["refined_subject_masks_001"].attrs["source_subject_mask_stale"])
    assert payload["state"] == "stale"


def test_main_apply_resolves_stale(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    zarr_path = tmp_path / "rec_training.zarr"
    _make_training_zarr(zarr_path)

    rc = mod.main(
        [
            str(zarr_path),
            "--apply",
            "--resolution",
            "manual_accept_after_keypoint_nudge_preserve_masks",
            "--reviewer",
            "tester",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "resolved=1" in out
    assert "Apply summary" in out

    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    payload = dict(root["refined_subject_masks_runs"]["refined_subject_masks_001"].attrs["source_subject_mask_stale"])
    assert payload["state"] == "resolved"
    assert payload["resolution"] == "manual_accept_after_keypoint_nudge_preserve_masks"
    assert payload["resolved_by"] == "tester"
    assert "resolved_at_utc" in payload


def test_main_zarr_use_filter_skips_nonmatching(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    zarr_path = tmp_path / "rec_analysis.zarr"
    _make_training_zarr(zarr_path)
    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    root.attrs["zarr_use"] = "analysis"

    rc = mod.main([str(zarr_path), "--zarr-use", "training"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "filtered_zarr_use=1" in out
