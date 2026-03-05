from pathlib import Path

import numpy as np
import zarr

from fisheye.utils import backfill_keypoint_auto_review_policy as mod


def _write_refined_run(
    zarr_path: Path,
    *,
    run_name: str = "refined_001",
    with_status: bool,
    method: str = "algorithmic",
    intended_use: str = "full_recording",
) -> zarr.Group:
    root = zarr.open_group(str(zarr_path), mode="w")
    refined_parent = root.create_group("refined_keypoints_runs")
    refined_parent.attrs["latest"] = run_name
    run = refined_parent.create_group(run_name)
    run.attrs["source_keypoints_run"] = "keypoints_001"
    run.create_array("keypoints_roi", data=np.zeros((4, 3, 2), dtype=np.float32), overwrite=True)
    run.create_array("refined_success", data=np.ones((4,), dtype=bool), overwrite=True)
    run.create_array("usable_keypoints", data=np.ones((4,), dtype=bool), overwrite=True)
    run.create_array("confidence_valid", data=np.ones((4,), dtype=bool), overwrite=True)
    run.create_array("geometry_valid", data=np.ones((4,), dtype=bool), overwrite=True)
    if with_status:
        run.attrs["keypoint_review_status"] = {
            "state": "approved",
            "method": method,
            "intended_use": intended_use,
            "timestamp_utc": "2026-03-02T00:00:00+00:00",
        }
    return run


def test_backfill_patches_existing_algorithmic_status(tmp_path: Path) -> None:
    zarr_path = tmp_path / "rec_analysis.zarr"
    _write_refined_run(zarr_path, with_status=True, method="algorithmic")

    rc = mod.main([str(zarr_path), "--apply"])
    assert rc == 0

    root = zarr.open_group(str(zarr_path), mode="r")
    payload = root["refined_keypoints_runs/refined_001"].attrs["keypoint_review_status"]
    assert payload["method"] == "algorithmic"
    assert payload["auto_review"]["policy_id"] == "keypoint_auto_review_v1"
    assert payload["auto_review"]["policy_version"] == 1


def test_backfill_skips_manual_status(tmp_path: Path) -> None:
    zarr_path = tmp_path / "rec_analysis.zarr"
    _write_refined_run(zarr_path, with_status=True, method="manual")

    rc = mod.main([str(zarr_path), "--apply"])
    assert rc == 0

    root = zarr.open_group(str(zarr_path), mode="r")
    payload = root["refined_keypoints_runs/refined_001"].attrs["keypoint_review_status"]
    assert payload["method"] == "manual"
    assert "auto_review" not in payload


def test_backfill_write_missing_creates_algorithmic_status(tmp_path: Path) -> None:
    zarr_path = tmp_path / "rec_analysis.zarr"
    _write_refined_run(zarr_path, with_status=False)

    rc = mod.main([str(zarr_path), "--write-missing", "--apply"])
    assert rc == 0

    root = zarr.open_group(str(zarr_path), mode="r")
    payload = root["refined_keypoints_runs/refined_001"].attrs["keypoint_review_status"]
    assert payload["method"] == "algorithmic"
    assert payload["intended_use"] == "full_recording"
    assert payload["auto_review"]["policy_id"] == "keypoint_auto_review_v1"


def test_backfill_dry_run_does_not_write(tmp_path: Path) -> None:
    zarr_path = tmp_path / "rec_analysis.zarr"
    _write_refined_run(zarr_path, with_status=True, method="algorithmic")

    rc = mod.main([str(zarr_path)])
    assert rc == 0

    root = zarr.open_group(str(zarr_path), mode="r")
    payload = root["refined_keypoints_runs/refined_001"].attrs["keypoint_review_status"]
    assert "auto_review" not in payload
