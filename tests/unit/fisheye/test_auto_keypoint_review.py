from pathlib import Path

import numpy as np
import zarr

from fisheye.shared.detect_reason_codec import write_reason_columns
from fisheye.utils.auto_keypoint_review import AutoReviewPolicy, apply_auto_review


def _bool_series(value: bool | list[bool], total: int) -> np.ndarray:
    if isinstance(value, list):
        return np.asarray(value, dtype=bool)
    return np.full((total,), bool(value), dtype=bool)


def _add_refined_run(
    parent: zarr.Group,
    *,
    run_name: str,
    source_keypoints_run: str,
    total: int = 4,
    refined_success: bool | list[bool] = True,
    usable_keypoints: bool | list[bool] = True,
    confidence_valid: bool | list[bool] = True,
    geometry_valid: bool | list[bool] = True,
    reason_labels: list[str] | None = None,
) -> zarr.Group:
    run = parent.create_group(run_name)
    run.attrs["source_keypoints_run"] = source_keypoints_run
    run.create_array("keypoints_roi", data=np.zeros((total, 2, 2), dtype=np.float32), overwrite=True)
    run.create_array("refined_success", data=_bool_series(refined_success, total), overwrite=True)
    run.create_array("usable_keypoints", data=_bool_series(usable_keypoints, total), overwrite=True)
    run.create_array("confidence_valid", data=_bool_series(confidence_valid, total), overwrite=True)
    run.create_array("geometry_valid", data=_bool_series(geometry_valid, total), overwrite=True)
    if reason_labels is not None:
        write_reason_columns(
            run,
            np.asarray(reason_labels, dtype=object),
            chunk_size=16,
            include_reason_text=False,
            overwrite=True,
        )
    return run


def test_apply_auto_review_selects_run_by_source_keypoints_run(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    parent = root.create_group("refined_keypoints_runs")
    _add_refined_run(parent, run_name="refined_a", source_keypoints_run="keypoints_a")
    _add_refined_run(parent, run_name="refined_b", source_keypoints_run="keypoints_b")
    parent.attrs["latest"] = "refined_b"

    result = apply_auto_review(zarr_path, source_keypoints_run="keypoints_a")
    assert result["refined_run"] == "refined_a"
    assert result["state"] == "approved"
    assert result["passed"] is True

    root_after = zarr.open_group(str(zarr_path), mode="r")
    attrs = dict(root_after["refined_keypoints_runs/refined_a"].attrs)
    payload = attrs["keypoint_review_status"]
    assert payload["method"] == "algorithmic"
    assert payload["intended_use"] == "full_recording"
    assert payload["auto_review"]["result"] == "approved"
    assert root_after["refined_keypoints_runs"].attrs["keypoint_review_status_latest"] == "refined_a"


def test_apply_auto_review_policy_failure_sets_needs_review(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    parent = root.create_group("refined_keypoints_runs")
    _add_refined_run(
        parent,
        run_name="refined_a",
        source_keypoints_run="keypoints_a",
        reason_labels=["clean", "detection_issue", "clean", "clean"],
    )
    parent.attrs["latest"] = "refined_a"

    policy = AutoReviewPolicy(
        disqualifying_tags=("detection_issue",),
    )
    result = apply_auto_review(zarr_path, policy=policy)
    assert result["state"] == "needs_review"
    assert result["passed"] is False

    root_after = zarr.open_group(str(zarr_path), mode="r")
    payload = root_after["refined_keypoints_runs/refined_a"].attrs["keypoint_review_status"]
    assert payload["auto_review"]["checks"]["disqualifying_tags"]["pass"] is False


def test_apply_auto_review_dry_run_does_not_write_attrs(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    parent = root.create_group("refined_keypoints_runs")
    _add_refined_run(parent, run_name="refined_a", source_keypoints_run="keypoints_a")
    parent.attrs["latest"] = "refined_a"

    result = apply_auto_review(zarr_path, dry_run=True)
    assert result["dry_run"] is True
    assert result["state"] == "approved"
    assert result["latest_updated"] is False

    root_after = zarr.open_group(str(zarr_path), mode="r")
    attrs = dict(root_after["refined_keypoints_runs/refined_a"].attrs)
    assert "keypoint_review_status" not in attrs
    parent_attrs = dict(root_after["refined_keypoints_runs"].attrs)
    assert "keypoint_review_status_latest" not in parent_attrs


def test_apply_auto_review_does_not_overwrite_existing_manual_status(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    parent = root.create_group("refined_keypoints_runs")
    run = _add_refined_run(parent, run_name="refined_a", source_keypoints_run="keypoints_a")
    parent.attrs["latest"] = "refined_a"
    run.attrs["keypoint_review_status"] = {
        "state": "approved",
        "method": "manual",
        "intended_use": "full_recording",
    }

    result = apply_auto_review(zarr_path)
    assert result["skipped"] is True
    assert result["skip_reason"] == "existing_review_status"
    assert result["existing_review_method"] == "manual"

    root_after = zarr.open_group(str(zarr_path), mode="r")
    payload = root_after["refined_keypoints_runs/refined_a"].attrs["keypoint_review_status"]
    assert payload["method"] == "manual"
