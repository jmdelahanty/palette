"""Tests for keypoint run resolution audit utility."""

from pathlib import Path
import sys

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.utils.audit_keypoint_run_resolution import audit_keypoint_resolution, main


def _seed_archive(
    path: Path,
    *,
    include_cross_method_review: bool,
) -> None:
    root = zarr.open_group(str(path), mode="w")

    kp_parent = root.create_group("keypoints_runs")

    kp_yolo = kp_parent.create_group("kp_yolo_001")
    kp_yolo.attrs["method"] = "yolo_pose"
    kp_yolo.attrs["keypoints_timestamp_utc"] = "2026-02-07T00:00:00+00:00"
    kp_yolo.create_array(
        "keypoints_roi",
        data=np.zeros((2, 3, 2), dtype=np.float32),
        chunks=(1, 3, 2),
    )

    kp_trad = kp_parent.create_group("kp_trad_002")
    kp_trad.attrs["method"] = "traditional_pose"
    kp_trad.attrs["keypoints_timestamp_utc"] = "2026-02-08T00:00:00+00:00"
    kp_trad.create_array(
        "keypoints_roi",
        data=np.zeros((2, 3, 2), dtype=np.float32),
        chunks=(1, 3, 2),
    )

    kp_parent.attrs["latest"] = "kp_trad_002"

    refined_parent = root.create_group("refined_keypoints_runs")
    refined_parent.attrs["latest"] = "refined_pose_001"
    refined = refined_parent.create_group("refined_pose_001")
    refined.attrs["created_utc"] = "2026-02-08T01:00:00+00:00"
    refined.attrs["keypoint_review_status"] = {
        "state": "approved",
        "intended_use": "training",
    }
    refined.attrs["source_keypoints_run"] = "kp_yolo_001" if include_cross_method_review else "kp_trad_002"


def test_audit_keypoint_resolution_detects_cross_method_review(tmp_path: Path) -> None:
    zarr_path = tmp_path / "sample.zarr"
    _seed_archive(zarr_path, include_cross_method_review=True)

    rows = audit_keypoint_resolution(
        [tmp_path],
        recursive=True,
        selector="latest_traditional",
        required_state="approved",
        required_intended_use="training",
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "cross_method_review"
    assert row["resolved_keypoint_run"] == "kp_trad_002"
    assert row["reviewed_source_keypoint_run"] == "kp_yolo_001"


def test_audit_keypoint_resolution_reports_aligned_when_methods_match(tmp_path: Path) -> None:
    zarr_path = tmp_path / "sample.zarr"
    _seed_archive(zarr_path, include_cross_method_review=False)

    rows = audit_keypoint_resolution(
        [tmp_path],
        recursive=True,
        selector="latest_traditional",
        required_state="approved",
        required_intended_use="training",
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "aligned"
    assert row["resolved_keypoint_run"] == "kp_trad_002"
    assert row["reviewed_source_keypoint_run"] == "kp_trad_002"


def test_main_strict_returns_nonzero_when_issues_found(tmp_path: Path) -> None:
    zarr_path = tmp_path / "sample.zarr"
    _seed_archive(zarr_path, include_cross_method_review=True)

    rc = main(
        [
            str(tmp_path),
            "--recursive",
            "--selector",
            "latest_traditional",
            "--strict",
        ]
    )
    assert rc == 2


def test_audit_keypoint_resolution_surfaces_review_status_divergence(monkeypatch, tmp_path: Path) -> None:
    zarr_path = tmp_path / "sample.zarr"
    _seed_archive(zarr_path, include_cross_method_review=False)

    from fisheye.utils import prepare_keypoint_training_from_registry as prep_pose

    original = prep_pose._resolve_review_status_sources

    def _with_conflict(refined_group, *, zarr_path, refined_parent_name, refined_run_name):
        result = original(
            refined_group,
            zarr_path=zarr_path,
            refined_parent_name=refined_parent_name,
            refined_run_name=refined_run_name,
        )
        if refined_run_name == "refined_pose_001":
            result = dict(result)
            result["disk"] = {"state": "pending", "intended_use": "full_recording"}
            result["divergence"] = "conflict"
        return result

    monkeypatch.setattr(prep_pose, "_resolve_review_status_sources", _with_conflict)

    rows = audit_keypoint_resolution(
        [tmp_path],
        recursive=True,
        selector="latest_traditional",
        required_state="approved",
        required_intended_use="training",
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "aligned"
    assert row["resolved_review_divergence"] == "conflict"

    rc = main([str(tmp_path), "--recursive", "--selector", "latest_traditional", "--strict"])
    assert rc == 2
