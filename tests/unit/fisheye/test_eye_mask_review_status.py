from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import zarr

from fisheye.tune import eye_mask_failure_review as failure_review
from fisheye.tune import eye_mask_review as review_mod


def test_apply_review_status_writes_eye_mask_review_attrs(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    refined_parent = root.create_group("refined_eye_masks_runs")
    refined = refined_parent.create_group("refined_eye_masks_001")
    refined.attrs.update(
        {
            "source_eye_masks_run": "eye_masks_001",
            "source_keypoints_run": "refined_keypoints_001",
            "source_keypoint_group": "refined_keypoints_runs",
        }
    )

    payload = failure_review._apply_review_status(
        refined_parent,
        "refined_eye_masks_001",
        refined,
        state="approved",
        method="manual",
        intended_use="training",
        reviewer="tester",
        notes="looks good",
    )

    status = dict(refined.attrs["eye_mask_review_status"])
    assert status["state"] == "approved"
    assert status["method"] == "manual"
    assert status["intended_use"] == "training"
    assert status["reviewer"] == "tester"
    assert status["notes"] == "looks good"
    assert status["source_eye_masks_run"] == "eye_masks_001"
    assert status["source_keypoints_run"] == "refined_keypoints_001"
    assert status["source_keypoint_group"] == "refined_keypoints_runs"
    assert "timestamp_utc" in status
    assert status["timestamp"] == status["timestamp_utc"]

    assert payload["state"] == "approved"
    assert refined_parent.attrs["eye_mask_review_status_latest"] == "refined_eye_masks_001"


def test_apply_review_status_rejects_derived_compat_run(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_compat.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    refined_parent = root.create_group("refined_eye_masks_runs")
    refined = refined_parent.create_group("refined_eye_masks_compat_001")
    refined.attrs.update(
        {
            "compatibility_role": "derived_from_refined_subject_masks",
            "source_refined_subject_masks_run": "refined_subject_masks_001",
        }
    )

    try:
        failure_review._apply_review_status(
            refined_parent,
            "refined_eye_masks_compat_001",
            refined,
            state="approved",
            method="manual",
            intended_use="training",
            reviewer="tester",
            notes=None,
        )
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected RuntimeError for derived compat run")

    assert "refined_subject_masks_runs/refined_subject_masks_001" in message
    assert "--manual --refined-run refined_eye_masks_compat_001" in message


def test_run_manual_review_forwards_review_status_args(
    monkeypatch,
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "recording.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    refined_parent = root.create_group("refined_eye_masks_runs")
    refined_parent.create_group("refined_eye_masks_001")
    refined_parent.attrs["latest"] = "refined_eye_masks_001"

    seen: Dict[str, Any] = {}

    def _fake_launch_review(_zarr_path: str, **kwargs: Any) -> Dict[str, str]:
        seen.update(kwargs)
        return {
            "subject_run": "subject_masks_from_refined_eye_masks_001",
            "refined_subject_run": "refined_subject_masks_from_refined_eye_masks_001",
        }

    monkeypatch.setattr(review_mod, "_launch_unified_subject_review", _fake_launch_review)
    monkeypatch.setattr(
        review_mod,
        "_update_postprocess_summary",
        lambda _root, _refined, print_summary=True: {"status": "ok"},
    )

    result = review_mod.run_manual_review(
        str(zarr_path),
        refined_run="refined_eye_masks_001",
        crop_run="crop_001",
        frame_flag_file="eye_mask_frame_flags.json",
        review_state="pending",
        review_method="spotcheck",
        review_intended_use="full_recording",
        reviewer="reviewer_a",
        review_notes="needs another pass",
    )

    assert result["status"] == "ok"
    assert result["review_surface"] == "refined_subject_masks_runs"
    assert result["subject_run"] == "subject_masks_from_refined_eye_masks_001"
    assert result["refined_subject_run"] == "refined_subject_masks_from_refined_eye_masks_001"
    assert seen["refined_run"] == "refined_eye_masks_001"
    assert seen["crop_run"] == "crop_001"
    assert seen["frame_flag_file"] == "eye_mask_frame_flags.json"
    assert seen["review_state"] == "pending"
    assert seen["review_method"] == "spotcheck"
    assert seen["review_intended_use"] == "full_recording"
    assert seen["reviewer"] == "reviewer_a"
    assert seen["review_notes"] == "needs another pass"


def test_run_manual_review_legacy_mode_forwards_to_eye_failure_review(
    monkeypatch,
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "recording.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    refined_parent = root.create_group("refined_eye_masks_runs")
    refined_parent.create_group("refined_eye_masks_001")
    refined_parent.attrs["latest"] = "refined_eye_masks_001"

    seen: Dict[str, Any] = {}

    def _fake_launch_review(_zarr_path: str, **kwargs: Any) -> None:
        seen.update(kwargs)

    monkeypatch.setattr("fisheye.tune.eye_mask_failure_review.launch_review", _fake_launch_review)
    monkeypatch.setattr(
        review_mod,
        "_update_postprocess_summary",
        lambda _root, _refined, print_summary=True: {"status": "ok"},
    )

    result = review_mod.run_manual_review(
        str(zarr_path),
        refined_run="refined_eye_masks_001",
        crop_run="crop_001",
        frame_flag_file="eye_mask_frame_flags.json",
        review_state="pending",
        review_method="spotcheck",
        review_intended_use="full_recording",
        reviewer="reviewer_a",
        review_notes="needs another pass",
        legacy=True,
    )

    assert result["status"] == "ok"
    assert seen["refined_run"] == "refined_eye_masks_001"
    assert seen["crop_run"] == "crop_001"
    assert seen["frame_flag_file"] == "eye_mask_frame_flags.json"
    assert seen["review_state"] == "pending"
    assert seen["review_method"] == "spotcheck"
    assert seen["review_intended_use"] == "full_recording"
    assert seen["reviewer"] == "reviewer_a"
    assert seen["review_notes"] == "needs another pass"
