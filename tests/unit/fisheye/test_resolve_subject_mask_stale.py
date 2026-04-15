from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.tune import refined_subject_mask_review as review_mod
from fisheye.utils import resolve_subject_mask_stale as mod


def _make_subject_review_zarr(path: Path) -> None:
    root = zarr.open_group(str(path), mode="w")
    root.attrs["zarr_use"] = "training"

    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "crop_001"
    crop = crop_parent.create_group("crop_001")
    crop.create_array("roi_images", data=np.zeros((1, 8, 8), dtype=np.uint8))

    subject_parent = root.create_group("subject_mask_runs")
    subject_parent.attrs["latest"] = "subject_masks_001"
    subject = subject_parent.create_group("subject_masks_001")
    subject.attrs["source_crop_run"] = "crop_001"
    subject.attrs["method"] = "subject_mask_threshold_lr_v1"
    subject.attrs["mask_labels"] = ["subject_body", "eye_left", "eye_right", "swim_bladder"]
    subject.attrs["label_schema_id"] = "subject_v1_lr"
    subject.create_array("detection_source", data=np.zeros((1,), dtype=np.int8))
    subject.create_array("frame_indices", data=np.asarray([10], dtype=np.int32))
    subject.create_array("detection_indices", data=np.asarray([0], dtype=np.int32))
    subject.create_array("frame_counts", data=np.asarray([1], dtype=np.int32))
    subject.create_array("available_channels", data=np.asarray([True, True, True, False], dtype=np.bool_))
    masks = np.zeros((1, 4, 8, 8), dtype=np.uint8)
    masks[0, 0, 1:7, 1:7] = 1
    subject.create_array("masks_roi", data=masks)
    metrics = subject.create_group("metrics")
    metrics.create_array("mask_present", data=np.asarray([[True, False, False, False]], dtype=np.bool_))


def _mark_subject_body_stale(zarr_path: Path) -> None:
    root = zarr.open_group(str(zarr_path), mode="a")
    source, refined = review_mod.prepare_refined_subject_run(
        root,
        subject_run="subject_masks_001",
        refined_run="refined_subject_masks_001",
        components=("subject_body", "swim_bladder"),
    )

    edited = np.asarray(refined.group["masks_roi"][0], dtype=np.uint8)
    edited[0, 1:7, 1:7] = 0
    review_mod.save_refined_subject_roi(
        source=source,
        refined=refined,
        roi_idx=0,
        edited_masks=edited,
    )

    subject = root["subject_mask_runs"]["subject_masks_001"]
    source_masks = np.asarray(subject["masks_roi"][:], dtype=np.uint8)
    source_masks[0, 0, 2:6, 2:6] = 0
    source_masks[0, 0, 3:5, 3:5] = 1
    subject["masks_roi"][:] = source_masks

    summary = review_mod.check_refined_subject_source_updates(
        zarr_path,
        refined_run="refined_subject_masks_001",
        component_name="subject_body",
        roi_indices=[0],
    )
    assert summary["stale_marked_roi_count"] == 1


def test_main_dry_run_reports_would_resolve(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    zarr_path = tmp_path / "rec_training.zarr"
    _make_subject_review_zarr(zarr_path)
    _mark_subject_body_stale(zarr_path)

    rc = mod.main([str(zarr_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "would_resolve=1" in out
    assert "Dry run summary" in out

    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    run = root["refined_subject_masks_runs"]["refined_subject_masks_001"]
    payload = dict(run.attrs["source_subject_mask_stale"])
    assert payload["state"] == "stale"
    body_group = run["components"]["subject_body"]
    assert bool(np.asarray(body_group["source_row_stale"][0], dtype=bool)) is True


def test_main_apply_resolves_subject_mask_stale(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    zarr_path = tmp_path / "rec_training.zarr"
    _make_subject_review_zarr(zarr_path)
    _mark_subject_body_stale(zarr_path)

    rc = mod.main(
        [
            str(zarr_path),
            "--apply",
            "--resolution",
            "manual_accept_after_subject_mask_source_update_preserve_masks",
            "--reviewer",
            "tester",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "resolved=1" in out
    assert "Apply summary" in out

    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    run = root["refined_subject_masks_runs"]["refined_subject_masks_001"]
    payload = dict(run.attrs["source_subject_mask_stale"])
    assert payload["state"] == "resolved"
    assert payload["resolution"] == "manual_accept_after_subject_mask_source_update_preserve_masks"
    assert payload["resolved_by"] == "tester"
    assert "resolved_at_utc" in payload
    assert "stale_timestamp_utc" in payload
    body_group = run["components"]["subject_body"]
    assert bool(np.asarray(body_group["source_row_stale"][0], dtype=bool)) is False
    assert body_group.attrs["source_update_pending_rows"] == []


def test_main_zarr_use_filter_skips_nonmatching(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    zarr_path = tmp_path / "rec_analysis.zarr"
    _make_subject_review_zarr(zarr_path)
    _mark_subject_body_stale(zarr_path)
    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    root.attrs["zarr_use"] = "analysis"

    rc = mod.main([str(zarr_path), "--zarr-use", "training"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "filtered_zarr_use=1" in out
