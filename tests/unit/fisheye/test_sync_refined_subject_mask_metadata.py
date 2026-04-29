from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import zarr

from fisheye.shared.detect_reason_codec import read_reason_labels
from fisheye.tune import refined_subject_mask_review as review_mod
from fisheye.utils import sync_refined_subject_mask_metadata as cli_mod
from fisheye.utils import write_refined_subject_mask_edit as write_cli_mod

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")


def _build_subject_review_archive(zarr_path: Path) -> None:
    root = zarr.open_group(str(zarr_path), mode="w")

    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "crop_001"
    crop = crop_parent.create_group("crop_001")
    roi_images = np.zeros((2, 8, 8), dtype=np.uint8)
    roi_images[0, 1:7, 1:7] = 70
    roi_images[1, 2:6, 2:6] = 120
    crop.create_array("roi_images", data=roi_images)

    subject_parent = root.create_group("subject_mask_runs")
    subject_parent.attrs["latest"] = "subject_masks_001"
    subject = subject_parent.create_group("subject_masks_001")
    subject.attrs["source_crop_run"] = "crop_001"
    subject.attrs["method"] = "subject_mask_threshold_lr_v1"
    subject.attrs["mask_labels"] = ["subject_body", "eye_left", "eye_right", "swim_bladder"]
    subject.attrs["label_schema_id"] = "subject_v1_lr"
    subject.attrs["source_keypoint_group"] = "refined_keypoints_runs"
    subject.attrs["source_keypoints_run"] = "refined_kp_001"
    subject.create_array("detection_source", data=np.zeros((2,), dtype=np.int8))
    subject.create_array("frame_indices", data=np.asarray([10, 11], dtype=np.int32))
    subject.create_array("detection_indices", data=np.asarray([0, 1], dtype=np.int32))
    subject.create_array("frame_counts", data=np.asarray([1, 1], dtype=np.int32))
    subject.create_array("available_channels", data=np.asarray([True, True, True, False], dtype=np.bool_))

    masks = np.zeros((2, 4, 8, 8), dtype=np.uint8)
    masks[0, 0, 1:7, 1:7] = 1
    masks[1, 0, 2:6, 2:6] = 1
    subject.create_array("masks_roi", data=masks)

    metrics = subject.create_group("metrics")
    metrics.create_array(
        "mask_present",
        data=np.asarray(
            [
                [True, False, False, False],
                [True, False, False, False],
            ],
            dtype=np.bool_,
        ),
    )


def test_sync_refined_subject_mask_metadata_updates_touched_component(tmp_path: Path) -> None:
    zarr_path = tmp_path / "subject_review.zarr"
    _build_subject_review_archive(zarr_path)

    root = zarr.open_group(str(zarr_path), mode="a")
    _source, refined = review_mod.prepare_refined_subject_run(
        root,
        subject_run="subject_masks_001",
        refined_run="refined_subject_masks_001",
        components=("subject_body", "swim_bladder"),
    )

    edited = np.asarray(refined.group["masks_roi"][0], dtype=np.uint8)
    edited[0, 1:7, 1:7] = 0
    refined.group["masks_roi"][0] = edited

    summary = review_mod.sync_refined_subject_mask_metadata(
        zarr_path,
        refined_run="refined_subject_masks_001",
        component_name="subject_body",
        roi_indices=[0],
    )

    assert summary["status"] == "updated"
    assert summary["component_name"] == "subject_body"
    assert summary["source_subject_mask_run"] == "subject_masks_001"
    assert summary["roi_indices"] == [0]
    assert summary["roi_count"] == 1
    assert summary["changed_roi_count"] == 1
    assert summary["noop_roi_count"] == 0
    assert summary["updated_at_utc"]

    run = root["refined_subject_masks_runs"]["refined_subject_masks_001"]
    np.testing.assert_array_equal(
        np.asarray(run["metrics/mask_present"][0], dtype=bool),
        np.asarray([False, False], dtype=bool),
    )
    np.testing.assert_allclose(
        np.asarray(run["metrics/area_px"][0], dtype=np.float32),
        np.asarray([0.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(run["metrics/centroid_xy"][0], dtype=np.float32),
        np.asarray([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(run["metrics/centroid_valid"][0], dtype=bool),
        np.asarray([False, False], dtype=bool),
    )
    np.testing.assert_allclose(
        np.asarray(run["metrics/bbox_xyxy"][0], dtype=np.float32),
        np.asarray([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(run["metrics/bbox_valid"][0], dtype=bool),
        np.asarray([False, False], dtype=bool),
    )
    np.testing.assert_array_equal(
        np.asarray(run["edit_applied"][0], dtype=bool),
        np.asarray([True, False], dtype=bool),
    )

    body_group = run["components/subject_body"]
    swim_group = run["components/swim_bladder"]
    body_provenance = body_group["provenance"]
    swim_provenance = swim_group["provenance"]
    assert bool(np.asarray(body_group["edit_applied"][0], dtype=bool)) is True
    assert bool(np.asarray(swim_group["edit_applied"][0], dtype=bool)) is False
    assert bool(np.asarray(body_group["manual_override"][0], dtype=bool)) is True
    assert bool(np.asarray(swim_group["manual_override"][0], dtype=bool)) is False
    assert bool(np.asarray(body_group["source_row_stale"][0], dtype=bool)) is False
    body_reasons = read_reason_labels(body_group)
    swim_reasons = read_reason_labels(swim_group)
    assert body_reasons is not None
    assert swim_reasons is not None
    assert body_reasons[0] == "manual_correction"
    assert swim_reasons[0] == "clean"
    assert int(np.asarray(body_group["metrics/component_count"][0], dtype=np.int32)) == 0
    assert float(np.asarray(body_group["metrics/largest_component_fraction"][0], dtype=np.float32)) == 0.0
    assert int(np.asarray(body_group["metrics/hole_count"][0], dtype=np.int32)) == 0
    assert float(np.asarray(body_group["metrics/hole_area_fraction"][0], dtype=np.float32)) == 0.0
    assert float(np.asarray(body_group["metrics/sigma_noise"][0], dtype=np.float32)) == 0.0
    assert float(np.asarray(body_group["metrics/curvature_var"][0], dtype=np.float32)) == 0.0
    assert float(np.asarray(body_group["metrics/ipr"][0], dtype=np.float32)) == 0.0
    assert float(np.asarray(body_group["metrics/solidity"][0], dtype=np.float32)) == 0.0
    assert body_provenance.attrs["last_update_stage"] == review_mod.REFINED_SUBJECT_STAGE_NAME
    assert body_provenance.attrs["last_update_mode"] == "interactive"
    assert body_provenance.attrs["last_update_method"] == review_mod.REFINED_SUBJECT_SYNC_METHOD
    assert body_provenance.attrs["updated_at_utc"] == run.attrs["updated_at_utc"]
    assert swim_provenance.attrs["last_update_mode"] == "create"


def test_sync_refined_subject_mask_metadata_batches_rows_and_tracks_noops(tmp_path: Path) -> None:
    zarr_path = tmp_path / "subject_review.zarr"
    _build_subject_review_archive(zarr_path)

    root = zarr.open_group(str(zarr_path), mode="a")
    _source, refined = review_mod.prepare_refined_subject_run(
        root,
        subject_run="subject_masks_001",
        refined_run="refined_subject_masks_001",
        components=("subject_body", "swim_bladder"),
    )

    edited0 = np.asarray(refined.group["masks_roi"][0], dtype=np.uint8)
    edited0[0, 1:7, 1:7] = 0
    refined.group["masks_roi"][0] = edited0

    edited1 = np.asarray(refined.group["masks_roi"][1], dtype=np.uint8)
    refined.group["masks_roi"][1] = edited1

    summary = review_mod.sync_refined_subject_mask_metadata(
        zarr_path,
        refined_run="refined_subject_masks_001",
        component_name="subject_body",
        roi_indices=[0, 1],
    )

    assert summary["status"] == "updated"
    assert summary["roi_indices"] == [0, 1]
    assert summary["roi_count"] == 2
    assert summary["changed_roi_count"] == 1
    assert summary["noop_roi_count"] == 1

    run = root["refined_subject_masks_runs"]["refined_subject_masks_001"]
    body_group = run["components/subject_body"]
    body_reasons = read_reason_labels(body_group)
    assert body_reasons is not None
    assert body_reasons.tolist() == ["manual_correction", "copied_from_source"]
    np.testing.assert_array_equal(
        np.asarray(body_group["manual_override"][:], dtype=bool),
        np.asarray([True, False], dtype=bool),
    )


def test_check_refined_subject_source_updates_auto_syncs_unedited_rows(tmp_path: Path) -> None:
    zarr_path = tmp_path / "subject_review.zarr"
    _build_subject_review_archive(zarr_path)

    root = zarr.open_group(str(zarr_path), mode="a")
    _source, refined = review_mod.prepare_refined_subject_run(
        root,
        subject_run="subject_masks_001",
        refined_run="refined_subject_masks_001",
        components=("subject_body", "swim_bladder"),
    )

    subject = root["subject_mask_runs"]["subject_masks_001"]
    source_masks = np.asarray(subject["masks_roi"][:], dtype=np.uint8)
    source_masks[0, 0, 1:7, 1:7] = 0
    source_masks[0, 0, 2:6, 2:6] = 1
    subject["masks_roi"][:] = source_masks

    summary = review_mod.check_refined_subject_source_updates(
        zarr_path,
        refined_run="refined_subject_masks_001",
        component_name="subject_body",
        roi_indices=[0],
    )

    assert summary["status"] == "updated"
    assert summary["source_changed_roi_count"] == 1
    assert summary["auto_synced_roi_count"] == 1
    assert summary["stale_marked_roi_count"] == 0
    assert summary["auto_synced_roi_indices"] == [0]
    assert summary["stale_roi_indices"] == []

    run = root["refined_subject_masks_runs"]["refined_subject_masks_001"]
    np.testing.assert_array_equal(
        np.asarray(run["masks_roi"][0, 0], dtype=np.uint8),
        np.asarray(subject["masks_roi"][0, 0], dtype=np.uint8),
    )
    body_group = run["components"]["subject_body"]
    assert bool(np.asarray(body_group["manual_override"][0], dtype=bool)) is False
    assert bool(np.asarray(body_group["source_row_stale"][0], dtype=bool)) is False


def test_check_refined_subject_source_updates_marks_manual_rows_stale(tmp_path: Path) -> None:
    zarr_path = tmp_path / "subject_review.zarr"
    _build_subject_review_archive(zarr_path)

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

    assert summary["status"] == "updated"
    assert summary["source_changed_roi_count"] == 1
    assert summary["auto_synced_roi_count"] == 0
    assert summary["stale_marked_roi_count"] == 1
    assert summary["stale_roi_indices"] == [0]
    assert summary["stale_total"] == 1

    run = root["refined_subject_masks_runs"]["refined_subject_masks_001"]
    body_group = run["components"]["subject_body"]
    assert bool(np.asarray(body_group["manual_override"][0], dtype=bool)) is True
    assert bool(np.asarray(body_group["source_row_stale"][0], dtype=bool)) is True
    assert body_group.attrs["source_update_pending_rows"] == [0]
    stale_payload = dict(run.attrs.get("source_subject_mask_stale") or {})
    assert stale_payload["state"] == "stale"
    assert stale_payload["reason"] == "source_subject_mask_rows_changed"
    assert stale_payload["roi_indices"] == [0]
    assert stale_payload["component_names"] == ["subject_body"]
    assert stale_payload["components"]["subject_body"]["roi_indices"] == [0]
    assert stale_payload["components"]["subject_body"]["source_subject_mask_run"] == "subject_masks_001"
    np.testing.assert_array_equal(
        np.asarray(run["masks_roi"][0, 0], dtype=np.uint8),
        np.asarray(edited[0], dtype=np.uint8),
    )
    payload = dict(run.attrs.get("component_review_statuses") or {}).get("subject_body", {})
    assert payload["state"] == "needs_review"


def test_save_refined_subject_roi_clears_active_run_level_stale_payload(tmp_path: Path) -> None:
    zarr_path = tmp_path / "subject_review.zarr"
    _build_subject_review_archive(zarr_path)

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

    review_mod.save_refined_subject_roi(
        source=source,
        refined=refined,
        roi_idx=0,
        edited_masks=np.asarray(subject["masks_roi"][0], dtype=np.uint8)[[0, 3]],
    )

    run = root["refined_subject_masks_runs"]["refined_subject_masks_001"]
    body_group = run["components"]["subject_body"]
    assert bool(np.asarray(body_group["source_row_stale"][0], dtype=bool)) is False
    assert body_group.attrs["source_update_pending_rows"] == []
    assert "source_subject_mask_stale" not in run.attrs


def test_sync_refined_subject_mask_metadata_cli_emits_json_summary(tmp_path: Path, capsys) -> None:
    zarr_path = tmp_path / "subject_review.zarr"
    _build_subject_review_archive(zarr_path)

    root = zarr.open_group(str(zarr_path), mode="a")
    _source, refined = review_mod.prepare_refined_subject_run(
        root,
        subject_run="subject_masks_001",
        refined_run="refined_subject_masks_001",
        components=("subject_body", "swim_bladder"),
    )

    edited = np.asarray(refined.group["masks_roi"][0], dtype=np.uint8)
    edited[0, 1:7, 1:7] = 0
    refined.group["masks_roi"][0] = edited

    rc = cli_mod.main(
        [
            "--zarr-path",
            str(zarr_path),
            "--refined-run",
            "refined_subject_masks_001",
            "--component-name",
            "subject_body",
            "--dataset",
            "labels/refined_subject_masks/refined_subject_masks_001/subject_body",
            "--roi-indices",
            "0,0",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["status"] == "updated"
    assert payload["dataset"] == "labels/refined_subject_masks/refined_subject_masks_001/subject_body"
    assert payload["roi_indices"] == [0]
    assert payload["changed_roi_count"] == 1
    assert payload["noop_roi_count"] == 0


def test_sync_refined_subject_mask_metadata_cli_can_check_source_updates(tmp_path: Path, capsys) -> None:
    zarr_path = tmp_path / "subject_review.zarr"
    _build_subject_review_archive(zarr_path)

    root = zarr.open_group(str(zarr_path), mode="a")
    _source, _refined = review_mod.prepare_refined_subject_run(
        root,
        subject_run="subject_masks_001",
        refined_run="refined_subject_masks_001",
        components=("subject_body", "swim_bladder"),
    )
    subject = root["subject_mask_runs"]["subject_masks_001"]
    source_masks = np.asarray(subject["masks_roi"][:], dtype=np.uint8)
    source_masks[0, 0, 1:7, 1:7] = 0
    source_masks[0, 0, 2:6, 2:6] = 1
    subject["masks_roi"][:] = source_masks

    rc = cli_mod.main(
        [
            "--zarr-path",
            str(zarr_path),
            "--refined-run",
            "refined_subject_masks_001",
            "--component-name",
            "subject_body",
            "--roi-indices",
            "0",
            "--check-source-updates",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["status"] == "updated"
    assert payload["auto_synced_roi_count"] == 1
    assert payload["stale_marked_roi_count"] == 0


def test_write_refined_subject_mask_edit_owns_pixels_and_metadata(tmp_path: Path) -> None:
    zarr_path = tmp_path / "subject_review.zarr"
    _build_subject_review_archive(zarr_path)

    root = zarr.open_group(str(zarr_path), mode="a")
    review_mod.prepare_refined_subject_run(
        root,
        subject_run="subject_masks_001",
        refined_run="refined_subject_masks_001",
        components=("subject_body", "swim_bladder"),
    )

    edited_mask = np.zeros((8, 8), dtype=np.uint8)
    edited_mask[2:5, 2:5] = 1
    summary = review_mod.write_refined_subject_mask_edit(
        zarr_path,
        refined_run="refined_subject_masks_001",
        component_name="subject_body",
        roi_index=0,
        mask=edited_mask,
        reason="crimson_refined_subject_mask_edit",
        validate=True,
    )

    assert summary["ok"] is True
    assert summary["status"] == "updated"
    assert summary["roi_index"] == 0
    assert summary["component_name"] == "subject_body"
    assert summary["row_revision_before"] == 0
    assert summary["row_revision_after"] == 1
    assert summary["edit_applied"] is True
    assert summary["mask_changed"] is True
    assert summary["contour_points"] > 0
    assert summary["updated_at_utc"]

    run = root["refined_subject_masks_runs"]["refined_subject_masks_001"]
    np.testing.assert_array_equal(np.asarray(run["masks_roi"][0, 0], dtype=np.uint8), edited_mask)
    assert float(np.asarray(run["metrics/area_px"][0, 0], dtype=np.float32)) == 9.0
    assert bool(np.asarray(run["metrics/mask_present"][0, 0], dtype=bool)) is True
    body_group = run["components"]["subject_body"]
    assert body_group.attrs["last_row_update_reason"] == "crimson_refined_subject_mask_edit"
    assert body_group["contours"]["len"][0] == summary["contour_points"]
    assert body_group["provenance"].attrs["last_update_method"] == review_mod.REFINED_SUBJECT_WRITEBACK_METHOD


def test_write_refined_subject_mask_edit_cli_emits_json_and_noops_same_pixels(
    tmp_path: Path,
    capsys,
) -> None:
    zarr_path = tmp_path / "subject_review.zarr"
    _build_subject_review_archive(zarr_path)

    root = zarr.open_group(str(zarr_path), mode="a")
    review_mod.prepare_refined_subject_run(
        root,
        subject_run="subject_masks_001",
        refined_run="refined_subject_masks_001",
        components=("subject_body", "swim_bladder"),
    )
    mask_path = tmp_path / "body.npy"
    edited_mask = np.zeros((8, 8), dtype=np.uint8)
    edited_mask[3:6, 3:6] = 1
    np.save(mask_path, edited_mask)

    rc = write_cli_mod.main(
        [
            "--zarr-path",
            str(zarr_path),
            "--refined-run",
            "refined_subject_masks_001",
            "--component-name",
            "subject_body",
            "--roi-index",
            "0",
            "--mask-path",
            str(mask_path),
            "--validate",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["ok"] is True
    assert payload["status"] == "updated"
    assert payload["row_revision_before"] == 0
    assert payload["row_revision_after"] == 1
    assert payload["mask_path"] == str(mask_path)

    rc = write_cli_mod.main(
        [
            "--zarr-path",
            str(zarr_path),
            "--refined-run",
            "refined_subject_masks_001",
            "--component-name",
            "subject_body",
            "--roi-index",
            "0",
            "--mask-path",
            str(mask_path),
            "--validate",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["status"] == "noop"
    assert payload["row_revision_before"] == 1
    assert payload["row_revision_after"] == 1
