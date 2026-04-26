from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.refinement import finalize_subject_masks as mod
from fisheye.shared.detect_reason_codec import read_reason_labels, write_reason_columns
from fisheye.tune import refined_subject_mask_review as review_mod


def _patch_refined_subject_provenance(monkeypatch) -> None:
    monkeypatch.setattr(
        review_mod,
        "get_git_info",
        lambda repo_path=None: {  # noqa: ARG005
            "commit_hash": "c" * 40,
            "short_hash": "cccccccc",
            "branch": "main",
            "is_dirty": False,
            "remote_url": "git@example.com:palette.git",
        },
    )
    monkeypatch.setattr(
        review_mod,
        "get_environment_info",
        lambda **kwargs: {  # noqa: ARG005
            "environment": {"python": "3.12"},
            "platform": {
                "hostname": "finalize-host",
                "system": "Linux",
                "release": "6.8",
                "python_version": "3.12.0",
                "machine": "x86_64",
            },
        },
    )


def _build_probability_root(store_path: Path | None = None) -> zarr.Group:
    root = zarr.open_group(str(store_path), mode="w") if store_path is not None else zarr.group()
    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "crop_001"
    crop = crop_parent.create_group("crop_001")
    crop.attrs["crop_storage_mode"] = "materialized"
    crop.attrs["crop_signature"] = {"signature_version": 2, "crop_revision": 7}
    crop.attrs["crop_revision"] = 7
    crop.attrs["detect_review_status_ref"] = "refined_detect_runs/refined_detect_001/review_status"
    crop.create_array("roi_images", data=np.zeros((2, 10, 10), dtype=np.uint8), overwrite=True)

    kp_parent = root.create_group("refined_keypoints_runs")
    kp_parent.attrs["latest"] = "refined_kp_001"
    kp = kp_parent.create_group("refined_kp_001")
    kp.attrs["keypoint_labels"] = ["swim_bladder", "eye_left", "eye_right"]
    kp.create_array(
        "keypoints_roi",
        data=np.asarray(
            [
                [[5.0, 6.0], [2.0, 2.0], [7.0, 2.0]],
                [[5.0, 6.0], [2.0, 4.0], [7.0, 4.0]],
            ],
            dtype=np.float32,
        ),
        overwrite=True,
    )
    kp.create_array("detection_success", data=np.asarray([True, True], dtype=bool), overwrite=True)

    parent = root.create_group("subject_mask_runs")
    parent.attrs["latest"] = "subject_probs_001"
    run = parent.create_group("subject_probs_001")
    run.attrs.update(
        {
            "source_crop_run": "crop_001",
            "source_crop_storage_mode": "materialized",
            "source_crop_signature": {"signature_version": 2, "crop_revision": 7},
            "source_crop_revision": 7,
            "source_detect_review_status_ref": "refined_detect_runs/refined_detect_001/review_status",
            "method": "unet_subject_masks_v1",
            "mask_labels": ["subject_body", "eyes_union", "swim_bladder"],
            "label_schema_id": "subject_v1_union_eyes",
            "source_keypoints_run": "refined_kp_001",
            "source_keypoint_run": "refined_kp_001",
            "source_keypoint_group": "refined_keypoints_runs",
            "created_at_utc": "2026-04-01T00:00:00+00:00",
            "probabilities_encoding": "linear_uint8_0_255",
            "mask_probability_threshold": 0.5,
        }
    )
    run.create_array("detection_source", data=np.asarray([0, 0], dtype=np.int8), overwrite=True)
    run.create_array("frame_indices", data=np.asarray([10, 11], dtype=np.int32), overwrite=True)
    run.create_array("detection_indices", data=np.asarray([0, 1], dtype=np.int32), overwrite=True)
    run.create_array("frame_counts", data=np.asarray([1, 1], dtype=np.int32), overwrite=True)
    run.create_array("available_channels", data=np.asarray([True, True, True], dtype=bool), overwrite=True)

    probs = np.zeros((2, 3, 10, 10), dtype=np.uint8)
    probs[:, 0, 2:9, 2:9] = 255
    probs[0, 0, 4:6, 4:6] = 0
    probs[0, 0, 0, 0] = 255
    probs[0, 1, 1:4, 1:4] = 255
    probs[0, 1, 1:4, 6:9] = 255
    probs[1, 1, 3:6, 1:4] = 255
    probs[1, 1, 3:6, 6:9] = 255
    probs[:, 2, 5:8, 4:7] = 255
    probs[0, 2, 6, 5] = 0
    run.create_array("mask_probs_roi", data=probs, overwrite=True)
    return root


def test_finalize_subject_mask_run_creates_refined_candidates_from_probabilities(monkeypatch) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    root = _build_probability_root()

    summary = mod.finalize_subject_mask_run(
        root,
        subject_run="subject_probs_001",
        refined_run="refined_subject_masks_smart_001",
        chunk_size=1,
        scheduler="threads",
        num_workers=2,
    )

    assert summary["status"] == "updated"
    assert summary["component_names"] == ["subject_body", "eye_left", "eye_right", "swim_bladder"]
    assert summary["chunk_count"] == 2
    assert summary["chunk_size"] == 1
    assert summary["metric_level"] == "cheap"
    assert summary["write_eye_geometry"] is False
    assert summary["execution_backend"] == "serial_driver"
    assert summary["dask_execution_enabled"] is False
    assert summary["dask_scheduler"] == "threads"
    assert summary["dask_num_workers"] == 2
    assert summary["timing_summary"]["chunk_count"] == 2
    assert summary["timing_summary"]["dask_scheduler"] == "threads"
    assert "finalize_subject_body" in summary["timing_summary"]["phase_seconds"]
    assert summary["review_counts"]["subject_body"]["needs_review"] >= 1

    run = root["refined_subject_masks_runs"]["refined_subject_masks_smart_001"]
    assert run.attrs["method"] == "smart_finalize_subject_masks_v1"
    assert run.attrs["finalization_semantics"] == "smart_probability_to_refined_candidate"
    assert run.attrs["smart_finalizer_chunk_count"] == 2
    assert run.attrs["smart_finalizer_chunk_size"] == 1
    assert run.attrs["smart_finalizer_metric_level"] == "cheap"
    assert run.attrs["smart_finalizer_execution_backend"] == "serial_driver"
    assert run.attrs["dask_execution_enabled"] is False
    assert run.attrs["dask_scheduler"] == "threads"
    assert run.attrs["dask_num_workers"] == 2
    assert run.attrs["smart_finalizer_timing_summary"]["chunk_count"] == 2
    assert len(run.attrs["smart_finalizer_chunk_timings"]) == 2
    assert run.attrs["eye_geometry_status"] == "deferred"
    assert run.attrs["refined_subject_mask_review_status"]["state"] == "pending"
    assert run.attrs["component_review_statuses"]["subject_body"]["state"] == "pending"
    assert run.attrs["summary_statistics"]["rows_total"] == 2

    labels = list(run.attrs["mask_labels"])
    body_idx = labels.index("subject_body")
    eye_left_idx = labels.index("eye_left")
    eye_right_idx = labels.index("eye_right")
    swim_idx = labels.index("swim_bladder")
    masks = np.asarray(run["masks_roi"][:], dtype=np.uint8)
    assert masks[0, body_idx, 4, 4] == 1
    assert masks[0, body_idx, 0, 0] == 0
    assert np.count_nonzero(masks[:, eye_left_idx]) > 0
    assert np.count_nonzero(masks[:, eye_right_idx]) > 0
    assert masks[0, swim_idx, 6, 5] == 1

    body_reasons = read_reason_labels(run["components/subject_body"])
    assert body_reasons is not None
    assert "cleanup_thresholded_probability" in str(body_reasons[0])
    assert "cleanup_closed_gaps" in str(body_reasons[0])
    assert "needs_review" in str(body_reasons[0])

    eye_left_reasons = read_reason_labels(run["components/eye_left"])
    assert eye_left_reasons is not None
    assert "assigned_from_eyes_union" in str(eye_left_reasons[0])
    assert "split_by_keypoint" in str(eye_left_reasons[0])

    provenance = run["components/subject_body/provenance"].attrs
    assert provenance["finalization_method"] == "smart_finalize_subject_masks_v1"
    assert provenance["source_binary_derivation"] == "smart_finalize(mask_probs_roi)"
    assert provenance["source_probability_path"] == "subject_mask_runs/subject_probs_001/mask_probs_roi"
    assert provenance["source_probability_threshold"] == pytest.approx(0.5)
    assert "finalization_metrics" in run["components/subject_body"]
    metrics = run["components/subject_body/finalization_metrics"]
    assert metrics.attrs["schema_id"] == "refined_subject_component_finalization_metrics_v1"
    assert np.asarray(metrics["quality_code"][:], dtype=np.int16).shape == (2,)
    component_metrics = run["components/subject_body/metrics"]
    assert component_metrics.attrs["schema_id"] == "refined_subject_component_mask_metrics_v1"
    assert component_metrics.attrs["qc_schema_id"] == "refined_subject_component_metric_qc_reasons_v1"
    assert component_metrics.attrs["qc_policy"]["component_name"] == "subject_body"
    assert component_metrics.attrs["metric_level"] == "cheap"
    assert np.isnan(np.asarray(component_metrics["sigma_noise"][:], dtype=np.float32)[0])
    assert "relations" not in run


def test_finalize_subject_mask_run_can_write_full_metrics_and_eye_geometry(monkeypatch) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    root = _build_probability_root()

    summary = mod.finalize_subject_mask_run(
        root,
        subject_run="subject_probs_001",
        refined_run="refined_subject_masks_smart_full_001",
        chunk_size=1,
        metric_level="full",
        write_eye_geometry=True,
    )

    assert summary["metric_level"] == "full"
    assert summary["write_eye_geometry"] is True
    run = root["refined_subject_masks_runs"]["refined_subject_masks_smart_full_001"]
    assert run.attrs["eye_geometry_status"] == "computed"
    assert "relations" in run
    assert run["components/subject_body"].attrs["shape_qc_metrics_status"] == "computed"
    assert run["components/subject_body/metrics"].attrs["metric_level"] == "full"


def test_refresh_refined_subject_mask_metrics_updates_metric_qc_reasons(monkeypatch) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    root = _build_probability_root()
    mod.finalize_subject_mask_run(
        root,
        subject_run="subject_probs_001",
        refined_run="refined_subject_masks_smart_001",
        chunk_size=1,
    )
    run = root["refined_subject_masks_runs"]["refined_subject_masks_smart_001"]
    labels = list(run.attrs["mask_labels"])
    body_idx = labels.index("subject_body")

    edited_body = np.zeros((10, 10), dtype=np.uint8)
    edited_body[1, 1] = 1
    edited_body[8, 8] = 1
    run["masks_roi"][1, body_idx] = edited_body
    write_reason_columns(
        run["components/subject_body"],
        np.asarray(["clean", "manual_correction|needs_review_metric_holes"], dtype=object),
        chunk_size=2,
        include_reason_text=True,
        overwrite=True,
    )

    summary = mod.refresh_refined_subject_mask_metrics_run(
        root,
        refined_run="refined_subject_masks_smart_001",
        components=["subject_body"],
        chunk_size=1,
        metric_level="cheap",
    )

    assert summary["components"] == ["subject_body"]
    assert summary["review_counts"]["subject_body"]["needs_review"] == 1
    assert float(np.asarray(run["metrics/area_px"][1, body_idx], dtype=np.float32)) == pytest.approx(2.0)
    component_metrics = run["components/subject_body/metrics"]
    assert int(np.asarray(component_metrics["component_count"][1], dtype=np.int32)) == 2
    assert component_metrics.attrs["schema_id"] == "refined_subject_component_mask_metrics_v1"
    run = root["refined_subject_masks_runs"]["refined_subject_masks_smart_001"]
    assert run.attrs["component_metric_qc_review_counts"]["subject_body"]["needs_review"] == 1

    body_reasons = read_reason_labels(run["components/subject_body"])
    assert body_reasons is not None
    assert "manual_correction" in str(body_reasons[1])
    assert "needs_review_metric_holes" not in str(body_reasons[1])
    assert "needs_review_metric_small_area" in str(body_reasons[1])
    assert "needs_review_metric_multiple_components" in str(body_reasons[1])


def test_refresh_refined_subject_mask_metrics_dask_worker_chunks_updates_metric_qc_reasons(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    zarr_path = tmp_path / "analysis.zarr"
    _build_probability_root(zarr_path)
    mod.finalize_subject_masks(
        zarr_path,
        subject_run="subject_probs_001",
        refined_run="refined_subject_masks_smart_refresh_dask_001",
        chunk_size=1,
    )
    root = zarr.open_group(str(zarr_path), mode="a")
    run = root["refined_subject_masks_runs/refined_subject_masks_smart_refresh_dask_001"]
    labels = list(run.attrs["mask_labels"])
    body_idx = labels.index("subject_body")

    edited_body = np.zeros((10, 10), dtype=np.uint8)
    edited_body[1, 1] = 1
    edited_body[8, 8] = 1
    run["masks_roi"][1, body_idx] = edited_body
    write_reason_columns(
        run["components/subject_body"],
        np.asarray(["clean", "manual_correction|needs_review_metric_holes"], dtype=object),
        chunk_size=2,
        include_reason_text=True,
        overwrite=True,
    )

    summary = mod.refresh_refined_subject_mask_metrics(
        zarr_path,
        refined_run="refined_subject_masks_smart_refresh_dask_001",
        components=["subject_body"],
        chunk_size=1,
        metric_level="cheap",
        execution_backend="dask_worker_chunks",
        scheduler="threads",
        num_workers=2,
    )

    assert summary["components"] == ["subject_body"]
    assert summary["execution_backend"] == "dask_worker_chunks"
    assert summary["dask_execution_enabled"] is True
    assert summary["dask_scheduler"] == "threads"
    assert summary["dask_num_workers"] == 2
    assert summary["review_counts"]["subject_body"]["needs_review"] == 1

    root = zarr.open_group(str(zarr_path), mode="r")
    run = root["refined_subject_masks_runs/refined_subject_masks_smart_refresh_dask_001"]
    assert run.attrs["component_metric_qc_execution_backend"] == "dask_worker_chunks"
    assert run.attrs["component_metric_qc_timing_summary"]["dask_execution_enabled"] is True
    assert "dask_compute" in run.attrs["component_metric_qc_timing_summary"]["phase_seconds"]
    assert len(run.attrs["component_metric_qc_chunk_timings"]) == 2
    assert float(np.asarray(run["metrics/area_px"][1, body_idx], dtype=np.float32)) == pytest.approx(2.0)
    component_metrics = run["components/subject_body/metrics"]
    assert int(np.asarray(component_metrics["component_count"][1], dtype=np.int32)) == 2

    body_reasons = read_reason_labels(run["components/subject_body"])
    assert body_reasons is not None
    assert "manual_correction" in str(body_reasons[1])
    assert "needs_review_metric_holes" not in str(body_reasons[1])
    assert "needs_review_metric_small_area" in str(body_reasons[1])
    assert "needs_review_metric_multiple_components" in str(body_reasons[1])


def test_finalize_subject_masks_dask_worker_chunks_writes_disjoint_rows(monkeypatch, tmp_path: Path) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    zarr_path = tmp_path / "analysis.zarr"
    _build_probability_root(zarr_path)

    summary = mod.finalize_subject_masks(
        zarr_path,
        subject_run="subject_probs_001",
        refined_run="refined_subject_masks_smart_dask_001",
        chunk_size=1,
        execution_backend="dask_worker_chunks",
        scheduler="threads",
        num_workers=2,
    )

    assert summary["execution_backend"] == "dask_worker_chunks"
    assert summary["dask_execution_enabled"] is True
    assert summary["dask_scheduler"] == "threads"
    assert summary["timing_summary"]["dask_execution_enabled"] is True
    root = zarr.open_group(str(zarr_path), mode="r")
    run = root["refined_subject_masks_runs/refined_subject_masks_smart_dask_001"]
    assert run.attrs["smart_finalizer_execution_backend"] == "dask_worker_chunks"
    assert run.attrs["dask_execution_enabled"] is True
    assert run.attrs["dask_scheduler"] == "threads"
    assert len(run.attrs["smart_finalizer_chunk_timings"]) == 2
    labels = list(run.attrs["mask_labels"])
    masks = np.asarray(run["masks_roi"][:], dtype=np.uint8)
    assert np.count_nonzero(masks[:, labels.index("subject_body")]) > 0
    assert np.count_nonzero(masks[:, labels.index("eye_left")]) > 0
    assert np.count_nonzero(masks[:, labels.index("eye_right")]) > 0
    assert np.count_nonzero(masks[:, labels.index("swim_bladder")]) > 0
    assert "dask_compute" in run.attrs["smart_finalizer_timing_summary"]["phase_seconds"]


def test_finalize_subject_mask_run_dry_run_and_overwrite_guard(monkeypatch) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    root = _build_probability_root()

    dry = mod.finalize_subject_mask_run(
        root,
        subject_run="subject_probs_001",
        refined_run="refined_subject_masks_smart_001",
        chunk_size=1,
        scheduler="single-thread",
        dry_run=True,
    )

    assert dry["status"] == "planned"
    assert dry["mutates_archive"] is False
    assert dry["metric_level"] == "cheap"
    assert dry["write_eye_geometry"] is False
    assert dry["dask_scheduler"] == "single-threaded"
    assert dry["dask_execution_enabled"] is False
    assert "refined_subject_masks_runs" not in root

    mod.finalize_subject_mask_run(
        root,
        subject_run="subject_probs_001",
        refined_run="refined_subject_masks_smart_001",
        chunk_size=1,
    )
    with pytest.raises(ValueError, match="already exists"):
        mod.finalize_subject_mask_run(
            root,
            subject_run="subject_probs_001",
            refined_run="refined_subject_masks_smart_001",
            chunk_size=1,
        )
