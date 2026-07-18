from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from fisheye.refinement import assemble_refined_subject_masks as assemble_mod
from fisheye.refinement import refine_subject_masks as batch_mod
from fisheye.shared.detect_reason_codec import read_reason_labels
from fisheye.shared.mask_store import write_component_rle_mask_store_from_dense
from fisheye.shared.subject_mask_chunks import (
    refined_subject_mask_metric_row_chunk,
    refined_subject_mask_storage_chunks,
)
from fisheye.tune import refined_subject_mask_review as review_mod

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")


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
                "hostname": "assemble-host",
                "system": "Linux",
                "release": "6.8",
                "python_version": "3.12.0",
                "machine": "x86_64",
            },
        },
    )


def _create_subject_run(
    root: zarr.Group,
    *,
    run_name: str,
    method: str,
    mask_labels: list[str],
    available_channels: np.ndarray,
    masks: np.ndarray,
    source_crop_run: str = "crop_001",
    source_crop_storage_mode: str = "geometry_only",
    source_crop_signature: str = "{'signature_version': 2, 'crop_revision': 4}",
    source_crop_revision: int = 4,
    source_detect_review_status_ref: str = "refined_detect_runs/refined_detect_001/review_status",
    source_keypoints_run: str = "refined_kp_001",
    source_keypoint_group: str = "refined_keypoints_runs",
    source_crop_row_ids: np.ndarray | None = None,
    write_source_crop_row_ids: bool = True,
) -> zarr.Group:
    parent = root.require_group("subject_mask_runs")
    parent.attrs["latest"] = run_name
    run = parent.create_group(run_name)
    run.attrs.update(
        {
            "source_crop_run": source_crop_run,
            "source_crop_storage_mode": source_crop_storage_mode,
            "source_crop_signature": source_crop_signature,
            "source_crop_revision": source_crop_revision,
            "source_detect_review_status_ref": source_detect_review_status_ref,
            "method": method,
            "mask_labels": list(mask_labels),
            "label_schema_id": "subject_v1_lr" if mask_labels == ["subject_body", "eye_left", "eye_right", "swim_bladder"] else "subject_v1_custom",
            "source_keypoints_run": source_keypoints_run,
            "source_keypoint_run": source_keypoints_run,
            "source_keypoint_group": source_keypoint_group,
            "created_at_utc": "2026-04-01T00:00:00+00:00",
        }
    )
    run.create_array("detection_source", data=np.asarray([0, 0], dtype=np.int8), overwrite=True)
    run.create_array("frame_indices", data=np.asarray([10, 11], dtype=np.int32), overwrite=True)
    run.create_array("detection_indices", data=np.asarray([0, 1], dtype=np.int32), overwrite=True)
    run.create_array("frame_counts", data=np.asarray([1, 1], dtype=np.int32), overwrite=True)
    if write_source_crop_row_ids:
        row_ids = np.asarray(
            [0, 1] if source_crop_row_ids is None else source_crop_row_ids,
            dtype=np.int64,
        )
        run.create_array("source_crop_row_ids", data=row_ids, overwrite=True)
    run.create_array("available_channels", data=np.asarray(available_channels, dtype=bool), overwrite=True)
    run.create_array("masks_roi", data=np.asarray(masks, dtype=np.uint8), overwrite=True)
    return run


def _create_keypoint_run(
    root: zarr.Group,
    *,
    run_name: str = "refined_kp_001",
    source_crop_row_ids: np.ndarray | None = None,
    write_source_crop_row_ids: bool = True,
) -> zarr.Group:
    parent = root.require_group("refined_keypoints_runs")
    parent.attrs["latest"] = run_name
    run = parent.create_group(run_name)
    run.attrs["keypoint_labels"] = ["swim_bladder", "eye_left", "eye_right"]
    keypoints_roi = np.asarray(
        [
            [[4.5, 5.0], [2.0, 2.0], [5.0, 2.0]],
            [[4.0, 4.0], [2.0, 4.0], [5.0, 4.0]],
        ],
        dtype=np.float32,
    )
    run.create_array("keypoints_roi", data=keypoints_roi, overwrite=True)
    run.create_array("heading", data=np.asarray([0.0, 0.0], dtype=np.float32), overwrite=True)
    run.create_array("detection_success", data=np.asarray([True, True], dtype=bool), overwrite=True)
    if write_source_crop_row_ids:
        row_ids = np.asarray(
            [0, 1] if source_crop_row_ids is None else source_crop_row_ids,
            dtype=np.int64,
        )
        run.create_array("source_crop_row_ids", data=row_ids, overwrite=True)
    return run


def test_resolve_keypoint_success_array_prefers_usable_keypoints() -> None:
    root = zarr.group()
    run = root.create_group("refined_kp_001")
    run.create_array("refined_success", data=np.asarray([True, True, True], dtype=bool))
    run.create_array("usable_keypoints", data=np.asarray([True, False, True], dtype=bool))

    success, dataset_name = assemble_mod._resolve_keypoint_success_array(run, "refined_kp_001")

    assert dataset_name == "usable_keypoints"
    np.testing.assert_array_equal(success, np.asarray([True, False, True], dtype=bool))


def test_resolve_keypoint_success_array_falls_back_for_legacy_runs() -> None:
    root = zarr.group()
    run = root.create_group("legacy_kp_001")
    run.create_array("refined_success", data=np.asarray([True, False], dtype=bool))

    success, dataset_name = assemble_mod._resolve_keypoint_success_array(run, "legacy_kp_001")

    assert dataset_name == "refined_success"
    np.testing.assert_array_equal(success, np.asarray([True, False], dtype=bool))


def _build_assembly_root() -> zarr.Group:
    root = zarr.group()
    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "crop_001"
    crop = crop_parent.create_group("crop_001")
    crop.attrs["crop_storage_mode"] = "geometry_only"
    crop.attrs["crop_signature"] = {"signature_version": 2, "crop_revision": 4}
    crop.attrs["crop_revision"] = 4
    crop.attrs["detect_review_status_ref"] = "refined_detect_runs/refined_detect_001/review_status"
    crop.create_array("roi_images", data=np.zeros((2, 8, 8), dtype=np.uint8), overwrite=True)
    crop.create_array("frame_indices", data=np.asarray([10, 11], dtype=np.int32), overwrite=True)

    body_masks = np.zeros((2, 1, 8, 8), dtype=np.uint8)
    body_masks[0, 0, 1:7, 1:7] = 1
    body_masks[1, 0, 2:6, 2:6] = 1
    _create_subject_run(
        root,
        run_name="body_run_001",
        method="sam_subject_body_v1",
        mask_labels=["subject_body"],
        available_channels=np.asarray([True], dtype=bool),
        masks=body_masks,
    )

    eye_masks = np.zeros((2, 2, 8, 8), dtype=np.uint8)
    eye_masks[0, 0, 1:4, 1:4] = 1
    eye_masks[0, 1, 1:4, 4:7] = 1
    eye_masks[1, 0, 3:6, 1:4] = 1
    eye_masks[1, 1, 3:6, 4:7] = 1
    _create_subject_run(
        root,
        run_name="eye_run_001",
        method="refined_eye_projection_v1",
        mask_labels=["eye_left", "eye_right"],
        available_channels=np.asarray([True, True], dtype=bool),
        masks=eye_masks,
    )

    swim_masks = np.zeros((2, 1, 8, 8), dtype=np.uint8)
    swim_masks[0, 0, 4:6, 4:6] = 1
    swim_masks[1, 0, 3:5, 3:5] = 1
    _create_subject_run(
        root,
        run_name="swim_run_001",
        method="traditional_swim_bladder_inference",
        mask_labels=["swim_bladder"],
        available_channels=np.asarray([True], dtype=bool),
        masks=swim_masks,
    )
    return root


def test_validate_source_alignment_rejects_crop_snapshot_mismatch_without_zarr() -> None:
    reference = SimpleNamespace(
        crop_run="crop_001",
        source_crop_snapshot={
            "source_crop_storage_mode": "geometry_only",
            "source_crop_signature": "sig-001",
            "source_crop_revision": 4,
            "source_detect_review_status_ref": "refined_detect_runs/refined_detect_001/review_status",
        },
        masks_roi=np.zeros((2, 1, 8, 8), dtype=np.uint8),
        detection_source=np.zeros((2,), dtype=np.int8),
        frame_indices=np.asarray([10, 11], dtype=np.int32),
        frame_counts=np.asarray([1, 1], dtype=np.int32),
        detection_indices=np.asarray([0, 1], dtype=np.int32),
        source_refined_row_ids=np.asarray([100, 101], dtype=np.int64),
        source_detect_row_index=np.asarray([0, 1], dtype=np.int32),
    )
    other = SimpleNamespace(
        crop_run="crop_001",
        source_crop_snapshot={
            "source_crop_storage_mode": "geometry_only",
            "source_crop_signature": "sig-002",
            "source_crop_revision": 4,
            "source_detect_review_status_ref": "refined_detect_runs/refined_detect_001/review_status",
        },
        masks_roi=np.zeros((2, 1, 8, 8), dtype=np.uint8),
        detection_source=np.zeros((2,), dtype=np.int8),
        frame_indices=np.asarray([10, 11], dtype=np.int32),
        frame_counts=np.asarray([1, 1], dtype=np.int32),
        detection_indices=np.asarray([0, 1], dtype=np.int32),
        source_refined_row_ids=np.asarray([100, 101], dtype=np.int64),
        source_detect_row_index=np.asarray([0, 1], dtype=np.int32),
    )

    with pytest.raises(ValueError, match="Alignment mismatch for crop snapshot fields"):
        assemble_mod._validate_source_alignment(reference, other)  # noqa: SLF001


def test_validate_source_alignment_allows_metadata_only_refined_source_view_mismatch_without_zarr() -> None:
    signature_base = {
        "signature_version": 1,
        "source_detect_run": "detect_001",
        "source_refined_run": "refined_detect_001",
        "roi_size": [512, 512],
        "parameter_source": "config_default",
        "parameters_hash": "abc123",
    }
    reference = SimpleNamespace(
        crop_run="crop_001",
        source_crop_snapshot={
            "source_crop_storage_mode": "materialized",
            "source_crop_signature": str(
                {
                    **signature_base,
                    "detection_source_path": "refined_detect_runs/refined_detect_001/instances",
                    "detection_source_type": "refined",
                }
            ),
            "source_crop_revision": 4,
            "source_detect_review_status_ref": "refined_detect_runs/refined_detect_001/review_status",
        },
        masks_roi=np.zeros((2, 2, 8, 8), dtype=np.uint8),
        detection_source=np.zeros((2,), dtype=np.int8),
        frame_indices=np.asarray([10, 11], dtype=np.int32),
        frame_counts=np.asarray([1, 1], dtype=np.int32),
        detection_indices=np.asarray([0, 1], dtype=np.int32),
        source_refined_row_ids=np.asarray([100, 101], dtype=np.int64),
        source_detect_row_index=np.asarray([0, 1], dtype=np.int32),
    )
    other = SimpleNamespace(
        crop_run="crop_001",
        source_crop_snapshot={
            "source_crop_storage_mode": "materialized",
            "source_crop_signature": str(
                {
                    **signature_base,
                    "detection_source_path": "refined_detect_runs/refined_detect_001/manual",
                    "detection_source_type": "manual",
                }
            ),
            "source_crop_revision": 4,
            "source_detect_review_status_ref": "refined_detect_runs/refined_detect_001/review_status",
        },
        masks_roi=np.zeros((2, 1, 8, 8), dtype=np.uint8),
        detection_source=np.zeros((2,), dtype=np.int8),
        frame_indices=np.asarray([10, 11], dtype=np.int32),
        frame_counts=np.asarray([1, 1], dtype=np.int32),
        detection_indices=np.asarray([0, 1], dtype=np.int32),
        source_refined_row_ids=np.asarray([100, 101], dtype=np.int64),
        source_detect_row_index=np.asarray([0, 1], dtype=np.int32),
    )

    assemble_mod._validate_source_alignment(reference, other)  # noqa: SLF001


def test_validate_source_alignment_rejects_one_sided_instance_key_loss_without_zarr() -> None:
    shared = {
        "crop_run": "crop_001",
        "source_crop_snapshot": {
            "source_crop_storage_mode": "geometry_only",
            "source_crop_signature": "sig-001",
            "source_crop_revision": 4,
        },
        "masks_roi": np.zeros((2, 1, 8, 8), dtype=np.uint8),
        "detection_source": np.zeros((2,), dtype=np.int8),
        "frame_indices": np.asarray([10, 11], dtype=np.int32),
        "frame_counts": np.asarray([1, 1], dtype=np.int32),
        "detection_indices": np.asarray([0, 1], dtype=np.int32),
        "source_refined_row_ids": np.asarray([100, 101], dtype=np.int64),
        "source_detect_row_index": np.asarray([0, 1], dtype=np.int32),
    }
    reference = SimpleNamespace(
        **shared,
        instance_key=np.asarray([1001, 1002], dtype=np.uint64),
    )
    other = SimpleNamespace(**shared)

    with pytest.raises(ValueError, match="one-sided key loss"):
        assemble_mod._validate_source_alignment(reference, other)  # noqa: SLF001


def test_validate_source_alignment_rejects_source_view_mismatch_with_real_signature_drift_without_zarr() -> None:
    reference = SimpleNamespace(
        crop_run="crop_001",
        source_crop_snapshot={
            "source_crop_storage_mode": "materialized",
            "source_crop_signature": {
                "signature_version": 1,
                "detection_source_path": "refined_detect_runs/refined_detect_001/instances",
                "detection_source_type": "refined",
                "source_detect_run": "detect_001",
                "source_refined_run": "refined_detect_001",
            },
            "source_crop_revision": 4,
        },
        masks_roi=np.zeros((2, 2, 8, 8), dtype=np.uint8),
        detection_source=np.zeros((2,), dtype=np.int8),
        frame_indices=np.asarray([10, 11], dtype=np.int32),
        frame_counts=np.asarray([1, 1], dtype=np.int32),
        detection_indices=np.asarray([0, 1], dtype=np.int32),
        source_refined_row_ids=np.asarray([100, 101], dtype=np.int64),
        source_detect_row_index=np.asarray([0, 1], dtype=np.int32),
    )
    other = SimpleNamespace(
        crop_run="crop_001",
        source_crop_snapshot={
            "source_crop_storage_mode": "materialized",
            "source_crop_signature": {
                "signature_version": 1,
                "detection_source_path": "refined_detect_runs/refined_detect_001/manual",
                "detection_source_type": "manual",
                "source_detect_run": "detect_002",
                "source_refined_run": "refined_detect_001",
            },
            "source_crop_revision": 4,
        },
        masks_roi=np.zeros((2, 1, 8, 8), dtype=np.uint8),
        detection_source=np.zeros((2,), dtype=np.int8),
        frame_indices=np.asarray([10, 11], dtype=np.int32),
        frame_counts=np.asarray([1, 1], dtype=np.int32),
        detection_indices=np.asarray([0, 1], dtype=np.int32),
        source_refined_row_ids=np.asarray([100, 101], dtype=np.int64),
        source_detect_row_index=np.asarray([0, 1], dtype=np.int32),
    )

    with pytest.raises(ValueError, match="Alignment mismatch for crop snapshot fields"):
        assemble_mod._validate_source_alignment(reference, other)  # noqa: SLF001


def test_assemble_refined_subject_run_creates_finalized_mixed_source_run(monkeypatch) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    root = _build_assembly_root()

    summary = assemble_mod.assemble_refined_subject_run(
        root,
        body_run="body_run_001",
        eye_run="eye_run_001",
        swim_run="swim_run_001",
        refined_run="refined_subject_masks_mixed_001",
    )

    assert summary["status"] == "updated"
    assert summary["row_identity_mode"] == "legacy_positional"
    assert summary["row_identity_mode_schema"] == "palette.row_identity_mode.v1"
    assert summary["component_names"] == ["subject_body", "eye_left", "eye_right", "swim_bladder"]
    assert summary["source_subject_mask_runs"] == {
        "subject_body": "body_run_001",
        "eye_left": "eye_run_001",
        "eye_right": "eye_run_001",
        "swim_bladder": "swim_run_001",
    }

    run = root["refined_subject_masks_runs"]["refined_subject_masks_mixed_001"]
    assert run.attrs["label_schema_id"] == "subject_v1_lr"
    assert run.attrs["method"] == assemble_mod.ASSEMBLE_REFINED_SUBJECT_METHOD
    assert run.attrs["assembly_semantics"] == "multi_source_component_seed"
    assert run.attrs["row_identity_mode"] == "legacy_positional"
    assert run.attrs["row_identity_mode_schema"] == "palette.row_identity_mode.v1"
    assert run.attrs["source_subject_mask_run"] == "body_run_001"
    assert run.attrs["source_body_subject_mask_run"] == "body_run_001"
    assert run.attrs["source_eye_subject_mask_run"] == "eye_run_001"
    assert run.attrs["source_swim_subject_mask_run"] == "swim_run_001"
    assert run.attrs["source_crop_storage_mode"] == "geometry_only"
    assert run.attrs["source_crop_signature"] == "{'signature_version': 2, 'crop_revision': 4}"
    assert run.attrs["source_crop_revision"] == 4
    assert run.attrs["source_detect_review_status_ref"] == "refined_detect_runs/refined_detect_001/review_status"
    np.testing.assert_array_equal(run["source_crop_row_ids"][:], np.asarray([0, 1], dtype=np.int64))
    np.testing.assert_array_equal(
        root[f"crop_runs/{run.attrs['source_crop_run']}"]["frame_indices"][run["source_crop_row_ids"][:]],
        run["frame_indices"][:],
    )
    np.testing.assert_array_equal(
        np.asarray(run["available_channels"][:], dtype=bool),
        np.asarray([True, True, True, True], dtype=bool),
    )
    np.testing.assert_array_equal(
        np.asarray(run["edit_applied"][:], dtype=bool),
        np.zeros((2, 4), dtype=bool),
    )
    assert tuple(int(v) for v in run["masks_roi"].chunks) == refined_subject_mask_storage_chunks(2, 8, 8)
    assert tuple(int(v) for v in run["edit_applied"].chunks) == (refined_subject_mask_metric_row_chunk(2), 1)
    np.testing.assert_array_equal(
        np.asarray(run["metrics/mask_present"][:], dtype=bool),
        np.asarray(
            [
                [True, True, True, True],
                [True, True, True, True],
            ],
            dtype=bool,
        ),
    )
    assert tuple(int(v) for v in run["metrics/mask_present"].chunks) == (refined_subject_mask_metric_row_chunk(2), 1)

    body_reasons = read_reason_labels(run["components/subject_body"])
    eye_left_reasons = read_reason_labels(run["components/eye_left"])
    swim_reasons = read_reason_labels(run["components/swim_bladder"])
    assert body_reasons is not None
    assert eye_left_reasons is not None
    assert swim_reasons is not None
    assert body_reasons.tolist() == ["copied_from_source", "copied_from_source"]
    assert eye_left_reasons.tolist() == ["copied_from_source", "copied_from_source"]
    assert swim_reasons.tolist() == ["copied_from_source", "copied_from_source"]

    assert run["components/subject_body/provenance"].attrs["source_run"] == "body_run_001"
    assert run["components/subject_body/provenance"].attrs["source_crop_run"] == "crop_001"
    assert run["components/subject_body/provenance"].attrs["source_crop_storage_mode"] == "geometry_only"
    assert run["components/subject_body/provenance"].attrs["source_crop_signature"] == "{'signature_version': 2, 'crop_revision': 4}"
    assert run["components/subject_body/provenance"].attrs["source_crop_revision"] == 4
    assert (
        run["components/subject_body/provenance"].attrs["source_detect_review_status_ref"]
        == "refined_detect_runs/refined_detect_001/review_status"
    )
    assert run["components/eye_left/provenance"].attrs["source_run"] == "eye_run_001"
    assert run["components/eye_right/provenance"].attrs["source_run"] == "eye_run_001"
    assert run["components/swim_bladder/provenance"].attrs["source_run"] == "swim_run_001"
    assert run["components/swim_bladder/provenance"].attrs["last_update_mode"] == "create"
    assert run.attrs["eye_geometry_schema_id"] == "refined_subject_eye_geometry_v1"
    left_geometry = run["components/eye_left/geometry"]
    right_geometry = run["components/eye_right/geometry"]
    assert left_geometry.attrs["geometry_schema_id"] == "refined_subject_eye_geometry_v1"
    assert right_geometry.attrs["geometry_schema_id"] == "refined_subject_eye_geometry_v1"
    assert left_geometry["ellipse_params"].shape == (2, 5)
    assert right_geometry["ellipse_params"].shape == (2, 5)
    np.testing.assert_array_equal(np.asarray(left_geometry["ellipse_success"][:], dtype=bool), np.ones((2,), dtype=bool))
    np.testing.assert_array_equal(np.asarray(right_geometry["ellipse_success"][:], dtype=bool), np.ones((2,), dtype=bool))
    left_contours = run["components/eye_left/contours"]
    right_contours = run["components/eye_right/contours"]
    assert left_contours["ptr"].shape == (2,)
    assert right_contours["ptr"].shape == (2,)
    assert left_contours["points_xy"].shape[1] == 2
    assert right_contours["points_xy"].shape[1] == 2
    eye_pair_metrics = run["relations/eye_pair/metrics"]
    assert eye_pair_metrics.attrs["relation_schema_id"] == "refined_subject_eye_pair_relation_v1"
    np.testing.assert_array_equal(
        np.asarray(eye_pair_metrics["separation_valid"][:], dtype=bool),
        np.ones((2,), dtype=bool),
    )
    assert np.all(np.asarray(eye_pair_metrics["separation_px"][:], dtype=np.float32) > 0.0)

    provenance = run.attrs["provenance"]
    assert provenance["inputs"]["source_crop_run"] == "crop_001"
    assert provenance["inputs"]["source_crop_storage_mode"] == "geometry_only"
    assert provenance["inputs"]["source_crop_signature"] == "{'signature_version': 2, 'crop_revision': 4}"
    assert provenance["inputs"]["source_crop_revision"] == 4
    assert provenance["inputs"]["source_detect_review_status_ref"] == "refined_detect_runs/refined_detect_001/review_status"


def test_assemble_refined_subject_run_copies_all_canonical_components_from_subject_run(monkeypatch) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    root = _build_assembly_root()
    masks = np.zeros((2, 4, 8, 8), dtype=np.uint8)
    masks[0, 0, 1:7, 1:7] = 1
    masks[1, 0, 2:6, 2:6] = 1
    masks[0, 1, 1:4, 1:4] = 1
    masks[1, 1, 3:6, 1:4] = 1
    masks[0, 2, 1:4, 4:7] = 1
    masks[1, 2, 3:6, 4:7] = 1
    masks[0, 3, 4:6, 4:6] = 1
    masks[1, 3, 3:5, 3:5] = 1
    _create_subject_run(
        root,
        run_name="subject_all_components_001",
        method="unet_subject_mask_segmenter",
        mask_labels=["subject_body", "eye_left", "eye_right", "swim_bladder"],
        available_channels=np.asarray([True, True, True, True], dtype=bool),
        masks=masks,
    )

    summary = assemble_mod.assemble_refined_subject_run(
        root,
        subject_run="subject_all_components_001",
        refined_run="refined_subject_masks_from_single_subject_001",
    )

    assert summary["status"] == "updated"
    assert summary["component_names"] == ["subject_body", "eye_left", "eye_right", "swim_bladder"]
    assert summary["source_subject_mask_run"] == "subject_all_components_001"
    assert summary["source_input_subject_mask_run"] == "subject_all_components_001"
    assert summary["source_subject_mask_runs"] == {
        "subject_body": "subject_all_components_001",
        "eye_left": "subject_all_components_001",
        "eye_right": "subject_all_components_001",
        "swim_bladder": "subject_all_components_001",
    }

    run = root["refined_subject_masks_runs"]["refined_subject_masks_from_single_subject_001"]
    assert run.attrs["assembly_semantics"] == "single_source_subject_run_seed"
    assert run.attrs["source_subject_mask_run"] == "subject_all_components_001"
    assert run.attrs["source_input_subject_mask_run"] == "subject_all_components_001"
    assert run.attrs["label_schema_id"] == "subject_v1_lr"
    np.testing.assert_array_equal(np.asarray(run["masks_roi"][:], dtype=np.uint8), masks)
    for component_name in ("subject_body", "eye_left", "eye_right", "swim_bladder"):
        provenance = run[f"components/{component_name}/provenance"].attrs
        assert provenance["source_stage"] == "subject_mask_runs"
        assert provenance["source_run"] == "subject_all_components_001"
        assert provenance["source_channels"] == [component_name]


def test_assemble_refined_subject_run_rejects_subject_run_eye_union_without_keypoints(monkeypatch) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    root = _build_assembly_root()
    masks = np.zeros((2, 3, 8, 8), dtype=np.uint8)
    masks[:, 0, 1:7, 1:7] = 1
    masks[:, 1, 1:4, 1:7] = 1
    masks[:, 2, 4:6, 4:6] = 1
    _create_subject_run(
        root,
        run_name="subject_body_eye_union_swim_001",
        method="unet_subject_mask_segmenter",
        mask_labels=["subject_body", "eyes_union", "swim_bladder"],
        available_channels=np.asarray([True, True, True], dtype=bool),
        masks=masks,
    )

    with pytest.raises(ValueError, match="references missing keypoint source"):
        assemble_mod.assemble_refined_subject_run(
            root,
            subject_run="subject_body_eye_union_swim_001",
            refined_run="refined_subject_masks_eye_union_rejected_001",
        )


def test_assemble_refined_subject_run_accepts_probability_only_subject_run(monkeypatch) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    root = _build_assembly_root()
    masks = np.zeros((2, 1, 8, 8), dtype=np.uint8)
    masks[0, 0, 1:7, 1:7] = 1
    masks[1, 0, 2:6, 2:6] = 1
    run = _create_subject_run(
        root,
        run_name="subject_body_probability_only_001",
        method="unet_subject_mask_segmenter",
        mask_labels=["subject_body"],
        available_channels=np.asarray([True], dtype=bool),
        masks=masks,
    )
    del run["masks_roi"]
    probabilities = masks * np.uint8(255)
    probabilities[0, 0, 0, 0] = np.uint8(127)
    run.create_array("mask_probs_roi", data=probabilities, overwrite=True)
    run.attrs["probabilities_encoding"] = "linear_uint8_0_255"
    run.attrs["mask_probability_threshold"] = 0.5

    summary = assemble_mod.assemble_refined_subject_run(
        root,
        subject_run="subject_body_probability_only_001",
        refined_run="refined_subject_masks_from_probability_only_001",
    )

    assert summary["component_names"] == ["subject_body"]
    refined = root["refined_subject_masks_runs"]["refined_subject_masks_from_probability_only_001"]
    np.testing.assert_array_equal(np.asarray(refined["masks_roi"][:], dtype=np.uint8), masks)
    provenance = refined["components/subject_body/provenance"].attrs
    assert provenance["source_probability_path"] == (
        "subject_mask_runs/subject_body_probability_only_001/mask_probs_roi"
    )
    assert provenance["source_probability_encoding"] == "linear_uint8_0_255"
    assert provenance["source_binary_derivation"] == "threshold(mask_probs_roi)"
    assert float(provenance["source_probability_threshold"]) == pytest.approx(0.5)


def test_assemble_refined_subject_run_assigns_subject_run_eye_union_with_keypoints(monkeypatch) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    root = _build_assembly_root()
    _create_keypoint_run(root)
    masks = np.zeros((2, 3, 8, 8), dtype=np.uint8)
    masks[0, 0, 1:7, 1:7] = 1
    masks[1, 0, 2:6, 2:6] = 1
    masks[0, 1, 1:4, 1:4] = 1
    masks[0, 1, 1:4, 4:7] = 1
    masks[1, 1, 3:6, 1:4] = 1
    masks[1, 1, 3:6, 4:7] = 1
    masks[0, 2, 4:6, 4:6] = 1
    masks[1, 2, 3:5, 3:5] = 1
    _create_subject_run(
        root,
        run_name="subject_body_eye_union_swim_001",
        method="unet_subject_mask_segmenter",
        mask_labels=["subject_body", "eyes_union", "swim_bladder"],
        available_channels=np.asarray([True, True, True], dtype=bool),
        masks=masks,
    )

    summary = assemble_mod.assemble_refined_subject_run(
        root,
        subject_run="subject_body_eye_union_swim_001",
        refined_run="refined_subject_masks_assigned_eye_union_001",
    )

    assert summary["component_names"] == ["subject_body", "eye_left", "eye_right", "swim_bladder"]
    assignment_summary = summary["eyes_union_assignment_summary"]
    assert assignment_summary["assignment_method"] == "subject_eyes_union_keypoint_assignment_v1"
    assert assignment_summary["assigned_rows"] == 2
    assert assignment_summary["failed_rows"] == 0
    assert assignment_summary["keypoint_source_kind"] == "source_keypoint_lineage"

    run = root["refined_subject_masks_runs"]["refined_subject_masks_assigned_eye_union_001"]
    assert run.attrs["assembly_semantics"] == "single_source_subject_run_seed"
    assert run.attrs["eyes_union_assignment_summary"]["assigned_rows"] == 2
    assert run.attrs["assignment_keypoints_run"] == "refined_kp_001"
    assert run.attrs["assignment_keypoint_group"] == "refined_keypoints_runs"
    assert run.attrs["assignment_keypoint_contract"] == "subject_eyes_union_assignment_keypoints_v1"
    assert run.attrs["assignment_keypoint_selection"] == "source_keypoint_lineage"
    expected_left = np.zeros((2, 8, 8), dtype=np.uint8)
    expected_left[0, 1:4, 1:4] = 1
    expected_left[1, 3:6, 1:4] = 1
    expected_right = np.zeros((2, 8, 8), dtype=np.uint8)
    expected_right[0, 1:4, 4:7] = 1
    expected_right[1, 3:6, 4:7] = 1
    np.testing.assert_array_equal(np.asarray(run["masks_roi"][:, 1], dtype=np.uint8), expected_left)
    np.testing.assert_array_equal(np.asarray(run["masks_roi"][:, 2], dtype=np.uint8), expected_right)
    np.testing.assert_array_equal(
        np.asarray(run["components/eye_left/source_seed_masks_roi"][:], dtype=np.uint8),
        expected_left,
    )
    np.testing.assert_array_equal(
        np.asarray(run["components/eye_right/source_seed_masks_roi"][:], dtype=np.uint8),
        expected_right,
    )
    np.testing.assert_array_equal(
        np.asarray(run["components/eye_left/manual_override"][:], dtype=bool),
        np.zeros((2,), dtype=bool),
    )

    eye_left_provenance = run["components/eye_left/provenance"].attrs
    eye_right_provenance = run["components/eye_right/provenance"].attrs
    assert eye_left_provenance["source_channels"] == ["eyes_union"]
    assert eye_right_provenance["source_channels"] == ["eyes_union"]
    assert eye_left_provenance["assignment_method"] == "subject_eyes_union_keypoint_assignment_v1"
    assert eye_left_provenance["assignment_keypoint_source_kind"] == "source_keypoint_lineage"
    assert eye_right_provenance["assignment_keypoint_group"] == "refined_keypoints_runs"

    eye_left_reasons = read_reason_labels(run["components/eye_left"])
    eye_right_reasons = read_reason_labels(run["components/eye_right"])
    assert eye_left_reasons is not None
    assert eye_right_reasons is not None
    assert eye_left_reasons.tolist() == [
        "assigned_from_eyes_union|split_by_keypoint",
        "assigned_from_eyes_union|split_by_keypoint",
    ]
    assert eye_right_reasons.tolist() == eye_left_reasons.tolist()

    left_geometry = run["components/eye_left/geometry"]
    right_geometry = run["components/eye_right/geometry"]
    np.testing.assert_array_equal(np.asarray(left_geometry["ellipse_success"][:], dtype=bool), np.ones((2,), dtype=bool))
    np.testing.assert_array_equal(np.asarray(right_geometry["ellipse_success"][:], dtype=bool), np.ones((2,), dtype=bool))
    np.testing.assert_array_equal(
        np.asarray(run["relations/eye_pair/metrics/separation_valid"][:], dtype=bool),
        np.ones((2,), dtype=bool),
    )


def test_assemble_refined_subject_run_rejects_eye_union_keypoint_row_identity_mismatch(monkeypatch) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    root = _build_assembly_root()
    _create_keypoint_run(root, source_crop_row_ids=np.asarray([1, 0], dtype=np.int64))
    masks = np.zeros((2, 3, 8, 8), dtype=np.uint8)
    masks[:, 0, 1:7, 1:7] = 1
    masks[0, 1, 1:4, 1:4] = 1
    masks[0, 1, 1:4, 4:7] = 1
    masks[1, 1, 3:6, 1:4] = 1
    masks[1, 1, 3:6, 4:7] = 1
    masks[:, 2, 4:6, 4:6] = 1
    _create_subject_run(
        root,
        run_name="subject_body_eye_union_swim_001",
        method="unet_subject_mask_segmenter",
        mask_labels=["subject_body", "eyes_union", "swim_bladder"],
        available_channels=np.asarray([True, True, True], dtype=bool),
        masks=masks,
    )

    with pytest.raises(ValueError, match="row identity mismatch.*refusing to split eye masks"):
        assemble_mod.assemble_refined_subject_run(
            root,
            subject_run="subject_body_eye_union_swim_001",
            refined_run="refined_subject_masks_bad_eye_identity_001",
        )

    refined_parent = root.get("refined_subject_masks_runs")
    assert refined_parent is None or "refined_subject_masks_bad_eye_identity_001" not in refined_parent


def test_assemble_refined_subject_run_eye_union_allows_legacy_missing_row_identity(monkeypatch) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    root = _build_assembly_root()
    _create_keypoint_run(root, write_source_crop_row_ids=False)
    masks = np.zeros((2, 3, 8, 8), dtype=np.uint8)
    masks[:, 0, 1:7, 1:7] = 1
    masks[0, 1, 1:4, 1:4] = 1
    masks[0, 1, 1:4, 4:7] = 1
    masks[1, 1, 3:6, 1:4] = 1
    masks[1, 1, 3:6, 4:7] = 1
    masks[:, 2, 4:6, 4:6] = 1
    _create_subject_run(
        root,
        run_name="subject_body_eye_union_swim_001",
        method="unet_subject_mask_segmenter",
        mask_labels=["subject_body", "eyes_union", "swim_bladder"],
        available_channels=np.asarray([True, True, True], dtype=bool),
        masks=masks,
    )

    summary = assemble_mod.assemble_refined_subject_run(
        root,
        subject_run="subject_body_eye_union_swim_001",
        refined_run="refined_subject_masks_legacy_eye_identity_001",
    )

    assert summary["eyes_union_assignment_summary"]["assigned_rows"] == 2


def test_assemble_refined_subject_run_prefers_assignment_keypoint_attrs(monkeypatch) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    root = _build_assembly_root()
    _create_keypoint_run(root, run_name="assignment_kp_001")
    masks = np.zeros((2, 3, 8, 8), dtype=np.uint8)
    masks[:, 0, 1:7, 1:7] = 1
    masks[0, 1, 1:4, 1:4] = 1
    masks[0, 1, 1:4, 4:7] = 1
    masks[1, 1, 3:6, 1:4] = 1
    masks[1, 1, 3:6, 4:7] = 1
    masks[:, 2, 4:6, 4:6] = 1
    run = _create_subject_run(
        root,
        run_name="subject_body_eye_union_swim_001",
        method="unet_subject_mask_segmenter",
        mask_labels=["subject_body", "eyes_union", "swim_bladder"],
        available_channels=np.asarray([True, True, True], dtype=bool),
        masks=masks,
        source_keypoints_run="missing_source_kp",
    )
    run.attrs["assignment_keypoint_group"] = "refined_keypoints_runs"
    run.attrs["assignment_keypoints_run"] = "assignment_kp_001"
    run.attrs["assignment_keypoint_contract"] = "subject_eyes_union_assignment_keypoints_v1"

    summary = assemble_mod.assemble_refined_subject_run(
        root,
        subject_run="subject_body_eye_union_swim_001",
        refined_run="refined_subject_masks_assignment_attrs_001",
    )

    assignment_summary = summary["eyes_union_assignment_summary"]
    assert assignment_summary["keypoint_group"] == "refined_keypoints_runs"
    assert assignment_summary["keypoint_run"] == "assignment_kp_001"
    assert assignment_summary["keypoint_source_kind"] == "assignment_keypoint_attrs"

    refined = root["refined_subject_masks_runs"]["refined_subject_masks_assignment_attrs_001"]
    assert refined.attrs["assignment_keypoints_run"] == "assignment_kp_001"
    eye_left_provenance = refined["components/eye_left/provenance"].attrs
    assert eye_left_provenance["assignment_keypoint_run"] == "assignment_kp_001"
    assert eye_left_provenance["assignment_keypoint_source_kind"] == "assignment_keypoint_attrs"


def test_refine_subject_masks_uses_component_provenance_for_assembled_run(monkeypatch) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    root = _build_assembly_root()
    monkeypatch.setattr(assemble_mod, "open_zarr_root", lambda *args, **kwargs: root)  # noqa: ARG005
    monkeypatch.setattr(batch_mod, "open_zarr_root", lambda *args, **kwargs: root)  # noqa: ARG005

    assemble_mod.assemble_refined_subject_masks(
        "in-memory.zarr",
        body_run="body_run_001",
        swim_run="swim_run_001",
        refined_run="refined_subject_masks_body_swim_001",
    )

    run = root["refined_subject_masks_runs"]["refined_subject_masks_body_swim_001"]
    edited_row = np.asarray(run["masks_roi"][0], dtype=np.uint8)
    edited_row[1, :, :] = 0
    run["masks_roi"][0] = edited_row

    summary = batch_mod.refine_subject_masks(
        "in-memory.zarr",
        refined_run="refined_subject_masks_body_swim_001",
        components=("swim_bladder",),
        roi_indices=[0, 1],
        chunk_size=1,
        scheduler="single-threaded",
    )

    assert summary["status"] == "updated"
    assert summary["source_subject_mask_run"] == "body_run_001"
    assert summary["changed_roi_count"] == 1
    assert summary["noop_roi_count"] == 1

    run = root["refined_subject_masks_runs"]["refined_subject_masks_body_swim_001"]
    np.testing.assert_array_equal(
        np.asarray(run["edit_applied"][:], dtype=bool),
        np.asarray(
            [
                [False, True],
                [False, False],
            ],
            dtype=bool,
        ),
    )
    swim_reasons = read_reason_labels(run["components/swim_bladder"])
    assert swim_reasons is not None
    assert swim_reasons.tolist() == ["manual_correction", "copied_from_source"]


def test_assemble_refined_subject_run_rejects_crop_snapshot_mismatch(monkeypatch) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    root = _build_assembly_root()
    root["subject_mask_runs"]["eye_run_001"].attrs["source_crop_signature"] = "sig-eye-mismatch"

    try:
        assemble_mod.assemble_refined_subject_run(
            root,
            body_run="body_run_001",
            eye_run="eye_run_001",
            refined_run="refined_subject_masks_bad_001",
        )
    except ValueError as exc:
        assert "Alignment mismatch for crop snapshot fields" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected crop snapshot mismatch to raise ValueError.")


def _mark_refined_component_review(run: zarr.Group, component_name: str, *, state: str = "approved") -> None:
    reviews = dict(run.attrs.get("component_review_statuses") or {})
    reviews[component_name] = {
        "state": state,
        "method": "manual",
        "intended_use": "training",
        "reviewer": "pytest",
        "timestamp_utc": "2026-04-03T00:00:00+00:00",
    }
    run.attrs["component_review_statuses"] = reviews
    run.attrs["refined_subject_mask_review_status"] = {
        "state": state,
        "method": "manual",
        "intended_use": "training",
        "reviewer": "pytest",
        "timestamp_utc": "2026-04-03T00:00:00+00:00",
    }


def _replace_refined_masks_with_rle(run: zarr.Group) -> np.ndarray:
    masks = np.asarray(run["masks_roi"][:], dtype=np.uint8)
    labels = [str(label) for label in run.attrs["mask_labels"]]
    del run["masks_roi"]
    write_component_rle_mask_store_from_dense(
        run,
        masks,
        component_names=labels,
        encode_row_chunk_size=1,
    )
    return masks


def test_assemble_refined_subject_run_can_import_components_from_refined_subject_runs(monkeypatch) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    root = _build_assembly_root()

    assemble_mod.assemble_refined_subject_run(
        root,
        body_run="body_run_001",
        refined_run="refined_subject_body_001",
    )
    assemble_mod.assemble_refined_subject_run(
        root,
        eye_run="eye_run_001",
        refined_run="refined_subject_eyes_001",
    )
    assemble_mod.assemble_refined_subject_run(
        root,
        swim_run="swim_run_001",
        refined_run="refined_subject_swim_001",
    )
    body_refined = root["refined_subject_masks_runs"]["refined_subject_body_001"]
    eye_refined = root["refined_subject_masks_runs"]["refined_subject_eyes_001"]
    swim_refined = root["refined_subject_masks_runs"]["refined_subject_swim_001"]
    _mark_refined_component_review(body_refined, "subject_body")
    _mark_refined_component_review(eye_refined, "eye_left")
    _mark_refined_component_review(eye_refined, "eye_right")
    _mark_refined_component_review(swim_refined, "swim_bladder")

    summary = assemble_mod.assemble_refined_subject_run(
        root,
        body_refined_run="refined_subject_body_001",
        eye_refined_run="refined_subject_eyes_001",
        swim_refined_run="refined_subject_swim_001",
        refined_run="refined_subject_assembled_from_refined_001",
    )

    assert summary["status"] == "updated"
    assert summary["component_names"] == ["subject_body", "eye_left", "eye_right", "swim_bladder"]
    assert summary["source_component_sources"]["subject_body"] == {
        "source_stage": "refined_subject_masks_runs",
        "source_run": "refined_subject_body_001",
    }
    assert summary["source_component_sources"]["eye_left"] == {
        "source_stage": "refined_subject_masks_runs",
        "source_run": "refined_subject_eyes_001",
    }
    assert summary["source_component_sources"]["swim_bladder"] == {
        "source_stage": "refined_subject_masks_runs",
        "source_run": "refined_subject_swim_001",
    }
    assert summary["source_subject_mask_run"] == "body_run_001"

    run = root["refined_subject_masks_runs"]["refined_subject_assembled_from_refined_001"]
    np.testing.assert_array_equal(
        np.asarray(run["masks_roi"][:, 0], dtype=np.uint8),
        np.asarray(body_refined["masks_roi"][:, 0], dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        np.asarray(run["masks_roi"][:, 1:3], dtype=np.uint8),
        np.asarray(eye_refined["masks_roi"][:, :], dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        np.asarray(run["masks_roi"][:, 3], dtype=np.uint8),
        np.asarray(swim_refined["masks_roi"][:, 0], dtype=np.uint8),
    )

    body_provenance = run["components/subject_body/provenance"].attrs
    swim_provenance = run["components/swim_bladder/provenance"].attrs
    assert body_provenance["source_stage"] == "refined_subject_masks_runs"
    assert body_provenance["source_run"] == "refined_subject_body_001"
    assert body_provenance["source_channels"] == ["subject_body"]
    assert body_provenance["upstream_component_provenance"]["source_stage"] == "subject_mask_runs"
    assert body_provenance["upstream_component_provenance"]["source_run"] == "body_run_001"
    assert run["components/eye_left/provenance"].attrs["source_run"] == "refined_subject_eyes_001"
    assert swim_provenance["source_stage"] == "refined_subject_masks_runs"
    assert swim_provenance["source_run"] == "refined_subject_swim_001"
    assert swim_provenance["upstream_component_provenance"]["source_run"] == "swim_run_001"

    reviews = run.attrs["component_review_statuses"]
    assert reviews["subject_body"]["state"] == "pending"
    assert reviews["swim_bladder"]["state"] == "pending"
    assert reviews["eye_left"]["state"] == "pending"
    assert reviews["eye_right"]["state"] == "pending"
    assert run.attrs["refined_subject_mask_review_status"]["state"] == "pending"
    assert body_provenance["source_review_status"]["state"] == "approved"
    assert swim_provenance["source_review_status"]["state"] == "approved"


def test_assemble_refined_subject_run_imports_compact_refined_subject_source(monkeypatch) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    root = _build_assembly_root()

    assemble_mod.assemble_refined_subject_run(
        root,
        body_run="body_run_001",
        refined_run="refined_subject_body_compact_source_001",
    )
    body_refined = root["refined_subject_masks_runs"]["refined_subject_body_compact_source_001"]
    expected_body_masks = _replace_refined_masks_with_rle(body_refined)
    _mark_refined_component_review(body_refined, "subject_body")

    summary = assemble_mod.assemble_refined_subject_run(
        root,
        body_refined_run="refined_subject_body_compact_source_001",
        refined_run="refined_subject_assembled_from_compact_refined_001",
    )

    assert summary["status"] == "updated"
    run = root["refined_subject_masks_runs"]["refined_subject_assembled_from_compact_refined_001"]
    np.testing.assert_array_equal(
        np.asarray(run["masks_roi"][:, 0], dtype=np.uint8),
        expected_body_masks[:, 0],
    )
    provenance = run["components/subject_body/provenance"].attrs
    assert provenance["source_stage"] == "refined_subject_masks_runs"
    assert provenance["source_run"] == "refined_subject_body_compact_source_001"
    assert provenance["source_mask_surface_path"] == "mask_rle"


def test_assemble_refined_subject_run_promotes_refined_component_reviews_when_requested(monkeypatch) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    root = _build_assembly_root()

    assemble_mod.assemble_refined_subject_run(
        root,
        body_run="body_run_001",
        refined_run="refined_subject_body_001",
    )
    assemble_mod.assemble_refined_subject_run(
        root,
        swim_run="swim_run_001",
        refined_run="refined_subject_swim_001",
    )
    body_refined = root["refined_subject_masks_runs"]["refined_subject_body_001"]
    swim_refined = root["refined_subject_masks_runs"]["refined_subject_swim_001"]
    _mark_refined_component_review(body_refined, "subject_body")
    _mark_refined_component_review(swim_refined, "swim_bladder")

    summary = assemble_mod.assemble_refined_subject_run(
        root,
        body_refined_run="refined_subject_body_001",
        swim_refined_run="refined_subject_swim_001",
        refined_run="refined_subject_assembled_promoted_from_refined_001",
        promote_source_review=True,
    )

    assert summary["status"] == "updated"
    assert summary["promote_source_review"] is True
    run = root["refined_subject_masks_runs"]["refined_subject_assembled_promoted_from_refined_001"]
    reviews = run.attrs["component_review_statuses"]
    assert reviews["subject_body"]["state"] == "approved"
    assert reviews["swim_bladder"]["state"] == "approved"
    assert run.attrs["refined_subject_mask_review_status"]["state"] == "approved"


def test_assemble_refined_subject_run_requires_approved_refined_components(monkeypatch) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    root = _build_assembly_root()
    assemble_mod.assemble_refined_subject_run(
        root,
        body_run="body_run_001",
        refined_run="refined_subject_body_001",
    )

    with pytest.raises(ValueError, match="component 'subject_body' is not approved"):
        assemble_mod.assemble_refined_subject_run(
            root,
            body_refined_run="refined_subject_body_001",
            refined_run="refined_subject_requires_approved_001",
        )

    summary = assemble_mod.assemble_refined_subject_run(
        root,
        body_refined_run="refined_subject_body_001",
        refined_run="refined_subject_draft_001",
        allow_unapproved_components=True,
    )
    assert summary["status"] == "updated"
    assert summary["component_names"] == ["subject_body"]


def test_assemble_refined_subject_run_rejects_duplicate_component_sources(monkeypatch) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    root = _build_assembly_root()
    assemble_mod.assemble_refined_subject_run(
        root,
        body_run="body_run_001",
        refined_run="refined_subject_body_001",
    )
    _mark_refined_component_review(
        root["refined_subject_masks_runs"]["refined_subject_body_001"],
        "subject_body",
    )

    with pytest.raises(ValueError, match="Duplicate source for component 'subject_body'"):
        assemble_mod.assemble_refined_subject_run(
            root,
            body_run="body_run_001",
            body_refined_run="refined_subject_body_001",
            refined_run="refined_subject_duplicate_001",
        )
