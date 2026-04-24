from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from fisheye.refinement import assemble_refined_subject_masks as assemble_mod
from fisheye.refinement import refine_subject_masks as batch_mod
from fisheye.shared.detect_reason_codec import read_reason_labels
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
    run.create_array("available_channels", data=np.asarray(available_channels, dtype=bool), overwrite=True)
    run.create_array("masks_roi", data=np.asarray(masks, dtype=np.uint8), overwrite=True)
    return run


def _create_refined_eye_run(root: zarr.Group, *, run_name: str = "refined_eye_masks_001") -> zarr.Group:
    parent = root.require_group("refined_eye_masks_runs")
    parent.attrs["latest"] = run_name
    run = parent.create_group(run_name)
    run.attrs.update(
        {
            "source_crop_run": "crop_001",
            "source_eye_masks_run": "eye_masks_001",
            "source_keypoints_run": "refined_kp_001",
            "source_keypoint_run": "refined_kp_001",
            "source_keypoint_group": "refined_keypoints_runs",
            "method": "refine_eye_masks_v1",
            "eye_labels": ["eye_left", "eye_right"],
            "created_at_utc": "2026-04-02T00:00:00+00:00",
        }
    )
    masks = np.zeros((2, 2, 8, 8), dtype=np.uint8)
    masks[0, 0, 1:4, 1:4] = 1
    masks[0, 1, 1:4, 4:7] = 1
    masks[1, 0, 3:6, 1:4] = 1
    masks[1, 1, 3:6, 4:7] = 1
    run.create_array("masks_roi", data=masks, overwrite=True)
    run.create_array("detection_source", data=np.asarray([0, 0], dtype=np.int8), overwrite=True)
    run.create_array("frame_indices", data=np.asarray([10, 11], dtype=np.int32), overwrite=True)
    run.create_array("detection_indices", data=np.asarray([0, 1], dtype=np.int32), overwrite=True)
    run.create_array("frame_counts", data=np.asarray([1, 1], dtype=np.int32), overwrite=True)
    return run


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
    assert run.attrs["source_subject_mask_run"] == "body_run_001"
    assert run.attrs["source_body_subject_mask_run"] == "body_run_001"
    assert run.attrs["source_eye_subject_mask_run"] == "eye_run_001"
    assert run.attrs["source_swim_subject_mask_run"] == "swim_run_001"
    assert run.attrs["source_crop_storage_mode"] == "geometry_only"
    assert run.attrs["source_crop_signature"] == "{'signature_version': 2, 'crop_revision': 4}"
    assert run.attrs["source_crop_revision"] == 4
    assert run.attrs["source_detect_review_status_ref"] == "refined_detect_runs/refined_detect_001/review_status"
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


def test_assemble_refined_subject_run_can_seed_eyes_directly_from_refined_eye_masks(monkeypatch) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    root = _build_assembly_root()
    del root["subject_mask_runs"]["eye_run_001"]
    _create_refined_eye_run(root)

    summary = assemble_mod.assemble_refined_subject_run(
        root,
        body_run="body_run_001",
        refined_eye_run="refined_eye_masks_001",
        swim_run="swim_run_001",
        refined_run="refined_subject_masks_direct_refined_eye_001",
    )

    assert summary["status"] == "updated"
    assert summary["component_names"] == ["subject_body", "eye_left", "eye_right", "swim_bladder"]
    assert summary["source_refined_eye_masks_run"] == "refined_eye_masks_001"
    assert summary["source_component_sources"]["eye_left"] == {
        "source_stage": "refined_eye_masks_runs",
        "source_run": "refined_eye_masks_001",
    }

    run = root["refined_subject_masks_runs"]["refined_subject_masks_direct_refined_eye_001"]
    assert run.attrs["source_subject_mask_run"] == "body_run_001"
    assert run.attrs["source_refined_eye_masks_run"] == "refined_eye_masks_001"
    assert "source_eye_subject_mask_run" not in run.attrs
    np.testing.assert_array_equal(
        np.asarray(run["available_channels"][:], dtype=bool),
        np.asarray([True, True, True, True], dtype=bool),
    )
    left_provenance = run["components/eye_left/provenance"].attrs
    right_provenance = run["components/eye_right/provenance"].attrs
    assert left_provenance["source_stage"] == "refined_eye_masks_runs"
    assert left_provenance["source_run"] == "refined_eye_masks_001"
    assert left_provenance["source_method"] == "refine_eye_masks_v1"
    assert left_provenance["source_channels"] == ["eye_left"]
    assert left_provenance["source_eye_masks_run"] == "eye_masks_001"
    assert right_provenance["source_stage"] == "refined_eye_masks_runs"
    assert right_provenance["source_run"] == "refined_eye_masks_001"
    assert right_provenance["source_channels"] == ["eye_right"]
    np.testing.assert_array_equal(
        np.asarray(run["components/eye_left/geometry/ellipse_success"][:], dtype=bool),
        np.ones((2,), dtype=bool),
    )
    np.testing.assert_array_equal(
        np.asarray(run["relations/eye_pair/metrics/separation_valid"][:], dtype=bool),
        np.ones((2,), dtype=bool),
    )
    refined = review_mod._open_existing_refined_subject_run(  # noqa: SLF001
        root,
        "refined_subject_masks_direct_refined_eye_001",
    )
    _primary_source, component_sources = review_mod._load_refined_component_source_runs(root, refined)  # noqa: SLF001
    assert component_sources["eye_left"].run_name == "refined_eye_masks_001"
    assert component_sources["eye_left"].mask_labels == ("eye_left", "eye_right")
    np.testing.assert_array_equal(
        np.asarray(component_sources["eye_left"].masks_roi[:, 0], dtype=np.uint8),
        np.asarray(run["masks_roi"][:, 1], dtype=np.uint8),
    )


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
    assert reviews["subject_body"]["state"] == "approved"
    assert reviews["swim_bladder"]["state"] == "approved"
    assert reviews["eye_left"]["state"] == "approved"
    assert reviews["eye_right"]["state"] == "approved"
    assert run.attrs["refined_subject_mask_review_status"]["state"] == "approved"


def test_assemble_refined_subject_run_rejects_duplicate_component_sources(monkeypatch) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    root = _build_assembly_root()
    assemble_mod.assemble_refined_subject_run(
        root,
        body_run="body_run_001",
        refined_run="refined_subject_body_001",
    )

    with pytest.raises(ValueError, match="Duplicate source for component 'subject_body'"):
        assemble_mod.assemble_refined_subject_run(
            root,
            body_run="body_run_001",
            body_refined_run="refined_subject_body_001",
            refined_run="refined_subject_duplicate_001",
        )
