from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from fisheye.shared.provenance_attrs import build_source_keypoints_attrs
from fisheye.shared.zarr.stage_arrays import SUBJECT_MASKS_SPEC, validate_run
from fisheye.utils import merge_subject_mask_runs as mod


def _create_subject_run(
    root: zarr.Group,
    *,
    run_name: str,
    mask_labels: list[str],
    available_channels: np.ndarray,
    masks: np.ndarray,
    probs: np.ndarray | None,
    source_crop_run: str = "crop_001",
    frame_indices: np.ndarray | None = None,
    frame_counts: np.ndarray | None = None,
    detection_indices: np.ndarray | None = None,
    detection_source: np.ndarray | None = None,
    source_keypoints_run: str = "refined_keypoints_001",
    source_keypoint_group: str = "refined_keypoints_runs",
    probabilities_encoding: str = "unit_float",
    source_crop_storage_mode: str = "geometry_only",
    source_crop_signature: str = "sig-001",
    source_crop_revision: int = 4,
    source_detect_review_status_ref: str | None = "refined_detect_runs/refined_detect_001/review_status",
) -> zarr.Group:
    parent = root.require_group("subject_mask_runs")
    parent.attrs["latest"] = run_name
    run = parent.create_group(run_name)
    run.attrs.update(
        {
            "source_crop_run": source_crop_run,
            "source_crop_storage_mode": source_crop_storage_mode,
            "source_crop_signature": source_crop_signature,
            "source_crop_revision": int(source_crop_revision),
            **build_source_keypoints_attrs(source_keypoints_run, include_legacy_alias=True),
            "source_keypoint_group": source_keypoint_group,
            "label_schema_id": "subject_v1_lr" if "eye_left" in mask_labels else "subject_v1_union",
            "mask_labels": list(mask_labels),
            "output_semantics": "multilabel",
            "overlap_policy": "independent_sigmoid",
            "probabilities_encoding": probabilities_encoding,
            "method": f"{run_name}_method",
            "created_at_utc": "2026-04-10T12:00:00+00:00",
        }
    )
    if source_detect_review_status_ref is not None:
        run.attrs["source_detect_review_status_ref"] = source_detect_review_status_ref

    frame_indices_arr = np.asarray([0, 0], dtype=np.int32) if frame_indices is None else np.asarray(frame_indices, dtype=np.int32)
    frame_counts_arr = np.asarray([2], dtype=np.int32) if frame_counts is None else np.asarray(frame_counts, dtype=np.int32)
    detection_indices_arr = (
        np.asarray([5, 6], dtype=np.int32) if detection_indices is None else np.asarray(detection_indices, dtype=np.int32)
    )
    detection_source_arr = (
        np.asarray([0, 1], dtype=np.int8) if detection_source is None else np.asarray(detection_source, dtype=np.int8)
    )

    run.create_array("frame_indices", data=frame_indices_arr, overwrite=True)
    run.create_array("frame_counts", data=frame_counts_arr, overwrite=True)
    run.create_array("detection_indices", data=detection_indices_arr, overwrite=True)
    run.create_array("detection_source", data=detection_source_arr, overwrite=True)
    run.create_array(
        "masks_roi",
        data=np.asarray(masks, dtype=np.uint8),
        chunks=(int(masks.shape[0]), 1, int(masks.shape[2]), int(masks.shape[3])),
        overwrite=True,
    )
    if probs is not None:
        run.create_array(
            "mask_probs_roi",
            data=np.asarray(probs),
            chunks=(int(masks.shape[0]), 1, int(masks.shape[2]), int(masks.shape[3])),
            overwrite=True,
        )
    run.create_array("available_channels", data=np.asarray(available_channels, dtype=bool), overwrite=True)
    return run


def test_merge_subject_mask_runs_combines_body_and_eye_components(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zarr_path = tmp_path / "recording_training.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    monkeypatch.setattr(
        mod,
        "get_git_info",
        lambda repo_path=None: {  # noqa: ARG005
            "commit_hash": "d" * 40,
            "short_hash": "dddddddd",
            "branch": "main",
            "is_dirty": False,
            "remote_url": "git@example.com:palette.git",
        },
    )
    monkeypatch.setattr(
        mod,
        "get_environment_info",
        lambda **_kwargs: {
            "environment": {"python": "3.11"},
            "platform": {
                "hostname": "merge-host",
                "system": "Linux",
                "release": "6.8",
                "python_version": "3.11.13",
                "machine": "x86_64",
            },
        },
    )

    body_masks = np.asarray(
        [
            [
                [[1, 1], [0, 0]],
                [[0, 0], [0, 0]],
                [[0, 0], [0, 0]],
            ],
            [
                [[0, 1], [1, 1]],
                [[0, 0], [0, 0]],
                [[0, 0], [0, 0]],
            ],
        ],
        dtype=np.uint8,
    )
    body_probs = np.asarray(
        [
            [
                [[0.9, 0.8], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ],
            [
                [[0.0, 0.7], [0.6, 0.5]],
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ],
        ],
        dtype=np.float16,
    )
    _create_subject_run(
        root,
        run_name="sam_subject_masks_canary_001",
        mask_labels=["subject_body", "eyes_union", "swim_bladder"],
        available_channels=np.asarray([True, False, False], dtype=bool),
        masks=body_masks,
        probs=body_probs,
    )

    eye_masks = np.asarray(
        [
            [
                [[0, 0], [0, 0]],
                [[1, 0], [0, 0]],
                [[0, 1], [0, 0]],
                [[0, 0], [0, 0]],
            ],
            [
                [[0, 0], [0, 0]],
                [[0, 0], [1, 0]],
                [[0, 0], [0, 1]],
                [[0, 0], [0, 0]],
            ],
        ],
        dtype=np.uint8,
    )
    _create_subject_run(
        root,
        run_name="subject_masks_from_refined_eye_masks_001",
        mask_labels=["subject_body", "eye_left", "eye_right", "swim_bladder"],
        available_channels=np.asarray([False, True, True, False], dtype=bool),
        masks=eye_masks,
        probs=None,
    )

    summary = mod.merge_subject_mask_runs(
        zarr_path,
        body_run="sam_subject_masks_canary_001",
        eye_run="subject_masks_from_refined_eye_masks_001",
        run_name="subject_masks_canary_body_eyes_001",
        apply=True,
    )

    assert summary["status"] == "updated"
    subject_parent = root["subject_mask_runs"]
    assert subject_parent.attrs["latest"] == "subject_masks_canary_body_eyes_001"
    run = subject_parent["subject_masks_canary_body_eyes_001"]

    expected_masks = np.asarray(
        [
            [
                [[1, 1], [0, 0]],
                [[1, 0], [0, 0]],
                [[0, 1], [0, 0]],
                [[0, 0], [0, 0]],
            ],
            [
                [[0, 1], [1, 1]],
                [[0, 0], [1, 0]],
                [[0, 0], [0, 1]],
                [[0, 0], [0, 0]],
            ],
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(np.asarray(run["masks_roi"][:], dtype=np.uint8), expected_masks)
    assert tuple(run["masks_roi"].chunks) == (2, 1, 2, 2)
    assert run["masks_roi"].fill_value == 0

    expected_body_probs = body_probs.astype(np.float32)
    expected_probs = np.asarray(
        [
            [
                [[expected_body_probs[0, 0, 0, 0], expected_body_probs[0, 0, 0, 1]], [0.0, 0.0]],
                [[1.0, 0.0], [0.0, 0.0]],
                [[0.0, 1.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ],
            [
                [
                    [0.0, expected_body_probs[1, 0, 0, 1]],
                    [expected_body_probs[1, 0, 1, 0], expected_body_probs[1, 0, 1, 1]],
                ],
                [[0.0, 0.0], [1.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(np.asarray(run["mask_probs_roi"][:], dtype=np.float32), expected_probs, atol=1e-6)
    assert tuple(run["mask_probs_roi"].chunks) == (2, 1, 2, 2)
    assert run["mask_probs_roi"].fill_value == 0.0

    np.testing.assert_array_equal(
        np.asarray(run["available_channels"][:], dtype=bool),
        np.asarray([True, True, True, False], dtype=bool),
    )
    np.testing.assert_array_equal(
        np.asarray(run["metrics/mask_present"][:], dtype=bool),
        np.asarray(
            [
                [True, True, True, False],
                [True, True, True, False],
            ],
            dtype=bool,
        ),
    )
    np.testing.assert_allclose(
        np.asarray(run["metrics/prob_max"][:], dtype=np.float32),
        np.asarray(
            [
                [expected_body_probs[0, 0].max(), 1.0, 1.0, 0.0],
                [expected_body_probs[1, 0].max(), 1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(run["metrics/area_px"][:], dtype=np.float32),
        np.asarray(
            [
                [2.0, 1.0, 1.0, 0.0],
                [3.0, 1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        atol=1e-6,
    )

    assert run.attrs["label_schema_id"] == "subject_v1_lr"
    assert run.attrs["run_semantics"] == "merged_subject_components"
    assert run.attrs["probabilities_dtype"] == "float32"
    assert run.attrs["probabilities_encoding"] == "unit_float"
    assert run.attrs["source_crop_storage_mode"] == "geometry_only"
    assert run.attrs["source_crop_signature"] == "sig-001"
    assert run.attrs["source_crop_revision"] == 4
    assert run.attrs["source_detect_review_status_ref"] == "refined_detect_runs/refined_detect_001/review_status"
    assert run.attrs["source_body_subject_mask_run"] == "sam_subject_masks_canary_001"
    assert run.attrs["source_eye_subject_mask_run"] == "subject_masks_from_refined_eye_masks_001"
    assert run.attrs["source_keypoints_run"] == "refined_keypoints_001"
    assert run.attrs["source_keypoint_run"] == "refined_keypoints_001"
    assert run.attrs["source_keypoint_group"] == "refined_keypoints_runs"
    assert run.attrs["summary_statistics"]["rows_total"] == 2
    assert run.attrs["summary_statistics"]["rows_with_nonempty_masks"] == 2
    assert run.attrs["summary_statistics"]["rows_with_subject_body_masks"] == 2
    assert run.attrs["summary_statistics"]["rows_with_eye_left_masks"] == 2
    assert run.attrs["summary_statistics"]["rows_with_eye_right_masks"] == 2
    assert run.attrs["summary_statistics"]["rows_with_swim_bladder_masks"] == 0

    component_provenance = run.attrs["component_provenance"]
    assert component_provenance["components"]["subject_body"]["source_run"] == "sam_subject_masks_canary_001"
    assert component_provenance["components"]["eye_left"]["source_run"] == "subject_masks_from_refined_eye_masks_001"
    assert component_provenance["components"]["eye_right"]["source_run"] == "subject_masks_from_refined_eye_masks_001"
    assert component_provenance["components"]["subject_body"]["source_crop_signature"] == "sig-001"
    assert component_provenance["components"]["eye_left"]["source_detect_review_status_ref"] == (
        "refined_detect_runs/refined_detect_001/review_status"
    )
    body_provenance = run["components"]["subject_body"]["provenance"].attrs
    left_provenance = run["components"]["eye_left"]["provenance"].attrs
    right_provenance = run["components"]["eye_right"]["provenance"].attrs
    assert body_provenance["source_run"] == "sam_subject_masks_canary_001"
    assert body_provenance["source_channels"] == ["subject_body"]
    assert body_provenance["source_label_schema_id"] == "subject_v1_union"
    assert body_provenance["source_crop_run"] == "crop_001"
    assert body_provenance["source_crop_signature"] == "sig-001"
    assert body_provenance["source_crop_revision"] == 4
    assert body_provenance["source_detect_review_status_ref"] == "refined_detect_runs/refined_detect_001/review_status"
    assert left_provenance["source_run"] == "subject_masks_from_refined_eye_masks_001"
    assert left_provenance["source_channels"] == ["eye_left"]
    assert left_provenance["source_label_schema_id"] == "subject_v1_lr"
    assert left_provenance["source_crop_signature"] == "sig-001"
    assert right_provenance["source_run"] == "subject_masks_from_refined_eye_masks_001"
    assert right_provenance["source_channels"] == ["eye_right"]
    assert right_provenance["source_label_schema_id"] == "subject_v1_lr"
    provenance = run.attrs["provenance"]
    assert provenance["stage"] == "subject_masks"
    assert provenance["parameters"]["run_semantics"] == "merged_subject_components"
    assert provenance["inputs"]["source_crop_run"] == "crop_001"
    assert provenance["inputs"]["source_crop_signature"] == "sig-001"
    assert provenance["inputs"]["source_crop_revision"] == 4
    assert provenance["inputs"]["source_detect_review_status_ref"] == (
        "refined_detect_runs/refined_detect_001/review_status"
    )
    assert provenance["inputs"]["source_body_subject_mask_run"] == "sam_subject_masks_canary_001"
    assert provenance["inputs"]["source_eye_subject_mask_run"] == "subject_masks_from_refined_eye_masks_001"
    assert provenance["git"]["commit"] == "d" * 40
    assert provenance["platform"]["hostname"] == "merge-host"
    assert provenance["artifacts"]["body_probability_source_path"] == "subject_mask_runs/sam_subject_masks_canary_001/mask_probs_roi"
    assert provenance["artifacts"]["eye_probability_source_path"] == (
        "subject_mask_runs/subject_masks_from_refined_eye_masks_001/masks_roi"
    )

    validation = validate_run(run, SUBJECT_MASKS_SPEC)
    assert validation.valid, validation.errors


def test_merge_subject_mask_runs_rejects_detection_index_mismatch(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_training.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")

    _create_subject_run(
        root,
        run_name="body_001",
        mask_labels=["subject_body", "eyes_union", "swim_bladder"],
        available_channels=np.asarray([True, False, False], dtype=bool),
        masks=np.zeros((2, 3, 2, 2), dtype=np.uint8),
        probs=np.zeros((2, 3, 2, 2), dtype=np.float16),
    )
    _create_subject_run(
        root,
        run_name="eyes_001",
        mask_labels=["subject_body", "eye_left", "eye_right", "swim_bladder"],
        available_channels=np.asarray([False, True, True, False], dtype=bool),
        masks=np.zeros((2, 4, 2, 2), dtype=np.uint8),
        probs=None,
        detection_indices=np.asarray([5, 99], dtype=np.int32),
    )

    with pytest.raises(ValueError, match="Alignment mismatch for detection_indices"):
        mod.merge_subject_mask_runs(
            zarr_path,
            body_run="body_001",
            eye_run="eyes_001",
            run_name="merged_001",
            apply=False,
        )


def test_merge_subject_mask_runs_rejects_crop_snapshot_mismatch(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_training.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")

    _create_subject_run(
        root,
        run_name="body_001",
        mask_labels=["subject_body", "eyes_union", "swim_bladder"],
        available_channels=np.asarray([True, False, False], dtype=bool),
        masks=np.zeros((2, 3, 2, 2), dtype=np.uint8),
        probs=np.zeros((2, 3, 2, 2), dtype=np.float16),
        source_crop_signature="sig-body",
    )
    _create_subject_run(
        root,
        run_name="eyes_001",
        mask_labels=["subject_body", "eye_left", "eye_right", "swim_bladder"],
        available_channels=np.asarray([False, True, True, False], dtype=bool),
        masks=np.zeros((2, 4, 2, 2), dtype=np.uint8),
        probs=None,
        source_crop_signature="sig-eyes",
    )

    with pytest.raises(ValueError, match="Alignment mismatch for crop snapshot fields"):
        mod.merge_subject_mask_runs(
            zarr_path,
            body_run="body_001",
            eye_run="eyes_001",
            run_name="merged_001",
            apply=False,
        )


def test_resolve_shared_crop_snapshot_returns_common_normalized_fields() -> None:
    body = SimpleNamespace(
        source_crop_snapshot={
            "source_crop_storage_mode": "geometry_only",
            "source_crop_signature": "{'signature_version': 2}",
            "source_crop_revision": 4,
            "source_detect_review_status_ref": "refined_detect_runs/refined_detect_001/review_status",
        }
    )
    eyes = SimpleNamespace(
        source_crop_snapshot={
            "source_crop_storage_mode": "geometry_only",
            "source_crop_signature": "{'signature_version': 2}",
            "source_crop_revision": 4,
            "source_detect_review_status_ref": "refined_detect_runs/refined_detect_001/review_status",
        }
    )

    resolved = mod._resolve_shared_crop_snapshot(body, eyes)

    assert resolved == {
        "source_crop_storage_mode": "geometry_only",
        "source_crop_signature": "{'signature_version': 2}",
        "source_crop_revision": 4,
        "source_detect_review_status_ref": "refined_detect_runs/refined_detect_001/review_status",
    }


def test_resolve_shared_crop_snapshot_requires_required_fields() -> None:
    body = SimpleNamespace(
        source_crop_snapshot={
            "source_crop_storage_mode": "geometry_only",
            "source_crop_signature": "sig-001",
            "source_crop_revision": 4,
        }
    )
    eyes = SimpleNamespace(
        source_crop_snapshot={
            "source_crop_signature": "sig-001",
            "source_crop_revision": 4,
        }
    )

    with pytest.raises(ValueError, match="Missing required crop snapshot fields for merge"):
        mod._resolve_shared_crop_snapshot(body, eyes)
