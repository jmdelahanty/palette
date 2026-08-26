from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from fisheye.analysis_workflows.eye_angle_candidate_execution import (
    eye_angle_logical_manifest_sha256,
)
from fisheye.analysis_workflows.eye_gaze_source_handle import (
    EyeGazeSourceHandleError,
    build_gaze_convention_review_receipt,
    load_eye_gaze_source_handle,
    validate_gaze_convention_review_receipt,
)
from fisheye.shared.json_safety import decode_null_terminated_text
from fisheye.shared.zarr_helpers import consolidate_metadata_capture_expected_warnings
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import mark_run_complete
from tests.unit.fisheye.test_check_recording_steps import (
    _populate_exact_eye_angle_v7_run,
)


def _numeric_validation(run_path: str) -> dict[str, object]:
    return {
        "schema_id": "palette.gaze_convention_validation.v1",
        "schema_version": 1,
        "created_at_utc": "2026-08-25T12:00:00+00:00",
        "status": "pass",
        "zarr_path": "/archive/recording.zarr",
        "eye_angle_run": run_path.rsplit("/", 1)[-1],
        "eye_angle_run_path": run_path,
        "read_only": True,
        "sampling": {"sample_rows": 12},
        "comparison_contract": {
            "object_angle_field": "egocentric_bearing/per_chaser/bearing_deg",
            "eye_angle_fields": ["left_gaze_signed_deg", "right_gaze_signed_deg"],
            "coordinate_frame": "fish_body_frame",
            "zero": "fish_forward",
            "positive": "anatomical_left",
            "explicitly_not_comparable_fields": [
                "left_eye_angle_deg",
                "right_eye_angle_deg",
            ],
        },
        "checks": [{"name": "all_numeric_identities", "passed": True}],
        "direction_assumption": {
            "name": "ellipse_axis_direction_assumption",
            "passed": None,
            "review_required": True,
        },
        "review_png": "/review/gaze.png",
        "review_mask_source_path": "analysis/refined_subject_masks_runs/masks",
        "review_row_indices": [0, 1],
    }


def _receipt(run_path: str, logical_digest: str) -> dict[str, object]:
    return build_gaze_convention_review_receipt(
        numeric_validation=_numeric_validation(run_path),
        source_eye_logical_sha256=logical_digest,
        reviewer="reviewer@example.org",
        reviewed_at_utc="2026-08-25T12:30:00+00:00",
        review_artifact_sha256="a" * 64,
    )


def test_review_receipt_is_exact_and_self_digesting() -> None:
    run_path = "analysis/eye_angle_runs/eye-v7"
    receipt = _receipt(run_path, "b" * 64)

    assert validate_gaze_convention_review_receipt(
        receipt,
        expected_run_path=run_path,
        expected_logical_sha256="b" * 64,
    ) == receipt
    altered = deepcopy(receipt)
    altered["biological_direction_review"]["reviewer"] = "somebody-else"
    with pytest.raises(EyeGazeSourceHandleError, match="self-digest"):
        validate_gaze_convention_review_receipt(
            altered,
            expected_run_path=run_path,
            expected_logical_sha256="b" * 64,
        )


def test_review_receipt_requires_a_rendered_biological_review() -> None:
    validation = _numeric_validation("analysis/eye_angle_runs/eye-v7")
    validation["review_png"] = None

    with pytest.raises(EyeGazeSourceHandleError, match="review_png"):
        build_gaze_convention_review_receipt(
            numeric_validation=validation,
            source_eye_logical_sha256="b" * 64,
            reviewer="reviewer@example.org",
            reviewed_at_utc="2026-08-25T12:30:00+00:00",
            review_artifact_sha256="a" * 64,
        )


def test_exact_eye_handle_aligns_only_acquisition_frame_identity(tmp_path) -> None:
    archive = tmp_path / "analysis.zarr"
    root = open_zarr_root(archive, mode="w-")
    root.attrs["recording_id"] = "recording-1"
    parent = root.require_group("analysis/eye_angle_runs")
    run = parent.create_group("eye-v7")
    _populate_exact_eye_angle_v7_run(run)
    run.attrs["stage_selector_eligible"] = False

    angle_names = tuple(
        decode_null_terminated_text(row)
        for row in np.asarray(run["angle_channel_index/name"][:])
    )
    qa_names = tuple(
        decode_null_terminated_text(row)
        for row in np.asarray(run["qa_channel_index/name"][:])
    )
    frame_angles = np.asarray(run["frame_angles"][:])
    frame_angles[:, angle_names.index("left_gaze_signed_deg")] = [1.0, 2.0, 3.0]
    frame_angles[:, angle_names.index("right_gaze_signed_deg")] = [4.0, 5.0, 6.0]
    frame_angles[:, angle_names.index("vergence_eye_angle_deg")] = [7.0, 8.0, 9.0]
    run["frame_angles"][:] = frame_angles
    frame_qa = np.asarray(run["frame_qa"][:])
    frame_qa[:, qa_names.index("valid_frame")] = 1
    run["frame_qa"][:] = frame_qa
    mark_run_complete(run, parent_group=parent, run_name="eye-v7")
    logical_digest = eye_angle_logical_manifest_sha256(run)
    receipt = _receipt("analysis/eye_angle_runs/eye-v7", logical_digest)
    consolidate_metadata_capture_expected_warnings(archive)

    handle = load_eye_gaze_source_handle(
        archive,
        run_name="eye-v7",
        convention_receipt=receipt,
        channel_variant="raw",
    )
    gaze, gaze_valid, vergence, vergence_valid = handle.align_to_acquisition_frames(
        np.asarray([2, 0], dtype=np.int64)
    )

    np.testing.assert_allclose(gaze, [[3.0, 6.0], [1.0, 4.0]])
    assert gaze_valid.tolist() == [[True, True], [True, True]]
    np.testing.assert_allclose(vergence, [9.0, 7.0])
    assert vergence_valid.tolist() == [True, True]
    assert handle.selector_eligible is False
    assert handle.convention_receipt_sha256 == receipt["receipt_sha256"]


def test_eye_handle_rejects_receipt_for_changed_payload(tmp_path) -> None:
    archive = tmp_path / "analysis.zarr"
    root = open_zarr_root(archive, mode="w-")
    root.attrs["recording_id"] = "recording-1"
    parent = root.require_group("analysis/eye_angle_runs")
    run = parent.create_group("eye-v7")
    _populate_exact_eye_angle_v7_run(run)
    run.attrs["stage_selector_eligible"] = False
    mark_run_complete(run, parent_group=parent, run_name="eye-v7")
    receipt = _receipt(
        "analysis/eye_angle_runs/eye-v7", eye_angle_logical_manifest_sha256(run)
    )
    run["frame_angles"][0, 0] = 99.0
    consolidate_metadata_capture_expected_warnings(archive)

    with pytest.raises(EyeGazeSourceHandleError, match="exact eye run payload"):
        load_eye_gaze_source_handle(
            archive,
            run_name="eye-v7",
            convention_receipt=receipt,
            channel_variant="raw",
        )
