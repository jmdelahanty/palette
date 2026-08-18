from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import fisheye.shared.subject_position_detection_source as detection_source
import fisheye.shared.subject_position_keypoint_source as keypoint_source
import fisheye.shared.subject_position_mask_source as mask_source
import fisheye.shared.subject_position_preparation as preparation
from fisheye.shared.anatomy_profile import (
    anatomy_profile_sha256,
    load_anatomy_profile,
    source_binding_sha256,
)
from fisheye.shared.coordinate_identity import (
    OBSERVATION_INSTANCE_DOMAIN,
    build_row_identity_contract,
)
from fisheye.shared.subject_position_expression import (
    ComponentSourceBinding,
    DETECTION_BBOX_CENTROID_ESTIMATOR_ID,
    KEYPOINT_ANATOMICAL_TRIAD_MEAN_ESTIMATOR_ID,
    MASK_COMPONENT_ANATOMICAL_TRIAD_MEAN_ESTIMATOR_ID,
    PointArrayBinding,
    PointExpressionBindings,
    SUBJECT_BODY_MASK_CENTROID_ESTIMATOR_ID,
)
from fisheye.shared.subject_position_policy import (
    SUBJECT_POSITION_CANARY_NO_DEFAULT_POLICY_ID,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


_PROFILE_PATH = (
    Path(__file__).resolve().parents[3]
    / "configs/fisheye/anatomy_profiles/zebrafish_larva_v1.json"
)


def _identity(keys: np.ndarray, run_path: str) -> SimpleNamespace:
    return SimpleNamespace(
        contract=build_row_identity_contract(
            domain=OBSERVATION_INSTANCE_DOMAIN,
            values=keys,
        ),
        record_ref=f"/{run_path}@row_identity_contract",
        record_sha256="a" * 64,
    )


def _frame() -> SimpleNamespace:
    return SimpleNamespace(
        record_ref="/coordinate_frames/source_camera@pixel_frame_authority",
        record_sha256="b" * 64,
        endpoint=SimpleNamespace(
            width=640,
            height=480,
            selector="pixel_frame_authority",
        ),
    )


def _detection_source() -> detection_source.BoundDetectionPositionSource:
    keys = np.array([11, 22], dtype=np.uint64)
    record = {
        "schema_id": detection_source.DETECTION_POSITION_SOURCE_SCHEMA_ID,
        "schema_version": 1,
        "source_modality": "detection",
        "run_path": "detect_runs/canonical",
    }
    return detection_source.BoundDetectionPositionSource(
        source_modality="detection",
        source_kind="native_detection",
        run_path="detect_runs/canonical",
        row_identity=_identity(keys, "detect_runs/canonical"),
        instance_key=keys,
        source_acquisition_frame_index=np.array([3, 4], dtype=np.int64),
        source_row_index=np.array([0, 1], dtype=np.int64),
        source_camera_frame=_frame(),
        source_binding_record=record,
        source_binding_digest=canonical_json_sha256(record),
        point_expression_bindings=PointExpressionBindings(
            bboxes={
                "bbox_img_xyxy": {
                    "xyxy": np.array(
                        [[0.0, 2.0, 4.0, 6.0], [10.0, 20.0, 14.0, 28.0]],
                        dtype=np.float32,
                    ),
                    "valid": np.array([True, True], dtype=bool),
                }
            }
        ),
        observation_validity=np.array([True, True], dtype=bool),
        direct_consolidated_evidence={"mode": "test"},
        _analysis_zarr=None,
        _root_node=SimpleNamespace(),
        _verification_seal=detection_source._BOUND_DETECTION_POSITION_SOURCE_SEAL,
    )


def _keypoint_source() -> keypoint_source.BoundKeypointPositionSource:
    profile = load_anatomy_profile(_PROFILE_PATH)
    binding_id = "zebrafish_larva_keypoint_traditional_v3_v1"
    binding = profile.binding(binding_id)
    keys = np.array([31, 32], dtype=np.uint64)
    points = {
        "swim_bladder": np.array([[0.0, 0.0], [3.0, 6.0]], dtype=np.float32),
        "eye_left": np.array([[3.0, 0.0], [6.0, 6.0]], dtype=np.float32),
        "eye_right": np.array([[0.0, 3.0], [3.0, 9.0]], dtype=np.float32),
    }
    bindings = PointExpressionBindings(
        keypoints={
            role: PointArrayBinding(
                values=value,
                valid=np.array([True, True], dtype=bool),
            )
            for role, value in points.items()
        }
    )
    return keypoint_source.BoundKeypointPositionSource(
        source_modality="keypoint",
        source_kind="canonical_keypoints_v2",
        run_path="keypoints_runs/canonical",
        row_identity=_identity(keys, "keypoints_runs/canonical"),
        instance_key=keys,
        source_acquisition_frame_index=np.array([7, 8], dtype=np.int64),
        source_row_index=np.array([0, 1], dtype=np.int64),
        source_camera_frame=_frame(),
        source_binding_record=binding,
        source_binding_digest=source_binding_sha256(binding),
        expression_bindings=bindings,
        run_manifest_digest="c" * 64,
        logical_content_digest="d" * 64,
        metadata_declarations_digest="e" * 64,
        skeleton_id="pose_skel_traditional_v3",
        skeleton_digest="f" * 64,
        pose_schema_binding_digest="1" * 64,
        _analysis_zarr=SimpleNamespace(),
        _anatomy_profile=profile,
        _binding_id=binding_id,
        _verification_seal=keypoint_source._BOUND_SOURCE_SEAL,
    )


def _mask_source() -> mask_source.BoundSubjectMaskPositionSource:
    profile = load_anatomy_profile(_PROFILE_PATH)
    binding_id = "zebrafish_larva_subject_mask_lr_v1"
    binding = profile.binding(binding_id)
    keys = np.array([41, 42], dtype=np.uint64)
    components = {
        "swim_bladder": np.array([[0.0, 0.0], [3.0, 6.0]], dtype=np.float32),
        "eye_left": np.array([[3.0, 0.0], [6.0, 6.0]], dtype=np.float32),
        "eye_right": np.array([[0.0, 3.0], [3.0, 9.0]], dtype=np.float32),
        "subject_body": np.array([[9.0, 12.0], [15.0, 18.0]], dtype=np.float32),
    }
    bindings = PointExpressionBindings(
        components={
            role: ComponentSourceBinding(
                centroids=value,
                valid=np.array([True, True], dtype=bool),
            )
            for role, value in components.items()
        }
    )
    return mask_source.BoundSubjectMaskPositionSource(
        source_modality="subject_mask",
        source_kind=mask_source.RAW_SUBJECT_MASK_SOURCE_KIND,
        run_path="subject_mask_runs/canonical",
        row_identity=_identity(keys, "subject_mask_runs/canonical"),
        instance_key=keys,
        source_acquisition_frame_index=np.array([9, 10], dtype=np.int64),
        source_row_index=np.array([0, 1], dtype=np.int64),
        source_camera_frame=_frame(),
        labels=("subject_body", "eye_left", "eye_right", "swim_bladder"),
        role_mapping={
            "subject_body": "subject_body",
            "eye_left": "eye_left",
            "eye_right": "eye_right",
            "swim_bladder": "swim_bladder",
        },
        source_binding_record=binding,
        source_binding_digest=source_binding_sha256(binding),
        expression_bindings=bindings,
        centroid_xy_source_camera=np.zeros((2, 4, 2), dtype=np.float32),
        centroid_valid=np.ones((2, 4), dtype=bool),
        available_channels=np.ones(4, dtype=bool),
        direct_consolidated_evidence={"status": "validated"},
        source_payload_digest="2" * 64,
        anatomy_profile_digest=anatomy_profile_sha256(profile),
        _analysis_zarr=Path("/unused"),
        _anatomy_profile_payload=profile.payload,
        _binding_id=binding_id,
        _required_role_ids=tuple(components),
        _seal=mask_source._BOUND_SOURCE_SEAL,
    )


def _software() -> dict[str, object]:
    return {"package": "palette", "commit": "3" * 40}


def test_prepare_detection_binds_no_default_policy(monkeypatch) -> None:
    source = _detection_source()
    monkeypatch.setattr(
        preparation,
        "require_bound_detection_position_source",
        lambda value: value,
    )

    prepared = preparation.prepare_subject_position_input(
        source,
        estimator_id=DETECTION_BBOX_CENTROID_ESTIMATOR_ID,
        software_record=_software(),
    )

    np.testing.assert_allclose(prepared.arrays["position_xy"], [[2, 4], [12, 24]])
    assert prepared.policy_record["policy_id"] == (
        SUBJECT_POSITION_CANARY_NO_DEFAULT_POLICY_ID
    )
    assert prepared.policy_record["default_estimator_id"] is None
    assert prepared.policy_record["selector_eligible"] is False
    assert prepared.anatomy_record["anatomy_profile_id"] is None
    assert (
        prepared.coordinate_record["coordinate_descriptor"]["reference_extent"]
        ["authority"]["selector"]
        == "record"
    )


def test_prepare_keypoint_triad_binds_exact_anatomy_recipe(monkeypatch) -> None:
    source = _keypoint_source()
    monkeypatch.setattr(
        preparation,
        "revalidate_bound_keypoint_position_source",
        lambda value: value,
    )

    prepared = preparation.prepare_subject_position_input(
        source,
        estimator_id=KEYPOINT_ANATOMICAL_TRIAD_MEAN_ESTIMATOR_ID,
        software_record=_software(),
    )

    np.testing.assert_allclose(prepared.arrays["position_xy"], [[1, 1], [4, 7]])
    assert prepared.anatomy_record["recipe_id"] == "head_triad_equal_mean"
    assert prepared.anatomy_record["source_modality"] == "keypoint"


@pytest.mark.parametrize(
    ("estimator_id", "expected", "recipe_id"),
    (
        (
            MASK_COMPONENT_ANATOMICAL_TRIAD_MEAN_ESTIMATOR_ID,
            np.array([[1, 1], [4, 7]], dtype=np.float32),
            "head_triad_equal_mean",
        ),
        (
            SUBJECT_BODY_MASK_CENTROID_ESTIMATOR_ID,
            np.array([[9, 12], [15, 18]], dtype=np.float32),
            "subject_body_centroid",
        ),
    ),
)
def test_prepare_mask_profiles_bind_distinct_recipes(
    monkeypatch,
    estimator_id: str,
    expected: np.ndarray,
    recipe_id: str,
) -> None:
    source = _mask_source()
    monkeypatch.setattr(
        mask_source.BoundSubjectMaskPositionSource,
        "revalidate",
        lambda self: self,
    )

    prepared = preparation.prepare_subject_position_input(
        source,
        estimator_id=estimator_id,
        software_record=_software(),
    )

    np.testing.assert_allclose(prepared.arrays["position_xy"], expected)
    assert prepared.anatomy_record["recipe_id"] == recipe_id


def test_prepare_refuses_cross_modality_and_unsealed_sources(monkeypatch) -> None:
    source = _detection_source()
    monkeypatch.setattr(
        preparation,
        "require_bound_detection_position_source",
        lambda value: value,
    )
    with pytest.raises(
        preparation.SubjectPositionPreparationError,
        match="modality",
    ):
        preparation.prepare_subject_position_input(
            source,
            estimator_id=KEYPOINT_ANATOMICAL_TRIAD_MEAN_ESTIMATOR_ID,
            software_record=_software(),
        )
    with pytest.raises(
        preparation.SubjectPositionPreparationError,
        match="sealed",
    ):
        preparation.prepare_subject_position_input(
            SimpleNamespace(),
            estimator_id=DETECTION_BBOX_CENTROID_ESTIMATOR_ID,
            software_record=_software(),
        )
