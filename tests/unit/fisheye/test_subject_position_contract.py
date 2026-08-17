from pathlib import Path

import numpy as np
import pytest

from fisheye.shared.anatomy_profile import AnatomyProfile, AnatomyProfileError
from fisheye.shared.coordinate_descriptor import (
    PIXEL_FRAME_AUTHORITY_RECORD_KIND,
    CanonicalFrameRecord,
    DigestBoundCoordinateRecordRef,
    build_canonical_coordinate_descriptor,
)
from fisheye.shared.coordinate_identity import (
    OBSERVATION_INSTANCE_DOMAIN,
    build_row_identity_contract,
)
from fisheye.shared.coordinate_surface_contract import SOURCE_CAMERA_POINT_XY
from fisheye.shared.subject_position_contract import (
    require_estimator_anatomy_expression,
    resolve_anatomy_point_expression,
)
from fisheye.shared.subject_position_expression import (
    KEYPOINT_ANATOMICAL_TRIAD_MEAN_PROFILE,
    MASK_COMPONENT_ANATOMICAL_TRIAD_MEAN_PROFILE,
    PointArrayBinding,
    evaluate_estimator_profile,
    point_expression_digest,
)
from fisheye.shared.subject_position_storage import (
    canonical_observation_position_logical_metadata,
    canonical_source_camera_coordinate_metadata,
    validate_observation_position_arrays,
)


_PROFILE_PATH = (
    Path(__file__).parents[3]
    / "configs/fisheye/anatomy_profiles/zebrafish_larva_v1.json"
)


def _profile() -> AnatomyProfile:
    return AnatomyProfile.from_json(_PROFILE_PATH)


@pytest.mark.parametrize(
    ("binding_id", "profile", "leaf_op"),
    [
        (
            "zebrafish_larva_keypoint_traditional_v3_v1",
            KEYPOINT_ANATOMICAL_TRIAD_MEAN_PROFILE,
            "keypoint",
        ),
        (
            "zebrafish_larva_subject_mask_lr_v1",
            MASK_COMPONENT_ANATOMICAL_TRIAD_MEAN_PROFILE,
            "component_centroid",
        ),
    ],
)
def test_initial_estimators_bind_exact_shared_anatomy_recipe(
    binding_id, profile, leaf_op
) -> None:
    resolved = require_estimator_anatomy_expression(
        profile,
        _profile(),
        binding_id=binding_id,
        recipe_id="head_triad_equal_mean",
    )

    assert {point["op"] for point in resolved.expression["points"]} == {leaf_op}
    assert resolved.record["expression_sha256"] == point_expression_digest(
        resolved.expression
    )
    assert len(resolved.digest) == 64


def test_resolved_recipe_binds_exact_source_schema_authority() -> None:
    keypoint = resolve_anatomy_point_expression(
        _profile(),
        binding_id="zebrafish_larva_keypoint_traditional_v3_v1",
        recipe_id="head_triad_equal_mean",
    )
    masks = resolve_anatomy_point_expression(
        _profile(),
        binding_id="zebrafish_larva_subject_mask_lr_v1",
        recipe_id="head_triad_equal_mean",
    )

    assert keypoint.record["anatomy_profile_id"] == "zebrafish_larva_anatomy.v1"
    assert keypoint.record["source_schema_id"] == "pose_skel_traditional_v3"
    assert masks.record["source_schema_id"] == "subject_v1_lr"
    assert len(keypoint.record["source_schema_sha256"]) == 64
    assert len(masks.record["source_schema_sha256"]) == 64

    detached = keypoint.record
    detached["source_schema_id"] = "tampered"
    assert keypoint.record["source_schema_id"] == "pose_skel_traditional_v3"
    with pytest.raises(TypeError):
        keypoint._record_json[0] = 0


def test_axis_recipe_is_not_silently_lowered_as_a_position() -> None:
    with pytest.raises(AnatomyProfileError, match="not a point recipe"):
        resolve_anatomy_point_expression(
            _profile(),
            binding_id="zebrafish_larva_keypoint_traditional_v3_v1",
            recipe_id="anterior_axis",
        )


def test_estimator_expression_mismatch_fails_closed() -> None:
    with pytest.raises(ValueError, match="does not match its v1 record"):
        require_estimator_anatomy_expression(
            {**KEYPOINT_ANATOMICAL_TRIAD_MEAN_PROFILE, "fallback": "detection"},
            _profile(),
            binding_id="zebrafish_larva_keypoint_traditional_v3_v1",
            recipe_id="head_triad_equal_mean",
        )


def test_evaluator_result_satisfies_materialized_storage_contract() -> None:
    instance_key = np.asarray([101, 102], dtype=np.uint64)
    result = evaluate_estimator_profile(
        KEYPOINT_ANATOMICAL_TRIAD_MEAN_PROFILE,
        {
            "keypoints": {
                "swim_bladder": PointArrayBinding(
                    [[2.0, 4.0], [3.0, 5.0]], valid=[True, True]
                ),
                "eye_left": PointArrayBinding(
                    [[4.0, 4.0], [5.0, 5.0]], valid=[True, True]
                ),
                "eye_right": PointArrayBinding(
                    [[6.0, 4.0], [7.0, 5.0]], valid=[True, False]
                ),
            },
            "components": {},
            "bboxes": {},
        },
    )
    row_identity = build_row_identity_contract(
        domain=OBSERVATION_INSTANCE_DOMAIN,
        values=instance_key,
    )
    frame_authority = DigestBoundCoordinateRecordRef(
        record_ref="/coordinate_frames/source_camera@pixel_frame_authority",
        record_sha256="1" * 64,
    )
    descriptor = build_canonical_coordinate_descriptor(
        **SOURCE_CAMERA_POINT_XY.descriptor_kwargs(),
        reference_width=4512,
        reference_height=4512,
        reference_authority=frame_authority,
        reference_selector="record",
        row_identity_contract=row_identity,
        row_identity_record_ref=(
            "/analysis/keypoint_runs/k1@row_identity_contract"
        ),
        frame_record=CanonicalFrameRecord(
            kind=PIXEL_FRAME_AUTHORITY_RECORD_KIND,
            record_ref=frame_authority.record_ref,
            record_sha256=frame_authority.record_sha256,
        ),
    )
    coordinate = canonical_source_camera_coordinate_metadata(descriptor)
    arrays = {
        "position_xy": result.position_xy,
        "valid": result.valid,
        "failure_reason_codes": result.failure_reason_codes,
        "instance_key": instance_key,
        "source_acquisition_frame_index": np.asarray([10, 11], dtype=np.int64),
        "source_row_index": np.asarray([0, 1], dtype=np.int64),
        "support/source_points_xy": result.source_points_xy,
        "support/source_points_valid": result.source_points_valid,
        "support/source_point_reason_codes": result.source_point_reason_codes,
    }

    report = validate_observation_position_arrays(
        arrays,
        coordinate_metadata=coordinate,
        manifest_metadata=canonical_observation_position_logical_metadata(
            coordinate
        ),
    )

    assert report.row_count == 2
    assert report.support_point_count == 3
    assert result.valid.tolist() == [True, False]
