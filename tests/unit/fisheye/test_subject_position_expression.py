from __future__ import annotations

import numpy as np
import pytest

from fisheye.shared.subject_position_expression import (
    DETECTION_BBOX_CENTROID_ESTIMATOR_ID,
    DETECTION_BBOX_CENTROID_PROFILE,
    ESTIMATOR_PROFILE_RECORDS,
    KEYPOINT_ANATOMICAL_TRIAD_MEAN_ESTIMATOR_ID,
    KEYPOINT_ANATOMICAL_TRIAD_MEAN_PROFILE,
    MASK_COMPONENT_ANATOMICAL_TRIAD_MEAN_PROFILE,
    SOURCE_CAMERA_BBOX_SURFACE_ID,
    SUBJECT_BODY_MASK_CENTROID_ESTIMATOR_ID,
    SUBJECT_BODY_MASK_CENTROID_PROFILE,
    BoundingBoxSourceBinding,
    ComponentSourceBinding,
    PointArrayBinding,
    PointExpressionBindings,
    canonicalize_estimator_profile,
    canonicalize_point_expression,
    estimator_profile_digest,
    evaluate_estimator_profile,
    evaluate_point_expression,
    get_estimator_profile,
    parse_point_expression_json,
    point_expression_envelope,
    point_expression_digest,
)
from fisheye.shared.subject_position_types import (
    CANONICAL_FLOAT32_QNAN_BITS,
    POSITION_FAILURE_REASON_CODES,
    POSITION_FAILURE_REASON_PRECEDENCE,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


def _point(op: str, name: str) -> dict[str, object]:
    return {"op": op, "role": name}


def _triad_expression(*roles: str) -> dict[str, object]:
    return {
        "op": "mean_points",
        "points": [_point("keypoint", role) for role in roles],
        "weighting": "equal_per_point",
    }


def test_mean_operands_are_canonicalized_order_independently() -> None:
    first = _triad_expression("eye_left", "eye_right", "swim_bladder")
    second = _triad_expression("swim_bladder", "eye_left", "eye_right")
    assert canonicalize_point_expression(first) == canonicalize_point_expression(second)
    assert point_expression_digest(first) == point_expression_digest(second)


def test_expression_digest_binds_schema_envelope() -> None:
    expression = _point("keypoint", "eye_left")
    envelope = point_expression_envelope(expression)
    assert envelope == {
        "schema_id": "palette.subject_position_point_expression",
        "schema_version": 1,
        "expression": expression,
    }
    assert point_expression_digest(expression) == canonical_json_sha256(envelope)
    assert point_expression_digest(expression) != canonical_json_sha256(expression)


def test_midpoint_has_deterministic_two_point_semantics() -> None:
    left = _point("keypoint", "eye_left")
    right = _point("keypoint", "eye_right")
    assert canonicalize_point_expression(
        {"op": "midpoint", "point_a": left, "point_b": right}
    ) == canonicalize_point_expression(
        {"op": "midpoint", "point_a": right, "point_b": left}
    )
    result = evaluate_point_expression(
        {"op": "midpoint", "point_a": left, "point_b": right},
        keypoints={
            "eye_left": PointArrayBinding([[0.0, 2.0]], valid=[True]),
            "eye_right": PointArrayBinding([[4.0, 6.0]], valid=[True]),
        },
    )
    np.testing.assert_array_equal(result.position_xy, [[2.0, 4.0]])
    assert result.source_points_xy.shape == (1, 2, 2)


@pytest.mark.parametrize(
    "expression",
    [
        {"op": "keypoint", "role": "eye_left", "extra": True},
        {"op": "keypoint"},
        {"op": "unknown", "role": "eye_left"},
        {"op": "mean_points", "points": [_point("keypoint", "a")]},
        {
            "op": "mean_points",
            "points": [_point("keypoint", "a"), _point("keypoint", "b")],
            "weighting": "area_weighted",
        },
        {
            "op": "mean_points",
            "points": [_point("keypoint", "a"), _point("keypoint", "a")],
            "weighting": "equal_per_point",
        },
        {
            "op": "midpoint",
            "point_a": _point("keypoint", "a"),
            "point_b": _point("keypoint", "a"),
        },
        {
            "op": "mean_points",
            "points": [_point("keypoint", "a"), _point("keypoint", "b")],
            "weighting": "equal_per_point",
            "weights": [0.5, 0.5],
        },
    ],
)
def test_expression_records_are_strict(expression: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        canonicalize_point_expression(expression)


def test_duplicate_json_fields_are_rejected() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        parse_point_expression_json('{"op":"keypoint","role":"a","role":"b"}')


def test_unknown_role_requires_an_explicit_binding() -> None:
    with pytest.raises(ValueError, match="explicit keypoint binding"):
        evaluate_point_expression(_point("keypoint", "not_bound"), keypoints={})


def test_keypoint_triad_requires_every_anchor_and_preserves_support() -> None:
    expression = _triad_expression("swim_bladder", "eye_left", "eye_right")
    sources = {
        "swim_bladder": PointArrayBinding([[1.0, 2.0]], valid=[True]),
        "eye_left": PointArrayBinding([[3.0, 4.0]], valid=[False]),
        "eye_right": PointArrayBinding([[5.0, 6.0]], valid=[True]),
    }
    result = evaluate_point_expression(expression, keypoints=sources)
    assert result.valid.tolist() == [False]
    assert result.failure_reason_codes.tolist() == [
        POSITION_FAILURE_REASON_CODES["required_anchor_invalid"]
    ]
    # Support evidence follows the canonical order: eye_left, eye_right,
    # swim_bladder.
    assert result.source_points_valid.tolist() == [[False, True, True]]
    assert (
        result.source_point_reason_codes[0, 0]
        == POSITION_FAILURE_REASON_CODES["required_anchor_invalid"]
    )
    np.testing.assert_array_equal(
        result.position_xy.view(np.uint32),
        np.full((1, 2), CANONICAL_FLOAT32_QNAN_BITS, dtype=np.uint32),
    )


def test_low_confidence_is_an_explicit_upstream_decision() -> None:
    result = evaluate_point_expression(
        _point("keypoint", "eye_left"),
        keypoints={
            "eye_left": PointArrayBinding(
                [[1.0, 2.0]],
                valid=[True],
                confidence=[0.1],
                confidence_valid=[False],
            )
        },
    )
    assert result.valid.tolist() == [False]
    assert result.failure_reason_codes.tolist() == [
        POSITION_FAILURE_REASON_CODES["required_anchor_low_confidence"]
    ]
    diagnostic_only = evaluate_point_expression(
        _point("keypoint", "eye_left"),
        keypoints={
            "eye_left": PointArrayBinding(
                [[1.0, 2.0]],
                valid=[True],
                confidence=[-100.0],
                confidence_valid=[True],
            )
        },
    )
    assert diagnostic_only.valid.tolist() == [True]
    np.testing.assert_array_equal(diagnostic_only.source_point_confidence, [[-100.0]])


def test_nonfinite_diagnostic_confidence_fails_closed() -> None:
    with pytest.raises(ValueError, match="confidence must be finite"):
        evaluate_point_expression(
            _point("keypoint", "eye_left"),
            keypoints={
                "eye_left": PointArrayBinding(
                    [[1.0, 2.0]],
                    valid=[True],
                    confidence=[np.nan],
                    confidence_valid=[True],
                )
            },
        )


def test_boolean_geometry_is_not_silently_cast_to_coordinates() -> None:
    with pytest.raises(ValueError, match="numeric array"):
        evaluate_point_expression(
            _point("keypoint", "eye_left"),
            keypoints={
                "eye_left": PointArrayBinding(
                    np.asarray([[True, False]], dtype=bool), valid=[True]
                )
            },
        )
    with pytest.raises(ValueError, match="numeric array"):
        evaluate_point_expression(
            {"op": "bbox_centroid", "array_ref": "boxes"},
            bboxes={
                "boxes": BoundingBoxSourceBinding(
                    np.asarray([[False, False, True, True]], dtype=bool),
                    valid=[True],
                )
            },
        )


def test_generic_keypoint_requires_explicit_upstream_validity() -> None:
    with pytest.raises(ValueError, match=r"keypoint\[eye_left\].*valid"):
        evaluate_point_expression(
            _point("keypoint", "eye_left"),
            keypoints={"eye_left": {"values": [[1.0, 2.0]]}},
        )
    with pytest.raises(ValueError, match="bare point arrays"):
        evaluate_point_expression(
            _point("keypoint", "eye_left"),
            keypoints={"eye_left": [[1.0, 2.0]]},
        )
    with pytest.raises(ValueError, match=r"valid.*bool\[1\]"):
        evaluate_point_expression(
            _point("keypoint", "eye_left"),
            keypoints={"eye_left": PointArrayBinding([[1.0, 2.0]], valid=[1])},
        )


def test_generic_bbox_requires_explicit_upstream_validity() -> None:
    with pytest.raises(ValueError, match=r"bbox\[boxes\].*valid"):
        evaluate_point_expression(
            {"op": "bbox_centroid", "array_ref": "boxes"},
            bboxes={"boxes": {"xyxy": [[0.0, 0.0, 2.0, 2.0]]}},
        )
    with pytest.raises(ValueError, match="bare bbox arrays"):
        evaluate_point_expression(
            {"op": "bbox_centroid", "array_ref": "boxes"},
            bboxes={"boxes": [[0.0, 0.0, 2.0, 2.0]]},
        )
    with pytest.raises(ValueError, match=r"valid.*bool\[1\]"):
        evaluate_point_expression(
            {"op": "bbox_centroid", "array_ref": "boxes"},
            bboxes={
                "boxes": BoundingBoxSourceBinding([[0.0, 0.0, 2.0, 2.0]], valid=[1])
            },
        )


def test_support_confidence_is_all_participating_leaves_or_absent() -> None:
    expression = _triad_expression("a", "b")
    partial = evaluate_point_expression(
        expression,
        keypoints={
            "a": PointArrayBinding([[1.0, 2.0]], valid=[True], confidence=[0.9]),
            "b": PointArrayBinding([[3.0, 4.0]], valid=[True]),
        },
    )
    assert partial.source_point_confidence is None
    complete = evaluate_point_expression(
        expression,
        keypoints={
            "a": PointArrayBinding([[1.0, 2.0]], valid=[True], confidence=[0.9]),
            "b": PointArrayBinding([[3.0, 4.0]], valid=[True], confidence=[0.8]),
        },
    )
    assert complete.source_point_confidence is not None
    assert complete.source_point_confidence.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(complete.source_point_confidence, [[0.9, 0.8]])


def test_nonfinite_anchor_takes_precedence_over_authority_failure() -> None:
    result = evaluate_point_expression(
        _point("keypoint", "eye_left"),
        keypoints={"eye_left": PointArrayBinding([[np.nan, 2.0]], valid=[False])},
    )
    assert (
        result.failure_reason_codes[0]
        == POSITION_FAILURE_REASON_CODES["nonfinite_source_geometry"]
    )


def test_combined_reason_selection_uses_shared_precedence() -> None:
    result = evaluate_point_expression(
        {
            "op": "mean_points",
            "points": [
                _point("keypoint", "eye_left"),
                {"op": "bbox_centroid", "array_ref": "boxes"},
            ],
            "weighting": "equal_per_point",
        },
        keypoints={"eye_left": PointArrayBinding([[np.nan, 2.0]], valid=[True])},
        bboxes={
            "boxes": BoundingBoxSourceBinding([[0.0, 0.0, 0.0, 2.0]], valid=[True])
        },
    )
    assert POSITION_FAILURE_REASON_PRECEDENCE[:2] == (
        "nonfinite_source_geometry",
        "degenerate_source_geometry",
    )
    assert result.failure_reason_codes.tolist() == [
        POSITION_FAILURE_REASON_CODES["nonfinite_source_geometry"]
    ]


def test_bbox_centroid_uses_source_camera_half_open_xyxy_surface() -> None:
    assert SOURCE_CAMERA_BBOX_SURFACE_ID == "source_camera_bbox_xyxy_v1"
    result = evaluate_point_expression(
        {"op": "bbox_centroid", "array_ref": "bbox_img_xyxy"},
        bboxes={
            "bbox_img_xyxy": BoundingBoxSourceBinding(
                [[2.0, 4.0, 8.0, 10.0]], valid=[True]
            )
        },
    )
    np.testing.assert_array_equal(result.position_xy, [[5.0, 7.0]])
    assert result.valid.tolist() == [True]


@pytest.mark.parametrize(
    "box,reason",
    [
        ([[1.0, 2.0, 1.0, 5.0]], "degenerate_source_geometry"),
        ([[1.0, 2.0, 5.0, 2.0]], "degenerate_source_geometry"),
        ([[np.nan, 2.0, 5.0, 6.0]], "nonfinite_source_geometry"),
        ([[1.0, 2.0, np.inf, 6.0]], "nonfinite_source_geometry"),
    ],
)
def test_bbox_centroid_rejects_degenerate_and_nonfinite_boxes(
    box: list[list[float]], reason: str
) -> None:
    result = evaluate_point_expression(
        {"op": "bbox_centroid", "array_ref": "boxes"},
        bboxes={"boxes": BoundingBoxSourceBinding(box, valid=[True])},
    )
    assert result.valid.tolist() == [False]
    assert result.failure_reason_codes.tolist() == [
        POSITION_FAILURE_REASON_CODES[reason]
    ]


def test_bbox_source_rejection_is_distinct_from_bad_geometry() -> None:
    result = evaluate_point_expression(
        {"op": "bbox_centroid", "array_ref": "boxes"},
        bboxes={
            "boxes": BoundingBoxSourceBinding([[0.0, 0.0, 2.0, 2.0]], valid=[False])
        },
    )
    assert result.failure_reason_codes.tolist() == [
        POSITION_FAILURE_REASON_CODES["source_observation_rejected"]
    ]


def test_component_centroid_empty_and_nonfinite_rows_are_explicit() -> None:
    result = evaluate_point_expression(
        _point("component_centroid", "eye_left"),
        components={
            "eye_left": ComponentSourceBinding(
                centroids=[[0.0, 0.0], [2.0, 1.0], [np.nan, 0.0]],
                valid=[False, True, True],
            )
        },
    )
    assert result.valid.tolist() == [False, True, False]
    assert result.failure_reason_codes.tolist() == [
        POSITION_FAILURE_REASON_CODES["empty_mask_component"],
        0,
        POSITION_FAILURE_REASON_CODES["nonfinite_source_geometry"],
    ]
    np.testing.assert_array_equal(result.position_xy[1], [2.0, 1.0])
    np.testing.assert_array_equal(
        result.position_xy[[0, 2]].view(np.uint32),
        np.full((2, 2), CANONICAL_FLOAT32_QNAN_BITS, dtype=np.uint32),
    )


@pytest.mark.parametrize(
    "component_source",
    [
        np.zeros((1, 4, 4), dtype=bool),
        np.full((1, 4, 4), 0.75, dtype=np.float32),
        {"masks": np.zeros((1, 4, 4), dtype=bool)},
        {"masks": np.full((1, 4, 4), 0.75, dtype=np.float32)},
    ],
)
def test_component_centroid_rejects_bare_and_probability_masks(
    component_source: object,
) -> None:
    with pytest.raises(ValueError, match="masks|centroids"):
        evaluate_point_expression(
            _point("component_centroid", "eye_left"),
            components={"eye_left": component_source},
        )


def test_component_centroid_requires_upstream_centroid_validity() -> None:
    with pytest.raises(ValueError, match="validity"):
        evaluate_point_expression(
            _point("component_centroid", "eye_left"),
            components={"eye_left": {"centroids": [[1.0, 2.0]]}},
        )


def test_equal_component_mean_is_not_pixel_union_centroid() -> None:
    left_mask = np.zeros((6, 8), dtype=bool)
    left_mask[0, 0] = True
    right_mask = np.zeros((6, 8), dtype=bool)
    right_mask[2, 4] = True
    right_mask[2, 5] = True
    right_mask[3, 5] = True
    assert not np.any(left_mask & right_mask)
    assert np.count_nonzero(left_mask) != np.count_nonzero(right_mask)

    def mask_pixel_centroid(mask: np.ndarray) -> np.ndarray:
        y_indices, x_indices = np.nonzero(mask)
        assert x_indices.size > 0
        return np.asarray(
            [
                np.mean(x_indices.astype(np.float64)),
                np.mean(y_indices.astype(np.float64)),
            ],
            dtype=np.float64,
        )

    left_centroid = mask_pixel_centroid(left_mask)
    right_centroid = mask_pixel_centroid(right_mask)
    union_centroid = mask_pixel_centroid(left_mask | right_mask)
    component_centroids = {
        "left": ComponentSourceBinding([left_centroid], valid=[True]),
        "right": ComponentSourceBinding([right_centroid], valid=[True]),
    }
    expression = {
        "op": "mean_points",
        "points": [
            {"op": "component_centroid", "role": "left"},
            {"op": "component_centroid", "role": "right"},
        ],
        "weighting": "equal_per_point",
    }
    result = evaluate_point_expression(expression, components=component_centroids)
    equal_component_mean = (left_centroid + right_centroid) / 2.0
    np.testing.assert_allclose(result.position_xy, [equal_component_mean])
    union_result = evaluate_point_expression(
        {"op": "component_centroid", "role": "union"},
        components={"union": ComponentSourceBinding([union_centroid], valid=[True])},
    )
    np.testing.assert_allclose(union_result.position_xy, [union_centroid])
    assert not np.allclose(result.position_xy, union_result.position_xy)


def test_float32_is_the_published_output_after_float64_evaluation() -> None:
    result = evaluate_point_expression(
        {
            "op": "mean_points",
            "points": [
                {"op": "keypoint", "role": "a"},
                {"op": "keypoint", "role": "b"},
                {"op": "keypoint", "role": "c"},
            ],
            "weighting": "equal_per_point",
        },
        keypoints={
            "a": PointArrayBinding([[16777217.0, 0.1]], valid=[True]),
            "b": PointArrayBinding([[16777219.0, 0.2]], valid=[True]),
            "c": PointArrayBinding([[16777221.0, 0.3]], valid=[True]),
        },
    )
    assert result.position_xy.dtype == np.dtype(np.float32)
    expected = np.asarray(
        [[np.float64(16777217.0 + 16777219.0 + 16777221.0) / 3.0, 0.2]],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(result.position_xy, expected)


def test_complete_empty_detection_rowset_is_supported() -> None:
    result = evaluate_point_expression(
        {"op": "bbox_centroid", "array_ref": "boxes"},
        bboxes={
            "boxes": BoundingBoxSourceBinding(
                np.empty((0, 4), dtype=np.float64),
                valid=np.empty((0,), dtype=bool),
            )
        },
    )
    assert result.position_xy.shape == (0, 2)
    assert result.position_xy.dtype == np.dtype(np.float32)
    assert result.valid.shape == (0,)
    assert result.source_points_xy.shape == (0, 1, 2)


def test_row_aligned_sources_must_have_identical_coverage() -> None:
    with pytest.raises(ValueError, match="Row-aligned source mismatch"):
        evaluate_point_expression(
            _triad_expression("a", "b"),
            keypoints={
                "a": PointArrayBinding([[1.0, 2.0]], valid=[True]),
                "b": PointArrayBinding([[1.0, 2.0], [2.0, 3.0]], valid=[True, True]),
            },
        )


@pytest.mark.parametrize(
    "profile",
    [
        DETECTION_BBOX_CENTROID_PROFILE,
        KEYPOINT_ANATOMICAL_TRIAD_MEAN_PROFILE,
        MASK_COMPONENT_ANATOMICAL_TRIAD_MEAN_PROFILE,
        SUBJECT_BODY_MASK_CENTROID_PROFILE,
    ],
)
def test_four_estimator_profiles_are_exact_and_have_no_default(
    profile: dict[str, object],
) -> None:
    canonical = canonicalize_estimator_profile(profile)
    assert canonical["estimator_id"] in ESTIMATOR_PROFILE_RECORDS
    assert estimator_profile_digest(profile) == estimator_profile_digest(canonical)
    assert canonical["fallback"] == "none"
    validity_policy = canonical["validity_policy"]
    assert validity_policy["confidence_policy"] == "upstream_confidence_valid_only"
    assert (
        tuple(item["tag"] for item in validity_policy["primary_reason_precedence"])
        == POSITION_FAILURE_REASON_PRECEDENCE
    )


def test_estimator_profile_tampering_and_unknown_fields_fail_closed() -> None:
    tampered = dict(KEYPOINT_ANATOMICAL_TRIAD_MEAN_PROFILE)
    tampered["source_modality"] = "subject_mask"
    with pytest.raises(ValueError):
        canonicalize_estimator_profile(tampered)
    extra = dict(DETECTION_BBOX_CENTROID_PROFILE)
    extra["default"] = True
    with pytest.raises(ValueError):
        canonicalize_estimator_profile(extra)
    precedence_tampered = dict(DETECTION_BBOX_CENTROID_PROFILE)
    precedence_tampered["validity_policy"] = dict(
        DETECTION_BBOX_CENTROID_PROFILE["validity_policy"]
    )
    precedence_tampered["validity_policy"]["primary_reason_precedence"] = list(
        reversed(precedence_tampered["validity_policy"]["primary_reason_precedence"])
    )
    with pytest.raises(ValueError, match="precedence"):
        canonicalize_estimator_profile(precedence_tampered)


def test_profile_accessors_cannot_mutate_builtin_validation_authority() -> None:
    returned = get_estimator_profile(DETECTION_BBOX_CENTROID_ESTIMATOR_ID)
    returned["source_modality"] = "tampered"
    returned["validity_policy"]["primary_reason_precedence"].reverse()
    fresh = get_estimator_profile(DETECTION_BBOX_CENTROID_ESTIMATOR_ID)
    assert fresh["source_modality"] == "detection"
    assert (
        tuple(
            item["tag"]
            for item in fresh["validity_policy"]["primary_reason_precedence"]
        )
        == POSITION_FAILURE_REASON_PRECEDENCE
    )
    assert canonicalize_estimator_profile(fresh) == fresh
    with pytest.raises(TypeError, match="immutable"):
        DETECTION_BBOX_CENTROID_PROFILE["validity_policy"][
            "confidence_policy"
        ] = "tampered"


def test_profile_evaluation_does_not_fallback_across_modalities() -> None:
    with pytest.raises(ValueError, match="explicit keypoint binding"):
        evaluate_estimator_profile(
            KEYPOINT_ANATOMICAL_TRIAD_MEAN_ESTIMATOR_ID,
            PointExpressionBindings(components={"swim_bladder": [[1.0, 2.0]]}),
        )


def test_named_keypoint_profile_requires_explicit_valid_for_every_leaf() -> None:
    bindings = PointExpressionBindings(
        keypoints={
            "eye_left": PointArrayBinding([[1.0, 2.0]], valid=[True]),
            "eye_right": PointArrayBinding([[3.0, 4.0]], valid=[True]),
            "swim_bladder": {"values": [[5.0, 6.0]]},
        }
    )
    with pytest.raises(ValueError, match=r"swim_bladder.*valid is missing"):
        evaluate_estimator_profile(
            KEYPOINT_ANATOMICAL_TRIAD_MEAN_ESTIMATOR_ID,
            bindings,
        )


def test_named_keypoint_profile_accepts_explicit_upstream_validity() -> None:
    result = evaluate_estimator_profile(
        KEYPOINT_ANATOMICAL_TRIAD_MEAN_ESTIMATOR_ID,
        PointExpressionBindings(
            keypoints={
                "eye_left": PointArrayBinding([[1.0, 2.0]], valid=[True]),
                "eye_right": PointArrayBinding([[3.0, 4.0]], valid=[True]),
                "swim_bladder": PointArrayBinding([[5.0, 9.0]], valid=[True]),
            }
        ),
    )
    assert result.valid.tolist() == [True]
    np.testing.assert_allclose(result.position_xy, [[3.0, 5.0]])


def test_named_detection_profile_requires_explicit_upstream_validity() -> None:
    with pytest.raises(ValueError, match=r"bbox_img_xyxy.*valid is missing"):
        evaluate_estimator_profile(
            DETECTION_BBOX_CENTROID_ESTIMATOR_ID,
            PointExpressionBindings(
                bboxes={"bbox_img_xyxy": {"xyxy": [[2.0, 4.0, 8.0, 10.0]]}}
            ),
        )


def test_named_detection_profile_accepts_explicit_upstream_validity() -> None:
    result = evaluate_estimator_profile(
        DETECTION_BBOX_CENTROID_ESTIMATOR_ID,
        PointExpressionBindings(
            bboxes={
                "bbox_img_xyxy": BoundingBoxSourceBinding(
                    [[2.0, 4.0, 8.0, 10.0]], valid=[True]
                )
            }
        ),
    )
    assert result.valid.tolist() == [True]
    np.testing.assert_array_equal(result.position_xy, [[5.0, 7.0]])


def test_profile_ids_are_explicit_and_detection_is_not_an_implicit_default() -> None:
    assert DETECTION_BBOX_CENTROID_ESTIMATOR_ID in ESTIMATOR_PROFILE_RECORDS
    assert SUBJECT_BODY_MASK_CENTROID_ESTIMATOR_ID in ESTIMATOR_PROFILE_RECORDS
    assert "default_estimator_id" not in DETECTION_BBOX_CENTROID_PROFILE
