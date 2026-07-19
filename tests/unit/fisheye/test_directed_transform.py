from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from fisheye.shared.directed_transform import (
    DIRECTED_TRANSFORM_DIRECTION,
    DIRECTED_TRANSFORM_KIND,
    DIRECTED_TRANSFORM_SCHEMA_ID,
    DirectedTransformError,
    TransformReferenceExtent,
    apply_directed_homography,
    build_directed_homography,
    canonical_directed_transform_json,
    directed_transform_digest,
    homography_matrix_sha256,
    invert_directed_homography,
    parse_directed_homography,
    serialize_directed_homography,
    validate_homography_matrix,
)


CAMERA_EXTENT = TransformReferenceExtent(
    width=4512,
    height=4512,
    units="px",
    authority="/raw_video/images_full.shape[-2:]",
)
CANVAS_EXTENT = TransformReferenceExtent(
    width=358,
    height=358,
    units="px",
    authority="/analysis/stimulus_runs/stim_1/calibration/arena_geometry",
)
NON_SELF_INVERSE = np.asarray(
    [
        [2.0, 0.25, 10.0],
        [0.1, 3.0, -5.0],
        [0.001, 0.002, 1.0],
    ],
    dtype=np.float64,
)


def _camera_to_canvas():
    return build_directed_homography(
        transform_id="camera_2010093_to_stimulus_canvas_v1",
        matrix=NON_SELF_INVERSE,
        from_space_id="source_camera_image_px",
        to_space_id="stimulus_canvas_px",
        source_reference_extent=CAMERA_EXTENT,
        target_reference_extent=CANVAS_EXTENT,
        calibration_ref="/analysis/calibration",
        camera_id="2010093",
    )


def _apply_camera_to_canvas(points: np.ndarray, transform=None) -> np.ndarray:
    return apply_directed_homography(
        points,
        NON_SELF_INVERSE,
        transform or _camera_to_canvas(),
        from_space_id="source_camera_image_px",
        to_space_id="stimulus_canvas_px",
        source_reference_extent=CAMERA_EXTENT,
        target_reference_extent=CANVAS_EXTENT,
    )


def test_round_trip_serialization_and_digests_are_canonical() -> None:
    transform = _camera_to_canvas()
    payload = transform.to_dict()

    assert payload["schema_id"] == DIRECTED_TRANSFORM_SCHEMA_ID
    assert payload["kind"] == DIRECTED_TRANSFORM_KIND
    assert payload["direction"] == DIRECTED_TRANSFORM_DIRECTION
    assert payload["matrix_sha256"] == homography_matrix_sha256(NON_SELF_INVERSE)
    assert parse_directed_homography(payload) == transform
    assert parse_directed_homography(json.dumps(payload)) == transform
    assert parse_directed_homography(json.dumps(payload).encode("utf-8")) == transform
    assert serialize_directed_homography(transform) == payload

    reordered = {key: payload[key] for key in reversed(tuple(payload))}
    assert directed_transform_digest(reordered) == transform.digest()
    assert canonical_directed_transform_json(reordered) == transform.canonical_json()


def test_transform_metadata_keeps_coordinate_and_calibration_records_separate() -> None:
    payload = _camera_to_canvas().to_dict()

    assert payload["calibration_ref"] == "/analysis/calibration"
    assert "coordinate_descriptor" not in payload
    assert "lineage_refs" not in payload
    assert "matrix" not in payload


@pytest.mark.parametrize("field", ["direction", "from_space_id", "calibration_ref"])
def test_parser_rejects_missing_required_metadata(field: str) -> None:
    payload = _camera_to_canvas().to_dict()
    del payload[field]

    with pytest.raises(DirectedTransformError, match="missing"):
        parse_directed_homography(payload)


def test_parser_and_apply_reject_wrong_direction() -> None:
    payload = _camera_to_canvas().to_dict()
    payload["direction"] = "target_to_source"

    with pytest.raises(DirectedTransformError, match="source_to_target"):
        parse_directed_homography(payload)

    with pytest.raises(DirectedTransformError, match="direction mismatch"):
        apply_directed_homography(
            np.asarray([[10.0, 20.0]]),
            NON_SELF_INVERSE,
            _camera_to_canvas(),
            from_space_id="stimulus_canvas_px",
            to_space_id="source_camera_image_px",
            source_reference_extent=CANVAS_EXTENT,
            target_reference_extent=CAMERA_EXTENT,
        )


def test_parser_requires_camera_id_for_camera_bound_transform() -> None:
    payload = _camera_to_canvas().to_dict()
    del payload["camera_id"]

    with pytest.raises(DirectedTransformError, match="camera_id is required"):
        parse_directed_homography(payload)


def test_camera_id_remains_optional_without_camera_bound_side() -> None:
    transform = build_directed_homography(
        transform_id="texture_to_canvas_v1",
        matrix=NON_SELF_INVERSE,
        from_space_id="stimulus_texture_px",
        to_space_id="stimulus_canvas_px",
        source_reference_extent=CANVAS_EXTENT,
        target_reference_extent=CANVAS_EXTENT,
        calibration_ref="/analysis/calibration",
    )

    assert transform.camera_id is None
    assert "camera_id" not in transform.to_dict()


def test_apply_uses_non_self_inverse_matrix_in_declared_direction() -> None:
    points = np.asarray([[100.0, 200.0], [1.5, 3.25]], dtype=np.float64)
    homogeneous = np.column_stack((points, np.ones(points.shape[0])))
    projected = (NON_SELF_INVERSE @ homogeneous.T).T
    expected = projected[:, :2] / projected[:, 2:3]

    actual = _apply_camera_to_canvas(points)

    np.testing.assert_allclose(actual, expected)
    assert not np.allclose(NON_SELF_INVERSE, np.linalg.inv(NON_SELF_INVERSE))


@pytest.mark.parametrize("side", ["source", "target"])
def test_apply_rejects_reference_extent_mismatch(side: str) -> None:
    source_extent = CAMERA_EXTENT
    target_extent = CANVAS_EXTENT
    if side == "source":
        source_extent = TransformReferenceExtent(
            width=640,
            height=640,
            units="px",
            authority=CAMERA_EXTENT.authority,
        )
    else:
        target_extent = TransformReferenceExtent(
            width=344,
            height=344,
            units="px",
            authority=CANVAS_EXTENT.authority,
        )

    with pytest.raises(DirectedTransformError, match="reference extent mismatch"):
        apply_directed_homography(
            np.asarray([[10.0, 20.0]]),
            NON_SELF_INVERSE,
            _camera_to_canvas(),
            from_space_id="source_camera_image_px",
            to_space_id="stimulus_canvas_px",
            source_reference_extent=source_extent,
            target_reference_extent=target_extent,
        )


def test_apply_rejects_reference_authority_mismatch() -> None:
    mismatched = TransformReferenceExtent(
        width=CAMERA_EXTENT.width,
        height=CAMERA_EXTENT.height,
        units=CAMERA_EXTENT.units,
        authority="/wrong/source.shape[-2:]",
    )

    with pytest.raises(DirectedTransformError, match="Source reference extent mismatch"):
        apply_directed_homography(
            np.asarray([[10.0, 20.0]]),
            NON_SELF_INVERSE,
            _camera_to_canvas(),
            from_space_id="source_camera_image_px",
            to_space_id="stimulus_canvas_px",
            source_reference_extent=mismatched,
            target_reference_extent=CANVAS_EXTENT,
        )


def test_apply_rejects_matrix_digest_mismatch() -> None:
    different = NON_SELF_INVERSE.copy()
    different[0, 2] += 1.0

    with pytest.raises(DirectedTransformError, match="digest"):
        apply_directed_homography(
            np.asarray([[10.0, 20.0]]),
            different,
            _camera_to_canvas(),
            from_space_id="source_camera_image_px",
            to_space_id="stimulus_canvas_px",
            source_reference_extent=CAMERA_EXTENT,
            target_reference_extent=CANVAS_EXTENT,
        )


def test_matrix_validator_rejects_wrong_shape_singular_and_nonfinite() -> None:
    with pytest.raises(DirectedTransformError, match="shape"):
        validate_homography_matrix(np.eye(2))
    with pytest.raises(DirectedTransformError, match="nonsingular"):
        validate_homography_matrix(
            np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        )
    nonfinite = np.eye(3)
    nonfinite[0, 0] = np.nan
    with pytest.raises(DirectedTransformError, match="finite"):
        validate_homography_matrix(nonfinite)


def test_apply_rejects_near_zero_homogeneous_w() -> None:
    matrix = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, -1.0]],
        dtype=np.float64,
    )
    transform = build_directed_homography(
        transform_id="near_zero_w_fixture",
        matrix=matrix,
        from_space_id="source_camera_image_px",
        to_space_id="stimulus_canvas_px",
        source_reference_extent=CAMERA_EXTENT,
        target_reference_extent=CANVAS_EXTENT,
        calibration_ref="/analysis/calibration",
        camera_id="2010093",
    )

    with pytest.raises(DirectedTransformError, match="near-zero homogeneous w"):
        apply_directed_homography(
            np.asarray([[1.0, 2.0]]),
            matrix,
            transform,
            from_space_id="source_camera_image_px",
            to_space_id="stimulus_canvas_px",
            source_reference_extent=CAMERA_EXTENT,
            target_reference_extent=CANVAS_EXTENT,
        )


def test_explicit_inverse_swaps_spaces_extents_and_records_source() -> None:
    forward = _camera_to_canvas()
    inverse, inverse_matrix = invert_directed_homography(
        forward,
        NON_SELF_INVERSE,
        transform_id="stimulus_canvas_to_camera_2010093_v1",
    )

    assert inverse.from_space_id == forward.to_space_id
    assert inverse.to_space_id == forward.from_space_id
    assert inverse.source_reference_extent == forward.target_reference_extent
    assert inverse.target_reference_extent == forward.source_reference_extent
    assert inverse.source_transform is not None
    assert inverse.source_transform.relationship == "inverse_of"
    assert inverse.source_transform.transform_id == forward.transform_id
    assert inverse.source_transform.transform_sha256 == forward.digest()
    assert inverse.matrix_sha256 == homography_matrix_sha256(inverse_matrix)

    points = np.asarray([[100.0, 200.0], [1.5, 3.25]], dtype=np.float64)
    canvas = _apply_camera_to_canvas(points, forward)
    restored = apply_directed_homography(
        canvas,
        inverse_matrix,
        inverse,
        from_space_id="stimulus_canvas_px",
        to_space_id="source_camera_image_px",
        source_reference_extent=CANVAS_EXTENT,
        target_reference_extent=CAMERA_EXTENT,
    )
    np.testing.assert_allclose(restored, points, rtol=1e-12, atol=1e-12)


def test_parser_rejects_unknown_fields_and_unsupported_spaces() -> None:
    payload = copy.deepcopy(_camera_to_canvas().to_dict())
    payload["unexpected"] = True
    with pytest.raises(DirectedTransformError, match="unknown"):
        parse_directed_homography(payload)

    payload = _camera_to_canvas().to_dict()
    payload["from_space_id"] = "presentation_viewport_px"
    with pytest.raises(DirectedTransformError, match="unsupported coordinate space"):
        parse_directed_homography(payload)
