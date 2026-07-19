from __future__ import annotations

import copy
import inspect
import json

import numpy as np
import pytest

from fisheye.shared import directed_transform as directed_transform_mod

from fisheye.shared.directed_transform import (
    DIRECTED_TRANSFORM_ATTR,
    DIRECTED_TRANSFORM_DIGEST_SUFFIX,
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
    load_bound_directed_homography,
    parse_directed_homography,
    serialize_directed_homography,
    stamp_directed_homography,
    validate_homography_matrix,
)


CAMERA_EXTENT = TransformReferenceExtent(
    width=4512,
    height=4512,
    units="px",
    authority="/raw_video/images_full.shape[-2:]",
)


def test_historical_v1_writer_helpers_are_not_public_exports() -> None:
    assert {
        "directed_homography_attrs",
        "stamp_directed_homography",
        "build_directed_homography",
        "invert_directed_homography",
    }.isdisjoint(directed_transform_mod.__all__)


CANVAS_EXTENT = TransformReferenceExtent(
    width=1920,
    height=1080,
    units="px",
    authority=(
        "analysis/stimulus_runs/stim_1/display_snapshot"
        "@selected_output_geometry"
    ),
)
CAMERA_CALIBRATION_PATH = "analysis/stimulus_runs/stim_1/calibration/2010093"
HOMOGRAPHY_ARRAY_PATH = f"{CAMERA_CALIBRATION_PATH}/homography_matrix"
NON_SELF_INVERSE = np.asarray(
    [
        [2.0, 0.25, 10.0],
        [0.1, 3.0, -5.0],
        [0.001, 0.002, 1.0],
    ],
    dtype=np.float64,
)


class FakeArray:
    def __init__(self, data: np.ndarray, *, path: str = HOMOGRAPHY_ARRAY_PATH) -> None:
        self.data = np.asarray(data, dtype=np.float64).copy()
        self.path = path
        self.attrs: dict[str, object] = {}

    def __getitem__(self, key):
        return self.data[key]


def _camera_to_canvas():
    return build_directed_homography(
        transform_id="camera_2010093_to_stimulus_canvas_v1",
        matrix=NON_SELF_INVERSE,
        from_space_id="source_camera_image_px",
        to_space_id="stimulus_canvas_px",
        source_reference_extent=CAMERA_EXTENT,
        target_reference_extent=CANVAS_EXTENT,
        calibration_ref=CAMERA_CALIBRATION_PATH,
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


@pytest.mark.parametrize(
    ("needle", "replacement"),
    [
        (
            '"direction":"source_to_target"',
            '"direction":"target_to_source","direction":"source_to_target"',
        ),
        (
            '"from_space_id":"source_camera_image_px"',
            '"from_space_id":"stimulus_canvas_px",'
            '"from_space_id":"source_camera_image_px"',
        ),
        (
            '"source_reference_extent":{"width":4512',
            '"source_reference_extent":{"width":640,"width":4512',
        ),
        (
            '"target_reference_extent":{"width":1920,"height":1080',
            '"target_reference_extent":{"width":1920,"height":720,'
            '"height":1080',
        ),
    ],
)
@pytest.mark.parametrize("as_bytes", [False, True])
def test_parser_rejects_duplicate_json_direction_and_extent_fields_recursively(
    needle: str,
    replacement: str,
    as_bytes: bool,
) -> None:
    raw = json.dumps(_camera_to_canvas().to_dict(), separators=(",", ":"))
    assert needle in raw
    raw = raw.replace(needle, replacement, 1)
    value: str | bytes = raw.encode("utf-8") if as_bytes else raw

    with pytest.raises(DirectedTransformError, match="duplicate key"):
        parse_directed_homography(value)


def test_transform_metadata_keeps_coordinate_and_calibration_records_separate() -> None:
    payload = _camera_to_canvas().to_dict()

    assert payload["calibration_ref"] == CAMERA_CALIBRATION_PATH
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
        calibration_ref=CAMERA_CALIBRATION_PATH,
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
        calibration_ref=CAMERA_CALIBRATION_PATH,
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


def _load_bound(node: FakeArray, **overrides):
    kwargs = {
        "array_path": HOMOGRAPHY_ARRAY_PATH,
        "expected_from_space_id": "source_camera_image_px",
        "expected_to_space_id": "stimulus_canvas_px",
        "expected_camera_id": "2010093",
        "expected_source_reference_extent": CAMERA_EXTENT,
        "expected_target_reference_extent": CANVAS_EXTENT,
        "expected_transform_sha256": _camera_to_canvas().digest(),
        "expected_calibration_ref": CAMERA_CALIBRATION_PATH,
    }
    kwargs.update(overrides)
    return load_bound_directed_homography(node, **kwargs)


def test_stamp_and_load_bind_exact_array_metadata_and_non_self_inverse_matrix() -> None:
    node = FakeArray(NON_SELF_INVERSE)
    transform = _camera_to_canvas()

    stamped = stamp_directed_homography(node, transform)
    loaded = _load_bound(node)

    assert stamped.array_path == HOMOGRAPHY_ARRAY_PATH
    assert loaded.array_path == HOMOGRAPHY_ARRAY_PATH
    assert loaded.transform == transform
    assert loaded.transform_sha256 == transform.digest()
    assert loaded.matrix_sha256 == homography_matrix_sha256(NON_SELF_INVERSE)
    assert loaded.matrix.flags.writeable is False
    assert node.attrs[DIRECTED_TRANSFORM_ATTR] == transform.to_dict()
    assert node.attrs[
        f"{DIRECTED_TRANSFORM_ATTR}{DIRECTED_TRANSFORM_DIGEST_SUFFIX}"
    ] == transform.digest()
    np.testing.assert_array_equal(loaded.matrix, NON_SELF_INVERSE)


def test_bound_loader_rejects_wrong_camera_direction_and_extent() -> None:
    node = FakeArray(NON_SELF_INVERSE)
    stamp_directed_homography(node, _camera_to_canvas())

    with pytest.raises(DirectedTransformError, match="camera mismatch"):
        _load_bound(node, expected_camera_id="different_camera")
    with pytest.raises(DirectedTransformError, match="direction mismatch"):
        _load_bound(
            node,
            expected_from_space_id="stimulus_canvas_px",
            expected_to_space_id="source_camera_image_px",
            expected_source_reference_extent=CANVAS_EXTENT,
            expected_target_reference_extent=CAMERA_EXTENT,
        )
    wrong_extent = TransformReferenceExtent(
        width=640,
        height=640,
        units="px",
        authority=CAMERA_EXTENT.authority,
    )
    with pytest.raises(DirectedTransformError, match="Source reference extent mismatch"):
        _load_bound(node, expected_source_reference_extent=wrong_extent)


def test_bound_loader_rejects_tampered_metadata_and_matrix() -> None:
    metadata_node = FakeArray(NON_SELF_INVERSE)
    stamp_directed_homography(metadata_node, _camera_to_canvas())
    metadata_node.attrs[DIRECTED_TRANSFORM_ATTR]["transform_id"] = "tampered"
    with pytest.raises(DirectedTransformError, match="digest does not match"):
        _load_bound(metadata_node)

    matrix_node = FakeArray(NON_SELF_INVERSE)
    stamp_directed_homography(matrix_node, _camera_to_canvas())
    matrix_node.data[0, 2] += 1.0
    with pytest.raises(DirectedTransformError, match="matrix digest"):
        _load_bound(matrix_node)


def test_bound_loader_rejects_node_path_mismatch() -> None:
    node = FakeArray(NON_SELF_INVERSE, path=f"{CAMERA_CALIBRATION_PATH}/other")
    stamp_directed_homography(node, _camera_to_canvas())

    with pytest.raises(DirectedTransformError, match="Array path mismatch"):
        _load_bound(node)


@pytest.mark.parametrize(
    "calibration_ref",
    [
        "/analysis/stimulus_runs/stim_1/calibration/2010093",
        "analysis/stimulus_runs/stim_1/calibration/2010093/",
        "analysis/stimulus_runs//stim_1/calibration/2010093",
        "analysis/stimulus_runs/../calibration/2010093",
    ],
)
def test_parse_build_and_stamp_reject_noncanonical_calibration_ref(
    calibration_ref: str,
) -> None:
    payload = _camera_to_canvas().to_dict()
    payload["calibration_ref"] = calibration_ref
    with pytest.raises(DirectedTransformError, match="canonical archive-relative"):
        parse_directed_homography(payload)

    with pytest.raises(DirectedTransformError, match="canonical archive-relative"):
        build_directed_homography(
            transform_id="invalid_calibration_ref",
            matrix=NON_SELF_INVERSE,
            from_space_id="source_camera_image_px",
            to_space_id="stimulus_canvas_px",
            source_reference_extent=CAMERA_EXTENT,
            target_reference_extent=CANVAS_EXTENT,
            calibration_ref=calibration_ref,
            camera_id="2010093",
        )

    node = FakeArray(NON_SELF_INVERSE)
    with pytest.raises(DirectedTransformError, match="canonical archive-relative"):
        stamp_directed_homography(node, payload)


def test_strict_bound_loader_requires_pointer_digest_and_calibration_ref() -> None:
    signature = inspect.signature(load_bound_directed_homography)

    assert (
        signature.parameters["expected_transform_sha256"].default
        is inspect.Parameter.empty
    )
    assert (
        signature.parameters["expected_calibration_ref"].default
        is inspect.Parameter.empty
    )


def test_parser_rejects_unknown_fields_and_unsupported_spaces() -> None:
    payload = copy.deepcopy(_camera_to_canvas().to_dict())
    payload["unexpected"] = True
    with pytest.raises(DirectedTransformError, match="unknown"):
        parse_directed_homography(payload)

    payload = _camera_to_canvas().to_dict()
    payload["from_space_id"] = "presentation_viewport_px"
    with pytest.raises(DirectedTransformError, match="unsupported coordinate space"):
        parse_directed_homography(payload)
