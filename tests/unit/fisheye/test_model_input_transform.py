import numpy as np
import pytest
import torch

from fisheye.shared.model_input_transform import (
    model_input_transform_from_attrs,
    resolve_model_input_transform,
    resolve_stride_aligned_square_input_transform,
)


def test_stride_aligned_348_contract_uses_centered_352_padding() -> None:
    transform = resolve_stride_aligned_square_input_transform((348, 348))

    assert transform.to_attrs() == {
        "name": "pad_to_size",
        "native_shape_hw": [348, 348],
        "model_shape_hw": [352, 352],
        "pad_top": 2,
        "pad_bottom": 2,
        "pad_left": 2,
        "pad_right": 2,
        "coordinate_mapping": "native_xy = model_xy - [pad_left, pad_top]",
    }
    native = np.ones((1, 348, 348), dtype=np.uint8)
    padded = transform.apply_numpy_luma_batch(native)
    assert padded.shape == (1, 352, 352)
    np.testing.assert_array_equal(padded[:, 2:350, 2:350], native)
    assert int(padded[:, :2, :].sum()) == 0
    np.testing.assert_allclose(
        transform.invert_points_xy(np.array([[2.0, 2.0], [349.0, 349.0]])),
        [[0.0, 0.0], [347.0, 347.0]],
    )


def test_pad_to_size_numpy_and_coordinate_inverse() -> None:
    transform = resolve_model_input_transform((4, 6), mode="pad_to_size", model_hw=(8, 10))

    assert transform.to_attrs()["name"] == "pad_to_size"
    assert transform.to_attrs()["pad_top"] == 2
    assert transform.to_attrs()["pad_left"] == 2

    batch = np.arange(24, dtype=np.uint8).reshape(1, 4, 6)
    padded = transform.apply_numpy_luma_batch(batch)
    assert padded.shape == (1, 8, 10)
    np.testing.assert_array_equal(padded[0, 2:6, 2:8], batch[0])
    assert int(padded[0, 0, 0]) == 0

    model_points = np.array([[2.0, 2.0], [7.0, 5.0]], dtype=np.float64)
    native_points = transform.invert_points_xy(model_points)
    np.testing.assert_allclose(native_points, [[0.0, 0.0], [5.0, 3.0]])

    model_box = np.array([2.0, 2.0, 7.0, 5.0], dtype=np.float32)
    native_box = transform.invert_boxes_xyxy(model_box)
    np.testing.assert_allclose(native_box, [0.0, 0.0, 5.0, 3.0])


def test_pad_to_size_torch_and_output_crop() -> None:
    transform = resolve_model_input_transform((4, 6), mode="auto", model_hw=(8, 10))

    batch = torch.ones((2, 1, 4, 6), dtype=torch.float32)
    padded = transform.apply_torch_image_batch(batch)
    assert tuple(padded.shape) == (2, 1, 8, 10)
    assert torch.equal(padded[:, :, 2:6, 2:8], batch)
    assert float(padded[:, :, :2, :].sum()) == 0.0

    model_output = torch.arange(2 * 3 * 8 * 10, dtype=torch.float32).reshape(2, 3, 8, 10)
    cropped = transform.crop_torch_output(model_output)
    assert tuple(cropped.shape) == (2, 3, 4, 6)
    assert torch.equal(cropped, model_output[:, :, 2:6, 2:8])


def test_identity_rejects_shape_mismatch() -> None:
    try:
        resolve_model_input_transform((4, 4), mode="identity", model_hw=(8, 8))
    except ValueError as exc:
        assert "identity" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("identity transform accepted mismatched shapes")


def test_persisted_transform_round_trips_without_defaults() -> None:
    expected = resolve_model_input_transform(
        (384, 384), mode="pad_to_size", model_hw=(512, 512)
    )

    loaded = model_input_transform_from_attrs(expected.to_attrs())

    assert loaded == expected


@pytest.mark.parametrize(
    "change",
    (
        lambda attrs: attrs.pop("pad_left"),
        lambda attrs: attrs.update({"pad_left": True}),
        lambda attrs: attrs.update({"model_shape_hw": [512, 384]}),
        lambda attrs: attrs.update({"coordinate_mapping": "guessed"}),
    ),
)
def test_persisted_transform_rejects_partial_or_inconsistent_attrs(change) -> None:
    attrs = resolve_model_input_transform(
        (384, 384), mode="pad_to_size", model_hw=(512, 512)
    ).to_attrs()
    change(attrs)

    with pytest.raises(ValueError):
        model_input_transform_from_attrs(attrs)
