import pytest

from fisheye.detection.detect_yolo import _actual_input_resize_dims


def test_decord_gpu_tensor_path_uses_canonical_resize_dims() -> None:
    assert _actual_input_resize_dims(
        [640, 640],
        None,
        decord_on_gpu=True,
    ) == [640, 640]


def test_non_decord_tensor_path_uses_canonical_resize_dims() -> None:
    assert _actual_input_resize_dims(
        [640, 640],
        None,
        decord_on_gpu=False,
        tensor_input=True,
    ) == [640, 640]


def test_numpy_path_leaves_canonical_resize_to_ultralytics() -> None:
    assert (
        _actual_input_resize_dims(
            [640, 640],
            None,
            decord_on_gpu=False,
        )
        is None
    )


def test_legacy_preresize_is_preserved_for_non_tensor_inputs() -> None:
    assert _actual_input_resize_dims(
        [480, 640],
        [480, 640],
        decord_on_gpu=False,
    ) == [480, 640]


def test_decord_gpu_tensor_path_requires_explicit_resize_dims() -> None:
    with pytest.raises(RuntimeError, match="GPU tensor decode paths require"):
        _actual_input_resize_dims(
            None,
            None,
            decord_on_gpu=True,
        )


def test_pynvvc_tensor_path_requires_explicit_resize_dims() -> None:
    with pytest.raises(RuntimeError, match="GPU tensor decode paths require"):
        _actual_input_resize_dims(
            None,
            None,
            decord_on_gpu=False,
            tensor_input=True,
        )


def test_numpy_path_without_requested_resize_uses_source_resolution() -> None:
    assert (
        _actual_input_resize_dims(
            None,
            None,
            decord_on_gpu=False,
        )
        is None
    )
