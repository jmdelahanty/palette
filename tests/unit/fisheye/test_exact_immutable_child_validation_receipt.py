from __future__ import annotations

import numpy as np
import pytest

from fisheye.analysis_workflows.exact_immutable_child_validation_receipt import (
    ExactImmutableChildValidationReceiptError,
    _streaming_declared_array_values_sha256,
)
from fisheye.shared.coordinate_frame_record import array_values_sha256


class _Array:
    def __init__(self, values: np.ndarray) -> None:
        self.values = np.asarray(values)
        self.dtype = self.values.dtype
        self.shape = self.values.shape
        self.attrs = {
            "palette_storage_schema_id": "palette.columnar_zarr_storage.v1",
            "palette_storage_writer": "fisheye.shared.zarr.columnar.store_array",
        }

    def __getitem__(self, key: object) -> np.ndarray:
        return self.values[key]


def _encoded(values: np.ndarray, *, width: int) -> np.ndarray:
    encoded = np.zeros((values.shape[0], width), dtype=np.uint8)
    for index, value in enumerate(values):
        payload = bytes(value).rstrip(b"\x00")
        encoded[index, : len(payload)] = np.frombuffer(payload, dtype=np.uint8)
    return encoded


def test_streaming_receipt_reconstructs_columnar_fixed_byte_identity() -> None:
    logical = np.asarray(
        [b"chaser_pre", b"chaser_training", b"chaser_post"],
        dtype="S32",
    )

    observed = _streaming_declared_array_values_sha256(
        _Array(_encoded(logical, width=16)),
        expected_dtype=logical.dtype.str,
        expected_shape=list(logical.shape),
    )

    assert observed == array_values_sha256(logical)


def test_streaming_receipt_rejects_bytes_outside_declared_logical_width() -> None:
    logical = np.asarray([b"pre", b"post"], dtype="S8")
    encoded = _encoded(logical, width=16)
    encoded[0, 9] = 1

    with pytest.raises(
        ExactImmutableChildValidationReceiptError,
        match="outside its logical width",
    ):
        _streaming_declared_array_values_sha256(
            _Array(encoded),
            expected_dtype=logical.dtype.str,
            expected_shape=list(logical.shape),
        )
