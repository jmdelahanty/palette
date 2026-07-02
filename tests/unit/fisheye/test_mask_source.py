from __future__ import annotations

import numpy as np
import pytest
import zarr

from fisheye.shared.mask_source import load_mask_bundle


def _probability_run(*, with_encoding: bool = True) -> zarr.Group:
    run = zarr.group()
    if with_encoding:
        run.attrs["probabilities_encoding"] = "linear_uint8_0_255"
    probs = np.asarray([0, 128, 255, 255], dtype=np.uint8).reshape(1, 1, 2, 2)
    run.create_array("mask_probs_roi", data=probs, overwrite=True)
    return run


def test_load_mask_bundle_dequantizes_uint8_probabilities() -> None:
    bundle = load_mask_bundle(_probability_run())

    assert bundle.probs is not None
    out = np.asarray(bundle.probs).reshape(-1)
    assert out.dtype == np.float32
    assert np.isclose(out[0], 0.0)
    assert np.isclose(out[1], 128.0 / 255.0)
    assert np.isclose(out[2], 1.0)


def test_load_mask_bundle_requires_probabilities_encoding() -> None:
    with pytest.raises(ValueError, match="mask_probs_roi.*missing.*uint8"):
        load_mask_bundle(_probability_run(with_encoding=False))
