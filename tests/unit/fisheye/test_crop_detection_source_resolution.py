from __future__ import annotations

import numpy as np
import pytest
import zarr

from fisheye.tracking.crop import _ensure_numpy_array, get_detection_source_info


def _build_root(tmp_path):
    root = zarr.open_group(str(tmp_path / "test.zarr"), mode="w")

    detect_runs = root.create_group("detect_runs")
    detect = detect_runs.create_group("detect_a")
    detect_runs.attrs["latest"] = "detect_a"
    detect.create_array(
        "frame_indices",
        data=np.array([0, 1], dtype=np.int32),
        overwrite=True,
    )
    detect.create_array(
        "bbox_norm_coords",
        data=np.array([[0.5, 0.5, 0.1, 0.1], [0.5, 0.5, 0.1, 0.1]], dtype=np.float32),
        overwrite=True,
    )
    return root


def test_preferred_falls_back_to_detect_when_refined_stage_is_incomplete(tmp_path):
    root = _build_root(tmp_path)
    refined_runs = root.create_group("refined_detect_runs")
    refined = refined_runs.create_group("refined_a")
    refined_runs.attrs["latest"] = "refined_a"
    interpolated = refined.create_group("interpolated")
    interpolated.create_array(
        "bbox_norm_coords",
        data=np.array([[0.5, 0.5, 0.1, 0.1]], dtype=np.float32),
        overwrite=True,
    )

    source_path, _group, _detection_source, source_type = get_detection_source_info(
        root=root,
        source_type="preferred",
    )

    assert source_path == "detect_runs/detect_a"
    assert source_type == "detect"


def test_explicit_refined_source_raises_clear_error_when_incomplete(tmp_path):
    root = _build_root(tmp_path)
    refined_runs = root.create_group("refined_detect_runs")
    refined = refined_runs.create_group("refined_a")
    refined_runs.attrs["latest"] = "refined_a"
    interpolated = refined.create_group("interpolated")
    interpolated.create_array(
        "bbox_norm_coords",
        data=np.array([[0.5, 0.5, 0.1, 0.1]], dtype=np.float32),
        overwrite=True,
    )

    with pytest.raises(ValueError, match="missing required arrays: frame_indices"):
        get_detection_source_info(root=root, source_type="interpolated")


class _FakeGpuArray:
    """Mimic CuPy-style explicit host transfer with implicit conversion disabled."""

    def __init__(self, values: np.ndarray) -> None:
        self._values = values

    def get(self) -> np.ndarray:
        return self._values

    def __array__(self, *args, **kwargs):  # pragma: no cover - should never be used
        raise TypeError(
            "Implicit conversion to a NumPy array is not allowed. "
            "Please use `.get()` to construct a NumPy array explicitly."
        )


def test_ensure_numpy_array_uses_get_for_gpu_like_arrays() -> None:
    gpu_like = _FakeGpuArray(np.array([0, 1, 0], dtype=np.int8))
    out = _ensure_numpy_array(gpu_like, dtype="i1", name="detection_source")
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.int8
    assert out.tolist() == [0, 1, 0]
