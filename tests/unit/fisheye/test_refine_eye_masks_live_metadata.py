from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from fisheye.refinement import refine_eye_masks as mod


def test_get_zarr_array_opens_without_consolidated_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    mod._ZARR_GROUP_CACHE.clear()
    mod._ZARR_ARRAY_CACHE.clear()

    seen: dict[str, Any] = {}
    fake_group = {"demo/path": np.array([1, 2, 3], dtype=np.int32)}

    def _fake_open(path: str, mode: str = "r", **kwargs: Any) -> dict[str, Any]:
        seen["path"] = path
        seen["mode"] = mode
        seen["kwargs"] = kwargs
        return fake_group

    monkeypatch.setattr(mod.zarr, "open", _fake_open)
    arr = mod._get_zarr_array("demo.zarr", "demo/path")
    assert np.array_equal(arr, np.array([1, 2, 3], dtype=np.int32))
    assert seen["mode"] == "r"
    assert seen["kwargs"].get("use_consolidated") is False


def test_refine_eye_masks_root_open_uses_live_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    class _FakeRoot(dict):
        def get(self, key: str, default: Any = None) -> Any:
            return super().get(key, default)

    def _fake_open(path: str, mode: str = "a", **kwargs: Any) -> _FakeRoot:
        seen["path"] = path
        seen["mode"] = mode
        seen["kwargs"] = kwargs
        return _FakeRoot()

    monkeypatch.setattr(mod.zarr, "open", _fake_open)

    with pytest.raises(ValueError, match="missing eye_masks_runs"):
        mod.refine_eye_masks("demo.zarr")

    assert seen["mode"] == "a"
    assert seen["kwargs"].get("use_consolidated") is False


def test_process_and_write_chunk_open_uses_live_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    roi_output = mod.ROIOutput(
        masks=np.ones((2, 3, 3), dtype=np.uint8),
        ellipse_params=np.zeros((2, 5), dtype=np.float32),
        ellipse_success=np.array([True, True], dtype=bool),
        centroids=np.zeros((2, 2), dtype=np.float32),
        contours=(None, None),
        eye_separation=5.0,
        reason="refined",
        smoothing_changed=np.array([False, False], dtype=bool),
        reassigned_pixels=0,
    )

    run_group = {
        "masks_roi": np.zeros((1, 2, 3, 3), dtype=np.uint8),
        "ellipse_params": np.zeros((1, 2, 5), dtype=np.float32),
        "ellipse_success": np.zeros((1, 2), dtype=bool),
        "eye_separation": np.full((1,), np.nan, dtype=np.float32),
    }
    root = {"refined_eye_masks_runs/refined_001": run_group}

    seen: dict[str, Any] = {}

    def _fake_open(path: str, mode: str = "a", **kwargs: Any) -> dict[str, Any]:
        seen["path"] = path
        seen["mode"] = mode
        seen["kwargs"] = kwargs
        return root

    monkeypatch.setattr(mod.zarr, "open", _fake_open)
    monkeypatch.setattr(
        mod,
        "_process_refine_chunk",
        lambda *args, **kwargs: [(0, roi_output)],
    )

    results = mod._process_and_write_chunk(
        binary_chunk=np.zeros((1, 2, 3, 3), dtype=np.uint8),
        probs_chunk=None,
        zarr_path="demo.zarr",
        run_group_path="refined_eye_masks_runs/refined_001",
        keypoints_path="unused/keypoints",
        heading_path="unused/heading",
        success_path="unused/success",
        start=0,
        stop=1,
        write_probabilities=False,
    )

    assert len(results) == 1
    assert seen["mode"] == "a"
    assert seen["kwargs"].get("use_consolidated") is False
    assert int(run_group["masks_roi"][0].sum()) > 0
