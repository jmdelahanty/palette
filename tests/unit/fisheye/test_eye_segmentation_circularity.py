from __future__ import annotations

import numpy as np
import pytest
import zarr

from fisheye.segmentation.eye_segmentation import (
    EyeSegmentationConfig,
    _apply_tuned_parameters,
    _process_roi_data,
    _select_region,
    _set_config_value,
)


def test_set_config_value_clamps_min_circularity() -> None:
    cfg = EyeSegmentationConfig()
    _set_config_value(cfg, "min_circularity", 1.7)
    assert cfg.min_circularity == pytest.approx(1.0)

    _set_config_value(cfg, "min_circularity", -0.4)
    assert cfg.min_circularity == pytest.approx(0.0)

    _set_config_value(cfg, "min_circularity", None)
    assert cfg.min_circularity is None


def test_select_region_marks_non_circular_filter() -> None:
    binary = np.zeros((32, 32), dtype=bool)
    binary[15:17, 6:26] = True  # Long strip -> low circularity

    cfg = EyeSegmentationConfig(
        min_area=5,
        max_area=None,
        min_circularity=0.7,
        closing_radius=0,
        opening_radius=0,
    )
    region_mask, meta = _select_region(binary, (16.0, 16.0), cfg)
    assert region_mask is None
    assert meta is not None
    assert bool(meta.get("filtered_non_circular")) is True


def test_select_region_accepts_circular_component() -> None:
    yy, xx = np.ogrid[:32, :32]
    binary = ((xx - 16) ** 2 + (yy - 16) ** 2) <= 25

    cfg = EyeSegmentationConfig(
        min_area=5,
        max_area=None,
        min_circularity=0.5,
        closing_radius=0,
        opening_radius=0,
    )
    region_mask, meta = _select_region(binary, (16.0, 16.0), cfg)
    assert region_mask is not None
    assert int(region_mask.sum()) > 0
    assert meta is not None
    assert bool(meta.get("filtered_non_circular")) is False


def test_process_roi_data_rejects_non_circular_regions(monkeypatch: pytest.MonkeyPatch) -> None:
    roi_img = np.full((40, 40), 220, dtype=np.uint8)
    kp = np.array([[20.0, 30.0], [11.0, 13.0], [28.0, 13.0]], dtype=np.float32)

    binary_strip = np.zeros((17, 17), dtype=bool)
    binary_strip[8:10, 2:15] = True

    def _mock_global_threshold(patch: np.ndarray, config: EyeSegmentationConfig) -> tuple[np.ndarray, np.ndarray]:
        return binary_strip.copy(), patch

    monkeypatch.setattr("fisheye.segmentation.eye_segmentation._global_threshold", _mock_global_threshold)

    cfg = EyeSegmentationConfig(
        roi_padding=8,
        pre_threshold=None,
        min_area=8,
        max_area=None,
        min_circularity=0.7,
        closing_radius=0,
        opening_radius=0,
        min_eye_separation=0.0,
        max_eye_separation=None,
    )
    result = _process_roi_data(0, roi_img, kp, True, cfg)
    assert result["reject_reason"] == "non_circular"
    assert result["ellipse_success"] == [False, False]


def test_process_roi_data_uses_no_region_when_nothing_detected() -> None:
    roi_img = np.full((40, 40), 220, dtype=np.uint8)
    kp = np.array([[20.0, 30.0], [11.0, 13.0], [28.0, 13.0]], dtype=np.float32)

    cfg = EyeSegmentationConfig(
        roi_padding=8,
        pre_threshold=1,
        min_area=8,
        max_area=None,
        min_circularity=0.7,
        closing_radius=0,
        opening_radius=0,
    )
    result = _process_roi_data(0, roi_img, kp, True, cfg)
    assert result["reject_reason"] == "no_region"


def test_apply_tuned_parameters_reads_min_circularity(tmp_path) -> None:
    zarr_path = tmp_path / "sample.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    analysis = root.create_group("analysis_metadata")
    analysis.attrs["eye_mask_tuning"] = {
        "tuned_parameters": {
            "min_circularity": 0.63,
        }
    }

    cfg = EyeSegmentationConfig()
    cfg = _apply_tuned_parameters(root, cfg)
    assert cfg.min_circularity == pytest.approx(0.63)
