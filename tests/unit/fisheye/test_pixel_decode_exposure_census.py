from __future__ import annotations

import numpy as np

from fisheye.utils.pixel_decode_exposure_census import classify_pixel_values


def test_classify_pixel_values_detects_range_expanded_lattice() -> None:
    source = np.arange(256, dtype=np.float64)
    expanded = np.clip(np.rint((source - 16.0) * 255.0 / 219.0), 0, 255).astype(np.uint8)
    values = np.tile(expanded, 512)

    classification, confidence, evidence = classify_pixel_values(values)

    assert classification == "range_expanded_like"
    assert confidence == "high"
    assert evidence["limited_expansion_forbidden_bins_present"] <= 2


def test_classify_pixel_values_detects_direct_y_for_forbidden_bins() -> None:
    values = np.tile(np.arange(256, dtype=np.uint8), 512)

    classification, confidence, evidence = classify_pixel_values(values)

    assert classification == "direct_y_like"
    assert confidence == "high"
    assert evidence["limited_expansion_forbidden_bins_present"] >= 20


def test_classify_pixel_values_marks_low_dynamic_range_indeterminate() -> None:
    values = np.full(20_000, 127, dtype=np.uint8)

    classification, confidence, evidence = classify_pixel_values(values)

    assert classification == "indeterminate"
    assert confidence == "low"
    assert evidence["reason"] == "sample_or_dynamic_range_too_small"
