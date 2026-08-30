from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import plotly.graph_objects as go
import pytest

from apps.marimo.components.chaser_exact.distance_distributions import (
    DISTANCE_DISTRIBUTION_DISPLAY_RECIPE,
    _distribution_tables,
    build_exact_distance_distributions_output,
)


class _RadialHandle:
    def __init__(self) -> None:
        metric_key = {
            "metric_epoch_role_code": np.asarray([0], dtype=np.uint8),
            "metric_epoch_window_id": np.asarray([0], dtype=np.int64),
            "metric_behavior_role_code": np.asarray([1], dtype=np.uint8),
            "metric_chaser_identity_code": np.asarray([2], dtype=np.uint16),
        }
        cdf_key = {
            "cdf_epoch_role_code": np.asarray([0, 0, 0], dtype=np.uint8),
            "cdf_epoch_window_id": np.asarray([0, 0, 0], dtype=np.int64),
            "cdf_behavior_role_code": np.asarray([1, 1, 1], dtype=np.uint8),
            "cdf_chaser_identity_code": np.asarray([2, 2, 2], dtype=np.uint16),
        }
        radial_key = {
            "radial_epoch_role_code": np.asarray([0, 0], dtype=np.uint8),
            "radial_epoch_window_id": np.asarray([0, 0], dtype=np.int64),
            "radial_behavior_role_code": np.asarray([1, 1], dtype=np.uint8),
            "radial_chaser_identity_code": np.asarray([2, 2], dtype=np.uint16),
        }
        self.arrays = {
            **metric_key,
            "metric_candidate_frame_count": np.asarray([5], dtype=np.int64),
            "metric_valid_distance_frame_count": np.asarray([4], dtype=np.int64),
            "metric_wall_excluded_valid_frame_count": np.asarray([2], dtype=np.int64),
            **cdf_key,
            "cdf_threshold_mm": np.asarray([5.0, 10.0, 15.0]),
            "cdf_fraction_at_or_below": np.asarray([0.25, 0.75, 1.0]),
            **radial_key,
            "radial_bin_start_mm": np.asarray([0.0, 5.0]),
            "radial_bin_end_mm": np.asarray([5.0, 10.0]),
            "radial_observed_count": np.asarray([1, 3], dtype=np.int64),
            "radial_observed_fraction": np.asarray([0.25, 0.75]),
            "radial_expected_available_area_mm2_frames": np.asarray([10.0, 30.0]),
            "radial_expected_fraction_geometric": np.asarray([0.25, 0.75]),
            "radial_selection_index_geometric": np.asarray([1.0, 1.0]),
            "radial_wall_excluded_observed_count": np.asarray([1, 1], dtype=np.int64),
            "radial_wall_excluded_observed_fraction": np.asarray([0.5, 0.5]),
            "radial_wall_excluded_expected_available_area_mm2_frames": np.asarray(
                [2.0, 2.0]
            ),
            "radial_wall_excluded_expected_fraction_geometric": np.asarray([0.5, 0.5]),
            "radial_wall_excluded_selection_index_geometric": np.asarray([1.0, 1.0]),
        }
        self.scientific_manifest = {
            "config": {"perimeter_band_mm": 5.0},
            "identity_registries": {
                "epoch_role": {"0": "chaser_pre"},
                "behavior_role": {"1": "aggressive"},
                "chaser": {"2": "stimulus-v1:chaser_index:0"},
            },
        }

    def array(self, name: str) -> np.ndarray:
        return self.arrays[name]

    def require_verified_arrays(self, names: tuple[str, ...]) -> None:
        assert set(names).issubset(self.arrays)


class _Marimo:
    @staticmethod
    def callout(value: str, *, kind: str) -> dict[str, str]:
        return {"value": value, "kind": kind}

    @staticmethod
    def vstack(values: list[Any]) -> list[Any]:
        return values


def _projection(*handles: _RadialHandle) -> Any:
    radial_handles = handles or (_RadialHandle(), _RadialHandle())
    return SimpleNamespace(
        radials=radial_handles,
        provider_ids=("keypoint.v1", "detection.v1"),
        provenance={"bundle_manifest_sha256": "a" * 64},
    )


def test_distance_distributions_render_only_persisted_thresholds_and_bins() -> None:
    output = build_exact_distance_distributions_output(_Marimo, go, _projection())

    assert len(output) == 5
    cdf, ordinary, wall, selection = output[1:]
    np.testing.assert_array_equal(cdf.data[0].x, [5.0, 10.0, 15.0])
    np.testing.assert_array_equal(cdf.data[0].y, [25.0, 75.0, 100.0])
    np.testing.assert_array_equal(ordinary.data[0].x, [2.5, 7.5])
    np.testing.assert_array_equal(ordinary.data[0].y, [25.0, 75.0])
    np.testing.assert_array_equal(ordinary.data[1].y, [25.0, 75.0])
    np.testing.assert_array_equal(wall.data[0].y, [50.0, 50.0])
    np.testing.assert_array_equal(selection.data[0].y, [1.0, 1.0])
    assert "5 mm wall-excluded" in wall.layout.title.text
    display = cdf.layout.meta["distance_distribution_display"]
    assert display["recipe_id"] == DISTANCE_DISTRIBUTION_DISPLAY_RECIPE
    assert display["cdf_thresholds"] == "persisted_exact_no_interpolation"
    assert display["radial_bin_edges"] == "persisted_exact_no_rebinning"
    assert display["scientific_recomputation"] is False
    assert display["perimeter_band_mm"] == 5.0
    assert display["strata"][0]["cdf_thresholds_mm"] == [5.0, 10.0, 15.0]
    assert display["strata"][0]["radial_bin_edges_mm"] == [0.0, 5.0, 10.0]
    assert (
        display["provider_denominators"][0]["strata"][0]["valid_distance_frame_count"]
        == 4
    )
    assert (
        display["provider_denominators"][0]["strata"][0][
            "wall_excluded_valid_frame_count"
        ]
        == 2
    )


def test_distance_distributions_reject_nonmonotone_persisted_cdf() -> None:
    handle = _RadialHandle()
    handle.arrays["cdf_fraction_at_or_below"] = np.asarray([0.25, 0.2, 1.0])

    with pytest.raises(ValueError, match="not monotone"):
        _distribution_tables(handle)


def test_distance_distributions_reject_unsorted_persisted_bins() -> None:
    handle = _RadialHandle()
    handle.arrays["radial_bin_start_mm"] = np.asarray([5.0, 0.0])
    handle.arrays["radial_bin_end_mm"] = np.asarray([10.0, 5.0])

    with pytest.raises(ValueError, match="invalid or noncontiguous"):
        _distribution_tables(handle)


def test_distance_distributions_reject_count_nonconservation() -> None:
    handle = _RadialHandle()
    handle.arrays["radial_observed_count"] = np.asarray([1, 2], dtype=np.int64)

    with pytest.raises(ValueError, match="conserve valid support"):
        _distribution_tables(handle)
