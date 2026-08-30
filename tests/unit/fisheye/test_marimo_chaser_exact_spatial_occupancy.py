from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import plotly.graph_objects as go
import pytest

from apps.marimo.components.chaser_exact.spatial_occupancy import (
    SPATIAL_OCCUPANCY_DISPLAY_RECIPE,
    _spatial_values,
    build_exact_spatial_occupancy_output,
)


class _SpatialHandle:
    successor_kind = "chaser_spatial_occupancy"
    deep_audited = True

    def __init__(self) -> None:
        shape = (2, 3, 2, 2)
        counts = np.zeros(shape, dtype=np.int64)
        counts[0, :, 0, 0] = 1
        counts[0, :, 0, 1] = 1
        counts[1, :, 0, 1] = 2
        candidate = np.full((2, 3), 4, dtype=np.int64)
        declared = np.full((2, 3), 3, dtype=np.int64)
        finite = np.full((2, 3), 3, dtype=np.int64)
        in_arena = np.full((2, 3), 2, dtype=np.int64)
        invalid = np.full((2, 3), 1, dtype=np.int64)
        out_of_arena = np.full((2, 3), 1, dtype=np.int64)
        self.arrays = {
            "occupancy_count": counts,
            "occupancy_density_valid_in_arena": counts / 2.0,
            "occupancy_fraction_candidate_epoch": counts / 4.0,
            "x_bin_edges_mm": np.asarray([-1.0, 0.0, 1.0]),
            "y_bin_edges_mm": np.asarray([-1.0, 0.0, 1.0]),
            "arena_bin_center_mask": np.asarray(
                [[True, False], [True, True]], dtype=bool
            ),
            "candidate_frame_count": candidate,
            "declared_valid_position_frame_count": declared,
            "finite_valid_position_frame_count": finite,
            "in_arena_position_frame_count": in_arena,
            "invalid_position_frame_count": invalid,
            "out_of_arena_position_frame_count": out_of_arena,
            "in_arena_coverage_fraction_candidate": in_arena / candidate,
            "in_arena_fraction_finite_valid": in_arena / finite,
        }
        self.scientific_manifest = {
            "dimensions": {
                "n_providers": 2,
                "n_epochs": 3,
                "grid_rows": 2,
                "grid_columns": 2,
            },
            "identity_registries": {
                "provider_role": {"0": "keypoint", "1": "detection"},
                "epoch_role": {
                    "0": "chaser_pre",
                    "1": "chaser_training",
                    "2": "chaser_post",
                },
            },
            "arena": {"radius_mm": 1.0},
            "grid": {
                "coordinate_orientation": "+x_right_+y_down",
                "normalization_policy_id": (
                    "valid_in_arena_and_candidate_epoch_denominators_v1"
                ),
                "bin_width_mm": 1.0,
            },
        }

    def array(self, name: str) -> np.ndarray:
        return self.arrays[name]

    def require_verified_arrays(self, names) -> None:
        assert set(names).issubset(self.arrays)


class _Marimo:
    @staticmethod
    def callout(value: str, *, kind: str) -> dict[str, str]:
        return {"value": value, "kind": kind}

    @staticmethod
    def vstack(values: list[Any]) -> list[Any]:
        return values


def _projection(handle: _SpatialHandle | None = None) -> Any:
    return SimpleNamespace(
        spatial=handle or _SpatialHandle(),
        recording_id="recording-1",
        provider_ids=("keypoint.v1", "detection.v1"),
        provenance={"bundle_manifest_sha256": "a" * 64},
    )


def test_spatial_heatmap_uses_persisted_density_and_provider_difference() -> None:
    output = build_exact_spatial_occupancy_output(_Marimo, go, _projection())

    figure = output[1]
    assert len(figure.data) == 9
    keypoint_pre = np.asarray(figure.data[0].z)
    detection_pre = np.asarray(figure.data[1].z)
    difference_pre = np.asarray(figure.data[2].z)
    assert keypoint_pre[0, 1] == 50.0
    assert detection_pre[0, 1] == 100.0
    assert difference_pre[0, 1] == 50.0
    assert not bool(np.asarray(figure.data[0].customdata)[0, 1])
    assert np.isfinite(keypoint_pre[0, 1])
    display = figure.layout.meta["spatial_occupancy_display"]
    assert display["recipe_id"] == SPATIAL_OCCUPANCY_DISPLAY_RECIPE
    assert display["source_array"] == "occupancy_density_valid_in_arena"
    assert display["scientific_recomputation"] is False
    assert display["interpolation"] == "prohibited"


def test_spatial_projection_rejects_denominator_nonconservation() -> None:
    handle = _SpatialHandle()
    handle.arrays["in_arena_position_frame_count"] = np.full((2, 3), 3, dtype=np.int64)

    with pytest.raises(ValueError, match="conservation"):
        _spatial_values(_projection(handle))
