from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import plotly.graph_objects as go

from apps.marimo.components.chaser_exact.radial_near_field import (
    _metric_rows,
    build_exact_radial_near_field_output,
)


class _RadialHandle:
    def __init__(self) -> None:
        metric_window = np.asarray([0, 1], dtype=np.int64)
        radial_window = np.asarray([0, 0, 1, 1], dtype=np.int64)
        self.arrays = {
            "metric_epoch_role_code": np.asarray([0, 0], dtype=np.uint8),
            "metric_epoch_window_id": metric_window,
            "metric_behavior_role_code": np.asarray([1, 1], dtype=np.uint8),
            "metric_chaser_identity_code": np.asarray([2, 2], dtype=np.uint16),
            "metric_distance_p25_mm": np.asarray([2.0, 3.0]),
            "metric_distance_p50_mm": np.asarray([4.0, 5.0]),
            "metric_distance_p75_mm": np.asarray([6.0, 7.0]),
            "metric_near_zone_fraction_valid": np.asarray([0.4, 0.5]),
            "metric_near_zone_dwell_s": np.asarray([1.0, 2.0]),
            "metric_near_zone_entry_rate_per_min_valid_time": np.asarray([3.0, 4.0]),
            "radial_epoch_role_code": np.asarray([0, 0, 0, 0], dtype=np.uint8),
            "radial_epoch_window_id": radial_window,
            "radial_behavior_role_code": np.asarray([1, 1, 1, 1], dtype=np.uint8),
            "radial_chaser_identity_code": np.asarray([2, 2, 2, 2], dtype=np.uint16),
            "radial_bin_start_mm": np.asarray([0.0, 5.0, 0.0, 5.0]),
            "radial_bin_end_mm": np.asarray([5.0, 10.0, 5.0, 10.0]),
            "radial_selection_index_geometric": np.asarray([0.8, 1.2, 0.9, 1.1]),
        }
        self.scientific_manifest = {
            "config": {"near_zone_radius_mm": 10.0},
            "identity_registries": {
                "epoch_role": {"0": "chaser_training"},
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


def test_radial_near_field_preserves_distinct_exact_epoch_windows() -> None:
    first = _RadialHandle()
    second = _RadialHandle()
    projection = SimpleNamespace(
        radials=(first, second),
        provider_ids=("keypoint.v1", "detection.v1"),
        recording_id="recording-1",
        provenance={"bundle_manifest_sha256": "a" * 64},
    )

    rows = _metric_rows(first)
    output = build_exact_radial_near_field_output(_Marimo, go, projection)

    assert set(rows) == {(0, 0, 1, 2), (0, 1, 1, 2)}
    distance, _, _, radial = output[1:]
    assert len(distance.data) == 2
    assert len(distance.data[0].x) == 2
    assert "window 0" in distance.data[0].x[0]
    assert "window 1" in distance.data[0].x[1]
    assert len(radial.data) == 4
