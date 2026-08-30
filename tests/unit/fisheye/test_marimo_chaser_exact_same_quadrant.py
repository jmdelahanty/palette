from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import plotly.graph_objects as go
import pytest

from apps.marimo.components.chaser_exact.same_quadrant import (
    SAME_QUADRANT_DISPLAY_RECIPE,
    _same_quadrant_rows,
    build_exact_same_quadrant_output,
)


class _RadialHandle:
    def __init__(self) -> None:
        self.arrays = {
            "metric_epoch_role_code": np.asarray([0], dtype=np.uint8),
            "metric_epoch_window_id": np.asarray([0], dtype=np.int64),
            "metric_behavior_role_code": np.asarray([1], dtype=np.uint8),
            "metric_chaser_identity_code": np.asarray([2], dtype=np.uint16),
            "metric_candidate_frame_count": np.asarray([5], dtype=np.int64),
            "metric_valid_distance_frame_count": np.asarray([4], dtype=np.int64),
            "metric_same_quadrant_valid_frame_count": np.asarray([2], dtype=np.int64),
            "metric_same_quadrant_fraction_valid": np.asarray([0.5]),
            "metric_same_quadrant_fraction_candidate": np.asarray([0.4]),
        }
        self.scientific_manifest = {
            "identity_registries": {
                "epoch_role": {"0": "chaser_pre"},
                "behavior_role": {"1": "aggressive"},
                "chaser": {"2": "stimulus-v1:chaser_index:0"},
            }
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


def test_same_quadrant_keeps_valid_and_candidate_denominators_separate() -> None:
    output = build_exact_same_quadrant_output(_Marimo, go, _projection())

    assert len(output) == 3
    valid, candidate = output[1:]
    np.testing.assert_array_equal(valid.data[0].y, [50.0])
    np.testing.assert_array_equal(candidate.data[0].y, [40.0])
    np.testing.assert_array_equal(valid.data[0].customdata, [[2, 4, 5]])
    display = valid.layout.meta["same_quadrant_display"]
    assert display["recipe_id"] == SAME_QUADRANT_DISPLAY_RECIPE
    assert display["valid_denominator"] == "metric_valid_distance_frame_count"
    assert display["candidate_denominator"] == "metric_candidate_frame_count"
    assert display["scientific_recomputation"] is False
    assert (
        display["provider_strata"][0]["strata"][0]["same_quadrant_valid_frame_count"]
        == 2
    )
    assert display["provider_strata"][0]["strata"][0]["candidate_frame_count"] == 5


def test_same_quadrant_rejects_fraction_that_disagrees_with_counts() -> None:
    handle = _RadialHandle()
    handle.arrays["metric_same_quadrant_fraction_candidate"] = np.asarray([0.5])

    with pytest.raises(ValueError, match="candidate fraction disagrees"):
        _same_quadrant_rows(handle)
