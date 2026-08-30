from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import plotly.graph_objects as go
import pytest

from apps.marimo.components.chaser_exact.array_requirements import (
    EPOCH_BEHAVIOR_ARRAYS,
)
from apps.marimo.components.chaser_exact.epoch_behavior import (
    EPOCH_BEHAVIOR_DISPLAY_RECIPE,
    _epoch_behavior_values,
    build_exact_epoch_behavior_output,
)
from fisheye.analysis_workflows.protocol_semantic_chaser_selection import (
    CHASER_WINDOW_ROLES,
)


class _Marimo:
    @staticmethod
    def callout(value: str, *, kind: str) -> dict[str, str]:
        return {"value": value, "kind": kind}

    @staticmethod
    def vstack(values: list[Any]) -> list[Any]:
        return values


class _Handle:
    def __init__(self) -> None:
        roles = np.asarray(CHASER_WINDOW_ROLES, dtype="S32")
        intervals = np.asarray([b"1" * 64, b"2" * 64, b"3" * 64])
        semantic_hash = np.asarray([f"sha256:{'a' * 64}".encode()] * 3)
        step_ref = np.asarray([b"protocol_semantic_snapshot@recipe.steps[1]"] * 3)
        numeric: dict[str, np.ndarray] = {
            "track_id": np.zeros(3, dtype=np.int64),
            "window_id": np.arange(3, dtype=np.int32),
            "window_index": np.arange(3, dtype=np.int32),
            "start_frame": np.asarray([0, 10, 20], dtype=np.int64),
            "end_frame": np.asarray([9, 19, 29], dtype=np.int64),
            "start_time_s": np.asarray([0.0, 1.0, 2.0]),
            "end_time_s": np.asarray([1.0, 2.0, 3.0]),
            "duration_s": np.ones(3),
            "total_span_frames": np.full(3, 10, dtype=np.int64),
            "provider_sample_count": np.full(3, 10, dtype=np.int64),
            "valid_tracked_frame_count": np.full(3, 10, dtype=np.int64),
            "missing_frame_count": np.zeros(3, dtype=np.int64),
            "tracking_dropout_fraction": np.zeros(3),
            "valid_tracked_duration_s": np.ones(3),
            "motion_valid_sample_count": np.full(3, 9, dtype=np.int64),
            "speed_sample_count": np.full(3, 9, dtype=np.int64),
            "mean_speed_mm_s": np.asarray([5.0, 8.0, 6.0]),
            "median_speed_mm_s": np.asarray([4.5, 7.5, 5.5]),
            "p05_speed_mm_s": np.asarray([1.0, 2.0, 1.5]),
            "p95_speed_mm_s": np.asarray([9.0, 12.0, 10.0]),
            "max_speed_mm_s": np.asarray([10.0, 13.0, 11.0]),
            "total_path_mm": np.asarray([5.0, 8.0, 6.0]),
            "bout_count": np.full(3, 2, dtype=np.int64),
            "bout_rate_per_min": np.full(3, 120.0),
            "median_bout_duration_s": np.full(3, 0.2),
            "mean_bout_duration_s": np.full(3, 0.25),
            "median_bout_path_length_mm": np.full(3, 1.2),
            "mean_bout_path_length_mm": np.full(3, 1.3),
            "bout_heading_sample_count": np.full(3, 2, dtype=np.int64),
            "mean_bout_net_heading_change_deg": np.asarray([5.0, 10.0, 7.0]),
            "median_bout_net_heading_change_deg": np.asarray([4.0, 9.0, 6.0]),
            "mean_abs_bout_net_heading_change_deg": np.asarray([6.0, 11.0, 8.0]),
            "median_abs_bout_net_heading_change_deg": np.asarray([5.0, 10.0, 7.0]),
            "mean_bout_heading_path_deg": np.asarray([8.0, 14.0, 10.0]),
            "median_bout_heading_path_deg": np.asarray([7.0, 13.0, 9.0]),
            "inter_bout_interval_count": np.ones(3, dtype=np.int64),
            "mean_inter_bout_interval_s": np.full(3, 0.5),
            "median_inter_bout_interval_s": np.full(3, 0.5),
            "p05_inter_bout_interval_s": np.full(3, 0.5),
            "p95_inter_bout_interval_s": np.full(3, 0.5),
            "inter_bout_interval_rate_per_min": np.full(3, 60.0),
            "protocol_semantic_step_index": np.ones(3, dtype=np.int32),
        }
        text = {
            "window_label": roles,
            "rate_denominator": np.asarray([b"valid_tracked_duration_s"] * 3),
            "motion_validity_rule": np.asarray(
                [b"linear_sample_valid_and_transition_valid"] * 3
            ),
            "analysis_role": roles,
            "source_interval_sha256": intervals,
            "protocol_semantic_hash": semantic_hash,
            "protocol_semantic_step_ref": step_ref,
        }
        self.arrays = {
            **{f"per_epoch_fish/{name}": value for name, value in numeric.items()},
            **{f"per_epoch_fish/{name}": value for name, value in text.items()},
        }
        self._add_histograms(
            "per_epoch_bout_histograms",
            (
                ("bout_duration_s", "s"),
                ("bout_path_length_mm", "mm"),
                ("bout_net_heading_change_deg", "deg"),
                ("abs_bout_net_heading_change_deg", "deg"),
                ("bout_heading_path_deg", "deg"),
            ),
            bin_count=2,
            sample_count=2,
        )
        self._add_histograms(
            "per_epoch_inter_bout_interval_histograms",
            (("inter_bout_interval_s", "s"),),
            bin_count=1,
            sample_count=1,
        )
        role_records = [
            {
                "analysis_role": role,
                "source_window_id": index,
                "source_interval_sha256": str(index + 1) * 64,
                "selected_start_frame": index * 10,
                "selected_end_frame_exclusive": (index + 1) * 10,
                "protocol_semantic_step_index": 1,
                "protocol_semantic_step_ref": (
                    "protocol_semantic_snapshot@recipe.steps[1]"
                ),
            }
            for index, role in enumerate(CHASER_WINDOW_ROLES)
        ]
        self.manifest = {
            "parameters": {
                "track_id": 0,
                "physical_speed_level": "filtered",
                "protocol_to_acquisition_alignment": (
                    "sealed_epoch_selection_proxy_not_physical_presentation"
                ),
            },
            "sources": {
                "protocol_semantic_selection": {
                    "protocol_semantic_hash": f"sha256:{'a' * 64}",
                    "semantic_role_bindings": role_records,
                },
                "provider_motion": {"manifest_sha256": "b" * 64},
                "swim_bouts": {"lineage_hash": "c" * 64},
            },
        }
        self.run_path = "analysis/stimulus_epoch_behavior_summary_runs/epoch-v2"
        self.manifest_sha256 = "d" * 64
        self.payload_digest = "e" * 64
        self.deep_audited = False
        self.verification_mode = "receipt_bound_targeted_array_rehash_v1"
        self.receipt_digest = "f" * 64
        self.verified_array_names = EPOCH_BEHAVIOR_ARRAYS

    @property
    def semantic_selection(self) -> dict[str, Any]:
        return self.manifest["sources"]["protocol_semantic_selection"]

    def _add_histograms(
        self,
        table: str,
        metrics: tuple[tuple[str, str], ...],
        *,
        bin_count: int,
        sample_count: int,
    ) -> None:
        rows: list[dict[str, Any]] = []
        for metric, units in metrics:
            for epoch_index, role in enumerate(CHASER_WINDOW_ROLES):
                for bin_index in range(bin_count):
                    rows.append(
                        {
                            "metric_name": metric,
                            "units": units,
                            "window_id": epoch_index,
                            "window_index": epoch_index,
                            "window_label": role,
                            "start_frame": epoch_index * 10,
                            "end_frame": (epoch_index + 1) * 10 - 1,
                            "start_time_s": float(epoch_index),
                            "end_time_s": float(epoch_index + 1),
                            "duration_s": 1.0,
                            "bin_index": bin_index,
                            "bin_left": float(bin_index),
                            "bin_right": float(bin_index + 1),
                            "bin_center": float(bin_index) + 0.5,
                            "bin_width": 1.0,
                            "hist_count": 1,
                            "hist_fraction": 1.0 / bin_count,
                            "source_sample_count": sample_count,
                            "finite_sample_count": sample_count,
                            "bin_policy": "fixed_persisted_fixture",
                            "analysis_role": role,
                            "source_interval_sha256": str(epoch_index + 1) * 64,
                            "protocol_semantic_hash": f"sha256:{'a' * 64}",
                            "protocol_semantic_step_index": 1,
                            "protocol_semantic_step_ref": (
                                "protocol_semantic_snapshot@recipe.steps[1]"
                            ),
                        }
                    )
        for name in rows[0]:
            self.arrays[f"{table}/{name}"] = np.asarray([row[name] for row in rows])

    def array(self, path: str) -> np.ndarray:
        return self.arrays[path]

    def require_verified_arrays(self, paths: tuple[str, ...]) -> None:
        assert tuple(paths) == EPOCH_BEHAVIOR_ARRAYS
        assert set(paths).issubset(self.arrays)


def _projection(handle: _Handle | None = None) -> Any:
    return SimpleNamespace(
        epoch_behavior=handle or _Handle(),
        provenance={"bundle_manifest_sha256": "1" * 64},
    )


def test_epoch_behavior_renders_only_persisted_summaries_and_bins() -> None:
    output = build_exact_epoch_behavior_output(_Marimo, go, _projection())

    assert len(output) == 5
    summary, bout, hist, ibi = output[1:]
    np.testing.assert_array_equal(summary.data[0].y, [5.0, 8.0, 6.0])
    np.testing.assert_array_equal(bout.data[1].y, [120.0, 120.0, 120.0])
    np.testing.assert_array_equal(hist.data[0].x, [0.5, 1.5])
    np.testing.assert_array_equal(hist.data[0].y, [50.0, 50.0])
    np.testing.assert_array_equal(ibi.data[0].x, [0.5])
    display = summary.layout.meta["epoch_behavior_display"]
    assert display["recipe_id"] == EPOCH_BEHAVIOR_DISPLAY_RECIPE
    assert display["source_speed_level"] == "filtered"
    assert display["rate_denominator"] == "valid_tracked_duration_s"
    assert display["histogram_bins"]["bout_duration_s"] == [0.0, 1.0, 2.0]
    assert display["viewer_epoch_recomputation"] == "prohibited"
    assert display["viewer_rebinning"] == "prohibited"
    assert display["scientific_recomputation"] is False


def test_epoch_behavior_rejects_recomputed_or_stale_rate() -> None:
    handle = _Handle()
    handle.arrays["per_epoch_fish/bout_rate_per_min"][1] = 121.0

    with pytest.raises(ValueError, match="rates or coverage"):
        _epoch_behavior_values(_projection(handle))


def test_epoch_behavior_rejects_histogram_count_nonconservation() -> None:
    handle = _Handle()
    handle.arrays["per_epoch_bout_histograms/hist_count"][0] = 0

    with pytest.raises(ValueError, match="bins or support"):
        _epoch_behavior_values(_projection(handle))
