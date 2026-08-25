from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

from fisheye.utils.plot_chaser_detailed_successors import (
    ChaserDetailedPlotError,
    detailed_plot_parameters,
    render_detailed_bundle,
    verify_detailed_plot_inputs,
)


@dataclass
class _Successor:
    successor_kind: str
    scientific_payload_sha256: str
    scientific_manifest: dict[str, Any]
    arrays: dict[str, np.ndarray]
    recording_id: str = "recording-1"
    run_path: str = "analysis/example/run"
    manifest_sha256: str = "f" * 64
    deep_audited: bool = True

    def array(self, name: str) -> np.ndarray:
        return self.arrays[name]


@dataclass
class _Relative:
    recording_id: str = "recording-1"
    run_path: str = "analysis/chaser_relative_frame_runs/relative-v3"
    manifest_sha256: str = "d" * 64
    payload_digest: str = "e" * 64
    verification_mode: str = "deep_array_content_hash"
    fish_provider: str = "keypoint_anatomical_triad_mean.v1"
    n_frames: int = 6
    n_chasers: int = 2
    n_rows: int = 12

    def __post_init__(self) -> None:
        self.source_authorities = {
            "fish_position": {"provider_id": self.fish_provider}
        }
        self.manifest = {
            "coordinate_policy": {"policy_id": "camera-v1"},
            "scale_policy": {"policy_id": "scale-v1"},
        }
        frame = np.arange(self.n_frames, dtype=np.int64)
        identities = np.tile(np.asarray([1, 2]), self.n_frames)
        self.values = {
            "acquisition_frame_id": np.repeat(frame, 2),
            "relative_distance_physical": np.column_stack(
                (10.0 + frame, 20.0 + frame)
            ).reshape(-1),
            "relative_physical_valid": np.ones(self.n_rows, dtype=bool),
            "timestamp_ns": np.repeat(frame * 10_000_000, 2),
            "timestamp_valid": np.ones(self.n_rows, dtype=bool),
            "selection_member": np.ones(self.n_rows, dtype=bool),
            "chaser_identity_code": identities,
            "chaser_occurrence_member": np.ones(self.n_rows, dtype=bool),
            "chaser_behavior_role_code": identities.copy(),
            "chaser_position_xy_px": np.column_stack(
                (identities.astype(float), identities.astype(float) + 1.0)
            ),
            "chaser_position_valid": np.ones(self.n_rows, dtype=bool),
        }

    def base_frame_chaser(self, name: str) -> np.ndarray:
        return self.values[name].reshape(self.n_frames, self.n_chasers)

    def base_array(self, name: str) -> np.ndarray:
        return self.values[name]


def _inputs() -> tuple[
    _Successor,
    _Successor,
    _Successor,
    _Relative,
    _Relative,
    _Successor,
    _Successor,
]:
    controller_digest = "a" * 64
    bout_digest = "b" * 64
    relative = _Relative()
    detection_relative = _Relative(
        run_path="analysis/chaser_relative_frame_runs/detection-relative-v3",
        manifest_sha256="9" * 64,
        payload_digest="8" * 64,
        fish_provider="detection_bbox_centroid.v1",
    )
    active = np.zeros(relative.n_rows, dtype=bool)
    active[[2, 4, 6, 8]] = True
    controller = _Successor(
        successor_kind="controller_chase_trials",
        scientific_payload_sha256=controller_digest,
        scientific_manifest={
            "source_relative_frame": {
                "run_path": relative.run_path,
                "manifest_sha256": relative.manifest_sha256,
            }
        },
        arrays={
            "start_source_frame_row": np.asarray([1, 3], dtype=np.int64),
            "end_source_frame_row_exclusive": np.asarray([3, 5], dtype=np.int64),
            "trial_ordinal": np.asarray([1, 2], dtype=np.int64),
            "logged_trial_id": np.asarray([11, 12], dtype=np.int64),
            "chaser_identity_code": np.asarray([1, 1], dtype=np.int64),
            "trigger_timestamp_ns": np.asarray([10_000_000, 30_000_000], dtype=np.int64),
            "trigger_timestamp_valid": np.asarray([True, True]),
            "logged_active_trial_member": active,
        },
    )
    summary_identity = {
        "summary_role_code": np.asarray([1, 1, 2, 2], dtype=np.int64),
        "summary_chaser_identity_code": np.asarray([1, 1, 2, 2], dtype=np.int64),
        "summary_distance_bin_index": np.asarray([0, 1, 0, 1], dtype=np.int64),
        "summary_distance_bin_start_mm": np.asarray([0, 8, 0, 8], dtype=np.float64),
        "summary_distance_bin_end_mm": np.asarray([8, np.inf, 8, np.inf], dtype=np.float64),
    }
    bout = _Successor(
        successor_kind="generalized_chaser_bout_response",
        scientific_payload_sha256=bout_digest,
        scientific_manifest={
            "sources": {"controller_trial_payload_sha256": controller_digest},
            "identity_registries": {
                "semantic_role": {"1": "chaser_pre", "2": "chaser_training"}
            },
        },
        arrays={
            **summary_identity,
            "summary_bout_rate_per_min": np.asarray([1, 2, 3, 4], dtype=float),
            "summary_median_peak_speed_mm_s": np.asarray([10, 20, 30, 40], dtype=float),
            "summary_median_net_displacement_mm": np.asarray([1, 2, 3, 4], dtype=float),
            "summary_median_duration_s": np.asarray([0.1, 0.2, 0.3, 0.4], dtype=float),
        },
    )
    escape = _Successor(
        successor_kind="chaser_escape_freeze",
        scientific_payload_sha256="c" * 64,
        scientific_manifest={
            "sources": {
                "controller_trial_payload_sha256": controller_digest,
                "bout_response_payload_sha256": bout_digest,
            },
            "identity_registries": {
                "response_class": {"1": "speed_escape", "2": "freeze_candidate"}
            },
        },
        arrays={
            "trial_ordinal": np.asarray([1, 2], dtype=np.int64),
            "trial_logged_id": np.asarray([11, 12], dtype=np.int64),
            "trial_response_class_code": np.asarray([1, 2], dtype=np.int64),
            "trial_escape_event_count": np.asarray([1, 0], dtype=np.int64),
            "trial_escape_event_rate_per_min": np.asarray([3.0, 0.0]),
            "trial_first_escape_latency_s": np.asarray([0.4, np.nan]),
            "trial_trigger_distance_mm": np.asarray([12.0, 14.0]),
            "trial_freeze_low_speed_fraction": np.asarray([0.1, 0.8]),
            "trial_freeze_valid_fraction": np.asarray([1.0, 0.9]),
            "trial_recapture_fraction": np.asarray([1.0, np.nan]),
            "trial_mean_separation_gain_mm": np.asarray([2.0, np.nan]),
            "event_controller_trial_row_id": np.asarray([0], dtype=np.int64),
            "event_latency_from_trigger_s": np.asarray([0.4]),
            "event_peak_speed_mm_s": np.asarray([24.0]),
            "event_distance_at_onset_mm": np.asarray([10.0]),
            "event_recaptured": np.asarray([True]),
            "sweep_speed_threshold_mm_s": np.asarray(
                [10.0, 20.0, 10.0, 20.0]
            ),
        },
    )

    def radial(provider: str, digest: str, source: _Relative) -> _Successor:
        return _Successor(
            successor_kind="chaser_radial_near_field",
            scientific_payload_sha256=digest,
            scientific_manifest={
                "position_provider": {
                    "provider_id": provider,
                    "status": "first_class_explicit_authority",
                },
                "sources": {
                    "relative_frame": {
                        "run_path": source.run_path,
                        "manifest_sha256": source.manifest_sha256,
                    },
                    "protocol_semantic_selection": {
                        "run_path": "analysis/semantic/run",
                        "manifest_sha256": "3" * 64,
                    },
                    "arena_geometry_and_scale": {
                        "selection_record_sha256": "4" * 64,
                        "physical_authority_sha256": "5" * 64,
                    },
                },
                "identity_registries": {
                    "epoch_role": {"1": "chaser_pre"},
                    "behavior_role": {"1": "aggressive"},
                },
                "config": {"near_zone_radius_mm": 5.0},
            },
            arrays={
                "cdf_epoch_role_code": np.asarray([1, 1], dtype=np.int64),
                "cdf_behavior_role_code": np.asarray([1, 1], dtype=np.int64),
                "cdf_chaser_identity_code": np.asarray([1, 1], dtype=np.int64),
                "cdf_threshold_mm": np.asarray([5.0, 10.0]),
                "cdf_fraction_at_or_below": np.asarray([0.1, 0.3]),
            },
        )

    return (
        controller,
        bout,
        escape,
        relative,
        detection_relative,
        radial("keypoint_anatomical_triad_mean.v1", "1" * 64, relative),
        radial("detection_bbox_centroid.v1", "2" * 64, detection_relative),
    )


def test_render_detailed_bundle_writes_eight_figures(tmp_path: Path) -> None:
    inputs = _inputs()
    outputs = render_detailed_bundle(
        *inputs, output_dir=tmp_path, bundle_name="detailed"
    )
    parameters = detailed_plot_parameters(
        inputs[0], inputs[1], inputs[2], inputs[5], inputs[6]
    )

    assert len(outputs) == 8
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs)
    assert parameters["scientific_coordinates"]["bout_distance_bins"][1][
        "end_mm_exclusive"
    ] is None
    assert parameters["scientific_coordinates"]["provider_distance_cdf"][0][
        "cdf_thresholds_mm"
    ] == [5.0, 10.0]
    assert parameters["rendering"]["trial_distance_traces"]["subplot_grid"] == [
        1,
        2,
    ]


def test_detailed_bundle_rejects_duplicate_position_provider() -> None:
    inputs = list(_inputs())
    inputs[-1].scientific_manifest["position_provider"]["provider_id"] = (
        "keypoint_anatomical_triad_mean.v1"
    )

    with pytest.raises(ChaserDetailedPlotError, match="distinct"):
        verify_detailed_plot_inputs(*inputs)


def test_detailed_bundle_rejects_relative_frame_mismatch() -> None:
    inputs = list(_inputs())
    inputs[0].scientific_manifest["source_relative_frame"]["manifest_sha256"] = (
        "0" * 64
    )

    with pytest.raises(ChaserDetailedPlotError, match="relative-frame"):
        verify_detailed_plot_inputs(*inputs)


def test_detailed_bundle_rejects_mismatched_chaser_arrays() -> None:
    inputs = list(_inputs())
    inputs[4].values["chaser_position_xy_px"] = inputs[4].values[
        "chaser_position_xy_px"
    ].copy()
    inputs[4].values["chaser_position_xy_px"][0, 0] += 1.0

    with pytest.raises(ChaserDetailedPlotError, match="chaser/timing evidence"):
        verify_detailed_plot_inputs(*inputs)


def test_detailed_bundle_accepts_frozen_manifest_bindings() -> None:
    inputs = list(_inputs())
    for relative in inputs[3:5]:
        relative.manifest = MappingProxyType(
            {
                key: MappingProxyType(value)
                for key, value in relative.manifest.items()
            }
        )
    for radial in inputs[5:]:
        radial.scientific_manifest["sources"] = MappingProxyType(
            {
                key: MappingProxyType(value)
                for key, value in radial.scientific_manifest["sources"].items()
            }
        )

    verify_detailed_plot_inputs(*inputs)
