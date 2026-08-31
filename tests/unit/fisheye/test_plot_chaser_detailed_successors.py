from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

from fisheye.utils import plot_chaser_detailed_successors as plot_module
from fisheye.utils.plot_chaser_detailed_successors import (
    ChaserDetailedPlotError,
    detailed_plot_parameters,
    render_detailed_bundle,
    verify_detailed_plot_inputs,
)
from fisheye.visualization.chaser_appearance import (
    ChaserAppearance,
    ChaserAppearanceProjection,
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
    verification_mode: str = "deep_audit"
    verified_array_names: tuple[str, ...] = ()
    receipt_digest: str | None = None

    def array(self, name: str) -> np.ndarray:
        return self.arrays[name]

    def require_verified_authority(self) -> None:
        if self.verification_mode not in {
            "deep_audit",
            "receipt_bound_targeted_array_rehash_v1",
        }:
            raise ValueError("unsupported verification mode")

    def require_verified_arrays(self, names: tuple[str, ...]) -> None:
        missing = set(names).difference(self.arrays)
        if missing:
            raise ValueError(f"missing arrays: {sorted(missing)!r}")


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
        is_keypoint = self.fish_provider.startswith("keypoint_")
        self.source_authorities = {
            "fish_position": {"provider_id": self.fish_provider},
            "body_frame": (
                {"provider_id": "accepted_keypoint_body_extension.v1"}
                if is_keypoint
                else None
            ),
        }
        self.manifest = {
            "coordinate_policy": {"policy_id": "camera-v1"},
            "scale_policy": {"policy_id": "scale-v1"},
        }
        frame = np.arange(self.n_frames, dtype=np.int64)
        identities = np.tile(np.asarray([1, 2]), self.n_frames)
        fish_xy = np.column_stack(
            (100.0 + frame.astype(float), 200.0 + frame.astype(float))
        )
        self.values = {
            "acquisition_frame_id": np.repeat(frame, 2),
            "relative_distance_physical": np.column_stack(
                (10.0 + frame, 20.0 + frame)
            ).reshape(-1),
            "relative_physical_valid": np.ones(self.n_rows, dtype=bool),
            "timestamp_ns": np.repeat(frame * 10_000_000, 2),
            "timestamp_valid": np.ones(self.n_rows, dtype=bool),
            "selection_member": np.ones(self.n_rows, dtype=bool),
            "fish_position_xy_px": np.repeat(fish_xy, self.n_chasers, axis=0),
            "fish_position_valid": np.ones(self.n_rows, dtype=bool),
            "chaser_identity_code": identities,
            "chaser_occurrence_member": np.ones(self.n_rows, dtype=bool),
            "chaser_behavior_role_code": identities.copy(),
            "chaser_position_xy_px": np.column_stack(
                (identities.astype(float), identities.astype(float) + 1.0)
            ),
            "chaser_position_valid": np.ones(self.n_rows, dtype=bool),
        }
        bearing = np.linspace(-150.0, 150.0, self.n_frames, dtype=np.float32)
        self.body_values = (
            {
                "body_bearing_deg": np.column_stack((bearing, -bearing)).reshape(-1),
                "body_bearing_valid": np.ones(self.n_rows, dtype=bool),
            }
            if is_keypoint
            else {}
        )

    def base_frame_chaser(self, name: str) -> np.ndarray:
        values = self.values[name]
        return values.reshape(
            (self.n_frames, self.n_chasers) + values.shape[1:]
        )

    def base_array(self, name: str) -> np.ndarray:
        return self.values[name]

    def body_frame_chaser(self, name: str) -> np.ndarray:
        values = self.body_values[name]
        return values.reshape((self.n_frames, self.n_chasers) + values.shape[1:])


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
                "epoch_records": [
                    {
                        "window_id": 0,
                        "analysis_role": "chaser_pre",
                        "start_frame": 0,
                        "end_frame_exclusive": 2,
                    },
                    {
                        "window_id": 1,
                        "analysis_role": "chaser_training",
                        "start_frame": 2,
                        "end_frame_exclusive": 4,
                    },
                    {
                        "window_id": 2,
                        "analysis_role": "chaser_post",
                        "start_frame": 4,
                        "end_frame_exclusive": 6,
                    },
                ],
                "arena": {
                    "center_x_px": 100.0,
                    "center_y_px": 200.0,
                    "radius_px": 50.0,
                    "radius_mm": 40.0,
                    "coordinate_space": (
                        "source_camera_continuous_pixel_xy_top_left_y_down"
                    ),
                },
                "config": {"near_zone_radius_mm": 5.0},
            },
            arrays={
                "cdf_epoch_role_code": np.asarray([1, 1], dtype=np.int64),
                "cdf_behavior_role_code": np.asarray([1, 1], dtype=np.int64),
                "cdf_chaser_identity_code": np.asarray([1, 1], dtype=np.int64),
                "cdf_threshold_mm": np.asarray([5.0, 10.0]),
                "cdf_fraction_at_or_below": np.asarray([0.1, 0.3]),
                "metric_epoch_role_code": np.asarray([1], dtype=np.int64),
                "metric_behavior_role_code": np.asarray([1], dtype=np.int64),
                "metric_chaser_identity_code": np.asarray([1], dtype=np.int64),
                "metric_distance_p25_mm": np.asarray([8.0]),
                "metric_distance_p50_mm": np.asarray([10.0]),
                "metric_distance_p75_mm": np.asarray([12.0]),
                "metric_near_zone_fraction_valid": np.asarray([0.1]),
                "metric_near_zone_dwell_s": np.asarray([1.5]),
                "metric_near_zone_entry_rate_per_min_valid_time": np.asarray([0.5]),
                "metric_valid_distance_frame_count": np.asarray([100], dtype=np.int64),
                "radial_epoch_role_code": np.asarray([1, 1], dtype=np.int64),
                "radial_behavior_role_code": np.asarray([1, 1], dtype=np.int64),
                "radial_chaser_identity_code": np.asarray([1, 1], dtype=np.int64),
                "radial_bin_start_mm": np.asarray([0.0, 5.0]),
                "radial_bin_end_mm": np.asarray([5.0, 10.0]),
                "radial_selection_index_geometric": np.asarray([0.2, -0.1]),
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


def _appearance() -> ChaserAppearanceProjection:
    appearances = []
    for identity_code, chaser_index, role_code, role, marker in (
        (1, 0, 1, "aggressive", "*"),
        (2, 1, 2, "inert", "o"),
    ):
        appearances.append(
            ChaserAppearance(
                identity_code=identity_code,
                chaser_index=chaser_index,
                identity=f"stimulus-v1:chaser_index:{chaser_index}",
                behavior_role_code=role_code,
                behavior_role=role,
                experimental_color_rgba=(0.0, 0.0, 1.0, 1.0),
                experimental_color_hex="#0000ff",
                experimental_color_css="rgba(0,0,255,1)",
                plotly_role_symbol="star" if role == "aggressive" else "circle",
                matplotlib_role_marker=marker,
                contrast_outline_hex="#ffffff",
            )
        )
    return ChaserAppearanceProjection(
        recording_id="recording-1",
        source_stimulus_run_path="analysis/stimulus_runs/stimulus-v1",
        source_protocol_sha256="a" * 64,
        occurrence_binding_sha256="b" * 64,
        appearances=tuple(appearances),
        projection_sha256="c" * 64,
    )


def test_render_detailed_bundle_writes_eighteen_files(tmp_path: Path) -> None:
    inputs = _inputs()
    appearance = _appearance()
    outputs = render_detailed_bundle(
        *inputs,
        output_dir=tmp_path,
        bundle_name="detailed",
        chaser_appearance=appearance,
    )
    parameters = detailed_plot_parameters(
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[3],
        inputs[5],
        inputs[6],
        chaser_appearance=appearance,
    )

    assert len(outputs) == 18
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
    assert parameters["rendering"]["provider_epoch_distance_traces"][
        "subplot_grid"
    ] == [4, 2]
    trajectory = parameters["rendering"]["provider_epoch_trajectory_overlays"]
    assert trajectory["chaser_color_source"] == "sealed_protocol_rgba"
    assert trajectory["chaser_role_encoding"] == (
        "independent_bounded_exact_row_marker_shape_and_legend_text"
    )
    assert trajectory["index_or_role_color_fallback"] == "prohibited"
    assert trajectory["appearance_projection"]["projection_sha256"] == "c" * 64
    assert parameters["output_families"][-2:] == [
        "keypoint_body_bearing_distance_point_cloud",
        "keypoint_body_bearing_distance_density",
    ]
    bearing = parameters["scientific_coordinates"]["keypoint_body_bearing_distance"]
    assert bearing["distance_bin_edges_mm"] == [
        0.0,
        5.0,
        10.0,
        15.0,
        20.0,
        25.0,
    ]
    assert bearing["bearing_bin_edges_deg"][0] == -180.0
    assert bearing["bearing_bin_edges_deg"][-1] == 180.0
    assert [row["valid_row_count"] for row in bearing["panel_denominators"]] == [
        6,
        6,
        2,
        2,
        2,
        2,
        2,
        2,
    ]


def test_detailed_bundle_rejects_duplicate_position_provider() -> None:
    inputs = list(_inputs())
    inputs[-1].scientific_manifest["position_provider"]["provider_id"] = (
        "keypoint_anatomical_triad_mean.v1"
    )

    with pytest.raises(ChaserDetailedPlotError, match="distinct"):
        verify_detailed_plot_inputs(*inputs)


def test_detailed_bundle_rejects_mismatched_appearance_projection(
    tmp_path: Path,
) -> None:
    inputs = _inputs()
    appearance = replace(_appearance(), recording_id="recording-2")

    with pytest.raises(ChaserDetailedPlotError, match="another recording"):
        render_detailed_bundle(
            *inputs,
            output_dir=tmp_path,
            bundle_name="detailed",
            chaser_appearance=appearance,
        )


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


def test_detailed_bundle_rejects_mismatched_epoch_evidence() -> None:
    inputs = list(_inputs())
    inputs[-1].scientific_manifest["epoch_records"][1]["end_frame_exclusive"] = 5

    with pytest.raises(ChaserDetailedPlotError, match="epoch_records"):
        verify_detailed_plot_inputs(*inputs)


def test_detailed_bundle_rejects_nonrepeated_fish_position() -> None:
    inputs = list(_inputs())
    inputs[3].values["fish_position_xy_px"] = inputs[3].values[
        "fish_position_xy_px"
    ].copy()
    inputs[3].values["fish_position_xy_px"][1, 0] += 1.0

    with pytest.raises(ChaserDetailedPlotError, match="repeated identically"):
        verify_detailed_plot_inputs(*inputs)


def test_detailed_bundle_rejects_missing_body_frame_authority() -> None:
    inputs = list(_inputs())
    inputs[3].source_authorities["body_frame"] = None

    with pytest.raises(ChaserDetailedPlotError, match="body-frame authority"):
        verify_detailed_plot_inputs(*inputs)


def test_detailed_bundle_rejects_declared_valid_nonfinite_body_bearing() -> None:
    inputs = list(_inputs())
    inputs[3].body_values["body_bearing_deg"] = (
        inputs[3].body_values["body_bearing_deg"].copy()
    )
    inputs[3].body_values["body_bearing_deg"][0] = np.nan

    with pytest.raises(ChaserDetailedPlotError, match="body-bearing values"):
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


def test_main_uses_receipt_bound_successor_array_rosters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs()
    controller, bout, escape = inputs[:3]
    relative_keypoint, relative_detection = inputs[3:5]
    radial_keypoint, radial_detection = inputs[5:]
    chain_by_kind = {
        value.successor_kind: value for value in (controller, bout, escape)
    }
    calls: list[dict[str, Any]] = []

    def load_successor(_archive: Path, **kwargs: Any) -> _Successor:
        calls.append(kwargs)
        if kwargs["successor_kind"] == "chaser_radial_near_field":
            handle = (
                radial_keypoint
                if kwargs["run_name"] == "keypoint-radial-v1"
                else radial_detection
            )
        else:
            handle = chain_by_kind[kwargs["successor_kind"]]
        handle.deep_audited = False
        handle.verification_mode = "receipt_bound_targeted_array_rehash_v1"
        handle.verified_array_names = tuple(sorted(kwargs["required_array_names"]))
        handle.receipt_digest = "7" * 64
        return handle

    def load_relative(_receipt: str, **kwargs: Any) -> _Relative:
        handle = (
            relative_keypoint
            if kwargs["expected_run_name"] == "keypoint-relative-v1"
            else relative_detection
        )
        handle.verification_mode = "receipt_bound_targeted_array_rehash_v1"
        handle.receipt_digest = "8" * 64
        return handle

    def render_stub(*_args: Any, output_dir: Path, bundle_name: str, **_kwargs: Any):
        output = output_dir / f"{bundle_name}_stub.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"png")
        return (output,)

    monkeypatch.setattr(
        plot_module, "load_composable_chaser_successor_source_handle", load_successor
    )
    monkeypatch.setattr(
        plot_module, "load_chaser_relative_frame_targeted_source_handle", load_relative
    )
    monkeypatch.setattr(plot_module, "_load_exact_chaser_appearance", lambda _: _appearance())
    monkeypatch.setattr(plot_module, "render_detailed_bundle", render_stub)
    monkeypatch.setattr(plot_module, "detailed_plot_parameters", lambda *_a, **_k: {})

    output_dir = tmp_path / "plots"
    assert plot_module.main(
        [
            str(tmp_path / "analysis.zarr"),
            "--run-name",
            "successors-v1",
            "--relative-frame-run",
            "keypoint-relative-v1",
            "--detection-relative-frame-run",
            "detection-relative-v1",
            "--keypoint-relative-frame-receipt",
            str(tmp_path / "keypoint-relative.json"),
            "--detection-relative-frame-receipt",
            str(tmp_path / "detection-relative.json"),
            "--controller-validation-receipt",
            str(tmp_path / "controller.json"),
            "--bout-validation-receipt",
            str(tmp_path / "bout.json"),
            "--escape-validation-receipt",
            str(tmp_path / "escape.json"),
            "--keypoint-radial-run",
            "keypoint-radial-v1",
            "--detection-radial-run",
            "detection-radial-v1",
            "--keypoint-radial-validation-receipt",
            str(tmp_path / "keypoint-radial.json"),
            "--detection-radial-validation-receipt",
            str(tmp_path / "detection-radial.json"),
            "--expected-recording-id",
            "recording-1",
            "--output-dir",
            str(output_dir),
            "--bundle-name",
            "detailed-receipt-bound-v6",
        ]
    ) == 0

    assert len(calls) == 5
    for call in calls:
        assert call["deep_audit"] is False
        assert call["required_array_names"] == (
            plot_module.DETAILED_SUCCESSOR_PLOT_ARRAY_NAMES[
                call["successor_kind"]
            ]
        )
    receipt = json.loads(
        (output_dir / "detailed-receipt-bound-v6_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["schema_version"] == 6
    assert receipt["plot_policy"]["source_validation"] == {
        "successors": "receipt_bound_targeted_array_rehash_v1",
        "relative_frames": "receipt_bound_targeted_array_rehash_v1",
    }
