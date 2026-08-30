from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from apps.marimo.components import chaser_exact_successors as facade
from apps.marimo.components.analysis_catalog import (
    CHASER_CANDIDATE_PROVIDER,
    CHASER_EXACT_SUCCESSOR_PROVIDER,
)
from apps.marimo.components.chaser_exact.distance_traces import (
    _trace_display_projection,
)
from apps.marimo.components.chaser_exact.array_requirements import (
    DISTANCE_DISTRIBUTION_ARRAYS,
    RADIAL_NEAR_FIELD_ARRAYS,
    SAME_QUADRANT_ARRAYS,
)
from apps.marimo.components.chaser_exact.controller_trials import (
    build_exact_controller_trials_output,
)
from apps.marimo.components.chaser_exact.bout_response import (
    build_exact_bout_response_output,
)
from apps.marimo.components.chaser_exact.escape_freeze import (
    build_exact_escape_freeze_output,
)
from apps.marimo.components.chaser_exact.provider import (
    ANALYSIS_IDS,
    EXACT_CHASER_PROVIDER_ADAPTER,
    ExactChaserAnalysisUnavailableError,
    ExactChaserStaleSelectionError,
    ExactChaserUnknownAnalysisError,
    load_exact_chaser_successor_projection,
)
from apps.marimo.components.chaser_exact.projection import (
    ExactChaserSelectionIdentity,
    RelativeFrameProjection,
    _RADIAL_ARRAYS_BY_ANALYSIS,
    load_exact_chaser_projection,
)
from apps.marimo.components.chaser_exact.trajectory_overlays import (
    _trajectory_display_indices,
)
from apps.marimo.components.registry import (
    CHASER_EXACT_SUCCESSOR_RENDERER,
    InteractiveSpecOption,
)


def _option(
    archive: Path,
    *,
    manifest_sha256: str = "a" * 64,
    run_name: str = "paired-spatial-v1",
    trace_max_points: int = 6_000,
    controller_manifest_sha256: str = "c" * 64,
    bout_manifest_sha256: str = "1" * 64,
    escape_manifest_sha256: str = "9" * 64,
) -> InteractiveSpecOption:
    run_path = f"analysis/chaser_spatial_occupancy_runs/{run_name}"
    spec = {
        "schema_id": "palette.chaser_exact_successor_explorer_spec",
        "schema_version": 8,
        "renderer": CHASER_EXACT_SUCCESSOR_RENDERER,
        "bundle_status": "exact_selector_ineligible",
        "bundle_manifest_sha256": manifest_sha256,
        "analysis_bindings": {
            "body_bearing": {
                "source_relative_frame": {
                    "run_path": "analysis/chaser_relative_frame_runs/keypoint-v1",
                    "manifest_sha256": "e" * 64,
                },
                "array_paths": [
                    "body/body_bearing_deg",
                    "body/body_bearing_valid",
                ],
                "body_axis_authority": "accepted_keypoint_body_extension",
                "position_substitution": "prohibited",
            },
            "body_heading": {
                "source_relative_frame": {
                    "run_path": "analysis/chaser_relative_frame_runs/keypoint-v1",
                    "manifest_sha256": "e" * 64,
                },
                "array_paths": [
                    "body/body_source_row_id",
                    "body/body_source_row_valid",
                    "body/body_heading_deg",
                    "body/body_heading_valid",
                    "body/body_heading_reason_code",
                ],
                "body_axis_authority": "accepted_keypoint_body_extension",
                "frame_collapse_policy": (
                    "exact_equality_across_flattened_chaser_rows_then_one_row_per_acquisition_frame"
                ),
                "position_substitution": "prohibited",
                "motion_heading_fallback": "prohibited",
            },
            "controller_trials": {
                "run_path": "analysis/controller_chase_trial_runs/controller-v1",
                "manifest_sha256": controller_manifest_sha256,
                "scientific_payload_sha256": "d" * 64,
                "source_relative_frame": {
                    "run_path": "analysis/chaser_relative_frame_runs/keypoint-v1",
                    "manifest_sha256": "e" * 64,
                },
                "semantic_selection": {
                    "run_path": (
                        "analysis/protocol_semantic_chaser_selection_runs/semantic-v1"
                    ),
                    "manifest_sha256": "f" * 64,
                },
            },
            "generalized_bout_response": {
                "run_path": ("analysis/generalized_chaser_bout_response_runs/bout-v1"),
                "manifest_sha256": bout_manifest_sha256,
                "scientific_payload_sha256": "2" * 64,
                "source_relative_frame": {
                    "run_path": "analysis/chaser_relative_frame_runs/keypoint-v1",
                    "manifest_sha256": "e" * 64,
                },
                "source_motion": {
                    "run_path": ("analysis/track_kinematics_runs/provider/motion-v1"),
                    "manifest_sha256": "3" * 64,
                    "relative_frame_projection": {
                        "schema_id": (
                            "palette.provider_motion.relative_frame_projection"
                        ),
                        "schema_version": 1,
                        "join_key": "exact_acquisition_frame_id",
                        "join_policy": (
                            "left_join_missing_provider_rows_invalid_no_interpolation"
                        ),
                        "provider_frame_count": 10,
                        "relative_frame_count": 10,
                        "matched_relative_frame_count": 10,
                        "missing_relative_frame_count": 0,
                        "provider_only_frame_count": 0,
                        "provider_frame_ids_sha256": "4" * 64,
                        "relative_frame_ids_sha256": "5" * 64,
                        "provider_row_index_by_relative_frame_sha256": "6" * 64,
                        "provider_frame_present_sha256": "7" * 64,
                        "fallback": "prohibited",
                    },
                },
                "source_swim_bouts": {
                    "run_path": "analysis/swim_bout_runs/bouts-v1",
                    "lineage_sha256": "8" * 64,
                    "signal_id": 4,
                    "signal_level": "speed_exponential",
                },
                "semantic_selection_manifest_sha256": "f" * 64,
                "controller_trial_payload_sha256": "d" * 64,
                "body_extension_present": True,
            },
            "escape_freeze": {
                "run_path": "analysis/chaser_escape_freeze_runs/escape-v1",
                "manifest_sha256": escape_manifest_sha256,
                "scientific_payload_sha256": "a" * 64,
                "source_motion": {
                    "run_path": "analysis/track_kinematics_runs/provider/motion-v1",
                    "manifest_sha256": "3" * 64,
                    "relative_frame_projection": {
                        "schema_id": (
                            "palette.provider_motion.relative_frame_projection"
                        ),
                        "schema_version": 1,
                        "join_key": "exact_acquisition_frame_id",
                        "join_policy": (
                            "left_join_missing_provider_rows_invalid_no_interpolation"
                        ),
                        "provider_frame_count": 10,
                        "relative_frame_count": 10,
                        "matched_relative_frame_count": 10,
                        "missing_relative_frame_count": 0,
                        "provider_only_frame_count": 0,
                        "provider_frame_ids_sha256": "4" * 64,
                        "relative_frame_ids_sha256": "5" * 64,
                        "provider_row_index_by_relative_frame_sha256": "6" * 64,
                        "provider_frame_present_sha256": "7" * 64,
                        "fallback": "prohibited",
                    },
                    "speed_level": "filtered",
                },
                "controller_trial_payload_sha256": "d" * 64,
                "bout_response_payload_sha256": "2" * 64,
                "classifier_parameters": {
                    "escape_speed_threshold_mm_s": 20.0,
                    "high_turn_threshold_deg": 45.0,
                    "freeze_speed_threshold_mm_s": 2.0,
                    "freeze_window_s": 1.0,
                    "freeze_fraction_threshold": 0.8,
                    "minimum_freeze_valid_fraction": 0.5,
                    "threshold_sweep_mm_s": [10.0, 20.0, 30.0],
                },
                "n_trials": 2,
                "n_events": 1,
                "n_sweep_rows": 6,
            },
            "gaze_tracking": {
                "run_path": "analysis/chaser_gaze_tracking_runs/gaze-v1",
                "manifest_sha256": "b" * 64,
                "scientific_payload_sha256": "c" * 64,
                "source_relative_frame": {
                    "run_path": "analysis/chaser_relative_frame_runs/keypoint-v1",
                    "manifest_sha256": "e" * 64,
                },
                "source_eye_orientation": {
                    "run_path": "analysis/eye_angle_runs/eye-v1",
                    "manifest_sha256": "1" * 64,
                    "convention_receipt_sha256": "2" * 64,
                    "channel_policy": "smoothed:left,right:vergence",
                },
                "source_radial_geometry": {
                    "run_path": (
                        "analysis/chaser_radial_near_field_runs/keypoint-radial"
                    ),
                    "manifest_sha256": "3" * 64,
                    "scientific_payload_sha256": "4" * 64,
                    "arena_geometry_and_scale": {"authority_sha256": "5" * 64},
                },
                "semantic_selection_manifest_sha256": "f" * 64,
                "parameters": {
                    "lock_threshold_deg": 10.0,
                    "minimum_lock_duration_s": 0.1,
                    "maximum_tracking_distance_mm": 50.0,
                    "accessible_quantiles": [0.025, 0.975],
                    "empirical_eye_range_deg": [[-40.0, 40.0], [-40.0, 40.0]],
                    "virtual_rotations_deg": [60.0, 120.0, 180.0, 240.0, 300.0],
                    "minimum_virtual_separation_mm": 8.0,
                    "maximum_virtual_collision_fraction": 0.05,
                    "maximum_dynamic_lag_s": 0.5,
                    "minimum_regression_samples": 30,
                    "minimum_regression_span_deg": 5.0,
                },
            },
        },
        "source_paths": {
            "position_providers": [
                {
                    "provider_role": "keypoint",
                    "relative_frame": {
                        "run_path": ("analysis/chaser_relative_frame_runs/keypoint-v1"),
                        "manifest_sha256": "e" * 64,
                    },
                    "radial_near_field": {
                        "run_path": (
                            "analysis/chaser_radial_near_field_runs/keypoint-radial"
                        ),
                        "manifest_sha256": "3" * 64,
                    },
                },
                {
                    "provider_role": "detection",
                    "relative_frame": {
                        "run_path": (
                            "analysis/chaser_relative_frame_runs/detection-v1"
                        ),
                        "manifest_sha256": "0" * 64,
                    },
                },
            ]
        },
        "display_parameters": {
            "distance_traces": {
                "algorithm": (
                    "source_order_bucket_first_last_min_max_missing_break_v1"
                ),
                "max_points_per_series": trace_max_points,
            },
            "scientific_recomputation": False,
            "interpolation": "prohibited",
        },
    }
    return InteractiveSpecOption(
        zarr_path=archive,
        artifact_path=f"{run_path}/interactive",
        run_path=run_path,
        artifact_name="chaser_exact_successor_bundle",
        renderer=CHASER_EXACT_SUCCESSOR_RENDERER,
        schema_id=str(spec["schema_id"]),
        title="Exact chaser successors",
        run_name=run_name,
        label=run_name,
        is_supported=True,
        attrs={},
        spec=spec,
    )


def test_analysis_listing_is_metadata_only(tmp_path: Path, monkeypatch) -> None:
    option = _option(tmp_path / "recording.zarr")

    def forbidden_loader(*args, **kwargs):
        raise AssertionError("analysis discovery must not open scientific arrays")

    monkeypatch.setattr(
        "apps.marimo.components.chaser_exact.provider.load_exact_chaser_projection",
        forbidden_loader,
    )
    assert (
        EXACT_CHASER_PROVIDER_ADAPTER.available_analysis_ids(option.zarr_path, option)
        == ANALYSIS_IDS
    )


def test_controller_trial_analysis_is_hidden_without_one_exact_binding(
    tmp_path: Path,
) -> None:
    option = _option(tmp_path / "recording.zarr")
    spec = dict(option.spec)
    spec["analysis_bindings"] = {}

    available = EXACT_CHASER_PROVIDER_ADAPTER.available_analysis_ids(
        option.zarr_path,
        replace(option, spec=spec),
    )

    assert "controller_trials" not in available
    assert "body_bearing_polar" not in available
    assert "body_bearing_distance" not in available
    assert "fish_heading" not in available
    assert "spatial_occupancy" in available


def test_escape_freeze_analysis_is_hidden_without_its_exact_binding(
    tmp_path: Path,
) -> None:
    option = _option(tmp_path / "recording.zarr")
    spec = dict(option.spec)
    bindings = dict(spec["analysis_bindings"])
    bindings.pop("escape_freeze")
    spec["analysis_bindings"] = bindings

    available = EXACT_CHASER_PROVIDER_ADAPTER.available_analysis_ids(
        option.zarr_path,
        replace(option, spec=spec),
    )

    assert "controller_trials" in available
    assert "generalized_bout_response" in available
    assert "escape_freeze" not in available


def test_provider_routes_are_closed_and_controls_are_explicit() -> None:
    assert ANALYSIS_IDS == (
        "radial_near_field",
        "distance_distributions",
        "same_quadrant_occupancy",
        "distance_traces",
        "body_bearing_polar",
        "body_bearing_distance",
        "fish_heading",
        "trajectory_overlays",
        "spatial_occupancy",
        "controller_trials",
        "generalized_bout_response",
        "escape_freeze",
        "gaze_tracking",
        "provenance",
    )
    assert EXACT_CHASER_PROVIDER_ADAPTER.requires_projection("distance_traces")
    assert not EXACT_CHASER_PROVIDER_ADAPTER.requires_projection("provenance")
    assert EXACT_CHASER_PROVIDER_ADAPTER.build_controls("radial_near_field") is None
    with pytest.raises(ExactChaserUnknownAnalysisError, match="Unsupported"):
        EXACT_CHASER_PROVIDER_ADAPTER.requires_projection("distance-ish")


def test_new_radial_routes_have_closed_receipt_target_rosters() -> None:
    assert _RADIAL_ARRAYS_BY_ANALYSIS == {
        "radial_near_field": RADIAL_NEAR_FIELD_ARRAYS,
        "distance_distributions": DISTANCE_DISTRIBUTION_ARRAYS,
        "same_quadrant_occupancy": SAME_QUADRANT_ARRAYS,
    }
    assert "cdf_fraction_at_or_below" in DISTANCE_DISTRIBUTION_ARRAYS
    assert "metric_same_quadrant_fraction_candidate" in SAME_QUADRANT_ARRAYS


def test_controller_trial_catalog_entry_belongs_to_exact_successors() -> None:
    exact_ids = tuple(
        item.analysis_id for item in CHASER_EXACT_SUCCESSOR_PROVIDER.analyses
    )
    candidate_ids = tuple(
        item.analysis_id for item in CHASER_CANDIDATE_PROVIDER.analyses
    )

    assert exact_ids == ANALYSIS_IDS
    assert "controller_trials" not in candidate_ids


def test_exact_source_defaults_only_when_discovery_is_unambiguous() -> None:
    assert EXACT_CHASER_PROVIDER_ADAPTER.initial_source_label(()) is None
    assert EXACT_CHASER_PROVIDER_ADAPTER.initial_source_label(("only",)) == "only"
    assert (
        EXACT_CHASER_PROVIDER_ADAPTER.initial_source_label(("newer", "older")) is None
    )


def test_only_selected_analysis_requests_relative_arrays(
    tmp_path: Path, monkeypatch
) -> None:
    option = _option(tmp_path / "recording.zarr")
    observed: list[
        tuple[str, bool, bool, bool, bool, bool, bool, bool, bool, bool]
    ] = []

    def fake_loader(
        zarr_path,
        selected_option,
        *,
        selection_identity,
        load_relative,
        load_relative_arrays,
        load_chaser_appearance,
        load_keypoint_body_bearing,
        load_keypoint_body_heading,
        load_controller_trials,
        load_generalized_bout_response,
        load_escape_freeze,
        load_gaze_tracking,
    ):
        assert zarr_path == option.zarr_path
        assert selected_option is option
        observed.append(
            (
                selection_identity.analysis_id,
                load_relative,
                load_relative_arrays,
                load_chaser_appearance,
                load_keypoint_body_bearing,
                load_keypoint_body_heading,
                load_controller_trials,
                load_generalized_bout_response,
                load_escape_freeze,
                load_gaze_tracking,
            )
        )
        return selection_identity

    monkeypatch.setattr(
        "apps.marimo.components.chaser_exact.provider.load_exact_chaser_projection",
        fake_loader,
    )

    radial = EXACT_CHASER_PROVIDER_ADAPTER.load_projection(
        option.zarr_path, option, analysis_id="radial_near_field"
    )
    distributions = EXACT_CHASER_PROVIDER_ADAPTER.load_projection(
        option.zarr_path, option, analysis_id="distance_distributions"
    )
    same_quadrant = EXACT_CHASER_PROVIDER_ADAPTER.load_projection(
        option.zarr_path, option, analysis_id="same_quadrant_occupancy"
    )
    spatial = EXACT_CHASER_PROVIDER_ADAPTER.load_projection(
        option.zarr_path, option, analysis_id="spatial_occupancy"
    )
    distance = EXACT_CHASER_PROVIDER_ADAPTER.load_projection(
        option.zarr_path, option, analysis_id="distance_traces"
    )
    body_bearing = EXACT_CHASER_PROVIDER_ADAPTER.load_projection(
        option.zarr_path, option, analysis_id="body_bearing_polar"
    )
    body_bearing_distance = EXACT_CHASER_PROVIDER_ADAPTER.load_projection(
        option.zarr_path, option, analysis_id="body_bearing_distance"
    )
    fish_heading = EXACT_CHASER_PROVIDER_ADAPTER.load_projection(
        option.zarr_path, option, analysis_id="fish_heading"
    )
    controller = EXACT_CHASER_PROVIDER_ADAPTER.load_projection(
        option.zarr_path, option, analysis_id="controller_trials"
    )
    bout_response = EXACT_CHASER_PROVIDER_ADAPTER.load_projection(
        option.zarr_path, option, analysis_id="generalized_bout_response"
    )
    escape_freeze = EXACT_CHASER_PROVIDER_ADAPTER.load_projection(
        option.zarr_path, option, analysis_id="escape_freeze"
    )
    gaze = EXACT_CHASER_PROVIDER_ADAPTER.load_projection(
        option.zarr_path, option, analysis_id="gaze_tracking"
    )

    assert radial.analysis_id == "radial_near_field"
    assert distributions.analysis_id == "distance_distributions"
    assert same_quadrant.analysis_id == "same_quadrant_occupancy"
    assert spatial.analysis_id == "spatial_occupancy"
    assert distance.analysis_id == "distance_traces"
    assert body_bearing.analysis_id == "body_bearing_polar"
    assert body_bearing_distance.analysis_id == "body_bearing_distance"
    assert fish_heading.analysis_id == "fish_heading"
    assert controller.analysis_id == "controller_trials"
    assert bout_response.analysis_id == "generalized_bout_response"
    assert escape_freeze.analysis_id == "escape_freeze"
    assert gaze.analysis_id == "gaze_tracking"
    assert observed == [
        (
            "radial_near_field",
            False,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ),
        (
            "distance_distributions",
            False,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ),
        (
            "same_quadrant_occupancy",
            False,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ),
        (
            "spatial_occupancy",
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
        ),
        (
            "distance_traces",
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ),
        (
            "body_bearing_polar",
            True,
            True,
            False,
            True,
            False,
            False,
            False,
            False,
            False,
        ),
        (
            "body_bearing_distance",
            True,
            True,
            False,
            True,
            False,
            False,
            False,
            False,
            False,
        ),
        (
            "fish_heading",
            True,
            True,
            False,
            False,
            True,
            False,
            False,
            False,
            False,
        ),
        (
            "controller_trials",
            True,
            True,
            False,
            False,
            False,
            True,
            False,
            False,
            False,
        ),
        (
            "generalized_bout_response",
            True,
            False,
            False,
            False,
            False,
            True,
            True,
            False,
            False,
        ),
        (
            "escape_freeze",
            True,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            False,
        ),
        (
            "gaze_tracking",
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
        ),
    ]


def test_gaze_projection_routes_keypoint_radial_handle(
    tmp_path: Path, monkeypatch
) -> None:
    archive = (tmp_path / "recording.zarr").resolve()
    option = _option(archive)
    records = (
        {
            "provider_id": "keypoint-provider",
            "provider_digest": "1" * 64,
            "relative_frame": {
                "run_path": "analysis/chaser_relative_frame_runs/keypoint-v1",
                "manifest_sha256": "e" * 64,
            },
            "radial_near_field": {
                "run_path": ("analysis/chaser_radial_near_field_runs/keypoint-radial"),
                "manifest_sha256": "3" * 64,
            },
        },
        {
            "provider_id": "detection-provider",
            "provider_digest": "2" * 64,
            "relative_frame": {
                "run_path": "analysis/chaser_relative_frame_runs/detection-v1",
                "manifest_sha256": "0" * 64,
            },
            "radial_near_field": {
                "run_path": ("analysis/chaser_radial_near_field_runs/detection-radial"),
                "manifest_sha256": "4" * 64,
            },
        },
    )
    spatial = SimpleNamespace(
        run_path=option.run_path,
        manifest_sha256="a" * 64,
        recording_id="recording-1",
        scientific_manifest={
            "epoch_records": [
                {
                    "analysis_role": "chaser_training",
                    "start_frame": 0,
                    "end_frame_exclusive": 1,
                }
            ]
        },
    )
    keypoint_radial = SimpleNamespace(
        run_path=records[0]["radial_near_field"]["run_path"]
    )
    detection_radial = SimpleNamespace(
        run_path=records[1]["radial_near_field"]["run_path"]
    )

    def fake_composable(_archive, *, successor_kind, run_name, **_kwargs):
        if successor_kind == "chaser_spatial_occupancy":
            return spatial
        return {
            "keypoint-radial": keypoint_radial,
            "detection-radial": detection_radial,
        }[run_name]

    arrays = {
        "acquisition_frame_id": np.asarray([0]),
        "timestamp_ns": np.asarray([0]),
        "timestamp_valid": np.asarray([True]),
        "selection_member": np.asarray([True]),
        "chaser_identity_code": np.asarray([1]),
        "chaser_behavior_role_code": np.asarray([1]),
        "chaser_occurrence_member": np.asarray([True]),
        "chaser_position_xy_px": np.asarray([[0.0, 0.0]]),
        "chaser_position_valid": np.asarray([True]),
    }

    def fake_relative(*, run_path, expected_manifest_sha256, **_kwargs):
        record = records[0] if "keypoint" in run_path else records[1]
        return RelativeFrameProjection(
            run_path=run_path,
            run_name=run_path.rsplit("/", 1)[-1],
            recording_id="recording-1",
            manifest_sha256=expected_manifest_sha256,
            n_frames=1,
            n_chasers=1,
            source_authorities={
                "fish_position": {
                    "provider_id": record["provider_id"],
                    "provider_digest": record["provider_digest"],
                }
            },
            arrays=arrays,
        )

    observed = {}

    def fake_gaze(
        _archive,
        _option,
        *,
        spatial,
        radial,
        expected_relative_binding,
        relative,
        direct_validation_receipt,
        required_array_names,
    ):
        observed.update(
            spatial=spatial,
            radial=radial,
            relative_binding=expected_relative_binding,
            relative=relative,
            receipt=direct_validation_receipt,
            arrays=required_array_names,
        )
        return SimpleNamespace(run_path="analysis/chaser_gaze_tracking_runs/gaze-v1")

    monkeypatch.setattr(
        "apps.marimo.components.chaser_exact.projection.load_composable_chaser_successor_source_handle",
        fake_composable,
    )
    monkeypatch.setattr(
        "apps.marimo.components.chaser_exact.projection._source_records",
        lambda _spatial: records,
    )
    monkeypatch.setattr(
        "apps.marimo.components.chaser_exact.projection._verify_bundle_children",
        lambda *_args: (("keypoint-provider", "detection-provider"), ({}, {})),
    )
    monkeypatch.setattr(
        "apps.marimo.components.chaser_exact.projection._load_targeted_relative",
        fake_relative,
    )
    monkeypatch.setattr(
        "apps.marimo.components.chaser_exact.projection.load_exact_gaze_tracking",
        fake_gaze,
    )
    monkeypatch.setattr(
        "apps.marimo.components.chaser_exact.projection.build_projection_provenance",
        lambda **_kwargs: {},
    )
    identity = ExactChaserSelectionIdentity(
        archive_path=str(archive),
        run_path=option.run_path,
        bundle_manifest_sha256="a" * 64,
        renderer=option.renderer,
        schema_id=option.schema_id,
        analysis_id="gaze_tracking",
        display_parameter_version="exact-gaze-tracking-display-v1",
        display_parameters_sha256="5" * 64,
        analysis_bindings_sha256="6" * 64,
        projection_receipt_path=None,
        projection_receipt_sha256=None,
        verification_mode="deep_audit",
    )

    result = load_exact_chaser_projection(
        archive,
        option,
        selection_identity=identity,
        load_relative=True,
        load_relative_arrays=False,
        load_gaze_tracking=True,
    )

    assert result.gaze_tracking.run_path.endswith("/gaze-v1")
    assert observed["spatial"] is spatial
    assert observed["radial"] is keypoint_radial
    assert observed["relative_binding"] == records[0]["relative_frame"]
    assert observed["relative"] is result.relatives[0]
    assert observed["receipt"] is None
    assert observed["arrays"] is None


def test_selection_identity_binds_display_parameters_and_exact_source(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "recording.zarr"
    option = _option(archive)
    identity = EXACT_CHASER_PROVIDER_ADAPTER.selection_identity(
        archive, option, analysis_id="distance_traces"
    )

    assert identity.archive_path == str(archive.resolve())
    assert identity.run_path == option.run_path
    assert identity.bundle_manifest_sha256 == "a" * 64
    assert identity.renderer == CHASER_EXACT_SUCCESSOR_RENDERER

    changed_display = _option(archive, trace_max_points=3_000)
    changed_manifest = _option(archive, manifest_sha256="b" * 64)
    changed_run = _option(archive, run_name="paired-spatial-v2")
    changed_controller = _option(archive, controller_manifest_sha256="9" * 64)
    changed_bout = _option(archive, bout_manifest_sha256="0" * 64)
    changed_escape = _option(archive, escape_manifest_sha256="0" * 64)
    for changed in (
        changed_display,
        changed_manifest,
        changed_run,
        changed_controller,
        changed_bout,
        changed_escape,
    ):
        assert (
            EXACT_CHASER_PROVIDER_ADAPTER.selection_identity(
                archive, changed, analysis_id="distance_traces"
            )
            != identity
        )


def test_stale_projection_cannot_render_under_new_selection(tmp_path: Path) -> None:
    archive = tmp_path / "recording.zarr"
    option = _option(archive)
    identity = EXACT_CHASER_PROVIDER_ADAPTER.selection_identity(
        archive, option, analysis_id="distance_traces"
    )
    projection = SimpleNamespace(
        analysis_id="distance_traces", selection_identity=identity
    )

    EXACT_CHASER_PROVIDER_ADAPTER.require_current_projection(
        projection,
        zarr_path=archive,
        option=option,
        analysis_id="distance_traces",
    )
    with pytest.raises(ExactChaserStaleSelectionError, match="earlier archive"):
        EXACT_CHASER_PROVIDER_ADAPTER.require_current_projection(
            projection,
            zarr_path=archive,
            option=_option(archive, trace_max_points=3_000),
            analysis_id="distance_traces",
        )


def test_shared_provenance_route_is_typed_unavailable(tmp_path: Path) -> None:
    archive = tmp_path / "recording.zarr"
    option = _option(archive)
    identity = EXACT_CHASER_PROVIDER_ADAPTER.selection_identity(
        archive, option, analysis_id="provenance"
    )
    projection = SimpleNamespace(analysis_id="provenance", selection_identity=identity)

    with pytest.raises(ExactChaserAnalysisUnavailableError, match="shared"):
        EXACT_CHASER_PROVIDER_ADAPTER.render(
            None,
            None,
            projection,
            zarr_path=archive,
            option=option,
            analysis_id="provenance",
        )


def test_compatibility_facade_reexports_focused_components() -> None:
    assert (
        facade.load_exact_chaser_successor_projection
        is load_exact_chaser_successor_projection
    )
    assert facade._trace_display_projection is _trace_display_projection
    assert facade._trajectory_display_indices is _trajectory_display_indices
    assert (
        facade.build_exact_controller_trials_output
        is build_exact_controller_trials_output
    )
    assert facade.build_exact_bout_response_output is build_exact_bout_response_output
    assert facade.build_exact_escape_freeze_output is build_exact_escape_freeze_output
    assert facade.EXACT_CHASER_PROVIDER_ADAPTER is EXACT_CHASER_PROVIDER_ADAPTER


def test_palette_explorer_uses_one_exact_provider_load_and_render_boundary() -> None:
    source = Path("apps/marimo/palette_explorer.py").read_text(encoding="utf-8")

    assert "build_exact_distance_traces_output" not in source
    assert "build_exact_radial_near_field_output" not in source
    assert "build_exact_trajectory_overlays_output" not in source
    assert source.count("EXACT_CHASER_PROVIDER_ADAPTER.load_projection(") == 1
    assert source.count("EXACT_CHASER_PROVIDER_ADAPTER.render(") == 1
    assert "EXACT_CHASER_PROVIDER_ADAPTER.initial_source_label(source_labels)" in source
    assert "source_picker.value is not None" in source
    assert "no analysis arrays will load until" in source
