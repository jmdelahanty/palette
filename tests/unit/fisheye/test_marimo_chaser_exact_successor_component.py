from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from apps.marimo.components.analysis_catalog import group_specs_by_provider
from apps.marimo.components.chaser_exact_successors import (
    _verify_bundle_children,
    _trace_display_projection,
    _trajectory_display_indices,
)
from apps.marimo.components.chaser_exact.provenance import freeze, plain
from apps.marimo.components.chaser_exact_bout_response_contract import (
    validate_scientific_manifest,
)
from apps.marimo.components.registry import (
    CHASER_EXACT_SUCCESSOR_RENDERER,
    discover_exact_chaser_successor_options,
)
from fisheye.analysis_workflows.exact_relative_frame_binding import (
    ExactRelativeFrameBindingError,
    MINIMAL_EXACT_CHILD_PROFILE,
    RECEIPT_BOUND_PROFILE,
    require_same_exact_relative_frame_child,
    validate_exact_relative_frame_binding,
)
from fisheye.analysis_workflows.gaze_tracking_successor import (
    GazeTrackingInput,
    prepare_gaze_tracking_successor,
)
from fisheye.analysis_workflows.generalized_bout_response_successor import ROLE_CODES
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.visualization.chaser_spatial_occupancy_display import (
    DEFAULT_DISPLAY_MODE_ID,
    DISPLAY_RECIPE_ID,
)


class _Group(dict[str, Any]):
    def __init__(self, *args: Any, attrs: dict[str, Any] | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.attrs = attrs or {}

    def group_keys(self) -> tuple[str, ...]:
        return tuple(self)


def _relative(path: str, provider_id: str, provider_digest: str) -> _Group:
    body_present = provider_id == "keypoint.v1"
    manifest = {
        "recording_id": "recording-1",
        "selector_eligible": False,
        "selection": "none",
        "dimensions": {"n_frames": 10, "n_chasers": 2, "n_rows": 20},
        "source_authorities": {
            "fish_position": {
                "provider_id": provider_id,
                "provider_digest": provider_digest,
            },
            "body_frame": (
                {
                    "source_authority_id": "accepted-body-frame-source",
                    "source_digest": "c" * 64,
                    "provider_id": "accepted-keypoint-body-frame",
                    "provider_digest": "d" * 64,
                }
                if body_present
                else None
            ),
        },
        "schema_binding": {"body_extension_present": body_present},
        "array_declarations": (
            [
                {
                    "path": "body/body_bearing_deg",
                    "dtype": "<f4",
                    "shape": [20],
                    "content_sha256": "1" * 64,
                },
                {
                    "path": "body/body_bearing_valid",
                    "dtype": "|b1",
                    "shape": [20],
                    "content_sha256": "2" * 64,
                },
                {
                    "path": "body/body_source_row_id",
                    "dtype": "<i8",
                    "shape": [20],
                    "content_sha256": "3" * 64,
                },
                {
                    "path": "body/body_source_row_valid",
                    "dtype": "|b1",
                    "shape": [20],
                    "content_sha256": "4" * 64,
                },
                {
                    "path": "body/body_heading_deg",
                    "dtype": "<f4",
                    "shape": [20],
                    "content_sha256": "5" * 64,
                },
                {
                    "path": "body/body_heading_valid",
                    "dtype": "|b1",
                    "shape": [20],
                    "content_sha256": "6" * 64,
                },
                {
                    "path": "body/body_heading_reason_code",
                    "dtype": "<u2",
                    "shape": [20],
                    "content_sha256": "7" * 64,
                },
            ]
            if body_present
            else []
        ),
    }
    digest = canonical_json_sha256(manifest)
    return _Group(
        attrs={
            "schema_id": "palette.analysis.chaser_relative_frame",
            "schema_version": 1,
            "run_path": path,
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": False,
            "chaser_relative_frame_manifest": manifest,
            "chaser_relative_frame_manifest_sha256": digest,
        }
    )


def _successor(
    path: str,
    kind: str,
    scientific: dict[str, Any],
) -> _Group:
    scientific = dict(scientific)
    scientific["payload_digest"] = canonical_json_sha256(scientific)
    manifest = {
        "successor_kind": kind,
        "run_path": path,
        "recording_id": "recording-1",
        "selector_eligible": False,
        "selection": "none",
        "scientific_manifest": scientific,
        "scientific_payload_sha256": scientific["payload_digest"],
    }
    digest = canonical_json_sha256(manifest)
    return _Group(
        attrs={
            "schema_id": "palette.analysis.composable_chaser_successor.run",
            "schema_version": 1,
            "successor_kind": kind,
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": False,
            "composable_chaser_successor_manifest": manifest,
            "composable_chaser_successor_manifest_sha256": digest,
        }
    )


def _archive() -> _Group:
    providers = (
        ("keypoint", "keypoint.v1", "a" * 64),
        ("detection", "detection.v1", "b" * 64),
    )
    root = _Group()
    semantic = {
        "run_path": ("analysis/protocol_semantic_chaser_selection_runs/semantic-v1"),
        "manifest_sha256": "c" * 64,
    }
    records = []
    for role, provider_id, provider_digest in providers:
        relative_path = f"analysis/chaser_relative_frame_runs/{role}-relative"
        radial_path = f"analysis/chaser_radial_near_field_runs/{role}-radial"
        relative = _relative(relative_path, provider_id, provider_digest)
        relative_binding = {
            "run_path": relative_path,
            "manifest_sha256": relative.attrs["chaser_relative_frame_manifest_sha256"],
        }
        receipt_bound_relative_binding = {
            **relative_binding,
            "validation_receipt_sha256": "f" * 64,
            "verification_mode": "receipt_bound_targeted_array_rehash_v1",
        }
        radial = _successor(
            radial_path,
            "chaser_radial_near_field",
            {
                "position_provider": {
                    "provider_id": provider_id,
                    "provider_digest": provider_digest,
                    "status": "first_class_explicit_authority",
                },
                "sources": {
                    "relative_frame": relative_binding,
                    "arena_geometry_and_scale": {"authority_sha256": "a" * 64},
                },
                "arena": {
                    "center_x_px": 100.0,
                    "center_y_px": 100.0,
                    "radius_px": 200.0,
                    "radius_mm": 20.0,
                    "boundary_role": "reviewed_arena_boundary",
                    "observed_feature": "reviewed_inner_boundary",
                    "coordinate_space": (
                        "source_camera_continuous_pixel_xy_top_left_y_down"
                    ),
                },
            },
        )
        root[relative_path] = relative
        root[radial_path] = radial
        records.append(
            {
                "provider_role": role,
                "provider_id": provider_id,
                "provider_digest": provider_digest,
                "relative_frame": receipt_bound_relative_binding,
                "radial_near_field": {
                    "run_path": radial_path,
                    "manifest_sha256": radial.attrs[
                        "composable_chaser_successor_manifest_sha256"
                    ],
                },
            }
        )
    spatial_name = "paired-spatial-v1"
    spatial_path = f"analysis/chaser_spatial_occupancy_runs/{spatial_name}"
    spatial = _successor(
        spatial_path,
        "chaser_spatial_occupancy",
        {
            "sources": {
                "position_providers": records,
                "protocol_semantic_selection": semantic,
            }
        },
    )
    parent = _Group({spatial_name: spatial})
    root["analysis/chaser_spatial_occupancy_runs"] = parent
    root[spatial_path] = spatial
    controller_name = "controller-v1"
    controller_path = f"analysis/controller_chase_trial_runs/{controller_name}"
    controller = _successor(
        controller_path,
        "controller_chase_trials",
        {
            "scientific_schema": {
                "schema_id": "palette.analysis.controller_chase_trials",
                "schema_version": 1,
                "method_id": "exact_logged_trial_id_active_membership_v1",
            },
            "source_relative_frame": {
                key: records[0]["relative_frame"][key]
                for key in ("run_path", "manifest_sha256")
            },
            "semantic_selection": semantic,
            "dimensions": {
                "n_frames": 10,
                "n_chasers": 2,
                "n_source_rows": 20,
                "n_trials": 2,
            },
            "policy": {
                "fallback": "prohibited_fail_closed",
                "legacy_contiguous_interval_reconstruction": "rejected",
            },
        },
    )
    controller_parent = _Group({controller_name: controller})
    root["analysis/controller_chase_trial_runs"] = controller_parent
    root[controller_path] = controller
    bout_name = "bout-v1"
    bout_path = f"analysis/generalized_chaser_bout_response_runs/{bout_name}"
    controller_payload = controller.attrs["composable_chaser_successor_manifest"][
        "scientific_manifest"
    ]["payload_digest"]
    bout = _successor(
        bout_path,
        "generalized_chaser_bout_response",
        {
            "scientific_schema": {
                "schema_id": "palette.analysis.generalized_chaser_bout_response",
                "schema_version": 1,
                "method_id": (
                    "exact_signal_bout_x_chaser_distance_motion_with_body_extension_v1"
                ),
                "row_unit": "selected_swim_bout_x_chaser",
                "summary_unit": "semantic_role_x_chaser_x_distance_band",
                "body_extension_present": True,
            },
            "sources": {
                "relative_frame": {
                    key: records[0]["relative_frame"][key]
                    for key in ("run_path", "manifest_sha256")
                },
                "motion": {
                    "run_path": ("analysis/track_kinematics_runs/provider/motion-v1"),
                    "manifest_sha256": "1" * 64,
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
                        "provider_frame_ids_sha256": "2" * 64,
                        "relative_frame_ids_sha256": "3" * 64,
                        "provider_row_index_by_relative_frame_sha256": "4" * 64,
                        "provider_frame_present_sha256": "5" * 64,
                        "fallback": "prohibited",
                    },
                },
                "swim_bouts": {
                    "run_path": "analysis/swim_bout_runs/bouts-v1",
                    "lineage_sha256": "6" * 64,
                    "signal_id": 4,
                    "signal_level": "speed_exponential",
                },
                "semantic_selection_manifest_sha256": semantic["manifest_sha256"],
                "controller_trial_payload_sha256": controller_payload,
            },
            "dimensions": {
                "n_frames": 10,
                "n_chasers": 2,
                "n_bouts": 3,
                "n_bout_chaser_rows": 6,
                "n_summary_rows": 30,
            },
            "distance_bin_edges_mm": [0.0, 8.0, 16.0, 30.0, 50.0, None],
            "policy": {
                "bout_signal": "one_explicit_default_signal_only",
                "bout_attachment": "exact_acquisition_frame_identity",
                "trial_attachment": "onset_row_exact_controller_trial_membership",
                "trial_envelope": (
                    "retained_for_visualization_and_censoring_not_event_membership"
                ),
                "rate_denominator": "valid_transition_time_in_distance_band",
                "directed_metrics": (
                    "optional_body_frame_extension_no_motion_heading_fallback"
                ),
                "unattached_bouts": "retained_with_reason_code",
            },
            "identity_registries": {
                "semantic_role": {
                    "1": "chaser_pre",
                    "2": "chaser_training",
                    "3": "chaser_post",
                },
                "attachment_reason": {
                    "0": "valid_or_trial_optional",
                    "1": "frame_unavailable",
                    "2": "outside_semantic_selection",
                    "3": "controller_trial_unavailable_at_onset",
                },
            },
        },
    )
    bout_parent = _Group({bout_name: bout})
    root["analysis/generalized_chaser_bout_response_runs"] = bout_parent
    root[bout_path] = bout
    escape_name = "escape-v1"
    escape_path = f"analysis/chaser_escape_freeze_runs/{escape_name}"
    bout_manifest = bout.attrs["composable_chaser_successor_manifest"]
    bout_scientific = bout_manifest["scientific_manifest"]
    escape = _successor(
        escape_path,
        "chaser_escape_freeze",
        {
            "scientific_schema": {
                "schema_id": "palette.analysis.chaser_escape_freeze",
                "schema_version": 2,
                "method_id": ("exact_trial_speed_escape_optional_high_turn_freeze_v1"),
                "event_unit": "speed_thresholded_exact_swim_bout_x_chaser",
                "trial_unit": "exact_logged_controller_trial",
            },
            "sources": {
                "motion": {
                    **bout_scientific["sources"]["motion"],
                    "speed_level": "filtered",
                },
                "controller_trial_payload_sha256": controller_payload,
                "bout_response_payload_sha256": bout_manifest[
                    "scientific_payload_sha256"
                ],
            },
            "parameters": {
                "escape_speed_threshold_mm_s": 20.0,
                "high_turn_threshold_deg": 45.0,
                "freeze_speed_threshold_mm_s": 2.0,
                "freeze_window_s": 1.0,
                "freeze_fraction_threshold": 0.8,
                "minimum_freeze_valid_fraction": 0.5,
                "threshold_sweep_mm_s": [10.0, 20.0, 30.0],
            },
            "dimensions": {"n_trials": 2, "n_events": 1, "n_sweep_rows": 6},
            "policy": {
                "speed_escape": "bout_peak_speed_greater_equal_threshold",
                "high_turn_tier": (
                    "optional_directed_annotation_separate_from_speed_class"
                ),
                "freeze": ("no_speed_escape_and_low_speed_fraction_with_coverage_gate"),
                "trial_attachment": ("exactly_one_controller_trial_row_at_bout_onset"),
                "event_counts": ("retained_even_when_recapture_trace_unusable"),
                "recapture": (
                    "first_post_event_exact_trial_member_at_or_below_onset_distance"
                ),
                "fallback_trial_segmentation": "prohibited",
                "trial_gaps": (
                    "excluded_from_membership_time_and_event_attachment;"
                    "retained_as_coverage_evidence"
                ),
            },
            "identity_registries": {
                "response_class": {
                    "0": "insufficient_valid_freeze_window",
                    "1": "speed_escape",
                    "2": "freeze_candidate",
                    "3": "other_response",
                },
                "trace_exclusion_reason": {
                    "0": "valid",
                    "1": "no_post_event_valid_distance_in_trial",
                    "2": "event_frame_unavailable",
                },
            },
            "selector_eligible": False,
            "selection": "none",
            "production_authority": False,
            "registry_update": False,
        },
    )
    escape_parent = _Group({escape_name: escape})
    root["analysis/chaser_escape_freeze_runs"] = escape_parent
    root[escape_path] = escape
    return root


def _add_gaze_successor(root: _Group, *, run_name: str = "gaze-v1") -> _Group:
    records = _provider_records(root)
    radial_manifest = root[records[0]["radial_near_field"]["run_path"]].attrs[
        "composable_chaser_successor_manifest"
    ]
    n_frames, n_chasers = 10, 2
    frame_bearing = np.linspace(-30.0, 30.0, n_frames, dtype=np.float32)
    bearing = np.column_stack((frame_bearing, -frame_bearing)).reshape(-1)
    center = np.asarray([100.0, 100.0], dtype=np.float64)
    radians = np.deg2rad(bearing.astype(np.float64))
    chaser_xy = center + np.column_stack(
        (100.0 * np.cos(radians), -100.0 * np.sin(radians))
    )
    source = GazeTrackingInput(
        recording_id="recording-1",
        source_relative_frame_run_path=records[0]["relative_frame"]["run_path"],
        source_relative_frame_manifest_sha256=records[0]["relative_frame"][
            "manifest_sha256"
        ],
        source_eye_run_path="analysis/eye_angle_runs/eye-v1",
        source_eye_manifest_sha256="7" * 64,
        source_eye_convention_receipt_sha256="8" * 64,
        source_eye_channel_policy="smoothed:left_gaze,right_gaze:vergence",
        source_semantic_selection_manifest_sha256="c" * 64,
        source_radial_run_path=records[0]["radial_near_field"]["run_path"],
        source_radial_manifest_sha256=records[0]["radial_near_field"][
            "manifest_sha256"
        ],
        source_radial_payload_sha256=radial_manifest["scientific_payload_sha256"],
        source_arena_geometry_and_scale={"authority_sha256": "a" * 64},
        arena_center_xy_px=center,
        arena_radius_px=200.0,
        arena_radius_mm=20.0,
        pixels_per_mm=10.0,
        n_frames=n_frames,
        n_chasers=n_chasers,
        acquisition_frame_id_by_frame=np.arange(100, 110, dtype=np.int64),
        timestamp_ns_by_frame=np.arange(n_frames, dtype=np.int64) * 100_000_000,
        timestamp_valid_by_frame=np.ones(n_frames, dtype=bool),
        semantic_role_code_by_frame=np.full(
            n_frames, ROLE_CODES["chaser_training"], dtype=np.uint8
        ),
        chaser_identity_code=np.tile(np.asarray([1, 2], dtype=np.uint16), n_frames),
        fish_position_xy_px=np.broadcast_to(center, (n_frames, 2)).copy(),
        fish_position_valid=np.ones(n_frames, dtype=bool),
        chaser_position_xy_px=chaser_xy,
        chaser_position_valid=np.ones(n_frames * n_chasers, dtype=bool),
        chaser_occurrence_member=np.ones(n_frames * n_chasers, dtype=bool),
        body_origin_xy_px=np.broadcast_to(center, (n_frames, 2)).copy(),
        body_forward_axis_xy=np.tile([1.0, 0.0], (n_frames, 1)),
        body_left_axis_xy=np.tile([0.0, -1.0], (n_frames, 1)),
        body_axes_valid=np.ones(n_frames, dtype=bool),
        distance_mm=np.full(n_frames * n_chasers, 12.0, dtype=np.float32),
        distance_valid=np.ones(n_frames * n_chasers, dtype=bool),
        chaser_bearing_deg=bearing.astype(np.float32),
        chaser_bearing_valid=np.ones(n_frames * n_chasers, dtype=bool),
        gaze_signed_deg=np.column_stack((frame_bearing, frame_bearing + 5.0)),
        gaze_valid=np.ones((n_frames, 2), dtype=bool),
        vergence_deg=np.full(n_frames, 10.0, dtype=np.float32),
        vergence_valid=np.ones(n_frames, dtype=bool),
        minimum_regression_samples=3,
    )
    prepared = prepare_gaze_tracking_successor(source)
    scientific = plain(prepared.manifest)
    scientific.pop("payload_digest")
    run_path = f"analysis/chaser_gaze_tracking_runs/{run_name}"
    run = _successor(run_path, "chaser_gaze_tracking", scientific)
    try:
        parent = root["analysis/chaser_gaze_tracking_runs"]
    except KeyError:
        parent = _Group()
        root["analysis/chaser_gaze_tracking_runs"] = parent
    parent[run_name] = run
    root[run_path] = run
    return run


def _redigest_spatial(root: _Group) -> None:
    spatial = root["analysis/chaser_spatial_occupancy_runs/paired-spatial-v1"]
    spatial.attrs["composable_chaser_successor_manifest_sha256"] = (
        canonical_json_sha256(spatial.attrs["composable_chaser_successor_manifest"])
    )


def _provider_records(root: _Group) -> list[dict[str, Any]]:
    spatial = root["analysis/chaser_spatial_occupancy_runs/paired-spatial-v1"]
    return spatial.attrs["composable_chaser_successor_manifest"]["scientific_manifest"][
        "sources"
    ]["position_providers"]


def test_exact_successor_discovery_uses_spatial_bundle_and_exact_children(
    monkeypatch,
) -> None:
    root = _archive()
    monkeypatch.setattr(
        "apps.marimo.components.registry.open_zarr_root",
        lambda *args, **kwargs: root,
    )

    options = discover_exact_chaser_successor_options("recording.zarr")

    assert len(options) == 1
    assert options[0].renderer == CHASER_EXACT_SUCCESSOR_RENDERER
    assert options[0].spec["bundle_status"] == "exact_selector_ineligible"
    assert options[0].spec["schema_version"] == 10
    assert options[0].spec["provider_ids"] == ["keypoint.v1", "detection.v1"]
    spatial_parameters = options[0].spec["display_parameters"]["spatial_occupancy"]
    assert spatial_parameters["recipe_id"] == DISPLAY_RECIPE_ID
    assert spatial_parameters["default_display_mode"] == DEFAULT_DISPLAY_MODE_ID
    assert spatial_parameters["source_arrays"] == [
        "occupancy_density_valid_in_arena",
        "occupancy_fraction_candidate_epoch",
    ]
    assert spatial_parameters["default_normalization"] == "valid_in_arena"
    assert spatial_parameters["available_normalizations"] == [
        "valid_in_arena",
        "candidate_epoch",
    ]
    assert spatial_parameters["density_multiplier_to_percent"] == 100.0
    assert spatial_parameters["density_color_normalization"] == (
        "shared_robust_p98_default_full_range_available"
    )
    assert spatial_parameters["display_bin_widths_mm"] == [2.0, 4.0]
    assert (
        spatial_parameters["provider_difference"]
        == "detection_minus_keypoint_percentage_points_per_bin"
    )
    controller_parameters = options[0].spec["display_parameters"]["controller_trials"]
    assert controller_parameters["max_points_per_trace"] == 6000
    assert controller_parameters["max_trial_panels"] == 32
    assert controller_parameters["max_gap_markers_per_panel"] == 2000
    proofs = options[0].spec["relative_frame_binding_proofs"]
    assert len(proofs) == 2
    assert proofs[0]["spatial_binding_profile"] == RECEIPT_BOUND_PROFILE
    assert proofs[0]["radial_binding_profile"] == MINIMAL_EXACT_CHILD_PROFILE
    assert proofs[0]["validation_receipt_sha256"] == "f" * 64
    assert proofs[0]["verification_mode"] == ("receipt_bound_targeted_array_rehash_v1")
    body = options[0].spec["analysis_bindings"]["body_bearing"]
    assert body["array_paths"] == [
        "body/body_bearing_deg",
        "body/body_bearing_valid",
    ]
    assert body["position_substitution"] == "prohibited"
    heading = options[0].spec["analysis_bindings"]["body_heading"]
    assert heading["array_paths"] == [
        "body/body_source_row_id",
        "body/body_source_row_valid",
        "body/body_heading_deg",
        "body/body_heading_valid",
        "body/body_heading_reason_code",
    ]
    assert heading["frame_collapse_policy"] == (
        "exact_equality_across_flattened_chaser_rows_then_one_row_per_acquisition_frame"
    )
    assert heading["motion_heading_fallback"] == "prohibited"
    heading_parameters = options[0].spec["display_parameters"]["fish_heading"]
    assert heading_parameters["bin_width_deg"] == 10.0
    assert heading_parameters["motion_heading_fallback"] == "prohibited"
    bearing_distance_parameters = options[0].spec["display_parameters"][
        "body_bearing_distance"
    ]
    assert bearing_distance_parameters["distance_bin_width_mm"] == 5.0
    assert bearing_distance_parameters["bearing_bin_width_deg"] == 30.0
    assert bearing_distance_parameters["density_normalization"] == (
        "probability_within_panel_chaser"
    )
    assert bearing_distance_parameters["interpolation"] == "prohibited"
    controller = options[0].spec["analysis_bindings"]["controller_trials"]
    assert controller["run_path"] == (
        "analysis/controller_chase_trial_runs/controller-v1"
    )
    assert controller["source_relative_frame"] == {
        key: _provider_records(root)[0]["relative_frame"][key]
        for key in ("run_path", "manifest_sha256")
    }
    bout = options[0].spec["analysis_bindings"]["generalized_bout_response"]
    assert bout["run_path"] == (
        "analysis/generalized_chaser_bout_response_runs/bout-v1"
    )
    assert (
        bout["controller_trial_payload_sha256"]
        == controller["scientific_payload_sha256"]
    )
    assert bout["body_extension_present"] is True
    bout_parameters = options[0].spec["display_parameters"]["generalized_bout_response"]
    assert bout_parameters["distance_band_edges"] == "persisted_no_rebinning"
    assert bout_parameters["bout_resegmentation"] == "prohibited"
    escape = options[0].spec["analysis_bindings"]["escape_freeze"]
    assert escape["run_path"] == "analysis/chaser_escape_freeze_runs/escape-v1"
    assert (
        escape["controller_trial_payload_sha256"]
        == controller["scientific_payload_sha256"]
    )
    assert escape["bout_response_payload_sha256"] == bout["scientific_payload_sha256"]
    assert escape["classifier_parameters"]["freeze_window_s"] == 1.0
    escape_parameters = options[0].spec["display_parameters"]["escape_freeze"]
    assert escape_parameters["response_classes"] == (
        "persisted_no_viewer_reclassification"
    )
    assert escape_parameters["event_trace_samples"] == (
        "not_persisted_no_viewer_reconstruction"
    )
    assert group_specs_by_provider(options) == {
        "stimulus_chaser_exact_successors": options
    }


def test_exact_gaze_capability_is_discovered_by_all_sealed_sources(
    monkeypatch,
) -> None:
    root = _archive()
    gaze = _add_gaze_successor(root)
    monkeypatch.setattr(
        "apps.marimo.components.registry.open_zarr_root",
        lambda *args, **kwargs: root,
    )

    options = discover_exact_chaser_successor_options("recording.zarr")

    assert len(options) == 1
    binding = options[0].spec["analysis_bindings"]["gaze_tracking"]
    assert binding["run_path"] == "analysis/chaser_gaze_tracking_runs/gaze-v1"
    assert (
        binding["manifest_sha256"]
        == gaze.attrs["composable_chaser_successor_manifest_sha256"]
    )
    assert binding["source_eye_orientation"]["run_path"] == (
        "analysis/eye_angle_runs/eye-v1"
    )
    assert (
        options[0].spec["display_parameters"]["gaze_tracking"][
            "rotated_spatial_controls"
        ]
        == "persisted_reviewed_arena_rotations_with_collision_exclusion"
    )


def test_exact_gaze_capability_is_hidden_when_source_join_is_ambiguous(
    monkeypatch,
) -> None:
    root = _archive()
    _add_gaze_successor(root, run_name="gaze-v1")
    _add_gaze_successor(root, run_name="gaze-v2")
    monkeypatch.setattr(
        "apps.marimo.components.registry.open_zarr_root",
        lambda *args, **kwargs: root,
    )

    options = discover_exact_chaser_successor_options("recording.zarr")

    assert len(options) == 1
    assert "gaze_tracking" not in options[0].spec["analysis_bindings"]


def test_exact_gaze_capability_is_hidden_for_mismatched_radial_arena(
    monkeypatch,
) -> None:
    root = _archive()
    gaze = _add_gaze_successor(root)
    manifest = gaze.attrs["composable_chaser_successor_manifest"]
    scientific = manifest["scientific_manifest"]
    scientific["arena"]["center_xy_px"] = [101.0, 100.0]
    scientific["payload_digest"] = canonical_json_sha256(
        {key: value for key, value in scientific.items() if key != "payload_digest"}
    )
    manifest["scientific_payload_sha256"] = scientific["payload_digest"]
    gaze.attrs["composable_chaser_successor_manifest_sha256"] = canonical_json_sha256(
        manifest
    )
    monkeypatch.setattr(
        "apps.marimo.components.registry.open_zarr_root",
        lambda *args, **kwargs: root,
    )

    options = discover_exact_chaser_successor_options("recording.zarr")

    assert len(options) == 1
    assert "gaze_tracking" not in options[0].spec["analysis_bindings"]


def test_bout_response_contract_hashes_frozen_loader_metadata() -> None:
    root = _archive()
    bout = root["analysis/generalized_chaser_bout_response_runs/bout-v1"]
    scientific = bout.attrs["composable_chaser_successor_manifest"][
        "scientific_manifest"
    ]

    verified = validate_scientific_manifest(
        freeze(scientific),
        expected_scientific_payload_sha256=scientific["payload_digest"],
        expected_n_frames=10,
        expected_n_chasers=2,
        expected_relative_binding=_provider_records(root)[0]["relative_frame"],
        expected_semantic_manifest_sha256="c" * 64,
        expected_controller_payload_sha256=scientific["sources"][
            "controller_trial_payload_sha256"
        ],
    )

    assert verified["n_bouts"] == 3
    assert verified["body_extension_present"] is True


def test_controller_trial_capability_is_hidden_when_exact_join_is_ambiguous(
    monkeypatch,
) -> None:
    root = _archive()
    parent = root["analysis/controller_chase_trial_runs"]
    original = parent["controller-v1"]
    scientific = dict(
        original.attrs["composable_chaser_successor_manifest"]["scientific_manifest"]
    )
    scientific.pop("payload_digest")
    duplicate_path = "analysis/controller_chase_trial_runs/controller-v2"
    duplicate = _successor(
        duplicate_path,
        "controller_chase_trials",
        scientific,
    )
    parent["controller-v2"] = duplicate
    root[duplicate_path] = duplicate
    monkeypatch.setattr(
        "apps.marimo.components.registry.open_zarr_root",
        lambda *args, **kwargs: root,
    )

    options = discover_exact_chaser_successor_options("recording.zarr")

    assert len(options) == 1
    assert set(options[0].spec["analysis_bindings"]) == {
        "body_bearing",
        "body_heading",
    }


def test_bout_response_capability_is_hidden_when_exact_join_is_ambiguous(
    monkeypatch,
) -> None:
    root = _archive()
    parent = root["analysis/generalized_chaser_bout_response_runs"]
    original = parent["bout-v1"]
    scientific = dict(
        original.attrs["composable_chaser_successor_manifest"]["scientific_manifest"]
    )
    scientific.pop("payload_digest")
    duplicate_path = "analysis/generalized_chaser_bout_response_runs/bout-v2"
    duplicate = _successor(
        duplicate_path,
        "generalized_chaser_bout_response",
        scientific,
    )
    parent["bout-v2"] = duplicate
    root[duplicate_path] = duplicate
    monkeypatch.setattr(
        "apps.marimo.components.registry.open_zarr_root",
        lambda *args, **kwargs: root,
    )

    options = discover_exact_chaser_successor_options("recording.zarr")

    assert len(options) == 1
    bindings = options[0].spec["analysis_bindings"]
    assert "controller_trials" in bindings
    assert "generalized_bout_response" not in bindings
    assert "escape_freeze" not in bindings


def test_escape_freeze_capability_is_hidden_when_exact_join_is_ambiguous(
    monkeypatch,
) -> None:
    root = _archive()
    parent = root["analysis/chaser_escape_freeze_runs"]
    original = parent["escape-v1"]
    scientific = dict(
        original.attrs["composable_chaser_successor_manifest"]["scientific_manifest"]
    )
    scientific.pop("payload_digest")
    duplicate_path = "analysis/chaser_escape_freeze_runs/escape-v2"
    duplicate = _successor(duplicate_path, "chaser_escape_freeze", scientific)
    parent["escape-v2"] = duplicate
    root[duplicate_path] = duplicate
    monkeypatch.setattr(
        "apps.marimo.components.registry.open_zarr_root",
        lambda *args, **kwargs: root,
    )

    options = discover_exact_chaser_successor_options("recording.zarr")

    assert len(options) == 1
    bindings = options[0].spec["analysis_bindings"]
    assert "generalized_bout_response" in bindings
    assert "escape_freeze" not in bindings


def test_escape_freeze_capability_is_hidden_for_wrong_bout_payload(
    monkeypatch,
) -> None:
    root = _archive()
    escape = root["analysis/chaser_escape_freeze_runs/escape-v1"]
    manifest = escape.attrs["composable_chaser_successor_manifest"]
    scientific = manifest["scientific_manifest"]
    scientific["sources"]["bout_response_payload_sha256"] = "9" * 64
    scientific["payload_digest"] = canonical_json_sha256(
        {key: value for key, value in scientific.items() if key != "payload_digest"}
    )
    manifest["scientific_payload_sha256"] = scientific["payload_digest"]
    escape.attrs["composable_chaser_successor_manifest_sha256"] = canonical_json_sha256(
        manifest
    )
    monkeypatch.setattr(
        "apps.marimo.components.registry.open_zarr_root",
        lambda *args, **kwargs: root,
    )

    options = discover_exact_chaser_successor_options("recording.zarr")

    assert len(options) == 1
    bindings = options[0].spec["analysis_bindings"]
    assert "generalized_bout_response" in bindings
    assert "escape_freeze" not in bindings


def test_bout_response_capability_is_hidden_for_wrong_controller_payload(
    monkeypatch,
) -> None:
    root = _archive()
    bout = root["analysis/generalized_chaser_bout_response_runs/bout-v1"]
    manifest = bout.attrs["composable_chaser_successor_manifest"]
    scientific = manifest["scientific_manifest"]
    scientific["sources"]["controller_trial_payload_sha256"] = "9" * 64
    scientific["payload_digest"] = canonical_json_sha256(
        {key: value for key, value in scientific.items() if key != "payload_digest"}
    )
    manifest["scientific_payload_sha256"] = scientific["payload_digest"]
    bout.attrs["composable_chaser_successor_manifest_sha256"] = canonical_json_sha256(
        manifest
    )
    monkeypatch.setattr(
        "apps.marimo.components.registry.open_zarr_root",
        lambda *args, **kwargs: root,
    )

    options = discover_exact_chaser_successor_options("recording.zarr")

    assert len(options) == 1
    bindings = options[0].spec["analysis_bindings"]
    assert "controller_trials" in bindings
    assert "generalized_bout_response" not in bindings


def test_bout_response_capability_is_hidden_for_stale_scientific_payload(
    monkeypatch,
) -> None:
    root = _archive()
    bout = root["analysis/generalized_chaser_bout_response_runs/bout-v1"]
    manifest = bout.attrs["composable_chaser_successor_manifest"]
    manifest["scientific_manifest"]["tampered_after_publication"] = True
    bout.attrs["composable_chaser_successor_manifest_sha256"] = canonical_json_sha256(
        manifest
    )
    monkeypatch.setattr(
        "apps.marimo.components.registry.open_zarr_root",
        lambda *args, **kwargs: root,
    )

    options = discover_exact_chaser_successor_options("recording.zarr")

    assert len(options) == 1
    bindings = options[0].spec["analysis_bindings"]
    assert "controller_trials" in bindings
    assert "generalized_bout_response" not in bindings


def test_controller_trial_capability_is_hidden_for_wrong_relative_source(
    monkeypatch,
) -> None:
    root = _archive()
    controller = root["analysis/controller_chase_trial_runs/controller-v1"]
    manifest = controller.attrs["composable_chaser_successor_manifest"]
    manifest["scientific_manifest"]["source_relative_frame"]["manifest_sha256"] = (
        "e" * 64
    )
    manifest["scientific_payload_sha256"] = manifest["scientific_manifest"][
        "payload_digest"
    ]
    controller.attrs["composable_chaser_successor_manifest_sha256"] = (
        canonical_json_sha256(manifest)
    )
    monkeypatch.setattr(
        "apps.marimo.components.registry.open_zarr_root",
        lambda *args, **kwargs: root,
    )

    options = discover_exact_chaser_successor_options("recording.zarr")

    assert len(options) == 1
    assert set(options[0].spec["analysis_bindings"]) == {
        "body_bearing",
        "body_heading",
    }


def test_controller_trial_capability_is_hidden_above_panel_bound(monkeypatch) -> None:
    root = _archive()
    controller = root["analysis/controller_chase_trial_runs/controller-v1"]
    manifest = controller.attrs["composable_chaser_successor_manifest"]
    scientific = manifest["scientific_manifest"]
    scientific["dimensions"]["n_trials"] = 33
    scientific["payload_digest"] = canonical_json_sha256(
        {key: value for key, value in scientific.items() if key != "payload_digest"}
    )
    manifest["scientific_payload_sha256"] = scientific["payload_digest"]
    controller.attrs["composable_chaser_successor_manifest_sha256"] = (
        canonical_json_sha256(manifest)
    )
    monkeypatch.setattr(
        "apps.marimo.components.registry.open_zarr_root",
        lambda *args, **kwargs: root,
    )

    options = discover_exact_chaser_successor_options("recording.zarr")

    assert len(options) == 1
    assert set(options[0].spec["analysis_bindings"]) == {
        "body_bearing",
        "body_heading",
    }


def test_production_binding_shapes_are_not_whole_object_equal() -> None:
    root = _archive()
    spatial_binding = _provider_records(root)[0]["relative_frame"]
    radial = root["analysis/chaser_radial_near_field_runs/keypoint-radial"]
    radial_binding = radial.attrs["composable_chaser_successor_manifest"][
        "scientific_manifest"
    ]["sources"]["relative_frame"]

    assert spatial_binding != radial_binding
    proof = require_same_exact_relative_frame_child(spatial_binding, radial_binding)

    assert dict(proof.normalized_identity) == {
        "run_path": spatial_binding["run_path"],
        "manifest_sha256": spatial_binding["manifest_sha256"],
    }
    assert proof.expected.profile_id == RECEIPT_BOUND_PROFILE
    assert proof.observed.profile_id == MINIMAL_EXACT_CHILD_PROFILE


def test_same_child_accepts_and_retains_independent_receipt_evidence() -> None:
    expected = {
        "run_path": "analysis/chaser_relative_frame_runs/exact-run",
        "manifest_sha256": "a" * 64,
        "validation_receipt_sha256": "b" * 64,
        "verification_mode": "receipt_bound_targeted_array_rehash_v1",
    }
    observed = {
        **expected,
        "validation_receipt_sha256": "c" * 64,
    }

    proof = require_same_exact_relative_frame_child(expected, observed)
    provenance = dict(proof.provenance_record())

    assert dict(proof.normalized_identity) == {
        "run_path": expected["run_path"],
        "manifest_sha256": expected["manifest_sha256"],
    }
    assert provenance["expected_validation_receipt_sha256"] == "b" * 64
    assert provenance["observed_validation_receipt_sha256"] == "c" * 64
    assert provenance["validation_receipt_sha256"] is None
    assert provenance["validation_receipt_sha256s"] == ("b" * 64, "c" * 64)
    assert provenance["receipt_evidence_relationship"] == (
        "independent_receipts_same_exact_child"
    )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("validation_receipt_sha256", "not-a-digest", "lowercase SHA-256"),
        ("verification_mode", "deep_audit", "unsupported"),
    ],
)
def test_receipt_bound_binding_rejects_invalid_evidence(
    field: str,
    value: str,
    match: str,
) -> None:
    binding = dict(_provider_records(_archive())[0]["relative_frame"])
    binding[field] = value

    with pytest.raises(ExactRelativeFrameBindingError, match=match):
        validate_exact_relative_frame_binding(binding)


def test_relative_binding_rejects_unrecognized_field_set() -> None:
    binding = dict(_provider_records(_archive())[0]["relative_frame"])
    binding["receipt_path"] = "/not/an/admitted/field"

    with pytest.raises(ExactRelativeFrameBindingError, match="unrecognized"):
        validate_exact_relative_frame_binding(binding)


@pytest.mark.parametrize(
    "run_name",
    ["latest", "latest_complete", "authoritative_run", "selected", "current_run"],
)
def test_relative_binding_rejects_selector_like_child_names(run_name: str) -> None:
    binding = dict(_provider_records(_archive())[0]["relative_frame"])
    binding["run_path"] = f"analysis/chaser_relative_frame_runs/{run_name}"

    with pytest.raises(ExactRelativeFrameBindingError, match="exact child"):
        validate_exact_relative_frame_binding(binding)


@pytest.mark.parametrize("field", ["run_path", "manifest_sha256"])
def test_relative_binding_equivalence_rejects_different_child_identity(
    field: str,
) -> None:
    root = _archive()
    spatial_binding = dict(_provider_records(root)[0]["relative_frame"])
    radial = root["analysis/chaser_radial_near_field_runs/keypoint-radial"]
    radial_binding = dict(
        radial.attrs["composable_chaser_successor_manifest"]["scientific_manifest"][
            "sources"
        ]["relative_frame"]
    )
    radial_binding[field] = (
        "analysis/chaser_relative_frame_runs/another-exact-run"
        if field == "run_path"
        else "e" * 64
    )

    with pytest.raises(ExactRelativeFrameBindingError, match="different exact"):
        require_same_exact_relative_frame_child(spatial_binding, radial_binding)


def test_loader_bundle_validation_accepts_enriched_spatial_and_minimal_radial() -> None:
    root = _archive()
    records = _provider_records(root)
    semantic = {"run_path": "analysis/protocol_semantic/run", "sha256": "1" * 64}
    geometry = {"authority_id": "reviewed-arena", "sha256": "2" * 64}
    epochs = [{"epoch_id": "pre", "start_frame": 0, "end_frame": 10}]
    arena = {"radius_mm": 10.0}
    radials = []
    for record in records:
        role = record["provider_role"]
        radial_group = root[f"analysis/chaser_radial_near_field_runs/{role}-radial"]
        radial_manifest = radial_group.attrs["composable_chaser_successor_manifest"]
        scientific = dict(radial_manifest["scientific_manifest"])
        scientific["sources"] = {
            **scientific["sources"],
            "protocol_semantic_selection": semantic,
            "arena_geometry_and_scale": geometry,
        }
        scientific["epoch_records"] = epochs
        scientific["arena"] = arena
        radials.append(
            SimpleNamespace(
                run_path=record["radial_near_field"]["run_path"],
                manifest_sha256=record["radial_near_field"]["manifest_sha256"],
                scientific_manifest=scientific,
            )
        )
    spatial = SimpleNamespace(
        scientific_manifest={
            "sources": {
                "position_providers": records,
                "protocol_semantic_selection": semantic,
                "arena_geometry_and_scale": geometry,
            },
            "epoch_records": epochs,
        }
    )

    provider_ids, proofs = _verify_bundle_children(
        spatial,
        radials,
        tuple(record["relative_frame"] for record in records),
    )

    assert provider_ids == ("keypoint.v1", "detection.v1")
    assert tuple(proof.expected.profile_id for proof in proofs) == (
        RECEIPT_BOUND_PROFILE,
        RECEIPT_BOUND_PROFILE,
    )
    assert tuple(proof.observed.profile_id for proof in proofs) == (
        MINIMAL_EXACT_CHILD_PROFILE,
        MINIMAL_EXACT_CHILD_PROFILE,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("validation_receipt_sha256", "invalid"),
        ("verification_mode", "unsupported"),
    ],
)
def test_discovery_hides_invalid_receipt_binding(
    monkeypatch,
    field: str,
    value: str,
) -> None:
    root = _archive()
    _provider_records(root)[0]["relative_frame"][field] = value
    _redigest_spatial(root)
    monkeypatch.setattr(
        "apps.marimo.components.registry.open_zarr_root",
        lambda *args, **kwargs: root,
    )

    assert discover_exact_chaser_successor_options("recording.zarr") == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (
            "run_path",
            "analysis/chaser_relative_frame_runs/another-exact-relative-run",
        ),
        ("manifest_sha256", "e" * 64),
    ],
)
def test_discovery_hides_wrong_relative_child_identity(
    monkeypatch,
    field: str,
    value: str,
) -> None:
    root = _archive()
    _provider_records(root)[0]["relative_frame"][field] = value
    _redigest_spatial(root)
    monkeypatch.setattr(
        "apps.marimo.components.registry.open_zarr_root",
        lambda *args, **kwargs: root,
    )

    assert discover_exact_chaser_successor_options("recording.zarr") == []


def test_discovery_hides_relative_child_from_another_recording(monkeypatch) -> None:
    root = _archive()
    relative = root["analysis/chaser_relative_frame_runs/keypoint-relative"]
    manifest = relative.attrs["chaser_relative_frame_manifest"]
    manifest["recording_id"] = "recording-2"
    digest = canonical_json_sha256(manifest)
    relative.attrs["chaser_relative_frame_manifest_sha256"] = digest
    _provider_records(root)[0]["relative_frame"]["manifest_sha256"] = digest
    _redigest_spatial(root)
    monkeypatch.setattr(
        "apps.marimo.components.registry.open_zarr_root",
        lambda *args, **kwargs: root,
    )

    assert discover_exact_chaser_successor_options("recording.zarr") == []


def test_discovery_hides_incomplete_relative_child(monkeypatch) -> None:
    root = _archive()
    relative = root["analysis/chaser_relative_frame_runs/keypoint-relative"]
    relative.attrs["palette_run_completion_status"] = "writing"
    monkeypatch.setattr(
        "apps.marimo.components.registry.open_zarr_root",
        lambda *args, **kwargs: root,
    )

    assert discover_exact_chaser_successor_options("recording.zarr") == []


def test_discovery_hides_reversed_provider_roles(monkeypatch) -> None:
    root = _archive()
    _provider_records(root).reverse()
    _redigest_spatial(root)
    monkeypatch.setattr(
        "apps.marimo.components.registry.open_zarr_root",
        lambda *args, **kwargs: root,
    )

    assert discover_exact_chaser_successor_options("recording.zarr") == []


@pytest.mark.parametrize("selector", ["latest", "latest_pending", "current_run"])
def test_discovery_hides_forbidden_parent_selector(monkeypatch, selector: str) -> None:
    root = _archive()
    root["analysis/chaser_spatial_occupancy_runs"].attrs[selector] = "paired-spatial-v1"
    monkeypatch.setattr(
        "apps.marimo.components.registry.open_zarr_root",
        lambda *args, **kwargs: root,
    )

    assert discover_exact_chaser_successor_options("recording.zarr") == []


def test_discovery_hides_controller_capability_with_parent_selector(
    monkeypatch,
) -> None:
    root = _archive()
    root["analysis/controller_chase_trial_runs"].attrs["authoritative_run"] = (
        "controller-v1"
    )
    monkeypatch.setattr(
        "apps.marimo.components.registry.open_zarr_root",
        lambda *args, **kwargs: root,
    )

    options = discover_exact_chaser_successor_options("recording.zarr")

    assert len(options) == 1
    assert set(options[0].spec["analysis_bindings"]) == {
        "body_bearing",
        "body_heading",
    }


def test_discovery_does_not_retry_unconsolidated_metadata(monkeypatch) -> None:
    calls = []

    def fail_open(*args, **kwargs):
        calls.append(kwargs)
        raise ValueError("missing consolidated generation")

    monkeypatch.setattr("apps.marimo.components.registry.open_zarr_root", fail_open)

    assert discover_exact_chaser_successor_options("recording.zarr") == []
    assert calls == [{"mode": "r", "use_consolidated": True}]


def test_exact_successor_discovery_hides_stale_child_provider_binding(
    monkeypatch,
) -> None:
    root = _archive()
    radial = root["analysis/chaser_radial_near_field_runs/detection-radial"]
    manifest = radial.attrs["composable_chaser_successor_manifest"]
    manifest["scientific_manifest"]["position_provider"]["provider_digest"] = "c" * 64
    radial.attrs["composable_chaser_successor_manifest_sha256"] = canonical_json_sha256(
        manifest
    )
    monkeypatch.setattr(
        "apps.marimo.components.registry.open_zarr_root",
        lambda *args, **kwargs: root,
    )

    assert discover_exact_chaser_successor_options("recording.zarr") == []


def test_trace_display_projection_preserves_extrema_and_missing_breaks() -> None:
    x = np.arange(100, dtype=np.float64)
    y = np.sin(x / 10.0)
    y[40] = 50.0
    valid = np.ones(100, dtype=bool)
    valid[48:53] = False

    display_x, display_y = _trace_display_projection(x, y, valid, max_points=24)

    assert 40.0 in display_x
    assert 50.0 in display_y
    assert np.isnan(display_y).any()
    finite_x = display_x[np.isfinite(display_y)]
    assert not np.any((finite_x >= 48) & (finite_x <= 52))


def test_trajectory_display_projection_retains_coordinate_extrema() -> None:
    xy = np.column_stack((np.arange(1000), -np.arange(1000))).astype(np.float64)
    valid = np.ones(1000, dtype=bool)

    indices = _trajectory_display_indices(xy, valid, max_points=40)

    assert indices.size <= 44
    assert {0, 999}.issubset(indices.tolist())
