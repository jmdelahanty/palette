from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import plotly.graph_objects as go

from apps.marimo.components.chaser_exact.body_bearing import (
    _bearing_histogram,
    build_exact_body_bearing_output,
)
from apps.marimo.components.chaser_exact.body_bearing_distance import (
    build_exact_body_bearing_distance_output,
)
from apps.marimo.components.chaser_exact.gaze_tracking import (
    _histogram_probability,
    _uniform_indices,
    build_exact_gaze_tracking_output,
)
from apps.marimo.components.chaser_exact.projection import RelativeFrameProjection
from fisheye.analysis_workflows.gaze_tracking_successor import (
    GazeTrackingInput,
    prepare_gaze_tracking_successor,
)
from fisheye.analysis_workflows.generalized_bout_response_successor import ROLE_CODES
from fisheye.visualization.chaser_body_bearing_distance import (
    bearing_bin_edges_deg,
    body_bearing_distance_histogram,
    body_bearing_distance_valid_mask,
    distance_bin_edges_mm,
)


class _Mo:
    @staticmethod
    def callout(value, *, kind):
        return {"value": value, "kind": kind}

    @staticmethod
    def vstack(values):
        return list(values)


def test_body_bearing_histogram_is_whole_circle_probability() -> None:
    centers, probability, counts = _bearing_histogram(
        np.asarray([-175.0, -5.0, 5.0, 175.0, np.nan]),
        np.asarray([True, True, True, True, False]),
    )

    assert centers.size == probability.size == counts.size == 36
    assert int(np.sum(counts)) == 4
    np.testing.assert_allclose(np.sum(probability), 1.0)


def test_body_bearing_renderer_uses_only_keypoint_body_extension() -> None:
    arrays = {
        "acquisition_frame_id": np.arange(4, dtype=np.int64),
        "selection_member": np.ones(4, dtype=bool),
        "chaser_occurrence_member": np.ones(4, dtype=bool),
        "chaser_identity_code": np.ones(4, dtype=np.uint16),
        "chaser_behavior_role_code": np.ones(4, dtype=np.uint8),
    }
    relative = RelativeFrameProjection(
        run_path="analysis/chaser_relative_frame_runs/keypoint-v1",
        run_name="keypoint-v1",
        recording_id="recording-1",
        manifest_sha256="a" * 64,
        n_frames=4,
        n_chasers=1,
        source_authorities={},
        arrays=arrays,
        body_arrays={
            "body_bearing_deg": np.asarray([-45, -5, 5, 45], dtype=np.float32),
            "body_bearing_valid": np.ones(4, dtype=bool),
        },
    )
    projection = SimpleNamespace(
        relatives=(relative, SimpleNamespace()),
        radials=(
            SimpleNamespace(
                scientific_manifest={"identity_registries": {"behavior_role": {}}}
            ),
            SimpleNamespace(),
        ),
        epoch_records=(
            {
                "analysis_role": "chaser_pre",
                "start_frame": 0,
                "end_frame_exclusive": 2,
            },
        ),
        provenance={},
        recording_id="recording-1",
    )

    output = build_exact_body_bearing_output(_Mo, go, projection)

    assert len(output) == 2
    figure = output[1]
    assert len(figure.data) == 2
    assert figure.layout.meta["display_recipe"]["bin_width_deg"] == 10.0
    assert (
        figure.layout.meta["display_recipe"]["detection_position_substitution"]
        == "prohibited"
    )


def test_body_bearing_distance_histogram_uses_exact_joint_validity() -> None:
    distance = np.asarray([2.0, 7.0, 12.0, np.nan])
    bearing = np.asarray([-170.0, -10.0, 40.0, 90.0])
    valid = body_bearing_distance_valid_mask(
        distance,
        bearing,
        np.asarray([True, True, True, False]),
        np.asarray([True, True, False, True]),
        np.asarray([True, True, True, True]),
        np.asarray([True, True, True, True]),
    )
    edges = distance_bin_edges_mm(distance, valid)
    histogram = body_bearing_distance_histogram(
        distance,
        bearing,
        valid,
        distance_edges_mm=edges,
        bearing_edges_deg=bearing_bin_edges_deg(),
    )

    np.testing.assert_array_equal(valid, np.asarray([True, True, False, False]))
    np.testing.assert_allclose(edges, np.asarray([0.0, 5.0, 10.0]))
    assert histogram.denominator == 2
    assert int(np.sum(histogram.counts)) == 2
    np.testing.assert_allclose(np.sum(histogram.probability), 1.0)


def test_body_bearing_distance_renderer_exposes_point_rows_and_joint_density() -> None:
    arrays = {
        "acquisition_frame_id": np.arange(5, dtype=np.int64),
        "selection_member": np.ones(5, dtype=bool),
        "chaser_occurrence_member": np.asarray([True, True, True, False, True]),
        "chaser_identity_code": np.ones(5, dtype=np.uint16),
        "chaser_behavior_role_code": np.ones(5, dtype=np.uint8),
        "relative_distance_physical": np.asarray(
            [2.0, 7.0, 12.0, np.nan, 17.0], dtype=np.float32
        ),
        "relative_physical_valid": np.asarray([True, True, True, False, True]),
    }
    relative = RelativeFrameProjection(
        run_path="analysis/chaser_relative_frame_runs/keypoint-v1",
        run_name="keypoint-v1",
        recording_id="recording-1",
        manifest_sha256="a" * 64,
        n_frames=5,
        n_chasers=1,
        source_authorities={},
        arrays=arrays,
        body_arrays={
            "body_bearing_deg": np.asarray(
                [-170.0, -10.0, 40.0, 90.0, 170.0], dtype=np.float32
            ),
            "body_bearing_valid": np.asarray([True, True, False, True, True]),
        },
    )
    projection = SimpleNamespace(
        relatives=(relative, SimpleNamespace()),
        radials=(
            SimpleNamespace(
                scientific_manifest={"identity_registries": {"behavior_role": {}}}
            ),
            SimpleNamespace(),
        ),
        epoch_records=(
            {
                "analysis_role": "chaser_pre",
                "start_frame": 0,
                "end_frame_exclusive": 2,
            },
        ),
        provenance={"verification_mode": "receipt_bound_targeted_array_rehash_v1"},
        recording_id="recording-1",
    )

    output = build_exact_body_bearing_distance_output(_Mo, go, projection)

    assert len(output) == 3
    point_cloud, density = output[1:]
    assert len(point_cloud.data) == len(density.data) == 2
    recipe = density.layout.meta["display_recipe"]
    assert recipe["distance_bin_width_mm"] == 5.0
    assert recipe["bearing_bin_width_deg"] == 30.0
    assert recipe["density_normalization"] == "probability_within_panel_chaser"
    assert [row["valid_row_count"] for row in recipe["panel_records"]] == [3, 2]
    assert int(np.sum(density.data[0].customdata[:, 0])) == 3
    np.testing.assert_allclose(np.sum(density.data[0].marker.color), 1.0)


def _prepared_gaze():
    n_frames = 6
    bearing = np.linspace(-20.0, 20.0, n_frames, dtype=np.float32)
    center = np.asarray([100.0, 100.0], dtype=np.float64)
    radians = np.deg2rad(bearing.astype(np.float64))
    chaser_xy = center + np.column_stack(
        (100.0 * np.cos(radians), -100.0 * np.sin(radians))
    )
    return prepare_gaze_tracking_successor(
        GazeTrackingInput(
            recording_id="recording-1",
            source_relative_frame_run_path=(
                "analysis/chaser_relative_frame_runs/keypoint-v1"
            ),
            source_relative_frame_manifest_sha256="a" * 64,
            source_eye_run_path="analysis/eye_angle_runs/eye-v1",
            source_eye_manifest_sha256="b" * 64,
            source_eye_convention_receipt_sha256="c" * 64,
            source_eye_channel_policy="smoothed:left,right:vergence",
            source_semantic_selection_manifest_sha256="d" * 64,
            source_radial_run_path=("analysis/chaser_radial_near_field_runs/radial-v1"),
            source_radial_manifest_sha256="e" * 64,
            source_radial_payload_sha256="f" * 64,
            source_arena_geometry_and_scale={"authority_sha256": "1" * 64},
            arena_center_xy_px=center,
            arena_radius_px=200.0,
            arena_radius_mm=20.0,
            pixels_per_mm=10.0,
            n_frames=n_frames,
            n_chasers=1,
            acquisition_frame_id_by_frame=np.arange(10, 16, dtype=np.int64),
            timestamp_ns_by_frame=np.arange(n_frames, dtype=np.int64) * 100_000_000,
            timestamp_valid_by_frame=np.ones(n_frames, dtype=bool),
            semantic_role_code_by_frame=np.full(
                n_frames, ROLE_CODES["chaser_training"], dtype=np.uint8
            ),
            chaser_identity_code=np.ones(n_frames, dtype=np.uint16),
            fish_position_xy_px=np.broadcast_to(center, (n_frames, 2)).copy(),
            fish_position_valid=np.ones(n_frames, dtype=bool),
            chaser_position_xy_px=chaser_xy,
            chaser_position_valid=np.ones(n_frames, dtype=bool),
            chaser_occurrence_member=np.ones(n_frames, dtype=bool),
            body_origin_xy_px=np.broadcast_to(center, (n_frames, 2)).copy(),
            body_forward_axis_xy=np.tile([1.0, 0.0], (n_frames, 1)),
            body_left_axis_xy=np.tile([0.0, -1.0], (n_frames, 1)),
            body_axes_valid=np.ones(n_frames, dtype=bool),
            distance_mm=np.full(n_frames, 10.0, dtype=np.float32),
            distance_valid=np.ones(n_frames, dtype=bool),
            chaser_bearing_deg=bearing,
            chaser_bearing_valid=np.ones(n_frames, dtype=bool),
            gaze_signed_deg=np.column_stack((bearing, bearing + 5.0)),
            gaze_valid=np.ones((n_frames, 2), dtype=bool),
            vergence_deg=np.full(n_frames, 10.0, dtype=np.float32),
            vergence_valid=np.ones(n_frames, dtype=bool),
            minimum_regression_samples=3,
        )
    )


def test_gaze_display_helpers_are_bounded_and_normalized() -> None:
    np.testing.assert_array_equal(
        _uniform_indices(np.arange(10), maximum=4), np.asarray([0, 3, 6, 9])
    )
    _centers, probability, counts = _histogram_probability(np.asarray([-2.0, 2.0, 2.0]))
    assert int(np.sum(counts)) == 3
    np.testing.assert_allclose(np.sum(probability), 1.0)


def test_exact_gaze_renderer_uses_persisted_rows_summaries_and_events() -> None:
    prepared = _prepared_gaze()

    class _Handle:
        scientific_manifest = prepared.manifest

        @staticmethod
        def array(name):
            return prepared.array(name)

        @staticmethod
        def require_verified_arrays(names):
            assert set(names) == set(prepared.arrays)

    projection = SimpleNamespace(
        gaze_tracking=_Handle(), provenance={}, recording_id="recording-1"
    )

    output = build_exact_gaze_tracking_output(_Mo, go, projection)

    assert len(output) == 7
    assert output[0]["kind"] == "info"
    scatter, error, summary, dynamic, controls, events = output[1:]
    assert len(scatter.data) == 3
    assert len(error.data) == 2
    assert len(summary.data) == 2
    assert len(dynamic.data) == 2
    assert len(controls.data) == 5
    assert len(events.data) == 2
    assert scatter.layout.meta["display_recipe"]["error_bin_width_deg"] == 5.0
    assert scatter.layout.meta["display_recipe"]["scientific_recomputation"] is False
