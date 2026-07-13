from __future__ import annotations

import numpy as np

from fisheye.analysis.heart_photometry_motion_controls import (
    integrate_gradient_displacement_control,
    resample_static_reference_control,
    tracking_feature_traces,
)
from fisheye.analysis.local_rostral_heartrate import LocalCoordinateDataset


def _moving_plane_dataset(*, frame_count: int = 80) -> LocalCoordinateDataset:
    yy, xx = np.mgrid[0:5, 0:5]
    pixel_xy = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float64)
    timestamps = np.arange(frame_count, dtype=np.float64) / 100.0
    offset = np.column_stack(
        [
            0.12 * np.sin(2.0 * np.pi * 2.0 * timestamps),
            0.08 * np.cos(2.0 * np.pi * 2.0 * timestamps),
        ]
    )
    source_xy = pixel_xy[None, :, :] + offset[:, None, :]
    traces = 90.0 + 2.0 * source_xy[:, :, 0] - 1.5 * source_xy[:, :, 1]
    displacement = np.zeros_like(source_xy)
    displacement[1:] = source_xy[1:] - source_xy[:-1]
    motion_prediction = 2.0 * displacement[:, :, 0] - 1.5 * displacement[:, :, 1]
    valid = np.ones(traces.shape, dtype=bool)
    frame_valid = np.ones(frame_count, dtype=bool)
    weights = np.full((frame_count, pixel_xy.shape[0], 4), 0.25, dtype=np.float64)
    nuisance = np.column_stack(
        [np.ones(frame_count, dtype=np.float64), np.zeros(frame_count, dtype=np.float64)]
    )
    uncertainty = np.ones(frame_count, dtype=np.float64)
    uncertainty[0] = 0.0
    return LocalCoordinateDataset(
        frame_indices=np.arange(frame_count, dtype=np.int64),
        timestamps_s=timestamps,
        traces=traces,
        pixel_xy=pixel_xy,
        pixel_valid=valid,
        frame_valid=frame_valid,
        source_xy=source_xy,
        bilinear_weights=weights,
        body_occupancy=np.ones_like(traces),
        eye_occupancy=np.zeros_like(traces),
        gradient_magnitude=np.full_like(traces, 2.5),
        motion_prediction=motion_prediction,
        nuisance_values=nuisance,
        nuisance_names=("constant", "zero"),
        image_shape_hw=(5, 5),
        administrative_boundary_distance_px=np.ones(pixel_xy.shape[0]),
        physical_boundary_distance_px=np.ones(pixel_xy.shape[0]),
        transform_uncertainty=uncertainty,
    ).validated()


def test_gradient_displacement_control_integrates_plane_motion_without_crossing_gap() -> None:
    dataset = _moving_plane_dataset(frame_count=30)
    frame_valid = np.asarray(dataset.frame_valid).copy()
    pixel_valid = np.asarray(dataset.pixel_valid).copy()
    traces = np.asarray(dataset.traces).copy()
    source_xy = np.asarray(dataset.source_xy).copy()
    weights = np.asarray(dataset.bilinear_weights).copy()
    motion = np.asarray(dataset.motion_prediction).copy()
    frame_valid[12] = False
    pixel_valid[12] = False
    traces[12] = np.nan
    source_xy[12] = np.nan
    weights[12] = np.nan
    motion[12] = np.nan
    dataset = LocalCoordinateDataset(
        **{
            **dataset.__dict__,
            "frame_valid": frame_valid,
            "pixel_valid": pixel_valid,
            "traces": traces,
            "source_xy": source_xy,
            "bilinear_weights": weights,
            "motion_prediction": motion,
        }
    ).validated()

    result = integrate_gradient_displacement_control(dataset)

    baseline = np.nanmedian(traces, axis=0)
    np.testing.assert_allclose(
        result.values[1:12] - result.values[0],
        traces[1:12] - traces[0],
        atol=1e-12,
    )
    assert not result.frame_valid[12]
    np.testing.assert_allclose(result.values[13], baseline, atol=1e-12)
    assert result.segment_index.shape == traces.shape
    assert result.reference_row.shape == traces.shape
    assert np.all(result.reference_row[0] == 0)
    assert np.all(result.reference_row[13] == 13)
    assert np.all(result.segment_index[0] != result.segment_index[13])
    assert result.diagnostics["provenance_axis"] == "time_by_pixel"
    assert result.diagnostics["interpretation"].endswith("not_optical_flow")


def test_static_reference_resampling_recovers_linear_surface_inside_hull() -> None:
    dataset = _moving_plane_dataset(frame_count=80)

    result = resample_static_reference_control(
        dataset,
        epoch_seconds=2.0,
        guard_seconds=0.0,
        minimum_template_pixels=8,
    )

    center_pixel = 2 * 5 + 2
    assert np.all(result.pixel_valid[:, center_pixel])
    np.testing.assert_allclose(
        result.values[:, center_pixel],
        dataset.traces[:, center_pixel],
        atol=1e-10,
    )
    assert result.segment_index.shape == dataset.traces.shape
    assert result.reference_row.shape == dataset.traces.shape
    assert np.all(result.reference_row[:, center_pixel] == 0)
    assert result.diagnostics["provenance_axis"] == "time_by_pixel"
    assert result.diagnostics["successful_epoch_count"] == 1
    assert result.diagnostics["interpretation"].endswith("not_optical_flow")


def test_tracking_features_report_measured_source_step_and_preserve_first_nan() -> None:
    dataset = _moving_plane_dataset(frame_count=30)
    region = np.ones(dataset.pixel_count, dtype=bool)

    features = tracking_feature_traces(dataset, region)

    assert np.isnan(features.source_step_px[0])
    expected = np.linalg.norm(dataset.source_xy[1, 0] - dataset.source_xy[0, 0])
    np.testing.assert_allclose(features.source_step_px[1], expected, atol=1e-12)
    assert np.all(features.valid_pixel_fraction == 1.0)
    np.testing.assert_allclose(
        features.abs_gradient_displacement[1],
        abs(dataset.motion_prediction[1, 0]),
        atol=1e-12,
    )
