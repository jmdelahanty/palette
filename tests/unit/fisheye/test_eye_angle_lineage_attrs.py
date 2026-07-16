from __future__ import annotations

import numpy as np
import pytest

import fisheye.analysis.eye_angle_analysis as eye_angle_analysis
from fisheye.analysis.eye_angle_analysis import _eye_angle_definition_attrs, _process_chunk
from fisheye.analysis.eye_angle_analysis import (
    _resolve_keypoint_run_name as resolve_eye_angle_keypoint_run,
)
from fisheye.shared.eye_geometry_source import (
    EYE_GEOMETRY_STAGE_REFINED_SUBJECT,
    EYE_GEOMETRY_STAGE_SUBJECT_SHAPE,
    resolve_eye_geometry_source,
)
from fisheye.shared.detect_reason_codec import decode_reason_bytes
from fisheye.shared.plot_artifacts import PNG_ARTIFACT_SCHEMA_ID
from fisheye.visualization.visualize_eye_angle_overlays import (
    _load_display_masks_and_geometry,
    _resolve_keypoint_run_name as resolve_overlay_keypoint_run,
)
from fisheye.visualization.visualize_eye_angles import (
    EYE_ANGLE_DASHBOARD_PLOT_SCHEMA_ID,
    EYE_ANGLE_DASHBOARD_RENDERER,
    _default_angle_source,
    _eye_angle_contract_metadata,
    _format_summary_lines,
    _select_angle_variant,
    _write_eye_angle_png_artifact,
)


def test_eye_angle_archive_opener_uses_palette_zarr_policy(monkeypatch, tmp_path) -> None:
    calls = []
    sentinel = object()

    def fake_open_zarr_root(path, *, mode):
        calls.append((path, mode))
        return sentinel

    zarr_path = tmp_path / "archive.zarr"
    monkeypatch.setattr(eye_angle_analysis, "open_zarr_root", fake_open_zarr_root)

    assert eye_angle_analysis._open_archive_for_eye_angle(zarr_path) is sentinel
    assert calls == [(zarr_path, "a")]


def test_eye_angle_layout_default_is_compact_dense_v2() -> None:
    parser = eye_angle_analysis.build_parser()
    args = parser.parse_args(["archive.zarr"])

    assert eye_angle_analysis.EYE_ANGLE_LAYOUT_DEFAULT == eye_angle_analysis.EYE_ANGLE_LAYOUT_COMPACT_DENSE_V2
    assert args.layout == eye_angle_analysis.EYE_ANGLE_LAYOUT_COMPACT_DENSE_V2


def test_eye_angle_definition_attrs_match_undirected_axis_vergence_math() -> None:
    attrs = _eye_angle_definition_attrs()

    assert attrs["signed_angles"] is True
    assert attrs["signed_angle_convention"] == "per-eye signed angles are body-frame anatomical-left-positive"
    assert attrs["canonical_eye_orientation_axis"] == "ellipse_major"
    assert attrs["canonical_eye_orientation_arrays"] == ["left_major_signed_deg", "right_major_signed_deg"]
    assert attrs["angle_zero"].startswith("major axis aligned")
    assert attrs["axis_ambiguity_resolution"].startswith("ellipse major axis is resolved")
    assert attrs["vergence_definition"] == "undirected_axis_separation(left_signed_deg, right_signed_deg)"
    assert attrs["vergence_signed_definition"] == "same as vergence_deg for directionless ellipse axes"
    assert attrs["version_definition"] == "0.5*(left_signed_deg + right_signed_deg)"
    assert attrs["major_vergence_definition"] == "undirected_axis_separation(left_major_signed_deg, right_major_signed_deg)"
    assert attrs["minor_signed_angles"] is True
    assert attrs["minor_signed_angle_convention"].startswith("per-eye minor/gaze signed angles")
    assert attrs["minor_vergence_definition"] == "undirected_axis_separation(left_minor_signed_deg, right_minor_signed_deg)"
    assert attrs["minor_vergence_signed_definition"] == "same as vergence_minor_deg for directionless ellipse axes"
    assert attrs["minor_version_definition"] == "0.5*(left_minor_signed_deg + right_minor_signed_deg)"
    assert attrs["preferred_angle_family"] == "gaze"
    assert attrs["preferred_eye_axis"] == "ellipse_major"
    assert attrs["gaze_angle_source"] == "ellipse_minor_derived_from_resolved_major_axis"
    assert attrs["gaze_angle_definition"].startswith("left/right_gaze_signed_deg are signed gaze")
    assert attrs["gaze_vergence_definition"] == "undirected_axis_separation(left_gaze_signed_deg, right_gaze_signed_deg)"
    assert attrs["gaze_vergence_signed_definition"] == "same as vergence_gaze_deg for directionless ellipse axes"
    assert attrs["gaze_total_vergence_definition"].startswith("vergence_gaze_deg retains the v3-compatible")
    assert attrs["nasal_gaze_definition"] == "90 - abs(outward_from_midline_gaze_axis_angle_deg)"
    assert attrs["mean_eye_vergence_gaze_definition"] == "0.5 * (left_nasal_gaze_deg + right_nasal_gaze_deg)"
    assert attrs["beast_comparable_eye_vergence"] == "mean_eye_vergence_gaze_deg"
    assert attrs["body_frame_schema_id"] == "fish_anatomical_body_frame"


def test_eye_angle_output_schema_describes_run_layout_and_conventions() -> None:
    schema = eye_angle_analysis._eye_angle_output_schema()

    assert schema["schema_id"] == "analysis.eye_angle_output_schema"
    assert schema["schema_version"] == 8
    assert schema["algorithm_contract"] == {
        "schema_id": "analysis.eye_angle_algorithm_contract",
        "schema_version": 1,
        "run_attr": "eye_angle_algorithm_contract",
    }
    assert schema["temporal_operators"]["delta"] == "absolute_adjacent_finite_difference"
    assert schema["variant_schema"]["schema_id"] == "analysis.eye_angle_variant_schema"
    assert schema["variant_schema"]["schema_version"] == 1
    assert schema["variant_schema"]["default_representation"] == "eye_frame"
    assert schema["variant_schema"]["representation_order"] == [
        "eye_frame",
        "gaze",
        "nasal_gaze",
        "major",
        "centroid",
        "legacy",
    ]
    representations = schema["variant_schema"]["representations"]
    assert representations["major"]["role"] == "canonical_geometry"
    assert representations["eye_frame"]["role"] == "biological_presentation"
    assert representations["eye_frame"]["primary_roi_fields"] == ["left_eye_angle_deg", "right_eye_angle_deg"]
    assert representations["eye_frame"]["aggregate_roi_fields"] == ["vergence_eye_angle_deg"]
    assert representations["gaze"]["vector_roi_fields"] == ["left_gaze_xy", "right_gaze_xy"]
    assert representations["nasal_gaze"]["aggregate_roi_fields"] == ["mean_eye_vergence_gaze_deg"]
    assert representations["legacy"]["alias_targets"]["left_minor_signed_deg"] == "left_gaze_signed_deg"
    fields = schema["variant_schema"]["fields"]
    assert fields["left_eye_angle_deg"]["representation"] == "eye_frame"
    assert fields["left_gaze_xy"]["representation"] == "gaze"
    assert fields["mean_eye_vergence_gaze_deg"]["representation"] == "nasal_gaze"
    assert schema["row_axes"]["roi"] == "keypoint_detection_rows"
    assert schema["row_axes"]["frame"] == "video_frame_rows"
    assert schema["groups"]["angles/roi"]["units"] == "deg"
    assert "left_signed_deg" in schema["groups"]["angles/roi"]["base_outputs"]
    assert "left_major_signed_deg" in schema["groups"]["angles/roi"]["base_outputs"]
    assert "right_major_signed_deg" in schema["groups"]["angles/roi"]["base_outputs"]
    assert "left_eye_angle_deg" in schema["groups"]["angles/roi"]["base_outputs"]
    assert "right_eye_angle_deg" in schema["groups"]["angles/roi"]["base_outputs"]
    assert "vergence_eye_angle_deg" in schema["groups"]["angles/roi"]["base_outputs"]
    assert "vergence_major_signed_deg" in schema["groups"]["angles/roi"]["base_outputs"]
    assert "version_major_deg" in schema["groups"]["angles/roi"]["base_outputs"]
    assert "left_gaze_signed_deg" in schema["groups"]["angles/roi"]["base_outputs"]
    assert "vergence_gaze_signed_deg" in schema["groups"]["angles/roi"]["base_outputs"]
    assert "left_nasal_gaze_deg" in schema["groups"]["angles/roi"]["base_outputs"]
    assert "right_nasal_gaze_deg" in schema["groups"]["angles/roi"]["base_outputs"]
    assert "mean_eye_vergence_gaze_deg" in schema["groups"]["angles/roi"]["base_outputs"]
    assert {"name": "left_gaze_xy", "shape": ["N", 2], "value_kind": "unit_vector_xy_roi"} in schema[
        "groups"
    ]["angles/roi"]["vector_outputs"]
    assert "mean_eye_vergence_gaze_speed_deg_s" in schema["groups"]["angles/roi"]["derivative_outputs"]
    assert "left_gaze_signed_deg" in schema["groups"]["angles/frame"]["base_outputs"]
    assert "right_gaze_signed_deg" in schema["groups"]["angles/frame"]["base_outputs"]
    assert "left_eye_angle_deg" in schema["groups"]["angles/frame"]["base_outputs"]
    assert "right_eye_angle_deg" in schema["groups"]["angles/frame"]["base_outputs"]
    assert "vergence_eye_angle_deg" in schema["groups"]["angles/frame"]["base_outputs"]
    assert "mean_eye_vergence_gaze_deg" in schema["groups"]["angles/frame"]["base_outputs"]
    assert "heading_deg" in schema["groups"]["angles/roi"]["base_outputs"]
    assert "left_speed_deg_s" in schema["groups"]["angles/roi"]["derivative_outputs"]
    assert "left_gaze_speed_deg_s" in schema["groups"]["angles/roi"]["derivative_outputs"]
    assert schema["groups"]["support"]["row_axis"] == "mixed"
    support_outputs = schema["groups"]["support"]["outputs"]
    assert {"name": "frame_time_seconds", "row_axis": "frame", "units": "s", "optional": True} in support_outputs
    assert schema["groups"]["qa/roi"]["outputs"] == [
        "valid_left",
        "valid_right",
        "valid_frame",
        "reason_codes",
        "left_major_axis_marginal",
        "right_major_axis_marginal",
        "major_axis_marginal",
    ]
    assert schema["signed_angle_convention"] == "per-eye signed angles are body-frame anatomical-left-positive"
    assert schema["canonical_eye_orientation_axis"] == "ellipse_major"
    assert schema["vergence_signed_definition"] == "same as vergence_deg for directionless ellipse axes"
    assert schema["eye_frame_angles"] is True
    assert schema["eye_frame_angle_convention"].startswith("left/right_eye_angle_deg are eye-frame")
    assert schema["vergence_eye_angle_definition"].startswith("vergence_eye_angle_deg = left_eye_angle_deg")
    assert schema["preferred_angle_family"] == "gaze"
    assert schema["preferred_eye_axis"] == "ellipse_major"
    assert schema["gaze_angle_source"] == "ellipse_minor_derived_from_resolved_major_axis"
    assert schema["gaze_vergence_signed_definition"] == "same as vergence_gaze_deg for directionless ellipse axes"
    assert schema["gaze_total_vergence_definition"].startswith("vergence_gaze_deg retains the v3-compatible")
    assert schema["mean_eye_vergence_gaze_definition"] == "0.5 * (left_nasal_gaze_deg + right_nasal_gaze_deg)"
    assert schema["beast_comparable_eye_vergence"] == "mean_eye_vergence_gaze_deg"
    assert schema["body_frame_group"] == "support/body_frame"
    assert schema["qa_reason_codes_attr"] == "reason_code_map"


def test_eye_angle_visualizer_minor_signed_variant_uses_signed_eye_traces() -> None:
    roi_angles = {
        "left_minor_signed": np.asarray([-20.0, -10.0], dtype=np.float32),
        "left_minor_signed_smoothed": np.asarray([-18.0, -12.0], dtype=np.float32),
        "right_minor_signed": np.asarray([15.0, 25.0], dtype=np.float32),
        "right_minor_signed_smoothed": np.asarray([14.0, 24.0], dtype=np.float32),
        "vergence_minor_signed": np.asarray([35.0, 35.0], dtype=np.float32),
        "version_minor": np.asarray([-2.5, 7.5], dtype=np.float32),
    }
    roi_deltas = {
        "left_minor_signed_delta_deg_smoothed": np.asarray([0.0, 6.0], dtype=np.float32),
        "right_minor_signed_delta_deg_smoothed": np.asarray([0.0, 10.0], dtype=np.float32),
        "vergence_minor_signed_delta_deg": np.asarray([0.0, 0.0], dtype=np.float32),
    }

    variant, label, meta = _select_angle_variant(roi_angles, roi_deltas, "minor_signed")

    assert label == "Ellipse minor axis signed (smoothed)"
    np.testing.assert_allclose(variant["left"], [-18.0, -12.0])
    np.testing.assert_allclose(variant["right"], [14.0, 24.0])
    assert meta["series_lookup"]["left"] == "left_minor_signed"
    assert meta["presentation"]["signed_eye_traces"] is True


def test_eye_angle_visualizer_eye_frame_variant_uses_bianco_angles() -> None:
    roi_angles = {
        "left_eye_angle": np.asarray([20.0, 22.0], dtype=np.float32),
        "left_eye_angle_smoothed": np.asarray([19.0, 21.0], dtype=np.float32),
        "right_eye_angle": np.asarray([18.0, 20.0], dtype=np.float32),
        "right_eye_angle_smoothed": np.asarray([17.0, 19.0], dtype=np.float32),
        "vergence_eye_angle": np.asarray([38.0, 42.0], dtype=np.float32),
        "vergence_eye_angle_smoothed": np.asarray([36.0, 40.0], dtype=np.float32),
    }
    roi_deltas = {
        "left_eye_angle_smoothed": np.asarray([0.0, 2.0], dtype=np.float32),
        "right_eye_angle_smoothed": np.asarray([0.0, 2.0], dtype=np.float32),
        "vergence_eye_angle_smoothed": np.asarray([0.0, 4.0], dtype=np.float32),
    }

    variant, label, meta = _select_angle_variant(roi_angles, roi_deltas, "eye_frame")

    assert label == "Eye-frame nasal-positive angles (Bianco/Engert) (smoothed)"
    np.testing.assert_allclose(variant["left"], [19.0, 21.0])
    np.testing.assert_allclose(variant["right"], [17.0, 19.0])
    np.testing.assert_allclose(variant["vergence"], [36.0, 40.0])
    assert meta["series_lookup"]["left"] == "left_eye_angle"
    assert meta["series_lookup"]["vergence"] == "vergence_eye_angle"
    np.testing.assert_allclose(meta["deltas"]["left_eye_angle"], [0.0, 2.0])
    assert meta["presentation"]["signed_eye_traces"] is True
    assert meta["presentation"]["signed_y_range"] == (-185.0, 185.0)
    assert meta["presentation"]["vergence_y_range"] == (-185.0, 185.0)


def test_eye_angle_dashboard_default_source_uses_variant_schema() -> None:
    assert _default_angle_source(
        {
            "eye_angle_variant_schema": {
                "default_representation": "eye_frame",
            }
        }
    ) == "eye_frame"
    assert _default_angle_source({}) == "gaze"


def test_eye_angle_source_geometry_kind_maps_known_sources() -> None:
    assert eye_angle_analysis._source_geometry_kind("analysis/subject_shape_runs") == "subject_shape_eye_geometry"
    assert eye_angle_analysis._source_geometry_kind("refined_subject_masks_runs") == "refined_subject_eye_geometry"
    assert eye_angle_analysis._source_geometry_kind("refined_eye_masks_runs") == "unknown_eye_geometry"
    assert eye_angle_analysis._source_geometry_kind("unknown") == "unknown_eye_geometry"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("serial", "serial_driver"),
        ("driver", "serial_driver"),
        ("dask", "dask_worker_chunks"),
        ("dask-chunks", "dask_worker_chunks"),
    ],
)
def test_eye_angle_execution_backend_aliases(raw: str, expected: str) -> None:
    assert eye_angle_analysis._normalize_execution_backend(raw) == expected


def test_eye_angle_frame_projection_flags_missing_and_multi_detection_frames() -> None:
    frame_arrays, frame_valid, frame_reason = eye_angle_analysis._project_detection_arrays_to_frames(
        np.asarray([0, 2, 2, 4], dtype=np.int64),
        num_frames=5,
        valid_frame=np.asarray([True, True, True, False], dtype=bool),
        reason_codes=np.asarray([0, 4, 8, 2], dtype=np.uint16),
        arrays={"left": np.asarray([10.0, 20.0, 30.0, 40.0], dtype=np.float32)},
    )

    assert frame_arrays["left"][0] == 10.0
    assert np.isnan(frame_arrays["left"][1])
    assert np.isnan(frame_arrays["left"][2])
    assert np.isnan(frame_arrays["left"][3])
    assert frame_arrays["left"][4] == 40.0
    assert frame_valid.tolist() == [True, False, False, False, False]
    assert int(frame_reason[1]) & int(eye_angle_analysis.REASON_NO_DETECTION)
    assert int(frame_reason[2]) & int(eye_angle_analysis.REASON_MULTI_DETECTION)
    assert int(frame_reason[3]) & int(eye_angle_analysis.REASON_NO_DETECTION)
    assert int(frame_reason[4]) & int(eye_angle_analysis.REASON_HEADING_INVALID)


def test_eye_angle_channel_formulas_describe_actual_temporal_operators() -> None:
    assert eye_angle_analysis._formula_for_angle_channel("left_deg_smoothed") == (
        "nan_aware_centered_boxcar(source_channel)"
    )
    assert eye_angle_analysis._formula_for_angle_channel("left_delta_deg") == (
        "abs(source_channel[row] - source_channel[row - 1])"
    )
    assert eye_angle_analysis._formula_for_angle_channel("left_delta_deg_smoothed") == (
        "abs(smoothed_source_channel[row] - smoothed_source_channel[row - 1])"
    )
    assert eye_angle_analysis._formula_for_angle_channel("left_speed_deg_s") == (
        "backward_difference_to_previous_valid(source_channel, time_seconds)"
    )


def _add_refined_subject_eye_geometry(root):
    parent = root.create_group("refined_subject_masks_runs")
    parent.attrs["latest"] = "refined_001"
    run = parent.create_group("refined_001")
    run.attrs.update(
        {
            "mask_labels": ["subject_body", "eye_left", "eye_right"],
            "source_keypoints_run": "kp_from_refined",
        }
    )
    run.create_array("masks_roi", data=np.zeros((2, 3, 4, 4), dtype=np.uint8), overwrite=True)
    for component in ("eye_left", "eye_right"):
        geometry = run.create_group(f"components/{component}/geometry")
        geometry.create_array("ellipse_params", data=np.ones((2, 5), dtype=np.float32), overwrite=True)
        geometry.create_array("ellipse_success", data=np.ones((2,), dtype=bool), overwrite=True)
    metrics = run.create_group("relations/eye_pair/metrics")
    metrics.create_array("separation_px", data=np.asarray([4.0, 4.5], dtype=np.float32), overwrite=True)
    return run


def _add_subject_shape_eye_geometry(root):
    analysis = root.create_group("analysis")
    parent = analysis.create_group("subject_shape_runs")
    parent.attrs["latest"] = "shape_001"
    run = parent.create_group("shape_001")
    run.attrs.update(
        {
            "source_refined_subject_masks_run": "refined_001",
            "source_keypoints_run": "kp_from_shape",
        }
    )
    for component, value in (("eye_left", 1.0), ("eye_right", 2.0)):
        group = run.create_group(f"components/{component}")
        group.create_array("ellipse_params", data=np.full((2, 5), value, dtype=np.float32), overwrite=True)
        group.create_array("ellipse_success", data=np.ones((2,), dtype=bool), overwrite=True)
    pair = run.create_group("relations/eye_pair")
    pair.create_array("separation_px", data=np.asarray([5.0, 5.5], dtype=np.float32), overwrite=True)
    return run


def test_eye_angle_algorithm_contract_records_resolved_sources_and_exact_methods(
    tmp_path,
) -> None:
    import zarr

    zarr_path = tmp_path / "eye-contract.zarr"
    root = zarr.open_group(str(zarr_path), mode="w", zarr_format=3)
    shape = _add_subject_shape_eye_geometry(root)
    shape.attrs.update(
        {
            "schema_id": "analysis.subject_shape_runs",
            "schema_version": 3,
            "method": "subject_shape",
            "method_version": "subject_shape.v3",
            "source_fingerprint": "shape-fingerprint",
        }
    )
    for component in ("eye_left", "eye_right"):
        shape[f"components/{component}"].attrs["ellipse_method"] = (
            "cv2.fitEllipse_component_contour_v1"
        )

    refined_parent = root.create_group("refined_keypoints_runs")
    refined_parent.attrs["latest"] = "kp_from_shape"
    refined = refined_parent.create_group("kp_from_shape")
    refined.attrs.update(
        {
            "schema_id": "refined_keypoints",
            "schema_version": 4,
            "method": "manual_plus_model_refinement",
            "method_version": "refined_keypoints.v4",
            "source_keypoints_run": "raw_001",
            "source_lineage_hash": "refined-lineage",
            "keypoint_labels": ["swim_bladder", "eye_left", "eye_right"],
        }
    )
    refined.create_array(
        "keypoints_roi",
        data=np.zeros((2, 3, 2), dtype=np.float32),
        overwrite=True,
    )
    refined.create_array("heading", data=np.zeros(2, dtype=np.float32), overwrite=True)
    refined.create_array("refined_success", data=np.ones(2, dtype=bool), overwrite=True)

    raw_parent = root.create_group("keypoints_runs")
    raw = raw_parent.create_group("raw_001")
    raw.attrs.update(
        {
            "schema_id": "keypoints",
            "schema_version": 2,
            "method": "yolo_pose",
            "method_version": "detector.v2",
            "lineage_hash": "raw-lineage",
        }
    )
    raw.create_array("detection_success", data=np.ones(2, dtype=bool), overwrite=True)
    raw.create_array("frame_indices", data=np.asarray([4, 5], dtype=np.int64), overwrite=True)

    context = eye_angle_analysis._resolve_eye_angle_inputs(
        root,
        subject_shape_run="shape_001",
        refined_subject_run=None,
        keypoint_run="kp_from_shape",
    )
    sources = eye_angle_analysis._eye_angle_source_contracts(context)
    contract = eye_angle_analysis._eye_angle_algorithm_contract(
        context,
        fps=200.0,
        fps_source="recording_metadata",
        smoothing_window_requested=7,
        smoothing_window_source="module_default",
        detection_smoothing_window=7,
        frame_smoothing_window=5,
    )

    assert sources["eye_geometry"]["path"] == "analysis/subject_shape_runs/shape_001"
    assert sources["eye_geometry"]["source_fingerprint"] == "shape-fingerprint"
    assert sources["eye_geometry"]["components"][0]["ellipse_source_contract"] == {
        "ellipse_method": "cv2.fitEllipse_component_contour_v1"
    }
    assert sources["refined_keypoints"]["source_lineage_hash"] == "refined-lineage"
    assert sources["source_keypoints"]["lineage_hash"] == "raw-lineage"
    assert sources["resolved_arrays"] == {
        "keypoints_roi": "refined_keypoints_runs/kp_from_shape/keypoints_roi",
        "heading": "refined_keypoints_runs/kp_from_shape/heading",
        "detection_success": "refined_keypoints_runs/kp_from_shape/refined_success",
        "frame_indices": "keypoints_runs/raw_001/frame_indices",
    }
    assert contract["schema_id"] == "analysis.eye_angle_algorithm_contract"
    assert contract["ellipse_input"]["parameter_order"][-1] == "major_axis_angle_deg"
    assert contract["ellipse_input"]["circularity_reject_condition"] == (
        "ellipse_ratio > 0.95"
    )
    assert contract["body_frame"]["resolved_keypoint_indices"] == {
        "swim_bladder": 0,
        "eye_left": 1,
        "eye_right": 2,
    }
    assert contract["smoothing"]["method"] == (
        "nan_aware_centered_boxcar_finite_count_normalized"
    )
    assert contract["delta"]["method"] == "absolute_adjacent_finite_difference"
    assert contract["derivative"]["maximum_dt_seconds"] == 0.25
    assert contract["frame_projection"]["multiple_detection_rule"].startswith(
        "leave values NaN"
    )

    args = eye_angle_analysis.build_parser().parse_args(
        [
            str(zarr_path),
            "--subject-shape-run",
            "shape_001",
            "--keypoint-run",
            "kp_from_shape",
            "--run-name",
            "eye_contract_001",
            "--fps",
            "200",
            "--chunk-size",
            "2",
            "--smoothing-window",
            "3",
            "--quiet",
        ]
    )
    eye_angle_analysis.run(args)

    persisted_root = zarr.open_group(
        str(zarr_path),
        mode="r",
        use_consolidated=False,
    )
    persisted = persisted_root["analysis/eye_angle_runs/eye_contract_001"]
    assert persisted.attrs["eye_angle_output_schema"]["schema_version"] == 8
    assert persisted.attrs["eye_angle_algorithm_contract"]["schema_version"] == 1
    assert persisted.attrs["eye_angle_source_contracts"]["resolved_arrays"] == (
        sources["resolved_arrays"]
    )
    assert persisted.attrs["angle_derivative_max_dt_seconds"] == 0.25
    assert persisted.attrs["provenance"]["algorithm_contract"]["delta"]["method"] == (
        "absolute_adjacent_finite_difference"
    )


def test_eye_geometry_resolution_prefers_latest_subject_shape_when_enabled() -> None:
    import zarr

    root = zarr.group()
    _add_refined_subject_eye_geometry(root)
    _add_subject_shape_eye_geometry(root)

    source = resolve_eye_geometry_source(root, prefer_subject_shape=True)

    assert source.stage_group == EYE_GEOMETRY_STAGE_SUBJECT_SHAPE
    assert source.run_name == "shape_001"
    assert source.source_subject_shape_run == "shape_001"
    assert source.source_refined_subject_run == "refined_001"
    assert source.ellipse_params.shape == (2, 2, 5)
    assert source.ellipse_success.shape == (2, 2)
    np.testing.assert_allclose(source.ellipse_params[:, 0, 0], [1.0, 1.0])
    np.testing.assert_allclose(source.ellipse_params[:, 1, 0], [2.0, 2.0])


def test_eye_geometry_resolution_default_keeps_mask_capable_source() -> None:
    import zarr

    root = zarr.group()
    _add_refined_subject_eye_geometry(root)
    _add_subject_shape_eye_geometry(root)

    source = resolve_eye_geometry_source(root)

    assert source.stage_group == EYE_GEOMETRY_STAGE_REFINED_SUBJECT
    assert source.run_name == "refined_001"
    assert source.masks_roi is not None


def test_eye_geometry_resolution_honors_explicit_refined_subject_run() -> None:
    import zarr

    root = zarr.group()
    _add_refined_subject_eye_geometry(root)
    _add_subject_shape_eye_geometry(root)

    source = resolve_eye_geometry_source(root, refined_subject_run="refined_001")

    assert source.stage_group == EYE_GEOMETRY_STAGE_REFINED_SUBJECT
    assert source.run_name == "refined_001"
    assert source.source_subject_shape_run is None


def test_eye_angle_keypoint_resolution_prefers_explicit() -> None:
    resolved = resolve_eye_angle_keypoint_run(
        explicit_keypoint_run="kp_explicit",
        refined_attrs={
            "source_keypoints_run": "kp_canonical",
            "source_keypoint_run": "kp_legacy",
        },
        parent_latest="kp_latest",
    )
    assert resolved == "kp_explicit"


def test_eye_angle_keypoint_resolution_prefers_canonical_over_legacy() -> None:
    resolved = resolve_eye_angle_keypoint_run(
        explicit_keypoint_run=None,
        refined_attrs={
            "source_keypoints_run": "kp_canonical",
            "source_keypoint_run": "kp_legacy",
        },
        parent_latest="kp_latest",
    )
    assert resolved == "kp_canonical"


def test_eye_angle_keypoint_resolution_falls_back_to_legacy_then_latest() -> None:
    resolved_legacy = resolve_eye_angle_keypoint_run(
        explicit_keypoint_run=None,
        refined_attrs={"source_keypoint_run": "kp_legacy"},
        parent_latest="kp_latest",
    )
    resolved_latest = resolve_eye_angle_keypoint_run(
        explicit_keypoint_run=None,
        refined_attrs={},
        parent_latest="kp_latest",
    )
    assert resolved_legacy == "kp_legacy"
    assert resolved_latest == "kp_latest"


def test_overlay_keypoint_resolution_prefers_explicit() -> None:
    resolved = resolve_overlay_keypoint_run(
        explicit_keypoint_run="kp_explicit",
        run_attrs={
            "source_keypoints_run": "kp_canonical",
            "source_keypoint_run": "kp_legacy",
        },
    )
    assert resolved == "kp_explicit"


def test_overlay_keypoint_resolution_prefers_canonical_over_legacy() -> None:
    resolved = resolve_overlay_keypoint_run(
        explicit_keypoint_run=None,
        run_attrs={
            "source_keypoints_run": "kp_canonical",
            "source_keypoint_run": "kp_legacy",
        },
    )
    assert resolved == "kp_canonical"


def test_overlay_keypoint_resolution_falls_back_to_legacy() -> None:
    resolved = resolve_overlay_keypoint_run(
        explicit_keypoint_run=None,
        run_attrs={"source_keypoint_run": "kp_legacy"},
    )
    assert resolved == "kp_legacy"


def test_eye_angle_dashboard_contract_metadata_prefers_canonical_keypoint_attr() -> None:
    metadata = _eye_angle_contract_metadata(
        {
            "schema_id": "analysis.eye_angle_runs",
            "schema_version": 4,
            "method": "ellipse_and_centroid_eye_angles",
            "row_axis": "keypoint_detection_rows",
            "source_geometry_kind": "subject_shape_eye_geometry",
            "source_eye_geometry_stage": "analysis/subject_shape_runs",
            "source_eye_geometry_run": "shape_001",
            "source_keypoints_run": "kp_canonical",
            "source_keypoint_run": "kp_legacy",
            "preferred_angle_family": "gaze",
            "preferred_eye_axis": "ellipse_minor",
            "gaze_angle_source": "ellipse_minor",
            "eye_angle_output_schema": {"schema_id": "analysis.eye_angle_output_schema"},
        }
    )

    assert metadata["schema_id"] == "analysis.eye_angle_runs"
    assert metadata["schema_version"] == 4
    assert metadata["source_geometry_kind"] == "subject_shape_eye_geometry"
    assert metadata["source_keypoints_run"] == "kp_canonical"
    assert metadata["preferred_angle_family"] == "gaze"
    assert metadata["preferred_eye_axis"] == "ellipse_minor"
    assert metadata["eye_angle_output_schema"] == {"schema_id": "analysis.eye_angle_output_schema"}


def test_eye_angle_dashboard_summary_includes_schema_and_lineage() -> None:
    summary = _format_summary_lines(
        {
            "run_name": "eye_angle_001",
            "schema_id": "analysis.eye_angle_runs",
            "schema_version": 4,
            "method": "ellipse_and_centroid_eye_angles",
            "preferred_angle_family": "gaze",
            "preferred_eye_axis": "ellipse_minor",
            "source_geometry_kind": "subject_shape_eye_geometry",
            "source_eye_geometry_stage": "analysis/subject_shape_runs",
            "source_eye_geometry_run": "shape_001",
            "source_keypoints_run": "kp_canonical",
            "num_detections": 2,
        },
        counts={},
        roi_valid=np.asarray([True, False]),
        frame_valid=None,
    )

    assert "Schema: analysis.eye_angle_runs v4" in summary
    assert "Method: ellipse_and_centroid_eye_angles" in summary
    assert "Preferred eye angle: gaze (ellipse_minor)" in summary
    assert "Eye geometry: subject_shape_eye_geometry (analysis/subject_shape_runs / shape_001)" in summary
    assert "Keypoints: kp_canonical" in summary


def test_eye_angle_dashboard_zarr_artifact_manifest(tmp_path) -> None:
    import zarr

    zarr_path = tmp_path / "archive.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    analysis = root.create_group("analysis")
    parent = analysis.create_group("eye_angle_runs")
    parent.attrs["latest"] = "eye_angle_001"
    run = parent.create_group("eye_angle_001")
    attrs = {
        "run_name": "eye_angle_001",
        "schema_id": "analysis.eye_angle_runs",
        "schema_version": 4,
        "method": "ellipse_and_centroid_eye_angles",
        "row_axis": "keypoint_detection_rows",
        "source_geometry_kind": "subject_shape_eye_geometry",
        "source_eye_geometry_stage": "analysis/subject_shape_runs",
        "source_eye_geometry_run": "shape_001",
        "source_keypoints_run": "kp_canonical",
        "preferred_angle_family": "gaze",
        "preferred_eye_axis": "ellipse_minor",
        "gaze_angle_source": "ellipse_minor",
    }
    run.attrs.update(attrs)

    artifact_path = _write_eye_angle_png_artifact(
        zarr_path=zarr_path,
        run_group=run,
        attrs=attrs,
        angle_source="ellipse",
        variant_label="Ellipse major axis",
        png_bytes=b"\x89PNG\r\n\x1a\nfake-png",
        generated_at_utc="2026-04-28T00:00:00+00:00",
        artifact_dpi=150,
        command="test-command",
    )

    assert artifact_path == "visualizations/eye_angle_dashboard_ellipse_png"
    artifact = run["visualizations"]["eye_angle_dashboard_ellipse_png"]
    assert bytes(np.asarray(artifact[:], dtype=np.uint8)[:8]) == b"\x89PNG\r\n\x1a\n"
    assert artifact.attrs["artifact_schema_id"] == PNG_ARTIFACT_SCHEMA_ID
    assert artifact.attrs["plot_schema_id"] == EYE_ANGLE_DASHBOARD_PLOT_SCHEMA_ID
    assert artifact.attrs["renderer"] == EYE_ANGLE_DASHBOARD_RENDERER
    assert artifact.attrs["visualization_contract_id"] == (
        "palette.core.eye_angles.summary.v1"
    )
    assert artifact.attrs["renderer_version"] == "1"
    assert artifact.attrs["parameters"]["angle_source"] == "ellipse"
    assert artifact.attrs["parameters"]["artifact_dpi"] == 150
    assert artifact.attrs["source_paths"]["angles_roi"].endswith("/angles/roi")
    assert artifact.attrs["source_runs"]["source_keypoints_run"] == "kp_canonical"
    assert artifact.attrs["eye_angle_contract"]["source_geometry_kind"] == "subject_shape_eye_geometry"
    assert artifact.attrs["eye_angle_contract"]["preferred_angle_family"] == "gaze"
    assert artifact.attrs["provenance"]["stage"] == "eye_angle_visualization"
    assert artifact.attrs["provenance"]["command"] == "test-command"
    assert artifact.attrs["provenance"]["artifacts"]["png_artifact"] == artifact_path

    manifest = run.attrs["visualizations"]
    assert manifest["eye_angle_dashboard_ellipse_png"]["path"] == artifact_path
    assert manifest["eye_angle_dashboard_ellipse_png"]["artifact_schema_id"] == PNG_ARTIFACT_SCHEMA_ID


def test_eye_angle_overlay_draws_vectors_from_actual_subject_shape_geometry() -> None:
    import zarr

    root = zarr.group()
    _add_refined_subject_eye_geometry(root)
    _add_subject_shape_eye_geometry(root)

    masks, ellipse_params = _load_display_masks_and_geometry(
        root,
        run_attrs={
            "source_eye_geometry_stage": EYE_GEOMETRY_STAGE_SUBJECT_SHAPE,
            "source_eye_geometry_run": "shape_001",
            "source_refined_subject_masks_run": "refined_001",
        },
        refined_subject_run="refined_001",
    )

    assert masks.shape == (2, 2, 4, 4)
    assert ellipse_params.shape == (2, 2, 5)
    np.testing.assert_allclose(ellipse_params[:, 0, 0], [1.0, 1.0])
    np.testing.assert_allclose(ellipse_params[:, 1, 0], [2.0, 2.0])


def test_eye_angle_chunk_uses_label_resolved_indices() -> None:
    ellipse_params = np.asarray(
        [[[3.0, 0.0, 4.0, 1.5, 0.0], [3.0, 2.0, 4.0, 1.5, 0.0]]],
        dtype=np.float32,
    )
    ellipse_success = np.asarray([[True, True]], dtype=bool)
    keypoints_roi = np.asarray(
        [
            [
                [3.0, 0.0],   # eye_left
                [9.0, 9.0],   # extra label
                [1.0, 1.0],   # swim_bladder
                [3.0, 2.0],   # eye_right
                [0.0, 0.0],   # extra label
            ]
        ],
        dtype=np.float32,
    )
    heading_deg = np.asarray([0.0], dtype=np.float32)
    detection_success = np.asarray([True], dtype=bool)

    result = _process_chunk(
        ellipse_params=ellipse_params,
        ellipse_success=ellipse_success,
        keypoints_roi=keypoints_roi,
        heading_deg=heading_deg,
        detection_success=detection_success,
        keypoint_indices={
            "swim_bladder": 2,
            "eye_left": 0,
            "eye_right": 3,
        },
    )

    assert bool(result.valid_left[0])
    assert bool(result.valid_right[0])
    assert bool(result.valid_frame[0])
    assert np.isfinite(result.left_deg[0])
    assert np.isfinite(result.right_deg[0])
    assert result.left_gaze_signed_deg[0] == result.left_minor_signed_deg[0]
    assert result.right_gaze_signed_deg[0] == result.right_minor_signed_deg[0]
    assert result.vergence_gaze_signed_deg[0] == result.vergence_minor_signed_deg[0]
    assert result.version_gaze_deg[0] == result.version_minor_deg[0]
    assert result.left_gaze_deg[0] == pytest.approx(abs(result.left_minor_signed_deg[0]))
    assert result.body_frame_valid.tolist() == [True]
    np.testing.assert_allclose(result.body_frame_origin_xy[0], [3.0, 1.0])


def test_eye_angle_chunk_computes_gaze_convergence_in_body_frame() -> None:
    ellipse_params = np.asarray(
        [
            [
                [2.0, -1.0, 4.0, 1.5, 110.0],  # left gaze signed -20 deg
                [2.0, 1.0, 4.0, 1.5, 70.0],    # right gaze signed +20 deg
            ]
        ],
        dtype=np.float32,
    )
    ellipse_success = np.asarray([[True, True]], dtype=bool)
    keypoints_roi = np.asarray(
        [
            [
                [0.0, 0.0],   # swim_bladder
                [2.0, -1.0],  # eye_left
                [2.0, 1.0],   # eye_right
            ]
        ],
        dtype=np.float32,
    )

    result = _process_chunk(
        ellipse_params=ellipse_params,
        ellipse_success=ellipse_success,
        keypoints_roi=keypoints_roi,
        heading_deg=np.asarray([0.0], dtype=np.float32),
        detection_success=np.asarray([True], dtype=bool),
        keypoint_indices={
            "swim_bladder": 0,
            "eye_left": 1,
            "eye_right": 2,
        },
    )

    assert bool(result.valid_frame[0])
    np.testing.assert_allclose(result.body_frame_forward_axis_xy[0], [1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(result.body_frame_left_axis_xy[0], [0.0, -1.0], atol=1e-6)
    assert result.left_major_signed_deg[0] == pytest.approx(70.0, abs=1e-4)
    assert result.right_major_signed_deg[0] == pytest.approx(-70.0, abs=1e-4)
    assert result.left_gaze_signed_deg[0] == pytest.approx(160.0, abs=1e-4)
    assert result.right_gaze_signed_deg[0] == pytest.approx(-160.0, abs=1e-4)
    assert result.vergence_gaze_signed_deg[0] == pytest.approx(40.0, abs=1e-4)
    assert result.vergence_gaze_deg[0] == pytest.approx(40.0, abs=1e-4)
    assert result.left_nasal_gaze_deg[0] == pytest.approx(-70.0, abs=1e-4)
    assert result.right_nasal_gaze_deg[0] == pytest.approx(-70.0, abs=1e-4)
    assert result.mean_eye_vergence_gaze_deg[0] == pytest.approx(-70.0, abs=1e-4)
    assert result.version_gaze_deg[0] == pytest.approx(0.0, abs=1e-4)


def test_eye_angle_chunk_computes_gaze_vergence_as_undirected_axis_separation() -> None:
    ellipse_params = np.asarray(
        [
            [
                [2.0, -1.0, 4.0, 1.5, 20.0],   # left gaze signed +70 deg
                [2.0, 1.0, 4.0, 1.5, 160.0],   # right gaze signed -70 deg
            ]
        ],
        dtype=np.float32,
    )
    ellipse_success = np.asarray([[True, True]], dtype=bool)
    keypoints_roi = np.asarray(
        [
            [
                [0.0, 0.0],   # swim_bladder
                [2.0, -1.0],  # eye_left
                [2.0, 1.0],   # eye_right
            ]
        ],
        dtype=np.float32,
    )

    result = _process_chunk(
        ellipse_params=ellipse_params,
        ellipse_success=ellipse_success,
        keypoints_roi=keypoints_roi,
        heading_deg=np.asarray([0.0], dtype=np.float32),
        detection_success=np.asarray([True], dtype=bool),
        keypoint_indices={
            "swim_bladder": 0,
            "eye_left": 1,
            "eye_right": 2,
        },
    )

    assert result.left_gaze_signed_deg[0] == pytest.approx(70.0, abs=1e-4)
    assert result.right_gaze_signed_deg[0] == pytest.approx(-70.0, abs=1e-4)
    assert result.left_major_signed_deg[0] == pytest.approx(-20.0, abs=1e-4)
    assert result.right_major_signed_deg[0] == pytest.approx(20.0, abs=1e-4)
    assert result.left_eye_angle_deg[0] == pytest.approx(20.0, abs=1e-4)
    assert result.right_eye_angle_deg[0] == pytest.approx(20.0, abs=1e-4)
    assert result.vergence_eye_angle_deg[0] == pytest.approx(40.0, abs=1e-4)
    assert result.vergence_gaze_deg[0] == pytest.approx(40.0, abs=1e-4)
    assert result.vergence_gaze_signed_deg[0] == pytest.approx(40.0, abs=1e-4)
    assert result.left_nasal_gaze_deg[0] == pytest.approx(20.0, abs=1e-4)
    assert result.right_nasal_gaze_deg[0] == pytest.approx(20.0, abs=1e-4)
    assert result.mean_eye_vergence_gaze_deg[0] == pytest.approx(20.0, abs=1e-4)
    assert result.version_gaze_deg[0] == pytest.approx(0.0, abs=1e-4)


def test_eye_angle_base_writer_persists_body_frame_support_group() -> None:
    import zarr

    run = zarr.group()
    eye_angle_analysis._prepare_base_output_arrays(run, total_detections=1, chunk_len=1)
    result = _process_chunk(
        ellipse_params=np.asarray([[[2.0, -1.0, 4.0, 1.5, 110.0], [2.0, 1.0, 4.0, 1.5, 70.0]]], dtype=np.float32),
        ellipse_success=np.asarray([[True, True]], dtype=bool),
        keypoints_roi=np.asarray([[[0.0, 0.0], [2.0, -1.0], [2.0, 1.0]]], dtype=np.float32),
        heading_deg=np.asarray([0.0], dtype=np.float32),
        detection_success=np.asarray([True], dtype=bool),
        keypoint_indices={"swim_bladder": 0, "eye_left": 1, "eye_right": 2},
    )

    eye_angle_analysis._write_base_eye_angle_result(
        run,
        slice(0, 1),
        result,
        frame_indices=np.asarray([12], dtype=np.int64),
        time_seconds=np.asarray([0.2], dtype=np.float32),
    )

    body_frame = run["support"]["body_frame"]
    assert body_frame.attrs["body_frame_schema_id"] == "fish_anatomical_body_frame"
    np.testing.assert_allclose(body_frame["origin_xy"][:], [[2.0, 0.0]])
    np.testing.assert_allclose(body_frame["forward_axis_xy"][:], [[1.0, 0.0]])
    np.testing.assert_allclose(body_frame["left_axis_xy"][:], [[0.0, -1.0]])
    assert body_frame["valid"][:].tolist() == [True]
    assert decode_reason_bytes(body_frame["failure_reason_bytes"][:]).tolist() == ["ok"]
    assert "left_gaze_xy" in run["angles"]["roi"]
    assert "right_gaze_xy" in run["angles"]["roi"]
    assert "left_eye_angle_deg" in run["angles"]["roi"]
    assert "right_eye_angle_deg" in run["angles"]["roi"]
    assert "vergence_eye_angle_deg" in run["angles"]["roi"]
    assert "major_axis_marginal" in run["qa"]["roi"]


def test_eye_angle_compact_dense_writer_packs_logical_tables(tmp_path) -> None:
    import zarr

    from fisheye.analysis.eye_angle_io import load_eye_angle_run_tables

    root = zarr.open_group(str(tmp_path / "eye_angle_compact_writer.zarr"), mode="w")
    parent = root.create_group("analysis").create_group("eye_angle_runs")
    parent.attrs["latest"] = "compact"
    run = parent.create_group("compact")
    run.attrs["schema_version"] = 5

    roi = run.create_group("angles").create_group("roi")
    roi.create_array("left_eye_angle_deg", data=np.asarray([10.0, 11.0], dtype=np.float32), chunks=(2,))
    roi.create_array("left_gaze_deg", data=np.asarray([80.0, 81.0], dtype=np.float32), chunks=(2,))
    roi.create_array("heading_deg", data=np.asarray([1.0, 2.0], dtype=np.float32), chunks=(2,))
    roi.create_array(
        "left_gaze_xy",
        data=np.asarray([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32),
        chunks=(2, 2),
    )
    frame = run["angles"].create_group("frame")
    frame.create_array("left_gaze_deg", data=np.asarray([80.0, 81.0, 82.0], dtype=np.float32), chunks=(3,))
    frame.create_array("left_eye_angle_deg", data=np.asarray([10.0, 11.0, 12.0], dtype=np.float32), chunks=(3,))

    qa_roi = run.create_group("qa").create_group("roi")
    qa_roi.create_array("valid_frame", data=np.asarray([True, False], dtype=bool), chunks=(2,))
    qa_roi.create_array("valid_left", data=np.asarray([True, True], dtype=bool), chunks=(2,))
    qa_roi.create_array("reason_codes", data=np.asarray([0, 4], dtype=np.uint16), chunks=(2,))
    qa_frame = run["qa"].create_group("frame")
    qa_frame.create_array("valid_frame", data=np.asarray([True, False, True], dtype=bool), chunks=(3,))
    qa_frame.create_array("reason_codes", data=np.asarray([0, 4, 0], dtype=np.uint16), chunks=(3,))
    run.create_group("support")

    eye_angle_analysis._write_compact_dense_layout(
        run,
        total_detections=2,
        num_frames=3,
        chunk_len=2,
        frame_chunk=3,
    )

    assert run.attrs["layout"] == eye_angle_analysis.EYE_ANGLE_LAYOUT_COMPACT_DENSE_V2
    assert "angles" not in run
    assert "qa" not in run
    assert run["roi_angles"].shape == (2, 3)
    assert run["frame_angles"].shape == (3, 3)
    assert run["roi_vectors"].shape == (2, 1, 2)
    assert run["roi_qa"].shape == (2, 3)

    tables = load_eye_angle_run_tables(root, run_name="latest")
    np.testing.assert_allclose(tables.roi["left_eye_angle_deg"], [10.0, 11.0])
    np.testing.assert_allclose(tables.roi["heading_deg"], [1.0, 2.0])
    assert "heading_deg" not in tables.frame
    np.testing.assert_allclose(tables.frame["left_gaze_deg"], [80.0, 81.0, 82.0])
    np.testing.assert_allclose(tables.roi["left_gaze_xy"], [[1.0, 0.0], [0.9, 0.1]])
    assert "left_gaze_xy" not in tables.frame
    assert tables.qa_roi["valid_left"].astype(bool).tolist() == [True, True]
    assert "valid_left" not in tables.qa_frame
    assert tables.qa_frame["valid_frame"].astype(bool).tolist() == [True, False, True]
    assert tables.source_paths[
        "analysis/eye_angle_runs/compact/angles/frame/left_gaze_deg"
    ].startswith("analysis/eye_angle_runs/compact/frame_angles[:,")
