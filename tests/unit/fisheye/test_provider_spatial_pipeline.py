from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.analysis.provider_occupancy_v2 import (
    OccupancyGrid,
    OccupancyTimingPolicy,
    calculate_provider_occupancy_v2,
)
from fisheye.analysis.provider_occupancy_contrast import compute_occupancy_contrast
from fisheye.analysis.provider_spatial_trajectory import (
    ProviderTrackSamples,
    SourceCameraToArenaMMTransform,
    TrajectoryAuthorityIdentities,
    prepare_provider_spatial_trajectory,
)
from fisheye.analysis_workflows.composable_stimulus_selection import (
    RoleMetadata,
    SelectionSpec,
    TimelineAuthority,
    compile_selection,
    member,
    stimulus_step_reference,
    union,
)
from fisheye.analysis_workflows.materializers.composable_stimulus_selection import (
    materialize_composable_stimulus_selection,
)
from fisheye.analysis_workflows.materializers.provider_occupancy_v2 import (
    PROVIDER_OCCUPANCY_MANIFEST_DIGEST_ATTR,
    PROVIDER_OCCUPANCY_MANIFEST_ATTR,
    materialize_provider_occupancy_v2,
)
from fisheye.analysis_workflows.materializers.provider_occupancy_contrast import (
    MANIFEST_ATTR as CONTRAST_MANIFEST_ATTR,
    SOURCE_SCOPE_POOLED,
    build_pooled_occupancy_contrast_summary,
    build_provider_occupancy_contrast_materialization_plan,
    publish_provider_occupancy_contrast_run,
)
from fisheye.analysis_workflows.materializers.provider_spatial_trajectory import (
    materialize_provider_spatial_trajectory_run,
)
from fisheye.analysis_workflows.provider_spatial_pipeline import (
    ProviderSpatialPipelineError,
    build_provider_occupancy_v2_source_bindings,
    compiled_selection_membership,
    occupancy_samples_from_provider_trajectory,
)
from fisheye.analysis_workflows.provider_spatial_grid_policy import (
    CircularArenaGeometryAuthority,
    PhysicalScaleAuthority,
    SelectionAuthority,
    build_arena_mm_grid_policy,
)
from fisheye.shared.zarr_io import open_zarr_root


RECORDING_ID = "recording-pipeline-canary"
TIMELINE_ID = "timeline-pipeline-canary"
PROVIDER_ID = "detection-provider-v1"
ESTIMATOR_ID = "estimator-v1"
SOURCE_ID = "detection-source-v1"
TRACK_SAMPLE_POLICY_ID = "one_tracked_subject_sample_per_acquisition_frame_v1"
TIMING_AUTHORITY_ID = "camera-timing-v1"
COORDINATE_AUTHORITY_ID = "camera-coordinate-v1"
TARGET_COORDINATE_AUTHORITY_ID = "arena-coordinate-v1"


def _timeline() -> TimelineAuthority:
    return TimelineAuthority(
        recording_id=RECORDING_ID,
        timeline_id=TIMELINE_ID,
        stimulus_authority_id="stimulus-authority-v1",
        stimulus_authority_sha256="a" * 64,
        acquisition_frame_domain="camera_acquisition_frame",
        acquisition_frame_count=8,
        source_video_metadata_ref="video-metadata-v1",
        source_video_metadata_sha256="b" * 64,
        acquisition_clock_authority_ref=TIMING_AUTHORITY_ID,
        acquisition_clock_authority_sha256="c" * 64,
        source_metadata_sha256="d" * 64,
    )


def _compiled(selection_id: str, intervals: tuple[tuple[int, int, str], ...]):
    authority = _timeline()
    expressions = tuple(
        member(
            stimulus_step_reference(
                reference_id=f"{reference_id}-frame-{frame}",
                label=reference_id,
                start_frame=frame,
                end_frame=frame + 1,
                authority=authority,
                occurrence_id=f"occ-{reference_id}",
            ),
            role=RoleMetadata(role=reference_id),
        )
        for start, end, reference_id in intervals
        for frame in range(start, end)
    )
    return compile_selection(
        SelectionSpec(
            selection_id=selection_id,
            expression=union(*expressions),
            aggregation_policy="keep_occurrences",
        )
    )


def _rows() -> ProviderTrackSamples:
    frames = np.asarray([0, 1, 2, 3, 5, 6, 7], dtype=np.int64)
    return ProviderTrackSamples(
        track_sample_key=np.column_stack(
            (np.zeros(frames.size, dtype=np.int64), frames)
        ),
        acquisition_frame=frames,
        subject_identity=("fish-1",) * frames.size,
        track_identity=("track-1",) * frames.size,
        source_position_xy=np.asarray(
            [
                [0.25, 0.25],
                [0.75, 0.25],
                [1.25, 0.25],
                [1.75, 0.25],
                [2.25, 0.25],
                [2.75, 0.25],
                [0.25, 0.75],
            ],
            dtype=np.float64,
        ),
        provider_present=np.ones(frames.size, dtype=bool),
        provider_valid=np.ones(frames.size, dtype=bool),
        provider_reason_code=("ok",) * frames.size,
        recording_ids=(RECORDING_ID,) * frames.size,
        timeline_authority_ids=(TIMELINE_ID,) * frames.size,
    )


def _trajectory(compiled):
    selection = compiled_selection_membership(compiled)
    transform = SourceCameraToArenaMMTransform(
        source_coordinate_authority_id=COORDINATE_AUTHORITY_ID,
        target_coordinate_authority_id=TARGET_COORDINATE_AUTHORITY_ID,
        matrix=np.eye(3, dtype=np.float64),
        grid_extent_mm=(0.0, 3.0, 0.0, 3.0),
        source_camera_extent_px=(0.0, 3.0, 0.0, 3.0),
    )
    authorities = TrajectoryAuthorityIdentities(
        recording_id=RECORDING_ID,
        provider_id=PROVIDER_ID,
        track_sample_policy_id=TRACK_SAMPLE_POLICY_ID,
        estimator_id=ESTIMATOR_ID,
        source_id=SOURCE_ID,
        timing_authority_id=TIMING_AUTHORITY_ID,
        timeline_authority_id=TIMELINE_ID,
        coordinate_authority_id=COORDINATE_AUTHORITY_ID,
        selection_authority_id=compiled.resolved_digest,
    )
    return prepare_provider_spatial_trajectory(
        authorities=authorities,
        rows=_rows(),
        selection=selection,
        transform=transform,
    )


def _authorities(result, trajectory):
    return {
        "provider": {
            "schema_id": "test.provider_authority",
            "schema_version": 1,
            "recording_id": RECORDING_ID,
            "provider_id": PROVIDER_ID,
            "estimator_id": ESTIMATOR_ID,
            "source_id": SOURCE_ID,
            "subject_id": "fish-1",
            "estimator": {
                "estimator_id": ESTIMATOR_ID,
                "model_id": "model-v1",
                "model_sha256": "e" * 64,
            },
        },
        "timing": {
            "schema_id": "test.timing_authority",
            "schema_version": 1,
            "recording_id": RECORDING_ID,
            "timeline_authority_id": TIMELINE_ID,
            "timing_authority_id": TIMING_AUTHORITY_ID,
            "fps_hz": result.fps_hz,
            "timing_policy_id": result.timing_policy_id,
        },
        "geometry": {
            "schema_id": "test.geometry_authority",
            "schema_version": 1,
            "recording_id": RECORDING_ID,
            "coordinate_authority_id": COORDINATE_AUTHORITY_ID,
            "geometry": {"geometry_id": "dish-geometry-v1"},
        },
        "transform": {
            "schema_id": "test.transform_authority",
            "schema_version": 1,
            "recording_id": RECORDING_ID,
            "source_coordinate_authority_id": COORDINATE_AUTHORITY_ID,
            "target_coordinate_authority_id": TARGET_COORDINATE_AUTHORITY_ID,
            "transform_sha256": trajectory.transform.sha256,
        },
        "fixed_grid_policy": {
            "schema_id": "test.fixed_grid_policy",
            "schema_version": 1,
            "grid_id": "arena-grid-v1",
            "config_digest": result.config_digest,
            "edge_policy_id": result.edge_policy_id,
            "timing_policy_id": result.timing_policy_id,
            "fps_hz": result.fps_hz,
            "x_edges": result.x_edges.tolist(),
            "y_edges": result.y_edges.tolist(),
        },
    }


def _publish_selection(archive: Path, compiled, scratch: Path, run_name: str) -> str:
    materialize_composable_stimulus_selection(
        archive,
        compiled_selection=compiled,
        scratch_root=scratch,
        run_name=run_name,
        apply=True,
    )
    return f"analysis/stimulus_selection_runs/{run_name}"


def _publish_trajectory(archive: Path, trajectory, scratch: Path, run_name: str) -> str:
    materialize_provider_spatial_trajectory_run(
        archive,
        trajectory,
        run_name=run_name,
        scratch_root=scratch,
        apply=True,
    )
    return f"analysis/provider_spatial_trajectory_runs/{run_name}"


def _canary(tmp_path: Path):
    archive = tmp_path / "analysis.zarr"
    zarr.open_group(str(archive), mode="w-", zarr_format=3, use_consolidated=False)
    compiled_a = _compiled("selection-a", ((1, 5, "pre"), (3, 7, "chaser")))
    compiled_b = _compiled("selection-b", ((0, 2, "baseline"), (6, 8, "post")))
    selection_a_path = _publish_selection(
        archive, compiled_a, tmp_path / "selection-a-scratch", "selection-a"
    )
    selection_b_path = _publish_selection(
        archive, compiled_b, tmp_path / "selection-b-scratch", "selection-b"
    )
    trajectory_a = _trajectory(compiled_a)
    trajectory_b = _trajectory(compiled_b)
    trajectory_a_path = _publish_trajectory(
        archive, trajectory_a, tmp_path / "trajectory-a-scratch", "trajectory-a"
    )
    trajectory_b_path = _publish_trajectory(
        archive, trajectory_b, tmp_path / "trajectory-b-scratch", "trajectory-b"
    )
    grid = OccupancyGrid([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0])
    timing = OccupancyTimingPolicy(10.0)
    samples_a = occupancy_samples_from_provider_trajectory(
        trajectory_a,
        selection=compiled_selection_membership(compiled_a),
    )
    result_a = calculate_provider_occupancy_v2(samples_a, grid, timing)
    authorities_a = _authorities(result_a, trajectory_a)
    bindings_a = build_provider_occupancy_v2_source_bindings(
        archive,
        selection_run_path=selection_a_path,
        trajectory_run_path=trajectory_a_path,
        compiled_selection=compiled_a,
        trajectory=trajectory_a,
        result=result_a,
        provider_authority=authorities_a["provider"],
        timing_authority=authorities_a["timing"],
        geometry_authority=authorities_a["geometry"],
        transform_authority=authorities_a["transform"],
        fixed_grid_policy_authority=authorities_a["fixed_grid_policy"],
    )
    samples_b = occupancy_samples_from_provider_trajectory(
        trajectory_b,
        selection=compiled_selection_membership(compiled_b),
    )
    result_b = calculate_provider_occupancy_v2(samples_b, grid, timing)
    authorities_b = _authorities(result_b, trajectory_b)
    bindings_b = build_provider_occupancy_v2_source_bindings(
        archive,
        selection_run_path=selection_b_path,
        trajectory_run_path=trajectory_b_path,
        compiled_selection=compiled_b,
        trajectory=trajectory_b,
        result=result_b,
        provider_authority=authorities_b["provider"],
        timing_authority=authorities_b["timing"],
        geometry_authority=authorities_b["geometry"],
        transform_authority=authorities_b["transform"],
        fixed_grid_policy_authority=authorities_b["fixed_grid_policy"],
    )
    return locals()


def test_selection_trajectory_occupancy_pipeline_publishes_exact_overlap_lineage(
    tmp_path: Path,
) -> None:
    values = _canary(tmp_path)
    output = materialize_provider_occupancy_v2(
        values["archive"],
        values["result_a"],
        values["bindings_a"],
        scratch_root=tmp_path / "occupancy-scratch",
        run_name="occupancy-a",
        apply=True,
        keep_scratch=False,
    )
    assert output["status"] == "complete"
    root = open_zarr_root(values["archive"], mode="r", use_consolidated=True)
    run = root["analysis/provider_occupancy_runs/occupancy-a"]
    manifest = run.attrs[PROVIDER_OCCUPANCY_MANIFEST_ATTR]
    assert manifest["payload"]["source_bindings"] == values["bindings_a"].as_record()
    assert run["pooled/expected_selected_frames"][:].tolist() == [6]
    assert run["pooled/provider_present_count"][:].tolist() == [5]
    assert int(run["pooled/counts"][:].sum()) == 5
    assert run["per_occurrence/expected_selected_frames"][:].tolist() == [4, 4]
    assert run["per_occurrence/counts"][:].sum(axis=(1, 2)).tolist() == [3, 3]
    assert values["bindings_a"].values["compiled_selection"]["record"]["run_path"] != values[
        "bindings_b"
    ].values["compiled_selection"]["record"]["run_path"]
    assert values["bindings_a"].values["trajectory"]["record"]["run_path"] != values[
        "bindings_b"
    ].values["trajectory"]["record"]["run_path"]
    assert values["bindings_a"].values["provider"]["record"]["estimator"] == values[
        "bindings_b"
    ].values["provider"]["record"]["estimator"]
    assert values["bindings_a"].values["trajectory"]["record"]["position_track_policy"] == values[
        "bindings_b"
    ].values["trajectory"]["record"]["position_track_policy"]
    assert values["bindings_a"].values["trajectory"]["record"]["position_track_policy"][
        "policy_id"
    ] == TRACK_SAMPLE_POLICY_ID
    assert values["bindings_a"].values["trajectory"]["record"][
        "position_track_policy"
    ]["policy_id"] != values["bindings_a"].values["trajectory"]["record"][
        "policy_id"
    ]
    assert "trajectory_run_manifest" not in values["bindings_a"].values[
        "trajectory"
    ]["record"]
    assert "trajectory_array_manifest" not in values["bindings_a"].values[
        "trajectory"
    ]["record"]
    assert values["bindings_a"].values["trajectory"]["record"]["sample_unit"] == values[
        "bindings_b"
    ].values["trajectory"]["record"]["sample_unit"]
    assert values["bindings_a"].values["transform"]["record"]["coordinate_frame"] == values[
        "bindings_b"
    ].values["transform"]["record"]["coordinate_frame"]
    assert values["bindings_a"].values["transform"]["record"]["coordinate_frame"][
        "coordinate_frame_id"
    ] == TARGET_COORDINATE_AUTHORITY_ID

    output_b = materialize_provider_occupancy_v2(
        values["archive"],
        values["result_b"],
        values["bindings_b"],
        scratch_root=tmp_path / "occupancy-b-scratch",
        run_name="occupancy-b",
        apply=True,
        keep_scratch=False,
    )
    assert output_b["status"] == "complete"
    root = open_zarr_root(values["archive"], mode="r", use_consolidated=True)
    occupancy_a_path = "analysis/provider_occupancy_runs/occupancy-a"
    occupancy_b_path = "analysis/provider_occupancy_runs/occupancy-b"
    occupancy_a_digest = root[occupancy_a_path].attrs[
        PROVIDER_OCCUPANCY_MANIFEST_DIGEST_ATTR
    ]
    occupancy_b_digest = root[occupancy_b_path].attrs[
        PROVIDER_OCCUPANCY_MANIFEST_DIGEST_ATTR
    ]
    baseline = build_pooled_occupancy_contrast_summary(
        values["archive"],
        run_path=occupancy_a_path,
        manifest_sha256=occupancy_a_digest,
        arm_role="baseline",
        source_scope=SOURCE_SCOPE_POOLED,
    )
    treatment = build_pooled_occupancy_contrast_summary(
        values["archive"],
        run_path=occupancy_b_path,
        manifest_sha256=occupancy_b_digest,
        arm_role="treatment",
        source_scope=SOURCE_SCOPE_POOLED,
    )
    assert baseline["source_manifest"]["run_id"] != treatment["source_manifest"]["run_id"]
    assert baseline["position_track_policy"] == treatment["position_track_policy"]
    assert baseline["coordinate_frame"] == treatment["coordinate_frame"]
    contrast = compute_occupancy_contrast(
        baseline,
        treatment,
        config={"selection_policy": "same-provider-v2-pooled-v1"},
    )
    contrast_plan = build_provider_occupancy_contrast_materialization_plan(
        values["archive"],
        baseline_run_path=occupancy_a_path,
        treatment_run_path=occupancy_b_path,
        baseline_manifest_digest=occupancy_a_digest,
        treatment_manifest_digest=occupancy_b_digest,
        contrast_result=contrast,
        run_name="occupancy-contrast-a-b",
        scratch_root=tmp_path / "contrast-scratch",
        source_scope=SOURCE_SCOPE_POOLED,
    )
    contrast_output = publish_provider_occupancy_contrast_run(
        contrast_plan,
        keep_scratch=False,
    )
    assert contrast_output["status"] == "complete"
    root = open_zarr_root(values["archive"], mode="r", use_consolidated=True)
    contrast_run = root["analysis/provider_occupancy_contrast_runs/occupancy-contrast-a-b"]
    assert contrast_run.attrs["stage_selector_eligible"] is False
    contrast_manifest = contrast_run.attrs[CONTRAST_MANIFEST_ATTR]
    contrast_payload = contrast_manifest["payload"]
    assert contrast_payload["source_runs"]["baseline"] == {
        "role": "baseline",
        "run_path": occupancy_a_path,
        "manifest_sha256": occupancy_a_digest,
        "manifest_envelope_sha256": baseline["source_manifest"]["sha256"],
        "source_bindings_sha256": baseline["bindings_sha256"],
        "run_schema_version": 2,
        "manifest_schema_version": 2,
    }
    assert contrast_payload["source_runs"]["treatment"] == {
        "role": "treatment",
        "run_path": occupancy_b_path,
        "manifest_sha256": occupancy_b_digest,
        "manifest_envelope_sha256": treatment["source_manifest"]["sha256"],
        "source_bindings_sha256": treatment["bindings_sha256"],
        "run_schema_version": 2,
        "manifest_schema_version": 2,
    }
    for path in (
        values["selection_a_path"],
        values["selection_b_path"],
        values["trajectory_a_path"],
        values["trajectory_b_path"],
        occupancy_a_path,
        occupancy_b_path,
        "analysis/provider_occupancy_contrast_runs/occupancy-contrast-a-b",
    ):
        assert root[path].attrs["stage_selector_eligible"] is False


def test_adapter_preserves_compact_array_backed_grid_authority(tmp_path: Path) -> None:
    values = _canary(tmp_path)
    selection_authority = SelectionAuthority(
        selection_id="grid-selection-v1",
        recording_id=RECORDING_ID,
        record_sha256="a" * 64,
        record_ref="analysis/arena_geometry_selection/grid-selection-v1",
    )
    grid_policy = build_arena_mm_grid_policy(
        recording_id=RECORDING_ID,
        geometry=CircularArenaGeometryAuthority(
            geometry_id="dish-geometry-v1",
            coordinate_authority_id=COORDINATE_AUTHORITY_ID,
            center_x_px=1.5,
            center_y_px=1.5,
            radius_px=1.5,
            record_ref="analysis/arena_geometry_candidates/dish-geometry-v1",
        ),
        scale=PhysicalScaleAuthority(
            scale_id="camera-scale-v1",
            coordinate_authority_id=COORDINATE_AUTHORITY_ID,
            mm_per_pixel=1.0,
            record_ref="analysis/source_camera_scale/camera-scale-v1",
        ),
        selection=selection_authority,
    )
    result = calculate_provider_occupancy_v2(
        occupancy_samples_from_provider_trajectory(
            values["trajectory_a"],
            selection=compiled_selection_membership(values["compiled_a"]),
        ),
        grid_policy.to_occupancy_grid(),
        values["timing"],
    )
    authorities = _authorities(result, values["trajectory_a"])
    authorities["fixed_grid_policy"] = {
        "schema_id": "palette.provider_spatial_fixed_grid_policy_authority",
        "schema_version": 2,
        "recording_id": RECORDING_ID,
        "grid_id": grid_policy.policy_id,
        "config_digest": result.config_digest,
        "edge_policy_id": result.edge_policy_id,
        "timing_policy_id": result.timing_policy_id,
        "fps_hz": result.fps_hz,
        "edge_count_xy": {
            "x": int(result.x_edges.size),
            "y": int(result.y_edges.size),
        },
        "bounds_mm": {
            "x": [float(result.x_edges[0]), float(result.x_edges[-1])],
            "y": [float(result.y_edges[0]), float(result.y_edges[-1])],
        },
        "edge_array_paths": {
            "x": "grid/x_edges",
            "y": "grid/y_edges",
            "path_scope": "relative_to_provider_occupancy_run",
        },
        "grid_policy": grid_policy.source_binding_authority_record(),
    }

    bindings = build_provider_occupancy_v2_source_bindings(
        values["archive"],
        selection_run_path=values["selection_a_path"],
        trajectory_run_path=values["trajectory_a_path"],
        compiled_selection=values["compiled_a"],
        trajectory=values["trajectory_a"],
        result=result,
        provider_authority=authorities["provider"],
        timing_authority=authorities["timing"],
        geometry_authority=authorities["geometry"],
        transform_authority=authorities["transform"],
        fixed_grid_policy_authority=authorities["fixed_grid_policy"],
    )

    fixed = bindings.values["fixed_grid_policy"]["record"]
    assert "x_edges" not in fixed
    assert "y_edges" not in fixed
    assert fixed["edge_array_paths"]["x"] == "grid/x_edges"
    assert fixed["grid_policy"]["x_edges_sha256"]
    assert fixed["grid_policy"]["y_edges_sha256"]

    wrong_grid_policy = build_arena_mm_grid_policy(
        recording_id=RECORDING_ID,
        geometry=CircularArenaGeometryAuthority(
            geometry_id="dish-geometry-wrong-v1",
            coordinate_authority_id=COORDINATE_AUTHORITY_ID,
            center_x_px=1.5,
            center_y_px=1.5,
            radius_px=2.5,
            record_ref="analysis/arena_geometry_candidates/dish-geometry-wrong-v1",
        ),
        scale=grid_policy.scale,
        selection=selection_authority,
    )
    mismatched_authority = dict(authorities["fixed_grid_policy"])
    mismatched_authority["grid_policy"] = (
        wrong_grid_policy.source_binding_authority_record()
    )
    with pytest.raises(ProviderSpatialPipelineError, match="x_edges_sha256"):
        build_provider_occupancy_v2_source_bindings(
            values["archive"],
            selection_run_path=values["selection_a_path"],
            trajectory_run_path=values["trajectory_a_path"],
            compiled_selection=values["compiled_a"],
            trajectory=values["trajectory_a"],
            result=result,
            provider_authority=authorities["provider"],
            timing_authority=authorities["timing"],
            geometry_authority=authorities["geometry"],
            transform_authority=authorities["transform"],
            fixed_grid_policy_authority=mismatched_authority,
        )


def test_adapter_rejects_direct_array_payload_tamper(tmp_path: Path) -> None:
    values = _canary(tmp_path)
    root = open_zarr_root(values["archive"], mode="a", use_consolidated=False)
    array = root[values["selection_a_path"]]["resolved_interval_bounds"]
    array[0, 0] = int(array[0, 0]) + 1
    with pytest.raises(ProviderSpatialPipelineError, match="content digest"):
        build_provider_occupancy_v2_source_bindings(
            values["archive"],
            selection_run_path=values["selection_a_path"],
            trajectory_run_path=values["trajectory_a_path"],
            compiled_selection=values["compiled_a"],
            trajectory=values["trajectory_a"],
            result=values["result_a"],
            provider_authority=values["authorities_a"]["provider"],
            timing_authority=values["authorities_a"]["timing"],
            geometry_authority=values["authorities_a"]["geometry"],
            transform_authority=values["authorities_a"]["transform"],
            fixed_grid_policy_authority=values["authorities_a"]["fixed_grid_policy"],
        )


def test_adapter_rejects_stale_consolidated_metadata(tmp_path: Path) -> None:
    values = _canary(tmp_path)
    root = open_zarr_root(values["archive"], mode="a", use_consolidated=False)
    root[values["trajectory_a_path"]].attrs["metadata_tamper"] = "direct-only"
    with pytest.raises(ProviderSpatialPipelineError, match="direct/consolidated metadata"):
        build_provider_occupancy_v2_source_bindings(
            values["archive"],
            selection_run_path=values["selection_a_path"],
            trajectory_run_path=values["trajectory_a_path"],
            compiled_selection=values["compiled_a"],
            trajectory=values["trajectory_a"],
            result=values["result_a"],
            provider_authority=values["authorities_a"]["provider"],
            timing_authority=values["authorities_a"]["timing"],
            geometry_authority=values["authorities_a"]["geometry"],
            transform_authority=values["authorities_a"]["transform"],
            fixed_grid_policy_authority=values["authorities_a"]["fixed_grid_policy"],
        )


def test_adapter_rejects_result_from_a_different_trajectory_selection(
    tmp_path: Path,
) -> None:
    values = _canary(tmp_path)
    with pytest.raises(ProviderSpatialPipelineError, match="occupancy result"):
        build_provider_occupancy_v2_source_bindings(
            values["archive"],
            selection_run_path=values["selection_a_path"],
            trajectory_run_path=values["trajectory_a_path"],
            compiled_selection=values["compiled_a"],
            trajectory=values["trajectory_a"],
            result=values["result_b"],
            provider_authority=values["authorities_a"]["provider"],
            timing_authority=values["authorities_a"]["timing"],
            geometry_authority=values["authorities_a"]["geometry"],
            transform_authority=values["authorities_a"]["transform"],
            fixed_grid_policy_authority=values["authorities_a"]["fixed_grid_policy"],
        )
