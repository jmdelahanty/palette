from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.analysis.provider_spatial_trajectory import (
    ProviderSpatialTrajectory,
    ProviderTrackSamples,
    SelectedFrameMembership,
    SourceCameraToArenaMMTransform,
    TrajectoryAuthorityIdentities,
    prepare_provider_spatial_trajectory,
)
from fisheye.analysis_workflows.materializers.provider_spatial_trajectory import (
    ARRAY_MANIFEST_ATTR,
    PARENT_PATH,
    ProviderSpatialTrajectoryMaterializationError,
    plan_provider_spatial_trajectory_run,
    publish_provider_spatial_trajectory_run,
)
from fisheye.shared.zarr.metadata_equivalence import (
    validate_direct_consolidated_subtree,
)
from fisheye.shared.zarr_io import open_zarr_root


pytestmark = pytest.mark.filterwarnings(
    "error::zarr.errors.UnstableSpecificationWarning"
)


def _iter_arrays(group: zarr.Group):
    for _name, array in group.arrays():
        yield array
    for _name, child in group.groups():
        yield from _iter_arrays(child)


def _decode_text(run: zarr.Group, prefix: str) -> list[str]:
    offsets = np.asarray(run[f"{prefix}_offsets"][:], dtype=np.int64)
    payload = np.asarray(run[f"{prefix}_utf8"][:], dtype=np.uint8).tobytes()
    return [
        payload[int(offsets[index]) : int(offsets[index + 1])].decode("utf-8")
        for index in range(offsets.size - 1)
    ]


def _trajectory() -> ProviderSpatialTrajectory:
    authorities = TrajectoryAuthorityIdentities(
        recording_id="recording-1",
        provider_id="detection",
        track_sample_policy_id="one_track_sample_per_subject_frame_v1",
        estimator_id="estimator-v1",
        source_id="source-run-v1",
        timing_authority_id="camera-timestamps-v1",
        timeline_authority_id="timeline-v1",
        coordinate_authority_id="camera-frame-v1",
        selection_authority_id="stimulus-selection-v1",
    )
    selection = SelectedFrameMembership(
        recording_id="recording-1",
        timeline_authority_id="timeline-v1",
        selection_authority_id="stimulus-selection-v1",
        acquisition_frames=np.asarray([0, 1, 2], dtype=np.int64),
        membership_keys=(
            ("atomic-pre", "atomic-chaser"),
            ("atomic-gap",),
            ("atomic-post",),
        ),
        occurrence_ids=(
            ("occ-pre", "occ-chaser"),
            ("occ-gap",),
            ("occ-post",),
        ),
        roles=(
            ("pre", "chaser"),
            ("gap",),
            ("post",),
        ),
    )
    rows = ProviderTrackSamples(
        track_sample_key=np.asarray([[0, 0], [0, 2]], dtype=np.int64),
        acquisition_frame=np.asarray([0, 2], dtype=np.int64),
        subject_identity=("fish-1", "fish-1"),
        track_identity=("track-1", "track-1"),
        source_position_xy=np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64),
        provider_present=np.asarray([True, True], dtype=bool),
        provider_valid=np.asarray([True, True], dtype=bool),
        provider_reason_code=("ok", "ok"),
    )
    transform = SourceCameraToArenaMMTransform(
        source_coordinate_authority_id="camera-frame-v1",
        target_coordinate_authority_id="arena-mm-v1",
        matrix=np.eye(3, dtype=np.float64),
        grid_extent_mm=(0.0, 100.0, 0.0, 100.0),
        source_camera_extent_px=(0.0, 100.0, 0.0, 100.0),
    )
    return prepare_provider_spatial_trajectory(
        authorities=authorities,
        rows=rows,
        selection=selection,
        transform=transform,
    )


def _archive(path: Path) -> Path:
    root = zarr.open_group(str(path), mode="w-", zarr_format=3, use_consolidated=False)
    parent = root.require_group("analysis").require_group(
        "provider_spatial_trajectory_runs"
    )
    parent.attrs.update(
        {
            "latest": "existing-run",
            "latest_complete": "existing-run",
            "latest_pending": None,
            "authoritative_run": "existing-run",
        }
    )
    return path


def _plan(tmp_path: Path):
    return plan_provider_spatial_trajectory_run(
        _archive(tmp_path / "analysis.zarr"),
        _trajectory(),
        run_name="trajectory-canary-v1",
        scratch_root=tmp_path / "scratch",
    )


def test_round_trip_publishes_exact_arrays_and_direct_consolidated_metadata(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    result = publish_provider_spatial_trajectory_run(plan, copy_backend="python")

    assert result["status"] == "complete"
    assert result["selector_eligible"] is False
    archive = open_zarr_root(plan.source_zarr, mode="r", use_consolidated=False)
    run = archive[plan.run_path]
    assert run.attrs["stage_selector_eligible"] is False
    assert run.attrs["palette_run_completion_status"] == "complete"
    assert archive[PARENT_PATH].attrs["palette_completion_epoch"] == 2
    assert run["track_sample_key"].dtype == np.dtype(np.int64)
    assert run["track_sample_key"].shape == (2, 2)
    assert run["source_extent_valid"][:].tolist() == [True, True]
    assert run.attrs["provider_spatial_trajectory_manifest"]["authorities"][
        "track_sample_policy_id"
    ] == "one_track_sample_per_subject_frame_v1"
    assert run["subject_identity_offsets"].dtype == np.dtype(np.int64)
    assert run["subject_identity_utf8"].dtype == np.dtype(np.uint8)
    assert run["track_identity_offsets"].dtype == np.dtype(np.int64)
    assert run["track_identity_utf8"].dtype == np.dtype(np.uint8)
    assert _decode_text(run, "subject_identity") == ["fish-1", "fish-1"]
    assert _decode_text(run, "track_identity") == ["track-1", "track-1"]
    assert all(
        array.dtype.kind in "iufb" and array.dtype.kind not in "OSU"
        for array in _iter_arrays(run)
    )
    assert run["selection/acquisition_frame"][:].tolist() == [0, 1, 2]
    assert run["selection/source_row_membership_offsets"][:].tolist() == [0, 2, 3]
    assert run["reasons/offsets"].shape == (3,)
    assert ARRAY_MANIFEST_ATTR in run.attrs
    assert result["validation"]["published_direct_consolidated"]["array_count"] >= 1
    validate_direct_consolidated_subtree(
        plan.source_zarr,
        subtree_path=plan.run_path,
    )
    consolidated = open_zarr_root(plan.source_zarr, mode="r", use_consolidated=True)
    assert plan.run_path in consolidated
    assert consolidated[plan.run_path]["track_sample_key"][:].tolist() == [[0, 0], [0, 2]]


def test_selected_frame_denominator_retains_missing_provider_frame(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    publish_provider_spatial_trajectory_run(plan)
    run = open_zarr_root(plan.source_zarr, mode="r", use_consolidated=False)[plan.run_path]

    assert run["selection/acquisition_frame"][:].tolist() == [0, 1, 2]
    assert run.attrs["provider_spatial_trajectory_manifest"][
        "selected_frame_denominator"
    ]["count"] == 3
    assert plan.trajectory.counts.missing_selected_frames == 1
    assert run["in_selection"][:].tolist() == [True, True]


def test_overlap_memberships_are_persisted_as_aligned_ragged_arrays(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    publish_provider_spatial_trajectory_run(plan)
    run = open_zarr_root(plan.source_zarr, mode="r", use_consolidated=False)[plan.run_path]

    assert run["selection/membership_offsets"][:].tolist() == [0, 2, 3, 4]
    assert _decode_text(run, "selection/membership_key") == [
        "atomic-pre",
        "atomic-chaser",
        "atomic-gap",
        "atomic-post",
    ]
    assert _decode_text(run, "selection/occurrence_id") == [
        "occ-pre",
        "occ-chaser",
        "occ-gap",
        "occ-post",
    ]
    assert _decode_text(run, "selection/role") == [
        "pre",
        "chaser",
        "gap",
        "post",
    ]


def test_one_interval_membership_identity_may_repeat_across_its_frames(
    tmp_path: Path,
) -> None:
    base = _trajectory()
    selection = SelectedFrameMembership(
        recording_id="recording-1",
        timeline_authority_id="timeline-v1",
        selection_authority_id="stimulus-selection-v1",
        acquisition_frames=np.asarray([0, 1, 2], dtype=np.int64),
        membership_keys=(("atomic-pre",), ("atomic-pre",), ("atomic-pre",)),
        occurrence_ids=(("occ-pre",), ("occ-pre",), ("occ-pre",)),
        roles=(("pre",), ("pre",), ("pre",)),
    )
    rows = ProviderTrackSamples(
        track_sample_key=base.track_sample_key,
        acquisition_frame=base.acquisition_frame,
        subject_identity=base.subject_identity,
        track_identity=base.track_identity,
        source_position_xy=base.source_position_xy,
        provider_present=base.provider_present,
        provider_valid=base.provider_valid,
        provider_reason_code=("ok", "ok"),
    )
    trajectory = prepare_provider_spatial_trajectory(
        authorities=base.authorities,
        rows=rows,
        selection=selection,
        transform=base.transform,
    )
    plan = plan_provider_spatial_trajectory_run(
        _archive(tmp_path / "analysis.zarr"),
        trajectory,
        run_name="repeated-membership-canary-v1",
        scratch_root=tmp_path / "scratch",
    )
    publish_provider_spatial_trajectory_run(plan)
    run = open_zarr_root(plan.source_zarr, mode="r", use_consolidated=False)[
        plan.run_path
    ]
    assert _decode_text(run, "selection/membership_key") == [
        "atomic-pre",
        "atomic-pre",
        "atomic-pre",
    ]


def test_source_trajectory_is_not_mutated(tmp_path: Path) -> None:
    trajectory = _trajectory()
    before = trajectory.source_position_xy.copy()
    plan = plan_provider_spatial_trajectory_run(
        _archive(tmp_path / "analysis.zarr"),
        trajectory,
        run_name="trajectory-canary-v1",
        scratch_root=tmp_path / "scratch",
    )
    publish_provider_spatial_trajectory_run(plan)
    np.testing.assert_array_equal(trajectory.source_position_xy, before)
    assert trajectory.source_position_xy.flags.writeable is False


def test_parent_selectors_are_preserved(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    before = dict(plan.parent_selector_attrs)
    publish_provider_spatial_trajectory_run(plan)
    root = open_zarr_root(plan.source_zarr, mode="r", use_consolidated=False)
    parent = root[PARENT_PATH]
    assert {
        name: parent.attrs[name]
        for name in before
    } == before


def test_retry_and_mutation_after_planning_fail_closed(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    publish_provider_spatial_trajectory_run(plan)
    with pytest.raises(FileExistsError):
        publish_provider_spatial_trajectory_run(plan)

    archive = _archive(tmp_path / "tamper.zarr")
    trajectory = _trajectory()
    tamper_plan = plan_provider_spatial_trajectory_run(
        archive,
        trajectory,
        run_name="tamper-canary-v1",
        scratch_root=tmp_path / "tamper-scratch",
    )
    trajectory.arena_position_xy.setflags(write=True)
    trajectory.arena_position_xy[0, 0] += 1.0
    with pytest.raises(ProviderSpatialTrajectoryMaterializationError, match="changed after planning"):
        publish_provider_spatial_trajectory_run(tamper_plan)


def test_existing_local_candidate_tamper_is_rejected(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    # The first call creates and validates the local immutable candidate but
    # intentionally does not publish it, allowing the physical-tamper gate to
    # be exercised before the atomic rename.
    from fisheye.analysis_workflows.materializers import provider_spatial_trajectory as materializer

    materializer._write_local(plan)
    local = open_zarr_root(plan.local_run_path, mode="a", use_consolidated=False)
    local["arena_position_xy"][0, 0] = 999.0
    with pytest.raises(ProviderSpatialTrajectoryMaterializationError, match="array drifted"):
        materializer._validate_run(plan.local_run_path, plan=plan)


def test_selector_like_child_attribute_is_rejected(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    publish_provider_spatial_trajectory_run(plan)
    run = open_zarr_root(plan.target_run_path, mode="a", use_consolidated=False)
    run.attrs["latest_materialized"] = plan.run_name
    from fisheye.analysis_workflows.materializers import provider_spatial_trajectory as materializer

    with pytest.raises(ProviderSpatialTrajectoryMaterializationError, match="selector attributes"):
        materializer._validate_run(plan.target_run_path, plan=plan)
