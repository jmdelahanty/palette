from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import zarr

from fisheye.analysis_workflows.materializers.subject_position import (
    SUBJECT_POSITION_PARENT_PATH,
    plan_subject_position_run,
    publish_subject_position_run,
)
from fisheye.analysis_workflows.provider_spatial_track_source import (
    TRACK_SAMPLE_POLICY_ID,
    ProviderSpatialTrackSourceError,
    ProviderTrackSourceAuthorities,
    build_provider_track_source,
)
from fisheye.analysis_workflows.subject_position_source_handle import (
    load_subject_position_source_handle,
)
from fisheye.analysis_workflows.tracking_source_handle import (
    load_tracking_source_handle,
)
from fisheye.shared.subject_position_expression import (
    KEYPOINT_ANATOMICAL_TRIAD_MEAN_ESTIMATOR_ID,
    estimator_profile_digest,
    get_estimator_profile,
)
from fisheye.tracking.single_subject_per_arena import (
    TrackingConflictError,
    write_single_subject_per_arena_tracking_run,
)
from tests.unit.fisheye.test_subject_position_source_handle import _prepared


RECORDING_ID = "recording-provider-track-source"
TIMELINE_ID = "timeline-provider-track-source-v1"
SUBJECT_ID = "fish-001"


def _authorities() -> ProviderTrackSourceAuthorities:
    return ProviderTrackSourceAuthorities(
        recording_id=RECORDING_ID,
        timeline_authority_id=TIMELINE_ID,
        subject_identity=SUBJECT_ID,
    )


def _position(tmp_path, *, keypoint: bool = False, row_count: int = 3):
    archive = tmp_path / "analysis.zarr"
    root = zarr.open_group(
        str(archive), mode="w-", zarr_format=3, use_consolidated=False
    )
    root.attrs.update(
        {
            "recording_id": RECORDING_ID,
            "timeline_authority_id": TIMELINE_ID,
            "subject_identity": SUBJECT_ID,
        }
    )
    root.require_group(SUBJECT_POSITION_PARENT_PATH)
    prepared = _prepared(row_count)
    if keypoint:
        estimator = get_estimator_profile(KEYPOINT_ANATOMICAL_TRIAD_MEAN_ESTIMATOR_ID)
        prepared = replace(
            prepared,
            estimator_record=estimator,
            estimator_sha256=estimator_profile_digest(estimator),
        )
    plan = plan_subject_position_run(
        archive,
        prepared,
        run_name="position_provider_track_source",
        scratch_root=tmp_path / "position-scratch",
    )
    publish_subject_position_run(plan)
    return archive, load_subject_position_source_handle(
        archive,
        plan.run_path,
        expected_selector_eligible=False,
        use_consolidated=True,
    )


def _tracking(
    archive,
    position,
    tmp_path,
    *,
    keys: np.ndarray | None = None,
    arenas: np.ndarray | None = None,
    frames: np.ndarray | None = None,
):
    keys = (
        np.asarray(position.instance_key[:], dtype=np.uint64)
        if keys is None
        else np.asarray(keys, dtype=np.uint64)
    )
    source_keys = np.asarray(position.instance_key[:], dtype=np.uint64)
    source_frames = np.asarray(
        position.source_acquisition_frame_index[:], dtype=np.int64
    )
    row_by_key = {int(key): row for row, key in enumerate(source_keys.tolist())}
    frames = (
        np.asarray([source_frames[row_by_key[int(key)]] for key in keys], dtype=np.int64)
        if frames is None
        else np.asarray(frames, dtype=np.int64)
    )
    arenas = (
        np.zeros(keys.size, dtype=np.int32)
        if arenas is None
        else np.asarray(arenas, dtype=np.int32)
    )
    root = zarr.open_group(
        str(archive), mode="r+", zarr_format=3, use_consolidated=False
    )
    run_name, _run, _summary = write_single_subject_per_arena_tracking_run(
        root=root,
        arena_ids=arenas,
        frame_indices=frames,
        source_detect_run="detect_provider_track_source",
        source_arena_assignment_run="arena_provider_track_source",
        source_rowset_path="detect_runs/detect_provider_track_source",
        instance_key=keys,
        exact_run_name="tracking_provider_track_source",
        stage_selector_eligible=False,
    )
    return load_tracking_source_handle(
        archive,
        f"tracking_runs/{run_name}",
        expected_selector_eligible=False,
        use_consolidated=False,
    )


@pytest.mark.parametrize("keypoint", [False, True], ids=["detection", "keypoint"])
def test_detection_and_keypoint_modalities_join_by_exact_instance_key(
    tmp_path, keypoint
) -> None:
    archive, position = _position(tmp_path, keypoint=keypoint)
    tracking = _tracking(
        archive,
        position,
        tmp_path,
        keys=np.asarray([102, 100, 101], dtype=np.uint64),
    )

    samples, evidence = build_provider_track_source(
        position,
        tracking,
        authorities=_authorities(),
    )

    assert samples.track_sample_key.tolist() == [[0, 0], [0, 1], [0, 2]]
    assert samples.acquisition_frame.tolist() == [0, 1, 2]
    assert samples.subject_identity == (SUBJECT_ID,) * 3
    assert samples.provider_reason_code == ("ok", "provider_invalid", "provider_invalid")
    assert evidence.record["provider"]["source_modality"] == (
        "keypoint" if keypoint else "detection"
    )
    assert evidence.source_id == evidence.sha256
    assert evidence.track_sample_policy_id == TRACK_SAMPLE_POLICY_ID
    assert evidence.record_sha256 == evidence.sha256
    position_source = evidence.record["subject_position_source"]
    assert "failure_reason_codes" not in position_source
    assert "failure_reason_tags" not in position_source
    assert position_source["arrays"]["failure_reason_code"]["array_path"] == (
        "failure_reason_codes"
    )
    assert sum(position_source["failure_reason_counts"].values()) == 3
    assert evidence.record["tracking_source"]["arrays"]["track_id"][
        "array_path"
    ] == "track_ids"
    assert "provider_reason_tags" not in evidence.record["keyed_join"]


def test_unassigned_tracking_rows_are_explicitly_excluded_after_full_key_join(
    tmp_path,
) -> None:
    archive, position = _position(tmp_path)
    tracking = _tracking(
        archive,
        position,
        tmp_path,
        arenas=np.asarray([0, -1, 0], dtype=np.int32),
    )

    samples, evidence = build_provider_track_source(
        position,
        tracking,
        authorities=_authorities(),
    )

    assert samples.acquisition_frame.tolist() == [0, 2]
    assert evidence.record["keyed_join"]["unassigned_row_count"] == 1
    assert evidence.record["tracking_source"]["unassigned_track_id"] == -1


def test_partial_keyed_rowsets_are_rejected_even_when_cardinality_matches(tmp_path) -> None:
    archive, position = _position(tmp_path)
    tracking = _tracking(
        archive,
        position,
        tmp_path,
        keys=np.asarray([100, 101, 999], dtype=np.uint64),
        frames=np.asarray([0, 1, 2], dtype=np.int64),
        arenas=np.asarray([0, 0, 0], dtype=np.int32),
    )

    with pytest.raises(ProviderSpatialTrackSourceError, match="different keyed rowsets"):
        build_provider_track_source(position, tracking, authorities=_authorities())


def test_duplicate_assigned_frame_is_rejected(tmp_path) -> None:
    archive, position = _position(tmp_path)
    with pytest.raises(TrackingConflictError, match="multiple detections"):
        _tracking(
            archive,
            position,
            tmp_path,
            frames=np.asarray([0, 0, 2], dtype=np.int64),
        )


def test_explicit_recording_authority_and_archive_are_checked(tmp_path) -> None:
    archive, position = _position(tmp_path)
    tracking = _tracking(archive, position, tmp_path)
    wrong = ProviderTrackSourceAuthorities(
        recording_id="another-recording",
        timeline_authority_id=TIMELINE_ID,
        subject_identity=SUBJECT_ID,
    )
    with pytest.raises(ProviderSpatialTrackSourceError, match="recording authority"):
        build_provider_track_source(position, tracking, authorities=wrong)

    other_archive, other_position = _position(tmp_path / "other")
    other_tracking = _tracking(other_archive, other_position, tmp_path / "other")
    with pytest.raises(ProviderSpatialTrackSourceError, match="different analysis archives"):
        build_provider_track_source(position, other_tracking, authorities=_authorities())


def test_stale_tracking_handle_is_revalidated_before_join(tmp_path) -> None:
    archive, position = _position(tmp_path)
    tracking = _tracking(archive, position, tmp_path)
    direct = zarr.open_group(
        str(archive), mode="r+", zarr_format=3, use_consolidated=False
    )
    direct[f"{tracking.run_path}/track_ids"][0] = np.int64(99)

    with pytest.raises(ProviderSpatialTrackSourceError, match="revalidation"):
        build_provider_track_source(position, tracking, authorities=_authorities())


def test_non_handle_inputs_cannot_enter_the_adapter(tmp_path) -> None:
    archive, position = _position(tmp_path)
    tracking = _tracking(archive, position, tmp_path)
    with pytest.raises(ProviderSpatialTrackSourceError, match="revalidation"):
        build_provider_track_source(
            object(),
            tracking,
            authorities=_authorities(),
        )
