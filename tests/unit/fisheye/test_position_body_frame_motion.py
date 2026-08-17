from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.analysis_workflows.body_frame_source_handle import (
    load_body_frame_source_handle,
)
from fisheye.analysis_workflows.materializers.subject_position import (
    SUBJECT_POSITION_PARENT_PATH,
    SubjectPositionPreparedInput,
    plan_subject_position_run,
    publish_subject_position_run,
)
from fisheye.analysis_workflows.position_body_frame_motion import (
    DETECTION_CENTROID_TRADITIONAL_V3_COMPATIBILITY_PROFILE_ID,
    PositionBodyFrameMotionError,
    _exact_key_join,
    bind_position_body_frame_to_tracking,
    compose_detection_centroid_traditional_v3_compatibility_authority,
    compose_position_body_frame_motion_authority,
)
from fisheye.analysis_workflows.subject_position_source_handle import (
    load_subject_position_source_handle,
)
from fisheye.analysis_workflows.tracking_source_handle import (
    load_tracking_source_handle,
)
from fisheye.shared.coordinate_descriptor import (
    CanonicalFrameRecord,
    DigestBoundCoordinateRecordRef,
    PIXEL_FRAME_AUTHORITY_RECORD_KIND,
    build_canonical_coordinate_descriptor,
)
from fisheye.shared.coordinate_identity import (
    OBSERVATION_INSTANCE_DOMAIN,
    build_row_identity_contract,
)
from fisheye.shared.coordinate_surface_contract import SOURCE_CAMERA_POINT_XY
from fisheye.shared.subject_position_expression import (
    DETECTION_BBOX_CENTROID_ESTIMATOR_ID,
    estimator_profile_digest,
    get_estimator_profile,
)
from fisheye.shared.subject_position_storage import (
    canonical_source_camera_coordinate_metadata,
)
from fisheye.shared.subject_position_types import POSITION_FAILURE_REASON_CODES
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.tracking.single_subject_per_arena import (
    write_single_subject_per_arena_tracking_run,
)
from fisheye.shared.traditional_heading_compatibility import (
    load_traditional_v3_heading_compatibility,
)
from tests.unit.fisheye.test_body_frame_source_handle import _published_fixture


def _position_coordinate(keys: np.ndarray) -> dict[str, object]:
    frame = DigestBoundCoordinateRecordRef(
        record_ref="/coordinate_frames/source_camera@pixel_frame_authority",
        record_sha256="a" * 64,
    )
    descriptor = build_canonical_coordinate_descriptor(
        **SOURCE_CAMERA_POINT_XY.descriptor_kwargs(),
        reference_width=640,
        reference_height=480,
        reference_authority=frame,
        reference_selector="record",
        row_identity_contract=build_row_identity_contract(
            domain=OBSERVATION_INSTANCE_DOMAIN,
            values=keys,
        ),
        row_identity_record_ref=(
            "/analysis/subject_position_source@row_identity_contract"
        ),
        overlay_transform_refs=(),
        frame_record=CanonicalFrameRecord(
            kind=PIXEL_FRAME_AUTHORITY_RECORD_KIND,
            record_ref=frame.record_ref,
            record_sha256=frame.record_sha256,
        ),
    )
    return canonical_source_camera_coordinate_metadata(descriptor)


def _publish_matching_position(destination, publication, tmp_path):  # type: ignore[no-untyped-def]
    keys = np.asarray(publication.prepared.arrays["instance_key"], dtype=np.uint64)
    frames = np.asarray(publication.prepared.arrays["frame_indices"], dtype=np.int64)
    rows = keys.shape[0]
    positions = np.column_stack(
        (
            np.arange(rows, dtype=np.float32) + 20.0,
            np.arange(rows, dtype=np.float32) + 30.0,
        )
    )
    estimator = get_estimator_profile(DETECTION_BBOX_CENTROID_ESTIMATOR_ID)
    anatomy = {"record_id": "non_anatomical_detection.v1"}
    source = {
        "run_path": "refined_detect_runs/refined_fixture",
        "row_axis": "observation_instance",
    }
    policy = {
        "policy_id": "subject_position_canary_no_default.v1",
        "fallback": "none",
    }
    software = {"package": "palette", "commit": "c" * 40}
    coordinate = _position_coordinate(keys)
    prepared = SubjectPositionPreparedInput(
        arrays={
            "position_xy": positions,
            "valid": np.ones(rows, dtype=bool),
            "failure_reason_codes": np.full(
                rows, POSITION_FAILURE_REASON_CODES["ok"], dtype=np.uint16
            ),
            "instance_key": keys,
            "source_acquisition_frame_index": frames,
            "source_row_index": np.arange(rows, dtype=np.int64),
        },
        estimator_record=estimator,
        estimator_sha256=estimator_profile_digest(estimator),
        anatomy_record=anatomy,
        anatomy_sha256=canonical_json_sha256(anatomy),
        source_record=source,
        source_sha256=canonical_json_sha256(source),
        policy_record=policy,
        policy_sha256=canonical_json_sha256(policy),
        software_record=software,
        software_sha256=canonical_json_sha256(software),
        coordinate_record=coordinate,
        coordinate_sha256=canonical_json_sha256(coordinate),
    )
    direct = zarr.open_group(
        str(destination), mode="r+", zarr_format=3, use_consolidated=False
    )
    direct.require_group(SUBJECT_POSITION_PARENT_PATH)
    plan = plan_subject_position_run(
        destination,
        prepared,
        run_name="position_for_body_frame_001",
        scratch_root=tmp_path / "position_scratch",
    )
    publish_subject_position_run(plan, keep_scratch=True)
    return plan


def _handles(tmp_path, **body_frame_options):  # type: ignore[no-untyped-def]
    destination, publication = _published_fixture(tmp_path, **body_frame_options)
    position_plan = _publish_matching_position(destination, publication, tmp_path)
    position = load_subject_position_source_handle(
        destination,
        position_plan.run_path,
        expected_selector_eligible=False,
    )
    body_frame = load_body_frame_source_handle(
        destination,
        run_path="analysis/body_frame_runs/body_frame_v1_001",
        expected_selector_eligible=False,
    )
    return position, body_frame


def _tracking_handle(
    source,
    *,
    keys: np.ndarray | None = None,
    arena_ids: np.ndarray | None = None,
):  # type: ignore[no-untyped-def]
    ordered_keys = (
        np.asarray(keys, dtype=np.uint64)
        if keys is not None
        else source.instance_key[[2, 0, 1]]
    )
    arenas = (
        np.asarray(arena_ids, dtype=np.int32)
        if arena_ids is not None
        else np.asarray([8, 7, 7], dtype=np.int32)
    )
    source_rows = {
        int(key): row for row, key in enumerate(source.instance_key.tolist())
    }
    if all(int(key) in source_rows for key in ordered_keys):
        frames = np.asarray(
            [
                source.source_acquisition_frame_index[source_rows[int(key)]]
                for key in ordered_keys
            ],
            dtype=np.int64,
        )
    else:
        frames = np.arange(ordered_keys.shape[0], dtype=np.int64)
    root = zarr.open_group(
        str(source.analysis_zarr_path),
        mode="r+",
        zarr_format=3,
        use_consolidated=False,
    )
    run_name, _run, _summary = write_single_subject_per_arena_tracking_run(
        root=root,
        arena_ids=arenas,
        frame_indices=frames,
        source_detect_run="detect_fixture",
        source_arena_assignment_run="arena_assignment_fixture",
        source_rowset_path="refined_detect_runs/refined_fixture/instances",
        instance_key=ordered_keys,
    )
    return load_tracking_source_handle(
        source.analysis_zarr_path,
        f"tracking_runs/{run_name}",
        expected_selector_eligible=True,
        use_consolidated=False,
    )


def test_composes_exact_position_and_body_frame_lineage(tmp_path) -> None:
    position, body_frame = _handles(tmp_path)

    authority = compose_position_body_frame_motion_authority(position, body_frame)

    assert authority.row_count == 3
    assert authority.row_alignment_mode == "exact_ordered_instance_key_equality_v1"
    assert authority.position_run_path == position.run_path
    assert authority.body_frame_run_path == body_frame.run_path
    assert authority.authority_record["linear_lineage"]["position_xy_sha256"]
    assert authority.authority_record["angular_lineage"]["heading_deg_sha256"]
    assert authority.position_valid.tolist() == [True, True, True]
    assert authority.heading_valid.tolist() == [True, True, True]
    with pytest.raises(ValueError):
        authority.heading_deg[0] = 90.0


def test_exact_key_join_supports_sealed_reorder_but_not_same_length_only() -> None:
    left = np.asarray([10, 20, 30], dtype=np.uint64)
    rows, mode = _exact_key_join(
        left,
        np.asarray([30, 10, 20], dtype=np.uint64),
        left_name="left",
        right_name="right",
    )
    assert mode == "exact_instance_key_set_reorder_v1"
    assert rows.tolist() == [1, 2, 0]

    with pytest.raises(PositionBodyFrameMotionError, match="different rowsets"):
        _exact_key_join(
            left,
            np.asarray([10, 20, 40], dtype=np.uint64),
            left_name="left",
            right_name="right",
        )


def test_tracking_join_reorders_by_identity_and_builds_independent_motion(
    tmp_path,
) -> None:
    position, body_frame = _handles(tmp_path)
    authority = compose_position_body_frame_motion_authority(position, body_frame)
    tracking = _tracking_handle(authority)

    tracked = bind_position_body_frame_to_tracking(authority, tracking)

    assert tracked.tracking_row_alignment_mode == "exact_instance_key_set_reorder_v1"
    assert tracked.track_ids.tolist() == [0, 0, 1]
    tracks, _summaries = tracked.build_track_datasets(
        fps=10.0,
        smooth_seconds=0.0,
        pixel_to_mm=None,
    )
    assert set(tracks) == {0, 1}
    assert tracks[0]["sample_validity_profile"] == (
        "explicit_position_body_frame_independent_validity.v1"
    )
    assert tracks[0]["linear_sample_valid"].all()
    assert tracks[0]["angular_sample_valid"].all()


def test_tracking_join_rejects_equal_length_different_identity(tmp_path) -> None:
    position, body_frame = _handles(tmp_path)
    authority = compose_position_body_frame_motion_authority(position, body_frame)

    tracking = _tracking_handle(
        authority,
        keys=np.asarray([101, 102, 999], dtype=np.uint64),
        arena_ids=np.asarray([0, 0, 0], dtype=np.int32),
    )
    with pytest.raises(PositionBodyFrameMotionError, match="different rowsets"):
        bind_position_body_frame_to_tracking(authority, tracking)


def test_traditional_v3_compatibility_profile_requires_and_binds_receipt(
    tmp_path,
) -> None:
    root = Path(__file__).parents[3]
    receipt = load_traditional_v3_heading_compatibility(
        pose_schema_path=root / "configs/fisheye/pose_schemas/traditional_v3.json",
        anatomy_profile_path=(
            root / "configs/fisheye/anatomy_profiles/zebrafish_larva_v1.json"
        ),
        source_binding_id="zebrafish_larva_keypoint_traditional_v3_v1",
    )
    position, body_frame = _handles(
        tmp_path,
        skeleton_id=receipt.skeleton_id,
        skeleton_digest=receipt.schema_sha256,
        heading_computation=receipt.as_dict()["heading_computation"],
    )

    authority = compose_detection_centroid_traditional_v3_compatibility_authority(
        position,
        body_frame,
        compatibility=receipt,
    )

    assert authority.profile_id == (
        DETECTION_CENTROID_TRADITIONAL_V3_COMPATIBILITY_PROFILE_ID
    )
    assert authority.authority_record["compatibility_receipt_sha256"] == (
        receipt.receipt_sha256
    )


def test_traditional_v3_compatibility_rejects_different_body_frame(tmp_path) -> None:
    root = Path(__file__).parents[3]
    receipt = load_traditional_v3_heading_compatibility(
        pose_schema_path=root / "configs/fisheye/pose_schemas/traditional_v3.json",
        anatomy_profile_path=(
            root / "configs/fisheye/anatomy_profiles/zebrafish_larva_v1.json"
        ),
        source_binding_id="zebrafish_larva_keypoint_traditional_v3_v1",
    )
    position, body_frame = _handles(tmp_path)

    with pytest.raises(PositionBodyFrameMotionError, match="receipt differs"):
        compose_detection_centroid_traditional_v3_compatibility_authority(
            position,
            body_frame,
            compatibility=receipt,
        )
