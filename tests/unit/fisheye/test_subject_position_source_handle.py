from __future__ import annotations

import numpy as np
import pytest
import zarr

from fisheye.analysis_workflows.materializers.subject_position import (
    SUBJECT_POSITION_PARENT_PATH,
    SubjectPositionPreparedInput,
    plan_subject_position_run,
    publish_subject_position_run,
)
from fisheye.analysis_workflows.subject_position_source_handle import (
    SubjectPositionSourceHandle,
    SubjectPositionSourceHandleError,
    load_subject_position_source_handle,
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
from fisheye.shared.subject_position_types import (
    POSITION_FAILURE_REASON_CODES,
    empty_position_xy,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


def _coordinate(row_count: int) -> dict[str, object]:
    keys = np.arange(100, 100 + row_count, dtype=np.uint64)
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
        row_identity_record_ref="/analysis/subject_position_source@row_identity_contract",
        overlay_transform_refs=(),
        frame_record=CanonicalFrameRecord(
            kind=PIXEL_FRAME_AUTHORITY_RECORD_KIND,
            record_ref=frame.record_ref,
            record_sha256=frame.record_sha256,
        ),
    )
    return canonical_source_camera_coordinate_metadata(descriptor)


def _prepared(row_count: int = 2) -> SubjectPositionPreparedInput:
    positions = empty_position_xy(row_count)
    valid = np.zeros(row_count, dtype=np.bool_)
    reasons = np.full(
        row_count,
        POSITION_FAILURE_REASON_CODES["source_observation_rejected"],
        dtype=np.uint16,
    )
    if row_count:
        positions[0] = (12.5, 24.25)
        valid[0] = True
        reasons[0] = POSITION_FAILURE_REASON_CODES["ok"]
    arrays = {
        "position_xy": positions,
        "valid": valid,
        "failure_reason_codes": reasons,
        "instance_key": np.arange(100, 100 + row_count, dtype=np.uint64),
        "source_acquisition_frame_index": np.arange(row_count, dtype=np.int64),
        "source_row_index": np.arange(row_count, dtype=np.int64),
    }
    records = {
        "anatomy": {"anatomy_profile_id": None, "record_id": "none.v1"},
        "source": {
            "run_path": "detect_runs/detect_authorized",
            "array_paths": ["instance_key", "bbox_img_xyxy"],
            "row_axis": "observation_instance",
        },
        "policy": {
            "provider_selection": "explicit_source_adapter",
            "validity": "upstream_authority_required.v1",
        },
        "software": {"package": "palette", "commit": "c" * 40},
    }
    estimator = get_estimator_profile(DETECTION_BBOX_CENTROID_ESTIMATOR_ID)
    coordinate = _coordinate(row_count)
    return SubjectPositionPreparedInput(
        arrays=arrays,
        estimator_record=estimator,
        estimator_sha256=estimator_profile_digest(estimator),
        anatomy_record=records["anatomy"],
        anatomy_sha256=canonical_json_sha256(records["anatomy"]),
        source_record=records["source"],
        source_sha256=canonical_json_sha256(records["source"]),
        policy_record=records["policy"],
        policy_sha256=canonical_json_sha256(records["policy"]),
        software_record=records["software"],
        software_sha256=canonical_json_sha256(records["software"]),
        coordinate_record=coordinate,
        coordinate_sha256=canonical_json_sha256(coordinate),
    )


def _published(tmp_path):
    archive = tmp_path / "analysis.zarr"
    root = zarr.open_group(str(archive), mode="w", zarr_format=3)
    root.require_group(SUBJECT_POSITION_PARENT_PATH)
    plan = plan_subject_position_run(
        archive,
        _prepared(),
        run_name="position_handle_canary_001",
        scratch_root=tmp_path / "scratch",
    )
    publish_subject_position_run(plan, keep_scratch=True)
    return archive, plan


def test_loader_seals_exact_complete_ineligible_run_and_binds_nodes(tmp_path) -> None:
    archive, plan = _published(tmp_path)

    handle = load_subject_position_source_handle(
        archive,
        plan.run_path,
        expected_selector_eligible=False,
    )

    assert isinstance(handle, SubjectPositionSourceHandle)
    assert handle.analysis_zarr_path == archive.resolve()
    assert handle.run_path == plan.run_path
    assert handle.run_name == plan.run_name
    assert handle.selector_eligible is False
    assert handle.row_count == 2
    assert handle.manifest_sha256 == plan.final_manifest_sha256
    assert handle.decoded_content_sha256
    assert handle.position_xy_node.shape == (2, 2)
    assert handle.instance_key_node.shape == (2,)
    assert handle.position_xy is handle.position_xy_node
    assert handle.source_points_xy_node is None
    assert handle.estimator_sha256 == handle.manifest["payload"]["estimator"]["sha256"]
    assert (
        handle.coordinate_sha256 == handle.manifest["payload"]["coordinate"]["sha256"]
    )
    with pytest.raises(TypeError):
        handle.array_nodes["position_xy"] = handle.position_xy_node


def test_expected_eligibility_is_required_and_wrong_disposition_fails(tmp_path) -> None:
    archive, plan = _published(tmp_path)

    with pytest.raises(TypeError):
        load_subject_position_source_handle(archive, plan.run_path)  # type: ignore[call-arg]
    with pytest.raises(SubjectPositionSourceHandleError, match="eligibility"):
        load_subject_position_source_handle(
            archive,
            plan.run_path,
            expected_selector_eligible=True,
        )


def test_loader_rejects_noncanonical_or_selector_paths(tmp_path) -> None:
    archive, plan = _published(tmp_path)
    bad_paths = (
        "analysis/subject_position_runs/track_sample/position_handle_canary_001",
        f"{plan.run_path}/",
        "analysis/subject_position_runs/observation/latest",
        "/" + plan.run_path,
    )
    for bad_path in bad_paths:
        with pytest.raises(SubjectPositionSourceHandleError):
            load_subject_position_source_handle(
                archive,
                bad_path,
                expected_selector_eligible=False,
            )


def test_loader_fails_on_stale_manifest_digest_and_incomplete_status(tmp_path) -> None:
    archive, plan = _published(tmp_path)

    direct_root = zarr.open_group(
        str(archive), mode="r+", zarr_format=3, use_consolidated=False
    )
    run = direct_root[plan.run_path]
    run.attrs["subject_position_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="digest"):
        load_subject_position_source_handle(
            archive,
            plan.run_path,
            expected_selector_eligible=False,
            use_consolidated=False,
        )

    run.attrs["subject_position_manifest_sha256"] = plan.final_manifest_sha256
    run.attrs["palette_run_completion_status"] = "running"
    with pytest.raises(ValueError, match="status|complete"):
        load_subject_position_source_handle(
            archive,
            plan.run_path,
            expected_selector_eligible=False,
            use_consolidated=False,
        )


def test_direct_construction_is_sealed(tmp_path) -> None:
    with pytest.raises(SubjectPositionSourceHandleError, match="constructed"):
        SubjectPositionSourceHandle(
            analysis_zarr_path=tmp_path / "analysis.zarr",
            run_path=f"{SUBJECT_POSITION_PARENT_PATH}/run",
            run_name="run",
            manifest={},
            manifest_sha256="0" * 64,
            decoded_content_sha256="0" * 64,
            estimator_record={},
            estimator_sha256="0" * 64,
            policy_record={},
            policy_sha256="0" * 64,
            source_record={},
            source_sha256="0" * 64,
            anatomy_record={},
            anatomy_sha256="0" * 64,
            coordinate_record={},
            coordinate_sha256="0" * 64,
            row_count=0,
            array_nodes={},
            selector_eligible=False,
        )
