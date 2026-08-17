from __future__ import annotations

import numpy as np
import pytest
import zarr

from fisheye.analysis_workflows.materializers.subject_position import (
    SUBJECT_POSITION_PARENT_PATH,
    SubjectPositionPreparedInput,
    build_subject_position_manifest,
    plan_subject_position_run,
    publish_subject_position_run,
    subject_position_manifest_digest,
    validate_subject_position_manifest,
    validate_subject_position_run,
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
    get_estimator_profile,
    estimator_profile_digest,
)
from fisheye.shared.subject_position_types import POSITION_FAILURE_REASON_CODES, empty_position_xy
from fisheye.shared.subject_position_storage import canonical_source_camera_coordinate_metadata
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


def _arrays(row_count: int) -> dict[str, np.ndarray]:
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
    return {
        "position_xy": positions,
        "valid": valid,
        "failure_reason_codes": reasons,
        "instance_key": np.arange(100, 100 + row_count, dtype=np.uint64),
        "source_acquisition_frame_index": np.arange(row_count, dtype=np.int64),
        "source_row_index": np.arange(row_count, dtype=np.int64),
    }


def _prepared(row_count: int = 2) -> SubjectPositionPreparedInput:
    estimator = get_estimator_profile(DETECTION_BBOX_CENTROID_ESTIMATOR_ID)
    coordinate = _coordinate(row_count)
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
        "software": {
            "package": "palette",
            "commit": "c" * 40,
        },
    }
    return SubjectPositionPreparedInput(
        arrays=_arrays(row_count),
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


def test_preparation_copies_arrays_and_accepts_only_registered_estimator() -> None:
    arrays = _arrays(2)
    prepared = _prepared()
    arrays["position_xy"][0, 0] = 99.0
    assert prepared.arrays["position_xy"][0, 0] == 12.5
    assert prepared.arrays["position_xy"].flags.writeable is False

    bad = get_estimator_profile(DETECTION_BBOX_CENTROID_ESTIMATOR_ID)
    bad["estimator_id"] = "union_not_registered.v1"
    with pytest.raises(ValueError, match="Unknown estimator"):
        SubjectPositionPreparedInput(
            arrays=arrays,
            estimator_record=bad,
            estimator_sha256="0" * 64,
            anatomy_record=prepared.anatomy_record,
            anatomy_sha256=prepared.anatomy_sha256,
            source_record=prepared.source_record,
            source_sha256=prepared.source_sha256,
            policy_record=prepared.policy_record,
            policy_sha256=prepared.policy_sha256,
            software_record=prepared.software_record,
            software_sha256=prepared.software_sha256,
            coordinate_record=prepared.coordinate_record,
            coordinate_sha256=prepared.coordinate_sha256,
        )


def test_plan_does_not_resolve_latest_and_captures_selector_attrs(tmp_path) -> None:
    archive = tmp_path / "analysis.zarr"
    root = zarr.open_group(str(archive), mode="w", zarr_format=3)
    parent = root.require_group(SUBJECT_POSITION_PARENT_PATH)
    parent.attrs["latest"] = "old_authority"
    parent.attrs["latest_complete"] = "old_authority"
    scratch = tmp_path / "scratch"
    plan = plan_subject_position_run(
        archive,
        _prepared(),
        run_name="position_attempt_001",
        scratch_root=scratch,
    )
    assert plan.run_path.endswith("/position_attempt_001")
    assert plan.parent_selector_attrs == {
        "latest": "old_authority",
        "latest_complete": "old_authority",
    }
    assert not plan.local_zarr.exists()


def test_empty_recording_manifest_is_valid(tmp_path) -> None:
    archive = tmp_path / "analysis.zarr"
    zarr.open_group(str(archive), mode="w", zarr_format=3)
    plan = plan_subject_position_run(
        archive,
        _prepared(0),
        run_name="position_empty",
        scratch_root=tmp_path / "scratch",
    )
    manifest = build_subject_position_manifest(plan, status="complete")
    result = validate_subject_position_manifest(
        manifest,
        expected_run_name="position_empty",
        expected_status="complete",
    )
    assert result["valid"] is True
    assert subject_position_manifest_digest(manifest) == manifest["payload_digest"]
    assert manifest["payload"]["decoded_content_sha256"]


def test_real_zarr_publication_is_complete_ineligible_and_nonpromoting(tmp_path) -> None:
    archive = tmp_path / "analysis.zarr"
    root = zarr.open_group(str(archive), mode="w", zarr_format=3)
    parent = root.require_group(SUBJECT_POSITION_PARENT_PATH)
    parent.attrs["latest"] = "preexisting"
    parent.attrs["latest_complete"] = "preexisting"
    plan = plan_subject_position_run(
        archive,
        _prepared(2),
        run_name="position_publication_001",
        scratch_root=tmp_path / "scratch",
    )
    result = publish_subject_position_run(plan, keep_scratch=True)
    assert result["acceptance"]["consolidated_validation"]["valid"] is True
    final = validate_subject_position_run(
        archive,
        plan.run_path,
        use_consolidated=True,
        expected_status="complete",
        expected_manifest_sha256=plan.final_manifest_sha256,
    )
    assert final["row_count"] == 2
    published_parent = zarr.open_group(
        str(archive), mode="r", zarr_format=3, use_consolidated=True
    )[SUBJECT_POSITION_PARENT_PATH]
    assert published_parent.attrs["latest"] == "preexisting"
    assert published_parent.attrs["latest_complete"] == "preexisting"
    published = published_parent[plan.run_name]
    assert published.attrs["stage_selector_eligible"] is False
    assert published.attrs["palette_run_completion_status"] == "complete"
    with pytest.raises(FileExistsError):
        plan_subject_position_run(
            archive,
            _prepared(2),
            run_name=plan.run_name,
            scratch_root=tmp_path / "retry-scratch",
        )
