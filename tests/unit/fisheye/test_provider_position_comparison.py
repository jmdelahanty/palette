from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pytest
import zarr

from fisheye.analysis_workflows.materializers.provider_position_comparison import (
    ProviderPositionComparisonError,
    build_provider_position_comparison,
    plan_provider_position_comparison_run,
    publish_provider_position_comparison_run,
)
from fisheye.analysis_workflows.materializers.subject_position import (
    SUBJECT_POSITION_PARENT_PATH,
    SubjectPositionPreparedInput,
    plan_subject_position_run,
    publish_subject_position_run,
)
from fisheye.analysis_workflows.subject_position_source_handle import (
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
from fisheye.utils.materialize_position_provider_comparison_canary import (
    DETECTION_SOURCE_KIND,
    KEYPOINT_SOURCE_KIND,
    TASK_SCHEMA_ID,
    TASK_SCHEMA_VERSION,
    PositionProviderCanaryError,
    load_task,
)


def _prepared(
    *,
    keys: np.ndarray,
    frames: np.ndarray,
    positions: np.ndarray,
    valid: np.ndarray,
    source_id: str,
) -> SubjectPositionPreparedInput:
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
        row_identity_record_ref=f"/{source_id}@row_identity_contract",
        overlay_transform_refs=(),
        frame_record=CanonicalFrameRecord(
            kind=PIXEL_FRAME_AUTHORITY_RECORD_KIND,
            record_ref=frame.record_ref,
            record_sha256=frame.record_sha256,
        ),
    )
    coordinate = canonical_source_camera_coordinate_metadata(descriptor)
    reasons = np.where(
        valid,
        POSITION_FAILURE_REASON_CODES["ok"],
        POSITION_FAILURE_REASON_CODES["source_observation_rejected"],
    ).astype(np.uint16)
    canonical_positions = empty_position_xy(keys.size)
    canonical_positions[valid] = np.asarray(positions, dtype=np.float32)[valid]
    arrays = {
        "position_xy": canonical_positions,
        "valid": np.asarray(valid, dtype=bool),
        "failure_reason_codes": reasons,
        "instance_key": np.asarray(keys, dtype=np.uint64),
        "source_acquisition_frame_index": np.asarray(frames, dtype=np.int64),
        "source_row_index": np.arange(keys.size, dtype=np.int64),
    }
    estimator = get_estimator_profile(DETECTION_BBOX_CENTROID_ESTIMATOR_ID)
    records = {
        "anatomy": {"anatomy_profile_id": None, "record_id": "none.v1"},
        "source": {"source_id": source_id},
        "policy": {"selection": "none", "fallback": "none"},
        "software": {"package": "palette", "commit": "c" * 40},
    }
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


def _publish_position(
    archive: Path,
    tmp_path: Path,
    *,
    run_name: str,
    keys: list[int],
    frames: list[int],
    positions: list[list[float]],
    valid: list[bool],
):
    plan = plan_subject_position_run(
        archive,
        _prepared(
            keys=np.asarray(keys, dtype=np.uint64),
            frames=np.asarray(frames, dtype=np.int64),
            positions=np.asarray(positions, dtype=np.float32),
            valid=np.asarray(valid, dtype=bool),
            source_id=run_name,
        ),
        run_name=run_name,
        scratch_root=tmp_path / f"scratch_{run_name}",
    )
    publish_subject_position_run(plan, keep_scratch=False)
    return load_subject_position_source_handle(
        archive,
        plan.run_path,
        expected_selector_eligible=False,
        use_consolidated=True,
    )


def _providers(tmp_path: Path, *, conflicting_frame: bool = False):
    archive = tmp_path / "analysis.zarr"
    root = zarr.open_group(str(archive), mode="w", zarr_format=3)
    root.require_group(SUBJECT_POSITION_PARENT_PATH)
    left = _publish_position(
        archive,
        tmp_path,
        run_name="left_position",
        keys=[10, 20],
        frames=[1, 2],
        positions=[[10.0, 20.0], [30.0, 40.0]],
        valid=[True, True],
    )
    right = _publish_position(
        archive,
        tmp_path,
        run_name="right_position",
        keys=[20, 30],
        frames=[99 if conflicting_frame else 2, 3],
        positions=[[33.0, 44.0], [50.0, 60.0]],
        valid=[True, False],
    )
    return archive, [("left", left), ("right", right)]


def test_union_comparison_preserves_presence_validity_and_failure(
    tmp_path: Path,
) -> None:
    _archive, providers = _providers(tmp_path)

    provider_ids, arrays, summary = build_provider_position_comparison(providers)

    assert provider_ids == ("left", "right")
    np.testing.assert_array_equal(arrays["rows/instance_key"], [10, 20, 30])
    np.testing.assert_array_equal(
        arrays["provider_present"], [[True, True, False], [False, True, True]]
    )
    np.testing.assert_array_equal(
        arrays["provider_valid"], [[True, True, False], [False, True, False]]
    )
    np.testing.assert_array_equal(arrays["pair_both_valid"], [[False, True, False]])
    assert arrays["pair_distance_px"][0, 1] == pytest.approx(5.0)
    assert np.isnan(arrays["pair_distance_px"][0, 0])
    assert summary["union_row_count"] == 3
    assert summary["providers"][0]["row_count"] == 2
    assert summary["providers"][1]["valid_count"] == 1
    assert summary["pairs"][0]["both_present_count"] == 1


def test_comparison_rejects_same_key_with_conflicting_frame_identity(
    tmp_path: Path,
) -> None:
    _archive, providers = _providers(tmp_path, conflicting_frame=True)
    with pytest.raises(ProviderPositionComparisonError, match="frame identity"):
        build_provider_position_comparison(providers)


def test_comparison_publication_is_immutable_and_nonpromoting(tmp_path: Path) -> None:
    archive, providers = _providers(tmp_path)
    plan = plan_provider_position_comparison_run(
        archive,
        providers,
        run_name="comparison_canary_v1",
        scratch_root=tmp_path / "comparison_scratch",
        software_record={"package": "palette", "commit": "d" * 40},
    )

    result = publish_provider_position_comparison_run(plan)

    assert result["status"] == "complete"
    assert result["selector_eligible"] is False
    assert result["selection"] == "none"
    root = zarr.open_group(str(archive), mode="r", use_consolidated=True)
    parent = root["analysis/provider_position_comparison_runs"]
    assert "latest" not in parent.attrs
    run = parent["comparison_canary_v1"]
    assert run.attrs["stage_selector_eligible"] is False
    np.testing.assert_array_equal(run["rows/instance_key"][:], [10, 20, 30])
    np.testing.assert_array_equal(
        run["provider_present"][:],
        [[True, True, False], [False, True, True]],
    )


def test_canary_task_requires_explicit_compatible_sources(tmp_path: Path) -> None:
    recording_id = "recording_001"
    archive = tmp_path / f"{recording_id}_analysis.zarr"
    zarr.open_group(str(archive), mode="w", zarr_format=3)
    task = {
        "schema_id": TASK_SCHEMA_ID,
        "schema_version": TASK_SCHEMA_VERSION,
        "recording_id": recording_id,
        "analysis_zarr": str(archive),
        "anatomy_profile": str(
            Path("configs/fisheye/anatomy_profiles/zebrafish_larva_v1.json").resolve()
        ),
        "software": {"package": "palette", "commit": "e" * 40},
        "providers": [
            {
                "provider_id": "detection",
                "estimator_id": DETECTION_BBOX_CENTROID_ESTIMATOR_ID,
                "position_run_name": "detection_position",
                "source": {
                    "kind": DETECTION_SOURCE_KIND,
                    "run_path": "detect_runs/canonical_v3",
                },
            },
            {
                "provider_id": "keypoint",
                "estimator_id": "keypoint_anatomical_triad_mean.v1",
                "position_run_name": "keypoint_position",
                "source": {
                    "kind": KEYPOINT_SOURCE_KIND,
                    "run_path": "keypoints_runs/coordinate_v2",
                    "binding_id": "zebrafish_larva_keypoint_traditional_v2_v1",
                },
            },
        ],
        "comparison_run_name": "provider_comparison_v1",
    }
    path = tmp_path / "task.json"
    path.write_text(json.dumps(task), encoding="utf-8")

    loaded = load_task(path)
    assert loaded["providers"][0]["source"]["run_path"] == ("detect_runs/canonical_v3")
    task["providers"][0]["estimator_id"] = "subject_body_mask_centroid.v1"
    path.write_text(json.dumps(task), encoding="utf-8")
    with pytest.raises(PositionProviderCanaryError, match="incompatible"):
        load_task(path)
