from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fisheye.analysis_workflows import core_authority_roster as roster_module
from fisheye.analysis_workflows import (
    core_behavior_cohort_adapter as core_cohort_module,
)
from fisheye.analysis_workflows.core_authority_roster import (
    BoundCoreMotionAndBouts,
    CoreAuthorityRosterError,
    build_bout_authority_identity,
    bout_authority_identity_from_phase_c_tables,
    build_core_authority_consumption_receipt,
    build_core_authority_roster,
    build_subject_body_frame_source_binding,
    compare_bout_authority_identities,
    selected_core_track_id_from_roster,
    validate_bout_authority_identity,
    validate_core_authority_consumption_receipt,
    validate_core_authority_roster,
)
from fisheye.analysis_workflows.core_behavior_cohort_adapter import (
    core_authority_roster_from_bundle_set_member,
    core_behavior_capability_contract,
)
from fisheye.analysis_workflows.core_motion_source_handle import (
    CoreMotionSourceHandleError,
    CoreMotionTrackSourceHandle,
    bind_core_motion_track_source_handle,
    core_motion_dependency_record,
    require_core_motion_track_source_handle,
    validate_core_motion_dependency_record,
)
from fisheye.analysis_workflows.core_subject_body_frame_source_handle import (
    CoreSubjectBodyFrameSourceHandle,
    CoreSubjectBodyFrameSourceHandleError,
    bind_core_subject_body_frame_source_handle,
    require_core_subject_body_frame_source_handle,
)
from fisheye.analysis_workflows.chaser_relative_frame_source_handle import (
    ChaserRelativeFrameSourceHandle,
)
from fisheye.analysis_workflows.chaser_relative_frame_storage import (
    validate_prepared_chaser_relative_frame,
)
from fisheye.analysis_workflows.core_chaser_relative_frame_adapter import (
    CORE_CHASER_RELATIVE_CONSUMER_ID,
    CoreChaserRelativeFrameAdapterError,
    prepare_core_chaser_relative_frame,
)
from fisheye.analytics_exports.validated_behavior_contracts import (
    CORE_TABLE_SPECS,
    compose_disjoint_table_specs,
)
from fisheye.analytics_exports.validated_behavior_core_behavior_contracts import (
    CORE_BEHAVIOR_CAPABILITY_KEYS,
    CROSS_GRAIN_JOIN_AUTHORITY,
    SUBJECT_BODY_FRAME_CAPABILITY,
    SUBJECT_BODY_FRAME_SOURCE_PROFILE_ID,
)
from fisheye.shared.subject_shape_coordinate_publication import (
    BoundSubjectShapeCoordinatePublication,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


def _sealed(**values: object) -> dict[str, object]:
    return {**values, "payload_sha256": canonical_json_sha256(values)}


def _capability_bindings(tmp_path: Path) -> dict[str, object]:
    join = _sealed(
        schema_id="palette.validated_behavior.cross_grain_join_authority",
        schema_version=1,
        recording_id="recording-a",
        camera_id="camera-a",
        source_total_frames=100,
        source_sample_rate_hz=30.0,
        acquisition_camera_frame_ref="/metadata/acquisition_camera_frame",
        acquisition_camera_frame_sha256="8" * 64,
        source_video_metadata_sha256="7" * 64,
    )
    join_sha = str(join["payload_sha256"])
    bindings: dict[str, object] = {CROSS_GRAIN_JOIN_AUTHORITY: join}
    for capability in CORE_BEHAVIOR_CAPABILITY_KEYS:
        if capability == CROSS_GRAIN_JOIN_AUTHORITY:
            continue
        bindings[capability] = {
            "profile_id": f"{capability}_v1",
            "source_binding": _sealed(
                schema_id=f"fixture.{capability}.source",
                schema_version=1,
                recording_id="recording-a",
                zarr_path=str((tmp_path / "recording-a.zarr").resolve()),
            ),
            "projection_contract": _sealed(
                schema_id=f"fixture.{capability}.projection",
                schema_version=1,
            ),
            "join_authority_sha256": join_sha,
        }
    return bindings


def _report_binding(tmp_path: Path) -> dict[str, object]:
    return {
        "role": "core_behavior_workflow_execution",
        "path": str((tmp_path / "execution-report.json").resolve()),
        "file_sha256": "a" * 64,
        "record_sha256": "b" * 64,
        "schema_id": "palette.analysis_workflow_execution",
        "schema_version": 3,
    }


def test_core_authority_roster_is_deterministic_and_closed(tmp_path: Path) -> None:
    capabilities = _capability_bindings(tmp_path)
    kwargs = {
        "recording_id": "recording-a",
        "analysis_zarr": tmp_path / "recording-a.zarr",
        "execution_report_binding": _report_binding(tmp_path),
        "capability_bindings": capabilities,
    }

    first = build_core_authority_roster(**kwargs)
    reversed_capabilities = dict(reversed(list(capabilities.items())))
    second = build_core_authority_roster(
        **{**kwargs, "capability_bindings": reversed_capabilities}
    )

    assert first["record_sha256"] == second["record_sha256"]
    assert (
        validate_core_authority_roster(first)["record_sha256"] == first["record_sha256"]
    )
    assert set(first["capability_bindings"]) == set(CORE_BEHAVIOR_CAPABILITY_KEYS)


def test_core_authority_roster_rejects_nested_tamper(tmp_path: Path) -> None:
    roster = build_core_authority_roster(
        recording_id="recording-a",
        analysis_zarr=tmp_path / "recording-a.zarr",
        execution_report_binding=_report_binding(tmp_path),
        capability_bindings=_capability_bindings(tmp_path),
    )
    tampered = deepcopy(roster)
    tampered["capability_bindings"]["kinematics_samples"]["source_binding"][
        "recording_id"
    ] = "recording-b"
    body = {key: value for key, value in tampered.items() if key != "record_sha256"}
    tampered["record_sha256"] = canonical_json_sha256(body)

    with pytest.raises(CoreAuthorityRosterError, match="payload_sha256 is stale"):
        validate_core_authority_roster(tampered)


def test_shared_roster_reader_requires_the_frozen_digest(tmp_path: Path) -> None:
    roster = build_core_authority_roster(
        recording_id="recording-a",
        analysis_zarr=tmp_path / "recording-a.zarr",
        execution_report_binding=_report_binding(tmp_path),
        capability_bindings=_capability_bindings(tmp_path),
    )
    path = tmp_path / "core-authority-roster.json"
    path.write_text(json.dumps(roster), encoding="utf-8")

    observed = roster_module.read_core_authority_roster(
        path,
        expected_record_sha256=roster["record_sha256"],
    )
    assert observed["record_sha256"] == roster["record_sha256"]
    with pytest.raises(CoreAuthorityRosterError, match="differs from the frozen task"):
        roster_module.read_core_authority_roster(
            path,
            expected_record_sha256="f" * 64,
        )


def test_paradigm_consumption_receipt_names_one_roster_and_track(
    tmp_path: Path,
) -> None:
    capabilities = _capability_bindings(tmp_path)
    capabilities["kinematics_samples"]["source_binding"] = _sealed(
        schema_id="fixture.kinematics_samples.source",
        schema_version=1,
        recording_id="recording-a",
        zarr_path=str((tmp_path / "recording-a.zarr").resolve()),
        tracks=[{"track_id": 7}],
    )
    roster = build_core_authority_roster(
        recording_id="recording-a",
        analysis_zarr=tmp_path / "recording-a.zarr",
        execution_report_binding=_report_binding(tmp_path),
        capability_bindings=capabilities,
    )
    receipt = build_core_authority_consumption_receipt(
        roster,
        consumer_id="goodbatbadbat.chaser_extension_v1",
        required_capabilities=(
            CROSS_GRAIN_JOIN_AUTHORITY,
            "kinematics_samples",
            "canonical_swim_bouts",
        ),
        selected_track_id=7,
    )

    validated = validate_core_authority_consumption_receipt(receipt, roster=roster)
    assert validated["core_authority_roster_sha256"] == roster["record_sha256"]
    assert validated["selected_track_id"] == 7
    assert set(validated["capability_binding_digests"]) == {
        CROSS_GRAIN_JOIN_AUTHORITY,
        "kinematics_samples",
        "canonical_swim_bouts",
    }


def test_paradigm_consumption_receipt_rejects_missing_join_authority(
    tmp_path: Path,
) -> None:
    capabilities = _capability_bindings(tmp_path)
    capabilities["kinematics_samples"]["source_binding"] = _sealed(
        schema_id="fixture.kinematics_samples.source",
        schema_version=1,
        recording_id="recording-a",
        zarr_path=str((tmp_path / "recording-a.zarr").resolve()),
        tracks=[{"track_id": 0}],
    )
    roster = build_core_authority_roster(
        recording_id="recording-a",
        analysis_zarr=tmp_path / "recording-a.zarr",
        execution_report_binding=_report_binding(tmp_path),
        capability_bindings=capabilities,
    )

    with pytest.raises(CoreAuthorityRosterError, match="cross-grain join"):
        build_core_authority_consumption_receipt(
            roster,
            consumer_id="goodbatbadbat.chaser_extension_v1",
            required_capabilities=("kinematics_samples",),
            selected_track_id=0,
        )


def test_selected_core_track_is_derived_from_matching_motion_and_bout_authority(
    tmp_path: Path,
) -> None:
    capabilities = _capability_bindings(tmp_path)
    capabilities["kinematics_samples"]["profile_id"] = "core_motion_physical_v2"
    capabilities["kinematics_samples"]["source_binding"] = _sealed(
        schema_id="fixture.kinematics_samples.source",
        schema_version=2,
        recording_id="recording-a",
        zarr_path=str((tmp_path / "recording-a.zarr").resolve()),
        tracks=[{"track_id": 7}],
    )
    capabilities["canonical_swim_bouts"]["source_binding"] = _sealed(
        schema_id="fixture.canonical_swim_bouts.source",
        schema_version=1,
        recording_id="recording-a",
        zarr_path=str((tmp_path / "recording-a.zarr").resolve()),
        track_id=7,
    )
    roster = build_core_authority_roster(
        recording_id="recording-a",
        analysis_zarr=tmp_path / "recording-a.zarr",
        execution_report_binding=_report_binding(tmp_path),
        capability_bindings=capabilities,
    )

    assert selected_core_track_id_from_roster(roster) == 7


def test_selected_core_track_rejects_multiple_or_mismatched_motion_tracks(
    tmp_path: Path,
) -> None:
    capabilities = _capability_bindings(tmp_path)
    capabilities["kinematics_samples"]["profile_id"] = "core_motion_physical_v2"
    capabilities["kinematics_samples"]["source_binding"] = _sealed(
        schema_id="fixture.kinematics_samples.source",
        schema_version=2,
        recording_id="recording-a",
        zarr_path=str((tmp_path / "recording-a.zarr").resolve()),
        tracks=[{"track_id": 7}, {"track_id": 9}],
    )
    capabilities["canonical_swim_bouts"]["source_binding"] = _sealed(
        schema_id="fixture.canonical_swim_bouts.source",
        schema_version=1,
        recording_id="recording-a",
        zarr_path=str((tmp_path / "recording-a.zarr").resolve()),
        track_id=7,
    )
    roster = build_core_authority_roster(
        recording_id="recording-a",
        analysis_zarr=tmp_path / "recording-a.zarr",
        execution_report_binding=_report_binding(tmp_path),
        capability_bindings=capabilities,
    )

    with pytest.raises(CoreAuthorityRosterError, match="exactly one motion track"):
        selected_core_track_id_from_roster(roster)


def test_core_roster_is_recovered_from_one_complete_generic_bundle_member(
    tmp_path: Path,
) -> None:
    capabilities = _capability_bindings(tmp_path)
    capabilities["kinematics_samples"]["source_binding"] = _sealed(
        schema_id="fixture.kinematics_samples.source",
        schema_version=2,
        recording_id="recording-a",
        zarr_path=str((tmp_path / "recording-a.zarr").resolve()),
        tracks=[{"track_id": 7}],
    )
    receipt = _report_binding(tmp_path)
    contract = core_behavior_capability_contract()
    bundle_set = {
        "bundle_profile": core_cohort_module._bundle_profile(
            contract,
            export_profile_id="validated_core_behavior_five_grain_v2",
        ),
        "capability_contract": contract,
    }
    capability_records = {
        key: {
            "state": "complete",
            "reason_code": None,
            "detail": None,
            "binding": value,
        }
        for key, value in capabilities.items()
    }
    inventory = {
        "execution_report": receipt,
        "capability_bindings": capabilities,
    }
    member = {
        "recording_id": "recording-a",
        "analysis_zarr": str((tmp_path / "recording-a.zarr").resolve()),
        "bundle_state": "complete",
        "reason_code": None,
        "bundle": {
            "receipt_bindings": [receipt],
            "binding_inventory_sha256": canonical_json_sha256(inventory),
        },
        "capabilities": capability_records,
    }

    roster = core_authority_roster_from_bundle_set_member(bundle_set, member)

    assert (
        roster["record_sha256"]
        == build_core_authority_roster(
            recording_id="recording-a",
            analysis_zarr=tmp_path / "recording-a.zarr",
            execution_report_binding=receipt,
            capability_bindings=capabilities,
        )["record_sha256"]
    )


def _bout_identity(
    tmp_path: Path,
    *,
    motion_scope: str = "offline",
    motion_run_path: str = "analysis/track_kinematics_runs/offline/motion-a",
    motion_manifest_sha256: str = "c" * 64,
    track_id: int = 0,
    events: np.ndarray | None = None,
):
    frame_axis = np.asarray([11, 12, 13], dtype=np.int64)
    if events is None:
        events = np.asarray(
            [(0, 11, 12)],
            dtype=[
                ("bout_id", "<i8"),
                ("start_frame", "<i8"),
                ("end_frame", "<i8"),
            ],
        )
    return build_bout_authority_identity(
        recording_id="recording-a",
        analysis_zarr=tmp_path / "recording-a.zarr",
        run_path="analysis/swim_bout_runs/bouts-a",
        run_schema_id="palette.swim_bout_run",
        run_schema_version=8,
        run_layout="compact_tabular_v2",
        completion_status="complete",
        selector_eligible=True,
        array_manifest_sha256="d" * 64,
        motion_scope=motion_scope,
        motion_run_path=motion_run_path,
        motion_manifest_sha256=motion_manifest_sha256,
        motion_verification_sha256="e" * 64,
        track_id=track_id,
        track_row_start=0,
        track_row_stop=3,
        source_sample_rate_hz=30.0,
        frame_axis=frame_axis,
        candidate_id=0,
        candidate_name="default",
        signal_id=0,
        signal_name="default_detector",
        signal_level="speed_smoothed",
        signal_role="detector_response",
        signal_source_level="speed_smoothed",
        selected_events=events,
        source_binding_sha256="f" * 64,
    )


def test_normalized_bout_identity_deduplicates_only_complete_equality(
    tmp_path: Path,
) -> None:
    left = _bout_identity(tmp_path)
    right = _bout_identity(tmp_path)

    result = compare_bout_authority_identities(left, right)

    assert validate_bout_authority_identity(left).record_sha256 == left.record_sha256
    assert result.status == "equal"
    assert result.reason_code == "same_normalized_bout_authority"
    assert result.differing_fields == ()
    assert result.selected_event_content_equal is True


def test_normalized_bout_identity_rejects_motion_authority_conflict_even_when_events_equal(
    tmp_path: Path,
) -> None:
    core = _bout_identity(tmp_path)
    provider = _bout_identity(
        tmp_path,
        motion_scope="provider",
        motion_run_path="analysis/track_kinematics_runs/provider/motion-a",
        motion_manifest_sha256="1" * 64,
    )

    result = compare_bout_authority_identities(core, provider)

    assert result.status == "conflict"
    assert result.reason_code == "bout_authority_identity_conflict"
    assert "motion_authority" in result.differing_fields
    assert result.selected_event_content_equal is True


def test_normalized_bout_identity_reports_missing_proof() -> None:
    result = compare_bout_authority_identities(None, None)

    assert result.status == "not_proven"
    assert result.reason_code == "missing_both_bout_authority"
    assert result.selected_event_content_equal is None


def test_phase_c_bout_identity_reopens_selected_event_content(tmp_path: Path) -> None:
    frame_axis = np.asarray([11, 12, 13], dtype=np.int64)
    events = np.asarray(
        [(0, 11, 12)],
        dtype=[
            ("bout_id", "<i8"),
            ("start_frame", "<i8"),
            ("end_frame", "<i8"),
        ],
    )
    binding_body = {
        "schema_id": "palette.selector_ineligible_swim_bout_binding.v1",
        "run_name": "bouts-a",
        "run_path": "analysis/swim_bout_runs/bouts-a",
        "lineage_hash": "1" * 64,
        "frame_axis_sha256": "2" * 64,
        "source_track_motion_manifest_sha256": "3" * 64,
        "source_track_motion_verification_digest": "4" * 64,
        "track_id": 0,
        "track_row_start": 0,
        "track_row_stop": 3,
        "default_candidate_id": 0,
        "default_signal_id": 0,
        "default_signal_level": "speed_smoothed",
    }
    binding = {**binding_body, "sha256": canonical_json_sha256(binding_body)}
    tables = SimpleNamespace(
        run_path="analysis/swim_bout_runs/bouts-a",
        run_attrs={
            "schema_id": "palette.swim_bout_run",
            "schema_version": 8,
            "layout": "compact_tabular_v2",
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": False,
            "source_track_kinematics_run": "motion-a",
            "fps": 30.0,
            "array_schema_manifest": {"fixture": "manifest"},
        },
        candidate=SimpleNamespace(
            track_id=0,
            candidate_id=0,
            candidate_name="default",
        ),
        signal=SimpleNamespace(
            signal_id=0,
            signal_name="default_detector",
            speed_level="speed_smoothed",
            role="detector_response",
            source_level="speed_smoothed",
        ),
        series={"frame_indices": frame_axis},
        bouts=events,
    )

    identity = bout_authority_identity_from_phase_c_tables(
        recording_id="recording-a",
        analysis_zarr=tmp_path / "recording-a.zarr",
        binding=binding,
        tables=tables,
        source_sample_rate_hz=30.0,
    )

    assert identity.record["motion_authority"]["scope"] == "provider"
    assert identity.record["selected_events"]["count"] == 1
    assert identity.record["selected_events"]["content_sha256"]


def test_table_spec_composition_rejects_collision_before_overwrite() -> None:
    names = tuple(CORE_TABLE_SPECS)
    midpoint = len(names) // 2
    first = {name: CORE_TABLE_SPECS[name] for name in names[:midpoint]}
    second = {name: CORE_TABLE_SPECS[name] for name in names[midpoint:]}

    composed = compose_disjoint_table_specs(("first", first), ("second", second))
    assert set(composed) == set(CORE_TABLE_SPECS)

    with pytest.raises(ValueError, match="collides before composition"):
        compose_disjoint_table_specs(
            ("core", CORE_TABLE_SPECS),
            ("duplicate", {names[0]: CORE_TABLE_SPECS[names[0]]}),
        )


class _FakeArray:
    def __init__(self, values: np.ndarray):
        self._values = values
        self.shape = values.shape
        self.dtype = values.dtype

    def __getitem__(self, selection: object) -> np.ndarray:
        return self._values[selection]


class _FakeGroup(dict[str, object]):
    def __init__(
        self, values: dict[str, object], *, attrs: dict[str, object] | None = None
    ):
        super().__init__(values)
        self.attrs = {} if attrs is None else attrs


def _body_publication(
    *,
    frames: np.ndarray | None = None,
    instance_keys: np.ndarray | None = None,
    axis_valid: np.ndarray | None = None,
) -> BoundSubjectShapeCoordinatePublication:
    frames = np.asarray(
        [11, 12, 13] if frames is None else frames,
        dtype=np.int64,
    )
    instance_keys = np.asarray(
        [101, 102, 103] if instance_keys is None else instance_keys,
        dtype=np.uint64,
    )
    axis_valid = np.asarray(
        np.ones(frames.size, dtype=bool) if axis_valid is None else axis_valid,
        dtype=bool,
    )
    origin = np.column_stack(
        (np.arange(frames.size, dtype=np.float32) + 2.0, np.ones(frames.size))
    ).astype(np.float32)
    forward = np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (frames.size, 1))
    left = np.tile(np.asarray([[0.0, -1.0]], dtype=np.float32), (frames.size, 1))
    origin[~axis_valid] = np.nan
    forward[~axis_valid] = np.nan
    left[~axis_valid] = np.nan
    body_group = _FakeGroup(
        {
            "origin_xy": _FakeArray(origin),
            "forward_axis_xy": _FakeArray(forward),
            "left_axis_xy": _FakeArray(left),
            "heading_deg": _FakeArray(
                np.where(axis_valid, 0.0, np.nan).astype(np.float32)
            ),
            "axis_valid": _FakeArray(axis_valid),
            "failure_reason_bytes": _FakeArray(
                np.zeros((frames.size, 64), dtype=np.uint8)
            ),
        }
    )
    run = _FakeGroup(
        {
            "instance_key": _FakeArray(instance_keys),
            "source_acquisition_frame_index": _FakeArray(frames),
            "body_frame": body_group,
        },
        attrs={
            "schema_id": "palette.subject_shape_coordinate_publication",
            "schema_version": 4,
            "palette_run_completion_status": "complete",
            "palette_run_completed_at_utc": "2026-09-05T00:00:00Z",
        },
    )
    acquisition_record = SimpleNamespace(
        recording_id="recording-a",
        camera_id="camera-a",
        source_total_frames=100,
        source_video_metadata={"fps": 30.0},
    )
    publication = object.__new__(BoundSubjectShapeCoordinatePublication)
    values = {
        "run_path": "analysis/subject_shape_runs/body-a",
        "row_identity": SimpleNamespace(
            leading_dimension=frames.size,
            record_sha256="1" * 64,
        ),
        "temporal_authority": SimpleNamespace(
            record=acquisition_record,
            acquisition_frame=SimpleNamespace(
                record=acquisition_record,
                record_sha256="8" * 64,
            ),
            record_sha256="2" * 64,
        ),
        "body_frame": SimpleNamespace(record_sha256="3" * 64),
        "heading_semantics": SimpleNamespace(record_sha256="4" * 64),
        "manifest": SimpleNamespace(record_sha256="5" * 64),
        "descriptors": {
            path: SimpleNamespace(
                descriptor=SimpleNamespace(
                    digest=(lambda value=value: value),
                )
            )
            for path, value in {
                "body_frame/origin_xy": "a" * 64,
                "body_frame/forward_axis_xy": "b" * 64,
                "body_frame/left_axis_xy": "c" * 64,
            }.items()
        },
        "selector_eligible": True,
        "_run": run,
    }
    for name, value in values.items():
        object.__setattr__(publication, name, value)
    return publication


def _bound_core_motion(
    tmp_path: Path,
    *,
    body_publication: BoundSubjectShapeCoordinatePublication | None = None,
) -> BoundCoreMotionAndBouts:
    frames = np.asarray([11, 12, 13], dtype=np.int64)
    positions = np.asarray([[1.0, 2.0], [2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    transition_valid = np.asarray([False, True, True], dtype=bool)
    filtered_speed = np.asarray([np.nan, 10.0, 12.0], dtype=np.float32)
    filtered_path = np.asarray([np.nan, 0.3, 0.4], dtype=np.float32)
    smoothed_speed = np.asarray([np.nan, 9.0, 11.0], dtype=np.float32)
    smoothed_path = np.asarray([np.nan, 0.25, 0.35], dtype=np.float32)
    heading = np.asarray([0.0, 10.0, 20.0], dtype=np.float32)
    smoothed_heading = np.asarray([0.0, 9.0, 19.0], dtype=np.float32)
    heading_usable = np.ones(3, dtype=bool)
    source_instances = np.zeros(
        frames.size,
        dtype=np.dtype([("valid", "?"), ("instance_key", "<u8")]),
    )
    source_instances["valid"] = True
    source_instances["instance_key"] = np.asarray([101, 102, 103], dtype=np.uint64)
    selected = {
        "source_acquisition_frame_index": {
            "shape": list(frames.shape),
            "dtype": frames.dtype.str,
        },
        "positions_mm": {
            "shape": list(positions.shape),
            "dtype": positions.dtype.str,
            "physical_authority_sha256": "9" * 64,
        },
        "transition_valid": {
            "shape": list(transition_valid.shape),
            "dtype": transition_valid.dtype.str,
        },
        "sample_valid": {
            "shape": [3],
            "dtype": np.dtype(bool).str,
        },
        "position_finite": {
            "shape": [3],
            "dtype": np.dtype(bool).str,
        },
        "source_instance_key": {
            "shape": list(source_instances.shape),
            "dtype": source_instances.dtype.str,
            "dtype_fields": [
                {"name": "valid", "dtype": np.dtype(bool).str, "offset": 0},
                {
                    "name": "instance_key",
                    "dtype": np.dtype("<u8").str,
                    "offset": 1,
                },
            ],
            "itemsize": source_instances.dtype.itemsize,
        },
        "movement/speed/filtered/mm": {
            "shape": list(filtered_speed.shape),
            "dtype": filtered_speed.dtype.str,
        },
        "movement/speed/filtered/frame_path_distance_mm": {
            "shape": list(filtered_path.shape),
            "dtype": filtered_path.dtype.str,
        },
        "movement/speed/smoothed/mm": {
            "shape": list(smoothed_speed.shape),
            "dtype": smoothed_speed.dtype.str,
        },
        "movement/speed/smoothed/frame_path_distance_mm": {
            "shape": list(smoothed_path.shape),
            "dtype": smoothed_path.dtype.str,
        },
        "heading_degrees": {
            "shape": list(heading.shape),
            "dtype": heading.dtype.str,
        },
        "smoothed_heading_degrees": {
            "shape": list(smoothed_heading.shape),
            "dtype": smoothed_heading.dtype.str,
        },
        "heading_usable": {
            "shape": list(heading_usable.shape),
            "dtype": heading_usable.dtype.str,
        },
    }
    capabilities = _capability_bindings(tmp_path)
    motion_source = _sealed(
        schema_id="fixture.kinematics_samples.source",
        schema_version=2,
        recording_id="recording-a",
        zarr_path=str((tmp_path / "recording-a.zarr").resolve()),
        run_path="analysis/track_kinematics_runs/offline/motion-a",
        run_name="motion-a",
        scope="offline",
        source_manifest_sha256="c" * 64,
        source_sample_rate_hz=30.0,
        tracks=[
            {
                "track_id": 7,
                "sample_count": 3,
                "selected_surfaces": selected,
            }
        ],
    )
    capabilities["kinematics_samples"] = {
        "profile_id": "core_motion_physical_v2",
        "source_binding": motion_source,
        "projection_contract": _sealed(
            schema_id="fixture.kinematics_samples.projection",
            schema_version=3,
        ),
        "join_authority_sha256": capabilities[CROSS_GRAIN_JOIN_AUTHORITY][
            "payload_sha256"
        ],
    }
    bout_source_binding = _sealed(
        schema_id="fixture.canonical_swim_bouts.source",
        schema_version=1,
        recording_id="recording-a",
        zarr_path=str((tmp_path / "recording-a.zarr").resolve()),
        run_path="analysis/swim_bout_runs/bouts-a",
        run_name="bouts-a",
        track_id=7,
    )
    capabilities["canonical_swim_bouts"]["source_binding"] = bout_source_binding
    if body_publication is None:
        body_publication = _body_publication()
    capabilities[SUBJECT_BODY_FRAME_CAPABILITY] = {
        "profile_id": SUBJECT_BODY_FRAME_SOURCE_PROFILE_ID,
        "source_binding": build_subject_body_frame_source_binding(body_publication),
        "projection_contract": _sealed(
            schema_id="fixture.subject_body_frame_samples.projection",
            schema_version=1,
        ),
        "join_authority_sha256": capabilities[CROSS_GRAIN_JOIN_AUTHORITY][
            "payload_sha256"
        ],
    }
    roster = build_core_authority_roster(
        recording_id="recording-a",
        analysis_zarr=tmp_path / "recording-a.zarr",
        execution_report_binding=_report_binding(tmp_path),
        capability_bindings=capabilities,
    )
    track_group = {
        "source_acquisition_frame_index": _FakeArray(frames),
        "positions_mm": _FakeArray(positions),
        "transition_valid": _FakeArray(transition_valid),
        "sample_valid": _FakeArray(np.ones(3, dtype=bool)),
        "position_finite": _FakeArray(np.ones(3, dtype=bool)),
        "source_instance_key": _FakeArray(source_instances),
        "movement": {
            "speed": {
                "filtered": {
                    "mm": _FakeArray(filtered_speed),
                    "frame_path_distance_mm": _FakeArray(filtered_path),
                },
                "smoothed": {
                    "mm": _FakeArray(smoothed_speed),
                    "frame_path_distance_mm": _FakeArray(smoothed_path),
                },
            }
        },
        "heading_degrees": _FakeArray(heading),
        "smoothed_heading_degrees": _FakeArray(smoothed_heading),
        "heading_usable": _FakeArray(heading_usable),
    }
    return BoundCoreMotionAndBouts(
        _verification_seal=roster_module._BOUND_CORE_MOTION_SEAL,
        roster=roster,
        root=None,
        track=SimpleNamespace(
            binding=motion_source,
            run_group={"tracks": {"id_7": track_group}},
        ),
        bouts=SimpleNamespace(
            bout_sources={7: SimpleNamespace(binding=bout_source_binding)}
        ),
        bout_identities={7: _bout_identity(tmp_path, track_id=7)},
    )


def test_core_motion_handle_is_resolver_minted_and_receipt_bound(
    tmp_path: Path,
) -> None:
    bound = _bound_core_motion(tmp_path)

    with pytest.raises(CoreMotionSourceHandleError, match="roster resolver"):
        CoreMotionTrackSourceHandle()

    handle = bind_core_motion_track_source_handle(
        bound,
        consumer_id="goodbatbadbat.chaser_extension_v1",
        required_capabilities=(
            CROSS_GRAIN_JOIN_AUTHORITY,
            "kinematics_samples",
            "canonical_swim_bouts",
        ),
        track_id=7,
    )

    assert require_core_motion_track_source_handle(handle) is handle
    assert handle.track_id == 7
    assert handle.core_authority_roster_sha256 == bound.roster_sha256
    assert np.array_equal(handle.frame_indices, np.asarray([11, 12, 13]))
    assert handle.positions_mm.flags.writeable is False
    assert handle.consumption_receipt["selected_track_id"] == 7
    dependency = core_motion_dependency_record(handle)
    assert dependency["core_authority_roster_sha256"] == bound.roster_sha256
    assert dependency["motion_run_path"] == handle.run_path
    assert dependency["swim_bout_run_path"] == "analysis/swim_bout_runs/bouts-a"
    assert dependency["track_id"] == 7
    assert (
        validate_core_motion_dependency_record(
            dependency,
            roster=bound.roster,
        )["record_sha256"]
        == dependency["record_sha256"]
    )
    with pytest.raises(KeyError, match="not selected"):
        handle.array("latest")


def test_core_motion_dependency_rejects_self_consistent_incomplete_receipt(
    tmp_path: Path,
) -> None:
    bound = _bound_core_motion(tmp_path)
    handle = bind_core_motion_track_source_handle(
        bound,
        consumer_id="goodbatbadbat.chaser_extension_v1",
        required_capabilities=(
            CROSS_GRAIN_JOIN_AUTHORITY,
            "kinematics_samples",
            "canonical_swim_bouts",
        ),
        track_id=7,
    )
    dependency = deepcopy(dict(core_motion_dependency_record(handle)))
    receipt = dependency["core_authority_consumption_receipt"]
    receipt["required_capabilities"].remove("canonical_swim_bouts")
    receipt["capability_binding_digests"].pop("canonical_swim_bouts")
    receipt_body = {
        key: value for key, value in receipt.items() if key != "record_sha256"
    }
    receipt["record_sha256"] = canonical_json_sha256(receipt_body)
    dependency_body = {
        key: value for key, value in dependency.items() if key != "record_sha256"
    }
    dependency["record_sha256"] = canonical_json_sha256(dependency_body)

    with pytest.raises(CoreMotionSourceHandleError, match="stale or incomplete"):
        validate_core_motion_dependency_record(dependency)


def test_core_motion_handle_rejects_implicit_or_mismatched_track(
    tmp_path: Path,
) -> None:
    bound = _bound_core_motion(tmp_path)

    for track_id in (-1, 0):
        with pytest.raises(CoreMotionSourceHandleError, match="track"):
            bind_core_motion_track_source_handle(
                bound,
                consumer_id="goodbatbadbat.chaser_extension_v1",
                required_capabilities=(
                    CROSS_GRAIN_JOIN_AUTHORITY,
                    "kinematics_samples",
                ),
                track_id=track_id,
            )


def _core_chaser_handles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    body_publication: BoundSubjectShapeCoordinatePublication | None = None,
) -> tuple[CoreMotionTrackSourceHandle, CoreSubjectBodyFrameSourceHandle]:
    publication = body_publication or _body_publication()
    bound = _bound_core_motion(tmp_path, body_publication=publication)
    monkeypatch.setattr(
        roster_module,
        "load_persisted_subject_shape_coordinate_publication",
        lambda *args, **kwargs: publication,
    )
    required = (
        CROSS_GRAIN_JOIN_AUTHORITY,
        "kinematics_samples",
        SUBJECT_BODY_FRAME_CAPABILITY,
    )
    motion = bind_core_motion_track_source_handle(
        bound,
        consumer_id=CORE_CHASER_RELATIVE_CONSUMER_ID,
        required_capabilities=required,
        track_id=7,
    )
    body = bind_core_subject_body_frame_source_handle(
        bound,
        consumer_id=CORE_CHASER_RELATIVE_CONSUMER_ID,
        required_capabilities=required,
        track_id=7,
    )
    return motion, body


def _chaser_source(tmp_path: Path) -> ChaserRelativeFrameSourceHandle:
    n_frames = 3
    n_chasers = 2
    frames = np.repeat(np.asarray([11, 12, 13], dtype=np.int64), n_chasers)
    controller = {
        "schema_id": "fixture.controller_state",
        "schema_version": 1,
        "recording_id": "recording-a",
        "session_timestamp_authority": {
            "recording_timing_authority_sha256": "6" * 64,
        },
    }
    chaser_authority = {
        "recording_id": "recording-a",
        "source_authority_id": "analysis/chaser_input/source-a",
        "source_digest": "a" * 64,
        "provider_id": "chaser-provider-a",
        "provider_digest": "b" * 64,
        "coordinate_authority_id": "/metadata/acquisition_camera_frame",
        "scale_authority_id": "/analysis/calibration/physical",
        "timing_authority_id": "analysis/chaser/timestamp_ns",
        "row_axis_authority_id": "analysis/chaser/frame_axis",
        "row_axis_authority_digest": "c" * 64,
    }
    manifest = {
        "coordinate_policy": {
            "policy_id": "source_camera_y_down_v1",
            "coordinate_authority_id": "/metadata/acquisition_camera_frame",
            "coordinate_frame": "source_camera_continuous_pixel_xy",
            "origin": "top_left",
            "x_axis_direction": "right",
            "y_axis_direction": "down",
        },
        "scale_policy": {
            "policy_id": "source_camera_scale_v1",
            "scale_authority_id": "/analysis/calibration/physical",
            "scale_digest": "9" * 64,
            "pixels_per_unit": 2.0,
            "unit": "mm",
        },
        "timing_policy": {
            "policy_id": "acquisition_camera_timing_v1",
            "timing_authority_id": "analysis/chaser/timestamp_ns",
            "timing_digest": "6" * 64,
            "frame_key_name": "acquisition_frame_id",
            "track_sample_key_name": "track_sample_id",
            "timestamp_field": "timestamp_ns",
        },
    }
    arrays = {
        "acquisition_frame_id": frames,
        "timestamp_ns": np.repeat(
            np.asarray([100, 200, 300], dtype=np.int64), n_chasers
        ),
        "timestamp_valid": np.ones(6, dtype=bool),
        "chaser_position_xy_px": np.asarray(
            [
                [4.0, 6.0],
                [6.0, 8.0],
                [6.0, 8.0],
                [8.0, 10.0],
                [10.0, 12.0],
                [12.0, 14.0],
            ],
            dtype=np.float32,
        ),
        "chaser_position_valid": np.ones(6, dtype=bool),
        "selection_member": np.ones(6, dtype=bool),
        "chaser_occurrence_member": np.ones(6, dtype=bool),
        "chaser_identity_code": np.tile(np.asarray([1, 2], dtype=np.uint16), 3),
        "chaser_behavior_role_code": np.tile(np.asarray([1, 2], dtype=np.uint8), 3),
        "chaser_source_row_id": np.arange(6, dtype=np.int64),
        "trial_id": np.ones(6, dtype=np.int64),
        "trial_valid": np.ones(6, dtype=bool),
        "active_state_code": np.ones(6, dtype=np.uint8),
        "active_state_valid": np.ones(6, dtype=bool),
    }

    def envelope(record: dict[str, object]) -> dict[str, object]:
        return {"record": record, "sha256": canonical_json_sha256(record)}

    context = {
        "fish_identity": "fish-a",
        "subject_identity": envelope(
            {
                "schema_id": "fixture.subject_identity",
                "schema_version": 1,
                "recording_id": "recording-a",
                "subject_id": "fish-a",
            }
        ),
        "temporal_selection": envelope(
            {
                "schema_id": "fixture.temporal_selection",
                "schema_version": 1,
                "recording_id": "recording-a",
                "selection_id": "source-selection-a",
            }
        ),
        "chaser_occurrence": envelope(
            {
                "schema_id": "fixture.chaser_occurrence",
                "schema_version": 1,
                "recording_id": "recording-a",
                "occurrence_policy_id": "logged-occurrence-a",
            }
        ),
        "acquisition_projection": envelope(
            {
                "schema_id": "fixture.acquisition_projection",
                "schema_version": 1,
                "recording_id": "recording-a",
                "policy_id": "direct-acquisition-projection-a",
            }
        ),
        "acquisition_projection_publication": None,
        "controller_state": envelope(controller),
        "analysis_profile": envelope(
            {
                "schema_id": "fixture.chaser_profile",
                "schema_version": 1,
                "recording_id": "recording-a",
                "profile_id": "source-chaser-profile-a",
            }
        ),
        "arena_geometry": None,
        "arena_to_source_camera_transform": None,
    }
    value = object.__new__(ChaserRelativeFrameSourceHandle)
    for name, item in {
        "analysis_zarr_path": (tmp_path / "recording-a.zarr").resolve(),
        "run_path": "analysis/chaser_relative_frame_runs/chaser-a",
        "run_name": "chaser-a",
        "recording_id": "recording-a",
        "selector_eligible": False,
        "n_frames": n_frames,
        "n_chasers": n_chasers,
        "n_rows": n_frames * n_chasers,
        "run_manifest": manifest,
        "source_authorities": {"chaser_position": chaser_authority},
        "context": context,
        "identity_registries": {
            "fish": {"1": "fish-a"},
            "chaser": {"1": "chaser-a", "2": "chaser-b"},
            "behavior_role": {"1": "good", "2": "bad"},
            "active_state": {"0": "inactive", "1": "active"},
        },
        "base_arrays": arrays,
        "body_arrays": None,
        "verification_digest": "d" * 64,
    }.items():
        object.__setattr__(value, name, item)
    return value


def _timing_authority() -> SimpleNamespace:
    return SimpleNamespace(
        sha256="6" * 64,
        recording_id="recording-a",
        camera_id="camera-a",
        frame_count=100,
        source_video_metadata_sha256="7" * 64,
    )


def test_core_chaser_adapter_reuses_existing_relative_frame_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fisheye.analysis_workflows import core_chaser_relative_frame_adapter as module

    monkeypatch.setattr(
        ChaserRelativeFrameSourceHandle, "assert_current", lambda self: None
    )
    monkeypatch.setattr(
        module,
        "load_provider_recording_timing_authority",
        lambda *args, **kwargs: _timing_authority(),
    )

    motion, body = _core_chaser_handles(tmp_path, monkeypatch)
    adapted = prepare_core_chaser_relative_frame(
        motion,
        body,
        _chaser_source(tmp_path),
    )
    prepared = adapted.prepared

    assert prepared.dimensions.n_rows == 6
    assert prepared.body_arrays is not None
    assert np.array_equal(
        prepared.body_arrays["body_source_row_id"],
        np.repeat(np.asarray([0, 1, 2], dtype=np.int64), 2),
    )
    assert np.array_equal(
        prepared.base_arrays["track_sample_id"],
        np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64),
    )
    assert np.array_equal(
        prepared.base_arrays["relative_physical_valid"],
        np.ones(6, dtype=bool),
    )
    assert np.allclose(
        prepared.base_arrays["relative_distance_physical"],
        np.asarray([2**0.5, 8**0.5] * 3, dtype=np.float32),
    )
    assert not any(
        token in name
        for name in prepared.base_arrays
        for token in ("speed", "acceleration", "cumulative")
    )
    core_binding = prepared.manifest["context"]["core_authority"]["record"]
    assert core_binding["core_motion_facts_repeated"] is False
    assert core_binding["core_subject_body_frame"]["run_path"] == (
        "analysis/subject_shape_runs/body-a"
    )
    assert core_binding["chaser_source"]["fish_position_authority"] == (
        "not_used_core_roster_selected_instead"
    )
    assert core_binding["chaser_source"]["body_frame_authority"] == (
        "not_used_core_roster_selected_instead"
    )
    assert (
        prepared.manifest["source_authorities"]["fish_position"]["source_authority_id"]
        == "analysis/track_kinematics_runs/offline/motion-a"
    )
    receipt = validate_prepared_chaser_relative_frame(prepared)
    assert receipt["payload_digest"] == prepared.payload_digest
    assert adapted.to_json()["publication_surface"] == (
        "analysis/chaser_relative_frame_runs"
    )


def test_core_chaser_adapter_rejects_coordinate_authority_conflict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ChaserRelativeFrameSourceHandle, "assert_current", lambda self: None
    )
    source = _chaser_source(tmp_path)
    source.run_manifest["coordinate_policy"][
        "coordinate_authority_id"
    ] = "/metadata/another_camera_frame"

    with pytest.raises(
        CoreChaserRelativeFrameAdapterError,
        match="source-camera authority",
    ):
        motion, body = _core_chaser_handles(tmp_path, monkeypatch)
        prepare_core_chaser_relative_frame(motion, body, source)


def test_core_chaser_adapter_rejects_a_missing_core_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fisheye.analysis_workflows import core_chaser_relative_frame_adapter as module

    monkeypatch.setattr(
        ChaserRelativeFrameSourceHandle, "assert_current", lambda self: None
    )
    monkeypatch.setattr(
        module,
        "load_provider_recording_timing_authority",
        lambda *args, **kwargs: _timing_authority(),
    )
    source = _chaser_source(tmp_path)
    source.base_arrays["acquisition_frame_id"][:] = np.repeat(
        np.asarray([11, 12, 14], dtype=np.int64), 2
    )

    with pytest.raises(
        CoreChaserRelativeFrameAdapterError,
        match="Every chaser frame",
    ):
        motion, body = _core_chaser_handles(tmp_path, monkeypatch)
        prepare_core_chaser_relative_frame(motion, body, source)


def test_core_subject_body_handle_is_resolver_minted_and_receipt_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(
        CoreSubjectBodyFrameSourceHandleError,
        match="minted by the resolver",
    ):
        CoreSubjectBodyFrameSourceHandle()

    motion, body = _core_chaser_handles(tmp_path, monkeypatch)

    assert require_core_subject_body_frame_source_handle(body) is body
    assert body.consumption_receipt == motion.consumption_receipt
    assert np.array_equal(body.frame_indices, np.asarray([11, 12, 13]))
    assert np.array_equal(body.instance_keys, np.asarray([101, 102, 103]))
    assert body.origin_xy.flags.writeable is False
    assert np.array_equal(
        motion.array("source_instance_key")["instance_key"],
        np.asarray([101, 102, 103], dtype=np.uint64),
    )


def test_core_chaser_adapter_keeps_missing_body_observation_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fisheye.analysis_workflows import core_chaser_relative_frame_adapter as module

    monkeypatch.setattr(
        ChaserRelativeFrameSourceHandle, "assert_current", lambda self: None
    )
    monkeypatch.setattr(
        module,
        "load_provider_recording_timing_authority",
        lambda *args, **kwargs: _timing_authority(),
    )
    publication = _body_publication(
        frames=np.asarray([11, 13]),
        instance_keys=np.asarray([101, 103]),
    )
    motion, body = _core_chaser_handles(
        tmp_path,
        monkeypatch,
        body_publication=publication,
    )

    prepared = prepare_core_chaser_relative_frame(
        motion,
        body,
        _chaser_source(tmp_path),
    ).prepared

    assert prepared.body_arrays is not None
    assert np.array_equal(
        prepared.body_arrays["body_source_row_id"],
        np.asarray([0, 0, -1, -1, 1, 1], dtype=np.int64),
    )
    assert np.array_equal(
        prepared.body_arrays["body_axes_valid"],
        np.asarray([True, True, False, False, True, True]),
    )
    projection = prepared.manifest["context"]["body_frame_projection"]["record"]
    assert projection["missing_source_row_count"] == 1
    assert projection["interpolation"] == "prohibited"


def test_core_chaser_adapter_rejects_duplicate_body_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fisheye.analysis_workflows import core_chaser_relative_frame_adapter as module

    monkeypatch.setattr(
        ChaserRelativeFrameSourceHandle, "assert_current", lambda self: None
    )
    monkeypatch.setattr(
        module,
        "load_provider_recording_timing_authority",
        lambda *args, **kwargs: _timing_authority(),
    )
    publication = _body_publication(
        frames=np.asarray([11, 11, 13]),
        instance_keys=np.asarray([101, 101, 103]),
    )
    motion, body = _core_chaser_handles(
        tmp_path,
        monkeypatch,
        body_publication=publication,
    )

    with pytest.raises(
        CoreChaserRelativeFrameAdapterError,
        match="duplicate frame/instance keys",
    ):
        prepare_core_chaser_relative_frame(
            motion,
            body,
            _chaser_source(tmp_path),
        )


def test_core_chaser_adapter_rejects_separate_body_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ChaserRelativeFrameSourceHandle, "assert_current", lambda self: None
    )
    motion, body = _core_chaser_handles(tmp_path, monkeypatch)
    separate = bind_core_subject_body_frame_source_handle(
        body._bound,
        consumer_id="another.paradigm.consumer.v1",
        required_capabilities=(
            CROSS_GRAIN_JOIN_AUTHORITY,
            "kinematics_samples",
            SUBJECT_BODY_FRAME_CAPABILITY,
        ),
        track_id=7,
    )

    with pytest.raises(
        CoreChaserRelativeFrameAdapterError,
        match="shared admission receipt",
    ):
        prepare_core_chaser_relative_frame(
            motion,
            separate,
            _chaser_source(tmp_path),
        )
