from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from fisheye.analysis_workflows.chaser_relative_frame_source_handle import (
    ChaserRelativeFrameSourceHandle,
)
from fisheye.analysis_workflows.core_paradigm_authority import (
    CoreParadigmAuthorityError,
    core_paradigm_dependency_from_relative_frame,
    validate_core_paradigm_dependency,
    validate_core_paradigm_source_dependency,
)
from fisheye.analysis_workflows.chaser_body_alignment_by_distance_successor import (
    prepare_chaser_body_alignment_by_distance_successor,
)
from fisheye.analysis_workflows.chaser_near_field_visit_successor import (
    prepare_chaser_near_field_visit_successor,
)
from fisheye.analysis_workflows.chaser_radial_near_field_successor import (
    prepare_chaser_radial_near_field_successor,
)
from fisheye.analysis_workflows.chaser_spatial_occupancy_successor import (
    prepare_chaser_spatial_occupancy_successor,
)
from fisheye.analysis_workflows.controller_trial_successor import (
    prepare_controller_trial_successor,
)
from fisheye.analysis_workflows.gaze_tracking_successor import (
    prepare_gaze_tracking_successor,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from tests.unit.fisheye.test_chaser_body_alignment_by_distance_successor import (
    _inputs as _body_alignment_inputs,
)
from tests.unit.fisheye.test_chaser_near_field_visit_successor import (
    _radial_inputs as _visit_radial_inputs,
    _visit_inputs,
)
from tests.unit.fisheye.test_chaser_radial_near_field_successor import (
    _inputs as _radial_inputs,
)
from tests.unit.fisheye.test_chaser_spatial_occupancy_successor import (
    _inputs as _spatial_inputs,
)
from tests.unit.fisheye.test_controller_trial_successor import (
    _source as _controller_input,
)
from tests.unit.fisheye.test_gaze_tracking_successor import _source as _gaze_input


def _seal(body: dict[str, object]) -> dict[str, object]:
    return {**body, "record_sha256": canonical_json_sha256(body)}


def _dependency(
    *,
    recording_id: str,
    run_path: str,
    manifest_sha256: str,
    roster_sha256: str = "1" * 64,
) -> dict[str, object]:
    body = {
        "schema_id": "palette.core_behavior.paradigm_relative_frame_dependency",
        "schema_version": 1,
        "recording_id": recording_id,
        "core_authority_roster_sha256": roster_sha256,
        "core_authority_consumption_receipt_sha256": "2" * 64,
        "selected_track_id": 7,
        "source_relative_frame_run_path": run_path,
        "source_relative_frame_manifest_sha256": manifest_sha256,
        "source_core_authority_binding_sha256": "3" * 64,
        "core_motion_source_binding_sha256": "4" * 64,
        "core_subject_body_frame_source_binding_sha256": "5" * 64,
        "fallback": "prohibited",
    }
    return _seal(body)


def _core_context() -> dict[str, object]:
    roster_sha = "1" * 64
    receipt_body = {
        "schema_id": "palette.core_behavior.authority_consumption_receipt",
        "schema_version": 1,
        "consumer_id": "palette.chaser.core_relative_frame.v1",
        "recording_id": "recording-1",
        "analysis_zarr": "/archive/analysis.zarr",
        "core_authority_roster_sha256": roster_sha,
        "required_capabilities": [
            "cross_grain_join_authority",
            "kinematics_samples",
            "subject_body_frame_samples",
        ],
        "capability_binding_digests": {
            "cross_grain_join_authority": {
                "profile_id": "cross_grain_join_authority_v1",
                "source_binding_sha256": "2" * 64,
                "projection_contract_sha256": None,
                "join_authority_sha256": "2" * 64,
            },
            "kinematics_samples": {
                "profile_id": "core_motion_physical_v2",
                "source_binding_sha256": "b" * 64,
                "projection_contract_sha256": "5" * 64,
                "join_authority_sha256": "2" * 64,
            },
            "subject_body_frame_samples": {
                "profile_id": "subject_body_frame_samples_v1",
                "source_binding_sha256": "e" * 64,
                "projection_contract_sha256": "8" * 64,
                "join_authority_sha256": "2" * 64,
            },
        },
        "selected_track_id": 7,
    }
    receipt = _seal(receipt_body)
    binding = {
        "schema_id": "palette.chaser_relative_frame.core_authority_binding",
        "schema_version": 1,
        "recording_id": "recording-1",
        "core_authority_roster_sha256": roster_sha,
        "core_authority_consumption_receipt": receipt,
        "core_motion": {
            "run_path": "analysis/track_kinematics_runs/provider/core",
            "source_manifest_sha256": "a" * 64,
            "source_binding_sha256": "b" * 64,
            "track_id": 7,
            "row_axis_sha256": "c" * 64,
        },
        "core_subject_body_frame": {
            "run_path": "analysis/subject_shape_runs/core",
            "publication_manifest_sha256": "d" * 64,
            "source_binding_sha256": "e" * 64,
            "row_identity_sha256": "f" * 64,
            "body_frame_record_sha256": "0" * 64,
            "projection_record_sha256": "1" * 64,
        },
        "chaser_source": {
            "run_path": "analysis/chaser_relative_frame_runs/source",
            "manifest_sha256": "2" * 64,
            "verification_digest": "3" * 64,
            "consumed_authority": "chaser_position",
            "fish_position_authority": "not_used_core_roster_selected_instead",
            "body_frame_authority": "not_used_core_roster_selected_instead",
        },
        "fish_pixel_projection": {
            "source": "core_positions_mm",
            "formula": "positions_mm * pixels_per_mm",
            "physical_authority_sha256": "4" * 64,
        },
        "core_motion_facts_repeated": False,
        "fallback": "prohibited",
    }
    profile = {
        "schema_id": "palette.chaser_relative_frame.core_analysis_profile",
        "schema_version": 1,
        "recording_id": "recording-1",
        "profile_id": "core_roster_chaser_relative_frame_v1",
        "core_authority_roster_sha256": roster_sha,
        "source_chaser_profile_sha256": "5" * 64,
        "body_frame": "core_roster_selected_subject_body_frame",
    }
    return {
        "core_authority": {
            "record": binding,
            "sha256": canonical_json_sha256(binding),
        },
        "analysis_profile": {
            "record": profile,
            "sha256": canonical_json_sha256(profile),
        },
    }


def _handle(*, context: dict[str, object]) -> ChaserRelativeFrameSourceHandle:
    handle = object.__new__(ChaserRelativeFrameSourceHandle)
    object.__setattr__(handle, "analysis_zarr_path", Path("/archive/analysis.zarr"))
    object.__setattr__(
        handle,
        "run_path",
        "analysis/chaser_relative_frame_runs/core-relative",
    )
    object.__setattr__(handle, "run_name", "core-relative")
    object.__setattr__(handle, "recording_id", "recording-1")
    object.__setattr__(handle, "run_manifest", {"payload_digest": "6" * 64})
    object.__setattr__(handle, "context", context)
    return handle


def test_projects_one_exact_core_roster_dependency(monkeypatch) -> None:
    monkeypatch.setattr(
        ChaserRelativeFrameSourceHandle,
        "assert_current",
        lambda self: None,
    )
    handle = _handle(context=_core_context())

    dependency = core_paradigm_dependency_from_relative_frame(handle, required=True)

    assert dependency is not None
    assert dependency["recording_id"] == "recording-1"
    assert dependency["core_authority_roster_sha256"] == "1" * 64
    assert dependency["selected_track_id"] == 7
    assert dependency["source_relative_frame_run_path"] == handle.run_path
    assert validate_core_paradigm_dependency(dependency) == dependency
    assert (
        validate_core_paradigm_source_dependency(
            dependency,
            recording_id=handle.recording_id,
            source_relative_frame_run_path=handle.run_path,
            source_relative_frame_manifest_sha256=handle.manifest_sha256,
        )
        == dependency
    )


def test_dependency_and_missing_core_authority_fail_closed(monkeypatch) -> None:
    monkeypatch.setattr(
        ChaserRelativeFrameSourceHandle,
        "assert_current",
        lambda self: None,
    )
    handle = _handle(context=_core_context())
    dependency = dict(
        core_paradigm_dependency_from_relative_frame(handle, required=True) or {}
    )
    dependency["core_authority_roster_sha256"] = "f" * 64
    with pytest.raises(CoreParadigmAuthorityError, match="digest is stale"):
        validate_core_paradigm_dependency(dependency)

    legacy = _handle(context={})
    assert core_paradigm_dependency_from_relative_frame(legacy) is None
    with pytest.raises(CoreParadigmAuthorityError, match="no selected core-authority"):
        core_paradigm_dependency_from_relative_frame(legacy, required=True)


def test_projection_rejects_an_incomplete_nested_consumption_receipt(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        ChaserRelativeFrameSourceHandle,
        "assert_current",
        lambda self: None,
    )
    context = _core_context()
    envelope = context["core_authority"]
    assert isinstance(envelope, dict)
    binding = dict(envelope["record"])
    receipt = dict(binding["core_authority_consumption_receipt"])
    receipt.pop("capability_binding_digests")
    receipt_body = {
        key: value for key, value in receipt.items() if key != "record_sha256"
    }
    receipt["record_sha256"] = canonical_json_sha256(receipt_body)
    binding["core_authority_consumption_receipt"] = receipt
    envelope["record"] = binding
    envelope["sha256"] = canonical_json_sha256(binding)

    with pytest.raises(CoreParadigmAuthorityError, match="incomplete or stale"):
        core_paradigm_dependency_from_relative_frame(_handle(context=context))


def test_all_direct_paradigm_successors_retain_the_selected_core_roster() -> None:
    controller_source = _controller_input()
    controller_dependency = _dependency(
        recording_id=controller_source.recording_id,
        run_path=controller_source.source_run_path,
        manifest_sha256=controller_source.source_manifest_sha256,
    )
    controller = prepare_controller_trial_successor(
        replace(
            controller_source,
            core_authority_dependency=controller_dependency,
        )
    )

    radial_source = _radial_inputs()
    radial_dependency = _dependency(
        recording_id=radial_source.recording_id,
        run_path=radial_source.source_relative_frame_run_path,
        manifest_sha256=radial_source.source_relative_frame_manifest_sha256,
    )
    radial = prepare_chaser_radial_near_field_successor(
        replace(radial_source, core_authority_dependency=radial_dependency)
    )

    visit_radial_source = replace(
        _visit_radial_inputs(),
        core_authority_dependency=_dependency(
            recording_id="recording",
            run_path="analysis/chaser_relative_frame_runs/exact",
            manifest_sha256="a" * 64,
        ),
    )
    visit_source = _visit_inputs(visit_radial_source)
    visit_source = replace(
        visit_source,
        core_authority_dependency=visit_radial_source.core_authority_dependency,
    )
    visits = prepare_chaser_near_field_visit_successor(visit_source)

    alignment_source = _body_alignment_inputs()
    alignment_dependency = _dependency(
        recording_id=alignment_source.recording_id,
        run_path=alignment_source.relative_frame_run_path,
        manifest_sha256=alignment_source.relative_frame_manifest_sha256,
    )
    alignment = prepare_chaser_body_alignment_by_distance_successor(
        replace(
            alignment_source,
            core_authority_dependency=alignment_dependency,
        )
    )

    gaze_source = _gaze_input()
    gaze_dependency = _dependency(
        recording_id=gaze_source.recording_id,
        run_path=gaze_source.source_relative_frame_run_path,
        manifest_sha256=gaze_source.source_relative_frame_manifest_sha256,
    )
    gaze = prepare_gaze_tracking_successor(
        replace(gaze_source, core_authority_dependency=gaze_dependency)
    )

    assert controller.manifest["core_authority"] == controller_dependency
    assert radial.manifest["core_authority"] == radial_dependency
    assert visits.manifest["core_authority"] == (
        visit_radial_source.core_authority_dependency
    )
    assert alignment.manifest["core_authority"] == alignment_dependency
    assert gaze.manifest["core_authority"] == gaze_dependency


def test_legacy_successors_do_not_mint_null_core_authority_records() -> None:
    prepared = (
        prepare_controller_trial_successor(_controller_input()),
        prepare_chaser_radial_near_field_successor(_radial_inputs()),
        prepare_chaser_near_field_visit_successor(
            _visit_inputs(_visit_radial_inputs())
        ),
        prepare_chaser_body_alignment_by_distance_successor(_body_alignment_inputs()),
        prepare_gaze_tracking_successor(_gaze_input()),
        prepare_chaser_spatial_occupancy_successor(_spatial_inputs()),
    )

    assert all("core_authority" not in result.manifest for result in prepared)


def test_paired_occupancy_requires_one_common_core_roster() -> None:
    source = _spatial_inputs()
    providers = tuple(
        replace(
            provider,
            core_authority_dependency=_dependency(
                recording_id=source.recording_id,
                run_path=provider.relative_frame_run_path,
                manifest_sha256=provider.relative_frame_manifest_sha256,
            ),
        )
        for provider in source.providers
    )

    prepared = prepare_chaser_spatial_occupancy_successor(
        replace(source, providers=providers)
    )

    assert (
        prepared.manifest["core_authority"]["core_authority_roster_sha256"] == "1" * 64
    )
    assert len(prepared.manifest["core_authority"]["provider_dependencies"]) == 2

    mismatched = replace(
        providers[1],
        core_authority_dependency=_dependency(
            recording_id=source.recording_id,
            run_path=providers[1].relative_frame_run_path,
            manifest_sha256=providers[1].relative_frame_manifest_sha256,
            roster_sha256="f" * 64,
        ),
    )
    with pytest.raises(
        ValueError,
        match="bind different core authorities",
    ):
        prepare_chaser_spatial_occupancy_successor(
            replace(source, providers=(providers[0], mismatched))
        )
