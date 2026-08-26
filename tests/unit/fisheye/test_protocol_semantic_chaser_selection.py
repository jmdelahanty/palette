from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.analysis.import_stimulus_to_zarr import (
    _bind_protocol_semantic_steps,
    _materialize_protocol_semantic_snapshot,
)
from fisheye.analysis.stimulus_epoch_runs import (
    StimulusEpochResult,
    StimulusEpochWindow,
    write_stimulus_epoch_run,
)
from fisheye.analysis_workflows.chaser_profile_applicability import (
    CapabilityState,
    ModuleApplicabilityState,
    ProfileReadiness,
    plan_chaser_profile_applicability,
)
from fisheye.analysis_workflows.composable_epoch_selection_adapter import (
    EpochRoleBinding,
)
from fisheye.analysis_workflows.materializers.stimulus_epochs import (
    materialize_stimulus_epoch_candidate,
)
from fisheye.analysis_workflows.protocol_semantic_chaser_selection import (
    CHASER_WINDOW_ROLES,
    PROTOCOL_SEMANTIC_CHASER_WINDOWS_CAPABILITY_ID,
    PROTOCOL_SEMANTIC_SELECTION_PRODUCTION_CAPABILITY_ID,
    STANDALONE_SOLID_BLACK_BASELINE_CAPABILITY_ID,
    STANDALONE_SOLID_BLACK_ROLE,
    STEP_END_EXCLUSIVE,
    STEP_END_INCLUSIVE,
    STEP_END_PENDING,
    ProtocolSemanticChaserSelectionError,
    ProtocolSemanticTimelineEvidence,
    _project_execution_interval_to_acquisition_bounds,
    compile_protocol_semantic_chaser_selections,
    load_protocol_semantic_selection_evidence,
    load_protocol_semantic_timeline_evidence,
)
from fisheye.shared.protocol_execution_contract import (
    ProtocolStimulusFrameInterval,
)
from fisheye.analysis_workflows.resolved_epoch_selection import (
    resolve_exact_stimulus_epoch_selection,
)
from fisheye.shared.zarr_helpers import (
    consolidate_metadata_capture_expected_warnings,
)
from tests.unit.fisheye.test_protocol_semantic_contract import _snapshot
from tests.unit.fisheye.test_provider_recording_timing_authority import (
    _install_clock_authority,
)


@dataclass(frozen=True)
class _Module:
    module_id: str
    requirement_class: str
    required_capabilities: tuple[str, ...]
    depends_on: tuple[str, ...] = ()


def _evidence(archive: Path, selection):
    return load_protocol_semantic_timeline_evidence(archive, selection)


def _chaser_bindings(selection, *, offset: int = 0):
    return {
        role: EpochRoleBinding.by_window_id(
            selection.intervals[index + offset].window_id
        )
        for index, role in enumerate(CHASER_WINDOW_ROLES)
    }


def _semantic_selection(
    tmp_path: Path,
    *,
    modes: tuple[str, ...] = ("CHASER",),
    step_bounds: tuple[tuple[int, int], ...] = ((0, 29),),
    include_standalone_window: bool = False,
    step_end_interval_semantics: str = STEP_END_PENDING,
):
    if modes == ("CHASER",):
        specs = [
            ("pre_event", 0, 9),
            ("training_event", 10, 19),
            ("post_event", 20, 29),
        ]
        total_frames = 30
    else:
        assert modes == ("SOLID_BLACK", "CHASER")
        specs = [
            ("pre_event", 10, 19),
            ("training_event", 20, 29),
            ("post_event", 30, 39),
        ]
        if include_standalone_window:
            specs.insert(0, ("standalone_solid_black", 0, 9))
        total_frames = 40

    stem = (
        "_".join(modes).lower()
        + f"_{step_bounds!r}_{step_end_interval_semantics}"
        + ("_baseline" if include_standalone_window else "")
    )
    archive = tmp_path / f"{stem}.zarr"
    root = zarr.open_group(str(archive), mode="w", zarr_format=3)
    root.attrs.update(
        {
            "recording_id": "recording_1",
            "fps": 10.0,
            "total_frames": total_frames,
        }
    )
    stimulus = (
        root.require_group("analysis")
        .require_group("stimulus_runs")
        .create_group("stimulus_1")
    )
    stimulus.attrs.update(
        {
            "schema_id": "palette.stimulus.import.v1",
            "schema_version": 1,
            "run_name": "stimulus_1",
        }
    )
    events = stimulus.create_group("events")
    events.create_array(
        "camera_frame_id",
        data=np.arange(0, total_frames + 1, 10, dtype=np.int64),
    )

    snapshot = _snapshot(modes)
    steps = stimulus.create_group("steps")
    for identity, (start, end) in zip(
        snapshot.steps,
        step_bounds,
        strict=True,
    ):
        step = steps.create_group(f"step_{identity.step_index}")
        step.attrs.update(
            {
                "step_index": identity.step_index,
                "stimulus_mode_id": identity.stimulus_mode_id,
                "stimulus_mode": identity.stimulus_mode,
                "duration_s": identity.duration_s,
                "start_camera_frame": start,
                "end_camera_frame": end,
            }
        )
    _bind_protocol_semantic_steps(stimulus, snapshot)
    _materialize_protocol_semantic_snapshot(stimulus, snapshot)

    windows = tuple(
        StimulusEpochWindow(
            window_id=index,
            label=label,
            start_frame=start,
            end_frame=end,
            start_time_s=start / 10.0,
            end_time_s=(end + 1) / 10.0,
            duration_s=(end - start + 1) / 10.0,
            source_start_event_name=f"{label.upper()}_START",
            source_end_event_name=f"{label.upper()}_END",
            source_start_event_frame=start,
            source_end_event_frame=end + 1,
            source_policy="inclusive_start_exclusive_end_event_boundary",
        )
        for index, (label, start, end) in enumerate(specs)
    )
    write_stimulus_epoch_run(
        archive,
        StimulusEpochResult(
            zarr_path=str(archive),
            recording_id="recording_1",
            run_name="source",
            stimulus_run_name="stimulus_1",
            stimulus_path="analysis/stimulus_runs/stimulus_1",
            fps=10.0,
            total_frames=total_frames,
            windows=windows,
            protocol_profile_id="semantic_test_profile",
            protocol_profile_version=1,
            protocol_profile_sha256="a" * 64,
            protocol_profile_source="semantic_test_profile.yaml",
            source_adapter_id="semantic_test_adapter",
            source_adapter_version=1,
            role_resolver_id="semantic_test_roles",
            role_resolver_version=1,
            window_policy_id="semantic_test_windows",
            window_policy_version=1,
        ),
    )
    reopened = zarr.open_group(str(archive), mode="a", use_consolidated=False)
    reopened["analysis/stimulus_epoch_runs/source"].attrs[
        "stage_selector_eligible"
    ] = True
    result = materialize_stimulus_epoch_candidate(
        archive,
        source_run="source",
        run_name="candidate",
        scratch_root=tmp_path / f"scratch-{stem}",
        copy_backend="python",
        apply=True,
    )
    assert result["status"] == "complete"
    _install_clock_authority(
        archive,
        tmp_path,
        frame_count=total_frames,
        fps=10.0,
        recording_id="recording_1",
    )
    selection = resolve_exact_stimulus_epoch_selection(
        archive,
        run_name="candidate",
    )
    evidence = load_protocol_semantic_selection_evidence(
        archive,
        selection,
        step_end_interval_semantics=step_end_interval_semantics,
    )
    return archive, selection, evidence


def test_chaser_only_recipe_compiles_nested_roles_and_excludes_pending_end(
    tmp_path: Path,
) -> None:
    archive, selection, evidence = _semantic_selection(tmp_path)
    result = compile_protocol_semantic_chaser_selections(
        selection,
        timeline_evidence=_evidence(archive, selection),
        protocol_evidence=evidence,
        role_bindings=_chaser_bindings(selection),
    )

    assert tuple(result.named) == CHASER_WINDOW_ROLES
    assert result.standalone_solid_black is None
    assert result.standalone_solid_black_status == (
        "not_applicable_protocol_has_no_standalone_solid_black"
    )
    assert result.chaser_post.resolved_intervals[0].end_frame == 29
    post_metadata = (
        result.chaser_post.resolved_intervals[0]
        .source_memberships[0]
        .role.metadata
    )
    assert post_metadata[
        "terminal_frame_excluded_pending_step_end_contract"
    ] is True
    assert result.identity_record()["selector_eligible"] is False
    assert evidence.source_direct_consolidated_metadata["subtree_path"] == (
        "analysis/stimulus_runs/stimulus_1"
    )
    with pytest.raises(KeyError, match="legacy role"):
        result["black_before"]


def test_semantic_timeline_evidence_cannot_be_fabricated(tmp_path: Path) -> None:
    archive, selection, _protocol = _semantic_selection(tmp_path)
    loaded = _evidence(archive, selection)

    with pytest.raises(
        ProtocolSemanticChaserSelectionError,
        match="requires its strict loader",
    ):
        ProtocolSemanticTimelineEvidence(evidence=loaded.evidence)


def test_two_step_recipe_reports_present_but_unselected_baseline(
    tmp_path: Path,
) -> None:
    archive, selection, evidence = _semantic_selection(
        tmp_path,
        modes=("SOLID_BLACK", "CHASER"),
        step_bounds=((0, 9), (10, 39)),
    )
    result = compile_protocol_semantic_chaser_selections(
        selection,
        timeline_evidence=_evidence(archive, selection),
        protocol_evidence=evidence,
        role_bindings=_chaser_bindings(selection),
    )

    assert result.standalone_solid_black_step is not None
    assert result.standalone_solid_black is None
    assert result.standalone_solid_black_status == "present_not_selected"
    assessments = {
        row.capability_id: row for row in result.capability_assessments()
    }
    assert assessments[
        STANDALONE_SOLID_BLACK_BASELINE_CAPABILITY_ID
    ].state is CapabilityState.MISSING


def test_two_step_recipe_can_select_real_standalone_baseline(
    tmp_path: Path,
) -> None:
    archive, selection, evidence = _semantic_selection(
        tmp_path,
        modes=("SOLID_BLACK", "CHASER"),
        step_bounds=((0, 9), (10, 39)),
        include_standalone_window=True,
    )
    bindings = _chaser_bindings(selection, offset=1)
    bindings[STANDALONE_SOLID_BLACK_ROLE] = EpochRoleBinding.by_window_id(
        selection.intervals[0].window_id
    )
    result = compile_protocol_semantic_chaser_selections(
        selection,
        timeline_evidence=_evidence(archive, selection),
        protocol_evidence=evidence,
        role_bindings=bindings,
    )

    assert result.standalone_solid_black is not None
    assert result.standalone_solid_black_status == "selected"
    assert result.standalone_solid_black.resolved_intervals[0].end_frame == 9
    assessments = {
        row.capability_id: row for row in result.capability_assessments()
    }
    assert assessments[
        STANDALONE_SOLID_BLACK_BASELINE_CAPABILITY_ID
    ].state is CapabilityState.READY


def test_chaser_only_capabilities_are_ready_without_fabricating_baseline(
    tmp_path: Path,
) -> None:
    archive, selection, evidence = _semantic_selection(tmp_path)
    result = compile_protocol_semantic_chaser_selections(
        selection,
        timeline_evidence=_evidence(archive, selection),
        protocol_evidence=evidence,
        role_bindings=_chaser_bindings(selection),
    )
    assessments = {
        row.capability_id: row for row in result.capability_assessments()
    }

    assert assessments[
        PROTOCOL_SEMANTIC_CHASER_WINDOWS_CAPABILITY_ID
    ].state is CapabilityState.READY
    assert assessments[
        STANDALONE_SOLID_BLACK_BASELINE_CAPABILITY_ID
    ].state is CapabilityState.NOT_APPLICABLE
    production = assessments[
        PROTOCOL_SEMANTIC_SELECTION_PRODUCTION_CAPABILITY_ID
    ]
    assert production.state is CapabilityState.REVIEW_REQUIRED
    assert production.evidence["pending_contracts"] == [
        "producer_trial_index_checksum",
        "producer_step_end_interval_semantics",
    ]

    plan = plan_chaser_profile_applicability(
        recording_id="recording_1",
        profile_id="semantic_chaser_candidate_v1",
        profile_version=1,
        profile_sha256="b" * 64,
        profile_scope="full",
        selected_modules=(
            _Module(
                "chaser_window_analytics",
                "required",
                (PROTOCOL_SEMANTIC_CHASER_WINDOWS_CAPABILITY_ID,),
            ),
            _Module(
                "standalone_baseline_contrast",
                "conditional_required",
                (STANDALONE_SOLID_BLACK_BASELINE_CAPABILITY_ID,),
            ),
            _Module(
                "production_publication",
                "required",
                (PROTOCOL_SEMANTIC_SELECTION_PRODUCTION_CAPABILITY_ID,),
            ),
        ),
        capability_assessments=result.capability_assessments(),
    )
    assert [decision.state for decision in plan.module_decisions] == [
        ModuleApplicabilityState.APPLICABLE,
        ModuleApplicabilityState.INAPPLICABLE,
        ModuleApplicabilityState.BLOCKED_REVIEW_REQUIRED,
    ]
    assert plan.readiness is ProfileReadiness.BLOCKED


@pytest.mark.parametrize(
    ("semantics", "bounds"),
    [
        (STEP_END_INCLUSIVE, ((0, 29),)),
        (STEP_END_EXCLUSIVE, ((0, 30),)),
    ],
)
def test_producer_declared_end_semantics_preserve_full_half_open_window(
    tmp_path: Path,
    semantics: str,
    bounds: tuple[tuple[int, int], ...],
) -> None:
    archive, selection, evidence = _semantic_selection(
        tmp_path,
        step_bounds=bounds,
        step_end_interval_semantics=semantics,
    )
    result = compile_protocol_semantic_chaser_selections(
        selection,
        timeline_evidence=_evidence(archive, selection),
        protocol_evidence=evidence,
        role_bindings=_chaser_bindings(selection),
    )

    assert result.chaser_post.resolved_intervals[0].end_frame == 30
    metadata = (
        result.chaser_post.resolved_intervals[0]
        .source_memberships[0]
        .role.metadata
    )
    assert metadata[
        "terminal_frame_excluded_pending_step_end_contract"
    ] is False


@pytest.mark.parametrize(
    ("bounds", "match"),
    [
        (((1, 29),), "starts before"),
        (((0, 28),), "extends beyond"),
    ],
)
def test_nested_chaser_windows_must_be_conservatively_contained(
    tmp_path: Path,
    bounds: tuple[tuple[int, int], ...],
    match: str,
) -> None:
    archive, selection, evidence = _semantic_selection(
        tmp_path,
        step_bounds=bounds,
    )
    with pytest.raises(ProtocolSemanticChaserSelectionError, match=match):
        compile_protocol_semantic_chaser_selections(
            selection,
            timeline_evidence=_evidence(archive, selection),
            protocol_evidence=evidence,
            role_bindings=_chaser_bindings(selection),
        )


def test_chaser_role_bindings_require_exact_source_labels_and_order(
    tmp_path: Path,
) -> None:
    archive, selection, evidence = _semantic_selection(tmp_path)
    bindings = _chaser_bindings(selection)
    bindings["chaser_pre"], bindings["chaser_post"] = (
        bindings["chaser_post"],
        bindings["chaser_pre"],
    )

    with pytest.raises(
        ProtocolSemanticChaserSelectionError,
        match="must bind the exact 'pre_event' source window",
    ):
        compile_protocol_semantic_chaser_selections(
            selection,
            timeline_evidence=_evidence(archive, selection),
            protocol_evidence=evidence,
            role_bindings=bindings,
        )


def test_two_step_exclusive_bounds_allow_exact_step_adjacency(
    tmp_path: Path,
) -> None:
    archive, selection, evidence = _semantic_selection(
        tmp_path,
        modes=("SOLID_BLACK", "CHASER"),
        step_bounds=((0, 10), (10, 40)),
        step_end_interval_semantics=STEP_END_EXCLUSIVE,
    )

    result = compile_protocol_semantic_chaser_selections(
        selection,
        timeline_evidence=_evidence(archive, selection),
        protocol_evidence=evidence,
        role_bindings=_chaser_bindings(selection),
    )

    assert result.standalone_solid_black_status == "present_not_selected"
    assert result.chaser_post.resolved_intervals[0].end_frame == 40


def test_legacy_role_names_are_rejected_before_role_reinterpretation(
    tmp_path: Path,
) -> None:
    archive, selection, evidence = _semantic_selection(tmp_path)
    with pytest.raises(ProtocolSemanticChaserSelectionError, match="Legacy role"):
        compile_protocol_semantic_chaser_selections(
            selection,
            timeline_evidence=_evidence(archive, selection),
            protocol_evidence=evidence,
            role_bindings={
                "black_before": EpochRoleBinding.by_window_id(0),
                "chaser": EpochRoleBinding.by_window_id(1),
                "black_after": EpochRoleBinding.by_window_id(2),
            },
        )


def test_chaser_only_recipe_rejects_standalone_baseline_binding(
    tmp_path: Path,
) -> None:
    archive, selection, evidence = _semantic_selection(tmp_path)
    bindings = _chaser_bindings(selection)
    bindings[STANDALONE_SOLID_BLACK_ROLE] = EpochRoleBinding.by_window_id(0)
    with pytest.raises(
        ProtocolSemanticChaserSelectionError,
        match="exact protocol has no standalone",
    ):
        compile_protocol_semantic_chaser_selections(
            selection,
            timeline_evidence=_evidence(archive, selection),
            protocol_evidence=evidence,
            role_bindings=bindings,
        )


def test_loader_rejects_source_fingerprint_drift(
    tmp_path: Path,
) -> None:
    archive, selection, _evidence_value = _semantic_selection(tmp_path)
    root = zarr.open_group(str(archive), mode="a", use_consolidated=False)
    root["analysis/stimulus_runs/stimulus_1"].attrs["tamper_marker"] = True
    consolidate_metadata_capture_expected_warnings(archive)

    with pytest.raises(
        ProtocolSemanticChaserSelectionError,
        match="fingerprint differs",
    ):
        load_protocol_semantic_selection_evidence(archive, selection)


def test_v2_execution_interval_projects_to_exact_acquisition_envelope() -> None:
    bounds = _project_execution_interval_to_acquisition_bounds(
        ProtocolStimulusFrameInterval(
            start_stimulus_frame_inclusive=10,
            end_stimulus_frame_exclusive=15,
            first_camera_frame_id=110,
            last_camera_frame_id=114,
        ),
        step_index=1,
        stimulus_frame_num=np.arange(10, 15, dtype=np.int64),
        acquisition_frame_index=np.asarray([20, 21, 23, 24, 25]),
    )

    assert bounds.frame_axis == "source_acquisition_frame_index"
    assert bounds.start_camera_frame == 20
    assert bounds.end_camera_frame == 26
    assert bounds.mapped_stimulus_frame_count == 5
    assert bounds.mapped_acquisition_frame_count == 5
    assert bounds.reused_acquisition_frame_count == 0
    assert bounds.acquisition_gap_count == 1


def test_v2_projection_uses_latest_stimulus_for_reused_acquisition_row() -> None:
    bounds = _project_execution_interval_to_acquisition_bounds(
        ProtocolStimulusFrameInterval(
            start_stimulus_frame_inclusive=10,
            end_stimulus_frame_exclusive=15,
            first_camera_frame_id=110,
            last_camera_frame_id=114,
        ),
        step_index=1,
        stimulus_frame_num=np.arange(10, 15, dtype=np.int64),
        acquisition_frame_index=np.asarray([20, 20, 21, 21, 22]),
    )

    assert bounds.start_camera_frame == 20
    assert bounds.end_camera_frame == 23
    assert bounds.mapped_acquisition_frame_count == 3
    assert bounds.reused_acquisition_frame_count == 2
    assert bounds.acquisition_gap_count == 0
    assert bounds.to_dict()["acquisition_projection_policy"] == (
        "latest_stimulus_frame_per_source_acquisition_index_v1"
    )


def test_v2_execution_interval_rejects_missing_stimulus_membership() -> None:
    with pytest.raises(
        ProtocolSemanticChaserSelectionError,
        match="does not cover every producer-authored stimulus frame",
    ):
        _project_execution_interval_to_acquisition_bounds(
            ProtocolStimulusFrameInterval(
                start_stimulus_frame_inclusive=10,
                end_stimulus_frame_exclusive=15,
                first_camera_frame_id=110,
                last_camera_frame_id=114,
            ),
            step_index=1,
            stimulus_frame_num=np.asarray([10, 11, 13, 14]),
            acquisition_frame_index=np.asarray([20, 21, 23, 24]),
        )
