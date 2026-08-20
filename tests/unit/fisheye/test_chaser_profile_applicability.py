from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy

import pytest

from fisheye.analysis.chaser_profiles import (
    full_chaser_analysis_profile_v3_path,
    load_chaser_analysis_profile,
    resolve_chaser_analysis_modules,
)
from fisheye.analysis_workflows.chaser_profile_applicability import (
    CHASER_TEMPORAL_ALIGNMENT_CAPABILITY_ID,
    CapabilityAssessment,
    CapabilityState,
    ChaserProfileApplicabilityError,
    ModuleApplicabilityState,
    ProfileReadiness,
    input_provenance_proxy_alignment_assessment,
    plan_chaser_profile_applicability,
    require_chaser_profile_applicability_plan,
    unavailable_physical_presentation_alignment_assessment,
)


@dataclass(frozen=True)
class _Module:
    module_id: str
    requirement_class: str
    required_capabilities: tuple[str, ...]
    depends_on: tuple[str, ...] = ()


def _capability(
    capability_id: str,
    state: CapabilityState = CapabilityState.READY,
) -> CapabilityAssessment:
    if capability_id == CHASER_TEMPORAL_ALIGNMENT_CAPABILITY_ID:
        if state is CapabilityState.READY:
            return input_provenance_proxy_alignment_assessment(
                proxy_projection_sha256="c" * 64,
                proxy_run_path="analysis/chaser_input_provenance_proxy_runs/proxy_v1",
                proxy_manifest_sha256="e" * 64,
            )
        if state is CapabilityState.MISSING:
            return unavailable_physical_presentation_alignment_assessment()
    return CapabilityAssessment(
        capability_id=capability_id,
        state=state,
        reason_code=f"fixture_{state.value}",
        evidence={"authority_sha256": "a" * 64} if state is CapabilityState.READY else {},
    )


def _plan(
    modules: tuple[_Module, ...],
    capabilities: tuple[CapabilityAssessment, ...],
    **kwargs: object,
):
    return plan_chaser_profile_applicability(
        recording_id="recording-001",
        profile_id="chaser_behavior_full_v3",
        profile_version=3,
        profile_sha256="b" * 64,
        profile_scope=str(kwargs.pop("profile_scope", "full")),
        selected_modules=modules,
        capability_assessments=capabilities,
        **kwargs,
    )


def test_position_modules_remain_applicable_when_heading_is_not_applicable() -> None:
    modules = (
        _Module("distance", "required", ("position",)),
        _Module(
            "bearing",
            "conditional_required",
            ("position", "body_frame"),
            ("distance",),
        ),
    )

    plan = _plan(
        modules,
        (
            _capability("position"),
            _capability("body_frame", CapabilityState.NOT_APPLICABLE),
        ),
    )

    assert [row.state for row in plan.module_decisions] == [
        ModuleApplicabilityState.APPLICABLE,
        ModuleApplicabilityState.INAPPLICABLE,
    ]
    assert plan.readiness is ProfileReadiness.PLANNED


def test_inapplicable_capability_dominates_irrelevant_missing_or_invalid_inputs() -> None:
    modules = (
        _Module(
            "gaze",
            "conditional_required",
            ("body_frame", "eye_orientation", "position"),
        ),
    )
    plan = _plan(
        modules,
        (
            _capability("body_frame", CapabilityState.NOT_APPLICABLE),
            _capability("eye_orientation", CapabilityState.MISSING),
            _capability("position", CapabilityState.INVALID),
        ),
    )

    assert plan.module_decisions[0].state is ModuleApplicabilityState.INAPPLICABLE
    assert plan.module_decisions[0].implicated_capability_ids == ("body_frame",)
    assert plan.readiness is ProfileReadiness.PLANNED


def test_inapplicable_dependency_dominates_irrelevant_block_for_conditional_module() -> None:
    modules = (
        _Module("bearing", "conditional_required", ("body_frame",)),
        _Module("motion", "required", ("position",)),
        _Module(
            "directed_response",
            "conditional_required",
            (),
            ("bearing", "motion"),
        ),
    )
    plan = _plan(
        modules,
        (
            _capability("body_frame", CapabilityState.NOT_APPLICABLE),
            _capability("position", CapabilityState.INVALID),
        ),
    )

    assert plan.module_decisions[2].state is ModuleApplicabilityState.INAPPLICABLE
    assert plan.module_decisions[2].implicated_dependency_ids == (
        "bearing",
        "motion",
    )


@pytest.mark.parametrize(
    ("capability_state", "expected_module_state", "expected_readiness"),
    [
        (
            CapabilityState.MISSING,
            ModuleApplicabilityState.BLOCKED_MISSING_CAPABILITY,
            ProfileReadiness.BLOCKED,
        ),
        (
            CapabilityState.INVALID,
            ModuleApplicabilityState.BLOCKED_INVALID_SOURCE,
            ProfileReadiness.BLOCKED,
        ),
        (
            CapabilityState.REVIEW_REQUIRED,
            ModuleApplicabilityState.BLOCKED_REVIEW_REQUIRED,
            ProfileReadiness.BLOCKED,
        ),
        (
            CapabilityState.STALE,
            ModuleApplicabilityState.STALE,
            ProfileReadiness.BLOCKED,
        ),
    ],
)
def test_missing_invalid_review_and_stale_are_never_called_inapplicable(
    capability_state: CapabilityState,
    expected_module_state: ModuleApplicabilityState,
    expected_readiness: ProfileReadiness,
) -> None:
    plan = _plan(
        (_Module("bearing", "conditional_required", ("body_frame",)),),
        (_capability("body_frame", capability_state),),
    )

    assert plan.module_decisions[0].state is expected_module_state
    assert plan.readiness is expected_readiness


def test_unassessed_capability_fails_closed_as_missing() -> None:
    plan = _plan(
        (_Module("distance", "required", ("position",)),),
        (),
    )

    decision = plan.module_decisions[0]
    assert decision.state is ModuleApplicabilityState.BLOCKED_MISSING_CAPABILITY
    assert decision.implicated_capability_ids == ("position",)


def test_proxy_alignment_explicitly_allows_chaser_modules_without_affecting_occupancy() -> None:
    modules = (
        _Module("epochs", "required", ()),
        _Module("occupancy", "required", ("position",), ("epochs",)),
        _Module(
            "distance",
            "required",
            ("position", CHASER_TEMPORAL_ALIGNMENT_CAPABILITY_ID),
            ("epochs",),
        ),
        _Module("response", "required", (), ("distance",)),
    )
    plan = _plan(
        modules,
        (
            _capability("position"),
            input_provenance_proxy_alignment_assessment(
                proxy_projection_sha256="d" * 64,
                proxy_run_path="analysis/chaser_input_provenance_proxy_runs/proxy_v1",
                proxy_manifest_sha256="e" * 64,
            ),
        ),
    )

    assert all(
        row.state is ModuleApplicabilityState.APPLICABLE
        for row in plan.module_decisions
    )
    temporal = next(
        row
        for row in plan.capability_assessments
        if row.capability_id == CHASER_TEMPORAL_ALIGNMENT_CAPABILITY_ID
    )
    assert temporal.evidence["temporal_alignment_class"] == (
        "controller_input_provenance_proxy"
    )
    assert temporal.evidence["physical_presentation_verified"] is False


def test_physical_alignment_request_blocks_chaser_chain_but_not_independent_occupancy() -> None:
    modules = (
        _Module("epochs", "required", ()),
        _Module("occupancy", "required", ("position",), ("epochs",)),
        _Module(
            "distance",
            "required",
            ("position", CHASER_TEMPORAL_ALIGNMENT_CAPABILITY_ID),
            ("epochs",),
        ),
        _Module("response", "required", (), ("distance",)),
    )
    plan = _plan(
        modules,
        (
            _capability("position"),
            unavailable_physical_presentation_alignment_assessment(),
        ),
    )
    by_id = {row.module_id: row for row in plan.module_decisions}

    assert by_id["occupancy"].state is ModuleApplicabilityState.APPLICABLE
    assert by_id["distance"].state is (
        ModuleApplicabilityState.BLOCKED_MISSING_CAPABILITY
    )
    assert by_id["distance"].implicated_capability_ids == (
        CHASER_TEMPORAL_ALIGNMENT_CAPABILITY_ID,
    )
    assert by_id["response"].state is (
        ModuleApplicabilityState.BLOCKED_MISSING_CAPABILITY
    )


def test_temporal_alignment_capability_rejects_implicit_or_rehashed_fallback() -> None:
    with pytest.raises(ChaserProfileApplicabilityError, match="controlled requirement"):
        CapabilityAssessment(
            capability_id=CHASER_TEMPORAL_ALIGNMENT_CAPABILITY_ID,
            state=CapabilityState.READY,
            reason_code="fixture_ready",
            evidence={"authority_sha256": "a" * 64},
        )

    proxy = input_provenance_proxy_alignment_assessment(
        proxy_projection_sha256="d" * 64,
        proxy_run_path="analysis/chaser_input_provenance_proxy_runs/proxy_v1",
        proxy_manifest_sha256="e" * 64,
    ).to_dict()
    proxy["evidence"]["physical_presentation_verified"] = True
    with pytest.raises(ChaserProfileApplicabilityError, match="invalid"):
        CapabilityAssessment.from_dict(proxy)


def test_dependency_block_propagates_without_blocking_independent_module() -> None:
    modules = (
        _Module("distance", "required", ("position",)),
        _Module("visit", "conditional_required", ("geometry",), ("distance",)),
        _Module("epochs", "required", ("stimulus",)),
    )
    plan = _plan(
        modules,
        (
            _capability("position", CapabilityState.INVALID),
            _capability("geometry"),
            _capability("stimulus"),
        ),
    )

    assert plan.module_decisions[0].state is ModuleApplicabilityState.BLOCKED_INVALID_SOURCE
    assert plan.module_decisions[1].state is ModuleApplicabilityState.BLOCKED_INVALID_SOURCE
    assert plan.module_decisions[1].implicated_dependency_ids == ("distance",)
    assert plan.module_decisions[2].state is ModuleApplicabilityState.APPLICABLE


def test_completion_requires_dependencies_to_be_complete() -> None:
    modules = (
        _Module("distance", "required", ("position",)),
        _Module("visit", "required", ("geometry",), ("distance",)),
    )
    partial = _plan(
        modules,
        (_capability("position"), _capability("geometry")),
        completed_module_ids=("visit",),
    )
    complete = _plan(
        modules,
        (_capability("position"), _capability("geometry")),
        completed_module_ids=("distance", "visit"),
    )

    assert partial.module_decisions[1].state is ModuleApplicabilityState.APPLICABLE
    assert partial.readiness is ProfileReadiness.PLANNED
    assert complete.readiness is ProfileReadiness.COMPLETE


def test_optional_block_does_not_block_full_profile_readiness() -> None:
    plan = _plan(
        (
            _Module("distance", "required", ("position",)),
            _Module("experimental", "optional", ("eye_orientation",)),
        ),
        (
            _capability("position"),
            _capability("eye_orientation", CapabilityState.MISSING),
        ),
        completed_module_ids=("distance",),
    )

    assert plan.module_decisions[1].state is ModuleApplicabilityState.BLOCKED_MISSING_CAPABILITY
    assert plan.readiness is ProfileReadiness.COMPLETE


def test_reduced_profile_never_claims_full_readiness() -> None:
    plan = _plan(
        (_Module("distance", "required", ("position",)),),
        (_capability("position"),),
        completed_module_ids=("distance",),
        profile_scope="reduced",
    )

    assert plan.readiness is ProfileReadiness.NOT_CLAIMED_REDUCED_PROFILE


def test_plan_is_canonical_and_preserves_explicit_overrides() -> None:
    plan = _plan(
        (_Module("distance", "required", ("position",)),),
        (_capability("position"),),
        explicit_enable=("distance",),
    )

    assert len(plan.sha256) == 64
    assert plan.as_envelope()["record"]["execution_order"] == ["distance"]
    assert plan.as_envelope()["record"]["explicit_overrides"] == {
        "enable": ["distance"],
        "disable": [],
    }
    assert require_chaser_profile_applicability_plan(
        plan.as_envelope(),
        expected_sha256=plan.sha256,
    ) == plan


def test_plan_envelope_rejects_tampering_and_rehashed_identity_swap() -> None:
    plan = _plan(
        (_Module("distance", "required", ("position",)),),
        (_capability("position"),),
    )
    tampered = deepcopy(plan.as_envelope())
    tampered["record"]["recording_id"] = "recording-002"
    with pytest.raises(ChaserProfileApplicabilityError, match="digest"):
        require_chaser_profile_applicability_plan(tampered)

    swapped = _plan(
        (_Module("distance", "required", ("position",)),),
        (_capability("position"),),
        explicit_enable=("distance",),
    )
    with pytest.raises(ChaserProfileApplicabilityError, match="expectation"):
        require_chaser_profile_applicability_plan(
            swapped.as_envelope(),
            expected_sha256=plan.sha256,
        )


def test_selected_modules_must_be_dependency_ordered() -> None:
    with pytest.raises(ChaserProfileApplicabilityError, match="dependency ordered"):
        _plan(
            (
                _Module("visit", "required", ("geometry",), ("distance",)),
                _Module("distance", "required", ("position",)),
            ),
            (_capability("position"), _capability("geometry")),
        )


def test_full_v3_profile_plans_position_modules_and_marks_absent_body_frame_inapplicable() -> None:
    profile = load_chaser_analysis_profile(full_chaser_analysis_profile_v3_path())
    capabilities = {
        capability_id
        for module in profile.modules
        for capability_id in module.required_capabilities
    }
    assessments = tuple(
        _capability(
            capability_id,
            (
                CapabilityState.NOT_APPLICABLE
                if capability_id in {"body_frame", "eye_orientation"}
                else CapabilityState.READY
            ),
        )
        for capability_id in sorted(capabilities)
    )

    plan = plan_chaser_profile_applicability(
        recording_id="recording-001",
        profile_id=profile.profile_id,
        profile_version=profile.profile_version,
        profile_sha256=profile.sha256,
        profile_scope=profile.profile_scope,
        selected_modules=resolve_chaser_analysis_modules(profile),
        capability_assessments=assessments,
    )
    by_id = {row.module_id: row for row in plan.module_decisions}

    assert by_id["chaser_distance"].state is ModuleApplicabilityState.APPLICABLE
    assert by_id["chaser_response_regimes"].state is ModuleApplicabilityState.APPLICABLE
    assert by_id["chaser_egocentric_bearing"].state is (
        ModuleApplicabilityState.INAPPLICABLE
    )
    assert by_id["chaser_gaze_tracking"].state is (
        ModuleApplicabilityState.INAPPLICABLE
    )
    assert plan.readiness is ProfileReadiness.PLANNED
