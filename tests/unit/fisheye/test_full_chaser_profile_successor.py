from __future__ import annotations

import pytest

from fisheye.analysis.chaser_profiles import (
    full_chaser_analysis_profile_v4_path,
    load_chaser_analysis_profile,
    resolve_chaser_analysis_modules,
    validate_chaser_runner_modules,
)
from fisheye.analysis_workflows.chaser_profile_applicability import (
    CHASER_TEMPORAL_ALIGNMENT_CAPABILITY_ID,
    CapabilityAssessment,
    CapabilityState,
    ProfileReadiness,
    input_provenance_proxy_alignment_assessment,
    plan_chaser_profile_applicability,
)
from fisheye.analysis_workflows.full_chaser_profile_successor import (
    FullChaserProfileSuccessorError,
    ImmutableModuleProductBinding,
    prepare_full_chaser_profile_successor,
)


def _profile():
    profile = load_chaser_analysis_profile(full_chaser_analysis_profile_v4_path())
    selected = resolve_chaser_analysis_modules(profile)
    validate_chaser_runner_modules(selected)
    return profile, selected


def _assessments(selected):
    capabilities = sorted(
        {value for module in selected for value in module.required_capabilities}
    )
    result = []
    for capability in capabilities:
        if capability == CHASER_TEMPORAL_ALIGNMENT_CAPABILITY_ID:
            result.append(
                input_provenance_proxy_alignment_assessment(
                    proxy_projection_sha256="a" * 64,
                    proxy_run_path=(
                        "analysis/chaser_input_provenance_proxy_runs/proxy-v1"
                    ),
                    proxy_manifest_sha256="b" * 64,
                )
            )
        else:
            result.append(
                CapabilityAssessment(
                    capability_id=capability,
                    state=CapabilityState.READY,
                    reason_code="fixture_ready",
                    evidence={"authority_sha256": "c" * 64},
                )
            )
    return tuple(result)


def _plan(*, complete: bool):
    profile, selected = _profile()
    plan = plan_chaser_profile_applicability(
        recording_id="recording-1",
        profile_id=profile.profile_id,
        profile_version=profile.profile_version,
        profile_sha256=profile.sha256,
        profile_scope=profile.profile_scope,
        selected_modules=selected,
        capability_assessments=_assessments(selected),
        completed_module_ids=(
            tuple(module.module_id for module in selected) if complete else ()
        ),
    )
    return profile, selected, plan


def _products(selected):
    return tuple(
        ImmutableModuleProductBinding(
            module_id=module.module_id,
            schema_id=module.schema_id,
            schema_version=module.schema_version,
            run_path=f"analysis/{module.module_id}_runs/recording-1-v1",
            manifest_sha256=(f"{index + 1:x}"[-1] * 64),
            payload_sha256=(f"{index + 2:x}"[-1] * 64),
        )
        for index, module in enumerate(selected)
    )


def test_v4_profile_resolves_new_successors_and_runner_adapters() -> None:
    profile, selected = _profile()
    by_id = {module.module_id: module for module in selected}

    assert profile.profile_id == "chaser_behavior_full_v4"
    assert profile.profile_scope == "full"
    assert by_id["controller_chase_trials"].schema_id == (
        "palette.analysis.controller_chase_trials"
    )
    assert by_id["generalized_chaser_bout_response"].depends_on == (
        "chaser_distance",
        "controller_chase_trials",
    )
    assert by_id["chaser_escape_freeze_v2"].depends_on == (
        "controller_chase_trials",
        "generalized_chaser_bout_response",
    )


def test_complete_full_profile_binds_every_exact_product_and_waves() -> None:
    profile, selected, plan = _plan(complete=True)
    assert plan.readiness is ProfileReadiness.COMPLETE
    result = prepare_full_chaser_profile_successor(
        profile=profile,
        applicability=plan,
        products=_products(selected),
    )

    assert result.full_profile_complete is True
    assert result.readiness == "complete"
    assert result.array("product_bound").tolist() == [True] * len(selected)
    waves = result.array("execution_wave")
    wave_by_id = {
        module.module_id: int(wave) for module, wave in zip(selected, waves)
    }
    assert wave_by_id["stimulus_epochs"] == 0
    assert wave_by_id["detection_occupancy"] == 1
    assert wave_by_id["chaser_distance"] == 1
    assert wave_by_id["controller_chase_trials"] == 2
    assert wave_by_id["chaser_gaze_tracking_v2"] == 3
    assert wave_by_id["generalized_chaser_bout_response"] == 3
    assert wave_by_id["chaser_escape_freeze_v2"] == 4
    assert len(result.manifest["module_products"]) == len(selected)
    assert result.manifest["normalized_profile_sha256"] == profile.sha256
    assert result.manifest["applicability_plan_sha256"] == plan.sha256
    assert result.manifest["policy"]["reuse"].startswith("exact_profile")
    assert result.manifest["selector_eligible"] is False


def test_planned_full_profile_is_honest_without_product_bindings() -> None:
    profile, _selected, plan = _plan(complete=False)
    assert plan.readiness is ProfileReadiness.PLANNED
    result = prepare_full_chaser_profile_successor(
        profile=profile,
        applicability=plan,
        products=(),
    )

    assert result.full_profile_complete is False
    assert not any(result.array("product_bound"))
    assert result.manifest["module_products"] == ()


def test_completed_module_without_exact_product_is_rejected() -> None:
    profile, selected, plan = _plan(complete=True)
    with pytest.raises(FullChaserProfileSuccessorError, match="lacks an immutable"):
        prepare_full_chaser_profile_successor(
            profile=profile,
            applicability=plan,
            products=_products(selected)[:-1],
        )


def test_selector_eligible_module_product_is_rejected() -> None:
    profile, selected, plan = _plan(complete=True)
    first = selected[0]
    with pytest.raises(FullChaserProfileSuccessorError, match="selector-ineligible"):
        ImmutableModuleProductBinding(
            module_id=first.module_id,
            schema_id=first.schema_id,
            schema_version=first.schema_version,
            run_path="analysis/example_runs/example-v1",
            manifest_sha256="a" * 64,
            payload_sha256="b" * 64,
            selector_eligible=True,
        )
