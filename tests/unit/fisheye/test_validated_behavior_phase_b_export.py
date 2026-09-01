from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import fisheye.analytics_exports.validated_behavior_phase_b_adapters as adapters
import fisheye.analytics_exports.validated_behavior_phase_b_contracts as contracts
import fisheye.analytics_exports.validated_behavior_profiles as profiles
from fisheye.analytics_exports.validated_behavior_contracts import (
    CORE_TABLE_NAMES,
    validate_table_specs,
)
from fisheye.analytics_exports.validated_behavior_dataset import (
    ValidatedBehaviorTable,
)
from fisheye.analytics_exports.validated_behavior_phase_a_contracts import (
    PHASE_A_TABLE_SPECS,
)

EXPECTED_DENSE_TABLES = (
    "provider_motion_samples",
    "bout_detector_signal_samples",
    "stimulus_native_state_support",
    "chaser_relative_samples",
    "body_frame_samples",
    "body_relative_samples",
    "controller_trial_membership",
    "controller_trial_gap_evidence",
)


def _fields(table_name: str) -> set[str]:
    return {
        field.name
        for field in contracts.PHASE_B_DENSE_TABLE_SPECS[table_name].contract.fields
    }


def test_phase_b_profile_extends_phase_a_with_exact_dense_roster() -> None:
    profile = profiles.resolve_validated_behavior_profile(contracts.PHASE_B_PROFILE_ID)

    assert profile.profile_id == contracts.PHASE_B_PROFILE_ID
    assert tuple(contracts.PHASE_B_DENSE_TABLE_SPECS) == EXPECTED_DENSE_TABLES
    assert tuple(profile.table_specs) == (
        *tuple(PHASE_A_TABLE_SPECS),
        *EXPECTED_DENSE_TABLES,
    )
    assert len(validate_table_specs(profile.table_specs)) == 30
    assert set(profile.row_extractors()) == (
        set(profile.table_specs) - set(CORE_TABLE_NAMES)
    )


def test_dense_specs_seal_streaming_and_semantic_contracts() -> None:
    for table_name in EXPECTED_DENSE_TABLES:
        spec = contracts.PHASE_B_DENSE_TABLE_SPECS[table_name]
        serialized = spec.to_dict()

        assert spec.capability_policy == "optional_explicit_coverage"
        assert spec.zero_rows_allowed is True
        assert spec.primary_key_validation == "strictly_increasing_v1"
        assert serialized["primary_key_validation"] == "strictly_increasing_v1"
        assert serialized["semantic_metadata"] == dict(spec.semantic_metadata)


def test_physical_motion_and_detector_response_remain_distinct() -> None:
    motion = _fields("provider_motion_samples")
    detector = _fields("bout_detector_signal_samples")
    detector_metadata = dict(
        contracts.PHASE_B_DENSE_TABLE_SPECS[
            "bout_detector_signal_samples"
        ].semantic_metadata
    )

    assert {
        "speed_raw_mm_s",
        "speed_filtered_mm_s",
        "speed_smoothed_mm_s",
        "speed_averaged_mm_s",
    }.issubset(motion)
    assert "detection_signal_mm_s" not in motion
    assert "detection_signal_mm_s" in detector
    assert not any(name.startswith("speed_") for name in detector)
    assert detector_metadata["signal_role"] == (
        "detector_response_not_physical_speed_estimator"
    )


def test_trial_gaps_are_evidence_not_trial_members() -> None:
    membership = contracts.PHASE_B_DENSE_TABLE_SPECS["controller_trial_membership"]
    gaps = contracts.PHASE_B_DENSE_TABLE_SPECS["controller_trial_gap_evidence"]

    assert "trial_row_id" in _fields("controller_trial_membership")
    assert "trial_gap_reason" not in _fields("controller_trial_membership")
    assert "trial_envelope_row_id" in _fields("controller_trial_gap_evidence")
    assert "trial_gap_reason" in _fields("controller_trial_gap_evidence")
    assert dict(membership.semantic_metadata)["membership"] == (
        "exact_logged_active_rows_only"
    )
    assert dict(gaps.semantic_metadata)["membership"] == (
        "evidence_rows_are_not_trial_members"
    )


def test_phase_b_adapter_is_protocol_neutral() -> None:
    source = Path(adapters.__file__).read_text(encoding="utf-8").casefold()

    assert "goodbatbadbat" not in source
    assert "goodcopbadcop" not in source
    assert "protocol_name" not in source
    assert "protocol_hash" not in source


def test_lazy_table_exposes_plan_bound_semantic_metadata() -> None:
    spec = contracts.PHASE_B_DENSE_TABLE_SPECS["body_relative_samples"]
    dataset = SimpleNamespace(
        export_run_id="fixture-export",
        manifest={
            "record_sha256": "a" * 64,
            "export_plan": {"plan_sha256": "b" * 64},
            "analysis_unit_policy": {"sha256": "c" * 64},
        },
    )
    table = ValidatedBehaviorTable(
        dataset=dataset,  # type: ignore[arg-type]
        name="body_relative_samples",
        spec=spec,
        part_paths=(),
    )

    assert dict(table.semantic_metadata) == dict(spec.semantic_metadata)
    assert table.query_identity()["semantic_metadata"] == {
        "bearing_convention": "positive_toward_anatomical_left_degrees",
        "alignment": "cos_body_bearing_and_sin_body_bearing",
        "distance_units": "mm",
        "interpolation": "prohibited",
    }


def test_foreign_keys_must_preserve_recording_ownership() -> None:
    original = contracts.PHASE_B_TABLE_SPECS["provider_motion_samples"]
    invalid = replace(
        original,
        foreign_keys=((("recording_id",), "cohort_recordings", ("recording_id",)),),
    )
    specs = {**contracts.PHASE_B_TABLE_SPECS, invalid.table_name: invalid}

    with pytest.raises(ValueError, match="foreign keys must be recording-scoped"):
        validate_table_specs(specs)
