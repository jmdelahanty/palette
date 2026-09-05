from __future__ import annotations

from fisheye.analytics_exports.validated_behavior_core_behavior_contracts import (
    CORE_BEHAVIOR_TABLE_SPECS,
)
from fisheye.analytics_exports.validated_behavior_core_chaser_contracts import (
    CHASER_EXTENSION_OMITTED_TABLES,
    CORE_BOUND_CHASER_RELATIVE_PAIR_CAPABILITY,
    CORE_CHASER_EXTENSION_TABLE_SPECS,
    CORE_CHASER_TABLE_SPECS,
)
from fisheye.analytics_exports.validated_behavior_phase_c_contracts import (
    PHASE_C_TABLE_SPECS,
)


def _foreign_key_targets(table_name: str) -> set[str]:
    return {
        target
        for _local, target, _target_fields in CORE_CHASER_TABLE_SPECS[
            table_name
        ].foreign_keys
    }


def test_composite_roster_subtracts_competing_core_like_tables_before_composition() -> None:
    assert set(CORE_BEHAVIOR_TABLE_SPECS).issubset(CORE_CHASER_TABLE_SPECS)
    assert set(CORE_CHASER_EXTENSION_TABLE_SPECS) == (
        set(PHASE_C_TABLE_SPECS) - CHASER_EXTENSION_OMITTED_TABLES
    )
    assert "provider_motion_samples" not in CORE_CHASER_TABLE_SPECS
    assert "bout_detector_signal_samples" not in CORE_CHASER_TABLE_SPECS
    assert "body_frame_samples" not in CORE_CHASER_TABLE_SPECS
    assert list(CORE_CHASER_TABLE_SPECS).count("canonical_swim_bouts") == 1


def test_paradigm_sample_relations_target_selected_core_rows() -> None:
    assert CORE_CHASER_TABLE_SPECS[
        "chaser_relative_samples"
    ].required_capability == CORE_BOUND_CHASER_RELATIVE_PAIR_CAPABILITY
    assert "kinematics_samples" in _foreign_key_targets("chaser_relative_samples")
    assert "subject_body_frame_samples" in _foreign_key_targets(
        "body_relative_samples"
    )
    assert "body_frame_samples" not in _foreign_key_targets("body_relative_samples")

    chaser_fk = next(
        foreign_key
        for foreign_key in CORE_CHASER_TABLE_SPECS[
            "chaser_relative_samples"
        ].foreign_keys
        if foreign_key[1] == "kinematics_samples"
    )
    assert chaser_fk == (
        ("export_run_id", "recording_id", "track_sample_id"),
        "kinematics_samples",
        ("export_run_id", "recording_id", "track_sample_index"),
    )

    body_fk = next(
        foreign_key
        for foreign_key in CORE_CHASER_TABLE_SPECS["body_relative_samples"].foreign_keys
        if foreign_key[1] == "subject_body_frame_samples"
    )
    assert body_fk == (
        ("export_run_id", "recording_id", "body_source_row_id"),
        "subject_body_frame_samples",
        ("export_run_id", "recording_id", "subject_shape_row_index"),
    )
